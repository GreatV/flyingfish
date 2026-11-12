//! Int8 linear with a block-diagonal regular Hadamard rotate-quant prologue.
//! `y = bf16((q(x R) @ qweight^T) * s_x * wscale)`, int32 products.
//! `R` is `kron(H4 / 2, ...)` of `group` (a power of 4), not stored.
//! `x` is cast to bf16. Each rotated group is an f32 dot of those bf16 values, rounded back to bf16.
//! `s_x = max(|xR|, 1e-12) / 127`. `q` is ties-to-even, clamped to [-127, 127].
use anyhow::{Result, ensure};
use half::bf16;

const H4: [[i32; 4]; 4] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]];

/// Entry of the regular Hadamard of `group`. `group` is a power of 4 and at least 4.
fn hadamard(group: usize, row: usize, col: usize) -> f32 {
    let mut sign = 1i32;
    let mut scale = 1.0f32;
    let mut n = group;
    let mut row = row;
    let mut col = col;
    while n > 1 {
        n /= 4;
        let rb = row / n;
        let cb = col / n;
        row %= n;
        col %= n;
        sign *= H4[rb][cb];
        scale *= 0.5;
    }
    (sign as f32) * scale
}

fn power_of_four(group: usize) -> bool {
    group >= 4 && group.is_power_of_two() && group.trailing_zeros().is_multiple_of(2)
}

/// `[rows, k]` f32 activation to per-row int8 codes and `s_x`.
fn rotate_quant(x: &[f32], k: usize, group: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    ensure!(
        k > 0 && x.len().is_multiple_of(k),
        "convrot activation shape"
    );
    ensure!(
        power_of_four(group) && k.is_multiple_of(group),
        "convrot group {group} does not divide {k}"
    );
    ensure!(x.iter().all(|v| v.is_finite()), "convrot activation");
    let rows = x.len() / k;
    let mut codes = vec![0i8; rows * k];
    let mut scales = vec![0.0f32; rows];
    for row in 0..rows {
        let src = &x[row * k..(row + 1) * k];
        let mut xr = vec![bf16::ZERO; k];
        for start in (0..k).step_by(group) {
            for col in 0..group {
                let mut sum = 0.0f32;
                for i in 0..group {
                    let value = bf16::from_f32(src[start + i]).to_f32();
                    sum += value * hadamard(group, i, col);
                }
                xr[start + col] = bf16::from_f32(sum);
            }
        }
        let mut peak = 0.0f32;
        for value in &xr {
            peak = peak.max(value.to_f32().abs());
        }
        let scale = peak.max(1e-12) / 127.0;
        scales[row] = scale;
        for (i, value) in xr.iter().enumerate() {
            let quantized = (value.to_f32() / scale)
                .round_ties_even()
                .clamp(-127.0, 127.0);
            codes[row * k + i] = quantized as i8;
        }
    }
    Ok((codes, scales))
}

fn int8_product(
    codes: &[i8],
    scales: &[f32],
    k: usize,
    qweight: &[i8],
    n: usize,
    wscale: &[f32],
) -> Result<Vec<bf16>> {
    let rows = scales.len();
    ensure!(
        codes.len() == rows * k && qweight.len() == n * k && wscale.len() == n && n > 0,
        "convrot product shape"
    );
    ensure!(wscale.iter().all(|v| v.is_finite()), "convrot wscale");
    let mut y = Vec::with_capacity(rows * n);
    for row in 0..rows {
        let xq = &codes[row * k..(row + 1) * k];
        let sx = scales[row];
        for col in 0..n {
            let mut acc = 0i32;
            let weight = &qweight[col * k..(col + 1) * k];
            for i in 0..k {
                acc += i32::from(xq[i]) * i32::from(weight[i]);
            }
            y.push(bf16::from_f32(acc as f32 * sx * wscale[col]));
        }
    }
    Ok(y)
}

/// `x` is `[rows, k]` f32. `qweight` is `[n, k]` i8. `wscale` is `[n]` f32.
/// The bf16 round of the rotated activation is used at every row count.
pub fn linear(
    x: &[f32],
    k: usize,
    group: usize,
    qweight: &[i8],
    n: usize,
    wscale: &[f32],
) -> Result<Vec<bf16>> {
    let (codes, scales) = rotate_quant(x, k, group)?;
    int8_product(&codes, &scales, k, qweight, n, wscale)
}

/// Same product with the rotation omitted: the bf16 activation is quantized directly.
#[cfg(test)]
fn linear_omitting_rotation(
    x: &[f32],
    k: usize,
    qweight: &[i8],
    n: usize,
    wscale: &[f32],
) -> Result<Vec<bf16>> {
    ensure!(
        k > 0 && x.len().is_multiple_of(k),
        "convrot activation shape"
    );
    let rows = x.len() / k;
    let mut codes = vec![0i8; rows * k];
    let mut scales = vec![0.0f32; rows];
    for row in 0..rows {
        let src = &x[row * k..(row + 1) * k];
        let xr: Vec<bf16> = src.iter().copied().map(bf16::from_f32).collect();
        let mut peak = 0.0f32;
        for value in &xr {
            peak = peak.max(value.to_f32().abs());
        }
        let scale = peak.max(1e-12) / 127.0;
        scales[row] = scale;
        for (i, value) in xr.iter().enumerate() {
            let quantized = (value.to_f32() / scale)
                .round_ties_even()
                .clamp(-127.0, 127.0);
            codes[row * k + i] = quantized as i8;
        }
    }
    int8_product(&codes, &scales, k, qweight, n, wscale)
}

#[cfg(test)]
mod tests {
    use super::{hadamard, linear, linear_omitting_rotation};
    use anyhow::{Result, ensure};
    use half::bf16;

    fn row(r: usize, i: usize) -> f32 {
        ((((r * 17 + i * 3) % 23) as f64) * 0.05 - 0.4) as f32
    }

    fn code(n: usize, i: usize) -> i8 {
        (((n * 13 + i * 5) % 255) as i16 - 127) as i8
    }

    fn activation(rows: usize, k: usize) -> Vec<f32> {
        (0..rows)
            .flat_map(|r| (0..k).map(move |i| row(r, i)))
            .collect()
    }

    fn weight(n: usize, k: usize) -> Vec<i8> {
        (0..n)
            .flat_map(|col| (0..k).map(move |i| code(col, i)))
            .collect()
    }

    fn bits(y: &[bf16]) -> Vec<i16> {
        y.iter().map(|v| v.to_bits() as i16).collect()
    }

    /// Definition rotate-quant-matmul, group 256, N = 2, wscale 0.0125 and 0.03125.
    /// High halves of the f32 view of the bf16 result. M = 1 is below 17 rows. M = 17 and M = 20 are not.
    const M1: [i16; 2] = [16484, 16398];
    const M17: [i16; 34] = [
        16484, 16398, -16336, -16045, -16089, -15970, -16076, -16099, -16997, -16278, 16435, 16585,
        -16278, -16477, -16356, 16681, 16678, 16796, 16521, 16572, -16230, -16464, -16395, 16690,
        16435, -16199, -16213, -15994, -16089, -16161, -16219, -16128, 16471, 16527,
    ];
    const M20: [i16; 40] = [
        16484, 16398, -16336, -16045, -16089, -15970, -16076, -16099, -16997, -16278, 16435, 16585,
        -16278, -16477, -16356, 16681, 16678, 16796, 16521, 16572, -16230, -16464, -16395, 16690,
        16435, -16199, -16213, -15994, -16089, -16161, -16219, -16128, 16471, 16527, -16492,
        -16506, -16300, 16675, 16586, 16782,
    ];

    #[test]
    fn hadamard_group_256_matches_the_regular_corners() {
        assert_eq!(hadamard(256, 0, 0), 0.0625);
        assert_eq!(hadamard(256, 0, 3), -0.0625);
        assert_eq!(hadamard(256, 3, 0), -0.0625);
        let mut sum = 0.0f32;
        for col in 0..256 {
            sum += hadamard(256, 0, col);
        }
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn convrot_matches_the_definition_at_both_row_counts() -> Result<()> {
        let k = 256;
        let n = 2;
        let group = 256;
        let qweight = weight(n, k);
        let wscale = [0.0125f64 as f32, 0.03125f64 as f32];
        for (rows, expected) in [(1, &M1[..]), (17, &M17[..]), (20, &M20[..])] {
            let y = linear(&activation(rows, k), k, group, &qweight, n, &wscale)?;
            let got = bits(&y);
            let mut differ = 0usize;
            for (a, b) in got.iter().zip(expected) {
                if a != b {
                    differ += 1;
                }
            }
            ensure!(
                differ == 0 && got.len() == expected.len(),
                "convrot M={rows} differing bits {differ}"
            );
        }
        let y = linear(&activation(1, k), k, group, &qweight, n, &wscale)?;
        let mut shifted = wscale;
        shifted[0] *= 2.0;
        let scale_off = linear(&activation(1, k), k, group, &qweight, n, &shifted)?;
        ensure!(
            bits(&scale_off) != bits(&y),
            "a doubled wscale reproduced the definition bits"
        );
        let omitted = linear_omitting_rotation(&activation(1, k), k, &qweight, n, &wscale)?;
        ensure!(
            bits(&omitted) != M1.to_vec() && bits(&omitted) != bits(&y),
            "omitting the rotation reproduced the definition bits"
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod device {
    use crate::backend::cuda::Device;
    use crate::backend::cuda::cubin;
    use crate::backend::gpu::{LaunchConfig, PushKernelArg};
    use crate::convrot::linear;
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    fn launch(
        device: &Device,
        x: &[f32],
        k: usize,
        group: usize,
        qweight: &[i8],
        n: usize,
        wscale: &[f32],
    ) -> Result<Vec<bf16>> {
        ensure!(
            k > 0 && n > 0 && x.len().is_multiple_of(k),
            "convrot launch shape"
        );
        let rows = x.len() / k;
        let total = rows
            .checked_mul(n)
            .context("convrot launch rows overflow")?;
        ensure!(
            total <= i32::MAX as usize && k <= i32::MAX as usize && group <= i32::MAX as usize,
            "convrot launch extent"
        );
        let stream = &device.stream;
        let x_d = stream.clone_htod(x)?;
        let weight_d = stream.clone_htod(qweight)?;
        let scale_d = stream.clone_htod(wscale)?;
        let mut y_d = stream.alloc_zeros::<bf16>(total)?;
        let function = cubin::module(&device.ctx, "convrot")?.load_function("convrot_linear")?;
        let threads = 128u32;
        let blocks = (total as u32).div_ceil(threads);
        let rows_i = rows as i32;
        let k_i = k as i32;
        let n_i = n as i32;
        let group_i = group as i32;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&x_d)
                .arg(&weight_d)
                .arg(&scale_d)
                .arg(&mut y_d)
                .arg(&rows_i)
                .arg(&k_i)
                .arg(&n_i)
                .arg(&group_i)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&y_d)?)
    }

    fn row(r: usize, i: usize) -> f32 {
        ((((r * 17 + i * 3) % 23) as f64) * 0.05 - 0.4) as f32
    }

    fn code(n: usize, i: usize) -> i8 {
        (((n * 13 + i * 5) % 255) as i16 - 127) as i8
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_linear_kernel_matches_the_host_bits() -> Result<()> {
        let device = Device::new(0)?;
        let k = 256usize;
        let n = 2usize;
        let group = 256usize;
        let qweight: Vec<i8> = (0..n)
            .flat_map(|col| (0..k).map(move |i| code(col, i)))
            .collect();
        let wscale = [0.0125f64 as f32, 0.03125f64 as f32];
        for rows in [1usize, 17, 20] {
            let x: Vec<f32> = (0..rows)
                .flat_map(|r| (0..k).map(move |i| row(r, i)))
                .collect();
            let host = linear(&x, k, group, &qweight, n, &wscale)?;
            let device_y = launch(&device, &x, k, group, &qweight, n, &wscale)?;
            let differ = host
                .iter()
                .zip(&device_y)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                differ == 0 && host.len() == device_y.len(),
                "convrot kernel M={rows} differing bits {differ}"
            );
        }
        Ok(())
    }
}
