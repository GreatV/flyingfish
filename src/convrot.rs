//! Int8 linear with a block-diagonal regular Hadamard rotate-quant prologue.
//! `y = bf16((q(x R) @ qweight^T) * s_x * wscale)`, int32 products.
//! `R` is `kron(H4 / 2, ...)` of `group` (a power of 4), not stored.
//! Below [`TENSOR_CORE_ROWS`], `x` is cast to bf16, each rotated group is an f32 dot rounded
//! back to bf16, `s_x = max(|xR|, 1e-12) / 127`, and `q` is ties-to-even of `xR / s_x`.
//! At [`TENSOR_CORE_ROWS`] and above, the device kernel's dot is `mma.m16n8k16` (bf16 in, f32
//! accumulator, K in groups of 16) and `q` is ties-to-even of `xR * (1/s_x)`.
//! [`linear`] is the below-[`TENSOR_CORE_ROWS`] form.
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{Result, ensure};
use half::bf16;

/// Row count where the device path switches to `mma.m16n8k16`.
pub const TENSOR_CORE_ROWS: usize = 17;

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
/// Rows at or above [`TENSOR_CORE_ROWS`] are the device kernel.
pub fn linear(
    x: &[f32],
    k: usize,
    group: usize,
    qweight: &[i8],
    n: usize,
    wscale: &[f32],
) -> Result<Vec<bf16>> {
    ensure!(
        k > 0 && x.len().is_multiple_of(k),
        "convrot activation shape"
    );
    ensure!(
        x.len() / k < TENSOR_CORE_ROWS,
        "convrot rows are the tensor-core path"
    );
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

    /// Definition forward below 17 rows, group 256, N = 2.
    /// bf16 bits, wscale 0.0125 and 0.03125.
    const M1: [i16; 2] = [16484, 16398];
    const M16: [i16; 32] = [
        16484, 16398, -16336, -16045, -16089, -15970, -16076, -16099, -16997, -16278, 16435, 16585,
        -16278, -16477, -16356, 16681, 16678, 16796, 16521, 16572, -16230, -16464, -16395, 16690,
        16435, -16199, -16213, -15994, -16089, -16161, -16219, -16128,
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
    fn convrot_matches_the_definition_below_the_tensor_core_rows() -> Result<()> {
        let k = 256;
        let n = 2;
        let group = 256;
        let qweight = weight(n, k);
        let wscale = [0.0125f64 as f32, 0.03125f64 as f32];
        for (rows, expected) in [(1, &M1[..]), (16, &M16[..])] {
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

#[cfg(feature = "cuda")]
pub struct ConvrotFns {
    linear: crate::backend::gpu::CudaFunction,
    group_dot: crate::backend::gpu::CudaFunction,
    quantize: crate::backend::gpu::CudaFunction,
    product: crate::backend::gpu::CudaFunction,
}

#[cfg(feature = "cuda")]
pub struct ConvrotArgs {
    pub x: u64,
    pub qweight: u64,
    pub wscale: u64,
    pub y: u64,
    pub rotated: u64,
    pub codes: u64,
    pub scales: u64,
    pub rows: i32,
    pub k: i32,
    pub n: i32,
    pub group: i32,
}

#[cfg(feature = "cuda")]
pub fn load_convrot(ctx: &std::sync::Arc<crate::backend::gpu::CudaContext>) -> Result<ConvrotFns> {
    let module = crate::backend::cuda::cubin::module(ctx, "convrot")?;
    Ok(ConvrotFns {
        linear: module.load_function("convrot_linear")?,
        group_dot: module.load_function("convrot_group_dot")?,
        quantize: module.load_function("convrot_quantize")?,
        product: module.load_function("convrot_product")?,
    })
}

#[cfg(feature = "cuda")]
pub fn launch_device(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
) -> Result<()> {
    ensure!(
        args.rows > 0 && args.k > 0 && args.n > 0 && args.group > 0,
        "convrot launch extent"
    );
    let total = args
        .rows
        .checked_mul(args.n)
        .context("convrot output extent")?;
    if args.rows < i32::try_from(TENSOR_CORE_ROWS)? {
        launch_rows(stream, &fns.linear, args, total)
    } else {
        launch_groups(stream, fns, args, total)
    }
}

#[cfg(feature = "cuda")]
fn launch_rows(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    function: &crate::backend::gpu::CudaFunction,
    args: &ConvrotArgs,
    total: i32,
) -> Result<()> {
    use crate::backend::gpu::{LaunchConfig, PushKernelArg};
    let threads = 128u32;
    unsafe {
        stream
            .launch_builder(function)
            .arg(&args.x)
            .arg(&args.qweight)
            .arg(&args.wscale)
            .arg(&args.y)
            .arg(&args.rows)
            .arg(&args.k)
            .arg(&args.n)
            .arg(&args.group)
            .launch(LaunchConfig {
                grid_dim: ((total as u32).div_ceil(threads), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_groups(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
    total: i32,
) -> Result<()> {
    use crate::backend::gpu::{LaunchConfig, PushKernelArg};
    ensure!(
        args.group % 16 == 0 && args.k % args.group == 0,
        "convrot tensor-core shape"
    );
    let rows = args.rows as u32;
    let group = args.group as u32;
    let k = args.k as u32;
    unsafe {
        stream
            .launch_builder(&fns.group_dot)
            .arg(&args.x)
            .arg(&args.rotated)
            .arg(&args.rows)
            .arg(&args.k)
            .arg(&args.group)
            .launch(LaunchConfig {
                grid_dim: (rows.div_ceil(16), group / 8, k / group),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let quant_threads = 128u32;
    unsafe {
        stream
            .launch_builder(&fns.quantize)
            .arg(&args.rotated)
            .arg(&args.codes)
            .arg(&args.scales)
            .arg(&args.rows)
            .arg(&args.k)
            .launch(LaunchConfig {
                grid_dim: (rows.div_ceil(quant_threads), 1, 1),
                block_dim: (quant_threads, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let threads = 128u32;
    unsafe {
        stream
            .launch_builder(&fns.product)
            .arg(&args.codes)
            .arg(&args.scales)
            .arg(&args.qweight)
            .arg(&args.wscale)
            .arg(&args.y)
            .arg(&args.rows)
            .arg(&args.k)
            .arg(&args.n)
            .launch(LaunchConfig {
                grid_dim: ((total as u32).div_ceil(threads), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod device {
    use crate::backend::cuda::Device;
    use crate::backend::gpu::DevicePtr;
    use crate::convrot::{ConvrotArgs, launch_device, load_convrot};
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    const M1: [i16; 2] = [16484, 16398];

    const M16: [i16; 32] = [
        16484, 16398, -16336, -16045, -16089, -15970, -16076, -16099, -16997, -16278, 16435, 16585,
        -16278, -16477, -16356, 16681, 16678, 16796, 16521, 16572, -16230, -16464, -16395, 16690,
        16435, -16199, -16213, -15994, -16089, -16161, -16219, -16128,
    ];

    const M17: [i16; 34] = [
        16483, 16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529,
    ];

    const M20: [i16; 40] = [
        16483, 16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782,
    ];

    const M256: [i16; 512] = [
        16483, 16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782, 16642, 16560, -16256, -16247, -16366, 16725, 16483,
        16396, -16338, -16046, -16089, -15969,
    ];

    fn row(r: usize, i: usize) -> f32 {
        ((((r * 17 + i * 3) % 23) as f64) * 0.05 - 0.4) as f32
    }

    fn code(col: usize, i: usize) -> i8 {
        (((col * 13 + i * 5) % 255) as i16 - 127) as i8
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

    fn wscale_row(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    0.0125f64 as f32
                } else {
                    0.03125f64 as f32
                }
            })
            .collect()
    }

    fn embedded(rows: usize, k: usize, n: usize) -> Option<&'static [i16]> {
        if k != 256 || n != 2 {
            return None;
        }
        match rows {
            1 => Some(&M1),
            16 => Some(&M16),
            17 => Some(&M17),
            20 => Some(&M20),
            256 => Some(&M256),
            _ => None,
        }
    }

    fn definition_bits(rows: usize, k: usize, n: usize) -> Result<Vec<i16>> {
        if let Some(bits) = embedded(rows, k, n) {
            return Ok(bits.to_vec());
        }
        let dir =
            std::env::var("FF_HYPERFLOW_CONVROT_BITS").context("FF_HYPERFLOW_CONVROT_BITS")?;
        let path = std::path::Path::new(&dir).join(format!("m{rows}_k{k}_n{n}.i16"));
        let bytes = std::fs::read(&path).context("convrot definition bits")?;
        let nbytes = bytes.len();
        let (words, rest) = bytes.as_chunks::<2>();
        ensure!(
            rest.is_empty() && words.len() == rows * n,
            "convrot definition bit length rows={rows} n={n} bytes={nbytes}"
        );
        Ok(words.iter().map(|word| i16::from_le_bytes(*word)).collect())
    }

    fn differ(y: &[bf16], expected: &[i16]) -> usize {
        let mismatched = y
            .iter()
            .zip(expected)
            .filter(|(got, want)| got.to_bits() as i16 != **want)
            .count();
        mismatched + y.len().abs_diff(expected.len())
    }

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
            k > 0 && n > 0 && group > 0 && x.len().is_multiple_of(k),
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
        let y_d = stream.alloc_zeros::<bf16>(total)?;
        let rotated = stream.alloc_zeros::<f32>(rows * k)?;
        let codes = stream.alloc_zeros::<i8>(rows * k)?;
        let scales = stream.alloc_zeros::<f32>(rows)?;
        let fns = load_convrot(&device.ctx)?;
        launch_device(
            stream,
            &fns,
            &ConvrotArgs {
                x: x_d.device_ptr(stream).0,
                qweight: weight_d.device_ptr(stream).0,
                wscale: scale_d.device_ptr(stream).0,
                y: y_d.device_ptr(stream).0,
                rotated: rotated.device_ptr(stream).0,
                codes: codes.device_ptr(stream).0,
                scales: scales.device_ptr(stream).0,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: i32::try_from(n)?,
                group: i32::try_from(group)?,
            },
        )?;
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&y_d)?)
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_kernel_matches_the_definition() -> Result<()> {
        let device = Device::new(0)?;
        let group = 256usize;
        let shapes = [
            (1usize, 256usize, 2usize),
            (16, 256, 2),
            (17, 256, 2),
            (20, 256, 2),
            (256, 256, 2),
            (17, 5376, 7168),
            (17, 7168, 5376),
            (17, 14336, 5376),
        ];
        for (rows, k, n) in shapes {
            let expected = definition_bits(rows, k, n)?;
            let y = launch(
                &device,
                &activation(rows, k),
                k,
                group,
                &weight(n, k),
                n,
                &wscale_row(n),
            )?;
            let mismatched = differ(&y, &expected);
            ensure!(
                mismatched == 0,
                "convrot kernel M={rows} K={k} N={n} differing bits {mismatched}"
            );
        }
        let rows = 17usize;
        let k = 256usize;
        let n = 2usize;
        let x = activation(rows, k);
        let qweight = weight(n, k);
        let wscale = wscale_row(n);
        let y = launch(&device, &x, k, group, &qweight, n, &wscale)?;
        let mut shifted = wscale.clone();
        shifted[0] *= 2.0;
        let scale_off = launch(&device, &x, k, group, &qweight, n, &shifted)?;
        ensure!(
            differ(&scale_off, &definition_bits(rows, k, n)?) > 0,
            "a doubled wscale reproduced the definition bits"
        );
        let omitted = super::linear_omitting_rotation(&x, k, &qweight, n, &wscale)?;
        ensure!(
            differ(&omitted, &definition_bits(rows, k, n)?) > 0
                && differ(
                    &omitted,
                    &y.iter().map(|v| v.to_bits() as i16).collect::<Vec<_>>()
                ) > 0,
            "omitting the rotation reproduced the definition bits"
        );
        Ok(())
    }
}
