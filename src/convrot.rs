//! Int8 linear with a block-diagonal regular Hadamard rotate-quant prologue.
//! `y = bf16((q(x R) @ qweight^T) * s_x * wscale)`, int32 products.
//! `R` is `kron(H4 / 2, ...)` of `group` (a power of 4). At [`TENSOR_CORE_ROWS`] the device holds
//! `group * group` bf16 entries, `table[col * group + row]`.
//! Below [`TENSOR_CORE_ROWS`], `x` is f32 cast to bf16, each rotated group is an f32 dot rounded
//! back to bf16, `s_x = max(|xR|, 1e-12) / 127`, and `q` is ties-to-even of `xR / s_x`.
//! At [`TENSOR_CORE_ROWS`] and above, `x` is bf16. The device kernel's dot is `mma.m16n8k16` (bf16 in, f32
//! accumulator, K in groups of 16) and `q` is ties-to-even of `xR * (1/s_x)`.
//! The int8 product is `mma.m16n8k32` on a 128×256×64 tile, 512 threads, 49,152 or 61,440 bytes dynamic smem.
//! [`linear`] is the below-[`TENSOR_CORE_ROWS`] form.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
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

fn write_product_row(
    xq: &[i8],
    qweight: &[i8],
    k: usize,
    sx: f32,
    wscale: &[f32],
    dst: &mut [bf16],
) {
    for (col, dst) in dst.iter_mut().enumerate() {
        let weight = &qweight[col * k..(col + 1) * k];
        let mut acc = 0i32;
        for (a, b) in xq.iter().zip(weight) {
            acc += i32::from(*a) * i32::from(*b);
        }
        *dst = bf16::from_f32(acc as f32 * sx * wscale[col]);
    }
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
    let mut y = vec![bf16::ZERO; rows * n];
    for row in 0..rows {
        write_product_row(
            &codes[row * k..(row + 1) * k],
            qweight,
            k,
            scales[row],
            wscale,
            &mut y[row * n..(row + 1) * n],
        );
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

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn group_dot_b_smem(arch: &crate::backend::setup::DeviceArchitecture) -> u32 {
    match arch {
        crate::backend::setup::DeviceArchitecture::Cuda(_) => 2048,
        crate::backend::setup::DeviceArchitecture::Hip(_) => 0,
    }
}

/// Dynamic shared bytes of one 16-row group tile (CUDA adds eight 8×16 bf16 Hadamard tiles).
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn group_dot_smem(group: u32) -> u32 {
    #[cfg(feature = "rocm")]
    {
        32 * group
    }
    #[cfg(not(feature = "rocm"))]
    {
        32 * group + 2048
    }
}

// A 32-row grid must fill at least one wave of resident CTAs.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn group_dot_32_fills_wave(rows: usize, k: usize, wave: Option<usize>) -> bool {
    let ctas = rows.div_ceil(32) * (k / 256);
    let enabled = rows >= TENSOR_CORE_ROWS && wave.is_some_and(|wave| ctas >= wave);
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if crate::backend::diagnostics_enabled() {
        let tile_rows = if enabled { 32 } else { 16 };
        let wave = wave.unwrap_or(0);
        eprintln!(
            "CONVROT_ROTATE_32_SELECT rows={rows} k={k} ctas={ctas} wave={wave} tile_rows={tile_rows}"
        );
    }
    enabled
}

/// Largest multiple of 16 whose [`group_dot_smem`] fits in `optin` bytes.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn max_group(optin: usize) -> Result<u32> {
    let bytes = u32::try_from(optin).context("convrot opt-in shared memory")?;
    let b_smem = group_dot_smem(0);
    ensure!(
        bytes >= b_smem + 512,
        "convrot opt-in shared memory {optin}"
    );
    let raw = (bytes - b_smem) / 32;
    let group = raw / 16 * 16;
    ensure!(group >= 16, "convrot opt-in shared memory {optin}");
    ensure!(
        group_dot_smem(group) <= bytes,
        "convrot group {group} tile exceeds opt-in {optin}"
    );
    Ok(group)
}

/// `group` fits the tile derived from `optin`. Returns that tile.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn ensure_group(group: u32, optin: usize) -> Result<u32> {
    let max = max_group(optin)?;
    ensure!(
        group <= max,
        "convrot group {group} exceeds device tile {max}"
    );
    Ok(max)
}

/// Rows in one `convrot_quantize` tile: `optin / (k * 4)`.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn quantize_rows(optin: usize, k: u32) -> Result<u32> {
    let bytes = u32::try_from(optin).context("convrot opt-in shared memory")?;
    let width = k.checked_mul(4).context("convrot quantize row")?;
    ensure!(
        width > 0 && width <= bytes,
        "convrot quantize k {k} exceeds opt-in {optin}"
    );
    Ok(bytes / width)
}

#[cfg(test)]
mod tests {
    use super::{hadamard, linear, linear_omitting_rotation};
    use anyhow::{Result, ensure};
    use half::bf16;

    #[test]
    fn group_dot_32_requires_a_full_wave() {
        assert!(!super::group_dot_32_fills_wave(960, 5376, Some(640)));
        assert!(super::group_dot_32_fills_wave(961, 5376, Some(640)));
        assert!(super::group_dot_32_fills_wave(960, 5376, Some(630)));
    }

    fn mock_device_info(
        sm: u32,
        sms: usize,
        l2_bytes: usize,
        optin_shared_bytes: usize,
    ) -> crate::backend::setup::DeviceInfo {
        crate::backend::setup::DeviceInfo {
            name: format!("mock-sm{sm}"),
            arch: crate::backend::setup::DeviceArchitecture::Cuda(sm),
            sms,
            l2_bytes,
            warp: 32,
            max_threads: 1024,
            shared_bytes: 49_152,
            sm_shared_bytes: 102_400,
            cooperative: true,
            optin_shared_bytes,
            memory_bytes_per_second: 1.0e12,
        }
    }

    fn mock_hip_device_info(
        gfx: &str,
        sms: usize,
        l2_bytes: usize,
        optin_shared_bytes: usize,
    ) -> crate::backend::setup::DeviceInfo {
        crate::backend::setup::DeviceInfo {
            name: format!("mock-{gfx}"),
            arch: crate::backend::setup::DeviceArchitecture::Hip(gfx.to_owned()),
            sms,
            l2_bytes,
            warp: 32,
            max_threads: 1024,
            shared_bytes: 65_536,
            sm_shared_bytes: 65_536,
            cooperative: true,
            optin_shared_bytes,
            memory_bytes_per_second: 8.64e11,
        }
    }

    #[test]
    fn convrot_selectors_qualify_sm80_and_decline_sm86_by_resources() {
        use super::{
            ProductKernel, gate_up_products, group_dot_32_resource_limits, qkv_product_eligible,
        };

        use ProductKernel::{MFastLdsm, MFastStriped, NFast};

        let a100 = mock_device_info(80, 108, 40 << 20, 166_912);
        let a4000 = mock_device_info(86, 48, 4 << 20, 101_376);
        let a4000_large_l2 = mock_device_info(86, 48, 40 << 20, 101_376);
        let rtx4090 = mock_device_info(89, 128, 72 << 20, 101_376);
        let l4 = mock_device_info(89, 58, 48 << 20, 101_376);
        let g4 = mock_device_info(120, 188, 128 << 20, 101_376);
        let t4 = mock_device_info(75, 40, 40 << 20, 65_536);
        let gfx1100 = mock_hip_device_info("gfx1100", 48, 6 << 20, 65_536);
        let gfx1100_large_l2 = mock_hip_device_info("gfx1100", 48, 40 << 20, 65_536);
        let gfx1100_low_smem = mock_hip_device_info("gfx1100", 48, 40 << 20, 32_768);
        let gfx1030 = mock_hip_device_info("gfx1030", 48, 40 << 20, 65_536);

        assert_eq!(ProductKernel::NFast.symbol_name(), "convrot_product");
        assert_eq!(
            ProductKernel::MFastLdsm.symbol_name(),
            "convrot_product_m_fast"
        );
        assert_eq!(
            ProductKernel::NFastLdsm.symbol_name(),
            "convrot_product_n_fast_ldsm"
        );
        assert_eq!(ProductKernel::NFast.shared_mem_bytes(&rtx4090.arch), 61_440);
        assert_eq!(
            ProductKernel::MFastLdsm.shared_mem_bytes(&rtx4090.arch),
            49_152
        );
        assert_eq!(
            ProductKernel::NFastLdsm.shared_mem_bytes(&rtx4090.arch),
            61_440
        );
        assert_eq!(ProductKernel::NFast.shared_mem_bytes(&gfx1100.arch), 32_768);
        assert_eq!(
            ProductKernel::MFastLdsm.shared_mem_bytes(&gfx1100.arch),
            32_768
        );
        assert_eq!(
            ProductKernel::NFastLdsm.shared_mem_bytes(&gfx1100.arch),
            49_152
        );
        assert_eq!(ProductKernel::NFast.tile_dims(&gfx1100.arch), (128, 128));
        assert_eq!(
            ProductKernel::MFastLdsm.tile_dims(&gfx1100.arch),
            (128, 128)
        );
        assert_eq!(
            ProductKernel::NFastLdsm.tile_dims(&gfx1100.arch),
            (128, 256)
        );

        assert!(qkv_product_eligible(&a100, 37_717, (5376, 7168)));
        assert!(!qkv_product_eligible(&a100, 16, (5376, 7168)));
        assert!(!qkv_product_eligible(&a100, 37_717, (5376, 7169)));
        assert!(!qkv_product_eligible(&a4000, 37_717, (5376, 7168)));
        assert!(qkv_product_eligible(&a4000_large_l2, 37_717, (5376, 7168)));
        assert!(qkv_product_eligible(&rtx4090, 37_717, (5376, 7168)));
        assert!(qkv_product_eligible(&l4, 37_717, (5376, 7168)));
        assert!(qkv_product_eligible(&g4, 37_717, (5376, 7168)));
        assert!(!qkv_product_eligible(&t4, 37_717, (5376, 7168)));
        assert!(!qkv_product_eligible(&gfx1100, 37_717, (5376, 7168)));
        assert!(qkv_product_eligible(
            &gfx1100_large_l2,
            37_717,
            (5376, 7168)
        ));
        assert!(!qkv_product_eligible(
            &gfx1100_low_smem,
            37_717,
            (5376, 7168)
        ));
        assert!(!qkv_product_eligible(&gfx1030, 37_717, (5376, 7168)));
        let low_smem = mock_device_info(80, 108, 40 << 20, 49_152);
        assert!(!qkv_product_eligible(&low_smem, 37_717, (5376, 7168)));

        for dims in [(5376, 7168), (7168, 5376), (14336, 5376)] {
            assert!(super::n_fast_swizzle_fits(&g4, 37_717, dims, 1));
            assert!(!super::n_fast_swizzle_fits(&gfx1100, 37_717, dims, 1));
            assert!(!super::n_fast_swizzle_fits(&rtx4090, 37_717, dims, 1));
            assert!(!super::n_fast_swizzle_fits(&g4, 17, dims, 1));
        }
        let small_l2 = mock_device_info(120, 188, 80 << 20, 101_376);
        assert!(!super::n_fast_swizzle_fits(
            &small_l2,
            37_717,
            (14336, 5376),
            1
        ));
        assert!(super::n_fast_swizzle_fits(
            &small_l2,
            37_717,
            (7168, 5376),
            1
        ));

        let gate_up = (5376, 28_672);
        assert_eq!(
            gate_up_products(&a100, 1, [1509, 1501], gate_up),
            [MFastLdsm; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [3057, 5400], gate_up),
            [MFastLdsm; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [5678, 39_740], gate_up),
            [NFast; 2]
        );
        assert_eq!(gate_up_products(&a100, 1, [128, 129], gate_up), [NFast; 2]);
        assert_eq!(
            gate_up_products(&a100, 1, [128, 513], gate_up),
            [NFast, MFastLdsm]
        );
        assert_eq!(
            gate_up_products(&a4000, 1, [1509, 1501], gate_up),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a4000, 1, [256, 37_717], gate_up),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a4000_large_l2, 1, [128, 129], gate_up),
            [NFast, MFastLdsm]
        );
        assert_eq!(
            gate_up_products(&rtx4090, 1, [9935, 39_740], gate_up),
            [MFastLdsm, NFast]
        );
        assert_eq!(
            gate_up_products(&g4, 1, [9935, 37_717], gate_up),
            [
                MFastLdsm,
                MFastStriped {
                    tiles: 59,
                    stripes: 5,
                    one_barrier: true,
                }
            ]
        );
        assert_eq!(gate_up_products(&t4, 1, [1509, 1501], gate_up), [NFast; 2]);
        assert_eq!(
            gate_up_products(&gfx1100, 2, [1509, 1501], gate_up),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&gfx1100, 2, [9935, 37_717], gate_up),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&gfx1100_large_l2, 2, [1509, 37_717], gate_up),
            [
                MFastLdsm,
                MFastStriped {
                    tiles: 19,
                    stripes: 16,
                    one_barrier: false,
                }
            ]
        );
        assert_eq!(
            gate_up_products(&gfx1030, 2, [1509, 37_717], gate_up),
            [NFast; 2]
        );

        assert_eq!(
            group_dot_32_resource_limits(&a100),
            Some([(48, 5), (48, 5)])
        );
        assert_eq!(
            group_dot_32_resource_limits(&a4000),
            Some([(48, 5), (48, 5)])
        );
        assert_eq!(
            group_dot_32_resource_limits(&rtx4090),
            Some([(48, 5), (48, 5)])
        );
        assert_eq!(group_dot_32_resource_limits(&g4), Some([(48, 5), (56, 4)]));
        assert_eq!(group_dot_32_resource_limits(&t4), None);
        assert_eq!(
            group_dot_32_resource_limits(&gfx1100),
            Some([(96, 4), (96, 4)])
        );
        assert_eq!(group_dot_32_resource_limits(&gfx1030), None);
        let too_small_smem = mock_device_info(80, 108, 40 << 20, 16_384);
        assert_eq!(group_dot_32_resource_limits(&too_small_smem), None);
        let tiny_hip_smem = mock_hip_device_info("gfx1100", 48, 6 << 20, 8_192);
        assert_eq!(group_dot_32_resource_limits(&tiny_hip_smem), None);
    }

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

    #[test]
    fn group_tile_fits_the_optin() -> Result<()> {
        let optin = 99usize << 10;
        let max = super::max_group(optin)?;
        let expected_1024 = if cfg!(feature = "rocm") {
            32_768
        } else {
            34_816
        };
        ensure!(
            max >= 1024
                && super::group_dot_smem(1024) == expected_1024
                && usize::try_from(super::group_dot_smem(max))? <= optin
                && super::ensure_group(1024, optin)? == max
                && super::max_group(64usize << 10)? >= 1024,
            "group tile optin={optin} max={max}"
        );
        ensure!(
            super::ensure_group(max + 16, optin).is_err(),
            "group above the tile was accepted"
        );
        Ok(())
    }

    #[test]
    fn quantize_rows_fit_the_optin() -> Result<()> {
        let optin = 99usize << 10;
        ensure!(
            super::quantize_rows(optin, 5376)? == 4
                && super::quantize_rows(optin, 7168)? == 3
                && super::quantize_rows(optin, 14336)? == 1,
            "quantize rows optin={optin}"
        );
        ensure!(
            super::quantize_rows(57_343, 14336).is_err(),
            "k above the opt-in was accepted"
        );
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub struct ConvrotFns {
    linear: crate::backend::gpu::CudaFunction,
    hadamard: crate::backend::gpu::CudaFunction,
    group_dot: [crate::backend::gpu::CudaFunction; 2],
    group_dot_32: [crate::backend::gpu::CudaFunction; 2],
    quantize: crate::backend::gpu::CudaFunction,
    product: [crate::backend::gpu::CudaFunction; 3],
    arch: crate::backend::setup::DeviceArchitecture,
    striped: bool,
    m_fast_one_barrier: Option<crate::backend::gpu::CudaFunction>,
    n_fast_swizzled: Option<crate::backend::gpu::CudaFunction>,
    table: std::sync::Mutex<Option<(u32, i32, crate::backend::gpu::CudaSlice<u16>)>>,
    max_group: u32,
    optin: usize,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
std::thread_local! {
    static HADAMARD_FLIP: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
}

#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
pub(crate) fn with_hadamard_flip<T>(index: i32, body: impl FnOnce() -> T) -> T {
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            HADAMARD_FLIP.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(HADAMARD_FLIP.with(|cell| cell.replace(index)));
    let result = body();
    drop(restore);
    result
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductKernel {
    NFast,
    MFastLdsm,
    MFastStriped {
        tiles: u32,
        stripes: u32,
        one_barrier: bool,
    },
    NFastLdsm,
    NFastSwizzled,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
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
    pub product: ProductKernel,
    pub fuse_silu: bool,
    pub rotate_32: bool,
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
impl ProductKernel {
    fn shared_mem_bytes(self, arch: &crate::backend::setup::DeviceArchitecture) -> u32 {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => match self {
                Self::MFastLdsm | Self::MFastStriped { .. } | Self::NFastSwizzled => 49_152,
                Self::NFast | Self::NFastLdsm => 61_440,
            },
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm | Self::MFastStriped { .. } => 32_768,
                Self::NFastLdsm | Self::NFastSwizzled => 49_152,
            },
        }
    }

    fn tile_dims(self, arch: &crate::backend::setup::DeviceArchitecture) -> (u32, u32) {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => (128, 256),
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm | Self::MFastStriped { .. } => (128, 128),
                Self::NFastLdsm | Self::NFastSwizzled => (128, 256),
            },
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn threads(self, arch: &crate::backend::setup::DeviceArchitecture) -> u32 {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => 512,
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm | Self::MFastStriped { .. } => 256,
                Self::NFastLdsm | Self::NFastSwizzled => 512,
            },
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn resource_limits(self, arch: &crate::backend::setup::DeviceArchitecture) -> (i32, u32) {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => (124, 1),
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm | Self::MFastStriped { .. } => (216, 2),
                Self::NFastLdsm | Self::NFastSwizzled => (216, 1),
            },
        }
    }

    const ONE_BARRIER_SYMBOL: &'static str = "convrot_product_m_fast_one_barrier";

    pub(crate) const fn symbol_name(self) -> &'static str {
        match self {
            Self::NFast => "convrot_product",
            Self::MFastStriped {
                one_barrier: true, ..
            } => Self::ONE_BARRIER_SYMBOL,
            Self::MFastLdsm | Self::MFastStriped { .. } => "convrot_product_m_fast",
            Self::NFastLdsm => "convrot_product_n_fast_ldsm",
            Self::NFastSwizzled => "convrot_product_n_fast_swizzled",
        }
    }
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn arch_supported(arch: &crate::backend::setup::DeviceArchitecture) -> bool {
    use crate::backend::setup::DeviceArchitecture::{Cuda, Hip};
    match arch {
        Cuda(sm) => *sm >= 80,
        Hip(gfx) => gfx.starts_with("gfx11"),
    }
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn arch_supports_striped(arch: &crate::backend::setup::DeviceArchitecture) -> bool {
    use crate::backend::setup::DeviceArchitecture::{Cuda, Hip};
    match arch {
        Cuda(120) => true,
        Hip(gfx) => gfx.starts_with("gfx11"),
        _ => false,
    }
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn qkv_product_eligible(
    info: &crate::backend::setup::DeviceInfo,
    rows: usize,
    (k, n): (usize, usize),
) -> bool {
    arch_supported(&info.arch)
        && info.warp == 32
        && info.optin_shared_bytes >= ProductKernel::NFastLdsm.shared_mem_bytes(&info.arch) as usize
        && rows >= TENSOR_CORE_ROWS
        && (k, n) == (5376, 7168)
        && k.saturating_mul(n) <= info.l2_bytes
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn gate_up_products(
    info: &crate::backend::setup::DeviceInfo,
    active_ctas: usize,
    row_counts: [usize; 2],
    (k, n): (usize, usize),
) -> [ProductKernel; 2] {
    let l2 = info.l2_bytes;
    if !arch_supported(&info.arch)
        || info.warp != 32
        || info.optin_shared_bytes < ProductKernel::MFastLdsm.shared_mem_bytes(&info.arch) as usize
        || active_ctas == 0
        || n.saturating_mul(k) <= l2
    {
        return [ProductKernel::NFast; 2];
    }
    let wave = info.sms * active_ctas;
    let (bm_u32, bn_u32) = ProductKernel::MFastLdsm.tile_dims(&info.arch);
    let (bm, bn) = (bm_u32 as usize, bn_u32 as usize);
    let n_tiles = n.div_ceil(bn);
    let working_set = |rows: usize, m_tiles: usize| {
        let ctas = wave.min(m_tiles * n_tiles);
        let columns = ((ctas + m_tiles - 1).div_ceil(m_tiles) * bn).min(n);
        let output_bytes = ctas * bm * bn * 2;
        let scale_bytes = (rows + columns) * size_of::<f32>();
        (rows + columns) * k + output_bytes + scale_bytes
    };
    row_counts.map(|rows| {
        let m_tiles = rows.div_ceil(bm);
        if m_tiles < 2 {
            return ProductKernel::NFast;
        }
        if working_set(rows, m_tiles) < l2 {
            return ProductKernel::MFastLdsm;
        }
        if arch_supports_striped(&info.arch) && rows.saturating_mul(k) > l2 {
            for stripes in 2..=m_tiles.min(u16::MAX as usize) {
                let tiles = m_tiles.div_ceil(stripes);
                if tiles < 2 || tiles * n_tiles < wave {
                    break;
                }
                // A wave straddles at most two stripes.
                if working_set(tiles * bm, tiles) < l2 / 2 {
                    return ProductKernel::MFastStriped {
                        tiles: tiles as u32,
                        stripes: m_tiles.div_ceil(tiles) as u32,
                        one_barrier: info.arch
                            == crate::backend::setup::DeviceArchitecture::Cuda(120)
                            && (rows, k, n, tiles, m_tiles.div_ceil(tiles))
                                == (37_717, 5376, 28_672, 59, 5),
                    };
                }
            }
        }
        ProductKernel::NFast
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn function_regs_and_local(function: &crate::backend::gpu::CudaFunction) -> Result<(i32, i32)> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        Ok((
            function.get_attribute(Attr::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            function.get_attribute(Attr::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
        ))
    }
    #[cfg(feature = "rocm")]
    {
        Ok((
            function.registers_per_thread()?,
            function.local_size_bytes()?,
        ))
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn active_blocks(
    function: &crate::backend::gpu::CudaFunction,
    threads: u32,
    shared: usize,
) -> Result<u32> {
    let raw = function.occupancy_max_active_blocks_per_multiprocessor(threads, shared, None)?;
    #[cfg(feature = "cuda")]
    {
        Ok(raw)
    }
    #[cfg(feature = "rocm")]
    {
        Ok(u32::try_from(raw)?)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn product_function(
    module: &std::sync::Arc<crate::backend::gpu::CudaModule>,
    arch: &crate::backend::setup::DeviceArchitecture,
    product: ProductKernel,
) -> Result<crate::backend::gpu::CudaFunction> {
    let function = module.load_function(product.symbol_name())?;
    crate::backend::cuda::set_max_dynamic_shared_bytes(&function, product.shared_mem_bytes(arch))?;
    Ok(function)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn product_active_ctas(
    device: &crate::backend::cuda::Device,
    product: ProductKernel,
) -> Result<usize> {
    let smem = product.shared_mem_bytes(&device.info.arch) as usize;
    ensure!(
        arch_supported(&device.info.arch)
            && device.info.warp == 32
            && device.info.optin_shared_bytes >= smem,
        "ConvRot {} requires supported warp32 arch and {} optin shared bytes (arch={} warp={} optin={})",
        product.symbol_name(),
        smem,
        device.info.arch,
        device.info.warp,
        device.info.optin_shared_bytes,
    );
    let module = crate::backend::cuda::cubin::module(&device.ctx, "convrot")?;
    let function = product_function(&module, &device.info.arch, product)?;
    let active = active_blocks(&function, product.threads(&device.info.arch), smem)?;
    let (regs, local) = function_regs_and_local(&function)?;
    let (max_regs, min_ctas) = product.resource_limits(&device.info.arch);
    ensure!(
        regs <= max_regs && local == 0,
        "ConvRot {} regs={regs} local={local} active_ctas={active}",
        product.symbol_name()
    );
    ensure!(
        active >= min_ctas,
        "ConvRot {} has {active} resident CTAs (min {min_ctas})",
        product.symbol_name()
    );
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "CONVROT_PRODUCT_RESOURCE kernel={} regs={regs} local={local} active_ctas={active}",
            product.symbol_name()
        );
    }
    Ok(usize::try_from(active)?)
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn n_fast_swizzle_fits(
    info: &crate::backend::setup::DeviceInfo,
    rows: usize,
    (k, n): (usize, usize),
    active_ctas: usize,
) -> bool {
    if info.arch != crate::backend::setup::DeviceArchitecture::Cuda(120)
        || info.warp != 32
        || rows < TENSOR_CORE_ROWS
        || info.optin_shared_bytes
            < ProductKernel::NFastSwizzled.shared_mem_bytes(&info.arch) as usize
        || !matches!((k, n), (5376, 7168) | (7168 | 14336, 5376))
    {
        return false;
    }
    let (bm, bn) = ProductKernel::NFastSwizzled.tile_dims(&info.arch);
    let (bm, bn) = (bm as usize, bn as usize);
    let wave = info.sms * active_ctas;
    let n_tiles = n / bn;
    if wave == 0 || rows.div_ceil(bm).saturating_mul(n_tiles) < wave {
        return false;
    }
    let live_rows = ((wave - 1).div_ceil(n_tiles) + 1) * bm;
    let operands = n * k + live_rows.min(rows) * (k + size_of::<f32>());
    let output = wave * bm * bn * size_of::<bf16>();
    operands + output + n * size_of::<f32>() < info.l2_bytes
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn n_fast_product(
    device: &crate::backend::cuda::Device,
    rows: usize,
    dims: (usize, usize),
) -> Result<ProductKernel> {
    let info = &device.info;
    if n_fast_swizzle_fits(info, rows, dims, 1) {
        let active = product_active_ctas(device, ProductKernel::NFastSwizzled)?;
        if n_fast_swizzle_fits(info, rows, dims, active) {
            return Ok(ProductKernel::NFastSwizzled);
        }
    }
    if qkv_product_eligible(info, rows, dims) {
        product_active_ctas(device, ProductKernel::NFastLdsm)?;
        return Ok(ProductKernel::NFastLdsm);
    }
    Ok(ProductKernel::NFast)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn gate_up_product_order(
    device: &crate::backend::cuda::Device,
    row_counts: [usize; 2],
    (k, n): (usize, usize),
) -> Result<[ProductKernel; 2]> {
    let info = &device.info;
    let selected = if arch_supported(&info.arch)
        && info.warp == 32
        && info.optin_shared_bytes >= ProductKernel::MFastLdsm.shared_mem_bytes(&info.arch) as usize
        && n.saturating_mul(k) > info.l2_bytes
    {
        let active = product_active_ctas(device, ProductKernel::MFastLdsm)?;
        let selected = gate_up_products(info, active, row_counts, (k, n));
        if let Some(&kernel) = selected.iter().find(|p| {
            matches!(
                p,
                ProductKernel::MFastStriped {
                    one_barrier: true,
                    ..
                }
            )
        }) {
            ensure!(
                product_active_ctas(device, kernel)? == active,
                "one-barrier stripe residency changed"
            );
        }
        selected
    } else {
        [ProductKernel::NFast; 2]
    };
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "CONVROT_GATE_UP_SELECT row_counts={row_counts:?} k={k} n={n} l2_bytes={} products={selected:?}",
            info.l2_bytes
        );
    }
    Ok(selected)
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn group_dot_32_smem(arch: &crate::backend::setup::DeviceArchitecture) -> u32 {
    2 * 8192 + group_dot_b_smem(arch)
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn group_dot_32_resource_limits(
    info: &crate::backend::setup::DeviceInfo,
) -> Option<[(i32, u32); 2]> {
    use crate::backend::setup::DeviceArchitecture::{Cuda, Hip};
    if info.warp != 32 || info.optin_shared_bytes < group_dot_32_smem(&info.arch) as usize {
        return None;
    }
    match &info.arch {
        Cuda(120) => Some([(48, 5), (56, 4)]),
        Cuda(sm) if *sm >= 80 => Some([(48, 5), (48, 5)]),
        Hip(gfx) if gfx.starts_with("gfx11") => Some([(96, 4), (96, 4)]),
        _ => None,
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
const GROUP_DOT_32_NAMES: [&str; 2] = ["convrot_group_dot_32", "convrot_group_dot_silu_32"];

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn group_dot_32_wave_sizes(
    device: &crate::backend::cuda::Device,
) -> Result<[Option<usize>; 2]> {
    let Some(limits) = group_dot_32_resource_limits(&device.info) else {
        return Ok([None; 2]);
    };
    let smem = group_dot_32_smem(&device.info.arch) as usize;
    let module = crate::backend::cuda::cubin::module(&device.ctx, "convrot")?;
    let mut waves = [None; 2];
    for ((wave, name), (max_regs, min_ctas)) in waves.iter_mut().zip(GROUP_DOT_32_NAMES).zip(limits)
    {
        let function = module.load_function(name)?;
        let (registers, local) = function_regs_and_local(&function)?;
        let active = active_blocks(&function, 256, smem)?;
        let enabled = registers <= max_regs && local == 0 && active >= min_ctas;
        if enabled {
            *wave = Some(device.info.sms * usize::try_from(active)?);
        }
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "CONVROT_ROTATE_32_RESOURCE kernel={name} regs={registers} local={local} active_ctas={active} enabled={enabled}"
            );
        }
    }
    Ok(waves)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn load_convrot(ctx: &std::sync::Arc<crate::backend::gpu::CudaContext>) -> Result<ConvrotFns> {
    #[cfg(feature = "rocm")]
    ctx.require_wmma()?;
    let arch = {
        #[cfg(feature = "cuda")]
        {
            crate::backend::setup::DeviceArchitecture::Cuda(
                crate::backend::cuda::cubin::capability(ctx)?,
            )
        }
        #[cfg(feature = "rocm")]
        {
            crate::backend::setup::DeviceArchitecture::Hip(ctx.architecture().to_owned())
        }
    };
    let module = crate::backend::cuda::cubin::module(ctx, "convrot")?;
    let product = [
        product_function(&module, &arch, ProductKernel::NFast)?,
        product_function(&module, &arch, ProductKernel::MFastLdsm)?,
        product_function(&module, &arch, ProductKernel::NFastLdsm)?,
    ];
    let striped = {
        #[cfg(feature = "cuda")]
        {
            product[1].get_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
            )? == 120
        }
        #[cfg(feature = "rocm")]
        {
            ctx.architecture().starts_with("gfx11")
        }
    };
    let hadamard = module.load_function("convrot_hadamard_table")?;
    let group_dot = [
        module.load_function("convrot_group_dot")?,
        module.load_function("convrot_group_dot_silu")?,
    ];
    let group_dot_32 = [
        module.load_function(GROUP_DOT_32_NAMES[0])?,
        module.load_function(GROUP_DOT_32_NAMES[1])?,
    ];
    let optin = crate::backend::cuda::optin_shared_bytes(ctx)?;
    let max_group = max_group(optin)?;
    for function in &group_dot {
        crate::backend::cuda::set_max_dynamic_shared_bytes(function, group_dot_smem(max_group))?;
    }
    for function in &group_dot_32 {
        crate::backend::cuda::set_max_dynamic_shared_bytes(function, group_dot_32_smem(&arch))?;
    }
    let quantize = module.load_function("convrot_quantize")?;
    let quant_dyn = u32::try_from(optin / 4 * 4).context("convrot quantize opt-in")?;
    crate::backend::cuda::set_max_dynamic_shared_bytes(&quantize, quant_dyn)?;
    let n_fast_swizzled = (arch == crate::backend::setup::DeviceArchitecture::Cuda(120) && striped)
        .then(|| product_function(&module, &arch, ProductKernel::NFastSwizzled))
        .transpose()?;
    let m_fast_one_barrier = if n_fast_swizzled.is_some() {
        let f = module.load_function(ProductKernel::ONE_BARRIER_SYMBOL)?;
        crate::backend::cuda::set_max_dynamic_shared_bytes(
            &f,
            ProductKernel::MFastLdsm.shared_mem_bytes(&arch),
        )?;
        Some(f)
    } else {
        None
    };
    Ok(ConvrotFns {
        linear: module.load_function("convrot_linear")?,
        hadamard,
        group_dot,
        group_dot_32,
        quantize,
        arch,
        striped,
        n_fast_swizzled,
        m_fast_one_barrier,
        product,
        table: std::sync::Mutex::new(None),
        max_group,
        optin,
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
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
        ensure!(!args.fuse_silu, "fused ConvRot requires tensor-core rows");
        launch_rows(stream, &fns.linear, args, total)
    } else {
        launch_groups(stream, fns, args)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
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

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn launch_product(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
) -> Result<()> {
    use crate::backend::gpu::{LaunchConfig, PushKernelArg};
    ensure!(
        args.rows > 0 && args.n > 0 && args.k > 0 && args.k % 64 == 0,
        "convrot product shape"
    );
    let (bm, bn) = args.product.tile_dims(&fns.arch);
    let m = (args.rows as u32).div_ceil(bm);
    let n = (args.n as u32).div_ceil(bn);
    let (product, grid) = match args.product {
        ProductKernel::NFast => (&fns.product[0], (n, m, 1)),
        ProductKernel::MFastLdsm => (&fns.product[1], (m, n, 1)),
        ProductKernel::MFastStriped {
            tiles,
            stripes,
            one_barrier,
        } => {
            ensure!(
                fns.striped && tiles > 0 && tiles <= m && stripes == m.div_ceil(tiles),
                "ConvRot striped product arch={} m_tiles={m} tiles={tiles} stripes={stripes}",
                fns.arch
            );
            let kernel = if one_barrier {
                fns.m_fast_one_barrier
                    .as_ref()
                    .context("one-barrier product requires sm120")?
            } else {
                &fns.product[1]
            };
            (kernel, (tiles, n, stripes))
        }
        ProductKernel::NFastLdsm => (&fns.product[2], (n, m, 1)),
        ProductKernel::NFastSwizzled => (
            fns.n_fast_swizzled
                .as_ref()
                .context("swizzled N-fast product requires sm120")?,
            (n, m, 1),
        ),
    };
    unsafe {
        stream
            .launch_builder(product)
            .arg(&args.codes)
            .arg(&args.scales)
            .arg(&args.qweight)
            .arg(&args.wscale)
            .arg(&args.y)
            .arg(&args.rows)
            .arg(&args.k)
            .arg(&args.n)
            .launch(LaunchConfig {
                grid_dim: grid,
                block_dim: (args.product.threads(&fns.arch), 1, 1),
                shared_mem_bytes: args.product.shared_mem_bytes(&fns.arch),
            })?;
    }
    Ok(())
}

/// Grid of `convrot_group_dot`: one block owns one row tile and one whole group.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn group_dot_grid(rows: u32, k: u32, group: u32, tile_rows: u32) -> (u32, u32, u32) {
    (rows.div_ceil(tile_rows), 1, k / group)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn launch_group_dot(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
    rows: u32,
    k: u32,
    group: u32,
) -> Result<()> {
    use crate::backend::gpu::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    ensure!(
        !args.rotate_32 || group == 256,
        "32-row ConvRot requires group 256, got group {group}"
    );
    let flip = HADAMARD_FLIP.with(|cell| cell.get());
    let mut slot = fns.table.lock().unwrap_or_else(|err| err.into_inner());
    let hit = slot.as_ref().is_some_and(|(cached_group, cached_flip, _)| {
        *cached_group == group && *cached_flip == flip
    });
    if !hit {
        let entries = u64::from(group) * u64::from(group);
        let count = u32::try_from(entries).context("convrot hadamard table")?;
        let mut table = stream.alloc_zeros::<u16>(usize::try_from(count)?)?;
        let group_i = i32::try_from(group)?;
        {
            let (ptr, _wrote) = table.device_ptr_mut(stream);
            let threads = 256u32;
            unsafe {
                stream
                    .launch_builder(&fns.hadamard)
                    .arg(&ptr)
                    .arg(&group_i)
                    .arg(&flip)
                    .launch(LaunchConfig {
                        grid_dim: (count.div_ceil(threads), 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        *slot = Some((group, flip, table));
    }
    let cached = slot.as_ref().context("convrot hadamard table")?;
    let entry = usize::from(args.fuse_silu);
    let (function, tile_rows, shared_mem_bytes) = if args.rotate_32 {
        (&fns.group_dot_32[entry], 32, group_dot_32_smem(&fns.arch))
    } else {
        (&fns.group_dot[entry], 16, group_dot_smem(group))
    };
    {
        let (table_ptr, _read) = cached.2.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(function)
                .arg(&args.x)
                .arg(&args.rotated)
                .arg(&table_ptr)
                .arg(&args.rows)
                .arg(&args.k)
                .arg(&args.group)
                .launch(LaunchConfig {
                    grid_dim: group_dot_grid(rows, k, group, tile_rows),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes,
                })?;
        }
    }
    Ok(())
}

/// `convrot_group_dot` then `convrot_quantize`. The product is a separate launch.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn launch_rotate_quant(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
) -> Result<()> {
    use crate::backend::gpu::{LaunchConfig, PushKernelArg};
    ensure!(
        args.group > 0 && args.group % 16 == 0 && args.k % args.group == 0,
        "convrot tensor-core shape"
    );
    let rows = args.rows as u32;
    let group = args.group as u32;
    let k = args.k as u32;
    ensure!(
        group <= fns.max_group,
        "convrot group {group} exceeds device tile {}",
        fns.max_group
    );
    launch_group_dot(stream, fns, args, rows, k, group)?;
    let tile_rows = quantize_rows(fns.optin, k)?;
    let tile = i32::try_from(tile_rows)?;
    let smem =
        u32::try_from(u64::from(tile_rows) * u64::from(k) * 4).context("convrot quantize tile")?;
    unsafe {
        stream
            .launch_builder(&fns.quantize)
            .arg(&args.rotated)
            .arg(&args.codes)
            .arg(&args.scales)
            .arg(&args.rows)
            .arg(&args.k)
            .arg(&tile)
            .launch(LaunchConfig {
                grid_dim: (rows.div_ceil(tile_rows), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: smem,
            })?;
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn launch_groups(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
) -> Result<()> {
    launch_rotate_quant(stream, fns, args)?;
    launch_product(stream, fns, args)
}

#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
mod device {
    use crate::backend::cuda::Device;
    use crate::backend::gpu::DevicePtr;
    use crate::convrot::{ConvrotArgs, ProductKernel, launch_device, load_convrot};
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    const M1: [i16; 2] = [16484, 16398];

    const M16: [i16; 32] = [
        16484, 16398, -16336, -16045, -16089, -15970, -16076, -16099, -16997, -16278, 16435, 16585,
        -16278, -16477, -16356, 16681, 16678, 16796, 16521, 16572, -16230, -16464, -16395, 16690,
        16435, -16199, -16213, -15994, -16089, -16161, -16219, -16128,
    ];

    #[cfg(feature = "cuda")]
    const M17: [i16; 34] = [
        16483, 16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529,
    ];

    #[cfg(feature = "cuda")]
    const M20: [i16; 40] = [
        16483, 16396, -16338, -16046, -16089, -15969, -16076, -16099, -16975, -16275, 16437, 16586,
        -16278, -16487, -16358, 16681, 16679, 16796, 16520, 16569, -16231, -16467, -16394, 16689,
        16437, -16199, -16213, -15994, -16088, -16157, -16221, -16131, 16469, 16529, -16495,
        -16508, -16299, 16675, 16585, 16782,
    ];

    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
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

    struct Ran {
        y: Vec<bf16>,
        rotated: Vec<f32>,
        codes: Vec<i8>,
        scales: Vec<f32>,
    }

    /// fmax of fabs, f32 round-to-nearest division, `rintf`, clamp [-127, 127].
    fn quantize_current(src: &[f32]) -> (Vec<i8>, f32) {
        let mut peak = 0.0f32;
        for &value in src {
            peak = peak.max(value.abs());
        }
        let scale = peak.max(1.0e-12f32) / 127.0f32;
        let inv = 1.0f32 / scale;
        let codes = src
            .iter()
            .map(|&value| {
                let q = (value * inv).round_ties_even().clamp(-127.0f32, 127.0f32);
                q as i8
            })
            .collect();
        (codes, scale)
    }

    fn code_scale_diff(rotated: &[f32], codes: &[i8], scales: &[f32], k: usize) -> (usize, usize) {
        let mut code_diff = 0usize;
        let mut scale_diff = 0usize;
        for (r, scale) in scales.iter().enumerate() {
            let (want, got_scale) = quantize_current(&rotated[r * k..(r + 1) * k]);
            let got = &codes[r * k..(r + 1) * k];
            code_diff += want.iter().zip(got).filter(|(a, b)| a != b).count();
            if got_scale.to_bits() != scale.to_bits() {
                scale_diff += 1;
            }
        }
        (code_diff, scale_diff)
    }

    fn check_codes(
        rotated: &[f32],
        codes: &[i8],
        scales: &[f32],
        rows: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(
            rotated.len() == rows * k && codes.len() == rows * k && scales.len() == rows,
            "convrot quantize download rows={rows} k={k}"
        );
        let (code_diff, scale_diff) = code_scale_diff(rotated, codes, scales, k);
        emit(&format!(
            "CONVROT_QUANT rows={rows} k={k} codes={code_diff} scales={scale_diff}"
        ));
        ensure!(
            code_diff == 0 && scale_diff == 0,
            "convrot quantize rows={rows} k={k} codes {code_diff} scales {scale_diff}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn rotated_mismatch(x: &[f32], got: &[f32], k: usize, group: usize) -> Result<usize> {
        ensure!(
            k > 0 && x.len() == got.len() && x.len().is_multiple_of(k),
            "convrot rotated shape"
        );
        let rows = x.len() / k;
        let h = hadamard_terms(group);
        let mut terms = Vec::new();
        let mut xr = Vec::new();
        let mut codes = vec![0i8; k];
        let mut scale = 0.0f32;
        let mut differ = 0usize;
        for r in 0..rows {
            mma_quantize_row(
                &x[r * k..(r + 1) * k],
                group,
                &h,
                &mut terms,
                &mut xr,
                &mut codes,
                &mut scale,
            );
            differ += xr
                .iter()
                .zip(&got[r * k..(r + 1) * k])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
        Ok(differ)
    }

    // 2x measured gfx1100 v_wmma_f32_16x16x16_bf16 vs exact f64 floor (activation / silu).
    #[cfg(feature = "rocm")]
    const ROCM_WMMA_ROT_MAX_ABS: f64 = 1.192_092_896e-7;
    #[cfg(feature = "rocm")]
    const ROCM_WMMA_ROT_RMSE: f64 = 2.132_190_794e-8;
    #[cfg(feature = "rocm")]
    const ROCM_WMMA_SILU_ROT_MAX_ABS: f64 = 2.086_162_568e-6;
    #[cfg(feature = "rocm")]
    const ROCM_WMMA_SILU_ROT_RMSE: f64 = 3.529_518_472e-7;

    #[cfg(feature = "rocm")]
    fn f64_rotated_stats(
        x: &[f32],
        got: &[f32],
        k: usize,
        group: usize,
        row_period: Option<usize>,
    ) -> Result<(f64, f64)> {
        ensure!(
            k > 0
                && group >= 16
                && k.is_multiple_of(group)
                && x.len() == got.len()
                && x.len().is_multiple_of(k),
            "convrot f64 rotated shape"
        );
        let rows = x.len() / k;
        let unique = row_period.map_or(rows, |period| rows.min(period));
        let mut table = vec![0.0f64; group * group];
        for col in 0..group {
            for r in 0..group {
                table[col * group + r] =
                    f64::from(bf16::from_f32(super::hadamard(group, r, col)).to_f32());
            }
        }
        let mut reference = vec![0.0f64; unique * k];
        let groups = k / group;
        let threads = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 8)
            .min(unique.max(1));
        let rows_per = unique.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut row0 = 0usize;
            for part in reference.chunks_mut(rows_per * k) {
                let rows_here = part.len() / k;
                let start = row0;
                row0 += rows_here;
                let table = table.as_slice();
                scope.spawn(move || {
                    let mut x_bf16 = vec![0.0f64; k];
                    for r in 0..rows_here {
                        let src = &x[(start + r) * k..(start + r + 1) * k];
                        for (dst, &v) in x_bf16.iter_mut().zip(src) {
                            *dst = f64::from(bf16::from_f32(v).to_f32());
                        }
                        let out = &mut part[r * k..(r + 1) * k];
                        for g in 0..groups {
                            let base = g * group;
                            for col in 0..group {
                                let mut sum = 0.0f64;
                                let hcol = &table[col * group..(col + 1) * group];
                                for i in 0..group {
                                    sum += x_bf16[base + i] * hcol[i];
                                }
                                out[base + col] = sum;
                            }
                        }
                    }
                });
            }
        });
        let mut max_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for r in 0..rows {
            let ref_row = &reference[(r % unique) * k..(r % unique + 1) * k];
            let got_row = &got[r * k..(r + 1) * k];
            for (&want, &actual) in ref_row.iter().zip(got_row) {
                ensure!(actual.is_finite(), "non-finite convrot rotated value");
                let err = (f64::from(actual) - want).abs();
                if err > max_abs {
                    max_abs = err;
                }
                sum_sq += err * err;
            }
        }
        let rmse = (sum_sq / (rows * k) as f64).sqrt();
        Ok((max_abs, rmse))
    }

    #[cfg(feature = "rocm")]
    fn check_rocm_rotation_and_product(
        input: &HostLaunch<'_>,
        ran: &Ran,
        row_period: Option<usize>,
    ) -> Result<()> {
        let HostLaunch {
            x,
            k,
            group,
            qweight,
            n,
            wscale,
            ..
        } = *input;
        let rows = x.len() / k;
        let (max_abs, rmse) = f64_rotated_stats(x, &ran.rotated, k, group, row_period)?;
        emit(&format!(
            "CONVROT_ROCM_ROTATED rows={rows} k={k} group={group} max_abs={max_abs:.9e} rmse={rmse:.9e}"
        ));
        ensure!(
            max_abs <= ROCM_WMMA_ROT_MAX_ABS && rmse <= ROCM_WMMA_ROT_RMSE,
            "convrot rocm rotated rows={rows} k={k} group={group} max_abs={max_abs:.9e} rmse={rmse:.9e}"
        );
        check_codes(&ran.rotated, &ran.codes, &ran.scales, rows, k)?;
        let host = host_product(&ran.codes, &ran.scales, k, qweight, n, wscale)?;
        let mismatched = bit_diff(&ran.y, &host);
        emit(&format!(
            "CONVROT_MMA rows={rows} k={k} n={n} group={group} differ={mismatched}"
        ));
        ensure!(
            mismatched == 0,
            "convrot rocm product rows={rows} k={k} n={n} group={group} differing bits {mismatched}"
        );
        Ok(())
    }

    fn tile_row_offset_diff(rotated: &[f32], codes: &[i8], k: usize) -> usize {
        let rows = codes.len() / k;
        let mut differ = 0usize;
        for r in 0..rows {
            let src = (r + 1) % rows;
            let (want, _) = quantize_current(&rotated[src * k..(src + 1) * k]);
            let got = &codes[r * k..(r + 1) * k];
            differ += want.iter().zip(got).filter(|(a, b)| a != b).count();
        }
        differ
    }

    struct HostLaunch<'a> {
        x: &'a [f32],
        k: usize,
        group: usize,
        qweight: &'a [i8],
        n: usize,
        wscale: &'a [f32],
        keep: bool,
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
        Ok(launch_kept(
            device,
            HostLaunch {
                x,
                k,
                group,
                qweight,
                n,
                wscale,
                keep: false,
            },
        )?
        .y)
    }

    struct HeldActivation {
        ptr: u64,
        host_bf16: Option<Vec<bf16>>,
        dev_f32: Option<crate::backend::gpu::CudaSlice<f32>>,
        dev_bf16: Option<crate::backend::gpu::CudaSlice<bf16>>,
    }

    impl HeldActivation {
        fn release(self) {
            let HeldActivation {
                host_bf16,
                dev_f32,
                dev_bf16,
                ptr: _,
            } = self;
            drop(host_bf16);
            drop(dev_f32);
            drop(dev_bf16);
        }
    }

    // At TENSOR_CORE_ROWS and above, the device bytes are bf16::from_f32 of x.
    fn upload_activation(
        stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
        x: &[f32],
        rows: usize,
    ) -> Result<HeldActivation> {
        if rows >= super::TENSOR_CORE_ROWS {
            let values: Vec<bf16> = x.iter().copied().map(bf16::from_f32).collect();
            let slice = stream.clone_htod(&values)?;
            let ptr = slice.device_ptr(stream).0;
            Ok(HeldActivation {
                ptr,
                host_bf16: Some(values),
                dev_f32: None,
                dev_bf16: Some(slice),
            })
        } else {
            let slice = stream.clone_htod(x)?;
            let ptr = slice.device_ptr(stream).0;
            Ok(HeldActivation {
                ptr,
                host_bf16: None,
                dev_f32: Some(slice),
                dev_bf16: None,
            })
        }
    }

    fn launch_kept(device: &Device, input: HostLaunch<'_>) -> Result<Ran> {
        launch_kept_tile(device, input, false)
    }

    fn launch_kept_tile(device: &Device, input: HostLaunch<'_>, rotate_32: bool) -> Result<Ran> {
        let HostLaunch {
            x,
            k,
            group,
            qweight,
            n,
            wscale,
            keep,
        } = input;
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
        let held = upload_activation(stream, x, rows)?;
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
                x: held.ptr,
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
                product: ProductKernel::NFast,
                fuse_silu: false,
                rotate_32,
            },
        )?;
        stream.synchronize()?;
        held.release();
        let y = stream.clone_dtoh(&y_d)?;
        let (rotated, codes, scales) = if keep {
            (
                stream.clone_dtoh(&rotated)?,
                stream.clone_dtoh(&codes)?,
                stream.clone_dtoh(&scales)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        Ok(Ran {
            y,
            rotated,
            codes,
            scales,
        })
    }

    #[cfg(feature = "cuda")]
    fn exp2i(exp: i32) -> f64 {
        if (-1022..=1023).contains(&exp) {
            f64::from_bits(((exp + 1023) as u64) << 52)
        } else if (-1074..-1022).contains(&exp) {
            f64::from_bits(1u64 << (exp + 1074) as u32)
        } else if exp < -1074 {
            0.0
        } else {
            f64::INFINITY
        }
    }

    #[cfg(feature = "cuda")]
    fn frexp_half(x: f64) -> (f64, i32) {
        let bits = x.to_bits();
        let exp_field = ((bits >> 52) & 0x7ff) as i32;
        if exp_field == 0 {
            if bits & 0x000f_ffff_ffff_ffff == 0 {
                return (0.0, 0);
            }
            let (mag, exp) = frexp_half(x * exp2i(54));
            return (mag, exp - 54);
        }
        let mag = (bits & 0x000f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000;
        (f64::from_bits(mag).copysign(x), exp_field - 1022)
    }

    #[cfg(feature = "cuda")]
    fn sig_exp(x: f64) -> (f64, i32) {
        if x.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {
            return (0.0, -126);
        }
        let (mag, exp) = frexp_half(x);
        let mut significand = mag * 2.0;
        let mut exponent = exp - 1;
        if exponent < -126 {
            significand *= exp2i(exponent + 126);
            exponent = -126;
        }
        if significand.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {
            exponent = -126;
        }
        (significand, exponent)
    }

    #[cfg(feature = "cuda")]
    fn rz_f32(sum: f64, e_max: i32) -> f32 {
        let (mut significand, exp) = sig_exp(sum * exp2i(e_max));
        significand = (significand * exp2i(23)).trunc() * exp2i(-23);
        let out = (significand * exp2i(exp)) as f32;
        if out.to_bits() & 0x7fff_ffff == 0 {
            0.0
        } else {
            out
        }
    }

    // m16n8k16 f32 accumulator: T-FDPA length 8, F=24, e_zero=-132, round toward zero to f32.
    // k = 0..7, then k = 8..15, each consuming the running accumulator.
    #[cfg(feature = "cuda")]
    fn t_fdpa8(a: &[(f64, i32)], b: &[(f64, i32)], c: f32) -> f32 {
        const F: i32 = 24;
        const E_ZERO: i32 = -132;
        let mut s = [0.0f64; 9];
        let mut e = [0i32; 9];
        for i in 0..8 {
            let prod = a[i].0 * b[i].0;
            s[i] = prod;
            e[i] = if prod.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {
                E_ZERO
            } else {
                a[i].1 + b[i].1
            };
        }
        let (sc, ec) = sig_exp(f64::from(c));
        s[8] = sc;
        e[8] = if sc.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {
            E_ZERO
        } else {
            ec
        };
        let mut e_max = e[0];
        for &value in &e[1..] {
            e_max = e_max.max(value);
        }
        let mut sum = 0.0f64;
        for i in 0..9 {
            sum += (s[i] * exp2i(e[i] - e_max + F)).trunc() * exp2i(-F);
        }
        rz_f32(sum, e_max)
    }

    #[cfg(feature = "cuda")]
    fn mma_k16(a: &[f32; 16], b: &[f32; 16], c: f32) -> f32 {
        let mut at = [(0.0f64, 0i32); 16];
        let mut bt = [(0.0f64, 0i32); 16];
        for i in 0..16 {
            at[i] = sig_exp(f64::from(a[i]));
            bt[i] = sig_exp(f64::from(b[i]));
        }
        let mid = t_fdpa8(&at[..8], &bt[..8], c);
        t_fdpa8(&at[8..], &bt[8..], mid)
    }

    #[cfg(feature = "cuda")]
    fn mma_k16_points() -> (f32, f32, f32) {
        let mut a = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        a[0] = -8192.0;
        a[1] = -0.5;
        a[2] = -0.25;
        a[3] = -0.125;
        b[0] = 1024.0;
        b[1] = 1.0;
        b[2] = 1.0;
        b[3] = 1.0;
        let paper = mma_k16(&a, &b, 8_388_608.0);
        a = [0.0; 16];
        b = [0.0; 16];
        a[0] = 1.0;
        b[0] = 1.0;
        let one = mma_k16(&a, &b, 0.0);
        a[1] = 1.0;
        b[1] = 1.0;
        let two = mma_k16(&a, &b, 0.0);
        (paper, one, two)
    }

    #[cfg(feature = "cuda")]
    fn hadamard_terms(group: usize) -> Vec<(f64, i32)> {
        let mut terms = vec![(0.0f64, 0i32); group * group];
        for col in 0..group {
            for k_index in 0..group {
                let value = bf16::from_f32(super::hadamard(group, k_index, col)).to_f32();
                terms[col * group + k_index] = sig_exp(f64::from(value));
            }
        }
        terms
    }

    #[cfg(feature = "cuda")]
    fn mma_quantize_row(
        src: &[f32],
        group: usize,
        h: &[(f64, i32)],
        terms: &mut Vec<(f64, i32)>,
        xr: &mut Vec<f32>,
        codes: &mut [i8],
        scale_out: &mut f32,
    ) {
        let k = src.len();
        terms.clear();
        xr.clear();
        terms.resize(k, (0.0, 0));
        xr.resize(k, 0.0);
        for (i, &value) in src.iter().enumerate() {
            let rounded = bf16::from_f32(value).to_f32();
            terms[i] = sig_exp(f64::from(rounded));
        }
        let mut peak = 0.0f32;
        let groups = k / group;
        for g in 0..groups {
            let base = g * group;
            for col in 0..group {
                let mut acc = 0.0f32;
                let b_col = &h[col * group..(col + 1) * group];
                let mut k0 = 0;
                while k0 < group {
                    acc = t_fdpa8(&terms[base + k0..base + k0 + 8], &b_col[k0..k0 + 8], acc);
                    acc = t_fdpa8(
                        &terms[base + k0 + 8..base + k0 + 16],
                        &b_col[k0 + 8..k0 + 16],
                        acc,
                    );
                    k0 += 16;
                }
                xr[base + col] = acc;
                peak = peak.max(acc.abs());
            }
        }
        let scale = peak.max(1e-12) / 127.0;
        *scale_out = scale;
        let inv = 1.0 / scale;
        for i in 0..k {
            let quantized = (xr[i] * inv).round_ties_even().clamp(-127.0, 127.0);
            codes[i] = quantized as i8;
        }
    }

    #[cfg(feature = "cuda")]
    fn mma_rotate_quant(x: &[f32], k: usize, group: usize) -> Result<(Vec<i8>, Vec<f32>)> {
        ensure!(
            k > 0
                && group >= 16
                && group.is_multiple_of(16)
                && k.is_multiple_of(group)
                && x.len().is_multiple_of(k),
            "convrot mma shape"
        );
        let rows = x.len() / k;
        let mut codes = vec![0i8; rows * k];
        let mut scales = vec![0.0f32; rows];
        if rows == 0 {
            return Ok((codes, scales));
        }
        let h = hadamard_terms(group);
        let threads = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 8)
            .min(rows);
        let rows_per = rows.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut row0 = 0usize;
            for (code_rows, scale_rows) in codes
                .chunks_mut(rows_per * k)
                .zip(scales.chunks_mut(rows_per))
            {
                let rows_here = scale_rows.len();
                let start = row0;
                row0 += rows_here;
                let x_rows = &x[start * k..(start + rows_here) * k];
                let h = h.as_slice();
                scope.spawn(move || {
                    let mut terms = Vec::new();
                    let mut xr = Vec::new();
                    for r in 0..rows_here {
                        mma_quantize_row(
                            &x_rows[r * k..(r + 1) * k],
                            group,
                            h,
                            &mut terms,
                            &mut xr,
                            &mut code_rows[r * k..(r + 1) * k],
                            &mut scale_rows[r],
                        );
                    }
                });
            }
        });
        Ok((codes, scales))
    }

    fn emit(line: &str) {
        println!("{line}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    fn check_hadamard_table_mutant(device: &Device) -> Result<()> {
        let rows = 17usize;
        let k = 256usize;
        let group = 256usize;
        let n = 2usize;
        let x = activation(rows, k);
        let qweight = weight(n, k);
        let wscale = wscale_row(n);
        let input = || HostLaunch {
            x: &x,
            k,
            group,
            qweight: &qweight,
            n,
            wscale: &wscale,
            keep: true,
        };
        let base = launch_kept(device, input())?;
        let flipped = super::with_hadamard_flip(0, || launch_kept(device, input()))?;
        let differ = count_f32(&flipped.rotated, &base.rotated);
        emit(&format!("CONVROT_TABLE_MUTANT differ={differ}"));
        ensure!(
            differ > 0,
            "one hadamard table entry matched the unflipped tile"
        );
        Ok(())
    }

    fn check_group_dot_tiles(device: &Device) -> Result<()> {
        check_hadamard_table_mutant(device)?;
        let group = 256usize;
        let n = 40usize;
        for rows in [17usize, 20, 42, 41770] {
            for k in [5376usize, 7168, 14336] {
                let (gx, gy, gz) = super::group_dot_grid(
                    u32::try_from(rows)?,
                    u32::try_from(k)?,
                    u32::try_from(group)?,
                    16,
                );
                ensure!(
                    gx == u32::try_from(rows)?.div_ceil(16)
                        && gy == 1
                        && gz == u32::try_from(k)? / u32::try_from(group)?
                        && gy != u32::try_from(group)? / 8,
                    "group_dot grid rows={rows} k={k} got {gx},{gy},{gz}"
                );
                emit(&format!("CONVROT_MMA rows={rows} k={k} n={n} begin"));
                let x = activation(rows, k);
                let qweight = weight(n, k);
                let wscale = wscale_row(n);
                #[cfg(feature = "cuda")]
                let (host_codes, host_scales) = mma_rotate_quant(&x, k, group)?;
                #[cfg(feature = "cuda")]
                let host = host_product(&host_codes, &host_scales, k, &qweight, n, &wscale)?;
                let quant = cfg!(feature = "rocm") || rows == 17 || rows == 20 || rows == 42;
                let launch_input = HostLaunch {
                    x: &x,
                    k,
                    group,
                    qweight: &qweight,
                    n,
                    wscale: &wscale,
                    keep: quant,
                };
                let ran = launch_kept(device, launch_input)?;
                #[cfg(feature = "cuda")]
                {
                    let mismatched = bit_diff(&ran.y, &host);
                    emit(&format!(
                        "CONVROT_MMA rows={rows} k={k} n={n} differ={mismatched}"
                    ));
                    ensure!(
                        mismatched == 0,
                        "convrot mma rows={rows} k={k} n={n} differing bits {mismatched}"
                    );
                    if quant {
                        check_codes(&ran.rotated, &ran.codes, &ran.scales, rows, k)?;
                        let rotated_diff = rotated_mismatch(&x, &ran.rotated, k, group)?;
                        emit(&format!(
                            "CONVROT_ROTATED rows={rows} k={k} n={n} differ={rotated_diff}"
                        ));
                        ensure!(
                            rotated_diff == 0,
                            "convrot rotated rows={rows} k={k} differing bits {rotated_diff}"
                        );
                        let host_code_diff = count_i8(&host_codes, &ran.codes);
                        let host_scale_diff = count_f32(&host_scales, &ran.scales);
                        emit(&format!(
                            "CONVROT_HOST rows={rows} k={k} codes={host_code_diff} scales={host_scale_diff}"
                        ));
                        ensure!(
                            host_code_diff == 0 && host_scale_diff == 0,
                            "convrot host rows={rows} k={k} codes {host_code_diff} scales {host_scale_diff}"
                        );
                    }
                }
                #[cfg(feature = "rocm")]
                check_rocm_rotation_and_product(
                    &HostLaunch {
                        x: &x,
                        k,
                        group,
                        qweight: &qweight,
                        n,
                        wscale: &wscale,
                        keep: true,
                    },
                    &ran,
                    Some(23),
                )?;
                if rows == 20 && k == 5376 {
                    let (seq_codes, seq_scales) = super::rotate_quant(&x, k, group)?;
                    let sequential =
                        super::int8_product(&seq_codes, &seq_scales, k, &qweight, n, &wscale)?;
                    let seq_diff = bit_diff(&ran.y, &sequential);
                    emit(&format!(
                        "CONVROT_SEQUENTIAL rows=20 k=5376 n=40 differ={seq_diff}"
                    ));
                    ensure!(seq_diff > 0, "sequential quantize matched the device");
                    let tile_diff = tile_row_offset_diff(&ran.rotated, &ran.codes, k);
                    emit(&format!(
                        "CONVROT_TILE_ROW rows=20 k=5376 differ={tile_diff}"
                    ));
                    ensure!(tile_diff > 0, "tile row offset reproduced the device codes");
                    let mut shifted = vec![0.0f32; rows * k];
                    for r in 0..rows {
                        for c in 0..k {
                            shifted[r * k + c] = x[r * k + (c + 1) % k];
                        }
                    }
                    #[cfg(feature = "cuda")]
                    let shift_codes = mma_rotate_quant(&shifted, k, group)?.0;
                    #[cfg(feature = "rocm")]
                    let shift_codes = launch_kept(
                        device,
                        HostLaunch {
                            x: &shifted,
                            k,
                            group,
                            qweight: &qweight,
                            n,
                            wscale: &wscale,
                            keep: true,
                        },
                    )?
                    .codes;
                    let col_diff = count_i8(&shift_codes, &ran.codes);
                    emit(&format!(
                        "CONVROT_BF16_COL rows=20 k=5376 differ={col_diff}"
                    ));
                    ensure!(
                        col_diff > 0,
                        "bf16 column offset reproduced the device codes"
                    );
                }
                if rows == 41770 && k == 5376 {
                    let mut swapped = x;
                    swapped.swap(0, 16);
                    let mutant = launch(device, &swapped, k, group, &qweight, n, &wscale)?;
                    let mut_diff = bit_diff(&mutant, &ran.y);
                    emit(&format!("CONVROT_GROUP_DOT_MUTANT differ={mut_diff}"));
                    ensure!(mut_diff > 0, "column swap reproduced the device product");
                }
            }
        }
        let wide = 1024usize;
        let rows = 20usize;
        let k = 7168usize;
        let max = super::max_group(device.info.optin_shared_bytes)?;
        ensure!(
            u32::try_from(wide)? <= max,
            "device tile {max} excludes group {wide}"
        );
        emit(&format!(
            "CONVROT_MAX_GROUP optin={} max={max}",
            device.info.optin_shared_bytes
        ));
        let (gx, gy, gz) = super::group_dot_grid(
            u32::try_from(rows)?,
            u32::try_from(k)?,
            u32::try_from(wide)?,
            16,
        );
        ensure!(
            gx == u32::try_from(rows)?.div_ceil(16)
                && gy == 1
                && gz == u32::try_from(k)? / u32::try_from(wide)?
                && gy != u32::try_from(wide)? / 8,
            "group_dot grid rows={rows} k={k} group={wide} got {gx},{gy},{gz}"
        );
        emit(&format!(
            "CONVROT_MMA rows={rows} k={k} n={n} group={wide} begin"
        ));
        let x = activation(rows, k);
        let qweight = weight(n, k);
        let wscale = wscale_row(n);
        #[cfg(feature = "cuda")]
        {
            let host = {
                let (codes, scales) = mma_rotate_quant(&x, k, wide)?;
                host_product(&codes, &scales, k, &qweight, n, &wscale)?
            };
            let y = launch(device, &x, k, wide, &qweight, n, &wscale)?;
            let mismatched = bit_diff(&y, &host);
            emit(&format!(
                "CONVROT_MMA rows={rows} k={k} n={n} group={wide} differ={mismatched}"
            ));
            ensure!(
                mismatched == 0,
                "convrot mma rows={rows} k={k} n={n} group={wide} differing bits {mismatched}"
            );
        }
        #[cfg(feature = "rocm")]
        {
            let ran = launch_kept(
                device,
                HostLaunch {
                    x: &x,
                    k,
                    group: wide,
                    qweight: &qweight,
                    n,
                    wscale: &wscale,
                    keep: true,
                },
            )?;
            check_rocm_rotation_and_product(
                &HostLaunch {
                    x: &x,
                    k,
                    group: wide,
                    qweight: &qweight,
                    n,
                    wscale: &wscale,
                    keep: true,
                },
                &ran,
                Some(23),
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_kernel_matches_the_definition() -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let (paper, one, two) = mma_k16_points();
            ensure!(
                paper.to_bits() == (-0.5f32).to_bits()
                    && one.to_bits() == 1.0f32.to_bits()
                    && two.to_bits() == 2.0f32.to_bits(),
                "mma k16 self-check paper={paper} one={one} two={two}"
            );
        }
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
            let x = activation(rows, k);
            let qweight = weight(n, k);
            let wscale = wscale_row(n);
            #[cfg(feature = "cuda")]
            {
                let expected = definition_bits(rows, k, n)?;
                let quant = rows == 17 && k != 256;
                let ran = launch_kept(
                    &device,
                    HostLaunch {
                        x: &x,
                        k,
                        group,
                        qweight: &qweight,
                        n,
                        wscale: &wscale,
                        keep: quant,
                    },
                )?;
                let mismatched = differ(&ran.y, &expected);
                ensure!(
                    mismatched == 0,
                    "convrot kernel M={rows} K={k} N={n} differing bits {mismatched}"
                );
                if quant {
                    check_codes(&ran.rotated, &ran.codes, &ran.scales, rows, k)?;
                    let rotated_diff = rotated_mismatch(&x, &ran.rotated, k, group)?;
                    emit(&format!(
                        "CONVROT_ROTATED rows={rows} k={k} n={n} differ={rotated_diff}"
                    ));
                    ensure!(
                        rotated_diff == 0,
                        "convrot rotated rows={rows} k={k} differing bits {rotated_diff}"
                    );
                }
            }
            #[cfg(feature = "rocm")]
            {
                let keep = rows >= super::TENSOR_CORE_ROWS;
                let ran = launch_kept(
                    &device,
                    HostLaunch {
                        x: &x,
                        k,
                        group,
                        qweight: &qweight,
                        n,
                        wscale: &wscale,
                        keep,
                    },
                )?;
                if rows == 1 {
                    let mismatched = differ(&ran.y, &M1);
                    ensure!(mismatched == 0, "convrot M=1 differing bits {mismatched}");
                } else if rows == 16 {
                    let mismatched = differ(&ran.y, &M16);
                    ensure!(mismatched == 0, "convrot M=16 differing bits {mismatched}");
                } else {
                    check_rocm_rotation_and_product(
                        &HostLaunch {
                            x: &x,
                            k,
                            group,
                            qweight: &qweight,
                            n,
                            wscale: &wscale,
                            keep: true,
                        },
                        &ran,
                        Some(23),
                    )?;
                }
            }
        }
        check_group_dot_tiles(&device)?;
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
        #[cfg(feature = "cuda")]
        ensure!(
            differ(&scale_off, &definition_bits(rows, k, n)?) > 0,
            "a doubled wscale reproduced the definition bits"
        );
        #[cfg(feature = "rocm")]
        ensure!(
            bit_diff(&scale_off, &y) > 0,
            "a doubled wscale reproduced the device output"
        );
        let omitted = super::linear_omitting_rotation(&x, k, &qweight, n, &wscale)?;
        #[cfg(feature = "cuda")]
        ensure!(
            differ(&omitted, &definition_bits(rows, k, n)?) > 0
                && differ(
                    &omitted,
                    &y.iter().map(|v| v.to_bits() as i16).collect::<Vec<_>>()
                ) > 0,
            "omitting the rotation reproduced the definition bits"
        );
        #[cfg(feature = "rocm")]
        ensure!(
            bit_diff(&omitted, &y) > 0,
            "omitting the rotation reproduced the device output"
        );
        Ok(())
    }

    const PRODUCT_FIXTURE_PERIOD: usize = 251;

    fn pattern_i8(a: usize, b: usize, p: usize, q: usize) -> i8 {
        (((a * p + b * q) % PRODUCT_FIXTURE_PERIOD) as i32 - 125) as i8
    }

    fn fixture(rows: usize, k: usize, n: usize) -> (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>) {
        let mut codes = vec![0i8; rows * k];
        for row in 0..rows {
            for i in 0..k {
                codes[row * k + i] = pattern_i8(row, i, 17, 3);
            }
        }
        let mut qweight = vec![0i8; n * k];
        for col in 0..n {
            for i in 0..k {
                qweight[col * k + i] = pattern_i8(col, i, 23, 13);
            }
        }
        let scales = (0..rows)
            .map(|row| ((row * 5) % 7 + 1) as f32 * 1.0e-3)
            .collect();
        let wscale = (0..n).map(|col| (col % 5 + 1) as f32 * 1.0e-2).collect();
        (codes, scales, qweight, wscale)
    }

    fn bit_diff(got: &[bf16], expected: &[bf16]) -> usize {
        got.iter()
            .zip(expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
            + got.len().abs_diff(expected.len())
    }

    fn host_product(
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
        let mut y = vec![bf16::ZERO; rows * n];
        let threads = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 8)
            .min(rows);
        let rows_per = rows.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut row0 = 0usize;
            for part in y.chunks_mut(rows_per * n) {
                let rows_here = part.len() / n;
                let start = row0;
                row0 += rows_here;
                scope.spawn(move || {
                    for r in 0..rows_here {
                        let row = start + r;
                        super::write_product_row(
                            &codes[row * k..(row + 1) * k],
                            qweight,
                            k,
                            scales[row],
                            wscale,
                            &mut part[r * n..(r + 1) * n],
                        );
                    }
                });
            }
        });
        Ok(y)
    }

    fn launch_product_at(
        device: &Device,
        codes: &[i8],
        scales: &[f32],
        qweight: &[i8],
        wscale: &[f32],
        k: usize,
        window: (u64, u64, ProductKernel),
    ) -> Result<Vec<bf16>> {
        let (weight_off, scale_off, product) = window;
        let n = wscale.len();
        let rows = scales.len();
        ensure!(
            rows > 0
                && k > 0
                && n > 0
                && codes.len() == rows * k
                && qweight.len() == n * k + weight_off as usize,
            "convrot product shape"
        );
        let off = usize::try_from(weight_off)?;
        let scale_extra = usize::try_from(scale_off)?;
        let stream = &device.stream;
        let codes_d = stream.clone_htod(codes)?;
        let scales_d = if scale_extra == 0 {
            stream.clone_htod(scales)?
        } else {
            let mut scale_buf = Vec::with_capacity(rows + scale_extra);
            scale_buf.extend_from_slice(scales);
            for i in 0..scale_extra {
                let row = rows + i;
                scale_buf.push(((row * 5) % 7 + 1) as f32 * 1.0e-3);
            }
            stream.clone_htod(&scale_buf)?
        };
        let weight_d = stream.clone_htod(&qweight[off..off + n * k])?;
        let wscale_d = stream.clone_htod(wscale)?;
        let y_d = stream.alloc_zeros::<bf16>(rows * n)?;
        let fns = load_convrot(&device.ctx)?;
        super::launch_product(
            stream,
            &fns,
            &ConvrotArgs {
                x: 0,
                qweight: weight_d.device_ptr(stream).0,
                wscale: wscale_d.device_ptr(stream).0,
                y: y_d.device_ptr(stream).0,
                rotated: 0,
                codes: codes_d.device_ptr(stream).0,
                scales: scales_d.device_ptr(stream).0 + scale_off * 4,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: i32::try_from(n)?,
                group: 256,
                product,
                fuse_silu: false,
                rotate_32: false,
            },
        )?;
        stream.synchronize()?;
        stream.clone_dtoh(&y_d).context("convrot product download")
    }

    fn fixture_product_oracle<'a>(
        codes: &[i8],
        scales: &'a [f32],
        qweight: &[i8],
        wscale: &'a [f32],
        k: usize,
    ) -> impl Fn(&[bf16]) -> usize + use<'a> {
        let n = wscale.len();
        let period = PRODUCT_FIXTURE_PERIOD;
        // Codes and weight rows have period 251; scales are indexed by the original coordinates.
        let mut accumulators = Vec::with_capacity(period * period);
        for x in codes.chunks_exact(k).take(period) {
            for w in qweight.chunks_exact(k).take(period) {
                accumulators.push(
                    x.iter()
                        .zip(w)
                        .map(|(&a, &b)| i32::from(a) * i32::from(b))
                        .sum::<i32>(),
                );
            }
        }
        move |output: &[bf16]| {
            output
                .chunks_exact(n)
                .enumerate()
                .map(|(row, values)| {
                    let accumulators = &accumulators[row % period * period..][..period];
                    values
                        .iter()
                        .enumerate()
                        .filter(|&(col, value)| {
                            let acc = accumulators[col % period] as f32;
                            let expected = bf16::from_f32(acc * scales[row] * wscale[col]);
                            value.to_bits() != expected.to_bits()
                        })
                        .count()
                })
                .sum::<usize>()
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "sm120 GPU window; validates the prime, drain and first shared-slot reuse"]
    fn convrot_gate_up_one_barrier_panels() -> Result<()> {
        let device = Device::new(0)?;
        ensure!(device.info.arch == crate::backend::setup::DeviceArchitecture::Cuda(120));
        for rows in [127usize, 128, 129] {
            for k in [64, 128, 192] {
                let n = 257;
                let (mut codes, scales, mut weights, wscale) = fixture(rows, k, n);
                let diff = fixture_product_oracle(&codes, &scales, &weights, &wscale, k);
                let stripes = u32::try_from(rows.div_ceil(128))?;
                let order = |one_barrier| ProductKernel::MFastStriped {
                    tiles: 1,
                    stripes,
                    one_barrier,
                };
                for flag in [false, true] {
                    let output = launch_product_at(
                        &device,
                        &codes,
                        &scales,
                        &weights,
                        &wscale,
                        k,
                        (0, 0, order(flag)),
                    )?;
                    ensure!(
                        output.len() == rows * n && diff(&output) == 0,
                        "panel oracle rows={rows} k={k} one_barrier={flag}"
                    );
                }
                if k == 192 {
                    for row in codes.chunks_exact_mut(k).chain(weights.chunks_exact_mut(k)) {
                        row.copy_within(0..64, 128);
                    }
                    let stale = launch_product_at(
                        &device,
                        &codes,
                        &scales,
                        &weights,
                        &wscale,
                        k,
                        (0, 0, order(true)),
                    )?;
                    ensure!(diff(&stale) > 0, "stale third panel matched at {rows} rows");
                }
            }
        }
        Ok(())
    }

    fn check_gate_up_product(device: &Device) -> Result<()> {
        let (rows, k, n) = (1509, 5376, 28672);
        let active = super::product_active_ctas(device, ProductKernel::MFastLdsm)?;
        let expected_order = super::gate_up_products(&device.info, active, [rows, 1501], (k, n));
        ensure!(
            super::gate_up_product_order(device, [rows, 1501], (k, n))? == expected_order,
            "gate/up M-fast selection mismatch on {}",
            device.info.arch
        );
        ensure!(
            super::gate_up_product_order(device, [39_740, 42], (k, n))?[1] == ProductKernel::NFast,
            "gate/up M-fast enabled below 2 M tiles"
        );
        let (codes, scales, qweight, wscale) = fixture(rows, k, n);
        let diff = fixture_product_oracle(&codes, &scales, &qweight, &wscale, k);
        for (rows, m_fast) in [(rows, false), (rows, true), (1501, false), (1501, true)] {
            let output = launch_product_at(
                device,
                &codes[..rows * k],
                &scales[..rows],
                &qweight,
                &wscale,
                k,
                (
                    0,
                    0,
                    if m_fast {
                        ProductKernel::MFastLdsm
                    } else {
                        ProductKernel::NFast
                    },
                ),
            )?;
            ensure!(output.len() == rows * n, "convrot gate/up output extent");
            let differing = diff(&output);
            println!(
                "CONVROT_PRODUCT_GATE_UP rows={rows} k={k} n={n} m_fast={m_fast} differ={differing}"
            );
            ensure!(
                differing == 0,
                "convrot gate/up rows={rows} differs in {differing} elements"
            );
        }
        let mut shifted = wscale.clone();
        shifted.rotate_left(1);
        let output = launch_product_at(
            device,
            &codes,
            &scales,
            &qweight,
            &shifted,
            k,
            (0, 0, ProductKernel::MFastLdsm),
        )?;
        ensure!(output.len() == rows * n, "convrot gate/up control extent");
        let differing = diff(&output);
        println!("CONVROT_PRODUCT_GATE_UP_SCALE_COLUMN_SHIFT differ={differing}");
        ensure!(
            differing > 0,
            "shifted gate/up scales matched unshifted output"
        );
        Ok(())
    }

    fn check_product_tail(
        device: &Device,
        (rows, k, n): (usize, usize, usize),
        baseline: ProductKernel,
        product: ProductKernel,
        salt: usize,
    ) -> Result<(usize, usize)> {
        let (mut codes, mut scales, mut weights, wscale) = fixture(rows, k, n);
        weights.rotate_left(k * salt);
        let run = |codes: &[i8], scales: &[f32], order| {
            launch_product_at(device, codes, scales, &weights, &wscale, k, (0, 0, order))
        };
        let reference = run(&codes, &scales, baseline)?;
        let diff = bit_diff(&run(&codes, &scales, product)?, &reference);
        ensure!(
            diff == 0,
            "{product:?} k={k} n={n} differs in {diff} elements"
        );

        let tile = product.tile_dims(&device.info.arch).0 as usize;
        let start = rows / tile * tile;
        let wrong = start - tile;
        codes.copy_within(wrong * k..(wrong + rows - start) * k, start * k);
        scales.copy_within(wrong..wrong + rows - start, start);
        let shifted = run(&codes, &scales, product)?;
        ensure!(bit_diff(&shifted[..start * n], &reference[..start * n]) == 0);
        let moved_rows = shifted[start * n..]
            .chunks_exact(n)
            .zip(reference[start * n..].chunks_exact(n))
            .filter(|(a, b)| a != b)
            .count();
        ensure!(
            moved_rows == rows - start,
            "wrong tail changed {moved_rows} rows"
        );
        ensure!(shifted[start * n..] == reference[wrong * n..(wrong + rows - start) * n]);
        Ok((diff, moved_rows))
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a reserved sm120 GPU window"]
    fn convrot_n_fast_swizzle_matches_stored_products() -> Result<()> {
        let device = Device::new(0)?;
        let rows = 37_717;
        for (role, k, n, salt, baseline) in [
            ("q", 5376, 7168, 0, ProductKernel::NFastLdsm),
            ("k", 5376, 7168, 1, ProductKernel::NFastLdsm),
            ("v", 5376, 7168, 2, ProductKernel::NFastLdsm),
            ("out", 7168, 5376, 0, ProductKernel::NFast),
            ("down", 14336, 5376, 0, ProductKernel::NFast),
        ] {
            let product = super::n_fast_product(&device, rows, (k, n))?;
            ensure!(
                product == ProductKernel::NFastSwizzled,
                "{role} selected {product:?}"
            );
            let (diff, moved) = check_product_tail(&device, (rows, k, n), baseline, product, salt)?;
            println!(
                "CONVROT_N_FAST_SWIZZLE role={role} rows={rows} k={k} n={n} differ={diff} wrong_tail_rows={moved}"
            );
        }
        Ok(())
    }

    fn check_gate_up_stripes(device: &Device) -> Result<()> {
        let (rows, k, n) = (37_717, 5376, 28_672);
        let [selected, _] = super::gate_up_product_order(device, [rows, 0], (k, n))?;
        let forced = match selected {
            ProductKernel::MFastStriped { .. } => false,
            _ if matches!(&device.info.arch, crate::backend::setup::DeviceArchitecture::Hip(gfx) if gfx.starts_with("gfx11")) => {
                true
            }
            _ => return Ok(()),
        };
        let product = if forced {
            ProductKernel::MFastStriped {
                tiles: 59,
                stripes: 5,
                one_barrier: false,
            }
        } else {
            selected
        };
        let ProductKernel::MFastStriped { tiles, stripes, .. } = product else {
            return Ok(());
        };
        if matches!(
            product,
            ProductKernel::MFastStriped {
                one_barrier: true,
                ..
            }
        ) {
            let (codes, scales, weights, wscale) = fixture(rows, k, n);
            let diff = fixture_product_oracle(&codes, &scales, &weights, &wscale, k);
            let baseline = ProductKernel::MFastStriped {
                tiles,
                stripes,
                one_barrier: false,
            };
            for order in [baseline, product] {
                let output = launch_product_at(
                    device,
                    &codes,
                    &scales,
                    &weights,
                    &wscale,
                    k,
                    (0, 0, order),
                )?;
                ensure!(
                    output.len() == rows * n && diff(&output) == 0,
                    "production gate/up oracle {order:?}"
                );
            }
        }
        let (diff, moved_rows) =
            check_product_tail(device, (rows, k, n), ProductKernel::NFast, product, 0)?;
        println!(
            "CONVROT_GATE_UP_STRIPES tiles={tiles} stripes={stripes} rows={rows} k={k} n={n} forced={forced} differ={diff} wrong_tail_rows={moved_rows}"
        );
        let tile = product.tile_dims(&device.info.arch).0 as usize;
        let padded_rows = rows + tile;
        let (codes, scales, weights, wscale) = fixture(padded_rows, 64, 257);
        let padded = if forced {
            ProductKernel::MFastStriped {
                tiles: 60,
                stripes: 5,
                one_barrier: false,
            }
        } else {
            super::gate_up_product_order(device, [padded_rows, 0], (k, n))?[0]
        };
        let narrow = |order| {
            launch_product_at(
                device,
                &codes,
                &scales,
                &weights,
                &wscale,
                64,
                (0, 0, order),
            )
        };
        let diff = bit_diff(&narrow(padded)?, &narrow(ProductKernel::NFast)?);
        ensure!(diff == 0, "partial final stripe differs in {diff} elements");
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_product_matches_host_int8() -> Result<()> {
        let device = Device::new(0)?;
        check_gate_up_product(&device)?;
        check_gate_up_stripes(&device)?;
        for rows in [17usize, 20, 42] {
            for n in [2usize, 40] {
                let k = 256usize;
                let (codes, scales, qweight, wscale) = fixture(rows, k, n);
                let host = host_product(&codes, &scales, k, &qweight, n, &wscale)?;
                let serial = super::int8_product(&codes, &scales, k, &qweight, n, &wscale)?;
                let host_diff = bit_diff(&host, &serial);
                ensure!(
                    host_diff == 0,
                    "host product rows={rows} n={n} differ {host_diff}"
                );
                for product in [
                    ProductKernel::NFast,
                    ProductKernel::MFastLdsm,
                    ProductKernel::NFastLdsm,
                ] {
                    let got = launch_product_at(
                        &device,
                        &codes,
                        &scales,
                        &qweight,
                        &wscale,
                        k,
                        (0, 0, product),
                    )?;
                    let differ = bit_diff(&got, &host);
                    println!(
                        "CONVROT_PRODUCT kernel={} rows={rows} k={k} n={n} differ={differ}",
                        product.symbol_name()
                    );
                    ensure!(
                        differ == 0,
                        "convrot product kernel={} rows={rows} n={n} differ {differ}",
                        product.symbol_name()
                    );
                }
            }
        }
        let rows = 41_770usize;
        let k = 14_336usize;
        let n = 40usize;
        let (codes, scales, qweight, wscale) = fixture(rows, k, n);
        let host = host_product(&codes, &scales, k, &qweight, n, &wscale)?;
        let got = launch_product_at(
            &device,
            &codes,
            &scales,
            &qweight,
            &wscale,
            k,
            (0, 0, ProductKernel::NFast),
        )?;
        let differ = bit_diff(&got, &host);
        println!(
            "CONVROT_PRODUCT rows={rows} k={k} n={n} differ={differ} macs={}",
            rows * k * n
        );
        ensure!(differ == 0, "convrot product production differ {differ}");
        let mut extended = qweight.clone();
        extended.push(0x5a);
        let shifted = launch_product_at(
            &device,
            &codes,
            &scales,
            &extended,
            &wscale,
            k,
            (1, 0, ProductKernel::NFast),
        )?;
        let moved = bit_diff(&shifted, &host);
        println!("CONVROT_PRODUCT_K_OFFSET differ={moved}");
        ensure!(
            moved > 0,
            "one-int8 K offset matched the unshifted host product"
        );
        let scale_shifted = launch_product_at(
            &device,
            &codes,
            &scales,
            &qweight,
            &wscale,
            k,
            (0, 1, ProductKernel::NFast),
        )?;
        let scale_moved = bit_diff(&scale_shifted, &host);
        println!("CONVROT_PRODUCT_SCALE_INDEX differ={scale_moved}");
        ensure!(
            scale_moved > 0,
            "scale index off by one matched the host product"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuobjdump_bin() -> std::path::PathBuf {
        if let Some(home) = std::env::var_os("CUDA_HOME") {
            let candidate = std::path::Path::new(&home).join("bin/cuobjdump");
            if candidate.is_file() {
                return candidate;
            }
        }
        std::path::PathBuf::from("cuobjdump")
    }

    #[cfg(feature = "cuda")]
    fn function_sass<'a>(dump: &'a str, name: &str) -> Option<&'a str> {
        let marker = format!("Function : {name}\n");
        let start = dump.find(&marker)? + marker.len();
        let rest = &dump[start..];
        let end = rest.find("Function :").unwrap_or(rest.len());
        Some(&rest[..end])
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn convrot_product_sass_is_imma16832() -> Result<()> {
        let sms = crate::backend::cuda::cubin::arches();
        ensure!(!sms.is_empty(), "convrot cubin missing");
        for &sm in sms {
            let bytes = crate::backend::cuda::cubin::bytes("convrot", sm)
                .with_context(|| format!("convrot cubin sm_{sm}"))?;
            let path = std::env::temp_dir()
                .join(format!("convrot-product-{}-{sm}.cubin", std::process::id()));
            std::fs::write(&path, bytes).context("convrot cubin")?;
            let output = std::process::Command::new(cuobjdump_bin())
                .arg("-sass")
                .arg(&path)
                .output()
                .context("cuobjdump")?;
            let resources = std::process::Command::new(cuobjdump_bin())
                .arg("-res-usage")
                .arg(&path)
                .output()
                .context("cuobjdump resources")?;
            let _ = std::fs::remove_file(&path);
            let text = String::from_utf8_lossy(&output.stdout);
            let err = String::from_utf8_lossy(&output.stderr);
            ensure!(
                output.status.success(),
                "cuobjdump sm_{sm} status {} {err}",
                output.status
            );
            let usage = String::from_utf8_lossy(&resources.stdout);
            let usage_err = String::from_utf8_lossy(&resources.stderr);
            ensure!(
                resources.status.success(),
                "cuobjdump resources sm_{sm} status {} {usage_err}",
                resources.status
            );
            for (name, want_lds, want_ldsm) in [
                ("convrot_product", 54, 0),
                ("convrot_product_m_fast", 6, 12),
                ("convrot_product_n_fast_ldsm", 6, 12),
            ] {
                let body = function_sass(&text, name)
                    .with_context(|| format!("{name} missing from cuobjdump"))?;
                let ops = sass_ops(body);
                let count = |op| ops.iter().filter(|&&code| code == op).count();
                let imma = count("IMMA.16832.S8.S8");
                let lds = count("LDS");
                let ldsm = count("LDSM.16.M88.4");
                let barriers = ops
                    .iter()
                    .filter(|code| code.starts_with("BAR.SYNC"))
                    .count();
                let (reg, stack, local) = function_resources(&usage, name)
                    .with_context(|| format!("sm_{sm} {name} resources"))?;
                if sm < 80 {
                    let trap = count("BPT.TRAP");
                    println!(
                        "CONVROT_PRODUCT_SASS sm_{sm} {name} IMMA={imma} TRAP={trap} REG={reg} STACK={stack} LOCAL={local}"
                    );
                    ensure!(
                        imma == 0 && trap >= 1 && reg <= 124 && stack == 0 && local == 0,
                        "sm_{sm} {name} IMMA={imma} TRAP={trap} REG={reg} STACK={stack} LOCAL={local}"
                    );
                    continue;
                }
                ensure!(
                    imma == 32 && lds == want_lds && ldsm == want_ldsm && barriers == 2,
                    "sm_{sm} {name} IMMA={imma} LDS={lds} LDSM={ldsm} BAR={barriers}"
                );
                let ldg = body.matches("LDG.E.CONSTANT").count();
                ensure!(ldg == 20, "sm_{sm} {name} scale LDG.E.CONSTANT {ldg}");
                ensure!(
                    reg <= 124 && stack == 0 && local == 0,
                    "sm_{sm} {name} REG={reg} STACK={stack} LOCAL={local}"
                );
                println!(
                    "CONVROT_PRODUCT_SASS sm_{sm} {name} IMMA={imma} LDS={lds} LDSM={ldsm} BAR={barriers} LDG={ldg} REG={reg} STACK={stack} LOCAL={local}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn sass_ops(body: &str) -> Vec<&str> {
        body.split_whitespace()
            .map(|word| word.trim_end_matches(';'))
            .filter(|word| word.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
            .collect()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn convrot_group_dot_sass_loads_the_table() -> Result<()> {
        let sms = crate::backend::cuda::cubin::arches();
        ensure!(!sms.is_empty(), "convrot cubin missing");
        for &sm in sms {
            let bytes = crate::backend::cuda::cubin::bytes("convrot", sm)
                .with_context(|| format!("convrot cubin sm_{sm}"))?;
            let path = std::env::temp_dir().join(format!(
                "convrot-group-dot-{}-{sm}.cubin",
                std::process::id()
            ));
            std::fs::write(&path, bytes).context("convrot cubin")?;
            let output = std::process::Command::new(cuobjdump_bin())
                .arg("-sass")
                .arg(&path)
                .output()
                .context("cuobjdump")?;
            let resources = std::process::Command::new(cuobjdump_bin())
                .arg("-res-usage")
                .arg(&path)
                .output()
                .context("cuobjdump resources")?;
            let _ = std::fs::remove_file(&path);
            let text = String::from_utf8_lossy(&output.stdout);
            let err = String::from_utf8_lossy(&output.stderr);
            ensure!(
                output.status.success(),
                "cuobjdump sm_{sm} status {} {err}",
                output.status
            );
            let body = function_sass(&text, "convrot_group_dot")
                .context("convrot_group_dot missing from cuobjdump")?;
            let ops = sass_ops(body);
            let ldg128 = ops.iter().filter(|op| **op == "LDG.E.128").count();
            let u16 = ops.iter().filter(|op| op.starts_with("LDG.E.U16")).count();
            let e64 = ops.iter().filter(|op| op.starts_with("LDG.E.64")).count();
            let constant = ops.iter().filter(|op| op.contains("CONSTANT")).count();
            let hmma = ops
                .iter()
                .filter(|op| **op == "HMMA.16816.F32.BF16")
                .count();
            let ldsm2 = ops.iter().filter(|op| **op == "LDSM.16.M88.2").count();
            let ldsm4 = ops.iter().filter(|op| **op == "LDSM.16.M88.4").count();
            let mufu = ops.iter().filter(|op| op.starts_with("MUFU")).count();
            let usage = String::from_utf8_lossy(&resources.stdout);
            let usage_err = String::from_utf8_lossy(&resources.stderr);
            ensure!(
                resources.status.success(),
                "cuobjdump resources sm_{sm} status {} {usage_err}",
                resources.status
            );
            let (reg, stack, local) = function_resources(&usage, "convrot_group_dot")?;
            println!(
                "CONVROT_GROUP_DOT_SASS sm_{sm} LDG.E.128={ldg128} U16={u16} E64={e64} CONSTANT={constant} HMMA={hmma} LDSM2={ldsm2} LDSM4={ldsm4} MUFU={mufu} REG={reg} STACK={stack} LOCAL={local}"
            );
            if sm < 80 {
                let trap = ops.iter().filter(|op| **op == "BPT.TRAP").count();
                ensure!(
                    hmma == 0
                        && trap >= 1
                        && u16 == 0
                        && e64 == 0
                        && constant == 0
                        && mufu == 0
                        && stack == 0
                        && local == 0,
                    "sm_{sm} convrot_group_dot HMMA={hmma} TRAP={trap} U16={u16} E64={e64} CONSTANT={constant} MUFU={mufu} STACK={stack} LOCAL={local}"
                );
                continue;
            }
            ensure!(
                ldg128 >= 2
                    && u16 == 0
                    && e64 == 0
                    && constant == 0
                    && hmma >= 1
                    && ldsm2 >= 1
                    && ldsm4 >= 1
                    && mufu == 0
                    && stack == 0
                    && local == 0,
                "sm_{sm} convrot_group_dot LDG.E.128={ldg128} U16={u16} E64={e64} CONSTANT={constant} HMMA={hmma} LDSM2={ldsm2} LDSM4={ldsm4} MUFU={mufu} STACK={stack} LOCAL={local}"
            );
            if sm >= 80 {
                let max_silu_reg = if sm == 120 { 56u32 } else { 48 };
                for (name, max_reg) in [
                    ("convrot_group_dot_32", 48u32),
                    ("convrot_group_dot_silu_32", max_silu_reg),
                ] {
                    let (reg, stack, local) = function_resources(&usage, name)
                        .with_context(|| format!("sm_{sm} {name} resources"))?;
                    println!(
                        "CONVROT_GROUP_DOT_32_SASS sm_{sm} {name} REG={reg} STACK={stack} LOCAL={local}"
                    );
                    ensure!(
                        reg <= max_reg && stack == 0 && local == 0,
                        "sm_{sm} {name} REG={reg} STACK={stack} LOCAL={local}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn function_resources(dump: &str, function: &str) -> Result<(u32, u32, u32)> {
        let marker = format!("Function {function}:");
        let rest = dump
            .find(&marker)
            .map(|start| &dump[start + marker.len()..])
            .with_context(|| format!("{function} resources"))?;
        let end = rest.find("Function ").unwrap_or(rest.len());
        let block = &rest[..end];
        let field = |name: &str| -> Result<u32> {
            block
                .split_whitespace()
                .find_map(|word| word.strip_prefix(name)?.parse().ok())
                .with_context(|| format!("missing {name}"))
        };
        Ok((field("REG:")?, field("STACK:")?, field("LOCAL:")?))
    }

    #[cfg(feature = "rocm")]
    fn rocm_llvm_bin(tool: &str) -> std::path::PathBuf {
        for var in ["HIP_HOME", "ROCM_PATH"] {
            if let Some(home) = std::env::var_os(var) {
                for rel in [format!("llvm/bin/{tool}"), format!("bin/{tool}")] {
                    let candidate = std::path::Path::new(&home).join(rel);
                    if candidate.is_file() {
                        return candidate;
                    }
                }
            }
        }
        if let Some(paths) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&paths) {
                let direct = dir.join(tool);
                if direct.is_file() {
                    return direct;
                }
                if let Some(parent) = dir.parent() {
                    let llvm = parent.join("llvm/bin").join(tool);
                    if llvm.is_file() {
                        return llvm;
                    }
                }
            }
        }
        std::path::PathBuf::from(tool)
    }

    #[cfg(feature = "rocm")]
    fn rocm_kernel_disasm<'a>(dump: &'a str, name: &str) -> Option<&'a str> {
        let marker = format!("<{name}>:\n");
        let start = dump.find(&marker)? + marker.len();
        let rest = &dump[start..];
        let end = rest
            .lines()
            .scan(0usize, |offset, line| {
                let pos = *offset;
                *offset += line.len() + 1;
                Some((pos, line))
            })
            .find(|(_, line)| line.ends_with(">:") && line.contains('<'))
            .map(|(pos, _)| pos)
            .unwrap_or(rest.len());
        Some(&rest[..end])
    }

    #[cfg(feature = "rocm")]
    fn rocm_kernel_metadata(notes: &str, name: &str) -> Result<(u32, u32)> {
        let block = notes
            .split("\n  - .")
            .find(|chunk| {
                chunk
                    .lines()
                    .any(|line| line.trim().strip_prefix(".name:").map(str::trim) == Some(name))
            })
            .with_context(|| format!("{name} missing from llvm-readelf --notes"))?;
        let field = |key: &str| -> Result<u32> {
            block
                .lines()
                .find_map(|line| {
                    line.trim()
                        .strip_prefix(key)
                        .and_then(|rest| rest.trim().parse().ok())
                })
                .with_context(|| format!("missing {key} for {name}"))
        };
        Ok((
            field(".vgpr_count:")?,
            field(".private_segment_fixed_size:")?,
        ))
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn convrot_rocm_hsaco_is_wmma_and_spill_free() -> Result<()> {
        let arches: Vec<&str> = crate::backend::cuda::cubin::rocm_arches()
            .iter()
            .copied()
            .filter(|arch| arch.starts_with("gfx11"))
            .collect();
        ensure!(!arches.is_empty(), "convrot gfx11 hsaco missing");
        for arch in arches {
            let bytes = crate::backend::cuda::cubin::rocm_bytes("convrot", arch)
                .with_context(|| format!("convrot hsaco {arch}"))?;
            let pid = std::process::id();
            let hsaco = std::env::temp_dir().join(format!("convrot-{pid}-{arch}.hsaco"));
            let elf = std::env::temp_dir().join(format!("convrot-{pid}-{arch}.elf"));
            std::fs::write(&hsaco, bytes).context("write convrot hsaco")?;
            let unbundle = std::process::Command::new(rocm_llvm_bin("clang-offload-bundler"))
                .arg("--type=o")
                .arg(format!("--targets=hipv4-amdgcn-amd-amdhsa--{arch}"))
                .arg(format!("--input={}", hsaco.display()))
                .arg(format!("--output={}", elf.display()))
                .arg("--unbundle")
                .output()
                .context("clang-offload-bundler")?;
            let _ = std::fs::remove_file(&hsaco);
            ensure!(
                unbundle.status.success(),
                "clang-offload-bundler {arch} failed: {}",
                String::from_utf8_lossy(&unbundle.stderr)
            );
            let disasm = std::process::Command::new(rocm_llvm_bin("llvm-objdump"))
                .arg("--disassemble")
                .arg(format!("--mcpu={arch}"))
                .arg(&elf)
                .output()
                .context("llvm-objdump")?;
            let notes = std::process::Command::new(rocm_llvm_bin("llvm-readelf"))
                .arg("--notes")
                .arg(&elf)
                .output()
                .context("llvm-readelf")?;
            let _ = std::fs::remove_file(&elf);
            ensure!(
                disasm.status.success() && notes.status.success(),
                "llvm-objdump/llvm-readelf {arch} failed"
            );
            let disasm_text = String::from_utf8_lossy(&disasm.stdout);
            let notes_text = String::from_utf8_lossy(&notes.stdout);
            for (name, want_bf16, max_vgprs) in [
                ("convrot_group_dot", 1usize, 96u32),
                ("convrot_group_dot_silu", 1, 96),
                ("convrot_group_dot_32", 2, 96),
                ("convrot_group_dot_silu_32", 2, 96),
            ] {
                let body = rocm_kernel_disasm(&disasm_text, name)
                    .with_context(|| format!("{arch} {name} missing from llvm-objdump"))?;
                let bf16 = body.matches("v_wmma_f32_16x16x16_bf16").count();
                let scratch =
                    body.matches("scratch_load").count() + body.matches("scratch_store").count();
                let (vgprs, priv_bytes) = rocm_kernel_metadata(&notes_text, name)?;
                println!(
                    "CONVROT_GROUP_DOT_ISA {arch} {name} BF16_WMMA={bf16} VGPR={vgprs} PRIV={priv_bytes} SCRATCH={scratch}"
                );
                ensure!(
                    bf16 == want_bf16 && vgprs <= max_vgprs && priv_bytes == 0 && scratch == 0,
                    "{arch} {name} BF16_WMMA={bf16} VGPR={vgprs} PRIV={priv_bytes} SCRATCH={scratch}"
                );
            }
            for name in [
                "convrot_product",
                "convrot_product_m_fast",
                "convrot_product_n_fast_ldsm",
            ] {
                let body = rocm_kernel_disasm(&disasm_text, name)
                    .with_context(|| format!("{arch} {name} missing from llvm-objdump"))?;
                let iu8 = body.matches("v_wmma_i32_16x16x16_iu8").count();
                let ds_ld128 = body.matches("ds_load_b128").count();
                let barriers = body.matches("s_barrier").count();
                let scratch =
                    body.matches("scratch_load").count() + body.matches("scratch_store").count();
                let (vgprs, priv_bytes) = rocm_kernel_metadata(&notes_text, name)?;
                println!(
                    "CONVROT_PRODUCT_ISA {arch} {name} IU8_WMMA={iu8} DS_LD128={ds_ld128} BAR={barriers} VGPR={vgprs} PRIV={priv_bytes} SCRATCH={scratch}"
                );
                ensure!(
                    iu8 == 32
                        && ds_ld128 == 24
                        && barriers == 2
                        && vgprs <= 216
                        && priv_bytes == 0
                        && scratch == 0,
                    "{arch} {name} IU8_WMMA={iu8} DS_LD128={ds_ld128} BAR={barriers} VGPR={vgprs} PRIV={priv_bytes} SCRATCH={scratch}"
                );
            }
        }
        Ok(())
    }

    fn count_i8(got: &[i8], expected: &[i8]) -> usize {
        got.iter().zip(expected).filter(|(a, b)| a != b).count()
            + got.len().abs_diff(expected.len())
    }

    fn count_f32(got: &[f32], expected: &[f32]) -> usize {
        got.iter()
            .zip(expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
            + got.len().abs_diff(expected.len())
    }

    fn check_rotation_bits(
        base: &Ran,
        wide: &Ran,
        (rows, k, n): (usize, usize, usize),
        fused: bool,
    ) -> Result<()> {
        let xr = count_f32(&base.rotated, &wide.rotated);
        let codes = count_i8(&base.codes, &wide.codes);
        let scales = count_f32(&base.scales, &wide.scales);
        let y = bit_diff(&base.y, &wide.y);
        println!(
            "CONVROT_ROTATE_32 rows={rows} k={k} n={n} fused={fused} xr={xr} codes={codes} scales={scales} y={y}"
        );
        ensure!(
            xr == 0 && codes == 0 && scales == 0 && y == 0,
            "32-row ConvRot differs"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_32_rows_matches_16_rows() -> Result<()> {
        let device = Device::new(0)?;
        let [ordinary, _] = super::group_dot_32_wave_sizes(&device)?;
        use crate::backend::setup::DeviceArchitecture::Cuda;
        if !super::arch_supported(&device.info.arch) {
            return Ok(());
        }
        ensure!(
            ordinary.is_some(),
            "32-row ConvRot resource gate did not admit the ordinary kernel"
        );
        let sm120 = device.info.arch == Cuda(120);
        if sm120 {
            for k in [5376, 7168] {
                ensure!(
                    !super::group_dot_32_fills_wave(1000, k, ordinary),
                    "sm120 ordinary 32-row grid fills a wave at M=1000 K={k}"
                );
            }
        }
        for (rows, k, n) in [
            (17, 256, 40),
            (37717, 5376, 7168),
            (37717, 7168, 5376),
            (1509, 5376, 28672),
            (1501, 5376, 28672),
        ] {
            let x = activation(rows, k);
            let qweight = weight(n, k);
            let wscale = wscale_row(n);
            let input = |group| HostLaunch {
                x: &x,
                k,
                group,
                qweight: &qweight,
                n,
                wscale: &wscale,
                keep: true,
            };
            let selected = super::group_dot_32_fills_wave(rows, k, ordinary);
            if !sm120 || matches!(rows, 17 | 37717) {
                ensure!(
                    selected == (rows != 17),
                    "ordinary 32-row selection M={rows} K={k}"
                );
            }
            let base = launch_kept_tile(&device, input(256), false)?;
            let wide = launch_kept_tile(&device, input(256), true)?;
            check_rotation_bits(&base, &wide, (rows, k, n), false)?;
            if rows == 17 {
                launch_kept_tile(&device, input(64), false)?;
                ensure!(
                    launch_kept_tile(&device, input(64), true).is_err(),
                    "32-row ConvRot accepted group 64"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_silu_matches_separate_activation() -> Result<()> {
        use crate::backend::gpu::PushKernelArg;

        let device = Device::new(0)?;
        let stream = &device.stream;
        let fns = load_convrot(&device.ctx)?;
        let silu =
            crate::backend::cuda::cubin::module(&device.ctx, "act")?.load_function("silu_mul")?;
        let [ordinary_32, fused_32] = super::group_dot_32_wave_sizes(&device)?;
        if super::arch_supported(&device.info.arch) {
            ensure!(
                fused_32.is_some(),
                "fused 32-row ConvRot resource gate did not admit the kernel on {}",
                device.info.arch
            );
            ensure!(
                super::group_dot_32_fills_wave(1000, 14336, fused_32),
                "fused 32-row grid did not qualify at M=1000 K=14336 on {}",
                device.info.arch
            );
        }
        let large_fused = device.ctx.mem_get_info()?.1 >= 32 << 30;
        let (k, n, group) = (14336usize, 5376usize, 256usize);
        let weight = stream.clone_htod(&salted_weight(n, k, 0))?;
        let scale = stream.clone_htod(&salted_wscale(n, 0))?;

        for rows in [17usize, 1509, 1501]
            .into_iter()
            .chain(large_fused.then_some(37_717))
        {
            let mut gu = vec![bf16::ZERO; rows * 2 * k];
            for (row, pair) in gu.chunks_exact_mut(2 * k).enumerate() {
                let (lower, upper) = pair.split_at_mut(k);
                for col in 0..k {
                    lower[col] = bf16::from_f32(((row * 13 + col * 71) % 251) as f32 / 32.0 - 4.0);
                    upper[col] = bf16::from_f32(((row * 37 + col * 41) % 127) as f32 / 16.0 - 4.0);
                }
            }
            let control = (rows - 1) * k + k - group;
            if rows == 1501 {
                let last = &mut gu[(rows - 1) * 2 * k..];
                last[k - group..k].fill(bf16::ZERO);
                last[2 * k - group..].fill(bf16::ZERO);
                last[k - group] = bf16::from_f32(3.25);
                last[2 * k - group] = bf16::from_f32(0.75);
            }

            let input = stream.clone_htod(&gu)?;
            let mut gated = stream.alloc_zeros::<bf16>(rows * k)?;
            let (gup, gap) = (input.device_ptr(stream).0, gated.device_ptr(stream).0);
            let cfg = crate::backend::cuda::ops::silu_mul_grid(rows, k, gup, gap)?;
            let rows_i = i32::try_from(rows)?;
            let k_i = i32::try_from(k)?;
            unsafe {
                stream
                    .launch_builder(&silu)
                    .arg(&gup)
                    .arg(&gap)
                    .arg(&rows_i)
                    .arg(&k_i)
                    .arg(&3i32)
                    .launch(cfg)?;
            }

            let run = |x, fuse_silu, rotate_32| -> Result<Ran> {
                let rotated = stream.alloc_zeros::<f32>((rows + 1) * k)?;
                let codes = stream.alloc_zeros::<i8>((rows + 1) * k)?;
                let scales = stream.alloc_zeros::<f32>(rows + 1)?;
                let y = stream.alloc_zeros::<bf16>((rows + 1) * n)?;
                launch_device(
                    stream,
                    &fns,
                    &ConvrotArgs {
                        x,
                        qweight: weight.device_ptr(stream).0,
                        wscale: scale.device_ptr(stream).0,
                        y: y.device_ptr(stream).0,
                        rotated: rotated.device_ptr(stream).0,
                        codes: codes.device_ptr(stream).0,
                        scales: scales.device_ptr(stream).0,
                        rows: rows_i,
                        k: k_i,
                        n: i32::try_from(n)?,
                        group: i32::try_from(group)?,
                        product: ProductKernel::NFast,
                        fuse_silu,
                        rotate_32,
                    },
                )?;
                stream.synchronize()?;
                let ran = Ran {
                    y: stream.clone_dtoh(&y)?,
                    rotated: stream.clone_dtoh(&rotated)?,
                    codes: stream.clone_dtoh(&codes)?,
                    scales: stream.clone_dtoh(&scales)?,
                };
                ensure!(
                    ran.rotated[rows * k..].iter().all(|v| v.to_bits() == 0)
                        && ran.codes[rows * k..].iter().all(|&v| v == 0)
                        && ran.scales[rows].to_bits() == 0
                        && ran.y[rows * n..].iter().all(|v| v.to_bits() == 0),
                    "ConvRot SiLU tail guard rows={rows} fused={fuse_silu}"
                );
                Ok(ran)
            };

            let separate = run(gap, false, false)?;
            let fused = run(gup, true, false)?;
            let xr = count_f32(&separate.rotated, &fused.rotated);
            let codes = count_i8(&separate.codes, &fused.codes);
            let scales = count_f32(&separate.scales, &fused.scales);
            let y = bit_diff(&separate.y, &fused.y);
            println!(
                "CONVROT_SILU rows={rows} k={k} n={n} xr={xr} codes={codes} scales={scales} y={y}"
            );
            ensure!(
                xr == 0 && codes == 0 && scales == 0 && y == 0,
                "fused SiLU rows={rows} differs"
            );
            #[cfg(feature = "rocm")]
            if rows == 17 {
                let gated_host: Vec<f32> = stream
                    .clone_dtoh(&gated)?
                    .into_iter()
                    .map(bf16::to_f32)
                    .collect();
                let (max_abs, rmse) =
                    f64_rotated_stats(&gated_host, &fused.rotated[..rows * k], k, group, None)?;
                println!(
                    "CONVROT_SILU_ROCM_F64 rows={rows} k={k} max_abs={max_abs:.9e} rmse={rmse:.9e}"
                );
                ensure!(
                    max_abs <= ROCM_WMMA_SILU_ROT_MAX_ABS && rmse <= ROCM_WMMA_SILU_ROT_RMSE,
                    "fused SiLU rocm f64 rows={rows} k={k} max_abs={max_abs:.9e} rmse={rmse:.9e}"
                );
            }
            if fused_32.is_some() {
                let selected = super::group_dot_32_fills_wave(rows, k, fused_32);
                ensure!(
                    selected == (rows != 17),
                    "fused 32-row selection M={rows} K={k}"
                );
                let wide = run(gup, true, true)?;
                check_rotation_bits(&fused, &wide, (rows, k, n), true)?;
            }

            if rows == 1501 {
                let actual = stream.clone_dtoh(&gated.slice(control..control + 1))?;
                ensure!(
                    actual == [bf16::from_f32(1.6484375)],
                    "SiLU control two-round value"
                );
                stream.memcpy_htod(
                    &[bf16::from_f32(1.65625)],
                    &mut gated.slice_mut(control..control + 1),
                )?;
                let mutant = run(gap, false, ordinary_32.is_some())?;
                let differing = count_f32(&separate.rotated, &mutant.rotated);
                let target = control..control + group;
                let inside = count_f32(&separate.rotated[target.clone()], &mutant.rotated[target]);
                println!(
                    "CONVROT_SILU_ROUND_CONTROL rows={rows} differ={differing} inside={inside}"
                );
                ensure!(
                    differing == group && inside == group,
                    "one-round SiLU control escaped selected group"
                );
            }
        }
        Ok(())
    }

    fn salted_weight(n: usize, k: usize, salt: usize) -> Vec<i8> {
        (0..n)
            .flat_map(|col| (0..k).map(move |i| code(col + salt * n, i)))
            .collect()
    }

    fn salted_wscale(n: usize, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|col| ((col + salt) % 5 + 1) as f32 * 1.0e-2)
            .collect()
    }

    fn per_product(
        device: &Device,
        fns: &super::ConvrotFns,
        x: &[f32],
        k: usize,
        group: usize,
        qweight: &[i8],
        wscale: &[f32],
    ) -> Result<(Vec<i8>, Vec<f32>, Vec<bf16>)> {
        let n = wscale.len();
        let rows = x.len() / k;
        let stream = &device.stream;
        let held = upload_activation(stream, x, rows)?;
        let weight_d = stream.clone_htod(qweight)?;
        let scale_d = stream.clone_htod(wscale)?;
        let y_d = stream.alloc_zeros::<bf16>(rows * n)?;
        let rotated = stream.alloc_zeros::<f32>(rows * k)?;
        let codes = stream.alloc_zeros::<i8>(rows * k)?;
        let scales = stream.alloc_zeros::<f32>(rows)?;
        launch_device(
            stream,
            fns,
            &ConvrotArgs {
                x: held.ptr,
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
                product: ProductKernel::NFast,
                fuse_silu: false,
                rotate_32: false,
            },
        )?;
        stream.synchronize()?;
        held.release();
        Ok((
            stream.clone_dtoh(&codes)?,
            stream.clone_dtoh(&scales)?,
            stream.clone_dtoh(&y_d)?,
        ))
    }

    fn check_shared(
        device: &Device,
        fns: &super::ConvrotFns,
        rows: usize,
        k: usize,
        n: usize,
        product: ProductKernel,
        mutant: bool,
    ) -> Result<()> {
        let group = 256usize;
        let x = activation(rows, k);
        let stream = &device.stream;
        let held = upload_activation(stream, &x, rows)?;
        let rotated = stream.alloc_zeros::<f32>(rows * k)?;
        let codes_d = stream.alloc_zeros::<i8>(rows * k)?;
        let scales_d = stream.alloc_zeros::<f32>(rows)?;
        super::launch_rotate_quant(
            stream,
            fns,
            &ConvrotArgs {
                x: held.ptr,
                qweight: 0,
                wscale: 0,
                y: 0,
                rotated: rotated.device_ptr(stream).0,
                codes: codes_d.device_ptr(stream).0,
                scales: scales_d.device_ptr(stream).0,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: 0,
                group: i32::try_from(group)?,
                product: ProductKernel::NFast,
                fuse_silu: false,
                rotate_32: false,
            },
        )?;
        stream.synchronize()?;
        held.release();
        drop(rotated);
        let shared_codes: Vec<i8> = stream.clone_dtoh(&codes_d)?;
        let shared_scales: Vec<f32> = stream.clone_dtoh(&scales_d)?;
        let mut codes_differ = 0usize;
        let mut scales_differ = 0usize;
        let mut product_differ = [0usize; 3];
        let mut kept_y = None;
        let mut kept_weight = None;
        let mut kept_wscale = None;
        for (salt, differ) in product_differ.iter_mut().enumerate() {
            let qweight = salted_weight(n, k, salt);
            let wscale = salted_wscale(n, salt);
            let (codes, scales, y) = per_product(device, fns, &x, k, group, &qweight, &wscale)?;
            codes_differ += count_i8(&shared_codes, &codes);
            scales_differ += count_f32(&shared_scales, &scales);
            let weight_d = stream.clone_htod(&qweight)?;
            let wscale_d = stream.clone_htod(&wscale)?;
            let y_d = stream.alloc_zeros::<bf16>(rows * n)?;
            super::launch_product(
                stream,
                fns,
                &ConvrotArgs {
                    x: 0,
                    qweight: weight_d.device_ptr(stream).0,
                    wscale: wscale_d.device_ptr(stream).0,
                    y: y_d.device_ptr(stream).0,
                    rotated: 0,
                    codes: codes_d.device_ptr(stream).0,
                    scales: scales_d.device_ptr(stream).0,
                    rows: i32::try_from(rows)?,
                    k: i32::try_from(k)?,
                    n: i32::try_from(n)?,
                    group: i32::try_from(group)?,
                    product,
                    fuse_silu: false,
                    rotate_32: false,
                },
            )?;
            stream.synchronize()?;
            let shared_y: Vec<bf16> = stream.clone_dtoh(&y_d)?;
            *differ = bit_diff(&shared_y, &y);
            if mutant && salt == 0 {
                kept_y = Some(shared_y);
                kept_weight = Some(qweight);
                kept_wscale = Some(wscale);
            }
        }
        println!(
            "CONVROT_SHARED rows={rows} k={k} n={n} codes={codes_differ} scales={scales_differ} y0={} y1={} y2={}",
            product_differ[0], product_differ[1], product_differ[2]
        );
        ensure!(
            codes_differ == 0,
            "shared codes rows={rows} differ {codes_differ}"
        );
        ensure!(
            scales_differ == 0,
            "shared scales rows={rows} differ {scales_differ}"
        );
        ensure!(
            product_differ == [0, 0, 0],
            "shared products rows={rows} differ {product_differ:?}"
        );
        if !mutant {
            return Ok(());
        }
        let kept_y = kept_y.context("shared product")?;
        let kept_weight = kept_weight.context("shared weight")?;
        let kept_wscale = kept_wscale.context("shared wscale")?;
        let mut wrong = shared_codes;
        let packed = matches!(product, ProductKernel::NFastLdsm);
        if packed {
            wrong[..32].rotate_left(16);
        } else {
            wrong[0] = if wrong[0] < 0 { 127 } else { -127 };
        }
        let shifted = launch_product_at(
            device,
            &wrong,
            &shared_scales,
            &kept_weight,
            &kept_wscale,
            k,
            (0, 0, product),
        )?;
        let moved = bit_diff(&shifted, &kept_y);
        if packed {
            let row = bit_diff(&shifted[..n], &kept_y[..n]);
            println!("CONVROT_SHARED_LDSM_K_HALF_CONTROL differ={moved} first_row={row}");
            ensure!(
                moved > 0 && moved == row,
                "QKV K-half control must change only the first output row"
            );
        } else {
            println!("CONVROT_SHARED_MUTANT differ={moved}");
            ensure!(moved > 0, "one changed shared code matched the product");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_shared_quant_matches_per_product() -> Result<()> {
        let device = Device::new(0)?;
        let fns = load_convrot(&device.ctx)?;
        let k = 5_376usize;
        let n = 7_168usize;
        let expected_qkv = if super::n_fast_swizzle_fits(&device.info, 37_717, (k, n), 1) {
            ProductKernel::NFastSwizzled
        } else if super::qkv_product_eligible(&device.info, 37_717, (k, n)) {
            super::product_active_ctas(&device, ProductKernel::NFastLdsm)?;
            ProductKernel::NFastLdsm
        } else {
            ProductKernel::NFast
        };
        ensure!(
            super::n_fast_product(&device, 37_717, (k, n))? == expected_qkv,
            "QKV product selection mismatch on {}",
            device.info.arch
        );
        for rows in [17usize, 20, 42] {
            check_shared(&device, &fns, rows, k, n, ProductKernel::NFast, false)?;
        }
        check_shared(&device, &fns, 42, k, n, ProductKernel::NFastLdsm, false)?;
        check_shared(&device, &fns, 41_770, k, n, ProductKernel::NFast, true)?;
        check_shared(&device, &fns, 37_717, k, n, ProductKernel::NFastLdsm, true)?;
        Ok(())
    }
}
