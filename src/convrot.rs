//! Int8 linear with a block-diagonal regular Hadamard rotate-quant prologue.
//! `y = bf16((q(x R) @ qweight^T) * s_x * wscale)`, int32 products.
//! `R` is `kron(H4 / 2, ...)` of `group` (a power of 4).
//! Below [`TENSOR_CORE_ROWS`], `x` is f32 cast to bf16, each rotated group is an f32 dot rounded
//! back to bf16, `s_x = max(|xR|, 1e-12) / 127`, and `q` is ties-to-even of `xR / s_x`.
//! At [`TENSOR_CORE_ROWS`] and above, `x` is bf16. The fused device kernel computes the Radix-4
//! regular Hadamard transform in f32 round-to-nearest (register-resident up to `group = 256`,
//! shared-memory staged above) and quantizes `q = rint(xR * (1/s_x))` in one pass with no
//! intermediate f32 rotation buffer in global memory.
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

/// Static shared bytes reserved for warp-peak reduction in the fused FWHT kernels (`32 * f32`).
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) const FWHT_PEAK_SMEM_BYTES: usize = 128;

/// Dynamic shared bytes available to the shared-memory FWHT kernel, 16-byte aligned.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn fwht_max_dynamic_smem(optin: usize) -> Result<u32> {
    let avail = optin.saturating_sub(FWHT_PEAK_SMEM_BYTES) / 16 * 16;
    ensure!(avail >= 16, "convrot opt-in shared memory {optin}");
    u32::try_from(avail).context("convrot opt-in shared memory")
}

/// Validates that `group` is a supported power of 4 and fits `optin` when staged in shared memory.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn ensure_group(group: u32, optin: usize) -> Result<()> {
    let g = usize::try_from(group)?;
    ensure!(
        power_of_four(g),
        "convrot group {group} is not a power of 4"
    );
    if g > 256 {
        let need = g.checked_mul(4).context("convrot group smem")?;
        let avail = usize::try_from(fwht_max_dynamic_smem(optin)?)?;
        ensure!(
            need <= avail,
            "convrot group {group} requires {need} shared bytes (available {avail})"
        );
    }
    Ok(())
}

/// Fused 1-pass Radix-4 FWHT + quantize execution plan, derived at setup from hardware resources.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotatePlan {
    /// Register-resident (`group <= 256`, `chunks_per_warp ∈ {1, 2, 4}`, `<= 128 B` static smem).
    Register { chunks_per_warp: u8, threads: u32 },
    /// Shared-memory staged (`group > 256` or `k` beyond register capacity; `smem_bytes = k * 4`).
    Shared { threads: u32, smem_bytes: u32 },
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
impl RotatePlan {
    pub(crate) const fn symbol_name(self, fuse_silu: bool) -> &'static str {
        match (self, fuse_silu) {
            (
                Self::Register {
                    chunks_per_warp: 1, ..
                },
                false,
            ) => "convrot_fwht_quantize_1",
            (
                Self::Register {
                    chunks_per_warp: 2, ..
                },
                false,
            ) => "convrot_fwht_quantize_2",
            (
                Self::Register {
                    chunks_per_warp: 4, ..
                },
                false,
            ) => "convrot_fwht_quantize_4",
            (
                Self::Register {
                    chunks_per_warp: 1, ..
                },
                true,
            ) => "convrot_fwht_quantize_silu_1",
            (
                Self::Register {
                    chunks_per_warp: 2, ..
                },
                true,
            ) => "convrot_fwht_quantize_silu_2",
            (
                Self::Register {
                    chunks_per_warp: 4, ..
                },
                true,
            ) => "convrot_fwht_quantize_silu_4",
            (Self::Shared { .. }, false) => "convrot_fwht_quantize_smem",
            (Self::Shared { .. }, true) => "convrot_fwht_quantize_silu_smem",
            _ => unreachable!(),
        }
    }
}

/// Derives the fused FWHT plan from detected device properties and per-variant `(chunks_per_warp, threads, local_bytes, active_ctas)`.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn select_rotate_plan(
    info: &crate::backend::setup::DeviceInfo,
    k: usize,
    group: usize,
    reg_candidates: &[(u8, u32, i32, u32)],
) -> Result<RotatePlan> {
    ensure!(
        info.warp == 32 && info.max_threads >= 32,
        "convrot FWHT requires warp32 device (warp={} max_threads={})",
        info.warp,
        info.max_threads
    );
    ensure!(
        power_of_four(group) && k > 0 && k.is_multiple_of(group) && k.is_multiple_of(8),
        "convrot FWHT shape k={k} group={group}"
    );
    let max_warps = ((info.max_threads / 32) as usize).min(32);
    let chunks = k.div_ceil(256);
    if group <= 256 {
        let mut best = None;
        for &(chunks_per_warp, threads, local, active) in reg_candidates {
            if !matches!(chunks_per_warp, 1 | 2 | 4)
                || local != 0
                || active == 0
                || threads == 0
                || !threads.is_multiple_of(32)
                || threads as usize > max_warps * 32
            {
                continue;
            }
            let warps = (threads as usize) / 32;
            if chunks > usize::from(chunks_per_warp) * warps {
                continue;
            }
            let score = (
                u64::from(active) * (chunks as u64) * u64::from(4 / chunks_per_warp),
                active >= 2,
                std::cmp::Reverse(chunks_per_warp),
                active,
            );
            let plan = RotatePlan::Register {
                chunks_per_warp,
                threads,
            };
            if best.is_none_or(|(prev, _)| score > prev) {
                best = Some((score, plan));
            }
        }
        if let Some((_, plan)) = best {
            return Ok(plan);
        }
    }
    let avail = usize::try_from(fwht_max_dynamic_smem(info.optin_shared_bytes)?)?;
    let need = k.checked_mul(4).context("convrot FWHT smem")?;
    ensure!(
        need <= avail,
        "convrot FWHT k={k} group={group} requires {need} shared bytes (available {avail})"
    );
    let target_warps = if need * 2 <= avail {
        chunks.clamp(1, 8)
    } else {
        chunks.clamp(1, 16)
    }
    .min(max_warps);
    Ok(RotatePlan::Shared {
        threads: u32::try_from(target_warps * 32)?,
        smem_bytes: u32::try_from(need)?,
    })
}

#[cfg(test)]
mod tests {
    use super::{hadamard, linear, linear_omitting_rotation};
    use anyhow::{Result, ensure};
    use half::bf16;

    #[test]
    fn fwht_rotate_plan_derives_register_and_shared_across_gpu_classes() -> Result<()> {
        use super::{RotatePlan, ensure_group, select_rotate_plan};

        let sm89 = mock_device_info(89, 128, 72 << 20, 101_376);
        let sm80 = mock_device_info(80, 108, 40 << 20, 166_912);
        let sm90 = mock_device_info(90, 132, 50 << 20, 232_448);
        let sm75 = mock_device_info(75, 40, 6 << 20, 65_536);
        let sm120 = mock_device_info(120, 188, 128 << 20, 101_376);
        let gfx1100 = mock_hip_device_info("gfx1100", 48, 6 << 20, 65_536);

        // sm89 / sm120 (48 warps/SM): K=5376 -> c=1 (21 warps, 2 CTAs); K=7168 -> c=2 (14 warps, 3 CTAs);
        // K=14336 -> c=4 (14 warps, 2 CTAs beats 1 CTA at c=2; falls to c=2 if c=4 spills).
        for info in [&sm89, &sm120] {
            assert_eq!(
                select_rotate_plan(
                    info,
                    5376,
                    256,
                    &[(1, 672, 0, 2), (2, 352, 0, 4), (4, 192, 0, 6)]
                )?,
                RotatePlan::Register {
                    chunks_per_warp: 1,
                    threads: 672
                }
            );
            assert_eq!(
                select_rotate_plan(
                    info,
                    7168,
                    256,
                    &[(1, 896, 0, 1), (2, 448, 0, 3), (4, 224, 0, 5)]
                )?,
                RotatePlan::Register {
                    chunks_per_warp: 2,
                    threads: 448
                }
            );
            assert_eq!(
                select_rotate_plan(info, 14336, 256, &[(2, 896, 0, 1), (4, 448, 0, 2)])?,
                RotatePlan::Register {
                    chunks_per_warp: 4,
                    threads: 448
                }
            );
            assert_eq!(
                select_rotate_plan(info, 14336, 256, &[(2, 896, 0, 1), (4, 448, 16, 2)])?,
                RotatePlan::Register {
                    chunks_per_warp: 2,
                    threads: 896
                }
            );
        }
        // sm80 / sm90 (64 warps/SM): K=5376 -> c=1 (3 CTAs); K=7168 -> c=1 (2 CTAs); K=14336 -> c=2 (2 CTAs).
        for info in [&sm80, &sm90] {
            assert_eq!(
                select_rotate_plan(info, 5376, 256, &[(1, 672, 0, 3), (2, 352, 0, 5)])?,
                RotatePlan::Register {
                    chunks_per_warp: 1,
                    threads: 672
                }
            );
            assert_eq!(
                select_rotate_plan(info, 7168, 256, &[(1, 896, 0, 2), (2, 448, 0, 4)])?,
                RotatePlan::Register {
                    chunks_per_warp: 1,
                    threads: 896
                }
            );
            assert_eq!(
                select_rotate_plan(info, 14336, 256, &[(2, 896, 0, 2), (4, 448, 0, 4)])?,
                RotatePlan::Register {
                    chunks_per_warp: 2,
                    threads: 896
                }
            );
        }
        // sm75 T4 (32 warps/SM): prefers 2 CTAs/SM over 1 CTA/SM at equal parallel chunks.
        assert_eq!(
            select_rotate_plan(&sm75, 5376, 256, &[(1, 672, 0, 1), (2, 352, 0, 2)])?,
            RotatePlan::Register {
                chunks_per_warp: 2,
                threads: 352
            }
        );
        // gfx1100 wave32: register-resident plan derived identically.
        assert_eq!(
            select_rotate_plan(&gfx1100, 7168, 256, &[(1, 896, 0, 1), (2, 448, 0, 2)])?,
            RotatePlan::Register {
                chunks_per_warp: 2,
                threads: 448
            }
        );
        // group > 256 derives Shared smem path; K beyond register capacity derives Shared on H100 and errors on sm89.
        let smem_plan = select_rotate_plan(&sm89, 7168, 1024, &[])?;
        assert_eq!(
            smem_plan,
            RotatePlan::Shared {
                threads: 256,
                smem_bytes: 28_672
            }
        );
        assert_eq!(smem_plan.symbol_name(false), "convrot_fwht_quantize_smem");
        assert_eq!(
            smem_plan.symbol_name(true),
            "convrot_fwht_quantize_silu_smem"
        );
        for (c, plain, silu) in [
            (
                1u8,
                "convrot_fwht_quantize_1",
                "convrot_fwht_quantize_silu_1",
            ),
            (2, "convrot_fwht_quantize_2", "convrot_fwht_quantize_silu_2"),
            (4, "convrot_fwht_quantize_4", "convrot_fwht_quantize_silu_4"),
        ] {
            let p = RotatePlan::Register {
                chunks_per_warp: c,
                threads: 256,
            };
            assert_eq!(p.symbol_name(false), plain);
            assert_eq!(p.symbol_name(true), silu);
        }
        assert_eq!(
            select_rotate_plan(&sm90, 49_152, 256, &[])?,
            RotatePlan::Shared {
                threads: 512,
                smem_bytes: 196_608
            }
        );
        assert!(select_rotate_plan(&sm89, 49_152, 256, &[]).is_err());
        // Unsupported non-power-of-4 groups and non-multiple k error (no fallback).
        assert!(select_rotate_plan(&sm89, 5376, 128, &[]).is_err());
        assert!(select_rotate_plan(&sm89, 5376, 3, &[]).is_err());
        assert!(select_rotate_plan(&sm89, 5380, 256, &[]).is_err());
        ensure_group(256, sm75.optin_shared_bytes)?;
        ensure_group(1024, sm75.optin_shared_bytes)?;
        assert!(ensure_group(128, sm89.optin_shared_bytes).is_err());
        assert!(ensure_group(16_384, sm75.optin_shared_bytes).is_err());
        Ok(())
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
        use super::{ProductKernel, gate_up_products, qkv_product_eligible};

        use ProductKernel::{MFastLdsm, MFastStriped, NFast};

        let a100 = mock_device_info(80, 108, 40 << 20, 166_912);
        let a4000 = mock_device_info(86, 48, 4 << 20, 101_376);
        let a4000_large_l2 = mock_device_info(86, 48, 40 << 20, 101_376);
        let rtx4090 = mock_device_info(89, 128, 72 << 20, 101_376);
        let rtx4090_low_smem = mock_device_info(89, 128, 72 << 20, 32_768);
        let l4 = mock_device_info(89, 58, 48 << 20, 101_376);
        let h100 = mock_device_info(90, 132, 50 << 20, 232_448);
        let g4 = mock_device_info(120, 188, 128 << 20, 101_376);
        let t4 = mock_device_info(75, 40, 40 << 20, 65_536);
        let gfx1100 = mock_hip_device_info("gfx1100", 48, 6 << 20, 65_536);
        let gfx1100_large_l2 = mock_hip_device_info("gfx1100", 48, 40 << 20, 65_536);
        let gfx1100_low_smem = mock_hip_device_info("gfx1100", 48, 40 << 20, 32_768);
        let gfx1030 = mock_hip_device_info("gfx1030", 48, 40 << 20, 65_536);

        assert_eq!(
            ProductKernel::NFast.symbol_name_for(&rtx4090.arch),
            "convrot_product"
        );
        assert_eq!(
            ProductKernel::NFastLdsm.symbol_name_for(&rtx4090.arch),
            "convrot_product_n_fast_ldsm"
        );
        for (stages, symbol, smem) in [
            (2u8, "convrot_product_m_fast_s2", 49_152u32),
            (3u8, "convrot_product_m_fast_s3", 73_728u32),
            (4u8, "convrot_product_m_fast_s4", 98_304u32),
        ] {
            let kernel = ProductKernel::MFastLdsm { stages };
            assert_eq!(kernel.symbol_name_for(&rtx4090.arch), symbol);
            assert_eq!(kernel.shared_mem_bytes(&rtx4090.arch), smem);
            assert_eq!(kernel.symbol_name_for(&a100.arch), symbol);
            assert_eq!(kernel.symbol_name_for(&h100.arch), symbol);
        }
        assert_eq!(
            ProductKernel::MFastLdsm { stages: 2 }.symbol_name_for(&g4.arch),
            "convrot_product_m_fast"
        );
        assert_eq!(
            ProductKernel::MFastStriped {
                tiles: 59,
                stripes: 5,
                one_barrier: true,
                stages: 2,
            }
            .symbol_name_for(&g4.arch),
            "convrot_product_m_fast_one_barrier"
        );
        assert_eq!(
            ProductKernel::MFastLdsm { stages: 2 }.symbol_name_for(&gfx1100.arch),
            "convrot_product_m_fast"
        );
        assert_eq!(
            ProductKernel::MFastLdsm { stages: 2 }.shared_mem_bytes(&gfx1100.arch),
            32_768
        );
        assert_eq!(
            ProductKernel::MFastLdsm { stages: 2 }.tile_dims(&gfx1100.arch),
            (128, 128)
        );
        for (stages, symbol, smem) in [
            (2u8, "convrot_product_n_fast_swizzled", 49_152u32),
            (3u8, "convrot_product_n_fast_swizzled_s3", 73_728u32),
            (4u8, "convrot_product_n_fast_swizzled_s4", 98_304u32),
        ] {
            let kernel = ProductKernel::NFastSwizzled { stages };
            assert_eq!(kernel.symbol_name_for(&rtx4090.arch), symbol);
            assert_eq!(kernel.shared_mem_bytes(&rtx4090.arch), smem);
            assert_eq!(kernel.shared_mem_bytes(&gfx1100.arch), 49_152);
            assert_eq!(kernel.tile_dims(&gfx1100.arch), (128, 256));
        }
        assert_eq!(ProductKernel::NFast.shared_mem_bytes(&rtx4090.arch), 61_440);
        assert_eq!(
            ProductKernel::NFastLdsm.shared_mem_bytes(&rtx4090.arch),
            61_440
        );
        assert_eq!(ProductKernel::NFast.shared_mem_bytes(&gfx1100.arch), 32_768);
        assert_eq!(
            ProductKernel::NFastLdsm.shared_mem_bytes(&gfx1100.arch),
            49_152
        );
        assert_eq!(ProductKernel::NFast.tile_dims(&gfx1100.arch), (128, 128));
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

        let optin_72k = mock_device_info(89, 128, 72 << 20, 73_728);
        for dims in [(5376, 7168), (7168, 5376), (14336, 5376)] {
            for info in [&a100, &a4000, &l4, &rtx4090, &h100, &g4] {
                assert_eq!(super::n_fast_swizzle_stage(info, 37_717, dims), Some(4));
            }
            assert_eq!(
                super::n_fast_swizzle_stage(&optin_72k, 37_717, dims),
                Some(3)
            );
            assert_eq!(
                super::n_fast_swizzle_stage(&low_smem, 37_717, dims),
                Some(2)
            );
            assert_eq!(super::n_fast_swizzle_stage(&rtx4090, 17, dims), Some(4));
            assert_eq!(
                super::n_fast_swizzle_stage(&rtx4090_low_smem, 37_717, dims),
                None
            );
            assert_eq!(super::n_fast_swizzle_stage(&t4, 37_717, dims), None);
            assert_eq!(super::n_fast_swizzle_stage(&gfx1100, 37_717, dims), None);
            assert_eq!(super::n_fast_swizzle_stage(&g4, 15, dims), None);
        }
        assert_eq!(
            super::n_fast_swizzle_stage(&rtx4090, 37_717, (5376 + 16, 7168)),
            None
        );

        for info in [&a100, &a4000, &l4, &rtx4090, &h100] {
            assert_eq!(super::m_fast_stage(info), Some(4));
        }
        assert_eq!(super::m_fast_stage(&optin_72k), Some(3));
        assert_eq!(super::m_fast_stage(&low_smem), Some(2));
        assert_eq!(super::m_fast_stage(&g4), Some(2));
        assert_eq!(super::m_fast_stage(&gfx1100), Some(2));
        assert_eq!(super::m_fast_stage(&gfx1100_low_smem), Some(2));
        assert_eq!(
            super::m_fast_stage(&mock_hip_device_info("gfx1100", 48, 40 << 20, 32_767)),
            None
        );
        assert_eq!(super::m_fast_stage(&rtx4090_low_smem), None);
        assert_eq!(super::m_fast_stage(&t4), None);
        assert_eq!(super::m_fast_stage(&gfx1030), None);

        let gate_up = (5376, 28_672);
        assert_eq!(
            gate_up_products(&a100, 1, [1509, 1501], gate_up, 4),
            [MFastLdsm { stages: 4 }; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [3057, 5400], gate_up, 4),
            [MFastLdsm { stages: 4 }; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [5678, 39_740], gate_up, 4),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [37_717, 128], gate_up, 4),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [128, 129], gate_up, 4),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a100, 1, [128, 513], gate_up, 4),
            [NFast, MFastLdsm { stages: 4 }]
        );
        assert_eq!(
            gate_up_products(&a4000, 1, [1509, 1501], gate_up, 4),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a4000, 1, [256, 37_717], gate_up, 4),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&a4000_large_l2, 1, [128, 129], gate_up, 4),
            [NFast, MFastLdsm { stages: 4 }]
        );
        assert_eq!(
            gate_up_products(&a4000_large_l2, 1, [1509, 37_717], gate_up, 4),
            [
                MFastLdsm { stages: 4 },
                MFastStriped {
                    tiles: 17,
                    stripes: 18,
                    one_barrier: false,
                    stages: 4,
                }
            ]
        );
        assert_eq!(
            gate_up_products(&l4, 1, [1509, 37_717], gate_up, 4),
            [
                MFastLdsm { stages: 4 },
                MFastStriped {
                    tiles: 23,
                    stripes: 13,
                    one_barrier: false,
                    stages: 4,
                }
            ]
        );
        assert_eq!(
            gate_up_products(&h100, 1, [1509, 37_717], gate_up, 4),
            [MFastLdsm { stages: 4 }, NFast]
        );
        assert_eq!(
            gate_up_products(&rtx4090, 1, [9935, 39_740], gate_up, 4),
            [
                MFastLdsm { stages: 4 },
                MFastStriped {
                    tiles: 32,
                    stripes: 10,
                    one_barrier: false,
                    stages: 4,
                }
            ]
        );
        assert_eq!(
            gate_up_products(&rtx4090, 1, [37_717, 0], gate_up, 4),
            [
                MFastStriped {
                    tiles: 30,
                    stripes: 10,
                    one_barrier: false,
                    stages: 4,
                },
                NFast
            ]
        );
        assert_eq!(
            gate_up_products(&optin_72k, 1, [37_717, 0], gate_up, 3),
            [
                MFastStriped {
                    tiles: 30,
                    stripes: 10,
                    one_barrier: false,
                    stages: 3,
                },
                NFast
            ]
        );
        assert_eq!(
            gate_up_products(&low_smem, 1, [1509, 1501], gate_up, 2),
            [MFastLdsm { stages: 2 }; 2]
        );
        assert_eq!(
            gate_up_products(&rtx4090_low_smem, 1, [9935, 37_717], gate_up, 2),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&g4, 1, [9935, 37_717], gate_up, 2),
            [
                MFastLdsm { stages: 2 },
                MFastStriped {
                    tiles: 59,
                    stripes: 5,
                    one_barrier: true,
                    stages: 2,
                }
            ]
        );
        assert_eq!(
            gate_up_products(&t4, 1, [1509, 37_717], gate_up, 2),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&gfx1100, 2, [1509, 1501], gate_up, 2),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&gfx1100, 2, [9935, 37_717], gate_up, 2),
            [NFast; 2]
        );
        assert_eq!(
            gate_up_products(&gfx1100_large_l2, 2, [1509, 37_717], gate_up, 2),
            [
                MFastLdsm { stages: 2 },
                MFastStriped {
                    tiles: 19,
                    stripes: 16,
                    one_barrier: false,
                    stages: 2,
                }
            ]
        );
        assert_eq!(
            gate_up_products(&gfx1030, 2, [1509, 37_717], gate_up, 2),
            [NFast; 2]
        );
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
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub struct ConvrotFns {
    linear: crate::backend::gpu::CudaFunction,
    fwht_reg: [[crate::backend::gpu::CudaFunction; 3]; 2],
    fwht_smem: [crate::backend::gpu::CudaFunction; 2],
    #[cfg(test)]
    fwht_probe: [crate::backend::gpu::CudaFunction; 3],
    product: [crate::backend::gpu::CudaFunction; 2],
    arch: crate::backend::setup::DeviceArchitecture,
    striped: bool,
    m_fast: [Option<crate::backend::gpu::CudaFunction>; 3],
    m_fast_one_barrier: Option<crate::backend::gpu::CudaFunction>,
    n_fast_swizzled: [Option<crate::backend::gpu::CudaFunction>; 3],
    optin: usize,
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductKernel {
    NFast,
    MFastLdsm {
        stages: u8,
    },
    MFastStriped {
        tiles: u32,
        stripes: u32,
        one_barrier: bool,
        stages: u8,
    },
    NFastLdsm,
    NFastSwizzled {
        stages: u8,
    },
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub struct ConvrotArgs {
    pub x: u64,
    pub qweight: u64,
    pub wscale: u64,
    pub y: u64,
    pub codes: u64,
    pub scales: u64,
    pub rows: i32,
    pub k: i32,
    pub n: i32,
    pub group: i32,
    pub product: ProductKernel,
    pub fuse_silu: bool,
    pub rotate: Option<RotatePlan>,
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
impl ProductKernel {
    fn shared_mem_bytes(self, arch: &crate::backend::setup::DeviceArchitecture) -> u32 {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => match self {
                Self::MFastLdsm { stages }
                | Self::MFastStriped { stages, .. }
                | Self::NFastSwizzled { stages } => u32::from(stages) * 24_576,
                Self::NFast | Self::NFastLdsm => 61_440,
            },
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm { .. } | Self::MFastStriped { .. } => 32_768,
                Self::NFastLdsm | Self::NFastSwizzled { .. } => 49_152,
            },
            crate::backend::setup::DeviceArchitecture::Tpu(_) => 0,
        }
    }

    fn tile_dims(self, arch: &crate::backend::setup::DeviceArchitecture) -> (u32, u32) {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => (128, 256),
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm { .. } | Self::MFastStriped { .. } => (128, 128),
                Self::NFastLdsm | Self::NFastSwizzled { .. } => (128, 256),
            },
            crate::backend::setup::DeviceArchitecture::Tpu(_) => (0, 0),
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn threads(self, arch: &crate::backend::setup::DeviceArchitecture) -> u32 {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => 512,
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm { .. } | Self::MFastStriped { .. } => 256,
                Self::NFastLdsm | Self::NFastSwizzled { .. } => 512,
            },
            crate::backend::setup::DeviceArchitecture::Tpu(_) => 0,
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn resource_limits(self, arch: &crate::backend::setup::DeviceArchitecture) -> (i32, u32) {
        match arch {
            crate::backend::setup::DeviceArchitecture::Cuda(_) => (124, 1),
            crate::backend::setup::DeviceArchitecture::Hip(_) => match self {
                Self::NFast | Self::MFastLdsm { .. } | Self::MFastStriped { .. } => (216, 2),
                Self::NFastLdsm | Self::NFastSwizzled { .. } => (216, 1),
            },
            crate::backend::setup::DeviceArchitecture::Tpu(_) => (0, 0),
        }
    }

    const ONE_BARRIER_SYMBOL: &'static str = "convrot_product_m_fast_one_barrier";

    /// Kernel entry symbol for `arch`.
    pub(crate) fn symbol_name_for(
        self,
        arch: &crate::backend::setup::DeviceArchitecture,
    ) -> &'static str {
        match self {
            Self::NFast => "convrot_product",
            Self::MFastStriped {
                one_barrier: true, ..
            } => Self::ONE_BARRIER_SYMBOL,
            Self::MFastLdsm { stages } | Self::MFastStriped { stages, .. } => match arch {
                crate::backend::setup::DeviceArchitecture::Cuda(sm) if (80..120).contains(sm) => {
                    match stages {
                        2 => "convrot_product_m_fast_s2",
                        3 => "convrot_product_m_fast_s3",
                        4 => "convrot_product_m_fast_s4",
                        _ => "convrot_product_m_fast_unsupported",
                    }
                }
                crate::backend::setup::DeviceArchitecture::Cuda(120)
                | crate::backend::setup::DeviceArchitecture::Hip(_) => "convrot_product_m_fast",
                crate::backend::setup::DeviceArchitecture::Cuda(_)
                | crate::backend::setup::DeviceArchitecture::Tpu(_) => {
                    "convrot_product_m_fast_unsupported"
                }
            },
            Self::NFastLdsm => "convrot_product_n_fast_ldsm",
            Self::NFastSwizzled { stages: 2 } => "convrot_product_n_fast_swizzled",
            Self::NFastSwizzled { stages: 3 } => "convrot_product_n_fast_swizzled_s3",
            Self::NFastSwizzled { stages: 4 } => "convrot_product_n_fast_swizzled_s4",
            Self::NFastSwizzled { .. } => "convrot_product_n_fast_swizzled_unsupported",
        }
    }
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn arch_supported(arch: &crate::backend::setup::DeviceArchitecture) -> bool {
    use crate::backend::setup::DeviceArchitecture::{Cuda, Hip, Tpu};
    match arch {
        Cuda(sm) => *sm >= 80,
        Hip(gfx) => gfx.starts_with("gfx11"),
        Tpu(_) => false,
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

/// Maximum M-fast pipeline stages fitting `info.optin_shared_bytes` (`2..=4` on CUDA sm80..90, `2` on sm120/gfx11).
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn m_fast_stage(info: &crate::backend::setup::DeviceInfo) -> Option<u8> {
    use crate::backend::setup::DeviceArchitecture::Cuda;
    if info.warp != 32 || info.sms == 0 {
        return None;
    }
    match &info.arch {
        Cuda(sm) if (80..120).contains(sm) => [4u8, 3, 2].into_iter().find(|&stages| {
            info.optin_shared_bytes
                >= ProductKernel::MFastLdsm { stages }.shared_mem_bytes(&info.arch) as usize
        }),
        arch if arch_supported(arch) => (info.optin_shared_bytes
            >= ProductKernel::MFastLdsm { stages: 2 }.shared_mem_bytes(arch) as usize)
            .then_some(2),
        _ => None,
    }
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn gate_up_products(
    info: &crate::backend::setup::DeviceInfo,
    active_ctas: usize,
    row_counts: [usize; 2],
    (k, n): (usize, usize),
    stages: u8,
) -> [ProductKernel; 2] {
    let l2 = info.l2_bytes;
    let ldsm = ProductKernel::MFastLdsm { stages };
    if !arch_supported(&info.arch)
        || info.warp != 32
        || info.optin_shared_bytes < ldsm.shared_mem_bytes(&info.arch) as usize
        || active_ctas == 0
        || n.saturating_mul(k) <= l2
    {
        return [ProductKernel::NFast; 2];
    }
    let wave = info.sms * active_ctas;
    let (bm_u32, bn_u32) = ldsm.tile_dims(&info.arch);
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
            return ldsm;
        }
        if rows.saturating_mul(k) > l2 {
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
                        stages,
                    };
                }
            }
        }
        ProductKernel::NFast
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn function_attributes(function: &crate::backend::gpu::CudaFunction) -> Result<(i32, i32, usize)> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        let version = function.get_attribute(Attr::CU_FUNC_ATTRIBUTE_BINARY_VERSION)?;
        let regs = function.get_attribute(Attr::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
        let local = function.get_attribute(Attr::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
        let static_smem = function.get_attribute(Attr::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
        ensure!(
            version > 0 && regs > 0 && local >= 0 && static_smem >= 0,
            "corrupt CUDA function attributes version={version} regs={regs} local={local} static_smem={static_smem}"
        );
        Ok((regs, local, usize::try_from(static_smem)?))
    }
    #[cfg(feature = "rocm")]
    {
        let regs = function.registers_per_thread()?;
        let local = function.local_size_bytes()?;
        let static_smem = function.shared_size_bytes()?;
        ensure!(
            regs > 0 && local >= 0 && static_smem >= 0,
            "corrupt HIP function attributes regs={regs} local={local} static_smem={static_smem}"
        );
        Ok((regs, local, usize::try_from(static_smem)?))
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

/// Validates stage count (`2..=4`) for pipelined ConvRot product variants.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn validate_product_stages(product: ProductKernel) -> Result<()> {
    match product {
        ProductKernel::MFastLdsm { stages }
        | ProductKernel::MFastStriped { stages, .. }
        | ProductKernel::NFastSwizzled { stages } => {
            ensure!(
                matches!(stages, 2..=4),
                "ConvRot {product:?} supports 2..=4 stages, got {stages}"
            );
        }
        ProductKernel::NFast | ProductKernel::NFastLdsm => {}
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn product_function(
    module: &std::sync::Arc<crate::backend::gpu::CudaModule>,
    arch: &crate::backend::setup::DeviceArchitecture,
    product: ProductKernel,
) -> Result<crate::backend::gpu::CudaFunction> {
    validate_product_stages(product)?;
    let function = module.load_function(product.symbol_name_for(arch))?;
    crate::backend::cuda::set_max_dynamic_shared_bytes(&function, product.shared_mem_bytes(arch))?;
    Ok(function)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn probe_product_stage(
    device: &crate::backend::cuda::Device,
    product: ProductKernel,
) -> Result<std::result::Result<usize, String>> {
    let arch = &device.info.arch;
    let smem = product.shared_mem_bytes(arch) as usize;
    let symbol = product.symbol_name_for(arch);
    ensure!(
        arch_supported(arch) && device.info.warp == 32,
        "ConvRot {symbol} requires supported warp32 arch (arch={} warp={})",
        device.info.arch,
        device.info.warp,
    );
    validate_product_stages(product)?;
    let module = crate::backend::cuda::cubin::module(&device.ctx, "convrot")?;
    let function = module.load_function(symbol)?;
    let (regs, local, static_smem) = function_attributes(&function)?;
    let (max_regs, min_ctas) = product.resource_limits(arch);
    ensure!(
        regs <= max_regs && local == 0,
        "ConvRot {symbol} regs={regs} (limit {max_regs}) local={local}"
    );
    let total_smem = static_smem
        .checked_add(smem)
        .context("ConvRot shared memory extent overflow")?;
    if device.info.optin_shared_bytes < total_smem {
        return Ok(Err(format!(
            "optin shared bytes {} < static {static_smem} + dynamic {smem}",
            device.info.optin_shared_bytes
        )));
    }
    crate::backend::cuda::set_max_dynamic_shared_bytes(&function, u32::try_from(smem)?)?;
    let active = active_blocks(&function, product.threads(arch), smem)?;
    if active < min_ctas {
        return Ok(Err(format!(
            "{active} resident CTAs < min {min_ctas} (regs={regs} local={local} static={static_smem} dynamic={smem})"
        )));
    }
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "CONVROT_PRODUCT_RESOURCE kernel={symbol} regs={regs} local={local} static_smem={static_smem} dynamic_smem={smem} active_ctas={active}"
        );
    }
    Ok(Ok(usize::try_from(active)?))
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn product_active_ctas(
    device: &crate::backend::cuda::Device,
    product: ProductKernel,
) -> Result<usize> {
    probe_product_stage(device, product)?.map_err(|reason| {
        anyhow::anyhow!(
            "ConvRot {} {reason}",
            product.symbol_name_for(&device.info.arch)
        )
    })
}

#[cfg(any(test, feature = "cuda", feature = "rocm"))]
fn n_fast_swizzle_stage(
    info: &crate::backend::setup::DeviceInfo,
    rows: usize,
    (k, n): (usize, usize),
) -> Option<u8> {
    if !matches!(info.arch, crate::backend::setup::DeviceArchitecture::Cuda(sm) if sm >= 80)
        || info.warp != 32
        || info.sms == 0
        || rows < TENSOR_CORE_ROWS
        || k < 64
        || !k.is_multiple_of(64)
        || n == 0
    {
        return None;
    }
    [4u8, 3, 2].into_iter().find(|&stages| {
        info.optin_shared_bytes
            >= ProductKernel::NFastSwizzled { stages }.shared_mem_bytes(&info.arch) as usize
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn n_fast_product(
    device: &crate::backend::cuda::Device,
    rows: usize,
    dims: (usize, usize),
) -> Result<ProductKernel> {
    let info = &device.info;
    if let Some(max_stages) = n_fast_swizzle_stage(info, rows, dims) {
        for stages in [4u8, 3, 2].into_iter().filter(|&s| s <= max_stages) {
            let kernel = ProductKernel::NFastSwizzled { stages };
            match probe_product_stage(device, kernel)? {
                Ok(_active) => return Ok(kernel),
                Err(reason) if crate::backend::diagnostics_enabled() => {
                    eprintln!(
                        "CONVROT_N_FAST_DECLINE kernel={} reason={reason}",
                        kernel.symbol_name_for(&info.arch)
                    );
                }
                Err(_) => {}
            }
        }
    } else if crate::backend::diagnostics_enabled() {
        eprintln!(
            "CONVROT_N_FAST_DECLINE stage=all arch={} warp={} optin={} rows={} dims={dims:?}",
            info.arch, info.warp, info.optin_shared_bytes, rows
        );
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
    let mut selected = [ProductKernel::NFast; 2];
    if n.saturating_mul(k) > info.l2_bytes
        && let Some(max_stages) = m_fast_stage(info)
    {
        let stages_to_try: &[u8] = if matches!(
            info.arch,
            crate::backend::setup::DeviceArchitecture::Cuda(sm) if (80..120).contains(&sm)
        ) {
            &[4, 3, 2]
        } else {
            &[2]
        };
        for &stages in stages_to_try.iter().filter(|&&s| s <= max_stages) {
            let probe = ProductKernel::MFastLdsm { stages };
            match probe_product_stage(device, probe)? {
                Ok(active) => {
                    let candidate = gate_up_products(info, active, row_counts, (k, n), stages);
                    if let Some(&kernel) = candidate.iter().find(|p| {
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
                    selected = candidate;
                    break;
                }
                Err(reason) if crate::backend::diagnostics_enabled() => {
                    eprintln!(
                        "CONVROT_M_FAST_DECLINE kernel={} reason={reason}",
                        probe.symbol_name_for(&info.arch)
                    );
                }
                Err(_) => {}
            }
        }
    }
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "CONVROT_GATE_UP_SELECT row_counts={row_counts:?} k={k} n={n} l2_bytes={} products={selected:?}",
            info.l2_bytes
        );
    }
    Ok(selected)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
const FWHT_REG_NAMES: [[&str; 3]; 2] = [
    [
        "convrot_fwht_quantize_1",
        "convrot_fwht_quantize_2",
        "convrot_fwht_quantize_4",
    ],
    [
        "convrot_fwht_quantize_silu_1",
        "convrot_fwht_quantize_silu_2",
        "convrot_fwht_quantize_silu_4",
    ],
];

#[cfg(any(feature = "cuda", feature = "rocm"))]
const FWHT_SMEM_NAMES: [&str; 2] = [
    "convrot_fwht_quantize_smem",
    "convrot_fwht_quantize_silu_smem",
];

#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
const FWHT_PROBE_NAMES: [&str; 3] = [
    "convrot_fwht_quantize_probe",
    "convrot_fwht_quantize_silu_probe",
    "convrot_fwht_quantize_smem_probe",
];

/// Derives the fused 1-pass Radix-4 FWHT + quantize plan once at setup from hardware resources.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn derive_rotate_plan(
    device: &crate::backend::cuda::Device,
    k: usize,
    group: usize,
    fuse_silu: bool,
) -> Result<RotatePlan> {
    let info = &device.info;
    ensure!(
        info.warp == 32 && info.max_threads >= 32,
        "convrot FWHT requires warp32 device (warp={} max_threads={})",
        info.warp,
        info.max_threads
    );
    ensure!(
        power_of_four(group) && k > 0 && k.is_multiple_of(group) && k.is_multiple_of(8),
        "convrot FWHT shape k={k} group={group}"
    );
    let module = crate::backend::cuda::cubin::module(&device.ctx, "convrot")?;
    let silu = usize::from(fuse_silu);
    let max_warps = ((info.max_threads / 32) as usize).min(32);
    let chunks = k.div_ceil(256);
    let mut reg_candidates = Vec::new();
    if group <= 256 {
        for (idx, chunks_per_warp) in [1u8, 2, 4].into_iter().enumerate() {
            let warps = chunks.div_ceil(usize::from(chunks_per_warp));
            if warps > max_warps {
                continue;
            }
            let threads = u32::try_from(warps * 32)?;
            let name = FWHT_REG_NAMES[silu][idx];
            let function = module.load_function(name)?;
            let (regs, local, _) = function_attributes(&function)?;
            let active = active_blocks(&function, threads, 0)?;
            if crate::backend::diagnostics_enabled() {
                eprintln!(
                    "CONVROT_FWHT_CANDIDATE kernel={name} k={k} group={group} threads={threads} regs={regs} local={local} active_ctas={active}"
                );
            }
            reg_candidates.push((chunks_per_warp, threads, local, active));
        }
    }
    let plan = select_rotate_plan(info, k, group, &reg_candidates)?;
    if let RotatePlan::Shared {
        threads,
        smem_bytes,
    } = plan
    {
        let name = FWHT_SMEM_NAMES[silu];
        let function = module.load_function(name)?;
        crate::backend::cuda::set_max_dynamic_shared_bytes(
            &function,
            fwht_max_dynamic_smem(info.optin_shared_bytes)?,
        )?;
        let (regs, local, _) = function_attributes(&function)?;
        let active = active_blocks(&function, threads, smem_bytes as usize)?;
        ensure!(
            local == 0 && active >= 1,
            "ConvRot {name} regs={regs} local={local} active_ctas={active}"
        );
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "CONVROT_FWHT_CANDIDATE kernel={name} k={k} group={group} threads={threads} smem={smem_bytes} regs={regs} local={local} active_ctas={active}"
            );
        }
    }
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "CONVROT_FWHT_SELECT k={k} group={group} silu={fuse_silu} kernel={} plan={plan:?}",
            plan.symbol_name(fuse_silu)
        );
    }
    Ok(plan)
}

/// Loads and validates one staged product kernel when `optin` fits static + dynamic shared memory.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn load_staged_slot(
    module: &std::sync::Arc<crate::backend::gpu::CudaModule>,
    arch: &crate::backend::setup::DeviceArchitecture,
    optin: usize,
    kernel: ProductKernel,
) -> Result<Option<crate::backend::gpu::CudaFunction>> {
    let symbol = kernel.symbol_name_for(arch);
    let function = module.load_function(symbol)?;
    let (regs, local, static_smem) = function_attributes(&function)?;
    let (max_regs, _) = kernel.resource_limits(arch);
    ensure!(
        regs <= max_regs && local == 0,
        "ConvRot {symbol} regs={regs} local={local}"
    );
    let needed = static_smem
        .checked_add(kernel.shared_mem_bytes(arch) as usize)
        .context("ConvRot shared memory extent overflow")?;
    if optin < needed {
        return Ok(None);
    }
    crate::backend::cuda::set_max_dynamic_shared_bytes(&function, kernel.shared_mem_bytes(arch))?;
    Ok(Some(function))
}

/// Loads the `2..=4`-stage family produced by `make_kernel`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn load_staged_products(
    module: &std::sync::Arc<crate::backend::gpu::CudaModule>,
    arch: &crate::backend::setup::DeviceArchitecture,
    optin: usize,
    make_kernel: impl Fn(u8) -> ProductKernel,
) -> Result<[Option<crate::backend::gpu::CudaFunction>; 3]> {
    let mut slots = [None, None, None];
    for (slot, stages) in slots.iter_mut().zip([2u8, 3, 4]) {
        *slot = load_staged_slot(module, arch, optin, make_kernel(stages))?;
    }
    Ok(slots)
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
        product_function(&module, &arch, ProductKernel::NFastLdsm)?,
    ];
    let (striped, sm120_binary) = {
        #[cfg(feature = "cuda")]
        {
            let version = product[0].get_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
            )?;
            ensure!(version > 0, "corrupt ConvRot binary version {version}");
            (version >= 80, version == 120)
        }
        #[cfg(feature = "rocm")]
        {
            (ctx.architecture().starts_with("gfx11"), false)
        }
    };
    let fwht_reg = [
        [
            module.load_function(FWHT_REG_NAMES[0][0])?,
            module.load_function(FWHT_REG_NAMES[0][1])?,
            module.load_function(FWHT_REG_NAMES[0][2])?,
        ],
        [
            module.load_function(FWHT_REG_NAMES[1][0])?,
            module.load_function(FWHT_REG_NAMES[1][1])?,
            module.load_function(FWHT_REG_NAMES[1][2])?,
        ],
    ];
    let fwht_smem = [
        module.load_function(FWHT_SMEM_NAMES[0])?,
        module.load_function(FWHT_SMEM_NAMES[1])?,
    ];
    let optin = crate::backend::cuda::optin_shared_bytes(ctx)?;
    let fwht_dyn = fwht_max_dynamic_smem(optin)?;
    for function in &fwht_smem {
        crate::backend::cuda::set_max_dynamic_shared_bytes(function, fwht_dyn)?;
    }
    #[cfg(test)]
    let fwht_probe = {
        let smem = module.load_function(FWHT_PROBE_NAMES[2])?;
        crate::backend::cuda::set_max_dynamic_shared_bytes(&smem, fwht_dyn)?;
        [
            module.load_function(FWHT_PROBE_NAMES[0])?,
            module.load_function(FWHT_PROBE_NAMES[1])?,
            smem,
        ]
    };
    let m_fast = match &arch {
        crate::backend::setup::DeviceArchitecture::Cuda(sm)
            if (80..120).contains(sm) && striped =>
        {
            load_staged_products(&module, &arch, optin, |stages| ProductKernel::MFastLdsm {
                stages,
            })?
        }
        other
            if (other == &crate::backend::setup::DeviceArchitecture::Cuda(120) && sm120_binary)
                || (matches!(other, crate::backend::setup::DeviceArchitecture::Hip(gfx) if gfx.starts_with("gfx11"))
                    && striped) =>
        {
            [
                load_staged_slot(
                    &module,
                    &arch,
                    optin,
                    ProductKernel::MFastLdsm { stages: 2 },
                )?,
                None,
                None,
            ]
        }
        _ => [None, None, None],
    };
    let n_fast_swizzled = {
        #[cfg(feature = "cuda")]
        {
            if matches!(
                arch,
                crate::backend::setup::DeviceArchitecture::Cuda(sm) if sm >= 80
            ) && striped
            {
                load_staged_products(&module, &arch, optin, |stages| {
                    ProductKernel::NFastSwizzled { stages }
                })?
            } else {
                [None, None, None]
            }
        }
        #[cfg(feature = "rocm")]
        {
            [None, None, None]
        }
    };
    let m_fast_one_barrier =
        if arch == crate::backend::setup::DeviceArchitecture::Cuda(120) && sm120_binary {
            let f = module.load_function(ProductKernel::ONE_BARRIER_SYMBOL)?;
            crate::backend::cuda::set_max_dynamic_shared_bytes(
                &f,
                ProductKernel::MFastLdsm { stages: 2 }.shared_mem_bytes(&arch),
            )?;
            Some(f)
        } else {
            None
        };
    Ok(ConvrotFns {
        linear: module.load_function("convrot_linear")?,
        fwht_reg,
        fwht_smem,
        #[cfg(test)]
        fwht_probe,
        arch,
        striped,
        m_fast,
        n_fast_swizzled,
        m_fast_one_barrier,
        product,
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

/// Resolves a `2..=4`-stage function slot from `slots`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn staged_product_kernel<'a>(
    slots: &'a [Option<crate::backend::gpu::CudaFunction>; 3],
    stages: u8,
    kind: &str,
    arch: &crate::backend::setup::DeviceArchitecture,
) -> Result<&'a crate::backend::gpu::CudaFunction> {
    let idx = usize::from(stages.checked_sub(2).context("pipelined stage count")?);
    slots
        .get(idx)
        .and_then(|slot| slot.as_ref())
        .with_context(|| format!("{kind} product stages={stages} unavailable on {arch}"))
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
        ProductKernel::MFastLdsm { stages } => (
            staged_product_kernel(&fns.m_fast, stages, "M-fast", &fns.arch)?,
            (m, n, 1),
        ),
        ProductKernel::MFastStriped {
            tiles,
            stripes,
            one_barrier,
            stages,
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
                staged_product_kernel(&fns.m_fast, stages, "M-fast", &fns.arch)?
            };
            (kernel, (tiles, n, stripes))
        }
        ProductKernel::NFastLdsm => (&fns.product[1], (n, m, 1)),
        ProductKernel::NFastSwizzled { stages } => (
            staged_product_kernel(&fns.n_fast_swizzled, stages, "swizzled N-fast", &fns.arch)?,
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

/// Fused 1-pass Radix-4 FWHT + per-row INT8 quantize. The product is a separate launch.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn launch_rotate_quant(
    stream: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    fns: &ConvrotFns,
    args: &ConvrotArgs,
) -> Result<()> {
    use crate::backend::gpu::{LaunchConfig, PushKernelArg};
    ensure!(
        args.rows > 0 && args.k > 0 && args.group > 0,
        "convrot FWHT extent"
    );
    let rows = u32::try_from(args.rows)?;
    let k = usize::try_from(args.k)?;
    let group = usize::try_from(args.group)?;
    ensure!(
        power_of_four(group) && k.is_multiple_of(group) && k.is_multiple_of(8),
        "convrot FWHT shape k={k} group={group}"
    );
    let plan = args.rotate.context("convrot FWHT plan missing")?;
    let silu = usize::from(args.fuse_silu);
    let (function, threads, shared_mem_bytes) = match plan {
        RotatePlan::Register {
            chunks_per_warp,
            threads,
        } => {
            ensure!(
                group <= 256
                    && threads > 0
                    && threads.is_multiple_of(32)
                    && threads <= 1024
                    && k.div_ceil(256) <= usize::from(chunks_per_warp) * (threads as usize / 32),
                "convrot FWHT register plan {plan:?} invalid for k={k} group={group}"
            );
            let idx = match chunks_per_warp {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => anyhow::bail!("convrot FWHT chunks_per_warp {chunks_per_warp}"),
            };
            (&fns.fwht_reg[silu][idx], threads, 0u32)
        }
        RotatePlan::Shared {
            threads,
            smem_bytes,
        } => {
            let need = u32::try_from(k.checked_mul(4).context("convrot FWHT smem")?)?;
            let avail = fwht_max_dynamic_smem(fns.optin)?;
            ensure!(
                smem_bytes == need
                    && smem_bytes <= avail
                    && threads > 0
                    && threads.is_multiple_of(32)
                    && threads <= 1024,
                "convrot FWHT shared plan {plan:?} invalid for k={k} avail={avail}"
            );
            (&fns.fwht_smem[silu], threads, smem_bytes)
        }
    };
    unsafe {
        stream
            .launch_builder(function)
            .arg(&args.x)
            .arg(&args.codes)
            .arg(&args.scales)
            .arg(&args.rows)
            .arg(&args.k)
            .arg(&args.group)
            .launch(LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
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
        codes: Vec<i8>,
        scales: Vec<f32>,
    }

    #[derive(Clone, Copy)]
    struct HostLaunch<'a> {
        x: &'a [f32],
        k: usize,
        group: usize,
        qweight: &'a [i8],
        n: usize,
        wscale: &'a [f32],
        keep: bool,
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

    fn launch_kept(device: &Device, input: HostLaunch<'_>) -> Result<Ran> {
        let rows = input.x.len() / input.k;
        let rotate = if rows >= super::TENSOR_CORE_ROWS {
            Some(super::derive_rotate_plan(
                device,
                input.k,
                input.group,
                false,
            )?)
        } else {
            None
        };
        launch_kept_opt_plan(device, input, rotate)
    }

    fn launch_kept_plan(
        device: &Device,
        input: HostLaunch<'_>,
        plan: super::RotatePlan,
    ) -> Result<Ran> {
        launch_kept_opt_plan(device, input, Some(plan))
    }

    fn launch_kept_opt_plan(
        device: &Device,
        input: HostLaunch<'_>,
        rotate: Option<super::RotatePlan>,
    ) -> Result<Ran> {
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
                codes: codes.device_ptr(stream).0,
                scales: scales.device_ptr(stream).0,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: i32::try_from(n)?,
                group: i32::try_from(group)?,
                product: ProductKernel::NFast,
                fuse_silu: false,
                rotate,
            },
        )?;
        stream.synchronize()?;
        held.release();
        let y = stream.clone_dtoh(&y_d)?;
        let (codes, scales) = if keep {
            (stream.clone_dtoh(&codes)?, stream.clone_dtoh(&scales)?)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Ran { y, codes, scales })
    }

    fn emit(line: &str) {
        println!("{line}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    #[derive(Clone, Copy, Debug)]
    struct F64Distance {
        rotated_max_abs: f64,
        rotated_rmse: f64,
        code_mismatch_count: usize,
        code_mismatch_frac: f64,
        code_max_abs_diff: i32,
        non_tie_code_mismatch: usize,
        scale_max_rel_err: f64,
    }

    #[derive(Clone, Copy, Debug)]
    struct IncumbentBar {
        rotated_max_abs: f64,
        code_mismatch_count: usize,
        code_mismatch_frac: f64,
        scale_max_rel_err: f64,
    }

    impl IncumbentBar {
        const fn max_f64(a: f64, b: f64) -> f64 {
            if a >= b { a } else { b }
        }

        const fn max_with(self, oracle: Self) -> Self {
            Self {
                rotated_max_abs: Self::max_f64(self.rotated_max_abs, oracle.rotated_max_abs),
                code_mismatch_count: if self.code_mismatch_count >= oracle.code_mismatch_count {
                    self.code_mismatch_count
                } else {
                    oracle.code_mismatch_count
                },
                code_mismatch_frac: Self::max_f64(
                    self.code_mismatch_frac,
                    oracle.code_mismatch_frac,
                ),
                scale_max_rel_err: Self::max_f64(self.scale_max_rel_err, oracle.scale_max_rel_err),
            }
        }
    }

    // 2026-11-24 gate correction after first candidate run: bar = max(incumbent_mma, oracle_f32),
    // where oracle_f32 is the exact f64 Kronecker H4/2 definition rounded to f32 RN.
    // Moves only step_k5376 scale_rel (1.045093079238e-7 -> 1.1086593136772697e-7) due to row 24594
    // exact 25-bit halfway tie (-0x1.088357p+0); all other bars stay at incumbent MMA.
    const INCUMBENT_SYNTHETIC_BAR: IncumbentBar = IncumbentBar {
        rotated_max_abs: 0.0,
        code_mismatch_count: 0,
        code_mismatch_frac: 0.0,
        scale_max_rel_err: 4.907_208_338_175e-8,
    }
    .max_with(IncumbentBar {
        rotated_max_abs: 0.0,
        code_mismatch_count: 0,
        code_mismatch_frac: 0.0,
        scale_max_rel_err: 4.907_208_338_175e-8,
    });
    const INCUMBENT_STEP_K5376_BAR: IncumbentBar = IncumbentBar {
        rotated_max_abs: 8.940_696_716_309e-8,
        code_mismatch_count: 195,
        code_mismatch_frac: 195.0 / (37_717.0 * 5_376.0),
        scale_max_rel_err: 1.045_093_079_238e-7,
    }
    .max_with(IncumbentBar {
        rotated_max_abs: 5.960_464_477_539e-8,
        code_mismatch_count: 195,
        code_mismatch_frac: 195.0 / (37_717.0 * 5_376.0),
        scale_max_rel_err: 1.108_659_313_677_269_7e-7,
    });
    const INCUMBENT_STEP_K7168_BAR: IncumbentBar = IncumbentBar {
        rotated_max_abs: 5.215_406_417_847e-7,
        code_mismatch_count: 236,
        code_mismatch_frac: 236.0 / (37_717.0 * 7_168.0),
        scale_max_rel_err: 1.555_275_536_410e-7,
    }
    .max_with(IncumbentBar {
        rotated_max_abs: 2.384_185_791_016e-7,
        code_mismatch_count: 236,
        code_mismatch_frac: 236.0 / (37_717.0 * 7_168.0),
        scale_max_rel_err: 1.156_987_463_809e-7,
    });
    const INCUMBENT_STEP_K14336_BAR: IncumbentBar = IncumbentBar {
        rotated_max_abs: 2.338_886_260_987e-4,
        code_mismatch_count: 691,
        code_mismatch_frac: 691.0 / (37_717.0 * 14_336.0),
        scale_max_rel_err: 2.867_293_573_185e-7,
    }
    .max_with(IncumbentBar {
        rotated_max_abs: 1.454_353_332_520e-5,
        code_mismatch_count: 635,
        code_mismatch_frac: 635.0 / (37_717.0 * 14_336.0),
        scale_max_rel_err: 1.175_828_724_085e-7,
    });
    const INCUMBENT_STEP_K14336_SILU_BAR: IncumbentBar = INCUMBENT_STEP_K14336_BAR;

    fn hadamard_f64_table(group: usize) -> Vec<f64> {
        let mut table = vec![0.0f64; group * group];
        for col in 0..group {
            for r in 0..group {
                let mut sign = 1i32;
                let mut scale = 1.0f64;
                let mut n = group;
                let mut rr = r;
                let mut cc = col;
                while n > 1 {
                    n /= 4;
                    let rb = rr / n;
                    let cb = cc / n;
                    rr %= n;
                    cc %= n;
                    sign *= super::H4[rb][cb];
                    scale *= 0.5;
                }
                table[col * group + r] = f64::from(sign) * scale;
            }
        }
        table
    }

    fn f64_distance_bf16(
        x_bf16: &[bf16],
        rotated: &[f32],
        codes: &[i8],
        scales: &[f32],
        k: usize,
        group: usize,
        row_period: Option<usize>,
    ) -> Result<F64Distance> {
        ensure!(
            k > 0
                && super::power_of_four(group)
                && k.is_multiple_of(group)
                && x_bf16.len() == rotated.len()
                && x_bf16.len() == codes.len()
                && x_bf16.len().is_multiple_of(k),
            "convrot f64 distance shape"
        );
        let rows = x_bf16.len() / k;
        ensure!(scales.len() == rows, "convrot f64 distance scales");
        let table = hadamard_f64_table(group);
        let groups = k / group;
        let threads = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .clamp(1, 8)
            .min(rows.max(1));

        // When rows are periodic (synthetic fixtures), compute one period of exact f64 rows once.
        let (ref_rot, ref_codes, ref_scales, unique) = if let Some(period) = row_period {
            let unique = rows.min(period);
            let mut rr = vec![0.0f64; unique * k];
            let mut rc = vec![0i8; unique * k];
            let mut rs = vec![0.0f64; unique];
            let mut x_row = vec![0.0f64; k];
            for r in 0..unique {
                for (dst, &v) in x_row.iter_mut().zip(&x_bf16[r * k..(r + 1) * k]) {
                    *dst = f64::from(v.to_f32());
                }
                let out = &mut rr[r * k..(r + 1) * k];
                let mut peak = 0.0f64;
                for g in 0..groups {
                    let base = g * group;
                    for col in 0..group {
                        let mut sum = 0.0f64;
                        let hcol = &table[col * group..(col + 1) * group];
                        for i in 0..group {
                            sum += x_row[base + i] * hcol[i];
                        }
                        out[base + col] = sum;
                        peak = peak.max(sum.abs());
                    }
                }
                let scale = peak.max(1.0e-12) / 127.0;
                rs[r] = scale;
                for i in 0..k {
                    rc[r * k + i] = (out[i] / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                }
            }
            (Some(rr), Some(rc), Some(rs), unique)
        } else {
            (None, None, None, 0)
        };

        let rows_per = rows.div_ceil(threads);
        let mut partials = vec![(0.0f64, 0.0f64, 0usize, 0i32, 0usize, 0.0f64, false); threads];
        std::thread::scope(|scope| {
            let table = table.as_slice();
            let ref_rot = ref_rot.as_deref();
            let ref_codes = ref_codes.as_deref();
            let ref_scales = ref_scales.as_deref();
            for (tid, slot) in partials.iter_mut().enumerate() {
                let start = tid * rows_per;
                let end = (start + rows_per).min(rows);
                if start >= end {
                    continue;
                }
                scope.spawn(move || {
                    let mut x_row = vec![0.0f64; k];
                    let mut out_row = vec![0.0f64; k];
                    let mut max_abs = 0.0f64;
                    let mut sum_sq = 0.0f64;
                    let mut code_mis = 0usize;
                    let mut code_max_diff = 0i32;
                    let mut non_tie = 0usize;
                    let mut scale_rel = 0.0f64;
                    let mut non_finite = false;
                    for r in start..end {
                        let got_rot = &rotated[r * k..(r + 1) * k];
                        let got_codes = &codes[r * k..(r + 1) * k];
                        let got_scale = f64::from(scales[r]);
                        if !got_scale.is_finite() {
                            non_finite = true;
                        }
                        let (want_rot, want_codes_opt, want_scale) = if let (
                            Some(rr),
                            Some(rc),
                            Some(rs),
                        ) =
                            (ref_rot, ref_codes, ref_scales)
                        {
                            let idx = r % unique;
                            (
                                &rr[idx * k..(idx + 1) * k],
                                Some(&rc[idx * k..(idx + 1) * k]),
                                rs[idx],
                            )
                        } else {
                            for (dst, &v) in x_row.iter_mut().zip(&x_bf16[r * k..(r + 1) * k]) {
                                *dst = f64::from(v.to_f32());
                            }
                            let mut peak = 0.0f64;
                            for g in 0..groups {
                                let base = g * group;
                                for col in 0..group {
                                    let mut sum = 0.0f64;
                                    let hcol = &table[col * group..(col + 1) * group];
                                    for i in 0..group {
                                        sum += x_row[base + i] * hcol[i];
                                    }
                                    out_row[base + col] = sum;
                                    peak = peak.max(sum.abs());
                                }
                            }
                            (out_row.as_slice(), None, peak.max(1.0e-12) / 127.0)
                        };
                        let srel = (got_scale - want_scale).abs() / want_scale;
                        if srel > scale_rel {
                            scale_rel = srel;
                        }
                        for i in 0..k {
                            let actual = got_rot[i];
                            if !actual.is_finite() {
                                non_finite = true;
                            }
                            let err = (f64::from(actual) - want_rot[i]).abs();
                            if err > max_abs {
                                max_abs = err;
                            }
                            sum_sq += err * err;
                            let scaled = want_rot[i] / want_scale;
                            let want_q = want_codes_opt.map_or_else(
                                || scaled.round_ties_even().clamp(-127.0, 127.0) as i8,
                                |rc| rc[i],
                            );
                            if got_codes[i] != want_q {
                                code_mis += 1;
                                let d = (i32::from(got_codes[i]) - i32::from(want_q)).abs();
                                if d > code_max_diff {
                                    code_max_diff = d;
                                }
                                let tie_margin = (scaled - (scaled.floor() + 0.5)).abs();
                                if d > 1 || tie_margin > 1.0e-4 {
                                    non_tie += 1;
                                }
                            }
                        }
                    }
                    *slot = (
                        max_abs,
                        sum_sq,
                        code_mis,
                        code_max_diff,
                        non_tie,
                        scale_rel,
                        non_finite,
                    );
                });
            }
        });
        let mut rotated_max_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut code_mismatch_count = 0usize;
        let mut code_max_abs_diff = 0i32;
        let mut non_tie_code_mismatch = 0usize;
        let mut scale_max_rel_err = 0.0f64;
        for (ma, sq, cm, cd, nt, sr, nf) in partials {
            ensure!(!nf, "non-finite convrot rotated or scale value");
            rotated_max_abs = rotated_max_abs.max(ma);
            sum_sq += sq;
            code_mismatch_count += cm;
            code_max_abs_diff = code_max_abs_diff.max(cd);
            non_tie_code_mismatch += nt;
            scale_max_rel_err = scale_max_rel_err.max(sr);
        }
        let total = (rows * k) as f64;
        Ok(F64Distance {
            rotated_max_abs,
            rotated_rmse: (sum_sq / total).sqrt(),
            code_mismatch_count,
            code_mismatch_frac: code_mismatch_count as f64 / total,
            code_max_abs_diff,
            non_tie_code_mismatch,
            scale_max_rel_err,
        })
    }

    fn launch_fwht_probe(
        device: &Device,
        x_ptr: u64,
        (rows, k, group): (usize, usize, usize),
        fuse_silu: bool,
        mutant: i32,
    ) -> Result<(Vec<f32>, Vec<i8>, Vec<f32>)> {
        use crate::backend::gpu::{LaunchConfig, PushKernelArg};
        let stream = &device.stream;
        let fns = load_convrot(&device.ctx)?;
        let rotated_d = stream.alloc_zeros::<f32>(rows * k)?;
        let codes_d = stream.alloc_zeros::<i8>(rows * k)?;
        let scales_d = stream.alloc_zeros::<f32>(rows)?;
        let rows_i = i32::try_from(rows)?;
        let k_i = i32::try_from(k)?;
        let group_i = i32::try_from(group)?;
        let (function, threads, smem) = if group <= 256 {
            let chunks = k.div_ceil(256);
            let warps = chunks.div_ceil(4).clamp(1, 32);
            (
                &fns.fwht_probe[usize::from(fuse_silu)],
                u32::try_from(warps * 32)?,
                0u32,
            )
        } else {
            ensure!(!fuse_silu, "smem probe only tests non-silu");
            let need = u32::try_from(k * 4)?;
            (&fns.fwht_probe[2], 256u32, need)
        };
        unsafe {
            stream
                .launch_builder(function)
                .arg(&x_ptr)
                .arg(&rotated_d.device_ptr(stream).0)
                .arg(&codes_d.device_ptr(stream).0)
                .arg(&scales_d.device_ptr(stream).0)
                .arg(&rows_i)
                .arg(&k_i)
                .arg(&group_i)
                .arg(&mutant)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(rows)?, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: smem,
                })?;
        }
        stream.synchronize()?;
        Ok((
            stream.clone_dtoh(&rotated_d)?,
            stream.clone_dtoh(&codes_d)?,
            stream.clone_dtoh(&scales_d)?,
        ))
    }

    fn check_within_bar(label: &str, dist: F64Distance, bar: IncumbentBar) -> Result<()> {
        emit(&format!(
            "CONVROT_FWHT_F64 label={label} rot_max={:.3e} (bar={:.3e}) rot_rmse={:.3e} codes={}/frac={:.3e} (bar={}/{:.3e}) max_cdiff={} non_tie={} scale_rel={:.3e} (bar={:.3e})",
            dist.rotated_max_abs,
            bar.rotated_max_abs,
            dist.rotated_rmse,
            dist.code_mismatch_count,
            dist.code_mismatch_frac,
            bar.code_mismatch_count,
            bar.code_mismatch_frac,
            dist.code_max_abs_diff,
            dist.non_tie_code_mismatch,
            dist.scale_max_rel_err,
            bar.scale_max_rel_err,
        ));
        ensure!(
            dist.rotated_max_abs <= bar.rotated_max_abs
                && dist.code_max_abs_diff <= 1
                && dist.non_tie_code_mismatch == 0
                && dist.code_mismatch_count <= bar.code_mismatch_count
                && dist.code_mismatch_frac <= bar.code_mismatch_frac
                && dist.scale_max_rel_err <= bar.scale_max_rel_err,
            "CONVROT_FWHT_F64 {label} exceeded incumbent bar: got {dist:?} vs {bar:?}"
        );
        Ok(())
    }

    fn check_planted_mutants(
        device: &Device,
        x_bf16: &[bf16],
        (rows, k, group): (usize, usize, usize),
        bar: IncumbentBar,
    ) -> Result<()> {
        let x_d = device.stream.clone_htod(x_bf16)?;
        let x_ptr = x_d.device_ptr(&device.stream).0;
        for (mutant, kind) in [(1i32, "stage_omitted"), (2i32, "sign_flipped")] {
            let (rot, codes, scales) =
                launch_fwht_probe(device, x_ptr, (rows, k, group), false, mutant)?;
            let dist = f64_distance_bf16(x_bf16, &rot, &codes, &scales, k, group, Some(23))?;
            emit(&format!(
                "CONVROT_FWHT_MUTANT kind={kind} group={group} rot_max={:.3e} codes={} non_tie={} scale_rel={:.3e}",
                dist.rotated_max_abs,
                dist.code_mismatch_count,
                dist.non_tie_code_mismatch,
                dist.scale_max_rel_err
            ));
            ensure!(
                dist.rotated_max_abs > bar.rotated_max_abs.max(1.0e-3)
                    && dist.code_mismatch_count > bar.code_mismatch_count,
                "planted mutant {kind} group={group} did not fail the gate"
            );
        }
        Ok(())
    }

    fn check_fwht_shape(
        device: &Device,
        (rows, k, n, group): (usize, usize, usize, usize),
        bar: IncumbentBar,
    ) -> Result<()> {
        let x = activation(rows, k);
        let qweight = weight(n, k);
        let wscale = wscale_row(n);
        let ran = launch_kept(
            device,
            HostLaunch {
                x: &x,
                k,
                group,
                qweight: &qweight,
                n,
                wscale: &wscale,
                keep: true,
            },
        )?;
        let host = host_product(&ran.codes, &ran.scales, k, &qweight, n, &wscale)?;
        let prod_diff = bit_diff(&ran.y, &host);
        ensure!(
            prod_diff == 0,
            "convrot product rows={rows} k={k} n={n} group={group} differ {prod_diff}"
        );
        let x_bf16: Vec<bf16> = x.iter().copied().map(bf16::from_f32).collect();
        let x_d = device.stream.clone_htod(&x_bf16)?;
        let (rot, p_codes, p_scales) = launch_fwht_probe(
            device,
            x_d.device_ptr(&device.stream).0,
            (rows, k, group),
            false,
            0,
        )?;
        ensure!(
            count_i8(&ran.codes, &p_codes) == 0 && count_f32(&ran.scales, &p_scales) == 0,
            "production FWHT diverged from mutant=0 probe rows={rows} k={k} group={group}"
        );
        let dist = f64_distance_bf16(&x_bf16, &rot, &ran.codes, &ran.scales, k, group, Some(23))?;
        check_within_bar(&format!("syn_m{rows}_k{k}_g{group}"), dist, bar)?;
        Ok(())
    }

    fn load_bf16_file(path: &std::path::Path, expected_elems: usize) -> Result<Vec<bf16>> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let (chunks, rest) = bytes.as_chunks::<2>();
        ensure!(
            rest.is_empty() && chunks.len() == expected_elems,
            "{} len {} != expected {}",
            path.display(),
            bytes.len(),
            expected_elems * 2
        );
        Ok(chunks
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
            .collect())
    }

    fn check_real_step_inputs(device: &Device, dir: &std::path::Path) -> Result<()> {
        let stream = &device.stream;
        let fns = load_convrot(&device.ctx)?;
        let rows = 37_717usize;
        let group = 256usize;
        let x_k14336_ref = load_bf16_file(&dir.join("x_k14336.bf16"), rows * 14_336)?;
        let cases = [
            (
                "step_k5376",
                "x_k5376.bf16",
                5376usize,
                false,
                INCUMBENT_STEP_K5376_BAR,
            ),
            (
                "step_k7168",
                "x_k7168.bf16",
                7168usize,
                false,
                INCUMBENT_STEP_K7168_BAR,
            ),
            (
                "step_k14336",
                "x_k14336.bf16",
                14336usize,
                false,
                INCUMBENT_STEP_K14336_BAR,
            ),
            (
                "step_k14336_silu",
                "gu_k14336.bf16",
                14336usize,
                true,
                INCUMBENT_STEP_K14336_SILU_BAR,
            ),
        ];
        for (label, file, k, fuse_silu, bar) in cases {
            let cols = k * (1 + usize::from(fuse_silu));
            let x_raw = load_bf16_file(&dir.join(file), rows * cols)?;
            let x_d = stream.clone_htod(&x_raw)?;
            let x_ptr = x_d.device_ptr(stream).0;
            let plan = super::derive_rotate_plan(device, k, group, fuse_silu)?;
            let codes_d = stream.alloc_zeros::<i8>(rows * k)?;
            let scales_d = stream.alloc_zeros::<f32>(rows)?;
            super::launch_rotate_quant(
                stream,
                &fns,
                &ConvrotArgs {
                    x: x_ptr,
                    qweight: 0,
                    wscale: 0,
                    y: 0,
                    codes: codes_d.device_ptr(stream).0,
                    scales: scales_d.device_ptr(stream).0,
                    rows: i32::try_from(rows)?,
                    k: i32::try_from(k)?,
                    n: 0,
                    group: i32::try_from(group)?,
                    product: ProductKernel::NFast,
                    fuse_silu,
                    rotate: Some(plan),
                },
            )?;
            stream.synchronize()?;
            let prod_codes = stream.clone_dtoh(&codes_d)?;
            let prod_scales = stream.clone_dtoh(&scales_d)?;
            let (rot, p_codes, p_scales) =
                launch_fwht_probe(device, x_ptr, (rows, k, group), fuse_silu, 0)?;
            ensure!(
                count_i8(&prod_codes, &p_codes) == 0 && count_f32(&prod_scales, &p_scales) == 0,
                "production FWHT diverged from probe on {label}"
            );
            let ref_in = if fuse_silu { &x_k14336_ref } else { &x_raw };
            let dist = f64_distance_bf16(ref_in, &rot, &prod_codes, &prod_scales, k, group, None)?;
            check_within_bar(label, dist, bar)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_kernel_matches_the_definition() -> Result<()> {
        let device = Device::new(0)?;
        let group = 256usize;
        // Below 17 rows: convrot_linear stays bit-exact to the embedded definition.
        for (rows, expected) in [(1usize, &M1[..]), (16usize, &M16[..])] {
            let x = activation(rows, 256);
            let qweight = weight(2, 256);
            let wscale = wscale_row(2);
            let y = launch(&device, &x, 256, group, &qweight, 2, &wscale)?;
            let mismatched = differ(&y, expected);
            ensure!(
                mismatched == 0,
                "convrot M={rows} differing bits {mismatched}"
            );
        }
        for (rows, k, n, g) in [
            (17usize, 256usize, 2usize, 256usize),
            (20, 256, 2, 256),
            (256, 256, 2, 256),
            (17, 5376, 7168, 256),
            (17, 7168, 5376, 256),
            (17, 14336, 5376, 256),
            (20, 5376, 40, 256),
            (42, 5376, 40, 256),
            (20, 7168, 40, 1024),
        ] {
            check_fwht_shape(&device, (rows, k, n, g), INCUMBENT_SYNTHETIC_BAR)?;
        }
        // Production-scale synthetic row count + negative column-swap control.
        {
            let (rows, k, n) = (41_770usize, 5376usize, 40usize);
            let x = activation(rows, k);
            let qweight = weight(n, k);
            let wscale = wscale_row(n);
            let ran = launch_kept(
                &device,
                HostLaunch {
                    x: &x,
                    k,
                    group,
                    qweight: &qweight,
                    n,
                    wscale: &wscale,
                    keep: true,
                },
            )?;
            let host = host_product(&ran.codes, &ran.scales, k, &qweight, n, &wscale)?;
            ensure!(
                bit_diff(&ran.y, &host) == 0,
                "production synthetic product diff"
            );
            let mut swapped = x;
            swapped.swap(0, 16);
            let mutant = launch(&device, &swapped, k, group, &qweight, n, &wscale)?;
            let mut_diff = bit_diff(&mutant, &ran.y);
            emit(&format!("CONVROT_FWHT_SWAP_MUTANT differ={mut_diff}"));
            ensure!(mut_diff > 0, "column swap reproduced the device product");
        }
        // Planted mutants (stage omitted & sign flipped) on both REG (group=256) and SMEM (group=1024).
        let x256: Vec<bf16> = activation(17, 256)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        check_planted_mutants(&device, &x256, (17, 256, 256), INCUMBENT_SYNTHETIC_BAR)?;
        let x1024: Vec<bf16> = activation(20, 7168)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        check_planted_mutants(&device, &x1024, (20, 7168, 1024), INCUMBENT_SYNTHETIC_BAR)?;

        // Doubled wscale and omitted-rotation negative controls.
        let (rows, k, n) = (17usize, 256usize, 2usize);
        let x = activation(rows, k);
        let qweight = weight(n, k);
        let wscale = wscale_row(n);
        let y = launch(&device, &x, k, group, &qweight, n, &wscale)?;
        let mut shifted = wscale.clone();
        shifted[0] *= 2.0;
        let scale_off = launch(&device, &x, k, group, &qweight, n, &shifted)?;
        ensure!(
            bit_diff(&scale_off, &y) > 0,
            "doubled wscale reproduced output"
        );
        let omitted = super::linear_omitting_rotation(&x, k, &qweight, n, &wscale)?;
        ensure!(
            bit_diff(&omitted, &y) > 0,
            "omitting rotation reproduced output"
        );

        // Real-step production-scale [37717, K] tensors for K in {5376, 7168, 14336} + fused SiLU.
        if let Ok(dir) = std::env::var("FF_HYPERFLOW_CONVROT_STEP_INPUTS") {
            check_real_step_inputs(&device, std::path::Path::new(&dir))?;
        }
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
                codes: codes_d.device_ptr(stream).0,
                scales: scales_d.device_ptr(stream).0 + scale_off * 4,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: i32::try_from(n)?,
                group: 256,
                product,
                fuse_silu: false,
                rotate: None,
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
                    stages: 2,
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
        let Some(stages) = super::m_fast_stage(&device.info) else {
            ensure!(
                super::gate_up_product_order(device, [rows, 1501], (k, n))?
                    == [ProductKernel::NFast; 2],
                "gate/up M-fast enabled without stage support on {}",
                device.info.arch
            );
            return Ok(());
        };
        let m_fast_kernel = ProductKernel::MFastLdsm { stages };
        let active = super::product_active_ctas(device, m_fast_kernel)?;
        let expected_order =
            super::gate_up_products(&device.info, active, [rows, 1501], (k, n), stages);
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
                        m_fast_kernel
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
            (0, 0, m_fast_kernel),
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
    fn check_pipelined_short_k(
        device: &Device,
        max_stage: u8,
        mut make_kernel: impl FnMut(u8, usize) -> ProductKernel,
    ) -> Result<()> {
        for stages in 2u8..=max_stage {
            for (rows, k, n) in [
                (85, 64, 500),
                (85, 128, 499),
                (85, 128, 500),
                (85, 192, 500),
                (257, 384, 512),
                (1100, 384, 500),
            ] {
                let kernel = make_kernel(stages, rows);
                let (mut codes, scales, qweight, wscale) = fixture(rows, k, n);
                let reference = launch_product_at(
                    device,
                    &codes,
                    &scales,
                    &qweight,
                    &wscale,
                    k,
                    (0, 0, ProductKernel::NFast),
                )?;
                let cand = launch_product_at(
                    device,
                    &codes,
                    &scales,
                    &qweight,
                    &wscale,
                    k,
                    (0, 0, kernel),
                )?;
                let diff = bit_diff(&cand, &reference);
                ensure!(
                    diff == 0,
                    "{kernel:?} rows={rows} k={k} n={n} differs in {diff} elements"
                );
                if k == 384 {
                    codes[64..128].rotate_left(16);
                    let shifted = launch_product_at(
                        device,
                        &codes,
                        &scales,
                        &qweight,
                        &wscale,
                        k,
                        (0, 0, kernel),
                    )?;
                    let first_row = bit_diff(&shifted[..n], &reference[..n]);
                    let total = bit_diff(&shifted, &reference);
                    ensure!(
                        first_row > 0 && first_row == total,
                        "{kernel:?} wrong K panel control row={first_row} total={total}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a reserved sm >= 80 GPU window; small fixture for compute-sanitizer"]
    fn convrot_n_fast_swizzle_racecheck_small_gpu() -> Result<()> {
        let device = Device::new(0)?;
        if !matches!(
            device.info.arch,
            crate::backend::setup::DeviceArchitecture::Cuda(sm) if sm >= 80
        ) {
            return Ok(());
        }
        let max_stage = super::n_fast_swizzle_stage(&device.info, 37_717, (5376, 7168))
            .context("expected swizzled N-fast stage on sm>=80")?;
        check_pipelined_short_k(&device, max_stage, |stages, _| {
            ProductKernel::NFastSwizzled { stages }
        })
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a reserved sm80..sm90 GPU window; small fixture for compute-sanitizer"]
    fn convrot_m_fast_racecheck_small_gpu() -> Result<()> {
        let device = Device::new(0)?;
        if !matches!(
            device.info.arch,
            crate::backend::setup::DeviceArchitecture::Cuda(sm) if (80..120).contains(&sm)
        ) {
            return Ok(());
        }
        let max_stage = super::m_fast_stage(&device.info)
            .context("expected pipelined M-fast stage on sm80..90")?;
        check_pipelined_short_k(&device, max_stage, |stages, rows| {
            ProductKernel::MFastStriped {
                tiles: 1,
                stripes: u32::try_from(rows.div_ceil(128)).unwrap(),
                one_barrier: false,
                stages,
            }
        })
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a reserved sm >= 80 GPU window"]
    fn convrot_n_fast_swizzle_matches_stored_products() -> Result<()> {
        let device = Device::new(0)?;
        if !matches!(
            device.info.arch,
            crate::backend::setup::DeviceArchitecture::Cuda(sm) if sm >= 80
        ) {
            return Ok(());
        }
        let max_stage = super::n_fast_swizzle_stage(&device.info, 37_717, (5376, 7168))
            .context("expected swizzled N-fast stage on sm>=80")?;
        check_pipelined_short_k(&device, max_stage, |stages, _| {
            ProductKernel::NFastSwizzled { stages }
        })?;
        for (role, rows, k, n, salt, baseline) in [
            ("q", 37_717, 5376, 7168, 0, ProductKernel::NFastLdsm),
            ("k", 37_717, 5376, 7168, 1, ProductKernel::NFastLdsm),
            ("v", 37_717, 5376, 7168, 2, ProductKernel::NFastLdsm),
            ("out", 37_717, 7168, 5376, 0, ProductKernel::NFast),
            ("down_chunk", 18_859, 14336, 5376, 0, ProductKernel::NFast),
            ("down", 37_717, 14336, 5376, 0, ProductKernel::NFast),
        ] {
            let product = super::n_fast_product(&device, rows, (k, n))?;
            ensure!(
                product == ProductKernel::NFastSwizzled { stages: max_stage },
                "{role} selected {product:?} expected stages={max_stage}"
            );
            let (diff, moved) = check_product_tail(&device, (rows, k, n), baseline, product, salt)?;
            println!(
                "CONVROT_N_FAST_SWIZZLE role={role} stages={max_stage} rows={rows} k={k} n={n} differ={diff} wrong_tail_rows={moved}"
            );
        }
        Ok(())
    }

    fn check_gate_up_stripes(device: &Device) -> Result<()> {
        let Some(stages) = super::m_fast_stage(&device.info) else {
            return Ok(());
        };
        #[cfg(feature = "cuda")]
        if matches!(
            device.info.arch,
            crate::backend::setup::DeviceArchitecture::Cuda(sm) if (80..120).contains(&sm)
        ) {
            check_pipelined_short_k(device, stages, |s, _| ProductKernel::MFastLdsm {
                stages: s,
            })?;
            check_pipelined_short_k(device, stages, |s, r| ProductKernel::MFastStriped {
                tiles: 1,
                stripes: u32::try_from(r.div_ceil(128)).expect("stripes"),
                one_barrier: false,
                stages: s,
            })?;
        }
        let (rows, k, n) = (37_717, 5376, 28_672);
        let active = super::product_active_ctas(device, ProductKernel::MFastLdsm { stages })?;
        let expected = super::gate_up_products(&device.info, active, [rows, 0], (k, n), stages)[0];
        let [selected, _] = super::gate_up_product_order(device, [rows, 0], (k, n))?;
        ensure!(
            selected == expected,
            "gate/up stripe selection mismatch on {}: got {selected:?} expected {expected:?}",
            device.info.arch
        );
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
                stages,
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
                stages,
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
                stages,
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
        let mut products = vec![ProductKernel::NFast, ProductKernel::NFastLdsm];
        if let Some(stages) = super::m_fast_stage(&device.info) {
            products.push(ProductKernel::MFastLdsm { stages });
        }
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
                for &product in &products {
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
                        product.symbol_name_for(&device.info.arch)
                    );
                    ensure!(
                        differ == 0,
                        "convrot product kernel={} rows={rows} n={n} differ {differ}",
                        product.symbol_name_for(&device.info.arch)
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
            let mut kernels = vec![
                ("convrot_product", 54, 0, 2),
                ("convrot_product_n_fast_ldsm", 6, 12, 2),
            ];
            let mut absent = Vec::new();
            if sm == 120 {
                kernels.push(("convrot_product_m_fast", 6, 12, 2));
                absent.extend_from_slice(&[
                    "convrot_product_m_fast_s2",
                    "convrot_product_m_fast_s3",
                    "convrot_product_m_fast_s4",
                ]);
            } else if sm >= 80 {
                kernels.extend_from_slice(&[
                    ("convrot_product_m_fast_s2", 6, 12, 6),
                    ("convrot_product_m_fast_s3", 6, 12, 4),
                    ("convrot_product_m_fast_s4", 6, 12, 4),
                ]);
                absent.extend_from_slice(&[
                    "convrot_product_m_fast",
                    "convrot_product_m_fast_one_barrier",
                ]);
            } else {
                absent.extend_from_slice(&[
                    "convrot_product_m_fast",
                    "convrot_product_m_fast_one_barrier",
                    "convrot_product_m_fast_s2",
                    "convrot_product_m_fast_s3",
                    "convrot_product_m_fast_s4",
                ]);
            }
            if sm >= 80 {
                kernels.extend_from_slice(&[
                    ("convrot_product_n_fast_swizzled", 6, 12, 2),
                    ("convrot_product_n_fast_swizzled_s3", 6, 12, 2),
                    ("convrot_product_n_fast_swizzled_s4", 6, 12, 2),
                ]);
            }
            for name in absent {
                ensure!(
                    function_sass(&text, name).is_none(),
                    "sm_{sm} unexpectedly contains unreachable {name}"
                );
            }
            for (name, want_lds, want_ldsm, want_bars) in kernels {
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
                let spills = ops
                    .iter()
                    .filter(|code| {
                        **code == "STL"
                            || code.starts_with("STL.")
                            || **code == "LDL"
                            || code.starts_with("LDL.")
                    })
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
                    imma == 32 && lds == want_lds && ldsm == want_ldsm && barriers == want_bars,
                    "sm_{sm} {name} IMMA={imma} LDS={lds} LDSM={ldsm} BAR={barriers}"
                );
                if name.starts_with("convrot_product_m_fast") {
                    ensure!(
                        body.contains("SR_CTAID.Z"),
                        "sm_{sm} {name} missing SR_CTAID.Z stripe index"
                    );
                }
                if name.starts_with("convrot_product_m_fast_s") {
                    let stg128 = count("STG.E.128");
                    let lds128 = count("LDS.128");
                    ensure!(
                        stg128 == 8 && lds128 == 8,
                        "sm_{sm} {name} STG128={stg128} LDS128={lds128}"
                    );
                }
                let ldg = body.matches("LDG.E.CONSTANT").count();
                ensure!(ldg == 20, "sm_{sm} {name} scale LDG.E.CONSTANT {ldg}");
                ensure!(
                    reg <= 124 && stack == 0 && local == 0 && spills == 0,
                    "sm_{sm} {name} REG={reg} STACK={stack} LOCAL={local} SPILLS={spills}"
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
    fn convrot_fwht_sass_is_register_butterfly_and_spill_free() -> Result<()> {
        let sms = crate::backend::cuda::cubin::arches();
        ensure!(!sms.is_empty(), "convrot cubin missing");
        let kernels = [
            "convrot_fwht_quantize_1",
            "convrot_fwht_quantize_2",
            "convrot_fwht_quantize_4",
            "convrot_fwht_quantize_silu_1",
            "convrot_fwht_quantize_silu_2",
            "convrot_fwht_quantize_silu_4",
            "convrot_fwht_quantize_smem",
            "convrot_fwht_quantize_silu_smem",
        ];
        for &sm in sms {
            let bytes = crate::backend::cuda::cubin::bytes("convrot", sm)
                .with_context(|| format!("convrot cubin sm_{sm}"))?;
            let path = std::env::temp_dir()
                .join(format!("convrot-fwht-{}-{sm}.cubin", std::process::id()));
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
            for name in kernels {
                let body = function_sass(&text, name)
                    .with_context(|| format!("sm_{sm} {name} missing from cuobjdump"))?;
                let ops = sass_ops(body);
                let ldg128 = ops.iter().filter(|op| **op == "LDG.E.128").count();
                let stg64 = ops.iter().filter(|op| **op == "STG.E.64").count();
                let shfl = ops.iter().filter(|op| op.starts_with("SHFL")).count();
                let hmma = ops.iter().filter(|op| op.starts_with("HMMA")).count();
                let (reg, stack, local) = function_resources(&usage, name)
                    .with_context(|| format!("sm_{sm} {name} resources"))?;
                println!(
                    "CONVROT_FWHT_SASS sm_{sm} {name} LDG.E.128={ldg128} STG.E.64={stg64} SHFL={shfl} HMMA={hmma} REG={reg} STACK={stack} LOCAL={local}"
                );
                ensure!(
                    ldg128 >= 1 && stg64 >= 1 && shfl >= 1 && hmma == 0 && stack == 0 && local == 0,
                    "sm_{sm} {name} LDG.E.128={ldg128} STG.E.64={stg64} SHFL={shfl} HMMA={hmma} REG={reg} STACK={stack} LOCAL={local}"
                );
            }
            for removed in [
                "convrot_hadamard_table",
                "convrot_group_dot",
                "convrot_group_dot_32",
                "convrot_group_dot_silu",
                "convrot_group_dot_silu_32",
                "convrot_quantize",
            ] {
                ensure!(
                    function_sass(&text, removed).is_none(),
                    "sm_{sm} obsolete kernel {removed} still present"
                );
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
            for (name, max_vgprs) in [
                ("convrot_fwht_quantize_1", 96u32),
                ("convrot_fwht_quantize_2", 128),
                ("convrot_fwht_quantize_4", 160),
                ("convrot_fwht_quantize_silu_1", 96),
                ("convrot_fwht_quantize_silu_2", 128),
                ("convrot_fwht_quantize_silu_4", 160),
                ("convrot_fwht_quantize_smem", 96),
                ("convrot_fwht_quantize_silu_smem", 96),
            ] {
                let body = rocm_kernel_disasm(&disasm_text, name)
                    .with_context(|| format!("{arch} {name} missing from llvm-objdump"))?;
                let bf16 = body.matches("v_wmma_f32_16x16x16_bf16").count();
                let scratch =
                    body.matches("scratch_load").count() + body.matches("scratch_store").count();
                let (vgprs, priv_bytes) = rocm_kernel_metadata(&notes_text, name)?;
                println!(
                    "CONVROT_FWHT_ISA {arch} {name} BF16_WMMA={bf16} VGPR={vgprs} PRIV={priv_bytes} SCRATCH={scratch}"
                );
                ensure!(
                    bf16 == 0 && vgprs <= max_vgprs && priv_bytes == 0 && scratch == 0,
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
        other: &Ran,
        (rows, k, n): (usize, usize, usize),
        label: &str,
    ) -> Result<()> {
        let codes = count_i8(&base.codes, &other.codes);
        let scales = count_f32(&base.scales, &other.scales);
        let y = bit_diff(&base.y, &other.y);
        println!(
            "CONVROT_FWHT_PARITY label={label} rows={rows} k={k} n={n} codes={codes} scales={scales} y={y}"
        );
        ensure!(
            codes == 0 && scales == 0 && y == 0,
            "ConvRot FWHT {label} differs rows={rows} k={k} codes={codes} scales={scales} y={y}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn convrot_fwht_variants_and_smem_agree() -> Result<()> {
        let device = Device::new(0)?;
        for (rows, k, n, group) in [
            (17usize, 256usize, 40usize, 256usize),
            (17, 256, 40, 64),
            (1501, 5376, 40, 256),
            (37_717, 5376, 40, 256),
            (37_717, 7168, 40, 256),
            (1501, 14_336, 40, 256),
        ] {
            let x = activation(rows, k);
            let qweight = weight(n, k);
            let wscale = wscale_row(n);
            let input = HostLaunch {
                x: &x,
                k,
                group,
                qweight: &qweight,
                n,
                wscale: &wscale,
                keep: true,
            };
            let derived = super::derive_rotate_plan(&device, k, group, false)?;
            let base = launch_kept_plan(&device, input, derived)?;
            let chunks = k.div_ceil(256);
            for chunks_per_warp in [1u8, 2, 4] {
                let warps = chunks.div_ceil(usize::from(chunks_per_warp));
                if warps > 32 {
                    continue;
                }
                let plan = super::RotatePlan::Register {
                    chunks_per_warp,
                    threads: u32::try_from(warps * 32)?,
                };
                if plan == derived {
                    continue;
                }
                let alt = launch_kept_plan(&device, input, plan)?;
                check_rotation_bits(
                    &base,
                    &alt,
                    (rows, k, n),
                    &format!("reg_c{chunks_per_warp}"),
                )?;
            }
            let smem_need = u32::try_from(k * 4)?;
            if smem_need <= super::fwht_max_dynamic_smem(device.info.optin_shared_bytes)? {
                let smem_plan = super::RotatePlan::Shared {
                    threads: 256,
                    smem_bytes: smem_need,
                };
                let alt = launch_kept_plan(&device, input, smem_plan)?;
                check_rotation_bits(&base, &alt, (rows, k, n), "smem")?;
            }
        }
        ensure!(
            super::derive_rotate_plan(&device, 5376, 128, false).is_err(),
            "derive_rotate_plan accepted non-power-of-4 group 128"
        );
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
        let large_fused = device.ctx.mem_get_info()?.1 >= 32 << 30;
        let (k, n, group) = (14336usize, 5376usize, 256usize);
        let plan_plain = super::derive_rotate_plan(&device, k, group, false)?;
        let plan_silu = super::derive_rotate_plan(&device, k, group, true)?;
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

            let run = |x, fuse_silu, plan| -> Result<Ran> {
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
                        codes: codes.device_ptr(stream).0,
                        scales: scales.device_ptr(stream).0,
                        rows: rows_i,
                        k: k_i,
                        n: i32::try_from(n)?,
                        group: i32::try_from(group)?,
                        product: ProductKernel::NFast,
                        fuse_silu,
                        rotate: Some(plan),
                    },
                )?;
                stream.synchronize()?;
                let ran = Ran {
                    y: stream.clone_dtoh(&y)?,
                    codes: stream.clone_dtoh(&codes)?,
                    scales: stream.clone_dtoh(&scales)?,
                };
                ensure!(
                    ran.codes[rows * k..].iter().all(|&v| v == 0)
                        && ran.scales[rows].to_bits() == 0
                        && ran.y[rows * n..].iter().all(|v| v.to_bits() == 0),
                    "ConvRot SiLU tail guard rows={rows} fused={fuse_silu}"
                );
                Ok(ran)
            };

            let separate = run(gap, false, plan_plain)?;
            let fused = run(gup, true, plan_silu)?;
            let codes = count_i8(&separate.codes, &fused.codes);
            let scales = count_f32(&separate.scales, &fused.scales);
            let y = bit_diff(&separate.y, &fused.y);
            println!("CONVROT_SILU rows={rows} k={k} n={n} codes={codes} scales={scales} y={y}");
            ensure!(
                codes == 0 && scales == 0 && y == 0,
                "fused SiLU rows={rows} differs"
            );

            if rows == 1501 {
                let actual = stream.clone_dtoh(&gated.slice(control..control + 1))?;
                ensure!(
                    actual == [bf16::from_f32(1.6484375)],
                    "SiLU control two-round value"
                );
                let (sep_rot, sep_p_codes, sep_p_scales) =
                    launch_fwht_probe(&device, gap, (rows, k, group), false, 0)?;
                ensure!(
                    count_i8(&separate.codes[..rows * k], &sep_p_codes) == 0
                        && count_f32(&separate.scales[..rows], &sep_p_scales) == 0,
                    "SiLU control probe diverged from production"
                );
                stream.memcpy_htod(
                    &[bf16::from_f32(1.65625)],
                    &mut gated.slice_mut(control..control + 1),
                )?;
                let (mut_rot, _, _) = launch_fwht_probe(&device, gap, (rows, k, group), false, 0)?;
                let differing = count_f32(&sep_rot, &mut_rot);
                let target = control..control + group;
                let inside = count_f32(&sep_rot[target.clone()], &mut_rot[target]);
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
        let codes = stream.alloc_zeros::<i8>(rows * k)?;
        let scales = stream.alloc_zeros::<f32>(rows)?;
        let rotate = if rows >= super::TENSOR_CORE_ROWS {
            Some(super::derive_rotate_plan(device, k, group, false)?)
        } else {
            None
        };
        launch_device(
            stream,
            fns,
            &ConvrotArgs {
                x: held.ptr,
                qweight: weight_d.device_ptr(stream).0,
                wscale: scale_d.device_ptr(stream).0,
                y: y_d.device_ptr(stream).0,
                codes: codes.device_ptr(stream).0,
                scales: scales.device_ptr(stream).0,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: i32::try_from(n)?,
                group: i32::try_from(group)?,
                product: ProductKernel::NFast,
                fuse_silu: false,
                rotate,
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
        let codes_d = stream.alloc_zeros::<i8>(rows * k)?;
        let scales_d = stream.alloc_zeros::<f32>(rows)?;
        let rotate = super::derive_rotate_plan(device, k, group, false)?;
        super::launch_rotate_quant(
            stream,
            fns,
            &ConvrotArgs {
                x: held.ptr,
                qweight: 0,
                wscale: 0,
                y: 0,
                codes: codes_d.device_ptr(stream).0,
                scales: scales_d.device_ptr(stream).0,
                rows: i32::try_from(rows)?,
                k: i32::try_from(k)?,
                n: 0,
                group: i32::try_from(group)?,
                product: ProductKernel::NFast,
                fuse_silu: false,
                rotate: Some(rotate),
            },
        )?;
        stream.synchronize()?;
        held.release();
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
                    codes: codes_d.device_ptr(stream).0,
                    scales: scales_d.device_ptr(stream).0,
                    rows: i32::try_from(rows)?,
                    k: i32::try_from(k)?,
                    n: i32::try_from(n)?,
                    group: i32::try_from(group)?,
                    product,
                    fuse_silu: false,
                    rotate: None,
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
        let expected_qkv =
            if let Some(stages) = super::n_fast_swizzle_stage(&device.info, 37_717, (k, n)) {
                ProductKernel::NFastSwizzled { stages }
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
        if matches!(expected_qkv, ProductKernel::NFastSwizzled { .. }) {
            check_shared(&device, &fns, 37_717, k, n, expected_qkv, true)?;
        }
        Ok(())
    }
}
