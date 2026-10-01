//! MMA GEMM launches (cuda/mma.cu in ff-cuda) for batched prefill
//! projections: y[t][r] = x[t] · w[r] with f32 accumulation, one launch
//! per projection. Weight policies: raw bf16/f16 bytes end to end (x
//! staged 16-bit first); int4 affine with the bias term folded into the
//! kernel's K-tile epilogue (prefill-design §2.1).

use anyhow::{Context, Result};
use cudarc::driver::DevicePtrMut;
use cudarc::driver::safe::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ff_core::quant::QuantFormat;
use ff_edge0::gpu::GpuContext;
use std::sync::Arc;

pub struct MmaKernels {
    bf16: CudaFunction,
    f16: CudaFunction,
    int4: CudaFunction,
    group_sums: CudaFunction,
    x_to_bf16: CudaFunction,
    x_to_f16: CudaFunction,
}

impl MmaKernels {
    pub fn load(ctx: &GpuContext) -> Result<Self> {
        let module = ff_edge0::kernel_assets::load_module(&ctx.context, &crate::kernel_assets::MMA)
            .context("mma module load failed")?;
        Ok(Self {
            bf16: module
                .load_function("bf16_mma")
                .context("bf16_mma missing")?,
            f16: module.load_function("f16_mma").context("f16_mma missing")?,
            int4: module
                .load_function("int4_mma")
                .context("int4_mma missing")?,
            group_sums: module
                .load_function("int4_group_sums")
                .context("int4_group_sums missing")?,
            x_to_bf16: module
                .load_function("x_to_bf16")
                .context("x_to_bf16 missing")?,
            x_to_f16: module
                .load_function("x_to_f16")
                .context("x_to_f16 missing")?,
        })
    }

    /// Convert x f32 to the policy's 16-bit staging buffer (cp.async
    /// stages raw bytes, so the conversion happens here, once per
    /// projection, with the same rounding the v1 in-kernel staging used).
    pub fn stage_x(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        xb: &mut CudaSlice<u8>,
        format: QuantFormat,
    ) -> Result<()> {
        let total = x.len() as i64;
        anyhow::ensure!(xb.len() >= x.len() * 2, "mma: short x staging buffer");
        let (xb_ptr, _guard) = xb.device_ptr_mut(stream);
        let kernel = match format {
            QuantFormat::GroupAffine { .. } | QuantFormat::Bf16 => &self.x_to_bf16,
            QuantFormat::F16 => &self.x_to_f16,
            QuantFormat::BlockFp8 { .. } => {
                anyhow::bail!("mma projection launched for an FP8-block checkpoint")
            }
        };
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(x)
                .arg(&xb_ptr)
                .arg(&total)
                .launch(LaunchConfig {
                    grid_dim: ((x.len() as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("x_to launch failed")?;
        Ok(())
    }

    /// xsum[g][t] over x [tokens, in] (the int4 bias path): the
    /// deleted kernel's body and order, verbatim.
    pub fn group_sums(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        xs: &mut CudaSlice<f32>,
        in_dim: usize,
        tokens: usize,
    ) -> Result<()> {
        let groups = in_dim / 64;
        let groups_i = groups as i32;
        unsafe {
            stream
                .launch_builder(&self.group_sums)
                .arg(x)
                .arg(xs)
                .arg(&(in_dim as i32))
                .arg(&groups_i)
                .launch(LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("int4_group_sums launch failed")?;
        Ok(())
    }

    /// y = x · dequant(w)ᵀ for one int4 projection; the per-group bias
    /// correction folds into the kernel from xs and the bf16 biases.
    pub fn project_int4(
        &self,
        stream: &Arc<CudaStream>,
        args: &Int4Args<'_>,
        y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
    ) -> Result<()> {
        let in_dim = args.in_dim;
        anyhow::ensure!(
            in_dim.is_multiple_of(64),
            "mma int4: in_dim {in_dim} not a multiple of 64"
        );
        anyhow::ensure!(
            args.rows > 0
                && args.tokens > 0
                && args.xb.len() >= args.tokens * in_dim * 2
                && y.len() >= args.tokens * args.rows,
            "mma int4: short operand buffers"
        );
        let (y_ptr, _y_guard) = y.device_ptr_mut(stream);
        let rows_i = args.rows as i32;
        let in_i = in_dim as i32;
        let tokens_i = args.tokens as i32;
        unsafe {
            stream
                .launch_builder(&self.int4)
                .arg(args.packed)
                .arg(args.scales)
                .arg(args.biases)
                .arg(args.xs)
                .arg(args.xb)
                .arg(&y_ptr)
                .arg(&rows_i)
                .arg(&in_i)
                .arg(&tokens_i)
                .launch(LaunchConfig {
                    grid_dim: (
                        (args.tokens).div_ceil(crate::kernel_assets::BT) as u32,
                        (args.rows).div_ceil(crate::kernel_assets::BM) as u32,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("int4_mma launch failed")?;
        Ok(())
    }

    /// y = x · wᵀ for one projection: w raw bf16 [rows, in], x staged
    /// 16-bit [tokens, in], y f32 [tokens, rows].
    pub fn project_bf16(
        &self,
        stream: &Arc<CudaStream>,
        args: &RawArgs<'_>,
        y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
    ) -> Result<()> {
        let (w, x, rows, in_dim, tokens) = (args.w, args.x, args.rows, args.in_dim, args.tokens);
        anyhow::ensure!(
            in_dim.is_multiple_of(64),
            "mma: in_dim {in_dim} not a multiple of 64 (BK)"
        );
        anyhow::ensure!(
            rows > 0 && tokens > 0 && x.len() >= tokens * in_dim * 2 && y.len() >= tokens * rows,
            "mma: short operand buffers"
        );
        let (y_ptr, _y_guard) = y.device_ptr_mut(stream);
        let rows_i = rows as i32;
        let in_i = in_dim as i32;
        let tokens_i = tokens as i32;
        unsafe {
            stream
                .launch_builder(&self.bf16)
                .arg(w)
                .arg(x)
                .arg(&y_ptr)
                .arg(&rows_i)
                .arg(&in_i)
                .arg(&tokens_i)
                .launch(LaunchConfig {
                    grid_dim: (
                        tokens.div_ceil(crate::kernel_assets::BT) as u32,
                        rows.div_ceil(crate::kernel_assets::BM) as u32,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("bf16_mma launch failed")?;
        Ok(())
    }

    /// y = x · wᵀ for one projection: w raw f16 [rows, in], x staged
    /// 16-bit [tokens, in], y f32 [tokens, rows].
    pub fn project_f16(
        &self,
        stream: &Arc<CudaStream>,
        args: &RawArgs<'_>,
        y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
    ) -> Result<()> {
        let (w, x, rows, in_dim, tokens) = (args.w, args.x, args.rows, args.in_dim, args.tokens);
        anyhow::ensure!(
            in_dim.is_multiple_of(64),
            "mma: in_dim {in_dim} not a multiple of 64 (BK)"
        );
        anyhow::ensure!(
            rows > 0 && tokens > 0 && x.len() >= tokens * in_dim * 2 && y.len() >= tokens * rows,
            "mma: short operand buffers"
        );
        let (y_ptr, _y_guard) = y.device_ptr_mut(stream);
        let rows_i = rows as i32;
        let in_i = in_dim as i32;
        let tokens_i = tokens as i32;
        unsafe {
            stream
                .launch_builder(&self.f16)
                .arg(w)
                .arg(x)
                .arg(&y_ptr)
                .arg(&rows_i)
                .arg(&in_i)
                .arg(&tokens_i)
                .launch(LaunchConfig {
                    grid_dim: (
                        tokens.div_ceil(crate::kernel_assets::BT) as u32,
                        rows.div_ceil(crate::kernel_assets::BM) as u32,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("f16_mma launch failed")?;
        Ok(())
    }
}

/// One raw 16-bit projection's tensors and geometry; x is the staged
/// 16-bit conversion of the activation slab (see stage_x).
pub struct RawArgs<'a> {
    pub w: &'a CudaSlice<u8>,
    pub x: &'a CudaSlice<u8>,
    pub rows: usize,
    pub in_dim: usize,
    pub tokens: usize,
}

/// One int4 projection's tensors and geometry; xb is the staged 16-bit
/// activation slab and xs the precomputed per-group f32 sums.
pub struct Int4Args<'a> {
    pub packed: &'a CudaSlice<u32>,
    pub scales: &'a CudaSlice<u16>,
    pub biases: &'a CudaSlice<u16>,
    pub xs: &'a CudaSlice<f32>,
    pub xb: &'a CudaSlice<u8>,
    pub rows: usize,
    pub in_dim: usize,
    pub tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::INT4;
    use anyhow::Result;

    #[test]
    #[ignore = "requires a CUDA device"]
    fn mma_bf16_matches_host_matvec() -> Result<()> {
        let ctx = GpuContext::new(0)?;
        let kernels = MmaKernels::load(&ctx)?;
        let (rows, in_dim, tokens) = (512usize, 512usize, 70usize);
        let w_bytes: Vec<u8> = (0..rows * in_dim)
            .flat_map(|i| half::bf16::from_f32(((i as f32) * 0.011).sin() * 0.05).to_le_bytes())
            .collect();
        let w_host: Vec<f32> = w_bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u32::from(c[0]) | (u32::from(c[1]) << 8)) << 16))
            .collect();
        let x_host: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        // The MMA stages x as bf16; the reference rounds the same way.
        let x_rounded: Vec<f32> = x_host
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let w = ctx.stream.clone_htod(&w_bytes)?;
        let x = ctx.upload_f32(&x_host)?;
        let mut xb = ctx.stream.alloc_zeros::<u8>(tokens * in_dim * 2)?;
        kernels.stage_x(&ctx.stream, &x, &mut xb, QuantFormat::Bf16)?;
        let mut y = ctx.stream.alloc_zeros::<f32>(tokens * rows)?;
        kernels.project_bf16(
            &ctx.stream,
            &RawArgs {
                w: &w,
                x: &xb,
                rows,
                in_dim,
                tokens,
            },
            &mut y,
        )?;
        let got = ctx.dtoh(&y)?;
        let mut max_abs = 0f32;
        let mut expect = vec![0f32; tokens * rows];
        for t in 0..tokens {
            for r in 0..rows {
                let mut acc = 0f32;
                for k in 0..in_dim {
                    acc += w_host[r * in_dim + k] * x_rounded[t * in_dim + k];
                }
                expect[t * rows + r] = acc;
                max_abs = max_abs.max((acc - got[t * rows + r]).abs());
            }
        }
        assert!(max_abs <= 0.01, "mma bf16 max_abs {max_abs}");
        Ok(())
    }

    fn small_int4(ctx: &GpuContext, kernels: &MmaKernels) -> Result<()> {
        let (rows, in_dim, tokens) = (512usize, 512usize, 70usize);
        let packed: Vec<u32> = (0..rows * in_dim / 8)
            .map(|i| (i as u32).wrapping_mul(2654435761))
            .collect();
        let groups = in_dim / 64;
        let scales: Vec<f32> = (0..rows * groups)
            .map(|i| 0.01 + (i % 7) as f32 * 0.003)
            .collect();
        let biases: Vec<f32> = scales.iter().map(|s| -7.0 * s).collect();
        let quant = ff_edge0::int4::GroupQuant::new(packed, scales, biases, rows, in_dim, 4)?;
        let x_host: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5)
            .collect();
        let gpu_q = ctx.upload(&quant, None)?;
        let (packed, scales, biases) = gpu_q.tensors();
        let x = ctx.upload_f32(&x_host)?;
        let mut xb = ctx.stream.alloc_zeros::<u8>(tokens * in_dim * 2)?;
        kernels.stage_x(&ctx.stream, &x, &mut xb, INT4)?;
        let mut xs = ctx.stream.alloc_zeros::<f32>(tokens * (in_dim / 64))?;
        kernels.group_sums(&ctx.stream, &x, &mut xs, in_dim, tokens)?;
        let mut y = ctx.stream.alloc_zeros::<f32>(tokens * rows)?;
        kernels.project_int4(
            &ctx.stream,
            &Int4Args {
                packed,
                scales,
                biases,
                xs: &xs,
                xb: &xb,
                rows,
                in_dim,
                tokens,
            },
            &mut y,
        )?;
        let got = ctx.dtoh(&y)?;
        let mut max_abs = 0f32;
        for t in 0..tokens {
            let expect = quant.matvec(&x_host[t * in_dim..(t + 1) * in_dim], None);
            for (r, e) in expect.iter().enumerate() {
                max_abs = max_abs.max((e - got[t * rows + r]).abs());
            }
        }
        assert!(max_abs <= 0.05, "mma int4 max_abs {max_abs}");
        Ok(())
    }

    /// Worst relative error of int4_mma against fp64 on the production
    /// projection shapes, plus a negative control: perturbing one bias
    /// must move the worst error, proving the harness sees the folded
    /// term. The pre-change path reports from the same harness in a
    /// pre-change tree.
    ///
    /// Row r of token t has one reference for every token count that
    /// includes it, so the reference is computed once per shape at the
    /// largest token count, for sampled rows (tile edges, a stride
    /// across the full range, the perturbed row) and sampled tokens.
    /// The perturbed arm differs only on the perturbed row and group,
    /// where its reference is the clean one plus the bias delta times
    /// that group's reference sum.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn int4_mma_fp64_relative_error() -> Result<()> {
        let ctx = GpuContext::new(0)?;
        let kernels = MmaKernels::load(&ctx)?;
        small_int4(&ctx, &kernels)?;
        let shapes: [(&str, usize, usize); 9] = [
            ("small", 512, 512),
            ("qkvz", 10240, 5120),
            ("ba", 96, 5120),
            ("gate_up", 34816, 5120),
            ("down", 5120, 17408),
            ("o", 5120, 6144),
            ("q", 6144, 5120),
            ("kv", 2048, 5120),
            ("lm_head", 248320, 5120),
        ];
        let tokens_set = [1usize, 31, 32, 33, 128, 257, 1024, 1025];
        let tokens_max = 1025usize;
        let t_sample = [0usize, 1, 512, 1023, 1024, 15, 16, 31, 32, 63, 64];
        for (name, rows, in_dim) in shapes {
            let groups = in_dim / 64;
            let packed: Vec<u32> = (0..rows * in_dim / 8)
                .map(|i| (i as u32).wrapping_mul(2654435761))
                .collect();
            let scales: Vec<f32> = (0..rows * groups)
                .map(|i| 0.01 + (i % 11) as f32 * 0.004)
                .collect();
            let biases: Vec<f32> = scales.iter().map(|s| -7.0 * s).collect();
            let x_host: Vec<f32> = (0..tokens_max * in_dim)
                .map(|i| ((i as f32) * 0.017).cos() * 0.5)
                .collect();
            // The kernel consumes bf16 x; the reference rounds the same
            // way so the harness isolates arithmetic error.
            let x_rounded: Vec<f32> = x_host
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_f32())
                .collect();
            let scales_bf: Vec<f64> = scales
                .iter()
                .map(|&s| half::bf16::from_f32(s).to_f32() as f64)
                .collect();
            let biases_bf: Vec<f64> = biases
                .iter()
                .map(|&s| half::bf16::from_f32(s).to_f32() as f64)
                .collect();
            let sumx_ref: Vec<Vec<f64>> = (0..tokens_max)
                .map(|t| {
                    let xv = &x_rounded[t * in_dim..(t + 1) * in_dim];
                    (0..groups)
                        .map(|g| {
                            xv[g * 64..g * 64 + 64]
                                .iter()
                                .map(|&v| v as f64)
                                .sum::<f64>()
                        })
                        .collect()
                })
                .collect();
            let pert_row = rows / 2;
            let pert_group = 3;
            let mut bp = biases.clone();
            bp[pert_row * groups + pert_group] *= 1.5;
            let bias_delta = (bp[pert_row * groups + pert_group]
                - biases[pert_row * groups + pert_group]) as f64;
            let quant_clean = ff_edge0::int4::GroupQuant::new(
                packed.clone(),
                scales.clone(),
                biases.clone(),
                rows,
                in_dim,
                4,
            )?;
            let quant_pert = ff_edge0::int4::GroupQuant::new(
                packed.clone(),
                scales.clone(),
                bp,
                rows,
                in_dim,
                4,
            )?;
            let handles = [
                ctx.upload(&quant_clean, None)?,
                ctx.upload(&quant_pert, None)?,
            ];
            let x = ctx.upload_f32(&x_host)?;
            let mut xb = ctx.stream.alloc_zeros::<u8>(tokens_max * in_dim * 2)?;
            let mut xs = ctx.stream.alloc_zeros::<f32>(tokens_max * groups)?;
            let mut y = ctx.stream.alloc_zeros::<f32>(tokens_max * rows)?;

            // Sampled rows: tile edges, a stride across the full range,
            // and the perturbed row.
            let mut sample: Vec<usize> = (0..16)
                .chain(rows.saturating_sub(16)..rows)
                .chain((0..48).map(|i| (i * 137 + 7) % rows))
                .chain([pert_row])
                .collect();
            sample.sort_unstable();
            sample.dedup();
            let t_check: Vec<usize> = t_sample
                .iter()
                .copied()
                .filter(|t| *t < tokens_max)
                .collect();

            // Reference once per shape, per sampled (t, r), with the
            // magnitude of its terms for a cancellation-aware error
            // denominator.
            let mut clean_ref = vec![0f64; t_check.len() * sample.len()];
            let mut ref_mag = vec![0f64; t_check.len() * sample.len()];
            for (ti, &t) in t_check.iter().enumerate() {
                let xv = &x_rounded[t * in_dim..(t + 1) * in_dim];
                for (ri, &r) in sample.iter().enumerate() {
                    let row_words = &packed[r * (in_dim / 8)..(r + 1) * (in_dim / 8)];
                    let mut acc = 0f64;
                    let mut mag = 0f64;
                    for (g, sxg) in sumx_ref[t].iter().enumerate() {
                        let mut dot = 0f64;
                        let xg = &xv[g * 64..g * 64 + 64];
                        let codes = &row_words[g * 8..g * 8 + 8];
                        for (wi, &word) in codes.iter().enumerate() {
                            for (j, &xk) in xg[wi * 8..wi * 8 + 8].iter().enumerate() {
                                let code = ((word >> (j * 4)) & 0xF) as f64;
                                let w_k =
                                    half::bf16::from_f32((scales_bf[r * groups + g] * code) as f32)
                                        .to_f32() as f64;
                                dot += w_k * xk as f64;
                            }
                        }
                        let term = biases_bf[r * groups + g] * sxg;
                        acc += dot + term;
                        mag += dot.abs() + term.abs();
                    }
                    clean_ref[ti * sample.len() + ri] = acc;
                    ref_mag[ti * sample.len() + ri] = mag;
                }
            }

            let mut worsts = [0f64; 2];
            let mut pert_jump_min = f64::INFINITY;
            for (arm, gpu_q) in handles.iter().enumerate() {
                let (packed_d, scales_d, biases_d) = gpu_q.tensors();
                let mut worst = 0f64;
                for tokens in tokens_set {
                    kernels.stage_x(&ctx.stream, &x, &mut xb, INT4)?;
                    kernels.group_sums(&ctx.stream, &x, &mut xs, in_dim, tokens)?;
                    kernels.project_int4(
                        &ctx.stream,
                        &Int4Args {
                            packed: packed_d,
                            scales: scales_d,
                            biases: biases_d,
                            xs: &xs,
                            xb: &xb,
                            rows,
                            in_dim,
                            tokens,
                        },
                        &mut y,
                    )?;
                    let got = ctx.dtoh(&y)?;
                    let pi = sample.iter().position(|r| *r == pert_row).unwrap();
                    for (ti, &t) in t_check.iter().enumerate() {
                        if t >= tokens {
                            continue;
                        }
                        let ci = ti * sample.len();
                        for (ri, &r) in sample.iter().enumerate() {
                            let mag_of = ref_mag[ci + ri];
                            let want = if arm == 1 && r == pert_row {
                                clean_ref[ci + ri] + bias_delta * sumx_ref[t][pert_group]
                            } else {
                                clean_ref[ci + ri]
                            };
                            let delta = (want - got[t * rows + r] as f64).abs();
                            worst = worst.max(delta / (want.abs() + mag_of).max(1e-9));
                        }
                        if arm == 1 {
                            let jump = (got[t * rows + pert_row] as f64 - clean_ref[ci + pi]).abs();
                            pert_jump_min = pert_jump_min.min(jump);
                        }
                    }
                }
                worsts[arm] = worst;
            }
            let clean = worsts[0];
            let perturbed = worsts[1];
            println!(
                "FP64 {name} [{rows},{in_dim}]: worst_rel {clean:.3e}, perturbed_bias {perturbed:.3e} (rows sampled {})",
                sample.len()
            );
            assert!(
                pert_jump_min > 20.0 * clean,
                "{name}: the perturbed bias moved row {pert_row} by only {pert_jump_min:.3e} against a clean worst of {clean:.3e} — the harness cannot see the folded term"
            );
            assert!(clean < 1e-2, "{name}: worst rel {clean:.3e}");
        }
        Ok(())
    }
}
