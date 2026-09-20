//! Wide-shape int4 GEMV launches (cuda/wide_gemv.cu): split-K for the
//! short-row-count wide projections where the in-4096 chunk primitive is
//! occupancy-starved (cold 161 GB/s on down [5120,17408] vs 456 on
//! lm_head-shaped chunks — wide_bw).

use anyhow::{Context, Result};
use cudarc::driver::safe::{CudaSlice, LaunchConfig, PushKernelArg};
use ff_edge0::gpu::GpuContext;

/// MUST match #define RPB in cuda/wide_gemv.cu (the launcher's grid
/// decomposition and the kernel's block->(slice,row) mapping must agree —
/// a mismatch silently drops slices).
pub const WIDE_RPB: usize = 16;

pub struct WideKernels {
    splitk: cudarc::driver::safe::CudaFunction,
    splitk_v4: cudarc::driver::safe::CudaFunction,
    combine: cudarc::driver::safe::CudaFunction,
    group: cudarc::driver::safe::CudaFunction,
    group_v4: cudarc::driver::safe::CudaFunction,
}

impl WideKernels {
    pub fn load(ctx: &GpuContext) -> Result<Self> {
        let module =
            ff_edge0::kernel_assets::load_module(&ctx.context, &crate::kernel_assets::WIDE_GEMV)
                .context("wide_gemv module load failed")?;
        Ok(Self {
            splitk: module
                .load_function("edge0_wide_gemv4_splitk")
                .context("edge0_wide_gemv4_splitk missing")?,
            splitk_v4: module
                .load_function("edge0_wide_gemv4_splitk_v4")
                .context("edge0_wide_gemv4_splitk_v4 missing")?,
            combine: module
                .load_function("edge0_wide_combine")
                .context("edge0_wide_combine missing")?,
            group: module
                .load_function("edge0_wide_group4")
                .context("edge0_wide_group4 missing")?,
            group_v4: module
                .load_function("edge0_wide_group4_v4")
                .context("edge0_wide_group4_v4 missing")?,
        })
    }

    /// y = quant(x) (column A) and yb = quant(xb) (column B, when
    /// cols == 2). scratch/scratch_b are `split * rows` f32 each. For
    /// cols == 1 pass x/y/scratch as the b-arguments too — the kernel
    /// only takes the B path when gridDim.y == 2, so the aliases are
    /// never dereferenced.
    #[allow(clippy::too_many_arguments)]
    pub fn down(
        &self,
        ctx: &GpuContext,
        packed: &CudaSlice<u32>,
        scales: &CudaSlice<f32>,
        biases: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        yb: &CudaSlice<f32>,
        scratch: &CudaSlice<f32>,
        scratch_b: &CudaSlice<f32>,
        rows: usize,
        in_dim: usize,
        split: usize,
        cols: u32,
    ) -> Result<()> {
        // The splitk row loop is #pragma-unrolled with no tail guard
        // (guarded variants miscompiled) — rows must be a full RPB multiple.
        anyhow::ensure!(
            rows.is_multiple_of(WIDE_RPB),
            "wide splitk: rows {rows} not a multiple of {WIDE_RPB} — the \
             unrolled row loop would read out of bounds"
        );
        let rows_i = rows as i32;
        let in_i = in_dim as i32;
        let split_i = split as i32;
        let blocks = rows.div_ceil(WIDE_RPB) * split;
        let stream = ctx.stream.clone();
        // uint4 variant when slices are 4-word aligned (all qwen shapes).
        let words = in_dim / 8;
        if words.is_multiple_of(4) && (words / split).is_multiple_of(4) && words / split / 4 <= 256
        {
            unsafe {
                stream
                    .launch_builder(&self.splitk_v4)
                    .arg(packed)
                    .arg(scales)
                    .arg(biases)
                    .arg(x)
                    .arg(xb)
                    .arg(scratch)
                    .arg(scratch_b)
                    .arg(&rows_i)
                    .arg(&in_i)
                    .arg(&split_i)
                    .launch(LaunchConfig {
                        grid_dim: (cols, blocks as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .context("splitk_v4 launch failed")?;
            unsafe {
                stream
                    .launch_builder(&self.combine)
                    .arg(scratch)
                    .arg(scratch_b)
                    .arg(y)
                    .arg(yb)
                    .arg(&rows_i)
                    .arg(&split_i)
                    .launch(LaunchConfig {
                        grid_dim: (cols, (rows as u32).div_ceil(256), 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .context("combine launch failed")?;
            return Ok(());
        }
        unsafe {
            stream
                .launch_builder(&self.splitk)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(x)
                .arg(xb)
                .arg(scratch)
                .arg(scratch_b)
                .arg(&rows_i)
                .arg(&in_i)
                .arg(&split_i)
                .launch(LaunchConfig {
                    grid_dim: (cols, blocks as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("splitk launch failed")?;
        unsafe {
            stream
                .launch_builder(&self.combine)
                .arg(scratch)
                .arg(scratch_b)
                .arg(y)
                .arg(yb)
                .arg(&rows_i)
                .arg(&split_i)
                .launch(LaunchConfig {
                    grid_dim: (cols, (rows as u32).div_ceil(256), 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("combine launch failed")?;
        Ok(())
    }
}

/// One wide grouped-GEMV launch over up to four same-x segments, full
/// width (in_dim <= 6144), no K-split. Segments carry no LoRA (qwen35
/// ships none); GroupSeg reuses the ff-edge0 type for call-site parity.
impl WideKernels {
    /// y_i = quant_i(x) (column A) and yb_i = quant_i(xb) (column B when
    /// cols == 2). yb entries are per-segment B outputs; for cols == 1
    /// pass the A slices again (aliases are never dereferenced at
    /// gridDim.y == 1).
    #[allow(clippy::too_many_arguments)]
    pub fn group(
        &self,
        ctx: &GpuContext,
        segs: &[ff_edge0::gpu::GroupSeg; 4],
        yb: [&CudaSlice<f32>; 4],
        x: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
        in_dim: usize,
        cols: u32,
    ) -> Result<()> {
        let [s0, s1, s2, s3] = [&segs[0], &segs[1], &segs[2], &segs[3]];
        for s in [s0, s1, s2, s3] {
            anyhow::ensure!(s.lora.is_none(), "wide group kernel has no lora path");
        }
        let r: [i32; 4] = [
            s0.rows as i32,
            s1.rows as i32,
            s2.rows as i32,
            s3.rows as i32,
        ];
        let in_i = in_dim as i32;
        let blocks: u32 = [s0.rows, s1.rows, s2.rows, s3.rows]
            .iter()
            .map(|n| n.div_ceil(WIDE_RPB) as u32)
            .sum();
        let words_g = in_dim / 8;
        if words_g.is_multiple_of(4) && words_g / 4 <= 256 {
            unsafe {
                ctx.stream
                    .launch_builder(&self.group_v4)
                    .arg(s0.packed)
                    .arg(s0.scales)
                    .arg(s0.biases)
                    .arg(s0.y)
                    .arg(yb[0])
                    .arg(&r[0])
                    .arg(s1.packed)
                    .arg(s1.scales)
                    .arg(s1.biases)
                    .arg(s1.y)
                    .arg(yb[1])
                    .arg(&r[1])
                    .arg(s2.packed)
                    .arg(s2.scales)
                    .arg(s2.biases)
                    .arg(s2.y)
                    .arg(yb[2])
                    .arg(&r[2])
                    .arg(s3.packed)
                    .arg(s3.scales)
                    .arg(s3.biases)
                    .arg(s3.y)
                    .arg(yb[3])
                    .arg(&r[3])
                    .arg(x)
                    .arg(xb)
                    .arg(&in_i)
                    .launch(LaunchConfig {
                        grid_dim: (cols, blocks, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .context("wide group_v4 launch failed")?;
            return Ok(());
        }
        unsafe {
            ctx.stream
                .launch_builder(&self.group)
                .arg(s0.packed)
                .arg(s0.scales)
                .arg(s0.biases)
                .arg(s0.y)
                .arg(yb[0])
                .arg(&r[0])
                .arg(s1.packed)
                .arg(s1.scales)
                .arg(s1.biases)
                .arg(s1.y)
                .arg(yb[1])
                .arg(&r[1])
                .arg(s2.packed)
                .arg(s2.scales)
                .arg(s2.biases)
                .arg(s2.y)
                .arg(yb[2])
                .arg(&r[2])
                .arg(s3.packed)
                .arg(s3.scales)
                .arg(s3.biases)
                .arg(s3.y)
                .arg(yb[3])
                .arg(&r[3])
                .arg(x)
                .arg(xb)
                .arg(&in_i)
                .launch(LaunchConfig {
                    grid_dim: (cols, blocks, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("wide group launch failed")?;
        Ok(())
    }
}

impl WideKernels {
    /// uint4-widened splitk A/B twin of `down` (same contract; requires
    /// slice_words % 4 == 0). Uses the SAME columned combine.
    #[allow(clippy::too_many_arguments)]
    pub fn down_v4(
        &self,
        ctx: &GpuContext,
        packed: &CudaSlice<u32>,
        scales: &CudaSlice<f32>,
        biases: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        yb: &CudaSlice<f32>,
        scratch: &CudaSlice<f32>,
        scratch_b: &CudaSlice<f32>,
        rows: usize,
        in_dim: usize,
        split: usize,
        cols: u32,
    ) -> Result<()> {
        let words = in_dim / 8;
        let slice_words = words / split;
        anyhow::ensure!(
            slice_words.is_multiple_of(4) && words.is_multiple_of(4),
            "uint4 splitk needs 4-word-aligned slices (in {in_dim}, split {split})"
        );
        // Same unguarded row loop as down(): full RPB multiples only.
        anyhow::ensure!(
            rows.is_multiple_of(WIDE_RPB),
            "wide splitk: rows {rows} not a multiple of {WIDE_RPB}"
        );
        let rows_i = rows as i32;
        let in_i = in_dim as i32;
        let split_i = split as i32;
        let blocks = rows.div_ceil(WIDE_RPB) * split;
        unsafe {
            ctx.stream
                .launch_builder(&self.splitk_v4)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(x)
                .arg(xb)
                .arg(scratch)
                .arg(scratch_b)
                .arg(&rows_i)
                .arg(&in_i)
                .arg(&split_i)
                .launch(LaunchConfig {
                    grid_dim: (cols, blocks as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("splitk_v4 launch failed")?;
        unsafe {
            ctx.stream
                .launch_builder(&self.combine)
                .arg(scratch)
                .arg(scratch_b)
                .arg(y)
                .arg(yb)
                .arg(&rows_i)
                .arg(&split_i)
                .launch(LaunchConfig {
                    grid_dim: (cols, (rows as u32).div_ceil(256), 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("combine launch failed")?;
        Ok(())
    }
}
