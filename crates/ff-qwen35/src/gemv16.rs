//! 16-bit grouped GEMV launches (cuda/gemv16.cu) for raw bf16/f16
//! checkpoints: warp-per-row, uint4 weight reads, f32 activation and
//! accumulation. One launch covers up to four same-x segments; single
//! projections launch with three zero-row segments.
use ff_core::quant::QuantFormat;

use anyhow::{Context, Result};
use cudarc::driver::safe::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use ff_edge0::gpu::GpuContext;

/// Rows per block; MUST match RPB16 in cuda/gemv16.cu.
pub const GEMV16_RPB: usize = 8;

/// One segment of a grouped launch: raw 16-bit weights and f32 output.
pub struct Seg16<'a> {
    pub w: &'a CudaSlice<u8>,
    pub y: &'a CudaSlice<f32>,
    pub rows: usize,
}

impl Seg16<'_> {
    /// A zero-row segment reusing the same pointers (never dereferenced).
    pub fn empty_like(&self) -> Seg16<'_> {
        Seg16 {
            w: self.w,
            y: self.y,
            rows: 0,
        }
    }
}

pub struct Gemv16Kernels {
    group_bf16: CudaFunction,
    group_f16: CudaFunction,
}

impl Gemv16Kernels {
    pub fn load(ctx: &GpuContext) -> Result<Self> {
        let module =
            ff_edge0::kernel_assets::load_module(&ctx.context, &crate::kernel_assets::GEMV16)
                .context("gemv16 module load failed")?;
        Ok(Self {
            group_bf16: module
                .load_function("qwen_gemv16_group4_bf16")
                .context("qwen_gemv16_group4_bf16 missing")?,
            group_f16: module
                .load_function("qwen_gemv16_group4_f16")
                .context("qwen_gemv16_group4_f16 missing")?,
        })
    }

    /// y_i = w_i · x for each segment; zero-row segments contribute no
    /// blocks (their pointers are never dereferenced).
    pub fn group(
        &self,
        ctx: &GpuContext,
        format: QuantFormat,
        segs: [&Seg16<'_>; 4],
        x: &CudaSlice<f32>,
        in_dim: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            in_dim.is_multiple_of(8),
            "gemv16: in_dim {in_dim} not a multiple of 8 (uint4 row tiling)"
        );
        let blocks: u32 = segs
            .iter()
            .map(|s| s.rows.div_ceil(GEMV16_RPB) as u32)
            .sum();
        if blocks == 0 {
            return Ok(());
        }
        let in_i = in_dim as i32;
        let r: [i32; 4] = [
            segs[0].rows as i32,
            segs[1].rows as i32,
            segs[2].rows as i32,
            segs[3].rows as i32,
        ];
        let function = match format {
            QuantFormat::Bf16 => &self.group_bf16,
            QuantFormat::F16 => &self.group_f16,
            QuantFormat::GroupAffine { .. } => {
                anyhow::bail!("gemv16 launched for an int4 checkpoint")
            }
            QuantFormat::BlockFp8 { .. } => {
                anyhow::bail!("gemv16 launched for an FP8-block checkpoint")
            }
        };
        let mut launch = ctx.stream.launch_builder(function);
        for (i, seg) in segs.iter().enumerate() {
            launch.arg(seg.w).arg(seg.y).arg(&r[i]);
        }
        launch.arg(x).arg(&in_i);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (32 * GEMV16_RPB as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .context("gemv16 group launch failed")?;
        Ok(())
    }
}
