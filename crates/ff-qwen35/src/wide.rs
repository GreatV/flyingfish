//! Wide-shape int4 GEMV launchers from int4_gemv_wide.cu.

use crate::kernel_assets::{RPB_V4, RPB_V4D8};
use anyhow::{Context, Result};
use cudarc::driver::safe::{CudaSlice, LaunchConfig, PushKernelArg};
use ff_edge0::gpu::GpuContext;

/// MUST match #define RPB in int4_gemv_wide.cu (the launcher's grid
/// decomposition and the kernel's block->(slice,row) mapping must agree —
/// a mismatch silently drops slices).
pub const WIDE_RPB: usize = 16;

pub struct WideKernels {
    group: cudarc::driver::safe::CudaFunction,
    pub(crate) bodies: ff_edge0::gpu::GroupKernels,
    group_v4_d8: cudarc::driver::safe::CudaFunction,
}

/// Every buffer and geometry value one wide `down` GEMV launch reads, named
/// one-to-one with the launcher's parameters.
pub struct DownBuffers<'a> {
    pub packed: &'a CudaSlice<u32>,
    pub scales: &'a CudaSlice<u16>,
    pub biases: &'a CudaSlice<u16>,
    pub x: &'a CudaSlice<f32>,
    pub xb: &'a CudaSlice<f32>,
    pub y: &'a CudaSlice<f32>,
    pub yb: &'a CudaSlice<f32>,
}

/// Copy the launch geometry for one wide `down` GEMV.
#[derive(Clone, Copy)]
pub struct WideGeom {
    pub rows: usize,
    pub in_dim: usize,
    pub cols: u32,
}

/// The input/output buffer pair one `group` launch reads.
pub struct GroupPair<'a> {
    pub x: &'a CudaSlice<f32>,
    pub xb: &'a CudaSlice<f32>,
}

impl WideKernels {
    pub fn load(ctx: &GpuContext) -> Result<Self> {
        let module = ff_edge0::kernel_assets::load_module(
            &ctx.context,
            &crate::kernel_assets::INT4_GEMV_WIDE,
        )
        .context("int4_gemv_wide module load failed")?;
        Ok(Self {
            group: module
                .load_function("int4_group4")
                .context("int4_group4 missing")?,
            bodies: ff_edge0::gpu::GroupKernels::new(
                &module,
                false,
                &ctx.context,
                &crate::kernel_assets::INT4_GEMV_WIDE,
            )?,
            group_v4_d8: module
                .load_function("int4_group4_v4d8")
                .context("int4_group4_v4d8 missing")?,
        })
    }

    pub fn capture_body(&self, body: Option<usize>) -> Result<()> {
        self.bodies.capture_body(body)
    }
    pub fn choices(&self) -> Vec<ff_core::probe::DecodeChoice> {
        self.bodies.choices()
    }

    /// y = quant(x) (column A) and yb = quant(xb) (column B, when
    /// cols == 2). For cols == 1 pass x/y as the b-arguments too — the
    /// kernel only takes the B path when gridDim.x == 2, so the aliases
    /// are never dereferenced.
    pub fn down(&self, ctx: &GpuContext, b: &DownBuffers<'_>, geom: &WideGeom) -> Result<()> {
        let DownBuffers {
            packed,
            scales,
            biases,
            x,
            xb,
            y,
            yb,
            ..
        } = *b;
        let WideGeom { rows, in_dim, cols } = *geom;
        anyhow::ensure!(
            in_dim > 8192,
            "wide down: in_dim {in_dim} is not the wide down projection (in_dim > 8192)"
        );
        anyhow::ensure!(
            in_dim % 64 == 0,
            "wide down: in_dim {in_dim} not a multiple of 64 — the 16-wide tile mapping would misindex groups"
        );
        anyhow::ensure!(
            rows % 4 == 0,
            "wide down: rows {rows} not a multiple of 4 — the 4-row tile would read past the segment"
        );
        let rows_i = rows as i32;
        let in_i = in_dim as i32;
        let blocks = rows.div_ceil(RPB_V4D8) as u32;
        unsafe {
            ctx.stream
                .launch_builder(&self.group_v4_d8)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(y)
                .arg(yb)
                .arg(&rows_i)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(y)
                .arg(yb)
                .arg(&0)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(y)
                .arg(yb)
                .arg(&0)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(y)
                .arg(yb)
                .arg(&0)
                .arg(x)
                .arg(xb)
                .arg(&in_i)
                .launch(LaunchConfig {
                    grid_dim: (cols, blocks, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("wide down launch failed")?;
        Ok(())
    }
}

/// The v4 group kernel serves 16-wide tiles: 64-column groups need in_dim
/// % 64 == 0, and the partials row caps at 128 groups (in_dim <= 8192).
fn group_v4_fits(in_dim: usize) -> bool {
    in_dim.is_multiple_of(64) && in_dim <= 8192
}

/// One wide grouped-GEMV launch over up to four same-x segments, full
/// width (in_dim <= 6144), no K-split. Segments carry no LoRA (qwen35
/// ships none); GroupSeg reuses the ff-edge0 type for call-site parity.
impl WideKernels {
    /// y_i = quant_i(x) (column A) and yb_i = quant_i(xb) (column B when
    /// cols == 2). yb entries are per-segment B outputs; for cols == 1
    /// pass the A slices again (aliases are never dereferenced at
    /// gridDim.y == 1).
    pub fn group(
        &self,
        ctx: &GpuContext,
        segs: &[ff_edge0::gpu::GroupSeg; 4],
        yb: [&CudaSlice<f32>; 4],
        pair: &GroupPair<'_>,
        in_dim: usize,
        cols: u32,
    ) -> Result<()> {
        let GroupPair { x, xb } = *pair;

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
            .map(|n| n.div_ceil(RPB_V4) as u32)
            .sum();

        if group_v4_fits(in_dim) {
            let blocks_g: u32 = blocks;
            let func = self.bodies.function(cols)?;
            unsafe {
                ctx.stream
                    .launch_builder(func)
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
                        grid_dim: (cols, blocks_g, 1),
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::ensure;

    #[test]
    fn group_v4_shape_selection() {
        assert!(group_v4_fits(5120));
        assert!(group_v4_fits(8192));
        assert!(!group_v4_fits(5152));
        assert!(!group_v4_fits(8224));
    }
    #[test]
    #[ignore = "requires a CUDA device"]
    fn group_bodies_match_f64() -> Result<()> {
        let model = ff_core::paths::checkpoint_dir("Qwen/Qwen3.8-27B-int4-rtn")
            .context("group4 fp64 fixture requires FF_MODELS_DIR")?;
        println!("group4 fixture checkpoint: {}", model.display());
        let config = crate::config::Qwen35Config::from_model_dir(&model)?;
        let weights = crate::weights::Qwen35Weights::open(&model)?;
        let mut shapes = std::collections::BTreeSet::new();
        for layer in 0..config.text_config.num_hidden_layers {
            for names in crate::gpu::grouped_names(
                config.text_config.layer_kind(layer),
                crate::prefill::Mode::Mma,
            ) {
                let dims = names
                    .iter()
                    .map(|name| {
                        weights.projection_shape(&format!(
                            "model.language_model.layers.{layer}.{name}"
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                shapes.insert((
                    dims[0].1,
                    std::array::from_fn::<_, 4, _>(|i| dims.get(i).map_or(0, |d| d.0)),
                ));
            }
        }
        let lm = weights.projection_shape("lm_head")?;
        shapes.insert((lm.1, [lm.0, 0, 0, 0]));
        shapes.insert((2048, [37, 1, 16, 5]));
        let ctx = GpuContext::new(0)?;
        let wide = WideKernels::load(&ctx)?;
        for (cols, rows) in shapes {
            let cpu = rows
                .iter()
                .filter(|n| **n > 0)
                .map(|n| {
                    let packed = (0..n * cols / 8)
                        .map(|i| {
                            (0..8).fold(0u32, |word, j| {
                                word | ((((i * 8 + j) % 61 % 16) as u32) << (4 * j))
                            })
                        })
                        .collect();
                    let scales = (0..n * cols / 64)
                        .map(|i| (1 + i % 47) as f32 / 1024.0)
                        .collect();
                    let biases = (0..n * cols / 64)
                        .map(|i| (i % 53) as f32 / 128.0 - 0.125)
                        .collect();
                    ff_edge0::int4::GroupQuant::new(packed, scales, biases, *n, cols, 4)
                })
                .collect::<Result<Vec<_>>>()?;
            let gpu = cpu
                .iter()
                .map(|q| ctx.upload(q, None))
                .collect::<Result<Vec<_>>>()?;
            let segs = std::array::from_fn(|i| {
                gpu.get(i)
                    .map_or_else(|| gpu[0].empty_seg_like(), |q| q.group_seg())
            });
            let xa: Vec<_> = (0..cols)
                .map(|c| (c % 43) as f32 / 32.0 - 21.0 / 32.0)
                .collect();
            let xb: Vec<_> = (0..cols)
                .map(|c| (c % 59) as f32 / 64.0 - 29.0 / 64.0)
                .collect();
            let x = ctx.upload_f32(&xa)?;
            let x_b = ctx.upload_f32(&xb)?;
            let yb = gpu
                .iter()
                .map(|q| ctx.stream.alloc_zeros::<f32>(q.out_dim))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let yb_refs = std::array::from_fn(|i| &yb[i.min(yb.len() - 1)]);
            let mut worst = [0.0_f64; 2];
            for (body, error) in worst.iter_mut().enumerate() {
                wide.capture_body(Some(body))?;
                wide.group(
                    &ctx,
                    &segs,
                    yb_refs,
                    &GroupPair { x: &x, xb: &x_b },
                    cols,
                    2,
                )?;
                for (segment, quant) in cpu.iter().enumerate() {
                    for (input, output) in [(&xa, gpu[segment].y_ref()), (&xb, &yb[segment])] {
                        let got = ctx.dtoh(output)?;
                        for row in [0, quant.out_dim / 2, quant.out_dim - 1] {
                            let mut want = 0.0;
                            let mut bound = 0.0;
                            for (c, x) in input.iter().enumerate() {
                                let word = quant.packed[row * cols / 8 + c / 8];
                                let nibble = ((word >> ((c % 8) * 4)) & 15) as f64;
                                let g = row * cols / 64 + c / 64;
                                let term = (quant.scales[g] as f64 * nibble
                                    + quant.biases[g] as f64)
                                    * *x as f64;
                                want += term;
                                bound += term.abs();
                            }
                            let ratio =
                                (got[row] as f64 - want).abs() / bound.max(f64::MIN_POSITIVE);
                            *error = error.max(ratio);
                            ensure!(
                                ratio <= (130 + cols / 64) as f64 * f32::EPSILON as f64,
                                "group4 body {body} shape {rows:?}/{cols} segment {segment} row {row}: {} vs f64 {want}, normalized {ratio}",
                                got[row]
                            );
                        }
                    }
                }
            }
            let (_, _, biases) = gpu[0].tensors();
            let mut bad = ctx.stream.clone_dtoh(biases)?;
            let value = f32::from_bits(u32::from(bad[0]) << 16) + 1.0;
            bad[0] = ff_edge0::int4::f32_to_bf16_bits(&[value])[0];
            let bad = ctx.stream.clone_htod(&bad)?;
            let mut corrupt = std::array::from_fn(|i| {
                gpu.get(i)
                    .map_or_else(|| gpu[0].empty_seg_like(), |q| q.group_seg())
            });
            corrupt[0].biases = &bad;
            let clean = ctx.dtoh(gpu[0].y_ref())?[0];
            wide.capture_body(Some(0))?;
            wide.group(
                &ctx,
                &corrupt,
                yb_refs,
                &GroupPair { x: &x, xb: &x_b },
                cols,
                2,
            )?;
            let changed = ctx.dtoh(gpu[0].y_ref())?[0];
            ensure!(
                (changed - clean).abs() > 1.0,
                "group4 bias negative control did not move output: {clean}/{changed}"
            );
            println!(
                "group4 shape {rows:?}/{cols}, paired columns: stock {}, xr16 {}, bias control {}",
                worst[0],
                worst[1],
                changed - clean
            );
        }
        Ok(())
    }
}
