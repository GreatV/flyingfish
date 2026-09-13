//! Structured latents to standard 3D Gaussian-splat PLY attributes.
use crate::{
    config::{AttentionMode, SLatCoderArgs},
    slat_ops::{AttentionPlan, Ops, layer_norm, position},
    sparse::Grid,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::ModelWeights;
use std::io::Write;

pub struct GaussianDecoder {
    args: SLatCoderArgs,
    ops: Ops,
}

impl GaussianDecoder {
    pub fn new(
        args: SLatCoderArgs,
        weights: ModelWeights,
        device: &Device,
        query_chunk: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            args.num_heads > 0
                && args.model_channels.is_multiple_of(args.num_heads)
                && args.num_blocks > 0
                && args.resolution > 0
                && args.window_size > 0
                && query_chunk > 0,
            "invalid Gaussian decoder geometry"
        );
        let dtype = if args.use_fp16 && !device.is_cpu() {
            DType::F16
        } else {
            DType::F32
        };
        Ok(Self {
            args,
            ops: Ops {
                weights,
                device: device.clone(),
                dtype,
                query_chunk,
                voxel_chunk: 256,
            },
        })
    }

    pub fn forward_raw(&self, grid: &Grid, latents: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            latents.dims() == [grid.coords.len(), self.args.latent_channels]
                && grid
                    .coords
                    .iter()
                    .all(|c| c[0] == 0
                        && c[1..].iter().all(|&v| (v as usize) < self.args.resolution)),
            "Gaussian decoder requires batch-one latents and in-range coordinates"
        );
        let mut h = self
            .ops
            .linear(&latents.to_dtype(DType::F32)?, "input_layer")?;
        h = (h + position(grid, self.args.model_channels, &self.ops.device, DType::F32)?)?
            .to_dtype(self.ops.dtype)?;
        let plans = match self.args.attn_mode {
            AttentionMode::Full => vec![AttentionPlan::new(grid, None, &self.ops.device)?],
            AttentionMode::Swin => vec![
                AttentionPlan::new(grid, Some((self.args.window_size, 0)), &self.ops.device)?,
                AttentionPlan::new(
                    grid,
                    Some((self.args.window_size, self.args.window_size / 2)),
                    &self.ops.device,
                )?,
            ],
        };
        for block in 0..self.args.num_blocks {
            let p = format!("blocks.{block}");
            let norm = self.ops.norm(&h, None, 1e-6)?;
            h = (&h
                + self.ops.self_attention(
                    &norm,
                    &format!("{p}.attn"),
                    self.args.num_heads,
                    false,
                    &plans[block % plans.len()],
                )?)?;
            h = (&h
                + self
                    .ops
                    .ff(&self.ops.norm(&h, None, 1e-6)?, &format!("{p}.mlp"))?)?;
        }
        self.ops.linear(
            &layer_norm(&h.to_dtype(latents.dtype())?, 1e-5)?,
            "out_layer",
        )
    }

    pub fn decode(&self, grid: &Grid, latents: &Tensor) -> Result<GaussianCloud> {
        let raw = self.forward_raw(grid, latents)?;
        let config = &self.args.representation_config;
        let count = usize::try_from(
            config
                .get("num_gaussians")
                .and_then(|v| v.as_u64())
                .context("missing num_gaussians")?,
        )?;
        anyhow::ensure!(
            count > 0 && raw.dims() == [grid.coords.len(), 14 * count],
            "Gaussian output layout mismatch"
        );
        let perturb = config
            .get("perturb_offset")
            .and_then(|v| v.as_bool())
            .context("missing Gaussian perturbation policy")?;
        let offsets = if perturb {
            self.ops
                .get("offset_perturbation", DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?
        } else {
            vec![vec![0.; 3]; count]
        };
        anyhow::ensure!(
            offsets.len() == count && offsets.iter().all(|row| row.len() == 3),
            "invalid Gaussian perturbation shape"
        );
        let scalar = |name: &str| {
            config
                .get(name)
                .and_then(|v| v.as_f64())
                .with_context(|| format!("missing Gaussian parameter {name}"))
        };
        let voxel = scalar("voxel_size")?;
        let kernel = scalar("3d_filter_kernel_size")?;
        let scale_bias = scalar("scaling_bias")?;
        let opacity_bias = scalar("opacity_bias")?;
        anyhow::ensure!(
            voxel.is_finite()
                && voxel > 0.
                && kernel.is_finite()
                && kernel >= 0.
                && scale_bias.is_finite()
                && scale_bias > 0.
                && opacity_bias > 0.
                && opacity_bias < 1.,
            "invalid Gaussian scale/opacity parameters"
        );
        let softplus = config
            .get("scaling_activation")
            .and_then(|v| v.as_str())
            .context("missing scaling activation")?;
        anyhow::ensure!(
            matches!(softplus, "softplus" | "exp"),
            "unsupported Gaussian scaling activation"
        );
        let scale_shift = if softplus == "softplus" {
            scale_bias + (-(-scale_bias).exp_m1()).ln()
        } else {
            scale_bias.ln()
        };
        let opacity_shift = (opacity_bias / (1. - opacity_bias)).ln();
        let gain = |name: &str| {
            config
                .get("lr")
                .and_then(|v| v.get(name))
                .and_then(|v| v.as_f64())
                .with_context(|| format!("missing Gaussian gain {name}"))
        };
        let gains = [
            gain("_xyz")?,
            gain("_features_dc")?,
            gain("_scaling")?,
            gain("_rotation")?,
            gain("_opacity")?,
        ];
        anyhow::ensure!(
            gains.iter().all(|v| v.is_finite()),
            "non-finite Gaussian gain"
        );
        let values = raw
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let mut vertices = Vec::with_capacity(
            values
                .len()
                .checked_mul(count)
                .context("Gaussian count overflow")?,
        );
        for (coord, row) in grid.coords.iter().zip(values) {
            anyhow::ensure!(
                row.iter().all(|v| v.is_finite()),
                "Gaussian network produced non-finite values"
            );
            for (g, perturb) in offsets.iter().enumerate() {
                let mut xyz = [0.; 3];
                let mut vertex = [0f32; 17];
                for axis in 0..3 {
                    let offset =
                        (f64::from(row[g * 3 + axis]) * gains[0] + f64::from(perturb[axis])).tanh();
                    xyz[axis] = (f64::from(coord[axis + 1]) + 0.5 + offset * 0.5 * voxel)
                        / self.args.resolution as f64
                        - 0.5;
                    vertex[6 + axis] = (f64::from(row[3 * count + g * 3 + axis]) * gains[1]) as f32;
                    let raw_scale =
                        f64::from(row[6 * count + g * 3 + axis]) * gains[2] + scale_shift;
                    let scale = if softplus == "exp" {
                        raw_scale.exp()
                    } else {
                        raw_scale.max(0.) + (-raw_scale.abs()).exp().ln_1p()
                    };
                    vertex[10 + axis] = (scale.hypot(kernel).ln()) as f32;
                }
                vertex[..3].copy_from_slice(&[xyz[0] as f32, -xyz[2] as f32, xyz[1] as f32]);
                vertex[9] = (f64::from(row[13 * count + g]) * gains[4] + opacity_shift) as f32;
                let mut q = [0f64; 4];
                for axis in 0..4 {
                    q[axis] = f64::from(row[9 * count + g * 4 + axis]) * gains[3]
                        + if axis == 0 { 1. } else { 0. };
                }
                let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
                anyhow::ensure!(norm.is_finite() && norm > 0., "invalid Gaussian quaternion");
                let factor = std::f64::consts::FRAC_1_SQRT_2 / norm;
                let rotated = [
                    (q[0] - q[1]) * factor,
                    (q[0] + q[1]) * factor,
                    (q[2] - q[3]) * factor,
                    (q[2] + q[3]) * factor,
                ];
                let sign = if rotated[0] < 0. { -1. } else { 1. };
                for axis in 0..4 {
                    vertex[13 + axis] = (rotated[axis] * sign) as f32;
                }
                anyhow::ensure!(
                    vertex.iter().all(|v| v.is_finite()),
                    "Gaussian export contains non-finite attributes"
                );
                vertices.push(vertex);
            }
        }
        Ok(GaussianCloud { vertices })
    }
}

/// Standard splat fields: xyz, zero normals, SH DC, opacity logit, log scales,
/// and a normalized wxyz quaternion. Positions use upstream's Y-up PLY axes.
pub struct GaussianCloud {
    pub vertices: Vec<[f32; 17]>,
}
impl GaussianCloud {
    pub fn write_ply(&self, mut writer: impl Write) -> Result<()> {
        writeln!(
            writer,
            "ply\nformat binary_little_endian 1.0\ncomment TRELLIS Gaussian splats; Y-up\nelement vertex {}",
            self.vertices.len()
        )?;
        for name in [
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
            "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ] {
            writeln!(writer, "property float {name}")?;
        }
        writeln!(writer, "end_header")?;
        for vertex in &self.vertices {
            for value in vertex {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }
}
