//! TRELLIS-1 structured-latent flow on an immutable sparse coordinate set.
use crate::{
    config::{PositionEmbeddingMode, SLatFlowArgs},
    sampler::SamplerParameters,
    slat_ops::{AttentionPlan, Ops, layer_norm, position},
    sparse::{Grid, PoolMap, conv3d},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::ModelWeights;

pub struct SLatFlow {
    args: SLatFlowArgs,
    ops: Ops,
    grid: Grid,
    pools: Vec<PoolMap>,
    neighbors: Vec<Tensor>,
    positions: Tensor,
    attention: AttentionPlan,
}

impl SLatFlow {
    pub fn new(
        args: SLatFlowArgs,
        weights: ModelWeights,
        grid: Grid,
        device: &Device,
        query_chunk: usize,
        voxel_chunk: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            query_chunk > 0
                && voxel_chunk > 0
                && args.num_heads > 0
                && args.model_channels.is_multiple_of(args.num_heads)
                && args.num_blocks > 0
                && args.mlp_ratio.is_finite()
                && args.mlp_ratio > 0.,
            "invalid SLat flow geometry/chunks"
        );
        anyhow::ensure!(
            args.pe_mode == PositionEmbeddingMode::Ape && !args.share_mod,
            "only TRELLIS-1 APE SLat flow is supported"
        );
        anyhow::ensure!(
            args.patch_size.is_power_of_two()
                && args.patch_size.trailing_zeros() as usize == args.io_block_channels.len()
                && (args.io_block_channels.is_empty() || args.num_io_res_blocks > 0),
            "invalid SLat IO packing configuration"
        );
        anyhow::ensure!(
            grid.coords
                .iter()
                .all(|c| c[0] == 0 && c[1..].iter().all(|&v| (v as usize) < args.resolution)),
            "SLat flow requires batch-one coordinates within its resolution"
        );
        let dtype = if args.use_fp16 && !device.is_cpu() {
            DType::F16
        } else {
            DType::F32
        };
        let mut pools = Vec::new();
        let mut current = grid.clone();
        let mut neighbors = vec![current.neighbor_tensor(device)?];
        for _ in &args.io_block_channels {
            let pool = current.downsample()?;
            current = pool.coarse.clone();
            neighbors.push(current.neighbor_tensor(device)?);
            pools.push(pool);
        }
        let positions = position(&current, args.model_channels, device, dtype)?;
        let attention = AttentionPlan::new(&current, None, device)?;
        Ok(Self {
            args,
            ops: Ops {
                weights,
                device: device.clone(),
                dtype,
                query_chunk,
                voxel_chunk,
            },
            grid,
            pools,
            neighbors,
            positions,
            attention,
        })
    }

    fn residual(
        &self,
        x: &Tensor,
        embedding: &Tensor,
        prefix: &str,
        level: usize,
    ) -> Result<Tensor> {
        let modulation = self.ops.linear(
            &candle_nn::ops::silu(embedding)?,
            &format!("{prefix}.emb_layers.1"),
        )?;
        let width = modulation.dim(1)? / 2;
        let scale = modulation.narrow(1, 0, width)?;
        let shift = modulation.narrow(1, width, width)?;
        let conv = |x: &Tensor, name: &str| -> Result<Tensor> {
            conv3d(
                x,
                &self.ops.get(&format!("{name}.conv.weight"), x.dtype())?,
                &self.ops.get(&format!("{name}.conv.bias"), x.dtype())?,
                &self.neighbors[level],
                self.ops.voxel_chunk,
            )
        };
        let h = candle_nn::ops::silu(&self.ops.norm(x, Some(&format!("{prefix}.norm1")), 1e-6)?)?;
        let h = conv(&h, &format!("{prefix}.conv1"))?;
        let h = self
            .ops
            .norm(&h, None, 1e-6)?
            .broadcast_mul(&(scale + 1.)?)?
            .broadcast_add(&shift)?;
        let h = conv(&candle_nn::ops::silu(&h)?, &format!("{prefix}.conv2"))?;
        let skip = if self
            .ops
            .weights
            .contains(&format!("{prefix}.skip_connection.weight"))
        {
            self.ops.linear(x, &format!("{prefix}.skip_connection"))?
        } else {
            x.clone()
        };
        Ok((h + skip)?)
    }

    pub fn forward(&self, x: &Tensor, time: &Tensor, condition: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            x.dims() == [self.grid.coords.len(), self.args.in_channels]
                && time.elem_count() == 1
                && condition.dims().len() == 3
                && condition.dims()[0] == 1
                && condition.dims()[2] == self.args.cond_channels,
            "SLat flow input geometry mismatch"
        );
        let embedding = self.ops.timestep(time, self.args.model_channels)?;
        let condition = condition.to_dtype(self.ops.dtype)?;
        let mut h = self
            .ops
            .linear(&x.to_dtype(DType::F32)?, "input_layer")?
            .to_dtype(self.ops.dtype)?;
        let mut skips = Vec::new();
        for stage in 0..self.pools.len() {
            for block in 0..self.args.num_io_res_blocks {
                let mut level = stage;
                if block + 1 == self.args.num_io_res_blocks {
                    h = self.pools[stage].pool(&h)?;
                    level += 1;
                }
                let p = format!(
                    "input_blocks.{}",
                    stage * self.args.num_io_res_blocks + block
                );
                h = self.residual(&h, &embedding, &p, level)?;
                skips.push(h.clone());
            }
        }
        h = (&h + &self.positions)?;
        for layer in 0..self.args.num_blocks {
            let p = format!("blocks.{layer}");
            let modulations = self.ops.linear(
                &candle_nn::ops::silu(&embedding)?,
                &format!("{p}.adaLN_modulation.1"),
            )?;
            let width = self.args.model_channels;
            let m = |i| modulations.narrow(1, i * width, width);
            let norm = self
                .ops
                .norm(&h, None, 1e-6)?
                .broadcast_mul(&(m(1)? + 1.)?)?
                .broadcast_add(&m(0)?)?;
            let attn = self.ops.self_attention(
                &norm,
                &format!("{p}.self_attn"),
                self.args.num_heads,
                self.args.qk_rms_norm,
                &self.attention,
            )?;
            h = (&h + attn.broadcast_mul(&m(2)?)?)?;
            let norm = self.ops.norm(&h, Some(&format!("{p}.norm2")), 1e-6)?;
            h = (&h
                + self.ops.cross_attention(
                    &norm,
                    &condition,
                    &format!("{p}.cross_attn"),
                    self.args.num_heads,
                    self.args.qk_rms_norm_cross,
                )?)?;
            let norm = self
                .ops
                .norm(&h, None, 1e-6)?
                .broadcast_mul(&(m(4)? + 1.)?)?
                .broadcast_add(&m(3)?)?;
            h = (&h
                + self
                    .ops
                    .ff(&norm, &format!("{p}.mlp"))?
                    .broadcast_mul(&m(5)?)?)?;
        }
        for (stage, pool) in self.pools.iter().enumerate().rev() {
            for block in 0..self.args.num_io_res_blocks {
                h = Tensor::cat(&[h, skips.pop().context("SLat skip stack exhausted")?], 1)?;
                if block == 0 {
                    h = pool.unpool(&h)?;
                }
                let index = (self.pools.len() - stage - 1) * self.args.num_io_res_blocks + block;
                h = self.residual(&h, &embedding, &format!("out_blocks.{index}"), stage)?;
            }
        }
        h = layer_norm(&h, 1e-5)?
            .to_dtype(self.ops.dtype)?
            .to_dtype(x.dtype())?;
        self.ops.linear(&h, "out_layer")
    }

    pub fn sample(
        &self,
        noise: &Tensor,
        condition: &Tensor,
        negative: &Tensor,
        parameters: SamplerParameters,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Tensor> {
        parameters.validate()?;
        let mut x = noise.clone();
        let times = parameters.timesteps();
        for (step, pair) in times.windows(2).enumerate() {
            let time = Tensor::new(&[(pair[0] * 1000.) as f32], &self.ops.device)?;
            let positive = self.forward(&x, &time, condition)?;
            let velocity =
                if parameters.cfg_interval.0 <= pair[0] && pair[0] <= parameters.cfg_interval.1 {
                    ((positive * (1. + parameters.cfg_strength))?
                        - (self.forward(&x, &time, negative)? * parameters.cfg_strength)?)?
                } else {
                    positive
                };
            x = (x - (velocity * (pair[0] - pair[1]))?)?;
            progress(step + 1, parameters.steps);
        }
        Ok(x)
    }
}
