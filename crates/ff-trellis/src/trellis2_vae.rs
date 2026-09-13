//! TRELLIS.2 sparse ConvNeXt VAEs and learned channel-to-space subdivision.
use crate::{
    slat_ops::{Ops, layer_norm},
    sparse::{Grid, conv3d},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::ModelWeights;
use serde_json::Value;

pub struct Subdivision {
    parent: Vec<[u32; 4]>,
    indices: Vec<u32>,
    child: Grid,
}
pub struct VaeOutput {
    pub grid: Grid,
    pub features: Tensor,
    pub subdivisions: Vec<Subdivision>,
}
pub struct Vae2 {
    ops: Ops,
    widths: Vec<usize>,
    blocks: Vec<usize>,
    latent: usize,
    output: usize,
    predict: bool,
}
impl Vae2 {
    pub fn new(
        config: &Value,
        weights: ModelWeights,
        device: &Device,
        voxel_chunk: usize,
        output: usize,
    ) -> Result<Self> {
        let array = |key: &str| -> Result<Vec<usize>> {
            config[key]
                .as_array()
                .with_context(|| format!("missing VAE {key}"))?
                .iter()
                .map(|v| Ok(usize::try_from(v.as_u64().context("invalid VAE integer")?)?))
                .collect()
        };
        let widths = array("model_channels")?;
        let blocks = array("num_blocks")?;
        anyhow::ensure!(
            widths.len() >= 2
                && widths.len() == blocks.len()
                && widths.iter().all(|&v| v > 0)
                && voxel_chunk > 0,
            "invalid VAE stage geometry"
        );
        let stages = widths.len();
        anyhow::ensure!(
            config["block_type"].as_array().is_some_and(
                |v| v.len() == stages && v.iter().all(|x| x == "SparseConvNeXtBlock3d")
            ) && config["up_block_type"].as_array().is_some_and(
                |v| v.len() + 1 == stages && v.iter().all(|x| x == "SparseResBlockC2S3d")
            ) && config["block_args"]
                .as_array()
                .is_some_and(|v| v.len() == stages
                    && v.iter()
                        .all(|x| x.as_object().is_some_and(|x| x.is_empty()))),
            "unsupported TRELLIS.2 VAE blocks"
        );
        let latent = usize::try_from(
            config["latent_channels"]
                .as_u64()
                .context("missing latent channels")?,
        )?;
        anyhow::ensure!(
            latent > 0 && output > 0,
            "VAE channel counts must be positive"
        );
        if let Some(margin) = config.get("voxel_margin") {
            anyhow::ensure!(
                margin.as_f64() == Some(0.5),
                "only the released dual-grid voxel margin is supported"
            );
        }
        let dtype = if !device.is_cpu() && config["use_fp16"] == true {
            DType::F16
        } else {
            DType::F32
        };
        let predict = config
            .get("pred_subdiv")
            .map(|v| v.as_bool().context("invalid subdivision flag"))
            .transpose()?
            .unwrap_or(true);
        Ok(Self {
            ops: Ops {
                weights,
                device: device.clone(),
                dtype,
                query_chunk: 128,
                voxel_chunk,
            },
            widths,
            blocks,
            latent,
            output,
            predict,
        })
    }
    fn conv(&self, x: &Tensor, prefix: &str, neighbors: &Tensor) -> Result<Tensor> {
        conv3d(
            x,
            &self.ops.get(&format!("{prefix}.weight"), x.dtype())?,
            &self.ops.get(&format!("{prefix}.bias"), x.dtype())?,
            neighbors,
            self.ops.voxel_chunk,
        )
    }

    fn prepare_blocks(&self, stage: usize, rows: usize) -> Result<()> {
        if self.ops.device.is_cuda()
            && let Some(cache) = self
                .ops
                .weights
                .device_cache()
                .filter(|cache| cache.is_enabled())
        {
            let reserve = crate::generation::memory::vae_blocks(
                &self.ops.weights,
                &self.ops.device,
                stage,
                rows,
                self.widths[stage],
                self.ops.voxel_chunk,
            )?;
            let kind = if self.predict { "shape" } else { "texture" };
            crate::generation::memory::prepare(
                cache,
                &self.ops.device,
                &format!("{kind}-vae-{stage} ({rows} voxels)"),
                reserve,
            )?;
        }
        Ok(())
    }

    fn prepare_subdivision(&self, stage: usize, parents: usize, children: usize) -> Result<()> {
        if self.ops.device.is_cuda()
            && let Some(cache) = self
                .ops
                .weights
                .device_cache()
                .filter(|cache| cache.is_enabled())
        {
            let reserve = crate::generation::memory::vae_subdivision(
                &self.ops.weights,
                &self.ops.device,
                parents,
                children,
                self.widths[stage],
                self.widths[stage + 1],
                self.ops.voxel_chunk,
            )?;
            let kind = if self.predict { "shape" } else { "texture" };
            crate::generation::memory::prepare(
                cache,
                &self.ops.device,
                &format!("{kind}-subdivision-{stage} ({parents}->{children} voxels)"),
                reserve,
            )?;
        }
        Ok(())
    }
    pub fn decode(
        &self,
        initial: &Grid,
        latent: &Tensor,
        guides: Option<&[Subdivision]>,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<VaeOutput> {
        anyhow::ensure!(
            latent.dims() == [initial.coords.len(), self.latent],
            "VAE latent geometry mismatch"
        );
        anyhow::ensure!(
            (self.predict && guides.is_none())
                || (!self.predict && guides.is_some_and(|g| g.len() + 1 == self.widths.len())),
            "invalid VAE subdivision guides"
        );
        self.prepare_blocks(0, initial.coords.len())?;
        let mut grid = initial.clone();
        let mut h = self
            .ops
            .linear(latent, "from_latent")?
            .to_dtype(self.ops.dtype)?;
        let mut subdivisions = Vec::new();
        for stage in 0..self.widths.len() {
            if stage > 0 {
                self.prepare_blocks(stage, grid.coords.len())?;
            }
            let neighbors = grid.neighbor_tensor(&self.ops.device)?;
            for block in 0..self.blocks[stage] {
                let p = format!("blocks.{stage}.{block}");
                let norm = self.ops.norm(
                    &self.conv(&h, &format!("{p}.conv"), &neighbors)?,
                    Some(&format!("{p}.norm")),
                    1e-6,
                )?;
                let projected =
                    candle_nn::ops::silu(&self.ops.linear(&norm, &format!("{p}.mlp.0"))?)?;
                h = (&h + self.ops.linear(&projected, &format!("{p}.mlp.2"))?)?;
            }
            if stage + 1 < self.widths.len() {
                let p = format!("blocks.{stage}.{}", self.blocks[stage]);
                let generated;
                let subdivision = if let Some(guides) = guides {
                    &guides[stage]
                } else {
                    let logits = self
                        .ops
                        .linear(&h, &format!("{p}.to_subdiv"))?
                        .to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .to_vec2::<f32>()?;
                    let mut coords = Vec::new();
                    let mut indices = Vec::new();
                    for (row, (coord, mask)) in grid.coords.iter().zip(logits).enumerate() {
                        anyhow::ensure!(
                            mask.len() == 8 && mask.iter().all(|x| x.is_finite()),
                            "invalid subdivision logits"
                        );
                        for (child, &logit) in mask.iter().enumerate() {
                            if logit > 0. {
                                let mut c = *coord;
                                for axis in 0..3 {
                                    c[axis + 1] = coord[axis + 1]
                                        .checked_mul(2)
                                        .and_then(|v| v.checked_add(((child >> axis) & 1) as u32))
                                        .context("subdivision coordinate overflow")?;
                                }
                                coords.push(c);
                                indices.push(u32::try_from(row * 8 + child)?);
                            }
                        }
                    }
                    generated = Subdivision {
                        parent: grid.coords.clone(),
                        indices,
                        child: Grid::new(coords)?,
                    };
                    &generated
                };
                anyhow::ensure!(
                    subdivision.parent == grid.coords,
                    "texture subdivision coordinates differ from shape decoder"
                );
                self.prepare_subdivision(stage, grid.coords.len(), subdivision.indices.len())?;
                let out_width = self.widths[stage + 1];
                let in_width = h.dim(1)?;
                anyhow::ensure!(
                    in_width.is_multiple_of(8) && out_width.is_multiple_of(in_width / 8),
                    "invalid channel-to-space skip geometry"
                );
                let normalized =
                    candle_nn::ops::silu(&self.ops.norm(&h, Some(&format!("{p}.norm1")), 1e-6)?)?;
                let expanded = self.conv(&normalized, &format!("{p}.conv1"), &neighbors)?;
                let rows = subdivision.indices.len();
                let ids = Tensor::from_vec(subdivision.indices.clone(), rows, &self.ops.device)?;
                let child = expanded
                    .reshape((grid.coords.len() * 8, out_width))?
                    .index_select(&ids, 0)?;
                let skip = h
                    .reshape((grid.coords.len() * 8, in_width / 8))?
                    .index_select(&ids, 0)?
                    .unsqueeze(2)?
                    .broadcast_as((rows, in_width / 8, out_width / (in_width / 8)))?
                    .reshape((rows, out_width))?
                    .contiguous()?;
                grid = subdivision.child.clone();
                let normalized = candle_nn::ops::silu(&self.ops.norm(&child, None, 1e-6)?)?;
                h = (self.conv(
                    &normalized,
                    &format!("{p}.conv2"),
                    &grid.neighbor_tensor(&self.ops.device)?,
                )? + skip)?;
                if self.predict {
                    subdivisions.push(Subdivision {
                        parent: subdivision.parent.clone(),
                        indices: subdivision.indices.clone(),
                        child: grid.clone(),
                    });
                }
            }
            progress(stage + 1, self.widths.len());
        }
        let features = self.ops.linear(
            &layer_norm(&h.to_dtype(latent.dtype())?, 1e-5)?,
            "output_layer",
        )?;
        anyhow::ensure!(
            features.dim(1)? == self.output,
            "VAE output channel mismatch"
        );
        Ok(VaeOutput {
            grid,
            features,
            subdivisions,
        })
    }
}
