//! Published Pixal3D flow geometry and checkpoint contracts.
use crate::checkpoint::{MappedCheckpoint, PublishedTensor};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct FlowSpec {
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub pe_mode: String,
    pub share_mod: bool,
    pub qk_rms_norm: bool,
    pub qk_rms_norm_cross: bool,
    pub image_attn_mode: String,
    pub dtype: String,
    pub proj_in_channels: Option<usize>,
}

impl FlowSpec {
    pub fn sparse(&self) -> bool {
        self.in_channels == 32
    }
    pub fn projection_width(&self) -> usize {
        self.proj_in_channels.unwrap_or(self.cond_channels)
    }
    pub fn feed_forward_width(&self) -> usize {
        (self.model_channels as f64 * self.mlp_ratio) as usize
    }

    pub fn read(dir: &Path, stem: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Definition {
            name: String,
            args: FlowSpec,
        }
        let d: Definition =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{stem}.json")))?)?;
        let s = d.args;
        ensure!(
            (d.name == "SparseStructureFlowModel" && s.in_channels == 8 && s.resolution == 16)
                || (d.name == "ElasticSLatFlowModel"
                    && s.in_channels == 32
                    && matches!(s.resolution, 32 | 64)),
            "unsupported Pixal3D flow definition"
        );
        ensure!(
            s.out_channels == s.in_channels
                && s.model_channels == 1536
                && s.cond_channels == 1024
                && s.num_blocks == 30
                && s.num_heads == 12
                && s.feed_forward_width() == 8192
                && s.pe_mode == "rope"
                && s.share_mod
                && s.qk_rms_norm
                && s.qk_rms_norm_cross
                && s.image_attn_mode == "proj"
                && s.dtype == "bfloat16"
                && s.projection_width() == if s.sparse() { 2048 } else { 1024 },
            "unsupported Pixal3D flow geometry/arithmetic"
        );
        Ok(s)
    }

    pub(crate) fn inventory(&self) -> BTreeMap<String, (Vec<usize>, bool)> {
        let mut out = BTreeMap::new();
        let h = self.model_channels;
        let mut linear = |name: String, output: usize, input: usize, bf16: bool| {
            out.insert(format!("{name}.weight"), (vec![output, input], bf16));
            out.insert(format!("{name}.bias"), (vec![output], bf16));
        };
        for (name, output, input) in [
            ("input_layer", h, self.in_channels),
            ("out_layer", self.out_channels, h),
            ("t_embedder.mlp.0", h, 256),
            ("t_embedder.mlp.2", h, h),
            ("adaLN_modulation.1", 6 * h, h),
        ] {
            linear(name.into(), output, input, false);
        }
        for i in 0..self.num_blocks {
            for (name, output, input) in [
                ("self_attn.to_qkv", 3 * h, h),
                ("self_attn.to_out", h, h),
                ("cross_attn.cross_attn_block.to_q", h, h),
                (
                    "cross_attn.cross_attn_block.to_kv",
                    2 * h,
                    self.cond_channels,
                ),
                ("cross_attn.cross_attn_block.to_out", h, h),
                ("cross_attn.proj_linear", h, self.projection_width()),
                ("mlp.mlp.0", self.feed_forward_width(), h),
                ("mlp.mlp.2", h, self.feed_forward_width()),
            ] {
                linear(format!("blocks.{i}.{name}"), output, input, true);
            }
        }
        for i in 0..self.num_blocks {
            out.insert(format!("blocks.{i}.modulation"), (vec![6 * h], false));
            for name in ["norm2.weight", "norm2.bias"] {
                out.insert(format!("blocks.{i}.{name}"), (vec![h], false));
            }
            for prefix in ["self_attn", "cross_attn.cross_attn_block"] {
                for qk in ["q", "k"] {
                    out.insert(
                        format!("blocks.{i}.{prefix}.{qk}_rms_norm.gamma"),
                        (vec![self.num_heads, h / self.num_heads], false),
                    );
                }
            }
        }
        if !self.sparse() {
            out.insert(
                "rope_phases".into(),
                (vec![self.resolution.pow(3), h / self.num_heads / 2], false),
            );
        }
        out
    }

    pub(crate) fn validate_tensors(
        &self,
        tensors: &BTreeMap<String, PublishedTensor<'_>>,
    ) -> Result<usize> {
        let inventory = self.inventory();
        ensure!(
            tensors.keys().eq(inventory.keys()),
            "Pixal3D checkpoint missing/unexpected keys"
        );
        let mut bytes = 0usize;
        for (name, (shape, bf16)) in inventory {
            let tensor = &tensors[&name];
            let dtype = if name == "rope_phases" { "C64" } else { "F32" };
            ensure!(
                tensor.shape == shape && tensor.dtype == dtype,
                "{name}: stored {} {:?}, expected {dtype} {shape:?}",
                tensor.dtype,
                tensor.shape
            );
            bytes = bytes
                .checked_add(if bf16 {
                    tensor.data.len() / 2
                } else {
                    tensor.data.len()
                })
                .context("Pixal3D runtime weight bytes overflow")?;
        }
        Ok(bytes)
    }
}

pub struct FlowCheckpoint {
    pub spec: FlowSpec,
    pub mapped: MappedCheckpoint,
    pub runtime_bytes: usize,
}
impl FlowCheckpoint {
    pub fn open(dir: &Path, stem: &str) -> Result<Self> {
        let mapped = MappedCheckpoint::open_component(dir, stem)?;
        let spec = FlowSpec::read(dir, stem)?;
        let runtime_bytes = spec.validate_tensors(&mapped.component_tensors()?)?;
        Ok(Self {
            spec,
            mapped,
            runtime_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the published Pixal3D checkpoint; CPU only"]
    fn pixal3d_flow_checkpoint_cpu() -> Result<()> {
        let root = std::env::var_os("FF_PIXAL_MODEL").context("FF_PIXAL_MODEL required")?;
        let root = Path::new(&root).join("ckpts");
        for stem in [
            "ss_flow_img_dit_1_3B_64_bf16",
            "ss_flow_img_dit_1_3B_64_bf16_mv",
            "slat_flow_img2shape_dit_1_3B_512_bf16",
            "slat_flow_img2shape_dit_1_3B_1024_bf16",
            "slat_flow_img2shape_dit_1_3B_512_bf16_mv",
            "slat_flow_img2shape_dit_1_3B_1024_bf16_mv",
        ] {
            let checkpoint = FlowCheckpoint::open(&root, stem)?;
            let expected = if checkpoint.spec.sparse() {
                2_808_490_112
            } else {
                2_715_920_416
            };
            ensure!(
                checkpoint.runtime_bytes == expected,
                "runtime cast inventory differs from definition"
            );
            let mut tensors = checkpoint.mapped.component_tensors()?;
            tensors.remove("blocks.0.self_attn.q_rms_norm.gamma");
            ensure!(
                checkpoint.spec.validate_tensors(&tensors).is_err(),
                "missing Q norm control accepted"
            );
            println!(
                "PASS PIXAL_FLOW_LOAD component={stem} runtime_bytes={expected} missing=0 unexpected=0 new=0 missing_norm_control=rejected"
            );
        }
        Ok(())
    }
}
