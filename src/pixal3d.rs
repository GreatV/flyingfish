//! Published Pixal3D flow geometry and checkpoint contracts.
use crate::checkpoint::{MappedCheckpoint, PublishedTensor};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde::Deserialize;
use serde_json::Value;
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

#[derive(Clone, Copy, Debug)]
pub enum DecoderKind {
    Structure,
    Shape,
}
pub struct DecoderCheckpoint {
    pub kind: DecoderKind,
    pub mapped: MappedCheckpoint,
    pub runtime_bytes: usize,
}
impl DecoderKind {
    pub fn stem(self) -> &'static str {
        match self {
            Self::Structure => "ss_dec_conv3d_16l8_fp16",
            Self::Shape => "shape_dec_next_dc_f16c32_fp16",
        }
    }
    pub(crate) fn inventory(self) -> BTreeMap<String, (Vec<usize>, Dtype, bool)> {
        let mut tensors = BTreeMap::new();
        let mut pair =
            |name: String, shape: Vec<usize>, width: usize, dtype: Dtype, f32_runtime: bool| {
                tensors.insert(format!("{name}.weight"), (shape, dtype, f32_runtime));
                tensors.insert(format!("{name}.bias"), (vec![width], dtype, f32_runtime));
            };
        match self {
            Self::Structure => {
                pair(
                    "input_layer".into(),
                    vec![512, 8, 3, 3, 3],
                    512,
                    Dtype::F32,
                    true,
                );
                pair(
                    "out_layer.2".into(),
                    vec![1, 32, 3, 3, 3],
                    1,
                    Dtype::F32,
                    true,
                );
                pair("out_layer.0".into(), vec![32], 32, Dtype::F32, true);
                for (name, c) in [
                    ("middle_block.0", 512),
                    ("middle_block.1", 512),
                    ("blocks.0", 512),
                    ("blocks.1", 512),
                    ("blocks.3", 128),
                    ("blocks.4", 128),
                    ("blocks.6", 32),
                    ("blocks.7", 32),
                ] {
                    for j in 1..=2 {
                        pair(format!("{name}.norm{j}"), vec![c], c, Dtype::F32, true);
                        pair(
                            format!("{name}.conv{j}"),
                            vec![c, c, 3, 3, 3],
                            c,
                            Dtype::F16,
                            false,
                        );
                    }
                }
                for (block, ci, co) in [(2, 512, 1024), (5, 128, 256)] {
                    pair(
                        format!("blocks.{block}.conv"),
                        vec![co, ci, 3, 3, 3],
                        co,
                        Dtype::F16,
                        false,
                    );
                }
            }
            Self::Shape => {
                pair("from_latent".into(), vec![1024, 32], 1024, Dtype::F16, true);
                pair("output_layer".into(), vec![7, 64], 7, Dtype::F16, true);
                for (stage, (c, blocks, co)) in
                    [(1024, 4, 512), (512, 16, 256), (256, 8, 128), (128, 4, 64)]
                        .into_iter()
                        .enumerate()
                {
                    for block in 0..blocks {
                        let name = format!("blocks.{stage}.{block}");
                        pair(
                            format!("{name}.conv"),
                            vec![c, 3, 3, 3, c],
                            c,
                            Dtype::F16,
                            false,
                        );
                        pair(format!("{name}.norm"), vec![c], c, Dtype::F16, true);
                        pair(
                            format!("{name}.mlp.0"),
                            vec![4 * c, c],
                            4 * c,
                            Dtype::F16,
                            false,
                        );
                        pair(
                            format!("{name}.mlp.2"),
                            vec![c, 4 * c],
                            c,
                            Dtype::F16,
                            false,
                        );
                    }
                    let name = format!("blocks.{stage}.{blocks}");
                    pair(format!("{name}.norm1"), vec![c], c, Dtype::F16, true);
                    pair(
                        format!("{name}.conv1"),
                        vec![8 * co, 3, 3, 3, c],
                        8 * co,
                        Dtype::F16,
                        false,
                    );
                    pair(
                        format!("{name}.conv2"),
                        vec![co, 3, 3, 3, co],
                        co,
                        Dtype::F16,
                        false,
                    );
                    pair(
                        format!("{name}.to_subdiv"),
                        vec![8, c],
                        8,
                        Dtype::F16,
                        false,
                    );
                }
            }
        }
        tensors
    }
    fn validate(self, config: &Value) -> Result<()> {
        let args = &config["args"];
        ensure!(
            args["use_fp16"] == true,
            "decoder precision differs from published contract"
        );
        match self {
            Self::Structure => ensure!(
                config["name"] == "SparseStructureDecoder"
                    && args["out_channels"] == 1
                    && args["latent_channels"] == 8
                    && args["num_res_blocks"] == 2
                    && args["num_res_blocks_middle"] == 2
                    && args["channels"] == serde_json::json!([512, 128, 32])
                    && args.get("norm_type").is_none_or(|v| v == "layer"),
                "unsupported dense structure decoder"
            ),
            Self::Shape => ensure!(
                config["name"] == "FlexiDualGridVaeDecoder"
                    && args["resolution"] == 256
                    && args["latent_channels"] == 32
                    && args["model_channels"] == serde_json::json!([1024, 512, 256, 128, 64])
                    && args["num_blocks"] == serde_json::json!([4, 16, 8, 4, 0])
                    && args["block_type"] == serde_json::json!(vec!["SparseConvNeXtBlock3d"; 5])
                    && args["up_block_type"] == serde_json::json!(vec!["SparseResBlockC2S3d"; 4])
                    && args["block_args"] == serde_json::json!([{}, {}, {}, {}, {}]),
                "unsupported sparse shape decoder"
            ),
        }
        Ok(())
    }
}
impl DecoderCheckpoint {
    pub fn open(dir: &Path, kind: DecoderKind) -> Result<Self> {
        let config: Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{}.json", kind.stem())))?)?;
        kind.validate(&config)?;
        let mapped = MappedCheckpoint::open_component(dir, kind.stem())?;
        let inventory = kind.inventory();
        ensure!(
            mapped.index.weight_map.keys().eq(inventory.keys()),
            "decoder checkpoint missing/unexpected keys"
        );
        let mut runtime_bytes = 0usize;
        for (name, (shape, dtype, f32_runtime)) in &inventory {
            let tensor = mapped.tensor(name)?;
            ensure!(
                tensor.shape() == shape && tensor.dtype() == *dtype,
                "decoder tensor {name} has unexpected stored dtype/layout"
            );
            let count = shape.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d).context("decoder shape overflow")
            })?;
            runtime_bytes = runtime_bytes
                .checked_add(
                    count
                        .checked_mul(if *f32_runtime { 4 } else { 2 })
                        .context("decoder bytes overflow")?,
                )
                .context("decoder bytes overflow")?;
        }
        Ok(Self {
            kind,
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
    #[test]
    #[ignore = "requires published Pixal3D checkpoints; CPU only"]
    fn pixal3d_decoder_checkpoint_cpu() -> Result<()> {
        let root = std::env::var_os("FF_PIXAL_MODEL").context("FF_PIXAL_MODEL required")?;
        for (kind, keys) in [(DecoderKind::Structure, 74), (DecoderKind::Shape, 292)] {
            let checkpoint = DecoderCheckpoint::open(&Path::new(&root).join("ckpts"), kind)?;
            ensure!(
                checkpoint.mapped.index.weight_map.len() == keys,
                "decoder key count differs"
            );
            let expected = match kind {
                DecoderKind::Structure => 147_585_412,
                DecoderKind::Shape => 948_598_492,
            };
            ensure!(
                checkpoint.runtime_bytes == expected,
                "decoder runtime dtype inventory differs"
            );
            println!(
                "PASS PIXAL_DECODER_LOAD kind={kind:?} keys={keys} runtime_bytes={} missing=0 unexpected=0",
                checkpoint.runtime_bytes
            );
        }
        Ok(())
    }
}
