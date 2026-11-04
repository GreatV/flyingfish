//! Published Pixal3D decoder layouts and runtime dtype inventory.
use crate::checkpoint::MappedCheckpoint;
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

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
    #[ignore = "requires published Pixal3D checkpoints; CPU only"]
    fn pixal3d_decoder_checkpoint_cpu() -> Result<()> {
        let root = std::env::var_os("FF_PIXAL_MODEL").context("FF_PIXAL_MODEL required")?;
        for (kind, keys) in [(DecoderKind::Structure, 74), (DecoderKind::Shape, 292)] {
            let checkpoint = DecoderCheckpoint::open(&Path::new(&root).join("ckpts"), kind)?;
            ensure!(
                checkpoint.mapped.index.weight_map.len() == keys,
                "decoder key count differs"
            );
            if matches!(kind, DecoderKind::Shape) {
                ensure!(
                    checkpoint.runtime_bytes == 948_598_492,
                    "sparse runtime dtype inventory differs"
                );
            }
            println!(
                "PASS PIXAL_DECODER_LOAD kind={kind:?} keys={keys} runtime_bytes={} missing=0 unexpected=0",
                checkpoint.runtime_bytes
            );
        }
        Ok(())
    }
}
