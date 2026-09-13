use candle_core::{Device, Tensor, safetensors};
use serde_json::json;
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

pub const PROBE_TENSOR: &str = "context_embedder.weight";

pub fn insert_test_qwen_contract(
    tensors: &mut HashMap<&'static str, Tensor>,
    language_rows: usize,
) {
    use flyingfish::h3::policy::{
        ExecutionBackendPolicy, H3QwenNumericalContract, H3QwenVisionLinearGeometry,
    };

    let rows = NonZeroUsize::new(language_rows).unwrap();
    let contract = H3QwenNumericalContract::for_verified_target(
        ExecutionBackendPolicy::Cpu,
        rows,
        rows,
        0,
        H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
    )
    .unwrap();
    contract
        .insert_artifact_tensors(tensors, &Device::Cpu)
        .unwrap();
}

const STAGE_TENSORS: [&str; 10] = [
    PROBE_TENSOR,
    "token_refiner.refiner_blocks.0.norm1.weight",
    "token_refiner.refiner_blocks.0.norm2.weight",
    "token_refiner.final_norm.weight",
    "time_embedder.linear_1.weight",
    "proj_in.weight",
    "transformer_blocks.0.adaln_proj.linear.weight",
    "transformer_blocks.0.norm1.weight",
    "transformer_blocks.0.norm2.weight",
    "norm_out.norm.weight",
];

pub struct TinyTransformerFixture {
    temporary: TempDir,
    model: PathBuf,
}

impl TinyTransformerFixture {
    pub fn checkpoint(&self) -> PathBuf {
        self.model.join("transformer")
    }

    pub fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("model");
        let transformer = model.join("transformer");
        fs::create_dir_all(&transformer).unwrap();

        let tensors = STAGE_TENSORS
            .iter()
            .enumerate()
            .map(|(index, name)| {
                (
                    (*name).to_owned(),
                    Tensor::new(&[index as f32], &Device::Cpu).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        safetensors::save(&tensors, transformer.join("weights.safetensors")).unwrap();

        let weight_map = STAGE_TENSORS
            .iter()
            .map(|name| ((*name).to_owned(), "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            transformer.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&json!({
                "metadata": {"total_size": 40},
                "weight_map": weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            transformer.join("config.json"),
            serde_json::to_vec_pretty(&json!({
                "_class_name": "MiniMaxH3Transformer3DModel",
                "num_attention_heads": 1,
                "attention_head_dim": 6,
                "hidden_size": 4,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "ffn_dim": 5,
                "in_channels": 1,
                "audio_in_channels": 1,
                "patch_size": [1, 1, 1],
                "text_dim": 2,
                "freq_dim": 2,
                "time_embed_hidden_dim": 2,
                "time_embed_dim": 2,
                "rope_freq_dim": 1,
                "rope_theta": 10_000.0,
                "norm_eps": 1e-5,
                "qk_norm_eps": 1e-5,
                "final_norm_eps": 1e-5
            }))
            .unwrap(),
        )
        .unwrap();

        Self { temporary, model }
    }

    pub fn model(&self) -> &Path {
        &self.model
    }

    pub fn scratch_path(&self, name: &str) -> PathBuf {
        self.temporary.path().join(name)
    }

    pub fn write_prompt_encoding(&self) -> PathBuf {
        let path = self.scratch_path("prompt.safetensors");
        let mut tensors = HashMap::from([
            (
                "prompt_embeddings",
                Tensor::zeros((1, 1, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "text_token_tags",
                Tensor::new(&[1u32], &Device::Cpu).unwrap(),
            ),
            ("token_ids", Tensor::new(&[64u32], &Device::Cpu).unwrap()),
        ]);
        insert_test_qwen_contract(&mut tensors, 1);
        safetensors::save(&tensors, &path).unwrap();
        path
    }
}
