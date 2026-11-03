use crate::{
    checkpoint::{ExpectedTensor, LoadPolicy, MappedCheckpoint, TensorManifest, TensorScope},
    model_spec::{Architecture, ModelSpec},
    prefill::DensePlacement,
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[cfg(feature = "cuda")]
pub use crate::backend::cuda::music3::Runtime;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepthConfig {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    version: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub audio_vocab_size: usize,
    pub num_codebooks: usize,
    pub max_position_embeddings: usize,
}

impl DepthConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.class == "MiniMaxMusic3RVQDepthDecoder" && !self.version.is_empty(),
            "invalid Music3 depth model definition"
        );
        ensure!(
            [
                self.hidden_size,
                self.intermediate_size,
                self.num_layers,
                self.num_attention_heads,
                self.audio_vocab_size,
                self.num_codebooks,
                self.max_position_embeddings
            ] == [4096, 6144, 4, 16, 1024, 8, 16],
            "unsupported Music3 depth geometry"
        );
        Ok(())
    }

    pub fn manifest(&self, mapped: &MappedCheckpoint) -> Result<TensorManifest> {
        self.validate()?;
        let h = self.hidden_size;
        let i = self.intermediate_size;
        let mut expected = BTreeMap::<String, ExpectedTensor>::new();
        let mut add = |name: String, shape: Vec<usize>| {
            expected.insert(name, (Dtype::BF16, shape, TensorScope::Text));
        };
        for (name, shape) in [
            ("projection.weight", vec![h, h]),
            (
                "pos_embedding.weight",
                vec![self.max_position_embeddings, h],
            ),
            (
                "audio_embeddings.weight",
                vec![self.audio_vocab_size * (self.num_codebooks - 1), h],
            ),
            ("norm.weight", vec![h]),
        ] {
            add(name.into(), shape);
        }
        for head in 0..self.num_codebooks - 1 {
            add(
                format!("audio_heads.{head}.weight"),
                vec![self.audio_vocab_size, h],
            );
        }
        for layer in 0..self.num_layers {
            for (name, shape) in [
                ("attn.to_q.weight", vec![h, h]),
                ("attn.to_k.weight", vec![h, h]),
                ("attn.to_v.weight", vec![h, h]),
                ("attn.to_out.weight", vec![h, h]),
                ("gate_proj.weight", vec![i, h]),
                ("up_proj.weight", vec![i, h]),
                ("down_proj.weight", vec![h, i]),
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
            ] {
                add(format!("layers.{layer}.{name}"), shape);
            }
        }
        let bytes = expected.values().try_fold(0usize, |sum, (_, shape, _)| {
            sum.checked_add(
                product(shape)?
                    .checked_mul(2)
                    .context("depth bytes overflow")?,
            )
            .context("depth manifest overflow")
        })?;
        mapped.assemble_manifest("Music3 depth", expected, LoadPolicy::TextOnly, bytes, None)
    }
}

pub struct ArCheckpoint {
    pub language: ModelSpec,
    pub depth: DepthConfig,
    pub language_weights: MappedCheckpoint,
    pub depth_weights: MappedCheckpoint,
    pub language_manifest: TensorManifest,
    pub depth_manifest: TensorManifest,
}

impl ArCheckpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let index: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("modular_model_index.json"))?)?;
        ensure!(
            index["_class_name"] == "MiniMaxMusic3ModularPipeline",
            "invalid Music3 pipeline class"
        );
        for (component, library, class) in [
            ("language_model", "transformers", "Qwen3ForCausalLM"),
            (
                "rvq_depth_decoder",
                "diffusers",
                "MiniMaxMusic3RVQDepthDecoder",
            ),
        ] {
            let entry = &index[component];
            ensure!(
                entry[0] == library && entry[1] == class && entry[2]["subfolder"] == component,
                "invalid Music3 component declaration: {component}"
            );
        }
        let language_dir = dir.join("language_model");
        let depth_dir = dir.join("rvq_depth_decoder");
        let language = ModelSpec::read(&language_dir)?;
        ensure!(
            language.architecture == Architecture::Qwen3
                && [
                    language.hidden,
                    language.intermediate,
                    language.vocab,
                    language.q_heads,
                    language.kv_heads,
                    language.head_dim,
                    language.max_position
                ] == [4096, 12288, 200000, 32, 8, 128, 10240]
                && language.layers.len() == 36
                && language.rms_eps == 1e-6
                && language.rope_theta == 1e6
                && language.eos.is_empty(),
            "unsupported Music3 language geometry/constants"
        );
        let depth: DepthConfig = serde_json::from_slice(&fs::read(depth_dir.join("config.json"))?)?;
        depth.validate()?;
        let language_weights = MappedCheckpoint::open(&language_dir)?;
        let depth_weights =
            MappedCheckpoint::open_component(&depth_dir, "diffusion_pytorch_model")?;
        let language_manifest = language_weights.manifest(&language, LoadPolicy::TextOnly)?;
        let depth_manifest = depth.manifest(&depth_weights)?;
        Ok(Self {
            language,
            depth,
            language_weights,
            depth_weights,
            language_manifest,
            depth_manifest,
        })
    }

    pub fn memory_plan(&self, request: &MemoryRequest) -> Result<MemoryPlan> {
        ensure!(
            request.prompt > 0
                && request.prompt <= 5000
                && request.frames > 0
                && request.frames <= 9000,
            "Music3 prompt/frame capacity exceeds the AR contract"
        );
        ensure!(
            request.device_free <= request.device_total && request.workspace_bytes > 0,
            "invalid Music3 device/workspace budget"
        );
        let capacity = request
            .prompt
            .checked_add(request.frames)
            .context("AR capacity overflow")?;
        let global_kv_bytes = product(&[
            2,
            2,
            self.language.layers.len(),
            self.language.kv_heads as usize,
            self.language.head_dim as usize,
            capacity,
            2,
        ])?;
        let depth_kv_bytes = product(&[
            2,
            2,
            self.depth.num_layers,
            self.depth.hidden_size,
            self.depth.max_position_embeddings,
            2,
        ])?;
        let rope_bytes = product(&[capacity, self.language.rotary_dim as usize, 2, 4])?;
        let feature_bytes = product(&[
            request.frames,
            self.depth.num_codebooks,
            self.depth.hidden_size,
            4,
        ])?;
        let mut fixed_weights = self.depth_manifest.resident_bytes;
        let mut layers = vec![0usize; self.language.layers.len()];
        let mut largest_matrix = 0;
        for entry in self.language_manifest.entries.values() {
            if entry.is_linear_projection()
                && let Some(layer) = entry.layer
            {
                let bytes = layers
                    .get_mut(layer as usize)
                    .context("AR manifest layer out of range")?;
                *bytes = bytes
                    .checked_add(entry.bytes)
                    .context("AR layer bytes overflow")?;
                largest_matrix = largest_matrix.max(entry.bytes);
            } else {
                fixed_weights = fixed_weights
                    .checked_add(entry.bytes)
                    .context("AR fixed weights overflow")?;
            }
        }
        let transient_bytes = [
            global_kv_bytes,
            depth_kv_bytes,
            rope_bytes,
            request.workspace_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |n, v| {
            n.checked_add(v).context("AR transient bytes overflow")
        })?;
        let budget = DensePlacement::device_budget(
            request.device_free,
            request.device_total,
            fixed_weights,
            transient_bytes,
            0,
        )?;
        let preliminary = DensePlacement::new(&layers, budget, 0)?;
        let host_staging = if preliminary.slot_bytes == 0 {
            0
        } else {
            preliminary
                .slot_bytes
                .checked_mul(2)
                .and_then(|n| n.checked_add(largest_matrix))
                .context("AR host staging overflow")?
        };
        let host_fixed_bytes = feature_bytes
            .checked_add(host_staging)
            .and_then(|n| n.checked_add(4 << 30))
            .context("AR host buffers overflow")?;
        let host_budget = request
            .host_available
            .checked_sub(host_fixed_bytes)
            .context("AR features, pinned staging and host reserve exceed available host memory")?;
        let placement = DensePlacement::new(&layers, budget, host_budget)?;
        let device_bytes = fixed_weights
            .checked_add(transient_bytes)
            .and_then(|n| n.checked_add(placement.resident_bytes))
            .and_then(|n| n.checked_add(placement.slot_bytes.checked_mul(2)?))
            .context("AR device footprint overflow")?;
        Ok(MemoryPlan {
            capacity,
            global_kv_bytes,
            depth_kv_bytes,
            rope_bytes,
            feature_bytes,
            host_fixed_bytes,
            device_bytes,
            placement,
        })
    }
}

pub struct MemoryRequest {
    pub prompt: usize,
    pub frames: usize,
    pub device_free: usize,
    pub device_total: usize,
    pub host_available: usize,
    pub workspace_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct MemoryPlan {
    pub capacity: usize,
    pub global_kv_bytes: usize,
    pub depth_kv_bytes: usize,
    pub rope_bytes: usize,
    pub feature_bytes: usize,
    pub host_fixed_bytes: usize,
    pub device_bytes: usize,
    pub placement: DensePlacement,
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &v| {
        n.checked_mul(v).context("Music3 extent overflow")
    })
}
