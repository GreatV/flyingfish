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
pub use crate::backend::cuda::music3::{Runtime, generate};

/// One production synthesis request: AR prompt token ids in the paired
/// cond/uncond layout [2, prompt_rows] (row 0 cond, row 1 uncond; the
/// length must be even), the audio frame count, and the sampling seed
/// that drives both the AR draws and the native initial-latent draw.
pub struct Music3Request {
    pub prompt_ids: Vec<u32>,
    pub frames: usize,
    pub seed: u64,
}

/// Synthesis output: stereo PCM [2, latent_length(frames) * upsampling].
pub struct Music3Output {
    pub pcm: Vec<f32>,
}

/// Synthesis-stage geometry from the per-component configs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynSpec {
    pub flow_layers: u32,
    pub flow_heads: u32,
    pub flow_head_dim: u32,
    pub flow_ff_inner: u32,
    pub fourier_dim: u32,
    pub in_channels: u32,
    pub condition_dim: u32,
    pub rotary_dim: u32,
    pub cond_hidden: u32,
    pub cond_layers: u32,
    pub cond_out: u32,
    pub latent_channels: u32,
    pub decoder_input: u32,
    pub decoder_hidden: u32,
    pub upsampling: Vec<u32>,
    pub input_hop: u32,
    pub input_rate: u32,
    pub output_hop: u32,
    pub output_rate: u32,
}

impl SynSpec {
    /// Flow model width; the conv/proj shapes below derive from it.
    pub fn flow_dim(&self) -> u32 {
        self.flow_heads * self.flow_head_dim
    }

    /// Channel width of the pre/post k=1 convs and the proj_in input:
    /// latent + zero latent + condition channels.
    pub fn concat_channels(&self) -> u32 {
        2 * self.in_channels + self.condition_dim
    }

    /// Vocoder channel width of one folded stereo stream.
    pub fn vocoder_stream_channels(&self) -> u32 {
        self.latent_channels / 2
    }

    /// Latent length for a frame count: int(frames * out_rate * in_hop /
    /// (in_rate * out_hop)), at least 1.
    pub fn latent_length(&self, frames: usize) -> usize {
        let numerator = self.output_rate as u64 * self.input_hop as u64 * frames as u64;
        let denominator = self.input_rate as u64 * self.output_hop as u64;
        usize::max(1, (numerator / denominator) as usize)
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let component = |name: &str| -> Result<serde_json::Value> {
            let path = dir.join(name).join("config.json");
            let text = fs::read_to_string(&path)
                .with_context(|| format!("missing component config {}", path.display()))?;
            serde_json::from_str(&text).with_context(|| format!("invalid {name} config"))
        };
        let int = |v: &serde_json::Value, key: &str| -> Result<u32> {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .with_context(|| format!("missing/invalid {key}"))
        };
        let root: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(dir.join("config.json")).context("missing config.json")?,
        )?;
        let bad = |v: &serde_json::Value, key: &str| -> anyhow::Error {
            anyhow::anyhow!("unsupported {key}: {:?}", v.get(key))
        };
        let keys = |v: &serde_json::Value| -> Result<Vec<String>> {
            let object = v.as_object().context("config is not an object")?;
            Ok(object.keys().cloned().collect())
        };
        ensure!(
            root.get("architectures")
                == Some(&serde_json::json!([
                    "MiniMaxMusic3ForConditionalGeneration"
                ])),
            "unsupported architectures"
        );
        ensure!(
            root.get("model_type") == Some(&serde_json::json!("minimax_music3")),
            "unsupported model_type"
        );

        let flow = component("transformer")?;
        let flow_keys = keys(&flow)?;
        let want_flow = [
            "_class_name",
            "_diffusers_version",
            "attention_head_dim",
            "condition_dim",
            "ff_inner_dim",
            "fourier_embedding_dim",
            "in_channels",
            "num_attention_heads",
            "num_layers",
            "rotary_dim",
        ];
        ensure!(
            flow_keys.iter().all(|k| want_flow.contains(&k.as_str())),
            "unknown transformer config key"
        );
        ensure!(
            flow.get("_class_name") == Some(&serde_json::json!("MiniMaxMusic3Transformer1DModel")),
            "unsupported transformer class"
        );
        let heads = int(&flow, "num_attention_heads")?;
        let head_dim = int(&flow, "attention_head_dim")?;
        let rotary = int(&flow, "rotary_dim")?;
        ensure!(
            rotary <= head_dim && rotary.is_multiple_of(2),
            "invalid rotary_dim"
        );

        let cond = component("condition_encoder")?;
        let cond_keys = keys(&cond)?;
        let want_cond = [
            "_class_name",
            "_diffusers_version",
            "condition_hidden_dim",
            "input_hop_length",
            "input_sampling_rate",
            "num_condition_layers",
            "out_dim",
            "output_hop_length",
            "output_sampling_rate",
        ];
        ensure!(
            cond_keys.iter().all(|k| want_cond.contains(&k.as_str())),
            "unknown condition_encoder config key"
        );
        ensure!(
            cond.get("_class_name") == Some(&serde_json::json!("MiniMaxMusic3ConditionEncoder")),
            "unsupported condition_encoder class"
        );

        let vocoder = component("vocoder")?;
        let vocoder_keys = keys(&vocoder)?;
        let want_vocoder = [
            "_class_name",
            "_diffusers_version",
            "decoder_hidden_dim",
            "decoder_input_dim",
            "latent_channels",
            "sampling_rate",
            "upsampling_ratios",
        ];
        ensure!(
            vocoder_keys
                .iter()
                .all(|k| want_vocoder.contains(&k.as_str())),
            "unknown vocoder config key"
        );
        ensure!(
            vocoder.get("_class_name") == Some(&serde_json::json!("MiniMaxMusic3Vocoder")),
            "unsupported vocoder class"
        );
        let ratios = vocoder
            .get("upsampling_ratios")
            .and_then(serde_json::Value::as_array)
            .context("missing/invalid upsampling_ratios")?;
        let upsampling = ratios
            .iter()
            .map(|v| {
                u32::try_from(v.as_u64().context("upsampling ratio not an integer")?)
                    .context("upsampling ratio overflows u32")
            })
            .collect::<Result<Vec<u32>>>()?;
        ensure!(upsampling.len() == 4, "vocoder needs 4 upsampling stages");

        let spec = Self {
            flow_layers: int(&flow, "num_layers")?,
            flow_heads: heads,
            flow_head_dim: head_dim,
            flow_ff_inner: int(&flow, "ff_inner_dim")?,
            fourier_dim: int(&flow, "fourier_embedding_dim")?,
            in_channels: int(&flow, "in_channels")?,
            condition_dim: int(&flow, "condition_dim")?,
            rotary_dim: rotary,
            cond_hidden: int(&cond, "condition_hidden_dim")?,
            cond_layers: int(&cond, "num_condition_layers")?,
            cond_out: int(&cond, "out_dim")?,
            latent_channels: int(&vocoder, "latent_channels")?,
            decoder_input: int(&vocoder, "decoder_input_dim")?,
            decoder_hidden: int(&vocoder, "decoder_hidden_dim")?,
            upsampling,
            input_hop: int(&cond, "input_hop_length")?,
            input_rate: int(&cond, "input_sampling_rate")?,
            output_hop: int(&cond, "output_hop_length")?,
            output_rate: int(&cond, "output_sampling_rate")?,
        };
        ensure!(
            vocoder.get("sampling_rate") == Some(&serde_json::json!(spec.output_rate)),
            "vocoder sampling_rate differs from the condition output rate"
        );
        ensure!(
            spec.latent_channels.is_multiple_of(2)
                && spec.flow_dim() > 0
                && spec.flow_ff_inner.is_multiple_of(2)
                && spec.fourier_dim.is_multiple_of(2)
                && spec.condition_dim == spec.cond_out
                && spec.condition_dim == spec.flow_dim(),
            "inconsistent synthesis geometry"
        );
        let _ = (&bad, &int);
        Ok(spec)
    }

    /// Audited manifest for the transformer component, all F32 as published.
    pub fn manifest_transformer(&self, mapped: &MappedCheckpoint) -> Result<TensorManifest> {
        let dim = self.flow_dim() as usize;
        let concat = self.concat_channels() as usize;
        let ff = self.flow_ff_inner as usize;
        let fourier = self.fourier_dim as usize;
        let channels = self.in_channels as usize;
        let mut expected = BTreeMap::<String, ExpectedTensor>::new();
        let add =
            |expected: &mut BTreeMap<String, ExpectedTensor>, name: String, shape: Vec<usize>| {
                expected.insert(name, (Dtype::F32, shape, TensorScope::Text));
            };
        for layer in 0..self.flow_layers as usize {
            let p = format!("transformer_blocks.{layer}");
            for role in ["to_q", "to_k", "to_v"] {
                add(
                    &mut expected,
                    format!("{p}.attn.{role}.weight"),
                    vec![dim, dim],
                );
            }
            add(
                &mut expected,
                format!("{p}.attn.to_out.0.weight"),
                vec![dim, dim],
            );
            for norm in ["norm1", "norm2"] {
                for suffix in ["weight", "bias"] {
                    add(&mut expected, format!("{p}.{norm}.{suffix}"), vec![dim]);
                }
            }
            add(
                &mut expected,
                format!("{p}.ff_in.weight"),
                vec![2 * ff, dim],
            );
            add(&mut expected, format!("{p}.ff_in.bias"), vec![2 * ff]);
            add(&mut expected, format!("{p}.ff_out.weight"), vec![dim, ff]);
            add(&mut expected, format!("{p}.ff_out.bias"), vec![dim]);
        }
        for (name, shape) in [
            ("preprocess_conv.weight", vec![concat, concat, 1]),
            ("postprocess_conv.weight", vec![channels, channels, 1]),
            ("proj_in.weight", vec![dim, concat]),
            ("proj_out.weight", vec![channels, dim]),
            ("time_proj.weight", vec![fourier / 2, 1]),
            ("time_embed.linear_1.weight", vec![dim, fourier]),
            ("time_embed.linear_1.bias", vec![dim]),
            ("time_embed.linear_2.weight", vec![dim, dim]),
            ("time_embed.linear_2.bias", vec![dim]),
        ] {
            add(&mut expected, name.into(), shape);
        }
        let bytes = manifest_bytes(&expected)?;
        mapped.assemble_manifest(
            "Music3Syn transformer",
            expected,
            LoadPolicy::TextOnly,
            bytes,
            None,
        )
    }

    /// Audited manifest for the condition encoder component.
    pub fn manifest_condition(&self, mapped: &MappedCheckpoint) -> Result<TensorManifest> {
        let mut expected = BTreeMap::<String, ExpectedTensor>::new();
        let insert =
            |expected: &mut BTreeMap<String, ExpectedTensor>, name: &str, shape: Vec<usize>| {
                expected.insert(name.into(), (Dtype::F32, shape, TensorScope::Text));
            };
        insert(
            &mut expected,
            "proj.weight",
            vec![self.cond_out as usize, self.cond_hidden as usize, 3],
        );
        insert(&mut expected, "proj.bias", vec![self.cond_out as usize]);
        insert(
            &mut expected,
            "layer_weight_logits",
            vec![self.cond_layers as usize],
        );
        insert(&mut expected, "layer_scale", vec![1]);
        let bytes = manifest_bytes(&expected)?;
        mapped.assemble_manifest(
            "Music3Syn condition",
            expected,
            LoadPolicy::TextOnly,
            bytes,
            None,
        )
    }

    /// Audited manifest for the vocoder component.
    pub fn manifest_vocoder(&self, mapped: &MappedCheckpoint) -> Result<TensorManifest> {
        let hidden = self.decoder_hidden as usize;
        let input = self.decoder_input as usize;
        let streams = self.vocoder_stream_channels() as usize;
        let mut expected = BTreeMap::<String, ExpectedTensor>::new();
        let insert =
            |expected: &mut BTreeMap<String, ExpectedTensor>, name: &str, shape: Vec<usize>| {
                expected.insert(name.into(), (Dtype::F32, shape, TensorScope::Text));
            };
        insert(&mut expected, "dec_in_proj.weight", vec![input, streams, 1]);
        insert(&mut expected, "dec_in_proj.bias", vec![input]);
        insert(&mut expected, "conv_in.weight_g", vec![hidden, 1, 1]);
        insert(&mut expected, "conv_in.weight_v", vec![hidden, input, 7]);
        insert(&mut expected, "conv_in.bias", vec![hidden]);
        let mut stage_in = hidden;
        for (index, ratio) in self.upsampling.iter().enumerate() {
            let (in_dim, out_dim, stride) = (stage_in, stage_in / 2, *ratio as usize);
            stage_in = out_dim;
            let block = format!("blocks.{index}");
            insert(
                &mut expected,
                &format!("{block}.snake1.alpha"),
                vec![1, in_dim, 1],
            );
            insert(
                &mut expected,
                &format!("{block}.conv_t1.weight_g"),
                vec![in_dim, 1, 1],
            );
            insert(
                &mut expected,
                &format!("{block}.conv_t1.weight_v"),
                vec![in_dim, out_dim, 2 * stride],
            );
            insert(
                &mut expected,
                &format!("{block}.conv_t1.bias"),
                vec![out_dim],
            );
            for unit in 1..=3 {
                let ru = format!("{block}.res_unit{unit}");
                for snake in ["snake1", "snake2"] {
                    insert(
                        &mut expected,
                        &format!("{ru}.{snake}.alpha"),
                        vec![1, out_dim, 1],
                    );
                }
                insert(
                    &mut expected,
                    &format!("{ru}.conv1.weight_g"),
                    vec![out_dim, 1, 1],
                );
                insert(
                    &mut expected,
                    &format!("{ru}.conv1.weight_v"),
                    vec![out_dim, out_dim, 7],
                );
                insert(&mut expected, &format!("{ru}.conv1.bias"), vec![out_dim]);
                insert(
                    &mut expected,
                    &format!("{ru}.conv2.weight_g"),
                    vec![out_dim, 1, 1],
                );
                insert(
                    &mut expected,
                    &format!("{ru}.conv2.weight_v"),
                    vec![out_dim, out_dim, 1],
                );
                insert(&mut expected, &format!("{ru}.conv2.bias"), vec![out_dim]);
            }
        }
        let tail = stage_in;
        insert(&mut expected, "snake_out.alpha", vec![1, tail, 1]);
        insert(&mut expected, "conv_out.weight_g", vec![1, 1, 1]);
        insert(&mut expected, "conv_out.weight_v", vec![1, tail, 7]);
        insert(&mut expected, "conv_out.bias", vec![1]);
        let bytes = manifest_bytes(&expected)?;
        mapped.assemble_manifest(
            "Music3Syn vocoder",
            expected,
            LoadPolicy::TextOnly,
            bytes,
            None,
        )
    }
}

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
        let preliminary = DensePlacement::new(&layers, None, budget, 0)?;
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
        let placement = DensePlacement::new(&layers, None, budget, host_budget)?;
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

fn manifest_bytes(expected: &BTreeMap<String, ExpectedTensor>) -> Result<usize> {
    expected.values().try_fold(0usize, |sum, (_, shape, _)| {
        sum.checked_add(
            product(shape)?
                .checked_mul(4)
                .context("manifest bytes overflow")?,
        )
        .context("manifest size overflow")
    })
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &v| {
        n.checked_mul(v).context("Music3 extent overflow")
    })
}
