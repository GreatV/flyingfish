//! Normalized model geometry. Device execution remains gated by backend capability.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Qwen35Text,
    MiniCpmV47,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Mixer {
    FullAttention,
    Gdn,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MoeConfig {
    pub experts: u32,
    pub experts_per_tok: u32,
    pub expert_intermediate: u32,
    pub shared_intermediate: u32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LinearFormat {
    Bf16,
    Fp8E4m3Block128,
    /// compressed-tensors FP8_DYNAMIC per-output-channel weights plus token activations.
    Fp8E4m3Channel,
}
/// Routed-expert checkpoint layout: fused BF16 slices or per-expert E4M3 weights with BF16
/// per-output-channel scales (compressed-tensors FP8_DYNAMIC).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExpertFormat {
    #[default]
    Bf16,
    Fp8W8a16,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GdnConfig {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerPlan {
    pub mixer: Mixer,
    pub state_index: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub abi_version: u32,
    pub architecture: Architecture,
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_width: u32,
    pub max_position: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub linear_format: LinearFormat,
    pub expert_format: ExpertFormat,
    pub eos: Vec<u32>,
    pub layers: Vec<LayerPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdn: Option<GdnConfig>,
    #[serde(default)]
    pub vision: Option<VisionConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiniCpmVision {
    pub image_size: u32,
    pub insert_layer: u32,
    pub slice_start_token: u32,
    pub slice_end_token: u32,
    pub newline_token: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    pub depth: u32,
    pub hidden: u32,
    pub heads: u32,
    pub intermediate: u32,
    pub channels: u32,
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub output: u32,
    pub position_side: u32,
    pub image_token: u32,
    pub video_token: u32,
    pub start_token: u32,
    pub end_token: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minicpm: Option<MiniCpmVision>,
}

impl VisionConfig {
    pub fn validate(&self, architecture: Architecture) -> Result<()> {
        match architecture {
            Architecture::Qwen35Text => {
                ensure!(
                    [
                        self.depth,
                        self.hidden,
                        self.heads,
                        self.intermediate,
                        self.channels,
                        self.patch,
                        self.temporal_patch,
                        self.merge,
                        self.output,
                        self.position_side
                    ] == [27, 1152, 16, 4304, 3, 16, 2, 2, 5120, 48],
                    "unsupported Q38 vision geometry"
                );
                ensure!(
                    [
                        self.image_token,
                        self.video_token,
                        self.start_token,
                        self.end_token
                    ] == [248056, 248057, 248053, 248054],
                    "unsupported Q38 image token IDs"
                );
                ensure!(self.minicpm.is_none(), "unexpected MiniCPM vision fields");
            }
            Architecture::MiniCpmV47 => {
                ensure!(
                    [
                        self.depth,
                        self.hidden,
                        self.heads,
                        self.intermediate,
                        self.channels,
                        self.patch,
                        self.temporal_patch,
                        self.merge,
                        self.output,
                        self.position_side
                    ] == [27, 1152, 16, 4304, 3, 14, 1, 4, 2048, 70],
                    "unsupported MiniCPM vision geometry"
                );
                ensure!(
                    [
                        self.image_token,
                        self.video_token,
                        self.start_token,
                        self.end_token
                    ] == [248056, 248057, 248078, 248079],
                    "unsupported MiniCPM image token IDs"
                );
                let v = self
                    .minicpm
                    .as_ref()
                    .context("MiniCPM vision configuration missing")?;
                ensure!(
                    v.image_size == 980,
                    "unsupported image_size: {}",
                    v.image_size
                );
                ensure!(
                    v.insert_layer == 6,
                    "unsupported insert_layer_id: {}",
                    v.insert_layer
                );
                ensure!(
                    v.slice_start_token == 248088,
                    "unsupported slice_start_id: {}",
                    v.slice_start_token
                );
                ensure!(
                    v.slice_end_token == 248089,
                    "unsupported slice_end_id: {}",
                    v.slice_end_token
                );
                ensure!(
                    v.newline_token == 198,
                    "unsupported newline_id: {}",
                    v.newline_token
                );
            }
            Architecture::Llama => bail!("vision configuration requires a multimodal architecture"),
        }
        Ok(())
    }

    pub fn patch_elements(&self, architecture: Architecture) -> Result<usize> {
        self.validate(architecture)?;
        Ok((self.channels * self.temporal_patch * self.patch * self.patch) as usize)
    }

    fn from_root(root: &Value) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config") else {
            return Ok(None);
        };
        match v.get("model_type").and_then(Value::as_str) {
            Some("qwen3_5") => Self::qwen_from_root(root, v).map(Some),
            Some("minicpmv4_7_vision") => Self::minicpm_from_root(root, v).map(Some),
            other => bail!("unsupported vision_config model_type: {other:?}"),
        }
    }

    fn qwen_from_root(root: &Value, v: &Value) -> Result<Self> {
        equal(v, "hidden_act", "gelu_pytorch_tanh".into())?;
        equal(v, "deepstack_visual_indexes", serde_json::json!([]))?;
        ensure!(
            integer(v, "num_position_embeddings")? == 2304,
            "unsupported vision position table"
        );
        let spec = Self {
            depth: integer(v, "depth")?,
            hidden: integer(v, "hidden_size")?,
            heads: integer(v, "num_heads")?,
            intermediate: integer(v, "intermediate_size")?,
            channels: integer(v, "in_channels")?,
            patch: integer(v, "patch_size")?,
            temporal_patch: integer(v, "temporal_patch_size")?,
            merge: integer(v, "spatial_merge_size")?,
            output: integer(v, "out_hidden_size")?,
            position_side: 48,
            image_token: integer(root, "image_token_id")?,
            video_token: integer(root, "video_token_id")?,
            start_token: integer(root, "vision_start_token_id")?,
            end_token: integer(root, "vision_end_token_id")?,
            minicpm: None,
        };
        spec.validate(Architecture::Qwen35Text)?;
        Ok(spec)
    }

    fn minicpm_from_root(root: &Value, v: &Value) -> Result<Self> {
        equal(v, "hidden_act", "gelu_pytorch_tanh".into())?;
        if let Some(kernel) = v.get("window_kernel_size") {
            ensure!(
                kernel == &serde_json::json!([2, 2]),
                "unsupported window_kernel_size: {kernel:?}"
            );
        }
        let image_size = integer(v, "image_size")?;
        let patch = integer(v, "patch_size")?;
        ensure!(
            patch > 0 && image_size.is_multiple_of(patch),
            "image_size is not a patch multiple"
        );
        // MiniCPMV4_7Config defaults: merge_kernel_size (2, 2), merger_times 1.
        if let Some(kernel) = root.get("merge_kernel_size") {
            ensure!(
                kernel == &serde_json::json!([2, 2]),
                "unsupported merge_kernel_size: {kernel:?}"
            );
        }
        if let Some(times) = root.get("merger_times") {
            ensure!(
                integer(root, "merger_times")? == 1,
                "unsupported merger_times: {times:?}"
            );
        }
        let merge = match root.get("downsample_mode").and_then(Value::as_str) {
            Some("16x") => 4,
            other => bail!("unsupported downsample_mode: {other:?}"),
        };
        let spec = Self {
            depth: integer(v, "num_hidden_layers")?,
            hidden: integer(v, "hidden_size")?,
            heads: integer(v, "num_attention_heads")?,
            intermediate: integer(v, "intermediate_size")?,
            channels: integer(v, "num_channels")?,
            patch,
            temporal_patch: 1,
            merge,
            output: u32::try_from(
                root.get("text_config")
                    .and_then(|t| t.get("hidden_size"))
                    .and_then(Value::as_u64)
                    .context("missing text_config.hidden_size")?,
            )
            .context("hidden_size overflows u32")?,
            position_side: image_size / patch,
            image_token: integer(root, "image_token_id")?,
            video_token: integer(root, "video_token_id")?,
            start_token: integer(root, "image_start_id")?,
            end_token: integer(root, "image_end_id")?,
            minicpm: Some(MiniCpmVision {
                image_size,
                insert_layer: integer(root, "insert_layer_id")?,
                slice_start_token: integer(root, "slice_start_id")?,
                slice_end_token: integer(root, "slice_end_id")?,
                newline_token: integer(root, "newline_id")?,
            }),
        };
        spec.validate(Architecture::MiniCpmV47)?;
        Ok(spec)
    }
}
fn integer(v: &Value, key: &str) -> Result<u32> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing/invalid {key}"))?;
    u32::try_from(n).with_context(|| format!("{key} overflows u32"))
}
fn positive(v: &Value, key: &str) -> Result<u32> {
    let n = integer(v, key)?;
    ensure!(n > 0, "unsupported {key}: 0");
    Ok(n)
}
fn equal(v: &Value, key: &str, expected: Value) -> Result<()> {
    ensure!(
        v.get(key) == Some(&expected),
        "unsupported {key}: {:?}",
        v.get(key)
    );
    Ok(())
}
fn exact_keys(v: &Value, keys: &[&str], label: &str) -> Result<()> {
    let object = v
        .as_object()
        .with_context(|| format!("{label} is not an object"))?;
    if let Some(key) = object.keys().find(|key| !keys.contains(&key.as_str())) {
        bail!("unknown {label} key {key}");
    }
    for key in keys {
        ensure!(object.contains_key(*key), "{label} missing key {key}");
    }
    Ok(())
}
impl LinearFormat {
    fn read(root: &Value) -> Result<Self> {
        let Some(quant) = root.get("quantization_config") else {
            return Ok(Self::Bf16);
        };
        for (key, val) in [
            ("quant_method", "fp8"),
            ("fmt", "e4m3"),
            ("activation_scheme", "dynamic"),
        ] {
            equal(quant, key, val.into())?;
        }
        equal(quant, "weight_block_size", serde_json::json!([128, 128]))?;
        Ok(Self::Fp8E4m3Block128)
    }
}
impl ExpertFormat {
    /// Parses the published compressed-tensors FP8_DYNAMIC form; a partial or unknown config errors.
    fn read(root: &Value) -> Result<Self> {
        let Some(quant) = root
            .get("quantization_config")
            .filter(|value| !value.is_null())
        else {
            return Ok(Self::Bf16);
        };
        exact_keys(
            quant,
            &[
                "config_groups",
                "format",
                "global_compression_ratio",
                "ignore",
                "kv_cache_scheme",
                "quant_method",
                "quantization_status",
                "sparsity_config",
                "transform_config",
                "version",
            ],
            "quantization_config",
        )?;
        for (key, value) in [
            ("quant_method", "compressed-tensors"),
            ("format", "float-quantized"),
            ("quantization_status", "compressed"),
        ] {
            equal(quant, key, value.into())?;
        }
        for key in ["global_compression_ratio", "kv_cache_scheme"] {
            ensure!(
                quant.get(key).is_none_or(Value::is_null),
                "unsupported quantization {key}: {:?}",
                quant.get(key)
            );
        }
        for key in ["sparsity_config", "transform_config"] {
            ensure!(
                quant
                    .get(key)
                    .is_none_or(|value| value == &serde_json::json!({})),
                "unsupported quantization {key}: {:?}",
                quant.get(key)
            );
        }
        ensure!(
            quant
                .get("version")
                .and_then(Value::as_str)
                .is_some_and(|version| !version.is_empty()),
            "quantization version missing"
        );
        ensure!(
            quant
                .get("ignore")
                .and_then(Value::as_array)
                .is_some_and(|ignore| !ignore.is_empty()),
            "quantization ignore list missing"
        );
        let groups = quant
            .get("config_groups")
            .and_then(Value::as_object)
            .context("quantization config_groups missing")?;
        ensure!(
            groups.len() == 1,
            "quantization expects one config group, got {}",
            groups.len()
        );
        let group = groups.values().next().context("empty quantization group")?;
        exact_keys(
            group,
            &[
                "format",
                "input_activations",
                "output_activations",
                "targets",
                "weights",
            ],
            "quantization config group",
        )?;
        equal(group, "format", "float-quantized".into())?;
        ensure!(
            group.get("output_activations").is_none_or(Value::is_null),
            "unsupported quantization output_activations"
        );
        equal(group, "targets", serde_json::json!(["Linear"]))?;
        let weights = group
            .get("weights")
            .context("quantization weights missing")?;
        exact_keys(
            weights,
            &[
                "actorder",
                "block_structure",
                "dynamic",
                "group_size",
                "num_bits",
                "observer",
                "observer_kwargs",
                "scale_dtype",
                "strategy",
                "symmetric",
                "type",
                "zp_dtype",
            ],
            "quantization weights",
        )?;
        equal(weights, "num_bits", 8.into())?;
        equal(weights, "type", "float".into())?;
        equal(weights, "strategy", "channel".into())?;
        equal(weights, "symmetric", true.into())?;
        equal(weights, "dynamic", false.into())?;
        equal(weights, "observer", "memoryless_minmax".into())?;
        for key in [
            "actorder",
            "block_structure",
            "group_size",
            "scale_dtype",
            "zp_dtype",
        ] {
            ensure!(
                weights.get(key).is_none_or(Value::is_null),
                "unsupported quantization weights {key}"
            );
        }
        ensure!(
            weights
                .get("observer_kwargs")
                .is_none_or(|value| value == &serde_json::json!({})),
            "unsupported quantization weights observer_kwargs"
        );
        let activations = group
            .get("input_activations")
            .context("quantization input_activations missing")?;
        exact_keys(
            activations,
            &[
                "actorder",
                "block_structure",
                "dynamic",
                "group_size",
                "num_bits",
                "observer",
                "observer_kwargs",
                "scale_dtype",
                "strategy",
                "symmetric",
                "type",
                "zp_dtype",
            ],
            "quantization input_activations",
        )?;
        equal(activations, "num_bits", 8.into())?;
        equal(activations, "type", "float".into())?;
        equal(activations, "strategy", "token".into())?;
        equal(activations, "symmetric", true.into())?;
        equal(activations, "dynamic", true.into())?;
        for key in [
            "actorder",
            "block_structure",
            "group_size",
            "observer",
            "scale_dtype",
            "zp_dtype",
        ] {
            ensure!(
                activations.get(key).is_none_or(Value::is_null),
                "unsupported quantization input_activations {key}"
            );
        }
        ensure!(
            activations
                .get("observer_kwargs")
                .is_none_or(|value| value == &serde_json::json!({})),
            "unsupported quantization input_activations observer_kwargs"
        );
        Ok(Self::Fp8W8a16)
    }

    /// The published ignore recipe: every tensor that stays in its stored dtype.
    fn expected_ignore(spec: &ModelSpec) -> Result<Vec<String>> {
        let vision = spec
            .vision
            .as_ref()
            .context("quantized checkpoints need the vision tower")?;
        let mut names = vec!["lm_head".to_owned()];
        for (layer, plan) in spec.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            if plan.mixer == Mixer::Gdn {
                names.push(format!("{prefix}.linear_attn"));
                for child in [
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "norm",
                    "out_proj",
                ] {
                    names.push(format!("{prefix}.linear_attn.{child}"));
                }
            }
            for suffix in [
                "mlp.gate",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert_gate",
            ] {
                names.push(format!("{prefix}.{suffix}"));
            }
        }
        names.push("model.merger.mlp.0.linear_1".to_owned());
        names.push("model.merger.mlp.0.linear_2".to_owned());
        for suffix in [
            "linear_1",
            "linear_2",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
        ] {
            names.push(format!("model.vision_tower.vit_merger.{suffix}"));
        }
        for layer in 0..vision.depth {
            for suffix in [
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
            ] {
                names.push(format!(
                    "model.vision_tower.encoder.layers.{layer}.{suffix}"
                ));
            }
        }
        names.sort_unstable();
        Ok(names)
    }

    /// Requires the published ignore list exactly, so an unexpected quantized role errors.
    fn check_ignore(&self, root: &Value, spec: &ModelSpec) -> Result<()> {
        if *self == Self::Bf16 {
            return Ok(());
        }
        let quant = root
            .get("quantization_config")
            .context("quantization_config missing")?;
        let ignore = quant
            .get("ignore")
            .and_then(Value::as_array)
            .context("quantization ignore missing")?;
        let mut actual = Vec::with_capacity(ignore.len());
        for entry in ignore {
            actual.push(
                entry
                    .as_str()
                    .context("quantization ignore entry is not a string")?
                    .to_owned(),
            );
        }
        actual.sort_unstable();
        let expected = Self::expected_ignore(spec)?;
        if let Some(entry) = expected
            .iter()
            .find(|entry| actual.binary_search(entry).is_err())
        {
            bail!("quantization ignore missing {entry}");
        }
        if let Some(entry) = actual
            .iter()
            .find(|entry| expected.binary_search(entry).is_err())
        {
            bail!("quantization ignore has unexpected entry {entry}");
        }
        ensure!(
            actual.len() == expected.len(),
            "quantization ignore repeats an entry"
        );
        Ok(())
    }
}
fn eos(v: &Value, vocab: u32) -> Result<Vec<u32>> {
    let values = match v {
        Value::Array(a) => a.clone(),
        Value::Number(_) => vec![v.clone()],
        _ => bail!("EOS must be an integer or array"),
    };
    let mut out = Vec::new();
    for value in values {
        let n = value.as_u64().context("EOS must be unsigned integer")?;
        ensure!(n < u64::from(vocab), "EOS outside vocabulary");
        out.push(n as u32);
    }
    out.sort_unstable();
    out.dedup();
    ensure!(!out.is_empty(), "empty EOS set");
    Ok(out)
}

struct TextGeometry {
    hidden: u32,
    intermediate: u32,
    vocab: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    q_width: u32,
    max_position: u32,
    rms_eps: f32,
    rope_theta: f32,
    rotary_dim: u32,
    layers: Vec<LayerPlan>,
    moe: Option<MoeConfig>,
    gdn: GdnConfig,
}

impl TextGeometry {
    fn read(t: &Value, model_type: &str) -> Result<Self> {
        equal(t, "model_type", model_type.into())?;
        if let Some(dtype) = t.get("dtype") {
            ensure!(
                dtype == &Value::String("bfloat16".into()),
                "unsupported dtype: {dtype:?}"
            );
        }
        for (key, val) in [("hidden_act", "silu"), ("mamba_ssm_dtype", "float32")] {
            equal(t, key, val.into())?;
        }
        if let Some(gate) = t.get("output_gate_type") {
            ensure!(
                gate == &Value::String("swish".into()),
                "unsupported output_gate_type: {gate:?}"
            );
        }
        equal(t, "attention_bias", false.into())?;
        equal(t, "attn_output_gate", true.into())?;
        equal(t, "attention_dropout", 0.0.into())?;
        equal(t, "tie_word_embeddings", false.into())?;
        if let Some(bias) = t.get("mlp_bias") {
            ensure!(
                bias == &Value::Bool(false),
                "unsupported mlp_bias: {bias:?}"
            );
        }
        let hidden = positive(t, "hidden_size")?;
        let vocab = positive(t, "vocab_size")?;
        let depth = positive(t, "num_hidden_layers")?;
        let q_heads = positive(t, "num_attention_heads")?;
        let kv_heads = positive(t, "num_key_value_heads")?;
        let head_dim = positive(t, "head_dim")?;
        let max_position = positive(t, "max_position_embeddings")?;
        ensure!(
            q_heads.is_multiple_of(kv_heads),
            "num_attention_heads is not a multiple of num_key_value_heads"
        );
        let interval = positive(t, "full_attention_interval")?;
        let eps = t
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .context("missing/invalid rms_norm_eps")?;
        ensure!(
            eps.is_finite() && eps > 0.0,
            "unsupported rms_norm_eps: {eps}"
        );
        let rope = t
            .get("rope_parameters")
            .context("missing rope_parameters")?;
        equal(rope, "rope_type", "default".into())?;
        equal(rope, "mrope_interleaved", true.into())?;
        equal(rope, "mrope_section", serde_json::json!([11, 11, 10]))?;
        let theta = positive(rope, "rope_theta")?;
        let top_partial = t.get("partial_rotary_factor");
        let rope_partial = rope.get("partial_rotary_factor");
        if let (Some(a), Some(b)) = (top_partial, rope_partial) {
            ensure!(
                a == b,
                "partial_rotary_factor differs between text_config and rope_parameters"
            );
        }
        let partial = top_partial
            .or(rope_partial)
            .and_then(Value::as_f64)
            .context("missing partial_rotary_factor")?;
        ensure!(
            partial > 0.0 && partial <= 1.0,
            "unsupported partial_rotary_factor: {partial}"
        );
        let rotary = f64::from(head_dim) * partial;
        ensure!(
            rotary.fract() == 0.0 && rotary >= 1.0,
            "partial_rotary_factor does not yield an integer rotary_dim: {partial}"
        );
        let types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        ensure!(
            types.len() == depth as usize,
            "layer_types length differs from num_hidden_layers"
        );
        let mut counts = [0u32, 0u32];
        let mut layers = Vec::with_capacity(types.len());
        for (i, kind) in types.iter().enumerate() {
            let full = (i as u32 + 1).is_multiple_of(interval);
            let (mixer, slot, expected) = if full {
                (Mixer::FullAttention, 0, "full_attention")
            } else {
                (Mixer::Gdn, 1, "linear_attention")
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported mixer at layer {i}: {kind:?}"
            );
            layers.push(LayerPlan {
                mixer,
                state_index: counts[slot],
            });
            counts[slot] += 1;
        }
        let gdn = GdnConfig {
            key_heads: positive(t, "linear_num_key_heads")?,
            value_heads: positive(t, "linear_num_value_heads")?,
            key_head_dim: positive(t, "linear_key_head_dim")?,
            value_head_dim: positive(t, "linear_value_head_dim")?,
            conv_kernel: positive(t, "linear_conv_kernel_dim")?,
        };
        ensure!(
            gdn.value_heads.is_multiple_of(gdn.key_heads),
            "linear_num_value_heads is not a multiple of linear_num_key_heads"
        );
        let moe = if model_type == "qwen3_5_moe_text" {
            equal(t, "output_router_logits", false.into())?;
            ensure!(
                t.get("intermediate_size").is_none(),
                "unexpected intermediate_size on a MoE text tower"
            );
            let experts = positive(t, "num_experts")?;
            let experts_per_tok = positive(t, "num_experts_per_tok")?;
            ensure!(
                experts_per_tok <= experts,
                "num_experts_per_tok exceeds num_experts"
            );
            Some(MoeConfig {
                experts,
                experts_per_tok,
                expert_intermediate: positive(t, "moe_intermediate_size")?,
                shared_intermediate: positive(t, "shared_expert_intermediate_size")?,
            })
        } else {
            for key in [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "output_router_logits",
            ] {
                ensure!(
                    t.get(key).is_none(),
                    "unexpected {key} on a dense text tower"
                );
            }
            None
        };
        let intermediate = match &moe {
            Some(moe) => moe.expert_intermediate,
            None => positive(t, "intermediate_size")?,
        };
        let q_width = q_heads
            .checked_mul(head_dim)
            .context("Q width overflows u32")?;
        Ok(Self {
            hidden,
            intermediate,
            vocab,
            q_heads,
            kv_heads,
            head_dim,
            q_width,
            max_position,
            rms_eps: eps as f32,
            rope_theta: theta as f32,
            rotary_dim: rotary as u32,
            layers,
            moe,
            gdn,
        })
    }
}

impl ModelSpec {
    /// `Config` view for the model/CLI surface; does not relax Config::read's Llama validation.
    pub fn config(&self) -> crate::config::Config {
        crate::config::Config {
            model_type: match self.architecture {
                Architecture::Llama => "llama",
                Architecture::Qwen35Text => "qwen3_5_text",
                Architecture::MiniCpmV47 => "qwen3_5_moe_text",
            }
            .into(),
            hidden_size: self.hidden as usize,
            intermediate_size: self.intermediate as usize,
            num_hidden_layers: self.layers.len(),
            num_attention_heads: self.q_heads as usize,
            num_key_value_heads: self.kv_heads as usize,
            head_dim: self.head_dim as usize,
            vocab_size: self.vocab as usize,
            max_position_embeddings: self.max_position as usize,
            rms_norm_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            rope_scaling: None,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            eos_token_id: self.eos.clone(),
            attention_bias: false,
            mlp_bias: false,
        }
    }
    pub fn read(dir: &Path) -> Result<Self> {
        let root: Value = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
        if root.get("model_type").and_then(Value::as_str) == Some("llama") {
            let c = crate::config::Config::read(dir)?;
            return Self::from_llama(&c);
        }
        let path = dir.join("generation_config.json");
        let generation = if path.exists() {
            serde_json::from_slice(&fs::read(path)?)?
        } else {
            Value::Null
        };
        if root.get("model_type").and_then(Value::as_str) == Some("minicpmv4_7") {
            Self::minicpm(&root, &generation)
        } else {
            Self::qwen(&root, &generation)
        }
    }
    pub fn from_llama(c: &crate::config::Config) -> Result<Self> {
        let u = |n: usize| u32::try_from(n).context("Llama dimension overflows u32");
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Llama,
            hidden: u(c.hidden_size)?,
            intermediate: u(c.intermediate_size)?,
            vocab: u(c.vocab_size)?,
            q_heads: u(c.num_attention_heads)?,
            kv_heads: u(c.num_key_value_heads)?,
            head_dim: u(c.head_dim)?,
            q_width: u(c.hidden_size)?,
            max_position: u(c.max_position_embeddings)?,
            rms_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            rotary_dim: u(c.head_dim)?,
            linear_format: LinearFormat::Bf16,
            expert_format: ExpertFormat::Bf16,
            eos: c.eos_token_id.clone(),
            vision: None,
            moe: None,
            gdn: None,
            layers: (0..u(c.num_hidden_layers)?)
                .map(|state_index| LayerPlan {
                    mixer: Mixer::FullAttention,
                    state_index,
                })
                .collect(),
        })
    }
    pub fn qwen(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "qwen3_5".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["Qwen3_5ForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        let t = root.get("text_config").context("missing text_config")?;
        let text = TextGeometry::read(t, "qwen3_5_text")?;
        Self::from_text(
            Architecture::Qwen35Text,
            text,
            t,
            generation,
            VisionConfig::from_root(root)?,
            LinearFormat::read(root)?,
            ExpertFormat::Bf16,
        )
    }
    pub fn minicpm(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "minicpmv4_7".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["MiniCPMV4_7ForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        equal(root, "dtype", "bfloat16".into())?;
        equal(root, "mrope_mode", "canvas".into())?;
        let t = root.get("text_config").context("missing text_config")?;
        // Declared by text_config; the checkpoint contains no mtp.* tensors.
        equal(t, "mtp_num_hidden_layers", 1.into())?;
        equal(t, "mtp_use_dedicated_embeddings", false.into())?;
        let text = TextGeometry::read(t, "qwen3_5_moe_text")?;
        let expert_format = ExpertFormat::read(root)?;
        let linear_format = match expert_format {
            ExpertFormat::Bf16 => LinearFormat::Bf16,
            ExpertFormat::Fp8W8a16 => LinearFormat::Fp8E4m3Channel,
        };
        let spec = Self::from_text(
            Architecture::MiniCpmV47,
            text,
            t,
            generation,
            VisionConfig::from_root(root)?,
            linear_format,
            expert_format,
        )?;
        expert_format.check_ignore(root, &spec)?;
        Ok(spec)
    }
    fn from_text(
        architecture: Architecture,
        text: TextGeometry,
        t: &Value,
        generation: &Value,
        vision: Option<VisionConfig>,
        linear_format: LinearFormat,
        expert_format: ExpertFormat,
    ) -> Result<Self> {
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture,
            hidden: text.hidden,
            intermediate: text.intermediate,
            vocab: text.vocab,
            q_heads: text.q_heads,
            kv_heads: text.kv_heads,
            head_dim: text.head_dim,
            q_width: text.q_width,
            max_position: text.max_position,
            rms_eps: text.rms_eps,
            rope_theta: text.rope_theta,
            rotary_dim: text.rotary_dim,
            linear_format,
            expert_format,
            eos: eos(eos_value, text.vocab)?,
            layers: text.layers,
            moe: text.moe,
            gdn: Some(text.gdn),
            vision,
        })
    }
}
