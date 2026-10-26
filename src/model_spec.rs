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
    Flint,
    Glm5Next,
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
    #[serde(
        default = "RouterScore::softmax",
        skip_serializing_if = "RouterScore::is_softmax"
    )]
    pub score: RouterScore,
    pub shared_gate: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RouterScore {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}
impl RouterScore {
    fn softmax() -> Self {
        Self::Softmax
    }
    fn is_softmax(&self) -> bool {
        *self == Self::Softmax
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwigluRounding {
    Twice,
    Once,
    ClampedTwice,
    ClampedOnce,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedWeightBoundary {
    AfterDownBf16,
    BeforeContributionBf16,
    BeforeDownQuantization,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LinearFormat {
    Bf16,
    Fp8E4m3Block {
        block: [u32; 2],
        scale_dtype: safetensors::Dtype,
    },
    /// compressed-tensors FP8_DYNAMIC per-output-channel weights plus token activations.
    Fp8E4m3Channel,
}
/// Routed-expert checkpoint layout: fused or separate BF16 tensors, per-channel BF16 scales,
/// or block scales beside per-expert E4M3 weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExpertFormat {
    Bf16,
    Bf16Separate,
    Fp8W8a16,
    Fp8W8a16Block {
        block: [u32; 2],
        scale_dtype: safetensors::Dtype,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GdnConfig {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm5NextConfig {
    pub dense_layers: u32,
    pub nextn_layers: u32,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub index_kpool: u32,
    pub gate_lower_bound: f32,
    pub swiglu_limit: f32,
    pub routed_scaling_factor: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerPlan {
    pub mixer: Mixer,
    pub state_index: u32,
    pub rope: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<u32>,
    pub ffn: FfnKind,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FfnKind {
    Dense,
    Moe,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_format: Option<ExpertFormat>,
    pub eos: Vec<u32>,
    pub layers: Vec<LayerPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdn: Option<GdnConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glm5_next: Option<Glm5NextConfig>,
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
            Architecture::Llama | Architecture::Flint | Architecture::Glm5Next => {
                bail!("vision configuration requires a supported multimodal architecture")
            }
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
fn integer_array(v: &Value, key: &str, limit: u32) -> Result<Vec<u32>> {
    let values = v
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing/invalid {key}"))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let n = value
            .as_u64()
            .with_context(|| format!("{key} contains a non-unsigned integer"))?;
        let n = u32::try_from(n).with_context(|| format!("{key} overflows u32"))?;
        ensure!(n < limit, "{key} contains out-of-range layer {n}");
        out.push(n);
    }
    Ok(out)
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
    pub fn fp8_block(block: [u32; 2], scale_dtype: safetensors::Dtype) -> Result<Self> {
        crate::fp8::Fp8BlockScale::new(block, scale_dtype)?;
        Ok(Self::Fp8E4m3Block { block, scale_dtype })
    }

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
        Self::fp8_block([128, 128], safetensors::Dtype::BF16)
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
                rope: full,
                sliding_window: None,
                ffn: if model_type == "qwen3_5_moe_text" {
                    FfnKind::Moe
                } else {
                    FfnKind::Dense
                },
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
                score: RouterScore::Softmax,
                shared_gate: true,
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
    pub fn swiglu_rounding(&self) -> SwigluRounding {
        match self.architecture {
            Architecture::Flint => SwigluRounding::Once,
            Architecture::Glm5Next => SwigluRounding::ClampedTwice,
            Architecture::Llama | Architecture::Qwen35Text | Architecture::MiniCpmV47 => {
                SwigluRounding::Twice
            }
        }
    }

    pub fn swiglu_limit(&self) -> f32 {
        self.glm5_next
            .as_ref()
            .map_or(0.0, |config| config.swiglu_limit)
    }

    pub fn routed_weight_boundary(&self) -> RoutedWeightBoundary {
        match self.architecture {
            Architecture::Flint | Architecture::Glm5Next => {
                RoutedWeightBoundary::BeforeContributionBf16
            }
            Architecture::Llama | Architecture::Qwen35Text | Architecture::MiniCpmV47 => {
                RoutedWeightBoundary::AfterDownBf16
            }
        }
    }

    /// `Config` view for the model/CLI surface; does not relax Config::read's Llama validation.
    pub fn config(&self) -> crate::config::Config {
        crate::config::Config {
            model_type: match self.architecture {
                Architecture::Llama => "llama",
                Architecture::Qwen35Text => "qwen3_5_text",
                Architecture::MiniCpmV47 => "qwen3_5_moe_text",
                Architecture::Flint => "flint",
                Architecture::Glm5Next => "glm5_next_text",
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
        } else if root.get("model_type").and_then(Value::as_str) == Some("flint") {
            Self::flint(&root, &generation)
        } else if root.get("model_type").and_then(Value::as_str) == Some("glm5_next") {
            Self::glm5_next(&root, &generation)
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
            expert_format: None,
            eos: c.eos_token_id.clone(),
            vision: None,
            moe: None,
            gdn: None,
            glm5_next: None,
            layers: (0..u(c.num_hidden_layers)?)
                .map(|state_index| LayerPlan {
                    mixer: Mixer::FullAttention,
                    state_index,
                    rope: true,
                    sliding_window: None,
                    ffn: FfnKind::Dense,
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
            None,
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
            ExpertFormat::Bf16 | ExpertFormat::Bf16Separate => LinearFormat::Bf16,
            ExpertFormat::Fp8W8a16 => LinearFormat::Fp8E4m3Channel,
            ExpertFormat::Fp8W8a16Block { .. } => {
                bail!("MiniCPM does not use F32 block-scale experts")
            }
        };
        let spec = Self::from_text(
            Architecture::MiniCpmV47,
            text,
            t,
            generation,
            VisionConfig::from_root(root)?,
            linear_format,
            Some(expert_format),
        )?;
        expert_format.check_ignore(root, &spec)?;
        Ok(spec)
    }
    pub fn flint(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "flint".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["FlintForCausalLM"]),
        )?;
        exact_keys(
            root,
            &[
                "architectures",
                "attention_bias",
                "attention_dropout",
                "bos_token_id",
                "decoder_sparse_step",
                "eos_token_id",
                "head_dim",
                "hidden_act",
                "hidden_size",
                "initializer_range",
                "intermediate_size",
                "layer_types",
                "max_position_embeddings",
                "model_type",
                "mlp_only_layers",
                "moe_intermediate_size",
                "norm_topk_prob",
                "num_attention_heads",
                "num_experts",
                "num_experts_per_tok",
                "num_hidden_layers",
                "num_key_value_heads",
                "no_rope_layer_indices",
                "output_router_logits",
                "pad_token_id",
                "rms_norm_eps",
                "rope_theta",
                "router_aux_loss_coef",
                "router_score_function",
                "shared_expert_intermediate_size",
                "sliding_window",
                "softmax_type",
                "tie_word_embeddings",
                "torch_dtype",
                "transformers_version",
                "use_cache",
                "use_sliding_window",
                "vocab_size",
            ],
            "flint config",
        )?;
        for (key, value) in [
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("bos_token_id", 150000.into()),
            ("decoder_sparse_step", 1.into()),
            ("hidden_act", "silu".into()),
            ("initializer_range", 0.02.into()),
            ("norm_topk_prob", true.into()),
            ("output_router_logits", false.into()),
            ("pad_token_id", 150001.into()),
            ("router_aux_loss_coef", 0.0005.into()),
            ("router_score_function", "sigmoid".into()),
            ("softmax_type", "learnable".into()),
            ("tie_word_embeddings", false.into()),
            ("torch_dtype", "bfloat16".into()),
            ("transformers_version", "5.12.1".into()),
            ("use_cache", true.into()),
            ("use_sliding_window", true.into()),
        ] {
            equal(root, key, value)?;
        }
        let hidden = positive(root, "hidden_size")?;
        let head_dim = positive(root, "head_dim")?;
        let q_heads = positive(root, "num_attention_heads")?;
        let kv_heads = positive(root, "num_key_value_heads")?;
        let rope_theta = root
            .get("rope_theta")
            .and_then(Value::as_f64)
            .context("missing/invalid rope_theta")?;
        ensure!(
            [
                hidden,
                positive(root, "intermediate_size")?,
                positive(root, "max_position_embeddings")?,
                positive(root, "vocab_size")?,
                head_dim,
                q_heads,
                kv_heads,
            ] == [2048, 6144, 262144, 150144, 128, 32, 4]
                && rope_theta == 10000.0,
            "unsupported Flint attention geometry"
        );
        ensure!(
            q_heads.is_multiple_of(kv_heads),
            "Flint Q heads are not a multiple of KV heads"
        );
        let depth = positive(root, "num_hidden_layers")?;
        ensure!(depth == 49, "unsupported Flint depth {depth}");
        let sliding_window = positive(root, "sliding_window")?;
        ensure!(
            sliding_window == 513,
            "unsupported sliding_window {sliding_window}"
        );
        equal(root, "mlp_only_layers", serde_json::json!([0, 1]))?;
        let no_rope = integer_array(root, "no_rope_layer_indices", depth)?;
        ensure!(
            no_rope == [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48],
            "unsupported Flint no-rope layer set"
        );
        let types = root
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing/invalid layer_types")?;
        ensure!(types.len() == depth as usize, "invalid layer_types length");
        let mut layers = Vec::with_capacity(types.len());
        for (i, kind) in types.iter().enumerate() {
            let full = no_rope.contains(&(i as u32));
            let expected = if full {
                "full_attention"
            } else {
                "sliding_attention"
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported Flint layer type at {i}: {kind:?}"
            );
            layers.push(LayerPlan {
                mixer: Mixer::FullAttention,
                state_index: u32::try_from(i).context("Flint layer index overflows u32")?,
                rope: !full,
                sliding_window: (!full).then_some(sliding_window),
                ffn: if i < 2 { FfnKind::Dense } else { FfnKind::Moe },
            });
        }
        let moe = MoeConfig {
            experts: positive(root, "num_experts")?,
            experts_per_tok: positive(root, "num_experts_per_tok")?,
            expert_intermediate: positive(root, "moe_intermediate_size")?,
            shared_intermediate: positive(root, "shared_expert_intermediate_size")?,
            score: RouterScore::Sigmoid,
            shared_gate: false,
        };
        ensure!(
            [
                moe.experts,
                moe.experts_per_tok,
                moe.expert_intermediate,
                moe.shared_intermediate
            ] == [256, 6, 512, 1024],
            "unsupported Flint MoE geometry"
        );
        let eps = root
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .context("missing/invalid rms_norm_eps")?;
        ensure!(eps == 1e-6, "unsupported rms_norm_eps {eps}");
        let vocab = positive(root, "vocab_size")?;
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| root.get("eos_token_id"))
            .context("missing Flint EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Flint,
            hidden,
            intermediate: positive(root, "intermediate_size")?,
            vocab,
            q_heads,
            kv_heads,
            head_dim,
            q_width: q_heads * head_dim,
            max_position: positive(root, "max_position_embeddings")?,
            rms_eps: eps as f32,
            rope_theta: rope_theta as f32,
            rotary_dim: head_dim,
            linear_format: LinearFormat::Bf16,
            expert_format: Some(ExpertFormat::Bf16Separate),
            eos: eos(eos_value, vocab)?,
            layers,
            moe: Some(moe),
            gdn: None,
            glm5_next: None,
            vision: None,
        })
    }
    pub fn glm5_next(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "glm5_next".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["Glm5NextForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        equal(root, "transformers_version", "5.16.0".into())?;
        let quant = root
            .get("quantization_config")
            .context("GLM-5.3 quantization_config missing")?;
        exact_keys(
            quant,
            &[
                "activation_scheme",
                "fmt",
                "modules_to_not_convert",
                "quant_method",
                "weight_block_size",
            ],
            "quantization_config",
        )?;
        for (key, value) in [
            ("activation_scheme", "dynamic"),
            ("fmt", "e4m3"),
            ("quant_method", "fp8"),
        ] {
            equal(quant, key, value.into())?;
        }
        equal(quant, "weight_block_size", serde_json::json!([128, 128]))?;
        ensure!(
            quant
                .get("modules_to_not_convert")
                .and_then(Value::as_array)
                .is_some_and(|entries| !entries.is_empty() && entries.iter().all(Value::is_string)),
            "quantization modules_to_not_convert must be a nonempty string array"
        );

        let t = root.get("text_config").context("missing text_config")?;
        exact_keys(
            t,
            &[
                "attention_bias",
                "attention_dropout",
                "dtype",
                "eos_token_id",
                "first_k_dense_replace",
                "hc_eps",
                "hc_mult",
                "hc_sinkhorn_iters",
                "head_dim",
                "hidden_act",
                "hidden_size",
                "index_head_dim",
                "index_kpool",
                "index_kpool_always_select_tail",
                "index_kpool_compress",
                "index_n_heads",
                "index_share_for_mtp_iteration",
                "index_topk",
                "indexer_rope_interleave",
                "indexer_types",
                "initializer_range",
                "intermediate_size",
                "kv_lora_rank",
                "layer_types",
                "linear_attn_config",
                "max_position_embeddings",
                "mhc",
                "mla_use_nope",
                "mlp_layer_types",
                "model_type",
                "moe_intermediate_size",
                "moe_router_dtype",
                "n_group",
                "n_routed_experts",
                "n_shared_experts",
                "norm_topk_prob",
                "num_attention_heads",
                "num_experts_per_tok",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_nextn_predict_layers",
                "output_router_logits",
                "pad_token_id",
                "q_lora_rank",
                "qk_head_dim",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "rms_norm_eps",
                "routed_scaling_factor",
                "router_aux_loss_coef",
                "scoring_func",
                "swiglu_limit",
                "tie_word_embeddings",
                "topk_group",
                "topk_method",
                "use_cache",
                "v_head_dim",
                "vocab_size",
            ],
            "text_config",
        )?;
        equal(t, "model_type", "glm5_next_text".into())?;
        for (key, value) in [
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("dtype", "bfloat16".into()),
            ("hidden_act", "silu".into()),
            ("mhc", true.into()),
            ("mla_use_nope", true.into()),
            ("moe_router_dtype", "float32".into()),
            ("norm_topk_prob", true.into()),
            ("output_router_logits", false.into()),
            ("scoring_func", "sigmoid".into()),
            ("tie_word_embeddings", false.into()),
            ("topk_method", "noaux_tc".into()),
            ("use_cache", true.into()),
            ("index_kpool_always_select_tail", true.into()),
            ("index_kpool_compress", true.into()),
            ("index_share_for_mtp_iteration", true.into()),
            ("indexer_rope_interleave", true.into()),
        ] {
            equal(t, key, value)?;
        }
        for (key, value) in [
            ("head_dim", 0),
            ("n_group", 1),
            ("n_shared_experts", 1),
            ("num_nextn_predict_layers", 1),
            ("qk_rope_head_dim", 0),
            ("topk_group", 1),
        ] {
            equal(t, key, value.into())?;
        }
        let hidden = positive(t, "hidden_size")?;
        let vocab = positive(t, "vocab_size")?;
        let depth = positive(t, "num_hidden_layers")?;
        let dense_layers = positive(t, "first_k_dense_replace")?;
        ensure!(
            dense_layers < depth,
            "first_k_dense_replace={dense_layers} must be below num_hidden_layers={depth}"
        );
        let q_heads = positive(t, "num_attention_heads")?;
        let kv_heads = positive(t, "num_key_value_heads")?;
        let qk_head_dim = positive(t, "qk_head_dim")?;
        let qk_nope_head_dim = positive(t, "qk_nope_head_dim")?;
        ensure!(
            qk_head_dim == qk_nope_head_dim && q_heads == kv_heads,
            "GLM-5.3 requires NoPE Q/K and equal Q/KV head counts"
        );
        let max_position = positive(t, "max_position_embeddings")?;
        let q_width = q_heads
            .checked_mul(qk_head_dim)
            .context("GLM-5.3 Q width overflow")?;
        let intermediate = positive(t, "intermediate_size")?;
        let expert_intermediate = positive(t, "moe_intermediate_size")?;
        let experts = positive(t, "n_routed_experts")?;
        let experts_per_tok = positive(t, "num_experts_per_tok")?;
        ensure!(
            experts_per_tok <= experts,
            "num_experts_per_tok exceeds n_routed_experts"
        );
        let shared_intermediate = expert_intermediate
            .checked_mul(positive(t, "n_shared_experts")?)
            .context("shared expert width overflow")?;
        let float = |key: &str| -> Result<f64> {
            t.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing/invalid {key}"))
        };
        let positive_f32 = |key: &str| -> Result<f32> {
            let value = float(key)? as f32;
            ensure!(value.is_finite() && value > 0.0, "invalid {key}");
            Ok(value)
        };
        for key in [
            "hc_eps",
            "initializer_range",
            "rms_norm_eps",
            "routed_scaling_factor",
            "swiglu_limit",
        ] {
            positive_f32(key)?;
        }
        ensure!(
            float("router_aux_loss_coef")?.is_finite() && float("router_aux_loss_coef")? >= 0.0,
            "invalid router_aux_loss_coef"
        );
        let rms_eps = positive_f32("rms_norm_eps")?;
        let hc_eps = positive_f32("hc_eps")?;
        let swiglu_limit = positive_f32("swiglu_limit")?;
        let routed_scaling_factor = positive_f32("routed_scaling_factor")?;
        let hc_mult = positive(t, "hc_mult")?;
        let hc_sinkhorn_iters = positive(t, "hc_sinkhorn_iters")?;
        ensure!(hc_mult == 4, "unsupported GLM-5.3 hc_mult={hc_mult}");
        let q_lora_rank = positive(t, "q_lora_rank")?;
        let kv_lora_rank = positive(t, "kv_lora_rank")?;
        let v_head_dim = positive(t, "v_head_dim")?;
        let index_heads = positive(t, "index_n_heads")?;
        let index_head_dim = positive(t, "index_head_dim")?;
        let index_topk = positive(t, "index_topk")?;
        let index_kpool = positive(t, "index_kpool")?;
        ensure!(
            index_topk.is_multiple_of(index_kpool),
            "index_topk is not an index_kpool multiple"
        );

        let linear = t
            .get("linear_attn_config")
            .context("missing linear_attn_config")?;
        exact_keys(
            linear,
            &[
                "num_heads",
                "gate_lower_bound",
                "head_dim",
                "short_conv_kernel_size",
                "kda_layers",
                "full_attn_layers",
            ],
            "linear_attn_config",
        )?;
        let linear_heads = positive(linear, "num_heads")?;
        let linear_head_dim = positive(linear, "head_dim")?;
        equal(linear, "short_conv_kernel_size", 4.into())?;
        let gate_lower_bound = linear
            .get("gate_lower_bound")
            .and_then(Value::as_f64)
            .context("missing/invalid gate_lower_bound")?;
        ensure!(
            gate_lower_bound.is_finite() && gate_lower_bound <= 0.0,
            "invalid gate_lower_bound"
        );
        let gate_lower_bound = gate_lower_bound as f32;
        ensure!(gate_lower_bound.is_finite(), "invalid gate_lower_bound");
        let layer_types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        let mlp_types = t
            .get("mlp_layer_types")
            .and_then(Value::as_array)
            .context("missing mlp_layer_types")?;
        let indexer_types = t
            .get("indexer_types")
            .and_then(Value::as_array)
            .context("missing indexer_types")?;
        ensure!(
            layer_types.len() == depth as usize
                && mlp_types.len() == depth as usize
                && indexer_types.len() == depth as usize,
            "GLM-5.3 layer type extent differs from num_hidden_layers"
        );
        let mut kda_layers = Vec::new();
        let mut full_attn_layers = Vec::new();
        let mut counts = [0u32; 2];
        let mut layers = Vec::with_capacity(depth as usize);
        for (i, kind) in layer_types.iter().enumerate() {
            let sparse = (i + 1).is_multiple_of(4);
            let expected = if sparse {
                "deepseek_sparse_attention"
            } else {
                "linear_attention"
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported mixer at layer {i}"
            );
            if sparse {
                full_attn_layers.push(i);
            } else {
                kda_layers.push(i);
            }
            let expected_ffn = if i < dense_layers as usize {
                "dense"
            } else {
                "sparse"
            };
            ensure!(
                mlp_types[i].as_str() == Some(expected_ffn),
                "unsupported FFN at layer {i}"
            );
            ensure!(
                indexer_types[i].as_str() == Some("full"),
                "unsupported indexer at layer {i}"
            );
            let (mixer, slot) = if sparse {
                (Mixer::FullAttention, 0)
            } else {
                (Mixer::Gdn, 1)
            };
            layers.push(LayerPlan {
                mixer,
                state_index: counts[slot],
                rope: false,
                sliding_window: None,
                ffn: if i < dense_layers as usize {
                    FfnKind::Dense
                } else {
                    FfnKind::Moe
                },
            });
            counts[slot] += 1;
        }
        ensure!(kda_layers.len() + full_attn_layers.len() == depth as usize);
        equal(linear, "kda_layers", serde_json::to_value(&kda_layers)?)?;
        equal(
            linear,
            "full_attn_layers",
            serde_json::to_value(&full_attn_layers)?,
        )?;
        ensure!(
            counts == [11, 34],
            "unsupported GLM-5.3 mixer counts {counts:?}"
        );
        let pad = integer(t, "pad_token_id")?;
        ensure!(pad < vocab, "pad_token_id outside vocabulary");
        let eos_value = generation
            .get("eos_token_id")
            .filter(|value| !value.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Glm5Next,
            hidden,
            intermediate,
            vocab,
            q_heads,
            kv_heads,
            head_dim: qk_head_dim,
            q_width,
            max_position,
            rms_eps,
            rope_theta: 1.0,
            rotary_dim: 0,
            linear_format: LinearFormat::fp8_block([128, 128], safetensors::Dtype::F32)?,
            expert_format: Some(ExpertFormat::Fp8W8a16Block {
                block: [128, 128],
                scale_dtype: safetensors::Dtype::F32,
            }),
            eos: eos(eos_value, vocab)?,
            layers,
            moe: Some(MoeConfig {
                experts,
                experts_per_tok,
                expert_intermediate,
                shared_intermediate,
                score: RouterScore::Sigmoid,
                shared_gate: false,
            }),
            gdn: Some(GdnConfig {
                key_heads: linear_heads,
                value_heads: linear_heads,
                key_head_dim: linear_head_dim,
                value_head_dim: linear_head_dim,
                conv_kernel: 4,
            }),
            glm5_next: Some(Glm5NextConfig {
                dense_layers,
                nextn_layers: positive(t, "num_nextn_predict_layers")?,
                hc_mult,
                hc_sinkhorn_iters,
                hc_eps,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                v_head_dim,
                index_heads,
                index_head_dim,
                index_topk,
                index_kpool,
                gate_lower_bound,
                swiglu_limit,
                routed_scaling_factor,
            }),
            vision: None,
        })
    }
    fn from_text(
        architecture: Architecture,
        text: TextGeometry,
        t: &Value,
        generation: &Value,
        vision: Option<VisionConfig>,
        linear_format: LinearFormat,
        expert_format: Option<ExpertFormat>,
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
            glm5_next: None,
            vision,
        })
    }
}
