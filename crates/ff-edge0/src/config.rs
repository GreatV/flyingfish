//! Edge0 groupwise-quantized Qwen3.5-MoE checkpoint configuration.

use anyhow::{Context, Result, ensure};
use ff_core::quant::QuantFormat;
use serde::Deserialize;
use std::{fs, path::Path};

pub const EDGE0_ARCHITECTURE: &str = "Qwen3_5MoeForConditionalGeneration";
pub const EDGE0_MODEL_TYPE: &str = "qwen3_5_moe";

/// `quantization.mode` value in the checkpoint's config.json; an upstream
/// BF16 Qwen3.5-MoE shares the architecture but carries no such object.
pub const EDGE0_QUANTIZATION_MODE: &str = "affine";

/// Groupwise affine quantization, verified from the checkpoint: body
/// tensors carry 8x unsigned int4 per U32 word (low nibble first), the
/// router and shared-expert gates carry 4x unsigned int8.
pub const EDGE0_BODY_BITS: u32 = 4;

/// Runtime routing width. The base model's trained width
/// (`num_experts_per_tok`) is 8; the official edge0 pipeline truncates to 4
/// (upstream `models/edge0_35b`: `top_k = 4`, `prerouter_top_k = 4`), and
/// the reference decode throughput only closes arithmetically at 4.
pub const EDGE0_RUNTIME_TOP_K: usize = 4;

/// Quantized projections pair `{projection}.weight` (U32-packed payload)
/// with `{projection}.scales` / `{projection}.biases` — the parameters hang
/// off the projection name, never off `.weight` (verified across the
/// self_attn, linear_attn, switch_mlp, shared_expert and embed families).

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct RopeConfig {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            mrope_section: [11, 11, 10],
            mrope_interleaved: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub full_attention_interval: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerKind>>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default, rename = "rope_parameters")]
    pub rope: RopeConfig,
}

impl TextConfig {
    /// Hybrid layout: `full_attention_interval - 1` linear-attention
    /// layers between full-attention layers, first full layer at index
    /// `interval - 1` (verified list starts linear, linear, linear, full).
    pub fn layer_kind(&self, index: usize) -> LayerKind {
        let interval = self.full_attention_interval.max(1);
        if (index + 1).is_multiple_of(interval) {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        }
    }

    pub fn full_attention_layers(&self) -> Vec<usize> {
        (0..self.num_hidden_layers)
            .filter(|&i| self.layer_kind(i) == LayerKind::FullAttention)
            .collect()
    }

    /// Routing width in force at run time: the pipeline truncates the base
    /// model's trained width to the official runtime width.
    pub fn effective_top_k(&self) -> usize {
        self.num_experts_per_tok.min(EDGE0_RUNTIME_TOP_K)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Edge0Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    /// HF writes eos_token_id as one id or a list.
    #[serde(deserialize_with = "token_id_list")]
    pub eos_token_id: Vec<u32>,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    #[serde(rename = "quantization", deserialize_with = "affine_int4")]
    pub quant: QuantFormat,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
}

fn affine_int4<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<QuantFormat, D::Error> {
    use serde::de::Error;
    #[derive(Deserialize)]
    struct Raw {
        group_size: usize,
        bits: u32,
        mode: String,
    }
    let raw = Raw::deserialize(deserializer)?;
    if raw.mode != EDGE0_QUANTIZATION_MODE {
        return Err(D::Error::custom(format!(
            "unsupported Edge0 quantization mode {:?}; expected {EDGE0_QUANTIZATION_MODE}",
            raw.mode
        )));
    }
    if raw.bits != EDGE0_BODY_BITS {
        return Err(D::Error::custom(format!(
            "unsupported Edge0 body bits {}; expected {EDGE0_BODY_BITS}",
            raw.bits
        )));
    }
    if raw.group_size != crate::int4::GROUP_SIZE {
        return Err(D::Error::custom(format!(
            "unsupported Edge0 group size {}; expected {}",
            raw.group_size,
            crate::int4::GROUP_SIZE
        )));
    }
    Ok(QuantFormat::GroupAffine {
        group: raw.group_size,
    })
}

fn token_id_list<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<u32>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        One(u32),
        Many(Vec<u32>),
    }
    Ok(match Repr::deserialize(deserializer)? {
        Repr::One(id) => vec![id],
        Repr::Many(ids) => ids,
    })
}

/// The checkpoint's chat template: one user turn, then an assistant think block.
pub fn chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

impl Edge0Config {
    pub fn detects(architectures: &[String], model_type: &str) -> bool {
        architectures.iter().any(|a| a == EDGE0_ARCHITECTURE) || model_type == EDGE0_MODEL_TYPE
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Edge0Config = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            Self::detects(&self.architectures, &self.model_type),
            "not an Edge0 checkpoint: architectures {:?}, model_type {:?}",
            self.architectures,
            self.model_type
        );
        ensure!(
            !self.eos_token_id.is_empty()
                && self
                    .eos_token_id
                    .iter()
                    .all(|&id| (id as usize) < self.text_config.vocab_size),
            "invalid EOS token ids {:?}",
            self.eos_token_id
        );
        if let Some(list) = &self.text_config.layer_types {
            ensure!(
                list.len() == self.text_config.num_hidden_layers,
                "layer_types length {} disagrees with num_hidden_layers {}",
                list.len(),
                self.text_config.num_hidden_layers
            );
            for (index, kind) in list.iter().enumerate() {
                ensure!(
                    *kind == self.text_config.layer_kind(index),
                    "layer_types[{index}] disagrees with full_attention_interval {}",
                    self.text_config.full_attention_interval
                );
            }
        }
        ensure!(
            self.text_config.effective_top_k() > 0,
            "routing width must be positive"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_layout_matches_the_verified_checkpoint() {
        let config: Edge0Config =
            serde_json::from_str(include_str!("testdata/config.json")).unwrap();
        assert_eq!(config.text_config.num_hidden_layers, 40);
        assert_eq!(
            config.text_config.full_attention_layers(),
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        );
        assert_eq!(config.text_config.num_experts_per_tok, 8);
        assert_eq!(config.text_config.effective_top_k(), 4);
        assert_eq!(config.text_config.num_experts, 256);
        config.validate().unwrap();
        assert_eq!(config.quant, QuantFormat::GroupAffine { group: 64 });
    }

    #[test]
    #[ignore = "requires FF_MODELS_DIR with Edge0/Edge0-35B-A3B-preview"]
    fn parses_the_real_checkpoint_config() {
        let dir = ff_core::paths::checkpoint_dir("Edge0/Edge0-35B-A3B-preview")
            .filter(|dir| dir.is_dir())
            .expect("requires FF_MODELS_DIR with Edge0/Edge0-35B-A3B-preview");
        let dir = &dir;
        let config = Edge0Config::from_model_dir(dir).unwrap();
        assert_eq!(config.quant, QuantFormat::GroupAffine { group: 64 });
    }

    #[test]
    fn an_unsupported_group_size_names_the_value() {
        let mut config: serde_json::Value =
            serde_json::from_str(include_str!("testdata/config.json")).unwrap();
        config["quantization"]["group_size"] = serde_json::json!(32);
        let error = serde_json::from_value::<Edge0Config>(config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("group size 32"), "{error}");
    }
}
