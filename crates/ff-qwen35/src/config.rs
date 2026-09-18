//! Qwen3.8-27B (dense `qwen3_5`) checkpoint configuration.
//!
//! Verified against `models/Qwen/Qwen3.8-27B/config.json` and the shard
//! index on 2026-09-17; see docs/qwen35-design.md. The checkpoint is all
//! BF16 — quantization is OURS (offline requant, groupwise affine int4,
//! group 64, the edge0 byte layout).

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

pub const QWEN35_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
pub const QWEN35_TEXT_MODEL_TYPE: &str = "qwen3_5_text";

/// Tensor prefix differs from Edge0: `model.language_model.layers.N.*`.
pub const TEXT_PREFIX: &str = "model.language_model";

pub const REQUANT_GROUP_SIZE: usize = 64;
pub const REQUANT_BITS: u32 = 4;

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
    /// Dense MLP width (no MoE in this family).
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    #[serde(default, rename = "rope_parameters")]
    pub rope: RopeConfig,
}

impl TextConfig {
    /// Same hybrid cadence as Edge0: `interval - 1` GDN layers between
    /// full-attention layers, first full layer at `interval - 1`.
    pub fn layer_kind(&self, index: usize) -> LayerKind {
        let interval = self.full_attention_interval.max(1);
        if (index + 1).is_multiple_of(interval) {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        }
    }

    pub fn conv_dim(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct VisionConfig {
    #[serde(default = "three")]
    pub in_channels: usize,
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
pub struct Qwen35Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
}

fn three() -> usize {
    3
}

/// The vision tower's rope theta: vision_config carries no rope_parameters,
/// HF resolves the hardcoded default 10000.0 (Qwen3_5VisionRotaryEmbedding).
pub const VISION_ROPE_THETA: f64 = 10_000.0;

impl Qwen35Config {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let raw = fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("config.json under {}", dir.display()))?;
        let cfg: Self = serde_json::from_str(&raw).context("parse config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.architectures.iter().any(|a| a == QWEN35_ARCHITECTURE),
            "unsupported architectures {:?}; expected {QWEN35_ARCHITECTURE}",
            self.architectures
        );
        let t = &self.text_config;
        ensure!(
            t.linear_conv_kernel_dim == 4,
            "conv kernel {}",
            t.linear_conv_kernel_dim
        );
        ensure!(
            t.linear_key_head_dim == 128 && t.linear_value_head_dim == 128,
            "gdn head dims {}x{}; kernels assume 128",
            t.linear_key_head_dim,
            t.linear_value_head_dim
        );
        ensure!(t.head_dim == 256, "head_dim {}", t.head_dim);
        if let Some(types) = &t.layer_types {
            ensure!(types.len() == t.num_hidden_layers, "layer_types length");
            for (i, kind) in types.iter().enumerate() {
                ensure!(
                    *kind == t.layer_kind(i),
                    "layer_types[{i}] contradicts full_attention_interval"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_real_checkpoint_config() {
        let dir = Path::new("../../models/Qwen/Qwen3.8-27B");
        if !dir.exists() {
            return;
        }
        let cfg = Qwen35Config::from_model_dir(dir).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 64);
        assert_eq!(t.hidden_size, 5120);
        assert_eq!(t.intermediate_size, 17408);
        assert_eq!(t.layer_types.as_ref().unwrap().len(), 64);
        assert_eq!(t.layer_kind(0), LayerKind::LinearAttention);
        assert_eq!(t.layer_kind(3), LayerKind::FullAttention);
        assert_eq!(t.conv_dim(), 10240);
        assert_eq!(t.mtp_num_hidden_layers, 1);
        assert!(!t.mtp_use_dedicated_embeddings);
        assert!(cfg.vision_config.is_some());
        let v = cfg.vision_config.as_ref().unwrap();
        assert_eq!(v.in_channels, 3);
        assert_eq!(v.out_hidden_size, 5120);
        assert_eq!(cfg.image_token_id, Some(248056));
        assert_eq!(cfg.vision_start_token_id, Some(248053));
        assert_eq!(cfg.vision_end_token_id, Some(248054));
        // The tower hardcodes tanh-gelu in blocks / erf-gelu in the merger
        // (vision.rs); fail loudly if the checkpoint ever changes that.
        let raw = fs::read_to_string(dir.join("config.json")).unwrap();
        assert!(raw.contains("\"gelu_pytorch_tanh\""));
    }
}
