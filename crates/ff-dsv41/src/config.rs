//! DeepSeek-V4.1-Flash checkpoint configuration.
//!
//! Field names mirror `config.json` in the checkpoint; the constants and
//! defaults were verified against
//! `models/deepseek-ai/DeepSeek-V4.1-Flash/config.json` on 2026-09-20.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

pub const DSV41_ARCHITECTURE: &str = "DeepseekV41ForCausalLM";
pub const DSV41_MODEL_TYPE: &str = "deepseek_v41";

/// One FP8 scale per `weight_block_size` square; routed experts carry one
/// E8M0 scale per 32 elements along the reduction axis.
pub const FP8_WEIGHT_BLOCK: usize = 32;
pub const FP4_WEIGHT_BLOCK: usize = 32;

#[derive(Clone, Debug, Deserialize)]
pub struct DeepseekV41Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    #[serde(default = "default_pad")]
    pub pad_token_id: u32,
    pub image_token_id: u32,
    pub quantization_config: QuantizationConfig,
    pub text_config: TextConfig,
    pub vision_config: Option<VisionConfig>,
}

fn default_bos() -> u32 {
    0
}
fn default_eos() -> u32 {
    1
}
fn default_pad() -> u32 {
    2
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_block_size: [usize; 2],
    pub scale_fmt: String,
    pub expert_dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub original_max_position_embeddings: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub initializer_range: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub sliding_window: usize,
    pub compress_ratios: Vec<u8>,
    pub compress_rope_theta: f64,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer_id: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<u64>,
    pub engram_max_ngram_size: usize,
    pub engram_vocab_size: u64,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_pad_token_id: u32,
    pub engram_compressed_vocab_size: u32,
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub dspark_block_size: usize,
    #[serde(default)]
    pub dspark_noise_token_id: u32,
    #[serde(default)]
    pub dspark_target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub dspark_markov_rank: usize,
    #[serde(default)]
    pub dspark_n_routed_experts: usize,
    #[serde(default)]
    pub dspark_num_experts_per_tok: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
}

impl DeepseekV41Config {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let raw = fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("config.json under {}", dir.display()))?;
        let cfg: Self = serde_json::from_str(&raw).context("parse config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.architectures.iter().any(|a| a == DSV41_ARCHITECTURE),
            "unsupported architectures {:?}; expected {DSV41_ARCHITECTURE}",
            self.architectures
        );
        ensure!(
            self.model_type == DSV41_MODEL_TYPE,
            "unsupported model_type {:?}; expected {DSV41_MODEL_TYPE}",
            self.model_type
        );
        let t = &self.text_config;
        ensure!(
            t.compress_ratios.len() == t.num_hidden_layers + t.num_nextn_predict_layers,
            "compress_ratios covers {} layers but the checkpoint has {} backbone + {} MTP",
            t.compress_ratios.len(),
            t.num_hidden_layers,
            t.num_nextn_predict_layers
        );
        ensure!(
            t.engram_num_embeddings.len() == t.engram_layer_ids.len(),
            "engram_num_embeddings has {} entries for {} engram layers",
            t.engram_num_embeddings.len(),
            t.engram_layer_ids.len()
        );
        ensure!(
            t.kv_source_layer_ids
                .iter()
                .all(|id| t.compress_ratios.get(*id).is_some_and(|r| *r > 0)),
            "every KV source layer must compress its own KV"
        );
        ensure!(
            t.index_source_layer_ids.len() >= t.kv_source_layer_ids.len()
                && t.kv_source_layer_ids
                    .iter()
                    .all(|id| t.index_source_layer_ids.contains(id)),
            "every KV source layer must also be an index source"
        );
        ensure!(
            t.index_source_layer_ids
                .contains(&t.candidate_source_layer_id),
            "candidate source layer {} must be an index source",
            t.candidate_source_layer_id
        );
        Ok(())
    }

    pub fn vision(&self) -> Result<&VisionConfig> {
        self.vision_config
            .as_ref()
            .context("checkpoint has no vision_config")
    }
}

impl TextConfig {
    pub fn compress_ratio(&self, layer_id: usize) -> u8 {
        self.compress_ratios[layer_id]
    }

    pub fn is_kv_source(&self, layer_id: usize) -> bool {
        self.kv_source_layer_ids.contains(&layer_id)
    }

    pub fn is_index_source(&self, layer_id: usize) -> bool {
        self.index_source_layer_ids.contains(&layer_id)
    }

    pub fn engram_layer(&self, layer_id: usize) -> Option<usize> {
        self.engram_layer_ids.iter().position(|id| *id == layer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_real_checkpoint_config() {
        let dir = Path::new("../../models/deepseek-ai/DeepSeek-V4.1-Flash");
        if !dir.exists() {
            return;
        }
        let cfg = DeepseekV41Config::from_model_dir(dir).unwrap();
        assert_eq!(cfg.architectures, [DSV41_ARCHITECTURE]);
        assert_eq!(cfg.model_type, DSV41_MODEL_TYPE);
        assert_eq!(cfg.image_token_id, 129_264);
        let t = &cfg.text_config;
        assert_eq!(t.hidden_size, 5120);
        assert_eq!(t.num_hidden_layers, 40);
        assert_eq!(t.n_routed_experts, 384);
        assert_eq!(t.num_experts_per_tok, 6);
        assert_eq!(t.compress_ratios.len(), 43);
        assert_eq!(t.compress_ratios[..5], [0, 0, 2, 2, 2]);
        assert_eq!(t.compress_ratios[19], 2);
        assert_eq!(t.compress_ratios[20], 1);
        assert_eq!(t.compress_ratios[39], 1);
        assert_eq!(&t.compress_ratios[40..], &[0, 0, 0]);
        assert_eq!(t.kv_source_layer_ids, [2, 8, 14, 20]);
        assert_eq!(t.index_source_layer_ids, [2, 8, 14, 20, 24, 28, 32, 36]);
        assert_eq!(t.hc_mult, 4);
        assert_eq!(t.hc_sinkhorn_iters, 20);
        assert_eq!(t.engram_layer_ids, [1, 14]);
        assert_eq!(t.engram_num_embeddings, [384_006_168, 384_016_682]);
        assert_eq!(t.engram_compressed_vocab_size, 99_092);
        assert_eq!(t.candidate_source_layer_id, 20);
        assert_eq!(t.candidate_topk_blocks, 2048);
        assert_eq!(t.candidate_block_size, 8);
        assert_eq!(t.index_topk, 512);
        assert_eq!(t.num_nextn_predict_layers, 3);
        assert_eq!(t.dspark_n_routed_experts, 128);
        assert_eq!(t.dspark_num_experts_per_tok, 3);
        assert_eq!(cfg.quantization_config.weight_block_size, [32, 32]);
        assert_eq!(cfg.quantization_config.scale_fmt, "ue8m0");
        assert_eq!(cfg.quantization_config.expert_dtype, "fp4");
        let v = cfg.vision().unwrap();
        assert_eq!(v.num_hidden_layers, 32);
        assert_eq!(v.hidden_size, 1024);
        assert_eq!(v.patch_size, 14);
        assert_eq!(v.downsample_ratio, 3);
    }

    #[test]
    fn mismatched_engram_arrays_are_rejected() {
        let dir = Path::new("../../models/deepseek-ai/DeepSeek-V4.1-Flash");
        if !dir.exists() {
            return;
        }
        let mut cfg = DeepseekV41Config::from_model_dir(dir).unwrap();
        cfg.text_config.engram_num_embeddings.pop();
        assert!(cfg.validate().is_err());
    }
}
