use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<serde_json::Value>,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub eos_token_id: Vec<u32>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
}

impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let c: Self = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )?;
        ensure!(c.model_type == "llama", "model_type must be llama");
        ensure!(c.torch_dtype == "bfloat16", "torch_dtype must be bfloat16");
        ensure!(c.hidden_act == "silu", "hidden_act must be silu");
        ensure!(!c.tie_word_embeddings, "tie_word_embeddings must be false");
        ensure!(
            !c.attention_bias && !c.mlp_bias,
            "projection biases are unsupported"
        );
        ensure!(c.rope_scaling.is_none(), "rope_scaling is unsupported");
        ensure!(c.head_dim == 128, "attention head_dim must be 128");
        ensure!(
            c.num_key_value_heads > 0
                && c.num_attention_heads.is_multiple_of(c.num_key_value_heads),
            "invalid GQA head counts"
        );
        ensure!(
            c.hidden_size == c.num_attention_heads * c.head_dim,
            "hidden_size does not match heads * head_dim"
        );
        ensure!(
            c.hidden_size > 0
                && c.intermediate_size > 0
                && c.vocab_size > 0
                && c.num_hidden_layers > 0,
            "zero model dimension"
        );
        ensure!(
            c.rms_norm_eps.is_finite()
                && c.rms_norm_eps > 0.0
                && c.rope_theta.is_finite()
                && c.rope_theta > 0.0,
            "invalid norm epsilon or rope theta"
        );
        ensure!(
            c.eos_token_id.iter().all(|&x| (x as usize) < c.vocab_size),
            "EOS outside vocabulary"
        );
        Ok(c)
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        self.hidden_size + 2 * self.kv_dim()
    }
}
