use anyhow::{Context, Result};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TransformerConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
    pub ffn_dim: usize,
    pub in_channels: usize,
    pub audio_in_channels: usize,
    pub patch_size: [usize; 3],
    pub text_dim: usize,
    pub freq_dim: usize,
    pub time_embed_hidden_dim: usize,
    pub time_embed_dim: usize,
    pub rope_freq_dim: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub qk_norm_eps: f64,
    pub final_norm_eps: f64,
}

impl TransformerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read(path)
            .with_context(|| format!("failed to read transformer config {}", path.display()))?;
        let config = serde_json::from_slice::<Self>(&data)
            .with_context(|| format!("invalid transformer config {}", path.display()))?;
        anyhow::ensure!(
            config.class_name == "MiniMaxH3Transformer3DModel",
            "{} describes {}, not MiniMaxH3Transformer3DModel",
            path.display(),
            config.class_name
        );
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.hidden_size > 0, "hidden_size must be non-zero");
        anyhow::ensure!(self.num_layers > 0, "num_layers must be non-zero");
        anyhow::ensure!(
            self.num_attention_heads > 0,
            "num_attention_heads must be non-zero"
        );
        anyhow::ensure!(
            self.attention_head_dim > 0,
            "attention_head_dim must be non-zero"
        );
        anyhow::ensure!(
            self.patch_size.iter().all(|&v| v > 0),
            "patch_size must be non-zero"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_shape() {
        let config: TransformerConfig = serde_json::from_str(
            r#"{
                "_class_name":"MiniMaxH3Transformer3DModel",
                "num_attention_heads":56,"attention_head_dim":128,"hidden_size":5376,
                "num_layers":50,"num_refiner_layers":2,"ffn_dim":14336,
                "in_channels":24,"audio_in_channels":32,"patch_size":[1,2,2],
                "text_dim":5120,"freq_dim":256,"time_embed_hidden_dim":5376,
                "time_embed_dim":2688,"rope_freq_dim":16,"rope_theta":10000.0,
                "norm_eps":0.00001,"qk_norm_eps":0.00001,"final_norm_eps":0.00001
            }"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.num_layers, 50);
        assert_eq!(config.num_attention_heads * config.attention_head_dim, 7168);
    }
}
