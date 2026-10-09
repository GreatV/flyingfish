use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct DraftConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub rms_norm_eps: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub projector_type: String,
    pub rope_parameters: Rope,
}

#[derive(Debug, Deserialize)]
pub struct Rope {
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Serialize)]
pub struct Rules {
    pub h_mean: f64,
    pub h_rel: f64,
    pub b_mean: f64,
    pub s_mean: f64,
    pub kv_rel: f64,
    pub margin: f64,
}

impl Rules {
    pub fn read(path: &Path) -> Result<Self> {
        let table: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let gates = table.get("gates").context("DSpark gates missing")?;
        let number = |name: &str| -> Result<f64> {
            gates
                .get(name)
                .and_then(serde_json::Value::as_f64)
                .with_context(|| format!("DSpark {name} numeric gate missing"))
        };
        let embedded = |name: &str, marker: &str| -> Result<f64> {
            let text = gates
                .get(name)
                .and_then(serde_json::Value::as_str)
                .with_context(|| format!("DSpark {name} gate description missing"))?;
            text.split_once(marker)
                .with_context(|| format!("DSpark {name} lacks {marker}"))?
                .1
                .split([' ', ';', ')', ','])
                .next()
                .context("DSpark gate number missing")?
                .parse()
                .with_context(|| format!("DSpark {name} invalid numeric gate"))
        };
        let rules = Self {
            h_mean: number("H_mean_abs")?,
            h_rel: number("H_rel_rmse")?,
            b_mean: number("B_mean_abs")?,
            s_mean: number("s_mean_abs")?,
            kv_rel: embedded("kv_inject", "rel_rmse <= ")?,
            margin: embedded("proposals", "top1-top2 < ")?,
        };
        ensure!(
            [
                rules.h_mean,
                rules.h_rel,
                rules.b_mean,
                rules.s_mean,
                rules.kv_rel,
                rules.margin
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0),
            "invalid DSpark numeric gate"
        );
        Ok(rules)
    }
}

impl DraftConfig {
    pub fn capture_width(&self) -> usize {
        self.target_layer_ids.len() * self.hidden_size
    }
    pub fn read(path: &Path, target: &Config) -> Result<Self> {
        let path = path.join("config.json");
        let c: Self = serde_json::from_slice(
            &std::fs::read(&path)
                .with_context(|| format!("read draft config {}", path.display()))?,
        )?;
        ensure!(
            c.model_type == "qwen3" && c.architectures == ["Qwen3DSparkModel"],
            "unsupported DSpark architecture"
        );
        ensure!(c.dtype == "bfloat16", "DSpark dtype must be bfloat16");
        ensure!(
            c.hidden_size == target.hidden_size
                && c.intermediate_size == target.intermediate_size
                && c.num_attention_heads == target.num_attention_heads
                && c.num_key_value_heads == target.num_key_value_heads
                && c.head_dim == target.head_dim,
            "DSpark/target dimensions do not match"
        );
        ensure!(
            c.vocab_size == target.vocab_size && c.draft_vocab_size == target.vocab_size,
            "DSpark shared vocabulary mismatch"
        );
        ensure!(
            c.num_hidden_layers == 5 && c.block_size == 7 && c.markov_rank == 256,
            "DSpark requires 5 layers, gamma7 and markov rank256"
        );
        ensure!(
            c.num_target_layers == target.num_hidden_layers,
            "invalid target hidden capture layers"
        );
        check_capture_layers(&c.target_layer_ids, target.num_hidden_layers)?;
        ensure!(
            (c.mask_token_id as usize) < target.vocab_size
                && c.rms_norm_eps.is_finite()
                && c.rms_norm_eps > 0.0,
            "invalid DSpark mask token or RMSNorm epsilon"
        );
        ensure!(
            c.projector_type == "dspark" && c.markov_head_type == "vanilla",
            "unsupported DSpark projector or Markov head"
        );
        ensure!(
            c.rope_parameters.rope_type == "default"
                && c.rope_parameters.rope_theta == target.rope_theta,
            "DSpark/target RoPE mismatch"
        );
        Ok(c)
    }
}

fn check_capture_layers(ids: &[usize], target_layers: usize) -> Result<()> {
    ensure!(
        ids.len() == 5 && ids.iter().all(|&i| i < target_layers),
        "invalid target hidden capture layers"
    );
    for (index, &id) in ids.iter().enumerate() {
        ensure!(
            !ids[..index].contains(&id),
            "duplicate target hidden capture layer {id}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accept {
    pub proposals: usize,
    pub bonus: u32,
}

pub fn accept(proposals: &[u32], predictions: &[u32]) -> Result<Accept> {
    ensure!(
        predictions.len() == proposals.len() + 1,
        "verification must contain anchor plus proposals"
    );
    let matched = proposals
        .iter()
        .zip(predictions)
        .take_while(|(p, t)| p == t)
        .count();
    Ok(Accept {
        proposals: matched,
        bonus: predictions[matched],
    })
}

// target_layer_ids 中 i 指 decoder 层 i 的输出，即 hidden_states[i + 1]。
pub fn capture_slot(ids: &[usize], decoder_layer: usize) -> Option<usize> {
    ids.iter().position(|&i| i == decoder_layer)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejection_commits_only_matching_prefix_and_target_bonus() {
        assert_eq!(
            accept(&[3, 4, 5], &[3, 7, 5, 8]).unwrap(),
            Accept {
                proposals: 1,
                bonus: 7
            }
        );
        assert_eq!(
            accept(&[3, 4, 5], &[9, 4, 5, 8]).unwrap(),
            Accept {
                proposals: 0,
                bonus: 9
            }
        );
        assert_eq!(
            accept(&[3, 4, 5], &[3, 4, 5, 8]).unwrap(),
            Accept {
                proposals: 3,
                bonus: 8
            }
        );
        assert!(accept(&[3, 4, 5], &[3, 4, 5]).is_err());
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Round {
    pub logits_rows: Vec<usize>,
    pub proposals: Vec<u32>,
    pub predictions: Vec<u32>,
    pub accepted: usize,
    pub committed: usize,
    pub output: Vec<u32>,
    pub position: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub inject_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tree: Option<TreeTiming>,
}

#[derive(Debug, serde::Serialize)]
pub struct TreeTiming {
    pub builder: crate::backend::TreeBuilder,
    pub base_ms: f64,
    pub build_ms: f64,
    pub waves: usize,
    pub requests: usize,
    pub used_requests: usize,
    pub batch_sizes: Vec<usize>,
    pub compared_serial: bool,
}
