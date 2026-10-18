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
    pub rope_dim: Option<usize>,
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
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim
    }
    pub fn read(path: &Path, target: &Config) -> Result<Self> {
        let c: Self = serde_json::from_value(read_config(path)?)?;
        ensure!(
            c.model_type == "qwen3" && c.architectures == ["Qwen3DSparkModel"],
            "unsupported DSpark architecture"
        );
        ensure!(c.dtype == "bfloat16", "DSpark dtype must be bfloat16");
        ensure!(
            c.hidden_size == target.hidden_size,
            "DSpark/target hidden_size mismatch"
        );
        ensure!(
            matches!(
                (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
                (16, 2, 128) | (8, 2, 256)
            ) && c.num_attention_heads.checked_mul(c.head_dim) == Some(c.hidden_size),
            "unsupported DSpark attention geometry"
        );
        ensure!(
            c.intermediate_size > 0 && c.intermediate_size.is_multiple_of(256),
            "unsupported DSpark intermediate_size"
        );
        ensure!(
            c.vocab_size == target.vocab_size,
            "DSpark shared vocabulary mismatch"
        );
        ensure!(
            c.draft_vocab_size == c.vocab_size,
            "DSpark draft_vocab_size differs from vocab_size"
        );
        ensure!(
            c.num_hidden_layers == 5 && c.markov_rank == 256,
            "DSpark requires five layers and Markov rank256"
        );
        ensure!(
            (1..=64).contains(&c.block_size),
            "DSpark block_size exceeds the 64-row attention mask"
        );
        ensure!(
            c.block_size.checked_mul(c.vocab_size).is_some()
                && c.vocab_size <= i32::MAX as usize
                && c.capture_width().checked_mul(c.hidden_size).is_some(),
            "DSpark tensor extent overflow"
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
                && c.rope_parameters.rope_theta.is_finite()
                && c.rope_parameters.rope_theta > 0.0
                && c.rope_parameters.rope_dim.unwrap_or(c.head_dim) == c.head_dim,
            "DSpark requires default full-head rotary geometry"
        );
        Ok(c)
    }
}

fn read_config(path: &Path) -> Result<serde_json::Value> {
    let path = path.join("config.json");
    let mut value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read draft config {}", path.display()))?,
    )?;
    if let Some(nested) = value.get("dflash_config").cloned() {
        let fields = nested.as_object().context("invalid DSpark dflash_config")?;
        let root = value.as_object_mut().context("invalid DSpark config")?;
        for name in [
            "target_layer_ids",
            "mask_token_id",
            "markov_rank",
            "markov_head_type",
            "projector_type",
        ] {
            if let Some(field) = fields.get(name) {
                if let Some(existing) = root.get(name) {
                    ensure!(
                        name != "target_layer_ids" && existing == field,
                        "ambiguous DSpark config field {name}"
                    );
                }
                root.insert(name.into(), field.clone());
            }
        }
        for (name, expected) in [
            ("attention_mode", serde_json::json!("gqa")),
            ("enable_confidence_head", serde_json::json!(true)),
            ("confidence_head_with_markov", serde_json::json!(true)),
            ("confidence_head_alpha", serde_json::json!(1.0)),
        ] {
            if let Some(actual) = fields.get(name) {
                ensure!(*actual == expected, "unsupported DSpark {name}");
            }
        }
    }
    Ok(value)
}

#[cfg(feature = "cuda")]
pub(crate) fn read_capture_layers(path: &Path, target: &Config) -> Result<Vec<usize>> {
    let value = read_config(path)?;
    ensure!(
        value["architectures"] == serde_json::json!(["Qwen3DSparkModel"]),
        "unsupported capture draft architecture"
    );
    for (name, expected) in [
        ("hidden_size", target.hidden_size),
        ("vocab_size", target.vocab_size),
        ("num_target_layers", target.num_hidden_layers),
    ] {
        ensure!(
            value[name].as_u64() == Some(u64::try_from(expected)?),
            "capture draft/target {name} mismatch"
        );
    }
    let ids: Vec<usize> = serde_json::from_value(
        value
            .get("target_layer_ids")
            .context("DSpark target_layer_ids missing")?
            .clone(),
    )?;
    check_capture_layers(&ids, target.num_hidden_layers)?;
    Ok(ids)
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

    #[test]
    #[cfg(feature = "cuda")]
    fn target_capture_metadata_preserves_slots_and_rejects_bad_extents() -> Result<()> {
        let target: Config = serde_json::from_value(serde_json::json!({
            "model_type":"llama", "hidden_size":2048, "intermediate_size":6144,
            "num_hidden_layers":42, "num_attention_heads":16, "num_key_value_heads":2,
            "head_dim":128, "vocab_size":130560, "max_position_embeddings":32768,
            "rms_norm_eps":1e-6, "rope_theta":5000000.0, "rope_scaling":null,
            "hidden_act":"silu", "tie_word_embeddings":false,
            "torch_dtype":"bfloat16", "eos_token_id":[2]
        }))?;
        let ids = [39usize, 1, 30, 10, 20];
        let flat = serde_json::json!({
            "architectures":["Qwen3DSparkModel"], "hidden_size":2048,
            "vocab_size":130560, "num_target_layers":42, "target_layer_ids":ids
        });
        let mut nested = flat.clone();
        nested
            .as_object_mut()
            .context("capture config object")?
            .remove("target_layer_ids");
        nested["dflash_config"] = serde_json::json!({"target_layer_ids":ids});
        let root = std::env::temp_dir().join(format!(
            "flyingfish-capture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let write = |value: &serde_json::Value| -> Result<()> {
                Ok(std::fs::write(
                    root.join("config.json"),
                    serde_json::to_vec(value)?,
                )?)
            };
            for value in [&flat, &nested] {
                write(value)?;
                ensure!(
                    read_capture_layers(&root, &target)? == ids,
                    "capture slot order changed"
                );
            }
            ensure!(
                serde_json::from_value::<Vec<usize>>(nested["target_layer_ids"].clone()).is_err(),
                "flat-only capture control accepted nested metadata"
            );
            let mut invalid = vec![flat.clone(); 7];
            invalid[0]["dflash_config"] = serde_json::json!({"target_layer_ids":ids});
            invalid[1]["target_layer_ids"] = serde_json::json!([1, 10, 20, 30, 42]);
            invalid[2]["target_layer_ids"] = serde_json::json!([1, 10, 20, 30, 30]);
            invalid[3]["target_layer_ids"] = serde_json::json!([1, 10, 20, 30]);
            invalid[4]["hidden_size"] = serde_json::json!(5120);
            invalid[5]["vocab_size"] = serde_json::json!(248144);
            invalid[6]["num_target_layers"] = serde_json::json!(40);
            for value in invalid {
                write(&value)?;
                ensure!(
                    read_capture_layers(&root, &target).is_err(),
                    "invalid capture extent admitted"
                );
            }
            let mut full = flat.clone();
            for (name, value) in serde_json::json!({
                "model_type":"qwen3", "dtype":"bfloat16", "intermediate_size":6144,
                "num_hidden_layers":5, "num_attention_heads":16, "num_key_value_heads":2,
                "head_dim":128, "draft_vocab_size":130560, "rms_norm_eps":1e-6,
                "block_size":7, "mask_token_id":75982, "markov_rank":256,
                "markov_head_type":"vanilla", "projector_type":"dspark",
                "rope_parameters":{"rope_theta":5000000.0,"rope_type":"default"}
            })
            .as_object()
            .context("draft fixture object")?
            {
                full[name] = value.clone();
            }
            write(&full)?;
            ensure!(
                DraftConfig::read(&root, &target)?.block_size == 7,
                "dense draft rejected"
            );
            let mut hybrid_target = target;
            hybrid_target.intermediate_size = 512;
            hybrid_target.num_attention_heads = 16;
            hybrid_target.head_dim = 256;
            hybrid_target.vocab_size = 248144;
            hybrid_target.num_hidden_layers = 40;
            hybrid_target.rope_theta = 1e7;
            for (name, value) in serde_json::json!({
                "num_attention_heads":8, "head_dim":256, "vocab_size":248144,
                "draft_vocab_size":248144, "num_target_layers":40, "block_size":15,
                "rope_parameters":{"rope_theta":1e7,"rope_type":"default","rope_dim":256}
            })
            .as_object()
            .context("hybrid draft fixture object")?
            {
                full[name] = value.clone();
            }
            let mut nested = serde_json::Map::new();
            for name in [
                "target_layer_ids",
                "mask_token_id",
                "markov_rank",
                "markov_head_type",
                "projector_type",
            ] {
                nested.insert(
                    name.into(),
                    full.as_object_mut()
                        .context("draft object")?
                        .remove(name)
                        .context("draft field")?,
                );
            }
            nested.insert(
                "target_layer_ids".into(),
                serde_json::json!([3, 11, 19, 27, 35]),
            );
            nested.insert("mask_token_id".into(), serde_json::json!(248076));
            full["dflash_config"] = nested.into();
            write(&full)?;
            let draft = DraftConfig::read(&root, &hybrid_target)?;
            ensure!(
                draft.block_size == 15 && draft.kv_dim() == 512 && draft.qkv_dim() == 3072,
                "draft geometry borrowed target dimensions"
            );
            for (field, value) in [
                ("draft_vocab_size", serde_json::json!(130560)),
                ("block_size", serde_json::json!(65)),
                ("num_attention_heads", serde_json::json!(16)),
                (
                    "rope_parameters",
                    serde_json::json!({"rope_theta":1e7,"rope_type":"default","rope_dim":64}),
                ),
            ] {
                let mut invalid = full.clone();
                invalid[field] = value;
                write(&invalid)?;
                ensure!(
                    DraftConfig::read(&root, &hybrid_target).is_err(),
                    "invalid draft geometry admitted"
                );
            }
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(root);
        result?;
        cleanup?;
        Ok(())
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
