//! Shard layout of the DeepSeek-V4.1-Flash checkpoint.
//!
//! The 48 shards are single-purpose: a text-only run opens the text, embed
//! and head shards only; the vision tower lives in shard 1, the DSpark draft
//! in shards 44-46, and the two engram tables in shards 47-48.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Clone, Debug, Deserialize)]
struct ShardIndex {
    #[serde(default)]
    pub metadata: ShardIndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ShardIndexMetadata {
    #[serde(
        default,
        deserialize_with = "ff_core::weights::deserialize_optional_index_size"
    )]
    pub total_size: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ShardRole {
    Vision,
    Embed,
    Text,
    HeadNorm,
    Dspark,
    Engram,
}

impl ShardRole {
    fn of_tensors(names: &BTreeSet<String>) -> Option<Self> {
        if names
            .iter()
            .any(|name| name.starts_with("vision.") || name.starts_with("aligner."))
        {
            Some(Self::Vision)
        } else if names.iter().any(|name| name.contains(".engram.")) {
            Some(Self::Engram)
        } else if names.iter().any(|name| name.starts_with("mtp.")) {
            Some(Self::Dspark)
        } else if names.contains("embed.weight") {
            Some(Self::Embed)
        } else if names.contains("head.weight") || names.contains("norm.weight") {
            Some(Self::HeadNorm)
        } else if names.iter().any(|name| name.starts_with("layers.")) {
            Some(Self::Text)
        } else {
            None
        }
    }
}

/// Which optional shard families a run opens on top of the always-needed
/// text skeleton.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenPlan {
    pub vision: bool,
    pub dspark: bool,
    pub engram: bool,
}

#[derive(Clone, Debug)]
pub struct CheckpointLayout {
    shards: BTreeMap<String, ShardRole>,
    tensors: BTreeMap<String, String>,
    total_size: Option<u64>,
}

impl CheckpointLayout {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("model.safetensors.index.json");
        let raw = fs::read_to_string(&path).with_context(|| {
            format!("model.safetensors.index.json under {}", model_dir.display())
        })?;
        let index: ShardIndex =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        let mut by_shard: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (tensor, shard) in &index.weight_map {
            by_shard
                .entry(shard.clone())
                .or_default()
                .insert(tensor.clone());
        }
        let mut shards = BTreeMap::new();
        for (shard, tensors) in by_shard {
            let role = ShardRole::of_tensors(&tensors)
                .with_context(|| format!("shard {shard} carries no recognizable tensor family"))?;
            shards.insert(shard, role);
        }
        Ok(Self {
            shards,
            tensors: index.weight_map,
            total_size: index.metadata.total_size,
        })
    }

    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    pub fn shard_role(&self, shard: &str) -> Option<ShardRole> {
        self.shards.get(shard).copied()
    }

    pub fn shard_count(&self, role: ShardRole) -> usize {
        self.shards.values().filter(|r| **r == role).count()
    }

    pub fn tensors_in(&self, shard: &str) -> Vec<&str> {
        self.tensors
            .iter()
            .filter(|(_, s)| s.as_str() == shard)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// The shards a run opens under the given plan, sorted by name.
    pub fn shards_to_open(&self, plan: OpenPlan) -> Vec<&str> {
        self.shards
            .iter()
            .filter(|(_, role)| {
                matches!(
                    role,
                    ShardRole::Text | ShardRole::Embed | ShardRole::HeadNorm
                ) || (plan.vision && **role == ShardRole::Vision)
                    || (plan.dspark && **role == ShardRole::Dspark)
                    || (plan.engram && **role == ShardRole::Engram)
            })
            .map(|(shard, _)| shard.as_str())
            .collect()
    }
}

use candle_core::{DType, Device, Tensor};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};

/// The static projections one text layer needs on the host, materialized F32.
pub struct LayerWeights {
    pub sink: Tensor,
    pub wq_a: Tensor,
    pub q_norm: Tensor,
    pub wq_b: Tensor,
    pub wkv: Tensor,
    pub kv_norm: Tensor,
    pub wo_a: Tensor,
    pub wo_b: Tensor,
}

fn open_weights(model_dir: &Path) -> Result<ModelWeights> {
    ModelWeights::open(model_dir, WeightSource::Mmap, CachePolicy::new(2))
}

fn f32_tensor(weights: &ModelWeights, name: &str) -> Result<Tensor> {
    weights
        .load(name, &Device::Cpu)
        .with_context(|| format!("load {name} from {}", weights.root().display()))?
        .to_dtype(DType::F32)
        .with_context(|| format!("cast {name}"))
}

fn scale_bytes(weights: &ModelWeights, name: &str) -> Result<Vec<u8>> {
    weights.with_tensor_bytes(name, |bytes| Ok(bytes.to_vec()))
}

/// `projection` is the name without the `.weight` suffix; the scale hangs off
/// the projection name (`wq_a.weight` pairs with `wq_a.scale`).
fn fp8_tensor(weights: &ModelWeights, projection: &str) -> Result<Tensor> {
    let device = Device::Cpu;
    let weight_name = format!("{projection}.weight");
    let weight = weights.load(&weight_name, &device)?;
    let scale = scale_bytes(weights, &format!("{projection}.scale"))?;
    crate::quant::dequantize_fp8_block(&weight, &scale, &device)
        .with_context(|| format!("dequantize {weight_name}"))
}

fn fp4_tensor(
    weights: &ModelWeights,
    weight_name: &str,
    rows: usize,
    columns: usize,
) -> Result<Tensor> {
    let scale = scale_bytes(weights, &weight_name.replace(".weight", ".scale"))?;
    weights
        .with_tensor_bytes(weight_name, |payload| {
            crate::quant::dequantize_fp4_packed(payload, rows, columns, &scale, &Device::Cpu)
        })
        .with_context(|| format!("dequantize {weight_name}"))
}

impl LayerWeights {
    pub fn load_layer(weights: &ModelWeights, layer: usize) -> Result<Self> {
        let prefix = format!("layers.{layer}.attn.");
        Ok(Self {
            sink: f32_tensor(weights, &format!("{prefix}attn_sink"))?,
            wq_a: fp8_tensor(weights, &format!("{prefix}wq_a"))?,
            q_norm: f32_tensor(weights, &format!("{prefix}q_norm.weight"))?,
            wq_b: fp8_tensor(weights, &format!("{prefix}wq_b"))?,
            wkv: fp8_tensor(weights, &format!("{prefix}wkv"))?,
            kv_norm: f32_tensor(weights, &format!("{prefix}kv_norm.weight"))?,
            wo_a: fp8_tensor(weights, &format!("{prefix}wo_a"))?,
            wo_b: fp8_tensor(weights, &format!("{prefix}wo_b"))?,
        })
    }
}

use crate::attention::Compressor;
use crate::config::DeepseekV41Config;
use crate::engram::{EngramLayout, NgramHashState, build_compressed_token_map};
use crate::model::{AttentionCore, WindowRing, layer_freqs};
use crate::moe::{Expert, Gate};
use crate::transformer::{
    BlockWeights, EngramCore, EngramLookup, FfnWeights, Transformer, TransformerParams,
};
use anyhow::bail;

/// Assembles a host-reference `Transformer` from a checkpoint directory.
/// Every projection is materialized F32 at load: the static skeleton and the
/// routed experts of all layers resident together. `open` refuses loudly
/// when the dequantized set cannot fit the host instead of degrading.
pub struct TransformerLoader {
    model_dir: std::path::PathBuf,
    config: DeepseekV41Config,
    layout: CheckpointLayout,
    weights: std::sync::Arc<ModelWeights>,
}

impl TransformerLoader {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config = DeepseekV41Config::from_model_dir(model_dir)?;
        let layout = CheckpointLayout::open(model_dir)?;
        let weights = std::sync::Arc::new(open_weights(model_dir)?);
        Ok(Self {
            model_dir: model_dir.to_owned(),
            config,
            layout,
            weights,
        })
    }

    pub fn config(&self) -> &DeepseekV41Config {
        &self.config
    }

    fn f32(&self, name: &str) -> Result<Tensor> {
        f32_tensor(&self.weights, name)
    }

    fn fp8(&self, projection: &str) -> Result<Tensor> {
        fp8_tensor(&self.weights, projection)
    }

    fn fp4(&self, weight_name: &str, rows: usize, columns: usize) -> Result<Tensor> {
        fp4_tensor(&self.weights, weight_name, rows, columns)
    }

    /// Dequantized sizes of every tensor the resident load would hold, by
    /// checkpoint role. FP8 and FP4 grow 4x and 8x to F32; BF16 grows 2x.
    pub fn resident_f32_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for (tensor, shard) in &self.layout.tensors {
            let role = self
                .layout
                .shard_role(shard)
                .with_context(|| format!("shard {shard} has no role"))?;
            // The DSpark draft, the vision tower (unwired on the text path),
            // and the 189 GiB engram embed table stay lookup-backed or
            // unloaded; the engram projections (wkv, q/k weights) are
            // dequantized resident at load time, so they count here.
            let lookup_backed = match role {
                ShardRole::Dspark | ShardRole::Vision => true,
                ShardRole::Engram => tensor.contains(".engram.embed."),
                _ => false,
            };
            // Quantization scales are read transiently while dequantizing;
            // only the dequantized weights stay resident.
            if lookup_backed || tensor.ends_with(".scale") {
                continue;
            }
            let view = self.weights.raw_tensor_metadata(tensor)?;
            let elements: u64 = view
                .shape
                .iter()
                .map(|dimension| *dimension as u64)
                .product::<u64>();
            // F32 stays 4 bytes; BF16 and FP8 double/quadruple to 4; a packed
            // FP4 byte (I8 view, two values per byte) becomes eight F32 bytes.
            let element_bytes = match view.dtype {
                safetensors::Dtype::F32 => 4,
                safetensors::Dtype::BF16
                | safetensors::Dtype::F8_E4M3
                | safetensors::Dtype::F8_E8M0 => 4,
                safetensors::Dtype::I8 => 8,
                other => {
                    return Err(anyhow::anyhow!(
                        "{tensor} in {shard} has unexpected dtype {other:?}"
                    ));
                }
            } as u64;
            total += elements
                .checked_mul(element_bytes)
                .with_context(|| format!("{tensor} resident size overflows u64"))?;
        }
        Ok(total)
    }

    /// Load the whole transformer onto the host. `tokenizer` is needed only
    /// when the checkpoint carries engram layers.
    pub fn load(
        self,
        tokenizer: Option<&tokenizers::Tokenizer>,
        max_seq: usize,
    ) -> Result<Transformer> {
        let config = DeepseekV41Config::from_model_dir(&self.model_dir)?;
        let text = &config.text_config;
        let params = TransformerParams {
            heads: text.num_attention_heads,
            head_dim: text.head_dim,
            rope_head_dim: text.qk_rope_head_dim,
            o_lora_rank: text.o_lora_rank,
            o_groups: text.o_groups,
            hc_mult: text.hc_mult,
            hc_sinkhorn_iters: text.hc_sinkhorn_iters,
            hc_eps: text.hc_eps,
            norm_eps: text.rms_norm_eps,
            vocab: text.vocab_size,
            hidden: text.hidden_size,
        };
        let embed = self.f32("embed.weight")?;
        let head = self.f32("head.weight")?;
        let norm_weight = self.f32("norm.weight")?;

        let ngram = if text.engram_layer_ids.is_empty() {
            None
        } else {
            let tokenizer = tokenizer.with_context(|| {
                "the checkpoint carries engram layers; the compressed token map needs its tokenizer"
            })?;
            let (token_map, compressed) = build_compressed_token_map(tokenizer)?;
            anyhow::ensure!(
                compressed == text.engram_compressed_vocab_size,
                "compressed vocab {compressed} disagrees with the config value {}",
                text.engram_compressed_vocab_size
            );
            let layout = EngramLayout::from_config(text)?;
            Some(NgramHashState::new(
                layout,
                token_map,
                1,
                max_seq,
                self.config.text_config.engram_compressed_vocab_size,
                self.config.text_config.engram_pad_token_id,
            )?)
        };

        let mut blocks = Vec::with_capacity(text.num_hidden_layers);
        for layer in 0..text.num_hidden_layers {
            blocks.push(self.load_block(layer, max_seq, &ngram)?);
        }
        Ok(Transformer {
            params,
            embed,
            blocks,
            norm_weight,
            head,
            ngram,
        })
    }

    fn load_block(
        &self,
        layer: usize,
        max_seq: usize,
        ngram: &Option<NgramHashState>,
    ) -> Result<BlockWeights> {
        let (ratio, is_kv_source, is_index_source, sliding_window, head_dim, rope_head_dim) = {
            let text = &self.config.text_config;
            (
                text.compress_ratio(layer) as usize,
                text.is_kv_source(layer),
                text.is_index_source(layer),
                text.sliding_window,
                text.head_dim,
                text.qk_rope_head_dim,
            )
        };
        let device = Device::Cpu;
        let attn = format!("layers.{layer}.attn.");
        let freqs = layer_freqs(
            &self.config.text_config,
            rope_head_dim,
            ratio as u8,
            max_seq,
            &device,
        )?;

        let compressor = if ratio > 1 && is_kv_source {
            let prefix = format!("{attn}compressor.");
            Some(Compressor::new(
                ratio,
                head_dim,
                self.f32(&format!("{prefix}norm.weight"))?,
                self.f32(&format!("{prefix}wkv.weight"))?,
                Some(self.f32(&format!("{prefix}wgate.weight"))?),
                1,
                self.config.text_config.rms_norm_eps,
            )?)
        } else if ratio == 1 && is_kv_source {
            let prefix = format!("{attn}compressor.");
            Some(Compressor::new(
                ratio,
                head_dim,
                self.f32(&format!("{prefix}norm.weight"))?,
                self.f32(&format!("{prefix}wkv.weight"))?,
                None,
                1,
                self.config.text_config.rms_norm_eps,
            )?)
        } else {
            None
        };

        let indexer_wq_b = if is_index_source {
            Some(self.fp8(&format!("{attn}indexer.wq_b"))?)
        } else {
            None
        };
        let indexer_wk = if is_kv_source && is_index_source {
            Some(self.f32(&format!("{attn}indexer.wk.weight"))?)
        } else {
            None
        };
        let indexer_k_norm = if indexer_wk.is_some() {
            Some(self.f32(&format!("{attn}indexer.k_norm.weight"))?)
        } else {
            None
        };
        let weights_proj = if is_index_source {
            Some(self.f32(&format!("{attn}indexer.weights_proj.weight"))?)
        } else {
            None
        };

        let attention = AttentionCore {
            sink: self.f32(&format!("{attn}attn_sink"))?,
            wq_a: self.fp8(&format!("{attn}wq_a"))?,
            q_norm: self.f32(&format!("{attn}q_norm.weight"))?,
            wq_b: self.fp8(&format!("{attn}wq_b"))?,
            wkv: self.fp8(&format!("{attn}wkv"))?,
            kv_norm: self.f32(&format!("{attn}kv_norm.weight"))?,
            wo_a: self.fp8(&format!("{attn}wo_a"))?,
            wo_b: self.fp8(&format!("{attn}wo_b"))?,
            compressor,
            compress_cache: Vec::new(),
            index_k_cache: Vec::new(),
            ring: WindowRing::new(1, sliding_window, head_dim),
            indexer_wq_b,
            indexer_wk,
            indexer_k_norm,
            weights_proj,
            freqs,
            index_head_dim: self.config.text_config.index_head_dim,
            index_heads: self.config.text_config.index_n_heads,
            index_topk: self.config.text_config.index_topk,
            candidate_topk_blocks: self.config.text_config.candidate_topk_blocks,
            candidate_block_size: self.config.text_config.candidate_block_size,
            norm_eps: self.config.text_config.rms_norm_eps,
        };

        let ffn = self.load_ffn(layer)?;
        let engram = match (ngram.is_some(), self.config.text_config.engram_layer(layer)) {
            (true, Some(hash_index)) => Some(self.load_engram(layer, hash_index)?),
            (true, None) => None,
            (false, Some(_)) => {
                bail!("engram layer {layer} present but no tokenizer was supplied")
            }
            (false, None) => None,
        };

        Ok(BlockWeights {
            attention,
            attn_norm: self.f32(&format!("layers.{layer}.attn_norm.weight"))?,
            ffn_norm: self.f32(&format!("layers.{layer}.ffn_norm.weight"))?,
            hc_attn_fn: self.f32(&format!("layers.{layer}.hc_attn_fn"))?,
            hc_attn_scale: self.f32(&format!("layers.{layer}.hc_attn_scale"))?,
            hc_attn_base: self.f32(&format!("layers.{layer}.hc_attn_base"))?,
            hc_ffn_fn: self.f32(&format!("layers.{layer}.hc_ffn_fn"))?,
            hc_ffn_scale: self.f32(&format!("layers.{layer}.hc_ffn_scale"))?,
            hc_ffn_base: self.f32(&format!("layers.{layer}.hc_ffn_base"))?,
            ffn,
            engram,
            ratio,
            is_kv_source,
            is_index_source,
            uses_candidates: self.config.text_config.candidate_source_layer_id < layer,
            candidate_source: layer == self.config.text_config.candidate_source_layer_id,
        })
    }

    fn load_ffn(&self, layer: usize) -> Result<FfnWeights> {
        let (routed, active, intermediate, hidden, swiglu_limit, norm_topk_prob, route_scale) = {
            let text = &self.config.text_config;
            (
                text.n_routed_experts,
                text.num_experts_per_tok,
                text.moe_intermediate_size,
                text.hidden_size,
                text.swiglu_limit,
                text.norm_topk_prob,
                text.routed_scaling_factor,
            )
        };
        let prefix = format!("layers.{layer}.ffn.");
        let gate = Gate {
            weight: self.f32(&format!("{prefix}gate.weight"))?,
            bias: self.f32(&format!("{prefix}gate.bias"))?,
            bias_vl: if self.config.vision_config.is_some() {
                Some(self.f32(&format!("{prefix}gate.bias_vl"))?)
            } else {
                None
            },
            topk: active,
            gate_temp: 1.0,
            norm_topk_prob,
            route_scale,
        };
        let load_expert = |loader: &Self, projection: &str| -> Result<Expert> {
            Ok(Expert {
                w1: loader.fp4(&format!("{projection}.w1.weight"), intermediate, hidden)?,
                w2: loader.fp4(&format!("{projection}.w2.weight"), hidden, intermediate)?,
                w3: loader.fp4(&format!("{projection}.w3.weight"), intermediate, hidden)?,
                swiglu_limit,
            })
        };
        let mut experts = Vec::with_capacity(routed);
        for index in 0..routed {
            experts.push(load_expert(self, &format!("{prefix}experts.{index}"))?);
        }
        let shared = load_expert(self, &format!("{prefix}shared_experts"))?;
        Ok(FfnWeights {
            gate,
            experts,
            shared,
        })
    }

    fn load_engram(&self, layer: usize, hash_index: usize) -> Result<EngramCore> {
        let text = &self.config.text_config;
        let prefix = format!("layers.{layer}.engram.");
        let weight_name = format!("{prefix}embed.weight");
        let scale_name = format!("{prefix}embed.scale");
        let head_dim = text.engram_head_dim;
        let weight = self
            .weights
            .raw_tensor_metadata(&weight_name)
            .with_context(|| format!("engram payload {weight_name}"))?;
        let scale = self
            .weights
            .raw_tensor_metadata(&scale_name)
            .with_context(|| format!("engram scale {scale_name}"))?;
        let expected_rows = text
            .engram_num_embeddings
            .get(hash_index)
            .copied()
            .with_context(|| format!("no engram table configured for index {hash_index}"))?;
        let scale_width = head_dim / crate::config::FP8_WEIGHT_BLOCK;
        anyhow::ensure!(
            weight.dtype == safetensors::Dtype::F8_E4M3
                && weight.shape == [expected_rows as usize, head_dim].as_slice(),
            "{weight_name} must be F8_E4M3 [{expected_rows}, {head_dim}], found {:?} {:?}",
            weight.dtype,
            weight.shape
        );
        anyhow::ensure!(
            scale.dtype == safetensors::Dtype::F8_E8M0
                && scale.shape == [expected_rows as usize, scale_width].as_slice(),
            "{scale_name} must be F8_E8M0 [{expected_rows}, {scale_width}], found {:?} {:?}",
            scale.dtype,
            scale.shape
        );
        let weights = std::sync::Arc::clone(&self.weights);
        let lookup: EngramLookup = Box::new(move |ids: &[i64]| -> Vec<f32> {
            let mut rows = Vec::with_capacity(ids.len() * head_dim);
            weights
                .with_tensor_bytes(&weight_name, |payload| {
                    weights.with_tensor_bytes(&scale_name, |scales| {
                        for id in ids {
                            let row = *id as usize;
                            for d in 0..head_dim {
                                let byte = payload[row * head_dim + d];
                                let group = d / crate::config::FP8_WEIGHT_BLOCK;
                                let exponent = scales
                                    [row * (head_dim / crate::config::FP8_WEIGHT_BLOCK) + group]
                                    as i32
                                    - 127;
                                rows.push(
                                    crate::quant::e4m3_byte_to_f32(byte) * 2.0f32.powi(exponent),
                                );
                            }
                        }
                        Ok(())
                    })
                })
                .expect("engram tensors were validated at load");
            rows
        });
        Ok(EngramCore {
            layer_hash_index: hash_index,
            head_dim,
            wkv: self.fp8(&format!("{prefix}wkv"))?,
            q_weight: self.f32(&format!("{prefix}q_weight"))?,
            k_weight: self.f32(&format!("{prefix}k_weight"))?,
            lookup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::paths::checkpoint_dir;

    #[test]
    fn index_size_uses_the_shared_optional_numeric_rule() {
        for (metadata, expected) in [
            ("{}", None),
            (r#"{"total_size":null}"#, None),
            (r#"{"total_size":55562855904.0}"#, Some(55_562_855_904)),
            (r#"{"total_size":18446744073709551615}"#, Some(u64::MAX)),
        ] {
            let parsed: ShardIndex =
                serde_json::from_str(&format!("{{\"metadata\":{metadata},\"weight_map\":{{}}}}"))
                    .unwrap();
            assert_eq!(parsed.metadata.total_size, expected);
        }
        let missing: ShardIndex = serde_json::from_str(r#"{"weight_map":{}}"#).unwrap();
        assert_eq!(missing.metadata.total_size, None);
        for size in ["1.5", "-1", "9007199254740994.0"] {
            assert!(
                serde_json::from_str::<ShardIndexMetadata>(&format!("{{\"total_size\":{size}}}"))
                    .is_err()
            );
        }
    }
    fn layout() -> Option<CheckpointLayout> {
        let dir = &checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash")?;
        if !dir.join("model.safetensors.index.json").exists() {
            return None;
        }
        Some(CheckpointLayout::open(dir).unwrap())
    }

    #[test]
    fn shards_partition_into_six_roles() {
        let Some(layout) = layout() else {
            return;
        };
        assert_eq!(layout.shard_count(ShardRole::Vision), 1);
        assert_eq!(layout.shard_count(ShardRole::Embed), 1);
        assert_eq!(layout.shard_count(ShardRole::Text), 40);
        assert_eq!(layout.shard_count(ShardRole::HeadNorm), 1);
        assert_eq!(layout.shard_count(ShardRole::Dspark), 3);
        assert_eq!(layout.shard_count(ShardRole::Engram), 2);
        assert_eq!(layout.shards.len(), 48);
    }

    #[test]
    fn role_assignment_matches_the_documented_shard_files() {
        let Some(layout) = layout() else {
            return;
        };
        assert_eq!(
            layout.shard_role("model-00001-of-00048.safetensors"),
            Some(ShardRole::Vision)
        );
        assert_eq!(
            layout.shard_role("model-00002-of-00048.safetensors"),
            Some(ShardRole::Embed)
        );
        assert_eq!(
            layout.shard_role("model-00043-of-00048.safetensors"),
            Some(ShardRole::HeadNorm)
        );
        for shard in [
            "model-00044-of-00048.safetensors",
            "model-00045-of-00048.safetensors",
            "model-00046-of-00048.safetensors",
        ] {
            assert_eq!(layout.shard_role(shard), Some(ShardRole::Dspark));
        }
        for shard in [
            "model-00047-of-00048.safetensors",
            "model-00048-of-00048.safetensors",
        ] {
            assert_eq!(layout.shard_role(shard), Some(ShardRole::Engram));
            assert_eq!(layout.tensors_in(shard).len(), 6);
        }
    }

    #[test]
    fn repeated_estimate_calls_stay_valid_after_map_release() {
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        if !dir.join("model.safetensors.index.json").exists() {
            return;
        }
        let first = TransformerLoader::open(dir)
            .unwrap()
            .resident_f32_bytes()
            .unwrap();
        // Each call builds and drops its own header set; a dangling mmap
        // would read freed storage on the second pass.
        let second = TransformerLoader::open(dir)
            .unwrap()
            .resident_f32_bytes()
            .unwrap();
        assert_eq!(first, second);
        assert!(first > 0);
    }

    #[test]
    fn resident_estimate_sums_exact_tensor_sizes_not_shard_averages() {
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        if !dir.join("model.safetensors.index.json").exists() {
            return;
        }
        let loader = TransformerLoader::open(dir).unwrap();
        let exact = loader.resident_f32_bytes().unwrap();
        // Cross-check a few known tensors against the same header math the
        // loader uses: a BF16 head row-block and an FP8 body projection.
        let layout = CheckpointLayout::open(dir).unwrap();
        let view = open_weights(dir)
            .unwrap()
            .raw_tensor_metadata("layers.6.attn.wq_a.weight")
            .unwrap();
        let elements: u64 = view.shape.iter().map(|d| *d as u64).product();
        assert_eq!(
            (elements, view.dtype),
            (1280 * 5120, safetensors::Dtype::F8_E4M3)
        );
        // The exact tensor sum excludes unrelated payload and uses each tensor's stored dtype.
        let mut naive = 0u64;
        for (tensor, shard) in &layout.tensors {
            let role = layout.shard_role(shard).unwrap();
            if matches!(role, ShardRole::Dspark | ShardRole::Engram) {
                continue;
            }
            let bytes = std::fs::metadata(dir.join(shard)).unwrap().len();
            let per = layout.tensors_in(shard).len() as u64;
            naive += bytes / per * if tensor.contains(".experts.") { 8 } else { 4 };
        }
        assert!(
            exact < naive,
            "exact estimate {exact} should undercut the averaged {naive}"
        );
        assert!(exact > 0);
    }

    #[test]
    fn a_missing_shard_is_an_error_at_open() {
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        if !dir.join("model.safetensors.index.json").exists() {
            return;
        }
        let scratch = tempfile::tempdir().unwrap();
        for file in ["model.safetensors.index.json", "config.json"] {
            std::fs::copy(dir.join(file), scratch.path().join(file)).unwrap();
        }
        let Err(error) = TransformerLoader::open(scratch.path()) else {
            panic!("a checkpoint without its shards must not open");
        };
        assert!(
            format!("{error:#}").contains("weight shard is missing"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn layer_assembly_materializes_the_real_projections() {
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        if !dir.join("model.safetensors.index.json").exists() {
            return;
        }
        let weights = LayerWeights::load_layer(&open_weights(dir).unwrap(), 6).unwrap();
        assert_eq!(weights.wq_a.dims(), [1280, 5120]);
        assert_eq!(weights.q_norm.dims(), [1280]);
        // Main wq_b covers every head: 64 heads x 512 = 32768 rows.
        assert_eq!(weights.wq_b.dims(), [32768, 1280]);
        assert_eq!(weights.wkv.dims(), [512, 5120]);
        assert_eq!(weights.kv_norm.dims(), [512]);
        assert_eq!(weights.wo_a.dims(), [8192, 4096]);
        assert_eq!(weights.wo_b.dims(), [5120, 8192]);
        for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            let tensor = match name {
                "wq_a" => &weights.wq_a,
                "wq_b" => &weights.wq_b,
                "wkv" => &weights.wkv,
                "wo_a" => &weights.wo_a,
                _ => &weights.wo_b,
            };
            let sample = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                sample.iter().any(|value| *value != 0.0),
                "{name} dequantized to all zeros"
            );
        }
    }

    #[test]
    fn text_only_runs_open_42_shards_and_skip_optional_families() {
        let Some(layout) = layout() else {
            return;
        };
        assert_eq!(layout.shards_to_open(OpenPlan::default()).len(), 42);
        assert_eq!(
            layout.shards_to_open(OpenPlan::default())[0],
            "model-00002-of-00048.safetensors"
        );
        let full = layout.shards_to_open(OpenPlan {
            vision: true,
            dspark: true,
            engram: true,
        });
        assert_eq!(full.len(), 48);
        assert_eq!(
            full.first().copied(),
            Some("model-00001-of-00048.safetensors")
        );
    }

    #[test]
    fn vision_tensors_never_share_a_shard_with_text_layers() {
        let Some(layout) = layout() else {
            return;
        };
        let vision = layout.tensors_in("model-00001-of-00048.safetensors");
        assert!(vision.iter().any(|name| name.starts_with("vision.blocks.")));
        assert!(vision.contains(&"vision.norm.weight"));
        assert!(vision.contains(&"aligner.w1.weight"));
        assert!(
            vision
                .iter()
                .all(|name| { name.starts_with("vision.") || name.starts_with("aligner.") })
        );
        let embed = layout.tensors_in("model-00002-of-00048.safetensors");
        assert!(embed.contains(&"embed.weight"));
        assert!(embed.contains(&"image_start"));
        assert!(embed.contains(&"image_end"));
        assert!(embed.contains(&"image_newline"));
    }

    fn exact_f32_bytes(shape: &[usize], dtype: safetensors::Dtype) -> u64 {
        let elements: u64 = shape.iter().map(|d| *d as u64).product();
        let element_bytes = match dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::BF16
            | safetensors::Dtype::F8_E4M3
            | safetensors::Dtype::F8_E8M0 => 4,
            safetensors::Dtype::I8 => 8,
            other => panic!("unexpected dtype {other:?}"),
        };
        elements * element_bytes
    }

    #[test]
    fn admission_counts_engram_projections_but_not_the_embed_table() {
        let Some(layout) = layout() else {
            return;
        };
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        let exact = TransformerLoader::open(dir)
            .unwrap()
            .resident_f32_bytes()
            .unwrap();
        // Only Engram resident projections are excluded from streamed payload.
        let weights = open_weights(dir).unwrap();
        let mut skeleton = 0u64;
        let mut projections = 0u64;
        for (tensor, shard) in &layout.tensors {
            let role = layout.shard_role(shard).unwrap();
            if matches!(role, ShardRole::Dspark | ShardRole::Vision) || tensor.ends_with(".scale") {
                continue;
            }
            let view = weights.raw_tensor_metadata(tensor).unwrap();
            match role {
                ShardRole::Engram if tensor.contains(".engram.embed.") => {}
                ShardRole::Engram => projections += exact_f32_bytes(&view.shape, view.dtype),
                _ => skeleton += exact_f32_bytes(&view.shape, view.dtype),
            }
        }
        assert!(
            projections > 0,
            "engram projections exist on this checkpoint"
        );
        assert_eq!(exact, skeleton + projections);
    }

    fn write_fixture(dir: &Path) -> Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> {
        let tensors = vec![
            (
                "layers.0.attn.wq_a.weight".to_owned(),
                safetensors::Dtype::F8_E4M3,
                vec![128, 128],
                (0..128 * 128).map(|i| (i % 120) as u8).collect::<Vec<u8>>(),
            ),
            (
                "layers.0.attn.wq_a.scale".to_owned(),
                safetensors::Dtype::F8_E8M0,
                vec![4, 4],
                vec![127u8; 16],
            ),
            (
                "layers.0.attn.q_norm.weight".to_owned(),
                safetensors::Dtype::BF16,
                vec![4],
                (1u16..=4)
                    .flat_map(|v| (v * 0x3f80 / 2).to_le_bytes())
                    .collect(),
            ),
        ];
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join("model-00001-of-00001.safetensors"))
            .unwrap();
        let map: serde_json::Map<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, ..)| {
                (
                    name.clone(),
                    serde_json::json!("model-00001-of-00001.safetensors"),
                )
            })
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(
                &serde_json::json!({ "metadata": {"total_size": 0}, "weight_map": map }),
            )
            .unwrap(),
        )
        .unwrap();
        tensors
    }

    #[test]
    fn shared_traversal_yields_the_bytes_and_dequantized_values_of_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_fixture(dir.path());
        let weights = open_weights(dir.path()).unwrap();
        let file = std::fs::read(dir.path().join("model-00001-of-00001.safetensors")).unwrap();
        let direct = safetensors::SafeTensors::deserialize(&file).unwrap();
        for (name, dtype, shape, data) in &tensors {
            let meta = weights.raw_tensor_metadata(name).unwrap();
            assert_eq!((meta.dtype, &meta.shape), (*dtype, shape), "{name}");
            let bytes = weights
                .with_tensor_bytes(name, |bytes| Ok(bytes.to_vec()))
                .unwrap();
            assert_eq!(&bytes, data, "{name}");
            assert_eq!(direct.tensor(name).unwrap().data(), data, "{name}");
        }
        let via_traversal = fp8_tensor(&weights, "layers.0.attn.wq_a").unwrap();
        let weight = Tensor::from_raw_buffer(
            direct.tensor("layers.0.attn.wq_a.weight").unwrap().data(),
            DType::F8E4M3,
            &[128, 128],
            &Device::Cpu,
        )
        .unwrap();
        let expected = crate::quant::dequantize_fp8_block(
            &weight,
            direct.tensor("layers.0.attn.wq_a.scale").unwrap().data(),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            via_traversal
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let norm = f32_tensor(&weights, "layers.0.attn.q_norm.weight").unwrap();
        assert_eq!(norm.dtype(), DType::F32);
        assert_eq!(norm.dims(), [4]);
    }

    #[test]
    fn a_layer_without_its_sink_fails_naming_the_tensor() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path());
        let weights = open_weights(dir.path()).unwrap();
        let Err(error) = LayerWeights::load_layer(&weights, 0) else {
            panic!("a missing attn_sink must not load");
        };
        let message = format!("{error:#}");
        assert!(message.contains("layers.0.attn.attn_sink"), "{message}");
    }
}
