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

    pub fn shard_of(&self, tensor: &str) -> Option<&str> {
        self.tensors.get(tensor).map(String::as_str)
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

use candle_core::safetensors::Load as _;
use candle_core::{Device, Tensor};

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

fn open_shard(
    model_dir: &Path,
    shard: &str,
) -> Result<(memmap2::Mmap, safetensors::SafeTensors<'static>)> {
    let file =
        std::fs::File::open(model_dir.join(shard)).with_context(|| format!("open {shard}"))?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let bytes: &'static [u8] = unsafe { std::mem::transmute(&map[..]) };
    let tensors = safetensors::SafeTensors::deserialize(bytes)?;
    Ok((map, tensors))
}

/// `projection` is the name without the `.weight` suffix; the scale hangs off
/// the projection name (`wq_a.weight` pairs with `wq_a.scale`).
fn body_fp8(
    tensors: &safetensors::SafeTensors<'_>,
    projection: &str,
    device: &Device,
) -> Result<Tensor> {
    let weight_name = format!("{projection}.weight");
    let scale_name = format!("{projection}.scale");
    let weight = tensors
        .tensor(&weight_name)
        .with_context(|| weight_name.clone())?
        .load(device)?;
    let scale = tensors
        .tensor(&scale_name)
        .with_context(|| scale_name.clone())?;
    crate::quant::dequantize_fp8_block(&weight, scale.data(), device)
        .with_context(|| format!("dequantize {weight_name}"))
}

fn bf16(tensors: &safetensors::SafeTensors<'_>, name: &str, device: &Device) -> Result<Tensor> {
    Ok(tensors
        .tensor(name)
        .with_context(|| name.to_owned())?
        .load(device)?
        .to_dtype(candle_core::DType::F32)?)
}

impl LayerWeights {
    pub fn load_layer(model_dir: &Path, layout: &CheckpointLayout, layer: usize) -> Result<Self> {
        let device = Device::Cpu;
        let prefix = format!("layers.{layer}.attn.");
        let shard_of = |name: &str| -> Result<(memmap2::Mmap, safetensors::SafeTensors<'static>)> {
            let shard = layout
                .shard_of(name)
                .with_context(|| format!("{name} is not in the index"))?;
            open_shard(model_dir, shard)
        };
        let sink = match shard_of(&format!("{prefix}attn_sink")) {
            Ok((map, tensors)) => {
                let loaded = bf16(&tensors, &format!("{prefix}attn_sink"), &device);
                drop(map);
                loaded?
            }
            Err(_) => Tensor::zeros(1, candle_core::DType::F32, &device)?,
        };
        // Every projection of one layer shares a shard in the real layout;
        // open once through the first name and load the rest from it.
        let (_map, tensors) = shard_of(&format!("{prefix}wq_a.weight"))?;
        Ok(Self {
            sink,
            wq_a: body_fp8(&tensors, &format!("{prefix}wq_a"), &device)?,
            q_norm: bf16(&tensors, &format!("{prefix}q_norm.weight"), &device)?,
            wq_b: body_fp8(&tensors, &format!("{prefix}wq_b"), &device)?,
            wkv: body_fp8(&tensors, &format!("{prefix}wkv"), &device)?,
            kv_norm: bf16(&tensors, &format!("{prefix}kv_norm.weight"), &device)?,
            wo_a: body_fp8(&tensors, &format!("{prefix}wo_a"), &device)?,
            wo_b: body_fp8(&tensors, &format!("{prefix}wo_b"), &device)?,
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
use std::collections::HashMap;

/// Assembles a host-reference `Transformer` from a checkpoint directory.
/// Every projection is materialized F32 at load: the static skeleton and the
/// routed experts of all layers resident together. `open` refuses loudly
/// when the dequantized set cannot fit the host instead of degrading.
pub struct TransformerLoader {
    model_dir: std::path::PathBuf,
    config: DeepseekV41Config,
    layout: CheckpointLayout,
    shards: HashMap<String, (memmap2::Mmap, safetensors::SafeTensors<'static>)>,
}

impl TransformerLoader {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config = DeepseekV41Config::from_model_dir(model_dir)?;
        let layout = CheckpointLayout::open(model_dir)?;
        Ok(Self {
            model_dir: model_dir.to_owned(),
            config,
            layout,
            shards: HashMap::new(),
        })
    }

    pub fn config(&self) -> &DeepseekV41Config {
        &self.config
    }

    fn tensors(&mut self, name: &str) -> Result<&safetensors::SafeTensors<'static>> {
        let shard = self
            .layout
            .shard_of(name)
            .with_context(|| format!("{name} is not in the index"))?
            .to_owned();
        let model_dir = self.model_dir.clone();
        let entry = self.shards.entry(shard).or_insert_with(|| {
            open_shard(&model_dir, self.layout.shard_of(name).unwrap_or_default()).unwrap()
        });
        let (_, tensors) = entry;
        Ok(tensors)
    }

    fn f32(&mut self, name: &str) -> Result<Tensor> {
        let device = Device::Cpu;
        let view = self
            .tensors(name)?
            .tensor(name)
            .with_context(|| format!("view {name}"))?;
        view.load(&device)
            .with_context(|| format!("load {name}"))?
            .to_dtype(candle_core::DType::F32)
            .with_context(|| format!("cast {name}"))
    }

    fn fp8(&mut self, projection: &str) -> Result<Tensor> {
        let device = Device::Cpu;
        let weight_name = format!("{projection}.weight");
        let scale_name = format!("{projection}.scale");
        let weight = self
            .tensors(&weight_name)?
            .tensor(&weight_name)
            .with_context(|| weight_name.clone())?
            .load(&device)?;
        let scale = self
            .tensors(&scale_name)?
            .tensor(&scale_name)
            .with_context(|| scale_name.clone())?;
        crate::quant::dequantize_fp8_block(&weight, scale.data(), &device)
            .with_context(|| format!("dequantize {weight_name}"))
    }

    fn fp4(&mut self, weight_name: &str, rows: usize, columns: usize) -> Result<Tensor> {
        let device = Device::Cpu;
        let scale_name = weight_name.replace(".weight", ".scale");
        let payload = self
            .tensors(weight_name)?
            .tensor(weight_name)
            .with_context(|| weight_name.to_owned())?;
        let scale = self
            .tensors(&scale_name)?
            .tensor(&scale_name)
            .with_context(|| scale_name.to_owned())?;
        crate::quant::dequantize_fp4_packed(payload.data(), rows, columns, scale.data(), &device)
            .with_context(|| format!("dequantize {weight_name}"))
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
            if matches!(role, ShardRole::Dspark | ShardRole::Engram) {
                continue;
            }
            let bytes = std::fs::metadata(self.model_dir.join(shard))?.len();
            let per_shard_tensors = self.layout.tensors_in(shard).len() as u64;
            let tensor_bytes = bytes.checked_div(per_shard_tensors).unwrap_or(0);
            let factor = if tensor.starts_with("layers.") && tensor.contains(".experts.") {
                8
            } else {
                4
            };
            total += tensor_bytes * factor;
        }
        Ok(total)
    }

    /// Load the whole transformer onto the host. `tokenizer` is needed only
    /// when the checkpoint carries engram layers.
    pub fn load(
        mut self,
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
            Some(NgramHashState::new(layout, token_map, 1, max_seq)?)
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
        &mut self,
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
            ring: WindowRing::new(1, sliding_window, head_dim, &device)?,
            indexer_wq_b,
            indexer_wk,
            indexer_k_norm,
            weights_proj,
            freqs,
            index_head_dim: self.config.text_config.index_head_dim,
            index_heads: self.config.text_config.index_n_heads,
            index_topk: self.config.text_config.index_topk,
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

    fn load_ffn(&mut self, layer: usize) -> Result<FfnWeights> {
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
        let load_expert = |loader: &mut Self, projection: &str| -> Result<Expert> {
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

    fn load_engram(&mut self, layer: usize, hash_index: usize) -> Result<EngramCore> {
        let text = &self.config.text_config;
        let prefix = format!("layers.{layer}.engram.");
        let weight_name = format!("{prefix}embed.weight");
        let scale_name = format!("{prefix}embed.scale");
        let head_dim = text.engram_head_dim;
        let model_dir = self.model_dir.clone();
        let weight_shard = self
            .layout
            .shard_of(&weight_name)
            .with_context(|| format!("{weight_name} is not in the index"))?
            .to_owned();
        let scale_shard = self
            .layout
            .shard_of(&scale_name)
            .with_context(|| format!("{scale_name} is not in the index"))?
            .to_owned();
        let lookup: EngramLookup = Box::new(move |ids: &[i64]| -> Vec<f32> {
            let mut rows = Vec::with_capacity(ids.len() * head_dim);
            let (weight_map, weight_view) =
                open_shard(&model_dir, &weight_shard).expect("engram payload shard");
            let (scale_map, scale_view) =
                open_shard(&model_dir, &scale_shard).expect("engram scale shard");
            let weight = weight_view
                .tensor(&weight_name)
                .expect("engram payload view");
            let scale = scale_view.tensor(&scale_name).expect("engram scale view");
            let payload = weight.data();
            let scales = scale.data();
            for id in ids {
                let row = *id as usize;
                for d in 0..head_dim {
                    let byte = payload[row * head_dim + d];
                    let group = d / crate::config::FP8_WEIGHT_BLOCK;
                    let exponent =
                        scales[row * (head_dim / crate::config::FP8_WEIGHT_BLOCK) + group] as i32
                            - 127;
                    rows.push(crate::quant::e4m3_byte_to_f32(byte) * 2.0f32.powi(exponent));
                }
            }
            drop(weight_map);
            drop(scale_map);
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

    fn layout() -> Option<CheckpointLayout> {
        let dir = Path::new("../../models/deepseek-ai/DeepSeek-V4.1-Flash");
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
    fn layer_assembly_materializes_the_real_projections() {
        let dir = Path::new("../../models/deepseek-ai/DeepSeek-V4.1-Flash");
        if !dir.join("model.safetensors.index.json").exists() {
            return;
        }
        let layout = CheckpointLayout::open(dir).unwrap();
        let weights = LayerWeights::load_layer(dir, &layout, 6).unwrap();
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
}
