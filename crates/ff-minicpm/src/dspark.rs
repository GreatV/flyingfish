//! DSpark's Qwen3 draft, target-feature KV conditioning and greedy verification.
//!
//! The draft uses the target's embeddings and output head. Its block is
//! bidirectional; only target verification is causal. Target layer IDs name
//! zero-based block outputs (HF hidden_states[id + 1]), before final RMSNorm.
use super::{Config as TargetConfig, Decoder, attention_with_mask};
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use ff_core::weights::ModelWeights;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    architectures: Vec<String>,
    model_type: String,
    hidden_size: usize,
    pub(crate) intermediate_size: usize,
    num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    draft_vocab_size: usize,
    num_target_layers: usize,
    pub(crate) target_layer_ids: Vec<usize>,
    pub(crate) block_size: usize,
    mask_token_id: u32,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_parameters: RopeParameters,
    attention_bias: bool,
    mlp_bias: bool,
    hidden_act: String,
    layer_types: Vec<String>,
    attention_mode: String,
    projector_type: String,
    markov_head_type: String,
    markov_rank: usize,
    enable_confidence_head: bool,
    confidence_head_with_markov: bool,
    #[serde(default = "default_true")]
    sample_from_anchor: bool,
    #[serde(default)]
    is_causal: bool,
    #[serde(default)]
    use_sliding_window: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    rope_type: String,
}

impl Config {
    pub(crate) fn kv_layer_bytes(&self, tokens: usize, element_bytes: u64) -> Result<u64> {
        super::memory::cache_bytes(
            1,
            self.num_key_value_heads,
            self.head_dim,
            tokens,
            element_bytes,
        )
    }

    pub(crate) fn kv_cache_bytes(&self, tokens: usize, element_bytes: u64) -> Result<u64> {
        anyhow::ensure!(
            tokens <= self.max_position_embeddings,
            "request exceeds draft KV context limit"
        );
        super::memory::cache_bytes(
            self.num_hidden_layers,
            self.num_key_value_heads,
            self.head_dim,
            tokens,
            element_bytes,
        )
    }

    pub fn read(root: &Path) -> Result<Self> {
        serde_json::from_slice(&std::fs::read(root.join("config.json"))?)
            .context("invalid DSpark config")
    }

    pub fn validate_for(&self, target: &TargetConfig) -> Result<()> {
        target.validate()?;
        anyhow::ensure!(
            self.architectures == ["Qwen3DSparkModel"] && self.model_type == "qwen3",
            "expected a Qwen3DSparkModel draft checkpoint"
        );
        anyhow::ensure!(
            self.hidden_size == target.hidden_size
                && self.vocab_size == target.vocab_size
                && self.draft_vocab_size == target.vocab_size
                && self.num_target_layers == target.num_hidden_layers
                && self.head_dim == target.head_dim
                && self.rope_parameters.rope_theta == target.rope_theta,
            "DSpark configuration does not match the target model"
        );
        anyhow::ensure!(
            self.num_hidden_layers > 0
                && self.intermediate_size > 0
                && self.num_attention_heads > 0
                && self.num_key_value_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads)
                && self.num_attention_heads.checked_mul(self.head_dim) == Some(self.hidden_size)
                && self.block_size > 0
                && self.block_size <= self.max_position_embeddings
                && (self.mask_token_id as usize) < self.vocab_size
                && self.markov_rank > 0,
            "invalid DSpark dimensions, block size or mask token"
        );
        anyhow::ensure!(
            !self.target_layer_ids.is_empty()
                && self
                    .target_layer_ids
                    .iter()
                    .all(|&id| id < target.num_hidden_layers)
                && self.target_layer_ids.windows(2).all(|ids| ids[0] < ids[1]),
            "DSpark target_layer_ids must be sorted, distinct target block IDs"
        );
        anyhow::ensure!(
            !self.attention_bias
                && !self.mlp_bias
                && self.hidden_act == "silu"
                && self.rope_parameters.rope_type == "default"
                && self.attention_mode == "gqa"
                && self.projector_type == "dspark"
                && self.markov_head_type == "vanilla"
                && self.sample_from_anchor
                && !self.is_causal
                && !self.use_sliding_window
                && self.layer_types.len() == self.num_hidden_layers
                && self.layer_types.iter().all(|kind| kind == "full_attention"),
            "unsupported DSpark attention, projector, Markov head or anchor policy"
        );
        anyhow::ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.,
            "invalid DSpark RMSNorm epsilon"
        );
        Ok(())
    }

    fn validate_weights(&self, weights: &ModelWeights) -> Result<()> {
        let h = self.hidden_size;
        let kv = self.num_key_value_heads * self.head_dim;
        let check = |name: &str, shape: &[usize]| -> Result<()> {
            let metadata = weights.metadata(name)?;
            anyhow::ensure!(
                metadata.shape == shape,
                "DSpark {name}: expected {shape:?}, got {:?}",
                metadata.shape
            );
            Ok(())
        };
        let features = h
            .checked_mul(self.target_layer_ids.len())
            .context("DSpark feature width overflow")?;
        check("fc.weight", &[h, features])?;
        check("hidden_norm.weight", &[h])?;
        check("norm.weight", &[h])?;
        for name in ["markov_w1", "markov_w2"] {
            check(
                &format!("markov_head.{name}.weight"),
                &[self.vocab_size, self.markov_rank],
            )?;
        }
        if self.enable_confidence_head {
            let rank = if self.confidence_head_with_markov {
                self.markov_rank
            } else {
                0
            };
            check(
                "confidence_head.proj.weight",
                &[1, h.checked_add(rank).context("confidence width overflow")?],
            )?;
            check("confidence_head.proj.bias", &[1])?;
        }
        for layer in 0..self.num_hidden_layers {
            let p = format!("layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                check(&format!("{p}.{name}.weight"), &[h])?;
            }
            for name in ["q_norm", "k_norm"] {
                check(&format!("{p}.self_attn.{name}.weight"), &[self.head_dim])?;
            }
            for (name, out, input) in [
                ("self_attn.q_proj", h, h),
                ("self_attn.k_proj", kv, h),
                ("self_attn.v_proj", kv, h),
                ("self_attn.o_proj", h, h),
                ("mlp.gate_proj", self.intermediate_size, h),
                ("mlp.up_proj", self.intermediate_size, h),
                ("mlp.down_proj", h, self.intermediate_size),
            ] {
                check(&format!("{p}.{name}.weight"), &[out, input])?;
            }
        }
        Ok(())
    }
}

pub struct Draft {
    config: Config,
    weights: ModelWeights,
    device: Device,
    dtype: DType,
    cache: Vec<Option<(Tensor, Tensor)>>,
    position: usize,
}

struct Proposal {
    tokens: Vec<u32>,
    confidence: Vec<f32>,
}

impl Draft {
    pub fn new(config: Config, weights: ModelWeights, target: &Decoder) -> Result<Self> {
        config.validate_for(target.config())?;
        config.validate_weights(&weights)?;
        let cache = vec![None; config.num_hidden_layers];
        Ok(Self {
            config,
            weights,
            device: target.device.clone(),
            dtype: target.dtype,
            cache,
            position: 0,
        })
    }

    fn clear(&mut self) {
        self.cache.fill(None);
        self.position = 0;
    }

    /// Return only the next token so prompt-wide features/logits are released
    /// before entering the generation loop. The committed KV is sufficient.
    fn prefill_anchor(&mut self, target: &mut Decoder, prompt: &[u32]) -> Result<u32> {
        let prefill = target.forward_capture(prompt, &self.config.target_layer_ids, false)?;
        self.commit(
            prefill
                .features
                .as_ref()
                .context("missing target features")?,
        )?;
        Ok(prefill.logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?)
    }
    fn get(&self, name: &str) -> Result<Tensor> {
        Ok(self
            .weights
            .load(name, &self.device)?
            .to_dtype(self.dtype)?)
    }
    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        Ok(x.matmul(&self.get(&format!("{prefix}.weight"))?.t()?)?)
    }
    fn norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let wide = x.to_dtype(DType::F32)?;
        let variance = wide.sqr()?.mean_keepdim(D::Minus1)?;
        Ok(wide
            .broadcast_div(&(variance + self.config.rms_norm_eps)?.sqrt()?)?
            .to_dtype(self.dtype)?
            .broadcast_mul(&self.get(&format!("{prefix}.weight"))?)?)
    }
    fn rotary(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, length, dim) = x.dims4()?;
        let half = dim / 2;
        let theta = self.config.rope_parameters.rope_theta as f32;
        let angles = (offset..offset + length)
            .flat_map(|position| {
                (0..half).map(move |i| position as f32 / theta.powf((2 * i) as f32 / dim as f32))
            })
            .collect::<Vec<_>>();
        let angles = Tensor::from_vec(angles, (length, half), &self.device)?;
        Ok(candle_nn::rotary_emb::rope(
            &x.contiguous()?,
            &angles.cos()?.to_dtype(self.dtype)?,
            &angles.sin()?.to_dtype(self.dtype)?,
        )?)
    }
    fn heads(&self, x: &Tensor, heads: usize) -> Result<Tensor> {
        Ok(x.reshape((1, x.dim(0)?, heads, self.config.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// Only verified target features enter the durable draft KV cache. The
    /// same projected features feed every draft layer, without input_layernorm.
    fn commit(&mut self, features: &Tensor) -> Result<()> {
        let (rows, width) = features.dims2()?;
        anyhow::ensure!(
            rows > 0 && width == self.config.hidden_size * self.config.target_layer_ids.len(),
            "invalid DSpark target feature shape"
        );
        let end = self
            .position
            .checked_add(rows)
            .context("DSpark context overflow")?;
        anyhow::ensure!(
            end <= self.config.max_position_embeddings,
            "DSpark context limit exceeded"
        );
        let result = self.commit_layers(features);
        if let Err(error) = result {
            self.clear();
            return Err(error);
        }
        self.position = end;
        Ok(())
    }

    fn commit_layers(&mut self, features: &Tensor) -> Result<()> {
        let hidden = self.norm(&self.linear(features, "fc")?, "hidden_norm")?;
        for layer in 0..self.config.num_hidden_layers {
            let p = format!("layers.{layer}.self_attn");
            let key = self.heads(
                &self.linear(&hidden, &format!("{p}.k_proj"))?,
                self.config.num_key_value_heads,
            )?;
            let key = self.rotary(&self.norm(&key, &format!("{p}.k_norm"))?, self.position)?;
            let value = self.heads(
                &self.linear(&hidden, &format!("{p}.v_proj"))?,
                self.config.num_key_value_heads,
            )?;
            let pair = match &self.cache[layer] {
                None => (key, value),
                Some((k, v)) => (Tensor::cat(&[k, &key], 2)?, Tensor::cat(&[v, &value], 2)?),
            };
            self.cache[layer] = Some(pair);
        }
        Ok(())
    }

    fn propose(&self, target: &Decoder, anchor: u32, count: usize) -> Result<Proposal> {
        anyhow::ensure!(
            self.position == target.position && count > 0 && count <= self.config.block_size,
            "invalid DSpark proposal position or length"
        );
        anyhow::ensure!(
            self.position + count <= self.config.max_position_embeddings,
            "draft query exceeds context"
        );
        let mut tokens = vec![self.config.mask_token_id; count];
        tokens[0] = anchor;
        let mut hidden = target
            .weights
            .load_rows("model.embed_tokens.weight", &tokens, &self.device)?
            .to_dtype(self.dtype)?;
        for layer in 0..self.config.num_hidden_layers {
            let p = format!("layers.{layer}");
            let x = self.norm(&hidden, &format!("{p}.input_layernorm"))?;
            let project = |name: &str, heads: usize| -> Result<Tensor> {
                self.heads(&self.linear(&x, &format!("{p}.self_attn.{name}"))?, heads)
            };
            let q = self.rotary(
                &self.norm(
                    &project("q_proj", self.config.num_attention_heads)?,
                    &format!("{p}.self_attn.q_norm"),
                )?,
                self.position,
            )?;
            let k = self.rotary(
                &self.norm(
                    &project("k_proj", self.config.num_key_value_heads)?,
                    &format!("{p}.self_attn.k_norm"),
                )?,
                self.position,
            )?;
            let v = project("v_proj", self.config.num_key_value_heads)?;
            let (past_k, past_v) = self.cache[layer]
                .as_ref()
                .context("draft prefix is not initialized")?;
            let k = Tensor::cat(&[past_k, &k], 2)?;
            let v = Tensor::cat(&[past_v, &v], 2)?;
            let context = attention_with_mask(&q, &k, &v, None, target.query_chunk)?
                .transpose(1, 2)?
                .reshape((count, self.config.hidden_size))?
                .contiguous()?;
            hidden = (&hidden + self.linear(&context, &format!("{p}.self_attn.o_proj"))?)?;
            let x = self.norm(&hidden, &format!("{p}.post_attention_layernorm"))?;
            let gate = candle_nn::ops::silu(&self.linear(&x, &format!("{p}.mlp.gate_proj"))?)?;
            let up = self.linear(&x, &format!("{p}.mlp.up_proj"))?;
            hidden = (&hidden + self.linear(&(&gate * up)?, &format!("{p}.mlp.down_proj"))?)?;
        }
        let hidden = self.norm(&hidden, "norm")?;
        let head = if target.config.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        };
        let base_logits = target.linear(&hidden, head)?.to_dtype(DType::F32)?;
        let mut tokens = Vec::with_capacity(count);
        let mut confidence = Vec::new();
        let markov_w2 = self.get("markov_head.markov_w2.weight")?;
        let mut previous = anchor;
        for row in 0..count {
            let previous_embedding = self
                .weights
                .load_rows("markov_head.markov_w1.weight", &[previous], &self.device)?
                .to_dtype(self.dtype)?;
            let bias = previous_embedding
                .matmul(&markov_w2.t()?)?
                .to_dtype(DType::F32)?;
            previous = (&base_logits.narrow(0, row, 1)? + &bias)?
                .squeeze(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            tokens.push(previous);
            if self.config.enable_confidence_head {
                let feature = hidden.narrow(0, row, 1)?;
                let feature = if self.config.confidence_head_with_markov {
                    Tensor::cat(&[&feature, &previous_embedding], 1)?
                } else {
                    feature
                };
                let w = self
                    .weights
                    .load("confidence_head.proj.weight", &self.device)?
                    .to_dtype(DType::F32)?;
                let b = self
                    .weights
                    .load("confidence_head.proj.bias", &self.device)?
                    .to_dtype(DType::F32)?;
                let raw = feature
                    .to_dtype(DType::F32)?
                    .matmul(&w.t()?)?
                    .broadcast_add(&b)?;
                confidence.push(
                    candle_nn::ops::sigmoid(&raw)?
                        .squeeze(0)?
                        .squeeze(0)?
                        .to_scalar::<f32>()?,
                );
            }
        }
        Ok(Proposal { tokens, confidence })
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GenerationStats {
    pub output_tokens: usize,
    pub draft_forwards: usize,
    pub target_verify_forwards: usize,
    pub proposed_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub rejected_blocks: usize,
    pub confidence_sum: f64,
    pub confidence_count: usize,
    pub target_decode_forwards: usize,
    pub fell_back_to_target: bool,
    pub observed_target_ms_per_token: Option<f64>,
    pub observed_speculative_ms_per_token: Option<f64>,
}

#[derive(Default)]
struct SpeculationCost {
    single_seconds: f64,
    single_steps: usize,
    speculative_seconds: f64,
    speculative_tokens: usize,
    speculative_blocks: usize,
    fallback: bool,
}

impl SpeculationCost {
    fn needs_probe(&self) -> bool {
        self.single_steps < 2
    }

    fn observe_single(&mut self, seconds: f64) {
        self.single_seconds += seconds;
        self.single_steps += 1;
    }

    fn observe_speculation(&mut self, seconds: f64, tokens: usize) {
        self.speculative_blocks += 1;
        if self.speculative_blocks == 1 {
            return;
        }
        self.speculative_seconds += seconds;
        self.speculative_tokens += tokens;
        if self.speculative_blocks >= 3 && self.single_steps >= 2 {
            let serial = self.single_seconds / self.single_steps as f64;
            let speculative = self.speculative_seconds / self.speculative_tokens as f64;
            self.fallback = speculative > serial * 1.1;
        }
    }
}

/// Each call starts a new request. Failed computations/callbacks clear both
/// models, so a partially verified request can never be reused as a prefix.
pub fn generate_greedy(
    target: &mut Decoder,
    draft: &mut Draft,
    prompt: &[u32],
    max_new_tokens: usize,
    emit: impl FnMut(u32) -> Result<()>,
) -> Result<GenerationStats> {
    let adaptive = target.batch_invariant_decode;
    generate_greedy_impl(
        target,
        draft,
        prompt,
        max_new_tokens,
        adaptive.then(SpeculationCost::default),
        emit,
    )
}

fn generate_greedy_impl(
    target: &mut Decoder,
    draft: &mut Draft,
    prompt: &[u32],
    max_new_tokens: usize,
    cost: Option<SpeculationCost>,
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<GenerationStats> {
    draft.config.validate_for(&target.config)?;
    anyhow::ensure!(
        max_new_tokens > 0 && !prompt.is_empty(),
        "prompt and output limit must be nonempty"
    );
    let context = target
        .config
        .max_position_embeddings
        .min(draft.config.max_position_embeddings);
    anyhow::ensure!(
        prompt
            .len()
            .checked_add(max_new_tokens)
            .is_some_and(|end| end <= context),
        "request exceeds the target/draft context limit"
    );
    target.clear_cache();
    draft.clear();
    let adaptive = cost.is_some();
    let mut cost = cost.unwrap_or_default();
    let result = (|| -> Result<GenerationStats> {
        let layers = draft.config.target_layer_ids.clone();
        let mut anchor = draft.prefill_anchor(target, prompt)?;
        let mut stats = GenerationStats::default();
        while stats.output_tokens < max_new_tokens && !target.config.eos_token_id.contains(&anchor)
        {
            let remaining = max_new_tokens - stats.output_tokens;
            if remaining == 1 {
                emit(anchor)?;
                stats.output_tokens += 1;
                break;
            }
            if adaptive && (cost.needs_probe() || cost.fallback) {
                let started = std::time::Instant::now();
                let (next, features) = if cost.fallback {
                    stats.fell_back_to_target = true;
                    (
                        target.forward(&[anchor])?.argmax(0)?.to_scalar::<u32>()?,
                        None,
                    )
                } else {
                    let output = target.forward_capture(&[anchor], &layers, false)?;
                    (
                        output.logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?,
                        output.features,
                    )
                };
                if cost.needs_probe() {
                    cost.observe_single(started.elapsed().as_secs_f64());
                }
                if let Some(features) = features {
                    draft.commit(&features)?;
                    target.device.synchronize()?;
                }
                stats.target_decode_forwards += 1;
                emit(anchor)?;
                stats.output_tokens += 1;
                anchor = next;
                continue;
            }
            let speculation_started = std::time::Instant::now();
            let count = draft.config.block_size.min(remaining - 1);
            let proposal = draft.propose(target, anchor, count)?;
            stats.draft_forwards += 1;
            stats.proposed_tokens += proposal.tokens.len();
            stats.confidence_count += proposal.confidence.len();
            stats.confidence_sum += proposal
                .confidence
                .iter()
                .map(|&x| f64::from(x))
                .sum::<f64>();
            let mut block = vec![anchor];
            block.extend_from_slice(&proposal.tokens);
            let start = target.position;
            let verified = target.forward_capture(&block, &layers, true)?;
            stats.target_verify_forwards += 1;
            let predictions = verified.logits.argmax(1)?.to_vec1::<u32>()?;
            let accepted = accepted_prefix(&proposal.tokens, &predictions)?;
            stats.accepted_draft_tokens += accepted;
            stats.rejected_blocks += usize::from(accepted < proposal.tokens.len());
            let consumed = accepted + 1;
            target.truncate_cache(start + consumed)?;
            draft.commit(
                &verified
                    .features
                    .context("missing verification features")?
                    .narrow(0, 0, consumed)?,
            )?;
            let speculation_seconds = if adaptive {
                target.device.synchronize()?;
                speculation_started.elapsed().as_secs_f64()
            } else {
                0.
            };
            for &token in &block[..consumed] {
                if target.config.eos_token_id.contains(&token) {
                    return Ok(stats);
                }
                emit(token)?;
                stats.output_tokens += 1;
            }
            anchor = predictions[accepted];
            if adaptive {
                cost.observe_speculation(speculation_seconds, consumed);
                if cost.fallback {
                    draft.clear();
                }
            }
        }
        Ok(stats)
    })();
    target.clear_cache();
    draft.clear();
    result.map(|mut stats| {
        stats.observed_target_ms_per_token =
            (cost.single_steps > 0).then(|| cost.single_seconds * 1000. / cost.single_steps as f64);
        stats.observed_speculative_ms_per_token = (cost.speculative_tokens > 0)
            .then(|| cost.speculative_seconds * 1000. / cost.speculative_tokens as f64);
        stats
    })
}

fn accepted_prefix(proposal: &[u32], predictions: &[u32]) -> Result<usize> {
    anyhow::ensure!(
        predictions.len() == proposal.len() + 1,
        "invalid verification logits length"
    );
    Ok(proposal
        .iter()
        .zip(predictions)
        .take_while(|(draft, target)| draft == target)
        .count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{decoder, fixture};
    use ff_core::weights::{CachePolicy, WeightSource};
    use std::collections::HashMap;

    #[test]
    fn partial_kv_replacement_failure_clears_state_and_allows_reuse() -> Result<()> {
        let (root, config) = fixture();
        let mut target = decoder(root.path(), config, 2);
        let (_draft_root, mut draft) = draft_fixture(&target);
        let features = target
            .forward_capture(&[2, 3, 4], &[0, 1], false)?
            .features
            .unwrap();
        draft.commit(&features)?;
        target.cache[1] = None;
        assert!(target.truncate_cache(2).is_err());
        assert_eq!(target.position, 0);
        assert!(target.cache.iter().all(Option::is_none));
        let _ = target.forward(&[2, 3])?;
        draft.config.num_hidden_layers = 2;
        draft.cache.push(None);
        assert!(draft.commit(&features.narrow(0, 0, 1)?).is_err());
        assert_eq!(draft.position, 0);
        assert!(draft.cache.iter().all(Option::is_none));
        draft.config.num_hidden_layers = 1;
        draft.cache.truncate(1);
        draft.commit(&features)?;
        assert_eq!(draft.position, 3);
        Ok(())
    }

    #[test]
    fn joint_kv_estimate_counts_both_live_caches_and_replacement_peak() -> Result<()> {
        let (root, config) = fixture();
        let mut target = decoder(root.path(), config, 2);
        let (_draft_root, mut draft) = draft_fixture(&target);
        let estimate = crate::memory::estimate_kv_memory(
            target.config(),
            Some(&draft.config),
            5,
            &Device::Cpu,
        )?;
        for tokens in [&[2, 3, 4][..], &[5, 6][..]] {
            let output = target.forward_capture(tokens, &draft.config.target_layer_ids, false)?;
            draft.commit(output.features.as_ref().unwrap())?;
        }
        let bytes = |cache: &[Option<(Tensor, Tensor)>]| {
            cache
                .iter()
                .flatten()
                .map(|(k, v)| {
                    (k.elem_count() * k.dtype().size_in_bytes()
                        + v.elem_count() * v.dtype().size_in_bytes()) as u64
                })
                .sum::<u64>()
        };
        assert_eq!(estimate.target_bytes, bytes(&target.cache));
        assert_eq!(estimate.draft_bytes, bytes(&draft.cache));
        assert_eq!(
            estimate.peak_bytes,
            bytes(&target.cache)
                + bytes(&draft.cache)
                + (bytes(&target.cache) / target.config.num_hidden_layers as u64)
                    .max(bytes(&draft.cache) / draft.config.num_hidden_layers as u64)
        );
        draft.config.max_position_embeddings = 4;
        assert!(
            crate::memory::estimate_kv_memory(
                target.config(),
                Some(&draft.config),
                5,
                &Device::Cpu
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn speculation_cost_ignores_cold_start_and_requires_warm_evidence() {
        let mut cost = SpeculationCost::default();
        cost.observe_single(0.01);
        assert!(cost.needs_probe());
        cost.observe_single(0.01);
        assert!(!cost.needs_probe());
        cost.observe_speculation(10., 1);
        cost.observe_speculation(0.02, 1);
        assert!(!cost.fallback);
        cost.observe_speculation(0.02, 1);
        assert!(cost.fallback);

        let mut fast = SpeculationCost::default();
        fast.observe_single(0.01);
        fast.observe_single(0.01);
        fast.observe_speculation(10., 1);
        fast.observe_speculation(0.04, 8);
        fast.observe_speculation(0.04, 8);
        assert!(!fast.fallback);
    }

    fn serial_tokens(target: &mut Decoder, prompt: &[u32], limit: usize) -> Result<Vec<u32>> {
        target.clear_cache();
        let mut input = prompt.to_vec();
        let mut tokens = Vec::new();
        for _ in 0..limit {
            let token = target.forward(&input)?.argmax(0)?.to_scalar::<u32>()?;
            if target.config.eos_token_id.contains(&token) {
                break;
            }
            tokens.push(token);
            input = vec![token];
        }
        target.clear_cache();
        Ok(tokens)
    }

    #[test]
    fn adaptive_fallback_preserves_tokens_and_cleans_up_callback_failure() -> Result<()> {
        let (dir, config) = fixture();
        let mut target = decoder(dir.path(), config, 2);
        let (_draft_dir, mut draft) = draft_fixture(&target);
        let expected = serial_tokens(&mut target, &[3, 5], 10)?;
        let prior = || {
            Some(SpeculationCost {
                single_steps: 2,
                speculative_blocks: 2,
                speculative_tokens: 2,
                ..Default::default()
            })
        };
        let mut actual = Vec::new();
        let stats = generate_greedy_impl(&mut target, &mut draft, &[3, 5], 10, prior(), |token| {
            actual.push(token);
            Ok(())
        })?;
        assert_eq!(actual, expected);
        assert_eq!(stats.draft_forwards, 1);
        assert!(stats.fell_back_to_target && stats.target_decode_forwards > 0);
        let mut emitted = 0;
        assert!(
            generate_greedy_impl(&mut target, &mut draft, &[3, 5], 10, prior(), |_| {
                emitted += 1;
                anyhow::ensure!(emitted < 6, "cancel during direct decoding");
                Ok(())
            })
            .is_err()
        );
        assert_eq!(target.position, 0);
        assert_eq!(draft.position, 0);
        assert!(target.cache.iter().all(Option::is_none));
        assert!(draft.cache.iter().all(Option::is_none));
        Ok(())
    }

    fn draft_fixture(target: &Decoder) -> (tempfile::TempDir, Draft) {
        let config: Config = serde_json::from_value(serde_json::json!({
            "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3", "hidden_size": 8,
            "intermediate_size": 12, "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 16, "draft_vocab_size": 16,
            "num_target_layers": 2, "target_layer_ids": [0, 1], "block_size": 3,
            "mask_token_id": 15, "max_position_embeddings": 16, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000., "rope_type": "default"},
            "attention_bias": false, "mlp_bias": false, "hidden_act": "silu",
            "layer_types": ["full_attention"], "attention_mode": "gqa", "projector_type": "dspark",
            "markov_head_type": "vanilla", "markov_rank": 2, "enable_confidence_head": true,
            "confidence_head_with_markov": true
        }))
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        let mut insert = |name: &str, shape: &[usize]| {
            let n: usize = shape.iter().product();
            let values = (0..n)
                .map(|i| {
                    if name.contains("norm.weight") {
                        1.
                    } else {
                        (i as f32 * 0.3).cos() * 0.1
                    }
                })
                .collect::<Vec<_>>();
            tensors.insert(
                name.to_string(),
                Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
            );
        };
        for (name, shape) in [
            ("fc.weight", vec![8, 16]),
            ("hidden_norm.weight", vec![8]),
            ("norm.weight", vec![8]),
            ("markov_head.markov_w1.weight", vec![16, 2]),
            ("markov_head.markov_w2.weight", vec![16, 2]),
            ("confidence_head.proj.weight", vec![1, 10]),
            ("confidence_head.proj.bias", vec![1]),
            ("layers.0.input_layernorm.weight", vec![8]),
            ("layers.0.post_attention_layernorm.weight", vec![8]),
            ("layers.0.self_attn.q_norm.weight", vec![4]),
            ("layers.0.self_attn.k_norm.weight", vec![4]),
            ("layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("layers.0.self_attn.k_proj.weight", vec![4, 8]),
            ("layers.0.self_attn.v_proj.weight", vec![4, 8]),
            ("layers.0.self_attn.o_proj.weight", vec![8, 8]),
            ("layers.0.mlp.gate_proj.weight", vec![12, 8]),
            ("layers.0.mlp.up_proj.weight", vec![12, 8]),
            ("layers.0.mlp.down_proj.weight", vec![8, 12]),
        ] {
            insert(name, &shape);
        }
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
        let draft = Draft::new(
            config,
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap(),
            target,
        )
        .unwrap();
        (dir, draft)
    }

    #[test]
    fn verification_accepts_only_contiguous_matching_prefix() {
        assert_eq!(accepted_prefix(&[2, 3, 4], &[9, 3, 4, 5]).unwrap(), 0);
        assert_eq!(accepted_prefix(&[2, 3, 4], &[2, 9, 4, 5]).unwrap(), 1);
        assert_eq!(accepted_prefix(&[2, 3, 4], &[2, 3, 4, 5]).unwrap(), 3);
        assert!(accepted_prefix(&[2], &[2]).is_err());
    }

    #[test]
    fn rejected_suffix_is_removed_from_every_target_layer() {
        let (dir, config) = fixture();
        let mut target = decoder(dir.path(), config.clone(), 2);
        let mut reference = decoder(dir.path(), config, 2);
        target.forward(&[3, 5]).unwrap();
        let captured = target.forward_capture(&[8, 2, 4], &[0, 1], true).unwrap();
        assert_eq!(captured.logits.dims(), [3, 16]);
        assert_eq!(captured.features.unwrap().dims(), [3, 16]);
        target.truncate_cache(3).unwrap();
        let actual = target.forward(&[6]).unwrap();
        let expected = reference.forward(&[3, 5, 8, 6]).unwrap();
        let error = (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "cache contamination: {error}");
        assert!(target.truncate_cache(5).is_err());
    }

    #[test]
    fn paired_generation_matches_target_and_handles_limits_and_callback_failure() {
        let (dir, config) = fixture();
        let mut target = decoder(dir.path(), config.clone(), 2);
        let mut baseline = decoder(dir.path(), config, 2);
        let (_dir, mut draft) = draft_fixture(&target);
        let mut input = vec![3, 5];
        let mut expected = Vec::new();
        for _ in 0..10 {
            let next = baseline
                .forward(&input)
                .unwrap()
                .argmax(0)
                .unwrap()
                .to_scalar::<u32>()
                .unwrap();
            if baseline.config.eos_token_id.contains(&next) {
                break;
            }
            expected.push(next);
            input = vec![next];
        }
        for limit in [1, 2, 4, 10] {
            let mut actual = Vec::new();
            let stats = generate_greedy(&mut target, &mut draft, &[3, 5], limit, |token| {
                actual.push(token);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual, expected[..expected.len().min(limit)]);
            assert_eq!(stats.output_tokens, actual.len());
            assert_eq!(target.position, 0);
            assert_eq!(draft.position, 0);
            assert!(stats.confidence_sum.is_finite());
        }
        assert!(
            generate_greedy(&mut target, &mut draft, &[3, 5], 10, |_| anyhow::bail!(
                "stop output"
            ))
            .is_err()
        );
        assert_eq!(target.position, 0);
        assert_eq!(draft.position, 0);
        assert!(generate_greedy(&mut target, &mut draft, &[3, 5], 15, |_| Ok(())).is_err());
    }

    #[test]
    fn proposal_leaves_draft_prefix_unchanged_and_rejects_mismatched_pair() {
        let (dir, config) = fixture();
        let mut target = decoder(dir.path(), config, 2);
        let (_dir, mut draft) = draft_fixture(&target);
        let result = target.forward_capture(&[3, 5], &[0, 1], false).unwrap();
        draft.commit(&result.features.unwrap()).unwrap();
        let first = draft.propose(&target, 2, 3).unwrap();
        let second = draft.propose(&target, 2, 3).unwrap();
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first.confidence, second.confidence);
        assert_eq!(draft.position, 2);
        draft.config.num_target_layers += 1;
        assert!(draft.config.validate_for(&target.config).is_err());
    }
}
