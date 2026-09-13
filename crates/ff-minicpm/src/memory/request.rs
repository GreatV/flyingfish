use super::{KvMemoryEstimate, estimate_kv_memory};
use crate::{Config, dspark};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use ff_core::weights::ModelWeights;

#[derive(Clone, Copy, Debug)]
pub struct RequestGeometry {
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub attention_query_chunk_size: usize,
    pub batch_invariant_decode: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestMemoryEstimate {
    pub kv: KvMemoryEstimate,
    pub target_prefill_activation_bytes: u64,
    pub target_decode_activation_bytes: u64,
    pub draft_activation_bytes: u64,
    pub activation_peak_bytes: u64,
    pub streamed_weight_bytes: u64,
    /// Known tensor demand; library workspaces and allocator slack are separate.
    /// Conservative: sums maxima that need not occur at the same instant.
    pub tensor_peak_bytes: u64,
}

fn product(factors: &[u64]) -> Result<u64> {
    factors.iter().try_fold(1u64, |n, &factor| {
        n.checked_mul(factor)
            .context("request memory product overflow")
    })
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |n, &value| {
        n.checked_add(value).context("request memory sum overflow")
    })
}

#[derive(Clone, Copy)]
struct Geometry {
    hidden: u64,
    intermediate: u64,
    heads: u64,
    kv_heads: u64,
    dim: u64,
    vocab: u64,
}

impl Geometry {
    /// Scratch inside attention_with_mask, beyond input Q/K/V and accumulated
    /// outputs. Candle 0.11 broadcast_matmul concretizes the head broadcast.
    fn attention(self, rows: u64, total: u64, chunk: u64) -> Result<u64> {
        let groups = self.heads / self.kv_heads;
        let tile = rows.min(chunk);
        sum(&[
            product(&[groups + 3, total, self.dim, 4])?,
            product(&[2 * groups + 1, tile, total, 4])?,
            product(&[3, groups, tile, self.dim, 4])?,
        ])
    }

    fn forward(
        self,
        rows: u64,
        total: u64,
        chunk: u64,
        bytes: u64,
        capture: u64,
        logits_rows: u64,
    ) -> Result<u64> {
        let hidden = product(&[rows, self.hidden, 8 * bytes + 12])?;
        let ffn = product(&[3, rows, self.intermediate, bytes])?;
        let attention = self.attention(rows, total, chunk)?;
        let features = product(&[2, capture, rows, self.hidden, bytes])?;
        let logits = product(&[logits_rows, self.vocab, bytes + 4])?;
        let rotary = product(&[rows, self.dim, bytes + 2])?;
        sum(&[hidden, features, rotary, ffn.max(attention).max(logits)])
    }
}

fn loaded_weight_bytes(
    weights: &ModelWeights,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<u64> {
    let metadata = weights.raw_tensor_metadata(name)?;
    let raw = DType::try_from(metadata.dtype)?;
    let produced = if device.is_cpu() && matches!(raw, DType::BF16 | DType::F16) {
        DType::F32
    } else {
        raw
    };
    let loaded = weights.produced_bytes(name, device)?;
    if produced == dtype {
        return Ok(loaded);
    }
    let elements = loaded / produced.size_in_bytes() as u64;
    sum(&[loaded, product(&[elements, dtype.size_in_bytes() as u64])?])
}

fn largest_load(weights: &ModelWeights, device: &Device, dtype: DType) -> Result<u64> {
    weights.tensor_names().try_fold(0, |largest, name| {
        let dtype = if name.starts_with("confidence_head.") {
            DType::F32
        } else {
            dtype
        };
        Ok(largest.max(loaded_weight_bytes(weights, name, device, dtype)?))
    })
}

/// Bound known batch-one tensor storage for the current unfused implementation.
/// This estimates logical live tensors, not CUDA pool reservations. The caller
/// must add library workspace/allocator headroom and retain runtime fallback.
pub fn estimate_request_memory(
    target: &Config,
    weights: &ModelWeights,
    draft: Option<(&dspark::Config, &ModelWeights)>,
    device: &Device,
    request: RequestGeometry,
) -> Result<RequestMemoryEstimate> {
    ensure!(
        request.prompt_tokens > 0
            && request.max_new_tokens > 0
            && request.attention_query_chunk_size > 0,
        "request dimensions must be positive"
    );
    let total = request
        .prompt_tokens
        .checked_add(request.max_new_tokens)
        .context("request context overflow")?;
    let kv = estimate_kv_memory(target, draft.map(|(config, _)| config), total, device)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let bytes = dtype.size_in_bytes() as u64;
    let geometry = Geometry {
        hidden: target.hidden_size as u64,
        intermediate: target.intermediate_size as u64,
        heads: target.num_attention_heads as u64,
        kv_heads: target.num_key_value_heads as u64,
        dim: target.head_dim as u64,
        vocab: target.vocab_size as u64,
    };
    let prompt = request.prompt_tokens as u64;
    let total = total as u64;
    let chunk = request.attention_query_chunk_size as u64;
    let vocab = target.vocab_size as u64;
    let capture = draft.map_or(0, |(config, _)| config.target_layer_ids.len() as u64);
    let verify_rows = draft.map_or(1, |(config, _)| {
        config
            .block_size
            .saturating_add(1)
            .min(request.max_new_tokens)
    }) as u64;
    let decode_rows = if request.batch_invariant_decode {
        verify_rows.max(8)
    } else {
        verify_rows
    };
    let target_prefill_activation_bytes =
        geometry.forward(prompt, prompt, chunk, bytes, capture, 1)?;
    let mut target_decode_activation_bytes =
        geometry.forward(decode_rows, total, chunk, bytes, capture, decode_rows)?;
    if request.batch_invariant_decode && verify_rows > 1 {
        target_decode_activation_bytes = sum(&[
            target_decode_activation_bytes,
            product(&[2, geometry.kv_heads, total, geometry.dim, bytes])?,
        ])?;
    }
    let mut draft_activation_bytes = 0;
    let mut streamed_weight_bytes = largest_load(weights, device, dtype)?;
    if let Some((config, weights)) = draft {
        let geometry = Geometry {
            intermediate: config.intermediate_size as u64,
            heads: config.num_attention_heads as u64,
            kv_heads: config.num_key_value_heads as u64,
            ..geometry
        };
        let rows = (config.block_size.min(request.max_new_tokens) as u64).max(1);
        let proposal = geometry.forward(rows, total, chunk, bytes, 0, rows)?;
        let commit_rows = prompt.max(verify_rows);
        let commit = sum(&[
            product(&[capture, commit_rows, geometry.hidden, bytes])?,
            product(&[commit_rows, geometry.hidden, 4 * bytes + 12])?,
            product(&[commit_rows, geometry.kv_heads, geometry.dim, 8 * bytes + 12])?,
            product(&[commit_rows, geometry.dim, bytes + 2])?,
            product(&[verify_rows, vocab, 4])?,
        ])?;
        draft_activation_bytes = commit.max(sum(&[proposal, product(&[3, vocab, 4])?])?);
        let markov = loaded_weight_bytes(weights, "markov_head.markov_w2.weight", device, dtype)?;
        streamed_weight_bytes =
            streamed_weight_bytes.max(sum(&[largest_load(weights, device, dtype)?, markov])?);
    }
    let activation_peak_bytes = target_prefill_activation_bytes
        .max(target_decode_activation_bytes)
        .max(draft_activation_bytes);
    let tensor_peak_bytes = sum(&[kv.peak_bytes, activation_peak_bytes, streamed_weight_bytes])?;
    Ok(RequestMemoryEstimate {
        kv,
        target_prefill_activation_bytes,
        target_decode_activation_bytes,
        draft_activation_bytes,
        activation_peak_bytes,
        streamed_weight_bytes,
        tensor_peak_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::weights::{CachePolicy, WeightSource};

    #[test]
    fn request_memory_uses_metadata_and_scales_with_geometry() -> Result<()> {
        let (root, mut config) = crate::tests::fixture();
        let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1))?;
        config.max_position_embeddings = 8192;
        let request = RequestGeometry {
            prompt_tokens: 16,
            max_new_tokens: 8,
            attention_query_chunk_size: 8,
            batch_invariant_decode: false,
        };
        let short = estimate_request_memory(&config, &weights, None, &Device::Cpu, request)?;
        let long = estimate_request_memory(
            &config,
            &weights,
            None,
            &Device::Cpu,
            RequestGeometry {
                prompt_tokens: 4096,
                ..request
            },
        )?;
        assert!(long.target_prefill_activation_bytes > short.target_prefill_activation_bytes);
        assert!(long.tensor_peak_bytes > short.tensor_peak_bytes);
        assert!(weights.tensor_reads().is_empty());
        assert_eq!(weights.cache_stats().resident_bytes, 0);
        assert_eq!(short.streamed_weight_bytes, 16 * 8 * 4);
        assert!(
            estimate_request_memory(
                &config,
                &weights,
                None,
                &Device::Cpu,
                RequestGeometry {
                    attention_query_chunk_size: 0,
                    ..request
                }
            )
            .is_err()
        );
        assert!(product(&[u64::MAX, 2]).is_err());
        assert!(sum(&[u64::MAX, 1]).is_err());
        Ok(())
    }
}
