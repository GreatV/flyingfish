//! Request tensor storage accounting, independent of checkpoint payload loading.
use crate::{Config, dspark};
use anyhow::{Context, Result, ensure};
use candle_core::Device;

mod request;
pub use request::{RequestGeometry, RequestMemoryEstimate, estimate_request_memory};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvMemoryEstimate {
    pub target_bytes: u64,
    pub draft_bytes: u64,
    /// Includes old and replacement storage during concatenation/truncation.
    /// Activations, weights, library workspaces and allocator slack are separate.
    pub peak_bytes: u64,
}

pub(crate) fn cache_bytes(
    layers: usize,
    heads: usize,
    dim: usize,
    tokens: usize,
    element_bytes: u64,
) -> Result<u64> {
    [layers, heads, dim, tokens]
        .into_iter()
        .try_fold(2 * element_bytes, |bytes, dimension| {
            bytes
                .checked_mul(u64::try_from(dimension)?)
                .context("KV storage byte count overflow")
        })
}

/// Bound KV storage for batch-one generation through the requested context.
/// Verification blocks are capped by remaining output tokens, so they never
/// extend beyond prompt length plus the requested generation length.
pub fn estimate_kv_memory(
    target: &Config,
    draft: Option<&dspark::Config>,
    tokens: usize,
    device: &Device,
) -> Result<KvMemoryEstimate> {
    target.validate()?;
    ensure!(
        tokens > 0 && tokens <= target.max_position_embeddings,
        "invalid request KV context length"
    );
    let element_bytes = if device.is_cpu() { 4 } else { 2 };
    let target_bytes = cache_bytes(
        target.num_hidden_layers,
        target.num_key_value_heads,
        target.head_dim,
        tokens,
        element_bytes,
    )?;
    let (draft_bytes, replacement_bytes) = if let Some(draft) = draft {
        draft.validate_for(target)?;
        let bytes = draft.kv_cache_bytes(tokens, element_bytes)?;
        (
            bytes,
            (target_bytes / target.num_hidden_layers as u64)
                .max(draft.kv_layer_bytes(tokens, element_bytes)?),
        )
    } else {
        (0, target_bytes / target.num_hidden_layers as u64)
    };
    let peak_bytes = target_bytes
        .checked_add(draft_bytes)
        .and_then(|bytes| bytes.checked_add(replacement_bytes))
        .context("request KV peak byte count overflow")?;
    Ok(KvMemoryEstimate {
        target_bytes,
        draft_bytes,
        peak_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{decoder, fixture};

    #[test]
    fn kv_estimate_matches_live_target_cache_and_rejects_invalid_context() -> Result<()> {
        let (root, config) = fixture();
        let estimate = estimate_kv_memory(&config, None, 5, &Device::Cpu)?;
        let mut target = decoder(root.path(), config.clone(), 2);
        let _ = target.forward(&[2, 3, 4])?;
        let _ = target.forward(&[5, 6])?;
        let actual = target
            .cache
            .iter()
            .flatten()
            .map(|(k, v)| {
                (k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()) as u64
            })
            .sum::<u64>();
        assert_eq!(estimate.target_bytes, actual);
        assert_eq!(estimate.draft_bytes, 0);
        assert_eq!(
            estimate.peak_bytes,
            actual + actual / config.num_hidden_layers as u64
        );
        assert!(estimate_kv_memory(&config, None, 0, &Device::Cpu).is_err());
        assert!(
            estimate_kv_memory(
                &config,
                None,
                config.max_position_embeddings + 1,
                &Device::Cpu
            )
            .is_err()
        );
        assert!(cache_bytes(usize::MAX, 2, 128, 8192, 2).is_err());
        Ok(())
    }
}
