//! Experimental layer-partitioned execution of one GLM request.
//! Layers and their recurrent/KV state have fixed owners. This is capacity
//! parallelism: layer dependencies remain sequential. Rank-scoped admission,
//! explicit policy recording and request-state reset preserve that boundary.
use super::*;
use crate::{admission::GlmAdmissionBreakdown, partition::GlmPartitionPolicy};
use candle_core::IndexOp;
use ff_core::interconnect::copy_tensor;
mod runtime;
pub use runtime::{
    GlmPartitionAdmission, GlmPartitionGeneration, GlmRankAdmission, GlmRankCacheStats,
    LayerPartitionOptions,
};

pub struct LayerPartitionedGlm {
    workers: Vec<StreamedGlm>,
    owners: Vec<usize>,
    caches: Vec<LayerCache>,
    tokens: usize,
    transfers: u64,
    transferred_bytes: u64,
    policy: GlmPartitionPolicy,
    breakdowns: Vec<GlmAdmissionBreakdown>,
    initialized: bool,
    failed: bool,
}

impl LayerPartitionedGlm {
    pub fn open(
        model: impl AsRef<Path>,
        devices: Vec<Device>,
        expert_cache_bytes_per_device: usize,
    ) -> Result<Self> {
        Self::prepare(
            model,
            devices,
            LayerPartitionOptions {
                expert_cache_bytes_per_device,
                ..Default::default()
            },
        )
    }

    pub fn layer_owners(&self) -> &[usize] {
        &self.owners
    }
    pub fn transfer_stats(&self) -> (u64, u64) {
        (self.transfers, self.transferred_bytes)
    }
    pub fn worker_materializations(&self) -> Vec<u64> {
        self.workers
            .iter()
            .map(|w| w.access_stats().device_tensor_materializations)
            .collect()
    }

    fn transfer(&mut self, input: &Tensor, owner: usize) -> Result<Tensor> {
        if !input.device().same_device(&self.workers[owner].device) {
            let copied = copy_tensor(input, &self.workers[owner].device)?;
            self.transfers += 1;
            self.transferred_bytes += (input.elem_count() * input.dtype().size_in_bytes()) as u64;
            Ok(copied)
        } else {
            Ok(input.clone())
        }
    }

    fn finish_logits(&self, streams: Tensor) -> Result<Tensor> {
        let last = self.workers.last().context("missing final worker")?;
        let hidden = streams
            .to_dtype(DType::F32)?
            .mean(0)?
            .to_dtype(last.compute_dtype)?;
        let norm = last.load_tensor(FINAL_NORM_WEIGHT)?;
        let hidden = math::rms_norm(&hidden, Some(&norm), last.config.text_config.rms_norm_eps)?;
        let head = last.load_linear_weight(LM_HEAD_WEIGHT)?;
        Ok(linear(&hidden.unsqueeze(0)?, &head)?
            .squeeze(0)?
            .to_dtype(DType::F32)?)
    }

    /// Layer-batched prefill, followed by logits for the next token.
    pub fn prefill_ids(&mut self, ids: &[u32]) -> Result<Tensor> {
        ensure!(
            !self.failed,
            "GLM partition request failed; reset before reuse"
        );
        ensure!(self.tokens == 0, "prefill requires a fresh request state");
        ensure!(
            !ids.is_empty() && ids.len() <= self.context_bound(),
            "prefill exceeds the exact short-context profile"
        );
        ensure!(
            ids.iter()
                .all(|&n| (n as usize) < self.workers[0].config.text_config.vocab_size),
            "token outside GLM vocabulary"
        );
        self.admission(ids.len())?;
        self.failed = true;
        self.initialize()?;
        let result = self.prefill_inner(ids);
        if result.is_ok() {
            self.failed = false;
        }
        result
    }

    fn prefill_inner(&mut self, ids: &[u32]) -> Result<Tensor> {
        let first = &self.workers[0];
        let embedded = first
            .weights
            .load_rows(EMBEDDING_WEIGHT, ids, &first.device)?
            .to_dtype(first.compute_dtype)?;
        let mut streams =
            embedded
                .unsqueeze(1)?
                .repeat((1, first.config.text_config.hc_mult, 1))?;
        for layer in 0..self.owners.len() {
            let owner = self.owners[layer];
            streams = self.transfer(&streams, owner)?;
            streams = self.workers[owner]
                .forward_layer_prefill(layer, &streams, &mut self.caches[layer])?
                .streams;
        }
        self.tokens = ids.len();
        self.finish_logits(streams.i(ids.len() - 1)?)
    }

    /// Append one token to the distributed request state and return next logits.
    pub fn decode_id(&mut self, token: u32) -> Result<Tensor> {
        ensure!(
            !self.failed,
            "GLM partition request failed; reset before reuse"
        );
        ensure!(
            self.tokens > 0 && self.tokens < self.context_bound(),
            "decode requires prefill and remaining context capacity"
        );
        ensure!(
            (token as usize) < self.workers[0].config.text_config.vocab_size,
            "token outside GLM vocabulary"
        );
        self.failed = true;
        let result = self.decode_inner(token);
        if result.is_ok() {
            self.failed = false;
        }
        result
    }

    fn decode_inner(&mut self, token: u32) -> Result<Tensor> {
        let first = &self.workers[0];
        let embedded = first
            .weights
            .load_rows(EMBEDDING_WEIGHT, &[token], &first.device)?
            .to_dtype(first.compute_dtype)?;
        let mut streams = embedded.repeat((first.config.text_config.hc_mult, 1))?;
        for layer in 0..self.owners.len() {
            let owner = self.owners[layer];
            streams = self.transfer(&streams, owner)?;
            streams = self.workers[owner].forward_layer(
                layer,
                self.tokens,
                RoutingTracePhase::Decode,
                &streams,
                &mut self.caches[layer],
                &mut None,
            )?;
        }
        self.tokens += 1;
        self.finish_logits(streams)
    }
}
