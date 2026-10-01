//! Experimental layer-partitioned execution of one GLM request.
//! Layers and their recurrent/KV state have fixed owners. This is capacity
//! parallelism: layer dependencies remain sequential. Rank-scoped admission,
//! explicit policy recording and request-state reset preserve that boundary.
use super::*;
use crate::{
    admission::GlmAdmissionBreakdown,
    partition::{GlmPartitionPolicy, GlmPartitionTransport},
};
use candle_core::IndexOp;
use ff_core::interconnect::{copy_tensor_into, stage_tensor_into};
mod runtime;
pub use runtime::{
    GlmPartitionAdmission, GlmPartitionGeneration, GlmRankAdmission, GlmRankCacheStats,
    LayerPartitionOptions,
};

pub struct LayerPartitionedGlm {
    caches: Vec<LayerCache>,
    buffers: Vec<[BoundaryBuffer; 2]>,
    workers: Vec<StreamedGlm>,
    owners: Vec<usize>,
    tokens: usize,
    transfers: u64,
    transferred_bytes: u64,
    policy: GlmPartitionPolicy,
    breakdowns: Vec<GlmAdmissionBreakdown>,
    initialized: bool,
    failed: bool,
}

impl GlmPartitionTransport {
    fn copy_into(self, input: &Tensor, destination: &Tensor, host: &mut [u8]) -> Result<()> {
        match self {
            Self::SynchronizedCudaDeviceCopyV1 => copy_tensor_into(input, destination)?,
            Self::HostStagedCopyV1 => stage_tensor_into(input, destination, host)?,
        }
        input
            .device()
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .bind_to_thread()?;
        Ok(())
    }
}

/// Destination and host staging for one boundary and transfer class, allocated once at setup.
struct BoundaryBuffer {
    destination: Tensor,
    host: Vec<u8>,
}

impl BoundaryBuffer {
    fn device_bytes(&self) -> usize {
        self.destination.elem_count() * self.destination.dtype().size_in_bytes()
    }
}

impl Drop for LayerPartitionedGlm {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.device.synchronize();
        }
        let _ = self.clear_caches();
    }
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
    pub fn transfer_stats(&self) -> (u64, u64) {
        (self.transfers, self.transferred_bytes)
    }
    pub fn worker_materializations(&self) -> Vec<u64> {
        self.workers
            .iter()
            .map(|w| w.access_stats().device_tensor_materializations)
            .collect()
    }

    /// The returned view aliases the boundary buffer and is valid until the next transfer of the same class across the same boundary.
    fn transfer(&mut self, input: &Tensor, owner: usize, class: usize) -> Result<Tensor> {
        if input.device().same_device(&self.workers[owner].device) {
            return Ok(input.clone());
        }
        let boundary = owner
            .checked_sub(1)
            .context("unexpected transfer to first rank")?;
        ensure!(
            input.device().same_device(&self.workers[boundary].device),
            "transfer did not follow adjacent GLM ranks"
        );
        let bytes = input.elem_count() * input.dtype().size_in_bytes();
        let buffer = &mut self.buffers[boundary][class];
        ensure!(
            bytes <= buffer.device_bytes(),
            "GLM boundary {boundary} class {class} needs {bytes} bytes, the setup buffer holds {}",
            buffer.device_bytes()
        );
        let destination = buffer
            .destination
            .narrow(0, 0, input.elem_count())?
            .reshape(input.shape())?;
        self.policy.transports[boundary][class].0.copy_into(
            input,
            &destination,
            &mut buffer.host,
        )?;
        self.transfers += 1;
        self.transferred_bytes += bytes as u64;
        Ok(destination)
    }

    fn clear_caches(&mut self) -> Result<()> {
        for (layer, cache) in self.caches.drain(..).enumerate() {
            self.workers[self.owners[layer]]
                .device
                .as_cuda_device()?
                .cuda_stream()
                .context()
                .bind_to_thread()?;
            drop(cache);
        }
        Ok(())
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

    fn embed_streams(&self, ids: &[u32]) -> Result<Tensor> {
        let first = &self.workers[0];
        let embedded = first
            .weights
            .load_rows(EMBEDDING_WEIGHT, ids, &first.device)?
            .to_dtype(first.compute_dtype)?;
        Ok(embedded
            .unsqueeze(1)?
            .repeat((1, first.config.text_config.hc_mult, 1))?)
    }

    fn prefill_inner(&mut self, ids: &[u32]) -> Result<Tensor> {
        let mut streams = self.embed_streams(ids)?;
        for layer in 0..self.owners.len() {
            let owner = self.owners[layer];
            streams = self.transfer(&streams, owner, 1)?;
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
        let mut streams = self.embed_streams(&[token])?.squeeze(0)?;
        for layer in 0..self.owners.len() {
            let owner = self.owners[layer];
            streams = self.transfer(&streams, owner, 0)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{quantize_tiny_linears, tiny_checkpoint_with_kda_width};

    #[test]
    #[ignore = "requires two CUDA devices"]
    fn derived_partition_matches_single_device() -> Result<()> {
        let checkpoint = tiny_checkpoint_with_kda_width(128);
        quantize_tiny_linears(checkpoint.path());
        let first = Device::new_cuda(0)?;
        let second = Device::new_cuda(1)?;
        let measure = |partition: &LayerPartitionedGlm| -> Result<()> {
            let worker = &partition.workers[0];
            let bytes = worker.config.text_config.hidden_size
                * worker.config.text_config.hc_mult
                * worker.compute_dtype.size_in_bytes();
            let devices = [&partition.workers[0].device, &partition.workers[1].device];
            let streams = [
                devices[0].as_cuda_device()?.cuda_stream(),
                devices[1].as_cuda_device()?.cuda_stream(),
            ];
            eprintln!(
                "allocation modes: {:?}",
                streams.each_ref().map(|s| s.context().has_async_alloc())
            );
            for bytes in [bytes, bytes * partition.context_bound()] {
                let mut pattern = vec![0u8; bytes];
                rand::Rng::fill(&mut rand::rng(), pattern.as_mut_slice());
                pattern[0] = 1;
                let input = Tensor::from_vec(pattern.clone(), bytes, devices[0])?;
                let reused = Tensor::zeros(bytes, DType::U8, devices[1])?;
                let mut host = vec![0u8; bytes];
                let mut samples = Vec::new();
                for sample in 0..25 {
                    let start = Instant::now();
                    let output = unsafe { streams[1].alloc::<u8>(bytes) }?;
                    streams[1].synchronize()?;
                    let allocation = start.elapsed().as_nanos();
                    drop(output);
                    streams[1].synchronize()?;
                    let mut row = vec![allocation];
                    for transport in [
                        GlmPartitionTransport::SynchronizedCudaDeviceCopyV1,
                        GlmPartitionTransport::HostStagedCopyV1,
                    ] {
                        let start = Instant::now();
                        transport.copy_into(&input, &reused, &mut host)?;
                        devices[1].synchronize()?;
                        row.push(start.elapsed().as_nanos());
                    }
                    if sample >= 5 {
                        samples.push(row);
                    }
                }
                eprintln!("GLM allocation/reused-peer/reused-host {bytes} B ns: {samples:?}");
            }
            Ok(())
        };
        let reference = StreamedGlm::open(
            checkpoint.path(),
            StreamedGlmOptions {
                io_readers: Some([0, 1]),
                ..StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), first.clone())
                    .with_resident_static(true)
            },
        )?;
        for pinned in [false, true] {
            let mut partition = LayerPartitionedGlm::prepare(
                checkpoint.path(),
                vec![first.clone(), second.clone()],
                LayerPartitionOptions {
                    transport: None,
                    io_readers: Some([0, 1]),
                    resident_static: true,
                    pinned_fp8_transfer: pinned,
                    max_context_tokens: Some(8),
                    ..Default::default()
                },
            )?;
            eprintln!(
                "partition policy: {}",
                String::from_utf8(partition.policy().canonical_json()?)?
            );
            if !pinned {
                measure(&partition)?;
            }
            assert!(partition.limit_context(0).is_err());
            assert!(partition.limit_context(9).is_err());
            partition.limit_context(4)?;
            assert_eq!(partition.context_bound(), 4);
            for rank in partition.admission(1)?.ranks {
                assert_eq!(
                    rank.breakdown.maximum_dsa_cache_bytes,
                    rank.breakdown.dsa_cache_bytes_per_token * 4
                );
            }
            let options = GlmGenerationOptions {
                max_new_tokens: 3,
                max_context_tokens: 8,
                reasoning_effort: "low".into(),
                temperature: 0.0,
                top_p: 1.0,
                seed: 7,
                progress: false,
            };
            for prompt in ["hello", "other", "hello"] {
                let expected = reference.generate(prompt, &options)?;
                let actual = partition.generate(prompt, &options)?;
                assert!(partition.limit_context(3).is_err());
                let mut oversized = options.clone();
                oversized.max_new_tokens = 4;
                assert!(partition.generate(prompt, &oversized).is_err());
                assert_eq!(actual.generated_token_ids, expected.generated_token_ids);
                assert_eq!(
                    actual.fp8_transfers.iter().any(|stats| stats.uploads > 0),
                    pinned
                );
                partition.reset()?;
                let expected =
                    reference.capture_first_next_token_parity(prompt, "low", 8, false)?;
                let logits = partition
                    .prefill_ids(&expected.prompt_token_ids)?
                    .to_vec1::<f32>()?;
                let expected = expected.next_token_logits.to_vec1::<f32>()?;
                let bits = logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits,
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                eprintln!(
                    "partition parity: pinned={pinned}, prompt={prompt:?}, ids={:?}, first_token_logit_bits={bits:?}",
                    actual.generated_token_ids
                );
            }
        }
        if let Some(model) =
            ff_core::paths::checkpoint_dir("zai-org/GLM-5.3-Flash").filter(|path| path.is_dir())
        {
            eprintln!("transport checkpoint: {}", model.display());
            let partition = LayerPartitionedGlm::prepare(
                model,
                vec![first, second],
                LayerPartitionOptions {
                    io_readers: Some([0, 1]),
                    ..Default::default()
                },
            )?;
            assert_eq!(partition.worker_materializations(), vec![0, 0]);
            measure(&partition)?;
            eprintln!(
                "full-size partition policy: {}",
                String::from_utf8(partition.policy().canonical_json()?)?
            );
        }
        Ok(())
    }
}
