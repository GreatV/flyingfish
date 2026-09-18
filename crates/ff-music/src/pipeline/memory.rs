use super::{Music3, Options};
use crate::Component;
use anyhow::{Context, Result};
use candle_core::DType;

mod acoustic;
pub use acoustic::AcousticMemoryEstimate;

#[derive(Clone, Debug, serde::Serialize)]
pub struct PhaseMemoryRelease {
    pub stage: String,
    pub required_free_bytes: u64,
    pub free_bytes_before: u64,
    pub free_bytes_after: u64,
    pub released_tensors: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RequestMemoryEstimate {
    pub prompt_tokens: usize,
    pub frames: usize,
    pub language_kv_bytes: u64,
    pub language_kv_peak_bytes: u64,
    pub prompt_embedding_bytes: u64,
    pub autoregressive_activation_bytes: u64,
    pub frame_stack_host_peak_bytes: u64,
    pub weight_load_reserve_bytes: u64,
    pub acoustic: AcousticMemoryEstimate,
    /// Maximum of autoregressive state/activation/load and acoustic stage demand.
    /// Library internals and allocator slack remain separate.
    pub known_device_reserve_bytes: u64,
}

fn product(factors: &[u64]) -> Result<u64> {
    factors.iter().try_fold(1u64, |n, &factor| {
        n.checked_mul(factor)
            .context("Music3 memory product overflow")
    })
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |n, &value| {
        n.checked_add(value).context("Music3 memory sum overflow")
    })
}

#[cfg(feature = "cuda")]
use ff_core::weights::is_cuda_allocation_error as is_cuda_oom;

/// Flat free-memory headroom required at an allocation instant (tensor bytes
/// plus this must fit free memory). Distinct from the pool-scaled admission
/// reserves on purpose: those size a planning budget, this guards one
/// allocation, and the two must not move together.
#[cfg(feature = "cuda")]
const STAGE_GUARD_HEADROOM_BYTES: u64 = 1 << 30;

fn language_state(component: &Component, prompt: usize, frames: usize) -> Result<(u64, u64, u64)> {
    let tokens = prompt
        .checked_add(frames)
        .context("Music3 context overflow")?;
    anyhow::ensure!(
        tokens <= component.n("max_position_embeddings")?,
        "Music3 language model context exceeded"
    );
    let bytes = component.dtype.size_in_bytes() as u64;
    let layer = product(&[
        2,
        2,
        component.n("num_key_value_heads")? as u64,
        component.n("head_dim")? as u64,
        tokens as u64,
        bytes,
    ])?;
    let steady = product(&[layer, component.n("num_hidden_layers")? as u64])?;
    let peak = sum(&[steady, layer])?;
    let embeddings = product(&[2, prompt as u64, component.n("hidden_size")? as u64, bytes])?;
    Ok((steady, peak, embeddings))
}

/// Conservative live-tensor envelope for the streamed decoder block. Weights
/// and durable KV are charged separately. This follows the RMS/MLP and grouped
/// F32 attention allocations in math.rs, including broadcasted K/V copies.
fn decoder_activations(
    component: &Component,
    rows: u64,
    key_rows: u64,
    kv_heads: u64,
    head_dim: u64,
    chunk: usize,
) -> Result<u64> {
    let hidden = component.n("hidden_size")? as u64;
    let heads = component.n("num_attention_heads")? as u64;
    anyhow::ensure!(
        kv_heads > 0 && heads.is_multiple_of(kv_heads),
        "invalid Music3 GQA heads"
    );
    let bytes = component.dtype.size_in_bytes() as u64;
    let plane = product(&[2, rows, hidden, bytes])?;
    let wide = product(&[2, rows, hidden, 4])?;
    let intermediate = product(&[2, rows, component.n("intermediate_size")? as u64, bytes])?;
    let norm = sum(&[product(&[2, plane])?, product(&[4, wide])?])?;
    let query = product(&[2, rows, heads, head_dim, bytes])?;
    let kv = product(&[2, rows, kv_heads, head_dim, bytes])?;
    let mlp = sum(&[
        product(&[5, plane])?,
        product(&[2, query])?,
        kv,
        product(&[3, intermediate])?,
    ])?;
    let group = heads / kv_heads;
    let query_rows = rows.min(chunk as u64);
    let score = product(&[2, group, query_rows, key_rows, 4])?;
    let mask = product(&[query_rows, key_rows, 4])?;
    let wide_kv = product(&[2, key_rows, head_dim, 4])?;
    let wide_query = product(&[2, group, query_rows, head_dim, 4])?;
    let attention = sum(&[
        product(&[2, plane])?,
        product(&[3, query])?,
        product(&[4, kv])?,
        product(&[4, score])?,
        mask,
        product(&[2, wide_kv])?,
        product(&[2, group, wide_kv])?,
        product(&[2, wide_query])?,
        product(&[6, rows, head_dim, 4])?,
    ])?;
    Ok(norm.max(mlp).max(attention))
}

fn weight_load_reserve(component: &Component, batched_linear: bool) -> Result<u64> {
    let mut largest = 0;
    for name in component.weights.tensor_names() {
        let metadata = component.weights.raw_tensor_metadata(name)?;
        let raw = DType::try_from(metadata.dtype)?;
        let produced = if component.device.is_cpu() && matches!(raw, DType::F16 | DType::BF16) {
            DType::F32
        } else {
            raw
        };
        let loaded = component.weights.produced_bytes(name, &component.device)?;
        let execution = product(&[
            loaded / produced.size_in_bytes() as u64,
            component.dtype.size_in_bytes() as u64,
        ])?;
        let conversion = if produced == component.dtype {
            0
        } else {
            execution
        };
        let mut reserve = sum(&[loaded, conversion])?;
        if name.ends_with(".weight_v") {
            let gain_name = format!("{}g", name.strip_suffix('v').unwrap());
            let gain = component
                .weights
                .raw_tensor_metadata(&gain_name)?
                .shape
                .iter()
                .try_fold(
                    component.dtype.size_in_bytes() as u64,
                    |bytes, &dimension| {
                        bytes
                            .checked_mul(dimension as u64)
                            .context("Music3 gain size overflow")
                    },
                )?;
            reserve = reserve.max(sum(&[product(&[3, execution])?, product(&[3, gain])?])?);
        }
        let row_only = name == "model.embed_tokens.weight"
            || name == "audio_embeddings.weight"
            || name == "pos_embedding.weight";
        let unbatched = name == "lm_head.weight"
            || name == "projection.weight"
            || name.starts_with("audio_heads.")
            || name.starts_with("time_embed.")
            || name == "time_proj.weight";
        if row_only {
            continue;
        }
        if batched_linear && metadata.shape.len() == 2 && !unbatched {
            let copies = if component.device.is_cuda() { 2 } else { 3 };
            reserve = reserve.max(product(&[copies, execution])?);
        }
        largest = largest.max(reserve);
    }
    Ok(largest)
}

impl Music3 {
    pub(super) fn vocode_with_memory_recovery(
        &mut self,
        latent: &candle_core::Tensor,
        tensor_bytes: u64,
    ) -> Result<candle_core::Tensor> {
        match crate::acoustic::vocode(&self.vocoder, latent) {
            Ok(output) => Ok(output),
            Err(error) => {
                if self.recover_vocoder_oom(&error, tensor_bytes)? {
                    crate::acoustic::vocode(&self.vocoder, latent)
                        .with_context(|| format!("vocoder retry after releasing retained weights; original error: {error:#}"))
                } else {
                    Err(error)
                }
            }
        }
    }

    fn recover_vocoder_oom(&mut self, error: &anyhow::Error, tensor_bytes: u64) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::{Device, cuda_backend::cudarc::driver::result};
            let device = self.vocoder.device.clone();
            if let Device::Cuda(cuda) = &device {
                if !is_cuda_oom(error.as_ref()) {
                    return Ok(false);
                }
                device
                    .synchronize()
                    .context("synchronize failed vocoder before recovery")?;
                let context = cuda.cuda_stream().context().clone();
                let before = context.mem_get_info()?.0 as u64;
                let released_tensors = self.residency.release();
                device.synchronize()?;
                if context.has_async_alloc() {
                    unsafe {
                        result::mem_pool::trim_to(
                            result::device::get_mem_pool(context.cu_device())?,
                            0,
                        )
                    }?;
                }
                let after = context.mem_get_info()?.0 as u64;
                self.memory_releases.push(PhaseMemoryRelease {
                    stage: "vocoder-oom-retry".to_owned(),
                    required_free_bytes: sum(&[
                        tensor_bytes,
                        // An instantaneous pre-allocation guard, not an
                        // admission reserve: a flat local headroom, so tuning
                        // the shared reserve never moves this gate.
                        STAGE_GUARD_HEADROOM_BYTES,
                    ])?,
                    free_bytes_before: before,
                    free_bytes_after: after,
                    released_tensors,
                });
                return Ok(true);
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (error, tensor_bytes);
        Ok(false)
    }

    pub(super) fn acoustic_memory(
        &self,
        frames: usize,
        chunk: usize,
    ) -> Result<AcousticMemoryEstimate> {
        acoustic::estimate(self, frames, chunk)
    }

    pub(super) fn prepare_acoustic_stage(&mut self, stage: &str, tensor_bytes: u64) -> Result<()> {
        self.activate_weight_stage(Some(stage));
        #[cfg(feature = "cuda")]
        {
            use candle_core::{Device, cuda_backend::cudarc::driver::result};
            let device = self.transformer.device.clone();
            if let Device::Cuda(cuda) = &device {
                if !self.residency.policy().is_enabled() {
                    return Ok(());
                }
                let required = sum(&[tensor_bytes, STAGE_GUARD_HEADROOM_BYTES])?;
                let context = cuda.cuda_stream().context().clone();
                device.synchronize()?;
                let before = context.mem_get_info()?.0 as u64;
                let pool = if context.has_async_alloc() {
                    Some(unsafe { result::device::get_mem_pool(context.cu_device()) }?)
                } else {
                    None
                };
                let reclaim = || -> Result<u64> {
                    device.synchronize()?;
                    if let Some(pool) = pool {
                        unsafe { result::mem_pool::trim_to(pool, 0) }?;
                    }
                    Ok(context.mem_get_info()?.0 as u64)
                };
                let mut free = if before < required {
                    reclaim()?
                } else {
                    before
                };
                let resident = self.residency.stats().resident_bytes;
                let mut released_tensors = self
                    .residency
                    .set_capacity_bytes(resident.saturating_add(free).saturating_sub(required));
                if released_tensors > 0 {
                    free = reclaim()?;
                }
                while free < required {
                    let bytes = self.residency.stats().resident_bytes;
                    if bytes == 0 {
                        break;
                    }
                    let deficit = (required - free).max(64 << 20);
                    let released = self
                        .residency
                        .set_capacity_bytes(bytes.saturating_sub(deficit));
                    if released == 0 {
                        break;
                    }
                    released_tensors += released;
                    free = reclaim()?;
                }
                self.memory_releases.push(PhaseMemoryRelease {
                    stage: stage.to_owned(),
                    required_free_bytes: required,
                    free_bytes_before: before,
                    free_bytes_after: free,
                    released_tensors,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = tensor_bytes;
        Ok(())
    }

    /// Validate/tokenize a request and estimate known state/load requirements
    /// using headers only. No checkpoint payload is loaded by this method.
    pub fn request_memory(
        &self,
        caption: &str,
        lyrics: &str,
        options: &Options,
    ) -> Result<RequestMemoryEstimate> {
        let (_, prompt_tokens, frames) = self.prepare_request(caption, lyrics, options)?;
        let (language_kv_bytes, language_kv_peak_bytes, prompt_embedding_bytes) =
            language_state(&self.language, prompt_tokens, frames)?;
        let key_rows = prompt_tokens
            .checked_add(frames)
            .context("Music3 context overflow")? as u64;
        let kv_heads = self.language.n("num_key_value_heads")? as u64;
        let head_dim = self.language.n("head_dim")? as u64;
        let prefill = decoder_activations(
            &self.language,
            prompt_tokens as u64,
            prompt_tokens as u64,
            kv_heads,
            head_dim,
            options.attention_query_chunk,
        )?;
        let decode = decoder_activations(
            &self.language,
            1,
            key_rows,
            kv_heads,
            head_dim,
            options.attention_query_chunk,
        )?;
        let depth_heads = self.depth.n("num_attention_heads")? as u64;
        let depth_hidden = self.depth.n("hidden_size")? as u64;
        anyhow::ensure!(
            depth_hidden.is_multiple_of(depth_heads),
            "invalid Music3 depth heads"
        );
        let codebooks = self.depth.n("num_codebooks")? as u64;
        let depth = decoder_activations(
            &self.depth,
            codebooks,
            codebooks,
            depth_heads,
            depth_hidden / depth_heads,
            options.attention_query_chunk,
        )?;
        let depth = sum(&[
            depth,
            product(&[
                8,
                codebooks,
                depth_hidden,
                self.depth.dtype.size_in_bytes() as u64,
            ])?,
        ])?;
        let logits = product(&[
            2,
            self.language
                .n("vocab_size")?
                .max(self.depth.n("audio_vocab_size")?) as u64,
            self.language.dtype.size_in_bytes() as u64 + 4,
        ])?;
        let autoregressive_activation_bytes = prefill.max(decode).max(depth).max(logits);
        let weight_load_reserve_bytes = [
            (&self.language, true),
            (&self.depth, true),
            (&self.condition, false),
            (&self.transformer, true),
            (&self.vocoder, false),
        ]
        .into_iter()
        .try_fold(0, |largest, (component, batched)| -> Result<_> {
            Ok(largest.max(weight_load_reserve(component, batched)?))
        })?;
        let frame_stack_host_peak_bytes = product(&[
            2,
            frames as u64,
            self.depth.n("num_codebooks")? as u64,
            self.language.n("hidden_size")? as u64,
            4,
        ])?;
        let acoustic = acoustic::estimate(self, frames, options.attention_query_chunk)?;
        let known_device_reserve_bytes = sum(&[
            language_kv_peak_bytes,
            prompt_embedding_bytes,
            autoregressive_activation_bytes,
            weight_load_reserve_bytes,
        ])?
        .max(acoustic.device_peak_bytes);
        Ok(RequestMemoryEstimate {
            prompt_tokens,
            frames,
            language_kv_bytes,
            language_kv_peak_bytes,
            prompt_embedding_bytes,
            autoregressive_activation_bytes,
            frame_stack_host_peak_bytes,
            weight_load_reserve_bytes,
            acoustic,
            known_device_reserve_bytes,
        })
    }
}

#[cfg(test)]
mod tests {

    #[test]
    #[cfg(feature = "cuda")]
    fn recovery_classifies_typed_oom_through_candle_contexts() {
        use super::is_cuda_oom;
        use candle_core::cuda_backend::{
            WrapErr,
            cudarc::driver::{DriverError, sys},
        };
        let error = Err::<(), _>(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY))
            .w()
            .unwrap_err();
        let error = candle_core::Error::Context {
            inner: Box::new(error),
            context: Box::new("vocoder convolution"),
        };
        assert!(is_cuda_oom(&error));
        assert!(!is_cuda_oom(&DriverError(
            sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
        )));
        assert!(!is_cuda_oom(&candle_core::Error::Msg(
            "CUDA_ERROR_OUT_OF_MEMORY".to_owned()
        )));
    }
}
