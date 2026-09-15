use super::{
    config::{AttentionKind, GlmConfig, GlmGenerationConfig, MlpKind},
    execution_manifest::{
        GlmCacheAdmissionDecision, GlmCacheReadmissionEvent, GlmExecutionManifest,
        GlmExecutionManifestRecorder,
    },
    execution_policy::GlmExecutionPolicy,
    expert_cache::{ExpertCache, ExpertCacheReplacementPolicy, ExpertCacheStats},
    expert_cache_manager::{ExpertCacheLayout, ExpertCacheManager},
    fp8, math,
    routing_trace::{
        ExpertCacheEntryDtype, RoutingTrace, RoutingTraceBuilder, RoutingTraceLayer,
        RoutingTracePhase,
    },
};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops;
use ff_core::probe::ResourceSnapshot;
use ff_core::weights::{CachePolicy, CacheStats, ModelWeights, WeightAccessStats, WeightSource};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ops::Not;
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    path::Path,
    sync::atomic::AtomicUsize,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

#[cfg(feature = "cuda")]
mod multi_gpu;
#[cfg(feature = "cuda")]
pub use multi_gpu::{
    GlmPartitionAdmission, GlmPartitionGeneration, GlmRankAdmission, GlmRankCacheStats,
    LayerPartitionOptions, LayerPartitionedGlm,
};

const EMBEDDING_WEIGHT: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM_WEIGHT: &str = "model.language_model.norm.weight";
pub(crate) const LM_HEAD_WEIGHT: &str = "lm_head.weight";

pub(crate) mod prefill;

#[derive(Clone, Copy, Debug)]
struct GlmAdmissionModel {
    maximum_dsa_cache_bytes: usize,
    dsa_cache_bytes_per_token: usize,
    streamed_transient_bytes: usize,
    pending_lm_head_bytes: usize,
    live_expert_bytes: usize,
    weight_load_staging_bytes: usize,
    safety_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CacheReadmissionPlan {
    available_memory_bytes: u64,
    reclaimable_capacity_bytes: u64,
    required_future_headroom_bytes: u64,
    admissible_cache_bound_bytes: u64,
    previous_bound_bytes: u64,
    new_bound_bytes: u64,
    realized_before_bytes: u64,
    decision: GlmCacheAdmissionDecision,
}

#[derive(Clone, Debug)]
pub struct GlmGenerationOptions {
    pub max_new_tokens: usize,
    pub max_context_tokens: usize,
    pub reasoning_effort: String,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
    pub progress: bool,
}

#[derive(Clone, Debug)]
pub struct GlmGeneration {
    pub prompt_tokens: usize,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub prefill_elapsed: Duration,
    pub decode_elapsed: Duration,
    pub token_elapsed: Vec<Duration>,
    pub execution_manifest: GlmExecutionManifest,
}

pub struct GlmParityCapture {
    pub prompt_token_ids: Vec<u32>,
    pub final_hidden_state: Tensor,
    pub next_token_logits: Tensor,
    pub next_token_id: u32,
}

#[derive(Clone, Debug)]
pub struct StreamedGlmOptions {
    pub weight_source: WeightSource,
    pub cache_policy: CachePolicy,
    pub device: Device,
    pub resident_static: bool,
    pub expert_cache_bytes: usize,
    pub expert_cache_layout: ExpertCacheLayout,
    pub expert_cache_replacement: ExpertCacheReplacementPolicy,
    pub expert_cache_min_bytes: usize,
    pub adaptive_expert_cache: bool,
    pub cpu_fp8_dequantization: bool,
    /// Host share, measured by `flyingfish::host_profile`.
    /// Set through `with_host_expert_share` to match the policy's per-mille precision.
    host_expert_share: f64,
    pub pinned_fp8_transfer: bool,
    /// Emit static-residency preload progress on stderr. Off by default so the
    /// library stays silent unless a caller asks for progress.
    pub progress: bool,
}

/// Evaluate an expert from quantized weights without materializing its matrices.
/// Worker threads return CPU tensors; the caller transfers them to the device.
fn host_expert_contribution(
    weights: &ModelWeights,
    swiglu_limit: f64,
    expert_prefix: &str,
    input: &Tensor,
    mixture: f64,
) -> Result<Tensor> {
    let gate = host_projection(weights, expert_prefix, "gate_proj", input)?;
    let up = host_projection(weights, expert_prefix, "up_proj", input)?;
    let activated =
        math::clamped_swiglu(&gate.unsqueeze(0)?, &up.unsqueeze(0)?, swiglu_limit)?.squeeze(0)?;
    host_projection(
        weights,
        expert_prefix,
        "down_proj",
        &activated.contiguous()?,
    )?
    .affine(mixture, 0.0)
    .map_err(Into::into)
}

/// One projection of a host-evaluated expert: quantized values and their block
/// scales, read straight from the checkpoint and multiplied through.
fn host_projection(
    weights: &ModelWeights,
    expert_prefix: &str,
    projection: &str,
    input: &Tensor,
) -> Result<Tensor> {
    let name = format!("{expert_prefix}.{projection}.weight");
    let metadata = weights.metadata(&name)?;
    ensure!(
        metadata.dtype == "F8_E4M3",
        "host expert evaluation needs a block-FP8 projection, but {name} is {}",
        metadata.dtype
    );
    let scale_name = format!("{name}_scale_inv");
    fp8::fused_block_fp8_matvec(
        &weights.load(&name, &Device::Cpu)?,
        &weights.load(&scale_name, &Device::Cpu)?,
        input,
    )
}

/// Quantize the host share to the policy's per-mille precision so replay uses
/// the same split as execution.
fn host_expert_share_per_mille(share: f64) -> u32 {
    if !share.is_finite() || share <= 0.0 {
        return 0;
    }
    (share.min(1.0) * 1000.0).round() as u32
}

/// Page-warming readers (`FF_GLM_PREFETCH_THREADS`, default 4).
fn expert_prefetch_threads() -> usize {
    std::env::var("FF_GLM_PREFETCH_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4)
}

/// Host expert workers (`FF_GLM_HOST_THREADS`, default 4).
fn host_evaluation_threads() -> usize {
    std::env::var("FF_GLM_HOST_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4)
}

/// Read checkpoint ranges into the page cache. Failures leave demand reads unchanged.
#[cfg(unix)]
fn warm_checkpoint_pages(ranges: &[(std::path::PathBuf, u64, u64)], next: &AtomicUsize) {
    use std::os::unix::fs::FileExt;
    let mut files: std::collections::HashMap<&Path, std::fs::File> =
        std::collections::HashMap::new();
    let mut buffer = vec![0u8; 2 << 20];
    loop {
        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Some((path, offset, len)) = ranges.get(index) else {
            return;
        };
        let file = match files.entry(path.as_path()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let Ok(opened) = std::fs::File::open(path) else {
                    continue;
                };
                entry.insert(opened)
            }
        };
        let mut position = *offset;
        let end = offset.saturating_add(*len);
        while position < end {
            let count = buffer.len().min((end - position) as usize);
            match file.read_at(&mut buffer[..count], position) {
                Ok(0) | Err(_) => break,
                Ok(read) => position += read as u64,
            }
        }
    }
}

#[cfg(not(unix))]
fn warm_checkpoint_pages(_ranges: &[(std::path::PathBuf, u64, u64)], _next: &AtomicUsize) {}

/// Fill-ahead worker count (`FF_GLM_FILL_AHEAD`, default 0/off, maximum 8).
#[cfg(feature = "cuda")]
fn fill_ahead_count() -> usize {
    static WORKERS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WORKERS.get_or_init(|| {
        std::env::var("FF_GLM_FILL_AHEAD")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
            .clamp(0, 8)
    })
}

/// Shared by the model and by `StreamedGlmOptions`, so the rule can be checked
/// without standing a model up.
///
/// `share` is already a share: every path reaches this through
/// `host_expert_share_per_mille`, which is where a non-finite, negative or
/// oversized value stops being one. A miss set of zero needs no special case
/// either, since it leaves nothing to divide.
fn host_expert_budget(share: f64, misses: usize) -> usize {
    let wanted = (misses as f64 * share).round() as usize;
    wanted.min(misses.saturating_sub(1))
}

impl StreamedGlmOptions {
    pub fn new(weight_source: WeightSource, cache_policy: CachePolicy, device: Device) -> Self {
        Self {
            weight_source,
            cache_policy,
            device,
            resident_static: false,
            expert_cache_bytes: 0,
            expert_cache_layout: ExpertCacheLayout::PerLayerSplit,
            expert_cache_replacement: ExpertCacheReplacementPolicy::Lru,
            expert_cache_min_bytes: 0,
            adaptive_expert_cache: false,
            cpu_fp8_dequantization: false,
            host_expert_share: 0.0,
            pinned_fp8_transfer: false,
            progress: false,
        }
    }

    pub fn with_resident_static(mut self, resident: bool) -> Self {
        self.resident_static = resident;
        self
    }

    /// CPU reference conversion on a CUDA model, for fallback and paired timing.
    pub fn with_pinned_fp8_transfer(mut self, enabled: bool) -> Self {
        self.pinned_fp8_transfer = enabled;
        self
    }

    pub fn with_cpu_fp8_dequantization(mut self, enabled: bool) -> Self {
        self.cpu_fp8_dequantization = enabled;
        self
    }

    #[cfg(test)]
    fn host_expert_budget_for(&self, misses: usize) -> usize {
        host_expert_budget(self.host_expert_share, misses)
    }

    /// Evaluate this fraction of each routed miss set on the host. Clamped to
    /// a fraction; a caller that has measured nothing should pass nothing.
    pub fn with_host_expert_share(mut self, share: f64) -> Self {
        self.host_expert_share = f64::from(host_expert_share_per_mille(share)) / 1000.0;
        self
    }

    pub fn with_progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    pub fn with_expert_cache_bytes(mut self, bytes: usize) -> Self {
        self.expert_cache_bytes = bytes;
        self
    }

    pub fn with_expert_cache_policy(
        mut self,
        layout: ExpertCacheLayout,
        replacement: ExpertCacheReplacementPolicy,
    ) -> Self {
        self.expert_cache_layout = layout;
        self.expert_cache_replacement = replacement;
        self
    }

    pub fn with_adaptive_expert_cache(mut self, minimum_bytes: usize) -> Self {
        self.expert_cache_min_bytes = minimum_bytes;
        self.adaptive_expert_cache = true;
        self
    }
}

pub struct StreamedGlm {
    weights: ModelWeights,
    config: GlmConfig,
    generation_config: GlmGenerationConfig,
    tokenizer: Tokenizer,
    device: Device,
    compute_dtype: DType,
    /// Fraction of a routed miss set to evaluate on the host, from a measured
    /// profile of this machine. Zero until one says otherwise.
    host_expert_share: f64,
    static_weights: BTreeMap<String, Tensor>,
    expert_cache: ExpertCacheManager,
    /// F32 mapping constants (weight, base, scale), cached by `hc_{site}` prefix.
    mhc_consts: std::sync::Mutex<std::collections::HashMap<String, (Tensor, Tensor, Tensor)>>,
    execution_policy: GlmExecutionPolicy,
    admission: Option<GlmAdmissionModel>,
    admission_breakdown: Option<crate::admission::GlmAdmissionBreakdown>,
    #[cfg(feature = "cuda")]
    weight_pool: std::sync::OnceLock<Option<fp8::cuda::WeightPool>>,
}

/// A validated metadata catalog with no weight payloads loaded. Capacity is
/// supplied explicitly at the transition to executable model residency.
pub struct PreparedGlm {
    model: StreamedGlm,
    breakdown: crate::admission::GlmAdmissionBreakdown,
    resident_static: bool,
    expert_cache_bytes: usize,
    progress: bool,
}

impl PreparedGlm {
    pub fn config(&self) -> &GlmConfig {
        &self.model.config
    }
    pub fn device(&self) -> &Device {
        &self.model.device
    }
    pub fn cache_stats(&self) -> CacheStats {
        self.model.weights.cache_stats()
    }
    pub fn access_stats(&self) -> WeightAccessStats {
        self.model.weights.access_stats()
    }
    pub fn execution_policy(&self) -> &GlmExecutionPolicy {
        &self.model.execution_policy
    }

    /// Apply only resource knobs while this object still owns metadata only.
    pub fn select_execution_policy(&mut self, policy: &GlmExecutionPolicy) -> Result<()> {
        policy.validate()?;
        let mut invariant = policy.clone();
        invariant.weights = self.model.execution_policy.weights.clone();
        invariant.resident_static = self.model.execution_policy.resident_static;
        invariant.expert_cache = self.model.execution_policy.expert_cache.clone();
        ensure!(
            invariant == self.model.execution_policy,
            "resource selection changes GLM numerical identity"
        );
        self.model
            .weights
            .configure_unloaded_cache(WeightSource::Mmap, policy.cache_policy()?)?;
        let layers = self
            .model
            .config
            .text_config
            .mlp_layer_types
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| (*kind == MlpKind::Sparse).then_some(layer))
            .collect::<Vec<_>>();
        let bytes = usize::try_from(policy.expert_cache.maximum_bound_bytes)?;
        self.model.expert_cache = ExpertCacheManager::new(
            self.model.config.text_config.num_hidden_layers,
            &layers,
            bytes,
            policy.expert_cache.layout,
            policy.expert_cache.replacement,
        )?;
        self.model.execution_policy = policy.clone();
        self.resident_static = policy.resident_static;
        self.expert_cache_bytes = bytes;
        Ok(())
    }

    pub fn prompt_token_count(&self, prompt: &str, reasoning_effort: &str) -> Result<usize> {
        ensure!(!prompt.trim().is_empty(), "GLM prompt must not be empty");
        let rendered = render_chat_prompt(prompt, reasoning_effort)?;
        let encoding = self
            .model
            .tokenizer
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!("failed to encode GLM prompt: {error}"))?;
        ensure!(
            !encoding.is_empty(),
            "GLM prompt tokenization produced no tokens"
        );
        Ok(encoding.len())
    }

    pub fn estimate(
        &self,
        prompt_tokens: usize,
    ) -> Result<crate::admission::GlmAdmissionBreakdown> {
        let mut breakdown = self.breakdown.clone();
        breakdown.prefill_workspace_bytes =
            prefill::prefill_workspace_bytes(&self.model.config.text_config, prompt_tokens)?;
        breakdown.prompt_tokens = prompt_tokens;
        breakdown.prefill_host_mask_bytes =
            crate::admission::host_mask_bytes(prompt_tokens, self.model.device.is_cpu())?;
        Ok(breakdown)
    }

    /// The supplied observation is captured after metadata preparation. The
    /// caller controls freshness; this operation never recaptures or reselects.
    pub fn open(
        mut self,
        prompt_tokens: usize,
        snapshot: &ResourceSnapshot,
    ) -> Result<StreamedGlm> {
        let breakdown = self.estimate(prompt_tokens)?;
        breakdown.validate_capacity(
            self.resident_static,
            self.expert_cache_bytes,
            self.model.weights.cache_policy(),
            snapshot,
        )?;
        self.model.admission = Some(GlmAdmissionModel {
            maximum_dsa_cache_bytes: breakdown.maximum_dsa_cache_bytes,
            dsa_cache_bytes_per_token: breakdown.dsa_cache_bytes_per_token,
            streamed_transient_bytes: if self.resident_static {
                0
            } else {
                breakdown.largest_streamed_group_bytes
            },
            pending_lm_head_bytes: if self.resident_static {
                0
            } else {
                breakdown.lm_head_bytes
            },
            live_expert_bytes: breakdown.live_expert_bytes,
            weight_load_staging_bytes: usize::try_from(if breakdown.compute_on_host {
                if self.resident_static {
                    breakdown.expert_load_host_bytes
                } else {
                    breakdown.streamed_load_host_bytes
                }
            } else if self.resident_static {
                breakdown.expert_load_device_bytes
            } else {
                breakdown
                    .static_load_device_bytes
                    .max(breakdown.expert_load_device_bytes)
            })?
            .checked_add(usize::try_from(breakdown.pinned_transfer_bytes)?)
            .context("FP8 staging admission overflow")?,
            safety_bytes: usize::try_from(crate::execution_policy::GLM_ADMISSION_SAFETY_BYTES)?,
        });
        self.model.admission_breakdown = Some(breakdown);
        if self.resident_static {
            self.model.preload_static_weights(self.progress)?;
        }
        Ok(self.model)
    }
}

#[cfg(feature = "cuda")]
struct FillPipeline<'a> {
    pool: &'a fp8::cuda::WeightPool,
    device: &'a candle_core::CudaDevice,
    ring: std::sync::Arc<fp8::staging::FillRing>,
    names: Vec<&'a str>,
    workers: usize,
}

/// Abort and wake both condvars on drop so scoped threads can join after failure.
/// The uploader stays armed; workers disarm on success to let pending fills finish.
#[cfg(feature = "cuda")]
struct FillAbortOnDrop<'a> {
    abort: &'a std::sync::atomic::AtomicBool,
    free_cv: &'a std::sync::Condvar,
    ready_cv: &'a std::sync::Condvar,
    armed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for FillAbortOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.abort.store(true, std::sync::atomic::Ordering::Relaxed);
            self.free_cv.notify_all();
            self.ready_cv.notify_all();
        }
    }
}

impl StreamedGlm {
    pub fn open(model_dir: impl AsRef<Path>, options: StreamedGlmOptions) -> Result<Self> {
        let prepared = Self::prepare(model_dir, options)?;
        let snapshot = ResourceSnapshot::capture(Some(prepared.device()));
        prepared.open(1, &snapshot)
    }

    pub fn prepare(
        model_dir: impl AsRef<Path>,
        options: StreamedGlmOptions,
    ) -> Result<PreparedGlm> {
        let model_dir = model_dir.as_ref();
        let config = GlmConfig::from_model_dir(model_dir)?;
        let generation_config = GlmGenerationConfig::from_model_dir(model_dir)?;
        generation_config.validate_for_text(&config.text_config)?;
        ensure!(
            config.text_config.n_shared_experts == 1,
            "the first GLM execution profile requires exactly one shared expert"
        );
        ensure!(
            config.text_config.n_group == 1 && config.text_config.topk_group == 1,
            "the first GLM execution profile requires the checkpoint's single router group"
        );
        ensure!(
            config.text_config.linear_lower_bound.is_some(),
            "the first GLM execution profile requires the safe KDA forget gate"
        );
        ensure!(
            options.device.is_cpu() || options.device.is_cuda(),
            "GLM text inference currently supports CPU or CUDA devices"
        );
        ensure!(
            !options.device.is_cuda() || config.text_config.linear_head_dim == 128,
            "GLM CUDA batched prefill requires KDA head width 128"
        );
        if options.device.is_cuda() {
            let text = &config.text_config;
            ensure!(
                text.n_routed_experts <= 288,
                "GLM CUDA router supports at most 288 experts"
            );
            ensure!(
                text.hc_mult < 64,
                "GLM CUDA mHC requires fewer than 64 streams"
            );
            for width in [
                text.hidden_size,
                text.hidden_size
                    .checked_mul(text.hc_mult)
                    .context("GLM mHC width overflow")?,
                text.linear_head_dim,
                text.q_lora_rank,
                text.kv_lora_rank,
            ] {
                ensure!(
                    math::normalization::CUDA_WIDTHS.contains(&width),
                    "unverified GLM CUDA normalization width {width}"
                );
            }
        }
        ensure!(
            options.weight_source == WeightSource::Mmap,
            "GLM inference requires --weight-source mmap; shard-granular memory caching would repeatedly read multi-GiB shards"
        );
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            anyhow::anyhow!(
                "failed to load GLM tokenizer {}: {error}",
                tokenizer_path.display()
            )
        })?;
        let weights = ModelWeights::open(model_dir, options.weight_source, options.cache_policy)?;
        let sparse_layers = config
            .text_config
            .mlp_layer_types
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| (*kind == MlpKind::Sparse).then_some(layer))
            .collect::<Vec<_>>();
        ensure!(
            !sparse_layers.is_empty(),
            "the supported GLM profile requires at least one sparse MLP layer"
        );
        let expert_cache = ExpertCacheManager::new(
            config.text_config.num_hidden_layers,
            &sparse_layers,
            options.expert_cache_bytes,
            options.expert_cache_layout,
            options.expert_cache_replacement,
        )?;
        let expert_cache_min_bytes = if options.adaptive_expert_cache {
            options.expert_cache_min_bytes
        } else {
            options.expert_cache_bytes
        };
        let mut execution_policy = GlmExecutionPolicy::from_runtime(
            &options.device,
            options.weight_source,
            options.cache_policy,
            options.resident_static,
            config.text_config.index_topk,
            options.expert_cache_layout,
            options.expert_cache_replacement,
            options.expert_cache_bytes,
            expert_cache_min_bytes,
            options.adaptive_expert_cache,
        )?;
        if options.cpu_fp8_dequantization {
            execution_policy.cpu_fp8_dequantization = true;
        }
        execution_policy.pinned_fp8_transfer = options.pinned_fp8_transfer;
        // Resolve FF_GLM_HOST_SHARE (per-mille) here so the policy records the actual split.
        let host_share_override = std::env::var("FF_GLM_HOST_SHARE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(|per_mille| f64::from(per_mille.min(1000)) / 1000.0);
        execution_policy.host_expert_share_per_mille =
            host_expert_share_per_mille(host_share_override.unwrap_or(options.host_expert_share));
        execution_policy.validate()?;
        let model = Self {
            weights,
            config,
            generation_config,
            compute_dtype: if options.device.is_cpu() {
                DType::F32
            } else {
                DType::BF16
            },
            host_expert_share: host_share_override.unwrap_or(options.host_expert_share),
            device: options.device,
            tokenizer,
            static_weights: BTreeMap::new(),
            expert_cache,
            mhc_consts: std::sync::Mutex::new(std::collections::HashMap::new()),
            execution_policy,
            admission: None,
            admission_breakdown: None,
            #[cfg(feature = "cuda")]
            weight_pool: std::sync::OnceLock::new(),
        };
        model.validate_inventory()?;
        let mut breakdown = crate::admission::GlmAdmissionBreakdown::from_metadata_with_fp8(
            &model.weights,
            &model.config.text_config,
            model.device.is_cpu(),
            1,
            model.execution_policy.cpu_fp8_dequantization,
        )?;
        if options.pinned_fp8_transfer {
            breakdown.enable_pinned_transfer()?;
        }
        Ok(PreparedGlm {
            model,
            breakdown,
            resident_static: options.resident_static,
            expert_cache_bytes: options.expert_cache_bytes,
            progress: options.progress,
        })
    }

    pub fn admission_breakdown(&self) -> &crate::admission::GlmAdmissionBreakdown {
        self.admission_breakdown
            .as_ref()
            .expect("opened GLM has an admission model")
    }

    pub fn config(&self) -> &GlmConfig {
        &self.config
    }

    pub fn execution_policy(&self) -> &GlmExecutionPolicy {
        &self.execution_policy
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.weights.cache_stats()
    }

    pub fn access_stats(&self) -> WeightAccessStats {
        self.weights.access_stats()
    }

    pub fn resident_static_bytes(&self) -> usize {
        self.static_weights
            .values()
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum()
    }

    pub fn fp8_transfer_stats(&self) -> Result<fp8::Fp8TransferStats> {
        #[cfg(feature = "cuda")]
        if let Some(pool) = self.weight_pool.get().and_then(Option::as_ref) {
            return pool.transfer_stats();
        }
        Ok(fp8::Fp8TransferStats::default())
    }
    #[cfg(feature = "cuda")]
    fn fp8_staging_bytes(&self) -> Result<u64> {
        self.weight_pool
            .get()
            .and_then(Option::as_ref)
            .map(|p| p.staging_bytes())
            .transpose()
            .map(|n| n.unwrap_or(0))
    }
    #[cfg(feature = "cuda")]
    fn begin_expert_compute(
        &self,
    ) -> Result<Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>> {
        if !self.execution_policy.pinned_fp8_transfer {
            return Ok(None);
        }
        if let (Some(pool), Device::Cuda(device)) = (
            self.weight_pool.get().and_then(Option::as_ref),
            &self.device,
        ) {
            return pool.begin_compute(device);
        }
        Ok(None)
    }
    #[cfg(feature = "cuda")]
    fn end_expert_compute(
        &self,
        start: Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    ) -> Result<()> {
        if let (Some(pool), Device::Cuda(device)) = (
            self.weight_pool.get().and_then(Option::as_ref),
            &self.device,
        ) {
            pool.end_compute(device, start)?;
        }
        Ok(())
    }

    pub fn expert_cache_stats(&self) -> ExpertCacheStats {
        self.expert_cache
            .stats()
            .expect("validated GLM expert cache statistics overflowed")
    }

    fn readmit_expert_cache(
        &self,
        completed_tokens: usize,
        phase: RoutingTracePhase,
        recorder: &mut GlmExecutionManifestRecorder,
    ) -> Result<()> {
        if self
            .execution_policy
            .expert_cache
            .readmission_interval_tokens
            .is_none()
        {
            return Ok(());
        }
        let admission = self
            .admission
            .context("GLM runtime admission model is unavailable")?;
        let before = self.expert_cache.stats()?;
        let snapshot = ResourceSnapshot::capture(Some(&self.device));
        let (available, memory_kind) = if self.device.is_cpu() {
            (crate::admission::host_available(&snapshot), "host")
        } else {
            (snapshot.device_free_memory_bytes, "CUDA")
        };
        let available = available.with_context(|| {
            format!("cannot measure free {memory_kind} memory for GLM cache re-admission")
        })?;
        let plan = plan_cache_readmission(
            &self.execution_policy,
            admission,
            completed_tokens,
            phase,
            available,
            before,
        )?;
        let resize = self.expert_cache.resize(
            usize::try_from(plan.new_bound_bytes)
                .context("GLM admitted cache bound exceeds usize")?,
        )?;
        ensure!(
            u64::try_from(resize.previous_max_bytes)? == plan.previous_bound_bytes
                && u64::try_from(resize.previous_resident_bytes)? == plan.realized_before_bytes
                && u64::try_from(resize.new_max_bytes)? == plan.new_bound_bytes,
            "GLM cache manager state changed between re-admission planning and resize"
        );
        if plan.decision == GlmCacheAdmissionDecision::Shrunk {
            self.device.synchronize()?;
        }
        recorder.record(GlmCacheReadmissionEvent {
            sequence: recorder.next_sequence()?,
            after_routed_token: u32::try_from(completed_tokens)
                .context("GLM cache readmission token index exceeds u32")?,
            phase,
            available_memory_bytes: plan.available_memory_bytes,
            reclaimable_capacity_bytes: plan.reclaimable_capacity_bytes,
            required_future_headroom_bytes: plan.required_future_headroom_bytes,
            admissible_cache_bound_bytes: plan.admissible_cache_bound_bytes,
            previous_bound_bytes: plan.previous_bound_bytes,
            new_bound_bytes: plan.new_bound_bytes,
            realized_before_bytes: plan.realized_before_bytes,
            realized_after_bytes: u64::try_from(resize.new_resident_bytes)
                .context("GLM new cache residency exceeds u64")?,
            evictions: resize.evictions,
            decision: plan.decision,
        })
    }

    pub fn generate(&self, prompt: &str, options: &GlmGenerationOptions) -> Result<GlmGeneration> {
        let (generation, trace) = self.generate_inner(prompt, options, None)?;
        debug_assert!(trace.is_none());
        Ok(generation)
    }

    /// Generate text while recording the router's already-selected expert IDs.
    ///
    /// Trace collection is a post-routing side channel: it never supplies an
    /// expert choice or changes the reference accumulation order.
    pub fn generate_with_routing_trace(
        &self,
        prompt: &str,
        options: &GlmGenerationOptions,
        domain: &str,
    ) -> Result<(GlmGeneration, RoutingTrace)> {
        let (generation, trace) = self.generate_inner(prompt, options, Some(domain))?;
        Ok((
            generation,
            trace.context("GLM routing-trace generation produced no trace")?,
        ))
    }

    /// Capture the exact first-next-token boundary used for reference parity.
    ///
    /// This deliberately exposes no alternative attention, precision, or
    /// sampling path: it runs ordinary prefill and exports the F32 hidden state
    /// consumed by the existing language-model head plus the complete F32
    /// logits produced by that head.
    pub fn capture_first_next_token_parity(
        &self,
        prompt: &str,
        reasoning_effort: &str,
        max_context_tokens: usize,
        progress: bool,
    ) -> Result<GlmParityCapture> {
        ensure!(
            self.execution_policy
                .expert_cache
                .readmission_interval_tokens
                .is_none(),
            "GLM parity capture requires a fixed expert-cache policy"
        );
        ensure!(!prompt.trim().is_empty(), "GLM prompt must not be empty");
        ensure!(
            max_context_tokens > 0 && max_context_tokens <= self.config.text_config.index_topk,
            "GLM parity capture supports max_context_tokens in 1..={}",
            self.config.text_config.index_topk
        );
        let rendered = render_chat_prompt(prompt, reasoning_effort)?;
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!("failed to tokenize GLM parity prompt: {error}"))?;
        let prompt_token_ids = encoding.get_ids().to_vec();
        ensure!(
            !prompt_token_ids.is_empty() && prompt_token_ids.len() <= max_context_tokens,
            "GLM parity prompt has {} tokens; expected 1..={max_context_tokens}",
            prompt_token_ids.len()
        );
        self.admit_prefill(prompt_token_ids.len())?;
        let mut state = DecoderState::new(&self.config, self.compute_dtype, &self.device)?;
        let mut no_trace = None;
        let hidden =
            self.forward_prefill(&prompt_token_ids, &mut state, &mut no_trace, progress)?;
        let lm_head = self.load_linear_weight(LM_HEAD_WEIGHT)?;
        let logits = linear(&hidden.unsqueeze(0)?, &lm_head)?.squeeze(0)?;
        self.device.synchronize()?;
        let next_token_id = argmax_token(&logits)?;
        let final_hidden_state = hidden
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?;
        let next_token_logits = logits
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?;
        ensure!(
            final_hidden_state
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.is_finite()),
            "GLM parity hidden state contains a non-finite value"
        );
        ensure!(
            next_token_logits
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.is_finite()),
            "GLM parity logits contain a non-finite value"
        );
        Ok(GlmParityCapture {
            prompt_token_ids,
            final_hidden_state,
            next_token_logits,
            next_token_id,
        })
    }

    fn generate_inner(
        &self,
        prompt: &str,
        options: &GlmGenerationOptions,
        trace_domain: Option<&str>,
    ) -> Result<(GlmGeneration, Option<RoutingTrace>)> {
        ensure!(!prompt.trim().is_empty(), "GLM prompt must not be empty");
        ensure!(
            options.max_new_tokens > 0,
            "GLM max_new_tokens must be positive"
        );
        ensure!(
            matches!(options.reasoning_effort.as_str(), "low" | "high" | "max"),
            "GLM reasoning effort must be low, high, or max"
        );
        ensure!(
            options.temperature.is_finite() && options.temperature >= 0.0,
            "GLM temperature must be finite and non-negative"
        );
        ensure!(
            options.temperature == 0.0 || options.temperature.recip().is_finite(),
            "GLM positive temperature is too small to invert without overflow"
        );
        ensure!(
            options.top_p.is_finite() && options.top_p > 0.0 && options.top_p <= 1.0,
            "GLM top_p must be finite and in (0, 1]"
        );
        let exact_limit = self.config.text_config.index_topk;
        ensure!(
            options.max_context_tokens > 0 && options.max_context_tokens <= exact_limit,
            "GLM text-only exact-attention profile supports max_context_tokens in 1..={exact_limit}"
        );
        ensure!(
            options.max_new_tokens <= options.max_context_tokens,
            "GLM max_new_tokens exceeds max_context_tokens"
        );

        let rendered = render_chat_prompt(prompt, &options.reasoning_effort)?;
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!("failed to tokenize GLM prompt: {error}"))?;
        let prompt_ids = encoding.get_ids();
        ensure!(
            !prompt_ids.is_empty(),
            "GLM prompt tokenized to an empty sequence"
        );
        let request_tokens = prompt_ids
            .len()
            .checked_add(options.max_new_tokens)
            .context("GLM request token count overflow")?;
        ensure!(
            request_tokens <= options.max_context_tokens,
            "GLM request needs at most {request_tokens} tokens but max_context_tokens is {}",
            options.max_context_tokens
        );

        let mut routing_trace = trace_domain
            .map(|domain| {
                self.new_routing_trace_builder(domain, prompt_ids.len(), options.max_context_tokens)
            })
            .transpose()?;
        let mut execution_recorder = GlmExecutionManifestRecorder::new(
            self.execution_policy.clone(),
            self.expert_cache_stats(),
        )?;

        self.admit_prefill(prompt_ids.len())?;
        let mut state = DecoderState::new(&self.config, self.compute_dtype, &self.device)?;
        let prefill_started = Instant::now();
        let mut hidden =
            self.forward_prefill(prompt_ids, &mut state, &mut routing_trace, options.progress)?;
        self.device.synchronize()?;
        self.readmit_expert_cache(
            state.tokens,
            RoutingTracePhase::Prefill,
            &mut execution_recorder,
        )?;
        let prefill_elapsed = prefill_started.elapsed();

        let lm_head = self.load_linear_weight(LM_HEAD_WEIGHT)?;
        let decode_started = Instant::now();
        let mut rng = StdRng::seed_from_u64(options.seed);
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        let mut token_elapsed = Vec::with_capacity(options.max_new_tokens);
        for step in 0..options.max_new_tokens {
            let started = Instant::now();
            let logits = linear(&hidden.unsqueeze(0)?, &lm_head)?.squeeze(0)?;
            let token = sample_token(&logits, options.temperature, options.top_p, &mut rng)?;
            generated.push(token);
            token_elapsed.push(started.elapsed());
            if options.progress {
                eprintln!(
                    "GLM decode token {}/{} selected id {} in {:.2}s",
                    step + 1,
                    options.max_new_tokens,
                    token,
                    started.elapsed().as_secs_f64()
                );
            }
            if self.generation_config.eos_token_ids.contains(&token) {
                break;
            }
            if step + 1 < options.max_new_tokens {
                let forward_started = Instant::now();
                hidden = self.forward_token(
                    token,
                    &mut state,
                    RoutingTracePhase::Decode,
                    &mut routing_trace,
                )?;
                self.device.synchronize()?;
                self.readmit_expert_cache(
                    state.tokens,
                    RoutingTracePhase::Decode,
                    &mut execution_recorder,
                )?;
                let elapsed = forward_started.elapsed();
                let recorded = token_elapsed
                    .last_mut()
                    .context("GLM decode timing row is missing after token selection")?;
                *recorded = recorded
                    .checked_add(elapsed)
                    .context("GLM per-token duration overflow")?;
                if options.progress {
                    eprintln!(
                        "GLM decode token {} state update completed in {:.2}s",
                        step + 1,
                        elapsed.as_secs_f64()
                    );
                }
            }
        }
        let decode_elapsed = decode_started.elapsed();
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!("failed to decode GLM output tokens: {error}"))?;
        let routing_trace = routing_trace
            .map(|builder| builder.finish(generated.len()))
            .transpose()?;
        let execution_manifest = execution_recorder.finish(
            prompt_ids.len(),
            generated.len(),
            state.tokens,
            self.expert_cache_stats(),
        )?;
        Ok((
            GlmGeneration {
                prompt_tokens: prompt_ids.len(),
                generated_token_ids: generated,
                text,
                prefill_elapsed,
                decode_elapsed,
                token_elapsed,
                execution_manifest,
            },
            routing_trace,
        ))
    }

    fn forward_token(
        &self,
        token: u32,
        state: &mut DecoderState,
        trace_phase: RoutingTracePhase,
        routing_trace: &mut Option<RoutingTraceBuilder>,
    ) -> Result<Tensor> {
        ensure!(
            state.tokens < self.config.text_config.index_topk,
            "GLM DSA exact-attention profile exceeded {} cached tokens",
            self.config.text_config.index_topk
        );
        let text = &self.config.text_config;
        let embedding = self
            .weights
            .load_rows(EMBEDDING_WEIGHT, &[token], &self.device)
            .with_context(|| format!("failed to gather GLM embedding row {token}"))?
            .to_dtype(self.compute_dtype)?;
        let mut streams = embedding.repeat((text.hc_mult, 1))?;
        for layer in 0..text.num_hidden_layers {
            streams = self
                .forward_layer(
                    layer,
                    state.tokens,
                    trace_phase,
                    &streams,
                    &mut state.layers[layer],
                    routing_trace,
                )
                .with_context(|| format!("GLM layer {layer} failed"))?;
        }
        state.tokens += 1;
        let collapsed = streams
            .to_dtype(DType::F32)?
            .mean(0)?
            .to_dtype(self.compute_dtype)?;
        let norm_weight = self.load_tensor(FINAL_NORM_WEIGHT)?;
        let normed = math::rms_norm(&collapsed, Some(&norm_weight), text.rms_norm_eps)?;
        Ok(normed)
    }

    fn forward_layer(
        &self,
        layer: usize,
        token_index: usize,
        trace_phase: RoutingTracePhase,
        input_streams: &Tensor,
        cache: &mut LayerCache,
        routing_trace: &mut Option<RoutingTraceBuilder>,
    ) -> Result<Tensor> {
        let text = &self.config.text_config;
        let prefix = format!("model.language_model.layers.{layer}");

        let residual = input_streams.clone();
        let (post, comb, collapsed) = self.hyper_map(&prefix, "attn", input_streams)?;
        let norm_weight = self.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
        let normalized = math::rms_norm(&collapsed, Some(&norm_weight), text.rms_norm_eps)?;
        let attended = match (text.layer_types[layer], cache) {
            (AttentionKind::LinearAttention, LayerCache::Kda(cache)) => {
                self.kda_attention(&prefix, &normalized, cache)?
            }
            (AttentionKind::DeepseekSparseAttention, LayerCache::Dsa(cache)) => {
                self.mla_attention(&prefix, &normalized, cache)?
            }
            _ => bail!("GLM layer cache kind disagrees with its configured attention kind"),
        };
        let streams = math::apply_mhc_residual(&residual, &attended, &post, &comb)?;

        let residual = streams.clone();
        let (post, comb, collapsed) = self.hyper_map(&prefix, "ffn", &streams)?;
        let norm_weight = self.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normalized = math::rms_norm(&collapsed, Some(&norm_weight), text.rms_norm_eps)?;
        let fed_forward = match text.mlp_layer_types[layer] {
            MlpKind::Dense => self.dense_mlp(&format!("{prefix}.mlp"), &normalized)?,
            MlpKind::Sparse => self.sparse_moe(
                layer,
                token_index,
                trace_phase,
                &format!("{prefix}.mlp"),
                &normalized,
                routing_trace,
            )?,
        };
        math::apply_mhc_residual(&residual, &fed_forward, &post, &comb)
    }

    fn hyper_map(
        &self,
        prefix: &str,
        site: &str,
        streams: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let text = &self.config.text_config;
        let key = format!("{prefix}.hc_{site}");
        let (weight, base, scale) = {
            let mut cache = self
                .mhc_consts
                .lock()
                .expect("GLM mHC constants mutex poisoned");
            match cache.get(&key) {
                Some(consts) => consts.clone(),
                None => {
                    let consts = (
                        self.load_tensor(&format!("{key}_fn"))?
                            .to_dtype(DType::F32)?,
                        self.load_tensor(&format!("{key}_base"))?
                            .to_dtype(DType::F32)?,
                        self.load_tensor(&format!("{key}_scale"))?
                            .to_dtype(DType::F32)?,
                    );
                    cache.insert(key, consts.clone());
                    consts
                }
            }
        };
        math::mhc_map(
            streams,
            &weight,
            &base,
            &scale,
            text.rms_norm_eps,
            text.hc_eps,
            text.hc_sinkhorn_iters,
        )
    }

    fn kda_attention(&self, prefix: &str, hidden: &Tensor, cache: &mut KdaCache) -> Result<Tensor> {
        let text = &self.config.text_config;
        let input = hidden.unsqueeze(0)?;
        let query = self.project_and_convolve(prefix, "q", &input, &mut cache.query_conv)?;
        let key = self.project_and_convolve(prefix, "k", &input, &mut cache.key_conv)?;
        let value = self.project_and_convolve(prefix, "v", &input, &mut cache.value_conv)?;

        let f_a = self.load_linear_weight(&format!("{prefix}.self_attn.f_a_proj.weight"))?;
        let f_b = self.load_linear_weight(&format!("{prefix}.self_attn.f_b_proj.weight"))?;
        let dt_bias = self.load_tensor(&format!("{prefix}.self_attn.dt_bias"))?;
        let a_log = self.load_tensor(&format!("{prefix}.self_attn.A_log"))?;
        let log_decay = math::kda_forget_gate(
            &input,
            &f_a,
            &f_b,
            &dt_bias,
            &a_log,
            text.linear_lower_bound
                .context("validated KDA lower bound is missing")?,
        )?
        .squeeze(0)?;

        let beta_weight = self.load_linear_weight(&format!("{prefix}.self_attn.b_proj.weight"))?;
        let beta =
            math::sigmoid_with_reference_rounding(&linear(&input, &beta_weight)?)?.squeeze(0)?;
        let query = query.reshape((text.linear_num_heads, text.linear_head_dim))?;
        let key = key.reshape((text.linear_num_heads, text.linear_head_dim))?;
        let value = value.reshape((text.linear_num_heads, text.linear_head_dim))?;
        let (attended, next_state) =
            math::kda_single_token(&query, &key, &value, &log_decay, &beta, &cache.recurrent)?;
        cache.recurrent = next_state;

        let g_a = self.load_linear_weight(&format!("{prefix}.self_attn.g_a_proj.weight"))?;
        let g_b = self.load_linear_weight(&format!("{prefix}.self_attn.g_b_proj.weight"))?;
        let gate = linear(&linear(&input, &g_a)?, &g_b)?
            .reshape((text.linear_num_heads, text.linear_head_dim))?;
        let o_norm = self.load_tensor(&format!("{prefix}.self_attn.o_norm.weight"))?;
        let attended = math::rms_norm_gated(&attended, &o_norm, &gate, text.rms_norm_eps)?
            .reshape((1, text.linear_num_heads * text.linear_head_dim))?;
        let o_proj = self.load_linear_weight(&format!("{prefix}.self_attn.o_proj.weight"))?;
        linear(&attended, &o_proj)?.squeeze(0).map_err(Into::into)
    }

    fn project_and_convolve(
        &self,
        prefix: &str,
        name: &str,
        input: &Tensor,
        conv_state: &mut Tensor,
    ) -> Result<Tensor> {
        let projection =
            self.load_linear_weight(&format!("{prefix}.self_attn.{name}_proj.weight"))?;
        let projected = linear(input, &projection)?.squeeze(0)?;
        let conv_weight = self
            .load_tensor(&format!("{prefix}.self_attn.{name}_conv1d.weight"))?
            .squeeze(1)?;
        let kernel = conv_weight.dim(1)?;
        ensure!(
            conv_state.dims() == [projected.dim(0)?, kernel],
            "KDA convolution state has the wrong shape"
        );
        let appended = Tensor::cat(&[conv_state, &projected.unsqueeze(1)?], 1)?;
        let start = appended.dim(1)? - kernel;
        let window = appended.narrow(1, start, kernel)?.contiguous()?;
        let convolved = window
            .to_dtype(DType::F32)?
            .mul(&conv_weight.to_dtype(DType::F32)?)?
            .sum(1)?
            .to_dtype(self.compute_dtype)?;
        let output = math::silu_with_reference_rounding(&convolved)?;
        *conv_state = window;
        Ok(output)
    }

    fn mla_attention(&self, prefix: &str, hidden: &Tensor, cache: &mut DsaCache) -> Result<Tensor> {
        let text = &self.config.text_config;
        let input = hidden.unsqueeze(0)?;
        let q_a_weight = self.load_linear_weight(&format!("{prefix}.self_attn.q_a_proj.weight"))?;
        let q_resid = linear(&input, &q_a_weight)?;
        let q_a_norm = self.load_tensor(&format!("{prefix}.self_attn.q_a_layernorm.weight"))?;
        let q_resid = math::rms_norm(&q_resid, Some(&q_a_norm), text.rms_norm_eps)?;
        let q_b_weight = self.load_linear_weight(&format!("{prefix}.self_attn.q_b_proj.weight"))?;
        let query = linear(&q_resid, &q_b_weight)?
            .reshape((text.num_attention_heads, text.mla_qk_head_dim()?))?;

        let kv_a_weight =
            self.load_linear_weight(&format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"))?;
        let compressed = linear(&input, &kv_a_weight)?;
        ensure!(
            text.qk_rope_head_dim == 0,
            "the exact short-context MLA path requires NoPE"
        );
        let kv_a_norm = self.load_tensor(&format!("{prefix}.self_attn.kv_a_layernorm.weight"))?;
        let compressed = math::rms_norm(&compressed, Some(&kv_a_norm), text.rms_norm_eps)?;
        let kv_b_weight =
            self.load_linear_weight(&format!("{prefix}.self_attn.kv_b_proj.weight"))?;
        let expanded = linear(&compressed, &kv_b_weight)?.reshape((
            text.num_attention_heads,
            text.qk_nope_head_dim + text.v_head_dim,
        ))?;
        let key = expanded.narrow(1, 0, text.qk_nope_head_dim)?.unsqueeze(1)?;
        let value = expanded
            .narrow(1, text.qk_nope_head_dim, text.v_head_dim)?
            .unsqueeze(1)?;
        cache.keys = Some(match cache.keys.take() {
            Some(previous) => Tensor::cat(&[&previous, &key], 1)?,
            None => key,
        });
        cache.values = Some(match cache.values.take() {
            Some(previous) => Tensor::cat(&[&previous, &value], 1)?,
            None => value,
        });
        let keys = cache
            .keys
            .as_ref()
            .context("DSA key cache was not populated")?;
        let values = cache
            .values
            .as_ref()
            .context("DSA value cache was not populated")?;
        let query = query.unsqueeze(1)?.contiguous()?;
        let key_t = keys.transpose(1, 2)?.contiguous()?;
        let scores = query
            .matmul(&key_t)?
            .affine(1.0 / (text.mla_qk_head_dim()? as f64).sqrt(), 0.0)?;
        let probabilities = ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(self.compute_dtype)?
            .contiguous()?;
        let values = values.contiguous()?;
        let attended = probabilities
            .matmul(&values)?
            .reshape((1, text.num_attention_heads * text.v_head_dim))?;
        let o_weight = self.load_linear_weight(&format!("{prefix}.self_attn.o_proj.weight"))?;
        linear(&attended, &o_weight)?.squeeze(0).map_err(Into::into)
    }

    fn dense_mlp(&self, prefix: &str, hidden: &Tensor) -> Result<Tensor> {
        self.mlp_projection(prefix, hidden, None)
    }

    fn sparse_moe(
        &self,
        layer: usize,
        token_index: usize,
        trace_phase: RoutingTracePhase,
        prefix: &str,
        hidden: &Tensor,
        routing_trace: &mut Option<RoutingTraceBuilder>,
    ) -> Result<Tensor> {
        let text = &self.config.text_config;
        let router_weight = self.load_tensor(&format!("{prefix}.gate.weight"))?;
        let correction = self.load_tensor(&format!("{prefix}.gate.e_score_correction_bias"))?;
        let routed = math::topk_router(
            &hidden.unsqueeze(0)?,
            &router_weight,
            &correction,
            text.num_experts_per_tok,
            text.routed_scaling_factor,
            text.norm_topk_prob,
        )?;
        let indices = routed
            .indices
            .to_device(&Device::Cpu)?
            .to_vec2::<u32>()?
            .into_iter()
            .next()
            .context("GLM router returned no token row")?;
        let weights = routed
            .weights
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
            .into_iter()
            .next()
            .context("GLM router returned no mixture-weight row")?;
        if let Some(trace) = routing_trace.as_mut() {
            trace.record(token_index, trace_phase, layer, &indices)?;
        }
        let mut selected = indices.into_iter().zip(weights).collect::<Vec<_>>();
        selected.sort_unstable_by_key(|(expert, _)| *expert);
        let cache = self.expert_cache.cache_for_layer(layer)?;
        let mut pending = selected
            .into_iter()
            .map(|(expert, mixture)| {
                ensure!(
                    (expert as usize) < text.n_routed_experts,
                    "GLM router selected out-of-range expert {expert}"
                );
                let expert_prefix = format!("{prefix}.experts.{expert}");
                let gate_name = format!("{expert_prefix}.gate_proj.weight");
                let up_name = format!("{expert_prefix}.up_proj.weight");
                let down_name = format!("{expert_prefix}.down_proj.weight");
                Ok(PendingExpert {
                    expert,
                    mixture,
                    gate: cache.get(&gate_name),
                    up: cache.get(&up_name),
                    down: cache.get(&down_name),
                    gate_name,
                    up_name,
                    down_name,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        pending.sort_by_key(|expert| !expert.is_complete());
        // Experts already held whole cost nothing to reuse and are never
        // rerouted; the split applies to the misses at the tail, and always
        // leaves one of them to the device path so the expert cache keeps
        // warming.
        let misses = pending.iter().filter(|e| !e.is_complete()).count();
        let host_budget = self.host_expert_budget(misses);
        let mut host_work = Vec::with_capacity(host_budget);
        let mut device_work = Vec::with_capacity(pending.len() - host_budget);
        for expert in pending {
            if host_work.len() < host_budget && !expert.is_complete() {
                host_work.push((
                    expert.expert,
                    format!("{prefix}.experts.{}", expert.expert),
                    expert.mixture as f64,
                ));
            } else {
                device_work.push(expert);
            }
        }

        // Warm pages for host experts and buffered device loads. O_DIRECT bypasses
        // the page cache, so warming direct device loads would duplicate disk reads.
        #[cfg(feature = "cuda")]
        let skip_device_warm = fp8::staging::direct_fill_active();
        #[cfg(not(feature = "cuda"))]
        let skip_device_warm = false;
        let mut prefetch_names: Vec<&str> = Vec::new();
        let mut device_iter = device_work.iter().filter(|expert| !expert.is_complete());
        let mut host_iter = host_work.iter();
        loop {
            let mut pushed = false;
            if !skip_device_warm && let Some(expert) = device_iter.next() {
                if expert.gate.is_none() {
                    prefetch_names.push(&expert.gate_name);
                }
                if expert.up.is_none() {
                    prefetch_names.push(&expert.up_name);
                }
                if expert.down.is_none() {
                    prefetch_names.push(&expert.down_name);
                }
                pushed = true;
            }
            if let Some((_, expert_prefix, _)) = host_iter.next() {
                prefetch_names.push(expert_prefix);
                pushed = true;
            }
            if !pushed {
                break;
            }
        }
        let mut prefetch_ranges: Vec<(std::path::PathBuf, u64, u64)> = Vec::new();
        for name in prefetch_names {
            // Host prefixes expand to three projections; include scales for every projection.
            let projections = if name.ends_with(".weight") {
                vec![name.to_owned()]
            } else {
                ["gate_proj", "up_proj", "down_proj"]
                    .iter()
                    .map(|projection| format!("{name}.{projection}.weight"))
                    .collect()
            };
            for projection in projections {
                for key in [projection.clone(), format!("{projection}_scale_inv")] {
                    if let Ok(metadata) = self.weights.raw_tensor_metadata(&key) {
                        prefetch_ranges.push((
                            self.weights.root().join(&metadata.shard),
                            metadata.file_offset as u64,
                            metadata.bytes as u64,
                        ));
                    }
                }
            }
        }

        // The host input is captured once, on this thread, because reading it
        // off the device is a device operation like any other.
        let host_input = host_work
            .is_empty()
            .not()
            .then(|| -> Result<Tensor> {
                Ok(hidden
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?)
            })
            .transpose()?;

        let mut contributions = Vec::with_capacity(device_work.len() + host_work.len());
        let weights = &self.weights;
        let swiglu_limit = self.config.text_config.swiglu_limit;
        let next_range = AtomicUsize::new(0);
        let next_host = AtomicUsize::new(0);
        let next_device = AtomicUsize::new(0);
        // Serialize matvecs because cuBLAS handles are not thread-safe; loads stay parallel.
        let matvec_lock = std::sync::Mutex::new(());
        // Fill workers write pinned buffers; the main thread uploads and evaluates in
        // routing order. Shared state lives outside the scope closure to outlive worker joins.
        let device_lanes = self.device_lanes().min(device_work.len().max(1));
        #[cfg(feature = "cuda")]
        let mut fill_pipeline: Option<FillPipeline<'_>> = None;
        #[cfg(feature = "cuda")]
        if device_lanes <= 1 && self.fill_ahead_workers() > 0 {
            let mut fill_names: Vec<&str> = Vec::new();
            for expert in &device_work {
                if expert.gate.is_none() {
                    fill_names.push(expert.gate_name.as_str());
                }
                if expert.up.is_none() {
                    fill_names.push(expert.up_name.as_str());
                }
                if expert.down.is_none() {
                    fill_names.push(expert.down_name.as_str());
                }
            }
            let mut block_fp8_only = !fill_names.is_empty();
            let mut max_bytes = 0usize;
            let mut max_scales = 0usize;
            for name in &fill_names {
                match self.weights.metadata(name) {
                    Ok(metadata) if metadata.dtype == "F8_E4M3" => {}
                    _ => {
                        block_fp8_only = false;
                        break;
                    }
                }
                if let Ok(raw) = self.weights.raw_tensor_metadata(name) {
                    max_bytes = max_bytes.max(raw.bytes);
                }
                if let Ok(raw) = self
                    .weights
                    .raw_tensor_metadata(&format!("{name}_scale_inv"))
                {
                    max_scales = max_scales.max(raw.bytes / 4);
                }
            }
            let workers = self.fill_ahead_workers().min(fill_names.len());
            if workers > 0 && block_fp8_only && max_bytes > 0 {
                self.ensure_weight_pool()?;
                if let (Some(pool), Device::Cuda(device)) = (
                    self.weight_pool.get().and_then(Option::as_ref),
                    &self.device,
                ) {
                    let ring =
                        pool.fill_ring(workers.saturating_mul(2).max(4), max_bytes, max_scales)?;
                    fill_pipeline = Some(FillPipeline {
                        pool,
                        device,
                        ring,
                        names: fill_names,
                        workers,
                    });
                }
            }
        }
        #[cfg(feature = "cuda")]
        let next_fill = AtomicUsize::new(0);
        #[cfg(feature = "cuda")]
        let free_buffers = std::sync::Mutex::new(Vec::<usize>::new());
        #[cfg(feature = "cuda")]
        let free_cv = std::sync::Condvar::new();
        #[cfg(feature = "cuda")]
        let ready_fills: std::sync::Mutex<
            std::collections::HashMap<usize, Result<(usize, fp8::staging::FillMeta)>>,
        > = std::sync::Mutex::new(std::collections::HashMap::new());
        #[cfg(feature = "cuda")]
        let ready_cv = std::sync::Condvar::new();
        #[cfg(feature = "cuda")]
        let abort = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| -> Result<()> {
            let warmers = expert_prefetch_threads().min(prefetch_ranges.len());
            for _ in 0..warmers {
                let ranges = &prefetch_ranges;
                let next = &next_range;
                scope.spawn(move || warm_checkpoint_pages(ranges, next));
            }
            // The whole point of the split: the host evaluates its share while
            // this thread is inside the device path, not before or after it.
            // Serially the host branch can only ever add its own time, which is
            // what the first version of this measured.
            let host_queue = &host_work;
            let host_next = &next_host;
            let workers: Vec<_> = match host_input.as_ref() {
                Some(input) => {
                    let threads = host_evaluation_threads().max(1).min(host_queue.len());
                    (0..threads)
                        .map(|_| {
                            scope.spawn(move || {
                                let mut evaluated = Vec::new();
                                loop {
                                    let index = host_next
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    let Some((expert, expert_prefix, mixture)) =
                                        host_queue.get(index)
                                    else {
                                        break;
                                    };
                                    evaluated.push((
                                        *expert,
                                        host_expert_contribution(
                                            weights,
                                            swiglu_limit,
                                            expert_prefix,
                                            input,
                                            *mixture,
                                        )?,
                                    ));
                                }
                                Ok::<_, anyhow::Error>(evaluated)
                            })
                        })
                        .collect()
                }
                None => Vec::new(),
            };

            // Each upload lane consumes a shared queue; a single lane runs inline.
            // Consumers wait for lane dequantization before matvecs on the compute stream.
            if device_lanes <= 1 {
                #[cfg(feature = "cuda")]
                let ran_pipeline = if let Some(FillPipeline {
                    pool,
                    device,
                    ring,
                    names: fill_names,
                    workers,
                }) = &fill_pipeline
                {
                    *free_buffers
                        .lock()
                        .map_err(|_| anyhow::anyhow!("GLM fill-ahead free lock poisoned"))? =
                        (0..ring.len()).collect();
                    // Wake waiting workers on every uploader exit so the scope can join.
                    let _uploader_guard = FillAbortOnDrop {
                        abort: &abort,
                        free_cv: &free_cv,
                        ready_cv: &ready_cv,
                        armed: true,
                    };
                    for _ in 0..*workers {
                        let next_fill = &next_fill;
                        let free_buffers = &free_buffers;
                        let free_cv = &free_cv;
                        let ready_fills = &ready_fills;
                        let ready_cv = &ready_cv;
                        let abort = &abort;
                        scope.spawn(move || -> Result<()> {
                            let mut worker_guard = FillAbortOnDrop {
                                abort,
                                free_cv,
                                ready_cv,
                                armed: true,
                            };
                            let result = (|| -> Result<()> {
                                loop {
                                    if abort.load(std::sync::atomic::Ordering::Relaxed) {
                                        return Ok(());
                                    }
                                    // Claim a buffer before an ordinal: otherwise later fills can occupy all
                                    // buffers while the ordered consumer waits for a worker without one.
                                    // Extra ring capacity keeps workers busy during out-of-order completion.
                                    let buffer_index = {
                                        let mut free = free_buffers.lock().map_err(|_| {
                                            anyhow::anyhow!("GLM fill-ahead free lock poisoned")
                                        })?;
                                        loop {
                                            if abort.load(std::sync::atomic::Ordering::Relaxed) {
                                                return Ok(());
                                            }
                                            if let Some(index) = free.pop() {
                                                break index;
                                            }
                                            free = free_cv.wait(free).map_err(|_| {
                                                anyhow::anyhow!("GLM fill-ahead free lock poisoned")
                                            })?;
                                        }
                                    };
                                    let ordinal = next_fill
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if ordinal >= fill_names.len() {
                                        free_buffers
                                            .lock()
                                            .map_err(|_| {
                                                anyhow::anyhow!("GLM fill-ahead free lock poisoned")
                                            })?
                                            .push(buffer_index);
                                        free_cv.notify_one();
                                        return Ok(());
                                    }
                                    let mut buffer =
                                        ring.buffer(buffer_index).lock().map_err(|_| {
                                            anyhow::anyhow!("GLM fill-ahead buffer lock poisoned")
                                        })?;
                                    if let Some(event) = buffer.event.take() {
                                        // Wait for H2D completion before refill; enqueue no GPU work.
                                        event.synchronize()?;
                                    }
                                    let scale_name = format!("{}_scale_inv", fill_names[ordinal]);
                                    let filled = fp8::staging::fill_prepared(
                                        weights,
                                        fill_names[ordinal],
                                        &scale_name,
                                        self.compute_dtype,
                                        &mut buffer,
                                    )
                                    .map(|meta| (buffer_index, meta));
                                    drop(buffer);
                                    ready_fills
                                        .lock()
                                        .map_err(|_| {
                                            anyhow::anyhow!("GLM fill-ahead ready lock poisoned")
                                        })?
                                        .insert(ordinal, filled);
                                    ready_cv.notify_all();
                                }
                            })();
                            // Keep the guard armed on failure; normal exits must not abort pending fills.
                            worker_guard.armed = result.is_err();
                            result
                        });
                    }
                    let mut ordinal = 0usize;
                    let mut take = |cached: &Option<Tensor>, name: &str| -> Result<Tensor> {
                        if let Some(tensor) = cached {
                            return Ok(tensor.clone());
                        }
                        let mine = ordinal;
                        ordinal += 1;
                        let (buffer_index, meta) = {
                            let mut ready = ready_fills.lock().map_err(|_| {
                                anyhow::anyhow!("GLM fill-ahead ready lock poisoned")
                            })?;
                            loop {
                                if let Some(filled) = ready.remove(&mine) {
                                    break filled;
                                }
                                if abort.load(std::sync::atomic::Ordering::Relaxed) {
                                    bail!("GLM fill-ahead aborted before miss {mine}");
                                }
                                // Fail on a missing fill so the scope can join.
                                let (guard, timeout) = ready_cv
                                    .wait_timeout(ready, std::time::Duration::from_secs(60))
                                    .map_err(|_| {
                                        anyhow::anyhow!("GLM fill-ahead ready lock poisoned")
                                    })?;
                                ready = guard;
                                if timeout.timed_out() {
                                    bail!("GLM fill-ahead timed out 60s waiting for miss {mine}");
                                }
                            }
                        }?;
                        let tensor = {
                            let mut buffer = ring.buffer(buffer_index).lock().map_err(|_| {
                                anyhow::anyhow!("GLM fill-ahead buffer lock poisoned")
                            })?;
                            let (tensor, drained) = pool.upload_prepared_on(
                                0,
                                device,
                                self.compute_dtype,
                                &mut buffer,
                                &meta,
                            )?;
                            buffer.event = Some(drained);
                            tensor
                        };
                        free_buffers
                            .lock()
                            .map_err(|_| anyhow::anyhow!("GLM fill-ahead free lock poisoned"))?
                            .push(buffer_index);
                        free_cv.notify_one();
                        cache.insert(name.to_owned(), tensor.clone());
                        Ok(tensor)
                    };
                    let walk = (|| -> Result<()> {
                        for expert in &device_work {
                            let gate = take(&expert.gate, &expert.gate_name)?;
                            let up = take(&expert.up, &expert.up_name)?;
                            let down = take(&expert.down, &expert.down_name)?;
                            self.wait_lane_ready(0)?;
                            let contribution = self.mlp_with_weights(
                                hidden,
                                &gate,
                                &up,
                                &down,
                                Some(expert.mixture as f64),
                            )?;
                            contributions.push((expert.expert, contribution));
                        }
                        Ok(())
                    })();
                    walk?;
                    true
                } else {
                    false
                };
                #[cfg(not(feature = "cuda"))]
                let ran_pipeline = false;
                if !ran_pipeline {
                    for expert in &device_work {
                        let gate = self.complete_cached_weight_on_lane(
                            cache,
                            &expert.gate_name,
                            expert.gate.clone(),
                            0,
                        )?;
                        let up = self.complete_cached_weight_on_lane(
                            cache,
                            &expert.up_name,
                            expert.up.clone(),
                            0,
                        )?;
                        let down = self.complete_cached_weight_on_lane(
                            cache,
                            &expert.down_name,
                            expert.down.clone(),
                            0,
                        )?;
                        self.wait_lane_ready(0)?;
                        let contribution = self.mlp_with_weights(
                            hidden,
                            &gate,
                            &up,
                            &down,
                            Some(expert.mixture as f64),
                        )?;
                        contributions.push((expert.expert, contribution));
                    }
                }
            } else {
                let device_queue = &device_work;
                let next = &next_device;
                let matvec = &matvec_lock;
                let mut consumers = Vec::with_capacity(device_lanes);
                for lane in 0..device_lanes {
                    consumers.push(scope.spawn(move || -> Result<Vec<(u32, Tensor)>> {
                        let mut evaluated = Vec::new();
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(expert) = device_queue.get(index) else {
                                break;
                            };
                            let gate = self.complete_cached_weight_on_lane(
                                cache,
                                &expert.gate_name,
                                expert.gate.clone(),
                                lane,
                            )?;
                            let up = self.complete_cached_weight_on_lane(
                                cache,
                                &expert.up_name,
                                expert.up.clone(),
                                lane,
                            )?;
                            let down = self.complete_cached_weight_on_lane(
                                cache,
                                &expert.down_name,
                                expert.down.clone(),
                                lane,
                            )?;
                            self.wait_lane_ready(lane)?;
                            let contribution = {
                                let _serial = matvec
                                    .lock()
                                    .map_err(|_| anyhow::anyhow!("GLM matvec lock poisoned"))?;
                                self.mlp_with_weights(
                                    hidden,
                                    &gate,
                                    &up,
                                    &down,
                                    Some(expert.mixture as f64),
                                )?
                            };
                            evaluated.push((expert.expert, contribution));
                        }
                        Ok(evaluated)
                    }));
                }
                for consumer in consumers {
                    let evaluated = consumer
                        .join()
                        .map_err(|_| anyhow::anyhow!("GLM device expert worker panicked"))??;
                    contributions.extend(evaluated);
                }
            }

            for worker in workers {
                let evaluated = worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("GLM host expert worker panicked"))??;
                for (expert, values) in evaluated {
                    contributions.push((
                        expert,
                        values
                            .to_dtype(self.compute_dtype)?
                            .to_device(&self.device)?,
                    ));
                }
            }
            Ok(())
        })?;
        contributions.sort_unstable_by_key(|(expert, _)| *expert);
        let mut output = Tensor::zeros(text.hidden_size, self.compute_dtype, &self.device)?;
        for (_, contribution) in contributions {
            output = output.add(&contribution)?;
        }
        let shared = self.mlp_projection(&format!("{prefix}.shared_experts"), hidden, None)?;
        output.add(&shared).map_err(Into::into)
    }

    fn mlp_projection(
        &self,
        prefix: &str,
        hidden: &Tensor,
        mixture: Option<f64>,
    ) -> Result<Tensor> {
        let gate_weight = self.load_linear_weight(&format!("{prefix}.gate_proj.weight"))?;
        let up_weight = self.load_linear_weight(&format!("{prefix}.up_proj.weight"))?;
        let down_weight = self.load_linear_weight(&format!("{prefix}.down_proj.weight"))?;
        self.mlp_with_weights(hidden, &gate_weight, &up_weight, &down_weight, mixture)
    }

    fn traced_linear(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.execution_policy.pinned_fp8_transfer
            && self
                .weight_pool
                .get()
                .and_then(Option::as_ref)
                .is_some_and(|p| p.tracing_enabled())
        {
            use candle_core::cuda_backend::{CudaStorageSlice, cudarc::driver::DevicePtr};
            let stream = self.device.as_cuda_device()?.cuda_stream();
            for tensor in [input, weight] {
                let (storage, _) = tensor.storage_and_layout();
                let candle_core::Storage::Cuda(storage) = &*storage else {
                    bail!("expected CUDA matrix");
                };
                macro_rules! ready {
                    ($data:expr) => {{
                        let (_, _read) = $data.device_ptr(&stream);
                    }};
                }
                match &storage.slice {
                    CudaStorageSlice::BF16(x) => ready!(x),
                    CudaStorageSlice::F16(x) => ready!(x),
                    CudaStorageSlice::F32(x) => ready!(x),
                    _ => bail!("unsupported matrix trace dtype"),
                }
            }
            let start = self.begin_expert_compute()?;
            let result = linear(input, weight)?;
            self.end_expert_compute(start)?;
            return Ok(result);
        }
        linear(input, weight)
    }

    fn mlp_with_weights(
        &self,
        hidden: &Tensor,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
        mixture: Option<f64>,
    ) -> Result<Tensor> {
        let input = hidden.unsqueeze(0)?;
        let gate = self.traced_linear(&input, gate_weight)?;
        let up = self.traced_linear(&input, up_weight)?;
        let activated = math::clamped_swiglu(&gate, &up, self.config.text_config.swiglu_limit)?;
        let output = self.traced_linear(&activated, down_weight)?.squeeze(0)?;
        match mixture {
            Some(weight) => output
                .to_dtype(DType::F32)?
                .affine(weight, 0.0)?
                .to_dtype(self.compute_dtype)
                .map_err(Into::into),
            None => Ok(output),
        }
    }

    /// Convert the configured host share to a miss count.
    /// Leave at least one miss on the device so its expert cache can warm.
    fn host_expert_budget(&self, misses: usize) -> usize {
        host_expert_budget(self.host_expert_share, misses)
    }

    fn complete_cached_weight(
        &self,
        cache: &ExpertCache,
        name: String,
        cached: Option<Tensor>,
    ) -> Result<Tensor> {
        self.complete_cached_weight_on_lane(cache, &name, cached, 0)
    }

    /// Use one consumer per upload lane, or a serial loop without a pinned pool.
    fn device_lanes(&self) -> usize {
        #[cfg(feature = "cuda")]
        if self.execution_policy.pinned_fp8_transfer
            && let Some(pool) = self.weight_pool.get().and_then(Option::as_ref)
        {
            return pool.lanes();
        }
        1
    }

    /// Fill-ahead worker count when pinned transfers and CUDA async allocation are enabled.
    #[cfg(feature = "cuda")]
    fn fill_ahead_workers(&self) -> usize {
        #[cfg(feature = "cuda")]
        if self.execution_policy.pinned_fp8_transfer
            && let Device::Cuda(device) = &self.device
            && device.cuda_stream().context().has_async_alloc()
        {
            return fill_ahead_count();
        }
        0
    }

    #[cfg(feature = "cuda")]
    fn ensure_weight_pool(&self) -> Result<()> {
        if self.weight_pool.get().is_none()
            && let Device::Cuda(device) = &self.device
        {
            let pool = if device.cuda_stream().context().has_async_alloc() {
                Some(fp8::cuda::WeightPool::new(device)?)
            } else {
                None
            };
            let _ = self.weight_pool.set(pool);
        }
        Ok(())
    }

    fn complete_cached_weight_on_lane(
        &self,
        cache: &ExpertCache,
        name: &str,
        cached: Option<Tensor>,
        lane: usize,
    ) -> Result<Tensor> {
        if let Some(tensor) = cached {
            return Ok(tensor);
        }
        let tensor = self.load_linear_weight_on_lane(name, lane)?;
        cache.insert(name.to_owned(), tensor.clone());
        Ok(tensor)
    }

    /// Order lane dequantization before compute-stream reads; non-pinned loads
    /// are already ordered.
    fn wait_lane_ready(&self, _lane: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.execution_policy.pinned_fp8_transfer
            && let (Device::Cuda(device), Some(pool)) = (
                &self.device,
                self.weight_pool.get().and_then(Option::as_ref),
            )
            && let Some(event) = pool.lane_ready(_lane)?
        {
            device.cuda_stream().wait(&event)?;
        }
        Ok(())
    }

    fn new_routing_trace_builder(
        &self,
        domain: &str,
        prompt_tokens: usize,
        max_routed_tokens: usize,
    ) -> Result<RoutingTraceBuilder> {
        let text = &self.config.text_config;
        let cache_entry_dtype = match self.compute_dtype {
            DType::BF16 => ExpertCacheEntryDtype::Bfloat16,
            DType::F32 => ExpertCacheEntryDtype::Float32,
            dtype => bail!("GLM routing trace does not support cache dtype {dtype:?}"),
        };
        let mut layers = Vec::new();
        for layer in 0..text.num_hidden_layers {
            if text.mlp_layer_types[layer] != MlpKind::Sparse {
                continue;
            }
            let mut expert_bytes = Vec::with_capacity(text.n_routed_experts);
            for expert in 0..text.n_routed_experts {
                let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
                let bytes = ["gate_proj", "up_proj", "down_proj"].into_iter().try_fold(
                    0u64,
                    |total, projection| {
                        let name = format!("{prefix}.{projection}.weight");
                        let metadata = self.weights.metadata(&name).with_context(|| {
                            format!("failed to size routed expert tensor {name}")
                        })?;
                        let elements =
                            metadata.shape.iter().try_fold(1u64, |count, &dimension| {
                                count
                                    .checked_mul(
                                        u64::try_from(dimension)
                                            .context("routed expert dimension exceeds u64")?,
                                    )
                                    .context("routed expert element count overflow")
                            })?;
                        let bytes = elements
                            .checked_mul(
                                u64::try_from(self.compute_dtype.size_in_bytes())
                                    .context("GLM compute dtype size exceeds u64")?,
                            )
                            .context("routed expert byte count overflow")?;
                        total
                            .checked_add(bytes)
                            .context("routed expert projection-byte sum overflow")
                    },
                )?;
                expert_bytes.push(bytes);
            }
            layers.push(RoutingTraceLayer {
                layer_index: u32::try_from(layer)
                    .context("routing-trace layer index exceeds u32")?,
                expert_bytes,
            });
        }
        RoutingTraceBuilder::new(
            domain,
            cache_entry_dtype,
            text.num_hidden_layers,
            text.n_routed_experts,
            text.num_experts_per_tok,
            prompt_tokens,
            max_routed_tokens,
            layers,
        )
    }

    fn load_tensor(&self, name: &str) -> Result<Tensor> {
        if let Some(tensor) = self.static_weights.get(name) {
            return Ok(tensor.clone());
        }
        self.weights
            .load(name, &self.device)
            .with_context(|| format!("failed to load GLM tensor {name}"))
    }

    fn load_linear_weight(&self, name: &str) -> Result<Tensor> {
        self.load_linear_weight_on_lane(name, 0)
    }

    /// Load through the selected lane's pinned staging area.
    fn load_linear_weight_on_lane(
        &self,
        name: &str,
        #[allow(unused)] lane: usize,
    ) -> Result<Tensor> {
        if let Some(tensor) = self.static_weights.get(name) {
            return Ok(tensor.clone());
        }
        let metadata = self.weights.metadata(name)?;
        let scale_name = format!("{name}_scale_inv");
        let scale = match metadata.dtype.as_str() {
            "F8_E4M3" => Some(scale_name.as_str()),
            "BF16" => None,
            dtype => bail!("GLM linear weight {name} has unsupported dtype {dtype}"),
        };
        if scale.is_some() && self.execution_policy.cpu_fp8_dequantization && self.device.is_cuda()
        {
            return Ok(fp8::load_linear_weight(
                &self.weights,
                name,
                scale,
                &Device::Cpu,
                self.compute_dtype,
            )?
            .to_device(&self.device)?);
        }
        #[cfg(feature = "cuda")]
        if let (Some(scale), Device::Cuda(device)) = (scale, &self.device) {
            self.ensure_weight_pool()?;
            if let Some(pool) = self.weight_pool.get().and_then(Option::as_ref) {
                if self.execution_policy.pinned_fp8_transfer {
                    return pool.load_pinned_on(
                        &self.weights,
                        name,
                        scale,
                        device,
                        self.compute_dtype,
                        lane,
                    );
                }
                return fp8::cuda::load_weight(
                    &self.weights,
                    name,
                    scale,
                    &self.device,
                    self.compute_dtype,
                    pool,
                );
            }
        }
        ensure!(
            !self.execution_policy.pinned_fp8_transfer || scale.is_none(),
            "pinned FP8 transfer requires CUDA asynchronous allocation support"
        );
        fp8::load_linear_weight(&self.weights, name, scale, &self.device, self.compute_dtype)
    }

    fn preload_static_weights(&mut self, progress: bool) -> Result<()> {
        let specs = static_weight_specs(&self.config.text_config);
        for (index, (name, linear_weight)) in specs.iter().enumerate() {
            let tensor = if *linear_weight {
                self.load_linear_weight(name)?
            } else {
                self.load_tensor(name)?
            };
            ensure!(
                self.static_weights.insert(name.clone(), tensor).is_none(),
                "duplicate GLM static residency tensor {name}"
            );
            if progress && ((index + 1).is_multiple_of(100) || index + 1 == specs.len()) {
                eprintln!(
                    "GLM static residency: {}/{} tensors, {:.2} GiB",
                    index + 1,
                    specs.len(),
                    self.resident_static_bytes() as f64 / 1024.0 / 1024.0 / 1024.0
                );
            }
        }
        Ok(())
    }

    fn validate_inventory(&self) -> Result<()> {
        for name in [EMBEDDING_WEIGHT, FINAL_NORM_WEIGHT] {
            self.require_tensor(name)?;
        }
        self.require_linear_weight(LM_HEAD_WEIGHT)?;
        let text = &self.config.text_config;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer}");
            for suffix in [
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ] {
                self.require_tensor(&format!("{prefix}.{suffix}"))?;
            }
            match text.layer_types[layer] {
                AttentionKind::LinearAttention => {
                    for suffix in [
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "f_a_proj.weight",
                        "f_b_proj.weight",
                        "b_proj.weight",
                        "g_a_proj.weight",
                        "g_b_proj.weight",
                        "o_proj.weight",
                    ] {
                        self.require_linear_weight(&format!("{prefix}.self_attn.{suffix}"))?;
                    }
                    for suffix in [
                        "q_conv1d.weight",
                        "k_conv1d.weight",
                        "v_conv1d.weight",
                        "dt_bias",
                        "A_log",
                        "o_norm.weight",
                    ] {
                        self.require_tensor(&format!("{prefix}.self_attn.{suffix}"))?;
                    }
                }
                AttentionKind::DeepseekSparseAttention => {
                    for suffix in [
                        "q_a_proj.weight",
                        "q_b_proj.weight",
                        "kv_a_proj_with_mqa.weight",
                        "kv_b_proj.weight",
                        "o_proj.weight",
                    ] {
                        self.require_linear_weight(&format!("{prefix}.self_attn.{suffix}"))?;
                    }
                    for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight"] {
                        self.require_tensor(&format!("{prefix}.self_attn.{suffix}"))?;
                    }
                }
            }
            match text.mlp_layer_types[layer] {
                MlpKind::Dense => self.require_mlp(&format!("{prefix}.mlp"))?,
                MlpKind::Sparse => {
                    self.require_tensor(&format!("{prefix}.mlp.gate.weight"))?;
                    self.require_tensor(&format!("{prefix}.mlp.gate.e_score_correction_bias"))?;
                    self.require_mlp(&format!("{prefix}.mlp.shared_experts"))?;
                    for expert in 0..text.n_routed_experts {
                        self.require_mlp(&format!("{prefix}.mlp.experts.{expert}"))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn require_mlp(&self, prefix: &str) -> Result<()> {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let weight = format!("{prefix}.{projection}.weight");
            self.require_linear_weight(&weight)?;
        }
        Ok(())
    }

    fn require_linear_weight(&self, name: &str) -> Result<()> {
        self.require_tensor(name)?;
        let metadata = self
            .weights
            .metadata(name)
            .with_context(|| format!("failed to inspect GLM linear weight {name}"))?;
        ensure!(
            metadata.shape.len() == 2 && metadata.shape.iter().all(|&dimension| dimension > 0),
            "GLM linear weight {name} must be a non-empty rank-2 matrix, found shape {:?}",
            metadata.shape
        );

        let scale_name = format!("{name}_scale_inv");
        match metadata.dtype.as_str() {
            "BF16" => ensure!(
                !self.weights.contains(&scale_name),
                "BF16 GLM linear weight {name} unexpectedly has inverse scale tensor {scale_name}"
            ),
            "F8_E4M3" => {
                self.require_tensor(&scale_name)?;
                let scale = self
                    .weights
                    .metadata(&scale_name)
                    .with_context(|| format!("failed to inspect GLM inverse scale {scale_name}"))?;
                ensure!(
                    scale.dtype == "F32",
                    "GLM inverse scale {scale_name} must have dtype F32, found {}",
                    scale.dtype
                );
                let expected = [
                    metadata.shape[0].div_ceil(fp8::FP8_BLOCK_SIZE),
                    metadata.shape[1].div_ceil(fp8::FP8_BLOCK_SIZE),
                ];
                ensure!(
                    scale.shape == expected,
                    "GLM inverse scale {scale_name} has shape {:?}, expected {:?} for weight shape {:?}",
                    scale.shape,
                    expected,
                    metadata.shape
                );
            }
            dtype => {
                bail!("GLM linear weight {name} must have dtype BF16 or F8_E4M3, found {dtype}")
            }
        }
        Ok(())
    }

    fn require_tensor(&self, name: &str) -> Result<()> {
        ensure!(
            self.weights.contains(name),
            "GLM checkpoint is missing required tensor {name}"
        );
        Ok(())
    }
}

fn plan_cache_readmission(
    policy: &GlmExecutionPolicy,
    admission: GlmAdmissionModel,
    completed_tokens: usize,
    phase: RoutingTracePhase,
    available_memory_bytes: u64,
    cache: ExpertCacheStats,
) -> Result<CacheReadmissionPlan> {
    policy.validate()?;
    ensure!(
        policy.expert_cache.readmission_interval_tokens.is_some(),
        "GLM cache re-admission planning requires an adaptive policy"
    );
    ensure!(
        completed_tokens > 0
            && u32::try_from(completed_tokens)
                .ok()
                .is_some_and(|tokens| tokens <= policy.dsa_context_bound_tokens),
        "GLM cache re-admission token boundary is outside the DSA context bound"
    );
    ensure!(
        cache.bytes <= cache.max_bytes,
        "GLM cache occupancy exceeds its current bound before re-admission"
    );
    let previous_bound_bytes =
        u64::try_from(cache.max_bytes).context("GLM current cache bound exceeds u64")?;
    let realized_before_bytes =
        u64::try_from(cache.bytes).context("GLM cache occupancy exceeds u64")?;
    ensure!(
        previous_bound_bytes >= policy.expert_cache.minimum_bound_bytes
            && previous_bound_bytes <= policy.expert_cache.maximum_bound_bytes,
        "GLM current cache bound is outside its execution policy"
    );
    let expected_maximum_dsa = admission
        .dsa_cache_bytes_per_token
        .checked_mul(policy.dsa_context_bound_tokens as usize)
        .context("GLM DSA admission-model byte count overflow")?;
    ensure!(
        expected_maximum_dsa == admission.maximum_dsa_cache_bytes,
        "GLM DSA admission model disagrees with the execution policy context bound"
    );
    ensure!(
        u64::try_from(admission.safety_bytes).context("GLM admission safety exceeds u64")?
            == policy.admission_safety_bytes,
        "GLM runtime admission safety disagrees with the execution policy"
    );
    let current_dsa = admission
        .dsa_cache_bytes_per_token
        .checked_mul(completed_tokens)
        .context("GLM current DSA cache byte count overflow")?;
    let remaining_dsa = admission
        .maximum_dsa_cache_bytes
        .checked_sub(current_dsa)
        .context("GLM remaining DSA cache byte count underflow")?;
    let pending_head = if phase == RoutingTracePhase::Prefill {
        admission.pending_lm_head_bytes
    } else {
        0
    };
    let required_future_headroom_bytes = remaining_dsa
        .checked_add(admission.streamed_transient_bytes)
        .and_then(|bytes| bytes.checked_add(pending_head))
        .and_then(|bytes| bytes.checked_add(admission.live_expert_bytes))
        .and_then(|bytes| bytes.checked_add(admission.weight_load_staging_bytes))
        .and_then(|bytes| bytes.checked_add(admission.safety_bytes))
        .context("GLM cache re-admission headroom overflow")?;
    let required_future_headroom_bytes = u64::try_from(required_future_headroom_bytes)
        .context("GLM cache re-admission headroom exceeds u64")?;
    let reclaimable_capacity_bytes = available_memory_bytes
        .checked_add(realized_before_bytes)
        .context("GLM cache reclaimable capacity overflow")?;
    let admissible_cache_bound_bytes =
        reclaimable_capacity_bytes.saturating_sub(required_future_headroom_bytes);
    ensure!(
        admissible_cache_bound_bytes >= policy.expert_cache.minimum_bound_bytes,
        "GLM cache re-admission can admit only {:.2} MiB after token {completed_tokens}, below the policy minimum {:.2} MiB; no cache bound was changed",
        admissible_cache_bound_bytes as f64 / 1024.0 / 1024.0,
        policy.expert_cache.minimum_bound_bytes as f64 / 1024.0 / 1024.0,
    );
    let new_bound_bytes = policy
        .expert_cache
        .maximum_bound_bytes
        .min(admissible_cache_bound_bytes);
    let decision = match new_bound_bytes.cmp(&previous_bound_bytes) {
        Ordering::Less => GlmCacheAdmissionDecision::Shrunk,
        Ordering::Equal => GlmCacheAdmissionDecision::Unchanged,
        Ordering::Greater => GlmCacheAdmissionDecision::Grown,
    };
    Ok(CacheReadmissionPlan {
        available_memory_bytes,
        reclaimable_capacity_bytes,
        required_future_headroom_bytes,
        admissible_cache_bound_bytes,
        previous_bound_bytes,
        new_bound_bytes,
        realized_before_bytes,
        decision,
    })
}

pub(crate) fn static_weight_specs(text: &super::config::GlmTextConfig) -> Vec<(String, bool)> {
    let mut specs = vec![
        (FINAL_NORM_WEIGHT.to_owned(), false),
        (LM_HEAD_WEIGHT.to_owned(), true),
    ];
    for layer in 0..text.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");
        for suffix in [
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            specs.push((format!("{prefix}.{suffix}"), false));
        }
        match text.layer_types[layer] {
            AttentionKind::LinearAttention => {
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "f_a_proj.weight",
                    "f_b_proj.weight",
                    "b_proj.weight",
                    "g_a_proj.weight",
                    "g_b_proj.weight",
                    "o_proj.weight",
                ] {
                    specs.push((format!("{prefix}.self_attn.{suffix}"), true));
                }
                for suffix in [
                    "q_conv1d.weight",
                    "k_conv1d.weight",
                    "v_conv1d.weight",
                    "dt_bias",
                    "A_log",
                    "o_norm.weight",
                ] {
                    specs.push((format!("{prefix}.self_attn.{suffix}"), false));
                }
            }
            AttentionKind::DeepseekSparseAttention => {
                for suffix in [
                    "q_a_proj.weight",
                    "q_b_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                    "kv_b_proj.weight",
                    "o_proj.weight",
                ] {
                    specs.push((format!("{prefix}.self_attn.{suffix}"), true));
                }
                for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight"] {
                    specs.push((format!("{prefix}.self_attn.{suffix}"), false));
                }
            }
        }
        match text.mlp_layer_types[layer] {
            MlpKind::Dense => {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    specs.push((format!("{prefix}.mlp.{projection}.weight"), true));
                }
            }
            MlpKind::Sparse => {
                specs.push((format!("{prefix}.mlp.gate.weight"), false));
                specs.push((format!("{prefix}.mlp.gate.e_score_correction_bias"), false));
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    specs.push((
                        format!("{prefix}.mlp.shared_experts.{projection}.weight"),
                        true,
                    ));
                }
            }
        }
    }
    specs
}

pub(crate) fn streamed_static_weight_groups(
    text: &super::config::GlmTextConfig,
) -> Vec<Vec<String>> {
    let mut groups = vec![vec![FINAL_NORM_WEIGHT.to_owned()]];
    for layer in 0..text.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");
        groups.push(
            ["hc_attn_fn", "hc_attn_base", "hc_attn_scale"]
                .map(|suffix| format!("{prefix}.{suffix}"))
                .to_vec(),
        );
        groups.push(vec![format!("{prefix}.input_layernorm.weight")]);
        match text.layer_types[layer] {
            AttentionKind::LinearAttention => {
                for projection in ["q", "k", "v"] {
                    groups.push(vec![
                        format!("{prefix}.self_attn.{projection}_proj.weight"),
                        format!("{prefix}.self_attn.{projection}_conv1d.weight"),
                    ]);
                }
                groups.push(
                    [
                        "f_a_proj.weight",
                        "f_b_proj.weight",
                        "dt_bias",
                        "A_log",
                        "b_proj.weight",
                        "g_a_proj.weight",
                        "g_b_proj.weight",
                        "o_norm.weight",
                        "o_proj.weight",
                    ]
                    .map(|suffix| format!("{prefix}.self_attn.{suffix}"))
                    .to_vec(),
                );
            }
            AttentionKind::DeepseekSparseAttention => groups.push(
                [
                    "q_a_proj.weight",
                    "q_a_layernorm.weight",
                    "q_b_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                    "kv_a_layernorm.weight",
                    "kv_b_proj.weight",
                    "o_proj.weight",
                ]
                .map(|suffix| format!("{prefix}.self_attn.{suffix}"))
                .to_vec(),
            ),
        }
        groups.push(
            ["hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"]
                .map(|suffix| format!("{prefix}.{suffix}"))
                .to_vec(),
        );
        groups.push(vec![format!("{prefix}.post_attention_layernorm.weight")]);
        match text.mlp_layer_types[layer] {
            MlpKind::Dense => groups.push(
                ["gate_proj", "up_proj", "down_proj"]
                    .map(|projection| format!("{prefix}.mlp.{projection}.weight"))
                    .to_vec(),
            ),
            MlpKind::Sparse => {
                let mut group = vec![
                    format!("{prefix}.mlp.gate.weight"),
                    format!("{prefix}.mlp.gate.e_score_correction_bias"),
                ];
                group.extend(
                    ["gate_proj", "up_proj", "down_proj"].map(|projection| {
                        format!("{prefix}.mlp.shared_experts.{projection}.weight")
                    }),
                );
                groups.push(group);
            }
        }
    }
    groups
}

struct DecoderState {
    tokens: usize,
    layers: Vec<LayerCache>,
}

impl DecoderState {
    fn new(config: &GlmConfig, dtype: DType, device: &Device) -> Result<Self> {
        let layers = config
            .text_config
            .layer_types
            .iter()
            .map(|kind| LayerCache::new(*kind, &config.text_config, dtype, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { tokens: 0, layers })
    }
}

enum LayerCache {
    Kda(KdaCache),
    Dsa(DsaCache),
}

impl LayerCache {
    fn new(
        kind: AttentionKind,
        text: &crate::config::GlmTextConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let qkv_dim = text.linear_qkv_dim()?;
        let convolution = || Tensor::zeros((qkv_dim, text.linear_conv_kernel_dim), dtype, device);
        Ok(match kind {
            AttentionKind::LinearAttention => Self::Kda(KdaCache {
                query_conv: convolution()?,
                key_conv: convolution()?,
                value_conv: convolution()?,
                recurrent: Tensor::zeros(
                    (
                        text.linear_num_heads,
                        text.linear_head_dim,
                        text.linear_head_dim,
                    ),
                    DType::F32,
                    device,
                )?,
            }),
            AttentionKind::DeepseekSparseAttention => Self::Dsa(DsaCache {
                keys: None,
                values: None,
            }),
        })
    }
}

struct KdaCache {
    query_conv: Tensor,
    key_conv: Tensor,
    value_conv: Tensor,
    recurrent: Tensor,
}

struct DsaCache {
    keys: Option<Tensor>,
    values: Option<Tensor>,
}

struct PendingExpert {
    expert: u32,
    mixture: f32,
    gate_name: String,
    up_name: String,
    down_name: String,
    gate: Option<Tensor>,
    up: Option<Tensor>,
    down: Option<Tensor>,
}

impl PendingExpert {
    fn is_complete(&self) -> bool {
        self.gate.is_some() && self.up.is_some() && self.down.is_some()
    }
}

fn render_chat_prompt(prompt: &str, reasoning_effort: &str) -> Result<String> {
    let label = match reasoning_effort {
        "low" => "Low",
        "high" => "High",
        "max" => "Max",
        value => bail!("GLM reasoning effort must be low, high, or max, got {value:?}"),
    };
    Ok(format!(
        "[gMASK]<sop><|system|>Reasoning Effort: {label}<|user|>{prompt}<|assistant|><think>"
    ))
}

fn linear(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let input_width = *input
        .dims()
        .last()
        .context("linear input must have at least one dimension")?;
    let (_, weight_width) = weight
        .dims2()
        .context("linear weight must have shape [output, input]")?;
    ensure!(
        input_width == weight_width,
        "linear input width {input_width} differs from weight width {weight_width}"
    );
    input.matmul(&weight.t()?).map_err(Into::into)
}

fn argmax_token(logits: &Tensor) -> Result<u32> {
    let values = logits
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "GLM logits contain a non-finite value"
    );
    ensure!(!values.is_empty(), "GLM logits are empty");
    let mut index = 0usize;
    for (candidate, value) in values.iter().enumerate().skip(1) {
        if *value > values[index] {
            index = candidate;
        }
    }
    u32::try_from(index).context("GLM token id exceeds u32")
}

fn sample_token(logits: &Tensor, temperature: f64, top_p: f64, rng: &mut StdRng) -> Result<u32> {
    if temperature == 0.0 {
        return argmax_token(logits);
    }
    ensure!(
        temperature.is_finite() && temperature > 0.0 && temperature.recip().is_finite(),
        "GLM sampling temperature must be positive and safely invertible"
    );
    let values = logits
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    ensure!(!values.is_empty(), "GLM logits are empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "GLM logits contain a non-finite value"
    );
    let inverse_temperature = 1.0 / temperature;
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    let maximum = values[order[0]] as f64 * inverse_temperature;
    let mut probabilities = order
        .iter()
        .map(|&index| (values[index] as f64 * inverse_temperature - maximum).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    ensure!(
        total.is_finite() && total > 0.0,
        "GLM softmax normalization is invalid"
    );
    for probability in &mut probabilities {
        *probability /= total;
    }
    let mut retained = 0usize;
    let mut cumulative = 0.0;
    for probability in &probabilities {
        cumulative += *probability;
        retained += 1;
        if cumulative >= top_p {
            break;
        }
    }
    let retained_total = probabilities[..retained].iter().sum::<f64>();
    let target = rng.random::<f64>() * retained_total;
    let mut cumulative = 0.0;
    for (rank, probability) in probabilities[..retained].iter().enumerate() {
        cumulative += *probability;
        if target <= cumulative {
            return u32::try_from(order[rank]).context("GLM token id exceeds u32");
        }
    }
    u32::try_from(order[retained - 1]).context("GLM token id exceeds u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::GlmTextConfig, execution_policy::GLM_ADMISSION_SAFETY_BYTES};
    use ff_core::weights::{CachePolicy, WeightSource};

    /// The split turns a measured fraction into a count, and always leaves one
    /// miss to the device path. A decision that sent every miss to the host
    /// would admit nothing to the expert cache, so the next decision would face
    /// the same misses forever.
    /// The execution policy records whole per-mille. Two shares that record the
    /// same number therefore have to split a miss set the same way, or the
    /// record would not describe the run: 0.2499 and 0.2501 both record 250,
    /// and un-quantised they straddle `round(2 * share)`.
    #[test]
    fn shares_that_record_alike_split_alike() {
        let options = |share: f64| {
            StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), Device::Cpu)
                .with_host_expert_share(share)
        };
        for (low, high, per_mille) in [(0.2499, 0.2501, 250u32), (0.4996, 0.5004, 500)] {
            assert_eq!(
                host_expert_share_per_mille(low),
                per_mille,
                "{low} should record {per_mille}"
            );
            assert_eq!(host_expert_share_per_mille(high), per_mille);
            for misses in 0..8 {
                assert_eq!(
                    options(low).host_expert_budget_for(misses),
                    options(high).host_expert_budget_for(misses),
                    "{low} and {high} both record {per_mille} but split {misses} misses apart"
                );
            }
        }
    }

    /// A share is only ever executed at a value the policy can hold.
    #[test]
    fn the_executed_share_is_the_recorded_one() {
        for share in [0.0, 0.2499, 0.2501, 0.484, 0.9999, 1.0, -1.0, f64::NAN] {
            let executed =
                StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), Device::Cpu)
                    .with_host_expert_share(share)
                    .host_expert_share;
            assert_eq!(
                executed,
                f64::from(host_expert_share_per_mille(share)) / 1000.0,
                "executed share for {share} is not the recorded one"
            );
            assert!((0.0..=1.0).contains(&executed));
        }
    }

    #[test]
    fn the_host_budget_follows_the_share_and_never_takes_every_miss() {
        let budget = |share: f64, misses: usize| {
            StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), Device::Cpu)
                .with_host_expert_share(share)
                .host_expert_budget_for(misses)
        };

        // Measured on this host: B_P/B_H of 0.52 leaves the host 48%.
        assert_eq!(budget(0.484, 8), 4);
        assert_eq!(budget(0.484, 1), 0);
        assert_eq!(budget(0.484, 0), 0);

        // No measurement, no scheduling.
        assert_eq!(budget(0.0, 8), 0);

        // Even asked for everything, one fill survives.
        assert_eq!(budget(1.0, 8), 7);
        assert_eq!(budget(1.0, 2), 1);

        // A share outside a fraction is not a share.
        assert_eq!(budget(f64::NAN, 8), 0);
        assert_eq!(budget(-1.0, 8), 0);
        assert_eq!(budget(4.0, 8), 7);
    }

    #[test]
    fn chat_prompt_matches_the_text_only_template_prefix() {
        assert_eq!(
            render_chat_prompt("hello", "low").unwrap(),
            "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>hello<|assistant|><think>"
        );
    }

    #[test]
    fn chat_prompt_rejects_unknown_reasoning_effort() {
        let error = render_chat_prompt("hello", "medium").unwrap_err();
        assert!(error.to_string().contains("low, high, or max"));
    }

    #[test]
    fn streamed_admission_counts_dense_projection_group_and_official_peak() {
        let text = GlmTextConfig::tiny();
        let groups = streamed_static_weight_groups(&text);
        let mut bytes = groups
            .iter()
            .flatten()
            .map(|name| (name.clone(), 1usize))
            .collect::<BTreeMap<_, _>>();
        let dense = [
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
        ];
        for name in dense {
            bytes.insert(name.to_owned(), 96 * 1024 * 1024);
        }
        bytes.insert(
            "model.language_model.layers.3.self_attn.o_proj.weight".to_owned(),
            128 * 1024 * 1024,
        );
        let old_single_tensor_peak = bytes.values().copied().max().unwrap();
        let grouped_peak = groups
            .iter()
            .map(|group| group.iter().map(|name| bytes[name]).sum::<usize>())
            .max()
            .unwrap();
        assert_eq!(old_single_tensor_peak, 128 * 1024 * 1024);
        assert_eq!(grouped_peak, 288 * 1024 * 1024);
        assert_eq!(grouped_peak - old_single_tensor_peak, 160 * 1024 * 1024);

        let lm_head = 154_880usize * 4_096 * 2;
        let dense_group = 3usize * 4_096 * 12_288 * 2;
        let kda_state = (64usize * 128 * 128 * 4 + 3 * (64 * 128) * 4 * 2) * 34;
        let dsa_cache = 2_048usize * 11 * 64 * (256 + 256) * 2;
        let live_expert = 4_096usize * 2_048 * 5 * 2;
        let safety = 1usize << 30;
        let required = lm_head + dense_group + kda_state + dsa_cache + live_expert + safety;
        assert_eq!(required, 4_354_080_768);
        assert_eq!(required as f64 / 1024f64.powi(3), 4.0550537109375);
    }

    #[test]
    fn current_residency_readmission_plans_shrink_grow_and_failure_without_double_counting() {
        let policy = GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            false,
            8,
            ExpertCacheLayout::SharedPool,
            ExpertCacheReplacementPolicy::Lfu,
            100,
            20,
            true,
        )
        .unwrap();
        let safety = GLM_ADMISSION_SAFETY_BYTES as usize;
        let admission = GlmAdmissionModel {
            weight_load_staging_bytes: 0,
            maximum_dsa_cache_bytes: 80,
            dsa_cache_bytes_per_token: 10,
            streamed_transient_bytes: 5,
            pending_lm_head_bytes: 7,
            live_expert_bytes: 11,
            safety_bytes: safety,
        };
        let cache = ExpertCacheStats {
            bytes: 60,
            entries: 3,
            max_bytes: 100,
            hits: 0,
            misses: 0,
            evictions: 0,
        };
        let shrink = plan_cache_readmission(
            &policy,
            admission,
            2,
            RoutingTracePhase::Prefill,
            GLM_ADMISSION_SAFETY_BYTES + 73,
            cache,
        )
        .unwrap();
        assert_eq!(
            shrink.reclaimable_capacity_bytes,
            GLM_ADMISSION_SAFETY_BYTES + 133
        );
        assert_eq!(
            shrink.required_future_headroom_bytes,
            GLM_ADMISSION_SAFETY_BYTES + 83
        );
        assert_eq!(shrink.admissible_cache_bound_bytes, 50);
        assert_eq!(shrink.new_bound_bytes, 50);
        assert_eq!(shrink.decision, GlmCacheAdmissionDecision::Shrunk);
        let with_staging = plan_cache_readmission(
            &policy,
            GlmAdmissionModel {
                weight_load_staging_bytes: 13,
                ..admission
            },
            2,
            RoutingTracePhase::Prefill,
            GLM_ADMISSION_SAFETY_BYTES + 73,
            cache,
        )
        .unwrap();
        assert_eq!(
            with_staging.required_future_headroom_bytes,
            shrink.required_future_headroom_bytes + 13
        );
        assert_eq!(with_staging.new_bound_bytes, shrink.new_bound_bytes - 13);

        let grown_cache = ExpertCacheStats {
            bytes: 40,
            max_bytes: 50,
            ..cache
        };
        let grow = plan_cache_readmission(
            &policy,
            admission,
            3,
            RoutingTracePhase::Decode,
            GLM_ADMISSION_SAFETY_BYTES + 146,
            grown_cache,
        )
        .unwrap();
        assert_eq!(grow.admissible_cache_bound_bytes, 120);
        assert_eq!(grow.new_bound_bytes, 100);
        assert_eq!(grow.decision, GlmCacheAdmissionDecision::Grown);

        let error =
            plan_cache_readmission(&policy, admission, 2, RoutingTracePhase::Prefill, 0, cache)
                .unwrap_err();
        assert!(error.to_string().contains("no cache bound was changed"));
    }

    #[test]
    fn argmax_is_deterministic() {
        let logits = Tensor::new(&[-1f32, 3., 2.], &Device::Cpu).unwrap();
        assert_eq!(argmax_token(&logits).unwrap(), 1);
    }

    #[test]
    fn argmax_breaks_ties_toward_the_first_maximum_like_torch() {
        for (values, expected) in [
            (vec![3f32, 3., 2.], 0u32),
            (vec![2f32, 3., 3.], 1),
            (vec![1f32, 1., 1.], 0),
            (vec![-0f32, 0., -1.], 0),
            (vec![f32::MIN, f32::MIN], 0),
        ] {
            let logits = Tensor::new(values.as_slice(), &Device::Cpu).unwrap();
            assert_eq!(
                argmax_token(&logits).unwrap(),
                expected,
                "tie handling for {values:?}"
            );
        }
    }

    #[test]
    fn argmax_rejects_empty_and_non_finite_logits() {
        let empty = Tensor::from_vec(Vec::<f32>::new(), (0,), &Device::Cpu).unwrap();
        assert!(argmax_token(&empty).is_err());
        let nan = Tensor::new(&[1f32, f32::NAN], &Device::Cpu).unwrap();
        assert!(argmax_token(&nan).is_err());
    }

    #[test]
    fn zero_temperature_sampling_is_greedy() {
        let logits = Tensor::new(&[-1f32, 3., 2.], &Device::Cpu).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(sample_token(&logits, 0.0, 0.95, &mut rng).unwrap(), 1);
    }

    #[test]
    fn positive_temperature_must_not_underflow_into_greedy_sampling() {
        let logits = Tensor::new(&[-1f32, 3., 2.], &Device::Cpu).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let error = sample_token(&logits, f64::from_bits(1), 0.95, &mut rng).unwrap_err();
        assert!(error.to_string().contains("safely invertible"));
    }

    #[test]
    fn greedy_sampling_rejects_non_finite_logits() {
        let logits = Tensor::new(&[f32::NAN, 1.], &Device::Cpu).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let error = sample_token(&logits, 0.0, 0.95, &mut rng).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }
    use super::super::test_support::*;
    use candle_core::DType;
    use candle_core::Device;
    use candle_core::Tensor;
    use candle_core::safetensors;

    #[test]
    fn generates_tokens_through_tiny_cpu_kda_mla_dense_moe_checkpoint() {
        let checkpoint = tiny_checkpoint();
        let model = StreamedGlm::open(checkpoint.path(), tiny_options()).unwrap();
        assert_eq!(model.config().text_config.num_hidden_layers, 2);
        assert!(model.resident_static_bytes() > 0);

        let generation = model
            .generate(
                "exercise every tiny GLM path",
                &GlmGenerationOptions {
                    max_new_tokens: 2,
                    max_context_tokens: 8,
                    reasoning_effort: "low".to_owned(),
                    temperature: 1.0,
                    top_p: 0.9,
                    seed: 7,
                    progress: false,
                },
            )
            .unwrap();

        assert_eq!(generation.prompt_tokens, 1);
        assert_eq!(generation.generated_token_ids, [TARGET_TOKEN, TARGET_TOKEN]);
        assert_eq!(generation.text, "winner winner");
        assert_eq!(generation.token_elapsed.len(), 2);
        assert!(generation.prefill_elapsed.as_secs_f64().is_finite());
        assert!(generation.decode_elapsed.as_secs_f64().is_finite());
        assert!(
            generation
                .token_elapsed
                .iter()
                .all(|elapsed| elapsed.as_secs_f64().is_finite())
        );
        assert!(
            generation
                .generated_token_ids
                .iter()
                .all(|&token| token < VOCAB as u32)
        );

        let access = model.access_stats();
        assert_eq!(access.device_row_materializations, 2);
        assert!(access.device_tensor_materializations > 0);
        let expert_cache = model.expert_cache_stats();
        assert!(expert_cache.hits > 0);
        assert!(expert_cache.misses > 0);
        assert!(expert_cache.bytes <= expert_cache.max_bytes);
    }
    #[test]
    fn cache_topologies_replacement_policies_and_adaptive_safe_points_preserve_output() {
        let checkpoint = tiny_checkpoint();
        let generation_options = GlmGenerationOptions {
            max_new_tokens: 2,
            max_context_tokens: 8,
            reasoning_effort: "low".to_owned(),
            temperature: 0.0,
            top_p: 0.95,
            seed: 7,
            progress: false,
        };
        let baseline = StreamedGlm::open(checkpoint.path(), tiny_options())
            .unwrap()
            .generate("exercise every tiny GLM path", &generation_options)
            .unwrap();

        for layout in [
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheLayout::SharedPool,
        ] {
            for replacement in [
                ExpertCacheReplacementPolicy::Lru,
                ExpertCacheReplacementPolicy::Lfu,
            ] {
                let model = StreamedGlm::open(
                    checkpoint.path(),
                    tiny_options()
                        .with_expert_cache_policy(layout, replacement)
                        .with_adaptive_expert_cache(0),
                )
                .unwrap();
                let generated = model
                    .generate("exercise every tiny GLM path", &generation_options)
                    .unwrap();
                assert_eq!(generated.generated_token_ids, baseline.generated_token_ids);
                assert_eq!(generated.text, baseline.text);
                assert_eq!(
                    generated.execution_manifest.readmissions.len(),
                    generated.execution_manifest.generated_tokens as usize
                );
                assert_eq!(
                    generated.execution_manifest.policy.expert_cache.layout,
                    layout
                );
                assert_eq!(
                    generated.execution_manifest.policy.expert_cache.replacement,
                    replacement
                );
                generated.execution_manifest.validate().unwrap();
            }
        }
    }
    #[test]
    fn parity_capture_uses_the_generation_prefill_and_retains_prompt_ids() {
        let checkpoint = tiny_checkpoint();
        let model = StreamedGlm::open(checkpoint.path(), tiny_options()).unwrap();
        let capture = model
            .capture_first_next_token_parity("exercise every tiny GLM path", "low", 8, false)
            .unwrap();
        assert_eq!(capture.prompt_token_ids, [0]);
        assert_eq!(capture.final_hidden_state.dtype(), DType::F32);
        assert_eq!(capture.final_hidden_state.dims(), [HIDDEN]);
        assert_eq!(capture.next_token_logits.dtype(), DType::F32);
        assert_eq!(capture.next_token_logits.dims(), [VOCAB]);
        assert_eq!(capture.next_token_id, TARGET_TOKEN);

        let generated = model
            .generate(
                "exercise every tiny GLM path",
                &GlmGenerationOptions {
                    max_new_tokens: 1,
                    max_context_tokens: 8,
                    reasoning_effort: "low".to_owned(),
                    temperature: 0.0,
                    top_p: 1.0,
                    seed: 0,
                    progress: false,
                },
            )
            .unwrap();
        assert_eq!(generated.generated_token_ids, [capture.next_token_id]);
    }
    #[test]
    fn raw_tensor_cache_granularity_preserves_glm_prefill() {
        use ff_core::weights::CacheGranularity;
        let checkpoint = tiny_checkpoint();
        let open = |granularity| {
            StreamedGlm::open(
                checkpoint.path(),
                StreamedGlmOptions::new(
                    WeightSource::Mmap,
                    CachePolicy::new(1)
                        .with_max_bytes(128)
                        .with_granularity(granularity),
                    Device::Cpu,
                )
                .with_resident_static(true)
                .with_expert_cache_bytes(1024),
            )
            .unwrap()
        };
        let shard = open(CacheGranularity::Shard);
        let tensor = open(CacheGranularity::Tensor);
        let capture = |model: &StreamedGlm| {
            model
                .capture_first_next_token_parity("exercise every tiny GLM path", "low", 8, false)
                .unwrap()
        };
        let expected = capture(&shard);
        let actual = capture(&tensor);
        assert_eq!(actual.prompt_token_ids, expected.prompt_token_ids);
        assert_eq!(actual.next_token_id, expected.next_token_id);
        assert_eq!(
            actual.final_hidden_state.to_vec1::<f32>().unwrap(),
            expected.final_hidden_state.to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            actual.next_token_logits.to_vec1::<f32>().unwrap(),
            expected.next_token_logits.to_vec1::<f32>().unwrap()
        );
        assert!(shard.cache_stats().tensor_retention.is_none());
        assert!(tensor.cache_stats().tensor_retention.is_some());
        assert!(tensor.cache_stats().evictions > 0);
    }
    #[test]
    fn memory_weight_source_is_rejected_instead_of_being_rewritten_to_mmap() {
        let checkpoint = tiny_checkpoint();
        let options =
            StreamedGlmOptions::new(WeightSource::Memory, CachePolicy::new(1), Device::Cpu);
        let error = match StreamedGlm::open(checkpoint.path(), options) {
            Ok(_) => panic!("the GLM mmap-only profile must not rewrite a memory source"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("requires --weight-source mmap"));
    }
    #[test]
    fn inventory_rejects_a_scale_that_would_be_ignored_for_bf16() {
        let checkpoint = tiny_checkpoint();
        let weights_path = checkpoint.path().join("model.safetensors");
        let mut tensors = safetensors::load(&weights_path, &Device::Cpu).unwrap();
        tensors.insert("lm_head.weight_scale_inv".to_owned(), f32_ones((1, 1)));
        safetensors::save(&tensors, &weights_path).unwrap();

        let error = match StreamedGlm::open(checkpoint.path(), tiny_options()) {
            Ok(_) => panic!("an ignored BF16 scale must be rejected"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("unexpectedly has inverse scale"));
    }
    #[test]
    fn inventory_rejects_an_invalid_fp8_scale_shape_before_generation() {
        let checkpoint = tiny_checkpoint();
        let weights_path = checkpoint.path().join("model.safetensors");
        let mut tensors = safetensors::load(&weights_path, &Device::Cpu).unwrap();
        let head = tensors
            .remove("lm_head.weight")
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        tensors.insert("lm_head.weight".to_owned(), head);
        tensors.insert("lm_head.weight_scale_inv".to_owned(), f32_ones((1, 2)));
        safetensors::save(&tensors, &weights_path).unwrap();

        let error = match StreamedGlm::open(checkpoint.path(), tiny_options()) {
            Ok(_) => panic!("an invalid FP8 scale shape must be rejected"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("expected [1, 1]"));
    }
}
