//! Lifecycle, sampling and rank admission for a persistent layer-partitioned model.
use super::*;
use crate::admission::{GlmAdmissionBreakdown, GlmLayerScope};
use crate::partition::{GlmPartitionPolicy, GlmPartitionTransport, GlmRankPolicy};
use ff_core::resource_selection::ResourcePhaseEstimate;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct LayerPartitionOptions {
    pub expert_cache_bytes_per_device: usize,
    pub resident_static: bool,
    pub cache_policy: CachePolicy,
    pub cache_layout: ExpertCacheLayout,
    pub replacement: ExpertCacheReplacementPolicy,
    pub cpu_fp8_dequantization: bool,
    pub pinned_fp8_transfer: bool,
    pub max_context_tokens: Option<usize>,
}
impl Default for LayerPartitionOptions {
    fn default() -> Self {
        Self {
            expert_cache_bytes_per_device: 0,
            resident_static: false,
            cache_policy: CachePolicy::new(1),
            cache_layout: ExpertCacheLayout::PerLayerSplit,
            replacement: ExpertCacheReplacementPolicy::Lru,
            cpu_fp8_dequantization: false,
            pinned_fp8_transfer: false,
            max_context_tokens: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmRankAdmission {
    pub rank: usize,
    /// Older fresh-request observations did not record a credit. Zero remains
    /// conservative when reading them.
    #[serde(default)]
    pub already_resident_device_bytes: u64,
    pub breakdown: GlmAdmissionBreakdown,
    pub phases: Vec<ResourcePhaseEstimate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmPartitionAdmission {
    pub ranks: Vec<GlmRankAdmission>,
    pub required_host_bytes: u64,
    pub snapshots: Vec<ResourceSnapshot>,
}

impl GlmPartitionAdmission {
    pub fn validate_capacity(&self) -> Result<()> {
        ensure!(
            self.ranks.len() >= 2 && self.ranks.len() == self.snapshots.len(),
            "partition admission rank count mismatch"
        );
        let mut host_sum = 0u64;
        let mut unified_device_sum = 0u64;
        let mut unified_pool: Option<u64> = None;
        for (rank, (record, snapshot)) in self.ranks.iter().zip(&self.snapshots).enumerate() {
            ensure!(
                record.rank == rank && !record.phases.is_empty(),
                "invalid partition admission rank/phases"
            );
            let cache_bound = record
                .phases
                .iter()
                .filter_map(|p| p.optional_device_bytes)
                .max()
                .unwrap_or(0);
            let maximum_retained = (record.breakdown.static_bytes as u64)
                .checked_add(record.breakdown.kda_state_bytes as u64)
                .and_then(|n| n.checked_add(record.breakdown.maximum_dsa_cache_bytes as u64))
                .and_then(|n| n.checked_add(cache_bound))
                .and_then(|n| n.checked_add(record.breakdown.pinned_transfer_bytes))
                .context("rank retained limit overflow")?;
            ensure!(
                record.already_resident_device_bytes <= maximum_retained,
                "rank retained credit exceeds its declared storage"
            );
            let mut host_peak = 0;
            // Phases are mutually exclusive, so a rank contributes its peak; the
            // ranks run together, so those peaks are what get summed.
            let mut rank_unified_peak = 0u64;
            for phase in &record.phases {
                host_peak = host_peak.max(phase.host_peak_bytes()?);
                let total = phase
                    .device_peak_bytes()?
                    .context("missing rank device bound")?;
                let required = total.saturating_sub(record.already_resident_device_bytes);
                // Integrated ranks are checked together after the loop: their
                // device allocations and every rank's host allocations draw from
                // one pool, so per-rank checks admit a combination it cannot hold.
                if let Some(pool) = snapshot.unified_pool_available_bytes() {
                    rank_unified_peak = rank_unified_peak.max(required);
                    unified_pool =
                        Some(unified_pool.map_or(pool, |current: u64| current.min(pool)));
                    continue;
                }
                ensure!(
                    !snapshot.unified_accounting_is_undecidable(),
                    "GLM rank {rank} is an integrated device whose shared pool could not be measured"
                );
                let available = snapshot
                    .device_free_memory_bytes
                    .context("rank device memory unavailable")?;
                ensure!(
                    required <= available,
                    "GLM rank {rank} {} requires {required} additional device bytes, only {available} available",
                    phase.phase
                );
            }
            host_sum = host_sum
                .checked_add(host_peak)
                .context("partition host peak overflow")?;
            unified_device_sum = unified_device_sum
                .checked_add(rank_unified_peak)
                .context("partition unified device sum overflow")?;
        }
        ensure!(
            host_sum == self.required_host_bytes,
            "partition aggregate host bound is inconsistent"
        );
        if let Some(pool) = unified_pool {
            let combined = unified_device_sum
                .checked_add(self.required_host_bytes)
                .context("partition unified peak overflow")?;
            ensure!(
                combined <= pool,
                "GLM integrated ranks jointly need {combined} bytes from the unified host/device pool, but only {pool} are available"
            );
        }
        let available = self
            .snapshots
            .iter()
            .filter_map(crate::admission::host_available)
            .min()
            .context("partition host memory unavailable")?;
        ensure!(
            host_sum <= available,
            "GLM ranks jointly require {host_sum} host bytes, only {available} available"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmRankCacheStats {
    pub rank: usize,
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub static_resident_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmPartitionGeneration {
    pub prompt_tokens: usize,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub prefill_elapsed_ms: u64,
    pub decode_elapsed_ms: u64,
    pub token_elapsed_ms: Vec<u64>,
    pub transfer_count: u64,
    pub transferred_bytes: u64,
    pub cache_before: Vec<GlmRankCacheStats>,
    pub cache_after: Vec<GlmRankCacheStats>,
    #[serde(default)]
    pub fp8_transfers: Vec<fp8::Fp8TransferStats>,
}

impl LayerPartitionedGlm {
    /// Prepare only metadata. No decoder state, static weights or experts are allocated.
    pub fn prepare(
        model: impl AsRef<Path>,
        devices: Vec<Device>,
        options: LayerPartitionOptions,
    ) -> Result<Self> {
        ensure!(
            devices.len() >= 2,
            "layer partitioning needs at least two devices"
        );
        let mut locations = std::collections::HashSet::new();
        for device in &devices {
            ensure!(
                device.is_cuda() && locations.insert(device.location()),
                "layer partitioning requires distinct CUDA devices"
            );
        }
        let config = GlmConfig::from_model_dir(model.as_ref())?;
        let context = options
            .max_context_tokens
            .unwrap_or(config.text_config.index_topk);
        ensure!(
            context > 0 && context <= config.text_config.index_topk,
            "partition context exceeds the exact GLM profile"
        );
        let mut ranks = Vec::new();
        let mut workers = Vec::new();
        let mut breakdowns = Vec::new();
        let count = devices.len();
        for (rank, device) in devices.into_iter().enumerate() {
            let scope = GlmLayerScope::for_rank(config.text_config.num_hidden_layers, rank, count)?;
            let prepared = StreamedGlm::prepare(
                model.as_ref(),
                StreamedGlmOptions::new(WeightSource::Mmap, options.cache_policy, device)
                    .with_cpu_fp8_dequantization(options.cpu_fp8_dequantization)
                    .with_pinned_fp8_transfer(options.pinned_fp8_transfer),
            )?;
            let mut worker = prepared.model;
            let mut breakdown = GlmAdmissionBreakdown::from_metadata_for_scope(
                &worker.weights,
                &worker.config.text_config,
                false,
                1,
                options.cpu_fp8_dequantization,
                scope,
            )?;
            if options.pinned_fp8_transfer {
                let (lanes, fill_ring_depth) = crate::model::pinned_staging_shape();
                breakdown.enable_pinned_transfer(lanes, fill_ring_depth)?;
            }
            breakdown.maximum_dsa_cache_bytes = breakdown
                .dsa_cache_bytes_per_token
                .checked_mul(context)
                .context("rank DSA context overflow")?;
            let dsa_layers = worker
                .config
                .text_config
                .sparse_attention_layers()
                .filter(|&n| scope.contains_layer(n))
                .count();
            breakdown.maximum_dsa_layer_cache_bytes = breakdown
                .maximum_dsa_cache_bytes
                .checked_div(dsa_layers)
                .unwrap_or(0);
            let sparse: Vec<_> = breakdown
                .sparse_layers
                .iter()
                .map(|&n| n as usize)
                .collect();
            let budget = if sparse.is_empty() {
                0
            } else {
                options.expert_cache_bytes_per_device
            };
            worker.expert_cache = ExpertCacheManager::for_owned_layers(
                scope.total_layers,
                &sparse,
                budget,
                options.cache_layout,
                options.replacement,
            )?;
            worker.execution_policy.resident_static = options.resident_static;
            worker.execution_policy.dsa_context_bound_tokens = u32::try_from(context)?;
            worker.execution_policy.expert_cache.maximum_bound_bytes = u64::try_from(budget)?;
            worker.execution_policy.expert_cache.minimum_bound_bytes = u64::try_from(budget)?;
            worker.execution_policy.expert_cache.layout = options.cache_layout;
            worker.execution_policy.expert_cache.replacement = options.replacement;
            let candle_core::DeviceLocation::Cuda { gpu_id: ordinal } = worker.device.location()
            else {
                unreachable!()
            };
            ranks.push(GlmRankPolicy {
                ordinal,
                scope,
                execution: worker.execution_policy.clone(),
            });
            breakdowns.push(breakdown);
            workers.push(worker);
        }
        let policy = GlmPartitionPolicy {
            schema_version: 1,
            transport: GlmPartitionTransport::SynchronizedCudaDeviceCopyV1,
            ranks,
        };
        policy.validate()?;
        let mut owners = vec![0; config.text_config.num_hidden_layers];
        for (rank, p) in policy.ranks.iter().enumerate() {
            for owner in &mut owners[p.scope.start..p.scope.end] {
                *owner = rank;
            }
        }
        Ok(Self {
            workers,
            owners,
            caches: vec![],
            tokens: 0,
            transfers: 0,
            transferred_bytes: 0,
            policy,
            breakdowns,
            initialized: false,
            failed: false,
        })
    }

    pub fn policy(&self) -> &GlmPartitionPolicy {
        &self.policy
    }

    /// Choose per-rank retention while workers still contain metadata only.
    /// Static choices are admitted against the shared host and all device
    /// limits; expert ceilings then use each rank's own remaining capacity.
    pub fn configure_automatic_residency(
        &mut self,
        prompt_tokens: usize,
        static_weights: bool,
        experts: bool,
    ) -> Result<()> {
        ensure!(
            !self.initialized && !self.failed,
            "automatic placement must precede execution"
        );
        if static_weights {
            for rank in 0..self.workers.len() {
                if self.policy.ranks[rank].execution.resident_static {
                    continue;
                }
                self.policy.ranks[rank].execution.resident_static = true;
                if self.admission(prompt_tokens)?.validate_capacity().is_err() {
                    self.policy.ranks[rank].execution.resident_static = false;
                }
                self.workers[rank].execution_policy.resident_static =
                    self.policy.ranks[rank].execution.resident_static;
            }
        }
        if experts {
            let admission = self.admission(prompt_tokens)?;
            // Integrated ranks draw their caches from one pool, and the
            // validator sums them, so each gets a share of the remainder rather
            // than all of it. Discrete ranks own their device outright.
            let unified_ranks = admission
                .snapshots
                .iter()
                .filter(|s| s.unified_pool_available_bytes().is_some())
                .count()
                .max(1) as u64;
            for rank in 0..self.workers.len() {
                let estimate = &admission.ranks[rank];
                let shared = admission.snapshots[rank]
                    .unified_pool_available_bytes()
                    .is_some();
                // Sized against the ledger the validator uses.
                let bytes = estimate
                    .breakdown
                    .automatic_expert_cache_bytes_against_host(
                        &estimate.phases,
                        &admission.snapshots[rank],
                        Some(admission.required_host_bytes),
                    )?;
                let bytes = if shared {
                    bytes / unified_ranks as usize
                } else {
                    bytes
                };
                self.workers[rank].expert_cache.resize(bytes)?;
                let policy = &mut self.workers[rank].execution_policy.expert_cache;
                policy.maximum_bound_bytes = bytes as u64;
                policy.minimum_bound_bytes = bytes as u64;
                self.policy.ranks[rank].execution.expert_cache = policy.clone();
            }
        }
        self.policy.validate()?;
        self.admission(prompt_tokens)?.validate_capacity()
    }
    pub fn devices(&self) -> Vec<Device> {
        self.workers.iter().map(|w| w.device.clone()).collect()
    }
    pub fn context_bound(&self) -> usize {
        self.policy.ranks[0].execution.dsa_context_bound_tokens as usize
    }
    pub fn cache_stats(&self) -> Vec<GlmRankCacheStats> {
        self.workers
            .iter()
            .enumerate()
            .map(|(rank, w)| {
                let s = w.expert_cache_stats();
                GlmRankCacheStats {
                    rank,
                    budget_bytes: s.max_bytes,
                    resident_bytes: s.bytes,
                    hits: s.hits,
                    misses: s.misses,
                    evictions: s.evictions,
                    static_resident_bytes: tensor_bytes(w.static_weights.values()),
                }
            })
            .collect()
    }

    pub fn enable_weight_read_audit(&self) {
        for w in &self.workers {
            w.weights.count_tensor_reads(true);
        }
    }
    pub fn weight_reads(&self) -> Vec<BTreeMap<String, u64>> {
        self.workers
            .iter()
            .map(|w| w.weights.tensor_reads())
            .collect()
    }

    pub fn tokenize(&self, prompt: &str, effort: &str) -> Result<Vec<u32>> {
        ensure!(!prompt.trim().is_empty(), "GLM prompt must not be empty");
        let rendered = render_chat_prompt(prompt, effort)?;
        Ok(self.workers[0]
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| anyhow::anyhow!("failed to tokenize GLM prompt: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// All ranks share one host limit; device limits remain independent. Already
    /// retained static/state/cache tensors are credited only on their own device.
    pub fn admission(&self, prompt_tokens: usize) -> Result<GlmPartitionAdmission> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.context_bound(),
            "invalid partition prompt length"
        );
        let text = &self.workers[0].config.text_config;
        let snapshots: Vec<_> = self
            .workers
            .iter()
            .map(|w| ResourceSnapshot::capture(Some(&w.device)))
            .collect();
        let mut required_host_bytes = 0u64;
        let mut ranks = Vec::new();
        for (rank, (base, worker)) in self.breakdowns.iter().zip(&self.workers).enumerate() {
            let mut b = base.clone();
            b.prompt_tokens = prompt_tokens;
            b.prefill_workspace_bytes = prefill::prefill_workspace_bytes(text, prompt_tokens)?;
            b.prefill_host_mask_bytes = crate::admission::host_mask_bytes(prompt_tokens, false)?;
            let rp = &self.policy.ranks[rank];
            // Each rank is judged against its own device, so it gets its own
            // pool-scaled reserve; `phases` would apply the flat constant.
            let mut phases = b.phases_with_safety(
                rp.execution.resident_static,
                worker.expert_cache_stats().max_bytes,
                worker.weights.cache_policy(),
                b.scaled_admission_safety_bytes(&snapshots[rank]),
            )?;
            let mut host_peak = 0;
            let mut retained = tensor_bytes(worker.static_weights.values())
                .checked_add(worker.fp8_staging_bytes()?)
                .context("FP8 staging retained credit overflow")?;
            retained = retained
                .checked_add(worker.expert_cache_stats().bytes as u64)
                .context("rank retained bytes overflow")?;
            for (layer, cache) in self.caches.iter().enumerate() {
                if self.owners[layer] == rank {
                    retained = retained
                        .checked_add(cache.bytes())
                        .context("rank state bytes overflow")?;
                }
            }
            for phase in &mut phases {
                let rows = if phase.phase == "prefill" {
                    prompt_tokens
                } else {
                    1
                };
                let boundary = rows
                    .checked_mul(text.hc_mult)
                    .and_then(|n| n.checked_mul(text.hidden_size))
                    .and_then(|n| n.checked_mul(4))
                    .context("GLM transfer workspace overflow")?;
                phase.required_device_bytes = Some(
                    phase
                        .required_device_bytes
                        .unwrap_or(0)
                        .checked_add(boundary as u64)
                        .context("rank transfer peak overflow")?,
                );
                host_peak = host_peak.max(phase.host_peak_bytes()?);
            }
            required_host_bytes = required_host_bytes
                .checked_add(host_peak)
                .context("partition host peak overflow")?;
            ranks.push(GlmRankAdmission {
                rank,
                already_resident_device_bytes: retained,
                breakdown: b,
                phases,
            });
        }
        let admission = GlmPartitionAdmission {
            ranks,
            required_host_bytes,
            snapshots,
        };
        admission.validate_capacity()?;
        Ok(admission)
    }

    pub(super) fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            let p = &self.policy.ranks[rank];
            for (name, linear_weight) in static_weight_specs(&worker.config.text_config) {
                if !p.scope.contains_static(&name)
                    || (!p.execution.resident_static && name != LM_HEAD_WEIGHT)
                {
                    continue;
                }
                let value = if linear_weight {
                    worker.load_linear_weight(&name)?
                } else {
                    worker.load_tensor(&name)?
                };
                worker.static_weights.insert(name, value);
            }
        }
        self.caches = self
            .owners
            .iter()
            .enumerate()
            .map(|(layer, &owner)| {
                LayerCache::new(
                    self.workers[owner].config.text_config.layer_types[layer],
                    &self.workers[owner].config.text_config,
                    self.workers[owner].compute_dtype,
                    &self.workers[owner].device,
                )
            })
            .collect::<Result<_>>()?;
        self.initialized = true;
        Ok(())
    }

    /// Drop only request state. Weight caches survive across independent requests.
    pub fn reset(&mut self) -> Result<()> {
        for w in &self.workers {
            w.device.synchronize()?;
        }
        self.caches.clear();
        self.tokens = 0;
        self.transfers = 0;
        self.transferred_bytes = 0;
        self.initialized = false;
        self.failed = false;
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GlmGenerationOptions,
    ) -> Result<GlmPartitionGeneration> {
        ensure!(
            options.max_new_tokens > 0
                && options.max_context_tokens > 0
                && options.max_context_tokens <= self.context_bound(),
            "invalid GLM generation context limits"
        );
        ensure!(
            options.temperature.is_finite()
                && options.temperature >= 0.
                && (options.temperature == 0. || options.temperature.recip().is_finite()),
            "invalid sampling temperature"
        );
        ensure!(
            options.top_p.is_finite() && options.top_p > 0. && options.top_p <= 1.,
            "invalid nucleus probability"
        );
        let ids = self.tokenize(prompt, &options.reasoning_effort)?;
        ensure!(
            ids.len()
                .checked_add(options.max_new_tokens)
                .is_some_and(|n| n <= options.max_context_tokens),
            "GLM prompt and output exceed request context"
        );
        ensure!(
            !self.failed,
            "GLM partition request failed; reset before reuse"
        );
        if self.tokens > 0 {
            self.reset()?;
        }
        let cache_before = self.cache_stats();
        let start = Instant::now();
        let mut logits = self.prefill_ids(&ids)?;
        self.workers.last().unwrap().device.synchronize()?;
        let prefill_elapsed_ms = millis(start.elapsed())?;
        let start = Instant::now();
        let mut tokens = Vec::new();
        let mut times = Vec::new();
        let mut rng = StdRng::seed_from_u64(options.seed);
        let result = (|| -> Result<()> {
            for step in 0..options.max_new_tokens {
                let tick = Instant::now();
                let token = sample_token(&logits, options.temperature, options.top_p, &mut rng)?;
                tokens.push(token);
                if self.workers[0]
                    .generation_config
                    .eos_token_ids
                    .contains(&token)
                {
                    times.push(millis(tick.elapsed())?);
                    break;
                }
                if step + 1 < options.max_new_tokens {
                    logits = self.decode_id(token)?;
                    self.workers.last().unwrap().device.synchronize()?;
                }
                times.push(millis(tick.elapsed())?);
                if options.progress {
                    eprintln!(
                        "GLM partition token {}/{} id {token}",
                        step + 1,
                        options.max_new_tokens
                    );
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        let text = self.workers[0]
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("GLM detokenization failed: {e}"))?;
        Ok(GlmPartitionGeneration {
            prompt_tokens: ids.len(),
            generated_token_ids: tokens,
            text,
            prefill_elapsed_ms,
            decode_elapsed_ms: millis(start.elapsed())?,
            token_elapsed_ms: times,
            transfer_count: self.transfers,
            transferred_bytes: self.transferred_bytes,
            cache_before,
            cache_after: self.cache_stats(),
            fp8_transfers: self
                .workers
                .iter()
                .map(StreamedGlm::fp8_transfer_stats)
                .collect::<Result<_>>()?,
        })
    }
}

fn tensor_bytes<'a>(tensors: impl Iterator<Item = &'a Tensor>) -> u64 {
    tensors
        .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
        .sum()
}
fn millis(d: Duration) -> Result<u64> {
    Ok(u64::try_from(d.as_millis())?)
}

impl LayerCache {
    fn bytes(&self) -> u64 {
        match self {
            Self::Kda(k) => {
                tensor_bytes([&k.query_conv, &k.key_conv, &k.value_conv, &k.recurrent].into_iter())
            }
            Self::Dsa(d) => tensor_bytes(d.keys.iter().chain(d.values.iter())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::probe::{CgroupMemoryLimit, ResourceMeasurementScopes};

    #[test]
    fn scoped_admission_sums_host_and_credits_only_the_owning_device() {
        let root = crate::test_support::tiny_checkpoint();
        let weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let config = GlmConfig::from_model_dir(root.path()).unwrap();
        let mut ranks = Vec::new();
        let mut snapshots = Vec::new();
        for rank in 0..2 {
            ranks.push(GlmRankAdmission {
                rank,
                already_resident_device_bytes: 0,
                breakdown: GlmAdmissionBreakdown::from_metadata_for_scope(
                    &weights,
                    &config.text_config,
                    false,
                    1,
                    false,
                    GlmLayerScope::for_rank(2, rank, 2).unwrap(),
                )
                .unwrap(),
                phases: vec![ResourcePhaseEstimate {
                    phase: "prefill".into(),
                    required_host_bytes: 60,
                    optional_host_bytes: 0,
                    reclaimable_host_bytes: 0,
                    host_promotion_reserve_bytes: 0,
                    required_device_bytes: Some(10),
                    optional_device_bytes: Some(0),
                    device_reserve_bytes: 0,
                }],
            });
            snapshots.push(ResourceSnapshot {
                schema_version: 1,
                measured_at_unix_ms: 1,
                host_memory_available_bytes: Some(100),
                cgroup_v2_memory_limit: Some(CgroupMemoryLimit::Bytes(100)),
                cgroup_v2_memory_current_bytes: Some(0),
                cgroup_v2_memory_available_bytes: Some(100),
                device_free_memory_bytes: Some(10),
                host_device_memory_is_unified: None,
                device_topology_probe_failed: false,
                host_memory_total_bytes: None,
                device_total_memory_bytes: None,
                measurement_scope: ResourceMeasurementScopes {
                    host_memory: None,
                    cgroup_memory: None,
                    device_memory: None,
                },
            });
        }
        let mut report = GlmPartitionAdmission {
            ranks,
            required_host_bytes: 120,
            snapshots,
        };
        assert!(
            report
                .validate_capacity()
                .unwrap_err()
                .to_string()
                .contains("jointly")
        );
        for s in &mut report.snapshots {
            s.host_memory_available_bytes = Some(120);
            s.cgroup_v2_memory_available_bytes = Some(120);
        }
        report.validate_capacity().unwrap();
        report.snapshots[0].device_free_memory_bytes = Some(3);
        report.ranks[1].already_resident_device_bytes = 7;
        assert!(
            report
                .validate_capacity()
                .unwrap_err()
                .to_string()
                .contains("rank 0")
        );
        report.ranks[0].already_resident_device_bytes = 7;
        report.validate_capacity().unwrap();
        report.required_host_bytes = 60;
        assert!(
            report
                .validate_capacity()
                .unwrap_err()
                .to_string()
                .contains("inconsistent")
        );
    }
}
