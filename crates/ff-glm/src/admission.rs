//! Metadata-derived phase peaks. No device allocation or live capacity probe.
use crate::{
    config::{GlmTextConfig, MlpKind},
    execution_policy::GLM_ADMISSION_SAFETY_BYTES,
    model::{
        LM_HEAD_WEIGHT, is_mhc_constant, prefill::prefill_workspace_bytes, static_weight_specs,
        streamed_static_weight_groups,
    },
};
use anyhow::{Context, Result, bail, ensure};
use candle_core::DType;
use ff_core::{
    probe::{ResourceSnapshot, admission_reserve_bytes},
    resource_selection::ResourcePhaseEstimate,
    weights::{
        CachePolicy, ModelWeights, TensorMetadata, WeightSource,
        accounting::{CacheInventory, CacheLoadLifetimes, estimate_cache_residency},
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmLayerScope {
    pub start: usize,
    pub end: usize,
    pub total_layers: usize,
}

impl GlmLayerScope {
    pub fn for_rank(total_layers: usize, rank: usize, ranks: usize) -> Result<Self> {
        ensure!(
            ranks > 0 && ranks <= total_layers && rank < ranks,
            "invalid GLM layer partition"
        );
        Ok(Self {
            start: rank
                .checked_mul(total_layers)
                .context("GLM partition overflow")?
                .div_ceil(ranks),
            end: (rank + 1)
                .checked_mul(total_layers)
                .context("GLM partition overflow")?
                .div_ceil(ranks),
            total_layers,
        })
    }
    pub fn contains_layer(self, layer: usize) -> bool {
        (self.start..self.end).contains(&layer)
    }
    pub fn owns_head(self) -> bool {
        self.end == self.total_layers
    }
    pub fn owns_embedding(self) -> bool {
        self.start == 0
    }
    pub fn contains_static(self, name: &str) -> bool {
        if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
            return rest
                .split('.')
                .next()
                .and_then(|n| n.parse().ok())
                .is_some_and(|n| self.contains_layer(n));
        }
        self.owns_head() && matches!(name, "lm_head.weight" | "model.language_model.norm.weight")
    }
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.start < self.end && self.end <= self.total_layers,
            "invalid GLM layer range"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmAdmissionBreakdown {
    pub scope: GlmLayerScope,
    pub compute_on_host: bool,
    pub cpu_fp8_dequantization: bool,
    /// Device (and host) bytes held by every configured upload lane's slots.
    #[serde(default)]
    pub pinned_transfer_bytes: u64,
    /// Host-only pinned bytes held by the fill-ahead ring.
    #[serde(default)]
    pub pinned_fill_ahead_bytes: u64,
    /// Loads that can be in flight at once, including the foreground one. Each
    /// owns its own decoded header, so this scales that exclusive charge.
    #[serde(default)]
    pub concurrent_loads: u64,
    pub prompt_tokens: usize,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub sparse_layers: Vec<u32>,
    pub static_bytes: usize,
    pub lm_head_bytes: usize,
    pub largest_streamed_group_bytes: usize,
    pub kda_state_bytes: usize,
    pub dsa_cache_bytes_per_token: usize,
    pub maximum_dsa_cache_bytes: usize,
    pub maximum_dsa_layer_cache_bytes: usize,
    pub live_expert_bytes: usize,
    pub prefill_workspace_bytes: usize,
    pub decode_workspace_bytes: usize,
    pub host_route_workspace_bytes: u64,
    pub host_sampling_workspace_bytes: u64,
    pub prefill_host_mask_bytes: u64,
    pub static_load_host_bytes: u64,
    pub streamed_load_host_bytes: u64,
    pub expert_load_host_bytes: u64,
    pub static_load_device_bytes: u64,
    pub expert_load_device_bytes: u64,
    pub raw_inventory: CacheInventory,
}

impl GlmAdmissionBreakdown {
    /// `weights` must have no materialized payloads at the caller's capture
    /// boundary. Includes static/initialization/scale names, not vision or MTP.
    pub fn from_metadata(
        weights: &ModelWeights,
        text: &GlmTextConfig,
        cpu: bool,
        prompt_tokens: usize,
    ) -> Result<Self> {
        Self::from_metadata_with_fp8(weights, text, cpu, prompt_tokens, cpu)
    }

    pub fn from_metadata_with_fp8(
        weights: &ModelWeights,
        text: &GlmTextConfig,
        cpu: bool,
        prompt_tokens: usize,
        cpu_fp8_dequantization: bool,
    ) -> Result<Self> {
        Self::from_metadata_for_scope(
            weights,
            text,
            cpu,
            prompt_tokens,
            cpu_fp8_dequantization,
            GlmLayerScope {
                start: 0,
                end: text.num_hidden_layers,
                total_layers: text.num_hidden_layers,
            },
        )
    }

    pub fn from_metadata_for_scope(
        weights: &ModelWeights,
        text: &GlmTextConfig,
        cpu: bool,
        prompt_tokens: usize,
        cpu_fp8_dequantization: bool,
        scope: GlmLayerScope,
    ) -> Result<Self> {
        scope.validate()?;
        ensure!(
            scope.total_layers == text.num_hidden_layers,
            "GLM scope disagrees with model layers"
        );
        let cpu_fp8_dequantization = cpu || cpu_fp8_dequantization;
        text.validate()?;
        let compute_dtype = if cpu { DType::F32 } else { DType::BF16 };
        let mut static_bytes = 0usize;
        let mut lm_head_bytes = 0usize;
        let mut static_tensor_bytes = BTreeMap::new();
        let specs: Vec<_> = static_weight_specs(text)
            .into_iter()
            .filter(|(name, _)| scope.contains_static(name))
            .collect();
        for (name, linear_weight) in &specs {
            let metadata = weights.metadata(name)?;
            let elements = metadata
                .shape
                .iter()
                .try_fold(1usize, |count, &dimension| {
                    count
                        .checked_mul(dimension)
                        .context("GLM tensor element count overflow")
                })?;
            let bytes_per_element = if cpu || is_mhc_constant(name) {
                DType::F32.size_in_bytes()
            } else if *linear_weight && metadata.dtype == "F8_E4M3" {
                compute_dtype.size_in_bytes()
            } else {
                match metadata.dtype.as_str() {
                    "BF16" => DType::BF16.size_in_bytes(),
                    "F32" => DType::F32.size_in_bytes(),
                    other => bail!("unsupported resident GLM tensor dtype {other}"),
                }
            };
            let bytes = elements
                .checked_mul(bytes_per_element)
                .context("GLM resident tensor byte count overflow")?;
            static_bytes = static_bytes
                .checked_add(bytes)
                .context("GLM static residency byte count overflow")?;
            if name == LM_HEAD_WEIGHT {
                lm_head_bytes = bytes;
            }
            ensure!(
                static_tensor_bytes.insert(name.clone(), bytes).is_none(),
                "duplicate GLM static admission tensor"
            );
        }

        let streamed_groups = streamed_static_weight_groups(text);
        let covered = streamed_groups
            .iter()
            .flatten()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        for name in static_tensor_bytes.keys() {
            if name != LM_HEAD_WEIGHT {
                ensure!(
                    covered.contains(name.as_str()),
                    "GLM streamed-static admission has no live group for {name}"
                );
            }
        }
        let largest_streamed_group_bytes =
            streamed_groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .filter(|name| scope.contains_static(name))
                        .try_fold(0usize, |total, name| {
                            total
                                .checked_add(*static_tensor_bytes.get(name).with_context(|| {
                                    format!("unknown GLM live-group tensor {name}")
                                })?)
                                .context("GLM streamed live-group byte count overflow")
                        })
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .context("GLM streamed-static live groups are empty")?;

        let qkv_dim = text.linear_qkv_dim()?;
        let recurrent_per_layer = text
            .linear_num_heads
            .checked_mul(text.linear_head_dim)
            .and_then(|value| value.checked_mul(text.linear_head_dim))
            .and_then(|value| value.checked_mul(DType::F32.size_in_bytes()))
            .context("GLM recurrent-state byte count overflow")?;
        let conv_per_layer = 3usize
            .checked_mul(qkv_dim)
            .and_then(|value| value.checked_mul(text.linear_conv_kernel_dim))
            .and_then(|value| value.checked_mul(compute_dtype.size_in_bytes()))
            .context("GLM convolution-state byte count overflow")?;
        let kda_state_bytes = recurrent_per_layer
            .checked_add(conv_per_layer)
            .and_then(|value| {
                value.checked_mul(
                    text.linear_attention_layers()
                        .filter(|&n| scope.contains_layer(n))
                        .count(),
                )
            })
            .context("GLM KDA state byte count overflow")?;
        let dsa_cache_bytes_per_token = text
            .sparse_attention_layers()
            .filter(|&n| scope.contains_layer(n))
            .count()
            .checked_mul(text.num_attention_heads)
            .and_then(|value| {
                value.checked_mul(text.qk_nope_head_dim.checked_add(text.v_head_dim)?)
            })
            .and_then(|value| value.checked_mul(compute_dtype.size_in_bytes()))
            .context("GLM per-token DSA cache byte count overflow")?;
        let dsa_cache_bytes = dsa_cache_bytes_per_token
            .checked_mul(text.index_topk)
            .context("GLM maximum DSA cache byte count overflow")?;
        let sparse_count = text
            .mlp_layer_types
            .iter()
            .enumerate()
            .filter(|(layer, kind)| scope.contains_layer(*layer) && **kind == MlpKind::Sparse)
            .count();
        let live_expert_bytes = if sparse_count == 0 {
            0
        } else {
            text.hidden_size
                .checked_mul(text.moe_intermediate_size)
                .and_then(|value| value.checked_mul(5))
                .and_then(|value| value.checked_mul(compute_dtype.size_in_bytes()))
                .context("GLM live expert byte count overflow")?
        };

        let mut names = BTreeSet::new();
        let mut static_load_host_bytes = 0;
        let mut expert_load_host_bytes = 0;
        for (name, linear) in specs {
            let metadata = weights.metadata(&name)?;
            let scale_name = format!("{name}_scale_inv");
            let scale = if linear && metadata.dtype == "F8_E4M3" {
                names.insert(scale_name.clone());
                Some(weights.metadata(&scale_name)?)
            } else {
                None
            };
            static_load_host_bytes = static_load_host_bytes.max(load_host_transient_bytes(
                &metadata,
                scale.as_ref(),
                cpu,
                cpu_fp8_dequantization,
            )?);
            names.insert(name);
        }
        if scope.owns_embedding() {
            names.insert("model.language_model.embed_tokens.weight".into());
        }
        for (layer, kind) in text.mlp_layer_types.iter().enumerate() {
            if *kind != MlpKind::Sparse || !scope.contains_layer(layer) {
                continue;
            }
            for expert in 0..text.n_routed_experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    let name = format!(
                        "model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    );
                    let metadata = weights.metadata(&name)?;
                    let scale_name = format!("{name}_scale_inv");
                    let scale = if metadata.dtype == "F8_E4M3" {
                        names.insert(scale_name.clone());
                        Some(weights.metadata(&scale_name)?)
                    } else {
                        None
                    };
                    expert_load_host_bytes = expert_load_host_bytes.max(load_host_transient_bytes(
                        &metadata,
                        scale.as_ref(),
                        cpu,
                        cpu_fp8_dequantization,
                    )?);
                    names.insert(name);
                }
            }
        }
        let mut static_load_device_bytes = 0;
        let mut expert_load_device_bytes = 0;
        if !cpu {
            for name in &names {
                let metadata = weights.metadata(name)?;
                if !cpu_fp8_dequantization && metadata.dtype == "F8_E4M3" {
                    let scale = weights.metadata(&format!("{name}_scale_inv"))?;
                    let bytes = u64::try_from(metadata.bytes)?
                        .checked_add(u64::try_from(scale.bytes)?)
                        .context("GLM raw GPU dequant staging overflow")?;
                    if name.contains(".experts.") {
                        expert_load_device_bytes = expert_load_device_bytes.max(bytes);
                    } else {
                        static_load_device_bytes = static_load_device_bytes.max(bytes);
                    }
                } else if is_mhc_constant(name) && metadata.dtype == "BF16" {
                    // The raw tensor remains live while its F32 replacement is allocated.
                    static_load_device_bytes =
                        static_load_device_bytes.max(u64::try_from(metadata.bytes)?);
                }
            }
        }
        let raw_inventory = weights.cache_inventory_for(names.iter().map(String::as_str))?;
        Ok(Self {
            scope,
            compute_on_host: cpu,
            cpu_fp8_dequantization,
            pinned_transfer_bytes: 0,
            pinned_fill_ahead_bytes: 0,
            concurrent_loads: 0,
            prompt_tokens,
            num_hidden_layers: u32::try_from(text.num_hidden_layers)?,
            num_experts: u32::try_from(text.n_routed_experts)?,
            experts_per_token: u32::try_from(text.num_experts_per_tok)?,
            sparse_layers: text
                .mlp_layer_types
                .iter()
                .enumerate()
                .filter_map(|(i, k)| {
                    (*k == MlpKind::Sparse && scope.contains_layer(i)).then_some(i)
                })
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            static_bytes,
            lm_head_bytes,
            largest_streamed_group_bytes,
            kda_state_bytes,
            dsa_cache_bytes_per_token,
            maximum_dsa_cache_bytes: dsa_cache_bytes,
            maximum_dsa_layer_cache_bytes: dsa_cache_bytes
                .checked_div(
                    text.sparse_attention_layers()
                        .filter(|&n| scope.contains_layer(n))
                        .count(),
                )
                .unwrap_or(0),
            live_expert_bytes,
            static_load_device_bytes,
            expert_load_device_bytes,
            prefill_workspace_bytes: prefill_workspace_bytes(text, prompt_tokens)?,
            decode_workspace_bytes: prefill_workspace_bytes(text, 1)?,
            host_route_workspace_bytes: u64::try_from(text.index_topk)?
                .checked_mul(sparse_count as u64)
                .and_then(|n| {
                    n.checked_mul(
                        128_u64
                            .checked_add(32_u64.checked_mul(text.num_experts_per_tok as u64)?)?,
                    )
                })
                .context("GLM host routing workspace overflow")?,
            prefill_host_mask_bytes: host_mask_bytes(prompt_tokens, cpu)?,
            host_sampling_workspace_bytes: if scope.owns_head() {
                u64::try_from(text.vocab_size)?
                    .checked_mul(24)
                    .context("GLM sampling workspace overflow")?
            } else {
                0
            },
            static_load_host_bytes,
            streamed_load_host_bytes: static_load_host_bytes.max(expert_load_host_bytes),
            expert_load_host_bytes,
            raw_inventory,
        })
    }

    /// Reserve two persistent slots per upload lane using the largest device
    /// staging requirement, and `fill_ring_depth` host-only fill-ahead buffers
    /// of the same ceiling. The lane reservation is charged independently to
    /// host and device; the fill-ahead ring is pinned host memory only.
    ///
    /// Both counts come from the runtime configuration (`FF_GLM_LOAD_LANES`,
    /// `FF_GLM_FILL_AHEAD`), so admission bounds what inference will allocate.
    pub fn enable_pinned_transfer(&mut self, lanes: usize, fill_ring_depth: usize) -> Result<()> {
        ensure!(
            !self.compute_on_host && !self.cpu_fp8_dequantization,
            "pinned FP8 staging requires CUDA conversion"
        );
        let slot_bytes = self
            .static_load_device_bytes
            .max(self.expert_load_device_bytes);
        self.pinned_transfer_bytes = u64::try_from(lanes.max(1))
            .ok()
            .and_then(|lanes| lanes.checked_mul(2))
            .and_then(|slots| slot_bytes.checked_mul(slots))
            .context("FP8 staging capacity overflow")?;
        self.pinned_fill_ahead_bytes = u64::try_from(fill_ring_depth)
            .ok()
            .and_then(|depth| slot_bytes.checked_mul(depth))
            .context("FP8 fill-ahead capacity overflow")?;
        Ok(())
    }

    pub fn phases(
        &self,
        resident_static: bool,
        expert_cache_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Vec<ResourcePhaseEstimate>> {
        self.phases_with_safety(
            resident_static,
            expert_cache_bytes,
            cache_policy,
            GLM_ADMISSION_SAFETY_BYTES,
        )
    }

    /// `safety_bytes` is the reserve applied per phase; admission-time callers
    /// pass the pool-scaled value from `scaled_admission_safety_bytes`, tests
    /// and the policy contract keep the declared constant via `phases()`.
    pub fn phases_with_safety(
        &self,
        resident_static: bool,
        expert_cache_bytes: usize,
        cache_policy: CachePolicy,
        safety_bytes: u64,
    ) -> Result<Vec<ResourcePhaseEstimate>> {
        // The mmap'd shard residency is page cache: file-backed and
        // kernel-reclaimable, already counted as available in MemAvailable.
        // It is reported as `reclaimable_host_bytes` (telemetry) and never
        // charged in fit decisions; only exclusive allocations are.
        // The largest shard header is an owned Vec: required, not reclaimable.
        let header_copies = self.largest_header_bytes(cache_policy);
        let raw = estimate_cache_residency(
            &self.raw_inventory,
            WeightSource::Mmap,
            cache_policy,
            CacheLoadLifetimes {
                additional_storage_bytes: header_copies,
                maximum_concurrent_loads: self.concurrent_loads(),
                ..CacheLoadLifetimes::SERIAL
            },
        )?
        .peak_storage_bytes
        .checked_sub(header_copies)
        .context("GLM cache residency is smaller than its header copies")?;
        let weights = if resident_static {
            self.static_bytes
        } else {
            self.lm_head_bytes
                .checked_add(self.largest_streamed_group_bytes)
                .context("GLM streamed weight residency overflow")?
        };
        let state = self
            .kda_state_bytes
            .checked_add(self.maximum_dsa_cache_bytes)
            .context("GLM state bytes overflow")?;
        let decode_state = state
            .checked_add(self.maximum_dsa_layer_cache_bytes)
            .context("GLM KV growth bytes overflow")?;
        let load_host = if resident_static {
            self.expert_load_host_bytes
        } else {
            self.streamed_load_host_bytes
        };
        let load_device = if resident_static {
            self.expert_load_device_bytes
        } else {
            self.static_load_device_bytes
                .max(self.expert_load_device_bytes)
        };
        let runtime = |phase: &str, state, workspace| -> Result<ResourcePhaseEstimate> {
            let host_workspace = self
                .host_route_workspace_bytes
                .checked_add(self.host_sampling_workspace_bytes)
                .and_then(|n| {
                    n.checked_add(if phase == "prefill" {
                        self.prefill_host_mask_bytes
                    } else {
                        0
                    })
                })
                .and_then(|n| n.checked_add(load_host))
                .and_then(|n| n.checked_add(header_copies))
                .context("GLM host workspace overflow")?;
            let compute = [weights, state, self.live_expert_bytes, workspace]
                .into_iter()
                .try_fold(0_u64, |sum, v| {
                    sum.checked_add(u64::try_from(v)?)
                        .context("GLM phase compute bytes overflow")
                })?;
            if self.compute_on_host {
                Ok(ResourcePhaseEstimate {
                    phase: phase.into(),
                    required_host_bytes: compute
                        .checked_add(host_workspace)
                        .and_then(|v| v.checked_add(safety_bytes))
                        .context("GLM CPU phase bytes overflow")?,
                    optional_host_bytes: u64::try_from(expert_cache_bytes)
                        .context("GLM CPU retention bytes overflow")?,
                    reclaimable_host_bytes: raw,
                    host_promotion_reserve_bytes: 0,
                    required_device_bytes: None,
                    optional_device_bytes: None,
                    device_reserve_bytes: 0,
                })
            } else {
                Ok(ResourcePhaseEstimate {
                    phase: phase.into(),
                    required_host_bytes: host_workspace,
                    optional_host_bytes: 0,
                    reclaimable_host_bytes: raw,
                    host_promotion_reserve_bytes: 0,
                    required_device_bytes: Some(
                        compute
                            .checked_add(load_device)
                            .context("GLM device staging peak overflow")?,
                    ),
                    optional_device_bytes: Some(u64::try_from(expert_cache_bytes)?),
                    device_reserve_bytes: safety_bytes,
                })
            }
        };
        let mut phases = vec![
            runtime("prefill", state, self.prefill_workspace_bytes)?,
            runtime("decode", decode_state, self.decode_workspace_bytes)?,
        ];
        if resident_static {
            phases.insert(
                0,
                if self.compute_on_host {
                    ResourcePhaseEstimate {
                        phase: "static_initialization".into(),
                        required_host_bytes: u64::try_from(self.static_bytes)?
                            .checked_add(self.static_load_host_bytes)
                            .and_then(|v| v.checked_add(safety_bytes))
                            .context("GLM CPU initialization overflow")?,
                        optional_host_bytes: 0,
                        reclaimable_host_bytes: raw,
                        host_promotion_reserve_bytes: 0,
                        required_device_bytes: None,
                        optional_device_bytes: None,
                        device_reserve_bytes: 0,
                    }
                } else {
                    ResourcePhaseEstimate {
                        phase: "static_initialization".into(),
                        required_host_bytes: self.static_load_host_bytes,
                        optional_host_bytes: 0,
                        reclaimable_host_bytes: raw,
                        host_promotion_reserve_bytes: 0,
                        required_device_bytes: Some(
                            u64::try_from(self.static_bytes)?
                                .checked_add(self.static_load_device_bytes)
                                .context("GLM static GPU staging peak overflow")?,
                        ),
                        optional_device_bytes: Some(0),
                        device_reserve_bytes: safety_bytes,
                    }
                },
            );
        }
        for phase in &mut phases {
            phase.required_host_bytes = phase
                .required_host_bytes
                .checked_add(self.pinned_transfer_bytes)
                .and_then(|bytes| bytes.checked_add(self.pinned_fill_ahead_bytes))
                .context("pinned host staging peak overflow")?;
            if let Some(bytes) = &mut phase.required_device_bytes {
                *bytes = bytes
                    .checked_add(self.pinned_transfer_bytes)
                    .context("pinned device staging peak overflow")?;
            }
        }
        Ok(phases)
    }

    /// Remaining CUDA capacity after the caller's complete phase peaks. Rank
    /// callers include their transfer buffers in these phases before sizing.
    /// Loads in flight at once; zero in a legacy record means serial.
    pub fn concurrent_loads(&self) -> u64 {
        self.concurrent_loads.max(1)
    }

    /// The owned `encoded_header` buffers a tensor-granularity miss allocates,
    /// one per loader in flight.
    pub fn largest_header_bytes(&self, cache_policy: CachePolicy) -> u64 {
        if cache_policy.granularity != ff_core::weights::CacheGranularity::Tensor {
            return 0;
        }
        self.raw_inventory
            .shards
            .iter()
            .map(|s| s.header_bytes)
            .max()
            .unwrap_or(0)
            .saturating_mul(self.concurrent_loads())
    }

    pub fn automatic_expert_cache_bytes(
        &self,
        phases: &[ResourcePhaseEstimate],
        snapshot: &ResourceSnapshot,
    ) -> Result<usize> {
        self.automatic_expert_cache_bytes_against_host(phases, snapshot, None)
    }

    /// `aggregate_host_bytes` replaces the phase's own host peak under the
    /// fold, for callers whose validator charges a wider host ledger.
    pub fn automatic_expert_cache_bytes_against_host(
        &self,
        phases: &[ResourcePhaseEstimate],
        snapshot: &ResourceSnapshot,
        aggregate_host_bytes: Option<u64>,
    ) -> Result<usize> {
        if self.compute_on_host || self.sparse_layers.is_empty() {
            return Ok(0);
        }
        // On a probed unified-memory topology the cache draws from the same
        // pool as every host charge, so size it from the combined peak instead
        // of the device view alone. Discrete and unprobed captures take the
        // device view exactly as before.
        // A confirmed pool of unknown size retains nothing.
        if snapshot.unified_accounting_is_undecidable() {
            return Ok(0);
        }
        let unified_pool = snapshot.unified_pool_available_bytes();
        let Some(free) = unified_pool.or(snapshot.device_free_memory_bytes) else {
            return Ok(0);
        };
        // The binding charge is the largest per-phase sum, as in
        // `validate_capacity`: maximising each axis charges a phantom phase.
        let required = phases.iter().try_fold(0u64, |peak, phase| {
            let device = phase
                .required_device_bytes
                .unwrap_or(0)
                .checked_add(phase.device_reserve_bytes)
                .context("GLM automatic cache reserve overflow")?;
            let charge = if unified_pool.is_some() {
                let host = match aggregate_host_bytes {
                    Some(aggregate) => aggregate,
                    None => phase.host_peak_bytes()?,
                };
                device
                    .checked_add(host)
                    .context("GLM automatic cache unified charge overflow")?
            } else {
                device
            };
            Ok::<_, anyhow::Error>(peak.max(charge))
        })?;
        // Only the reserve the phases already carry in `device_reserve_bytes`.
        let available = free.saturating_sub(required);
        let available = available / (1 << 20) * (1 << 20);
        let all_experts = (self.live_expert_bytes as u64 / 5)
            .checked_mul(3)
            .and_then(|n| n.checked_mul(self.num_experts as u64))
            .and_then(|n| n.checked_mul(self.sparse_layers.len() as u64))
            .context("GLM expert working set overflow")?;
        usize::try_from(available.min(all_experts)).context("GLM cache bound exceeds usize")
    }

    /// The admission-time safety reserve: 5% of the pool's TOTAL bytes, capped
    /// at the declared 1 GiB. The denominator is the hardware total, not the
    /// instantaneous available view — a busier machine does not get a smaller
    /// safety margin, and sidecar-recorded reserves stay comparable across
    /// runs. The crossover is a 20 GiB pool: below it (including small
    /// discrete cards) the reserve relaxes below 1 GiB. Axis: device total for
    /// CUDA, host total for CPU, the smaller of both totals when unified.
    /// Legacy records without totals fall back to the available views.
    pub fn scaled_admission_safety_bytes(&self, snapshot: &ResourceSnapshot) -> u64 {
        let unified = snapshot.host_device_memory_is_unified == Some(true);
        let total = if self.compute_on_host {
            snapshot
                .host_pool_total_bytes()
                .or(snapshot.host_memory_available_bytes)
        } else if unified {
            [
                snapshot.host_pool_total_bytes(),
                snapshot.device_total_memory_bytes,
            ]
            .into_iter()
            .flatten()
            .min()
            .or_else(|| snapshot.unified_pool_available_bytes())
        } else {
            snapshot
                .device_total_memory_bytes
                .or(snapshot.device_free_memory_bytes)
                .or_else(|| snapshot.host_pool_total_bytes())
                .or(snapshot.host_memory_available_bytes)
        };
        admission_reserve_bytes(total)
    }

    /// The host-promotion reserve, on the same footing as the safety reserve:
    /// 5% of the host (or unified pool) total, capped at 1 GiB. This reserve
    /// gates weight-source promotions; the reverse-scaled `max(1 GiB, pool/20)`
    /// it replaces never shrank and grew with the pool.
    pub fn scaled_promotion_reserve_bytes(snapshot: &ResourceSnapshot) -> u64 {
        let unified = snapshot.host_device_memory_is_unified == Some(true);
        let total = if unified {
            [
                snapshot.host_pool_total_bytes(),
                snapshot.device_total_memory_bytes,
            ]
            .into_iter()
            .flatten()
            .min()
            .or_else(|| snapshot.unified_pool_available_bytes())
        } else {
            snapshot
                .host_pool_total_bytes()
                .or(snapshot.host_memory_available_bytes)
        };
        admission_reserve_bytes(total)
    }

    /// Whether a weight promotion still has its reserve of headroom. The
    /// selector admits a promotion only when every phase plus this reserve
    /// fits, so a later snapshot must be judged by the same rule.
    pub fn promotion_headroom_error(
        &self,
        resident_static: bool,
        expert_cache_bytes: usize,
        cache_policy: CachePolicy,
        snapshot: &ResourceSnapshot,
    ) -> Option<String> {
        let unified = snapshot.unified_pool_available_bytes();
        let host = unified.or_else(|| host_available(snapshot))?;
        let reserve = Self::scaled_promotion_reserve_bytes(snapshot);
        let phases = self
            .phases_with_safety(
                resident_static,
                expert_cache_bytes,
                cache_policy,
                self.scaled_admission_safety_bytes(snapshot),
            )
            .ok()?;
        for phase in phases {
            let mut peak = phase.host_peak_bytes().ok()?;
            if unified.is_some() {
                peak = peak.checked_add(phase.device_peak_bytes().ok()?.unwrap_or(0))?;
            }
            let needed = peak.checked_add(reserve)?;
            if needed > host {
                return Some(format!(
                    "GLM {} promotion needs {needed} bytes including its {reserve} byte reserve, but only {host} are available",
                    phase.phase
                ));
            }
        }
        None
    }

    pub fn validate_capacity(
        &self,
        resident_static: bool,
        expert_cache_bytes: usize,
        cache_policy: CachePolicy,
        snapshot: &ResourceSnapshot,
    ) -> Result<()> {
        // A probed unified-memory topology (e.g. Jetson) charges host and
        // device peaks against one pool; independent per-axis checks would
        // admit a combined footprint the machine cannot hold. `None` (unprobed
        // or legacy record) keeps the split-axis behavior: degrading to
        // discrete checks is unsafe only when the pool is known shared, and
        // legacy records predate unified support entirely.
        let unified_pool = snapshot.unified_pool_available_bytes();
        ensure!(
            !snapshot.unified_accounting_is_undecidable(),
            "GLM admission needs the shared host/device pool size on a unified-memory device, \
             but one of the host and CUDA views could not be measured"
        );
        let safety = self.scaled_admission_safety_bytes(snapshot);
        for phase in
            self.phases_with_safety(resident_static, expert_cache_bytes, cache_policy, safety)?
        {
            let host_peak = phase.host_peak_bytes()?;
            let device_peak = phase.device_peak_bytes()?;
            if let Some(pool) = unified_pool {
                let combined = host_peak
                    .checked_add(device_peak.unwrap_or(0))
                    .context("GLM unified-pool peak overflow")?;
                ensure!(
                    combined <= pool,
                    "GLM {} needs {combined} bytes from the unified host/device pool, but only {pool} are available",
                    phase.phase
                );
                continue;
            }
            let host = host_available(snapshot)
                .context("cannot measure free host memory for GLM admission")?;
            ensure!(
                host_peak <= host,
                "GLM {} needs {host_peak} free host bytes, but only {host} are available",
                phase.phase
            );
            if let Some(required) = device_peak {
                let free = snapshot
                    .device_free_memory_bytes
                    .context("cannot measure free CUDA memory for GLM admission")?;
                ensure!(
                    required <= free,
                    "GLM {} needs {required} free CUDA bytes, but only {free} are available",
                    phase.phase
                );
            }
        }
        Ok(())
    }
}

pub(crate) fn host_mask_bytes(tokens: usize, cpu: bool) -> Result<u64> {
    if cpu {
        return Ok(0);
    }
    u64::try_from(tokens)?
        .checked_mul(u64::try_from(tokens)?)
        .context("GLM host mask size overflow")
}

pub fn host_available(snapshot: &ResourceSnapshot) -> Option<u64> {
    [
        snapshot.host_memory_available_bytes,
        snapshot.cgroup_v2_memory_available_bytes,
    ]
    .into_iter()
    .flatten()
    .min()
}

/// CPU: extra bytes beyond the final F32 compute tensor already charged in
/// its live group. CUDA: compact raw upload staging, with no CPU dequantization.
fn load_host_transient_bytes(
    metadata: &TensorMetadata,
    scale: Option<&TensorMetadata>,
    cpu: bool,
    cpu_fp8_dequantization: bool,
) -> Result<u64> {
    let elements = metadata.shape.iter().try_fold(1_u64, |n, &d| {
        n.checked_mul(u64::try_from(d)?)
            .context("GLM tensor element overflow")
    })?;
    match metadata.dtype.as_str() {
        "F8_E4M3" => {
            let scale = scale.context("GLM FP8 admission requires inverse scale metadata")?;
            ensure!(
                metadata.shape.len() == 2
                    && scale.dtype == "F32"
                    && scale.shape
                        == [
                            metadata.shape[0].div_ceil(128),
                            metadata.shape[1].div_ceil(128)
                        ],
                "GLM admission FP8 scale shape/dtype mismatch"
            );
            if !cpu_fp8_dequantization {
                return elements
                    .checked_add(u64::try_from(scale.bytes)?)
                    .context("GLM compact upload staging overflow");
            }
            let aligned = metadata.shape.iter().all(|d| d.is_multiple_of(128));
            let per_element = (if cpu { 5 } else { 11 }) + if aligned { 0 } else { 8 };
            let index_bytes = if aligned {
                0
            } else {
                u64::try_from(metadata.shape[0])?
                    .checked_add(u64::try_from(metadata.shape[1])?)
                    .and_then(|n| n.checked_mul(4))
                    .context("GLM scale indices overflow")?
            };
            elements
                .checked_mul(per_element)
                .and_then(|n| n.checked_add(scale.bytes as u64))
                .and_then(|n| n.checked_add(index_bytes))
                .context("GLM dequant staging overflow")
        }
        "BF16" => {
            if cpu {
                elements
                    .checked_mul(2)
                    .context("GLM BF16 source copy overflow")
            } else {
                Ok(0)
            }
        }
        "F32" => Ok(0),
        other => bail!("unsupported GLM admission tensor dtype {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::GlmConfig, test_support::tiny_checkpoint};
    use ff_core::probe::{CgroupMemoryLimit, ResourceMeasurementScopes};

    #[test]
    fn mhc_promotion_counts_residency_and_conversion_peaks() -> Result<()> {
        let root = tiny_checkpoint();
        let config = GlmConfig::from_model_dir(root.path())?;
        let estimate = || -> Result<GlmAdmissionBreakdown> {
            let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1))?;
            GlmAdmissionBreakdown::from_metadata(&weights, &config.text_config, false, 2)
        };
        let bf16 = estimate()?;
        let path = root.path().join("model.safetensors");
        let mut tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)?;
        let mut raw_peak = 0;
        for (name, tensor) in &mut tensors {
            if is_mhc_constant(name) && tensor.dtype() == DType::BF16 {
                raw_peak = raw_peak.max((tensor.elem_count() * 2) as u64);
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        candle_core::safetensors::save(&tensors, path)?;
        let f32 = estimate()?;
        assert!(raw_peak > 0);
        assert_eq!(bf16.static_bytes, f32.static_bytes);
        assert_eq!(
            bf16.largest_streamed_group_bytes,
            f32.largest_streamed_group_bytes
        );
        assert_eq!(bf16.static_load_device_bytes, raw_peak);
        assert_eq!(f32.static_load_device_bytes, 0);
        for resident in [false, true] {
            let before = bf16.phases(resident, 0, CachePolicy::new(1))?;
            let after = f32.phases(resident, 0, CachePolicy::new(1))?;
            for (before, after) in before.iter().zip(&after) {
                let transient = if !resident || before.phase == "static_initialization" {
                    raw_peak
                } else {
                    0
                };
                assert_eq!(
                    before.required_device_bytes.unwrap(),
                    after.required_device_bytes.unwrap() + transient
                );
            }
        }
        Ok(())
    }

    #[test]
    fn layer_scopes_partition_static_weights_state_and_global_endpoints() {
        for layers in 1usize..=65 {
            for ranks in 1..=layers {
                let scopes = (0..ranks)
                    .map(|rank| GlmLayerScope::for_rank(layers, rank, ranks).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(scopes[0].start, 0);
                assert_eq!(scopes.last().unwrap().end, layers);
                assert!(scopes.windows(2).all(|p| p[0].end == p[1].start));
                assert!(scopes.iter().all(|s| s.start < s.end));
            }
        }
        assert!(GlmLayerScope::for_rank(2, 0, 3).is_err());
        let root = tiny_checkpoint();
        crate::test_support::quantize_tiny_linears(root.path());
        let w = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let c = GlmConfig::from_model_dir(root.path()).unwrap();
        let whole = GlmAdmissionBreakdown::from_metadata(&w, &c.text_config, false, 3).unwrap();
        let parts = (0..2)
            .map(|rank| {
                GlmAdmissionBreakdown::from_metadata_for_scope(
                    &w,
                    &c.text_config,
                    false,
                    3,
                    false,
                    GlmLayerScope::for_rank(2, rank, 2).unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            parts.iter().map(|p| p.static_bytes).sum::<usize>(),
            whole.static_bytes
        );
        assert_eq!(
            parts.iter().map(|p| p.kda_state_bytes).sum::<usize>(),
            whole.kda_state_bytes
        );
        assert_eq!(
            parts
                .iter()
                .map(|p| p.maximum_dsa_cache_bytes)
                .sum::<usize>(),
            whole.maximum_dsa_cache_bytes
        );
        assert_eq!(parts[0].lm_head_bytes, 0);
        assert_eq!(parts[1].lm_head_bytes, whole.lm_head_bytes);
        assert_eq!(parts[0].host_sampling_workspace_bytes, 0);
        assert_eq!(
            parts[1].host_sampling_workspace_bytes,
            whole.host_sampling_workspace_bytes
        );
        assert!(parts[0].sparse_layers.is_empty());
        assert_eq!(parts[0].live_expert_bytes, 0);
        assert_eq!(parts[1].sparse_layers, whole.sparse_layers);
        let selected = |p: &GlmAdmissionBreakdown| {
            p.raw_inventory
                .shards
                .iter()
                .map(|s| s.selected_tensor_bytes)
                .sum::<u64>()
        };
        assert_eq!(parts.iter().map(selected).sum::<u64>(), selected(&whole));
        assert_eq!(w.access_stats().device_tensor_materializations, 0);
    }

    fn snapshot(host: u64, cgroup: u64, device: Option<u64>) -> ResourceSnapshot {
        ResourceSnapshot {
            schema_version: 1,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(host),
            // `u64::MAX` means "no cgroup constraint" at these call sites. A
            // real probe reports that as `Unlimited` — `memory.max` reading
            // `max` never parses to a finite byte count — so expressing it as
            // `Bytes(u64::MAX)` would hand reserve scaling an 18-exabyte pool.
            cgroup_v2_memory_limit: Some(if cgroup == u64::MAX {
                CgroupMemoryLimit::Unlimited
            } else {
                CgroupMemoryLimit::Bytes(cgroup)
            }),
            cgroup_v2_memory_current_bytes: Some(0),
            cgroup_v2_memory_available_bytes: (cgroup != u64::MAX).then_some(cgroup),
            device_free_memory_bytes: device,
            host_device_memory_is_unified: None,
            device_topology_probe_failed: false,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            measurement_scope: ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        }
    }

    #[test]
    fn metadata_phase_model_includes_static_ranges_and_preserves_empty_payload_cache() {
        let root = tiny_checkpoint();
        crate::test_support::quantize_tiny_linears(root.path());
        let weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let config = GlmConfig::from_model_dir(root.path()).unwrap();
        let cpu =
            GlmAdmissionBreakdown::from_metadata(&weights, &config.text_config, true, 2).unwrap();
        let gpu =
            GlmAdmissionBreakdown::from_metadata(&weights, &config.text_config, false, 2).unwrap();
        let cpu_conversion = GlmAdmissionBreakdown::from_metadata_with_fp8(
            &weights,
            &config.text_config,
            false,
            2,
            true,
        )
        .unwrap();
        assert_eq!(gpu.static_bytes, cpu_conversion.static_bytes);
        assert_eq!(gpu.live_expert_bytes, cpu_conversion.live_expert_bytes);
        assert!(gpu.expert_load_device_bytes > 0);
        assert_eq!(cpu_conversion.expert_load_device_bytes, 0);
        let raw_mhc_bytes = weights
            .metadata("model.language_model.layers.0.hc_attn_fn")
            .unwrap()
            .bytes as u64;
        assert_eq!(cpu_conversion.static_load_device_bytes, raw_mhc_bytes);
        assert!(cpu_conversion.expert_load_host_bytes > gpu.expert_load_host_bytes);
        assert!(cpu.static_bytes > gpu.static_bytes);
        let head = weights.metadata("lm_head.weight").unwrap();
        assert!(
            gpu.raw_inventory
                .largest_unit_bytes(ff_core::weights::CacheGranularity::Tensor)
                >= head.bytes as u64
        );
        let stats = weights.cache_stats();
        assert_eq!((stats.hits, stats.misses, stats.resident_bytes), (0, 0, 0));
        let phases = gpu.phases(true, 10, CachePolicy::new(1)).unwrap();
        assert_eq!(
            phases[0].required_device_bytes,
            Some(gpu.static_bytes as u64 + gpu.static_load_device_bytes)
        );
        let fallback_phases = cpu_conversion
            .phases(true, 10, CachePolicy::new(1))
            .unwrap();
        assert_eq!(
            phases[1].required_device_bytes.unwrap(),
            fallback_phases[1].required_device_bytes.unwrap() + gpu.expert_load_device_bytes
        );
        assert_eq!(
            phases.iter().map(|p| p.phase.as_str()).collect::<Vec<_>>(),
            ["static_initialization", "prefill", "decode"]
        );
        for phase in &phases {
            assert_eq!(phase.device_reserve_bytes, GLM_ADMISSION_SAFETY_BYTES);
            assert!(phase.device_peak_bytes().unwrap().unwrap() >= GLM_ADMISSION_SAFETY_BYTES);
        }
        for phase in cpu.phases(true, 10, CachePolicy::new(1)).unwrap() {
            assert_eq!(phase.device_peak_bytes().unwrap(), None);
            assert!(phase.required_host_bytes >= GLM_ADMISSION_SAFETY_BYTES);
        }
        assert!(
            cpu.validate_capacity(true, 0, CachePolicy::new(1), &snapshot(u64::MAX, 1, None))
                .is_err()
        );
        cpu.validate_capacity(
            true,
            0,
            CachePolicy::new(1),
            &snapshot(u64::MAX, u64::MAX, None),
        )
        .unwrap();
        assert!(
            gpu.validate_capacity(
                true,
                0,
                CachePolicy::new(1),
                &snapshot(u64::MAX, u64::MAX, Some(1))
            )
            .is_err()
        );
    }

    #[test]
    fn unified_pool_charges_host_and_device_peaks_together() {
        let root = tiny_checkpoint();
        crate::test_support::quantize_tiny_linears(root.path());
        let weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let config = GlmConfig::from_model_dir(root.path()).unwrap();
        let gpu =
            GlmAdmissionBreakdown::from_metadata(&weights, &config.text_config, false, 2).unwrap();
        let phases = gpu
            .phases_with_safety(false, 0, CachePolicy::new(1), 0)
            .unwrap();
        let host_peak = phases
            .iter()
            .map(|p| p.host_peak_bytes().unwrap())
            .max()
            .unwrap();
        let device_peak = phases
            .iter()
            .map(|p| p.device_peak_bytes().unwrap().unwrap())
            .max()
            .unwrap();
        assert!(host_peak > 0 && device_peak > 0);
        // validate_capacity applies the shared pool-scaled reserve, so the
        // pool must cover each axis *plus* that reserve in split mode while
        // staying below the merged sum. Iterate to the fixed point: pool =
        // device_peak + reserve(pool) converges in a few steps at these sizes.
        let mut pool = host_peak.max(device_peak);
        for _ in 0..16 {
            let reserve = ff_core::probe::admission_reserve_bytes(Some(pool));
            let split_need = (device_peak + reserve).max(host_peak);
            let merged_need = host_peak + device_peak + reserve;
            if split_need <= pool && merged_need > pool {
                break;
            }
            assert!(split_need > pool, "fixed point not converging");
            pool = split_need;
        }
        // (The reserve alone can exceed the tiny host peak at these fixture
        // sizes, so pool may legitimately exceed host_peak + device_peak;
        // what matters is the split/merged outcome pair below.)
        let unified = |unified: Option<bool>, pool: u64| ResourceSnapshot {
            host_device_memory_is_unified: unified,
            device_topology_probe_failed: false,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            ..snapshot(pool, u64::MAX, Some(pool))
        };
        assert!(
            gpu.validate_capacity(false, 0, CachePolicy::new(1), &unified(Some(true), pool))
                .is_err()
        );
        // Unprobed (None) and confirmed-discrete (Some(false)) records keep the
        // split-axis behavior: the same numbers are admitted.
        for legacy in [None, Some(false)] {
            gpu.validate_capacity(false, 0, CachePolicy::new(1), &unified(legacy, pool))
                .unwrap();
        }
        // A pool that holds the combined peak plus the pool-scaled reserve is
        // admitted — iterate to the merged fixed point, same reserve fn.
        let mut merged_pool = host_peak + device_peak;
        for _ in 0..16 {
            let need = host_peak
                + device_peak
                + ff_core::probe::admission_reserve_bytes(Some(merged_pool));
            if need <= merged_pool {
                break;
            }
            merged_pool = need;
        }
        gpu.validate_capacity(
            false,
            0,
            CachePolicy::new(1),
            &unified(Some(true), merged_pool),
        )
        .unwrap();
    }

    #[test]
    fn safety_reserve_scales_down_on_small_pools_and_never_up() {
        let root = tiny_checkpoint();
        crate::test_support::quantize_tiny_linears(root.path());
        let weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let config = GlmConfig::from_model_dir(root.path()).unwrap();
        let gpu =
            GlmAdmissionBreakdown::from_metadata(&weights, &config.text_config, false, 2).unwrap();
        // Discrete workstation (24 GiB card): the declared 1 GiB is under
        // device_total/20 there.
        let desktop = ResourceSnapshot {
            device_total_memory_bytes: Some(24 << 30),
            ..snapshot(62 << 30, u64::MAX, Some(24 << 30))
        };
        assert_eq!(
            gpu.scaled_admission_safety_bytes(&desktop),
            GLM_ADMISSION_SAFETY_BYTES
        );
        // A mid-size unified pool: 5% of its total (above the floor, below
        // the cap), fixed across runs (not the instantaneous available view).
        let mid = ResourceSnapshot {
            host_device_memory_is_unified: Some(true),
            device_topology_probe_failed: false,
            host_memory_total_bytes: Some(12 << 30),
            device_total_memory_bytes: Some(12 << 30),
            ..snapshot(11 << 30, u64::MAX, Some(11 << 30))
        };
        let scaled = gpu.scaled_admission_safety_bytes(&mid);
        assert_eq!(scaled, (12 << 30) / 20);
        assert!(scaled < GLM_ADMISSION_SAFETY_BYTES);
        // A small pool keeps the floor instead of shrinking past the fixed
        // CUDA-context component the reserve also covers.
        let small = ResourceSnapshot {
            host_memory_total_bytes: Some(7_849_050_112),
            ..mid
        };
        assert_eq!(gpu.scaled_admission_safety_bytes(&small), 512 << 20);
    }

    #[test]
    fn unified_pool_shrinks_automatic_expert_cache_by_host_peaks() {
        // Hand-rolled CUDA breakdown: the all-experts clamp must not bind, so
        // expert geometry is sized large enough to leave the pool arithmetic
        // observable.
        let inventory = ff_core::weights::accounting::CacheInventory {
            shards: vec![ff_core::weights::accounting::CacheShardInventory {
                name: "a.safetensors".into(),
                file_bytes: 108,
                header_bytes: 8,
                selected_tensor_bytes: 100,
                selected_tensor_count: 1,
                largest_tensor_bytes: 100,
            }],
        };
        let breakdown = GlmAdmissionBreakdown {
            scope: GlmLayerScope {
                start: 0,
                end: 2,
                total_layers: 2,
            },
            cpu_fp8_dequantization: false,
            pinned_transfer_bytes: 0,
            pinned_fill_ahead_bytes: 0,
            concurrent_loads: 0,
            static_load_device_bytes: 256 << 20,
            expert_load_device_bytes: 8 << 20,
            compute_on_host: false,
            prompt_tokens: 2,
            num_hidden_layers: 2,
            num_experts: 288,
            experts_per_token: 8,
            sparse_layers: vec![1],
            static_bytes: 1 << 30,
            lm_head_bytes: 64 << 20,
            largest_streamed_group_bytes: 32 << 20,
            kda_state_bytes: 4,
            dsa_cache_bytes_per_token: 4,
            maximum_dsa_cache_bytes: 128,
            maximum_dsa_layer_cache_bytes: 64,
            live_expert_bytes: 80 << 20,
            prefill_workspace_bytes: 128,
            decode_workspace_bytes: 64,
            host_route_workspace_bytes: 32,
            host_sampling_workspace_bytes: 0,
            prefill_host_mask_bytes: 0,
            static_load_host_bytes: 16 << 20,
            streamed_load_host_bytes: 16 << 20,
            expert_load_host_bytes: 4 << 20,
            raw_inventory: inventory,
        };
        let phases = breakdown.phases(false, 0, CachePolicy::new(1)).unwrap();
        let device_required = phases
            .iter()
            .map(|p| p.required_device_bytes.unwrap_or(0) + p.device_reserve_bytes)
            .max()
            .unwrap();
        let host_required = phases
            .iter()
            .map(|p| p.host_peak_bytes().unwrap())
            .max()
            .unwrap();
        assert!(host_required > 0);
        let all_experts = (80u64 << 20) / 5 * 3 * 288;
        // Discrete: sized from the device view alone. The 8 GiB device view
        // keeps the all-experts clamp (13.8 GiB) from binding.
        let discrete = breakdown
            .automatic_expert_cache_bytes(&phases, &snapshot(8 << 30, u64::MAX, Some(8 << 30)))
            .unwrap();
        let expected_discrete = ((8u64 << 30) - device_required) / (1 << 20) * (1 << 20);
        assert!(expected_discrete < all_experts, "clamp must not bind");
        assert_eq!(discrete as u64, expected_discrete);
        // Unified: same numbers, but the host phase peaks are subtracted from
        // the shared pool too, so the result is strictly smaller.
        let unified_snapshot = ResourceSnapshot {
            host_device_memory_is_unified: Some(true),
            device_topology_probe_failed: false,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            ..snapshot(8 << 30, u64::MAX, Some(8 << 30))
        };
        let unified = breakdown
            .automatic_expert_cache_bytes(&phases, &unified_snapshot)
            .unwrap();
        let pool = 8u64 << 30; // min(host, cgroup, device) views
        // Both maxima fall in one phase here, so this fixture alone cannot tell
        // the two formulas apart; the split phases below can.
        let combined = phases
            .iter()
            .map(|p| {
                p.host_peak_bytes().unwrap()
                    + p.required_device_bytes.unwrap_or(0)
                    + p.device_reserve_bytes
            })
            .max()
            .unwrap();
        assert_eq!(combined, device_required + host_required);
        let expected_unified = (pool - combined) / (1 << 20) * (1 << 20);
        assert_eq!(unified as u64, expected_unified);
        assert!(unified < discrete);

        let split = |phase: &str, host: u64, device: u64| ResourcePhaseEstimate {
            phase: phase.into(),
            required_host_bytes: host,
            optional_host_bytes: 0,
            reclaimable_host_bytes: 0,
            host_promotion_reserve_bytes: 0,
            required_device_bytes: Some(device),
            optional_device_bytes: Some(0),
            device_reserve_bytes: 0,
        };
        let skewed = vec![
            split("prefill", 3 << 30, 1 << 30),
            split("decode", 1 << 30, 3 << 30),
        ];
        let sized = breakdown
            .automatic_expert_cache_bytes(&skewed, &unified_snapshot)
            .unwrap() as u64;
        assert_eq!(sized, pool - (4 << 30));
        assert!(sized > pool - (6 << 30));
    }

    #[test]
    fn large_static_fp8_staging_uses_actual_shape_and_cpu_output_is_not_charged_twice() {
        let metadata = |shape: Vec<usize>, dtype: &str, bytes| TensorMetadata {
            name: "static.weight".into(),
            shard: "s.safetensors".into(),
            dtype: dtype.into(),
            shape,
            bytes,
        };
        let static_weight = metadata(vec![4096, 16384], "F8_E4M3", 64 << 20);
        let scale = metadata(vec![32, 128], "F32", 16 << 10);
        assert_eq!(
            load_host_transient_bytes(&static_weight, Some(&scale), false, false).unwrap(),
            (64 << 20) + (16 << 10)
        );
        assert_eq!(
            load_host_transient_bytes(&static_weight, Some(&scale), true, true).unwrap(),
            (320 << 20) + (16 << 10)
        );
        let expert = metadata(vec![2048, 4096], "F8_E4M3", 8 << 20);
        let expert_scale = metadata(vec![16, 32], "F32", 2048);
        assert_eq!(
            load_host_transient_bytes(&expert, Some(&expert_scale), false, false).unwrap(),
            (8 << 20) + 2048
        );
        assert!(
            load_host_transient_bytes(&static_weight, Some(&expert_scale), false, false).is_err()
        );
        assert!(load_host_transient_bytes(&static_weight, None, false, false).is_err());
    }

    #[test]
    fn pinned_buffers_are_charged_to_both_tiers_in_every_phase() {
        let root = crate::test_support::tiny_checkpoint();
        crate::test_support::quantize_tiny_linears(root.path());
        let weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let config = GlmConfig::from_model_dir(root.path()).unwrap();
        let mut breakdown = GlmAdmissionBreakdown::from_metadata_with_fp8(
            &weights,
            &config.text_config,
            false,
            3,
            false,
        )
        .unwrap();
        let before = breakdown.phases(true, 4096, CachePolicy::new(1)).unwrap();
        breakdown.enable_pinned_transfer(1, 0).unwrap();
        let bytes = breakdown.pinned_transfer_bytes;
        assert!(bytes > 0);
        assert_eq!(breakdown.pinned_fill_ahead_bytes, 0);
        for (before, after) in before
            .iter()
            .zip(breakdown.phases(true, 4096, CachePolicy::new(1)).unwrap())
        {
            assert_eq!(
                after.host_peak_bytes().unwrap(),
                before.host_peak_bytes().unwrap() + bytes
            );
            assert_eq!(
                after.device_peak_bytes().unwrap().unwrap(),
                before.device_peak_bytes().unwrap().unwrap() + bytes
            );
        }
        // Every configured lane holds its own slot pair; the fill-ahead ring is
        // pinned host memory alone.
        breakdown.enable_pinned_transfer(3, 4).unwrap();
        let ring = bytes / 2 * 4;
        assert_eq!(breakdown.pinned_transfer_bytes, bytes * 3);
        assert_eq!(breakdown.pinned_fill_ahead_bytes, ring);
        for (before, after) in before
            .iter()
            .zip(breakdown.phases(true, 4096, CachePolicy::new(1)).unwrap())
        {
            assert_eq!(
                after.host_peak_bytes().unwrap(),
                before.host_peak_bytes().unwrap() + bytes * 3 + ring
            );
            assert_eq!(
                after.device_peak_bytes().unwrap().unwrap(),
                before.device_peak_bytes().unwrap().unwrap() + bytes * 3
            );
        }
        assert_eq!(weights.access_stats().device_tensor_materializations, 0);
        breakdown.compute_on_host = true;
        assert!(breakdown.enable_pinned_transfer(1, 0).is_err());
    }

    #[test]
    fn selected_residency_and_tensor_cache_execute_the_same_tokens() {
        let root = tiny_checkpoint();
        let baseline =
            crate::StreamedGlm::open(root.path(), crate::test_support::tiny_options()).unwrap();
        let mut prepared =
            crate::StreamedGlm::prepare(root.path(), crate::test_support::tiny_options()).unwrap();
        let mut policy = prepared.execution_policy().clone();
        policy.resident_static = true;
        policy.weights.granularity = ff_core::weights::CacheGranularity::Tensor;
        policy.weights.cache_bytes = Some(4096);
        policy.weights.cache_shards = usize::MAX as u64;
        prepared.select_execution_policy(&policy).unwrap();
        assert_eq!(prepared.cache_stats().misses, 0);
        let observed = ResourceSnapshot::capture(Some(prepared.device()));
        let selected = prepared.open(1, &observed).unwrap();
        assert_eq!(*selected.execution_policy(), policy);
        let options = crate::GlmGenerationOptions {
            max_new_tokens: 2,
            max_context_tokens: 8,
            reasoning_effort: "low".into(),
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
            progress: false,
        };
        let a = baseline.generate("hello", &options).unwrap();
        let b = selected.generate("hello", &options).unwrap();
        assert_eq!(a.generated_token_ids, b.generated_token_ids);
        assert_eq!(a.text, b.text);
        assert!(selected.resident_static_bytes() > 0);
    }

    #[test]
    fn prepared_open_obeys_supplied_capacity_before_static_preload() {
        let root = tiny_checkpoint();
        let options = crate::test_support::tiny_options().with_resident_static(true);
        let prepared = crate::StreamedGlm::prepare(root.path(), options).unwrap();
        assert_eq!(prepared.cache_stats().resident_bytes, 0);
        assert_eq!(prepared.access_stats().device_tensor_materializations, 0);
        let small = prepared.estimate(1).unwrap();
        let large = prepared.estimate(2).unwrap();
        assert!(large.prefill_workspace_bytes >= small.prefill_workspace_bytes);
        assert!(prepared.estimate(0).is_err());
        let error = prepared
            .open(2, &snapshot(u64::MAX, 0, None))
            .err()
            .unwrap();
        assert!(error.to_string().contains("free host bytes"));
    }
}
