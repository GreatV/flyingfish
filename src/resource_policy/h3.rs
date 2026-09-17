use super::MeasuredScore;
use super::evidence::{EvidenceContext, EvidencePolicy, ResourceEvidence};
use crate::{
    h3::{
        config::TransformerConfig,
        policy::ExecutionPolicy,
        resources::{
            ResourceAssumptions, ResourceBudget, ResourceEstimate, SequenceRows, T2vaGeometry,
            TransformerShape, host_weight_residency_charges,
        },
    },
    runtime::{
        probe::ResourceSnapshot,
        resource_selection::{
            CandidateDisposition, ResourceCandidateObservation, ResourcePhaseEstimate,
            ResourcePolicyMode, ResourceSelectionProvenance, SelectedResourceAxis, SelectionOrigin,
        },
        weights::{WeightSource, accounting::CacheInventory},
    },
};
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

pub struct H3SelectionRequest<'a> {
    pub baseline: &'a ExecutionPolicy,
    pub config: &'a TransformerConfig,
    pub geometry: T2vaGeometry,
    pub rows: SequenceRows,
    pub assumptions: ResourceAssumptions,
    pub inventory: &'a CacheInventory,
    pub budget: ResourceBudget,
    pub snapshot: &'a ResourceSnapshot,
    pub context: &'a EvidenceContext,
    pub mode: ResourcePolicyMode,
    pub explicit_axes: &'a BTreeSet<String>,
    pub evidence: Option<&'a ResourceEvidence>,
    pub locked_origin: Option<SelectionOrigin>,
    /// Inputs already present at capture, restricted to bytes also represented
    /// by prompt_embedding_bytes/latent_state_bytes in the full estimate.
    pub resident_input_bytes: u64,
    pub additional_host_allowance_bytes: u64,
}

pub struct H3Selection {
    pub policy: ExecutionPolicy,
    pub estimate: ResourceEstimate,
    pub provenance: ResourceSelectionProvenance,
}

/// Retain weights with capacity left after modeled compute allocations.
/// Explicit/replayed policies and supplied evidence keep their recorded axes.
pub fn automatic_device_cache(
    request: &H3SelectionRequest<'_>,
    headroom_bytes: u64,
) -> Result<Option<crate::runtime::weights::DeviceCachePolicy>> {
    use crate::h3::policy::ExecutionBackendPolicy;
    use crate::runtime::weights::{CacheGranularity, CudaWeightAllocator, DeviceCachePolicy};
    if request.mode != ResourcePolicyMode::Performance
        || request.locked_origin.is_some()
        || request.evidence.is_some()
        || request.baseline.execution_backend != ExecutionBackendPolicy::Cuda
        || request.baseline.weights.device_cache.is_enabled()
        || request.explicit_axes.contains("weights.device_cache")
    {
        return Ok(None);
    }
    // A unified-memory CUDA device still has a device: retaining weights there
    // avoids re-fetching them from storage, which is the binding term on such
    // machines. (The host-less CPU path never reaches here — the backend check
    // above excludes it.)
    // A confirmed shared pool whose size is unknown retains nothing: the device
    // view would authorize retention past an unmeasured host constraint.
    if request.snapshot.unified_pool_is_unmeasurable() {
        return Ok(None);
    }
    let unified_pool = request.snapshot.unified_pool_available_bytes();
    let Some(free) = unified_pool.or(request.snapshot.device_free_memory_bytes) else {
        return Ok(None);
    };
    // Under the unified fold every charge — including this retention — lands on
    // the host axis, so an explicit `--max-host-mib` bounds it as well. Sizing
    // from the device bound alone would fill the measured pool past that limit,
    // and because the retained policy replaces the baseline before selection,
    // the run is then refused outright instead of choosing a smaller cache.
    let mut capacity = request.budget.max_device_bytes.unwrap_or(free).min(free);
    if unified_pool.is_some()
        && let Some(host_bound) = request.budget.max_host_bytes
    {
        capacity = capacity.min(host_bound);
    }
    let modeled = estimate(request, request.baseline)?;
    let (host, required) = future_peaks(request, &modeled)?;
    // The retention draws from the same pool as every other charge. Under the
    // unified fold the host figure already includes the device peak, so it is
    // the binding subtrahend; on discrete topology the device axis is used as
    // before.
    let subtrahend = if unified_pool.is_some() {
        host
    } else {
        required
    };
    let ceiling = capacity
        .saturating_sub(subtrahend)
        .saturating_sub(headroom_bytes)
        .min(request.inventory.total_bytes(CacheGranularity::Tensor)?);
    let ceiling = ceiling / (1 << 20) * (1 << 20);
    Ok((ceiling > 0).then(|| {
        DeviceCachePolicy::with_max_bytes(ceiling).with_cuda_allocator(CudaWeightAllocator::Direct)
    }))
}

/// Record the budget a boundary was judged against. `refused` selects the
/// refusal ledger: the peaks are known to exceed the budget — that is why the
/// refusal happened — so the admitted-peak invariant must not be applied, and
/// the overshoot is recorded as a shortfall instead of a remainder. Applying
/// the invariant here would replace `AdmissionRefused` with a generic error and
/// lose the candidate ledger the refusal exists to publish.
fn record_budget(
    record: &mut ResourceSelectionProvenance,
    boundary: &str,
    budget: ResourceBudget,
    host_peak: u64,
    compute_peak: u64,
    refused: bool,
) -> Result<()> {
    ensure!(
        refused || budget.check_peaks(host_peak, compute_peak).within_budget,
        "recorded H3 admission budget is below its peak"
    );
    record.workload.insert("budget_record_version".into(), 1);
    for (axis, limit, peak) in [
        ("host", budget.max_host_bytes, host_peak),
        ("compute", budget.max_device_bytes, compute_peak),
    ] {
        let prefix = format!("{boundary}_{axis}");
        record.workload.insert(format!("{prefix}_peak_bytes"), peak);
        record.workload.insert(
            format!("{prefix}_budget_is_bounded"),
            u64::from(limit.is_some()),
        );
        record.workload.remove(&format!("{prefix}_budget_bytes"));
        record.workload.remove(&format!("{prefix}_remaining_bytes"));
        record.workload.remove(&format!("{prefix}_shortfall_bytes"));
        if let Some(limit) = limit {
            record
                .workload
                .insert(format!("{prefix}_budget_bytes"), limit);
            match limit.checked_sub(peak) {
                Some(remaining) => {
                    record
                        .workload
                        .insert(format!("{prefix}_remaining_bytes"), remaining);
                }
                None => {
                    ensure!(refused, "recorded H3 admission budget is below its peak");
                    record
                        .workload
                        .insert(format!("{prefix}_shortfall_bytes"), peak - limit);
                }
            }
        }
    }
    Ok(())
}

/// Keep the effective limits and checked peaks with the observation that used
/// them. Compute limits also apply to the compute portion of a CPU host ledger.
pub fn record_final_admission(
    record: &mut ResourceSelectionProvenance,
    snapshot: ResourceSnapshot,
    budget: ResourceBudget,
    host_peak: u64,
    compute_peak: u64,
) -> Result<()> {
    record_budget(
        record,
        "final_admission",
        budget,
        host_peak,
        compute_peak,
        false,
    )?;
    record.final_admission_snapshot = Some(snapshot);
    Ok(())
}

pub fn axes(policy: &ExecutionPolicy) -> Result<BTreeMap<String, String>> {
    let p = serde_json::to_value(&policy.weights)?;
    Ok([
        "source",
        "cache_bytes",
        "cache_shards",
        "granularity",
        "device_cache",
        "host_phase_priority",
    ]
    .into_iter()
    .map(|name| {
        let value = &p[name];
        (
            format!("weights.{name}"),
            if name == "granularity" && value.is_null() {
                "shard".into()
            } else {
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            },
        )
    })
    .collect())
}

pub fn estimate(
    request: &H3SelectionRequest<'_>,
    policy: &ExecutionPolicy,
) -> Result<ResourceEstimate> {
    let charge = host_weight_residency_charges(policy, request.inventory)?;
    let mut assumptions = request.assumptions;
    assumptions.device_weight_cache_bytes = policy.weights.device_cache.max_bytes;
    assumptions.host_weight_cache_bytes = charge
        .owned_weight_bytes
        .checked_add(request.additional_host_allowance_bytes)
        .context("H3 host allowance overflow")?;
    assumptions.mapped_weight_residency_bytes = charge.mapped_weight_bytes;
    assumptions.checkpoint_weight_bytes = Some(
        request
            .inventory
            .total_bytes(crate::runtime::weights::CacheGranularity::Tensor)?,
    );
    ResourceEstimate::for_shape_rows(
        TransformerShape::from_config(request.config),
        request.geometry,
        assumptions,
        request.rows,
    )
}

fn future_peaks(
    request: &H3SelectionRequest<'_>,
    estimate: &ResourceEstimate,
) -> Result<(u64, u64)> {
    let represented = estimate
        .activations
        .prompt_embedding_bytes
        .checked_add(estimate.activations.latent_state_bytes)
        .context("H3 input residency overflow")?;
    ensure!(
        request.resident_input_bytes <= represented,
        "input residency credit exceeds the modeled inputs"
    );
    let device = estimate
        .peak_device_bytes
        .checked_sub(request.resident_input_bytes)
        .context("H3 input credit exceeds device estimate")?;
    let host = if request.assumptions.device_memory_is_host {
        estimate
            .peak_host_bytes
            .checked_sub(request.resident_input_bytes)
            .context("H3 input credit exceeds host estimate")?
    } else {
        estimate.peak_host_bytes
    };
    Ok((host, device))
}

pub fn select(request: H3SelectionRequest<'_>) -> Result<H3Selection> {
    request.baseline.validate()?;
    request.context.validate()?;

    let base_axes = axes(request.baseline)?;
    let mut candidates = vec![request.baseline.clone()];
    if request.mode == ResourcePolicyMode::Performance
        && request.locked_origin.is_none()
        && let Some(record) = request.evidence
    {
        record.validate()?;
        for row in &record.candidates {
            if let EvidencePolicy::H3(policy) = &row.candidate {
                candidates.push(policy.clone());
            }
        }
    }
    // A candidate's number labels it inside this record; two candidates are the
    // same when their policies are equal, which is what `seen` compares.
    let mut seen: Vec<ExecutionPolicy> = Vec::new();
    let mut observations: Vec<ResourceCandidateObservation> = Vec::new();
    let mut selected: Option<(
        ExecutionPolicy,
        ResourceEstimate,
        usize,
        Option<MeasuredScore>,
    )> = None;
    for policy in candidates {
        if seen.contains(&policy) {
            continue;
        }
        seen.push(policy.clone());
        let is_baseline = policy == *request.baseline;
        let candidate_id = format!("candidate-{}", seen.len() - 1);
        let mut observation = ResourceCandidateObservation {
            candidate_id,
            disposition: CandidateDisposition::NoBenefitEvidence,
            reason: "baseline_no_benefit_evidence".into(),
            expected_cost: None,
            evidence: vec![],
        };
        let candidate_axes = axes(&policy)?;
        let mut invariant = policy.clone();
        invariant.weights = request.baseline.weights.clone();
        if invariant != *request.baseline
            || policy.weights.host_phase_priority != request.baseline.weights.host_phase_priority
            || policy.weights.device_cache != request.baseline.weights.device_cache
            || policy.weights.granularity != request.baseline.weights.granularity
            || request
                .explicit_axes
                .iter()
                .any(|a| base_axes.get(a) != candidate_axes.get(a))
        {
            observation.disposition = CandidateDisposition::OperatorExcluded;
            observation.reason =
                "candidate changes an explicit, numerical or nonautomatic axis".into();
            observations.push(observation);
            continue;
        }
        let charge = host_weight_residency_charges(&policy, request.inventory)?;
        if !is_baseline && policy.weight_source() == WeightSource::Mmap {
            let cache = policy.cache_policy()?;
            let two_shards = request
                .inventory
                .largest_unit_bytes(crate::runtime::weights::CacheGranularity::Shard)
                .checked_mul(2)
                .context("two-shard retention size overflow")?;
            let baseline_charge =
                host_weight_residency_charges(request.baseline, request.inventory)?;
            if cache.max_shards < 2
                || cache.max_bytes.is_some_and(|bytes| bytes < two_shards)
                || charge.retained_bytes <= baseline_charge.retained_bytes
            {
                observation.disposition = CandidateDisposition::OperatorExcluded;
                observation.reason = "automatic mmap retention must increase the cache and retain at least two shards without the oversized-unit exception".into();
                observations.push(observation);
                continue;
            }
        }
        if !is_baseline
            && policy.weight_source() == WeightSource::Memory
            && !charge.complete_set_fits
        {
            observation.disposition = CandidateDisposition::OperatorExcluded;
            observation.reason = "automatic Memory requires complete actual-unit retention".into();
            observations.push(observation);
            continue;
        }
        let modeled = estimate(&request, &policy)?;
        let (host, device) = future_peaks(&request, &modeled)?;
        let admitted = request.budget.check_peaks(host, device);
        if !admitted.within_budget {
            observation.disposition = CandidateDisposition::CapacityRejected;
            observation.reason = format!(
                "H3 requires host {host} and compute {device} bytes; configured bounds are {:?} / {:?}",
                request.budget.max_host_bytes, request.budget.max_device_bytes
            );
            observations.push(observation);
            continue;
        }
        if !is_baseline {
            let available = request
                .budget
                .max_host_bytes
                .context("automatic host retention requires measured or explicit capacity")?;
            let reserve = (1_u64 << 30).max(available / 20);
            if host.checked_add(reserve).is_none_or(|n| n > available) {
                observation.disposition = CandidateDisposition::CapacityRejected;
                observation.reason = "host retention would consume the promotion reserve".into();
                observations.push(observation);
                continue;
            }
        }
        let cost = if is_baseline {
            if request.locked_origin.is_some() {
                observation.reason = "replay sealed policy without resource reselection".into();
            } else if request.mode == ResourcePolicyMode::Conservative {
                observation.reason = "operator selected conservative defaults".into();
            }
            None
        } else {
            let Some(record) = request.evidence else {
                observations.push(observation);
                continue;
            };
            let row = record.candidates.iter().find(
                |r| matches!(&r.candidate, EvidencePolicy::H3(candidate) if *candidate == policy),
            );
            let context_difference = record.context.first_difference(request.context);
            let baseline_matches = row.is_some_and(
                |r| matches!(&r.baseline_policy, EvidencePolicy::H3(base) if base == request.baseline),
            );
            if context_difference.is_some() || !baseline_matches {
                observation.disposition = CandidateDisposition::EvidenceMismatch;
                observation.reason = match context_difference {
                    Some(field) => format!("evidence was measured under a different {field}"),
                    None => "evidence does not name this baseline policy".into(),
                };
                observations.push(observation);
                continue;
            }
            let row = row.unwrap();
            observation.evidence.push(serde_json::json!({
                "pairs": row.pairs.len(),
                "median_us": row.median_us(),
                "minimum_improvement_basis_points": row.minimum_improvement_basis_points,
            }));
            if !row.qualifies()
                || row.observed_peak_deltas.is_none_or(|(h, d)| {
                    // `device_memory_is_host` now marks two different things:
                    // the host-only CPU path, where a device observation cannot
                    // belong to the trial at all, and the unified fold, where a
                    // real CUDA device records one and both axes draw from the
                    // same pool. Rejecting the fold for merely having a device
                    // delta marks every qualified CUDA candidate a regression.
                    let unified = request.snapshot.host_device_memory_is_unified == Some(true);
                    h > host
                        || if unified {
                            d.is_none_or(|d| h.saturating_add(d) > host)
                        } else if request.assumptions.device_memory_is_host {
                            d.is_some()
                        } else {
                            d.is_none_or(|d| d > device)
                        }
                })
            {
                observation.disposition = CandidateDisposition::KnownRegression;
                observation.reason =
                    "paired timing/output/peak observations do not qualify this candidate".into();
                observations.push(observation);
                continue;
            }
            observation.reason = format!("qualified paired command median {} us", row.median_us());
            Some(MeasuredScore {
                median_us: row.median_us(),
                uncertainty_us: row.pairs.iter().map(|p| p.candidate_wall_us).max().unwrap()
                    - row.pairs.iter().map(|p| p.candidate_wall_us).min().unwrap(),
                retained_peak_bytes: u128::from(host)
                    + if request.assumptions.device_memory_is_host {
                        0
                    } else {
                        u128::from(device)
                    },
            })
        };
        if selected
            .as_ref()
            .is_none_or(|(_, _, _, old)| match (cost, old) {
                (Some(a), Some(b)) => a.prefers(*b),
                (Some(_), None) => true,
                _ => false,
            })
        {
            if let Some((_, _, index, _)) = &selected {
                observations[*index].disposition = CandidateDisposition::Slower;
            }
            observation.disposition = CandidateDisposition::Selected;
            selected = Some((policy, modeled, observations.len(), cost));
        } else {
            observation.disposition = CandidateDisposition::Slower;
        }
        observations.push(observation);
    }
    let Some((policy, estimate, _, _)) = selected else {
        // A refusal must still ship its numbers. The record names the
        // baseline (that is what was evaluated) and marks refused=1.
        let baseline_estimate = estimate(&request, request.baseline)?;
        let provenance = build_provenance(
            &request,
            request.baseline,
            &baseline_estimate,
            observations,
            true,
        )?;
        let summary = provenance
            .candidates
            .iter()
            .map(|r| r.reason.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(super::AdmissionRefused {
            provenance,
            summary: format!("H3 resource admission refused: {summary}"),
        }
        .into());
    };
    let provenance = build_provenance(&request, &policy, &estimate, observations, false)?;
    Ok(H3Selection {
        policy,
        estimate,
        provenance,
    })
}

/// Assemble the selection record for a policy: its denoise phase, axis
/// origins, the budget ledger, and every candidate observation. `refused`
/// marks a record whose baseline was evaluated but admitted nothing.
fn build_provenance(
    request: &H3SelectionRequest<'_>,
    policy: &ExecutionPolicy,
    estimate: &crate::h3::resources::ResourceEstimate,
    observations: Vec<ResourceCandidateObservation>,
    refused: bool,
) -> Result<ResourceSelectionProvenance> {
    let base_axes = axes(request.baseline)?;
    let (host, device) = future_peaks(request, estimate)?;
    let retained = host_weight_residency_charges(policy, request.inventory)?.peak_storage_bytes;
    let promoted = !refused && policy != request.baseline;
    let phase = ResourcePhaseEstimate {
        phase: "denoise".into(),
        required_host_bytes: host
            .checked_sub(retained)
            .context("H3 retained bytes exceed host peak")?,
        optional_host_bytes: retained,
        reclaimable_host_bytes: 0,
        host_promotion_reserve_bytes: if promoted {
            (1 << 30).max(request.budget.max_host_bytes.unwrap_or(0) / 20)
        } else {
            0
        },
        required_device_bytes: (!request.assumptions.device_memory_is_host)
            .then_some(device - policy.weights.device_cache.max_bytes),
        optional_device_bytes: (!request.assumptions.device_memory_is_host)
            .then_some(policy.weights.device_cache.max_bytes),
        device_reserve_bytes: 0,
    };
    let selected_axes = axes(policy)?
        .into_iter()
        .map(|(axis, value)| {
            let origin = request.locked_origin.unwrap_or_else(|| {
                if request.explicit_axes.contains(&axis) {
                    SelectionOrigin::OperatorExplicit
                } else if base_axes.get(&axis) != Some(&value) {
                    SelectionOrigin::MeasuredEvidence
                } else {
                    SelectionOrigin::Baseline
                }
            });
            SelectedResourceAxis {
                axis,
                value,
                origin,
            }
        })
        .collect();
    let mut provenance = ResourceSelectionProvenance {
        schema_version: 1,
        policy: serde_json::to_value(policy)?,
        selector_revision: "h3-resource-measured-v1".into(),
        request: request.context.request.clone(),
        input: None,
        model: serde_json::to_value(&request.context.model)?,
        hardware: Some(serde_json::to_value(&request.context.hardware)?),
        executable: request
            .context
            .executable_metadata
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?,
        environment: Some(serde_json::to_value(&request.context.environment)?),
        mode: request.mode,
        selection_snapshot: request.snapshot.clone(),
        final_admission_snapshot: None,
        already_present_at_capture: vec![format!(
            "metadata and {0} represented input bytes",
            request.resident_input_bytes
        )],
        inventory: request.inventory.clone(),
        phases: vec![phase],
        workload: BTreeMap::from([
            ("text_rows".into(), request.rows.text),
            ("video_rows".into(), request.rows.video),
            ("audio_rows".into(), request.rows.audio),
            ("packed_rows".into(), request.rows.total),
            ("timestep_rows".into(), request.assumptions.timestep_rows),
            ("evaluations".into(), request.assumptions.evaluation_count),
            ("resident_input_bytes".into(), request.resident_input_bytes),
            (
                "latent_frames".into(),
                request.geometry.latent_frames as u64,
            ),
            (
                "latent_height".into(),
                request.geometry.latent_height as u64,
            ),
            ("latent_width".into(), request.geometry.latent_width as u64),
            ("audio_frames".into(), request.geometry.audio_frames as u64),
            (
                "audio_channels".into(),
                request.geometry.audio_channels as u64,
            ),
            (
                "additional_host_allowance_bytes".into(),
                request.additional_host_allowance_bytes,
            ),
            ("refused".into(), u64::from(refused)),
        ]),
        axes: selected_axes,
        candidates: observations,
    };
    record_budget(
        &mut provenance,
        "selection",
        request.budget,
        host,
        device,
        refused,
    )?;
    provenance.validate()?;
    Ok(provenance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        h3::{core::AttentionChunking, model::TransformerChunking},
        resource_policy::evidence::{EvidenceArtifact, MeasuredCandidate, PairedObservation},
        runtime::{
            probe::ResourceMeasurementScopes,
            weights::{CachePolicy, accounting::CacheShardInventory},
        },
    };
    use candle_core::Device;
    fn config() -> TransformerConfig {
        TransformerConfig {
            class_name: "MiniMaxH3Transformer3DModel".into(),
            num_attention_heads: 1,
            attention_head_dim: 6,
            hidden_size: 4,
            num_layers: 2,
            num_refiner_layers: 1,
            ffn_dim: 5,
            in_channels: 1,
            audio_in_channels: 1,
            patch_size: [1, 1, 1],
            text_dim: 2,
            freq_dim: 1,
            time_embed_hidden_dim: 2,
            time_embed_dim: 2,
            rope_freq_dim: 1,
            rope_theta: 10000.,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }
    fn baseline() -> ExecutionPolicy {
        ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            TransformerChunking {
                attention: AttentionChunking {
                    projection_chunk_size: 1.try_into().unwrap(),
                    query_chunk_size: 1.try_into().unwrap(),
                    key: crate::h3::core::AttentionKeyChunkPolicy::Full,
                },
                feed_forward_chunk_size: 1.try_into().unwrap(),
                output_chunk_size: 1.try_into().unwrap(),
            },
            false,
            true,
        )
        .unwrap()
    }
    fn geometry() -> T2vaGeometry {
        T2vaGeometry {
            text_rows: 1,
            latent_frames: 1,
            latent_height: 1,
            latent_width: 1,
            audio_frames: 1,
            audio_channels: 1,
            attention_projection_chunk_size: 1,
            attention_query_chunk_size: 1,
            attention_key_chunk_policy: crate::h3::core::AttentionKeyChunkPolicy::Full,
            ffn_token_chunk_size: 1,
            output_token_chunk_size: 1,
        }
    }
    fn context() -> EvidenceContext {
        crate::resource_policy::evidence::tests::sample_context()
    }
    fn snapshot() -> ResourceSnapshot {
        ResourceSnapshot {
            schema_version: 1,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(8 << 30),
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: None,
            device_free_memory_bytes: None,
            host_device_memory_is_unified: None,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            measurement_scope: ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        }
    }
    fn inventory() -> CacheInventory {
        CacheInventory {
            shards: ["a", "b"]
                .into_iter()
                .map(|n| CacheShardInventory {
                    name: format!("{n}.safetensors"),
                    file_bytes: 108,
                    header_bytes: 8,
                    selected_tensor_bytes: 100,
                    selected_tensor_count: 1,
                    largest_tensor_bytes: 100,
                })
                .collect(),
        }
    }
    fn run(
        policy: &ExecutionPolicy,
        locked: Option<SelectionOrigin>,
        evidence: Option<&ResourceEvidence>,
        credit: u64,
    ) -> Result<H3Selection> {
        run_with_budget(
            policy,
            locked,
            evidence,
            credit,
            ResourceBudget {
                max_host_bytes: Some(8 << 30),
                max_device_bytes: None,
            },
        )
    }
    fn run_with_budget(
        policy: &ExecutionPolicy,
        locked: Option<SelectionOrigin>,
        evidence: Option<&ResourceEvidence>,
        credit: u64,
        budget: ResourceBudget,
    ) -> Result<H3Selection> {
        let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
        assumptions.device_memory_is_host = true;
        assumptions.weight_element_bytes = 4;
        assumptions.activation_element_bytes = 4;
        select(H3SelectionRequest {
            baseline: policy,
            config: &config(),
            geometry: geometry(),
            rows: geometry().sequence_rows([1, 1, 1])?,
            assumptions,
            inventory: &inventory(),
            budget,
            snapshot: &snapshot(),
            context: &context(),
            mode: ResourcePolicyMode::Performance,
            explicit_axes: &BTreeSet::new(),
            evidence,
            locked_origin: locked,
            resident_input_bytes: credit,
            additional_host_allowance_bytes: 0,
        })
    }
    #[test]
    fn refusal_publishes_its_ledger_instead_of_a_budget_recording_error() {
        // A budget below the baseline peak is the ordinary reason selection
        // refuses. The refusal must still reach `AdmissionRefused` carrying the
        // candidate ledger: recording the budget must not apply the invariant
        // that peaks fit, since the whole point is that they do not.
        let error = run_with_budget(
            &baseline(),
            None,
            None,
            0,
            ResourceBudget {
                max_host_bytes: Some(1),
                max_device_bytes: Some(1),
            },
        )
        .err()
        .expect("a budget below the baseline peak must refuse");
        let refusal = error
            .downcast_ref::<super::super::AdmissionRefused>()
            .expect("refusal must carry its record, not a budget-recording error");
        assert_eq!(refusal.provenance.workload.get("refused"), Some(&1));
        assert!(!refusal.provenance.candidates.is_empty());
        assert!(
            refusal
                .provenance
                .candidates
                .iter()
                .all(|c| c.disposition != CandidateDisposition::Selected)
        );
        // The overshoot is recorded as a shortfall, where an admitted record
        // would carry a remainder.
        assert!(
            refusal
                .provenance
                .workload
                .contains_key("selection_host_shortfall_bytes")
        );
        assert!(
            !refusal
                .provenance
                .workload
                .contains_key("selection_host_remaining_bytes")
        );
        refusal.provenance.validate().unwrap();
    }

    #[test]
    fn explicit_device_retention_is_charged_before_admission_and_not_auto_enabled() {
        let base = baseline();
        let old = run(&base, None, None, 0).unwrap();
        let mut cached = base.clone();
        cached.weights.device_cache =
            crate::runtime::weights::DeviceCachePolicy::with_max_bytes(4096);
        let selected = run(&cached, Some(SelectionOrigin::Pinned), None, 0).unwrap();
        assert_eq!(
            selected.estimate.peak_host_bytes,
            old.estimate.peak_host_bytes + 4096
        );
        assert_eq!(
            selected.estimate.peak_device_bytes,
            old.estimate.peak_device_bytes + 4096
        );
        assert!(
            run_with_budget(
                &cached,
                None,
                None,
                0,
                ResourceBudget {
                    max_host_bytes: Some(old.estimate.peak_host_bytes),
                    max_device_bytes: None,
                }
            )
            .is_err()
        );
        let mut e = evidence();
        if let EvidencePolicy::H3(p) = &mut e.candidates[0].candidate {
            p.weights.device_cache = cached.weights.device_cache;
        }
        let excluded = run(&base, None, Some(&e), 0).unwrap();
        assert_eq!(excluded.policy, base);
        assert!(
            excluded
                .provenance
                .candidates
                .iter()
                .any(|c| c.disposition == CandidateDisposition::OperatorExcluded)
        );
    }

    #[test]
    fn unified_cuda_keeps_automatic_device_cache_sized_from_folded_host_peak() {
        // A unified-memory CUDA device still has a device; the CPU-shaped
        // device_memory_is_host flag must not disable retention that avoids
        // re-fetching weights from storage (the binding term there).
        let mut base = baseline();
        base.execution_backend = crate::h3::policy::ExecutionBackendPolicy::Cuda;
        *base.numerics = crate::h3::policy::H3NumericalContract::for_target(
            crate::h3::policy::ExecutionBackendPolicy::Cuda,
            Some(crate::h3::policy::CudaCapabilities::NONE),
            crate::h3::policy::AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        let mut unified = snapshot();
        unified.device_free_memory_bytes = Some(8 << 30);
        unified.host_device_memory_is_unified = Some(true);
        let inventory = CacheInventory {
            shards: vec![CacheShardInventory {
                name: "w.safetensors".into(),
                file_bytes: 80 << 20,
                header_bytes: 8,
                selected_tensor_bytes: 64 << 20,
                selected_tensor_count: 1,
                largest_tensor_bytes: 64 << 20,
            }],
        };
        let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
        assumptions.device_memory_is_host = true;
        let geometry = geometry();
        let request = H3SelectionRequest {
            baseline: &base,
            config: &config(),
            rows: geometry.sequence_rows([1, 1, 1]).unwrap(),
            geometry,
            assumptions,
            inventory: &inventory,
            budget: ResourceBudget {
                max_host_bytes: Some(8 << 30),
                max_device_bytes: Some(8 << 30),
            },
            snapshot: &unified,
            context: &context(),
            mode: ResourcePolicyMode::Performance,
            explicit_axes: &BTreeSet::new(),
            evidence: None,
            locked_origin: None,
            resident_input_bytes: 0,
            additional_host_allowance_bytes: 0,
        };
        let cache = automatic_device_cache(&request, 0).unwrap();
        assert_eq!(
            cache.map(|c| c.max_bytes),
            Some(64 << 20),
            "inventory clamp bounds the folded-pool ceiling"
        );
    }

    #[test]
    fn host_priority_needs_explicit_policy_instead_of_host_source_evidence() {
        let mut base = baseline();
        base.weights.granularity = crate::runtime::weights::CacheGranularity::Tensor;
        base.weights.cache_bytes = Some(216);
        let mut e = evidence();
        e.candidates[0].baseline_policy = EvidencePolicy::H3(base.clone());
        if let EvidencePolicy::H3(p) = &mut e.candidates[0].candidate {
            p.weights.granularity = base.weights.granularity;
            p.weights.host_phase_priority = true;
        }
        let selection = run(&base, None, Some(&e), 0).unwrap();
        assert_eq!(selection.policy, base);
        assert!(
            selection
                .provenance
                .candidates
                .iter()
                .any(|c| c.disposition == CandidateDisposition::OperatorExcluded
                    && c.reason.contains("nonautomatic axis"))
        );
    }

    #[test]
    fn effective_budget_is_recorded_independently_of_snapshot_and_policy() {
        let policy = baseline();
        for host_limit in [4 << 30, 16 << 30] {
            let selected = run_with_budget(
                &policy,
                None,
                None,
                0,
                ResourceBudget {
                    max_host_bytes: Some(host_limit),
                    max_device_bytes: Some(1 << 30),
                },
            )
            .unwrap();
            let record = selected.provenance;
            assert_eq!(record.selection_snapshot, snapshot());
            assert_eq!(record.policy, serde_json::to_value(&policy).unwrap());
            assert_eq!(record.workload["selection_host_budget_bytes"], host_limit);
            assert_eq!(
                record.workload["selection_host_remaining_bytes"],
                host_limit - record.phases[0].host_peak_bytes().unwrap()
            );
            assert_eq!(record.workload["selection_compute_budget_bytes"], 1 << 30);
            assert_eq!(
                record.workload["selection_compute_remaining_bytes"],
                (1 << 30) - selected.estimate.peak_device_bytes
            );
            let restored =
                ResourceSelectionProvenance::from_json(&record.canonical_json().unwrap()).unwrap();
            assert_eq!(restored.workload, record.workload);
        }
    }

    #[test]
    fn final_admission_replaces_budget_with_its_own_observation() {
        let mut record = run(&baseline(), None, None, 0).unwrap().provenance;
        let mut final_snapshot = snapshot();
        final_snapshot.measured_at_unix_ms = 2;
        final_snapshot.host_memory_available_bytes = Some(3 << 30);
        record_final_admission(
            &mut record,
            final_snapshot.clone(),
            ResourceBudget {
                max_host_bytes: Some(1 << 30),
                max_device_bytes: Some(1 << 29),
            },
            1024,
            2048,
        )
        .unwrap();
        assert_eq!(
            record.final_admission_snapshot,
            Some(final_snapshot.clone())
        );
        assert_eq!(record.workload["selection_host_budget_bytes"], 8 << 30);
        assert_eq!(
            record.workload["final_admission_host_budget_bytes"],
            1 << 30
        );
        assert_eq!(
            record.workload["final_admission_host_remaining_bytes"],
            (1 << 30) - 1024
        );
        let before_refusal = record.canonical_json().unwrap();
        assert!(
            record_final_admission(
                &mut record,
                final_snapshot.clone(),
                ResourceBudget {
                    max_host_bytes: Some(0),
                    max_device_bytes: None
                },
                1024,
                2048,
            )
            .is_err()
        );
        assert_eq!(record.canonical_json().unwrap(), before_refusal);
        record_final_admission(
            &mut record,
            final_snapshot,
            ResourceBudget {
                max_host_bytes: None,
                max_device_bytes: None,
            },
            1024,
            2048,
        )
        .unwrap();
        assert_eq!(record.workload["final_admission_host_budget_is_bounded"], 0);
        assert_eq!(
            record.workload["final_admission_compute_budget_is_bounded"],
            0
        );
        assert!(
            !record
                .workload
                .contains_key("final_admission_host_budget_bytes")
        );
        assert!(
            !record
                .workload
                .contains_key("final_admission_compute_remaining_bytes")
        );
        assert_eq!(record.policy, serde_json::to_value(baseline()).unwrap());
    }

    fn evidence() -> ResourceEvidence {
        let base = baseline();
        let mut candidate = base.clone();
        candidate.weights.source = crate::h3::policy::WeightSourcePolicy::Memory;
        candidate.weights.cache_bytes = Some(216);
        ResourceEvidence {
            schema_version: 1,
            context: context(),
            candidates: vec![MeasuredCandidate {
                baseline_policy: EvidencePolicy::H3(base.clone()),
                candidate: EvidencePolicy::H3(candidate),
                minimum_improvement_basis_points: 200,
                pairs: (0..3)
                    .map(|n| PairedObservation {
                        baseline_wall_us: 1000,
                        candidate_wall_us: 500,
                        baseline_record: EvidenceArtifact {
                            file: format!("b{n}.json"),
                            bytes: 20 + n,
                        },
                        candidate_record: EvidenceArtifact {
                            file: format!("c{n}.json"),
                            bytes: 30 + n,
                        },
                        // `load` compares the retained outputs; this fixture
                        // states the outcome it stands in for.
                        outputs_match: true,
                    })
                    .collect(),
                routing_trace: None,
                routing_replay: None,
                observed_peak_deltas: Some((100, None)),
                routing_verified: false,
                routing_profile: None,
            }],
        }
    }
    #[test]
    fn measured_complete_memory_can_promote_but_pinned_and_recorded_policies_replay() {
        let base = baseline();
        assert_eq!(run(&base, None, None, 0).unwrap().policy, base);
        let evidence = evidence();
        assert_eq!(
            run(&base, None, Some(&evidence), 0)
                .unwrap()
                .policy
                .weight_source(),
            WeightSource::Memory
        );
        for locked in [SelectionOrigin::Pinned, SelectionOrigin::Recorded] {
            let selected = run(&base, Some(locked), Some(&evidence), 0).unwrap();
            assert_eq!(selected.policy, base);
            assert!(selected.provenance.axes.iter().all(|a| a.origin == locked));
        }
    }
    #[test]
    fn measured_mmap_candidate_still_needs_two_shard_retention() {
        let mut e = evidence();
        if let EvidencePolicy::H3(p) = &mut e.candidates[0].candidate {
            p.weights.source = crate::h3::policy::WeightSourcePolicy::Mmap;
            p.weights.cache_bytes = Some(108);
        }
        assert_eq!(
            run(&baseline(), None, Some(&e), 0).unwrap().policy,
            baseline()
        );
        if let EvidencePolicy::H3(p) = &mut e.candidates[0].candidate {
            p.weights.cache_bytes = Some(216);
        }
        assert_eq!(
            run(&baseline(), None, Some(&e), 0)
                .unwrap()
                .policy
                .weights
                .cache_bytes,
            Some(216)
        );
    }

    #[test]
    fn payload_only_memory_bound_and_numerical_changes_cannot_promote() {
        let mut e = evidence();
        if let EvidencePolicy::H3(p) = &mut e.candidates[0].candidate {
            p.weights.cache_bytes = Some(200);
        }
        assert_eq!(
            run(&baseline(), None, Some(&e), 0).unwrap().policy,
            baseline()
        );
        let mut e = evidence();
        if let EvidencePolicy::H3(p) = &mut e.candidates[0].candidate {
            p.precompute_adaln = false;
        }
        assert_eq!(
            run(&baseline(), None, Some(&e), 0).unwrap().policy,
            baseline()
        );
        assert!(run(&baseline(), None, None, u64::MAX).is_err());
    }
}
