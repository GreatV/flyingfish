use super::evidence::{EvidenceContext, EvidencePolicy, ResourceEvidence};
use crate::{
    glm::{GlmExecutionPolicy, admission::GlmAdmissionBreakdown},
    runtime::{
        probe::ResourceSnapshot,
        resource_selection::{
            CandidateDisposition, CapacityShortfall, ResourceCandidateObservation,
            ResourcePhaseEstimate, ResourcePolicyMode, ResourceSelectionProvenance,
            SelectedResourceAxis, SelectionOrigin,
        },
        weights::CachePolicy,
    },
};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};

use super::MeasuredScore;

pub struct GlmSelection {
    pub policy: GlmExecutionPolicy,
    pub provenance: ResourceSelectionProvenance,
}

pub fn cache_policy(policy: &GlmExecutionPolicy) -> Result<CachePolicy> {
    policy.cache_policy()
}

/// The host peak a promotion phase draws from the pool it is judged against,
/// with the device peak folded in under a unified topology.
fn promotion_peak(phase: &ResourcePhaseEstimate, unified: bool) -> Option<u64> {
    let peak = phase.host_peak_bytes().ok()?;
    if unified {
        let device = phase.device_peak_bytes().ok()?;
        return peak.checked_add(device.unwrap_or(0));
    }
    Some(peak)
}

pub fn axes(policy: &GlmExecutionPolicy) -> Result<BTreeMap<String, String>> {
    let p = serde_json::to_value(policy)?;
    let mut result = BTreeMap::new();
    for name in [
        "resident_static",
        "weights.source",
        "weights.cache_bytes",
        "weights.cache_shards",
        "weights.granularity",
        "expert_cache.maximum_bound_bytes",
        "expert_cache.minimum_bound_bytes",
        "expert_cache.layout",
        "expert_cache.replacement",
        "expert_cache.readmission_interval_tokens",
    ] {
        let value = name.split('.').fold(&p, |v, key| &v[key]);
        result.insert(
            name.into(),
            if name == "weights.granularity" && value.is_null() {
                "shard".into()
            } else {
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            },
        );
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub fn select(
    baseline: &GlmExecutionPolicy,
    breakdown: &GlmAdmissionBreakdown,
    snapshot: &ResourceSnapshot,
    context: &EvidenceContext,
    mode: ResourcePolicyMode,
    explicit_axes: &BTreeSet<String>,
    evidence: Option<&ResourceEvidence>,
) -> Result<GlmSelection> {
    baseline.validate()?;
    context.validate()?;
    let baseline_axes = axes(baseline)?;
    let mut candidates = vec![baseline.clone()];
    if mode == ResourcePolicyMode::Performance {
        let mut resident = baseline.clone();
        resident.resident_static = true;
        candidates.push(resident);
        if let Some(evidence) = evidence {
            evidence.validate()?;
            for row in &evidence.candidates {
                if let EvidencePolicy::Glm(policy) = &row.candidate {
                    candidates.push(policy.clone());
                }
            }
        }
    }
    // Candidates are numbered in the order they were proposed. The number is
    // a label for this record, not an identity: two candidates are the same
    // when their policies are equal, which is what `seen` compares.
    let mut seen: Vec<GlmExecutionPolicy> = Vec::new();
    let mut observations = Vec::new();
    let mut selected = None::<(GlmExecutionPolicy, usize, Option<MeasuredScore>)>;
    for candidate in candidates {
        if seen.contains(&candidate) {
            continue;
        }
        seen.push(candidate.clone());
        let candidate_id = format!("candidate-{}", seen.len() - 1);
        let mut observation = ResourceCandidateObservation {
            candidate_id,
            disposition: CandidateDisposition::NoBenefitEvidence,
            reason: "baseline_no_benefit_evidence".into(),
            shortfall: None,
            expected_cost: None,
            evidence: vec![],
        };
        let candidate_axes = axes(&candidate)?;
        let mut invariant = candidate.clone();
        invariant.weights = baseline.weights.clone();
        invariant.resident_static = baseline.resident_static;
        invariant.expert_cache = baseline.expert_cache.clone();
        let changes_adaptive = candidate.expert_cache.readmission_interval_tokens
            != baseline.expert_cache.readmission_interval_tokens;
        if invariant != *baseline
            || changes_adaptive
            || explicit_axes
                .iter()
                .any(|axis| candidate_axes.get(axis) != baseline_axes.get(axis))
        {
            observation.disposition = CandidateDisposition::OperatorExcluded;
            observation.reason = "candidate changes an explicit or numerical axis".into();
            observations.push(observation);
            continue;
        }
        let cache = match cache_policy(&candidate) {
            Ok(cache) => cache,
            Err(error) => {
                observation.disposition = CandidateDisposition::CapacityRejected;
                observation.reason = format!("candidate cache policy is not executable: {error}");
                observations.push(observation);
                continue;
            }
        };
        let expert_bytes = match usize::try_from(candidate.expert_cache.maximum_bound_bytes) {
            Ok(bytes) => bytes,
            Err(error) => {
                observation.disposition = CandidateDisposition::CapacityRejected;
                observation.reason = format!("expert-cache bound is not a byte count: {error}");
                observations.push(observation);
                continue;
            }
        };
        if let Err(rejection) =
            breakdown.validate_capacity(candidate.resident_static, expert_bytes, cache, snapshot)
        {
            observation.disposition = CandidateDisposition::CapacityRejected;
            observation.reason = rejection.to_string();
            observation.shortfall = rejection.shortfall();
            observations.push(observation);
            continue;
        }
        if candidate.weights != baseline.weights {
            // Unified-memory topologies charge the device peak to the same
            // pool; reserving against the host view alone would overspend it.
            let unified_pool = snapshot.unified_pool_available_bytes();
            let host = match unified_pool {
                Some(pool) => pool,
                None => match crate::glm::admission::host_available(snapshot) {
                    Some(host) => host,
                    None => {
                        observation.disposition = CandidateDisposition::CapacityRejected;
                        observation.reason =
                            "host capacity unavailable for the promotion check".into();
                        observations.push(observation);
                        continue;
                    }
                },
            };
            let reserve =
                crate::glm::admission::GlmAdmissionBreakdown::scaled_promotion_reserve_bytes(
                    snapshot,
                );
            let promotion_phases = match breakdown.phases_with_safety(
                candidate.resident_static,
                expert_bytes,
                cache,
                breakdown.scaled_admission_safety_bytes(snapshot),
            ) {
                Ok(phases) => phases,
                Err(error) => {
                    observation.disposition = CandidateDisposition::CapacityRejected;
                    observation.reason = format!("GLM phase estimate failed: {error}");
                    observations.push(observation);
                    continue;
                }
            };
            let overspent =
                promotion_phases
                    .iter()
                    .find(|p| match promotion_peak(p, unified_pool.is_some()) {
                        Some(peak) => peak.checked_add(reserve).is_none_or(|n| n > host),
                        None => true,
                    });
            if let Some(phase) = overspent {
                observation.disposition = CandidateDisposition::CapacityRejected;
                observation.reason = "host promotion would spend the safety reserve".into();
                observation.shortfall = promotion_peak(phase, unified_pool.is_some())
                    .and_then(|peak| peak.checked_add(reserve))
                    .map(|needed_bytes| CapacityShortfall {
                        predicate: "host_promotion_reserve".into(),
                        needed_bytes,
                        available_bytes: host,
                    });
                observations.push(observation);
                continue;
            }
        }
        let cost = if candidate == *baseline {
            if mode == ResourcePolicyMode::Conservative {
                observation.reason = "operator selected conservative defaults".into();
            }
            None
        } else if let Some(record) = evidence {
            let row = record.candidates.iter().find(
                |row| matches!(&row.candidate, EvidencePolicy::Glm(policy) if *policy == candidate),
            );
            let context_difference = record.context.first_difference(context);
            let baseline_matches = row.is_some_and(
                |row| matches!(&row.baseline_policy, EvidencePolicy::Glm(policy) if policy == baseline),
            );
            if context_difference.is_some() || !baseline_matches {
                observation.disposition = CandidateDisposition::EvidenceMismatch;
                observation.reason = match context_difference {
                    Some(field) => {
                        format!("evidence was measured under a different {field}")
                    }
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
            if candidate.expert_cache != baseline.expert_cache
                && expert_bytes > 0
                && !row.routing_verified
            {
                observation.reason =
                    "expert-cache promotion requires matching F1 trace and replay".into();
                observations.push(observation);
                continue;
            }
            if candidate.expert_cache != baseline.expert_cache && expert_bytes > 0 {
                let expected_expert_bytes = u64::try_from(breakdown.live_expert_bytes)? / 5 * 3;
                if row.routing_profile.as_ref().is_none_or(|profile| {
                    profile.prompt_tokens as usize != breakdown.prompt_tokens
                        || profile.num_hidden_layers != breakdown.num_hidden_layers
                        || profile.num_experts != breakdown.num_experts
                        || profile.experts_per_token != breakdown.experts_per_token
                        || profile.layers != breakdown.sparse_layers
                        || profile
                            .expert_bytes
                            .iter()
                            .any(|bytes| *bytes != expected_expert_bytes)
                }) {
                    observation.disposition = CandidateDisposition::EvidenceMismatch;
                    observation.reason =
                        "routing evidence does not match actual model dimensions or prompt length"
                            .into();
                    observations.push(observation);
                    continue;
                }
            }
            let phases = match breakdown.phases_with_safety(
                candidate.resident_static,
                expert_bytes,
                cache,
                breakdown.scaled_admission_safety_bytes(snapshot),
            ) {
                Ok(phases) => phases,
                Err(error) => {
                    observation.disposition = CandidateDisposition::CapacityRejected;
                    observation.reason = format!("GLM phase estimate failed: {error}");
                    observations.push(observation);
                    continue;
                }
            };
            // The evidence is a process-RSS delta, which counts faulted
            // mmap-backed pages, so this bound adds reclaimable residency back.
            let host_peak = phases
                .iter()
                .map(|p| {
                    p.host_peak_bytes()?
                        .checked_add(p.reclaimable_host_bytes)
                        .context("GLM evidence host bound overflow")
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            let device_peak = phases
                .iter()
                .map(|p| p.device_peak_bytes())
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .max();
            // Under the fold both deltas draw from one pool.
            let combined_peak = if snapshot.unified_pool_available_bytes().is_some() {
                Some(
                    phases
                        .iter()
                        .map(|p| {
                            p.host_peak_bytes()?
                                .checked_add(p.reclaimable_host_bytes)
                                .and_then(|n| {
                                    n.checked_add(p.device_peak_bytes().ok()?.unwrap_or(0))
                                })
                                .context("GLM unified evidence bound overflow")
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .max()
                        .unwrap_or(0),
                )
            } else {
                None
            };
            if row.observed_peak_deltas.is_none_or(|(host, device)| {
                if let Some(bound) = combined_peak {
                    // A missing device observation is not a zero one; the H3
                    // fold rejects it and so does this.
                    let Some(device) = device else {
                        return true;
                    };
                    return host
                        .checked_add(device)
                        .is_none_or(|combined| combined > bound);
                }
                host > host_peak
                    || match (device, device_peak) {
                        (Some(d), Some(bound)) => d > bound,
                        (None, None) => false,
                        _ => true,
                    }
            }) {
                observation.disposition = CandidateDisposition::KnownRegression;
                observation.reason =
                    "validated trial peaks are missing or exceed the candidate model".into();
                observations.push(observation);
                continue;
            }
            if !row.qualifies() {
                observation.disposition = CandidateDisposition::KnownRegression;
                observation.reason =
                    "paired observations do not establish an output-preserving improvement".into();
                observations.push(observation);
                continue;
            }
            observation.reason = format!(
                "qualified paired whole-command median {} us",
                row.median_us()
            );
            let fastest = row.pairs.iter().map(|p| p.candidate_wall_us).min().unwrap();
            let slowest = row.pairs.iter().map(|p| p.candidate_wall_us).max().unwrap();
            Some(MeasuredScore {
                median_us: row.median_us(),
                uncertainty_us: slowest - fastest,
                retained_peak_bytes: u128::from(host_peak) + u128::from(device_peak.unwrap_or(0)),
            })
        } else {
            observation.reason =
                "no applicable measured benefit; capacity alone does not qualify retention".into();
            observations.push(observation);
            continue;
        };
        let choose = selected
            .as_ref()
            .is_none_or(|(_, _, previous)| match (cost, previous) {
                (Some(new), Some(old)) => new.prefers(*old),
                (Some(_), None) => true,
                _ => false,
            });
        if choose {
            if let Some((_, index, _)) = &selected {
                let old: &mut ResourceCandidateObservation = &mut observations[*index];
                old.disposition = CandidateDisposition::Slower;
                old.reason = "another qualified candidate has lower cost or lower retention within measurement uncertainty".into();
            }
            observation.disposition = CandidateDisposition::Selected;
            selected = Some((candidate, observations.len(), cost));
        } else {
            observation.disposition = CandidateDisposition::Slower;
        }
        observations.push(observation);
    }
    let build_provenance = |policy: &GlmExecutionPolicy,
                            axes: Vec<SelectedResourceAxis>,
                            phases: Vec<ResourcePhaseEstimate>,
                            observations: Vec<ResourceCandidateObservation>,
                            refused: bool|
     -> Result<ResourceSelectionProvenance> {
        Ok(ResourceSelectionProvenance {
            schema_version: 1,
            policy: serde_json::to_value(policy)?,
            selector_revision: "resource-policy-measured-v1".into(),
            request: context.request.clone(),
            input: None,
            model: serde_json::to_value(&context.model)?,
            hardware: Some(serde_json::to_value(&context.hardware)?),
            executable: context
                .executable_metadata
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?,
            environment: Some(serde_json::to_value(&context.environment)?),
            mode,
            selection_snapshot: snapshot.clone(),
            final_admission_snapshot: None,
            already_present_at_capture: vec!["device context, tokenizer and header catalog".into()],
            inventory: breakdown.raw_inventory.clone(),
            phases,
            workload: BTreeMap::from([
                ("prompt_tokens".into(), breakdown.prompt_tokens as u64),
                (
                    "context_bound_tokens".into(),
                    policy.dsa_context_bound_tokens as u64,
                ),
                ("refused".into(), u64::from(refused)),
                // The reserve actually applied, after pool scaling — the
                // calibration reader needs the number, not the rule.
                (
                    "admission_safety_bytes_applied".into(),
                    breakdown.scaled_admission_safety_bytes(snapshot),
                ),
            ]),
            axes,
            candidates: observations,
        })
    };
    let Some((policy, _, measured)) = selected else {
        // A refusal must still ship its numbers: per-candidate dispositions
        // and the phase estimates they were judged against. The record names
        // the baseline (that is what was evaluated) and marks refused=1.
        // Same rule as the admitted path; nothing was selected, so no evidence.
        let refusal_axes = baseline_axes
            .iter()
            .map(|(axis, value)| SelectedResourceAxis {
                axis: axis.clone(),
                value: value.clone(),
                origin: if explicit_axes.contains(axis) {
                    SelectionOrigin::OperatorExplicit
                } else {
                    SelectionOrigin::Baseline
                },
            })
            .collect();
        let refusal_phases = breakdown.phases_with_safety(
            baseline.resident_static,
            usize::try_from(baseline.expert_cache.maximum_bound_bytes)?,
            cache_policy(baseline)?,
            breakdown.scaled_admission_safety_bytes(snapshot),
        )?;
        let provenance =
            build_provenance(baseline, refusal_axes, refusal_phases, observations, true)?;
        return Err(super::AdmissionRefused {
            provenance,
            summary: "no admissible GLM resource policy; the baseline and qualified candidates were refused".into(),
        }
        .into());
    };
    let selected_axes = axes(&policy)?
        .into_iter()
        .map(|(axis, value)| {
            let origin = if explicit_axes.contains(&axis) {
                SelectionOrigin::OperatorExplicit
            } else if measured.is_some() && baseline_axes.get(&axis) != Some(&value) {
                SelectionOrigin::MeasuredEvidence
            } else {
                SelectionOrigin::Baseline
            };
            SelectedResourceAxis {
                axis,
                value,
                origin,
            }
        })
        .collect();
    let mut phases = breakdown.phases_with_safety(
        policy.resident_static,
        usize::try_from(policy.expert_cache.maximum_bound_bytes)?,
        cache_policy(&policy)?,
        breakdown.scaled_admission_safety_bytes(snapshot),
    )?;
    if policy.weights != baseline.weights {
        for phase in &mut phases {
            phase.host_promotion_reserve_bytes =
                crate::glm::admission::GlmAdmissionBreakdown::scaled_promotion_reserve_bytes(
                    snapshot,
                );
        }
    }
    let provenance = build_provenance(&policy, selected_axes, phases, observations, false)?;
    provenance.validate()?;
    provenance.validate_policy_binding(&serde_json::to_value(&policy)?)?;
    Ok(GlmSelection { policy, provenance })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        glm::{ExpertCacheLayout, ExpertCacheReplacementPolicy},
        resource_policy::evidence::{EvidenceArtifact, MeasuredCandidate, PairedObservation},
        runtime::{
            probe::{CgroupMemoryLimit, ResourceMeasurementScopes},
            weights::{
                WeightSource,
                accounting::{CacheInventory, CacheShardInventory},
            },
        },
    };
    use candle_core::Device;

    fn baseline() -> GlmExecutionPolicy {
        GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            false,
            32,
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheReplacementPolicy::Lru,
            0,
            0,
            false,
        )
        .unwrap()
    }
    fn context() -> EvidenceContext {
        crate::resource_policy::evidence::tests::sample_context()
    }
    fn snapshot() -> ResourceSnapshot {
        ResourceSnapshot {
            schema_version: 1,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(8 << 30),
            cgroup_v2_memory_available_bytes: Some(8 << 30),
            cgroup_v2_memory_limit: Some(CgroupMemoryLimit::Bytes(8 << 30)),
            cgroup_v2_memory_current_bytes: Some(0),
            device_free_memory_bytes: None,
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
    fn breakdown() -> GlmAdmissionBreakdown {
        GlmAdmissionBreakdown {
            scope: crate::glm::admission::GlmLayerScope {
                start: 0,
                end: 2,
                total_layers: 2,
            },
            cpu_fp8_dequantization: false,
            pinned_transfer_bytes: 0,
            pinned_fill_ahead_bytes: 0,
            concurrent_loads: 0,
            static_load_device_bytes: 0,
            expert_load_device_bytes: 0,
            compute_on_host: true,
            prompt_tokens: 2,
            num_hidden_layers: 2,
            num_experts: 4,
            experts_per_token: 2,
            sparse_layers: vec![1],
            static_bytes: 500,
            lm_head_bytes: 10,
            largest_streamed_group_bytes: 20,
            kda_state_bytes: 4,
            dsa_cache_bytes_per_token: 4,
            maximum_dsa_cache_bytes: 128,
            maximum_dsa_layer_cache_bytes: 64,
            live_expert_bytes: 24,
            prefill_workspace_bytes: 128,
            decode_workspace_bytes: 64,
            host_route_workspace_bytes: 32,
            host_sampling_workspace_bytes: 0,
            prefill_host_mask_bytes: 0,
            static_load_host_bytes: 8,
            streamed_load_host_bytes: 8,
            expert_load_host_bytes: 4,
            raw_inventory: CacheInventory {
                shards: vec![CacheShardInventory {
                    name: "a.safetensors".into(),
                    file_bytes: 108,
                    header_bytes: 8,
                    selected_tensor_bytes: 100,
                    selected_tensor_count: 1,
                    largest_tensor_bytes: 100,
                }],
            },
        }
    }
    fn evidence() -> ResourceEvidence {
        let base = baseline();
        let mut policy = base.clone();
        policy.resident_static = true;
        ResourceEvidence {
            schema_version: 1,
            context: context(),
            candidates: vec![MeasuredCandidate {
                baseline_policy: EvidencePolicy::Glm(base.clone()),
                candidate: EvidencePolicy::Glm(policy),
                minimum_improvement_basis_points: 200,
                routing_trace: None,
                routing_replay: None,
                observed_peak_deltas: Some((100, None)),
                routing_verified: false,
                routing_profile: None,
                pairs: (0..3)
                    .map(|n| PairedObservation {
                        baseline_wall_us: 1000,
                        candidate_wall_us: 500,
                        baseline_record: EvidenceArtifact {
                            file: format!("base{n}.json"),
                            bytes: 20 + n,
                        },
                        candidate_record: EvidenceArtifact {
                            file: format!("candidate{n}.json"),
                            bytes: 30 + n,
                        },
                        // `load` is what compares the retained outputs; this
                        // fixture states the outcome it is standing in for.
                        outputs_match: true,
                    })
                    .collect(),
            }],
        }
    }
    #[test]
    fn uncertain_timings_prefer_lower_retention_but_clear_speed_wins() {
        let current = MeasuredScore {
            median_us: 500,
            uncertainty_us: 20,
            retained_peak_bytes: 1000,
        };
        assert!(
            MeasuredScore {
                median_us: 510,
                uncertainty_us: 20,
                retained_peak_bytes: 500
            }
            .prefers(current)
        );
        assert!(
            MeasuredScore {
                median_us: 100,
                uncertainty_us: 5,
                retained_peak_bytes: 2000
            }
            .prefers(current)
        );
        assert!(
            !MeasuredScore {
                median_us: 900,
                uncertainty_us: 5,
                retained_peak_bytes: 1
            }
            .prefers(current)
        );
    }

    #[test]
    fn capacity_alone_keeps_baseline_but_matching_pairs_can_promote() {
        let base = baseline();
        let selected = select(
            &base,
            &breakdown(),
            &snapshot(),
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            None,
        )
        .unwrap();
        assert_eq!(selected.policy, base);
        assert_eq!(
            selected.provenance.policy,
            serde_json::to_value(&base).unwrap()
        );
        let evidence = evidence();
        let selected = select(
            &base,
            &breakdown(),
            &snapshot(),
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            Some(&evidence),
        )
        .unwrap();
        assert!(selected.policy.resident_static);
        assert!(
            selected
                .provenance
                .axes
                .iter()
                .any(|a| a.axis == "resident_static"
                    && a.origin == SelectionOrigin::MeasuredEvidence)
        );
    }
    #[test]
    fn unified_pool_blocks_weight_promotion_that_split_axes_admit() {
        let base = baseline();
        // Candidate differs only in the weights axis, so it reaches the
        // host-promotion reserve check.
        let mut promoted = base.clone();
        promoted.weights.cache_bytes = Some(64 << 30);
        let evidence = ResourceEvidence {
            candidates: vec![MeasuredCandidate {
                baseline_policy: EvidencePolicy::Glm(base.clone()),
                candidate: EvidencePolicy::Glm(promoted.clone()),
                minimum_improvement_basis_points: 200,
                routing_trace: None,
                routing_replay: None,
                observed_peak_deltas: Some((1, None)),
                routing_verified: false,
                routing_profile: None,
                pairs: (0..3)
                    .map(|n| PairedObservation {
                        baseline_wall_us: 1000,
                        candidate_wall_us: 500,
                        baseline_record: EvidenceArtifact {
                            file: format!("wbase{n}.json"),
                            bytes: 20 + n,
                        },
                        candidate_record: EvidenceArtifact {
                            file: format!("wcandidate{n}.json"),
                            bytes: 30 + n,
                        },
                        outputs_match: true,
                    })
                    .collect(),
            }],
            ..evidence()
        };
        // Pool (the device view is smallest) passes plain capacity but not the
        // promotion reserve; the discrete host view admits both. Totals are
        // set explicitly so the total-scaled reserves are fixed (host_total
        // 8 GiB → both reserves at the 512 MiB floor) and the discriminating
        // margin is not pool-dependent: promoted peak 0.404 GiB + 0.512 GiB
        // > 0.9 GiB pool, while the baseline peak + the same reserve fits.
        // The discrete device view is widened further so its own safety
        // reserve fits with room to spare.
        let mut unified_snapshot = snapshot();
        unified_snapshot.host_memory_total_bytes = Some(8 << 30);
        unified_snapshot.device_free_memory_bytes = Some(966_367_640); // 0.9 GiB
        unified_snapshot.host_device_memory_is_unified = Some(true);
        let mut discrete_snapshot = unified_snapshot.clone();
        discrete_snapshot.host_device_memory_is_unified = None;
        discrete_snapshot.device_free_memory_bytes = Some(2 << 30);

        let selected = select(
            &base,
            &breakdown(),
            &discrete_snapshot,
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            Some(&evidence),
        )
        .unwrap();
        assert_eq!(
            selected.policy.weights.cache_bytes,
            Some(64 << 30),
            "discrete axes admit the promotion reserve"
        );

        let selected = select(
            &base,
            &breakdown(),
            &unified_snapshot,
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            Some(&evidence),
        )
        .unwrap();
        assert_eq!(
            selected.policy, base,
            "unified pool must charge device bytes to the promotion reserve"
        );
    }

    #[test]
    fn refusal_carries_the_full_candidate_record() {
        let base = baseline();
        let mut tight = snapshot();
        // Nothing fits: 1 byte of host capacity.
        tight.host_memory_available_bytes = Some(1);
        tight.cgroup_v2_memory_available_bytes = Some(1);
        tight.cgroup_v2_memory_limit = Some(CgroupMemoryLimit::Bytes(1));
        let error = match select(
            &base,
            &breakdown(),
            &tight,
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            None,
        ) {
            Ok(_) => panic!("a 1-byte capacity must refuse"),
            Err(error) => error,
        };
        let refusal = error
            .downcast_ref::<crate::resource_policy::AdmissionRefused>()
            .expect("refusals must carry the candidate record");
        assert_eq!(refusal.provenance.workload.get("refused"), Some(&1));
        assert!(!refusal.provenance.candidates.is_empty());
        assert!(
            refusal
                .provenance
                .candidates
                .iter()
                .all(|c| c.disposition != CandidateDisposition::Selected)
        );
        assert!(
            refusal
                .provenance
                .candidates
                .iter()
                .any(|c| !c.reason.is_empty())
        );
        // The record must carry calibratable numbers, not just the refusal
        // marker: at least one phase with a non-zero host peak. (This fixture
        // is host-only, so no device peaks exist here; CUDA phases are covered
        // by the unified-pool tests above.)
        assert!(
            refusal
                .provenance
                .phases
                .iter()
                .any(|p| p.host_peak_bytes().is_ok_and(|peak| peak > 0)),
            "refusal record must carry non-zero phase estimates"
        );
    }

    #[test]
    fn explicit_false_and_conservative_mode_own_the_choice() {
        for (mode, explicit) in [
            (ResourcePolicyMode::Conservative, BTreeSet::new()),
            (
                ResourcePolicyMode::Performance,
                BTreeSet::from(["resident_static".into()]),
            ),
        ] {
            let selected = select(
                &baseline(),
                &breakdown(),
                &snapshot(),
                &context(),
                mode,
                &explicit,
                Some(&evidence()),
            )
            .unwrap();
            assert!(!selected.policy.resident_static);
        }
    }
    #[test]
    fn mismatch_regression_and_missing_f1_cannot_promote() {
        for case in 0..4 {
            let mut evidence = evidence();
            match case {
                0 => evidence.context.request = serde_json::json!({"prompt": "another"}),
                1 => evidence.candidates[0].pairs[1].candidate_wall_us = 2000,
                2 => evidence.candidates[0].pairs[1].outputs_match = false,
                _ => {
                    if let EvidencePolicy::Glm(policy) = &mut evidence.candidates[0].candidate {
                        policy.expert_cache.maximum_bound_bytes = 100;
                        policy.expert_cache.minimum_bound_bytes = 100;
                    }
                }
            }
            let selected = select(
                &baseline(),
                &breakdown(),
                &snapshot(),
                &context(),
                ResourcePolicyMode::Performance,
                &BTreeSet::new(),
                Some(&evidence),
            )
            .unwrap();
            assert!(!selected.policy.resident_static, "case {case}");
            assert_eq!(selected.policy.expert_cache.maximum_bound_bytes, 0);
        }
    }
    #[test]
    fn verified_routing_from_different_model_shape_cannot_promote() {
        let mut evidence = evidence();
        if let EvidencePolicy::Glm(policy) = &mut evidence.candidates[0].candidate {
            policy.expert_cache.maximum_bound_bytes = 100;
            policy.expert_cache.minimum_bound_bytes = 100;
        }
        evidence.candidates[0].routing_verified = true;
        evidence.candidates[0].routing_profile =
            Some(super::super::evidence::VerifiedRoutingProfile {
                prompt_tokens: 999,
                num_hidden_layers: 2,
                num_experts: 4,
                experts_per_token: 2,
                layers: vec![1],
                expert_bytes: vec![12; 4],
            });
        let selected = select(
            &baseline(),
            &breakdown(),
            &snapshot(),
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            Some(&evidence),
        )
        .unwrap();
        assert_eq!(selected.policy.expert_cache.maximum_bound_bytes, 0);
        assert!(
            selected
                .provenance
                .candidates
                .iter()
                .any(|c| c.reason.contains("actual model dimensions"))
        );
    }

    #[test]
    fn tighter_capacity_refuses_unchanged_policy_without_reshaping_it() {
        let selected = select(
            &baseline(),
            &breakdown(),
            &snapshot(),
            &context(),
            ResourcePolicyMode::Performance,
            &BTreeSet::new(),
            Some(&evidence()),
        )
        .unwrap();
        let mut current = snapshot();
        current.cgroup_v2_memory_available_bytes = Some(1);
        assert!(
            breakdown()
                .validate_capacity(
                    selected.policy.resident_static,
                    0,
                    selected.policy.cache_policy().unwrap(),
                    &current
                )
                .is_err()
        );
        selected
            .provenance
            .validate_policy_binding(&serde_json::to_value(&selected.policy).unwrap())
            .unwrap();
    }

    #[test]
    fn capacity_rejections_carry_a_structured_shortfall() {
        let mut current = snapshot();
        current.host_memory_available_bytes = Some(1);
        current.cgroup_v2_memory_available_bytes = Some(1);
        let error = match select(
            &baseline(),
            &breakdown(),
            &current,
            &context(),
            ResourcePolicyMode::Conservative,
            &BTreeSet::new(),
            None,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a snapshot below the peak must refuse the baseline"),
        };
        let refusal = error
            .downcast_ref::<super::super::AdmissionRefused>()
            .expect("the refusal carries its record");
        let rejection = refusal
            .provenance
            .candidates
            .iter()
            .find(|c| c.disposition == CandidateDisposition::CapacityRejected)
            .expect("the capacity rejection is recorded");
        let shortfall = rejection
            .shortfall
            .as_ref()
            .expect("the rejection binds by bytes");
        assert_eq!(shortfall.predicate, "host_available");
        assert_eq!(shortfall.available_bytes, 1);
        assert!(shortfall.needed_bytes > 1);
        refusal.provenance.validate().unwrap();
    }

    #[test]
    fn unmeasured_capacity_is_a_recorded_rejection_not_an_abort() {
        let mut current = snapshot();
        current.host_memory_available_bytes = None;
        current.cgroup_v2_memory_available_bytes = None;
        let error = match select(
            &baseline(),
            &breakdown(),
            &current,
            &context(),
            ResourcePolicyMode::Conservative,
            &BTreeSet::new(),
            None,
        ) {
            Err(error) => error,
            Ok(_) => panic!("an unmeasured snapshot must refuse the baseline"),
        };
        let refusal = error
            .downcast_ref::<super::super::AdmissionRefused>()
            .expect("the refusal carries its record");
        let observation = &refusal.provenance.candidates[0];
        assert_eq!(
            observation.disposition,
            CandidateDisposition::CapacityRejected
        );
        assert!(
            observation
                .reason
                .contains("cannot measure free host memory")
        );
        assert!(observation.shortfall.is_none());
        refusal.provenance.validate().unwrap();
    }
}
