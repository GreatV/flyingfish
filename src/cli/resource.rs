use super::output_hygiene::{ensure_new_output, mib_to_bytes};
use super::{
    DEFAULT_FLASH_BACKEND_WORKSPACE_MIB, DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB, DeviceCacheArgs,
    H3AdmissionArgs, H3ResourceArgs, OptionalWeightCacheArgs,
};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use flyingfish::h3::config::TransformerConfig;
use flyingfish::h3::execution::H3ExecutionPlan;
use flyingfish::h3::model::validate_h3_numerical_backend;
use flyingfish::h3::policy::ExecutionBackendPolicy;
use flyingfish::h3::policy::ExecutionPolicy;
use flyingfish::h3::resources::{ResourceAssumptions, SequenceRows, T2vaGeometry};
use flyingfish::resource_policy::evidence::{EvidenceContext, ResourceEvidence};
use flyingfish::resource_policy::h3::{H3Selection, H3SelectionRequest};
use flyingfish::runtime::probe::ResourceSnapshot;
use flyingfish::runtime::resource_selection::SelectionOrigin;
use flyingfish::runtime::weights::{
    CachePolicy, DeviceCache, DeviceCachePolicy, ModelWeights, WeightSource,
};
use std::collections::BTreeSet;
use std::path::Path;

/// Headroom the residency ladder leaves free on the device for activations,
/// workspaces and allocator slack: the shared admission-reserve shape (5% of
/// the pool total, capped at 1 GiB), so small and unified pools are not
/// over-reserved — a 24 GiB discrete card keeps the historical flat 1 GiB
/// exactly.
///
/// A declared margin, not a measured requirement. Adapters add known request
/// allocations separately. Runtime cache demotion can recover weight-load
/// allocation failures; this margin does not guarantee that every activation
/// allocation will fit.
pub(crate) fn device_residency_reserve_bytes(
    snapshot: &flyingfish::runtime::probe::ResourceSnapshot,
) -> u64 {
    let total = if snapshot.host_device_memory_is_unified == Some(true) {
        [
            snapshot.host_pool_total_bytes(),
            snapshot.device_total_memory_bytes,
        ]
        .into_iter()
        .flatten()
        .min()
    } else {
        snapshot.device_total_memory_bytes
    };
    // The free-view fallback below is for legacy recorded snapshots only: a
    // live capture produces the device total and free together (both come
    // from the same CUDA device handle).
    flyingfish::runtime::probe::admission_reserve_bytes(total.or(snapshot.device_free_memory_bytes))
}

/// Default CUDA placement for adapters with request tensor estimates. Explicit
/// ceilings (including zero) and non-CUDA devices keep their existing behavior.
/// `required_device_bytes` is one caller's device charge and is multiplied by
/// `unified_share`, the number of callers drawing from this pool.
/// `unified_host_bytes` is already the total across every concurrent host
/// allocation, because those exist per caller whether or not the caller shares
/// this pool — the two counts differ and only the caller knows the second. All
/// of it applies under the fold alone.
pub(crate) fn decide_auto_residency_with_required_memory(
    demands: &[flyingfish::runtime::residency::PhaseResidencyDemand],
    device: &candle_core::Device,
    args: DeviceCacheArgs,
    required_device_bytes: u64,
    unified_host_bytes: u64,
    unified_share: u64,
) -> Result<DeviceCache> {
    use flyingfish::runtime::{probe::ResourceSnapshot, residency::plan_device_residency};
    if !device.is_cuda() || args.device_cache_mib.is_some() {
        return decide_residency_with_required_memory(
            demands,
            device,
            args,
            required_device_bytes,
            unified_host_bytes,
            unified_share,
        );
    }
    let snapshot = ResourceSnapshot::capture(Some(device));
    // Each concurrent caller needs its own fixed margin, but only where the
    // pool is actually shared; a discrete device owes nothing to the others.
    let shared = snapshot.unified_pool_available_bytes().is_some();
    let share = if shared { unified_share.max(1) } else { 1 };
    let reserve = required_device_bytes
        .saturating_mul(share)
        .checked_add(device_residency_reserve_bytes(&snapshot).saturating_mul(share))
        .context("device residency reserve overflow")?;
    let capacity = if snapshot.unified_accounting_is_undecidable() {
        0
    } else {
        snapshot
            .unified_pool_available_bytes()
            .or(snapshot.device_free_memory_bytes)
            .unwrap_or(0)
    };
    let reserve = if shared {
        reserve
            .checked_add(unified_host_bytes)
            .context("unified host reserve overflow")?
    } else {
        reserve
    };
    // Reserving each concurrent caller's charges is not enough: the remainder
    // is also shared, so one caller may retain only its portion of it.
    let available = capacity.saturating_sub(reserve) / share;
    let plan = plan_device_residency(demands, available, 0)?;
    eprintln!(
        "device residency: auto retains {} MiB in {}/{} weight groups after reserving {} MiB for request tensors and workspace",
        plan.resident_bytes / (1 << 20),
        plan.placed.len(),
        demands.len(),
        reserve / (1 << 20),
    );
    Ok(DeviceCache::with_selected_phases(
        DeviceCachePolicy::with_max_bytes(plan.resident_bytes)
            .with_cuda_allocator(args.device_cache_allocator),
        plan.placed,
    ))
}

pub(crate) fn decide_residency_with_required_memory(
    demands: &[flyingfish::runtime::residency::PhaseResidencyDemand],
    device: &candle_core::Device,
    args: DeviceCacheArgs,
    required_device_bytes: u64,
    unified_host_bytes: u64,
    unified_share: u64,
) -> Result<DeviceCache> {
    use flyingfish::runtime::{probe::ResourceSnapshot, residency::decide_device_residency};
    let snapshot = ResourceSnapshot::capture(Some(device));
    let shared = snapshot.unified_pool_available_bytes().is_some();
    let share = if shared { unified_share.max(1) } else { 1 };
    let reserve = device_residency_reserve_bytes(&snapshot)
        .saturating_mul(share)
        .checked_add(required_device_bytes.saturating_mul(share))
        .context("device residency reserve overflow")?;
    // An explicit ceiling is still clamped against the shared pool, so the
    // caller's separately modelled host allocation is reserved here too.
    let reserve = if shared {
        reserve
            .checked_add(unified_host_bytes)
            .context("unified host reserve overflow")?
    } else {
        reserve
    };
    // The share is taken before planning, by withholding the other callers'
    // portions as reserve: dividing the decision afterwards would leave a plan
    // built against the whole remainder, selecting phases that can never be
    // retained. The operator's own ceiling still clamps on top, so a ceiling
    // that already fits the share is not reduced.
    let reserve = match snapshot.unified_pool_available_bytes() {
        Some(pool) if share > 1 => pool.saturating_sub(pool.saturating_sub(reserve) / share),
        _ => reserve,
    };
    let decision = decide_device_residency(&snapshot, demands, reserve, args.authorization()?)?;
    if decision.authorized_budget_bytes > 0 {
        eprintln!(
            "device residency: {} MiB authorized, {}/{} phase working sets fit, {} MiB planned weight peak",
            decision.authorized_budget_bytes / (1 << 20),
            decision.plan.placed.len(),
            demands.len(),
            decision.plan.resident_bytes / (1 << 20),
        );
        for phase in &decision.plan.spilled {
            eprintln!("device residency: {phase} cannot be retained in full within this budget");
        }
    }
    Ok(DeviceCache::with_selected_phases(
        DeviceCachePolicy::with_max_bytes(decision.authorized_budget_bytes)
            .with_cuda_allocator(args.device_cache_allocator),
        decision.plan.placed,
    ))
}

pub(super) struct H3ResourceRequest<'a> {
    pub component: &'a Path,
    pub device: &'a Device,
    pub baseline: &'a ExecutionPolicy,
    pub geometry: T2vaGeometry,
    pub rows: Option<SequenceRows>,
    pub timestep_rows: u64,
    pub evaluations: usize,
    pub limits: H3AdmissionArgs,
    pub resources: &'a H3ResourceArgs,
    pub weights: OptionalWeightCacheArgs,
    pub locked_origin: Option<SelectionOrigin>,
    pub resident_input_bytes: u64,
    pub additional_host_allowance_bytes: u64,
    pub request: serde_json::Value,
}

pub(super) fn select_h3(request: H3ResourceRequest<'_>) -> Result<H3Selection> {
    if request.limits.max_host_mib == Some(0) {
        bail!("H3 preflight refused: binding budget: host (zero byte limit)");
    }
    if request.limits.max_device_mib == Some(0) {
        bail!("H3 preflight refused: binding budget: device (zero byte limit)");
    }
    let config = TransformerConfig::from_file(request.component.join("config.json"))?;
    let metadata = ModelWeights::open(request.component, WeightSource::Mmap, CachePolicy::new(1))?;
    let inventory = metadata.cache_inventory()?;
    let evidence = if request.locked_origin.is_none() {
        request
            .resources
            .resource_evidence
            .as_deref()
            .map(ResourceEvidence::load)
            .transpose()?
    } else {
        None
    };
    let mut context =
        EvidenceContext::collect(request.component, request.device, request.request.clone())?;
    let cold_components = vec![request.component.to_owned()];
    let cache_preparation = if request.resources.resource_cold_cache {
        let observed =
            flyingfish::resource_policy::prepare_cold_cache_components(&cold_components)?;
        context.cache_state = flyingfish::resource_policy::evidence::CacheState::Cold;
        Some(observed)
    } else {
        None
    };
    let snapshot = ResourceSnapshot::capture(Some(request.device));
    let budget = super::generate::probed_budget(
        &snapshot,
        request.baseline.execution_backend,
        request.limits.max_host_mib,
        request.limits.max_device_mib,
    )?;
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    if request.device.is_cpu() {
        assumptions.weight_element_bytes = 4;
        assumptions.activation_element_bytes = 4;
        assumptions.device_memory_is_host = true;
    }
    // Unified-memory devices (e.g. Jetson) share one pool across both axes.
    let unified = snapshot.host_device_memory_is_unified == Some(true);
    if unified {
        assumptions.device_memory_is_host = true;
    }
    assumptions.evaluation_count = u64::try_from(request.evaluations)?;
    assumptions.precompute_adaln_steps = if request.baseline.precompute_adaln {
        assumptions.evaluation_count
    } else {
        0
    };
    assumptions.timestep_rows = request.timestep_rows;
    assumptions.use_flash_attention = request.baseline.flash_attention();
    let minimum = if request.device.is_cpu() {
        0
    } else if request.baseline.flash_attention() {
        DEFAULT_FLASH_BACKEND_WORKSPACE_MIB
    } else {
        DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB
    };
    let workspace = request.limits.backend_workspace_mib.unwrap_or(minimum);
    anyhow::ensure!(
        workspace >= minimum,
        "backend workspace is below the {minimum} MiB minimum"
    );
    assumptions.backend_workspace_bytes = mib_to_bytes(workspace)?;
    assumptions.peak_materialized_weight_bytes_override = Some(
        H3ExecutionPlan::from_config(&metadata, &config)?
            .peak_stage_weight_bytes()
            .checked_mul(if request.device.is_cpu() { 2 } else { 1 })
            .context("materialized H3 stage size overflow")?,
    );
    let rows = request
        .rows
        .unwrap_or(request.geometry.sequence_rows(config.patch_size)?);
    validate_h3_numerical_backend(
        request.device,
        request.baseline.flash_attention(),
        request.geometry.attention_key_chunk_policy,
        usize::try_from(rows.total)?,
        request.geometry.text_rows,
        usize::try_from(request.timestep_rows)?,
    )?;
    let mut explicit = BTreeSet::new();
    for (present, axis) in [
        (request.weights.weight_source.is_some(), "weights.source"),
        (
            request.weights.host_cache_mib.is_some(),
            "weights.cache_bytes",
        ),
        (
            request.weights.host_cache_mib.is_some(),
            "weights.cache_shards",
        ),
        (
            request.weights.host_cache_granularity.is_some(),
            "weights.granularity",
        ),
        (
            request.baseline.weights.device_cache.is_enabled(),
            "weights.device_cache",
        ),
        (
            request.baseline.weights.host_phase_priority,
            "weights.host_phase_priority",
        ),
    ] {
        if present {
            explicit.insert(axis.to_owned());
        }
    }
    let mut selection_request = H3SelectionRequest {
        baseline: request.baseline,
        config: &config,
        geometry: request.geometry,
        rows,
        assumptions,
        inventory: &inventory,
        budget,
        snapshot: &snapshot,
        context: &context,
        mode: request.resources.resource_policy,
        explicit_axes: &explicit,
        evidence: evidence.as_ref().map(|(record, _)| record),
        locked_origin: request.locked_origin,
        resident_input_bytes: request.resident_input_bytes,
        // Under the unified fold the once-only device reserve is charged to
        // the host axis instead of stamped on phases (host-only phases carry
        // no device reserve). This keeps "modelled peaks plus the shared
        // pool-scaled reserve of slack" charged exactly once, including the
        // paths where automatic_device_cache early-returns.
        additional_host_allowance_bytes: request
            .additional_host_allowance_bytes
            .checked_add(if unified {
                device_residency_reserve_bytes(&snapshot)
            } else {
                0
            })
            .context("H3 host allowance overflow")?,
    };
    // Under the unified fold the reserve already sits in the host peak via
    // additional_host_allowance_bytes; passing it again as sizing headroom
    // would charge it twice. Discrete keeps the headroom-sized path.
    let headroom_bytes = if unified {
        0
    } else {
        device_residency_reserve_bytes(&snapshot)
    };
    let automatic_cache = flyingfish::resource_policy::h3::automatic_device_cache(
        &selection_request,
        headroom_bytes,
    )?;
    let mut baseline = request.baseline.clone();
    if let Some(cache) = automatic_cache {
        baseline.weights.device_cache = cache;
        selection_request.baseline = &baseline;
        eprintln!(
            "H3 device residency: auto retains up to {} MiB after compute/workspace estimates and {} MiB additional headroom",
            cache.max_bytes / (1 << 20),
            headroom_bytes / (1 << 20),
        );
    }
    let mut selection = flyingfish::resource_policy::h3::select(selection_request)?;
    if automatic_cache.is_some() {
        selection.provenance.selector_revision = "h3-resource-capacity-v2".into();
        for axis in &mut selection.provenance.axes {
            if axis.axis == "weights.device_cache" {
                axis.origin = SelectionOrigin::CostModel;
            }
        }
        for candidate in &mut selection.provenance.candidates {
            if candidate.disposition
                == flyingfish::runtime::resource_selection::CandidateDisposition::Selected
            {
                candidate.reason = "automatic CUDA retention within remaining capacity".into();
            }
        }
        for phase in &mut selection.provenance.phases {
            // Host-only phases (CPU, or the unified-memory fold that charges
            // device bytes into the host axis) must keep a zero device
            // reserve: the phase type invariant rejects the combination, and
            // the sizing already accounted the headroom against the pool.
            if phase.required_device_bytes.is_some() {
                phase.device_reserve_bytes = device_residency_reserve_bytes(&snapshot);
            }
        }
    }
    selection.provenance.workload.insert(
        "identity_component_count".into(),
        cold_components.len() as u64,
    );
    if let Some(cache) = cache_preparation {
        selection
            .provenance
            .workload
            .insert("cold_cache_advised_bytes".into(), cache.advised_bytes);
        selection
            .provenance
            .workload
            .insert("cold_cache_checked_pages".into(), cache.checked_pages);
        selection
            .provenance
            .workload
            .insert("cold_cache_resident_pages".into(), cache.resident_pages);
    }
    let final_observation = ResourceSnapshot::capture(Some(request.device));
    let final_budget = super::generate::probed_budget(
        &final_observation,
        selection.policy.execution_backend,
        request.limits.max_host_mib,
        request.limits.max_device_mib,
    )?;
    let phase = &selection.provenance.phases[0];
    // The fold emits a host-only phase, so reading the device peak from it
    // records zero and contradicts the selection ledger. Fit is unaffected.
    let device_peak = if selection.policy.execution_backend == ExecutionBackendPolicy::Cpu
        || final_observation.host_device_memory_is_unified == Some(true)
    {
        selection
            .estimate
            .peak_device_bytes
            .checked_sub(request.resident_input_bytes)
            .context("input credit exceeds compute peak")?
    } else {
        phase.device_peak_bytes()?.unwrap_or(0)
    };
    // A promotion was selected only if its peak plus this reserve fitted, so
    // the same requirement is carried into the final check. Under the fold the
    // device peak is already inside the host ledger the reserve sits beside.
    let host_peak = phase.host_peak_bytes()?;
    let checked_host = host_peak
        .checked_add(phase.host_promotion_reserve_bytes)
        .context("H3 final promotion headroom overflow")?;
    let final_report = final_budget.check_peaks(checked_host, device_peak);
    if !final_report.within_budget {
        // A bare error would leave the caller nothing to downcast.
        let mut provenance = selection.provenance.clone();
        flyingfish::resource_policy::h3::record_refused_final_admission(
            &mut provenance,
            final_observation,
            final_budget,
            checked_host,
            device_peak,
        )?;
        provenance.workload.insert("refused".into(), 1);
        for candidate in &mut provenance.candidates {
            if candidate.disposition
                == flyingfish::runtime::resource_selection::CandidateDisposition::Selected
            {
                candidate.disposition =
                    flyingfish::runtime::resource_selection::CandidateDisposition::CapacityRejected;
                candidate.reason = "refused by final H3 admission".into();
                candidate.shortfall = final_report
                    .violations
                    .first()
                    .map(|violation| violation.shortfall());
            }
        }
        return Err(flyingfish::resource_policy::AdmissionRefused {
            provenance,
            summary: "live H3 admission refused the unchanged selected policy".into(),
        }
        .into());
    }
    flyingfish::resource_policy::h3::record_final_admission(
        &mut selection.provenance,
        final_observation,
        final_budget,
        host_peak,
        device_peak,
    )?;
    selection.provenance.workload.insert(
        "host_budget_operator_override".into(),
        u64::from(request.limits.max_host_mib.is_some()),
    );
    selection.provenance.workload.insert(
        "compute_budget_operator_cap".into(),
        u64::from(request.limits.max_device_mib.is_some()),
    );
    eprintln!(
        "H3 resource policy {:?}: {} candidates",
        request.resources.resource_policy,
        selection.provenance.candidates.len()
    );
    Ok(selection)
}

pub(super) fn geometry_from_tensors(
    policy: &ExecutionPolicy,
    prompt: &Tensor,
    video: &Tensor,
    audio: &Tensor,
) -> Result<T2vaGeometry> {
    let (batch, text_rows, _) = prompt.dims3()?;
    let (video_batch, _, latent_frames, latent_height, latent_width) = video.dims5()?;
    let (audio_channels, _, audio_frames) = audio.dims3()?;
    anyhow::ensure!(
        batch == 1 && video_batch == 1,
        "H3 resource admission requires single-batch inputs"
    );
    let chunks = policy.transformer_chunking()?;
    Ok(T2vaGeometry {
        text_rows,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        attention_projection_chunk_size: chunks.attention.projection_chunk_size.get(),
        attention_query_chunk_size: chunks.attention.query_chunk_size.get(),
        attention_key_chunk_policy: chunks.attention.key,
        ffn_token_chunk_size: chunks.feed_forward_chunk_size.get(),
        output_token_chunk_size: chunks.output_chunk_size.get(),
    })
}

pub(super) fn represented_input_bytes(
    device: &Device,
    prompt: &Tensor,
    video: &Tensor,
    audio: &Tensor,
) -> Result<u64> {
    let activation = if device.is_cpu() {
        candle_core::DType::F32
    } else {
        candle_core::DType::BF16
    };
    [
        (prompt, activation),
        (video, candle_core::DType::F32),
        (audio, candle_core::DType::F32),
    ]
    .into_iter()
    .filter(|(tensor, dtype)| tensor.dtype() == *dtype && tensor.device().same_device(device))
    .try_fold(0u64, |sum, (tensor, _)| {
        sum.checked_add(
            u64::try_from(tensor.elem_count())?
                .checked_mul(tensor.dtype().size_in_bytes() as u64)
                .context("input size overflow")?,
        )
        .context("input residency sum overflow")
    })
}

pub(super) fn validate_sidecar_output(
    path: &Path,
    output: &Path,
    others: &[Option<&Path>],
) -> Result<()> {
    ensure_new_output(path, "resource selection")?;
    for other in std::iter::once(output).chain(others.iter().filter_map(|path| *path)) {
        anyhow::ensure!(
            path != other && !path.starts_with(other) && !other.starts_with(path),
            "resource selection conflicts with another output: {}",
            other.display()
        );
    }
    Ok(())
}

pub(super) fn validate_recorded_selection(path: &Path, policy: &ExecutionPolicy) -> Result<()> {
    if !path.try_exists()? {
        return Ok(());
    }
    let bytes = flyingfish::runtime::artifact::read_artifact_snapshot(
        path,
        flyingfish::runtime::resource_selection::MAX_RESOURCE_SELECTION_BYTES as u64,
    )?;
    let record = flyingfish::runtime::resource_selection::ResourceSelectionProvenance::from_json(
        &bytes.bytes,
    )?;
    anyhow::ensure!(
        !record.is_refusal(),
        "recorded resource selection at {} is a refusal record, not an admitted selection",
        path.display()
    );
    record.validate_policy_binding(&serde_json::to_value(policy)?)
}

pub(super) fn verify_evidence(path: &Path) -> Result<()> {
    let (evidence, evidence_bytes) = ResourceEvidence::load(path)?;
    let candidates = evidence
        .candidates
        .iter()
        .map(|row| {
            Ok::<_, anyhow::Error>(serde_json::json!({
                "candidate":row.candidate,"pairs":row.pairs.len(),
                "paired_timing_and_output_eligible":row.qualifies()
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"schema_version":1,"artifact_verified":true,
        "evidence_bytes":evidence_bytes,"context":evidence.context,"candidates":candidates,"live_admission_performed":false})
        )?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::DenoiseChunkArgs;
    use crate::cli::denoise::resolve_execution_policy;
    #[test]
    fn selection_output_conflicts_are_refused_before_publication() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("latents.safetensors");
        let sidecar = output.with_extension("resource-selection.json");
        assert!(validate_sidecar_output(&sidecar, &output, &[Some(&sidecar)]).is_err());
        assert!(
            validate_sidecar_output(&sidecar, &output, &[Some(&sidecar.join("checkpoint"))])
                .is_err()
        );
        assert!(!sidecar.exists());
    }
    #[test]
    fn old_runs_need_no_sidecar_and_present_records_must_match_the_sealed_policy() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("resource-selection.json");
        let policy = resolve_execution_policy(
            None,
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            DenoiseChunkArgs::default().configured(false),
            false,
            true,
        )
        .unwrap();
        validate_recorded_selection(&path, &policy).unwrap();
        let recorded_policy = serde_json::to_value(&policy).unwrap();
        let record = serde_json::json!({"schema_version":1,"policy":recorded_policy,"selector_revision":"test",
            "request":{"command":"test"},"model":{"component":"transformer"},"mode":"performance",
            "selection_snapshot":ResourceSnapshot::capture(None),"final_admission_snapshot":null,
            "already_present_at_capture":["metadata"],"inventory":{"shards":[{"name":"a.safetensors","file_bytes":8,"header_bytes":8,"selected_tensor_bytes":0,"selected_tensor_count":1,"largest_tensor_bytes":0}]},
            "phases":[{"phase":"denoise","required_host_bytes":1,"optional_host_bytes":0,"host_promotion_reserve_bytes":0,"required_device_bytes":null,"optional_device_bytes":null,"device_reserve_bytes":0}],
            "axes":[{"axis":"weights.source","value":"mmap","origin":"baseline"}],
            "candidates":[{"candidate_id":"baseline","disposition":"selected","reason":"test","expected_cost":null,"evidence":[]}]});
        std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        validate_recorded_selection(&path, &policy).unwrap();
        let mut changed = record;
        changed["policy"] = serde_json::json!({"schema_version": 0});
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(validate_recorded_selection(&path, &policy).is_err());
    }
}
