use super::device_parse::parse_device;
use super::output_hygiene::{
    ensure_new_output, publish_staged_bytes, resolve_output_outside_model,
};
use super::{GlmCommand, kit, resolve_optional_new_output};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Tensor, safetensors};
use flyingfish::glm::config::GlmConfig;
use flyingfish::glm::routing_trace::{MAX_ROUTING_TRACE_JSON_BYTES, validate_routing_trace_domain};
use flyingfish::glm::{
    GlmGenerationOptions, GlmParityCapture, RoutingReplayOptions, RoutingReplayReport,
    RoutingTrace, StreamedGlm, StreamedGlmOptions,
};
use flyingfish::runtime::artifact::{ArtifactStaging, read_artifact_snapshot};
use flyingfish::runtime::telemetry::TelemetryMonitor;
use flyingfish::runtime::weights::{WeightAccessStats, WeightSource};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

const GLM_PARITY_CAPTURE_SCHEMA_VERSION: u32 = 2;

/// State the measured split between moving an expert to the device and
/// evaluating it here, or say plainly that there is no measurement.
///
/// Both rates belong to this machine and to no other, so nothing is assumed in
/// their absence: a run without a profile reports that rather than scheduling
/// against a number it does not have.
fn report_host_profile(
    explicit: Option<&Path>,
    model: &Path,
    device: &candle_core::Device,
) -> Result<f64> {
    let path = explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(|| model.join(flyingfish::host_profile::DEFAULT_HOST_PROFILE_NAME));
    match flyingfish::host_profile::HostProfile::load(&path, device)? {
        Ok(profile) => {
            let rates = profile.expert_bandwidths();
            eprintln!(
                "preflight host profile: B_P {:.3} GiB/s, B_H {:.3} GiB/s evaluating \
                 ({:.3} GiB/s end to end), B_P/B_H {:.2}; host share of a miss set {:.0}%",
                rates.transfer_bytes_per_second / 1024f64.powi(3),
                rates.host_evaluation_bytes_per_second / 1024f64.powi(3),
                rates.host_service_bytes_per_second / 1024f64.powi(3),
                rates.transfer_over_host(),
                rates.host_share() * 100.0,
            );
            return Ok(rates.host_share());
        }
        // An explicit path that yields nothing is an operator error; an
        // unnamed one simply has no profile yet, which is not a failure.
        Err(absence) if explicit.is_some() => bail!("{absence}"),
        Err(_) => {}
    }
    Ok(0.0)
}

pub(super) fn run_generate(command: GlmCommand) -> Result<()> {
    let GlmCommand::Generate {
        model,
        prompt,
        max_new_tokens,
        max_context_tokens,
        reasoning_effort,
        sampling,
        device,
        weights,
        resource_policy,
        resource_evidence,
        resource_cold_cache,
        resource_selection,
        resident_static,
        no_resident_static,
        host_profile,
        cpu_fp8_dequantization,
        pinned_fp8_transfer,
        expert_cache_mib,
        expert_cache_layout,
        expert_cache_replacement,
        expert_cache_readmit,
        expert_cache_min_mib,
        no_progress,
        output,
        routing_trace,
        routing_trace_domain,
        execution_manifest,
    } = command
    else {
        unreachable!("run_generate received a non-generate GLM command")
    };
    let kit::SamplingArgs {
        temperature,
        top_p,
        seed,
    } = sampling;
    let kit::OutputArgs {
        output,
        json: emit_json,
        telemetry_json,
    } = output;
    let mut explicit_axes = std::collections::BTreeSet::new();
    for (present, axis) in [
        (weights.weight_source.is_some(), "weights.source"),
        (weights.host_cache_mib.is_some(), "weights.cache_bytes"),
        (weights.host_cache_mib.is_some(), "weights.cache_shards"),
        (
            weights.host_cache_granularity.is_some(),
            "weights.granularity",
        ),
        (
            resident_static.is_some() || no_resident_static,
            "resident_static",
        ),
        (
            expert_cache_mib.is_some(),
            "expert_cache.maximum_bound_bytes",
        ),
        (
            expert_cache_mib.is_some() || expert_cache_min_mib.is_some(),
            "expert_cache.minimum_bound_bytes",
        ),
        (expert_cache_layout.is_some(), "expert_cache.layout"),
        (
            expert_cache_replacement.is_some(),
            "expert_cache.replacement",
        ),
        (
            expert_cache_readmit,
            "expert_cache.readmission_interval_tokens",
        ),
    ] {
        if present {
            explicit_axes.insert(axis.to_owned());
        }
    }
    let mut resident_static = resident_static.unwrap_or(false) && !no_resident_static;
    let expert_cache_mib = expert_cache_mib.unwrap_or(0);
    let expert_cache_layout = expert_cache_layout.unwrap_or_default();
    let expert_cache_replacement = expert_cache_replacement.unwrap_or_default();
    let weights = weights.configured();
    let request_reasoning_effort = reasoning_effort.clone();
    let output = output
        .map(|path| {
            let path = resolve_output_outside_model(&path, &model)?;
            ensure_new_output(&path, "GLM output")?;
            Ok::<_, anyhow::Error>(path)
        })
        .transpose()?;
    let resource_selection = resolve_optional_new_output(resource_selection, &model)?;
    let evidence = resource_evidence
        .as_deref()
        .map(flyingfish::resource_policy::evidence::ResourceEvidence::load)
        .transpose()?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    let routing_trace = routing_trace
        .map(|path| {
            let path = resolve_output_outside_model(&path, &model)?;
            ensure_new_output(&path, "GLM routing trace")?;
            Ok::<_, anyhow::Error>(path)
        })
        .transpose()?;
    let execution_manifest = execution_manifest
        .map(|path| {
            let path = resolve_output_outside_model(&path, &model)?;
            ensure_new_output(&path, "GLM execution manifest")?;
            Ok::<_, anyhow::Error>(path)
        })
        .transpose()?;
    let resource_selection = match resource_selection {
        Some(path) => Some(path),
        None => execution_manifest
            .as_ref()
            .or(output.as_ref())
            .map(|path| path.with_extension("resource-selection.json"))
            .map(|path| {
                let path = resolve_output_outside_model(&path, &model)?;
                ensure_new_output(&path, "GLM resource selection")?;
                Ok::<_, anyhow::Error>(path)
            })
            .transpose()?,
    };
    ensure!(
        routing_trace.is_some() || routing_trace_domain.is_none(),
        "--routing-trace-domain requires --routing-trace"
    );
    if routing_trace.is_some() {
        validate_routing_trace_domain(routing_trace_domain.as_deref().unwrap_or("unspecified"))?;
    }
    ensure!(
        !expert_cache_readmit || execution_manifest.is_some(),
        "--expert-cache-readmit requires --execution-manifest"
    );
    ensure!(
        expert_cache_readmit || expert_cache_min_mib.is_none(),
        "--expert-cache-min-mib requires --expert-cache-readmit"
    );
    ensure_distinct_outputs(&[
        ("GLM output", output.as_deref()),
        ("GLM resource selection", resource_selection.as_deref()),
        ("GLM telemetry", telemetry_json.as_deref()),
        ("GLM routing trace", routing_trace.as_deref()),
        ("GLM execution manifest", execution_manifest.as_deref()),
    ])?;
    let cache_policy = weights.cache_policy()?;
    let expert_cache_bytes =
        usize::try_from(super::output_hygiene::mib_to_bytes(expert_cache_mib)?)?;
    let expert_cache_min_bytes = usize::try_from(super::output_hygiene::mib_to_bytes(
        expert_cache_min_mib.unwrap_or(0),
    )?)?;
    ensure!(
        !expert_cache_readmit || expert_cache_bytes > 0,
        "--expert-cache-readmit requires a positive --expert-cache-mib maximum"
    );
    ensure!(
        expert_cache_min_bytes <= expert_cache_bytes,
        "--expert-cache-min-mib exceeds --expert-cache-mib"
    );
    let device = parse_device(&device)?;
    ensure!(
        device.is_cpu() || device.is_cuda(),
        "GLM text inference currently supports CPU or CUDA devices"
    );
    let host_expert_share = report_host_profile(host_profile.as_deref(), &model, &device)?;
    ensure!(
        weights.weight_source == WeightSource::Mmap,
        "GLM inference requires --weight-source mmap"
    );
    let config = GlmConfig::from_model_dir(&model)?;
    validate_generation_request(
        &prompt,
        max_new_tokens.get(),
        max_context_tokens.get(),
        &reasoning_effort,
        temperature,
        top_p,
        config.text_config.index_topk,
    )?;
    let weight_compute = if device.is_cpu() {
        "block_fp8_dequantized_to_f32"
    } else {
        "block_fp8_dequantized_to_bf16"
    };
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let mut engine_options = StreamedGlmOptions::new(weights.weight_source, cache_policy, device)
        .with_cpu_fp8_dequantization(cpu_fp8_dequantization)
        .with_host_expert_share(host_expert_share)
        .with_pinned_fp8_transfer(pinned_fp8_transfer)
        .with_resident_static(resident_static)
        .with_progress(!no_progress)
        .with_expert_cache_bytes(expert_cache_bytes)
        .with_expert_cache_policy(expert_cache_layout, expert_cache_replacement);
    if expert_cache_readmit {
        engine_options = engine_options.with_adaptive_expert_cache(expert_cache_min_bytes);
    }
    let mut prepared = StreamedGlm::prepare(&model, engine_options)?;
    let prompt_tokens = prepared.prompt_token_count(&prompt, &reasoning_effort)?;
    let request_tokens = prompt_tokens
        .checked_add(max_new_tokens.get())
        .context("GLM request token count overflow")?;
    ensure!(
        request_tokens <= max_context_tokens.get(),
        "GLM request needs at most {request_tokens} tokens but max_context_tokens is {}",
        max_context_tokens.get()
    );
    let request_identity = json!({
        "family":"glm", "prompt":prompt, "reasoning_effort":reasoning_effort,
        "max_new_tokens":max_new_tokens.get(), "max_context_tokens":max_context_tokens.get(),
        "temperature":temperature, "top_p":top_p, "seed":seed,
    });
    let mut context = flyingfish::resource_policy::evidence::EvidenceContext::collect(
        &model,
        prepared.device(),
        request_identity,
    )?;
    let cache_preparation = if resource_cold_cache {
        let observation = flyingfish::resource_policy::prepare_cold_cache(&model)?;
        context.cache_state = flyingfish::resource_policy::evidence::CacheState::Cold;
        Some(observation)
    } else {
        None
    };
    let breakdown = prepared.estimate(prompt_tokens)?;
    let selection_snapshot =
        flyingfish::runtime::probe::ResourceSnapshot::capture(Some(prepared.device()));
    let mut baseline = prepared.execution_policy().clone();
    let mut automatic_axes = Vec::new();
    if prepared.device().is_cuda()
        && resource_policy
            == flyingfish::runtime::resource_selection::ResourcePolicyMode::Performance
        && evidence.is_none()
    {
        if !baseline.resident_static
            && !explicit_axes.contains("resident_static")
            && breakdown
                .validate_capacity(
                    true,
                    flyingfish::glm::admission::ExpertCacheBound::new(
                        expert_cache_bytes,
                        baseline.expert_cache.layout,
                    ),
                    baseline.cache_policy()?,
                    &selection_snapshot,
                )
                .is_ok()
        {
            baseline.resident_static = true;
            automatic_axes.push("resident_static");
        }
        if !explicit_axes.contains("expert_cache.maximum_bound_bytes") {
            let phases = breakdown.phases_with_safety(
                baseline.resident_static,
                flyingfish::glm::admission::ExpertCacheBound::new(0, baseline.expert_cache.layout),
                baseline.cache_policy()?,
                breakdown.scaled_admission_safety_bytes(&selection_snapshot),
            )?;
            let bytes = breakdown.automatic_expert_cache_bytes(
                &phases,
                &selection_snapshot,
                baseline.expert_cache.layout,
            )? as u64;
            baseline.expert_cache.maximum_bound_bytes = bytes;
            baseline.expert_cache.minimum_bound_bytes = bytes;
            if bytes > 0 && !explicit_axes.contains("expert_cache.replacement") {
                baseline.expert_cache.replacement =
                    flyingfish::glm::ExpertCacheReplacementPolicy::Lfu;
                automatic_axes.push("expert_cache.replacement");
            }
            automatic_axes.extend([
                "expert_cache.maximum_bound_bytes",
                "expert_cache.minimum_bound_bytes",
            ]);
        }
    }
    let mut selection = match flyingfish::resource_policy::glm::select(
        &baseline,
        &breakdown,
        &selection_snapshot,
        &context,
        resource_policy,
        &explicit_axes,
        evidence.as_ref().map(|(record, _)| record),
    ) {
        Ok(selection) => selection,
        Err(error) => {
            // A refusal still publishes its per-candidate record when a
            // sidecar path was given, so the rejection can be calibrated
            // against instead of guessed about.
            flyingfish::resource_policy::report_refusal(
                &error,
                resource_selection
                    .as_deref()
                    .map(flyingfish::resource_policy::refusal_path_for)
                    .as_deref(),
            );
            return Err(error);
        }
    };
    if !automatic_axes.is_empty() {
        selection.provenance.selector_revision = "glm-resource-capacity-v2".into();
        for axis in &mut selection.provenance.axes {
            if automatic_axes.contains(&axis.axis.as_str()) {
                axis.origin = flyingfish::runtime::resource_selection::SelectionOrigin::CostModel;
            }
        }
        for candidate in &mut selection.provenance.candidates {
            if candidate.disposition
                == flyingfish::runtime::resource_selection::CandidateDisposition::Selected
            {
                candidate.reason = "CUDA static/expert retention within measured capacity".into();
            }
        }
    }
    prepared.select_execution_policy(&selection.policy)?;
    resident_static = selection.policy.resident_static;
    let admission_snapshot =
        flyingfish::runtime::probe::ResourceSnapshot::capture(Some(prepared.device()));
    // A promotion was admitted only because its peaks plus the promotion
    // reserve fitted; that gate is re-run here, not just the ordinary peaks.
    let selection_cache =
        flyingfish::glm::admission::ExpertCacheBound::from_policy(&selection.policy.expert_cache)?;
    let promotion_refusal = if selection.policy.weights != baseline.weights {
        flyingfish::glm::admission::GlmAdmissionBreakdown::promotion_headroom_error(
            &breakdown,
            resident_static,
            selection_cache,
            selection.policy.cache_policy()?,
            &admission_snapshot,
        )
    } else {
        None
    };
    if let Err(error) = breakdown
        .validate_capacity(
            resident_static,
            selection_cache,
            selection.policy.cache_policy()?,
            &admission_snapshot,
        )
        .map_err(anyhow::Error::from)
        .and_then(|()| match promotion_refusal {
            Some(message) => anyhow::bail!("{message}"),
            None => Ok(()),
        })
    {
        // The capacity race this second snapshot detects publishes its ledger
        // like any other refusal; it is the one that carries both snapshots.
        // Rebuilt from the final snapshot: `validate_capacity` scales its
        // reserve from that snapshot's pool, so selection-time phases and
        // applied reserve cannot reproduce the check that rejected the run.
        let mut provenance = selection.provenance.clone();
        let final_safety = breakdown.scaled_admission_safety_bytes(&admission_snapshot);
        provenance.phases = breakdown.phases_with_safety(
            resident_static,
            selection_cache,
            selection.policy.cache_policy()?,
            final_safety,
        )?;
        provenance
            .workload
            .insert("admission_safety_bytes_applied".into(), final_safety);
        if selection.policy.weights != baseline.weights {
            let promotion =
                flyingfish::glm::admission::GlmAdmissionBreakdown::scaled_promotion_reserve_bytes(
                    &admission_snapshot,
                );
            for phase in &mut provenance.phases {
                phase.host_promotion_reserve_bytes = promotion;
            }
        }
        provenance.final_admission_snapshot = Some(admission_snapshot);
        provenance.workload.insert("refused".into(), 1);
        for candidate in &mut provenance.candidates {
            if candidate.disposition
                == flyingfish::runtime::resource_selection::CandidateDisposition::Selected
            {
                candidate.disposition =
                    flyingfish::runtime::resource_selection::CandidateDisposition::CapacityRejected;
                candidate.reason = format!("refused by final admission: {error}");
                candidate.shortfall = error
                    .downcast_ref::<flyingfish::glm::admission::CapacityRejection>()
                    .and_then(flyingfish::glm::admission::CapacityRejection::shortfall);
            }
        }
        let refusal = anyhow::Error::from(flyingfish::resource_policy::AdmissionRefused {
            provenance,
            summary: format!("GLM final admission refused: {error}"),
        });
        flyingfish::resource_policy::report_refusal(
            &refusal,
            resource_selection
                .as_deref()
                .map(flyingfish::resource_policy::refusal_path_for)
                .as_deref(),
        );
        return Err(refusal);
    }
    // The admitted record is rebuilt from the same snapshot for the same
    // reason: what it reports must be what the final check actually applied.
    let final_safety = breakdown.scaled_admission_safety_bytes(&admission_snapshot);
    selection.provenance.phases = breakdown.phases_with_safety(
        resident_static,
        selection_cache,
        selection.policy.cache_policy()?,
        final_safety,
    )?;
    selection
        .provenance
        .workload
        .insert("admission_safety_bytes_applied".into(), final_safety);
    if selection.policy.weights != baseline.weights {
        let promotion =
            flyingfish::glm::admission::GlmAdmissionBreakdown::scaled_promotion_reserve_bytes(
                &admission_snapshot,
            );
        for phase in &mut selection.provenance.phases {
            phase.host_promotion_reserve_bytes = promotion;
        }
    }
    selection.provenance.final_admission_snapshot = Some(admission_snapshot.clone());
    eprintln!(
        "GLM resource policy {:?}: {} candidates, static={}, expert cache={} bytes",
        resource_policy,
        selection.provenance.candidates.len(),
        resident_static,
        selection.policy.expert_cache.maximum_bound_bytes
    );
    if let Some(chosen) = selection.provenance.candidates.iter().find(|row| {
        row.disposition == flyingfish::runtime::resource_selection::CandidateDisposition::Selected
    }) {
        eprintln!("resource selection: {}", chosen.reason);
    }
    if let Some(path) = &resource_selection {
        flyingfish::resource_policy::publish_selection(path, &selection.provenance)?;
    }
    let engine = prepared.open(prompt_tokens, &admission_snapshot)?;
    let before = engine.access_stats();
    let generation_options = GlmGenerationOptions {
        max_new_tokens: max_new_tokens.get(),
        max_context_tokens: max_context_tokens.get(),
        reasoning_effort,
        temperature,
        top_p,
        seed,
        progress: !no_progress,
    };
    let (result, emitted_trace) = if routing_trace.is_some() {
        let (result, trace) = engine.generate_with_routing_trace(
            &prompt,
            &generation_options,
            routing_trace_domain.as_deref().unwrap_or("unspecified"),
        )?;
        (result, Some(trace))
    } else {
        (engine.generate(&prompt, &generation_options)?, None)
    };
    let access = engine.access_stats().delta_since(&before);
    let cache = engine.cache_stats();
    let expert_cache = engine.expert_cache_stats();
    if expert_cache.max_bytes > 0 && expert_cache.hits == 0 && expert_cache.misses > 1 {
        eprintln!(
            "expert cache observed zero hits across {} misses in this request; its occupancy is not a performance benefit",
            expert_cache.misses
        );
    }
    if let Some(path) = execution_manifest.as_ref() {
        let bytes = result.execution_manifest.canonical_json()?;
        let staging = ArtifactStaging::new(path).with_context(|| {
            format!("failed to stage GLM execution manifest {}", path.display())
        })?;
        publish_staged_bytes(staging, &bytes)?;
        eprintln!("saved GLM execution manifest to {}", path.display());
    }
    if let (Some(path), Some(trace)) = (routing_trace.as_ref(), emitted_trace) {
        let bytes = trace.canonical_json()?;
        let staging = ArtifactStaging::new(path)
            .with_context(|| format!("failed to stage GLM routing trace {}", path.display()))?;
        publish_staged_bytes(staging, &bytes)?;
        eprintln!("saved GLM routing trace to {}", path.display());
    }
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        let report = monitor.finish()?;
        let bytes = serde_json::to_vec_pretty(&report)?;
        let staging = ArtifactStaging::new(path)
            .with_context(|| format!("failed to stage GLM telemetry report {}", path.display()))?;
        publish_staged_bytes(staging, &bytes)?;
        eprintln!("saved GLM runtime telemetry to {}", path.display());
    }

    let rendered = if emit_json {
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "model_family": "glm5_next",
            "execution_profile": "text_only_exact_dsa_short_context",
            "weight_compute": weight_compute,
            "model": model,
            "prompt": prompt,
            "prompt_tokens": result.prompt_tokens,
            "generated_token_ids": &result.generated_token_ids,
            "text": &result.text,
            "prefill_elapsed_ms": duration_ms(result.prefill_elapsed)?,
            "decode_elapsed_ms": duration_ms(result.decode_elapsed)?,
            "token_elapsed_ms": result
                .token_elapsed
                .iter()
                .copied()
                .map(duration_ms)
                .collect::<Result<Vec<_>>>()?,
            "sampling": {
                "reasoning_effort": request_reasoning_effort,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            },
            "execution_policy": engine.execution_policy(),
            "resource_selection": selection.provenance,
            "evidence_context": context,
            "cache_preparation":cache_preparation,
            "admission_snapshot": admission_snapshot,
            "admission_breakdown": engine.admission_breakdown(),
            "resident_static": resident_static,
            "resident_static_bytes": engine.resident_static_bytes(),
            "expert_cache": {
                "bytes": expert_cache.bytes,
                "entries": expert_cache.entries,
                "max_bytes": expert_cache.max_bytes,
                "hits": expert_cache.hits,
                "misses": expert_cache.misses,
                "evictions": expert_cache.evictions,
            },
            "weight_access": weight_access_json(&access),
            "host_cache": cache,
        }))?
    } else {
        result.text.clone()
    };
    if let Some(path) = output.as_ref() {
        let staging = ArtifactStaging::new(path)
            .with_context(|| format!("failed to stage GLM output {}", path.display()))?;
        publish_staged_bytes(staging, rendered.as_bytes())?;
        eprintln!("saved GLM output to {}", path.display());
    } else {
        println!("{rendered}");
    }
    if !emit_json {
        eprintln!(
            "GLM completed: {} prompt tokens, {} generated tokens, prefill {:.2}s, decode {:.2}s",
            result.prompt_tokens,
            result.generated_token_ids.len(),
            result.prefill_elapsed.as_secs_f64(),
            result.decode_elapsed.as_secs_f64()
        );
        eprintln!(
            "GLM weight materializations: {} full tensors, {} gathered rows",
            access.device_tensor_materializations, access.device_row_materializations
        );
        eprintln!(
            "GLM expert cache: {} hits, {} misses, {} evictions, {:.2}/{:.2} GiB",
            expert_cache.hits,
            expert_cache.misses,
            expert_cache.evictions,
            expert_cache.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            expert_cache.max_bytes as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }
    Ok(())
}

pub(super) fn run_replay_routing(command: GlmCommand) -> Result<()> {
    let GlmCommand::ReplayRouting {
        trace,
        output,
        segment_lengths,
        cache_mib,
    } = command
    else {
        unreachable!("run_replay_routing received a non-replay GLM command")
    };
    ensure_new_output(&output, "GLM routing-replay output")?;
    let options = RoutingReplayOptions::new(
        segment_lengths
            .into_iter()
            .map(|length| length.get())
            .collect(),
        cache_mib
            .into_iter()
            .map(|mib| super::output_hygiene::mib_to_bytes(mib.get()))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let snapshot = read_artifact_snapshot(
        &trace,
        u64::try_from(MAX_ROUTING_TRACE_JSON_BYTES)
            .context("routing-trace byte limit exceeds u64")?,
    )
    .with_context(|| format!("failed to read GLM routing trace {}", trace.display()))?;
    let trace = RoutingTrace::from_json(&snapshot.bytes)?;
    let report = RoutingReplayReport::analyze(&trace, snapshot.bytes.len() as u64, &options)?;
    let bytes = report.canonical_json()?;
    let staging = ArtifactStaging::new(&output).with_context(|| {
        format!(
            "failed to stage GLM routing-replay report {}",
            output.display()
        )
    })?;
    publish_staged_bytes(staging, &bytes)?;
    println!(
        "saved GLM routing replay to {} ({} routed tokens, {} cache comparisons)",
        output.display(),
        report.trace.routed_tokens,
        report.cache_replays.len()
    );
    Ok(())
}

pub(super) fn run_capture_parity(command: GlmCommand) -> Result<()> {
    let GlmCommand::CaptureParity {
        model,
        prompt,
        max_context_tokens,
        reasoning_effort,
        device,
        weights,
        resident_static,
        no_progress,
        output,
    } = command
    else {
        unreachable!("run_capture_parity received a non-capture GLM command")
    };
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "GLM parity capture")?;
    ensure!(
        !prompt.trim().is_empty(),
        "GLM parity prompt must not be empty"
    );
    ensure!(
        matches!(reasoning_effort.as_str(), "low" | "high" | "max"),
        "GLM reasoning effort must be low, high, or max"
    );
    ensure!(
        weights.weight_source == WeightSource::Mmap,
        "GLM parity capture requires --weight-source mmap"
    );
    let config = GlmConfig::from_model_dir(&model)?;
    ensure!(
        max_context_tokens.get() <= config.text_config.index_topk,
        "GLM parity capture max_context_tokens exceeds index_topk {}",
        config.text_config.index_topk
    );
    let device = parse_device(&device)?;
    ensure!(
        device.is_cpu() || device.is_cuda(),
        "GLM parity capture supports CPU or CUDA"
    );
    let engine = StreamedGlm::open(
        &model,
        StreamedGlmOptions::new(weights.weight_source, weights.cache_policy()?, device)
            .with_resident_static(resident_static)
            .with_progress(!no_progress),
    )?;
    let capture = engine.capture_first_next_token_parity(
        &prompt,
        &reasoning_effort,
        max_context_tokens.get(),
        !no_progress,
    )?;
    let prompt_tokens = capture.prompt_token_ids.len();
    let next_token_id = capture.next_token_id;
    let tensors = parity_capture_tensors(capture)?;
    let staging = ArtifactStaging::new_for_path_producer(&output)
        .with_context(|| format!("failed to stage GLM parity capture {}", output.display()))?;
    safetensors::save(&tensors, staging.producer_path())
        .with_context(|| format!("failed to save GLM parity capture {}", output.display()))?;
    let published = staging.publish()?;
    println!(
        "saved GLM parity capture to {} ({prompt_tokens} prompt tokens, next token {next_token_id})",
        published.destination.display(),
    );
    Ok(())
}

fn parity_capture_tensors(capture: GlmParityCapture) -> Result<HashMap<String, Tensor>> {
    let prompt_tokens = capture.prompt_token_ids.len();
    let mut tensors = HashMap::new();
    tensors.insert(
        "schema_version".to_owned(),
        Tensor::new(GLM_PARITY_CAPTURE_SCHEMA_VERSION, &Device::Cpu)?,
    );
    tensors.insert(
        "prompt_token_count".to_owned(),
        Tensor::new(
            u32::try_from(prompt_tokens).context("GLM parity prompt count exceeds u32")?,
            &Device::Cpu,
        )?,
    );
    tensors.insert(
        "prompt_token_ids".to_owned(),
        Tensor::from_vec(capture.prompt_token_ids, prompt_tokens, &Device::Cpu)?,
    );
    tensors.insert("final_hidden_state".to_owned(), capture.final_hidden_state);
    tensors.insert("next_token_logits".to_owned(), capture.next_token_logits);
    tensors.insert(
        "next_token_id".to_owned(),
        Tensor::new(capture.next_token_id, &Device::Cpu)?,
    );
    Ok(tensors)
}

pub(super) fn ensure_distinct_outputs(outputs: &[(&str, Option<&std::path::Path>)]) -> Result<()> {
    for (index, (left_label, left)) in outputs.iter().enumerate() {
        let Some(left) = left else {
            continue;
        };
        for (right_label, right) in &outputs[index + 1..] {
            if right.is_some_and(|right| right == *left) {
                anyhow::bail!(
                    "{left_label} conflicts with {right_label}: {}",
                    left.display()
                );
            }
        }
    }
    Ok(())
}

fn duration_ms(duration: std::time::Duration) -> Result<u64> {
    u64::try_from(duration.as_millis()).context("GLM duration in milliseconds exceeds u64")
}

pub(super) fn validate_generation_request(
    prompt: &str,
    max_new_tokens: usize,
    max_context_tokens: usize,
    reasoning_effort: &str,
    temperature: f64,
    top_p: f64,
    exact_context_limit: usize,
) -> Result<()> {
    ensure!(!prompt.trim().is_empty(), "GLM prompt must not be empty");
    ensure!(
        matches!(reasoning_effort, "low" | "high" | "max"),
        "GLM reasoning effort must be low, high, or max"
    );
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "GLM temperature must be finite and non-negative"
    );
    ensure!(
        temperature == 0.0 || temperature.recip().is_finite(),
        "GLM positive temperature is too small to invert without overflow"
    );
    ensure!(
        top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
        "GLM top_p must be finite and in (0, 1]"
    );
    ensure!(
        max_context_tokens > 0 && max_context_tokens <= exact_context_limit,
        "GLM text-only exact-attention profile supports max_context_tokens in 1..={exact_context_limit}"
    );
    ensure!(
        max_new_tokens > 0 && max_new_tokens <= max_context_tokens,
        "GLM max_new_tokens must be in 1..=max_context_tokens"
    );
    Ok(())
}

fn weight_access_json(access: &WeightAccessStats) -> serde_json::Value {
    json!({
        "device_tensor_materializations": access.device_tensor_materializations,
        "device_row_materializations": access.device_row_materializations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flyingfish::glm::routing_trace::{
        ExpertCacheEntryDtype, ExpertCacheEntryUnit, ROUTING_TRACE_SCHEMA_VERSION, RoutingDecision,
        RoutingModelFamily, RoutingTraceLayer, RoutingTracePhase,
    };

    #[test]
    fn duration_conversion_rejects_u64_millisecond_overflow() {
        let duration = Duration::new(u64::MAX, 999_999_999);
        let error = duration_ms(duration).unwrap_err();
        assert!(error.to_string().contains("exceeds u64"));
    }

    #[test]
    fn glm_outputs_must_be_pairwise_distinct() {
        let path = std::path::Path::new("same.json");
        let error = ensure_distinct_outputs(&[
            ("result", Some(path)),
            ("trace", Some(path)),
            ("telemetry", None),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("result conflicts with trace"));
    }

    #[test]
    fn request_validation_rejects_invalid_inputs_before_model_open() {
        for (prompt, max_new, max_context, effort, temperature, top_p, expected) in [
            (" ", 1, 8, "max", 1.0, 0.95, "prompt must not be empty"),
            ("x", 1, 8, "medium", 1.0, 0.95, "reasoning effort"),
            (
                "x",
                1,
                8,
                "max",
                f64::from_bits(1),
                0.95,
                "too small to invert",
            ),
            ("x", 1, 8, "max", 1.0, 0.0, "top_p"),
            ("x", 1, 9, "max", 1.0, 0.95, "1..=8"),
            ("x", 9, 8, "max", 1.0, 0.95, "1..=max_context_tokens"),
        ] {
            let error = validate_generation_request(
                prompt,
                max_new,
                max_context,
                effort,
                temperature,
                top_p,
                8,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "unexpected error: {error:#}"
            );
        }
    }

    #[test]
    fn routing_replay_handler_reads_strict_trace_and_publishes_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let trace_path = directory.path().join("trace.json");
        let output = directory.path().join("replay.json");
        let sequence = [0, 0, 1, 0];
        let trace = RoutingTrace {
            schema_version: ROUTING_TRACE_SCHEMA_VERSION,
            routed_scaling_factor: Some(1.0),
            norm_topk_prob: Some(false),
            prefill_schedule: flyingfish::glm::routing_trace::RoutingPrefillSchedule::TokenSerial,
            model_family: RoutingModelFamily::Glm5Next,
            domain: "cli-test".to_owned(),
            cache_entry_dtype: ExpertCacheEntryDtype::Bfloat16,
            cache_entry_unit: ExpertCacheEntryUnit::RoutedExpertAllProjections,
            num_hidden_layers: 2,
            num_experts: 4,
            experts_per_token: 1,
            prompt_tokens: 2,
            generated_tokens: 3,
            routed_tokens: 4,
            layers: vec![RoutingTraceLayer {
                layer_index: 1,
                expert_bytes: vec![1024 * 1024; 4],
            }],
            decisions: sequence
                .into_iter()
                .enumerate()
                .map(|(token_index, expert)| RoutingDecision {
                    token_index: token_index as u32,
                    phase: if token_index < 2 {
                        RoutingTracePhase::Prefill
                    } else {
                        RoutingTracePhase::Decode
                    },
                    layer_index: 1,
                    experts: vec![expert],
                    gate_weights: vec![0.5],
                })
                .collect(),
        };
        std::fs::write(&trace_path, trace.canonical_json().unwrap()).unwrap();
        let command = || GlmCommand::ReplayRouting {
            trace: trace_path.clone(),
            output: output.clone(),
            segment_lengths: vec![std::num::NonZeroUsize::new(2).unwrap()],
            cache_mib: vec![std::num::NonZeroU64::new(1).unwrap()],
        };
        run_replay_routing(command()).unwrap();
        let bytes = std::fs::read(&output).unwrap();
        let report = RoutingReplayReport::from_json(&bytes).unwrap();
        assert_eq!(report.trace.domain, "cli-test");
        assert_eq!(report.cache_replays.len(), 4);

        let error = run_replay_routing(command()).unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert_eq!(std::fs::read(&output).unwrap(), bytes);
    }
}
