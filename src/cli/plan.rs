use super::H3Command;
use super::checkpoint::resolve_component;
use super::device_parse::parse_device_single;
use super::output_hygiene::mib_to_bytes;
use super::{
    DEFAULT_FLASH_BACKEND_WORKSPACE_MIB, DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB, H3AdmissionArgs,
    OptionalWeightCacheArgs,
};
use anyhow::{Context, Result, bail};
use flyingfish::h3::config::TransformerConfig;
use flyingfish::h3::core::{
    AttentionKeyChunkPolicy, CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS,
    DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE, DEFAULT_ATTENTION_QUERY_CHUNK_SIZE,
    DEFAULT_FFN_TOKEN_CHUNK_SIZE,
};
use flyingfish::h3::execution::H3ExecutionPlan;
use flyingfish::h3::model::DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE;
use flyingfish::h3::policy::ExecutionPolicy;
use flyingfish::h3::resources::{
    H3ResourceBudgetExt, ResourceAssumptions, ResourceBudget, ResourceEstimate, T2vaGeometry,
    TransformerShape,
};
use flyingfish::h3::solver::{
    AttentionBackendAllowlist, PolicySearchSpace, solve_feasible_policies,
};
use flyingfish::runtime::weights::{CacheGranularity, CachePolicy, ModelWeights, WeightSource};

#[derive(serde::Serialize)]
struct SolveT2vaCandidate {
    candidate_id: String,
    peak_host_bytes: u64,
    peak_device_bytes: u64,
    policy: ExecutionPolicy,
    estimate: ResourceEstimate,
}

#[derive(serde::Serialize)]
struct SolveT2vaSearch {
    attention_backends: [&'static str; 1],
    weight_source: &'static str,
    winner_selected: bool,
    presentation_order: [&'static str; 3],
    attention_projection_rows: Vec<usize>,
    attention_query_rows: Vec<usize>,
    attention_key_rows: Vec<usize>,
    feed_forward_rows: Vec<usize>,
    output_rows: Vec<usize>,
    precompute_adaln: Vec<bool>,
}

#[derive(serde::Serialize)]
struct MmapHostAdmission {
    largest_shard_file_bytes: u64,
    mmap_transition_mapping_count: u64,
    mmap_transition_bytes: u64,
    additional_allowance_bytes: u64,
    charged_host_weight_residency_bytes: u64,
    owned_weight_bytes: u64,
}

#[derive(serde::Serialize)]
struct SolveT2vaReport {
    schema_version: u32,
    resource_estimate_schema_version: u32,
    model: TransformerShape,
    geometry: T2vaGeometry,
    assumptions: ResourceAssumptions,
    host_admission: MmapHostAdmission,
    search: SolveT2vaSearch,
    resource_policy: flyingfish::runtime::resource_selection::ResourcePolicyMode,
    weight_selection: Option<flyingfish::runtime::resource_selection::ResourceSelectionProvenance>,
    hard_budget: ResourceBudget,
    total_feasible_candidates: usize,
    returned_candidates: usize,
    truncated: bool,
    candidates: Vec<SolveT2vaCandidate>,
}

pub(super) fn run_plan_t2va(command: H3Command) -> Result<()> {
    let H3Command::PlanT2va {
        model,
        component,
        text_rows,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        target,
        chunks,
        sigma_points,
        start_step,
        max_steps,
        cpu,
        no_precompute_adaln,
        flash_attention,
        host_cache_mib,
        backend_workspace_mib,
        max_host_mib,
        max_device_mib,
        execution_plan,
        json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for plan-t2va");
    };
    anyhow::ensure!(!(cpu && flash_attention), "FlashAttention requires CUDA");
    let chunks = chunks.configured(flash_attention);
    let (latent_frames, latent_height, latent_width, audio_frames) = target
        .resolve_latent_geometry((latent_frames, latent_height, latent_width, audio_frames))?;
    let component_dir = resolve_component(&model, &component)?;
    let config = TransformerConfig::from_file(component_dir.join("config.json"))?;
    let mut geometry = T2vaGeometry {
        text_rows,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        attention_query_chunk_size: chunks.attention_query_chunk_size.get(),
        attention_projection_chunk_size: chunks.attention_projection_chunk_size.get(),
        attention_key_chunk_policy: chunks.attention_key_policy(),
        ffn_token_chunk_size: chunks.ffn_token_chunk_size.get(),
        output_token_chunk_size: chunks.output_token_chunk_size.get(),
    };
    let sequence_rows = geometry.sequence_rows(config.patch_size)?.total;
    let models_tuned_kernels = !cpu && !flyingfish::h3::cuda::profile::tuned_kernels_disabled();
    // What a run of this request would use. Full softmax covers a bounded
    // number of packed rows; past it the run would be planned and executed on
    // online softmax over key blocks, so the plan models that rather than
    // refusing to answer.
    if models_tuned_kernels
        && !flash_attention
        && chunks.attention_key_policy().is_full()
        && sequence_rows > CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS as u64
    {
        geometry.attention_key_chunk_policy =
            flyingfish::h3::core::AttentionKeyChunkPolicy::chunked(CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS)
                .context("verified exact-softmax row bound is not a usable key block")?;
        eprintln!(
            "attention: {sequence_rows} packed rows exceed the \
             {CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS} full softmax covers; planning online softmax \
             over {CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS}-row key blocks"
        );
    }
    let weights = ModelWeights::open(&component_dir, WeightSource::Mmap, CachePolicy::new(1))?;
    let plan = H3ExecutionPlan::from_config(&weights, &config)?;
    let inventory = weights.inventory();
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    if cpu {
        assumptions.weight_element_bytes = 4;
        assumptions.activation_element_bytes = 4;
        assumptions.device_memory_is_host = true;
    }
    assumptions.use_flash_attention = flash_attention;
    assumptions.evaluation_count = selected_evaluation_count(sigma_points, start_step, max_steps)?;
    assumptions.precompute_adaln_steps = if no_precompute_adaln {
        0
    } else {
        assumptions.evaluation_count
    };
    let host_admission = conservative_mmap_host_admission(&weights, host_cache_mib)?;
    assumptions.host_weight_cache_bytes = host_admission.additional_allowance_bytes;
    assumptions.mapped_weight_residency_bytes = host_admission.mmap_transition_bytes;
    let backend_workspace_mib = backend_workspace_mib.unwrap_or(if cpu {
        0
    } else if flash_attention {
        DEFAULT_FLASH_BACKEND_WORKSPACE_MIB
    } else {
        DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB
    });
    assumptions.backend_workspace_bytes = mib_to_bytes(backend_workspace_mib)?;
    assumptions.checkpoint_weight_bytes = inventory.indexed_bytes;
    assumptions.peak_materialized_weight_bytes_override = Some(
        plan.peak_stage_weight_bytes()
            .checked_mul(if cpu { 2 } else { 1 })
            .context("materialized stage size overflow")?,
    );
    let estimate = ResourceEstimate::for_t2va(&config, geometry, assumptions)?;
    let budget = resource_budget(max_host_mib, max_device_mib)?;
    let report = estimate.check_budget(budget);
    if json {
        println!("{}", estimate.to_pretty_json()?);
    } else {
        if execution_plan {
            print_execution_plan_stages(&plan);
        }
        println!("{estimate}");
        println!("{report}");
    }
    budget.validate(&estimate)?;
    Ok(())
}

pub(super) fn run_solve_t2va(command: H3Command) -> Result<()> {
    let H3Command::SolveT2va {
        resources,
        model,
        component,
        text_rows,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        target,
        sigma_points,
        start_step,
        video_shift,
        audio_shift,
        max_steps,
        cpu,
        device,
        host_cache_mib,
        backend_workspace_mib,
        max_host_mib,
        max_device_mib,
        limit,
        json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for solve-t2va");
    };

    let (latent_frames, latent_height, latent_width, audio_frames) = target
        .resolve_latent_geometry((latent_frames, latent_height, latent_width, audio_frames))?;

    let evaluation_count = selected_evaluation_count(sigma_points, start_step, max_steps)?;

    let component_dir = resolve_component(&model, &component)?;
    let config = TransformerConfig::from_file(component_dir.join("config.json"))?;
    let weights = ModelWeights::open(&component_dir, WeightSource::Mmap, CachePolicy::new(1))?;
    let execution_plan = H3ExecutionPlan::from_config(&weights, &config)?;
    let inventory = weights.inventory();

    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    if cpu {
        assumptions.weight_element_bytes = 4;
        assumptions.activation_element_bytes = 4;
        assumptions.device_memory_is_host = true;
    } else {
        // The solver requires `device_memory_is_host = false` for CUDA and
        // checks each axis against its own bound, which on a shared pool can
        // call a candidate feasible whose two axes together exceed it. Refused
        // rather than reported wrongly until the solver models a combined bound.
        // Only an omitted device skips probing: a named one that cannot be
        // opened is an error, not the symbolic mode, and silently treating it
        // as such would emit a split-axis solution for an unknown topology.
        let probe = device
            .as_deref()
            .map(|name| -> Result<_> {
                Ok(flyingfish::runtime::probe::ResourceSnapshot::capture(Some(
                    &parse_device_single(name)?,
                )))
            })
            .transpose()?;
        anyhow::ensure!(
            probe.is_none_or(|snapshot| {
                snapshot.host_device_memory_is_unified != Some(true)
                    && !snapshot.unified_accounting_is_undecidable()
            }),
            "solve-t2va cannot model a unified-memory device: its candidate search checks the host \
             and device axes independently, which overstates feasibility on one shared pool"
        );
    }
    assumptions.evaluation_count = evaluation_count;
    assumptions.precompute_adaln_steps = assumptions.evaluation_count;
    let mut host_admission = conservative_mmap_host_admission(&weights, host_cache_mib)?;
    assumptions.host_weight_cache_bytes = host_admission.additional_allowance_bytes;
    assumptions.mapped_weight_residency_bytes = host_admission.mmap_transition_bytes;
    assumptions.backend_workspace_bytes = mib_to_bytes(backend_workspace_mib.unwrap_or(if cpu {
        0
    } else {
        DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB
    }))?;
    assumptions.checkpoint_weight_bytes = inventory.indexed_bytes;
    assumptions.peak_materialized_weight_bytes_override = Some(
        execution_plan
            .peak_stage_weight_bytes()
            .checked_mul(if cpu { 2 } else { 1 })
            .context("materialized stage size overflow")?,
    );

    let budget = resource_budget(max_host_mib, max_device_mib)?;
    let geometry = T2vaGeometry {
        text_rows,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        attention_query_chunk_size: DEFAULT_ATTENTION_QUERY_CHUNK_SIZE,
        attention_projection_chunk_size: DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE,
        attention_key_chunk_policy: AttentionKeyChunkPolicy::Full,
        ffn_token_chunk_size: DEFAULT_FFN_TOKEN_CHUNK_SIZE,
        output_token_chunk_size: DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE,
    };
    let model_shape = TransformerShape::from_config(&config);
    let mut base_policy = ExecutionPolicy::conservative(cpu)?;
    if let Some(mib) = host_cache_mib {
        base_policy.weights.cache_bytes = Some(mib_to_bytes(mib)?);
    }
    let weight_selection = if let Some(device_name) = device {
        let device = parse_device_single(&device_name)?;
        anyhow::ensure!(
            device.is_cpu() == cpu,
            "--device must match the --cpu modeling choice"
        );
        base_policy = ExecutionPolicy::from_runtime(
            &device,
            base_policy.weight_source(),
            base_policy.cache_policy()?,
            base_policy.transformer_chunking()?,
            base_policy.flash_attention(),
            base_policy.precompute_adaln,
        )?;
        let selected = super::resource::select_h3(super::resource::H3ResourceRequest {
            additional_host_allowance_bytes: host_admission.additional_allowance_bytes,
            component: &component_dir,
            device: &device,
            baseline: &base_policy,
            geometry,
            rows: None,
            timestep_rows: 2,
            evaluations: usize::try_from(evaluation_count)?,
            limits: H3AdmissionArgs {
                max_host_mib,
                max_device_mib,
                backend_workspace_mib,
            },
            resources: &resources,
            weights: OptionalWeightCacheArgs {
                host_cache_mib,
                ..Default::default()
            },
            locked_origin: None,
            resident_input_bytes: 0,
            request: serde_json::json!({"command":"h3.transformer","geometry":geometry,"evaluations":evaluation_count,"sigma_points":sigma_points,"start_step":start_step,"video_shift":video_shift,"audio_shift":audio_shift}),
        })?;
        base_policy = selected.policy;
        Some(selected.provenance)
    } else {
        anyhow::ensure!(
            resources.resource_evidence.is_none(),
            "evidence-qualified solve requires --device; offline solve cannot establish the execution hardware"
        );
        None
    };
    assumptions.device_weight_cache_bytes = base_policy.weights.device_cache.max_bytes;
    let charge = flyingfish::h3::resources::host_weight_residency_charges(
        &base_policy,
        &weights.cache_inventory()?,
    )?;
    assumptions.host_weight_cache_bytes = charge
        .owned_weight_bytes
        .checked_add(host_admission.additional_allowance_bytes)
        .context("solver host allowance overflow")?;
    assumptions.mapped_weight_residency_bytes = charge.mapped_weight_bytes;
    host_admission.mmap_transition_bytes = charge.mapped_weight_bytes;
    host_admission.owned_weight_bytes = charge.owned_weight_bytes;
    host_admission.charged_host_weight_residency_bytes = assumptions
        .host_weight_cache_bytes
        .checked_add(charge.mapped_weight_bytes)
        .context("solver host residency overflow")?;
    host_admission.mmap_transition_mapping_count =
        if base_policy.weight_source() == WeightSource::Memory {
            0
        } else {
            mapping_count_bound(&weights.cache_inventory()?, base_policy.cache_policy()?)
        };
    let search_space = PolicySearchSpace {
        attention_backends: AttentionBackendAllowlist::parity_safe(),
        ..PolicySearchSpace::default()
    };
    let search_report = SolveT2vaSearch {
        attention_backends: ["full"],
        weight_source: match base_policy.weight_source() {
            WeightSource::Mmap => "mmap",
            WeightSource::Memory => "memory",
        },
        winner_selected: false,
        presentation_order: ["peak_device_bytes", "peak_host_bytes", "candidate_id"],
        attention_projection_rows: search_space
            .attention_projection_rows
            .iter()
            .map(|rows| rows.get())
            .collect(),
        attention_query_rows: search_space
            .attention_query_rows
            .iter()
            .map(|rows| rows.get())
            .collect(),
        attention_key_rows: Vec::new(),
        feed_forward_rows: search_space
            .feed_forward_rows
            .iter()
            .map(|rows| rows.get())
            .collect(),
        output_rows: search_space
            .output_rows
            .iter()
            .map(|rows| rows.get())
            .collect(),
        precompute_adaln: search_space.precompute_adaln.clone(),
    };
    let feasible = solve_feasible_policies(
        &base_policy,
        model_shape,
        geometry,
        assumptions,
        budget,
        &search_space,
        flyingfish::h3::solver::SolverHostWeights {
            inventory: &weights.cache_inventory()?,
            additional_host_allowance_bytes: host_admission.additional_allowance_bytes,
        },
    )?;
    let total_feasible_candidates = feasible.len();
    let candidates = feasible
        .into_iter()
        .take(limit.get())
        .map(|candidate| {
            let peak_host_bytes = candidate.estimate.peak_host_bytes;
            let peak_device_bytes = candidate.estimate.peak_device_bytes;
            SolveT2vaCandidate {
                candidate_id: candidate.candidate_id,
                peak_host_bytes,
                peak_device_bytes,
                policy: candidate.policy,
                estimate: candidate.estimate,
            }
        })
        .collect::<Vec<_>>();
    let report = SolveT2vaReport {
        schema_version: 1,
        resource_estimate_schema_version:
            flyingfish::h3::resources::RESOURCE_ESTIMATE_SCHEMA_VERSION,
        model: model_shape,
        geometry,
        assumptions,
        host_admission,
        search: search_report,
        resource_policy: resources.resource_policy,
        weight_selection,
        hard_budget: budget,
        total_feasible_candidates,
        returned_candidates: candidates.len(),
        truncated: candidates.len() < total_feasible_candidates,
        candidates,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_solve_t2va_report(&report)?;
    }
    anyhow::ensure!(
        total_feasible_candidates > 0,
        "no candidate in the built-in conservative full-softmax mmap search grid fits the hard budget (host {}, device {})",
        display_optional_budget(max_host_mib),
        display_optional_budget(max_device_mib)
    );
    Ok(())
}

fn mapping_count_bound(
    inventory: &ff_core::weights::accounting::CacheInventory,
    policy: CachePolicy,
) -> u64 {
    let total = inventory.shards.len();
    let mut sizes = inventory
        .shards
        .iter()
        .map(|s| s.file_bytes)
        .collect::<Vec<_>>();
    sizes.sort_unstable();
    let mut sum = 0u64;
    let mut retained = 0usize;
    for bytes in sizes.into_iter().take(policy.max_shards) {
        let Some(next) = sum.checked_add(bytes) else {
            break;
        };
        if policy.max_bytes.is_some_and(|limit| next > limit) {
            break;
        }
        sum = next;
        retained += 1;
    }
    retained
        .max(1)
        .saturating_add(usize::from(retained < total))
        .min(total) as u64
}

fn conservative_mmap_host_admission(
    weights: &ModelWeights,
    host_cache_mib: Option<u64>,
) -> Result<MmapHostAdmission> {
    let inventory = weights.cache_inventory()?;
    let largest_shard_file_bytes = inventory.largest_unit_bytes(CacheGranularity::Shard);
    let charge = ff_core::weights::accounting::estimate_cache_residency(
        &inventory,
        WeightSource::Mmap,
        match host_cache_mib {
            Some(mib) => CachePolicy::unbounded_units().with_max_bytes(mib_to_bytes(mib)?),
            None => CachePolicy::new(1),
        },
        ff_core::weights::accounting::CacheLoadLifetimes::SERIAL,
    )?;
    let mmap_transition_bytes = charge.mapped_weight_bytes;
    // Nothing asks for an allowance beyond the mapping any more: the flag that
    // did was the deprecated `--host-weight-cache-mib`.
    let additional_allowance_bytes = 0;
    let charged_host_weight_residency_bytes = mmap_transition_bytes;
    Ok(MmapHostAdmission {
        largest_shard_file_bytes,
        mmap_transition_mapping_count: mapping_count_bound(
            &inventory,
            match host_cache_mib {
                Some(mib) => CachePolicy::unbounded_units().with_max_bytes(mib_to_bytes(mib)?),
                None => CachePolicy::new(1),
            },
        ),
        mmap_transition_bytes,
        additional_allowance_bytes,
        charged_host_weight_residency_bytes,
        owned_weight_bytes: 0,
    })
}

pub(super) fn selected_evaluation_count(
    sigma_points: usize,
    start_step: usize,
    max_steps: Option<usize>,
) -> Result<u64> {
    anyhow::ensure!(sigma_points >= 2, "sigma_points must be at least two");
    let schedule_steps = sigma_points - 1;
    anyhow::ensure!(
        start_step < schedule_steps,
        "start_step {start_step} is outside {schedule_steps} evaluations"
    );
    let remaining = schedule_steps - start_step;
    let selected_steps = match max_steps {
        Some(requested) => {
            anyhow::ensure!(
                requested > 0,
                "max_steps must select at least one evaluation"
            );
            anyhow::ensure!(
                requested <= remaining,
                "max_steps {requested} exceeds the {remaining} remaining evaluations"
            );
            requested
        }
        None => remaining,
    };
    u64::try_from(selected_steps).context("selected step count exceeds u64")
}

fn resource_budget(
    max_host_mib: Option<u64>,
    max_device_mib: Option<u64>,
) -> Result<ResourceBudget> {
    Ok(ResourceBudget {
        max_host_bytes: max_host_mib.map(mib_to_bytes).transpose()?,
        max_device_bytes: max_device_mib.map(mib_to_bytes).transpose()?,
    })
}

fn print_execution_plan_stages(plan: &H3ExecutionPlan) {
    for stage in plan.stages() {
        println!(
            "stage {:<28} {:>4} tensors {:>8.2} MiB",
            stage.kind,
            stage.tensor_names.len(),
            stage.weight_bytes as f64 / 1024f64.powi(2)
        );
    }
    println!(
        "peak stage weights: {:.2} MiB",
        plan.peak_stage_weight_bytes() as f64 / 1024f64.powi(2)
    );
}

fn print_solve_t2va_report(report: &SolveT2vaReport) -> Result<()> {
    println!(
        "feasible policies: {} (showing {}, truncated: {})",
        report.total_feasible_candidates, report.returned_candidates, report.truncated
    );
    println!(
        "hard budget: host {}, device {}",
        display_optional_budget_bytes(report.hard_budget.max_host_bytes),
        display_optional_budget_bytes(report.hard_budget.max_device_bytes)
    );
    println!("search: full softmax only, mmap weights; no winner selected");
    for (index, candidate) in report.candidates.iter().enumerate() {
        println!(
            "candidate {}: {}, peak_host_bytes={}, peak_device_bytes={}",
            index + 1,
            candidate.candidate_id,
            candidate.peak_host_bytes,
            candidate.peak_device_bytes
        );
        println!("  policy={}", serde_json::to_string(&candidate.policy)?);
    }
    Ok(())
}

fn display_optional_budget(value: Option<u64>) -> String {
    value.map_or_else(|| "unbounded".to_owned(), |value| format!("{value} MiB"))
}

fn display_optional_budget_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "unbounded".to_owned(), |value| format!("{value} bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "cuda"))]
    use candle_core::Device;

    #[test]
    fn symbolic_cuda_solver_policy_is_bound_without_a_cuda_build() {
        let policy = ExecutionPolicy::conservative(false).unwrap();
        assert_eq!(
            policy.execution_backend,
            flyingfish::h3::policy::ExecutionBackendPolicy::Cuda
        );
        policy.validate().unwrap();
        #[cfg(not(feature = "cuda"))]
        assert!(
            policy
                .validate_device(&Device::Cpu)
                .unwrap_err()
                .to_string()
                .contains("selected device uses Cpu")
        );
    }

    #[test]
    fn evaluation_selection_is_shared_and_bounded() {
        assert_eq!(selected_evaluation_count(50, 0, None).unwrap(), 49);
        assert_eq!(selected_evaluation_count(50, 7, Some(3)).unwrap(), 3);
        assert!(
            selected_evaluation_count(50, 47, Some(9))
                .unwrap_err()
                .to_string()
                .contains("exceeds the 2 remaining evaluations")
        );
        assert!(selected_evaluation_count(1, 0, None).is_err());
        assert!(selected_evaluation_count(50, 49, None).is_err());
        assert!(selected_evaluation_count(50, 0, Some(0)).is_err());
    }
}
