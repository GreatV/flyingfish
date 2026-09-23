use super::H3Command;
use super::checkpoint::resolve_component;
use super::device_parse::parse_device_single;
use super::output_hygiene::{
    create_new_directory, ensure_new_output, publish_staged_bytes, resolve_output_outside_model,
    write_telemetry,
};
use super::progress;
use super::prompt::{sorted_tensor_names, take_input};
use super::qwen_numerical::validate_qwen_numerical_contract;
use super::{
    TransformerChunkArgs, build_transformer_options, ensure_optional_output_is_distinct,
    resolve_optional_new_output,
};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, safetensors};
use flyingfish::h3::conditioning_provenance::{ExternalPromptProvenance, H3ConditioningProvenance};
use flyingfish::h3::config::TransformerConfig;
use flyingfish::h3::core::AttentionKeyChunkPolicy;
use flyingfish::h3::model::StreamedTransformer;
use flyingfish::h3::pipeline::{
    DenoiseCheckpointEvent, DenoiseObserver, DenoisePreparationEvent, DenoiseStepEvent,
    T2vaExecutionOptions, T2vaLatents, T2vaSchedule, denoise_t2va_with_options_and_observer,
};
use flyingfish::h3::policy::ExecutionPolicy;
use flyingfish::h3::resources::T2vaGeometry;
use flyingfish::recovery::{CheckpointIdentity, PolicyHistory, take_t2va_checkpoint_metadata};
use flyingfish::runtime::artifact::{ArtifactStaging, PublishedArtifact};
use flyingfish::runtime::telemetry::TelemetryMonitor;
use flyingfish::runtime::weights::{CachePolicy, WeightSource};
use memmap2::MmapOptions;
use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(crate) struct CliDenoiseObserver {
    progress: progress::DenoiseProgress,
}

impl CliDenoiseObserver {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            progress: progress::DenoiseProgress::for_stderr(enabled),
        }
    }
}

impl DenoiseObserver for CliDenoiseObserver {
    fn synchronize_device_timings(&self) -> bool {
        self.progress.synchronize_device_timings()
    }

    fn on_preparation_completed(&mut self, event: DenoisePreparationEvent) -> Result<()> {
        self.progress
            .on_preparation_completed(event, &mut std::io::stderr().lock())
    }

    fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()> {
        self.progress
            .on_step_completed(event, &mut std::io::stderr().lock())
    }
}

pub(crate) fn report_h3_device_cache(transformer: &StreamedTransformer) {
    if let Some(plan) = transformer.host_residency_plan() {
        eprintln!(
            "H3 host cache: {}",
            serde_json::json!({"planned_resident_bytes":plan.resident_bytes,"stats":transformer.cache_stats()})
        );
    }
    if let Some(plan) = transformer.device_residency_plan() {
        eprintln!(
            "H3 device cache: {}",
            serde_json::json!({
                "planned_resident_bytes": plan.resident_bytes,
                "stats": transformer.device_cache_stats(),
            })
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_execution_policy(
    path: Option<&Path>,
    device: &Device,
    weight_source: WeightSource,
    cache_policy: CachePolicy,
    chunks: TransformerChunkArgs,
    flash_attention: bool,
    precompute_adaln: bool,
) -> Result<ExecutionPolicy> {
    let policy = match path {
        Some(path) => ExecutionPolicy::load(path)?,
        None => ExecutionPolicy::from_runtime(
            device,
            weight_source,
            cache_policy,
            chunks.model_chunking(),
            flash_attention,
            precompute_adaln,
        )?,
    };
    validate_executable_policy(&policy, device)?;
    Ok(policy)
}

pub(crate) fn validate_executable_policy(policy: &ExecutionPolicy, device: &Device) -> Result<()> {
    policy.validate_device(device)?;
    anyhow::ensure!(
        !policy.flash_attention() || cfg!(feature = "flash-attn"),
        "execution policy selects FlashAttention, but this binary was not compiled with --features flash-attn"
    );
    Ok(())
}

/// Choose an attention backend the request can actually run on.
///
/// The exact full-softmax path covers a bounded number of packed rows. Past
/// that bound it is not that the operator prefers another backend -- full
/// softmax does not run the request at all -- and the row count is known here.
/// Online softmax processes keys in blocks instead, so the bound applies per
/// block rather than to the sequence.
///
/// The block is the largest the verified exact-softmax kernels cover, which is
/// the same bound the full path ran out of: larger blocks mean fewer passes,
/// and the modelled peak grows slowly enough with block size that memory does
/// not decide this. A pinned or replayed policy is left alone, and so is one
/// that already names a backend without the bound.
pub(crate) fn promote_attention_backend_for_rows(
    policy: &mut ExecutionPolicy,
    on_cuda: bool,
    packed_rows: u64,
    operator_named_the_backend: bool,
) -> Result<()> {
    use flyingfish::h3::core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS;
    use flyingfish::h3::policy::AttentionBackendPolicy;

    let bound = u64::try_from(CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS)
        .context("exact softmax row bound exceeds u64")?;
    if operator_named_the_backend
        || !on_cuda
        || policy.attention.backend != AttentionBackendPolicy::FullSoftmax
        || packed_rows <= bound
    {
        return Ok(());
    }
    policy.attention.backend = AttentionBackendPolicy::OnlineSoftmax;
    policy.attention.configured_key_rows = Some(bound);
    policy.rebind_attention_numerics()?;
    eprintln!(
        "attention: {packed_rows} packed rows exceed the {bound} full softmax covers; \
         using online softmax over {bound}-row key blocks"
    );
    Ok(())
}

pub(crate) fn select_resume_execution_policy(
    mut requested: ExecutionPolicy,
    recorded: Option<&ExecutionPolicy>,
    explicit_policy: bool,
    explicit_policy_settings: bool,
    requires_recorded_policy: bool,
) -> Result<ExecutionPolicy> {
    match recorded {
        Some(recorded) => {
            if explicit_policy || explicit_policy_settings {
                // Device residency is planned against whatever the card has
                // free at that moment, so the recorded ceiling describes the
                // first run's machine rather than anything the operator asked
                // for. Two runs minutes apart can plan differently and agree on
                // every choice a person made. Carry the recorded ceiling
                // forward -- keeping the run's placement stable across a
                // resume -- and let admission decide whether it still fits;
                // refusing the resume for it would name a field no H3 command
                // even exposes as a flag.
                if !explicit_policy {
                    requested.weights.device_cache = recorded.weights.device_cache;
                }
                if let Some(field) = recorded.first_difference(&requested) {
                    bail!(
                        "the checkpoint's execution policy disagrees with the requested one at {field}"
                    );
                }
                Ok(requested)
            } else {
                Ok(recorded.clone())
            }
        }
        None if requires_recorded_policy => bail!("checkpoint has no execution policy"),
        None => Ok(requested),
    }
}

const CONDITIONED_SCHEMA_MARKER: &str = "conditioned_schema_version";
const CONDITIONED_MODE_MARKER: &str = "conditioned_mode";
const OFFICIAL_A1_PROMPT_ROWS: usize = 357;
const OFFICIAL_A1_PACKED_ROWS: usize = 808;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PolicyHistoryCheckpointFormat {
    T2va,
    Conditioned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExternalPromptInputKind {
    InitialFixture,
    RecoveryCheckpoint,
}

fn classify_external_prompt_input(path: &Path) -> Result<ExternalPromptInputKind> {
    let path_before = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect external prompt input {}", path.display()))?;
    anyhow::ensure!(
        path_before.file_type().is_file() && path_before.len() > 0,
        "external prompt input must be a non-empty regular non-symlink file"
    );
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open external prompt input {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("failed to inspect external prompt input {}", path.display()))?;
    anyhow::ensure!(
        opened.is_file() && opened.len() == path_before.len(),
        "external prompt input changed while opening"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            (opened.dev(), opened.ino()) == (path_before.dev(), path_before.ino()),
            "external prompt input path changed while opening"
        );
    }
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to map external prompt input {}", path.display()))?;
    let tensors = ::safetensors::SafeTensors::deserialize(&mmap)
        .with_context(|| format!("invalid external prompt safetensors {}", path.display()))?;
    let names = tensors.names().into_iter().collect::<BTreeSet<_>>();
    let external = [
        flyingfish::h3::conditioning_provenance::EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR,
        flyingfish::h3::conditioning_provenance::EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR,
    ]
    .map(|name| names.contains(name));
    anyhow::ensure!(
        external.iter().all(|present| *present) || external.iter().all(|present| !present),
        "external prompt input has an incomplete provenance pair"
    );
    let qwen = [
        flyingfish::h3::policy::H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR,
        flyingfish::h3::policy::H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR,
    ]
    .map(|name| names.contains(name));
    anyhow::ensure!(
        qwen.iter().all(|present| *present) || qwen.iter().all(|present| !present),
        "external prompt input has an incomplete Flyingfish Qwen contract pair"
    );
    anyhow::ensure!(
        !(external[0] && qwen[0]),
        "external prompt input cannot contain both external and Flyingfish Qwen provenance"
    );
    anyhow::ensure!(
        !qwen[0],
        "--external-prompt-manifest cannot be used with Flyingfish Qwen provenance"
    );
    let recovery = [
        "completed_steps",
        flyingfish::recovery::POLICY_HISTORY_JSON_TENSOR,
        flyingfish::recovery::POLICY_HISTORY_SCHEMA_TENSOR,
    ]
    .map(|name| names.contains(name));
    anyhow::ensure!(
        recovery.iter().all(|present| *present) || recovery.iter().all(|present| !present),
        "external prompt input has incomplete recovery metadata"
    );
    match (external[0], recovery[0]) {
        (false, false) => Ok(ExternalPromptInputKind::InitialFixture),
        (true, true) => Ok(ExternalPromptInputKind::RecoveryCheckpoint),
        (false, true) => {
            anyhow::bail!("external recovery checkpoint is missing its provenance pair")
        }
        (true, false) => {
            anyhow::bail!("external provenance pair is only valid in a recovery checkpoint")
        }
    }
}

fn validate_official_a1_execution_policy(
    policy: &ExecutionPolicy,
    prompt_rows: usize,
    packed_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        prompt_rows == OFFICIAL_A1_PROMPT_ROWS && packed_rows == OFFICIAL_A1_PACKED_ROWS,
        "official A1 fixture geometry changed: expected {OFFICIAL_A1_PROMPT_ROWS} prompt and \
         {OFFICIAL_A1_PACKED_ROWS} packed rows, got {prompt_rows} and {packed_rows}"
    );
    anyhow::ensure!(
        policy.attention.backend == flyingfish::h3::policy::AttentionBackendPolicy::FullSoftmax
            && policy.attention.configured_key_rows.is_none(),
        "official A1 fixture requires the full-softmax attention backend"
    );
    for (name, rows) in [
        (
            "attention projection",
            policy.attention.configured_projection_rows,
        ),
        ("attention query", policy.attention.configured_query_rows),
        ("feed-forward", policy.configured_ffn_rows),
        ("output", policy.configured_output_rows),
    ] {
        anyhow::ensure!(
            rows == OFFICIAL_A1_PACKED_ROWS as u64,
            "official A1 fixture requires {name} rows={OFFICIAL_A1_PACKED_ROWS}, got {rows}"
        );
    }
    policy.validate_official_full_packed_output_heads(
        NonZeroUsize::new(packed_rows).context("official A1 packed rows must be non-zero")?,
    )
}

fn official_a1_packed_rows(
    component_dir: &Path,
    prompt_embeddings: &Tensor,
    video_latents: &Tensor,
    audio_latents: &Tensor,
) -> Result<usize> {
    let (prompt_batch, text_rows, _) = prompt_embeddings
        .dims3()
        .context("official A1 prompt embeddings must be rank three")?;
    let (video_batch, _, latent_frames, latent_height, latent_width) = video_latents
        .dims5()
        .context("official A1 video latents must be rank five")?;
    let (audio_channels, _, audio_frames) = audio_latents
        .dims3()
        .context("official A1 audio latents must be rank three")?;
    anyhow::ensure!(
        prompt_batch == 1 && video_batch == 1,
        "official A1 fixture requires single-batch prompt/video tensors"
    );
    let config = TransformerConfig::from_file(component_dir.join("config.json"))?;
    let geometry = T2vaGeometry {
        text_rows,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        attention_projection_chunk_size: 1,
        attention_query_chunk_size: 1,
        attention_key_chunk_policy: AttentionKeyChunkPolicy::Full,
        ffn_token_chunk_size: 1,
        output_token_chunk_size: 1,
    };
    usize::try_from(geometry.sequence_rows(config.patch_size)?.total)
        .context("official A1 packed rows exceed usize")
}

pub(super) struct EvaluationCheckpointObserver<'a> {
    pub(super) progress: CliDenoiseObserver,
    pub(super) directory: Option<PathBuf>,
    pub(super) prompt_embeddings: &'a Tensor,
    pub(super) text_token_tags: &'a Tensor,
    pub(super) sigma_points: usize,
    pub(super) video_shift: f32,
    pub(super) audio_shift: f32,
    pub(super) execution_policy: ExecutionPolicy,
    pub(super) policy_history: PolicyHistory,
    pub(super) conditioning_provenance: &'a H3ConditioningProvenance,
    pub(super) announce_checkpoints: bool,
    pub(super) staging_parent: Option<PathBuf>,
}

impl DenoiseObserver for EvaluationCheckpointObserver<'_> {
    fn synchronize_device_timings(&self) -> bool {
        self.progress.synchronize_device_timings()
    }

    fn on_preparation_completed(&mut self, event: DenoisePreparationEvent) -> Result<()> {
        self.progress.on_preparation_completed(event)
    }

    fn checkpoint_each_evaluation(&self) -> bool {
        self.directory.is_some()
    }

    fn on_checkpoint_boundary(&mut self, event: DenoiseCheckpointEvent<'_>) -> Result<()> {
        let directory = self
            .directory
            .as_ref()
            .context("evaluation checkpoint directory is not configured")?;
        let completed_evaluations = event.latents.completed_steps;
        let mut policy_history = self.policy_history.clone();
        policy_history.append_successful_evaluations(
            u64::try_from(completed_evaluations)
                .context("completed evaluation count exceeds u64")?,
            self.execution_policy.clone(),
        )?;
        let destination = directory.join(format!(
            "checkpoint-step{completed_evaluations:06}.safetensors"
        ));
        let write_started = std::time::Instant::now();
        let (published, identity) = publish_t2va_checkpoint(
            &destination,
            self.staging_parent.as_deref(),
            event.latents,
            self.prompt_embeddings,
            self.text_token_tags,
            self.sigma_points,
            self.video_shift,
            self.audio_shift,
            &policy_history,
            self.conditioning_provenance,
        )
        .with_context(|| {
            format!("failed to publish evaluation checkpoint {completed_evaluations}")
        })?;
        flyingfish::h3::timing::record_checkpoint_write(
            u64::try_from(completed_evaluations)
                .context("completed evaluation count exceeds u64")?,
            write_started.elapsed(),
        );
        anyhow::ensure!(
            identity.completed_evaluations
                == u64::try_from(completed_evaluations)
                    .context("completed evaluation count exceeds u64")?,
            "published checkpoint completed evaluation changed"
        );
        anyhow::ensure!(
            identity.policy_history == policy_history,
            "published checkpoint policy history changed"
        );
        self.policy_history = policy_history;
        if self.announce_checkpoints {
            println!(
                "evaluation checkpoint {}/{}: {} bytes, {}",
                completed_evaluations,
                event.total_steps,
                identity.checkpoint_bytes,
                published.destination.display()
            );
        }
        Ok(())
    }

    fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()> {
        self.progress.on_step_completed(event)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn publish_t2va_checkpoint(
    destination: &Path,
    staging_parent: Option<&Path>,
    latents: &T2vaLatents,
    prompt_embeddings: &Tensor,
    text_token_tags: &Tensor,
    sigma_points: usize,
    video_shift: f32,
    audio_shift: f32,
    policy_history: &PolicyHistory,
    conditioning_provenance: &H3ConditioningProvenance,
) -> Result<(PublishedArtifact, CheckpointIdentity)> {
    let device = latents.video.device();
    anyhow::ensure!(
        latents.audio.device().same_device(device)
            && prompt_embeddings.device().same_device(device)
            && text_token_tags.device().same_device(device),
        "checkpoint tensors are on different devices"
    );
    let completed_steps = Tensor::new(
        u32::try_from(latents.completed_steps).context("completed_steps exceeds u32")?,
        device,
    )?;
    let saved_sigma_points = Tensor::new(
        u32::try_from(sigma_points).context("sigma_points exceeds u32")?,
        device,
    )?;
    let saved_video_shift = Tensor::new(video_shift, device)?;
    let saved_audio_shift = Tensor::new(audio_shift, device)?;
    let mut saved = HashMap::from([
        ("video_latents", latents.video.clone()),
        ("audio_latents", latents.audio.clone()),
        ("prompt_embeddings", prompt_embeddings.clone()),
        ("text_token_tags", text_token_tags.clone()),
        ("completed_steps", completed_steps),
        ("sigma_points", saved_sigma_points),
        ("video_shift", saved_video_shift),
        ("audio_shift", saved_audio_shift),
    ]);
    policy_history.insert_checkpoint_tensors(&mut saved, device)?;
    conditioning_provenance.insert_artifact_tensors(&mut saved, device)?;
    let staging = match staging_parent {
        Some(parent) => {
            ArtifactStaging::new_for_path_producer_with_staging_parent(destination, parent)
        }
        None => ArtifactStaging::new_for_path_producer(destination),
    }
    .with_context(|| format!("failed to stage checkpoint {}", destination.display()))?;
    safetensors::save(&saved, staging.producer_path())
        .with_context(|| format!("failed to save checkpoint {}", destination.display()))?;
    let identity = CheckpointIdentity::collect(staging.producer_path())
        .context("staged checkpoint failed identity/schema validation before publication")?;
    let published = staging.publish()?;
    Ok((published, identity))
}

fn ensure_checkpoint_outputs_are_disjoint(
    checkpoint_directory: Option<&Path>,
    output: &Path,
    telemetry: Option<&Path>,
) -> Result<()> {
    let Some(checkpoint_directory) = checkpoint_directory else {
        return Ok(());
    };
    for (label, path) in [
        ("denoise output", Some(output)),
        ("telemetry output", telemetry),
    ] {
        if let Some(path) = path {
            anyhow::ensure!(
                path != checkpoint_directory
                    && !path.starts_with(checkpoint_directory)
                    && !checkpoint_directory.starts_with(path),
                "checkpoint directory conflicts with {label}: {}",
                path.display()
            );
        }
    }
    Ok(())
}

pub(super) fn run_show_policy_history(command: H3Command) -> Result<()> {
    let H3Command::ShowPolicyHistory { checkpoint, output } = command else {
        bail!("internal CLI dispatch mismatch for show-policy-history");
    };
    anyhow::ensure!(
        checkpoint.is_file(),
        "checkpoint does not exist: {}",
        checkpoint.display()
    );
    let output = output
        .map(|output| -> Result<_> {
            ensure_new_output(&output, "policy-history output")?;
            Ok(output)
        })
        .transpose()?;
    let policy_history = match identify_policy_history_checkpoint_format(&checkpoint)? {
        PolicyHistoryCheckpointFormat::T2va => {
            CheckpointIdentity::collect(&checkpoint)?.policy_history
        }
        PolicyHistoryCheckpointFormat::Conditioned => {
            super::conditioned::validated_policy_history(&checkpoint)?
        }
    };
    let json = policy_history.canonical_json()?;
    match output {
        Some(output) => {
            let staging = ArtifactStaging::new(&output)?;
            publish_staged_bytes(staging, &json)?;
            println!("saved policy history to {}", output.display());
        }
        None => println!(
            "{}",
            String::from_utf8(json).context("policy-history JSON is not UTF-8")?
        ),
    }
    Ok(())
}

fn identify_policy_history_checkpoint_format(path: &Path) -> Result<PolicyHistoryCheckpointFormat> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect checkpoint format {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && metadata.len() > 0,
        "policy-history checkpoint must be a non-empty regular non-symlink file"
    );
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open checkpoint format {}", path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to map checkpoint format {}", path.display()))?;
    let tensors = ::safetensors::SafeTensors::deserialize(&mmap)
        .with_context(|| format!("invalid safetensors checkpoint {}", path.display()))?;
    let names = tensors.names().into_iter().collect::<BTreeSet<_>>();
    let has_conditioned_schema = names.contains(CONDITIONED_SCHEMA_MARKER);
    let has_conditioned_mode = names.contains(CONDITIONED_MODE_MARKER);
    anyhow::ensure!(
        has_conditioned_schema == has_conditioned_mode,
        "checkpoint has an incomplete conditioned format marker; both {CONDITIONED_SCHEMA_MARKER} and {CONDITIONED_MODE_MARKER} are required"
    );
    if has_conditioned_schema {
        return Ok(PolicyHistoryCheckpointFormat::Conditioned);
    }

    const T2VA_SIGNATURE: [&str; 10] = [
        "video_latents",
        "audio_latents",
        "prompt_embeddings",
        "text_token_tags",
        "completed_steps",
        "sigma_points",
        "video_shift",
        "audio_shift",
        "ff_policy_history_json",
        "ff_policy_history_schema",
    ];
    anyhow::ensure!(
        T2VA_SIGNATURE.iter().all(|name| names.contains(name)),
        "checkpoint has neither the explicit conditioned markers nor the complete T2VA checkpoint signature"
    );
    Ok(PolicyHistoryCheckpointFormat::T2va)
}

pub(super) fn run_denoise_t2va(command: H3Command) -> Result<()> {
    let H3Command::DenoiseT2va {
        resources,
        admission,
        model,
        component,
        inputs,
        external_prompt_manifest,
        output,
        device,
        policy,
        weights: optional_weight_args,
        chunks,
        no_precompute_adaln,
        flash_attention,
        no_progress,
        telemetry_json,
        checkpoint_dir,
        max_steps,
        sigma_points,
        video_shift,
        audio_shift,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for denoise-t2va");
    };
    let component_dir = resolve_component(&model, &component)?;
    let output = resolve_output_outside_model(&output, &model)?;
    anyhow::ensure!(
        inputs.is_file(),
        "input safetensors does not exist: {}",
        inputs.display()
    );
    ensure_new_output(&output, "denoise output")?;
    let explicit_policy_settings = optional_weight_args.is_explicit()
        || chunks.is_explicit()
        || no_precompute_adaln
        || flash_attention;
    let weight_args = optional_weight_args.configured();
    let chunks = chunks.configured(flash_attention);
    let device = parse_device_single(&device)?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output])?;
    let checkpoint_dir = checkpoint_dir
        .map(|path| -> Result<PathBuf> {
            let path = resolve_output_outside_model(&path, &model)?;
            ensure_new_output(&path, "checkpoint directory")?;
            Ok(path)
        })
        .transpose()?;
    ensure_checkpoint_outputs_are_disjoint(
        checkpoint_dir.as_deref(),
        &output,
        telemetry_json.as_deref(),
    )?;
    let explicit_policy = policy.is_some();
    let requested_policy = resolve_execution_policy(
        policy.as_deref(),
        &device,
        weight_args.weight_source,
        weight_args.cache_policy()?,
        chunks,
        flash_attention,
        !no_precompute_adaln,
    )?;
    let prevalidated_external = external_prompt_manifest
        .as_deref()
        .map(|manifest| -> Result<Option<ExternalPromptProvenance>> {
            ExternalPromptProvenance::verify_official_fixture_manifest(manifest)?;
            match classify_external_prompt_input(&inputs)? {
                ExternalPromptInputKind::InitialFixture => Ok(Some(
                    ExternalPromptProvenance::from_official_fixture_manifest(manifest, &inputs)?,
                )),
                ExternalPromptInputKind::RecoveryCheckpoint => Ok(None),
            }
        })
        .transpose()?
        .flatten();
    let mut values = safetensors::load(&inputs, &device)
        .with_context(|| format!("failed to load T2VA inputs {}", inputs.display()))?;
    let embedded_provenance = H3ConditioningProvenance::take_artifact_tensors(&mut values)?;
    let conditioning_provenance = match (
        embedded_provenance,
        external_prompt_manifest.as_deref(),
        prevalidated_external,
    ) {
        (Some(H3ConditioningProvenance::FlyingfishQwen(contract)), None, None) => {
            H3ConditioningProvenance::FlyingfishQwen(contract)
        }
        (Some(H3ConditioningProvenance::ExternallyValidated(provenance)), Some(manifest), None) => {
            provenance.verify_manifest(manifest)?;
            H3ConditioningProvenance::ExternallyValidated(provenance)
        }
        (None, Some(_), Some(provenance)) => {
            provenance.verify_initial_input(&inputs)?;
            H3ConditioningProvenance::ExternallyValidated(provenance)
        }
        (None, None, None) => bail!(
            "T2VA inputs have no conditioning provenance; legacy audit-only/unbound artifacts cannot execute. Pass a Flyingfish Qwen artifact or explicitly provide --external-prompt-manifest"
        ),
        (Some(H3ConditioningProvenance::FlyingfishQwen(_)), Some(_), _) => bail!(
            "--external-prompt-manifest cannot be mixed with embedded Flyingfish Qwen provenance"
        ),
        (Some(H3ConditioningProvenance::ExternallyValidated(_)), None, _) => bail!(
            "externally validated prompt provenance requires --external-prompt-manifest on every execution/resume"
        ),
        (None, Some(_), None) => {
            bail!("external recovery checkpoint lost its preflight-validated provenance pair")
        }
        (None, None, Some(_)) => {
            bail!("external prompt provenance was prevalidated without a manifest argument")
        }
        (Some(_), _, Some(_)) => {
            bail!("external prompt input changed its provenance classification while loading")
        }
    };
    let checkpoint_metadata = take_t2va_checkpoint_metadata(&mut values)?;
    let checkpoint_step = checkpoint_metadata
        .as_ref()
        .map(|metadata| usize::try_from(metadata.completed_evaluations))
        .transpose()
        .context("completed_steps exceeds usize")?;
    let checkpoint_history = checkpoint_metadata
        .as_ref()
        .map(|metadata| metadata.policy_history.clone());
    if let Some(history) = checkpoint_history.as_ref() {
        history
            .validate_resume_numerics(&device)
            .context("checkpoint policy history cannot resume on the selected runtime")?;
    }
    let checkpoint_policy = checkpoint_history
        .as_ref()
        .and_then(|history| history.segments.last())
        .map(|segment| &segment.policy);
    let mut execution_policy = {
        select_resume_execution_policy(
            requested_policy,
            checkpoint_policy,
            explicit_policy,
            explicit_policy_settings,
            checkpoint_step.is_some_and(|steps| steps > 0),
        )?
    };
    validate_executable_policy(&execution_policy, &device)?;
    if let Some(metadata) = checkpoint_metadata.as_ref() {
        anyhow::ensure!(
            usize::try_from(metadata.sigma_points)? == sigma_points,
            "checkpoint sigma_points {} disagrees with --sigma-points {sigma_points}",
            metadata.sigma_points
        );
        anyhow::ensure!(
            metadata.video_shift == video_shift,
            "checkpoint video_shift {} disagrees with requested {video_shift}",
            metadata.video_shift
        );
        anyhow::ensure!(
            metadata.audio_shift == audio_shift,
            "checkpoint audio_shift {} disagrees with requested {audio_shift}",
            metadata.audio_shift
        );
    }
    let start_step = checkpoint_step.unwrap_or(0);
    let prompt_embeddings = take_input(&mut values, "prompt_embeddings")?;
    let text_token_tags_tensor = take_input(&mut values, "text_token_tags")?;
    let text_token_tags = text_token_tags_tensor
        .to_vec1::<u32>()
        .context("text_token_tags must be a U32 vector")?;
    let video_latents = take_input(&mut values, "video_latents")?;
    let audio_latents = take_input(&mut values, "audio_latents")?;
    let (prompt_batch, prompt_rows, _) = prompt_embeddings
        .dims3()
        .context("prompt_embeddings must be [batch, rows, width]")?;
    let language_rows = prompt_batch
        .checked_mul(prompt_rows)
        .context("prompt language row count overflow")?;
    if let H3ConditioningProvenance::FlyingfishQwen(contract) = &conditioning_provenance {
        validate_qwen_numerical_contract(contract, &device, language_rows, 0, 0, 0)?;
    }
    anyhow::ensure!(
        values.is_empty(),
        "T2VA inputs contain unexpected tensors: {}",
        sorted_tensor_names(&values).join(", ")
    );
    if matches!(
        &conditioning_provenance,
        H3ConditioningProvenance::ExternallyValidated(_)
    ) {
        let packed_rows = official_a1_packed_rows(
            &component_dir,
            &prompt_embeddings,
            &video_latents,
            &audio_latents,
        )?;
        validate_official_a1_execution_policy(&execution_policy, prompt_rows, packed_rows)
            .context("external official fixture requires its pinned full-shape A1 policy")?;
    }
    let geometry = super::resource::geometry_from_tensors(
        &execution_policy,
        &prompt_embeddings,
        &video_latents,
        &audio_latents,
    )?;
    let evaluations = usize::try_from(super::plan::selected_evaluation_count(
        sigma_points,
        start_step,
        max_steps.map(|n| n.get()),
    )?)?;
    let locked_origin = if checkpoint_policy.is_some() {
        Some(flyingfish::runtime::resource_selection::SelectionOrigin::Recorded)
    } else if explicit_policy {
        Some(flyingfish::runtime::resource_selection::SelectionOrigin::Pinned)
    } else {
        None
    };
    let input_identity = serde_json::json!({
        "file": inputs.display().to_string(),
        "bytes": flyingfish::runtime::artifact::FileStat::of_target(&inputs)?.len(),
    });
    let selection_path = output.with_extension("resource-selection.json");
    super::resource::validate_sidecar_output(
        &selection_path,
        &output,
        &[telemetry_json.as_deref(), checkpoint_dir.as_deref()],
    )?;
    let mut selected = match super::resource::select_h3(super::resource::H3ResourceRequest {
        additional_host_allowance_bytes: 0,
        component: &component_dir,
        device: &device,
        baseline: &execution_policy,
        geometry,
        rows: None,
        timestep_rows: 2,
        evaluations,
        limits: admission,
        resources: &resources,
        weights: optional_weight_args,
        locked_origin,
        resident_input_bytes: super::resource::represented_input_bytes(
            &device,
            &prompt_embeddings,
            &video_latents,
            &audio_latents,
        )?,
        // The input belongs to the request identity: without it two refusals
        // for different inputs with the same geometry are indistinguishable,
        // and the ownership check that replaces a refusal compares exactly this.
        request: serde_json::json!({"command":"h3.transformer", "geometry":geometry,
            "sigma_points":sigma_points, "start_step":start_step, "evaluations":evaluations,
            "video_shift":video_shift, "audio_shift":audio_shift, "input":input_identity}),
    }) {
        Ok(selected) => selected,
        Err(error) => {
            flyingfish::resource_policy::report_refusal(
                &error,
                Some(&flyingfish::resource_policy::refusal_path_for(
                    &selection_path,
                )),
            );
            return Err(error);
        }
    };
    selected.provenance.input = Some(input_identity.clone());
    execution_policy = selected.policy;
    flyingfish::resource_policy::publish_selection(&selection_path, &selected.provenance)?;
    if let Some(path) = checkpoint_dir.as_ref() {
        create_new_directory(path, "checkpoint directory")?;
    }
    let transformer_options = build_transformer_options(device.clone(), &execution_policy)?;
    let transformer = StreamedTransformer::open(component_dir, transformer_options)?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let mut observer = EvaluationCheckpointObserver {
        progress: CliDenoiseObserver::new(!no_progress),
        directory: checkpoint_dir,
        prompt_embeddings: &prompt_embeddings,
        text_token_tags: &text_token_tags_tensor,
        sigma_points,
        video_shift,
        audio_shift,
        execution_policy: execution_policy.clone(),
        policy_history: checkpoint_history.unwrap_or_else(PolicyHistory::new),
        conditioning_provenance: &conditioning_provenance,
        announce_checkpoints: true,
        staging_parent: None,
    };
    let result = denoise_t2va_with_options_and_observer(
        &transformer,
        &prompt_embeddings,
        &text_token_tags,
        &video_latents,
        &audio_latents,
        T2vaSchedule {
            sigma_points,
            video_shift,
            audio_shift,
        },
        T2vaExecutionOptions {
            precompute_adaln: execution_policy.precompute_adaln,
            start_step,
            max_steps: max_steps.map(NonZeroUsize::get),
        },
        &mut observer,
    );
    report_h3_device_cache(&transformer);
    let result = result?;
    let checkpointing = observer.directory.is_some();
    let mut policy_history = observer.policy_history.clone();
    if checkpointing {
        anyhow::ensure!(
            policy_history.completed_evaluations
                == u64::try_from(result.completed_steps).context("completed_steps exceeds u64")?,
            "evaluation checkpoint history did not reach the denoise result"
        );
    } else {
        policy_history.append_successful_evaluations(
            u64::try_from(result.completed_steps).context("completed_steps exceeds u64")?,
            execution_policy.clone(),
        )?;
    }
    drop(observer);
    let (published, published_identity) = publish_t2va_checkpoint(
        &output,
        None,
        &result,
        &prompt_embeddings,
        &text_token_tags_tensor,
        sigma_points,
        video_shift,
        audio_shift,
        &policy_history,
        &conditioning_provenance,
    )
    .with_context(|| format!("failed to publish denoise output {}", output.display()))?;
    let cache = transformer.cache_stats();
    let unit = if cache.tensor_retention.is_some() {
        "tensor"
    } else {
        "shard"
    };
    let residency = match &cache.tensor_retention {
        Some(tensors) => format!(
            "{} retained tensors across {} source shards",
            tensors.resident_tensors, cache.resident_shards
        ),
        None => format!("{} resident shards", cache.resident_shards),
    };
    let warning = if cache.over_budget {
        format!(" [single {unit} exceeds byte budget]")
    } else {
        String::new()
    };
    println!(
        "host {unit} cache: {} hits, {} misses, {} evictions, {} header parses, {residency} ({:.2} GiB){warning}",
        cache.hits,
        cache.misses,
        cache.evictions,
        cache.header_parses,
        cache.resident_bytes as f64 / 1024f64.powi(3),
    );
    println!("saved denoised H3 latents to {}", output.display());
    println!(
        "checkpoint artifact: {} bytes, {:?}",
        published_identity.checkpoint_bytes, published.durability
    );
    println!(
        "execution policy: schema {}",
        execution_policy.schema_version
    );
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::default_execution_policy;

    #[test]
    fn external_checkpoint_writer_and_classifier_use_the_same_metadata() {
        let root = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let history = PolicyHistory::new();
        let provenance = H3ConditioningProvenance::ExternallyValidated(ExternalPromptProvenance {
            schema_version: 1,
            binding: "externally_validated_unbound_to_flyingfish_qwen".into(),
            manifest_bytes: 29_478,
            input_bytes: 3_726_732,
            manifest_producer: "official MiniMax-H3 Modular Diffusers pipeline".into(),
            manifest_schema_version: 2,
        });
        let mut tensors = HashMap::from([("completed_steps", Tensor::new(0u32, &device).unwrap())]);
        history
            .insert_checkpoint_tensors(&mut tensors, &device)
            .unwrap();
        provenance
            .insert_artifact_tensors(&mut tensors, &device)
            .unwrap();
        let path = root.path().join("checkpoint.safetensors");
        safetensors::save(&tensors, &path).unwrap();
        assert_eq!(
            classify_external_prompt_input(&path).unwrap(),
            ExternalPromptInputKind::RecoveryCheckpoint
        );
        let mut loaded = safetensors::load(&path, &device).unwrap();
        assert_eq!(
            H3ConditioningProvenance::take_artifact_tensors(&mut loaded).unwrap(),
            Some(provenance)
        );
        assert_eq!(
            PolicyHistory::take_checkpoint_tensors(&mut loaded).unwrap(),
            Some(history)
        );

        tensors.remove(flyingfish::recovery::POLICY_HISTORY_SCHEMA_TENSOR);
        safetensors::save(&tensors, &path).unwrap();
        assert!(
            classify_external_prompt_input(&path)
                .unwrap_err()
                .to_string()
                .contains("incomplete recovery metadata")
        );
        let initial = HashMap::from([("video_latents", Tensor::new(&[0f32], &device).unwrap())]);
        safetensors::save(&initial, &path).unwrap();
        assert_eq!(
            classify_external_prompt_input(&path).unwrap(),
            ExternalPromptInputKind::InitialFixture
        );
    }

    #[test]
    fn external_official_fixture_requires_the_full_808_row_policy() {
        let mut policy = default_execution_policy(&Device::Cpu);
        policy.attention.configured_projection_rows = OFFICIAL_A1_PACKED_ROWS as u64;
        policy.attention.configured_query_rows = OFFICIAL_A1_PACKED_ROWS as u64;
        policy.configured_ffn_rows = OFFICIAL_A1_PACKED_ROWS as u64;
        policy.configured_output_rows = OFFICIAL_A1_PACKED_ROWS as u64;
        validate_official_a1_execution_policy(
            &policy,
            OFFICIAL_A1_PROMPT_ROWS,
            OFFICIAL_A1_PACKED_ROWS,
        )
        .unwrap();

        let mut chunked = policy.clone();
        chunked.attention.configured_query_rows -= 1;
        assert!(
            validate_official_a1_execution_policy(
                &chunked,
                OFFICIAL_A1_PROMPT_ROWS,
                OFFICIAL_A1_PACKED_ROWS,
            )
            .unwrap_err()
            .to_string()
            .contains("attention query")
        );
        assert!(
            validate_official_a1_execution_policy(
                &policy,
                OFFICIAL_A1_PROMPT_ROWS,
                OFFICIAL_A1_PACKED_ROWS + 1,
            )
            .unwrap_err()
            .to_string()
            .contains("fixture geometry changed")
        );
    }
}
