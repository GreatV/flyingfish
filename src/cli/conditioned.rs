use super::*;
use candle_core::DType;
use flyingfish::{
    h3::audio_vae_encoder::StreamedAudioVaeEncoder,
    h3::fl2va::{Fl2vaCanvas, Fl2vaOptions, PreparedFl2va, prepare_fl2va},
    h3::h3_conditioning::{
        ConditionedLayout, KeyframeAnchor, ReferenceBlock, denoise_conditioned_with_observer,
    },
    h3::multimodal_text_encoder::{
        RgbImage, StreamedMultimodalTextEncoder, resolve_fl2va_canvas_size,
    },
    h3::ref2va::{PreparedRef2va, Ref2vaPipeline, Ref2vaReference, Ref2vaTarget},
    h3::video_vae_encoder::StreamedVideoVaeEncoder,
    recovery::PolicyHistory,
    runtime::artifact::ArtifactStaging,
    runtime::frame_manifest::{MAX_PNG_FRAME_BYTES, MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES},
};
use serde::Deserialize;
use std::{fs, io::BufReader};

const CONDITIONED_SCHEMA_VERSION: u32 = 2;
const QWEN_IMAGE_PAD_TOKEN_ID: u32 = 151_655;
const QWEN_VIDEO_PAD_TOKEN_ID: u32 = 151_656;
const MAX_PROMPT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REFERENCE_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CONDITIONED_BUNDLE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_REFERENCE_WAV_BYTES: u64 = 8 * 1024 * 1024;
const MIN_REFERENCE_VIDEO_FRAMES: usize = 2 * 24;
const MAX_REFERENCE_VIDEO_FRAMES: usize = 15 * 24;
const MIN_REFERENCE_AUDIO_SAMPLES: usize = 2 * 32_000;
const MAX_REFERENCE_AUDIO_SAMPLES: usize = 15 * 32_000;
const FRAME_MANIFEST_NAME: &str = "frames.manifest.json";

const SCHEMA: &str = "conditioned_schema_version";
const MODE: &str = "conditioned_mode";
const NUM_FRAMES: &str = "num_frames";
const CANVAS_HEIGHT: &str = "canvas_height";
const CANVAS_WIDTH: &str = "canvas_width";
const PATCH_SIZE: &str = "patch_size";
const AUDIO_CHANNELS: &str = "audio_channels";
const COMPLETED_STEPS: &str = "completed_steps";
const SIGMA_POINTS: &str = "sigma_points";
const VIDEO_SHIFT: &str = "video_shift";
const AUDIO_SHIFT: &str = "audio_shift";
const PROMPT_EMBEDDINGS: &str = "prompt_embeddings";
const TEXT_TOKEN_TAGS: &str = "text_token_tags";
const TOKEN_IDS: &str = "token_ids";
const CONDITION_VIDEO_ROWS: &str = "condition_video_rows";
const CONDITION_AUDIO_ROWS: &str = "condition_audio_rows";
const VIDEO_LATENTS: &str = "video_latents";
const AUDIO_LATENTS: &str = "audio_latents";
const FL_ANCHORS: &str = "fl_anchors";
const REF_KINDS: &str = "reference_kinds";
const REF_VIDEO_FRAMES: &str = "reference_video_frames";
const REF_VIDEO_HEIGHTS: &str = "reference_video_heights";
const REF_VIDEO_WIDTHS: &str = "reference_video_widths";
const REF_AUDIO_LATENTS: &str = "reference_audio_latents";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceManifest {
    schema_version: u32,
    references: Vec<ReferenceManifestEntry>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ReferenceManifestEntry {
    Image {
        png: PathBuf,
    },
    Video {
        frames_manifest: PathBuf,
        fps: u32,
        #[serde(default)]
        soundtrack_wav: Option<PathBuf>,
    },
    Audio {
        wav: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConditionedMode {
    Fl2va,
    Ref2va,
}

impl ConditionedMode {
    const fn code(self) -> u32 {
        match self {
            Self::Fl2va => 0,
            Self::Ref2va => 1,
        }
    }

    fn from_code(code: u32) -> Result<Self> {
        match code {
            0 => Ok(Self::Fl2va),
            1 => Ok(Self::Ref2va),
            _ => bail!("unknown conditioned bundle mode {code}"),
        }
    }

    const fn transformer_component(self) -> &'static str {
        match self {
            Self::Fl2va => "transformer",
            Self::Ref2va => "transformer_ref",
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
        }
    }
}

enum BundleLayout {
    Fl2va { anchors: Vec<KeyframeAnchor> },
    Ref2va { references: Vec<ReferenceBlock> },
}

struct ConditionedBundle {
    mode: ConditionedMode,
    num_frames: usize,
    canvas_height: usize,
    canvas_width: usize,
    patch_size: [usize; 3],
    audio_channels: usize,
    completed_steps: usize,
    sigma_points: usize,
    video_shift: f32,
    audio_shift: f32,
    history: PolicyHistory,
    prompt_embeddings: Tensor,
    text_token_tags: Tensor,
    token_ids: Tensor,
    condition_video_rows: Tensor,
    condition_audio_rows: Tensor,
    video_latents: Tensor,
    audio_latents: Tensor,
    layout_spec: BundleLayout,
    layout: ConditionedLayout,
    qwen_numerical_contract: Option<H3QwenNumericalContract>,
}

pub(super) fn run_prepare_fl2va(command: H3Command) -> Result<()> {
    let H3Command::PrepareFl2va {
        model,
        prompt,
        prompt_file,
        image,
        last_image,
        height,
        width,
        num_frames,
        target,
        seed,
        sigma_points,
        video_shift,
        audio_shift,
        output,
        device,
        weights,
        attention_query_chunk_size,
        telemetry_json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for prepare-fl2va");
    };
    validate_schedule(sigma_points, video_shift, audio_shift)?;
    anyhow::ensure!(
        image.is_some() || last_image.is_some(),
        "prepare-fl2va requires --image, --last-image, or both"
    );
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "conditioned bundle output")?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output])?;
    let prompt = resolve_prompt(prompt, prompt_file.as_deref())?;
    let first_image = image.as_deref().map(RgbImage::from_png).transpose()?;
    let last_image = last_image.as_deref().map(RgbImage::from_png).transpose()?;
    let source_canvas = first_image
        .as_ref()
        .or(last_image.as_ref())
        .map(|image| (image.width(), image.height()));
    let (canvas, num_frames) =
        target.resolve_canvas_geometry((height, width), num_frames, source_canvas)?;
    let canvas = canvas
        .map(|(height, width)| Fl2vaCanvas::new(height, width))
        .transpose()?;
    let device = parse_device(&device)?;
    validate_h3_selected_cuda_profile(&device)
        .context("FL2VA preparation exact CUDA profile preflight")?;
    let cache_policy = weights.cache_policy()?;
    let tokenizer_path = required_component_file(&model, "tokenizer/tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let transformer_dir = resolve_component(&model, Path::new("transformer"))?;
    let transformer_config = TransformerConfig::from_file(transformer_dir.join("config.json"))?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let text_encoder = StreamedMultimodalTextEncoder::open_h3_default(
        resolve_component(&model, Path::new("text_encoder"))?,
        weights.weight_source,
        cache_policy,
        device.clone(),
        attention_query_chunk_size,
    )?;
    let video_encoder = StreamedVideoVaeEncoder::open_from_model_root(
        &model,
        weights.weight_source,
        cache_policy,
        device.clone(),
    )?;
    let mut request_rng = StdRng::seed_from_u64(seed);
    let mut options = Fl2vaOptions::official(num_frames);
    if let Some(canvas) = canvas {
        options = options.with_canvas(canvas);
    }
    let prepared = prepare_fl2va(
        &text_encoder,
        &video_encoder,
        &tokenizer,
        &prompt,
        first_image.as_ref(),
        last_image.as_ref(),
        options,
        &mut request_rng,
    )?;
    let bundle = ConditionedBundle::from_fl2va(prepared, sigma_points, video_shift, audio_shift)?;
    bundle.validate_transformer_config(&transformer_config, &device)?;
    publish_bundle(&output, &bundle)?;
    println!(
        "prepared {}-frame FL2VA bundle at {}x{} in {}",
        bundle.num_frames,
        bundle.canvas_width,
        bundle.canvas_height,
        output.display()
    );
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}

struct ConditionedCheckpointObserver<'a> {
    progress: CliDenoiseObserver,
    directory: Option<PathBuf>,
    bundle: &'a ConditionedBundle,
    execution_policy: ExecutionPolicy,
    history: PolicyHistory,
}

impl DenoiseObserver for ConditionedCheckpointObserver<'_> {
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
            .context("conditioned checkpoint directory is not configured")?;
        let mut history = self.history.clone();
        history.append_successful_evaluations(
            u64::try_from(event.latents.completed_steps).context("completed steps exceed u64")?,
            self.execution_policy.clone(),
        )?;
        let destination = directory.join(format!(
            "checkpoint-step{:06}.safetensors",
            event.latents.completed_steps
        ));
        publish_bundle_state(
            &destination,
            self.bundle,
            &event.latents.video,
            &event.latents.audio,
            event.latents.completed_steps,
            &history,
        )?;
        self.history = history;
        println!(
            "conditioned checkpoint {}/{}: {}",
            event.latents.completed_steps,
            event.total_steps,
            destination.display()
        );
        Ok(())
    }

    fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()> {
        self.progress.on_step_completed(event)
    }
}

pub(super) fn run_denoise_conditioned(command: H3Command) -> Result<()> {
    let H3Command::DenoiseConditioned {
        resources,
        admission,
        model,
        inputs,
        output,
        device,
        policy,
        weights,
        chunks,
        no_precompute_adaln,
        flash_attention,
        no_progress,
        max_steps,
        checkpoint_dir,
        telemetry_json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for denoise-conditioned");
    };
    anyhow::ensure!(
        inputs.is_file(),
        "conditioned bundle does not exist: {}",
        inputs.display()
    );
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "conditioned denoise output")?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output])?;
    let checkpoint_dir = checkpoint_dir
        .map(|path| -> Result<PathBuf> {
            let path = resolve_output_outside_model(&path, &model)?;
            ensure_new_output(&path, "conditioned checkpoint directory")?;
            Ok(path)
        })
        .transpose()?;
    ensure_checkpoint_outputs_are_disjoint(
        checkpoint_dir.as_deref(),
        &output,
        telemetry_json.as_deref(),
    )?;
    let explicit_policy_settings =
        weights.is_explicit() || chunks.is_explicit() || no_precompute_adaln || flash_attention;
    let configured_weights = weights.configured();
    let configured_chunks = chunks.configured(flash_attention);
    let device = parse_device(&device)?;
    validate_h3_selected_cuda_profile(&device)
        .context("conditioned denoise exact CUDA profile preflight")?;
    let mut bundle = load_bundle(&inputs, &device)?;
    let (prompt_batch, prompt_rows, _) = bundle.prompt_embeddings.dims3()?;
    let language_rows = prompt_batch
        .checked_mul(prompt_rows)
        .context("conditioned Qwen language row count overflow")?;
    let qwen_numerical_contract = bundle
        .qwen_numerical_contract
        .as_ref()
        .context(
            "conditioned bundle has legacy/unbound Qwen provenance and cannot execute; history remains audit-readable",
        )?;
    validate_qwen_numerical_contract(
        qwen_numerical_contract,
        &device,
        language_rows,
        usize::try_from(
            qwen_numerical_contract
                .vision_linear_geometry
                .image_total_patch_rows,
        )
        .context("recorded Qwen image patch rows exceed usize")?,
        usize::try_from(
            qwen_numerical_contract
                .vision_linear_geometry
                .video_total_patch_rows,
        )
        .context("recorded Qwen video patch rows exceed usize")?,
        usize::try_from(qwen_numerical_contract.max_vision_segment_rows)
            .context("recorded Qwen vision segment rows exceed usize")?,
    )?;
    let explicit_policy = policy.is_some();
    let requested_policy = resolve_execution_policy(
        policy.as_deref(),
        &device,
        configured_weights.weight_source,
        configured_weights.cache_policy()?,
        configured_chunks,
        flash_attention,
        !no_precompute_adaln,
    )?;
    bundle
        .history
        .validate_resume_numerics(&device)
        .context("conditioned checkpoint policy history cannot resume on the selected runtime")?;
    let checkpoint_policy = bundle
        .history
        .segments
        .last()
        .map(|segment| &segment.policy);
    let mut execution_policy = select_resume_execution_policy(
        requested_policy,
        checkpoint_policy,
        explicit_policy,
        explicit_policy_settings,
        bundle.completed_steps > 0,
    )?;
    validate_executable_policy(&execution_policy, &device)?;

    let component = bundle.mode.transformer_component();
    let transformer_dir = resolve_component(&model, Path::new(component))?;
    let transformer_config = TransformerConfig::from_file(transformer_dir.join("config.json"))?;
    bundle.validate_transformer_config(&transformer_config, &device)?;
    let (context_batch, context_rows, _) = bundle.prompt_embeddings.dims3()?;
    let context_projection_rows = context_batch
        .checked_mul(context_rows)
        .context("conditioned context projection row count overflows usize")?;
    let maximum_timestep_rows = match bundle.mode {
        ConditionedMode::Fl2va => 3,
        ConditionedMode::Ref2va => 4,
    };
    validate_h3_numerical_backend(
        &device,
        execution_policy.flash_attention(),
        execution_policy.transformer_chunking()?.attention.key,
        bundle.layout.sequence_length(),
        context_projection_rows,
        maximum_timestep_rows,
    )
    .context("conditioned numerical-backend preflight refused before transformer payload access")?;
    let geometry = super::resource::geometry_from_tensors(
        &execution_policy,
        &bundle.prompt_embeddings,
        &bundle.video_latents,
        &bundle.audio_latents,
    )?;
    let mut rows = geometry.sequence_rows(transformer_config.patch_size)?;
    rows.video = u64::try_from(bundle.layout.video_rows())?;
    rows.audio = u64::try_from(bundle.layout.audio_rows())?;
    rows.total = u64::try_from(bundle.layout.sequence_length())?;
    let evaluations = usize::try_from(super::plan::selected_evaluation_count(
        bundle.sigma_points,
        bundle.completed_steps,
        max_steps.map(|n| n.get()),
    )?)?;
    let locked_origin = if checkpoint_policy.is_some() {
        Some(flyingfish::runtime::resource_selection::SelectionOrigin::Recorded)
    } else if explicit_policy {
        Some(flyingfish::runtime::resource_selection::SelectionOrigin::Pinned)
    } else {
        None
    };
    let selection_path = output.with_extension("resource-selection.json");
    super::resource::validate_sidecar_output(
        &selection_path,
        &output,
        &[telemetry_json.as_deref(), checkpoint_dir.as_deref()],
    )?;
    // What the run was conditioned on, recorded as the file's own properties.
    let input_stat = flyingfish::runtime::artifact::FileStat::of_target(&inputs)?;
    anyhow::ensure!(
        input_stat.is_file(),
        "conditioning input is not a regular file: {}",
        inputs.display()
    );
    let input_evidence = serde_json::json!({
        "file": inputs.display().to_string(),
        "bytes": input_stat.len(),
    });
    let mut selected = match super::resource::select_h3(super::resource::H3ResourceRequest {
        additional_host_allowance_bytes: 0,
        component: &transformer_dir,
        device: &device,
        baseline: &execution_policy,
        geometry,
        rows: Some(rows),
        timestep_rows: maximum_timestep_rows as u64,
        evaluations,
        limits: admission,
        resources: &resources,
        weights,
        locked_origin,
        resident_input_bytes: super::resource::represented_input_bytes(
            &device,
            &bundle.prompt_embeddings,
            &bundle.video_latents,
            &bundle.audio_latents,
        )?,
        request: serde_json::json!({
            "command":"h3.denoise-conditioned","inputs":input_evidence.clone(),
            "geometry":geometry,"packed_rows":rows,"evaluations":evaluations,
        }),
    }) {
        Ok(selected) => selected,
        Err(error) => {
            flyingfish::resource_policy::report_refusal(&error, Some(&selection_path));
            return Err(error);
        }
    };
    selected.provenance.input = Some(input_evidence);
    execution_policy = selected.policy;
    flyingfish::resource_policy::publish_selection(&selection_path, &selected.provenance)?;
    let transformer_options = build_transformer_options(device.clone(), &execution_policy)?;
    let transformer = StreamedTransformer::open(&transformer_dir, transformer_options)?;
    bundle.validate_transformer_config(transformer.config(), transformer.device())?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let tags = bundle
        .text_token_tags
        .to_vec1::<u32>()
        .context("conditioned text_token_tags must be a U32 vector")?;
    if let Some(directory) = checkpoint_dir.as_ref() {
        create_new_directory(directory, "conditioned checkpoint directory")?;
    }
    let checkpointing = checkpoint_dir.is_some();
    let mut observer = ConditionedCheckpointObserver {
        progress: CliDenoiseObserver::new(!no_progress),
        directory: checkpoint_dir,
        bundle: &bundle,
        execution_policy: execution_policy.clone(),
        history: bundle.history.clone(),
    };
    let result = denoise_conditioned_with_observer(
        &transformer,
        &bundle.prompt_embeddings,
        &tags,
        &bundle.condition_video_rows,
        &bundle.condition_audio_rows,
        &bundle.video_latents,
        &bundle.audio_latents,
        &bundle.layout,
        T2vaSchedule {
            sigma_points: bundle.sigma_points,
            video_shift: bundle.video_shift,
            audio_shift: bundle.audio_shift,
        },
        T2vaExecutionOptions {
            precompute_adaln: execution_policy.precompute_adaln,
            start_step: bundle.completed_steps,
            max_steps: max_steps.map(NonZeroUsize::get),
        },
        &mut observer,
    );
    report_h3_device_cache(&transformer);
    let result = result?;
    let history = if checkpointing {
        anyhow::ensure!(
            observer.history.completed_evaluations
                == u64::try_from(result.completed_steps).context("completed steps exceed u64")?,
            "conditioned checkpoint history did not reach the denoise result"
        );
        observer.history.clone()
    } else {
        let mut history = observer.history.clone();
        history.append_successful_evaluations(
            u64::try_from(result.completed_steps).context("completed steps exceed u64")?,
            execution_policy,
        )?;
        history
    };
    drop(observer);
    bundle.video_latents = result.video;
    bundle.audio_latents = result.audio;
    bundle.completed_steps = result.completed_steps;
    bundle.history = history;
    publish_bundle(&output, &bundle)?;
    println!(
        "saved {} conditioned checkpoint after {}/{} evaluations to {}",
        bundle.mode.name(),
        bundle.completed_steps,
        bundle.sigma_points - 1,
        output.display()
    );
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}

pub(super) fn run_prepare_ref2va(command: H3Command) -> Result<()> {
    let H3Command::PrepareRef2va {
        model,
        prompt,
        prompt_file,
        references_json,
        height,
        width,
        num_frames,
        target,
        seed,
        sigma_points,
        video_shift,
        audio_shift,
        output,
        device,
        weights,
        attention_query_chunk_size,
        telemetry_json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for prepare-ref2va");
    };
    validate_schedule(sigma_points, video_shift, audio_shift)?;
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "conditioned bundle output")?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output])?;
    let device = parse_device(&device)?;
    validate_h3_selected_cuda_profile(&device)
        .context("Ref2VA preparation exact CUDA profile preflight")?;
    let cache_policy = weights.cache_policy()?;
    let prompt = resolve_prompt(prompt, prompt_file.as_deref())?;
    let (canvas, num_frames) = target.resolve_canvas_geometry((height, width), num_frames, None)?;
    let (height, width) = canvas.context(
        "prepare-ref2va needs a target canvas: pass --height and --width, or the official --short-edge target",
    )?;
    let aligned_target_frames = align_h3_frames(num_frames)?;
    let references = load_reference_manifest(&references_json, aligned_target_frames)?;
    let tokenizer_path = required_component_file(&model, "tokenizer/tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let text_encoder = StreamedMultimodalTextEncoder::open_h3_default(
        resolve_component(&model, Path::new("text_encoder"))?,
        weights.weight_source,
        cache_policy,
        device.clone(),
        attention_query_chunk_size,
    )?;
    let video_encoder = StreamedVideoVaeEncoder::open_from_model_root(
        &model,
        weights.weight_source,
        cache_policy,
        device.clone(),
    )?;
    let audio_encoder = StreamedAudioVaeEncoder::open(
        resolve_component(&model, Path::new("audio_vae"))?,
        weights.weight_source,
        cache_policy,
        device.clone(),
    )?;
    let transformer_dir = resolve_component(&model, Path::new("transformer_ref"))?;
    let transformer = StreamedTransformer::open(
        &transformer_dir,
        StreamedTransformerOptions::new(weights.weight_source, cache_policy, device.clone()),
    )?;
    let pipeline = Ref2vaPipeline::new(
        &tokenizer,
        &text_encoder,
        &video_encoder,
        &audio_encoder,
        &transformer,
    )?;
    let prepared = pipeline.prepare(
        &prompt,
        &references,
        Ref2vaTarget {
            num_frames,
            height,
            width,
        },
        seed,
    )?;
    let bundle = ConditionedBundle::from_ref2va(
        prepared,
        transformer.config().patch_size,
        sigma_points,
        video_shift,
        audio_shift,
    )?;
    bundle.validate_transformer_config(transformer.config(), transformer.device())?;
    publish_bundle(&output, &bundle)?;
    println!(
        "prepared {}-frame Ref2VA bundle with {} references at {}x{} in {}",
        bundle.num_frames,
        match &bundle.layout_spec {
            BundleLayout::Ref2va { references } => references.len(),
            BundleLayout::Fl2va { .. } => 0,
        },
        bundle.canvas_width,
        bundle.canvas_height,
        output.display()
    );
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}

impl ConditionedBundle {
    fn from_fl2va(
        prepared: PreparedFl2va,
        sigma_points: usize,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
        let device = prepared.prompt.embeddings.device();
        let text_token_tags = Tensor::from_slice(
            &prepared.prompt.text_token_tags,
            prepared.prompt.text_token_tags.len(),
            device,
        )?;
        let token_ids = Tensor::from_slice(
            &prepared.prompt.token_ids,
            prepared.prompt.token_ids.len(),
            device,
        )?;
        let bundle = Self {
            mode: ConditionedMode::Fl2va,
            num_frames: prepared.num_frames,
            canvas_height: prepared.canvas.height,
            canvas_width: prepared.canvas.width,
            patch_size: prepared.patch_size,
            audio_channels: prepared.audio_channels,
            completed_steps: 0,
            sigma_points,
            video_shift,
            audio_shift,
            history: PolicyHistory::new(),
            prompt_embeddings: prepared.prompt.embeddings,
            text_token_tags,
            token_ids,
            condition_video_rows: prepared.condition_video_rows,
            condition_audio_rows: prepared.condition_audio_rows,
            video_latents: prepared.initial_target_video_latents,
            audio_latents: prepared.initial_target_audio_latents,
            layout_spec: BundleLayout::Fl2va {
                anchors: prepared.anchors,
            },
            layout: prepared.layout,
            qwen_numerical_contract: Some(prepared.prompt.numerical_contract),
        };
        bundle.validate_common()?;
        Ok(bundle)
    }

    fn from_ref2va(
        prepared: PreparedRef2va,
        patch_size: [usize; 3],
        sigma_points: usize,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
        let device = prepared.prompt_embeddings.device();
        let text_token_tags = Tensor::from_slice(
            &prepared.text_token_tags,
            prepared.text_token_tags.len(),
            device,
        )?;
        let token_ids = Tensor::from_slice(&prepared.token_ids, prepared.token_ids.len(), device)?;
        let audio_channels = prepared.initial_target_audio_latents.dim(0)?;
        let bundle = Self {
            mode: ConditionedMode::Ref2va,
            num_frames: prepared.target.num_frames,
            canvas_height: prepared.target.height,
            canvas_width: prepared.target.width,
            patch_size,
            audio_channels,
            completed_steps: 0,
            sigma_points,
            video_shift,
            audio_shift,
            history: PolicyHistory::new(),
            prompt_embeddings: prepared.prompt_embeddings,
            text_token_tags,
            token_ids,
            condition_video_rows: prepared.condition_video_rows,
            condition_audio_rows: prepared.condition_audio_rows,
            video_latents: prepared.initial_target_video_latents,
            audio_latents: prepared.initial_target_audio_latents,
            layout_spec: BundleLayout::Ref2va {
                references: prepared.reference_blocks,
            },
            layout: prepared.layout,
            qwen_numerical_contract: Some(prepared.qwen_numerical_contract),
        };
        bundle.validate_common()?;
        Ok(bundle)
    }

    fn validate_common(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(
                (&self.mode, &self.layout_spec),
                (ConditionedMode::Fl2va, BundleLayout::Fl2va { .. })
                    | (ConditionedMode::Ref2va, BundleLayout::Ref2va { .. })
            ),
            "conditioned bundle mode and layout metadata disagree"
        );
        if let BundleLayout::Ref2va { references } = &self.layout_spec {
            let images = references
                .iter()
                .filter(|reference| matches!(reference, ReferenceBlock::Image { .. }))
                .count();
            let videos = references
                .iter()
                .filter(|reference| matches!(reference, ReferenceBlock::Video { .. }))
                .count();
            let audios = references
                .iter()
                .filter(|reference| matches!(reference, ReferenceBlock::Audio { .. }))
                .count();
            anyhow::ensure!(
                !references.is_empty()
                    && references.len() <= 12
                    && images <= 9
                    && videos <= 3
                    && audios <= 3
                    && images + videos > 0,
                "conditioned Ref2VA bundle violates the explicit 9-image/3-video/3-audio/12-total limits"
            );
        }
        validate_schedule(self.sigma_points, self.video_shift, self.audio_shift)?;
        anyhow::ensure!(
            self.completed_steps < self.sigma_points,
            "completed steps exceed conditioned schedule"
        );
        self.history.validate()?;
        anyhow::ensure!(
            self.history.completed_evaluations
                == u64::try_from(self.completed_steps).context("completed steps exceed u64")?,
            "policy history and completed conditioned steps disagree"
        );
        anyhow::ensure!(
            self.patch_size == [1, 2, 2] && self.audio_channels == 2,
            "conditioned bundle is not released H3 patch/stereo geometry"
        );
        let (prompt_batch, prompt_rows, prompt_width) = self
            .prompt_embeddings
            .dims3()
            .context("prompt_embeddings must be [1, rows, width]")?;
        anyhow::ensure!(
            prompt_batch == 1 && prompt_rows > 0 && prompt_width > 0,
            "conditioned prompt embedding shape is invalid"
        );
        anyhow::ensure!(
            matches!(
                self.prompt_embeddings.dtype(),
                DType::F32 | DType::F16 | DType::BF16
            ),
            "prompt_embeddings must use a floating-point dtype"
        );
        anyhow::ensure!(
            self.text_token_tags.dtype() == DType::U32
                && self.text_token_tags.dims() == [prompt_rows]
                && self
                    .text_token_tags
                    .to_vec1::<u32>()?
                    .iter()
                    .all(|&tag| tag < 3),
            "text_token_tags must be a valid U32 vector matching prompt rows"
        );
        anyhow::ensure!(
            self.token_ids.dtype() == DType::U32 && self.token_ids.dims() == [prompt_rows],
            "token_ids must be a U32 vector matching prompt rows"
        );
        anyhow::ensure!(
            self.condition_video_rows.dtype() == DType::F32
                && self.condition_video_rows.rank() == 2,
            "condition_video_rows must be an F32 matrix"
        );
        anyhow::ensure!(
            self.condition_audio_rows.dtype() == DType::F32
                && self.condition_audio_rows.rank() == 2,
            "condition_audio_rows must be an F32 matrix"
        );
        let (video_batch, video_channels, video_frames, latent_height, latent_width) = self
            .video_latents
            .dims5()
            .context("video_latents must be [1, channels, frames, height, width]")?;
        anyhow::ensure!(
            self.video_latents.dtype() == DType::F32
                && video_batch == 1
                && video_channels > 0
                && video_frames > 0
                && latent_height > 0
                && latent_width > 0,
            "conditioned video_latents shape or dtype is invalid"
        );
        let (audio_channels, audio_width, audio_frames) = self
            .audio_latents
            .dims3()
            .context("audio_latents must be [channels, width, frames]")?;
        anyhow::ensure!(
            self.audio_latents.dtype() == DType::F32
                && audio_channels == self.audio_channels
                && audio_width > 0
                && audio_frames > 0,
            "conditioned audio_latents shape or dtype is invalid"
        );
        let expected_height = latent_height
            .checked_mul(16)
            .context("conditioned canvas height overflow")?;
        let expected_width = latent_width
            .checked_mul(16)
            .context("conditioned canvas width overflow")?;
        anyhow::ensure!(
            self.canvas_height == expected_height && self.canvas_width == expected_width,
            "conditioned canvas does not match the released 16x VAE latent geometry"
        );
        anyhow::ensure!(
            (flyingfish::h3::fl2va::MINIMAX_H3_MIN_ALIGNED_FRAMES
                ..=flyingfish::h3::fl2va::MINIMAX_H3_MAX_ALIGNED_FRAMES)
                .contains(&self.num_frames)
                && self.num_frames % 17 == 5,
            "conditioned num_frames must be an aligned H3 17*n+5 geometry within the released 4-15 second range"
        );
        let expected_video_frames = (self.num_frames - 5) / 17 * 5 + 2;
        anyhow::ensure!(
            video_frames == expected_video_frames,
            "conditioned video has {video_frames} latent frames, expected {expected_video_frames}"
        );
        let expected_audio_frames = round_ratio_ties_even(
            self.num_frames
                .checked_mul(40)
                .context("conditioned audio frame count overflow")?,
            24,
        )?;
        anyhow::ensure!(
            audio_frames == expected_audio_frames,
            "conditioned audio has {audio_frames} latents, expected {expected_audio_frames}"
        );
        let patch_volume = self.patch_size.iter().try_fold(1usize, |product, &part| {
            product
                .checked_mul(part)
                .context("conditioned patch volume overflow")
        })?;
        let video_row_width = video_channels
            .checked_mul(patch_volume)
            .context("conditioned video row width overflow")?;
        anyhow::ensure!(
            self.condition_video_rows.dim(1)? == video_row_width,
            "conditioned video row width does not match latent channels and patch"
        );
        anyhow::ensure!(
            self.condition_audio_rows.dim(1)? == audio_width,
            "conditioned audio row width does not match target audio width"
        );
        anyhow::ensure!(
            self.condition_video_rows.dim(0)? == self.layout.condition_video_rows()
                && self.condition_audio_rows.dim(0)? == self.layout.condition_audio_rows(),
            "condition row counts disagree with the reconstructed layout"
        );
        let target_video_rows = video_frames
            .checked_mul(latent_height / 2)
            .and_then(|rows| rows.checked_mul(latent_width / 2))
            .context("conditioned target video row count overflow")?;
        let target_audio_rows = audio_channels
            .checked_mul(audio_frames)
            .context("conditioned target audio row count overflow")?;
        anyhow::ensure!(
            self.layout.target_video_rows() == target_video_rows
                && self.layout.target_audio_rows() == target_audio_rows,
            "target row counts disagree with conditioned tensor geometry"
        );
        let device = self.prompt_embeddings.device();
        for (name, tensor) in [
            ("text_token_tags", &self.text_token_tags),
            ("token_ids", &self.token_ids),
            ("condition_video_rows", &self.condition_video_rows),
            ("condition_audio_rows", &self.condition_audio_rows),
            ("video_latents", &self.video_latents),
            ("audio_latents", &self.audio_latents),
        ] {
            anyhow::ensure!(
                tensor.device().same_device(device),
                "{name} is on a different device from prompt_embeddings"
            );
        }
        if let Some(contract) = &self.qwen_numerical_contract {
            contract.validate()?;
            let (batch, rows, _) = self.prompt_embeddings.dims3()?;
            let language_rows = batch
                .checked_mul(rows)
                .context("conditioned Qwen language row count overflow")?;
            anyhow::ensure!(
                contract.language_rows
                    == u64::try_from(language_rows)
                        .context("conditioned Qwen language rows exceed u64")?,
                "conditioned Qwen numerical contract language rows disagree with prompt embeddings"
            );
            let ids = self
                .token_ids
                .to_vec1::<u32>()
                .context("conditioned token_ids must be a U32 vector")?;
            let (image_patch_rows, video_patch_rows, max_segment_rows) =
                qwen_vision_geometry_from_token_ids(&ids)?;
            anyhow::ensure!(
                contract.vision_linear_geometry.image_total_patch_rows
                    == u64::try_from(image_patch_rows)
                        .context("derived image patch rows exceed u64")?
                    && contract.vision_linear_geometry.video_total_patch_rows
                        == u64::try_from(video_patch_rows)
                            .context("derived video patch rows exceed u64")?
                    && contract.max_vision_segment_rows
                        == u64::try_from(max_segment_rows)
                            .context("derived vision segment rows exceed u64")?,
                "conditioned Qwen contract vision geometry disagrees with canonical pad-token runs"
            );
            let observed_runs = qwen_vision_pad_runs(&ids)?;
            let mut expected_runs = Vec::new();
            for grid in &contract.ordered_vision_grids {
                let temporal = usize::try_from(grid.temporal)
                    .context("recorded Qwen grid temporal count exceeds usize")?;
                let merged_segment_rows = usize::try_from(grid.attention_segment_rows / 4)
                    .context("recorded Qwen merged segment rows exceed usize")?;
                expected_runs.extend(std::iter::repeat_n(
                    (grid.modality, merged_segment_rows),
                    temporal,
                ));
            }
            anyhow::ensure!(
                observed_runs == expected_runs,
                "conditioned Qwen ordered vision grids disagree with canonical pad-token run order/lengths"
            );
        }
        Ok(())
    }

    fn validate_transformer_config(
        &self,
        config: &TransformerConfig,
        device: &Device,
    ) -> Result<()> {
        self.validate_common()?;
        anyhow::ensure!(
            config.patch_size == self.patch_size,
            "conditioned patch {:?} differs from transformer patch {:?}",
            self.patch_size,
            config.patch_size
        );
        anyhow::ensure!(
            self.prompt_embeddings.dim(2)? == config.text_dim,
            "conditioned prompt width differs from transformer text_dim"
        );
        anyhow::ensure!(
            self.video_latents.dim(1)? == config.in_channels
                && self.audio_latents.dim(1)? == config.audio_in_channels,
            "conditioned latent channels differ from transformer inputs"
        );
        let transformer_video_width =
            config
                .patch_size
                .iter()
                .try_fold(config.in_channels, |width, &part| {
                    width
                        .checked_mul(part)
                        .context("transformer video input width overflow")
                })?;
        anyhow::ensure!(
            self.condition_video_rows.dim(1)? == transformer_video_width
                && self.condition_audio_rows.dim(1)? == config.audio_in_channels,
            "conditioned row widths differ from transformer inputs"
        );
        anyhow::ensure!(
            self.prompt_embeddings.device().same_device(device),
            "conditioned bundle and transformer are on different devices"
        );
        Ok(())
    }
}

fn publish_bundle(path: &Path, bundle: &ConditionedBundle) -> Result<()> {
    publish_bundle_state(
        path,
        bundle,
        &bundle.video_latents,
        &bundle.audio_latents,
        bundle.completed_steps,
        &bundle.history,
    )
}

fn publish_bundle_state(
    path: &Path,
    bundle: &ConditionedBundle,
    video_latents: &Tensor,
    audio_latents: &Tensor,
    completed_steps: usize,
    history: &PolicyHistory,
) -> Result<()> {
    bundle.validate_common()?;
    anyhow::ensure!(
        completed_steps >= bundle.completed_steps,
        "conditioned checkpoint cannot move completed steps backwards"
    );
    anyhow::ensure!(
        video_latents.dims() == bundle.video_latents.dims()
            && video_latents.dtype() == DType::F32
            && audio_latents.dims() == bundle.audio_latents.dims()
            && audio_latents.dtype() == DType::F32,
        "conditioned checkpoint latents differ from the prepared target geometry"
    );
    anyhow::ensure!(
        completed_steps < bundle.sigma_points,
        "conditioned checkpoint completed steps exceed its schedule"
    );
    history.validate()?;
    history.validate_extends(&bundle.history)?;
    anyhow::ensure!(
        history.completed_evaluations
            == u64::try_from(completed_steps).context("completed steps exceed u64")?,
        "conditioned checkpoint history and completed steps disagree"
    );
    let device = bundle.prompt_embeddings.device();
    anyhow::ensure!(
        video_latents.device().same_device(device) && audio_latents.device().same_device(device),
        "conditioned checkpoint latents are on a different device"
    );
    let patch_size = [
        checked_u32(bundle.patch_size[0], PATCH_SIZE)?,
        checked_u32(bundle.patch_size[1], PATCH_SIZE)?,
        checked_u32(bundle.patch_size[2], PATCH_SIZE)?,
    ];
    let mut tensors: HashMap<&'static str, Tensor> = HashMap::from([
        (SCHEMA, Tensor::new(CONDITIONED_SCHEMA_VERSION, device)?),
        (MODE, Tensor::new(bundle.mode.code(), device)?),
        (
            NUM_FRAMES,
            scalar_usize(bundle.num_frames, NUM_FRAMES, device)?,
        ),
        (
            CANVAS_HEIGHT,
            scalar_usize(bundle.canvas_height, CANVAS_HEIGHT, device)?,
        ),
        (
            CANVAS_WIDTH,
            scalar_usize(bundle.canvas_width, CANVAS_WIDTH, device)?,
        ),
        (PATCH_SIZE, Tensor::from_slice(&patch_size, 3, device)?),
        (
            AUDIO_CHANNELS,
            scalar_usize(bundle.audio_channels, AUDIO_CHANNELS, device)?,
        ),
        (
            COMPLETED_STEPS,
            scalar_usize(completed_steps, COMPLETED_STEPS, device)?,
        ),
        (
            SIGMA_POINTS,
            scalar_usize(bundle.sigma_points, SIGMA_POINTS, device)?,
        ),
        (VIDEO_SHIFT, Tensor::new(bundle.video_shift, device)?),
        (AUDIO_SHIFT, Tensor::new(bundle.audio_shift, device)?),
        (PROMPT_EMBEDDINGS, bundle.prompt_embeddings.clone()),
        (TEXT_TOKEN_TAGS, bundle.text_token_tags.clone()),
        (TOKEN_IDS, bundle.token_ids.clone()),
        (CONDITION_VIDEO_ROWS, bundle.condition_video_rows.clone()),
        (CONDITION_AUDIO_ROWS, bundle.condition_audio_rows.clone()),
        (VIDEO_LATENTS, video_latents.clone()),
        (AUDIO_LATENTS, audio_latents.clone()),
    ]);
    match &bundle.layout_spec {
        BundleLayout::Fl2va { anchors } => {
            let anchors = anchors
                .iter()
                .map(|anchor| match anchor {
                    KeyframeAnchor::First => 0u32,
                    KeyframeAnchor::Last => 1u32,
                })
                .collect::<Vec<_>>();
            tensors.insert(
                FL_ANCHORS,
                Tensor::from_slice(&anchors, anchors.len(), device)?,
            );
        }
        BundleLayout::Ref2va { references } => {
            let mut kinds = Vec::with_capacity(references.len());
            let mut video_frames = Vec::with_capacity(references.len());
            let mut video_heights = Vec::with_capacity(references.len());
            let mut video_widths = Vec::with_capacity(references.len());
            let mut audio_latents = Vec::with_capacity(references.len());
            for reference in references {
                let (kind, frames, height, width, audio) = match *reference {
                    ReferenceBlock::Image {
                        latent_frames,
                        latent_height,
                        latent_width,
                    } => (0u32, latent_frames, latent_height, latent_width, 0),
                    ReferenceBlock::Audio { audio_latents } => (1u32, 0, 0, 0, audio_latents),
                    ReferenceBlock::Video {
                        latent_frames,
                        latent_height,
                        latent_width,
                        audio_latents,
                    } => (
                        2u32,
                        latent_frames,
                        latent_height,
                        latent_width,
                        audio_latents,
                    ),
                };
                kinds.push(kind);
                video_frames.push(checked_u32(frames, REF_VIDEO_FRAMES)?);
                video_heights.push(checked_u32(height, REF_VIDEO_HEIGHTS)?);
                video_widths.push(checked_u32(width, REF_VIDEO_WIDTHS)?);
                audio_latents.push(checked_u32(audio, REF_AUDIO_LATENTS)?);
            }
            for (name, values) in [
                (REF_KINDS, kinds),
                (REF_VIDEO_FRAMES, video_frames),
                (REF_VIDEO_HEIGHTS, video_heights),
                (REF_VIDEO_WIDTHS, video_widths),
                (REF_AUDIO_LATENTS, audio_latents),
            ] {
                tensors.insert(name, Tensor::from_slice(&values, values.len(), device)?);
            }
        }
    }
    history.insert_checkpoint_tensors(&mut tensors, device)?;
    bundle
        .qwen_numerical_contract
        .as_ref()
        .context("refusing to publish a conditioned bundle without Qwen numerical provenance")?
        .insert_artifact_tensors(&mut tensors, device)?;
    let staging = ArtifactStaging::new_for_path_producer(path)
        .with_context(|| format!("failed to stage conditioned bundle {}", path.display()))?;
    safetensors::save(&tensors, staging.producer_path())
        .with_context(|| format!("failed to save conditioned bundle {}", path.display()))?;
    staging.publish()?;
    Ok(())
}

fn load_bundle(path: &Path, device: &Device) -> Result<ConditionedBundle> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect conditioned bundle {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file()
            && metadata.len() > 0
            && metadata.len() <= MAX_CONDITIONED_BUNDLE_BYTES,
        "conditioned bundle must be a regular non-symlink file in 1..={MAX_CONDITIONED_BUNDLE_BYTES} bytes"
    );
    let mut tensors = safetensors::load(path, device)
        .with_context(|| format!("failed to load conditioned bundle {}", path.display()))?;
    let qwen_numerical_contract = H3QwenNumericalContract::take_artifact_tensors(&mut tensors)?;
    let schema = take_scalar_u32(&mut tensors, SCHEMA)?;
    anyhow::ensure!(
        schema == CONDITIONED_SCHEMA_VERSION,
        "unsupported conditioned bundle schema {schema}; expected {CONDITIONED_SCHEMA_VERSION}"
    );
    anyhow::ensure!(
        qwen_numerical_contract.is_some(),
        "conditioned bundle schema {schema} requires a Qwen numerical contract"
    );
    let mode = ConditionedMode::from_code(take_scalar_u32(&mut tensors, MODE)?)?;
    let history = PolicyHistory::take_checkpoint_tensors(&mut tensors)?
        .context("conditioned bundle is missing policy-history metadata")?;
    let num_frames = take_scalar_usize(&mut tensors, NUM_FRAMES)?;
    let canvas_height = take_scalar_usize(&mut tensors, CANVAS_HEIGHT)?;
    let canvas_width = take_scalar_usize(&mut tensors, CANVAS_WIDTH)?;
    let patch_size_values = take_u32_vector(&mut tensors, PATCH_SIZE)?;
    let patch_size: [usize; 3] = patch_size_values
        .into_iter()
        .map(|value| usize::try_from(value).context("patch size exceeds usize"))
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|values: Vec<usize>| {
            anyhow::anyhow!("patch_size must have 3 values, found {}", values.len())
        })?;
    let audio_channels = take_scalar_usize(&mut tensors, AUDIO_CHANNELS)?;
    let completed_steps = take_scalar_usize(&mut tensors, COMPLETED_STEPS)?;
    let sigma_points = take_scalar_usize(&mut tensors, SIGMA_POINTS)?;
    let video_shift = take_scalar_f32(&mut tensors, VIDEO_SHIFT)?;
    let audio_shift = take_scalar_f32(&mut tensors, AUDIO_SHIFT)?;
    let prompt_embeddings = take_input(&mut tensors, PROMPT_EMBEDDINGS)?;
    let text_token_tags = take_input(&mut tensors, TEXT_TOKEN_TAGS)?;
    let token_ids = take_input(&mut tensors, TOKEN_IDS)?;
    let condition_video_rows = take_input(&mut tensors, CONDITION_VIDEO_ROWS)?;
    let condition_audio_rows = take_input(&mut tensors, CONDITION_AUDIO_ROWS)?;
    let video_latents = take_input(&mut tensors, VIDEO_LATENTS)?;
    let audio_latents = take_input(&mut tensors, AUDIO_LATENTS)?;
    let tags = text_token_tags
        .to_vec1::<u32>()
        .context("conditioned text_token_tags must be U32")?;
    let (layout_spec, layout) = match mode {
        ConditionedMode::Fl2va => {
            let anchors = take_u32_vector(&mut tensors, FL_ANCHORS)?
                .into_iter()
                .map(|value| match value {
                    0 => Ok(KeyframeAnchor::First),
                    1 => Ok(KeyframeAnchor::Last),
                    _ => bail!("unknown FL2VA anchor code {value}"),
                })
                .collect::<Result<Vec<_>>>()?;
            let (_, _, target_frames, latent_height, latent_width) = video_latents.dims5()?;
            let (_, _, target_audio_frames) = audio_latents.dims3()?;
            let layout = ConditionedLayout::fl2va(
                &tags,
                target_frames,
                latent_height,
                latent_width,
                target_audio_frames,
                patch_size,
                audio_channels,
                &anchors,
                device,
            )?;
            (BundleLayout::Fl2va { anchors }, layout)
        }
        ConditionedMode::Ref2va => {
            let kinds = take_u32_vector(&mut tensors, REF_KINDS)?;
            let frames = take_u32_vector(&mut tensors, REF_VIDEO_FRAMES)?;
            let heights = take_u32_vector(&mut tensors, REF_VIDEO_HEIGHTS)?;
            let widths = take_u32_vector(&mut tensors, REF_VIDEO_WIDTHS)?;
            let audios = take_u32_vector(&mut tensors, REF_AUDIO_LATENTS)?;
            anyhow::ensure!(
                !kinds.is_empty()
                    && frames.len() == kinds.len()
                    && heights.len() == kinds.len()
                    && widths.len() == kinds.len()
                    && audios.len() == kinds.len(),
                "Ref2VA reference metadata vectors have different or zero lengths"
            );
            let mut references = Vec::with_capacity(kinds.len());
            for index in 0..kinds.len() {
                let frames =
                    usize::try_from(frames[index]).context("reference frames exceed usize")?;
                let height =
                    usize::try_from(heights[index]).context("reference height exceeds usize")?;
                let width =
                    usize::try_from(widths[index]).context("reference width exceeds usize")?;
                let audio = usize::try_from(audios[index])
                    .context("reference audio count exceeds usize")?;
                references.push(match kinds[index] {
                    0 => {
                        anyhow::ensure!(
                            frames == 1 && height > 0 && width > 0 && audio == 0,
                            "invalid image-reference metadata at index {index}"
                        );
                        ReferenceBlock::Image {
                            latent_frames: frames,
                            latent_height: height,
                            latent_width: width,
                        }
                    }
                    1 => {
                        anyhow::ensure!(
                            frames == 0 && height == 0 && width == 0 && audio > 0,
                            "invalid audio-reference metadata at index {index}"
                        );
                        ReferenceBlock::Audio {
                            audio_latents: audio,
                        }
                    }
                    2 => {
                        anyhow::ensure!(
                            frames > 0 && height > 0 && width > 0,
                            "invalid video-reference metadata at index {index}"
                        );
                        ReferenceBlock::Video {
                            latent_frames: frames,
                            latent_height: height,
                            latent_width: width,
                            audio_latents: audio,
                        }
                    }
                    value => bail!("unknown Ref2VA reference kind {value} at index {index}"),
                });
            }
            let (_, _, target_frames, latent_height, latent_width) = video_latents.dims5()?;
            let (_, _, target_audio_frames) = audio_latents.dims3()?;
            let layout = ConditionedLayout::ref2va(
                &tags,
                &references,
                target_frames,
                latent_height,
                latent_width,
                target_audio_frames,
                patch_size,
                audio_channels,
                device,
            )?;
            (BundleLayout::Ref2va { references }, layout)
        }
    };
    anyhow::ensure!(
        tensors.is_empty(),
        "conditioned bundle contains unknown tensors: {}",
        sorted_tensor_names(&tensors).join(", ")
    );
    let bundle = ConditionedBundle {
        mode,
        num_frames,
        canvas_height,
        canvas_width,
        patch_size,
        audio_channels,
        completed_steps,
        sigma_points,
        video_shift,
        audio_shift,
        history,
        prompt_embeddings,
        text_token_tags,
        token_ids,
        condition_video_rows,
        condition_audio_rows,
        video_latents,
        audio_latents,
        layout_spec,
        layout,
        qwen_numerical_contract,
    };
    bundle.validate_common()?;
    Ok(bundle)
}

fn qwen_vision_geometry_from_token_ids(token_ids: &[u32]) -> Result<(usize, usize, usize)> {
    let image_tokens = token_ids
        .iter()
        .filter(|&&token| token == QWEN_IMAGE_PAD_TOKEN_ID)
        .count();
    let video_tokens = token_ids
        .iter()
        .filter(|&&token| token == QWEN_VIDEO_PAD_TOKEN_ID)
        .count();
    let mut maximum_run = 0usize;
    let mut current_token = None;
    let mut current_run = 0usize;
    for &token in token_ids {
        if matches!(token, QWEN_IMAGE_PAD_TOKEN_ID | QWEN_VIDEO_PAD_TOKEN_ID) {
            if current_token == Some(token) {
                current_run = current_run
                    .checked_add(1)
                    .context("Qwen visual pad run length overflow")?;
            } else {
                current_token = Some(token);
                current_run = 1;
            }
            maximum_run = maximum_run.max(current_run);
        } else {
            current_token = None;
            current_run = 0;
        }
    }
    Ok((
        image_tokens
            .checked_mul(4)
            .context("Qwen image patch row count overflow")?,
        video_tokens
            .checked_mul(4)
            .context("Qwen video patch row count overflow")?,
        maximum_run
            .checked_mul(4)
            .context("Qwen vision segment row count overflow")?,
    ))
}

fn qwen_vision_pad_runs(token_ids: &[u32]) -> Result<Vec<(H3QwenVisionGridModality, usize)>> {
    let mut runs = Vec::new();
    let mut cursor = 0usize;
    while cursor < token_ids.len() {
        let modality = match token_ids[cursor] {
            QWEN_IMAGE_PAD_TOKEN_ID => H3QwenVisionGridModality::Image,
            QWEN_VIDEO_PAD_TOKEN_ID => H3QwenVisionGridModality::Video,
            _ => {
                cursor += 1;
                continue;
            }
        };
        let token = token_ids[cursor];
        let start = cursor;
        while cursor < token_ids.len() && token_ids[cursor] == token {
            cursor += 1;
        }
        let rows = cursor
            .checked_sub(start)
            .context("Qwen visual pad run underflow")?;
        runs.push((modality, rows));
    }
    Ok(runs)
}

pub(super) fn validated_policy_history(path: &Path) -> Result<PolicyHistory> {
    let bundle = load_bundle(path, &Device::Cpu)?;
    Ok(bundle.history)
}

fn load_reference_manifest(
    path: &Path,
    maximum_video_frames: usize,
) -> Result<Vec<Ref2vaReference>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect reference manifest {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && metadata.len() <= MAX_REFERENCE_MANIFEST_BYTES,
        "reference manifest must be a regular file no larger than {MAX_REFERENCE_MANIFEST_BYTES} bytes"
    );
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read reference manifest {}", path.display()))?;
    let manifest: ReferenceManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid reference manifest {}", path.display()))?;
    anyhow::ensure!(
        manifest.schema_version == 1,
        "unsupported reference manifest schema {}; expected 1",
        manifest.schema_version
    );
    anyhow::ensure!(
        !manifest.references.is_empty(),
        "reference manifest contains no references"
    );
    anyhow::ensure!(
        manifest.references.len() <= 12,
        "reference manifest exceeds H3's 12-reference limit"
    );
    let images = manifest
        .references
        .iter()
        .filter(|entry| matches!(entry, ReferenceManifestEntry::Image { .. }))
        .count();
    let videos = manifest
        .references
        .iter()
        .filter(|entry| matches!(entry, ReferenceManifestEntry::Video { .. }))
        .count();
    let audios = manifest
        .references
        .iter()
        .filter(|entry| matches!(entry, ReferenceManifestEntry::Audio { .. }))
        .count();
    anyhow::ensure!(
        images <= 9 && videos <= 3 && audios <= 3 && images + videos > 0,
        "reference manifest violates H3 limits (9 images, 3 videos, 3 standalone audios, and at least one visual)"
    );
    let base = path
        .parent()
        .context("reference manifest path has no parent directory")?;
    let mut references = Vec::with_capacity(manifest.references.len());
    let mut total_video_frames = 0usize;
    let mut total_audio_samples = 0usize;
    for (index, entry) in manifest.references.into_iter().enumerate() {
        references.push(match entry {
            ReferenceManifestEntry::Image { png } => {
                let png = resolve_regular_reference(base, &png, "reference image")?;
                ensure_file_size(&png, MAX_PNG_FRAME_BYTES, "reference image")?;
                let image = RgbImage::from_png(&png)?;
                anyhow::ensure!(
                    image.width().min(image.height()) == 2048
                        && image.width().is_multiple_of(32)
                        && image.height().is_multiple_of(32),
                    "reference image {index} must be preprocessed to a 2048-pixel short edge and 32-aligned dimensions"
                );
                let ratio = image.width() as f64 / image.height() as f64;
                anyhow::ensure!(
                    (0.25..=4.).contains(&ratio),
                    "reference image {index} aspect ratio is outside 1:4 through 4:1"
                );
                Ref2vaReference::Image(image)
            }
            ReferenceManifestEntry::Video {
                frames_manifest,
                fps,
                soundtrack_wav,
            } => {
                anyhow::ensure!(fps == 24, "reference video {index} must declare fps=24");
                let frames_manifest =
                    resolve_regular_reference(base, &frames_manifest, "frame-set manifest")?;
                let remaining_video_frames = MAX_REFERENCE_VIDEO_FRAMES
                    .checked_sub(total_video_frames)
                    .context("total reference-video duration already exceeds 15 seconds")?;
                let (frames, source_frame_count) = load_sealed_frames(
                    &frames_manifest,
                    maximum_video_frames,
                    remaining_video_frames,
                )?;
                total_video_frames = total_video_frames
                    .checked_add(source_frame_count)
                    .context("total reference-video duration overflow")?;
                anyhow::ensure!(
                    total_video_frames <= MAX_REFERENCE_VIDEO_FRAMES,
                    "total reference-video duration exceeds 15 seconds at 24 fps"
                );
                let first = frames
                    .first()
                    .context("sealed reference video contains no frames")?;
                let (expected_height, expected_width) =
                    resolve_fl2va_canvas_size(first.width(), first.height())?;
                anyhow::ensure!(
                    first.height() == expected_height && first.width() == expected_width,
                    "reference video {index} is {}x{}, expected its released H3 canvas {}x{}",
                    first.width(),
                    first.height(),
                    expected_width,
                    expected_height
                );
                let soundtrack = soundtrack_wav
                    .map(|wav| -> Result<Tensor> {
                        let wav = resolve_regular_reference(base, &wav, "reference soundtrack")?;
                        let remaining = MAX_REFERENCE_AUDIO_SAMPLES
                            .checked_sub(total_audio_samples)
                            .context("total reference-audio duration already exceeds 15 seconds")?;
                        read_wav_32k(&wav, remaining)
                    })
                    .transpose()?;
                if let Some(soundtrack) = &soundtrack {
                    total_audio_samples = total_audio_samples
                        .checked_add(soundtrack.dim(1)?)
                        .context("total reference-audio duration overflow")?;
                    anyhow::ensure!(
                        total_audio_samples <= MAX_REFERENCE_AUDIO_SAMPLES,
                        "total reference-audio duration exceeds 15 seconds at 32 kHz"
                    );
                }
                Ref2vaReference::Video {
                    frames,
                    soundtrack,
                }
            }
            ReferenceManifestEntry::Audio { wav } => {
                let wav = resolve_regular_reference(base, &wav, "audio reference")?;
                let remaining = MAX_REFERENCE_AUDIO_SAMPLES
                    .checked_sub(total_audio_samples)
                    .context("total reference-audio duration already exceeds 15 seconds")?;
                let audio = read_wav_32k(&wav, remaining)?;
                total_audio_samples = total_audio_samples
                    .checked_add(audio.dim(1)?)
                    .context("total reference-audio duration overflow")?;
                anyhow::ensure!(
                    total_audio_samples <= MAX_REFERENCE_AUDIO_SAMPLES,
                    "total reference-audio duration exceeds 15 seconds at 32 kHz"
                );
                Ref2vaReference::Audio(audio)
            }
        });
    }
    Ok(references)
}

fn resolve_regular_reference(base: &Path, supplied: &Path, label: &str) -> Result<PathBuf> {
    let unresolved = if supplied.is_absolute() {
        supplied.to_path_buf()
    } else {
        base.join(supplied)
    };
    let metadata = fs::symlink_metadata(&unresolved)
        .with_context(|| format!("failed to inspect {label} {}", unresolved.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "{label} is not a regular non-symlink file: {}",
        unresolved.display()
    );
    fs::canonicalize(&unresolved)
        .with_context(|| format!("failed to resolve {label} {}", unresolved.display()))
}

fn ensure_file_size(path: &Path, maximum: u64, label: &str) -> Result<()> {
    let bytes = fs::metadata(path)
        .with_context(|| format!("failed to inspect {label} size {}", path.display()))?
        .len();
    anyhow::ensure!(
        (1..=maximum).contains(&bytes),
        "{label} must contain 1..={maximum} bytes, got {bytes}: {}",
        path.display()
    );
    Ok(())
}

fn load_sealed_frames(
    manifest_path: &Path,
    target_frame_limit: usize,
    total_frame_limit: usize,
) -> Result<(Vec<RgbImage>, usize)> {
    anyhow::ensure!(
        target_frame_limit > 0,
        "target frame limit must be positive"
    );
    anyhow::ensure!(
        manifest_path.file_name().and_then(|name| name.to_str()) == Some(FRAME_MANIFEST_NAME),
        "reference video completion must be named {FRAME_MANIFEST_NAME}"
    );
    ensure_file_size(
        manifest_path,
        MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES as u64,
        "frame-set manifest",
    )?;
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("failed to read frame manifest {}", manifest_path.display()))?;
    let manifest = PngFrameSetManifest::from_canonical_json(&bytes)?;
    let source_frame_count = usize::try_from(manifest.frame_count)
        .context("reference-video frame count exceeds usize")?;
    anyhow::ensure!(
        (MIN_REFERENCE_VIDEO_FRAMES..=MAX_REFERENCE_VIDEO_FRAMES).contains(&source_frame_count),
        "reference video must contain 2 through 15 seconds at 24 fps"
    );
    anyhow::ensure!(
        source_frame_count <= total_frame_limit,
        "total reference-video duration exceeds 15 seconds at 24 fps"
    );
    let directory = manifest_path
        .parent()
        .context("frame manifest has no parent directory")?;
    manifest.verify_completed_directory(directory, FRAME_MANIFEST_NAME)?;
    let retained_frame_count = source_frame_count.min(target_frame_limit);
    let frames = manifest
        .frames
        .iter()
        .take(retained_frame_count)
        .map(|member| RgbImage::from_png(directory.join(&member.file_name)))
        .collect::<Result<Vec<_>>>()?;
    Ok((frames, source_frame_count))
}

fn align_h3_frames(requested: usize) -> Result<usize> {
    let minimum =
        flyingfish::h3::fl2va::MINIMAX_H3_MIN_SECONDS * flyingfish::h3::fl2va::MINIMAX_H3_FPS;
    let maximum =
        flyingfish::h3::fl2va::MINIMAX_H3_MAX_SECONDS * flyingfish::h3::fl2va::MINIMAX_H3_FPS;
    anyhow::ensure!(
        (minimum..=maximum).contains(&requested),
        "H3 supports {minimum} through {maximum} frames at 24 fps; requested {requested}"
    );
    let remainder = requested % 17;
    requested
        .checked_add((5 + 17 - remainder) % 17)
        .context("H3 frame alignment overflow")
}

fn read_wav_32k(path: &Path, total_sample_limit: usize) -> Result<Tensor> {
    ensure_file_size(path, MAX_REFERENCE_WAV_BYTES, "reference WAV")?;
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open reference WAV {}", path.display()))?;
    let mut reader = hound::WavReader::new(BufReader::new(file))
        .with_context(|| format!("invalid reference WAV {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 32_000 && matches!(spec.channels, 1 | 2),
        "reference WAV must be 32 kHz mono or stereo, got {} Hz / {} channels",
        spec.sample_rate,
        spec.channels
    );
    let sample_frames =
        usize::try_from(reader.duration()).context("reference WAV duration exceeds usize")?;
    anyhow::ensure!(
        (MIN_REFERENCE_AUDIO_SAMPLES..=MAX_REFERENCE_AUDIO_SAMPLES).contains(&sample_frames),
        "reference WAV must contain 2 through 15 seconds at 32 kHz"
    );
    anyhow::ensure!(
        sample_frames <= total_sample_limit,
        "total reference-audio duration exceeds 15 seconds at 32 kHz"
    );
    let interleaved = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| value as f32 / 32768.)
                    .context("invalid PCM16 reference WAV sample")
            })
            .collect::<Result<Vec<_>>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|sample| sample.context("invalid float32 reference WAV sample"))
            .collect::<Result<Vec<_>>>()?,
        _ => bail!(
            "reference WAV must use PCM16 or float32 samples, got {:?}/{} bits",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };
    let channels = usize::from(spec.channels);
    anyhow::ensure!(
        !interleaved.is_empty()
            && interleaved.len().is_multiple_of(channels)
            && interleaved.iter().all(|value| value.is_finite()),
        "reference WAV samples are empty, truncated, or non-finite"
    );
    let frames = interleaved.len() / channels;
    let mut channel_major = vec![0f32; interleaved.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            channel_major[channel * frames + frame] = interleaved[frame * channels + channel];
        }
    }
    Tensor::from_vec(channel_major, (channels, frames), &Device::Cpu).map_err(Into::into)
}

fn validate_schedule(sigma_points: usize, video_shift: f32, audio_shift: f32) -> Result<()> {
    anyhow::ensure!(sigma_points >= 2, "sigma_points must be at least two");
    anyhow::ensure!(
        video_shift.is_finite() && video_shift > 0.,
        "video_shift must be finite and positive"
    );
    anyhow::ensure!(
        audio_shift.is_finite() && audio_shift > 0.,
        "audio_shift must be finite and positive"
    );
    checked_u32(sigma_points, SIGMA_POINTS)?;
    Ok(())
}

fn resolve_prompt(prompt: Option<String>, prompt_file: Option<&Path>) -> Result<String> {
    match (prompt, prompt_file) {
        (Some(prompt), None) => {
            anyhow::ensure!(!prompt.is_empty(), "prompt must not be empty");
            anyhow::ensure!(
                prompt.chars().count() <= 7_000,
                "MiniMax-H3 prompt exceeds the 7000-character limit"
            );
            Ok(prompt)
        }
        (None, Some(path)) => {
            let metadata = fs::symlink_metadata(path)
                .with_context(|| format!("failed to inspect prompt file {}", path.display()))?;
            anyhow::ensure!(
                metadata.file_type().is_file() && metadata.len() <= MAX_PROMPT_BYTES,
                "prompt file must be a regular file no larger than {MAX_PROMPT_BYTES} bytes"
            );
            let prompt = fs::read_to_string(path)
                .with_context(|| format!("failed to read UTF-8 prompt file {}", path.display()))?;
            anyhow::ensure!(!prompt.is_empty(), "prompt file must not be empty");
            anyhow::ensure!(
                prompt.chars().count() <= 7_000,
                "MiniMax-H3 prompt exceeds the 7000-character limit"
            );
            Ok(prompt)
        }
        _ => bail!("pass exactly one of --prompt or --prompt-file"),
    }
}

fn required_component_file(model: &Path, relative: &str) -> Result<PathBuf> {
    let model = fs::canonicalize(model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let path = model.join(relative);
    let resolved = fs::canonicalize(&path)
        .with_context(|| format!("required model file does not exist: {}", path.display()))?;
    anyhow::ensure!(
        resolved.starts_with(&model) && resolved.is_file(),
        "required model file escapes the model or is not regular: {}",
        resolved.display()
    );
    Ok(resolved)
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} exceeds u32"))
}

fn scalar_usize(value: usize, name: &str, device: &Device) -> Result<Tensor> {
    Ok(Tensor::new(checked_u32(value, name)?, device)?)
}

fn take_scalar_u32(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<u32> {
    let tensor = take_input(tensors, name)?;
    anyhow::ensure!(
        tensor.dtype() == DType::U32 && tensor.rank() == 0,
        "{name} must be a U32 scalar"
    );
    tensor
        .to_scalar::<u32>()
        .with_context(|| format!("failed to read {name} scalar"))
}

fn take_scalar_usize(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<usize> {
    usize::try_from(take_scalar_u32(tensors, name)?)
        .with_context(|| format!("{name} exceeds usize"))
}

fn take_scalar_f32(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<f32> {
    let tensor = take_input(tensors, name)?;
    anyhow::ensure!(
        tensor.dtype() == DType::F32 && tensor.rank() == 0,
        "{name} must be an F32 scalar"
    );
    let value = tensor
        .to_scalar::<f32>()
        .with_context(|| format!("failed to read {name} scalar"))?;
    anyhow::ensure!(value.is_finite(), "{name} must be finite");
    Ok(value)
}

fn take_u32_vector(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Vec<u32>> {
    let tensor = take_input(tensors, name)?;
    anyhow::ensure!(
        tensor.dtype() == DType::U32 && tensor.rank() == 1,
        "{name} must be a U32 vector"
    );
    tensor
        .to_vec1::<u32>()
        .with_context(|| format!("failed to read {name} vector"))
}

fn round_ratio_ties_even(numerator: usize, denominator: usize) -> Result<usize> {
    anyhow::ensure!(denominator > 0, "rounding denominator must be non-zero");
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled = remainder
        .checked_mul(2)
        .context("rounding remainder overflow")?;
    if doubled < denominator || (doubled == denominator && quotient.is_multiple_of(2)) {
        Ok(quotient)
    } else {
        quotient.checked_add(1).context("rounded ratio overflow")
    }
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
        ("conditioned denoise output", Some(output)),
        ("telemetry output", telemetry),
    ] {
        if let Some(path) = path {
            anyhow::ensure!(
                path != checkpoint_directory
                    && !path.starts_with(checkpoint_directory)
                    && !checkpoint_directory.starts_with(path),
                "conditioned checkpoint directory conflicts with {label}: {}",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn audit_only_qwen_contract(language_rows: usize) -> H3QwenNumericalContract {
        H3QwenNumericalContract::for_verified_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cpu,
            NonZeroUsize::new(language_rows).unwrap(),
            NonZeroUsize::new(language_rows).unwrap(),
            0,
            H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
        )
        .unwrap()
    }

    fn fake_fl_bundle() -> ConditionedBundle {
        let device = Device::Cpu;
        let tags = vec![1u32, 0];
        let anchors = vec![KeyframeAnchor::First];
        let layout =
            ConditionedLayout::fl2va(&tags, 37, 2, 2, 207, [1, 2, 2], 2, &anchors, &device)
                .unwrap();
        ConditionedBundle {
            mode: ConditionedMode::Fl2va,
            num_frames: 124,
            canvas_height: 32,
            canvas_width: 32,
            patch_size: [1, 2, 2],
            audio_channels: 2,
            completed_steps: 0,
            sigma_points: 3,
            video_shift: 12.,
            audio_shift: 3.,
            history: PolicyHistory::new(),
            prompt_embeddings: Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap(),
            text_token_tags: Tensor::from_slice(&tags, 2, &device).unwrap(),
            token_ids: Tensor::from_slice(&[7u32, 8], 2, &device).unwrap(),
            condition_video_rows: Tensor::zeros((1, 4), DType::F32, &device).unwrap(),
            condition_audio_rows: Tensor::zeros((0, 1), DType::F32, &device).unwrap(),
            video_latents: Tensor::zeros((1, 1, 37, 2, 2), DType::F32, &device).unwrap(),
            audio_latents: Tensor::zeros((2, 1, 207), DType::F32, &device).unwrap(),
            layout_spec: BundleLayout::Fl2va { anchors },
            layout,
            qwen_numerical_contract: Some(audit_only_qwen_contract(2)),
        }
    }

    fn fake_ref_bundle() -> ConditionedBundle {
        let mut bundle = fake_fl_bundle();
        let references = vec![ReferenceBlock::Image {
            latent_frames: 1,
            latent_height: 2,
            latent_width: 2,
        }];
        bundle.mode = ConditionedMode::Ref2va;
        bundle.layout = ConditionedLayout::ref2va(
            &[1u32, 0],
            &references,
            37,
            2,
            2,
            207,
            [1, 2, 2],
            2,
            &Device::Cpu,
        )
        .unwrap();
        bundle.layout_spec = BundleLayout::Ref2va { references };
        bundle
    }

    fn fake_vision_bundle(
        grids: &[(H3QwenVisionGridModality, usize, usize, usize)],
        token_ids: &[u32],
    ) -> ConditionedBundle {
        let device = Device::Cpu;
        let tags = vec![1u32; token_ids.len()];
        let anchors = vec![KeyframeAnchor::First];
        let layout =
            ConditionedLayout::fl2va(&tags, 37, 2, 2, 207, [1, 2, 2], 2, &anchors, &device)
                .unwrap();
        let (mut image_rows, mut video_rows, mut max_segment) = (0usize, 0usize, 0usize);
        for &(modality, temporal, height, width) in grids {
            let segment = height.checked_mul(width).unwrap();
            let patch_rows = temporal.checked_mul(segment).unwrap();
            match modality {
                H3QwenVisionGridModality::Image => image_rows += patch_rows,
                H3QwenVisionGridModality::Video => video_rows += patch_rows,
            }
            max_segment = max_segment.max(segment);
        }
        let geometry = H3QwenVisionLinearGeometry::from_patch_rows(image_rows, video_rows).unwrap();
        let rows = NonZeroUsize::new(token_ids.len()).unwrap();
        let contract = H3QwenNumericalContract::for_verified_target_with_grids(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cpu,
            rows,
            rows,
            max_segment,
            geometry,
            grids,
        )
        .unwrap();
        ConditionedBundle {
            mode: ConditionedMode::Fl2va,
            num_frames: 124,
            canvas_height: 32,
            canvas_width: 32,
            patch_size: [1, 2, 2],
            audio_channels: 2,
            completed_steps: 0,
            sigma_points: 3,
            video_shift: 12.,
            audio_shift: 3.,
            history: PolicyHistory::new(),
            prompt_embeddings: Tensor::zeros((1, token_ids.len(), 4), DType::F32, &device).unwrap(),
            text_token_tags: Tensor::from_slice(&tags, tags.len(), &device).unwrap(),
            token_ids: Tensor::from_slice(token_ids, token_ids.len(), &device).unwrap(),
            condition_video_rows: Tensor::zeros((1, 4), DType::F32, &device).unwrap(),
            condition_audio_rows: Tensor::zeros((0, 1), DType::F32, &device).unwrap(),
            video_latents: Tensor::zeros((1, 1, 37, 2, 2), DType::F32, &device).unwrap(),
            audio_latents: Tensor::zeros((2, 1, 207), DType::F32, &device).unwrap(),
            layout_spec: BundleLayout::Fl2va { anchors },
            layout,
            qwen_numerical_contract: Some(contract),
        }
    }

    #[test]
    fn conditioned_bundle_cross_checks_ordered_vision_pad_modalities_and_run_lengths() {
        let image = H3QwenVisionGridModality::Image;
        let video = H3QwenVisionGridModality::Video;
        let mixed_grids = [(image, 1, 2, 2), (video, 1, 2, 2)];
        let valid = fake_vision_bundle(
            &mixed_grids,
            &[QWEN_IMAGE_PAD_TOKEN_ID, 42, QWEN_VIDEO_PAD_TOKEN_ID],
        );
        valid.validate_common().unwrap();
        let swapped = fake_vision_bundle(
            &mixed_grids,
            &[QWEN_VIDEO_PAD_TOKEN_ID, 42, QWEN_IMAGE_PAD_TOKEN_ID],
        );
        let error = swapped.validate_common().unwrap_err().to_string();
        assert!(error.contains("ordered vision grids"), "{error}");

        let unequal_runs = [(image, 1, 2, 4), (image, 1, 2, 2)];
        let valid = fake_vision_bundle(
            &unequal_runs,
            &[
                QWEN_IMAGE_PAD_TOKEN_ID,
                QWEN_IMAGE_PAD_TOKEN_ID,
                42,
                QWEN_IMAGE_PAD_TOKEN_ID,
            ],
        );
        valid.validate_common().unwrap();
        let swapped_lengths = fake_vision_bundle(
            &unequal_runs,
            &[
                QWEN_IMAGE_PAD_TOKEN_ID,
                42,
                QWEN_IMAGE_PAD_TOKEN_ID,
                QWEN_IMAGE_PAD_TOKEN_ID,
            ],
        );
        let error = swapped_lengths.validate_common().unwrap_err().to_string();
        assert!(error.contains("ordered vision grids"), "{error}");
    }

    #[test]
    fn conditioned_bundle_roundtrips_and_rejects_unknown_tensors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prepared.safetensors");
        let bundle = fake_fl_bundle();
        publish_bundle(&path, &bundle).unwrap();
        let loaded = load_bundle(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.mode, ConditionedMode::Fl2va);
        assert_eq!(loaded.num_frames, 124);
        assert_eq!(loaded.completed_steps, 0);
        assert_eq!(loaded.video_latents.dims(), &[1, 1, 37, 2, 2]);
        assert_eq!(loaded.audio_latents.dims(), &[2, 1, 207]);

        let mut tensors = safetensors::load(&path, &Device::Cpu).unwrap();
        tensors.insert(
            "unknown_tensor".to_owned(),
            Tensor::new(1u32, &Device::Cpu).unwrap(),
        );
        let unknown = directory.path().join("unknown.safetensors");
        safetensors::save(&tensors, &unknown).unwrap();
        let error = load_bundle(&unknown, &Device::Cpu)
            .err()
            .expect("unknown tensor must be rejected")
            .to_string();
        assert!(
            error.contains("unknown tensors"),
            "unexpected error: {error}"
        );

        let mut tensors = safetensors::load(&path, &Device::Cpu).unwrap();
        tensors.remove(TOKEN_IDS);
        let missing = directory.path().join("missing.safetensors");
        safetensors::save(&tensors, &missing).unwrap();
        let error = load_bundle(&missing, &Device::Cpu)
            .err()
            .expect("missing tensor must be rejected")
            .to_string();
        assert!(error.contains(TOKEN_IDS), "unexpected error: {error}");
    }

    #[test]
    fn history_cli_accepts_strict_fl_and_ref_bundles_and_rejects_arbitrary_safetensors() {
        let directory = tempfile::tempdir().unwrap();
        let mut ref_checkpoint = None;
        for (name, mut bundle) in [("fl2va", fake_fl_bundle()), ("ref2va", fake_ref_bundle())] {
            let mut expected = PolicyHistory::new();
            expected
                .append_successful_evaluations(1, default_execution_policy(&Device::Cpu))
                .unwrap();
            bundle.completed_steps = 1;
            bundle.history = expected.clone();
            let checkpoint = directory.path().join(format!("{name}.safetensors"));
            let output = directory.path().join(format!("{name}-history.json"));
            publish_bundle(&checkpoint, &bundle).unwrap();
            super::super::denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
                checkpoint: checkpoint.clone(),
                output: Some(output.clone()),
            })
            .unwrap();
            let actual = PolicyHistory::from_json(&fs::read(output).unwrap()).unwrap();
            assert_eq!(actual, expected);
            if name == "ref2va" {
                ref_checkpoint = Some(checkpoint);
            }
        }

        let arbitrary = directory.path().join("arbitrary.safetensors");
        safetensors::save(
            &HashMap::from([("arbitrary", Tensor::new(1u32, &Device::Cpu).unwrap())]),
            &arbitrary,
        )
        .unwrap();
        let output = directory.path().join("arbitrary-history.json");
        let error = super::super::denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
            checkpoint: arbitrary,
            output: Some(output.clone()),
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("neither the explicit conditioned markers"));
        assert!(!output.exists());

        let partial = directory
            .path()
            .join("partial-conditioned-marker.safetensors");
        safetensors::save(
            &HashMap::from([(
                SCHEMA,
                Tensor::new(CONDITIONED_SCHEMA_VERSION, &Device::Cpu).unwrap(),
            )]),
            &partial,
        )
        .unwrap();
        let error = super::super::denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
            checkpoint: partial,
            output: None,
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("incomplete conditioned format marker"));

        let mut tensors = safetensors::load(ref_checkpoint.unwrap(), &Device::Cpu).unwrap();
        let unsupported_schema = CONDITIONED_SCHEMA_VERSION + 1;
        tensors.insert(
            SCHEMA.to_owned(),
            Tensor::new(unsupported_schema, &Device::Cpu).unwrap(),
        );
        let corrupt = directory
            .path()
            .join("conditioned-unsupported-schema.safetensors");
        safetensors::save(&tensors, &corrupt).unwrap();
        let error = super::super::denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
            checkpoint: corrupt,
            output: None,
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains(&format!(
            "unsupported conditioned bundle schema {unsupported_schema}"
        )));
    }

    #[test]
    fn checkpoint_publish_extends_history_and_advances_absolute_step() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("step-000001.safetensors");
        let bundle = fake_fl_bundle();
        let mut history = bundle.history.clone();
        history
            .append_successful_evaluations(1, default_execution_policy(&Device::Cpu))
            .unwrap();
        publish_bundle_state(
            &path,
            &bundle,
            &bundle.video_latents,
            &bundle.audio_latents,
            1,
            &history,
        )
        .unwrap();
        let loaded = load_bundle(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded.completed_steps, 1);
        assert_eq!(loaded.history, history);

        let error = publish_bundle_state(
            &directory.path().join("backwards.safetensors"),
            &loaded,
            &loaded.video_latents,
            &loaded.audio_latents,
            0,
            &PolicyHistory::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("backwards"), "unexpected error: {error}");
    }

    #[test]
    fn prompt_file_and_inline_prompt_share_the_public_limit() {
        assert_eq!(
            resolve_prompt(Some("hello".to_owned()), None).unwrap(),
            "hello"
        );
        let too_long = "界".repeat(7_001);
        assert!(resolve_prompt(Some(too_long), None).is_err());

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prompt.txt");
        fs::write(&path, "long form prompt").unwrap();
        assert_eq!(
            resolve_prompt(None, Some(&path)).unwrap(),
            "long form prompt"
        );
    }

    #[test]
    fn reference_wav_loader_requires_32k_and_deinterleaves() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("reference.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 32_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in [0i16, 16_384, -16_384, 8_192]
            .into_iter()
            .chain(std::iter::repeat_n(0, 2 * 64_000 - 4))
        {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let waveform = read_wav_32k(&path, MAX_REFERENCE_AUDIO_SAMPLES).unwrap();
        assert_eq!(waveform.dims(), &[2, 64_000]);
        let values = waveform.to_vec2::<f32>().unwrap();
        assert_eq!(&values[0][..2], &[0., -0.5]);
        assert_eq!(&values[1][..2], &[0.5, 0.25]);

        let short = directory.path().join("short.wav");
        let mut writer = hound::WavWriter::create(&short, spec).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.finalize().unwrap();
        let error = read_wav_32k(&short, MAX_REFERENCE_AUDIO_SAMPLES)
            .unwrap_err()
            .to_string();
        assert!(error.contains("2 through 15 seconds"));
    }

    #[test]
    fn reference_manifest_schema_is_ordered_tagged_and_closed() {
        let manifest: ReferenceManifest = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "references": [
                    {"kind":"image","png":"portrait.png"},
                    {"kind":"video","frames_manifest":"frames/frames.manifest.json","fps":24,"soundtrack_wav":"sound.wav"},
                    {"kind":"audio","wav":"voice.wav"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(manifest.references.len(), 3);
        assert!(matches!(
            &manifest.references[0],
            ReferenceManifestEntry::Image { png } if png == Path::new("portrait.png")
        ));
        assert!(matches!(
            &manifest.references[1],
            ReferenceManifestEntry::Video { fps: 24, .. }
        ));
        assert!(matches!(
            &manifest.references[2],
            ReferenceManifestEntry::Audio { .. }
        ));
        assert!(
            serde_json::from_str::<ReferenceManifest>(
                r#"{"schema_version":1,"references":[],"unknown":true}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<ReferenceManifest>(
                r#"{"schema_version":1,"references":[{"kind":"image","png":"x.png","fps":24}]}"#
            )
            .is_err()
        );
        assert_eq!(align_h3_frames(240).unwrap(), 243);

        let directory = tempfile::tempdir().unwrap();
        let too_many = directory.path().join("too-many.json");
        let entries = (0..13)
            .map(|index| format!(r#"{{"kind":"audio","wav":"missing-{index}.wav"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        fs::write(
            &too_many,
            format!(r#"{{"schema_version":1,"references":[{entries}]}}"#),
        )
        .unwrap();
        let error = load_reference_manifest(&too_many, 243)
            .err()
            .expect("excess references must fail before media paths are opened")
            .to_string();
        assert!(error.contains("12-reference limit"));
        assert!(!error.contains("missing-0.wav"));

        let wrong_fps = directory.path().join("wrong-fps.json");
        fs::write(
            &wrong_fps,
            r#"{"schema_version":1,"references":[{"kind":"video","frames_manifest":"missing/frames.manifest.json","fps":30}]}"#,
        )
        .unwrap();
        let error = load_reference_manifest(&wrong_fps, 243)
            .err()
            .expect("wrong fps must fail before the frame path is opened")
            .to_string();
        assert!(error.contains("must declare fps=24"));
        assert!(!error.contains("frames.manifest.json"));
    }

    #[test]
    fn conditioned_cli_surface_parses_all_staged_commands() {
        for arguments in [
            vec![
                "ff",
                "h3",
                "prepare-fl2va",
                "--model",
                "model",
                "--prompt-file",
                "prompt.txt",
                "--last-image",
                "last.png",
                "--output",
                "prepared.safetensors",
            ],
            vec![
                "ff",
                "h3",
                "prepare-ref2va",
                "--model",
                "model",
                "--prompt",
                "prompt",
                "--references-json",
                "references.json",
                "--height",
                "512",
                "--width",
                "896",
                "--output",
                "prepared.safetensors",
            ],
            vec![
                "ff",
                "h3",
                "denoise-conditioned",
                "--model",
                "model",
                "--inputs",
                "prepared.safetensors",
                "--output",
                "next.safetensors",
                "--checkpoint-dir",
                "checkpoints",
            ],
        ] {
            Args::try_parse_from(arguments).unwrap();
        }
    }
}
