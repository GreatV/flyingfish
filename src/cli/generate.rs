use super::H3Command;
use super::checkpoint::resolve_component;
use super::denoise::{
    CliDenoiseObserver, EvaluationCheckpointObserver, publish_t2va_checkpoint,
    report_h3_device_cache, resolve_execution_policy, select_resume_execution_policy,
    validate_executable_policy,
};
use super::device_parse::parse_device;
use super::output_hygiene::{
    create_new_directory, ensure_new_output, mib_to_bytes, publish_png_frame_manifest,
    publish_staged_bytes, resolve_output_outside_model, write_telemetry,
};
use super::prompt::{sorted_tensor_names, take_input};
use super::qwen_numerical::{build_qwen_numerical_contract, validate_qwen_numerical_contract};
use super::{
    DEFAULT_FLASH_BACKEND_WORKSPACE_MIB, DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB, H3AdmissionArgs,
    H3ResourceArgs, OptionalWeightCacheArgs, build_transformer_options,
    ensure_optional_output_is_distinct, resolve_optional_new_output,
};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, safetensors};
use flyingfish::h3::audio_vae::AudioVaeConfig;
use flyingfish::h3::audio_vae::{StreamedAudioVae, WavSampleFormat, write_wav};
use flyingfish::h3::conditioning_provenance::H3ConditioningProvenance;
use flyingfish::h3::config::TransformerConfig;
use flyingfish::h3::model::{
    StreamedTransformer, TransformerChunking, validate_h3_numerical_backend,
};
use flyingfish::h3::pipeline::{
    T2vaExecutionOptions, T2vaLatents, T2vaSchedule, denoise_t2va_with_options_and_observer,
};
use flyingfish::h3::policy::ExecutionBackendPolicy;
use flyingfish::h3::policy::{ExecutionPolicy, H3QwenNumericalContract};
use flyingfish::h3::resources::{
    H3ResourceBudgetExt, ResourceAssumptions, ResourceBudget, ResourceDomain, ResourceEstimate,
    T2vaGeometry, format_bytes,
};
use flyingfish::h3::text_encoder::StreamedTextEncoder;
use flyingfish::h3::video_vae::StreamedVideoVae;
use flyingfish::h3::video_vae::VideoVaeConfig;
use flyingfish::recovery::{CheckpointIdentity, PolicyHistory, take_t2va_checkpoint_metadata};
use flyingfish::runtime::artifact::{
    ArtifactStaging, FileStat, read_artifact_snapshot, sync_parent_directory,
};
use flyingfish::runtime::frame_manifest::{
    MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES, PngFrameSetManifest,
};
use flyingfish::runtime::probe::ResourceSnapshot;
use flyingfish::runtime::storage::{RECOMMENDED_READ_AHEAD_BYTES, read_ahead_window};
use flyingfish::runtime::telemetry::TelemetryMonitor;
use flyingfish::runtime::weights::{CachePolicy, ModelWeights, WeightSource};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::{fs, thread};
use tokenizers::Tokenizer;

const GENERATION_REQUEST_SCHEMA_VERSION: u32 = 1;
const GENERATION_INITIALIZATION_SCHEMA_VERSION: u32 = 2;
const GENERATION_REQUEST_FILE: &str = "generation-request.json";
const GENERATION_INITIALIZATION_FILE: &str = "generation-initialization.json";
const GENERATION_READY_FILE: &str = "generation-ready";
const EXECUTION_POLICY_FILE: &str = "execution-policy.json";
const RESOURCE_SELECTION_FILE: &str = "resource-selection.json";

#[allow(clippy::too_many_arguments)]
pub(crate) fn make_t2va_noise(
    config: &TransformerConfig,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_frames: usize,
    audio_channels: usize,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let video_count = config
        .in_channels
        .checked_mul(latent_frames)
        .and_then(|value| value.checked_mul(latent_height))
        .and_then(|value| value.checked_mul(latent_width))
        .context("video latent element count overflow")?;
    let video_values = (0..video_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    let video_latents = Tensor::from_vec(
        video_values,
        (
            1,
            config.in_channels,
            latent_frames,
            latent_height,
            latent_width,
        ),
        device,
    )?;
    let audio_count = audio_channels
        .checked_mul(config.audio_in_channels)
        .and_then(|value| value.checked_mul(audio_frames))
        .context("audio latent element count overflow")?;
    let audio_values = (0..audio_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    let audio_latents = Tensor::from_vec(
        audio_values,
        (audio_channels, config.audio_in_channels, audio_frames),
        device,
    )?;
    Ok((video_latents, audio_latents))
}

/// A preflight rejected for capacity; only this becomes a refusal. It carries
/// the observation that rejected the run, not just its text: a record naming
/// the selection's older snapshot cannot reproduce the budget that refused.
#[derive(Debug)]
struct PreflightCapacityRefusal {
    message: String,
    snapshot: ResourceSnapshot,
    budget: ResourceBudget,
    host_peak: u64,
    device_peak: u64,
}

impl std::fmt::Display for PreflightCapacityRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PreflightCapacityRefusal {}
const RESOURCE_REFUSAL_FILE: &str = "resource-refusal.json";

/// Where a refusal ledger is published: always its own file, never the
/// selection artifact. `ArtifactStaging` refuses to replace an existing
/// destination, so sharing the path would lose a resumed run's refusal behind
/// the admitted record of the original attempt — and, in the other direction,
/// would leave a fresh run'"'"'s refusal occupying the path a successful retry then
/// needs. A separate name removes both collisions and lets a refusal-only
/// directory be recognised as retryable state.
fn refusal_artifact_path(output_dir: &Path) -> PathBuf {
    output_dir.join(RESOURCE_REFUSAL_FILE)
}
const CHECKPOINT_DIRECTORY: &str = "checkpoints";
const STAGING_DIRECTORY: &str = "staging";
const FINAL_LATENTS_FILE: &str = "denoised-latents.safetensors";
const WAV_FILE: &str = "generated.wav";
const FRAMES_DIRECTORY: &str = "frames";
const FRAME_MANIFEST_FILE: &str = "frames.manifest.json";
const GENERATION_COMPLETE_FILE: &str = "generation-complete.json";
const MAX_GENERATION_REQUEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GENERATION_INITIALIZATION_BYTES: u64 = 32 * 1024 * 1024;
const MAX_EXECUTION_POLICY_BYTES: u64 = 1024 * 1024;
const MAX_GENERATION_COMPLETION_BYTES: u64 = 1024 * 1024;
const GENERATION_READY_BYTES: &[u8] = b"flyingfish-h3-generation-schema-2\n";
static FRAME_STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

const GENERATION_COMPLETION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DecodeBranchAction {
    Decoded,
    VerifiedExisting,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct DecodeBranchMeasurement {
    action: DecodeBranchAction,
    started_after_ns: u64,
    ended_after_ns: u64,
    elapsed_ns: u64,
}

impl DecodeBranchMeasurement {
    fn end_after_ns(&self) -> Result<u64> {
        let expected = self
            .started_after_ns
            .checked_add(self.elapsed_ns)
            .context("decode branch end time overflow")?;
        anyhow::ensure!(
            self.ended_after_ns == expected,
            "decode branch end offset disagrees with its start and elapsed time"
        );
        Ok(self.ended_after_ns)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationCompletion {
    schema_version: u32,
    execution: DecodeExecution,
    timing_scope: DecodeTimingScope,
    decode_wall_elapsed_ns: u64,
    branch_overlap_ns: u64,
    audio: DecodeBranchMeasurement,
    video: DecodeBranchMeasurement,
    wav_bytes: u64,
    frame_manifest_bytes: u64,
    frame_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DecodeExecution {
    ConcurrentInProcess,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DecodeTimingScope {
    HostMonotonicThroughStagedOutput,
}

impl GenerationCompletion {
    fn new(
        audio: DecodeBranchMeasurement,
        video: DecodeBranchMeasurement,
        decode_wall_elapsed: Duration,
        wav_bytes: u64,
        frame_manifest_bytes: u64,
        frame_count: usize,
    ) -> Result<Self> {
        let completion = Self {
            schema_version: GENERATION_COMPLETION_SCHEMA_VERSION,
            execution: DecodeExecution::ConcurrentInProcess,
            timing_scope: DecodeTimingScope::HostMonotonicThroughStagedOutput,
            decode_wall_elapsed_ns: duration_ns(decode_wall_elapsed)?,
            branch_overlap_ns: branch_overlap_ns(&audio, &video)?,
            audio,
            video,
            wav_bytes,
            frame_manifest_bytes,
            frame_count: u64::try_from(frame_count).context("decoded frame count exceeds u64")?,
        };
        completion.validate()?;
        Ok(completion)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == GENERATION_COMPLETION_SCHEMA_VERSION,
            "unsupported generation-completion schema {}; this build supports schema {GENERATION_COMPLETION_SCHEMA_VERSION}",
            self.schema_version
        );
        anyhow::ensure!(
            self.execution == DecodeExecution::ConcurrentInProcess,
            "generation completion does not record in-process concurrent decoding"
        );
        anyhow::ensure!(
            self.timing_scope == DecodeTimingScope::HostMonotonicThroughStagedOutput,
            "generation completion has an unsupported decode timing scope"
        );
        let audio_end = self.audio.end_after_ns()?;
        let video_end = self.video.end_after_ns()?;
        anyhow::ensure!(
            audio_end <= self.decode_wall_elapsed_ns && video_end <= self.decode_wall_elapsed_ns,
            "decode branch timing exceeds the recorded decode wall time"
        );
        anyhow::ensure!(
            self.branch_overlap_ns == branch_overlap_ns(&self.audio, &self.video)?,
            "decode branch overlap disagrees with the recorded branch intervals"
        );
        anyhow::ensure!(
            self.wav_bytes > 0 && self.frame_manifest_bytes > 0,
            "completed decode records an empty WAV or frame manifest"
        );
        anyhow::ensure!(self.frame_count > 0, "completed decode has no video frames");
        Ok(())
    }

    fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(self).context("failed to serialize generation completion")?;
        anyhow::ensure!(
            u64::try_from(bytes.len()).context("generation completion size exceeds u64")?
                <= MAX_GENERATION_COMPLETION_BYTES,
            "generation completion exceeds {MAX_GENERATION_COMPLETION_BYTES} bytes"
        );
        Ok(bytes)
    }

    fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            !bytes.is_empty()
                && u64::try_from(bytes.len()).context("generation completion size exceeds u64")?
                    <= MAX_GENERATION_COMPLETION_BYTES,
            "generation completion must contain 1..={MAX_GENERATION_COMPLETION_BYTES} bytes"
        );
        let completion: Self =
            serde_json::from_slice(bytes).context("invalid generation-completion JSON")?;
        completion.validate()?;
        anyhow::ensure!(
            bytes == completion.canonical_json()?,
            "generation-completion JSON is not in canonical schema-1 encoding"
        );
        Ok(completion)
    }
}

#[derive(Debug)]
struct TimedDecodeBranch<T> {
    value: T,
    measurement: DecodeBranchMeasurement,
}

#[derive(Debug)]
struct ConcurrentDecode<A, V> {
    audio: TimedDecodeBranch<A>,
    video: TimedDecodeBranch<V>,
    wall_elapsed: Duration,
}

#[derive(Debug)]
enum PreparedAudioDecode {
    Existing,
    Staged(Box<ArtifactStaging>),
}

impl PreparedAudioDecode {
    /// Publish if this run produced the WAV, and report its size either way.
    fn publish(self, wav_path: &Path) -> Result<u64> {
        if let Self::Staged(staging) = self {
            (*staging).publish()?;
        }
        let stat = FileStat::of_target(wav_path)
            .with_context(|| format!("failed to inspect generated WAV {}", wav_path.display()))?;
        anyhow::ensure!(
            stat.is_file(),
            "generated WAV is not a regular file: {}",
            wav_path.display()
        );
        Ok(stat.len())
    }
}

#[derive(Debug)]
enum PreparedVideoDecode {
    Existing { frames: usize },
    Staged { directory: PathBuf, frames: usize },
}

impl PreparedVideoDecode {
    fn frame_count(&self) -> usize {
        match self {
            Self::Existing { frames } | Self::Staged { frames, .. } => *frames,
        }
    }

    fn publish(self, frames_dir: &Path, output_dir: &Path) -> Result<u64> {
        match self {
            Self::Existing { .. } => frame_manifest_bytes(frames_dir),
            Self::Staged { directory, .. } => {
                let digest = frame_manifest_bytes(&directory)?;
                anyhow::ensure!(
                    symlink_metadata_if_exists(frames_dir, "frame output")?.is_none(),
                    "frame output appeared during decode: {}",
                    frames_dir.display()
                );
                fs::rename(&directory, frames_dir).with_context(|| {
                    format!(
                        "failed to atomically publish completed frame directory {}",
                        frames_dir.display()
                    )
                })?;
                sync_parent_directory(output_dir).with_context(|| {
                    format!(
                        "failed to synchronize published frame directory {}",
                        output_dir.display()
                    )
                })?;
                Ok(digest)
            }
        }
    }
}

fn duration_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).context("decode timing exceeds u64 nanoseconds")
}

fn branch_overlap_ns(
    audio: &DecodeBranchMeasurement,
    video: &DecodeBranchMeasurement,
) -> Result<u64> {
    let start = audio.started_after_ns.max(video.started_after_ns);
    let end = audio.end_after_ns()?.min(video.end_after_ns()?);
    Ok(end.saturating_sub(start))
}

fn panic_description(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn timed_decode_branch<T>(
    wall_started: Instant,
    branch: impl FnOnce() -> Result<(T, DecodeBranchAction)>,
) -> Result<TimedDecodeBranch<T>> {
    let started_after_ns = duration_ns(wall_started.elapsed())?;
    let (value, action) = branch()?;
    let ended_after_ns = duration_ns(wall_started.elapsed())?;
    Ok(TimedDecodeBranch {
        value,
        measurement: DecodeBranchMeasurement {
            action,
            started_after_ns,
            ended_after_ns,
            elapsed_ns: ended_after_ns
                .checked_sub(started_after_ns)
                .context("decode monotonic clock moved backwards")?,
        },
    })
}

fn execute_decode_branches<A, V, AF, VF>(audio: AF, video: VF) -> Result<ConcurrentDecode<A, V>>
where
    A: Send,
    V: Send,
    AF: FnOnce() -> Result<(A, DecodeBranchAction)> + Send,
    VF: FnOnce() -> Result<(V, DecodeBranchAction)> + Send,
{
    let wall_started = Instant::now();
    let (audio_result, video_result) = thread::scope(|scope| -> Result<_> {
        let audio_handle = thread::Builder::new()
            .name("ff-decode-audio".to_owned())
            .spawn_scoped(scope, || timed_decode_branch(wall_started, audio))
            .context("failed to start in-process audio decode branch")?;
        let video_handle = thread::Builder::new()
            .name("ff-decode-video".to_owned())
            .spawn_scoped(scope, || timed_decode_branch(wall_started, video))
            .context("failed to start in-process video decode branch")?;

        let audio_joined = audio_handle.join();
        let video_joined = video_handle.join();
        let audio_result = audio_joined.map_err(|payload| {
            anyhow::anyhow!(
                "in-process audio decode branch panicked: {}",
                panic_description(payload)
            )
        });
        let video_result = video_joined.map_err(|payload| {
            anyhow::anyhow!(
                "in-process video decode branch panicked: {}",
                panic_description(payload)
            )
        });
        match (audio_result, video_result) {
            (Ok(audio), Ok(video)) => Ok((audio, video)),
            (Err(audio), Err(video)) => {
                bail!("both in-process decode threads panicked; audio: {audio:#}; video: {video:#}")
            }
            (Err(error), Ok(_)) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    })?;

    let (audio, video) = match (audio_result, video_result) {
        (Ok(audio), Ok(video)) => (audio, video),
        (Err(audio), Err(video)) => {
            bail!("both in-process decode branches failed; audio: {audio:#}; video: {video:#}")
        }
        (Err(error), Ok(_)) => return Err(error).context("in-process audio decode branch failed"),
        (Ok(_), Err(error)) => return Err(error).context("in-process video decode branch failed"),
    };
    Ok(ConcurrentDecode {
        audio,
        video,
        wall_elapsed: wall_started.elapsed(),
    })
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RecordedWavFormat {
    Pcm16,
    Float32,
}

impl From<WavSampleFormat> for RecordedWavFormat {
    fn from(value: WavSampleFormat) -> Self {
        match value {
            WavSampleFormat::Pcm16 => Self::Pcm16,
            WavSampleFormat::Float32 => Self::Float32,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationRequest {
    schema_version: u32,
    model_root: String,
    prompt: String,
    token_ids: Vec<u32>,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_frames: usize,
    audio_channels: usize,
    wav_format: RecordedWavFormat,
    seed: u64,
    target_hidden_state: usize,
    sigma_points: usize,
    video_shift_f32_bits: u32,
    audio_shift_f32_bits: u32,
    /// The policy this request was sealed against, recorded in full so a
    /// resume can say which field disagrees.
    execution_policy: ExecutionPolicy,
}

/// Caller-facing values a sealed generation request is built from.
///
/// Several of these are same-typed counts and shifts. Naming them at the call
/// site keeps a transposed pair from silently sealing the wrong geometry or
/// schedule into the request.
struct GenerationRequestSpec {
    model_root: String,
    prompt: String,
    token_ids: Vec<u32>,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_frames: usize,
    audio_channels: usize,
    wav_format: WavSampleFormat,
    seed: u64,
    target_hidden_state: usize,
    sigma_points: usize,
    video_shift: f32,
    audio_shift: f32,
}

impl GenerationRequest {
    fn new(spec: GenerationRequestSpec, execution_policy: &ExecutionPolicy) -> Result<Self> {
        let GenerationRequestSpec {
            model_root,
            prompt,
            token_ids,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
            wav_format,
            seed,
            target_hidden_state,
            sigma_points,
            video_shift,
            audio_shift,
        } = spec;
        let request = Self {
            schema_version: GENERATION_REQUEST_SCHEMA_VERSION,
            model_root,
            prompt,
            token_ids,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
            wav_format: wav_format.into(),
            seed,
            target_hidden_state,
            sigma_points,
            video_shift_f32_bits: video_shift.to_bits(),
            audio_shift_f32_bits: audio_shift.to_bits(),
            execution_policy: execution_policy.clone(),
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == GENERATION_REQUEST_SCHEMA_VERSION,
            "unsupported generation-request schema {}; this build supports schema {GENERATION_REQUEST_SCHEMA_VERSION}",
            self.schema_version
        );
        anyhow::ensure!(
            !self.model_root.is_empty() && !self.prompt.is_empty() && !self.token_ids.is_empty(),
            "generation request model root, prompt, and token IDs must be non-empty"
        );
        for (name, value) in [
            ("latent_frames", self.latent_frames),
            ("latent_height", self.latent_height),
            ("latent_width", self.latent_width),
            ("audio_frames", self.audio_frames),
            ("audio_channels", self.audio_channels),
            ("target_hidden_state", self.target_hidden_state),
        ] {
            anyhow::ensure!(value > 0, "generation request {name} must be non-zero");
        }
        anyhow::ensure!(
            self.sigma_points >= 2,
            "generation request sigma_points must be at least two"
        );
        for (name, bits) in [
            ("video_shift", self.video_shift_f32_bits),
            ("audio_shift", self.audio_shift_f32_bits),
        ] {
            let value = f32::from_bits(bits);
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "generation request {name} must be finite and positive"
            );
        }
        self.execution_policy.validate()?;
        Ok(())
    }

    fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("failed to serialize generation request")?;
        anyhow::ensure!(
            u64::try_from(bytes.len()).context("generation request size exceeds u64")?
                <= MAX_GENERATION_REQUEST_BYTES,
            "generation request exceeds {MAX_GENERATION_REQUEST_BYTES} bytes"
        );
        Ok(bytes)
    }

    fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            !bytes.is_empty()
                && u64::try_from(bytes.len()).context("generation request size exceeds u64")?
                    <= MAX_GENERATION_REQUEST_BYTES,
            "generation request must contain 1..={MAX_GENERATION_REQUEST_BYTES} bytes"
        );
        let request: Self =
            serde_json::from_slice(bytes).context("invalid generation-request JSON")?;
        request.validate()?;
        anyhow::ensure!(
            bytes == request.canonical_json()?,
            "generation-request JSON is not in canonical schema-1 encoding"
        );
        Ok(request)
    }

    fn first_difference(&self, requested: &Self) -> Option<&'static str> {
        if self.model_root != requested.model_root {
            Some("model_root")
        } else if self.prompt != requested.prompt {
            Some("prompt")
        } else if self.token_ids != requested.token_ids {
            Some("token_ids")
        } else if self.latent_frames != requested.latent_frames {
            Some("latent_frames")
        } else if self.latent_height != requested.latent_height {
            Some("latent_height")
        } else if self.latent_width != requested.latent_width {
            Some("latent_width")
        } else if self.audio_frames != requested.audio_frames {
            Some("audio_frames")
        } else if self.audio_channels != requested.audio_channels {
            Some("audio_channels")
        } else if self.wav_format != requested.wav_format {
            Some("wav_format")
        } else if self.seed != requested.seed {
            Some("seed")
        } else if self.target_hidden_state != requested.target_hidden_state {
            Some("target_hidden_state")
        } else if self.sigma_points != requested.sigma_points {
            Some("sigma_points")
        } else if self.video_shift_f32_bits != requested.video_shift_f32_bits {
            Some("video_shift")
        } else if self.audio_shift_f32_bits != requested.audio_shift_f32_bits {
            Some("audio_shift")
        } else if let Some(field) = self
            .execution_policy
            .first_difference(&requested.execution_policy)
        {
            Some(field)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationInitialization {
    schema_version: u32,
    request: GenerationRequest,
    execution_policy: ExecutionPolicy,
    qwen_numerical_contract: H3QwenNumericalContract,
}

impl GenerationInitialization {
    fn new(
        request: GenerationRequest,
        execution_policy: ExecutionPolicy,
        qwen_numerical_contract: H3QwenNumericalContract,
    ) -> Result<Self> {
        let initialization = Self {
            schema_version: GENERATION_INITIALIZATION_SCHEMA_VERSION,
            request,
            execution_policy,
            qwen_numerical_contract,
        };
        initialization.validate()?;
        Ok(initialization)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == GENERATION_INITIALIZATION_SCHEMA_VERSION,
            "unsupported generation-initialization schema {}; this build supports schema {GENERATION_INITIALIZATION_SCHEMA_VERSION}",
            self.schema_version
        );
        self.request.validate()?;
        self.execution_policy.validate()?;
        self.qwen_numerical_contract.validate()?;
        if let Some(field) = self
            .request
            .execution_policy
            .first_difference(&self.execution_policy)
        {
            anyhow::bail!(
                "generation initialization request and execution policy disagree at {field}"
            );
        }
        anyhow::ensure!(
            self.qwen_numerical_contract.execution_backend
                == self.execution_policy.execution_backend
                && self.qwen_numerical_contract.configured_query_rows
                    == self.execution_policy.attention.configured_query_rows
                && self.qwen_numerical_contract.language_rows
                    == u64::try_from(self.request.token_ids.len())
                        .context("generation request token count exceeds u64")?
                && self.qwen_numerical_contract.max_vision_segment_rows == 0
                && self
                    .qwen_numerical_contract
                    .vision_linear_geometry
                    .image_total_patch_rows
                    == 0
                && self
                    .qwen_numerical_contract
                    .vision_linear_geometry
                    .video_total_patch_rows
                    == 0,
            "generation initialization Qwen contract disagrees with its text-only request/policy"
        );
        Ok(())
    }

    fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(self).context("failed to serialize generation initialization")?;
        anyhow::ensure!(
            u64::try_from(bytes.len()).context("generation initialization size exceeds u64")?
                <= MAX_GENERATION_INITIALIZATION_BYTES,
            "generation initialization exceeds {MAX_GENERATION_INITIALIZATION_BYTES} bytes"
        );
        Ok(bytes)
    }

    fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            !bytes.is_empty()
                && u64::try_from(bytes.len())
                    .context("generation initialization size exceeds u64")?
                    <= MAX_GENERATION_INITIALIZATION_BYTES,
            "generation initialization must contain 1..={MAX_GENERATION_INITIALIZATION_BYTES} bytes"
        );
        let envelope: serde_json::Value =
            serde_json::from_slice(bytes).context("invalid generation-initialization JSON")?;
        let schema = envelope
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .context("generation initialization has no integer schema_version")?;
        anyhow::ensure!(
            schema != 1,
            "legacy generation-initialization schema 1 has no Qwen numerical provenance and is audit-only; resume is refused"
        );
        anyhow::ensure!(
            schema == u64::from(GENERATION_INITIALIZATION_SCHEMA_VERSION),
            "unsupported generation-initialization schema {schema}; this build supports {GENERATION_INITIALIZATION_SCHEMA_VERSION}"
        );
        let initialization: Self =
            serde_json::from_slice(bytes).context("invalid generation-initialization JSON")?;
        initialization.validate()?;
        anyhow::ensure!(
            bytes == initialization.canonical_json()?,
            "generation-initialization JSON is not in canonical schema-2 encoding"
        );
        Ok(initialization)
    }
}

#[derive(Clone, Debug)]
struct ResumeCheckpoint {
    path: PathBuf,
    identity: CheckpointIdentity,
}

#[derive(Debug)]
struct GenerationPreflight {
    estimate: ResourceEstimate,
    budget: ResourceBudget,
    snapshot: ResourceSnapshot,
}

#[derive(Clone, Copy, Debug)]
enum PolicyOrigin {
    Recorded,
    PinnedFile,
    ExplicitFlags,
    Defaults,
    Promoted,
}

impl PolicyOrigin {
    const fn description(self) -> &'static str {
        match self {
            Self::Recorded => "recorded resume policy",
            Self::PinnedFile => "explicit pinned policy",
            Self::ExplicitFlags => "explicit resource-policy flags",
            Self::Defaults => "documented conservative defaults",
            Self::Promoted => "measured resource selection",
        }
    }
}

struct GenerationState {
    prompt_embeddings: Tensor,
    text_token_tags_tensor: Tensor,
    text_token_tags: Vec<u32>,
    video_latents: Tensor,
    audio_latents: Tensor,
    completed_steps: usize,
    policy_history: PolicyHistory,
    qwen_numerical_contract: H3QwenNumericalContract,
}

/// State the readahead window the weight streaming will run under, and how to
/// widen it when it is the binding constraint.
///
/// Every stage's weights reach the device through it, so it decides cold-start
/// throughput more than any flag this command takes: on the pinned host a
/// 128 KiB window held mmap fault-in near 594 MB/s where 2 MiB reached about
/// 2 GB/s on the same NVMe. Nothing in the process can raise it, so the run
/// reports it instead of silently absorbing it.
fn report_weight_streaming_window(model_root: &Path) {
    let Some(window) = read_ahead_window(model_root) else {
        return;
    };
    if !window.throttles_weight_streaming() {
        println!(
            "preflight storage: {} readahead window",
            format_bytes(window.bytes)
        );
        return;
    }
    println!(
        "preflight storage: {} readahead window throttles weight streaming; \
         raise it with `echo {} | sudo tee {}`",
        format_bytes(window.bytes),
        RECOMMENDED_READ_AHEAD_BYTES / 1024,
        window.control_file.display()
    );
}

/// The request `select_generation_resources` plans against.
///
/// These are the generation request's own terms, carried as one value so the
/// planner's signature stays a signature rather than a list.
struct GenerationResourceRequest<'a> {
    /// Where a refusal publishes its candidate ledger.
    refusal_sidecar: &'a Path,
    model: &'a Path,
    model_root_record: &'a str,
    device: &'a Device,
    resources: &'a H3ResourceArgs,
    optional_weight_args: OptionalWeightCacheArgs,
    checkpoint_policy: Option<&'a ExecutionPolicy>,
    explicit_policy: bool,
    token_ids: &'a [u32],
    prompt: &'a str,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_frames: usize,
    audio_channels: usize,
    sigma_points: usize,
    seed: u64,
    target_hidden_state: usize,
    video_shift: f32,
    audio_shift: f32,
    wav_format: WavSampleFormat,
    max_host_mib: Option<u64>,
    max_device_mib: Option<u64>,
    backend_workspace_mib: Option<u64>,
}

/// Choose the execution policy the remaining denoise evaluations run under.
///
/// A run with nothing left to evaluate asks nothing of the planner and keeps
/// the policy it resumed with. Otherwise the planner may promote the baseline,
/// which is a change to both the policy and its recorded provenance, so it
/// reports both rather than leaving the caller to infer the second.
#[allow(clippy::too_many_arguments)]
fn select_generation_resources(
    remaining_evaluations: usize,
    execution_policy: &mut ExecutionPolicy,
    policy_origin: &mut PolicyOrigin,
    request: GenerationResourceRequest<'_>,
) -> Result<Option<flyingfish::resource_policy::h3::H3Selection>> {
    if remaining_evaluations == 0 {
        return Ok(None);
    }
    let GenerationResourceRequest {
        refusal_sidecar,
        model,
        model_root_record,
        device,
        resources,
        optional_weight_args,
        checkpoint_policy,
        explicit_policy,
        token_ids,
        prompt,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        sigma_points,
        seed,
        target_hidden_state,
        video_shift,
        audio_shift,
        wav_format,
        max_host_mib,
        max_device_mib,
        backend_workspace_mib,
    } = request;
    let transformer_dir = resolve_component(model, Path::new("transformer"))?;
    super::denoise::promote_attention_backend_for_rows(
        execution_policy,
        device.is_cuda(),
        // Row count only: the chunk fields play no part in it.
        T2vaGeometry {
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
            ..T2vaGeometry::h3_default(token_ids.len())
        }
        .sequence_rows(
            TransformerConfig::from_file(transformer_dir.join("config.json"))?.patch_size,
        )?
        .total,
        explicit_policy,
    )?;
    let chunks = execution_policy.transformer_chunking()?;
    let geometry = T2vaGeometry {
        text_rows: token_ids.len(),
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
    };
    let locked_origin = if checkpoint_policy.is_some() {
        Some(flyingfish::runtime::resource_selection::SelectionOrigin::Recorded)
    } else if explicit_policy {
        Some(flyingfish::runtime::resource_selection::SelectionOrigin::Pinned)
    } else {
        None
    };
    let selected = match super::resource::select_h3(super::resource::H3ResourceRequest {
        additional_host_allowance_bytes: 0,
        component: &transformer_dir,
        device,
        baseline: execution_policy,
        geometry,
        rows: None,
        timestep_rows: 2,
        evaluations: remaining_evaluations,
        limits: H3AdmissionArgs {
            max_host_mib,
            max_device_mib,
            backend_workspace_mib,
        },
        resources,
        weights: optional_weight_args,
        locked_origin,
        resident_input_bytes: 0,
        request: serde_json::json!({"command":"h3.generate", "model_root":model_root_record, "prompt":prompt,
                    "token_ids":token_ids, "geometry":geometry, "sigma_points":sigma_points, "seed":seed, "target_hidden_state":target_hidden_state,
                    "video_shift":video_shift, "audio_shift":audio_shift, "wav_format":format!("{wav_format:?}")}),
    }) {
        Ok(selected) => selected,
        Err(error) => {
            // An early bail has no ledger, and leaves no directory behind.
            let destination = error
                .downcast_ref::<flyingfish::resource_policy::AdmissionRefused>()
                .map(|_| refusal_sidecar);
            flyingfish::resource_policy::report_refusal(&error, destination);
            return Err(error);
        }
    };
    if selected.policy != *execution_policy {
        *policy_origin = PolicyOrigin::Promoted;
    }
    *execution_policy = selected.policy.clone();
    Ok(Some(selected))
}

/// What an existing output directory says about a run already in progress.
///
/// Resuming has to agree with the directory on three separate questions before
/// anything is generated: whether initialization was ever sealed, which policy
/// it was sealed under, and how far denoising actually got. Answering them in
/// one place is what keeps a half-initialized directory from being read as a
/// resumable one.
struct RecordedRun {
    initialization: Option<GenerationInitialization>,
    initialization_complete: bool,
    policy: Option<ExecutionPolicy>,
    checkpoint: Option<ResumeCheckpoint>,
    final_checkpoint: Option<ResumeCheckpoint>,
}

impl RecordedRun {
    #[allow(clippy::too_many_arguments)]
    fn discover(
        existing_run: bool,
        output_dir: &Path,
        initialization_path: &Path,
        ready_path: &Path,
        policy_path: &Path,
        request_path: &Path,
        staging_dir: &Path,
        checkpoint_dir: &Path,
        latent_path: &Path,
        sigma_points: usize,
    ) -> Result<Self> {
        let recorded_initialization = if existing_run && initialization_path.exists() {
            Some(load_generation_initialization(initialization_path)?)
        } else {
            None
        };
        let initialization_complete = if existing_run {
            validate_generation_directory_entries(output_dir)?;
            if ready_path.exists() {
                validate_ready_marker(ready_path)?;
                validate_staging_directory(staging_dir)?;
                let initialization = recorded_initialization
                    .as_ref()
                    .context("existing generation has no initialization record")?;
                anyhow::ensure!(
                    load_canonical_execution_policy(policy_path)?
                        == initialization.execution_policy,
                    "recorded execution-policy artifact disagrees with generation initialization"
                );
                anyhow::ensure!(
                    load_generation_request(request_path)? == initialization.request,
                    "recorded generation-request artifact disagrees with generation initialization"
                );
                true
            } else {
                validate_incomplete_initialization(output_dir, checkpoint_dir)?;
                if recorded_initialization.is_none() {
                    validate_empty_uninitialized_directory(output_dir)?;
                }
                false
            }
        } else {
            false
        };
        let recorded_policy = recorded_initialization
            .as_ref()
            .map(|initialization| initialization.execution_policy.clone());
        let checkpoint = if initialization_complete {
            discover_latest_checkpoint(checkpoint_dir, sigma_points)?
        } else {
            None
        };
        let final_checkpoint = if existing_run && latent_path.exists() {
            Some(load_resume_checkpoint(latent_path, sigma_points)?)
        } else {
            None
        };
        validate_checkpoint_chain(
            checkpoint.as_ref(),
            final_checkpoint.as_ref(),
            recorded_policy.as_ref(),
            sigma_points,
        )?;
        Ok(Self {
            initialization: recorded_initialization,
            initialization_complete,
            policy: recorded_policy,
            checkpoint,
            final_checkpoint,
        })
    }

    /// The checkpoint a resume starts from: the published final latents when
    /// they exist, otherwise the newest step checkpoint.
    fn resume_from(&self) -> Option<&ResumeCheckpoint> {
        self.final_checkpoint.as_ref().or(self.checkpoint.as_ref())
    }
}

/// The denoise state a run starts from: a verified checkpoint's tensors, or a
/// freshly encoded prompt and freshly sampled noise.
///
/// Both paths have to produce a state the sealed initialization accepts, so
/// they are written as the two arms of one decision rather than as two places
/// that separately decide what a starting state is.
#[allow(clippy::too_many_arguments)]
fn load_or_initialize_state(
    resume_checkpoint: Option<&ResumeCheckpoint>,
    schedule_steps: usize,
    no_progress: bool,
    device: &Device,
    transformer_config: &TransformerConfig,
    generation_request: &GenerationRequest,
    model: &Path,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    prompt: &str,
    target_hidden_state: usize,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_frames: usize,
    audio_channels: usize,
    seed: u64,
    _weight_source: WeightSource,
    execution_cache_policy: CachePolicy,
    transformer_chunking: TransformerChunking,
) -> Result<GenerationState> {
    Ok(match resume_checkpoint {
        Some(checkpoint) => {
            if !no_progress {
                eprintln!(
                    "resume: verified evaluation {}/{} from {}",
                    checkpoint.identity.completed_evaluations,
                    schedule_steps,
                    checkpoint.path.display()
                );
            }
            load_generation_state(
                &checkpoint.path,
                device,
                transformer_config,
                generation_request,
            )?
        }
        None => {
            let encode_start = Instant::now();
            // Encode is one-pass: every encoder weight is read exactly once,
            // so stream mmap with drop-behind instead of retaining 63 GiB.
            let mut encoder = StreamedTextEncoder::open(
                resolve_component(model, Path::new("text_encoder"))?,
                WeightSource::Mmap,
                execution_cache_policy,
                device.clone(),
                target_hidden_state,
                transformer_chunking.attention.query_chunk_size.get(),
            )?;
            encoder.set_drop_evicted_pages(true);
            let encoded = encoder.encode_prompt(tokenizer, prompt)?;
            anyhow::ensure!(
                encoded.token_ids == token_ids,
                "tokenizer output changed after preflight"
            );
            drop(encoder);
            if !no_progress {
                eprintln!(
                    "encode: {} prompt tokens completed in {:.2}s",
                    encoded.token_ids.len(),
                    encode_start.elapsed().as_secs_f64(),
                );
            }
            let (video_latents, audio_latents) = make_t2va_noise(
                transformer_config,
                latent_frames,
                latent_height,
                latent_width,
                audio_frames,
                audio_channels,
                seed,
                device,
            )?;
            let text_token_tags_tensor = Tensor::from_vec(
                encoded.text_token_tags.clone(),
                encoded.text_token_tags.len(),
                device,
            )?;
            GenerationState {
                prompt_embeddings: encoded.embeddings,
                text_token_tags_tensor,
                text_token_tags: encoded.text_token_tags,
                video_latents,
                audio_latents,
                completed_steps: 0,
                policy_history: PolicyHistory::new(),
                qwen_numerical_contract: encoded.numerical_contract,
            }
        }
    })
}

/// What a denoise phase produces: the latents, the policy history that records
/// how it got there, and the transformer's own cache statistics when it ran.
type DenoiseOutcome = (
    T2vaLatents,
    PolicyHistory,
    Option<flyingfish::runtime::weights::CacheStats>,
);

/// Run the denoise schedule to completion, or hand back a state that already
/// reached it.
///
/// A resumed run whose checkpoint is already at the final evaluation must not
/// open the transformer at all: doing so would spend the residency budget and
/// the load time on a model with no work left for it.
#[allow(clippy::too_many_arguments)]
fn denoise_remaining_steps(
    state: GenerationState,
    schedule_steps: usize,
    device: &Device,
    execution_policy: &ExecutionPolicy,
    transformer_dir: PathBuf,
    conditioning_provenance: &H3ConditioningProvenance,
    checkpoint_dir: &Path,
    staging_dir: &Path,
    output_dir: &Path,
    sigma_points: usize,
    video_shift: f32,
    audio_shift: f32,
    no_progress: bool,
) -> Result<DenoiseOutcome> {
    Ok(if state.completed_steps < schedule_steps {
        let transformer_options = build_transformer_options(device.clone(), execution_policy)?;
        let transformer = StreamedTransformer::open(transformer_dir, transformer_options)?;
        let mut observer = EvaluationCheckpointObserver {
            progress: CliDenoiseObserver::new(!no_progress),
            directory: Some(checkpoint_dir.to_path_buf()),
            prompt_embeddings: &state.prompt_embeddings,
            text_token_tags: &state.text_token_tags_tensor,
            sigma_points,
            video_shift,
            audio_shift,
            execution_policy: execution_policy.clone(),
            policy_history: state.policy_history,
            conditioning_provenance,
            announce_checkpoints: false,
            staging_parent: Some(staging_dir.to_path_buf()),
        };
        let result = denoise_t2va_with_options_and_observer(
            &transformer,
            &state.prompt_embeddings,
            &state.text_token_tags,
            &state.video_latents,
            &state.audio_latents,
            T2vaSchedule {
                sigma_points,
                video_shift,
                audio_shift,
            },
            T2vaExecutionOptions {
                precompute_adaln: execution_policy.precompute_adaln,
                start_step: state.completed_steps,
                max_steps: None,
            },
            &mut observer,
        );
        report_h3_device_cache(&transformer);
        let result = result.with_context(|| {
                format!(
                    "generation denoise stopped; rerun the same ff video generate invocation with --output-dir {} to resume from the last complete evaluation",
                    output_dir.display()
                )
            })?;
        anyhow::ensure!(
            observer.policy_history.completed_evaluations
                == u64::try_from(result.completed_steps)
                    .context("completed evaluation count exceeds u64")?,
            "evaluation checkpoint history did not reach the generation result"
        );
        let history = observer.policy_history.clone();
        let cache = transformer.cache_stats();
        drop(observer);
        drop(transformer);
        (result, history, Some(cache))
    } else {
        (
            T2vaLatents {
                video: state.video_latents,
                audio: state.audio_latents,
                completed_steps: state.completed_steps,
            },
            state.policy_history,
            None,
        )
    })
}

/// The decode phase: turn denoised latents into a WAV and a PNG frame set, or
/// verify what an earlier run already published.
///
/// Both branches are gathered behind one call because they answer to the same
/// expectations and the same atomic completion record. The phase reads a lot
/// of the request, so it takes it as one value rather than twenty arguments.
struct DecodeStage<'a> {
    device: &'a Device,
    output_dir: &'a Path,
    staging_dir: &'a Path,
    completion_path: &'a Path,
    wav_path: &'a Path,
    frames_dir: &'a Path,
    audio_vae_dir: &'a Path,
    video_vae_dir: &'a Path,
    audio_vae_config: &'a AudioVaeConfig,
    audio_channels: usize,
    wav_format: WavSampleFormat,
    expected_audio_samples: usize,
    expected_frame_count: usize,
    expected_frame_height: usize,
    expected_frame_width: usize,
    weight_source: WeightSource,
    execution_cache_policy: CachePolicy,
    transformer_chunking: TransformerChunking,
    no_progress: bool,
}

impl DecodeStage<'_> {
    /// The number of frames the run published.
    fn run(self, result: &T2vaLatents) -> Result<usize> {
        let Self {
            device,
            output_dir,
            staging_dir,
            completion_path,
            wav_path,
            frames_dir,
            audio_vae_dir,
            video_vae_dir,
            audio_vae_config,
            audio_channels,
            wav_format,
            expected_audio_samples,
            expected_frame_count,
            expected_frame_height,
            expected_frame_width,
            weight_source,
            execution_cache_policy,
            transformer_chunking,
            no_progress,
        } = self;
        let frames = if symlink_metadata_if_exists(completion_path, "generation completion")?
            .is_some()
        {
            let completion = load_generation_completion(completion_path)?;
            validate_completed_decode(
                &completion,
                wav_path,
                frames_dir,
                audio_vae_config.sampling_rate,
                audio_channels,
                expected_audio_samples,
                wav_format,
                expected_frame_count,
                expected_frame_width,
                expected_frame_height,
            )?;
            if !no_progress {
                eprintln!(
                    "decode: verified atomic completion record {} (audio {:.2}s, video {:.2}s, overlap {:.2}s, wall {:.2}s)",
                    completion_path.display(),
                    completion.audio.elapsed_ns as f64 / 1e9,
                    completion.video.elapsed_ns as f64 / 1e9,
                    completion.branch_overlap_ns as f64 / 1e9,
                    completion.decode_wall_elapsed_ns as f64 / 1e9,
                );
            }
            usize::try_from(completion.frame_count)
                .context("completed frame count exceeds usize")?
        } else {
            let audio_latents = result.audio.clone();
            let video_latents = result.video.clone();
            let audio_device = device.clone();
            let video_device = device.clone();
            let audio = || -> Result<(PreparedAudioDecode, DecodeBranchAction)> {
                if symlink_metadata_if_exists(wav_path, "generated WAV")?.is_some() {
                    validate_existing_wav(
                        wav_path,
                        audio_vae_config.sampling_rate,
                        audio_channels,
                        expected_audio_samples,
                        wav_format,
                    )?;
                    return Ok((
                        PreparedAudioDecode::Existing,
                        DecodeBranchAction::VerifiedExisting,
                    ));
                }
                let decoder = StreamedAudioVae::open(
                    audio_vae_dir,
                    weight_source,
                    execution_cache_policy,
                    audio_device,
                )?;
                let waveform = decoder.decode(&audio_latents)?;
                let staging = ArtifactStaging::new_for_path_producer_with_staging_parent(
                    wav_path,
                    staging_dir,
                )
                .with_context(|| {
                    format!("failed to stage generated audio {}", wav_path.display())
                })?;
                write_wav(
                    staging.producer_path(),
                    &waveform,
                    audio_vae_config.sampling_rate,
                    wav_format,
                )?;
                Ok((
                    PreparedAudioDecode::Staged(Box::new(staging)),
                    DecodeBranchAction::Decoded,
                ))
            };
            let video = || -> Result<(PreparedVideoDecode, DecodeBranchAction)> {
                if symlink_metadata_if_exists(frames_dir, "frame output")?.is_some() {
                    let frames = validate_existing_frames(
                        frames_dir,
                        expected_frame_count,
                        expected_frame_width,
                        expected_frame_height,
                    )?;
                    return Ok((
                        PreparedVideoDecode::Existing { frames },
                        DecodeBranchAction::VerifiedExisting,
                    ));
                }
                let staging_directory = create_frame_staging_directory(staging_dir)?;
                let decoder = StreamedVideoVae::open(
                    video_vae_dir,
                    weight_source,
                    execution_cache_policy,
                    video_device,
                    transformer_chunking.attention.query_chunk_size.get(),
                )?;
                let frames = decoder
                    .decode_to_png_frames(&video_latents, &staging_directory)
                    .with_context(|| {
                        format!(
                            "video decode failed; incomplete frames remain isolated at {}",
                            staging_directory.display()
                        )
                    })?;
                publish_png_frame_manifest(&staging_directory, frames)?;
                Ok((
                    PreparedVideoDecode::Staged {
                        directory: staging_directory,
                        frames,
                    },
                    DecodeBranchAction::Decoded,
                ))
            };

            let decode = execute_decode_branches(audio, video)?;
            let frames = decode.video.value.frame_count();
            let audio_measurement = decode.audio.measurement.clone();
            let video_measurement = decode.video.measurement.clone();
            let decode_wall_elapsed = decode.wall_elapsed;
            let wav_digest = decode.audio.value.publish(wav_path)?;
            let frame_manifest_digest = decode.video.value.publish(frames_dir, output_dir)?;
            validate_existing_wav(
                wav_path,
                audio_vae_config.sampling_rate,
                audio_channels,
                expected_audio_samples,
                wav_format,
            )?;
            anyhow::ensure!(
                validate_existing_frames(
                    frames_dir,
                    expected_frame_count,
                    expected_frame_width,
                    expected_frame_height,
                )? == frames,
                "published frame count changed after concurrent decode"
            );
            let completion = GenerationCompletion::new(
                audio_measurement,
                video_measurement,
                decode_wall_elapsed,
                wav_digest,
                frame_manifest_digest,
                frames,
            )?;
            publish_new_bytes_with_staging_parent(
                completion_path,
                &completion.canonical_json()?,
                "generation completion",
                staging_dir,
            )?;
            if !no_progress {
                eprintln!(
                    "decode audio: {:?} in {:.2}s",
                    completion.audio.action,
                    completion.audio.elapsed_ns as f64 / 1e9
                );
                eprintln!(
                    "decode video: {:?}, {frames} frames in {:.2}s",
                    completion.video.action,
                    completion.video.elapsed_ns as f64 / 1e9
                );
                eprintln!(
                    "decode overlap: {:.2}s of {:.2}s wall; completion atomically recorded in {}",
                    completion.branch_overlap_ns as f64 / 1e9,
                    completion.decode_wall_elapsed_ns as f64 / 1e9,
                    completion_path.display()
                );
            }
            frames
        };
        Ok(frames)
    }
}

/// What the checkpoint's own VAE configurations say the decoded outputs must
/// be, resolved once so the verify and decode branches cannot disagree.
///
/// Both branches check the same shapes: one against files an earlier run left
/// behind, the other against tensors it is about to write. Deriving them twice
/// is how those two answers drift apart.
struct DecodeExpectations {
    audio_vae_dir: PathBuf,
    audio_vae_config: AudioVaeConfig,
    expected_audio_samples: usize,
    video_vae_dir: PathBuf,
    expected_frame_count: usize,
    expected_frame_height: usize,
    expected_frame_width: usize,
}

impl DecodeExpectations {
    fn resolve(
        model: &Path,
        audio_frames: usize,
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
    ) -> Result<Self> {
        let audio_vae_dir = resolve_component(model, Path::new("audio_vae"))?;
        let audio_vae_config = AudioVaeConfig::from_file(audio_vae_dir.join("config.json"))?;
        let expected_audio_samples = audio_frames
            .checked_mul(audio_vae_config.hop_length()?)
            .context("expected generated WAV sample count overflow")?;
        let video_vae_dir = resolve_component(model, Path::new("vae"))?;
        let video_vae_config = VideoVaeConfig::from_file(video_vae_dir.join("config.json"))?;
        let expected_frame_count = video_vae_config.decoded_frame_count(latent_frames)?;
        let spatial_ratio = video_vae_config.spatial_ratio()?;
        let expected_frame_height = latent_height
            .checked_mul(spatial_ratio)
            .context("expected decoded frame height overflow")?;
        let expected_frame_width = latent_width
            .checked_mul(spatial_ratio)
            .context("expected decoded frame width overflow")?;
        Ok(Self {
            audio_vae_dir,
            audio_vae_config,
            expected_audio_samples,
            video_vae_dir,
            expected_frame_count,
            expected_frame_height,
            expected_frame_width,
        })
    }
}

fn device_ordinal(device: &str) -> usize {
    device
        .strip_prefix("cuda:")
        .and_then(|ordinal| ordinal.parse().ok())
        .unwrap_or(0)
}

fn directory_tree_bytes(root: &Path) -> Result<u64> {
    if !root
        .try_exists()
        .with_context(|| format!("failed to inspect {}", root.display()))?
    {
        return Ok(0);
    }
    let mut total = 0u64;
    let mut queue = vec![root.to_path_buf()];
    while let Some(path) = queue.pop() {
        let entries = fs::read_dir(&path)
            .with_context(|| format!("failed to list {}", path.display()))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("failed to list {}", path.display()))?;
            let metadata = entry
                .metadata()
                .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
            if metadata.is_dir() {
                queue.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

fn load_or_capture_topology(primary: &Device) -> Result<flyingfish::runtime::topology::TopologyProfile> {
    use flyingfish::runtime::topology::TopologyProfile;
    let Some(path) = std::env::var_os("FF_TOPOLOGY_PROFILE") else {
        return Ok(TopologyProfile::capture(primary));
    };
    let path = std::path::PathBuf::from(path);
    match TopologyProfile::load(&path, primary)? {
        Ok(profile) => Ok(profile),
        Err(absence) => {
            eprintln!(
                "topology profile {} is not reusable ({}); recapturing",
                path.display(),
                match absence {
                    flyingfish::runtime::topology::TopologyProfileAbsence::NotFound(_) => {
                        "absent"
                    }
                    flyingfish::runtime::topology::TopologyProfileAbsence::ForeignHost { .. } => {
                        "recorded on another machine"
                    }
                    flyingfish::runtime::topology::TopologyProfileAbsence::StaleSchema { .. } => {
                        "recorded under an older schema"
                    }
                }
            );
            let profile = TopologyProfile::capture(primary);
            profile.save(&path)?;
            Ok(profile)
        }
    }
}

fn derive_t2va_configuration(
    model_root: &Path,
    geometry: T2vaGeometry,
    ordinal: usize,
    primary: &Device,
) -> Result<(ff_core::configure::DerivedConfig, Option<u64>)> {
    use flyingfish::h3::resources::{
        H3T2vaRequirement, ResourceAssumptions, ResourceEstimate, cache_charge,
    };
    let config = TransformerConfig::from_file(model_root.join("transformer/config.json"))?;
    let estimate =
        ResourceEstimate::for_t2va(&config, geometry, ResourceAssumptions::h3_bf16_mmap())?;
    let encoder_bytes = directory_tree_bytes(&model_root.join("text_encoder"))?;
    let requirement = H3T2vaRequirement::from_estimate(&estimate, encoder_bytes)?;
    let profile = load_or_capture_topology(primary)?;
    let derived = ff_core::configure::derive(ordinal, &profile, &requirement)?;
    let host_bound_bytes = match (derived.weight_source, derived.host_cache_ceiling_bytes) {
        (ff_core::configure::WeightSourceChoice::Memory, Some(ceiling)) => {
            match (|| -> Result<u64> {
                // The two steps admission itself performs: charge the cache at
                // the backfilled cap's exact integer, then read the estimate's
                // host peak with that charge. Both numbers agree bit for bit.
                let weights = flyingfish::runtime::weights::ModelWeights::open(
                    model_root.join("transformer"),
                    WeightSource::Memory,
                    CachePolicy::new(1),
                )?;
                let inventory = weights.cache_inventory()?;
                let charge = cache_charge(
                    &inventory,
                    WeightSource::Memory,
                    CachePolicy::unbounded_units().with_max_bytes((ceiling >> 20) << 20),
                )?;
                let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
                assumptions.host_weight_cache_bytes = charge.owned_weight_bytes;
                Ok(ResourceEstimate::for_t2va(&config, geometry, assumptions)?.peak_host_bytes)
            })() {
                Ok(demand) => Some(demand),
                Err(error) => {
                    eprintln!(
                        "config: host bound was not derived ({}); the admission default applies",
                        error
                    );
                    None
                }
            }
        }
        _ => None,
    };
    Ok((derived, host_bound_bytes))
}

pub(super) fn run_generate_t2va(command: H3Command) -> Result<()> {
    let H3Command::GenerateT2va {
        resources,
        model,
        prompt,
        output_dir,
        device,
        policy,
        weights: optional_weight_args,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        target,
        wav_format,
        seed,
        target_hidden_state,
        chunks: optional_chunks,
        sigma_points,
        video_shift,
        audio_shift,
        no_precompute_adaln,
        flash_attention,
        no_progress,
        explain_config,
        telemetry_json,
        max_host_mib,
        max_device_mib,
        backend_workspace_mib,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for generate-t2va");
    };

    let (latent_frames, latent_height, latent_width, audio_frames) = target
        .resolve_latent_geometry((latent_frames, latent_height, latent_width, audio_frames))?;

    let supplied_output_metadata =
        symlink_metadata_if_exists(&output_dir, "generation output directory")?;
    anyhow::ensure!(
        supplied_output_metadata
            .as_ref()
            .is_none_or(|metadata| !metadata.file_type().is_symlink()),
        "generation output directory must not be a symlink: {}",
        output_dir.display()
    );
    let model_root = fs::canonicalize(&model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let model_root_record = model_root
        .to_str()
        .context("resolved model directory is not valid UTF-8")?
        .to_owned();
    let output_dir = resolve_output_outside_model(&output_dir, &model_root)?;
    let existing_run = output_dir.exists();
    if existing_run {
        anyhow::ensure!(
            output_dir.is_dir(),
            "generation output exists but is not a directory: {}",
            output_dir.display()
        );
    } else {
        ensure_new_output(&output_dir, "output directory")?;
    }
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output_dir])?;
    if let Some(telemetry_json) = telemetry_json.as_deref() {
        anyhow::ensure!(
            !telemetry_json.starts_with(&output_dir) && !output_dir.starts_with(telemetry_json),
            "generation telemetry output must be disjoint from the resumable run directory: {}",
            telemetry_json.display()
        );
    }
    for (name, value) in [
        ("latent_frames", latent_frames),
        ("latent_height", latent_height),
        ("latent_width", latent_width),
        ("audio_frames", audio_frames),
        ("audio_channels", audio_channels),
    ] {
        anyhow::ensure!(value > 0, "{name} must be non-zero");
    }
    anyhow::ensure!(sigma_points >= 2, "sigma_points must be at least two");

    let selected_device_ordinal = device_ordinal(&device);
    let device = parse_device(&device)?;
    let tokenizer_path = required_model_file(&model, Path::new("tokenizer/tokenizer.json"))?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let tokenization = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|error| anyhow::anyhow!("failed to tokenize prompt: {error}"))?;
    let token_ids = tokenization.get_ids().to_vec();
    anyhow::ensure!(
        !token_ids.is_empty(),
        "prompt tokenized to an empty sequence"
    );

    let explicit_policy = policy.is_some();
    let explicit_policy_settings = optional_weight_args.is_explicit()
        || optional_chunks.is_explicit()
        || no_precompute_adaln
        || flash_attention;
    let mut derived_host_bound_mib: Option<u64> = None;
    let derived = if explicit_policy {
        None
    } else {
        let mut geometry = T2vaGeometry::h3_default(token_ids.len());
        geometry.latent_frames = latent_frames;
        geometry.latent_height = latent_height;
        geometry.latent_width = latent_width;
        geometry.audio_frames = audio_frames;
        geometry.audio_channels = audio_channels;
        let (derived, host_bound_bytes) = derive_t2va_configuration(
            &model_root,
            geometry,
            selected_device_ordinal,
            &device,
        )?;
        derived_host_bound_mib = host_bound_bytes.map(|bytes| bytes >> 20);
        Some(derived)
    };
    let derived_source = derived.as_ref().map(|derived| match derived.weight_source {
        ff_core::configure::WeightSourceChoice::Memory => WeightSource::Memory,
        ff_core::configure::WeightSourceChoice::Mmap => WeightSource::Mmap,
    });
    let effective_source = optional_weight_args.weight_source.or(derived_source);
    let derived_host_cache_mib = matches!(effective_source, Some(WeightSource::Memory))
        .then(|| {
            derived
                .as_ref()
                .and_then(|derived| derived.host_cache_ceiling_bytes)
                .map(|bytes| bytes >> 20)
        })
        .flatten();
    let optional_weight_args =
        optional_weight_args.with_derived(derived_source, derived_host_cache_mib);
    let optional_chunks = optional_chunks.with_derived(derived.as_ref().and_then(|d| d.chunks));
    let max_host_mib = max_host_mib.or(derived_host_bound_mib);
    if explain_config {
        match &derived {
            Some(derived) => {
                for step in &derived.provenance {
                    eprintln!("config: {step}");
                }
            }
            None => eprintln!("config: policy is operator-pinned; derivation skipped"),
        }
    }
    let weight_args = optional_weight_args.configured();
    let chunks = optional_chunks.configured(flash_attention);
    let requested_policy = resolve_execution_policy(
        policy.as_deref(),
        &device,
        weight_args.weight_source,
        weight_args.cache_policy()?,
        chunks,
        flash_attention,
        !no_precompute_adaln,
    )?;

    let policy_path = output_dir.join(EXECUTION_POLICY_FILE);
    let request_path = output_dir.join(GENERATION_REQUEST_FILE);
    let initialization_path = output_dir.join(GENERATION_INITIALIZATION_FILE);
    let ready_path = output_dir.join(GENERATION_READY_FILE);
    let checkpoint_dir = output_dir.join(CHECKPOINT_DIRECTORY);
    let staging_dir = output_dir.join(STAGING_DIRECTORY);
    let latent_path = output_dir.join(FINAL_LATENTS_FILE);
    let wav_path = output_dir.join(WAV_FILE);
    let frames_dir = output_dir.join(FRAMES_DIRECTORY);
    let completion_path = output_dir.join(GENERATION_COMPLETE_FILE);

    let recorded = RecordedRun::discover(
        existing_run,
        &output_dir,
        &initialization_path,
        &ready_path,
        &policy_path,
        &request_path,
        &staging_dir,
        &checkpoint_dir,
        &latent_path,
        sigma_points,
    )?;
    let recorded_initialization = recorded.initialization.clone();
    let initialization_complete = recorded.initialization_complete;
    let recorded_policy = recorded.policy.clone();
    let resume_checkpoint = recorded.resume_from();
    if let Some(checkpoint) = resume_checkpoint {
        checkpoint
            .identity
            .policy_history
            .validate_resume_numerics(&device)
            .context(
                "generation checkpoint policy history cannot resume on the selected runtime",
            )?;
    }
    let checkpoint_policy = resume_checkpoint
        .and_then(|checkpoint| checkpoint.identity.policy_history.segments.last())
        .map(|segment| &segment.policy)
        .or(recorded_policy.as_ref());
    let mut execution_policy = select_resume_execution_policy(
        requested_policy,
        checkpoint_policy,
        explicit_policy,
        explicit_policy_settings,
        recorded_initialization.is_some(),
    )?;
    validate_executable_policy(&execution_policy, &device)?;
    let mut policy_origin =
        if recorded_initialization.is_some() && !explicit_policy && !explicit_policy_settings {
            PolicyOrigin::Recorded
        } else if explicit_policy {
            PolicyOrigin::PinnedFile
        } else if explicit_policy_settings {
            PolicyOrigin::ExplicitFlags
        } else {
            PolicyOrigin::Defaults
        };

    let remaining_evaluations = (sigma_points - 1).saturating_sub(
        resume_checkpoint
            .map(|c| c.identity.completed_evaluations as usize)
            .unwrap_or(0),
    );
    let mut resource_selection = select_generation_resources(
        remaining_evaluations,
        &mut execution_policy,
        &mut policy_origin,
        GenerationResourceRequest {
            refusal_sidecar: &refusal_artifact_path(&output_dir),
            model: &model,
            model_root_record: &model_root_record,
            device: &device,
            resources: &resources,
            optional_weight_args,
            checkpoint_policy,
            explicit_policy,
            token_ids: &token_ids,
            prompt: &prompt,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
            sigma_points,
            seed,
            target_hidden_state,
            video_shift,
            audio_shift,
            wav_format,
            max_host_mib,
            max_device_mib,
            backend_workspace_mib,
        },
    )?;
    if let Some(initialization) = &recorded_initialization {
        let sidecar = output_dir.join(RESOURCE_SELECTION_FILE);
        super::resource::validate_recorded_selection(&sidecar, &initialization.execution_policy)?;
    }

    let generation_request = GenerationRequest::new(
        GenerationRequestSpec {
            model_root: model_root_record,
            prompt: prompt.clone(),
            token_ids: token_ids.clone(),
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
            wav_format,
            seed,
            target_hidden_state,
            sigma_points,
            video_shift,
            audio_shift,
        },
        &execution_policy,
    )?;
    let qwen_numerical_contract = build_qwen_numerical_contract(
        &device,
        usize::try_from(execution_policy.attention.configured_query_rows)
            .context("execution-policy Qwen query rows exceed usize")?,
        token_ids.len(),
        0,
        0,
        0,
    )?;
    let generation_initialization = GenerationInitialization::new(
        generation_request.clone(),
        execution_policy.clone(),
        qwen_numerical_contract,
    )?;
    if let Some(recorded_initialization) = recorded_initialization.as_ref() {
        let recorded_request = &recorded_initialization.request;
        if let Some(field) = recorded_request.first_difference(&generation_request) {
            bail!("generation resume refused: recorded {field} disagrees with the requested run");
        }
        anyhow::ensure!(
            *recorded_initialization == generation_initialization,
            "generation initialization disagrees with the requested run"
        );
    }

    let start_step = resume_checkpoint
        .map(|checkpoint| {
            usize::try_from(checkpoint.identity.completed_evaluations)
                .context("completed evaluation count exceeds usize")
        })
        .transpose()?
        .unwrap_or(0);
    let schedule_steps = sigma_points - 1;
    anyhow::ensure!(
        start_step <= schedule_steps,
        "generation checkpoint completes {start_step} evaluations, beyond the {schedule_steps}-evaluation schedule"
    );

    let transformer_dir = resolve_component(&model, Path::new("transformer"))?;
    let transformer_config = TransformerConfig::from_file(transformer_dir.join("config.json"))?;
    super::denoise::promote_attention_backend_for_rows(
        &mut execution_policy,
        device.is_cuda(),
        // Row count only: the chunk fields below do not take part in it.
        T2vaGeometry {
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
            ..T2vaGeometry::h3_default(token_ids.len())
        }
        .sequence_rows(transformer_config.patch_size)?
        .total,
        explicit_policy,
    )?;
    let transformer_chunking = execution_policy.transformer_chunking()?;
    if start_step < schedule_steps {
        let preflight = preflight_generation(
            &transformer_dir,
            &transformer_config,
            &execution_policy,
            T2vaGeometry {
                text_rows: token_ids.len(),
                latent_frames,
                latent_height,
                latent_width,
                audio_frames,
                audio_channels,
                attention_projection_chunk_size: transformer_chunking
                    .attention
                    .projection_chunk_size
                    .get(),
                attention_query_chunk_size: transformer_chunking.attention.query_chunk_size.get(),
                attention_key_chunk_policy: transformer_chunking.attention.key,
                ffn_token_chunk_size: transformer_chunking.feed_forward_chunk_size.get(),
                output_token_chunk_size: transformer_chunking.output_chunk_size.get(),
            },
            schedule_steps - start_step,
            max_host_mib,
            max_device_mib,
            backend_workspace_mib,
            &device,
        );
        let preflight = match preflight {
            Ok(preflight) => preflight,
            Err(error) => {
                let Some(refusal) = error.downcast_ref::<PreflightCapacityRefusal>() else {
                    return Err(error);
                };
                if let Some(selected) = resource_selection.as_ref() {
                    let mut provenance = selected.provenance.clone();
                    flyingfish::resource_policy::h3::record_refused_final_admission(
                        &mut provenance,
                        refusal.snapshot.clone(),
                        refusal.budget,
                        refusal.host_peak,
                        refusal.device_peak,
                    )?;
                    provenance.workload.insert("refused".into(), 1);
                    for candidate in &mut provenance.candidates {
                        if candidate.disposition
                            == flyingfish::runtime::resource_selection::CandidateDisposition::Selected
                        {
                            candidate.disposition =
                                flyingfish::runtime::resource_selection::CandidateDisposition::CapacityRejected;
                            candidate.reason = format!("refused by generation preflight: {error}");
                        }
                    }
                    let refusal =
                        anyhow::Error::from(flyingfish::resource_policy::AdmissionRefused {
                            provenance,
                            summary: format!("generation preflight refused: {error}"),
                        });
                    flyingfish::resource_policy::report_refusal(
                        &refusal,
                        Some(&refusal_artifact_path(&output_dir)),
                    );
                    return Err(refusal);
                }
                return Err(error);
            }
        };
        if let Some(selected) = resource_selection.as_mut() {
            flyingfish::resource_policy::h3::record_final_admission(
                &mut selected.provenance,
                preflight.snapshot.clone(),
                preflight.budget,
                preflight.estimate.peak_host_bytes,
                preflight.estimate.peak_device_bytes,
            )?;
        }
        print_preflight(
            &preflight,
            policy_origin,
            start_step,
            schedule_steps,
            &model_root,
        )?;
    } else {
        println!(
            "preflight: denoise already completed {schedule_steps}/{schedule_steps}; transformer admission is not applicable"
        );
    }
    if !existing_run {
        create_new_directory(&output_dir, "generation directory")?;
    }
    if recorded_initialization.is_none() {
        publish_new_bytes_with_staging_parent(
            &initialization_path,
            &generation_initialization.canonical_json()?,
            "generation initialization",
            output_dir
                .parent()
                .context("generation output directory has no parent")?,
        )?;
    }
    if !initialization_complete {
        complete_generation_initialization(
            &checkpoint_dir,
            &staging_dir,
            &policy_path,
            &request_path,
            &ready_path,
            &generation_initialization,
        )?;
    }

    if let Some(selected) = &resource_selection {
        if recorded_initialization.is_none() {
            flyingfish::resource_policy::publish_selection(
                &output_dir.join(RESOURCE_SELECTION_FILE),
                &selected.provenance,
            )?;
        }
        // Outside the fresh-run branch: a resume that first refused and then
        // succeeded leaves the same stale record beside its original selection.
        flyingfish::resource_policy::remove_superseded_refusal(
            &refusal_artifact_path(&output_dir),
            &selected.provenance,
        );
    }

    let weight_source = execution_policy.weight_source();
    let execution_cache_policy = execution_policy.cache_policy()?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;

    let state = load_or_initialize_state(
        resume_checkpoint,
        schedule_steps,
        no_progress,
        &device,
        &transformer_config,
        &generation_request,
        &model,
        &tokenizer,
        &token_ids,
        &prompt,
        target_hidden_state,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        seed,
        weight_source,
        execution_cache_policy,
        transformer_chunking,
    )?;
    anyhow::ensure!(
        state.qwen_numerical_contract == generation_initialization.qwen_numerical_contract,
        "generation prompt/checkpoint Qwen numerical contract disagrees with initialization"
    );
    let conditioning_provenance =
        H3ConditioningProvenance::FlyingfishQwen(Box::new(state.qwen_numerical_contract.clone()));

    let prompt_embeddings = state.prompt_embeddings.clone();
    let text_token_tags_tensor = state.text_token_tags_tensor.clone();
    let (result, policy_history, transformer_stats) = denoise_remaining_steps(
        state,
        schedule_steps,
        &device,
        &execution_policy,
        transformer_dir,
        &conditioning_provenance,
        &checkpoint_dir,
        &staging_dir,
        &output_dir,
        sigma_points,
        video_shift,
        audio_shift,
        no_progress,
    )?;

    if latent_path.exists() {
        anyhow::ensure!(
            resume_checkpoint.is_some_and(|checkpoint| checkpoint.path == latent_path),
            "final latent artifact appeared after resume preflight: {}",
            latent_path.display()
        );
    } else {
        publish_t2va_checkpoint(
            &latent_path,
            Some(&staging_dir),
            &result,
            &prompt_embeddings,
            &text_token_tags_tensor,
            sigma_points,
            video_shift,
            audio_shift,
            &policy_history,
            &conditioning_provenance,
        )
        .with_context(|| {
            format!(
                "failed to publish final generation latents {}",
                latent_path.display()
            )
        })?;
    }

    let DecodeExpectations {
        audio_vae_dir,
        audio_vae_config,
        expected_audio_samples,
        video_vae_dir,
        expected_frame_count,
        expected_frame_height,
        expected_frame_width,
    } = DecodeExpectations::resolve(
        &model,
        audio_frames,
        latent_frames,
        latent_height,
        latent_width,
    )?;

    let frames = DecodeStage {
        device: &device,
        output_dir: &output_dir,
        staging_dir: &staging_dir,
        completion_path: &completion_path,
        wav_path: &wav_path,
        frames_dir: &frames_dir,
        audio_vae_dir: &audio_vae_dir,
        video_vae_dir: &video_vae_dir,
        audio_vae_config: &audio_vae_config,
        audio_channels,
        wav_format,
        expected_audio_samples,
        expected_frame_count,
        expected_frame_height,
        expected_frame_width,
        weight_source,
        execution_cache_policy,
        transformer_chunking,
        no_progress,
    }
    .run(&result)?;

    println!(
        "generated {frames} frames, stereo WAV, and reusable latents in {}",
        output_dir.display()
    );
    if let Some(cache) = transformer_stats {
        let unit = if cache.tensor_retention.is_some() {
            "tensor"
        } else {
            "shard"
        };
        println!(
            "transformer host {unit} cache: {} hits, {} misses, {} evictions, {} header parses",
            cache.hits, cache.misses, cache.evictions, cache.header_parses
        );
    }
    println!(
        "execution policy: schema {} ({})",
        execution_policy.schema_version,
        policy_path.display()
    );
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}

fn create_frame_staging_directory(staging_dir: &Path) -> Result<PathBuf> {
    validate_staging_directory(staging_dir)?;
    for _ in 0..1024 {
        let nonce = FRAME_STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = staging_dir.join(format!(".frames-{}-{nonce:016x}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to create frame staging directory {}",
                        path.display()
                    )
                });
            }
        }
    }
    bail!(
        "failed to allocate a unique frame staging directory in {}",
        staging_dir.display()
    )
}

#[allow(clippy::too_many_arguments)]
fn preflight_generation(
    transformer_dir: &Path,
    config: &TransformerConfig,
    policy: &ExecutionPolicy,
    geometry: T2vaGeometry,
    evaluation_count: usize,
    max_host_mib: Option<u64>,
    max_device_mib: Option<u64>,
    backend_workspace_mib: Option<u64>,
    device: &Device,
) -> Result<GenerationPreflight> {
    policy.validate_device(device)?;
    let sequence_rows = geometry.sequence_rows(config.patch_size)?;
    validate_h3_numerical_backend(
        device,
        policy.flash_attention(),
        geometry.attention_key_chunk_policy,
        usize::try_from(sequence_rows.total).context("packed sequence rows exceed usize")?,
        geometry.text_rows,
        2,
    )
    .context("generation numerical-backend preflight refused before model payload access")?;
    let snapshot = ResourceSnapshot::capture(Some(device));
    let mut budget = probed_budget(
        &snapshot,
        policy.execution_backend,
        max_host_mib,
        max_device_mib,
    )?;
    // Selection promised this reserve on the folded host axis. Charged on the
    // budget side so it applies once; unifying the two homes is an A7 follow-up.
    if snapshot.host_device_memory_is_unified == Some(true) {
        budget.max_host_bytes = Some(
            budget
                .max_host_bytes
                .context("unified generation preflight needs a host bound")?
                .saturating_sub(super::resource::device_residency_reserve_bytes(&snapshot)),
        );
    }

    require_indexed_transformer(transformer_dir)?;
    let weights = ModelWeights::open(transformer_dir, WeightSource::Mmap, CachePolicy::new(1))?;
    let inventory = weights.inventory();
    let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
    if policy.execution_backend == ExecutionBackendPolicy::Cpu {
        assumptions.weight_element_bytes = 4;
        assumptions.activation_element_bytes = 4;
        assumptions.device_memory_is_host = true;
    }
    // A probed unified-memory device (e.g. Jetson) draws device allocations
    // from the host pool; fold the device axis into host accounting instead of
    // checking the two axes against the same bytes independently.
    if snapshot.host_device_memory_is_unified == Some(true) {
        assumptions.device_memory_is_host = true;
    }
    assumptions.use_flash_attention = policy.flash_attention();
    assumptions.evaluation_count =
        u64::try_from(evaluation_count).context("evaluation count exceeds u64")?;
    assumptions.precompute_adaln_steps = if policy.precompute_adaln {
        assumptions.evaluation_count
    } else {
        0
    };
    assumptions.host_weight_cache_bytes = 0;
    assumptions.mapped_weight_residency_bytes = 0;
    let minimum_backend_workspace_mib = if policy.execution_backend == ExecutionBackendPolicy::Cpu {
        0
    } else if policy.flash_attention() {
        DEFAULT_FLASH_BACKEND_WORKSPACE_MIB
    } else {
        DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB
    };
    let backend_workspace_mib = backend_workspace_mib.unwrap_or(minimum_backend_workspace_mib);
    anyhow::ensure!(
        backend_workspace_mib >= minimum_backend_workspace_mib,
        "generation preflight --backend-workspace-mib {backend_workspace_mib} is below the conservative {minimum_backend_workspace_mib} MiB minimum for this backend"
    );
    assumptions.device_weight_cache_bytes = policy.weights.device_cache.max_bytes;
    assumptions.backend_workspace_bytes = mib_to_bytes(backend_workspace_mib)?;
    assumptions.checkpoint_weight_bytes = inventory.indexed_bytes;
    assumptions.peak_materialized_weight_bytes_override = None;
    let mut estimate = ResourceEstimate::for_t2va(config, geometry, assumptions)?;
    if budget.check(&estimate).within_budget {
        let cache_inventory = weights.cache_inventory()?;
        let charge =
            flyingfish::h3::resources::host_weight_residency_charges(policy, &cache_inventory)?;
        assumptions.host_weight_cache_bytes = charge.owned_weight_bytes;
        assumptions.mapped_weight_residency_bytes = charge.mapped_weight_bytes;
        estimate = ResourceEstimate::for_t2va(config, geometry, assumptions)?;
    }
    if let Err(error) = budget.validate(&estimate) {
        let report = budget.check(&estimate);
        let binding = report
            .violations
            .iter()
            .map(|violation| match violation.domain {
                ResourceDomain::Host => "host",
                ResourceDomain::Device => "device",
            })
            .collect::<Vec<_>>()
            .join("+");
        return Err(PreflightCapacityRefusal {
            message: format!(
                "generation preflight admission refused before any model tensor was materialized (binding budget: {binding}): {error}"
            ),
            snapshot,
            budget,
            host_peak: estimate.peak_host_bytes,
            device_peak: estimate.peak_device_bytes,
        }
        .into());
    }
    Ok(GenerationPreflight {
        estimate,
        budget,
        snapshot,
    })
}

fn require_indexed_transformer(transformer_dir: &Path) -> Result<()> {
    let candidates = [
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    ];
    let mut present = Vec::new();
    for name in candidates {
        let path = transformer_dir.join(name);
        if symlink_metadata_if_exists(&path, "transformer weight index")?
            .is_some_and(|metadata| metadata.file_type().is_file())
        {
            present.push(path);
        }
    }
    anyhow::ensure!(
        present.len() == 1,
        "generation metadata-only preflight requires exactly one regular non-symlink transformer weight index, found {}",
        present.len()
    );
    Ok(())
}

fn symlink_metadata_if_exists(path: &Path, label: &str) -> Result<Option<fs::Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("failed to inspect {label} {}", path.display()))
        }
    }
}

pub(super) fn probed_budget(
    snapshot: &ResourceSnapshot,
    backend: ExecutionBackendPolicy,
    max_host_mib: Option<u64>,
    max_device_mib: Option<u64>,
) -> Result<ResourceBudget> {
    let probed_host = [
        snapshot.host_memory_available_bytes,
        snapshot.cgroup_v2_memory_available_bytes,
    ]
    .into_iter()
    .flatten()
    .min();
    let requested_host = max_host_mib.map(mib_to_bytes).transpose()?;
    let max_host_bytes = requested_host.or(probed_host).context(
        "generation preflight cannot measure available host memory; provide --max-host-mib explicitly",
    )?;
    // Under the fold the combined peak is charged against this one bound, and
    // the host view alone can exceed the pool: CUDA free tracks MemFree while
    // MemAvailable counts reclaimable cache.
    anyhow::ensure!(
        !snapshot.unified_accounting_is_undecidable(),
        "generation preflight needs the shared host/device pool size on a unified-memory device, \
         but one of the host and CUDA views could not be measured"
    );
    let max_host_bytes = match snapshot.unified_pool_available_bytes() {
        Some(pool) => max_host_bytes.min(pool),
        None => max_host_bytes,
    };
    let requested_device = max_device_mib.map(mib_to_bytes).transpose()?;
    let max_device_bytes = match backend {
        ExecutionBackendPolicy::Cpu => requested_device,
        ExecutionBackendPolicy::Cuda | ExecutionBackendPolicy::Metal => Some(
            minimum_present(requested_device, snapshot.device_free_memory_bytes).context(
                "generation preflight cannot measure free device memory; provide --max-device-mib explicitly",
            )?,
        ),
    };
    Ok(ResourceBudget {
        max_host_bytes: Some(max_host_bytes),
        max_device_bytes,
    })
}

fn minimum_present(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn print_preflight(
    preflight: &GenerationPreflight,
    policy_origin: PolicyOrigin,
    start_step: usize,
    schedule_steps: usize,
    model_root: &Path,
) -> Result<()> {
    let host_budget = preflight
        .budget
        .max_host_bytes
        .context("preflight host budget is missing")?;
    let device_budget = preflight
        .budget
        .max_device_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "host-shared (CPU)".to_owned());
    println!(
        "preflight admitted: policy from {}, host peak {} / {}, device peak {} / {}",
        policy_origin.description(),
        format_bytes(preflight.estimate.peak_host_bytes),
        format_bytes(host_budget),
        format_bytes(preflight.estimate.peak_device_bytes),
        device_budget
    );
    println!(
        "preflight snapshot: host available {}, cgroup available {}, device free {}",
        display_optional_bytes(preflight.snapshot.host_memory_available_bytes),
        display_optional_bytes(preflight.snapshot.cgroup_v2_memory_available_bytes),
        display_optional_bytes(preflight.snapshot.device_free_memory_bytes)
    );
    report_weight_streaming_window(model_root);
    println!(
        "forecast: evaluations {start_step}..{schedule_steps}; modeled {} per evaluation; wall-clock ETA stabilizes after one cold-start observation plus 5 trailing samples",
        flyingfish::h3::resources::format_flops(
            preflight
                .estimate
                .compute_and_traffic
                .total_flops_per_evaluation
        )
    );
    Ok(())
}

fn display_optional_bytes(value: Option<u64>) -> String {
    value
        .map(format_bytes)
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn required_model_file(model: &Path, relative: &Path) -> Result<PathBuf> {
    let model = fs::canonicalize(model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let candidate = model.join(relative);
    let metadata = fs::symlink_metadata(&candidate)
        .with_context(|| format!("failed to inspect model file {}", candidate.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "model file must be a regular non-symlink file: {}",
        candidate.display()
    );
    let candidate = fs::canonicalize(&candidate)
        .with_context(|| format!("failed to resolve model file {}", candidate.display()))?;
    anyhow::ensure!(
        candidate.starts_with(&model),
        "model file escapes the model root: {}",
        candidate.display()
    );
    Ok(candidate)
}

fn publish_new_bytes_with_staging_parent(
    path: &Path,
    bytes: &[u8],
    label: &str,
    staging_parent: &Path,
) -> Result<()> {
    let staging = ArtifactStaging::new_with_staging_parent(path, staging_parent)
        .with_context(|| format!("failed to stage {label} {}", path.display()))?;
    publish_staged_bytes(staging, bytes)?;
    Ok(())
}

fn load_bounded_regular_file(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && (1..=maximum).contains(&metadata.len()),
        "{label} must be a non-empty regular non-symlink file no larger than {maximum} bytes: {}",
        path.display()
    );
    fs::read(path).with_context(|| format!("failed to read {label} {}", path.display()))
}

fn load_generation_request(path: &Path) -> Result<GenerationRequest> {
    let bytes =
        load_bounded_regular_file(path, MAX_GENERATION_REQUEST_BYTES, "generation request")?;
    GenerationRequest::from_canonical_json(&bytes)
        .with_context(|| format!("failed to load generation request {}", path.display()))
}

fn load_generation_initialization(path: &Path) -> Result<GenerationInitialization> {
    let bytes = load_bounded_regular_file(
        path,
        MAX_GENERATION_INITIALIZATION_BYTES,
        "generation initialization",
    )?;
    GenerationInitialization::from_canonical_json(&bytes).with_context(|| {
        format!(
            "failed to load generation initialization {}",
            path.display()
        )
    })
}

fn load_generation_completion(path: &Path) -> Result<GenerationCompletion> {
    let bytes = load_bounded_regular_file(
        path,
        MAX_GENERATION_COMPLETION_BYTES,
        "generation completion",
    )?;
    GenerationCompletion::from_canonical_json(&bytes)
        .with_context(|| format!("failed to load generation completion {}", path.display()))
}

fn frame_manifest_bytes(directory: &Path) -> Result<u64> {
    let path = directory.join(FRAME_MANIFEST_FILE);
    let snapshot = read_artifact_snapshot(
        &path,
        u64::try_from(MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES)
            .context("frame manifest limit exceeds u64")?,
    )
    .with_context(|| format!("failed to read frame manifest {}", path.display()))?;
    u64::try_from(snapshot.bytes.len()).context("frame manifest size exceeds u64")
}

#[allow(clippy::too_many_arguments)]
fn validate_completed_decode(
    completion: &GenerationCompletion,
    wav_path: &Path,
    frames_dir: &Path,
    expected_sample_rate: u32,
    expected_channels: usize,
    expected_samples_per_channel: usize,
    expected_format: WavSampleFormat,
    expected_frame_count: usize,
    expected_frame_width: usize,
    expected_frame_height: usize,
) -> Result<()> {
    completion.validate()?;
    validate_existing_wav(
        wav_path,
        expected_sample_rate,
        expected_channels,
        expected_samples_per_channel,
        expected_format,
    )?;
    let frames = validate_existing_frames(
        frames_dir,
        expected_frame_count,
        expected_frame_width,
        expected_frame_height,
    )?;
    anyhow::ensure!(
        u64::try_from(frames).context("verified frame count exceeds u64")?
            == completion.frame_count,
        "completed frame count disagrees with the verified frame set"
    );
    let wav_stat = FileStat::of_target(wav_path)
        .with_context(|| format!("failed to inspect generated WAV {}", wav_path.display()))?;
    anyhow::ensure!(
        wav_stat.is_file() && wav_stat.len() == completion.wav_bytes,
        "generated WAV size disagrees with the atomic completion record"
    );
    anyhow::ensure!(
        frame_manifest_bytes(frames_dir)? == completion.frame_manifest_bytes,
        "frame-manifest size disagrees with the atomic completion record"
    );
    Ok(())
}

fn load_canonical_execution_policy(path: &Path) -> Result<ExecutionPolicy> {
    let bytes = load_bounded_regular_file(path, MAX_EXECUTION_POLICY_BYTES, "execution policy")?;
    let policy = ExecutionPolicy::from_json(&bytes)
        .with_context(|| format!("failed to load execution policy {}", path.display()))?;
    anyhow::ensure!(
        bytes == policy.canonical_json()?,
        "execution-policy JSON is not in canonical schema-{} encoding: {}",
        policy.schema_version,
        path.display()
    );
    Ok(policy)
}

fn validate_generation_directory_entries(directory: &Path) -> Result<()> {
    let allowed = [
        GENERATION_INITIALIZATION_FILE,
        GENERATION_READY_FILE,
        GENERATION_REQUEST_FILE,
        EXECUTION_POLICY_FILE,
        RESOURCE_SELECTION_FILE,
        RESOURCE_REFUSAL_FILE,
        CHECKPOINT_DIRECTORY,
        STAGING_DIRECTORY,
        FINAL_LATENTS_FILE,
        WAV_FILE,
        FRAMES_DIRECTORY,
        GENERATION_COMPLETE_FILE,
    ];
    for entry in fs::read_dir(directory).with_context(|| {
        format!(
            "failed to inspect generation directory {}",
            directory.display()
        )
    })? {
        let entry = entry.context("failed to inspect generation directory entry")?;
        let name = entry.file_name();
        let name = name.to_str().with_context(|| {
            format!(
                "generation directory entry is not valid UTF-8: {}",
                entry.path().display()
            )
        })?;
        anyhow::ensure!(
            allowed.contains(&name),
            "unexpected generation directory entry {name:?}; refusing an ambiguous resume"
        );
    }
    Ok(())
}

fn validate_ready_marker(path: &Path) -> Result<()> {
    let bytes = load_bounded_regular_file(
        path,
        u64::try_from(GENERATION_READY_BYTES.len()).expect("ready marker length fits u64"),
        "generation ready marker",
    )?;
    anyhow::ensure!(
        bytes == GENERATION_READY_BYTES,
        "generation ready marker has unexpected contents: {}",
        path.display()
    );
    Ok(())
}

fn validate_incomplete_initialization(output_dir: &Path, checkpoint_dir: &Path) -> Result<()> {
    for path in [
        output_dir.join(FINAL_LATENTS_FILE),
        output_dir.join(WAV_FILE),
        output_dir.join(FRAMES_DIRECTORY),
        output_dir.join(GENERATION_COMPLETE_FILE),
    ] {
        anyhow::ensure!(
            !path.exists(),
            "incomplete generation initialization already has execution artifact {}; refusing repair",
            path.display()
        );
    }
    if checkpoint_dir.exists() {
        let metadata = fs::symlink_metadata(checkpoint_dir).with_context(|| {
            format!(
                "failed to inspect incomplete checkpoint directory {}",
                checkpoint_dir.display()
            )
        })?;
        anyhow::ensure!(
            metadata.file_type().is_dir(),
            "incomplete checkpoint path is not a non-symlink directory: {}",
            checkpoint_dir.display()
        );
        anyhow::ensure!(
            fs::read_dir(checkpoint_dir)
                .with_context(|| format!("failed to read {}", checkpoint_dir.display()))?
                .next()
                .is_none(),
            "incomplete generation initialization has checkpoint entries; refusing repair"
        );
    }
    let staging_dir = output_dir.join(STAGING_DIRECTORY);
    if staging_dir.exists() {
        validate_staging_directory(&staging_dir)?;
    }
    Ok(())
}

fn validate_staging_directory(staging_dir: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(staging_dir).with_context(|| {
        format!(
            "failed to inspect generation staging directory {}",
            staging_dir.display()
        )
    })?;
    anyhow::ensure!(
        metadata.file_type().is_dir(),
        "generation staging path is not a non-symlink directory: {}",
        staging_dir.display()
    );
    let resolved = fs::canonicalize(staging_dir).with_context(|| {
        format!(
            "failed to resolve generation staging directory {}",
            staging_dir.display()
        )
    })?;
    let parent = fs::canonicalize(
        staging_dir
            .parent()
            .context("generation staging directory has no run parent")?,
    )?;
    anyhow::ensure!(
        resolved.parent() == Some(parent.as_path()),
        "generation staging directory resolves outside its run directory"
    );
    Ok(())
}

/// A directory holding only refusal diagnostics is retryable, not partial.
fn validate_empty_uninitialized_directory(output_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(output_dir)
        .with_context(|| format!("failed to read {}", output_dir.display()))?
    {
        let name = entry
            .with_context(|| format!("failed to read an entry of {}", output_dir.display()))?
            .file_name();
        anyhow::ensure!(
            name == RESOURCE_REFUSAL_FILE,
            "generation directory has partial state without an initialization record"
        );
    }
    Ok(())
}

fn complete_generation_initialization(
    checkpoint_dir: &Path,
    staging_dir: &Path,
    policy_path: &Path,
    request_path: &Path,
    ready_path: &Path,
    initialization: &GenerationInitialization,
) -> Result<()> {
    anyhow::ensure!(
        !ready_path.exists(),
        "generation ready marker appeared during initialization: {}",
        ready_path.display()
    );
    if policy_path.exists() {
        anyhow::ensure!(
            load_canonical_execution_policy(policy_path)? == initialization.execution_policy,
            "partial generation execution-policy artifact disagrees with initialization"
        );
    }
    if request_path.exists() {
        anyhow::ensure!(
            load_generation_request(request_path)? == initialization.request,
            "partial generation-request artifact disagrees with initialization"
        );
    }
    if checkpoint_dir.exists() {
        validate_incomplete_initialization(
            checkpoint_dir
                .parent()
                .context("checkpoint directory has no generation parent")?,
            checkpoint_dir,
        )?;
    } else {
        create_new_directory(checkpoint_dir, "generation checkpoint directory")?;
    }
    if !staging_dir.exists() {
        create_new_directory(staging_dir, "generation staging directory")?;
    } else {
        validate_staging_directory(staging_dir)?;
    }
    if !policy_path.exists() {
        publish_new_bytes_with_staging_parent(
            policy_path,
            &initialization.execution_policy.canonical_json()?,
            "execution policy",
            staging_dir,
        )?;
    }
    if !request_path.exists() {
        publish_new_bytes_with_staging_parent(
            request_path,
            &initialization.request.canonical_json()?,
            "generation request",
            staging_dir,
        )?;
    }
    publish_new_bytes_with_staging_parent(
        ready_path,
        GENERATION_READY_BYTES,
        "generation ready marker",
        staging_dir,
    )?;
    Ok(())
}

fn discover_latest_checkpoint(
    directory: &Path,
    sigma_points: usize,
) -> Result<Option<ResumeCheckpoint>> {
    let metadata = fs::symlink_metadata(directory).with_context(|| {
        format!(
            "existing generation is missing checkpoint directory {}",
            directory.display()
        )
    })?;
    anyhow::ensure!(
        metadata.file_type().is_dir(),
        "generation checkpoint path is not a directory: {}",
        directory.display()
    );
    let mut checkpoints = BTreeMap::new();
    for entry in fs::read_dir(directory).with_context(|| {
        format!(
            "failed to read checkpoint directory {}",
            directory.display()
        )
    })? {
        let entry = entry.context("failed to inspect checkpoint directory entry")?;
        let file_name = entry.file_name();
        let file_name = file_name.to_str().with_context(|| {
            format!(
                "checkpoint directory entry is not valid UTF-8: {}",
                entry.path().display()
            )
        })?;
        let digits = file_name
            .strip_prefix("checkpoint-step")
            .and_then(|value| value.strip_suffix(".safetensors"))
            .with_context(|| format!("unexpected checkpoint directory entry {file_name:?}"))?;
        anyhow::ensure!(
            digits.len() >= 6 && digits.bytes().all(|byte| byte.is_ascii_digit()),
            "malformed generation checkpoint filename {file_name:?}"
        );
        let step = digits
            .parse::<usize>()
            .with_context(|| format!("checkpoint step exceeds usize in {file_name:?}"))?;
        anyhow::ensure!(
            file_name == format!("checkpoint-step{step:06}.safetensors"),
            "generation checkpoint filename is not canonical: {file_name:?}"
        );
        let entry_metadata = fs::symlink_metadata(entry.path()).with_context(|| {
            format!(
                "failed to inspect generation checkpoint {}",
                entry.path().display()
            )
        })?;
        anyhow::ensure!(
            entry_metadata.file_type().is_file() && entry_metadata.len() > 0,
            "generation checkpoint must be a non-empty regular non-symlink file: {}",
            entry.path().display()
        );
        anyhow::ensure!(
            checkpoints.insert(step, entry.path()).is_none(),
            "duplicate generation checkpoint step {step}"
        );
    }
    for (expected, actual) in (1usize..).zip(checkpoints.keys().copied()) {
        anyhow::ensure!(
            actual == expected,
            "generation checkpoint sequence is not contiguous: expected step {expected}, found {actual}"
        );
    }
    let mut latest: Option<ResumeCheckpoint> = None;
    for (step, path) in checkpoints {
        anyhow::ensure!(
            step < sigma_points,
            "generation checkpoint step {step} exceeds the {}-evaluation schedule",
            sigma_points - 1
        );
        let checkpoint = load_resume_checkpoint(&path, sigma_points)?;
        anyhow::ensure!(
            checkpoint.identity.completed_evaluations
                == u64::try_from(step).context("checkpoint step exceeds u64")?,
            "generation checkpoint filename step {step} disagrees with its embedded completed evaluation {}",
            checkpoint.identity.completed_evaluations
        );
        if let Some(previous) = latest.as_ref() {
            checkpoint
                .identity
                .policy_history
                .validate_extends(&previous.identity.policy_history)
                .with_context(|| {
                    format!("generation checkpoint step {step} rewrites the prior policy history")
                })?;
        }
        latest = Some(checkpoint);
    }
    Ok(latest)
}

fn load_resume_checkpoint(path: &Path, sigma_points: usize) -> Result<ResumeCheckpoint> {
    let identity = CheckpointIdentity::collect(path).with_context(|| {
        format!(
            "failed to validate generation checkpoint {}",
            path.display()
        )
    })?;
    anyhow::ensure!(
        usize::try_from(identity.sigma_points).context("checkpoint sigma_points exceeds usize")?
            == sigma_points,
        "generation checkpoint sigma_points {} disagrees with requested {sigma_points}",
        identity.sigma_points
    );
    Ok(ResumeCheckpoint {
        path: path.to_path_buf(),
        identity,
    })
}

fn validate_checkpoint_chain(
    latest: Option<&ResumeCheckpoint>,
    final_checkpoint: Option<&ResumeCheckpoint>,
    recorded_policy: Option<&ExecutionPolicy>,
    sigma_points: usize,
) -> Result<()> {
    let schedule_steps = sigma_points - 1;
    if let Some(final_checkpoint) = final_checkpoint {
        anyhow::ensure!(
            usize::try_from(final_checkpoint.identity.completed_evaluations)
                .context("final completed evaluation count exceeds usize")?
                == schedule_steps,
            "final latent artifact records {} evaluations, expected {schedule_steps}",
            final_checkpoint.identity.completed_evaluations
        );
        let latest = latest.context(
            "final latent artifact exists without the required per-evaluation checkpoint chain",
        )?;
        anyhow::ensure!(
            latest.identity == final_checkpoint.identity,
            "final latent artifact identity disagrees with the last per-evaluation checkpoint"
        );
    }
    if let (Some(checkpoint), Some(policy)) = (final_checkpoint.or(latest), recorded_policy) {
        for segment in &checkpoint.identity.policy_history.segments {
            anyhow::ensure!(
                segment.policy == *policy,
                "generation checkpoint evaluations [{}, {}) use a different policy than the recorded execution policy",
                segment.start_evaluation,
                segment.end_evaluation_exclusive
            );
        }
    }
    Ok(())
}

fn load_generation_state(
    path: &Path,
    device: &Device,
    config: &TransformerConfig,
    request: &GenerationRequest,
) -> Result<GenerationState> {
    let mut values = safetensors::load(path, device)
        .with_context(|| format!("failed to load generation checkpoint {}", path.display()))?;
    let qwen_numerical_contract = H3QwenNumericalContract::take_artifact_tensors(&mut values)?
        .context("generation checkpoint is missing the required Qwen numerical contract")?;
    let metadata = take_t2va_checkpoint_metadata(&mut values)?
        .context("generation resume source is not a recovery checkpoint")?;
    anyhow::ensure!(
        usize::try_from(metadata.sigma_points).context("checkpoint sigma_points exceeds usize")?
            == request.sigma_points
            && metadata.video_shift.to_bits() == request.video_shift_f32_bits
            && metadata.audio_shift.to_bits() == request.audio_shift_f32_bits,
        "generation checkpoint schedule disagrees with the recorded generation request"
    );
    let prompt_embeddings = take_input(&mut values, "prompt_embeddings")?;
    let text_token_tags_tensor = take_input(&mut values, "text_token_tags")?;
    let text_token_tags = text_token_tags_tensor
        .to_vec1::<u32>()
        .context("text_token_tags must be a U32 vector")?;
    let video_latents = take_input(&mut values, "video_latents")?;
    let audio_latents = take_input(&mut values, "audio_latents")?;
    anyhow::ensure!(
        values.is_empty(),
        "generation checkpoint contains unexpected tensors: {}",
        sorted_tensor_names(&values).join(", ")
    );
    anyhow::ensure!(
        prompt_embeddings.dims3()? == (1, request.token_ids.len(), config.text_dim),
        "generation checkpoint prompt embedding shape disagrees with the recorded request"
    );
    anyhow::ensure!(
        text_token_tags.len() == request.token_ids.len(),
        "generation checkpoint text-token tag count disagrees with the recorded request"
    );
    validate_qwen_numerical_contract(
        &qwen_numerical_contract,
        device,
        request.token_ids.len(),
        0,
        0,
        0,
    )?;
    anyhow::ensure!(
        video_latents.dims5()?
            == (
                1,
                config.in_channels,
                request.latent_frames,
                request.latent_height,
                request.latent_width,
            ),
        "generation checkpoint video latent shape disagrees with the recorded request"
    );
    anyhow::ensure!(
        audio_latents.dims3()?
            == (
                request.audio_channels,
                config.audio_in_channels,
                request.audio_frames,
            ),
        "generation checkpoint audio latent shape disagrees with the recorded request"
    );
    Ok(GenerationState {
        prompt_embeddings,
        text_token_tags_tensor,
        text_token_tags,
        video_latents,
        audio_latents,
        completed_steps: usize::try_from(metadata.completed_evaluations)
            .context("completed evaluation count exceeds usize")?,
        policy_history: metadata.policy_history,
        qwen_numerical_contract,
    })
}

fn validate_existing_wav(
    path: &Path,
    expected_sample_rate: u32,
    expected_channels: usize,
    expected_samples_per_channel: usize,
    expected_format: WavSampleFormat,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect generated WAV {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && metadata.len() > 0,
        "generated WAV must be a non-empty regular non-symlink file: {}",
        path.display()
    );
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to validate generated WAV {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        usize::from(spec.channels) == expected_channels && spec.sample_rate == expected_sample_rate,
        "existing generated WAV channel/rate metadata disagrees with this run"
    );
    let expected_total_samples = u64::try_from(expected_channels)
        .context("expected WAV channel count exceeds u64")?
        .checked_mul(
            u64::try_from(expected_samples_per_channel)
                .context("expected WAV sample count exceeds u64")?,
        )
        .context("expected WAV interleaved sample count overflow")?;
    match expected_format {
        WavSampleFormat::Pcm16 => {
            anyhow::ensure!(
                spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16,
                "existing generated WAV is not the requested pcm16 format"
            );
            let samples =
                reader
                    .samples::<i16>()
                    .try_fold(0u64, |count, sample| -> Result<u64> {
                        sample.context("existing generated WAV contains an invalid PCM sample")?;
                        count.checked_add(1).context("WAV sample count overflow")
                    })?;
            anyhow::ensure!(
                samples == expected_total_samples,
                "existing generated WAV has {samples} interleaved samples, expected {expected_total_samples}"
            );
        }
        WavSampleFormat::Float32 => {
            anyhow::ensure!(
                spec.sample_format == hound::SampleFormat::Float && spec.bits_per_sample == 32,
                "existing generated WAV is not the requested float32 format"
            );
            let samples =
                reader
                    .samples::<f32>()
                    .try_fold(0u64, |count, sample| -> Result<u64> {
                        let sample = sample
                            .context("existing generated WAV contains an invalid float sample")?;
                        anyhow::ensure!(
                            sample.is_finite(),
                            "existing generated WAV contains NaN/Inf"
                        );
                        count.checked_add(1).context("WAV sample count overflow")
                    })?;
            anyhow::ensure!(
                samples == expected_total_samples,
                "existing generated WAV has {samples} interleaved samples, expected {expected_total_samples}"
            );
        }
    }
    Ok(())
}

fn validate_existing_frames(
    directory: &Path,
    expected_frame_count: usize,
    expected_width: usize,
    expected_height: usize,
) -> Result<usize> {
    let metadata = fs::symlink_metadata(directory)
        .with_context(|| format!("failed to inspect frame directory {}", directory.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_dir(),
        "frame output must be a non-symlink directory: {}",
        directory.display()
    );
    let manifest_path = directory.join(FRAME_MANIFEST_FILE);
    let snapshot = read_artifact_snapshot(
        &manifest_path,
        u64::try_from(MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES)
            .context("frame manifest limit exceeds u64")?,
    )
    .with_context(|| {
        format!(
            "existing frame directory is incomplete; its completion manifest is unavailable at {}",
            manifest_path.display()
        )
    })?;
    let manifest = PngFrameSetManifest::from_canonical_json(&snapshot.bytes)?;
    anyhow::ensure!(
        usize::try_from(manifest.frame_count).context("frame count exceeds usize")?
            == expected_frame_count
            && usize::try_from(manifest.width).context("frame width exceeds usize")?
                == expected_width
            && usize::try_from(manifest.height).context("frame height exceeds usize")?
                == expected_height,
        "existing frame manifest geometry/count disagrees with this run"
    );
    manifest.verify_completed_directory(directory, FRAME_MANIFEST_FILE)?;
    usize::try_from(manifest.frame_count).context("frame count exceeds usize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::default_execution_policy;
    use candle_core::{DType, Shape, safetensors};
    use flyingfish::h3::conditioning_provenance::ExternalPromptProvenance;
    use flyingfish::h3::core::AttentionKeyChunkPolicy;
    use flyingfish::h3::policy::H3QwenVisionLinearGeometry;
    use serde_json::json;
    use std::num::NonZeroUsize;
    use std::{
        collections::HashMap,
        sync::{Arc, Barrier},
    };

    fn policy() -> ExecutionPolicy {
        default_execution_policy(&Device::Cpu)
    }

    fn qwen_contract(policy: &ExecutionPolicy, rows: usize) -> H3QwenNumericalContract {
        H3QwenNumericalContract::for_verified_target(
            ExecutionBackendPolicy::Cpu,
            NonZeroUsize::new(policy.attention.configured_query_rows as usize).unwrap(),
            NonZeroUsize::new(rows).unwrap(),
            0,
            H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
        )
        .unwrap()
    }

    fn request(policy: &ExecutionPolicy) -> GenerationRequest {
        GenerationRequest::new(
            GenerationRequestSpec {
                model_root: std::env::temp_dir()
                    .join("flyingfish-generation-test-model")
                    .to_string_lossy()
                    .into_owned(),
                prompt: "a kite".to_owned(),
                token_ids: vec![1, 2],
                latent_frames: 2,
                latent_height: 2,
                latent_width: 2,
                audio_frames: 2,
                audio_channels: 2,
                wav_format: WavSampleFormat::Pcm16,
                seed: 42,
                target_hidden_state: 2,
                sigma_points: 3,
                video_shift: 12.0,
                audio_shift: 3.0,
            },
            policy,
        )
        .unwrap()
    }

    fn checkpoint(path: &str, history: PolicyHistory) -> ResumeCheckpoint {
        let identity = CheckpointIdentity {
            schema_version: flyingfish::recovery::CHECKPOINT_IDENTITY_SCHEMA_VERSION,
            checkpoint_bytes: 4096,
            completed_evaluations: history.completed_evaluations,
            sigma_points: 3,
            video_shift_f32_bits: 12.0f32.to_bits(),
            audio_shift_f32_bits: 3.0f32.to_bits(),
            policy_history: history,
        };
        identity.validate().unwrap();
        ResumeCheckpoint {
            path: PathBuf::from(path),
            identity,
        }
    }

    fn publish_checkpoint_at(path: &Path, completed_steps: usize) {
        let policy = policy();
        let mut history = PolicyHistory::new();
        history
            .append_successful_evaluations(completed_steps as u64, policy.clone())
            .unwrap();
        let latents = T2vaLatents {
            video: Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            audio: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            completed_steps,
        };
        let prompt = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let tags = Tensor::new(&[1u32], &Device::Cpu).unwrap();
        publish_t2va_checkpoint(
            path,
            None,
            &latents,
            &prompt,
            &tags,
            4,
            12.0,
            3.0,
            &history,
            &H3ConditioningProvenance::FlyingfishQwen(Box::new(qwen_contract(&policy, 1))),
        )
        .unwrap();
    }

    fn publish_checkpoint_file(directory: &Path, completed_steps: usize) -> PathBuf {
        let path = directory.join(format!("checkpoint-step{completed_steps:06}.safetensors"));
        publish_checkpoint_at(&path, completed_steps);
        path
    }

    #[test]
    fn checkpoint_publication_refuses_invalid_conditioning_before_linking_output() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("invalid-conditioning.safetensors");
        let latents = T2vaLatents {
            video: Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            audio: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            completed_steps: 0,
        };
        let prompt = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let tags = Tensor::new(&[1u32], &Device::Cpu).unwrap();
        let invalid = H3ConditioningProvenance::ExternallyValidated(ExternalPromptProvenance {
            schema_version: 0,
            binding: "invalid".to_owned(),
            manifest_bytes: 1,
            input_bytes: 1,
            manifest_producer: "invalid".to_owned(),
            manifest_schema_version: 0,
        });
        let error = publish_t2va_checkpoint(
            &path,
            None,
            &latents,
            &prompt,
            &tags,
            2,
            12.0,
            3.0,
            &PolicyHistory::new(),
            &invalid,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("external prompt provenance schema")
        );
        assert!(!path.exists());
    }

    fn test_ones(shape: impl Into<Shape>) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn test_zeros(shape: impl Into<Shape>) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn write_tiny_audio_decoder(directory: &Path) {
        fs::create_dir(directory).unwrap();
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "AutoencoderKLMiniMaxH3Audio",
                "latent_dim": 2,
                "latent_channels": 1,
                "decoder_dim": 2,
                "decoder_rates": [2],
                "decoder_kernel_sizes": [4],
                "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1]],
                "sampling_rate": 32000,
                "latents_mean": [0.0],
                "latents_std": [1.0]
            }))
            .unwrap(),
        )
        .unwrap();
        let mut weights = HashMap::from([
            ("dec_in_proj.weight".to_owned(), test_ones((2, 1, 1))),
            ("dec_in_proj.bias".to_owned(), test_ones(2)),
            ("decoder.conv_pre.weight_g".to_owned(), test_ones((2, 1, 1))),
            ("decoder.conv_pre.weight_v".to_owned(), test_ones((2, 2, 7))),
            ("decoder.conv_pre.bias".to_owned(), test_ones(2)),
            ("decoder.ups.0.0.weight_g".to_owned(), test_ones((2, 1, 1))),
            ("decoder.ups.0.0.weight_v".to_owned(), test_ones((2, 1, 4))),
            ("decoder.ups.0.0.bias".to_owned(), test_ones(1)),
            (
                "decoder.resblocks.0.convs1.0.weight_g".to_owned(),
                test_ones((1, 1, 1)),
            ),
            (
                "decoder.resblocks.0.convs1.0.weight_v".to_owned(),
                test_ones((1, 1, 3)),
            ),
            ("decoder.resblocks.0.convs1.0.bias".to_owned(), test_ones(1)),
            (
                "decoder.resblocks.0.convs2.0.weight_g".to_owned(),
                test_ones((1, 1, 1)),
            ),
            (
                "decoder.resblocks.0.convs2.0.weight_v".to_owned(),
                test_ones((1, 1, 3)),
            ),
            ("decoder.resblocks.0.convs2.0.bias".to_owned(), test_ones(1)),
            (
                "decoder.conv_post.weight_g".to_owned(),
                test_ones((1, 1, 1)),
            ),
            (
                "decoder.conv_post.weight_v".to_owned(),
                test_ones((1, 1, 7)),
            ),
        ]);
        for prefix in [
            "decoder.resblocks.0.activations.0",
            "decoder.resblocks.0.activations.1",
            "decoder.activation_post",
        ] {
            weights.insert(format!("{prefix}.act.alpha"), test_zeros(1));
            weights.insert(format!("{prefix}.act.beta"), test_zeros(1));
            weights.insert(format!("{prefix}.upsample.filter"), test_ones((1, 1, 12)));
            weights.insert(
                format!("{prefix}.downsample.lowpass.filter"),
                test_ones((1, 1, 12)),
            );
        }
        safetensors::save(
            &weights,
            directory.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
    }

    fn write_tiny_video_decoder(directory: &Path) {
        fs::create_dir(directory).unwrap();
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "AutoencoderKLMiniMaxH3",
                "out_channels": 3,
                "latent_channels": 1,
                "spatial_downsample_factors": [2],
                "temporal_downsample_factors": [2],
                "decoder_num_layers": 1,
                "decoder_num_attention_heads": 1,
                "decoder_attention_head_dim": 6,
                "decoder_num_register_tokens": 1,
                "decoder_ffn_mult": 1,
                "decoder_rope_theta": 100.0,
                "decoder_rope_dim_ratio": 1.0,
                "decoder_norm_eps": 1e-5,
                "clip_length": 1,
                "token_drop": 0,
                "latents_mean": [0.0],
                "latents_std": [1.0]
            }))
            .unwrap(),
        )
        .unwrap();
        let prefix = "decoder.transformer_blocks.0";
        let mut weights = HashMap::from([
            (
                "post_quant_conv.weight".to_owned(),
                test_ones((1, 1, 1, 1, 1)),
            ),
            ("post_quant_conv.bias".to_owned(), test_zeros(1)),
            ("decoder.proj_in.weight".to_owned(), test_ones((6, 1))),
            ("decoder.proj_in.bias".to_owned(), test_zeros(6)),
            ("decoder.register_tokens".to_owned(), test_zeros((1, 1, 6))),
            (format!("{prefix}.norm1.weight"), test_ones(6)),
            (format!("{prefix}.norm2.weight"), test_ones(6)),
            (format!("{prefix}.scale1"), test_zeros(6)),
            (format!("{prefix}.scale2"), test_zeros(6)),
            (format!("{prefix}.ff.net.0.proj.weight"), test_ones((12, 6))),
            (format!("{prefix}.ff.net.0.proj.bias"), test_zeros(12)),
            (format!("{prefix}.ff.net.2.weight"), test_ones((6, 6))),
            (format!("{prefix}.ff.net.2.bias"), test_zeros(6)),
            ("decoder.norm_out.weight".to_owned(), test_ones(6)),
            ("decoder.norm_out.bias".to_owned(), test_zeros(6)),
            ("decoder.proj_out.weight".to_owned(), test_ones((24, 6))),
            ("decoder.proj_out.bias".to_owned(), test_zeros(24)),
        ]);
        for name in ["to_q", "to_k", "to_v", "to_out.0"] {
            weights.insert(format!("{prefix}.attn.{name}.weight"), test_ones((6, 6)));
            weights.insert(format!("{prefix}.attn.{name}.bias"), test_zeros(6));
        }
        safetensors::save(
            &weights,
            directory.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
    }

    fn frame_file_names(directory: &Path) -> Vec<String> {
        let mut files = fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        files.sort();
        files
    }

    fn assert_frame_directories_byte_identical(left: &Path, right: &Path) {
        let left_names = frame_file_names(left);
        let right_names = frame_file_names(right);
        assert_eq!(left_names, right_names);
        for name in left_names {
            assert_eq!(
                fs::read(left.join(&name)).unwrap(),
                fs::read(right.join(&name)).unwrap(),
                "frame artifact {name} differs"
            );
        }
    }

    #[test]
    fn concurrent_decode_is_byte_identical_to_serial_and_records_real_overlap() {
        let temporary = tempfile::tempdir().unwrap();
        let serial = temporary.path().join("serial");
        let concurrent = temporary.path().join("concurrent");
        let audio_component = temporary.path().join("audio-vae");
        let video_component = temporary.path().join("video-vae");
        write_tiny_audio_decoder(&audio_component);
        write_tiny_video_decoder(&video_component);
        let audio_latents = test_zeros((2, 1, 3));
        let video_latents = test_ones((1, 1, 1, 1, 1));
        fs::create_dir(&serial).unwrap();
        fs::create_dir(&concurrent).unwrap();
        let serial_frames = serial.join(FRAMES_DIRECTORY);
        fs::create_dir(&serial_frames).unwrap();
        let serial_wav = serial.join(WAV_FILE);
        let serial_audio_decoder = StreamedAudioVae::open(
            &audio_component,
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
        )
        .unwrap();
        let serial_waveform = serial_audio_decoder.decode(&audio_latents).unwrap();
        write_wav(
            &serial_wav,
            &serial_waveform,
            32_000,
            WavSampleFormat::Pcm16,
        )
        .unwrap();
        let serial_video_decoder = StreamedVideoVae::open(
            &video_component,
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
            1,
        )
        .unwrap();
        let serial_frame_count = serial_video_decoder
            .decode_to_png_frames(&video_latents, &serial_frames)
            .unwrap();
        publish_png_frame_manifest(&serial_frames, serial_frame_count).unwrap();

        let staging_dir = concurrent.join(STAGING_DIRECTORY);
        fs::create_dir(&staging_dir).unwrap();
        let concurrent_wav = concurrent.join(WAV_FILE);
        let concurrent_frames = concurrent.join(FRAMES_DIRECTORY);
        let gate = Arc::new(Barrier::new(2));
        let audio_gate = Arc::clone(&gate);
        let video_gate = Arc::clone(&gate);
        let concurrent_audio_latents = audio_latents.clone();
        let concurrent_video_latents = video_latents.clone();
        let decode = execute_decode_branches(
            || {
                audio_gate.wait();
                thread::sleep(Duration::from_millis(40));
                let staging = ArtifactStaging::new_for_path_producer_with_staging_parent(
                    &concurrent_wav,
                    &staging_dir,
                )?;
                let decoder = StreamedAudioVae::open(
                    &audio_component,
                    WeightSource::Mmap,
                    CachePolicy::new(1),
                    Device::Cpu,
                )?;
                let waveform = decoder.decode(&concurrent_audio_latents)?;
                write_wav(
                    staging.producer_path(),
                    &waveform,
                    32_000,
                    WavSampleFormat::Pcm16,
                )?;
                Ok((
                    PreparedAudioDecode::Staged(Box::new(staging)),
                    DecodeBranchAction::Decoded,
                ))
            },
            || {
                video_gate.wait();
                thread::sleep(Duration::from_millis(40));
                let directory = create_frame_staging_directory(&staging_dir)?;
                let decoder = StreamedVideoVae::open(
                    &video_component,
                    WeightSource::Mmap,
                    CachePolicy::new(1),
                    Device::Cpu,
                    1,
                )?;
                let frames = decoder.decode_to_png_frames(&concurrent_video_latents, &directory)?;
                publish_png_frame_manifest(&directory, frames)?;
                Ok((
                    PreparedVideoDecode::Staged { directory, frames },
                    DecodeBranchAction::Decoded,
                ))
            },
        )
        .unwrap();

        assert!(!concurrent_wav.exists());
        assert!(!concurrent_frames.exists());
        let frames = decode.video.value.frame_count();
        let audio_measurement = decode.audio.measurement.clone();
        let video_measurement = decode.video.measurement.clone();
        let wall_elapsed = decode.wall_elapsed;
        let wav_digest = decode.audio.value.publish(&concurrent_wav).unwrap();
        let manifest_digest = decode
            .video
            .value
            .publish(&concurrent_frames, &concurrent)
            .unwrap();
        assert_eq!(
            fs::read(&serial_wav).unwrap(),
            fs::read(&concurrent_wav).unwrap()
        );
        assert_frame_directories_byte_identical(&serial_frames, &concurrent_frames);

        let completion = GenerationCompletion::new(
            audio_measurement,
            video_measurement,
            wall_elapsed,
            wav_digest,
            manifest_digest,
            frames,
        )
        .unwrap();
        assert!(completion.branch_overlap_ns >= 20_000_000);
        let completion_path = concurrent.join(GENERATION_COMPLETE_FILE);
        assert!(!completion_path.exists());
        publish_new_bytes_with_staging_parent(
            &completion_path,
            &completion.canonical_json().unwrap(),
            "test generation completion",
            &staging_dir,
        )
        .unwrap();
        let loaded = load_generation_completion(&completion_path).unwrap();
        assert_eq!(loaded, completion);
        validate_completed_decode(
            &loaded,
            &concurrent_wav,
            &concurrent_frames,
            32_000,
            2,
            6,
            WavSampleFormat::Pcm16,
            1,
            2,
            2,
        )
        .unwrap();
    }

    #[test]
    fn decode_failure_propagates_without_publishing_the_successful_branch() {
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("run");
        let staging_dir = output.join(STAGING_DIRECTORY);
        fs::create_dir(&output).unwrap();
        fs::create_dir(&staging_dir).unwrap();
        let wav_path = output.join(WAV_FILE);
        let frames_path = output.join(FRAMES_DIRECTORY);

        let error = execute_decode_branches(
            || {
                let staging = ArtifactStaging::new_for_path_producer_with_staging_parent(
                    &wav_path,
                    &staging_dir,
                )?;
                staging.write_bytes(b"staged audio")?;
                Ok((
                    PreparedAudioDecode::Staged(Box::new(staging)),
                    DecodeBranchAction::Decoded,
                ))
            },
            || -> Result<(PreparedVideoDecode, DecodeBranchAction)> {
                bail!("synthetic video decode failure")
            },
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("synthetic video decode failure"));
        assert!(!wav_path.exists());
        assert!(!frames_path.exists());
        assert!(!output.join(GENERATION_COMPLETE_FILE).exists());
    }

    #[cfg(unix)]
    #[test]
    fn metadata_checks_propagate_io_errors_instead_of_treating_them_as_absence() {
        let temporary = tempfile::tempdir().unwrap();
        let non_directory = temporary.path().join("not-a-directory");
        fs::write(&non_directory, b"x").unwrap();
        let inaccessible_child = non_directory.join("child");
        let error = symlink_metadata_if_exists(&inaccessible_child, "test artifact").unwrap_err();
        assert!(format!("{error:#}").contains("failed to inspect test artifact"));

        let error = require_indexed_transformer(&non_directory).unwrap_err();
        assert!(format!("{error:#}").contains("failed to inspect transformer weight index"));
    }

    #[test]
    fn generation_request_is_canonical_and_names_the_first_resume_mismatch() {
        let recorded = request(&policy());
        let bytes = recorded.canonical_json().unwrap();
        assert_eq!(
            GenerationRequest::from_canonical_json(&bytes).unwrap(),
            recorded
        );
        let mut changed = recorded.clone();
        changed.seed += 1;
        assert_eq!(recorded.first_difference(&changed), Some("seed"));
        let pretty = serde_json::to_vec_pretty(&recorded).unwrap();
        assert!(GenerationRequest::from_canonical_json(&pretty).is_err());
    }

    #[test]
    fn incomplete_initialization_is_completed_from_its_canonical_record() {
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("run");
        fs::create_dir(&output).unwrap();
        let policy = policy();
        let initialization = GenerationInitialization::new(
            request(&policy),
            policy.clone(),
            qwen_contract(&policy, 2),
        )
        .unwrap();
        let initialization_path = output.join(GENERATION_INITIALIZATION_FILE);
        fs::write(
            &initialization_path,
            initialization.canonical_json().unwrap(),
        )
        .unwrap();
        let checkpoint_dir = output.join(CHECKPOINT_DIRECTORY);
        let staging_dir = output.join(STAGING_DIRECTORY);
        let policy_path = output.join(EXECUTION_POLICY_FILE);
        let request_path = output.join(GENERATION_REQUEST_FILE);
        let ready_path = output.join(GENERATION_READY_FILE);

        validate_generation_directory_entries(&output).unwrap();
        validate_incomplete_initialization(&output, &checkpoint_dir).unwrap();
        complete_generation_initialization(
            &checkpoint_dir,
            &staging_dir,
            &policy_path,
            &request_path,
            &ready_path,
            &initialization,
        )
        .unwrap();

        validate_ready_marker(&ready_path).unwrap();
        assert!(checkpoint_dir.is_dir());
        assert_eq!(
            load_canonical_execution_policy(&policy_path).unwrap(),
            policy
        );
        assert_eq!(
            load_generation_request(&request_path).unwrap(),
            initialization.request
        );
    }

    #[test]
    fn empty_claimed_output_directory_is_the_only_state_without_an_initialization_record() {
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("run");
        fs::create_dir(&output).unwrap();
        let checkpoint_dir = output.join(CHECKPOINT_DIRECTORY);
        validate_generation_directory_entries(&output).unwrap();
        validate_incomplete_initialization(&output, &checkpoint_dir).unwrap();
        validate_empty_uninitialized_directory(&output).unwrap();

        let policy = policy();
        let initialization = GenerationInitialization::new(
            request(&policy),
            policy.clone(),
            qwen_contract(&policy, 2),
        )
        .unwrap();
        let initialization_path = output.join(GENERATION_INITIALIZATION_FILE);
        publish_new_bytes_with_staging_parent(
            &initialization_path,
            &initialization.canonical_json().unwrap(),
            "generation initialization",
            temporary.path(),
        )
        .unwrap();
        complete_generation_initialization(
            &checkpoint_dir,
            &output.join(STAGING_DIRECTORY),
            &output.join(EXECUTION_POLICY_FILE),
            &output.join(GENERATION_REQUEST_FILE),
            &output.join(GENERATION_READY_FILE),
            &initialization,
        )
        .unwrap();
        assert_eq!(
            load_generation_initialization(&initialization_path).unwrap(),
            initialization
        );
        validate_ready_marker(&output.join(GENERATION_READY_FILE)).unwrap();

        let ambiguous = temporary.path().join("ambiguous");
        fs::create_dir(&ambiguous).unwrap();
        fs::write(ambiguous.join(EXECUTION_POLICY_FILE), b"partial").unwrap();
        validate_generation_directory_entries(&ambiguous).unwrap();
        assert!(
            validate_empty_uninitialized_directory(&ambiguous)
                .unwrap_err()
                .to_string()
                .contains("partial state")
        );
    }

    #[cfg(unix)]
    #[test]
    fn generation_staging_symlink_is_rejected_before_checkpoint_publication() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("run");
        let outside = temporary.path().join("model-weights");
        fs::create_dir(&output).unwrap();
        fs::create_dir(&outside).unwrap();
        let staging = output.join(STAGING_DIRECTORY);
        symlink(&outside, &staging).unwrap();
        let error = validate_staging_directory(&staging).unwrap_err();
        assert!(error.to_string().contains("non-symlink directory"));
        assert_eq!(fs::read_dir(outside).unwrap().count(), 0);
    }

    #[test]
    fn stale_private_staging_residue_does_not_hide_the_latest_complete_checkpoint() {
        let temporary = tempfile::tempdir().unwrap();
        let checkpoints = temporary.path().join(CHECKPOINT_DIRECTORY);
        let staging = temporary.path().join(STAGING_DIRECTORY);
        fs::create_dir(&checkpoints).unwrap();
        fs::create_dir(&staging).unwrap();
        fs::create_dir(staging.join(".ff-stage-dir-crash-residue")).unwrap();
        publish_checkpoint_file(&checkpoints, 1);
        publish_checkpoint_file(&checkpoints, 2);

        let latest = discover_latest_checkpoint(&checkpoints, 4)
            .unwrap()
            .unwrap();
        assert_eq!(latest.identity.completed_evaluations, 2);
    }

    #[test]
    fn corrupt_or_misnamed_latest_checkpoint_is_not_silently_skipped() {
        let temporary = tempfile::tempdir().unwrap();
        let checkpoints = temporary.path().join(CHECKPOINT_DIRECTORY);
        fs::create_dir(&checkpoints).unwrap();
        let first = publish_checkpoint_file(&checkpoints, 1);
        let second = checkpoints.join("checkpoint-step000002.safetensors");
        fs::write(&second, b"corrupt latest checkpoint").unwrap();
        let error = discover_latest_checkpoint(&checkpoints, 4).unwrap_err();
        assert!(format!("{error:#}").contains("checkpoint-step000002"));

        fs::remove_file(&second).unwrap();
        fs::copy(first, &second).unwrap();
        let error = discover_latest_checkpoint(&checkpoints, 4).unwrap_err();
        assert!(error.to_string().contains("filename step 2 disagrees"));
    }

    #[test]
    fn corruption_anywhere_in_the_checkpoint_chain_is_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let checkpoints = temporary.path().join(CHECKPOINT_DIRECTORY);
        fs::create_dir(&checkpoints).unwrap();
        let first = publish_checkpoint_file(&checkpoints, 1);
        publish_checkpoint_file(&checkpoints, 2);
        fs::write(first, b"corrupt earlier checkpoint").unwrap();

        let error = discover_latest_checkpoint(&checkpoints, 4).unwrap_err();
        assert!(format!("{error:#}").contains("checkpoint-step000001"));
    }

    #[test]
    fn final_and_per_evaluation_publications_of_the_same_state_are_byte_identical() {
        let temporary = tempfile::tempdir().unwrap();
        let checkpoint_path = temporary.path().join("checkpoint-step000003.safetensors");
        let final_path = temporary.path().join(FINAL_LATENTS_FILE);
        publish_checkpoint_at(&checkpoint_path, 3);
        publish_checkpoint_at(&final_path, 3);
        assert_eq!(
            fs::read(checkpoint_path).unwrap(),
            fs::read(final_path).unwrap()
        );
    }

    #[test]
    fn probed_budget_honours_an_explicit_host_bound_and_the_tighter_device_bound() {
        let snapshot = ResourceSnapshot {
            schema_version: flyingfish::runtime::probe::RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(10 * 1024 * 1024),
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: Some(8 * 1024 * 1024),
            device_free_memory_bytes: Some(6 * 1024 * 1024),
            host_device_memory_is_unified: None,
            device_topology_probe_failed: false,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            measurement_scope: flyingfish::runtime::probe::ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        };
        let budget =
            probed_budget(&snapshot, ExecutionBackendPolicy::Cuda, Some(9), Some(5)).unwrap();
        assert_eq!(budget.max_host_bytes, Some(9 * 1024 * 1024));
        assert_eq!(budget.max_device_bytes, Some(5 * 1024 * 1024));

        let probed = probed_budget(&snapshot, ExecutionBackendPolicy::Cuda, None, Some(5)).unwrap();
        assert_eq!(probed.max_host_bytes, Some(8 * 1024 * 1024));

        let device = probed_budget(&snapshot, ExecutionBackendPolicy::Cuda, None, Some(9)).unwrap();
        assert_eq!(device.max_device_bytes, Some(6 * 1024 * 1024));

        let snapshot = ResourceSnapshot {
            host_memory_available_bytes: Some(2 * 1024 * 1024),
            ..snapshot
        };
        let budget = probed_budget(&snapshot, ExecutionBackendPolicy::Cpu, None, None).unwrap();
        assert_eq!(budget.max_host_bytes, Some(2 * 1024 * 1024));
    }

    #[test]
    fn probed_budget_fails_when_a_required_measurement_and_bound_are_both_missing() {
        let snapshot = ResourceSnapshot {
            schema_version: flyingfish::runtime::probe::RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: None,
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: None,
            device_free_memory_bytes: None,
            host_device_memory_is_unified: None,
            device_topology_probe_failed: false,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            measurement_scope: flyingfish::runtime::probe::ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        };
        assert!(
            probed_budget(&snapshot, ExecutionBackendPolicy::Cpu, None, None)
                .unwrap_err()
                .to_string()
                .contains("cannot measure available host memory")
        );
        assert!(
            probed_budget(&snapshot, ExecutionBackendPolicy::Cuda, Some(1), None)
                .unwrap_err()
                .to_string()
                .contains("cannot measure free device memory")
        );
    }

    #[test]
    fn probed_budget_clamps_the_host_bound_to_a_probed_unified_pool() {
        let snapshot = |unified: Option<bool>| ResourceSnapshot {
            schema_version: flyingfish::runtime::probe::RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(10 * 1024 * 1024),
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: None,
            device_free_memory_bytes: Some(6 * 1024 * 1024),
            host_device_memory_is_unified: unified,
            device_topology_probe_failed: false,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            measurement_scope: flyingfish::runtime::probe::ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        };
        for legacy in [None, Some(false)] {
            let budget =
                probed_budget(&snapshot(legacy), ExecutionBackendPolicy::Cuda, None, None).unwrap();
            assert_eq!(budget.max_host_bytes, Some(10 * 1024 * 1024));
        }
        let folded = probed_budget(
            &snapshot(Some(true)),
            ExecutionBackendPolicy::Cuda,
            None,
            None,
        )
        .unwrap();
        assert_eq!(folded.max_host_bytes, Some(6 * 1024 * 1024));
        let requested = probed_budget(
            &snapshot(Some(true)),
            ExecutionBackendPolicy::Cuda,
            Some(9),
            None,
        )
        .unwrap();
        assert_eq!(requested.max_host_bytes, Some(6 * 1024 * 1024));
        let unmeasurable = ResourceSnapshot {
            device_free_memory_bytes: None,
            ..snapshot(Some(true))
        };
        assert!(
            probed_budget(
                &unmeasurable,
                ExecutionBackendPolicy::Cuda,
                Some(9),
                Some(9)
            )
            .unwrap_err()
            .to_string()
            .contains("shared host/device pool size")
        );
    }

    #[test]
    fn admission_refuses_before_parsing_or_materializing_an_invalid_weight_payload() {
        let temporary = tempfile::tempdir().unwrap();
        let transformer = temporary.path().join("transformer");
        fs::create_dir(&transformer).unwrap();
        fs::write(
            transformer.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "_class_name": "MiniMaxH3Transformer3DModel",
                "num_attention_heads": 1,
                "attention_head_dim": 6,
                "hidden_size": 4,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "ffn_dim": 5,
                "in_channels": 1,
                "audio_in_channels": 1,
                "patch_size": [1, 1, 1],
                "text_dim": 2,
                "freq_dim": 2,
                "time_embed_hidden_dim": 2,
                "time_embed_dim": 2,
                "rope_freq_dim": 1,
                "rope_theta": 10000.0,
                "norm_eps": 1e-5,
                "qk_norm_eps": 1e-5,
                "final_norm_eps": 1e-5
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(transformer.join("invalid.safetensors"), b"not safetensors").unwrap();
        fs::write(
            transformer.join("model.safetensors.index.json"),
            br#"{"metadata":{"total_size":1},"weight_map":{"proj_in.weight":"invalid.safetensors"}}"#,
        )
        .unwrap();
        let config = TransformerConfig::from_file(transformer.join("config.json")).unwrap();
        let error = preflight_generation(
            &transformer,
            &config,
            &policy(),
            T2vaGeometry {
                text_rows: 1,
                latent_frames: 1,
                latent_height: 1,
                latent_width: 1,
                audio_frames: 1,
                audio_channels: 1,
                attention_projection_chunk_size: 1,
                attention_query_chunk_size: 1,
                attention_key_chunk_policy: AttentionKeyChunkPolicy::Full,
                ffn_token_chunk_size: 1,
                output_token_chunk_size: 1,
            },
            1,
            Some(0),
            None,
            Some(0),
            &Device::Cpu,
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("binding budget: host"), "{message}");
        assert!(!message.contains("invalid safetensors"), "{message}");
    }

    #[test]
    fn final_latents_must_have_the_same_identity_as_the_last_evaluation_checkpoint() {
        let policy = policy();
        let mut history = PolicyHistory::new();
        history
            .append_successful_evaluations(2, policy.clone())
            .unwrap();
        let latest = checkpoint("checkpoint-step000002.safetensors", history.clone());
        let mut final_checkpoint = checkpoint("denoised-latents.safetensors", history);
        final_checkpoint.identity.checkpoint_bytes += 1;

        let error =
            validate_checkpoint_chain(Some(&latest), Some(&final_checkpoint), Some(&policy), 3)
                .unwrap_err();
        assert!(error.to_string().contains("identity disagrees"));
    }

    #[test]
    fn every_checkpoint_policy_segment_must_match_the_recorded_policy() {
        let recorded = policy();
        let mut earlier = recorded.clone();
        earlier.configured_output_rows = 17;
        earlier.validate().unwrap();
        let mut history = PolicyHistory::new();
        history.append_successful_evaluations(1, earlier).unwrap();
        history
            .append_successful_evaluations(2, recorded.clone())
            .unwrap();
        let latest = checkpoint("checkpoint-step000002.safetensors", history);

        let error = validate_checkpoint_chain(Some(&latest), None, Some(&recorded), 3).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("evaluations [0, 1) use a different policy")
        );
    }
}
