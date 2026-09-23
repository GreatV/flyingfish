#[derive(Debug, Subcommand)]
pub(in crate::cli) enum H3Command {
    #[command(
        name = "init",
        about = "Convert raw T2VA inputs into a canonical step-zero recovery checkpoint"
    )]
    InitializeT2vaCheckpoint {
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = NonZeroUsize::new(50).unwrap())]
        sigma_points: NonZeroUsize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
    },
    #[command(
        name = "plan",
        about = "Estimate T2VA memory and enforce optional bounds on the modeled peaks"
    )]
    PlanT2va {
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "transformer")]
        component: PathBuf,
        #[arg(long)]
        text_rows: usize,
        #[arg(long, default_value_t = 37, conflicts_with = "short_edge")]
        latent_frames: usize,
        #[arg(long, default_value_t = 48, conflicts_with = "short_edge")]
        latent_height: usize,
        #[arg(long, default_value_t = 84, conflicts_with = "short_edge")]
        latent_width: usize,
        #[arg(long, default_value_t = 207, conflicts_with = "short_edge")]
        audio_frames: usize,
        #[command(flatten)]
        target: TargetGeometryArgs,
        #[arg(long, default_value_t = 2)]
        audio_channels: usize,
        #[command(flatten)]
        chunks: DenoiseChunkArgs,
        #[arg(long, default_value_t = 50)]
        sigma_points: usize,
        #[arg(long, default_value_t = 0)]
        start_step: usize,
        #[arg(long)]
        max_steps: Option<usize>,
        #[arg(help = "Estimate CPU F32 execution instead of checkpoint-precision GPU execution")]
        #[arg(long)]
        cpu: bool,
        #[arg(long)]
        no_precompute_adaln: bool,
        #[arg(help = "Estimate optional CUDA FlashAttention working memory")]
        #[arg(long)]
        flash_attention: bool,
        #[arg(
            long,
            help = "Logical host-cache byte ceiling in MiB; replaces the one-shard bound"
        )]
        host_cache_mib: Option<u64>,
        #[arg(help = "Backend allocator/cuBLAS allowance; defaults to zero on CPU")]
        #[arg(long)]
        backend_workspace_mib: Option<u64>,
        #[arg(help = "Hard bound on modeled host memory, including conservative mmap residency")]
        #[arg(long)]
        max_host_mib: Option<u64>,
        #[arg(help = "Hard bound on modeled device memory. Omit for no device limit")]
        #[arg(long)]
        max_device_mib: Option<u64>,
        #[arg(help = "Print the H3 layer-scoped device-residency plan")]
        #[arg(long)]
        execution_plan: bool,
        #[arg(long)]
        json: bool,
    },
    #[command(
        name = "solve",
        about = "Enumerate a finite grid of conservative full-softmax T2VA policies"
    )]
    SolveT2va {
        #[command(flatten)]
        resources: H3ResourceArgs,
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "transformer")]
        component: PathBuf,
        #[arg(long)]
        text_rows: usize,
        #[arg(long, default_value_t = 37, conflicts_with = "short_edge")]
        latent_frames: usize,
        #[arg(long, default_value_t = 48, conflicts_with = "short_edge")]
        latent_height: usize,
        #[arg(long, default_value_t = 84, conflicts_with = "short_edge")]
        latent_width: usize,
        #[arg(long, default_value_t = 207, conflicts_with = "short_edge")]
        audio_frames: usize,
        #[command(flatten)]
        target: TargetGeometryArgs,
        #[arg(long, default_value_t = 2)]
        audio_channels: usize,
        #[arg(long, default_value_t = 50)]
        sigma_points: usize,
        #[arg(long, default_value_t = 0)]
        start_step: usize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
        #[arg(long)]
        max_steps: Option<usize>,
        #[arg(help = "Estimate CPU F32 execution instead of checkpoint-precision GPU execution")]
        #[arg(long)]
        cpu: bool,
        #[arg(
            long,
            help = "Live device for evidence-qualified weight selection; omitted for offline numerical planning"
        )]
        device: Option<String>,
        #[arg(
            long,
            help = "Logical host-cache byte ceiling in MiB; replaces the one-shard bound"
        )]
        host_cache_mib: Option<u64>,
        #[arg(
            help = "Backend allocator safety allowance added to device peak; defaults to zero on CPU"
        )]
        #[arg(long)]
        backend_workspace_mib: Option<u64>,
        #[arg(help = "Hard bound on modeled host memory, including conservative mmap residency")]
        #[arg(long)]
        max_host_mib: Option<u64>,
        #[arg(help = "Hard bound on modeled device memory. Omit for no device limit")]
        #[arg(long)]
        max_device_mib: Option<u64>,
        #[arg(help = "Maximum number of feasible policies to print; does not limit the search")]
        #[arg(long, default_value_t = NonZeroUsize::new(25).unwrap())]
        limit: NonZeroUsize,
        #[arg(long)]
        json: bool,
    },
    #[command(
        name = "history",
        about = "Extract and verify the policy history embedded in a denoise checkpoint"
    )]
    ShowPolicyHistory {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(help = "Write canonical JSON to a new file instead of stdout")]
        #[arg(long)]
        output: Option<PathBuf>,
    },
    #[command(
        name = "calibrate",
        about = "Measure prevalidated pinned T2VA policies in isolated child processes"
    )]
    CalibrateT2va {
        #[arg(help = "MiniMax-H3 repository directory")]
        #[arg(long)]
        model: PathBuf,
        #[arg(help = "Transformer component path relative to the repository")]
        #[arg(hide = true, long, default_value = "transformer")]
        component: PathBuf,
        #[arg(help = "Initial, step-zero T2VA safetensors")]
        #[arg(long)]
        inputs: PathBuf,
        #[arg(help = "Budgeted full-softmax policy JSON; repeat once per candidate")]
        #[arg(long, required = true)]
        policy: Vec<PathBuf>,
        #[arg(help = "New path receiving the versioned calibration report")]
        #[arg(long)]
        output: PathBuf,
        #[arg(help = "Explicit child execution device: cpu, cuda:N, or metal:N")]
        #[arg(long)]
        device: String,
        #[arg(
            help = "Leading evaluations that advance the trajectory but are excluded from statistics"
        )]
        #[arg(long, default_value_t = NonZeroUsize::new(1).unwrap())]
        warmup_prefix_evaluations: NonZeroUsize,
        #[arg(help = "Evaluations measured after the warmup prefix")]
        #[arg(long, default_value_t = NonZeroUsize::new(1).unwrap())]
        measured_evaluations: NonZeroUsize,
        #[arg(help = "Isolated child trials per candidate policy")]
        #[arg(long, default_value_t = NonZeroUsize::new(3).unwrap())]
        trials: NonZeroUsize,
        #[arg(help = "Sigma grid points including terminal zero")]
        #[arg(long, default_value_t = NonZeroUsize::new(50).unwrap())]
        sigma_points: NonZeroUsize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
    },
    #[command(
        name = "denoise",
        about = "Run T2VA denoising from precomputed prompt embeddings and initial noise"
    )]
    DenoiseT2va {
        #[command(flatten)]
        admission: H3AdmissionArgs,
        #[command(flatten)]
        resources: H3ResourceArgs,
        #[arg(help = "MiniMax-H3 repository directory")]
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "transformer")]
        component: PathBuf,
        #[arg(
            help = "Safetensors containing prompt_embeddings, text_token_tags, video_latents, and audio_latents"
        )]
        #[arg(long)]
        inputs: PathBuf,
        #[arg(
            long,
            help = "Explicit schema-2 official fixture manifest for externally validated prompt embeddings"
        )]
        external_prompt_manifest: Option<PathBuf>,
        #[arg(help = "Output safetensors path for video_latents and audio_latents")]
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(
            help = "Replay a pinned execution policy; resource-policy flags may not be mixed with it"
        )]
        #[arg(
            long,
            conflicts_with_all = [
                "weight_source",
                "host_cache_mib",
                "host_cache_granularity",
                "attention_query_chunk_size",
                "attention_projection_chunk_size",
                "attention_key_chunk_size",
                "ffn_token_chunk_size",
                "output_token_chunk_size",
                "no_precompute_adaln",
                "flash_attention"
            ]
        )]
        policy: Option<PathBuf>,
        #[command(flatten)]
        weights: OptionalWeightCacheArgs,
        #[command(flatten)]
        chunks: DenoiseChunkArgs,
        #[arg(help = "Disable host-side precomputation of the complete AdaLN schedule")]
        #[arg(long)]
        no_precompute_adaln: bool,
        #[arg(help = "Use optional CUDA FlashAttention (requires the flash-attn build feature)")]
        #[arg(long)]
        flash_attention: bool,
        #[arg(help = "Suppress one-line timing output after each denoising evaluation")]
        #[arg(long)]
        no_progress: bool,
        #[arg(help = "Write sampled RSS/CUDA peaks as JSON after a successful run")]
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
        #[arg(
            help = "Publish a complete recovery checkpoint after every evaluation into a new directory"
        )]
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        #[arg(help = "Stop after this many evaluations and save a resumable checkpoint")]
        #[arg(long)]
        max_steps: Option<NonZeroUsize>,
        #[arg(help = "Sigma grid points including terminal zero; model evaluations are one fewer")]
        #[arg(long, default_value_t = 50)]
        sigma_points: usize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
    },
    #[command(
        name = "encode",
        about = "Encode a text-only H3 prompt with streamed Qwen3-VL layers 0 through 49"
    )]
    EncodePrompt {
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "text_encoder")]
        component: PathBuf,
        #[arg(long, default_value = "tokenizer/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(long, default_value_t = DEFAULT_ATTENTION_QUERY_CHUNK_SIZE)]
        attention_query_chunk_size: usize,
        #[arg(help = "H3 consumes `hidden_states[50]`, before the encoder's final norm")]
        #[arg(long, default_value_t = 50)]
        target_hidden_state: usize,
    },
    #[command(
        name = "prepare",
        about = "Combine a prompt encoding with reproducible initial video/audio noise"
    )]
    PrepareT2vaInputs {
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "transformer")]
        component: PathBuf,
        #[arg(long)]
        prompt_encoding: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(
            long,
            required_unless_present = "short_edge",
            conflicts_with = "short_edge"
        )]
        latent_frames: Option<usize>,
        #[arg(
            long,
            required_unless_present = "short_edge",
            conflicts_with = "short_edge"
        )]
        latent_height: Option<usize>,
        #[arg(
            long,
            required_unless_present = "short_edge",
            conflicts_with = "short_edge"
        )]
        latent_width: Option<usize>,
        #[arg(
            long,
            required_unless_present = "short_edge",
            conflicts_with = "short_edge"
        )]
        audio_frames: Option<usize>,
        #[command(flatten)]
        target: TargetGeometryArgs,
        #[arg(long, default_value_t = 2)]
        audio_channels: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    #[command(
        name = "generate",
        about = "Start or resume admitted prompt encoding, checkpointed denoising, and WAV/PNG decoding"
    )]
    GenerateT2va {
        #[command(flatten)]
        resources: H3ResourceArgs,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(help = "New run directory, or an existing initialized run to resume by default")]
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(
            help = "Replay a pinned execution policy; resource-policy flags may not be mixed with it"
        )]
        #[arg(
            long,
            conflicts_with_all = [
                "weight_source",
                "host_cache_mib",
                "host_cache_granularity",
                "attention_query_chunk_size",
                "attention_projection_chunk_size",
                "attention_key_chunk_size",
                "ffn_token_chunk_size",
                "output_token_chunk_size",
                "no_precompute_adaln",
                "flash_attention"
            ]
        )]
        policy: Option<PathBuf>,
        #[command(flatten)]
        weights: OptionalWeightCacheArgs,
        #[arg(long, default_value_t = 37, conflicts_with = "short_edge")]
        latent_frames: usize,
        #[arg(long, default_value_t = 48, conflicts_with = "short_edge")]
        latent_height: usize,
        #[arg(long, default_value_t = 84, conflicts_with = "short_edge")]
        latent_width: usize,
        #[arg(long, default_value_t = 207, conflicts_with = "short_edge")]
        audio_frames: usize,
        #[command(flatten)]
        target: TargetGeometryArgs,
        #[arg(long, default_value_t = 2)]
        audio_channels: usize,
        #[arg(help = "WAV encoding: pcm16 for playback compatibility or float32 for parity")]
        #[arg(long, value_enum, default_value = "pcm16")]
        wav_format: WavSampleFormat,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(help = "H3 consumes `hidden_states[50]` by default")]
        #[arg(long, default_value_t = 50)]
        target_hidden_state: usize,
        #[command(flatten)]
        chunks: DenoiseChunkArgs,
        #[arg(long, default_value_t = 50)]
        sigma_points: usize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
        #[arg(long)]
        no_precompute_adaln: bool,
        #[arg(help = "Use optional CUDA FlashAttention (requires the flash-attn build feature)")]
        #[arg(long)]
        flash_attention: bool,
        #[arg(long)]
        no_progress: bool,
        #[arg(help = "Print the topology-derived configuration and its provenance to stderr")]
        #[arg(long)]
        explain_config: bool,
        #[arg(help = "Write sampled RSS/CUDA peaks as JSON after a successful run")]
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
        #[arg(
            long,
            help = "Hard host-memory admission bound in MiB; defaults to probed availability"
        )]
        max_host_mib: Option<u64>,
        #[arg(
            long,
            help = "Hard device-memory admission bound in MiB; defaults to probed CUDA/Metal availability"
        )]
        max_device_mib: Option<u64>,
        #[arg(
            long,
            help = "Backend allocator safety allowance in MiB used by preflight admission"
        )]
        backend_workspace_mib: Option<u64>,
    },
    #[command(
        name = "prepare-fl2va",
        about = "Prepare a self-contained resumable H3 first/last-frame conditioning bundle"
    )]
    PrepareFl2va {
        #[arg(long)]
        model: PathBuf,
        #[arg(
            long,
            required_unless_present = "prompt_file",
            conflicts_with = "prompt_file"
        )]
        prompt: Option<String>,
        #[arg(long, required_unless_present = "prompt", conflicts_with = "prompt")]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        image: Option<PathBuf>,
        #[arg(long)]
        last_image: Option<PathBuf>,
        #[arg(long, conflicts_with = "short_edge")]
        height: Option<usize>,
        #[arg(long, conflicts_with = "short_edge")]
        width: Option<usize>,
        #[arg(long, default_value_t = 240, conflicts_with = "short_edge")]
        num_frames: usize,
        #[command(flatten)]
        target: TargetGeometryArgs,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 50)]
        sigma_points: usize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "auto")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(long, default_value_t = DEFAULT_ATTENTION_QUERY_CHUNK_SIZE)]
        attention_query_chunk_size: usize,
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
    },
    #[command(
        name = "prepare-ref2va",
        about = "Prepare a self-contained resumable H3 omni-reference conditioning bundle"
    )]
    PrepareRef2va {
        #[arg(long)]
        model: PathBuf,
        #[arg(
            long,
            required_unless_present = "prompt_file",
            conflicts_with = "prompt_file"
        )]
        prompt: Option<String>,
        #[arg(long, required_unless_present = "prompt", conflicts_with = "prompt")]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        references_json: PathBuf,
        #[arg(
            long,
            required_unless_present = "short_edge",
            conflicts_with = "short_edge"
        )]
        height: Option<usize>,
        #[arg(
            long,
            required_unless_present = "short_edge",
            conflicts_with = "short_edge"
        )]
        width: Option<usize>,
        #[arg(long, default_value_t = 240, conflicts_with = "short_edge")]
        num_frames: usize,
        #[command(flatten)]
        target: TargetGeometryArgs,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 50)]
        sigma_points: usize,
        #[arg(long, default_value_t = 12.0)]
        video_shift: f32,
        #[arg(long, default_value_t = 3.0)]
        audio_shift: f32,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "auto")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(long, default_value_t = DEFAULT_ATTENTION_QUERY_CHUNK_SIZE)]
        attention_query_chunk_size: usize,
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
    },
    #[command(
        name = "denoise-conditioned",
        about = "Denoise or resume a prepared FL2VA/Ref2VA conditioning bundle"
    )]
    DenoiseConditioned {
        #[command(flatten)]
        admission: H3AdmissionArgs,
        #[command(flatten)]
        resources: H3ResourceArgs,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(
            long,
            conflicts_with_all = [
                "weight_source",
                "host_cache_mib",
                "host_cache_granularity",
                "attention_query_chunk_size",
                "attention_projection_chunk_size",
                "attention_key_chunk_size",
                "ffn_token_chunk_size",
                "output_token_chunk_size",
                "no_precompute_adaln",
                "flash_attention"
            ]
        )]
        policy: Option<PathBuf>,
        #[command(flatten)]
        weights: OptionalWeightCacheArgs,
        #[command(flatten)]
        chunks: DenoiseChunkArgs,
        #[arg(long)]
        no_precompute_adaln: bool,
        #[arg(long)]
        flash_attention: bool,
        #[arg(long)]
        no_progress: bool,
        #[arg(long)]
        max_steps: Option<NonZeroUsize>,
        #[arg(
            long,
            help = "Publish a complete conditioned bundle after every evaluation into a new directory"
        )]
        checkpoint_dir: Option<PathBuf>,
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
    },
    #[command(subcommand)]
    #[command(about = "Decode H3 audio or video latents")]
    Decode(H3DecodeCommand),
}

#[derive(Debug, Subcommand)]
pub(in crate::cli) enum H3DecodeCommand {
    #[command(about = "Decode normalized H3 audio latents into a 32 kHz WAV file")]
    Audio {
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "audio_vae")]
        component: PathBuf,
        #[arg(help = "Safetensors containing `audio_latents`")]
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(help = "WAV encoding: pcm16 for playback compatibility or float32 for parity")]
        #[arg(long, value_enum, default_value = "pcm16")]
        wav_format: WavSampleFormat,
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
    },
    #[command(about = "Decode normalized H3 video latents into a numbered RGB PNG sequence")]
    Video {
        #[arg(long)]
        model: PathBuf,
        #[arg(hide = true, long, default_value = "vae")]
        component: PathBuf,
        #[arg(help = "Safetensors containing `video_latents`")]
        #[arg(long)]
        inputs: PathBuf,
        #[arg(help = "New directory that will receive frame_00000.png, etc")]
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(long, default_value_t = DEFAULT_ATTENTION_QUERY_CHUNK_SIZE)]
        attention_query_chunk_size: usize,
        #[arg(long)]
        telemetry_json: Option<PathBuf>,
    },
}

impl H3Command {
    pub(in crate::cli) fn run(self) -> Result<()> {
        let command = self;
        match command {
            command @ H3Command::InitializeT2vaCheckpoint { .. } => {
                initialize::run_initialize_t2va_checkpoint(command)
            }
            command @ H3Command::PlanT2va { .. } => plan::run_plan_t2va(command),
            command @ H3Command::SolveT2va { .. } => plan::run_solve_t2va(command),
            command @ H3Command::ShowPolicyHistory { .. } => {
                denoise::run_show_policy_history(command)
            }
            command @ H3Command::CalibrateT2va { .. } => calibrate::run_calibrate_t2va(command),
            command @ H3Command::DenoiseT2va { .. } => denoise::run_denoise_t2va(command),
            command @ H3Command::EncodePrompt { .. } => prompt::run_encode_prompt(command),
            command @ H3Command::PrepareT2vaInputs { .. } => {
                prompt::run_prepare_t2va_inputs(command)
            }
            command @ H3Command::GenerateT2va { .. } => generate::run_generate_t2va(command),
            command @ H3Command::PrepareFl2va { .. } => conditioned::run_prepare_fl2va(command),
            command @ H3Command::PrepareRef2va { .. } => conditioned::run_prepare_ref2va(command),
            command @ H3Command::DenoiseConditioned { .. } => {
                conditioned::run_denoise_conditioned(command)
            }
            H3Command::Decode(command) => match command {
                command @ H3DecodeCommand::Audio { .. } => decode::run_decode_audio(command),
                command @ H3DecodeCommand::Video { .. } => decode::run_decode_video(command),
            },
        }
    }
}

use super::{Adapter, Task};
use crate::cli::{
    DenoiseChunkArgs, H3AdmissionArgs, H3ResourceArgs, OptionalWeightCacheArgs, TargetGeometryArgs,
    WeightCacheArgs,
};
use crate::cli::{calibrate, conditioned, decode, denoise, generate, initialize, plan, prompt};
use anyhow::Result;
use clap::{FromArgMatches, Subcommand};
use flyingfish::h3::audio_vae::WavSampleFormat;
use flyingfish::h3::core::DEFAULT_ATTENTION_QUERY_CHUNK_SIZE;
use std::num::NonZeroUsize;
use std::path::PathBuf;

pub(super) const ADAPTER: Adapter = Adapter {
    id: "h3",
    task: Task::Video,
    recognizes: |metadata| {
        metadata.architecture("MiniMaxH3ModularPipeline")
            || metadata.architecture("MiniMaxH3Transformer3DModel")
    },
    command: || H3Command::augment_subcommands(clap::Command::new(Task::Video.name())),
    run: |matches| H3Command::from_arg_matches(matches)?.run(),
};
