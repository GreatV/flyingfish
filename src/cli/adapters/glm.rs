#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
pub(in crate::cli) enum GlmCommand {
    #[command(
        name = "generate-multi",
        about = "Generate GLM requests across reusable CUDA layer partitions"
    )]
    GenerateMulti(glm_multi::Args),
    #[command(
        name = "generate",
        about = "Generate text with the short-context disk-streamed GLM-5.3-Flash profile"
    )]
    Generate {
        #[arg(long, help = "GLM-5.3-Flash checkpoint directory")]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = NonZeroUsize::new(16).unwrap())]
        max_new_tokens: NonZeroUsize,
        #[arg(
            long,
            default_value_t = NonZeroUsize::new(2048).unwrap(),
            help = "Hard request limit; the exact short-context profile supports at most index_topk"
        )]
        max_context_tokens: NonZeroUsize,
        #[arg(long, default_value = "max", value_parser = ["low", "high", "max"])]
        reasoning_effort: String,
        #[command(flatten)]
        sampling: kit::SamplingArgs,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[command(flatten)]
        weights: OptionalWeightCacheArgs,
        #[arg(long, value_enum, default_value_t = flyingfish::runtime::resource_selection::ResourcePolicyMode::Performance)]
        resource_policy: flyingfish::runtime::resource_selection::ResourcePolicyMode,
        #[arg(long, help = "Qualified paired resource evidence")]
        resource_evidence: Option<PathBuf>,
        #[arg(
            long,
            help = "Linux benchmark only: evict and verify checkpoint file-cache pages before payload loading"
        )]
        resource_cold_cache: bool,
        #[arg(long, help = "Write resource selection to a new JSON sidecar")]
        resource_selection: Option<PathBuf>,
        #[arg(
            long, num_args = 0..=1, default_missing_value = "true", require_equals = true,
            conflicts_with = "no_resident_static",
            help = "Keep the non-routed text skeleton (about 15.3 GiB); absent selects by CUDA capacity in performance mode"
        )]
        resident_static: Option<bool>,
        #[arg(
            long,
            conflicts_with = "resident_static",
            help = "Explicitly keep the static skeleton streamed"
        )]
        no_resident_static: bool,
        #[arg(
            long,
            help = "Host profile measured on this machine by `ff bench io --profile \
                    local-interconnect`; reports the transfer-versus-host expert split"
        )]
        host_profile: Option<PathBuf>,
        #[arg(
            long,
            help = "Use the CPU FP8 conversion reference with CUDA (fallback and paired benchmarks)"
        )]
        cpu_fp8_dequantization: bool,
        #[arg(
            long,
            conflicts_with = "cpu_fp8_dequantization",
            help = "Use reusable pinned buffers for FP8 uploads (experimental)"
        )]
        pinned_fp8_transfer: bool,
        #[arg(
            long,
            help = "Expert-cache budget in MiB; absent selects by CUDA capacity in performance mode, zero disables"
        )]
        expert_cache_mib: Option<u64>,
        #[arg(long, value_enum)]
        expert_cache_layout: Option<ExpertCacheLayout>,
        #[arg(long, value_enum)]
        expert_cache_replacement: Option<ExpertCacheReplacementPolicy>,
        #[arg(
            long,
            requires = "execution_manifest",
            help = "Re-admit the expert-cache bound after every synchronized routed token"
        )]
        expert_cache_readmit: bool,
        #[arg(
            long,
            requires = "expert_cache_readmit",
            help = "Hard adaptive expert-cache floor in MiB; defaults to zero"
        )]
        expert_cache_min_mib: Option<u64>,
        #[arg(long, help = "Suppress per-token progress on stderr")]
        no_progress: bool,
        #[command(flatten)]
        output: kit::OutputArgs,
        #[arg(
            long,
            help = "Atomically write a bounded router-only trace outside the model"
        )]
        routing_trace: Option<PathBuf>,
        #[arg(
            long,
            requires = "routing_trace",
            help = "Bounded workload/domain label recorded in --routing-trace"
        )]
        routing_trace_domain: Option<String>,
        #[arg(
            long,
            help = "Atomically write the versioned GLM policy and cache-residency manifest"
        )]
        execution_manifest: Option<PathBuf>,
    },
    #[command(
        name = "capture-parity",
        about = "Export one bounded GLM prefill boundary for offline reference comparison"
    )]
    CaptureParity {
        #[arg(long, help = "GLM-5.3-Flash checkpoint directory")]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(
            long,
            default_value_t = NonZeroUsize::new(2048).unwrap(),
            help = "Hard prompt limit no larger than index_topk"
        )]
        max_context_tokens: NonZeroUsize,
        #[arg(long, default_value = "max", value_parser = ["low", "high", "max"])]
        reasoning_effort: String,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(
            long,
            help = "Keep the non-routed text skeleton on the selected device"
        )]
        resident_static: bool,
        #[arg(long, help = "Suppress layer-wise parity prefill progress")]
        no_progress: bool,
        #[arg(
            long,
            help = "New atomically published parity safetensors outside the model"
        )]
        output: PathBuf,
    },
    #[command(
        name = "replay-routing",
        about = "Analyze a bounded GLM routing trace with SRP/SCH and equal-byte cache replays"
    )]
    ReplayRouting {
        #[arg(long, help = "Versioned routing-trace JSON emitted by glm generate")]
        trace: PathBuf,
        #[arg(long, help = "New atomically published routing-replay JSON report")]
        output: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "4,16,64",
            help = "Comma-separated positive SRP/SCH segment lengths"
        )]
        segment_lengths: Vec<NonZeroUsize>,
        #[arg(
            long,
            required = true,
            value_delimiter = ',',
            help = "Comma-separated equal total cache budgets in MiB"
        )]
        cache_mib: Vec<NonZeroU64>,
    },
}

impl GlmCommand {
    pub(in crate::cli) fn run(self) -> Result<()> {
        let command = self;
        match command {
            command @ GlmCommand::Generate { .. } => glm::run_generate(command),
            GlmCommand::GenerateMulti(args) => glm_multi::run(args),
            command @ GlmCommand::CaptureParity { .. } => glm::run_capture_parity(command),
            command @ GlmCommand::ReplayRouting { .. } => glm::run_replay_routing(command),
        }
    }
}

use super::{Adapter, Task};
use crate::cli::glm_multi;
use crate::cli::{OptionalWeightCacheArgs, WeightCacheArgs, glm, kit};
use anyhow::Result;
use clap::{FromArgMatches, Subcommand};
use flyingfish::glm::{ExpertCacheLayout, ExpertCacheReplacementPolicy};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;

pub(super) const ADAPTER: Adapter = Adapter {
    id: "glm",
    task: Task::Text,
    recognizes: |metadata| metadata.architecture("Glm5NextForConditionalGeneration"),
    command: || GlmCommand::augment_subcommands(clap::Command::new(Task::Text.name())),
    run: |matches| GlmCommand::from_arg_matches(matches)?.run(),
};
