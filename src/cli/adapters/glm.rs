#[derive(Debug, Subcommand)]
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
    Generate(Box<glm::Args>),
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
            GlmCommand::Generate(args) => glm::run_generate(*args),
            GlmCommand::GenerateMulti(args) => glm_multi::run(args),
            command @ GlmCommand::CaptureParity { .. } => glm::run_capture_parity(command),
            command @ GlmCommand::ReplayRouting { .. } => glm::run_replay_routing(command),
        }
    }
}

use super::{Adapter, Task};
use crate::cli::glm_multi;
use crate::cli::{WeightCacheArgs, glm};
use anyhow::Result;
use clap::{FromArgMatches, Subcommand};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;

pub(super) const ADAPTER: Adapter = Adapter {
    id: "glm",
    task: Task::Text,
    recognizes: |metadata| metadata.architecture("Glm5NextForConditionalGeneration"),
    command: || GlmCommand::augment_subcommands(clap::Command::new(Task::Text.name())),
    run: |matches| GlmCommand::from_arg_matches(matches)?.run(),
};
