use crate::cli::trellis;

#[derive(Debug, Subcommand)]
pub(in crate::cli) enum TrellisCommand {
    #[command(about = "Generate TRELLIS Gaussian splats or a TRELLIS.2 direct-512 colored mesh")]
    Generate(trellis_generate::GenerateArgs),
    #[command(about = "Decode saved structured latents into a 3D Gaussian splat PLY")]
    DecodeGaussians(trellis_generate::DecodeArgs),
    #[command(about = "Report the components a TRELLIS pipeline names and where they resolved to")]
    Inspect {
        #[arg(long, help = "TRELLIS checkpoint directory holding pipeline.json")]
        model: PathBuf,
        #[arg(
            long,
            help = "Directory holding the published checkpoints, for cross-checkpoint \
                    references. Defaults to this checkpoint's owner directory and its parent"
        )]
        models_root: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    #[command(
        name = "structure",
        about = "Generate a sparse structure: text prompt to occupied voxels on the decoded grid"
    )]
    Structure {
        #[arg(long, help = "TRELLIS text checkpoint directory")]
        model: PathBuf,
        #[arg(
            long,
            help = "CLIP text checkpoint the pipeline conditions on, which the TRELLIS \
                    checkpoint does not contain"
        )]
        conditioner: PathBuf,
        #[arg(long)]
        models_root: Option<PathBuf>,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = NonZeroUsize::new(1).unwrap())]
        samples: NonZeroUsize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, help = "Override the pipeline's sampling steps")]
        steps: Option<usize>,
        #[arg(long, help = "Override the pipeline's guidance strength")]
        cfg_strength: Option<f64>,
        #[arg(long, help = "Override the start of the pipeline's guidance interval")]
        cfg_interval_start: Option<f64>,
        #[arg(long, help = "Override the end of the pipeline's guidance interval")]
        cfg_interval_end: Option<f64>,
        #[arg(long, help = "Override the pipeline's timestep rescale factor")]
        rescale_t: Option<f64>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long)]
        json: bool,
        #[arg(long, help = "Write the occupied voxel coordinates to this JSON file")]
        output: Option<PathBuf>,
    },
}

impl TrellisCommand {
    pub(in crate::cli) fn run(self) -> Result<()> {
        let command = self;
        match command {
            TrellisCommand::Generate(args) => trellis_generate::generate(args),
            TrellisCommand::DecodeGaussians(args) => trellis_generate::decode(args),
            command @ TrellisCommand::Inspect { .. } => trellis::run_inspect(command),
            command @ TrellisCommand::Structure { .. } => trellis::run_structure(command),
        }
    }
}

use super::{Adapter, Task};
use crate::cli::trellis_generate;
use anyhow::Result;
use clap::{FromArgMatches, Subcommand};
use std::num::NonZeroUsize;
use std::path::PathBuf;

pub(super) const ADAPTER: Adapter = Adapter {
    id: "trellis",
    task: Task::ThreeD,
    recognizes: |metadata| {
        [
            "TrellisTextTo3DPipeline",
            "TrellisImageTo3DPipeline",
            "Trellis2ImageTo3DPipeline",
        ]
        .iter()
        .any(|name| metadata.architecture(name))
    },
    command: || TrellisCommand::augment_subcommands(clap::Command::new(Task::ThreeD.name())),
    run: |matches| TrellisCommand::from_arg_matches(matches)?.run(),
};
