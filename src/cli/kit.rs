//! Shared argument groups for model adapters.

use std::path::PathBuf;

#[derive(Clone, Debug, clap::Args)]
pub struct SamplingArgs {
    #[arg(
        long,
        default_value_t = 1.0,
        help = "Sampling temperature; 0 selects greedy decoding"
    )]
    pub temperature: f64,
    #[arg(
        long,
        default_value_t = 0.95,
        help = "Nucleus probability; no additional top-k filter is applied"
    )]
    pub top_p: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Clone, Debug, clap::Args)]
pub struct DeviceArgs {
    #[arg(long, default_value = "auto", help = "cpu, auto, or cuda:N; some adapters also accept cuda:N[,M...] or metal:N")]
    pub device: String,
}

#[derive(Clone, Debug, clap::Args)]
pub struct OutputArgs {
    #[arg(
        long,
        help = "Atomically write the generated text or JSON result outside the model directory"
    )]
    pub output: Option<PathBuf>,
    #[arg(long, help = "Emit one JSON result instead of plain generated text")]
    pub json: bool,
    #[arg(
        long,
        value_name = "PATH",
        help = "Write sampled RSS/CUDA peaks to a new JSON file outside the model"
    )]
    pub telemetry_json: Option<PathBuf>,
}
