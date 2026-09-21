use crate::cli::output_hygiene::{ensure_new_output, resolve_output_outside_model};
use crate::cli::{Result, WeightCacheArgs, kit};
use anyhow::bail;
use candle_core::Tensor;
use clap::Subcommand;
use flyingfish::dsv41::config::DeepseekV41Config;
use flyingfish::dsv41::weights::TransformerLoader;
use std::{num::NonZeroUsize, path::PathBuf};

#[derive(Debug, Subcommand)]
pub(super) enum Dsv41Command {
    #[command(about = "Generate text with a DeepSeek-V4.1-Flash checkpoint")]
    Generate {
        #[arg(long, help = "DeepSeek-V4.1-Flash checkpoint directory")]
        model: PathBuf,
        #[arg(
            long,
            help = "Prompt text; an image path grounds it when --image is set"
        )]
        prompt: String,
        #[arg(
            long,
            help = "PNG image the prompt refers to; the tower runs on the host"
        )]
        image: Option<PathBuf>,
        #[arg(
            long,
            default_value = "chat",
            value_parser = ["chat", "thinking"],
            help = "chat closes the thinking block; thinking leaves it open for the model to reason"
        )]
        thinking_mode: String,
        #[arg(
            long,
            default_value = "high",
            help = "Reasoning budget for thinking mode: low, high, max, or an integer 1-100"
        )]
        reasoning_effort: String,
        #[arg(
            long,
            help = "System message prepended to the conversation; thinking mode adds the effort header"
        )]
        system: Option<String>,
        #[command(flatten)]
        sampling: kit::SamplingArgs,
        #[command(flatten)]
        device: kit::DeviceArgs,
        #[command(flatten)]
        output: kit::OutputArgs,
        #[command(flatten)]
        limits: kit::DecodeLimitArgs,
        #[command(flatten)]
        weights: WeightCacheArgs,
    },
    #[command(
        name = "capture-parity",
        about = "Export one bounded DeepSeek-V4.1 prefill boundary for offline reference comparison"
    )]
    CaptureParity {
        #[arg(long, help = "DeepSeek-V4.1-Flash checkpoint directory")]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(
            long,
            default_value_t = NonZeroUsize::new(128).unwrap(),
            help = "Hard prompt limit; the capture runs a single prefill"
        )]
        max_context_tokens: NonZeroUsize,
        #[arg(
            long,
            default_value = "max",
            help = "Reasoning budget: low, high, max, or an integer 1-100"
        )]
        reasoning_effort: String,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
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
        about = "Schema placeholder; the replay lands with the routed-expert trace"
    )]
    ReplayRouting {
        #[arg(long, help = "Versioned routing-trace JSON emitted by dsv41 generate")]
        trace: PathBuf,
        #[arg(long, help = "New atomically published routing-replay JSON report")]
        output: PathBuf,
    },
}

pub(super) fn run(command: Dsv41Command) -> Result<()> {
    match command {
        Dsv41Command::Generate {
            model: model_dir,
            prompt,
            image,
            thinking_mode,
            reasoning_effort,
            system,
            sampling,
            device,
            output,
            limits,
            weights,
        } => {
            let config = DeepseekV41Config::from_model_dir(&model_dir)?;
            let effort = flyingfish::dsv41::encoding::ReasoningEffort::parse(&reasoning_effort)
                .map_err(|error| anyhow::anyhow!(error))?;
            if image.is_some() {
                anyhow::bail!(
                    "--image generation is not wired yet; the vision tower runs only in capture-parity fixtures"
                );
            }
            let _ = &weights;
            let loader = TransformerLoader::open(&model_dir)?;
            admit_resident_footprint(&loader)?;
            let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .map_err(|error| anyhow::anyhow!("load tokenizer: {error}"))?;
            let thinking = match thinking_mode.as_str() {
                "chat" => false,
                "thinking" => true,
                other => bail!("unknown thinking mode {other:?}"),
            };
            let encoded = flyingfish::dsv41::encoding::chat_prompt(
                &prompt,
                system.as_deref(),
                thinking,
                effort,
            );
            let ids = tokenizer
                .encode(encoded, false)
                .map_err(|error| anyhow::anyhow!("tokenize prompt: {error}"))?
                .get_ids()
                .to_vec();
            anyhow::ensure!(!ids.is_empty(), "prompt tokenized to an empty sequence");
            let max_context = limits.max_context_tokens.get();
            anyhow::ensure!(
                ids.len() <= max_context,
                "prompt of {} tokens exceeds the {}-token request limit",
                ids.len(),
                max_context
            );
            let _ = &device;
            let max_seq =
                max_context.max(config.text_config.max_position_embeddings.min(max_context));
            let mut transformer = loader.load(Some(&tokenizer), max_seq)?;
            let device = candle_core::Device::Cpu;
            let mut position = 0usize;
            let chunk = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let (mut token, logits) = transformer.forward(&chunk, position)?;
            let mut generated = vec![token];
            position = ids.len();
            let budget = limits.max_new_tokens.get();
            while generated.len() < budget {
                let step = Tensor::from_vec(vec![token], (1, 1), &device)?;
                let (next, _) = transformer.forward(&step, position)?;
                token = next;
                generated.push(token);
                position += 1;
                if token == config.eos_token_id {
                    generated.pop();
                    break;
                }
            }
            let text = tokenizer
                .decode(&generated, true)
                .map_err(|error| anyhow::anyhow!("decode output: {error}"))?;
            let _ = (sampling, &logits, &output);
            println!("{text}");
            eprintln!("generated {} tokens (greedy)", generated.len());
            Ok(())
        }
        Dsv41Command::CaptureParity {
            model: model_dir,
            prompt,
            max_context_tokens,
            reasoning_effort,
            device,
            weights,
            no_progress,
            output,
        } => {
            let _ = (device, weights, no_progress);
            let effort = flyingfish::dsv41::encoding::ReasoningEffort::parse(&reasoning_effort)
                .map_err(|error| anyhow::anyhow!(error))?;
            let loader = TransformerLoader::open(&model_dir)?;
            admit_resident_footprint(&loader)?;
            let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .map_err(|error| anyhow::anyhow!("load tokenizer: {error}"))?;
            let encoded = flyingfish::dsv41::encoding::chat_prompt(&prompt, None, true, effort);
            let mut ids = tokenizer
                .encode(encoded, false)
                .map_err(|error| anyhow::anyhow!("tokenize prompt: {error}"))?
                .get_ids()
                .to_vec();
            let limit = max_context_tokens.get();
            anyhow::ensure!(!ids.is_empty(), "prompt tokenized to an empty sequence");
            ids.truncate(limit);
            let mut transformer = loader.load(Some(&tokenizer), limit.max(ids.len()))?;
            let device = candle_core::Device::Cpu;
            let chunk = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let mut snapshots = Vec::new();
            let (token, _) = transformer.forward_with_capture(&chunk, 0, Some(&mut snapshots))?;
            anyhow::ensure!(
                snapshots.iter().all(|snapshot| snapshot.dims().len() == 4),
                "layer snapshots must keep the hc stream shape"
            );
            let mut tensors = std::collections::HashMap::new();
            for (index, snapshot) in snapshots.iter().enumerate() {
                tensors.insert(format!("layer_{index:02}"), snapshot.clone());
            }
            let prompt_tensor = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            tensors.insert("prompt_tokens".to_owned(), prompt_tensor);
            let logits_row = Tensor::from_vec(vec![token], (1,), &device)?;
            tensors.insert("greedy_token".to_owned(), logits_row);
            let output = resolve_output_outside_model(&output, &model_dir)?;
            ensure_new_output(&output, "parity capture output")?;
            let parent = output.parent().unwrap_or(std::path::Path::new("."));
            anyhow::ensure!(
                parent.is_dir(),
                "parity output parent {} does not exist",
                parent.display()
            );
            candle_core::safetensors::save(&tensors, &output)?;
            println!("saved parity capture to {}", output.display());
            Ok(())
        }
        Dsv41Command::ReplayRouting { .. } => {
            bail!("dsv41 replay-routing is a schema placeholder")
        }
    }
}

/// Refuse loudly when the dequantized resident set cannot fit this host, in
/// line with the design doc's admission stance for oversized checkpoints.
fn admit_resident_footprint(loader: &TransformerLoader) -> Result<()> {
    let needed = loader.resident_f32_bytes()?;
    let available = host_available_bytes();
    anyhow::ensure!(
        needed <= available,
        "the resident F32 load needs {} GiB but only {} GiB is available;          the routed experts need streaming, which is not wired yet",
        needed / (1 << 30),
        available / (1 << 30),
    );
    Ok(())
}

fn host_available_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemAvailable:"))
                .and_then(|line| {
                    line.split_whitespace()
                        .nth(1)
                        .and_then(|kib| kib.parse::<u64>().ok())
                })
                .map(|kib| kib * 1024)
        })
        .unwrap_or(u64::MAX)
}
