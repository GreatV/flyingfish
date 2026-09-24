use super::kit;
use super::text_runtime::{TextDevice, greedy_token, resolve_text_device};
#[cfg(not(feature = "cuda"))]
use anyhow::bail;
use anyhow::{Context, Result};
use clap::Subcommand;
use flyingfish::edge0::config::{Edge0Config, chat_prompt};
use flyingfish::edge0::model::{Edge0Text, configured_max_ctx};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug, Subcommand)]
pub(super) enum Edge0Command {
    #[command(about = "Generate text with an Edge0-35B-A3B checkpoint using greedy decoding")]
    Generate {
        #[arg(long, help = "Edge0-35B-A3B checkpoint directory")]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = NonZeroUsize::new(128).unwrap())]
        max_new_tokens: NonZeroUsize,
        #[command(flatten)]
        device: kit::DeviceArgs,
        #[arg(
            long,
            help = "Upload the MoE expert set to the device; a capacity planner verifies it first"
        )]
        resident_experts: bool,
        #[arg(help = "Print the topology-derived expert-residency bound and its provenance to stderr")]
        #[arg(long)]
        explain_config: bool,
    },
}

pub(super) fn run(command: Edge0Command) -> Result<()> {
    let Edge0Command::Generate {
        model: model_dir,
        prompt,
        max_new_tokens,
        device,
        resident_experts,
        explain_config,
    } = command;
    let kit::DeviceArgs { device } = device;
    let (device, auto) = resolve_text_device(&device)?;
    let config = Edge0Config::from_model_dir(&model_dir)?;
    if explain_config {
        let weights = flyingfish::edge0::weights::Edge0Weights::open(&model_dir)
            .context("open edge0 checkpoint for the configuration derivation")?;
        let sizes = flyingfish::edge0::mode::WeightSizes::from_weights(&weights)?;
        let selected_ordinal = match &device {
            TextDevice::Cuda(ordinals) => ordinals.first().copied().unwrap_or(0),
            TextDevice::Cpu => 0,
        };
        let capture_device = match &device {
            TextDevice::Cuda(_) => candle_core::Device::new_cuda(selected_ordinal)?,
            TextDevice::Cpu => candle_core::Device::Cpu,
        };
        let profile = ff_core::topology::TopologyProfile::capture(&capture_device);
        let derived =
            flyingfish::edge0::resources::derive_edge0_configuration(&sizes, &profile, selected_ordinal)?;
        // GLM and edge0 stream through mmap by contract, so the weight-source
        // rules do not describe their runtimes; the pool rule is the derived
        // guidance.
        for step in derived
            .provenance
            .iter()
            .filter(|step| step.rule == ff_core::configure::RULE_POOL_RESIDENCY)
        {
            eprintln!("config: {step}");
        }
    }
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("load tokenizer: {error}"))?;
    let ids = tokenizer
        .encode(chat_prompt(&prompt).as_str(), false)
        .map_err(|error| anyhow::anyhow!("tokenize prompt: {error}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!ids.is_empty(), "prompt tokenized to an empty sequence");
    anyhow::ensure!(
        ids.len()
            .checked_add(max_new_tokens.get())
            .is_some_and(|total| total <= config.text_config.max_position_embeddings),
        "prompt and requested output exceed model context"
    );
    let device = match device {
        TextDevice::Cuda(ordinals) => {
            if auto && ids.len() + max_new_tokens.get() > configured_max_ctx() {
                eprintln!(
                    "auto: prompt + generation exceeds the {} KV-cache capacity; falling back to CPU",
                    configured_max_ctx()
                );
                TextDevice::Cpu
            } else {
                TextDevice::Cuda(ordinals)
            }
        }
        TextDevice::Cpu => TextDevice::Cpu,
    };
    let eos_token_ids = config.eos_token_id.clone();
    let mut model = Edge0Text::load(&model_dir, config)?;
    let generated = match device {
        TextDevice::Cpu => {
            anyhow::ensure!(
                !resident_experts,
                "--resident-experts requires a CUDA device"
            );
            generate_greedy(&mut model, &ids, max_new_tokens.get(), &eos_token_ids)?
        }
        TextDevice::Cuda(ordinals) => generate_cuda(
            &ordinals,
            &mut model,
            &ids,
            max_new_tokens.get(),
            resident_experts,
            &eos_token_ids,
        )?,
    };
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|error| anyhow::anyhow!("decode output: {error}"))?;
    println!("{text}");
    eprintln!("generated {} tokens (greedy)", generated.len());
    Ok(())
}

fn generate_greedy(
    model: &mut Edge0Text,
    ids: &[u32],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    let mut hidden = None;
    for &id in ids {
        hidden = Some(model.forward(id)?);
    }
    decode_from_hidden(model, hidden, max_new_tokens, eos_token_ids)
}

fn decode_from_hidden(
    model: &mut Edge0Text,
    mut hidden: Option<Vec<f32>>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    let mut generated = Vec::with_capacity(max_new_tokens);
    while generated.len() < max_new_tokens {
        let logits = model.logits(hidden.as_ref().context("missing prefill state")?)?;
        let best = greedy_token(&logits)?;
        generated.push(best);
        if eos_token_ids.contains(&best) || generated.len() == max_new_tokens {
            break;
        }
        hidden = Some(model.forward(best)?);
    }
    Ok(generated)
}

#[cfg(feature = "cuda")]
fn generate_cuda(
    ordinals: &[usize],
    model: &mut Edge0Text,
    ids: &[u32],
    max_new_tokens: usize,
    resident_experts: bool,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    anyhow::ensure!(
        !ordinals.is_empty(),
        "--device cuda:N[,M...] requires at least one ordinal"
    );
    model.enable_gpu_multi(ordinals, resident_experts)?;
    if ordinals.len() > 1 {
        return generate_cuda_multi(model, ids, max_new_tokens, eos_token_ids);
    }
    let needed = ids.len() + max_new_tokens;
    let max_ctx = model.gpu_max_ctx().context("resident runtime")?;
    anyhow::ensure!(
        needed <= max_ctx,
        "prompt + generation {needed} exceeds the {max_ctx} KV-cache capacity; \
         raise EDGE0_MAX_CTX (kernel cap 8192)"
    );
    let mut hidden = None;
    for &id in ids {
        hidden = Some(model.forward(id)?);
    }
    if model.has_resident_experts() {
        let mut generated = vec![model.first_token()?];
        while generated.len() < max_new_tokens && !eos_token_ids.contains(generated.last().unwrap())
        {
            let prev = *generated.last().expect("first token");
            generated.push(model.forward_token(prev)?);
        }
        return Ok(generated);
    }
    decode_from_hidden(model, hidden, max_new_tokens, eos_token_ids)
}

#[cfg(feature = "cuda")]
fn generate_cuda_multi(
    model: &mut Edge0Text,
    ids: &[u32],
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    // Prefill runs on the multi-device runtime one token at a time so the
    // device state (KV, GDN, position counters) is warm when decode starts.
    // Single-device behaviour is bitwise-equivalent because each peer runs
    // the same per-layer kernels as the single-context path.
    let max_ctx = configured_max_ctx();
    anyhow::ensure!(
        ids.len() + max_new_tokens <= max_ctx,
        "prompt + generation {} exceeds the {max_ctx} KV-cache capacity; \
         raise EDGE0_MAX_CTX (kernel cap 8192)",
        ids.len() + max_new_tokens,
    );
    for &id in ids {
        model.forward_multi(id)?;
    }
    let mut generated = vec![model.first_token_multi()?];
    while generated.len() < max_new_tokens && !eos_token_ids.contains(generated.last().unwrap()) {
        let prev = *generated.last().expect("first token");
        generated.push(model.forward_token_multi(prev)?);
    }
    Ok(generated)
}

#[cfg(not(feature = "cuda"))]
fn generate_cuda(
    _: &[usize],
    _: &mut Edge0Text,
    _: &[u32],
    _: usize,
    _: bool,
    _: &[u32],
) -> Result<Vec<u32>> {
    bail!("CUDA decoding requires a binary built with --features cuda")
}
