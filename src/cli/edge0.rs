use super::*;
use flyingfish::edge0::{
    config::{Edge0Config, chat_prompt},
    model::{Edge0Text, configured_max_ctx},
};

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
        #[arg(long, default_value = "auto", help = "cpu, auto, or cuda:N")]
        device: String,
        #[arg(
            long,
            help = "Upload the MoE expert set to the device; a capacity planner verifies it first"
        )]
        resident_experts: bool,
    },
}

pub(super) fn run(command: Edge0Command) -> Result<()> {
    let Edge0Command::Generate {
        model: model_dir,
        prompt,
        max_new_tokens,
        device,
        resident_experts,
    } = command;
    let (device, auto) = resolve_text_device(&device)?;
    let config = Edge0Config::from_model_dir(&model_dir)?;
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
            anyhow::ensure!(
                ordinals.len() == 1,
                "edge0 decode is single-device; pass one --device (cuda:N or cpu), not cuda:{}",
                ordinals
                    .iter()
                    .map(|o| o.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
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
            ordinals[0],
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
    ordinal: usize,
    model: &mut Edge0Text,
    ids: &[u32],
    max_new_tokens: usize,
    resident_experts: bool,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    model.enable_gpu(ordinal, resident_experts)?;
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

#[cfg(not(feature = "cuda"))]
fn generate_cuda(
    _: usize,
    _: &mut Edge0Text,
    _: &[u32],
    _: usize,
    _: bool,
    _: &[u32],
) -> Result<Vec<u32>> {
    bail!("CUDA decoding requires a binary built with --features cuda")
}
