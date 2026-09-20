use super::*;
use flyingfish::qwen35::{
    config::{Qwen35Config, chat_prompt, chat_prompt_with_image},
    model::Qwen35Text,
    vision::{self, VisionGrid},
    weights::Qwen35Weights,
};

/// The attn_scores CUDA kernel's shared-memory bound.
const GPU_CONTEXT_CAP: usize = 8192;

#[derive(Debug, Subcommand)]
pub(super) enum Qwen35Command {
    #[command(about = "Generate text with a Qwen3.8-27B checkpoint using greedy decoding")]
    Generate {
        #[arg(long, help = "Qwen3.8-27B checkpoint directory")]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(
            long,
            help = "PNG image the prompt refers to; the tower runs on the host"
        )]
        image: Option<PathBuf>,
        #[arg(long, default_value_t = NonZeroUsize::new(128).unwrap())]
        max_new_tokens: NonZeroUsize,
        #[arg(long, default_value = "auto", help = "cpu, auto, or cuda:N[,M...]")]
        device: String,
    },
}

pub(super) fn run(command: Qwen35Command) -> Result<()> {
    let Qwen35Command::Generate {
        model: model_dir,
        prompt,
        image,
        max_new_tokens,
        device,
    } = command;
    let (device, auto) = resolve_text_device(&device)?;
    let config = Qwen35Config::from_model_dir(&model_dir)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("load tokenizer: {error}"))?;
    let vision_input = match &image {
        Some(path) => Some(preprocess(&model_dir, path)?),
        None => None,
    };
    let templated = if vision_input.is_some() {
        chat_prompt_with_image(&prompt)
    } else {
        chat_prompt(&prompt)
    };
    let mut ids = tokenizer
        .encode(templated.as_str(), false)
        .map_err(|error| anyhow::anyhow!("tokenize prompt: {error}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!ids.is_empty(), "prompt tokenized to an empty sequence");
    let (pos3, mrope_delta) = match &vision_input {
        Some((_, grid)) => expand_image_tokens(&config, &mut ids, *grid)?,
        None => (Vec::new(), 0),
    };
    let total = ids
        .len()
        .checked_add(max_new_tokens.get())
        .context("prompt and requested output overflow the context arithmetic")?;
    anyhow::ensure!(
        total <= config.text_config.max_position_embeddings,
        "prompt and requested output exceed model context"
    );
    let device = match device {
        TextDevice::Cuda(ordinals) => {
            if total > GPU_CONTEXT_CAP {
                if auto {
                    eprintln!(
                        "auto: prompt + generation {total} exceeds the {GPU_CONTEXT_CAP} kernel cap; falling back to CPU"
                    );
                    TextDevice::Cpu
                } else {
                    bail!("prompt + generation {total} exceeds the {GPU_CONTEXT_CAP} kernel cap");
                }
            } else {
                match checkpoint_fits_free_vram(&ordinals, &model_dir, &config, total)? {
                    true => TextDevice::Cuda(ordinals),
                    false if auto => {
                        eprintln!(
                            "auto: resident and streaming plans exceed free VRAM; falling back to CPU"
                        );
                        TextDevice::Cpu
                    }
                    false => {
                        let list = ordinals
                            .iter()
                            .map(|o| o.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        bail!(
                            "resident and streaming plans exceed free VRAM on cuda:{list}; \
                             pass another --device (cuda:N with more memory, or cpu)"
                        )
                    }
                }
            }
        }
        TextDevice::Cpu => TextDevice::Cpu,
    };
    let vision = match vision_input {
        Some((patches, grid)) => {
            Some((run_vision_tower(&model_dir, &config, patches, grid)?, grid))
        }
        None => None,
    };
    let generated = match device {
        TextDevice::Cpu => {
            let mut model = Qwen35Text::load(&model_dir, config.clone())?;
            let mut hidden = prefill(&config, &mut model, &ids, &pos3, vision.as_ref())?;
            if vision.is_some() {
                model.set_mrope_delta(mrope_delta);
            }
            let mut generated = Vec::with_capacity(max_new_tokens.get());
            while generated.len() < max_new_tokens.get() {
                let logits = model.logits(&hidden)?;
                let best = greedy_token(&logits)?;
                generated.push(best);
                if config.text_config.eos_token_id.contains(&best)
                    || generated.len() == max_new_tokens.get()
                {
                    break;
                }
                hidden = model.forward(best)?;
            }
            generated
        }
        TextDevice::Cuda(ordinals) => generate_cuda(
            &ordinals,
            &model_dir,
            &config,
            &ids,
            &pos3,
            vision.as_ref(),
            max_new_tokens.get(),
        )?,
    };
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|error| anyhow::anyhow!("decode output: {error}"))?;
    println!("{text}");
    eprintln!("generated {} tokens (greedy)", generated.len());
    Ok(())
}

fn preprocess(model_dir: &Path, image: &Path) -> Result<(Vec<f32>, VisionGrid)> {
    let processor = vision::ProcessorConfig::from_model_dir(model_dir)?;
    let image = vision::RgbImage::from_png(image)?;
    vision::preprocess_image(&image, &processor)
}

fn run_vision_tower(
    model_dir: &Path,
    config: &Qwen35Config,
    patches: Vec<f32>,
    grid: VisionGrid,
) -> Result<Vec<f32>> {
    let weights = Qwen35Weights::open(model_dir)?;
    let vision_config = config
        .vision_config
        .as_ref()
        .context("checkpoint has no vision_config")?;
    anyhow::ensure!(
        vision_config.out_hidden_size == config.text_config.hidden_size,
        "tower out_hidden {} != text hidden {}",
        vision_config.out_hidden_size,
        config.text_config.hidden_size
    );
    let tower = vision::VisionTower::load(&weights, vision_config)?;
    tower.forward(&patches, grid)
}

/// The runtime's own residency decision: a device keeps a resident prefix
/// of its layers and streams the rest; only a device that cannot host the
/// always-resident set (statics, slots, KV, scratch) rejects the request.
#[cfg(feature = "cuda")]
fn checkpoint_fits_free_vram(
    ordinals: &[usize],
    model_dir: &Path,
    config: &Qwen35Config,
    total_tokens: usize,
) -> Result<bool> {
    anyhow::ensure!(
        !ordinals.is_empty(),
        "--device cuda:N[,M...] requires at least one ordinal"
    );
    let max_ctx = if total_tokens > 4096 {
        total_tokens.next_power_of_two()
    } else {
        4096
    };
    let weights = Qwen35Weights::open(model_dir)
        .with_context(|| format!("open weights for {}", model_dir.display()))?;
    let ranges = flyingfish::qwen35::gpu::partition_layers(&weights, config, ordinals.len())?;
    let free: Vec<u64> = ordinals
        .iter()
        .map(|&ordinal| {
            let context = cudarc::driver::CudaContext::new(ordinal)
                .with_context(|| format!("open CUDA device {ordinal}"))?;
            Ok(context.mem_get_info().context("mem_get_info")?.0 as u64)
        })
        .collect::<Result<Vec<_>>>()?;
    let plans = flyingfish::qwen35::gpu::plan_residency(&weights, config, &ranges, max_ctx, &free)?;
    Ok(plans
        .iter()
        .all(|plan| plan.residency != flyingfish::qwen35::gpu::Residency::Insufficient))
}

#[cfg(not(feature = "cuda"))]
fn checkpoint_fits_free_vram(_: &[usize], _: &Path, _: &Qwen35Config, _: usize) -> Result<bool> {
    Ok(true)
}

/// Replace the <|image_pad|> placeholder with one token per merged row and
/// compute the per-token mrope positions.
fn expand_image_tokens(
    config: &Qwen35Config,
    ids: &mut Vec<u32>,
    grid: VisionGrid,
) -> Result<(Vec<[i32; 3]>, i64)> {
    let image_token = config
        .image_token_id
        .context("checkpoint has no image_token_id")?;
    let merge = config
        .vision_config
        .as_ref()
        .map(|vision| vision.spatial_merge_size)
        .unwrap_or(2);
    let merged = grid.merged_count(merge)?;
    let at = ids
        .iter()
        .position(|&token| token == image_token)
        .context("no <|image_pad|> in the tokenized prompt")?;
    let mut expanded = Vec::with_capacity(ids.len() + merged - 1);
    expanded.extend_from_slice(&ids[..at]);
    expanded.extend(std::iter::repeat_n(image_token, merged));
    expanded.extend_from_slice(&ids[at + 1..]);
    *ids = expanded;
    let types: Vec<u8> = ids
        .iter()
        .map(|&token| (token == image_token) as u8)
        .collect();
    vision::multimodal_positions(&types, &[grid], merge)
}

fn prefill(
    config: &Qwen35Config,
    model: &mut Qwen35Text,
    ids: &[u32],
    pos3: &[[i32; 3]],
    vision: Option<&(Vec<f32>, VisionGrid)>,
) -> Result<Vec<f32>> {
    let mut hidden = None;
    match vision {
        Some((rows, _)) => {
            let hidden_size = config.text_config.hidden_size;
            let image_token = config
                .image_token_id
                .context("checkpoint has no image_token_id")?;
            let mut pad_row = 0usize;
            for (index, &id) in ids.iter().enumerate() {
                let position = [
                    pos3[index][0] as usize,
                    pos3[index][1] as usize,
                    pos3[index][2] as usize,
                ];
                hidden = Some(if id == image_token {
                    let row = rows[pad_row * hidden_size..(pad_row + 1) * hidden_size].to_vec();
                    pad_row += 1;
                    model.forward_vision_row(row, position)?.1
                } else {
                    model.forward_at(id, position)?.1
                });
            }
        }
        None => {
            for &id in ids {
                hidden = Some(model.forward(id)?);
            }
        }
    }
    hidden.context("missing prefill state")
}

#[cfg(feature = "cuda")]
fn generate_cuda(
    ordinals: &[usize],
    model_dir: &Path,
    config: &Qwen35Config,
    ids: &[u32],
    pos3: &[[i32; 3]],
    vision: Option<&(Vec<f32>, VisionGrid)>,
    max_new_tokens: usize,
) -> Result<Vec<u32>> {
    use flyingfish::qwen35::gpu::QwenGpu;

    anyhow::ensure!(
        !ordinals.is_empty(),
        "--device cuda:N[,M...] requires at least one ordinal"
    );
    let total = ids.len() + max_new_tokens;
    anyhow::ensure!(
        total <= GPU_CONTEXT_CAP,
        "prompt + generation {total} exceeds the {GPU_CONTEXT_CAP} kernel cap"
    );
    let weights = Qwen35Weights::open(model_dir)?;
    let mut gpu = if total > 4096 {
        QwenGpu::with_max_ctx(ordinals, &weights, config, total.next_power_of_two())?
    } else {
        QwenGpu::new(ordinals, &weights, config)?
    };
    match vision {
        Some((rows, _)) => {
            let hidden_size = config.text_config.hidden_size;
            let image_token = config
                .image_token_id
                .context("checkpoint has no image_token_id")?;
            let mut pad_row = 0usize;
            for (index, &id) in ids.iter().enumerate() {
                if id == image_token {
                    let row = &rows[pad_row * hidden_size..(pad_row + 1) * hidden_size];
                    pad_row += 1;
                    gpu.push_vision_row(row, pos3[index])?;
                } else {
                    gpu.push_token_at(id, pos3[index])?;
                }
            }
        }
        None => {
            for &id in ids {
                gpu.push_token(id)?;
            }
        }
    }
    let mut generated = vec![gpu.read_token()?];
    while generated.len() < max_new_tokens
        && !config
            .text_config
            .eos_token_id
            .contains(generated.last().unwrap())
    {
        gpu.step()?;
        generated.push(gpu.read_token()?);
    }
    Ok(generated)
}

#[cfg(not(feature = "cuda"))]
fn generate_cuda(
    _: &[usize],
    _: &Path,
    _: &Qwen35Config,
    _: &[u32],
    _: &[[i32; 3]],
    _: Option<&(Vec<f32>, VisionGrid)>,
    _: usize,
) -> Result<Vec<u32>> {
    bail!("CUDA decoding requires a binary built with --features cuda")
}
