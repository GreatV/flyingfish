use super::kit;
use super::text_runtime::{TextDevice, greedy_token, resolve_text_device};
use anyhow::{Context, Result, bail};
use clap::Subcommand;
use flyingfish::qwen35::config::{Qwen35Config, chat_prompt, chat_prompt_with_image};
use flyingfish::qwen35::model::Qwen35Text;
use flyingfish::qwen35::vision::{self, VisionGrid};
use flyingfish::qwen35::weights::Qwen35Weights;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

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
        #[arg(
            help = "Draft one token per round with the checkpoint's MTP head and verify both \
                    (CUDA, resident weights, single device)"
        )]
        #[arg(long)]
        speculative: bool,
        #[arg(
            help = "Skip drafting when the model's top1-top2 logit margin falls below this \
                    threshold; requires --speculative"
        )]
        #[arg(long)]
        speculative_gate: Option<f32>,
        #[command(flatten)]
        device: kit::DeviceArgs,
    },
}

pub(super) fn run(command: Qwen35Command) -> Result<()> {
    let Qwen35Command::Generate {
        model: model_dir,
        prompt,
        image,
        max_new_tokens,
        speculative,
        speculative_gate,
        device,
    } = command;
    let kit::DeviceArgs { device } = device;
    let (device, auto) = resolve_text_device(&device)?;
    if speculative && image.is_some() {
        bail!("--speculative is text-only; it cannot be combined with --image");
    }
    if speculative_gate.is_some() && !speculative {
        bail!("--speculative-gate requires --speculative");
    }
    if let Some(gate) = speculative_gate {
        anyhow::ensure!(
            gate > 0.0,
            "--speculative-gate must be positive, got {gate}"
        );
    }
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
    if speculative && matches!(device, TextDevice::Cpu) {
        bail!("--speculative requires CUDA decode; the resolved device is cpu");
    }
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
        TextDevice::Cuda(ordinals) => {
            #[cfg(feature = "cuda")]
            {
                generate_cuda(CudaDecode {
                    ordinals: &ordinals,
                    model_dir: &model_dir,
                    config: &config,
                    ids: &ids,
                    pos3: &pos3,
                    vision: vision.as_ref(),
                    max_new_tokens: max_new_tokens.get(),
                    speculative,
                    gate: speculative_gate,
                })?
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!(
                    "CUDA decoding requires a binary built with --features cuda \
                     (requested ordinals {ordinals:?})"
                );
            }
        }
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

/// One CUDA decode request: everything `generate_cuda` needs to open the
/// checkpoint, run prefill, and decode.
#[cfg(feature = "cuda")]
struct CudaDecode<'a> {
    ordinals: &'a [usize],
    model_dir: &'a Path,
    config: &'a Qwen35Config,
    ids: &'a [u32],
    pos3: &'a [[i32; 3]],
    vision: Option<&'a (Vec<f32>, VisionGrid)>,
    max_new_tokens: usize,
    speculative: bool,
    gate: Option<f32>,
}

#[cfg(feature = "cuda")]
fn generate_cuda(request: CudaDecode<'_>) -> Result<Vec<u32>> {
    use flyingfish::qwen35::gpu::QwenGpu;

    let CudaDecode {
        ordinals,
        model_dir,
        config,
        ids,
        pos3,
        vision,
        max_new_tokens,
        speculative,
        gate,
    } = request;

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
            gpu.push_tokens(ids)?;
        }
    }
    if speculative {
        return generate_cuda_speculative(&mut gpu, &weights, config, max_new_tokens, gate);
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

/// Greedy decode with one MTP-drafted token verified per round. The emitted
/// ids are the non-speculative path's by construction: a draft is only
/// emitted after the model's own greedy token confirmed it.
#[cfg(feature = "cuda")]
fn generate_cuda_speculative(
    gpu: &mut flyingfish::qwen35::gpu::QwenGpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    max_new_tokens: usize,
    gate: Option<f32>,
) -> Result<Vec<u32>> {
    use flyingfish::qwen35::spec::QwenSpec;
    use std::time::Instant;

    let eos = |token: &u32| config.text_config.eos_token_id.contains(token);
    let mut spec = QwenSpec::new(gpu, weights)?;
    spec.margins_enabled = gate.is_some();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let (mut accepts, mut rejects, mut skips) = (0usize, 0usize, 0usize);
    let mut pending = gpu.read_token()?;
    spec.draft(gpu, Some(&gpu.hidden), pending)?;
    let mut have_draft = true;
    let started = Instant::now();
    while generated.len() < max_new_tokens {
        // `pending` is a model-confirmed token carried from the previous
        // round (or the prefill's first); emit it alone and stop when it is
        // EOS rather than staging it under a fresh draft.
        if eos(&pending) {
            generated.push(pending);
            break;
        }
        gpu.ctx
            .stream
            .memcpy_htod(&[pending as i32], &mut gpu.next_token)?;
        // A verify round emits two tokens, so it needs two slots; the last
        // slot fills with a plain step's verified token instead of an
        // unverified draft.
        if have_draft && generated.len() + 2 <= max_new_tokens {
            let (a, b) = spec.verify_round(gpu)?;
            let accepted = a == spec.draft_id;
            if accepted {
                accepts += 1;
                generated.push(pending);
                generated.push(spec.draft_id);
                gpu.ctx.glue_inc(&mut gpu.pos)?;
                gpu.ctx.glue_inc(&mut gpu.pos)?;
                gpu.ctx.glue_inc3(&mut gpu.rope_pos)?;
                gpu.ctx.glue_inc3(&mut gpu.rope_pos)?;
                gpu.position += 2;
                pending = b;
                have_draft = gate.is_none_or(|t| spec.margin_b >= t);
            } else {
                rejects += 1;
                generated.push(pending);
                spec.reject_restore(gpu)?;
                gpu.ctx.glue_inc(&mut gpu.pos)?;
                gpu.ctx.glue_inc3(&mut gpu.rope_pos)?;
                gpu.position += 1;
                pending = a;
                have_draft = gate.is_none_or(|t| spec.margin_a >= t);
            }
            if accepted && eos(&spec.draft_id) {
                break;
            }
            if generated.len() >= max_new_tokens {
                break;
            }
            if have_draft {
                if accepted {
                    spec.draft(gpu, None, b)?;
                } else {
                    spec.draft(gpu, Some(&gpu.hidden), a)?;
                }
            } else {
                skips += 1;
            }
        } else {
            generated.push(pending);
            if generated.len() >= max_new_tokens {
                break;
            }
            gpu.step()?;
            pending = gpu.read_token()?;
            let margin = spec.step_margin(gpu)?;
            have_draft = gate.is_none_or(|t| margin >= t);
            if have_draft {
                spec.draft(gpu, Some(&gpu.hidden), pending)?;
            } else {
                skips += 1;
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let emitted = generated.len();
    let rounds = accepts + rejects;
    let rate = if rounds > 0 {
        accepts as f64 / rounds as f64 * 100.0
    } else {
        0.0
    };
    eprintln!(
        "spec decode: {emitted} tokens in {elapsed:.2}s = {:.1} tok/s; accepts {accepts} \
         rejects {rejects} skips {skips} (accept rate {rate:.0}%)",
        emitted as f64 / elapsed
    );
    Ok(generated)
}
