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
use std::time::Instant;
use tokenizers::Tokenizer;

mod serve;

/// Validated end-to-end CUDA context limit.

#[derive(Debug, Subcommand)]
pub(super) enum Qwen35Command {
    #[command(about = "Keep a Qwen model loaded and serve sequential greedy requests over HTTP")]
    Serve(serve::Args),
    #[command(about = "Generate text with a Qwen3.8-27B checkpoint using greedy decoding")]
    Generate {
        #[arg(long, help = "Qwen3.8-27B checkpoint directory")]
        model: PathBuf,
        #[arg(long)]
        host_profile: Option<PathBuf>,
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
            long,
            help = "Write timings and generated token IDs to a new JSON file"
        )]
        report: Option<PathBuf>,
        #[command(flatten)]
        device: kit::DeviceArgs,
    },
}

pub(super) fn run(command: Qwen35Command) -> Result<()> {
    let started = Instant::now();
    let Qwen35Command::Generate {
        model: model_dir,
        host_profile,
        prompt,
        image,
        max_new_tokens,
        speculative,
        report,
        device,
    } = command
    else {
        let Qwen35Command::Serve(args) = command else {
            unreachable!()
        };
        return serve::run(args);
    };
    #[cfg(not(feature = "cuda"))]
    let _ = &host_profile;
    eprintln!("model: {}", model_dir.display());
    let mut report_file = report
        .as_ref()
        .map(|path| {
            eprintln!("report: {}", path.display());
            std::fs::File::create_new(path)
                .with_context(|| format!("create report {}", path.display()))
        })
        .transpose()?;
    let kit::DeviceArgs { device } = device;
    let (device, auto) = resolve_text_device(&device)?;
    if speculative && image.is_some() {
        bail!("--speculative is text-only; it cannot be combined with --image");
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
            match checkpoint_fits_free_vram(&ordinals, &model_dir, &config, total, speculative)? {
                true => TextDevice::Cuda(ordinals),
                false if auto => {
                    eprintln!(
                        "auto: resident and streaming plans exceed free VRAM at {total} tokens; falling back to CPU"
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
                        "resident and streaming plans exceed free VRAM on cuda:{list} at {total} tokens; \
                         pass another --device (cuda:N with more memory, or cpu)"
                    )
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
    let mut timings = Timings::default();
    let generated = match device {
        TextDevice::Cpu => {
            let load = Instant::now();
            let mut model = Qwen35Text::load(&model_dir, config.clone())?;
            timings.load_ms = load.elapsed().as_secs_f64() * 1000.0;
            let prefill_start = Instant::now();
            let mut hidden = prefill(&config, &mut model, &ids, &pos3, vision.as_ref())?;
            if vision.is_some() {
                model.set_mrope_delta(mrope_delta);
            }
            let mut generated = Vec::with_capacity(max_new_tokens.get());
            let mut decode_start = Instant::now();
            while generated.len() < max_new_tokens.get() {
                let logits = model.logits(&hidden)?;
                let best = greedy_token(&logits)?;
                generated.push(best);
                if generated.len() == 1 {
                    timings.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
                    decode_start = Instant::now();
                }
                if config.text_config.eos_token_id.contains(&best)
                    || generated.len() == max_new_tokens.get()
                {
                    break;
                }
                hidden = model.forward(best)?;
            }
            timings.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            generated
        }
        TextDevice::Cuda(ordinals) => {
            #[cfg(feature = "cuda")]
            {
                generate_cuda(CudaDecode {
                    ordinals: &ordinals,
                    host_profile: host_profile.as_deref(),
                    model_dir: &model_dir,
                    config: &config,
                    ids: &ids,
                    pos3: &pos3,
                    vision: vision.as_ref(),
                    max_new_tokens: max_new_tokens.get(),
                    speculative,
                    timings: &mut timings,
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
    if let Some(file) = report_file.as_mut() {
        let decode_tokens = generated.len().saturating_sub(1);
        serde_json::to_writer_pretty(
            file,
            &serde_json::json!({
                "schema_version": 1,
                "prompt_tokens": ids.len(),
                "generated_ids": generated,
                "load_ms": timings.load_ms,
                "profile_ms": timings.profile_ms,
                "prefill_ms": timings.prefill_ms,
                "prefill_packed_tokens": timings.prefill_packed_tokens,
                "prefill_device_modes": timings.prefill_device_modes,
                "prefill_block": timings.prefill_block,
                "group4": timings.group4,
                "decode_ms": timings.decode_ms,
                "decode_tokens": decode_tokens,
                "decode_tokens_per_second": if timings.decode_ms > 0.0 {
                    Some(decode_tokens as f64 * 1000.0 / timings.decode_ms)
                } else { None },
                "total_ms": started.elapsed().as_secs_f64() * 1000.0,
                "stream16": stream16_json(&timings),
                "speculative": speculative,
                "spec": spec_json(&timings),
            }),
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn stream16_json(timings: &Timings) -> Option<serde_json::Value> {
    timings.stream16.map(|s| {
        serde_json::json!({
            "fill_ms": s.fill_ms,
            "gate_wait_ms": s.gate_wait_ms,
            "h2d_ms": s.h2d_ms,
            "h2d_bytes": s.h2d_bytes,
        })
    })
}

#[cfg(not(feature = "cuda"))]
fn stream16_json(_timings: &Timings) -> Option<serde_json::Value> {
    None
}

#[cfg(feature = "cuda")]
fn spec_json(timings: &Timings) -> Option<serde_json::Value> {
    timings
        .spec
        .as_ref()
        .map(|s| serde_json::json!({"mode": s.mode, "r": s.r, "c": s.c}))
}

#[cfg(not(feature = "cuda"))]
fn spec_json(_timings: &Timings) -> Option<serde_json::Value> {
    None
}

#[derive(Default)]
struct Timings {
    load_ms: f64,
    profile_ms: f64,
    prefill_ms: f64,
    prefill_packed_tokens: usize,
    prefill_device_modes: Vec<String>,
    prefill_block: usize,
    group4: serde_json::Value,
    decode_ms: f64,
    #[cfg(feature = "cuda")]
    spec: Option<flyingfish::qwen35::spec::SpecStats>,
    #[cfg(feature = "cuda")]
    stream16: Option<flyingfish::qwen35::gpu::Stream16Stats>,
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
    speculative: bool,
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
    // QwenSpec adds the B-column activation set, the A/B-seam GDN scratch,
    // a whole extra MTP decoder layer with its own KV cache and weights,
    // and the round ring on top of the resident plan; charge their derived
    // size up front so admission reflects the speculative footprint.
    let spec_bytes = flyingfish::qwen35::spec::workspace_bytes(
        &weights,
        &config.text_config,
        max_ctx,
        total_tokens + 2,
    );
    let free: Vec<u64> = ordinals
        .iter()
        .map(|&ordinal| {
            let context = cudarc::driver::CudaContext::new(ordinal)
                .with_context(|| format!("open CUDA device {ordinal}"))?;
            Ok(context.mem_get_info().context("mem_get_info")?.0 as u64)
        })
        .collect::<Result<Vec<_>>>()?;
    let free = free
        .into_iter()
        .map(|free| {
            if speculative {
                free.saturating_sub(spec_bytes)
            } else {
                free
            }
        })
        .collect::<Vec<_>>();
    let ring_slots = if weights.format().is_16bit() {
        let context = cudarc::driver::CudaContext::new(ordinals[0])
            .with_context(|| format!("open CUDA device {}", ordinals[0]))?;
        flyingfish::qwen35::gpu::ring_geom(&context, &weights, config)?.depth
    } else {
        2
    };
    let plans = flyingfish::qwen35::gpu::plan_residency(
        &weights,
        config,
        flyingfish::qwen35::gpu::PlanParams {
            ranges: &ranges,
            max_ctx,
            free: &free,
            ring_slots,
            force_stream: ff_qwen35::gpu::force_stream_requested(),
            granularity: ff_qwen35::gpu::pool_granularity_on(ordinals[0])?,
        },
    )?;
    let residency_ok = if speculative {
        plans
            .iter()
            .all(|plan| plan.residency == flyingfish::qwen35::gpu::Residency::Resident)
    } else {
        plans
            .iter()
            .all(|plan| plan.residency != flyingfish::qwen35::gpu::Residency::Insufficient)
    };
    if speculative && !residency_ok {
        eprintln!(
            "--speculative needs a fully resident plan plus its {} MiB workspace; \
                   this device only admits streaming or too-tight residency",
            spec_bytes / (1 << 20)
        );
    }
    Ok(residency_ok)
}

#[cfg(not(feature = "cuda"))]
fn checkpoint_fits_free_vram(
    _: &[usize],
    _: &Path,
    _: &Qwen35Config,
    _: usize,
    _: bool,
) -> Result<bool> {
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
    host_profile: Option<&'a Path>,
    model_dir: &'a Path,
    config: &'a Qwen35Config,
    ids: &'a [u32],
    pos3: &'a [[i32; 3]],
    vision: Option<&'a (Vec<f32>, VisionGrid)>,
    max_new_tokens: usize,
    speculative: bool,
    timings: &'a mut Timings,
}

#[cfg(feature = "cuda")]
fn generate_cuda(request: CudaDecode<'_>) -> Result<Vec<u32>> {
    use flyingfish::qwen35::gpu::QwenGpu;

    let CudaDecode {
        ordinals,
        host_profile,
        model_dir,
        config,
        ids,
        pos3,
        vision,
        max_new_tokens,
        speculative,
        timings,
    } = request;

    anyhow::ensure!(
        !ordinals.is_empty(),
        "--device cuda:N[,M...] requires at least one ordinal"
    );
    let total = ids.len() + max_new_tokens;
    anyhow::ensure!(
        total <= config.text_config.max_position_embeddings,
        "prompt + generation {total} exceeds the model context {}",
        config.text_config.max_position_embeddings
    );
    let load = Instant::now();
    let weights = Qwen35Weights::open(model_dir)?;
    let mut gpu = if total > 4096 {
        QwenGpu::with_max_ctx(
            ordinals,
            &weights,
            config,
            total.next_power_of_two(),
            ff_qwen35::gpu::force_stream_requested(),
        )?
    } else {
        QwenGpu::new(
            ordinals,
            &weights,
            config,
            ff_qwen35::gpu::force_stream_requested(),
        )?
    };
    let (records, profile_ms) = flyingfish::host_profile::HostProfile::group_records(
        host_profile,
        ordinals[0],
        gpu.forced_group() || weights.format().is_16bit(),
    )?;
    let binary = flyingfish::collect_binary_identity()?;
    gpu.bind_groups(&records, &binary)?;
    if speculative {
        gpu.bind_round(&records, &binary, max_new_tokens + 2)?;
    }
    timings.profile_ms = profile_ms;
    gpu.ctx.stream.synchronize()?;
    timings.load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let prefill_start = Instant::now();
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
    let first = gpu.read_token()?;
    timings.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    timings.prefill_packed_tokens = gpu.prefill_packed_tokens();
    timings.prefill_device_modes = gpu.prefill_modes().into_iter().map(str::to_owned).collect();
    timings.prefill_block = gpu.prefill_block();
    timings.group4 = serde_json::to_value(gpu.group_choices())?;
    let decode_start = Instant::now();
    if speculative {
        let (generated, stats) =
            generate_cuda_speculative(&mut gpu, &weights, config, max_new_tokens)?;
        timings.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        timings.spec = Some(stats);
        timings.group4 = serde_json::to_value(gpu.group_choices())?;
        return Ok(generated);
    }
    let generated = gpu.decode(first, max_new_tokens)?;
    timings.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    timings.stream16 = gpu.stream16_stats();
    Ok(generated)
}

/// Greedy decode with one MTP-drafted token verified per round; the round
/// economics and the enablement rule live in [`flyingfish::qwen35::spec::decode`].
#[cfg(feature = "cuda")]
fn generate_cuda_speculative(
    gpu: &mut flyingfish::qwen35::gpu::QwenGpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    max_new_tokens: usize,
) -> Result<(Vec<u32>, flyingfish::qwen35::spec::SpecStats)> {
    let eos = |token: &u32| config.text_config.eos_token_id.contains(token);
    flyingfish::qwen35::spec::decode(gpu, weights, &eos, max_new_tokens)
}
