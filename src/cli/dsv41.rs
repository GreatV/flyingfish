use crate::cli::output_hygiene::{
    ensure_new_output, publish_staged_bytes, resolve_output_outside_model,
};
use crate::cli::{CachePolicy, Result, WeightCacheArgs, WeightSource, kit};
use anyhow::{Context, bail};
use candle_core::{Device, Tensor};
use clap::Subcommand;
use flyingfish::dsv41::config::DeepseekV41Config;
use flyingfish::dsv41::weights::TransformerLoader;
use rand::{Rng as _, SeedableRng, rngs::StdRng};
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
        #[arg(long, default_value = "cpu")]
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

/// Greedy pick from a logits row.
fn greedy(values: &[f32]) -> Result<u32> {
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "model produced non-finite logits"
    );
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(right.1).expect("finite logits"))
        .map(|(index, _)| index as u32)
        .context("model produced no logits")
}

/// Nucleus sampling with a seeded generator; mirrors the GLM reference
/// (softmax over the vocabulary, keep the smallest prefix reaching `top_p`).
fn sample_token(logits: &Tensor, temperature: f64, top_p: f64, rng: &mut StdRng) -> Result<u32> {
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "sampling temperature must be finite and non-negative"
    );
    anyhow::ensure!(
        top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
        "top-p must lie in (0, 1]"
    );
    if temperature == 0.0 {
        return greedy(&logits.flatten_all()?.to_vec1::<f32>()?);
    }
    anyhow::ensure!(
        temperature.recip().is_finite(),
        "sampling temperature must be safely invertible"
    );
    let values = logits
        .to_dtype(candle_core::DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    anyhow::ensure!(!values.is_empty(), "logits are empty");
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "logits contain a non-finite value"
    );
    let inverse = 1.0 / temperature;
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    let maximum = values[order[0]] as f64 * inverse;
    let mut probabilities = order
        .iter()
        .map(|&index| (values[index] as f64 * inverse - maximum).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    anyhow::ensure!(
        total.is_finite() && total > 0.0,
        "softmax normalization is invalid"
    );
    for probability in &mut probabilities {
        *probability /= total;
    }
    let mut retained = 0usize;
    let mut cumulative = 0.0;
    for probability in &probabilities {
        cumulative += *probability;
        retained += 1;
        if cumulative >= top_p {
            break;
        }
    }
    let retained_total = probabilities[..retained].iter().sum::<f64>();
    let target = rng.random::<f64>() * retained_total;
    let mut cumulative = 0.0;
    for (rank, probability) in probabilities[..retained].iter().enumerate() {
        cumulative += *probability;
        if target <= cumulative {
            return u32::try_from(order[rank]).context("token id exceeds u32");
        }
    }
    u32::try_from(order[retained - 1]).context("token id exceeds u32")
}

fn reject_unsupported_device(device: &str) -> Result<()> {
    // auto lands on CPU here: nothing GPU-side is wired, so the automatic
    // choice must resolve rather than fail.
    if device == "auto" {
        return Ok(());
    }
    anyhow::ensure!(
        device == "cpu",
        "CUDA and Metal inference for dsv41 are not wired yet; pass --device cpu"
    );
    Ok(())
}

fn reject_nondefault_weights(weights: &WeightCacheArgs) -> Result<()> {
    anyhow::ensure!(
        weights.weight_source == WeightSource::Mmap,
        "streamed and in-memory weight sources are not wired for dsv41 yet; keep the mmap default"
    );
    anyhow::ensure!(
        weights.host_cache_mib.is_none(),
        "host cache ceilings are not wired for dsv41 yet; drop --host-cache-mib"
    );
    anyhow::ensure!(
        weights.cache_policy()? == CachePolicy::new(1),
        "weight-cache granularity other than the shard default is not wired for dsv41 yet"
    );
    Ok(())
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
                bail!(
                    "--image generation is not wired yet; the vision tower runs only in capture-parity fixtures"
                );
            }
            reject_unsupported_device(&device.device)?;
            reject_nondefault_weights(&weights)?;
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
            // The sequence budget covers the prompt plus every requested
            // generation step; it must fit both the request limit and the
            // model's trained span.
            let budget = limits.max_new_tokens.get();
            let needed = ids
                .len()
                .checked_add(budget)
                .context("request length overflows usize")?;
            anyhow::ensure!(
                needed <= max_context,
                "prompt of {} tokens plus {} new tokens exceeds the {}-token request limit",
                ids.len(),
                budget,
                max_context
            );
            anyhow::ensure!(
                needed <= config.text_config.max_position_embeddings,
                "prompt of {} tokens plus {} new tokens exceeds the model's {}-token span",
                ids.len(),
                budget,
                config.text_config.max_position_embeddings
            );
            let max_seq = needed;
            let telemetry_monitor = output
                .telemetry_json
                .as_ref()
                .map(|_| {
                    flyingfish::runtime::telemetry::TelemetryMonitor::start(
                        Some(Device::Cpu),
                        std::time::Duration::from_millis(250),
                    )
                })
                .transpose()?;
            {
                let primary = output
                    .output
                    .as_deref()
                    .map(|path| resolve_output_outside_model(path, &model_dir))
                    .transpose()?;
                if let Some(primary) = &primary {
                    ensure_new_output(primary, "generation output")?;
                }
                let telemetry = output
                    .telemetry_json
                    .as_deref()
                    .map(|path| resolve_output_outside_model(path, &model_dir))
                    .transpose()?;
                if let Some(telemetry) = &telemetry {
                    ensure_new_output(telemetry, "telemetry output")?;
                }
                if let (Some(telemetry), Some(primary)) = (&telemetry, &primary) {
                    anyhow::ensure!(
                        telemetry != primary,
                        "telemetry output conflicts with the primary output: {}",
                        telemetry.display()
                    );
                }
            }
            let mut transformer = loader.load(Some(&tokenizer), max_seq)?;
            let device = Device::Cpu;
            let mut rng = StdRng::seed_from_u64(sampling.seed);
            let mut position = 0usize;
            let chunk = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let (mut token, logits) = transformer.forward(&chunk, position)?;
            if sampling.temperature != 0.0 {
                token = sample_token(&logits, sampling.temperature, sampling.top_p, &mut rng)?;
            }
            // The prompt already answered: an immediate EOS ends the turn
            // before any decode step runs.
            if token == config.eos_token_id {
                let text = tokenizer
                    .decode(&[] as &[u32], true)
                    .map_err(|error| anyhow::anyhow!("decode output: {error}"))?;
                emit_result(
                    &output,
                    &model_dir,
                    &text,
                    &[],
                    sampling.temperature == 0.0,
                    telemetry_monitor,
                )?;
                return Ok(());
            }
            let mut generated = vec![token];
            position = ids.len();
            while generated.len() < budget {
                let step = Tensor::from_vec(vec![token], (1, 1), &device)?;
                let (next, logits) = transformer.forward(&step, position)?;
                token = if sampling.temperature != 0.0 {
                    sample_token(&logits, sampling.temperature, sampling.top_p, &mut rng)?
                } else {
                    next
                };
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
            let greedy = sampling.temperature == 0.0;
            emit_result(
                &output,
                &model_dir,
                &text,
                &generated,
                greedy,
                telemetry_monitor,
            )
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
            reject_unsupported_device(&device)?;
            reject_nondefault_weights(&weights)?;
            let effort = flyingfish::dsv41::encoding::ReasoningEffort::parse(&reasoning_effort)
                .map_err(|error| anyhow::anyhow!(error))?;
            let loader = TransformerLoader::open(&model_dir)?;
            admit_resident_footprint(&loader)?;
            let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .map_err(|error| anyhow::anyhow!("load tokenizer: {error}"))?;
            let encoded = flyingfish::dsv41::encoding::chat_prompt(&prompt, None, true, effort);
            let ids = tokenizer
                .encode(encoded, false)
                .map_err(|error| anyhow::anyhow!("tokenize prompt: {error}"))?
                .get_ids()
                .to_vec();
            let limit = max_context_tokens.get();
            anyhow::ensure!(!ids.is_empty(), "prompt tokenized to an empty sequence");
            // Truncating would silently drop the assistant/thinking header
            // the template appended; refuse an oversized prompt instead. The
            // model's trained span is a second gate, same as generate.
            anyhow::ensure!(
                ids.len() <= limit,
                "prompt of {} tokens exceeds the {}-token capture limit",
                ids.len(),
                limit
            );
            anyhow::ensure!(
                ids.len() <= loader.config().text_config.max_position_embeddings,
                "prompt of {} tokens exceeds the model's {}-token span",
                ids.len(),
                loader.config().text_config.max_position_embeddings
            );
            let mut transformer = loader.load(Some(&tokenizer), limit.max(ids.len()))?;
            let device = Device::Cpu;
            let chunk = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let mut snapshots = Vec::new();
            let mut progress = |captured: usize| {
                eprintln!("parity prefill: captured block {captured}");
            };
            let observer = (!no_progress).then_some(&mut progress as &mut dyn FnMut(usize));
            let (token, logits) =
                transformer.forward_with_capture(&chunk, 0, Some(&mut snapshots), observer)?;
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
            // The argmax alone cannot see a final-norm/head error; keep the
            // full logits row in the capture, as the GLM parity path does.
            tensors.insert("logits".to_owned(), logits);
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
            // Serialize to memory first, then publish through the staging
            // path: a failed write never replaces a complete capture.
            let bytes = safetensors::serialize(
                tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
                None,
            )
            .map_err(|error| anyhow::anyhow!("serialize parity capture: {error}"))?;
            let staging = flyingfish::runtime::artifact::ArtifactStaging::new(&output)
                .with_context(|| format!("failed to stage parity capture {}", output.display()))?;
            publish_staged_bytes(staging, &bytes)?;
            println!("saved parity capture to {}", output.display());
            Ok(())
        }
        Dsv41Command::ReplayRouting { .. } => {
            bail!("dsv41 replay-routing is a schema placeholder")
        }
    }
}

fn emit_result(
    output: &kit::OutputArgs,
    model_dir: &PathBuf,
    text: &str,
    generated: &[u32],
    greedy: bool,
    telemetry_monitor: Option<flyingfish::runtime::telemetry::TelemetryMonitor>,
) -> Result<()> {
    let telemetry_path = output
        .telemetry_json
        .as_ref()
        .map(|path| resolve_output_outside_model(path, model_dir))
        .transpose()?;
    if let Some(path) = &telemetry_path {
        ensure_new_output(path, "telemetry output")?;
    }
    let rendered = if output.json {
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "model_family": "deepseek_v41",
            "model": model_dir,
            "text": text,
            "generated_token_ids": generated,
            "sampling": {
                "greedy": greedy,
            },
        }))?
    } else {
        text.to_owned()
    };
    if let Some(path) = output.output.as_ref() {
        let path = resolve_output_outside_model(path, model_dir)?;
        ensure_new_output(&path, "dsv41 output")?;
        let staging = flyingfish::runtime::artifact::ArtifactStaging::new(&path)
            .with_context(|| format!("failed to stage dsv41 output {}", path.display()))?;
        publish_staged_bytes(staging, rendered.as_bytes())?;
        eprintln!("saved dsv41 output to {}", path.display());
    } else {
        println!("{rendered}");
    }
    if let (Some(path), Some(monitor)) = (telemetry_path, telemetry_monitor) {
        let report = monitor.finish()?;
        let bytes = serde_json::to_vec_pretty(&report)?;
        let staging = flyingfish::runtime::artifact::ArtifactStaging::new(&path)
            .with_context(|| format!("failed to stage dsv41 telemetry {}", path.display()))?;
        publish_staged_bytes(staging, &bytes)?;
        eprintln!("saved dsv41 runtime telemetry to {}", path.display());
    } else {
        eprintln!(
            "generated {} tokens ({})",
            generated.len(),
            if greedy { "greedy" } else { "sampled" }
        );
    }
    Ok(())
}

/// Refuse loudly when the dequantized resident set cannot fit this host.
fn admit_resident_footprint(loader: &TransformerLoader) -> Result<()> {
    let needed = loader.resident_f32_bytes()?;
    let host = host_available_bytes("/proc/meminfo")?;
    // meminfo is host-wide; a cgroup-v2 container may hold a smaller budget.
    let snapshot = flyingfish::runtime::probe::ResourceSnapshot::capture(Some(&Device::Cpu));
    let available = snapshot
        .cgroup_v2_memory_available_bytes
        .map_or(host, |limit| host.min(limit));
    anyhow::ensure!(
        needed <= available,
        "the resident F32 load needs {} GiB but only {} GiB is available; the routed experts need streaming, which is not wired yet",
        needed / (1 << 30),
        available / (1 << 30),
    );
    Ok(())
}

fn host_available_bytes(source: &str) -> Result<u64> {
    let text = std::fs::read_to_string(source)
        .with_context(|| format!("read {source} to measure host memory"))?;
    let kib = text
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kib| kib.parse::<u64>().ok())
        .with_context(|| format!("parse MemAvailable from {source}"))?;
    Ok(kib * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_rejects_out_of_range_top_p_and_temperature() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        for (temperature, top_p) in [
            (1.0f64, 0.0f64),
            (1.0, -1.0),
            (1.0, f64::NAN),
            (1.0, 1.5),
            (-1.0, 0.95),
            (f64::INFINITY, 0.95),
        ] {
            let error = sample_token(&logits, temperature, top_p, &mut rng)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("top-p") || error.contains("temperature"),
                "({temperature}, {top_p}): {error}"
            );
        }
        // Valid edges still sample.
        sample_token(&logits, 1.0, 1.0, &mut rng).unwrap();
    }

    #[test]
    fn unreadable_meminfo_fails_closed() {
        let error = host_available_bytes("/nonexistent/meminfo").unwrap_err();
        assert!(
            error.to_string().contains("measure host memory"),
            "{error:#}"
        );
        let scratch = std::env::temp_dir().join(format!("ff-dsv41-meminfo-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::write(
            scratch.join("meminfo"),
            b"MemTotal:       1 kB
",
        )
        .unwrap();
        let error = host_available_bytes(scratch.join("meminfo").to_str().unwrap()).unwrap_err();
        assert!(
            error.to_string().contains("parse MemAvailable"),
            "{error:#}"
        );
        std::fs::remove_dir_all(&scratch).ok();
    }
}
