use anyhow::{Result, ensure};
use clap::Args as ClapArgs;
use std::num::NonZeroUsize;
use std::path::PathBuf;

#[derive(Debug, ClapArgs)]
pub(super) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(
        long,
        required_unless_present = "prompts_file",
        conflicts_with = "prompts_file"
    )]
    prompt: Option<String>,
    #[arg(
        long,
        conflicts_with = "prompt",
        help = "JSON array of prompts processed sequentially by the same model workers"
    )]
    prompts_file: Option<PathBuf>,
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        num_args = 1,
        help = "Ordered CUDA devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3"
    )]
    devices: Vec<String>,
    #[arg(
        long,
        value_enum,
        help = "Override the measured boundary transport; peer still requires bidirectional validation"
    )]
    transport: Option<flyingfish::glm::partition::GlmPartitionTransport>,
    #[arg(
        long,
        help = "Host profile containing calibrated streaming reader counts"
    )]
    host_profile: Option<PathBuf>,
    #[arg(
        long,
        help = "Linux benchmark only: evict and verify checkpoint file-cache pages before loading"
    )]
    resource_cold_cache: bool,
    #[arg(long, default_value = "16")]
    max_new_tokens: NonZeroUsize,
    #[arg(
        long,
        default_value = "2048",
        help = "Hard request limit; memory admission uses the supplied prompts and output budget"
    )]
    max_context_tokens: NonZeroUsize,
    #[arg(long, default_value = "max", value_parser = ["low","high","max"])]
    reasoning_effort: String,
    #[arg(long, default_value_t = 1.)]
    temperature: f64,
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true,
        help = "Retain static weights; absent selects by capacity, =false disables")]
    resident_static: Option<bool>,
    #[arg(
        long,
        help = "Expert-cache ceiling per device in MiB; absent selects by capacity, zero disables"
    )]
    expert_cache_mib: Option<u64>,
    #[arg(long, value_enum, default_value = "per-layer-split")]
    expert_cache_layout: flyingfish::glm::ExpertCacheLayout,
    #[arg(long)]
    cpu_fp8_dequantization: bool,
    #[arg(
        long,
        conflicts_with = "cpu_fp8_dequantization",
        help = "Use reusable pinned buffers for FP8 uploads (experimental)"
    )]
    pinned_fp8_transfer: bool,
    #[arg(long)]
    host_cache_mib: Option<u64>,
    #[arg(long)]
    no_progress: bool,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(
        long,
        help = "Write the request, rank policies and result to this new path"
    )]
    execution_manifest: Option<PathBuf>,
    #[arg(
        long,
        help = "One JSON file with separate per-device telemetry reports"
    )]
    telemetry_json: Option<PathBuf>,
}

fn ordinals(values: &[String]) -> Result<Vec<usize>> {
    ensure!(
        values.len() >= 2,
        "--devices requires at least two CUDA devices"
    );
    let mut seen = std::collections::BTreeSet::new();
    values
        .iter()
        .map(|value| {
            let ordinal = value
                .strip_prefix("cuda:")
                .and_then(|v| v.parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("expected cuda:N, found {value:?}"))?;
            ensure!(
                format!("cuda:{ordinal}") == *value && seen.insert(ordinal),
                "CUDA device names must be canonical and distinct"
            );
            Ok(ordinal)
        })
        .collect()
}

#[cfg(any(feature = "cuda", test))]
fn request_bounds(prompts: &[usize], output: usize, limit: usize) -> Result<(usize, usize)> {
    use anyhow::Context;
    let largest = *prompts.iter().max().context("GLM prompt queue is empty")?;
    ensure!(
        output > 0 && prompts.iter().all(|&n| n > 0),
        "GLM token budgets must be positive"
    );
    let total = largest
        .checked_add(output)
        .context("GLM request context overflow")?;
    ensure!(
        total <= limit,
        "GLM prompt and output exceed request context"
    );
    Ok((largest, total))
}

pub(super) fn run(args: Args) -> Result<()> {
    let ordinals = ordinals(&args.devices)?;
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (args, ordinals);
        anyhow::bail!("GLM multi-device generation requires the cuda build feature");
    }
    #[cfg(feature = "cuda")]
    run_cuda(args, ordinals)
}

#[cfg(feature = "cuda")]
fn run_cuda(args: Args, ordinals: Vec<usize>) -> Result<()> {
    use anyhow::Context;
    use candle_core::Device;
    use flyingfish::{
        glm::{GlmGenerationOptions, LayerPartitionOptions, LayerPartitionedGlm},
        runtime::{
            artifact::ArtifactStaging, probe::HardwareFingerprint, telemetry::TelemetryMonitor,
            weights::CachePolicy,
        },
    };
    use serde_json::json;
    use std::time::Duration;
    let batch = args.prompts_file.is_some();
    let prompts: Vec<String> = match (&args.prompt, &args.prompts_file) {
        (Some(prompt), None) => vec![prompt.clone()],
        (None, Some(path)) => serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("read prompt queue {}", path.display()))?,
        )
        .context("prompt queue must be a JSON array of strings")?,
        _ => anyhow::bail!("supply --prompt or --prompts-file"),
    };
    ensure!(!prompts.is_empty(), "GLM prompt queue is empty");
    let model_config = flyingfish::glm::config::GlmConfig::from_model_dir(&args.model)?;
    for prompt in &prompts {
        super::glm::validate_generation_request(
            prompt,
            args.max_new_tokens.get(),
            args.max_context_tokens.get(),
            &args.reasoning_effort,
            args.temperature,
            args.top_p,
            model_config.text_config.index_topk,
        )?;
    }
    let output = super::resolve_optional_new_output(args.output, &args.model)?;
    let manifest = super::resolve_optional_new_output(
        args.execution_manifest
            .or_else(|| output.as_ref().map(|p| p.with_extension("manifest.json"))),
        &args.model,
    )?;
    let telemetry = super::resolve_optional_new_output(args.telemetry_json, &args.model)?;
    super::glm::ensure_distinct_outputs(&[
        ("output", output.as_deref()),
        ("manifest", manifest.as_deref()),
        ("telemetry", telemetry.as_deref()),
    ])?;
    let request_outputs = if batch {
        output
            .as_ref()
            .map(|path| {
                (0..prompts.len())
                    .map(|index| path.with_extension(format!("request-{index:04}.json")))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    for path in &request_outputs {
        super::ensure_new_output(path, "GLM request result")?;
    }
    let mut all_outputs = vec![
        ("output", output.as_deref()),
        ("manifest", manifest.as_deref()),
        ("telemetry", telemetry.as_deref()),
    ];
    all_outputs.extend(
        request_outputs
            .iter()
            .map(|path| ("request result", Some(path.as_path()))),
    );
    super::glm::ensure_distinct_outputs(&all_outputs)?;
    let cache_policy = match args.host_cache_mib {
        None => CachePolicy::new(1),
        Some(mib) => {
            ensure!(mib > 0, "host cache ceiling must be positive");
            CachePolicy::unbounded_units().with_max_bytes(super::mib_to_bytes(mib)?)
        }
    };
    let devices = ordinals
        .iter()
        .map(|&ordinal| Device::new_cuda(ordinal))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let io_readers = super::glm::report_host_profile(args.host_profile.as_deref(), &devices[0])?
        .map(|profile| profile.reader_counts(&args.model))
        .transpose()?;
    let hardware = devices
        .iter()
        .map(HardwareFingerprint::collect)
        .collect::<Result<Vec<_>>>()?;
    let monitors = if telemetry.is_some() {
        devices
            .iter()
            .map(|device| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![]
    };
    let cache_preparation = args
        .resource_cold_cache
        .then(|| flyingfish::resource_policy::prepare_cold_cache(&args.model))
        .transpose()?;
    let mut engine = LayerPartitionedGlm::prepare(
        &args.model,
        devices,
        LayerPartitionOptions {
            transport: args.transport,
            io_readers,
            expert_cache_bytes_per_device: usize::try_from(super::mib_to_bytes(
                args.expert_cache_mib.unwrap_or(0),
            )?)?,
            resident_static: args.resident_static.unwrap_or(false),
            cache_policy,
            cache_layout: args.expert_cache_layout,
            cpu_fp8_dequantization: args.cpu_fp8_dequantization,
            pinned_fp8_transfer: args.pinned_fp8_transfer,
            max_context_tokens: Some(args.max_context_tokens.get()),
        },
    )?;
    let prompt_tokens = prompts
        .iter()
        .map(|prompt| -> Result<usize> {
            let ids = engine.tokenize(prompt, &args.reasoning_effort)?;
            Ok(ids.len())
        })
        .collect::<Result<Vec<_>>>()?;
    let (largest_prompt, planned_context) = request_bounds(
        &prompt_tokens,
        args.max_new_tokens.get(),
        args.max_context_tokens.get(),
    )?;
    engine.limit_context(planned_context)?;
    if args.resident_static.is_none() || args.expert_cache_mib.is_none() {
        engine.configure_automatic_residency(
            largest_prompt,
            args.resident_static.is_none(),
            args.expert_cache_mib.is_none(),
        )?;
    }
    let admission = engine.admission(largest_prompt)?;
    let policy = engine.policy().clone();
    let mut request = json!({
        "planned_context_tokens":engine.context_bound(),
        "max_new_tokens":args.max_new_tokens.get(),"max_context_tokens":args.max_context_tokens.get(),
        "reasoning_effort":args.reasoning_effort,"temperature":args.temperature,"top_p":args.top_p,"seed":args.seed});
    if batch {
        request["prompts"] = json!(prompts);
        request["prompt_tokens"] = json!(prompt_tokens);
    } else {
        request["prompt"] = json!(prompts[0]);
        request["prompt_tokens"] = json!(prompt_tokens[0]);
    }
    eprintln!(
        "GLM layer partition: {} devices, host budget {} bytes",
        ordinals.len(),
        admission.required_host_bytes
    );
    let write = |path: &std::path::Path, bytes: &[u8]| -> Result<()> {
        let staging =
            ArtifactStaging::new(path).with_context(|| format!("stage {}", path.display()))?;
        super::output_hygiene::publish_staged_bytes(staging, bytes)?;
        Ok(())
    };
    let options = GlmGenerationOptions {
        max_new_tokens: args.max_new_tokens.get(),
        max_context_tokens: args.max_context_tokens.get(),
        reasoning_effort: args.reasoning_effort,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
        progress: !args.no_progress,
    };
    let started = std::time::Instant::now();
    let mut single_generation = None;
    let mut results = Vec::new();
    let mut failure = None::<String>;
    for (index, prompt) in prompts.iter().enumerate() {
        if failure.is_some() {
            results.push(json!({"index":index,"status":"skipped"}));
            continue;
        }
        let request_started = std::time::Instant::now();
        if batch {
            eprintln!("GLM request {}/{}: started", index + 1, prompts.len());
        }
        match engine.generate(prompt, &options) {
            Ok(generation) => {
                if batch {
                    eprintln!(
                        "GLM request {}/{}: completed in {:.3}s",
                        index + 1,
                        prompts.len(),
                        request_started.elapsed().as_secs_f64()
                    );
                    let mut result = json!({
                        "schema_version":1,"index":index,"status":"completed",
                        "model":args.model,"policy":policy,
                        "request":{
                            "prompt":prompt,"max_new_tokens":options.max_new_tokens,
                            "max_context_tokens":options.max_context_tokens,
                            "reasoning_effort":options.reasoning_effort,
                            "temperature":options.temperature,"top_p":options.top_p,"seed":options.seed
                        },
                        "elapsed_seconds":request_started.elapsed().as_secs_f64(),
                        "generation":generation,"artifact":request_outputs.get(index)
                    });
                    if let Some(path) = request_outputs.get(index)
                        && let Err(error) = write(path, &serde_json::to_vec(&result)?)
                    {
                        let error = format!("publish request {index}: {error:#}");
                        result["status"] = json!("publication_failed");
                        result["error"] = json!(error);
                        failure = Some(error);
                    }
                    results.push(result);
                } else {
                    single_generation = Some(generation);
                }
            }
            Err(error) if !batch => return Err(error),
            Err(error) => {
                let error = format!("{error:#}");
                eprintln!(
                    "GLM request {}/{}: failed: {error}",
                    index + 1,
                    prompts.len()
                );
                results.push(json!({"index":index,"status":"failed",
                    "elapsed_seconds":request_started.elapsed().as_secs_f64(),"error":error}));
                failure = Some(error);
            }
        }
    }
    let reports = monitors
        .into_iter()
        .enumerate()
        .map(|(rank, monitor)| {
            Ok(json!({"rank":rank,"ordinal":ordinals[rank],"report":monitor.finish()?}))
        })
        .collect::<Result<Vec<_>>>()?;
    let record = if batch {
        json!({"schema_version":2,"policy":policy,"request":request,"cache_preparation":cache_preparation,
            "hardware":hardware,"admission":admission,"results":results,
            "wall_seconds":started.elapsed().as_secs_f64(),"error":failure})
    } else {
        json!({"schema_version":1,"policy":policy,"request":request,"cache_preparation":cache_preparation,
            "hardware":hardware,"admission":admission,"generation":single_generation})
    };
    let canonical = serde_json::to_vec(&record)?;
    if let Some(path) = manifest {
        write(&path, &canonical)?;
        eprintln!("execution report {}", path.display());
    }
    if let Some(path) = telemetry {
        write(
            &path,
            &serde_json::to_vec_pretty(&json!({"schema_version":1,"devices":reports}))?,
        )?;
    }
    let rendered = if args.json || batch {
        serde_json::to_string_pretty(&json!({"manifest":record}))?
    } else {
        single_generation
            .context("missing GLM generation result")?
            .text
    };
    if let Some(path) = output {
        write(&path, rendered.as_bytes())?;
        println!("saved {}", path.display());
    } else {
        println!("{rendered}");
    }
    if let Some(error) = failure {
        anyhow::bail!("GLM prompt queue failed: {error}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_list_rejects_aliases_duplicates_and_non_cuda_before_opening_devices() {
        for list in [
            vec!["cuda:0"],
            vec!["cuda:0", "cuda:0"],
            vec!["cuda:0", "cpu"],
            vec!["cuda:01", "cuda:2"],
        ] {
            assert!(ordinals(&list.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
        }
        assert_eq!(
            ordinals(&["cuda:2".into(), "cuda:0".into()]).unwrap(),
            [2, 0]
        );
    }

    #[test]
    fn request_bounds_cover_the_whole_queue_and_enforce_the_limit() {
        assert_eq!(request_bounds(&[20, 50, 10], 32, 2048).unwrap(), (50, 82));
        assert_eq!(request_bounds(&[20], 32, 52).unwrap(), (20, 52));
        assert!(request_bounds(&[20, 50], 32, 81).is_err());
        assert!(request_bounds(&[usize::MAX], 1, usize::MAX).is_err());
        assert!(request_bounds(&[], 32, 2048).is_err());
        assert!(request_bounds(&[0, 20], 32, 2048).is_err());
        assert!(request_bounds(&[20], 0, 2048).is_err());
    }
}
