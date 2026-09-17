use super::*;
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    /// JSON array of {id, prompt, lyrics, options}; options use Music3 defaults.
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, value_delimiter = ',', required = true)]
    devices: Vec<String>,
    #[arg(long, value_enum, default_value = "pcm16")]
    wav_format: WavSampleFormat,
    #[command(flatten)]
    weights: WeightCacheArgs,
    #[command(flatten)]
    device_cache: DeviceCacheArgs,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    prompt: String,
    lyrics: String,
    #[serde(default)]
    options: Options,
}

#[derive(Serialize)]
struct Completion {
    index: usize,
    id: String,
    device: String,
    elapsed_seconds: f64,
    cache_before: flyingfish::runtime::weights::DeviceCacheStats,
    cache_after: flyingfish::runtime::weights::DeviceCacheStats,
    memory_releases: Vec<flyingfish::music::pipeline::PhaseMemoryRelease>,
    error: Option<String>,
}

fn validate_requests(requests: &[Request]) -> Result<()> {
    ensure!(!requests.is_empty(), "Music3 batch is empty");
    let mut ids = HashSet::new();
    for request in requests {
        ensure!(
            !request.id.is_empty()
                && request.id.len() <= 128
                && request
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "request id must contain 1..=128 ASCII letters, digits, '-' or '_'"
        );
        ensure!(
            ids.insert(&request.id),
            "duplicate request id {}",
            request.id
        );
        flyingfish::music::prompt::format_prompt(&request.prompt, &request.lyrics)?;
        let options = &request.options;
        ensure!(
            options.duration_seconds.is_finite()
                && (0.04..=360.).contains(&options.duration_seconds),
            "invalid duration for {}",
            request.id
        );
        ensure!(
            options.steps > 0 && options.attention_query_chunk > 0,
            "steps and attention chunk must be positive for {}",
            request.id
        );
    }
    Ok(())
}

fn validate_devices(devices: &[String]) -> Result<()> {
    ensure!(!devices.is_empty(), "at least one CUDA device is required");
    let mut ordinals = HashSet::new();
    for device in devices {
        let ordinal = device
            .strip_prefix("cuda:")
            .and_then(|value| value.parse::<usize>().ok())
            .with_context(|| format!("expected cuda:N, got {device}"))?;
        ensure!(
            *device == format!("cuda:{ordinal}"),
            "use canonical cuda:N device names"
        );
        ensure!(ordinals.insert(ordinal), "duplicate device {device}");
    }
    Ok(())
}

pub(super) fn run(args: Args) -> Result<()> {
    validate_devices(&args.devices)?;
    let requests: Vec<Request> = serde_json::from_slice(&std::fs::read(&args.requests)?)?;
    validate_requests(&requests)?;
    let output_dir = resolve_output_outside_model(&args.output_dir, &args.model)?;
    std::fs::create_dir_all(&output_dir)?;
    let report_path = output_dir.join("report.json");
    ensure_new_output(&report_path, "Music3 batch report")?;
    for request in &requests {
        ensure_new_output(&output_dir.join(format!("{}.wav", request.id)), "music WAV")?;
    }
    let max_frames = requests
        .iter()
        .map(|r| (r.options.duration_seconds * 25.) as usize)
        .max()
        .unwrap();
    let max_steps = requests.iter().map(|r| r.options.steps).max().unwrap();
    let cache_policy = args.weights.cache_policy()?;
    let requests = Arc::new(requests);
    let next = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let completions = Mutex::new(Vec::<Completion>::new());
    let started = Instant::now();
    // Workers run at once, so on a shared pool their frame stacks and device
    // charges are all live together. Each integrated worker therefore reserves
    // the concurrent sum of both, and takes a share of what is left rather than
    // all of it. Discrete workers are unaffected: the fold does not apply.
    let concurrent_workers = args.devices.len().min(requests.len()).max(1) as u64;
    let unified_workers = args
        .devices
        .iter()
        .take(requests.len())
        .filter(|name| {
            parse_device(name).is_ok_and(|device| {
                flyingfish::runtime::probe::ResourceSnapshot::capture(Some(&device))
                    .unified_pool_available_bytes()
                    .is_some()
            })
        })
        .count()
        .max(1) as u64;
    let worker_errors = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for name in args.devices.iter().take(requests.len()) {
            let requests = &requests;
            let next = &next;
            let cancelled = &cancelled;
            let completions = &completions;
            let output_dir = &output_dir;
            let args = &args;
            handles.push(scope.spawn(move || -> Result<()> {
                let run = || -> Result<()> {
                    let device = parse_device(name)?;
                    let mut model = Music3::open(
                        &args.model,
                        &device,
                        args.weights.weight_source,
                        cache_policy,
                        DeviceCache::disabled(),
                    )?;
                    let demands = model.residency_demands(max_frames, max_steps, &device)?;
                    let (reserve, host_reserve) = requests.iter().try_fold(
                        (0u64, 0u64),
                        |(device, host), request| -> Result<_> {
                            let memory = model.request_memory(
                                &request.prompt,
                                &request.lyrics,
                                &request.options,
                            )?;
                            Ok((
                                device.max(memory.known_device_reserve_bytes),
                                host.max(memory.frame_stack_host_peak_bytes),
                            ))
                        },
                    )?;
                    eprintln!("{name}: maximum queued Music3 known tensor reserve {reserve} bytes");
                    model.configure_device_cache(decide_auto_residency_with_required_memory(
                        &demands,
                        &device,
                        args.device_cache,
                        reserve.saturating_mul(unified_workers),
                        host_reserve.saturating_mul(concurrent_workers),
                        unified_workers,
                    )?)?;
                    while !cancelled.load(Ordering::Acquire) {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(request) = requests.get(index) else {
                            break;
                        };
                        let cache_before = model.device_cache_stats();
                        let request_started = Instant::now();
                        let result = (|| -> Result<()> {
                            let audio = model.generate_cancellable(
                                &request.prompt,
                                &request.lyrics,
                                &request.options,
                                |_, _, _| {
                                    ensure!(
                                        !cancelled.load(Ordering::Acquire),
                                        "batch cancelled after a worker failed"
                                    );
                                    Ok(())
                                },
                            )?;
                            let waveform = audio.waveform.squeeze(0)?;
                            ensure!(
                                waveform.abs()?.max_all()?.to_scalar::<f32>()?.is_finite()
                                    && waveform.sqr()?.mean_all()?.to_scalar::<f32>()?.is_finite(),
                                "Music3 produced non-finite waveform values"
                            );
                            let path = output_dir.join(format!("{}.wav", request.id));
                            let staging = ArtifactStaging::new_for_path_producer(&path)?;
                            write_wav(
                                staging.producer_path(),
                                &waveform.unsqueeze(1)?,
                                audio.sample_rate,
                                args.wav_format,
                            )?;
                            staging.publish()?;
                            Ok(())
                        })();
                        let error = result.err().map(|error| format!("{error:#}"));
                        if error.is_some() {
                            cancelled.store(true, Ordering::Release);
                        }
                        eprintln!(
                            "{} on {name}: {} ({:.3}s)",
                            request.id,
                            if error.is_some() { "failed" } else { "saved" },
                            request_started.elapsed().as_secs_f64()
                        );
                        completions
                            .lock()
                            .map_err(|_| anyhow::anyhow!("batch results lock poisoned"))?
                            .push(Completion {
                                index,
                                id: request.id.clone(),
                                device: name.clone(),
                                elapsed_seconds: request_started.elapsed().as_secs_f64(),
                                cache_before,
                                cache_after: model.device_cache_stats(),
                                memory_releases: model.memory_releases().to_vec(),
                                error,
                            });
                    }
                    Ok(())
                };
                let result = run();
                if result.is_err() {
                    cancelled.store(true, Ordering::Release);
                }
                result
            }));
        }
        handles
            .into_iter()
            .filter_map(|handle| match handle.join() {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("{error:#}")),
                Err(_) => Some("Music3 worker panicked".to_owned()),
            })
            .collect::<Vec<_>>()
    });
    let mut completions = completions
        .into_inner()
        .map_err(|_| anyhow::anyhow!("batch results lock poisoned"))?;
    completions.sort_by_key(|entry| entry.index);
    let skipped = requests
        .iter()
        .enumerate()
        .filter(|(index, _)| !completions.iter().any(|c| c.index == *index))
        .map(|(_, request)| request.id.as_str())
        .collect::<Vec<_>>();
    let failed = !worker_errors.is_empty()
        || !skipped.is_empty()
        || completions.iter().any(|c| c.error.is_some());
    let report = serde_json::json!({"wall_seconds":started.elapsed().as_secs_f64(),
        "requests":completions,"skipped":skipped,"worker_errors":worker_errors});
    let staging = ArtifactStaging::new(&report_path)?;
    publish_staged_bytes(staging, &serde_json::to_vec_pretty(&report)?)?;
    ensure!(
        !failed,
        "Music3 batch failed; see {}",
        report_path.display()
    );
    println!(
        "saved {} requests; report {}",
        requests.len(),
        report_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_rejects_output_collisions_and_device_aliases() {
        let request: Request =
            serde_json::from_str(r#"{"id":"song","prompt":"guitar","lyrics":"[Instrumental]"}"#)
                .unwrap();
        validate_requests(std::slice::from_ref(&request)).unwrap();
        assert!(validate_requests(&[request.clone(), request.clone()]).is_err());
        for id in ["../song", "", "a/b", "."] {
            let mut invalid = request.clone();
            invalid.id = id.into();
            assert!(validate_requests(&[invalid]).is_err());
        }
        assert!(validate_devices(&["cuda:0".into(), "cuda:0".into()]).is_err());
        assert!(validate_devices(&["cuda:00".into()]).is_err());
        assert!(validate_devices(&["cpu".into()]).is_err());
        validate_devices(&["cuda:0".into(), "cuda:1".into()]).unwrap();
    }
}
