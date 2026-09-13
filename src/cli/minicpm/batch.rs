use super::*;
use flyingfish::minicpm::memory::RequestGeometry;
use flyingfish::runtime::artifact::ArtifactStaging;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    draft_model: Option<PathBuf>,
    /// JSON array of {id, prompt, max_new_tokens, raw}.
    #[arg(long)]
    requests: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    /// One reusable worker per canonical cpu or cuda:N device.
    #[arg(long, value_delimiter = ',', required = true)]
    devices: Vec<String>,
    #[arg(long, default_value_t = 32)]
    attention_query_chunk_size: usize,
    #[arg(long)]
    batch_invariant_decode: bool,
    #[command(flatten)]
    weights: WeightCacheArgs,
    #[command(flatten)]
    device_cache: DeviceCacheArgs,
}

fn default_tokens() -> usize {
    128
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    prompt: String,
    #[serde(default = "default_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    raw: bool,
}

#[derive(Serialize)]
struct Completion {
    index: usize,
    id: String,
    device: String,
    started_offset_seconds: f64,
    finished_offset_seconds: f64,
    elapsed_seconds: f64,
    prompt_tokens: usize,
    output_tokens: usize,
    required_tensor_bytes: u64,
    cache_before: flyingfish::runtime::weights::DeviceCacheStats,
    cache_after: flyingfish::runtime::weights::DeviceCacheStats,
    error: Option<String>,
}

fn validate_devices(devices: &[String]) -> Result<()> {
    anyhow::ensure!(
        !devices.is_empty(),
        "at least one worker device is required"
    );
    let mut seen = HashSet::new();
    for name in devices {
        let valid = name == "cpu"
            || name
                .strip_prefix("cuda:")
                .and_then(|value| value.parse::<usize>().ok())
                .is_some_and(|ordinal| *name == format!("cuda:{ordinal}"));
        anyhow::ensure!(
            valid && seen.insert(name),
            "use unique canonical cpu or cuda:N devices"
        );
    }
    Ok(())
}

fn validate_requests(requests: &[Request]) -> Result<()> {
    anyhow::ensure!(!requests.is_empty(), "MiniCPM batch is empty");
    let mut ids = HashSet::new();
    for request in requests {
        anyhow::ensure!(
            !request.id.is_empty()
                && request.id.len() <= 128
                && request
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "request id must contain 1..=128 ASCII letters, digits, '-' or '_'"
        );
        anyhow::ensure!(
            ids.insert(&request.id),
            "duplicate request id {}",
            request.id
        );
        anyhow::ensure!(
            request.max_new_tokens > 0,
            "max_new_tokens must be positive"
        );
    }
    Ok(())
}

struct CancelOnFailure<'a> {
    cancelled: &'a AtomicBool,
    completed: bool,
}
impl Drop for CancelOnFailure<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

pub(super) fn run(args: Args) -> Result<()> {
    let started = Instant::now();
    validate_devices(&args.devices)?;
    anyhow::ensure!(
        args.attention_query_chunk_size > 0,
        "attention query chunk must be positive"
    );
    let requests: Vec<Request> = serde_json::from_slice(&std::fs::read(&args.requests)?)?;
    validate_requests(&requests)?;
    let config = Config::read(&args.model)?;
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let inputs = requests
        .iter()
        .map(|r| encode_prompt(&tokenizer, &config, &r.prompt, r.raw, r.max_new_tokens))
        .collect::<Result<Vec<_>>>()?;
    let geometries = requests
        .iter()
        .zip(&inputs)
        .map(|(r, t)| RequestGeometry {
            prompt_tokens: t.len(),
            max_new_tokens: r.max_new_tokens,
            attention_query_chunk_size: args.attention_query_chunk_size,
            batch_invariant_decode: args.batch_invariant_decode,
        })
        .collect::<Vec<_>>();
    let output = resolve_output_outside_model(&args.output_dir, &args.model)?;
    if let Some(draft) = &args.draft_model {
        resolve_output_outside_model(&output, draft)?;
    }
    std::fs::create_dir_all(&output)?;
    let report_path = output.join("report.json");
    ensure_new_output(&report_path, "MiniCPM batch report")?;
    let request_dir = resolve_output_outside_model(&output.join("requests"), &args.model)?;
    if let Some(draft) = &args.draft_model {
        resolve_output_outside_model(&request_dir, draft)?;
    }
    std::fs::create_dir_all(&request_dir)?;
    for r in &requests {
        ensure_new_output(
            &request_dir.join(format!("{}.json", r.id)),
            "MiniCPM request output",
        )?;
    }
    let next = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let completions = Mutex::new(Vec::<Completion>::new());
    let worker_errors = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for name in args.devices.iter().take(requests.len()) {
            let (args, requests, inputs, geometries, config, tokenizer) =
                (&args, &requests, &inputs, &geometries, &config, &tokenizer);
            let (next, cancelled, completions, request_dir) =
                (&next, &cancelled, &completions, &request_dir);
            handles.push((name.clone(), scope.spawn(move || -> Result<()> {
                let mut guard = CancelOnFailure { cancelled, completed: false };
                let mut worker = Worker::open(WorkerOptions {
                    model: &args.model, draft_model: args.draft_model.as_deref(), config: config.clone(),
                    device: parse_device(name)?, weights: args.weights, device_cache: args.device_cache,
                    query_chunk: args.attention_query_chunk_size, batch_invariant: args.batch_invariant_decode,
                }, geometries)?;
                while !cancelled.load(Ordering::Acquire) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(request) = requests.get(index) else { break; };
                    let cache_before = worker.cache.stats();
                    let began = Instant::now();
                    eprintln!("{} on {name}: started", request.id);
                    let mut ids = Vec::new();
                    let result = (|| -> Result<()> {
                        anyhow::ensure!(!cancelled.load(Ordering::Acquire), "batch cancelled before starting request after a worker failed");
                        let speculation = worker.generate(&inputs[index], request.max_new_tokens, |token| {
                            anyhow::ensure!(!cancelled.load(Ordering::Acquire), "batch cancelled at token emission after a worker failed");
                            ids.push(token);
                            Ok(())
                        })?;
                        // Decoding the whole sequence once, rather than concatenating
                        // streamed pieces, is what keeps `text` a faithful reading of
                        // `token_ids`: a `DecodeStream` withholds a piece whose bytes do
                        // not yet form a character, so a request that stops mid-sequence
                        // would otherwise be saved a character short of its own ids.
                        let text = tokenizer.decode(&ids, true)
                            .map_err(|e| anyhow::anyhow!("decode output: {e}"))?;
                        let artifact = serde_json::json!({"id":request.id,"prompt_tokens":inputs[index].len(),
                            "token_ids":ids,"text":text,"speculation":speculation});
                        let path = request_dir.join(format!("{}.json", request.id));
                        publish_staged_bytes(ArtifactStaging::new(&path)?, &serde_json::to_vec_pretty(&artifact)?)?;
                        Ok(())
                    })();
                    let error = result.err().map(|e| format!("{e:#}"));
                    if error.is_some() { cancelled.store(true, Ordering::Release); }
                    let finished = Instant::now();
                    eprintln!("{} on {name}: {} ({:.3}s)", request.id,
                        if error.is_some() {"failed"} else {"saved"}, finished.duration_since(began).as_secs_f64());
                    completions.lock().map_err(|_| anyhow::anyhow!("batch results lock poisoned"))?.push(Completion {
                        index, id: request.id.clone(), device: name.clone(), elapsed_seconds: finished.duration_since(began).as_secs_f64(),
                        started_offset_seconds: began.duration_since(started).as_secs_f64(),
                        finished_offset_seconds: finished.duration_since(started).as_secs_f64(),
                        prompt_tokens: inputs[index].len(), output_tokens: ids.len(), required_tensor_bytes: worker.required_tensor_bytes,
                        cache_before, cache_after: worker.cache.stats(), error,
                    });
                }
                guard.completed = true;
                Ok(())
            })));
        }
        handles
            .into_iter()
            .filter_map(|(device, handle)| match handle.join() {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(format!("{device}: {e:#}")),
                Err(_) => Some(format!("MiniCPM worker on {device} panicked")),
            })
            .collect::<Vec<_>>()
    });
    let mut completions = completions
        .into_inner()
        .map_err(|_| anyhow::anyhow!("batch results lock poisoned"))?;
    completions.sort_by_key(|c| c.index);
    let done = completions.iter().map(|c| c.index).collect::<HashSet<_>>();
    let skipped = requests
        .iter()
        .enumerate()
        .filter(|(index, _)| !done.contains(index))
        .map(|(_, r)| r.id.as_str())
        .collect::<Vec<_>>();
    let failed = !worker_errors.is_empty()
        || !skipped.is_empty()
        || completions.iter().any(|c| c.error.is_some());
    let report = serde_json::json!({"schema_version":1,"wall_seconds":started.elapsed().as_secs_f64(),
        "model":args.model.display().to_string(),"draft_model":args.draft_model.as_ref().map(|p| p.display().to_string()),
        "devices":args.devices,"attention_query_chunk_size":args.attention_query_chunk_size,
        "batch_invariant_decode":args.batch_invariant_decode,
        "requests":completions,"skipped":skipped,"worker_errors":worker_errors});
    publish_staged_bytes(
        ArtifactStaging::new(&report_path)?,
        &serde_json::to_vec_pretty(&report)?,
    )?;
    anyhow::ensure!(
        !failed,
        "MiniCPM batch failed; see {}",
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
    fn queue_and_device_validation_prevent_output_collisions() {
        let request: Request = serde_json::from_str(r#"{"id":"a","prompt":"hello"}"#).unwrap();
        assert!(validate_requests(std::slice::from_ref(&request)).is_ok());
        assert!(validate_requests(&[request.clone(), request.clone()]).is_err());
        assert!(
            validate_requests(&[Request {
                id: "../a".into(),
                ..request
            }])
            .is_err()
        );
        assert!(validate_devices(&["cpu".into(), "cuda:0".into()]).is_ok());
        for devices in [
            vec![],
            vec!["cuda:00".into()],
            vec!["cpu".into(), "cpu".into()],
            vec!["auto".into()],
        ] {
            assert!(validate_devices(&devices).is_err());
        }
    }
}
