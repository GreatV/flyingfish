//! Teacher-forced full-vocabulary logit dumps and the numerical corpus gate.
//!
//! Dump one case (teacher-forced when an ids file is given):
//!   logit_dump <checkpoint> <out-prefix> <steps> <prompt|@file|@raw:file> [truncate] [ids-file]
//! `@raw:` feeds the file through the tokenizer without the chat template.
//! Rows of `vocab` f32 values: row 0 holds the prefill-end logits, row j > 0
//! the logits after token j. Writes <out-prefix>.bin and <out-prefix>.ids.
//! Greedy mode takes the ids from the tree's own argmax; an argmax
//! disagreement with a fed id is printed to stderr.
//!
//! Self-check that greedy and fed modes agree on the greedy chain:
//!   logit_dump <checkpoint> selfcheck <steps> <prompt|@file|@raw:file>
//!
//! Perplexity pass (reference arm writes .bin; mode arms pass it and
//! write per-row max|Δ| to .d):
//!   logit_dump <checkpoint> ppl <out-prefix> <prompt|@file|@raw:file> <truncate> <rows> [reference.bin]
//!
//! Run the 13-case numerical corpus, comparing against a reference when one
//! is given (nonzero exit on gate failure):
//!   logit_dump <checkpoint> corpus <out-dir> [reference-dir]
//! Multi-GPU: QWEN35_CHECK_DEVICES=0,1.

use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use ff_qwen35::{
    config::{Qwen35Config, chat_prompt},
    gpu::QwenGpu,
    weights::Qwen35Weights,
};
use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("model dir")?;
    let second = args.next().context("out prefix or corpus")?;
    if second == "corpus" {
        let out = args.next().context("output directory")?;
        return run_corpus(
            Path::new(&dir),
            Path::new(&out),
            args.next().map(PathBuf::from),
        );
    }
    if second == "selfcheck" {
        let n: usize = args.next().context("n steps")?.parse()?;
        let prompt_arg = args.next().context("prompt")?;
        return run_selfcheck(Path::new(&dir), n, &prompt_arg);
    }
    if second == "ppl" {
        let out = args.next().context("out prefix")?;
        let prompt_arg = args.next().context("prompt")?;
        let truncate: usize = args.next().context("truncate")?.parse()?;
        let rows: usize = args.next().context("rows")?.parse()?;
        let reference = args.next();
        return run_ppl(
            Path::new(&dir),
            &out,
            &prompt_arg,
            truncate,
            rows,
            reference.as_deref(),
        );
    }
    let n: usize = args.next().context("n steps")?.parse()?;
    let prompt_arg = args.next().context("prompt")?;
    let truncate: Option<usize> = args.next().and_then(|v| v.parse().ok());
    let feed: Option<Vec<u32>> = args
        .next()
        .map(|path| -> Result<Vec<u32>> {
            let file = std::fs::File::open(&path)?;
            BufReader::new(file)
                .lines()
                .map(|line| Ok(line?.parse()?))
                .collect()
        })
        .transpose()?;
    run_dump(Path::new(&dir), &second, n, &prompt_arg, truncate, feed)
}

fn tokenize_prompt(tokenizer: &Tokenizer, prompt_arg: &str) -> Result<Vec<u32>> {
    let (text, raw) = match prompt_arg.strip_prefix("@raw:") {
        Some(path) => (std::fs::read_to_string(path)?, true),
        None => match prompt_arg.strip_prefix('@') {
            Some(path) => (std::fs::read_to_string(path)?, false),
            None => (prompt_arg.to_string(), false),
        },
    };
    let encoded = if raw {
        tokenizer.encode(text.as_str(), false)
    } else {
        tokenizer.encode(chat_prompt(&text).as_str(), false)
    };
    Ok(encoded
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec())
}

fn run_dump(
    dir: &Path,
    out: &str,
    n: usize,
    prompt_arg: &str,
    truncate: Option<usize>,
    feed: Option<Vec<u32>>,
) -> Result<()> {
    let config = Qwen35Config::from_model_dir(dir)?;
    let tokenizer =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ids = tokenize_prompt(&tokenizer, prompt_arg)?;
    if let Some(n) = truncate {
        ids.truncate(n);
    }
    eprintln!("prompt ids: {}", ids.len());
    let weights = Qwen35Weights::open(dir)?;
    let total = ids.len() + n + 4;
    let mut gpu = if total > 4096 {
        QwenGpu::with_max_ctx(
            &[0],
            &weights,
            &config,
            total.next_power_of_two(),
            ff_qwen35::gpu::force_stream_requested(),
        )?
    } else {
        QwenGpu::new(
            &[0],
            &weights,
            &config,
            ff_qwen35::gpu::force_stream_requested(),
        )?
    };
    let mut rows = std::fs::File::create(format!("{out}.bin"))?;
    let mut ids_out = std::fs::File::create(format!("{out}.ids"))?;
    let mut prompt_out = std::fs::File::create(format!("{out}.prompt_ids"))?;
    for id in &ids {
        writeln!(prompt_out, "{id}")?;
    }
    for &id in &ids {
        gpu.push_token(id)?;
    }
    let mut pending = gpu.read_token()?;
    let mut generated = vec![pending];
    let mut nll = 0.0;
    let mut scored = 0usize;
    let mut nll_out = feed
        .is_some()
        .then(|| std::fs::File::create(format!("{out}.nll")))
        .transpose()?;
    let mut row = |gpu: &QwenGpu, rows: &mut std::fs::File| -> Result<()> {
        let logits = gpu.read_logits()?;
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite model logits"
        );
        if let Some(ids) = &feed
            && let Some(&target) = ids.get(scored)
        {
            let row_nll = logsumexp(&logits) - logits[target as usize] as f64;
            nll += row_nll;
            scored += 1;
            if let Some(out) = nll_out.as_mut() {
                writeln!(out, "{row_nll}")?;
            }
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(logits.as_ptr().cast::<u8>(), logits.len() * 4) };
        rows.write_all(bytes)?;
        Ok(())
    };
    row(&gpu, &mut rows)?;
    for step in 1..n {
        let forced = feed.as_ref().map_or(pending, |ids| {
            let id = ids[step - 1];
            if id != pending {
                eprintln!("step {step}: argmax {pending} != fed {id}");
            }
            id
        });
        gpu.push_token(forced)?;
        row(&gpu, &mut rows)?;
        pending = gpu.read_token()?;
        generated.push(pending);
    }
    if feed.is_some() {
        eprintln!(
            "mean_nll: {:.6} over {scored} positions",
            nll / scored as f64
        );
    }
    for id in generated {
        writeln!(ids_out, "{id}")?;
    }
    Ok(())
}

/// Greedy chain must reproduce itself under teacher forcing: run greedy,
/// feed the generated ids back, and require bitwise-identical logit rows.
fn run_selfcheck(dir: &Path, n: usize, prompt_arg: &str) -> Result<()> {
    let tmp = std::env::temp_dir().join(format!("logit-dump-selfcheck-{}", std::process::id()));
    let greedy = tmp.join("greedy");
    let fed = tmp.join("fed");
    let greedy_prefix = greedy.to_string_lossy().into_owned();
    let fed_prefix = fed.to_string_lossy().into_owned();
    std::fs::create_dir_all(&tmp)?;
    let result = (|| -> Result<()> {
        run_dump(dir, &greedy_prefix, n, prompt_arg, None, None)?;
        let feed: Vec<u32> = BufReader::new(std::fs::File::open(format!("{greedy_prefix}.ids"))?)
            .lines()
            .map(|line| Ok(line?.parse()?))
            .collect::<Result<_>>()?;
        run_dump(dir, &fed_prefix, n, prompt_arg, None, Some(feed))?;
        let a = std::fs::read(format!("{greedy_prefix}.bin"))?;
        let b = std::fs::read(format!("{fed_prefix}.bin"))?;
        ensure!(
            a == b,
            "greedy and fed logits diverge: self-check failed ({n} steps)"
        );
        eprintln!("selfcheck: greedy == fed bitwise over {n} steps");
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

/// Perplexity pass: prefill the corpus in blocks and score every row of
/// each block (logits of every position, not per-position decode steps).
/// Row for corpus position p predicts the token at p+1; per-row NLL goes
/// to <out>.nll, argmax ids to <out>.ids, top-2 margins to <out>.margins.
/// The reference arm also writes raw f32 rows to <out>.bin; a mode arm
/// instead passes the reference bin and writes per-row max|Δ| to <out>.d.
fn run_ppl(
    dir: &Path,
    out: &str,
    prompt_arg: &str,
    truncate: usize,
    rows: usize,
    reference: Option<&str>,
) -> Result<()> {
    let config = Qwen35Config::from_model_dir(dir)?;
    let tokenizer =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = tokenize_prompt(&tokenizer, prompt_arg)?;
    ensure!(
        truncate + rows <= ids.len(),
        "corpus of {} ids cannot feed prompt {truncate} + {rows} rows",
        ids.len()
    );
    let weights = Qwen35Weights::open(dir)?;
    let total = truncate + rows + 4;
    let mut gpu = if total > 4096 {
        QwenGpu::with_max_ctx(
            &[0],
            &weights,
            &config,
            total.next_power_of_two(),
            ff_qwen35::gpu::force_stream_requested(),
        )?
    } else {
        QwenGpu::new(
            &[0],
            &weights,
            &config,
            ff_qwen35::gpu::force_stream_requested(),
        )?
    };
    let vocab = config.text_config.vocab_size;
    let mut nll_out = std::fs::File::create(format!("{out}.nll"))?;
    let mut ids_out = std::fs::File::create(format!("{out}.ids"))?;
    let mut margins_out = std::fs::File::create(format!("{out}.margins"))?;
    let mut bin_out = reference
        .is_none()
        .then(|| std::fs::File::create(format!("{out}.bin")))
        .transpose()?;
    let mut ref_in = reference
        .map(|path| {
            std::fs::File::open(path).map(|f| std::io::BufReader::with_capacity(64 << 20, f))
        })
        .transpose()?;
    let mut d_out = reference
        .map(|_| std::fs::File::create(format!("{out}.d")))
        .transpose()?;
    let mut ref_row: Vec<f32> = Vec::new();
    let mut score = |logits: &[f32], target: u32| -> Result<u32> {
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite model logits"
        );
        let z = logsumexp(logits);
        writeln!(nll_out, "{}", z - logits[target as usize] as f64)?;
        let (mut top, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut argmax = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > top {
                second = top;
                top = v;
                argmax = i;
            } else if v > second {
                second = v;
            }
        }
        writeln!(ids_out, "{argmax}")?;
        writeln!(margins_out, "{}", top - second)?;
        if let Some(bin) = bin_out.as_mut() {
            let bytes = unsafe {
                std::slice::from_raw_parts(logits.as_ptr().cast::<u8>(), logits.len() * 4)
            };
            bin.write_all(bytes)?;
        }
        if let (Some(ref_in), Some(d_out)) = (ref_in.as_mut(), d_out.as_mut()) {
            ref_row.resize(logits.len(), 0.0);
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(ref_row.as_mut_ptr().cast::<u8>(), logits.len() * 4)
            };
            std::io::Read::read_exact(ref_in, bytes)?;
            ensure!(
                ref_row.iter().all(|v| v.is_finite()),
                "non-finite reference logits"
            );
            let d = logits
                .iter()
                .zip(ref_row.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            writeln!(d_out, "{d}")?;
        }
        Ok(argmax as u32)
    };
    gpu.push_tokens(&ids[..truncate])?;
    let block = gpu.prefill_block();
    let tail = match truncate % block {
        0 => block,
        t => t,
    };
    let logits = gpu.block_logits(tail)?;
    let mut targets = ids[truncate..truncate + rows].iter();
    score(
        &logits[(tail - 1) * vocab..tail * vocab],
        *targets.next().expect("rows checked"),
    )?;
    for chunk in ids[truncate..truncate + rows].chunks(block) {
        gpu.push_tokens(chunk)?;
        let logits = gpu.block_logits(chunk.len())?;
        for t in 0..chunk.len() {
            let Some(&target) = targets.next() else { break };
            score(&logits[t * vocab..(t + 1) * vocab], target)?;
        }
    }
    eprintln!("ppl: scored {rows} rows over prompt {truncate}");
    Ok(())
}

#[derive(serde::Deserialize)]
struct SavedCase {
    case: String,
    ids: Vec<u32>,
    targets: Vec<u32>,
}

fn logsumexp(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + logits
        .iter()
        .map(|&v| (v as f64 - max).exp())
        .sum::<f64>()
        .ln()
}

fn compare(
    actual: &[f32],
    expected: &[f32],
    vocab: usize,
    targets: &[u32],
) -> Result<serde_json::Value> {
    ensure!(actual.len() == expected.len(), "logit shape changed");
    let bitwise_equal = actual
        .iter()
        .zip(expected)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let mut max_abs = 0.0f64;
    let mut squared = 0.0;
    let mut max_kl = 0.0f64;
    let mut max_nll = 0.0f64;
    let mut top1_equal = true;
    for ((a, b), &target) in actual
        .chunks(vocab)
        .zip(expected.chunks(vocab))
        .zip(targets)
    {
        let za = logsumexp(a);
        let zb = logsumexp(b);
        let mut kl = 0.0;
        for (&x, &y) in a.iter().zip(b) {
            let d = x as f64 - y as f64;
            max_abs = max_abs.max(d.abs());
            squared += d * d;
            kl += (y as f64 - zb).exp() * (y as f64 - zb - x as f64 + za);
        }
        max_kl = max_kl.max(kl);
        max_nll =
            max_nll.max((za - a[target as usize] as f64 - zb + b[target as usize] as f64).abs());
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .unwrap()
                .0
        };
        top1_equal &= argmax(a) == argmax(b);
    }
    let rmse = (squared / actual.len() as f64).sqrt();
    let passed =
        max_abs <= 0.001 && rmse <= 0.0001 && max_kl <= 0.000001 && max_nll <= 0.0001 && top1_equal;
    Ok(serde_json::json!({
        "passed": passed,
        "bitwise_equal": bitwise_equal,
        "max_abs": max_abs,
        "rmse": rmse,
        "max_kl": max_kl,
        "max_nll_delta": max_nll,
        "top1_equal": top1_equal,
    }))
}

fn run_corpus(model: &Path, output: &Path, reference: Option<PathBuf>) -> Result<()> {
    eprintln!("model: {}\noutput: {}", model.display(), output.display());
    if let Some(path) = &reference {
        eprintln!("reference: {}", path.display());
    }
    let saved = reference
        .as_ref()
        .map(|path| -> Result<HashMap<String, SavedCase>> {
            let cases: Vec<SavedCase> =
                serde_json::from_reader(std::fs::File::open(path.join("report.json"))?)?;
            let count = cases.len();
            let cases: HashMap<_, _> = cases
                .into_iter()
                .map(|case| (case.case.clone(), case))
                .collect();
            ensure!(cases.len() == count, "duplicate reference case");
            Ok(cases)
        })
        .transpose()?;
    std::fs::create_dir(output)?;
    let config = Qwen35Config::from_model_dir(model)?;
    let weights = Qwen35Weights::open(model)?;
    let tokenizer =
        Tokenizer::from_file(model.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let prompts = [
        ("paging", "Explain paging to a systems programmer."),
        (
            "code",
            "Write a Python function that reverses a linked list. Explain its complexity.",
        ),
        ("chinese", "请解释虚拟内存、页表和缺页异常之间的关系。"),
        (
            "math",
            "Prove that the sum of the first n positive integers is n(n+1)/2.",
        ),
    ];
    let mut cases = prompts
        .iter()
        .map(|(name, text)| Ok((name.to_string(), encode(&tokenizer, &chat_prompt(text))?)))
        .collect::<Result<Vec<_>>>()?;
    let long = encode(
        &tokenizer,
        &chat_prompt(
            &"A page table maps virtual pages to physical frames. A translation cache avoids repeated table walks. "
                .repeat(100),
        ),
    )?;
    for n in [31, 32, 127, 128, 129, 255, 256, 257, 1024] {
        ensure!(long.len() >= n, "boundary prompt too short");
        cases.push((format!("length-{n}"), long[..n].to_vec()));
    }
    let targets = encode(
        &tokenizer,
        "Virtual memory provides each process with its own address space.",
    )?;
    let targets = &targets[..8];
    let devices = std::env::var("QWEN35_CHECK_DEVICES").unwrap_or_else(|_| "0".to_owned());
    eprintln!("devices: {devices}");
    let devices: Vec<usize> = devices
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut gpu = None;
    let mut records = Vec::new();
    let mut passed = true;
    for (name, ids) in cases {
        if let Some(saved) = &saved {
            let expected = saved.get(&name).context("reference case missing")?;
            ensure!(
                expected.ids == ids && expected.targets == targets,
                "reference inputs changed for {name}"
            );
        }
        drop(gpu.take());
        gpu = Some(QwenGpu::new(
            &devices,
            &weights,
            &config,
            ff_qwen35::gpu::force_stream_requested(),
        )?);
        let gpu = gpu.as_mut().context("missing request model")?;
        let start = Instant::now();
        gpu.push_tokens(&ids)?;
        gpu.read_token()?;
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut logits = Vec::new();
        let mut nll = 0.0;
        for &target in targets {
            let values = gpu.read_logits()?;
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "non-finite logits in {name}"
            );
            nll += logsumexp(&values) - values[target as usize] as f64;
            logits.extend(values);
            gpu.push_token(target)?;
        }
        let vocab = config.text_config.vocab_size;
        let tensor = Tensor::from_vec(logits.clone(), (targets.len(), vocab), &Device::Cpu)?;
        candle_core::safetensors::save(
            &HashMap::from([("logits", tensor)]),
            output.join(format!("{name}.safetensors")),
        )?;
        let comparison = if let Some(root) = &reference {
            let tensors = candle_core::safetensors::load(
                root.join(format!("{name}.safetensors")),
                &Device::Cpu,
            )?;
            let expected = tensors.get("logits").context("reference has no logits")?;
            ensure!(
                expected.dims() == [targets.len(), vocab],
                "reference logit shape changed"
            );
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            let metrics = compare(&logits, &expected, vocab, targets)?;
            passed &= metrics["passed"].as_bool().unwrap();
            Some(metrics)
        } else {
            None
        };
        eprintln!("{name}: {prefill_ms:.1} ms, comparison {comparison:?}");
        records.push(serde_json::json!({"case":name,"ids":ids,"targets":targets,"prefill_ms":prefill_ms,"mean_nll":nll/targets.len() as f64,"comparison":comparison}));
    }
    serde_json::to_writer_pretty(
        std::fs::File::create_new(output.join("report.json"))?,
        &records,
    )?;
    ensure!(passed, "prefill logit gate failed; see report.json");
    Ok(())
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec())
}
