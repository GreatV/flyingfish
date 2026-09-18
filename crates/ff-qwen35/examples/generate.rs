//! Greedy decode on the int4 CPU path; prints prompt/gen token ids as JSON
//! for comparison against scripts/qwen35_reference.py (HF bf16).
//!
//! Usage: generate <model_dir> [n_tokens] [prompt]

use ff_qwen35::config::Qwen35Config;
use ff_qwen35::model::Qwen35Text;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "models/Qwen/Qwen3.8-27B-int4".to_string());
    let n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    let prompt = args
        .next()
        .unwrap_or_else(|| "Explain paging to a systems programmer.".to_string());

    let dir = Path::new(&dir);
    let config = Qwen35Config::from_model_dir(dir)?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let mut model = Qwen35Text::load(dir, config.clone())?;

    // The checkpoint's chat_template.jinja prepends this system block
    // (verified against apply_chat_template 2026-09-17). Divergent prompts
    // were the whole "first token mismatch" bug class — keep byte-exact.
    let templated = format!(
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let mut ids = tokenizer
        .encode(templated.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if let Some(n) = std::env::var("QWEN35_PREFIX")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        ids.truncate(n);
    }
    println!("prompt ids: {} {:?}", ids.len(), ids);
    #[cfg(feature = "cuda")]
    if std::env::var_os("QWEN35_GPU").is_some() {
        let weights = ff_qwen35::weights::Qwen35Weights::open(dir)?;
        let mut gpu = ff_qwen35::gpu::QwenGpu::new(&weights, &config)?;
        let spec_mode = std::env::var("QWEN35_MTP").unwrap_or_default() == "spec";
        let started = Instant::now();
        for &id in &ids {
            gpu.push_token(id)?;
        }
        println!("prefill: {:.2}s (gpu)", started.elapsed().as_secs_f32());
        if spec_mode {
            let mut spec = ff_qwen35::spec::QwenSpec::new(&gpu, &weights)?;
            let mut generated: Vec<u32> = Vec::new();
            let (mut accepts, mut rejects) = (0usize, 0usize);
            let mut pending = gpu.read_token()?;
            spec.draft(&gpu, Some(&gpu.hidden), pending)?;
            let started = Instant::now();
            let (mut t_verify, mut t_draft, mut t_rounds) = (0.0f64, 0.0f64, 0usize);
            while generated.len() < n {
                // Stage the real token into the A-side input buffer.
                gpu.ctx
                    .stream
                    .memcpy_htod(&[pending as i32], &mut gpu.next_token)?;
                let tv = Instant::now();
                let (a, b) = spec.verify_round(&mut gpu)?;
                t_verify += tv.elapsed().as_secs_f64();
                t_rounds += 1;
                if a == spec.draft_id {
                    accepts += 1;
                    generated.push(pending);
                    generated.push(spec.draft_id);
                    gpu.ctx.glue_inc(&mut gpu.pos)?;
                    gpu.ctx.glue_inc(&mut gpu.pos)?;
                    gpu.position += 2;
                    pending = b;
                    let td = Instant::now();
                    spec.draft(&gpu, None, b)?;
                    t_draft += td.elapsed().as_secs_f64();
                } else {
                    rejects += 1;
                    generated.push(pending);
                    spec.reject_restore(&mut gpu)?;
                    gpu.ctx.glue_inc(&mut gpu.pos)?;
                    gpu.position += 1;
                    pending = a;
                    let td = Instant::now();
                    spec.draft(&gpu, Some(&gpu.hidden), a)?;
                    t_draft += td.elapsed().as_secs_f64();
                }
            }
            let el = started.elapsed().as_secs_f64();
            let total = accepts * 2 + rejects;
            println!(
                "spec decode: {total} tokens in {el:.2}s = {:.1} tok/s; accepts {accepts} rejects {rejects}",
                total as f64 / el
            );
            println!(
                "  verify {:.1} ms/round, draft {:.1} ms/round, rest {:.1} ms/round",
                t_verify * 1000.0 / t_rounds as f64,
                t_draft * 1000.0 / t_rounds as f64,
                (el - t_verify - t_draft) * 1000.0 / t_rounds as f64
            );
            println!("gen ids: {generated:?}");
            println!("text: {:?}", tokenizer.decode(&generated, false));
            return Ok(());
        }
        let mut generated = vec![gpu.read_token()?];
        for step in 1..n {
            let tick = Instant::now();
            gpu.step()?;
            generated.push(gpu.read_token()?);
            println!(
                "token {step}: id {} in {:.3}s",
                generated[step],
                tick.elapsed().as_secs_f32()
            );
        }
        println!("gen ids: {generated:?}");
        println!("text: {:?}", tokenizer.decode(&generated, false));
        return Ok(());
    }

    let started = Instant::now();
    let mut hidden = None;
    for &id in &ids {
        hidden = Some(model.forward(id)?);
    }
    println!("prefill: {:.1}s", started.elapsed().as_secs_f32());
    if std::env::var_os("QWEN35_DUMP_MIXER").is_some() {
        let mut bytes = Vec::new();
        for h in &model.mixer_dump {
            for v in h {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write("/tmp/qwen_rust_mixer.bin", &bytes)?;
        println!("mixer dumped {} vecs", model.mixer_dump.len());
    }
    if std::env::var_os("QWEN35_DUMP").is_some() {
        let mut bytes = Vec::new();
        for h in &model.dump {
            for v in h {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write("/tmp/qwen_rust_dump.bin", &bytes)?;
        println!(
            "dumped {} layers x {} floats",
            model.dump.len(),
            model.dump[0].len()
        );
    }

    // QWEN35_MTP=1: measure acceptance off-path. QWEN35_MTP=spec: full
    // speculative loop on CPU (no time savings — it gates that speculation
    // is output-neutral vs plain greedy before the GPU batch-2 port).
    let mtp_mode = std::env::var("QWEN35_MTP").unwrap_or_default();
    let measure_mtp = mtp_mode == "1";
    let spec_mtp = mtp_mode == "spec";
    let (mut accepts, mut drafts) = (0usize, 0usize);
    let mut pending: Option<u32> = None;
    if spec_mtp {
        let (mut h, mut normed) = (None, hidden.unwrap());
        let mut generated: Vec<u32> = Vec::new();
        let mut accepts = 0usize;
        let mut drafts = 0usize;
        // First real token from the prefill's hidden.
        let mut pending_tok = {
            let logits = model.logits(&normed)?;
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32
        };
        // First draft needs h after prefill — not exposed; draft starts
        // after the first generated token is processed.
        let mut draft: Option<u32> = None;
        while generated.len() < n {
            let (h1, normed1) = model.forward_raw(pending_tok)?;
            generated.push(pending_tok);
            let logits1 = model.logits(&normed1)?;
            let actual_next = logits1
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            if let Some(d) = draft {
                drafts += 1;
                if d == actual_next {
                    // Accept: process the draft (a real token) this round.
                    accepts += 1;
                    generated.push(d);
                    let (h2, normed2) = model.forward_raw(d)?;
                    let logits2 = model.logits(&normed2)?;
                    let next_tok = logits2
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0 as u32;
                    let pos = model.position() - 1;
                    let dl = model.mtp_draft(&h2, next_tok, pos)?;
                    draft = Some(
                        dl.iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap()
                            .0 as u32,
                    );
                    pending_tok = next_tok;
                    let _ = h1;
                    continue;
                }
            }
            let pos = model.position() - 1;
            let dl = model.mtp_draft(&h1, actual_next, pos)?;
            draft = Some(
                dl.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u32,
            );
            pending_tok = actual_next;
            h = Some(h1);
            normed = normed1;
        }
        let _ = (h, normed);
        println!(
            "spec acceptance: {accepts}/{drafts} = {:.2}",
            accepts as f64 / drafts.max(1) as f64
        );
        println!("gen ids: {generated:?}");
        println!("text: {:?}", tokenizer.decode(&generated, false));
        return Ok(());
    }
    let mut generated = Vec::new();
    let mut next_hidden = hidden;
    for step in 0..n {
        let tick = Instant::now();
        let logits = model.logits(next_hidden.as_ref().unwrap())?;
        let mut order: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let (best, margin) = (order[0].0 as u32, order[0].1 - order[1].1);
        generated.push(best);
        if let Some(d) = pending {
            drafts += 1;
            accepts += (d == best) as usize;
        }
        println!(
            "token {step}: id {best} margin {margin:.4} in {:.2}s",
            tick.elapsed().as_secs_f32()
        );
        if measure_mtp {
            let (h, normed) = model.forward_raw(best)?;
            let pos = model.position() - 1;
            let dlogits = model.mtp_draft(&h, best, pos)?;
            pending = Some(
                dlogits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u32,
            );
            next_hidden = Some(normed);
        } else {
            next_hidden = Some(model.forward(best)?);
        }
    }
    if measure_mtp {
        println!(
            "mtp acceptance: {accepts}/{drafts} = {:.2}",
            accepts as f64 / drafts.max(1) as f64
        );
    }
    println!("gen ids: {generated:?}");
    println!("text: {:?}", tokenizer.decode(&generated, false));
    Ok(())
}
