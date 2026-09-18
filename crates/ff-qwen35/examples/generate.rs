//! Greedy decode on the int4 CPU path; prints prompt/gen token ids as JSON
//! for comparison against scripts/qwen35_reference.py (HF bf16).
//!
//! Usage: generate <model_dir> [n_tokens] [prompt]

use anyhow::Context as _;
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
    let image_path = args.next();

    let dir = Path::new(&dir);
    let config = Qwen35Config::from_model_dir(dir)?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let mut model = Qwen35Text::load(dir, config.clone())?;

    // The checkpoint's chat_template.jinja prepends this system block
    // (verified against apply_chat_template 2026-09-17). Divergent prompts
    // were the whole "first token mismatch" bug class — keep byte-exact.
    // Vision: preprocess + tower forward (CPU f32); the <|image_pad|>
    // placeholder expands to the merged-token count.
    let vision = match &image_path {
        Some(path) => {
            anyhow::ensure!(
                std::env::var("QWEN35_MTP").unwrap_or_default().is_empty(),
                "image prompts do not support QWEN35_MTP (spec is text-only)"
            );
            anyhow::ensure!(
                std::env::var_os("QWEN35_PREFIX").is_none(),
                "QWEN35_PREFIX with an image is unsupported"
            );
            let proc = ff_qwen35::vision::ProcessorConfig::from_model_dir(dir)?;
            let img = ff_qwen35::vision::RgbImage::from_png(path)?;
            let (patches, grid) = ff_qwen35::vision::preprocess_image(&img, &proc)?;
            let n_merged = grid.merged_count(proc.merge_size())?;
            let weights = ff_qwen35::weights::Qwen35Weights::open(dir)?;
            let tower = ff_qwen35::vision::VisionTower::load(
                &weights,
                config.vision_config.as_ref().context("no vision_config")?,
            )?;
            let started = Instant::now();
            anyhow::ensure!(
                config.vision_config.as_ref().unwrap().out_hidden_size
                    == config.text_config.hidden_size,
                "tower out_hidden {} != text hidden {}",
                config.vision_config.as_ref().unwrap().out_hidden_size,
                config.text_config.hidden_size
            );
            let rows = tower.forward(&patches, grid)?;
            println!(
                "vision: grid [{}, {}, {}] -> {n_merged} tokens in {:.1}s",
                grid.temporal,
                grid.height,
                grid.width,
                started.elapsed().as_secs_f32()
            );
            Some((rows, grid))
        }
        None => None,
    };
    let templated = if vision.is_some() {
        format!(
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
    } else {
        format!(
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
    };
    let mut ids = tokenizer
        .encode(templated.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    // Expand the placeholder; build per-token mrope positions.
    let (pos3, mrope_delta) = if let Some((_, grid)) = &vision {
        let image_token = config.image_token_id.context("no image_token_id")?;
        let merge = config
            .vision_config
            .as_ref()
            .map(|v| v.spatial_merge_size)
            .unwrap_or(2);
        let n_merged = grid.merged_count(merge)?;
        let at = ids
            .iter()
            .position(|&t| t == image_token)
            .context("no <|image_pad|> in the tokenized prompt")?;
        let mut expanded = Vec::with_capacity(ids.len() + n_merged - 1);
        expanded.extend_from_slice(&ids[..at]);
        expanded.extend(std::iter::repeat_n(image_token, n_merged));
        expanded.extend_from_slice(&ids[at + 1..]);
        ids = expanded;
        let types: Vec<u8> = ids.iter().map(|&t| (t == image_token) as u8).collect();
        let (p, d) = ff_qwen35::vision::multimodal_positions(&types, &[*grid], merge)?;
        (p, d)
    } else {
        (Vec::new(), 0)
    };

    if let Some(n) = std::env::var("QWEN35_PREFIX")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        ids.truncate(n);
    }
    // After QWEN35_PREFIX truncation; the cap is the attn_scores kernel's
    // shared-memory bound.
    let total_ctx = ids.len() + n + 4;
    anyhow::ensure!(
        total_ctx <= 8192,
        "prompt + generation {total_ctx} exceeds the 8192 kernel cap"
    );
    println!("prompt ids: {} {:?}", ids.len(), ids);
    #[cfg(feature = "cuda")]
    if std::env::var_os("QWEN35_GPU").is_some() {
        let weights = ff_qwen35::weights::Qwen35Weights::open(dir)?;
        let mut gpu = if total_ctx > 4096 {
            ff_qwen35::gpu::QwenGpu::with_max_ctx(&weights, &config, total_ctx.next_power_of_two())?
        } else {
            ff_qwen35::gpu::QwenGpu::new(&weights, &config)?
        };
        let spec_mode = std::env::var("QWEN35_MTP").unwrap_or_default() == "spec";
        let started = Instant::now();
        if let Some((rows, _)) = &vision {
            let n_hidden = config.text_config.hidden_size;
            let image_token = config.image_token_id.unwrap();
            let mut pad_row = 0usize;
            for (i, &id) in ids.iter().enumerate() {
                let p = pos3[i];
                if id == image_token {
                    let row = &rows[pad_row * n_hidden..(pad_row + 1) * n_hidden];
                    pad_row += 1;
                    gpu.push_vision_row(row, p)?; // splice the tower row
                } else {
                    gpu.push_token_at(id, p)?;
                }
            }
        } else {
            for &id in &ids {
                gpu.push_token(id)?;
            }
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
                    gpu.ctx.glue_inc3(&mut gpu.rope_pos)?;
                    gpu.ctx.glue_inc3(&mut gpu.rope_pos)?;
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
                    gpu.ctx.glue_inc3(&mut gpu.rope_pos)?;
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
    if let Some((rows, _)) = &vision {
        let n_hidden = config.text_config.hidden_size;
        let image_token = config.image_token_id.unwrap();
        let mut pad_row = 0usize;
        for (i, &id) in ids.iter().enumerate() {
            let p = [
                pos3[i][0] as usize,
                pos3[i][1] as usize,
                pos3[i][2] as usize,
            ];
            hidden = Some(if id == image_token {
                let row = rows[pad_row * n_hidden..(pad_row + 1) * n_hidden].to_vec();
                pad_row += 1;
                model.forward_vision_row(row, p)?.1
            } else {
                model.forward_at(id, p)?.1
            });
        }
        model.set_mrope_delta(mrope_delta);
    } else {
        for &id in &ids {
            hidden = Some(model.forward(id)?);
        }
    }
    println!("prefill: {:.1}s", started.elapsed().as_secs_f32());
    // Dump path comes from the env var's VALUE (unset = no dump).
    if let Some(path) = std::env::var_os("QWEN35_DUMP_MIXER") {
        let mut bytes = Vec::new();
        for h in &model.mixer_dump {
            for v in h {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes)?;
        println!(
            "mixer dumped {} vecs -> {}",
            model.mixer_dump.len(),
            path.to_string_lossy()
        );
    }
    if let Some(path) = std::env::var_os("QWEN35_DUMP") {
        let mut bytes = Vec::new();
        for h in &model.dump {
            for v in h {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes)?;
        println!(
            "dumped {} layers x {} floats -> {}",
            model.dump.len(),
            model.dump[0].len(),
            path.to_string_lossy()
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
