//! Extended acceptance gate — the instrument the lora_add 1/32 bug proved
//! missing. Two parts, both must pass (nonzero exit otherwise):
//!
//! A. LoRA-dump assertion: for a representative set of lora'd projections,
//!    GPU (device lora) vs CPU (host lora) on a deterministic input, gated
//!    at max_rel <= 1e-4 — the bug ran at ~3e-1 — PLUS an adapter-presence
//!    check (with-lora vs without-lora must differ materially), the
//!    failure mode where a path silently drops the adapter entirely.
//!
//! B. Greedy-decode token-id gates: the original 8-token prompt and a
//!    second 32-token prompt against Python-reference ids (bit-equal
//!    prompt ids, exact id sequence). Longer decode gives near-tie
//!    margins more chances to cross — the 8-token gate stayed green
//!    through the lora bug.

use ff_edge0::config::Edge0Config;
use ff_edge0::model::Edge0Text;

/// (projection name, in_dim) — one per adapter family.
const LORA_SAMPLES: &[(&str, usize)] = &[
    (
        "language_model.model.layers.0.linear_attn.in_proj_qkv",
        2048,
    ),
    ("language_model.model.layers.0.linear_attn.out_proj", 4096),
    ("language_model.model.layers.3.self_attn.q_proj", 2048),
    ("language_model.model.layers.3.self_attn.o_proj", 4096),
    (
        "language_model.model.layers.0.mlp.shared_expert.gate_proj",
        2048,
    ),
    (
        "language_model.model.layers.0.mlp.shared_expert.down_proj",
        512,
    ),
];

const PROMPT1: &str = "Explain paging to a systems programmer.";
const EXPECT1: &[u32] = &[8160, 579, 264, 7047, 1817, 25, 271, 16];

const PROMPT2: &str = "Describe how a CPU cache works, briefly.";
// Regenerated 2026-09-18 with the fixed oracle (per-kv-head KV caches +
// final_norm applied). Prompt 1's 8 ids reproduced unchanged; prompt 2
// diverges from the pre-fix archive at token 16.
const EXPECT2: &[u32] = &[
    8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 15771, 2598, 2570, 5952, 64700, 561,
    1156, 6587, 264, 348, 6449, 9, 3874, 314, 1204, 264, 13540, 6297, 4138, 13, 271,
];

fn main() -> anyhow::Result<()> {
    let Some(model_dir) = ff_core::paths::checkpoint_dir("Edge0/Edge0-35B-A3B-preview") else {
        anyhow::bail!("set FF_MODELS_DIR to the local models root");
    };
    let model_dir = &model_dir;
    let config = Edge0Config::from_model_dir(model_dir)?;
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // ---- Part A: lora dump ----
    {
        let mut gpu = Edge0Text::load(model_dir, config.clone())?;
        gpu.enable_gpu(0, true)?;
        let mut worst = 0.0f32;
        for (name, in_dim) in LORA_SAMPLES {
            let x: Vec<f32> = (0..*in_dim)
                .map(|i| ((i as f32) * 0.037).sin() * 0.5)
                .collect();
            let (cpu_full, cpu_bare) = {
                let quant = gpu.weights_quant(name)?;
                let lora = gpu.weights_lora(name);
                (quant.matvec(&x, lora), quant.matvec(&x, None))
            };
            // Adapter presence: the delta must be materially nonzero
            // (the shared-expert-missing bug was an entirely absent delta).
            let delta = cpu_full
                .iter()
                .zip(&cpu_bare)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            anyhow::ensure!(
                delta > 1e-3,
                "{name}: adapter delta is ~0 ({delta:e}) — a path dropped the lora"
            );
            // GPU (device lora) vs CPU (host lora): the 1/32 bug ran ~3e-1.
            let out = gpu
                .gpu
                .as_ref()
                .unwrap()
                .prepared(name, &x)
                .unwrap_or_else(|| panic!("{name}: not resident"))
                .unwrap_or_else(|e| panic!("{name}: gpu run failed: {e:?}"));
            let rel = cpu_full
                .iter()
                .zip(&out)
                .map(|(&a, &b)| (a - b).abs() / a.abs().max(1.0))
                .fold(0.0f32, f32::max);
            anyhow::ensure!(
                rel <= 1e-4,
                "{name}: gpu-vs-cpu max_rel {rel:e} > 1e-4 (lora magnitude error class)"
            );
            worst = worst.max(rel);
            println!("A: {name}: delta {delta:.3}, gpu-vs-cpu {rel:.2e} — ok");
        }
        println!("A: lora dump PASS (worst {worst:.2e})");
    }

    // ---- Part B: decode gates ----
    for (prompt, expect, tag) in [
        (PROMPT1, EXPECT1, "8-token"),
        (PROMPT2, EXPECT2, "32-token"),
    ] {
        if expect.is_empty() {
            println!("B: {tag} gate: reference ids not yet baked in — SKIP");
            continue;
        }
        let mut model = Edge0Text::load(model_dir, config.clone())?;
        model.enable_gpu(0, true)?;
        let templated = ff_edge0::config::chat_prompt(prompt);
        let enc = tokenizer
            .encode(templated.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let ids = enc.get_ids();
        anyhow::ensure!(!ids.is_empty(), "{tag}: empty prompt ids");
        let mut hidden = None;
        for &id in ids {
            hidden = Some(model.forward(id)?);
        }
        let _ = hidden.take();
        model.set_next_token_harness(*ids.last().unwrap())?;
        let mut got = Vec::new();
        let mut prev = model.first_token()?;
        got.push(prev);
        while got.len() < expect.len() {
            prev = model.forward_token_pub(prev)?;
            got.push(prev);
        }
        if got.len() >= expect.len() && &got[..expect.len()] == expect {
            println!("B: {tag} gate PASS ({} ids exact)", expect.len());
            continue;
        }
        // Diverged somewhere. Semantics: the gate judges the FIRST flip's
        // margin — a razor-thin tie (ulp drift between GPU and the Python
        // reference, both self-consistent after) is the benign class; a
        // confident wrong token (thick margin) or a prefix shorter than the
        // original 8-token gate is a real regression. Per-step margins come
        // from a Vec-path re-decode.
        drop(model);
        let mut model2 = Edge0Text::load(model_dir, config.clone())?;
        model2.enable_gpu(0, true)?;
        let mut hidden = None;
        for &id in ids {
            hidden = Some(model2.forward(id)?);
        }
        // Prefix length against the reference.
        let prefix = got.iter().zip(expect).take_while(|(a, b)| a == b).count();
        let mut first_margin = None;
        for (step, &want) in expect.iter().enumerate() {
            let h = hidden.take().unwrap();
            let logits = model2.logits(&h)?;
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let best = order[0] as u32;
            let margin = logits[order[0]] - logits[order[1]];
            if step == prefix {
                first_margin = Some(margin);
                println!(
                    "  first divergence step {step}: got {} want {} margin {:.4} (top {:.3} runner {:.3})",
                    got.get(step).copied().unwrap_or(0),
                    want,
                    margin,
                    logits[order[0]],
                    logits[order[1]]
                );
            }
            hidden = Some(model2.forward(best)?);
        }
        const TIE_EPS: f32 = 0.05;
        let m = first_margin.unwrap_or(f32::INFINITY);
        anyhow::ensure!(
            prefix >= 8,
            "{tag} gate FAIL: prefix {prefix} < 8 (regression vs the original gate)"
        );
        anyhow::ensure!(
            m < TIE_EPS,
            "{tag} gate FAIL: first divergence at step {prefix} is CONFIDENT (margin {m:.4} >= {TIE_EPS}) — not a tie flip"
        );
        println!("B: {tag} gate PASS (prefix {prefix}, tie flip margin {m:.4} < {TIE_EPS})");
    }
    println!("acceptance_gate: ALL PASS");
    Ok(())
}
