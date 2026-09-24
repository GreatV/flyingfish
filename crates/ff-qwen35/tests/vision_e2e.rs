//! E2E vision acceptance vs the HF fixtures (skips without the model dir).

use ff_core::paths::checkpoint_dir;
use ff_qwen35::config::Qwen35Config;
use ff_qwen35::model::Qwen35Text;
use ff_qwen35::vision::{
    ProcessorConfig, RgbImage, VisionTower, multimodal_positions, preprocess_image,
};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct E2eFixture {
    prompt: String,
    gen_ids: Vec<u32>,
    margins: Vec<f32>,
}

/// CI-unconditional: decode-position arithmetic from the committed rope
/// fixture (130 ids, delta -63 -> 67). The id-level e2e below skips without
/// the model dir.
#[test]
fn decode_rope_pos_matches_fixture_arithmetic() {
    let f: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("src/testdata/vision_rope_fixture.json").unwrap(),
    )
    .unwrap();
    let seq_len = f["input_ids"].as_array().unwrap().len();
    let delta = f["rope_delta"].as_f64().unwrap() as i64;
    let max_pos = f["position_ids"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|axis| axis.as_array().unwrap().iter())
        .filter_map(|v| v.as_i64())
        .max()
        .unwrap();
    // First decode token sits at KV index seq_len; its rope position must be
    // max(prefill)+1 — negative delta included (the usize-clamp regression).
    assert_eq!(
        ff_qwen35::model::decode_rope_pos(seq_len, delta),
        (max_pos + 1) as usize,
        "decode continuation position mismatch"
    );
    assert!(delta < 0, "fixture should exercise the negative-delta path");
}

/// Debug-profile CPU decode runs ~50 minutes; opt in with `--ignored`.
#[test]
#[ignore]
fn image_prompt_decode_matches_hf_fixture() {
    let Some(dir) = checkpoint_dir("Qwen/Qwen3.8-27B-int4") else {
        return;
    };
    let dir = &dir;
    let fixture_path = Path::new("src/testdata/vision_e2e_fixture.json");
    let png = Path::new("src/testdata/vision_test.png");
    if !dir.exists() || !fixture_path.exists() {
        return;
    }
    let fixtures: Vec<E2eFixture> =
        serde_json::from_str(&std::fs::read_to_string(fixture_path).unwrap()).unwrap();
    let config = Qwen35Config::from_model_dir(dir).unwrap();
    let weights = ff_qwen35::weights::Qwen35Weights::open(dir).unwrap();
    let proc = ProcessorConfig::from_model_dir(dir).unwrap();
    let tower = VisionTower::load(&weights, config.vision_config.as_ref().unwrap()).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let image = RgbImage::from_png(png).unwrap();
    let (patches, grid) = preprocess_image(&image, &proc).unwrap();
    let rows = tower.forward(&patches, grid).unwrap();
    let merge = proc.merge_size();
    let n_merged = grid.merged_count(merge).unwrap();
    let image_token = config.image_token_id.unwrap();
    let n_hidden = config.text_config.hidden_size;

    for fixture in &fixtures {
        let templated = format!(
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            fixture.prompt
        );
        let mut ids = tokenizer
            .encode(templated.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let at = ids.iter().position(|&t| t == image_token).unwrap();
        let mut expanded = Vec::with_capacity(ids.len() + n_merged - 1);
        expanded.extend_from_slice(&ids[..at]);
        expanded.extend(std::iter::repeat_n(image_token, n_merged));
        expanded.extend_from_slice(&ids[at + 1..]);
        ids = expanded;
        let types: Vec<u8> = ids.iter().map(|&t| (t == image_token) as u8).collect();
        let (pos3, delta) = multimodal_positions(&types, &[grid], merge).unwrap();

        let mut model = Qwen35Text::load(dir, config.clone()).unwrap();
        let mut pad_row = 0usize;
        let mut hidden = None;
        for (i, &id) in ids.iter().enumerate() {
            let p = [
                pos3[i][0] as usize,
                pos3[i][1] as usize,
                pos3[i][2] as usize,
            ];
            hidden = Some(if id == image_token {
                let row = rows[pad_row * n_hidden..(pad_row + 1) * n_hidden].to_vec();
                pad_row += 1;
                model.forward_vision_row(row, p).unwrap().1
            } else {
                model.forward_at(id, p).unwrap().1
            });
        }
        model.set_mrope_delta(delta);
        // First decode token's rope position must be max(prefill)+1, not
        // the KV index (delta is negative for image prompts).
        let max_prefill = pos3.iter().flatten().max().copied().unwrap() as usize;
        assert_eq!(
            model.next_rope_pos(),
            max_prefill + 1,
            "decode rope position {} != max_prefill+1 {}",
            model.next_rope_pos(),
            max_prefill + 1
        );

        let mut got = Vec::new();
        let mut h = hidden;
        for _ in 0..fixture.gen_ids.len() {
            let logits = model.logits(h.as_ref().unwrap()).unwrap();
            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            got.push(best);
            h = Some(model.forward(best).unwrap());
        }
        let prefix = got
            .iter()
            .zip(&fixture.gen_ids)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!("e2e {:?}: ids {:?}", fixture.prompt, got);
        eprintln!(
            "e2e {:?}: prefix {}/{} (HF margin at flip: {:.3})",
            fixture.prompt,
            prefix,
            fixture.gen_ids.len(),
            fixture.margins.get(prefix).copied().unwrap_or(f32::NAN)
        );
        // Per-prompt measured baselines (flips sit at HF margins <= 0.50,
        // the int4 quality class); GPU produced the same prefixes.
        let baseline = ["Describe this image.", "What colors dominate?"]
            .iter()
            .position(|p| *p == fixture.prompt)
            .map(|i| [17usize, 11][i])
            .unwrap();
        assert!(
            prefix >= baseline,
            "e2e prefix {prefix} < baseline {baseline} — regression below the measured int4 class"
        );
    }
}
