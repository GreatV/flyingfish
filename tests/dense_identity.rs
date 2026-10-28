#![cfg(any(feature = "cuda", feature = "rocm"))]

use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::{Settings, SpecBudget, TreeBuilder},
    kv::KvFormat,
    model::Model,
};
use std::{io::Write, path::PathBuf};

#[test]
#[ignore = "requires GPU lease, FF_ARGMAX_MODEL, FLYINGFISH_RUNTIME_DIR and FF_DENSE_EVIDENCE"]
fn dense_model_identity_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let output = PathBuf::from(std::env::var("FF_DENSE_EVIDENCE")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: KvFormat::Bf16,
    };
    #[cfg(feature = "cuda")]
    let chunk = 16;
    #[cfg(feature = "rocm")]
    let chunk = 33;
    #[cfg(feature = "rocm")]
    drop(Model::calibrate(
        &target,
        0,
        256,
        Some(chunk),
        settings.clone(),
    )?);
    let mut model = Model::load(&target, 0, 256, Some(chunk), settings)?;
    ensure!(
        model.config.hidden_size == 2048
            && model.config.num_hidden_layers == 42
            && model.config.vocab_size == 130560,
        "dense MiniCPM5-2B geometry required"
    );
    model.check_ready()?;
    for rows in [1, 17, 129] {
        let ids: Vec<_> = (0..rows)
            .map(|i| ((i * 269 + 23) % 130560) as u32)
            .collect();
        let mut eager = Vec::new();
        let modes = [false, true];
        for (repeat, graph) in modes.into_iter().enumerate() {
            model.reset()?;
            let mut token = model.prefill(&ids, None)?;
            let name = format!("rows{rows}-graph{graph}.bin");
            let mut file = std::fs::File::create_new(output.join(name))?;
            let mut margin = f32::INFINITY;
            for step in 0..16 {
                if graph && step == 1 {
                    model.capture()?;
                }
                if step > 0 {
                    token = model.decode(graph, None)?;
                }
                let logits = model.decode_logits()?;
                ensure!(
                    logits.len() == 130560 && logits.iter().all(|v| v.is_finite()),
                    "invalid full-vocabulary logits"
                );
                let mut order: Vec<_> = (0..logits.len()).collect();
                order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
                ensure!(
                    order[0] == token as usize,
                    "sample differs from logits argmax"
                );
                margin = margin.min(logits[order[0]] - logits[order[1]]);
                let bits: Vec<_> = logits.into_iter().map(f32::to_bits).collect();
                if repeat == 1 {
                    ensure!(
                        eager.get(step).context("eager step missing")? == &(token, bits.clone()),
                        "repeated execution token or logit bits differ rows={rows} step={step}"
                    );
                } else {
                    eager.push((token, bits.clone()));
                }
                file.write_all(&token.to_le_bytes())?;
                for value in bits {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
            file.flush()?;
            model.check_ready()?;
            println!(
                "DENSE_MODEL_IDENTITY rows={rows} graph={graph} steps=16 vocab=130560 min_top_two_margin={margin} token_logits_written=true PASS"
            );
        }
    }
    let prefix: Vec<_> = (0..255).map(|i| ((i * 269 + 23) % 130560) as u32).collect();
    let mut boundary = None;
    for graph in [false, true] {
        model.reset()?;
        let pending = model.prefill(&prefix, None)?;
        let token = model.decode(graph, None)?;
        let logits: Vec<_> = model
            .decode_logits()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        ensure!(
            model.position() == 256,
            "decode did not reach exact KV capacity"
        );
        ensure!(
            model.decode(graph, None).is_err(),
            "decode exceeded KV capacity"
        );
        model.rewind(255, pending)?;
        ensure!(model.decode(graph, None)? == token, "rewind token identity");
        ensure!(
            model
                .decode_logits()?
                .into_iter()
                .map(f32::to_bits)
                .eq(logits.iter().copied()),
            "rewind full-logit identity"
        );
        if let Some(expected) = &boundary {
            ensure!(
                expected == &(token, logits.clone()),
                "graph/eager capacity token/logit identity"
            );
        } else {
            boundary = Some((token, logits));
        }
        model.check_ready()?;
    }
    println!(
        "DENSE_GRAPH_BOUNDARY capacity=256 rewind=255 graph_eager_bit_identity=true overflow_rejected=true PASS"
    );
    println!("DENSE_MODEL_IDENTITY cells=6 graph_eager_bit_identity=true PASS");
    Ok(())
}

#[derive(serde::Deserialize)]
struct ReferenceFixtures {
    steps: usize,
    chunks: Vec<usize>,
    cases: Vec<ReferenceCase>,
}

#[derive(serde::Deserialize)]
struct ReferenceCase {
    name: String,
    rows: usize,
    capacity: usize,
    forced: Vec<u32>,
    limits: ReferenceLimits,
}

#[derive(serde::Deserialize)]
struct ReferenceLimits {
    max_normalized: f64,
    rms_normalized: f64,
}

fn reference_settings(runtime: &std::path::Path) -> Settings {
    Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(runtime.to_owned()),
        prefix_cache: false,
        kv_format: KvFormat::Bf16,
    }
}

fn reference_prefill(model: &mut Model, ids: &[u32]) -> Result<u32> {
    if ids.len() == 1 {
        return model.prefill(ids, None);
    }
    let split = 17.min(ids.len() - 1);
    model.prefill(&ids[..split], None)?;
    ensure!(
        model.position() == split,
        "nonzero prefix position mismatch"
    );
    let token = model.prefill(&ids[split..], None)?;
    ensure!(
        model.position() == ids.len(),
        "continued prefix position mismatch"
    );
    Ok(token)
}

#[test]
#[ignore = "requires GPU lease, FF_ARGMAX_MODEL, FF_DENSE_REFERENCE and runtime/evidence directories"]
fn dense_model_reference_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let output = PathBuf::from(std::env::var("FF_DENSE_EVIDENCE")?);
    let reference = PathBuf::from(std::env::var("FF_DENSE_REFERENCE")?);
    let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
    let fixtures: ReferenceFixtures =
        serde_json::from_str(include_str!("fixtures/dense_reference.json"))?;
    ensure!(
        fixtures
            .cases
            .iter()
            .all(|c| c.limits.max_normalized.is_finite()
                && c.limits.max_normalized >= 0.0
                && c.limits.rms_normalized.is_finite()
                && c.limits.rms_normalized >= 0.0),
        "model comparison requires finite nonnegative committed limits"
    );
    let mut defects = 0;
    let mut cells = 0;
    for case in &fixtures.cases {
        ensure!(
            case.forced.len() + 1 == fixtures.steps && case.rows + fixtures.steps <= case.capacity,
            "invalid reference fixture extent"
        );
        let bytes = std::fs::read(reference.join(format!(
            "{}-chunk{}-repeat0.safetensors",
            case.name, case.rows
        )))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor("logits")?;
        ensure!(
            view.dtype() == safetensors::Dtype::F32 && view.shape() == [fixtures.steps, 130560],
            "reference logits layout must be F32[steps,130560]"
        );
        let reference_logits: Vec<_> = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&x| f32::from_le_bytes(x))
            .collect();
        ensure!(
            reference_logits.iter().all(|v| v.is_finite()),
            "nonfinite reference logits"
        );
        for &chunk in &fixtures.chunks {
            let settings = reference_settings(&runtime);
            drop(Model::calibrate(
                &target,
                0,
                case.capacity,
                Some(chunk),
                settings.clone(),
            )?);
            let mut model = Model::load(&target, 0, case.capacity, Some(chunk), settings)?;
            let ids: Vec<_> = (0..case.rows)
                .map(|i| ((i * 269 + 23) % 130560) as u32)
                .collect();
            let mut token = reference_prefill(&mut model, &ids)?;
            let mut max_normalized = 0.0f64;
            let mut rms_normalized = 0.0f64;
            let mut id_disagreements = 0;
            let mut argmax_defects = 0;
            let mut margin_excluded_ids = 0;
            let mut min_margin = f32::INFINITY;
            let mut file = std::fs::File::create_new(
                output.join(format!("{}-chunk{chunk}-teacher.bin", case.name)),
            )?;
            for step in 0..fixtures.steps {
                if step > 0 {
                    model.set_token(case.forced[step - 1])?;
                    token = model.decode(false, None)?;
                }
                ensure!(
                    model.position() == case.rows + step,
                    "absolute decode position mismatch"
                );
                let logits = model.decode_logits()?;
                ensure!(
                    logits.len() == 130560 && logits.iter().all(|v| v.is_finite()),
                    "nonfinite candidate logits or invalid layout"
                );
                let expected = &reference_logits[step * 130560..(step + 1) * 130560];
                let scale =
                    (expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 130560.0).sqrt();
                ensure!(scale.is_finite() && scale > 0.0, "invalid reference RMS");
                let mut max_error = 0.0f64;
                let mut squared_error = 0.0;
                for (&actual, &expected) in logits.iter().zip(expected) {
                    let error = f64::from(actual) - f64::from(expected);
                    max_error = max_error.max(error.abs());
                    squared_error += error * error;
                }
                max_normalized = max_normalized.max(max_error / scale);
                rms_normalized = rms_normalized.max((squared_error / 130560.0).sqrt() / scale);
                let mut order: Vec<_> = (0..130560).collect();
                order
                    .sort_unstable_by(|&a, &b| expected[b].total_cmp(&expected[a]).then(a.cmp(&b)));
                let margin = expected[order[0]] - expected[order[1]];
                min_margin = min_margin.min(margin);
                if token as usize != order[0] {
                    id_disagreements += 1;
                    if f64::from(margin) > 2.0 * case.limits.max_normalized * scale {
                        margin_excluded_ids += 1;
                    }
                }
                let actual_argmax = logits
                    .iter()
                    .enumerate()
                    .max_by(|(a, x), (b, y)| x.total_cmp(y).then(b.cmp(a)))
                    .context("empty logits")?
                    .0;
                argmax_defects += usize::from(token as usize != actual_argmax);
                file.write_all(&token.to_le_bytes())?;
                for value in logits {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
            file.flush()?;
            let numerical_defect = max_normalized > case.limits.max_normalized
                || rms_normalized > case.limits.rms_normalized;
            defects += usize::from(numerical_defect) + margin_excluded_ids + argmax_defects;
            cells += 1;
            println!(
                "{}",
                serde_json::json!({"dense_model_reference":{"case":case.name,"chunk":chunk,"steps":fixtures.steps,"max_normalized":max_normalized,"rms_normalized":rms_normalized,"id_disagreements":id_disagreements,"margin_excluded_ids":margin_excluded_ids,"argmax_defects":argmax_defects,"min_reference_margin":min_margin,"result":if numerical_defect || margin_excluded_ids>0 || argmax_defects>0{"DEFECT"}else{"PASS"}}})
            );
            model.reset()?;
            let mut greedy = Vec::new();
            greedy.push(reference_prefill(&mut model, &ids)?);
            for _ in 1..fixtures.steps {
                greedy.push(model.decode(false, None)?);
            }
            std::fs::write(
                output.join(format!("{}-chunk{chunk}-greedy.json", case.name)),
                serde_json::to_vec(&greedy)?,
            )?;
            model.check_ready()?;
        }
    }
    println!("DENSE_MODEL_REFERENCE cells={cells} defects={defects} limits_frozen=true");
    ensure!(
        defects == 0,
        "dense model reference candidate defects={defects}"
    );
    Ok(())
}
