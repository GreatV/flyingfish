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

#[test]
#[ignore = "requires GPU lease, published MiniCPM-V-4.7 FP8 and runtime/evidence directories"]
fn hybrid_model_identity_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let specification = flyingfish::model_spec::ModelSpec::read(&target)?;
    ensure!(
        specification.linear_format == flyingfish::model_spec::LinearFormat::Fp8E4m3Channel
            && specification.expert_format == Some(flyingfish::model_spec::ExpertFormat::Fp8W8a16),
        "hybrid gate requires published per-channel FP8 projections and experts"
    );
    let output = PathBuf::from(std::env::var("FF_DENSE_EVIDENCE")?);
    let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
    let settings = reference_settings(&runtime);
    drop(Model::calibrate(
        &target,
        0,
        8192,
        Some(129),
        settings.clone(),
    )?);
    let mut model = Model::load(&target, 0, 8192, Some(129), settings)?;
    ensure!(
        model.config.hidden_size == 2048
            && model.config.num_hidden_layers == 40
            && model.config.vocab_size == 248144,
        "MiniCPM-V-4.7 geometry required"
    );
    let mut cells = 0;
    let mut defects = 0;
    for rows in [33, 1024, 4096] {
        let ids: Vec<_> = (0..rows)
            .map(|i| ((i * 269 + 23) % 248144) as u32)
            .collect();
        let mut first = Vec::new();
        for (repeat, graph) in [false, true, false].into_iter().enumerate() {
            let mut cell_defects = 0;
            model.reset()?;
            let mut token = reference_prefill(&mut model, &ids)?;
            let mut file = std::fs::File::create_new(
                output.join(format!("text{rows}-repeat{repeat}-graph{graph}.bin")),
            )?;
            let mut margin = f32::INFINITY;
            for step in 0..4 {
                if step > 0 {
                    model.set_token([115, 1301, 3809][step - 1])?;
                    if graph && step == 1 {
                        model.capture()?;
                    }
                    token = model.decode(graph, None)?;
                }
                ensure!(
                    model.position() == rows + step,
                    "hybrid absolute position mismatch"
                );
                let logits = model.decode_logits()?;
                ensure!(
                    logits.len() == 248144 && logits.iter().all(|v| v.is_finite()),
                    "hybrid logits extent/nonfinite"
                );
                let mut order: Vec<_> = (0..logits.len()).collect();
                order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
                cell_defects += usize::from(token as usize != order[0]);
                margin = margin.min(logits[order[0]] - logits[order[1]]);
                let bits: Vec<_> = logits.into_iter().map(f32::to_bits).collect();
                if repeat == 0 {
                    first.push((token, bits.clone()));
                } else {
                    cell_defects += usize::from(first[step] != (token, bits.clone()));
                }
                file.write_all(&token.to_le_bytes())?;
                for value in bits {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
            file.flush()?;
            model.check_ready()?;
            cells += 1;
            defects += cell_defects;
            println!(
                "HYBRID_MODEL_IDENTITY rows={rows} chunk=129 repeat={repeat} graph={graph} steps=4 min_top_two_margin={margin} defects={cell_defects} result={}",
                if cell_defects == 0 { "PASS" } else { "DEFECT" }
            );
        }
    }
    println!("HYBRID_MODEL_IDENTITY cells={cells} defects={defects}");
    ensure!(defects == 0, "hybrid model identity defects={defects}");
    Ok(())
}

#[test]
#[ignore = "requires FF_DENSE_REFERENCE and captured hybrid-model FF_DENSE_EVIDENCE"]
fn hybrid_model_reference_outputs() -> Result<()> {
    let reference = PathBuf::from(std::env::var("FF_DENSE_REFERENCE")?);
    let output = PathBuf::from(std::env::var("FF_DENSE_EVIDENCE")?);
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/minicpm_reference.json"))?;
    let cases = fixtures["cases"]
        .as_array()
        .context("hybrid fixture cases missing")?;
    let steps = fixtures["steps"]
        .as_u64()
        .context("hybrid fixture steps missing")? as usize;
    let mut cells = 0;
    let mut defects = 0;
    for case in cases {
        let name = case["name"]
            .as_str()
            .context("hybrid fixture name missing")?;
        let rows = case["rows"]
            .as_u64()
            .context("hybrid fixture rows missing")?;
        let limits: ReferenceLimits = serde_json::from_value(case["limits"].clone())
            .context("hybrid reference limits must be frozen before HIP model execution")?;
        ensure!(
            limits.max_normalized.is_finite()
                && limits.max_normalized >= 0.0
                && limits.rms_normalized.is_finite()
                && limits.rms_normalized >= 0.0,
            "invalid hybrid reference limits"
        );
        let bytes =
            std::fs::read(reference.join(format!("{name}-chunk{rows}-repeat0.safetensors")))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let view = tensors.tensor("logits")?;
        let vocab = 248144;
        ensure!(
            view.dtype() == safetensors::Dtype::F32 && view.shape() == [steps, vocab],
            "RUNNER_FAIL hybrid reference requires F32[steps,248144]"
        );
        let expected: Vec<_> = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&x| f32::from_le_bytes(x))
            .collect();
        ensure!(
            expected.iter().all(|v| v.is_finite()),
            "RUNNER_FAIL nonfinite hybrid reference"
        );
        for (repeat, graph) in [false, true, false].into_iter().enumerate() {
            let actual =
                std::fs::read(output.join(format!("text{rows}-repeat{repeat}-graph{graph}.bin")))?;
            let stride = 4 * (vocab + 1);
            ensure!(
                actual.len() == steps * stride,
                "RUNNER_FAIL hybrid capture extent"
            );
            let (mut maximum, mut rms) = (0.0f64, 0.0f64);
            let mut margin_defects = 0;
            let mut min_margin = f32::INFINITY;
            for (step, record) in actual.chunks_exact(stride).enumerate() {
                let token = u32::from_le_bytes(record[..4].try_into()?);
                let expected = &expected[step * vocab..(step + 1) * vocab];
                let scale = (expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                    / vocab as f64)
                    .sqrt();
                ensure!(
                    scale.is_finite() && scale > 0.0,
                    "RUNNER_FAIL hybrid reference RMS"
                );
                let (mut max_error, mut sum_error) = (0.0f64, 0.0f64);
                for (&word, &value) in record[4..].as_chunks::<4>().0.iter().zip(expected) {
                    let observed = f32::from_le_bytes(word);
                    ensure!(
                        observed.is_finite(),
                        "RUNNER_FAIL hybrid candidate nonfinite"
                    );
                    let error = f64::from(observed) - f64::from(value);
                    max_error = max_error.max(error.abs());
                    sum_error += error * error;
                }
                maximum = maximum.max(max_error / scale);
                rms = rms.max((sum_error / vocab as f64).sqrt() / scale);
                let mut order: Vec<_> = (0..vocab).collect();
                order
                    .sort_unstable_by(|&a, &b| expected[b].total_cmp(&expected[a]).then(a.cmp(&b)));
                let margin = expected[order[0]] - expected[order[1]];
                min_margin = min_margin.min(margin);
                margin_defects += usize::from(
                    token as usize != order[0]
                        && f64::from(margin) > 2.0 * limits.max_normalized * scale,
                );
            }
            let bad = usize::from(maximum > limits.max_normalized || rms > limits.rms_normalized)
                + margin_defects;
            defects += bad;
            cells += 1;
            println!(
                "HYBRID_MODEL_REFERENCE name={name} repeat={repeat} graph={graph} max_normalized={maximum:e} rms_normalized={rms:e} max_limit={} rms_limit={} min_margin={min_margin} margin_defects={margin_defects} result={}",
                limits.max_normalized,
                limits.rms_normalized,
                if bad == 0 { "PASS" } else { "DEFECT" }
            );
        }
    }
    println!("HYBRID_MODEL_REFERENCE_OUTPUTS cells={cells} defects={defects} limits_frozen=true");
    ensure!(defects == 0, "hybrid model reference defects={defects}");
    Ok(())
}

#[test]
#[ignore = "requires GPU lease, target/draft weights and runtime/evidence directories"]
fn dense_dspark_identity_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let output = PathBuf::from(std::env::var("FF_DENSE_EVIDENCE")?);
    let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
    let mut cells = 0;
    for (budget, chunk) in [
        (SpecBudget::Tree32, 9),
        (SpecBudget::Tree64, 17),
        (SpecBudget::Tree32, 17),
        (SpecBudget::Tree16, 9),
        (SpecBudget::Chain, 33),
        (SpecBudget::Tree16, 33),
    ] {
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: budget,
            spec_graph: true,
            draft_model: Some(draft.clone()),
            runtime_dir: Some(runtime.clone()),
            prefix_cache: false,
            kv_format: KvFormat::Bf16,
        };
        drop(Model::calibrate(
            &target,
            0,
            256,
            Some(chunk),
            settings.clone(),
        )?);
        let mut model = Model::load(&target, 0, 256, Some(chunk), settings)?;
        ensure!(
            model.draft_config()?.block_size == 7 && model.draft_config()?.markov_rank == 256,
            "MiniCPM5-2B DSpark block7/rank256 required"
        );
        for rows in [1usize, 17, 129] {
            let ids: Vec<_> = (0..rows)
                .map(|i| ((i * 269 + 23) % 130560) as u32)
                .collect();
            let mut expected = Vec::new();
            for (repeat, graph) in [false, true, true].into_iter().enumerate() {
                model.reset()?;
                model.set_spec_graph(graph)?;
                let first = model.prefill(&ids, None)?;
                let mut file = std::fs::File::create_new(output.join(format!(
                    "dspark-{budget:?}-chunk{chunk}-rows{rows}-repeat{repeat}.bin"
                )))?;
                file.write_all(&first.to_le_bytes())?;
                let mut min_margin = f32::INFINITY;
                let mut compared_logits = 0;
                for (round_index, limit) in [1usize, 3, 8, 5].into_iter().enumerate() {
                    let start = model.position();
                    let round = model.spec_round(limit, None, "")?;
                    ensure!(
                        round.committed == round.output.len()
                            && round.committed > 0
                            && round.committed <= limit
                            && model.position() == start + round.committed,
                        "DSpark commit position/output extent mismatch"
                    );
                    let mut all_logits = Vec::new();
                    for (row, &prediction) in round.predictions.iter().enumerate() {
                        let logits = model.verification_logits(row)?;
                        ensure!(
                            logits.len() == 130560 && logits.iter().all(|x| x.is_finite()),
                            "RUNNER_FAIL DSpark logits"
                        );
                        let mut order: Vec<_> = (0..logits.len()).collect();
                        order.sort_unstable_by(|&a, &b| {
                            logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
                        });
                        ensure!(
                            prediction as usize == order[0],
                            "DSpark prediction differs from full-logit argmax"
                        );
                        min_margin = min_margin.min(logits[order[0]] - logits[order[1]]);
                        compared_logits += logits.len();
                        all_logits.extend(logits.into_iter().map(f32::to_bits));
                    }
                    if !budget.is_tree() {
                        let accepted =
                            flyingfish::dspark::accept(&round.proposals, &round.predictions)?;
                        let mut output = round.proposals[..accepted.proposals].to_vec();
                        output.push(accepted.bonus);
                        output.truncate(limit);
                        ensure!(
                            round.output == output
                                && round.accepted == accepted.proposals.min(output.len() - 1),
                            "chain acceptance differs from CPU prefix selection"
                        );
                    }
                    let state = model.spec_state_bits(start, round.committed)?;
                    let result = (round.output.clone(), all_logits, state);
                    if repeat == 0 {
                        expected.push(result);
                    } else {
                        ensure!(
                            expected.get(round_index) == Some(&result),
                            "DSpark graph/eager/repeat token, logits or KV/state bits differ budget={budget:?} prefix={rows} round={round_index}"
                        );
                    }
                    for token in round.output {
                        file.write_all(&token.to_le_bytes())?;
                    }
                    for value in model.decode_logits()? {
                        file.write_all(&value.to_bits().to_le_bytes())?;
                    }
                }
                file.flush()?;
                model.check_ready()?;
                println!(
                    "DENSE_DSPARK_IDENTITY budget={budget:?} chunk={chunk} prefix={rows} graph={graph} repeat={repeat} rounds=4 compared_logits={compared_logits} min_top_two_margin={min_margin} token_logits_state_bit_identity=true PASS"
                );
                cells += 1;
            }
        }
        if budget.is_tree() {
            for graph in [false, true] {
                model.reset()?;
                model.set_spec_graph(graph)?;
                let prefix: Vec<_> = (0..17).map(|i| ((i * 269 + 23) % 130560) as u32).collect();
                let anchor = model.prefill(&prefix, None)?;
                let tokens = [anchor, 29, 37, 43, 53, 61, 71, 79, 89];
                let parents = [-1, 0, 0, 1, 1, 2, 2, 5, 5];
                let tree = flyingfish::tree::Tree::edges(&tokens, &parents, 17, 256)?;
                let predictions = model.verify_tree(&tree, None)?;
                let clean: Vec<_> = (0..tokens.len())
                    .map(|r| model.verification_logits(r))
                    .collect::<Result<_>>()?;
                model.poison_tree_padding(&tree)?;
                ensure!(
                    model.verify_tree(&tree, None)? == predictions,
                    "poisoned tree padding changed predictions"
                );
                for (row, expected) in clean.iter().enumerate() {
                    ensure!(
                        model.verification_logits(row)? == *expected,
                        "poisoned tree padding changed valid logits"
                    );
                }
                let mut forced = vec![997; tokens.len()];
                forced[0] = tokens[2];
                forced[2] = tokens[5];
                forced[5] = tokens[8];
                let plan = tree.select(&forced, 4, &[])?;
                ensure!(
                    plan.rows == [0, 2, 5, 8],
                    "tree fixture did not select a noncontiguous path"
                );
                let expected = model.tree_kv_path(&plan.rows)?;
                ensure!(
                    expected != model.tree_kv_path(&[0, 1, 2, 3])?,
                    "tree KV indexing control lacks signal"
                );
                let mut mutant = plan.clone();
                mutant.rows[1] = 1;
                ensure!(
                    model
                        .commit_tree_fixture(&tree, &mutant, &[], &forced)
                        .is_err(),
                    "wrong tree path was accepted"
                );
                ensure!(
                    model.position() == 17 && model.tree_kv_path(&plan.rows)? == expected,
                    "rejected tree commit mutated KV/position"
                );
                model.commit_tree_fixture(&tree, &plan, &[], &forced)?;
                ensure!(
                    model.kv_snapshot(17, 4)? == expected && model.position() == 21,
                    "tree gather/scatter changed selected KV rows"
                );
                model.check_ready()?;
                println!(
                    "DENSE_DSPARK_TREE chunk={chunk} graph={graph} prefix=17 path=0,2,5,8 padding_nan_invariant=true invalid_path_rejected=true kv_compaction_bit_identity=true PASS"
                );
            }
        }
    }
    println!("DENSE_DSPARK_IDENTITY cells={cells} defects=0 PASS");
    Ok(())
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
