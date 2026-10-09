use super::*;

#[test]
#[ignore = "requires a reserved GPU window and FF_ARGMAX_MODEL/FF_ARGMAX_DRAFT/FLYINGFISH_RUNTIME_DIR"]
fn closure_graphs_restore_state_and_match_eager() -> Result<()> {
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        linear_choices: Vec::new(),
        draft_model: None,
        runtime_dir: Some(runtime),
    };
    Engine::calibrate(&target, 0, 33824, Some(1024), settings.clone())?;
    let mut model = Engine::load(&target, 0, 33824, Some(1024), settings)?;
    model.check_ready()?;
    for kv in &model.kv {
        for head in 0..model.config.num_key_value_heads {
            let lo = head * model.capacity * model.config.head_dim;
            for values in [&kv.k, &kv.v] {
                ensure!(
                    model
                        .device
                        .stream
                        .clone_dtoh(&values.slice(lo..lo + model.config.head_dim))?
                        .iter()
                        .all(|v| v.to_bits() == 0),
                    "setup left target KV data"
                );
            }
        }
    }
    for rows in (1..=17).chain([1024, 8192, 32768]) {
        let ids = vec![1; rows];
        model.reset()?;
        let first = model.prefill(&ids, None)?;
        let mut eager = Vec::new();
        for _ in 0..8 {
            let token = model.decode(false, None)?;
            eager.push((
                token,
                model
                    .decode_logits()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
            ));
        }
        let kv = model.kv_snapshot(rows, 8)?;
        model.reset()?;
        ensure!(model.prefill(&ids, None)? == first, "prefill token changed");
        let poison = vec![bf16::from_bits(0x7fc0); 8 * model.config.head_dim];
        for cache in &mut model.kv {
            for head in 0..model.config.num_key_value_heads {
                let lo = (head * model.capacity + rows) * model.config.head_dim;
                model
                    .device
                    .stream
                    .memcpy_htod(&poison, &mut cache.k.slice_mut(lo..lo + poison.len()))?;
                model
                    .device
                    .stream
                    .memcpy_htod(&poison, &mut cache.v.slice_mut(lo..lo + poison.len()))?;
            }
        }
        model.capture()?;
        for (token, logits) in eager {
            ensure!(
                model.decode(true, None)? == token,
                "Graph token changed at prefill rows {rows}"
            );
            ensure!(
                model
                    .decode_logits()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
                    == logits,
                "Graph logits changed at prefill rows {rows}"
            );
        }
        ensure!(
            model.kv_snapshot(rows, 8)? == kv,
            "Graph KV changed at prefill rows {rows}"
        );
        model.check_ready()?;
        model.closure.expect_captures(model.graphs.len())?;
    }
    println!(
        "closure_target_graph rows1..17/1k/8k/32k token/logits/KV bitwise PASS; unwritten KV poisoned"
    );
    drop(model);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Tree16,
        spec_graph: true,
        linear_choices: Vec::new(),
        draft_model: Some(draft),
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
    };
    Engine::calibrate(&target, 0, 33824, Some(1024), settings.clone())?;
    let mut model = Engine::load(&target, 0, 33824, Some(1024), settings)?;
    let spec = model.spec.as_ref().context("draft missing")?;
    let state = spec.tree.as_ref().context("tree missing")?;
    ensure!(
        state.rows() == 0 && state.keep() == 0 && spec.active_rows == 0,
        "setup left active tree metadata"
    );
    let mut cleared = Vec::new();
    spec.draft.state_bits(0, 8, &mut cleared)?;
    ensure!(
        cleared.iter().all(|&v| v == 0),
        "setup left draft state or KV data"
    );
    for kv in &model.kv {
        for head in 0..model.config.num_key_value_heads {
            let lo = head * model.capacity * model.config.head_dim;
            for values in [&kv.k, &kv.v] {
                ensure!(
                    model
                        .device
                        .stream
                        .clone_dtoh(&values.slice(lo..lo + 16 * model.config.head_dim))?
                        .iter()
                        .all(|v| v.to_bits() == 0),
                    "setup left tree KV/padding data"
                );
            }
        }
    }
    model.prefill(&[1], None)?;
    let token = model.decode(false, None)?;
    model.step(false)?;
    model.check_logits()?;
    model.check_ready()?;
    println!("draft_profile_eager_decode tree16 decode/step PASS token={token}");
    model.reset()?;
    model.prefill(&[1], None)?;
    let start = model.position();
    let tokens: Vec<_> = (0..16).map(|r| if r < 8 { 0 } else { r }).collect();
    let parents: Vec<_> = (0..16)
        .map(|r| {
            if r == 0 {
                -1
            } else if r < 8 {
                r - 1
            } else {
                0
            }
        })
        .collect();
    let tree = crate::tree::Tree::edges(&tokens, &parents, start, model.capacity)?;
    model.verify_tree(&tree, None)?;
    for rows in 1..=8 {
        let plan = tree.select(&[0; 16], rows, &[])?;
        model
            .spec
            .as_mut()
            .context("draft missing")?
            .tree
            .as_mut()
            .context("tree missing")?
            .load_commit(&plan)?;
        model.tree_gather_scatter()?;
        model.tree_inject(start)?;
        let mut expected = Vec::new();
        model
            .spec
            .as_ref()
            .context("draft missing")?
            .draft
            .state_bits(start, rows, &mut expected)?;
        ensure!(
            expected.iter().any(|&v| v != 0),
            "inject control has no nonzero signal"
        );
        let zero = model.device.stream.alloc_zeros::<bf16>(rows * 10240)?;
        model.spec.as_mut().context("draft missing")?.draft.inject(
            &zero,
            rows,
            start,
            super::super::draft::Compute {
                ops: &model.ops,
                blas: &mut model.blas,
                config: &model.config,
            },
            None,
        )?;
        model.spec.as_ref().context("draft missing")?.inject_graphs[rows - 1]
            .as_ref()
            .context("inject Graph missing")?
            .launch()?;
        let mut actual = Vec::new();
        model
            .spec
            .as_ref()
            .context("draft missing")?
            .draft
            .state_bits(start, rows, &mut actual)?;
        ensure!(
            actual == expected,
            "inject M{rows} Graph differs from eager"
        );
    }
    let removed = model.spec.as_mut().context("draft missing")?.inject_graphs[2].take();
    ensure!(
        model.check_ready().is_err(),
        "missing inject Graph passed Ready validation"
    );
    model.spec.as_mut().context("draft missing")?.inject_graphs[2] = removed;
    model.check_ready()?;
    model.closure.expect_captures(model.graphs.len())?;
    Ok(())
}

#[test]
#[ignore = "requires a reserved GPU window and FF_ARGMAX_MODEL/FF_ARGMAX_DRAFT/FF_ARGMAX_INPUTS/FLYINGFISH_RUNTIME_DIR"]
fn chain_argmax_matches_single_on_real_logits() -> Result<()> {
    let target =
        PathBuf::from(std::env::var("FF_ARGMAX_MODEL").context("FF_ARGMAX_MODEL missing")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT").context("FF_ARGMAX_DRAFT missing")?);
    let inputs =
        PathBuf::from(std::env::var("FF_ARGMAX_INPUTS").context("FF_ARGMAX_INPUTS missing")?);
    let runtime = PathBuf::from(
        std::env::var("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
    );
    println!(
        "argmax_target={} draft={}",
        target.display(),
        draft.display()
    );
    let mut compared = 0;
    let mut rounds = 0;
    for prompt in [
        "p1", "p2", "p3", "p4", "math1", "math2", "math3", "code1", "code2", "code3", "gen1",
        "gen2", "gen3",
    ] {
        #[derive(serde::Deserialize)]
        struct Input {
            input_ids: Vec<u32>,
        }
        let ids: Input =
            serde_json::from_slice(&std::fs::read(inputs.join(format!("{prompt}.json")))?)?;
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: true,
            linear_choices: Vec::new(),
            draft_model: None,
            runtime_dir: Some(runtime.clone()),
        };
        let mut model = Engine::load(
            &target,
            0,
            (ids.input_ids.len() + 512).max(4096),
            None,
            settings.with_draft(&draft),
        )?;

        let first = model.prefill(&ids.input_ids, None)?;
        let mut actual_ids = vec![first];
        let mut old_ids = vec![first];
        let s = model.device.stream.clone();
        let mut old = s.alloc_zeros::<u32>(8)?;
        let mut generated = 1;
        let first_round = rounds;
        while generated < 256 {
            let round = model.spec_round(256 - generated, None, "")?;
            let spec = model.spec.as_ref().context("missing draft state")?;
            let vocab = model.config.vocab_size;
            ensure!(
                spec.active_rows == 8,
                "chain verification must have eight rows"
            );
            for row in 0..8 {
                unsafe {
                    s.launch_builder(&model.ops.argmax)
                        .arg(&spec.logits.slice(row * vocab..(row + 1) * vocab))
                        .arg(&mut old.slice_mut(row..row + 1))
                        .arg(&(vocab as i32))
                        .launch(grid(1, 1024))?;
                }
            }
            let expected = s.clone_dtoh(&old)?;
            let actual = s.clone_dtoh(&spec.predictions)?;
            ensure!(
                actual == expected,
                "chain argmax differs prompt={prompt} round={rounds}: actual={actual:?} expected={expected:?}"
            );
            let accepted = crate::dspark::accept(&round.proposals, &expected)?;
            let mut old_output = round.proposals[..accepted.proposals].to_vec();
            old_output.push(accepted.bonus);
            old_output.truncate(256 - generated);
            ensure!(
                round.accepted == accepted.proposals && round.output == old_output,
                "chain acceptance/output differs from single-row sampler"
            );
            actual_ids.extend(&round.output);
            old_ids.extend(old_output);
            compared += actual.len();
            generated += round.committed;
            rounds += 1;
        }
        ensure!(
            generated == 256 && actual_ids == old_ids,
            "chain output count mismatch"
        );
        println!(
            "{}",
            serde_json::json!({"chain_argmax_equivalence":{"prompt":prompt,"output_tokens":generated,"rounds":rounds-first_round,"prediction_bits_equal":true,"output_ids_equal":true,"actual_ids":actual_ids,"single_sampler_ids":old_ids,"scope":"same actual BF16 logits; batched production predictions vs eight original single-row launches; reconstruct original sampler acceptance and committed outputs"}})
        );
    }
    println!(
        "{}",
        serde_json::json!({"chain_argmax_equivalence_summary":{"prompts":13,"rounds":rounds,"predictions_compared":compared,"pass":true}})
    );
    Ok(())
}
