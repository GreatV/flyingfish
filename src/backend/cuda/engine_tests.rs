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
        draft_model: None,
        runtime_dir: Some(runtime),
    };
    Engine::calibrate(&target, 0, 33824, Some(16), settings.clone())?;
    let mut model = Engine::load(&target, 0, 33824, Some(1024), settings)?;
    model.check_ready()?;
    for kv in &model.state.kv {
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
        for cache in &mut model.state.kv {
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
    for kv in &model.state.kv {
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
    assert!(model.capture().is_err());
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
    drop(model);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Tree16,
        spec_graph: true,
        draft_model: Some(PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?)),
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
    };
    Engine::calibrate(&target, 0, 33824, Some(8), settings.clone())?;
    let mut model = Engine::load(&target, 0, 33824, Some(8), settings.clone())?;
    model.prefill(&[1; 8], None)?;
    let before = model.device.stream.clone_dtoh(&model.work.x)?;
    for rows in 9..16 {
        let step = crate::backend::Step {
            rows,
            input: crate::backend::Input::Ids,
            head: Head::Last,
            mask: crate::backend::Mask::Causal,
        };
        ensure!(
            model.run_step(step, None).is_err(),
            "undeclared forward rows passed admission"
        );
    }
    ensure!(
        model.device.stream.clone_dtoh(&model.work.x)? == before,
        "rejected rows mutated workspace"
    );
    model.run_step(
        crate::backend::Step {
            rows: 16,
            input: crate::backend::Input::Ids,
            head: Head::Last,
            mask: crate::backend::Mask::Causal,
        },
        None,
    )?;
    model.check_logits()?;
    model.check_ready()?;
    println!("small_chunk_forward tree16/chunk8 rejects9..15 without writes; declared16 PASS");
    drop(model);
    let settings = Settings {
        spec_budget: SpecBudget::Tree64,
        ..settings
    };
    Engine::calibrate(&target, 0, 33824, Some(8), settings.clone())?;
    let mut model = Engine::load(&target, 0, 33824, Some(8), settings)?;
    model.prefill(&[1; 8], None)?;
    model.run_step(
        crate::backend::Step {
            rows: 17,
            input: crate::backend::Input::Ids,
            head: Head::Last,
            mask: crate::backend::Mask::Causal,
        },
        None,
    )?;
    model.check_logits()?;
    model.check_ready()?;
    println!("small_chunk_forward tree64/chunk8 CublasOnly17 PASS");
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

/// Full-model G7; local primitive grants do not authorize this test.
#[test]
#[ignore = "requires a full-model GPU window and FF_Q38_MODEL/FLYINGFISH_RUNTIME_DIR"]
fn q38_engine_capture_gpu() -> Result<()> {
    use crate::backend::{Input, Mask, Step};
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "full-model GPU window not acknowledged"
    );
    let dir = PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("FF_Q38_MODEL missing")?);
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime directory missing")?,
    );
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(runtime),
    };
    // Calibration/setup returns Frozen before production graph capture for a fixed plan.
    let image_fixture = std::env::var_os("FF_Q38_IMAGE_CAPTURE_FIXTURE");
    let capacity = if image_fixture.is_some() { 128 } else { 32 };
    let chunk = if image_fixture.is_some() { 9 } else { 8 };
    let mut engine = Engine::load_kind(&dir, 0, capacity, Some(chunk), settings, true)?;
    ensure!(engine.qwen.is_some(), "G7 fixture requires Q38");
    let s = engine.device.stream.clone();
    engine.reset()?;
    let ids = if let Some(path) = image_fixture.as_ref() {
        use safetensors::{Dtype, SafeTensors};
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;
        let integers = |name: &str| -> Result<Vec<i64>> {
            let t = tensors.tensor(name)?;
            ensure!(
                t.dtype() == Dtype::I64,
                "image fixture integer dtype mismatch"
            );
            Ok(t.data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| i64::from_le_bytes(*v))
                .collect())
        };
        let ids = integers("input_ids")?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let modalities = integers("mm_token_type_ids")?
            .into_iter()
            .map(u8::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let grid = integers("image_grid_thw")?;
        ensure!(grid.len() == 3, "capture fixture must contain one image");
        let grids = [crate::vision_input::ImageGrid {
            t: grid[0].try_into()?,
            h: grid[1].try_into()?,
            w: grid[2].try_into()?,
        }];
        let pixels = tensors.tensor("pixel_values_fp32")?;
        ensure!(pixels.dtype() == Dtype::F32, "capture pixels must be FP32");
        let pixels: Vec<_> = pixels
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let input = crate::vision_input::MultimodalInput {
            input_ids: &ids,
            modality_ids: &modalities,
            grids: &grids,
            pixel_values: &pixels,
        };
        engine.prepare_multimodal(&input, None)?;
        ensure!(
            ids.len() % chunk == 1 && ids.len() + 3 <= capacity,
            "use the tiny/marker fixture for M1 tail and replay room"
        );
        ids
    } else {
        vec![1, 2, 3, 4, 5]
    };
    for part in ids.chunks(chunk) {
        engine.prepare(part.len())?;
        engine
            .state
            .set_sequence(&s, engine.position as u32, part.len() as u32, chunk as u32)?;
        s.memcpy_htod(part, &mut engine.work.ids.slice_mut(..part.len()))?;
        engine.run_step(
            Step {
                rows: part.len(),
                input: Input::Ids,
                head: Head::Last,
                mask: Mask::Causal,
            },
            None,
        )?;
        engine.position += part.len();
    }
    let prefix = ids.len();
    s.memcpy_htod(&[prefix as i32], &mut engine.work.position)?;
    s.memcpy_htod(&[prefix as i32 + 1], &mut engine.work.length)?;
    engine
        .state
        .set_sequence(&s, prefix as u32, 1, chunk as u32)?;
    s.synchronize()?;

    fn state_bits(engine: &Engine) -> Result<Vec<u8>> {
        let s = &engine.device.stream;
        let mut bits = Vec::with_capacity(engine.state.bytes() + 32);
        for state in &engine.state.gdn {
            for buffer in [&state.s, &state.history] {
                for value in s.clone_dtoh(buffer)? {
                    bits.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        for kv in &engine.state.kv {
            for buffer in [&kv.k, &kv.v] {
                for value in s.clone_dtoh(buffer)? {
                    bits.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        for value in s.clone_dtoh(engine.state.sequence.as_ref().context("metadata missing")?)? {
            bits.extend_from_slice(&value.to_le_bytes());
        }
        for value in engine
            .qwen
            .as_ref()
            .context("Q38 runtime missing")?
            .position_values(s)?
        {
            bits.extend_from_slice(&value.to_le_bytes());
        }
        for value in s.clone_dtoh(&engine.work.token)? {
            bits.extend_from_slice(&value.to_le_bytes());
        }
        for buffer in [&engine.work.position, &engine.work.length] {
            for value in s.clone_dtoh(buffer)? {
                bits.extend_from_slice(&value.to_le_bytes());
            }
        }
        bits.extend_from_slice(&engine.position.to_le_bytes());
        Ok(bits)
    }
    let committed = state_bits(&engine)?;
    for occurrence in [1, 2] {
        engine.capture_fail_after_layer = Some((0, occurrence));
        ensure!(
            engine.capture().is_err(),
            "injected warmup failure was ignored"
        );
        ensure!(
            engine.poisoned && engine.graph.is_none(),
            "failed capture published a graph or remained healthy"
        );
        ensure!(
            state_bits(&engine)? == committed,
            "failed warmup did not restore committed state"
        );
        engine.capture_fail_after_layer = None;
        // The fixture reuses the restored state; ordinary load returns its capture error.
        engine.poisoned = false;
    }
    engine.capture()?;
    ensure!(
        state_bits(&engine)? == committed,
        "nonzero-prefix capture changed committed state"
    );
    // Capture's write window is one row. The replay comparison needs three rows.
    let mut seed = crate::backend::cuda::state::StateCheckpoint::new(
        &engine.state,
        &engine.model_spec,
        3,
        &s,
    )?;
    seed.save(&engine.state, &engine.work.token, prefix, &s)?;
    let mut replay_tokens = Vec::new();
    for _ in 0..3 {
        engine
            .graph
            .as_ref()
            .context("decode graph missing")?
            .launch()?;
        engine.position += 1;
        replay_tokens.push(engine.token()?);
    }
    engine.check_logits()?;
    let replay_state = state_bits(&engine)?;
    let replay_logits = s.clone_dtoh(&engine.work.logits)?;
    seed.restore(&mut engine.state, &mut engine.work.token, &s)?;
    s.memcpy_htod(&[prefix as i32], &mut engine.work.position)?;
    s.memcpy_htod(&[prefix as i32 + 1], &mut engine.work.length)?;
    engine.position = prefix;
    ensure!(
        state_bits(&engine)? == committed,
        "replay seed restore changed committed state"
    );
    let mut eager_tokens = Vec::new();
    for _ in 0..3 {
        engine.run_decode_step(None)?;
        engine.position += 1;
        eager_tokens.push(engine.token()?);
    }
    ensure!(eager_tokens == replay_tokens, "eager/graph tokens differ");
    ensure!(
        state_bits(&engine)? == replay_state,
        "eager/graph S/history/KV/meta/token differ"
    );
    ensure!(
        s.clone_dtoh(&engine.work.logits)? == replay_logits,
        "eager/graph logits differ"
    );
    engine.closure.expect_captures(3)?;
    engine.check_graphs()?;
    engine.reset()?;
    ensure!(engine.images.is_none(), "reset retained image features");
    ensure!(
        engine
            .qwen
            .as_ref()
            .context("Q38 runtime missing")?
            .position_values(&s)?
            == crate::vision_input::PositionPlan::text(1, capacity)?.packed_axes()?,
        "reset retained image coordinates"
    );
    for state in &engine.state.gdn {
        for buffer in [&state.s, &state.history] {
            ensure!(
                s.clone_dtoh(buffer)?.iter().all(|v| v.to_bits() == 0),
                "reset retained recurrent state"
            );
        }
    }
    engine.closure.ready(Instant::now())?;
    engine.check_ready()?;
    engine.poisoned = true;
    ensure!(
        engine.check_ready().is_err(),
        "poisoned engine reported Ready"
    );
    engine.reset()?;
    engine.check_ready()?;
    eprintln!(
        "Q38_ENGINE_G7 nonzero_capture_restore/warmup_and_capture_failure_restore/repeated_graph_eager/state_bits/logits/reset/ready_poison PASS"
    );
    Ok(())
}
