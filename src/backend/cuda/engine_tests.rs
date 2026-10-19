use super::*;

#[test]
fn chain_geometry_is_checked_before_device_setup() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "flyingfish-draft-block-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&root)?;
    let result = (|| -> Result<()> {
        let target = root.join("target");
        let draft = root.join("draft");
        std::fs::create_dir(&target)?;
        std::fs::create_dir(&draft)?;
        let value = serde_json::json!({
            "model_type":"llama", "hidden_size":2048, "intermediate_size":6144,
            "num_hidden_layers":42, "num_attention_heads":16, "num_key_value_heads":2,
            "head_dim":128, "vocab_size":130560, "max_position_embeddings":131072,
            "rms_norm_eps":1e-6, "rope_theta":5000000.0, "hidden_act":"silu",
            "tie_word_embeddings":false, "torch_dtype":"bfloat16", "eos_token_id":[2]
        });
        std::fs::write(target.join("config.json"), serde_json::to_vec(&value)?)?;
        let target_config: Config = serde_json::from_value(value)?;
        let mut value = serde_json::json!({
            "model_type":"qwen3", "architectures":["Qwen3DSparkModel"], "dtype":"bfloat16",
            "hidden_size":2048, "intermediate_size":6144, "num_hidden_layers":5,
            "num_attention_heads":16, "num_key_value_heads":2, "head_dim":128,
            "vocab_size":130560, "draft_vocab_size":130560, "rms_norm_eps":1e-6,
            "block_size":7, "mask_token_id":75982, "target_layer_ids":[1,10,20,30,39],
            "num_target_layers":42, "markov_rank":256, "markov_head_type":"vanilla",
            "projector_type":"dspark", "rope_parameters":{"rope_theta":5000000.0,"rope_type":"default"}
        });
        let runtime = root.join("blocked-runtime");
        std::fs::write(&runtime, b"not a directory")?;
        let mut rejected = 0;
        let mut declared = 0;
        for block in 1..=8 {
            value["block_size"] = block.into();
            std::fs::write(draft.join("config.json"), serde_json::to_vec(&value)?)?;
            let geometry = crate::dspark::DraftConfig::read(&draft, &target_config)?;
            assert_eq!(geometry.block_size, block);
            assert_eq!(SpecPhase::Draft.rows(&geometry)?, block);
            assert_eq!(SpecPhase::Verify.rows(&geometry)?, block + 1);
            assert_eq!(SpecPhase::Inject(block + 1).rows(&geometry)?, block + 1);
            for budget in [SpecBudget::Chain, SpecBudget::Tree16] {
                for write in [false, true] {
                    for capacity in [block + 1, 32] {
                        let settings = Settings {
                            tree_builder: TreeBuilder::Waves,
                            spec_budget: budget,
                            spec_graph: true,
                            draft_model: Some(draft.clone()),
                            runtime_dir: Some(runtime.clone()),
                            prefix_cache: false,
                            kv_format: crate::kv::KvFormat::Bf16,
                        };
                        let error = if write {
                            Engine::calibrate(&target, 0, capacity, Some(16), settings).err()
                        } else {
                            Engine::load(&target, 0, capacity, Some(16), settings).err()
                        }
                        .context("draft metadata reached device setup")?;
                        let expected = if budget.is_tree() && block + 1 != SpecBudget::Chain.rows()
                        {
                            rejected += 1;
                            Some("tree draft block_size=")
                        } else if capacity == block + 1 {
                            rejected += 1;
                            Some("DSpark block_size=")
                        } else {
                            declared += 1;
                            None
                        };
                        let reached = match expected {
                            Some(prefix) => error.to_string().starts_with(prefix),
                            None if write => error
                                .downcast_ref::<std::io::Error>()
                                .is_some_and(|e| e.kind() == std::io::ErrorKind::AlreadyExists),
                            None => error.to_string() == "runtime path is not a directory",
                        };
                        ensure!(
                            reached,
                            "draft block={block} capacity={capacity} budget={budget:?} validation order: {error:#}"
                        );
                    }
                }
            }
        }
        println!(
            "PASS chain_geometry_order block=1..8 pre_device_bounds={rejected} runtime_sentinel={declared} tree_chain_rows={}",
            SpecBudget::Chain.rows()
        );
        Ok(())
    })();
    std::fs::remove_dir_all(&root)?;
    result
}

#[test]
fn load_dispatch_selects_shared_hybrid_and_llama() -> Result<()> {
    assert_eq!(
        super::load_dispatch(crate::model_spec::Architecture::Qwen35Text)?,
        super::LoadDispatch::Qwen
    );
    assert_eq!(
        super::load_dispatch(crate::model_spec::Architecture::Llama)?,
        super::LoadDispatch::Llama
    );
    assert_eq!(
        super::load_dispatch(crate::model_spec::Architecture::MiniCpmV47)?,
        super::LoadDispatch::Qwen
    );
    Ok(())
}

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
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
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
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
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
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
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
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
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

#[test]
#[ignore = "requires a GPU window and FF_ARGMAX_MODEL/FF_ARGMAX_DRAFT/FLYINGFISH_RUNTIME_DIR"]
fn target_hidden_capture_gpu() -> Result<()> {
    use cudarc::driver::DevicePtr;
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let mut model = Engine::load_kind(&target, 0, 128, Some(17), settings, true)?;
    ensure!(
        model.qwen.is_none(),
        "Llama operator capture fixture required"
    );
    let layers = crate::dspark::read_capture_layers(&draft, &model.config)?;
    let mut capture = Some(HiddenCapture::new(
        &layers,
        &model.config,
        model.workspace_rows,
        &model.device.stream,
    )?);
    let s = model.device.stream.clone();
    let h = model.config.hidden_size;
    let width = capture.as_ref().context("capture missing")?.width;
    let poison = bf16::from_f32(-17.0);
    let prepare = |model: &mut Engine, start: usize, ids: &[u32]| -> Result<()> {
        model.position = start;
        model
            .device
            .stream
            .memcpy_htod(ids, &mut model.work.ids.slice_mut(..ids.len()))?;
        model
            .device
            .stream
            .memcpy_htod(&[start as i32], &mut model.work.position)?;
        model
            .device
            .stream
            .memcpy_htod(&[(start + ids.len()) as i32], &mut model.work.length)?;
        Ok(())
    };
    let run =
        |model: &mut Engine, start: usize, ids: &[u32], trace: Option<&mut Trace>| -> Result<()> {
            prepare(model, start, ids)?;
            model.forward(
                ids.len(),
                false,
                true,
                trace.map(|trace| Target {
                    trace,
                    prefix: "tap",
                    skip: 0,
                }),
            )
        };
    let state = |model: &Engine| -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for layer in &model.state.kv {
            for value in [&layer.k, &layer.v] {
                copy::bits(&s, value, &mut out)?;
            }
        }
        for value in [&model.work.x, &model.work.n, &model.work.logits] {
            copy::bits(&s, value, &mut out)?;
        }
        Ok(out)
    };
    let mut defects = 0;
    let mut cells = 0;
    let mut graph_replays = 0;
    let mut wrong_mismatches = 0;
    for prefix in [0usize, 7, 17] {
        for rows in [1usize, 2, 5, 17] {
            let ids: Vec<_> = (0..prefix + rows).map(|i| (i * 263 + 17) as u32).collect();
            model.hidden_capture = None;
            model.reset()?;
            if prefix > 0 {
                run(&mut model, 0, &ids[..prefix], None)?;
            }
            let mut baseline_trace = Trace::default();
            run(
                &mut model,
                prefix,
                &ids[prefix..],
                Some(&mut baseline_trace),
            )?;
            model.check_logits()?;
            let baseline = state(&model)?;
            model.hidden_capture = capture.take();
            model.reset()?;
            if prefix > 0 {
                run(&mut model, 0, &ids[..prefix], None)?;
            }
            let binding = model.hidden_capture.as_mut().context("capture missing")?;
            let pointer = binding.values.device_ptr(&s).0;
            s.memcpy_htod(&vec![poison; binding.values.len()], &mut binding.values)?;
            let mut trace = Trace::default();
            run(&mut model, prefix, &ids[prefix..], Some(&mut trace))?;
            model.check_logits()?;
            let binding = model.hidden_capture.as_ref().context("capture missing")?;
            let actual = s.clone_dtoh(&binding.values)?;
            ensure!(
                actual.iter().all(|x| x.to_f32().is_finite()),
                "non-finite target capture"
            );
            let mut expected = vec![poison; actual.len()];
            let mut mismatches = 0;
            for (slot, layer) in layers.iter().enumerate() {
                let name = format!("tap.layer.{layer}");
                let values = baseline_trace.values(&name)?;
                ensure!(
                    values.len() == rows * h && values.iter().all(|x| x.is_finite()),
                    "target trace shape/finiteness"
                );
                let enabled = trace.values(&name)?;
                ensure!(
                    enabled.len() == values.len() && enabled.iter().all(|x| x.is_finite()),
                    "enabled target trace shape/finiteness"
                );
                mismatches += values
                    .iter()
                    .zip(enabled)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                for row in 0..rows {
                    for d in 0..h {
                        expected[row * width + slot * h + d] = bf16::from_f32(values[row * h + d]);
                    }
                }
            }
            mismatches += actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            defects += usize::from(mismatches != 0 || state(&model)? != baseline);
            prepare(&mut model, prefix, &ids[prefix..])?;
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = model.forward(rows, false, true, None);
            let graph = s.end_capture(sys::CUgraphInstantiate_flags(0))?;
            launched?;
            let graph = graph.context("target capture graph missing")?;
            for _ in 0..2 {
                model.reset()?;
                if prefix > 0 {
                    run(&mut model, 0, &ids[..prefix], None)?;
                }
                prepare(&mut model, prefix, &ids[prefix..])?;
                let binding = model.hidden_capture.as_mut().context("capture missing")?;
                s.memcpy_htod(&vec![poison; binding.values.len()], &mut binding.values)?;
                ensure!(
                    binding.values.device_ptr(&s).0 == pointer,
                    "capture graph pointer moved"
                );
                graph.launch()?;
                s.synchronize()?;
                let values = s.clone_dtoh(
                    &model
                        .hidden_capture
                        .as_ref()
                        .context("capture missing")?
                        .values,
                )?;
                ensure!(
                    values.iter().all(|x| x.to_f32().is_finite()),
                    "non-finite graph target capture"
                );
                let mismatch = values
                    .iter()
                    .zip(&expected)
                    .any(|(a, b)| a.to_bits() != b.to_bits());
                defects += usize::from(mismatch || state(&model)? != baseline);
                graph_replays += 1;
            }
            if prefix == 7 && rows == 17 {
                let mut wrong = layers.clone();
                wrong[0] += 1;
                let mut binding =
                    HiddenCapture::new(&wrong, &model.config, model.workspace_rows, &s)?;
                s.memcpy_htod(&vec![poison; binding.values.len()], &mut binding.values)?;
                let healthy = model.hidden_capture.replace(binding);
                model.reset()?;
                run(&mut model, 0, &ids[..prefix], None)?;
                run(&mut model, prefix, &ids[prefix..], None)?;
                let values = s.clone_dtoh(
                    &model
                        .hidden_capture
                        .as_ref()
                        .context("capture missing")?
                        .values,
                )?;
                for row in 0..rows {
                    wrong_mismatches += values[row * width..row * width + h]
                        .iter()
                        .zip(&expected[row * width..row * width + h])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
                defects += usize::from(wrong_mismatches == 0);
                model.hidden_capture = healthy;
            }
            println!(
                "TARGET_CAPTURE operator=Llama prefix={prefix} rows={rows} taps={layers:?} bit_mismatches={mismatches} defects={defects}"
            );
            capture = model.hidden_capture.take();
            cells += 1;
        }
    }
    println!(
        "TARGET_HIDDEN_CAPTURE operator=Llama cells={cells} graph_replays={graph_replays} wrong_layer_bit_mismatches={wrong_mismatches} defects={defects} {}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "target capture has {defects} finite defects");
    Ok(())
}

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
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
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
    seed.save(&engine.state, &engine.work.token, prefix, &s, None)?;
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

#[test]
#[ignore = "requires a full-model GPU window, FF_Q38_MODEL and a calibrated FLYINGFISH_RUNTIME_DIR"]
fn q38_tail_prefill_after_ready_uses_bound_linear_bucket() -> Result<()> {
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
        spec_budget: SpecBudget::Tree16,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(runtime),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let mut engine = Engine::load_kind(&dir, 0, 128, Some(64), settings, false)?;
    engine.check_ready()?;
    let plans_before = engine.blas.fixed_plan_count();
    ensure!(
        engine.blas.has_fixed_plan(32) && !engine.blas.has_fixed_plan(31),
        "chunk64 setup did not bind the intermediate 32-row bucket"
    );
    let ids = vec![1u32; 95];
    let token = engine.prefill(&ids, None)?;
    engine.check_logits()?;
    let plans_after = engine.blas.fixed_plan_count();
    ensure!(
        plans_before == plans_after,
        "tail prefill created Lt plans after Ready: {plans_before} -> {plans_after}"
    );
    ensure!(
        token < engine.config.vocab_size as u32,
        "invalid tail token"
    );
    ensure!(!engine.poisoned, "tail prefill poisoned Ready engine");
    engine.check_ready()?;
    eprintln!(
        "Q38_TAIL_SINGLE_ARM tail31_bucket32 token={token} finite_logits=true plans_before={plans_before} plans_after={plans_after} PASS"
    );
    Ok(())
}
