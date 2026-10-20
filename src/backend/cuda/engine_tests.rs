use super::*;

#[test]
fn llama_rejects_unsupported_draft_blocks_before_device_setup() -> Result<()> {
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
        for (heads, head_dim, limit) in [(16, 128, 8), (8, 256, 16)] {
            value["num_attention_heads"] = heads.into();
            value["head_dim"] = head_dim.into();
            value["rope_parameters"]["rope_theta"] = if head_dim == 256 { 1e7 } else { 5e6 }.into();
            for block in 1..=limit {
                value["block_size"] = block.into();
                std::fs::write(draft.join("config.json"), serde_json::to_vec(&value)?)?;
                assert_eq!(
                    crate::dspark::DraftConfig::read(&draft, &target_config)?.block_size,
                    block
                );
                if block == 7 {
                    continue;
                }
                for budget in [SpecBudget::Chain, SpecBudget::Tree16] {
                    for write in [false, true] {
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
                            Engine::calibrate(&target, 0, 32, Some(16), settings).err()
                        } else {
                            Engine::load(&target, 0, 32, Some(16), settings).err()
                        }
                        .context("unsupported draft block reached device setup")?;
                        ensure!(
                            error
                                .to_string()
                                .starts_with("unsupported Llama DSpark block_size="),
                            "draft block {block} reached runtime/device setup: {error:#}"
                        );
                        rejected += 1;
                    }
                }
            }
        }
        println!(
            "PASS llama_draft_block pre_device_rejections={rejected} block7_config_valid=true"
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
#[ignore = "requires a GPU window and FF_Q38_GPU_WINDOW=granted"]
fn target_capture_wave_offsets_gpu() -> Result<()> {
    use super::super::{
        contracts::{RowDomain, SeqRef, TensorMut},
        ops::{CommonOpsRef, ResidualRmsInput, RmsMode, RmsOptions},
    };
    use cudarc::driver::DevicePtr;
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let config: Config = serde_json::from_value(serde_json::json!({
        "model_type":"llama", "hidden_size":2048, "intermediate_size":6144,
        "num_hidden_layers":42, "num_attention_heads":16, "num_key_value_heads":2,
        "head_dim":128, "vocab_size":130560, "max_position_embeddings":32768,
        "rms_norm_eps":1e-6, "rope_theta":5000000.0, "rope_scaling":null,
        "hidden_act":"silu", "tie_word_embeddings":false,
        "torch_dtype":"bfloat16", "eos_token_id":[2]
    }))?;
    let device = Device::new(0)?;
    let s = &device.stream;
    let ops = Ops::new(&device.ctx)?;
    let common = CommonOpsRef::new(&device.ctx)?;
    let rows = 263;
    let hidden = config.hidden_size;
    let poison = bf16::from_f32(-17.0);
    let x_host: Vec<_> = (0..rows * hidden)
        .map(|i| bf16::from_f32(((i % 257) as f32 - 128.0) / 128.0))
        .collect();
    let add_host: Vec<_> = (0..rows * hidden)
        .map(|i| bf16::from_f32(((i % 263) as f32 - 131.0) / 256.0))
        .collect();
    let weights_host: Vec<_> = (0..hidden)
        .map(|i| bf16::from_f32((i % 251) as f32 / 256.0 + 0.5))
        .collect();
    let add = s.clone_htod(&add_host)?;
    let weights = s.clone_htod(&weights_host)?;
    let baseline = s.clone_htod(&x_host)?;
    let baseline_norm = s.alloc_zeros::<bf16>(rows * hidden)?;
    let mut metadata = s.clone_htod(&[0i32, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?;
    let seq = SeqRef {
        ptr: metadata.device_ptr(s).0,
    };
    let options = RmsOptions {
        mode: RmsMode::ZeroCentered,
        eps: config.rms_norm_eps,
    };
    common.norm.rms_add_with(
        s,
        RowDomain {
            meta: seq,
            row_base: 0,
            row_capacity: rows as u32,
        },
        ResidualRmsInput {
            x: &mut TensorMut(qwen::heads(&baseline, s, 0, rows, 1, hidden as u32)?),
            residual: qwen::heads(&add, s, 0, rows, 1, hidden as u32)?,
            weight: qwen::vector(&weights, s)?,
        },
        &mut TensorMut(qwen::heads(&baseline_norm, s, 0, rows, 1, hidden as u32)?),
        options,
    )?;
    let expected_x = s.clone_dtoh(&baseline)?;
    let expected_norm = s.clone_dtoh(&baseline_norm)?;
    ensure!(
        expected_x
            .iter()
            .chain(&expected_norm)
            .all(|v| v.to_f32().is_finite()),
        "non-finite wave reference"
    );
    let unequal = |a: &[bf16], b: &[bf16]| {
        a.iter()
            .zip(b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    let mut expected_capture = vec![poison; rows * 5 * hidden];
    for row in 0..rows {
        expected_capture[row * 5 * hidden + 2 * hidden..row * 5 * hidden + 3 * hidden]
            .copy_from_slice(&expected_x[row * hidden..(row + 1) * hidden]);
    }
    let mut defects = 0;
    let mut control_mismatch = 0;
    for wrong in [false, true] {
        let mut capture = HiddenCapture::new(&[1, 10, 20, 30, 39], &config, rows, s)?;
        s.memcpy_htod(&vec![poison; capture.values.len()], &mut capture.values)?;
        let mut x = s.clone_htod(&x_host)?;
        let normalized = s.alloc_zeros::<bf16>(rows * hidden)?;
        for base in [0, 256] {
            let count = (rows - base).min(256);
            let meta = crate::op_contract::SequenceMeta {
                prefix: base as i32,
                valid_rows: count as i32,
                key_count: (base + count) as i32,
                capacity: rows as i32,
                causal: 1,
                reserved: [0; 3],
            };
            meta.validate(rows as u32, count as u32)?;
            s.memcpy_htod(
                &[
                    meta.prefix,
                    meta.valid_rows,
                    meta.key_count,
                    meta.capacity,
                    1,
                    0,
                    0,
                    0,
                ],
                &mut metadata,
            )?;
            // SAFETY: each wave covers count*hidden values in the owned row buffers.
            unsafe {
                s.launch_builder(&ops.residual)
                    .arg(&mut x.slice_mut(base * hidden..(base + count) * hidden))
                    .arg(&add.slice(base * hidden..(base + count) * hidden))
                    .arg(&i32::try_from(count * hidden)?)
                    .launch(crate::backend::cuda::ops::flat(count * hidden))?;
            }
            copy::columns(
                s,
                &x.slice(base * hidden..(base + count) * hidden),
                &mut capture
                    .values
                    .slice_mut(if wrong { 0 } else { base * capture.width }..),
                copy::Rect {
                    rows: count,
                    width: hidden,
                    src_pitch: hidden,
                    src_col: 0,
                    dst_pitch: capture.width,
                    dst_col: 2 * hidden,
                },
            )?;
            common.norm.rms_with(
                s,
                RowDomain {
                    meta: seq,
                    row_base: 0,
                    row_capacity: count as u32,
                },
                qwen::heads(&x, s, base * hidden, count, 1, hidden as u32)?,
                qwen::vector(&weights, s)?,
                &mut TensorMut(qwen::heads(
                    &normalized,
                    s,
                    base * hidden,
                    count,
                    1,
                    hidden as u32,
                )?),
                options,
            )?;
        }
        let actual_x = s.clone_dtoh(&x)?;
        let actual_norm = s.clone_dtoh(&normalized)?;
        let actual_capture = s.clone_dtoh(&capture.values)?;
        ensure!(
            actual_x
                .iter()
                .chain(&actual_norm)
                .chain(&actual_capture)
                .all(|x| x.to_f32().is_finite()),
            "non-finite capture waves"
        );
        let mismatch = unequal(&actual_capture, &expected_capture);
        if wrong {
            control_mismatch = mismatch;
            defects += usize::from(mismatch == 0);
        } else {
            defects += usize::from(mismatch != 0);
        }
        let residual_mismatch = unequal(&actual_x, &expected_x);
        let norm_mismatch = unequal(&actual_norm, &expected_norm);
        defects += usize::from(residual_mismatch != 0 || norm_mismatch != 0);
        println!(
            "CAPTURE_WAVE_ARM wrong_base={wrong} residual_bit_mismatches={residual_mismatch} norm_bit_mismatches={norm_mismatch} capture_bit_mismatches={mismatch} defects={defects}"
        );
    }
    println!(
        "CAPTURE_WAVES rows=263 hidden=2048 waves=256,7 writer=residual_then_columns wrong_base_bit_mismatches={control_mismatch} defects={defects} {}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(
        defects == 0,
        "capture wave copy has {defects} finite defects"
    );
    Ok(())
}

fn captured_trace_bits(
    actual: &[bf16],
    shape: (usize, usize),
    layers: &[usize],
    trace: &Trace,
    prefix: &str,
    poison: bf16,
) -> Result<(usize, usize)> {
    let (rows, hidden) = shape;
    let width = layers
        .len()
        .checked_mul(hidden)
        .context("capture trace width overflow")?;
    ensure!(
        rows.checked_mul(width) == Some(actual.len())
            && actual.iter().all(|x| x.to_f32().is_finite()),
        "capture trace shape/finiteness"
    );
    let mut mismatch = 0;
    let mut old_noop = 0;
    for (slot, layer) in layers.iter().enumerate() {
        let expected = trace.values(&format!("{prefix}.layer.{layer}"))?;
        ensure!(
            expected.len() == rows * hidden && expected.iter().all(|x| x.is_finite()),
            "layer trace shape/finiteness"
        );
        for row in 0..rows {
            for d in 0..hidden {
                let expected = bf16::from_f32(expected[row * hidden + d]).to_bits();
                mismatch +=
                    usize::from(actual[row * width + slot * hidden + d].to_bits() != expected);
                old_noop += usize::from(poison.to_bits() != expected);
            }
        }
    }
    Ok((mismatch, old_noop))
}

#[test]
#[ignore = "requires a MiniCPM BF16 full-model window and FF_ARGMAX_MODEL/FF_ARGMAX_DRAFT/FLYINGFISH_RUNTIME_DIR"]
fn qwen_target_hidden_capture_gpu() -> Result<()> {
    use crate::experts::plan::Tier;
    use cudarc::driver::DevicePtr;
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "full-model GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let spec = crate::model_spec::ModelSpec::read(&target)?;
    ensure!(
        spec.architecture == crate::model_spec::Architecture::MiniCpmV47
            && spec.linear_format == crate::model_spec::LinearFormat::Bf16,
        "capture gate requires MiniCPM-V BF16"
    );
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: Some(draft.clone()),
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let started = Instant::now();
    let mut model = Engine::load_kind(&target, 0, 512, Some(17), settings, true)?;
    ensure!(
        model.qwen.is_some() && model.moe_prefill.is_some() && model.spec.is_some(),
        "draft-attached Qwen capture binding missing"
    );
    ensure!(
        !model.graphs.is_empty(),
        "capture parity gate requires declared target graphs"
    );
    model.reset()?;
    model.prepare_graphs()?;
    model.closure.expect_captures(model.graphs.len())?;
    model.check_graphs()?;
    model.reset()?;
    model.closure.ready(started)?;
    let mut draft_state = model.spec.take();
    let layers = crate::dspark::read_capture_layers(&draft, &model.config)?;
    let mut capture = Some(
        model
            .hidden_capture
            .take()
            .context("target capture missing")?,
    );
    let s = model.device.stream.clone();
    let hidden = model.config.hidden_size;
    let width = capture.as_ref().context("target capture missing")?.width;
    let pointer = capture
        .as_ref()
        .context("target capture missing")?
        .values
        .device_ptr(&s)
        .0;
    let span_rows = model
        .moe_prefill
        .as_ref()
        .context("prefill missing")?
        .reference_geometry()[0];
    let poison = bf16::from_f32(-17.0);
    let signature = |model: &Engine, rows: usize| -> Result<[Vec<u8>; 4]> {
        let logits = s.clone_dtoh(&model.work.logits)?;
        ensure!(
            logits.iter().all(|value| value.to_f32().is_finite()),
            "non-finite capture signature logits"
        );
        let mut bits = [qwen_state_bits(model)?, Vec::new(), Vec::new(), Vec::new()];
        for (index, value) in [&model.work.x, &model.work.n].into_iter().enumerate() {
            copy::bits(&s, &value.slice(..rows * hidden), &mut bits[index + 1])?;
        }
        copy::bits(&s, &model.work.logits, &mut bits[3])?;
        Ok(bits)
    };
    let prefill = |model: &mut Engine,
                   ids: &[u32],
                   trace: Option<&mut Trace>,
                   observe: bool|
     -> Result<Vec<(usize, Vec<bf16>)>> {
        let start = model.position;
        let mut observed = Vec::new();
        model.prefill_moe_chunks(ids, trace, ids.len(), &mut |engine, part| {
            if observe {
                let base = (engine.position - start) % span_rows;
                let binding = engine
                    .hidden_capture
                    .as_ref()
                    .context("target capture missing")?;
                observed.push((
                    part.len(),
                    s.clone_dtoh(
                        &binding
                            .values
                            .slice(base * width..(base + part.len()) * width),
                    )?,
                ));
            }
            Ok(())
        })?;
        Ok(observed)
    };
    let poison_capture = |model: &mut Engine| -> Result<()> {
        let binding = model
            .hidden_capture
            .as_mut()
            .context("target capture missing")?;
        ensure!(
            binding.values.device_ptr(&s).0 == pointer,
            "capture graph pointer moved"
        );
        s.memcpy_htod(&vec![poison; binding.values.len()], &mut binding.values)?;
        Ok(())
    };
    let mut defects = 0;
    let mut old_noop = 0;
    let seed: Vec<_> = (0..17).map(|i| (i * 263 + 17) as u32).collect();
    prefill(&mut model, &seed, None, false)?;
    let frequency = model
        .moe_prefill
        .as_ref()
        .context("prefill missing")?
        .capture_frequency();
    ensure!(
        frequency
            .iter()
            .all(|layer| layer.iter().all(|x| x.is_finite() && *x >= 0.0)
                && layer.iter().sum::<f64>() > 0.0),
        "capture seed frequency is invalid"
    );
    println!(
        "QWEN_CAPTURE_SEED rows=17 layers={} frequency=frozen_actual_prefill",
        frequency.len()
    );
    let pin = |model: &mut Engine| -> Result<_> {
        let slots = model.moe_slots.as_mut().context("expert slots missing")?;
        slots.pin_capture_request(&frequency)
    };
    let resident_limits = [2usize, 5, 17, 18, 35, 263]
        .into_iter()
        .map(|rows| {
            Ok((
                rows,
                model
                    .moe_prefill
                    .as_ref()
                    .context("prefill missing")?
                    .capture_resident_limit(rows)?,
            ))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let placement_matches = |actual: Vec<(Tier, Option<u32>)>,
                             expected: &[(Tier, Option<u32>)],
                             phase: &str,
                             prefill_rows: Option<usize>,
                             routed: Option<&[Vec<f64>]>|
     -> Result<bool> {
        ensure!(
            actual.len() == expected.len(),
            "Runner FAIL capture placement geometry differs"
        );
        ensure!(
            prefill_rows.is_some() == routed.is_some(),
            "Runner FAIL capture activity geometry differs"
        );
        let routed = routed.map(|layers| layers.iter().flatten().copied().collect::<Vec<_>>());
        ensure!(
            routed
                .as_ref()
                .is_none_or(|counts| counts.len() == actual.len()
                    && counts
                        .iter()
                        .all(|count| count.is_finite() && *count >= 0.0)),
            "Runner FAIL capture activity is invalid"
        );
        let gpu_cpu = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| (a.0 == Tier::Gpu) != (b.0 == Tier::Gpu))
            .count();
        let storage = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| a.0 != b.0)
            .count();
        let slots = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| a.0 == Tier::Gpu && b.0 == Tier::Gpu && a.1 != b.1)
            .count();
        let limit = prefill_rows
            .map(|rows| {
                resident_limits
                    .get(&rows)
                    .copied()
                    .context("capture row count missing")
            })
            .transpose()?
            .unwrap_or(u32::MAX);
        let resident = |entry: &(Tier, Option<u32>)| {
            entry.0 == Tier::Gpu && entry.1.is_some_and(|slot| slot < limit)
        };
        let borrowed = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| resident(a) != resident(b))
            .count();
        let executed = actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(index, (a, b))| {
                resident(a) != resident(b)
                    && routed.as_ref().is_none_or(|counts| counts[*index] > 0.0)
            })
            .count();
        println!(
            "QWEN_CAPTURE_PLACEMENT phase={phase} gpu_cpu_mismatches={gpu_cpu} post_borrow_gpu_cpu_mismatches={borrowed} executed_post_borrow_gpu_cpu_mismatches={executed} resident_slot_limit={limit} storage_tier_changes={storage} physical_gpu_slot_changes={slots}"
        );
        Ok(gpu_cpu == 0 && executed == 0)
    };
    for rows in [18usize, 35, 263] {
        let ids: Vec<_> = (0..rows).map(|i| (i * 263 + 17) as u32).collect();
        model.reset()?;
        pin(&mut model)?;
        prefill(&mut model, &ids, None, false)?;
        model.reset()?;
        let placement = pin(&mut model)?;
        prefill(&mut model, &ids, None, false)?;
        let tail = rows % model.chunk;
        let active = if tail == 0 { model.chunk } else { tail };
        let expected = signature(&model, active)?;
        let routed = model
            .moe_prefill
            .as_ref()
            .context("prefill missing")?
            .capture_frequency();
        model.reset()?;
        ensure!(
            placement_matches(
                pin(&mut model)?,
                &placement,
                "prefill_seed",
                Some(rows),
                Some(&routed)
            )?,
            "Runner FAIL native placement differs before capture pair"
        );
        model.spec = draft_state.take();
        model.hidden_capture = capture.take();
        poison_capture(&mut model)?;
        let mut trace = Trace::default();
        let observed = prefill(&mut model, &ids, Some(&mut trace), true)?;
        let mut mismatch = 0;
        for (part, (count, values)) in observed.iter().enumerate() {
            let result = captured_trace_bits(
                values,
                (*count, hidden),
                &layers,
                &trace,
                &format!("prefill.{part}"),
                poison,
            )?;
            mismatch += result.0;
            old_noop += result.1;
        }
        let state_equal = signature(&model, active)? == expected;
        defects += usize::from(mismatch != 0 || !state_equal);
        println!(
            "QWEN_CAPTURE_PREFILL rows={rows} mixer_rows={} span_rows={span_rows} trace_bit_mismatches={mismatch} residual_norm_logits_state_bits_equal={state_equal} defects={defects}",
            model.chunk
        );
        draft_state = model.spec.take();
        capture = model.hidden_capture.take();
    }
    let prefix: Vec<_> = (0..17).map(|i| (i * 263 + 17) as u32).collect();
    for rows in [2usize, 5] {
        let ids: Vec<_> = (0..rows).map(|i| (i * 269 + 23) as u32).collect();
        let causal = |model: &mut Engine, trace: Option<&mut Trace>| -> Result<()> {
            model
                .moe_slots
                .as_mut()
                .context("expert slots missing")?
                .end_request()?;
            model.prefill_moe_chunks(&ids, trace, ids.len(), &mut |_, _| Ok(()))?;
            Ok(())
        };
        model.reset()?;
        pin(&mut model)?;
        prefill(&mut model, &prefix, None, false)?;
        causal(&mut model, None)?;
        model.reset()?;
        let placement = pin(&mut model)?;
        prefill(&mut model, &prefix, None, false)?;
        let prefix_routed = model
            .moe_prefill
            .as_ref()
            .context("prefill missing")?
            .capture_frequency();
        let before = model
            .moe_slots
            .as_ref()
            .context("expert slots missing")?
            .capture_placement()?;
        causal(&mut model, None)?;
        let expected = signature(&model, rows)?;
        let append_routed = model
            .moe_prefill
            .as_ref()
            .context("prefill missing")?
            .capture_frequency();
        model.reset()?;
        ensure!(
            placement_matches(
                pin(&mut model)?,
                &placement,
                "causal_seed",
                Some(prefix.len()),
                Some(&prefix_routed)
            )?,
            "Runner FAIL native causal placement differs"
        );
        model.spec = draft_state.take();
        model.hidden_capture = capture.take();
        prefill(&mut model, &prefix, None, false)?;
        defects += usize::from(!placement_matches(
            model
                .moe_slots
                .as_ref()
                .context("expert slots missing")?
                .capture_placement()?,
            &before,
            "causal_prefix",
            Some(rows),
            Some(&append_routed),
        )?);
        poison_capture(&mut model)?;
        let mut trace = Trace::default();
        causal(&mut model, Some(&mut trace))?;
        let binding = model
            .hidden_capture
            .as_ref()
            .context("target capture missing")?;
        let values = s.clone_dtoh(&binding.values)?;
        let result = captured_trace_bits(
            &values[..rows * width],
            (rows, hidden),
            &layers,
            &trace,
            "prefill.0",
            poison,
        )?;
        let inactive = values[rows * width..]
            .iter()
            .filter(|x| x.to_bits() != poison.to_bits())
            .count();
        old_noop += result.1;
        let state_equal = signature(&model, rows)? == expected;
        defects += usize::from(result.0 != 0 || inactive != 0 || !state_equal);
        println!(
            "QWEN_CAPTURE_CAUSAL_APPEND prefix=17 rows={rows} trace_bit_mismatches={} inactive_bit_mismatches={inactive} residual_norm_logits_state_bits_equal={state_equal} defects={defects}",
            result.0
        );
        draft_state = model.spec.take();
        capture = model.hidden_capture.take();
    }
    model.reset()?;
    pin(&mut model)?;
    prefill(&mut model, &prefix, None, false)?;
    let disabled_placement = model
        .moe_slots
        .as_ref()
        .context("expert slots missing")?
        .capture_placement()?;
    model.decode(false, None)?;
    let disabled_state = signature(&model, 1)?;
    model.spec = draft_state.take();
    model.hidden_capture = capture.take();
    model.reset()?;
    pin(&mut model)?;
    prefill(&mut model, &prefix, None, false)?;
    model.reset()?;
    pin(&mut model)?;
    prefill(&mut model, &prefix, None, false)?;
    let decode_placement = model
        .moe_slots
        .as_ref()
        .context("expert slots missing")?
        .capture_placement()?;
    defects += usize::from(!placement_matches(
        decode_placement.clone(),
        &disabled_placement,
        "decode_prefix",
        None,
        None,
    )?);
    poison_capture(&mut model)?;
    let mut trace = Trace::default();
    model.profiler_start()?;
    model.decode(false, Some((&mut trace, "decode")))?;
    let expected = signature(&model, 1)?;
    let disabled_equal = expected == disabled_state;
    defects += usize::from(!disabled_equal);
    println!(
        "QWEN_CAPTURE_DECODE rows=1 capture_off_state_bits_equal={disabled_equal} defects={defects}"
    );
    let expected_capture = s.clone_dtoh(
        &model
            .hidden_capture
            .as_ref()
            .context("target capture missing")?
            .values,
    )?;
    let result = captured_trace_bits(
        &expected_capture[..width],
        (1, hidden),
        &layers,
        &trace,
        "decode",
        poison,
    )?;
    defects += usize::from(result.0 != 0);
    old_noop += result.1;
    let graph_rows = model
        .spec
        .as_ref()
        .context("draft state missing")?
        .draft
        .config
        .verify_rows()?;
    let graph_ids: Vec<_> = (0..graph_rows).map(|row| (row * 269 + 23) as u32).collect();
    model.reset()?;
    let graph_seed = pin(&mut model)?;
    prefill(&mut model, &prefix, None, false)?;
    let graph_placement = model
        .moe_slots
        .as_ref()
        .context("expert slots missing")?
        .capture_placement()?;
    let graph_routed = model
        .moe_prefill
        .as_ref()
        .context("prefill missing")?
        .capture_frequency();
    poison_capture(&mut model)?;
    s.memcpy_htod(&graph_ids, &mut model.work.ids.slice_mut(..graph_rows))?;
    let mut verify_trace = Trace::default();
    model.run_step(
        crate::backend::Step {
            rows: graph_rows,
            input: crate::backend::Input::Ids,
            head: Head::All,
            mask: crate::backend::Mask::Causal,
        },
        Some(Target {
            trace: &mut verify_trace,
            prefix: "verify",
            skip: 0,
        }),
    )?;
    let graph_expected = signature(&model, graph_rows)?;
    let graph_capture = s.clone_dtoh(
        &model
            .hidden_capture
            .as_ref()
            .context("target capture missing")?
            .values,
    )?;
    let spec = model.spec.as_ref().context("draft state missing")?;
    let graph_logits = s.clone_dtoh(&spec.logits)?;
    let graph_predictions = s.clone_dtoh(&spec.predictions)?;
    ensure!(
        graph_logits.iter().all(|x| x.to_f32().is_finite()),
        "non-finite verify capture reference logits"
    );
    let result = captured_trace_bits(
        &graph_capture[..graph_rows * width],
        (graph_rows, hidden),
        &layers,
        &verify_trace,
        "verify",
        poison,
    )?;
    defects += usize::from(result.0 != 0);
    old_noop += result.1;
    for _ in 0..2 {
        model.reset()?;
        ensure!(
            placement_matches(
                pin(&mut model)?,
                &graph_seed,
                "graph_seed",
                Some(prefix.len()),
                Some(&graph_routed)
            )?,
            "Runner FAIL graph seed placement differs"
        );
        prefill(&mut model, &prefix, None, false)?;
        defects += usize::from(!placement_matches(
            model
                .moe_slots
                .as_ref()
                .context("expert slots missing")?
                .capture_placement()?,
            &graph_placement,
            "graph_prefix",
            None,
            None,
        )?);
        poison_capture(&mut model)?;
        s.memcpy_htod(&graph_ids, &mut model.work.ids.slice_mut(..graph_rows))?;
        model.launch_spec_graph(SpecPhase::Verify)?;
        let actual = s.clone_dtoh(
            &model
                .hidden_capture
                .as_ref()
                .context("target capture missing")?
                .values,
        )?;
        let spec = model.spec.as_ref().context("draft state missing")?;
        let logits = s.clone_dtoh(&spec.logits)?;
        ensure!(
            actual.iter().chain(&logits).all(|x| x.to_f32().is_finite()),
            "non-finite verify capture replay"
        );
        let equal = actual
            .iter()
            .zip(&graph_capture)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && logits
                .iter()
                .zip(&graph_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && s.clone_dtoh(&spec.predictions)? == graph_predictions
            && signature(&model, graph_rows)? == graph_expected;
        defects += usize::from(!equal);
        println!(
            "QWEN_CAPTURE_GRAPH prefix=17 rows={graph_rows} phase=verify bitwise_equal={equal} defects={defects}"
        );
    }
    let original = model
        .hidden_capture
        .as_ref()
        .context("target capture missing")?
        .layer_slots
        .clone();
    draft_state = model.spec.take();
    capture = model.hidden_capture.take();
    let final_ids: Vec<_> = (0..35).map(|i| (i * 263 + 17) as u32).collect();
    model.reset()?;
    pin(&mut model)?;
    prefill(&mut model, &final_ids, None, false)?;
    model.reset()?;
    let final_placement = pin(&mut model)?;
    prefill(&mut model, &final_ids, None, false)?;
    let final_expected = signature(&model, 1)?;
    let final_routed = model
        .moe_prefill
        .as_ref()
        .context("prefill missing")?
        .capture_frequency();
    model.reset()?;
    ensure!(
        placement_matches(
            pin(&mut model)?,
            &final_placement,
            "final_seed",
            Some(final_ids.len()),
            Some(&final_routed)
        )?,
        "Runner FAIL final-tap seed placement differs"
    );
    model.spec = draft_state.take();
    model.hidden_capture = capture.take();
    let mut final_layers = layers.clone();
    let slot = final_layers.len() - 1;
    let last = model.config.num_hidden_layers - 1;
    {
        let binding = model
            .hidden_capture
            .as_mut()
            .context("target capture missing")?;
        binding.layer_slots[final_layers[slot]] = None;
        binding.layer_slots[last] = Some(u32::try_from(slot)?);
        final_layers[slot] = last;
    }
    poison_capture(&mut model)?;
    let mut final_trace = Trace::default();
    let observed = prefill(&mut model, &final_ids, Some(&mut final_trace), true)?;
    let mut final_mismatches = 0;
    for (part, (count, values)) in observed.iter().enumerate() {
        final_mismatches += captured_trace_bits(
            values,
            (*count, hidden),
            &final_layers,
            &final_trace,
            &format!("prefill.{part}"),
            poison,
        )?
        .0;
    }
    let final_actual = signature(&model, 1)?;
    for ((name, actual), expected) in ["state", "residual", "norm", "logits"]
        .into_iter()
        .zip(&final_actual)
        .zip(&final_expected)
    {
        ensure!(
            actual.len() == expected.len(),
            "final-tap signature extent differs"
        );
        let mismatch = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
        println!("QWEN_CAPTURE_FINAL_COMPONENT component={name} byte_mismatches={mismatch}");
    }
    let final_equal = final_actual == final_expected;
    defects += usize::from(final_mismatches != 0 || !final_equal);
    println!(
        "QWEN_CAPTURE_FINAL_TAP layer={last} trace_bit_mismatches={final_mismatches} capture_off_state_bits_equal={final_equal} defects={defects}"
    );
    model
        .hidden_capture
        .as_mut()
        .context("target capture missing")?
        .layer_slots = original.clone();
    {
        let binding = model
            .hidden_capture
            .as_mut()
            .context("target capture missing")?;
        binding.layer_slots[layers[0]] = None;
        binding.layer_slots[layers[0] + 1] = Some(0);
    }
    model.reset()?;
    pin(&mut model)?;
    poison_capture(&mut model)?;
    let ids: Vec<_> = (0..35).map(|i| (i * 263 + 17) as u32).collect();
    let mut trace = Trace::default();
    let observed = prefill(&mut model, &ids, Some(&mut trace), true)?;
    let mut wrong = 0;
    for (part, (count, values)) in observed.iter().enumerate() {
        wrong += captured_trace_bits(
            values,
            (*count, hidden),
            &layers,
            &trace,
            &format!("prefill.{part}"),
            poison,
        )?
        .0;
    }
    model
        .hidden_capture
        .as_mut()
        .context("target capture missing")?
        .layer_slots = original;
    defects += usize::from(wrong == 0 || old_noop == 0);
    model.profiler_stop()?;
    println!(
        "QWEN_TARGET_CAPTURE prefill_cells=3 causal_append_cells=2 final_tap_cells=1 graph_replays=2 wrong_layer_bit_mismatches={wrong} old_noop_bit_mismatches={old_noop} defects={defects} {}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "Qwen capture has {defects} finite defects");
    Ok(())
}

fn qwen_state_bits(engine: &Engine) -> Result<Vec<u8>> {
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
