use super::*;
use crate::backend::cuda::state::StateStore;
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};

struct Reference {
    record: serde_json::Value,
    tokens: Vec<u32>,
    limit: f64,
}

impl Reference {
    fn load(root: &Path) -> Result<Self> {
        let record: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("composition.json"))?)?;
        let token_bytes = std::fs::read(root.join("tokens.json"))?;
        ensure!(
            format!("{:x}", Sha256::digest(&token_bytes))
                == "9c5eb391cc756e43aee6bd95b0877f7f32e0f78c6afeb72dac7b9eb285f231c7",
            "Runner FAIL frozen token integrity"
        );
        let tokens: serde_json::Value = serde_json::from_slice(&token_bytes)?;
        let tokens: Vec<u32> = serde_json::from_value(tokens["input_ids"].clone())?;
        let floor = record["publisher_path_floor"]["largest_pair_mean_position_max_abs"]
            .as_f64()
            .context("missing publisher path floor")?;
        ensure!(
            record["round"] == "r2"
                && record["frozen"] == true
                && record["candidate_run"] == false
                && record["publisher_only"] == true
                && record["model_definition"] == "50e35cc0ca18ec19b80653690de553e6decf74f5"
                && record["native_backend"] == "0e5764aa5c27d0f005117a92645156481540950a"
                && record["tokens"]["source_sha256"]
                    == "9c5eb391cc756e43aee6bd95b0877f7f32e0f78c6afeb72dac7b9eb285f231c7"
                && record["tokens"]["count"] == 600
                && record["tokens"]["teacher_forced_logits"] == 599
                && record["tokens"]["added_bos"] == false
                && record["repeat_identity"]["different_fp32_bits"] == 0
                && tokens.len() == 600
                && tokens.iter().all(|&id| id < 154880)
                && floor.is_finite()
                && floor > 0.0,
            "Runner FAIL frozen GLM composition contract"
        );
        let loads = record["clean_load_summary"]
            .as_array()
            .context("clean loads missing")?;
        ensure!(
            loads.len() == 45
                && loads.iter().enumerate().all(|(i, load)| {
                    load["layer"] == i
                        && load["missing"] == 0
                        && load["unexpected"] == 0
                        && load["newly_initialized"] == 0
                }),
            "Runner FAIL publisher clean-load report"
        );
        let positions = record["positions"]
            .as_array()
            .context("reference rows missing")?;
        ensure!(positions.len() == 599, "Runner FAIL reference row count");
        for (i, row) in positions.iter().enumerate() {
            ensure!(
                row["position"] == i && row["target_id"] == tokens[i + 1],
                "Runner FAIL frozen token alignment at {i}"
            );
            ensure!(
                row["max_publisher_logit_spread"]
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && v >= 0.0),
                "Runner FAIL reference spread at {i}"
            );
            for arm in ["R1", "R2", "R3", "R4", "R5"] {
                ensure!(
                    row[format!("{arm}_id")]
                        .as_u64()
                        .is_some_and(|v| v < 154880)
                        && row[format!("{arm}_margin")]
                            .as_f64()
                            .is_some_and(|v| v.is_finite() && v >= 0.0),
                    "Runner FAIL reference ID/margin {arm} at {i}"
                );
            }
        }
        for arm in ["R1", "R2", "R3", "R4", "R5"] {
            let schedule: Vec<usize> = serde_json::from_value(record["arms"][arm].clone())?;
            ensure!(
                schedule.iter().all(|&v| v > 0)
                    && schedule.iter().sum::<usize>() == 599
                    && record["arms"][arm] == record["head_schedules"][arm]
                    && record["arm_nll"][arm].as_f64().is_some_and(f64::is_finite),
                "Runner FAIL reference arm/head schedule {arm}"
            );
            if matches!(arm, "R1" | "R2") {
                ensure!(schedule == vec![1; 599], "Runner FAIL singleton arm {arm}");
            }
        }
        let mut means = [0.0; 5];
        for (arm, mean) in ["R1", "R2", "R3", "R4", "R5"].into_iter().zip(&mut means) {
            let file = std::fs::File::open(root.join(format!("{arm}-logits.fp32")))?;
            ensure!(
                file.metadata()?.len() == 599 * 154880 * 4,
                "Runner FAIL reference extent {arm}"
            );
            let mut reader = BufReader::new(file);
            for &target in &tokens[1..] {
                *mean += nll(&read_logits(&mut reader, 154880)?, target as usize)?;
            }
            *mean /= 599.0;
        }
        let limit = nll_limit(&means)?;
        ensure!(
            (limit - 0.04456232760365619).abs() < 1e-12,
            "Runner FAIL frozen reference NLL limit {limit:.17}"
        );
        println!(
            "G1_REFERENCE_NLL means={means:?} limit={limit:.17} logit_spread_diagnostic={floor:.17}"
        );
        Ok(Self {
            record,
            tokens,
            limit,
        })
    }
}

fn read_logits(reader: &mut impl Read, width: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; width * 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}

fn nll_limit(means: &[f64; 5]) -> Result<f64> {
    ensure!(
        means.iter().all(|v| v.is_finite()),
        "Runner FAIL reference NLL non-finite"
    );
    let low = means.iter().copied().fold(f64::INFINITY, f64::min);
    let high = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(high > low, "Runner FAIL reference NLL floor absent");
    Ok(2.0 * (high - low))
}

fn nll(logits: &[f32], target: usize) -> Result<f64> {
    ensure!(
        target < logits.len() && logits.iter().all(|v| v.is_finite()),
        "Runner FAIL composition logit extent/non-finite value"
    );
    let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    Ok(max
        + logits
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln()
        - f64::from(logits[target]))
}

fn id_defect(id: usize, expected: usize, margin: f64, spread: f64) -> bool {
    margin > spread && id != expected
}

struct Score<'a> {
    reference: &'a Reference,
    readers: Vec<BufReader<std::fs::File>>,
    output: BufWriter<std::fs::File>,
    total_nll: f64,
    reference_nll: [f64; 5],
    defects: usize,
    exempt: usize,
    control: usize,
    first_id: Option<usize>,
    first_required_id: Option<usize>,
    first_logit: Option<usize>,
    rows: usize,
}

impl<'a> Score<'a> {
    fn new(root: &Path, output: &Path, reference: &'a Reference) -> Result<Self> {
        let readers = ["R1", "R2", "R3", "R4", "R5"]
            .into_iter()
            .map(|arm| {
                let file = std::fs::File::open(root.join(format!("{arm}-logits.fp32")))?;
                ensure!(
                    file.metadata()?.len() == 599 * 154880 * 4,
                    "Runner FAIL reference logits extent {arm}"
                );
                Ok(BufReader::new(file))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            reference,
            readers,
            output: BufWriter::new(std::fs::File::create_new(output)?),
            total_nll: 0.0,
            reference_nll: [0.0; 5],
            defects: 0,
            exempt: 0,
            control: 0,
            first_id: None,
            first_required_id: None,
            first_logit: None,
            rows: 0,
        })
    }

    fn row(&mut self, position: usize, logits: &[f32]) -> Result<()> {
        ensure!(
            position == self.rows && logits.len() == 154880,
            "Runner FAIL candidate row alignment"
        );
        let target = self.reference.tokens[position + 1] as usize;
        let loss = nll(logits, target)?;
        self.total_nll += loss;
        for value in logits {
            self.output.write_all(&value.to_le_bytes())?;
        }
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.select_nth_unstable_by(1, |&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let id = order[0];
        let margin = f64::from(logits[id]) - f64::from(logits[order[1]]);
        let row = &self.reference.record["positions"][position];
        let expected = row["R1_id"].as_u64().context("R1 ID missing")? as usize;
        let reference_margin = row["R1_margin"].as_f64().context("R1 margin missing")?;
        let spread = row["max_publisher_logit_spread"]
            .as_f64()
            .context("spread missing")?;
        let required = reference_margin > spread;
        let defect = id_defect(id, expected, reference_margin, spread);
        self.defects += usize::from(defect);
        self.exempt += usize::from(!required);
        self.control += usize::from(id_defect(
            (expected + 1) % logits.len(),
            expected,
            reference_margin,
            spread,
        ));
        if id != expected {
            self.first_id.get_or_insert(position);
        }
        if defect {
            self.first_required_id.get_or_insert(position);
        }
        let mut distances = Vec::new();
        let mut reference_losses = Vec::new();
        for (arm, reader) in self.readers.iter_mut().enumerate() {
            let values = read_logits(reader, logits.len())?;
            let reference_loss = nll(&values, target)?;
            self.reference_nll[arm] += reference_loss;
            reference_losses.push(reference_loss);
            let mut max = 0.0f64;
            let mut squares = 0.0;
            for (&got, &want) in logits.iter().zip(&values) {
                let delta = (f64::from(got) - f64::from(want)).abs();
                max = max.max(delta);
                squares += delta * delta;
            }
            distances.push((max, (squares / logits.len() as f64).sqrt()));
        }
        if distances[0].0 > spread {
            self.first_logit.get_or_insert(position);
        }
        println!(
            "G1_COMPOSITION_ROW position={position} id={id} margin={margin:.12e} nll={loss:.12e} required={required} defect={defect} reference={row} reference_nll={reference_losses:?} distances_max_rms={distances:?}"
        );
        self.rows += 1;
        Ok(())
    }

    fn finish(&mut self, informational: bool) -> Result<bool> {
        self.output.flush()?;
        ensure!(
            self.rows == 599 && self.control > 0,
            "Runner FAIL composition extent/control"
        );
        let reference = self.reference_nll[0] / self.rows as f64;
        let delta = self.total_nll / self.rows as f64 - reference;
        let pass = delta.abs() <= self.reference.limit && self.defects == 0;
        println!(
            "{} G1_COMPOSITION informational={informational} rows={} mean_nll={:.12e} reference_nll={reference:.12e} delta={delta:.12e} limit={:.12e} id_defects={} margin_exempt={} first_id={:?} first_required_id={:?} first_logit_diagnostic={:?} vocabulary_rotation_control={}",
            if informational {
                "DIAGNOSTIC"
            } else if pass {
                "PASS"
            } else {
                "DEFECT"
            },
            self.rows,
            self.total_nll / self.rows as f64,
            self.reference.limit,
            self.defects,
            self.exempt,
            self.first_id,
            self.first_required_id,
            self.first_logit,
            self.control
        );
        Ok(pass)
    }
}

fn check_counts(calls: &[[usize; 10]], end: usize) -> Result<()> {
    ensure!(
        calls.len() == 45 && calls.iter().all(|v| v[..9] == [end; 9]),
        "DEFECT skipped or repeated GLM layer stage"
    );
    Ok(())
}

fn live_state(engine: &Engine, end: usize) -> Result<()> {
    let q = engine.qwen.as_ref().context("GLM runtime absent")?;
    let calls = q
        .glm_calls
        .as_ref()
        .context("GLM execution counts absent")?
        .borrow();
    check_counts(&calls, end)?;
    ensure!(
        engine.state.gdn.len() == 34
            && engine.state.dsa.len() == 11
            && q.layers.iter().filter(|v| v.native.is_some()).count() == 42
            && engine.hc.is_some()
            && engine.moe_slots.is_some(),
        "DEFECT GLM live binding counts"
    );
    let stream = &engine.device.stream;
    let sequence = stream.clone_dtoh(
        engine
            .state
            .sequence
            .as_ref()
            .context("GLM sequence absent")?,
    )?;
    let scheduled_end = end + usize::from(engine.position == end);
    ensure!(
        sequence[2] == scheduled_end as i32 && sequence[0] + sequence[1] == sequence[2],
        "DEFECT GLM scheduled sequence end={end} meta={sequence:?}"
    );
    for (layer, state) in engine.state.dsa.iter().enumerate() {
        let meta = stream.clone_dtoh(&state.meta)?;
        let ids = stream.clone_dtoh(&state.token_ids.slice(..end))?;
        ensure!(
            meta[..4] == sequence[..4]
                && meta[4] == 0
                && ids.iter().enumerate().all(|(i, &v)| v == i as i32),
            "DEFECT stale DSA state layer={layer} end={end} meta={meta:?}"
        );
    }
    println!(
        "G1_COMPOSITION_LIVE end={end} layers=45 stages_per_layer=9 KDA=34 DSA=11 native_expert_layers=42 head_calls={}",
        calls[0][9]
    );
    Ok(())
}

#[test]
fn composition_controls_reject_label_shift_and_skipped_layer() -> Result<()> {
    let limit = nll_limit(&[1.0, 1.0, 1.01, 1.02, 1.015])?;
    assert!((limit - 0.04).abs() < 1e-12);
    assert!(0.05 > limit && 0.05 < 2.0 * 1.2718517780303955);
    check_reset_meta(&[0, 1, 1, 37, 0, 0, 0, 0], 37)?;
    assert!(check_reset_meta(&[0, 0, 0, 37, 0, 0, 0, 0], 37).is_err());
    let logits = [2.0, 1.0, -1.0];
    assert!(nll(&[-1.0, 2.0, 1.0], 0)? - nll(&logits, 0)? > 2.0);
    assert!((nll(&[102.0, 101.0, 99.0], 0)? - nll(&logits, 0)?).abs() < 1e-12);
    assert!(id_defect(1, 0, 1.0, 0.5));
    assert!(!id_defect(1, 0, 1.0, 1.0));
    let mut calls = vec![[7; 10]; 45];
    check_counts(&calls, 7)?;
    calls[17][3] -= 1;
    assert!(check_counts(&calls, 7).is_err());
    calls[17][3] += 2;
    assert!(check_counts(&calls, 7).is_err());
    Ok(())
}

fn check_reset_meta(meta: &[i32], capacity: u32) -> Result<()> {
    // Metadata describes the scheduled first token, with no completed prefix.
    ensure!(
        meta == [0, 1, 1, capacity as i32, 0, 0, 0, 0],
        "DEFECT stale DSA reset meta={meta:?}"
    );
    Ok(())
}

#[test]
#[ignore = "frozen r2 reference logits; CPU only"]
fn glm_composition_reference() -> Result<()> {
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    Reference::load(&root)?;
    println!("PASS G1_REFERENCE_NLL arms=5 rows=599 candidate_inputs=0");
    Ok(())
}

fn reject_control(name: &str, values: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        values.len() == expected.len() && values.iter().all(|v| v.is_finite()),
        "Runner FAIL control {name} extent/non-finite logits"
    );
    let different = values
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(different > 0, "Runner FAIL undetected control {name}");
    println!("PASS G1_COMPOSITION_CONTROL name={name} different_logit_bits={different}");
    Ok(())
}

fn composition_controls(
    engine: &mut Engine,
    tokens: &[u32],
    first: &[f32],
    second: &[f32],
) -> Result<()> {
    engine.reset()?;
    engine.prefill(&tokens[1..2], None)?;
    reject_control("token_shift", &engine.decode_logits()?, first)?;
    engine.reset()?;
    engine.qwen.as_mut().context("GLM runtime absent")?.layers[3]
        .native
        .as_mut()
        .context("GLM layer3 native experts absent")?
        .wrong_route = true;
    let result = engine.prefill(&tokens[..1], None);
    engine.qwen.as_mut().context("GLM runtime absent")?.layers[3]
        .native
        .as_mut()
        .context("GLM layer3 native experts absent")?
        .wrong_route = false;
    result?;
    reject_control("wrong_selected_expert", &engine.decode_logits()?, first)?;
    for kda in [true, false] {
        engine.reset()?;
        engine.prefill(&tokens[..1], None)?;
        let stream = &engine.device.stream;
        if kda {
            for state in &mut engine.state.gdn {
                stream.memset_zeros(&mut state.s)?;
                stream.memset_zeros(&mut state.history)?;
            }
        } else {
            for state in &mut engine.state.dsa {
                stream.memcpy_htod(&[-1i32], &mut state.token_ids.slice_mut(..1))?;
            }
        }
        engine.set_token(tokens[1])?;
        engine.decode(false, None)?;
        reject_control(
            if kda { "stale_KDA" } else { "stale_DSA" },
            &engine.decode_logits()?,
            second,
        )?;
    }
    engine.reset()?;
    Ok(())
}

#[test]
#[ignore = "official GLM model, frozen r2 composition reference and exclusive GPU lease"]
fn glm_composition_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    let reference = Reference::load(&root)?;
    let model = PathBuf::from(std::env::var_os("FF_GLM_MODEL").context("GLM model missing")?);
    let output =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_OUTPUT").context("output missing")?);
    std::fs::create_dir(&output)?;
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?,
        )),
    };
    #[cfg(feature = "rocm")]
    Engine::calibrate(&model, 0, 640, None, settings.clone())?;
    let mut engine = Engine::load(&model, 0, 640, None, settings)?;
    ensure!(
        engine.workspace_rows
            == engine
                .qwen
                .as_ref()
                .context("GLM runtime missing")?
                .workspace_rows as usize,
        "Runner FAIL allocated GLM projection row pitch changed during slot admission"
    );
    ensure!(
        engine.model_spec.architecture == crate::model_spec::Architecture::Glm5Next
            && engine.config.vocab_size == 154880
            && engine.config.num_hidden_layers == 45,
        "Runner FAIL GLM geometry"
    );
    let mut score = Score::new(&root, &output.join("decode.fp32"), &reference)?;
    engine.prefill(&reference.tokens[..1], None)?;
    live_state(&engine, 1)?;
    let first = engine.decode_logits()?;
    score.row(0, &first)?;
    let mut second = None;
    for position in 1..599 {
        engine.set_token(reference.tokens[position])?;
        engine.decode(false, None)?;
        live_state(&engine, position + 1)?;
        let logits = engine.decode_logits()?;
        score.row(position, &logits)?;
        if position == 1 {
            second = Some(logits);
        }
    }
    let accepted = score.finish(false)?;
    drop(score);
    engine.reset()?;
    let stream = &engine.device.stream;
    for state in &engine.state.gdn {
        ensure!(
            stream.clone_dtoh(&state.s)?.iter().all(|&v| v == 0.0)
                && stream.clone_dtoh(&state.history)?.iter().all(|&v| v == 0.0),
            "DEFECT stale KDA reset"
        );
    }
    for state in &engine.state.dsa {
        check_reset_meta(&stream.clone_dtoh(&state.meta)?, engine.capacity as u32)?;
    }
    engine.prefill(&reference.tokens[..1], None)?;
    ensure!(
        engine
            .decode_logits()?
            .iter()
            .zip(&first)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "DEFECT reset first-token logit identity"
    );
    println!("PASS G1_COMPOSITION_RESET KDA=34 DSA=11 first_token_bits=0");
    composition_controls(
        &mut engine,
        &reference.tokens,
        &first,
        &second.context("second candidate row missing")?,
    )?;
    engine.reset()?;
    let mut chunks = Score::new(&root, &output.join("chunked.fp32"), &reference)?;
    engine.prefill_chunks(&reference.tokens[..599], None, 599, |e, ids| {
        live_state(e, e.position + ids.len())?;
        for row in 0..ids.len() {
            e.state.set_sequence(
                &e.device.stream,
                e.position as u32,
                (row + 1) as u32,
                e.workspace_rows as u32,
            )?;
            e.qwen_operator(
                crate::backend::Operator::Head,
                0,
                crate::backend::Step {
                    rows: row + 1,
                    input: crate::backend::Input::Ids,
                    head: Head::Last,
                    mask: crate::backend::Mask::Causal,
                },
                &mut None,
            )?;
            e.logits_cache.mark_valid();
            chunks.row(e.position + row, &e.decode_logits()?)?;
        }
        Ok(())
    })?;
    chunks.finish(true)?;
    ensure!(accepted, "DEFECT GLM production composition");
    Ok(())
}

#[test]
#[ignore = "GPU GLM DSA sequence publication and append slots"]
fn glm_dsa_publication_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glm53-flash-config.json"
    )))?;
    let model = crate::model_spec::ModelSpec::glm5_next(&root, &serde_json::Value::Null)?;
    let device = Device::new(0)?;
    let s = &device.stream;
    let mut state = StateStore::new(&model, 37, s)?;
    ensure!(
        crate::backend::cuda::state::StateCheckpoint::new(&state, &model, 1, s).is_err(),
        "DEFECT incomplete DSA checkpoint admitted"
    );
    let advance = cubin::module(&device.ctx, "q38_common")?.load_function("q38_advance")?;
    state.set_sequence(s, 7, 1, 16)?;
    let meta = state.seq_ref(s)?;
    let null = 0u64;
    unsafe {
        s.launch_builder(&advance)
            .arg(&meta.ptr)
            .arg(&null)
            .arg(&null)
            .launch(grid(1, 1))?;
    }
    let mut mutant = 0;
    for dsa in &state.dsa {
        mutant += usize::from(s.clone_dtoh(&dsa.meta)?[0] != 8);
    }
    ensure!(mutant == 11, "Runner FAIL global-only advance control");
    state.reset(s)?;
    let mut old_assertion_rejections = 0;
    for dsa in &mut state.dsa {
        let meta = s.clone_dtoh(&dsa.meta)?;
        check_reset_meta(&meta, 37)?;
        old_assertion_rejections += usize::from(meta[2] != 0);
        dsa.reset(s)?;
        ensure!(
            check_reset_meta(&s.clone_dtoh(&dsa.meta)?, 37).is_err(),
            "Runner FAIL unscheduled DSA reset control"
        );
    }
    ensure!(
        old_assertion_rejections == 11,
        "Runner FAIL old reset assertion control"
    );
    state.reset(s)?;
    println!(
        "PASS G1_DSA_RESET layers=11 scheduled_key_count=1 completed_prefix=0 old_assertion_rejections={old_assertion_rejections} unscheduled_reset_rejections=11"
    );
    let plan = DsaPlan::new(
        &device.ctx,
        &device.info,
        crate::op_contract::DsaSpec::glm53(),
        37,
        1,
    )?;
    let mut latent = s.alloc_zeros::<bf16>(512)?;
    let packed = s.alloc_zeros::<bf16>(257)?;
    for position in 0..5usize {
        let values: Vec<_> = (0..512)
            .map(|i| bf16::from_f32(((i * 37 + position * 17) % 131) as f32 / 16.0))
            .collect();
        s.memcpy_htod(&values, &mut latent)?;
        let input = DsaAppendInput {
            latent: matrix(&latent, s, 0, 1, 512)?,
            packed: matrix(&packed, s, 0, 1, 257)?,
        };
        for dsa in &state.dsa {
            plan.append(dsa, &input, s)?;
        }
        state.advance_sequence(s, &advance, None, None)?;
        for dsa in &state.dsa {
            let meta = s.clone_dtoh(&dsa.meta)?;
            ensure!(
                meta == [position as i32 + 1, 1, position as i32 + 2, 37, 0, 0, 0, 0]
                    && s.clone_dtoh(&dsa.token_ids.slice(..position + 1))?
                        == (0..=position as i32).collect::<Vec<_>>()
                    && s.clone_dtoh(&dsa.latent.slice(position * 512..(position + 1) * 512))?
                        == values,
                "DEFECT DSA publication/slot at position={position}"
            );
        }
    }
    println!(
        "PASS G1_DSA_PUBLICATION layers=11 steps=5 latent_width=512 metadata_mismatches=0 slot_mismatches=0 incomplete_checkpoint_rejected=true global_only_control={mutant}"
    );
    Ok(())
}

#[test]
#[ignore = "GPU GLM final stream mean at production width"]
fn glm_head_collapse_gpu() -> Result<()> {
    use crate::backend::cuda::{hc::HcBinding, mhc};
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = Device::new(0)?;
    let stream = &device.stream;
    let (capacity, mult, hidden) = (16, 4, 4096);
    let binding = HcBinding {
        params: mhc::HcParams {
            mult: mult as u32,
            hidden: hidden as u32,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-6,
        },
        policy: crate::model::hc_pre_policy(crate::model_spec::Architecture::Glm5Next)?,
        rounding: mhc::HcRounding::PostCombBf16,
        orientation: mhc::HcCombOrientation::Transposed,
        fn_dtype: mhc::HcFnDtype::Bf16,
    };
    let mut runtime = HcRuntime::new(&device.ctx, stream, binding, 1, 1)?;
    runtime.grow(stream, capacity)?;
    let input: Vec<_> = (0..capacity * mult * hidden)
        .map(|i| bf16::from_f32(((i * 37 + 11) % 131) as f32 / 16.0 - 4.0))
        .collect();
    let pre: Vec<_> = (0..capacity * mult)
        .map(|i| (i % mult + 1) as f32 / 8.0)
        .collect();
    let embedded = stream.clone_htod(&input[..capacity * hidden])?;
    runtime.begin_embedded(stream, capacity, &embedded)?;
    let initialized = stream.clone_dtoh(&runtime.stream)?;
    ensure!(
        initialized
            .iter()
            .enumerate()
            .all(|(i, &value)| { value == input[(i / (mult * hidden)) * hidden + i % hidden] }),
        "DEFECT GLM stream embedding broadcast"
    );
    println!(
        "PASS G1_STREAM_INIT rows={capacity} streams={mult} hidden={hidden} bit_mismatches=0 grown_from=1"
    );
    stream.memcpy_htod(&input, &mut runtime.stream)?;
    stream.memcpy_htod(&pre, &mut runtime.mix_pre)?;
    let mut output = stream.alloc_zeros::<bf16>(capacity * hidden)?;
    for rows in [1, 3, 16] {
        runtime.collapse_final(stream, rows, &mut output)?;
        let values = stream.clone_dtoh(&output)?;
        let mut mismatch = 0;
        let mut mutant = 0;
        for row in 0..rows {
            for column in 0..hidden {
                let mut sum = 0.0f32;
                let mut weighted = 0.0f32;
                for copy in 0..mult {
                    let value = input[(row * mult + copy) * hidden + column].to_f32();
                    sum += value;
                    weighted = pre[row * mult + copy].mul_add(value, weighted);
                }
                let expected = bf16::from_f32(sum / mult as f32);
                mismatch += usize::from(values[row * hidden + column] != expected);
                mutant += usize::from(bf16::from_f32(weighted) != expected);
            }
        }
        ensure!(
            mismatch == 0 && mutant > 0,
            "DEFECT GLM head rows={rows} mismatch={mismatch} old_pre_control={mutant}"
        );
        println!(
            "PASS G1_HEAD_MEAN rows={rows} streams={mult} hidden={hidden} bit_mismatches={mismatch} old_pre_control={mutant}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU DSA production index-packing layout"]
fn glm_index_pack_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = Device::new(0)?;
    let stream = &device.stream;
    let capacity = 16;
    let input: Vec<_> = (0..capacity * 257)
        .map(|i| bf16::from_f32(((i * 37 + 11) % 131) as f32 / 16.0 - 4.0))
        .collect();
    let source = stream.clone_htod(&input)?;
    let mut output = GlmDsaOutputs::new(capacity, stream)?;
    for rows in [1, 3, 16] {
        output.pack_index(stream, &source, [0, capacity * 128], rows)?;
        let values = stream.clone_dtoh(&output.packed)?;
        let mut mismatch = 0;
        let mut mutant = 0;
        for row in 0..rows {
            for column in 0..256 {
                let source_index = (column / 128) * capacity * 128 + row * 128 + column % 128;
                mismatch += usize::from(values[row * 257 + column] != input[source_index]);
                mutant += usize::from(input[row * 257 + column] != input[source_index]);
            }
        }
        ensure!(
            mismatch == 0 && mutant > 0,
            "DEFECT DSA packing rows={rows} mismatch={mismatch} old_layout_control={mutant}"
        );
        println!(
            "PASS G1_INDEX_PACK rows={rows} capacity={capacity} key_gate_columns=256 row_pitch=257 bit_mismatches={mismatch} wrong_layout_control={mutant}"
        );
    }
    Ok(())
}
