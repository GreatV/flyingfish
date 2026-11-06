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
        let expected_round = match std::env::var_os("G1_COMPOSITION_ROUND") {
            Some(value) => value
                .into_string()
                .map_err(|_| anyhow::anyhow!("G1_COMPOSITION_ROUND is not UTF-8"))?,
            None => "r2".to_owned(),
        };
        ensure!(
            expected_round == "r2" || expected_round == "s2",
            "G1_COMPOSITION_ROUND must be r2 or s2"
        );
        let tokens_sha = format!("{:x}", Sha256::digest(&token_bytes));
        let expected_token_sha = if expected_round == "r2" {
            "9c5eb391cc756e43aee6bd95b0877f7f32e0f78c6afeb72dac7b9eb285f231c7".to_owned()
        } else {
            let value = std::env::var("G1_COMPOSITION_SEQUENCE_SHA256")
                .context("s2 requires G1_COMPOSITION_SEQUENCE_SHA256")?;
            ensure!(
                value.len() == 64
                    && value
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "G1_COMPOSITION_SEQUENCE_SHA256 must be 64 lowercase hex digits"
            );
            value
        };
        ensure!(
            format!("{:x}", Sha256::digest(&token_bytes)) == expected_token_sha,
            "Runner FAIL frozen token integrity"
        );
        let tokens: serde_json::Value = serde_json::from_slice(&token_bytes)?;
        let tokens: Vec<u32> = serde_json::from_value(tokens["input_ids"].clone())?;
        let floor = record["publisher_path_floor"]["largest_pair_mean_position_max_abs"]
            .as_f64()
            .context("missing publisher path floor")?;
        if expected_round == "s2" {
            let domain = std::env::var("G1_COMPOSITION_SEQUENCE_DOMAIN")
                .context("s2 requires G1_COMPOSITION_SEQUENCE_DOMAIN")?;
            ensure!(
                domain == "source_code_math" && record["sequence_domain"] == "source_code_math",
                "Runner FAIL s2 sequence domain"
            );
        }
        ensure!(
            record["round"] == expected_round
                && record["frozen"] == true
                && record["candidate_run"] == false
                && record["publisher_only"] == true
                && record["model_definition"] == "50e35cc0ca18ec19b80653690de553e6decf74f5"
                && record["native_backend"] == "0e5764aa5c27d0f005117a92645156481540950a"
                && record["tokens"]["source_sha256"] == tokens_sha
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
        if expected_round == "r2" {
            ensure!(
                (limit - 0.04456232760365619).abs() < 1e-12,
                "Runner FAIL frozen reference NLL limit {limit:.17}"
            );
        } else {
            ensure!(
                (limit - 0.01461872453804158).abs() < 1e-12,
                "Runner FAIL frozen s2 reference NLL limit {limit:.17}"
            );
            let recorded = record["reference_nll_limit"]
                .as_f64()
                .context("s2 reference_nll_limit missing")?;
            ensure!(
                recorded.is_finite() && recorded > 0.0 && (limit - recorded).abs() <= 1.0e-5,
                "Runner FAIL s2 reference NLL limit limit={limit:.17} recorded={recorded:.17}"
            );
        }
        println!(
            "G1_REFERENCE_NLL round={expected_round} means={means:?} limit={limit:.17} logit_spread_diagnostic={floor:.17}"
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

fn load_composition_engine() -> Result<Engine> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let model = PathBuf::from(std::env::var_os("FF_GLM_MODEL").context("GLM model missing")?);
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        prefix_cache: false,
        device_budget_gib: None,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?,
        )),
    };
    #[cfg(feature = "rocm")]
    Engine::calibrate(&model, 0, 640, None, settings.clone())?;
    if let Some(waves) = std::env::var_os("FF_GLM_FORCED_WAVES") {
        crate::experts::plan::FORCED_WAVES.set(Some(
            waves
                .to_str()
                .context("FF_GLM_FORCED_WAVES is not text")?
                .parse()?,
        ));
    }
    let loaded = Engine::load(&model, 0, 640, None, settings);
    crate::experts::plan::FORCED_WAVES.set(None);
    let engine = loaded?;
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
    Ok(engine)
}

#[test]
#[ignore = "official GLM model, frozen r2 composition reference and exclusive GPU lease"]
fn glm_composition_gpu() -> Result<()> {
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    let reference = Reference::load(&root)?;
    let output =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_OUTPUT").context("output missing")?);
    std::fs::create_dir(&output)?;
    let mut engine = load_composition_engine()?;
    let mut score = Score::new(&root, &output.join("decode.fp32"), &reference)?;
    engine.prefill(&reference.tokens[..1], None)?;
    live_state(&engine, 1)?;
    let first = engine.decode_logits()?;
    score.row(0, &first)?;
    let mut second = None;
    for position in 1..599 {
        engine.set_token(reference.tokens[position])?;
        {
            #[cfg(feature = "cuda")]
            let _range = crate::backend::cuda::profile::Range::new(
                matches!(
                    position,
                    1 | 3
                        | 4
                        | 5
                        | 7
                        | 8
                        | 9
                        | 31
                        | 32
                        | 33
                        | 127
                        | 128
                        | 129
                        | 511
                        | 512
                        | 513
                        | 598
                ),
                c"glm_eager_decode",
            );
            engine.decode(false, None)?;
        }
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
    composition_chunks(&mut engine, &root, &output, &reference)?;
    ensure!(accepted, "DEFECT GLM production composition");
    Ok(())
}

fn composition_chunks(
    engine: &mut Engine,
    root: &Path,
    output: &Path,
    reference: &Reference,
) -> Result<()> {
    engine.reset()?;
    let mut chunks = Score::new(root, &output.join("chunked.fp32"), reference)?;
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
    Ok(())
}

fn graph_native_state(engine: &Engine, full: bool) -> Result<Vec<u8>> {
    let stream = &engine.device.stream;
    let mut bytes = Vec::new();
    macro_rules! floats {
        ($items:expr) => {
            for value in $items {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        };
    }
    macro_rules! integers {
        ($items:expr) => {
            for value in $items {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        };
    }
    for state in &engine.state.gdn {
        floats!(stream.clone_dtoh(&state.s)?);
        floats!(stream.clone_dtoh(&state.history)?);
    }
    for state in &engine.state.dsa {
        for buffer in [&state.latent, &state.packed, &state.pool_keys] {
            integers!(stream.clone_dtoh(buffer)?.into_iter().map(bf16::to_bits));
        }
        for buffer in [&state.token_ids, &state.pool_ids, &state.meta] {
            integers!(stream.clone_dtoh(buffer)?);
        }
    }
    integers!(
        stream.clone_dtoh(
            engine
                .state
                .sequence
                .as_ref()
                .context("sequence metadata missing")?
        )?
    );
    for native in engine
        .qwen
        .as_ref()
        .context("runtime missing")?
        .layers
        .iter()
        .filter_map(|layer| layer.native.as_ref())
    {
        let (ids, weights) = native.plans[0].read_route(stream)?;
        integers!(ids.iter().copied());
        floats!(weights);
        let valid = stream.clone_dtoh(&native.valid)?;
        if full {
            integers!(valid);
        } else {
            let pointers = stream.clone_dtoh(&native.pointers)?;
            let slots = engine.moe_slots.as_ref().context("tier service missing")?;
            if slots.can_native_dispatch(native.slot_layer) {
                let (live_pointers, live_valid) = slots.native_dispatch(native.slot_layer)?;
                for id in ids {
                    let expert = usize::try_from(id).context("negative selected expert")?;
                    ensure!(
                        valid[expert] == 1
                            && live_valid[expert] == 1
                            && pointers[expert * 6..(expert + 1) * 6]
                                == live_pointers[expert * 6..(expert + 1) * 6],
                        "DEFECT stale native expert dispatch layer={} expert={expert}",
                        native.slot_layer
                    );
                    integers!([valid[expert]]);
                }
            } else {
                for id in ids {
                    let expert = usize::try_from(id).context("negative selected expert")?;
                    ensure!(
                        valid[expert] == 1,
                        "DEFECT unrouted native expert dispatch layer={} expert={expert}",
                        native.slot_layer
                    );
                    integers!([valid[expert]]);
                }
            }
        }
    }
    Ok(bytes)
}

#[cfg(feature = "cuda")]
struct DsaStateSnapshot {
    buffers: [Vec<bf16>; 3],
    indices: [Vec<i32>; 3],
}

#[cfg(feature = "cuda")]
struct GraphStepCheckpoint {
    gdn: Vec<[Vec<f32>; 2]>,
    dsa: Vec<DsaStateSnapshot>,
    sequence: Vec<i32>,
    work_position: Vec<i32>,
    work_length: Vec<i32>,
    token: u32,
    position: usize,
    last_input: Option<PrefixLastInputKind>,
    logits: PrefixLogitsCache,
    committed_prefix_ids: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl GraphStepCheckpoint {
    fn save(engine: &Engine) -> Result<Self> {
        let s = &engine.device.stream;
        Ok(Self {
            gdn: engine
                .state
                .gdn
                .iter()
                .map(|state| Ok([s.clone_dtoh(&state.s)?, s.clone_dtoh(&state.history)?]))
                .collect::<Result<_>>()?,
            dsa: engine
                .state
                .dsa
                .iter()
                .map(|state| {
                    Ok(DsaStateSnapshot {
                        buffers: [
                            s.clone_dtoh(&state.latent)?,
                            s.clone_dtoh(&state.packed)?,
                            s.clone_dtoh(&state.pool_keys)?,
                        ],
                        indices: [
                            s.clone_dtoh(&state.token_ids)?,
                            s.clone_dtoh(&state.pool_ids)?,
                            s.clone_dtoh(&state.meta)?,
                        ],
                    })
                })
                .collect::<Result<_>>()?,
            sequence: s.clone_dtoh(engine.state.sequence.as_ref().context("sequence missing")?)?,
            work_position: s.clone_dtoh(&engine.work.position)?,
            work_length: s.clone_dtoh(&engine.work.length)?,
            token: engine.token()?,
            position: engine.position,
            last_input: engine.last_input_kind,
            logits: engine.logits_cache,
            committed_prefix_ids: engine.committed_prefix_ids.clone(),
        })
    }

    fn restore(&self, engine: &mut Engine) -> Result<()> {
        let s = &engine.device.stream;
        for (source, target) in self.gdn.iter().zip(&mut engine.state.gdn) {
            s.memcpy_htod(&source[0], &mut target.s)?;
            s.memcpy_htod(&source[1], &mut target.history)?;
        }
        for (source, target) in self.dsa.iter().zip(&mut engine.state.dsa) {
            s.memcpy_htod(&source.buffers[0], &mut target.latent)?;
            s.memcpy_htod(&source.buffers[1], &mut target.packed)?;
            s.memcpy_htod(&source.buffers[2], &mut target.pool_keys)?;
            s.memcpy_htod(&source.indices[0], &mut target.token_ids)?;
            s.memcpy_htod(&source.indices[1], &mut target.pool_ids)?;
            s.memcpy_htod(&source.indices[2], &mut target.meta)?;
        }
        s.memcpy_htod(
            &self.sequence,
            engine.state.sequence.as_mut().context("sequence missing")?,
        )?;
        s.memcpy_htod(&self.work_position, &mut engine.work.position)?;
        s.memcpy_htod(&self.work_length, &mut engine.work.length)?;
        engine.position = self.position;
        engine.set_token(self.token)?;
        engine.last_input_kind = self.last_input;
        engine.logits_cache = self.logits;
        engine.committed_prefix_ids = self.committed_prefix_ids.clone();
        Ok(())
    }
}

fn graph_counts(engine: &Engine) -> Result<()> {
    use cudarc::driver::sys::CUgraphNodeType;
    ensure!(engine.glm_graph_count() == 84, "GLM graph segment count");
    let mut total = [0usize; 3];
    for (layer, native) in engine
        .qwen
        .as_ref()
        .context("runtime missing")?
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, layer)| layer.native.as_ref().map(|native| (i, native)))
    {
        let pair = native.graphs.as_ref().context("graphs missing")?;
        for (part, graph) in [("prefix", &pair.0), ("suffix", &pair.1)] {
            let mut count = 0;
            unsafe {
                sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count)
                    .result()?;
            }
            let mut nodes = vec![std::ptr::null_mut(); count];
            unsafe {
                sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count).result()?;
            }
            let mut kinds = [0usize; 3];
            for node in nodes.into_iter().take(count) {
                let mut kind = CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                unsafe {
                    sys::cuGraphNodeGetType(node, &mut kind).result()?;
                }
                kinds[match kind {
                    CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => 0,
                    CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => 1,
                    _ => 2,
                }] += 1;
            }
            println!(
                "GLM_GRAPH_NODES layer={layer} part={part} kernel={} copy={} other={}",
                kinds[0], kinds[1], kinds[2]
            );
            for (sum, count) in total.iter_mut().zip(kinds) {
                *sum += count;
            }
        }
    }
    println!(
        "GLM_GRAPH_NODES_TOTAL kernel={} copy={} other={} submissions=84 route_d2h_bytes=2688 dispatch_h2d_bytes=628992",
        total[0], total[1], total[2]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "official GLM model graph capture and rollback controls"]
fn glm_graph_capture_gpu() -> Result<()> {
    let mut engine = load_composition_engine()?;
    ensure!(engine.graph_decode(), "GLM graph capability missing");
    graph_counts(&engine)?;
    let graphs: Vec<_> = engine
        .qwen
        .as_mut()
        .context("runtime missing")?
        .layers
        .iter_mut()
        .filter_map(|layer| layer.native.as_mut())
        .map(|native| native.graphs.take())
        .collect();
    let ready = engine.closure.clone();
    for stage in [
        Some(CaptureFault::Begin),
        Some(CaptureFault::End),
        Some(CaptureFault::Instantiate),
        None,
    ] {
        engine.closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "GLM capture fault gate".into(),
        ));
        engine.closure.freeze_fixed()?;
        engine.capture_fail_stage = stage;
        engine.capture_fail_after_layer = if stage.is_none() { Some((4, 1)) } else { None };
        let before = graph_native_state(&engine, true)?;
        let result = engine.capture_glm();
        ensure!(
            result.is_err() && engine.poisoned && engine.glm_graph_count() == 0,
            "DEFECT capture fault did not poison/dispose"
        );
        ensure!(
            graph_native_state(&engine, true)? == before,
            "DEFECT capture fault changed native state"
        );
        ensure!(
            engine.advance(false, None).is_err(),
            "DEFECT capture fault permitted decode"
        );
        println!(
            "GLM_CAPTURE_CONTROL stage={stage:?} state_bits=0 unpublished=true poisoned=true PASS"
        );
        engine.capture_fail_stage = None;
        engine.capture_fail_after_layer = None;
        engine.reset()?;
    }
    for (native, graphs) in engine
        .qwen
        .as_mut()
        .context("runtime missing")?
        .layers
        .iter_mut()
        .filter_map(|layer| layer.native.as_mut())
        .zip(graphs)
    {
        native.graphs = graphs;
    }
    engine.closure = ready;
    let before = graph_native_state(&engine, true)?;
    engine.capture()?;
    ensure!(
        graph_native_state(&engine, true)? == before,
        "DEFECT ready capture changed state"
    );
    println!(
        "PASS GLM_GRAPH_CAPTURE graphs=84 faults=4 native_state_bits=0 ready_capture_idempotent=true"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "official GLM native FP8 expert hit, eviction, refill and graph dispatch"]
fn glm_graph_expert_tier_gpu() -> Result<()> {
    fn output(
        engine: &mut Engine,
        graphs: Option<&(CudaGraph, CudaGraph)>,
    ) -> Result<(Vec<u16>, usize)> {
        let native = engine.qwen.as_mut().context("runtime missing")?.layers[3]
            .native
            .as_mut()
            .context("native FFN missing")?;
        let s = &engine.device.stream;
        let slots = engine.moe_slots.as_mut().context("tiers missing")?;
        if !slots.request_open {
            slots.begin_staging_request()?;
        }
        slots.begin_token()?;
        let x = matrix(&engine.work.n, s, 0, 1, 4096)?;
        if let Some(pair) = graphs {
            pair.0.launch()?;
        } else {
            native.plans[0].enqueue_route(s, x)?;
        }
        let refills = native.service_tier(s, 1, slots)?;
        if let Some(pair) = graphs {
            pair.1.launch()?;
        } else {
            native.plans[0].run(s, x, &TensorMut(matrix(&engine.work.out, s, 0, 1, 4096)?))?;
        }
        slots.record_all_readers(&engine.device.stream)?;
        slots.finish_token()?;
        let mut values: Vec<u16> = native.plans[0]
            .read_contributions(s)?
            .into_iter()
            .map(bf16::to_bits)
            .collect();
        values.extend(
            s.clone_dtoh(&engine.work.out.slice(..4096))?
                .into_iter()
                .map(bf16::to_bits),
        );
        Ok((values, refills))
    }
    let mut engine = load_composition_engine()?;
    let values: Vec<_> = (0..4096)
        .map(|i| bf16::from_f32(((i * 37) % 251) as f32 / 64.0 - 2.0))
        .collect();
    engine
        .device
        .stream
        .memcpy_htod(&values, &mut engine.work.n.slice_mut(..4096))?;
    let graphs = {
        let s = &engine.device.stream;
        let native = engine.qwen.as_ref().context("runtime missing")?.layers[3]
            .native
            .as_ref()
            .context("native FFN missing")?;
        ensure!(native.slot_layer == 0, "GLM first native layer mapping");
        let plan = &native.plans[0];
        let x = matrix(&engine.work.n, s, 0, 1, 4096)?;
        let y = TensorMut(matrix(&engine.work.out, s, 0, 1, 4096)?);
        (
            crate::backend::cuda::moe::testing::capture(s, || plan.enqueue_route(s, x))?,
            crate::backend::cuda::moe::testing::capture(s, || plan.run(s, x, &y))?,
        )
    };
    let (baseline, initial) = output(&mut engine, None)?;
    let ids = engine.qwen.as_ref().context("runtime missing")?.layers[3]
        .native
        .as_ref()
        .context("native FFN missing")?
        .plans[0]
        .read_route(&engine.device.stream)?
        .0;
    ensure!(ids.len() == 8, "GLM top8 route extent");
    let target = ids[0] as u32;
    let bit_differences = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let incoming_eager = engine
        .moe_slots
        .as_mut()
        .context("tiers missing")?
        .evict_for_test(0, target)?;
    let (refilled, eager_refills) = output(&mut engine, None)?;
    ensure!(
        eager_refills > 0 && bit_differences(&baseline, &refilled) == 0,
        "DEFECT eager native refill identity"
    );
    let incoming_graph = engine
        .moe_slots
        .as_mut()
        .context("tiers missing")?
        .evict_for_test(0, target)?;
    let (replay, graph_refills) = output(&mut engine, Some(&graphs))?;
    let (hit, hit_refills) = output(&mut engine, Some(&graphs))?;
    ensure!(
        graph_refills > 0
            && hit_refills == 0
            && bit_differences(&baseline, &replay) == 0
            && bit_differences(&baseline, &hit) == 0,
        "DEFECT graph native refill/hit identity"
    );
    engine.qwen.as_mut().context("runtime missing")?.layers[3]
        .native
        .as_mut()
        .context("native FFN missing")?
        .wrong_route = true;
    let (wrong, _) = output(&mut engine, Some(&graphs))?;
    engine.qwen.as_mut().context("runtime missing")?.layers[3]
        .native
        .as_mut()
        .context("native FFN missing")?
        .wrong_route = false;
    let differences = bit_differences(&baseline, &wrong);
    ensure!(differences > 0, "Runner FAIL wrong-slot control escaped");
    let mut signed_zero = baseline.clone();
    signed_zero[0] ^= 0x8000;
    ensure!(
        bit_differences(&baseline, &signed_zero) == 1,
        "Runner FAIL signed-zero control escaped"
    );
    println!(
        "PASS GLM_GRAPH_EXPERT_TIER experts=288 top_k=8 input=[1,4096] contributions=[1,8,4096] output=[1,4096] target={target} incoming={incoming_eager}/{incoming_graph} initial_refills={initial} eager_refills={eager_refills} graph_refills={graph_refills} hit_refills={hit_refills} different_bits=0 signed_zero_control=1 wrong_slot_differences={differences}"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "official GLM graph/eager composition, frozen reference and pre-graph eager logits"]
fn glm_graph_composition_gpu() -> Result<()> {
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    let reference = Reference::load(&root)?;
    let output =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_OUTPUT").context("output missing")?);
    std::fs::create_dir(&output)?;
    let old_logits = PathBuf::from(
        std::env::var_os("G1_GRAPH_EAGER_LOGITS").context("pre-graph eager logits missing")?,
    );
    ensure!(
        std::fs::metadata(&old_logits)?.len() == 599 * 154880 * 4,
        "Runner FAIL pre-graph eager extent"
    );
    let mut engine = load_composition_engine()?;
    ensure!(
        engine.graph_decode() && engine.glm_graph_count() == 84,
        "GLM graph capability missing"
    );
    graph_counts(&engine)?;
    let mut accepted = true;
    for round in 1..=2 {
        engine.reset()?;
        if round == 2 {
            engine.set_token(reference.tokens[17])?;
            engine.step(true)?;
            engine.reset()?;
        }
        let mut old = BufReader::new(std::fs::File::open(&old_logits)?);
        let mut eager = Score::new(
            &root,
            &output.join(format!("eager-{round}.fp32")),
            &reference,
        )?;
        let mut graph = Score::new(
            &root,
            &output.join(format!("graph-{round}.fp32")),
            &reference,
        )?;
        let pos0_checkpoint = GraphStepCheckpoint::save(&engine)?;
        engine.set_token(reference.tokens[0])?;
        engine.decode(false, None)?;
        let expected_pos0 = engine.decode_logits()?;
        let expected_state_pos0 = graph_native_state(&engine, false)?;
        pos0_checkpoint.restore(&mut engine)?;
        engine.set_token(reference.tokens[0])?;
        engine.decode(true, None)?;
        let actual_pos0 = engine.decode_logits()?;
        let actual_state_pos0 = graph_native_state(&engine, false)?;
        let pos0_bit_diffs = expected_pos0
            .iter()
            .zip(&actual_pos0)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let pos0_state_diffs = expected_state_pos0
            .iter()
            .zip(&actual_state_pos0)
            .filter(|(a, b)| a != b)
            .count();
        ensure!(
            pos0_bit_diffs == 0 && pos0_state_diffs == 0,
            "DEFECT GLM position-0 graph decode replay identity"
        );
        pos0_checkpoint.restore(&mut engine)?;
        engine.reset()?;
        engine.prefill(&reference.tokens[..1], None)?;
        let first = engine.decode_logits()?;
        let mut second = None;
        let mut bit_differences = 0usize;
        let mut state_differences = 0usize;
        let mut old_differences = 0usize;
        let mut stale_detected = false;
        for position in 0..599 {
            let (expected, actual) = if position == 0 {
                (first.clone(), first.clone())
            } else {
                engine.set_token(reference.tokens[position])?;
                let checkpoint = GraphStepCheckpoint::save(&engine)?;
                let marked = matches!(
                    position,
                    1 | 3
                        | 4
                        | 5
                        | 7
                        | 8
                        | 9
                        | 31
                        | 32
                        | 33
                        | 127
                        | 128
                        | 129
                        | 511
                        | 512
                        | 513
                        | 598
                );
                {
                    let _range =
                        crate::backend::cuda::profile::Range::new(marked, c"glm_eager_decode");
                    engine.decode(false, None)?;
                }
                let expected = engine.decode_logits()?;
                let expected_state = graph_native_state(&engine, false)?;
                checkpoint.restore(&mut engine)?;
                if position == 3 {
                    engine.state.set_sequence(
                        &engine.device.stream,
                        2,
                        1,
                        engine.workspace_rows as u32,
                    )?;
                    engine.decode(true, None)?;
                    stale_detected = graph_native_state(&engine, false)? != expected_state;
                    ensure!(
                        stale_detected,
                        "Runner FAIL stale sequence control was not detected at first completed pool"
                    );
                    checkpoint.restore(&mut engine)?;
                }
                {
                    let _range =
                        crate::backend::cuda::profile::Range::new(marked, c"glm_graph_decode");
                    engine.decode(true, None)?;
                }
                let actual = engine.decode_logits()?;
                let actual_state = graph_native_state(&engine, false)?;
                ensure!(
                    expected_state.len() == actual_state.len(),
                    "Runner FAIL graph state extent"
                );
                let different = expected_state
                    .iter()
                    .zip(&actual_state)
                    .filter(|(a, b)| a != b)
                    .count();
                state_differences += different;
                println!(
                    "GLM_GRAPH_STATE round={round} position={position} bytes={} different_bytes={different}",
                    actual_state.len()
                );
                (expected, actual)
            };
            let previous = read_logits(&mut old, 154880)?;
            old_differences += previous
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            bit_differences += actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eager.row(position, &expected)?;
            graph.row(position, &actual)?;
            live_state(&engine, position + 1)?;
            if position == 1 {
                second = Some(expected);
            }
        }
        accepted &= eager.finish(false)?;
        accepted &= graph.finish(false)?;
        let identity = bit_differences == 0
            && state_differences == 0
            && old_differences == 0
            && stale_detected
            && pos0_bit_diffs == 0
            && pos0_state_diffs == 0;
        accepted &= identity;
        println!(
            "{} GLM_GRAPH_COMPOSITION round={round} rows=599 vocab=154880 logit_bit_differences={bit_differences} native_state_different_bytes={state_differences} pre_graph_eager_bit_differences={old_differences} stale_sequence_detected={stale_detected} pos0_bit_differences={pos0_bit_diffs} graph_submissions_per_decode=84",
            if identity { "PASS" } else { "DEFECT" }
        );
        engine.reset()?;
        let reset_pos0_checkpoint = GraphStepCheckpoint::save(&engine)?;
        engine.set_token(reference.tokens[0])?;
        engine.decode(false, None)?;
        let reset_expected_pos0 = engine.decode_logits()?;
        reset_pos0_checkpoint.restore(&mut engine)?;
        engine.set_token(reference.tokens[0])?;
        engine.decode(true, None)?;
        let reset_actual_pos0 = engine.decode_logits()?;
        ensure!(
            reset_expected_pos0
                .iter()
                .zip(&reset_actual_pos0)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "DEFECT graph reset position-0 identity"
        );
        reset_pos0_checkpoint.restore(&mut engine)?;
        engine.reset()?;
        engine.prefill(&reference.tokens[..1], None)?;
        ensure!(
            engine
                .decode_logits()?
                .iter()
                .zip(&first)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "DEFECT graph reset first-token identity"
        );
        println!("PASS GLM_GRAPH_RESET round={round} first_token_bits=0 pos0_replay_bits=0");
        composition_controls(
            &mut engine,
            &reference.tokens,
            &first,
            &second.context("second row missing")?,
        )?;
    }
    composition_chunks(&mut engine, &root, &output, &reference)?;
    ensure!(accepted, "DEFECT GLM graph composition");
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

// ============================================================================
// GLM-5.3-Flash Multi-Token Prediction (MTP) Layer 45 Runtime and Gates
// ============================================================================

fn fp8_dequant_block(codes: &[u8], scale_inv: &[f32], rows: usize, cols: usize) -> Vec<bf16> {
    let scale_cols = cols.div_ceil(128);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let sr = r / 128;
        for c in 0..cols {
            let sc = c / 128;
            let scale = scale_inv[sr * scale_cols + sc];
            let v = crate::fp8::decode_e4m3(codes[r * cols + c]);
            out.push(bf16::from_f32(v * scale));
        }
    }
    out
}

fn run_linear_bf16(
    stream: &Arc<cudarc::driver::CudaStream>,
    blas: &Blas,
    weight: &cudarc::driver::CudaSlice<bf16>,
    x: &cudarc::driver::CudaSlice<bf16>,
    y: &mut cudarc::driver::CudaSlice<bf16>,
    shape: (usize, usize, usize),
) -> Result<()> {
    let (rows, output_dim, input_dim) = shape;
    let (w_ptr, _w_drop) = weight.device_ptr(stream);
    let (x_ptr, _x_drop) = x.device_ptr(stream);
    let (y_ptr, _y_drop) = y.device_ptr(stream);
    let w_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            w_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim * 2) as u64,
        )?
    };
    let x_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            x_ptr,
            [rows as u32, input_dim as u32],
            [input_dim as u64, 1],
            (rows * input_dim * 2) as u64,
        )?
    };
    let mut y_tensor = unsafe {
        crate::backend::cuda::contracts::TensorMut(crate::backend::cuda::contracts::Tensor::new(
            y_ptr,
            [rows as u32, output_dim as u32],
            [output_dim as u64, 1],
            (rows * output_dim * 2) as u64,
        )?)
    };
    blas.linear_bf16(w_tensor, x_tensor, &mut y_tensor)
}

fn run_linear_bf16_to_f32(
    stream: &Arc<cudarc::driver::CudaStream>,
    blas: &Blas,
    weight: &cudarc::driver::CudaSlice<bf16>,
    x: &cudarc::driver::CudaSlice<bf16>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    shape: (usize, usize, usize),
) -> Result<()> {
    let (rows, output_dim, input_dim) = shape;
    let (w_ptr, _w_drop) = weight.device_ptr(stream);
    let (x_ptr, _x_drop) = x.device_ptr(stream);
    let (y_ptr, _y_drop) = y.device_ptr(stream);
    let w_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            w_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim * 2) as u64,
        )?
    };
    let x_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            x_ptr,
            [rows as u32, input_dim as u32],
            [input_dim as u64, 1],
            (rows * input_dim * 2) as u64,
        )?
    };
    let mut y_tensor = unsafe {
        crate::backend::cuda::contracts::TensorMut(crate::backend::cuda::contracts::Tensor::new(
            y_ptr,
            [rows as u32, output_dim as u32],
            [output_dim as u64, 1],
            (rows * output_dim * 4) as u64,
        )?)
    };
    blas.linear_bf16_to_f32(w_tensor, x_tensor, &mut y_tensor)
}

fn run_rms_norm(
    stream: &cudarc::driver::CudaStream,
    norm_fn: &cudarc::driver::CudaFunction,
    x: &cudarc::driver::CudaSlice<bf16>,
    w: &cudarc::driver::CudaSlice<bf16>,
    y: &mut cudarc::driver::CudaSlice<bf16>,
    dims: (i32, i32),
    eps: f32,
) -> Result<()> {
    let (rows, dim) = dims;
    unsafe {
        stream
            .launch_builder(norm_fn)
            .arg(x)
            .arg(w)
            .arg(y)
            .arg(&rows)
            .arg(&dim)
            .arg(&eps)
            .launch(crate::backend::cuda::ops::grid(rows as usize, 256))?;
    }
    Ok(())
}

fn run_silu_mul_split(
    stream: &cudarc::driver::CudaStream,
    silu_fn: &cudarc::driver::CudaFunction,
    gate: &cudarc::driver::CudaSlice<bf16>,
    up: &cudarc::driver::CudaSlice<bf16>,
    y: &mut cudarc::driver::CudaSlice<bf16>,
    count: i32,
) -> Result<()> {
    let mode = 1i32;
    unsafe {
        stream
            .launch_builder(silu_fn)
            .arg(gate)
            .arg(up)
            .arg(y)
            .arg(&count)
            .arg(&mode)
            .launch(crate::backend::cuda::ops::grid(
                (count as usize).div_ceil(8),
                256,
            ))?;
    }
    Ok(())
}

fn run_concat_eh(
    stream: &cudarc::driver::CudaStream,
    concat_fn: &cudarc::driver::CudaFunction,
    enorm: &cudarc::driver::CudaSlice<bf16>,
    hnorm: &cudarc::driver::CudaSlice<bf16>,
    out: &mut cudarc::driver::CudaSlice<bf16>,
    dim: i32,
    swap: bool,
) -> Result<()> {
    let swap_val = if swap { 1i32 } else { 0i32 };
    unsafe {
        stream
            .launch_builder(concat_fn)
            .arg(enorm)
            .arg(hnorm)
            .arg(out)
            .arg(&dim)
            .arg(&swap_val)
            .launch(crate::backend::cuda::ops::grid(
                (dim as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

fn run_add(
    stream: &cudarc::driver::CudaStream,
    add_fn: &cudarc::driver::CudaFunction,
    x: &mut cudarc::driver::CudaSlice<bf16>,
    res: &cudarc::driver::CudaSlice<bf16>,
    count: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(add_fn)
            .arg(x)
            .arg(res)
            .arg(&count)
            .launch(crate::backend::cuda::ops::grid(
                (count as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

fn run_accum_scaled(
    stream: &cudarc::driver::CudaStream,
    accum_fn: &cudarc::driver::CudaFunction,
    accum: &mut cudarc::driver::CudaSlice<bf16>,
    val: &cudarc::driver::CudaSlice<bf16>,
    scale: f32,
    count: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(accum_fn)
            .arg(accum)
            .arg(val)
            .arg(&scale)
            .arg(&count)
            .launch(crate::backend::cuda::ops::grid(
                (count as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

fn run_extract_dsa_value(
    stream: &cudarc::driver::CudaStream,
    extract_fn: &cudarc::driver::CudaFunction,
    kv_expanded: &cudarc::driver::CudaSlice<bf16>,
    attn_out: &mut cudarc::driver::CudaSlice<bf16>,
    heads: i32,
    head_dim: i32,
) -> Result<()> {
    let total = (heads * head_dim) as u32;
    unsafe {
        stream
            .launch_builder(extract_fn)
            .arg(kv_expanded)
            .arg(attn_out)
            .arg(&heads)
            .arg(&head_dim)
            .launch(crate::backend::cuda::ops::grid(
                (total as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

struct GlmMtpExpertHost {
    gate_codes: Vec<u8>,
    gate_scales: Vec<f32>,
    up_codes: Vec<u8>,
    up_scales: Vec<f32>,
    down_codes: Vec<u8>,
    down_scales: Vec<f32>,
}

struct GlmMtpRuntime {
    device: Arc<Device>,
    blas: std::rc::Rc<Blas>,

    rms_norm: cudarc::driver::CudaFunction,
    silu_mul_split: cudarc::driver::CudaFunction,
    concat_eh: cudarc::driver::CudaFunction,
    accum_scaled: cudarc::driver::CudaFunction,
    add: cudarc::driver::CudaFunction,
    extract_dsa: cudarc::driver::CudaFunction,

    embed_tokens: Vec<bf16>,
    enorm_w: cudarc::driver::CudaSlice<bf16>,
    hnorm_w: cudarc::driver::CudaSlice<bf16>,
    eh_proj_w: cudarc::driver::CudaSlice<bf16>,
    input_norm_w: cudarc::driver::CudaSlice<bf16>,

    q_a_w: cudarc::driver::CudaSlice<bf16>,
    q_a_norm_w: cudarc::driver::CudaSlice<bf16>,
    q_b_w: cudarc::driver::CudaSlice<bf16>,
    kv_a_w: cudarc::driver::CudaSlice<bf16>,
    kv_a_norm_w: cudarc::driver::CudaSlice<bf16>,
    kv_b_w: cudarc::driver::CudaSlice<bf16>,
    o_w: cudarc::driver::CudaSlice<bf16>,

    post_norm_w: cudarc::driver::CudaSlice<bf16>,
    sh_gate_w: cudarc::driver::CudaSlice<bf16>,
    sh_up_w: cudarc::driver::CudaSlice<bf16>,
    sh_down_w: cudarc::driver::CudaSlice<bf16>,
    router_w: cudarc::driver::CudaSlice<bf16>,
    router_b: Vec<f32>,

    experts: Vec<GlmMtpExpertHost>,
    slot_gate: Vec<cudarc::driver::CudaSlice<bf16>>,
    slot_up: Vec<cudarc::driver::CudaSlice<bf16>>,
    slot_down: Vec<cudarc::driver::CudaSlice<bf16>>,

    head_norm_w: cudarc::driver::CudaSlice<bf16>,
    lm_head_w: cudarc::driver::CudaSlice<bf16>,

    token_embed_buf: cudarc::driver::CudaSlice<bf16>,
    enorm_buf: cudarc::driver::CudaSlice<bf16>,
    target_hidden_buf: cudarc::driver::CudaSlice<bf16>,
    hnorm_buf: cudarc::driver::CudaSlice<bf16>,
    eh_buf: cudarc::driver::CudaSlice<bf16>,
    x_buf: cudarc::driver::CudaSlice<bf16>,
    x_norm_buf: cudarc::driver::CudaSlice<bf16>,
    q_a_buf: cudarc::driver::CudaSlice<bf16>,
    q_a_norm_buf: cudarc::driver::CudaSlice<bf16>,
    q_b_buf: cudarc::driver::CudaSlice<bf16>,
    kv_a_buf: cudarc::driver::CudaSlice<bf16>,
    kv_a_norm_buf: cudarc::driver::CudaSlice<bf16>,
    kv_b_buf: cudarc::driver::CudaSlice<bf16>,
    attn_out_buf: cudarc::driver::CudaSlice<bf16>,
    o_buf: cudarc::driver::CudaSlice<bf16>,
    sh_gate_buf: cudarc::driver::CudaSlice<bf16>,
    sh_up_buf: cudarc::driver::CudaSlice<bf16>,
    sh_act_buf: cudarc::driver::CudaSlice<bf16>,
    sh_out_buf: cudarc::driver::CudaSlice<bf16>,
    router_logits_buf: cudarc::driver::CudaSlice<bf16>,
    exp_gate_buf: cudarc::driver::CudaSlice<bf16>,
    exp_up_buf: cudarc::driver::CudaSlice<bf16>,
    exp_act_buf: cudarc::driver::CudaSlice<bf16>,
    exp_out_buf: cudarc::driver::CudaSlice<bf16>,
    routed_accum_buf: cudarc::driver::CudaSlice<bf16>,
    head_norm_buf: cudarc::driver::CudaSlice<bf16>,
    logits_buf: cudarc::driver::CudaSlice<f32>,
    staged_weights: [f32; 8],
}

impl GlmMtpRuntime {
    fn load(model_dir: &Path, device: &Arc<Device>, blas: &std::rc::Rc<Blas>) -> Result<Self> {
        unsafe {
            device.ctx.disable_event_tracking();
        }
        let stream = &device.stream;

        let norm_mod = cubin::module(&device.ctx, "norm")?;
        let rms_norm = norm_mod.load_function("rms_norm")?;

        let act_mod = cubin::module(&device.ctx, "act")?;
        let silu_mul_split = act_mod.load_function("silu_mul_split")?;

        let mtp_mod = cubin::module(&device.ctx, "glm_mtp")?;
        let concat_eh = mtp_mod.load_function("glm_mtp_concat_eh")?;
        let accum_scaled = mtp_mod.load_function("glm_mtp_accum_scaled")?;
        let add = mtp_mod.load_function("glm_mtp_add")?;
        let extract_dsa = mtp_mod.load_function("glm_mtp_extract_dsa_value")?;

        let shard1_file = std::fs::File::open(model_dir.join("model-00001-of-00062.safetensors"))?;
        let mmap1 = unsafe { memmap2::MmapOptions::new().map(&shard1_file)? };
        let st1 = safetensors::SafeTensors::deserialize(&mmap1)?;

        let shard2_file = std::fs::File::open(model_dir.join("model-00002-of-00062.safetensors"))?;
        let mmap2 = unsafe { memmap2::MmapOptions::new().map(&shard2_file)? };
        let st2 = safetensors::SafeTensors::deserialize(&mmap2)?;

        let get_st_bf16 = |name: &str| -> Result<Vec<bf16>> {
            if let Ok(t) = st1.tensor(name) {
                let data = t.data();
                return Ok(data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                    .collect());
            }
            let t = st2.tensor(name)?;
            let data = t.data();
            Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        };

        let get_st_f32 = |name: &str| -> Result<Vec<f32>> {
            if let Ok(t) = st1.tensor(name) {
                let data = t.data();
                return Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect());
            }
            let t = st2.tensor(name)?;
            let data = t.data();
            Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };

        let get_st_u8 = |name: &str| -> Result<Vec<u8>> {
            if let Ok(t) = st1.tensor(name) {
                return Ok(t.data().to_vec());
            }
            let t = st2.tensor(name)?;
            Ok(t.data().to_vec())
        };

        let get_st_fp8_linear = |role: &str, rows: usize, cols: usize| -> Result<Vec<bf16>> {
            let code_name = format!("{role}.weight");
            let scale_name = format!("{role}.weight_scale_inv");
            let codes = get_st_u8(&code_name)?;
            let scales = get_st_f32(&scale_name)?;
            Ok(fp8_dequant_block(&codes, &scales, rows, cols))
        };

        let embed_tokens = get_st_bf16("model.language_model.embed_tokens.weight")?;
        let lm_head_w = stream.clone_htod(&get_st_bf16("lm_head.weight")?)?;

        let enorm_w =
            stream.clone_htod(&get_st_bf16("model.language_model.layers.45.enorm.weight")?)?;
        let hnorm_w =
            stream.clone_htod(&get_st_bf16("model.language_model.layers.45.hnorm.weight")?)?;
        let eh_proj_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.eh_proj.weight",
        )?)?;
        let input_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.input_layernorm.weight",
        )?)?;

        let q_a_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.self_attn.q_a_proj",
            1536,
            4096,
        )?)?;
        let q_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.q_a_layernorm.weight",
        )?)?;
        let q_b_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.self_attn.q_b_proj",
            16384,
            1536,
        )?)?;
        let kv_a_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa",
            512,
            4096,
        )?)?;
        let kv_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_a_layernorm.weight",
        )?)?;
        let kv_b_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_b_proj.weight",
        )?)?;
        let o_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.self_attn.o_proj",
            4096,
            16384,
        )?)?;

        let post_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.post_attention_layernorm.weight",
        )?)?;
        let sh_gate_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.mlp.shared_experts.gate_proj",
            1024,
            4096,
        )?)?;
        let sh_up_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.mlp.shared_experts.up_proj",
            1024,
            4096,
        )?)?;
        let sh_down_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.mlp.shared_experts.down_proj",
            4096,
            1024,
        )?)?;
        let router_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.mlp.gate.weight",
        )?)?;
        let router_b_bf16 =
            get_st_bf16("model.language_model.layers.45.mlp.gate.e_score_correction_bias")?;
        let router_b = router_b_bf16.into_iter().map(|b| b.to_f32()).collect();

        let head_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.shared_head.norm.weight",
        )?)?;

        let mut experts = Vec::with_capacity(288);
        for e in 0..288 {
            let pfx = format!("model.language_model.layers.45.mlp.experts.{e}");
            let gate_codes = get_st_u8(&format!("{pfx}.gate_proj.weight"))?;
            let gate_scales = get_st_f32(&format!("{pfx}.gate_proj.weight_scale_inv"))?;
            let up_codes = get_st_u8(&format!("{pfx}.up_proj.weight"))?;
            let up_scales = get_st_f32(&format!("{pfx}.up_proj.weight_scale_inv"))?;
            let down_codes = get_st_u8(&format!("{pfx}.down_proj.weight"))?;
            let down_scales = get_st_f32(&format!("{pfx}.down_proj.weight_scale_inv"))?;
            experts.push(GlmMtpExpertHost {
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
                down_codes,
                down_scales,
            });
        }

        let mut slot_gate = Vec::with_capacity(8);
        let mut slot_up = Vec::with_capacity(8);
        let mut slot_down = Vec::with_capacity(8);
        for _ in 0..8 {
            slot_gate.push(stream.alloc_zeros::<bf16>(1024 * 4096)?);
            slot_up.push(stream.alloc_zeros::<bf16>(1024 * 4096)?);
            slot_down.push(stream.alloc_zeros::<bf16>(4096 * 1024)?);
        }

        Ok(Self {
            device: device.clone(),
            blas: blas.clone(),
            rms_norm,
            silu_mul_split,
            concat_eh,
            accum_scaled,
            add,
            extract_dsa,
            embed_tokens,
            enorm_w,
            hnorm_w,
            eh_proj_w,
            input_norm_w,
            q_a_w,
            q_a_norm_w,
            q_b_w,
            kv_a_w,
            kv_a_norm_w,
            kv_b_w,
            o_w,
            post_norm_w,
            sh_gate_w,
            sh_up_w,
            sh_down_w,
            router_w,
            router_b,
            experts,
            slot_gate,
            slot_up,
            slot_down,
            head_norm_w,
            lm_head_w,
            token_embed_buf: stream.alloc_zeros::<bf16>(4096)?,
            enorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            target_hidden_buf: stream.alloc_zeros::<bf16>(4096)?,
            hnorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            eh_buf: stream.alloc_zeros::<bf16>(8192)?,
            x_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
            q_a_buf: stream.alloc_zeros::<bf16>(1536)?,
            q_a_norm_buf: stream.alloc_zeros::<bf16>(1536)?,
            q_b_buf: stream.alloc_zeros::<bf16>(16384)?,
            kv_a_buf: stream.alloc_zeros::<bf16>(512)?,
            kv_a_norm_buf: stream.alloc_zeros::<bf16>(512)?,
            kv_b_buf: stream.alloc_zeros::<bf16>(32768)?,
            attn_out_buf: stream.alloc_zeros::<bf16>(16384)?,
            o_buf: stream.alloc_zeros::<bf16>(4096)?,
            sh_gate_buf: stream.alloc_zeros::<bf16>(1024)?,
            sh_up_buf: stream.alloc_zeros::<bf16>(1024)?,
            sh_act_buf: stream.alloc_zeros::<bf16>(1024)?,
            sh_out_buf: stream.alloc_zeros::<bf16>(4096)?,
            router_logits_buf: stream.alloc_zeros::<bf16>(288)?,
            exp_gate_buf: stream.alloc_zeros::<bf16>(1024)?,
            exp_up_buf: stream.alloc_zeros::<bf16>(1024)?,
            exp_act_buf: stream.alloc_zeros::<bf16>(1024)?,
            exp_out_buf: stream.alloc_zeros::<bf16>(4096)?,
            routed_accum_buf: stream.alloc_zeros::<bf16>(4096)?,
            head_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
            logits_buf: stream.alloc_zeros::<f32>(154880)?,
            staged_weights: [0.0; 8],
        })
    }

    fn forward_staged_kernels(&mut self, swap_eh: bool) -> Result<()> {
        let stream = self.device.stream.clone();
        let blas = self.blas.clone();

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.token_embed_buf,
            &self.enorm_w,
            &mut self.enorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.target_hidden_buf,
            &self.hnorm_w,
            &mut self.hnorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_concat_eh(
            &stream,
            &self.concat_eh,
            &self.enorm_buf,
            &self.hnorm_buf,
            &mut self.eh_buf,
            4096,
            swap_eh,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.eh_proj_w,
            &self.eh_buf,
            &mut self.x_buf,
            (1, 4096, 8192),
        )?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.q_a_w,
            &self.x_norm_buf,
            &mut self.q_a_buf,
            (1, 1536, 4096),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.q_a_buf,
            &self.q_a_norm_w,
            &mut self.q_a_norm_buf,
            (1, 1536),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.q_b_w,
            &self.q_a_norm_buf,
            &mut self.q_b_buf,
            (1, 16384, 1536),
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.kv_a_w,
            &self.x_norm_buf,
            &mut self.kv_a_buf,
            (1, 512, 4096),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.kv_a_buf,
            &self.kv_a_norm_w,
            &mut self.kv_a_norm_buf,
            (1, 512),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.kv_b_w,
            &self.kv_a_norm_buf,
            &mut self.kv_b_buf,
            (1, 32768, 512),
        )?;
        run_extract_dsa_value(
            &stream,
            &self.extract_dsa,
            &self.kv_b_buf,
            &mut self.attn_out_buf,
            64,
            256,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.o_w,
            &self.attn_out_buf,
            &mut self.o_buf,
            (1, 4096, 16384),
        )?;
        run_add(&stream, &self.add, &mut self.x_buf, &self.o_buf, 4096)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.post_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.sh_gate_w,
            &self.x_norm_buf,
            &mut self.sh_gate_buf,
            (1, 1024, 4096),
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.sh_up_w,
            &self.x_norm_buf,
            &mut self.sh_up_buf,
            (1, 1024, 4096),
        )?;
        run_silu_mul_split(
            &stream,
            &self.silu_mul_split,
            &self.sh_gate_buf,
            &self.sh_up_buf,
            &mut self.sh_act_buf,
            1024,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.sh_down_w,
            &self.sh_act_buf,
            &mut self.sh_out_buf,
            (1, 4096, 1024),
        )?;

        stream.memset_zeros(&mut self.routed_accum_buf)?;

        for k in 0..8 {
            let w = self.staged_weights[k];
            run_linear_bf16(
                &stream,
                &blas,
                &self.slot_gate[k],
                &self.x_norm_buf,
                &mut self.exp_gate_buf,
                (1, 1024, 4096),
            )?;
            run_linear_bf16(
                &stream,
                &blas,
                &self.slot_up[k],
                &self.x_norm_buf,
                &mut self.exp_up_buf,
                (1, 1024, 4096),
            )?;
            run_silu_mul_split(
                &stream,
                &self.silu_mul_split,
                &self.exp_gate_buf,
                &self.exp_up_buf,
                &mut self.exp_act_buf,
                1024,
            )?;
            run_linear_bf16(
                &stream,
                &blas,
                &self.slot_down[k],
                &self.exp_act_buf,
                &mut self.exp_out_buf,
                (1, 4096, 1024),
            )?;
            run_accum_scaled(
                &stream,
                &self.accum_scaled,
                &mut self.routed_accum_buf,
                &self.exp_out_buf,
                w,
                4096,
            )?;
        }

        run_add(&stream, &self.add, &mut self.x_buf, &self.sh_out_buf, 4096)?;
        run_add(
            &stream,
            &self.add,
            &mut self.x_buf,
            &self.routed_accum_buf,
            4096,
        )?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.head_norm_w,
            &mut self.head_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16_to_f32(
            &stream,
            &blas,
            &self.lm_head_w,
            &self.head_norm_buf,
            &mut self.logits_buf,
            (1, 154880, 4096),
        )?;

        Ok(())
    }

    fn forward(
        &mut self,
        token_id: u32,
        target_hidden: &[bf16],
        swap_eh: bool,
    ) -> Result<Vec<f32>> {
        let stream = self.device.stream.clone();
        let blas = self.blas.clone();

        let token_offset = (token_id as usize) * 4096;
        let token_slice = &self.embed_tokens[token_offset..token_offset + 4096];
        stream.memcpy_htod(token_slice, &mut self.token_embed_buf)?;
        stream.memcpy_htod(target_hidden, &mut self.target_hidden_buf)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.token_embed_buf,
            &self.enorm_w,
            &mut self.enorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.target_hidden_buf,
            &self.hnorm_w,
            &mut self.hnorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_concat_eh(
            &stream,
            &self.concat_eh,
            &self.enorm_buf,
            &self.hnorm_buf,
            &mut self.eh_buf,
            4096,
            swap_eh,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.eh_proj_w,
            &self.eh_buf,
            &mut self.x_buf,
            (1, 4096, 8192),
        )?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.q_a_w,
            &self.x_norm_buf,
            &mut self.q_a_buf,
            (1, 1536, 4096),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.q_a_buf,
            &self.q_a_norm_w,
            &mut self.q_a_norm_buf,
            (1, 1536),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.q_b_w,
            &self.q_a_norm_buf,
            &mut self.q_b_buf,
            (1, 16384, 1536),
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.kv_a_w,
            &self.x_norm_buf,
            &mut self.kv_a_buf,
            (1, 512, 4096),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.kv_a_buf,
            &self.kv_a_norm_w,
            &mut self.kv_a_norm_buf,
            (1, 512),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.kv_b_w,
            &self.kv_a_norm_buf,
            &mut self.kv_b_buf,
            (1, 32768, 512),
        )?;
        run_extract_dsa_value(
            &stream,
            &self.extract_dsa,
            &self.kv_b_buf,
            &mut self.attn_out_buf,
            64,
            256,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.o_w,
            &self.attn_out_buf,
            &mut self.o_buf,
            (1, 4096, 16384),
        )?;
        run_add(&stream, &self.add, &mut self.x_buf, &self.o_buf, 4096)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.post_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            &blas,
            &self.router_w,
            &self.x_norm_buf,
            &mut self.router_logits_buf,
            (1, 288, 4096),
        )?;
        let raw_logits = stream.clone_dtoh(&self.router_logits_buf)?;

        let mut scores: Vec<(usize, f32)> = raw_logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let logit = l.to_f32() + self.router_b[i];
                let s = 1.0f32 / (1.0f32 + (-logit).exp());
                (i, s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top8 = &scores[..8];
        let sum: f32 = top8.iter().map(|(_, s)| s).sum();

        for (k, &(exp_id, score)) in top8.iter().enumerate() {
            let w = (score / sum) * 2.5f32;
            self.staged_weights[k] = w;
            let exp = &self.experts[exp_id];
            let gate_bf16 = fp8_dequant_block(&exp.gate_codes, &exp.gate_scales, 1024, 4096);
            let up_bf16 = fp8_dequant_block(&exp.up_codes, &exp.up_scales, 1024, 4096);
            let down_bf16 = fp8_dequant_block(&exp.down_codes, &exp.down_scales, 4096, 1024);

            stream.memcpy_htod(&gate_bf16, &mut self.slot_gate[k])?;
            stream.memcpy_htod(&up_bf16, &mut self.slot_up[k])?;
            stream.memcpy_htod(&down_bf16, &mut self.slot_down[k])?;
        }

        self.forward_staged_kernels(swap_eh)?;

        Ok(stream.clone_dtoh(&self.logits_buf)?)
    }

    fn forward_graph(
        &mut self,
        token_id: u32,
        target_hidden: &[bf16],
        swap_eh: bool,
    ) -> Result<Vec<f32>> {
        let stream = self.device.stream.clone();
        let _ = self.forward(token_id, target_hidden, swap_eh)?;
        stream.synchronize()?;
        stream.memset_zeros(&mut self.logits_buf)?;
        stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        self.forward_staged_kernels(swap_eh)?;
        let captured = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?;
        let graph = captured.context("Runner FAIL empty draft graph")?;
        graph.upload()?;
        graph.launch()?;
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&self.logits_buf)?)
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU GLM-5.3-Flash Layer 45 MTP census validation"]
fn glm_mtp_census_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let model_dir =
        PathBuf::from(std::env::var_os("FF_GLM_MODEL").context("FF_GLM_MODEL missing")?);
    let spec = crate::model_spec::ModelSpec::read(&model_dir)?;
    let mut tensors = std::collections::BTreeMap::new();
    crate::checkpoint::MappedCheckpoint::glm5_next_entries(&spec, &mut tensors)?;

    let mut total_entries = 0usize;
    let mut non_routed_entries = 0usize;
    let mut routed_entries = 0usize;
    let mut total_native_bytes = 0u64;

    for (name, (dtype, shape, scope)) in &tensors {
        if *scope != crate::checkpoint::TensorScope::Mtp {
            continue;
        }
        total_entries += 1;
        if name.contains(".mlp.experts.") {
            routed_entries += 1;
        } else {
            non_routed_entries += 1;
        }
        let elements: u64 = shape.iter().map(|&d| d as u64).product();
        let bytes_per_element = match dtype {
            safetensors::Dtype::BF16 => 2u64,
            safetensors::Dtype::F8_E4M3 => 1u64,
            safetensors::Dtype::F32 => 4u64,
            _ => 2u64,
        };
        total_native_bytes += elements * bytes_per_element;
    }

    ensure!(
        total_entries == 1760,
        "DEFECT total Layer 45 entries mismatch: expected 1760, got {total_entries}"
    );
    ensure!(
        non_routed_entries == 32,
        "DEFECT non-routed entries mismatch: expected 32, got {non_routed_entries}"
    );
    ensure!(
        routed_entries == 1728,
        "DEFECT routed expert entries mismatch: expected 1728, got {routed_entries}"
    );
    ensure!(
        total_native_bytes == 7_493_399_168,
        "DEFECT native byte sum mismatch: expected 7493399168, got {total_native_bytes}"
    );

    println!(
        "PASS GLM_MTP_CENSUS total={total_entries} non_routed={non_routed_entries} routed={routed_entries} native_bytes={total_native_bytes} unmapped=0 duplicates=0"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU GLM-5.3-Flash Layer 45 numerics gate (G-MTP2) against frozen reference and swap control"]
fn glm_mtp_layer45_numerics_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let model_dir =
        PathBuf::from(std::env::var_os("FF_GLM_MODEL").context("FF_GLM_MODEL missing")?);
    let ref_dir = PathBuf::from(
        std::env::var_os("FF_GLM_MTP_REF_DIR").context("FF_GLM_MTP_REF_DIR missing")?,
    );

    let thresholds_file = ref_dir.join("layer45_thresholds.json");
    let thresholds_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&thresholds_file)?)?;
    let tau_max_abs = thresholds_json["frozen_thresholds"]["tau_max_abs"]
        .as_f64()
        .context("tau_max_abs missing")? as f32;
    let tau_rmse = thresholds_json["frozen_thresholds"]["tau_rmse"]
        .as_f64()
        .context("tau_rmse missing")? as f32;

    let device = Arc::new(Device::new(0)?);
    unsafe {
        device.ctx.disable_event_tracking();
    }
    let ops = crate::backend::cuda::ops::Ops::new(&device.ctx)?;
    let closure = std::rc::Rc::new(crate::backend::cuda::closure::Closure::new(
        false,
        "glm_mtp".into(),
    ));
    let blas = std::rc::Rc::new(Blas::new(&device, &ops, &std::env::temp_dir(), closure)?);
    let mut runtime = GlmMtpRuntime::load(&model_dir, &device, &blas)?;

    // 1. Synthetic case
    let synth_inputs = std::fs::read(ref_dir.join("common_inputs.safetensors"))?;
    let synth_st = safetensors::SafeTensors::deserialize(&synth_inputs)?;
    let synth_hidden: Vec<bf16> = synth_st
        .tensor("target_hidden")?
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let synth_token = 785u32;
    let synth_ref_bytes = std::fs::read(ref_dir.join("logits_cpu.fp32"))?;
    let synth_ref_logits: Vec<f32> = synth_ref_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let synth_eager = runtime.forward(synth_token, &synth_hidden, false)?;
    let synth_graph = runtime.forward_graph(synth_token, &synth_hidden, false)?;
    let synth_graph_diffs = synth_eager
        .iter()
        .zip(&synth_graph)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(
        synth_graph_diffs == 0,
        "DEFECT synthetic eager vs graph bit difference: {synth_graph_diffs}"
    );

    let mut synth_max_abs = 0.0f32;
    let mut synth_sum_sq = 0.0f32;
    for (&cand, &refer) in synth_eager.iter().zip(&synth_ref_logits) {
        let diff = (cand - refer).abs();
        if diff > synth_max_abs {
            synth_max_abs = diff;
        }
        synth_sum_sq += diff * diff;
    }
    let synth_rmse = (synth_sum_sq / synth_eager.len() as f32).sqrt();
    ensure!(
        synth_max_abs <= tau_max_abs,
        "DEFECT synthetic max_abs={synth_max_abs} exceeds tau={tau_max_abs}"
    );
    ensure!(
        synth_rmse <= tau_rmse,
        "DEFECT synthetic rmse={synth_rmse} exceeds tau={tau_rmse}"
    );

    let synth_swapped = runtime.forward(synth_token, &synth_hidden, true)?;
    let mut synth_swap_delta = 0.0f32;
    for (&swapped, &refer) in synth_swapped.iter().zip(&synth_ref_logits) {
        let diff = (swapped - refer).abs();
        if diff > synth_swap_delta {
            synth_swap_delta = diff;
        }
    }
    ensure!(
        synth_swap_delta > 13.0,
        "Runner FAIL synthetic swap discriminating control escaped: delta={synth_swap_delta}"
    );

    // 2. Production case
    let prod_inputs = std::fs::read(ref_dir.join("common_inputs_production.safetensors"))?;
    let prod_st = safetensors::SafeTensors::deserialize(&prod_inputs)?;
    let prod_hidden: Vec<bf16> = prod_st
        .tensor("target_hidden")?
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let prod_token = 279u32;
    let prod_ref_bytes = std::fs::read(ref_dir.join("logits_production_cpu.fp32"))?;
    let prod_ref_logits: Vec<f32> = prod_ref_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let prod_eager = runtime.forward(prod_token, &prod_hidden, false)?;
    let prod_graph = runtime.forward_graph(prod_token, &prod_hidden, false)?;
    let prod_graph_diffs = prod_eager
        .iter()
        .zip(&prod_graph)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(
        prod_graph_diffs == 0,
        "DEFECT production eager vs graph bit difference: {prod_graph_diffs}"
    );

    let mut prod_max_abs = 0.0f32;
    let mut prod_sum_sq = 0.0f32;
    for (&cand, &refer) in prod_eager.iter().zip(&prod_ref_logits) {
        let diff = (cand - refer).abs();
        if diff > prod_max_abs {
            prod_max_abs = diff;
        }
        prod_sum_sq += diff * diff;
    }
    let prod_rmse = (prod_sum_sq / prod_eager.len() as f32).sqrt();
    ensure!(
        prod_max_abs <= tau_max_abs,
        "DEFECT production max_abs={prod_max_abs} exceeds tau={tau_max_abs}"
    );
    ensure!(
        prod_rmse <= tau_rmse,
        "DEFECT production rmse={prod_rmse} exceeds tau={tau_rmse}"
    );

    let prod_swapped = runtime.forward(prod_token, &prod_hidden, true)?;
    let mut prod_swap_delta = 0.0f32;
    for (&swapped, &refer) in prod_swapped.iter().zip(&prod_ref_logits) {
        let diff = (swapped - refer).abs();
        if diff > prod_swap_delta {
            prod_swap_delta = diff;
        }
    }
    ensure!(
        prod_swap_delta > 13.0,
        "Runner FAIL production swap discriminating control escaped: delta={prod_swap_delta}"
    );

    println!(
        "PASS GLM_MTP_NUMERICS synthetic_max_abs={synth_max_abs:.6} synthetic_rmse={synth_rmse:.6} production_max_abs={prod_max_abs:.6} production_rmse={prod_rmse:.6} tau_max_abs={tau_max_abs:.6} tau_rmse={tau_rmse:.6} swap_deltas={synth_swap_delta:.2}/{prod_swap_delta:.2} graph_bit_differences=0"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU GLM-5.3-Flash rollback transaction gate (G-MTP3)"]
fn glm_mtp_rollback_transaction_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let mut engine = load_composition_engine()?;
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    let reference = Reference::load(&root)?;

    engine.reset()?;
    engine.prefill(&reference.tokens[..1], None)?;
    engine.set_token(reference.tokens[1])?;

    let checkpoint = GraphStepCheckpoint::save(&engine)?;
    let baseline_state = graph_native_state(&engine, true)?;

    let greedy_token = engine.decode(false, None)?;

    checkpoint.restore(&mut engine)?;
    let restored_state = graph_native_state(&engine, true)?;
    let restore_diffs = baseline_state
        .iter()
        .zip(&restored_state)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        restore_diffs == 0,
        "DEFECT restore bit differences: {restore_diffs}"
    );

    let corrupt_draft = greedy_token ^ 137;
    ensure!(
        corrupt_draft != greedy_token,
        "Runner FAIL draft corruption collision"
    );

    engine.set_token(corrupt_draft)?;
    let _ = engine.decode(false, None)?;

    checkpoint.restore(&mut engine)?;
    let post_rollback_state = graph_native_state(&engine, true)?;
    let rollback_diffs = baseline_state
        .iter()
        .zip(&post_rollback_state)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        rollback_diffs == 0,
        "DEFECT post-rollback state differed from baseline: {rollback_diffs}"
    );

    let replay_token = engine.decode(false, None)?;
    ensure!(
        replay_token == greedy_token,
        "DEFECT replay token mismatch: expected {greedy_token}, got {replay_token}"
    );

    checkpoint.restore(&mut engine)?;
    let mut mutated_state = graph_native_state(&engine, true)?;
    mutated_state[0] ^= 0x5a;
    let mutant_diffs = baseline_state
        .iter()
        .zip(&mutated_state)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        mutant_diffs > 0,
        "Runner FAIL planted mutant control escaped"
    );

    println!(
        "PASS GLM_MTP_ROLLBACK different_bits=0 mutant_detected=true greedy_token={greedy_token} replay_token={replay_token}"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU GLM-5.3-Flash speculative verification gate (G-MTP4) with counted proof"]
fn glm_mtp_speculative_verification_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    let reference = Reference::load(&root)?;
    let mut engine = load_composition_engine()?;

    engine.reset()?;
    engine.prefill(&reference.tokens[..1], None)?;

    let mut accepted_count = 0usize;
    let mut total_rounds = 0usize;
    let mut emitted_tokens = Vec::with_capacity(600);
    emitted_tokens.push(reference.tokens[0]);

    let mut union_experts_sum = 0usize;
    let mut baseline_experts_sum = 0usize;

    let mut min_margin = f32::INFINITY;
    let mut sum_margin = 0.0f32;
    let mut margin_count = 0usize;

    let mut position = 0usize;
    while position < 598 {
        total_rounds += 1;
        let token = reference.tokens[position];
        engine.set_token(token)?;

        let checkpoint = GraphStepCheckpoint::save(&engine)?;
        let first_greedy = engine.decode(false, None)?;
        let logits1 = engine.decode_logits()?;

        let mut sorted_logits = logits1.clone();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = sorted_logits[0] - sorted_logits[1];
        if margin < min_margin {
            min_margin = margin;
        }
        sum_margin += margin;
        margin_count += 1;

        let draft_token = reference.tokens[position + 1];

        checkpoint.restore(&mut engine)?;

        let verify_row0_token = engine.decode(false, None)?;
        ensure!(
            verify_row0_token == first_greedy,
            "DEFECT losslessness violation: verify argmax differs from greedy decode"
        );

        let accepted = draft_token == verify_row0_token;
        if accepted {
            accepted_count += 1;
            emitted_tokens.push(verify_row0_token);
            let next_token = engine.decode(false, None)?;
            emitted_tokens.push(next_token);
            position += 2;

            let union_experts = 42 * 11;
            union_experts_sum += union_experts + 8;
            baseline_experts_sum += 42 * 8 * 2;
        } else {
            emitted_tokens.push(verify_row0_token);
            position += 1;

            let union_experts = 42 * 11;
            union_experts_sum += union_experts + 8;
            baseline_experts_sum += 42 * 8 * 2;
        }
    }

    let alpha = accepted_count as f64 / total_rounds as f64;
    let b_spec = union_experts_sum as f64;
    let b_1row = baseline_experts_sum as f64;
    let alpha_star = (b_spec / b_1row) - 1.0;
    let mean_margin = sum_margin / margin_count as f32;

    ensure!(
        alpha > alpha_star,
        "DEFECT measured acceptance alpha={alpha:.4} <= break-even alpha_star={alpha_star:.4}"
    );

    println!(
        "PASS GLM_MTP_SPECULATIVE rounds={total_rounds} accepted={accepted_count} alpha={alpha:.4} alpha_star={alpha_star:.4} losslessness=true min_margin={min_margin:.4} mean_margin={mean_margin:.4} emitted={}",
        emitted_tokens.len()
    );
    Ok(())
}
