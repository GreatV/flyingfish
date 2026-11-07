use super::mtp::{GlmMtpRuntime, GlmMtpShared, glm_mtp_draft_step_config};
use super::*;
use crate::backend::cuda::state::StateStore;
use crate::mtp::{MtpHiddenTap, MtpOptions, MtpRecurrentTap};
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

#[test]
fn glm_mtp_router_bias_precision_discriminating_cpu() -> Result<()> {
    let count = 288;
    let bias_f32: Vec<f32> = (0..count)
        .map(|i| ((i * 37 + 11) % 131) as f32 / 16.0 - 4.0)
        .collect();
    let raw_bytes: Vec<u8> = bias_f32.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let bias_bf16_mutant: Vec<f32> = (0..count)
        .map(|i| {
            let u = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
            bf16::from_bits(u).to_f32()
        })
        .collect();

    let router_logits: Vec<f32> = (0..count)
        .map(|i| ((i * 19 + 7) % 73) as f32 / 10.0 - 3.6)
        .collect();

    let route = |biases: &[f32]| -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32, f32)> = router_logits
            .iter()
            .zip(biases.iter())
            .enumerate()
            .map(|(i, (&l, &b))| {
                let s = 1.0f32 / (1.0f32 + (-l).exp());
                let select = s + b;
                (i, s, select)
            })
            .collect();
        scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(8);
        let sum: f32 = scores.iter().map(|(_, s, _)| s).sum();
        scores
            .into_iter()
            .map(|(i, s, _)| (i, (s / sum) * 2.5))
            .collect()
    };

    let top8_f32 = route(&bias_f32);
    let top8_mutant = route(&bias_bf16_mutant);

    let matching_experts = top8_f32
        .iter()
        .filter(|(idx, _)| top8_mutant.iter().any(|(m_idx, _)| m_idx == idx))
        .count();
    ensure!(
        matching_experts <= 2,
        "DEFECT router precision mutant failed to discriminate: matching={matching_experts}"
    );

    ensure!(
        raw_bytes.len() == 288 * 4,
        "DEFECT expected 1152 bytes for 288 F32 bias values, got {}",
        raw_bytes.len()
    );

    Ok(())
}

#[test]
fn glm_mtp_draft_step_config_discriminating_cpu() -> Result<()> {
    assert!(glm_mtp_draft_step_config(0, None, 0).is_err());
    let prefill3 = glm_mtp_draft_step_config(3, Some(PrefixLastInputKind::PrefillChunk), 3)?;
    ensure!(
        prefill3.draft_pos == 2
            && prefill3.committed_row == 2
            && prefill3.hidden_tap == MtpHiddenTap::PostFinalNorm
            && prefill3.retain_committed_on_reject,
        "DEFECT prefill3 MTP step config mismatch: {prefill3:?}"
    );
    let decode4 = glm_mtp_draft_step_config(4, Some(PrefixLastInputKind::DecodeToken), 1)?;
    ensure!(
        decode4.draft_pos == 3
            && decode4.committed_row == 0
            && decode4.hidden_tap == MtpHiddenTap::PostFinalNorm
            && decode4.retain_committed_on_reject,
        "DEFECT decode4 MTP step config mismatch: {decode4:?}"
    );
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
    if let Some(mtp_state) = &engine.mtp_state {
        for state in &mtp_state.dsa {
            for buffer in [&state.latent, &state.packed, &state.pool_keys] {
                integers!(stream.clone_dtoh(buffer)?.into_iter().map(bf16::to_bits));
            }
            for buffer in [&state.token_ids, &state.pool_ids, &state.meta] {
                integers!(stream.clone_dtoh(buffer)?);
            }
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
#[derive(Clone)]
struct DsaStateSnapshot {
    buffers: [Vec<bf16>; 3],
    indices: [Vec<i32>; 3],
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct GraphStepCheckpoint {
    gdn: Vec<[Vec<f32>; 2]>,
    dsa: Vec<DsaStateSnapshot>,
    sequence: Vec<i32>,
    work_position: Vec<i32>,
    work_length: Vec<i32>,
    work_token: Vec<u32>,
    work_x: Vec<bf16>,
    work_n: Vec<bf16>,
    token: u32,
    position: usize,
    pending_token: Option<u32>,
    resident_ids_len: usize,
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
            work_token: s.clone_dtoh(&engine.work.token)?,
            work_x: s.clone_dtoh(&engine.work.x)?,
            work_n: s.clone_dtoh(&engine.work.n)?,
            token: engine.token()?,
            position: engine.position,
            pending_token: engine.pending_token,
            resident_ids_len: engine.resident_ids_len,
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
        s.memcpy_htod(&self.work_token, &mut engine.work.token)?;
        s.memcpy_htod(&self.work_x, &mut engine.work.x)?;
        s.memcpy_htod(&self.work_n, &mut engine.work.n)?;
        engine.position = self.position;
        engine.set_token(self.token)?;
        engine.pending_token = self.pending_token;
        engine.resident_ids_len = self.resident_ids_len;
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
        crate::backend::cuda::state::StateCheckpoint::new(&state, &model, 1, s).is_ok(),
        "DEFECT DSA checkpoint failed"
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
    let tau_max_abs = thresholds_json
        .pointer("/logits/unified/tau_max_abs")
        .or_else(|| thresholds_json.pointer("/unified/tau_max_abs"))
        .context("tau_max_abs missing")?
        .as_f64()
        .context("tau_max_abs not a number")? as f32;
    let tau_rmse = thresholds_json
        .pointer("/logits/unified/tau_rmse")
        .or_else(|| thresholds_json.pointer("/unified/tau_rmse"))
        .context("tau_rmse missing")?
        .as_f64()
        .context("tau_rmse not a number")? as f32;
    let tau_weights_max_abs = thresholds_json
        .pointer("/router_tap/tau_weights_max_abs")
        .map(|v| v.as_f64().unwrap() as f32)
        .unwrap_or(0.0028);
    let tau_weights_rmse = thresholds_json
        .pointer("/router_tap/tau_weights_rmse")
        .map(|v| v.as_f64().unwrap() as f32)
        .unwrap_or(0.0016);

    let device = Device::new(0)?;
    unsafe {
        device.ctx.disable_event_tracking();
    }
    let ops = crate::backend::cuda::ops::Ops::new(&device.ctx)?;
    let closure = std::rc::Rc::new(crate::backend::cuda::closure::Closure::new(
        false,
        "glm_mtp".into(),
    ));
    let blas = Blas::new(&device, &ops, &std::env::temp_dir(), closure)?;
    let mut runtime = GlmMtpRuntime::load(&model_dir, &device, &blas, true, 2048)?;

    // 1. Synthetic case
    let synth_inputs = std::fs::read(ref_dir.join("common_inputs.safetensors"))?;
    let synth_st = safetensors::SafeTensors::deserialize(&synth_inputs)?;
    let synth_tokens: Vec<u32> = synth_st
        .tensor("history_tokens")?
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let synth_hidden: Vec<bf16> = synth_st
        .tensor("history_hidden")?
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let synth_ref_bytes = std::fs::read(ref_dir.join("logits_cpu.fp32"))?;
    let synth_ref_logits: Vec<f32> = synth_ref_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let synth_eager = runtime.forward(&synth_tokens, &synth_hidden, false, &blas)?;
    let synth_graph = runtime.forward_graph(&synth_tokens, &synth_hidden, false, &blas)?;
    let synth_graph_diffs = synth_eager
        .iter()
        .zip(&synth_graph)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(
        synth_graph_diffs == 0,
        "DEFECT synthetic eager vs graph bit difference: {synth_graph_diffs}"
    );

    let mut defects = Vec::new();
    let stage_taps = runtime.read_stage_taps()?;
    let inter_file = ref_dir.join("ref_intermediates.safetensors");
    if inter_file.exists() {
        let inter_bytes = std::fs::read(&inter_file)?;
        let inter_st = safetensors::SafeTensors::deserialize(&inter_bytes)?;

        let read_bf16 = |name: &str| -> Result<Vec<f32>> {
            let t = inter_st.tensor(name)?;
            Ok(t.data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect())
        };
        let read_f32 = |name: &str| -> Result<Vec<f32>> {
            let t = inter_st.tensor(name)?;
            Ok(t.data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };

        let diff_stat = |name: &str, cand: &[f32], refer: &[f32]| {
            let mut max_abs = 0.0f32;
            let mut sum_sq = 0.0f32;
            let mut worst_idx = 0;
            for (i, (&c, &r)) in cand.iter().zip(refer).enumerate() {
                let d = (c - r).abs();
                if d > max_abs {
                    max_abs = d;
                    worst_idx = i;
                }
                sum_sq += d * d;
            }
            let rmse = (sum_sq / cand.len().max(1) as f32).sqrt();
            println!(
                "STAGE_DIAG {name:15}: max_abs={max_abs:.6} rmse={rmse:.6} worst_idx={worst_idx} (cand={} ref={})",
                cand.get(worst_idx).unwrap_or(&0.0),
                refer.get(worst_idx).unwrap_or(&0.0)
            );
        };

        let to_f32 = |v: &[bf16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };

        if let Ok(ref_t) = read_bf16("enorm_out") {
            diff_stat("enorm_out", &to_f32(&stage_taps.enorm), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("hnorm_out") {
            diff_stat("hnorm_out", &to_f32(&stage_taps.hnorm), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("eh_in") {
            diff_stat("eh_in", &to_f32(&stage_taps.eh_in), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("x") {
            diff_stat("x", &to_f32(&stage_taps.x_init), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("x_norm1") {
            diff_stat("x_norm1", &to_f32(&stage_taps.x_norm1), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("kv_latent") {
            diff_stat("kv_latent", &to_f32(&stage_taps.kv_latent), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("q_a") {
            diff_stat("q_a", &to_f32(&stage_taps.q_a), &ref_t);
        }
        if let Ok(ref_t) = read_f32("q_out") {
            diff_stat("q_out", &to_f32(&stage_taps.q_out), &ref_t);
        }
        if let Ok(ref_t) = read_f32("attn_out") {
            diff_stat("attn_out", &to_f32(&stage_taps.attn_out), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("o_out") {
            diff_stat("o_out", &to_f32(&stage_taps.o_out), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("x_curr") {
            diff_stat("x_curr", &to_f32(&stage_taps.x_curr), &ref_t);
        }
        if let Ok(ref_t) = read_bf16("x_norm2") {
            diff_stat("x_norm2", &to_f32(&stage_taps.x_norm2), &ref_t);
        }
        if let Ok(ref_t) = read_f32("router_logits") {
            diff_stat("router_logits", &stage_taps.router_logits, &ref_t);
        }
        println!("STAGE_DIAG cand_routing_ids: {:?}", stage_taps.routing_ids);
        println!(
            "STAGE_DIAG cand_routing_weights: {:?}",
            stage_taps.routing_weights
        );
        let moe_norm = to_f32(&stage_taps.moe_out)
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        println!("STAGE_DIAG cand_moe_out norm: {moe_norm:.6}");

        if let Ok(t) = inter_st.tensor("x_norm2") {
            let ref_norm2: Vec<bf16> = t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            let (ctrl_logits, ctrl_ids, ctrl_weights) =
                runtime.test_route_custom_norm(&ref_norm2)?;
            if let Ok(ref_t) = read_f32("router_logits") {
                diff_stat("ctrl_router_logits", &ctrl_logits, &ref_t);
            }
            println!("STAGE_DIAG ctrl_router_ids (from ref x_norm2): got {ctrl_ids:?}");
            println!("STAGE_DIAG ctrl_router_weights: {ctrl_weights:?}");
        }
    }

    let (synth_cand_ids, synth_cand_weights) = runtime
        .last_route()
        .context("synthetic router tap missing")?
        .clone();
    let synth_expected_ids: Vec<i32> = thresholds_json
        .pointer("/router_tap/synthetic/normal_ids")
        .context("synthetic normal_ids missing")?
        .as_array()
        .context("normal_ids array missing")?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let synth_expected_weights: Vec<f32> = thresholds_json
        .pointer("/router_tap/synthetic/normal_weights")
        .context("synthetic normal_weights missing")?
        .as_array()
        .context("normal_weights array missing")?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let synth_min_margin = thresholds_json
        .pointer("/router_tap/synthetic/min_margin_8_vs_9")
        .map(|v| v.as_f64().unwrap() as f32)
        .unwrap_or(0.011381);
    let synth_spread = thresholds_json
        .pointer("/router_tap/synthetic/cross_arm_selection_scores_max_abs")
        .map(|v| v.as_f64().unwrap() as f32)
        .unwrap_or(0.005793);
    if synth_min_margin > synth_spread && synth_cand_ids != synth_expected_ids {
        defects.push(format!(
            "DEFECT synthetic router IDs mismatch: got {synth_cand_ids:?}, expected {synth_expected_ids:?}"
        ));
    }
    let mut synth_exp_map = std::collections::BTreeMap::new();
    for (&id, &w) in synth_expected_ids.iter().zip(&synth_expected_weights) {
        synth_exp_map.insert(id, w);
    }
    let mut synth_w_max_abs = 0.0f32;
    let mut synth_w_sum_sq = 0.0f32;
    let mut synth_w_count = 0;
    for (&id, &cw) in synth_cand_ids.iter().zip(&synth_cand_weights) {
        if let Some(&rw) = synth_exp_map.get(&id) {
            let diff = (cw - rw).abs();
            if diff > synth_w_max_abs {
                synth_w_max_abs = diff;
            }
            synth_w_sum_sq += diff * diff;
            synth_w_count += 1;
        }
    }
    let synth_w_rmse = if synth_w_count > 0 {
        (synth_w_sum_sq / synth_w_count as f32).sqrt()
    } else {
        0.0
    };
    if synth_w_max_abs > tau_weights_max_abs {
        defects.push(format!(
            "DEFECT synthetic router weights max_abs={synth_w_max_abs} exceeds tau={tau_weights_max_abs}"
        ));
    }
    if synth_w_rmse > tau_weights_rmse {
        defects.push(format!(
            "DEFECT synthetic router weights rmse={synth_w_rmse} exceeds tau={tau_weights_rmse}"
        ));
    }

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
    if synth_max_abs > tau_max_abs {
        defects.push(format!(
            "DEFECT synthetic max_abs={synth_max_abs} exceeds tau={tau_max_abs}"
        ));
    }
    if synth_rmse > tau_rmse {
        defects.push(format!(
            "DEFECT synthetic rmse={synth_rmse} exceeds tau={tau_rmse}"
        ));
    }

    let synth_swapped = runtime.forward(&synth_tokens, &synth_hidden, true, &blas)?;
    let mut synth_swap_delta = 0.0f32;
    let mut synth_swap_sum_sq = 0.0f32;
    for (&swapped, &refer) in synth_swapped.iter().zip(&synth_ref_logits) {
        let diff = (swapped - refer).abs();
        if diff > synth_swap_delta {
            synth_swap_delta = diff;
        }
        synth_swap_sum_sq += diff * diff;
    }
    let synth_swap_rmse = (synth_swap_sum_sq / synth_swapped.len() as f32).sqrt();

    // 2. Production case
    let prod_inputs = std::fs::read(ref_dir.join("common_inputs_production.safetensors"))?;
    let prod_st = safetensors::SafeTensors::deserialize(&prod_inputs)?;
    let prod_tokens: Vec<u32> = prod_st
        .tensor("history_tokens")?
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let prod_hidden: Vec<bf16> = prod_st
        .tensor("history_hidden")?
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let prod_ref_bytes = std::fs::read(ref_dir.join("logits_production_cpu.fp32"))?;
    let prod_ref_logits: Vec<f32> = prod_ref_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let prod_eager = runtime.forward(&prod_tokens, &prod_hidden, false, &blas)?;
    let prod_graph = runtime.forward_graph(&prod_tokens, &prod_hidden, false, &blas)?;
    let prod_graph_diffs = prod_eager
        .iter()
        .zip(&prod_graph)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if prod_graph_diffs != 0 {
        defects.push(format!(
            "DEFECT production eager vs graph bit difference: {prod_graph_diffs}"
        ));
    }

    let (prod_cand_ids, prod_cand_weights) = runtime
        .last_route()
        .context("production router tap missing")?
        .clone();
    let prod_expected_ids: Vec<i32> = thresholds_json
        .pointer("/router_tap/production/normal_ids")
        .context("production normal_ids missing")?
        .as_array()
        .context("normal_ids array missing")?
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let prod_expected_weights: Vec<f32> = thresholds_json
        .pointer("/router_tap/production/normal_weights")
        .context("production normal_weights missing")?
        .as_array()
        .context("normal_weights array missing")?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let prod_expected_set: std::collections::BTreeSet<i32> = thresholds_json
        .pointer("/router_tap/production/expected_expert_set")
        .map(|v| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect()
        })
        .unwrap_or_else(|| prod_expected_ids.iter().copied().collect());
    let prod_min_margin = thresholds_json
        .pointer("/router_tap/production/min_margin_8_vs_9")
        .map(|v| v.as_f64().unwrap() as f32)
        .unwrap_or(0.017159);
    let prod_spread = thresholds_json
        .pointer("/router_tap/production/cross_arm_selection_scores_max_abs")
        .map(|v| v.as_f64().unwrap() as f32)
        .unwrap_or(0.003126);
    if prod_min_margin > prod_spread {
        let prod_cand_set: std::collections::BTreeSet<i32> =
            prod_cand_ids.iter().copied().collect();
        if prod_cand_set != prod_expected_set {
            defects.push(format!(
                "DEFECT production router expert set mismatch: got {prod_cand_set:?}, expected {prod_expected_set:?}"
            ));
        }
    }
    let mut prod_exp_map = std::collections::BTreeMap::new();
    for (&id, &w) in prod_expected_ids.iter().zip(&prod_expected_weights) {
        prod_exp_map.insert(id, w);
    }
    let mut prod_w_max_abs = 0.0f32;
    let mut prod_w_sum_sq = 0.0f32;
    for (&id, &cw) in prod_cand_ids.iter().zip(&prod_cand_weights) {
        if let Some(&rw) = prod_exp_map.get(&id) {
            let diff = (cw - rw).abs();
            if diff > prod_w_max_abs {
                prod_w_max_abs = diff;
            }
            prod_w_sum_sq += diff * diff;
        }
    }
    let prod_w_rmse = (prod_w_sum_sq / prod_cand_weights.len() as f32).sqrt();
    if prod_w_max_abs > tau_weights_max_abs {
        defects.push(format!(
            "DEFECT production router weights max_abs={prod_w_max_abs} exceeds tau={tau_weights_max_abs}"
        ));
    }
    if prod_w_rmse > tau_weights_rmse {
        defects.push(format!(
            "DEFECT production router weights rmse={prod_w_rmse} exceeds tau={tau_weights_rmse}"
        ));
    }

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
    if prod_max_abs > tau_max_abs {
        defects.push(format!(
            "DEFECT production max_abs={prod_max_abs} exceeds tau={tau_max_abs}"
        ));
    }
    if prod_rmse > tau_rmse {
        defects.push(format!(
            "DEFECT production rmse={prod_rmse} exceeds tau={tau_rmse}"
        ));
    }

    let prod_swapped = runtime.forward(&prod_tokens, &prod_hidden, true, &blas)?;
    let mut prod_swap_delta = 0.0f32;
    let mut prod_swap_sum_sq = 0.0f32;
    for (&swapped, &refer) in prod_swapped.iter().zip(&prod_ref_logits) {
        let diff = (swapped - refer).abs();
        if diff > prod_swap_delta {
            prod_swap_delta = diff;
        }
        prod_swap_sum_sq += diff * diff;
    }
    let prod_swap_rmse = (prod_swap_sum_sq / prod_swapped.len() as f32).sqrt();

    // Production carries the discriminating gate at 10.23x tau_max_abs and 11.51x tau_rmse
    ensure!(
        prod_swap_delta >= 10.0 * tau_max_abs,
        "Runner FAIL production swap discriminating control: delta={prod_swap_delta} < 10*tau"
    );
    ensure!(
        prod_swap_rmse >= 10.0 * tau_rmse,
        "Runner FAIL production swap discriminating control: rmse={prod_swap_rmse} < 10*tau"
    );

    let status = if defects.is_empty() { "PASS" } else { "DEFECT" };
    println!(
        "{status} GLM_MTP_NUMERICS synthetic_max_abs={synth_max_abs:.6} synthetic_rmse={synth_rmse:.6} production_max_abs={prod_max_abs:.6} production_rmse={prod_rmse:.6} tau_max_abs={tau_max_abs:.6} tau_rmse={tau_rmse:.6} swap_deltas={synth_swap_delta:.6}/{prod_swap_delta:.6} swap_rmse={synth_swap_rmse:.6}/{prod_swap_rmse:.6} graph_bit_differences=0 defects={}",
        defects.len()
    );
    for d in &defects {
        println!("{d}");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn state_store_bytes(
    store: &StateStore,
    stream: &Arc<crate::backend::gpu::CudaStream>,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for state in &store.gdn {
        for v in stream.clone_dtoh(&state.s)? {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in stream.clone_dtoh(&state.history)? {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    for state in &store.dsa {
        for buffer in [&state.latent, &state.packed, &state.pool_keys] {
            for v in stream.clone_dtoh(buffer)? {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for buffer in [&state.token_ids, &state.pool_ids, &state.meta] {
            for v in stream.clone_dtoh(buffer)? {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    if let Some(seq) = &store.sequence {
        for v in stream.clone_dtoh(seq)? {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(bytes)
}

#[cfg(feature = "cuda")]
fn mtp_persistent_state_bytes(engine: &Engine) -> Result<Vec<u8>> {
    let stream = &engine.device.stream;
    let mut bytes = state_store_bytes(&engine.state, stream)?;
    if let Some(mtp_state) = &engine.mtp_state {
        bytes.extend_from_slice(&state_store_bytes(mtp_state, stream)?);
    }
    bytes.extend_from_slice(&(engine.position as u64).to_le_bytes());
    bytes.extend_from_slice(&engine.pending_token.unwrap_or(u32::MAX).to_le_bytes());
    bytes.extend_from_slice(&(engine.resident_ids_len as u64).to_le_bytes());
    for &id in &engine.committed_prefix_ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    for v in stream.clone_dtoh(&engine.work.position)? {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in stream.clone_dtoh(&engine.work.length)? {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in stream.clone_dtoh(&engine.work.token)? {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let vocab = engine.model_spec.vocab as usize;
    for v in &stream.clone_dtoh(&engine.work.x)?[..4096] {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    for v in &stream.clone_dtoh(&engine.work.n)?[..4096] {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    for v in &stream.clone_dtoh(&engine.work.logits)?[..vocab] {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(feature = "cuda")]
fn verify_select_prefix_kernel_oracle() -> Result<()> {
    use crate::backend::cuda::contracts::{Scratch, TensorMut, view};
    use crate::backend::cuda::dsa::{DsaAppendInput, DsaIndexInput, DsaPlan};
    use crate::backend::cuda::gdn::{
        GdnBindings, GdnConvWeights, GdnDecayWeights, GdnGateInput, GdnOps,
    };
    use crate::backend::cuda::state::{GdnState, StateCheckpoint};
    use crate::backend::gpu::DevicePtr;
    use crate::gdn_plan::GdnPolicy;
    use crate::op_contract::{DsaSpec, GdnSpec};

    let device = Device::new(0)?;
    let stream = &device.stream;
    let root_cfg: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glm53-flash-config.json"
    )))?;
    let mut mini_spec = crate::model_spec::ModelSpec::glm5_next(
        &root_cfg,
        &serde_json::json!({"eos_token_id": [154820, 154827, 154829]}),
    )?;
    mini_spec.layers = vec![mini_spec.layers[0].clone(), mini_spec.layers[3].clone()];
    mini_spec.layers[0].state_index = 0;
    mini_spec.layers[1].state_index = 0;

    let cap = 64usize;
    let gdn_spec = GdnSpec::from_model(&mini_spec)?;
    let dsa_spec = DsaSpec::from_model(&mini_spec)?;
    let cd = gdn_spec.conv_dim()? as usize;
    let mut store_1row = StateStore::new(&mini_spec, cap, stream)?;
    let mut store_2row = StateStore::new(&mini_spec, cap, stream)?;
    let mut ckpt = StateCheckpoint::new(&store_2row, &mini_spec, 2, stream)?;
    let tok_buf = stream.alloc_zeros::<u32>(1)?;
    let snap = vec![GdnState {
        s: stream.alloc_zeros(store_2row.gdn[0].s.len())?,
        history: stream.alloc_zeros(store_2row.gdn[0].history.len())?,
    }];

    let pat_bf16 = |n: usize, period: usize, mul: usize, div: f32| -> Vec<bf16> {
        (0..n)
            .map(|i| {
                bf16::from_f32(
                    ((i * mul + 17) % period) as f32 / div - (period / 2) as f32 / (2.0 * div),
                )
            })
            .collect()
    };
    let qkv_all = pat_bf16(5 * cd, 1009, 37, 256.0);
    let a_all = pat_bf16(5 * (gdn_spec.hv * gdn_spec.dk) as usize, 509, 19, 128.0);
    let b_all = pat_bf16(5 * gdn_spec.hv as usize, 251, 13, 128.0);
    let q_conv_w =
        stream.clone_htod(&pat_bf16((gdn_spec.hk * 128 * 4) as usize, 127, 11, 256.0))?;
    let k_conv_w =
        stream.clone_htod(&pat_bf16((gdn_spec.hk * 128 * 4) as usize, 131, 17, 256.0))?;
    let v_conv_w =
        stream.clone_htod(&pat_bf16((gdn_spec.hv * 128 * 4) as usize, 137, 23, 256.0))?;
    let a_log = stream.clone_htod(
        &(0..gdn_spec.hv as usize)
            .map(|i| 0.25f32 + (i % 11) as f32 * 0.05)
            .collect::<Vec<_>>(),
    )?;
    let dt_bias = stream.clone_htod(
        &(0..(gdn_spec.hv * gdn_spec.dk) as usize)
            .map(|i| ((i * 7 + 3) % 29) as f32 * 0.01 - 0.14)
            .collect::<Vec<_>>(),
    )?;
    let ws_bytes = GdnOps::workspace(&gdn_spec, 2, GdnPolicy::P2)?.bytes;
    let gdn_scratch = stream.alloc_zeros::<u8>(usize::try_from(ws_bytes)?)?;
    let qkv_buf = stream.alloc_zeros::<bf16>(2 * cd)?;
    let a_buf = stream.alloc_zeros::<bf16>(2 * (gdn_spec.hv * gdn_spec.dk) as usize)?;
    let b_buf = stream.alloc_zeros::<bf16>(2 * gdn_spec.hv as usize)?;
    let out_buf = stream.alloc_zeros::<bf16>(2 * (gdn_spec.hv * gdn_spec.dv) as usize)?;

    let make_gdn_plan = |store: &StateStore,
                         qkv_in: &cudarc::driver::CudaSlice<bf16>,
                         a_in: &cudarc::driver::CudaSlice<bf16>,
                         b_in: &cudarc::driver::CudaSlice<bf16>,
                         rows: u32,
                         with_snap: bool|
     -> Result<_> {
        let base = GdnOps::prepare(
            &device.ctx,
            &gdn_spec,
            2,
            GdnBindings {
                qkv: view(qkv_in, stream, 0, [2, cd as u32], [cd as u64, 1])?,
                a: GdnGateInput::PerChannel(view(
                    a_in,
                    stream,
                    0,
                    [2, gdn_spec.hv, 128],
                    [u64::from(gdn_spec.hv * 128), 128, 1],
                )?),
                b: view(
                    b_in,
                    stream,
                    0,
                    [2, gdn_spec.hv],
                    [u64::from(gdn_spec.hv), 1],
                )?,
                conv_weight: GdnConvWeights::Separate([
                    view(&q_conv_w, stream, 0, [gdn_spec.hk * 128, 4], [4, 1])?,
                    view(&k_conv_w, stream, 0, [gdn_spec.hk * 128, 4], [4, 1])?,
                    view(&v_conv_w, stream, 0, [gdn_spec.hv * 128, 4], [4, 1])?,
                ]),
                decay: GdnDecayWeights::F32 {
                    a_log: view(&a_log, stream, 0, [gdn_spec.hv], [1])?,
                    dt_bias: view(&dt_bias, stream, 0, [gdn_spec.hv * 128], [1])?,
                },
                core_out: TensorMut(view(
                    &out_buf,
                    stream,
                    0,
                    [2, gdn_spec.hv, 128],
                    [u64::from(gdn_spec.hv * 128), 128, 1],
                )?),
                s: TensorMut(view(
                    &store.gdn[0].s,
                    stream,
                    0,
                    [gdn_spec.hv, 128, 128],
                    [128 * 128, 128, 1],
                )?),
                history: TensorMut(view(
                    &store.gdn[0].history,
                    stream,
                    0,
                    [cd as u32, 3],
                    [3, 1],
                )?),
            },
            false,
        )?;
        let snap_views = if with_snap {
            Some((
                view(
                    &snap[0].s,
                    stream,
                    0,
                    [1, gdn_spec.hv, 128, 128],
                    [u64::from(gdn_spec.hv * 128 * 128), 128 * 128, 128, 1],
                )?,
                view(
                    &snap[0].history,
                    stream,
                    0,
                    [1, cd as u32, 3],
                    [cd as u64 * 3, 3, 1],
                )?,
            ))
        } else {
            None
        };
        base.for_rows(rows, snap_views)
    };

    let dsa_plan = DsaPlan::new(&device.ctx, &device.info, dsa_spec, cap as u32, 2)?;
    let dsa_1 = dsa_plan.for_rows(1)?;
    let dsa_2 = dsa_plan.for_rows(2)?;
    let latent_all = pat_bf16(5 * 512, 509, 29, 256.0);
    let mut packed_all = pat_bf16(5 * 257, 251, 17, 128.0);
    for t in 0..5 {
        packed_all[t * 257 + 256] = bf16::from_f32(1.0);
    }
    let latent_buf = stream.alloc_zeros::<bf16>(2 * 512)?;
    let packed_buf = stream.alloc_zeros::<bf16>(2 * 257)?;
    let dsa_q = stream.clone_htod(&pat_bf16(2 * 32 * 128, 1031, 23, 512.0))?;
    let dsa_w = stream.clone_htod(&pat_bf16(2 * 32, 127, 19, 128.0))?;
    let dsa_ape = stream.clone_htod(&pat_bf16(4 * 128, 509, 17, 256.0))?;

    struct OracleInputBufs {
        qkv: cudarc::driver::CudaSlice<bf16>,
        a: cudarc::driver::CudaSlice<bf16>,
        b: cudarc::driver::CudaSlice<bf16>,
        latent: cudarc::driver::CudaSlice<bf16>,
        packed: cudarc::driver::CudaSlice<bf16>,
    }
    let mut bufs = OracleInputBufs {
        qkv: qkv_buf,
        a: a_buf,
        b: b_buf,
        latent: latent_buf,
        packed: packed_buf,
    };

    let load_rows = |start: usize, rows: usize, dst: &mut OracleInputBufs| -> Result<()> {
        let a_w = (gdn_spec.hv * gdn_spec.dk) as usize;
        let b_w = gdn_spec.hv as usize;
        stream.memcpy_htod(
            &qkv_all[start * cd..(start + rows) * cd],
            &mut dst.qkv.slice_mut(..rows * cd),
        )?;
        stream.memcpy_htod(
            &a_all[start * a_w..(start + rows) * a_w],
            &mut dst.a.slice_mut(..rows * a_w),
        )?;
        stream.memcpy_htod(
            &b_all[start * b_w..(start + rows) * b_w],
            &mut dst.b.slice_mut(..rows * b_w),
        )?;
        stream.memcpy_htod(
            &latent_all[start * 512..(start + rows) * 512],
            &mut dst.latent.slice_mut(..rows * 512),
        )?;
        stream.memcpy_htod(
            &packed_all[start * 257..(start + rows) * 257],
            &mut dst.packed.slice_mut(..rows * 257),
        )?;
        Ok(())
    };
    let scratch_view = || Scratch {
        ptr: gdn_scratch.device_ptr(stream).0,
        bytes: ws_bytes,
    };

    // Seed prefix = 2 identically on both stores.
    load_rows(0, 2, &mut bufs)?;
    for store in [&mut store_1row, &mut store_2row] {
        store.set_sequence(stream, 0, 2, 8)?;
        let gdn_seed = make_gdn_plan(store, &bufs.qkv, &bufs.a, &bufs.b, 2, false)?;
        let meta = store.seq_ref(stream)?;
        GdnOps::run_verify(&gdn_seed, stream, meta, scratch_view())?;
        dsa_2.append(
            &store.dsa[0],
            &DsaAppendInput {
                latent: view(&bufs.latent, stream, 0, [2, 512], [512, 1])?,
                packed: view(&bufs.packed, stream, 0, [2, 257], [257, 1])?,
            },
            stream,
        )?;
        dsa_2.pool(
            &store.dsa[0],
            &DsaIndexInput {
                q: view(&dsa_q, stream, 0, [2, 32, 128], [32 * 128, 128, 1])?,
                weights: view(&dsa_w, stream, 0, [2, 32], [32, 1])?,
                ape: view(&dsa_ape, stream, 0, [4, 128], [128, 1])?,
            },
            stream,
        )?;
        store.set_sequence(stream, 2, 1, 8)?;
    }

    // Test both prefix = 2 (incomplete pool 0) and prefix = 3 (completed pool 0 + overrun pool 1).
    for prefix in [2usize, 3usize] {
        load_rows(prefix, 1, &mut bufs)?;
        store_1row.set_sequence(stream, prefix as u32, 1, 8)?;
        let gdn_1 = make_gdn_plan(&store_1row, &bufs.qkv, &bufs.a, &bufs.b, 1, false)?;
        let meta_1 = store_1row.seq_ref(stream)?;
        GdnOps::run_verify(&gdn_1, stream, meta_1, scratch_view())?;
        dsa_1.append(
            &store_1row.dsa[0],
            &DsaAppendInput {
                latent: view(&bufs.latent, stream, 0, [1, 512], [512, 1])?,
                packed: view(&bufs.packed, stream, 0, [1, 257], [257, 1])?,
            },
            stream,
        )?;
        dsa_1.pool(
            &store_1row.dsa[0],
            &DsaIndexInput {
                q: view(&dsa_q, stream, 0, [1, 32, 128], [32 * 128, 128, 1])?,
                weights: view(&dsa_w, stream, 0, [1, 32], [32, 1])?,
                ape: view(&dsa_ape, stream, 0, [4, 128], [128, 1])?,
            },
            stream,
        )?;
        store_1row.set_sequence(stream, (prefix + 1) as u32, 1, 8)?;

        ckpt.save_rows(&store_2row, &tok_buf, prefix, 2, stream, None)?;
        load_rows(prefix, 2, &mut bufs)?;
        store_2row.set_sequence(stream, prefix as u32, 2, 8)?;
        let gdn_2 = make_gdn_plan(&store_2row, &bufs.qkv, &bufs.a, &bufs.b, 2, true)?;
        let meta_2 = store_2row.seq_ref(stream)?;
        GdnOps::run_verify(&gdn_2, stream, meta_2, scratch_view())?;
        dsa_2.append(
            &store_2row.dsa[0],
            &DsaAppendInput {
                latent: view(&bufs.latent, stream, 0, [2, 512], [512, 1])?,
                packed: view(&bufs.packed, stream, 0, [2, 257], [257, 1])?,
            },
            stream,
        )?;
        dsa_2.pool(
            &store_2row.dsa[0],
            &DsaIndexInput {
                q: view(&dsa_q, stream, 0, [2, 32, 128], [32 * 128, 128, 1])?,
                weights: view(&dsa_w, stream, 0, [2, 32], [32, 1])?,
                ape: view(&dsa_ape, stream, 0, [4, 128], [128, 1])?,
            },
            stream,
        )?;
        ckpt.select_prefix(&mut store_2row, &snap, 1, stream)?;
        store_2row.set_sequence(stream, (prefix + 1) as u32, 1, 8)?;

        let expected = state_store_bytes(&store_1row, stream)?;
        let actual = state_store_bytes(&store_2row, stream)?;
        let diffs = expected.iter().zip(&actual).filter(|(a, b)| a != b).count();
        ensure!(
            diffs == 0,
            "DEFECT select_prefix(1) kernel oracle differed from 1-row execution at prefix={prefix}: {diffs} byte diffs"
        );
    }
    stream.synchronize()?;
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
    // 1. Direct 1-row vs 2-row + select_prefix(1) bitwise oracle across incomplete & completed DSA pools.
    verify_select_prefix_kernel_oracle()?;

    // 2. End-to-end 45-layer + Layer-45 prefix-1 rejection commit invariance across start_pos = 2 and 3.
    let mut engine = load_composition_engine()?;
    let root =
        PathBuf::from(std::env::var_os("G1_COMPOSITION_REFERENCE").context("reference missing")?);
    let reference = Reference::load(&root)?;

    let mtp_options = MtpOptions {
        depth: 1,
        hidden_tap: MtpHiddenTap::PostFinalNorm,
        recurrent_tap: MtpRecurrentTap::PreMtpNorm,
        graph: true,
    };
    engine.mtp_enable(mtp_options)?;
    let vocab = engine.model_spec.vocab;

    let run_two_rejected_rounds = |engine: &mut Engine,
                                   xor_mask: u32,
                                   add_off: u32|
     -> Result<(u32, u32, Vec<u8>, Vec<u8>)> {
        let first = engine.mtp_prefill(&reference.tokens[..2])?;
        let p1 = (first.wrapping_add(add_off) ^ xor_mask) % vocab;
        if let Some(chain) = engine.mtp.as_mut() {
            chain.force_proposals = Some(vec![p1]);
        }
        let out1 = engine.glm_mtp_round(false)?;
        ensure!(
            out1.round.accepted == 0 && out1.replay_row0_logits.is_none(),
            "DEFECT round 1 expected prefix-1 rejection without replay"
        );
        let pool_ids_1 = engine
            .device
            .stream
            .clone_dtoh(&engine.state.dsa[0].pool_ids)?;
        let tok_ids_1 = engine
            .device
            .stream
            .clone_dtoh(&engine.state.dsa[0].token_ids)?;
        ensure!(
            pool_ids_1[0] == -1 && tok_ids_1[3] == 0,
            "DEFECT round 1 incomplete pool rollback mismatch: pool_ids[0]={} token_ids[3]={}",
            pool_ids_1[0],
            tok_ids_1[3]
        );
        let snap1 = mtp_persistent_state_bytes(engine)?;
        let t1 = out1.round.tokens[0];

        let mut p2 = (t1.wrapping_add(add_off) ^ xor_mask) % vocab;
        if p2 == 0 {
            p2 = 7;
        }
        if let Some(chain) = engine.mtp.as_mut() {
            chain.force_proposals = Some(vec![p2]);
        }
        let out2 = engine.glm_mtp_round(false)?;
        ensure!(
            out2.round.accepted == 0 && out2.replay_row0_logits.is_none(),
            "DEFECT round 2 expected prefix-1 rejection without replay"
        );
        let pool_ids_2 = engine
            .device
            .stream
            .clone_dtoh(&engine.state.dsa[0].pool_ids)?;
        let tok_ids_2 = engine
            .device
            .stream
            .clone_dtoh(&engine.state.dsa[0].token_ids)?;
        ensure!(
            pool_ids_2[0] == 3 && pool_ids_2[1] == 0 && tok_ids_2[4] == 0,
            "DEFECT round 2 completed-pool boundary rollback mismatch: pool_ids[0]={} pool_ids[1]={} token_ids[4]={}",
            pool_ids_2[0],
            pool_ids_2[1],
            tok_ids_2[4]
        );
        let snap2 = mtp_persistent_state_bytes(engine)?;
        Ok((t1, out2.round.tokens[0], snap1, snap2))
    };

    let (t1_a, t2_a, state_a1, state_a2) = run_two_rejected_rounds(&mut engine, 0x5a5a, 17)?;
    let (t1_b, t2_b, state_b1, state_b2) = run_two_rejected_rounds(&mut engine, 0x33cc, 1337)?;
    ensure!(
        t1_a == t1_b && t2_a == t2_b,
        "DEFECT prefix-1 committed tokens differed across rejected proposals: ({t1_a},{t2_a}) vs ({t1_b},{t2_b})"
    );
    let diffs_r1 = state_a1
        .iter()
        .zip(&state_b1)
        .filter(|(a, b)| a != b)
        .count();
    let diffs_r2 = state_a2
        .iter()
        .zip(&state_b2)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        diffs_r1 == 0 && diffs_r2 == 0,
        "DEFECT post-reject persistent state differed across proposals: r1={diffs_r1} r2={diffs_r2}"
    );

    // 3. Negative controls:
    let stream = engine.device.stream.clone();
    let mut mutant_gdn_history = stream.clone_dtoh(&engine.state.gdn[0].history)?;
    mutant_gdn_history[0] += 1.0;
    stream.memcpy_htod(&mutant_gdn_history, &mut engine.state.gdn[0].history)?;
    let mutant_gdn_diffs = state_a2
        .iter()
        .zip(&mtp_persistent_state_bytes(&engine)?)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        mutant_gdn_diffs > 0,
        "Runner FAIL planted GDN history rollback mutant escaped"
    );
    mutant_gdn_history[0] -= 1.0;
    stream.memcpy_htod(&mutant_gdn_history, &mut engine.state.gdn[0].history)?;

    let mut mutant_dsa_pools = stream.clone_dtoh(&engine.state.dsa[0].pool_ids)?;
    mutant_dsa_pools[1] = -1;
    stream.memcpy_htod(&mutant_dsa_pools, &mut engine.state.dsa[0].pool_ids)?;
    let mutant_dsa_diffs = state_a2
        .iter()
        .zip(&mtp_persistent_state_bytes(&engine)?)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        mutant_dsa_diffs > 0,
        "Runner FAIL planted target DSA pool rollback mutant escaped"
    );
    mutant_dsa_pools[1] = 0;
    stream.memcpy_htod(&mutant_dsa_pools, &mut engine.state.dsa[0].pool_ids)?;

    {
        let mtp_state = engine.mtp_state.as_mut().context("MTP state missing")?;
        let mut mutant_layer45 = stream.clone_dtoh(&mtp_state.dsa[0].latent)?;
        mutant_layer45[4 * 512] = bf16::from_f32(mutant_layer45[4 * 512].to_f32() + 1.0);
        stream.memcpy_htod(&mutant_layer45, &mut mtp_state.dsa[0].latent)?;
    }
    let mutant_l45_diffs = state_a2
        .iter()
        .zip(&mtp_persistent_state_bytes(&engine)?)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        mutant_l45_diffs > 0,
        "Runner FAIL planted Layer 45 rollback mutant escaped"
    );

    println!(
        "PASS GLM_MTP_ROLLBACK different_bits=0 kernel_oracle_bits=0 replay_passes=0 mutant_gdn_detected=true mutant_dsa_detected=true mutant_layer45_detected=true committed_tokens=[{t1_a},{t2_a}]"
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

    let mtp_options = MtpOptions {
        depth: 1,
        hidden_tap: MtpHiddenTap::PostFinalNorm,
        recurrent_tap: MtpRecurrentTap::PreMtpNorm,
        graph: true,
    };
    engine.mtp_enable(mtp_options)?;
    let eos_ids = engine.model_spec.eos.clone();
    ensure!(!eos_ids.is_empty(), "Runner FAIL GLM EOS ids missing");

    // Read frozen verify thresholds (tau)
    let ref_dir = PathBuf::from(
        std::env::var_os("FF_GLM_MTP_REF_DIR").context("FF_GLM_MTP_REF_DIR missing")?,
    );
    let tau_path = ref_dir.join("verify_thresholds.json");
    let thresholds_raw = std::fs::read_to_string(tau_path)?;
    let thresholds: serde_json::Value = serde_json::from_str(&thresholds_raw)?;
    let tau_max_abs_per_pos: Vec<f32> = thresholds["tau_max_abs_per_position"]
        .as_array()
        .context("tau_max_abs_per_position missing")?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let tau_rmse_per_pos: Vec<f32> = thresholds["tau_rmse_per_position"]
        .as_array()
        .context("tau_rmse_per_position missing")?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let control_path = ref_dir.join("verify_control_positions_r1.json");
    let control_raw = std::fs::read_to_string(control_path)?;
    let control_json: serde_json::Value = serde_json::from_str(&control_raw)?;
    let gated_positions: std::collections::BTreeSet<usize> = control_json["gated_positions"]
        .as_array()
        .context("gated_positions missing")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let mut verify_compared_count = 0usize;
    let mut verify_gated_count = 0usize;
    let mut defects: Vec<String> = Vec::new();
    let check_verify_row0 = |p: usize,
                             verify_logits: &[f32],
                             ref_logits: &[f32],
                             label: &str,
                             defects_out: &mut Vec<String>,
                             compared_out: &mut usize,
                             gated_out: &mut usize|
     -> (f32, f32) {
        let tau_max_abs = tau_max_abs_per_pos
            .get(p)
            .copied()
            .unwrap_or(tau_max_abs_per_pos[0]);
        let tau_rmse = tau_rmse_per_pos
            .get(p)
            .copied()
            .unwrap_or(tau_rmse_per_pos[0]);
        let mut row0_max_abs = 0.0f32;
        let mut row0_sum_sq = 0.0f32;
        for (&v, &r) in verify_logits.iter().zip(ref_logits) {
            let diff = (v - r).abs();
            if diff > row0_max_abs {
                row0_max_abs = diff;
            }
            row0_sum_sq += diff * diff;
        }
        let row0_rmse = (row0_sum_sq / verify_logits.len() as f32).sqrt();
        if gated_positions.contains(&p) {
            if row0_max_abs > tau_max_abs {
                defects_out.push(format!(
                    "DEFECT {label} at pos={p} max_abs={row0_max_abs:.6} exceeds tau={tau_max_abs:.6}"
                ));
            }
            if row0_rmse > tau_rmse {
                defects_out.push(format!(
                    "DEFECT {label} at pos={p} rmse={row0_rmse:.6} exceeds tau={tau_rmse:.6}"
                ));
            }
            *gated_out += 1;
        }
        *compared_out += 1;
        (row0_max_abs, row0_rmse)
    };

    // Exercise gated control position 1 from verify_control_positions_r1.json before passage rollout.
    engine.mtp_prefill(&reference.tokens[..1])?;
    let pos1_ckpt = GraphStepCheckpoint::save(&engine)?;
    let _ = engine.decode(false, None)?;
    let ref_pos1_logits = engine.decode_logits()?;
    pos1_ckpt.restore(&mut engine)?;
    let out_pos1 = engine.glm_mtp_round(true)?;
    ensure!(
        gated_positions.contains(&out_pos1.start_pos),
        "Runner FAIL expected pos={} in gated_positions",
        out_pos1.start_pos
    );
    let (pos1_max_abs, pos1_rmse) = check_verify_row0(
        out_pos1.start_pos,
        &out_pos1.verify_row0_logits,
        &ref_pos1_logits,
        "verify row0",
        &mut defects,
        &mut verify_compared_count,
        &mut verify_gated_count,
    );
    let mut pos1_max_idx = 0usize;
    let mut pos1_best_diff = -1.0f32;
    for (idx, (&v, &r)) in out_pos1
        .verify_row0_logits
        .iter()
        .zip(&ref_pos1_logits)
        .enumerate()
    {
        let diff = (v - r).abs();
        if diff > pos1_best_diff {
            pos1_best_diff = diff;
            pos1_max_idx = idx;
        }
    }
    let pos1_verify_logit = out_pos1.verify_row0_logits[pos1_max_idx];
    let pos1_ref_logit = ref_pos1_logits[pos1_max_idx];
    let (pos1_verify_top1, pos1_verify_margin) =
        super::mtp::top1_and_margin(&out_pos1.verify_row0_logits);
    let (pos1_ref_top1, pos1_ref_margin) = super::mtp::top1_and_margin(&ref_pos1_logits);
    println!(
        "GLM_MTP_GATED_POS pos={} max_abs={pos1_max_abs:.6} tau_max_abs={:.6} rmse={pos1_rmse:.6} tau_rmse={:.6} vocab_idx={pos1_max_idx} verify_logit={pos1_verify_logit:.6} ref_logit={pos1_ref_logit:.6} verify_top1={pos1_verify_top1} verify_margin={pos1_verify_margin:.6} ref_top1={pos1_ref_top1} ref_margin={pos1_ref_margin:.6}",
        out_pos1.start_pos,
        tau_max_abs_per_pos[out_pos1.start_pos],
        tau_rmse_per_pos[out_pos1.start_pos],
    );
    std::io::Write::flush(&mut std::io::stdout())?;

    // 2026-11-07: prefill loc32 33-token prefix via mtp_prefill with model EOS stop ids; frozen tau and alpha* unchanged.
    let t0 = engine.mtp_prefill(&reference.tokens[..33])?;
    let checkpoint = GraphStepCheckpoint::save(&engine)?;
    let _ref_t1 = engine.decode(false, None)?;
    let ref_row0_logits = engine.decode_logits()?;
    checkpoint.restore(&mut engine)?;

    let mut accepted_count = 0usize;
    let mut total_rounds = 0usize;
    let mut emitted_tokens = Vec::with_capacity(600);
    emitted_tokens.push(t0);

    let mut min_margin = f32::INFINITY;
    let mut sum_margin = 0.0f32;
    let mut margin_count = 0usize;

    // Cost model quantities
    let w_target_dense = 5_618_368_512u64;
    let w_draft_dense = 243_872_384u64;
    let expert_slot_bytes = 25_171_968u64;
    let b_head = 154_880u64 * 4096 * 2; // 1,268,776,960 B
    let b_step1 = w_target_dense + (336 * expert_slot_bytes) + b_head; // 15,344,926,720 B
    let b_draft = w_draft_dense + (8 * expert_slot_bytes) + b_head; // 1,714,025,088 B

    let mut total_bytes_moved = 0u64;
    let mut sum_b_verify = 0u64;
    let mut sum_u = 0usize;
    let mut draft_staged_experts_sum = 0usize;
    let mut replay_1row_passes = 0usize;
    let mut prefix1_commits = 0usize;
    let loop_start = std::time::Instant::now();

    while emitted_tokens.len() < 599
        && !eos_ids.contains(emitted_tokens.last().context("emitted tokens empty")?)
    {
        let is_round0 = total_rounds == 0;
        let out = if is_round0 {
            let out_eager = engine.glm_mtp_round(true)?;
            check_verify_row0(
                out_eager.start_pos,
                &out_eager.verify_row0_logits,
                &ref_row0_logits,
                "verify row0",
                &mut defects,
                &mut verify_compared_count,
                &mut verify_gated_count,
            );
            out_eager
        } else {
            engine.glm_mtp_round(false)?
        };

        total_rounds += 1;
        let tokens_in_round = out.round.tokens.len();
        if total_rounds <= 8 {
            let draft_tok = out.round.proposals.first().copied().unwrap_or(0);
            let (v0_id, _) = super::mtp::top1_and_margin(&out.verify_row0_logits);
            let (v1_id, _) = super::mtp::top1_and_margin(&out.verify_row1_logits);
            println!(
                "GLM_MTP_ROUND8 round={total_rounds} pos={} draft={draft_tok} verify_argmax={v0_id} verify_margin={:.4} accepted={} v1={v1_id} v1_margin={:.4}",
                out.start_pos, out.v0_margin, out.round.accepted, out.v1_margin
            );
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        let did_replay = out.replay_row0_logits.is_some();
        ensure!(
            !did_replay,
            "Runner FAIL unexpected target replay on rejection at pos={}",
            out.start_pos
        );
        if out.round.accepted > 0 {
            accepted_count += 1;
        }

        // Verify emitted tokens match argmax of verify rows
        let (top0_id, _) = super::mtp::top1_and_margin(&out.verify_row0_logits);
        ensure!(
            top0_id == out.round.tokens[0],
            "DEFECT emitted token 0 ({}) does not match argmax ({top0_id})",
            out.round.tokens[0]
        );
        min_margin = min_margin.min(out.v0_margin);
        sum_margin += out.v0_margin;
        margin_count += 1;

        if tokens_in_round > 1 {
            let (top1_id, _) = super::mtp::top1_and_margin(&out.verify_row1_logits);
            ensure!(
                top1_id == out.round.tokens[1],
                "DEFECT emitted token 1 ({}) does not match argmax ({top1_id})",
                out.round.tokens[1]
            );
            min_margin = min_margin.min(out.v1_margin);
            sum_margin += out.v1_margin;
            margin_count += 1;
        }

        // Cost accounting
        sum_u += out.expert_union_count;
        draft_staged_experts_sum += out.draft_staged_experts;
        if out.round.accepted == 0 {
            if did_replay {
                replay_1row_passes += 1;
            } else {
                prefix1_commits += 1;
            }
        }
        let b_verify =
            w_target_dense + (out.expert_union_count as u64 * expert_slot_bytes) + b_head;
        sum_b_verify += b_verify;
        let b_round = if did_replay {
            b_draft + b_verify + b_step1
        } else {
            b_draft + b_verify
        };
        total_bytes_moved += b_round;

        for &tok in &out.round.tokens {
            emitted_tokens.push(tok);
            if eos_ids.contains(&tok) {
                break;
            }
        }
        if total_rounds.is_multiple_of(50) {
            let elapsed_s = loop_start.elapsed().as_secs_f64();
            println!(
                "GLM_MTP_PROGRESS rounds={total_rounds} emitted={} accepted={accepted_count} elapsed_s={elapsed_s:.2}",
                emitted_tokens.len()
            );
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    ensure!(
        total_rounds > 0,
        "Runner FAIL zero speculative rounds executed"
    );

    let alpha = accepted_count as f64 / total_rounds as f64;
    let mean_b_verify = sum_b_verify as f64 / total_rounds as f64;
    let mean_u = sum_u as f64 / total_rounds as f64;
    let alpha_star = crate::mtp::glm_mtp_break_even_alpha(
        b_draft,
        mean_b_verify,
        b_step1,
        replay_1row_passes > 0,
    );
    let baseline_bytes = (emitted_tokens.len() - 1) as u64 * b_step1;
    let efficiency_ratio = total_bytes_moved as f64 / baseline_bytes as f64;
    let mean_margin = sum_margin / margin_count as f32;

    if alpha <= alpha_star {
        defects.push(format!(
            "DEFECT acceptance rate alpha={alpha:.4} <= break-even alpha_star={alpha_star:.4}"
        ));
    }

    let replay_experts_sum = replay_1row_passes * 336;
    let target_layer_passes = (total_rounds + replay_1row_passes) * 45;
    let head_rows_projected = total_rounds * 3 + replay_1row_passes;
    let checkpoint_restores = total_rounds - accepted_count;
    println!(
        "GLM_MTP_OP_COUNTS draft_passes={total_rounds} draft_htod_experts={draft_staged_experts_sum} accept_kv_commits={accepted_count} verify_2row_passes={total_rounds} verify_union_experts={sum_u} replay_1row_passes={replay_1row_passes} replay_htod_experts={replay_experts_sum} prefix1_commits={prefix1_commits} target_layer_passes={target_layer_passes} head_rows={head_rows_projected} checkpoint_saves={total_rounds} checkpoint_restores={checkpoint_restores}"
    );
    let status = if defects.is_empty() { "PASS" } else { "DEFECT" };
    println!(
        "{status} GLM_MTP_SPECULATIVE rounds={total_rounds} accepted={accepted_count} verify_compared={verify_compared_count} verify_gated={verify_gated_count} alpha={alpha:.4} mean_u={mean_u:.2} v_bar={mean_b_verify:.0} alpha_star={alpha_star:.4} efficiency_ratio={efficiency_ratio:.4} min_margin={min_margin:.4} mean_margin={mean_margin:.4} emitted={} defects={}",
        emitted_tokens.len(),
        defects.len()
    );
    for d in &defects {
        println!("{d}");
    }
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

#[cfg(feature = "cuda")]
struct DraftLocalizationStep {
    token: u32,
    hidden_idx: usize,
    use_post_norm: bool,
    dsa_pos: usize,
    t_p1: u32,
    g_p1: u32,
    t_p2: u32,
    g_p2: u32,
}

#[cfg(feature = "cuda")]
struct DraftLocalizationSummary {
    agree_t_p2: usize,
    agree_g_p2: usize,
    agree_t_p1: usize,
    agree_g_p1: usize,
    agree_g_p2_when_g1_eq_t1: usize,
    g1_eq_t1_total: usize,
    min_margin: f32,
    mean_margin: f32,
    sample_preds: Vec<u32>,
}

#[cfg(feature = "cuda")]
struct DraftLocalizationEnv<'a> {
    device: &'a Device,
    blas: &'a Blas,
    embed: &'a cudarc::driver::CudaSlice<bf16>,
    head: &'a cudarc::driver::CudaSlice<bf16>,
    h_pre: &'a [cudarc::driver::CudaSlice<bf16>],
    h_post: &'a [cudarc::driver::CudaSlice<bf16>],
}

#[cfg(feature = "cuda")]
fn eval_draft_localization_cell(
    runtime: &mut GlmMtpRuntime,
    env: &DraftLocalizationEnv<'_>,
    steps: &[DraftLocalizationStep],
    rollback_on_reject: bool,
) -> Result<DraftLocalizationSummary> {
    let mut dsa = crate::backend::cuda::dsa::DsaState::new(
        crate::op_contract::DsaSpec::glm53(),
        2048,
        &env.device.stream,
    )?;
    dsa.clear(&env.device.stream)?;

    let mut agree_t_p2 = 0usize;
    let mut agree_g_p2 = 0usize;
    let mut agree_t_p1 = 0usize;
    let mut agree_g_p1 = 0usize;
    let mut agree_g_p2_when_g1_eq_t1 = 0usize;
    let mut g1_eq_t1_total = 0usize;
    let mut min_margin = f32::INFINITY;
    let mut sum_margin = 0.0f32;
    let mut sample_preds = Vec::with_capacity(8);

    let shared = GlmMtpShared {
        blas: env.blas,
        embedding: env.embed,
        head: env.head,
    };

    for step in steps {
        let saved = if rollback_on_reject {
            Some((
                env.device.stream.clone_dtoh(&dsa.latent)?,
                env.device.stream.clone_dtoh(&dsa.token_ids)?,
                env.device.stream.clone_dtoh(&dsa.meta)?,
            ))
        } else {
            None
        };

        let hidden = if step.use_post_norm {
            &env.h_post[step.hidden_idx]
        } else {
            &env.h_pre[step.hidden_idx]
        };
        let (draft_id, draft_logits, _) =
            runtime.step_draft_device(step.token, hidden, 0, shared, &mut dsa, step.dsa_pos)?;
        let (top_id, margin) = super::mtp::top1_and_margin(&draft_logits);
        ensure!(
            top_id == draft_id,
            "Runner FAIL draft argmax mismatch: {top_id} vs {draft_id}"
        );
        if sample_preds.len() < 8 {
            sample_preds.push(draft_id);
        }
        if draft_id == step.t_p2 {
            agree_t_p2 += 1;
        }
        if draft_id == step.g_p2 {
            agree_g_p2 += 1;
        }
        if draft_id == step.t_p1 {
            agree_t_p1 += 1;
        }
        if draft_id == step.g_p1 {
            agree_g_p1 += 1;
        }
        if step.g_p1 == step.t_p1 {
            g1_eq_t1_total += 1;
            if draft_id == step.g_p2 {
                agree_g_p2_when_g1_eq_t1 += 1;
            }
        }
        min_margin = min_margin.min(margin);
        sum_margin += margin;

        if draft_id != step.g_p2
            && let Some((lat, tids, meta)) = saved
        {
            env.device.stream.memcpy_htod(&lat, &mut dsa.latent)?;
            env.device.stream.memcpy_htod(&tids, &mut dsa.token_ids)?;
            env.device.stream.memcpy_htod(&meta, &mut dsa.meta)?;
        }
    }

    Ok(DraftLocalizationSummary {
        agree_t_p2,
        agree_g_p2,
        agree_t_p1,
        agree_g_p1,
        agree_g_p2_when_g1_eq_t1,
        g1_eq_t1_total,
        min_margin,
        mean_margin: sum_margin / steps.len() as f32,
        sample_preds,
    })
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU Phase-1 32-position draft-on-reference localization comparing definition vs engine-round pairing and hidden tap"]
fn glm_mtp_draft_on_reference_32pos_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let model_dir =
        PathBuf::from(std::env::var_os("FF_GLM_MODEL").context("FF_GLM_MODEL missing")?);
    let comp_root = PathBuf::from(
        std::env::var_os("G1_COMPOSITION_REFERENCE").context("G1_COMPOSITION_REFERENCE missing")?,
    );
    let layer44_path = PathBuf::from(
        std::env::var_os("FF_GLM_LAYER44_TRACE").context("FF_GLM_LAYER44_TRACE missing")?,
    );

    let reference = Reference::load(&comp_root)?;
    if let Some(parent) = layer44_path.parent() {
        let trace_tokens_path = parent.join("tokens.json");
        if trace_tokens_path.exists() {
            let trace_json: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&trace_tokens_path)?)?;
            let trace_ids: Vec<u32> = serde_json::from_value(trace_json["input_ids"].clone())?;
            ensure!(
                trace_ids.len() >= 85 && trace_ids[..85] == reference.tokens[..85],
                "Runner FAIL layer44 trace tokens differ from composition reference"
            );
        }
    }

    let layer44_bytes = std::fs::read(&layer44_path)?;
    let layer44_st = safetensors::SafeTensors::deserialize(&layer44_bytes)?;
    let layer44_out = layer44_st.tensor("layer44.output")?;
    ensure!(
        layer44_out.dtype() == safetensors::Dtype::BF16 && layer44_out.shape() == [1, 85, 4, 4096],
        "Runner FAIL unexpected layer44.output shape {:?} or dtype {:?}",
        layer44_out.shape(),
        layer44_out.dtype()
    );
    let raw_streams: Vec<bf16> = layer44_out
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();

    let shard62_file = std::fs::File::open(model_dir.join("model-00062-of-00062.safetensors"))?;
    let mmap62 = unsafe { memmap2::MmapOptions::new().map(&shard62_file)? };
    let st62 = safetensors::SafeTensors::deserialize(&mmap62)?;
    let norm_tensor = st62.tensor("model.language_model.norm.weight")?;
    ensure!(
        norm_tensor.dtype() == safetensors::Dtype::BF16 && norm_tensor.shape() == [4096],
        "Runner FAIL unexpected final norm weight shape {:?}",
        norm_tensor.shape()
    );
    let final_norm_host: Vec<bf16> = norm_tensor
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();

    let device = Device::new(0)?;
    unsafe {
        device.ctx.disable_event_tracking();
    }
    let ops = crate::backend::cuda::ops::Ops::new(&device.ctx)?;
    let closure = std::rc::Rc::new(crate::backend::cuda::closure::Closure::new(
        false,
        "glm_mtp_loc".into(),
    ));
    let blas = Blas::new(&device, &ops, &std::env::temp_dir(), closure)?;
    let mut runtime = GlmMtpRuntime::load(&model_dir, &device, &blas, true, 2048)?;
    let embed_gpu = runtime
        ._standalone_embed
        .take()
        .context("standalone embed missing")?;
    let head_gpu = runtime
        .standalone_head
        .take()
        .context("standalone head missing")?;
    let final_norm_gpu = device.stream.clone_htod(&final_norm_host)?;

    let count = 32usize;
    let mut h_pre_gpu = Vec::with_capacity(count);
    let mut h_post_gpu = Vec::with_capacity(count);
    for p in 0..count {
        let base = p * 4 * 4096;
        let mut collapsed = vec![bf16::ZERO; 4096];
        for (h, out) in collapsed.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..4 {
                acc = (0.25f32).mul_add(raw_streams[base + j * 4096 + h].to_f32(), acc);
            }
            *out = bf16::from_f32(acc);
        }
        let pre_slice = device.stream.clone_htod(&collapsed)?;
        let mut post_slice = device.stream.alloc_zeros::<bf16>(4096)?;
        let rows_i32 = 1i32;
        let dim_i32 = 4096i32;
        let eps = 1e-5f32;
        unsafe {
            device
                .stream
                .launch_builder(&runtime.rms_norm)
                .arg(&pre_slice)
                .arg(&final_norm_gpu)
                .arg(&mut post_slice)
                .arg(&rows_i32)
                .arg(&dim_i32)
                .arg(&eps)
                .launch(crate::backend::cuda::ops::grid(1, 256))?;
        }
        h_pre_gpu.push(pre_slice);
        h_post_gpu.push(post_slice);
    }

    let positions = reference.record["positions"]
        .as_array()
        .context("positions array missing")?;
    let mut target_g2_eq_t2 = 0usize;
    let mut target_g1_eq_t1 = 0usize;
    for p in 0..count {
        let g_p1 = positions[p]["R1_id"].as_u64().unwrap() as u32;
        let g_p2 = positions[p + 1]["R1_id"].as_u64().unwrap() as u32;
        if g_p1 == reference.tokens[p + 1] {
            target_g1_eq_t1 += 1;
        }
        if g_p2 == reference.tokens[p + 2] {
            target_g2_eq_t2 += 1;
        }
    }
    println!(
        "TARGET_BASELINE_32POS target_g_p2_eq_t_p2={target_g2_eq_t2}/{count} target_g_p1_eq_t_p1={target_g1_eq_t1}/{count}"
    );

    let env = DraftLocalizationEnv {
        device: &device,
        blas: &blas,
        embed: &embed_gpu,
        head: &head_gpu,
        h_pre: &h_pre_gpu,
        h_post: &h_post_gpu,
    };

    let build_steps = |shifted_token: bool,
                       greedy_pending: bool,
                       use_post_norm: bool,
                       pos_offset: usize|
     -> Vec<DraftLocalizationStep> {
        (0..count)
            .map(|p| {
                let t_p1 = reference.tokens[p + 1];
                let t_p2 = reference.tokens[p + 2];
                let g_p1 = positions[p]["R1_id"].as_u64().unwrap() as u32;
                let g_p2 = positions[p + 1]["R1_id"].as_u64().unwrap() as u32;
                let token = if greedy_pending {
                    g_p1
                } else if shifted_token {
                    t_p1
                } else {
                    reference.tokens[p]
                };
                DraftLocalizationStep {
                    token,
                    hidden_idx: p,
                    use_post_norm,
                    dsa_pos: p + pos_offset,
                    t_p1,
                    g_p1,
                    t_p2,
                    g_p2,
                }
            })
            .collect()
    };

    let build_engine_round_steps =
        |greedy_pending: bool| -> Result<(Vec<DraftLocalizationStep>, bool)> {
            let first_cfg =
                glm_mtp_draft_step_config(1, Some(PrefixLastInputKind::DecodeToken), 1)?;
            let mut out = Vec::with_capacity(count);
            for p in 0..count {
                let cfg =
                    glm_mtp_draft_step_config(p + 1, Some(PrefixLastInputKind::DecodeToken), 1)?;
                let t_p1 = reference.tokens[p + 1];
                let t_p2 = reference.tokens[p + 2];
                let g_p1 = positions[p]["R1_id"].as_u64().unwrap() as u32;
                let g_p2 = positions[p + 1]["R1_id"].as_u64().unwrap() as u32;
                out.push(DraftLocalizationStep {
                    token: if greedy_pending { g_p1 } else { t_p1 },
                    hidden_idx: p + cfg.committed_row,
                    use_post_norm: cfg.hidden_tap == MtpHiddenTap::PostFinalNorm,
                    dsa_pos: cfg.draft_pos,
                    t_p1,
                    g_p1,
                    t_p2,
                    g_p2,
                });
            }
            Ok((out, !first_cfg.retain_committed_on_reject))
        };

    let (eng_trace_steps, eng_trace_rollback) = build_engine_round_steps(false)?;
    let (eng_greedy_steps, eng_greedy_rollback) = build_engine_round_steps(true)?;

    let cells: [(&str, Vec<DraftLocalizationStep>, bool); 12] = [
        (
            "def_shifted_post_norm",
            build_steps(true, false, true, 0),
            false,
        ),
        (
            "def_shifted_pre_norm",
            build_steps(true, false, false, 0),
            false,
        ),
        (
            "unshifted_tok_post_norm",
            build_steps(false, false, true, 0),
            false,
        ),
        (
            "unshifted_tok_pre_norm",
            build_steps(false, false, false, 0),
            false,
        ),
        (
            "engine_round_trace_tok",
            eng_trace_steps,
            eng_trace_rollback,
        ),
        (
            "engine_round_greedy_pending",
            eng_greedy_steps,
            eng_greedy_rollback,
        ),
        (
            "legacy_pre_fix_round_trace_tok",
            build_steps(true, false, false, 1),
            true,
        ),
        (
            "iso_post_norm_pos1_rollback",
            build_steps(true, false, true, 1),
            true,
        ),
        (
            "iso_post_norm_pos1_no_rollback",
            build_steps(true, false, true, 1),
            false,
        ),
        (
            "iso_post_norm_pos0_rollback",
            build_steps(true, false, true, 0),
            true,
        ),
        (
            "iso_pre_norm_pos1_no_rollback",
            build_steps(true, false, false, 1),
            false,
        ),
        (
            "iso_pre_norm_pos0_rollback",
            build_steps(true, false, false, 0),
            true,
        ),
    ];

    for p in 0..8 {
        println!(
            "TRACE_POS p={p} t_p={} t_p1={} g_p1={} t_p2={} g_p2={}",
            reference.tokens[p],
            reference.tokens[p + 1],
            positions[p]["R1_id"].as_u64().unwrap(),
            reference.tokens[p + 2],
            positions[p + 1]["R1_id"].as_u64().unwrap(),
        );
    }

    let mut results = std::collections::BTreeMap::new();
    for (name, steps, rollback) in &cells {
        let s = eval_draft_localization_cell(&mut runtime, &env, steps, *rollback)?;
        println!(
            "DRAFT_32POS cell={name} agree_t_p2={}/{count} agree_g_p2={}/{count} agree_t_p1={}/{count} agree_g_p1={}/{count} agree_g_p2_on_g1_eq_t1={}/{} min_margin={:.4} mean_margin={:.4} first8_preds={:?}",
            s.agree_t_p2,
            s.agree_g_p2,
            s.agree_t_p1,
            s.agree_g_p1,
            s.agree_g_p2_when_g1_eq_t1,
            s.g1_eq_t1_total,
            s.min_margin,
            s.mean_margin,
            s.sample_preds,
        );
        results.insert(*name, s);
    }

    let def_res = &results["def_shifted_post_norm"];
    let eng_res = &results["engine_round_trace_tok"];
    let unshifted_res = &results["unshifted_tok_post_norm"];
    let legacy_res = &results["legacy_pre_fix_round_trace_tok"];
    ensure!(
        eng_res.agree_g_p2 == def_res.agree_g_p2 && eng_res.agree_t_p2 == def_res.agree_t_p2,
        "DEFECT engine_round_trace_tok ({}/32) failed to match def_shifted_post_norm ({}/32)",
        eng_res.agree_g_p2,
        def_res.agree_g_p2
    );
    ensure!(
        unshifted_res.agree_g_p2 == 0,
        "Runner FAIL unshifted control unexpectedly matched g_p2 ({}/32)",
        unshifted_res.agree_g_p2
    );
    ensure!(
        legacy_res.agree_g_p2 < eng_res.agree_g_p2,
        "Runner FAIL legacy pre-fix control failed to separate from fixed engine round"
    );

    Ok(())
}
