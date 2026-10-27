use super::*;
use crate::mtp::teacher_forced_rows;
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, path::PathBuf};

struct Fixture {
    sequence: Vec<u32>,
    pre: Vec<bf16>,
    post: Vec<bf16>,
    logits_pre: Vec<bf16>,
    logits_post: Vec<bf16>,
}

#[test]
fn canonical_repair_uses_single_row_fc() -> Result<()> {
    for rows in 1..=4 {
        let mut calls = Vec::new();
        OutputMode::Repair.for_each_fc(rows, |row, count| {
            calls.push((row, count));
            Ok(())
        })?;
        assert_eq!(calls, (0..rows).map(|row| (row, 1)).collect::<Vec<_>>());
    }
    for rows in [1, 2, 3, 16, 53] {
        for output in [
            OutputMode::Predict,
            OutputMode::PredictDevice,
            OutputMode::StateOnly,
            OutputMode::Queued,
        ] {
            let mut calls = Vec::new();
            output.for_each_fc(rows, |row, count| {
                calls.push((row, count));
                Ok(())
            })?;
            assert_eq!(calls, [(0, rows)]);
        }
    }
    let mut calls = 0;
    assert!(
        OutputMode::Repair
            .for_each_fc(3, |_, _| {
                calls += 1;
                anyhow::bail!("injected FC error")
            })
            .is_err()
    );
    assert_eq!(calls, 1);
    Ok(())
}

#[test]
fn canonical_repair_preserves_decode_attention() {
    for phase in [ExecutionPhase::Decode, ExecutionPhase::Prefill] {
        assert_eq!(
            OutputMode::Repair.attention_phase(phase),
            ExecutionPhase::Decode
        );
        for output in [
            OutputMode::Predict,
            OutputMode::StateOnly,
            OutputMode::Queued,
        ] {
            assert_eq!(output.attention_phase(phase), phase);
        }
    }
}
impl Fixture {
    fn load(root: &Path, case: &str) -> Result<Self> {
        let ids: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join(format!("inputs/{case}.input_ids.json")),
        )?)?;
        let traces: BTreeMap<String, Vec<u32>> = serde_json::from_slice(&std::fs::read(
            root.join("sota/cloud-runs/VLLM_TRACE_IDS.json"),
        )?)?;
        let mut sequence: Vec<u32> = serde_json::from_value(ids["input_ids"].clone())?;
        let prompt = sequence.len();
        let generated = traces
            .get(case)
            .context("missing frozen candidate sequence")?;
        ensure!(
            generated.len() == 64,
            "MTP fixture expects 64 frozen candidate tokens"
        );
        sequence.extend_from_slice(generated);
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join("fixtures-bf16/mtp-tf-index.json"),
        )?)?;
        ensure!(
            index["prompts"][case]["rows"].as_u64() == Some(sequence.len() as u64)
                && index["prompts"][case]["prompt_len"].as_u64() == Some(prompt as u64),
            "MTP index/token length mismatch"
        );
        let bytes = std::fs::read(root.join(format!("fixtures-bf16/mtp-tf-{case}.safetensors")))?;
        let sf = SafeTensors::deserialize(&bytes)?;
        let read = |name: &str, rows: usize, width: usize| -> Result<Vec<bf16>> {
            let t = sf.tensor(name)?;
            ensure!(
                t.dtype() == Dtype::BF16 && t.shape() == [rows, width],
                "wrong fixture type/shape {name}"
            );
            let values: Vec<_> = t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite fixture {name}"
            );
            Ok(values)
        };
        let n = sequence.len();
        teacher_forced_rows(&sequence)?;
        Ok(Self {
            sequence,
            pre: read("target.head_input.prenorm", n, HIDDEN)?,
            post: read("target.head_input.postnorm", n, HIDDEN)?,
            logits_pre: read("mtp.logits", n - 1, VOCAB)?,
            logits_post: read("mtp.logits.postnorm_h2", n - 1, VOCAB)?,
        })
    }
    fn hidden(&self, tap: MtpHiddenTap) -> &[bf16] {
        match tap {
            MtpHiddenTap::PreFinalNorm => &self.pre,
            MtpHiddenTap::PostFinalNorm => &self.post,
        }
    }
    fn logits(&self, tap: MtpHiddenTap) -> &[bf16] {
        match tap {
            MtpHiddenTap::PreFinalNorm => &self.logits_pre,
            MtpHiddenTap::PostFinalNorm => &self.logits_post,
        }
    }
}

struct Probe {
    device: Device,
    ops: Ops,
    blas: Blas,
    catalog: WeightCatalog,
    spec: ModelSpec,
    root: PathBuf,
    output: PathBuf,
}
impl Probe {
    fn new() -> Result<Self> {
        ensure!(
            std::env::var("FF_Q38_MTP_WINDOW").as_deref() == Ok("granted"),
            "MTP local 4090 window missing"
        );
        let path = |name: &str| -> Result<PathBuf> {
            Ok(std::env::var_os(name)
                .with_context(|| format!("{name} missing"))?
                .into())
        };
        let model = path("FF_Q38_MODEL")?;
        let root = path("FF_Q38_MTP_REF_ROOT")?;
        let output = path("FF_Q38_MTP_OUTPUT")?;
        std::fs::create_dir_all(&output)?;
        let spec = ModelSpec::read(&model)?;
        let device = Device::new(0)?;
        ensure!(
            matches!(
                device.info.arch,
                crate::backend::setup::DeviceArchitecture::Cuda(86 | 89)
            ),
            "M1 requires an authorized sm_86 or sm_89 device"
        );
        let budget = plan(
            MtpHiddenTap::PreFinalNorm,
            116,
            128,
            ExecutionPhase::Prefill,
        )
        .memory(&spec)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16)
                .is_some_and(|b| budget.total <= b),
            "MTP memory budget exceeded: required={} free={free}",
            budget.total
        );
        eprintln!("MTP_MEMORY {}", serde_json::to_string(&budget)?);
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &output,
            Rc::new(Closure::new(false, "MTP M1".into())),
        )?;
        blas.enable_fp8(&device.ctx)?;
        let catalog =
            WeightCatalog::load_with_policy(&model, &spec, &device, LoadPolicy::MtpProbe)?;
        ensure!(
            catalog.bytes == crate::mtp::PROBE_WEIGHT_BYTES && catalog.tensors.len() == 24,
            "MTP probe selected unexpected weights"
        );
        Ok(Self {
            device,
            ops,
            blas,
            catalog,
            spec,
            root,
            output,
        })
    }
    fn state(&self, capacity: usize) -> Result<StateStore> {
        let mut spec = self.spec.clone();
        spec.layers = vec![LayerPlan {
            mixer: Mixer::FullAttention,
            state_index: 0,
            rope: true,
            sliding_window: None,
            ffn: crate::model_spec::FfnKind::Dense,
        }];
        let state = StateStore::new(&spec, capacity, &self.device.stream)?;
        ensure!(
            state.kv.len() == 1 && state.gdn.is_empty(),
            "MTP probe allocated target state"
        );
        Ok(state)
    }
    fn runtime(&self, plan: MtpPlan, state: &StateStore) -> Result<MtpRuntime> {
        let runtime = MtpRuntime::new(
            &self.device,
            &self.spec,
            &self.catalog,
            state,
            &self.blas,
            plan,
        )?;
        let budget = plan.memory(&self.spec)?;
        let allocated =
            self.catalog.bytes + runtime.allocation_bytes() + state.bytes() + budget.blas;
        ensure!(
            allocated == budget.total,
            "MTP allocation/budget mismatch: allocated={allocated} planned={}",
            budget.total
        );
        Ok(runtime)
    }
    fn run(
        &self,
        runtime: &mut MtpRuntime,
        state: &mut StateStore,
        input: RunInput<'_>,
    ) -> Result<Dump> {
        runtime.run(
            Execution {
                device: &self.device,
                ops: &self.ops,
                blas: &self.blas,
                catalog: &self.catalog,
                embedding: self
                    .catalog
                    .bf16("model.language_model.embed_tokens.weight")?,
                head: self.catalog.bf16("lm_head.weight")?,
            },
            state,
            input,
        )
    }
    fn save(
        &self,
        dump: &Dump,
        name: &str,
        plan: MtpPlan,
        prefix: usize,
        control: &str,
    ) -> Result<()> {
        let metadata = HashMap::from([
            ("contract".into(), MtpPlan::CONTRACT.into()),
            ("device".into(), self.device.info.name.clone()),
            ("hidden_tap".into(), format!("{:?}", plan.hidden_tap)),
            ("recurrent_tap".into(), format!("{:?}", plan.recurrent_tap)),
            ("phase".into(), format!("{:?}", plan.phase)),
            ("capacity".into(), plan.capacity.to_string()),
            ("prefix".into(), prefix.to_string()),
            ("rows".into(), plan.max_rows.to_string()),
            ("control".into(), control.into()),
            (
                "rules".into(),
                "aeac42dfc373d41a5e96cbac771ca9d4e900de5c19b5ddf226a9629c5ea54f38".into(),
            ),
            (
                "numeric_verdict".into(),
                "PENDING independent OpRef judgement".into(),
            ),
        ]);
        dump.save(&self.output.join(format!("{name}.safetensors")), metadata)
    }
}
fn plan(tap: MtpHiddenTap, rows: usize, capacity: usize, phase: ExecutionPhase) -> MtpPlan {
    MtpPlan {
        hidden_tap: tap,
        recurrent_tap: MtpRecurrentTap::PostMtpNorm,
        max_rows: rows,
        capacity,
        phase,
    }
}
fn values(dump: &Dump, name: &str) -> Result<Vec<bf16>> {
    let (dtype, _, data) = dump
        .tensors
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(*dtype == Dtype::BF16, "expected BF16 {name}");
    Ok(data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
        .collect())
}
/// Reference logit spread ε at a recursion depth (1-based), from `<root>/mtp-gate-bounds.json`.
fn near_tie_epsilon(root: &Path, depth: usize) -> Result<f32> {
    let path = root.join("mtp-gate-bounds.json");
    let bounds: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let epsilon = bounds["near_tie"]["epsilon_by_depth"][depth - 1]
        .as_f64()
        .with_context(|| format!("near_tie.epsilon_by_depth[{}] missing", depth - 1))?;
    Ok(epsilon as f32)
}
/// A row is a DEFECT when its max |Δlogit| exceeds `epsilon`, or when its argmax differs and the
/// reference margin exceeds the cut 2ε; a differing argmax within the cut is a NEAR-TIE.
fn score(candidate: &[bf16], reference: &[bf16], epsilon: f32) -> Result<serde_json::Value> {
    let cut = 2.0 * epsilon;
    ensure!(
        !candidate.is_empty()
            && candidate.len() == reference.len()
            && candidate.len().is_multiple_of(VOCAB),
        "empty/misaligned MTP logits"
    );
    let mut exact = 0;
    let mut near = 0;
    let mut defects = Vec::new();
    let mut max_abs = 0f32;
    let argmax = |row: &[bf16]| {
        let mut best = 0;
        for i in 1..row.len() {
            if row[i] > row[best] {
                best = i;
            }
        }
        best
    };
    for (r, (c, t)) in candidate
        .as_chunks::<VOCAB>()
        .0
        .iter()
        .zip(reference.as_chunks::<VOCAB>().0.iter())
        .enumerate()
    {
        ensure!(
            c.iter().chain(t).all(|x| x.is_finite()),
            "nonfinite logits row {r}"
        );
        let row_abs = c
            .iter()
            .zip(t)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0f32, f32::max);
        max_abs = max_abs.max(row_abs);
        let ci = argmax(c);
        let ti = argmax(t);
        let margin = t[ti].to_f32() - t[ci].to_f32();
        if row_abs > epsilon {
            defects.push(serde_json::json!({"row":r,"candidate":ci,"reference":ti,"margin":margin,"max_abs":row_abs}));
        } else if ci == ti {
            exact += 1;
        } else if margin <= cut {
            near += 1;
        } else {
            defects
                .push(serde_json::json!({"row":r,"candidate":ci,"reference":ti,"margin":margin}));
        }
    }
    Ok(
        serde_json::json!({"argmax":exact,"near_tie":near,"epsilon":epsilon,"cut":cut,"defects":defects,"max_abs_logits":max_abs,"opref":"PENDING"}),
    )
}

#[test]
#[ignore = "local 4090 MTP window; FF_Q38_MODEL, FF_Q38_MTP_REF_ROOT, FF_Q38_MTP_OUTPUT required"]
fn q38_mtp_single_step_gpu() -> Result<()> {
    let probe = Probe::new()?;
    let epsilon = near_tie_epsilon(&probe.root, 1)?;
    for case in ["math1", "code1", "gen1", "gen2", "math2"] {
        let f = Fixture::load(&probe.root, case)?;
        let rows = f.sequence.len() - 1;
        for tap in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
            let p = plan(tap, rows, rows.div_ceil(32) * 32, ExecutionPhase::Prefill);
            let mut state = probe.state(p.capacity)?;
            let mut runtime = probe.runtime(p, &state)?;
            let dump = probe.run(
                &mut runtime,
                &mut state,
                RunInput {
                    device_ids: None,
                    prefix: 0,
                    ids: &f.sequence[1..],
                    hidden: &f.hidden(tap)[..rows * HIDDEN],
                    control: Control::None,
                    detailed: case == "math1",
                },
            )?;
            let name = format!("{case}-{tap:?}-full");
            probe.save(&dump, &name, p, 0, "None")?;
            let result = score(&values(&dump, "logits")?, f.logits(tap), epsilon)?;
            std::fs::write(
                probe.output.join(format!("{name}-score.json")),
                serde_json::to_vec_pretty(&result)?,
            )?;
            eprintln!("MTP_SINGLE_STEP {name} {result}");
            ensure!(
                result["defects"]
                    .as_array()
                    .context("score defects missing")?
                    .is_empty(),
                "MTP first logits DEFECT: {name}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "local 4090 MTP window; cached row fixture comparison"]
fn q38_mtp_cached_rows_gpu() -> Result<()> {
    let probe = Probe::new()?;
    let epsilon = near_tie_epsilon(&probe.root, 1)?;
    for case in ["math1", "code1", "gen1", "gen2", "math2"] {
        let f = Fixture::load(&probe.root, case)?;
        let rows = f.sequence.len() - 1;
        for tap in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
            // Exact final slot, including a non-multiple-of-32 capacity.
            let p = plan(tap, 1, rows, ExecutionPhase::Decode);
            let mut state = probe.state(p.capacity)?;
            let mut runtime = probe.runtime(p, &state)?;
            let mut logits = Vec::with_capacity(rows * VOCAB);
            for r in 0..rows {
                let dump = probe.run(
                    &mut runtime,
                    &mut state,
                    RunInput {
                        device_ids: None,
                        prefix: r,
                        ids: &f.sequence[r + 1..r + 2],
                        hidden: &f.hidden(tap)[r * HIDDEN..(r + 1) * HIDDEN],
                        control: Control::None,
                        detailed: case == "math1" && r == rows - 1,
                    },
                )?;
                logits.extend(values(&dump, "logits")?);
                if case == "math1" && r == rows - 1 {
                    probe.save(&dump, &format!("{case}-{tap:?}-cached-last"), p, r, "None")?;
                }
            }
            let mut dump = Dump::default();
            dump.bf16("logits", vec![rows, VOCAB], logits.clone())?;
            let name = format!("{case}-{tap:?}-cached");
            probe.save(&dump, &name, p, 0, "None")?;
            let result = score(&logits, f.logits(tap), epsilon)?;
            std::fs::write(
                probe.output.join(format!("{name}-score.json")),
                serde_json::to_vec_pretty(&result)?,
            )?;
            eprintln!("MTP_CACHED {name} {result}");
            ensure!(
                result["defects"]
                    .as_array()
                    .context("score defects missing")?
                    .is_empty(),
                "MTP first cached logits DEFECT: {name}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "authorized GPU window; StateOnly Decode must preserve every intermediate and KV bit"]
fn q38_mtp_state_only_decode_gpu() -> Result<()> {
    let probe = Probe::new()?;
    let fixture = Fixture::load(&probe.root, "math1")?;
    let s = &probe.device.stream;
    for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
        for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
            let mut p = plan(h2, 1, 64, ExecutionPhase::Decode);
            p.recurrent_tap = h5;
            let mut full_state = probe.state(p.capacity)?;
            let mut light_state = probe.state(p.capacity)?;
            let mut full = probe.runtime(p, &full_state)?;
            let mut light = probe.runtime(p, &light_state)?;
            let sentinel = vec![bf16::from_f32(-17.0); VOCAB];
            s.memcpy_htod(&sentinel, &mut light.logits)?;
            let mut hidden = fixture.hidden(h2)[..HIDDEN].to_vec();
            for prefix in 0..8 {
                let input = || RunInput {
                    device_ids: None,
                    prefix,
                    ids: &fixture.sequence[prefix + 1..prefix + 2],
                    hidden: &hidden,
                    control: Control::None,
                    detailed: true,
                };
                let mut full_dump = probe.run(&mut full, &mut full_state, input())?;
                let light_dump = light.run_decode(
                    Execution {
                        device: &probe.device,
                        ops: &probe.ops,
                        blas: &probe.blas,
                        catalog: &probe.catalog,
                        embedding: probe
                            .catalog
                            .bf16("model.language_model.embed_tokens.weight")?,
                        head: probe.catalog.bf16("lm_head.weight")?,
                    },
                    &mut light_state,
                    input(),
                    OutputMode::StateOnly,
                )?;
                assert!(full_dump.tensors.remove("logits").is_some());
                assert_eq!(full_dump.tensors, light_dump.tensors);
                assert_eq!(full.head_runs, prefix + 1);
                assert_eq!(light.head_runs, 0);
                let bits = |v: Vec<bf16>| v.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
                assert_eq!(bits(s.clone_dtoh(&light.logits)?), bits(sentinel.clone()));
                for (a, b) in [
                    (&full_state.kv[0].k, &light_state.kv[0].k),
                    (&full_state.kv[0].v, &light_state.kv[0].v),
                ] {
                    assert_eq!(bits(s.clone_dtoh(a)?), bits(s.clone_dtoh(b)?));
                }
                assert_eq!(
                    s.clone_dtoh(
                        full_state
                            .sequence
                            .as_ref()
                            .context("full metadata missing")?
                    )?,
                    s.clone_dtoh(
                        light_state
                            .sequence
                            .as_ref()
                            .context("StateOnly metadata missing")?
                    )?
                );
                hidden = s.clone_dtoh(
                    &match h5 {
                        MtpRecurrentTap::PreMtpNorm => &full.work.x,
                        MtpRecurrentTap::PostMtpNorm => &full.work.n,
                    }
                    .slice(..HIDDEN),
                )?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "authorized MTP GPU window; headless Prefill must preserve hidden and KV bits"]
fn q38_mtp_headless_prefill_gpu() -> Result<()> {
    let probe = Probe::new()?;
    let fixture = Fixture::load(&probe.root, "math1")?;
    let s = &probe.device.stream;
    for rows in [1, 2, 16] {
        let p = plan(
            MtpHiddenTap::PostFinalNorm,
            rows,
            64,
            ExecutionPhase::Prefill,
        );
        let mut full_state = probe.state(p.capacity)?;
        let mut light_state = probe.state(p.capacity)?;
        let mut full = probe.runtime(p, &full_state)?;
        let mut light = MtpRuntime::allocate(
            &probe.device,
            &probe.spec,
            &probe.catalog,
            &light_state,
            &probe.blas,
            p,
            1,
        )?;
        let hidden = &fixture.hidden(p.hidden_tap)[..rows * HIDDEN];
        let device_hidden = s.clone_htod(hidden)?;
        let ids = &fixture.sequence[1..=rows];
        probe.run(
            &mut full,
            &mut full_state,
            RunInput {
                device_ids: None,
                prefix: 0,
                ids,
                hidden,
                control: Control::None,
                detailed: false,
            },
        )?;
        light.run_prefill(
            Execution {
                device: &probe.device,
                ops: &probe.ops,
                blas: &probe.blas,
                catalog: &probe.catalog,
                embedding: probe
                    .catalog
                    .bf16("model.language_model.embed_tokens.weight")?,
                head: probe.catalog.bf16("lm_head.weight")?,
            },
            &mut light_state,
            0,
            ids,
            matrix(&device_hidden, s, 0, rows, HIDDEN)?,
        )?;
        assert_eq!(full.head_runs, 1);
        assert_eq!(light.head_runs, 0);
        assert_eq!(light.logits.len(), VOCAB);
        let bits = |v: Vec<bf16>| v.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
        for (a, b) in [(&full.work.x, &light.work.x), (&full.work.n, &light.work.n)] {
            assert_eq!(
                bits(s.clone_dtoh(&a.slice(..rows * HIDDEN))?),
                bits(s.clone_dtoh(&b.slice(..rows * HIDDEN))?)
            );
        }
        for (a, b) in [
            (&full_state.kv[0].k, &light_state.kv[0].k),
            (&full_state.kv[0].v, &light_state.kv[0].v),
        ] {
            assert_eq!(bits(s.clone_dtoh(a)?), bits(s.clone_dtoh(b)?));
        }
        assert_eq!(
            s.clone_dtoh(full_state.sequence.as_ref().unwrap())?,
            s.clone_dtoh(light_state.sequence.as_ref().unwrap())?
        );
    }
    repair_prefill_matches_decode(&probe, &fixture)
}

fn repair_prefill_matches_decode(probe: &Probe, fixture: &Fixture) -> Result<()> {
    let s = &probe.device.stream;
    let execution = || -> Result<Execution<'_>> {
        Ok(Execution {
            device: &probe.device,
            ops: &probe.ops,
            blas: &probe.blas,
            catalog: &probe.catalog,
            embedding: probe
                .catalog
                .bf16("model.language_model.embed_tokens.weight")?,
            head: probe.catalog.bf16("lm_head.weight")?,
        })
    };
    let bits = |v: Vec<bf16>| v.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
    let mut receipts = Vec::new();
    for tap in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
        let mut seed_state = probe.state(64)?;
        let mut seed = probe.runtime(plan(tap, 53, 64, ExecutionPhase::Prefill), &seed_state)?;
        seed.run_inner(
            execution()?,
            &mut seed_state,
            RunInput {
                prefix: 0,
                ids: &fixture.sequence[1..54],
                device_ids: None,
                hidden: &fixture.hidden(tap)[..53 * HIDDEN],
                control: Control::None,
                detailed: false,
            },
            None,
            ExecutionPhase::Prefill,
            OutputMode::StateOnly,
        )?;
        for rows in 1..=4 {
            let mut serial_state = probe.state(64)?;
            let mut repair_state = probe.state(64)?;
            let mut old_state = probe.state(64)?;
            for state in [&mut serial_state, &mut repair_state, &mut old_state] {
                s.memcpy_dtod(&seed_state.kv[0].k, &mut state.kv[0].k)?;
                s.memcpy_dtod(&seed_state.kv[0].v, &mut state.kv[0].v)?;
            }
            let mut serial =
                probe.runtime(plan(tap, 1, 64, ExecutionPhase::Decode), &serial_state)?;
            let p = plan(tap, rows, 64, ExecutionPhase::Prefill);
            let mut repair = probe.runtime(p, &repair_state)?;
            let mut old = probe.runtime(p, &old_state)?;
            let mut pre = Vec::new();
            let mut post = Vec::new();
            let mut serial_dump = Dump::default();
            let mut serial_fc = Dump::default();
            for row in 0..rows {
                serial_dump = serial.run_inner(
                    execution()?,
                    &mut serial_state,
                    RunInput {
                        prefix: 53 + row,
                        ids: &fixture.sequence[54 + row..55 + row],
                        device_ids: None,
                        hidden: &fixture.hidden(tap)[(53 + row) * HIDDEN..(54 + row) * HIDDEN],
                        control: Control::None,
                        detailed: true,
                    },
                    None,
                    ExecutionPhase::Decode,
                    OutputMode::StateOnly,
                )?;
                for name in [
                    "pre_fc_norm_embedding",
                    "pre_fc_norm_hidden",
                    "concat",
                    "fc",
                    "norm1",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ] {
                    let (dtype, shape, bytes) = serial_dump
                        .tensors
                        .get(name)
                        .context("serial FC diagnostic missing")?;
                    ensure!(
                        *dtype == Dtype::BF16 && shape.len() == 2 && shape[0] == 1,
                        "serial FC diagnostic shape mismatch"
                    );
                    let value = serial_fc
                        .tensors
                        .entry(name.into())
                        .or_insert_with(|| (*dtype, vec![rows, shape[1]], Vec::new()));
                    value.2.extend_from_slice(bytes);
                }
                pre.extend(bits(s.clone_dtoh(&serial.work.x.slice(..HIDDEN))?));
                post.extend(bits(s.clone_dtoh(&serial.work.n.slice(..HIDDEN))?));
            }
            let mut dumps = Vec::new();
            for (runtime, state, output) in [
                (&mut repair, &mut repair_state, OutputMode::Repair),
                (&mut old, &mut old_state, OutputMode::StateOnly),
            ] {
                dumps.push(runtime.run_inner(
                    execution()?,
                    state,
                    RunInput {
                        prefix: 53,
                        ids: &fixture.sequence[54..54 + rows],
                        device_ids: None,
                        hidden: &fixture.hidden(tap)[53 * HIDDEN..(53 + rows) * HIDDEN],
                        control: Control::None,
                        detailed: true,
                    },
                    None,
                    ExecutionPhase::Prefill,
                    output,
                )?);
            }
            let stem = format!("repair-{tap:?}-rows{rows}");
            serial_fc.save(
                &probe.output.join(format!("{stem}-serial-fc.safetensors")),
                HashMap::new(),
            )?;
            for (name, dump) in [("repair", &dumps[0]), ("old_fa2", &dumps[1])] {
                dump.save(
                    &probe.output.join(format!("{stem}-{name}.safetensors")),
                    HashMap::new(),
                )?;
            }
            let expected = [
                pre,
                post,
                bits(s.clone_dtoh(&serial_state.kv[0].k)?),
                bits(s.clone_dtoh(&serial_state.kv[0].v)?),
            ];
            let mut checks = Vec::new();
            for (name, runtime, state, dump) in [
                ("repair", &repair, &repair_state, &dumps[0]),
                ("old_fa2", &old, &old_state, &dumps[1]),
            ] {
                let actual = [
                    bits(s.clone_dtoh(&runtime.work.x.slice(..rows * HIDDEN))?),
                    bits(s.clone_dtoh(&runtime.work.n.slice(..rows * HIDDEN))?),
                    bits(s.clone_dtoh(&state.kv[0].k)?),
                    bits(s.clone_dtoh(&state.kv[0].v)?),
                ];
                let differences: Vec<_> = ["pre_mtp_norm", "post_mtp_norm", "k_cache", "v_cache"]
                    .into_iter()
                    .zip(expected.iter().zip(&actual))
                    .filter_map(|(field, (a, b))| {
                        a.iter().zip(b).position(|(a, b)| a != b).map(|index| {
                            serde_json::json!({"field":field,"word":index,"expected":a[index],"actual":b[index]})
                        })
                    })
                    .collect();
                let mut fc_differences = Vec::new();
                for (field, expected) in &serial_fc.tensors {
                    let actual = dump
                        .tensors
                        .get(field)
                        .context("batched FC diagnostic missing")?;
                    if expected != actual {
                        let word = expected
                            .2
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .zip(actual.2.as_chunks::<2>().0)
                            .position(|(a, b)| a != b);
                        fc_differences.push(serde_json::json!({"field":field,"word":word,"expected_shape":expected.1,"actual_shape":actual.1}));
                    }
                }
                checks.push(serde_json::json!({"arm":name,"differences":differences,"fc_differences":fc_differences}));
            }
            let first_old_op = if rows == 1 {
                [
                    "pre_fc_norm_embedding",
                    "pre_fc_norm_hidden",
                    "concat",
                    "fc",
                    "norm1",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "k_norm",
                    "q_norm",
                    "q_rope",
                    "k_cache",
                    "v_cache",
                    "attention_core",
                    "attention_gated",
                    "o_proj",
                    "residual1",
                    "norm2",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp_activation",
                    "mlp_down",
                    "pre_mtp_norm",
                    "post_mtp_norm",
                ]
                .into_iter()
                .find(|name| serial_dump.tensors.get(*name) != dumps[1].tensors.get(*name))
            } else {
                [
                    "pre_fc_norm_embedding",
                    "pre_fc_norm_hidden",
                    "concat",
                    "fc",
                    "norm1",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ]
                .into_iter()
                .find(|name| serial_fc.tensors.get(*name) != dumps[1].tensors.get(*name))
            };
            receipts.push(serde_json::json!({"hidden_tap":format!("{tap:?}"),"prefix":53,"rows":rows,"hidden_shape":[rows,HIDDEN],"old_first_different_op":first_old_op,"checks":checks}));
            std::fs::write(
                probe.output.join("repair-prefill.json"),
                serde_json::to_vec_pretty(&receipts)?,
            )?;
            ensure!(
                checks[0]["differences"]
                    .as_array()
                    .is_some_and(Vec::is_empty)
                    && checks[0]["fc_differences"]
                        .as_array()
                        .is_some_and(Vec::is_empty),
                "MTP repair Prefill differs from serial Decode tap={tap:?} rows={rows}"
            );
        }
    }
    ensure!(
        receipts
            .iter()
            .any(|r| r["rows"] == 2 && r["old_first_different_op"] == "fc"),
        "old batched FC negative control did not discriminate"
    );
    ensure!(
        receipts.iter().any(|r| r["checks"][1]["differences"]
            .as_array()
            .is_some_and(|v| !v.is_empty())),
        "old FA2 repair negative control did not discriminate"
    );
    println!(
        "MTP_REPAIR_PREFILL_BITWISE_PASS cases=8 prefix=53 rows=1,2,3,4 fields=pre_mtp_norm,post_mtp_norm,k_cache,v_cache fc_intermediates=true old_fc_control=true old_fa2_control=true"
    );
    Ok(())
}

#[test]
#[ignore = "local 4090 MTP window; exports controls for independent OpRef judgement"]
fn q38_mtp_negative_controls_gpu() -> Result<()> {
    let probe = Probe::new()?;
    let f = Fixture::load(&probe.root, "math1")?;
    for tap in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
        let p = plan(tap, 1, 128, ExecutionPhase::Decode);
        let mut state = probe.state(p.capacity)?;
        let mut runtime = probe.runtime(p, &state)?;
        // Establish identical past. Only slot 10 is overwritten by each control.
        for r in 0..10 {
            probe.run(
                &mut runtime,
                &mut state,
                RunInput {
                    device_ids: None,
                    prefix: r,
                    ids: &f.sequence[r + 1..r + 2],
                    hidden: &f.hidden(tap)[r * HIDDEN..(r + 1) * HIDDEN],
                    control: Control::None,
                    detailed: false,
                },
            )?;
        }
        let run = |runtime: &mut MtpRuntime,
                   state: &mut StateStore,
                   control: Control,
                   hidden_tap: MtpHiddenTap| {
            probe.run(
                runtime,
                state,
                RunInput {
                    device_ids: None,
                    prefix: 10,
                    ids: &f.sequence[11..12],
                    hidden: &f.hidden(hidden_tap)[10 * HIDDEN..11 * HIDDEN],
                    control,
                    detailed: true,
                },
            )
        };
        let baseline = run(&mut runtime, &mut state, Control::None, tap)?;
        probe.save(
            &baseline,
            &format!("math1-{tap:?}-control-baseline"),
            p,
            10,
            "None",
        )?;
        let other = match tap {
            MtpHiddenTap::PreFinalNorm => MtpHiddenTap::PostFinalNorm,
            MtpHiddenTap::PostFinalNorm => MtpHiddenTap::PreFinalNorm,
        };
        for (control, input_tap, label, key) in [
            (Control::SwapConcat, tap, "swap_concat", "fc"),
            (
                Control::ShiftNewPosition,
                tap,
                "shift_new_position",
                "q_rope",
            ),
            (Control::None, other, "swap_hidden_tap", "fc"),
            (
                Control::StandardNorm,
                tap,
                "standard_norm",
                "pre_fc_norm_hidden",
            ),
        ] {
            let dump = run(&mut runtime, &mut state, control, input_tap)?;
            probe.save(
                &dump,
                &format!("math1-{tap:?}-control-{label}"),
                p,
                10,
                label,
            )?;
            ensure!(
                values(&dump, key)?
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(values(&baseline, key)?.iter().map(|v| v.to_bits())),
                "negative control {label} did not perturb {key}"
            );
            // A perturbation is only a harness self-check, never a numeric verdict.
            eprintln!(
                "MTP_CONTROL_EXPORTED tap={tap:?} control={label}; independent frozen-rule detection PENDING"
            );
        }
        // Each slot r runs the control over an uncontrolled past 0..r-1, then is rewritten
        // without the control; the controlled rows must score at least one DEFECT.
        let epsilon = near_tie_epsilon(&probe.root, 1)?;
        let rows = f.sequence.len() - 1;
        for (control, label) in [
            (Control::ShiftNewPosition, "shift_new_position"),
            (Control::SwapConcat, "swap_concat"),
        ] {
            let p = plan(tap, 1, rows, ExecutionPhase::Decode);
            let mut state = probe.state(p.capacity)?;
            let mut runtime = probe.runtime(p, &state)?;
            let mut logits = Vec::with_capacity(rows * VOCAB);
            for r in 0..rows {
                for c in [control, Control::None] {
                    let dump = probe.run(
                        &mut runtime,
                        &mut state,
                        RunInput {
                            device_ids: None,
                            prefix: r,
                            ids: &f.sequence[r + 1..r + 2],
                            hidden: &f.hidden(tap)[r * HIDDEN..(r + 1) * HIDDEN],
                            control: c,
                            detailed: false,
                        },
                    )?;
                    if !matches!(c, Control::None) {
                        logits.extend(values(&dump, "logits")?);
                    }
                }
            }
            let result = score(&logits, f.logits(tap), epsilon)?;
            std::fs::write(
                probe
                    .output
                    .join(format!("math1-{tap:?}-control-{label}-slots-score.json")),
                serde_json::to_vec_pretty(&result)?,
            )?;
            eprintln!("MTP_CONTROL_SCORE tap={tap:?} control={label} {result}");
            let detected = !result["defects"]
                .as_array()
                .context("score defects missing")?
                .is_empty();
            ensure!(
                detected,
                "control {label} scored no DEFECT at epsilon {epsilon}"
            );
        }
    }
    Ok(())
}

#[test]
fn score_separates_near_tie_from_defect_at_the_cut() -> Result<()> {
    let mut reference = vec![bf16::ZERO; 3 * VOCAB];
    let mut candidate = reference.clone();
    let mut set = |row: usize, i: usize, t: f32, c: f32| {
        reference[row * VOCAB + i] = bf16::from_f32(t);
        candidate[row * VOCAB + i] = bf16::from_f32(c);
    };
    // ε = 0.25, cut 0.5. Row 0: margin 0.375, |Δ| 0.25, argmax flips.
    set(0, 7, 16.0, 15.75);
    set(0, 11, 15.625, 15.875);
    // Row 1: argmax kept, |Δ| 0.375.
    set(1, 7, 16.0, 16.0);
    set(1, 3, 1.0, 1.375);
    // Row 2: margin 0.75, |Δ| 0.4375, argmax flips.
    set(2, 7, 16.0, 15.625);
    set(2, 11, 15.25, 15.6875);
    let result = score(&candidate, &reference, 0.25)?;
    assert_eq!(result["near_tie"], 1);
    assert_eq!(result["argmax"], 0);
    let defects = result["defects"].as_array().context("defects")?;
    assert_eq!(
        defects.iter().map(|d| d["row"].clone()).collect::<Vec<_>>(),
        [1, 2]
    );
    Ok(())
}

#[test]
#[ignore = "cloud full-model window; M2a K1..3, both H2/H5, ordinary target token equality"]
fn q38_mtp_chain_gpu() -> Result<()> {
    super::chain::tests::q38_mtp_chain_gpu()
}

#[test]
#[ignore = "cloud full-model window; all acceptance lengths and bit-exact committed state"]
fn q38_mtp_transaction_gpu() -> Result<()> {
    super::chain::tests::q38_mtp_transaction_gpu()
}

#[test]
#[ignore = "cloud full-model window; rollback, warmup/capture failures and retry"]
fn q38_mtp_failure_and_capture_gpu() -> Result<()> {
    super::chain::tests::q38_mtp_failure_and_capture_gpu()
}

#[test]
#[ignore = "MTP window and qualified FF_Q38_MTP_CHAIN_INDEX required; numerical recursion fixture"]
fn q38_mtp_recursive_fixture_gpu() -> Result<()> {
    let index_path = PathBuf::from(
        std::env::var_os("FF_Q38_MTP_CHAIN_INDEX").context("qualified chain index required")?,
    );
    fn check_sha(path: &Path, expected: &str) -> Result<()> {
        ensure!(
            expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid fixture SHA256"
        );
        ensure!(
            crate::prefix_state::file_sha256(path)? == expected,
            "fixture SHA256 mismatch: {}",
            path.display()
        );
        Ok(())
    }
    check_sha(
        &index_path,
        "4c6c2b69a6fdf2208905b65c1e82cc2036411fc58bc86dda9087dd4d7e51042e",
    )?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    ensure!(
        index["architecture"].as_str() == Some("sm_89"),
        "chain reference architecture mismatch"
    );
    let cross = index["crosscheck_full_length"]
        .as_object()
        .context("chain fixture lacks construction identity gate")?;
    ensure!(
        cross.len() == 16 && cross.values().all(|v| v.as_str() == Some("bit-equal")),
        "chain construction identity is not qualified (requires same-architecture 16/16)"
    );
    ensure!(
        index["h5_control"].as_str() == Some("FAIL(detected)"),
        "chain H5 control is not qualified"
    );
    let file = index["fixture"]["file"]
        .as_str()
        .context("chain fixture filename missing")?;
    ensure!(
        std::path::Path::new(file).components().count() == 1 && !file.starts_with('.'),
        "invalid chain fixture filename"
    );
    let fixture_path = index_path
        .parent()
        .context("index parent missing")?
        .join(file);
    check_sha(
        &fixture_path,
        index["fixture"]["sha256"]
            .as_str()
            .context("chain SHA missing")?,
    )?;
    let bytes = std::fs::read(fixture_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let read = |name: &str, width: usize| -> Result<Vec<bf16>> {
        let t = tensors.tensor(name)?;
        ensure!(
            t.dtype() == Dtype::BF16 && t.shape() == [width],
            "chain tensor shape/dtype mismatch {name}"
        );
        Ok(t.data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
            .collect())
    };
    let chains = index["chains"]
        .as_object()
        .context("chain records missing")?;
    ensure!(chains.len() == 32, "expected 8 anchors x H2 x H5");
    // The index must identify which target-hidden fixture produced its chains.
    let source = index["hidden_source"]["file"]
        .as_str()
        .context("chain index must pin hidden_source filename")?;
    ensure!(
        std::path::Path::new(source).components().count() == 1 && !source.starts_with('.'),
        "invalid hidden source filename"
    );
    let hidden_path = index_path
        .parent()
        .context("index parent missing")?
        .join(source);
    check_sha(
        &hidden_path,
        index["hidden_source"]["sha256"]
            .as_str()
            .context("hidden source SHA missing")?,
    )?;
    let hidden_bytes = std::fs::read(hidden_path)?;
    let hidden_sf = SafeTensors::deserialize(&hidden_bytes)?;
    let root =
        PathBuf::from(std::env::var_os("FF_Q38_MTP_REF_ROOT").context("reference root missing")?);
    let fixture = Fixture::load(&root, "math1")?;
    let probe = Probe::new()?;
    for (key, record) in chains {
        let parts: Vec<_> = key.split('.').collect();
        ensure!(parts.len() == 3, "invalid chain key");
        let h2 = match parts[1] {
            "prenorm" => MtpHiddenTap::PreFinalNorm,
            "postnorm" => MtpHiddenTap::PostFinalNorm,
            _ => anyhow::bail!("unknown H2"),
        };
        let h5 = match parts[2] {
            "PreFinalNorm" => MtpRecurrentTap::PreMtpNorm,
            "PostFinalNorm" => MtpRecurrentTap::PostMtpNorm,
            _ => anyhow::bail!("unknown H5"),
        };
        let j = usize::try_from(record["j0"].as_u64().context("chain j0 missing")?)?;
        ensure!(
            j > 0 && j < fixture.sequence.len(),
            "chain anchor outside sequence"
        );
        let t = hidden_sf.tensor(h2.fixture_key())?;
        ensure!(
            t.dtype() == Dtype::BF16 && t.shape() == [fixture.sequence.len(), HIDDEN],
            "chain hidden source geometry mismatch"
        );
        let hidden: Vec<_> = t
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
            .collect();
        let p = MtpPlan {
            recurrent_tap: h5,
            ..plan(h2, 1, 128, ExecutionPhase::Decode)
        };
        let mut state = probe.state(p.capacity)?;
        let mut runtime = probe.runtime(p, &state)?;
        for r in 0..j - 1 {
            probe.run(
                &mut runtime,
                &mut state,
                RunInput {
                    device_ids: None,
                    prefix: r,
                    ids: &fixture.sequence[r + 1..r + 2],
                    hidden: &hidden[r * HIDDEN..(r + 1) * HIDDEN],
                    control: Control::None,
                    detailed: false,
                },
            )?;
        }
        let steps = record["steps"].as_array().context("chain steps missing")?;
        ensure!(steps.len() == 3, "chain requires depths1..3");
        let mut recurrent = hidden[(j - 1) * HIDDEN..j * HIDDEN].to_vec();
        for (depth, step) in steps.iter().enumerate() {
            let token = u32::try_from(step["tok_in"].as_u64().context("input token missing")?)?;
            ensure!(
                step["abs_pos"].as_u64() == Some((j + depth) as u64),
                "chain absolute position mismatch"
            );
            if depth == 0 {
                ensure!(token == fixture.sequence[j], "chain initial token mismatch");
            }
            let dump = probe.run(
                &mut runtime,
                &mut state,
                RunInput {
                    device_ids: None,
                    prefix: j - 1 + depth,
                    ids: &[token],
                    hidden: &recurrent,
                    control: Control::None,
                    detailed: true,
                },
            )?;
            let prefix = format!("{key}.step{}", depth + 1);
            let expected = read(&format!("{prefix}.logits"), VOCAB)?;
            let result = score(
                &values(&dump, "logits")?,
                &expected,
                near_tie_epsilon(&root, depth + 1)?,
            )?;
            probe.save(
                &dump,
                &format!("recursive-{prefix}"),
                p,
                j - 1 + depth,
                "reference tokens; own recurrent hidden",
            )?;
            let mut diagnostic = serde_json::Map::new();
            for (actual, key) in [
                ("fc", "fc_last"),
                ("pre_mtp_norm", "hid_pre"),
                ("post_mtp_norm", "hid_post"),
            ] {
                let a = values(&dump, actual)?;
                let r = read(&format!("{prefix}.{key}"), HIDDEN)?;
                let max_abs = a
                    .iter()
                    .zip(&r)
                    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                    .fold(0.0f32, f32::max);
                diagnostic.insert(actual.into(),serde_json::json!({"max_abs":max_abs,"numeric_opref":"informational; whole-head gate governs"}));
            }
            std::fs::write(
                probe.output.join(format!("recursive-{prefix}-score.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"head":result,"hidden":diagnostic,"mode":"same reference token path, own H5 recurrence"}),
                )?,
            )?;
            ensure!(
                result["defects"]
                    .as_array()
                    .context("score defects missing")?
                    .is_empty(),
                "recursive head DEFECT {prefix}"
            );
            recurrent = values(
                &dump,
                match h5 {
                    MtpRecurrentTap::PreMtpNorm => "pre_mtp_norm",
                    MtpRecurrentTap::PostMtpNorm => "post_mtp_norm",
                },
            )?;
        }
    }
    Ok(())
}

#[test]
fn mtp_dump_retains_bits_and_rejects_nonfinite_logits() -> Result<()> {
    let mut dump = Dump::default();
    let values = vec![bf16::from_bits(0x8000), bf16::ZERO, bf16::from_f32(1.0)];
    dump.bf16("logits", vec![1, 3], values)?;
    assert_eq!(dump.tensors["logits"].2, [0, 128, 0, 0, 128, 63]);
    for bits in [0x7fc0, 0x7f80, 0xff80] {
        let error = Dump::default()
            .bf16("logits", vec![1, 1], vec![bf16::from_bits(bits)])
            .expect_err("nonfinite logits must retain the existing error");
        assert_eq!(error.to_string(), "nonfinite MTP dump logits");
    }
    Ok(())
}

#[test]
#[ignore = "authorized GPU window; argmax ties, finite flag, sticky/reset and tail controls"]
fn q38_mtp_host_trim_primitives_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_MTP_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = Device::new(0)?;
    let ops = Ops::new(&device.ctx)?;
    let s = &device.stream;
    let mut results = Vec::new();
    for count in [1usize, 7, 8, 31, 1025, VOCAB, 4 * VOCAB] {
        for offset in [0usize, 1] {
            for case in ["ties", "zeros", "negative", "nan", "posinf", "neginf"] {
                let mut host = vec![bf16::from_f32(-2.0); count + offset + 1];
                let values = &mut host[offset..offset + count];
                match case {
                    "ties" => {
                        values[count / 3] = bf16::from_f32(4.0);
                        values[count - 1] = bf16::from_f32(4.0);
                    }
                    "zeros" => {
                        values.fill(bf16::from_bits(0x8000));
                        values[count - 1] = bf16::ZERO;
                    }
                    "negative" => values[count - 1] = bf16::from_f32(-1.0),
                    "nan" => values[count - 1] = bf16::from_bits(0x7fc0),
                    "posinf" => values[count - 1] = bf16::from_bits(0x7f80),
                    "neginf" => values[count - 1] = bf16::from_bits(0xff80),
                    _ => unreachable!(),
                }
                let finite = values.iter().all(|x| x.is_finite());
                let mut expected = (0u32, f32::NEG_INFINITY);
                for (i, value) in values.iter().enumerate() {
                    if value.to_f32() > expected.1 {
                        expected = (i as u32, value.to_f32());
                    }
                }
                host[offset + count] = bf16::from_bits(0x7fc0);
                let input = s.clone_htod(&host)?;
                let view = input.slice(offset..offset + count);
                let mut token = s.alloc_zeros::<u32>(1)?;
                let mut flag = s.alloc_zeros::<u32>(1)?;
                let scan = |flag: &mut CudaSlice<u32>| -> Result<()> {
                    unsafe {
                        s.launch_builder(&ops.finite_bf16)
                            .arg(&view)
                            .arg(flag)
                            .arg(&(count as u64))
                            .launch(flat(count))?;
                    }
                    Ok(())
                };
                unsafe {
                    s.launch_builder(&ops.argmax)
                        .arg(&view)
                        .arg(&mut token)
                        .arg(&(count as i32))
                        .launch(crate::backend::cuda::ops::grid(1, 1024))?;
                }
                scan(&mut flag)?;
                let id = s.clone_dtoh(&token)?[0];
                let bad = s.clone_dtoh(&flag)?[0];
                s.memcpy_htod(&[1u32], &mut flag)?;
                scan(&mut flag)?;
                let sticky = s.clone_dtoh(&flag)?[0];
                s.memset_zeros(&mut flag)?;
                scan(&mut flag)?;
                let reset = s.clone_dtoh(&flag)?[0];
                results.push(serde_json::json!({"count":count,"offset":offset,"case":case,"id":id,"expected":expected.0,"bad":bad,"finite":finite,"sticky":sticky,"reset":reset}));
                if let Some(out) = std::env::var_os("FF_Q38_MTP_OUTPUT") {
                    let out = std::path::PathBuf::from(out);
                    std::fs::create_dir_all(&out)?;
                    std::fs::write(
                        out.join("primitive-observations.json"),
                        serde_json::to_vec_pretty(&results)?,
                    )?;
                }
                ensure!(
                    bad == u32::from(!finite) && reset == bad && sticky == 1,
                    "finite/sticky/reset/tail mismatch"
                );
                ensure!(!finite || id == expected.0, "lowest-index argmax mismatch");
            }
        }
    }
    Ok(())
}
