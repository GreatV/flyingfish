use super::super::tests::{
    committed_bits, committed_difference, configure, load, options, output, prompt, restore, save,
    state_bits,
};
use super::*;
use crate::mtp::batched::BatchedBackend;
use std::path::PathBuf;

#[test]
fn eager_setup_creates_no_verify_graphs_or_capture_work() -> Result<()> {
    let graphs = optional_verify_graphs::<()>(false, || {
        anyhow::bail!("capture is unavailable; eager setup must never call this transaction")
    })?;
    assert!(graphs.is_empty());
    Ok(())
}

#[test]
fn graph_setup_preserves_capture_results_and_errors() -> Result<()> {
    let mut calls = 0;
    let graphs = optional_verify_graphs(true, || {
        calls += 1;
        Ok(vec![1, 2, 3, 4])
    })?;
    assert_eq!(graphs, [1, 2, 3, 4]);
    assert_eq!(calls, 1);
    let error = optional_verify_graphs::<()>(true, || anyhow::bail!("capture failed"))
        .expect_err("graph setup must propagate capture failure");
    assert_eq!(error.to_string(), "capture failed");
    Ok(())
}

pub(super) fn staged() -> Result<Engine> {
    let mut e = load(true)?;
    e.mtp_enable(options(
        3,
        MtpHiddenTap::PreFinalNorm,
        MtpRecurrentTap::PostMtpNorm,
        true,
    ))?;
    Ok(e)
}
pub(super) fn ready() -> Result<Engine> {
    let mut e = staged()?;
    e.mtp_batch_enable()?;
    e.capture()?;
    e.closure.ready(std::time::Instant::now())?;
    Ok(e)
}
fn call<T>(e: &mut Engine, f: impl FnOnce(&mut Bound<'_>) -> Result<T>) -> Result<T> {
    let mut c = e.mtp.take().context("MTP missing")?;
    let r = f(&mut Bound {
        engine: e,
        chain: &mut c,
    });
    e.mtp = Some(c);
    r
}
fn counts(e: &Engine) -> Result<(usize, usize, usize)> {
    let b = e
        .mtp
        .as_ref()
        .context("MTP missing")?
        .batch
        .as_ref()
        .context("batch missing")?;
    Ok((b.calls, b.target_restores, b.mtp_restores))
}
// Causal teacher forcing at exactly K+1 rows constructs all-accept proposals.
// Future placeholders cannot influence the row being chosen. Every trial is
// rolled back; no recurrent-decode oracle is used for batch state equality.
fn proposals(e: &mut Engine, k: usize) -> Result<Vec<u32>> {
    let anchor = e.token()?;
    let mut inputs = vec![1; k + 1];
    inputs[0] = anchor;
    for depth in 0..k {
        let predictions = call(e, |b| b.verify_batch(&inputs))?;
        inputs[depth + 1] = predictions[depth];
        restore(e)?;
    }
    let predictions = call(e, |b| b.verify_batch(&inputs))?;
    restore(e)?;
    ensure!(
        inputs[1..] == predictions[..k],
        "causal batch fixture is inconsistent"
    );
    ensure!(
        !inputs.iter().any(|t| e.model_spec.eos.contains(t)),
        "fixture reaches EOS too early"
    );
    Ok(inputs[1..].to_vec())
}
// This oracle materializes the chosen GPU slot without the production selector.
// Primitive gates independently check the slot contents against row recurrence.
fn canonical(e: &mut Engine, inputs: &[u32], rows: usize) -> Result<Vec<u32>> {
    ensure!(
        !e.mtp.as_ref().context("MTP missing")?.options.graph,
        "oracle must run eagerly"
    );
    call(e, |b| {
        let start = b.engine.position;
        let hidden = b.chain.last_target.clone();
        let predictions = b.verify_batch(inputs)?;
        let batch = b.chain.batch.as_ref().context("batch missing")?;
        let frame = b
            .engine
            .device
            .stream
            .clone_dtoh(&batch.frame.slice(..rows * HIDDEN))?;
        let s = &b.engine.device.stream;
        for (state, snapshot) in b.engine.state.gdn.iter_mut().zip(&batch.snapshots) {
            for (target, source) in [
                (&mut state.s, &snapshot.s),
                (&mut state.history, &snapshot.history),
            ] {
                let n = target.len();
                let values = s.clone_dtoh(&source.slice((rows - 1) * n..rows * n))?;
                s.memcpy_htod(&values, target)?;
            }
        }
        s.memcpy_dtod(
            &batch.logits.slice((rows - 1) * VOCAB..rows * VOCAB),
            &mut b.engine.work.logits,
        )?;
        s.memcpy_htod(&[predictions[rows - 1]], &mut b.engine.work.token)?;
        let position = start + rows;
        s.memcpy_htod(&[position as i32], &mut b.engine.work.position)?;
        s.memcpy_htod(&[(position + 1) as i32], &mut b.engine.work.length)?;
        b.engine
            .state
            .set_sequence(s, position as u32, 1, b.engine.workspace_rows as u32)?;
        b.engine.position = position;
        b.chain.cursor = b.chain.saved.cursor;
        b.chain.mtp_checkpoint.restore(
            b.engine.mtp_state.as_mut().context("MTP state missing")?,
            &mut b.chain.runtime.work.token,
            s,
        )?;
        for (row, &token) in inputs[..rows].iter().enumerate() {
            let h = if row == 0 {
                matrix(&hidden, &b.engine.device.stream, 0, 1, HIDDEN)?
            } else {
                let buffer = b
                    .engine
                    .device
                    .stream
                    .clone_htod(&frame[(row - 1) * HIDDEN..row * HIDDEN])?;
                b.row(
                    token,
                    matrix(&buffer, &b.engine.device.stream, 0, 1, HIDDEN)?,
                    OutputMode::Predict,
                )?;
                continue;
            };
            b.row(token, h, OutputMode::Predict)?;
        }
        b.engine.device.stream.memcpy_htod(
            &frame[(rows - 1) * HIDDEN..rows * HIDDEN],
            &mut b.chain.last_target,
        )?;
        b.chain.pending = predictions[rows - 1];
        b.chain.committed_position = b.engine.position;
        Ok(predictions[..rows].to_vec())
    })
}
fn receipt(name: &str, value: serde_json::Value) -> Result<()> {
    std::fs::write(
        output()?.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&value)?,
    )?;
    Ok(())
}

#[test]
#[ignore = "authorized full-model GPU window required"]
fn q38_mtp_batch_transaction_gpu() -> Result<()> {
    let mut e = ready()?;
    let ids = prompt("math1")?;
    let mut cases = 0;
    for k in 1..=3 {
        for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
            for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                configure(&mut e, options(k, h2, h5, false))?;
                e.mtp_prefill(&ids)?;
                save(&mut e)?;
                let full = proposals(&mut e, k)?;
                let anchor = e.token()?;
                for j in 0..=k {
                    let mut forced = full.clone();
                    if j < k {
                        forced[j] = (forced[j] + 1) % VOCAB as u32;
                    }
                    let inputs: Vec<_> = std::iter::once(anchor)
                        .chain(forced.iter().copied())
                        .collect();
                    // Eager canonical is also the graph parity oracle.
                    e.mtp.as_mut().context("MTP missing")?.options.graph = false;
                    let expected = canonical(&mut e, &inputs, j + 1)?;
                    let bits = committed_bits(&e)?;
                    restore(&mut e)?;
                    for graph in [false, true] {
                        let c = e.mtp.as_mut().context("MTP missing")?;
                        c.options.graph = graph;
                        c.force_proposals = Some(forced.clone());
                        let head_runs = c.runtime.head_runs;
                        let before = counts(&e)?;
                        let r = e.mtp_batch_round(64, None)?;
                        let after = counts(&e)?;
                        ensure!(
                            e.mtp.as_ref().context("MTP missing")?.runtime.head_runs - head_runs
                                == k,
                            "canonical repair evaluated an unused MTP head"
                        );
                        let mut tokens = full[..j].to_vec();
                        tokens.push(*expected.last().context("empty canonical head")?);
                        ensure!(
                            r.round.accepted == j
                                && r.round.tokens == tokens
                                && r.kept_verify_state == (j == k),
                            "M2b commit/pending mismatch"
                        );
                        ensure!(
                            (after.0 - before.0, after.1 - before.1, after.2 - before.2)
                                == (1, 0, 1),
                            "M2b full/partial path did extra target work"
                        );
                        let actual = committed_bits(&e)?;
                        if actual != bits {
                            let difference = committed_difference(&e, &bits, &actual)?;
                            receipt("m2b-transaction-difference", difference.clone())?;
                            anyhow::bail!(
                                "M2b state bits mismatch K{k} j{j} {h2:?}/{h5:?} graph{graph}: {difference}"
                            );
                        }
                        restore(&mut e)?;
                        cases += 1;
                    }
                }
            }
        }
    }
    // Metadata tail, K_eff=0 recurrent fallback, budget zero and EOS pending.
    configure(
        &mut e,
        options(
            3,
            MtpHiddenTap::PreFinalNorm,
            MtpRecurrentTap::PostMtpNorm,
            true,
        ),
    )?;
    let mut tail = ids.clone();
    tail.resize(e.capacity - 2, 1);
    e.mtp_prefill(&tail)?;
    save(&mut e)?;
    let expected = super::super::tests::ordinary(&mut e, 1)?;
    let bits = state_bits(&e)?;
    restore(&mut e)?;
    ensure!(
        e.mtp_batch_round(64, None)?.round.tokens == expected && state_bits(&e)? == bits,
        "M2b tail fallback differs"
    );
    ensure!(
        e.mtp_batch_round(64, None)?.round.tokens.is_empty(),
        "M2b exceeded capacity"
    );
    e.mtp_prefill(&ids)?;
    let bits = state_bits(&e)?;
    ensure!(
        e.mtp_batch_round(0, None)?.round.tokens.is_empty() && state_bits(&e)? == bits,
        "zero budget changed state"
    );
    let pending = e.token()?;
    e.model_spec.eos.push(pending);
    ensure!(
        e.mtp_batch_round(64, None)?.round.tokens.is_empty() && state_bits(&e)? == bits,
        "EOS pending was consumed"
    );
    e.model_spec.eos.pop();
    // A verified EOS proposal is left pending: only the anchor is consumed.
    save(&mut e)?;
    let full = proposals(&mut e, 3)?;
    e.model_spec.eos.push(full[0]);
    e.mtp.as_mut().context("MTP missing")?.force_proposals = Some(full);
    let r = e.mtp_batch_round(64, None)?;
    ensure!(
        r.round.accepted == 0 && r.round.consumed == 1 && !r.kept_verify_state,
        "EOS proposal was consumed"
    );
    e.model_spec.eos.pop();
    e.mtp_prefill(&ids)?;
    e.reset()?;
    e.prefill(&ids, None)?;
    let before = state_bits(&e)?;
    ensure!(
        e.mtp_batch_round(1, None).is_err() && state_bits(&e)? == before,
        "stale epoch accepted or mutated"
    );
    receipt(
        "m2b-transaction",
        serde_json::json!({"result":"PASS","cases":cases,"state_oracle":"independent slot materialization; all target/MTP states and physical KV tail","paths":"all:1batch+0target_restore+1mtp_restore","tail_eos_epoch":"PASS"}),
    )
}

#[test]
#[ignore = "authorized full-model GPU window required"]
fn q38_mtp_batch_failure_capture_gpu() -> Result<()> {
    let mut e = staged()?;
    let before = state_bits(&e)?;
    for occurrence in [1, 2, 3] {
        e.capture_fail_after_layer = Some((0, occurrence));
        ensure!(e.mtp_batch_enable().is_err(), "batch capture fault ignored");
        ensure!(
            e.poisoned && e.mtp.as_ref().unwrap().batch.is_none() && state_bits(&e)? == before,
            "batch capture rollback/publication failure"
        );
        e.capture_fail_after_layer = None;
        e.poisoned = false;
    }
    use crate::backend::cuda::engine::CaptureFault;
    for fault in [
        CaptureFault::Begin,
        CaptureFault::End,
        CaptureFault::Instantiate,
    ] {
        e.capture_fail_stage = Some(fault);
        ensure!(e.mtp_batch_enable().is_err(), "batch stage fault ignored");
        ensure!(
            e.poisoned && e.mtp.as_ref().unwrap().batch.is_none() && state_bits(&e)? == before,
            "batch stage rollback/publication failure"
        );
        e.capture_fail_stage = None;
        e.poisoned = false;
    }
    e.mtp_batch_enable()?;
    ensure!(
        state_bits(&e)? == before,
        "successful batch capture mutated committed state"
    );
    e.capture()?;
    e.closure.ready(std::time::Instant::now())?;
    e.mtp_prefill(&prompt("math1")?)?;
    save(&mut e)?;
    let full = proposals(&mut e, 3)?;
    // Prove the state oracle detects a missed GDN S/history restore and a
    // stale MTP cursor, even when the target token has not changed.
    let oracle = state_bits(&e)?;
    for history in [false, true] {
        let s = e.device.stream.clone();
        let g = &mut e.state.gdn[0];
        let data = if history { &mut g.history } else { &mut g.s };
        let value = s.clone_dtoh(&data.slice(..1))?[0];
        s.memcpy_htod(
            &[f32::from_bits(value.to_bits() ^ 1)],
            &mut data.slice_mut(..1),
        )?;
        ensure!(
            state_bits(&e)? != oracle,
            "GDN restore negative control escaped"
        );
        restore(&mut e)?;
        ensure!(
            state_bits(&e)? == oracle,
            "negative control failed to restore"
        );
    }
    e.mtp.as_mut().unwrap().cursor += 1;
    ensure!(state_bits(&e)? != oracle, "MTP cursor control escaped");
    restore(&mut e)?;
    let mut faults = 0;
    for graph in [false, true] {
        for partial in [false, true] {
            let mut forced = full.clone();
            if partial {
                forced[1] = (forced[1] + 1) % VOCAB as u32;
            }
            let c = e.mtp.as_mut().context("MTP missing")?;
            c.options.graph = graph;
            c.force_proposals = Some(forced);
            let before = state_bits(&e)?;
            let mut points = vec![
                policy::Fault::AfterVerify,
                policy::Fault::AfterRepair,
                policy::Fault::BeforePublish,
            ];
            points.extend((0..3).map(policy::Fault::AfterDraft));
            if partial {
                points.extend([
                    policy::Fault::AfterPrefixSelect,
                    policy::Fault::AfterMtpRestore,
                ]);
            } else {
                points.push(policy::Fault::AfterMtpRestore);
            }
            for fault in points {
                ensure!(
                    e.mtp_batch_round(64, Some(fault)).is_err(),
                    "batch round fault ignored {fault:?}"
                );
                ensure!(
                    e.poisoned && state_bits(&e)? == before,
                    "batch rollback mismatch {fault:?}"
                );
                e.poisoned = false;
                faults += 1;
            }
            for row in 0..if partial { 2 } else { 4 } {
                e.mtp
                    .as_mut()
                    .unwrap()
                    .batch
                    .as_mut()
                    .unwrap()
                    .repair_fail_after = Some(row);
                ensure!(
                    e.mtp_batch_round(64, None).is_err(),
                    "canonical row fault ignored"
                );
                ensure!(
                    e.poisoned && state_bits(&e)? == before,
                    "canonical row rollback mismatch"
                );
                e.mtp
                    .as_mut()
                    .unwrap()
                    .batch
                    .as_mut()
                    .unwrap()
                    .repair_fail_after = None;
                e.poisoned = false;
                faults += 1;
            }
            let r = e.mtp_batch_round(64, None)?;
            let bits = state_bits(&e)?;
            restore(&mut e)?;
            let again = e.mtp_batch_round(64, None)?;
            ensure!(
                r == again && state_bits(&e)? == bits,
                "batch retry changed published tokens/state"
            );
            restore(&mut e)?;
        }
    }
    receipt(
        "m2b-failure-capture",
        serde_json::json!({"result":"PASS","capture_boundaries":6,"round_faults":faults,"state_negative_controls":3,"poison":"required after every failed round; test-only manual clear for further injected faults"}),
    )
}

#[test]
#[ignore = "requires a full-model GPU window, FF_Q38_MODEL and FLYINGFISH_RUNTIME_DIR"]
fn q38_snapshot_poison_rows_gpu() -> Result<()> {
    // Read contract for the verify GDN snapshot rows: select_prefix reads exactly one row
    // per layer (snapshot_row(committed) = committed - 1). Every other row is poisoned
    // between verify and restore; the restored GDN state must match the canonical oracle.
    let mut e = ready()?;
    let ids = prompt("math1")?;
    let garbage_s = vec![1.0e30f32; 48 * 128 * 128];
    let garbage_h = vec![1.0e30f32; 10240 * 3];
    for k in 1..=3 {
        configure(
            &mut e,
            options(
                k,
                MtpHiddenTap::PreFinalNorm,
                MtpRecurrentTap::PreMtpNorm,
                false,
            ),
        )?;
        e.mtp_prefill(&ids)?;
        save(&mut e)?;
        let full = proposals(&mut e, k)?;
        let anchor = e.token()?;
        for j in 0..=k {
            let mut forced = full.clone();
            if j < k {
                forced[j] = (forced[j] + 1) % VOCAB as u32;
            }
            let inputs: Vec<_> = std::iter::once(anchor)
                .chain(forced.iter().copied())
                .collect();
            let _ = canonical(&mut e, &inputs, j + 1)?;
            let expected = gdn_bits(&e)?;
            restore(&mut e)?;
            let _ = call(&mut e, |b| b.verify_batch(&inputs))?;
            // j == k keeps the verify state and never selects; poison every row.
            let keep_row = (j < k).then_some(j);
            poison_snapshot_rows(&mut e, keep_row, &garbage_s, &garbage_h)?;
            if j < k {
                call(&mut e, |b| b.select_prefix(j + 1))?;
            }
            let actual = gdn_bits(&e)?;
            ensure!(
                actual == expected,
                "poisoned snapshot rows changed the GDN state K{k} j{j}"
            );
            restore(&mut e)?;
        }
        restore(&mut e)?;
    }
    Ok(())
}

fn gdn_bits(e: &Engine) -> Result<Vec<u8>> {
    let s = &e.device.stream;
    let mut bits = Vec::new();
    for g in &e.state.gdn {
        for buf in [&g.s, &g.history] {
            for v in s.clone_dtoh(buf)? {
                bits.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    Ok(bits)
}

fn poison_snapshot_rows(
    e: &mut Engine,
    keep_row: Option<usize>,
    garbage_s: &[f32],
    garbage_h: &[f32],
) -> Result<()> {
    let s = e.device.stream.clone();
    let c = e.mtp.as_mut().context("MTP missing")?;
    let batch = c.batch.as_mut().context("M2b not bound")?;
    for snap in &mut batch.snapshots {
        for r in 0..4 {
            if Some(r) == keep_row {
                continue;
            }
            s.memcpy_htod(
                garbage_s,
                &mut snap
                    .s
                    .slice_mut(r * garbage_s.len()..(r + 1) * garbage_s.len()),
            )?;
            s.memcpy_htod(
                garbage_h,
                &mut snap
                    .history
                    .slice_mut(r * garbage_h.len()..(r + 1) * garbage_h.len()),
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "authorized full-model GPU window required; DRef scoring follows export"]
fn q38_mtp_batch_chain_gpu() -> Result<()> {
    let mut e = ready()?;
    let out = output()?;
    let mut cases = Vec::new();
    for k in 1..=3 {
        for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
            for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                for case in ["math1", "code1", "gen1", "gen2", "math2"] {
                    configure(&mut e, options(k, h2, h5, true))?;
                    let ids = prompt(case)?;
                    let first = e.prefill(&ids, None)?;
                    let mut plain = vec![first];
                    while plain.len() < 64 && !e.model_spec.eos.contains(plain.last().unwrap()) {
                        plain.push(e.decode(true, None)?);
                    }
                    let mut graph_tokens = None;
                    for graph in [false, true] {
                        e.mtp.as_mut().context("MTP missing")?.options.graph = graph;
                        ensure!(
                            e.mtp_prefill(&ids)? == first,
                            "batch bootstrap changed prefill"
                        );
                        let mut tokens = vec![first];
                        let mut rounds = Vec::new();
                        while tokens.len() < 64
                            && !e.model_spec.eos.contains(tokens.last().unwrap())
                        {
                            let r = e.mtp_batch_round(64 - tokens.len(), None)?;
                            ensure!(!r.round.tokens.is_empty(), "batch stopped before budget");
                            rounds.push(serde_json::json!({"matched":r.round.matched,"accepted":r.round.accepted,"consumed":r.round.consumed,"proposals":r.round.proposals,"tokens":r.round.tokens,"kept_verify_state":r.kept_verify_state,"verify_predictions":r.verify_predictions}));
                            tokens.extend_from_slice(&r.round.tokens);
                        }
                        if let Some(expected) = &graph_tokens {
                            ensure!(&tokens == expected, "M2b eager/graph trajectory differs");
                        } else {
                            graph_tokens = Some(tokens.clone());
                        }
                        let divergence = tokens.iter().zip(&plain).filter(|(a, b)| a != b).count()
                            + tokens.len().abs_diff(plain.len());
                        let name = format!("{case}-K{k}-{h2:?}-{h5:?}-graph{graph}");
                        std::fs::write(
                            out.join(format!("{name}.json")),
                            serde_json::to_vec_pretty(
                                &serde_json::json!({"case":case,"depth":k,"h2":h2,"h5":h5,"graph":graph,"prompt_ids":ids,"tokens":tokens,"plain_tokens":plain,"batch_vs_decode_divergences":divergence,"rounds":rounds,"numeric_gate":"PENDING_DREF_G7; exporter rc0 is not numeric PASS"}),
                            )?,
                        )?;
                        cases.push(name);
                    }
                }
            }
        }
    }
    receipt(
        "m2b-chain-export",
        serde_json::json!({"result":"EXPORTED_PENDING_DREF","cases":cases,"eager_graph_tokens":"BIT_EQUAL","rule":"DRef G7 ARGMAX/NEAR-TIE/DEFECT T=1.0; zero DEFECT required"}),
    )
}

#[test]
#[ignore = "authorized review GPU window; frozen K2 bootstrap acceptance regression"]
fn q38_mtp_bootstrap_acceptance_gpu() -> Result<()> {
    let mut e = super::super::tests::load_capacity(true, 512)?;
    let selected = options(
        2,
        MtpHiddenTap::PostFinalNorm,
        MtpRecurrentTap::PostMtpNorm,
        true,
    );
    e.mtp_enable(selected)?;
    e.mtp_batch_enable()?;
    e.capture()?;
    let inputs = ["math1", "code1", "gen1", "gen2", "math2"]
        .into_iter()
        .map(|name| Ok((name, prompt(name)?)))
        .collect::<Result<Vec<_>>>()?;
    for (_, ids) in &inputs {
        for chunk in ids.chunks(e.chunk) {
            e.prepare(chunk.len())?;
        }
    }
    e.closure.ready(std::time::Instant::now())?;
    e.model_spec.eos.clear();
    let (mut published, mut rounds) = (0usize, 0usize);
    let mut cases = Vec::new();
    for (name, ids) in &inputs {
        for iteration in 0..5 {
            let mut tokens = vec![e.mtp_prefill(ids)?];
            let (mut emitted, mut count) = (0usize, 0usize);
            while emitted < 32 {
                let r = e.mtp_batch_round(32 - emitted, None)?.round;
                ensure!(!r.tokens.is_empty(), "bootstrap warmup stopped early");
                emitted += r.tokens.len();
                tokens.extend(r.tokens);
            }
            emitted = 0;
            while emitted < 200 {
                let r = e.mtp_batch_round(200 - emitted, None)?.round;
                ensure!(!r.tokens.is_empty(), "bootstrap measurement stopped early");
                emitted += r.tokens.len();
                count += 1;
                tokens.extend(r.tokens);
            }
            ensure!(
                emitted == 200 && tokens.len() == 233,
                "bootstrap output budget changed"
            );
            published += emitted;
            rounds += count;
            cases.push(serde_json::json!({"case":name,"iteration":iteration,"tokens":tokens,"measured_tokens":emitted,"rounds":count}));
        }
    }
    let reference = 5000.0 / 1920.0;
    let actual = published as f64 / rounds as f64;
    let pass = (reference * 0.97..=reference * 1.03).contains(&actual);
    receipt(
        "bootstrap-acceptance",
        serde_json::json!({
            "result":if pass {"PASS"} else {"FAIL"},"depth":2,"warmup_tokens":32,
            "reference_tokens":5000,"reference_rounds":1920,"relative_limit":0.03,
            "published":published,"rounds":rounds,"mean_tokens_per_round":actual,"cases":cases
        }),
    )?;
    ensure!(
        pass,
        "K2 bootstrap acceptance {actual} is outside +/-3% of {reference}"
    );
    Ok(())
}

#[test]
#[ignore = "authorized full-model GPU window required; score exported FP32 logits against frozen DRef"]
fn q38_mtp_snapshot_teacher_forcing_gpu() -> Result<()> {
    use std::io::Write;
    let root =
        PathBuf::from(std::env::var_os("FF_Q38_MTP_REF_ROOT").context("reference root missing")?);
    let paths: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(root.join("PATHS.json"))?)?;
    ensure!(paths.len() == 5, "five DREF16K paths required");
    let model = PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("model missing")?);
    let runtime =
        PathBuf::from(std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?);
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let options = crate::spec::Options {
        device: 0,
        capacity: 16640,
        chunk: Some(2048),
        count: 256,
        ignore_eos: true,
        backend: crate::backend::Settings {
            runtime_dir: Some(runtime),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            tree_builder: crate::backend::TreeBuilder::Waves,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
        },
    };
    let mut e = Engine::load_mtp(&model, &options, 3, true)?;
    let out = output()?;
    let job: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("MEASURE.json"))?)?;
    let smoke = job["count"] == 16;
    let length = if smoke { 12 } else { 64 };
    let mut completed = Vec::new();
    for path in paths.into_iter().take(if smoke { 1 } else { 5 }) {
        let name = path["id"].as_str().context("path id missing")?;
        let input: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join(path["input_file"].as_str().context("input file missing")?),
        )?)?;
        let mut ids: Vec<u32> = serde_json::from_value(input["input_ids"].clone())?;
        let history: Vec<u32> = serde_json::from_value(path["tokens"].clone())?;
        ensure!(
            ids.len() == 16320 && history.len() == 64,
            "DREF16K extent mismatch"
        );
        if smoke {
            ids.truncate(32);
        }
        e.mtp_prefill(&ids)?;
        let mut predictions = vec![e.token()?];
        let mut logits =
            std::io::BufWriter::new(std::fs::File::create(out.join(format!("{name}.f32le")))?);
        for value in e.decode_logits()? {
            ensure!(value.is_finite(), "nonfinite prefill logits");
            logits.write_all(&value.to_le_bytes())?;
        }
        let mut offset = 0;
        let mut round = 0;
        let mut branches = [0usize; 4];
        while offset < length - 1 {
            let rows = (round % 4 + 1).min(length - 1 - offset);
            let inputs: Vec<_> = (0..4).map(|i| history[(offset + i).min(63)]).collect();
            let (next, values) = call(&mut e, |b| {
                b.save(4)?;
                let mut draft = b.pending()?;
                for depth in 0..3 {
                    draft = b.draft(draft, depth)?;
                }
                let full = b.verify_batch(&inputs)?;
                let batch = b.chain.batch.as_ref().context("batch missing")?;
                let values = b
                    .engine
                    .device
                    .stream
                    .clone_dtoh(&batch.logits.slice(..rows * VOCAB))?;
                if rows < 4 {
                    b.select_prefix(rows)?;
                }
                b.restore_mtp()?;
                b.repair_mtp(&inputs[..rows])?;
                b.chain.committed_position = b.engine.position;
                b.chain
                    .batch
                    .as_mut()
                    .context("batch missing")?
                    .invalidate_frame();
                Ok((full[..rows].to_vec(), values))
            })?;
            ensure!(
                values.len() == rows * VOCAB && values.iter().all(|v| v.is_finite()),
                "nonfinite teacher-forced verify logits"
            );
            for value in values {
                logits.write_all(&value.to_f32().to_le_bytes())?;
            }
            predictions.extend(next);
            offset += rows;
            branches[rows - 1] += 1;
            round += 1;
            ensure!(
                e.position == ids.len() + offset
                    && e.mtp.as_ref().context("MTP missing")?.cursor + 1 == e.position,
                "teacher-forced commit metadata mismatch"
            );
        }
        logits.flush()?;
        e.synchronize()?;
        ensure!(
            predictions.len() == length && branches.iter().all(|n| *n > 0),
            "branch coverage incomplete"
        );
        let receipt = serde_json::json!({"path":path,"predictions":predictions,"branches":branches,"logits_shape":[length,VOCAB],"stored_dtype":"FP32 little-endian, widened from native BF16","first_absolute_logit_row":ids.len()-1,"teacher_forcing":"prefill row then forced history, cyclic committed rows 1/2/3/4; full verify with raw snapshots and canonical MTP repair","candidate_DEFECT":"score only"});
        std::fs::write(
            out.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&receipt)?,
        )?;
        completed.push(receipt);
    }
    receipt(
        "snapshot-teacher-forcing",
        serde_json::json!({"validity":"PASS","paths":completed,"positions":if smoke {12} else {320},"smoke":smoke}),
    )
}

#[test]
#[ignore = "authorized full-model GPU window required; frozen MEASURE.json"]
fn q38_mtp_snapshot_measure_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root = PathBuf::from(std::env::var_os("FF_Q38_MTP_REF_ROOT").context("root missing")?);
    let job: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("MEASURE.json"))?)?;
    let capacity = usize::try_from(job["capacity"].as_u64().context("capacity missing")?)?;
    let count = usize::try_from(job["count"].as_u64().context("count missing")?)?;
    let samples = usize::try_from(job["samples"].as_u64().context("samples missing")?)?;
    ensure!(
        [4096, 131072].contains(&capacity)
            && [16, 256].contains(&count)
            && (1..=4).contains(&samples),
        "measurement job outside protocol"
    );
    let model = PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("model missing")?);
    let runtime =
        PathBuf::from(std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?);
    let chunk = if job["auto_chunk"].as_bool() == Some(true) {
        None
    } else {
        Some(2048)
    };
    let options = crate::spec::Options {
        device: 0,
        capacity,
        chunk,
        count,
        ignore_eos: true,
        backend: crate::backend::Settings {
            runtime_dir: Some(runtime),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            tree_builder: crate::backend::TreeBuilder::Waves,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
        },
    };
    let mut e = Engine::load_mtp(&model, &options, 3, true)?;
    let (free, total) = e.device.ctx.mem_get_info()?;
    ensure!(
        free >= total / 16,
        "Ready allocations consumed the unchanged reserve"
    );
    receipt(
        "snapshot-admission",
        serde_json::json!({"validity":"PASS","capacity":capacity,"requested_chunk":chunk,"chunk":e.chunk,"reserve_divisor":16,"total_bytes":total,"free_bytes":free}),
    )?;
    let ids = prompt("math1")?;
    let mut records = Vec::new();
    for sample in 0..=samples {
        let mut rounds = 0;
        let mut accepted = 0;
        let mut publications = 0;
        let result = e.generate_mtp_prepared(
            &ids,
            count,
            true,
            crate::mtp::VerifyPolicy::Batched,
            true,
            &mut |_, a| {
                if publications > 0 {
                    rounds += 1;
                    accepted += a;
                }
                publications += 1;
                true
            },
        )?;
        ensure!(
            result.ids.len() == count && result.ids.iter().all(|t| (*t as usize) < VOCAB),
            "incomplete or invalid MTP3 generation"
        );
        ensure!(
            result.decode_ms.is_finite() && result.prefill_ms.is_finite() && result.decode_ms > 0.0,
            "nonfinite generation timing"
        );
        ensure!(
            e.decode_logits()?.iter().all(|v| v.is_finite()),
            "nonfinite final MTP3 head"
        );
        if sample > 0 {
            records.push(serde_json::json!({"decode_ms":result.decode_ms,"prefill_ms":result.prefill_ms,"ids":result.ids,"rounds":rounds,"accepted":accepted}));
        }
    }
    receipt(
        "snapshot-measure",
        serde_json::json!({"validity":"PASS","capacity":capacity,"chunk":e.chunk,"samples":records,"reserve_divisor":16}),
    )
}

#[test]
#[ignore = "authorized GPU window required; pins main calibration for candidate"]
fn q38_mtp_snapshot_calibrate_gpu() -> Result<()> {
    let mut e = super::super::tests::load_capacity(true, 131072)?;
    e.mtp_enable(options(
        3,
        MtpHiddenTap::PostFinalNorm,
        MtpRecurrentTap::PostMtpNorm,
        true,
    ))?;
    e.mtp_batch_enable()?;
    receipt(
        "snapshot-calibration",
        serde_json::json!({"validity":"PASS","capacity":131072,"linear_rows":"1..16; larger prefill index0","reserve_divisor":16,"plans":e.blas.bound_plan_receipt()?}),
    )
}

#[test]
#[ignore = "authorized full-model GPU window required"]
fn q38_compact_failed_round_then_plain_decode_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "compact rollback regression requires full-model window"
    );
    let mut ids = prompt("math1")?;
    ids.resize(2242, 1);
    {
        let mut e = super::super::tests::load_format(true, 2304, crate::kv::KvFormat::Affine8)?;
        let o = options(
            3,
            MtpHiddenTap::PreFinalNorm,
            MtpRecurrentTap::PostMtpNorm,
            false,
        );
        configure(&mut e, o)?;
        e.mtp_batch_enable()?;
        e.closure.ready(std::time::Instant::now())?;
        e.mtp_prefill(&ids)?;
        e.model_spec.eos.clear();
        e.check_logits()?;
        let before = e.decode_logits()?;
        let failure = e
            .mtp_batch_round(1, Some(policy::Fault::BeforePublish))
            .err()
            .context("single-row fault was not injected")?;
        ensure!(
            failure.to_string() == "injected MTP failure at BeforePublish",
            "fault transaction failed before injection: {failure:#}"
        );
        ensure!(
            !e.poisoned && e.position == ids.len(),
            "failed single-row rollback left unhealthy engine"
        );
        e.check_logits()?;
        ensure!(
            e.decode_logits()? == before,
            "failed round changed boundary logits"
        );
        for kv in &e.state.kv {
            let c = kv.compact.as_ref().context("compact cache missing")?;
            ensure!(
                e.device.stream.clone_dtoh(&c.cursor)?[2] == 0,
                "failed round retained seal lock"
            );
        }
        let mut compact_logits = Vec::new();
        let mut inputs = Vec::new();
        for _ in 0..16 {
            inputs.push(e.device.stream.clone_dtoh(&e.work.token)?[0]);
            e.decode(false, None)?;
            e.check_logits()?;
            compact_logits.push(e.decode_logits()?);
        }
        configure(&mut e, o)?;
        e.mtp_prefill(&ids)?;
        for (row, (&input, candidate)) in inputs.iter().zip(&compact_logits).enumerate() {
            e.device.stream.memcpy_htod(&[input], &mut e.work.token)?;
            e.decode(false, None)?;
            e.check_logits()?;
            let healthy = e.decode_logits()?;
            ensure!(
                candidate
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(healthy.iter().map(|x| x.to_bits())),
                "post-fault/healthy compact bits differ at row {row}"
            );
        }
        drop(e);
        let mut bf16 = super::super::tests::load_format(false, 2304, crate::kv::KvFormat::Bf16)?;
        bf16.prefill(&ids, None)?;
        for (row, (&input, candidate)) in inputs.iter().zip(&compact_logits).enumerate() {
            bf16.device
                .stream
                .memcpy_htod(&[input], &mut bf16.work.token)?;
            bf16.decode(false, None)?;
            let reference = bf16.decode_logits()?;
            ensure!(
                candidate.iter().chain(&reference).all(|x| x.is_finite()),
                "nonfinite post-fault logits row {row}"
            );
            let max_error = candidate
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "COMPACT_FAILED_ROUND_BF16_COMPARISON affine8 row={row} max_logit_error={max_error}"
            );
        }
    }
    println!(
        "COMPACT_FAILED_ROUND_PLAIN_PASS: affine8; injected k0 failure; 16 rows beyond wrap; healthy compact exact bits; BF16 per-row finite logit differences reported"
    );
    Ok(())
}
