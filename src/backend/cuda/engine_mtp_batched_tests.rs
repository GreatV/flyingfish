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
        Ok(vec![1, 2, 3, 4, 5])
    })?;
    assert_eq!(graphs, [1, 2, 3, 4, 5]);
    assert_eq!(calls, 1);
    let error = optional_verify_graphs::<()>(true, || anyhow::bail!("capture failed"))
        .expect_err("graph setup must propagate capture failure");
    assert_eq!(error.to_string(), "capture failed");
    Ok(())
}

pub(super) fn staged() -> Result<Engine> {
    let mut e = load(true)?;
    e.mtp_enable(options(
        4,
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
// Causal teacher forcing at exactly K+1 rows constructs all-accept proposals; every trial is rolled back.
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
// Materializes the chosen GPU slot without the production selector.
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
        if rows < inputs.len() {
            // On keep (rows == inputs.len()) the live verify state is the last row's state.
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
    for k in 1..=4 {
        for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
            for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                configure(&mut e, options(k, h2, h5, false))?;
                e.mtp_prefill(&ids)?;
                save(&mut e)
                    .with_context(|| format!("BF16 K{k} {h2:?}/{h5:?} five-row checkpoint save"))?;
                if k == 1 && h2 == MtpHiddenTap::PreFinalNorm && h5 == MtpRecurrentTap::PreMtpNorm {
                    println!("MTP_CHECKPOINT_CAPACITY_PASS selected_depth=1 saved_rows=5");
                }
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
    drop(e);
    let mut affine = super::super::tests::load_format(true, 128, crate::kv::KvFormat::Affine8)?;
    affine.mtp_enable(options(
        4,
        MtpHiddenTap::PreFinalNorm,
        MtpRecurrentTap::PostMtpNorm,
        true,
    ))?;
    affine.mtp_batch_enable()?;
    affine.capture()?;
    affine.closure.ready(std::time::Instant::now())?;
    affine.mtp_prefill(&ids)?;
    save(&mut affine)?;
    let full = proposals(&mut affine, 4)?;
    let anchor = affine.token()?;
    let mut affine_cases = 0;
    for accepted in 0..=4 {
        let mut forced = full.clone();
        if accepted < 4 {
            forced[accepted] = (forced[accepted] + 1) % VOCAB as u32;
        }
        let inputs: Vec<_> = std::iter::once(anchor)
            .chain(forced.iter().copied())
            .collect();
        affine.mtp.as_mut().context("MTP missing")?.options.graph = false;
        let expected = canonical(&mut affine, &inputs, accepted + 1)?;
        let expected_bits = committed_bits(&affine)?;
        restore(&mut affine)?;
        for graph in [false, true] {
            let chain = affine.mtp.as_mut().context("MTP missing")?;
            chain.options.graph = graph;
            chain.force_proposals = Some(forced.clone());
            let round = affine.mtp_batch_round(64, None)?;
            let mut tokens = full[..accepted].to_vec();
            tokens.push(*expected.last().context("empty affine8 canonical head")?);
            ensure!(
                round.round.accepted == accepted && round.round.tokens == tokens,
                "affine8 K4 commit/pending mismatch"
            );
            ensure!(
                committed_bits(&affine)? == expected_bits,
                "affine8 K4 canonical state mismatch accepted={accepted} graph={graph}"
            );
            restore(&mut affine)?;
            affine_cases += 1;
        }
    }
    println!("MTP_AFFINE8_K4_TRANSACTIONS_PASS cases={affine_cases} graph_and_eager");
    receipt(
        "m2b-transaction",
        serde_json::json!({"result":"PASS","cases":cases,"affine8_k4_cases":affine_cases,"state_oracle":"independent slot materialization; all target/MTP states and physical KV tail","paths":"all:1batch+0target_restore+1mtp_restore","tail_eos_epoch":"PASS"}),
    )
}

#[test]
#[ignore = "authorized full-model GPU window required"]
fn q38_mtp_batch_failure_capture_gpu() -> Result<()> {
    let mut e = staged()?;
    let before = state_bits(&e)?;
    for occurrence in 1..=5 {
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
    let full = proposals(&mut e, 4)?;
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
        for accepted in 0..=4 {
            let mut forced = full.clone();
            if accepted < 4 {
                forced[accepted] = (forced[accepted] + 1) % VOCAB as u32;
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
            points.extend((0..4).map(policy::Fault::AfterDraft));
            if accepted < 4 {
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
            for row in 0..accepted + 1 {
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
        serde_json::json!({"result":"PASS","capture_boundaries":8,"round_faults":faults,"state_negative_controls":3,"poison":"required after every failed round; test-only manual clear for further injected faults"}),
    )
}

#[test]
#[ignore = "requires a full-model GPU window, FF_Q38_MODEL and FLYINGFISH_RUNTIME_DIR"]
fn q38_snapshot_poison_rows_gpu() -> Result<()> {
    // select_prefix reads one verify snapshot row per layer (committed - 1); every other row is poisoned.
    let mut e = ready()?;
    let ids = prompt("math1")?;
    let garbage_s = vec![1.0e30f32; 48 * 128 * 128];
    let garbage_h = vec![1.0e30f32; 10240 * 3];
    let mut cases = 0;
    for k in 1..=4 {
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
            cases += 1;
        }
        restore(&mut e)?;
    }
    println!(
        "MTP_SNAPSHOT_POISON_PASS K1..4 cases={cases} allocated_rows=4 full_accept_keeps_live_state=true"
    );
    receipt(
        "snapshot-poison",
        serde_json::json!({"result":"PASS","cases":cases,"allocated_rows":4,"max_depth":4,"keep":"live final state"}),
    )
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

fn chain_margin(logits: impl IntoIterator<Item = f32>) -> Result<f32> {
    let mut top = [f32::NEG_INFINITY; 2];
    let mut count = 0;
    for value in logits {
        ensure!(value.is_finite(), "Runner FAIL nonfinite chain logits");
        count += 1;
        if value > top[0] {
            top[1] = top[0];
            top[0] = value;
        } else if value > top[1] {
            top[1] = value;
        }
    }
    let margin = top[0] - top[1];
    ensure!(
        count == VOCAB && margin.is_finite() && margin >= 0.0,
        "Runner FAIL chain logits extent={count} margin={margin}"
    );
    Ok(margin)
}

const CHAIN_LOGIT_TOLERANCE: f32 = 0.6875;

fn chain_identity(
    case: &str,
    plain: &[(u32, Vec<f32>)],
    batched: &[(u32, Vec<f32>)],
) -> Result<bool> {
    ensure!(
        !plain.is_empty() && !batched.is_empty(),
        "Runner FAIL empty chain trajectory"
    );
    let mut defects = 0;
    let mut near_ties = 0;
    let mut compared = 0;
    let mut maximum = 0.0f32;
    let mut diverged_at = None;
    for (position, (expected, actual)) in plain.iter().zip(batched).enumerate() {
        let plain_margin = chain_margin(expected.1.iter().copied())?;
        let batched_margin = chain_margin(actual.1.iter().copied())?;
        let drift = expected
            .1
            .iter()
            .zip(&actual.1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let different_id = expected.0 != actual.0;
        let valid = drift <= CHAIN_LOGIT_TOLERANCE
            && (!different_id || plain_margin <= 2.0 * CHAIN_LOGIT_TOLERANCE);
        let near_tie = valid && different_id;
        defects += usize::from(!valid);
        near_ties += usize::from(near_tie);
        maximum = maximum.max(drift);
        compared += 1;
        println!(
            "{} q38_mtp_batch_chain_position {}",
            if !valid {
                "DEFECT"
            } else if near_tie {
                "NEAR_TIE"
            } else {
                "PASS"
            },
            serde_json::json!({"case":case,"position":position,"shared_prefix":true,
                "plain_id":expected.0,"batched_id":actual.0,
                "plain_margin":plain_margin,"batched_margin":batched_margin,
                "max_abs_logit_delta":drift,"tolerance":CHAIN_LOGIT_TOLERANCE,
                "margin_cut":2.0*CHAIN_LOGIT_TOLERANCE})
        );
        if different_id {
            diverged_at = Some(position);
            break;
        }
    }
    if diverged_at.is_none() && plain.len() != batched.len() {
        defects += 1;
        println!(
            "DEFECT q38_mtp_batch_chain_length case={case} plain={} batched={}",
            plain.len(),
            batched.len()
        );
    }
    println!(
        "{} q38_mtp_batch_chain_case {}",
        if defects != 0 {
            "DEFECT"
        } else if near_ties != 0 {
            "NEAR_TIE"
        } else {
            "PASS"
        },
        serde_json::json!({"case":case,"plain_tokens":plain.len(),"batched_tokens":batched.len(),
            "compared_shared_prefix_positions":compared,"diverged_at":diverged_at,
            "post_divergence_positions_scored":0,"max_abs_logit_delta":maximum,
            "defective_positions":defects,"near_ties":near_ties,
            "tolerance":CHAIN_LOGIT_TOLERANCE,"margin_cut":2.0*CHAIN_LOGIT_TOLERANCE})
    );
    Ok(defects == 0)
}

#[test]
fn batch_chain_identity_rejects_logit_perturbation() -> Result<()> {
    let mut logits = vec![0.0; VOCAB];
    logits[17] = 8.0;
    logits[263] = 0.5;
    let plain = [(17, logits)];
    ensure!(
        chain_identity("control-aligned", &plain, &plain)?,
        "aligned control failed"
    );
    let mut changed = plain.clone();
    changed[0].1[VOCAB - 17] += 2.0 * CHAIN_LOGIT_TOLERANCE;
    ensure!(
        !chain_identity("control-planted-logit", &plain, &changed)?,
        "planted logit perturbation was not rejected"
    );
    println!(
        "PASS batch_chain_noise_control planted_position=0 vocab_index={} unchanged_id=17 delta={} tolerance={}",
        VOCAB - 17,
        2.0 * CHAIN_LOGIT_TOLERANCE,
        CHAIN_LOGIT_TOLERANCE
    );
    Ok(())
}

#[test]
#[ignore = "authorized full-model GPU window; production depth4 Batched/graph shared-prefix logits"]
fn q38_mtp_batch_chain_gpu() -> Result<()> {
    let mut e = load(true)?;
    e.mtp_enable(options(
        4,
        MtpHiddenTap::PostFinalNorm,
        MtpRecurrentTap::PostMtpNorm,
        true,
    ))?;
    e.mtp_batch_enable()?;
    e.capture()?;
    e.closure.ready(std::time::Instant::now())?;
    let mut defects = 0;
    for case in ["math1", "code1", "gen1", "gen2", "math2"] {
        let ids = prompt(case)?;
        e.reset()?;
        let first = e.prefill(&ids, None)?;
        let logits = e.decode_logits()?;
        chain_margin(logits.iter().copied())?;
        let mut plain = vec![(first, logits)];
        while plain.len() < 64 && !e.model_spec.eos.contains(&plain.last().unwrap().0) {
            let token = e.decode(true, None)?;
            let logits = e.decode_logits()?;
            chain_margin(logits.iter().copied())?;
            plain.push((token, logits));
        }
        let first = e.mtp_prefill(&ids)?;
        let logits = e.decode_logits()?;
        chain_margin(logits.iter().copied())?;
        let mut batched = vec![(first, logits)];
        while batched.len() < 64 && !e.model_spec.eos.contains(&batched.last().unwrap().0) {
            let round = e.mtp_batch_round(64 - batched.len(), None)?;
            ensure!(
                !round.round.tokens.is_empty(),
                "Runner FAIL batch stopped before budget"
            );
            if round.verify_predictions.is_empty() {
                ensure!(
                    round.round.tokens.len() == 1,
                    "Runner FAIL single-row tail extent"
                );
                let logits = e.decode_logits()?;
                chain_margin(logits.iter().copied())?;
                batched.push((round.round.tokens[0], logits));
            } else {
                let rows = round.verify_predictions.len();
                ensure!(
                    (2..=crate::mtp::MAX_VERIFY_ROWS).contains(&rows)
                        && round.round.tokens.len() <= rows,
                    "Runner FAIL verify output/logit alignment"
                );
                let logits = e.device.stream.clone_dtoh(
                    &e.mtp
                        .as_ref()
                        .context("MTP missing")?
                        .batch
                        .as_ref()
                        .context("batch missing")?
                        .logits
                        .slice(..rows * VOCAB),
                )?;
                for (&token, values) in round.round.tokens.iter().zip(logits.as_chunks::<VOCAB>().0)
                {
                    let values: Vec<_> = values.iter().map(|x| x.to_f32()).collect();
                    chain_margin(values.iter().copied())?;
                    batched.push((token, values));
                }
            }
        }
        println!(
            "q38_mtp_batch_chain_termination {}",
            serde_json::json!({"case":case,"plain_tokens":plain.len(),
                "batched_tokens":batched.len(),
                "plain_eos":e.model_spec.eos.contains(&plain.last().unwrap().0),
                "batched_eos":e.model_spec.eos.contains(&batched.last().unwrap().0)})
        );
        defects += usize::from(!chain_identity(case, &plain, &batched)?);
    }
    println!(
        "{} q38_mtp_batch_chain_gate cases=5 depth=4 graph=true taps=post/post horizon=64 natural_eos=true tolerance={} margin_cut={} defective_cases={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" },
        CHAIN_LOGIT_TOLERANCE,
        2.0 * CHAIN_LOGIT_TOLERANCE
    );
    ensure!(
        defects == 0,
        "DEFECT batched/greedy shared-prefix cases={defects}"
    );
    Ok(())
}

#[test]
#[ignore = "authorized review GPU window; frozen K2 bootstrap acceptance regression"]
fn q38_mtp_bootstrap_acceptance_gpu() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("FF_Q38_MTP_REF_ROOT").context("FF_Q38_MTP_REF_ROOT missing")?,
    );
    let path = root.join("mtp-gate-bounds.json");
    let bounds: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let band = &bounds["bootstrap_acceptance"];
    let z = band["z"]
        .as_f64()
        .context("mtp-gate-bounds.json bootstrap_acceptance.z missing")?;
    let reference_rounds: BTreeMap<String, Vec<usize>> =
        serde_json::from_value(band["round_tokens"].clone())
            .context("mtp-gate-bounds.json bootstrap_acceptance.round_tokens missing")?;
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
            let mut round_tokens = Vec::new();
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
                round_tokens.push(r.tokens.len());
                tokens.extend(r.tokens);
            }
            ensure!(
                emitted == 200 && tokens.len() == 233,
                "bootstrap output budget changed"
            );
            published += emitted;
            rounds += count;
            cases.push(serde_json::json!({"case":name,"iteration":iteration,"tokens":tokens,"measured_tokens":emitted,"rounds":count,"round_tokens":round_tokens}));
        }
    }
    ensure!(
        inputs
            .iter()
            .all(|(name, _)| reference_rounds.contains_key(*name)),
        "bootstrap_acceptance.round_tokens must cover every prompt"
    );
    let (reference, half_width) =
        acceptance_band(&reference_rounds.into_values().collect::<Vec<_>>(), z)?;
    let actual = published as f64 / rounds as f64;
    let pass = (actual / reference - 1.0).abs() <= half_width;
    receipt(
        "bootstrap-acceptance",
        serde_json::json!({
            "result":if pass {"PASS"} else {"FAIL"},"depth":2,"warmup_tokens":32,
            "reference_mean_tokens_per_round":reference,"z":z,"relative_half_width":half_width,
            "published":published,"rounds":rounds,"mean_tokens_per_round":actual,"cases":cases
        }),
    )?;
    ensure!(
        pass,
        "K2 bootstrap acceptance {actual} is outside +/-{half_width} of {reference}"
    );
    Ok(())
}

/// Mean tokens per round of the reference streams and the relative half-width
/// z·√2·SE/mean, SE = sqrt(σ²·τ/n), τ = 1 + 2·Σρ_k over lags before the first negative ρ_k.
/// ρ_k pairs rounds within one stream; σ² is the population variance over all n rounds.
fn acceptance_band(streams: &[Vec<usize>], z: f64) -> Result<(f64, f64)> {
    let all: Vec<f64> = streams.iter().flatten().map(|&x| x as f64).collect();
    ensure!(all.len() > 1, "reference round counts missing");
    let n = all.len() as f64;
    let mean = all.iter().sum::<f64>() / n;
    let ss: f64 = all.iter().map(|x| (x - mean).powi(2)).sum();
    ensure!(ss > 0.0, "reference round counts have zero variance");
    let longest = streams.iter().map(Vec::len).max().unwrap_or(0);
    let mut tau = 1.0;
    for k in 1..longest {
        let rho = streams
            .iter()
            .flat_map(|s| s.windows(k + 1))
            .map(|w| (w[0] as f64 - mean) * (w[k] as f64 - mean))
            .sum::<f64>()
            / ss;
        if rho < 0.0 {
            break;
        }
        tau += 2.0 * rho;
    }
    let se = (ss / n * tau / n).sqrt();
    Ok((mean, z * std::f64::consts::SQRT_2 * se / mean))
}

#[test]
fn acceptance_band_truncates_at_first_negative_autocorrelation() -> Result<()> {
    // [3,3,1,1]: mean 2, σ² 1, ρ1 = 1/4, ρ2 = -1/2, τ = 3/2.
    let (mean, h) = acceptance_band(&[vec![3, 3, 1, 1]], 2.0)?;
    assert_eq!(mean, 2.0);
    let expected = 2.0 * std::f64::consts::SQRT_2 * (1.5f64 / 4.0).sqrt() / 2.0;
    assert!((h - expected).abs() <= 4.0 * f64::EPSILON * expected);
    // [3,1,3,1]: ρ1 = -3/4, τ = 1.
    let (_, h) = acceptance_band(&[vec![3, 1, 3, 1]], 2.0)?;
    let expected = 2.0 * std::f64::consts::SQRT_2 * 0.25f64.sqrt() / 2.0;
    assert!((h - expected).abs() <= 4.0 * f64::EPSILON * expected);
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
        4,
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
