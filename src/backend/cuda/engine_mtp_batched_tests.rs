use super::super::tests::{configure, load, options, output, prompt, restore, save, state_bits};
use super::*;

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
// Independent same-partition rebuild: one target batch, then canonical MTP
// rows directly from C(P)'s H2 and that batch's H2 rows. It never invokes the
// transaction controller, restore_mtp or repair_mtp.
fn canonical(e: &mut Engine, inputs: &[u32]) -> Result<Vec<u32>> {
    call(e, |b| {
        let hidden = b.chain.last_target.clone();
        let predictions = b.verify_batch(inputs)?;
        let frame = b
            .chain
            .batch
            .as_ref()
            .context("batch missing")?
            .frame
            .clone();
        for (row, &token) in inputs.iter().enumerate() {
            let h = if row == 0 {
                &hidden
            } else {
                &frame[(row - 1) * HIDDEN..row * HIDDEN]
            };
            // Retain the old head-producing row as the canonical state oracle.
            b.row(token, h, OutputMode::Predict)?;
        }
        b.chain.last_target = frame[(inputs.len() - 1) * HIDDEN..].to_vec();
        b.chain.committed_position = b.engine.position;
        Ok(predictions)
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
                    let inputs: Vec<_> = std::iter::once(anchor)
                        .chain(full[..j].iter().copied())
                        .collect();
                    // Eager canonical is also the graph parity oracle.
                    e.mtp.as_mut().context("MTP missing")?.options.graph = false;
                    let expected = canonical(&mut e, &inputs)?;
                    let bits = state_bits(&e)?;
                    restore(&mut e)?;
                    for graph in [false, true] {
                        let c = e.mtp.as_mut().context("MTP missing")?;
                        c.options.graph = graph;
                        let mut forced = full.clone();
                        if j < k {
                            forced[j] = (forced[j] + 1) % VOCAB as u32;
                        }
                        c.force_proposals = Some(forced);
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
                                == if j == k { (1, 0, 1) } else { (2, 1, 0) },
                            "M2b full/partial path did extra target work"
                        );
                        ensure!(
                            state_bits(&e)? == bits,
                            "M2b state bits mismatch K{k} j{j} {h2:?}/{h5:?} graph{graph}"
                        );
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
        serde_json::json!({"result":"PASS","cases":cases,"state_oracle":"independent same-partition eager rebuild; all target/MTP states and invalid KV tail","paths":"full:1batch+0target_restore; partial:2batch+1target_restore","tail_eos_epoch":"PASS"}),
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
                    policy::Fault::AfterTargetRestore,
                    policy::Fault::AfterReplay,
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
                    !e.poisoned && state_bits(&e)? == before,
                    "batch rollback mismatch {fault:?}"
                );
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
                    !e.poisoned && state_bits(&e)? == before,
                    "canonical row rollback mismatch"
                );
                e.mtp
                    .as_mut()
                    .unwrap()
                    .batch
                    .as_mut()
                    .unwrap()
                    .repair_fail_after = None;
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
        serde_json::json!({"result":"PASS","capture_boundaries":6,"round_faults":faults,"state_negative_controls":3,"retry":"identical tokens and state"}),
    )
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
                            rounds.push(serde_json::json!({"matched":r.round.matched,"accepted":r.round.accepted,"consumed":r.round.consumed,"proposals":r.round.proposals,"tokens":r.round.tokens,"kept_verify_state":r.kept_verify_state,"verify_predictions":r.verify_predictions,"replay_predictions":r.replay_predictions}));
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
