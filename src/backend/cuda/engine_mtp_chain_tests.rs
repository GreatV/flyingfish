use super::*;
use crate::backend::{Settings, SpecBudget, TreeBuilder};
use std::path::PathBuf;

pub(super) fn load(write: bool) -> Result<Engine> {
    load_capacity(write, 128)
}
pub(super) fn load_capacity(write: bool, capacity: usize) -> Result<Engine> {
    load_format(write, capacity, crate::kv::KvFormat::Bf16)
}
pub(super) fn load_format(
    write: bool,
    capacity: usize,
    format: crate::kv::KvFormat,
) -> Result<Engine> {
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "M2a requires a separate full-model cloud window"
    );
    let dir = PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("FF_Q38_MODEL missing")?);
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
    );
    Engine::load_kind(
        &dir,
        0,
        capacity,
        Some(16),
        Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            runtime_dir: Some(runtime),
            prefix_cache: false,
            kv_format: format,
        },
        write,
    )
}
pub(super) fn options(k: usize, h2: MtpHiddenTap, h5: MtpRecurrentTap, graph: bool) -> MtpOptions {
    MtpOptions {
        depth: k,
        hidden_tap: h2,
        recurrent_tap: h5,
        graph,
    }
}
pub(super) fn configure(e: &mut Engine, o: MtpOptions) -> Result<()> {
    e.reset()?;
    if let Some(c) = e.mtp.as_mut() {
        o.validate()?;
        c.options = o;
        c.runtime.plan.hidden_tap = o.hidden_tap;
        c.runtime.plan.recurrent_tap = o.recurrent_tap;
    } else {
        e.mtp_enable(o)?;
    }
    Ok(())
}
pub(super) fn prompt(case: &str) -> Result<Vec<u32>> {
    let root = PathBuf::from(
        std::env::var_os("FF_Q38_MTP_REF_ROOT").context("FF_Q38_MTP_REF_ROOT missing")?,
    );
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.join(format!("inputs/{case}.input_ids.json")),
    )?)?;
    Ok(serde_json::from_value(v["input_ids"].clone())?)
}
pub(super) fn output() -> Result<PathBuf> {
    let p =
        PathBuf::from(std::env::var_os("FF_Q38_MTP_OUTPUT").context("FF_Q38_MTP_OUTPUT missing")?);
    std::fs::create_dir_all(&p)?;
    Ok(p)
}
pub(super) fn save(e: &mut Engine) -> Result<()> {
    let mut c = e.mtp.take().context("MTP missing")?;
    let r = Bound {
        engine: e,
        chain: &mut c,
    }
    .save(5);
    e.mtp = Some(c);
    r
}
pub(super) fn restore(e: &mut Engine) -> Result<()> {
    let mut c = e.mtp.take().context("MTP missing")?;
    let r = Bound {
        engine: e,
        chain: &mut c,
    }
    .restore(false);
    e.mtp = Some(c);
    r
}
pub(super) fn ordinary(e: &mut Engine, count: usize) -> Result<Vec<u32>> {
    let mut c = e.mtp.take().context("MTP missing")?;
    let result = (|| -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut b = Bound {
            engine: e,
            chain: &mut c,
        };
        for _ in 0..count {
            let token = b.pending()?;
            // Keep the old head-producing canonical row as the state oracle.
            let hidden = b.chain.last_target.clone();
            b.row(
                token,
                matrix(&hidden, &b.engine.device.stream, 0, 1, HIDDEN)?,
                OutputMode::Predict,
            )?;
            out.push(b.target_decode(token)?);
        }
        b.chain.committed_position = b.engine.position;
        Ok(out)
    })();
    e.mtp = Some(c);
    result
}
pub(super) fn state_bits(e: &Engine) -> Result<Vec<u8>> {
    state_bits_with_visibility(e, false)
}
pub(super) fn committed_bits(e: &Engine) -> Result<Vec<u8>> {
    state_bits_with_visibility(e, true)
}
pub(super) fn committed_difference(
    e: &Engine,
    expected: &[u8],
    actual: &[u8],
) -> Result<serde_json::Value> {
    if expected.len() != actual.len() {
        return Ok(
            serde_json::json!([{"field":"packed_extent","expected":expected.len(),"actual":actual.len()}]),
        );
    }
    let c = e.mtp.as_ref().context("MTP missing")?;
    let mut fields = Vec::new();
    for (name, store, prefix) in [
        ("target", &e.state, e.position),
        (
            "mtp",
            e.mtp_state.as_ref().context("MTP state missing")?,
            c.cursor,
        ),
    ] {
        for (layer, g) in store.gdn.iter().enumerate() {
            fields.push((format!("{name}.gdn{layer}.s"), g.s.len() * 4));
            fields.push((format!("{name}.gdn{layer}.history"), g.history.len() * 4));
        }
        for layer in 0..store.kv.len() {
            for part in ["k", "v"] {
                for head in 0..4 {
                    fields.push((
                        format!("{name}.kv{layer}.{part}.head{head}"),
                        prefix * 256 * 2,
                    ));
                }
            }
        }
        fields.push((
            format!("{name}.sequence"),
            store.sequence.as_ref().context("metadata missing")?.len() * 4,
        ));
    }
    fields.extend([
        ("target.position".into(), e.work.position.len() * 4),
        ("target.length".into(), e.work.length.len() * 4),
        ("target.token".into(), e.work.token.len() * 4),
    ]);
    if e.position > 0 {
        fields.push(("target.logits".into(), e.work.logits.len() * 2));
    }
    fields.extend([
        ("mtp.pending".into(), 4),
        ("mtp.last_target".into(), c.last_target.len() * 2),
        ("mtp.recurrent".into(), c.recurrent.len() * 2),
        ("target.host_position".into(), size_of::<usize>()),
        ("mtp.cursor".into(), size_of::<usize>()),
        ("mtp.committed_position".into(), size_of::<usize>()),
        ("target.epoch".into(), 8),
        ("mtp.epoch".into(), 8),
    ]);
    let used = fields.iter().map(|(_, bytes)| bytes).sum::<usize>();
    ensure!(
        used <= expected.len(),
        "committed state partition exceeds receipt"
    );
    fields.push(("target.qwen_positions".into(), expected.len() - used));
    let mut offset = 0;
    let mut differences = Vec::new();
    for (field, bytes) in fields {
        let a = &expected[offset..offset + bytes];
        let b = &actual[offset..offset + bytes];
        if let Some(index) = a.iter().zip(b).position(|(a, b)| a != b) {
            let end = (index + 8).min(bytes);
            differences.push(serde_json::json!({"field":field,"byte":index,"expected":a[index..end],"actual":b[index..end]}));
        }
        offset += bytes;
    }
    Ok(serde_json::json!(differences))
}
fn state_bits_with_visibility(e: &Engine, visible_only: bool) -> Result<Vec<u8>> {
    let s = &e.device.stream;
    let mut bits = Vec::new();
    for (index, store) in [&e.state, e.mtp_state.as_ref().context("MTP state missing")?]
        .into_iter()
        .enumerate()
    {
        for g in &store.gdn {
            for buf in [&g.s, &g.history] {
                for v in s.clone_dtoh(buf)? {
                    bits.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
        for kv in &store.kv {
            for buf in [&kv.k, &kv.v] {
                if visible_only {
                    let prefix = if index == 0 {
                        e.position
                    } else {
                        e.mtp.as_ref().context("MTP missing")?.cursor
                    };
                    for head in 0..4 {
                        for v in s.clone_dtoh(
                            &buf.slice(head * e.capacity * 256..(head * e.capacity + prefix) * 256),
                        )? {
                            bits.extend_from_slice(&v.to_bits().to_le_bytes());
                        }
                    }
                    continue;
                }
                for v in s.clone_dtoh(buf)? {
                    bits.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
        for v in s.clone_dtoh(store.sequence.as_ref().context("metadata missing")?)? {
            bits.extend_from_slice(&v.to_le_bytes());
        }
    }
    for buf in [&e.work.position, &e.work.length] {
        for v in s.clone_dtoh(buf)? {
            bits.extend_from_slice(&v.to_le_bytes());
        }
    }
    for v in s.clone_dtoh(&e.work.token)? {
        bits.extend_from_slice(&v.to_le_bytes());
    }
    // Before the first prefill, logits are scratch, including capture warmup.
    if e.position > 0 {
        for v in s.clone_dtoh(&e.work.logits)? {
            bits.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    let c = e.mtp.as_ref().context("MTP missing")?;
    bits.extend_from_slice(&c.pending.to_le_bytes());
    for buf in [&c.last_target, &c.recurrent] {
        for v in s.clone_dtoh(buf)? {
            bits.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    for n in [e.position, c.cursor, c.committed_position] {
        bits.extend_from_slice(&n.to_le_bytes());
    }
    bits.extend_from_slice(&e.mtp_epoch.to_le_bytes());
    bits.extend_from_slice(&c.epoch.to_le_bytes());
    for v in e
        .qwen
        .as_ref()
        .context("Qwen missing")?
        .position_values(s)?
    {
        bits.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bits)
}

pub(crate) fn q38_mtp_chain_gpu() -> Result<()> {
    let mut e = load(false)?;
    let out = output()?;
    for k in 1..=3 {
        for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
            for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                for case in ["math1", "code1", "gen1", "gen2", "math2"] {
                    configure(&mut e, options(k, h2, h5, true))?;
                    let ids = prompt(case)?;
                    let first = e.prefill(&ids, None)?;
                    // The ordinary target consumes the same pending-anchor convention.
                    let mut plain = vec![first];
                    for _ in 1..64 {
                        if e.model_spec
                            .eos
                            .contains(plain.last().context("no target output")?)
                        {
                            break;
                        }
                        let t = e.decode(true, None)?;
                        plain.push(t);
                        if e.model_spec.eos.contains(&t) {
                            break;
                        }
                    }
                    // Rebuild the prompt; a four-row transaction journal cannot
                    // rewind a complete 64-token baseline trajectory.
                    ensure!(
                        e.mtp_prefill(&ids)? == first,
                        "MTP bootstrap changed target prefill"
                    );
                    let mut got = vec![first];
                    let mut rounds = Vec::new();
                    while got.len() < plain.len()
                        && !e.model_spec.eos.contains(got.last().context("no output")?)
                    {
                        let r = e.mtp_round(plain.len() - got.len(), None)?;
                        ensure!(!r.tokens.is_empty(), "MTP stopped before target budget");
                        rounds.push(serde_json::json!({"matched":r.matched,"accepted":r.accepted,"consumed":r.consumed,"proposals":r.proposals,"tokens":r.tokens}));
                        got.extend_from_slice(&r.tokens);
                    }
                    ensure!(
                        got == plain,
                        "M2a token divergence {case} K{k} {h2:?}/{h5:?}"
                    );
                    std::fs::write(
                        out.join(format!("{case}-K{k}-{h2:?}-{h5:?}.json")),
                        serde_json::to_vec_pretty(
                            &serde_json::json!({"case":case,"depth":k,"h2":h2,"h5":h5,"tokens":got,"plain_tokens":plain,"rounds":rounds,"target_token_gate":"BIT_EQUAL","recursive_numeric_gate":"requires corrected 6d chain fixture; old 0de341ed not qualified"}),
                        )?,
                    )?;
                    eprintln!("M2A_CHAIN {case} K{k} {h2:?}/{h5:?} target_token_bits PASS");
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn q38_mtp_transaction_gpu() -> Result<()> {
    let mut e = load(false)?;
    let ids = prompt("math1")?;
    for graph in [false, true] {
        for k in 1..=3 {
            for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
                for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                    configure(&mut e, options(k, h2, h5, graph))?;
                    e.mtp_prefill(&ids)?;
                    save(&mut e)?;
                    let predictions = ordinary(&mut e, k + 1)?;
                    restore(&mut e)?;
                    ensure!(
                        !predictions[..k]
                            .iter()
                            .any(|t| e.model_spec.eos.contains(t)),
                        "transaction fixture hit EOS; use an earlier prefix"
                    );
                    for j in 0..=k {
                        let expected = ordinary(&mut e, j + 1)?;
                        let bits = state_bits(&e)?;
                        restore(&mut e)?;
                        let mut proposals = predictions[..k].to_vec();
                        if j < k {
                            proposals[j] = (proposals[j] + 1) % VOCAB as u32;
                        }
                        e.mtp.as_mut().context("MTP missing")?.force_proposals = Some(proposals);
                        let head_runs = e.mtp.as_ref().context("MTP missing")?.runtime.head_runs;
                        let round = e.mtp_round(64, None)?;
                        ensure!(
                            e.mtp.as_ref().context("MTP missing")?.runtime.head_runs - head_runs
                                == k,
                            "M2a canonical repair evaluated an unused MTP head"
                        );
                        ensure!(
                            round.accepted == j && round.tokens == expected,
                            "wrong forced acceptance count/output"
                        );
                        ensure!(
                            state_bits(&e)? == bits,
                            "M2a committed S/history/KV/meta/hidden mismatch at K{k}/j{j}/{h2:?}/{h5:?}/graph{graph}"
                        );
                        restore(&mut e)?;
                        e.mtp.as_mut().context("MTP missing")?.force_proposals = None;
                    }
                }
            }
        }
    }
    // A snapshot near the last consumable slot must clip its copy window.
    configure(
        &mut e,
        options(
            3,
            MtpHiddenTap::PreFinalNorm,
            MtpRecurrentTap::PostMtpNorm,
            true,
        ),
    )?;
    let mut tail = ids;
    tail.resize(e.capacity - 2, 1);
    e.mtp_prefill(&tail)?;
    save(&mut e)?;
    let expected = ordinary(&mut e, 1)?;
    let bits = state_bits(&e)?;
    restore(&mut e)?;
    let r = e.mtp_round(64, None)?;
    ensure!(
        r.tokens == expected && state_bits(&e)? == bits,
        "last consumable slot mismatch"
    );
    ensure!(
        e.mtp_round(64, None)?.tokens.is_empty(),
        "MTP ran beyond legal next-row metadata"
    );
    // A different prefill with the same length must not revive an old MTP cache.
    let ids = prompt("math1")?;
    e.mtp_prefill(&ids)?;
    e.reset()?;
    e.prefill(&ids, None)?;
    let before = state_bits(&e)?;
    ensure!(
        e.mtp_round(1, None).is_err(),
        "MTP accepted a stale state epoch"
    );
    ensure!(
        state_bits(&e)? == before,
        "stale-epoch rejection mutated state"
    );
    Ok(())
}

pub(crate) fn q38_mtp_failure_and_capture_gpu() -> Result<()> {
    let mut e = load(true)?;
    e.mtp_enable(options(
        3,
        MtpHiddenTap::PreFinalNorm,
        MtpRecurrentTap::PostMtpNorm,
        true,
    ))?;
    let before = state_bits(&e)?;
    for occurrence in [1, 2] {
        e.capture_fail_after_layer = Some((0, occurrence));
        ensure!(e.capture().is_err(), "capture fault ignored");
        ensure!(
            e.poisoned && e.graph.is_none() && state_bits(&e)? == before,
            "capture failed to restore state or published graph"
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
        ensure!(
            e.capture().is_err(),
            "capture stage fault ignored: {fault:?}"
        );
        ensure!(
            e.poisoned && e.graph.is_none() && state_bits(&e)? == before,
            "capture stage rollback/publication failure: {fault:?}"
        );
        e.capture_fail_stage = None;
        e.poisoned = false;
    }
    e.capture()?;
    ensure!(
        state_bits(&e)? == before,
        "capture changed target/MTP state"
    );
    e.closure.ready(std::time::Instant::now())?;
    e.mtp_prefill(&prompt("math1")?)?;
    save(&mut e)?;
    let predictions = ordinary(&mut e, 4)?;
    restore(&mut e)?;
    e.mtp.as_mut().context("MTP missing")?.force_proposals = Some(predictions[..3].to_vec());
    let mut points = vec![Fault::AfterRestore, Fault::BeforePublish];
    points.extend((0..3).map(Fault::AfterDraft));
    points.extend((0..4).map(Fault::AfterVerify));
    points.extend((0..4).map(Fault::AfterCanonical));
    points.extend((0..4).map(Fault::AfterReplay));
    let before = state_bits(&e)?;
    for fault in points {
        ensure!(
            e.mtp_round(64, Some(fault)).is_err(),
            "MTP fault ignored: {fault:?}"
        );
        ensure!(
            !e.poisoned && state_bits(&e)? == before,
            "rollback state mismatch: {fault:?}"
        );
    }
    let expected = ordinary(&mut e, 4)?;
    let bits = state_bits(&e)?;
    restore(&mut e)?;
    let round = e.mtp_round(64, None)?;
    ensure!(
        round.tokens == expected && state_bits(&e)? == bits,
        "retry after rollback differs from ordinary decode"
    );
    Ok(())
}
