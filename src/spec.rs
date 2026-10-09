use crate::{backend::Settings, dspark::Round, model::Model, tokenizer, trace::Trace};
use anyhow::{Result, ensure};
use std::{path::Path, time::Instant};

pub struct Options {
    pub device: usize,
    pub capacity: usize,
    pub chunk: Option<usize>,
    pub backend: Settings,
    pub count: usize,
    pub ignore_eos: bool,
}

pub struct Output {
    pub ids: Vec<u32>,
    pub rounds: Vec<Round>,
    pub decode_ms: f64,
    pub prefill_ms: f64,
}

pub fn check_tree_greedy(
    target: &Path,
    draft: &Path,
    ids: &[u32],
    options: &Options,
    report: &Path,
) -> Result<()> {
    ensure!(
        options.backend.spec_budget.is_tree(),
        "check-tree-greedy requires --spec-budget tree16, tree32 or tree64"
    );
    check(ids, options)?;
    ensure!(
        options.ignore_eos,
        "strict tree greedy check requires --ignore-eos"
    );
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)?;
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    let mut plain_settings = options.backend.clone();
    plain_settings.draft_model = None;
    plain_settings.spec_budget = crate::backend::SpecBudget::Chain;
    let mut plain = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        plain_settings,
    )?;
    let first = model.prefill(ids, None)?;
    let ordinary = plain.prefill(ids, None)?;
    plain.capture()?;
    let mut generated = vec![first];
    let mut ordinary_ids = vec![ordinary];
    let mut divergence = if first != ordinary {
        Some(serde_json::json!({"step":0,"actual":first,"expected":ordinary,"common_prefix":[]}))
    } else {
        None
    };
    let mut rounds = Vec::new();
    while divergence.is_none() && generated.len() < options.count {
        let round = model.spec_round(options.count - generated.len(), None, "")?;
        for &actual in &round.output {
            let expected = plain.decode(true, None)?;
            let step = generated.len();
            generated.push(actual);
            ordinary_ids.push(expected);
            if actual != expected {
                divergence = Some(
                    serde_json::json!({"step":step,"actual":actual,"expected":expected,"common_prefix":&ordinary_ids[..step]}),
                );
                break;
            }
        }
        rounds.push(round);
    }
    let count = rounds.len().max(1) as f64;
    let result = serde_json::json!({"schema":"m6-strict-greedy-v1","budget":options.backend.spec_budget,"strict":true,"ignore_eos":true,
        "ordinary_ids":ordinary_ids,"generated_ids":generated,"rounds":rounds,
        "mean_accepted_including_bonus":rounds.iter().map(|r|r.accepted+1).sum::<usize>() as f64/count,
        "mean_committed":rounds.iter().map(|r|r.committed).sum::<usize>() as f64/count,"first_divergence":divergence,"pass":divergence.is_none()});
    serde_json::to_writer_pretty(&mut file, &result)?;
    use std::io::Write;
    file.write_all(b"\n")?;
    file.sync_all()?;
    println!(
        "{}",
        serde_json::json!({"tree_greedy_report":report,"pass":divergence.is_none(),"compared_tokens":generated.len()})
    );
    ensure!(
        divergence.is_none(),
        "strict tree greedy mismatch; first divergence saved in {}",
        report.display()
    );
    Ok(())
}

pub fn compare_padding(
    target: &Path,
    draft: &Path,
    ids: &[u32],
    options: &Options,
    nodes: usize,
) -> Result<()> {
    check(ids, options)?;
    ensure!(
        options.backend.spec_budget.is_tree()
            && nodes > 1
            && nodes < options.backend.spec_budget.rows(),
        "padding comparison requires1<N<tree budget"
    );
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    let anchor = model.prefill(ids, None)?;
    let mut tokens = vec![anchor];
    tokens.extend((1..nodes).map(|i| ((anchor as usize + i) % model.config.vocab_size) as u32));
    let parents: Vec<i32> = (0..nodes)
        .map(|i| if i == 0 { -1 } else { ((i - 1) / 2) as i32 })
        .collect();
    let tree = crate::tree::Tree::edges(&tokens, &parents, ids.len(), options.capacity)?;
    let expected = model.verify_tree(&tree, None)?;
    let hidden = model.tree_hidden()?;
    let logits = (0..nodes)
        .map(|i| model.verification_logits(i))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        hidden
            .iter()
            .chain(logits.iter().flatten())
            .all(|v| v.is_finite()),
        "nonfinite unpoisoned tree output"
    );
    model.poison_tree_padding(&tree)?;
    let got = model.verify_tree(&tree, None)?;
    ensure!(got == expected, "padding poison changed target predictions");
    let poisoned_hidden = model.tree_hidden()?;
    ensure!(
        hidden
            .iter()
            .zip(&poisoned_hidden)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "padding poison changed valid hidden bits"
    );
    for (row, old) in logits.iter().enumerate() {
        let new = model.verification_logits(row)?;
        ensure!(
            old.iter()
                .zip(&new)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "padding poison changed valid logits at row{row}"
        );
    }
    println!(
        "{}",
        serde_json::json!({"tree_padding_compare":{"N":nodes,"budget":options.backend.spec_budget.rows(),
        "graph":options.backend.spec_graph,"poison":"all42 layers K/V [L+N,L+budget) BF16 NaN","finite":true,"bitwise":true,"pass":true,
        "scope":"structural negative control; no generated accuracy reference"}})
    );
    Ok(())
}

pub fn compare_graph(target: &Path, draft: &Path, ids: &[u32], options: &Options) -> Result<()> {
    check(ids, options)?;
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    model.set_spec_graph(false)?;
    model.reset()?;
    let first = model.prefill(ids, None)?;
    let first_logits = model.decode_logits()?;
    let mut eager = Vec::new();
    let mut emitted = 1;
    while emitted < options.count {
        let start = model.position();
        let round = model.spec_round(options.count - emitted, None, "")?;
        let bits = model.spec_state_bits(start, round.committed)?;
        emitted += round.committed;
        eager.push((round, bits));
    }
    model.set_spec_graph(true)?;
    model.reset()?;
    ensure!(
        model.prefill(ids, None)? == first,
        "Graph/eager prefill token mismatch"
    );
    let logits = model.decode_logits()?;
    ensure!(
        logits.len() == first_logits.len()
            && logits
                .iter()
                .zip(&first_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Graph/eager prefill logits differ bitwise"
    );
    emitted = 1;
    for (index, (expected, bits)) in eager.iter().enumerate() {
        let start = model.position();
        let actual = model.spec_round(options.count - emitted, None, "")?;
        ensure!(
            actual.proposals == expected.proposals
                && actual.predictions == expected.predictions
                && actual.accepted == expected.accepted
                && actual.committed == expected.committed
                && actual.output == expected.output
                && actual.position == expected.position
                && actual.logits_rows == expected.logits_rows,
            "Graph/eager token/commit mismatch in round {index}"
        );
        let got = model.spec_state_bits(start, actual.committed)?;
        let mismatch = got.iter().zip(bits).position(|(a, b)| a != b);
        ensure!(
            got.len() == bits.len() && mismatch.is_none(),
            "Graph/eager numerical byte mismatch in round {index}: offset {mismatch:?}"
        );
        emitted += actual.committed;
        println!(
            "{}",
            serde_json::json!({"spec_graph_compare_round":index,"bitwise":true,"compared_bytes":got.len(),"committed":actual.committed,"position":actual.position,"pass":true})
        );
    }
    ensure!(
        emitted == options.count,
        "Graph comparison output count mismatch"
    );
    println!(
        "{}",
        serde_json::json!({"spec_graph_compare":{"prompt_tokens":ids.len(),"output_tokens":emitted,"rounds":eager.len(),"scope":"draft H/B/s, all verification target logits, current logits, newly committed target/draft KV, device token/position/length, proposals/predictions/commit state","bitwise":true,"pass":true}})
    );
    Ok(())
}

fn produce(
    m: &mut Model,
    ids: &[u32],
    options: &Options,
    mut trace: Option<&mut Trace>,
) -> Result<Output> {
    let eos = if options.ignore_eos {
        Vec::new()
    } else {
        m.config.eos_token_id.clone()
    };
    m.set_spec_eos(&eos)?;
    m.reset()?;
    let start = Instant::now();
    let first = m.prefill(ids, None)?;
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut output = vec![first];
    let mut rounds = Vec::new();
    let start = Instant::now();
    while output.len() < options.count
        && (options.ignore_eos
            || !m
                .config
                .eos_token_id
                .contains(output.last().expect("nonempty output")))
    {
        let prefix = format!("round{}", rounds.len() + 1);
        let mut round =
            m.spec_round(options.count - output.len(), trace.as_deref_mut(), &prefix)?;
        if !options.ignore_eos
            && let Some(index) = round
                .output
                .iter()
                .position(|v| m.config.eos_token_id.contains(v))
        {
            round.output.truncate(index + 1);
            let position = round.position - round.committed + round.output.len();
            m.rewind(position, *round.output.last().expect("nonempty EOS prefix"))?;
            round.committed = round.output.len();
            round.position = position;
        }
        output.extend(&round.output);
        rounds.push(round);
    }
    m.synchronize()?;
    m.check_logits()?;
    Ok(Output {
        ids: output,
        rounds,
        decode_ms: start.elapsed().as_secs_f64() * 1000.0,
        prefill_ms,
    })
}

fn check(ids: &[u32], options: &Options) -> Result<()> {
    ensure!(
        !ids.is_empty() && options.count > 0,
        "spec input and output count must be positive"
    );
    let budget = options.backend.spec_budget;
    let headroom = if budget.is_tree() { budget.rows() } else { 7 };
    let required = ids
        .len()
        .checked_add(options.count)
        .and_then(|v| v.checked_add(headroom))
        .ok_or_else(|| {
            anyhow::anyhow!("spec budget {budget:?} required KV capacity overflows usize")
        })?;
    ensure!(
        required <= options.capacity,
        "spec budget {budget:?} requires KV capacity {required} (prompt {} + output {} + scratch {headroom}), got {}",
        ids.len(),
        options.count,
        options.capacity
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{SpecBudget, TreeBuilder};

    fn options(budget: SpecBudget, capacity: usize) -> Options {
        Options {
            device: 0,
            capacity,
            chunk: None,
            backend: Settings {
                tree_builder: TreeBuilder::Waves,
                spec_budget: budget,
                spec_graph: true,
                linear_choices: Vec::new(),
                draft_model: None,
                runtime_dir: None,
            },
            count: 256,
            ignore_eos: true,
        }
    }

    #[test]
    fn budget_capacity_accepts_exact_fit_and_rejects_one_row_short() -> Result<()> {
        let ids = [1, 2, 3];
        for (budget, headroom) in [
            (SpecBudget::Chain, 7),
            (SpecBudget::Tree16, 16),
            (SpecBudget::Tree32, 32),
            (SpecBudget::Tree64, 64),
        ] {
            let required = ids.len() + 256 + headroom;
            check(&ids, &options(budget, required))?;
            let error = check(&ids, &options(budget, required - 1))
                .expect_err("one-row-short capacity must fail")
                .to_string();
            assert!(error.contains(&format!("{budget:?}")));
            assert!(error.contains(&format!("requires KV capacity {required}")));
            assert!(error.contains(&format!("got {}", required - 1)));
        }
        Ok(())
    }

    #[test]
    fn budget_capacity_rejects_overflow() {
        let mut options = options(SpecBudget::Tree64, usize::MAX);
        options.count = usize::MAX;
        let error = check(&[1], &options)
            .expect_err("overflow must fail")
            .to_string();
        assert!(error.contains("Tree64") && error.contains("overflows usize"));
    }
}

pub fn generate(
    target: &Path,
    draft: &Path,
    ids: &[u32],
    options: &Options,
    dump: Option<&Path>,
) -> Result<()> {
    check(ids, options)?;
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    let mut trace = Trace::default();
    let result = produce(&mut model, ids, options, dump.map(|_| &mut trace))?;
    if let Some(dump) = dump {
        trace.save(dump)?;
    }
    for (i, round) in result.rounds.iter().enumerate() {
        println!("{}", serde_json::json!({"spec_round":i,"stats":round}));
    }
    // SGLang acceptance = output tokens / verify rounds, including the first prefill token.
    let sglang_accept_len = result.ids.len() as f64 / result.rounds.len().max(1) as f64;
    println!(
        "{}",
        serde_json::json!({"spec_generation":{"input_ids":ids,"generated_ids":result.ids,"text":tokenizer::decode(target,&result.ids)?,"prefill_ms":result.prefill_ms,"decode_ms":result.decode_ms,"decode_tok_s":(result.ids.len()-1) as f64*1000.0/result.decode_ms,"sglang_accept_len":sglang_accept_len,"end_to_end_tok_s":result.ids.len() as f64*1000.0/(result.prefill_ms+result.decode_ms),"mean_accept_including_bonus":result.rounds.iter().map(|r|r.accepted+1).sum::<usize>() as f64/result.rounds.len().max(1) as f64,"ignore_eos":options.ignore_eos,"draft_attention":"noncausal full block","verify_attention":model.verification_impl()}})
    );
    Ok(())
}

pub fn bench(
    target: &Path,
    draft: &Path,
    ids: &[u32],
    options: &Options,
    runs: usize,
) -> Result<()> {
    check(ids, options)?;
    ensure!(
        runs > 0 && options.count >= 200 && options.ignore_eos,
        "spec bench needs >=200 output tokens, positive runs and ignore EOS"
    );
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    produce(&mut model, ids, options, None)?;
    let mut times = Vec::new();
    let mut end_to_end = Vec::new();
    let mut all_rounds = Vec::new();
    for run in 0..runs {
        let result = produce(&mut model, ids, options, None)?;
        times.push(result.decode_ms);
        end_to_end.push(result.prefill_ms + result.decode_ms);
        for (i, round) in result.rounds.into_iter().enumerate() {
            println!(
                "{}",
                serde_json::json!({"spec_bench_round":i,"run":run,"stats":round})
            );
            all_rounds.push(round);
        }
        println!(
            "{}",
            serde_json::json!({"spec_bench_run":run,"decode_ms":result.decode_ms,"prefill_ms":result.prefill_ms,"generated_ids":result.ids})
        );
    }
    times.sort_by(f64::total_cmp);
    let middle = (times.len() - 1) as f64 / 2.0;
    let ms = (times[middle.floor() as usize] + times[middle.ceil() as usize]) / 2.0;
    end_to_end.sort_by(f64::total_cmp);
    let total_ms = (end_to_end[middle.floor() as usize] + end_to_end[middle.ceil() as usize]) / 2.0;
    let rounds = all_rounds.len() as f64;
    // SGLang acceptance = output tokens / verify rounds, including the first prefill token.
    let sglang_accept_len = options.count as f64 * runs as f64 / rounds;
    let trees: Vec<_> = all_rounds.iter().filter_map(|r| r.tree.as_ref()).collect();
    let tree_summary = (!trees.is_empty()).then(|| serde_json::json!({
        "builder": options.backend.tree_builder,
        "base_ms_per_round": trees.iter().map(|t| t.base_ms).sum::<f64>() / rounds,
        "build_ms_per_round": trees.iter().map(|t| t.build_ms).sum::<f64>() / rounds,
        "waves_per_round": trees.iter().map(|t| t.waves).sum::<usize>() as f64 / rounds,
        "requests_per_round": trees.iter().map(|t| t.requests).sum::<usize>() as f64 / rounds,
        "used_requests_per_round": trees.iter().map(|t| t.used_requests).sum::<usize>() as f64 / rounds,
    }));
    println!(
        "{}",
        serde_json::json!({"spec_benchmark":{"prompt_tokens":ids.len(),"output_tokens":options.count,"runs":runs,"warmup_runs":1,"decode_wall_ms_median":ms,"end_to_end_ms_median":total_ms,"end_to_end_tok_s":options.count as f64*1000.0/total_ms,"sglang_accept_len":sglang_accept_len,"decode_tok_s":(options.count-1) as f64*1000.0/ms,"mean_accept_including_bonus":all_rounds.iter().map(|r|r.accepted+1).sum::<usize>() as f64/rounds,"mean_emitted":all_rounds.iter().map(|r|r.committed).sum::<usize>() as f64/rounds,"draft_ms_per_round":all_rounds.iter().map(|r|r.draft_ms).sum::<f64>()/rounds,"verify_ms_per_round":all_rounds.iter().map(|r|r.verify_ms).sum::<f64>()/rounds,"inject_ms_per_round":all_rounds.iter().map(|r|r.inject_ms).sum::<f64>()/rounds,"timing":"wall time; token transfers and synchronization included; initial prefill excluded","draft_attention":"full block","verify":model.verification_impl(),"confidence_budget":"fixed 7","tree_build":tree_summary}})
    );
    Ok(())
}

pub fn compare_builders(target: &Path, draft: &Path, ids: &[u32], options: &Options) -> Result<()> {
    check(ids, options)?;
    ensure!(
        options.backend.spec_budget.is_tree(),
        "builder comparison requires a tree budget"
    );
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    model.compare_tree_builders(true)?;
    let result = produce(&mut model, ids, options, None)?;
    for (index, round) in result.rounds.iter().enumerate() {
        let tree = round
            .tree
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing tree comparison stats"))?;
        ensure!(
            tree.compared_serial,
            "serial/wave tree comparison was not performed"
        );
        println!(
            "{}",
            serde_json::json!({"tree_builder_compare_round": index,"stats":tree,"pass":true})
        );
    }
    println!(
        "{}",
        serde_json::json!({"tree_builder_compare":{"budget": options.backend.spec_budget,
        "prompt_tokens":ids.len(),"output_tokens":result.ids.len(),"rounds":result.rounds.len(),
        "scope":"same actual draft B; independent real Markov GPU requests; nodes and f64 logp bits equal", "pass":true}})
    );
    Ok(())
}

pub fn profile(target: &Path, draft: &Path, ids: &[u32], options: &Options) -> Result<()> {
    check(ids, options)?;
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    model.prefill(ids, None)?;
    model.spec_round(8, None, "")?;
    model.reset()?;
    model.prefill(ids, None)?;
    model.synchronize()?;
    model.profile_rounds(true);
    let round = model.spec_round(8, None, "")?;
    model.profile_rounds(false);
    model.check_logits()?;
    println!(
        "{}",
        serde_json::json!({"profile_spec_round":round,"prompt_tokens":ids.len(),"warmup_rounds":1})
    );
    Ok(())
}

pub fn verify_greedy(
    target: &Path,
    draft: &Path,
    ids: &[u32],
    options: &Options,
    limits: &[f64],
    reference: Option<&Path>,
    rules: Option<&crate::verify::Rules>,
) -> Result<()> {
    check(ids, options)?;
    ensure!(
        limits.len() >= options.count,
        "greedy comparison needs per-step max_abs thresholds"
    );
    ensure!(
        limits.iter().all(|v| v.is_finite() && *v >= 0.0),
        "invalid max_abs threshold"
    );
    let reference = reference
        .map(|path| {
            crate::verify::GreedyReference::read(
                path,
                crate::config::Config::read(target)?.vocab_size,
                options.count,
            )
        })
        .transpose()?;
    if let Some(r) = &reference {
        ensure!(
            r.input == ids,
            "FP32 reference input_ids differ from the requested prompt"
        );
    }
    ensure!(
        rules.is_none() || reference.is_some(),
        "formal logits gates require the FP32 reference fixture"
    );
    let mut plain_settings = options.backend.clone();
    plain_settings.draft_model = None;
    let mut plain = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        plain_settings,
    )?;
    let mut spec = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    let first = plain.prefill(ids, None)?;
    let spec_first = spec.prefill(ids, None)?;
    let mut logit_metrics = Vec::new();
    if let (Some(rules), Some(reference)) = (rules, reference.as_ref()) {
        logit_metrics.push(rules.check_step(&spec.decode_logits()?, reference, 0, spec_first)?);
    }
    let expected_first = reference.as_ref().map_or(first, |r| r.tokens[0]);
    let first_near = reference
        .as_ref()
        .is_some_and(|r| (r.margins[0] as f64) < limits[0]);
    ensure!(
        (first == expected_first && spec_first == expected_first) || first_near,
        "plain/spec prefill first token differs from reference outside near-tie"
    );
    plain.set_token(expected_first)?;
    spec.set_token(expected_first)?;
    plain.capture()?;
    let mut emitted = 1;
    let mut rounds = 0;
    let mut near_ties = usize::from(spec_first != expected_first);
    let mut ordinary_near_ties = usize::from(first != expected_first);
    while emitted < options.count {
        let start = spec.position();
        let round = spec.spec_round(options.count - emitted, None, "")?;
        let mut committed = 0;
        for (row, &actual) in round.output.iter().enumerate() {
            if let (Some(rules), Some(reference)) = (rules, reference.as_ref()) {
                logit_metrics.push(rules.check_step(
                    &spec.verification_logits(round.logits_rows[row])?,
                    reference,
                    emitted,
                    actual,
                )?);
            }
            let ordinary = plain.decode(true, None)?;
            let logits = plain.decode_logits()?;
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.select_nth_unstable_by(1, |&a, &b| {
                logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
            });
            let ordinary_margin = (logits[order[0]] - logits[order[1]]) as f64;
            let (expected, margin) = reference.as_ref().map_or((ordinary, ordinary_margin), |r| {
                (r.tokens[emitted], r.margins[emitted] as f64)
            });
            let exact = actual == expected;
            let near = !exact && margin < limits[emitted];
            let ordinary_exact = ordinary == expected;
            let ordinary_near = !ordinary_exact && margin < limits[emitted];
            println!(
                "{}",
                serde_json::json!({"spec_greedy_step":emitted,"actual":actual,"expected":expected,"margin":margin,"ordinary_token":ordinary,"ordinary_margin":ordinary_margin,"ordinary_near_tie":ordinary_near,"max_abs_threshold":limits[emitted],"near_tie":near,"pass":(exact||near)&&(ordinary_exact||ordinary_near)})
            );
            ensure!(
                (exact || near) && (ordinary_exact || ordinary_near),
                "spec greedy mismatch at step {emitted}: margin={margin}"
            );
            if near {
                near_ties += 1;
                spec.rewind(plain.position(), expected)?;
            }
            if ordinary_near {
                ordinary_near_ties += 1;
                plain.set_token(expected)?;
            }
            emitted += 1;
            committed += 1;
            if near {
                break;
            }
        }
        ensure!(
            spec.position() == plain.position(),
            "spec/ordinary valid KV length mismatch"
        );
        let a = spec.kv_snapshot(start, committed)?;
        let b = plain.kv_snapshot(start, committed)?;
        let error = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>();
        let scale = b.iter().map(|&b| (b as f64).powi(2)).sum::<f64>();
        let max = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| ((a - b) as f64).abs())
            .fold(0.0f64, f64::max);
        println!(
            "{}",
            serde_json::json!({"spec_kv_round":rounds,"position":spec.position(),"rows":committed,"rel_rmse":(error/scale.max(f64::MIN_POSITIVE)).sqrt(),"max_abs":max,"gate":"diagnostic; peer numerical thresholds pending"})
        );
        rounds += 1;
    }
    if let Some(rules) = rules {
        ensure!(
            rules.systematic(&logit_metrics)?,
            "formal speculative logits distribution gates failed"
        );
    }
    println!(
        "{}",
        serde_json::json!({"spec_greedy_summary":{"pass":true,"output_tokens":emitted,"rounds":rounds,"near_tie_exceptions":near_ties,"ordinary_near_tie_exceptions":ordinary_near_ties,"margin_source":if reference.is_some() { "peer FP32 fixture" } else { "ordinary engine BF16 logits; FP32 reference margin recheck pending" },"reference_path":"at a permitted near-tie the speculative block is truncated and both models continue with the reference token"}})
    );
    Ok(())
}

#[derive(serde::Deserialize)]
pub struct Replay {
    pub initial: std::path::PathBuf,
    pub rounds: Vec<ReplayRound>,
}

#[derive(serde::Deserialize)]
pub struct ReplayRound {
    pub draft: std::path::PathBuf,
    pub inject: std::path::PathBuf,
}

fn errors(a: &[f32], b: &[f32]) -> Result<(f64, f64, f64)> {
    ensure!(
        !a.is_empty() && a.len() == b.len(),
        "replay tensor sizes differ"
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "nonfinite replay tensor"
    );
    let max = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    let mean = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (a - b).abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    let square = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum::<f64>();
    let norm = b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    Ok((mean, max, (square / norm.max(f64::MIN_POSITIVE)).sqrt()))
}

fn inject_reference(m: &mut Model, path: &Path, rules: &crate::dspark::Rules) -> Result<usize> {
    let data = Trace::read(path)?;
    let hidden = data.values("inject_hidden")?;
    let positions = data.values("inject_positions")?;
    let rows = data.shape("inject_hidden")?[0];
    let commit_values = data.values("inject_commit_lens")?;
    ensure!(
        commit_values.len() == 1
            && commit_values[0].is_finite()
            && commit_values[0] > 0.0
            && commit_values[0].fract() == 0.0,
        "invalid injection commit length"
    );
    let commit = commit_values[0] as usize;
    ensure!(
        positions.len() == rows && commit > 0 && commit <= rows,
        "invalid injection positions/commit length"
    );
    let start = positions[0] as usize;
    ensure!(
        positions
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (start + i) as f32),
        "replay requires contiguous integer injection positions"
    );
    m.inject_hidden(&hidden[..commit * 10240], start, commit)?;
    let mut actual = Trace::default();
    m.draft_kv(&mut actual, start, commit)?;
    for layer in 0..5 {
        for kind in ["k", "v"] {
            let name = format!("kv_{kind}_{layer}");
            {
                let expected = data.values(&name)?;
                ensure!(
                    expected.len() >= commit * 256,
                    "reference {name} is shorter than committed rows"
                );
                let (mean, max, rel) = errors(&actual.values(&name)?, &expected[..commit * 256])?;
                let pass = rel <= rules.kv_rel;
                println!(
                    "{}",
                    serde_json::json!({"replay_inject":path,"tensor":name,"mean_abs":mean,"max_abs":max,"rel_rmse":rel,"rel_limit":rules.kv_rel,"gate":"hard","pass":pass})
                );
                ensure!(
                    pass,
                    "DSpark injection {} {name} rel_rmse {rel} exceeds {}",
                    path.display(),
                    rules.kv_rel
                );
            }
        }
    }
    Ok(start + commit)
}

pub fn replay(
    target: &Path,
    draft: &Path,
    options: &Options,
    plan: &Path,
    thresholds: &Path,
) -> Result<()> {
    let replay: Replay = serde_json::from_slice(&std::fs::read(plan)?)?;
    let rules = crate::dspark::Rules::read(thresholds)?;
    println!(
        "{}",
        serde_json::json!({"dspark_rules":rules,"source":thresholds})
    );
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.with_draft(draft),
    )?;

    let mut position = inject_reference(&mut model, &replay.initial, &rules)?;
    let mut near = 0;
    let mut exact = 0;
    let mut pass = true;
    for (index, round) in replay.rounds.iter().enumerate() {
        let expected = Trace::read(&round.draft)?;
        let block = expected.values("draft_block_ids")?;
        let proposals = expected.values("proposals")?;
        let scores = expected.values("s")?;
        ensure!(
            block.len() == 7 && proposals.len() == 7 && scores.len() == 7 * model.config.vocab_size,
            "invalid draft replay block"
        );
        let mut actual = Trace::default();
        let reference_tokens: Vec<u32> = proposals.iter().map(|&v| v as u32).collect();
        let got = model.draft_proposals(
            block[0] as u32,
            position,
            &mut actual,
            "draft",
            Some(&reference_tokens),
        )?;
        for (name, reference) in [("H", "draft_hidden"), ("B", "B"), ("s", "s")] {
            let actual_values = actual.values(&format!("draft.{name}"))?;
            let expected_values = expected.values(reference)?;
            let (mean, max, rel) = errors(&actual_values, &expected_values)?;
            let peak = actual_values
                .iter()
                .zip(&expected_values)
                .enumerate()
                .max_by(|(_, (a, b)), (_, (c, d))| ((*a - *b).abs()).total_cmp(&((*c - *d).abs())))
                .map(
                    |(index, (&a, &b))| serde_json::json!({"index":index,"actual":a,"reference":b}),
                );
            println!(
                "{}",
                serde_json::json!({"replay_peak":name,"round":index,"value":peak})
            );
            let gate = match name {
                "H" => mean <= rules.h_mean && rel <= rules.h_rel,
                "B" => mean <= rules.b_mean,
                "s" => mean <= rules.s_mean,
                _ => unreachable!(),
            };
            pass &= gate;
            println!(
                "{}",
                serde_json::json!({"replay_round":index,"tensor":name,"mean_abs":mean,"max_abs":max,"rel_rmse":rel,"pass":gate,"gate":"formal §13; max_abs diagnostic"})
            );
        }
        for row in 0..7 {
            let s = &scores[row * model.config.vocab_size..(row + 1) * model.config.vocab_size];
            let mut order: Vec<usize> = (0..s.len()).collect();
            order.select_nth_unstable_by(1, |&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
            let margin = s[order[0]] - s[order[1]];
            let matched = got[row] == proposals[row] as u32;
            let tie = !matched && (margin as f64) < rules.margin;
            if matched {
                exact += 1;
            }
            if tie {
                near += 1;
            }
            pass &= matched || tie;
            println!(
                "{}",
                serde_json::json!({"replay_round":index,"proposal_row":row,"actual":got[row],"expected":proposals[row] as u32,"margin":margin,"near_tie":tie,"pass":matched||tie})
            );
        }
        position = inject_reference(&mut model, &round.inject, &rules)?;
    }
    println!(
        "{}",
        serde_json::json!({"replay_summary":{"rounds":replay.rounds.len(),"exact_proposals":exact,"near_tie_exceptions":near,"pass":pass,"path":"SGLang prefix+injected-hidden+Markov predecessor teacher forcing"}})
    );
    ensure!(pass, "SGLang draft replay failed");
    Ok(())
}
