use crate::{model::Model, spec::Options, tensors::Tensors, trace::Trace, tree::Tree};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Deserialize)]
pub struct Request {
    schema: String,
    fixture_file: PathBuf,
    fixture_sha256: String,
    prompt_id: String,
    fixture_budget: usize,
    engine_budget: usize,
    vocab_size: usize,
    hidden_size: usize,
    input_ids: Vec<u32>,
    target_layer_ids: Vec<i32>,
    rounds: Vec<Round>,
}

#[derive(Deserialize)]
struct Round {
    index: usize,
    seq: i64,
    prefix_ids: Vec<u32>,
    prefix_len: usize,
    output_offset: i32,
    limit: usize,
    eos_ids: Vec<u32>,
    tokens: Vec<u32>,
    parent: Vec<i32>,
    depth: Vec<i32>,
    position: Vec<i32>,
    slot: Vec<i32>,
    anc: Vec<u64>,
    reference_predictions: Vec<u32>,
    accepted_path: Vec<usize>,
    bonus: u32,
    commit_path: Vec<usize>,
    output_ids: Vec<u32>,
    next_anchor: u32,
    next_prefix_len: usize,
    commit_logits_row: usize,
}

pub fn sha256(path: &Path) -> Result<String> {
    let result = Command::new("sha256sum").arg(path).output()?;
    ensure!(
        result.status.success(),
        "sha256sum failed for {}",
        path.display()
    );
    Ok(String::from_utf8(result.stdout)?
        .split_whitespace()
        .next()
        .context("SHA output empty")?
        .to_owned())
}

fn ids(t: &mut Tensors, name: String, values: &[u32]) -> Result<()> {
    t.i64(
        name,
        vec![values.len()],
        &values.iter().map(|&v| v as i64).collect::<Vec<_>>(),
    )
}
fn rows(t: &mut Tensors, name: String, values: &[usize]) -> Result<()> {
    let data = values
        .iter()
        .map(|&v| i32::try_from(v).context("row exceeds I32"))
        .collect::<Result<Vec<_>>>()?;
    t.i32(name, vec![data.len()], &data)
}
fn scalar(t: &mut Tensors, name: String, value: usize) -> Result<()> {
    t.i32(
        name,
        vec![1],
        &[i32::try_from(value).context("scalar exceeds I32")?],
    )
}

pub fn run(target: &Path, draft: &Path, plan: &Path, dump: &Path, options: &Options) -> Result<()> {
    let request: Request = serde_json::from_slice(&std::fs::read(plan)?)?;
    ensure!(
        request.schema == "m6-replay-request-v1",
        "unsupported tree request schema"
    );
    let fixture = if request.fixture_file.is_absolute() {
        request.fixture_file.clone()
    } else {
        plan.parent()
            .unwrap_or(Path::new("."))
            .join(&request.fixture_file)
    }
    .canonicalize()?;
    ensure!(
        sha256(&fixture)? == request.fixture_sha256,
        "tree fixture SHA256 mismatch"
    );
    ensure!(
        dump.file_name() == fixture.file_name(),
        "tree dump must retain fixture filename"
    );
    ensure!(
        request.engine_budget == options.backend.spec_budget.rows()
            && options.backend.spec_budget.is_tree(),
        "request engine budget does not match CLI tree budget"
    );
    ensure!(
        !request.rounds.is_empty() && request.target_layer_ids == [1, 10, 20, 30, 39],
        "invalid tree request rounds/capture layers"
    );
    let mut model = Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.clone(),
    )?;
    ensure!(
        model.config.vocab_size == request.vocab_size
            && model.config.hidden_size == request.hidden_size,
        "request model dimensions mismatch"
    );
    model.enable_draft(draft)?;
    let mut tensors = Tensors::default();
    ids(&mut tensors, "input_ids".into(), &request.input_ids)?;
    scalar(&mut tensors, "budget".into(), request.fixture_budget)?;
    scalar(&mut tensors, "rounds".into(), request.rounds.len())?;
    tensors.i32(
        "target_layer_ids".into(),
        vec![5],
        &request.target_layer_ids,
    )?;
    for (index, r) in request.rounds.iter().enumerate() {
        ensure!(
            r.index == index && r.prefix_len == r.prefix_ids.len(),
            "request round index/prefix mismatch"
        );
        let tree = Tree::edges(&r.tokens, &r.parent, r.prefix_len, options.capacity)?;
        let n = tree.nodes().len();
        ensure!(
            tree.nodes()
                .iter()
                .map(|v| v.depth as i32)
                .collect::<Vec<_>>()
                == r.depth
                && tree
                    .nodes()
                    .iter()
                    .map(|v| v.position as i32)
                    .collect::<Vec<_>>()
                    == r.position
                && tree.nodes().iter().map(|v| v.anc).collect::<Vec<_>>() == r.anc,
            "request tree topology/position/anc mismatch"
        );
        ensure!(
            (0..n)
                .map(|i| (r.prefix_len + i) as i32)
                .collect::<Vec<_>>()
                == r.slot,
            "request physical slots mismatch"
        );
        model.reset()?;
        model.prefill(&r.prefix_ids, None)?;
        model.set_token(r.tokens[0])?;
        let predictions = model.verify_tree(&tree, None)?;
        let metadata = model.tree_metadata()?;
        ensure!(
            metadata.ids == r.tokens
                && metadata.ancestors == r.anc
                && metadata.positions == r.position
                && metadata.slots == r.slot
                && metadata.valid_rows == n as i32
                && metadata.prefix == r.prefix_len as i32,
            "device tree metadata differs from request"
        );
        let hidden = model.tree_hidden()?;
        let logits = (0..n)
            .map(|i| model.verification_logits(i))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let actual = tree.select(&predictions, r.limit, &r.eos_ids)?;
        let actual_walk = tree.walk(&predictions)?;
        let selected = tree.select(&r.reference_predictions, r.limit, &r.eos_ids)?;
        let walk = tree.walk(&r.reference_predictions)?;
        ensure!(
            walk.rows == r.accepted_path
                && walk.bonus == r.bonus
                && selected.rows == r.commit_path
                && selected.output == r.output_ids
                && selected.next == r.next_anchor
                && selected.position == r.next_prefix_len
                && selected.logits_row == r.commit_logits_row,
            "reference tree Commit is inconsistent with request"
        );
        let kv_path = model.tree_kv_path(&selected.rows)?;
        model.commit_tree_fixture(&tree, &selected, &r.eos_ids, &r.reference_predictions)?;
        let injected = model.tree_injected_hidden()?;
        let committed = model.kv_snapshot(r.prefix_len, selected.rows.len())?;
        ensure!(
            kv_path
                .iter()
                .zip(&committed)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "tree KV gather/scatter changed bytes"
        );
        let h = model.config.hidden_size;
        let expected_hidden = selected
            .rows
            .iter()
            .flat_map(|&i| hidden[i * 5 * h..(i + 1) * 5 * h].iter().copied())
            .collect::<Vec<_>>();
        ensure!(
            injected
                .iter()
                .zip(&expected_hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "GPU compact hidden differs from selected tree hidden"
        );
        let mut dt = Trace::default();
        let keep = selected.rows.len();
        model.draft_kv(&mut dt, r.prefix_len, keep)?;
        let mut draft_kv = Vec::new();
        for layer in 0..5 {
            for kind in ["k", "v"] {
                let data = dt.values(&format!("kv_{kind}_{layer}"))?;
                for head in 0..2 {
                    for row in 0..keep {
                        draft_kv.extend_from_slice(
                            &data[row * 256 + head * 128..row * 256 + (head + 1) * 128],
                        );
                    }
                }
            }
        }
        let pre = format!("r{index}.");
        tensors.i64(format!("{pre}seq"), vec![1], &[r.seq])?;
        ids(&mut tensors, format!("{pre}prefix_ids"), &r.prefix_ids)?;
        scalar(&mut tensors, format!("{pre}prefix_len"), r.prefix_len)?;
        tensors.i32(format!("{pre}output_offset"), vec![1], &[r.output_offset])?;
        scalar(&mut tensors, format!("{pre}limit"), r.limit)?;
        ids(&mut tensors, format!("{pre}eos_ids"), &r.eos_ids)?;
        ids(&mut tensors, format!("{pre}tokens"), &metadata.ids)?;
        tensors.i32(format!("{pre}parent"), vec![n], &r.parent)?;
        tensors.i32(format!("{pre}depth"), vec![n], &r.depth)?;
        tensors.i32(format!("{pre}position"), vec![n], &metadata.positions)?;
        tensors.i32(format!("{pre}slot"), vec![n], &metadata.slots)?;
        tensors.u64(format!("{pre}anc"), vec![n], &metadata.ancestors)?;
        tensors.f32(format!("{pre}logits"), vec![n, request.vocab_size], &logits)?;
        tensors.f32(format!("{pre}hidden"), vec![n, 5, h], &hidden)?;
        ids(&mut tensors, format!("{pre}predictions"), &predictions)?;
        rows(
            &mut tensors,
            format!("{pre}actual_accepted_path"),
            &actual_walk.rows,
        )?;
        ids(
            &mut tensors,
            format!("{pre}actual_output_ids"),
            &actual.output,
        )?;
        ids(
            &mut tensors,
            format!("{pre}actual_bonus"),
            &[actual_walk.bonus],
        )?;
        rows(&mut tensors, format!("{pre}accepted_path"), &walk.rows)?;
        ids(&mut tensors, format!("{pre}output_ids"), &selected.output)?;
        ids(&mut tensors, format!("{pre}bonus"), &[walk.bonus])?;
        rows(&mut tensors, format!("{pre}commit_path"), &selected.rows)?;
        ids(&mut tensors, format!("{pre}next_anchor"), &[selected.next])?;
        scalar(
            &mut tensors,
            format!("{pre}next_prefix_len"),
            selected.position,
        )?;
        scalar(
            &mut tensors,
            format!("{pre}commit_logits_row"),
            selected.logits_row,
        )?;
        tensors.f32(format!("{pre}injected_hidden"), vec![keep, 5, h], &injected)?;
        tensors.f32(
            format!("{pre}target_kv_path"),
            vec![42, 2, 2, keep, 128],
            &kv_path,
        )?;
        tensors.f32(
            format!("{pre}target_kv_committed"),
            vec![42, 2, 2, keep, 128],
            &committed,
        )?;
        tensors.f32(
            format!("{pre}draft_kv_committed"),
            vec![5, 2, 2, keep, 128],
            &draft_kv,
        )?;
        println!(
            "{}",
            serde_json::json!({"tree_replay_round":index,"actual_predictions":predictions,"actual_output":actual.output,"forced_output":selected.output,"copy_bitwise":true,"commit_policy":"fixture","pass":true})
        );
    }
    let mut metadata = HashMap::new();
    for (k, v) in [
        ("schema", "tree-verify-v1"),
        ("role", "engine"),
        ("compute_dtype", "bfloat16"),
        ("storage_dtype", "float32"),
        ("prefix_policy", "independent_teacher_forced"),
        ("commit_policy", "fixture"),
        ("capture_layers", "1,10,20,30,39"),
    ] {
        metadata.insert(k.into(), v.into());
    }
    metadata.insert("prompt_id".into(), request.prompt_id);
    metadata.insert(
        "mode".into(),
        if request.fixture_budget == 8 {
            "chain"
        } else {
            "tree"
        }
        .into(),
    );
    metadata.insert("fixture_file".into(), fixture.display().to_string());
    metadata.insert("fixture_sha256".into(), request.fixture_sha256);
    metadata.insert("ff_sha256".into(), sha256(&std::env::current_exe()?)?);
    metadata.insert("engine_budget".into(), request.engine_budget.to_string());
    metadata.insert("spec_graph".into(), options.backend.spec_graph.to_string());
    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()?;
    ensure!(date.status.success(), "date UTC failed");
    metadata.insert(
        "created_utc".into(),
        String::from_utf8(date.stdout)?.trim().to_owned(),
    );
    tensors.save_new(dump, metadata)?;
    Ok(())
}
