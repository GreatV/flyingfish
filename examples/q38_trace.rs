//! Teacher-forced engine trajectory on the exact path carried by a reference score file.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::{Settings, SpecBudget, TreeBuilder},
    model::Model,
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "coordinator GPU window required"
    );
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 7,
        "usage: q38_trace MODEL INPUTS_DIR REFERENCE_JSON RUNTIME_DIR OUTPUT_JSON eager|graph CHUNK"
    );
    let paths: Vec<_> = args[..5].iter().map(PathBuf::from).collect();
    let graph = match args[5].to_str() {
        Some("eager") => false,
        Some("graph") => true,
        _ => anyhow::bail!("expected eager or graph"),
    };
    let chunk: usize = args[6].to_str().context("invalid chunk")?.parse()?;
    ensure!(chunk > 0, "chunk must be positive");
    let reference: Value = serde_json::from_slice(&fs::read(&paths[2])?)?;
    let prompts = reference["prompts"]
        .as_object()
        .context("missing reference prompts")?;
    ensure!(
        prompts.len() == 5,
        "G8 requires exactly five reference prompts"
    );
    let mut cases = Vec::new();
    let mut capacity = 32.max(chunk);
    for (name, entry) in prompts {
        let input: Value =
            serde_json::from_slice(&fs::read(paths[1].join(format!("{name}.input_ids.json")))?)?;
        let ids: Vec<u32> = serde_json::from_value(input["input_ids"].clone())?;
        let positions = entry["positions"]
            .as_array()
            .context("reference positions missing")?;
        ensure!(
            positions.len() == 64 && entry["prompt_len"].as_u64() == Some(ids.len() as u64),
            "prompt/position count mismatch"
        );
        let tokens = positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                ensure!(
                    p["i"].as_u64() == Some(i as u64),
                    "noncontiguous reference positions"
                );
                Ok(u32::try_from(
                    p["tok"].as_u64().context("missing reference token")?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        capacity = capacity.max(ids.len() + 64);
        cases.push((name, entry, ids, tokens));
    }
    ensure!(
        chunk <= capacity,
        "requested chunk {chunk} exceeds sequence capacity {capacity}"
    );
    eprintln!(
        "{}",
        json!({"trajectory_setup":{"capacity":capacity,"requested_chunk":chunk,"prompt_lengths":cases.iter().map(|(name,_,ids,_)| (*name,ids.len())).collect::<std::collections::BTreeMap<_,_>>()}})
    );
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(paths[3].clone()),
        prefix_cache: false,
    };
    let mut model = Model::load(&paths[0], 0, capacity, Some(chunk), settings)?;
    model.check_ready()?;
    let mut report = json!({"schema":1,"mode":"teacher_forced","graph":graph,"chunk":chunk,"reference":paths[2],"prompts":{}});
    for (name, entry, ids, forced) in cases {
        model.reset()?;
        let mut predicted = model.prefill(&ids, None)?;
        let mut positions = Vec::new();
        for (i, &token) in forced.iter().enumerate() {
            model.check_logits()?;
            let logits = model.decode_logits()?;
            ensure!(
                (predicted as usize) < logits.len(),
                "engine token outside logits"
            );
            let ref_pos = &entry["positions"][i];
            let t1 = usize::try_from(ref_pos["top1"].as_u64().context("top1 missing")?)?;
            let t2 = usize::try_from(ref_pos["top2"].as_u64().context("top2 missing")?)?;
            ensure!(
                t1 < logits.len() && t2 < logits.len(),
                "reference token outside logits"
            );
            positions.push(json!({"i":i,"tok":predicted,"forced_token":token,"engine_top1_v":logits[predicted as usize],"reference_top1_v":logits[t1],"reference_top2_v":logits[t2]}));
            if i + 1 < forced.len() {
                model.set_token(token)?;
                predicted = model.decode(graph, None)?;
            }
        }
        report["prompts"][name] = json!({"prompt_len":ids.len(),"input_ids":ids,"cand_sha16":entry["cand_sha16"],"forced_tokens":forced,"positions":positions});
        fs::write(
            &paths[4],
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        model.check_ready()?;
    }
    Ok(())
}
