//! Capture all layer boundaries on the reference's prompt plus forced continuation.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::{Settings, SpecBudget, TreeBuilder},
    model::Model,
    trace::Trace,
};
use serde_json::Value;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "coordinator full-model GPU window required"
    );
    let paths: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(
        paths.len() == 6,
        "usage: q38_layers MODEL INPUTS_DIR REFERENCE_JSON RUNTIME_DIR math1|code1 OUTPUT"
    );
    let name = paths[4].to_str().context("invalid prompt name")?;
    ensure!(
        matches!(name, "math1" | "code1"),
        "unsupported layer fixture"
    );
    let reference: Value = serde_json::from_slice(&fs::read(&paths[2])?)?;
    let entry = &reference["prompts"][name];
    let input: Value =
        serde_json::from_slice(&fs::read(paths[1].join(format!("{name}.input_ids.json")))?)?;
    let mut ids: Vec<u32> = serde_json::from_value(input["input_ids"].clone())?;
    ensure!(
        entry["prompt_len"].as_u64() == Some(ids.len() as u64),
        "prompt mismatch"
    );
    let positions = entry["positions"]
        .as_array()
        .context("missing forced positions")?;
    ensure!(
        positions.len() == 64,
        "layer fixture requires 64 forced tokens"
    );
    for (i, p) in positions.iter().enumerate() {
        ensure!(p["i"].as_u64() == Some(i as u64), "noncontiguous positions");
        ids.push(u32::try_from(
            p["tok"].as_u64().context("missing forced token")?,
        )?);
    }
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(paths[3].clone()),
    };
    let mut model = Model::load(&paths[0], 0, 128, Some(128), settings)?;
    model.check_ready()?;
    model.reset()?;
    let mut trace = Trace::all_layers();
    model.prefill(&ids, Some(&mut trace))?;
    for i in 0..64 {
        for (point, width) in [
            ("norm1", 5120),
            ("mixer_out", 6144),
            ("o_proj", 5120),
            ("post_attention", 5120),
            ("norm2", 5120),
        ] {
            ensure!(
                trace.shape(&format!("layer.{i}.debug.{point}"))? == [1, ids.len(), width],
                "incomplete layer trace"
            );
        }
    }
    trace.save(&paths[5])?;
    model.check_ready()?;
    Ok(())
}
