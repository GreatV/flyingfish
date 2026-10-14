//! In-process GDN P1 integration timing: one model load per process, CUDA events
//! bracket prefill and decode only. Arms are separate processes (the planner derives
//! the core; there is no runtime switch), so the driver interleaves process pairs.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use flyingfish::model::Model;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    runtime_dir: PathBuf,
    #[arg(long)]
    tokens_file: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Arm label written into the record (e.g. main / cand).
    #[arg(long)]
    label: String,
    #[arg(long, default_value_t = 20000)]
    capacity: usize,
    /// Measured prefill repetitions, each from a reset.
    #[arg(long, default_value_t = 4)]
    prefill_reps: usize,
    /// Decode steps measured after each prefill rep.
    #[arg(long, default_value_t = 16)]
    decode_steps: usize,
    #[arg(long)]
    chunk: Option<usize>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.prefill_reps > 0 && a.decode_steps > 0, "positive reps");
    let ids: Vec<u32> =
        serde_json::from_slice(&std::fs::read(&a.tokens_file)?).context("tokens file")?;
    ensure!(
        !ids.is_empty() && ids.len() + 64 <= a.capacity,
        "tokens vs capacity"
    );
    let settings = flyingfish::backend::Settings {
        tree_builder: flyingfish::backend::TreeBuilder::Waves,
        spec_budget: flyingfish::backend::SpecBudget::Chain,
        draft_model: None,
        spec_graph: true,
        runtime_dir: Some(a.runtime_dir.clone()),
        prefix_cache: false,
    };
    let started = Instant::now();
    let mut model = Model::load(&a.model, 0, a.capacity, a.chunk, settings)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    // One discarded prefill+decode warms caches and graphs.
    model.reset()?;
    let first = model.prefill(&ids, None)?;
    for _ in 0..4 {
        model.step(false)?;
    }
    // Q38 has no scalar rewind; reset is the Q38-native clean state for measurement.
    model.reset()?;
    let _ = first;
    let mut prefill_gpu_ms = Vec::new();
    let mut decode_gpu_ms = Vec::new();
    for _ in 0..a.prefill_reps {
        model.reset()?;
        let begin = model.event()?;
        let end = model.event()?;
        model.record(&begin)?;
        model.prefill(&ids, None)?;
        model.record(&end)?;
        end.synchronize()?;
        prefill_gpu_ms.push(begin.elapsed_ms(&end)? as f64);
        let dbegin = model.event()?;
        let dend = model.event()?;
        model.record(&dbegin)?;
        for _ in 0..a.decode_steps {
            model.step(false)?;
        }
        model.record(&dend)?;
        dend.synchronize()?;
        decode_gpu_ms.push(dbegin.elapsed_ms(&dend)? as f64 / a.decode_steps as f64);
    }
    let out = json!({
        "label": a.label,
        "tokens": ids.len(),
        "load_ms": load_ms,
        "prefill_gpu_ms": prefill_gpu_ms,
        "decode_step_gpu_ms": decode_gpu_ms,
        "decode_steps": a.decode_steps,
        "capacity": a.capacity,
    });
    std::fs::write(&a.output, serde_json::to_vec_pretty(&out)?)?;
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
