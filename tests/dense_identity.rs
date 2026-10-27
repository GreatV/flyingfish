#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::{Settings, SpecBudget, TreeBuilder},
    kv::KvFormat,
    model::Model,
};
use std::{io::Write, path::PathBuf};

#[test]
#[ignore = "requires GPU lease, FF_ARGMAX_MODEL, FLYINGFISH_RUNTIME_DIR and FF_DENSE_EVIDENCE"]
fn dense_model_identity_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let output = PathBuf::from(std::env::var("FF_DENSE_EVIDENCE")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: KvFormat::Bf16,
    };
    let mut model = Model::load(&target, 0, 256, Some(16), settings)?;
    ensure!(
        model.config.hidden_size == 2048
            && model.config.num_hidden_layers == 42
            && model.config.vocab_size == 130560,
        "dense MiniCPM5-2B geometry required"
    );
    model.check_ready()?;
    for rows in [1, 17, 129] {
        let ids: Vec<_> = (0..rows)
            .map(|i| ((i * 269 + 23) % 130560) as u32)
            .collect();
        let mut eager = Vec::new();
        for graph in [false, true] {
            model.reset()?;
            let mut token = model.prefill(&ids, None)?;
            let mut file =
                std::fs::File::create_new(output.join(format!("rows{rows}-graph{graph}.bin")))?;
            let mut margin = f32::INFINITY;
            for step in 0..16 {
                if graph && step == 1 {
                    model.capture()?;
                }
                if step > 0 {
                    token = model.decode(graph, None)?;
                }
                let logits = model.decode_logits()?;
                ensure!(
                    logits.len() == 130560 && logits.iter().all(|v| v.is_finite()),
                    "invalid full-vocabulary logits"
                );
                let mut order: Vec<_> = (0..logits.len()).collect();
                order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
                ensure!(
                    order[0] == token as usize,
                    "sample differs from logits argmax"
                );
                margin = margin.min(logits[order[0]] - logits[order[1]]);
                let bits: Vec<_> = logits.into_iter().map(f32::to_bits).collect();
                if graph {
                    ensure!(
                        eager.get(step).context("eager step missing")? == &(token, bits.clone()),
                        "graph/eager token or logit bits differ rows={rows} step={step}"
                    );
                } else {
                    eager.push((token, bits.clone()));
                }
                file.write_all(&token.to_le_bytes())?;
                for value in bits {
                    file.write_all(&value.to_le_bytes())?;
                }
            }
            file.flush()?;
            model.check_ready()?;
            println!(
                "DENSE_MODEL_IDENTITY rows={rows} graph={graph} steps=16 vocab=130560 min_top_two_margin={margin} token_logits_written=true PASS"
            );
        }
    }
    println!("DENSE_MODEL_IDENTITY cells=6 graph_eager_bit_identity=true PASS");
    Ok(())
}
