use ff_edge0::config::Edge0Config;
use ff_edge0::model::Edge0Text;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_dir = Path::new("models/Edge0/Edge0-35B-A3B-preview");
    let config = Edge0Config::from_model_dir(model_dir)?;
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let mut model = Edge0Text::load(model_dir, config)?;
    let prompt = "Explain paging to a systems programmer.";
    let templated = ff_edge0::config::chat_prompt(prompt);
    let ids = tokenizer
        .encode(templated.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    let mut hidden = None;
    for &id in &ids {
        hidden = Some(model.forward(id)?);
    }
    let steps = std::env::var("EDGE0_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    let mut logits = Vec::new();
    for _ in 0..steps {
        logits = model.logits(&hidden.clone().unwrap())?;
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        println!(
            "step margin {:.4} (top {} {:.3}, runner {} {:.3})",
            sorted[0] - sorted[1],
            best,
            sorted[0],
            sorted[1],
            sorted[2]
        );
        hidden = Some(model.forward(best)?);
    }
    let tag = std::env::var("EDGE0_LOGIT_TAG").unwrap_or_else(|_| "default".into());
    std::fs::write(
        format!("/tmp/edge0_logits_{tag}.f32"),
        bytemuck::cast_slice(&logits),
    )?;
    println!(
        "wrote /tmp/edge0_logits_{tag}.f32 ({} floats)",
        logits.len()
    );
    Ok(())
}
