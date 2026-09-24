use ff_edge0::config::Edge0Config;
use ff_edge0::model::Edge0Text;

fn main() -> anyhow::Result<()> {
    let Some(model_dir) = ff_core::paths::checkpoint_dir("Edge0/Edge0-35B-A3B-preview") else {
        anyhow::bail!("set FF_MODELS_DIR to the local models root");
    };
    let model_dir = &model_dir;
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
    let dir = std::env::var("EDGE0_LOGIT_DIR").unwrap_or_else(|_| "output".into());
    std::fs::create_dir_all(&dir)?;
    let path = format!("{dir}/edge0_logits_{tag}.f32");
    std::fs::write(&path, bytemuck::cast_slice(&logits))?;
    println!("wrote {path} ({} floats)", logits.len());
    Ok(())
}
