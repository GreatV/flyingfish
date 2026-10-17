use anyhow::{Context, Result};
fn main() -> Result<()> {
    let arg = std::env::args_os()
        .nth(1)
        .context("usage: q38_audit MODEL_DIR")?;
    let dir = std::path::Path::new(&arg);
    let spec = flyingfish::model_spec::ModelSpec::read(dir)?;
    let mapped = flyingfish::checkpoint::MappedCheckpoint::open(dir)?;
    let manifest = mapped.manifest(&spec, flyingfish::checkpoint::LoadPolicy::TextOnly)?;
    let plans = (0..64)
        .map(|i| flyingfish::linear_plan::LayerDequantPlan::from_manifest(&manifest, i))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::json!({"text_tensors":manifest.entries.values().filter(|e|e.load).count(),"all_tensors":manifest.entries.len(),"resident_bytes":manifest.resident_bytes,"max_layer_arena_bytes":plans.iter().map(|p|p.bytes).max(),"layers":spec.layers,"q_width":spec.q_width,"hidden":spec.hidden,"eos":spec.eos})
    );
    Ok(())
}
