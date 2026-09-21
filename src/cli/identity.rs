use super::output_hygiene::{
    ensure_new_output, publish_staged_bytes, resolve_output_outside_model,
};
use anyhow::{Context, Result};
use flyingfish::runtime::artifact::ArtifactStaging;
use flyingfish::runtime::identity::WeakModelIdentity;
use std::path::PathBuf;

pub(super) fn run_identify_model(checkpoint: PathBuf, output: PathBuf) -> Result<()> {
    let component_dir = std::fs::canonicalize(&checkpoint).with_context(|| {
        format!(
            "failed to resolve checkpoint directory {}",
            checkpoint.display()
        )
    })?;
    anyhow::ensure!(
        component_dir.is_dir(),
        "checkpoint directory does not exist: {}",
        component_dir.display()
    );
    let output = resolve_output_outside_model(&output, &component_dir)?;
    ensure_new_output(&output, "model identity output")?;
    let staging = ArtifactStaging::new(&output)
        .with_context(|| format!("failed to stage model identity output {}", output.display()))?;
    let identity = WeakModelIdentity::collect(&component_dir)?;
    let json = identity.canonical_json()?;
    let published = publish_staged_bytes(staging, &json)?;
    println!(
        "identified {} bytes of weight files (file count {}) to {} ({:?})",
        identity.weight_file_bytes(),
        identity.weight_file_count(),
        published.destination.display(),
        published.durability
    );
    Ok(())
}
