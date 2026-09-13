use super::*;
use flyingfish::{runtime::artifact::ArtifactStaging, runtime::identity::WeakModelIdentity};

pub(super) fn run_identify_model(command: Command) -> Result<()> {
    let Command::Identify { checkpoint, output } = command else {
        bail!("internal CLI dispatch mismatch for identify");
    };

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
