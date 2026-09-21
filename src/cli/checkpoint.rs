use super::CheckpointCommand;
use super::output_hygiene::{
    ensure_new_output, publish_staged_bytes, resolve_output_outside_model,
};
use anyhow::Context;
use anyhow::Result;
use flyingfish::checkpoint_layout;
use flyingfish::runtime::artifact::ArtifactStaging;
use std::path::{Path, PathBuf};

pub(crate) fn resolve_component(model: &Path, component: &Path) -> Result<PathBuf> {
    anyhow::ensure!(
        model.is_dir(),
        "model directory does not exist: {}",
        model.display()
    );
    anyhow::ensure!(
        !component.is_absolute(),
        "component must be relative to the model directory"
    );
    anyhow::ensure!(
        component
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_))),
        "component path may not contain . or .."
    );
    let model = std::fs::canonicalize(model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let unresolved = model.join(component);
    let result = std::fs::canonicalize(&unresolved).with_context(|| {
        format!(
            "failed to resolve component directory {}",
            unresolved.display()
        )
    })?;
    anyhow::ensure!(
        result.is_dir(),
        "component directory does not exist: {}",
        result.display()
    );
    anyhow::ensure!(
        result.starts_with(&model),
        "component directory escapes the model root: {}",
        result.display()
    );
    Ok(result)
}

pub(super) fn run(command: CheckpointCommand) -> Result<()> {
    match command {
        CheckpointCommand::StageLayout { checkpoint, output } => {
            let source = std::fs::canonicalize(&checkpoint)
                .context("failed to resolve Base transformer component")?;
            let model_root = source
                .parent()
                .context("Base transformer has no model root")?;
            let output = output
                .map(|path| resolve_output_outside_model(&path, model_root))
                .transpose()?;
            if let Some(path) = &output {
                ensure_new_output(path, "stage-layout report")?;
            }
            let report = checkpoint_layout::inspect_h3_base_exact_stage_layout(&checkpoint)?;
            match output {
                Some(path) => {
                    let staging = ArtifactStaging::new(&path)?;
                    publish_staged_bytes(staging, &report.canonical_json()?)?;
                    println!("saved stage-layout report to {}", path.display());
                }
                None => println!("{}", String::from_utf8(report.pretty_json()?)?),
            }
        }
        CheckpointCommand::RepackExactStages { checkpoint, output } => {
            let manifest = checkpoint_layout::repack_h3_base_exact_stages(&checkpoint, &output)?;
            println!(
                "verified {} tensors ({} payload bytes) across {} stage files in {}",
                manifest.layout.tensor_count,
                manifest.layout.total_payload_bytes,
                manifest.layout.stages.len(),
                output.display()
            );
        }
    }
    Ok(())
}
