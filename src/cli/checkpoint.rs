use super::*;
use flyingfish::{checkpoint_layout, runtime::artifact::ArtifactStaging};

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
