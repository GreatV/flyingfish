use anyhow::{Context, Result};
use flyingfish::runtime::artifact::{ArtifactStaging, sync_parent_directory};
use flyingfish::runtime::frame_manifest::PngFrameSetManifest;
use flyingfish::runtime::telemetry::TelemetryMonitor;
use std::path::{Path, PathBuf};

pub(crate) fn mib_to_bytes(value: u64) -> Result<u64> {
    value
        .checked_mul(1024 * 1024)
        .context("MiB resource limit exceeds u64")
}

/// The directory a new output will be created in.
///
/// `Path::parent` answers `Some("")` for a bare relative file name, not `None`,
/// so `unwrap_or(".")` never fires there and leaves an empty path that every
/// later `is_dir` or `canonicalize` rejects. Both spellings name the working
/// directory. `ArtifactStaging` already resolves this the same way.
pub(crate) fn output_parent(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

pub(crate) fn ensure_new_output(path: &Path, label: &str) -> Result<()> {
    anyhow::ensure!(!path.exists(), "{label} already exists: {}", path.display());
    let parent = output_parent(path);
    anyhow::ensure!(
        parent.is_dir(),
        "{label} parent directory does not exist: {}",
        parent.display()
    );
    Ok(())
}

pub(crate) fn create_new_directory(path: &Path, label: &str) -> Result<()> {
    std::fs::create_dir(path)
        .with_context(|| format!("failed to create {label} {}", path.display()))?;
    let parent = output_parent(path);
    sync_parent_directory(parent)
        .with_context(|| format!("failed to synchronize {label} parent {}", parent.display()))?;
    Ok(())
}

pub(crate) fn resolve_output_outside_model(path: &Path, model: &Path) -> Result<PathBuf> {
    let model = std::fs::canonicalize(model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let candidate = if std::fs::symlink_metadata(path).is_ok() {
        std::fs::canonicalize(path)
            .with_context(|| format!("failed to resolve existing output {}", path.display()))?
    } else {
        let parent = output_parent(path);
        let parent = std::fs::canonicalize(parent)
            .with_context(|| format!("failed to resolve output parent {}", parent.display()))?;
        match path.file_name() {
            Some(name) => parent.join(name),
            None => parent,
        }
    };
    anyhow::ensure!(
        !candidate.starts_with(&model),
        "refusing to write output inside the model directory: {}",
        candidate.display()
    );
    Ok(candidate)
}

pub(crate) fn write_telemetry(path: &Path, monitor: TelemetryMonitor) -> Result<()> {
    let report = monitor.finish()?;
    let json = serde_json::to_vec_pretty(&report)?;
    let staging = ArtifactStaging::new(path)
        .with_context(|| format!("failed to stage telemetry report {}", path.display()))?;
    publish_staged_bytes(staging, &json)?;
    println!("saved runtime telemetry to {}", path.display());
    Ok(())
}

pub(crate) fn publish_staged_bytes(
    staging: ArtifactStaging,
    bytes: &[u8],
) -> Result<flyingfish::runtime::artifact::PublishedArtifact> {
    staging.write_bytes(bytes)?;
    Ok(staging.publish()?)
}

pub(crate) fn publish_png_frame_manifest(directory: &Path, frame_count: usize) -> Result<()> {
    const FILE_NAME: &str = "frames.manifest.json";
    let manifest = PngFrameSetManifest::collect(directory, FILE_NAME, frame_count)?;
    let json = manifest.canonical_json()?;
    let path = directory.join(FILE_NAME);
    let staging_parent = directory
        .parent()
        .context("PNG frame directory has no staging parent")?;
    let staging = ArtifactStaging::new_with_staging_parent(&path, staging_parent)
        .with_context(|| format!("failed to stage PNG frame-set manifest {}", path.display()))?;
    publish_staged_bytes(staging, &json)?;
    manifest.verify_completed_directory(directory, FILE_NAME)?;
    Ok(())
}
