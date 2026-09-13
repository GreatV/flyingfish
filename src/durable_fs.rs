//! Durable filesystem primitives shared by the checkpoint layouts.

use crate::runtime::artifact::FileStat;
use crate::runtime::artifact::sync_parent_directory;

use anyhow::{Context, Result};
use std::{fs, path::Path};

/// A builder for a directory private to this user where the platform can
/// express that.
///
/// Unix sets the mode at creation so the directory is never briefly readable.
/// Windows has no mode to set here and inherits the parent's ACL.
#[cfg(unix)]
fn private_directory_builder() -> fs::DirBuilder {
    use std::os::unix::fs::DirBuilderExt as _;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    builder
}

#[cfg(not(unix))]
fn private_directory_builder() -> fs::DirBuilder {
    fs::DirBuilder::new()
}

pub(crate) fn create_private_directory(path: &Path, label: &str) -> Result<()> {
    private_directory_builder()
        .create(path)
        .with_context(|| format!("failed to create private directory {}", path.display()))?;
    validate_private_directory(path, label)
}

pub(crate) fn validate_private_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = FileStat::of_path(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_dir(),
        "{label} is not a non-symlink directory: {}",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        anyhow::ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "{label} grants group or other access: {}",
            path.display()
        );
    }
    Ok(())
}

/// Make a directory's own entries durable, as far as the platform allows.
///
/// Only Unix can flush a directory's metadata. Windows offers no equivalent, so
/// there the durability of an entry comes from the write-through publication
/// that creates it: that flush commits the filesystem's metadata log, which
/// already contains the directory the entry was written into. Reporting the
/// weaker level is the honest answer; failing would refuse a platform that has
/// simply arranged the same guarantee differently.
pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    sync_parent_directory(path)
        .with_context(|| format!("failed to synchronize directory {}", path.display()))?;
    Ok(())
}
