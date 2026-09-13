//! Bounded artifact reads.
//!
//! A read is taken from an open handle rather than a path, and the file's
//! identity and length are checked before and after, so what was read is one
//! file that did not change underneath the read. That is a narrower claim than
//! knowing the content: nothing here attests to what the bytes are.

use super::identity::*;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSnapshot {
    pub bytes: Vec<u8>,
}

pub fn read_artifact_snapshot(
    path: impl AsRef<Path>,
    maximum_bytes: u64,
) -> Result<ArtifactSnapshot> {
    anyhow::ensure!(maximum_bytes > 0, "artifact read limit must be non-zero");
    let path = path.as_ref();
    let path_before = FileStat::of_path(path)
        .with_context(|| format!("failed to inspect artifact {}", path.display()))?;
    anyhow::ensure!(
        path_before.file_type().is_file(),
        "artifact is not a regular non-symlink file: {}",
        path.display()
    );
    anyhow::ensure!(
        path_before.len() <= maximum_bytes,
        "artifact {} is {} bytes, exceeding the {}-byte read limit",
        path.display(),
        path_before.len(),
        maximum_bytes
    );
    let mut file =
        File::open(path).with_context(|| format!("failed to open artifact {}", path.display()))?;
    let opened_before = FileStat::of_file(&file)
        .with_context(|| format!("failed to inspect artifact {}", path.display()))?;
    anyhow::ensure!(
        opened_before.is_file()
            && identifies_same_file(&path_before, &opened_before)
            && opened_before.len() == path_before.len()
            && opened_before.modified().ok() == path_before.modified().ok()
            && change_marker(&opened_before) == change_marker(&path_before),
        "artifact path changed while opening {}",
        path.display()
    );
    let capacity = usize::try_from(opened_before.len()).context("artifact size exceeds usize")?;
    let mut bytes = Vec::with_capacity(capacity);
    (&mut file)
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read artifact {}", path.display()))?;
    anyhow::ensure!(
        u64::try_from(bytes.len()).context("artifact read length exceeds u64")? <= maximum_bytes,
        "artifact {} grew beyond the {}-byte read limit",
        path.display(),
        maximum_bytes
    );
    let opened_after = FileStat::of_file(&file)
        .with_context(|| format!("failed to re-inspect artifact {}", path.display()))?;
    let path_after = FileStat::of_path(path)
        .with_context(|| format!("failed to re-inspect artifact path {}", path.display()))?;
    anyhow::ensure!(
        path_after.file_type().is_file()
            && identifies_same_file(&opened_before, &opened_after)
            && identifies_same_file(&opened_after, &path_after)
            && opened_after.len() == opened_before.len()
            && path_after.len() == opened_before.len()
            && opened_after.modified().ok() == opened_before.modified().ok()
            && path_after.modified().ok() == opened_before.modified().ok()
            && change_marker(&opened_after) == change_marker(&opened_before)
            && change_marker(&path_after) == change_marker(&opened_before)
            && u64::try_from(bytes.len()).context("artifact read length exceeds u64")?
                == opened_before.len(),
        "artifact changed while reading {}",
        path.display()
    );
    Ok(ArtifactSnapshot { bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn a_bounded_snapshot_returns_the_bytes_and_refuses_to_exceed_its_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("manifest.json");
        fs::write(&path, b"{\"schema_version\":1}").unwrap();
        let snapshot = read_artifact_snapshot(&path, 1024).unwrap();
        assert_eq!(snapshot.bytes, b"{\"schema_version\":1}");
        assert!(read_artifact_snapshot(&path, 1).is_err());
        assert!(read_artifact_snapshot(&path, 0).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_snapshot_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        let link = directory.path().join("link.json");
        fs::write(&target, b"{}").unwrap();
        symlink(&target, &link).unwrap();
        assert!(read_artifact_snapshot(&link, 1024).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn path_producer_publication_seals_bytes_away_from_a_retained_writer() {
        use std::io::{Seek, SeekFrom, Write};

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        let staging = ArtifactStaging::new_for_path_producer(&destination).unwrap();
        let mut producer = fs::File::options()
            .write(true)
            .create_new(true)
            .open(staging.producer_path())
            .unwrap();
        producer.write_all(b"original").unwrap();
        producer.flush().unwrap();
        let published = staging.publish().unwrap();

        producer.seek(SeekFrom::Start(0)).unwrap();
        producer.write_all(b"mutated!").unwrap();
        producer.flush().unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"original");
        assert_eq!(published.destination, destination);
    }
}
