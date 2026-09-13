//! Direct comparison of large outputs using fixed-size read buffers.
use super::{FileStat, identity::change_marker};
use anyhow::{Context, Result, ensure};
use std::{fs::File, io::Read, path::Path};

const COMPARE_BUFFER_BYTES: usize = 64 * 1024;

struct Input<'a> {
    path: &'a Path,
    file: File,
    before: FileStat,
}

impl<'a> Input<'a> {
    fn open(path: &'a Path, expected_bytes: u64) -> Result<Self> {
        let before = FileStat::of_path(path)?;
        ensure!(
            before.file_type().is_file(),
            "output is not a regular non-symlink file: {}",
            path.display()
        );
        ensure!(
            before.len() == expected_bytes,
            "output {} is {} bytes, expected {expected_bytes}",
            path.display(),
            before.len()
        );
        let input = Self {
            path,
            file: File::open(path)?,
            before,
        };
        input.check_unchanged()?;
        Ok(input)
    }

    fn check_unchanged(&self) -> Result<()> {
        let opened = FileStat::of_file(&self.file)?;
        let after = FileStat::of_path(self.path)?;
        for stat in [&opened, &after] {
            ensure!(
                stat.file_type().is_file()
                    && self.before.identifies_same_file_as(stat)
                    && self.before.len() == stat.len()
                    && self.before.modified()? == stat.modified()?
                    && change_marker(&self.before) == change_marker(stat),
                "output changed while comparing: {}",
                self.path.display()
            );
        }
        Ok(())
    }
}

/// Compare complete output bytes without imposing a JSON-report size limit.
/// Both lengths must match their recorded observations. Invalid or changing
/// files are errors; different stable outputs return `false`.
pub fn compare_artifact_files(
    left: &Path,
    left_bytes: u64,
    right: &Path,
    right_bytes: u64,
) -> Result<bool> {
    let mut left =
        Input::open(left, left_bytes).with_context(|| format!("open output {}", left.display()))?;
    let mut right = Input::open(right, right_bytes)
        .with_context(|| format!("open output {}", right.display()))?;
    let mut equal = left_bytes == right_bytes;
    if equal {
        let mut a = vec![0; COMPARE_BUFFER_BYTES];
        let mut b = vec![0; COMPARE_BUFFER_BYTES];
        let mut remaining = left_bytes;
        while remaining > 0 {
            let count = remaining.min(COMPARE_BUFFER_BYTES as u64) as usize;
            left.file
                .read_exact(&mut a[..count])
                .context("read baseline output")?;
            right
                .file
                .read_exact(&mut b[..count])
                .context("read candidate output")?;
            if a[..count] != b[..count] {
                equal = false;
                break;
            }
            remaining -= count as u64;
        }
    }
    left.check_unchanged()?;
    right.check_unchanged()?;
    Ok(equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn large_outputs_check_middle_tail_and_recorded_lengths() {
        let dir = tempfile::tempdir().unwrap();
        let left = dir.path().join("a");
        let right = dir.path().join("b");
        let size = 5 * 1024 * 1024 + 7;
        let a = File::create(&left).unwrap();
        a.set_len(size).unwrap();
        let mut b = File::create(&right).unwrap();
        b.set_len(size).unwrap();
        assert!(compare_artifact_files(&left, size, &right, size).unwrap());
        for offset in [COMPARE_BUFFER_BYTES as u64 + 1, size - 1] {
            b.seek(SeekFrom::Start(offset)).unwrap();
            b.write_all(&[1]).unwrap();
            assert!(!compare_artifact_files(&left, size, &right, size).unwrap());
            b.seek(SeekFrom::Start(offset)).unwrap();
            b.write_all(&[0]).unwrap();
        }
        b.set_len(size - 1).unwrap();
        assert!(compare_artifact_files(&left, size, &right, size).is_err());
        assert!(!compare_artifact_files(&left, size, &right, size - 1).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_outputs_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let link = dir.path().join("link");
        std::fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(compare_artifact_files(&target, 1, &link, 1).is_err());
    }
}
