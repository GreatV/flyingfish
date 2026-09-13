//! Explicit benchmark-only file-cache preparation. Never a default policy action.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColdCacheObservation {
    pub files: usize,
    pub advised_bytes: u64,
    pub checked_pages: u64,
    pub resident_pages: u64,
}

/// Drop only clean pages belonging to these files, then inspect residency
/// without faulting payloads in. Other readers can repopulate pages afterward;
/// this is a verified starting observation, not a cache reservation.
#[cfg(target_os = "linux")]
pub fn evict_and_verify(paths: &[PathBuf]) -> Result<ColdCacheObservation> {
    use anyhow::{Context, ensure};
    use std::{fs::File, os::fd::AsRawFd};
    ensure!(
        !paths.is_empty(),
        "cold-cache preparation requires checkpoint files"
    );
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(page > 0, "cannot determine page size");
    let page = usize::try_from(page)?;
    let mut report = ColdCacheObservation {
        files: paths.len(),
        advised_bytes: 0,
        checked_pages: 0,
        resident_pages: 0,
    };
    for path in paths {
        let file = File::open(path)
            .with_context(|| format!("cannot open cache-control file {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file(),
            "cache-control target is not a regular file"
        );
        let code =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        ensure!(
            code == 0,
            "file-cache advice failed: {}",
            std::io::Error::from_raw_os_error(code)
        );
        report.advised_bytes = report
            .advised_bytes
            .checked_add(metadata.len())
            .context("cache-control size overflow")?;
    }
    for path in paths {
        let file = File::open(path)?;
        if file.metadata()?.len() == 0 {
            continue;
        }
        let mapping = unsafe { memmap2::MmapOptions::new().map(&file) }?;
        let mut residency = vec![0u8; mapping.len().div_ceil(page)];
        let status = unsafe {
            libc::mincore(
                mapping.as_ptr().cast_mut().cast(),
                mapping.len(),
                residency.as_mut_ptr(),
            )
        };
        ensure!(
            status == 0,
            "file residency query failed: {}",
            std::io::Error::last_os_error()
        );
        report.checked_pages = report
            .checked_pages
            .checked_add(residency.len() as u64)
            .context("page count overflow")?;
        report.resident_pages = report
            .resident_pages
            .checked_add(residency.iter().filter(|v| **v & 1 != 0).count() as u64)
            .context("resident page count overflow")?;
    }
    ensure!(
        report.resident_pages == 0,
        "cold file-cache state was not established: {} of {} pages remain resident (dirty or concurrently used pages are not forcibly reclaimed)",
        report.resident_pages,
        report.checked_pages
    );
    Ok(report)
}

#[cfg(not(target_os = "linux"))]
pub fn evict_and_verify(_paths: &[PathBuf]) -> Result<ColdCacheObservation> {
    anyhow::bail!("verified cold file-cache preparation is available only on Linux")
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    #[test]
    fn cache_preparation_preserves_file_contents_and_rejects_nonfiles() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("weights");
        std::fs::write(&path, vec![17u8; 1024 * 1024]).unwrap();
        std::fs::File::open(&path).unwrap().sync_all().unwrap();
        let before = std::fs::read(&path).unwrap();
        match evict_and_verify(std::slice::from_ref(&path)) {
            Ok(report) => {
                assert_eq!(report.resident_pages, 0);
                assert_eq!(report.advised_bytes, 1024 * 1024);
            }
            Err(error) => assert!(
                error
                    .to_string()
                    .contains("cold file-cache state was not established")
            ),
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(evict_and_verify(&[root.path().to_owned()]).is_err());
        assert!(evict_and_verify(&[]).is_err());
    }
}
