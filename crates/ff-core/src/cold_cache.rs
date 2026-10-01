//! Explicit benchmark-only file-cache preparation. Never a default policy action.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Measurements defining one trial’s residual-cache bound.
pub struct ColdCacheEvidence {
    pub read_bytes_per_second: f64,
    pub shortest_phase_seconds: f64,
    pub phase_source: String,
    pub first_tensors: Vec<ColdCacheTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// A checkpoint tensor read during trial startup.
pub struct ColdCacheTensor {
    pub name: String,
    pub file: PathBuf,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Resident pages and the first eight byte offsets in one file.
pub struct ColdFileResidency {
    pub file: PathBuf,
    pub resident_pages: u64,
    pub first_offsets: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColdCacheObservation {
    pub files: usize,
    pub advised_bytes: u64,
    pub checked_pages: u64,
    pub resident_pages: u64,
    pub resident_bytes: u64,
    pub residency: Vec<ColdFileResidency>,
    pub evidence: Option<ColdCacheEvidence>,
    pub first_tensor_resident_pages: u64,
    pub resident_read_seconds: Option<f64>,
    pub allowed_resident_read_seconds: Option<f64>,
}

impl ColdCacheObservation {
    fn verify(&mut self) -> Result<()> {
        use anyhow::ensure;
        if let Some(evidence) = &self.evidence {
            self.resident_read_seconds =
                Some(self.resident_bytes as f64 / evidence.read_bytes_per_second);
            self.allowed_resident_read_seconds = Some(evidence.shortest_phase_seconds * 0.001);
            ensure!(
                self.first_tensor_resident_pages == 0,
                "cold file-cache criterion failed: {} resident pages intersect first-loaded tensors; {:?}",
                self.first_tensor_resident_pages,
                self.residency
            );
            ensure!(
                self.resident_bytes as f64 / evidence.read_bytes_per_second
                    <= evidence.shortest_phase_seconds * 0.001,
                "cold file-cache criterion failed: {} resident bytes cost {:.9} s, exceeding {:.9} s (0.1% of shortest phase); {:?}",
                self.resident_bytes,
                self.resident_read_seconds.unwrap(),
                self.allowed_resident_read_seconds.unwrap(),
                self.residency
            );
        } else if self.resident_pages != 0 {
            anyhow::bail!(
                "cold file-cache state was not established: {} of {} pages remain resident; measured FF_COLD_CACHE_EVIDENCE is required; {:?}",
                self.resident_pages,
                self.checked_pages,
                self.residency
            );
        }
        Ok(())
    }
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
    let evidence = std::env::var_os("FF_COLD_CACHE_EVIDENCE")
        .map(|path| -> Result<ColdCacheEvidence> {
            let path = std::fs::canonicalize(path).context("cannot resolve FF_COLD_CACHE_EVIDENCE")?;
            eprintln!("cold-cache evidence: {}", path.display());
            let mut evidence: ColdCacheEvidence = serde_json::from_slice(&std::fs::read(&path)?)
                .context("invalid cold-cache evidence JSON")?;
            ensure!(
                evidence.read_bytes_per_second.is_finite() && evidence.read_bytes_per_second > 0.0
                    && evidence.shortest_phase_seconds.is_finite() && evidence.shortest_phase_seconds > 0.0
                    && !evidence.phase_source.trim().is_empty() && !evidence.first_tensors.is_empty(),
                "cold-cache evidence needs positive measured throughput/phase and first-loaded tensors"
            );
            for tensor in &mut evidence.first_tensors {
                tensor.file = std::fs::canonicalize(&tensor.file)?;
                ensure!(paths.contains(&tensor.file), "first-loaded tensor {} is outside the cold-cache file set", tensor.name);
                let file_bytes = std::fs::metadata(&tensor.file)?.len();
                ensure!(tensor.bytes > 0 && tensor.offset.checked_add(tensor.bytes).is_some_and(|end| end <= file_bytes), "invalid first-loaded tensor range for {}", tensor.name);
            }
            Ok(evidence)
        }).transpose()?;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(page > 0, "cannot determine page size");
    let page = usize::try_from(page)?;
    let mut report = ColdCacheObservation {
        files: paths.len(),
        advised_bytes: 0,
        checked_pages: 0,
        resident_pages: 0,
        resident_bytes: 0,
        residency: Vec::new(),
        evidence,
        first_tensor_resident_pages: 0,
        resident_read_seconds: None,
        allowed_resident_read_seconds: None,
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
        let resident = residency.iter().filter(|value| **value & 1 != 0).count();
        report.resident_pages = report
            .resident_pages
            .checked_add(resident as u64)
            .context("resident page count overflow")?;
        if resident > 0 {
            let offsets = residency
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (*value & 1 != 0).then_some(index * page))
                .take(8)
                .collect::<Vec<_>>();
            report.residency.push(ColdFileResidency {
                file: path.clone(),
                resident_pages: resident as u64,
                first_offsets: offsets.into_iter().map(|offset| offset as u64).collect(),
            });
            if let Some(evidence) = &report.evidence {
                for (index, _) in residency
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value & 1 != 0)
                {
                    let offset = (index * page) as u64;
                    if evidence.first_tensors.iter().any(|tensor| {
                        tensor.file == *path
                            && offset < tensor.offset + tensor.bytes
                            && offset + page as u64 > tensor.offset
                    }) {
                        report.first_tensor_resident_pages += 1;
                    }
                }
            }
        }
    }
    report.resident_bytes = report
        .resident_pages
        .checked_mul(page as u64)
        .context("resident byte count overflow")?;
    report.verify()?;
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
    fn measured_budget_accepts_small_residue_but_never_first_tensor_pages() {
        let mut report = ColdCacheObservation {
            files: 1,
            advised_bytes: 1 << 30,
            checked_pages: 1 << 18,
            resident_pages: 4,
            resident_bytes: 16 << 10,
            residency: Vec::new(),
            evidence: Some(ColdCacheEvidence {
                read_bytes_per_second: 2_900_000_000.0,
                shortest_phase_seconds: 50.0,
                phase_source: "measured-phase".into(),
                first_tensors: vec![ColdCacheTensor {
                    name: "first".into(),
                    file: "shard".into(),
                    offset: 0,
                    bytes: 4096,
                }],
            }),
            first_tensor_resident_pages: 0,
            resident_read_seconds: None,
            allowed_resident_read_seconds: None,
        };
        report.verify().unwrap();
        assert_eq!(report.allowed_resident_read_seconds, Some(0.05));
        assert!(report.resident_read_seconds.unwrap() < 0.000006);
        report.first_tensor_resident_pages = 1;
        assert!(
            report
                .verify()
                .unwrap_err()
                .to_string()
                .contains("first-loaded")
        );
        report.first_tensor_resident_pages = 0;
        report.resident_bytes = 145_000_001;
        assert!(
            report
                .verify()
                .unwrap_err()
                .to_string()
                .contains("exceeding")
        );
        report.evidence = None;
        assert!(
            report
                .verify()
                .unwrap_err()
                .to_string()
                .contains("FF_COLD_CACHE_EVIDENCE")
        );
    }

    #[test]
    fn mapped_pages_name_the_file_and_offset_in_the_failure() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("mapped-weights");
        std::fs::write(&path, vec![19u8; 1024 * 1024]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        file.sync_all().unwrap();
        let mapping = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
        std::hint::black_box(mapping[0]);
        let error = format!(
            "{:#}",
            evict_and_verify(std::slice::from_ref(&path)).unwrap_err()
        );
        assert!(error.contains("mapped-weights"));
        assert!(error.contains("resident_pages"));
        assert!(error.contains("first_offsets: [0"));
        assert_eq!(mapping[0], 19);
    }

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
            Err(error) => {
                assert!(format!("{error:#}").contains("cold file-cache state was not established"))
            }
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(evict_and_verify(&[root.path().to_owned()]).is_err());
        assert!(evict_and_verify(&[]).is_err());
    }
}
