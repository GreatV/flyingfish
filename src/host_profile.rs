//! Measured properties of the machine a run is actually on.
//!
//! Everything here is a number this host produced about itself. None of it is
//! a literal, because none of it transfers: `B_P` is whatever this PCIe link
//! delivers, `B_H` is whatever these cores and this memory deliver, and a
//! constant carrying one machine's answer would silently mis-schedule every
//! other machine. `ff bench io --profile local-interconnect` writes the
//! measurements; this reads them back and refuses any that were not taken
//! here.

use crate::{interconnect_benchmark::LocalIoBenchmarkReport, runtime::probe::HardwareFingerprint};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Where a run looks for its host profile when the operator names no path.
pub const DEFAULT_HOST_PROFILE_NAME: &str = "host-profile.json";

/// Why a profile is not in play. A run says which of these it is rather than
/// quietly proceeding as though it had measurements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostProfileAbsence {
    /// No profile file where one was looked for.
    NotFound(PathBuf),
    /// A profile that another machine produced. Its numbers describe that
    /// machine's links and cores, so they are not evidence about this one.
    ForeignHost {
        path: PathBuf,
        recorded_device: Option<String>,
        current_device: Option<String>,
    },
}

impl std::fmt::Display for HostProfileAbsence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(path) => {
                write!(
                    f,
                    "no host profile at {}; measure one with `ff bench io --profile \
                     local-interconnect`",
                    path.display()
                )
            }
            Self::ForeignHost {
                path,
                recorded_device,
                current_device,
            } => write!(
                f,
                "host profile {} was measured on {} but this host is {}; re-measure it here",
                path.display(),
                recorded_device
                    .as_deref()
                    .unwrap_or("an unidentified device"),
                current_device.as_deref().unwrap_or("unidentified"),
            ),
        }
    }
}

/// The transfer and host-evaluation rates a scheduling split is decided from,
/// both over the same logical expert bytes.
///
/// `B_H` is reported two ways because only one of them decides a split. A miss
/// has to be read out of the page cache whichever side handles it, so that
/// read is common to both branches and cancels; what differs is the transfer
/// against the evaluation. `host_evaluation_bytes_per_second` is the marginal
/// rate and is what `host_share` uses. `host_service_bytes_per_second` adds the
/// shared read back in and is the end-to-end cost of one expert on the host,
/// which is the right number for asking whether the host path is worth having
/// at all, and the wrong one for dividing a miss set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertBandwidths {
    pub transfer_bytes_per_second: f64,
    pub host_evaluation_bytes_per_second: f64,
    pub host_service_bytes_per_second: f64,
}

impl ExpertBandwidths {
    /// `B_P / B_H`. At or above one the host is the slower path for every
    /// expert and should take none of them.
    pub fn transfer_over_host(&self) -> f64 {
        self.transfer_bytes_per_second / self.host_evaluation_bytes_per_second
    }

    /// The share of a miss set the host should evaluate, from the balance of
    /// the two branches: transferring `q` of `m` experts takes `qS / B_P`
    /// while the host evaluates the rest against the bandwidth the transfer
    /// leaves it, `B_H - B_P`. Equating them gives `q = m * B_P / B_H`, so the
    /// host's share is `1 - B_P / B_H`, and nothing when that is not positive.
    pub fn host_share(&self) -> f64 {
        (1.0 - self.transfer_over_host()).max(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct HostProfile {
    path: PathBuf,
    report: Box<LocalIoBenchmarkReport>,
}

impl HostProfile {
    /// Read a profile and accept it only if this machine is the one that
    /// produced it.
    pub fn load(
        path: &Path,
        device: &candle_core::Device,
    ) -> Result<Result<Self, HostProfileAbsence>> {
        if !path.is_file() {
            return Ok(Err(HostProfileAbsence::NotFound(path.to_path_buf())));
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read host profile {}", path.display()))?;
        let report = LocalIoBenchmarkReport::from_json(&bytes)
            .with_context(|| format!("invalid host profile {}", path.display()))?;
        let current = HardwareFingerprint::collect(device);
        if !describes_same_host(&report.fingerprint, &current) {
            return Ok(Err(HostProfileAbsence::ForeignHost {
                path: path.to_path_buf(),
                recorded_device: report.fingerprint.device_name.clone(),
                current_device: current.device_name.clone(),
            }));
        }
        Ok(Ok(Self {
            path: path.to_path_buf(),
            report: Box::new(report),
        }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn fingerprint(&self) -> &HardwareFingerprint {
        &self.report.fingerprint
    }

    pub fn expert_bandwidths(&self) -> ExpertBandwidths {
        let expert = &self.report.glm_routed_expert;
        ExpertBandwidths {
            transfer_bytes_per_second: expert
                .pinned_host_to_device_b_p
                .statistics
                .median_bytes_per_second,
            host_evaluation_bytes_per_second: expert.geometry.logical_transfer_bytes as f64 * 1e9
                / expert
                    .host_evaluation_b_h
                    .compute_statistics
                    .median_elapsed_ns as f64,
            host_service_bytes_per_second: expert
                .host_evaluation_b_h
                .median_service_bytes_per_second,
        }
    }
}

/// Whether two fingerprints describe the same machine.
///
/// The CUDA device UUID is the strongest identifier available and is what a
/// transfer rate actually belongs to; the host fields catch a profile carried
/// between machines that happen to hold the same card model.
fn describes_same_host(recorded: &HardwareFingerprint, current: &HardwareFingerprint) -> bool {
    crate::runtime::probe::describes_same_machine(recorded, current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn recorded() -> LocalIoBenchmarkReport {
        LocalIoBenchmarkReport::from_json(include_bytes!("testdata/local-io-report.json")).unwrap()
    }

    #[test]
    fn a_missing_profile_is_named_rather_than_assumed() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(DEFAULT_HOST_PROFILE_NAME);
        let absence = HostProfile::load(&path, &Device::Cpu).unwrap().unwrap_err();
        assert_eq!(absence, HostProfileAbsence::NotFound(path.clone()));
        assert!(absence.to_string().contains("ff bench io"));
    }

    #[test]
    fn the_split_follows_the_two_measured_rates() {
        // Host strictly faster than the link: it takes the share the link
        // cannot carry.
        let faster_host = ExpertBandwidths {
            transfer_bytes_per_second: 20.0,
            host_evaluation_bytes_per_second: 80.0,
            host_service_bytes_per_second: 10.0,
        };
        assert_eq!(faster_host.transfer_over_host(), 0.25);
        assert_eq!(faster_host.host_share(), 0.75);

        // A slow end-to-end service rate does not by itself deny the host a
        // share: the read it includes is one both branches pay, so the split
        // follows the marginal evaluation rate above it.
        assert_eq!(faster_host.host_share(), 0.75);

        // The rates this host reports with the fused kernels. Evaluating an
        // expert is faster than moving it, so the host takes about half.
        let measured = ExpertBandwidths {
            transfer_bytes_per_second: 18.937 * 1024.0 * 1024.0 * 1024.0,
            host_evaluation_bytes_per_second: 36.702 * 1024.0 * 1024.0 * 1024.0,
            host_service_bytes_per_second: 8.275 * 1024.0 * 1024.0 * 1024.0,
        };
        assert!((measured.transfer_over_host() - 0.516).abs() < 0.005);
        assert!((measured.host_share() - 0.484).abs() < 0.005);

        // A link faster than the host evaluation takes everything.
        let faster_link = ExpertBandwidths {
            transfer_bytes_per_second: 80.0,
            host_evaluation_bytes_per_second: 20.0,
            host_service_bytes_per_second: 10.0,
        };
        assert_eq!(faster_link.host_share(), 0.0);
    }

    #[test]
    fn a_profile_reads_back_the_rates_it_recorded() {
        let report = recorded();
        let expected_b_p = report
            .glm_routed_expert
            .pinned_host_to_device_b_p
            .statistics
            .median_bytes_per_second;
        let profile = HostProfile {
            path: PathBuf::from(DEFAULT_HOST_PROFILE_NAME),
            report: Box::new(report),
        };
        assert_eq!(
            profile.expert_bandwidths().transfer_bytes_per_second,
            expected_b_p
        );
        assert!(profile.expert_bandwidths().host_evaluation_bytes_per_second > 0.0);
    }
}
