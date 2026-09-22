//! Machine topology capture for configuration derivation.
//!
//! A profile records the devices, host memory and interconnect the derivation
//! rules in `configure` decide against. Profiles persist as JSON and reload
//! only on the machine that produced them, keyed by the same
//! `HardwareFingerprint` the calibration caches use.

use anyhow::{Context, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::probe::{
    CudaComputeCapability, DeviceBackend, HardwareFingerprint, ResourceSnapshot,
    describes_same_machine,
};

pub const TOPOLOGY_PROFILE_SCHEMA_VERSION: u32 = 1;
const MAX_ENUMERATED_DEVICES: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum InterconnectLevel {
    SingleDevice,
    /// More than one device is present; the link class is not measured yet.
    Unknown,
    PciExpressPeer,
    Nvlink,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyDevice {
    pub ordinal: usize,
    pub backend: DeviceBackend,
    pub name: Option<String>,
    pub total_memory_bytes: Option<u64>,
    pub compute_capability: Option<CudaComputeCapability>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyProfile {
    pub schema_version: u32,
    pub fingerprint: HardwareFingerprint,
    pub host_memory_total_bytes: Option<u64>,
    pub devices: Vec<TopologyDevice>,
    pub interconnect: InterconnectLevel,
    pub storage_bytes_per_second: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TopologyProfileAbsence {
    NotFound(PathBuf),
    ForeignHost {
        path: PathBuf,
        recorded_device: Option<String>,
        current_device: Option<String>,
    },
    StaleSchema {
        path: PathBuf,
        recorded: u32,
    },
}

impl TopologyProfile {
    pub fn capture(primary: &Device) -> Self {
        let fingerprint = HardwareFingerprint::collect(primary);
        let host_memory_total_bytes = ResourceSnapshot::capture(None).host_pool_total_bytes();
        let devices = enumerate_devices(&fingerprint);
        let interconnect = if devices.len() > 1 {
            InterconnectLevel::Unknown
        } else {
            InterconnectLevel::SingleDevice
        };
        Self {
            schema_version: TOPOLOGY_PROFILE_SCHEMA_VERSION,
            fingerprint,
            host_memory_total_bytes,
            devices,
            interconnect,
            storage_bytes_per_second: None,
        }
    }

    pub fn with_storage_bandwidth(mut self, bytes_per_second: Option<u64>) -> Self {
        self.storage_bytes_per_second = bytes_per_second;
        self
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = serde_json::to_vec(self)
            .with_context(|| format!("serialize topology profile {}", path.display()))?;
        std::fs::write(path, &bytes)
            .with_context(|| format!("write topology profile {}", path.display()))
    }

    pub fn load(path: &Path, primary: &Device) -> Result<Result<Self, TopologyProfileAbsence>> {
        if !path.is_file() {
            return Ok(Err(TopologyProfileAbsence::NotFound(path.to_path_buf())));
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("read topology profile {}", path.display()))?;
        let recorded = Self::from_json(&bytes)
            .with_context(|| format!("invalid topology profile {}", path.display()))?;
        if recorded.schema_version != TOPOLOGY_PROFILE_SCHEMA_VERSION {
            return Ok(Err(TopologyProfileAbsence::StaleSchema {
                path: path.to_path_buf(),
                recorded: recorded.schema_version,
            }));
        }
        let current = HardwareFingerprint::collect(primary);
        if !describes_same_machine(&recorded.fingerprint, &current) {
            return Ok(Err(TopologyProfileAbsence::ForeignHost {
                path: path.to_path_buf(),
                recorded_device: recorded.fingerprint.device_name.clone(),
                current_device: current.device_name.clone(),
            }));
        }
        Ok(Ok(recorded))
    }

    fn from_json(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("parse topology profile")
    }
}

fn enumerate_devices(primary: &HardwareFingerprint) -> Vec<TopologyDevice> {
    let mut devices = Vec::new();
    if primary.backend != DeviceBackend::Cuda {
        return devices;
    }
    for ordinal in 0..MAX_ENUMERATED_DEVICES {
        let Ok(device) = Device::new_cuda(ordinal) else {
            break;
        };
        let fingerprint = HardwareFingerprint::collect(&device);
        devices.push(TopologyDevice {
            ordinal,
            backend: fingerprint.backend,
            name: fingerprint.device_name,
            total_memory_bytes: fingerprint.device_total_memory_bytes,
            compute_capability: fingerprint.cuda_compute_capability,
        });
    }
    devices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::HARDWARE_FINGERPRINT_SCHEMA_VERSION;

    fn cpu_profile() -> TopologyProfile {
        TopologyProfile::capture(&Device::Cpu)
    }

    #[test]
    fn cpu_capture_records_the_host_without_devices() {
        let profile = cpu_profile();
        assert_eq!(profile.schema_version, TOPOLOGY_PROFILE_SCHEMA_VERSION);
        assert_eq!(
            profile.fingerprint.schema_version,
            HARDWARE_FINGERPRINT_SCHEMA_VERSION
        );
        assert!(profile.host_memory_total_bytes.is_some());
        assert!(profile.devices.is_empty());
        assert_eq!(profile.interconnect, InterconnectLevel::SingleDevice);
        assert!(profile.storage_bytes_per_second.is_none());
    }

    #[test]
    fn profile_round_trips_through_disk_on_the_same_machine() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("topology.json");
        let profile = cpu_profile().with_storage_bandwidth(Some(1_600_000_000));
        profile.save(&path).unwrap();
        let loaded = TopologyProfile::load(&path, &Device::Cpu).unwrap().unwrap();
        assert_eq!(loaded, profile);
    }

    #[test]
    fn a_missing_profile_reports_not_found() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("absent.json");
        assert_eq!(
            TopologyProfile::load(&path, &Device::Cpu)
                .unwrap()
                .unwrap_err(),
            TopologyProfileAbsence::NotFound(path.clone())
        );
    }

    #[test]
    fn a_profile_from_another_machine_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("foreign.json");
        cpu_profile().save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut edited: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        edited["fingerprint"]["logical_cpu_count"] = 1_000_000.into();
        std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
        match TopologyProfile::load(&path, &Device::Cpu)
            .unwrap()
            .unwrap_err()
        {
            TopologyProfileAbsence::ForeignHost { .. } => {}
            other => panic!("expected a foreign host, got {other:?}"),
        }
    }

    #[test]
    fn a_stale_schema_version_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("stale.json");
        cpu_profile().save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut edited: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        edited["schema_version"] = (TOPOLOGY_PROFILE_SCHEMA_VERSION + 1).into();
        std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
        match TopologyProfile::load(&path, &Device::Cpu)
            .unwrap()
            .unwrap_err()
        {
            TopologyProfileAbsence::StaleSchema { recorded, .. } => {
                assert_eq!(recorded, TOPOLOGY_PROFILE_SCHEMA_VERSION + 1)
            }
            other => panic!("expected a stale schema, got {other:?}"),
        }
    }

    #[test]
    fn fingerprints_of_one_machine_agree_with_themselves() {
        let profile = cpu_profile();
        assert!(describes_same_machine(
            &profile.fingerprint,
            &profile.fingerprint
        ));
    }
}
