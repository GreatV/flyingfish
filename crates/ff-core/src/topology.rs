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

use crate::probe::{CudaComputeCapability, DeviceBackend, HardwareFingerprint, ResourceSnapshot};

pub const TOPOLOGY_PROFILE_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkClass {
    Nvlink,
    PciExpress,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerLink {
    pub a: usize,
    pub b: usize,
    pub reachable: bool,
    pub link_class: Option<LinkClass>,
}

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
    pub cuda_device_uuid: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyProfile {
    pub schema_version: u32,
    pub fingerprint: HardwareFingerprint,
    pub host_memory_total_bytes: Option<u64>,
    /// The cgroup limit at capture time: a container whose limit changes
    /// invalidates the recorded pool total even though the machine is the
    /// same.
    pub cgroup_memory_limit_bytes: Option<u64>,
    pub devices: Vec<TopologyDevice>,
    pub peer_links: Vec<PeerLink>,
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
    CgroupLimitChanged {
        path: PathBuf,
        recorded: Option<u64>,
        current: Option<u64>,
    },
}

impl TopologyProfile {
    pub fn capture(primary: &Device) -> Self {
        let fingerprint = HardwareFingerprint::collect(primary);
        let snapshot = ResourceSnapshot::capture(None);
        let host_memory_total_bytes = snapshot.host_pool_total_bytes();
        let cgroup_memory_limit_bytes = match snapshot.cgroup_v2_memory_limit {
            Some(crate::probe::CgroupMemoryLimit::Bytes(bytes)) => Some(bytes),
            _ => None,
        };
        let devices = enumerate_devices(&fingerprint);
        let peer_links = capture_peer_links(devices.len());
        let interconnect = synthesize_interconnect(devices.len(), &peer_links);
        Self {
            schema_version: TOPOLOGY_PROFILE_SCHEMA_VERSION,
            fingerprint,
            host_memory_total_bytes,
            cgroup_memory_limit_bytes,
            devices,
            peer_links,
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
        // The schema version is read loosely first: a newer schema's fields
        // would fail the strict decode before the version check could name
        // the real reason.
        let version = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|value| value.get("schema_version").and_then(|v| v.as_u64()))
            .unwrap_or(0) as u32;
        if version != TOPOLOGY_PROFILE_SCHEMA_VERSION {
            return Ok(Err(TopologyProfileAbsence::StaleSchema {
                path: path.to_path_buf(),
                recorded: version,
            }));
        }
        let recorded = Self::from_json(&bytes)
            .with_context(|| format!("invalid topology profile {}", path.display()))?;
        let current_snapshot = ResourceSnapshot::capture(None);
        let current_limit = match current_snapshot.cgroup_v2_memory_limit {
            Some(crate::probe::CgroupMemoryLimit::Bytes(bytes)) => Some(bytes),
            _ => None,
        };
        if recorded.cgroup_memory_limit_bytes != current_limit {
            return Ok(Err(TopologyProfileAbsence::CgroupLimitChanged {
                path: path.to_path_buf(),
                recorded: recorded.cgroup_memory_limit_bytes,
                current: current_limit,
            }));
        }
        // A CUDA_VISIBLE_DEVICES reorder keeps the machine but renumbers the
        // ordinals, so the recorded per-ordinal capacity map no longer
        // applies; the derivation recaptures under the new mapping.
        let current = HardwareFingerprint::collect(primary);
        let current_devices = enumerate_devices(&current);
        if device_uuid_sequence(&recorded.devices) != device_uuid_sequence(&current_devices) {
            return Ok(Err(TopologyProfileAbsence::ForeignHost {
                path: path.to_path_buf(),
                recorded_device: recorded
                    .devices
                    .first()
                    .and_then(|device| device.name.clone()),
                current_device: current_devices
                    .first()
                    .and_then(|device| device.name.clone()),
            }));
        }
        if !same_host(&recorded.fingerprint, &current, &current_devices) {
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
    let mut ordinal = 0;
    while let Ok(device) = Device::new_cuda(ordinal) {
        let fingerprint = HardwareFingerprint::collect(&device);
        devices.push(TopologyDevice {
            ordinal,
            backend: fingerprint.backend,
            name: fingerprint.device_name,
            total_memory_bytes: fingerprint.device_total_memory_bytes,
            compute_capability: fingerprint.cuda_compute_capability,
            cuda_device_uuid: fingerprint.cuda_device_uuid,
        });
        ordinal += 1;
    }
    devices
}

fn device_uuid_sequence(devices: &[TopologyDevice]) -> Vec<Option<String>> {
    devices
        .iter()
        .map(|device| device.cuda_device_uuid.clone())
        .collect()
}

#[cfg(feature = "cuda")]
fn capture_peer_links(device_count: usize) -> Vec<PeerLink> {
    if device_count < 2 {
        return Vec::new();
    }
    let labels = nvidia_smi_link_labels(device_count);
    let mut links = Vec::new();
    for a in 0..device_count {
        for b in (a + 1)..device_count {
            let reachable = crate::probe::can_access_peer(a as u32, b as u32)
                .or_else(|| crate::probe::can_access_peer(b as u32, a as u32))
                .unwrap_or(false);
            links.push(PeerLink {
                a,
                b,
                reachable,
                link_class: labels[a * device_count + b],
            });
        }
    }
    links
}

#[cfg(not(feature = "cuda"))]
fn capture_peer_links(_device_count: usize) -> Vec<PeerLink> {
    Vec::new()
}

/// The synthesis records link facts only. Link labels come from the
/// topology table, and `nvidia-smi topo -m` was measured not to predict
/// concurrent transfer behaviour on this fleet, so nothing downstream may
/// read a bandwidth figure out of these levels.
fn synthesize_interconnect(device_count: usize, links: &[PeerLink]) -> InterconnectLevel {
    if device_count < 2 {
        return InterconnectLevel::SingleDevice;
    }
    let any_nvlink = links
        .iter()
        .any(|link| link.reachable && link.link_class == Some(LinkClass::Nvlink));
    if any_nvlink {
        return InterconnectLevel::Nvlink;
    }
    let any_pcie_peer = links
        .iter()
        .any(|link| link.reachable && link.link_class == Some(LinkClass::PciExpress))
        || links.iter().any(|link| link.reachable);
    if any_pcie_peer {
        return InterconnectLevel::PciExpressPeer;
    }
    InterconnectLevel::Unknown
}

/// One link class per ordinal pair, read from `nvidia-smi topo -m`'s matrix.
/// Unreadable cells stay `None` rather than guessing.
#[cfg(feature = "cuda")]
fn nvidia_smi_link_labels(device_count: usize) -> Vec<Option<LinkClass>> {
    let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["topo", "-m"])
        .output()
    else {
        return vec![None; device_count * device_count];
    };
    let Ok(text) = String::from_utf8(output.stdout) else {
        return vec![None; device_count * device_count];
    };
    parse_topo_matrix(&text, device_count)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn parse_topo_matrix(text: &str, device_count: usize) -> Vec<Option<LinkClass>> {
    let mut labels = vec![None; device_count * device_count];
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<Option<usize>> = lines
        .next()
        .map(|line| {
            line.split_whitespace()
                .filter_map(|token| token.strip_prefix("GPU"))
                .map(|suffix| suffix.parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    for line in lines {
        let mut tokens = line.split_whitespace();
        let Some(row) = tokens
            .next()
            .and_then(|token| token.strip_prefix("GPU"))
            .and_then(|suffix| suffix.parse::<usize>().ok())
        else {
            continue;
        };
        for (column, token) in tokens.enumerate() {
            let Some(column_device) = header.get(column).copied().flatten() else {
                break;
            };
            if row >= device_count || column_device >= device_count {
                continue;
            }
            let class = match token {
                value if value.starts_with("NV") => Some(LinkClass::Nvlink),
                "PIX" | "PHB" | "NODE" | "SYS" => Some(LinkClass::PciExpress),
                _ => None,
            };
            labels[row * device_count + column_device] = class;
        }
    }
    labels
}

/// Whether a profile recorded on this machine still describes it.
///
/// The profile is keyed to the host rather than to one card: the recorded
/// card only has to be among the devices the machine still enumerates, so a
/// run selecting a different ordinal reuses it.
fn same_host(
    recorded: &HardwareFingerprint,
    current: &HardwareFingerprint,
    current_devices: &[TopologyDevice],
) -> bool {
    recorded.validate().is_ok()
        && current.validate().is_ok()
        && recorded.backend == current.backend
        && recorded.architecture == current.architecture
        && recorded.operating_system == current.operating_system
        && recorded.logical_cpu_count == current.logical_cpu_count
        && match &recorded.cuda_device_uuid {
            Some(recorded_uuid) => current_devices
                .iter()
                .find(|device| device.cuda_device_uuid.as_ref() == Some(recorded_uuid))
                .is_some_and(|device| {
                    device.name.as_deref() == recorded.device_name.as_deref()
                        && device.total_memory_bytes == recorded.device_total_memory_bytes
                }),
            None => {
                recorded.device_name == current.device_name
                    && recorded.device_total_memory_bytes == current.device_total_memory_bytes
                    && current.backend != DeviceBackend::Cuda
            }
        }
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
        assert!(same_host(
            &profile.fingerprint,
            &profile.fingerprint,
            &profile.devices
        ));
    }

    #[test]
    fn interconnect_levels_follow_the_measured_links() {
        let link = |reachable: bool, class: Option<LinkClass>| PeerLink {
            a: 0,
            b: 1,
            reachable,
            link_class: class,
        };
        assert_eq!(
            synthesize_interconnect(1, &[link(true, Some(LinkClass::Nvlink))]),
            InterconnectLevel::SingleDevice
        );
        assert_eq!(
            synthesize_interconnect(
                2,
                &[
                    link(true, Some(LinkClass::Nvlink)),
                    link(true, Some(LinkClass::PciExpress))
                ]
            ),
            InterconnectLevel::Nvlink
        );
        assert_eq!(
            synthesize_interconnect(2, &[link(true, None)]),
            InterconnectLevel::PciExpressPeer
        );
        assert_eq!(
            synthesize_interconnect(2, &[link(false, Some(LinkClass::Nvlink))]),
            InterconnectLevel::Unknown
        );
    }

    #[test]
    fn topo_matrix_parses_nvlink_and_pcie_cells() {
        let text = "\
            \tGPU0\tGPU1\tGPU2\tCPU Affinity\tNUMA Affinity\n\
            GPU0\tX\tNV1\tPIX\t0-31\t0\n\
            GPU1\tNV1\tX\tPHB\t0-31\t0\n\
            GPU2\tPIX\tPHB\tX\t0-31\t0\n";
        let labels = parse_topo_matrix(text, 3);
        let (gpu0, gpu1, gpu2) = (0usize, 1usize, 2usize);
        assert_eq!(labels[gpu0 * 3 + gpu1], Some(LinkClass::Nvlink));
        assert_eq!(labels[gpu1 * 3 + gpu0], Some(LinkClass::Nvlink));
        assert_eq!(labels[gpu0 * 3 + gpu2], Some(LinkClass::PciExpress));
        assert_eq!(labels[gpu1 * 3 + gpu2], Some(LinkClass::PciExpress));
        assert_eq!(labels[gpu2 * 3 + gpu2], None);
    }
}
