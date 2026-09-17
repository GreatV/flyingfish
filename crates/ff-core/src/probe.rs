use anyhow::{Context, Result};
use candle_core::{Device, DeviceLocation};
use serde::{Deserialize, Serialize};
use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const HARDWARE_FINGERPRINT_SCHEMA_VERSION: u32 = 1;
pub const RESOURCE_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceBackend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintValueSource {
    ProcfsNvidiaDriver,
    CudaDriverApi,
    CudarcBuildBindings,
    CudaRuntimeLibraryBuild,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FingerprintUnavailableReason {
    ProcfsUnavailable,
    ProcfsNvidiaDriverParseFailed,
    CudaDriverApiQueryFailed,
    CudarcBindingVersionUnavailable,
    CudaRuntimeBuildIdentityUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaComputeCapability {
    pub major: u32,
    pub minor: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFingerprint {
    pub schema_version: u32,
    pub operating_system: String,
    #[serde(deserialize_with = "crate::required_option")]
    pub operating_system_version: Option<String>,
    pub architecture: String,
    pub logical_cpu_count: u64,
    pub backend: DeviceBackend,
    #[serde(deserialize_with = "crate::required_option")]
    pub device_name: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub device_total_memory_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub driver_version: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub runtime_version: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_compute_capability: Option<CudaComputeCapability>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_compute_capability_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_compute_capability_unavailable_reason: Option<FingerprintUnavailableReason>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_device_uuid: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_device_uuid_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_device_uuid_unavailable_reason: Option<FingerprintUnavailableReason>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_pci_bus_id: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_pci_bus_id_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_pci_bus_id_unavailable_reason: Option<FingerprintUnavailableReason>,
    #[serde(deserialize_with = "crate::required_option")]
    pub driver_version_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub driver_version_unavailable_reason: Option<FingerprintUnavailableReason>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_driver_api_version: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_driver_api_version_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_driver_api_version_unavailable_reason: Option<FingerprintUnavailableReason>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_binding_api_version: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_binding_api_version_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_binding_api_version_unavailable_reason: Option<FingerprintUnavailableReason>,
    #[serde(deserialize_with = "crate::required_option")]
    pub runtime_version_source: Option<FingerprintValueSource>,
    #[serde(deserialize_with = "crate::required_option")]
    pub runtime_version_unavailable_reason: Option<FingerprintUnavailableReason>,
}

impl HardwareFingerprint {
    pub fn collect(device: &Device) -> Self {
        hardware_fingerprint_with_source(device, &SystemProbeSource)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.operating_system.trim().is_empty(),
            "hardware fingerprint operating_system must not be empty"
        );
        anyhow::ensure!(
            !self.architecture.trim().is_empty(),
            "hardware fingerprint architecture must not be empty"
        );
        anyhow::ensure!(
            self.logical_cpu_count > 0,
            "hardware fingerprint logical_cpu_count must be non-zero"
        );
        for (name, value) in [
            (
                "operating_system_version",
                self.operating_system_version.as_deref(),
            ),
            ("device_name", self.device_name.as_deref()),
            ("driver_version", self.driver_version.as_deref()),
            ("runtime_version", self.runtime_version.as_deref()),
            ("cuda_device_uuid", self.cuda_device_uuid.as_deref()),
            ("cuda_pci_bus_id", self.cuda_pci_bus_id.as_deref()),
            (
                "cuda_driver_api_version",
                self.cuda_driver_api_version.as_deref(),
            ),
            (
                "cuda_binding_api_version",
                self.cuda_binding_api_version.as_deref(),
            ),
        ] {
            if let Some(value) = value {
                anyhow::ensure!(
                    !value.trim().is_empty(),
                    "hardware fingerprint {name} must not be empty when present"
                );
            }
        }
        if let Some(bytes) = self.device_total_memory_bytes {
            anyhow::ensure!(
                bytes > 0,
                "hardware fingerprint device_total_memory_bytes must be non-zero when present"
            );
        }

        anyhow::ensure!(
            self.schema_version == HARDWARE_FINGERPRINT_SCHEMA_VERSION,
            "unsupported hardware-fingerprint schema {}; this build supports schema {}",
            self.schema_version,
            HARDWARE_FINGERPRINT_SCHEMA_VERSION
        );
        self.validate_current()
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let fingerprint: Self =
            serde_json::from_slice(bytes).context("invalid hardware-fingerprint JSON")?;
        fingerprint.validate()?;
        Ok(fingerprint)
    }

    pub fn supports_calibration_cache_reuse(&self) -> bool {
        self.validate().is_ok()
            && self.schema_version == HARDWARE_FINGERPRINT_SCHEMA_VERSION
            && self.backend == DeviceBackend::Cuda
            && self.operating_system_version.is_some()
            && self.device_name.is_some()
            && self.device_total_memory_bytes.is_some()
            && self.driver_version.is_some()
            && self.cuda_compute_capability.is_some()
            && self.cuda_device_uuid.is_some()
            && self.cuda_driver_api_version.is_some()
            && self.cuda_binding_api_version.is_some()
            && self.runtime_version.is_some()
            && self.runtime_version_source == Some(FingerprintValueSource::CudaRuntimeLibraryBuild)
    }

    fn validate_current(&self) -> Result<()> {
        match self.backend {
            DeviceBackend::Cpu => {
                self.ensure_non_cuda_metadata_absent("CPU fingerprint")?;
            }
            DeviceBackend::Metal => {
                anyhow::ensure!(
                    self.device_total_memory_bytes.is_none(),
                    "Metal fingerprint device_total_memory_bytes must be null"
                );
                self.ensure_non_cuda_metadata_absent("Metal fingerprint")?;
            }
            DeviceBackend::Cuda => {
                validate_provenance(
                    "driver_version",
                    self.driver_version.is_some(),
                    self.driver_version_source,
                    self.driver_version_unavailable_reason,
                    FingerprintValueSource::ProcfsNvidiaDriver,
                    &[
                        FingerprintUnavailableReason::ProcfsUnavailable,
                        FingerprintUnavailableReason::ProcfsNvidiaDriverParseFailed,
                    ],
                )?;
                validate_provenance(
                    "runtime_version",
                    self.runtime_version.is_some(),
                    self.runtime_version_source,
                    self.runtime_version_unavailable_reason,
                    FingerprintValueSource::CudaRuntimeLibraryBuild,
                    &[FingerprintUnavailableReason::CudaRuntimeBuildIdentityUnavailable],
                )?;
                validate_provenance(
                    "cuda_compute_capability",
                    self.cuda_compute_capability.is_some(),
                    self.cuda_compute_capability_source,
                    self.cuda_compute_capability_unavailable_reason,
                    FingerprintValueSource::CudaDriverApi,
                    &[FingerprintUnavailableReason::CudaDriverApiQueryFailed],
                )?;
                if let Some(capability) = self.cuda_compute_capability {
                    anyhow::ensure!(
                        capability.major > 0,
                        "CUDA compute-capability major version must be non-zero"
                    );
                }
                validate_provenance(
                    "cuda_device_uuid",
                    self.cuda_device_uuid.is_some(),
                    self.cuda_device_uuid_source,
                    self.cuda_device_uuid_unavailable_reason,
                    FingerprintValueSource::CudaDriverApi,
                    &[FingerprintUnavailableReason::CudaDriverApiQueryFailed],
                )?;
                if let Some(uuid) = self.cuda_device_uuid.as_deref() {
                    anyhow::ensure!(
                        is_canonical_cuda_uuid(uuid),
                        "CUDA device UUID must use lowercase 8-4-4-4-12 hexadecimal form"
                    );
                }
                validate_provenance(
                    "cuda_pci_bus_id",
                    self.cuda_pci_bus_id.is_some(),
                    self.cuda_pci_bus_id_source,
                    self.cuda_pci_bus_id_unavailable_reason,
                    FingerprintValueSource::CudaDriverApi,
                    &[FingerprintUnavailableReason::CudaDriverApiQueryFailed],
                )?;
                validate_provenance(
                    "cuda_driver_api_version",
                    self.cuda_driver_api_version.is_some(),
                    self.cuda_driver_api_version_source,
                    self.cuda_driver_api_version_unavailable_reason,
                    FingerprintValueSource::CudaDriverApi,
                    &[FingerprintUnavailableReason::CudaDriverApiQueryFailed],
                )?;
                validate_provenance(
                    "cuda_binding_api_version",
                    self.cuda_binding_api_version.is_some(),
                    self.cuda_binding_api_version_source,
                    self.cuda_binding_api_version_unavailable_reason,
                    FingerprintValueSource::CudarcBuildBindings,
                    &[FingerprintUnavailableReason::CudarcBindingVersionUnavailable],
                )?;
            }
        }
        Ok(())
    }

    fn ensure_cuda_fields_absent(&self, record: &str) -> Result<()> {
        anyhow::ensure!(
            self.cuda_compute_capability.is_none()
                && self.cuda_compute_capability_source.is_none()
                && self.cuda_compute_capability_unavailable_reason.is_none()
                && self.cuda_device_uuid.is_none()
                && self.cuda_device_uuid_source.is_none()
                && self.cuda_device_uuid_unavailable_reason.is_none()
                && self.cuda_pci_bus_id.is_none()
                && self.cuda_pci_bus_id_source.is_none()
                && self.cuda_pci_bus_id_unavailable_reason.is_none()
                && self.cuda_driver_api_version.is_none()
                && self.cuda_driver_api_version_source.is_none()
                && self.cuda_driver_api_version_unavailable_reason.is_none()
                && self.cuda_binding_api_version.is_none()
                && self.cuda_binding_api_version_source.is_none()
                && self.cuda_binding_api_version_unavailable_reason.is_none(),
            "{record} must not contain CUDA metadata"
        );
        Ok(())
    }

    fn ensure_non_cuda_metadata_absent(&self, record: &str) -> Result<()> {
        self.ensure_cuda_fields_absent(record)?;
        anyhow::ensure!(
            self.driver_version.is_none()
                && self.driver_version_source.is_none()
                && self.driver_version_unavailable_reason.is_none()
                && self.runtime_version.is_none()
                && self.runtime_version_source.is_none()
                && self.runtime_version_unavailable_reason.is_none(),
            "{record} must not contain CUDA driver/runtime metadata"
        );
        Ok(())
    }
}

fn validate_provenance(
    field: &str,
    has_value: bool,
    source: Option<FingerprintValueSource>,
    unavailable_reason: Option<FingerprintUnavailableReason>,
    expected_source: FingerprintValueSource,
    allowed_unavailable_reasons: &[FingerprintUnavailableReason],
) -> Result<()> {
    if has_value {
        anyhow::ensure!(
            source == Some(expected_source) && unavailable_reason.is_none(),
            "hardware fingerprint {field} value requires source {expected_source:?} and no unavailable reason"
        );
    } else {
        anyhow::ensure!(
            source.is_none()
                && unavailable_reason
                    .is_some_and(|reason| allowed_unavailable_reasons.contains(&reason)),
            "hardware fingerprint {field} without a value requires no source and an applicable unavailable reason"
        );
    }
    Ok(())
}

fn is_canonical_cuda_uuid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| match index {
            8 | 13 | 18 | 23 => byte == b'-',
            _ => byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte),
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    tag = "kind",
    content = "bytes",
    deny_unknown_fields
)]
pub enum CgroupMemoryLimit {
    Unlimited,
    Bytes(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMeasurementScope {
    HostWide,
    ProcessCgroupV2,
    DeviceWide,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceMeasurementScopes {
    #[serde(deserialize_with = "crate::required_option")]
    pub host_memory: Option<MemoryMeasurementScope>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cgroup_memory: Option<MemoryMeasurementScope>,
    #[serde(deserialize_with = "crate::required_option")]
    pub device_memory: Option<MemoryMeasurementScope>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSnapshot {
    pub schema_version: u32,
    pub measured_at_unix_ms: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub host_memory_available_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cgroup_v2_memory_limit: Option<CgroupMemoryLimit>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cgroup_v2_memory_current_bytes: Option<u64>,
    /// Minimum estimated headroom across finite visible ancestors, including
    /// clean, unmapped file cache. The raw current usage above is unchanged.
    /// Missing or invalid cache statistics contribute no reclaimable bytes.
    #[serde(deserialize_with = "crate::required_option")]
    pub cgroup_v2_memory_available_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub device_free_memory_bytes: Option<u64>,
    /// Whether host and device memory are one shared physical pool (the CUDA
    /// `integrated` device attribute, e.g. Jetson unified memory). `Some(true)`
    /// switches admission to combined-pool accounting. `None` — legacy records,
    /// non-CUDA backends, and (warned-once) failed probes — keeps the split
    /// per-axis checks every consumer used before this field existed, as does
    /// a confirmed-discrete `Some(false)`. The distinction that matters for
    /// correctness is `Some(true)` versus everything else; the distinction
    /// between `None` and `Some(false)` is provenance for diagnostics.
    #[serde(default)]
    pub host_device_memory_is_unified: Option<bool>,
    /// Hardware-level totals (MemTotal / device total memory), unlike the
    /// instantaneous `*_available`/`device_free` views. Reserve sizes scale
    /// with totals so that a run's margin does not shrink when the machine is
    /// busy and sidecar-recorded reserves stay comparable across runs.
    #[serde(default)]
    pub host_memory_total_bytes: Option<u64>,
    #[serde(default)]
    pub device_total_memory_bytes: Option<u64>,
    pub measurement_scope: ResourceMeasurementScopes,
}

impl ResourceSnapshot {
    pub fn capture(device: Option<&Device>) -> Self {
        resource_snapshot_with_source(device, &SystemProbeSource, unix_time_ms())
    }

    /// Available bytes of the single memory pool when the probe confirmed a
    /// unified host/device topology. Both axes' charges draw from it, so the
    /// binding budget is the smaller of the two views. Returns `None` when
    /// the topology is discrete or was not probed, and when either view of
    /// the pool could not be measured.
    pub fn unified_pool_available_bytes(&self) -> Option<u64> {
        if self.host_device_memory_is_unified != Some(true) {
            return None;
        }
        [
            self.host_memory_available_bytes,
            self.cgroup_v2_memory_available_bytes,
            self.device_free_memory_bytes,
        ]
        .into_iter()
        .flatten()
        .min()
    }
}

trait ProbeSource {
    fn read_to_string(&self, path: &Path) -> io::Result<String>;

    /// Host-wide available memory, for platforms that report it outside the
    /// filesystem.
    ///
    /// Linux answers through `/proc/meminfo`, which `read_to_string` already
    /// covers and which tests can substitute. Windows has no such file, so the
    /// measurement has to come from a system call instead.
    fn host_available_memory_bytes(&self) -> Option<u64> {
        None
    }
}

struct SystemProbeSource;

impl ProbeSource for SystemProbeSource {
    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path)
    }

    #[cfg(windows)]
    fn host_available_memory_bytes(&self) -> Option<u64> {
        windows_available_physical_bytes()
    }
}

/// Windows' host-wide available physical memory.
///
/// `ullAvailPhys` is the closest counterpart to Linux's `MemAvailable`: both
/// answer "how much can a new allocation expect without paging". They are not
/// computed the same way — `MemAvailable` includes reclaimable page cache,
/// `ullAvailPhys` reports free physical pages — so a snapshot is comparable
/// across runs on one host, not across platforms.
#[cfg(windows)]
fn windows_available_physical_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = unsafe { std::mem::zeroed::<MEMORYSTATUSEX>() };
    status.dwLength = u32::try_from(size_of::<MEMORYSTATUSEX>()).ok()?;
    (unsafe { GlobalMemoryStatusEx(&raw mut status) } != 0).then_some(status.ullAvailPhys)
}

fn hardware_fingerprint_with_source(
    source_device: &Device,
    source: &impl ProbeSource,
) -> HardwareFingerprint {
    let cpu_info = source.read_to_string(Path::new("/proc/cpuinfo")).ok();
    let logical_cpu_count = cpu_info
        .as_deref()
        .and_then(parse_logical_cpu_count)
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(|count| count.get())
        })
        .and_then(|count| u64::try_from(count).ok())
        .unwrap_or(1);
    let os_version = source
        .read_to_string(Path::new("/proc/sys/kernel/osrelease"))
        .ok()
        .and_then(|value| nonempty_trimmed(&value));

    let mut fingerprint = HardwareFingerprint {
        schema_version: HARDWARE_FINGERPRINT_SCHEMA_VERSION,
        operating_system: std::env::consts::OS.to_owned(),
        operating_system_version: os_version,
        architecture: std::env::consts::ARCH.to_owned(),
        logical_cpu_count,
        backend: DeviceBackend::Cpu,
        device_name: cpu_info.as_deref().and_then(parse_cpu_name),
        device_total_memory_bytes: source
            .read_to_string(Path::new("/proc/meminfo"))
            .ok()
            .as_deref()
            .and_then(|contents| parse_meminfo_bytes(contents, "MemTotal")),
        driver_version: None,
        runtime_version: None,
        cuda_compute_capability: None,
        cuda_compute_capability_source: None,
        cuda_compute_capability_unavailable_reason: None,
        cuda_device_uuid: None,
        cuda_device_uuid_source: None,
        cuda_device_uuid_unavailable_reason: None,
        cuda_pci_bus_id: None,
        cuda_pci_bus_id_source: None,
        cuda_pci_bus_id_unavailable_reason: None,
        driver_version_source: None,
        driver_version_unavailable_reason: None,
        cuda_driver_api_version: None,
        cuda_driver_api_version_source: None,
        cuda_driver_api_version_unavailable_reason: None,
        cuda_binding_api_version: None,
        cuda_binding_api_version_source: None,
        cuda_binding_api_version_unavailable_reason: None,
        runtime_version_source: None,
        runtime_version_unavailable_reason: None,
    };

    match source_device.location() {
        DeviceLocation::Cpu => {}
        DeviceLocation::Cuda { .. } => {
            fingerprint.backend = DeviceBackend::Cuda;
            fingerprint.device_name = None;
            fingerprint.device_total_memory_bytes = None;
            populate_cuda_driver_version(source, &mut fingerprint);
            fingerprint.runtime_version_unavailable_reason =
                Some(FingerprintUnavailableReason::CudaRuntimeBuildIdentityUnavailable);
            populate_cuda_fingerprint(source_device, &mut fingerprint);
        }
        DeviceLocation::Metal { .. } => {
            fingerprint.backend = DeviceBackend::Metal;
            fingerprint.device_name = None;
            fingerprint.device_total_memory_bytes = None;
            populate_metal_fingerprint(source_device, &mut fingerprint);
        }
    }
    fingerprint
}

fn resource_snapshot_with_source(
    device: Option<&Device>,
    source: &impl ProbeSource,
    measured_at_unix_ms: u64,
) -> ResourceSnapshot {
    let host_memory_available_bytes = source
        .read_to_string(Path::new("/proc/meminfo"))
        .ok()
        .as_deref()
        .and_then(|contents| parse_meminfo_bytes(contents, "MemAvailable"))
        .or_else(|| source.host_available_memory_bytes());
    let (cgroup_v2_memory_limit, cgroup_v2_memory_current_bytes, cgroup_v2_memory_available_bytes) =
        read_cgroup_v2_memory(source).unwrap_or((None, None, None));
    let device_free_memory_bytes = device.and_then(device_free_memory);
    let host_device_memory_is_unified = device.and_then(device_memory_is_unified);
    let host_memory_total_bytes = source
        .read_to_string(Path::new("/proc/meminfo"))
        .ok()
        .as_deref()
        .and_then(|contents| parse_meminfo_bytes(contents, "MemTotal"));
    let device_total_memory_bytes = device.and_then(device_total_memory);

    ResourceSnapshot {
        schema_version: RESOURCE_SNAPSHOT_SCHEMA_VERSION,
        measured_at_unix_ms,
        host_memory_available_bytes,
        cgroup_v2_memory_limit,
        cgroup_v2_memory_current_bytes,
        cgroup_v2_memory_available_bytes,
        device_free_memory_bytes,
        host_device_memory_is_unified,
        host_memory_total_bytes,
        device_total_memory_bytes,
        measurement_scope: ResourceMeasurementScopes {
            host_memory: host_memory_available_bytes.map(|_| MemoryMeasurementScope::HostWide),
            cgroup_memory: (cgroup_v2_memory_limit.is_some()
                || cgroup_v2_memory_current_bytes.is_some())
            .then_some(MemoryMeasurementScope::ProcessCgroupV2),
            device_memory: device_free_memory_bytes.map(|_| MemoryMeasurementScope::DeviceWide),
        },
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| u64::try_from(elapsed.as_millis()).ok())
        .unwrap_or(0)
}

fn nonempty_trimmed(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn parse_logical_cpu_count(cpu_info: &str) -> Option<usize> {
    let count = cpu_info
        .lines()
        .filter(|line| {
            line.split_once(':')
                .is_some_and(|(name, _)| name.trim() == "processor")
        })
        .count();
    (count > 0).then_some(count)
}

fn parse_cpu_name(cpu_info: &str) -> Option<String> {
    ["model name", "Model", "Hardware", "Processor"]
        .into_iter()
        .find_map(|key| {
            cpu_info.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == key).then(|| value.trim().to_owned())
            })
        })
        .filter(|value| !value.is_empty())
}

fn parse_meminfo_bytes(contents: &str, key: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name != key {
            return None;
        }
        let mut fields = value.split_whitespace();
        let amount = fields.next()?.parse::<u64>().ok()?;
        match fields.next() {
            Some("kB") => amount.checked_mul(1024),
            None | Some("B") => Some(amount),
            _ => None,
        }
    })
}

fn parse_nvidia_driver_version(contents: &str) -> Option<String> {
    let version_line = contents
        .lines()
        .find(|line| line.trim_start().starts_with("NVRM version:"))?;
    version_line
        .split_whitespace()
        .find(|field| {
            field.as_bytes().first().is_some_and(u8::is_ascii_digit) && field.contains('.')
        })
        .map(|value| {
            value
                .trim_matches(|character: char| {
                    !character.is_ascii_alphanumeric() && character != '.'
                })
                .to_owned()
        })
        .filter(|value| !value.is_empty())
}

fn populate_cuda_driver_version(source: &impl ProbeSource, fingerprint: &mut HardwareFingerprint) {
    fingerprint.driver_version = None;
    fingerprint.driver_version_source = None;
    fingerprint.driver_version_unavailable_reason = None;
    match source.read_to_string(Path::new("/proc/driver/nvidia/version")) {
        Err(_) => {
            fingerprint.driver_version_unavailable_reason =
                Some(FingerprintUnavailableReason::ProcfsUnavailable);
        }
        Ok(contents) => match parse_nvidia_driver_version(&contents) {
            Some(version) => {
                fingerprint.driver_version = Some(version);
                fingerprint.driver_version_source =
                    Some(FingerprintValueSource::ProcfsNvidiaDriver);
            }
            None => {
                fingerprint.driver_version_unavailable_reason =
                    Some(FingerprintUnavailableReason::ProcfsNvidiaDriverParseFailed);
            }
        },
    }
}

#[cfg(any(feature = "cuda", test))]
fn format_cuda_api_version(encoded: u32) -> Option<String> {
    let major = encoded / 1000;
    let minor = (encoded % 1000) / 10;
    (major > 0).then(|| format!("{major}.{minor}"))
}

#[cfg(any(feature = "cuda", test))]
fn format_cuda_uuid(bytes: [u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

fn read_cgroup_v2_memory(
    source: &impl ProbeSource,
) -> Option<(Option<CgroupMemoryLimit>, Option<u64>, Option<u64>)> {
    let cgroup = source.read_to_string(Path::new("/proc/self/cgroup")).ok()?;
    let mountinfo = source
        .read_to_string(Path::new("/proc/self/mountinfo"))
        .ok()?;
    let (mount_point, directory) = resolve_cgroup_v2_directory(&cgroup, &mountinfo)?;
    let leaf_current = source
        .read_to_string(&directory.join("memory.current"))
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok());

    let mut saw_limit = false;
    let mut finite_limit: Option<u64> = None;
    let mut effective_available: Option<u64> = None;
    let mut hierarchy_complete = true;
    for ancestor in directory.ancestors() {
        if !ancestor.starts_with(&mount_point) {
            break;
        }
        let limit = source
            .read_to_string(&ancestor.join("memory.max"))
            .ok()
            .as_deref()
            .and_then(parse_cgroup_memory_limit);
        let Some(limit) = limit else {
            if ancestor == mount_point {
                saw_limit = true;
                break;
            }
            hierarchy_complete = false;
            continue;
        };
        saw_limit = true;
        if let CgroupMemoryLimit::Bytes(bytes) = limit {
            finite_limit = Some(finite_limit.map_or(bytes, |current| current.min(bytes)));
            let current = source
                .read_to_string(&ancestor.join("memory.current"))
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok());
            let Some(current) = current else {
                hierarchy_complete = false;
                continue;
            };
            let reclaimable = source
                .read_to_string(&ancestor.join("memory.stat"))
                .ok()
                .as_deref()
                .and_then(cgroup_reclaimable_file_bytes)
                .unwrap_or(0)
                .min(current);
            let available = bytes.saturating_sub(current - reclaimable);
            effective_available =
                Some(effective_available.map_or(available, |current| current.min(available)));
        }
        if ancestor == mount_point {
            break;
        }
    }
    let effective_limit = hierarchy_complete.then(|| {
        finite_limit
            .map(CgroupMemoryLimit::Bytes)
            .unwrap_or(CgroupMemoryLimit::Unlimited)
    });
    let effective_available = hierarchy_complete.then_some(effective_available).flatten();
    debug_assert!(!hierarchy_complete || saw_limit);
    Some((effective_limit, leaf_current, effective_available))
}

fn cgroup_reclaimable_file_bytes(stat: &str) -> Option<u64> {
    const KEYS: [&str; 6] = [
        "file",
        "shmem",
        "file_mapped",
        "file_dirty",
        "file_writeback",
        "unevictable",
    ];
    let mut values = [None; KEYS.len()];
    for line in stat.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else {
            continue;
        };
        let Some(index) = KEYS.iter().position(|candidate| *candidate == key) else {
            continue;
        };
        let value = fields.next()?.parse::<u64>().ok()?;
        if fields.next().is_some() || values[index].replace(value).is_some() {
            return None;
        }
    }
    let mut reclaimable = values[0]?;
    for excluded in &values[1..] {
        reclaimable = reclaimable.saturating_sub((*excluded)?);
    }
    Some(reclaimable)
}

fn parse_cgroup_memory_limit(value: &str) -> Option<CgroupMemoryLimit> {
    match value.trim() {
        "max" => Some(CgroupMemoryLimit::Unlimited),
        value => value.parse::<u64>().ok().map(CgroupMemoryLimit::Bytes),
    }
}

fn resolve_cgroup_v2_directory(cgroup: &str, mountinfo: &str) -> Option<(PathBuf, PathBuf)> {
    let cgroup_path = cgroup.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        (fields.next()? == "0" && fields.next()?.is_empty())
            .then(|| fields.next())
            .flatten()
    })?;
    let cgroup_path = Path::new(cgroup_path);

    mountinfo.lines().find_map(|line| {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        let separator = fields.iter().position(|field| *field == "-")?;
        if fields.get(separator + 1).copied() != Some("cgroup2") {
            return None;
        }
        let mount_root = PathBuf::from(decode_mountinfo_path(fields.get(3)?));
        let mount_point = PathBuf::from(decode_mountinfo_path(fields.get(4)?));
        let relative = cgroup_path.strip_prefix(&mount_root).ok()?;
        let directory = mount_point.join(relative);
        Some((mount_point, directory))
    })
}

fn decode_mountinfo_path(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

#[cfg(any(feature = "cuda", test))]
#[derive(Default)]
struct CudaFingerprintDetails {
    device_name: Option<String>,
    device_total_memory_bytes: Option<u64>,
    compute_capability: Option<CudaComputeCapability>,
    device_uuid: Option<String>,
    pci_bus_id: Option<String>,
    driver_api_version: Option<String>,
    binding_api_version: Option<String>,
}

#[cfg(any(feature = "cuda", test))]
fn apply_cuda_fingerprint_details(
    fingerprint: &mut HardwareFingerprint,
    details: CudaFingerprintDetails,
) {
    fingerprint.device_name = details.device_name;
    fingerprint.device_total_memory_bytes = details.device_total_memory_bytes;

    fingerprint.cuda_compute_capability = details.compute_capability;
    if fingerprint.cuda_compute_capability.is_some() {
        fingerprint.cuda_compute_capability_source = Some(FingerprintValueSource::CudaDriverApi);
    } else {
        fingerprint.cuda_compute_capability_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed);
    }

    fingerprint.cuda_device_uuid = details.device_uuid;
    if fingerprint.cuda_device_uuid.is_some() {
        fingerprint.cuda_device_uuid_source = Some(FingerprintValueSource::CudaDriverApi);
    } else {
        fingerprint.cuda_device_uuid_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed);
    }

    fingerprint.cuda_pci_bus_id = details.pci_bus_id;
    if fingerprint.cuda_pci_bus_id.is_some() {
        fingerprint.cuda_pci_bus_id_source = Some(FingerprintValueSource::CudaDriverApi);
    } else {
        fingerprint.cuda_pci_bus_id_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed);
    }

    fingerprint.cuda_driver_api_version = details.driver_api_version;
    if fingerprint.cuda_driver_api_version.is_some() {
        fingerprint.cuda_driver_api_version_source = Some(FingerprintValueSource::CudaDriverApi);
    } else {
        fingerprint.cuda_driver_api_version_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed);
    }

    fingerprint.cuda_binding_api_version = details.binding_api_version;
    if fingerprint.cuda_binding_api_version.is_some() {
        fingerprint.cuda_binding_api_version_source =
            Some(FingerprintValueSource::CudarcBuildBindings);
    } else {
        fingerprint.cuda_binding_api_version_unavailable_reason =
            Some(FingerprintUnavailableReason::CudarcBindingVersionUnavailable);
    }
}

#[cfg(feature = "cuda")]
fn populate_cuda_fingerprint(device: &Device, fingerprint: &mut HardwareFingerprint) {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let Ok(cuda) = device.as_cuda_device() else {
        apply_cuda_fingerprint_details(fingerprint, CudaFingerprintDetails::default());
        return;
    };
    let stream = cuda.cuda_stream();
    let context = stream.context();
    let compute_capability = context
        .compute_capability()
        .ok()
        .and_then(|(major, minor)| {
            Some(CudaComputeCapability {
                major: u32::try_from(major).ok()?,
                minor: u32::try_from(minor).ok()?,
            })
        });
    let device_uuid = context.uuid().ok().map(|uuid| {
        let bytes = uuid.bytes.map(|byte| byte as u8);
        format_cuda_uuid(bytes)
    });
    let pci_bus_id = cuda_pci_bus_id(context.cu_device());
    let driver_api_version = cuda_driver_api_version();
    let binding_api_version = format_cuda_api_version(sys::CUDA_VERSION);

    apply_cuda_fingerprint_details(
        fingerprint,
        CudaFingerprintDetails {
            device_name: context.name().ok().and_then(|name| nonempty_trimmed(&name)),
            device_total_memory_bytes: context
                .total_mem()
                .ok()
                .and_then(|bytes| u64::try_from(bytes).ok()),
            compute_capability,
            device_uuid,
            pci_bus_id,
            driver_api_version,
            binding_api_version,
        },
    );
}

#[cfg(feature = "cuda")]
fn cuda_pci_bus_id(
    device: candle_core::cuda_backend::cudarc::driver::sys::CUdevice,
) -> Option<String> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let mut buffer = [0_u8; 32];
    let result = unsafe {
        sys::cuDeviceGetPCIBusId(
            buffer.as_mut_ptr().cast(),
            i32::try_from(buffer.len()).ok()?,
            device,
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    std::ffi::CStr::from_bytes_until_nul(&buffer)
        .ok()
        .and_then(|value| value.to_str().ok())
        .and_then(nonempty_trimmed)
}

#[cfg(feature = "cuda")]
fn cuda_driver_api_version() -> Option<String> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let mut encoded = 0_i32;
    let result = unsafe { sys::cuDriverGetVersion(&mut encoded) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    u32::try_from(encoded)
        .ok()
        .and_then(format_cuda_api_version)
}

#[cfg(not(feature = "cuda"))]
fn populate_cuda_fingerprint(_device: &Device, _fingerprint: &mut HardwareFingerprint) {}

#[cfg(feature = "metal")]
fn populate_metal_fingerprint(device: &Device, fingerprint: &mut HardwareFingerprint) {
    let Ok(metal) = device.as_metal_device() else {
        return;
    };
    fingerprint.device_name = nonempty_trimmed(metal.device().name());
}

#[cfg(not(feature = "metal"))]
fn populate_metal_fingerprint(_device: &Device, _fingerprint: &mut HardwareFingerprint) {}

#[cfg(feature = "cuda")]
fn device_free_memory(device: &Device) -> Option<u64> {
    let cuda = device.as_cuda_device().ok()?;
    let stream = cuda.cuda_stream();
    let (free, _) = stream.context().mem_get_info().ok()?;
    u64::try_from(free).ok()
}

#[cfg(not(feature = "cuda"))]
fn device_free_memory(_device: &Device) -> Option<u64> {
    None
}

#[cfg(feature = "cuda")]
fn device_total_memory(device: &Device) -> Option<u64> {
    let cuda = device.as_cuda_device().ok()?;
    let stream = cuda.cuda_stream();
    u64::try_from(stream.context().total_mem().ok()?).ok()
}

#[cfg(not(feature = "cuda"))]
fn device_total_memory(_device: &Device) -> Option<u64> {
    None
}

/// Probe the CUDA `integrated` device attribute once at snapshot capture.
/// `Some(false)` is a confirmed discrete topology; `None` means unprobed
/// (non-CUDA device or build). A query failure on a CUDA device is not
/// silent: it warns once per process before falling back to `None`, so a real
/// integrated device cannot regress to split-axis accounting unnoticed.
#[cfg(feature = "cuda")]
fn device_memory_is_unified(device: &Device) -> Option<bool> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    static WARNED: std::sync::Once = std::sync::Once::new();
    let cuda = device.as_cuda_device().ok()?;
    let stream = cuda.cuda_stream();
    let mut value = 0_i32;
    let result = unsafe {
        sys::cuDeviceGetAttribute(
            &raw mut value,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED,
            stream.context().cu_device(),
        )
    };
    if result == sys::CUresult::CUDA_SUCCESS {
        return Some(value != 0);
    }
    WARNED.call_once(|| {
        eprintln!(
            "warning: CUDA integrated-topology attribute query failed ({result:?}); \
             this capture records host_device_memory_is_unified as unprobed"
        );
    });
    None
}

#[cfg(not(feature = "cuda"))]
fn device_memory_is_unified(_device: &Device) -> Option<bool> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct FakeProbeSource {
        files: HashMap<PathBuf, String>,
    }

    impl FakeProbeSource {
        fn with(mut self, path: &str, contents: &str) -> Self {
            self.files.insert(PathBuf::from(path), contents.to_owned());
            self
        }
    }

    impl ProbeSource for FakeProbeSource {
        fn read_to_string(&self, path: &Path) -> io::Result<String> {
            self.files.get(path).cloned().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing {}", path.display()),
                )
            })
        }
    }

    fn valid_current_cuda_fingerprint() -> HardwareFingerprint {
        let mut fingerprint = hardware_fingerprint_with_source(
            &Device::Cpu,
            &FakeProbeSource::default()
                .with("/proc/cpuinfo", "processor: 0\nmodel name: Test CPU\n")
                .with("/proc/meminfo", "MemTotal: 32768 kB\n")
                .with("/proc/sys/kernel/osrelease", "6.8.0-test\n"),
        );
        fingerprint.backend = DeviceBackend::Cuda;
        fingerprint.driver_version = Some("570.124.06".to_owned());
        fingerprint.driver_version_source = Some(FingerprintValueSource::ProcfsNvidiaDriver);
        fingerprint.runtime_version_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaRuntimeBuildIdentityUnavailable);
        apply_cuda_fingerprint_details(
            &mut fingerprint,
            CudaFingerprintDetails {
                device_name: Some("Injected GPU".to_owned()),
                device_total_memory_bytes: Some(24 << 30),
                compute_capability: Some(CudaComputeCapability { major: 8, minor: 9 }),
                device_uuid: Some("00010203-0405-0607-0809-0a0b0c0d0e0f".to_owned()),
                pci_bus_id: Some("0000:65:00.0".to_owned()),
                driver_api_version: Some("12.8".to_owned()),
                binding_api_version: Some("12.8".to_owned()),
            },
        );
        fingerprint
    }

    #[test]
    fn parses_linux_meminfo_units_and_rejects_unknown_units() {
        let meminfo = "MemTotal:       16384 kB\nMemAvailable: 4096 kB\nSwapTotal: 2 MB\n";
        assert_eq!(parse_meminfo_bytes(meminfo, "MemTotal"), Some(16 << 20));
        assert_eq!(parse_meminfo_bytes(meminfo, "MemAvailable"), Some(4 << 20));
        assert_eq!(parse_meminfo_bytes(meminfo, "SwapTotal"), None);
        assert_eq!(parse_meminfo_bytes(meminfo, "Missing"), None);
    }

    #[test]
    fn resolves_namespaced_cgroup_v2_mount_and_reads_finite_values() {
        let source = FakeProbeSource::default()
            .with("/proc/meminfo", "MemAvailable: 8192 kB\n")
            .with("/proc/self/cgroup", "0::/docker/abc/workload\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 /docker/abc /sys/fs/cgroup rw,nosuid - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/workload/memory.max", "1073741824\n")
            .with("/sys/fs/cgroup/workload/memory.current", "268435456\n")
            .with("/sys/fs/cgroup/memory.max", "max\n");

        let snapshot = resource_snapshot_with_source(None, &source, 1234);
        assert_eq!(snapshot.measured_at_unix_ms, 1234);
        assert_eq!(snapshot.host_memory_available_bytes, Some(8 << 20));
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Bytes(1 << 30))
        );
        assert_eq!(snapshot.cgroup_v2_memory_current_bytes, Some(1 << 28));
        assert_eq!(
            snapshot.cgroup_v2_memory_available_bytes,
            Some((1 << 30) - (1 << 28))
        );
        assert_eq!(
            snapshot.measurement_scope,
            ResourceMeasurementScopes {
                host_memory: Some(MemoryMeasurementScope::HostWide),
                cgroup_memory: Some(MemoryMeasurementScope::ProcessCgroupV2),
                device_memory: None,
            }
        );
    }

    #[test]
    fn preserves_unlimited_cgroup_limit_and_decodes_mount_path() {
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/tenant/a\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup\\040unified rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup unified/tenant/a/memory.max", "max\n")
            .with("/sys/fs/cgroup unified/tenant/a/memory.current", "42\n")
            .with("/sys/fs/cgroup unified/tenant/memory.max", "max\n")
            .with("/sys/fs/cgroup unified/memory.max", "max\n");

        let snapshot = resource_snapshot_with_source(None, &source, 7);
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Unlimited)
        );
        assert_eq!(snapshot.cgroup_v2_memory_current_bytes, Some(42));
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, None);
    }

    #[test]
    fn cgroup_capacity_recovers_after_model_pages_are_unmapped() {
        let gib = 1_u64 << 30;
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/memory.max", &(62 * gib).to_string())
            .with("/sys/fs/cgroup/memory.current", &(58 * gib).to_string());
        let stat = |mapped| {
            format!(
                "file {}\nshmem 0\nfile_mapped {mapped}\nfile_dirty 0\n\
                 file_writeback 0\nunevictable 0\nactive_file {}\ninactive_file {}\n",
                56 * gib,
                53 * gib,
                3 * gib,
            )
        };
        let source = source.with("/sys/fs/cgroup/memory.stat", &stat(56 * gib));
        let running = resource_snapshot_with_source(None, &source, 1);
        assert_eq!(running.cgroup_v2_memory_available_bytes, Some(4 * gib));

        let source = source.with("/sys/fs/cgroup/memory.stat", &stat(0));
        let exited = resource_snapshot_with_source(None, &source, 2);
        assert_eq!(exited.cgroup_v2_memory_current_bytes, Some(58 * gib));
        assert_eq!(exited.cgroup_v2_memory_available_bytes, Some(60 * gib));
    }

    #[test]
    fn cgroup_cache_excludes_shared_mapped_dirty_and_unevictable_pages() {
        let stat = "file 1000\nshmem 100\nfile_mapped 200\nfile_dirty 30\n\
                    file_writeback 40\nunevictable 50\nunknown_future_counter 999\n";
        assert_eq!(cgroup_reclaimable_file_bytes(stat), Some(580));
        assert_eq!(
            cgroup_reclaimable_file_bytes(&stat.replace("shmem 100", "shmem 18446744073709551615")),
            Some(0)
        );
    }

    #[test]
    fn cgroup_cache_requires_complete_unambiguous_statistics() {
        let valid =
            "file 100\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n";
        for line in valid.lines() {
            let missing = valid.replace(&format!("{line}\n"), "");
            assert_eq!(cgroup_reclaimable_file_bytes(&missing), None);
            let key = line.split_whitespace().next().unwrap();
            for invalid in ["", "-1", "invalid", "18446744073709551616", "0 extra"] {
                assert_eq!(
                    cgroup_reclaimable_file_bytes(
                        &valid.replace(line, &format!("{key} {invalid}"))
                    ),
                    None
                );
            }
            assert_eq!(
                cgroup_reclaimable_file_bytes(&format!("{valid}{line}\n")),
                None
            );
        }
    }

    #[test]
    fn cgroup_capacity_bounds_cache_and_falls_back_when_stats_are_invalid() {
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/memory.max", "1000")
            .with("/sys/fs/cgroup/memory.current", "1200");
        let stat =
            "file 300\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n";
        let source = source.with("/sys/fs/cgroup/memory.stat", stat);
        let snapshot = resource_snapshot_with_source(None, &source, 1);
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(100));

        let source = source.with("/sys/fs/cgroup/memory.current", "200");
        let snapshot = resource_snapshot_with_source(None, &source, 2);
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(1000));

        let source = source.with("/sys/fs/cgroup/memory.stat", "file 300\n");
        let snapshot = resource_snapshot_with_source(None, &source, 3);
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(800));
    }

    #[test]
    fn cgroup_capacity_uses_the_most_restrictive_ancestor() {
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/tenant/child\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/tenant/child/memory.max", "max\n")
            .with("/sys/fs/cgroup/tenant/child/memory.current", "100\n")
            .with("/sys/fs/cgroup/tenant/memory.max", "1000\n")
            .with("/sys/fs/cgroup/tenant/memory.current", "900\n")
            .with("/sys/fs/cgroup/memory.max", "2000\n")
            .with("/sys/fs/cgroup/memory.current", "1000\n");

        let snapshot = resource_snapshot_with_source(None, &source, 11);
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Bytes(1000))
        );
        assert_eq!(snapshot.cgroup_v2_memory_current_bytes, Some(100));
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(100));

        let stat =
            "file 800\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n";
        let source = source
            .with("/sys/fs/cgroup/tenant/memory.stat", stat)
            .with("/sys/fs/cgroup/memory.current", "1800\n");
        let snapshot = resource_snapshot_with_source(None, &source, 12);
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(200));

        let source = source.with("/sys/fs/cgroup/memory.stat", stat);
        let snapshot = resource_snapshot_with_source(None, &source, 13);
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(900));
    }

    #[test]
    fn incomplete_cgroup_hierarchy_never_reports_a_partial_capacity() {
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/tenant/child\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/tenant/child/memory.max", "1000\n")
            .with("/sys/fs/cgroup/tenant/child/memory.current", "100\n");

        let snapshot = resource_snapshot_with_source(None, &source, 12);
        assert_eq!(snapshot.cgroup_v2_memory_limit, None);
        assert_eq!(snapshot.cgroup_v2_memory_current_bytes, Some(100));
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, None);
    }

    #[test]
    fn cgroup_root_may_omit_memory_controller_limit_files() {
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/tenant/child\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/tenant/child/memory.max", "max\n")
            .with("/sys/fs/cgroup/tenant/child/memory.current", "100\n")
            .with("/sys/fs/cgroup/tenant/memory.max", "1000\n")
            .with("/sys/fs/cgroup/tenant/memory.current", "900\n");

        let snapshot = resource_snapshot_with_source(None, &source, 13);
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Bytes(1000))
        );
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(100));
    }

    #[test]
    fn missing_proc_and_cgroup_files_are_reported_as_unavailable() {
        let snapshot = resource_snapshot_with_source(None, &FakeProbeSource::default(), 9);
        assert_eq!(
            snapshot,
            ResourceSnapshot {
                schema_version: RESOURCE_SNAPSHOT_SCHEMA_VERSION,
                measured_at_unix_ms: 9,
                host_memory_available_bytes: None,
                cgroup_v2_memory_limit: None,
                cgroup_v2_memory_current_bytes: None,
                cgroup_v2_memory_available_bytes: None,
                device_free_memory_bytes: None,
                host_device_memory_is_unified: None,
                host_memory_total_bytes: None,
                device_total_memory_bytes: None,
                measurement_scope: ResourceMeasurementScopes {
                    host_memory: None,
                    cgroup_memory: None,
                    device_memory: None,
                },
            }
        );
    }

    #[test]
    fn unified_pool_available_bytes_requires_a_probed_unified_topology() {
        let snapshot = |unified: Option<bool>| ResourceSnapshot {
            schema_version: RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(10),
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: Some(8),
            device_free_memory_bytes: Some(6),
            host_device_memory_is_unified: unified,
            host_memory_total_bytes: None,
            device_total_memory_bytes: None,
            measurement_scope: ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        };
        // Unprobed (legacy) and confirmed-discrete records expose no pool.
        assert_eq!(snapshot(None).unified_pool_available_bytes(), None);
        assert_eq!(snapshot(Some(false)).unified_pool_available_bytes(), None);
        // A probed unified topology binds on the smallest view of the pool.
        assert_eq!(snapshot(Some(true)).unified_pool_available_bytes(), Some(6));
    }

    #[test]
    fn cpu_fingerprint_is_stable_and_never_contains_dynamic_free_memory() {
        let source = FakeProbeSource::default()
            .with(
                "/proc/cpuinfo",
                "processor: 0\nmodel name: Test CPU\n\nprocessor: 1\nmodel name: Test CPU\n",
            )
            .with("/proc/meminfo", "MemTotal: 32768 kB\nMemAvailable: 1 kB\n")
            .with("/proc/sys/kernel/osrelease", "6.8.0-test\n");
        let fingerprint = hardware_fingerprint_with_source(&Device::Cpu, &source);

        assert_eq!(
            fingerprint.schema_version,
            HARDWARE_FINGERPRINT_SCHEMA_VERSION
        );
        assert_eq!(fingerprint.logical_cpu_count, 2);
        assert_eq!(fingerprint.backend, DeviceBackend::Cpu);
        assert_eq!(fingerprint.device_name.as_deref(), Some("Test CPU"));
        assert_eq!(fingerprint.device_total_memory_bytes, Some(32 << 20));
        assert_eq!(
            fingerprint.operating_system_version.as_deref(),
            Some("6.8.0-test")
        );
        let json = serde_json::to_value(&fingerprint).unwrap();
        assert!(json.get("device_free_memory_bytes").is_none());
        assert!(json.get("host_memory_available_bytes").is_none());
        fingerprint.validate().unwrap();
        assert!(!fingerprint.supports_calibration_cache_reuse());
    }

    #[test]
    fn parses_driver_and_cuda_api_versions() {
        assert_eq!(
            parse_nvidia_driver_version(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.124.06  Wed\n"
            )
            .as_deref(),
            Some("570.124.06")
        );
        assert_eq!(format_cuda_api_version(12_080).as_deref(), Some("12.8"));
        assert_eq!(format_cuda_api_version(11_040).as_deref(), Some("11.4"));
        assert_eq!(format_cuda_api_version(0), None);
    }

    #[test]
    fn cuda_identity_fields_are_injectable_without_a_cuda_device() {
        let mut fingerprint = hardware_fingerprint_with_source(
            &Device::Cpu,
            &FakeProbeSource::default()
                .with("/proc/cpuinfo", "processor: 0\nmodel name: Test CPU\n"),
        );
        apply_cuda_fingerprint_details(
            &mut fingerprint,
            CudaFingerprintDetails {
                device_name: Some("Injected GPU".to_owned()),
                device_total_memory_bytes: Some(24 << 30),
                compute_capability: Some(CudaComputeCapability { major: 8, minor: 9 }),
                device_uuid: Some("00010203-0405-0607-0809-0a0b0c0d0e0f".to_owned()),
                pci_bus_id: Some("0000:65:00.0".to_owned()),
                driver_api_version: Some("12.8".to_owned()),
                binding_api_version: Some("12.8".to_owned()),
            },
        );

        assert_eq!(fingerprint.device_name.as_deref(), Some("Injected GPU"));
        assert_eq!(
            fingerprint.cuda_compute_capability,
            Some(CudaComputeCapability { major: 8, minor: 9 })
        );
        assert_eq!(
            fingerprint.cuda_compute_capability_source,
            Some(FingerprintValueSource::CudaDriverApi)
        );
        assert_eq!(
            fingerprint.cuda_device_uuid.as_deref(),
            Some("00010203-0405-0607-0809-0a0b0c0d0e0f")
        );
        assert_eq!(fingerprint.cuda_pci_bus_id.as_deref(), Some("0000:65:00.0"));
        assert_eq!(fingerprint.cuda_driver_api_version.as_deref(), Some("12.8"));
        assert_eq!(
            fingerprint.cuda_binding_api_version_source,
            Some(FingerprintValueSource::CudarcBuildBindings)
        );
    }

    #[test]
    fn current_cuda_fingerprint_validates_but_cache_reuse_is_fail_closed() {
        let fingerprint = valid_current_cuda_fingerprint();

        fingerprint.validate().unwrap();
        assert!(!fingerprint.supports_calibration_cache_reuse());

        let mut with_runtime_build = fingerprint;
        with_runtime_build.runtime_version = Some("libcudart.so.12.8.89".to_owned());
        with_runtime_build.runtime_version_source =
            Some(FingerprintValueSource::CudaRuntimeLibraryBuild);
        with_runtime_build.runtime_version_unavailable_reason = None;
        with_runtime_build.validate().unwrap();
        assert!(with_runtime_build.supports_calibration_cache_reuse());

        let mut missing_os_identity = with_runtime_build.clone();
        missing_os_identity.operating_system_version = None;
        missing_os_identity.validate().unwrap();
        assert!(!missing_os_identity.supports_calibration_cache_reuse());

        let mut missing_compute_identity = with_runtime_build;
        missing_compute_identity.cuda_compute_capability = None;
        missing_compute_identity.cuda_compute_capability_source = None;
        missing_compute_identity.cuda_compute_capability_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed);
        missing_compute_identity.validate().unwrap();
        assert!(!missing_compute_identity.supports_calibration_cache_reuse());
    }

    #[test]
    fn fingerprint_rejects_tampered_provenance_and_backend_metadata() {
        let encoded = serde_json::to_value(valid_current_cuda_fingerprint()).unwrap();

        let mut wrong_source = encoded.clone();
        wrong_source["cuda_device_uuid_source"] = serde_json::json!("cudarc_build_bindings");
        assert!(
            HardwareFingerprint::from_json(&serde_json::to_vec(&wrong_source).unwrap()).is_err()
        );

        let mut value_and_reason = encoded.clone();
        value_and_reason["cuda_device_uuid_unavailable_reason"] =
            serde_json::json!("cuda_driver_api_query_failed");
        assert!(
            HardwareFingerprint::from_json(&serde_json::to_vec(&value_and_reason).unwrap())
                .is_err()
        );

        let mut cpu_with_cuda_metadata = encoded.clone();
        cpu_with_cuda_metadata["backend"] = serde_json::json!("cpu");
        assert!(
            HardwareFingerprint::from_json(&serde_json::to_vec(&cpu_with_cuda_metadata).unwrap())
                .is_err()
        );

        let mut unknown_field = encoded;
        unknown_field["future_identity"] = serde_json::json!(true);
        assert!(
            HardwareFingerprint::from_json(&serde_json::to_vec(&unknown_field).unwrap()).is_err()
        );
    }

    #[test]
    fn failed_cuda_driver_queries_are_explicit() {
        let mut fingerprint = hardware_fingerprint_with_source(
            &Device::Cpu,
            &FakeProbeSource::default().with("/proc/cpuinfo", "processor: 0\n"),
        );
        apply_cuda_fingerprint_details(&mut fingerprint, CudaFingerprintDetails::default());

        assert_eq!(
            fingerprint.cuda_compute_capability_unavailable_reason,
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed)
        );
        assert_eq!(
            fingerprint.cuda_device_uuid_unavailable_reason,
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed)
        );
        assert_eq!(
            fingerprint.cuda_pci_bus_id_unavailable_reason,
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed)
        );
        assert_eq!(
            fingerprint.cuda_driver_api_version_unavailable_reason,
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed)
        );
        assert_eq!(
            fingerprint.cuda_binding_api_version_unavailable_reason,
            Some(FingerprintUnavailableReason::CudarcBindingVersionUnavailable)
        );
    }

    #[test]
    fn cuda_driver_source_is_injectable_and_toolkit_file_is_not_runtime_identity() {
        let source = FakeProbeSource::default()
            .with(
                "/proc/driver/nvidia/version",
                "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.124.06\n",
            )
            .with("/usr/local/cuda/version.txt", "CUDA Version 99.0\n");
        let mut fingerprint = hardware_fingerprint_with_source(
            &Device::Cpu,
            &FakeProbeSource::default().with("/proc/cpuinfo", "processor: 0\n"),
        );
        populate_cuda_driver_version(&source, &mut fingerprint);

        assert_eq!(fingerprint.driver_version.as_deref(), Some("570.124.06"));
        assert_eq!(
            fingerprint.driver_version_source,
            Some(FingerprintValueSource::ProcfsNvidiaDriver)
        );
        assert_eq!(fingerprint.runtime_version, None);
        assert_eq!(fingerprint.runtime_version_source, None);

        populate_cuda_driver_version(
            &FakeProbeSource::default().with(
                "/proc/driver/nvidia/version",
                "present but not an NVRM version record\n",
            ),
            &mut fingerprint,
        );
        assert_eq!(fingerprint.driver_version, None);
        assert_eq!(fingerprint.driver_version_source, None);
        assert_eq!(
            fingerprint.driver_version_unavailable_reason,
            Some(FingerprintUnavailableReason::ProcfsNvidiaDriverParseFailed)
        );

        populate_cuda_driver_version(&FakeProbeSource::default(), &mut fingerprint);
        assert_eq!(
            fingerprint.driver_version_unavailable_reason,
            Some(FingerprintUnavailableReason::ProcfsUnavailable)
        );
    }

    #[test]
    fn cuda_uuid_has_a_canonical_text_encoding() {
        assert_eq!(
            format_cuda_uuid([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            "00010203-0405-0607-0809-0a0b0c0d0e0f"
        );
    }

    #[test]
    fn public_cpu_probe_works_without_a_gpu() {
        let fingerprint = HardwareFingerprint::collect(&Device::Cpu);
        let snapshot = ResourceSnapshot::capture(Some(&Device::Cpu));
        assert_eq!(fingerprint.backend, DeviceBackend::Cpu);
        assert!(fingerprint.logical_cpu_count > 0);
        fingerprint.validate().unwrap();
        assert!(!fingerprint.supports_calibration_cache_reuse());
        assert_eq!(snapshot.schema_version, RESOURCE_SNAPSHOT_SCHEMA_VERSION);
        assert!(snapshot.measured_at_unix_ms > 0);
        assert_eq!(snapshot.device_free_memory_bytes, None);
        assert_eq!(snapshot.measurement_scope.device_memory, None);
    }

    #[test]
    fn json_round_trip_preserves_versioned_probe_records() {
        let fingerprint = HardwareFingerprint::collect(&Device::Cpu);
        let encoded = serde_json::to_vec(&fingerprint).unwrap();
        let decoded = HardwareFingerprint::from_json(&encoded).unwrap();
        assert_eq!(decoded, fingerprint);

        let mut stale = serde_json::to_value(&fingerprint).unwrap();
        stale["schema_version"] = serde_json::json!(2);
        assert!(HardwareFingerprint::from_json(&serde_json::to_vec(&stale).unwrap()).is_err());

        let mut incomplete = serde_json::to_value(&fingerprint).unwrap();
        incomplete
            .as_object_mut()
            .unwrap()
            .remove("cuda_device_uuid");
        assert!(HardwareFingerprint::from_json(&serde_json::to_vec(&incomplete).unwrap()).is_err());

        let snapshot = ResourceSnapshot::capture(None);
        let encoded = serde_json::to_vec(&snapshot).unwrap();
        let decoded: ResourceSnapshot = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, snapshot);
    }
}
