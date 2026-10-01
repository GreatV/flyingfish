use anyhow::{Context, Result, ensure};
use candle_core::{Device, DeviceLocation};
use serde::{Deserialize, Serialize};
use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// A measured sample range in milliseconds.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProbeRange {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

impl ProbeRange {
    fn from_samples(samples: &mut [f64]) -> Result<Self> {
        ensure!(
            !samples.is_empty() && samples.iter().all(|v| v.is_finite() && *v > 0.0),
            "probe samples must be nonempty, finite and positive"
        );
        samples.sort_by(f64::total_cmp);
        Ok(Self {
            min: samples[0],
            median: samples[samples.len() / 2],
            max: samples[samples.len() - 1],
        })
    }
}

/// Warm each candidate, then time candidates in interleaved order.
pub fn probe(
    candidates: usize,
    mut prepare: impl FnMut(usize) -> Result<()>,
    mut run: impl FnMut(usize) -> Result<()>,
    mut fence: impl FnMut() -> Result<()>,
    mut elapsed: impl FnMut(usize, Duration) -> Result<f64>,
) -> Result<Vec<ProbeRange>> {
    ensure!(candidates > 0, "probe has no candidates");
    let mut samples = vec![Vec::with_capacity(12); candidates];
    for rep in 0..15 {
        for (i, values) in samples.iter_mut().enumerate() {
            prepare(i).with_context(|| format!("probe candidate {i} preparation failed"))?;
            fence().with_context(|| format!("probe candidate {i} preparation fence failed"))?;
            let start = Instant::now();
            run(i).with_context(|| format!("probe candidate {i} launch failed"))?;
            fence().with_context(|| format!("probe candidate {i} completion fence failed"))?;
            let ms = elapsed(i, start.elapsed())
                .with_context(|| format!("probe candidate {i} duration measurement failed"))?;
            ensure!(
                ms.is_finite() && ms > 0.0,
                "probe candidate {i} duration is invalid: {ms}"
            );
            if rep >= 3 {
                values.push(ms);
            }
        }
    }
    samples
        .iter_mut()
        .enumerate()
        .map(|(i, values)| {
            ProbeRange::from_samples(values)
                .with_context(|| format!("probe candidate {i} sample range failed"))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayProbe {
    pub batches: usize,
    pub ranges: Vec<ProbeRange>,
    pub repeats: usize,
    pub pilot_spread: f64,
    pub spread: f64,
}

pub fn probe_replays(
    candidates: usize,
    capacity: usize,
    mut prepare: impl FnMut(usize) -> Result<()>,
    mut run: impl FnMut(usize, usize) -> Result<()>,
    mut fence: impl FnMut() -> Result<()>,
    mut elapsed: impl FnMut(usize, Duration) -> Result<f64>,
) -> Result<ReplayProbe> {
    ensure!(capacity > 0, "decode probe has no replay-state capacity");
    let mut repeats = 1;
    let mut pilot_spread = None;
    let mut batches = 0;
    loop {
        batches += 1;
        let mut ranges = probe(
            candidates,
            &mut prepare,
            |i| run(i, repeats),
            &mut fence,
            &mut elapsed,
        )?;
        for r in &mut ranges {
            r.min /= repeats as f64;
            r.median /= repeats as f64;
            r.max /= repeats as f64;
        }
        let spread = ranges
            .iter()
            .map(|r| (r.max - r.min) / r.median)
            .fold(0.0_f64, f64::max);
        let pilot = *pilot_spread.get_or_insert(spread);
        if spread <= 0.01 {
            return Ok(ReplayProbe {
                batches,
                ranges,
                repeats,
                pilot_spread: pilot,
                spread,
            });
        }
        let required = (repeats as f64 * (spread / 0.01).powi(2)).ceil();
        ensure!(
            required.is_finite() && required <= capacity as f64,
            "decode probe needs {required} replays to resolve 1%, state capacity {capacity}, measured spread {spread}"
        );
        let required = required as usize;
        repeats = required.max(
            repeats
                .checked_add(1)
                .context("decode probe replay count overflow")?,
        );
        ensure!(
            repeats <= capacity,
            "decode probe needs {repeats} replays, state capacity {capacity}"
        );
    }
}

/// Prefer a measured winner only when its whole range is faster.
pub fn probe_choice(ranges: &[ProbeRange], preferred: Option<usize>) -> Result<(usize, bool)> {
    ensure!(!ranges.is_empty(), "probe selection has no candidates");
    if let Some(i) = preferred {
        ensure!(i < ranges.len(), "probe preferred candidate {i} is absent");
    }
    for (i, r) in ranges.iter().enumerate() {
        ensure!(
            r.min.is_finite()
                && r.median.is_finite()
                && r.max.is_finite()
                && r.min > 0.0
                && r.min <= r.median
                && r.median <= r.max,
            "probe candidate {i} range is invalid: {r:?}"
        );
    }
    let best = (0..ranges.len())
        .min_by(|&a, &b| {
            ranges[a]
                .median
                .total_cmp(&ranges[b].median)
                .then(a.cmp(&b))
        })
        .context("probe selection has no median")?;
    let separated = ranges
        .iter()
        .enumerate()
        .all(|(i, r)| i == best || ranges[best].max < r.min);
    Ok((
        if separated {
            best
        } else {
            preferred.unwrap_or(best)
        },
        separated,
    ))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeChoice {
    pub adapter: String,
    pub geometry: serde_json::Value,
    pub class: String,
    pub settings: serde_json::Value,
    pub split: Vec<Vec<(usize, usize, usize, usize)>>,
    pub fingerprint: Option<HardwareFingerprint>,
    pub binary: Option<crate::identity::BinaryIdentity>,
    pub image: String,
    pub trials: Vec<ReplayProbe>,
    pub device: usize,
    pub cols: u32,
    pub derived: String,
    pub body: String,
    pub reason: String,
    pub stock: ProbeRange,
    pub xr16: ProbeRange,
    pub repeats: usize,
    pub warmups: usize,
    pub samples: usize,
    pub capacity: usize,
    pub state_bytes: usize,
    pub seed_token: u32,
    pub topology: String,
    pub pilot_spread: f64,
    pub spread: f64,
    pub capture_ms: f64,
    pub replay_ms: f64,
    pub setup_ms: f64,
}

impl DecodeChoice {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            ["qwen35", "edge0"].contains(&self.adapter.as_str()),
            "unknown group4 adapter; run ff bench group4"
        );
        ensure!(
            [
                "resident_graph",
                "streamed",
                "host_expert",
                "fully_streamed"
            ]
            .contains(&self.class.as_str()),
            "unknown group4 execution class; run ff bench group4"
        );
        ensure!(
            [1, 2].contains(&self.cols) && self.geometry.is_object() && self.settings.is_object(),
            "invalid group4 program key; run ff bench group4"
        );
        ensure!(
            ["stock", "xr16"].contains(&self.body.as_str()) && self.body == self.derived,
            "invalid persisted group4 body; run ff bench group4"
        );
        ensure!(
            self.image.len() == 64
                && self
                    .image
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "group4 kernel image digest missing/invalid; run ff bench group4"
        );
        let fingerprint = self
            .fingerprint
            .as_ref()
            .context("group4 fingerprint missing; run ff bench group4")?;
        fingerprint.validate()?;
        ensure!(
            fingerprint.backend == DeviceBackend::Cuda,
            "group4 fingerprint is not CUDA; run ff bench group4"
        );
        for (field, present) in [
            (
                "operating_system_version",
                fingerprint.operating_system_version.is_some(),
            ),
            ("device_name", fingerprint.device_name.is_some()),
            (
                "device_total_memory_bytes",
                fingerprint.device_total_memory_bytes.is_some(),
            ),
            ("driver_version", fingerprint.driver_version.is_some()),
            (
                "cuda_compute_capability",
                fingerprint.cuda_compute_capability.is_some(),
            ),
            ("cuda_device_uuid", fingerprint.cuda_device_uuid.is_some()),
            (
                "cuda_driver_api_version",
                fingerprint.cuda_driver_api_version.is_some(),
            ),
            (
                "cuda_binding_api_version",
                fingerprint.cuda_binding_api_version.is_some(),
            ),
        ] {
            ensure!(
                present,
                "group4 fingerprint {field} missing; run ff bench group4"
            );
        }
        self.binary
            .as_ref()
            .context("group4 binary identity missing; run ff bench group4")?
            .validate()?;
        ensure!(
            self.trials.len() == 3
                && self.split.len() == 3
                && self.split.iter().all(|s| !s.is_empty()
                    && s.iter()
                        .all(|&(_, a, resident, b)| a <= resident && resident <= b && a < b)),
            "group4 needs three independent calibration trials; run ff bench group4"
        );
        let mut unanimous = true;
        for trial in &self.trials {
            ensure!(
                trial.ranges.len() == 2
                    && trial.repeats > 0
                    && trial.batches > 0
                    && trial.spread.is_finite()
                    && trial.spread <= 0.01
                    && trial.spread >= 0.0
                    && trial.pilot_spread.is_finite()
                    && trial.pilot_spread >= 0.0,
                "invalid group4 trial; run ff bench group4"
            );
            let (body, separated) = probe_choice(&trial.ranges, Some(0))?;
            unanimous &= body == 1 && separated;
        }
        ensure!(
            self.body == if unanimous { "xr16" } else { "stock" },
            "group4 body disagrees with independent evidence; run ff bench group4"
        );
        Ok(())
    }

    pub fn key(&self) -> serde_json::Value {
        serde_json::json!({"adapter":self.adapter,"device":self.device,"cols":self.cols,"geometry":self.geometry,"class":self.class,"settings":self.settings,"fingerprint":self.fingerprint,"binary":self.binary.as_ref().map(|b| serde_json::json!({"schema_version":b.schema_version,"package_name":b.package_name,"package_version":b.package_version,"compiled_features":b.compiled_features})),"image":self.image})
    }

    pub fn set_key(
        &mut self,
        key: &serde_json::Value,
        split: Vec<(usize, usize, usize, usize)>,
    ) -> Result<()> {
        self.adapter = serde_json::from_value(key["adapter"].clone())?;
        self.geometry = key["geometry"].clone();
        self.class = serde_json::from_value(key["class"].clone())?;
        self.settings = key["settings"].clone();
        self.fingerprint = Some(serde_json::from_value(key["fingerprint"].clone())?);
        self.binary = Some(serde_json::from_value(key["binary"].clone())?);
        self.image = serde_json::from_value(key["image"].clone())?;
        self.split = vec![split];
        Ok(())
    }

    pub fn combine(mut trials: Vec<Self>) -> Result<Self> {
        ensure!(
            trials.len() == 3,
            "group4 needs three independent calibrations"
        );
        let mut record = trials.remove(0);
        ensure!(
            trials.iter().all(|t| record.matches(t)),
            "group4 execution class changed during calibration; rerun ff bench group4"
        );
        for trial in trials {
            record.capture_ms += trial.capture_ms;
            record.replay_ms += trial.replay_ms;
            record.setup_ms += trial.setup_ms;
            record.trials.extend(trial.trials);
            record.split.extend(trial.split);
        }
        let choices = record
            .trials
            .iter()
            .map(|t| probe_choice(&t.ranges, Some(0)))
            .collect::<Result<Vec<_>>>()?;
        let xr16 = choices
            .iter()
            .all(|&(body, separated)| body == 1 && separated);
        let agreed = choices
            .iter()
            .all(|&(body, separated)| body == choices[0].0 && separated);
        record.body = if xr16 { "xr16" } else { "stock" }.into();
        record.derived = record.body.clone();
        record.reason = if agreed {
            "independent separated trials agree"
        } else {
            "overlap or independent disagreement"
        }
        .into();
        record.validate()?;
        Ok(record)
    }

    pub fn matches(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

pub fn image_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

pub const HARDWARE_FINGERPRINT_SCHEMA_VERSION: u32 = 1;
pub const RESOURCE_SNAPSHOT_SCHEMA_VERSION: u32 = 2;

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
    pub fn collect(device: &Device) -> Result<Self> {
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
    /// switches admission to combined-pool accounting; `None` records a
    /// non-CUDA backend or a topology the probe could not classify; a
    /// confirmed-discrete `Some(false)` is split accounting. The field is
    /// required in stored records: a record without it predates the probe
    /// and must be recaptured.
    #[serde(deserialize_with = "crate::required_option")]
    pub host_device_memory_is_unified: Option<bool>,
    /// Hardware-level totals (MemTotal / device total memory), unlike the
    /// instantaneous `*_available`/`device_free` views. Reserve sizes scale
    /// with totals so that a run's margin does not shrink when the machine is
    /// busy and sidecar-recorded reserves stay comparable across runs.
    #[serde(deserialize_with = "crate::required_option")]
    pub host_memory_total_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub device_total_memory_bytes: Option<u64>,
    pub measurement_scope: ResourceMeasurementScopes,
}

impl ResourceSnapshot {
    pub fn capture(device: Option<&Device>) -> anyhow::Result<Self> {
        resource_snapshot_with_source(device, &SystemProbeSource, unix_time_ms()?)
            .context("resource probe failed")
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
        // Both binding views are required; an absent cgroup limit is not one.
        let pool = self
            .host_memory_available_bytes?
            .min(self.device_free_memory_bytes?);
        Some(match self.cgroup_v2_memory_available_bytes {
            Some(limit) => pool.min(limit),
            None => pool,
        })
    }

    /// The host pool a run is confined to: physical memory, or a smaller
    /// finite cgroup limit. Both are totals, so scaled reserves stay stable.
    pub fn host_pool_total_bytes(&self) -> Option<u64> {
        let limit = match self.cgroup_v2_memory_limit {
            Some(CgroupMemoryLimit::Bytes(bytes)) => Some(bytes),
            Some(CgroupMemoryLimit::Unlimited) | None => None,
        };
        match (self.host_memory_total_bytes, limit) {
            (Some(host), Some(limit)) => Some(host.min(limit)),
            (host, limit) => host.or(limit),
        }
    }

    pub fn pool_total(&self, device: bool) -> Result<u64> {
        let host = || {
            self.host_memory_total_bytes
                .filter(|&n| n > 0)
                .map(|total| match self.cgroup_v2_memory_limit {
                    Some(CgroupMemoryLimit::Bytes(limit)) => total.min(limit),
                    _ => total,
                })
                .context("host_memory_total_bytes unavailable; recapture with ff probe --json")
        };
        if !device {
            return host();
        }
        let unified = self.host_device_memory_is_unified.context("host_device_memory_is_unified unavailable; recapture with ff probe --device cuda:N --json")?;
        let total = self.device_total_memory_bytes.filter(|&n| n > 0).context(
            "device_total_memory_bytes unavailable; recapture with ff probe --device cuda:N --json",
        )?;
        if unified {
            Ok(total.min(host()?))
        } else {
            Ok(total)
        }
    }

    pub fn device_capacity(&self) -> Result<u64> {
        let unified = self.host_device_memory_is_unified.context("host_device_memory_is_unified unavailable; recapture with ff probe --device cuda:N --json")?;
        if unified {
            self.unified_pool_available_bytes().context("unified_pool_available_bytes unavailable; recapture with ff probe --device cuda:N --json")
        } else {
            self.device_free_memory_bytes.context("device_free_memory_bytes unavailable; recapture with ff probe --device cuda:N --json")
        }
    }

    /// Whether admission must refuse rather than fall back to per-axis checks:
    /// a shared pool of unknown size, or a failed live topology query.
    pub fn unified_accounting_is_undecidable(&self) -> bool {
        self.unified_pool_is_unmeasurable()
            || (self.host_device_memory_is_unified.is_none()
                && self.measurement_scope.device_memory.is_some())
    }

    /// Whether a confirmed shared pool's size could not be measured.
    pub fn unified_pool_is_unmeasurable(&self) -> bool {
        self.host_device_memory_is_unified == Some(true)
            && self.unified_pool_available_bytes().is_none()
    }
}

/// Cap every pool-scaled admission reserve converges to: 1 GiB.
pub const ADMISSION_RESERVE_CAP_BYTES: u64 = 1 << 30;

/// Minimum allowance for fixed context and allocator costs.
pub const ADMISSION_RESERVE_FLOOR_BYTES: u64 = 512 << 20;

/// Reserve 5% of the pool total, between 512 MiB and 1 GiB; an unknown total uses the cap.
pub fn admission_reserve_bytes(pool_total: Option<u64>) -> u64 {
    pool_total.map_or(ADMISSION_RESERVE_CAP_BYTES, |total| {
        ADMISSION_RESERVE_FLOOR_BYTES.max(ADMISSION_RESERVE_CAP_BYTES.min(total / 20))
    })
}

/// Device-pool safety allowance for allocation rounding and unobserved transient usage.
pub const DEVICE_RESERVE_SAFETY_BYTES: u64 = 64 << 20;

/// The device-pool admission reserve: a fixed safety term over the charged
/// transient planes. It does not scale with the pool — the planner's charge
/// already carries the pool-dependent terms.
pub fn device_admission_reserve_bytes() -> u64 {
    DEVICE_RESERVE_SAFETY_BYTES
}

/// Whether two fingerprints describe the same machine.
///
/// The CUDA device UUID is the strongest identifier available; the host fields
/// catch a profile carried between machines that happen to hold the same card
/// model.
pub fn describes_same_machine(
    recorded: &HardwareFingerprint,
    current: &HardwareFingerprint,
) -> bool {
    recorded.validate().is_ok()
        && current.validate().is_ok()
        && recorded.backend == current.backend
        && recorded.architecture == current.architecture
        && recorded.operating_system == current.operating_system
        && recorded.logical_cpu_count == current.logical_cpu_count
        && recorded.device_name == current.device_name
        && recorded.device_total_memory_bytes == current.device_total_memory_bytes
        && match (&recorded.cuda_device_uuid, &current.cuda_device_uuid) {
            (Some(recorded), Some(current)) => recorded == current,
            (None, None) => recorded.backend != DeviceBackend::Cuda,
            _ => false,
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

    /// Host-wide total physical memory, for platforms outside the filesystem.
    fn host_total_memory_bytes(&self) -> Option<u64> {
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
        windows_physical_memory_bytes().map(|(available, _)| available)
    }

    #[cfg(windows)]
    fn host_total_memory_bytes(&self) -> Option<u64> {
        windows_physical_memory_bytes().map(|(_, total)| total)
    }
}

/// Windows' host-wide available and total physical memory, in that order.
///
/// `ullAvailPhys` is the closest counterpart to Linux's `MemAvailable`: both
/// answer "how much can a new allocation expect without paging". They are not
/// computed the same way — `MemAvailable` includes reclaimable page cache,
/// `ullAvailPhys` reports free physical pages — so a snapshot is comparable
/// across runs on one host, not across platforms. `ullTotalPhys` is the
/// hardware counterpart to `MemTotal`, which is why reserves scale from it.
#[cfg(windows)]
fn windows_physical_memory_bytes() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = unsafe { std::mem::zeroed::<MEMORYSTATUSEX>() };
    status.dwLength = u32::try_from(size_of::<MEMORYSTATUSEX>()).ok()?;
    (unsafe { GlobalMemoryStatusEx(&raw mut status) } != 0)
        .then_some((status.ullAvailPhys, status.ullTotalPhys))
}

fn hardware_fingerprint_with_source(
    source_device: &Device,
    source: &impl ProbeSource,
) -> Result<HardwareFingerprint> {
    let cpu_info = source.read_to_string(Path::new("/proc/cpuinfo")).ok();
    let logical_cpu_count = cpu_count(
        cpu_info.as_deref(),
        std::thread::available_parallelism().ok().map(|n| n.get()),
    )?;
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
    Ok(fingerprint)
}

fn resource_snapshot_with_source(
    device: Option<&Device>,
    source: &impl ProbeSource,
    measured_at_unix_ms: u64,
) -> anyhow::Result<ResourceSnapshot> {
    let host_memory_available_bytes = source
        .read_to_string(Path::new("/proc/meminfo"))
        .ok()
        .as_deref()
        .and_then(|contents| parse_meminfo_bytes(contents, "MemAvailable"))
        .or_else(|| source.host_available_memory_bytes());
    let views = read_cgroup_v2_memory(source)?.unwrap_or(CgroupMemoryViews {
        limit: None,
        current: None,
        available: None,
    });
    let (cgroup_v2_memory_limit, cgroup_v2_memory_current_bytes, cgroup_v2_memory_available_bytes) =
        (views.limit, views.current, views.available);
    let device_free_memory_bytes = device.and_then(device_free_memory);
    let host_device_memory_is_unified = device.map(device_memory_is_unified).transpose()?.flatten();
    // Live topology-query errors propagate from capture.
    let host_memory_total_bytes = source
        .read_to_string(Path::new("/proc/meminfo"))
        .ok()
        .as_deref()
        .and_then(|contents| parse_meminfo_bytes(contents, "MemTotal"))
        .or_else(|| source.host_total_memory_bytes());
    let device_total_memory_bytes = device.and_then(device_total_memory);

    Ok(ResourceSnapshot {
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
    })
}

fn cpu_count(cpu_info: Option<&str>, platform_count: Option<usize>) -> Result<u64> {
    let count = match cpu_info {
        Some(info) => parse_logical_cpu_count(info),
        None => platform_count,
    }
    .filter(|&count| count > 0)
    .context("logical_cpu_count unavailable; run ff probe --device cpu --json")?;
    u64::try_from(count).context("logical_cpu_count exceeds u64; run ff probe --device cpu --json")
}

fn unix_time_ms() -> Result<u64> {
    timestamp_ms(SystemTime::now())
}

fn timestamp_ms(time: SystemTime) -> Result<u64> {
    let elapsed = time.duration_since(UNIX_EPOCH).context(
        "measured_at_unix_ms precedes the epoch; correct the clock and run ff probe --json",
    )?;
    u64::try_from(elapsed.as_millis())
        .context("measured_at_unix_ms exceeds u64; correct the clock and run ff probe --json")
}

pub fn env_value<T: std::str::FromStr>(name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    parse_env(name, std::env::var_os(name))
}

fn parse_env<T: std::str::FromStr>(
    name: &str,
    value: Option<std::ffi::OsString>,
) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    value
        .map(|value| {
            let value = value.into_string().map_err(|_| {
                anyhow::anyhow!("{name} is not Unicode; supply a valid value and rerun ff")
            })?;
            value.parse::<T>().map_err(|error| {
                anyhow::anyhow!(
                    "invalid {name}={value:?}: {error}; supply a valid value and rerun ff"
                )
            })
        })
        .transpose()
}

pub fn env_usize(name: &str, default: usize, min: usize, max: usize) -> Result<usize> {
    parse_env_usize(name, std::env::var_os(name), default, min, max)
}

fn parse_env_usize(
    name: &str,
    source: Option<std::ffi::OsString>,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize> {
    let value = parse_env(name, source)?.unwrap_or(default);
    anyhow::ensure!(
        (min..=max).contains(&value),
        "{name} must be in {min}..={max}, got {value}; supply a valid value and rerun ff"
    );
    Ok(value)
}

pub fn cached_env_usize(
    cache: &std::sync::OnceLock<std::result::Result<usize, String>>,
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize> {
    cache
        .get_or_init(|| env_usize(name, default, min, max).map_err(|error| error.to_string()))
        .as_ref()
        .copied()
        .map_err(|error| anyhow::anyhow!(error.clone()))
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

/// The cgroup v2 views the admission reads: the effective limit, the leaf
/// usage, and the usage net of reclaimable file cache.
pub(crate) struct CgroupMemoryViews {
    limit: Option<CgroupMemoryLimit>,
    current: Option<u64>,
    available: Option<u64>,
}

fn read_cgroup_v2_memory(source: &impl ProbeSource) -> anyhow::Result<Option<CgroupMemoryViews>> {
    // Missing /proc entries mean the OS has no cgroup v2 hierarchy at all:
    // not applicable, not a failure.
    let cgroup = match source.read_to_string(Path::new("/proc/self/cgroup")) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(anyhow::anyhow!(
                "cgroup v2 probe: read /proc/self/cgroup: {error}"
            ));
        }
    };
    let mountinfo = match source.read_to_string(Path::new("/proc/self/mountinfo")) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(anyhow::anyhow!(
                "cgroup v2 probe: read /proc/self/mountinfo: {error}"
            ));
        }
    };
    let Some((mount_point, directory)) = resolve_cgroup_v2_directory(&cgroup, &mountinfo) else {
        return Ok(None);
    };
    let leaf_current = read_cgroup_counter(source, &directory.join("memory.current"))?;

    let mut finite_limit: Option<u64> = None;
    let mut effective_available: Option<u64> = None;
    for ancestor in directory.ancestors() {
        if !ancestor.starts_with(&mount_point) {
            break;
        }
        let limit = read_cgroup_limit(source, &ancestor.join("memory.max"))?;
        let Some(limit) = limit else {
            if ancestor == mount_point {
                break;
            }
            anyhow::bail!(
                "cgroup v2 probe: {} has no readable memory.max",
                ancestor.display()
            );
        };
        // memory.high throttles before memory.max; the effective ceiling is
        // the lower of whichever of the two is finite. The file itself is
        // optional: no memory.high means no throttle.
        let high = match source.read_to_string(&ancestor.join("memory.high")) {
            Ok(text) => match parse_cgroup_memory_limit(&text) {
                Some(limit) => Some(limit),
                None => {
                    return Err(anyhow::anyhow!(
                        "cgroup v2 probe: unparsable memory.high in {}",
                        ancestor.join("memory.high").display()
                    ));
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "cgroup v2 probe: read {}: {error}",
                    ancestor.join("memory.high").display()
                ));
            }
        };
        let limit = match high {
            Some(CgroupMemoryLimit::Bytes(high)) => match limit {
                CgroupMemoryLimit::Bytes(max) => CgroupMemoryLimit::Bytes(max.min(high)),
                CgroupMemoryLimit::Unlimited => CgroupMemoryLimit::Bytes(high),
            },
            _ => limit,
        };
        if let CgroupMemoryLimit::Bytes(bytes) = limit {
            finite_limit = Some(finite_limit.map_or(bytes, |current| current.min(bytes)));
            let current = read_cgroup_counter(source, &ancestor.join("memory.current"))?;
            let reclaimable = source
                .read_to_string(&ancestor.join("memory.stat"))
                .map_err(|error| {
                    anyhow::anyhow!(
                        "cgroup v2 probe: read {}: {error}",
                        ancestor.join("memory.stat").display()
                    )
                })?;
            let reclaimable = cgroup_reclaimable_file_bytes(&reclaimable)
                .context("cgroup v2 probe: unparsable memory.stat")?
                .min(current);
            let available = bytes.saturating_sub(current - reclaimable);
            effective_available =
                Some(effective_available.map_or(available, |current| current.min(available)));
        }
        if ancestor == mount_point {
            break;
        }
    }
    let effective_limit = finite_limit
        .map(CgroupMemoryLimit::Bytes)
        .unwrap_or(CgroupMemoryLimit::Unlimited);
    Ok(Some(CgroupMemoryViews {
        limit: Some(effective_limit),
        current: Some(leaf_current),
        available: effective_available,
    }))
}

/// One cgroup control file whose content is a byte count or the literal
/// `max`. `Ok(None)` is a clean unlimited value; anything the OS cannot
/// read is an error naming the file.
fn read_cgroup_limit(
    source: &impl ProbeSource,
    path: &Path,
) -> anyhow::Result<Option<CgroupMemoryLimit>> {
    // A missing limit file means the controller exposes no ceiling here:
    // clean unlimited, not a failure.
    let text = match source.read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(anyhow::anyhow!(
                "cgroup v2 probe: read {}: {error}",
                path.display()
            ));
        }
    };
    parse_cgroup_memory_limit(&text)
        .context("cgroup v2 probe: unparsable memory limit")
        .map(Some)
}

fn read_cgroup_counter(source: &impl ProbeSource, path: &Path) -> anyhow::Result<u64> {
    let text = source
        .read_to_string(path)
        .map_err(|error| anyhow::anyhow!("cgroup v2 probe: read {}: {error}", path.display()))?;
    text.trim()
        .parse::<u64>()
        .context("cgroup v2 probe: unparsable byte count")
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

/// Directional peer reachability from the driver's own device query.
///
/// The answer says only whether a peer mapping is possible; it says nothing
/// about the link's class or speed.
#[cfg(feature = "cuda")]
pub fn can_access_peer(device: u32, peer_device: u32) -> Option<bool> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let mut reachable: i32 = 0;
    let status = unsafe {
        sys::cuDeviceCanAccessPeer(
            &mut reachable,
            device as sys::CUdevice,
            peer_device as sys::CUdevice,
        )
    };
    if status != sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    Some(reachable != 0)
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
fn device_memory_is_unified(device: &Device) -> anyhow::Result<Option<bool>> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let Ok(cuda) = device.as_cuda_device() else {
        return Ok(None);
    };
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
        return Ok(Some(value != 0));
    }
    Err(anyhow::anyhow!(
        "CUDA integrated-topology attribute query failed: {result:?}"
    ))
}

#[cfg(not(feature = "cuda"))]
fn device_memory_is_unified(_device: &Device) -> anyhow::Result<Option<bool>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_measurements_name_the_field_and_recapture_command() {
        let error = cpu_count(None, None).unwrap_err().to_string();
        assert!(error.contains("logical_cpu_count") && error.contains("ff probe"));
        let error = timestamp_ms(UNIX_EPOCH - Duration::from_secs(1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("measured_at_unix_ms") && error.contains("ff probe"));
        let mut snapshot = ResourceSnapshot::capture(None).unwrap();
        snapshot.host_memory_total_bytes = None;
        let error = snapshot.pool_total(false).unwrap_err().to_string();
        assert!(error.contains("host_memory_total_bytes") && error.contains("ff probe"));
        snapshot.host_device_memory_is_unified = Some(false);
        snapshot.device_total_memory_bytes = None;
        let error = snapshot.pool_total(true).unwrap_err().to_string();
        assert!(error.contains("device_total_memory_bytes") && error.contains("ff probe"));
        snapshot.device_free_memory_bytes = None;
        let error = snapshot.device_capacity().unwrap_err().to_string();
        assert!(error.contains("device_free_memory_bytes") && error.contains("ff probe"));
        snapshot.device_free_memory_bytes = Some(0);
        assert_eq!(snapshot.device_capacity().unwrap(), 0);
        snapshot.host_device_memory_is_unified = None;
        let error = snapshot.device_capacity().unwrap_err().to_string();
        assert!(error.contains("host_device_memory_is_unified") && error.contains("ff probe"));
    }

    #[test]
    fn invalid_overrides_error_instead_of_selecting_defaults() {
        for value in ["wrong", "0", "9"] {
            let error = parse_env_usize("FF_GLM_LOAD_LANES", Some(value.into()), 1, 1, 8)
                .unwrap_err()
                .to_string();
            assert!(error.contains("FF_GLM_LOAD_LANES") && error.contains("rerun ff"));
        }
        for field in ["FF_GLM_DIRECT_FILL", "FF_GLM_IO_TRACE", "QWEN35_GRAPH"] {
            let error = parse_env_usize(field, Some("2".into()), 0, 0, 1)
                .unwrap_err()
                .to_string();
            assert!(error.contains(field) && error.contains("rerun ff"));
            assert_eq!(
                parse_env_usize(field, Some("0".into()), 1, 0, 1).unwrap(),
                0
            );
        }
        assert_eq!(
            parse_env_usize("FF_GLM_LOAD_LANES", None, 1, 1, 8).unwrap(),
            1
        );
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let error = parse_env::<usize>(
                "FF_WEIGHT_LOAD_THREADS",
                Some(std::ffi::OsString::from_vec(vec![255])),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("FF_WEIGHT_LOAD_THREADS") && error.contains("rerun ff"));
        }
    }

    #[test]
    fn snapshots_require_explicit_total_fields() {
        let snapshot = ResourceSnapshot::capture(None).unwrap();
        for field in ["host_memory_total_bytes", "device_total_memory_bytes"] {
            let mut value = serde_json::to_value(&snapshot).unwrap();
            value.as_object_mut().unwrap().remove(field);
            let error = serde_json::from_value::<ResourceSnapshot>(value)
                .unwrap_err()
                .to_string();
            assert!(error.contains(field));
        }
    }

    #[test]
    fn probe_range_rejects_invalid_samples() {
        for mut values in [
            vec![],
            vec![0.0],
            vec![-1.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(ProbeRange::from_samples(&mut values).is_err());
        }
        let range = ProbeRange::from_samples(&mut [4.0, 1.0, 2.0]).unwrap();
        assert_eq!((range.min, range.median, range.max), (1.0, 2.0, 4.0));
    }

    #[test]
    fn probe_choice_requires_separated_ranges() {
        let cases = [
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], Some(1), (0, true)),
            ([4.0, 5.0, 6.0], [1.0, 2.0, 3.0], Some(0), (1, true)),
            ([4.0, 6.0, 8.0], [3.5, 3.7, 5.0], Some(0), (0, false)),
            ([4.0, 5.0, 6.0], [1.0, 2.0, 4.0], Some(0), (0, false)),
            ([1.0, 4.0, 9.0], [2.0, 3.0, 5.0], None, (1, false)),
            ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], Some(1), (1, false)),
        ];
        for (a, b, preferred, expected) in cases {
            let ranges = [a, b].map(|[min, median, max]| ProbeRange { min, median, max });
            assert_eq!(probe_choice(&ranges, preferred).unwrap(), expected);
        }
        let ranges = [
            ProbeRange {
                min: 1.0,
                median: 2.0,
                max: 4.0,
            },
            ProbeRange {
                min: 5.0,
                median: 6.0,
                max: 7.0,
            },
            ProbeRange {
                min: 3.0,
                median: 8.0,
                max: 9.0,
            },
        ];
        assert_eq!(probe_choice(&ranges, Some(1)).unwrap(), (1, false));
    }

    #[test]
    fn probe_choice_rejects_invalid_ranges() {
        assert!(probe_choice(&[], None).is_err());
        let valid = ProbeRange {
            min: 1.0,
            median: 2.0,
            max: 3.0,
        };
        assert!(probe_choice(&[valid], Some(1)).is_err());
        assert_eq!(probe_choice(&[valid], Some(0)).unwrap(), (0, true));
        for bad in [
            ProbeRange { min: 0.0, ..valid },
            ProbeRange {
                median: 0.5,
                ..valid
            },
            ProbeRange { max: 1.5, ..valid },
            ProbeRange {
                median: f64::NAN,
                ..valid
            },
            ProbeRange {
                max: f64::INFINITY,
                ..valid
            },
        ] {
            assert!(probe_choice(&[bad], None).is_err());
        }
    }

    #[test]
    fn probe_interleaves_and_stops_on_error() {
        let mut order = Vec::new();
        let mut fences = 0;
        let ranges = probe(
            2,
            |_| Ok(()),
            |i| {
                order.push(i);
                Ok(())
            },
            || {
                fences += 1;
                Ok(())
            },
            |_, d| Ok(d.as_secs_f64() * 1000.0),
        )
        .unwrap();
        assert_eq!(order, [0, 1].repeat(15));
        assert_eq!(fences, 60);
        assert_eq!(ranges.len(), 2);
        order.clear();
        let err = probe(
            2,
            |_| Ok(()),
            |i| {
                order.push(i);
                ensure!(i != 1, "fixture launch error");
                Ok(())
            },
            || Ok(()),
            |_, d| Ok(d.as_secs_f64() * 1000.0),
        )
        .unwrap_err();
        assert_eq!(order, [0, 1]);
        assert!(format!("{err:#}").contains("probe candidate 1 launch failed"));
        assert!(format!("{err:#}").contains("fixture launch error"));
        assert!(
            probe(
                0,
                |_| Ok(()),
                |_| Ok(()),
                || Ok(()),
                |_, d| Ok(d.as_secs_f64() * 1000.0)
            )
            .is_err()
        );
    }
    #[test]
    fn replay_probe_derives_count_and_names_capacity() {
        use std::cell::Cell;
        let count = Cell::new(0usize);
        let batch = Cell::new(1usize);
        let sample = Cell::new(0usize);
        let result = probe_replays(
            2,
            1000,
            |_| Ok(()),
            |_, n| {
                count.set(count.get() + n);
                batch.set(n);
                Ok(())
            },
            || Ok(()),
            |_, _| {
                let i = sample.get();
                sample.set(i + 1);
                Ok(10.0 * batch.get() as f64 + if (i / 2).is_multiple_of(2) { 0.0 } else { 0.3 })
            },
        )
        .unwrap();
        assert!(result.repeats > 1 && result.pilot_spread > 0.01 && result.spread <= 0.01);
        assert!(count.get() >= 30 * result.repeats);
        let tick = Cell::new(false);
        let error = probe_replays(
            1,
            1,
            |_| Ok(()),
            |_, _| Ok(()),
            || Ok(()),
            |_, _| {
                tick.set(!tick.get());
                Ok(if tick.get() { 1.0 } else { 2.0 })
            },
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("state capacity 1"));
        assert!(
            probe(
                1,
                |_| Ok(()),
                |_| Ok(()),
                || Ok(()),
                |_, _| Err(anyhow::anyhow!("clock failed"))
            )
            .unwrap_err()
            .to_string()
            .contains("duration measurement failed")
        );
    }

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

    /// A platform with no `/proc/meminfo`, as Windows answers both figures.
    struct SyscallOnlyProbeSource {
        available: Option<u64>,
        total: Option<u64>,
    }

    impl ProbeSource for SyscallOnlyProbeSource {
        fn read_to_string(&self, _: &Path) -> io::Result<String> {
            Err(io::Error::new(io::ErrorKind::NotFound, "no procfs"))
        }
        fn host_available_memory_bytes(&self) -> Option<u64> {
            self.available
        }
        fn host_total_memory_bytes(&self) -> Option<u64> {
            self.total
        }
    }

    #[test]
    fn host_pool_total_binds_on_a_finite_cgroup_limit() {
        let snapshot = |host: Option<u64>, limit: Option<CgroupMemoryLimit>| ResourceSnapshot {
            schema_version: RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(1 << 30),
            cgroup_v2_memory_limit: limit,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: None,
            device_free_memory_bytes: None,
            host_device_memory_is_unified: None,
            host_memory_total_bytes: host,
            device_total_memory_bytes: None,
            measurement_scope: ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        };
        let host = 64u64 << 30;
        let container = 8u64 << 30;
        assert_eq!(
            snapshot(Some(host), Some(CgroupMemoryLimit::Bytes(container))).host_pool_total_bytes(),
            Some(container)
        );
        for limit in [Some(CgroupMemoryLimit::Unlimited), None] {
            assert_eq!(
                snapshot(Some(host), limit).host_pool_total_bytes(),
                Some(host)
            );
        }
        assert_eq!(
            snapshot(Some(host), Some(CgroupMemoryLimit::Bytes(host * 2))).host_pool_total_bytes(),
            Some(host)
        );
        assert_eq!(
            snapshot(None, Some(CgroupMemoryLimit::Bytes(container))).host_pool_total_bytes(),
            Some(container)
        );
        assert_eq!(snapshot(None, None).host_pool_total_bytes(), None);
    }

    #[test]
    fn snapshot_takes_both_host_figures_from_the_source_without_procfs() {
        let snapshot = resource_snapshot_with_source(
            None,
            &SyscallOnlyProbeSource {
                available: Some(4 << 30),
                total: Some(16 << 30),
            },
            1,
        )
        .unwrap();
        assert_eq!(snapshot.host_memory_available_bytes, Some(4 << 30));
        assert_eq!(snapshot.host_memory_total_bytes, Some(16 << 30));
        let blind = resource_snapshot_with_source(
            None,
            &SyscallOnlyProbeSource {
                available: Some(4 << 30),
                total: None,
            },
            1,
        )
        .unwrap();
        assert_eq!(blind.host_memory_total_bytes, None);
    }

    fn valid_current_cuda_fingerprint() -> HardwareFingerprint {
        let mut fingerprint = hardware_fingerprint_with_source(
            &Device::Cpu,
            &FakeProbeSource::default()
                .with("/proc/cpuinfo", "processor: 0\nmodel name: Test CPU\n")
                .with("/proc/meminfo", "MemTotal: 32768 kB\n")
                .with("/proc/sys/kernel/osrelease", "6.8.0-test\n"),
        )
        .unwrap();
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

    fn decode_record() -> DecodeChoice {
        let stock = ProbeRange {
            min: 10.0,
            median: 10.01,
            max: 10.02,
        };
        let xr16 = ProbeRange {
            min: 9.0,
            median: 9.01,
            max: 9.02,
        };
        let fingerprint = valid_current_cuda_fingerprint();
        DecodeChoice {
            adapter: "qwen35".into(),
            geometry: serde_json::json!({"hidden":1024}),
            class: "streamed".into(),
            settings: serde_json::json!({"max_context":4096}),
            split: vec![vec![(0, 0, 52, 64)]],
            fingerprint: Some(fingerprint),
            binary: Some(crate::identity::BinaryIdentity {
                schema_version: 1,
                package_name: "ff".into(),
                package_version: "0.1.0".into(),
                compiled_features: vec!["cuda".into()],
            }),
            image: image_digest(b"kernel"),
            trials: vec![ReplayProbe {
                batches: 1,
                ranges: vec![stock, xr16],
                repeats: 1,
                pilot_spread: 0.003,
                spread: 0.003,
            }],
            device: 0,
            cols: 1,
            derived: "xr16".into(),
            body: "xr16".into(),
            reason: "separated".into(),
            stock,
            xr16,
            repeats: 1,
            warmups: 6,
            samples: 24,
            capacity: 4096,
            state_bytes: 0,
            seed_token: 0,
            topology: "decode".into(),
            pilot_spread: 0.003,
            spread: 0.003,
            capture_ms: 1.0,
            replay_ms: 2.0,
            setup_ms: 3.0,
        }
    }

    #[test]
    fn group4_record_keys_measured_content_and_execution_class() {
        let record = decode_record();
        let mut other = record.clone();
        other.split = vec![vec![(0, 0, 51, 64)]];
        assert!(record.matches(&other));
        other.image = image_digest(b"changed kernel");
        assert!(!record.matches(&other));
        other = record.clone();
        other.class = "resident_graph".into();
        assert!(!record.matches(&other));
        other = record.clone();
        other.settings = serde_json::json!({"max_context":8192});
        assert!(!record.matches(&other));
        other = record.clone();
        other.fingerprint.as_mut().unwrap().driver_version = Some("new driver".into());
        assert!(!record.matches(&other));
        other = record.clone();
        other.binary.as_mut().unwrap().package_version = "0.2.0".into();
        assert!(!record.matches(&other));
        assert_eq!(
            image_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn group4_requires_the_loaded_driver_image_identity() {
        let record = DecodeChoice::combine(vec![decode_record(); 3]).unwrap();
        assert!(
            record
                .fingerprint
                .as_ref()
                .unwrap()
                .runtime_version
                .is_none()
        );
        assert!(
            !record
                .fingerprint
                .as_ref()
                .unwrap()
                .supports_calibration_cache_reuse()
        );
        let mut missing = record.clone();
        let fingerprint = missing.fingerprint.as_mut().unwrap();
        fingerprint.cuda_driver_api_version = None;
        fingerprint.cuda_driver_api_version_source = None;
        fingerprint.cuda_driver_api_version_unavailable_reason =
            Some(FingerprintUnavailableReason::CudaDriverApiQueryFailed);
        assert!(
            missing
                .validate()
                .unwrap_err()
                .to_string()
                .contains("cuda_driver_api_version")
        );
    }

    #[test]
    fn group4_independent_disagreement_persists_stock() {
        let record = decode_record();
        let unanimous = DecodeChoice::combine(vec![record.clone(); 3]).unwrap();
        assert_eq!(unanimous.body, "xr16");
        assert_eq!(unanimous.setup_ms, 9.0);
        let mut stock = record.clone();
        stock.trials[0].ranges.swap(0, 1);
        let stock = DecodeChoice::combine(vec![stock; 3]).unwrap();
        assert_eq!(stock.body, "stock");
        assert_eq!(stock.reason, "independent separated trials agree");
        let mut overlap = record.clone();
        overlap.trials[0].ranges[1] = overlap.stock;
        let combined =
            DecodeChoice::combine(vec![record.clone(), overlap, record.clone()]).unwrap();
        assert_eq!(combined.body, "stock");
        combined.validate().unwrap();
        let mut corrupt = combined.clone();
        corrupt.body = "xr16".into();
        corrupt.derived = "xr16".into();
        assert!(corrupt.validate().is_err());
        assert!(DecodeChoice::combine(vec![record.clone(); 2]).is_err());
        let mut wrong = record.clone();
        wrong.cols = 2;
        assert!(DecodeChoice::combine(vec![record.clone(), record, wrong]).is_err());
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
            .with("/sys/fs/cgroup/workload/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n")
            .with("/sys/fs/cgroup/memory.max", "max\n");

        let snapshot = resource_snapshot_with_source(None, &source, 1234).unwrap();
        assert_eq!(snapshot.measured_at_unix_ms, 1234);
        assert_eq!(snapshot.host_memory_available_bytes, Some(8 << 20));
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Bytes(1 << 30))
        );
        assert_eq!(snapshot.cgroup_v2_memory_current_bytes, Some(1 << 28));
        // The whole file-cache view is reclaimable, so the available bytes
        // recover the full limit.
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(1 << 30));
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
            .with("/sys/fs/cgroup unified/tenant/a/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n")
            .with("/sys/fs/cgroup unified/tenant/memory.max", "max\n")
            .with("/sys/fs/cgroup unified/memory.max", "max\n");

        let snapshot = resource_snapshot_with_source(None, &source, 7).unwrap();
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
        let running = resource_snapshot_with_source(None, &source, 1).unwrap();
        assert_eq!(running.cgroup_v2_memory_available_bytes, Some(4 * gib));

        let source = source.with("/sys/fs/cgroup/memory.stat", &stat(0));
        let exited = resource_snapshot_with_source(None, &source, 2).unwrap();
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
    fn cgroup_capacity_bounds_cache_and_excludes_reclaimable_file_cache() {
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
        let snapshot = resource_snapshot_with_source(None, &source, 1).unwrap();
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(100));

        let source = source.with("/sys/fs/cgroup/memory.current", "200");
        let snapshot = resource_snapshot_with_source(None, &source, 2).unwrap();
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(1000));

        let source = source.with(
            "/sys/fs/cgroup/memory.stat",
            "file 300\nshmem 300\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n",
        );
        let snapshot = resource_snapshot_with_source(None, &source, 3).unwrap();
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
            .with("/sys/fs/cgroup/tenant/child/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n")
            .with("/sys/fs/cgroup/tenant/memory.max", "1000\n")
            .with("/sys/fs/cgroup/tenant/memory.current", "900\n")
            .with("/sys/fs/cgroup/tenant/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n")
            .with("/sys/fs/cgroup/memory.max", "2000\n")
            .with("/sys/fs/cgroup/memory.current", "1000\n")
            .with("/sys/fs/cgroup/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n");

        let snapshot = resource_snapshot_with_source(None, &source, 11).unwrap();
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Bytes(1000))
        );
        assert_eq!(snapshot.cgroup_v2_memory_current_bytes, Some(100));
        // The file cache is reclaimable, so /tenant's usage nets to zero and
        // its full 1000-byte limit is available.
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(1000));

        let stat =
            "file 800\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n";
        let source = source
            .with("/sys/fs/cgroup/tenant/memory.stat", stat)
            .with("/sys/fs/cgroup/memory.current", "1800\n")
            .with("/sys/fs/cgroup/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n");
        let snapshot = resource_snapshot_with_source(None, &source, 12).unwrap();
        // /tenant nets to 1000 - (900 - 800) = 900; the root nets to zero.
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(900));

        let source = source.with("/sys/fs/cgroup/memory.stat", stat);
        let snapshot = resource_snapshot_with_source(None, &source, 13).unwrap();
        // The root's file cache backfills its headroom again.
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(900));
    }

    #[test]
    fn incomplete_cgroup_hierarchy_errors_instead_of_partial_capacity() {
        let source = FakeProbeSource::default()
            .with("/proc/self/cgroup", "0::/tenant/child\n")
            .with(
                "/proc/self/mountinfo",
                "31 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            )
            .with("/sys/fs/cgroup/tenant/child/memory.max", "1000\n")
            .with("/sys/fs/cgroup/tenant/child/memory.current", "100\n")
            .with(
                "/sys/fs/cgroup/tenant/child/memory.stat",
                "file 50\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n",
            );

        let error = resource_snapshot_with_source(None, &source, 12)
            .expect_err("an incomplete cgroup hierarchy must error, not degrade");
        assert!(
            error.to_string().contains("memory.max"),
            "unexpected error: {error}"
        );
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
            .with("/sys/fs/cgroup/tenant/child/memory.stat", "file 268435456\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n")
            .with("/sys/fs/cgroup/tenant/memory.max", "1000\n")
            .with("/sys/fs/cgroup/tenant/memory.current", "900\n")
            .with(
                "/sys/fs/cgroup/tenant/memory.stat",
                "file 100\nshmem 0\nfile_mapped 0\nfile_dirty 0\nfile_writeback 0\nunevictable 0\n",
            );

        let snapshot = resource_snapshot_with_source(None, &source, 13).unwrap();
        assert_eq!(
            snapshot.cgroup_v2_memory_limit,
            Some(CgroupMemoryLimit::Bytes(1000))
        );
        assert_eq!(snapshot.cgroup_v2_memory_available_bytes, Some(200));
    }

    /// A stored record without the unified-topology field predates the
    /// probe and must be recaptured, not silently treated as split.
    #[test]
    fn record_without_unified_field_fails_the_load() {
        let old = serde_json::json!({
            "schema_version": RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            "measured_at_unix_ms": 7,
            "host_memory_available_bytes": null,
            "cgroup_v2_memory_limit": null,
            "cgroup_v2_memory_current_bytes": null,
            "cgroup_v2_memory_available_bytes": null,
            "device_free_memory_bytes": null,
            "host_memory_total_bytes": null,
            "device_total_memory_bytes": null,
            "measurement_scope": {
                "host_memory": null,
                "cgroup_memory": null,
                "device_memory": null
            }
        });
        let error = serde_json::from_value::<ResourceSnapshot>(old)
            .expect_err("a record missing host_device_memory_is_unified must fail");
        assert!(
            error.to_string().contains("host_device_memory_is_unified"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_proc_and_cgroup_files_are_reported_as_unavailable() {
        let snapshot = resource_snapshot_with_source(None, &FakeProbeSource::default(), 9).unwrap();
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
        // Only confirmed unified records expose a combined pool.
        assert_eq!(snapshot(None).unified_pool_available_bytes(), None);
        assert_eq!(snapshot(Some(false)).unified_pool_available_bytes(), None);
        // A probed unified topology binds on the smallest view of the pool.
        assert_eq!(snapshot(Some(true)).unified_pool_available_bytes(), Some(6));
        assert!(!snapshot(Some(true)).unified_pool_is_unmeasurable());
        for missing in [
            ResourceSnapshot {
                device_free_memory_bytes: None,
                ..snapshot(Some(true))
            },
            ResourceSnapshot {
                host_memory_available_bytes: None,
                ..snapshot(Some(true))
            },
        ] {
            assert_eq!(missing.unified_pool_available_bytes(), None);
            assert!(missing.unified_pool_is_unmeasurable());
        }
        let no_cgroup = ResourceSnapshot {
            cgroup_v2_memory_available_bytes: None,
            ..snapshot(Some(true))
        };
        assert_eq!(no_cgroup.unified_pool_available_bytes(), Some(6));
        assert!(!no_cgroup.unified_pool_is_unmeasurable());
        assert!(!snapshot(Some(false)).unified_pool_is_unmeasurable());
        assert!(!snapshot(None).unified_pool_is_unmeasurable());
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
        let fingerprint = hardware_fingerprint_with_source(&Device::Cpu, &source).unwrap();

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
        )
        .unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
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
        let fingerprint = HardwareFingerprint::collect(&Device::Cpu).unwrap();
        let snapshot = ResourceSnapshot::capture(Some(&Device::Cpu)).unwrap();
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
        let fingerprint = HardwareFingerprint::collect(&Device::Cpu).unwrap();
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

        let snapshot = ResourceSnapshot::capture(None).unwrap();
        let encoded = serde_json::to_vec(&snapshot).unwrap();
        let decoded: ResourceSnapshot = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn admission_reserve_scales_below_the_crossover_and_caps_above_it() {
        const GIB: u64 = 1 << 30;
        const FLOOR: u64 = 512 << 20;
        assert_eq!(admission_reserve_bytes(None), GIB);
        // Small pools scale down to the floor, never past it: the reserve
        // still has to cover the fixed CUDA-context component.
        assert_eq!(admission_reserve_bytes(Some(6 * GIB)), FLOOR);
        assert_eq!(admission_reserve_bytes(Some(8 * GIB)), FLOOR);
        assert_eq!(admission_reserve_bytes(Some(0)), FLOOR);
        // Above ~10 GiB the percentage clears the floor and dominates.
        assert_eq!(admission_reserve_bytes(Some(16 * GIB)), 16 * GIB / 20);
        assert_eq!(
            admission_reserve_bytes(Some(20 * GIB)),
            GIB,
            "exactly at the crossover the cap binds"
        );
        assert_eq!(
            admission_reserve_bytes(Some(24 * GIB)),
            GIB,
            "the 24 GiB baseline keeps the flat 1 GiB margin"
        );
        assert_eq!(admission_reserve_bytes(Some(u64::MAX)), GIB);
    }
}
