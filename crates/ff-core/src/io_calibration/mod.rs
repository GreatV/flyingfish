use crate::{
    probe::{DeviceBackend, HardwareFingerprint},
    required_option,
};

use anyhow::{Context, Result, bail};

use candle_core::{DType, Device, Tensor};

use serde::{Deserialize, Serialize};

use std::{
    fs::File,
    io::Read,
    path::Path,
    time::{Duration, Instant},
};

mod measure;

pub use measure::*;

pub const IO_CALIBRATION_SCHEMA_VERSION: u32 = 1;

pub const IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL: &str = "raw_sequential";

pub const MIN_IO_CALIBRATION_PAYLOAD_BYTES: u64 = 4_096;

pub const MAX_IO_CALIBRATION_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IoCalibrationKey {
    pub backend: DeviceBackend,
    pub operating_system: String,
    pub architecture: String,
    #[serde(deserialize_with = "required_option")]
    pub device_name: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub cuda_device_uuid: Option<String>,
    pub payload_format: String,
}

impl IoCalibrationKey {
    pub fn from_fingerprint(
        fingerprint: &HardwareFingerprint,
        payload_format: &str,
    ) -> Result<Self> {
        let key = Self {
            backend: fingerprint.backend,
            operating_system: fingerprint.operating_system.clone(),
            architecture: fingerprint.architecture.clone(),
            device_name: fingerprint.device_name.clone(),
            cuda_device_uuid: fingerprint.cuda_device_uuid.clone(),
            payload_format: payload_format.to_owned(),
        };
        key.validate()?;
        Ok(key)
    }

    pub fn validate(&self) -> Result<()> {
        validate_payload_format(&self.payload_format)?;
        anyhow::ensure!(
            !self.operating_system.trim().is_empty(),
            "I/O calibration key operating_system must not be empty"
        );
        anyhow::ensure!(
            !self.architecture.trim().is_empty(),
            "I/O calibration key architecture must not be empty"
        );
        if let Some(name) = self.device_name.as_deref() {
            anyhow::ensure!(
                !name.trim().is_empty(),
                "I/O calibration key device_name must not be empty when present"
            );
        }
        if let Some(uuid) = self.cuda_device_uuid.as_deref() {
            anyhow::ensure!(
                !uuid.trim().is_empty(),
                "I/O calibration key cuda_device_uuid must not be empty when present"
            );
        }
        match self.backend {
            DeviceBackend::Cpu | DeviceBackend::Metal => anyhow::ensure!(
                self.cuda_device_uuid.is_none(),
                "I/O calibration key for {:?} must not carry a CUDA UUID",
                self.backend
            ),
            DeviceBackend::Cuda => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoCopyUnavailableReason {
    CpuDeviceHasNoHostToDevicePath,
    MetalHostToDeviceNotMeasuredInSchema1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BandwidthSample {
    pub bytes: u64,
    pub elapsed_ns: u64,
    pub bytes_per_second: f64,
}

impl BandwidthSample {
    pub fn from_bytes_and_elapsed(bytes: u64, elapsed: Duration) -> Result<Self> {
        anyhow::ensure!(bytes > 0, "bandwidth sample transferred no bytes");
        let elapsed_ns =
            u64::try_from(elapsed.as_nanos()).context("elapsed time exceeds u64 ns")?;
        anyhow::ensure!(
            elapsed_ns > 0,
            "bandwidth sample elapsed below timer resolution; use a larger payload"
        );
        Ok(Self {
            bytes,
            elapsed_ns,
            bytes_per_second: (bytes as f64) * 1_000_000_000.0 / (elapsed_ns as f64),
        })
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.bytes > 0, "bandwidth sample transferred no bytes");
        anyhow::ensure!(
            self.elapsed_ns > 0,
            "bandwidth sample elapsed_ns must be positive"
        );
        let expected = (self.bytes as f64) * 1_000_000_000.0 / (self.elapsed_ns as f64);
        anyhow::ensure!(
            (self.bytes_per_second - expected).abs() <= expected.abs() * 1e-9 + 1e-6,
            "bandwidth sample bytes_per_second {0} disagrees with bytes/elapsed",
            self.bytes_per_second
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalBandwidthSample {
    #[serde(deserialize_with = "required_option")]
    pub sample: Option<BandwidthSample>,
    #[serde(deserialize_with = "required_option")]
    pub unavailable_reason: Option<IoCopyUnavailableReason>,
}

impl OptionalBandwidthSample {
    pub fn measured(sample: BandwidthSample) -> Self {
        Self {
            sample: Some(sample),
            unavailable_reason: None,
        }
    }

    pub fn unavailable(reason: IoCopyUnavailableReason) -> Self {
        Self {
            sample: None,
            unavailable_reason: Some(reason),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match (&self.sample, self.unavailable_reason) {
            (Some(sample), None) => sample.validate(),
            (None, Some(_)) => Ok(()),
            (Some(_), Some(_)) => {
                bail!("host-to-device sample must not include an unavailable reason")
            }
            (None, None) => {
                bail!("host-to-device sample requires a measurement or an unavailable reason")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IoCalibrationReport {
    pub schema_version: u32,
    pub key: IoCalibrationKey,
    pub fingerprint: HardwareFingerprint,
    pub payload_bytes: u64,
    pub host_sequential_read: BandwidthSample,
    pub host_to_device: OptionalBandwidthSample,
    #[serde(deserialize_with = "required_option")]
    pub host_to_device_over_host_read_ratio: Option<f64>,
}

impl IoCalibrationReport {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let report: Self =
            serde_json::from_slice(bytes).context("invalid I/O calibration-report JSON")?;
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == IO_CALIBRATION_SCHEMA_VERSION,
            "unsupported I/O calibration-report schema {}; this build supports schema {}",
            self.schema_version,
            IO_CALIBRATION_SCHEMA_VERSION
        );
        self.key.validate()?;
        self.fingerprint.validate()?;
        anyhow::ensure!(
            self.payload_bytes >= MIN_IO_CALIBRATION_PAYLOAD_BYTES,
            "I/O calibration payload is {} bytes; schema 1 requires at least {MIN_IO_CALIBRATION_PAYLOAD_BYTES}",
            self.payload_bytes
        );
        anyhow::ensure!(
            self.payload_bytes <= MAX_IO_CALIBRATION_PAYLOAD_BYTES,
            "I/O calibration payload is {} bytes; schema 1 accepts at most {MAX_IO_CALIBRATION_PAYLOAD_BYTES}",
            self.payload_bytes
        );
        anyhow::ensure!(
            self.key.backend == self.fingerprint.backend
                && self.key.operating_system == self.fingerprint.operating_system
                && self.key.architecture == self.fingerprint.architecture
                && self.key.device_name == self.fingerprint.device_name
                && self.key.cuda_device_uuid == self.fingerprint.cuda_device_uuid,
            "I/O calibration key does not match the embedded hardware fingerprint"
        );
        self.host_sequential_read.validate()?;
        self.host_to_device.validate()?;
        match (
            &self.host_to_device.sample,
            self.host_to_device_over_host_read_ratio,
        ) {
            (Some(h2d), Some(ratio)) => {
                let expected = h2d.bytes_per_second / self.host_sequential_read.bytes_per_second;
                anyhow::ensure!(
                    (ratio - expected).abs() <= expected.abs() * 1e-9 + 1e-9,
                    "host-to-device over host-read ratio {ratio} disagrees with measured samples"
                );
            }
            (None, None) => {}
            (Some(_), None) => {
                bail!(
                    "measured host-to-device sample requires a host-to-device over host-read ratio"
                )
            }
            (None, Some(_)) => {
                bail!(
                    "host-to-device over host-read ratio requires a measured host-to-device sample"
                )
            }
        }
        Ok(())
    }

    pub fn require_matching_key(&self, live: &IoCalibrationKey) -> Result<()> {
        self.validate()?;
        anyhow::ensure!(
            self.key == *live,
            "I/O calibration profile is keyed for backend {:?}, UUID {:?}, format {:?}; it cannot be applied to backend {:?}, UUID {:?}, format {:?}",
            self.key.backend,
            self.key.cuda_device_uuid,
            self.key.payload_format,
            live.backend,
            live.cuda_device_uuid,
            live.payload_format
        );
        Ok(())
    }
}

pub fn validate_payload_format(format: &str) -> Result<()> {
    anyhow::ensure!(
        format == IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL,
        "unsupported I/O payload format {format:?}; schema 1 accepts {IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_1_accepts_only_raw_sequential_payload_format() {
        validate_payload_format(IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL).unwrap();
        let error = validate_payload_format("nvfp4").unwrap_err().to_string();
        assert!(error.contains("unsupported I/O payload format"));
        assert!(error.contains("raw_sequential"));
        assert!(!error.contains("2.0"));
    }
}
