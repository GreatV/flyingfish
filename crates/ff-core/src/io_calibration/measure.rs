//! The sequential payload measurement itself.
//!
//! Separated from the schema it fills so a report can be parsed, compared and
//! replayed on a host that never runs a measurement.

use super::*;

pub fn measure_io_bandwidth(
    payload: &Path,
    device: &Device,
    payload_format: &str,
) -> Result<IoCalibrationReport> {
    validate_payload_format(payload_format)?;
    let fingerprint = HardwareFingerprint::collect(device);
    fingerprint.validate()?;
    let key = IoCalibrationKey::from_fingerprint(&fingerprint, payload_format)?;

    let metadata = payload.metadata().with_context(|| {
        format!(
            "failed to stat I/O calibration payload {}",
            payload.display()
        )
    })?;
    anyhow::ensure!(
        metadata.is_file(),
        "I/O calibration payload is not a regular file: {}",
        payload.display()
    );
    let payload_bytes = metadata.len();
    anyhow::ensure!(
        payload_bytes >= MIN_IO_CALIBRATION_PAYLOAD_BYTES,
        "I/O calibration payload is {payload_bytes} bytes; schema 1 requires at least {MIN_IO_CALIBRATION_PAYLOAD_BYTES}"
    );
    anyhow::ensure!(
        payload_bytes <= MAX_IO_CALIBRATION_PAYLOAD_BYTES,
        "I/O calibration payload is {payload_bytes} bytes; schema 1 accepts at most {MAX_IO_CALIBRATION_PAYLOAD_BYTES}"
    );
    let len = usize::try_from(payload_bytes).context("I/O calibration payload exceeds usize")?;
    let mut buf = vec![0u8; len];
    let mut file = File::open(payload).with_context(|| {
        format!(
            "failed to open I/O calibration payload {}",
            payload.display()
        )
    })?;
    let start = Instant::now();
    file.read_exact(&mut buf).with_context(|| {
        format!(
            "failed to read I/O calibration payload {}",
            payload.display()
        )
    })?;
    let mut elapsed = start.elapsed();
    std::hint::black_box(&buf[..]);
    let mut transferred = payload_bytes;
    while elapsed.as_nanos() == 0 && transferred < MAX_IO_CALIBRATION_PAYLOAD_BYTES {
        let mut file = File::open(payload).with_context(|| {
            format!(
                "failed to reopen I/O calibration payload {}",
                payload.display()
            )
        })?;
        let start = Instant::now();
        file.read_exact(&mut buf).with_context(|| {
            format!(
                "failed to reread I/O calibration payload {}",
                payload.display()
            )
        })?;
        elapsed += start.elapsed();
        transferred = transferred
            .checked_add(payload_bytes)
            .context("I/O calibration transferred-byte count overflow")?;
        std::hint::black_box(&buf[..]);
    }

    let host_sequential_read = BandwidthSample::from_bytes_and_elapsed(transferred, elapsed)?;
    let host_to_device = measure_host_to_device(&buf, device)?;
    let host_to_device_over_host_read_ratio = host_to_device
        .sample
        .as_ref()
        .map(|sample| sample.bytes_per_second / host_sequential_read.bytes_per_second);
    let report = IoCalibrationReport {
        schema_version: IO_CALIBRATION_SCHEMA_VERSION,
        key,
        fingerprint,
        payload_bytes,
        host_sequential_read,
        host_to_device,
        host_to_device_over_host_read_ratio,
    };
    report.validate()?;
    Ok(report)
}

pub(super) fn measure_host_to_device(
    bytes: &[u8],
    device: &Device,
) -> Result<OptionalBandwidthSample> {
    if device.is_cpu() {
        return Ok(OptionalBandwidthSample::unavailable(
            IoCopyUnavailableReason::CpuDeviceHasNoHostToDevicePath,
        ));
    }
    if device.is_metal() {
        return Ok(OptionalBandwidthSample::unavailable(
            IoCopyUnavailableReason::MetalHostToDeviceNotMeasuredInSchema1,
        ));
    }
    anyhow::ensure!(
        device.is_cuda(),
        "I/O calibration host-to-device copy requires a CUDA device"
    );
    let len = bytes.len();
    let chunk = u64::try_from(len).context("payload exceeds u64")?;
    let host = Tensor::from_raw_buffer(bytes, DType::U8, &[len], &Device::Cpu)
        .context("failed to wrap I/O calibration payload as a host tensor")?;
    device
        .synchronize()
        .context("failed to synchronize device before host-to-device copy")?;
    let mut elapsed = Duration::ZERO;
    let mut transferred = 0u64;
    while elapsed.as_nanos() == 0 && transferred < MAX_IO_CALIBRATION_PAYLOAD_BYTES {
        let start = Instant::now();
        let on_device = host
            .to_device(device)
            .context("failed to copy I/O calibration payload to the device")?;
        device
            .synchronize()
            .context("failed to synchronize device after host-to-device copy")?;
        elapsed += start.elapsed();
        transferred = transferred
            .checked_add(chunk)
            .context("I/O calibration host-to-device byte count overflow")?;
        drop(on_device);
    }
    BandwidthSample::from_bytes_and_elapsed(transferred, elapsed)
        .map(OptionalBandwidthSample::measured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::DeviceBackend;
    use candle_core::Device;
    use std::fs;

    fn write_payload(bytes: usize) -> (tempfile::TempDir, std::path::PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("payload.bin");
        fs::write(&path, vec![0x5a; bytes]).unwrap();
        (temporary, path)
    }

    #[test]
    fn payload_driven_measurement_emits_a_versioned_keyed_cpu_report() {
        let (_temporary, path) =
            write_payload(usize::try_from(MIN_IO_CALIBRATION_PAYLOAD_BYTES).unwrap());
        let report =
            measure_io_bandwidth(&path, &Device::Cpu, IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL).unwrap();
        report.validate().unwrap();
        assert_eq!(report.schema_version, IO_CALIBRATION_SCHEMA_VERSION);
        assert_eq!(report.key.backend, DeviceBackend::Cpu);
        assert_eq!(report.key.payload_format, IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL);
        assert!(report.key.cuda_device_uuid.is_none());
        assert_eq!(report.payload_bytes, MIN_IO_CALIBRATION_PAYLOAD_BYTES);
        assert!(report.host_sequential_read.bytes >= MIN_IO_CALIBRATION_PAYLOAD_BYTES);
        assert!(report.host_sequential_read.elapsed_ns > 0);
        assert!(report.host_sequential_read.bytes_per_second > 0.0);
        assert!(report.host_to_device.sample.is_none());
        assert_eq!(
            report.host_to_device.unavailable_reason,
            Some(IoCopyUnavailableReason::CpuDeviceHasNoHostToDevicePath)
        );
        assert!(report.host_to_device_over_host_read_ratio.is_none());
        let json = serde_json::to_value(&report).unwrap();
        assert!(json.get("threshold").is_none());
        assert!(json.get("hybrid").is_none());
        assert!(json.get("moe_backend").is_none());
        assert_ne!(
            json.get("host_to_device_over_host_read_ratio"),
            Some(&serde_json::json!(2.0))
        );
    }

    #[test]
    fn matching_live_key_is_accepted_and_gpu_or_format_mismatch_is_rejected() {
        let (_temporary, path) =
            write_payload(usize::try_from(MIN_IO_CALIBRATION_PAYLOAD_BYTES).unwrap());
        let report =
            measure_io_bandwidth(&path, &Device::Cpu, IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL).unwrap();
        report.require_matching_key(&report.key).unwrap();

        let mut other_format = report.key.clone();
        other_format.payload_format = "nvfp4".to_owned();
        let format_error = report
            .require_matching_key(&other_format)
            .unwrap_err()
            .to_string();
        assert!(format_error.contains("cannot be applied"));
        assert!(format_error.contains("nvfp4"));

        let other_gpu = IoCalibrationKey {
            backend: DeviceBackend::Cuda,
            operating_system: report.key.operating_system.clone(),
            architecture: report.key.architecture.clone(),
            device_name: Some("Other GPU".to_owned()),
            cuda_device_uuid: Some("00010203-0405-0607-0809-0a0b0c0d0e0f".to_owned()),
            payload_format: IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL.to_owned(),
        };
        other_gpu.validate().unwrap();
        let gpu_error = report
            .require_matching_key(&other_gpu)
            .unwrap_err()
            .to_string();
        assert!(gpu_error.contains("cannot be applied"));
        assert!(gpu_error.contains("raw_sequential"));
    }

    #[test]
    fn from_json_round_trips_a_measured_report() {
        let (_temporary, path) =
            write_payload(usize::try_from(MIN_IO_CALIBRATION_PAYLOAD_BYTES).unwrap());
        let report =
            measure_io_bandwidth(&path, &Device::Cpu, IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL).unwrap();
        let bytes = serde_json::to_vec_pretty(&report).unwrap();
        let loaded = super::IoCalibrationReport::from_json(&bytes).unwrap();
        assert_eq!(loaded.schema_version, report.schema_version);
        assert_eq!(loaded.key, report.key);
        assert_eq!(loaded.payload_bytes, report.payload_bytes);
        assert_eq!(
            loaded.host_to_device.unavailable_reason,
            report.host_to_device.unavailable_reason
        );
    }

    #[test]
    fn rejects_an_undersized_payload_before_timing() {
        let (_temporary, path) = write_payload(16);
        let error = measure_io_bandwidth(&path, &Device::Cpu, IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires at least"));
    }
}
