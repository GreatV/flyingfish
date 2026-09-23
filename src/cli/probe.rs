use super::device_parse::parse_device_single;
use anyhow::Result;
use flyingfish::runtime::probe::{DeviceBackend, HardwareFingerprint, ResourceSnapshot};
use serde::Serialize;

#[derive(Serialize)]
struct ProbeReport {
    fingerprint: HardwareFingerprint,
    calibration_cache_reuse_supported: bool,
    snapshot: ResourceSnapshot,
}

pub(super) fn run_probe(device: String, json: bool) -> Result<()> {
    let device = parse_device_single(&device)?;
    let fingerprint = HardwareFingerprint::collect(&device);
    let report = ProbeReport {
        calibration_cache_reuse_supported: fingerprint.supports_calibration_cache_reuse(),
        fingerprint,
        snapshot: ResourceSnapshot::capture(Some(&device)),
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("Best-effort hardware fingerprint (calibration identity component):");
        println!("{}", serde_json::to_string_pretty(&report.fingerprint)?);
        if report.fingerprint.backend == DeviceBackend::Cuda {
            println!(
                "CUDA values identify their query source; unavailable runtime/API values stay null with an explicit reason."
            );
        }
        if report.calibration_cache_reuse_supported {
            println!("Calibration cache reuse: supported by this fingerprint.");
        } else {
            println!(
                "Calibration cache reuse: disabled; retain this fingerprint as provenance only."
            );
        }
        println!();
        println!("Dynamic resource snapshot (point-in-time; excluded from fingerprint):");
        println!("{}", serde_json::to_string_pretty(&report.snapshot)?);
        println!("Measurement scopes are reported explicitly in snapshot.measurement_scope.");
    }
    Ok(())
}
