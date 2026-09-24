use super::IoBenchmarkProfile;
use super::device_parse::parse_device_single;
use super::output_hygiene::{
    ensure_new_output, publish_staged_bytes, resolve_output_outside_model,
};
use anyhow::{Context, Result, bail};
use flyingfish::interconnect_benchmark::{
    DEFAULT_LOCAL_IO_SAMPLE_ITERATIONS, DEFAULT_LOCAL_IO_WARMUP_ITERATIONS,
    LocalIoBenchmarkOptions, LocalIoBenchmarkReport, PeerD2dMeasurement,
    measure_local_io_benchmark,
};
use flyingfish::runtime::artifact::ArtifactStaging;
use flyingfish::runtime::io_calibration::{
    IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL, IoCalibrationReport, measure_io_bandwidth,
    validate_payload_format,
};
use std::path::PathBuf;

/// The shared operator inputs of every IO-calibration profile.
pub(super) struct CalibrateIoConfig {
    pub(super) payload: Option<PathBuf>,
    pub(super) model: Option<PathBuf>,
    pub(super) output: PathBuf,
    pub(super) device: String,
    pub(super) format: String,
    pub(super) peer_device: Option<String>,
    pub(super) warmups: usize,
    pub(super) samples: usize,
    pub(super) expert_layer: usize,
    pub(super) expert_index: usize,
}

pub(super) fn run_calibrate_io(
    profile: IoBenchmarkProfile,
    config: CalibrateIoConfig,
) -> Result<()> {
    match profile {
        IoBenchmarkProfile::Sequential => run_sequential(config),
        IoBenchmarkProfile::LocalInterconnect => run_local_interconnect(config),
    }
}

fn run_sequential(config: CalibrateIoConfig) -> Result<()> {
    let CalibrateIoConfig {
        payload,
        model,
        output,
        device,
        format,
        peer_device,
        warmups,
        samples,
        expert_layer,
        expert_index,
    } = config;
    anyhow::ensure!(
        model.is_none()
            && peer_device.is_none()
            && warmups == DEFAULT_LOCAL_IO_WARMUP_ITERATIONS
            && samples == DEFAULT_LOCAL_IO_SAMPLE_ITERATIONS
            && expert_layer == 3
            && expert_index == 0,
        "local-interconnect-only options cannot be used with --profile sequential"
    );
    validate_payload_format(&format)?;
    let device = parse_device_single(&device)?;
    let payload = payload.context("--payload is required for --profile sequential")?;
    let payload = std::fs::canonicalize(&payload).with_context(|| {
        format!(
            "failed to resolve I/O calibration payload {}",
            payload.display()
        )
    })?;
    anyhow::ensure!(
        payload.is_file(),
        "I/O calibration payload is not a file: {}",
        payload.display()
    );
    ensure_new_output(&output, "I/O calibration output")?;
    let staging = ArtifactStaging::new(&output).with_context(|| {
        format!(
            "failed to stage I/O calibration report {}",
            output.display()
        )
    })?;

    let report = measure_io_bandwidth(&payload, &device, &format)?;
    let json =
        serde_json::to_vec_pretty(&report).context("failed to serialize I/O calibration report")?;
    let _ = IoCalibrationReport::from_json(&json)?;
    let published = publish_staged_bytes(staging, &json)?;
    println!(
        "wrote I/O calibration schema {} report to {} (format {IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL})",
        report.schema_version,
        published.destination.display()
    );
    match report.host_to_device.sample.as_ref() {
        Some(sample) => println!(
            "host sequential read {:.3} GiB/s; host-to-device {:.3} GiB/s; measured ratio {:.3}",
            gib_per_second(report.host_sequential_read.bytes_per_second),
            gib_per_second(sample.bytes_per_second),
            report
                .host_to_device_over_host_read_ratio
                .expect("validated reports include a ratio when host-to-device is measured")
        ),
        None => println!(
            "host sequential read {:.3} GiB/s; host-to-device unavailable ({})",
            gib_per_second(report.host_sequential_read.bytes_per_second),
            report
                .host_to_device
                .unavailable_reason
                .map(|reason| format!("{reason:?}"))
                .unwrap_or_else(|| "unspecified".to_owned())
        ),
    }
    Ok(())
}

fn run_local_interconnect(config: CalibrateIoConfig) -> Result<()> {
    let CalibrateIoConfig {
        payload,
        model,
        output,
        device,
        format,
        peer_device,
        warmups,
        samples,
        expert_layer,
        expert_index,
    } = config;
    anyhow::ensure!(
        payload.is_none(),
        "--payload cannot be used with --profile local-interconnect"
    );
    anyhow::ensure!(
        format == IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL,
        "--format applies only to --profile sequential"
    );
    let model = model.context("--model is required for --profile local-interconnect")?;
    let primary_cuda_ordinal = explicit_cuda_ordinal(&device, "--device")?;
    let peer_cuda_ordinal = peer_device
        .as_deref()
        .map(|value| explicit_cuda_ordinal(value, "--peer-device"))
        .transpose()?;
    anyhow::ensure!(
        peer_cuda_ordinal != Some(primary_cuda_ordinal),
        "--peer-device must differ from --device"
    );
    let device = parse_device_single(&device)?;
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "local-interconnect I/O output")?;
    let staging = ArtifactStaging::new(&output).with_context(|| {
        format!(
            "failed to stage local-interconnect report {}",
            output.display()
        )
    })?;
    let report = measure_local_io_benchmark(
        &model,
        &device,
        LocalIoBenchmarkOptions {
            primary_cuda_ordinal,
            peer_cuda_ordinal,
            warmup_iterations: warmups,
            sample_iterations: samples,
            expert_layer,
            expert_index,
        },
    )?;
    let json = serde_json::to_vec_pretty(&report)
        .context("failed to serialize local-interconnect I/O report")?;
    let _ = LocalIoBenchmarkReport::from_json(&json)?;
    let published = publish_staged_bytes(staging, &json)?;
    println!(
        "wrote local-interconnect schema {} report to {}",
        report.schema_version,
        published.destination.display()
    );
    println!(
        "H3 cut: same-device D2D {:.3} GiB/s; pinned host-staged roundtrip {:.3} GiB/s; TCP loopback {:.3} GiB/s",
        gib_per_second(
            report
                .h3_standard_block_cut
                .same_device_d2d
                .statistics
                .median_bytes_per_second,
        ),
        gib_per_second(
            report
                .h3_standard_block_cut
                .pinned_host_staged_roundtrip
                .statistics
                .median_bytes_per_second,
        ),
        gib_per_second(
            report
                .h3_standard_block_cut
                .tcp_loopback_roundtrip
                .statistics
                .median_bytes_per_second,
        ),
    );
    match &report.h3_standard_block_cut.peer_device_d2d {
        PeerD2dMeasurement::Measured { series, .. } => println!(
            "peer-device D2D {:.3} GiB/s",
            gib_per_second(series.statistics.median_bytes_per_second)
        ),
        PeerD2dMeasurement::Unavailable {
            reason,
            visible_cuda_devices,
        } => println!(
            "peer-device D2D unavailable ({reason:?}; visible CUDA devices {visible_cuda_devices})"
        ),
    }
    println!(
        "GLM expert: B_P {:.3} GiB/s; B_H {:.3} GiB/s; B_P/B_H {:.3}",
        gib_per_second(
            report
                .glm_routed_expert
                .pinned_host_to_device_b_p
                .statistics
                .median_bytes_per_second,
        ),
        gib_per_second(
            report
                .glm_routed_expert
                .host_evaluation_b_h
                .median_service_bytes_per_second,
        ),
        report.glm_routed_expert.b_p_over_b_h,
    );
    Ok(())
}

fn explicit_cuda_ordinal(value: &str, flag: &str) -> Result<usize> {
    let ordinal = value.strip_prefix("cuda:").with_context(|| {
        format!("{flag} must be an explicit cuda:N for the local-interconnect profile")
    })?;
    if ordinal.is_empty() || !ordinal.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{flag} has an invalid CUDA ordinal: {value:?}");
    }
    ordinal
        .parse::<usize>()
        .with_context(|| format!("{flag} CUDA ordinal exceeds usize"))
}

fn gib_per_second(bytes_per_second: f64) -> f64 {
    bytes_per_second / 1024.0 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_profile_requires_explicit_cuda_ordinals_without_fallback() {
        assert_eq!(explicit_cuda_ordinal("cuda:0", "--device").unwrap(), 0);
        for value in ["auto", "cpu", "metal:0", "cuda:", "cuda:-1", "cuda:1x"] {
            let error = explicit_cuda_ordinal(value, "--device")
                .unwrap_err()
                .to_string();
            assert!(error.contains("--device"), "{error}");
        }

        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("must-not-exist.json");
        let error = run_local_interconnect(CalibrateIoConfig {
            payload: None,
            model: Some(temporary.path().join("missing-model")),
            output: output.clone(),
            device: "auto".to_owned(),
            format: IO_PAYLOAD_FORMAT_RAW_SEQUENTIAL.to_owned(),
            peer_device: None,
            warmups: 1,
            samples: 1,
            expert_layer: 3,
            expert_index: 0,
        })
        .unwrap_err();
        assert!(error.to_string().contains("explicit cuda:N"));
        assert!(!output.exists());
    }
}
