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
    {
        let [prefetch, fill] = report
            .glm_routed_expert
            .as_ref()
            .context("local I/O report has no GLM section")?
            .reader_counts;
        println!("GLM calibrated readers: prefetch={prefetch}, fill={fill}");
    }
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
                .as_ref()
                .context("local I/O report has no H3 section")?
                .same_device_d2d
                .statistics
                .median_bytes_per_second,
        ),
        gib_per_second(
            report
                .h3_standard_block_cut
                .as_ref()
                .context("local I/O report has no H3 section")?
                .pinned_host_staged_roundtrip
                .statistics
                .median_bytes_per_second,
        ),
        gib_per_second(
            report
                .h3_standard_block_cut
                .as_ref()
                .context("local I/O report has no H3 section")?
                .tcp_loopback_roundtrip
                .statistics
                .median_bytes_per_second,
        ),
    );
    match &report
        .h3_standard_block_cut
        .as_ref()
        .context("local I/O report has no H3 section")?
        .peer_device_d2d
    {
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
                .as_ref()
                .context("local I/O report has no GLM section")?
                .pinned_host_to_device_b_p
                .statistics
                .median_bytes_per_second,
        ),
        gib_per_second(
            report
                .glm_routed_expert
                .as_ref()
                .context("local I/O report has no GLM section")?
                .host_evaluation_b_h
                .median_service_bytes_per_second,
        ),
        report
            .glm_routed_expert
            .as_ref()
            .context("local I/O report has no GLM section")?
            .b_p_over_b_h,
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

#[derive(Debug, clap::Args)]
pub(super) struct Group4Args {
    #[arg(long, value_parser=["qwen35", "edge0"])]
    adapter: String,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    device: String,
    #[arg(long)]
    host_profile: PathBuf,
    #[arg(long, default_value_t=std::num::NonZeroUsize::new(4096).unwrap())]
    max_context: std::num::NonZeroUsize,
    #[arg(long, requires = "rounds", conflicts_with = "resident_experts")]
    speculative: bool,
    #[arg(long, requires = "speculative")]
    rounds: Option<std::num::NonZeroUsize>,
    #[arg(long)]
    resident_experts: bool,
}

#[cfg(not(feature = "cuda"))]
pub(super) fn run_group4(_args: Group4Args) -> Result<()> {
    bail!("ff bench group4 requires --features cuda")
}

#[cfg(feature = "cuda")]
pub(super) fn run_group4(args: Group4Args) -> Result<()> {
    use flyingfish::{host_profile::HostProfile, runtime::probe::DecodeChoice};
    anyhow::ensure!(
        std::env::var_os("FF_GROUP4_BODY").is_none(),
        "calibration refuses FF_GROUP4_BODY; unset it before ff bench group4"
    );
    anyhow::ensure!(
        args.adapter != "qwen35" || std::env::var_os("QWEN35_GRAPH").is_none_or(|v| v != "0"),
        "ff bench group4 measures graph replay; unset QWEN35_GRAPH=0"
    );
    let ordinals = super::device_parse::parse_device_ordinals(&args.device)?;
    anyhow::ensure!(
        !ordinals.is_empty() && args.device.starts_with("cuda:"),
        "ff bench group4 requires explicit --device cuda:N[,M...]"
    );
    anyhow::ensure!(
        args.adapter == "qwen35" || !args.speculative,
        "Edge0 calibration does not support --speculative"
    );
    anyhow::ensure!(
        args.adapter == "edge0" || !args.resident_experts,
        "--resident-experts applies to Edge0"
    );
    let output = resolve_output_outside_model(&args.host_profile, &args.model)?;
    eprintln!(
        "model: {}; host profile: {}",
        args.model.display(),
        output.display()
    );
    let device = candle_core::Device::new_cuda(ordinals[0])?;
    let mut report = if output.exists() {
        HostProfile::load(&output, &device)?
            .map_err(|e| anyhow::anyhow!("{e}; run ff bench group4"))?
            .into_report()
    } else {
        LocalIoBenchmarkReport {
            schema_version: flyingfish::interconnect_benchmark::LOCAL_IO_BENCHMARK_SCHEMA_VERSION,
            measured_at_unix_ms: u64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_millis(),
            )?,
            model_root: None,
            model_config: None,
            primary_cuda_ordinal: u32::try_from(ordinals[0])?,
            visible_cuda_devices: u32::try_from(cudarc::driver::CudaContext::device_count()?)?,
            fingerprint: flyingfish::runtime::probe::HardwareFingerprint::collect(&device).unwrap(),
            warmup_iterations: 0,
            sample_iterations: 0,
            h3_standard_block_cut: None,
            glm_routed_expert: None,
            group4: Vec::new(),
        }
    };
    let binary = flyingfish::collect_binary_identity()?;
    let mut trials = Vec::new();
    for trial in 0..3 {
        eprintln!("group4 independent calibration {}/3", trial + 1);
        let records = match args.adapter.as_str() {
            "qwen35" => {
                use flyingfish::qwen35::{
                    config::Qwen35Config, gpu::QwenGpu, spec::QwenSpec, weights::Qwen35Weights,
                };
                let config = Qwen35Config::from_model_dir(&args.model)?;
                anyhow::ensure!(
                    args.max_context.get() <= config.text_config.max_position_embeddings,
                    "calibration context exceeds model capacity"
                );
                let weights = Qwen35Weights::open(&args.model)?;
                anyhow::ensure!(
                    !weights.format().is_16bit(),
                    "group4 calibration requires an int4 checkpoint"
                );
                let mut gpu = QwenGpu::with_max_ctx(
                    &ordinals,
                    &weights,
                    &config,
                    args.max_context.get(),
                    ff_qwen35::gpu::force_stream_requested(),
                )?;
                gpu.calibrate_groups()?;
                if args.speculative {
                    let rounds = args
                        .rounds
                        .context("--rounds is required with --speculative")?
                        .get();
                    gpu.push_token(0)?;
                    let mut spec = QwenSpec::new(&mut gpu, &weights, rounds)?;
                    spec.calibrate_groups(&mut gpu)?;
                }
                gpu.calibration_records(&binary, args.rounds.map(|n| n.get()))?
            }
            "edge0" => {
                use flyingfish::edge0::{
                    config::Edge0Config,
                    model::{Edge0Text, configured_max_ctx},
                };
                anyhow::ensure!(
                    args.max_context.get() == configured_max_ctx()?,
                    "Edge0 --max-context must match EDGE0_MAX_CTX ({})",
                    configured_max_ctx()?
                );
                let config = Edge0Config::from_model_dir(&args.model)?;
                let mut model = Edge0Text::load(&args.model, config)?;
                model.enable_gpu_multi(&ordinals, args.resident_experts)?;
                if ordinals.len() == 1 {
                    model.calibrate_group()?;
                } else {
                    model.calibrate_multi_groups()?;
                }
                model.calibration_records(&binary)?
            }
            _ => bail!("unknown group4 adapter"),
        };
        trials.push(records);
    }
    let mut records = Vec::new();
    for record in &trials[0] {
        let repeated = trials
            .iter()
            .map(|trial| {
                let matches = trial
                    .iter()
                    .filter(|other| record.matches(other))
                    .collect::<Vec<_>>();
                anyhow::ensure!(
                    matches.len() == 1,
                    "group4 execution key changed between independent trials; rerun ff bench group4"
                );
                Ok(matches[0].clone())
            })
            .collect::<Result<Vec<_>>>()?;
        records.push(DecodeChoice::combine(repeated)?);
    }
    anyhow::ensure!(
        trials.iter().all(|t| t.len() == records.len()),
        "group4 program count changed during calibration"
    );
    for record in records {
        eprintln!("group4 persisted: {record:?}");
        report.group4.retain(|old| !old.matches(&record));
        report.group4.push(record);
    }
    report.validate()?;
    let json = serde_json::to_vec_pretty(&report)?;
    LocalIoBenchmarkReport::from_json(&json)?;
    let staging = ArtifactStaging::new(&output)?;
    staging.write_bytes(&json)?;
    std::fs::File::open(staging.producer_path())?.sync_all()?;
    flyingfish::runtime::artifact::replace_file_durably(staging.producer_path(), &output)?;
    flyingfish::runtime::artifact::sync_parent_directory(
        output.parent().context("profile parent missing")?,
    )?;
    eprintln!(
        "wrote host profile schema {}: {}",
        report.schema_version,
        output.display()
    );
    Ok(())
}
