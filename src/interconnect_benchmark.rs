//! Explicit local-interconnect and GLM expert-service measurements.
//!
//! "Local" here means the interconnects inside one machine — device-to-device,
//! pinned host staging and TCP loopback — as opposed to a LAN.
//!
//! The legacy payload-driven I/O calibration remains schema 1 in
//! `io_calibration`.  This module owns a separate closed schema because its
//! fixed H3 cut and GLM expert geometries are not interchangeable with an
//! arbitrary sequential file payload.

use crate::{
    glm::{
        config::{GlmConfig, MlpKind},
        fp8, math,
    },
    runtime::identity::{FileStamp, stamp_file},
    runtime::io_calibration::BandwidthSample,
    runtime::probe::HardwareFingerprint,
    runtime::weights::{CachePolicy, ModelWeights, TensorMetadata, WeightSource},
};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    net::{Ipv4Addr, Shutdown, SocketAddrV4, TcpListener, TcpStream},
    path::Path,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice};

pub const LOCAL_IO_BENCHMARK_SCHEMA_VERSION: u32 = 1;
pub const H3_STANDARD_BLOCK_CUT_BYTES: u64 = 405_468_672;
pub const H3_STANDARD_BLOCK_CUT_ROWS: usize = 37_711;
pub const H3_HIDDEN_SIZE: usize = 5_376;
pub const GLM_HIDDEN_SIZE: usize = 4_096;
pub const GLM_ROUTED_EXPERT_INTERMEDIATE_SIZE: usize = 2_048;
pub const GLM_ROUTED_EXPERT_PROJECTIONS: usize = 3;
pub const GLM_ROUTED_EXPERT_BF16_BYTES: u64 = 50_331_648;
pub const DEFAULT_LOCAL_IO_WARMUP_ITERATIONS: usize = 1;
pub const DEFAULT_LOCAL_IO_SAMPLE_ITERATIONS: usize = 5;
pub const MAX_LOCAL_IO_WARMUP_ITERATIONS: usize = 16;
pub const MAX_LOCAL_IO_SAMPLE_ITERATIONS: usize = 31;

const MAX_LOCAL_IO_REPORT_BYTES: usize = 4 * 1024 * 1024;
const LOOPBACK_ACK: u8 = 0xa5;
#[cfg(feature = "cuda")]
const VERIFY_WINDOW_BYTES: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalIoBenchmarkOptions {
    pub primary_cuda_ordinal: usize,
    pub peer_cuda_ordinal: Option<usize>,
    pub warmup_iterations: usize,
    pub sample_iterations: usize,
    pub expert_layer: usize,
    pub expert_index: usize,
}

impl Default for LocalIoBenchmarkOptions {
    fn default() -> Self {
        Self {
            primary_cuda_ordinal: 0,
            peer_cuda_ordinal: None,
            warmup_iterations: DEFAULT_LOCAL_IO_WARMUP_ITERATIONS,
            sample_iterations: DEFAULT_LOCAL_IO_SAMPLE_ITERATIONS,
            expert_layer: 3,
            expert_index: 0,
        }
    }
}

impl LocalIoBenchmarkOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.warmup_iterations <= MAX_LOCAL_IO_WARMUP_ITERATIONS,
            "local I/O warmup count {} exceeds {MAX_LOCAL_IO_WARMUP_ITERATIONS}",
            self.warmup_iterations
        );
        ensure!(
            (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(&self.sample_iterations),
            "local I/O sample count must be in 1..={MAX_LOCAL_IO_SAMPLE_ITERATIONS}"
        );
        if let Some(peer) = self.peer_cuda_ordinal {
            ensure!(
                peer != self.primary_cuda_ordinal,
                "peer CUDA ordinal must differ from the primary ordinal"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleStatistics {
    pub sample_count: u32,
    pub minimum_elapsed_ns: u64,
    pub median_elapsed_ns: u64,
    pub p95_elapsed_ns: u64,
    pub maximum_elapsed_ns: u64,
    pub median_absolute_deviation_ns: u64,
    pub median_bytes_per_second: f64,
}

impl SampleStatistics {
    fn from_samples(samples: &[BandwidthSample]) -> Result<Self> {
        ensure!(!samples.is_empty(), "bandwidth series has no samples");
        let bytes = samples[0].bytes;
        ensure!(
            samples.iter().all(|sample| sample.bytes == bytes),
            "bandwidth series samples do not use one transfer size"
        );
        let elapsed = samples
            .iter()
            .map(|sample| sample.elapsed_ns)
            .collect::<Vec<_>>();
        let summary = summarize_elapsed(&elapsed)?;
        Ok(Self {
            sample_count: u32::try_from(samples.len()).context("sample count exceeds u32")?,
            minimum_elapsed_ns: summary.minimum,
            median_elapsed_ns: summary.median,
            p95_elapsed_ns: summary.p95,
            maximum_elapsed_ns: summary.maximum,
            median_absolute_deviation_ns: summary.mad,
            median_bytes_per_second: bytes as f64 * 1_000_000_000.0 / summary.median as f64,
        })
    }

    fn validate_for(&self, samples: &[BandwidthSample]) -> Result<()> {
        let expected = Self::from_samples(samples)?;
        ensure!(
            self.sample_count == expected.sample_count
                && self.minimum_elapsed_ns == expected.minimum_elapsed_ns
                && self.median_elapsed_ns == expected.median_elapsed_ns
                && self.p95_elapsed_ns == expected.p95_elapsed_ns
                && self.maximum_elapsed_ns == expected.maximum_elapsed_ns
                && self.median_absolute_deviation_ns == expected.median_absolute_deviation_ns,
            "bandwidth statistics disagree with their samples"
        );
        ensure!(
            approximately_equal(
                self.median_bytes_per_second,
                expected.median_bytes_per_second
            ),
            "median bytes/s disagrees with the median sample"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BandwidthSeries {
    pub operation: String,
    pub timing_scope: String,
    pub bytes_per_iteration: u64,
    pub warmup_iterations: u32,
    pub samples: Vec<BandwidthSample>,
    pub statistics: SampleStatistics,
}

/// Build a validated series from measured durations. Exposed to
/// `collective_benchmark`, which measures different operations against the same
/// statistics contract.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn new_bandwidth_series(
    operation: &str,
    timing_scope: &str,
    bytes_per_iteration: u64,
    warmup_iterations: usize,
    elapsed: Vec<Duration>,
) -> Result<BandwidthSeries> {
    BandwidthSeries::new(
        operation,
        timing_scope,
        bytes_per_iteration,
        warmup_iterations,
        elapsed,
    )
}

impl BandwidthSeries {
    fn new(
        operation: &str,
        timing_scope: &str,
        bytes_per_iteration: u64,
        warmup_iterations: usize,
        elapsed: Vec<Duration>,
    ) -> Result<Self> {
        ensure!(
            !operation.trim().is_empty() && !timing_scope.trim().is_empty(),
            "bandwidth operation and timing scope must be non-empty"
        );
        let samples = elapsed
            .into_iter()
            .map(|elapsed| BandwidthSample::from_bytes_and_elapsed(bytes_per_iteration, elapsed))
            .collect::<Result<Vec<_>>>()?;
        let series = Self {
            operation: operation.to_owned(),
            timing_scope: timing_scope.to_owned(),
            bytes_per_iteration,
            warmup_iterations: u32::try_from(warmup_iterations)
                .context("warmup count exceeds u32")?,
            statistics: SampleStatistics::from_samples(&samples)?,
            samples,
        };
        series.validate()?;
        Ok(series)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.operation.trim().is_empty() && !self.timing_scope.trim().is_empty(),
            "bandwidth operation and timing scope must be non-empty"
        );
        ensure!(
            self.bytes_per_iteration > 0,
            "bandwidth series bytes_per_iteration must be positive"
        );
        ensure!(
            usize::try_from(self.warmup_iterations).context("warmup count exceeds usize")?
                <= MAX_LOCAL_IO_WARMUP_ITERATIONS,
            "bandwidth series warmup count is unbounded"
        );
        ensure!(
            (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(&self.samples.len()),
            "bandwidth series sample count is out of bounds"
        );
        for sample in &self.samples {
            sample.validate()?;
            ensure!(
                sample.bytes == self.bytes_per_iteration,
                "bandwidth sample byte count disagrees with its series"
            );
        }
        self.statistics.validate_for(&self.samples)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeerD2dUnavailableReason {
    FewerThanTwoVisibleCudaDevices,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case", deny_unknown_fields)]
pub enum PeerD2dMeasurement {
    Measured {
        source_cuda_ordinal: u32,
        destination_cuda_ordinal: u32,
        destination_fingerprint: Box<HardwareFingerprint>,
        series: BandwidthSeries,
    },
    Unavailable {
        reason: PeerD2dUnavailableReason,
        visible_cuda_devices: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3CutMeasurements {
    pub packed_rows: usize,
    pub hidden_size: usize,
    pub dtype: String,
    pub bytes: u64,
    pub same_device_d2d: BandwidthSeries,
    pub pinned_device_to_host: BandwidthSeries,
    pub pinned_host_to_device: BandwidthSeries,
    pub pinned_host_staged_roundtrip: BandwidthSeries,
    pub tcp_loopback_roundtrip: BandwidthSeries,
    pub peer_device_d2d: PeerD2dMeasurement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertTensorInput {
    pub name: String,
    pub scale_inv_name: String,
    pub shape: [usize; 2],
    pub source_dtype: String,
    pub source_bytes: u64,
    pub scale_shape: [usize; 2],
    pub scale_dtype: String,
    pub scale_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmExpertGeometry {
    pub layer: usize,
    pub expert: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_count: usize,
    pub logical_transfer_dtype: String,
    pub logical_transfer_bytes: u64,
    pub host_compute_dtype: String,
    pub projection_matmul_flops: u64,
    pub tensors: Vec<ExpertTensorInput>,
}

impl GlmExpertGeometry {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size == GLM_HIDDEN_SIZE
                && self.intermediate_size == GLM_ROUTED_EXPERT_INTERMEDIATE_SIZE
                && self.projection_count == GLM_ROUTED_EXPERT_PROJECTIONS,
            "GLM expert benchmark does not use the released [2048,4096] geometry"
        );
        ensure!(
            self.logical_transfer_dtype == "bf16"
                && self.logical_transfer_bytes == GLM_ROUTED_EXPERT_BF16_BYTES
                && self.host_compute_dtype == "f32",
            "GLM expert benchmark dtype/byte contract changed"
        );
        ensure!(
            self.projection_matmul_flops == glm_projection_matmul_flops()?,
            "GLM expert projection FLOP count changed"
        );
        ensure!(
            self.tensors.len() == GLM_ROUTED_EXPERT_PROJECTIONS,
            "GLM expert benchmark must bind three projection tensors"
        );
        let expected_shapes = [
            [self.intermediate_size, self.hidden_size],
            [self.intermediate_size, self.hidden_size],
            [self.hidden_size, self.intermediate_size],
        ];
        let prefix = format!(
            "model.language_model.layers.{}.mlp.experts.{}",
            self.layer, self.expert
        );
        for ((tensor, expected_shape), projection) in self
            .tensors
            .iter()
            .zip(expected_shapes)
            .zip(["gate_proj", "up_proj", "down_proj"])
        {
            let expected_name = format!("{prefix}.{projection}.weight");
            ensure!(
                tensor.shape == expected_shape
                    && tensor.source_dtype == "F8_E4M3"
                    && tensor.scale_dtype == "F32"
                    && tensor.name == expected_name
                    && tensor.scale_inv_name == format!("{expected_name}_scale_inv"),
                "GLM expert tensor metadata disagrees with the released FP8 projection contract"
            );
            ensure!(
                tensor.scale_shape
                    == [
                        expected_shape[0].div_ceil(fp8::FP8_BLOCK_SIZE),
                        expected_shape[1].div_ceil(fp8::FP8_BLOCK_SIZE),
                    ],
                "GLM expert inverse-scale shape is invalid"
            );
            let expected_source_bytes = expected_shape[0]
                .checked_mul(expected_shape[1])
                .and_then(|value| u64::try_from(value).ok())
                .context("GLM expert source byte count overflow")?;
            let expected_scale_bytes = tensor.scale_shape[0]
                .checked_mul(tensor.scale_shape[1])
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .context("GLM expert scale byte count overflow")?;
            ensure!(
                tensor.source_bytes == expected_source_bytes
                    && tensor.scale_bytes == expected_scale_bytes,
                "GLM expert tensor byte counts disagree with dtype and shape"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertEvaluationSample {
    pub load_and_dequantize_elapsed_ns: u64,
    pub compute_elapsed_ns: u64,
    pub service_elapsed_ns: u64,
    pub logical_weight_bytes: u64,
    pub service_bytes_per_second: f64,
    pub projection_matmul_flops: u64,
    pub compute_flops_per_second: f64,
}

impl ExpertEvaluationSample {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.load_and_dequantize_elapsed_ns > 0
                && self.compute_elapsed_ns > 0
                && self.service_elapsed_ns > 0,
            "expert evaluation timing must be positive"
        );
        ensure!(
            self.service_elapsed_ns
                >= self
                    .load_and_dequantize_elapsed_ns
                    .checked_add(self.compute_elapsed_ns)
                    .context("expert timing sum overflow")?,
            "expert service time is shorter than load plus compute"
        );
        ensure!(
            self.logical_weight_bytes == GLM_ROUTED_EXPERT_BF16_BYTES
                && self.projection_matmul_flops == glm_projection_matmul_flops()?,
            "expert evaluation byte/FLOP contract changed"
        );
        let expected_bytes =
            self.logical_weight_bytes as f64 * 1e9 / self.service_elapsed_ns as f64;
        let expected_flops =
            self.projection_matmul_flops as f64 * 1e9 / self.compute_elapsed_ns as f64;
        ensure!(
            approximately_equal(self.service_bytes_per_second, expected_bytes)
                && approximately_equal(self.compute_flops_per_second, expected_flops),
            "expert evaluation rates disagree with their timings"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertEvaluationSeries {
    pub operation: String,
    pub timing_scope: String,
    pub warmup_iterations: u32,
    pub samples: Vec<ExpertEvaluationSample>,
    pub service_statistics: ElapsedStatistics,
    pub load_and_dequantize_statistics: ElapsedStatistics,
    pub compute_statistics: ElapsedStatistics,
    pub median_service_bytes_per_second: f64,
    pub median_compute_flops_per_second: f64,
}

impl ExpertEvaluationSeries {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.operation == "glm_routed_expert_mmap_service"
                && self.timing_scope
                    == "host_monotonic_load_dequantize_and_single_token_cpu_evaluation",
            "unknown GLM expert evaluation measurement scope"
        );
        ensure!(
            (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(&self.samples.len()),
            "expert evaluation sample count is out of bounds"
        );
        for sample in &self.samples {
            sample.validate()?;
        }
        self.service_statistics.validate_for(
            &self
                .samples
                .iter()
                .map(|sample| sample.service_elapsed_ns)
                .collect::<Vec<_>>(),
        )?;
        self.load_and_dequantize_statistics.validate_for(
            &self
                .samples
                .iter()
                .map(|sample| sample.load_and_dequantize_elapsed_ns)
                .collect::<Vec<_>>(),
        )?;
        self.compute_statistics.validate_for(
            &self
                .samples
                .iter()
                .map(|sample| sample.compute_elapsed_ns)
                .collect::<Vec<_>>(),
        )?;
        let expected_service = GLM_ROUTED_EXPERT_BF16_BYTES as f64 * 1e9
            / self.service_statistics.median_elapsed_ns as f64;
        let expected_compute = glm_projection_matmul_flops()? as f64 * 1e9
            / self.compute_statistics.median_elapsed_ns as f64;
        ensure!(
            approximately_equal(self.median_service_bytes_per_second, expected_service)
                && approximately_equal(self.median_compute_flops_per_second, expected_compute),
            "expert evaluation median rates disagree with their summaries"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElapsedStatistics {
    pub sample_count: u32,
    pub minimum_elapsed_ns: u64,
    pub median_elapsed_ns: u64,
    pub p95_elapsed_ns: u64,
    pub maximum_elapsed_ns: u64,
    pub median_absolute_deviation_ns: u64,
}

impl ElapsedStatistics {
    fn from_elapsed(elapsed: &[u64]) -> Result<Self> {
        let summary = summarize_elapsed(elapsed)?;
        Ok(Self {
            sample_count: u32::try_from(elapsed.len()).context("sample count exceeds u32")?,
            minimum_elapsed_ns: summary.minimum,
            median_elapsed_ns: summary.median,
            p95_elapsed_ns: summary.p95,
            maximum_elapsed_ns: summary.maximum,
            median_absolute_deviation_ns: summary.mad,
        })
    }

    fn validate_for(&self, elapsed: &[u64]) -> Result<()> {
        ensure!(
            *self == Self::from_elapsed(elapsed)?,
            "elapsed statistics disagree with their samples"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmExpertMeasurements {
    pub geometry: GlmExpertGeometry,
    pub pinned_host_to_device_b_p: BandwidthSeries,
    pub host_evaluation_b_h: ExpertEvaluationSeries,
    pub b_p_over_b_h: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalIoBenchmarkReport {
    pub schema_version: u32,
    pub measured_at_unix_ms: u64,
    pub model_root: String,
    pub model_config: FileStamp,
    pub primary_cuda_ordinal: u32,
    pub visible_cuda_devices: u32,
    pub fingerprint: HardwareFingerprint,
    pub warmup_iterations: u32,
    pub sample_iterations: u32,
    pub h3_standard_block_cut: H3CutMeasurements,
    pub glm_routed_expert: GlmExpertMeasurements,
}

impl LocalIoBenchmarkReport {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_LOCAL_IO_REPORT_BYTES,
            "local I/O benchmark JSON must contain 1..={MAX_LOCAL_IO_REPORT_BYTES} bytes"
        );
        let report: Self =
            serde_json::from_slice(bytes).context("invalid local I/O benchmark JSON")?;
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == LOCAL_IO_BENCHMARK_SCHEMA_VERSION,
            "unsupported local I/O benchmark schema {}; this build supports {LOCAL_IO_BENCHMARK_SCHEMA_VERSION}",
            self.schema_version
        );
        ensure!(
            self.measured_at_unix_ms > 0 && !self.model_root.trim().is_empty(),
            "local I/O report is missing its timestamp or model root"
        );
        self.model_config.validate()?;
        self.fingerprint.validate()?;
        ensure!(
            self.fingerprint.backend == crate::runtime::probe::DeviceBackend::Cuda,
            "local I/O benchmark requires a CUDA fingerprint"
        );
        ensure!(
            self.visible_cuda_devices > 0
                && self.primary_cuda_ordinal < self.visible_cuda_devices
                && usize::try_from(self.warmup_iterations).context("warmups exceed usize")?
                    <= MAX_LOCAL_IO_WARMUP_ITERATIONS
                && (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(
                    &usize::try_from(self.sample_iterations).context("samples exceed usize")?,
                ),
            "local I/O report iteration/device counts are invalid"
        );
        ensure!(
            self.h3_standard_block_cut.packed_rows == H3_STANDARD_BLOCK_CUT_ROWS
                && self.h3_standard_block_cut.hidden_size == H3_HIDDEN_SIZE
                && self.h3_standard_block_cut.dtype == "bf16"
                && self.h3_standard_block_cut.bytes == H3_STANDARD_BLOCK_CUT_BYTES
                && u64::try_from(H3_STANDARD_BLOCK_CUT_ROWS)
                    .context("H3 packed rows exceed u64")?
                    .checked_mul(
                        u64::try_from(H3_HIDDEN_SIZE).context("H3 hidden size exceeds u64")?
                    )
                    .and_then(|value| value.checked_mul(2))
                    == Some(H3_STANDARD_BLOCK_CUT_BYTES),
            "H3 block-cut geometry or byte count changed"
        );
        let roundtrip_bytes = H3_STANDARD_BLOCK_CUT_BYTES
            .checked_mul(2)
            .context("H3 roundtrip byte count overflow")?;
        for (series, operation, bytes) in [
            (
                &self.h3_standard_block_cut.same_device_d2d,
                "same_device_cuda_d2d",
                H3_STANDARD_BLOCK_CUT_BYTES,
            ),
            (
                &self.h3_standard_block_cut.pinned_device_to_host,
                "cuda_device_to_pinned_host",
                H3_STANDARD_BLOCK_CUT_BYTES,
            ),
            (
                &self.h3_standard_block_cut.pinned_host_to_device,
                "pinned_host_to_cuda_device",
                H3_STANDARD_BLOCK_CUT_BYTES,
            ),
            (
                &self.h3_standard_block_cut.pinned_host_staged_roundtrip,
                "pinned_host_staged_cuda_roundtrip",
                roundtrip_bytes,
            ),
            (
                &self.h3_standard_block_cut.tcp_loopback_roundtrip,
                "tcp_ipv4_loopback_framed_roundtrip",
                H3_STANDARD_BLOCK_CUT_BYTES,
            ),
        ] {
            series.validate()?;
            ensure!(
                series.operation == operation
                    && series.bytes_per_iteration == bytes
                    && series.warmup_iterations == self.warmup_iterations
                    && series.samples.len()
                        == usize::try_from(self.sample_iterations)
                            .context("sample count exceeds usize")?,
                "local I/O series identity or iteration counts disagree with the report"
            );
        }
        validate_peer_measurement(
            &self.h3_standard_block_cut.peer_device_d2d,
            self.primary_cuda_ordinal,
            self.visible_cuda_devices,
            self.warmup_iterations,
            self.sample_iterations,
        )?;
        self.glm_routed_expert.geometry.validate()?;
        self.glm_routed_expert
            .pinned_host_to_device_b_p
            .validate()?;
        self.glm_routed_expert.host_evaluation_b_h.validate()?;
        ensure!(
            self.glm_routed_expert.pinned_host_to_device_b_p.operation
                == "glm_expert_pinned_host_to_device_b_p"
                && self
                    .glm_routed_expert
                    .pinned_host_to_device_b_p
                    .bytes_per_iteration
                    == self.glm_routed_expert.geometry.logical_transfer_bytes
                && self
                    .glm_routed_expert
                    .pinned_host_to_device_b_p
                    .warmup_iterations
                    == self.warmup_iterations
                && self
                    .glm_routed_expert
                    .pinned_host_to_device_b_p
                    .samples
                    .len()
                    == self.sample_iterations as usize
                && self.glm_routed_expert.host_evaluation_b_h.warmup_iterations
                    == self.warmup_iterations
                && self.glm_routed_expert.host_evaluation_b_h.samples.len()
                    == self.sample_iterations as usize,
            "B_P/B_H identity, geometry, or iteration counts disagree with the report"
        );
        let expected_ratio = self
            .glm_routed_expert
            .pinned_host_to_device_b_p
            .statistics
            .median_bytes_per_second
            / self
                .glm_routed_expert
                .host_evaluation_b_h
                .median_service_bytes_per_second;
        ensure!(
            approximately_equal(self.glm_routed_expert.b_p_over_b_h, expected_ratio),
            "B_P/B_H ratio disagrees with measured medians"
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ElapsedSummary {
    minimum: u64,
    median: u64,
    p95: u64,
    maximum: u64,
    mad: u64,
}

fn summarize_elapsed(elapsed: &[u64]) -> Result<ElapsedSummary> {
    ensure!(!elapsed.is_empty(), "timing series is empty");
    ensure!(
        elapsed.iter().all(|value| *value > 0),
        "timing series contains a zero-duration sample"
    );
    let mut sorted = elapsed.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let p95_index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
    let mut deviations = sorted
        .iter()
        .map(|value| value.abs_diff(median))
        .collect::<Vec<_>>();
    deviations.sort_unstable();
    Ok(ElapsedSummary {
        minimum: sorted[0],
        median,
        p95: sorted[p95_index],
        maximum: sorted[sorted.len() - 1],
        mad: deviations[deviations.len() / 2],
    })
}

pub(crate) fn approximately_equal(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= expected.abs() * 1e-9 + 1e-6
}

fn duration_ns(duration: Duration) -> Result<u64> {
    let value = u64::try_from(duration.as_nanos()).context("timing exceeds u64 nanoseconds")?;
    ensure!(value > 0, "timing elapsed below clock resolution");
    Ok(value)
}

fn glm_projection_matmul_flops() -> Result<u64> {
    u64::try_from(GLM_HIDDEN_SIZE)
        .context("GLM hidden size exceeds u64")?
        .checked_mul(
            u64::try_from(GLM_ROUTED_EXPERT_INTERMEDIATE_SIZE)
                .context("GLM expert width exceeds u64")?,
        )
        .and_then(|value| value.checked_mul(6))
        .context("GLM expert projection FLOPs overflow")
}

fn validate_peer_measurement(
    measurement: &PeerD2dMeasurement,
    primary: u32,
    visible: u32,
    warmups: u32,
    samples: u32,
) -> Result<()> {
    match measurement {
        PeerD2dMeasurement::Measured {
            source_cuda_ordinal,
            destination_cuda_ordinal,
            destination_fingerprint,
            series,
        } => {
            ensure!(
                visible >= 2
                    && *source_cuda_ordinal == primary
                    && destination_cuda_ordinal != source_cuda_ordinal
                    && *destination_cuda_ordinal < visible,
                "peer D2D ordinals disagree with visible hardware"
            );
            destination_fingerprint.validate()?;
            ensure!(
                destination_fingerprint.backend == crate::runtime::probe::DeviceBackend::Cuda,
                "peer D2D destination is not CUDA"
            );
            series.validate()?;
            ensure!(
                series.operation == "peer_cuda_d2d"
                    && series.bytes_per_iteration == H3_STANDARD_BLOCK_CUT_BYTES
                    && series.warmup_iterations == warmups
                    && series.samples.len() == samples as usize,
                "peer D2D identity or iteration counts disagree with the report"
            );
        }
        PeerD2dMeasurement::Unavailable {
            reason,
            visible_cuda_devices,
        } => {
            ensure!(
                *reason == PeerD2dUnavailableReason::FewerThanTwoVisibleCudaDevices
                    && *visible_cuda_devices == visible
                    && visible < 2,
                "peer D2D unavailability is not justified by visible hardware"
            );
        }
    }
    Ok(())
}

pub fn measure_local_io_benchmark(
    glm_model_root: &Path,
    device: &Device,
    options: LocalIoBenchmarkOptions,
) -> Result<LocalIoBenchmarkReport> {
    options.validate()?;
    ensure!(
        device.is_cuda(),
        "local-interconnect I/O benchmark requires an explicitly selected CUDA device"
    );
    let supplied_metadata = fs::symlink_metadata(glm_model_root).with_context(|| {
        format!(
            "failed to inspect GLM model root {}",
            glm_model_root.display()
        )
    })?;
    ensure!(
        supplied_metadata.file_type().is_dir(),
        "GLM model root must be a non-symlink directory: {}",
        glm_model_root.display()
    );
    let model_root = fs::canonicalize(glm_model_root).with_context(|| {
        format!(
            "failed to resolve GLM model root {}",
            glm_model_root.display()
        )
    })?;
    let config_path = model_root.join("config.json");
    let model_config = stamp_file(&config_path)?;
    model_config.validate()?;
    let config = GlmConfig::from_file(&config_path)?;
    let fingerprint = HardwareFingerprint::collect(device);
    fingerprint.validate()?;

    let selected = Device::new_cuda(options.primary_cuda_ordinal).with_context(|| {
        format!(
            "failed to initialize selected CUDA device {}",
            options.primary_cuda_ordinal
        )
    })?;
    let selected_fingerprint = HardwareFingerprint::collect(&selected);
    selected_fingerprint.validate()?;
    ensure!(
        fingerprint == selected_fingerprint,
        "provided device does not match primary CUDA ordinal {}",
        options.primary_cuda_ordinal
    );

    let visible_cuda_devices = visible_cuda_device_count()?;
    ensure!(
        options.primary_cuda_ordinal < visible_cuda_devices,
        "primary CUDA ordinal {} is outside the {visible_cuda_devices} visible devices",
        options.primary_cuda_ordinal
    );
    if visible_cuda_devices >= 2 && options.peer_cuda_ordinal.is_none() {
        bail!(
            "{visible_cuda_devices} CUDA devices are visible; --peer-device is required so peer D2D never selects hardware implicitly"
        );
    }
    if let Some(peer) = options.peer_cuda_ordinal {
        ensure!(
            peer < visible_cuda_devices,
            "peer CUDA ordinal {peer} is outside the {visible_cuda_devices} visible devices"
        );
    }

    let geometry = inspect_glm_expert_geometry(
        &model_root,
        &config,
        options.expert_layer,
        options.expert_index,
    )?;
    let (same_device_d2d, pinned_d2h, pinned_h2d, pinned_roundtrip, peer_device_d2d) =
        measure_h3_cut_transfers(
            device,
            options,
            visible_cuda_devices,
            H3_STANDARD_BLOCK_CUT_BYTES,
        )?;
    let tcp_loopback_roundtrip = measure_tcp_loopback(
        H3_STANDARD_BLOCK_CUT_BYTES,
        options.warmup_iterations,
        options.sample_iterations,
    )?;
    let pinned_host_to_device_b_p = measure_pinned_expert_h2d(
        device,
        geometry.logical_transfer_bytes,
        options.warmup_iterations,
        options.sample_iterations,
    )?;
    let host_evaluation_b_h = measure_host_expert_evaluation(
        &model_root,
        &config,
        &geometry,
        options.warmup_iterations,
        options.sample_iterations,
    )?;
    let b_p_over_b_h = pinned_host_to_device_b_p.statistics.median_bytes_per_second
        / host_evaluation_b_h.median_service_bytes_per_second;
    let report = LocalIoBenchmarkReport {
        schema_version: LOCAL_IO_BENCHMARK_SCHEMA_VERSION,
        measured_at_unix_ms: u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("system clock is before the Unix epoch")?
                .as_millis(),
        )
        .context("measurement timestamp exceeds u64")?,
        model_root: model_root
            .to_str()
            .context("resolved GLM model root is not valid UTF-8")?
            .to_owned(),
        model_config,
        primary_cuda_ordinal: u32::try_from(options.primary_cuda_ordinal)
            .context("primary CUDA ordinal exceeds u32")?,
        visible_cuda_devices: u32::try_from(visible_cuda_devices)
            .context("visible CUDA device count exceeds u32")?,
        fingerprint,
        warmup_iterations: u32::try_from(options.warmup_iterations)
            .context("warmup count exceeds u32")?,
        sample_iterations: u32::try_from(options.sample_iterations)
            .context("sample count exceeds u32")?,
        h3_standard_block_cut: H3CutMeasurements {
            packed_rows: H3_STANDARD_BLOCK_CUT_ROWS,
            hidden_size: H3_HIDDEN_SIZE,
            dtype: "bf16".to_owned(),
            bytes: H3_STANDARD_BLOCK_CUT_BYTES,
            same_device_d2d,
            pinned_device_to_host: pinned_d2h,
            pinned_host_to_device: pinned_h2d,
            pinned_host_staged_roundtrip: pinned_roundtrip,
            tcp_loopback_roundtrip,
            peer_device_d2d,
        },
        glm_routed_expert: GlmExpertMeasurements {
            geometry,
            pinned_host_to_device_b_p,
            host_evaluation_b_h,
            b_p_over_b_h,
        },
    };
    report.validate()?;
    Ok(report)
}

#[cfg(feature = "cuda")]
fn visible_cuda_device_count() -> Result<usize> {
    let count = CudaContext::device_count().context("failed to query visible CUDA device count")?;
    ensure!(count > 0, "CUDA reported no visible devices");
    usize::try_from(count).context("visible CUDA device count exceeds usize")
}

#[cfg(not(feature = "cuda"))]
fn visible_cuda_device_count() -> Result<usize> {
    bail!("local-interconnect I/O benchmark requires a build with the cuda feature")
}

fn inspect_glm_expert_geometry(
    model_root: &Path,
    config: &GlmConfig,
    layer: usize,
    expert: usize,
) -> Result<GlmExpertGeometry> {
    let text = &config.text_config;
    ensure!(
        text.hidden_size == GLM_HIDDEN_SIZE
            && text.moe_intermediate_size == GLM_ROUTED_EXPERT_INTERMEDIATE_SIZE,
        "GLM checkpoint geometry is [{},{}], expected released [{GLM_ROUTED_EXPERT_INTERMEDIATE_SIZE},{GLM_HIDDEN_SIZE}] experts",
        text.moe_intermediate_size,
        text.hidden_size
    );
    ensure!(
        layer < text.num_hidden_layers && text.mlp_layer_types.get(layer) == Some(&MlpKind::Sparse),
        "GLM layer {layer} is not a routed-expert layer"
    );
    ensure!(
        expert < text.n_routed_experts,
        "GLM expert {expert} is outside 0..{}",
        text.n_routed_experts
    );
    let weights = ModelWeights::open(model_root, WeightSource::Mmap, CachePolicy::new(1))?;
    let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
    let tensors = ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|projection| {
            let name = format!("{prefix}.{projection}.weight");
            let scale_inv_name = format!("{prefix}.{projection}.weight_scale_inv");
            expert_tensor_input(
                weights.metadata(&name)?,
                weights.metadata(&scale_inv_name)?,
                scale_inv_name,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let geometry = GlmExpertGeometry {
        layer,
        expert,
        hidden_size: text.hidden_size,
        intermediate_size: text.moe_intermediate_size,
        projection_count: tensors.len(),
        logical_transfer_dtype: "bf16".to_owned(),
        logical_transfer_bytes: GLM_ROUTED_EXPERT_BF16_BYTES,
        host_compute_dtype: "f32".to_owned(),
        projection_matmul_flops: glm_projection_matmul_flops()?,
        tensors,
    };
    geometry.validate()?;
    Ok(geometry)
}

fn expert_tensor_input(
    weight: TensorMetadata,
    scale: TensorMetadata,
    scale_inv_name: String,
) -> Result<ExpertTensorInput> {
    let shape: [usize; 2] = weight
        .shape
        .as_slice()
        .try_into()
        .context("GLM expert projection must be rank two")?;
    let scale_shape: [usize; 2] = scale
        .shape
        .as_slice()
        .try_into()
        .context("GLM expert inverse scale must be rank two")?;
    Ok(ExpertTensorInput {
        name: weight.name,
        scale_inv_name,
        shape,
        source_dtype: weight.dtype,
        source_bytes: u64::try_from(weight.bytes).context("expert weight bytes exceed u64")?,
        scale_shape,
        scale_dtype: scale.dtype,
        scale_bytes: u64::try_from(scale.bytes).context("expert scale bytes exceed u64")?,
    })
}

fn measure_elapsed(
    warmup_iterations: usize,
    sample_iterations: usize,
    mut operation: impl FnMut() -> Result<()>,
) -> Result<Vec<Duration>> {
    ensure!(
        warmup_iterations <= MAX_LOCAL_IO_WARMUP_ITERATIONS
            && (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(&sample_iterations),
        "local I/O iteration counts are out of bounds"
    );
    for _ in 0..warmup_iterations {
        operation()?;
    }
    let mut samples = Vec::with_capacity(sample_iterations);
    for _ in 0..sample_iterations {
        let started = Instant::now();
        operation()?;
        let elapsed = started.elapsed();
        ensure!(
            elapsed.as_nanos() > 0,
            "local I/O sample elapsed below timer resolution"
        );
        samples.push(elapsed);
    }
    Ok(samples)
}

fn pattern_byte(index: usize) -> u8 {
    ((index.wrapping_mul(131).wrapping_add(17)) & 0xff) as u8
}

fn fill_pattern(bytes: &mut [u8]) {
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = pattern_byte(index);
    }
}

#[cfg(feature = "cuda")]
fn verify_cuda_boundaries(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    value: &CudaSlice<u8>,
    length: usize,
) -> Result<()> {
    let window = VERIFY_WINDOW_BYTES.min(length);
    ensure!(window > 0, "CUDA verification buffer is empty");
    let first = stream
        .clone_dtoh(&value.slice(0..window))
        .context("failed to verify leading CUDA transfer bytes")?;
    let tail_start = length - window;
    let last = stream
        .clone_dtoh(&value.slice(tail_start..length))
        .context("failed to verify trailing CUDA transfer bytes")?;
    stream
        .synchronize()
        .context("failed to synchronize CUDA transfer verification")?;
    ensure!(
        first
            .iter()
            .enumerate()
            .all(|(index, byte)| *byte == pattern_byte(index))
            && last
                .iter()
                .enumerate()
                .all(|(index, byte)| *byte == pattern_byte(tail_start + index)),
        "CUDA transfer boundary verification failed"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn measure_h3_cut_transfers(
    device: &Device,
    options: LocalIoBenchmarkOptions,
    visible_cuda_devices: usize,
    bytes: u64,
) -> Result<(
    BandwidthSeries,
    BandwidthSeries,
    BandwidthSeries,
    BandwidthSeries,
    PeerD2dMeasurement,
)> {
    let length = usize::try_from(bytes).context("H3 block cut bytes exceed usize")?;
    let stream = device
        .as_cuda_device()
        .context("H3 transfer benchmark requires CUDA")?
        .cuda_stream();
    let mut initialization = vec![0u8; length];
    fill_pattern(&mut initialization);
    let source = stream
        .clone_htod(&initialization)
        .context("failed to initialize H3 D2D source")?;
    drop(initialization);
    let mut destination =
        unsafe { stream.alloc::<u8>(length) }.context("failed to allocate H3 D2D destination")?;
    stream
        .synchronize()
        .context("failed to synchronize H3 transfer setup")?;

    let same_device_elapsed =
        measure_elapsed(options.warmup_iterations, options.sample_iterations, || {
            stream
                .memcpy_dtod(&source, &mut destination)
                .context("same-device D2D copy failed")?;
            stream
                .synchronize()
                .context("same-device D2D synchronization failed")
        })?;
    verify_cuda_boundaries(&stream, &destination, length)?;
    let same_device = BandwidthSeries::new(
        "same_device_cuda_d2d",
        "host_monotonic_preallocated_copy_plus_stream_synchronize",
        bytes,
        options.warmup_iterations,
        same_device_elapsed,
    )?;

    let peer = measure_peer_d2d(
        &stream,
        &source,
        length,
        bytes,
        options,
        visible_cuda_devices,
    )?;

    let context = stream.context().clone();
    let mut pinned = unsafe { context.alloc_pinned::<u8>(length) }
        .context("failed to allocate page-locked H3 host staging buffer")?;
    let d2h_elapsed =
        measure_elapsed(options.warmup_iterations, options.sample_iterations, || {
            stream
                .memcpy_dtoh(&source, &mut pinned)
                .context("pinned D2H copy failed")?;
            stream
                .synchronize()
                .context("pinned D2H synchronization failed")
        })?;
    let pinned_d2h = BandwidthSeries::new(
        "cuda_device_to_pinned_host",
        "host_monotonic_preallocated_copy_plus_stream_synchronize",
        bytes,
        options.warmup_iterations,
        d2h_elapsed,
    )?;
    let h2d_elapsed =
        measure_elapsed(options.warmup_iterations, options.sample_iterations, || {
            stream
                .memcpy_htod(&pinned, &mut destination)
                .context("pinned H2D copy failed")?;
            stream
                .synchronize()
                .context("pinned H2D synchronization failed")
        })?;
    verify_cuda_boundaries(&stream, &destination, length)?;
    let pinned_h2d = BandwidthSeries::new(
        "pinned_host_to_cuda_device",
        "host_monotonic_preallocated_copy_plus_stream_synchronize",
        bytes,
        options.warmup_iterations,
        h2d_elapsed,
    )?;
    let roundtrip_elapsed =
        measure_elapsed(options.warmup_iterations, options.sample_iterations, || {
            stream
                .memcpy_dtoh(&source, &mut pinned)
                .context("host-staged roundtrip D2H leg failed")?;
            stream
                .memcpy_htod(&pinned, &mut destination)
                .context("host-staged roundtrip H2D leg failed")?;
            stream
                .synchronize()
                .context("host-staged roundtrip synchronization failed")
        })?;
    verify_cuda_boundaries(&stream, &destination, length)?;
    let roundtrip_bytes = bytes
        .checked_mul(2)
        .context("host-staged roundtrip byte count overflow")?;
    let roundtrip = BandwidthSeries::new(
        "pinned_host_staged_cuda_roundtrip",
        "host_monotonic_preallocated_d2h_then_h2d_plus_stream_synchronize",
        roundtrip_bytes,
        options.warmup_iterations,
        roundtrip_elapsed,
    )?;
    Ok((same_device, pinned_d2h, pinned_h2d, roundtrip, peer))
}

#[cfg(not(feature = "cuda"))]
fn measure_h3_cut_transfers(
    _device: &Device,
    _options: LocalIoBenchmarkOptions,
    _visible_cuda_devices: usize,
    _bytes: u64,
) -> Result<(
    BandwidthSeries,
    BandwidthSeries,
    BandwidthSeries,
    BandwidthSeries,
    PeerD2dMeasurement,
)> {
    bail!("local-interconnect I/O benchmark requires a build with the cuda feature")
}

#[cfg(feature = "cuda")]
fn measure_peer_d2d(
    source_stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    source: &CudaSlice<u8>,
    length: usize,
    bytes: u64,
    options: LocalIoBenchmarkOptions,
    visible_cuda_devices: usize,
) -> Result<PeerD2dMeasurement> {
    if visible_cuda_devices < 2 {
        ensure!(
            options.peer_cuda_ordinal.is_none(),
            "peer CUDA device was requested but fewer than two devices are visible"
        );
        return Ok(PeerD2dMeasurement::Unavailable {
            reason: PeerD2dUnavailableReason::FewerThanTwoVisibleCudaDevices,
            visible_cuda_devices: u32::try_from(visible_cuda_devices)
                .context("visible CUDA device count exceeds u32")?,
        });
    }
    let peer_ordinal = options
        .peer_cuda_ordinal
        .context("peer CUDA ordinal is required when multiple devices are visible")?;
    let peer_device = Device::new_cuda(peer_ordinal)
        .with_context(|| format!("failed to initialize peer CUDA device {peer_ordinal}"))?;
    let destination_fingerprint = HardwareFingerprint::collect(&peer_device);
    destination_fingerprint.validate()?;
    let peer_stream = peer_device
        .as_cuda_device()
        .context("peer device is not CUDA")?
        .cuda_stream();
    source_stream
        .synchronize()
        .context("failed to synchronize peer-copy source")?;
    let mut destination = unsafe { peer_stream.alloc::<u8>(length) }
        .context("failed to allocate peer D2D destination")?;
    let elapsed = measure_elapsed(options.warmup_iterations, options.sample_iterations, || {
        peer_stream
            .memcpy_dtod(source, &mut destination)
            .context("peer CUDA D2D copy failed")?;
        peer_stream
            .synchronize()
            .context("peer CUDA D2D synchronization failed")
    })?;
    verify_cuda_boundaries(&peer_stream, &destination, length)?;
    Ok(PeerD2dMeasurement::Measured {
        source_cuda_ordinal: u32::try_from(options.primary_cuda_ordinal)
            .context("source CUDA ordinal exceeds u32")?,
        destination_cuda_ordinal: u32::try_from(peer_ordinal)
            .context("destination CUDA ordinal exceeds u32")?,
        destination_fingerprint: Box::new(destination_fingerprint),
        series: BandwidthSeries::new(
            "peer_cuda_d2d",
            "host_monotonic_preallocated_peer_copy_plus_destination_stream_synchronize",
            bytes,
            options.warmup_iterations,
            elapsed,
        )?,
    })
}

fn measure_tcp_loopback(
    bytes: u64,
    warmup_iterations: usize,
    sample_iterations: usize,
) -> Result<BandwidthSeries> {
    let length = usize::try_from(bytes).context("loopback payload exceeds usize")?;
    let mut payload = vec![0u8; length];
    fill_pattern(&mut payload);
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
        .context("failed to bind TCP loopback benchmark listener")?;
    let address = listener
        .local_addr()
        .context("failed to inspect TCP loopback listener address")?;
    let iterations = warmup_iterations
        .checked_add(sample_iterations)
        .context("loopback iteration count overflow")?;
    let receiver = thread::Builder::new()
        .name("ff-io-loopback".to_owned())
        .spawn(move || -> Result<Vec<u8>> {
            let (mut stream, peer) = listener
                .accept()
                .context("failed to accept TCP loopback benchmark connection")?;
            ensure!(
                peer.ip().is_loopback(),
                "TCP benchmark peer is not loopback"
            );
            stream
                .set_nodelay(true)
                .context("failed to set TCP loopback TCP_NODELAY")?;
            let mut received = vec![0u8; length];
            for _ in 0..iterations {
                let mut header = [0u8; 8];
                stream
                    .read_exact(&mut header)
                    .context("failed to read TCP loopback frame header")?;
                ensure!(
                    u64::from_le_bytes(header) == bytes,
                    "TCP loopback frame length changed"
                );
                stream
                    .read_exact(&mut received)
                    .context("failed to read TCP loopback payload")?;
                stream
                    .write_all(&[LOOPBACK_ACK])
                    .context("failed to write TCP loopback acknowledgement")?;
            }
            Ok(received)
        })
        .context("failed to start TCP loopback receiver")?;
    let sender_result = (|| -> Result<Vec<Duration>> {
        let mut stream = TcpStream::connect(address)
            .context("failed to connect TCP loopback benchmark sender")?;
        stream
            .set_nodelay(true)
            .context("failed to set TCP loopback sender TCP_NODELAY")?;
        measure_elapsed(warmup_iterations, sample_iterations, || {
            stream
                .write_all(&bytes.to_le_bytes())
                .context("failed to write TCP loopback frame header")?;
            stream
                .write_all(&payload)
                .context("failed to write TCP loopback payload")?;
            stream
                .flush()
                .context("failed to flush TCP loopback payload")?;
            let mut ack = [0u8; 1];
            stream
                .read_exact(&mut ack)
                .context("failed to read TCP loopback acknowledgement")?;
            ensure!(
                ack[0] == LOOPBACK_ACK,
                "TCP loopback acknowledgement changed"
            );
            Ok(())
        })
        .inspect(|_| {
            let _ = stream.shutdown(Shutdown::Both);
        })
    })();
    let receiver_result = receiver
        .join()
        .map_err(|_| anyhow::anyhow!("TCP loopback receiver panicked"))?;
    let elapsed = sender_result?;
    ensure!(
        receiver_result? == payload,
        "TCP loopback delivered different bytes than it sent"
    );
    BandwidthSeries::new(
        "tcp_ipv4_loopback_framed_roundtrip",
        "host_monotonic_write_length_and_payload_through_one_byte_ack",
        bytes,
        warmup_iterations,
        elapsed,
    )
}

#[cfg(feature = "cuda")]
fn measure_pinned_expert_h2d(
    device: &Device,
    bytes: u64,
    warmup_iterations: usize,
    sample_iterations: usize,
) -> Result<BandwidthSeries> {
    ensure!(
        bytes == GLM_ROUTED_EXPERT_BF16_BYTES,
        "B_P must transfer one complete dequantized GLM routed expert"
    );
    let length = usize::try_from(bytes).context("GLM expert bytes exceed usize")?;
    let stream = device
        .as_cuda_device()
        .context("B_P requires CUDA")?
        .cuda_stream();
    let mut pinned = unsafe { stream.context().alloc_pinned::<u8>(length) }
        .context("failed to allocate page-locked GLM expert buffer")?;
    fill_pattern(
        pinned
            .as_mut_slice()
            .context("failed to initialize page-locked GLM expert buffer")?,
    );
    let mut destination = unsafe { stream.alloc::<u8>(length) }
        .context("failed to allocate GLM expert CUDA destination")?;
    stream
        .synchronize()
        .context("failed to synchronize GLM expert transfer setup")?;
    let elapsed = measure_elapsed(warmup_iterations, sample_iterations, || {
        stream
            .memcpy_htod(&pinned, &mut destination)
            .context("GLM expert pinned H2D copy failed")?;
        stream
            .synchronize()
            .context("GLM expert pinned H2D synchronization failed")
    })?;
    verify_cuda_boundaries(&stream, &destination, length)?;
    BandwidthSeries::new(
        "glm_expert_pinned_host_to_device_b_p",
        "host_monotonic_preallocated_write_combined_pinned_copy_plus_stream_synchronize",
        bytes,
        warmup_iterations,
        elapsed,
    )
}

#[cfg(not(feature = "cuda"))]
fn measure_pinned_expert_h2d(
    _device: &Device,
    _bytes: u64,
    _warmup_iterations: usize,
    _sample_iterations: usize,
) -> Result<BandwidthSeries> {
    bail!("B_P requires a build with the cuda feature")
}

fn measure_host_expert_evaluation(
    model_root: &Path,
    config: &GlmConfig,
    geometry: &GlmExpertGeometry,
    warmup_iterations: usize,
    sample_iterations: usize,
) -> Result<ExpertEvaluationSeries> {
    let weights = ModelWeights::open(model_root, WeightSource::Mmap, CachePolicy::new(1))?;
    let hidden = Tensor::arange(0f32, geometry.hidden_size as f32, &Device::Cpu)?
        .affine(1.0 / geometry.hidden_size as f64, -0.5)?;
    for _ in 0..warmup_iterations {
        let (_, output) = evaluate_one_host_expert(&weights, config, geometry, &hidden)?;
        std::hint::black_box(output);
    }
    let mut samples = Vec::with_capacity(sample_iterations);
    let mut final_output = None;
    for _ in 0..sample_iterations {
        let (sample, output) = evaluate_one_host_expert(&weights, config, geometry, &hidden)?;
        samples.push(sample);
        final_output = Some(output);
    }
    final_output.context("GLM expert benchmark produced no output")?;
    let service_elapsed = samples
        .iter()
        .map(|sample| sample.service_elapsed_ns)
        .collect::<Vec<_>>();
    let load_elapsed = samples
        .iter()
        .map(|sample| sample.load_and_dequantize_elapsed_ns)
        .collect::<Vec<_>>();
    let compute_elapsed = samples
        .iter()
        .map(|sample| sample.compute_elapsed_ns)
        .collect::<Vec<_>>();
    let service_statistics = ElapsedStatistics::from_elapsed(&service_elapsed)?;
    let load_and_dequantize_statistics = ElapsedStatistics::from_elapsed(&load_elapsed)?;
    let compute_statistics = ElapsedStatistics::from_elapsed(&compute_elapsed)?;
    let series = ExpertEvaluationSeries {
        operation: "glm_routed_expert_mmap_service".to_owned(),
        timing_scope: "host_monotonic_load_dequantize_and_single_token_cpu_evaluation".to_owned(),
        warmup_iterations: u32::try_from(warmup_iterations).context("warmups exceed u32")?,
        median_service_bytes_per_second: GLM_ROUTED_EXPERT_BF16_BYTES as f64 * 1e9
            / service_statistics.median_elapsed_ns as f64,
        median_compute_flops_per_second: glm_projection_matmul_flops()? as f64 * 1e9
            / compute_statistics.median_elapsed_ns as f64,
        samples,
        service_statistics,
        load_and_dequantize_statistics,
        compute_statistics,
    };
    series.validate()?;
    Ok(series)
}

fn evaluate_one_host_expert(
    weights: &ModelWeights,
    config: &GlmConfig,
    geometry: &GlmExpertGeometry,
    hidden: &Tensor,
) -> Result<(ExpertEvaluationSample, Vec<f32>)> {
    let service_started = Instant::now();
    let load_started = Instant::now();
    // Fused evaluation never materializes a dequantized projection, so "load"
    // here is the quantized bytes and their scales and nothing more.
    let gate = load_expert_projection(weights, &geometry.tensors[0])?;
    let up = load_expert_projection(weights, &geometry.tensors[1])?;
    let down = load_expert_projection(weights, &geometry.tensors[2])?;
    let load_elapsed_ns = duration_ns(load_started.elapsed())?;
    let compute_started = Instant::now();
    let gate_output =
        fp8::fused_block_fp8_matvec(&gate.weight, &gate.scale_inv, hidden)?.unsqueeze(0)?;
    let up_output = fp8::fused_block_fp8_matvec(&up.weight, &up.scale_inv, hidden)?.unsqueeze(0)?;
    let activated =
        math::clamped_swiglu(&gate_output, &up_output, config.text_config.swiglu_limit)?;
    let output = fp8::fused_block_fp8_matvec(
        &down.weight,
        &down.scale_inv,
        &activated.squeeze(0)?.contiguous()?,
    )?;
    ensure!(
        output.dims() == [geometry.hidden_size],
        "GLM expert output shape changed"
    );
    let values = output.to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "GLM expert CPU evaluation produced NaN/Inf"
    );
    let compute_elapsed_ns = duration_ns(compute_started.elapsed())?;
    let service_elapsed_ns = duration_ns(service_started.elapsed())?;
    let sample = ExpertEvaluationSample {
        load_and_dequantize_elapsed_ns: load_elapsed_ns,
        compute_elapsed_ns,
        service_elapsed_ns,
        logical_weight_bytes: geometry.logical_transfer_bytes,
        service_bytes_per_second: geometry.logical_transfer_bytes as f64 * 1e9
            / service_elapsed_ns as f64,
        projection_matmul_flops: geometry.projection_matmul_flops,
        compute_flops_per_second: geometry.projection_matmul_flops as f64 * 1e9
            / compute_elapsed_ns as f64,
    };
    sample.validate()?;
    Ok((sample, values))
}

/// One projection as the checkpoint stores it: quantized values and the block
/// scales that interpret them, neither expanded.
struct QuantizedProjection {
    weight: Tensor,
    scale_inv: Tensor,
}

fn load_expert_projection(
    weights: &ModelWeights,
    input: &ExpertTensorInput,
) -> Result<QuantizedProjection> {
    Ok(QuantizedProjection {
        weight: weights.load(&input.name, &Device::Cpu)?,
        scale_inv: weights.load(&input.scale_inv_name, &Device::Cpu)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_statistics_are_derived_and_tampering_is_rejected() {
        let series = BandwidthSeries::new(
            "test_copy",
            "host_monotonic_test",
            1_000,
            1,
            vec![
                Duration::from_nanos(10),
                Duration::from_nanos(30),
                Duration::from_nanos(20),
                Duration::from_nanos(50),
                Duration::from_nanos(40),
            ],
        )
        .unwrap();
        assert_eq!(series.statistics.minimum_elapsed_ns, 10);
        assert_eq!(series.statistics.median_elapsed_ns, 30);
        assert_eq!(series.statistics.p95_elapsed_ns, 50);
        assert_eq!(series.statistics.maximum_elapsed_ns, 50);
        assert_eq!(series.statistics.median_absolute_deviation_ns, 10);
        series.validate().unwrap();

        let mut tampered = series;
        tampered.statistics.median_elapsed_ns += 1;
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn tcp_loopback_uses_real_framing_and_retains_each_sample() {
        let series = measure_tcp_loopback(4_096, 1, 3).unwrap();
        series.validate().unwrap();
        assert_eq!(series.operation, "tcp_ipv4_loopback_framed_roundtrip");
        assert_eq!(series.bytes_per_iteration, 4_096);
        assert_eq!(series.warmup_iterations, 1);
        assert_eq!(series.samples.len(), 3);
        assert!(
            series
                .samples
                .iter()
                .all(|sample| sample.elapsed_ns > 0 && sample.bytes_per_second > 0.0)
        );
    }

    #[test]
    fn local_options_reject_implicit_or_unbounded_sampling() {
        let options = LocalIoBenchmarkOptions {
            sample_iterations: 0,
            ..Default::default()
        };
        assert!(options.validate().is_err());
        let options = LocalIoBenchmarkOptions {
            sample_iterations: MAX_LOCAL_IO_SAMPLE_ITERATIONS + 1,
            ..Default::default()
        };
        assert!(options.validate().is_err());
        let options = LocalIoBenchmarkOptions {
            sample_iterations: 1,
            warmup_iterations: MAX_LOCAL_IO_WARMUP_ITERATIONS + 1,
            ..Default::default()
        };
        assert!(options.validate().is_err());
        let options = LocalIoBenchmarkOptions {
            warmup_iterations: 0,
            peer_cuda_ordinal: Some(0),
            ..Default::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn released_expert_byte_and_flop_geometry_is_locked() {
        assert_eq!(
            GLM_ROUTED_EXPERT_BF16_BYTES,
            (GLM_HIDDEN_SIZE * GLM_ROUTED_EXPERT_INTERMEDIATE_SIZE * 2 * 3) as u64
        );
        assert_eq!(glm_projection_matmul_flops().unwrap(), 50_331_648);
        assert_eq!(H3_STANDARD_BLOCK_CUT_BYTES, 405_468_672);
        assert_eq!(
            H3_STANDARD_BLOCK_CUT_ROWS * H3_HIDDEN_SIZE * 2,
            H3_STANDARD_BLOCK_CUT_BYTES as usize
        );
    }

    #[test]
    fn local_io_report_fixture_passes_the_strict_schema() {
        let report =
            LocalIoBenchmarkReport::from_json(include_bytes!("testdata/local-io-report.json"))
                .unwrap();
        assert_eq!(report.visible_cuda_devices, 1);
        assert_eq!(report.sample_iterations, 11);
        assert!(matches!(
            report.h3_standard_block_cut.peer_device_d2d,
            PeerD2dMeasurement::Unavailable {
                reason: PeerD2dUnavailableReason::FewerThanTwoVisibleCudaDevices,
                visible_cuda_devices: 1,
            }
        ));
        assert!((report.glm_routed_expert.b_p_over_b_h - 36.18138119669186).abs() < 1e-12);
    }
}
