//! Concurrent ring all-reduce measurement.
//!
//! Every inter-device rate this project has recorded so far was measured point
//! to point, with one pair of devices active (`interconnect_benchmark`'s
//! `peer_cuda_d2d`). A ring all-reduce runs several pairs at once over shared
//! PCIe, and the tensor-split cost model charges its communication at that
//! point-to-point rate. This module measures the collective itself to find out
//! whether that substitution is legitimate: N ranks, one CUDA device each,
//! exchanging the H3 standard block cut in the ring schedule a Megatron-shaped
//! all-reduce would use, with every rank driving its own link at the same time.
//!
//! What is deliberately not here: a collective implementation for the runtime
//! to use. Nothing below is called by any model path, and the report it writes
//! authorizes nothing. It answers one question — does concurrency collapse the
//! per-link rate — and records the answer in a closed schema.

use crate::interconnect_benchmark::{
    BandwidthSeries, H3_HIDDEN_SIZE, H3_STANDARD_BLOCK_CUT_BYTES, H3_STANDARD_BLOCK_CUT_ROWS,
    MAX_LOCAL_IO_SAMPLE_ITERATIONS, MAX_LOCAL_IO_WARMUP_ITERATIONS, approximately_equal,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(any(feature = "cuda", test))]
use crate::interconnect_benchmark::new_bandwidth_series;
#[cfg(feature = "cuda")]
use crate::runtime::probe::HardwareFingerprint;
#[cfg(any(feature = "cuda", test))]
use anyhow::anyhow;
#[cfg(not(feature = "cuda"))]
use anyhow::bail;
#[cfg(any(feature = "cuda", test))]
use candle_core::{DType, Device, Tensor};
#[cfg(any(feature = "cuda", test))]
#[cfg(any(feature = "cuda", test))]
use std::sync::{Condvar, Mutex, RwLock};
#[cfg(any(feature = "cuda", test))]
use std::time::Duration;
#[cfg(any(feature = "cuda", test))]
use std::time::Instant;

pub const COLLECTIVE_BENCHMARK_SCHEMA_VERSION: u32 = 2;

/// One raw BF16 hidden state, as `interconnect_benchmark` already names it.
/// The collective moves this object because it is what a Megatron-shaped
/// all-reduce reduces: the output of one attention or feed-forward projection.
pub const COLLECTIVE_PAYLOAD_BYTES: u64 = H3_STANDARD_BLOCK_CUT_BYTES;
pub const COLLECTIVE_PAYLOAD_ELEMENTS: u64 = H3_STANDARD_BLOCK_CUT_BYTES / 2;

/// The H3 shape: 50 transformer blocks and 2 refiner blocks, each contributing
/// one all-reduce after the attention output projection and one after the
/// feed-forward down projection.
pub const H3_TRANSFORMER_BLOCKS: u64 = 50;
pub const H3_REFINER_BLOCKS: u64 = 2;
pub const H3_COLLECTIVES_PER_EVALUATION: u64 = 2 * (H3_TRANSFORMER_BLOCKS + H3_REFINER_BLOCKS);

/// The two rates a tensor-split projection divides, both of which belong to the
/// host they were taken on.
///
/// Neither has a defensible default. A single-device evaluation time is a
/// property of one GPU running one model, and a charged interconnect rate is a
/// property of one set of links; carrying either as a constant would project
/// the machine it was measured on onto every machine that later ran the tool,
/// and would do so most confidently on hosts that measured nothing themselves.
/// So the projection is only made when a caller supplies both.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostModelBaseline {
    /// One evaluation on one device, measured on the host being projected.
    pub single_device_evaluation_seconds: f64,
    /// The interconnect rate the projection charges communication at. The
    /// natural source is this run's own point-to-point reference.
    pub charged_bytes_per_second: f64,
}

impl CostModelBaseline {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.single_device_evaluation_seconds.is_finite()
                && self.single_device_evaluation_seconds > 0.0
                && self.charged_bytes_per_second.is_finite()
                && self.charged_bytes_per_second > 0.0,
            "cost-model baseline rates must be positive and finite"
        );
        Ok(())
    }
}

/// How far a measured evaluation total may sit from the derived one before the
/// derived row is reported as replaced rather than standing.
pub const COST_MODEL_TOTAL_TOLERANCE: f64 = 0.10;

pub const MIN_COLLECTIVE_RANKS: usize = 2;
pub const MAX_COLLECTIVE_RANKS: usize = 8;
pub const DEFAULT_COLLECTIVE_RANK_COUNTS: [usize; 2] = [2, 4];

/// Elements read back per verified window at each end of every reduced chunk.
pub const COLLECTIVE_VERIFY_WINDOW_ELEMENTS: usize = 4_096;

/// Values are integers small enough that every partial sum the ring forms is
/// exact in BF16, so the reduced buffer can be checked bit for bit rather than
/// within a tolerance. The modulus does not divide the payload or any chunk
/// length, so a chunk delivered to the wrong offset shifts the phase and is
/// caught.
#[cfg(any(feature = "cuda", test))]
const COLLECTIVE_VALUE_MODULUS: u64 = 5;

const MAX_COLLECTIVE_REPORT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectiveBenchmarkOptions {
    /// The ring, in order. Rank `r` receives from rank `r - 1`; naming the
    /// order explicitly keeps this benchmark from inferring a topology, which
    /// `nvidia-smi topo -m` was measured not to predict on this hardware.
    pub cuda_ordinals: Vec<usize>,
    pub rank_counts: Vec<usize>,
    pub warmup_iterations: usize,
    pub sample_iterations: usize,
}

impl CollectiveBenchmarkOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.cuda_ordinals.is_empty() && self.cuda_ordinals.len() <= MAX_COLLECTIVE_RANKS,
            "the collective benchmark needs 1..={MAX_COLLECTIVE_RANKS} explicit CUDA ordinals"
        );
        let mut seen = std::collections::BTreeSet::new();
        for ordinal in &self.cuda_ordinals {
            ensure!(
                seen.insert(*ordinal),
                "CUDA ordinal {ordinal} is named twice; a rank cannot share a device"
            );
        }
        ensure!(
            !self.rank_counts.is_empty(),
            "the collective benchmark needs at least one rank count"
        );
        ensure!(
            self.rank_counts.windows(2).all(|pair| pair[0] < pair[1]),
            "rank counts must be strictly increasing and unique"
        );
        for count in &self.rank_counts {
            ensure!(
                (MIN_COLLECTIVE_RANKS..=MAX_COLLECTIVE_RANKS).contains(count),
                "rank count {count} is outside {MIN_COLLECTIVE_RANKS}..={MAX_COLLECTIVE_RANKS}"
            );
            ensure!(
                COLLECTIVE_PAYLOAD_ELEMENTS.is_multiple_of(*count as u64),
                "rank count {count} does not divide the {COLLECTIVE_PAYLOAD_ELEMENTS}-element \
                 payload; the ring would move unequal chunks and its algorithmic byte count \
                 would no longer be 2(N-1)/N of the payload"
            );
        }
        ensure!(
            self.warmup_iterations <= MAX_LOCAL_IO_WARMUP_ITERATIONS,
            "collective warmup count {} exceeds {MAX_LOCAL_IO_WARMUP_ITERATIONS}",
            self.warmup_iterations
        );
        ensure!(
            (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(&self.sample_iterations),
            "collective sample count must be in 1..={MAX_LOCAL_IO_SAMPLE_ITERATIONS}"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectivePayload {
    pub packed_rows: usize,
    pub hidden_size: usize,
    pub dtype: String,
    pub bytes: u64,
    pub elements: u64,
}

impl CollectivePayload {
    fn current() -> Self {
        Self {
            packed_rows: H3_STANDARD_BLOCK_CUT_ROWS,
            hidden_size: H3_HIDDEN_SIZE,
            dtype: "bf16".to_owned(),
            bytes: COLLECTIVE_PAYLOAD_BYTES,
            elements: COLLECTIVE_PAYLOAD_ELEMENTS,
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            *self == Self::current(),
            "collective payload geometry disagrees with the H3 standard block cut"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveUnavailableReason {
    /// The host, or the operator's device list, offers fewer devices than the
    /// rank count needs. Ranks never share a device: two ranks on one device
    /// would measure that device's own memory, not the interconnect.
    FewerDevicesThanRanks,
}

/// The single-pair rate the concurrent measurements are read against. It is
/// re-measured here rather than quoted from an earlier report so the ratio is
/// taken on one host, in one process, at one payload size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case", deny_unknown_fields)]
pub enum PointToPointReference {
    Measured {
        source_cuda_ordinal: u32,
        destination_cuda_ordinal: u32,
        series: BandwidthSeries,
    },
    Unavailable {
        reason: CollectiveUnavailableReason,
        available_cuda_devices: u32,
    },
}

/// A positive record that the reduced buffer was read back and checked, not an
/// assertion that it was. The digest covers every window that was compared, so
/// two runs that agree bit for bit publish one digest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveVerification {
    pub window_elements: u32,
    pub verified_windows: u32,
    pub verified_elements: u64,
}

impl CollectiveVerification {
    fn validate(&self, rank_count: u32) -> Result<()> {
        let expected_windows = 2 * rank_count * rank_count;
        ensure!(
            self.window_elements == COLLECTIVE_VERIFY_WINDOW_ELEMENTS as u32
                && self.verified_windows == expected_windows
                && self.verified_elements
                    == u64::from(self.verified_windows) * u64::from(self.window_elements),
            "collective verification did not cover both ends of every chunk on every rank"
        );
        Ok(())
    }
}

/// What one rank count measured. Boxed into its own struct because the ring's
/// three series dwarf the unavailable arm, and most hosts record the latter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredRing {
    pub rank_count: u32,
    pub cuda_ordinals: Vec<u32>,
    pub chunk_elements: u64,
    /// `2(N-1)/N × payload`: what one rank sends, and receives, to complete
    /// one ring all-reduce. The rate every derived second below is charged at
    /// divides this by the collective's wall time.
    pub algorithmic_bytes_per_rank: u64,
    /// Every rank pulls the whole payload from its ring predecessor at once.
    /// This is the point-to-point copy with N pairs active, and is the most
    /// direct answer to T0's question.
    pub concurrent_neighbor_d2d: BandwidthSeries,
    /// The ring schedule with the reduction removed, so the collective's
    /// transfer cost is separable from its arithmetic.
    pub ring_all_reduce_transfer_only: BandwidthSeries,
    /// The collective itself: scatter-reduce then all-gather, BF16 adds
    /// included.
    pub ring_all_reduce: BandwidthSeries,
    pub verification: CollectiveVerification,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case", deny_unknown_fields)]
pub enum RankCountMeasurement {
    Measured(Box<MeasuredRing>),
    Unavailable {
        rank_count: u32,
        reason: CollectiveUnavailableReason,
        available_cuda_devices: u32,
    },
}

impl RankCountMeasurement {
    pub fn rank_count(&self) -> u32 {
        match self {
            Self::Measured(ring) => ring.rank_count,
            Self::Unavailable { rank_count, .. } => *rank_count,
        }
    }

    fn collective_bytes_per_second(&self) -> Option<f64> {
        match self {
            Self::Measured(ring) => Some(ring.ring_all_reduce.statistics.median_bytes_per_second),
            Self::Unavailable { .. } => None,
        }
    }

    fn concurrent_bytes_per_second(&self) -> Option<f64> {
        match self {
            Self::Measured(ring) => Some(
                ring.concurrent_neighbor_d2d
                    .statistics
                    .median_bytes_per_second,
            ),
            Self::Unavailable { .. } => None,
        }
    }

    fn validate(&self, warmups: u32, samples: u32) -> Result<()> {
        match self {
            Self::Measured(ring) => {
                let MeasuredRing {
                    rank_count,
                    cuda_ordinals,
                    chunk_elements,
                    algorithmic_bytes_per_rank,
                    concurrent_neighbor_d2d,
                    ring_all_reduce_transfer_only,
                    ring_all_reduce,
                    verification,
                } = ring.as_ref();
                let ranks = usize::try_from(*rank_count).context("rank count exceeds usize")?;
                ensure!(
                    (MIN_COLLECTIVE_RANKS..=MAX_COLLECTIVE_RANKS).contains(&ranks)
                        && cuda_ordinals.len() == ranks,
                    "measured rank count disagrees with its device list"
                );
                let mut seen = std::collections::BTreeSet::new();
                for ordinal in cuda_ordinals {
                    ensure!(
                        seen.insert(*ordinal),
                        "a measured ring names CUDA ordinal {ordinal} twice"
                    );
                }
                ensure!(
                    *chunk_elements == COLLECTIVE_PAYLOAD_ELEMENTS / u64::from(*rank_count)
                        && COLLECTIVE_PAYLOAD_ELEMENTS.is_multiple_of(u64::from(*rank_count)),
                    "ring chunk element count disagrees with the payload and rank count"
                );
                ensure!(
                    *algorithmic_bytes_per_rank == algorithmic_bytes_per_rank_for(ranks)?,
                    "ring algorithmic byte count is not 2(N-1)/N of the payload"
                );
                for (series, operation, bytes) in [
                    (
                        concurrent_neighbor_d2d,
                        "concurrent_ring_neighbor_cuda_d2d",
                        COLLECTIVE_PAYLOAD_BYTES,
                    ),
                    (
                        ring_all_reduce_transfer_only,
                        "ring_all_reduce_transfer_only",
                        *algorithmic_bytes_per_rank,
                    ),
                    (
                        ring_all_reduce,
                        "ring_all_reduce_bf16_sum",
                        *algorithmic_bytes_per_rank,
                    ),
                ] {
                    series.validate()?;
                    ensure!(
                        series.operation == operation
                            && series.bytes_per_iteration == bytes
                            && series.warmup_iterations == warmups
                            && series.samples.len()
                                == usize::try_from(samples).context("samples exceed usize")?,
                        "collective series identity or iteration counts disagree with the report"
                    );
                }
                verification.validate(*rank_count)
            }
            Self::Unavailable {
                rank_count,
                reason,
                available_cuda_devices,
            } => {
                ensure!(
                    *reason == CollectiveUnavailableReason::FewerDevicesThanRanks
                        && u64::from(*available_cuda_devices) < u64::from(*rank_count),
                    "collective unavailability is not justified by the available devices"
                );
                Ok(())
            }
        }
    }
}

/// One row of the tensor-split cost table, recomputed here from the same
/// constants and then, where a rank count was measured, recomputed again from
/// the measured collective rate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostModelRow {
    pub rank_count: u32,
    pub bytes_per_rank_per_evaluation: u64,
    /// Present only when a baseline was supplied. Absent means no projection
    /// was made, which is what a host that measured nothing should report.
    pub derived: Option<DerivedCostModelRow>,
    pub measured: Option<MeasuredCostModelRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedCostModelRow {
    pub communication_seconds: f64,
    pub compute_seconds: f64,
    pub total_seconds: f64,
    pub speedup: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCostModelRow {
    pub collective_bytes_per_second: f64,
    /// The concurrent neighbour rate over the single-pair rate. T0's gate is
    /// written against this number.
    pub concurrent_fraction_of_point_to_point: Option<f64>,
    pub communication_seconds: f64,
    /// A measured communication time stands on its own; a total, a speedup and
    /// an error against the projection all need a single-device baseline, so
    /// they are absent exactly when no baseline was supplied.
    pub total_seconds: Option<f64>,
    pub speedup: Option<f64>,
    pub relative_total_error: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostModelVerdict {
    /// No rank count could be measured on this host, so the table is neither
    /// confirmed nor replaced. This is the honest outcome on a single-GPU host
    /// and is not a pass.
    Unmeasured,
    /// Every measured row lands within the tolerance of its derived row.
    Stands,
    /// A measured row differs from its derived row by more than the tolerance,
    /// but still finishes an evaluation faster than one device does. The
    /// measured row replaces the derived one.
    Replaced,
    /// A measured row shows no latency win at all. This outcome stops the
    /// design.
    Refuted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostModelReview {
    pub hidden_state_bytes: u64,
    pub collectives_per_evaluation: u64,
    pub baseline: Option<CostModelBaseline>,
    pub total_tolerance: f64,
    pub rows: Vec<CostModelRow>,
    pub verdict: CostModelVerdict,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveBenchmarkReport {
    pub schema_version: u32,
    pub measured_at_unix_ms: u64,
    pub visible_cuda_devices: u32,
    /// The ring the operator named, whether or not every rank count used it.
    pub cuda_ordinals: Vec<u32>,
    pub fingerprints: Vec<crate::runtime::probe::HardwareFingerprint>,
    pub warmup_iterations: u32,
    pub sample_iterations: u32,
    pub payload: CollectivePayload,
    pub point_to_point_reference: PointToPointReference,
    pub rank_counts: Vec<RankCountMeasurement>,
    pub cost_model: CostModelReview,
}

impl CollectiveBenchmarkReport {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_COLLECTIVE_REPORT_BYTES,
            "collective benchmark JSON must contain 1..={MAX_COLLECTIVE_REPORT_BYTES} bytes"
        );
        let report: Self =
            serde_json::from_slice(bytes).context("invalid collective benchmark JSON")?;
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == COLLECTIVE_BENCHMARK_SCHEMA_VERSION,
            "unsupported collective benchmark schema {}; this build supports \
             {COLLECTIVE_BENCHMARK_SCHEMA_VERSION}",
            self.schema_version
        );
        ensure!(
            self.measured_at_unix_ms > 0,
            "collective report is missing its timestamp"
        );
        ensure!(
            !self.cuda_ordinals.is_empty()
                && self.cuda_ordinals.len() <= MAX_COLLECTIVE_RANKS
                && self.fingerprints.len() == self.cuda_ordinals.len(),
            "collective report device list and fingerprint list disagree"
        );
        let mut seen = std::collections::BTreeSet::new();
        for ordinal in &self.cuda_ordinals {
            ensure!(
                seen.insert(*ordinal),
                "collective report names CUDA ordinal {ordinal} twice"
            );
            ensure!(
                *ordinal < self.visible_cuda_devices,
                "collective report names CUDA ordinal {ordinal} outside the {} visible devices",
                self.visible_cuda_devices
            );
        }
        for fingerprint in &self.fingerprints {
            fingerprint.validate()?;
            ensure!(
                fingerprint.backend == crate::runtime::probe::DeviceBackend::Cuda,
                "collective benchmark requires CUDA fingerprints"
            );
        }
        ensure!(
            usize::try_from(self.warmup_iterations).context("warmups exceed usize")?
                <= MAX_LOCAL_IO_WARMUP_ITERATIONS
                && (1..=MAX_LOCAL_IO_SAMPLE_ITERATIONS).contains(
                    &usize::try_from(self.sample_iterations).context("samples exceed usize")?
                ),
            "collective report iteration counts are invalid"
        );
        self.payload.validate()?;
        match &self.point_to_point_reference {
            PointToPointReference::Measured {
                source_cuda_ordinal,
                destination_cuda_ordinal,
                series,
            } => {
                ensure!(
                    source_cuda_ordinal != destination_cuda_ordinal
                        && self.cuda_ordinals.contains(source_cuda_ordinal)
                        && self.cuda_ordinals.contains(destination_cuda_ordinal),
                    "the point-to-point reference names devices outside the ring"
                );
                series.validate()?;
                ensure!(
                    series.operation == "point_to_point_peer_cuda_d2d"
                        && series.bytes_per_iteration == COLLECTIVE_PAYLOAD_BYTES
                        && series.warmup_iterations == self.warmup_iterations
                        && series.samples.len()
                            == usize::try_from(self.sample_iterations)
                                .context("samples exceed usize")?,
                    "point-to-point reference identity or iteration counts disagree"
                );
            }
            PointToPointReference::Unavailable {
                reason,
                available_cuda_devices,
            } => ensure!(
                *reason == CollectiveUnavailableReason::FewerDevicesThanRanks
                    && *available_cuda_devices < 2,
                "point-to-point unavailability is not justified by the available devices"
            ),
        }
        ensure!(
            !self.rank_counts.is_empty()
                && self
                    .rank_counts
                    .windows(2)
                    .all(|pair| pair[0].rank_count() < pair[1].rank_count()),
            "collective rank counts must be present, unique and increasing"
        );
        for measurement in &self.rank_counts {
            measurement.validate(self.warmup_iterations, self.sample_iterations)?;
        }
        // Re-derived from the report's own recorded baseline, so validation
        // checks the report against itself rather than against whatever this
        // build happens to think a baseline should be.
        let expected = review_cost_model(
            &self.rank_counts,
            point_to_point_bytes_per_second(&self.point_to_point_reference),
            self.cost_model.baseline,
        )?;
        ensure!(
            cost_model_matches(&self.cost_model, &expected),
            "collective cost-model review disagrees with the measurements it was derived from"
        );
        Ok(())
    }
}

fn point_to_point_bytes_per_second(reference: &PointToPointReference) -> Option<f64> {
    match reference {
        PointToPointReference::Measured { series, .. } => {
            Some(series.statistics.median_bytes_per_second)
        }
        PointToPointReference::Unavailable { .. } => None,
    }
}

/// `2(N-1)/N × payload`, exactly: the payload's element count is divisible by
/// every admitted rank count, so this is integer arithmetic rather than a
/// rounded product.
fn algorithmic_bytes_per_rank(ranks: usize) -> Result<u64> {
    algorithmic_bytes_per_rank_for(ranks)
}

fn algorithmic_bytes_per_rank_for(ranks: usize) -> Result<u64> {
    let ranks = u64::try_from(ranks).context("rank count exceeds u64")?;
    ensure!(
        (MIN_COLLECTIVE_RANKS as u64..=MAX_COLLECTIVE_RANKS as u64).contains(&ranks),
        "rank count {ranks} is outside the admitted range"
    );
    let chunk_bytes = COLLECTIVE_PAYLOAD_BYTES
        .checked_div(ranks)
        .context("rank count is zero")?;
    ensure!(
        chunk_bytes * ranks == COLLECTIVE_PAYLOAD_BYTES,
        "rank count {ranks} does not divide the payload"
    );
    chunk_bytes
        .checked_mul(2 * (ranks - 1))
        .context("ring algorithmic byte count overflow")
}

fn review_cost_model(
    measurements: &[RankCountMeasurement],
    point_to_point: Option<f64>,
    baseline: Option<CostModelBaseline>,
) -> Result<CostModelReview> {
    let mut rows = Vec::with_capacity(measurements.len());
    let mut verdict = CostModelVerdict::Unmeasured;
    for measurement in measurements {
        let ranks =
            usize::try_from(measurement.rank_count()).context("rank count exceeds usize")?;
        let bytes_per_rank_per_evaluation = algorithmic_bytes_per_rank(ranks)?
            .checked_mul(H3_COLLECTIVES_PER_EVALUATION)
            .context("per-evaluation collective byte count overflow")?;
        let derived = baseline.map(|baseline| {
            let communication_seconds =
                bytes_per_rank_per_evaluation as f64 / baseline.charged_bytes_per_second;
            let compute_seconds = baseline.single_device_evaluation_seconds / ranks as f64;
            let total_seconds = communication_seconds + compute_seconds;
            DerivedCostModelRow {
                communication_seconds,
                compute_seconds,
                total_seconds,
                speedup: baseline.single_device_evaluation_seconds / total_seconds,
            }
        });
        let measured = measurement.collective_bytes_per_second().map(|rate| {
            let communication_seconds = bytes_per_rank_per_evaluation as f64 / rate;
            MeasuredCostModelRow {
                collective_bytes_per_second: rate,
                concurrent_fraction_of_point_to_point: measurement
                    .concurrent_bytes_per_second()
                    .zip(point_to_point)
                    .map(|(concurrent, reference)| concurrent / reference),
                communication_seconds,
                // A measured communication time can stand alone; a total and a
                // speedup cannot, because both need a single-device baseline.
                total_seconds: derived
                    .map(|derived| communication_seconds + derived.compute_seconds),
                speedup: derived.map(|derived| {
                    baseline
                        .expect("a derived row implies a baseline")
                        .single_device_evaluation_seconds
                        / (communication_seconds + derived.compute_seconds)
                }),
                relative_total_error: derived.map(|derived| {
                    ((communication_seconds + derived.compute_seconds) - derived.total_seconds)
                        .abs()
                        / derived.total_seconds
                }),
            }
        });
        if let Some(measured) = measured.as_ref() {
            verdict = worst_verdict(verdict, row_verdict(measured));
        }
        rows.push(CostModelRow {
            rank_count: measurement.rank_count(),
            bytes_per_rank_per_evaluation,
            derived,
            measured,
        });
    }
    Ok(CostModelReview {
        hidden_state_bytes: COLLECTIVE_PAYLOAD_BYTES,
        collectives_per_evaluation: H3_COLLECTIVES_PER_EVALUATION,
        baseline,
        total_tolerance: COST_MODEL_TOTAL_TOLERANCE,
        rows,
        verdict,
    })
}

/// A measured row can only stand against, or refute, a projection there is one
/// to compare with. Without a baseline it is a measurement and nothing more.
fn row_verdict(row: &MeasuredCostModelRow) -> CostModelVerdict {
    let (Some(speedup), Some(error)) = (row.speedup, row.relative_total_error) else {
        return CostModelVerdict::Unmeasured;
    };
    if speedup <= 1.0 {
        CostModelVerdict::Refuted
    } else if error <= COST_MODEL_TOTAL_TOLERANCE {
        CostModelVerdict::Stands
    } else {
        CostModelVerdict::Replaced
    }
}

fn optionally_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => approximately_equal(left, right),
        _ => false,
    }
}

/// The report carries one verdict for every measured row, so the least
/// favourable one wins: a table that predicts N=2 and misses N=4 has not stood.
fn worst_verdict(left: CostModelVerdict, right: CostModelVerdict) -> CostModelVerdict {
    fn rank(verdict: CostModelVerdict) -> u8 {
        match verdict {
            CostModelVerdict::Stands => 0,
            CostModelVerdict::Unmeasured => 1,
            CostModelVerdict::Replaced => 2,
            CostModelVerdict::Refuted => 3,
        }
    }
    match (left, right) {
        (CostModelVerdict::Unmeasured, other) => other,
        (other, CostModelVerdict::Unmeasured) => other,
        _ => {
            if rank(left) >= rank(right) {
                left
            } else {
                right
            }
        }
    }
}

fn cost_model_matches(actual: &CostModelReview, expected: &CostModelReview) -> bool {
    actual.hidden_state_bytes == expected.hidden_state_bytes
        && actual.collectives_per_evaluation == expected.collectives_per_evaluation
        && actual.baseline == expected.baseline
        && approximately_equal(actual.total_tolerance, expected.total_tolerance)
        && actual.verdict == expected.verdict
        && actual.rows.len() == expected.rows.len()
        && actual
            .rows
            .iter()
            .zip(&expected.rows)
            .all(|(actual, expected)| {
                actual.rank_count == expected.rank_count
                    && actual.bytes_per_rank_per_evaluation
                        == expected.bytes_per_rank_per_evaluation
                    && match (actual.derived, expected.derived) {
                        (None, None) => true,
                        (Some(actual), Some(expected)) => {
                            approximately_equal(
                                actual.communication_seconds,
                                expected.communication_seconds,
                            ) && approximately_equal(
                                actual.compute_seconds,
                                expected.compute_seconds,
                            ) && approximately_equal(actual.total_seconds, expected.total_seconds)
                                && approximately_equal(actual.speedup, expected.speedup)
                        }
                        _ => false,
                    }
                    && match (&actual.measured, &expected.measured) {
                        (None, None) => true,
                        (Some(actual), Some(expected)) => {
                            approximately_equal(
                                actual.collective_bytes_per_second,
                                expected.collective_bytes_per_second,
                            ) && match (
                                actual.concurrent_fraction_of_point_to_point,
                                expected.concurrent_fraction_of_point_to_point,
                            ) {
                                (None, None) => true,
                                (Some(actual), Some(expected)) => {
                                    approximately_equal(actual, expected)
                                }
                                _ => false,
                            } && approximately_equal(
                                actual.communication_seconds,
                                expected.communication_seconds,
                            ) && optionally_equal(actual.total_seconds, expected.total_seconds)
                                && optionally_equal(actual.speedup, expected.speedup)
                                && optionally_equal(
                                    actual.relative_total_error,
                                    expected.relative_total_error,
                                )
                        }
                        _ => false,
                    }
            })
}

fn unix_millis() -> Result<u64> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_millis(),
    )
    .context("measurement timestamp exceeds u64")
}

#[cfg(not(feature = "cuda"))]
pub fn measure_collective_benchmark(
    _options: CollectiveBenchmarkOptions,
    _baseline: impl FnOnce(Option<f64>) -> Option<CostModelBaseline>,
) -> Result<CollectiveBenchmarkReport> {
    let _ = unix_millis()?;
    bail!("the collective benchmark requires a build with the cuda feature")
}

#[cfg(feature = "cuda")]
pub fn measure_collective_benchmark(
    options: CollectiveBenchmarkOptions,
    baseline: impl FnOnce(Option<f64>) -> Option<CostModelBaseline>,
) -> Result<CollectiveBenchmarkReport> {
    options.validate()?;
    let visible = visible_cuda_device_count()?;
    for ordinal in &options.cuda_ordinals {
        ensure!(
            *ordinal < visible,
            "CUDA ordinal {ordinal} is outside the {visible} visible devices"
        );
    }
    let devices = options
        .cuda_ordinals
        .iter()
        .map(|ordinal| {
            Device::new_cuda(*ordinal)
                .with_context(|| format!("failed to initialize CUDA device {ordinal}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let fingerprints = devices
        .iter()
        .map(|device| {
            let fingerprint = HardwareFingerprint::collect(device);
            fingerprint.validate()?;
            ensure!(
                fingerprint.backend == crate::runtime::probe::DeviceBackend::Cuda,
                "a named collective device is not CUDA"
            );
            Ok(fingerprint)
        })
        .collect::<Result<Vec<_>>>()?;

    let available = u32::try_from(devices.len()).context("device count exceeds u32")?;
    let point_to_point_reference = if devices.len() >= 2 {
        let series = measure_point_to_point(&devices[0], &devices[1], &options)?;
        PointToPointReference::Measured {
            source_cuda_ordinal: u32::try_from(options.cuda_ordinals[1])
                .context("CUDA ordinal exceeds u32")?,
            destination_cuda_ordinal: u32::try_from(options.cuda_ordinals[0])
                .context("CUDA ordinal exceeds u32")?,
            series,
        }
    } else {
        PointToPointReference::Unavailable {
            reason: CollectiveUnavailableReason::FewerDevicesThanRanks,
            available_cuda_devices: available,
        }
    };

    let mut rank_counts = Vec::with_capacity(options.rank_counts.len());
    for count in &options.rank_counts {
        rank_counts.push(if *count <= devices.len() {
            measure_rank_count(
                &devices[..*count],
                &options.cuda_ordinals[..*count],
                &options,
            )?
        } else {
            RankCountMeasurement::Unavailable {
                rank_count: u32::try_from(*count).context("rank count exceeds u32")?,
                reason: CollectiveUnavailableReason::FewerDevicesThanRanks,
                available_cuda_devices: available,
            }
        });
    }

    let measured_point_to_point = point_to_point_bytes_per_second(&point_to_point_reference);
    let baseline = baseline(measured_point_to_point);
    if let Some(baseline) = baseline.as_ref() {
        baseline.validate()?;
    }
    let cost_model = review_cost_model(&rank_counts, measured_point_to_point, baseline)?;
    let report = CollectiveBenchmarkReport {
        schema_version: COLLECTIVE_BENCHMARK_SCHEMA_VERSION,
        measured_at_unix_ms: unix_millis()?,
        visible_cuda_devices: u32::try_from(visible).context("visible device count exceeds u32")?,
        cuda_ordinals: options
            .cuda_ordinals
            .iter()
            .map(|ordinal| u32::try_from(*ordinal).context("CUDA ordinal exceeds u32"))
            .collect::<Result<Vec<_>>>()?,
        fingerprints,
        warmup_iterations: u32::try_from(options.warmup_iterations)
            .context("warmup count exceeds u32")?,
        sample_iterations: u32::try_from(options.sample_iterations)
            .context("sample count exceeds u32")?,
        payload: CollectivePayload::current(),
        point_to_point_reference,
        rank_counts,
        cost_model,
    };
    report.validate()?;
    Ok(report)
}

#[cfg(feature = "cuda")]
fn visible_cuda_device_count() -> Result<usize> {
    let count = candle_core::cuda_backend::cudarc::driver::CudaContext::device_count()
        .context("failed to query visible CUDA device count")?;
    ensure!(count > 0, "CUDA reported no visible devices");
    usize::try_from(count).context("visible CUDA device count exceeds usize")
}

/// A barrier that also carries a failure flag. A rank that fails mid-collective
/// still reaches every remaining barrier, so its peers abort with it instead of
/// blocking forever on a step that will never complete.
#[cfg(any(feature = "cuda", test))]
struct FailableBarrier {
    parties: usize,
    state: Mutex<BarrierState>,
    signal: Condvar,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Default)]
struct BarrierState {
    arrived: usize,
    generation: u64,
    failed: bool,
    healthy: bool,
}

#[cfg(any(feature = "cuda", test))]
impl FailableBarrier {
    fn new(parties: usize) -> Self {
        Self {
            parties,
            state: Mutex::new(BarrierState {
                healthy: true,
                ..BarrierState::default()
            }),
            signal: Condvar::new(),
        }
    }

    /// Returns whether every party reached this barrier healthy.
    fn wait(&self, healthy: bool) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("collective barrier mutex poisoned");
        state.failed |= !healthy;
        state.arrived += 1;
        if state.arrived == self.parties {
            state.arrived = 0;
            state.healthy = !state.failed;
            state.failed = false;
            state.generation = state.generation.wrapping_add(1);
            self.signal.notify_all();
            return state.healthy;
        }
        let generation = state.generation;
        while state.generation == generation {
            state = self
                .signal
                .wait(state)
                .expect("collective barrier condvar poisoned");
        }
        state.healthy
    }
}

/// One rank's device-side state. `chunks` is the only mutable part, and every
/// write to it is separated from every peer's read of it by a ring barrier and
/// a device synchronization.
#[cfg(any(feature = "cuda", test))]
struct RankState {
    device: Device,
    /// The whole payload, never written. The concurrent neighbour copy reads
    /// this so that its transfer is one copy of exactly the block cut, which is
    /// what the point-to-point reference copies too.
    payload: Tensor,
    /// The rank's contribution, kept so every timed iteration starts from the
    /// same inputs and the reduced result stays checkable.
    pristine: Vec<Tensor>,
    chunks: Vec<RwLock<Tensor>>,
    chunk_elements: usize,
}

#[cfg(any(feature = "cuda", test))]
impl RankState {
    fn reset(&self) -> Result<()> {
        for (chunk, pristine) in self.chunks.iter().zip(&self.pristine) {
            *chunk.write().expect("chunk lock poisoned") = pristine.clone();
        }
        Ok(())
    }
}

/// Rank `r`'s contribution at global element `e` is `(e mod 5) + 1 + r`. Every
/// value and every partial sum the ring forms is a small integer, so BF16 holds
/// all of them exactly and the reduced buffer has one right answer.
#[cfg(any(feature = "cuda", test))]
fn contribution(element: u64, rank: usize) -> f32 {
    (element % COLLECTIVE_VALUE_MODULUS + 1 + rank as u64) as f32
}

#[cfg(any(feature = "cuda", test))]
fn reduced_expectation(element: u64, ranks: usize) -> f32 {
    let ranks = ranks as u64;
    ((element % COLLECTIVE_VALUE_MODULUS + 1) * ranks + ranks * (ranks - 1) / 2) as f32
}

/// `elements` is the whole payload; the real measurement always passes the H3
/// block cut, and a test passes something small enough to check by hand.
#[cfg(any(feature = "cuda", test))]
fn build_rank_state(
    device: &Device,
    rank: usize,
    ranks: usize,
    elements: usize,
) -> Result<RankState> {
    ensure!(
        ranks > 0 && elements > 0 && elements.is_multiple_of(ranks),
        "a ring of {ranks} ranks cannot split {elements} elements evenly"
    );
    let chunk_elements = elements / ranks;
    let payload = Tensor::zeros(elements, DType::BF16, device)
        .context("failed to allocate the collective payload")?;
    let mut pristine = Vec::with_capacity(ranks);
    for index in 0..ranks {
        let start = u64::try_from(index * chunk_elements).context("chunk offset exceeds u64")?;
        let values = (0..chunk_elements)
            .map(|offset| {
                half::bf16::from_f32(contribution(
                    start + u64::try_from(offset).expect("chunk offset exceeds u64"),
                    rank,
                ))
            })
            .collect::<Vec<_>>();
        pristine.push(
            Tensor::from_vec(values, chunk_elements, device)
                .context("failed to upload a collective chunk")?,
        );
    }
    device
        .synchronize()
        .context("failed to synchronize collective setup")?;
    let chunks = pristine
        .iter()
        .map(|chunk| RwLock::new(chunk.clone()))
        .collect();
    Ok(RankState {
        device: device.clone(),
        payload,
        pristine,
        chunks,
        chunk_elements,
    })
}

/// The chunk rank `r` receives from rank `r-1` at `round` of `phase`: the
/// scatter-reduce (phase 0) walks the chunk index backwards from `r-1`, and the
/// all-gather (phase 1) walks it backwards from `r`, one step behind. Every
/// index below is `mod count`, and the schedule is the whole correctness
/// argument for the ring, so it is a function that can be checked on its own.
#[cfg(any(feature = "cuda", test))]
fn ring_chunk(rank: usize, count: usize, phase: usize, round: usize) -> usize {
    debug_assert!(phase < 2 && round < count.saturating_sub(1));
    let offset = if phase == 0 { round + 1 } else { round };
    (rank + count - offset) % count
}

/// Run `operation` on every rank at once, `warmups + samples` times, and return
/// the wall time of each sampled round. The clock is the coordinating thread's:
/// a collective is finished when its slowest rank is, which is what the ranks'
/// own timers would each have to be reduced to anyway.
/// What one rank does in one round of a concurrent measurement, given every
/// rank's state, its own index, and the ring barrier its steps synchronize on.
#[cfg(any(feature = "cuda", test))]
type RankOperation<'a> = dyn Fn(&[RankState], usize, &FailableBarrier) -> Result<()> + Sync + 'a;

#[cfg(any(feature = "cuda", test))]
fn run_concurrent(
    ranks: &[RankState],
    options: &CollectiveBenchmarkOptions,
    reset_between_rounds: bool,
    operation: &RankOperation<'_>,
) -> Result<Vec<Duration>> {
    let count = ranks.len();
    let rounds = options
        .warmup_iterations
        .checked_add(options.sample_iterations)
        .context("collective round count overflow")?;
    let ready = FailableBarrier::new(count + 1);
    let done = FailableBarrier::new(count + 1);
    let step = FailableBarrier::new(count);
    let mut elapsed = Vec::with_capacity(options.sample_iterations);
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::with_capacity(count);
        for rank in 0..count {
            let (ready, done, step) = (&ready, &done, &step);
            workers.push(
                std::thread::Builder::new()
                    .name(format!("ff-collective-{rank}"))
                    .spawn_scoped(scope, move || -> Result<()> {
                        let mut outcome = Ok(());
                        for _ in 0..rounds {
                            if outcome.is_ok() && reset_between_rounds {
                                outcome = ranks[rank].reset();
                            }
                            ready.wait(outcome.is_ok());
                            if outcome.is_ok() {
                                outcome = operation(ranks, rank, step);
                            }
                            done.wait(outcome.is_ok());
                        }
                        outcome
                    })
                    .with_context(|| format!("failed to start collective rank {rank}"))?,
            );
        }
        for round in 0..rounds {
            ready.wait(true);
            let started = Instant::now();
            let healthy = done.wait(true);
            let round_elapsed = started.elapsed();
            if healthy && round >= options.warmup_iterations {
                ensure!(
                    round_elapsed.as_nanos() > 0,
                    "collective round elapsed below timer resolution"
                );
                elapsed.push(round_elapsed);
            }
        }
        for (rank, worker) in workers.into_iter().enumerate() {
            worker
                .join()
                .map_err(|_| anyhow!("collective rank {rank} panicked"))??;
        }
        Ok(())
    })?;
    ensure!(
        elapsed.len() == options.sample_iterations,
        "the collective produced {} of {} sampled rounds",
        elapsed.len(),
        options.sample_iterations
    );
    Ok(elapsed)
}

/// Pull `chunk` from the ring predecessor and either add it into this rank's
/// own copy of that chunk or replace it. The transfer is a peer copy issued on
/// the destination's stream, which is how `interconnect_benchmark` measures a
/// single pair too.
#[cfg(any(feature = "cuda", test))]
fn exchange_chunk(ranks: &[RankState], rank: usize, chunk: usize, reduce: bool) -> Result<()> {
    let count = ranks.len();
    let predecessor = (rank + count - 1) % count;
    let received = {
        let source = ranks[predecessor].chunks[chunk]
            .read()
            .expect("chunk lock poisoned");
        source
            .to_device(&ranks[rank].device)
            .context("peer chunk transfer failed")?
    };
    let replacement = if reduce {
        let own = ranks[rank].chunks[chunk]
            .read()
            .expect("chunk lock poisoned");
        (&*own + &received).context("ring reduction failed")?
    } else {
        received
    };
    *ranks[rank].chunks[chunk]
        .write()
        .expect("chunk lock poisoned") = replacement;
    ranks[rank]
        .device
        .synchronize()
        .context("failed to synchronize a ring step")
}

/// Every rank pulls the whole payload from its ring predecessor at once: the
/// point-to-point copy with every link busy, which is the comparison T0's gate
/// is written against.
#[cfg(any(feature = "cuda", test))]
fn neighbor_copy(ranks: &[RankState], rank: usize, _step: &FailableBarrier) -> Result<()> {
    let predecessor = (rank + ranks.len() - 1) % ranks.len();
    let received = ranks[predecessor]
        .payload
        .to_device(&ranks[rank].device)
        .context("concurrent neighbour copy failed")?;
    ranks[rank]
        .device
        .synchronize()
        .context("failed to synchronize a concurrent neighbour copy")?;
    drop(received);
    Ok(())
}

/// The ring: `N-1` scatter-reduce steps, then `N-1` all-gather steps. At step
/// `s` rank `r` receives chunk `r-s-1` (reduce) or `r-s` (gather) from rank
/// `r-1`, and sends the chunk its own successor is receiving. Within a step the
/// chunk a rank writes is never the chunk its successor is reading, so one
/// barrier per step is enough.
#[cfg(any(feature = "cuda", test))]
fn ring_all_reduce(
    ranks: &[RankState],
    rank: usize,
    step: &FailableBarrier,
    reduce: bool,
) -> Result<()> {
    let count = ranks.len();
    let mut failure: Option<anyhow::Error> = None;
    for phase in 0..2 {
        for round in 0..(count - 1) {
            let chunk = ring_chunk(rank, count, phase, round);
            if failure.is_none()
                && let Err(error) = exchange_chunk(ranks, rank, chunk, reduce && phase == 0)
            {
                failure = Some(error);
            }
            if !step.wait(failure.is_none()) && failure.is_none() {
                failure = Some(anyhow!(
                    "a peer rank failed during the ring collective; this rank aborted with it"
                ));
            }
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(any(feature = "cuda", test))]
fn verify_reduced(ranks: &[RankState]) -> Result<CollectiveVerification> {
    let count = ranks.len();
    let chunk_elements = ranks
        .first()
        .context("a ring has at least one rank")?
        .chunk_elements;
    let window = COLLECTIVE_VERIFY_WINDOW_ELEMENTS.min(chunk_elements);
    ensure!(window > 0, "collective verification window is empty");
    let mut windows = 0u32;
    for rank in ranks {
        for (index, chunk) in rank.chunks.iter().enumerate() {
            let chunk = chunk.read().expect("chunk lock poisoned");
            let base = u64::try_from(index * chunk_elements).context("chunk offset exceeds u64")?;
            for start in [0usize, chunk_elements - window] {
                let values = chunk
                    .narrow(0, start, window)
                    .and_then(|slice| slice.to_dtype(DType::F32))
                    .and_then(|slice| slice.to_vec1::<f32>())
                    .context("failed to read back a reduced chunk window")?;
                for (offset, value) in values.iter().enumerate() {
                    let element = base
                        + u64::try_from(start + offset).context("element index exceeds u64")?;
                    let expected = reduced_expectation(element, count);
                    ensure!(
                        *value == expected,
                        "ring all-reduce produced {value} at element {element}, expected {expected}"
                    );
                }
                windows += 1;
            }
        }
    }
    Ok(CollectiveVerification {
        window_elements: u32::try_from(window).context("window size exceeds u32")?,
        verified_windows: windows,
        verified_elements: u64::from(windows)
            * u64::try_from(window).context("window exceeds u64")?,
    })
}

#[cfg(feature = "cuda")]
fn measure_point_to_point(
    destination: &Device,
    source: &Device,
    options: &CollectiveBenchmarkOptions,
) -> Result<BandwidthSeries> {
    let elements = usize::try_from(COLLECTIVE_PAYLOAD_ELEMENTS).context("payload exceeds usize")?;
    let payload = Tensor::zeros(elements, DType::BF16, source)
        .context("failed to allocate the point-to-point payload")?;
    source
        .synchronize()
        .context("failed to synchronize the point-to-point source")?;
    let mut elapsed = Vec::with_capacity(options.sample_iterations);
    for round in 0..(options.warmup_iterations + options.sample_iterations) {
        let started = Instant::now();
        let received = payload
            .to_device(destination)
            .context("point-to-point peer copy failed")?;
        destination
            .synchronize()
            .context("failed to synchronize the point-to-point destination")?;
        let round_elapsed = started.elapsed();
        drop(received);
        if round >= options.warmup_iterations {
            ensure!(
                round_elapsed.as_nanos() > 0,
                "point-to-point round elapsed below timer resolution"
            );
            elapsed.push(round_elapsed);
        }
    }
    new_bandwidth_series(
        "point_to_point_peer_cuda_d2d",
        "host_monotonic_single_pair_peer_copy_plus_destination_synchronize",
        COLLECTIVE_PAYLOAD_BYTES,
        options.warmup_iterations,
        elapsed,
    )
}

#[cfg(feature = "cuda")]
fn measure_rank_count(
    devices: &[Device],
    ordinals: &[usize],
    options: &CollectiveBenchmarkOptions,
) -> Result<RankCountMeasurement> {
    let count = devices.len();
    let elements = usize::try_from(COLLECTIVE_PAYLOAD_ELEMENTS).context("payload exceeds usize")?;
    let ranks = devices
        .iter()
        .enumerate()
        .map(|(rank, device)| build_rank_state(device, rank, count, elements))
        .collect::<Result<Vec<_>>>()?;

    let neighbor = run_concurrent(&ranks, options, false, &neighbor_copy)?;
    let concurrent_neighbor_d2d = new_bandwidth_series(
        "concurrent_ring_neighbor_cuda_d2d",
        "host_monotonic_all_ranks_copy_the_payload_from_their_predecessor",
        COLLECTIVE_PAYLOAD_BYTES,
        options.warmup_iterations,
        neighbor,
    )?;

    let algorithmic = algorithmic_bytes_per_rank(count)?;
    let transfer_only = run_concurrent(&ranks, options, true, &|ranks, rank, step| {
        ring_all_reduce(ranks, rank, step, false)
    })?;
    let ring_all_reduce_transfer_only = new_bandwidth_series(
        "ring_all_reduce_transfer_only",
        "host_monotonic_ring_schedule_without_the_reduction",
        algorithmic,
        options.warmup_iterations,
        transfer_only,
    )?;

    let reduced = run_concurrent(&ranks, options, true, &|ranks, rank, step| {
        ring_all_reduce(ranks, rank, step, true)
    })?;
    let ring_all_reduce_series = new_bandwidth_series(
        "ring_all_reduce_bf16_sum",
        "host_monotonic_ring_scatter_reduce_and_all_gather_with_per_step_synchronization",
        algorithmic,
        options.warmup_iterations,
        reduced,
    )?;
    let verification = verify_reduced(&ranks)?;

    Ok(RankCountMeasurement::Measured(Box::new(MeasuredRing {
        rank_count: u32::try_from(count).context("rank count exceeds u32")?,
        cuda_ordinals: ordinals
            .iter()
            .map(|ordinal| u32::try_from(*ordinal).context("CUDA ordinal exceeds u32"))
            .collect::<Result<Vec<_>>>()?,
        chunk_elements: COLLECTIVE_PAYLOAD_ELEMENTS / count as u64,
        algorithmic_bytes_per_rank: algorithmic,
        concurrent_neighbor_d2d,
        ring_all_reduce_transfer_only,
        ring_all_reduce: ring_all_reduce_series,
        verification,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unavailable(rank_count: u32) -> RankCountMeasurement {
        RankCountMeasurement::Unavailable {
            rank_count,
            reason: CollectiveUnavailableReason::FewerDevicesThanRanks,
            available_cuda_devices: 1,
        }
    }

    #[test]
    fn the_documented_ring_byte_counts_are_reproduced_exactly() {
        assert_eq!(H3_COLLECTIVES_PER_EVALUATION, 104);
        let two = algorithmic_bytes_per_rank(2).unwrap() * H3_COLLECTIVES_PER_EVALUATION;
        let four = algorithmic_bytes_per_rank(4).unwrap() * H3_COLLECTIVES_PER_EVALUATION;
        assert_eq!(two, 42_168_741_888);
        assert_eq!(four, 63_253_112_832);
        let gib = 1024.0 * 1024.0 * 1024.0;
        assert!((two as f64 / gib - 39.27).abs() < 0.01, "{two}");
        assert!((four as f64 / gib - 58.91).abs() < 0.01, "{four}");
    }

    /// A host that measured nothing projects nothing. The rows still name the
    /// bytes each rank would move, because that is arithmetic over the payload
    /// rather than a claim about any machine.
    #[test]
    fn without_a_baseline_no_projection_is_made() {
        let review = review_cost_model(&[unavailable(2), unavailable(4)], None, None).unwrap();
        assert_eq!(review.verdict, CostModelVerdict::Unmeasured);
        assert_eq!(review.baseline, None);
        assert!(review.rows.iter().all(|row| row.derived.is_none()));
        assert!(
            review
                .rows
                .iter()
                .all(|row| row.bytes_per_rank_per_evaluation > 0)
        );
    }

    /// Supplied a baseline, the projection is exactly that baseline divided:
    /// compute by the rank count, communication by the charged rate. These are
    /// the numbers the retained RTX 4090 table was built from, now carried in
    /// as measurements instead of compiled in as constants.
    #[test]
    fn a_supplied_baseline_is_what_the_projection_divides() {
        let baseline = CostModelBaseline {
            single_device_evaluation_seconds: 39.63,
            charged_bytes_per_second: 17.1 * 1024.0 * 1024.0 * 1024.0,
        };
        baseline.validate().unwrap();
        let review =
            review_cost_model(&[unavailable(2), unavailable(4)], None, Some(baseline)).unwrap();
        assert_eq!(review.baseline, Some(baseline));
        let two = review.rows[0].derived.unwrap();
        assert!((two.communication_seconds - 2.30).abs() < 0.01);
        assert!((two.compute_seconds - 19.82).abs() < 0.01);
        assert!((two.total_seconds - 22.11).abs() < 0.01);
        assert!((two.speedup - 1.79).abs() < 0.01);
        let four = review.rows[1].derived.unwrap();
        assert!((four.communication_seconds - 3.44).abs() < 0.01);
        assert!((four.compute_seconds - 9.91).abs() < 0.01);
        assert!((four.total_seconds - 13.35).abs() < 0.01);
        assert!((four.speedup - 2.97).abs() < 0.01);

        for broken in [
            CostModelBaseline {
                single_device_evaluation_seconds: 0.0,
                ..baseline
            },
            CostModelBaseline {
                charged_bytes_per_second: f64::NAN,
                ..baseline
            },
        ] {
            assert!(broken.validate().is_err());
        }
    }

    #[test]
    fn a_rank_count_that_does_not_divide_the_payload_is_refused() {
        let options = CollectiveBenchmarkOptions {
            cuda_ordinals: vec![0, 1, 2, 3, 4],
            rank_counts: vec![5],
            warmup_iterations: 1,
            sample_iterations: 1,
        };
        let error = options.validate().unwrap_err().to_string();
        assert!(error.contains("does not divide"), "{error}");
    }

    #[test]
    fn options_refuse_a_shared_device_and_unbounded_sampling() {
        let base = CollectiveBenchmarkOptions {
            cuda_ordinals: vec![0, 1],
            rank_counts: vec![2],
            warmup_iterations: 1,
            sample_iterations: 5,
        };
        base.validate().unwrap();
        for (options, expected) in [
            (
                CollectiveBenchmarkOptions {
                    cuda_ordinals: vec![0, 0],
                    ..base.clone()
                },
                "twice",
            ),
            (
                CollectiveBenchmarkOptions {
                    rank_counts: vec![1],
                    ..base.clone()
                },
                "outside",
            ),
            (
                CollectiveBenchmarkOptions {
                    rank_counts: vec![4, 2],
                    ..base.clone()
                },
                "increasing",
            ),
            (
                CollectiveBenchmarkOptions {
                    sample_iterations: 0,
                    ..base.clone()
                },
                "sample count",
            ),
            (
                CollectiveBenchmarkOptions {
                    sample_iterations: MAX_LOCAL_IO_SAMPLE_ITERATIONS + 1,
                    ..base.clone()
                },
                "sample count",
            ),
            (
                CollectiveBenchmarkOptions {
                    warmup_iterations: MAX_LOCAL_IO_WARMUP_ITERATIONS + 1,
                    ..base.clone()
                },
                "warmup count",
            ),
        ] {
            let error = options.validate().unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn an_unmeasured_verdict_is_not_a_pass() {
        let review = review_cost_model(&[unavailable(2), unavailable(4)], None, None).unwrap();
        assert_eq!(review.verdict, CostModelVerdict::Unmeasured);
        assert!(review.rows.iter().all(|row| row.measured.is_none()));
    }

    #[test]
    fn a_collapsed_collective_refutes_the_table_and_a_slow_one_replaces_it() {
        const SINGLE_DEVICE_SECONDS: f64 = 39.63;
        let row = |seconds: f64, ranks: u32| {
            let bytes = algorithmic_bytes_per_rank(ranks as usize).unwrap() as f64
                * H3_COLLECTIVES_PER_EVALUATION as f64;
            let total = seconds + SINGLE_DEVICE_SECONDS / f64::from(ranks);
            MeasuredCostModelRow {
                collective_bytes_per_second: bytes / seconds,
                concurrent_fraction_of_point_to_point: None,
                communication_seconds: seconds,
                total_seconds: Some(total),
                speedup: Some(SINGLE_DEVICE_SECONDS / total),
                relative_total_error: Some(0.0),
            }
        };
        let mut stands = row(3.44, 4);
        stands.relative_total_error = Some(0.02);
        assert_eq!(row_verdict(&stands), CostModelVerdict::Stands);
        let mut replaced = row(13.8, 4);
        replaced.relative_total_error = Some(0.77);
        assert_eq!(row_verdict(&replaced), CostModelVerdict::Replaced);
        let mut refuted = row(40.0, 4);
        refuted.relative_total_error = Some(2.7);
        assert_eq!(row_verdict(&refuted), CostModelVerdict::Refuted);
        assert_eq!(
            worst_verdict(CostModelVerdict::Stands, CostModelVerdict::Refuted),
            CostModelVerdict::Refuted
        );
        assert_eq!(
            worst_verdict(CostModelVerdict::Unmeasured, CostModelVerdict::Stands),
            CostModelVerdict::Stands
        );
    }

    #[test]
    fn a_report_whose_cost_model_was_edited_is_rejected() {
        let report = CollectiveBenchmarkReport {
            schema_version: COLLECTIVE_BENCHMARK_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            visible_cuda_devices: 1,
            cuda_ordinals: vec![0],
            fingerprints: vec![cuda_fingerprint()],
            warmup_iterations: 1,
            sample_iterations: 5,
            payload: CollectivePayload::current(),
            point_to_point_reference: PointToPointReference::Unavailable {
                reason: CollectiveUnavailableReason::FewerDevicesThanRanks,
                available_cuda_devices: 1,
            },
            rank_counts: vec![unavailable(2), unavailable(4)],
            cost_model: review_cost_model(&[unavailable(2), unavailable(4)], None, None).unwrap(),
        };
        report.validate().unwrap();
        let json = serde_json::to_vec(&report).unwrap();
        assert_eq!(CollectiveBenchmarkReport::from_json(&json).unwrap(), report);

        // A report with no baseline cannot have a projection attached to it
        // after the fact, and one whose baseline is edited no longer matches
        // the rows that were derived from it.
        let mut tampered = report.clone();
        tampered.cost_model.baseline = Some(CostModelBaseline {
            single_device_evaluation_seconds: 39.63,
            charged_bytes_per_second: 17.1 * 1024.0 * 1024.0 * 1024.0,
        });
        assert!(tampered.validate().is_err());
        let mut tampered = report.clone();
        tampered.cost_model.verdict = CostModelVerdict::Stands;
        assert!(tampered.validate().is_err());
        let mut tampered = report.clone();
        tampered.payload.bytes += 1;
        assert!(tampered.validate().is_err());
        let mut tampered = report;
        tampered.rank_counts.reverse();
        assert!(tampered.validate().is_err());
    }

    /// The measured arm is an internally tagged newtype variant, which serde
    /// flattens into the tag's own object. A round trip proves the schema the
    /// report publishes is the one it reads back.
    #[test]
    fn a_measured_ring_round_trips_through_its_tagged_schema() {
        let ring = measured_ring(2);
        let report = CollectiveBenchmarkReport {
            schema_version: COLLECTIVE_BENCHMARK_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            visible_cuda_devices: 2,
            cuda_ordinals: vec![0, 1],
            fingerprints: vec![cuda_fingerprint(), cuda_fingerprint()],
            warmup_iterations: 1,
            sample_iterations: 5,
            payload: CollectivePayload::current(),
            point_to_point_reference: PointToPointReference::Measured {
                source_cuda_ordinal: 1,
                destination_cuda_ordinal: 0,
                series: series("point_to_point_peer_cuda_d2d", COLLECTIVE_PAYLOAD_BYTES, 25),
            },
            rank_counts: vec![ring.clone(), unavailable(4)],
            cost_model: review_cost_model(
                &[ring, unavailable(4)],
                Some(
                    series("point_to_point_peer_cuda_d2d", COLLECTIVE_PAYLOAD_BYTES, 25)
                        .statistics
                        .median_bytes_per_second,
                ),
                Some(CostModelBaseline {
                    single_device_evaluation_seconds: 39.63,
                    charged_bytes_per_second: series(
                        "point_to_point_peer_cuda_d2d",
                        COLLECTIVE_PAYLOAD_BYTES,
                        25,
                    )
                    .statistics
                    .median_bytes_per_second,
                }),
            )
            .unwrap(),
        };
        report.validate().unwrap();
        let json = serde_json::to_vec_pretty(&report).unwrap();
        assert_eq!(CollectiveBenchmarkReport::from_json(&json).unwrap(), report);
        let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(value["rank_counts"][0]["availability"], "measured");
        assert_eq!(value["rank_counts"][0]["rank_count"], 2);
        let row = &report.cost_model.rows[0];
        let measured = row.measured.as_ref().unwrap();
        assert!(measured.total_seconds.unwrap() > 0.0);
        assert!(
            (measured.concurrent_fraction_of_point_to_point.unwrap() - 1.0).abs() < 1e-9,
            "equal medians must read as an equal rate"
        );
    }

    #[test]
    fn a_measured_ring_that_lost_a_device_is_rejected() {
        let RankCountMeasurement::Measured(mut ring) = measured_ring(2) else {
            unreachable!("measured_ring builds a measured arm")
        };
        ring.cuda_ordinals = vec![0, 0];
        assert!(
            RankCountMeasurement::Measured(ring.clone())
                .validate(1, 5)
                .is_err()
        );
        ring.cuda_ordinals = vec![0, 1];
        ring.algorithmic_bytes_per_rank += 1;
        assert!(RankCountMeasurement::Measured(ring).validate(1, 5).is_err());
    }

    fn series(operation: &str, bytes: u64, nanos_per_gib: u64) -> BandwidthSeries {
        let elapsed = (0..5)
            .map(|index| {
                Duration::from_nanos(
                    bytes / (1024 * 1024) * nanos_per_gib + u64::from(index as u8) * 1_000,
                )
            })
            .collect();
        new_bandwidth_series(operation, "test_scope", bytes, 1, elapsed).unwrap()
    }

    fn measured_ring(rank_count: u32) -> RankCountMeasurement {
        let algorithmic = algorithmic_bytes_per_rank(rank_count as usize).unwrap();
        RankCountMeasurement::Measured(Box::new(MeasuredRing {
            rank_count,
            cuda_ordinals: (0..rank_count).collect(),
            chunk_elements: COLLECTIVE_PAYLOAD_ELEMENTS / u64::from(rank_count),
            algorithmic_bytes_per_rank: algorithmic,
            concurrent_neighbor_d2d: series(
                "concurrent_ring_neighbor_cuda_d2d",
                COLLECTIVE_PAYLOAD_BYTES,
                25,
            ),
            ring_all_reduce_transfer_only: series("ring_all_reduce_transfer_only", algorithmic, 30),
            ring_all_reduce: series("ring_all_reduce_bf16_sum", algorithmic, 35),
            verification: CollectiveVerification {
                window_elements: COLLECTIVE_VERIFY_WINDOW_ELEMENTS as u32,
                verified_windows: 2 * rank_count * rank_count,
                verified_elements: u64::from(2 * rank_count * rank_count)
                    * COLLECTIVE_VERIFY_WINDOW_ELEMENTS as u64,
            },
        }))
    }

    /// The ring's index schedule is its whole correctness argument, so it is
    /// checked directly: within one step no rank writes the chunk its
    /// successor is reading from it, which is why one barrier per step is
    /// enough, and after the scatter-reduce each rank owns exactly one fully
    /// reduced chunk.
    #[test]
    fn the_ring_schedule_never_writes_a_chunk_a_peer_is_reading() {
        for count in MIN_COLLECTIVE_RANKS..=MAX_COLLECTIVE_RANKS {
            for phase in 0..2 {
                for round in 0..(count - 1) {
                    let mut written = std::collections::BTreeSet::new();
                    for rank in 0..count {
                        let chunk = ring_chunk(rank, count, phase, round);
                        assert!(chunk < count);
                        let successor = (rank + 1) % count;
                        let successor_reads = ring_chunk(successor, count, phase, round);
                        assert_ne!(
                            chunk, successor_reads,
                            "count {count} phase {phase} round {round} rank {rank}"
                        );
                        assert!(written.insert((rank, chunk)));
                    }
                }
            }
            for rank in 0..count {
                let owned = ring_chunk(rank, count, 0, count - 2);
                assert_eq!(owned, (rank + 1) % count, "count {count} rank {rank}");
            }
        }
    }

    /// The whole ring — barriers, schedule, reduction, verification — run over
    /// small tensors on CPU devices. The transport differs from a peer copy;
    /// the arithmetic and the schedule are the same code the CUDA path runs.
    #[test]
    fn the_ring_reduces_every_rank_to_the_same_sum() {
        for count in [2usize, 3, 4] {
            let elements = count * 8;
            let ranks = (0..count)
                .map(|rank| build_rank_state(&Device::Cpu, rank, count, elements).unwrap())
                .collect::<Vec<_>>();
            let options = CollectiveBenchmarkOptions {
                cuda_ordinals: (0..count).collect(),
                rank_counts: vec![2],
                warmup_iterations: 1,
                sample_iterations: 2,
            };
            let elapsed = run_concurrent(&ranks, &options, true, &|ranks, rank, step| {
                ring_all_reduce(ranks, rank, step, true)
            })
            .unwrap();
            assert_eq!(elapsed.len(), 2);
            assert_eq!(
                run_concurrent(&ranks, &options, false, &neighbor_copy)
                    .unwrap()
                    .len(),
                2
            );
            let verification = verify_reduced(&ranks).unwrap();
            assert_eq!(verification.verified_windows as usize, 2 * count * count);
            for rank in &ranks {
                for (index, chunk) in rank.chunks.iter().enumerate() {
                    let values = chunk
                        .read()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    for (offset, value) in values.iter().enumerate() {
                        let element = (index * rank.chunk_elements + offset) as u64;
                        assert_eq!(*value, reduced_expectation(element, count));
                    }
                }
            }
        }
    }

    /// A rank that fails takes the ring down with it rather than leaving its
    /// peers parked on a step barrier that will never complete. The failure is
    /// a real one — a rank whose chunks are the wrong length, so the reduction
    /// itself refuses — rather than an injected abort, because what is being
    /// checked is that the ring's own error path reaches every barrier.
    #[test]
    fn a_failing_rank_aborts_the_ring_instead_of_hanging_it() {
        let count = 4;
        let ranks = (0..count)
            .map(|rank| {
                let elements = if rank == 2 { count * 4 } else { count * 8 };
                build_rank_state(&Device::Cpu, rank, count, elements).unwrap()
            })
            .collect::<Vec<_>>();
        let options = CollectiveBenchmarkOptions {
            cuda_ordinals: (0..count).collect(),
            rank_counts: vec![2],
            warmup_iterations: 0,
            sample_iterations: 1,
        };
        let error = run_concurrent(&ranks, &options, true, &|ranks, rank, step| {
            ring_all_reduce(ranks, rank, step, true)
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("ring reduction failed") || error.contains("a peer rank failed"),
            "{error}"
        );
    }

    /// The barrier the ring aborts through, on its own: every party learns of a
    /// failure reported by any one of them, and a healthy generation that
    /// follows is not tainted by it.
    #[test]
    fn the_failable_barrier_tells_every_party_about_one_partys_failure() {
        let parties = 4;
        let barrier = FailableBarrier::new(parties);
        let observed = std::thread::scope(|scope| {
            let workers = (0..parties)
                .map(|party| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let first = barrier.wait(party != 2);
                        let second = barrier.wait(true);
                        (first, second)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(
            observed.iter().all(|(first, _)| !first),
            "every party must see the failure: {observed:?}"
        );
        assert!(
            observed.iter().all(|(_, second)| *second),
            "the next generation must start healthy: {observed:?}"
        );
    }

    #[test]
    fn an_unavailable_arm_cannot_claim_devices_it_had() {
        let report = RankCountMeasurement::Unavailable {
            rank_count: 2,
            reason: CollectiveUnavailableReason::FewerDevicesThanRanks,
            available_cuda_devices: 2,
        };
        assert!(report.validate(1, 5).is_err());
    }

    /// The local I/O report fixture supplies a validated CUDA fingerprint for
    /// report tests.
    fn cuda_fingerprint() -> crate::runtime::probe::HardwareFingerprint {
        crate::interconnect_benchmark::LocalIoBenchmarkReport::from_json(include_bytes!(
            "testdata/local-io-report.json"
        ))
        .unwrap()
        .fingerprint
    }
}
