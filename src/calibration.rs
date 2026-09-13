use crate::{
    h3::pipeline::{DenoiseObserver, DenoisePreparationEvent, DenoiseStepEvent},
    h3::policy::ExecutionPolicy,
    runtime::identity::{BinaryIdentity, InputIdentity, WeakModelIdentity},
    runtime::probe::{HardwareFingerprint, ResourceSnapshot},
    runtime::telemetry::{ProcessWideIoFaultDelta, TraceMeasurement},
    runtime::weights::{CacheStats, WeightAccessStats},
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

mod protocol;
mod summary;
mod validation;

pub use protocol::{CalibrationTrialObservations, CalibrationTrialRequestSpec};
pub use summary::{T2vaLatentSummary, TensorSummary};
#[cfg(test)]
use validation::calibration_schedule_timesteps;
pub use validation::validate_calibration_policy;

pub const CALIBRATION_TIMING_SCHEMA_VERSION: u32 = 1;
pub const CALIBRATION_TRIAL_REQUEST_SCHEMA_VERSION: u32 = 1;
pub const CALIBRATION_TRIAL_RESULT_SCHEMA_VERSION: u32 = 1;
pub const CALIBRATION_CANDIDATE_SCHEMA_VERSION: u32 = 2;
pub const CALIBRATION_REPORT_SCHEMA_VERSION: u32 = 2;
pub const MAX_CALIBRATION_SIGMA_POINTS: u64 = 10_000;

const MAX_DEVICE_SELECTOR_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTimingProtocol {
    pub warmup_prefix_evaluations: u64,
    pub measured_evaluations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSchedule {
    pub sigma_points: u64,
    pub video_shift_bits: u32,
    pub audio_shift_bits: u32,
    pub first_step_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationCacheCondition {
    TrajectoryWarmed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTrialRequest {
    pub schema_version: u32,
    pub policy_index: u64,
    pub policy_count: u64,
    pub trial_index: u64,
    pub trial_count: u64,
    pub invocation_order: u64,
    pub device_selector: String,
    pub binary_identity: BinaryIdentity,
    pub model_identity: WeakModelIdentity,
    pub input_identity: InputIdentity,
    pub policy: ExecutionPolicy,
    pub schedule: CalibrationSchedule,
    pub protocol: CalibrationTimingProtocol,
    pub cache_condition: CalibrationCacheCondition,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynchronizedEvaluationTiming {
    pub step_index: u64,
    pub total_schedule_steps: u64,
    pub video_timestep: f32,
    pub audio_timestep: f32,
    pub elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTimingSummary {
    pub samples: u64,
    pub minimum_ns: u64,
    pub median_ns: u64,
    pub median_absolute_deviation_ns: u64,
    pub maximum_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTrialTimings {
    pub schema_version: u32,
    pub protocol: CalibrationTimingProtocol,
    pub preparation_elapsed_ns: u64,
    pub warmup: Vec<SynchronizedEvaluationTiming>,
    pub measured: Vec<SynchronizedEvaluationTiming>,
    pub measured_summary: CalibrationTimingSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTrialResult {
    pub schema_version: u32,
    pub request: CalibrationTrialRequest,
    pub observed_binary_identity: BinaryIdentity,
    pub observed_model_identity: WeakModelIdentity,
    pub observed_input_identity: InputIdentity,
    pub hardware_fingerprint: HardwareFingerprint,
    pub resource_snapshot_before: ResourceSnapshot,
    pub resource_snapshot_after: ResourceSnapshot,
    pub cache_stats_before: CacheStats,
    pub cache_stats_after: CacheStats,
    pub weight_access_stats_before: WeightAccessStats,
    pub weight_access_stats_after: WeightAccessStats,
    pub process_wide_io_fault_delta: TraceMeasurement<ProcessWideIoFaultDelta>,
    pub output: T2vaLatentSummary,
    pub timings: CalibrationTrialTimings,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationCandidateResult {
    pub schema_version: u32,
    pub policy_index: u64,
    pub policy_count: u64,
    pub trial_count: u64,
    pub device_selector: String,
    pub binary_identity: BinaryIdentity,
    pub model_identity: WeakModelIdentity,
    pub input_identity: InputIdentity,
    pub policy: ExecutionPolicy,
    pub schedule: CalibrationSchedule,
    pub protocol: CalibrationTimingProtocol,
    pub cache_condition: CalibrationCacheCondition,
    pub hardware_fingerprint: HardwareFingerprint,
    pub output: T2vaLatentSummary,
    /// Whether repeated trials have equal statistics; not tensor equality.
    pub trial_output_statistics_all_equal: bool,
    pub trials: Vec<CalibrationTrialResult>,
    pub measured_summary: CalibrationTimingSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationReport {
    pub schema_version: u32,
    pub device_selector: String,
    pub binary_identity: BinaryIdentity,
    pub model_identity: WeakModelIdentity,
    pub input_identity: InputIdentity,
    pub hardware_fingerprint: HardwareFingerprint,
    pub schedule: CalibrationSchedule,
    pub protocol: CalibrationTimingProtocol,
    pub cache_condition: CalibrationCacheCondition,
    pub candidates: Vec<CalibrationCandidateResult>,
    /// Whether all trials share the same statistics; not tensor equality.
    pub candidate_output_statistics_all_equal: bool,
    cacheable: bool,
    winner_selected: bool,
    #[serde(deserialize_with = "crate::required_option")]
    selection: Option<String>,
}

pub struct EvaluationTimingRecorder {
    protocol: CalibrationTimingProtocol,
    preparation_elapsed_ns: Option<u64>,
    warmup: Vec<SynchronizedEvaluationTiming>,
    measured: Vec<SynchronizedEvaluationTiming>,
}

impl EvaluationTimingRecorder {
    pub fn new(protocol: CalibrationTimingProtocol) -> Result<Self> {
        protocol.validate()?;
        Ok(Self {
            protocol,
            preparation_elapsed_ns: None,
            warmup: Vec::new(),
            measured: Vec::new(),
        })
    }

    pub fn finish(self) -> Result<CalibrationTrialTimings> {
        let preparation_elapsed_ns = self
            .preparation_elapsed_ns
            .context("calibration preparation event was not observed")?;
        anyhow::ensure!(
            u64::try_from(self.warmup.len()).context("warmup sample count exceeds u64")?
                == self.protocol.warmup_prefix_evaluations,
            "calibration observed {} warmup evaluations, expected {}",
            self.warmup.len(),
            self.protocol.warmup_prefix_evaluations
        );
        anyhow::ensure!(
            u64::try_from(self.measured.len()).context("measured sample count exceeds u64")?
                == self.protocol.measured_evaluations,
            "calibration observed {} measured evaluations, expected {}",
            self.measured.len(),
            self.protocol.measured_evaluations
        );
        let measured_values = self
            .measured
            .iter()
            .map(|timing| timing.elapsed_ns)
            .collect::<Vec<_>>();
        let record = CalibrationTrialTimings {
            schema_version: CALIBRATION_TIMING_SCHEMA_VERSION,
            protocol: self.protocol,
            preparation_elapsed_ns,
            warmup: self.warmup,
            measured: self.measured,
            measured_summary: CalibrationTimingSummary::from_measurements(&measured_values)?,
        };
        record.validate()?;
        Ok(record)
    }
}

impl DenoiseObserver for EvaluationTimingRecorder {
    fn synchronize_device_timings(&self) -> bool {
        true
    }

    fn on_preparation_completed(&mut self, event: DenoisePreparationEvent) -> Result<()> {
        anyhow::ensure!(
            event.timing_synchronized,
            "calibration preparation timing was not device-synchronized"
        );
        anyhow::ensure!(
            self.preparation_elapsed_ns.is_none(),
            "calibration observed more than one preparation event"
        );
        anyhow::ensure!(
            u64::try_from(event.prepared_evaluations)
                .context("preparation evaluation count exceeds u64")?
                == self.protocol.total_evaluations()?,
            "calibration preparation covered {} evaluations, expected {}",
            event.prepared_evaluations,
            self.protocol.total_evaluations()?
        );
        self.preparation_elapsed_ns = Some(duration_ns(event.elapsed)?);
        Ok(())
    }

    fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()> {
        anyhow::ensure!(
            event.timing_synchronized,
            "calibration evaluation timing was not device-synchronized"
        );
        anyhow::ensure!(
            self.preparation_elapsed_ns.is_some(),
            "calibration observed an evaluation before preparation"
        );
        let observed = u64::try_from(
            self.warmup
                .len()
                .checked_add(self.measured.len())
                .context("calibration observation count overflow")?,
        )
        .context("calibration observation count exceeds u64")?;
        anyhow::ensure!(
            observed < self.protocol.total_evaluations()?,
            "calibration observed more evaluations than requested"
        );
        let timing = SynchronizedEvaluationTiming {
            step_index: u64::try_from(event.step_index).context("step index exceeds u64")?,
            total_schedule_steps: u64::try_from(event.total_steps)
                .context("total step count exceeds u64")?,
            video_timestep: event.video_timestep,
            audio_timestep: event.audio_timestep,
            elapsed_ns: duration_ns(event.step_elapsed)?,
        };
        if observed < self.protocol.warmup_prefix_evaluations {
            self.warmup.push(timing);
        } else {
            self.measured.push(timing);
        }
        Ok(())
    }
}

fn duration_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).context("calibration duration exceeds u64 nanoseconds")
}

fn integer_median(sorted: &[u64]) -> u64 {
    let upper = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[upper]
    } else {
        let lower_value = sorted[upper - 1];
        lower_value + (sorted[upper] - lower_value) / 2
    }
}

#[cfg(test)]
mod tests;
