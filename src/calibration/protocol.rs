use super::validation::{
    calibration_schedule_timesteps, parse_calibration_device_selector,
    summarize_trial_measurements, validate_cache_stats, validate_cache_stats_pair,
    validate_calibration_policy, validate_device_selector, validate_fingerprint_backend,
    validate_process_wide_io_fault_measurement, validate_resource_snapshot,
    validate_snapshot_backend, validate_timings_against_request, validate_weight_access_stats_pair,
};
use super::{
    CALIBRATION_CANDIDATE_SCHEMA_VERSION, CALIBRATION_REPORT_SCHEMA_VERSION,
    CALIBRATION_TIMING_SCHEMA_VERSION, CALIBRATION_TRIAL_REQUEST_SCHEMA_VERSION,
    CALIBRATION_TRIAL_RESULT_SCHEMA_VERSION, CalibrationCacheCondition, CalibrationCandidateResult,
    CalibrationReport, CalibrationSchedule, CalibrationTimingProtocol, CalibrationTimingSummary,
    CalibrationTrialRequest, CalibrationTrialResult, CalibrationTrialTimings,
    EvaluationTimingRecorder, MAX_CALIBRATION_SIGMA_POINTS, T2vaLatentSummary, integer_median,
};
use crate::{
    h3::policy::ExecutionPolicy,
    runtime::identity::{BinaryIdentity, InputIdentity, WeakModelIdentity},
    runtime::probe::{HARDWARE_FINGERPRINT_SCHEMA_VERSION, HardwareFingerprint, ResourceSnapshot},
    runtime::telemetry::{ProcessWideIoFaultDelta, TraceMeasurement},
    runtime::weights::{CacheStats, WeightAccessStats},
};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};

impl CalibrationTimingProtocol {
    pub fn validate(self) -> Result<()> {
        anyhow::ensure!(
            self.warmup_prefix_evaluations > 0,
            "trajectory-warmed calibration requires at least one warmup-prefix evaluation"
        );
        anyhow::ensure!(
            self.measured_evaluations > 0,
            "calibration must measure at least one evaluation"
        );
        let total = self.total_evaluations()?;
        usize::try_from(total).context("calibration evaluation count exceeds usize")?;
        Ok(())
    }

    pub fn total_evaluations(self) -> Result<u64> {
        self.warmup_prefix_evaluations
            .checked_add(self.measured_evaluations)
            .context("calibration evaluation count overflow")
    }
}

impl CalibrationSchedule {
    pub fn new(
        sigma_points: u64,
        video_shift: f32,
        audio_shift: f32,
        first_step_index: u64,
        protocol: CalibrationTimingProtocol,
    ) -> Result<Self> {
        let schedule = Self {
            sigma_points,
            video_shift_bits: video_shift.to_bits(),
            audio_shift_bits: audio_shift.to_bits(),
            first_step_index,
        };
        schedule.validate(protocol)?;
        Ok(schedule)
    }

    pub fn validate(self, protocol: CalibrationTimingProtocol) -> Result<()> {
        protocol.validate()?;
        anyhow::ensure!(
            self.sigma_points >= 2,
            "calibration schedule requires at least two sigma points"
        );
        anyhow::ensure!(
            self.sigma_points <= MAX_CALIBRATION_SIGMA_POINTS,
            "calibration sigma-point count {} exceeds the protocol limit {}",
            self.sigma_points,
            MAX_CALIBRATION_SIGMA_POINTS
        );
        usize::try_from(self.sigma_points)
            .context("calibration sigma-point count exceeds usize")?;
        let video_shift = self.video_shift();
        let audio_shift = self.audio_shift();
        anyhow::ensure!(
            video_shift.is_finite() && video_shift > 0.0,
            "calibration video shift must be finite and positive"
        );
        anyhow::ensure!(
            audio_shift.is_finite() && audio_shift > 0.0,
            "calibration audio shift must be finite and positive"
        );
        let total_schedule_steps = self.total_schedule_steps()?;
        anyhow::ensure!(
            self.first_step_index < total_schedule_steps,
            "calibration first step {} is outside its {}-step schedule",
            self.first_step_index,
            total_schedule_steps
        );
        let end = self
            .first_step_index
            .checked_add(protocol.total_evaluations()?)
            .context("calibration schedule interval overflow")?;
        anyhow::ensure!(
            end <= total_schedule_steps,
            "calibration interval ends at step {end}, beyond its {}-step schedule",
            total_schedule_steps
        );
        Ok(())
    }

    pub fn total_schedule_steps(self) -> Result<u64> {
        let (video, _) = calibration_schedule_timesteps(self)?;
        u64::try_from(video.len()).context("calibration schedule length exceeds u64")
    }

    pub const fn video_shift(self) -> f32 {
        f32::from_bits(self.video_shift_bits)
    }

    pub const fn audio_shift(self) -> f32 {
        f32::from_bits(self.audio_shift_bits)
    }
}

impl CalibrationCacheCondition {
    fn validate(self, protocol: CalibrationTimingProtocol) -> Result<()> {
        match self {
            Self::TrajectoryWarmed => anyhow::ensure!(
                protocol.warmup_prefix_evaluations > 0,
                "trajectory-warmed calibration requires at least one warmup-prefix evaluation"
            ),
        }
        Ok(())
    }
}
/// Everything a calibration trial request is sealed from.
///
/// The four grid counters are same-typed `u64`s; naming them at the call site
/// keeps a transposed pair from sealing a request into the wrong grid slot.
pub struct CalibrationTrialRequestSpec {
    pub device_selector: String,
    pub binary_identity: BinaryIdentity,
    pub model_identity: WeakModelIdentity,
    pub input_identity: InputIdentity,
    pub policy: ExecutionPolicy,
    pub schedule: CalibrationSchedule,
    pub protocol: CalibrationTimingProtocol,
    pub cache_condition: CalibrationCacheCondition,
    pub policy_index: u64,
    pub policy_count: u64,
    pub trial_index: u64,
    pub trial_count: u64,
}

impl CalibrationTrialRequest {
    pub fn new(spec: CalibrationTrialRequestSpec) -> Result<Self> {
        let CalibrationTrialRequestSpec {
            device_selector,
            binary_identity,
            model_identity,
            input_identity,
            policy,
            schedule,
            protocol,
            cache_condition,
            policy_index,
            policy_count,
            trial_index,
            trial_count,
        } = spec;
        let invocation_order = trial_index
            .checked_mul(policy_count)
            .and_then(|value| value.checked_add(policy_index))
            .context("calibration invocation-order overflow")?;
        let request = Self {
            schema_version: CALIBRATION_TRIAL_REQUEST_SCHEMA_VERSION,
            policy_index,
            policy_count,
            trial_index,
            trial_count,
            invocation_order,
            device_selector,
            binary_identity,
            model_identity,
            input_identity,
            policy,
            schedule,
            protocol,
            cache_condition,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == CALIBRATION_TRIAL_REQUEST_SCHEMA_VERSION,
            "unsupported calibration-trial-request schema {}; this build supports schema {}",
            self.schema_version,
            CALIBRATION_TRIAL_REQUEST_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.policy_count > 0 && self.policy_index < self.policy_count,
            "calibration policy index {} is outside candidate count {}",
            self.policy_index,
            self.policy_count
        );
        anyhow::ensure!(
            self.trial_count > 0 && self.trial_index < self.trial_count,
            "calibration trial index {} is outside trial count {}",
            self.trial_index,
            self.trial_count
        );
        self.policy_count
            .checked_mul(self.trial_count)
            .context("calibration trial-grid size overflow")?;
        let expected_invocation_order = self
            .trial_index
            .checked_mul(self.policy_count)
            .and_then(|value| value.checked_add(self.policy_index))
            .context("calibration invocation-order overflow")?;
        anyhow::ensure!(
            self.invocation_order == expected_invocation_order,
            "calibration invocation order {} disagrees with trial/policy indices ({expected_invocation_order})",
            self.invocation_order
        );
        let (selector_backend, _) = parse_calibration_device_selector(&self.device_selector)?;
        self.binary_identity.validate()?;
        self.model_identity.validate()?;
        self.input_identity.validate()?;
        validate_calibration_policy(&self.policy)?;
        anyhow::ensure!(
            selector_backend == self.policy.execution_backend,
            "calibration device selector backend disagrees with the execution policy"
        );
        self.protocol.validate()?;
        self.cache_condition.validate(self.protocol)?;
        self.schedule.validate(self.protocol)?;
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let request: Self =
            serde_json::from_slice(bytes).context("invalid calibration-trial-request JSON")?;
        request.validate()?;
        Ok(request)
    }

    pub fn timing_recorder(&self) -> Result<EvaluationTimingRecorder> {
        self.validate()?;
        EvaluationTimingRecorder::new(self.protocol)
    }
}

impl CalibrationTimingSummary {
    pub fn from_measurements(values: &[u64]) -> Result<Self> {
        anyhow::ensure!(
            !values.is_empty(),
            "cannot summarize an empty calibration measurement set"
        );
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let median_ns = integer_median(&sorted);
        let mut deviations = sorted
            .iter()
            .map(|value| value.abs_diff(median_ns))
            .collect::<Vec<_>>();
        deviations.sort_unstable();
        Ok(Self {
            samples: u64::try_from(sorted.len()).context("calibration sample count exceeds u64")?,
            minimum_ns: sorted[0],
            median_ns,
            median_absolute_deviation_ns: integer_median(&deviations),
            maximum_ns: *sorted.last().context("calibration samples disappeared")?,
        })
    }
}

impl CalibrationTrialTimings {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == CALIBRATION_TIMING_SCHEMA_VERSION,
            "unsupported calibration-timing schema {}; this build supports schema {}",
            self.schema_version,
            CALIBRATION_TIMING_SCHEMA_VERSION
        );
        self.protocol.validate()?;
        anyhow::ensure!(
            self.preparation_elapsed_ns > 0,
            "calibration preparation timing must be non-zero"
        );
        anyhow::ensure!(
            u64::try_from(self.warmup.len()).context("warmup sample count exceeds u64")?
                == self.protocol.warmup_prefix_evaluations,
            "calibration timing record has {} warmup samples, expected {}",
            self.warmup.len(),
            self.protocol.warmup_prefix_evaluations
        );
        anyhow::ensure!(
            u64::try_from(self.measured.len()).context("measured sample count exceeds u64")?
                == self.protocol.measured_evaluations,
            "calibration timing record has {} measured samples, expected {}",
            self.measured.len(),
            self.protocol.measured_evaluations
        );
        let all = self.warmup.iter().chain(&self.measured);
        let mut previous: Option<u64> = None;
        let mut total_schedule_steps: Option<u64> = None;
        for timing in all {
            anyhow::ensure!(
                timing.elapsed_ns > 0,
                "calibration evaluation timing must be non-zero"
            );
            anyhow::ensure!(
                timing.video_timestep.is_finite() && timing.audio_timestep.is_finite(),
                "calibration timing record contains a non-finite timestep"
            );
            anyhow::ensure!(
                timing.total_schedule_steps > 0 && timing.step_index < timing.total_schedule_steps,
                "calibration step {} is outside its {}-step schedule",
                timing.step_index,
                timing.total_schedule_steps
            );
            if let Some(expected_total) = total_schedule_steps {
                anyhow::ensure!(
                    timing.total_schedule_steps == expected_total,
                    "calibration timing record mixes schedule lengths"
                );
            } else {
                total_schedule_steps = Some(timing.total_schedule_steps);
            }
            if let Some(previous) = previous {
                let expected = previous
                    .checked_add(1)
                    .context("calibration step-index sequence overflow")?;
                anyhow::ensure!(
                    timing.step_index == expected,
                    "calibration step indices are not consecutive"
                );
            }
            previous = Some(timing.step_index);
        }
        let measured_values = self
            .measured
            .iter()
            .map(|timing| timing.elapsed_ns)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            self.measured_summary == CalibrationTimingSummary::from_measurements(&measured_values)?,
            "calibration timing summary disagrees with its measured samples"
        );
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let record: Self =
            serde_json::from_slice(bytes).context("invalid calibration-timing JSON")?;
        record.validate()?;
        Ok(record)
    }
}

/// What a finished calibration child reports back about one trial.
///
/// The before/after pairs are same-typed; naming them at the call site keeps a
/// swapped pair from being sealed into the report as a measured delta.
pub struct CalibrationTrialObservations {
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

impl CalibrationTrialResult {
    pub fn new(
        request: CalibrationTrialRequest,
        observations: CalibrationTrialObservations,
    ) -> Result<Self> {
        let CalibrationTrialObservations {
            observed_binary_identity,
            observed_model_identity,
            observed_input_identity,
            hardware_fingerprint,
            resource_snapshot_before,
            resource_snapshot_after,
            cache_stats_before,
            cache_stats_after,
            weight_access_stats_before,
            weight_access_stats_after,
            process_wide_io_fault_delta,
            output,
            timings,
        } = observations;
        let result = Self {
            schema_version: CALIBRATION_TRIAL_RESULT_SCHEMA_VERSION,
            request,
            observed_binary_identity,
            observed_model_identity,
            observed_input_identity,
            hardware_fingerprint,
            resource_snapshot_before,
            resource_snapshot_after,
            cache_stats_before,
            cache_stats_after,
            weight_access_stats_before,
            weight_access_stats_after,
            process_wide_io_fault_delta,
            output,
            timings,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == CALIBRATION_TRIAL_RESULT_SCHEMA_VERSION,
            "unsupported calibration-trial-result schema {}; this build supports schema {}",
            self.schema_version,
            CALIBRATION_TRIAL_RESULT_SCHEMA_VERSION
        );
        self.request.validate()?;
        self.observed_binary_identity.validate()?;
        self.observed_model_identity.validate()?;
        self.observed_input_identity.validate()?;
        anyhow::ensure!(
            self.observed_binary_identity == self.request.binary_identity
                && self.observed_model_identity == self.request.model_identity
                && self.observed_input_identity == self.request.input_identity,
            "calibration child identities disagree with its request"
        );
        self.hardware_fingerprint.validate()?;
        anyhow::ensure!(
            self.hardware_fingerprint.schema_version == HARDWARE_FINGERPRINT_SCHEMA_VERSION,
            "calibration trials require current hardware-fingerprint schema {}",
            HARDWARE_FINGERPRINT_SCHEMA_VERSION
        );
        validate_fingerprint_backend(
            &self.hardware_fingerprint,
            &self.request.policy,
            &self.request.device_selector,
        )?;
        validate_resource_snapshot(&self.resource_snapshot_before)?;
        validate_resource_snapshot(&self.resource_snapshot_after)?;
        validate_snapshot_backend(
            "before",
            &self.resource_snapshot_before,
            &self.hardware_fingerprint,
        )?;
        validate_snapshot_backend(
            "after",
            &self.resource_snapshot_after,
            &self.hardware_fingerprint,
        )?;
        anyhow::ensure!(
            self.resource_snapshot_before.measured_at_unix_ms
                <= self.resource_snapshot_after.measured_at_unix_ms,
            "calibration after-snapshot predates its before-snapshot"
        );
        validate_cache_stats("before", &self.cache_stats_before)?;
        validate_cache_stats("after", &self.cache_stats_after)?;
        validate_cache_stats_pair(
            &self.cache_stats_before,
            &self.cache_stats_after,
            &self.request.policy,
        )?;
        validate_weight_access_stats_pair(
            &self.weight_access_stats_before,
            &self.weight_access_stats_after,
        )?;
        validate_process_wide_io_fault_measurement(&self.process_wide_io_fault_delta)?;
        self.output.validate()?;
        self.timings.validate()?;
        validate_timings_against_request(&self.timings, &self.request)?;
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let result: Self =
            serde_json::from_slice(bytes).context("invalid calibration-trial-result JSON")?;
        result.validate()?;
        Ok(result)
    }
}

impl CalibrationCandidateResult {
    pub fn aggregate(mut trials: Vec<CalibrationTrialResult>) -> Result<Self> {
        anyhow::ensure!(
            !trials.is_empty(),
            "cannot aggregate an empty calibration trial set"
        );
        for trial in &trials {
            trial.validate()?;
        }
        trials.sort_by_key(|trial| trial.request.trial_index);

        let first = trials
            .first()
            .context("calibration trial set disappeared while aggregating")?;
        let candidate = Self {
            schema_version: CALIBRATION_CANDIDATE_SCHEMA_VERSION,
            policy_index: first.request.policy_index,
            policy_count: first.request.policy_count,
            trial_count: first.request.trial_count,
            device_selector: first.request.device_selector.clone(),
            binary_identity: first.request.binary_identity.clone(),
            model_identity: first.request.model_identity.clone(),
            input_identity: first.request.input_identity.clone(),
            policy: first.request.policy.clone(),
            schedule: first.request.schedule,
            protocol: first.request.protocol,
            cache_condition: first.request.cache_condition,
            hardware_fingerprint: first.hardware_fingerprint.clone(),
            output: first.output.clone(),
            trial_output_statistics_all_equal: trials
                .iter()
                .all(|trial| trial.output == first.output),
            measured_summary: summarize_trial_measurements(&trials)?,
            trials,
        };
        candidate.validate()?;
        Ok(candidate)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == CALIBRATION_CANDIDATE_SCHEMA_VERSION,
            "unsupported calibration-candidate schema {}; this build supports schema {}",
            self.schema_version,
            CALIBRATION_CANDIDATE_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.policy_count > 0 && self.policy_index < self.policy_count,
            "calibration candidate policy index {} is outside candidate count {}",
            self.policy_index,
            self.policy_count
        );
        anyhow::ensure!(
            self.trial_count > 0,
            "calibration candidate trial_count must be non-zero"
        );
        anyhow::ensure!(
            u64::try_from(self.trials.len()).context("calibration trial count exceeds u64")?
                == self.trial_count,
            "calibration candidate contains {} trials, expected {}",
            self.trials.len(),
            self.trial_count
        );
        validate_device_selector(&self.device_selector)?;
        anyhow::ensure!(
            !self.trials.is_empty(),
            "calibration candidate must contain at least one trial"
        );
        self.binary_identity.validate()?;
        self.model_identity.validate()?;
        self.input_identity.validate()?;
        validate_calibration_policy(&self.policy)?;
        self.protocol.validate()?;
        self.cache_condition.validate(self.protocol)?;
        self.schedule.validate(self.protocol)?;
        self.hardware_fingerprint.validate()?;
        self.output.validate()?;

        for (expected_trial_index, trial) in self.trials.iter().enumerate() {
            trial.validate()?;
            let expected_trial_index = u64::try_from(expected_trial_index)
                .context("calibration trial count exceeds u64")?;
            anyhow::ensure!(
                trial.request.trial_index == expected_trial_index,
                "calibration candidate trial index {} is out of order; expected {expected_trial_index}",
                trial.request.trial_index
            );
            anyhow::ensure!(
                trial.request.policy_index == self.policy_index
                    && trial.request.policy_count == self.policy_count
                    && trial.request.trial_count == self.trial_count,
                "calibration candidate mixes policy positions"
            );
            anyhow::ensure!(
                trial.request.schedule == self.schedule
                    && trial.request.protocol == self.protocol
                    && trial.request.cache_condition == self.cache_condition,
                "calibration candidate mixes schedules, protocols, or cache conditions"
            );
            anyhow::ensure!(
                trial.request.binary_identity == self.binary_identity
                    && trial.request.model_identity == self.model_identity
                    && trial.request.input_identity == self.input_identity,
                "calibration candidate mixes binary, model, or input identities"
            );
            anyhow::ensure!(
                trial.request.device_selector == self.device_selector,
                "calibration candidate mixes device selectors"
            );
            anyhow::ensure!(
                trial.hardware_fingerprint == self.hardware_fingerprint,
                "calibration candidate mixes hardware fingerprints"
            );
        }
        anyhow::ensure!(
            self.output == self.trials[0].output,
            "candidate statistics differ from the first trial"
        );
        anyhow::ensure!(
            self.trial_output_statistics_all_equal
                == self.trials.iter().all(|trial| trial.output == self.output),
            "repeated-trial statistics flag is inconsistent"
        );
        anyhow::ensure!(
            self.measured_summary == summarize_trial_measurements(&self.trials)?,
            "calibration candidate summary disagrees with its flattened measured evaluations"
        );
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let candidate: Self =
            serde_json::from_slice(bytes).context("invalid calibration-candidate JSON")?;
        candidate.validate()?;
        Ok(candidate)
    }
}

impl CalibrationReport {
    pub fn from_trials(trials: Vec<CalibrationTrialResult>) -> Result<Self> {
        // Grouped by the policy itself, in its canonical form: trials of one
        // policy belong together, and the key is that policy rather than a
        // stand-in for it.
        let mut groups = BTreeMap::<Vec<u8>, Vec<CalibrationTrialResult>>::new();
        for trial in trials {
            trial.validate()?;
            groups
                .entry(trial.request.policy.canonical_json()?)
                .or_default()
                .push(trial);
        }
        let candidates = groups
            .into_values()
            .map(CalibrationCandidateResult::aggregate)
            .collect::<Result<Vec<_>>>()?;
        Self::new(candidates)
    }

    pub fn new(mut candidates: Vec<CalibrationCandidateResult>) -> Result<Self> {
        anyhow::ensure!(
            !candidates.is_empty(),
            "calibration report must contain at least one candidate"
        );
        for candidate in &candidates {
            candidate.validate()?;
        }
        candidates.sort_by(|left, right| {
            left.measured_summary
                .median_ns
                .cmp(&right.measured_summary.median_ns)
                .then_with(|| left.policy_index.cmp(&right.policy_index))
        });
        let first_trial = candidates
            .first()
            .and_then(|candidate| candidate.trials.first())
            .context("calibration report has no trial identity")?;
        let candidate_output_statistics_all_equal = candidates
            .iter()
            .flat_map(|candidate| &candidate.trials)
            .all(|trial| trial.output == candidates[0].output);
        let report = Self {
            schema_version: CALIBRATION_REPORT_SCHEMA_VERSION,
            device_selector: first_trial.request.device_selector.clone(),
            binary_identity: first_trial.request.binary_identity.clone(),
            model_identity: first_trial.request.model_identity.clone(),
            input_identity: first_trial.request.input_identity.clone(),
            hardware_fingerprint: first_trial.hardware_fingerprint.clone(),
            schedule: first_trial.request.schedule,
            protocol: first_trial.request.protocol,
            cache_condition: first_trial.request.cache_condition,
            candidates,
            candidate_output_statistics_all_equal,
            cacheable: false,
            winner_selected: false,
            selection: None,
        };
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == CALIBRATION_REPORT_SCHEMA_VERSION,
            "unsupported calibration-report schema {}; this build supports schema {}",
            self.schema_version,
            CALIBRATION_REPORT_SCHEMA_VERSION
        );
        validate_device_selector(&self.device_selector)?;
        anyhow::ensure!(
            !self.candidates.is_empty(),
            "calibration report must contain at least one candidate"
        );
        anyhow::ensure!(
            !self.cacheable,
            "calibration reports cannot authorize cache reuse"
        );
        anyhow::ensure!(
            !self.winner_selected && self.selection.is_none(),
            "calibration reports cannot select a winner"
        );
        self.binary_identity.validate()?;
        self.model_identity.validate()?;
        self.input_identity.validate()?;
        self.hardware_fingerprint.validate()?;
        self.protocol.validate()?;
        self.cache_condition.validate(self.protocol)?;
        self.schedule.validate(self.protocol)?;
        anyhow::ensure!(
            self.candidate_output_statistics_all_equal
                == self
                    .candidates
                    .iter()
                    .flat_map(|candidate| &candidate.trials)
                    .all(|trial| trial.output == self.candidates[0].output),
            "calibration output-statistics flag is inconsistent"
        );

        let first = self
            .candidates
            .first()
            .context("calibration candidate set disappeared while validating")?;
        let candidate_count = u64::try_from(self.candidates.len())
            .context("calibration candidate count exceeds u64")?;
        let trial_count = first.trial_count;
        let expected_trial_count = usize::try_from(trial_count)
            .context("calibration declared trial count exceeds usize")?;
        let expected_execution_count = candidate_count
            .checked_mul(trial_count)
            .context("calibration report trial-grid size overflow")?;
        anyhow::ensure!(
            u64::try_from(
                self.candidates
                    .iter()
                    .map(|candidate| candidate.trials.len())
                    .sum::<usize>(),
            )
            .context("calibration execution-order length exceeds u64")?
                == expected_execution_count,
            "calibration execution-order length disagrees with its declared trial grid"
        );
        let mut policy_indices = BTreeSet::new();
        let mut policies = BTreeSet::<Vec<u8>>::new();
        let precompute_adaln = first.policy.precompute_adaln;
        let mut previous_key: Option<(u64, u64)> = None;
        for candidate in &self.candidates {
            candidate.validate()?;
            anyhow::ensure!(
                candidate.policy_count == candidate_count,
                "calibration candidate records policy_count {}, expected {candidate_count}",
                candidate.policy_count
            );
            anyhow::ensure!(
                candidate.trial_count == trial_count
                    && candidate.trials.len() == expected_trial_count,
                "calibration candidates contain different trial counts"
            );
            anyhow::ensure!(
                policy_indices.insert(candidate.policy_index),
                "calibration report contains duplicate policy index {}",
                candidate.policy_index
            );
            anyhow::ensure!(
                policies.insert(candidate.policy.canonical_json()?),
                "calibration report contains duplicate policy at index {}",
                candidate.policy_index
            );
            anyhow::ensure!(
                candidate.policy.precompute_adaln == precompute_adaln,
                "calibration candidates must use the same AdaLN precomputation setting"
            );
            anyhow::ensure!(
                candidate.hardware_fingerprint == self.hardware_fingerprint
                    && candidate.hardware_fingerprint == first.hardware_fingerprint,
                "calibration report mixes hardware fingerprints"
            );
            anyhow::ensure!(
                candidate.binary_identity == self.binary_identity
                    && candidate.model_identity == self.model_identity
                    && candidate.input_identity == self.input_identity,
                "calibration report mixes binary, model, or input identities"
            );
            anyhow::ensure!(
                candidate.device_selector == self.device_selector,
                "calibration report mixes device selectors"
            );
            anyhow::ensure!(
                candidate.schedule == self.schedule
                    && candidate.protocol == self.protocol
                    && candidate.cache_condition == self.cache_condition
                    && candidate.schedule == first.schedule
                    && candidate.protocol == first.protocol
                    && candidate.cache_condition == first.cache_condition,
                "calibration report mixes schedules, protocols, or cache conditions"
            );
            let key = (candidate.measured_summary.median_ns, candidate.policy_index);
            if let Some(previous) = previous_key {
                anyhow::ensure!(
                    previous < key,
                    "calibration candidates are duplicated or not sorted by median and policy index"
                );
            }
            previous_key = Some(key);
        }
        let expected_policy_indices = (0..candidate_count).collect::<BTreeSet<_>>();
        anyhow::ensure!(
            policy_indices == expected_policy_indices,
            "calibration report policy indices are not contiguous from zero"
        );

        let mut invocations = self
            .candidates
            .iter()
            .flat_map(|candidate| &candidate.trials)
            .map(|trial| trial.request.invocation_order)
            .collect::<Vec<_>>();
        invocations.sort_unstable();
        for (expected_order, invocation_order) in invocations.iter().enumerate() {
            let expected_order = u64::try_from(expected_order)
                .context("calibration invocation count exceeds u64")?;
            anyhow::ensure!(
                *invocation_order == expected_order,
                "calibration invocation order {invocation_order} is out of sequence; expected {expected_order}"
            );
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let report: Self =
            serde_json::from_slice(bytes).context("invalid calibration-report JSON")?;
        report.validate()?;
        Ok(report)
    }

    pub const fn winner_selected(&self) -> bool {
        self.winner_selected
    }

    pub fn selection(&self) -> Option<&str> {
        self.selection.as_deref()
    }

    pub const fn cacheable(&self) -> bool {
        self.cacheable
    }
}
