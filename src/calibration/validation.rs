use super::{
    CalibrationSchedule, CalibrationTimingSummary, CalibrationTrialRequest, CalibrationTrialResult,
    CalibrationTrialTimings, MAX_CALIBRATION_SIGMA_POINTS, MAX_DEVICE_SELECTOR_BYTES,
};
use crate::{
    h3::policy::{
        AttentionBackendPolicy, ExecutionBackendPolicy, ExecutionPolicy, WeightSourcePolicy,
    },
    h3::scheduler::H3Scheduler,
    runtime::probe::{
        CgroupMemoryLimit, DeviceBackend, HardwareFingerprint, MemoryMeasurementScope,
        RESOURCE_SNAPSHOT_SCHEMA_VERSION, ResourceSnapshot,
    },
    runtime::telemetry::{ProcessWideIoFaultDelta, TraceMeasurement},
    runtime::weights::{CacheStats, WeightAccessStats},
};
use anyhow::{Context, Result};

pub fn validate_calibration_policy(policy: &ExecutionPolicy) -> Result<()> {
    policy.validate()?;
    anyhow::ensure!(
        policy.attention.backend == AttentionBackendPolicy::FullSoftmax,
        "calibration profile admits only full-softmax attention"
    );
    anyhow::ensure!(
        policy.weights.source == WeightSourcePolicy::Mmap,
        "calibration profile admits only mmap weight access"
    );
    anyhow::ensure!(
        policy.weights.cache_shards == 1 && policy.weights.cache_bytes.is_none(),
        "calibration profile requires a one-shard cache without a byte limit"
    );
    anyhow::ensure!(
        !policy.weights.device_cache.is_enabled(),
        "calibration profile does not model retained device weights"
    );
    Ok(())
}
pub(super) fn summarize_trial_measurements(
    trials: &[CalibrationTrialResult],
) -> Result<CalibrationTimingSummary> {
    let total_samples = trials.iter().try_fold(0usize, |total, trial| {
        total
            .checked_add(trial.timings.measured.len())
            .context("flattened calibration sample count overflow")
    })?;
    anyhow::ensure!(
        total_samples > 0,
        "cannot summarize calibration trials without measured evaluations"
    );
    let mut measurements = Vec::new();
    measurements
        .try_reserve_exact(total_samples)
        .map_err(|error| {
            anyhow::anyhow!("failed to reserve flattened calibration samples: {error}")
        })?;
    for trial in trials {
        measurements.extend(
            trial
                .timings
                .measured
                .iter()
                .map(|timing| timing.elapsed_ns),
        );
    }
    CalibrationTimingSummary::from_measurements(&measurements)
}

pub(super) fn validate_timings_against_request(
    timings: &CalibrationTrialTimings,
    request: &CalibrationTrialRequest,
) -> Result<()> {
    anyhow::ensure!(
        timings.protocol == request.protocol,
        "calibration trial timings disagree with the requested protocol"
    );
    let (video_timesteps, audio_timesteps) = calibration_schedule_timesteps(request.schedule)?;
    let total_schedule_steps =
        u64::try_from(video_timesteps.len()).context("calibration schedule length exceeds u64")?;
    for (offset, timing) in timings.warmup.iter().chain(&timings.measured).enumerate() {
        let offset = u64::try_from(offset).context("calibration timing offset exceeds u64")?;
        let expected_step = request
            .schedule
            .first_step_index
            .checked_add(offset)
            .context("calibration timing step-index overflow")?;
        anyhow::ensure!(
            timing.step_index == expected_step,
            "calibration timing step {} disagrees with requested step {expected_step}",
            timing.step_index
        );
        anyhow::ensure!(
            timing.total_schedule_steps == total_schedule_steps,
            "calibration timing reports {} schedule steps, expected {total_schedule_steps}",
            timing.total_schedule_steps
        );
        let expected_step_usize =
            usize::try_from(expected_step).context("calibration step index exceeds usize")?;
        let expected_video_timestep = video_timesteps[expected_step_usize];
        let expected_audio_timestep = audio_timesteps[expected_step_usize];
        anyhow::ensure!(
            timing.video_timestep.to_bits() == expected_video_timestep.to_bits()
                && timing.audio_timestep.to_bits() == expected_audio_timestep.to_bits(),
            "calibration timing timestep values disagree with the requested schedule at step {expected_step}"
        );
    }
    Ok(())
}

pub(super) fn calibration_schedule_timesteps(
    schedule: CalibrationSchedule,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let points = usize::try_from(schedule.sigma_points)
        .context("calibration sigma-point count exceeds usize")?;
    anyhow::ensure!(
        schedule.sigma_points <= MAX_CALIBRATION_SIGMA_POINTS,
        "calibration sigma-point count {} exceeds the protocol limit {}",
        schedule.sigma_points,
        MAX_CALIBRATION_SIGMA_POINTS
    );
    let mut video = H3Scheduler::new(schedule.video_shift())?;
    let video_timesteps = video.set_timesteps(points)?.to_vec();
    let mut audio = H3Scheduler::new(schedule.audio_shift())?;
    let audio_timesteps = audio.set_timesteps(points)?.to_vec();
    anyhow::ensure!(
        video_timesteps.len() == audio_timesteps.len(),
        "calibration video/audio schedules have different evaluation counts ({} and {})",
        video_timesteps.len(),
        audio_timesteps.len()
    );
    Ok((video_timesteps, audio_timesteps))
}

pub(super) fn validate_fingerprint_backend(
    fingerprint: &HardwareFingerprint,
    policy: &ExecutionPolicy,
    device_selector: &str,
) -> Result<()> {
    let matches = matches!(
        (policy.execution_backend, fingerprint.backend),
        (ExecutionBackendPolicy::Cpu, DeviceBackend::Cpu)
            | (ExecutionBackendPolicy::Cuda, DeviceBackend::Cuda)
            | (ExecutionBackendPolicy::Metal, DeviceBackend::Metal)
    );
    anyhow::ensure!(
        matches,
        "calibration hardware fingerprint backend does not match the execution policy"
    );
    let (selector_backend, _) = parse_calibration_device_selector(device_selector)?;
    anyhow::ensure!(
        selector_backend == policy.execution_backend,
        "calibration device selector backend does not match the execution policy"
    );
    Ok(())
}

pub(super) fn validate_resource_snapshot(snapshot: &ResourceSnapshot) -> Result<()> {
    anyhow::ensure!(
        snapshot.schema_version == RESOURCE_SNAPSHOT_SCHEMA_VERSION,
        "unsupported calibration resource-snapshot schema {}; this build supports schema {}",
        snapshot.schema_version,
        RESOURCE_SNAPSHOT_SCHEMA_VERSION
    );
    anyhow::ensure!(
        snapshot.measurement_scope.host_memory
            == snapshot
                .host_memory_available_bytes
                .map(|_| MemoryMeasurementScope::HostWide),
        "calibration host-memory value disagrees with its measurement scope"
    );
    let has_cgroup_measurement = snapshot.cgroup_v2_memory_limit.is_some()
        || snapshot.cgroup_v2_memory_current_bytes.is_some()
        || snapshot.cgroup_v2_memory_available_bytes.is_some();
    anyhow::ensure!(
        snapshot.measurement_scope.cgroup_memory
            == has_cgroup_measurement.then_some(MemoryMeasurementScope::ProcessCgroupV2),
        "calibration cgroup-memory values disagree with their measurement scope"
    );
    anyhow::ensure!(
        snapshot.measurement_scope.device_memory
            == snapshot
                .device_free_memory_bytes
                .map(|_| MemoryMeasurementScope::DeviceWide),
        "calibration device-memory value disagrees with its measurement scope"
    );
    if let Some(CgroupMemoryLimit::Unlimited) = snapshot.cgroup_v2_memory_limit {
        anyhow::ensure!(
            snapshot.cgroup_v2_memory_available_bytes.is_none(),
            "an unlimited cgroup cannot report finite available bytes"
        );
    }
    if snapshot.cgroup_v2_memory_available_bytes.is_some() {
        anyhow::ensure!(
            matches!(
                snapshot.cgroup_v2_memory_limit,
                Some(CgroupMemoryLimit::Bytes(_))
            ),
            "finite cgroup available bytes require a finite cgroup limit"
        );
    }
    if let (Some(CgroupMemoryLimit::Bytes(limit)), Some(available)) = (
        snapshot.cgroup_v2_memory_limit,
        snapshot.cgroup_v2_memory_available_bytes,
    ) {
        anyhow::ensure!(
            available <= limit,
            "cgroup available bytes exceed its finite limit"
        );
    }
    Ok(())
}

pub(super) fn validate_snapshot_backend(
    label: &str,
    snapshot: &ResourceSnapshot,
    fingerprint: &HardwareFingerprint,
) -> Result<()> {
    match fingerprint.backend {
        DeviceBackend::Cpu | DeviceBackend::Metal => anyhow::ensure!(
            snapshot.device_free_memory_bytes.is_none(),
            "calibration {label} snapshot cannot report device-free memory for the {:?} backend",
            fingerprint.backend
        ),
        DeviceBackend::Cuda => {
            if let Some(free_bytes) = snapshot.device_free_memory_bytes {
                let total_bytes = fingerprint.device_total_memory_bytes.context(
                    "CUDA calibration snapshot reports free memory without fingerprint total memory",
                )?;
                anyhow::ensure!(
                    free_bytes <= total_bytes,
                    "calibration {label} snapshot device-free memory exceeds fingerprint total memory"
                );
            }
        }
    }
    Ok(())
}

pub(super) fn validate_cache_stats(label: &str, stats: &CacheStats) -> Result<()> {
    anyhow::ensure!(
        stats.max_shards > 0,
        "calibration {label} cache max_shards must be non-zero"
    );
    if let Some(max_bytes) = stats.max_bytes {
        anyhow::ensure!(
            max_bytes > 0,
            "calibration {label} cache max_bytes must be non-zero when present"
        );
    }
    anyhow::ensure!(
        stats.resident_shards <= stats.max_shards,
        "calibration {label} cache resident shard count exceeds its capacity"
    );
    let resident_shards = u64::try_from(stats.resident_shards)
        .context("calibration resident shard count exceeds u64")?;
    anyhow::ensure!(
        resident_shards <= stats.header_parses,
        "calibration {label} payload residency exceeds cataloged shard count"
    );
    anyhow::ensure!(
        (stats.resident_shards == 0) == (stats.resident_bytes == 0),
        "calibration {label} cache resident shard and byte counts disagree"
    );
    let expected_over_budget = stats
        .max_bytes
        .is_some_and(|max_bytes| stats.resident_bytes > max_bytes);
    anyhow::ensure!(
        stats.over_budget == expected_over_budget,
        "calibration {label} cache over_budget flag disagrees with residency"
    );
    anyhow::ensure!(
        !stats.over_budget || stats.resident_shards == 1,
        "calibration {label} cache over-budget exception must contain exactly one shard"
    );
    Ok(())
}

pub(super) fn validate_cache_stats_pair(
    before: &CacheStats,
    after: &CacheStats,
    policy: &ExecutionPolicy,
) -> Result<()> {
    let policy_max_shards = usize::try_from(policy.weights.cache_shards)
        .context("calibration policy cache_shards exceeds usize")?;
    anyhow::ensure!(
        before.max_shards == policy_max_shards
            && after.max_shards == policy_max_shards
            && before.max_bytes == policy.weights.cache_bytes
            && after.max_bytes == policy.weights.cache_bytes,
        "calibration cache snapshots disagree with the execution policy"
    );
    if policy.weights.source == WeightSourcePolicy::Mmap {
        anyhow::ensure!(
            before.memory_source_reads == 0
                && before.memory_source_read_bytes == 0
                && after.memory_source_reads == 0
                && after.memory_source_read_bytes == 0,
            "mmap calibration cannot report memory-source shard reads"
        );
    }
    for (name, before, after) in [
        ("hits", before.hits, after.hits),
        ("misses", before.misses, after.misses),
        ("evictions", before.evictions, after.evictions),
        ("header_parses", before.header_parses, after.header_parses),
        (
            "memory_source_reads",
            before.memory_source_reads,
            after.memory_source_reads,
        ),
        (
            "memory_source_read_bytes",
            before.memory_source_read_bytes,
            after.memory_source_read_bytes,
        ),
    ] {
        anyhow::ensure!(
            after >= before,
            "calibration cache counter {name} moved backwards ({before} -> {after})"
        );
    }
    Ok(())
}

pub(super) fn validate_weight_access_stats_pair(
    before: &WeightAccessStats,
    after: &WeightAccessStats,
) -> Result<()> {
    for (name, before, after) in [
        (
            "device_tensor_materializations",
            before.device_tensor_materializations,
            after.device_tensor_materializations,
        ),
        (
            "device_row_materializations",
            before.device_row_materializations,
            after.device_row_materializations,
        ),
    ] {
        anyhow::ensure!(
            after >= before,
            "calibration weight-access counter {name} moved backwards ({before} -> {after})"
        );
    }
    Ok(())
}

pub(super) fn validate_process_wide_io_fault_measurement(
    measurement: &TraceMeasurement<ProcessWideIoFaultDelta>,
) -> Result<()> {
    match measurement {
        TraceMeasurement::Available { value } => value.validate()?,
        TraceMeasurement::Unavailable { reason } => anyhow::ensure!(
            !reason.trim().is_empty(),
            "unavailable calibration process-counter measurement requires a reason"
        ),
    }
    Ok(())
}

pub(super) fn validate_device_selector(value: &str) -> Result<()> {
    parse_calibration_device_selector(value).map(|_| ())
}

pub(super) fn parse_calibration_device_selector(
    value: &str,
) -> Result<(ExecutionBackendPolicy, Option<u64>)> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= MAX_DEVICE_SELECTOR_BYTES,
        "calibration device selector must contain 1..={MAX_DEVICE_SELECTOR_BYTES} bytes"
    );
    anyhow::ensure!(
        value.trim() == value && !value.chars().any(char::is_control),
        "calibration device selector must be trimmed and contain no control characters"
    );
    if value == "cpu" {
        return Ok((ExecutionBackendPolicy::Cpu, None));
    }
    for (prefix, backend) in [
        ("cuda:", ExecutionBackendPolicy::Cuda),
        ("metal:", ExecutionBackendPolicy::Metal),
    ] {
        if let Some(ordinal) = value.strip_prefix(prefix) {
            anyhow::ensure!(!ordinal.is_empty(), "calibration device ordinal is empty");
            let ordinal = ordinal
                .parse::<u64>()
                .with_context(|| format!("invalid calibration device selector {value:?}"))?;
            usize::try_from(ordinal).context("calibration device ordinal exceeds usize")?;
            anyhow::ensure!(
                value == format!("{prefix}{ordinal}"),
                "calibration device selector must use canonical decimal ordinal syntax"
            );
            return Ok((backend, Some(ordinal)));
        }
    }
    anyhow::bail!("unknown calibration device {value:?}; use cpu, cuda:N, or metal:N")
}
