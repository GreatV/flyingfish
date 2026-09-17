use super::{
    CALIBRATION_TIMING_SCHEMA_VERSION, CalibrationCacheCondition, CalibrationCandidateResult,
    CalibrationSchedule, CalibrationTimingProtocol, CalibrationTimingSummary,
    CalibrationTrialObservations, CalibrationTrialRequest, CalibrationTrialRequestSpec,
    CalibrationTrialResult, CalibrationTrialTimings, EvaluationTimingRecorder,
    SynchronizedEvaluationTiming, T2vaLatentSummary, calibration_schedule_timesteps,
};
use crate::{
    h3::model::TransformerChunking,
    h3::pipeline::{DenoiseObserver, DenoisePreparationEvent, DenoiseStepEvent},
    h3::policy::ExecutionPolicy,
    runtime::identity::{
        BinaryIdentity, CALIBRATION_IDENTITY_SCHEMA_VERSION, FileStamp, InputIdentity,
        ModelIdentityStrength, NamedFileStamp, WeakModelIdentity, WeakShardIdentity,
    },
    runtime::probe::{
        HardwareFingerprint, MemoryMeasurementScope, RESOURCE_SNAPSHOT_SCHEMA_VERSION,
        ResourceMeasurementScopes, ResourceSnapshot,
    },
    runtime::telemetry::TraceMeasurement,
    runtime::weights::{CachePolicy, CacheStats, WeightAccessStats, WeightSource},
};
use candle_core::{Device, Tensor};
use std::time::Duration;

/// One latent pair, and the same pair with a value moved. Trials of one policy
/// must agree on the first; the second is what disagreement looks like.
fn sample_output(shifted: bool) -> T2vaLatentSummary {
    let video = Tensor::new(&[[1.0f32, if shifted { 2.5 } else { 2.0 }]], &Device::Cpu).unwrap();
    let audio = Tensor::new(&[3.0f32], &Device::Cpu).unwrap();
    T2vaLatentSummary::collect(&video, &audio).unwrap()
}

fn preparation(prepared_evaluations: usize, synchronized: bool) -> DenoisePreparationEvent {
    DenoisePreparationEvent {
        prepared_evaluations,
        elapsed: Duration::from_nanos(77),
        timing_synchronized: synchronized,
    }
}

fn step(index: usize, elapsed_ns: u64, synchronized: bool) -> DenoiseStepEvent {
    DenoiseStepEvent {
        step_index: index,
        total_steps: 49,
        video_timestep: index as f32,
        audio_timestep: index as f32 / 2.,
        step_elapsed: Duration::from_nanos(elapsed_ns),
        total_elapsed: Duration::from_nanos(elapsed_ns),
        timing_synchronized: synchronized,
    }
}

fn protocol() -> CalibrationTimingProtocol {
    CalibrationTimingProtocol {
        warmup_prefix_evaluations: 1,
        measured_evaluations: 2,
    }
}

fn schedule() -> CalibrationSchedule {
    CalibrationSchedule::new(50, 12.0, 3.0, 0, protocol()).unwrap()
}

fn binary_identity() -> BinaryIdentity {
    BinaryIdentity {
        schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
        package_name: "flyingfish".to_owned(),
        package_version: "0.1.0-test".to_owned(),
        executable: FileStamp {
            bytes: 3,
            modified_ns: 1,
        },
        compiled_features: Vec::new(),
    }
}

fn model_identity() -> WeakModelIdentity {
    WeakModelIdentity {
        schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
        strength: ModelIdentityStrength::LocalMetadataManifest,
        cacheable: false,
        canonical_component_path: std::env::temp_dir()
            .join("flyingfish-calibration-test-transformer")
            .to_string_lossy()
            .into_owned(),
        config: NamedFileStamp {
            relative_path: "config.json".to_owned(),
            bytes: 2,
            modified_ns: 1,
        },
        index: Some(NamedFileStamp {
            relative_path: "model.safetensors.index.json".to_owned(),
            bytes: 2,
            modified_ns: 1,
        }),
        indexed_checkpoint_bytes: 1024,
        shards: vec![WeakShardIdentity {
            relative_path: "model-00001-of-00001.safetensors".to_owned(),
            file_bytes: 1024,
            modified_ns: 7,
        }],
    }
}

fn input_identity() -> InputIdentity {
    InputIdentity {
        schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
        bytes: 17,
        modified_ns: 1,
    }
}

fn policy(query_rows: usize) -> ExecutionPolicy {
    let mut chunking = TransformerChunking::default();
    chunking.attention.query_chunk_size = std::num::NonZeroUsize::new(query_rows).unwrap();
    ExecutionPolicy::from_runtime(
        &Device::Cpu,
        WeightSource::Mmap,
        CachePolicy::new(1),
        chunking,
        false,
        true,
    )
    .unwrap()
}

fn request(query_rows: usize, trial_index: u64, trial_count: u64) -> CalibrationTrialRequest {
    let policy_index = match query_rows {
        32 => 0,
        16 => 1,
        _ => panic!("test policy has no assigned index"),
    };
    CalibrationTrialRequest::new(CalibrationTrialRequestSpec {
        device_selector: "cpu".to_owned(),
        binary_identity: binary_identity(),
        model_identity: model_identity(),
        input_identity: input_identity(),
        policy: policy(query_rows),
        schedule: schedule(),
        protocol: protocol(),
        cache_condition: CalibrationCacheCondition::TrajectoryWarmed,
        policy_index,
        policy_count: 2,
        trial_index,
        trial_count,
    })
    .unwrap()
}

fn resource_snapshot(measured_at_unix_ms: u64) -> ResourceSnapshot {
    ResourceSnapshot {
        schema_version: RESOURCE_SNAPSHOT_SCHEMA_VERSION,
        measured_at_unix_ms,
        host_memory_available_bytes: None,
        cgroup_v2_memory_limit: None,
        cgroup_v2_memory_current_bytes: None,
        cgroup_v2_memory_available_bytes: None,
        device_free_memory_bytes: None,
        host_device_memory_is_unified: None,
        device_topology_probe_failed: false,
        host_memory_total_bytes: None,
        device_total_memory_bytes: None,
        measurement_scope: ResourceMeasurementScopes {
            host_memory: None,
            cgroup_memory: None,
            device_memory: None,
        },
    }
}

fn cache_stats() -> CacheStats {
    CacheStats {
        max_shards: 1,
        max_bytes: None,
        ..CacheStats::default()
    }
}

fn timings(measured_elapsed: [u64; 2]) -> CalibrationTrialTimings {
    let schedule = schedule();
    let (video_timesteps, audio_timesteps) = calibration_schedule_timesteps(schedule).unwrap();
    let make = |step_index: u64, elapsed_ns| SynchronizedEvaluationTiming {
        step_index,
        total_schedule_steps: schedule.total_schedule_steps().unwrap(),
        video_timestep: video_timesteps[step_index as usize],
        audio_timestep: audio_timesteps[step_index as usize],
        elapsed_ns,
    };
    let measured = vec![make(1, measured_elapsed[0]), make(2, measured_elapsed[1])];
    CalibrationTrialTimings {
        schema_version: CALIBRATION_TIMING_SCHEMA_VERSION,
        protocol: protocol(),
        preparation_elapsed_ns: 90,
        warmup: vec![make(0, 900)],
        measured_summary: CalibrationTimingSummary::from_measurements(&measured_elapsed).unwrap(),
        measured,
    }
}

fn trial(
    query_rows: usize,
    trial_index: u64,
    trial_count: u64,
    measured_elapsed: [u64; 2],
    output: T2vaLatentSummary,
) -> CalibrationTrialResult {
    let request = request(query_rows, trial_index, trial_count);
    CalibrationTrialResult::new(
        request,
        CalibrationTrialObservations {
            observed_binary_identity: binary_identity(),
            observed_model_identity: model_identity(),
            observed_input_identity: input_identity(),
            hardware_fingerprint: HardwareFingerprint::collect(&Device::Cpu),
            resource_snapshot_before: resource_snapshot(1),
            resource_snapshot_after: resource_snapshot(2),
            cache_stats_before: cache_stats(),
            cache_stats_after: cache_stats(),
            weight_access_stats_before: WeightAccessStats::default(),
            weight_access_stats_after: WeightAccessStats::default(),
            process_wide_io_fault_delta: TraceMeasurement::unavailable(
                "test process counters are intentionally unavailable",
            ),
            output,
            timings: timings(measured_elapsed),
        },
    )
    .unwrap()
}

#[test]
fn protocol_rejects_empty_measurements_and_overflow() {
    assert!(
        CalibrationTimingProtocol {
            warmup_prefix_evaluations: 0,
            measured_evaluations: 1,
        }
        .validate()
        .is_err()
    );
    assert!(
        CalibrationTimingProtocol {
            warmup_prefix_evaluations: 1,
            measured_evaluations: 0,
        }
        .validate()
        .is_err()
    );
    assert!(
        CalibrationTimingProtocol {
            warmup_prefix_evaluations: u64::MAX,
            measured_evaluations: 1,
        }
        .validate()
        .is_err()
    );
}

#[test]
fn recorder_excludes_warmup_and_summarizes_synchronized_measurements() {
    let protocol = CalibrationTimingProtocol {
        warmup_prefix_evaluations: 1,
        measured_evaluations: 3,
    };
    let mut recorder = EvaluationTimingRecorder::new(protocol).unwrap();
    assert!(recorder.synchronize_device_timings());
    recorder
        .on_preparation_completed(preparation(4, true))
        .unwrap();
    for (index, elapsed) in [900, 100, 300, 200].into_iter().enumerate() {
        recorder
            .on_step_completed(step(index, elapsed, true))
            .unwrap();
    }
    let result = recorder.finish().unwrap();
    assert_eq!(result.schema_version, CALIBRATION_TIMING_SCHEMA_VERSION);
    assert_eq!(result.preparation_elapsed_ns, 77);
    assert_eq!(result.warmup[0].elapsed_ns, 900);
    assert_eq!(
        result
            .measured
            .iter()
            .map(|timing| timing.elapsed_ns)
            .collect::<Vec<_>>(),
        vec![100, 300, 200]
    );
    assert_eq!(
        CalibrationTrialTimings::from_json(&serde_json::to_vec(&result).unwrap()).unwrap(),
        result
    );
    assert_eq!(
        result.measured_summary,
        CalibrationTimingSummary {
            samples: 3,
            minimum_ns: 100,
            median_ns: 200,
            median_absolute_deviation_ns: 100,
            maximum_ns: 300,
        }
    );
}

#[test]
fn median_uses_an_overflow_safe_midpoint() {
    let summary = CalibrationTimingSummary::from_measurements(&[u64::MAX, u64::MAX - 2]).unwrap();
    assert_eq!(summary.median_ns, u64::MAX - 1);
    assert_eq!(summary.median_absolute_deviation_ns, 1);
}

#[test]
fn recorder_rejects_unsynchronized_or_incomplete_events() {
    let protocol = CalibrationTimingProtocol {
        warmup_prefix_evaluations: 1,
        measured_evaluations: 1,
    };
    let mut recorder = EvaluationTimingRecorder::new(protocol).unwrap();
    assert!(
        recorder
            .on_preparation_completed(preparation(2, false))
            .is_err()
    );

    let mut recorder = EvaluationTimingRecorder::new(protocol).unwrap();
    recorder
        .on_preparation_completed(preparation(2, true))
        .unwrap();
    assert!(recorder.on_step_completed(step(0, 1, false)).is_err());
    assert!(recorder.finish().is_err());
}

#[test]
fn serialized_record_rejects_schema_count_order_and_summary_tampering() {
    let protocol = CalibrationTimingProtocol {
        warmup_prefix_evaluations: 1,
        measured_evaluations: 2,
    };
    let mut recorder = EvaluationTimingRecorder::new(protocol).unwrap();
    recorder
        .on_preparation_completed(preparation(3, true))
        .unwrap();
    recorder.on_step_completed(step(4, 10, true)).unwrap();
    recorder.on_step_completed(step(5, 20, true)).unwrap();
    recorder.on_step_completed(step(6, 30, true)).unwrap();
    let record = recorder.finish().unwrap();

    let mut json = serde_json::to_value(&record).unwrap();
    json["schema_version"] = serde_json::json!(2);
    assert!(CalibrationTrialTimings::from_json(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&record).unwrap();
    json["measured"][1]["step_index"] = serde_json::json!(7);
    assert!(CalibrationTrialTimings::from_json(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&record).unwrap();
    json["measured_summary"]["median_ns"] = serde_json::json!(999);
    assert!(CalibrationTrialTimings::from_json(&serde_json::to_vec(&json).unwrap()).is_err());
}

#[test]
fn schedule_rejects_nonfinite_out_of_range_and_overflowing_intervals() {
    assert!(CalibrationSchedule::new(50, f32::NAN, 3.0, 0, protocol()).is_err());
    assert!(CalibrationSchedule::new(3, 12.0, 3.0, 1, protocol()).is_err());
    let overflowing = CalibrationSchedule {
        sigma_points: u64::MAX,
        video_shift_bits: 12.0f32.to_bits(),
        audio_shift_bits: 3.0f32.to_bits(),
        first_step_index: u64::MAX - 1,
    };
    assert!(overflowing.validate(protocol()).is_err());
}

#[test]
fn trial_result_round_trips_and_rejects_identity_counter_and_timing_tampering() {
    let result = trial(32, 0, 1, [100, 300], sample_output(false));
    assert_eq!(
        CalibrationTrialResult::from_json(&serde_json::to_vec(&result).unwrap()).unwrap(),
        result
    );

    let mut tampered = result.clone();
    tampered.observed_input_identity.bytes += 1;
    assert!(tampered.validate().is_err());

    let mut tampered = result.clone();
    tampered.cache_stats_before.hits = 1;
    assert!(tampered.validate().is_err());

    let mut tampered = result.clone();
    tampered.timings.measured[0].video_timestep = 0.25;
    tampered.timings.measured_summary = CalibrationTimingSummary::from_measurements(
        &tampered
            .timings
            .measured
            .iter()
            .map(|timing| timing.elapsed_ns)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(tampered.validate().is_err());

    let mut tampered = result.clone();
    tampered
        .resource_snapshot_before
        .cgroup_v2_memory_available_bytes = Some(1);
    tampered
        .resource_snapshot_before
        .measurement_scope
        .cgroup_memory = Some(MemoryMeasurementScope::ProcessCgroupV2);
    assert!(tampered.validate().is_err());

    let mut tampered = result.clone();
    tampered.resource_snapshot_after.device_free_memory_bytes = Some(1);
    tampered
        .resource_snapshot_after
        .measurement_scope
        .device_memory = Some(MemoryMeasurementScope::DeviceWide);
    assert!(tampered.validate().is_err());

    let mut json = serde_json::to_value(&result).unwrap();
    json["unexpected"] = serde_json::json!(0);
    assert!(CalibrationTrialResult::from_json(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&result).unwrap();
    json["resource_snapshot_before"]["unexpected"] = serde_json::json!(0);
    assert!(CalibrationTrialResult::from_json(&serde_json::to_vec(&json).unwrap()).is_err());
}

#[test]
fn candidate_flattens_measurements_and_reports_repeat_statistics() {
    let candidate = CalibrationCandidateResult::aggregate(vec![
        trial(32, 1, 2, [700, 900], sample_output(false)),
        trial(32, 0, 2, [100, 300], sample_output(false)),
    ])
    .unwrap();
    assert_eq!(
        candidate
            .trials
            .iter()
            .map(|trial| trial.request.trial_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        candidate.measured_summary,
        CalibrationTimingSummary {
            samples: 4,
            minimum_ns: 100,
            median_ns: 500,
            median_absolute_deviation_ns: 300,
            maximum_ns: 900,
        }
    );
    assert_eq!(
        CalibrationCandidateResult::from_json(&serde_json::to_vec(&candidate).unwrap()).unwrap(),
        candidate
    );

    let mut missing_tail = candidate.clone();
    missing_tail.trials.pop();
    assert!(missing_tail.validate().is_err());

    let differing = CalibrationCandidateResult::aggregate(vec![
        trial(32, 0, 2, [100, 300], sample_output(false)),
        trial(32, 1, 2, [700, 900], sample_output(true)),
    ])
    .unwrap();
    assert!(!differing.trial_output_statistics_all_equal);
    assert!(
        CalibrationCandidateResult::aggregate(vec![
            trial(32, 0, 3, [100, 300], sample_output(false)),
            trial(32, 2, 3, [700, 900], sample_output(false)),
        ])
        .is_err()
    );
}

#[test]
fn latent_summary_is_shape_and_modality_bound_and_rejects_nonfinite_values() {
    let video = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
    let audio = Tensor::new(&[3.0f32], &Device::Cpu).unwrap();
    let summary = T2vaLatentSummary::collect(&video, &audio).unwrap();
    summary.validate().unwrap();
    assert_eq!(summary.video.dims, vec![1, 2]);
    assert_eq!(summary.video.elements, 2);
    assert_eq!(f64::from_bits(summary.video.sum_bits), 3.0);
    assert_eq!(f32::from_bits(summary.video.minimum_bits), 1.0);
    assert_eq!(f32::from_bits(summary.video.maximum_bits), 2.0);
    assert_eq!(summary.first_statistics_difference(&summary), None);

    // The same values in another shape are another output, and so is the same
    // pair of tensors with the modalities exchanged.
    let reshaped = video.reshape((2, 1)).unwrap();
    assert_eq!(
        summary
            .first_statistics_difference(&T2vaLatentSummary::collect(&reshaped, &audio).unwrap()),
        Some("video")
    );
    assert_eq!(
        summary.first_statistics_difference(&T2vaLatentSummary::collect(&audio, &video).unwrap()),
        Some("video")
    );

    let nonfinite = Tensor::new(&[f32::NAN], &Device::Cpu).unwrap();
    assert!(T2vaLatentSummary::collect(&video, &nonfinite).is_err());
}

#[test]
fn permuted_latents_do_not_get_a_tensor_equality_claim() {
    let values = (0..32).map(|i| i as f32).collect::<Vec<_>>();
    let mut permuted = values.clone();
    permuted.swap(12, 13);
    let a = Tensor::from_vec(values, (1, 32), &Device::Cpu).unwrap();
    let b = Tensor::from_vec(permuted, (1, 32), &Device::Cpu).unwrap();
    let audio = Tensor::new(&[0f32], &Device::Cpu).unwrap();
    assert_eq!(
        (&a - &b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
        1.0
    );
    let a_stats = T2vaLatentSummary::collect(&a, &audio).unwrap();
    let b_stats = T2vaLatentSummary::collect(&b, &audio).unwrap();
    let report = super::CalibrationReport::from_trials(vec![
        trial(32, 0, 2, [100, 300], a_stats.clone()),
        trial(32, 1, 2, [100, 300], a_stats),
        trial(16, 0, 2, [100, 300], b_stats.clone()),
        trial(16, 1, 2, [100, 300], b_stats),
    ])
    .unwrap();
    // The diagnostics agree, but the report makes no tensor-equality claim.
    assert!(report.candidate_output_statistics_all_equal);
    let json = serde_json::to_value(&report).unwrap();
    assert!(json.get("candidate_outputs_all_equal").is_none());
    let mut old = json;
    old["schema_version"] = serde_json::json!(1);
    assert!(super::CalibrationReport::from_json(&serde_json::to_vec(&old).unwrap()).is_err());
}

#[test]
fn summary_preserves_statistics_across_chunk_boundaries() {
    let values = (0..65_540)
        .map(|i| (i % 1024) as f32 * 0.25)
        .collect::<Vec<_>>();
    let tensor = Tensor::from_vec(values.clone(), (2, 32_770), &Device::Cpu).unwrap();
    let result = T2vaLatentSummary::collect(&tensor, &tensor).unwrap().video;
    assert_eq!(
        f64::from_bits(result.sum_bits),
        values.iter().map(|&v| f64::from(v)).sum::<f64>()
    );
    assert_eq!(
        f64::from_bits(result.sum_of_squares_bits),
        values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
    );
    assert_eq!(
        result.leading_bits,
        values[..8].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        result.trailing_bits,
        values[values.len() - 8..]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}
