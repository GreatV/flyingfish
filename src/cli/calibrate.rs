use super::*;
use flyingfish::{
    calibration::{
        CalibrationCacheCondition, CalibrationReport, CalibrationSchedule,
        CalibrationTimingProtocol, CalibrationTrialObservations, CalibrationTrialRequest,
        CalibrationTrialRequestSpec, CalibrationTrialResult, T2vaLatentSummary,
        validate_calibration_policy,
    },
    h3::policy::ExecutionBackendPolicy,
    runtime::artifact::ArtifactStaging,
    runtime::identity::{InputIdentity, WeakModelIdentity},
    runtime::probe::{HardwareFingerprint, ResourceSnapshot},
    runtime::telemetry::{process_wide_io_fault_delta, process_wide_io_fault_sample},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Write},
    process::{Command as ProcessCommand, Stdio},
};

const TRIAL_WORKER_REQUEST_SCHEMA_VERSION: u32 = 1;
const TRIAL_WORKER_RESPONSE_SCHEMA_VERSION: u32 = 1;
const CLI_CALIBRATION_REPORT_SCHEMA_VERSION: u32 = 1;
const MAX_TRIAL_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_TRIAL_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
const MAX_CHILD_ERROR_BYTES: usize = 16 * 1024;
const MAX_CALIBRATION_INPUT_BYTES: usize = 512 * 1024 * 1024;
const MAX_CALIBRATION_POLICIES: usize = 32;
const MAX_CALIBRATION_TRIALS: u64 = 100;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialWorkerRequest {
    schema_version: u32,
    model: PathBuf,
    component: PathBuf,
    inputs: PathBuf,
    policy_path: PathBuf,
    calibration: CalibrationTrialRequest,
}

impl TrialWorkerRequest {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == TRIAL_WORKER_REQUEST_SCHEMA_VERSION,
            "unsupported calibration trial-worker request schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.model.is_absolute() && self.inputs.is_absolute() && self.policy_path.is_absolute(),
            "calibration worker paths must be absolute"
        );
        anyhow::ensure!(
            !self.component.as_os_str().is_empty()
                && !self.component.is_absolute()
                && self
                    .component
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
            "calibration component must be a non-empty safe relative path"
        );
        self.calibration.validate()?;
        parse_calibration_device_selector(&self.calibration.device_selector)?;
        validate_selector_policy(&self.calibration.device_selector, &self.calibration.policy)?;
        anyhow::ensure!(
            self.calibration.schedule.first_step_index == 0,
            "public calibration accepts only step-zero initial inputs"
        );
        validate_candidate_policy(&self.calibration.policy)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialWorkerResponse {
    schema_version: u32,
    request: TrialWorkerRequest,
    result: CalibrationTrialResult,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestedPolicySource {
    policy_index: u64,
    path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CliCalibrationReport {
    schema_version: u32,
    device: String,
    model: PathBuf,
    component: PathBuf,
    inputs: PathBuf,
    requested_policies: Vec<RequestedPolicySource>,
    trials_per_policy: u64,
    calibration: CalibrationReport,
}

impl CliCalibrationReport {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == CLI_CALIBRATION_REPORT_SCHEMA_VERSION,
            "unsupported CLI calibration-report schema {}",
            self.schema_version
        );
        parse_calibration_device_selector(&self.device)?;
        anyhow::ensure!(
            self.model.is_absolute() && self.inputs.is_absolute(),
            "CLI calibration report paths must be absolute"
        );
        anyhow::ensure!(
            !self.component.as_os_str().is_empty()
                && !self.component.is_absolute()
                && self
                    .component
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
            "CLI calibration component must be a non-empty safe relative path"
        );
        anyhow::ensure!(
            self.trials_per_policy > 0,
            "calibration report has zero trials"
        );
        self.calibration.validate()?;
        anyhow::ensure!(
            self.calibration.device_selector == self.device,
            "CLI device selector disagrees with the calibration report"
        );
        anyhow::ensure!(
            !self.calibration.cacheable() && !self.calibration.winner_selected(),
            "CLI calibration schema 1 is report-only"
        );
        anyhow::ensure!(
            self.requested_policies.len() == self.calibration.candidates.len(),
            "requested-policy count disagrees with calibrated candidates"
        );
        let mut requested_indices = BTreeSet::new();
        for (expected_index, source) in self.requested_policies.iter().enumerate() {
            let expected_index =
                u64::try_from(expected_index).context("requested-policy count exceeds u64")?;
            anyhow::ensure!(
                source.policy_index == expected_index,
                "requested policy index {} is out of order; expected {expected_index}",
                source.policy_index
            );
            anyhow::ensure!(source.path.is_absolute(), "policy path must be absolute");
            self.calibration
                .candidates
                .iter()
                .find(|candidate| candidate.policy_index == source.policy_index)
                .context("requested policy index is absent from calibrated candidates")?;
            anyhow::ensure!(
                requested_indices.insert(source.policy_index),
                "duplicate requested policy index {}",
                source.policy_index
            );
        }
        let candidate_indices = self
            .calibration
            .candidates
            .iter()
            .map(|candidate| candidate.policy_index)
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            requested_indices == candidate_indices,
            "requested-policy indices disagree with calibrated candidates"
        );
        for candidate in &self.calibration.candidates {
            anyhow::ensure!(
                u64::try_from(candidate.trials.len())
                    .context("candidate trial count exceeds u64")?
                    == self.trials_per_policy,
                "candidate {} has {} trials, expected {}",
                candidate.policy_index,
                candidate.trials.len(),
                self.trials_per_policy
            );
        }
        Ok(())
    }
}

struct CandidatePolicy {
    path: PathBuf,
    policy: ExecutionPolicy,
}

pub(super) fn run_calibrate_t2va(command: H3Command) -> Result<()> {
    let H3Command::CalibrateT2va {
        model,
        component,
        inputs,
        policy,
        output,
        device,
        warmup_prefix_evaluations,
        measured_evaluations,
        trials,
        sigma_points,
        video_shift,
        audio_shift,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for calibrate-t2va");
    };

    parse_calibration_device_selector(&device)?;
    let model = std::fs::canonicalize(&model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let component_dir = resolve_component(&model, &component)?;
    let inputs = canonical_regular_file(&inputs, "calibration inputs")?;
    ensure_file_within_limit(&inputs, "calibration inputs", MAX_CALIBRATION_INPUT_BYTES)?;
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "calibration output")?;
    let report_staging = ArtifactStaging::new(&output).with_context(|| {
        format!(
            "failed to stage new calibration report {}",
            output.display()
        )
    })?;

    let protocol = CalibrationTimingProtocol {
        warmup_prefix_evaluations: u64::try_from(warmup_prefix_evaluations.get())
            .context("warmup-prefix count exceeds u64")?,
        measured_evaluations: u64::try_from(measured_evaluations.get())
            .context("measured evaluation count exceeds u64")?,
    };
    protocol.validate()?;
    let schedule = CalibrationSchedule::new(
        u64::try_from(sigma_points.get()).context("sigma-point count exceeds u64")?,
        video_shift,
        audio_shift,
        0,
        protocol,
    )?;
    let trials = u64::try_from(trials.get()).context("trial count exceeds u64")?;
    anyhow::ensure!(
        trials <= MAX_CALIBRATION_TRIALS,
        "calibration trial count {trials} exceeds the limit {MAX_CALIBRATION_TRIALS}"
    );

    anyhow::ensure!(
        policy.len() <= MAX_CALIBRATION_POLICIES,
        "calibration policy count {} exceeds the finalist limit {MAX_CALIBRATION_POLICIES}",
        policy.len()
    );
    let policies = load_candidate_policies(&policy)?;
    validate_policy_family(&policies)?;
    let policy_count =
        u64::try_from(policies.len()).context("calibration policy count exceeds u64")?;

    let current_exe = std::env::current_exe().context("failed to locate current ff executable")?;
    let binary_identity = flyingfish::collect_binary_identity(&current_exe)?;
    let model_identity = WeakModelIdentity::collect(&component_dir)?;
    let input_identity = InputIdentity::collect(&inputs)?;

    let result_capacity = policies
        .len()
        .checked_mul(usize::try_from(trials).context("trial count exceeds usize")?)
        .context("calibration result count overflow")?;
    let mut results = Vec::new();
    results
        .try_reserve_exact(result_capacity)
        .map_err(|error| anyhow::anyhow!("failed to reserve calibration results: {error}"))?;
    let mut expected_fingerprint: Option<HardwareFingerprint> = None;

    for trial_index in 0..trials {
        for (policy_index, candidate) in policies.iter().enumerate() {
            let policy_index = u64::try_from(policy_index).context("policy index exceeds u64")?;
            let calibration = CalibrationTrialRequest::new(CalibrationTrialRequestSpec {
                device_selector: device.clone(),
                binary_identity: binary_identity.clone(),
                model_identity: model_identity.clone(),
                input_identity: input_identity.clone(),
                policy: candidate.policy.clone(),
                schedule,
                protocol,
                cache_condition: CalibrationCacheCondition::TrajectoryWarmed,
                policy_index,
                policy_count,
                trial_index,
                trial_count: trials,
            })?;
            let request = TrialWorkerRequest {
                schema_version: TRIAL_WORKER_REQUEST_SCHEMA_VERSION,
                model: model.clone(),
                component: component.clone(),
                inputs: inputs.clone(),
                policy_path: candidate.path.clone(),
                calibration,
            };
            request.validate()?;
            let result = run_trial_child(&current_exe, &request)?;
            if let Some(expected) = expected_fingerprint.as_ref() {
                anyhow::ensure!(
                    result.hardware_fingerprint == *expected,
                    "calibration child hardware fingerprint changed between trials"
                );
            } else {
                expected_fingerprint = Some(result.hardware_fingerprint.clone());
            }
            results.push(result);
        }
    }

    let calibration = CalibrationReport::from_trials(results)?;
    let requested_policies = policies
        .iter()
        .enumerate()
        .map(|(policy_index, candidate)| {
            Ok(RequestedPolicySource {
                policy_index: u64::try_from(policy_index).context("policy index exceeds u64")?,
                path: candidate.path.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let report = CliCalibrationReport {
        schema_version: CLI_CALIBRATION_REPORT_SCHEMA_VERSION,
        device,
        model,
        component,
        inputs,
        requested_policies,
        trials_per_policy: trials,
        calibration,
    };
    report.validate()?;
    let json =
        serde_json::to_vec_pretty(&report).context("failed to serialize calibration report")?;
    publish_staged_bytes(report_staging, &json)
        .with_context(|| format!("failed to write calibration report {}", output.display()))?;
    println!("saved calibration report to {}", output.display());
    Ok(())
}

pub(super) fn run_calibrate_t2va_trial() -> Result<()> {
    let request = read_trial_request()?;
    let result = execute_trial(&request)?;
    let response = TrialWorkerResponse {
        schema_version: TRIAL_WORKER_RESPONSE_SCHEMA_VERSION,
        request,
        result,
    };
    let json = serde_json::to_vec(&response).context("failed to serialize trial response")?;
    anyhow::ensure!(
        json.len() <= MAX_TRIAL_RESPONSE_BYTES,
        "calibration trial response exceeds {} bytes",
        MAX_TRIAL_RESPONSE_BYTES
    );
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    stdout
        .write_all(&json)
        .context("failed to write calibration trial response")?;
    stdout
        .write_all(b"\n")
        .context("failed to terminate calibration trial response")?;
    stdout
        .flush()
        .context("failed to flush calibration trial response")?;
    Ok(())
}

fn execute_trial(request: &TrialWorkerRequest) -> Result<CalibrationTrialResult> {
    request.validate()?;
    let current_exe = std::env::current_exe().context("failed to locate trial executable")?;
    let observed_binary_identity = flyingfish::collect_binary_identity(&current_exe)?;
    let component_dir = resolve_component(&request.model, &request.component)?;
    let observed_model_identity = WeakModelIdentity::collect(&component_dir)?;
    let input_bytes = read_bounded_file(
        &request.inputs,
        "calibration inputs",
        MAX_CALIBRATION_INPUT_BYTES,
    )?;
    let observed_input_identity = InputIdentity::collect(&request.inputs)?;
    anyhow::ensure!(
        observed_binary_identity == request.calibration.binary_identity
            && observed_model_identity == request.calibration.model_identity
            && observed_input_identity == request.calibration.input_identity,
        "calibration child recomputed identities disagree with its request"
    );

    let observed_policy = ExecutionPolicy::load(&request.policy_path)?;
    anyhow::ensure!(
        observed_policy == request.calibration.policy,
        "calibration policy file changed after the supervisor request"
    );
    validate_candidate_policy(&observed_policy)?;

    let mut values = safetensors::load_buffer(&input_bytes, &Device::Cpu)
        .context("failed to decode the identity-checked calibration input bytes")?;
    drop(input_bytes);
    match H3ConditioningProvenance::take_artifact_tensors(&mut values)? {
        Some(H3ConditioningProvenance::FlyingfishQwen(_)) => {}
        Some(H3ConditioningProvenance::ExternallyValidated(_)) => {
            bail!("calibration does not accept external official-fixture prompt provenance")
        }
        None => bail!("calibration inputs are missing Flyingfish Qwen provenance"),
    }
    anyhow::ensure!(
        take_t2va_checkpoint_metadata(&mut values)
            .context("calibration requires initial inputs")?
            .is_none(),
        "calibration requires initial inputs, not a checkpoint"
    );
    let prompt_embeddings = take_input(&mut values, "prompt_embeddings")?;
    let text_token_tags = take_input(&mut values, "text_token_tags")?
        .to_vec1::<u32>()
        .context("text_token_tags must be a U32 vector")?;
    let video_latents = take_input(&mut values, "video_latents")?;
    let audio_latents = take_input(&mut values, "audio_latents")?;
    anyhow::ensure!(
        values.is_empty(),
        "calibration inputs contain unexpected tensors: {}",
        sorted_tensor_names(&values).join(", ")
    );

    let device = parse_device(&request.calibration.device_selector)?;
    validate_executable_policy(&observed_policy, &device)?;
    let prompt_embeddings = prompt_embeddings.to_device(&device)?;
    let video_latents = video_latents.to_device(&device)?;
    let audio_latents = audio_latents.to_device(&device)?;
    let transformer_options = build_transformer_options(device.clone(), &observed_policy)?;
    let transformer = StreamedTransformer::open(&component_dir, transformer_options)?;

    let hardware_fingerprint = HardwareFingerprint::collect(&device);
    hardware_fingerprint.validate()?;
    let resource_snapshot_before = ResourceSnapshot::capture(Some(&device));
    let cache_stats_before = transformer.cache_stats();
    let weight_access_stats_before = transformer.access_stats();
    let process_before = process_wide_io_fault_sample();
    let mut recorder = request.calibration.timing_recorder()?;
    let max_steps = usize::try_from(request.calibration.protocol.total_evaluations()?)
        .context("calibration evaluation count exceeds usize")?;
    let sigma_points = usize::try_from(request.calibration.schedule.sigma_points)
        .context("calibration sigma-point count exceeds usize")?;
    let output = denoise_t2va_with_options_and_observer(
        &transformer,
        &prompt_embeddings,
        &text_token_tags,
        &video_latents,
        &audio_latents,
        T2vaSchedule {
            sigma_points,
            video_shift: request.calibration.schedule.video_shift(),
            audio_shift: request.calibration.schedule.audio_shift(),
        },
        T2vaExecutionOptions {
            precompute_adaln: observed_policy.precompute_adaln,
            start_step: 0,
            max_steps: Some(max_steps),
        },
        &mut recorder,
    )?;
    anyhow::ensure!(
        output.completed_steps == max_steps,
        "calibration child completed {} evaluations, expected {max_steps}",
        output.completed_steps
    );
    device.synchronize()?;
    let process_after = process_wide_io_fault_sample();
    let process_wide_io_fault_delta = process_wide_io_fault_delta(&process_before, &process_after);
    let resource_snapshot_after = ResourceSnapshot::capture(Some(&device));
    let cache_stats_after = transformer.cache_stats();
    let weight_access_stats_after = transformer.access_stats();
    let timings = recorder.finish()?;
    let output = T2vaLatentSummary::collect(&output.video, &output.audio)?;
    let model_identity_after = WeakModelIdentity::collect(&component_dir)?;
    anyhow::ensure!(
        model_identity_after == observed_model_identity
            && model_identity_after == request.calibration.model_identity,
        "calibration model identity changed while the child was running"
    );

    CalibrationTrialResult::new(
        request.calibration.clone(),
        CalibrationTrialObservations {
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
        },
    )
}

fn run_trial_child(
    current_exe: &Path,
    request: &TrialWorkerRequest,
) -> Result<CalibrationTrialResult> {
    request.validate()?;
    let request_json =
        serde_json::to_vec(request).context("failed to serialize trial-worker request")?;
    anyhow::ensure!(
        request_json.len() <= MAX_TRIAL_REQUEST_BYTES,
        "calibration trial request exceeds {} bytes",
        MAX_TRIAL_REQUEST_BYTES
    );

    let mut child = ProcessCommand::new(current_exe)
        .arg("__calibrate-t2va-trial")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn isolated calibration trial")?;
    let write_error = match child.stdin.take() {
        Some(mut stdin) => stdin
            .write_all(&request_json)
            .err()
            .or_else(|| stdin.flush().err()),
        None => Some(std::io::Error::other(
            "spawned calibration child has no piped stdin",
        )),
    };
    let output = child
        .wait_with_output()
        .context("failed to wait for isolated calibration trial")?;
    if !output.status.success() {
        bail!(
            "calibration trial {} failed with {}: {}",
            request.calibration.invocation_order,
            output.status,
            bounded_child_error(&output.stderr)
        );
    }
    if let Some(error) = write_error {
        return Err(error).context("failed to send calibration trial request");
    }
    anyhow::ensure!(
        output.stdout.len() <= MAX_TRIAL_RESPONSE_BYTES,
        "calibration trial response exceeds {} bytes",
        MAX_TRIAL_RESPONSE_BYTES
    );
    let response: TrialWorkerResponse = serde_json::from_slice(&output.stdout)
        .context("calibration trial stdout is not exactly one JSON response")?;
    anyhow::ensure!(
        response.schema_version == TRIAL_WORKER_RESPONSE_SCHEMA_VERSION,
        "unsupported calibration trial-worker response schema {}",
        response.schema_version
    );
    anyhow::ensure!(
        response.request == *request,
        "calibration child response does not echo its exact request"
    );
    response.result.validate()?;
    anyhow::ensure!(
        response.result.request == request.calibration,
        "calibration child result request disagrees with its envelope"
    );
    Ok(response.result)
}

fn read_trial_request() -> Result<TrialWorkerRequest> {
    let stdin = std::io::stdin();
    let mut bytes = Vec::new();
    stdin
        .lock()
        .take(u64::try_from(MAX_TRIAL_REQUEST_BYTES + 1).expect("request limit fits u64"))
        .read_to_end(&mut bytes)
        .context("failed to read calibration trial request")?;
    anyhow::ensure!(
        bytes.len() <= MAX_TRIAL_REQUEST_BYTES,
        "calibration trial request exceeds {} bytes",
        MAX_TRIAL_REQUEST_BYTES
    );
    let request: TrialWorkerRequest =
        serde_json::from_slice(&bytes).context("invalid calibration trial-worker request JSON")?;
    request.validate()?;
    Ok(request)
}

fn load_candidate_policies(paths: &[PathBuf]) -> Result<Vec<CandidatePolicy>> {
    anyhow::ensure!(!paths.is_empty(), "at least one --policy is required");
    let mut seen = BTreeSet::new();
    paths
        .iter()
        .map(|path| {
            let path = canonical_regular_file(path, "execution policy")?;
            let policy = ExecutionPolicy::load(&path)?;
            validate_candidate_policy(&policy)?;
            anyhow::ensure!(
                seen.insert(policy.canonical_json()?),
                "duplicate calibration policy {}",
                path.display()
            );
            Ok(CandidatePolicy { path, policy })
        })
        .collect()
}

fn validate_candidate_policy(policy: &ExecutionPolicy) -> Result<()> {
    validate_calibration_policy(policy)
}

fn validate_selector_policy(selector: &str, policy: &ExecutionPolicy) -> Result<()> {
    let expected = match parse_calibration_device_selector(selector)? {
        CalibrationDeviceSelector::Cpu => ExecutionBackendPolicy::Cpu,
        CalibrationDeviceSelector::Cuda => ExecutionBackendPolicy::Cuda,
        CalibrationDeviceSelector::Metal => ExecutionBackendPolicy::Metal,
    };
    anyhow::ensure!(
        policy.execution_backend == expected,
        "calibration device selector {selector:?} disagrees with policy backend {:?}",
        policy.execution_backend
    );
    Ok(())
}

fn validate_policy_family(policies: &[CandidatePolicy]) -> Result<()> {
    let first = policies
        .first()
        .context("at least one --policy is required")?;
    let backend = first.policy.execution_backend;
    let precompute_adaln = first.policy.precompute_adaln;
    for candidate in policies {
        validate_candidate_policy(&candidate.policy)?;
        anyhow::ensure!(
            candidate.policy.execution_backend == backend,
            "calibration candidates must use one execution backend family"
        );
        anyhow::ensure!(
            candidate.policy.precompute_adaln == precompute_adaln,
            "calibration candidates must use the same AdaLN precomputation setting"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum CalibrationDeviceSelector {
    Cpu,
    Cuda,
    Metal,
}

fn parse_calibration_device_selector(value: &str) -> Result<CalibrationDeviceSelector> {
    if value == "cpu" {
        return Ok(CalibrationDeviceSelector::Cpu);
    }
    if let Some(ordinal) = value.strip_prefix("cuda:") {
        let ordinal = ordinal
            .parse::<usize>()
            .with_context(|| format!("invalid device ordinal in {value:?}"))?;
        anyhow::ensure!(
            value == format!("cuda:{ordinal}"),
            "calibration device selector must use canonical decimal syntax"
        );
        return Ok(CalibrationDeviceSelector::Cuda);
    }
    if let Some(ordinal) = value.strip_prefix("metal:") {
        let ordinal = ordinal
            .parse::<usize>()
            .with_context(|| format!("invalid device ordinal in {value:?}"))?;
        anyhow::ensure!(
            value == format!("metal:{ordinal}"),
            "calibration device selector must use canonical decimal syntax"
        );
        return Ok(CalibrationDeviceSelector::Metal);
    }
    bail!("unknown calibration device {value:?}; use cpu, cuda:N, or metal:N")
}

fn canonical_regular_file(path: &Path, label: &str) -> Result<PathBuf> {
    let path = std::fs::canonicalize(path)
        .with_context(|| format!("failed to resolve {label} {}", path.display()))?;
    anyhow::ensure!(path.is_file(), "{label} is not a file: {}", path.display());
    Ok(path)
}

fn ensure_file_within_limit(path: &Path, label: &str, limit: usize) -> Result<()> {
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("failed to stat {label} {}", path.display()))?
        .len();
    anyhow::ensure!(
        bytes <= u64::try_from(limit).context("calibration file limit exceeds u64")?,
        "{label} is {bytes} bytes, exceeding the {limit}-byte calibration limit"
    );
    Ok(())
}

fn read_bounded_file(path: &Path, label: &str, limit: usize) -> Result<Vec<u8>> {
    ensure_file_within_limit(path, label, limit)?;
    let file =
        File::open(path).with_context(|| format!("failed to open {label} {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(u64::try_from(limit).context("calibration file limit exceeds u64")? + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label} {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() <= limit,
        "{label} grew beyond the {limit}-byte calibration limit while reading"
    );
    Ok(bytes)
}

fn bounded_child_error(stderr: &[u8]) -> String {
    let end = stderr.len().min(MAX_CHILD_ERROR_BYTES);
    let mut text = String::from_utf8_lossy(&stderr[..end]).into_owned();
    if stderr.len() > end {
        text.push_str(" [truncated]");
    }
    if text.trim().is_empty() {
        "child produced no stderr".to_owned()
    } else {
        text
    }
}
