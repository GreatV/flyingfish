use crate::{
    h3::policy::ExecutionPolicy,
    h3::resources::{ResourceBudget, ResourceEstimate},
    h3::scheduler::H3Scheduler,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

mod history;
mod validation;

pub use history::take_t2va_checkpoint_metadata;

pub const POLICY_HISTORY_SCHEMA_VERSION: u32 = 1;
pub const CHECKPOINT_IDENTITY_SCHEMA_VERSION: u32 = 1;
pub const RECOVERY_PLAN_SCHEMA_VERSION: u32 = 1;
pub const ATTEMPT_JOURNAL_ENTRY_SCHEMA_VERSION: u32 = 1;
pub const WORKER_OUTCOME_ENVELOPE_SCHEMA_VERSION: u32 = 1;

pub const POLICY_HISTORY_JSON_TENSOR: &str = "ff_policy_history_json";
pub const POLICY_HISTORY_SCHEMA_TENSOR: &str = "ff_policy_history_schema";

pub const MAX_RECOVERY_JSON_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_RECOVERY_CHECKPOINT_BYTES: u64 = 512 * 1024 * 1024;

const MAX_POLICY_SEGMENTS: usize = 10_000;
const RECOVERY_RESOURCE_ESTIMATE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySegment {
    pub start_evaluation: u64,
    pub end_evaluation_exclusive: u64,
    pub policy: ExecutionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyHistory {
    pub schema_version: u32,
    pub completed_evaluations: u64,
    pub segments: Vec<PolicySegment>,
}

/// The retry ceiling a bounded-recovery contract may authorize.
pub const MAX_RECOVERY_RETRIES: u64 = 31;

/// The resource estimate a policy was admitted against.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAdmission {
    pub resource_estimate_schema_version: u32,
    pub estimate: ResourceEstimate,
    pub budget: ResourceBudget,
}

impl ResourceAdmission {
    pub fn new(estimate: ResourceEstimate, budget: ResourceBudget) -> Result<Self> {
        let admission = Self {
            resource_estimate_schema_version: RECOVERY_RESOURCE_ESTIMATE_SCHEMA_VERSION,
            estimate,
            budget,
        };
        admission.validate()?;
        Ok(admission)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.resource_estimate_schema_version == RECOVERY_RESOURCE_ESTIMATE_SCHEMA_VERSION,
            "unsupported admission resource-estimate schema {}; recovery schema 1 requires {}",
            self.resource_estimate_schema_version,
            RECOVERY_RESOURCE_ESTIMATE_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.estimate.schema_version == RECOVERY_RESOURCE_ESTIMATE_SCHEMA_VERSION,
            "embedded resource-estimate schema {} disagrees with recovery schema 1 ({})",
            self.estimate.schema_version,
            RECOVERY_RESOURCE_ESTIMATE_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.estimate.check_budget(self.budget).within_budget,
            "admitted policy does not fit its recorded budget"
        );
        Ok(())
    }
}

/// One policy a bounded-recovery contract may fall back to, with the admission
/// that authorized it. The admission travels inside the policy it belongs to,
/// so the two cannot be paired wrongly.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedRecoveryPolicy {
    pub policy: ExecutionPolicy,
    pub admission: ResourceAdmission,
}

impl AdmittedRecoveryPolicy {
    pub fn new(policy: ExecutionPolicy, admission: ResourceAdmission) -> Result<Self> {
        let admitted = Self { policy, admission };
        admitted.validate()?;
        Ok(admitted)
    }

    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.admission.validate()
    }
}

/// The largest denoise schedule this build will plan or resume.
pub const MAX_SCHEDULE_POINTS: u64 = 10_000;

/// Check that a requested schedule actually produces the evaluations it asks
/// for.
///
/// A shift value can collapse the schedule to fewer timesteps than requested,
/// which would silently shorten the run. Catching that before any weight is
/// read is what keeps a doomed request from spending the residency budget.
pub fn validate_t2va_schedule(sigma_points: u64, video_shift: f32, audio_shift: f32) -> Result<()> {
    anyhow::ensure!(
        (2..=MAX_SCHEDULE_POINTS).contains(&sigma_points),
        "sigma points must be in 2..={MAX_SCHEDULE_POINTS}"
    );
    for (name, value) in [("video shift", video_shift), ("audio shift", audio_shift)] {
        anyhow::ensure!(
            value.is_finite() && value > 0.0,
            "{name} must be finite and positive"
        );
        let mut scheduler = H3Scheduler::new(value)?;
        let actual = scheduler
            .set_timesteps(usize::try_from(sigma_points).context("sigma points exceed usize")?)?;
        anyhow::ensure!(
            actual.len()
                == usize::try_from(sigma_points - 1).context("evaluation count exceeds usize")?,
            "{name} collapses the requested {}-evaluation schedule to {} evaluations",
            sigma_points - 1,
            actual.len()
        );
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointIdentity {
    pub schema_version: u32,
    pub checkpoint_bytes: u64,
    pub completed_evaluations: u64,
    pub sigma_points: u32,
    pub video_shift_f32_bits: u32,
    pub audio_shift_f32_bits: u32,
    pub policy_history: PolicyHistory,
}

#[derive(Clone, Debug)]
pub struct T2vaCheckpointMetadata {
    pub completed_evaluations: u32,
    pub policy_history: PolicyHistory,
    pub sigma_points: u32,
    pub video_shift: f32,
    pub audio_shift: f32,
}

#[cfg(test)]
mod test_support {
    use super::*;
    use crate::h3::policy::AttentionBackendPolicy;
    use crate::h3::policy::AttentionExecutionPolicy;
    use crate::h3::policy::EXECUTION_POLICY_SCHEMA_VERSION;
    use crate::h3::policy::ExecutionBackendPolicy;
    use crate::h3::policy::ExecutionPolicy;
    use crate::h3::policy::H3NumericalContract;
    use crate::h3::policy::WeightExecutionPolicy;
    use crate::h3::policy::WeightSourcePolicy;
    use candle_core::DType;
    use candle_core::Device;
    use candle_core::Tensor;
    use std::collections::HashMap;

    pub(super) fn policy(query_rows: u64) -> ExecutionPolicy {
        let policy = ExecutionPolicy {
            schema_version: EXECUTION_POLICY_SCHEMA_VERSION,
            execution_backend: ExecutionBackendPolicy::Cuda,
            numerics: Box::new(
                H3NumericalContract::for_verified_target(
                    ExecutionBackendPolicy::Cuda,
                    AttentionBackendPolicy::FullSoftmax,
                )
                .unwrap(),
            ),
            attention: AttentionExecutionPolicy {
                backend: AttentionBackendPolicy::FullSoftmax,
                configured_projection_rows: query_rows,
                configured_query_rows: query_rows,
                configured_key_rows: None,
            },
            configured_ffn_rows: query_rows,
            configured_output_rows: query_rows,
            weights: WeightExecutionPolicy {
                host_phase_priority: false,
                device_cache: Default::default(),
                source: WeightSourcePolicy::Mmap,
                granularity: crate::runtime::weights::CacheGranularity::Shard,
                cache_shards: 1,
                cache_bytes: None,
            },
            precompute_adaln: true,
        };
        policy.validate().unwrap();
        policy
    }

    pub(super) fn checkpoint(history: PolicyHistory) -> CheckpointIdentity {
        CheckpointIdentity::new(4096, history, 5, 12.0, 3.0).unwrap()
    }

    pub(super) fn policy_history(
        completed_evaluations: u64,
        policy: ExecutionPolicy,
    ) -> PolicyHistory {
        let mut history = PolicyHistory::new();
        if completed_evaluations > 0 {
            history
                .append_successful_evaluations(completed_evaluations, policy)
                .unwrap();
        }
        history
    }

    pub(super) fn checkpoint_tensors() -> HashMap<&'static str, Tensor> {
        let device = Device::Cpu;
        HashMap::from([
            (
                "video_latents",
                Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &device).unwrap(),
            ),
            (
                "audio_latents",
                Tensor::zeros((1, 1, 1), DType::F32, &device).unwrap(),
            ),
            (
                "prompt_embeddings",
                Tensor::zeros((1, 1, 1), DType::F32, &device).unwrap(),
            ),
            (
                "text_token_tags",
                Tensor::zeros(1, DType::U32, &device).unwrap(),
            ),
        ])
    }

    pub(super) fn checkpoint_tensors_with_metadata(
        history: &PolicyHistory,
    ) -> HashMap<&'static str, Tensor> {
        let mut tensors = checkpoint_tensors();
        tensors.insert(
            "completed_steps",
            Tensor::new(
                u32::try_from(history.completed_evaluations).unwrap(),
                &Device::Cpu,
            )
            .unwrap(),
        );
        tensors.insert("sigma_points", Tensor::new(3u32, &Device::Cpu).unwrap());
        tensors.insert("video_shift", Tensor::new(12f32, &Device::Cpu).unwrap());
        tensors.insert("audio_shift", Tensor::new(3f32, &Device::Cpu).unwrap());
        history
            .insert_checkpoint_tensors(&mut tensors, &Device::Cpu)
            .unwrap();
        tensors
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::h3::policy::ExecutionBackendPolicy;
    use crate::h3::policy::ExecutionPolicy;
    use crate::h3::policy::H3QwenNumericalContract;
    use crate::h3::policy::H3QwenVisionLinearGeometry;
    use candle_core::DType;
    use candle_core::Device;
    use candle_core::Tensor;
    use std::collections::HashMap;
    use std::num::NonZeroUsize;

    #[test]
    fn policy_history_rejects_gaps_duplicates_and_hash_tampering() {
        let first = policy(32);
        let mut gap =
            PolicyHistory::from_segments(3, vec![PolicySegment::new(0, 3, first.clone()).unwrap()])
                .unwrap();
        gap.segments[0].start_evaluation = 2;
        assert!(
            gap.validate()
                .unwrap_err()
                .to_string()
                .contains("contiguous")
        );

        let duplicate = PolicyHistory {
            schema_version: POLICY_HISTORY_SCHEMA_VERSION,
            completed_evaluations: 2,
            segments: vec![
                PolicySegment::new(0, 1, first.clone()).unwrap(),
                PolicySegment::new(1, 2, first.clone()).unwrap(),
            ],
        };
        assert!(
            duplicate
                .validate()
                .unwrap_err()
                .to_string()
                .contains("merge")
        );

        let mut reversed = PolicySegment::new(0, 1, first).unwrap();
        reversed.end_evaluation_exclusive = 0;
        assert!(
            reversed
                .validate()
                .unwrap_err()
                .to_string()
                .contains("empty or reversed")
        );
    }

    #[test]
    fn resume_rejects_another_backend_or_mixed_numerics_but_allows_resource_changes() {
        let current = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            crate::runtime::weights::WeightSource::Mmap,
            crate::runtime::weights::CachePolicy::new(1),
            crate::h3::model::TransformerChunking::default(),
            false,
            true,
        )
        .unwrap();
        let mut other_backend = current.clone();
        other_backend.execution_backend = crate::h3::policy::ExecutionBackendPolicy::Cuda;
        *other_backend.numerics = crate::h3::policy::H3NumericalContract::for_verified_target(
            crate::h3::policy::ExecutionBackendPolicy::Cuda,
            other_backend.attention.backend,
        )
        .unwrap();
        let other_history =
            PolicyHistory::from_segments(1, vec![PolicySegment::new(0, 1, other_backend).unwrap()])
                .unwrap();
        let error = other_history
            .validate_resume_numerics(&Device::Cpu)
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("execution policy requires"),
            "unexpected error: {error:#}"
        );

        let mut online_chunking = crate::h3::model::TransformerChunking::default();
        online_chunking.attention.key =
            crate::h3::core::AttentionKeyChunkPolicy::chunked(16).unwrap();
        let online = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            crate::runtime::weights::WeightSource::Mmap,
            crate::runtime::weights::CachePolicy::new(1),
            online_chunking,
            false,
            true,
        )
        .unwrap();
        let mixed = PolicyHistory::from_segments(
            2,
            vec![
                PolicySegment::new(0, 1, current.clone()).unwrap(),
                PolicySegment::new(1, 2, online).unwrap(),
            ],
        )
        .unwrap();
        let error = mixed.validate_resume_numerics(&Device::Cpu).unwrap_err();
        assert!(error.to_string().contains("numerics.attention"));

        let mut tighter = current.clone();
        tighter.configured_output_rows = 1;
        let resource_only = PolicyHistory::from_segments(
            2,
            vec![
                PolicySegment::new(0, 1, current).unwrap(),
                PolicySegment::new(1, 2, tighter).unwrap(),
            ],
        )
        .unwrap();
        resource_only
            .validate_resume_numerics(&Device::Cpu)
            .unwrap();
    }

    #[test]
    fn policy_history_checkpoint_codec_roundtrips_and_rejects_corruption() {
        let history = policy_history(2, policy(32));
        let mut encoded = checkpoint_tensors();
        encoded.insert("completed_steps", Tensor::new(2u32, &Device::Cpu).unwrap());
        history
            .insert_checkpoint_tensors(&mut encoded, &Device::Cpu)
            .unwrap();
        let mut loaded = encoded
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        assert_eq!(
            PolicyHistory::take_checkpoint_tensors(&mut loaded).unwrap(),
            Some(history.clone())
        );

        let mut corrupted = checkpoint_tensors();
        history
            .insert_checkpoint_tensors(&mut corrupted, &Device::Cpu)
            .unwrap();
        corrupted.remove(super::POLICY_HISTORY_SCHEMA_TENSOR);
        let mut corrupted = corrupted
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        assert!(PolicyHistory::take_checkpoint_tensors(&mut corrupted).is_err());

        // A history recorded on another backend still round-trips through a
        // checkpoint, and is refused only when a device is named for it.
        let mut other_backend = policy(32);
        other_backend.execution_backend = crate::h3::policy::ExecutionBackendPolicy::Cuda;
        *other_backend.numerics = crate::h3::policy::H3NumericalContract::for_verified_target(
            crate::h3::policy::ExecutionBackendPolicy::Cuda,
            other_backend.attention.backend,
        )
        .unwrap();
        let other_history = policy_history(2, other_backend.clone());
        let mut recorded = checkpoint_tensors();
        recorded.insert("completed_steps", Tensor::new(2u32, &Device::Cpu).unwrap());
        other_history
            .insert_checkpoint_tensors(&mut recorded, &Device::Cpu)
            .unwrap();
        let mut recorded = recorded
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        assert_eq!(
            PolicyHistory::take_checkpoint_tensors(&mut recorded).unwrap(),
            Some(other_history)
        );
        let error = other_backend.validate_device(&Device::Cpu).unwrap_err();
        assert!(error.to_string().contains("execution policy requires"));
    }

    #[test]
    fn checkpoint_collects_step_zero_and_new_history_only() {
        let directory = tempfile::tempdir().unwrap();

        let step_zero = directory.path().join("step-zero.safetensors");
        let step_zero_tensors = checkpoint_tensors_with_metadata(&PolicyHistory::new());
        candle_core::safetensors::save(&step_zero_tensors, &step_zero).unwrap();
        let step_zero_identity = CheckpointIdentity::collect(&step_zero).unwrap();
        assert_eq!(step_zero_identity.completed_evaluations, 0);
        assert_eq!(step_zero_identity.policy_history, PolicyHistory::new());
        step_zero_identity.verify(&step_zero).unwrap();

        let history = policy_history(2, policy(32));
        let history_path = directory.path().join("history.safetensors");
        let history_tensors = checkpoint_tensors_with_metadata(&history);
        candle_core::safetensors::save(&history_tensors, &history_path).unwrap();
        let history_identity = CheckpointIdentity::collect(&history_path).unwrap();
        assert_eq!(history_identity.policy_history, history);
    }

    #[test]
    fn checkpoint_collection_accepts_complete_qwen_provenance_and_rejects_partial_or_unknown_ff() {
        let directory = tempfile::tempdir().unwrap();
        let contract = H3QwenNumericalContract::for_verified_target(
            ExecutionBackendPolicy::Cpu,
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(1).unwrap(),
            0,
            H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
        )
        .unwrap();
        let mut complete = checkpoint_tensors_with_metadata(&PolicyHistory::new());
        contract
            .insert_artifact_tensors(&mut complete, &Device::Cpu)
            .unwrap();
        let complete_path = directory.path().join("complete-qwen.safetensors");
        candle_core::safetensors::save(&complete, &complete_path).unwrap();
        CheckpointIdentity::collect(&complete_path).unwrap();

        let mut partial = complete.clone();
        partial.remove(crate::h3::policy::H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR);
        let partial_path = directory.path().join("partial-qwen.safetensors");
        candle_core::safetensors::save(&partial, &partial_path).unwrap();
        assert!(CheckpointIdentity::collect(&partial_path).is_err());

        let mut unknown = complete;
        unknown.insert(
            "ff_unknown_metadata",
            Tensor::new(1u32, &Device::Cpu).unwrap(),
        );
        let unknown_path = directory.path().join("unknown-ff.safetensors");
        candle_core::safetensors::save(&unknown, &unknown_path).unwrap();
        assert!(CheckpointIdentity::collect(&unknown_path).is_err());
    }

    #[test]
    fn checkpoint_collection_rejects_missing_provenance_tensors_and_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let initial = directory.path().join("initial.safetensors");
        candle_core::safetensors::save(&checkpoint_tensors(), &initial).unwrap();
        assert!(CheckpointIdentity::collect(&initial).is_err());

        let unproven = directory.path().join("unproven.safetensors");
        let mut tensors = checkpoint_tensors();
        tensors.insert("completed_steps", Tensor::new(1u32, &Device::Cpu).unwrap());
        candle_core::safetensors::save(&tensors, &unproven).unwrap();
        assert!(CheckpointIdentity::collect(&unproven).is_err());

        let incomplete = directory.path().join("incomplete.safetensors");
        let only_metadata =
            HashMap::from([("completed_steps", Tensor::new(0u32, &Device::Cpu).unwrap())]);
        candle_core::safetensors::save(&only_metadata, &incomplete).unwrap();
        assert!(CheckpointIdentity::collect(&incomplete).is_err());

        let unexpected = directory.path().join("unexpected.safetensors");
        let mut tensors = checkpoint_tensors_with_metadata(&PolicyHistory::new());
        tensors.insert("junk", Tensor::new(1u32, &Device::Cpu).unwrap());
        candle_core::safetensors::save(&tensors, &unexpected).unwrap();
        assert!(CheckpointIdentity::collect(&unexpected).is_err());

        let wrong_dtype = directory.path().join("wrong-dtype.safetensors");
        let mut tensors = checkpoint_tensors_with_metadata(&PolicyHistory::new());
        tensors.insert(
            "video_latents",
            Tensor::zeros((1, 1, 1, 1, 1), DType::U8, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &wrong_dtype).unwrap();
        assert!(CheckpointIdentity::collect(&wrong_dtype).is_err());

        let wrong_tag = directory.path().join("wrong-tag.safetensors");
        let mut tensors = checkpoint_tensors_with_metadata(&PolicyHistory::new());
        tensors.insert(
            "text_token_tags",
            Tensor::new(&[3u32], &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &wrong_tag).unwrap();
        assert!(CheckpointIdentity::collect(&wrong_tag).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let real = directory.path().join("real.safetensors");
            candle_core::safetensors::save(&checkpoint_tensors(), &real).unwrap();
            let link = directory.path().join("link.safetensors");
            symlink(&real, &link).unwrap();
            assert!(CheckpointIdentity::collect(&link).is_err());
        }
    }

    #[test]
    fn checkpoint_verify_detects_replaced_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.safetensors");
        let tensors = checkpoint_tensors_with_metadata(&PolicyHistory::new());
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let identity = CheckpointIdentity::collect(&path).unwrap();

        let mut changed = tensors;
        changed.insert(
            "video_latents",
            Tensor::ones(1, DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&changed, &path).unwrap();
        assert!(identity.verify(&path).is_err());
    }
}
