use super::{
    CHECKPOINT_IDENTITY_SCHEMA_VERSION, CheckpointIdentity, MAX_POLICY_SEGMENTS,
    MAX_RECOVERY_CHECKPOINT_BYTES, POLICY_HISTORY_JSON_TENSOR, POLICY_HISTORY_SCHEMA_TENSOR,
    POLICY_HISTORY_SCHEMA_VERSION, PolicyHistory, PolicySegment, T2vaCheckpointMetadata,
    validation::{canonical_json, from_json, validate_schema},
};
use crate::h3::conditioning_provenance::H3ConditioningProvenance;
use crate::h3::core::MODALITY_COUNT;
use crate::h3::policy::ExecutionPolicy;
use crate::h3::scheduler::H3Scheduler;
use crate::runtime::artifact::read_artifact_snapshot;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::Path};

impl PolicySegment {
    pub fn new(
        start_evaluation: u64,
        end_evaluation_exclusive: u64,
        policy: ExecutionPolicy,
    ) -> Result<Self> {
        let segment = Self {
            start_evaluation,
            end_evaluation_exclusive,
            policy,
        };
        segment.validate()?;
        Ok(segment)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.start_evaluation < self.end_evaluation_exclusive,
            "policy-history segment [{}, {}) is empty or reversed",
            self.start_evaluation,
            self.end_evaluation_exclusive
        );
        self.policy.validate()?;
        Ok(())
    }
}

impl PolicyHistory {
    pub fn new() -> Self {
        Self {
            schema_version: POLICY_HISTORY_SCHEMA_VERSION,
            completed_evaluations: 0,
            segments: Vec::new(),
        }
    }

    /// Validate the invariant required before any checkpoint history is used
    /// to resume computation. Resource bounds may change between segments,
    /// but arithmetic identity may not, and every recorded policy must be
    /// executable by the selected device/current binary.
    pub fn validate_resume_numerics(&self, device: &Device) -> Result<()> {
        self.validate()?;
        let Some(first) = self.segments.first() else {
            return Ok(());
        };
        for segment in self.segments.iter().skip(1) {
            if let Some(field) = first.policy.first_numerical_difference(&segment.policy) {
                anyhow::bail!(
                    "checkpoint policy history changes numerical contract at evaluations [{}, {}) at {field}",
                    segment.start_evaluation,
                    segment.end_evaluation_exclusive
                );
            }
        }
        for segment in &self.segments {
            segment.policy.validate_device(device).with_context(|| {
                format!(
                    "checkpoint policy history evaluations [{}, {}) are not executable",
                    segment.start_evaluation, segment.end_evaluation_exclusive
                )
            })?;
        }
        Ok(())
    }

    pub fn from_segments(completed_evaluations: u64, segments: Vec<PolicySegment>) -> Result<Self> {
        let history = Self {
            schema_version: POLICY_HISTORY_SCHEMA_VERSION,
            completed_evaluations,
            segments,
        };
        history.validate()?;
        Ok(history)
    }

    pub fn append_successful_evaluations(
        &mut self,
        completed_evaluations: u64,
        policy: ExecutionPolicy,
    ) -> Result<()> {
        self.validate()?;
        anyhow::ensure!(
            completed_evaluations > self.completed_evaluations,
            "successful policy-history append must advance beyond evaluation {}",
            self.completed_evaluations
        );
        policy.validate()?;
        let mut updated = self.clone();
        if let Some(last) = updated.segments.last_mut()
            && last.policy == policy
        {
            last.end_evaluation_exclusive = completed_evaluations;
        } else {
            updated.segments.push(PolicySegment::new(
                updated.completed_evaluations,
                completed_evaluations,
                policy,
            )?);
        }
        updated.completed_evaluations = completed_evaluations;
        updated.validate()?;
        *self = updated;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema(
            self.schema_version,
            POLICY_HISTORY_SCHEMA_VERSION,
            "policy-history",
        )?;
        anyhow::ensure!(
            self.segments.len() <= MAX_POLICY_SEGMENTS,
            "policy history has {} segments, exceeding the schema-1 limit of {MAX_POLICY_SEGMENTS}",
            self.segments.len()
        );

        let mut next = 0;
        let mut previous_policy: Option<&ExecutionPolicy> = None;
        for segment in &self.segments {
            segment.validate()?;
            anyhow::ensure!(
                segment.start_evaluation == next,
                "policy history is not contiguous at evaluation {next}: next segment starts at {}",
                segment.start_evaluation
            );
            anyhow::ensure!(
                segment.end_evaluation_exclusive <= self.completed_evaluations,
                "policy-history segment ends at {}, beyond completed evaluation {}",
                segment.end_evaluation_exclusive,
                self.completed_evaluations
            );
            anyhow::ensure!(
                previous_policy != Some(&segment.policy),
                "adjacent policy-history segments use the same policy; merge them"
            );
            next = segment.end_evaluation_exclusive;
            previous_policy = Some(&segment.policy);
        }
        anyhow::ensure!(
            next == self.completed_evaluations,
            "policy history covers through evaluation {next}, but checkpoint completed evaluation {}",
            self.completed_evaluations
        );
        Ok(())
    }

    pub fn validate_extends(&self, prior: &Self) -> Result<()> {
        self.validate()?;
        prior.validate()?;
        anyhow::ensure!(
            self.completed_evaluations >= prior.completed_evaluations,
            "policy-history extension moved backwards from evaluation {} to {}",
            prior.completed_evaluations,
            self.completed_evaluations
        );
        anyhow::ensure!(
            self.prefix_through(prior.completed_evaluations)? == *prior,
            "policy-history extension rewrote successful policy provenance before evaluation {}",
            prior.completed_evaluations
        );
        Ok(())
    }

    fn prefix_through(&self, completed_evaluations: u64) -> Result<Self> {
        anyhow::ensure!(
            completed_evaluations <= self.completed_evaluations,
            "policy-history prefix boundary {completed_evaluations} exceeds {}",
            self.completed_evaluations
        );
        let mut segments = Vec::new();
        for segment in &self.segments {
            if segment.start_evaluation >= completed_evaluations {
                break;
            }
            let mut segment = segment.clone();
            segment.end_evaluation_exclusive =
                segment.end_evaluation_exclusive.min(completed_evaluations);
            segments.push(segment);
        }
        Self::from_segments(completed_evaluations, segments)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        from_json(bytes, "policy-history")
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        canonical_json(self, "policy-history")
    }

    pub fn insert_checkpoint_tensors(
        &self,
        tensors: &mut HashMap<&'static str, Tensor>,
        device: &Device,
    ) -> Result<()> {
        self.validate()?;
        for name in [POLICY_HISTORY_JSON_TENSOR, POLICY_HISTORY_SCHEMA_TENSOR] {
            anyhow::ensure!(
                !tensors.contains_key(name),
                "checkpoint already contains policy provenance tensor {name}"
            );
        }
        let json = self.canonical_json()?;
        tensors.insert(
            POLICY_HISTORY_JSON_TENSOR,
            Tensor::from_vec(json.clone(), json.len(), device)?,
        );
        tensors.insert(
            POLICY_HISTORY_SCHEMA_TENSOR,
            Tensor::new(POLICY_HISTORY_SCHEMA_VERSION, device)?,
        );
        Ok(())
    }

    pub fn take_checkpoint_tensors(tensors: &mut HashMap<String, Tensor>) -> Result<Option<Self>> {
        let present = [POLICY_HISTORY_JSON_TENSOR, POLICY_HISTORY_SCHEMA_TENSOR]
            .map(|name| tensors.contains_key(name));
        if present.iter().all(|present| !present) {
            return Ok(None);
        }
        anyhow::ensure!(
            present.iter().all(|present| *present),
            "checkpoint has incomplete policy-history metadata"
        );
        let json_tensor = tensors
            .remove(POLICY_HISTORY_JSON_TENSOR)
            .context("checkpoint is missing policy-history JSON")?;
        anyhow::ensure!(
            json_tensor.dtype() == DType::U8
                && json_tensor.rank() == 1
                && json_tensor.elem_count() <= super::MAX_RECOVERY_JSON_BYTES,
            "policy-history JSON must be a bounded U8 vector"
        );
        let json = json_tensor
            .to_vec1::<u8>()
            .context("policy-history JSON must be a U8 vector")?;
        let schema_tensor = tensors
            .remove(POLICY_HISTORY_SCHEMA_TENSOR)
            .context("checkpoint is missing policy-history schema")?;
        anyhow::ensure!(
            schema_tensor.dtype() == DType::U32 && schema_tensor.rank() == 0,
            "policy-history schema must be a U32 scalar"
        );
        let schema = schema_tensor
            .to_scalar::<u32>()
            .context("policy-history schema must be a U32 scalar")?;
        anyhow::ensure!(
            schema == POLICY_HISTORY_SCHEMA_VERSION,
            "unsupported checkpoint policy-history schema {schema}; this build supports schema {POLICY_HISTORY_SCHEMA_VERSION}"
        );
        let history = Self::from_json(&json)?;
        anyhow::ensure!(
            history.schema_version == schema,
            "policy-history JSON schema {} disagrees with checkpoint schema {schema}",
            history.schema_version
        );
        Ok(Some(history))
    }
}

impl Default for PolicyHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointIdentity {
    pub(crate) fn new(
        checkpoint_bytes: u64,
        policy_history: PolicyHistory,
        sigma_points: u32,
        video_shift: f32,
        audio_shift: f32,
    ) -> Result<Self> {
        let completed_evaluations = policy_history.completed_evaluations;
        let identity = Self {
            schema_version: CHECKPOINT_IDENTITY_SCHEMA_VERSION,
            checkpoint_bytes,
            completed_evaluations,
            sigma_points,
            video_shift_f32_bits: video_shift.to_bits(),
            audio_shift_f32_bits: audio_shift.to_bits(),
            policy_history,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema(
            self.schema_version,
            CHECKPOINT_IDENTITY_SCHEMA_VERSION,
            "checkpoint identity",
        )?;
        anyhow::ensure!(
            (1..=MAX_RECOVERY_CHECKPOINT_BYTES).contains(&self.checkpoint_bytes),
            "checkpoint size must be in 1..={MAX_RECOVERY_CHECKPOINT_BYTES} bytes"
        );
        anyhow::ensure!(
            self.sigma_points >= 2,
            "checkpoint sigma_points must be at least two"
        );
        anyhow::ensure!(
            self.completed_evaluations < u64::from(self.sigma_points),
            "checkpoint completed evaluation exceeds its schedule"
        );
        for (name, bits) in [
            ("video shift", self.video_shift_f32_bits),
            ("audio shift", self.audio_shift_f32_bits),
        ] {
            let value = f32::from_bits(bits);
            let mut scheduler = H3Scheduler::new(value)?;
            let timesteps = scheduler.set_timesteps(
                usize::try_from(self.sigma_points).context("sigma_points exceeds usize")?,
            )?;
            anyhow::ensure!(
                timesteps.len()
                    == usize::try_from(self.sigma_points - 1)
                        .context("checkpoint evaluation count exceeds usize")?,
                "checkpoint {name} collapses its declared schedule"
            );
        }
        self.policy_history.validate()?;
        anyhow::ensure!(
            self.completed_evaluations == self.policy_history.completed_evaluations,
            "checkpoint completed evaluation {} disagrees with policy history {}",
            self.completed_evaluations,
            self.policy_history.completed_evaluations
        );
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        from_json(bytes, "checkpoint identity")
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        canonical_json(self, "checkpoint identity")
    }

    pub fn collect(path: &Path) -> Result<Self> {
        let snapshot = read_artifact_snapshot(path, MAX_RECOVERY_CHECKPOINT_BYTES)?;
        let mut tensors = candle_core::safetensors::load_buffer(&snapshot.bytes, &Device::Cpu)
            .with_context(|| format!("failed to load checkpoint snapshot {}", path.display()))?;
        let metadata = take_t2va_checkpoint_metadata(&mut tensors)?
            .context("recovery identity requires a checkpoint, not initial T2VA inputs")?;
        Self::new(
            u64::try_from(snapshot.bytes.len()).context("checkpoint size exceeds u64")?,
            metadata.policy_history,
            metadata.sigma_points,
            metadata.video_shift,
            metadata.audio_shift,
        )
    }

    pub fn verify(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let actual = Self::collect(path)?;
        anyhow::ensure!(
            actual == *self,
            "checkpoint identity or policy provenance changed for {}",
            path.display()
        );
        Ok(())
    }
}

pub fn take_t2va_checkpoint_metadata(
    tensors: &mut HashMap<String, Tensor>,
) -> Result<Option<T2vaCheckpointMetadata>> {
    let has_metadata = [
        "completed_steps",
        "sigma_points",
        "video_shift",
        "audio_shift",
        POLICY_HISTORY_JSON_TENSOR,
        POLICY_HISTORY_SCHEMA_TENSOR,
    ]
    .iter()
    .any(|name| tensors.contains_key(*name));
    if !has_metadata {
        validate_t2va_payload(tensors)?;
        return Ok(None);
    }
    let completed_evaluations = tensors
        .remove("completed_steps")
        .context("recovery checkpoint is missing completed_steps")?;
    anyhow::ensure!(
        completed_evaluations.dtype() == DType::U32 && completed_evaluations.rank() == 0,
        "completed_steps must be a U32 scalar"
    );
    let completed_evaluations = u64::from(
        completed_evaluations
            .to_scalar::<u32>()
            .context("completed_steps must be a U32 scalar")?,
    );
    let history = PolicyHistory::take_checkpoint_tensors(tensors)?
        .context("recovery checkpoint is missing policy-history metadata")?;
    let _conditioning_provenance = H3ConditioningProvenance::take_artifact_tensors(tensors)?;
    anyhow::ensure!(
        history.completed_evaluations == completed_evaluations,
        "policy history completes {} evaluations, but checkpoint records {completed_evaluations}",
        history.completed_evaluations
    );
    anyhow::ensure!(
        !tensors.keys().any(|name| name.starts_with("ff_")),
        "checkpoint contains unsupported ff metadata"
    );
    let sigma_points = take_scalar_u32(tensors, "sigma_points")?;
    let video_shift = take_scalar_f32(tensors, "video_shift")?;
    let audio_shift = take_scalar_f32(tensors, "audio_shift")?;
    anyhow::ensure!(sigma_points >= 2, "sigma_points must be at least two");
    anyhow::ensure!(
        completed_evaluations < u64::from(sigma_points),
        "completed_steps {completed_evaluations} exceeds the {}-evaluation schedule",
        sigma_points - 1
    );
    anyhow::ensure!(
        video_shift > 0.0 && audio_shift > 0.0,
        "video_shift and audio_shift must be positive"
    );
    validate_t2va_payload(tensors)?;
    Ok(Some(T2vaCheckpointMetadata {
        completed_evaluations: u32::try_from(completed_evaluations)
            .context("completed_steps exceeds u32")?,
        policy_history: history,
        sigma_points,
        video_shift,
        audio_shift,
    }))
}

fn take_scalar_u32(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<u32> {
    let tensor = tensors
        .remove(name)
        .with_context(|| format!("recovery checkpoint is missing {name}"))?;
    anyhow::ensure!(
        tensor.dtype() == DType::U32 && tensor.rank() == 0,
        "{name} must be a U32 scalar"
    );
    tensor
        .to_scalar::<u32>()
        .with_context(|| format!("{name} must be a U32 scalar"))
}

fn take_scalar_f32(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<f32> {
    let tensor = tensors
        .remove(name)
        .with_context(|| format!("recovery checkpoint is missing {name}"))?;
    anyhow::ensure!(
        tensor.dtype() == DType::F32 && tensor.rank() == 0,
        "{name} must be an F32 scalar"
    );
    let value = tensor
        .to_scalar::<f32>()
        .with_context(|| format!("{name} must be an F32 scalar"))?;
    anyhow::ensure!(value.is_finite(), "{name} must be finite");
    Ok(value)
}

fn validate_t2va_payload(tensors: &HashMap<String, Tensor>) -> Result<()> {
    let mut names = tensors.keys().map(String::as_str).collect::<Vec<_>>();
    names.sort_unstable();
    anyhow::ensure!(
        names
            == [
                "audio_latents",
                "prompt_embeddings",
                "text_token_tags",
                "video_latents",
            ],
        "T2VA payload must contain exactly audio_latents, prompt_embeddings, text_token_tags, and video_latents; found {}",
        names.join(", ")
    );
    let video = &tensors["video_latents"];
    anyhow::ensure!(
        video.dtype() == DType::F32
            && video.rank() == 5
            && video.dims()[0] == 1
            && video.dims().iter().all(|dimension| *dimension > 0),
        "video_latents must be a non-empty F32 [1, channels, frames, height, width] tensor"
    );
    let audio = &tensors["audio_latents"];
    anyhow::ensure!(
        audio.dtype() == DType::F32
            && audio.rank() == 3
            && audio.dims().iter().all(|dimension| *dimension > 0),
        "audio_latents must be a non-empty F32 [channels, latent_channels, frames] tensor"
    );
    let prompt = &tensors["prompt_embeddings"];
    anyhow::ensure!(
        matches!(prompt.dtype(), DType::F32 | DType::F16 | DType::BF16)
            && prompt.rank() == 3
            && prompt.dims()[0] == 1
            && prompt.dims().iter().all(|dimension| *dimension > 0),
        "prompt_embeddings must be a non-empty floating-point [1, rows, width] tensor"
    );
    let tags = &tensors["text_token_tags"];
    anyhow::ensure!(
        tags.dtype() == DType::U32 && tags.rank() == 1 && tags.dims()[0] == prompt.dims()[1],
        "text_token_tags must be a U32 vector matching prompt rows"
    );
    anyhow::ensure!(
        tags.to_vec1::<u32>()?
            .iter()
            .all(|tag| *tag < MODALITY_COUNT as u32),
        "text_token_tags contains a modality outside H3's three modalities"
    );
    Ok(())
}
