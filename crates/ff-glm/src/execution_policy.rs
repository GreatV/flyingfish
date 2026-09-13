use super::{
    expert_cache::ExpertCacheReplacementPolicy, expert_cache_manager::ExpertCacheLayout,
    routing_trace::RoutingModelFamily,
};
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use ff_core::weights::{CacheGranularity, CachePolicy, WeightSource};
use serde::{Deserialize, Serialize};

pub const GLM_EXECUTION_POLICY_SCHEMA_VERSION: u32 = 8;
pub const MAX_GLM_EXECUTION_POLICY_JSON_BYTES: usize = 1024 * 1024;
pub const GLM_ADMISSION_SAFETY_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_GLM_EXPERT_CACHE_BOUND_BYTES: u64 = 1 << 50;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmExecutionBackend {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmWeightSourcePolicy {
    Mmap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmExpertCacheEntryUnit {
    DequantizedProjectionTensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmExecutionProfile {
    TextOnlyExactDsaShortContext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmActivationMath {
    F32SiluAndSigmoidThenInputDtypeCastV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmPrefillMath {
    LayerBatchedKda64GroupedExpertsV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmNormalizationMath {
    CpuTensorCudaTorchOrderedV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmRoutingMath {
    CpuScoreSortedCudaThresholdScanV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmHostWeightPolicy {
    pub source: GlmWeightSourcePolicy,
    pub cache_shards: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub cache_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "CacheGranularity::is_shard")]
    pub granularity: CacheGranularity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmExpertCacheExecutionPolicy {
    pub entry_unit: GlmExpertCacheEntryUnit,
    pub layout: ExpertCacheLayout,
    pub replacement: ExpertCacheReplacementPolicy,
    pub maximum_bound_bytes: u64,
    pub minimum_bound_bytes: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub readmission_interval_tokens: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmExecutionPolicy {
    pub schema_version: u32,
    pub model_family: RoutingModelFamily,
    pub execution_profile: GlmExecutionProfile,
    pub activation_math: GlmActivationMath,
    pub prefill_math: GlmPrefillMath,
    pub normalization_math: GlmNormalizationMath,
    pub routing_math: GlmRoutingMath,
    /// The share of each routed miss set evaluated on the host rather than
    /// moved to the device, in parts per thousand.
    ///
    /// Recorded because it changes which arithmetic ran, not merely where: the
    /// host path sums an expert in F32 through the fused block-FP8 matvec while
    /// the device path runs its own GEMM over dequantized weights. Per mille
    /// rather than a float so the policy stays exactly comparable. Zero, the
    /// default, means every expert went to the device and the policy describes
    /// the same execution it always did.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub host_expert_share_per_mille: u32,
    pub cpu_fp8_dequantization: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub pinned_fp8_transfer: bool,
    pub backend: GlmExecutionBackend,
    pub resident_static: bool,
    pub dsa_context_bound_tokens: u32,
    pub admission_safety_bytes: u64,
    pub weights: GlmHostWeightPolicy,
    pub expert_cache: GlmExpertCacheExecutionPolicy,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl GlmExecutionPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn from_runtime(
        device: &Device,
        weight_source: WeightSource,
        host_cache: CachePolicy,
        resident_static: bool,
        dsa_context_bound_tokens: usize,
        layout: ExpertCacheLayout,
        replacement: ExpertCacheReplacementPolicy,
        maximum_bound_bytes: usize,
        minimum_bound_bytes: usize,
        adaptive: bool,
    ) -> Result<Self> {
        ensure!(
            weight_source == WeightSource::Mmap,
            "GLM execution policy supports only mmap weights"
        );
        let backend = if device.is_cuda() {
            GlmExecutionBackend::Cuda
        } else {
            ensure!(device.is_cpu(), "GLM execution policy supports CPU or CUDA");
            GlmExecutionBackend::Cpu
        };
        let policy = Self {
            schema_version: GLM_EXECUTION_POLICY_SCHEMA_VERSION,
            model_family: RoutingModelFamily::Glm5Next,
            execution_profile: GlmExecutionProfile::TextOnlyExactDsaShortContext,
            activation_math: GlmActivationMath::F32SiluAndSigmoidThenInputDtypeCastV1,
            prefill_math: GlmPrefillMath::LayerBatchedKda64GroupedExpertsV1,
            normalization_math: GlmNormalizationMath::CpuTensorCudaTorchOrderedV1,
            routing_math: GlmRoutingMath::CpuScoreSortedCudaThresholdScanV1,
            // Nothing is scheduled onto the host until a caller, holding a
            // measurement of this machine, sets it.
            host_expert_share_per_mille: 0,
            cpu_fp8_dequantization: device.is_cpu(),
            pinned_fp8_transfer: false,
            backend,
            resident_static,
            dsa_context_bound_tokens: u32::try_from(dsa_context_bound_tokens)
                .context("GLM DSA context bound exceeds u32")?,
            admission_safety_bytes: GLM_ADMISSION_SAFETY_BYTES,
            weights: GlmHostWeightPolicy {
                source: GlmWeightSourcePolicy::Mmap,
                cache_shards: u64::try_from(host_cache.max_shards)
                    .context("GLM host shard-cache bound exceeds u64")?,
                cache_bytes: host_cache.max_bytes,
                granularity: host_cache.granularity,
            },
            expert_cache: GlmExpertCacheExecutionPolicy {
                entry_unit: GlmExpertCacheEntryUnit::DequantizedProjectionTensor,
                layout,
                replacement,
                maximum_bound_bytes: u64::try_from(maximum_bound_bytes)
                    .context("GLM expert-cache maximum exceeds u64")?,
                minimum_bound_bytes: u64::try_from(minimum_bound_bytes)
                    .context("GLM expert-cache minimum exceeds u64")?,
                readmission_interval_tokens: adaptive.then_some(1),
            },
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == GLM_EXECUTION_POLICY_SCHEMA_VERSION,
            "unsupported GLM execution-policy schema {}; this build supports schema {}",
            self.schema_version,
            GLM_EXECUTION_POLICY_SCHEMA_VERSION
        );
        ensure!(
            self.dsa_context_bound_tokens > 0,
            "GLM execution policy DSA context bound must be positive"
        );
        ensure!(
            !self.pinned_fp8_transfer
                || (self.backend == GlmExecutionBackend::Cuda && !self.cpu_fp8_dequantization),
            "pinned FP8 transfer requires GPU dequantization on CUDA"
        );
        if self.backend == GlmExecutionBackend::Cpu {
            ensure!(
                self.cpu_fp8_dequantization,
                "CPU GLM policy requires CPU FP8 conversion"
            );
        }
        ensure!(
            self.admission_safety_bytes == GLM_ADMISSION_SAFETY_BYTES,
            "GLM execution policy safety allowance must be {GLM_ADMISSION_SAFETY_BYTES} bytes"
        );
        ensure!(
            self.weights.cache_shards > 0,
            "GLM execution policy host shard-cache count must be positive"
        );
        usize::try_from(self.weights.cache_shards)
            .context("GLM host shard-cache count exceeds usize")?;
        if let Some(bytes) = self.weights.cache_bytes {
            ensure!(
                bytes > 0,
                "GLM host shard-cache bytes must be positive when present"
            );
        }
        ensure!(
            self.expert_cache.maximum_bound_bytes <= MAX_GLM_EXPERT_CACHE_BOUND_BYTES,
            "GLM expert-cache maximum exceeds {MAX_GLM_EXPERT_CACHE_BOUND_BYTES} bytes"
        );
        ensure!(
            self.expert_cache.minimum_bound_bytes <= self.expert_cache.maximum_bound_bytes,
            "GLM expert-cache minimum exceeds its maximum"
        );
        usize::try_from(self.expert_cache.maximum_bound_bytes)
            .context("GLM expert-cache maximum exceeds usize")?;
        usize::try_from(self.expert_cache.minimum_bound_bytes)
            .context("GLM expert-cache minimum exceeds usize")?;
        match self.expert_cache.readmission_interval_tokens {
            None => ensure!(
                self.expert_cache.minimum_bound_bytes == self.expert_cache.maximum_bound_bytes,
                "a fixed GLM expert cache must have equal minimum and maximum bounds"
            ),
            Some(1) => ensure!(
                self.expert_cache.maximum_bound_bytes > 0,
                "adaptive GLM expert-cache admission requires a positive maximum"
            ),
            Some(interval) => anyhow::bail!(
                "GLM expert-cache readmission interval must be exactly one token, found {interval}"
            ),
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let json = serde_json::to_vec(self).context("failed to serialize GLM execution policy")?;
        ensure!(
            json.len() <= MAX_GLM_EXECUTION_POLICY_JSON_BYTES,
            "GLM execution-policy JSON exceeds {MAX_GLM_EXECUTION_POLICY_JSON_BYTES} bytes"
        );
        Ok(json)
    }

    pub fn cache_policy(&self) -> Result<CachePolicy> {
        self.validate()?;
        let mut policy = CachePolicy::new(usize::try_from(self.weights.cache_shards)?)
            .with_granularity(self.weights.granularity);
        if let Some(bytes) = self.weights.cache_bytes {
            policy = policy.with_max_bytes(bytes);
        }
        Ok(policy)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_GLM_EXECUTION_POLICY_JSON_BYTES,
            "GLM execution-policy JSON exceeds {MAX_GLM_EXECUTION_POLICY_JSON_BYTES} bytes"
        );
        let policy: Self =
            serde_json::from_slice(bytes).context("invalid GLM execution-policy JSON")?;
        policy.validate()?;
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(adaptive: bool) -> GlmExecutionPolicy {
        GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            false,
            2_048,
            ExpertCacheLayout::SharedPool,
            ExpertCacheReplacementPolicy::Lfu,
            4_096,
            if adaptive { 1_024 } else { 4_096 },
            adaptive,
        )
        .unwrap()
    }

    /// A host share is part of what ran, so it has to reach the policy. It is
    /// absent when zero, which is what every run that measured nothing does.
    #[test]
    fn a_host_expert_share_is_recorded_and_absent_when_nothing_ran_there() {
        let mut policy = policy(false);
        assert_eq!(policy.host_expert_share_per_mille, 0);
        let canonical = String::from_utf8(policy.canonical_json().unwrap()).unwrap();
        assert!(
            !canonical.contains("host_expert_share_per_mille"),
            "an unused split should not appear in the policy: {canonical}"
        );

        policy.host_expert_share_per_mille = 484;
        let canonical = String::from_utf8(policy.canonical_json().unwrap()).unwrap();
        assert!(
            canonical.contains("\"host_expert_share_per_mille\":484"),
            "a used split must appear in the policy: {canonical}"
        );
        assert_eq!(
            GlmExecutionPolicy::from_json(canonical.as_bytes()).unwrap(),
            policy
        );
    }

    #[test]
    fn canonical_policy_preserves_cache_topology_replacement_and_bounds() {
        let first = policy(true);
        let json = first.canonical_json().unwrap();
        assert_eq!(GlmExecutionPolicy::from_json(&json).unwrap(), first);
        assert_eq!(
            String::from_utf8(json).unwrap(),
            r#"{"schema_version":8,"model_family":"glm5_next","execution_profile":"text_only_exact_dsa_short_context","activation_math":"f32_silu_and_sigmoid_then_input_dtype_cast_v1","prefill_math":"layer_batched_kda64_grouped_experts_v1","normalization_math":"cpu_tensor_cuda_torch_ordered_v1","routing_math":"cpu_score_sorted_cuda_threshold_scan_v1","cpu_fp8_dequantization":true,"backend":"cpu","resident_static":false,"dsa_context_bound_tokens":2048,"admission_safety_bytes":1073741824,"weights":{"source":"mmap","cache_shards":1,"cache_bytes":null},"expert_cache":{"entry_unit":"dequantized_projection_tensor","layout":"shared_pool","replacement":"lfu","maximum_bound_bytes":4096,"minimum_bound_bytes":1024,"readmission_interval_tokens":1}}"#
        );
        let mut changed = first.clone();
        changed.expert_cache.maximum_bound_bytes += 1;
        assert_ne!(
            first.canonical_json().unwrap(),
            changed.canonical_json().unwrap()
        );
        let mut changed = first.clone();
        changed.expert_cache.layout = ExpertCacheLayout::PerLayerSplit;
        assert_ne!(
            first.canonical_json().unwrap(),
            changed.canonical_json().unwrap()
        );
        let mut changed = first.clone();
        changed.expert_cache.replacement = ExpertCacheReplacementPolicy::Lru;
        assert_ne!(
            first.canonical_json().unwrap(),
            changed.canonical_json().unwrap()
        );
    }

    #[test]
    fn raw_tensor_cache_granularity_is_distinct_from_expert_cache_policy() {
        let shard = policy(false);
        let mut tensor = shard.clone();
        tensor.weights.granularity = CacheGranularity::Tensor;
        assert_eq!(tensor.expert_cache, shard.expert_cache);
        assert_ne!(
            tensor.canonical_json().unwrap(),
            shard.canonical_json().unwrap()
        );
        let restored: GlmExecutionPolicy =
            serde_json::from_slice(&tensor.canonical_json().unwrap()).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored, tensor);
        assert!(
            !String::from_utf8(shard.canonical_json().unwrap())
                .unwrap()
                .contains("granularity")
        );
    }

    #[test]
    fn fixed_and_adaptive_bounds_are_closed_and_fail_fast() {
        let mut fixed = policy(false);
        fixed.expert_cache.minimum_bound_bytes -= 1;
        assert!(fixed.validate().unwrap_err().to_string().contains("fixed"));

        let mut adaptive = policy(true);
        adaptive.expert_cache.readmission_interval_tokens = Some(2);
        assert!(
            adaptive
                .validate()
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
        let mut adaptive = policy(true);
        adaptive.expert_cache.minimum_bound_bytes = 4_097;
        assert!(
            adaptive
                .validate()
                .unwrap_err()
                .to_string()
                .contains("minimum")
        );
    }

    #[test]
    fn parser_rejects_unknown_schema_fields_and_oversize() {
        let json = policy(true).canonical_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&json).unwrap();
        value["unknown"] = serde_json::json!(true);
        assert!(
            GlmExecutionPolicy::from_json(&serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .to_string()
                .contains("invalid")
        );
        for (field, invalid) in [
            ("model_family", "minimax_h3"),
            ("execution_profile", "unbounded_or_fallback"),
            ("activation_math", "native_bf16"),
            ("activation_math", "f32_silu_then_input_dtype_cast_v1"),
            ("prefill_math", "token_serial"),
            ("normalization_math", "candle_cuda"),
            ("routing_math", "cuda_score_sorted"),
        ] {
            let mut value: serde_json::Value = serde_json::from_slice(&json).unwrap();
            value[field] = serde_json::json!(invalid);
            assert!(
                GlmExecutionPolicy::from_json(&serde_json::to_vec(&value).unwrap())
                    .unwrap_err()
                    .to_string()
                    .contains("invalid GLM execution-policy JSON")
            );
        }
        assert!(
            GlmExecutionPolicy::from_json(&vec![b' '; MAX_GLM_EXECUTION_POLICY_JSON_BYTES + 1])
                .unwrap_err()
                .to_string()
                .contains("exceeds")
        );
    }

    #[test]
    fn activation_contract_is_required_and_old_schema_is_rejected() {
        let json = policy(true).canonical_json().unwrap();
        let mut missing: serde_json::Value = serde_json::from_slice(&json).unwrap();
        missing.as_object_mut().unwrap().remove("activation_math");
        let error =
            GlmExecutionPolicy::from_json(&serde_json::to_vec(&missing).unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains("missing field `activation_math`"));
        let mut missing: serde_json::Value = serde_json::from_slice(&json).unwrap();
        missing.as_object_mut().unwrap().remove("prefill_math");
        assert!(GlmExecutionPolicy::from_json(&serde_json::to_vec(&missing).unwrap()).is_err());
        for field in ["normalization_math", "routing_math"] {
            let mut missing: serde_json::Value = serde_json::from_slice(&json).unwrap();
            missing.as_object_mut().unwrap().remove(field);
            assert!(GlmExecutionPolicy::from_json(&serde_json::to_vec(&missing).unwrap()).is_err());
        }
        // The PTX digest keys these policies once carried are no longer
        // tolerated on the way in: the format migrated in one step rather than
        // keeping a parallel path for records that name a digest.
        for legacy_field in ["cuda_normalization_ptx_sha256", "cuda_fp8_ptx_sha256"] {
            let mut legacy: serde_json::Value = serde_json::from_slice(&json).unwrap();
            legacy[legacy_field] = serde_json::json!("a legacy digest");
            assert!(GlmExecutionPolicy::from_json(&serde_json::to_vec(&legacy).unwrap()).is_err());
        }
        for version in [1, 2, 3, 4, 5, 6] {
            let mut old: serde_json::Value = serde_json::from_slice(&json).unwrap();
            old["schema_version"] = serde_json::json!(version);
            assert!(
                GlmExecutionPolicy::from_json(&serde_json::to_vec(&old).unwrap())
                    .unwrap_err()
                    .to_string()
                    .contains(&format!(
                        "unsupported GLM execution-policy schema {version}"
                    ))
            );
        }
    }
}
