//! Serializable layer ownership and per-device execution contracts.
use crate::{
    admission::GlmLayerScope,
    execution_policy::{GlmExecutionBackend, GlmExecutionPolicy},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmRankPolicy {
    pub ordinal: usize,
    pub scope: GlmLayerScope,
    /// Static residency and expert-cache bounds apply only to this rank's scope.
    pub execution: GlmExecutionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmPartitionPolicy {
    pub schema_version: u32,
    pub transport: GlmPartitionTransport,
    pub ranks: Vec<GlmRankPolicy>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmPartitionTransport {
    SynchronizedCudaDeviceCopyV1,
}

impl GlmPartitionPolicy {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1 && self.ranks.len() >= 2,
            "invalid GLM partition policy schema/rank count"
        );
        let total = self.ranks[0].scope.total_layers;
        let mut seen = std::collections::BTreeSet::new();
        for (rank, p) in self.ranks.iter().enumerate() {
            ensure!(
                seen.insert(p.ordinal),
                "duplicate CUDA ordinal in GLM partition"
            );
            ensure!(
                p.scope == GlmLayerScope::for_rank(total, rank, self.ranks.len())?,
                "noncanonical GLM layer assignment"
            );
            p.execution.validate()?;
            ensure!(
                p.execution.backend == GlmExecutionBackend::Cuda,
                "GLM partitions require CUDA"
            );
            ensure!(
                p.execution
                    .expert_cache
                    .readmission_interval_tokens
                    .is_none(),
                "partition cache bounds must be fixed"
            );
            let mut math = p.execution.clone();
            let reference = &self.ranks[0].execution;
            math.weights = reference.weights.clone();
            math.resident_static = reference.resident_static;
            math.expert_cache = reference.expert_cache.clone();
            ensure!(math == *reference, "GLM rank numerical contracts disagree");
        }
        Ok(())
    }
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::execution_policy::ExpertCacheOptions;
    #[test]
    fn partition_policy_binds_devices_scopes_and_consistent_numerics() {
        use crate::{
            expert_cache::ExpertCacheReplacementPolicy, expert_cache_manager::ExpertCacheLayout,
        };
        use candle_core::Device;
        use ff_core::weights::{CachePolicy, WeightSource};
        let mut execution = GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            true,
            8,
            ExpertCacheOptions {
                layout: ExpertCacheLayout::PerLayerSplit,
                replacement: ExpertCacheReplacementPolicy::Lru,
                maximum_bound_bytes: 1024,
                minimum_bound_bytes: 1024,
                adaptive: false,
            },
        )
        .unwrap();
        execution.backend = GlmExecutionBackend::Cuda;
        let policy = GlmPartitionPolicy {
            schema_version: 1,
            transport: GlmPartitionTransport::SynchronizedCudaDeviceCopyV1,
            ranks: (0..2)
                .map(|rank| GlmRankPolicy {
                    ordinal: rank,
                    scope: GlmLayerScope::for_rank(4, rank, 2).unwrap(),
                    execution: execution.clone(),
                })
                .collect(),
        };
        let bytes = policy.canonical_json().unwrap();
        let restored: GlmPartitionPolicy = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, policy);
        let mut altered = policy.clone();
        altered.ranks[1].ordinal = 0;
        assert!(altered.validate().is_err());
        let mut altered = policy.clone();
        altered.ranks[1].scope.start = 1;
        assert!(altered.validate().is_err());
        let mut altered = policy.clone();
        altered.ranks[1].execution.cpu_fp8_dequantization = false;
        assert!(
            altered
                .validate()
                .unwrap_err()
                .to_string()
                .contains("numerical")
        );
        let mut altered = policy.clone();
        altered.ranks[1].ordinal = 2;
        altered.validate().unwrap();
        assert_ne!(
            altered.canonical_json().unwrap(),
            policy.canonical_json().unwrap()
        );
        let mut altered = policy.clone();
        altered.ranks[1].execution.expert_cache.maximum_bound_bytes = 2048;
        altered.ranks[1].execution.expert_cache.minimum_bound_bytes = 2048;
        assert_ne!(
            altered.canonical_json().unwrap(),
            policy.canonical_json().unwrap()
        );
    }
}
