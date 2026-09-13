//! Raw host weight storage under the same cache policy used for execution.
use crate::policy::ExecutionPolicy;
use anyhow::Result;
use ff_core::weights::accounting::{
    CacheInventory, CacheLoadLifetimes, CacheResidencyEstimate, estimate_cache_residency,
};

/// H3 materialization copies weights into owned arithmetic tensors before the
/// stage closure. Those copies belong to the existing compute/live-stage model,
/// so only the single foreground loader is charged here.
pub fn host_weight_residency_charges(
    policy: &ExecutionPolicy,
    inventory: &CacheInventory,
) -> Result<CacheResidencyEstimate> {
    let cache_policy = policy.cache_policy()?;
    let loaders = 1;
    let header_copies = if cache_policy.granularity == ff_core::weights::CacheGranularity::Tensor {
        inventory
            .shards
            .iter()
            .map(|shard| shard.header_bytes)
            .max()
            .unwrap_or(0)
            .checked_mul(loaders)
            .ok_or_else(|| anyhow::anyhow!("tensor header validation buffers overflow"))?
    } else {
        0
    };
    estimate_cache_residency(
        inventory,
        policy.weight_source(),
        cache_policy,
        CacheLoadLifetimes {
            maximum_concurrent_loads: loaders,
            additional_storage_bytes: header_copies,
            ..CacheLoadLifetimes::SERIAL
        },
    )
}
