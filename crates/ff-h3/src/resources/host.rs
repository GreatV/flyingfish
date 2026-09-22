//! Raw host weight storage under the same cache policy used for execution.
use crate::policy::ExecutionPolicy;
use anyhow::Result;
use ff_core::weights::accounting::{
    CacheInventory, CacheLoadLifetimes, CacheResidencyEstimate, estimate_cache_residency,
};
use ff_core::weights::{CacheGranularity, CachePolicy, WeightSource};

/// The residency charge one foreground loader imposes for a cache policy.
///
/// Both the execution-policy path and the configuration derivation charge
/// through here so an admitted run and its derived host bound disagree by
/// nothing.
pub fn cache_charge(
    inventory: &CacheInventory,
    source: WeightSource,
    cache_policy: CachePolicy,
) -> Result<CacheResidencyEstimate> {
    let loaders = 1;
    let header_copies = if cache_policy.granularity == CacheGranularity::Tensor {
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
        source,
        cache_policy,
        CacheLoadLifetimes {
            maximum_concurrent_loads: loaders,
            additional_storage_bytes: header_copies,
            ..CacheLoadLifetimes::SERIAL
        },
    )
}

/// H3 materialization copies weights into owned arithmetic tensors before the
/// stage closure. Those copies belong to the existing compute/live-stage model,
/// so only the single foreground loader is charged here.
pub fn host_weight_residency_charges(
    policy: &ExecutionPolicy,
    inventory: &CacheInventory,
) -> Result<CacheResidencyEstimate> {
    cache_charge(inventory, policy.weight_source(), policy.cache_policy()?)
}
