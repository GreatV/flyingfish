//! Raw host weight storage under the same cache policy used for execution.
use crate::policy::ExecutionPolicy;
use anyhow::{Context, Result};
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

/// Return the cache ceiling and host peak, including the incoming unit and fixed allocations.
pub fn fit_host_cache(
    inventory: &CacheInventory,
    policy: CachePolicy,
    budget: u64,
    fixed: u64,
) -> Result<Option<(u64, u64)>> {
    let total = inventory.total_bytes(policy.granularity)?;
    let complete = total
        .div_ceil(1 << 20)
        .checked_mul(1 << 20)
        .context("host cache ceiling overflow")?;
    let ceiling = policy.max_bytes.unwrap_or(complete).min(complete);
    let peak = |mib| -> Result<u64> {
        cache_charge(
            inventory,
            WeightSource::Memory,
            policy.with_max_bytes(mib << 20),
        )?
        .owned_weight_bytes
        .checked_add(fixed)
        .context("host cache peak overflow")
    };
    let mut high = ceiling >> 20;
    if high == 0 {
        return Ok(None);
    }
    let full = peak(high)?;
    if full <= budget {
        return Ok(Some((high << 20, full)));
    }
    if peak(1)? > budget {
        return Ok(None);
    }
    let mut low = 1;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if peak(middle)? <= budget {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(Some((low << 20, peak(low)?)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::weights::accounting::CacheShardInventory;

    fn inventory() -> CacheInventory {
        CacheInventory {
            shards: (0..4)
                .map(|n| CacheShardInventory {
                    name: format!("shard-{n}.safetensors"),
                    file_bytes: 8 << 20,
                    header_bytes: 8,
                    selected_tensor_bytes: (8 << 20) - 8,
                    selected_tensor_count: 4,
                    largest_tensor_bytes: 2 << 20,
                })
                .collect(),
        }
    }

    #[test]
    fn full_cache_does_not_charge_an_extra_incoming_shard() -> Result<()> {
        let fit = fit_host_cache(
            &inventory(),
            CachePolicy::unbounded_units(),
            35 << 20,
            3 << 20,
        )?;
        assert_eq!(fit, Some((32 << 20, 35 << 20)));
        Ok(())
    }

    #[test]
    fn partial_cache_reserves_the_incoming_shard_and_fixed_allocations() -> Result<()> {
        let fit = fit_host_cache(
            &inventory(),
            CachePolicy::unbounded_units(),
            30 << 20,
            3 << 20,
        )?;
        assert_eq!(fit, Some((19 << 20, 30 << 20)));
        Ok(())
    }

    #[test]
    fn tensor_units_can_fit_when_two_shards_cannot() -> Result<()> {
        let policy = CachePolicy::unbounded_units();
        assert_eq!(
            fit_host_cache(&inventory(), policy, 10 << 20, 3 << 20)?,
            None
        );
        let fit = fit_host_cache(
            &inventory(),
            policy.with_granularity(CacheGranularity::Tensor),
            10 << 20,
            3 << 20,
        )?;
        assert_eq!(fit, Some((4 << 20, (9 << 20) + 8)));
        Ok(())
    }

    #[test]
    fn full_tensor_cache_rounds_up_to_cover_the_last_partial_mib() -> Result<()> {
        let policy = CachePolicy::unbounded_units().with_granularity(CacheGranularity::Tensor);
        let fit = fit_host_cache(&inventory(), policy, 35 << 20, 3 << 20)?;
        assert_eq!(fit, Some((32 << 20, (35 << 20) - 24)));
        Ok(())
    }

    #[test]
    fn refuses_when_even_tensor_units_exceed_the_budget() -> Result<()> {
        let policy = CachePolicy::unbounded_units().with_granularity(CacheGranularity::Tensor);
        assert_eq!(
            fit_host_cache(&inventory(), policy, 7 << 20, 3 << 20)?,
            None
        );
        Ok(())
    }

    #[test]
    fn rounds_capacity_down_and_honors_the_requested_ceiling() -> Result<()> {
        let policy = CachePolicy::unbounded_units().with_max_bytes((18 << 20) + 17);
        let fit = fit_host_cache(&inventory(), policy, 40 << 20, 3 << 20)?;
        assert_eq!(fit, Some((18 << 20, 29 << 20)));
        Ok(())
    }
}
