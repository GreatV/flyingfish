//! Metadata-only storage units and conservative cache-lifetime accounting.
//!
//! Logical mmap storage is deliberately separate from measured physical RSS.
//! Callers must supply concurrent-load and borrowed-storage lifetime bounds.
use super::{CacheGranularity, CachePolicy, ModelWeights, WeightSource, validate_shard_name};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheShardInventory {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub selected_tensor_bytes: u64,
    pub selected_tensor_count: u64,
    pub largest_tensor_bytes: u64,
}

/// Actual units the caller can access. Shard mode necessarily loads the whole
/// file, including headers and tensors outside a selected subset.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheInventory {
    pub shards: Vec<CacheShardInventory>,
}

impl CacheInventory {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.shards.is_empty(), "cache access inventory is empty");
        let mut names = BTreeSet::new();
        for shard in &self.shards {
            validate_shard_name(&shard.name)?;
            ensure!(names.insert(&shard.name), "duplicate cache inventory shard");
            ensure!(
                shard.header_bytes >= 8,
                "invalid cache inventory header size"
            );
            let payload = shard
                .file_bytes
                .checked_sub(shard.header_bytes)
                .context("cache inventory header exceeds file length")?;
            ensure!(
                shard.selected_tensor_count > 0,
                "cache inventory shard has no selected tensors"
            );
            ensure!(
                shard.selected_tensor_bytes <= payload,
                "selected tensors exceed shard payload"
            );
            ensure!(
                shard.largest_tensor_bytes <= shard.selected_tensor_bytes,
                "largest tensor exceeds selected bytes"
            );
            ensure!(
                u128::from(shard.selected_tensor_bytes)
                    <= u128::from(shard.largest_tensor_bytes)
                        * u128::from(shard.selected_tensor_count),
                "cache inventory tensor count cannot cover selected bytes"
            );
        }
        self.total_bytes(CacheGranularity::Shard)?;
        self.total_bytes(CacheGranularity::Tensor)?;
        self.header_bytes()?;
        Ok(())
    }

    pub fn total_bytes(&self, granularity: CacheGranularity) -> Result<u64> {
        self.shards.iter().try_fold(0_u64, |total, shard| {
            total
                .checked_add(shard.bytes(granularity))
                .context("cache inventory byte total overflow")
        })
    }

    pub fn header_bytes(&self) -> Result<u64> {
        self.shards.iter().try_fold(0_u64, |total, shard| {
            total
                .checked_add(shard.header_bytes)
                .context("cache inventory header total overflow")
        })
    }

    pub fn largest_unit_bytes(&self, granularity: CacheGranularity) -> u64 {
        self.shards
            .iter()
            .map(|shard| match granularity {
                CacheGranularity::Shard => shard.file_bytes,
                CacheGranularity::Tensor => shard.largest_tensor_bytes,
            })
            .max()
            .unwrap_or(0)
    }
}

impl CacheShardInventory {
    fn bytes(&self, granularity: CacheGranularity) -> u64 {
        match granularity {
            CacheGranularity::Shard => self.file_bytes,
            CacheGranularity::Tensor => self.selected_tensor_bytes,
        }
    }
}

/// Additional raw-storage lifetimes outside the retained-cache bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheLoadLifetimes {
    /// Includes the foreground loader. A concurrent duplicate load still owns
    /// storage before discovering that another loader inserted the same unit.
    pub maximum_concurrent_loads: u64,
    /// Evicted units kept alive by callers beyond the loader itself.
    pub externally_held_bytes: u64,
    /// Mapping alignment/page tables or other measured per-cache overhead.
    /// Header catalogs and arithmetic staging are separate phase allocations.
    pub additional_storage_bytes: u64,
}

impl CacheLoadLifetimes {
    pub const SERIAL: Self = Self {
        maximum_concurrent_loads: 1,
        externally_held_bytes: 0,
        additional_storage_bytes: 0,
    };
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheResidencyEstimate {
    pub complete_set_fits: bool,
    pub total_unit_bytes: u64,
    pub largest_unit_bytes: u64,
    pub retained_bytes: u64,
    pub loading_bytes: u64,
    pub externally_held_bytes: u64,
    pub additional_storage_bytes: u64,
    pub peak_storage_bytes: u64,
    pub owned_weight_bytes: u64,
    pub mapped_weight_bytes: u64,
}

pub fn estimate_cache_residency(
    inventory: &CacheInventory,
    source: WeightSource,
    policy: CachePolicy,
    lifetimes: CacheLoadLifetimes,
) -> Result<CacheResidencyEstimate> {
    inventory.validate()?;
    policy.validate()?;
    ensure!(
        lifetimes.maximum_concurrent_loads > 0,
        "cache requires at least one loader"
    );
    let total = inventory.total_bytes(policy.granularity)?;
    let unit = inventory.largest_unit_bytes(policy.granularity);
    let mut source_bytes = inventory
        .shards
        .iter()
        .map(|shard| shard.bytes(policy.granularity))
        .collect::<Vec<_>>();
    source_bytes.sort_unstable_by(|a, b| b.cmp(a));
    let count_bound =
        source_bytes
            .into_iter()
            .take(policy.max_shards)
            .try_fold(0_u64, |sum, bytes| {
                sum.checked_add(bytes)
                    .context("count-bounded cache bytes overflow")
            })?;
    let retained = count_bound.min(policy.max_bytes.map_or(total, |bound| bound.max(unit)));
    let complete_set_fits = policy.max_shards >= inventory.shards.len()
        && policy.max_bytes.is_none_or(|bound| bound >= total);
    let loading_units = if complete_set_fits {
        lifetimes.maximum_concurrent_loads - 1
    } else {
        lifetimes.maximum_concurrent_loads
    };
    let loading_bytes = loading_units
        .checked_mul(unit)
        .context("cache concurrent-load bytes overflow")?;
    let peak = retained
        .checked_add(loading_bytes)
        .and_then(|v| v.checked_add(lifetimes.externally_held_bytes))
        .and_then(|v| v.checked_add(lifetimes.additional_storage_bytes))
        .context("cache peak storage bytes overflow")?;
    Ok(CacheResidencyEstimate {
        complete_set_fits,
        total_unit_bytes: total,
        largest_unit_bytes: unit,
        retained_bytes: retained,
        loading_bytes,
        externally_held_bytes: lifetimes.externally_held_bytes,
        additional_storage_bytes: lifetimes.additional_storage_bytes,
        peak_storage_bytes: peak,
        owned_weight_bytes: if source == WeightSource::Memory {
            peak
        } else {
            0
        },
        mapped_weight_bytes: if source == WeightSource::Mmap {
            peak
        } else {
            0
        },
    })
}

impl ModelWeights {
    /// Catalog all indexed ranges without mapping or reading weight payloads.
    /// Check total_size against actual headers instead of trusting the index.
    pub fn cache_inventory(&self) -> Result<CacheInventory> {
        let inventory = self.cache_inventory_for(self.tensor_names())?;
        ensure!(
            inventory.total_bytes(CacheGranularity::Tensor)? == self.indexed_payload_bytes(),
            "indexed payload byte total disagrees with tensor headers"
        );
        Ok(inventory)
    }

    /// Catalog a deduplicated request access set. The caller is responsible
    /// for including initialization/static/scales, not just recurrent experts.
    pub fn cache_inventory_for<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<CacheInventory> {
        let names = names.into_iter().collect::<BTreeSet<_>>();
        let mut shards = BTreeMap::<String, CacheShardInventory>::new();
        for name in names {
            let tensor = self.raw_tensor_metadata(name)?;
            let bytes = u64::try_from(tensor.bytes).context("tensor size exceeds u64")?;
            let header = self.cache.headers.get(&self.root.join(&tensor.shard))?;
            let shard = shards
                .entry(tensor.shard.clone())
                .or_insert_with(|| CacheShardInventory {
                    name: tensor.shard,
                    file_bytes: header.file_bytes,
                    header_bytes: header.payload_offset as u64,
                    selected_tensor_bytes: 0,
                    selected_tensor_count: 0,
                    largest_tensor_bytes: 0,
                });
            shard.selected_tensor_bytes = shard
                .selected_tensor_bytes
                .checked_add(bytes)
                .context("selected tensor byte total overflow")?;
            shard.selected_tensor_count = shard
                .selected_tensor_count
                .checked_add(1)
                .context("selected tensor count overflow")?;
            shard.largest_tensor_bytes = shard.largest_tensor_bytes.max(bytes);
        }
        let inventory = CacheInventory {
            shards: shards.into_values().collect(),
        };
        inventory.validate()?;
        Ok(inventory)
    }
}

#[cfg(test)]
mod tests;
