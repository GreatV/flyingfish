use super::expert_cache::{
    ExpertCache, ExpertCacheReplacementPolicy, ExpertCacheResize, ExpertCacheStats,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ExpertCacheLayout {
    #[default]
    #[value(help = "Divide the exact total byte budget across sparse layers")]
    PerLayerSplit,
    #[value(help = "Let all (layer, expert) projection entries share one byte pool")]
    SharedPool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertCacheManagerResize {
    pub previous_max_bytes: usize,
    pub new_max_bytes: usize,
    pub previous_resident_bytes: usize,
    pub new_resident_bytes: usize,
    pub evictions: u64,
}

/// Owns the routed-expert caches behind a topology-neutral total-budget API.
///
/// The model asks only for the cache serving a sparse layer and changes only a
/// total byte bound at token safe points. The manager, not the decode loop,
/// owns whether that bound is split or shared.
pub struct ExpertCacheManager {
    layout: ExpertCacheLayout,
    replacement: ExpertCacheReplacementPolicy,
    layer_to_cache: Vec<Option<usize>>,
    caches: Vec<ExpertCache>,
}

impl ExpertCacheManager {
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn for_owned_layers(
        num_layers: usize,
        sparse_layers: &[usize],
        total_max_bytes: usize,
        layout: ExpertCacheLayout,
        replacement: ExpertCacheReplacementPolicy,
    ) -> Result<Self> {
        if sparse_layers.is_empty() {
            ensure!(
                num_layers > 0 && total_max_bytes == 0,
                "a dense-only worker cannot retain routed experts"
            );
            return Ok(Self {
                layout,
                replacement,
                layer_to_cache: vec![None; num_layers],
                caches: vec![],
            });
        }
        Self::new(
            num_layers,
            sparse_layers,
            total_max_bytes,
            layout,
            replacement,
        )
    }
    pub fn new(
        num_layers: usize,
        sparse_layers: &[usize],
        total_max_bytes: usize,
        layout: ExpertCacheLayout,
        replacement: ExpertCacheReplacementPolicy,
    ) -> Result<Self> {
        ensure!(num_layers > 0, "expert cache manager requires model layers");
        ensure!(
            !sparse_layers.is_empty() && sparse_layers.len() <= num_layers,
            "expert cache manager requires 1..=num_layers sparse layers"
        );
        ensure!(
            sparse_layers.windows(2).all(|pair| pair[0] < pair[1]),
            "expert cache manager sparse layers must be strictly increasing"
        );
        ensure!(
            sparse_layers.iter().all(|&layer| layer < num_layers),
            "expert cache manager sparse layer is out of range"
        );
        let mut layer_to_cache = vec![None; num_layers];
        let caches = match layout {
            ExpertCacheLayout::PerLayerSplit => {
                let quotas = split_total_budget(total_max_bytes, sparse_layers.len())?;
                sparse_layers
                    .iter()
                    .copied()
                    .zip(quotas)
                    .enumerate()
                    .map(|(cache_index, (layer, quota))| {
                        layer_to_cache[layer] = Some(cache_index);
                        ExpertCache::with_replacement(quota, replacement)
                    })
                    .collect()
            }
            ExpertCacheLayout::SharedPool => {
                for &layer in sparse_layers {
                    layer_to_cache[layer] = Some(0);
                }
                vec![ExpertCache::with_replacement(total_max_bytes, replacement)]
            }
        };
        let manager = Self {
            layout,
            replacement,
            layer_to_cache,
            caches,
        };
        ensure!(
            manager.stats()?.max_bytes == total_max_bytes,
            "expert cache manager did not allocate its exact total budget"
        );
        Ok(manager)
    }

    pub fn layout(&self) -> ExpertCacheLayout {
        self.layout
    }

    pub fn replacement(&self) -> ExpertCacheReplacementPolicy {
        self.replacement
    }

    pub fn cache_for_layer(&self, layer: usize) -> Result<&ExpertCache> {
        let cache = self
            .layer_to_cache
            .get(layer)
            .copied()
            .flatten()
            .with_context(|| format!("GLM layer {layer} has no routed-expert cache"))?;
        self.caches
            .get(cache)
            .context("expert cache manager mapping is out of range")
    }

    pub fn stats(&self) -> Result<ExpertCacheStats> {
        self.caches
            .iter()
            .try_fold(ExpertCacheStats::default(), |mut total, cache| {
                let cache = cache.stats();
                total.bytes = total
                    .bytes
                    .checked_add(cache.bytes)
                    .context("GLM expert cache resident byte count overflow")?;
                total.entries = total
                    .entries
                    .checked_add(cache.entries)
                    .context("GLM expert cache entry count overflow")?;
                total.max_bytes = total
                    .max_bytes
                    .checked_add(cache.max_bytes)
                    .context("GLM expert cache budget overflow")?;
                total.hits = total
                    .hits
                    .checked_add(cache.hits)
                    .context("GLM expert cache hit count overflow")?;
                total.misses = total
                    .misses
                    .checked_add(cache.misses)
                    .context("GLM expert cache miss count overflow")?;
                total.evictions = total
                    .evictions
                    .checked_add(cache.evictions)
                    .context("GLM expert cache eviction count overflow")?;
                Ok(total)
            })
    }

    pub fn resize(&self, new_total_max_bytes: usize) -> Result<ExpertCacheManagerResize> {
        let before = self.stats()?;
        let resizes = if self.caches.is_empty() {
            ensure!(
                new_total_max_bytes == 0,
                "a dense-only worker cannot retain routed experts"
            );
            vec![]
        } else {
            match self.layout {
                ExpertCacheLayout::PerLayerSplit => {
                    let quotas = split_total_budget(new_total_max_bytes, self.caches.len())?;
                    self.caches
                        .iter()
                        .zip(quotas)
                        .map(|(cache, quota)| cache.resize(quota))
                        .collect::<Vec<_>>()
                }
                ExpertCacheLayout::SharedPool => {
                    vec![self.caches[0].resize(new_total_max_bytes)]
                }
            }
        };
        let after = self.stats()?;
        ensure!(
            after.max_bytes == new_total_max_bytes,
            "expert cache manager resize did not apply its exact total budget"
        );
        validate_resize_sums(&resizes, before, after)?;
        Ok(ExpertCacheManagerResize {
            previous_max_bytes: before.max_bytes,
            new_max_bytes: after.max_bytes,
            previous_resident_bytes: before.bytes,
            new_resident_bytes: after.bytes,
            evictions: after
                .evictions
                .checked_sub(before.evictions)
                .context("expert cache eviction count moved backwards")?,
        })
    }
}

fn validate_resize_sums(
    resizes: &[ExpertCacheResize],
    before: ExpertCacheStats,
    after: ExpertCacheStats,
) -> Result<()> {
    let sum = |field: fn(&ExpertCacheResize) -> usize| -> Result<usize> {
        resizes.iter().try_fold(0usize, |total, resize| {
            total
                .checked_add(field(resize))
                .context("expert cache resize byte sum overflow")
        })
    };
    ensure!(
        sum(|resize| resize.previous_max_bytes)? == before.max_bytes
            && sum(|resize| resize.new_max_bytes)? == after.max_bytes
            && sum(|resize| resize.previous_resident_bytes)? == before.bytes
            && sum(|resize| resize.new_resident_bytes)? == after.bytes,
        "expert cache manager resize summary is inconsistent"
    );
    let evictions = resizes.iter().try_fold(0u64, |total, resize| {
        total
            .checked_add(resize.evictions)
            .context("expert cache resize eviction sum overflow")
    })?;
    let eviction_delta = after
        .evictions
        .checked_sub(before.evictions)
        .context("expert cache manager eviction counter moved backwards")?;
    ensure!(
        evictions == eviction_delta,
        "expert cache manager resize eviction summary is inconsistent"
    );
    Ok(())
}

pub(crate) fn split_total_budget(total: usize, parts: usize) -> Result<Vec<usize>> {
    ensure!(parts > 0, "cannot split expert cache across zero layers");
    let base = total / parts;
    let remainder = total % parts;
    let quotas = (0..parts)
        .map(|index| base + usize::from(index < remainder))
        .collect::<Vec<_>>();
    ensure!(
        quotas
            .iter()
            .try_fold(0usize, |sum, &quota| sum.checked_add(quota))
            == Some(total),
        "expert cache split does not preserve its total byte budget"
    );
    Ok(quotas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_layer_budgets_exclude_other_ranks_and_dense_only_workers() {
        let cache = ExpertCacheManager::for_owned_layers(
            45,
            &[12, 13],
            101,
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheReplacementPolicy::Lru,
        )
        .unwrap();
        assert_eq!(cache.cache_for_layer(12).unwrap().stats().max_bytes, 51);
        assert_eq!(cache.cache_for_layer(13).unwrap().stats().max_bytes, 50);
        assert!(cache.cache_for_layer(11).is_err());
        for layout in [
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheLayout::SharedPool,
        ] {
            let empty = ExpertCacheManager::for_owned_layers(
                45,
                &[],
                0,
                layout,
                ExpertCacheReplacementPolicy::Lru,
            )
            .unwrap();
            assert_eq!(empty.stats().unwrap().max_bytes, 0);
            empty.resize(0).unwrap();
            assert!(empty.resize(1).is_err());
            assert!(
                ExpertCacheManager::for_owned_layers(
                    45,
                    &[],
                    1,
                    layout,
                    ExpertCacheReplacementPolicy::Lru
                )
                .is_err()
            );
        }
    }
    use candle_core::{DType, Device, Tensor};

    fn tensor() -> Tensor {
        Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn layouts_receive_the_same_exact_total_in_different_topologies() {
        for layout in [
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheLayout::SharedPool,
        ] {
            let manager = ExpertCacheManager::new(
                5,
                &[1, 3, 4],
                11,
                layout,
                ExpertCacheReplacementPolicy::Lru,
            )
            .unwrap();
            assert_eq!(manager.stats().unwrap().max_bytes, 11);
            assert!(manager.cache_for_layer(0).is_err());
            for layer in [1, 3, 4] {
                assert!(manager.cache_for_layer(layer).is_ok());
            }
        }
    }

    #[test]
    fn shared_pool_can_borrow_bytes_and_never_aliases_layer_keys() {
        let manager = ExpertCacheManager::new(
            2,
            &[0, 1],
            8,
            ExpertCacheLayout::SharedPool,
            ExpertCacheReplacementPolicy::Lru,
        )
        .unwrap();
        assert!(
            manager
                .cache_for_layer(0)
                .unwrap()
                .insert("layer.0.expert".to_owned(), tensor())
        );
        assert!(
            manager
                .cache_for_layer(1)
                .unwrap()
                .insert("layer.1.expert".to_owned(), tensor())
        );
        assert_eq!(manager.stats().unwrap().entries, 2);
        assert!(
            manager
                .cache_for_layer(0)
                .unwrap()
                .get("layer.0.expert")
                .is_some()
        );
        assert!(
            manager
                .cache_for_layer(1)
                .unwrap()
                .get("layer.1.expert")
                .is_some()
        );
    }

    #[test]
    fn total_resize_is_synchronous_and_topology_neutral() {
        for layout in [
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheLayout::SharedPool,
        ] {
            let manager =
                ExpertCacheManager::new(2, &[0, 1], 16, layout, ExpertCacheReplacementPolicy::Lfu)
                    .unwrap();
            for layer in [0, 1] {
                let cache = manager.cache_for_layer(layer).unwrap();
                assert!(cache.insert(format!("layer.{layer}.a"), tensor()));
                assert!(cache.insert(format!("layer.{layer}.b"), tensor()));
            }
            let resize = manager.resize(4).unwrap();
            assert_eq!(resize.previous_max_bytes, 16);
            assert_eq!(resize.new_max_bytes, 4);
            assert_eq!(resize.previous_resident_bytes, 16);
            let expected_resident = if layout == ExpertCacheLayout::SharedPool {
                4
            } else {
                0
            };
            let expected_evictions = if layout == ExpertCacheLayout::SharedPool {
                3
            } else {
                4
            };
            assert_eq!(resize.new_resident_bytes, expected_resident);
            assert_eq!(resize.evictions, expected_evictions);
            assert_eq!(manager.stats().unwrap().bytes, expected_resident);

            let grow = manager.resize(12).unwrap();
            assert_eq!(grow.previous_max_bytes, 4);
            assert_eq!(grow.new_max_bytes, 12);
            assert_eq!(grow.previous_resident_bytes, expected_resident);
            assert_eq!(grow.new_resident_bytes, expected_resident);
            assert_eq!(grow.evictions, 0);
        }
    }

    #[test]
    fn invalid_sparse_layer_maps_fail_closed() {
        assert!(
            ExpertCacheManager::new(
                3,
                &[1, 1],
                8,
                ExpertCacheLayout::PerLayerSplit,
                ExpertCacheReplacementPolicy::Lru,
            )
            .is_err()
        );
        assert!(
            ExpertCacheManager::new(
                3,
                &[3],
                8,
                ExpertCacheLayout::SharedPool,
                ExpertCacheReplacementPolicy::Lru,
            )
            .is_err()
        );
    }
}
