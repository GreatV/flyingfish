use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ExpertCacheReplacementPolicy {
    #[default]
    #[value(help = "Evict the least recently accessed resident projection")]
    Lru,
    #[value(help = "Retain frequent projections; reject colder newcomers under pressure")]
    Lfu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertCacheResize {
    pub previous_max_bytes: usize,
    pub new_max_bytes: usize,
    pub previous_resident_bytes: usize,
    pub new_resident_bytes: usize,
    pub evictions: u64,
}

/// A snapshot of an [`ExpertCache`]'s residency and access counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertCacheStats {
    /// Logical tensor bytes currently retained by the cache.
    pub bytes: usize,
    /// Number of tensors currently retained by the cache.
    pub entries: usize,
    /// Configured upper bound for `bytes`.
    pub max_bytes: usize,
    /// Successful calls to [`ExpertCache::get`].
    pub hits: u64,
    /// Unsuccessful calls to [`ExpertCache::get`].
    pub misses: u64,
    /// Entries removed to make room for an insertion.
    pub evictions: u64,
}

/// A thread-safe, byte-bounded least-recently-used cache of expert tensors.
///
/// Tensor handles are cheap to clone, so a hit returns a cloned handle while
/// keeping the cached tensor resident. The byte charge is the tensor's logical
/// element count multiplied by its dtype size. A zero byte budget disables the
/// cache, and a tensor larger than the configured budget is never retained.
pub struct ExpertCache {
    replacement: ExpertCacheReplacementPolicy,
    state: Mutex<State>,
}

/// Where a resident entry sits in its policy's eviction order, lowest first.
///
/// LRU ranks by access clock alone. LFU ranks by frequency and breaks ties on
/// that same clock, so the colder of two equally frequent projections leaves
/// first. Both policies therefore share one ordered index whose first element
/// is always the next victim, which is what keeps eviction off a linear scan.
type Rank = (u64, u64);

struct State {
    max_bytes: usize,
    entries: HashMap<Arc<str>, Entry>,
    /// Every resident entry filed under its own `Rank`, so the least valuable
    /// one is `first` rather than a search. Ranks are unique because each ends
    /// in an access clock handed out exactly once, which is what lets a reorder
    /// carry the shared key across without touching a reference count.
    order: BTreeMap<Rank, Arc<str>>,
    bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    clock: u64,
    /// LFU access history, retained past eviction so a re-admitted projection
    /// keeps the frequency it earned.
    history: HashMap<Arc<str>, AccessHistory>,
}

struct Entry {
    tensor: Tensor,
    bytes: usize,
    rank: Rank,
}

#[derive(Clone, Copy, Debug, Default)]
struct AccessHistory {
    frequency: u64,
    last_access: u64,
}

impl ExpertCache {
    pub fn new(max_bytes: usize) -> Self {
        Self::with_replacement(max_bytes, ExpertCacheReplacementPolicy::Lru)
    }

    pub fn with_replacement(max_bytes: usize, replacement: ExpertCacheReplacementPolicy) -> Self {
        Self {
            replacement,
            state: Mutex::new(State {
                max_bytes,
                entries: HashMap::new(),
                order: BTreeMap::new(),
                bytes: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
                clock: 0,
                history: HashMap::new(),
            }),
        }
    }

    /// Returns a cloned tensor handle and marks `key` as most recently used.
    pub fn get(&self, key: &str) -> Option<Tensor> {
        let mut state = self.state.lock().expect("expert cache mutex poisoned");
        state.record_access(key, self.replacement);
        let tensor = match state.entries.get(key) {
            Some(entry) => entry.tensor.clone(),
            None => {
                state.misses = state
                    .misses
                    .checked_add(1)
                    .expect("expert cache miss counter overflow");
                return None;
            }
        };

        state.hits = state
            .hits
            .checked_add(1)
            .expect("expert cache hit counter overflow");
        state.touch(key, self.replacement);
        Some(tensor)
    }

    /// Inserts `tensor`, returning whether it was retained by the cache.
    ///
    /// Replacing a key does not itself count as an eviction. If the replacement
    /// cannot be cached, any older value for that key is removed so a later
    /// lookup cannot return stale expert weights.
    pub fn insert(&self, key: String, tensor: Tensor) -> bool {
        let tensor_bytes = tensor
            .elem_count()
            .checked_mul(tensor.dtype().size_in_bytes())
            .expect("expert cache tensor byte count overflow");
        let mut state = self.state.lock().expect("expert cache mutex poisoned");

        state.remove(&key);
        if state.max_bytes == 0 || tensor_bytes > state.max_bytes {
            return false;
        }

        if self.replacement == ExpertCacheReplacementPolicy::Lru {
            while state.bytes > state.max_bytes - tensor_bytes {
                state.evict();
            }
        }

        let key = state.interned(&key);
        let rank = match self.replacement {
            ExpertCacheReplacementPolicy::Lru => {
                state.clock = state
                    .clock
                    .checked_add(1)
                    .expect("expert cache access clock overflow");
                (0, state.clock)
            }
            ExpertCacheReplacementPolicy::Lfu => {
                state.clock = state
                    .clock
                    .checked_add(1)
                    .expect("expert cache access clock overflow");
                let admitted = state.clock;
                let history = *state
                    .history
                    .entry(Arc::clone(&key))
                    .or_insert(AccessHistory {
                        frequency: 0,
                        last_access: admitted,
                    });
                (history.frequency, history.last_access)
            }
        };

        state.bytes = state
            .bytes
            .checked_add(tensor_bytes)
            .expect("expert cache resident byte count overflow");
        let ordered = state.order.insert(rank, Arc::clone(&key));
        debug_assert!(ordered.is_none());
        let replaced = state.entries.insert(
            Arc::clone(&key),
            Entry {
                tensor,
                bytes: tensor_bytes,
                rank,
            },
        );
        debug_assert!(replaced.is_none());
        while state.bytes > state.max_bytes {
            state.evict();
        }
        state.entries.contains_key(&key)
    }

    pub fn stats(&self) -> ExpertCacheStats {
        let state = self.state.lock().expect("expert cache mutex poisoned");
        ExpertCacheStats {
            bytes: state.bytes,
            entries: state.entries.len(),
            max_bytes: state.max_bytes,
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
        }
    }

    /// Change the performance-only residency bound and synchronously evict
    /// entries until the realized occupancy satisfies it.
    pub fn resize(&self, new_max_bytes: usize) -> ExpertCacheResize {
        let mut state = self.state.lock().expect("expert cache mutex poisoned");
        let previous_max_bytes = state.max_bytes;
        let previous_resident_bytes = state.bytes;
        let previous_evictions = state.evictions;
        state.max_bytes = new_max_bytes;
        while state.bytes > state.max_bytes {
            state.evict();
        }
        ExpertCacheResize {
            previous_max_bytes,
            new_max_bytes,
            previous_resident_bytes,
            new_resident_bytes: state.bytes,
            evictions: state
                .evictions
                .checked_sub(previous_evictions)
                .expect("expert cache eviction counter moved backwards"),
        }
    }
}

impl State {
    /// The shared key for `key`, reusing the allocation an existing resident
    /// or remembered access already holds.
    fn interned(&self, key: &str) -> Arc<str> {
        self.entries
            .get_key_value(key)
            .map(|(key, _)| key)
            .or_else(|| self.history.get_key_value(key).map(|(key, _)| key))
            .map_or_else(|| Arc::from(key), Arc::clone)
    }

    /// Move `key` to `rank`, keeping `order` and the entry itself in step.
    ///
    /// A key that is not resident has no place in the order; LFU still records
    /// its access history, which is what a later admission ranks it by.
    fn rerank(&mut self, key: &str, rank: Rank) {
        let Some(entry) = self.entries.get_mut(key) else {
            return;
        };
        let previous = std::mem::replace(&mut entry.rank, rank);
        if previous == rank {
            return;
        }
        let Some(stored) = self.order.remove(&previous) else {
            debug_assert!(false, "expert cache order index is missing a resident key");
            return;
        };
        let replaced = self.order.insert(rank, stored);
        debug_assert!(replaced.is_none(), "expert cache order index reused a rank");
    }

    fn record_access(&mut self, key: &str, replacement: ExpertCacheReplacementPolicy) {
        if replacement == ExpertCacheReplacementPolicy::Lru {
            return;
        }
        self.clock = self
            .clock
            .checked_add(1)
            .expect("expert cache access clock overflow");
        let clock = self.clock;
        let interned = self.interned(key);
        let history = self.history.entry(interned).or_default();
        history.frequency = history
            .frequency
            .checked_add(1)
            .expect("expert cache frequency overflow");
        history.last_access = clock;
        let rank = (history.frequency, history.last_access);
        self.rerank(key, rank);
    }

    /// Mark a hit. LFU already reranked the entry when it recorded the access,
    /// so only LRU has anything left to do here.
    fn touch(&mut self, key: &str, replacement: ExpertCacheReplacementPolicy) {
        if replacement != ExpertCacheReplacementPolicy::Lru {
            return;
        }
        self.clock = self
            .clock
            .checked_add(1)
            .expect("expert cache access clock overflow");
        self.rerank(key, (0, self.clock));
    }

    fn remove(&mut self, key: &str) {
        let Some(entry) = self.entries.remove(key) else {
            return;
        };
        let removed = self.order.remove(&entry.rank);
        debug_assert!(
            removed.is_some(),
            "expert cache order index is missing a resident key"
        );
        self.bytes -= entry.bytes;
    }

    fn evict(&mut self) {
        let (rank, key) = self
            .order
            .pop_first()
            .expect("expert cache cannot exceed its budget while empty");
        let entry = self
            .entries
            .remove(&key)
            .expect("expert cache order index contains a non-resident key");
        debug_assert_eq!(entry.rank, rank);
        self.bytes -= entry.bytes;
        self.evictions = self
            .evictions
            .checked_add(1)
            .expect("expert cache eviction counter overflow");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::sync::Arc;

    fn f32_tensor(elements: usize) -> Tensor {
        Tensor::zeros(elements, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn tracks_logical_bytes_hits_and_misses() {
        let cache = ExpertCache::new(32);
        let bf16 = Tensor::zeros(8, DType::BF16, &Device::Cpu).unwrap();

        assert!(cache.insert("expert".to_owned(), bf16));
        assert_eq!(cache.get("expert").unwrap().dims(), [8]);
        assert!(cache.get("missing").is_none());
        assert_eq!(
            cache.stats(),
            ExpertCacheStats {
                bytes: 16,
                entries: 1,
                max_bytes: 32,
                hits: 1,
                misses: 1,
                evictions: 0,
            }
        );
    }

    #[test]
    fn refreshes_recency_and_evicts_the_least_recent_entry() {
        let cache = ExpertCache::new(8);
        assert!(cache.insert("a".to_owned(), f32_tensor(1)));
        assert!(cache.insert("b".to_owned(), f32_tensor(1)));

        assert!(cache.get("a").is_some());
        assert!(cache.insert("c".to_owned(), f32_tensor(1)));

        assert!(cache.get("b").is_none());
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
        assert_eq!(
            cache.stats(),
            ExpertCacheStats {
                bytes: 8,
                entries: 2,
                max_bytes: 8,
                hits: 3,
                misses: 1,
                evictions: 1,
            }
        );
    }

    #[test]
    fn evicts_multiple_entries_until_the_new_tensor_fits() {
        let cache = ExpertCache::new(12);
        for key in ["a", "b", "c"] {
            assert!(cache.insert(key.to_owned(), f32_tensor(1)));
        }

        assert!(cache.insert("large".to_owned(), f32_tensor(2)));
        assert_eq!(
            cache.stats(),
            ExpertCacheStats {
                bytes: 12,
                entries: 2,
                max_bytes: 12,
                hits: 0,
                misses: 0,
                evictions: 2,
            }
        );
        assert!(cache.get("c").is_some());
        assert!(cache.get("large").is_some());
    }

    #[test]
    fn oversize_replacement_is_not_cached_or_counted_as_an_eviction() {
        let cache = ExpertCache::new(8);
        assert!(cache.insert("expert".to_owned(), f32_tensor(1)));

        assert!(!cache.insert("expert".to_owned(), f32_tensor(3)));
        assert!(cache.get("expert").is_none());
        assert_eq!(
            cache.stats(),
            ExpertCacheStats {
                bytes: 0,
                entries: 0,
                max_bytes: 8,
                hits: 0,
                misses: 1,
                evictions: 0,
            }
        );
    }

    #[test]
    fn zero_budget_disables_the_cache() {
        let cache = ExpertCache::new(0);
        assert!(!cache.insert("expert".to_owned(), f32_tensor(1)));
        assert!(cache.get("expert").is_none());
        assert_eq!(
            cache.stats(),
            ExpertCacheStats {
                bytes: 0,
                entries: 0,
                max_bytes: 0,
                hits: 0,
                misses: 1,
                evictions: 0,
            }
        );
    }

    #[test]
    fn lfu_retains_frequency_across_eviction_and_breaks_ties_by_recency() {
        let cache = ExpertCache::with_replacement(8, ExpertCacheReplacementPolicy::Lfu);
        assert!(cache.get("a").is_none());
        assert!(cache.insert("a".to_owned(), f32_tensor(1)));
        assert!(cache.get("b").is_none());
        assert!(cache.insert("b".to_owned(), f32_tensor(1)));
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_none());
        assert!(cache.insert("c".to_owned(), f32_tensor(1)));
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
        assert!(cache.get("cold").is_none());
        assert!(!cache.insert("cold".to_owned(), f32_tensor(1)));
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn resize_synchronously_satisfies_a_smaller_bound() {
        let cache = ExpertCache::new(12);
        for key in ["a", "b", "c"] {
            assert!(cache.insert(key.to_owned(), f32_tensor(1)));
        }
        assert!(cache.get("a").is_some());
        let resized = cache.resize(8);
        assert_eq!(resized.previous_max_bytes, 12);
        assert_eq!(resized.new_max_bytes, 8);
        assert_eq!(resized.previous_resident_bytes, 12);
        assert_eq!(resized.new_resident_bytes, 8);
        assert_eq!(resized.evictions, 1);
        assert!(cache.get("b").is_none());
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());

        let grown = cache.resize(16);
        assert_eq!(grown.previous_max_bytes, 8);
        assert_eq!(grown.new_max_bytes, 16);
        assert_eq!(grown.previous_resident_bytes, 8);
        assert_eq!(grown.new_resident_bytes, 8);
        assert_eq!(grown.evictions, 0);
    }

    /// The ordered index must hold exactly the resident keys, each filed under
    /// the rank its own entry records.
    ///
    /// A reorder that removed the wrong rank, or failed to remove the old one,
    /// leaves the cache still evicting something on every admission — just the
    /// wrong entry, silently and only under load. Nothing observable through
    /// `stats` catches that, so the invariant is asserted directly.
    fn assert_index_is_consistent(cache: &ExpertCache) {
        let state = cache.state.lock().unwrap();
        assert_eq!(
            state.order.len(),
            state.entries.len(),
            "order index and resident set disagree on size"
        );
        for (rank, key) in &state.order {
            let entry = state
                .entries
                .get(key)
                .expect("order index names a non-resident key");
            assert_eq!(
                entry.rank, *rank,
                "order index holds a stale rank for {key}"
            );
        }
        let bytes = state
            .entries
            .values()
            .map(|entry| entry.bytes)
            .sum::<usize>();
        assert_eq!(bytes, state.bytes, "resident byte total drifted");
        assert!(state.bytes <= state.max_bytes, "cache is over its budget");
    }

    #[test]
    fn interleaved_access_keeps_the_order_index_in_step_under_both_policies() {
        for replacement in [
            ExpertCacheReplacementPolicy::Lru,
            ExpertCacheReplacementPolicy::Lfu,
        ] {
            let cache = ExpertCache::with_replacement(24, replacement);
            let keys = ["a", "b", "c", "d", "e", "f"];
            let mut step = 1u64;
            for round in 0..200u64 {
                step = step.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = keys[(step >> 33) as usize % keys.len()];
                if round % 3 == 0 {
                    cache.insert(key.to_owned(), f32_tensor(1));
                } else {
                    cache.get(key);
                }
                assert_index_is_consistent(&cache);
            }
            cache.resize(8);
            assert_index_is_consistent(&cache);
            cache.resize(40);
            assert_index_is_consistent(&cache);
        }
    }

    #[test]
    fn eviction_follows_access_order_rather_than_insertion_order() {
        let cache = ExpertCache::new(12);
        for key in ["a", "b", "c"] {
            assert!(cache.insert(key.to_owned(), f32_tensor(1)));
        }

        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_some());
        assert!(cache.insert("d".to_owned(), f32_tensor(1)));

        assert!(
            cache.get("c").is_none(),
            "the least recently used key survived"
        );
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_some());
        assert!(cache.get("d").is_some());

        assert!(cache.insert("e".to_owned(), f32_tensor(1)));
        assert!(
            cache.get("a").is_none(),
            "recency did not advance past the oldest hit"
        );
        assert_index_is_consistent(&cache);
    }

    #[test]
    fn access_counters_are_thread_safe() {
        let cache = Arc::new(ExpertCache::new(4));
        assert!(cache.insert("expert".to_owned(), f32_tensor(1)));
        let threads = (0..4)
            .map(|_| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        assert!(cache.get("expert").is_some());
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(cache.stats().hits, 400);
        assert_eq!(cache.stats().misses, 0);
    }
}
