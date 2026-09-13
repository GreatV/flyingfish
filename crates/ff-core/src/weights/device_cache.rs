//! Device-resident tensor retention: the top tier of the residency ladder. A
//! hit returns the tensor exactly as the loader produced it — same bytes, dtype
//! and device — so caching cannot change an output. The byte charge is the
//! tensor's logical element count multiplied by its dtype size. A zero byte
//! budget — the default — disables the cache, and a tensor larger than the
//! budget is never retained.
//!
//! One [`DeviceCache`] is a single ceiling shared by every store attached to
//! it, **applied to each device separately**. A pipeline that opens several
//! checkpoints hands the same handle to all of them, so the operator's ceiling
//! bounds the request rather than each component separately; a process driving
//! several devices spends that ceiling on each of them, because a ceiling one
//! device's worth of memory wide, divided among four devices, is not what
//! `--device-cache-mib` means. Retention, eviction and the priority reserve are
//! therefore all scoped to a [`DeviceLocation`]; a single-device request sees
//! exactly the behaviour it saw before. Stores are told apart by the id
//! [`DeviceCache::attach`] returns, so two checkpoints that name a tensor alike
//! never collide.
use candle_core::{DType, Device, DeviceLocation, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CudaWeightAllocator {
    #[default]
    #[value(name = "pool")]
    StreamPool,
    /// Experimental: retain weights in direct allocations outside the stream
    /// pool. May synchronize uploads; requires workload-specific measurement.
    Direct,
}

impl CudaWeightAllocator {
    fn is_stream_pool(&self) -> bool {
        *self == Self::StreamPool
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceCachePolicy {
    /// Ceiling on retained device tensor bytes per device, shared by every
    /// attached store on that device. Zero disables retention.
    #[serde(default)]
    pub max_bytes: u64,
    #[serde(default, skip_serializing_if = "CudaWeightAllocator::is_stream_pool")]
    pub cuda_allocator: CudaWeightAllocator,
}

impl DeviceCachePolicy {
    pub const DISABLED: Self = Self {
        max_bytes: 0,
        cuda_allocator: CudaWeightAllocator::StreamPool,
    };

    pub const fn with_max_bytes(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            ..Self::DISABLED
        }
    }

    pub const fn with_cuda_allocator(mut self, allocator: CudaWeightAllocator) -> Self {
        self.cuda_allocator = allocator;
        self
    }

    pub const fn is_enabled(&self) -> bool {
        self.max_bytes > 0
    }
}

/// A snapshot of the device cache's residency and access counters, summed over
/// every store attached to it.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_tensors: usize,
    pub resident_bytes: u64,
    pub max_bytes: u64,
    /// Set once a device allocation failed with tensors resident. The cache
    /// released them and stopped retaining, so the request finishes from the
    /// host tiers. It is reported because a demoted cache and a useless cache
    /// look alike in the hit counters and are not the same problem.
    pub demoted: bool,
    /// Resident bytes currently protected by active phase priorities.
    #[serde(default)]
    pub prioritized_bytes: u64,
    /// Resident bytes per device, in device order. `max_bytes` bounds each
    /// entry rather than their sum, so a rank-aware request can see which
    /// device is full. A single-device request reports one entry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resident_bytes_by_device: Vec<DeviceResidentBytes>,
    /// Inserts bypassed because they would displace a selected phase.
    #[serde(default)]
    pub priority_bypasses: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub demotion: Option<DeviceCacheDemotion>,
}

/// One device's share of a [`DeviceCacheStats`]. The device is named the way
/// the demotion record names it, by its `DeviceLocation`, because two `Device`
/// handles for one location share storage and must not read as two devices.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceResidentBytes {
    pub device: String,
    pub resident_tensors: usize,
    pub resident_bytes: u64,
    pub prioritized_bytes: u64,
}

/// First failed device load or compute allocation and the cache occupancy
/// before release. CUDA counters are best-effort, unsynchronized observations,
/// not reservations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceCacheDemotion {
    pub tensor: String,
    pub device: String,
    pub requested_bytes: Option<u64>,
    pub error: String,
    pub resident_bytes_before: u64,
    pub prioritized_bytes_before: u64,
    pub cuda_free_bytes: Option<u64>,
    pub cuda_pool_used_bytes: Option<u64>,
    pub cuda_pool_reserved_bytes: Option<u64>,
}

impl DeviceCacheDemotion {
    pub(super) fn capture(
        name: &str,
        requested_bytes: Option<u64>,
        error: String,
        device: &Device,
    ) -> Self {
        let result = Self {
            tensor: name.into(),
            device: format!("{:?}", device.location()),
            requested_bytes,
            error,
            resident_bytes_before: 0,
            prioritized_bytes_before: 0,
            cuda_free_bytes: None,
            cuda_pool_used_bytes: None,
            cuda_pool_reserved_bytes: None,
        };
        #[cfg(feature = "cuda")]
        let mut result = result;
        #[cfg(feature = "cuda")]
        if let Ok(cuda) = device.as_cuda_device() {
            use candle_core::cuda_backend::cudarc::driver::{result as driver, sys};
            let stream = cuda.cuda_stream();
            let context = stream.context();
            if driver::ctx::get_current().ok().flatten() == Some(context.cu_ctx()) {
                result.cuda_free_bytes = driver::mem_get_info().ok().map(|(free, _)| free as u64);
                if let Ok(pool) = unsafe { driver::device::get_mem_pool(context.cu_device()) } {
                    let read = |attribute| {
                        let mut bytes = 0u64;
                        unsafe {
                            sys::cuMemPoolGetAttribute(
                                pool,
                                attribute,
                                (&mut bytes as *mut u64).cast(),
                            )
                        }
                        .result()
                        .ok()
                        .map(|_| bytes)
                    };
                    result.cuda_pool_used_bytes =
                        read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT);
                    result.cuda_pool_reserved_bytes =
                        read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
                }
            }
        }
        result
    }
}

/// Which attached store a key belongs to. Two checkpoints may name a tensor
/// alike; the id keeps their entries apart in one shared cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceCacheStore(u64);

/// Which axis of a rank-two tensor a split runs along. A checkpoint stores a
/// linear weight as `[out_features, in_features]` in row-major order, so a
/// column-parallel linear splits `Rows` (a contiguous byte range) and a
/// row-parallel linear splits `Columns` (a strided gather).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorAxis {
    Rows,
    Columns,
}

/// Which part of a named tensor an entry holds.
///
/// Under a tensor split two ranks hold different slices of one named tensor,
/// and today they fail to collide only because their device locations differ.
/// That is an accident, not a rule — one process could hold a slice and the
/// whole tensor on one device — and a cache that returned the slice to a caller
/// who asked for the whole tensor would silently change an output. So the
/// slice's identity is part of the key. `Whole` is what every existing loader
/// asks for, and is the only thing that can ever answer a whole-tensor read.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorPartition {
    #[default]
    Whole,
    /// Rank `rank` of `ranks`, along `axis`. The rank count is part of the
    /// identity because the same rank of a different-sized ring is a different
    /// slice.
    Shard {
        axis: TensorAxis,
        rank: u32,
        ranks: u32,
    },
    /// This rank's share of *each* of `segments` equal, contiguous segments of
    /// one axis, concatenated in segment order.
    ///
    /// A checkpoint may fuse several projections that a split has to divide
    /// separately. H3's feed-forward input is one `[2 × ffn_dim, hidden]`
    /// tensor holding the SwiGLU values above the gates, and a column-parallel
    /// split of it must take each rank's share of the values *and* its share of
    /// the gates: one contiguous range would hand the first rank nothing but
    /// values and the last nothing but gates, which is not a shard of that
    /// linear at all. The share stays in segment order, so a rank's own slice
    /// has the same internal layout the whole tensor has.
    SegmentedShard {
        axis: TensorAxis,
        rank: u32,
        ranks: u32,
        segments: u32,
    },
}

impl TensorPartition {
    /// Every half-open range of `length` this partition names, in the order the
    /// slice concatenates them, and an error when the split does not divide it.
    ///
    /// Refusing an uneven split is deliberate: a Megatron-shaped partition of a
    /// head-structured tensor is only correct when the ranks divide the heads,
    /// and silently rounding would produce a plausible, wrong slice.
    pub fn ranges(&self, length: usize) -> anyhow::Result<Vec<std::ops::Range<usize>>> {
        match self {
            Self::Whole => Ok(std::iter::once(0..length).collect()),
            Self::Shard { rank, ranks, .. } => {
                let share = Self::share(length, *rank, *ranks, 1)?;
                let rank = *rank as usize;
                Ok(std::iter::once(rank * share..(rank + 1) * share).collect())
            }
            Self::SegmentedShard {
                rank,
                ranks,
                segments,
                ..
            } => {
                let share = Self::share(length, *rank, *ranks, *segments)?;
                let (rank, segments) = (*rank as usize, *segments as usize);
                let segment = length / segments;
                Ok((0..segments)
                    .map(|index| {
                        let start = index * segment + rank * share;
                        start..start + share
                    })
                    .collect())
            }
        }
    }

    /// The elements of `length` this partition holds, summed over its ranges.
    pub fn extent(&self, length: usize) -> anyhow::Result<usize> {
        Ok(self
            .ranges(length)?
            .into_iter()
            .map(|range| range.end - range.start)
            .sum())
    }

    /// The single range this partition names, for a caller that cannot handle a
    /// segmented one. A segmented shard is several ranges by construction and
    /// is refused here rather than collapsed into a span that would include
    /// elements this rank does not hold.
    pub fn range(&self, length: usize) -> anyhow::Result<std::ops::Range<usize>> {
        let mut ranges = self.ranges(length)?;
        anyhow::ensure!(
            ranges.len() == 1,
            "a {}-segment shard names {} ranges, not one",
            match self {
                Self::SegmentedShard { segments, .. } => *segments,
                _ => 1,
            },
            ranges.len()
        );
        Ok(ranges.remove(0))
    }

    fn share(length: usize, rank: u32, ranks: u32, segments: u32) -> anyhow::Result<usize> {
        let (rank, ranks, segments) = (rank as usize, ranks as usize, segments as usize);
        anyhow::ensure!(ranks > 0, "a tensor partition needs at least one rank");
        anyhow::ensure!(
            segments > 0,
            "a tensor partition needs at least one segment"
        );
        anyhow::ensure!(rank < ranks, "rank {rank} is outside a {ranks}-rank split");
        anyhow::ensure!(
            length.is_multiple_of(segments),
            "a {segments}-segment tensor of {length} has unequal segments"
        );
        let segment = length / segments;
        anyhow::ensure!(
            segment.is_multiple_of(ranks),
            "a {ranks}-rank split does not divide {segment} evenly"
        );
        Ok(segment / ranks)
    }

    pub fn axis(&self) -> TensorAxis {
        match self {
            Self::Whole => TensorAxis::Rows,
            Self::Shard { axis, .. } | Self::SegmentedShard { axis, .. } => *axis,
        }
    }
}

/// Retention is keyed by what a load produces: the store it came from, the
/// tensor's name, the part of that tensor the entry holds, the device location
/// it was materialized on, and the dtype the loader returned. Two `Device`
/// handles naming one location share storage, so the location — not the
/// handle — is the key's device component.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Key {
    store: DeviceCacheStore,
    name: String,
    partition: TensorPartition,
    location: DeviceLocation,
    dtype: DType,
}

struct Entry {
    tensor: Option<Tensor>,
    bytes: u64,
    prioritized: bool,
    last_access: u64,
    #[cfg(feature = "cuda")]
    owner_thread: std::thread::ThreadId,
    #[cfg(feature = "cuda")]
    cross_thread_access: bool,
}

impl Drop for Entry {
    fn drop(&mut self) {
        let Some(tensor) = self.tensor.take() else {
            return;
        };
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = tensor.device() {
            use candle_core::cuda_backend::cudarc::driver::result;
            let stream = cuda.cuda_stream();
            let context = stream.context().clone();
            let owner = context.cu_ctx();
            let previous = result::ctx::get_current().ok();
            let bound =
                previous == Some(Some(owner)) || unsafe { result::ctx::set_current(owner) }.is_ok();
            let synchronize_default = bound
                && stream.cu_stream() as usize == 0x2
                && (self.cross_thread_access || self.owner_thread != std::thread::current().id());
            if synchronize_default {
                let _ = result::ctx::synchronize();
            }
            drop(tensor);
            if synchronize_default {
                let _ = unsafe { result::stream::synchronize(stream.cu_stream()) };
            }
            drop(stream);
            drop(context);
            if let Some(previous) = previous.filter(|previous| *previous != Some(owner)) {
                let _ =
                    unsafe { result::ctx::set_current(previous.unwrap_or(std::ptr::null_mut())) };
            }
            return;
        }
        drop(tensor);
    }
}

/// One device's occupancy. The ceiling applies here rather than to the sum, so
/// a rank on one device can neither be starved nor displaced by a rank on
/// another.
#[derive(Default)]
struct LocationState {
    /// Separate LRU orders for best-effort and selected weights, each keyed by
    /// the access clock so the victim is the first entry rather than a search.
    /// The clock is assigned under this lock and never reused, so it alone
    /// identifies a resident entry's place; the key rides along as the value.
    recency: [BTreeMap<u64, Arc<Key>>; 2],
    bytes: u64,
    prioritized_bytes: u64,
}

#[derive(Default)]
struct State {
    entries: HashMap<Arc<Key>, Entry>,
    locations: HashMap<DeviceLocation, LocationState>,
    /// Retained bytes summed over every device, reported but never compared
    /// against `max_bytes`.
    bytes: u64,
    /// The ceiling for each device, not for their sum.
    max_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    demoted: bool,
    priorities: HashMap<DeviceCacheStore, Arc<HashSet<String>>>,
    inactive_priorities: HashSet<DeviceCacheStore>,
    clock: u64,
    prioritized_bytes: u64,
    priority_bypasses: u64,
    demotion: Option<DeviceCacheDemotion>,
}

impl State {
    fn prioritized(&self, store: DeviceCacheStore, name: &str) -> bool {
        !self.inactive_priorities.contains(&store)
            && self
                .priorities
                .get(&store)
                .is_some_and(|names| names.contains(name))
    }

    fn location(&mut self, location: DeviceLocation) -> &mut LocationState {
        self.locations.entry(location).or_default()
    }

    fn location_bytes(&self, location: &DeviceLocation) -> u64 {
        self.locations
            .get(location)
            .map_or(0, |occupancy| occupancy.bytes)
    }

    fn location_prioritized_bytes(&self, location: &DeviceLocation) -> u64 {
        self.locations
            .get(location)
            .map_or(0, |occupancy| occupancy.prioritized_bytes)
    }

    fn make_room(
        &mut self,
        location: DeviceLocation,
        bytes: u64,
        limit: u64,
        prioritized: bool,
    ) -> bool {
        if bytes > limit
            || (!prioritized && self.location_prioritized_bytes(&location) > limit - bytes)
        {
            return false;
        }
        while self.location_bytes(&location) > limit - bytes {
            let evict_prioritized = self.location(location).recency[0].is_empty();
            assert!(
                !evict_prioritized || prioritized,
                "unselected admission cannot evict selected tensors"
            );
            self.evict_class(location, evict_prioritized);
        }
        true
    }

    /// Move a resident entry from `previous` to `next` on its device's order.
    ///
    /// The caller has already stamped `next` onto the entry, so this reorders
    /// the index alone and shares the stored key rather than allocating one.
    fn touch(&mut self, key: &Key, prioritized: bool, previous: u64, next: u64) {
        let queue = &mut self.location(key.location).recency[usize::from(prioritized)];
        let Some(stored) = queue.remove(&previous) else {
            debug_assert!(false, "device cache order is missing a resident key");
            return;
        };
        let replaced = queue.insert(next, stored);
        debug_assert!(
            replaced.is_none(),
            "device cache order reused an access clock"
        );
    }

    fn charge(&mut self, key: &Key, bytes: u64, prioritized: bool) {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .expect("device cache resident byte count overflow");
        let occupancy = self.location(key.location);
        occupancy.bytes = occupancy
            .bytes
            .checked_add(bytes)
            .expect("device cache resident byte count overflow");
        if prioritized {
            occupancy.prioritized_bytes += bytes;
            self.prioritized_bytes += bytes;
        }
    }

    fn discharge(&mut self, key: &Key, bytes: u64, prioritized: bool) {
        self.bytes = self
            .bytes
            .checked_sub(bytes)
            .expect("device cache byte count is inconsistent");
        if prioritized {
            self.prioritized_bytes = self
                .prioritized_bytes
                .checked_sub(bytes)
                .expect("priority byte count is inconsistent");
        }
        let occupancy = self.location(key.location);
        occupancy.bytes = occupancy
            .bytes
            .checked_sub(bytes)
            .expect("device cache byte count is inconsistent");
        if prioritized {
            occupancy.prioritized_bytes = occupancy
                .prioritized_bytes
                .checked_sub(bytes)
                .expect("priority byte count is inconsistent");
        }
    }

    fn remove(&mut self, key: &Key) {
        let Some(entry) = self.entries.remove(key) else {
            return;
        };
        self.discharge(key, entry.bytes, entry.prioritized);
        let queue = &mut self.location(key.location).recency[usize::from(entry.prioritized)];
        let removed = queue.remove(&entry.last_access);
        debug_assert!(
            removed.is_some(),
            "device cache recency index is missing a resident key"
        );
    }

    /// The device holding the least recently used best-effort entry, or, when
    /// no device holds one, the least recently used selected entry. Releasing
    /// everything walks every device this way rather than favouring one.
    fn eviction_target(&self) -> Option<(DeviceLocation, bool)> {
        let mut best: Option<(u64, DeviceLocation, bool)> = None;
        for prioritized in [false, true] {
            for (location, occupancy) in &self.locations {
                let Some((&age, _)) = occupancy.recency[usize::from(prioritized)].first_key_value()
                else {
                    continue;
                };
                if best.as_ref().is_none_or(|(oldest, _, _)| age < *oldest) {
                    best = Some((age, *location, prioritized));
                }
            }
            if let Some((_, location, prioritized)) = best {
                return Some((location, prioritized));
            }
        }
        None
    }

    /// Charge one entry against its device and record it. The caller has
    /// already removed any older entry under the same key and checked that the
    /// cache is enabled, undemoted, and wide enough to hold `bytes`. Returns
    /// whether the entry was admitted; a refusal counts a priority bypass.
    fn admit(&mut self, key: Key, bytes: u64, prioritized: bool, tensor: Option<Tensor>) -> bool {
        let max_bytes = self.max_bytes;
        if !self.make_room(key.location, bytes, max_bytes, prioritized) {
            self.priority_bypasses = self
                .priority_bypasses
                .checked_add(1)
                .expect("priority bypass count overflow");
            return false;
        }
        self.charge(&key, bytes, prioritized);
        self.clock = self
            .clock
            .checked_add(1)
            .expect("cache access clock overflow");
        let last_access = self.clock;
        let key = Arc::new(key);
        let replaced = self.location(key.location).recency[usize::from(prioritized)]
            .insert(last_access, Arc::clone(&key));
        debug_assert!(
            replaced.is_none(),
            "device cache order reused an access clock"
        );
        let replaced = self.entries.insert(
            key,
            Entry {
                tensor,
                bytes,
                prioritized,
                last_access,
                #[cfg(feature = "cuda")]
                owner_thread: std::thread::current().id(),
                #[cfg(feature = "cuda")]
                cross_thread_access: false,
            },
        );
        debug_assert!(replaced.is_none());
        true
    }

    /// Apply one ceiling to every device, evicting each device's own least
    /// recently used entries until it fits. A device already inside the bound
    /// loses nothing to a device that is over it.
    fn trim_each_location(&mut self, maximum_resident_bytes: u64) {
        let locations = self.locations.keys().copied().collect::<Vec<_>>();
        for location in locations {
            while self.location_bytes(&location) > maximum_resident_bytes {
                let prioritized = self.location(location).recency[0].is_empty();
                self.evict_class(location, prioritized);
            }
        }
    }

    fn evict(&mut self) {
        let (location, prioritized) = self
            .eviction_target()
            .expect("device cache cannot exceed its budget while empty");
        self.evict_class(location, prioritized);
    }

    fn evict_class(&mut self, location: DeviceLocation, prioritized: bool) {
        let (_, key) = self.location(location).recency[usize::from(prioritized)]
            .pop_first()
            .expect("device cache cannot exceed its budget while empty");
        let entry = self
            .entries
            .remove(&key)
            .expect("device cache recency index contains a non-resident key");
        self.discharge(&key, entry.bytes, entry.prioritized);
        self.evictions = self
            .evictions
            .checked_add(1)
            .expect("device cache eviction counter overflow");
    }
}

struct Inner {
    policy: DeviceCachePolicy,
    state: Mutex<State>,
    next_store: AtomicU64,
    selected_phases: HashSet<String>,
}

/// A shared device residency budget. Cloning shares one ceiling and one
/// resident set; every clone sees the same statistics and the same demotion.
#[derive(Clone)]
pub struct DeviceCache(Arc<Inner>);

impl DeviceCache {
    pub(super) fn is_prioritized(&self, store: DeviceCacheStore, name: &str) -> bool {
        self.lock().prioritized(store, name)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn can_retain(
        &self,
        store: DeviceCacheStore,
        name: &str,
        device: &Device,
        bytes: u64,
    ) -> bool {
        let state = self.lock();
        self.0.policy.is_enabled()
            && !state.demoted
            && bytes <= state.max_bytes
            && (state.prioritized(store, name)
                || state.location_prioritized_bytes(&device.location()) <= state.max_bytes - bytes)
    }
    pub fn new(policy: DeviceCachePolicy) -> Self {
        Self(Arc::new(Inner {
            policy,
            state: Mutex::new(State {
                max_bytes: policy.max_bytes,
                ..State::default()
            }),
            next_store: AtomicU64::new(0),
            selected_phases: HashSet::new(),
        }))
    }

    /// Protect selected phase tensors from unrelated best-effort admissions.
    /// Stores attach the tensor names belonging to these phases before loading.
    pub fn with_selected_phases(
        policy: DeviceCachePolicy,
        phases: impl IntoIterator<Item = String>,
    ) -> Self {
        let mut cache = Self::new(policy);
        if policy.is_enabled() {
            Arc::get_mut(&mut cache.0)
                .expect("new cache has one owner")
                .selected_phases = phases.into_iter().collect();
        }
        cache
    }

    pub fn phase_selected(&self, name: &str) -> bool {
        self.0.selected_phases.contains(name)
    }

    /// The streaming baseline: retains nothing, and is what every
    /// `ModelWeights` uses until one is configured.
    pub fn disabled() -> Self {
        Self::new(DeviceCachePolicy::DISABLED)
    }

    /// Register one weight store against this budget. Entries the returned id
    /// keys are distinct from every other store's, so sharing a cache across
    /// checkpoints cannot confuse two tensors that share a name.
    pub fn attach(&self) -> DeviceCacheStore {
        DeviceCacheStore(self.0.next_store.fetch_add(1, Ordering::Relaxed))
    }

    pub(super) fn attach_prioritized(&self, names: HashSet<String>) -> DeviceCacheStore {
        let store = self.attach();
        if self.0.policy.is_enabled() && !names.is_empty() {
            self.lock().priorities.insert(store, Arc::new(names));
        }
        store
    }

    pub(super) fn set_priority_active(&self, store: DeviceCacheStore, active: bool) {
        let mut state = self.lock();
        let Some(names) = state.priorities.get(&store).cloned() else {
            return;
        };
        let changed = if active {
            state.inactive_priorities.remove(&store)
        } else {
            state.inactive_priorities.insert(store)
        };
        if !changed {
            return;
        }
        let mut reclassify = false;
        for (key, entry) in &mut state.entries {
            if key.store == store {
                let prioritized = active && names.contains(&key.name);
                reclassify |= entry.prioritized != prioritized;
                entry.prioritized = prioritized;
            }
        }
        if !reclassify {
            return;
        }
        let keys = state.entries.keys().map(Arc::clone).collect::<Vec<_>>();
        for occupancy in state.locations.values_mut() {
            occupancy.recency.iter_mut().for_each(BTreeMap::clear);
            occupancy.prioritized_bytes = 0;
        }
        state.prioritized_bytes = 0;
        for key in keys {
            let entry = &state.entries[&key];
            let (prioritized, bytes, last_access) =
                (entry.prioritized, entry.bytes, entry.last_access);
            if prioritized {
                state.prioritized_bytes += bytes;
            }
            let occupancy = state.location(key.location);
            if prioritized {
                occupancy.prioritized_bytes += bytes;
            }
            occupancy.recency[usize::from(prioritized)].insert(last_access, key);
        }
    }

    pub fn policy(&self) -> DeviceCachePolicy {
        self.0.policy
    }

    /// Whether a read should consult this cache: it has a budget and has not
    /// been demoted by an allocation failure.
    pub fn is_enabled(&self) -> bool {
        self.0.policy.is_enabled() && !self.is_demoted()
    }

    pub fn is_demoted(&self) -> bool {
        self.lock().demoted
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.state.lock().expect("device cache mutex poisoned")
    }

    /// A cloned handle to the resident tensor, marked most recently used.
    pub(super) fn get(
        &self,
        store: DeviceCacheStore,
        name: &str,
        partition: TensorPartition,
        device: &Device,
        dtype: DType,
    ) -> Option<Tensor> {
        let key = Key {
            store,
            name: name.to_owned(),
            partition,
            location: device.location(),
            dtype,
        };
        let mut state = self.lock();
        let clock = state
            .clock
            .checked_add(1)
            .expect("cache access clock overflow");
        let (tensor, prioritized, previous_access) = match state.entries.get_mut(&key) {
            Some(entry) => {
                let previous_access = std::mem::replace(&mut entry.last_access, clock);
                #[cfg(feature = "cuda")]
                {
                    entry.cross_thread_access |= entry.owner_thread != std::thread::current().id();
                }
                (
                    entry.tensor.as_ref().expect("live cache entry").clone(),
                    entry.prioritized,
                    previous_access,
                )
            }
            None => {
                state.misses = state
                    .misses
                    .checked_add(1)
                    .expect("device cache miss counter overflow");
                return None;
            }
        };
        state.clock = clock;
        state.hits = state
            .hits
            .checked_add(1)
            .expect("device cache hit counter overflow");
        state.touch(&key, prioritized, previous_access, clock);
        Some(tensor)
    }

    /// Whether the tensor is resident. This is a report, not an access: it
    /// counts neither a hit nor a miss and does not change recency, so asking
    /// what a phase actually holds cannot alter what it holds next.
    pub(super) fn contains(
        &self,
        store: DeviceCacheStore,
        name: &str,
        partition: TensorPartition,
        device: &Device,
        dtype: DType,
    ) -> bool {
        let key = Key {
            store,
            name: name.to_owned(),
            partition,
            location: device.location(),
            dtype,
        };
        self.lock().entries.contains_key(&key)
    }

    /// Retains `tensor`, returning whether it is now cached. Never fails: a
    /// zero budget, a demoted cache, or a tensor larger than the budget simply
    /// leaves the cache unchanged, and replacing a key first removes the older
    /// value so no stale tensor can be returned later.
    pub(super) fn insert(
        &self,
        store: DeviceCacheStore,
        name: &str,
        partition: TensorPartition,
        device: &Device,
        tensor: &Tensor,
    ) -> bool {
        let bytes = tensor
            .elem_count()
            .checked_mul(tensor.dtype().size_in_bytes())
            .expect("device cache tensor byte count overflow") as u64;
        let key = Key {
            store,
            name: name.to_owned(),
            partition,
            location: device.location(),
            dtype: tensor.dtype(),
        };
        let mut state = self.lock();
        state.remove(&key);
        if !self.0.policy.is_enabled() || state.demoted || bytes > state.max_bytes {
            return false;
        }
        let prioritized = state.prioritized(store, name);
        state.admit(key, bytes, prioritized, Some(tensor.clone()))
    }

    /// Release everything one store holds, returning how many were dropped.
    ///
    /// A dropped weight store can never be read again, so its tensors are dead
    /// weight the moment it goes away. An LRU will not discover this: it
    /// evicts only as much as a newcomer needs, so a budget large enough to
    /// hold a finished component alongside the next one keeps the finished one
    /// resident for the rest of the request. `ModelWeights` calls this when it
    /// drops, which is what makes residency follow the phase boundaries rather
    /// than accumulating across them.
    pub(super) fn release_store(&self, store: DeviceCacheStore) -> usize {
        let mut state = self.lock();
        let doomed: Vec<Arc<Key>> = state
            .entries
            .keys()
            .filter(|key| key.store == store)
            .cloned()
            .collect();
        let dropped = doomed.len();
        for key in doomed {
            state.remove(&key);
        }
        state.priorities.remove(&store);
        state.inactive_priorities.remove(&store);
        dropped
    }

    /// Release every resident tensor but stay enabled, returning how many were
    /// dropped.
    ///
    /// This is a phase boundary. A phase's
    /// tensors are dead weight once the next phase starts, and an LRU alone
    /// will not say so: it evicts only as much as the newcomer needs, so a
    /// budget large enough to hold both keeps the finished phase resident for
    /// the rest of the request. Releasing at the boundary is what makes the
    /// declared per-phase working set the thing that is actually held.
    ///
    /// Unlike `demote` this is not a failure path: retention
    /// resumes for the next phase.
    pub fn release(&self) -> usize {
        let mut state = self.lock();
        let dropped = state.entries.len();
        while !state.entries.is_empty() {
            state.evict();
        }
        dropped
    }

    /// Trim current retention on each device without disabling or changing the
    /// authorized ceiling. Best-effort entries are released before prioritized
    /// entries. The bound is per device, like the ceiling itself.
    pub fn trim_to_bytes(&self, maximum_resident_bytes: u64) -> usize {
        let mut state = self.lock();
        let before = state.entries.len();
        state.trim_each_location(maximum_resident_bytes);
        before - state.entries.len()
    }

    /// Bound subsequent inserts as well as current retention. A phase can
    /// preserve workspace headroom until the next capacity adjustment; the
    /// original policy remains the upper bound when capacity is raised again.
    pub fn set_capacity_bytes(&self, maximum_resident_bytes: u64) -> usize {
        let mut state = self.lock();
        state.max_bytes = maximum_resident_bytes.min(self.0.policy.max_bytes);
        let before = state.entries.len();
        let ceiling = state.max_bytes;
        state.trim_each_location(ceiling);
        before - state.entries.len()
    }

    /// Release every resident tensor and stop retaining, returning how many
    /// were dropped. This is the demotion path of the runtime fallback chain:
    /// an optimistic placement that blocked a device allocation is given up so
    /// the request proceeds from the host tiers.
    ///
    /// Demotion is permanent for the life of the cache, and deliberately so.
    /// Releasing without disabling would refill the budget from the very next
    /// read and fail again, turning sustained pressure into an evict-everything
    /// cycle that is slower than never having retained anything.
    pub fn demote_after_allocation_failure(
        &self,
        label: &str,
        device: &Device,
        error: &anyhow::Error,
    ) -> usize {
        self.demote(Some(DeviceCacheDemotion::capture(
            label,
            None,
            format!("{error:#}"),
            device,
        )))
    }

    pub(super) fn demote(&self, failure: Option<DeviceCacheDemotion>) -> usize {
        let mut state = self.lock();
        if !state.demoted
            && let Some(mut failure) = failure
        {
            failure.resident_bytes_before = state.bytes;
            failure.prioritized_bytes_before = state.prioritized_bytes;
            state.demotion = Some(failure);
        }
        state.demoted = true;
        let dropped = state.entries.len();
        while !state.entries.is_empty() {
            state.evict();
        }
        dropped
    }

    pub fn stats(&self) -> DeviceCacheStats {
        let state = self.lock();
        let mut resident_bytes_by_device = state
            .locations
            .iter()
            .filter(|(_, occupancy)| occupancy.bytes > 0)
            .map(|(location, occupancy)| DeviceResidentBytes {
                device: format!("{location:?}"),
                resident_tensors: state
                    .entries
                    .keys()
                    .filter(|key| key.location == *location)
                    .count(),
                resident_bytes: occupancy.bytes,
                prioritized_bytes: occupancy.prioritized_bytes,
            })
            .collect::<Vec<_>>();
        resident_bytes_by_device.sort_by(|left, right| left.device.cmp(&right.device));
        DeviceCacheStats {
            resident_bytes_by_device,
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            resident_tensors: state.entries.len(),
            resident_bytes: state.bytes,
            max_bytes: state.max_bytes,
            demoted: state.demoted,
            prioritized_bytes: state.prioritized_bytes,
            priority_bypasses: state.priority_bypasses,
            demotion: state.demotion.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_policy_preserves_legacy_defaults() {
        let policy: DeviceCachePolicy =
            serde_json::from_value(serde_json::json!({"max_bytes":64})).unwrap();
        assert_eq!(policy.cuda_allocator, CudaWeightAllocator::StreamPool);
        assert_eq!(
            serde_json::to_value(policy).unwrap(),
            serde_json::json!({"max_bytes":64})
        );
        let direct = policy.with_cuda_allocator(CudaWeightAllocator::Direct);
        assert_eq!(
            serde_json::from_value::<DeviceCachePolicy>(serde_json::to_value(direct).unwrap())
                .unwrap(),
            direct
        );
    }

    #[test]
    fn pressure_trim_preserves_priority_and_allows_later_retention() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(64));
        let hot = cache.attach_prioritized(HashSet::from(["hot".to_owned()]));
        let cold = cache.attach();
        cache.insert(
            hot,
            "hot",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4),
        );
        cache.insert(
            cold,
            "a",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4),
        );
        cache.insert(
            cold,
            "b",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4),
        );
        assert_eq!(cache.trim_to_bytes(16), 2);
        assert_eq!(cache.stats().prioritized_bytes, 16);
        assert!(!cache.stats().demoted);
        assert!(cache.insert(
            cold,
            "c",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4)
        ));
        assert_eq!(cache.stats().resident_bytes, 32);
        assert_eq!(cache.trim_to_bytes(0), 2);
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    fn f32_tensor(elements: usize) -> Tensor {
        Tensor::zeros(elements, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn demotion_preserves_first_error_and_pre_release_occupancy() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(16));
        let store = cache.attach_prioritized(HashSet::from(["a".into()]));
        assert!(cache.insert(
            store,
            "a",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(2)
        ));
        let failure = DeviceCacheDemotion::capture(
            "large",
            Some(32),
            "allocation failed".into(),
            &Device::Cpu,
        );
        assert_eq!(cache.demote(Some(failure)), 1);
        let before = cache.stats().demotion.unwrap();
        assert_eq!(before.resident_bytes_before, 8);
        assert_eq!(before.prioritized_bytes_before, 8);
        assert_eq!(before.requested_bytes, Some(32));
        assert_eq!(before.error, "allocation failed");
        assert_eq!(cache.stats().resident_bytes, 0);
        let later =
            DeviceCacheDemotion::capture("other", Some(64), "later failure".into(), &Device::Cpu);
        cache.demote(Some(later));
        assert_eq!(cache.stats().demotion.as_ref(), Some(&before));
        let encoded = serde_json::to_value(cache.stats()).unwrap();
        assert_eq!(
            serde_json::from_value::<DeviceCacheStats>(encoded).unwrap(),
            cache.stats()
        );
    }

    #[test]
    fn selected_weights_survive_lower_priority_scans_and_release_their_budget() {
        let disabled =
            DeviceCache::with_selected_phases(DeviceCachePolicy::DISABLED, ["hot".to_owned()]);
        assert!(!disabled.phase_selected("hot"));
        let cache = DeviceCache::with_selected_phases(
            DeviceCachePolicy::with_max_bytes(64),
            ["hot".to_owned()],
        );
        assert!(cache.phase_selected("hot"));
        assert!(!cache.phase_selected("cold"));
        let hot = cache.attach_prioritized(HashSet::from(["weight".into(), "next".into()]));
        let cold = cache.attach();
        assert!(cache.insert(
            hot,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(8)
        ));
        assert!(cache.insert(
            cold,
            "small",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(2)
        ));
        assert!(!cache.insert(
            cold,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(10)
        ));
        assert!(cache.contains(
            hot,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            DType::F32
        ));
        assert!(cache.contains(
            cold,
            "small",
            TensorPartition::Whole,
            &Device::Cpu,
            DType::F32
        ));
        assert_eq!(cache.stats().evictions, 0);
        assert!(cache.insert(
            cold,
            "medium",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(6)
        ));
        assert!(cache.insert(
            hot,
            "next",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(8)
        ));
        assert_eq!(cache.stats().prioritized_bytes, 64);
        assert_eq!(cache.stats().evictions, 2);
        assert!(!cache.insert(
            cold,
            "tiny",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert_eq!(cache.stats().priority_bypasses, 2);
        assert_eq!(cache.release_store(hot), 2);
        assert_eq!(cache.stats().prioritized_bytes, 0);
        assert!(cache.insert(
            cold,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(16)
        ));
        assert_eq!(cache.stats().resident_bytes, 64);
    }

    #[test]
    fn priority_accounting_survives_replacement_and_demotion() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(64));
        let store = cache.attach_prioritized(HashSet::from(["weight".into()]));
        assert!(cache.insert(
            store,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(8)
        ));
        assert!(cache.insert(
            store,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4)
        ));
        assert_eq!(cache.stats().prioritized_bytes, 16);
        assert_eq!(cache.demote(None), 1);
        assert_eq!(cache.stats().prioritized_bytes, 0);
        assert!(!cache.insert(
            store,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4)
        ));
    }

    #[test]
    fn inactive_phase_can_be_evicted_in_original_access_order() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(64));
        let a = cache.attach_prioritized(HashSet::from(["a".into()]));
        let b = cache.attach_prioritized(HashSet::from(["b".into()]));
        let other = cache.attach();
        assert!(cache.insert(a, "a", TensorPartition::Whole, &Device::Cpu, &f32_tensor(8)));
        assert!(cache.insert(
            other,
            "old",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(4)
        ));
        assert!(cache.insert(b, "b", TensorPartition::Whole, &Device::Cpu, &f32_tensor(4)));
        cache.set_priority_active(a, false);
        assert_eq!(cache.stats().prioritized_bytes, 16);
        assert_eq!(cache.stats().evictions, 0);
        assert!(cache.contains(a, "a", TensorPartition::Whole, &Device::Cpu, DType::F32));
        assert!(cache.insert(
            other,
            "new",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(6)
        ));
        assert!(!cache.contains(a, "a", TensorPartition::Whole, &Device::Cpu, DType::F32));
        assert!(cache.contains(
            other,
            "old",
            TensorPartition::Whole,
            &Device::Cpu,
            DType::F32
        ));
        assert!(cache.contains(b, "b", TensorPartition::Whole, &Device::Cpu, DType::F32));
        assert_eq!(cache.stats().evictions, 1);
        cache.set_priority_active(a, true);
        assert!(cache.insert(a, "a", TensorPartition::Whole, &Device::Cpu, &f32_tensor(8)));
        assert_eq!(cache.stats().prioritized_bytes, 48);
        cache.set_priority_active(a, false);
        cache.set_priority_active(b, false);
        assert_eq!(cache.stats().prioritized_bytes, 0);
    }

    #[test]
    fn tracks_logical_bytes_hits_and_misses() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(32));
        let store = cache.attach();
        let tensor = Tensor::zeros(8, DType::BF16, &Device::Cpu).unwrap();

        assert!(cache.insert(
            store,
            "weight",
            TensorPartition::Whole,
            &Device::Cpu,
            &tensor
        ));
        assert_eq!(
            cache
                .get(
                    store,
                    "weight",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::BF16
                )
                .unwrap()
                .dims(),
            [8]
        );
        assert!(
            cache
                .get(
                    store,
                    "missing",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::BF16
                )
                .is_none()
        );
        assert!(
            cache
                .get(
                    store,
                    "weight",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32
                )
                .is_none()
        );
        assert_eq!(
            cache.stats(),
            DeviceCacheStats {
                hits: 1,
                misses: 2,
                evictions: 0,
                resident_tensors: 1,
                resident_bytes: 16,
                max_bytes: 32,
                demoted: false,
                prioritized_bytes: 0,
                priority_bypasses: 0,
                resident_bytes_by_device: vec![DeviceResidentBytes {
                    device: "Cpu".to_owned(),
                    resident_tensors: 1,
                    resident_bytes: 16,
                    prioritized_bytes: 0,
                }],
                demotion: None,
            }
        );
    }

    #[test]
    fn attached_stores_share_one_budget_without_colliding() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(8));
        let first = cache.attach();
        let second = cache.attach();
        assert_ne!(first, second);

        assert!(cache.insert(
            first,
            "shared.name",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert!(cache.insert(
            second,
            "shared.name",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert_eq!(cache.stats().resident_tensors, 2);
        assert_eq!(cache.stats().resident_bytes, 8);

        assert!(cache.insert(
            first,
            "other.name",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        let stats = cache.stats();
        assert_eq!(stats.resident_bytes, 8);
        assert_eq!(stats.evictions, 1);
        assert!(
            cache
                .get(
                    first,
                    "shared.name",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32
                )
                .is_none()
        );
        assert!(
            cache
                .get(
                    second,
                    "shared.name",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32
                )
                .is_some()
        );
    }

    /// Every device's order must file exactly its own resident entries, each
    /// under the access clock that entry records.
    ///
    /// A reorder that left a stale clock behind would keep evicting on every
    /// admission, just the wrong entry, and no counter would show it.
    fn assert_index_is_consistent(cache: &DeviceCache) {
        let state = cache.lock();
        let indexed = state
            .locations
            .values()
            .map(|occupancy| occupancy.recency.iter().map(BTreeMap::len).sum::<usize>())
            .sum::<usize>();
        assert_eq!(
            indexed,
            state.entries.len(),
            "order index and resident set disagree on size"
        );
        for (location, occupancy) in &state.locations {
            for (prioritized, queue) in occupancy.recency.iter().enumerate() {
                for (clock, key) in queue {
                    let entry = state
                        .entries
                        .get(key)
                        .expect("order index names a non-resident key");
                    assert_eq!(entry.last_access, *clock, "order index holds a stale clock");
                    assert_eq!(
                        usize::from(entry.prioritized),
                        prioritized,
                        "entry is filed under the wrong priority class"
                    );
                    assert_eq!(
                        &key.location, location,
                        "entry is filed on the wrong device"
                    );
                }
            }
        }
    }

    #[test]
    fn interleaved_access_keeps_every_device_order_in_step() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(24));
        let store = cache.attach();
        let names = ["a", "b", "c", "d", "e", "f"];
        let mut step = 1u64;
        for round in 0..200u64 {
            step = step.wrapping_mul(6364136223846793005).wrapping_add(1);
            let name = names[(step >> 33) as usize % names.len()];
            if round % 3 == 0 {
                cache.insert(
                    store,
                    name,
                    TensorPartition::Whole,
                    &Device::Cpu,
                    &f32_tensor(1),
                );
            } else {
                cache.get(
                    store,
                    name,
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32,
                );
            }
            assert_index_is_consistent(&cache);
        }
        cache.trim_to_bytes(8);
        assert_index_is_consistent(&cache);
        cache.release_store(store);
        assert_index_is_consistent(&cache);
        assert_eq!(cache.stats().resident_tensors, 0);
    }

    #[test]
    fn refreshes_recency_and_evicts_the_least_recent_entry() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(8));
        let store = cache.attach();
        assert!(cache.insert(
            store,
            "a",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert!(cache.insert(
            store,
            "b",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));

        assert!(
            cache
                .get(store, "a", TensorPartition::Whole, &Device::Cpu, DType::F32)
                .is_some()
        );
        assert!(cache.insert(
            store,
            "c",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));

        assert!(
            cache
                .get(store, "b", TensorPartition::Whole, &Device::Cpu, DType::F32)
                .is_none()
        );
        assert!(
            cache
                .get(store, "a", TensorPartition::Whole, &Device::Cpu, DType::F32)
                .is_some()
        );
        assert!(
            cache
                .get(store, "c", TensorPartition::Whole, &Device::Cpu, DType::F32)
                .is_some()
        );
        assert_eq!(
            cache.stats(),
            DeviceCacheStats {
                hits: 3,
                misses: 1,
                evictions: 1,
                resident_tensors: 2,
                resident_bytes: 8,
                max_bytes: 8,
                demoted: false,
                prioritized_bytes: 0,
                priority_bypasses: 0,
                resident_bytes_by_device: vec![DeviceResidentBytes {
                    device: "Cpu".to_owned(),
                    resident_tensors: 2,
                    resident_bytes: 8,
                    prioritized_bytes: 0,
                }],
                demotion: None,
            }
        );
    }

    #[test]
    fn evicts_multiple_entries_until_the_new_tensor_fits() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(12));
        let store = cache.attach();
        for name in ["a", "b", "c"] {
            assert!(cache.insert(
                store,
                name,
                TensorPartition::Whole,
                &Device::Cpu,
                &f32_tensor(1)
            ));
        }

        assert!(cache.insert(
            store,
            "large",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(2)
        ));
        assert_eq!(
            cache.stats(),
            DeviceCacheStats {
                hits: 0,
                misses: 0,
                evictions: 2,
                resident_tensors: 2,
                resident_bytes: 12,
                max_bytes: 12,
                demoted: false,
                prioritized_bytes: 0,
                priority_bypasses: 0,
                resident_bytes_by_device: vec![DeviceResidentBytes {
                    device: "Cpu".to_owned(),
                    resident_tensors: 2,
                    resident_bytes: 12,
                    prioritized_bytes: 0,
                }],
                demotion: None,
            }
        );
        assert!(
            cache
                .get(store, "c", TensorPartition::Whole, &Device::Cpu, DType::F32)
                .is_some()
        );
        assert!(
            cache
                .get(
                    store,
                    "large",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32
                )
                .is_some()
        );
    }

    #[test]
    fn oversize_replacement_is_not_cached_or_counted_as_an_eviction() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(8));
        let store = cache.attach();
        assert!(cache.insert(
            store,
            "expert",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));

        assert!(!cache.insert(
            store,
            "expert",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(3)
        ));
        assert!(
            cache
                .get(
                    store,
                    "expert",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32
                )
                .is_none()
        );
        assert_eq!(
            cache.stats(),
            DeviceCacheStats {
                hits: 0,
                misses: 1,
                evictions: 0,
                resident_tensors: 0,
                resident_bytes: 0,
                max_bytes: 8,
                demoted: false,
                prioritized_bytes: 0,
                priority_bypasses: 0,
                resident_bytes_by_device: Vec::new(),
                demotion: None,
            }
        );
    }

    #[test]
    fn zero_budget_disables_the_cache() {
        let cache = DeviceCache::disabled();
        let store = cache.attach();
        assert!(!cache.is_enabled());
        assert!(!cache.insert(
            store,
            "expert",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert!(
            cache
                .get(
                    store,
                    "expert",
                    TensorPartition::Whole,
                    &Device::Cpu,
                    DType::F32
                )
                .is_none()
        );
        assert_eq!(
            cache.stats(),
            DeviceCacheStats {
                hits: 0,
                misses: 1,
                evictions: 0,
                resident_tensors: 0,
                resident_bytes: 0,
                max_bytes: 0,
                demoted: false,
                prioritized_bytes: 0,
                priority_bypasses: 0,
                resident_bytes_by_device: Vec::new(),
                demotion: None,
            }
        );
    }

    #[test]
    fn demotion_releases_everything_and_stops_retaining() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(16));
        let store = cache.attach();
        assert_eq!(cache.demote(None), 0);
        assert!(cache.is_demoted());
        assert!(!cache.is_enabled());

        assert!(!cache.insert(
            store,
            "a",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert_eq!(cache.stats().resident_tensors, 0);
    }

    #[test]
    fn demotion_drops_resident_tensors_and_is_visible_in_statistics() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(16));
        let store = cache.attach();
        for name in ["a", "b"] {
            assert!(cache.insert(
                store,
                name,
                TensorPartition::Whole,
                &Device::Cpu,
                &f32_tensor(1)
            ));
        }
        assert_eq!(cache.demote(None), 2);
        let stats = cache.stats();
        assert_eq!(stats.resident_tensors, 0);
        assert_eq!(stats.resident_bytes, 0);
        assert_eq!(stats.evictions, 2);
        assert!(stats.demoted);
    }

    #[test]
    fn clones_share_one_resident_set_and_one_demotion() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(16));
        let other = cache.clone();
        let store = cache.attach();
        assert!(cache.insert(
            store,
            "a",
            TensorPartition::Whole,
            &Device::Cpu,
            &f32_tensor(1)
        ));
        assert_eq!(other.stats().resident_tensors, 1);
        assert!(
            other
                .get(store, "a", TensorPartition::Whole, &Device::Cpu, DType::F32)
                .is_some()
        );

        other.demote(None);
        assert!(cache.is_demoted());
        assert!(!cache.is_enabled());
    }

    /// Two ranks on two devices, in one process, under one `--device-cache-mib`
    /// ceiling. The ceiling is each device's, so neither rank is starved by the
    /// other, and neither rank's
    /// admissions evict the other's tensors. Real CUDA ordinals are not needed
    /// to check the accounting, and this host has one device, so the locations
    /// are named directly.
    #[test]
    fn each_device_gets_the_whole_ceiling_and_evicts_only_its_own() {
        let mut state = State {
            max_bytes: 96,
            ..State::default()
        };
        let key = |gpu_id: usize, name: &str| Key {
            store: DeviceCacheStore(0),
            name: name.to_owned(),
            partition: TensorPartition::Whole,
            location: DeviceLocation::Cuda { gpu_id },
            dtype: DType::BF16,
        };
        for gpu_id in [0usize, 1] {
            assert!(state.admit(key(gpu_id, "shard.a"), 64, false, None));
            assert!(state.admit(key(gpu_id, "shard.b"), 32, false, None));
        }
        assert_eq!(state.bytes, 192);
        assert_eq!(
            state.location_bytes(&DeviceLocation::Cuda { gpu_id: 0 }),
            96
        );
        assert_eq!(
            state.location_bytes(&DeviceLocation::Cuda { gpu_id: 1 }),
            96
        );
        assert_eq!(state.evictions, 0);

        assert!(state.admit(key(0, "shard.c"), 64, false, None));
        assert_eq!(state.evictions, 1);
        assert!(!state.entries.contains_key(&key(0, "shard.a")));
        assert!(state.entries.contains_key(&key(1, "shard.a")));
        assert!(state.entries.contains_key(&key(1, "shard.b")));
        assert_eq!(
            state.location_bytes(&DeviceLocation::Cuda { gpu_id: 1 }),
            96
        );
    }

    /// The priority reserve is a per-device reserve too: a selected phase on
    /// one device does not lock a best-effort admission out of another.
    #[test]
    fn a_selected_phase_reserves_only_its_own_device() {
        let mut state = State {
            max_bytes: 64,
            ..State::default()
        };
        let key = |gpu_id: usize, name: &str| Key {
            store: DeviceCacheStore(0),
            name: name.to_owned(),
            partition: TensorPartition::Whole,
            location: DeviceLocation::Cuda { gpu_id },
            dtype: DType::BF16,
        };
        assert!(state.admit(key(0, "selected"), 64, true, None));
        assert!(!state.admit(key(0, "best-effort"), 32, false, None));
        assert_eq!(state.priority_bypasses, 1);
        assert!(state.admit(key(1, "best-effort"), 64, false, None));
        assert_eq!(state.prioritized_bytes, 64);
        assert_eq!(
            state.location_prioritized_bytes(&DeviceLocation::Cuda { gpu_id: 1 }),
            0
        );
    }

    /// Trimming and releasing walk every device rather than emptying one.
    #[test]
    fn trimming_bounds_each_device_and_releasing_empties_all_of_them() {
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(96));
        {
            let mut state = cache.lock();
            let key = |gpu_id: usize, name: &str| Key {
                store: DeviceCacheStore(0),
                name: name.to_owned(),
                partition: TensorPartition::Whole,
                location: DeviceLocation::Cuda { gpu_id },
                dtype: DType::BF16,
            };
            for gpu_id in [0usize, 1] {
                for name in ["shard.a", "shard.b", "shard.c"] {
                    assert!(state.admit(key(gpu_id, name), 32, false, None));
                }
            }
        }
        assert_eq!(cache.stats().resident_bytes, 192);
        assert_eq!(cache.stats().resident_bytes_by_device.len(), 2);
        assert_eq!(cache.trim_to_bytes(64), 2);
        let stats = cache.stats();
        assert_eq!(stats.resident_bytes, 128);
        assert!(
            stats
                .resident_bytes_by_device
                .iter()
                .all(|device| device.resident_bytes == 64),
            "{stats:?}"
        );
        assert_eq!(cache.release(), 4);
        assert_eq!(cache.stats().resident_bytes, 0);
        assert!(cache.stats().resident_bytes_by_device.is_empty());
    }
}
