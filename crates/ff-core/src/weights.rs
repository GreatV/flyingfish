use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, safetensors::Load};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

const MAX_SAFETENSORS_HEADER_BYTES: usize = 100_000_000;

mod tensor_cache;
use tensor_cache::TensorCache;
pub mod accounting;
#[cfg(feature = "cuda")]
mod cuda_allocation;
mod device_cache;
pub use device_cache::{
    CudaWeightAllocator, DeviceCache, DeviceCacheDemotion, DeviceCachePolicy, DeviceCacheStats,
    DeviceCacheStore, DeviceResidentBytes, TensorAxis, TensorPartition,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CacheGranularity {
    #[default]
    Shard,
    Tensor,
}

impl CacheGranularity {
    pub fn is_shard(&self) -> bool {
        *self == Self::Shard
    }
}

const INDEX_NAMES: [&str; 2] = [
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
];
const SINGLE_FILE_NAMES: [&str; 2] = ["diffusion_pytorch_model.safetensors", "model.safetensors"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum WeightSource {
    #[value(
        help = "Read-only virtual-memory mapping. This is the lowest-RAM mode: the OS pages tensor bytes in from disk as they are touched"
    )]
    Mmap,
    #[value(help = "Read shard files into host RAM. A bounded shard LRU controls host usage")]
    Memory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CachePolicy {
    /// Maximum distinct source shards represented by retained units.
    pub max_shards: usize,
    pub max_bytes: Option<u64>,
    pub granularity: CacheGranularity,
}

impl CachePolicy {
    pub const fn new(max_shards: usize) -> Self {
        Self {
            max_shards,
            max_bytes: None,
            granularity: CacheGranularity::Shard,
        }
    }

    pub const fn unbounded_units() -> Self {
        Self {
            max_shards: usize::MAX,
            max_bytes: None,
            granularity: CacheGranularity::Shard,
        }
    }

    pub const fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    pub const fn with_granularity(mut self, granularity: CacheGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    fn validate(self) -> Result<Self> {
        anyhow::ensure!(
            self.max_shards > 0,
            "host shard cache capacity must be at least one"
        );
        if let Some(max_bytes) = self.max_bytes {
            anyhow::ensure!(
                max_bytes > 0,
                "host shard cache byte budget must be non-zero"
            );
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Successful parses retained by the model-open header catalog. Header
    /// inspection is independent from payload-cache hits and misses.
    pub header_parses: u64,
    pub memory_source_reads: u64,
    pub memory_source_read_bytes: u64,
    pub resident_shards: usize,
    pub resident_bytes: u64,
    pub max_shards: usize,
    #[serde(deserialize_with = "crate::required_option")]
    pub max_bytes: Option<u64>,
    pub over_budget: bool,
    /// Present even when empty in tensor mode. Omitted in the legacy shard
    /// representation so existing sealed observation digests remain unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_retention: Option<TensorRetentionStats>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRetentionStats {
    pub resident_tensors: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<HostCachePriorityStats>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCachePriorityStats {
    pub reserved_bytes: u64,
    pub resident_bytes: u64,
    pub bypasses: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRetentionDelta {
    pub resident_tensors_before: usize,
    pub resident_tensors_after: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheStatsDelta {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub header_parses: u64,
    pub memory_source_reads: u64,
    pub memory_source_read_bytes: u64,
    pub resident_shards_before: usize,
    pub resident_shards_after: usize,
    pub resident_bytes_before: u64,
    pub resident_bytes_after: u64,
    pub over_budget_before: bool,
    pub over_budget_after: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_retention: Option<TensorRetentionDelta>,
}

impl CacheStats {
    pub fn delta_since(&self, before: &Self) -> CacheStatsDelta {
        assert_eq!(
            self.tensor_retention.is_some(),
            before.tensor_retention.is_some(),
            "cannot compare cache counters from different granularities"
        );
        CacheStatsDelta {
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            evictions: self.evictions.saturating_sub(before.evictions),
            header_parses: self.header_parses.saturating_sub(before.header_parses),
            memory_source_reads: self
                .memory_source_reads
                .saturating_sub(before.memory_source_reads),
            memory_source_read_bytes: self
                .memory_source_read_bytes
                .saturating_sub(before.memory_source_read_bytes),
            resident_shards_before: before.resident_shards,
            resident_shards_after: self.resident_shards,
            resident_bytes_before: before.resident_bytes,
            resident_bytes_after: self.resident_bytes,
            over_budget_before: before.over_budget,
            over_budget_after: self.over_budget,
            tensor_retention: self
                .tensor_retention
                .as_ref()
                .map(|after| TensorRetentionDelta {
                    resident_tensors_before: before
                        .tensor_retention
                        .as_ref()
                        .unwrap()
                        .resident_tensors,
                    resident_tensors_after: after.resident_tensors,
                }),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightAccessStats {
    pub device_tensor_materializations: u64,
    pub device_row_materializations: u64,
}

impl WeightAccessStats {
    pub fn delta_since(&self, before: &Self) -> Self {
        Self {
            device_tensor_materializations: self
                .device_tensor_materializations
                .saturating_sub(before.device_tensor_materializations),
            device_row_materializations: self
                .device_row_materializations
                .saturating_sub(before.device_row_materializations),
        }
    }
}

#[derive(Default)]
struct WeightAccessCounters {
    device_tensor_materializations: AtomicU64,
    device_row_materializations: AtomicU64,
    /// Whether to keep `tensor_reads`. Off by default: counting locks a mutex
    /// and allocates the tensor's name on every read, and every adapter's
    /// hottest path is a read.
    count_tensor_reads: AtomicBool,
    /// Per-tensor read counts across `load` and `load_rows`, whether the read
    /// hit a resident copy or materialized one. Phase declarations are
    /// verified against these counts. Only populated while counting is on.
    tensor_reads: Mutex<BTreeMap<String, u64>>,
}

impl WeightAccessCounters {
    fn snapshot(&self) -> WeightAccessStats {
        WeightAccessStats {
            device_tensor_materializations: self
                .device_tensor_materializations
                .load(Ordering::Relaxed),
            device_row_materializations: self.device_row_materializations.load(Ordering::Relaxed),
        }
    }

    fn record_tensor_read(&self, name: &str) {
        if !self.count_tensor_reads.load(Ordering::Relaxed) {
            return;
        }
        let mut reads = self
            .tensor_reads
            .lock()
            .expect("tensor read counters mutex poisoned");
        *reads.entry(name.to_owned()).or_default() += 1;
    }

    fn tensor_reads(&self) -> BTreeMap<String, u64> {
        self.tensor_reads
            .lock()
            .expect("tensor read counters mutex poisoned")
            .clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMetadata {
    pub name: String,
    pub shard: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawTensorMetadata {
    pub name: String,
    pub shard: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub file_offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawShardMetadata {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<RawTensorMetadata>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightInventory {
    pub tensors: usize,
    pub shards: usize,
    pub indexed_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VerificationReport {
    pub checked_shards: usize,
    pub checked_tensors: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsIndex {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct IndexMetadata {
    total_size: u64,
}

enum ShardBytes {
    Mmap(MappedShard),
    Memory(Box<[u8]>),
    Empty,
}

/// A mapped shard, holding its descriptor open only when the checkpoint is too
/// large for the host to retain. The descriptor is what lets the shard drop its
/// own cached pages when it is evicted, which is worth doing exactly when the
/// pages would never be hit again -- see `storage::advise_dropped`.
struct MappedShard {
    mapping: Mmap,
    /// Backing file for positional reads and page eviction.
    file: Option<File>,
    droppable: bool,
}

impl Drop for MappedShard {
    fn drop(&mut self) {
        if self.droppable
            && let Some(file) = &self.file
        {
            let _ = crate::storage::advise_dropped(file);
        }
    }
}

impl ShardBytes {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mmap(value) => &value.mapping,
            Self::Memory(value) => value,
            Self::Empty => &[],
        }
    }
}

#[derive(Clone)]
struct CachedTensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    offset: usize,
    len: usize,
}

struct ShardHeader {
    file_bytes: u64,
    payload_offset: usize,
    encoded_header: Box<[u8]>,
    tensors: BTreeMap<String, CachedTensorInfo>,
}

impl ShardHeader {
    fn read(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("failed to open weight shard header {}", path.display()))?;
        let file_bytes = file
            .metadata()
            .with_context(|| format!("failed to stat weight shard {}", path.display()))?
            .len();
        let mut encoded_len = [0u8; size_of::<u64>()];
        file.read_exact(&mut encoded_len)
            .with_context(|| format!("failed to read weight shard header {}", path.display()))?;
        let json_bytes = usize::try_from(u64::from_le_bytes(encoded_len))
            .context("safetensors header length exceeds usize")?;
        anyhow::ensure!(
            json_bytes <= MAX_SAFETENSORS_HEADER_BYTES,
            "safetensors header exceeds the {MAX_SAFETENSORS_HEADER_BYTES}-byte limit"
        );
        let payload_offset = size_of::<u64>()
            .checked_add(json_bytes)
            .context("safetensors payload offset overflow")?;
        let mut encoded_header = Vec::with_capacity(payload_offset);
        encoded_header.extend_from_slice(&encoded_len);
        encoded_header.resize(payload_offset, 0);
        file.read_exact(&mut encoded_header[size_of::<u64>()..])
            .with_context(|| format!("failed to read weight shard header {}", path.display()))?;
        let metadata: safetensors::tensor::Metadata =
            serde_json::from_slice(&encoded_header[size_of::<u64>()..])
                .with_context(|| format!("invalid safetensors header {}", path.display()))?;
        let expected_file_bytes = u64::try_from(payload_offset)
            .context("safetensors payload offset exceeds u64")?
            .checked_add(
                u64::try_from(metadata.data_len())
                    .context("safetensors payload length exceeds u64")?,
            )
            .context("safetensors file length overflow")?;
        anyhow::ensure!(
            file_bytes == expected_file_bytes,
            "safetensors header describes {expected_file_bytes} bytes, but {} has {file_bytes} bytes",
            path.display()
        );

        let mut tensors = BTreeMap::new();
        for name in metadata.offset_keys() {
            let info = metadata
                .info(&name)
                .with_context(|| format!("safetensors metadata lost tensor {name}"))?;
            let offset = payload_offset
                .checked_add(info.data_offsets.0)
                .context("safetensors tensor offset overflow")?;
            let len = info
                .data_offsets
                .1
                .checked_sub(info.data_offsets.0)
                .context("safetensors tensor byte range is reversed")?;
            tensors.insert(
                name,
                CachedTensorInfo {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    offset,
                    len,
                },
            );
        }
        Ok(Self {
            file_bytes,
            payload_offset,
            encoded_header: encoded_header.into_boxed_slice(),
            tensors,
        })
    }

    fn tensor_info(&self, name: &str) -> Result<&CachedTensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor is absent from cached shard header: {name}"))
    }
}

#[derive(Default)]
struct HeaderCatalogState {
    shards: HashMap<PathBuf, Arc<ShardHeader>>,
    parses: u64,
}

#[derive(Default)]
struct ShardHeaderCatalog {
    state: Mutex<HeaderCatalogState>,
}

impl ShardHeaderCatalog {
    fn get(&self, path: &Path) -> Result<Arc<ShardHeader>> {
        let mut state = self
            .state
            .lock()
            .expect("shard header catalog mutex poisoned");
        if let Some(header) = state.shards.get(path) {
            return Ok(Arc::clone(header));
        }
        let header = Arc::new(ShardHeader::read(path)?);
        state.parses = state.parses.saturating_add(1);
        state.shards.insert(path.to_path_buf(), Arc::clone(&header));
        Ok(header)
    }

    fn parses(&self) -> u64 {
        self.state
            .lock()
            .expect("shard header catalog mutex poisoned")
            .parses
    }
}

struct ShardData {
    storage: ShardBytes,
    header: Arc<ShardHeader>,
}

impl ShardData {
    fn new(storage: ShardBytes, header: Arc<ShardHeader>) -> Result<Self> {
        anyhow::ensure!(
            storage.bytes().len() as u64 == header.file_bytes,
            "weight shard length changed after its header was cataloged"
        );
        anyhow::ensure!(
            storage.bytes().starts_with(&header.encoded_header),
            "weight shard header changed after it was cataloged"
        );
        Ok(Self { storage, header })
    }

    fn bytes(&self) -> &[u8] {
        self.storage.bytes()
    }

    fn tensor_info(&self, name: &str) -> Result<&CachedTensorInfo> {
        self.header.tensor_info(name)
    }

    fn tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self.tensor_info(name)?;
        let end = info
            .offset
            .checked_add(info.len)
            .context("cached tensor byte range overflow")?;
        let data = self
            .bytes()
            .get(info.offset..end)
            .context("cached tensor byte range exceeds shard")?;
        TensorView::new(info.dtype, info.shape.clone(), data).map_err(Into::into)
    }

    /// Returns an mmap tensor's backing file, byte offset and length.
    #[cfg(feature = "cuda")]
    fn tensor_file_range(&self, name: &str) -> Option<(&File, usize, usize)> {
        let info = self.tensor_info(name).ok()?;
        if let ShardBytes::Mmap(mapped) = &self.storage {
            let file = mapped.file.as_ref()?;
            Some((file, info.offset, info.len))
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.bytes().len()
    }
}

struct CacheLookup {
    shard: Arc<ShardData>,
}

struct CacheState {
    shards: HashMap<PathBuf, Arc<ShardData>>,
    lru: VecDeque<PathBuf>,
    hits: u64,
    misses: u64,
    evictions: u64,
    memory_source_reads: u64,
    memory_source_read_bytes: u64,
    resident_bytes: u64,
}

struct ShardCache {
    source: WeightSource,
    policy: CachePolicy,
    headers: ShardHeaderCatalog,
    /// Whether an evicted shard should drop its cached pages, decided once from
    /// the checkpoint's own size against the host's own free memory.
    drop_evicted_pages: bool,
    state: Mutex<CacheState>,
}

/// Read a shard with parallel positional reads.
/// FF_WEIGHT_LOAD_THREADS sets the reader count (default 8).
#[cfg(unix)]
fn read_shard_parallel(path: &Path) -> Result<Box<[u8]>> {
    use std::os::unix::fs::FileExt;
    let file = File::open(path)
        .with_context(|| format!("failed to open weight shard {}", path.display()))?;
    let len = usize::try_from(file.metadata()?.len()).context("weight shard size exceeds usize")?;
    let mut bytes = vec![0u8; len];
    let threads = std::env::var("FF_WEIGHT_LOAD_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8usize)
        .max(1)
        .min(len.div_ceil(4 << 20).max(1));
    let chunk = len.div_ceil(threads);
    let base = bytes.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for reader in 0..threads {
            let start = reader * chunk;
            if start >= len {
                break;
            }
            let end = (start + chunk).min(len);
            let file = &file;
            handles.push(scope.spawn(move || -> Result<()> {
                let mut cursor = start;
                while cursor < end {
                    let wanted = (end - cursor).min(4 << 20);
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut((base + cursor) as *mut u8, wanted)
                    };
                    let read = file.read_at(slice, cursor as u64).with_context(|| {
                        format!("failed to read weight shard {}", path.display())
                    })?;
                    anyhow::ensure!(read > 0, "weight shard {} is truncated", path.display());
                    cursor += read;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("weight shard reader panicked")))?;
        }
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(bytes.into_boxed_slice())
}

#[cfg(not(unix))]
fn read_shard_parallel(path: &Path) -> Result<Box<[u8]>> {
    Ok(fs::read(path)
        .with_context(|| format!("failed to read weight shard {}", path.display()))?
        .into_boxed_slice())
}

impl ShardCache {
    fn new(source: WeightSource, policy: CachePolicy) -> Self {
        Self {
            source,
            policy,
            headers: ShardHeaderCatalog::default(),
            drop_evicted_pages: false,
            state: Mutex::new(CacheState {
                shards: HashMap::new(),
                lru: VecDeque::new(),
                hits: 0,
                misses: 0,
                evictions: 0,
                memory_source_reads: 0,
                memory_source_read_bytes: 0,
                resident_bytes: 0,
            }),
        }
    }

    fn get(&self, path: &Path) -> Result<Arc<ShardData>> {
        Ok(self.lookup(path)?.shard)
    }

    fn lookup(&self, path: &Path) -> Result<CacheLookup> {
        anyhow::ensure!(
            self.policy.granularity == CacheGranularity::Shard,
            "whole-shard payload lookup is invalid under tensor granularity"
        );
        {
            let mut state = self.state.lock().expect("shard cache mutex poisoned");
            if let Some(shard) = state.shards.get(path).cloned() {
                state.hits += 1;
                touch(&mut state.lru, path);
                return Ok(CacheLookup { shard });
            }
            state.misses += 1;
        }

        let header = self.headers.get(path)?;
        let storage = match self.source {
            WeightSource::Mmap => {
                let file = File::open(path)
                    .with_context(|| format!("failed to open weight shard {}", path.display()))?;
                let mmap = unsafe { MmapOptions::new().map(&file) }
                    .with_context(|| format!("failed to mmap weight shard {}", path.display()))?;
                ShardBytes::Mmap(MappedShard {
                    mapping: mmap,
                    file: Some(file),
                    droppable: self.drop_evicted_pages,
                })
            }
            WeightSource::Memory => ShardBytes::Memory(read_shard_parallel(path)?),
        };
        let loaded = Arc::new(
            ShardData::new(storage, header)
                .with_context(|| format!("invalid safetensors shard {}", path.display()))?,
        );
        let loaded_bytes = u64::try_from(loaded.len()).context("weight shard size exceeds u64")?;

        let mut state = self.state.lock().expect("shard cache mutex poisoned");
        if self.source == WeightSource::Memory {
            state.memory_source_reads = state.memory_source_reads.saturating_add(1);
            state.memory_source_read_bytes =
                state.memory_source_read_bytes.saturating_add(loaded_bytes);
        }
        if let Some(existing) = state.shards.get(path).cloned() {
            touch(&mut state.lru, path);
            return Ok(CacheLookup { shard: existing });
        }
        while !state.shards.is_empty() && self.insertion_exceeds_policy(&state, loaded_bytes) {
            anyhow::ensure!(
                evict_oldest(&mut state),
                "host shard cache LRU is inconsistent"
            );
        }
        state.shards.insert(path.to_path_buf(), Arc::clone(&loaded));
        state.lru.push_back(path.to_path_buf());
        state.resident_bytes = state
            .resident_bytes
            .checked_add(loaded_bytes)
            .context("host shard cache byte count overflow")?;
        Ok(CacheLookup { shard: loaded })
    }

    fn insertion_exceeds_policy(&self, state: &CacheState, loaded_bytes: u64) -> bool {
        if state.shards.len() >= self.policy.max_shards {
            return true;
        }
        self.policy.max_bytes.is_some_and(|max_bytes| {
            state
                .resident_bytes
                .checked_add(loaded_bytes)
                .is_none_or(|total| total > max_bytes)
        })
    }

    fn stats(&self) -> CacheStats {
        let state = self.state.lock().expect("shard cache mutex poisoned");
        CacheStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            header_parses: self.headers.parses(),
            memory_source_reads: state.memory_source_reads,
            memory_source_read_bytes: state.memory_source_read_bytes,
            resident_shards: state.shards.len(),
            resident_bytes: state.resident_bytes,
            max_shards: self.policy.max_shards,
            max_bytes: self.policy.max_bytes,
            over_budget: self
                .policy
                .max_bytes
                .is_some_and(|max_bytes| state.resident_bytes > max_bytes),
            tensor_retention: None,
        }
    }
}

fn evict_oldest(state: &mut CacheState) -> bool {
    while let Some(oldest) = state.lru.pop_front() {
        if let Some(shard) = state.shards.remove(&oldest) {
            state.resident_bytes = state
                .resident_bytes
                .checked_sub(shard.len() as u64)
                .expect("cached shard byte count is inconsistent");
            state.evictions += 1;
            return true;
        }
    }
    false
}

fn touch(lru: &mut VecDeque<PathBuf>, path: &Path) {
    if let Some(index) = lru.iter().position(|entry| entry == path) {
        lru.remove(index);
    }
    lru.push_back(path.to_path_buf());
}

pub struct ModelWeights {
    root: PathBuf,
    index_path: PathBuf,
    index: SafetensorsIndex,
    cache: ShardCache,
    tensor_cache: Option<TensorCache>,
    device_cache: Option<AttachedDeviceCache>,
    host_complements_device: bool,
    access: WeightAccessCounters,
}

/// One weight store's registration against a shared device budget.
struct AttachedDeviceCache {
    cache: DeviceCache,
    store: DeviceCacheStore,
}

impl Drop for AttachedDeviceCache {
    /// Hand the budget back when the store goes away. Nothing can read this
    /// store's tensors again, so holding them would spend the request's
    /// ceiling on a component that has finished.
    fn drop(&mut self) {
        self.cache.release_store(self.store);
    }
}

/// Build a one-shard index over a single safetensors file.
fn single_file_index(model_path: &Path, file_name: &str) -> Result<SafetensorsIndex> {
    let file = File::open(model_path)
        .with_context(|| format!("failed to open weight file {}", model_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to mmap weight file {}", model_path.display()))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("invalid safetensors file {}", model_path.display()))?;
    let mut total_size = 0u64;
    let mut weight_map = BTreeMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name)?;
        total_size = total_size
            .checked_add(view.data().len() as u64)
            .context("single-file safetensors size overflow")?;
        weight_map.insert(name.to_owned(), file_name.to_owned());
    }
    Ok(SafetensorsIndex {
        metadata: IndexMetadata { total_size },
        weight_map,
    })
}

impl ModelWeights {
    /// Open one named safetensors file inside a directory that holds several.
    ///
    /// [`Self::open`] discovers a checkpoint by its conventional index or
    /// single-file name, which assumes the directory belongs to one model. Some
    /// checkpoints instead place several unrelated components side by side and
    /// name each after its role, so the caller has to say which one it wants.
    pub fn open_component(
        root: impl AsRef<Path>,
        file_name: &str,
        source: WeightSource,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let cache_policy = cache_policy.validate()?;
        let root = root.as_ref();
        anyhow::ensure!(
            root.is_dir(),
            "checkpoint directory does not exist: {}",
            root.display()
        );
        validate_shard_name(file_name)?;
        let model_path = root.join(file_name);
        anyhow::ensure!(
            model_path.is_file(),
            "checkpoint component is missing: {}",
            model_path.display()
        );
        let index = single_file_index(&model_path, file_name)?;
        anyhow::ensure!(
            !index.weight_map.is_empty(),
            "checkpoint component contains no tensors"
        );
        Ok(Self {
            root: root.to_path_buf(),
            index_path: model_path,
            index,
            cache: ShardCache::new(source, cache_policy),
            tensor_cache: (cache_policy.granularity == CacheGranularity::Tensor)
                .then(|| TensorCache::new(source, cache_policy)),
            device_cache: None,
            host_complements_device: false,
            access: WeightAccessCounters::default(),
        })
    }

    pub fn open(
        root: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let cache_policy = cache_policy.validate()?;
        let root = root.as_ref();
        anyhow::ensure!(
            root.is_dir(),
            "checkpoint directory does not exist: {}",
            root.display()
        );
        let index_path = INDEX_NAMES
            .iter()
            .map(|name| root.join(name))
            .find(|path| path.is_file());
        let (index_path, index) = if let Some(index_path) = index_path {
            let bytes = fs::read(&index_path)
                .with_context(|| format!("failed to read weight index {}", index_path.display()))?;
            let index = serde_json::from_slice(&bytes)
                .with_context(|| format!("invalid weight index {}", index_path.display()))?;
            (index_path, index)
        } else {
            let model_path = SINGLE_FILE_NAMES
                .iter()
                .map(|name| root.join(name))
                .find(|path| path.is_file())
                .with_context(|| {
                    format!(
                        "no safetensors weights found in {} (looked for an index or {})",
                        root.display(),
                        SINGLE_FILE_NAMES.join(" or ")
                    )
                })?;
            let file_name = model_path
                .file_name()
                .and_then(|value| value.to_str())
                .context("safetensors filename is not valid UTF-8")?
                .to_owned();
            let index = single_file_index(&model_path, &file_name)?;
            (model_path, index)
        };
        anyhow::ensure!(
            !index.weight_map.is_empty(),
            "weight index contains no tensors"
        );

        for shard in index.weight_map.values() {
            validate_shard_name(shard)?;
            let path = root.join(shard);
            anyhow::ensure!(
                path.is_file(),
                "weight shard is missing: {}",
                path.display()
            );
        }

        Ok(Self {
            root: root.to_path_buf(),
            index_path,
            index,
            cache: ShardCache::new(source, cache_policy),
            tensor_cache: (cache_policy.granularity == CacheGranularity::Tensor)
                .then(|| TensorCache::new(source, cache_policy)),
            device_cache: None,
            host_complements_device: false,
            access: WeightAccessCounters::default(),
        })
    }

    /// Drop a shard's cached pages when the shard is evicted.
    ///
    /// Only the caller knows whether this helps, because it depends entirely on
    /// reuse distance and not on how large the checkpoint is. An adapter that
    /// walks every block once per evaluation has a reuse distance of the whole
    /// model: if that exceeds host memory, nothing it reads is ever hit again,
    /// and the page cache pays to hold pages it will evict before their next
    /// use. Dropping each shard behind that scan was measured at 1.82 GB/s
    /// against 0.13 GB/s for the same scan without it.
    ///
    /// An adapter with locality wants the opposite, and a checkpoint far larger
    /// than memory is no evidence against locality: GLM-5.3-Flash is 305 GiB
    /// against 58 GiB of host memory, but its routed experts repeat across
    /// tokens, and dropping its pages on eviction measured 53% slower
    /// end-to-end. Size is not the signal; declared reuse is.
    ///
    /// Inert for `WeightSource::Memory`, which has no page cache behind it, and
    /// off Unix, where there is no primitive for it.
    pub fn drop_evicted_pages(&mut self, enabled: bool) {
        self.cache.drop_evicted_pages = enabled && self.cache.source == WeightSource::Mmap;
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    pub fn cache_policy(&self) -> CachePolicy {
        self.cache.policy
    }

    /// Configure retention after metadata inspection but before any payload
    /// access. Retains the header catalog; never reinterprets a live cache.
    pub fn configure_unloaded_cache(
        &mut self,
        source: WeightSource,
        policy: CachePolicy,
    ) -> Result<()> {
        let policy = policy.validate()?;
        let stats = self.cache_stats();
        anyhow::ensure!(
            stats.hits == 0 && stats.misses == 0 && stats.resident_bytes == 0,
            "cannot reconfigure a weight cache after payload access"
        );
        self.cache.source = source;
        self.cache.policy = policy;
        self.tensor_cache = (policy.granularity == CacheGranularity::Tensor)
            .then(|| TensorCache::new(source, policy));
        Ok(())
    }

    /// Reserve bounded raw tensor-cache capacity for recurrent weights before
    /// payload access. Other tensors can use only the unreserved remainder.
    pub fn configure_host_cache_priority(
        &mut self,
        names: impl IntoIterator<Item = String>,
    ) -> Result<u64> {
        let names = names.into_iter().collect::<std::collections::HashSet<_>>();
        let mut bytes = 0u64;
        let mut shards = BTreeSet::new();
        for name in &names {
            bytes = bytes
                .checked_add(self.raw_tensor_metadata(name)?.bytes as u64)
                .context("host priority reservation overflow")?;
            shards.insert(&self.index.weight_map[name]);
        }
        anyhow::ensure!(
            shards.len() <= self.cache.policy.max_shards,
            "host priority tensors exceed the source shard limit"
        );
        let access = self.access_stats();
        anyhow::ensure!(
            access.device_tensor_materializations == 0 && access.device_row_materializations == 0,
            "cannot configure host priority after payload access"
        );
        self.tensor_cache
            .as_ref()
            .context("host priority requires tensor cache granularity")?
            .configure_priority(names, bytes)?;
        Ok(bytes)
    }

    /// Plan host retention for weights outside the protected device set. Raw
    /// host copies are released only after protected CUDA retention succeeds.
    pub fn configure_complementary_host_cache(
        &mut self,
        names: impl IntoIterator<Item = String>,
    ) -> Result<u64> {
        let bytes = self.configure_host_cache_priority(names)?;
        self.host_complements_device = true;
        Ok(bytes)
    }

    fn release_host_duplicate(
        &self,
        attached: &AttachedDeviceCache,
        device: &Device,
        name: &str,
        shard: &str,
    ) {
        if self.host_complements_device
            && device.is_cuda()
            && attached.cache.is_prioritized(attached.store, name)
            && let Some(cache) = &self.tensor_cache
        {
            cache.release(&self.root.join(shard), name);
        }
    }

    /// Configure device-tier retention after metadata inspection but before
    /// any payload access. The default is a zero budget: tensors are
    /// materialized on every read, as before. Enabling it never changes what a
    /// read returns — a hit is the tensor the loader would have produced.
    pub fn configure_device_cache(&mut self, cache: DeviceCache) -> Result<()> {
        self.configure_device_cache_with_priority(cache, std::iter::empty())
    }

    /// Protect the supplied tensor names from lower-priority cache admissions.
    /// Names are validated using the index; configuration never reads payloads.
    pub fn configure_device_cache_with_priority(
        &mut self,
        cache: DeviceCache,
        names: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let names = names.into_iter().collect::<std::collections::HashSet<_>>();
        anyhow::ensure!(
            names.iter().all(|name| self.contains(name)),
            "priority set contains an unknown weight tensor"
        );
        let stats = self.cache_stats();
        let access = self.access_stats();
        anyhow::ensure!(
            stats.hits == 0
                && stats.misses == 0
                && stats.resident_bytes == 0
                && access.device_tensor_materializations == 0
                && access.device_row_materializations == 0,
            "cannot configure a device weight cache after payload access"
        );
        let store = cache.attach_prioritized(names);
        self.device_cache = Some(AttachedDeviceCache { cache, store });
        Ok(())
    }

    pub fn device_cache_policy(&self) -> DeviceCachePolicy {
        self.device_cache
            .as_ref()
            .map_or(DeviceCachePolicy::DISABLED, |attached| {
                attached.cache.policy()
            })
    }

    /// The shared retention budget, for stage-local workspace adjustments.
    pub fn device_cache(&self) -> Option<&DeviceCache> {
        self.device_cache.as_ref().map(|attached| &attached.cache)
    }

    /// Activate protection while this store's phase runs. Deactivation keeps
    /// tensors available for reuse but lets subsequent phases evict them.
    pub fn set_device_priority_active(&self, active: bool) {
        if let Some(attached) = &self.device_cache {
            attached.cache.set_priority_active(attached.store, active);
        }
    }

    /// Statistics for the whole shared budget, not this store's slice of it:
    /// several stores attached to one `DeviceCache` report the same numbers.
    pub fn device_cache_stats(&self) -> DeviceCacheStats {
        self.device_cache
            .as_ref()
            .map_or_else(DeviceCacheStats::default, |attached| attached.cache.stats())
    }

    /// Whether a `load` of this tensor would be served from device residency.
    ///
    /// Probing is not an access: it counts no hit or miss and does not change
    /// recency, so a caller can ask what a phase actually holds without
    /// changing what it holds next.
    pub fn device_resident(&self, name: &str, device: &Device) -> Result<bool> {
        let Some(attached) = self.device_cache.as_ref() else {
            return Ok(false);
        };
        let dtype = produced_dtype(self.raw_tensor_metadata(name)?.dtype, device)?;
        Ok(attached
            .cache
            .contains(attached.store, name, TensorPartition::Whole, device, dtype))
    }

    /// Start or stop per-tensor read counting.
    ///
    /// Off by default. Counting takes a lock and allocates the tensor's name
    /// on every read, so it is a diagnostic a caller asks for — when verifying
    /// a phase declaration — not a cost every request pays.
    pub fn count_tensor_reads(&self, enabled: bool) {
        self.access
            .count_tensor_reads
            .store(enabled, Ordering::Relaxed);
    }

    /// Per-tensor read counts across `load` and `load_rows`, in name order.
    /// Resident reads count too: this is the read set a phase declaration is
    /// checked against, independent of where the bytes came from. Empty unless
    /// [`ModelWeights::count_tensor_reads`] turned counting on.
    pub fn tensor_reads(&self) -> BTreeMap<String, u64> {
        self.access.tensor_reads()
    }

    pub fn inventory(&self) -> WeightInventory {
        WeightInventory {
            tensors: self.index.weight_map.len(),
            shards: self
                .index
                .weight_map
                .values()
                .collect::<BTreeSet<_>>()
                .len(),
            indexed_bytes: Some(self.index.metadata.total_size),
        }
    }

    pub fn max_shard_file_bytes(&self) -> Result<u64> {
        self.index
            .weight_map
            .values()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|shard| {
                let path = self.root.join(shard);
                fs::metadata(&path)
                    .with_context(|| format!("failed to stat weight shard {}", path.display()))
                    .map(|metadata| metadata.len())
            })
            .try_fold(0u64, |largest, bytes| bytes.map(|bytes| largest.max(bytes)))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.index.weight_map.keys().map(String::as_str)
    }

    pub fn indexed_payload_bytes(&self) -> u64 {
        self.index.metadata.total_size
    }

    pub fn indexed_shard_names(&self) -> Vec<String> {
        self.index
            .weight_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.weight_map.contains_key(name)
    }

    pub fn access_stats(&self) -> WeightAccessStats {
        self.access.snapshot()
    }

    pub fn metadata(&self, name: &str) -> Result<TensorMetadata> {
        let raw = self.raw_tensor_metadata(name)?;
        Ok(TensorMetadata {
            name: raw.name,
            shard: self.root.join(raw.shard),
            dtype: format!("{:?}", raw.dtype),
            shape: raw.shape,
            bytes: raw.bytes,
        })
    }

    pub fn raw_tensor_metadata(&self, name: &str) -> Result<RawTensorMetadata> {
        let shard = self.index.weight_map.get(name).with_context(|| {
            format!(
                "tensor is not present in {}: {name}",
                self.index_path.display()
            )
        })?;
        let path = self.root.join(shard);
        let header = self.cache.headers.get(&path)?;
        let info = header.tensor_info(name).with_context(|| {
            format!("index maps {name} to {}, but it is absent", path.display())
        })?;
        Ok(RawTensorMetadata {
            name: name.to_owned(),
            shard: shard.clone(),
            dtype: info.dtype,
            shape: info.shape.clone(),
            file_offset: info.offset,
            bytes: info.len,
        })
    }

    /// Bytes the tensor occupies once materialized on `device`, after the
    /// loader's promotion of CPU BF16/F16 to F32. This is the byte charge the
    /// device cache and the residency ladder use for the tensor.
    pub fn produced_bytes(&self, name: &str, device: &Device) -> Result<u64> {
        let raw = self.raw_tensor_metadata(name)?;
        let dtype = produced_dtype(raw.dtype, device)?;
        let mut elements = 1u64;
        for &dimension in &raw.shape {
            elements = elements
                .checked_mul(u64::try_from(dimension)?)
                .with_context(|| format!("element count of {name} overflows u64"))?;
        }
        elements
            .checked_mul(dtype.size_in_bytes() as u64)
            .with_context(|| format!("byte count of {name} overflows u64"))
    }

    pub fn raw_shard_metadata(&self, shard: &str) -> Result<RawShardMetadata> {
        validate_shard_name(shard)?;
        anyhow::ensure!(
            self.index.weight_map.values().any(|value| value == shard),
            "shard is not referenced by {}: {shard}",
            self.index_path.display()
        );
        let path = self.root.join(shard);
        let header = self.cache.headers.get(&path)?;
        let mut tensors = Vec::with_capacity(header.tensors.len());
        for (name, info) in &header.tensors {
            tensors.push(RawTensorMetadata {
                name: name.clone(),
                shard: shard.to_owned(),
                dtype: info.dtype,
                shape: info.shape.clone(),
                file_offset: info.offset,
                bytes: info.len,
            });
        }
        Ok(RawShardMetadata {
            name: shard.to_owned(),
            file_bytes: header.file_bytes,
            header_bytes: u64::try_from(header.payload_offset)
                .context("safetensors header length exceeds u64")?,
            payload_bytes: header
                .file_bytes
                .checked_sub(
                    u64::try_from(header.payload_offset)
                        .context("safetensors header length exceeds u64")?,
                )
                .context("safetensors header exceeds file length")?,
            tensors,
        })
    }

    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.access.record_tensor_read(name);
        let attached = self
            .device_cache
            .as_ref()
            .filter(|attached| attached.cache.is_enabled());
        let Some(attached) = attached else {
            let tensor = self.materialize(name, device)?;
            self.access
                .device_tensor_materializations
                .fetch_add(1, Ordering::Relaxed);
            return Ok(tensor);
        };
        let metadata = self.raw_tensor_metadata(name)?;
        let dtype = produced_dtype(metadata.dtype, device)?;
        if let Some(tensor) =
            attached
                .cache
                .get(attached.store, name, TensorPartition::Whole, device, dtype)
        {
            self.release_host_duplicate(attached, device, name, &metadata.shard);
            return Ok(tensor);
        }
        let tensor = match self.materialize_for_cache(name, device, attached) {
            Ok(tensor) => tensor,
            Err(error) if is_device_error(&error) => {
                let failure = DeviceCacheDemotion::capture(
                    name,
                    self.produced_bytes(name, device).ok(),
                    format!("{error:#}"),
                    device,
                );
                let released = attached.cache.demote(Some(failure));
                if device.is_cuda()
                    && attached.cache.policy().cuda_allocator == CudaWeightAllocator::Direct
                {
                    device.synchronize()?;
                }
                self.materialize(name, device).with_context(|| {
                    format!("retried after releasing {released} device-resident tensors: {error:#}")
                })?
            }
            Err(error) => return Err(error),
        };
        self.access
            .device_tensor_materializations
            .fetch_add(1, Ordering::Relaxed);
        if attached.cache.insert(
            attached.store,
            name,
            TensorPartition::Whole,
            device,
            &tensor,
        ) {
            self.release_host_duplicate(attached, device, name, &metadata.shard);
        }
        Ok(tensor)
    }

    /// Borrow a tensor's original bytes while its mmap/cache entry stays live.
    /// Backends can copy into reusable staging without first creating a Tensor.
    /// This records a tensor read, but no owned tensor materialization.
    pub fn with_tensor_bytes<T>(
        &self,
        name: &str,
        f: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        self.access.record_tensor_read(name);
        self.with_view(name, |view, _| f(view.data()))
    }

    fn materialize(&self, name: &str, device: &Device) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            let metadata = self.raw_tensor_metadata(name)?;
            let dtype = produced_dtype(metadata.dtype, device)?;
            if metadata.bytes > 0 && cuda_allocation::supports(dtype) {
                return self.with_source(name, |view, source| {
                    cuda_allocation::upload(view.data(), source, view.shape(), dtype, cuda)
                        .with_context(|| {
                            format!("failed to materialize tensor {name} on {device:?}")
                        })
                });
            }
        }
        self.with_view(name, |view, _| {
            let tensor = view
                .load(device)
                .with_context(|| format!("failed to materialize tensor {name} on {device:?}"))?;
            if device.is_cpu()
                && matches!(
                    tensor.dtype(),
                    candle_core::DType::BF16 | candle_core::DType::F16
                )
            {
                tensor.to_dtype(candle_core::DType::F32).map_err(Into::into)
            } else {
                Ok(tensor)
            }
        })
    }

    fn materialize_for_cache(
        &self,
        name: &str,
        device: &Device,
        attached: &AttachedDeviceCache,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if attached.cache.policy().cuda_allocator == CudaWeightAllocator::Direct
            && let Device::Cuda(cuda) = device
        {
            let metadata = self.raw_tensor_metadata(name)?;
            let dtype = produced_dtype(metadata.dtype, device)?;
            if metadata.bytes > 0
                && cuda_allocation::supports(dtype)
                && attached
                    .cache
                    .can_retain(attached.store, name, device, metadata.bytes as u64)
            {
                return self.with_source(name, |view, source| {
                    cuda_allocation::upload(view.data(), source, view.shape(), dtype, cuda)
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = attached;
        self.materialize(name, device)
    }

    /// One rank's slice of a rank-two tensor, materialized on `device` and
    /// retained under the slice's own cache identity.
    ///
    /// This is the loader a tensor split calls for. `load_rows` forgoes the
    /// cache entirely because a gathered subset must
    /// not be keyed by the tensor's name; a tensor split cannot forgo it and
    /// keep its point, so the partition is part of the key instead. A
    /// `TensorPartition::Whole` request is `load`, cache and all, which is why
    /// nothing changes for a caller that does not split.
    ///
    /// `TensorAxis::Rows` is a contiguous range of a row-major checkpoint and
    /// splits a linear's output features (Megatron's column-parallel case);
    /// `TensorAxis::Columns` gathers a strided range and splits its input
    /// features (the row-parallel case). Neither reorders anything within the
    /// slice, so a rank computes over exactly the released numbers.
    pub fn load_shard(
        &self,
        name: &str,
        partition: TensorPartition,
        device: &Device,
    ) -> Result<Tensor> {
        if partition == TensorPartition::Whole {
            return self.load(name, device);
        }
        self.access.record_tensor_read(name);
        let dtype = produced_dtype(self.raw_tensor_metadata(name)?.dtype, device)?;
        if let Some(attached) = self.device_cache.as_ref()
            && let Some(tensor) = attached
                .cache
                .get(attached.store, name, partition, device, dtype)
        {
            return Ok(tensor);
        }
        let tensor = self.materialize_shard(name, partition, device)?;
        self.access
            .device_row_materializations
            .fetch_add(1, Ordering::Relaxed);
        if let Some(attached) = self.device_cache.as_ref() {
            attached
                .cache
                .insert(attached.store, name, partition, device, &tensor);
        }
        Ok(tensor)
    }

    fn materialize_shard(
        &self,
        name: &str,
        partition: TensorPartition,
        device: &Device,
    ) -> Result<Tensor> {
        self.with_view(name, |view, _| {
            anyhow::ensure!(
                view.shape().len() == 2,
                "a tensor split requires a 2-D tensor; {name} is {:?}",
                view.shape()
            );
            let (rows, columns) = (view.shape()[0], view.shape()[1]);
            let dtype = candle_core::DType::try_from(view.dtype())?;
            anyhow::ensure!(
                view.dtype().bitsize().is_multiple_of(8),
                "a tensor split does not support sub-byte tensor dtypes"
            );
            let element = dtype.size_in_bytes();
            let row_bytes = columns
                .checked_mul(element)
                .context("split row size overflow")?;
            let data = view.data();
            let (shape, bytes) = match partition.axis() {
                TensorAxis::Rows => {
                    let ranges = partition.ranges(rows)?;
                    let share = partition.extent(rows)?;
                    let mut gathered = Vec::with_capacity(
                        share
                            .checked_mul(row_bytes)
                            .context("split row size overflow")?,
                    );
                    for range in ranges {
                        gathered.extend_from_slice(
                            &data[range.start * row_bytes..range.end * row_bytes],
                        );
                    }
                    (vec![share, columns], gathered)
                }
                TensorAxis::Columns => {
                    let ranges = partition.ranges(columns)?;
                    let share = partition.extent(columns)?;
                    let mut gathered = Vec::with_capacity(
                        rows.checked_mul(share)
                            .and_then(|count| count.checked_mul(element))
                            .context("split column size overflow")?,
                    );
                    for row in 0..rows {
                        for range in &ranges {
                            let start = row * row_bytes + range.start * element;
                            let length = (range.end - range.start) * element;
                            gathered.extend_from_slice(&data[start..start + length]);
                        }
                    }
                    (vec![rows, share], gathered)
                }
            };
            let tensor = Tensor::from_raw_buffer(&bytes, dtype, &shape, device)
                .with_context(|| format!("failed to materialize a split of {name}"))?;
            if device.is_cpu()
                && matches!(dtype, candle_core::DType::BF16 | candle_core::DType::F16)
            {
                tensor.to_dtype(candle_core::DType::F32).map_err(Into::into)
            } else {
                Ok(tensor)
            }
        })
    }

    pub fn load_rows(&self, name: &str, rows: &[u32], device: &Device) -> Result<Tensor> {
        self.access.record_tensor_read(name);
        let tensor = self.with_view(name, |view, _| {
            anyhow::ensure!(
                view.shape().len() == 2,
                "row gathering requires a 2-D tensor"
            );
            let source_rows = view.shape()[0];
            let columns = view.shape()[1];
            let dtype = candle_core::DType::try_from(view.dtype())?;
            anyhow::ensure!(
                view.dtype().bitsize().is_multiple_of(8),
                "row gathering does not support sub-byte tensor dtypes"
            );
            let row_bytes = columns
                .checked_mul(dtype.size_in_bytes())
                .context("embedding row size overflow")?;
            let total_bytes = rows
                .len()
                .checked_mul(row_bytes)
                .context("gathered embedding size overflow")?;
            let mut gathered = Vec::with_capacity(total_bytes);
            for &row in rows {
                let row = row as usize;
                anyhow::ensure!(row < source_rows, "row {row} is outside tensor {name}");
                let start = row * row_bytes;
                gathered.extend_from_slice(&view.data()[start..start + row_bytes]);
            }
            let tensor = Tensor::from_raw_buffer(&gathered, dtype, &[rows.len(), columns], device)
                .with_context(|| format!("failed to materialize gathered rows from {name}"))?;
            if device.is_cpu()
                && matches!(dtype, candle_core::DType::BF16 | candle_core::DType::F16)
            {
                tensor.to_dtype(candle_core::DType::F32).map_err(Into::into)
            } else {
                Ok(tensor)
            }
        })?;
        self.access
            .device_row_materializations
            .fetch_add(1, Ordering::Relaxed);
        Ok(tensor)
    }

    pub fn with_group<T>(
        &self,
        names: &[&str],
        device: &Device,
        f: impl FnOnce(&BTreeMap<String, Tensor>) -> Result<T>,
    ) -> Result<T> {
        let mut tensors = BTreeMap::new();
        for &name in names {
            tensors.insert(name.to_owned(), self.load(name, device)?);
        }
        f(&tensors)
    }

    pub fn verify(&self) -> Result<VerificationReport> {
        let mut by_shard: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (name, shard) in &self.index.weight_map {
            by_shard.entry(shard).or_default().push(name);
        }
        for (shard_name, names) in &by_shard {
            let shard_path = self.root.join(shard_name);
            let header = self.cache.headers.get(&shard_path)?;
            for name in names {
                header.tensor_info(name).with_context(|| {
                    format!(
                        "index maps {name} to {}, but it is absent",
                        shard_path.display()
                    )
                })?;
            }
        }
        Ok(VerificationReport {
            checked_shards: by_shard.len(),
            checked_tensors: self.index.weight_map.len(),
        })
    }

    pub fn cache_stats(&self) -> CacheStats {
        match &self.tensor_cache {
            Some(cache) => cache.stats(self.cache.headers.parses()),
            None => self.cache.stats(),
        }
    }

    fn with_view<T>(
        &self,
        name: &str,
        f: impl FnOnce(&safetensors::tensor::TensorView<'_>, &Path) -> Result<T>,
    ) -> Result<T> {
        let shard_name = self.index.weight_map.get(name).with_context(|| {
            format!(
                "tensor is not present in {}: {name}",
                self.index_path.display()
            )
        })?;
        let shard_path = self.root.join(shard_name);
        if let Some(cache) = &self.tensor_cache {
            let header = self.cache.headers.get(&shard_path)?;
            let lookup = cache.lookup(&shard_path, name, &header)?;
            return f(&lookup.data.view()?, &shard_path);
        }
        let shard = self.cache.get(&shard_path)?;
        let view = shard.tensor_view(name).with_context(|| {
            format!(
                "index maps {name} to {}, but it is absent",
                shard_path.display()
            )
        })?;
        f(&view, &shard_path)
    }

    /// Like `with_view`, with an optional backing file, offset and length for uploads.
    #[cfg(feature = "cuda")]
    fn with_source<T>(
        &self,
        name: &str,
        f: impl FnOnce(&safetensors::tensor::TensorView<'_>, Option<(&File, usize, usize)>) -> Result<T>,
    ) -> Result<T> {
        let shard_name = self.index.weight_map.get(name).with_context(|| {
            format!(
                "tensor is not present in {}: {name}",
                self.index_path.display()
            )
        })?;
        let shard_path = self.root.join(shard_name);
        let shard = self.cache.get(&shard_path)?;
        let view = shard.tensor_view(name).with_context(|| {
            format!(
                "index maps {name} to {}, but it is absent",
                shard_path.display()
            )
        })?;
        f(&view, shard.tensor_file_range(name))
    }
}

fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        bail!("unsafe shard path in safetensors index: {name}");
    }
    Ok(())
}

/// Recognize typed CUDA allocation failures through Candle error wrappers.
#[cfg(feature = "cuda")]
pub fn is_cuda_allocation_error(error: &(dyn std::error::Error + 'static)) -> bool {
    use candle_core::{
        Error,
        cuda_backend::{
            CudaError,
            cudarc::driver::{DriverError, sys},
        },
    };
    if let Some(error) = error.downcast_ref::<DriverError>() {
        return error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    }
    if let Some(error) = error.downcast_ref::<CudaError>() {
        return match error {
            CudaError::Cuda(driver) | CudaError::Load { cuda: driver, .. } => {
                is_cuda_allocation_error(driver)
            }
            _ => false,
        };
    }
    if let Some(error) = error.downcast_ref::<Error>() {
        return match error {
            Error::Cuda(inner) | Error::WrappedContext { wrapped: inner, .. } => {
                is_cuda_allocation_error(inner.as_ref())
            }
            Error::Context { inner, .. }
            | Error::WithPath { inner, .. }
            | Error::WithBacktrace { inner, .. } => is_cuda_allocation_error(inner.as_ref()),
            _ => false,
        };
    }
    error.source().is_some_and(is_cuda_allocation_error)
}

/// Whether an error came from the compute device rather than from the index,
/// the shard header or host I/O. Only `view.load` and the dtype promotion in
/// [`ModelWeights::materialize`] reach the device, and both surface a
/// `candle_core::Error`; everything else in that path is this crate's own
/// `anyhow` error or a `std::io::Error`.
fn is_device_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if cause.is::<candle_core::Error>() {
            return true;
        }
        #[cfg(feature = "cuda")]
        if cause.is::<candle_core::cuda_backend::cudarc::driver::DriverError>() {
            return true;
        }
        false
    })
}

/// The dtype `load` returns for a stored tensor on this device, including the
/// BF16/F16 to F32 promotion a CPU device applies. The device cache is keyed
/// on this dtype, and phase working-set estimates charge in it.
pub(crate) fn produced_dtype(raw: Dtype, device: &Device) -> Result<candle_core::DType> {
    let dtype = candle_core::DType::try_from(raw)?;
    Ok(
        if device.is_cpu() && matches!(dtype, candle_core::DType::BF16 | candle_core::DType::F16) {
            candle_core::DType::F32
        } else {
            dtype
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use serde_json::json;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        Tensor::new(&[1f32, 2., 3.], &Device::Cpu)
            .unwrap()
            .save_safetensors("a", dir.path().join("one.safetensors"))
            .unwrap();
        Tensor::new(&[[4f32, 5.], [6., 7.]], &Device::Cpu)
            .unwrap()
            .save_safetensors("b", dir.path().join("two.safetensors"))
            .unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 28},
                "weight_map": {"a": "one.safetensors", "b": "two.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        dir
    }

    /// Dropping a shard's cached pages is a hint to the kernel, not a change to
    /// what the shard holds: the next read faults the same bytes back. Forcing
    /// the flag on a one-shard cache makes every load evict and re-map, so the
    /// advice fires between each pair of reads here.
    #[test]
    fn dropping_evicted_pages_does_not_change_what_is_read() {
        let root = fixture();
        let mut weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights.drop_evicted_pages(true);
        for _ in 0..4 {
            let a = weights.load("a", &Device::Cpu).unwrap();
            assert_eq!(a.to_vec1::<f32>().unwrap(), vec![1., 2., 3.]);
            let b = weights.load("b", &Device::Cpu).unwrap();
            assert_eq!(
                b.to_vec2::<f32>().unwrap(),
                vec![vec![4., 5.], vec![6., 7.]]
            );
        }
        assert!(weights.cache_stats().evictions > 0);
    }

    /// The mechanism is off unless a caller turns it on, and a caller cannot
    /// turn it on for a source with no page cache behind it.
    #[test]
    fn page_dropping_is_off_until_asked_for_and_only_where_it_means_something() {
        let root = fixture();
        let mut weights =
            ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert!(!weights.cache.drop_evicted_pages);
        weights.drop_evicted_pages(true);
        assert!(weights.cache.drop_evicted_pages);
        weights.drop_evicted_pages(false);
        assert!(!weights.cache.drop_evicted_pages);

        let mut memory =
            ModelWeights::open(root.path(), WeightSource::Memory, CachePolicy::new(1)).unwrap();
        memory.drop_evicted_pages(true);
        assert!(!memory.cache.drop_evicted_pages);
    }

    #[test]
    fn accepts_extensible_index_metadata_without_relaxing_weight_map_validation() {
        let dir = fixture();
        let path = dir.path().join("model.safetensors.index.json");
        let mut index: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        index["metadata"]["total_parameters"] = json!(7);
        fs::write(&path, serde_json::to_vec(&index).unwrap()).unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert_eq!(weights.verify().unwrap().checked_tensors, 2);
        assert_eq!(weights.inventory().indexed_bytes, Some(28));
        index["weight_map"]["a"] = json!("../outside.safetensors");
        fs::write(&path, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).is_err());
    }

    fn differently_sized_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (tensor, shard, elements) in [
            ("a", "small.safetensors", 4usize),
            ("b", "large.safetensors", 128usize),
            ("c", "medium.safetensors", 16usize),
        ] {
            Tensor::from_vec(vec![1f32; elements], elements, &Device::Cpu)
                .unwrap()
                .save_safetensors(tensor, dir.path().join(shard))
                .unwrap();
        }
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 592},
                "weight_map": {
                    "a": "small.safetensors",
                    "b": "large.safetensors",
                    "c": "medium.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        dir
    }

    fn file_bytes(directory: &Path, name: &str) -> u64 {
        fs::metadata(directory.join(name)).unwrap().len()
    }

    #[test]
    fn mmap_loads_real_tensor_and_evicts_shards() {
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert_eq!(
            weights.inventory(),
            WeightInventory {
                tensors: 2,
                shards: 2,
                indexed_bytes: Some(28)
            }
        );
        assert_eq!(
            weights.max_shard_file_bytes().unwrap(),
            file_bytes(dir.path(), "one.safetensors")
                .max(file_bytes(dir.path(), "two.safetensors"))
        );
        assert_eq!(
            weights
                .load("a", &Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 2., 3.]
        );
        assert_eq!(weights.load("b", &Device::Cpu).unwrap().dims(), &[2, 2]);
        let stats = weights.cache_stats();
        assert_eq!(stats.resident_shards, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.max_shards, 1);
        assert_eq!(stats.max_bytes, None);
        assert_eq!(stats.memory_source_reads, 0);
        assert_eq!(stats.memory_source_read_bytes, 0);
        assert!(!stats.over_budget);
    }

    #[test]
    fn memory_mode_reuses_cached_shard() {
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Memory, CachePolicy::new(2)).unwrap();
        weights.metadata("a").unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.header_parses, 1);
        assert_eq!(stats.memory_source_reads, 1);
        assert_eq!(
            stats.memory_source_read_bytes,
            file_bytes(dir.path(), "one.safetensors")
        );
    }

    #[test]
    fn borrowed_tensor_bytes_survive_cache_eviction_without_materializing() {
        let dir = fixture();
        for source in [WeightSource::Mmap, WeightSource::Memory] {
            for granularity in [CacheGranularity::Shard, CacheGranularity::Tensor] {
                let weights = ModelWeights::open(
                    dir.path(),
                    source,
                    CachePolicy::new(1).with_granularity(granularity),
                )
                .unwrap();
                let expected = [1f32, 2., 3.]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>();
                weights
                    .with_tensor_bytes("a", |bytes| {
                        weights.with_tensor_bytes("b", |other| {
                            assert_eq!(other.len(), 16);
                            Ok(())
                        })?;
                        assert_eq!(bytes, expected);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(weights.access_stats().device_tensor_materializations, 0);
                assert!(weights.with_tensor_bytes("missing", |_| Ok(())).is_err());
                assert!(
                    weights
                        .with_tensor_bytes("a", |_| -> Result<()> { bail!("caller failure") })
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn header_catalog_is_independent_from_payload_residency_and_parses_once() {
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Memory, CachePolicy::new(1)).unwrap();

        weights.metadata("a").unwrap();
        weights.metadata("a").unwrap();
        let metadata_only = weights.cache_stats();
        assert_eq!(metadata_only.header_parses, 1);
        assert_eq!(metadata_only.misses, 0);
        assert_eq!(metadata_only.resident_shards, 0);
        assert_eq!(metadata_only.memory_source_reads, 0);

        weights.load("a", &Device::Cpu).unwrap();
        weights.load("b", &Device::Cpu).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        let after_reload = weights.cache_stats();
        assert_eq!(after_reload.header_parses, 2);
        assert_eq!(after_reload.misses, 3);
        assert_eq!(after_reload.evictions, 2);
        assert_eq!(after_reload.memory_source_reads, 3);
    }

    #[test]
    fn byte_budget_evicts_the_least_recently_used_real_shard_bytes() {
        let dir = differently_sized_fixture();
        let small_bytes = file_bytes(dir.path(), "small.safetensors");
        let large_bytes = file_bytes(dir.path(), "large.safetensors");
        let medium_bytes = file_bytes(dir.path(), "medium.safetensors");
        assert!(small_bytes < medium_bytes && medium_bytes < large_bytes);
        let byte_budget = small_bytes + large_bytes;
        let weights = ModelWeights::open(
            dir.path(),
            WeightSource::Memory,
            CachePolicy::new(3).with_max_bytes(byte_budget),
        )
        .unwrap();

        weights
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        weights
            .cache
            .get(&dir.path().join("large.safetensors"))
            .unwrap();
        weights
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        weights
            .cache
            .get(&dir.path().join("medium.safetensors"))
            .unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.resident_shards, 2);
        assert_eq!(stats.resident_bytes, small_bytes + medium_bytes);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
        assert!(!stats.over_budget);

        weights
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        weights
            .cache
            .get(&dir.path().join("large.safetensors"))
            .unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 2);
        assert_eq!(stats.resident_bytes, byte_budget);
    }

    #[test]
    fn byte_ceiling_without_a_unit_bound_allows_multiple_shards_to_coreside() {
        let dir = differently_sized_fixture();
        let small_bytes = file_bytes(dir.path(), "small.safetensors");
        let large_bytes = file_bytes(dir.path(), "large.safetensors");
        let medium_bytes = file_bytes(dir.path(), "medium.safetensors");
        let byte_budget = small_bytes + medium_bytes;
        assert!(large_bytes > byte_budget);

        let unit_bound = ModelWeights::open(
            dir.path(),
            WeightSource::Memory,
            CachePolicy::new(1).with_max_bytes(byte_budget),
        )
        .unwrap();
        unit_bound
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        unit_bound
            .cache
            .get(&dir.path().join("medium.safetensors"))
            .unwrap();
        assert_eq!(unit_bound.cache_stats().resident_shards, 1);

        let weights = ModelWeights::open(
            dir.path(),
            WeightSource::Memory,
            CachePolicy::unbounded_units().with_max_bytes(byte_budget),
        )
        .unwrap();
        weights
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        weights
            .cache
            .get(&dir.path().join("medium.safetensors"))
            .unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.resident_shards, 2);
        assert_eq!(stats.resident_bytes, byte_budget);
        assert_eq!(stats.max_shards, usize::MAX);
        assert_eq!(stats.max_bytes, Some(byte_budget));
        assert!(!stats.over_budget);

        weights
            .cache
            .get(&dir.path().join("large.safetensors"))
            .unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.resident_shards, 1);
        assert_eq!(stats.resident_bytes, large_bytes);
        assert!(stats.over_budget);
    }

    #[test]
    fn a_single_oversized_shard_is_retained_for_forward_progress() {
        let dir = differently_sized_fixture();
        let small_bytes = file_bytes(dir.path(), "small.safetensors");
        let large_bytes = file_bytes(dir.path(), "large.safetensors");
        let policy = CachePolicy::new(4).with_max_bytes(small_bytes);
        let weights = ModelWeights::open(dir.path(), WeightSource::Memory, policy).unwrap();

        weights
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        assert!(!weights.cache_stats().over_budget);
        weights
            .cache
            .get(&dir.path().join("large.safetensors"))
            .unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.resident_shards, 1);
        assert_eq!(stats.resident_bytes, large_bytes);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.max_shards, 4);
        assert_eq!(stats.max_bytes, Some(small_bytes));
        assert!(stats.over_budget);

        weights
            .cache
            .get(&dir.path().join("small.safetensors"))
            .unwrap();
        let stats = weights.cache_stats();
        assert_eq!(stats.resident_shards, 1);
        assert_eq!(stats.resident_bytes, small_bytes);
        assert_eq!(stats.evictions, 2);
        assert!(!stats.over_budget);
    }

    #[test]
    fn rejects_empty_cache_limits() {
        let dir = fixture();
        let error = ModelWeights::open(dir.path(), WeightSource::Memory, CachePolicy::new(0))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("capacity must be at least one"));

        let error = ModelWeights::open(
            dir.path(),
            WeightSource::Memory,
            CachePolicy::new(1).with_max_bytes(0),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("byte budget must be non-zero"));
    }

    #[test]
    fn gathers_embedding_rows_without_loading_the_table_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let embedding = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu).unwrap();
        embedding
            .save_safetensors("embedding", dir.path().join("table.safetensors"))
            .unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 24},
                "weight_map": {"embedding": "table.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert_eq!(
            weights
                .load_rows("embedding", &[2, 0, 2], &Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![5., 6.], vec![1., 2.], vec![5., 6.]]
        );
    }

    #[test]
    fn verifies_index_against_headers() {
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert_eq!(
            weights.verify().unwrap(),
            VerificationReport {
                checked_shards: 2,
                checked_tensors: 2
            }
        );
    }

    /// A 4x4 projection: enough to split either axis four ways and check every
    /// element of every rank's slice.
    fn projection_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        Tensor::from_vec(values, (4, 4), &Device::Cpu)
            .unwrap()
            .save_safetensors("proj.weight", dir.path().join("proj.safetensors"))
            .unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 64},
                "weight_map": {"proj.weight": "proj.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        dir
    }

    /// Both Megatron axes produce exactly the released numbers, in order: a row
    /// split takes whole output features, a column split takes the same input
    /// features out of every row.
    #[test]
    fn a_tensor_split_slices_the_released_weights_without_reordering_them() {
        let dir = projection_fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        for rank in 0..2u32 {
            let rows = weights
                .load_shard(
                    "proj.weight",
                    TensorPartition::Shard {
                        axis: TensorAxis::Rows,
                        rank,
                        ranks: 2,
                    },
                    &Device::Cpu,
                )
                .unwrap();
            assert_eq!(rows.dims(), [2, 4]);
            let expected = (0..8)
                .map(|index| (rank * 8 + index) as f32)
                .collect::<Vec<_>>();
            assert_eq!(
                rows.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected
            );

            let columns = weights
                .load_shard(
                    "proj.weight",
                    TensorPartition::Shard {
                        axis: TensorAxis::Columns,
                        rank,
                        ranks: 2,
                    },
                    &Device::Cpu,
                )
                .unwrap();
            assert_eq!(columns.dims(), [4, 2]);
            let expected = (0..4)
                .flat_map(|row| (0..2).map(move |column| (row * 4 + rank * 2 + column) as f32))
                .collect::<Vec<_>>();
            assert_eq!(
                columns.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected
            );
        }
        let whole = weights.load("proj.weight", &Device::Cpu).unwrap();
        assert_eq!(whole.dims(), [4, 4]);
    }

    /// Under a tensor split, a resident slice must never answer a request for
    /// the whole tensor, and two ranks' slices must not answer for each other —
    /// on one device, under one name, at one dtype, which is the case the old
    /// key could not tell apart.
    #[test]
    fn a_resident_split_never_answers_a_whole_tensor_read() {
        let dir = projection_fixture();
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(1 << 20)))
            .unwrap();
        let shard = |rank: u32| TensorPartition::Shard {
            axis: TensorAxis::Rows,
            rank,
            ranks: 2,
        };
        let first = weights
            .load_shard("proj.weight", shard(0), &Device::Cpu)
            .unwrap();
        assert_eq!(first.dims(), [2, 4]);
        let whole = weights.load("proj.weight", &Device::Cpu).unwrap();
        assert_eq!(whole.dims(), [4, 4]);
        let second = weights
            .load_shard("proj.weight", shard(1), &Device::Cpu)
            .unwrap();
        assert_eq!(
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (8..16).map(|value| value as f32).collect::<Vec<_>>()
        );
        let stats = weights.device_cache_stats();
        assert_eq!(stats.resident_tensors, 3);
        assert_eq!(stats.hits, 0);
        for (expected, tensor) in [
            (
                first,
                weights
                    .load_shard("proj.weight", shard(0), &Device::Cpu)
                    .unwrap(),
            ),
            (
                second,
                weights
                    .load_shard("proj.weight", shard(1), &Device::Cpu)
                    .unwrap(),
            ),
            (whole, weights.load("proj.weight", &Device::Cpu).unwrap()),
        ] {
            assert_eq!(
                tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
        assert_eq!(weights.device_cache_stats().hits, 3);
    }

    /// A fused gate/up projection is split segment by segment: each rank takes
    /// its share of the values *and* its share of the gates, so its own slice
    /// has the same internal layout the whole tensor has. A single contiguous
    /// range would hand rank 0 nothing but values and rank 1 nothing but gates.
    #[test]
    fn a_fused_projection_splits_each_of_its_segments() {
        let dir = projection_fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let fused = |rank: u32| TensorPartition::SegmentedShard {
            axis: TensorAxis::Rows,
            rank,
            ranks: 2,
            segments: 2,
        };
        let first = weights
            .load_shard("proj.weight", fused(0), &Device::Cpu)
            .unwrap();
        let second = weights
            .load_shard("proj.weight", fused(1), &Device::Cpu)
            .unwrap();
        assert_eq!(first.dims(), [2, 4]);
        assert_eq!(second.dims(), [2, 4]);
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0., 1., 2., 3., 8., 9., 10., 11.]
        );
        assert_eq!(
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![4., 5., 6., 7., 12., 13., 14., 15.]
        );
        let contiguous = weights
            .load_shard(
                "proj.weight",
                TensorPartition::Shard {
                    axis: TensorAxis::Rows,
                    rank: 0,
                    ranks: 2,
                },
                &Device::Cpu,
            )
            .unwrap();
        assert_ne!(
            contiguous.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_ne!(
            fused(0),
            TensorPartition::Shard {
                axis: TensorAxis::Rows,
                rank: 0,
                ranks: 2
            }
        );
        let mut covered = (0..2)
            .flat_map(|rank| fused(rank).ranges(4).unwrap())
            .flat_map(|range| range.collect::<Vec<_>>())
            .collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, (0..4).collect::<Vec<_>>());
        let columns = weights
            .load_shard(
                "proj.weight",
                TensorPartition::SegmentedShard {
                    axis: TensorAxis::Columns,
                    rank: 1,
                    ranks: 2,
                    segments: 2,
                },
                &Device::Cpu,
            )
            .unwrap();
        assert_eq!(columns.dims(), [4, 2]);
        assert_eq!(
            columns.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 3., 5., 7., 9., 11., 13., 15.]
        );
    }

    /// A segmented shard names several ranges by construction, so a caller that
    /// can only take one is refused rather than handed a span covering
    /// elements it does not hold.
    #[test]
    fn a_segmented_shard_refuses_to_collapse_into_one_range() {
        let segmented = TensorPartition::SegmentedShard {
            axis: TensorAxis::Rows,
            rank: 0,
            ranks: 2,
            segments: 2,
        };
        assert_eq!(segmented.ranges(8).unwrap(), vec![0..2, 4..6]);
        assert_eq!(segmented.extent(8).unwrap(), 4);
        let error = segmented.range(8).unwrap_err().to_string();
        assert!(error.contains("names 2 ranges"), "{error}");
        assert!(segmented.ranges(9).is_err());
        assert!(
            TensorPartition::SegmentedShard {
                axis: TensorAxis::Rows,
                rank: 0,
                ranks: 4,
                segments: 2,
            }
            .ranges(12)
            .is_err()
        );
    }

    /// A split that does not divide the axis is refused rather than rounded: a
    /// plausible wrong slice is worse than a failure at planning time.
    #[test]
    fn an_uneven_split_is_refused() {
        let dir = projection_fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let error = weights
            .load_shard(
                "proj.weight",
                TensorPartition::Shard {
                    axis: TensorAxis::Rows,
                    rank: 0,
                    ranks: 3,
                },
                &Device::Cpu,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not divide"), "{error}");
        let error = weights
            .load_shard(
                "proj.weight",
                TensorPartition::Shard {
                    axis: TensorAxis::Rows,
                    rank: 2,
                    ranks: 2,
                },
                &Device::Cpu,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside a 2-rank split"), "{error}");
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let error = weights
            .load_shard(
                "a",
                TensorPartition::Shard {
                    axis: TensorAxis::Rows,
                    rank: 0,
                    ranks: 1,
                },
                &Device::Cpu,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires a 2-D tensor"), "{error}");
    }

    #[test]
    fn device_cache_hits_are_bit_identical_and_bounded() {
        let dir = differently_sized_fixture();
        let budget = 16 + 64;
        let mut cached =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        cached
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(budget)))
            .unwrap();
        cached.count_tensor_reads(true);
        let uncached =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();

        for name in ["a", "b", "c", "a", "b", "c"] {
            let resident = cached.load(name, &Device::Cpu).unwrap();
            let streamed = uncached.load(name, &Device::Cpu).unwrap();
            assert_eq!(
                resident.to_vec1::<f32>().unwrap(),
                streamed.to_vec1::<f32>().unwrap(),
                "cached and uncached loads of {name} differ"
            );
        }
        let stats = cached.device_cache_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.resident_tensors, 2);
        assert_eq!(stats.resident_bytes, budget);
        assert_eq!(stats.max_bytes, budget);
        assert!(!stats.demoted);
        assert_eq!(cached.access_stats().device_tensor_materializations, 4);
        assert_eq!(
            cached.tensor_reads(),
            BTreeMap::from([
                ("a".to_owned(), 2),
                ("b".to_owned(), 2),
                ("c".to_owned(), 2)
            ])
        );
    }

    #[test]
    fn device_cache_zero_budget_preserves_the_streaming_baseline() {
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert_eq!(weights.device_cache_policy(), DeviceCachePolicy::DISABLED);
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        let stats = weights.device_cache_stats();
        assert_eq!(
            stats,
            DeviceCacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                resident_tensors: 0,
                resident_bytes: 0,
                max_bytes: 0,
                demoted: false,
                prioritized_bytes: 0,
                resident_bytes_by_device: Vec::new(),
                priority_bypasses: 0,
                demotion: None,
            }
        );
        assert_eq!(weights.access_stats().device_tensor_materializations, 2);
    }

    #[test]
    fn device_cache_keeps_an_oversized_tensor_out_without_evicting() {
        let dir = differently_sized_fixture();
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(80)))
            .unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("c", &Device::Cpu).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("b", &Device::Cpu).unwrap();
        weights.load("c", &Device::Cpu).unwrap();
        let stats = weights.device_cache_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.resident_bytes, 80);
    }

    #[test]
    fn device_cache_evicts_least_recently_used_under_pressure() {
        let dir = differently_sized_fixture();
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(64)))
            .unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("c", &Device::Cpu).unwrap();
        let stats = weights.device_cache_stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_tensors, 1);
        assert_eq!(stats.resident_bytes, 64);
        weights.load("a", &Device::Cpu).unwrap();
        let stats = weights.device_cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 2);
        assert_eq!(stats.resident_bytes, 16);
    }

    #[test]
    fn one_shared_budget_bounds_every_attached_store() {
        let dir = differently_sized_fixture();
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(64));
        let mut first =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        first.configure_device_cache(cache.clone()).unwrap();
        let mut second =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        second.configure_device_cache(cache.clone()).unwrap();

        first.load("c", &Device::Cpu).unwrap();
        second.load("c", &Device::Cpu).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.resident_bytes, 64);
        assert_eq!(stats.resident_tensors, 1);
        assert_eq!(stats.evictions, 1);
        assert_eq!(first.device_cache_stats(), stats);
        assert_eq!(second.device_cache_stats(), stats);
    }

    #[test]
    fn priority_configuration_protects_only_declared_tensors_without_changing_values() {
        let dir = differently_sized_fixture();
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(544));
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert!(
            weights
                .configure_device_cache_with_priority(cache.clone(), ["missing".to_owned()])
                .is_err()
        );
        weights
            .configure_device_cache_with_priority(cache.clone(), ["c".to_owned()])
            .unwrap();
        assert_eq!(weights.access_stats().device_tensor_materializations, 0);
        let plain =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        for name in ["c", "a", "b", "c", "a"] {
            assert_eq!(
                weights
                    .load(name, &Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                plain
                    .load(name, &Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );
        }
        let stats = cache.stats();
        assert_eq!(stats.prioritized_bytes, 64);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.priority_bypasses, 1);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.resident_bytes, 80);
    }

    #[test]
    fn sequential_phase_plan_matches_store_release_and_cache_reuse() {
        use crate::residency::{PhaseResidencyDemand, WeightPhase, plan_device_residency};
        let dir = differently_sized_fixture();
        let metadata =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let demands = (0..2)
            .map(|index| {
                let phase = WeightPhase::new(format!("phase{index}"), ["c".to_owned()], 2);
                let mut demand =
                    PhaseResidencyDemand::from_phase(&phase, &metadata, &Device::Cpu).unwrap();
                demand.lifetime = Some(index..index + 1);
                demand
            })
            .collect::<Vec<_>>();
        assert_eq!(metadata.access_stats().device_tensor_materializations, 0);
        let plan = plan_device_residency(&demands, 64, 0).unwrap();
        assert_eq!(plan.placed.len(), 2);
        assert_eq!(plan.resident_bytes, 64);
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(plan.budget_bytes));
        for _ in 0..2 {
            assert_eq!(cache.stats().resident_bytes, 0);
            {
                let mut weights =
                    ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1))
                        .unwrap();
                weights.configure_device_cache(cache.clone()).unwrap();
                let first = weights.load("c", &Device::Cpu).unwrap();
                let second = weights.load("c", &Device::Cpu).unwrap();
                assert_eq!(
                    first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    second.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                );
                assert_eq!(cache.stats().resident_bytes, 64);
                assert_eq!(weights.access_stats().device_tensor_materializations, 1);
            }
            assert_eq!(cache.stats().resident_bytes, 0);
        }
        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().evictions, 0);
    }

    #[test]
    fn device_cache_cannot_be_configured_after_payload_access() {
        let dir = fixture();
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(64)))
            .unwrap();
        let mut second =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        second.load("a", &Device::Cpu).unwrap();
        let error = second
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(64)))
            .unwrap_err()
            .to_string();
        assert!(error.contains("after payload access"));
        drop(weights);
    }

    #[test]
    fn read_counting_is_off_until_a_caller_asks_for_it() {
        let dir = fixture();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        assert!(weights.tensor_reads().is_empty());

        weights.count_tensor_reads(true);
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        assert_eq!(
            weights.tensor_reads(),
            BTreeMap::from([("a".to_owned(), 2)]),
            "only reads after the switch are counted"
        );

        weights.count_tensor_reads(false);
        weights.load("a", &Device::Cpu).unwrap();
        assert_eq!(
            weights.tensor_reads(),
            BTreeMap::from([("a".to_owned(), 2)])
        );
    }

    #[test]
    fn dropping_a_store_hands_its_budget_back() {
        let dir = differently_sized_fixture();
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(1 << 20));
        let mut first =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        first.configure_device_cache(cache.clone()).unwrap();
        let mut second =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        second.configure_device_cache(cache.clone()).unwrap();

        first.load("c", &Device::Cpu).unwrap();
        second.load("a", &Device::Cpu).unwrap();
        assert_eq!(cache.stats().resident_bytes, 80);
        assert_eq!(cache.stats().resident_tensors, 2);

        drop(first);
        let stats = cache.stats();
        assert_eq!(stats.resident_bytes, 16, "the finished store let go");
        assert_eq!(stats.resident_tensors, 1);
        assert!(second.device_resident("a", &Device::Cpu).unwrap());

        drop(second);
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[test]
    fn only_a_device_error_gives_up_the_resident_set() {
        let candle = anyhow::Error::new(candle_core::Error::Msg("out of memory".into()))
            .context("failed to materialize tensor w");
        assert!(is_device_error(&candle));

        let missing = anyhow::anyhow!("tensor is not present in index.json: w");
        assert!(!is_device_error(&missing));

        let io = anyhow::Error::new(std::io::Error::other("shard read failed"))
            .context("failed to read shard 0");
        assert!(!is_device_error(&io));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn direct_cuda_reclamation_errors_keep_the_device_fallback_path() {
        use candle_core::cuda_backend::cudarc::driver::{DriverError, sys};
        let error = anyhow::Error::new(DriverError(sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY))
            .context("preparing cache space");
        assert!(is_device_error(&error));
    }

    #[test]
    fn a_missing_tensor_keeps_the_resident_set() {
        let dir = differently_sized_fixture();
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(1 << 20)))
            .unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        assert!(weights.load("absent", &Device::Cpu).is_err());
        let stats = weights.device_cache_stats();
        assert_eq!(stats.resident_tensors, 1);
        assert!(!stats.demoted);
    }

    #[test]
    fn a_missing_payload_does_not_pre_evict_resident_weights() {
        let dir = differently_sized_fixture();
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(64));
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights.configure_device_cache(cache.clone()).unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        weights.metadata("c").unwrap();
        fs::remove_file(dir.path().join("medium.safetensors")).unwrap();
        assert!(weights.load("c", &Device::Cpu).is_err());
        assert_eq!(cache.stats().resident_bytes, 16);
        assert_eq!(cache.stats().evictions, 0);
        assert!(!cache.stats().demoted);
    }

    #[test]
    fn device_cache_keys_on_the_dtype_the_loader_produces() {
        let dir = tempfile::tempdir().unwrap();
        let raw = [0x00u8, 0x80, 0x00, 0x00, 0x00, 0x3f, 0x80, 0x3f];
        let view = safetensors::tensor::TensorView::new(Dtype::BF16, vec![4], &raw).unwrap();
        safetensors::tensor::serialize_to_file(
            [("raw", view)],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let mut weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(1024)))
            .unwrap();
        let first = weights.load("raw", &Device::Cpu).unwrap();
        assert_eq!(first.dtype(), candle_core::DType::F32);
        let second = weights.load("raw", &Device::Cpu).unwrap();
        assert_eq!(
            first.to_vec1::<f32>().unwrap(),
            second.to_vec1::<f32>().unwrap()
        );
        let stats = weights.device_cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.resident_bytes, 16);
    }

    #[test]
    fn rejects_index_path_traversal() {
        let dir = fixture();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 12},
                "weight_map": {"a": "../outside.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("unsafe shard path"));
    }

    #[test]
    fn opens_unsharded_safetensors_without_an_index() {
        let dir = tempfile::tempdir().unwrap();
        Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)
            .unwrap()
            .save_safetensors(
                "value",
                dir.path().join("diffusion_pytorch_model.safetensors"),
            )
            .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        assert_eq!(weights.inventory().tensors, 1);
        assert_eq!(weights.inventory().indexed_bytes, Some(16));
        assert_eq!(
            weights
                .load("value", &Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1., 2.], vec![3., 4.]]
        );
    }
}
