//! Raw, independently retained tensor ranges. No dtype conversion happens here.
use super::{
    CachePolicy, CacheStats, CachedTensorInfo, HostCachePriorityStats, ShardBytes, ShardHeader,
    TensorRetentionStats, WeightSource,
};
use anyhow::{Context, Result, ensure};
use memmap2::MmapOptions;
use safetensors::tensor::TensorView;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Key {
    path: PathBuf,
    name: String,
}

pub(super) struct TensorData {
    storage: ShardBytes,
    info: CachedTensorInfo,
}

impl TensorData {
    pub(super) fn view(&self) -> Result<TensorView<'_>> {
        TensorView::new(
            self.info.dtype,
            self.info.shape.clone(),
            self.storage.bytes(),
        )
        .map_err(Into::into)
    }
}

pub(super) struct TensorLookup {
    pub data: Arc<TensorData>,
}

#[derive(Default)]
struct State {
    entries: HashMap<Key, Arc<TensorData>>,
    source_shards: HashMap<PathBuf, usize>,
    recency: VecDeque<Key>,
    bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    memory_reads: u64,
    memory_read_bytes: u64,
    priority_names: HashSet<String>,
    priority_reserved_bytes: u64,
    priority_resident_bytes: u64,
    priority_bypasses: u64,
}

impl State {
    fn touch(&mut self, key: &Key) {
        if let Some(index) = self.recency.iter().position(|value| value == key) {
            self.recency.remove(index);
        }
        self.recency.push_back(key.clone());
    }

    fn evict(&mut self, allow_priority: bool) -> bool {
        let position = self
            .recency
            .iter()
            .position(|key| !self.priority_names.contains(&key.name))
            .or_else(|| (allow_priority && !self.recency.is_empty()).then_some(0));
        let Some(position) = position else {
            return false;
        };
        let key = self.recency[position].clone();
        self.remove(&key)
    }

    fn remove(&mut self, key: &Key) -> bool {
        let Some(data) = self.entries.remove(key) else {
            return false;
        };
        self.recency.retain(|entry| entry != key);
        self.bytes = self
            .bytes
            .checked_sub(data.info.len as u64)
            .expect("tensor cache bytes are inconsistent");
        if self.priority_names.contains(&key.name) {
            self.priority_resident_bytes -= data.info.len as u64;
        }
        let count = self
            .source_shards
            .get_mut(&key.path)
            .expect("tensor source shard is missing");
        *count -= 1;
        if *count == 0 {
            self.source_shards.remove(&key.path);
        }
        self.evictions += 1;
        true
    }
}

pub(super) struct TensorCache {
    source: WeightSource,
    policy: CachePolicy,
    state: Mutex<State>,
}

impl TensorCache {
    pub(super) fn new(source: WeightSource, policy: CachePolicy) -> Self {
        Self {
            source,
            policy,
            state: Mutex::new(State::default()),
        }
    }

    pub(super) fn configure_priority(&self, names: HashSet<String>, bytes: u64) -> Result<()> {
        ensure!(
            self.policy.max_bytes.is_none_or(|limit| bytes <= limit),
            "host priority reservation exceeds cache ceiling"
        );
        let mut state = self.state.lock().expect("tensor cache mutex poisoned");
        ensure!(
            state.hits == 0 && state.misses == 0 && state.entries.is_empty(),
            "cannot configure host priority after payload access"
        );
        state.priority_names = names;
        state.priority_reserved_bytes = bytes;
        Ok(())
    }

    pub(super) fn release(&self, path: &Path, name: &str) {
        self.state
            .lock()
            .expect("tensor cache mutex poisoned")
            .remove(&Key {
                path: path.to_owned(),
                name: name.to_owned(),
            });
    }

    pub(super) fn lookup(
        &self,
        path: &Path,
        name: &str,
        header: &ShardHeader,
    ) -> Result<TensorLookup> {
        let key = Key {
            path: path.to_owned(),
            name: name.to_owned(),
        };
        {
            let mut state = self.state.lock().expect("tensor cache mutex poisoned");
            if let Some(data) = state.entries.get(&key).cloned() {
                state.hits += 1;
                state.touch(&key);
                return Ok(TensorLookup { data });
            }
            state.misses += 1;
        }
        let info = header.tensor_info(name)?.clone();
        let mut file = File::open(path)
            .with_context(|| format!("failed to open tensor source {}", path.display()))?;
        ensure!(
            file.metadata()?.len() == header.file_bytes,
            "tensor source shard length changed after header cataloging"
        );
        let mut encoded_header = vec![0; header.encoded_header.len()];
        file.read_exact(&mut encoded_header)?;
        ensure!(
            encoded_header.as_slice() == header.encoded_header.as_ref(),
            "tensor source shard header changed after cataloging"
        );
        let storage = if info.len == 0 {
            ShardBytes::Empty
        } else {
            match self.source {
                WeightSource::Mmap => {
                    let mapping = unsafe {
                        MmapOptions::new()
                            .offset(info.offset as u64)
                            .len(info.len)
                            .map(&file)
                    }
                    .with_context(|| format!("failed to map tensor {name}"))?;
                    // One entry here is one tensor's byte range, so an eviction
                    // says nothing about the rest of the file; page dropping is
                    // the shard cache's decision, not this one's.
                    ShardBytes::Mmap(super::MappedShard {
                        mapping,
                        droppable: None,
                    })
                }
                WeightSource::Memory => {
                    file.seek(SeekFrom::Start(info.offset as u64))?;
                    let mut bytes = vec![0; info.len];
                    file.read_exact(&mut bytes)
                        .with_context(|| format!("failed to read tensor {name}"))?;
                    ShardBytes::Memory(bytes.into_boxed_slice())
                }
            }
        };
        ensure!(
            file.metadata()?.len() == header.file_bytes,
            "tensor source length changed while loading"
        );
        let bytes = info.len as u64;
        let io_bytes = if self.source == WeightSource::Memory {
            bytes
                .checked_add(encoded_header.len() as u64)
                .context("tensor I/O byte count overflow")?
        } else {
            0
        };
        let data = Arc::new(TensorData { storage, info });
        let mut state = self.state.lock().expect("tensor cache mutex poisoned");
        if self.source == WeightSource::Memory {
            state.memory_reads += 1;
            state.memory_read_bytes = state
                .memory_read_bytes
                .checked_add(io_bytes)
                .context("tensor cache read byte count overflow")?;
        }
        if let Some(existing) = state.entries.get(&key).cloned() {
            state.touch(&key);
            return Ok(TensorLookup { data: existing });
        }
        let prioritized = state.priority_names.contains(name);
        let low_limit = if !state.priority_names.is_empty() && !prioritized {
            self.policy
                .max_bytes
                .map(|limit| limit - state.priority_reserved_bytes)
        } else {
            None
        };
        if low_limit.is_some_and(|limit| bytes > limit) {
            state.priority_bypasses += 1;
            return Ok(TensorLookup { data });
        }
        let allow_priority_eviction = prioritized || state.priority_names.is_empty();
        while !state.entries.is_empty()
            && ((!state.source_shards.contains_key(path)
                && state.source_shards.len() >= self.policy.max_shards)
                || self.policy.max_bytes.is_some_and(|limit| {
                    state.bytes.checked_add(bytes).is_none_or(|sum| sum > limit)
                })
                || low_limit.is_some_and(|limit| {
                    (state.bytes - state.priority_resident_bytes)
                        .checked_add(bytes)
                        .is_none_or(|sum| sum > limit)
                }))
        {
            if !state.evict(allow_priority_eviction) {
                state.priority_bypasses += 1;
                return Ok(TensorLookup { data });
            }
        }
        state.bytes = state
            .bytes
            .checked_add(bytes)
            .context("tensor cache byte count overflow")?;
        *state.source_shards.entry(path.to_owned()).or_default() += 1;
        state.entries.insert(key.clone(), Arc::clone(&data));
        if prioritized {
            state.priority_resident_bytes += bytes;
        }
        state.touch(&key);
        Ok(TensorLookup { data })
    }

    pub(super) fn stats(&self, header_parses: u64) -> CacheStats {
        let state = self.state.lock().expect("tensor cache mutex poisoned");
        CacheStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            header_parses,
            memory_source_reads: state.memory_reads,
            memory_source_read_bytes: state.memory_read_bytes,
            resident_shards: state.source_shards.len(),
            resident_bytes: state.bytes,
            max_shards: self.policy.max_shards,
            max_bytes: self.policy.max_bytes,
            over_budget: self
                .policy
                .max_bytes
                .is_some_and(|limit| state.bytes > limit),
            tensor_retention: Some(TensorRetentionStats {
                resident_tensors: state.entries.len(),
                priority: (!state.priority_names.is_empty()).then_some(HostCachePriorityStats {
                    reserved_bytes: state.priority_reserved_bytes,
                    resident_bytes: state.priority_resident_bytes,
                    bypasses: state.priority_bypasses,
                }),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use safetensors::tensor::serialize_to_file;

    fn fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let a = [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let b = [-1f32, -2., -3., -4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let padding = vec![17u8; 8192];
        let raw = [0x00, 0x80, 0x00, 0x00, 0xc1, 0x7f, 0x80, 0x3f];
        let tensors = BTreeMap::from([
            ("a", TensorView::new(Dtype::F32, vec![2, 2], &a).unwrap()),
            ("b", TensorView::new(Dtype::F32, vec![2, 2], &b).unwrap()),
            ("empty", TensorView::new(Dtype::F32, vec![0], &[]).unwrap()),
            (
                "padding",
                TensorView::new(Dtype::U8, vec![8192], &padding).unwrap(),
            ),
            ("raw", TensorView::new(Dtype::BF16, vec![4], &raw).unwrap()),
        ]);
        serialize_to_file(&tensors, None, &root.path().join("part.safetensors")).unwrap();
        let x = 9f32.to_le_bytes();
        serialize_to_file(
            [("x", TensorView::new(Dtype::F32, vec![1], &x).unwrap())],
            None,
            &root.path().join("other.safetensors"),
        )
        .unwrap();
        fs::write(root.path().join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": 8236},
            "weight_map": {"a":"part.safetensors", "b":"part.safetensors", "empty":"part.safetensors",
                "padding":"part.safetensors", "raw":"part.safetensors", "x":"other.safetensors"}
        })).unwrap()).unwrap();
        root
    }

    fn policy(bytes: u64) -> CachePolicy {
        CachePolicy::new(1)
            .with_max_bytes(bytes)
            .with_granularity(CacheGranularity::Tensor)
    }

    #[test]
    fn host_priority_reserves_capacity_before_fill_and_survives_static_reads() {
        for source in [WeightSource::Mmap, WeightSource::Memory] {
            let root = fixture();
            let mut weights = ModelWeights::open(
                root.path(),
                source,
                CachePolicy::unbounded_units()
                    .with_max_bytes(40)
                    .with_granularity(CacheGranularity::Tensor),
            )
            .unwrap();
            assert_eq!(
                weights
                    .configure_host_cache_priority(["a".into(), "b".into()])
                    .unwrap(),
                32
            );
            weights.load("padding", &Device::Cpu).unwrap();
            assert_eq!(weights.cache_stats().resident_bytes, 0);
            for name in ["a", "b", "x", "raw", "x", "padding"] {
                weights.load(name, &Device::Cpu).unwrap();
                assert!(weights.cache_stats().resident_bytes <= 40);
            }
            let before = weights.cache_stats();
            assert_eq!(
                weights
                    .load("a", &Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                [1., 2., 3., 4.]
            );
            assert_eq!(
                weights
                    .load("b", &Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                [-1., -2., -3., -4.]
            );
            let after = weights.cache_stats();
            assert_eq!(after.hits - before.hits, 2);
            let priority = after.tensor_retention.unwrap().priority.unwrap();
            assert_eq!(priority.reserved_bytes, 32);
            assert_eq!(priority.resident_bytes, 32);
            assert_eq!(priority.bypasses, 2);
            assert!(weights.configure_host_cache_priority(["a".into()]).is_err());
        }
    }

    #[test]
    fn low_priority_shard_cannot_displace_protected_host_tensors() {
        let root = fixture();
        let mut weights =
            ModelWeights::open(root.path(), WeightSource::Memory, policy(40)).unwrap();
        assert!(
            weights
                .configure_host_cache_priority(["missing".into()])
                .is_err()
        );
        assert!(
            weights
                .configure_host_cache_priority(["a".into(), "x".into()])
                .is_err()
        );
        assert!(
            weights
                .configure_host_cache_priority(["padding".into()])
                .is_err()
        );
        weights
            .configure_host_cache_priority(["a".into(), "b".into()])
            .unwrap();
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("b", &Device::Cpu).unwrap();
        assert_eq!(
            weights
                .load("x", &Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [9.]
        );
        let before = weights.cache_stats();
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("b", &Device::Cpu).unwrap();
        assert_eq!(weights.cache_stats().hits - before.hits, 2);
        assert_eq!(weights.cache_stats().resident_bytes, 32);
    }

    #[test]
    fn tensor_cache_retains_and_evicts_individual_units() {
        for source in [WeightSource::Mmap, WeightSource::Memory] {
            let root = fixture();
            let weights = ModelWeights::open(root.path(), source, policy(32)).unwrap();
            let before = weights.cache_stats();
            assert_eq!(before.tensor_retention.unwrap().resident_tensors, 0);
            for name in ["a", "b", "a"] {
                weights.load(name, &Device::Cpu).unwrap();
            }
            let stats = weights.cache_stats();
            assert_eq!((stats.hits, stats.misses, stats.evictions), (1, 2, 0));
            assert_eq!((stats.resident_shards, stats.resident_bytes), (1, 32));
            assert_eq!(stats.tensor_retention.unwrap().resident_tensors, 2);
            weights
                .with_view("raw", |view, _| {
                    assert_eq!(
                        view.data(),
                        &[0x00, 0x80, 0x00, 0x00, 0xc1, 0x7f, 0x80, 0x3f]
                    );
                    Ok(())
                })
                .unwrap();
            assert_eq!(weights.cache_stats().evictions, 1);
            assert_eq!(
                weights
                    .load_rows("b", &[1, 0], &Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap(),
                vec![vec![-3., -4.], vec![-1., -2.]]
            );
            weights.load("padding", &Device::Cpu).unwrap();
            let oversized = weights.cache_stats();
            assert!(oversized.over_budget);
            assert_eq!(oversized.resident_bytes, 8192);
            assert_eq!(oversized.tensor_retention.unwrap().resident_tensors, 1);
            weights.load("a", &Device::Cpu).unwrap();
            assert!(!weights.cache_stats().over_budget);
            assert_eq!(weights.cache_stats().header_parses, 1);
        }
    }

    #[test]
    fn memory_tensor_reads_do_not_load_whole_shards() {
        let root = fixture();
        let shard =
            ModelWeights::open(root.path(), WeightSource::Memory, CachePolicy::new(1)).unwrap();
        let tensor = ModelWeights::open(root.path(), WeightSource::Memory, policy(32)).unwrap();
        for name in ["a", "b"] {
            assert_eq!(
                shard
                    .load(name, &Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap(),
                tensor
                    .load(name, &Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
            );
        }
        let shard_stats = shard.cache_stats();
        let tensor_stats = tensor.cache_stats();
        assert_eq!(shard_stats.memory_source_reads, 1);
        assert_eq!(tensor_stats.memory_source_reads, 2);
        assert_eq!(
            shard_stats.resident_bytes,
            fs::metadata(root.path().join("part.safetensors"))
                .unwrap()
                .len()
        );
        assert_eq!(tensor_stats.resident_bytes, 32);
        assert!(tensor_stats.memory_source_read_bytes < shard_stats.memory_source_read_bytes);
        assert!(
            serde_json::to_value(&shard_stats)
                .unwrap()
                .get("tensor_retention")
                .is_none()
        );
        let encoded = serde_json::to_vec(&tensor_stats).unwrap();
        assert_eq!(
            serde_json::from_slice::<CacheStats>(&encoded).unwrap(),
            tensor_stats
        );
    }

    #[test]
    fn tensor_source_shard_limit_and_empty_ranges_are_explicit() {
        for source in [WeightSource::Mmap, WeightSource::Memory] {
            let root = fixture();
            let weights = ModelWeights::open(root.path(), source, policy(64)).unwrap();
            let empty = weights.load("empty", &Device::Cpu).unwrap();
            assert_eq!(empty.elem_count(), 0);
            assert_eq!(weights.cache_stats().resident_bytes, 0);
            assert_eq!(
                weights
                    .cache_stats()
                    .tensor_retention
                    .unwrap()
                    .resident_tensors,
                1
            );
            weights.load("a", &Device::Cpu).unwrap();
            weights.load("b", &Device::Cpu).unwrap();
            assert_eq!(
                weights
                    .load("x", &Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                [9.]
            );
            let stats = weights.cache_stats();
            assert_eq!(
                (stats.evictions, stats.resident_shards, stats.resident_bytes),
                (3, 1, 4)
            );
            assert_eq!(stats.tensor_retention.unwrap().resident_tensors, 1);
        }
    }

    #[test]
    fn tensor_cache_rejects_changed_catalog_header_before_payload_access() {
        for source in [WeightSource::Mmap, WeightSource::Memory] {
            let root = fixture();
            let weights = ModelWeights::open(root.path(), source, policy(32)).unwrap();
            weights.verify().unwrap();
            let path = root.path().join("part.safetensors");
            let mut bytes = fs::read(&path).unwrap();
            let offset = bytes
                .windows(3)
                .position(|window| window == b"\"a\"")
                .unwrap();
            bytes[offset + 1] = b'z';
            fs::write(path, bytes).unwrap();
            let error = weights.load("a", &Device::Cpu).unwrap_err();
            assert!(format!("{error:#}").contains("header changed"));
            assert_eq!(weights.cache_stats().resident_bytes, 0);
        }
    }
}
