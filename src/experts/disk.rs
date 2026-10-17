//! Disk tier: bounded aligned native-range reads with per-slot completion leases.
use crate::experts::catalogue::{ExpertCatalogue, ExpertRange};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::ptr::NonNull;

/// Alignment of a ring slot; one slot holds one expert's gate-up then down bytes.
const RING_ALIGN: usize = 4096;

struct AlignedSlot {
    ptr: NonNull<u8>,
    layout: Layout,
    leased: Cell<bool>,
}

impl AlignedSlot {
    fn new(bytes: usize) -> Result<Self> {
        let layout = Layout::from_size_align(bytes, RING_ALIGN)
            .with_context(|| format!("invalid ring slot layout: {bytes} B"))?;
        // SAFETY: the layout has a nonzero size and a power-of-two alignment.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .with_context(|| format!("allocate {bytes} B ring slot"))?;
        Ok(Self {
            ptr,
            layout,
            leased: Cell::new(false),
        })
    }
}

impl Drop for AlignedSlot {
    fn drop(&mut self) {
        // SAFETY: the pointer came from alloc_zeroed with this layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Read-only disk tier over the native checkpoint shards with a bounded aligned ring.
pub struct DiskReader {
    shards: BTreeMap<String, File>,
    per_expert: usize,
    gate_up_bytes: u64,
    ring: Vec<AlignedSlot>,
    next: Cell<usize>,
}

// SAFETY: the ring holds a private allocation owned by this reader; moving the reader to one
// worker thread transfers that ownership.
unsafe impl Send for DiskReader {}

impl DiskReader {
    pub fn open(catalogue: &ExpertCatalogue, model_dir: &Path, ring_slots: u32) -> Result<Self> {
        ensure!(ring_slots > 0, "zero disk ring slots");
        let geometry = catalogue.geometry();
        let slot_bytes = usize::try_from(geometry.slot_bytes()?)?;
        let gate_up_bytes = geometry.gate_up_bytes()?;
        let per_expert = slot_bytes / std::mem::size_of::<bf16>();
        let mut names = BTreeSet::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                let range = catalogue.range(layer, expert)?;
                names.insert(range.gate_up.shard.clone());
                names.insert(range.down.shard.clone());
            }
        }
        let mut shards = BTreeMap::new();
        let mut lengths = BTreeMap::new();
        for name in names {
            let file = File::open(model_dir.join(name.as_ref()))
                .with_context(|| format!("open disk shard {name}"))?;
            let length = file
                .metadata()
                .with_context(|| format!("stat disk shard {name}"))?
                .len();
            shards.insert(name.to_string(), file);
            lengths.insert(name.to_string(), length);
        }
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                let range = catalogue.range(layer, expert)?;
                for native in [&range.gate_up, &range.down] {
                    let name = native.shard.as_ref();
                    let length = lengths[name];
                    let end = native
                        .offset
                        .checked_add(native.len)
                        .context("disk range end overflow")?;
                    ensure!(
                        end <= length,
                        "disk range {}..{end} of shard {name} exceeds {length} bytes",
                        native.offset
                    );
                }
            }
        }
        let ring = (0..ring_slots)
            .map(|_| AlignedSlot::new(slot_bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shards,
            per_expert,
            gate_up_bytes,
            ring,
            next: Cell::new(0),
        })
    }

    /// Reads one expert's native ranges into a free ring slot and leases it; release with
    /// `release` after the consumer is done.
    pub fn read_expert(&self, layer: u32, expert: u32, catalogue: &ExpertCatalogue) -> Result<u32> {
        let range: &ExpertRange = catalogue.range(layer, expert)?;
        let slots = self.ring.len();
        let mut chosen = None;
        for step in 0..slots {
            let index = (self.next.get() + step) % slots;
            if !self.ring[index].leased.get() {
                chosen = Some(index);
                break;
            }
        }
        let slot = chosen.with_context(|| format!("all {slots} disk ring slots are leased"))?;
        self.next.set((slot + 1) % slots);
        let bytes = self.ring[slot].layout.size();
        let ptr = self.ring[slot].ptr.as_ptr();
        // SAFETY: the slot owns `bytes` writable bytes with the ring alignment.
        let buffer = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
        let gate = self
            .shards
            .get(range.gate_up.shard.as_ref())
            .with_context(|| format!("disk shard {} not open", range.gate_up.shard))?;
        gate.read_exact_at(
            &mut buffer[..range.gate_up.len as usize],
            range.gate_up.offset,
        )
        .with_context(|| {
            format!(
                "read gate-up {}..{} of shard {}",
                range.gate_up.offset,
                range.gate_up.offset + range.gate_up.len,
                range.gate_up.shard
            )
        })?;
        let start = usize::try_from(self.gate_up_bytes)?;
        let down = self
            .shards
            .get(range.down.shard.as_ref())
            .with_context(|| format!("disk shard {} not open", range.down.shard))?;
        down.read_exact_at(
            &mut buffer[start..start + range.down.len as usize],
            range.down.offset,
        )
        .with_context(|| {
            format!(
                "read down {}..{} of shard {}",
                range.down.offset,
                range.down.offset + range.down.len,
                range.down.shard
            )
        })?;
        self.ring[slot].leased.set(true);
        u32::try_from(slot).context("disk ring slot exceeds u32")
    }

    /// Values of a leased ring slot.
    pub fn values(&self, slot: u32) -> Result<&[bf16]> {
        let index = usize::try_from(slot).context("disk ring slot exceeds usize")?;
        ensure!(
            index < self.ring.len() && self.ring[index].leased.get(),
            "disk ring slot {slot} is not leased"
        );
        let ptr = self.ring[index].ptr.as_ptr();
        // SAFETY: the slot is leased and holds `per_expert` initialized bf16 values.
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const bf16, self.per_expert) })
    }

    /// Releases a leased slot for reuse.
    pub fn release(&self, slot: u32) -> Result<()> {
        let index = usize::try_from(slot).context("disk ring slot exceeds usize")?;
        ensure!(
            index < self.ring.len() && self.ring[index].leased.get(),
            "disk ring slot {slot} is not leased"
        );
        self.ring[index].leased.set(false);
        Ok(())
    }
}

/// Bounded layer-ahead disk prefetch; demand entries jump ahead of prefetches.
pub struct Prefetch {
    depth: u32,
    queue: Vec<(u32, u32)>,
    loaded: Vec<(u32, u32, u32)>,
}

impl Prefetch {
    pub fn new(depth: u32) -> Result<Self> {
        ensure!(depth > 0, "zero prefetch depth");
        Ok(Self {
            depth,
            queue: Vec::with_capacity(depth as usize),
            loaded: Vec::with_capacity(depth as usize),
        })
    }

    /// Entries holding a ring slot's worth of capacity: queued and loaded.
    pub fn entries(&self) -> u32 {
        u32::try_from(self.queue.len() + self.loaded.len()).expect("prefetch entry count fits u32")
    }

    /// Queues one read; a demand entry moves to the front and repeats are no-ops.
    pub fn request(&mut self, layer: u32, expert: u32, demand: bool) -> Result<()> {
        if self
            .loaded
            .iter()
            .any(|&(l, e, _)| l == layer && e == expert)
        {
            return Ok(());
        }
        if let Some(index) = self
            .queue
            .iter()
            .position(|&(l, e)| l == layer && e == expert)
        {
            if demand && index > 0 {
                let entry = self.queue.remove(index);
                self.queue.insert(0, entry);
            }
            return Ok(());
        }
        ensure!(
            self.entries() < self.depth,
            "prefetch queue full at depth {}",
            self.depth
        );
        if demand {
            self.queue.insert(0, (layer, expert));
        } else {
            self.queue.push((layer, expert));
        }
        Ok(())
    }

    /// Front request for the reader; it stays queued until `complete`.
    pub fn front(&self) -> Option<(u32, u32)> {
        self.queue.first().copied()
    }

    /// Records the ring slot of the front request.
    pub fn complete(&mut self, layer: u32, expert: u32, slot: u32) -> Result<()> {
        ensure!(
            self.front() == Some((layer, expert)),
            "completed request ({layer}, {expert}) is not the front"
        );
        self.queue.remove(0);
        self.loaded.push((layer, expert, slot));
        Ok(())
    }

    /// Drops a queued request that became resident elsewhere.
    pub fn cancel(&mut self, layer: u32, expert: u32) -> bool {
        self.queue
            .iter()
            .position(|&(l, e)| l == layer && e == expert)
            .map(|index| self.queue.remove(index))
            .is_some()
    }

    /// Takes a loaded request's ring slot.
    pub fn take(&mut self, layer: u32, expert: u32) -> Option<u32> {
        self.loaded
            .iter()
            .position(|&(l, e, _)| l == layer && e == expert)
            .map(|index| self.loaded.remove(index).2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::experts::catalogue::ExpertGeometry;
    use std::collections::BTreeMap;

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
            layers: 2,
            experts: 4,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn entry(role: &str, shape: Vec<usize>, shard: &str, start: u64, len: u64) -> TensorEntry {
        TensorEntry {
            dtype: safetensors::Dtype::BF16,
            shape,
            shard: shard.into(),
            bytes: len as usize,
            load: true,
            data_start: 0,
            data_offsets: [start, start + len],
            scope: TensorScope::Text,
            layer: None,
            role: role.into(),
            scale_name: None,
        }
    }

    fn manifest(gate_shard: &str, down_shard: &str) -> TensorManifest {
        let gate = 4 * 8 * 8 * 2u64;
        let down = 4 * 8 * 4 * 2u64;
        let mut entries = BTreeMap::new();
        for layer in 0..2u32 {
            let base = u64::from(layer) * (gate + down);
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                entry(
                    "mlp.experts.gate_up_proj",
                    vec![4, 8, 8],
                    gate_shard,
                    base,
                    gate,
                ),
            );
            let down_start = if gate_shard == down_shard {
                base + gate
            } else {
                base
            };
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                entry(
                    "mlp.experts.down_proj",
                    vec![4, 8, 4],
                    down_shard,
                    down_start,
                    down,
                ),
            );
        }
        TensorManifest {
            entries,
            resident_bytes: 0,
        }
    }

    fn pattern(bytes: usize, salt: usize) -> Vec<u8> {
        (0..bytes)
            .map(|i| ((i * 37 + i / 251 + salt) % 256) as u8)
            .collect()
    }

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(tag: &str) -> Result<TempDir> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "flyingfish-disk-{}-{}-{}",
            std::process::id(),
            tag,
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)?;
        Ok(TempDir(dir))
    }

    fn staged(gate_shard: &str, down_shard: &str) -> Result<(TempDir, ExpertCatalogue)> {
        let dir = temp_dir("ranges")?;
        std::fs::write(dir.path().join(gate_shard), pattern(1536, 0))?;
        if down_shard != gate_shard {
            std::fs::write(dir.path().join(down_shard), pattern(1536, 91))?;
        }
        let catalogue =
            ExpertCatalogue::from_manifest(&manifest(gate_shard, down_shard), geometry())?;
        Ok((dir, catalogue))
    }

    #[test]
    fn reads_native_ranges_at_the_catalogue_offsets() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let pattern = pattern(1536, 0);
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        for (layer, expert) in [(0u32, 0u32), (0, 3), (1, 0), (1, 3)] {
            let slot = reader.read_expert(layer, expert, &catalogue)?;
            let values = reader.values(slot)?;
            assert_eq!(values.len(), 96);
            let layer_base = layer as usize * 768;
            let gate = layer_base + expert as usize * 128;
            let down = layer_base + 512 + expert as usize * 64;
            for (index, value) in values.iter().enumerate() {
                let byte = if index < 64 {
                    gate + index * 2
                } else {
                    down + (index - 64) * 2
                };
                assert_eq!(
                    value.to_bits(),
                    u16::from_le_bytes([pattern[byte], pattern[byte + 1]]),
                    "layer {layer} expert {expert} element {index}"
                );
            }
            reader.release(slot)?;
        }
        Ok(())
    }

    #[test]
    fn gate_up_and_down_may_live_in_different_shards() -> Result<()> {
        let (dir, catalogue) = staged("gate.safetensors", "down.safetensors")?;
        let gate_pattern = pattern(1536, 0);
        let down_pattern = pattern(1536, 91);
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let slot = reader.read_expert(0, 2, &catalogue)?;
        let values = reader.values(slot)?;
        for (index, value) in values.iter().enumerate() {
            let (pattern, byte) = if index < 64 {
                (&gate_pattern, 2 * 128 + index * 2)
            } else {
                (&down_pattern, 2 * 64 + (index - 64) * 2)
            };
            assert_eq!(
                value.to_bits(),
                u16::from_le_bytes([pattern[byte], pattern[byte + 1]]),
                "element {index}"
            );
        }
        reader.release(slot)?;
        Ok(())
    }

    #[test]
    fn a_truncated_shard_names_the_range() -> Result<()> {
        let dir = temp_dir("truncated")?;
        std::fs::write(dir.path().join("shard-0.safetensors"), vec![0u8; 1000])?;
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest("shard-0.safetensors", "shard-0.safetensors"),
            geometry(),
        )?;
        let Err(error) = DiskReader::open(&catalogue, dir.path(), 1) else {
            panic!("open succeeded on a truncated shard");
        };
        let error = error.to_string();
        assert!(error.contains("exceeds 1000 bytes"), "{error}");
        Ok(())
    }

    #[test]
    fn ring_slots_are_leased_until_released() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        let first = reader.read_expert(0, 0, &catalogue)?;
        let second = reader.read_expert(0, 1, &catalogue)?;
        assert_ne!(first, second);
        let Err(error) = reader.read_expert(0, 2, &catalogue) else {
            panic!("a third read succeeded with every slot leased");
        };
        let error = error.to_string();
        assert!(
            error.contains("all 2 disk ring slots are leased"),
            "{error}"
        );
        reader.release(first)?;
        assert!(reader.values(first).is_err());
        let third = reader.read_expert(0, 2, &catalogue)?;
        assert_eq!(third, first);
        assert!(reader.release(third).is_ok());
        assert!(reader.release(third).is_err());
        Ok(())
    }

    #[test]
    fn a_slot_can_be_filled_while_another_is_read() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        let first = reader.read_expert(0, 0, &catalogue)?;
        let values = reader.values(first)?;
        let second = reader.read_expert(0, 1, &catalogue)?;
        assert_ne!(first, second);
        let second_values = reader.values(second)?;
        assert_eq!(values.len(), 96);
        assert_eq!(second_values.len(), 96);
        assert_ne!(values[0].to_bits(), second_values[0].to_bits());
        reader.release(first)?;
        reader.release(second)?;
        assert!(reader.values(first).is_err());
        let single = DiskReader::open(&catalogue, dir.path(), 1)?;
        let held = single.read_expert(0, 0, &catalogue)?;
        let error = single
            .read_expert(0, 1, &catalogue)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("all 1 disk ring slots are leased"),
            "{error}"
        );
        single.release(held)?;
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only real checkpoint; set FF_MODEL"]
    fn reads_the_published_checkpoint_ranges() -> Result<()> {
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let model = crate::model_spec::ModelSpec::read(&dir)?;
        let geometry = ExpertGeometry::from_model(&model)?;
        let mapped = crate::checkpoint::MappedCheckpoint::open(&dir)?;
        let manifest = mapped.manifest(&model, crate::checkpoint::LoadPolicy::TextOnly)?;
        let catalogue = ExpertCatalogue::from_manifest(&manifest, geometry)?;
        let reader = DiskReader::open(&catalogue, &dir, 2)?;
        let gate_up_elements = 2 * geometry.intermediate as usize * geometry.hidden as usize;
        let down_elements = geometry.hidden as usize * geometry.intermediate as usize;
        for (layer, expert) in [(0u32, 0u32), (39, 255)] {
            let slot = reader.read_expert(layer, expert, &catalogue)?;
            let values = reader.values(slot)?.to_vec();
            for (name, range, offset, elements) in [
                (
                    format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                    catalogue.range(layer, expert)?.gate_up.clone(),
                    0usize,
                    gate_up_elements,
                ),
                (
                    format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                    catalogue.range(layer, expert)?.down.clone(),
                    gate_up_elements,
                    down_elements,
                ),
            ] {
                let file = std::fs::File::open(dir.join(range.shard.as_ref()))?;
                let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&map)?;
                let data = tensors.tensor(&name)?.data();
                let start = expert as usize * elements * 2;
                for index in 0..elements {
                    let value = values[offset + index].to_bits();
                    let expected =
                        u16::from_le_bytes([data[start + index * 2], data[start + index * 2 + 1]]);
                    assert_eq!(value, expected, "{name} element {index}");
                }
            }
            reader.release(slot)?;
        }
        Ok(())
    }

    #[test]
    fn prefetch_serves_demand_first_within_its_depth() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let reader = DiskReader::open(&catalogue, dir.path(), 4)?;
        let mut prefetch = Prefetch::new(2)?;
        prefetch.request(0, 0, false)?;
        prefetch.request(0, 1, false)?;
        let error = prefetch.request(0, 2, false).unwrap_err().to_string();
        assert!(error.contains("prefetch queue full at depth 2"), "{error}");
        prefetch.request(0, 1, true)?;
        assert_eq!(prefetch.front(), Some((0, 1)));
        prefetch.request(0, 1, true)?;
        assert_eq!(prefetch.entries(), 2);
        let error = prefetch.complete(0, 0, 0).unwrap_err().to_string();
        assert!(error.contains("is not the front"), "{error}");
        let (layer, expert) = prefetch.front().expect("front request");
        let slot = reader.read_expert(layer, expert, &catalogue)?;
        prefetch.complete(layer, expert, slot)?;
        assert_eq!(prefetch.front(), Some((0, 0)));
        assert_eq!(prefetch.take(0, 1), Some(slot));
        assert_eq!(prefetch.take(0, 1), None);
        reader.release(slot)?;
        assert_eq!(prefetch.entries(), 1);
        assert!(prefetch.cancel(0, 0));
        assert!(!prefetch.cancel(0, 0));
        assert_eq!(prefetch.entries(), 0);
        assert!(prefetch.take(0, 0).is_none());
        assert!(Prefetch::new(0).is_err());
        Ok(())
    }
}
