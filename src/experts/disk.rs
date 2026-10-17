//! Disk tier: bounded aligned native-range reads for disk-owned experts.
use crate::experts::catalogue::{ExpertCatalogue, ExpertGeometry, ExpertRange};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::ptr::NonNull;

/// Alignment of a ring slot; one slot holds one expert's gate-up then down bytes.
const RING_ALIGN: usize = 4096;

struct AlignedSlot {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedSlot {
    fn new(bytes: usize) -> Result<Self> {
        let layout = Layout::from_size_align(bytes, RING_ALIGN)
            .with_context(|| format!("invalid ring slot layout: {bytes} B"))?;
        // SAFETY: the layout has a nonzero size and a power-of-two alignment.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .with_context(|| format!("allocate {bytes} B ring slot"))?;
        Ok(Self { ptr, layout })
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
    geometry: ExpertGeometry,
    per_expert: usize,
    gate_up_bytes: u64,
    ring: Vec<AlignedSlot>,
    next: usize,
}

impl DiskReader {
    pub fn open(catalogue: &ExpertCatalogue, model_dir: &Path, ring_slots: u32) -> Result<Self> {
        ensure!(ring_slots > 0, "zero disk ring slots");
        let geometry = catalogue.geometry();
        let slot_bytes = usize::try_from(geometry.slot_bytes()?)?;
        let gate_up_bytes = geometry.gate_up_bytes()?;
        let per_expert = slot_bytes / std::mem::size_of::<bf16>();
        let mut shards = BTreeMap::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                let range = catalogue.range(layer, expert)?;
                for native in [&range.gate_up, &range.down] {
                    let name = native.shard.as_ref();
                    if !shards.contains_key(name) {
                        let file = File::open(model_dir.join(name))
                            .with_context(|| format!("open disk shard {name}"))?;
                        shards.insert(name.to_owned(), file);
                    }
                    let file = &shards[name];
                    let length = file
                        .metadata()
                        .with_context(|| format!("stat disk shard {name}"))?
                        .len();
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
            geometry,
            per_expert,
            gate_up_bytes,
            ring,
            next: 0,
        })
    }

    /// Reads one expert's native gate-up and down ranges into the next ring slot.
    pub fn read_expert(
        &mut self,
        layer: u32,
        expert: u32,
        catalogue: &ExpertCatalogue,
    ) -> Result<&[bf16]> {
        let range: &ExpertRange = catalogue.range(layer, expert)?;
        let slot = (self.next + 1) % self.ring.len();
        self.next = slot;
        let bytes = self.ring[slot].layout.size();
        let ptr = self.ring[slot].ptr.as_ptr();
        // SAFETY: the slot owns `bytes` readable and writable bytes with the ring alignment.
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
        let start = self.gate_up_bytes as usize;
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
        ensure!(
            range.gate_up.len + range.down.len == bytes as u64,
            "expert {expert} of layer {layer} does not fill one ring slot"
        );
        // SAFETY: the slot is aligned and holds `per_expert` initialized bf16 values.
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const bf16, self.per_expert) })
    }

    /// Ring slot count.
    pub fn ring_slots(&self) -> u32 {
        u32::try_from(self.ring.len()).expect("ring count fits u32")
    }

    /// Next ring cursor, for diagnostics and tests.
    pub fn cursor(&self) -> usize {
        self.next
    }

    pub fn geometry(&self) -> ExpertGeometry {
        self.geometry
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

    fn entry(role: &str, shape: Vec<usize>, start: u64, len: u64) -> TensorEntry {
        TensorEntry {
            dtype: safetensors::Dtype::BF16,
            shape,
            shard: "shard-0.safetensors".into(),
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

    fn manifest() -> TensorManifest {
        let gate = 4 * 8 * 8 * 2u64;
        let down = 4 * 8 * 4 * 2u64;
        let mut entries = BTreeMap::new();
        for layer in 0..2u32 {
            let base = u64::from(layer) * (gate + down);
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                entry("mlp.experts.gate_up_proj", vec![4, 8, 8], base, gate),
            );
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                entry("mlp.experts.down_proj", vec![4, 8, 4], base + gate, down),
            );
        }
        TensorManifest {
            entries,
            resident_bytes: 0,
        }
    }

    fn pattern(bytes: usize) -> Vec<u8> {
        (0..bytes)
            .map(|i| ((i * 37 + i / 251) % 256) as u8)
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

    fn staged() -> Result<(TempDir, ExpertCatalogue, u64)> {
        let dir = temp_dir("ranges")?;
        let bytes = pattern(1536);
        std::fs::write(dir.path().join("shard-0.safetensors"), &bytes)?;
        let catalogue = ExpertCatalogue::from_manifest(&manifest(), geometry())?;
        Ok((dir, catalogue, 1536))
    }

    #[test]
    fn reads_native_ranges_at_the_catalogue_offsets() -> Result<()> {
        let (dir, catalogue, bytes) = staged()?;
        let mut reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        let pattern = pattern(bytes as usize);
        for (layer, expert) in [(0u32, 0u32), (0, 3), (1, 0), (1, 3)] {
            let values = reader.read_expert(layer, expert, &catalogue)?;
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
        }
        assert_eq!(reader.ring_slots(), 2);
        Ok(())
    }

    #[test]
    fn a_truncated_shard_names_the_range() -> Result<()> {
        let dir = temp_dir("truncated")?;
        std::fs::write(dir.path().join("shard-0.safetensors"), vec![0u8; 1000])?;
        let catalogue = ExpertCatalogue::from_manifest(&manifest(), geometry())?;
        let Err(error) = DiskReader::open(&catalogue, dir.path(), 1) else {
            panic!("open succeeded on a truncated shard");
        };
        let error = error.to_string();
        assert!(error.contains("exceeds 1000 bytes"), "{error}");
        Ok(())
    }

    #[test]
    fn the_ring_rotates_over_bounded_slots() -> Result<()> {
        let (dir, catalogue, _) = staged()?;
        let mut reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        assert_eq!(reader.cursor(), 0);
        reader.read_expert(0, 0, &catalogue)?;
        assert_eq!(reader.cursor(), 1);
        reader.read_expert(0, 1, &catalogue)?;
        assert_eq!(reader.cursor(), 0);
        reader.read_expert(1, 2, &catalogue)?;
        assert_eq!(reader.cursor(), 1);
        assert_eq!(reader.ring_slots(), 2);
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
        let mut reader = DiskReader::open(&catalogue, &dir, 2)?;
        let gate_up_elements = 2 * geometry.intermediate as usize * geometry.hidden as usize;
        let down_elements = geometry.hidden as usize * geometry.intermediate as usize;
        for (layer, expert) in [(0u32, 0u32), (39, 255)] {
            let values = reader.read_expert(layer, expert, &catalogue)?;
            for (name, offset) in [
                (
                    format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                    0usize,
                ),
                (
                    format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                    gate_up_elements,
                ),
            ] {
                let shard = catalogue.range(layer, expert)?.gate_up.shard.clone();
                let file = std::fs::File::open(dir.join(shard.as_ref()))?;
                let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&map)?;
                let data = tensors.tensor(&name)?.data();
                let elements = if offset == 0 {
                    gate_up_elements
                } else {
                    down_elements
                };
                let start = expert as usize * elements * 2;
                for index in 0..elements {
                    let value = values[offset + index].to_bits();
                    let expected =
                        u16::from_le_bytes([data[start + index * 2], data[start + index * 2 + 1]]);
                    assert_eq!(value, expected, "{name} element {index}");
                }
            }
        }
        Ok(())
    }
}
