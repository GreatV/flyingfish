//! Disk tier: bounded aligned native-range reads with per-slot completion leases.
use crate::experts::catalogue::{ExpertCatalogue, ExpertLayout, ExpertRange, NativeRange};
use anyhow::{Context, Result, bail, ensure};
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
    state: Cell<RingState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RingState {
    Free,
    Filled,
    Guarded,
    Copied { event: u64 },
}

/// Lease over one filled ring slot; dropping it returns the slot to the ring unless `copied`
/// recorded a device copy event.
pub struct SlotLease<'a> {
    reader: &'a DiskReader,
    slot: usize,
    release: bool,
}

impl SlotLease<'_> {
    pub fn values(&self) -> &[bf16] {
        let ring = &self.reader.ring[self.slot];
        // SAFETY: the slot holds `per_expert` initialized bf16 values for the guard's lifetime.
        unsafe {
            std::slice::from_raw_parts(ring.ptr.as_ptr() as *const bf16, self.reader.per_expert)
        }
    }

    /// Marks the slot's data copied to the device; the slot is reusable only through `reclaim`
    /// with this event.
    pub fn copied(mut self, event: u64) -> Result<()> {
        ensure!(
            self.reader.ring[self.slot].state.get() == RingState::Guarded,
            "disk ring slot {} is not guarded",
            self.slot
        );
        self.reader.ring[self.slot]
            .state
            .set(RingState::Copied { event });
        self.release = false;
        Ok(())
    }
}

impl std::ops::Deref for SlotLease<'_> {
    type Target = [bf16];

    fn deref(&self) -> &[bf16] {
        self.values()
    }
}

impl Drop for SlotLease<'_> {
    fn drop(&mut self) {
        if self.release {
            self.reader.ring[self.slot].state.set(RingState::Free);
        }
    }
}

impl AlignedSlot {
    fn new(bytes: usize) -> Result<Self> {
        ensure!(bytes > 0, "zero disk ring slot bytes");
        let layout = Layout::from_size_align(bytes, RING_ALIGN)
            .with_context(|| format!("invalid ring slot layout: {bytes} B"))?;
        // SAFETY: the layout has a nonzero size and a power-of-two alignment.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .with_context(|| format!("allocate {bytes} B ring slot"))?;
        Ok(Self {
            ptr,
            layout,
            state: Cell::new(RingState::Free),
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
    ring: Vec<AlignedSlot>,
    next: Cell<usize>,
}

// SAFETY: the ring holds a private allocation owned by this reader; moving the reader to one
// worker thread transfers that ownership.
unsafe impl Send for DiskReader {}

/// Visits one expert's native ranges in the single documented in-slot order.
fn visit_slices(
    range: &ExpertRange,
    visit: &mut impl FnMut(&NativeRange) -> Result<()>,
) -> Result<()> {
    match range {
        ExpertRange::Bf16 { gate_up, down } => {
            visit(gate_up)?;
            visit(down)?;
        }
        ExpertRange::Fp8 {
            gate,
            up,
            down,
            gate_scale,
            up_scale,
            down_scale,
        } => {
            for native in [gate, up, down, gate_scale, up_scale, down_scale] {
                visit(native)?;
            }
        }
    }
    Ok(())
}

impl DiskReader {
    pub fn open(catalogue: &ExpertCatalogue, model_dir: &Path, ring_slots: u32) -> Result<Self> {
        ensure!(ring_slots > 0, "zero disk ring slots");
        let geometry = catalogue.geometry();
        let slot_bytes = usize::try_from(catalogue.slot_bytes())?;
        let per_expert = slot_bytes / std::mem::size_of::<bf16>();
        let mut names = BTreeSet::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                visit_slices(catalogue.range(layer, expert)?, &mut |native| {
                    names.insert(native.shard.clone());
                    Ok(())
                })?;
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
                visit_slices(catalogue.range(layer, expert)?, &mut |native| {
                    let name = native.shard.as_ref();
                    let length = lengths
                        .get(name)
                        .copied()
                        .with_context(|| format!("disk shard {name} not open"))?;
                    let end = native
                        .offset
                        .checked_add(native.len)
                        .context("disk range end overflow")?;
                    ensure!(
                        end <= length,
                        "disk range {}..{end} of shard {name} exceeds {length} bytes",
                        native.offset
                    );
                    Ok(())
                })?;
            }
        }
        let ring = (0..ring_slots)
            .map(|_| AlignedSlot::new(slot_bytes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shards,
            per_expert,
            ring,
            next: Cell::new(0),
        })
    }

    /// Reads one expert's native ranges into a free ring slot in the layout's slot order.
    pub fn read_expert(&self, layer: u32, expert: u32, catalogue: &ExpertCatalogue) -> Result<u32> {
        let range: &ExpertRange = catalogue.range(layer, expert)?;
        let layout = catalogue.layout();
        let slots = self.ring.len();
        let mut chosen = None;
        for step in 0..slots {
            let index = (self.next.get() + step) % slots;
            if self.ring[index].state.get() == RingState::Free {
                chosen = Some(index);
                break;
            }
        }
        let slot = chosen.with_context(|| format!("all {slots} disk ring slots are in use"))?;
        self.next.set((slot + 1) % slots);
        let bytes = self.ring[slot].layout.size();
        let ptr = self.ring[slot].ptr.as_ptr();
        // SAFETY: the slot owns `bytes` writable bytes with the ring alignment.
        let buffer: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
        let mut read = |offset: u64, native: &NativeRange| -> Result<()> {
            let start = usize::try_from(offset).context("expert slice offset exceeds usize")?;
            let len = usize::try_from(native.len).context("expert slice length exceeds usize")?;
            let end = start
                .checked_add(len)
                .context("expert slice end overflow")?;
            ensure!(
                end <= buffer.len(),
                "expert slice {start}..{end} exceeds the {}-byte slot",
                buffer.len()
            );
            let shard = self
                .shards
                .get(native.shard.as_ref())
                .with_context(|| format!("disk shard {} not open", native.shard))?;
            shard
                .read_exact_at(&mut buffer[start..end], native.offset)
                .with_context(|| {
                    format!(
                        "read {}..{} of shard {}",
                        native.offset,
                        native.offset + native.len,
                        native.shard
                    )
                })
        };
        match (layout, range) {
            (
                ExpertLayout::Bf16Fused { gate_up_bytes, .. },
                ExpertRange::Bf16 { gate_up, down },
            ) => {
                read(0, gate_up)?;
                read(*gate_up_bytes, down)?;
            }
            (
                ExpertLayout::Fp8W8a16(slot),
                ExpertRange::Fp8 {
                    gate,
                    up,
                    down,
                    gate_scale,
                    up_scale,
                    down_scale,
                },
            ) => {
                read(slot.gate.start, gate)?;
                read(slot.up.start, up)?;
                read(slot.down.start, down)?;
                read(slot.gate_scale.start, gate_scale)?;
                read(slot.up_scale.start, up_scale)?;
                read(slot.down_scale.start, down_scale)?;
            }
            _ => bail!("expert range does not match the catalogue layout"),
        }
        self.ring[slot].state.set(RingState::Filled);
        u32::try_from(slot).context("disk ring slot exceeds u32")
    }

    /// Lease over a filled ring slot; the guard holds the values until it drops or records the
    /// device copy event.
    pub fn values(&self, slot: u32) -> Result<SlotLease<'_>> {
        let index = usize::try_from(slot).context("disk ring slot exceeds usize")?;
        ensure!(
            index < self.ring.len(),
            "disk ring slot {slot} outside {}",
            self.ring.len()
        );
        ensure!(
            self.ring[index].state.get() == RingState::Filled,
            "disk ring slot {slot} is not filled"
        );
        self.ring[index].state.set(RingState::Guarded);
        Ok(SlotLease {
            reader: self,
            slot: index,
            release: true,
        })
    }

    /// Discards unconsumed prefetched bytes.
    pub fn discard(&self, slot: u32) -> Result<()> {
        let index = usize::try_from(slot).context("disk ring slot exceeds usize")?;
        ensure!(
            index < self.ring.len() && self.ring[index].state.get() == RingState::Filled,
            "disk ring slot {slot} is not filled"
        );
        self.ring[index].state.set(RingState::Free);
        Ok(())
    }

    /// Returns a copied slot to the ring after its device copy event completed.
    pub fn reclaim(&self, slot: u32, event: u64) -> Result<()> {
        let index = usize::try_from(slot).context("disk ring slot exceeds usize")?;
        ensure!(
            index < self.ring.len()
                && self.ring[index].state.get() == (RingState::Copied { event }),
            "disk ring slot {slot} has no copy with event {event}"
        );
        self.ring[index].state.set(RingState::Free);
        Ok(())
    }

    pub fn ring_slots(&self) -> u32 {
        u32::try_from(self.ring.len()).expect("ring slot count fits u32")
    }
}

struct Work {
    layer: u32,
    hidden: Vec<bf16>,
    routes: Vec<crate::experts::runtime::Route>,
    output: Vec<bf16>,
}

struct Completed {
    work: Work,
    error: Option<String>,
}

struct Worker {
    input: Option<std::sync::mpsc::SyncSender<Work>>,
    output: std::sync::mpsc::Receiver<Completed>,
    handle: Option<std::thread::JoinHandle<()>>,
    work: Option<Work>,
}

/// Bounded native disk reads and CPU contributions, with one persistent reader per worker.
pub struct DiskExperts {
    geometry: crate::experts::catalogue::ExpertGeometry,
    max_rows: usize,
    max_routes: usize,
    running: bool,
    workers: Vec<Worker>,
}

impl DiskExperts {
    pub fn new(
        catalogue: std::sync::Arc<ExpertCatalogue>,
        directory: &Path,
        workers: u32,
        max_rows: u32,
    ) -> Result<Self> {
        let geometry = catalogue.geometry();
        ensure!(
            workers > 0 && workers <= 256 && max_rows > 0,
            "disk expert bounds workers={workers} rows={max_rows}"
        );
        let shape = crate::cpu::expert_ffn::Shape::new(
            geometry.hidden as usize,
            geometry.intermediate as usize,
        )?;
        let max_routes = (max_rows as usize)
            .checked_mul(geometry.top_k as usize)
            .context("disk route capacity overflow")?;
        let elements = max_routes
            .checked_mul(shape.hidden)
            .context("disk output capacity overflow")?;
        let mut pool = Self {
            geometry,
            max_rows: max_rows as usize,
            max_routes,
            running: false,
            workers: Vec::with_capacity(workers as usize),
        };
        for index in 0..workers {
            let reader = DiskReader::open(&catalogue, directory, 1)?;
            let mut scratch = crate::cpu::expert_ffn::Scratch::new(&shape);
            let mut vector = vec![bf16::ZERO; shape.hidden];
            let native = crate::cpu::expert_ffn::ExpertWeights::from_slot(
                shape,
                0..usize::try_from(catalogue.slot_bytes() / 2)?,
                catalogue.layout(),
            )?;
            let isa = crate::cpu::expert_ffn::Isa::detect();
            let compute = if catalogue.layout().fp8_slot().is_some() {
                disk_contributions::<true>
            } else {
                disk_contributions::<false>
            };
            let source = catalogue.clone();
            let (input, requests) = std::sync::mpsc::sync_channel::<Work>(1);
            let (results, output) = std::sync::mpsc::sync_channel(1);
            let handle = std::thread::Builder::new()
                .name(format!("expert-disk-{index}"))
                .spawn(move || {
                    while let Ok(mut work) = requests.recv() {
                        let result = compute(
                            &reader,
                            &source,
                            &mut work,
                            &native,
                            &mut scratch,
                            &mut vector,
                            isa,
                        );
                        let error = result.err().map(|error| format!("{error:#}"));
                        if results.send(Completed { work, error }).is_err() {
                            break;
                        }
                    }
                })
                .context("spawn native disk expert worker")?;
            pool.workers.push(Worker {
                input: Some(input),
                output,
                handle: Some(handle),
                work: Some(Work {
                    layer: 0,
                    hidden: Vec::with_capacity(max_rows as usize * shape.hidden),
                    routes: Vec::with_capacity(max_routes),
                    output: vec![bf16::ZERO; elements],
                }),
            });
        }
        Ok(pool)
    }

    pub fn start(
        &mut self,
        layer: u32,
        hidden: &[bf16],
        rows: usize,
        routes: &[crate::experts::runtime::Route],
    ) -> Result<()> {
        ensure!(!self.running, "disk contribution job is already running");
        ensure!(
            layer < self.geometry.layers
                && rows > 0
                && rows <= self.max_rows
                && hidden.len() == rows * self.geometry.hidden as usize
                && routes.len() <= self.max_routes,
            "disk contribution bounds layer={layer} rows={rows} hidden={} routes={}",
            hidden.len(),
            routes.len()
        );
        for (index, route) in routes.iter().enumerate() {
            ensure!(
                (route.row as usize) < rows
                    && route.expert < self.geometry.experts
                    && (route.choice as usize) < self.max_routes
                    && route.weight.to_f32().is_finite(),
                "disk route outside job bounds: {route:?}"
            );
            ensure!(
                !routes[..index]
                    .iter()
                    .any(|previous| previous.choice == route.choice),
                "disk choice={} has duplicate routes",
                route.choice
            );
        }
        let workers = self.workers.len();
        for (index, worker) in self.workers.iter_mut().enumerate() {
            let mut work = worker
                .work
                .take()
                .context("disk worker has no reusable job")?;
            work.layer = layer;
            work.hidden.clear();
            work.hidden.extend_from_slice(hidden);
            work.routes.clear();
            work.routes.extend(
                routes
                    .iter()
                    .filter(|r| r.expert as usize % workers == index),
            );
            worker
                .input
                .as_ref()
                .context("disk worker input closed")?
                .send(work)
                .map_err(|_| anyhow::anyhow!("disk expert worker {index} stopped"))?;
        }
        self.running = true;
        Ok(())
    }

    pub fn finish(&mut self, output: &mut [bf16]) -> Result<()> {
        ensure!(
            self.running,
            "disk contribution finish without a running job"
        );
        let width = self.geometry.hidden as usize;
        let mut failure = None;
        for (index, worker) in self.workers.iter_mut().enumerate() {
            let completed = worker
                .output
                .recv()
                .with_context(|| format!("receive disk worker {index}"))?;
            for route in &completed.work.routes {
                let first = route.choice as usize * width;
                ensure!(
                    first + width <= output.len(),
                    "disk contribution choice={} outside output={}",
                    route.choice,
                    output.len()
                );
                output[first..first + width]
                    .copy_from_slice(&completed.work.output[first..first + width]);
            }
            if completed.error.is_some() {
                failure = completed.error;
            }
            worker.work = Some(completed.work);
        }
        self.running = false;
        if let Some(error) = failure {
            anyhow::bail!("disk expert contribution: {error}");
        }
        Ok(())
    }
}

impl Drop for DiskExperts {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.input.take();
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
fn native_weights(shape: crate::cpu::expert_ffn::Shape) -> crate::cpu::expert_ffn::ExpertWeights {
    let down = 2 * shape.intermediate * shape.hidden;
    crate::cpu::expert_ffn::ExpertWeights {
        gate_up: 0..down,
        gate_up_stride: shape.hidden,
        down: down..down + shape.hidden * shape.intermediate,
        down_stride: shape.intermediate,
    }
}

fn disk_contributions<const FP8: bool>(
    reader: &DiskReader,
    catalogue: &ExpertCatalogue,
    work: &mut Work,
    weights: &crate::cpu::expert_ffn::ExpertWeights,
    scratch: &mut crate::cpu::expert_ffn::Scratch,
    vector: &mut [bf16],
    isa: crate::cpu::expert_ffn::Isa,
) -> Result<()> {
    use crate::cpu::expert_ffn::{
        Contribution, Fp8Contribution, Shape, contribution, fp8_contribution, native_bytes,
    };
    let geometry = catalogue.geometry();
    let shape = Shape::new(geometry.hidden as usize, geometry.intermediate as usize)?;
    for expert in 0..geometry.experts {
        if !work.routes.iter().any(|route| route.expert == expert) {
            continue;
        }
        let slot = reader.read_expert(work.layer, expert, catalogue)?;
        let values = reader.values(slot)?;
        for route in work.routes.iter().filter(|route| route.expert == expert) {
            let first = route.row as usize * shape.hidden;
            if FP8 {
                fp8_contribution(
                    isa,
                    Fp8Contribution {
                        shape: &shape,
                        slot: native_bytes(&values),
                        slot_layout: catalogue
                            .layout()
                            .fp8_slot()
                            .context("FP8 disk slot missing")?,
                        hidden: &work.hidden[first..first + shape.hidden],
                        routing: route.weight,
                    },
                    vector,
                    scratch,
                )?;
            } else {
                contribution(
                    isa,
                    Contribution {
                        shape: &shape,
                        arena: &values,
                        weights,
                        hidden: &work.hidden[first..first + shape.hidden],
                        routing: route.weight,
                    },
                    vector,
                    scratch,
                )?;
            }
            ensure!(
                vector.iter().all(|value| value.to_f32().is_finite()),
                "non-finite disk contribution layer={} expert={expert} row={}",
                work.layer,
                route.row
            );
            let first = route.choice as usize * shape.hidden;
            work.output[first..first + shape.hidden].copy_from_slice(vector);
        }
    }
    Ok(())
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

    fn fp8_entry(
        role: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        shard: &str,
        start: u64,
        len: u64,
    ) -> TensorEntry {
        TensorEntry {
            dtype,
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

    fn fp8_manifest(geometry: ExpertGeometry) -> TensorManifest {
        let mut entries = BTreeMap::new();
        let mut next = 0u64;
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                let name = |role: &str| {
                    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{role}")
                };
                for (role, dtype, shape) in [
                    (
                        "gate_proj.weight",
                        safetensors::Dtype::F8_E4M3,
                        vec![geometry.intermediate as usize, geometry.hidden as usize],
                    ),
                    (
                        "up_proj.weight",
                        safetensors::Dtype::F8_E4M3,
                        vec![geometry.intermediate as usize, geometry.hidden as usize],
                    ),
                    (
                        "down_proj.weight",
                        safetensors::Dtype::F8_E4M3,
                        vec![geometry.hidden as usize, geometry.intermediate as usize],
                    ),
                    (
                        "gate_proj.weight_scale",
                        safetensors::Dtype::BF16,
                        vec![geometry.intermediate as usize, 1],
                    ),
                    (
                        "up_proj.weight_scale",
                        safetensors::Dtype::BF16,
                        vec![geometry.intermediate as usize, 1],
                    ),
                    (
                        "down_proj.weight_scale",
                        safetensors::Dtype::BF16,
                        vec![geometry.hidden as usize, 1],
                    ),
                ] {
                    let elements: usize = shape.iter().product();
                    let len = elements as u64
                        * match dtype {
                            safetensors::Dtype::BF16 => 2,
                            _ => 1,
                        };
                    entries.insert(
                        name(role),
                        fp8_entry(role, dtype, shape, "fp8-0.safetensors", next, len),
                    );
                    next += len;
                }
            }
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
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest(gate_shard, down_shard),
            geometry(),
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        Ok((dir, catalogue))
    }

    #[test]
    fn reads_native_ranges_at_the_catalogue_offsets() -> Result<()> {
        assert!(AlignedSlot::new(0).is_err());
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let pattern = pattern(1536, 0);
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        for (layer, expert) in [(0u32, 0u32), (0, 3), (1, 0), (1, 3)] {
            let slot = reader.read_expert(layer, expert, &catalogue)?;
            {
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
            }
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
        Ok(())
    }

    #[test]
    fn a_truncated_shard_names_the_range() -> Result<()> {
        let dir = temp_dir("truncated")?;
        std::fs::write(dir.path().join("shard-0.safetensors"), vec![0u8; 1000])?;
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest("shard-0.safetensors", "shard-0.safetensors"),
            geometry(),
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        let Err(error) = DiskReader::open(&catalogue, dir.path(), 1) else {
            panic!("open succeeded on a truncated shard");
        };
        let error = error.to_string();
        assert!(error.contains("exceeds 1000 bytes"), "{error}");
        Ok(())
    }

    #[test]
    fn ring_slots_are_guarded_until_their_lease_drops() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        let first = reader.read_expert(0, 0, &catalogue)?;
        let second = reader.read_expert(0, 1, &catalogue)?;
        assert_ne!(first, second);
        let error = reader
            .read_expert(0, 2, &catalogue)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("all 2 disk ring slots are in use"),
            "{error}"
        );
        drop(reader.values(first)?);
        assert!(reader.values(first).is_err());
        let third = reader.read_expert(0, 2, &catalogue)?;
        assert_eq!(third, first);
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
        drop(values);
        drop(second_values);
        assert!(reader.values(first).is_err());
        let single = DiskReader::open(&catalogue, dir.path(), 1)?;
        let held = single.read_expert(0, 0, &catalogue)?;
        let error = single
            .read_expert(0, 1, &catalogue)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("all 1 disk ring slots are in use"),
            "{error}"
        );
        drop(single.values(held)?);
        Ok(())
    }

    #[test]
    fn a_copied_slot_returns_only_through_its_copy_event() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let slot = reader.read_expert(0, 0, &catalogue)?;
        reader.values(slot)?.copied(5)?;
        let error = reader
            .read_expert(0, 1, &catalogue)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("all 1 disk ring slots are in use"),
            "{error}"
        );
        let error = reader.reclaim(slot, 4).unwrap_err().to_string();
        assert!(error.contains("has no copy with event 4"), "{error}");
        reader.reclaim(slot, 5)?;
        let again = reader.read_expert(0, 1, &catalogue)?;
        assert_eq!(again, slot);
        Ok(())
    }

    #[test]
    fn disk_workers_preserve_compacted_rows_choices_and_native_offsets() -> Result<()> {
        use crate::cpu::expert_ffn::{Contribution, Isa, Scratch, Shape, contribution};
        use crate::experts::runtime::Route;
        use std::sync::Arc;
        let dir = temp_dir("workers")?;
        let geometry = ExpertGeometry {
            intermediate: 8,
            ..geometry()
        };
        let shape = Shape::new(8, 8)?;
        let mut bytes = Vec::new();
        let mut manifest = TensorManifest {
            entries: BTreeMap::new(),
            resident_bytes: 0,
        };
        for layer in 0..geometry.layers {
            for (role, dims) in [
                ("gate_up_proj", vec![4, 16, 8]),
                ("down_proj", vec![4, 8, 8]),
            ] {
                let start = bytes.len() as u64;
                for index in 0..dims.iter().product::<usize>() {
                    let value = bf16::from_f32(
                        ((index * 37 + layer as usize * 11) % 53) as f32 / 64.0 - 0.4,
                    );
                    bytes.extend(value.to_bits().to_le_bytes());
                }
                manifest.entries.insert(
                    format!("model.language_model.layers.{layer}.mlp.experts.{role}"),
                    entry(role, dims, "native.bin", start, bytes.len() as u64 - start),
                );
            }
        }
        std::fs::write(dir.path().join("native.bin"), bytes)?;
        let catalogue = Arc::new(ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            crate::model_spec::ExpertFormat::Bf16,
        )?);
        let mut pool = DiskExperts::new(catalogue.clone(), dir.path(), 2, 3)?;
        let oracle = DiskReader::open(&catalogue, dir.path(), 1)?;
        let mut scratch = Scratch::new(&shape);
        let hidden = (0..24)
            .map(|index| bf16::from_f32(((index * 11) % 29) as f32 / 32.0))
            .collect::<Vec<_>>();
        for layer in [1, 0, 1] {
            let routes = [
                Route {
                    row: 1,
                    choice: 5,
                    expert: 3,
                    weight: bf16::from_f32(0.25),
                },
                Route {
                    row: 0,
                    choice: 1,
                    expert: 0,
                    weight: bf16::from_f32(0.5),
                },
                Route {
                    row: 2,
                    choice: 4,
                    expert: 3,
                    weight: bf16::from_f32(0.75),
                },
            ];
            let mut output = vec![bf16::from_f32(-99.0); 6 * 8];
            let mut expected = output.clone();
            for route in routes {
                let slot = oracle.read_expert(layer, route.expert, &catalogue)?;
                let values = oracle.values(slot)?;
                let first = route.row as usize * 8;
                contribution(
                    Isa::detect(),
                    Contribution {
                        shape: &shape,
                        arena: &values,
                        weights: &native_weights(shape),
                        hidden: &hidden[first..first + 8],
                        routing: route.weight,
                    },
                    &mut expected[route.choice as usize * 8..(route.choice as usize + 1) * 8],
                    &mut scratch,
                )?;
            }
            assert!(
                pool.start(layer, &hidden, 3, &[routes[0], routes[0]])
                    .is_err()
            );
            pool.start(layer, &hidden, 3, &routes)?;
            assert!(pool.start(layer, &hidden, 3, &routes).is_err());
            pool.finish(&mut output)?;
            assert_eq!(output, expected);
            assert!(pool.finish(&mut output).is_err());
        }
        Ok(())
    }

    #[test]
    fn fp8_disk_workers_preserve_native_scales_rows_and_choices() -> Result<()> {
        use crate::cpu::expert_ffn::{
            Fp8Contribution, Isa, Scratch, Shape, fp8_contribution, native_bytes,
        };
        use crate::experts::runtime::Route;
        use std::sync::Arc;
        let dir = temp_dir("fp8-workers")?;
        let geometry = ExpertGeometry {
            hidden: 16,
            intermediate: 8,
            ..geometry()
        };
        let manifest = fp8_manifest(geometry);
        let len = manifest
            .entries
            .values()
            .map(|e| e.data_offsets[1])
            .max()
            .context("empty fixture")?;
        let mut bytes = vec![0u8; len as usize];
        for (name, entry) in &manifest.entries {
            let data = &mut bytes[entry.data_offsets[0] as usize..entry.data_offsets[1] as usize];
            if name.ends_with("weight_scale") {
                for (index, value) in data.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let scale = bf16::from_f32(
                        (1.0 + ((index + entry.data_offsets[0] as usize / 2) % 17) as f32 / 16.0)
                            / 8.0,
                    );
                    value.copy_from_slice(&scale.to_le_bytes());
                }
            } else {
                for (index, value) in data.iter_mut().enumerate() {
                    *value = (((index * 37 + entry.data_offsets[0] as usize) % 64) as u8)
                        | (((index / 7 % 2) as u8) << 7);
                }
            }
        }
        std::fs::write(dir.path().join("fp8-0.safetensors"), bytes)?;
        let catalogue = Arc::new(ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            crate::model_spec::ExpertFormat::Fp8W8a16,
        )?);
        let mut pool = DiskExperts::new(catalogue.clone(), dir.path(), 2, 3)?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let shape = Shape::new(16, 8)?;
        let mut scratch = Scratch::new(&shape);
        let hidden = (0..3 * shape.hidden)
            .map(|k| bf16::from_f32(((k * 11 % 29) as f32 - 13.0) / 32.0))
            .collect::<Vec<_>>();
        for layer in [1, 0, 1] {
            let routes = [(1, 5, 3, 0.25), (0, 1, 0, 0.5), (2, 4, 3, 0.75)].map(
                |(row, choice, expert, weight)| Route {
                    row,
                    choice,
                    expert,
                    weight: bf16::from_f32(weight),
                },
            );
            let mut actual = vec![bf16::from_f32(-99.0); 6 * shape.hidden];
            let mut expected = actual.clone();
            for route in routes {
                let slot = reader.read_expert(layer, route.expert, &catalogue)?;
                let values = reader.values(slot)?;
                fp8_contribution(
                    Isa::Scalar,
                    Fp8Contribution {
                        shape: &shape,
                        slot: native_bytes(&values),
                        slot_layout: catalogue.layout().fp8_slot().context("FP8 fixture slot")?,
                        hidden: &hidden[route.row as usize * shape.hidden..][..shape.hidden],
                        routing: route.weight,
                    },
                    &mut expected[route.choice as usize * shape.hidden..][..shape.hidden],
                    &mut scratch,
                )?;
            }
            pool.start(layer, &hidden, 3, &routes)?;
            pool.finish(&mut actual)?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        println!(
            "PASS FP8_DISK_POOL layers=2 jobs=3 rows=3 choices=3 native_offsets=true bit_mismatches=0"
        );
        Ok(())
    }

    #[test]
    fn fp8_slot_packs_the_six_ranges_in_the_documented_order() -> Result<()> {
        let geometry = geometry();
        let dir = temp_dir("fp8-order")?;
        let manifest = fp8_manifest(geometry);
        let mut bytes = 0u64;
        for entry in manifest.entries.values() {
            bytes = bytes.max(entry.data_offsets[1]);
        }
        std::fs::write(
            dir.path().join("fp8-0.safetensors"),
            pattern(bytes as usize, 5),
        )?;
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            crate::model_spec::ExpertFormat::Fp8W8a16,
        )?;
        assert_eq!(catalogue.slot_bytes(), 128);
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let slot = reader.read_expert(1, 2, &catalogue)?;
        let ExpertRange::Fp8 {
            gate,
            up,
            down,
            gate_scale,
            up_scale,
            down_scale,
        } = catalogue.range(1, 2)?
        else {
            bail!("FP8 range expected");
        };
        let file = std::fs::File::open(dir.path().join("fp8-0.safetensors"))?;
        let mut expected = Vec::new();
        for native in [gate, up, down, gate_scale, up_scale, down_scale] {
            let mut raw = vec![0u8; native.len as usize];
            file.read_exact_at(&mut raw, native.offset)?;
            expected.extend_from_slice(&raw);
        }
        assert_eq!(expected.len(), catalogue.slot_bytes() as usize);
        let expected: Vec<bf16> = expected
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)))
            .collect();
        assert_eq!(reader.values(slot)?.to_vec(), expected);
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
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        let reader = DiskReader::open(&catalogue, &dir, 2)?;
        let gate_up_elements = 2 * geometry.intermediate as usize * geometry.hidden as usize;
        let down_elements = geometry.hidden as usize * geometry.intermediate as usize;
        for (layer, expert) in [(0u32, 0u32), (39, 255)] {
            let slot = reader.read_expert(layer, expert, &catalogue)?;
            let values = reader.values(slot)?.to_vec();
            let ExpertRange::Bf16 { gate_up, down } = catalogue.range(layer, expert)? else {
                bail!("BF16 range expected");
            };
            for (name, range, offset, elements) in [
                (
                    format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                    gate_up,
                    0usize,
                    gate_up_elements,
                ),
                (
                    format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                    down,
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
        }
        Ok(())
    }
}
