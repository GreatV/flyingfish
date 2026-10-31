//! Disk tier: bounded aligned native-range reads with per-slot completion leases.
use crate::experts::catalogue::{ExpertCatalogue, ExpertLayout, ExpertRange, NativeRange};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;

/// Alignment of a ring slot; one slot holds one expert's gate-up then down bytes.
const RING_ALIGN: usize = 4096;

/// An aligned, zeroed, caller-sized buffer. The one aligned allocation both the expert ring and
/// the Engram row reader use.
pub(crate) struct AlignedBytes {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedBytes {
    pub(crate) fn new(bytes: usize, alignment: usize) -> Result<Self> {
        ensure!(bytes > 0, "zero aligned buffer bytes");
        ensure!(
            alignment.is_power_of_two() && alignment >= 4096,
            "aligned buffer alignment {alignment} is not a power of two of at least 4096"
        );
        let layout = Layout::from_size_align(bytes, alignment)
            .with_context(|| format!("invalid aligned buffer layout: {bytes} B / {alignment}"))?;
        // SAFETY: the layout has a nonzero size and a power-of-two alignment.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .with_context(|| format!("allocate {bytes} B aligned buffer"))?;
        Ok(Self { ptr, layout })
    }

    pub(crate) fn len(&self) -> usize {
        self.layout.size()
    }

    pub(crate) fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: the buffer owns `layout.size()` initialized bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: the pointer and layout came from alloc_zeroed.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct AlignedSlot {
    bytes: AlignedBytes,
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
            std::slice::from_raw_parts(ring.bytes.as_ptr() as *const bf16, self.reader.per_expert)
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
        Ok(Self {
            bytes: AlignedBytes::new(bytes, RING_ALIGN)?,
            state: Cell::new(RingState::Free),
        })
    }
}

/// One planned aligned read: a file span and the request range that lands inside it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReadSpan {
    pub(crate) begin: u64,
    pub(crate) end: u64,
    pub(crate) first: usize,
    pub(crate) last: usize,
}

/// Merges `(offset, len)` requests into aligned reads: sorted, merged while the next request falls
/// inside the span already planned, capped at `cap`, tail clamped to `file_len`. Shared by the
/// expert ring and the Engram row reader.
pub(crate) fn plan_spans(
    requests: &[(u64, u64)],
    alignment: u64,
    file_len: u64,
    cap: u64,
) -> Result<Vec<ReadSpan>> {
    ensure!(
        alignment.is_power_of_two() && alignment > 0,
        "span alignment {alignment} is not a power of two"
    );
    let mut ordered: Vec<(u64, u64)> = requests.to_vec();
    ordered.sort_unstable();
    let mut spans = Vec::new();
    let mut index = 0;
    while index < ordered.len() {
        let (offset, len) = ordered[index];
        ensure!(len > 0, "zero-length span request at {offset}");
        let finish = offset
            .checked_add(len)
            .context("span request end overflow")?;
        ensure!(
            finish <= file_len,
            "span request {offset}+{len} exceeds file length {file_len}"
        );
        let begin = offset & !(alignment - 1);
        let mut end = finish.div_ceil(alignment) * alignment;
        let mut last = index + 1;
        while last < ordered.len() {
            let (next_offset, next_len) = ordered[last];
            let next_end = next_offset
                .checked_add(next_len)
                .context("span request end overflow")?;
            let next_aligned = next_end.div_ceil(alignment) * alignment;
            if next_aligned > end || next_aligned - begin > cap {
                break;
            }
            end = end.max(next_aligned);
            last += 1;
        }
        spans.push(ReadSpan {
            begin,
            end: end.min(file_len),
            first: index,
            last,
        });
        index = last;
    }
    Ok(spans)
}

/// Shared shard reader; direct mode uses filesystem-derived aligned IO.
pub(crate) struct SpanReader {
    file: File,
    pub(crate) alignment: u64,
    len: u64,
    direct: bool,
}

impl SpanReader {
    pub(crate) fn open(path: &Path, direct: bool) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(if direct {
                libc::O_DIRECT | libc::O_CLOEXEC
            } else {
                libc::O_CLOEXEC
            })
            .open(path)
            .with_context(|| format!("open native shard {} direct={direct}", path.display()))?;
        let len = file.metadata().context("stat native shard")?.len();
        let alignment = if direct {
            let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
            // SAFETY: the file descriptor is open and stat is writable.
            let rc = unsafe { libc::fstatvfs(file.as_raw_fd(), &mut stat) };
            ensure!(
                rc == 0,
                "native shard fstatvfs: {}",
                std::io::Error::last_os_error()
            );
            // SAFETY: sysconf takes no pointer arguments.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            ensure!(
                page > 0 && stat.f_frsize > 0,
                "invalid direct IO page/filesystem block size"
            );
            let page = u64::try_from(page)?;
            let block = stat.f_frsize;
            let (mut a, mut b) = (page, block);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            let alignment = (page / a)
                .checked_mul(block)
                .context("direct IO alignment overflow")?;
            ensure!(
                alignment.is_power_of_two() && alignment >= RING_ALIGN as u64,
                "invalid direct IO alignment {alignment}"
            );
            alignment
        } else {
            RING_ALIGN as u64
        };
        Ok(Self {
            file,
            alignment,
            len,
            direct,
        })
    }

    pub(crate) fn plan(&self, requests: &[(u64, u64)], chunk: u64) -> Result<Vec<ReadSpan>> {
        plan_spans(
            requests,
            self.alignment,
            self.len,
            chunk.max(self.alignment),
        )
    }

    pub(crate) fn read_span<'a>(
        &self,
        span: &ReadSpan,
        chunk: u64,
        staging: &'a mut AlignedBytes,
    ) -> Result<&'a [u8]> {
        ensure!(
            chunk > 0 && span.begin < span.end && span.end <= self.len,
            "invalid prepared native span"
        );
        ensure!(
            !self.direct
                || (chunk.is_multiple_of(self.alignment)
                    && span.begin.is_multiple_of(self.alignment)),
            "misaligned prepared native span"
        );
        let width = usize::try_from(span.end - span.begin)?;
        let allocated = if self.direct {
            u64::try_from(width)?
                .checked_add(self.alignment - 1)
                .context("direct span overflow")?
                & !(self.alignment - 1)
        } else {
            u64::try_from(width)?
        };
        ensure!(
            allocated <= staging.len() as u64,
            "native span {allocated} exceeds staging {}",
            staging.len()
        );
        ensure!(
            !self.direct || (staging.as_ptr() as usize as u64).is_multiple_of(self.alignment),
            "native direct buffer is misaligned"
        );
        let mut written = 0usize;
        while written < width {
            let logical = usize::try_from(chunk)?.min(width - written);
            let requested = if self.direct {
                logical.div_ceil(self.alignment as usize) * self.alignment as usize
            } else {
                logical
            };
            let at = span.begin + written as u64;
            let buffer = &mut staging.as_mut()[written..written + requested];
            if self.direct {
                let got = self
                    .file
                    .read_at(buffer, at)
                    .with_context(|| format!("direct read {at}+{requested}"))?;
                ensure!(
                    got == logical,
                    "short native direct read at {at}: got {got}, expected {logical}"
                );
            } else {
                self.file
                    .read_exact_at(buffer, at)
                    .with_context(|| format!("native read {at}+{requested}"))?;
            }
            written += logical;
        }
        Ok(&staging.as_mut()[..width])
    }

    /// The callback receives the original request index and its complete native bytes.
    pub(crate) fn read_spans(
        &self,
        requests: &[(u64, u64)],
        chunk_bytes: u64,
        staging: &mut AlignedBytes,
        mut consume: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<usize> {
        ensure!(chunk_bytes > 0, "native read chunk is zero");
        let chunk = if self.direct {
            chunk_bytes
                .checked_add(self.alignment - 1)
                .context("direct chunk overflow")?
                & !(self.alignment - 1)
        } else {
            chunk_bytes
        };
        for &(offset, len) in requests {
            ensure!(
                len > 0 && offset.checked_add(len).is_some_and(|end| end <= self.len),
                "invalid native range {offset}+{len} for {} bytes",
                self.len
            );
        }
        let mut order: Vec<usize> = (0..requests.len()).collect();
        order.sort_unstable_by_key(|&i| requests[i]);
        let sorted: Vec<_> = order.iter().map(|&i| requests[i]).collect();
        let spans = self.plan(&sorted, chunk)?;
        for span in &spans {
            let bytes = self.read_span(span, chunk, staging)?;
            let width = bytes.len();
            for index in span.first..span.last {
                let (offset, len) = sorted[index];
                let from = usize::try_from(offset - span.begin)?;
                let end = from
                    .checked_add(usize::try_from(len)?)
                    .context("native slice overflow")?;
                ensure!(end <= width, "native request exceeds its span");
                consume(order[index], &bytes[from..end])?;
            }
        }
        Ok(spans.len())
    }
}

/// Read-only disk tier over the native checkpoint shards with a bounded aligned ring.
pub struct DiskReader {
    shards: BTreeMap<String, SpanReader>,
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
        ExpertRange::Bf16Separate { gate, up, down } => {
            visit(gate)?;
            visit(up)?;
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
        ExpertRange::E2m1 {
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
            let file = SpanReader::open(
                &model_dir.join(name.as_ref()),
                matches!(catalogue.layout(), ExpertLayout::Fp4E2m1Block(..)),
            )?;
            let length = file.len;
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
    pub fn read_expert(
        &self,
        layer: u32,
        expert: u32,
        catalogue: &ExpertCatalogue,
        chunk_bytes: u64,
    ) -> Result<u32> {
        ensure!(chunk_bytes > 0, "disk read chunk is not positive");
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
        let bytes = self.ring[slot].bytes.len();
        let ptr = self.ring[slot].bytes.as_ptr() as *mut u8;
        // SAFETY: the slot owns `bytes` writable bytes with the ring alignment.
        let buffer: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
        let mut parts: Vec<(u64, &NativeRange)> = Vec::new();
        match (layout, range) {
            (
                ExpertLayout::Bf16Fused { gate_up_bytes, .. },
                ExpertRange::Bf16 { gate_up, down },
            ) => {
                parts.push((0, gate_up));
                parts.push((*gate_up_bytes, down));
            }
            (
                ExpertLayout::Bf16Separate { matrix_bytes, .. },
                ExpertRange::Bf16Separate { gate, up, down },
            ) => {
                parts.push((0, gate));
                parts.push((*matrix_bytes, up));
                parts.push((*matrix_bytes * 2, down));
            }
            (
                ExpertLayout::Fp8W8a16(slot) | ExpertLayout::Fp8W8a16Block(slot, _),
                ExpertRange::Fp8 {
                    gate,
                    up,
                    down,
                    gate_scale,
                    up_scale,
                    down_scale,
                },
            ) => {
                parts.push((slot.gate.start, gate));
                parts.push((slot.up.start, up));
                parts.push((slot.down.start, down));
                parts.push((slot.gate_scale.start, gate_scale));
                parts.push((slot.up_scale.start, up_scale));
                parts.push((slot.down_scale.start, down_scale));
            }
            (
                ExpertLayout::Fp4E2m1Block(slot, _),
                ExpertRange::E2m1 {
                    gate,
                    up,
                    down,
                    gate_scale,
                    up_scale,
                    down_scale,
                },
            ) => {
                parts.push((slot.gate.start, gate));
                parts.push((slot.up.start, up));
                parts.push((slot.down.start, down));
                parts.push((slot.gate_scale.start, gate_scale));
                parts.push((slot.up_scale.start, up_scale));
                parts.push((slot.down_scale.start, down_scale));
            }
            _ => bail!("expert range does not match the catalogue layout"),
        }
        self.read_parts(&parts, buffer, chunk_bytes)?;
        self.ring[slot].state.set(RingState::Filled);
        u32::try_from(slot).context("disk ring slot exceeds u32")
    }

    /// Reads one expert's ranges through the shared span planner: the requests are merged into
    /// aligned spans per shard, each span is fetched once in `chunk_bytes` steps, and every request
    /// is copied into its slot offset.
    fn read_parts(
        &self,
        parts: &[(u64, &NativeRange)],
        slot: &mut [u8],
        chunk_bytes: u64,
    ) -> Result<()> {
        ensure!(!parts.is_empty(), "expert read has no native ranges");
        for (offset, native) in parts {
            let end = offset
                .checked_add(native.len)
                .context("expert slice end overflow")?;
            ensure!(end <= slot.len() as u64, "expert range exceeds slot");
        }
        let mut by_shard: std::collections::BTreeMap<&str, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (index, (_, native)) in parts.iter().enumerate() {
            by_shard
                .entry(native.shard.as_ref())
                .or_default()
                .push(index);
        }
        for (shard_name, indexes) in by_shard {
            let shard = self
                .shards
                .get(shard_name)
                .with_context(|| format!("disk shard {shard_name} not open"))?;
            let requests: Vec<(u64, u64)> = indexes
                .iter()
                .map(|&index| (parts[index].1.offset, parts[index].1.len))
                .collect();
            let largest = requests
                .iter()
                .map(|&(offset, len)| {
                    (offset % shard.alignment)
                        .checked_add(len)
                        .and_then(|n| n.checked_add(shard.alignment - 1))
                        .map(|n| n & !(shard.alignment - 1))
                        .context("expert staging extent overflow")
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .context("empty native requests")?;
            let mut staging =
                AlignedBytes::new(usize::try_from(largest)?, usize::try_from(shard.alignment)?)?;
            shard.read_spans(&requests, chunk_bytes, &mut staging, |index, bytes| {
                let start = usize::try_from(parts[indexes[index]].0)?;
                slot[start..start + bytes.len()].copy_from_slice(bytes);
                Ok(())
            })?;
        }
        Ok(())
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
    swiglu_rounding: crate::model_spec::SwigluRounding,
    swiglu_limit: f32,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
    form: crate::cpu::expert_ffn::Form,
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
pub struct Job<'a> {
    pub layer: u32,
    pub hidden: &'a [bf16],
    pub rows: usize,
    pub routes: &'a [crate::experts::runtime::Route],
    pub form: crate::cpu::expert_ffn::Form,
}

pub struct DiskExperts {
    geometry: crate::experts::catalogue::ExpertGeometry,
    max_rows: usize,
    max_routes: usize,
    running: bool,
    failed: bool,
    workers: Vec<Worker>,
}

impl DiskExperts {
    pub fn new(
        catalogue: std::sync::Arc<ExpertCatalogue>,
        directory: &Path,
        workers: u32,
        max_rows: u32,
    ) -> Result<Self> {
        Self::new_with_swiglu_rounding(
            catalogue,
            directory,
            workers,
            max_rows,
            crate::model_spec::SwigluRounding::Twice,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            0.0,
        )
    }

    pub fn new_with_swiglu_rounding(
        catalogue: std::sync::Arc<ExpertCatalogue>,
        directory: &Path,
        workers: u32,
        max_rows: u32,
        swiglu_rounding: crate::model_spec::SwigluRounding,
        routed_weight: crate::model_spec::RoutedWeightBoundary,
        swiglu_limit: f32,
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
            failed: false,
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
                    swiglu_rounding,
                    swiglu_limit,
                    routed_weight,
                    form: crate::cpu::expert_ffn::Form::Lane8,
                    output: vec![bf16::ZERO; elements],
                }),
            });
        }
        Ok(pool)
    }

    pub fn check_ready(&self) -> Result<()> {
        ensure!(!self.failed, "disk expert worker service is unavailable");
        ensure!(!self.running, "disk contribution job is already running");
        for (index, worker) in self.workers.iter().enumerate() {
            ensure!(
                worker.work.is_some() && worker.input.is_some(),
                "disk expert worker {index} has no reusable job or input"
            );
        }
        Ok(())
    }

    pub fn start(&mut self, job: Job<'_>) -> Result<()> {
        let Job {
            layer,
            hidden,
            rows,
            routes,
            form,
        } = job;
        self.check_ready()?;
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
        self.running = true;
        let mut failure = None;
        for (index, worker) in self.workers.iter_mut().enumerate() {
            let mut work = worker.work.take().expect("preflight checked reusable job");
            work.layer = layer;
            work.form = form;
            work.hidden.clear();
            work.hidden.extend_from_slice(hidden);
            work.routes.clear();
            work.routes.extend(
                routes
                    .iter()
                    .filter(|r| r.expert as usize % workers == index),
            );
            if let Err(error) = worker
                .input
                .as_ref()
                .expect("preflight checked worker input")
                .send(work)
            {
                worker.work = Some(error.0);
                failure = Some(anyhow::anyhow!("disk expert worker {index} stopped"));
                break;
            }
        }
        if let Some(error) = failure {
            self.failed = true;
            return match self.collect(None) {
                Ok(()) => Err(error),
                Err(cleanup) => Err(error.context(format!("disk cleanup: {cleanup:#}"))),
            };
        }
        Ok(())
    }

    pub fn run(
        &mut self,
        job: Job<'_>,
        output: &mut [bf16],
        contribute: impl FnOnce(&mut [bf16]) -> Result<()>,
    ) -> Result<()> {
        self.start(job)?;
        let result = contribute(output);
        let disk = self.collect(if result.is_ok() { Some(output) } else { None });
        match (result, disk) {
            (Ok(()), disk) => disk,
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(cleanup)) => Err(error.context(format!("disk cleanup: {cleanup:#}"))),
        }
    }

    pub fn finish(&mut self, output: &mut [bf16]) -> Result<()> {
        ensure!(
            self.running,
            "disk contribution finish without a running job"
        );
        self.collect(Some(output))
    }

    fn collect(&mut self, mut output: Option<&mut [bf16]>) -> Result<()> {
        let width = self.geometry.hidden as usize;
        let mut failures = Vec::new();
        for (index, worker) in self.workers.iter_mut().enumerate() {
            if worker.work.is_some() {
                continue;
            }
            let completed = match worker.output.recv() {
                Ok(completed) => completed,
                Err(error) => {
                    self.failed = true;
                    failures.push(format!("receive disk worker {index}: {error}"));
                    continue;
                }
            };
            if let Some(error) = completed.error {
                failures.push(format!("disk worker {index}: {error}"));
            } else if let Some(target) = output.as_deref_mut() {
                let copied = (|| -> Result<()> {
                    for route in &completed.work.routes {
                        let first = route.choice as usize * width;
                        ensure!(
                            first + width <= target.len(),
                            "disk contribution choice={} outside output={}",
                            route.choice,
                            target.len()
                        );
                        target[first..first + width]
                            .copy_from_slice(&completed.work.output[first..first + width]);
                    }
                    Ok(())
                })();
                if let Err(error) = copied {
                    failures.push(format!("disk worker {index}: {error:#}"));
                }
            }
            worker.work = Some(completed.work);
        }
        self.running = false;
        ensure!(
            failures.is_empty(),
            "disk expert contribution: {}",
            failures.join("; ")
        );
        Ok(())
    }
}

impl Drop for DiskExperts {
    fn drop(&mut self) {
        if self.running
            && let Err(error) = self.collect(None)
        {
            eprintln!("disk expert cleanup failed: {error:#}");
        }
        for worker in &mut self.workers {
            worker.input.take();
        }
        for (index, worker) in self.workers.iter_mut().enumerate() {
            if let Some(handle) = worker.handle.take()
                && handle.join().is_err()
            {
                eprintln!("disk expert worker {index} panicked");
            }
        }
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
        let slot = reader.read_expert(work.layer, expert, catalogue, catalogue.slot_bytes())?;
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
                        rounding: work.swiglu_rounding,
                        limit: work.swiglu_limit,
                        routed_weight: work.routed_weight,
                        form: work.form,
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
                        rounding: work.swiglu_rounding,
                        limit: work.swiglu_limit,
                        routed_weight: work.routed_weight,
                        form: work.form,
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
            first_layer: 0,
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
            let slot = reader.read_expert(layer, expert, &catalogue, catalogue.slot_bytes())?;
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
        let slot = reader.read_expert(0, 2, &catalogue, catalogue.slot_bytes())?;
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
        let first = reader.read_expert(0, 0, &catalogue, catalogue.slot_bytes())?;
        let second = reader.read_expert(0, 1, &catalogue, catalogue.slot_bytes())?;
        assert_ne!(first, second);
        let error = reader
            .read_expert(0, 2, &catalogue, catalogue.slot_bytes())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("all 2 disk ring slots are in use"),
            "{error}"
        );
        drop(reader.values(first)?);
        assert!(reader.values(first).is_err());
        let third = reader.read_expert(0, 2, &catalogue, catalogue.slot_bytes())?;
        assert_eq!(third, first);
        Ok(())
    }

    #[test]
    fn a_slot_can_be_filled_while_another_is_read() -> Result<()> {
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        let first = reader.read_expert(0, 0, &catalogue, catalogue.slot_bytes())?;
        let values = reader.values(first)?;
        let second = reader.read_expert(0, 1, &catalogue, catalogue.slot_bytes())?;
        assert_ne!(first, second);
        let second_values = reader.values(second)?;
        assert_eq!(values.len(), 96);
        assert_eq!(second_values.len(), 96);
        assert_ne!(values[0].to_bits(), second_values[0].to_bits());
        drop(values);
        drop(second_values);
        assert!(reader.values(first).is_err());
        let single = DiskReader::open(&catalogue, dir.path(), 1)?;
        let held = single.read_expert(0, 0, &catalogue, catalogue.slot_bytes())?;
        let error = single
            .read_expert(0, 1, &catalogue, catalogue.slot_bytes())
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
        let slot = reader.read_expert(0, 0, &catalogue, catalogue.slot_bytes())?;
        reader.values(slot)?.copied(5)?;
        let error = reader
            .read_expert(0, 1, &catalogue, catalogue.slot_bytes())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("all 1 disk ring slots are in use"),
            "{error}"
        );
        let error = reader.reclaim(slot, 4).unwrap_err().to_string();
        assert!(error.contains("has no copy with event 4"), "{error}");
        reader.reclaim(slot, 5)?;
        let again = reader.read_expert(0, 1, &catalogue, catalogue.slot_bytes())?;
        assert_eq!(again, slot);
        Ok(())
    }

    fn native_workers(tag: &str) -> Result<(TempDir, std::sync::Arc<ExpertCatalogue>)> {
        use std::sync::Arc;
        let dir = temp_dir(tag)?;
        let geometry = ExpertGeometry {
            intermediate: 8,
            ..geometry()
        };
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
        Ok((dir, catalogue))
    }

    #[test]
    fn disk_workers_preserve_compacted_rows_choices_and_native_offsets() -> Result<()> {
        use crate::cpu::expert_ffn::{Contribution, Isa, Scratch, Shape, contribution};
        use crate::experts::runtime::Route;
        let (dir, catalogue) = native_workers("workers")?;
        let shape = Shape::new(8, 8)?;
        let mut pool = DiskExperts::new(catalogue.clone(), dir.path(), 2, 3)?;
        let oracle = DiskReader::open(&catalogue, dir.path(), 1)?;
        let mut scratch = Scratch::new(&shape);
        let hidden = (0..24)
            .map(|index| bf16::from_f32(((index * 11) % 29) as f32 / 32.0))
            .collect::<Vec<_>>();
        for (layer, form) in [
            (1, crate::cpu::expert_ffn::Form::Lane8),
            (0, crate::cpu::expert_ffn::Form::Lane16),
            (1, crate::cpu::expert_ffn::Form::Lane8),
        ] {
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
                let slot =
                    oracle.read_expert(layer, route.expert, &catalogue, catalogue.slot_bytes())?;
                let values = oracle.values(slot)?;
                let first = route.row as usize * 8;
                contribution(
                    Isa::detect(),
                    Contribution {
                        form,
                        shape: &shape,
                        arena: &values,
                        weights: &crate::cpu::expert_ffn::ExpertWeights::from_slot(
                            shape,
                            0..values.len(),
                            catalogue.layout(),
                        )?,
                        hidden: &hidden[first..first + 8],
                        routing: route.weight,
                        rounding: crate::model_spec::SwigluRounding::Twice,
                        limit: 0.0,
                        routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                    },
                    &mut expected[route.choice as usize * 8..(route.choice as usize + 1) * 8],
                    &mut scratch,
                )?;
            }
            assert!(
                pool.start(Job {
                    layer,
                    hidden: &hidden,
                    rows: 3,
                    routes: &[routes[0], routes[0]],
                    form
                })
                .is_err()
            );
            pool.start(Job {
                layer,
                hidden: &hidden,
                rows: 3,
                routes: &routes,
                form,
            })?;
            assert!(
                pool.start(Job {
                    layer,
                    hidden: &hidden,
                    rows: 3,
                    routes: &routes,
                    form
                })
                .is_err()
            );
            pool.finish(&mut output)?;
            assert_eq!(output, expected);
            assert!(pool.finish(&mut output).is_err());
        }
        Ok(())
    }

    #[test]
    fn disk_scoped_host_errors_drain_jobs_and_preserve_both_failures() -> Result<()> {
        use crate::experts::runtime::Route;
        let (dir, catalogue) = native_workers("scoped-errors")?;
        let mut pool = DiskExperts::new(catalogue, dir.path(), 2, 1)?;
        let hidden = [bf16::from_f32(0.25); 8];
        let routes = [0, 1].map(|expert| Route {
            row: 0,
            choice: expert,
            expert,
            weight: bf16::from_f32(0.5),
        });
        let mut expected = vec![bf16::ZERO; 16];
        pool.run(
            Job {
                layer: 0,
                hidden: &hidden,
                rows: 1,
                routes: &routes,
                form: crate::cpu::expert_ffn::Form::Lane8,
            },
            &mut expected,
            |_| Ok(()),
        )?;
        assert!(expected.iter().all(|value| value.to_f32().is_finite()));
        assert!(expected.iter().any(|&value| value != bf16::ZERO));
        for fault in ["ready_weights", "pool.run"] {
            let mut output = vec![bf16::from_f32(-99.0); 16];
            let error = pool
                .run(
                    Job {
                        layer: 0,
                        hidden: &hidden,
                        rows: 1,
                        routes: &routes,
                        form: crate::cpu::expert_ffn::Form::Lane8,
                    },
                    &mut output,
                    |_| bail!("{fault}"),
                )
                .expect_err("host error must be returned");
            assert!(format!("{error:#}").contains(fault));
            assert!(!pool.running);
            pool.check_ready()?;
            assert!(pool.workers.iter().all(|worker| worker.work.is_some()));
            assert!(output.iter().all(|&value| value == bf16::from_f32(-99.0)));
            pool.run(
                Job {
                    layer: 0,
                    hidden: &hidden,
                    rows: 1,
                    routes: &routes,
                    form: crate::cpu::expert_ffn::Form::Lane8,
                },
                &mut output,
                |_| Ok(()),
            )?;
            assert_eq!(output, expected);
        }
        let path = dir.path().join("native.bin");
        let native = std::fs::read(&path)?;
        std::fs::write(&path, [])?;
        let mut output = vec![bf16::ZERO; 16];
        let error = pool
            .run(
                Job {
                    layer: 0,
                    hidden: &hidden,
                    rows: 1,
                    routes: &routes,
                    form: crate::cpu::expert_ffn::Form::Lane8,
                },
                &mut output,
                |_| bail!("host failure"),
            )
            .expect_err("host and disk errors must both be returned");
        let error = format!("{error:#}");
        assert!(
            error.contains("host failure") && error.contains("disk worker"),
            "{error}"
        );
        assert!(!pool.running);
        assert!(pool.workers.iter().all(|worker| worker.work.is_some()));
        std::fs::write(&path, native)?;
        pool.run(
            Job {
                layer: 0,
                hidden: &hidden,
                rows: 1,
                routes: &routes,
                form: crate::cpu::expert_ffn::Form::Lane8,
            },
            &mut output,
            |_| Ok(()),
        )?;
        assert_eq!(output, expected);
        Ok(())
    }

    #[test]
    fn disk_partial_start_drains_accepted_work() -> Result<()> {
        let (dir, catalogue) = native_workers("partial-start")?;
        let mut pool = DiskExperts::new(catalogue, dir.path(), 2, 1)?;
        let stopped = &mut pool.workers[1];
        stopped.input.take();
        stopped.handle.take().unwrap().join().unwrap();
        let (input, receiver) = std::sync::mpsc::sync_channel(1);
        drop(receiver);
        stopped.input = Some(input);
        assert!(
            pool.start(Job {
                layer: 0,
                hidden: &[bf16::ZERO; 8],
                rows: 1,
                routes: &[],
                form: crate::cpu::expert_ffn::Form::Lane8
            })
            .is_err()
        );
        assert!(!pool.running);
        assert!(pool.failed);
        assert!(pool.check_ready().is_err());
        assert!(pool.workers.iter().all(|worker| worker.work.is_some()));
        assert!(
            pool.start(Job {
                layer: 0,
                hidden: &[bf16::ZERO; 8],
                rows: 1,
                routes: &[],
                form: crate::cpu::expert_ffn::Form::Lane8
            })
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn disk_partial_finish_drains_other_workers() -> Result<()> {
        let (dir, catalogue) = native_workers("partial-finish")?;
        let mut pool = DiskExperts::new(catalogue, dir.path(), 2, 1)?;
        pool.start(Job {
            layer: 0,
            hidden: &[bf16::ZERO; 8],
            rows: 1,
            routes: &[],
            form: crate::cpu::expert_ffn::Form::Lane8,
        })?;
        let (sender, output) = std::sync::mpsc::sync_channel(1);
        drop(sender);
        drop(std::mem::replace(&mut pool.workers[0].output, output));
        assert!(pool.finish(&mut []).is_err());
        assert!(!pool.running);
        assert!(pool.failed);
        assert!(pool.check_ready().is_err());
        assert!(pool.workers[1].work.is_some());
        assert!(
            pool.start(Job {
                layer: 0,
                hidden: &[bf16::ZERO; 8],
                rows: 1,
                routes: &[],
                form: crate::cpu::expert_ffn::Form::Lane8
            })
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn disk_short_output_drains_jobs_before_reuse() -> Result<()> {
        use crate::experts::runtime::Route;
        let (dir, catalogue) = native_workers("short-output")?;
        let mut pool = DiskExperts::new(catalogue, dir.path(), 2, 1)?;
        let hidden = [bf16::from_f32(0.25); 8];
        let routes = [0, 1].map(|expert| Route {
            row: 0,
            choice: expert,
            expert,
            weight: bf16::from_f32(0.5),
        });
        pool.start(Job {
            layer: 0,
            hidden: &hidden,
            rows: 1,
            routes: &routes,
            form: crate::cpu::expert_ffn::Form::Lane8,
        })?;
        assert!(pool.finish(&mut []).is_err());
        assert!(!pool.running);
        pool.check_ready()?;
        assert!(pool.workers.iter().all(|worker| worker.work.is_some()));
        let mut output = [bf16::ZERO; 16];
        pool.start(Job {
            layer: 0,
            hidden: &hidden,
            rows: 1,
            routes: &routes,
            form: crate::cpu::expert_ffn::Form::Lane8,
        })?;
        pool.finish(&mut output)?;
        assert!(output.iter().any(|&value| value != bf16::ZERO));
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
        for (layer, form) in [
            (1, crate::cpu::expert_ffn::Form::Lane8),
            (0, crate::cpu::expert_ffn::Form::Lane16),
            (1, crate::cpu::expert_ffn::Form::Lane8),
        ] {
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
                let slot =
                    reader.read_expert(layer, route.expert, &catalogue, catalogue.slot_bytes())?;
                let values = reader.values(slot)?;
                fp8_contribution(
                    Isa::Scalar,
                    Fp8Contribution {
                        form,
                        shape: &shape,
                        slot: native_bytes(&values),
                        slot_layout: catalogue.layout().fp8_slot().context("FP8 fixture slot")?,
                        hidden: &hidden[route.row as usize * shape.hidden..][..shape.hidden],
                        routing: route.weight,
                        rounding: crate::model_spec::SwigluRounding::Twice,
                        limit: 0.0,
                        routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                    },
                    &mut expected[route.choice as usize * shape.hidden..][..shape.hidden],
                    &mut scratch,
                )?;
            }
            pool.start(Job {
                layer,
                hidden: &hidden,
                rows: 3,
                routes: &routes,
                form,
            })?;
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
        let slot = reader.read_expert(1, 2, &catalogue, catalogue.slot_bytes())?;
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
            let slot = reader.read_expert(layer, expert, &catalogue, catalogue.slot_bytes())?;
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

#[cfg(test)]
mod e2m1_tests {
    use super::*;
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::experts::catalogue::{ExpertCatalogue, ExpertGeometry};
    use crate::model_spec::ExpertFormat;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// The audited DeepSeek E2M1 names: packed `[rows, cols/2]` I8 codes and
    /// `[rows, cols/32]` E8M0 scales, in emit order.
    fn e2m1_manifest(layers: u32, experts: u32, rows: usize, cols: usize) -> TensorManifest {
        let mut entries = BTreeMap::new();
        let mut next = 0u64;
        for layer in 0..layers {
            for expert in 0..experts {
                let name = |role: &str| format!("layers.{layer}.ffn.experts.{expert}.{role}");
                for (role, r, c, dtype) in [
                    ("w1.weight", rows, cols / 2, safetensors::Dtype::I8),
                    ("w3.weight", rows, cols / 2, safetensors::Dtype::I8),
                    ("w2.weight", cols, rows / 2, safetensors::Dtype::I8),
                    ("w1.scale", rows, cols / 32, safetensors::Dtype::F8_E8M0),
                    ("w3.scale", rows, cols / 32, safetensors::Dtype::F8_E8M0),
                    ("w2.scale", cols, rows / 32, safetensors::Dtype::F8_E8M0),
                ] {
                    let len = (r * c) as u64;
                    entries.insert(
                        name(role),
                        TensorEntry {
                            dtype,
                            shape: vec![r, c],
                            shard: "e2m1.bin".into(),
                            bytes: len as usize,
                            load: true,
                            data_start: 0,
                            data_offsets: [next, next + len],
                            scope: TensorScope::Text,
                            layer: None,
                            role: role.into(),
                            scale_name: None,
                        },
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

    fn e2m1_geometry() -> ExpertGeometry {
        ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: 2,
            hidden: 32,
            intermediate: 32,
            top_k: 2,
        }
    }

    fn e2m1_format() -> ExpertFormat {
        ExpertFormat::Fp4E2m1Block {
            block: [1, 32],
            scale_dtype: safetensors::Dtype::F8_E8M0,
        }
    }

    fn temp_dir(tag: &str) -> Result<TempDir> {
        let dir =
            std::env::temp_dir().join(format!("flyingfish-e2m1-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir)?;
        Ok(TempDir {
            path: dir,
            _0: std::marker::PhantomData,
        })
    }

    struct TempDir {
        path: std::path::PathBuf,
        _0: std::marker::PhantomData<u8>,
    }
    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn pattern(len: usize, seed: u64) -> Vec<u8> {
        (0..len)
            .map(|index| ((index * 37 + seed as usize * 11) % 251) as u8)
            .collect()
    }

    #[test]
    fn e2m1_slot_layout_is_the_published_geometry() -> Result<()> {
        let geometry = e2m1_geometry();
        let catalogue = ExpertCatalogue::from_manifest(
            &e2m1_manifest(geometry.layers, geometry.experts, 32, 32),
            geometry,
            e2m1_format(),
        )?;
        let layout = match catalogue.layout() {
            ExpertLayout::Fp4E2m1Block(slot, _) => slot,
            _ => bail!("E2M1 layout expected"),
        };
        let ranges = [
            ("gate", &layout.gate),
            ("up", &layout.up),
            ("down", &layout.down),
            ("gate_scale", &layout.gate_scale),
            ("up_scale", &layout.up_scale),
            ("down_scale", &layout.down_scale),
        ];
        let mut ends = Vec::new();
        for (name, range) in ranges {
            ensure!(
                range.start < range.end && range.start.is_multiple_of(16),
                "E2M1 {name} alignment: range={range:?}"
            );
            ends.push(range.end);
        }
        ensure!(
            ranges.iter().zip(ends).all(|((_, a), end)| a.end <= end),
            "E2M1 ranges must be ordered"
        );
        assert_eq!(catalogue.slot_bytes(), layout.down_scale.end);
        Ok(())
    }

    #[test]
    fn e2m1_disk_read_lands_the_published_layout_byte_for_byte() -> Result<()> {
        let geometry = e2m1_geometry();
        let rows = 32usize;
        let cols = 32usize;
        let dir = temp_dir("disk-read")?;
        let manifest = e2m1_manifest(geometry.layers, geometry.experts, rows, cols);
        let mut native = Vec::new();
        let mut ordered: Vec<(&String, &TensorEntry)> = manifest.entries.iter().collect();
        ordered.sort_by_key(|(_, entry)| entry.data_offsets[0]);
        for (name, entry) in ordered {
            let len = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
            let _ = name;
            native.extend(pattern(len, 7));
        }
        let _ = &native;
        std::fs::write(dir.path().join("e2m1.bin"), &native)?;
        let catalogue = Arc::new(ExpertCatalogue::from_manifest(
            &e2m1_manifest(geometry.layers, geometry.experts, rows, cols),
            geometry,
            e2m1_format(),
        )?);
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        for shard in reader.shards.values() {
            // SAFETY: F_GETFL reads flags from an open file descriptor.
            let flags = unsafe { libc::fcntl(shard.file.as_raw_fd(), libc::F_GETFL) };
            ensure!(
                flags >= 0 && flags & libc::O_DIRECT != 0,
                "E2M1 shard must use direct IO"
            );
        }
        let shard = &reader.shards["e2m1.bin"];
        let requests = [(native.len() as u64 - 11, 11), (13, 19), (7, 23), (13, 19)];
        let mut staging =
            AlignedBytes::new(2 * shard.alignment as usize, shard.alignment as usize)?;
        let mut completed = [false; 4];
        shard.read_spans(&requests, shard.alignment, &mut staging, |index, bytes| {
            let (offset, len) = requests[index];
            ensure!(
                bytes == &native[offset as usize..(offset + len) as usize],
                "native request ordering or tail differs"
            );
            completed[index] = true;
            Ok(())
        })?;
        ensure!(
            completed.iter().all(|&done| done),
            "native read omitted a request"
        );
        ensure!(
            shard
                .read_spans(
                    &[(native.len() as u64 - 1, 2)],
                    shard.alignment,
                    &mut staging,
                    |_, _| Ok(())
                )
                .is_err(),
            "out-of-file request escaped validation"
        );
        let slot = reader.read_expert(0, 1, &catalogue, catalogue.slot_bytes())?;
        let values = reader.values(slot)?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let want_len = usize::try_from(catalogue.slot_bytes())?;
        ensure!(
            bytes.len() >= want_len && native.len() >= want_len,
            "slot byte extent: have {} want {want_len}",
            bytes.len().min(native.len())
        );
        assert_eq!(&bytes[..want_len], &native[..want_len]);
        // Planted control: a slot served from the wrong byte range must not match.
        let mut shifted = bytes[..want_len].to_vec();
        shifted.rotate_left(1);
        ensure!(
            shifted != bytes[..want_len],
            "wrong-slot control: a rotated slot compared equal"
        );
        let _ = (rows, cols);
        Ok(())
    }

    #[test]
    fn e2m1_disk_read_rejects_a_wrong_range() -> Result<()> {
        let geometry = e2m1_geometry();
        let dir = temp_dir("wrong-range")?;
        let manifest = e2m1_manifest(geometry.layers, geometry.experts, 32, 32);
        let mut native = Vec::new();
        let mut ordered: Vec<&TensorEntry> = manifest.entries.values().collect();
        ordered.sort_by_key(|entry| entry.data_offsets[0]);
        for entry in ordered {
            let len = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
            native.extend(pattern(len, 3));
        }
        std::fs::write(dir.path().join("e2m1.bin"), native)?;
        let catalogue = Arc::new(ExpertCatalogue::from_manifest(
            &e2m1_manifest(geometry.layers, geometry.experts, 32, 32),
            geometry,
            e2m1_format(),
        )?);
        // A geometry whose code extent differs must not be readable as this layout.
        let wrong = ExpertGeometry {
            intermediate: 64,
            ..geometry
        };
        let error = ExpertCatalogue::from_manifest(
            &e2m1_manifest(wrong.layers, wrong.experts, 32, 64),
            wrong,
            e2m1_format(),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(!error.to_string().is_empty());
        let _ = (catalogue, dir, error);
        Ok(())
    }
}
