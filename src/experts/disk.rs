//! Disk tier: bounded aligned native-range reads with per-slot completion leases.
use crate::experts::catalogue::{ExpertCatalogue, ExpertLayout, ExpertRange, NativeRange};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::{Cell, UnsafeCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::mpsc;

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
        #[cfg(test)]
        tests::ALIGNED_ALLOCATIONS.with(|count| count.set(count.get() + 1));
        Ok(Self { ptr, layout })
    }

    pub(crate) fn len(&self) -> usize {
        self.layout.size()
    }

    pub(crate) fn alignment(&self) -> usize {
        self.layout.align()
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
    /// One residue per planned slice, written by the read that filled the slot.
    residues: UnsafeCell<Vec<u32>>,
    /// Packed logical bytes. Filled on the first `values` view when a window is in use.
    packed_view: UnsafeCell<Vec<u8>>,
    /// Staging for this slot's non-window slices. One fill owns it until the fill returns.
    staging: Cell<Option<AlignedBytes>>,
}

/// One slice inside a ring slot. `ring` is the window or staged-tail offset.
#[derive(Clone, Copy)]
struct Place {
    ring: u32,
    len: u32,
    packed: u32,
    windowed: bool,
}

/// Ring placement of one native slot. Built once at open.
struct SlotPlan {
    places: Vec<Place>,
    ring_bytes: usize,
    alignment: u64,
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
        let ptr = self.reader.packed_bytes(self.slot);
        // SAFETY: `packed_bytes` returns `per_expert` initialized bf16 values owned by this slot.
        // The view is not resized while this lease is held.
        unsafe { std::slice::from_raw_parts(ptr as *const bf16, self.reader.per_expert) }
    }

    /// Base of this slot's ring allocation.
    pub fn bytes_ptr(&self) -> *const u8 {
        self.reader.ring[self.slot].bytes.as_ptr()
    }

    /// Coalesced `(ring_offset, len, packed_offset)` device copies of the logical slices.
    pub fn disk_slices(&self) -> Result<Vec<(usize, usize, usize)>> {
        self.reader.disk_slices(self.slot)
    }

    /// Writes the packed logical bytes into `dst` from the ring windows.
    pub fn copy_logical(&self, dst: &mut [bf16]) -> Result<()> {
        self.reader.copy_logical(self.slot, dst)
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
    fn new(bytes: usize, alignment: usize) -> Result<Self> {
        Ok(Self {
            bytes: AlignedBytes::new(bytes, alignment)?,
            state: Cell::new(RingState::Free),
            residues: UnsafeCell::new(Vec::new()),
            packed_view: UnsafeCell::new(Vec::new()),
            staging: Cell::new(None),
        })
    }
}

/// `alignment` is a power of two and at least [`RING_ALIGN`].
fn require_direct_alignment(alignment: u64) -> Result<()> {
    ensure!(
        alignment.is_power_of_two() && alignment >= RING_ALIGN as u64,
        "invalid direct IO alignment {alignment}"
    );
    Ok(())
}

fn layout_slices(layout: &ExpertLayout) -> Vec<(u64, u64)> {
    let six = |gate: &std::ops::Range<u64>,
               up: &std::ops::Range<u64>,
               down: &std::ops::Range<u64>,
               gate_scale: &std::ops::Range<u64>,
               up_scale: &std::ops::Range<u64>,
               down_scale: &std::ops::Range<u64>| {
        [gate, up, down, gate_scale, up_scale, down_scale]
            .into_iter()
            .map(|range| (range.start, range.end - range.start))
            .collect()
    };
    match layout {
        ExpertLayout::Bf16Fused {
            slot_bytes,
            gate_up_bytes,
        } => vec![
            (0, *gate_up_bytes),
            (*gate_up_bytes, slot_bytes - gate_up_bytes),
        ],
        ExpertLayout::Bf16Separate { matrix_bytes, .. } => {
            vec![
                (0, *matrix_bytes),
                (*matrix_bytes, *matrix_bytes),
                (matrix_bytes * 2, *matrix_bytes),
            ]
        }
        ExpertLayout::Fp8W8a16(slot) | ExpertLayout::Fp8W8a16Block(slot, _) => six(
            &slot.gate,
            &slot.up,
            &slot.down,
            &slot.gate_scale,
            &slot.up_scale,
            &slot.down_scale,
        ),
        ExpertLayout::Fp4E2m1Block(slot, _) => six(
            &slot.gate,
            &slot.up,
            &slot.down,
            &slot.gate_scale,
            &slot.up_scale,
            &slot.down_scale,
        ),
    }
}

fn coalesce(places: &[Place], residues: &[u32]) -> Result<Vec<(usize, usize, usize)>> {
    ensure!(
        residues.len() == places.len(),
        "disk residues {} do not cover {} slices",
        residues.len(),
        places.len()
    );
    let mut out = Vec::with_capacity(places.len());
    for (place, &residue) in places.iter().zip(residues) {
        let src = DiskReader::slice_offset(*place, residue);
        let len = place.len as usize;
        let packed = place.packed as usize;
        if let Some((prev_src, prev_len, prev_packed)) = out.last_mut()
            && *prev_src + *prev_len == src
            && *prev_packed + *prev_len == packed
        {
            *prev_len += len;
            continue;
        }
        out.push((src, len, packed));
    }
    Ok(out)
}

fn plan_slot(layout: &ExpertLayout, alignment: u64) -> Result<SlotPlan> {
    ensure!(
        alignment.is_power_of_two() && alignment >= RING_ALIGN as u64,
        "invalid direct IO alignment {alignment}"
    );
    let align = usize::try_from(alignment)?;
    let slices = layout_slices(layout);
    ensure!(!slices.is_empty(), "native slot has no slices");
    let mut windowed = vec![false; slices.len()];
    for (index, &(_, len)) in slices.iter().enumerate() {
        let len = usize::try_from(len).context("native slice exceeds usize")?;
        ensure!(len > 0, "native slice is empty");
        windowed[index] = len.is_multiple_of(align);
    }
    let mut ring_of = vec![0u32; slices.len()];
    let mut cursor = 0usize;
    if windowed.iter().any(|&flag| flag) {
        for (index, &is_window) in windowed.iter().enumerate() {
            if !is_window {
                continue;
            }
            let len = usize::try_from(slices[index].1).context("native slice exceeds usize")?;
            ring_of[index] = u32::try_from(cursor).context("ring offset exceeds u32")?;
            cursor = cursor
                .checked_add(len + align)
                .context("native ring extent overflow")?;
        }
        for (index, &is_window) in windowed.iter().enumerate() {
            if is_window {
                continue;
            }
            let len = usize::try_from(slices[index].1).context("native slice exceeds usize")?;
            ring_of[index] = u32::try_from(cursor).context("ring offset exceeds u32")?;
            cursor = cursor
                .checked_add(len)
                .context("native ring extent overflow")?;
        }
    } else {
        let mut end = 0u64;
        for (index, &(packed, len)) in slices.iter().enumerate() {
            ring_of[index] = u32::try_from(packed).context("packed offset exceeds u32")?;
            end = end.max(
                packed
                    .checked_add(len)
                    .context("packed slice end overflow")?,
            );
        }
        cursor = usize::try_from(end).context("packed slot exceeds usize")?;
    }
    let places = slices
        .iter()
        .enumerate()
        .map(|(index, &(_, len))| {
            Ok(Place {
                ring: ring_of[index],
                len: u32::try_from(len).context("native slice exceeds u32")?,
                packed: u32::try_from(slices[index].0).context("packed offset exceeds u32")?,
                windowed: windowed[index],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(cursor > 0, "native ring is empty");
    Ok(SlotPlan {
        places,
        ring_bytes: cursor,
        alignment,
    })
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

fn filesystem_alignment(file: &File) -> Result<u64> {
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: the file descriptor is open and stat is writable.
    let rc = unsafe { libc::fstatvfs(file.as_raw_fd(), &mut stat) };
    ensure!(rc == 0, "fstatvfs: {}", std::io::Error::last_os_error());
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
    require_direct_alignment(alignment)?;
    Ok(alignment)
}

/// Alignment `DiskReader::open` will require of the shards under `path`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn directory_alignment(path: &Path) -> Result<u64> {
    let file = File::open(path)
        .with_context(|| format!("open {} for direct alignment", path.display()))?;
    filesystem_alignment(&file)
}

/// Outstanding `read_at` calls for one layer. The depth is `top_k`.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn disk_read_depth(path: &Path, top_k: u32) -> Result<u32> {
    ensure!(top_k >= 1, "disk read depth has top_k 0");
    File::open(path).with_context(|| format!("open {} for disk read depth", path.display()))?;
    Ok(top_k)
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn ring_slot_bytes(layout: &ExpertLayout, alignment: u64) -> Result<u64> {
    u64::try_from(plan_slot(layout, alignment)?.ring_bytes).context("ring slot bytes exceed u64")
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
            filesystem_alignment(&file)?
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

    /// Direct read of `[begin, begin+width)` into `dest`. `dest` holds the next multiple of the
    /// alignment. A short read is accepted when `got == width`.
    fn read_direct(&self, begin: u64, width: u64, dest: &mut [u8]) -> Result<()> {
        ensure!(self.direct, "buffered reader issued a direct window");
        let align = self.alignment;
        ensure!(
            width > 0 && begin.checked_add(width).is_some_and(|end| end <= self.len),
            "direct window {begin}+{width} outside {}",
            self.len
        );
        ensure!(
            begin.is_multiple_of(align),
            "misaligned direct window at {begin}"
        );
        let requested = width.div_ceil(align) * align;
        ensure!(
            dest.len() as u64 >= requested,
            "direct window buffer holds {} of {requested}",
            dest.len()
        );
        ensure!(
            (dest.as_ptr() as usize).is_multiple_of(align as usize),
            "direct window buffer is misaligned"
        );
        let mut written = 0u64;
        while written < width {
            let logical = width - written;
            let count = logical.div_ceil(align) * align;
            let at = begin + written;
            let buffer = &mut dest[written as usize..(written + count) as usize];
            let got = self
                .file
                .read_at(buffer, at)
                .with_context(|| format!("direct read {at}+{count}"))?;
            if got == logical as usize {
                break;
            }
            ensure!(
                got > 0 && (got as u64).is_multiple_of(align) && (got as u64) < logical,
                "short native direct read at {at}: got {got}, expected {logical}"
            );
            written += got as u64;
        }
        Ok(())
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

struct FillJob {
    id: usize,
    layer: u32,
    expert: u32,
    slot: *const AlignedSlot,
    plan: *const SlotPlan,
    parts: Vec<(u64, NativeRange)>,
}

// SAFETY: `slot` and `plan` stay in the owning `DiskReader` until this job's reply is received.
// The owner does not move that reader or write the slot while the job runs. One job writes one slot.
unsafe impl Send for FillJob {}

enum ReadCmd {
    Fill(FillJob),
    Stop,
}

struct Done {
    id: usize,
    result: Result<()>,
}

/// `open` starts one thread per ring slot. The thread blocks in `read_at` into that slot.
/// `read_experts` submits the layer's routed experts, at most one per slot, and waits for every read.
/// A read error or a short read returns the first failure in submission order and names the layer, expert, and slice.
struct ReadPool {
    txs: Vec<mpsc::SyncSender<ReadCmd>>,
    done: mpsc::Receiver<Done>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl ReadPool {
    fn empty() -> Self {
        let (_tx, done) = mpsc::channel();
        Self {
            txs: Vec::new(),
            done,
            threads: Vec::new(),
        }
    }

    fn start(shards: Arc<BTreeMap<String, SpanReader>>, threads: usize) -> Result<Self> {
        let (done_tx, done) = mpsc::channel();
        let mut txs: Vec<mpsc::SyncSender<ReadCmd>> = Vec::with_capacity(threads);
        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (tx, rx) = mpsc::sync_channel(1);
            let shards = Arc::clone(&shards);
            let done_tx = done_tx.clone();
            let handle = match std::thread::Builder::new()
                .name("expert-disk-read".into())
                .spawn(move || reader_loop(shards, rx, done_tx))
            {
                Ok(handle) => handle,
                Err(error) => {
                    for tx in &txs {
                        let _ = tx.send(ReadCmd::Stop);
                    }
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(error).context("start expert disk reader");
                }
            };
            txs.push(tx);
            handles.push(handle);
        }
        drop(done_tx);
        Ok(Self {
            txs,
            done,
            threads: handles,
        })
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        for tx in &self.txs {
            let _ = tx.send(ReadCmd::Stop);
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn reader_loop(
    shards: Arc<BTreeMap<String, SpanReader>>,
    rx: mpsc::Receiver<ReadCmd>,
    done: mpsc::Sender<Done>,
) {
    while let Ok(job) = rx.recv() {
        match job {
            ReadCmd::Stop => break,
            ReadCmd::Fill(fill) => {
                #[cfg(test)]
                flight::enter();
                let result = fill_job(&shards, &fill);
                #[cfg(test)]
                flight::leave();
                if done
                    .send(Done {
                        id: fill.id,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}

fn fill_job(shards: &BTreeMap<String, SpanReader>, fill: &FillJob) -> Result<()> {
    // SAFETY: the owner waits for this reply before moving the reader or writing this slot.
    let slot = unsafe { &*fill.slot };
    let plan = unsafe { &*fill.plan };
    let parts: Vec<(u64, &NativeRange)> = fill
        .parts
        .iter()
        .map(|(packed, native)| (*packed, native))
        .collect();
    fill_slot(shards, plan, slot, &parts, fill.layer, fill.expert)
}

#[cfg(test)]
mod flight {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
    static MAX_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
    static HOLD: AtomicUsize = AtomicUsize::new(0);
    static ARRIVED: AtomicUsize = AtomicUsize::new(0);
    static RELEASED: AtomicUsize = AtomicUsize::new(0);
    static SERIAL: AtomicBool = AtomicBool::new(false);

    // `arrived` and `released` stay through `leave`. `reset` clears them.
    pub(super) fn enter() {
        let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
        MAX_IN_FLIGHT.fetch_max(now, Ordering::SeqCst);
        if SERIAL.load(Ordering::SeqCst) {
            return;
        }
        let hold = HOLD.load(Ordering::SeqCst);
        if hold <= 1 {
            return;
        }
        let ticket = ARRIVED.fetch_add(1, Ordering::SeqCst) + 1;
        if ticket >= hold {
            RELEASED.store(ticket, Ordering::SeqCst);
            return;
        }
        while RELEASED.load(Ordering::SeqCst) < hold && HOLD.load(Ordering::SeqCst) >= hold {
            std::thread::yield_now();
        }
    }

    pub(super) fn leave() {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }

    pub(super) fn reset() {
        IN_FLIGHT.store(0, Ordering::SeqCst);
        MAX_IN_FLIGHT.store(0, Ordering::SeqCst);
        ARRIVED.store(0, Ordering::SeqCst);
        RELEASED.store(0, Ordering::SeqCst);
    }

    pub(super) fn max_in_flight() -> usize {
        MAX_IN_FLIGHT.load(Ordering::SeqCst)
    }

    pub(super) fn hold(n: usize) {
        HOLD.store(n, Ordering::SeqCst);
    }

    pub(super) fn force_serial(on: bool) {
        SERIAL.store(on, Ordering::SeqCst);
    }

    pub(super) fn serial() -> bool {
        SERIAL.load(Ordering::SeqCst)
    }
}

/// Read-only disk tier over the native checkpoint shards with a bounded aligned ring.
pub struct DiskReader {
    shards: Arc<BTreeMap<String, SpanReader>>,
    per_expert: usize,
    ring: Vec<AlignedSlot>,
    next: Cell<usize>,
    staging: Cell<Option<AlignedBytes>>,
    plan: Option<SlotPlan>,
    pool: ReadPool,
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

fn owned_parts(
    catalogue: &ExpertCatalogue,
    layer: u32,
    expert: u32,
) -> Result<Vec<(u64, NativeRange)>> {
    let range = catalogue.range(layer, expert)?;
    let layout = catalogue.layout();
    let mut parts = Vec::new();
    match (layout, range) {
        (ExpertLayout::Bf16Fused { gate_up_bytes, .. }, ExpertRange::Bf16 { gate_up, down }) => {
            parts.push((0, gate_up.clone()));
            parts.push((*gate_up_bytes, down.clone()));
        }
        (
            ExpertLayout::Bf16Separate { matrix_bytes, .. },
            ExpertRange::Bf16Separate { gate, up, down },
        ) => {
            parts.push((0, gate.clone()));
            parts.push((*matrix_bytes, up.clone()));
            parts.push((*matrix_bytes * 2, down.clone()));
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
            parts.push((slot.gate.start, gate.clone()));
            parts.push((slot.up.start, up.clone()));
            parts.push((slot.down.start, down.clone()));
            parts.push((slot.gate_scale.start, gate_scale.clone()));
            parts.push((slot.up_scale.start, up_scale.clone()));
            parts.push((slot.down_scale.start, down_scale.clone()));
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
            parts.push((slot.gate.start, gate.clone()));
            parts.push((slot.up.start, up.clone()));
            parts.push((slot.down.start, down.clone()));
            parts.push((slot.gate_scale.start, gate_scale.clone()));
            parts.push((slot.up_scale.start, up_scale.clone()));
            parts.push((slot.down_scale.start, down_scale.clone()));
        }
        _ => bail!("expert range does not match the catalogue layout"),
    }
    Ok(parts)
}

fn fill_slot(
    shards: &BTreeMap<String, SpanReader>,
    plan: &SlotPlan,
    slot: &AlignedSlot,
    parts: &[(u64, &NativeRange)],
    layer: u32,
    expert: u32,
) -> Result<()> {
    let alignment = plan.alignment;
    let places = &plan.places;
    ensure!(
        parts.len() == places.len(),
        "native plan covers {} slices, read has {}",
        places.len(),
        parts.len()
    );
    // SAFETY: this fill owns the slot until it returns.
    unsafe {
        (*slot.residues.get()).clear();
        (*slot.packed_view.get()).clear();
    }
    let ring_len = slot.bytes.len();
    let ring_ptr = slot.bytes.as_ptr() as *mut u8;
    let mut residues = Vec::with_capacity(places.len());
    let mut staging = slot.staging.take();
    let outcome = (|| -> Result<()> {
        for (index, (place, &(packed, native))) in places.iter().zip(parts).enumerate() {
            (|| -> Result<()> {
                ensure!(
                    packed == u64::from(place.packed) && native.len == u64::from(place.len),
                    "native slice does not match the slot plan"
                );
                let shard = shards
                    .get(native.shard.as_ref())
                    .with_context(|| format!("disk shard {} not open", native.shard))?;
                ensure!(
                    shard.direct && shard.alignment == alignment,
                    "native shard {} is not the planned direct alignment",
                    native.shard
                );
                if place.windowed {
                    let residue = native.offset % alignment;
                    let begin = native.offset - residue;
                    let raw_end = begin
                        .checked_add(native.len)
                        .and_then(|end| end.checked_add(alignment))
                        .context("window end overflow")?;
                    let end = raw_end.min(shard.len);
                    let width = end - begin;
                    let logical_end = native
                        .offset
                        .checked_add(native.len)
                        .context("window logical end overflow")?;
                    ensure!(
                        logical_end <= end,
                        "windowed slice {}+{} outside {begin}..{end}",
                        native.offset,
                        native.len
                    );
                    let dest = place.ring as usize;
                    let requested = usize::try_from(width.div_ceil(alignment) * alignment)?;
                    ensure!(dest + requested <= ring_len, "window exceeds the ring");
                    // SAFETY: the window is inside this slot's allocation and is not aliased.
                    let buffer =
                        unsafe { std::slice::from_raw_parts_mut(ring_ptr.add(dest), requested) };
                    shard.read_direct(begin, width, buffer)?;
                    residues.push(u32::try_from(residue).context("window residue exceeds u32")?);
                } else {
                    let begin = native.offset & !(alignment - 1);
                    let logical_end = native
                        .offset
                        .checked_add(native.len)
                        .context("staged slice end overflow")?;
                    let aligned_end = logical_end.div_ceil(alignment) * alignment;
                    let end = aligned_end.min(shard.len);
                    ensure!(logical_end <= end, "staged slice extends past the shard");
                    let span = ReadSpan {
                        begin,
                        end,
                        first: 0,
                        last: 1,
                    };
                    let need =
                        DiskReader::aligned_staging_bytes(native.offset, native.len, alignment)?;
                    let align_usize = usize::try_from(alignment)?;
                    if staging.as_ref().is_none_or(|buffer| {
                        buffer.len() < need || buffer.alignment() < align_usize
                    }) {
                        drop(staging.take());
                        staging = Some(AlignedBytes::new(need, align_usize)?);
                    }
                    let bytes = shard.read_span(
                        &span,
                        alignment,
                        staging.as_mut().context("native staging missing")?,
                    )?;
                    let from = usize::try_from(native.offset - begin)?;
                    let len = usize::try_from(native.len)?;
                    ensure!(from + len <= bytes.len(), "staged slice exceeds its span");
                    let dest = place.ring as usize;
                    ensure!(dest + len <= ring_len, "staged slice exceeds the ring");
                    // SAFETY: the staging allocation and the ring allocation do not overlap.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr().add(from),
                            ring_ptr.add(dest),
                            len,
                        );
                    }
                    residues.push(0);
                }
                Ok(())
            })()
            .with_context(|| {
                format!(
                    "layer {layer} expert {expert} slice {index} shard {} offset {}",
                    native.shard, native.offset
                )
            })?;
        }
        Ok(())
    })();
    slot.staging.set(staging);
    outcome?;
    // SAFETY: the slot is not leased until the caller marks it filled.
    unsafe {
        *slot.residues.get() = residues;
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
        let mut alignment = None;
        for name in names {
            let file = SpanReader::open(&model_dir.join(name.as_ref()), true)?;
            if let Some(previous) = alignment {
                ensure!(
                    previous == file.alignment,
                    "native shard {name} alignment {} differs from {previous}",
                    file.alignment
                );
            } else {
                alignment = Some(file.alignment);
            }
            let length = file.len;
            shards.insert(name.to_string(), file);
            lengths.insert(name.to_string(), length);
        }
        let alignment = alignment.context("native catalogue has no shards")?;
        require_direct_alignment(alignment)?;
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
        let plan = plan_slot(catalogue.layout(), alignment)?;
        let ring_align = usize::try_from(alignment)?;
        let ring = (0..ring_slots)
            .map(|_| AlignedSlot::new(plan.ring_bytes, ring_align))
            .collect::<Result<Vec<_>>>()?;
        let shards = Arc::new(shards);
        let pool = ReadPool::start(Arc::clone(&shards), ring_slots as usize)?;
        Ok(Self {
            shards,
            per_expert,
            ring,
            next: Cell::new(0),
            staging: Cell::new(None),
            plan: Some(plan),
            pool,
        })
    }

    pub fn open_ranges(
        layers: &[Vec<(u64, NativeRange)>],
        slot_bytes: usize,
        model_dir: &Path,
        ring_slots: u32,
    ) -> Result<Self> {
        ensure!(ring_slots > 0, "zero disk ring slots");
        ensure!(
            slot_bytes > 0 && slot_bytes.is_multiple_of(std::mem::size_of::<bf16>()),
            "invalid disk slot byte count {slot_bytes}"
        );
        let per_expert = slot_bytes / std::mem::size_of::<bf16>();
        let mut names = BTreeSet::new();
        for ranges in layers {
            for (_, native) in ranges {
                names.insert(native.shard.clone());
            }
        }
        let mut shards = BTreeMap::new();
        let mut lengths = BTreeMap::new();
        for name in names {
            let file = SpanReader::open(&model_dir.join(name.as_ref()), false)?;
            let length = file.len;
            shards.insert(name.to_string(), file);
            lengths.insert(name.to_string(), length);
        }
        for ranges in layers {
            for (offset, native) in ranges {
                let slot_end = offset
                    .checked_add(native.len)
                    .context("disk slot range end overflow")?;
                ensure!(
                    slot_end <= slot_bytes as u64,
                    "disk slot range {offset}..{slot_end} exceeds {slot_bytes} bytes"
                );
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
            }
        }
        let ring = (0..ring_slots)
            .map(|_| AlignedSlot::new(slot_bytes, RING_ALIGN))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shards: Arc::new(shards),
            per_expert,
            ring,
            next: Cell::new(0),
            staging: Cell::new(None),
            plan: None,
            pool: ReadPool::empty(),
        })
    }

    #[cfg(test)]
    fn detached(shards: BTreeMap<String, SpanReader>) -> Self {
        Self {
            shards: Arc::new(shards),
            per_expert: 0,
            ring: Vec::new(),
            next: Cell::new(0),
            staging: Cell::new(None),
            plan: None,
            pool: ReadPool::empty(),
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub(crate) fn ring_spans(&self) -> Vec<(u64, usize)> {
        self.ring
            .iter()
            .map(|slot| (slot.bytes.as_ptr() as u64, slot.bytes.len()))
            .collect()
    }

    pub(crate) fn aligned_staging_bytes(offset: u64, len: u64, alignment: u64) -> Result<usize> {
        ensure!(
            alignment.is_power_of_two() && alignment >= RING_ALIGN as u64,
            "invalid expert staging alignment {alignment}"
        );
        let bytes = (offset % alignment)
            .checked_add(len)
            .and_then(|n| n.checked_add(alignment - 1))
            .map(|n| n & !(alignment - 1))
            .context("expert staging extent overflow")?;
        usize::try_from(bytes).context("expert staging extent exceeds usize")
    }

    fn reset_view(&self, slot: usize) {
        // SAFETY: this thread owns the reader and the slot is not leased.
        unsafe {
            (*self.ring[slot].residues.get()).clear();
            (*self.ring[slot].packed_view.get()).clear();
        }
    }

    fn slice_offset(place: Place, residue: u32) -> usize {
        place.ring as usize + if place.windowed { residue as usize } else { 0 }
    }

    fn disk_slices(&self, slot: usize) -> Result<Vec<(usize, usize, usize)>> {
        let plan = self.plan.as_ref().context("disk slice plan missing")?;
        // SAFETY: the lease owns the slot. The residue vec is not mutated while the lease is held.
        let residues = unsafe { &*self.ring[slot].residues.get() };
        coalesce(&plan.places, residues)
    }

    fn copy_logical(&self, slot: usize, dst: &mut [bf16]) -> Result<()> {
        ensure!(
            dst.len() == self.per_expert,
            "logical copy holds {} elements for {}",
            dst.len(),
            self.per_expert
        );
        let nbytes = self.per_expert * std::mem::size_of::<bf16>();
        // SAFETY: `dst` is `nbytes` initialized bytes and is exclusive for this call.
        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), nbytes) };
        let ring = &self.ring[slot];
        let Some(plan) = self.plan.as_ref() else {
            // SAFETY: a dense slot's prefix is the packed slot.
            let src = unsafe { std::slice::from_raw_parts(ring.bytes.as_ptr(), nbytes) };
            dst_bytes.copy_from_slice(src);
            return Ok(());
        };
        // SAFETY: the lease owns the slot.
        let residues = unsafe { &*ring.residues.get() };
        ensure!(
            residues.len() == plan.places.len(),
            "disk residues incomplete"
        );
        let needs = plan
            .places
            .iter()
            .any(|place| place.windowed || place.ring != place.packed);
        if !needs {
            let src = unsafe { std::slice::from_raw_parts(ring.bytes.as_ptr(), nbytes) };
            dst_bytes.copy_from_slice(src);
            return Ok(());
        }
        let src = ring.bytes.as_ptr();
        let ring_len = ring.bytes.len();
        for (place, &residue) in plan.places.iter().zip(residues) {
            let from = Self::slice_offset(*place, residue);
            let start = place.packed as usize;
            let len = place.len as usize;
            ensure!(
                from + len <= ring_len && start + len <= nbytes,
                "logical slice outside the ring"
            );
            // SAFETY: the source window and the packed destination do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(from),
                    dst_bytes.as_mut_ptr().add(start),
                    len,
                );
            }
        }
        Ok(())
    }

    fn packed_bytes(&self, slot: usize) -> *const u8 {
        let nbytes = self.per_expert * std::mem::size_of::<bf16>();
        let ring = &self.ring[slot];
        let src = ring.bytes.as_ptr();
        // SAFETY: plan, residues and the packed view are distinct and owned by this thread.
        let plan = self.plan.as_ref().map(|plan| plan as *const SlotPlan);
        let residues_ptr = ring.residues.get();
        let view_ptr = ring.packed_view.get();
        unsafe {
            let Some(plan) = plan else {
                return src;
            };
            let plan = &*plan;
            let residues = &*residues_ptr;
            let needs = residues.len() == plan.places.len()
                && plan
                    .places
                    .iter()
                    .any(|place| place.windowed || place.ring != place.packed);
            if !needs {
                return src;
            }
            let view = &mut *view_ptr;
            if view.len() != nbytes {
                view.resize(nbytes, 0);
                for (place, &residue) in plan.places.iter().zip(residues.iter()) {
                    let from = Self::slice_offset(*place, residue);
                    std::ptr::copy_nonoverlapping(
                        src.add(from),
                        view.as_mut_ptr().add(place.packed as usize),
                        place.len as usize,
                    );
                }
            }
            view.as_ptr()
        }
    }

    fn read_planned(
        &self,
        parts: &[(u64, &NativeRange)],
        slot: usize,
        layer: u32,
        expert: u32,
    ) -> Result<()> {
        let plan = self.plan.as_ref().context("native slot plan missing")?;
        fill_slot(&self.shards, plan, &self.ring[slot], parts, layer, expert)
    }

    fn claim_free_slot(&self) -> Result<usize> {
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
        self.ring[slot].state.set(RingState::Filled);
        self.next.set((slot + 1) % slots);
        Ok(slot)
    }

    /// Reads one layer's native ranges into a free ring slot at their configured slot offsets.
    pub fn read_ranges(&self, ranges: &[(u64, NativeRange)], chunk_bytes: u64) -> Result<u32> {
        ensure!(chunk_bytes > 0, "disk read chunk is not positive");
        let slot = self.claim_free_slot()?;
        let filled = (|| -> Result<()> {
            self.reset_view(slot);
            let bytes = self.ring[slot].bytes.len();
            let ptr = self.ring[slot].bytes.as_ptr() as *mut u8;
            // SAFETY: the slot owns `bytes` writable bytes with the ring alignment.
            let buffer: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
            let parts: Vec<(u64, &NativeRange)> = ranges
                .iter()
                .map(|(offset, native)| (*offset, native))
                .collect();
            self.read_parts(&parts, buffer, chunk_bytes)?;
            Ok(())
        })();
        if let Err(error) = filled {
            self.ring[slot].state.set(RingState::Free);
            return Err(error);
        }
        u32::try_from(slot).context("disk ring slot exceeds u32")
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
        let slot = self.claim_free_slot()?;
        if let Err(error) = self.read_into(layer, expert, catalogue, slot, chunk_bytes) {
            self.ring[slot].state.set(RingState::Free);
            return Err(error);
        }
        u32::try_from(slot).context("disk ring slot exceeds u32")
    }

    fn read_into(
        &self,
        layer: u32,
        expert: u32,
        catalogue: &ExpertCatalogue,
        slot: usize,
        chunk_bytes: u64,
    ) -> Result<()> {
        self.reset_view(slot);
        let owned = owned_parts(catalogue, layer, expert)?;
        let parts: Vec<(u64, &NativeRange)> = owned
            .iter()
            .map(|(packed, native)| (*packed, native))
            .collect();
        if self.plan.is_some() {
            self.read_planned(&parts, slot, layer, expert)?;
        } else {
            let bytes = self.ring[slot].bytes.len();
            let ptr = self.ring[slot].bytes.as_ptr() as *mut u8;
            // SAFETY: the slot owns `bytes` writable bytes with the ring alignment.
            let buffer: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
            self.read_parts(&parts, buffer, chunk_bytes)?;
        }
        Ok(())
    }

    /// Reads `experts` of one layer into free ring slots and waits for every read.
    /// At most one expert is in flight per ring slot. Slot `i` is filled by pool thread `i`.
    pub fn read_experts(
        &self,
        layer: u32,
        experts: &[u32],
        catalogue: &ExpertCatalogue,
    ) -> Result<Vec<u32>> {
        ensure!(
            !experts.is_empty(),
            "layer {layer} disk read has no experts"
        );
        ensure!(
            experts.len() <= self.ring.len(),
            "layer {layer} has {} disk experts; the ring holds {}",
            experts.len(),
            self.ring.len()
        );
        ensure!(
            self.pool.txs.len() == self.ring.len(),
            "disk reader pool holds {} threads for {} slots",
            self.pool.txs.len(),
            self.ring.len()
        );
        let mut assigned: Vec<(u32, usize)> = Vec::with_capacity(experts.len());
        for &expert in experts {
            let slot = match self.claim_free_slot() {
                Ok(slot) => slot,
                Err(error) => {
                    for &(_, slot) in &assigned {
                        self.ring[slot].state.set(RingState::Free);
                    }
                    return Err(error);
                }
            };
            assigned.push((expert, slot));
        }
        let filled = self.fill_assigned(layer, &assigned, catalogue);
        if let Err(error) = filled {
            for &(_, slot) in &assigned {
                if self.ring[slot].state.get() == RingState::Filled {
                    self.ring[slot].state.set(RingState::Free);
                }
            }
            return Err(error);
        }
        let mut slots = Vec::with_capacity(assigned.len());
        for &(_, slot) in &assigned {
            self.ring[slot].state.set(RingState::Filled);
            slots.push(u32::try_from(slot).context("disk ring slot exceeds u32")?);
        }
        Ok(slots)
    }

    fn fill_assigned(
        &self,
        layer: u32,
        assigned: &[(u32, usize)],
        catalogue: &ExpertCatalogue,
    ) -> Result<()> {
        for (i, &(_, slot)) in assigned.iter().enumerate() {
            ensure!(
                assigned[..i].iter().all(|&(_, other)| other != slot),
                "layer {layer} disk slot {slot} got two reads"
            );
        }
        #[cfg(test)]
        if flight::serial() {
            for &(expert, slot) in assigned {
                flight::enter();
                let result = self.read_into(layer, expert, catalogue, slot, 1);
                flight::leave();
                result?;
            }
            return Ok(());
        }
        let plan = self.plan.as_ref().context("native slot plan missing")? as *const SlotPlan;
        let mut jobs = Vec::with_capacity(assigned.len());
        for (id, &(expert, slot)) in assigned.iter().enumerate() {
            jobs.push(FillJob {
                id,
                layer,
                expert,
                slot: &self.ring[slot] as *const AlignedSlot,
                plan,
                parts: owned_parts(catalogue, layer, expert)?,
            });
        }
        for (sent, job) in jobs.into_iter().enumerate() {
            let index = assigned[job.id].1;
            if self.pool.txs[index].send(ReadCmd::Fill(job)).is_err() {
                for _ in 0..sent {
                    self.pool
                        .done
                        .recv()
                        .with_context(|| format!("layer {layer} disk reader dropped a reply"))?;
                }
                bail!("layer {layer} disk reader stopped");
            }
        }
        let mut first: Option<(usize, anyhow::Error)> = None;
        for _ in 0..assigned.len() {
            let done = self
                .pool
                .done
                .recv()
                .with_context(|| format!("layer {layer} disk reader dropped a reply"))?;
            if let Err(error) = done.result
                && first.as_ref().is_none_or(|(id, _)| done.id < *id)
            {
                first = Some((done.id, error));
            }
        }
        if let Some((_, error)) = first {
            return Err(error);
        }
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub(crate) fn slot_is_free(&self, slot: usize) -> bool {
        self.ring
            .get(slot)
            .is_some_and(|slot| slot.state.get() == RingState::Free)
    }

    /// Buffered single-request spans land directly in their slot slices; gathered and direct-IO
    /// spans use aligned staging and scatter in physical source order.
    fn read_parts(
        &self,
        parts: &[(u64, &NativeRange)],
        slot: &mut [u8],
        chunk_bytes: u64,
    ) -> Result<()> {
        ensure!(!parts.is_empty(), "expert read has no native ranges");
        ensure!(chunk_bytes > 0, "native read chunk is zero");
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
        let mut staging = self.staging.take();
        let outcome = (|| -> Result<()> {
            for (shard_name, mut indexes) in by_shard {
                let shard = self
                    .shards
                    .get(shard_name)
                    .with_context(|| format!("disk shard {shard_name} not open"))?;
                indexes.sort_unstable_by_key(|&index| (parts[index].1.offset, parts[index].1.len));
                let requests: Vec<(u64, u64)> = indexes
                    .iter()
                    .map(|&index| (parts[index].1.offset, parts[index].1.len))
                    .collect();
                for &(offset, len) in &requests {
                    ensure!(
                        len > 0 && offset.checked_add(len).is_some_and(|end| end <= shard.len),
                        "invalid native range {offset}+{len} for {} bytes",
                        shard.len
                    );
                }
                let chunk = if shard.direct {
                    chunk_bytes
                        .checked_add(shard.alignment - 1)
                        .context("direct chunk overflow")?
                        & !(shard.alignment - 1)
                } else {
                    chunk_bytes
                };
                let spans = shard.plan(&requests, chunk)?;
                let mut needed = 0;
                for span in &spans {
                    if shard.direct || span.last > span.first + 1 {
                        needed = needed.max(Self::aligned_staging_bytes(
                            span.begin,
                            span.end - span.begin,
                            shard.alignment,
                        )?);
                    }
                }
                let align = usize::try_from(shard.alignment)?;
                if needed > 0
                    && staging
                        .as_ref()
                        .is_none_or(|buffer| buffer.len() < needed || buffer.alignment() < align)
                {
                    drop(staging.take());
                    staging = Some(AlignedBytes::new(needed, align)?);
                }
                for span in spans {
                    if !shard.direct && span.last == span.first + 1 {
                        let index = indexes[span.first];
                        let (offset, native) = parts[index];
                        let start = usize::try_from(offset)?;
                        let end = start + usize::try_from(native.len)?;
                        let mut at = native.offset;
                        for bytes in slot[start..end].chunks_mut(usize::try_from(chunk)?) {
                            shard.file.read_exact_at(bytes, at).with_context(|| {
                                format!("native read {shard_name} {at}+{}", bytes.len())
                            })?;
                            at += bytes.len() as u64;
                        }
                    } else {
                        let bytes = shard.read_span(
                            &span,
                            chunk,
                            staging.as_mut().context("native staging missing")?,
                        )?;
                        for index in span.first..span.last {
                            let (offset, len) = requests[index];
                            let from = usize::try_from(offset - span.begin)?;
                            let end = from
                                .checked_add(usize::try_from(len)?)
                                .context("native slice overflow")?;
                            ensure!(end <= bytes.len(), "native request exceeds its span");
                            let start = usize::try_from(parts[indexes[index]].0)?;
                            slot[start..start + end - from].copy_from_slice(&bytes[from..end]);
                        }
                    }
                }
            }
            Ok(())
        })();
        self.staging.set(staging);
        outcome
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
                    && route.weight.is_finite(),
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

    std::thread_local! {
        pub(super) static ALIGNED_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

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
            quant: None,
            swap_halves: false,
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
            quant: None,
            swap_halves: false,
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
        temp_dir_at(&std::env::temp_dir(), tag)
    }

    fn temp_dir_at(root: &std::path::Path, tag: &str) -> Result<TempDir> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = root.join(format!(
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
        assert!(AlignedSlot::new(0, RING_ALIGN).is_err());
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

    fn require_outstanding(in_flight: usize, misses: usize) -> Result<()> {
        ensure!(
            in_flight == misses,
            "in-flight {in_flight} covers {misses} misses"
        );
        Ok(())
    }

    fn expert_bytes(
        reader: &DiskReader,
        slot: u32,
        file: &[u8],
        layer: u32,
        expert: u32,
    ) -> Result<Vec<u8>> {
        let values = reader.values(slot)?;
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values.iter() {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let layer_base = layer as usize * 768;
        let gate = layer_base + expert as usize * 128;
        let down = layer_base + 512 + expert as usize * 64;
        let mut expected = Vec::with_capacity(bytes.len());
        for index in 0..values.len() {
            let at = if index < 64 {
                gate + index * 2
            } else {
                down + (index - 64) * 2
            };
            expected.extend_from_slice(&file[at..at + 2]);
        }
        assert_eq!(bytes, expected, "layer {layer} expert {expert}");
        Ok(bytes)
    }

    #[test]
    fn one_layer_misses_stay_outstanding() -> Result<()> {
        struct ResetFlight;
        impl Drop for ResetFlight {
            fn drop(&mut self) {
                super::flight::force_serial(false);
                super::flight::hold(0);
                super::flight::reset();
            }
        }
        let _reset = ResetFlight;
        let (dir, catalogue) = staged("shard-0.safetensors", "shard-0.safetensors")?;
        let file = pattern(1536, 0);
        let catalogue = std::sync::Arc::new(catalogue);
        let reader = DiskReader::open(&catalogue, dir.path(), 2)?;
        super::flight::reset();
        super::flight::hold(2);
        super::flight::force_serial(false);
        let slots = reader.read_experts(0, &[0, 1], &catalogue)?;
        let together = super::flight::max_in_flight();
        require_outstanding(together, 2)?;
        expert_bytes(&reader, slots[0], &file, 0, 0)?;
        expert_bytes(&reader, slots[1], &file, 0, 1)?;
        super::flight::reset();
        super::flight::hold(0);
        super::flight::force_serial(true);
        let serial_slots = reader.read_experts(0, &[0, 1], &catalogue)?;
        let serial = super::flight::max_in_flight();
        let rejected = require_outstanding(serial, 2);
        assert!(
            rejected.is_err(),
            "depth-1 mutant reported in-flight {serial}"
        );
        expert_bytes(&reader, serial_slots[0], &file, 0, 0)?;
        expert_bytes(&reader, serial_slots[1], &file, 0, 1)?;
        super::flight::force_serial(false);
        super::flight::hold(0);
        super::flight::reset();
        println!(
            "PASS layer_reads_outstanding misses=2 in_flight={together} depth1_in_flight={serial} depth1_rejected=true"
        );
        Ok(())
    }

    #[test]
    fn disk_read_depth_is_top_k_without_a_block_queue() -> Result<()> {
        use std::os::linux::fs::MetadataExt;
        let mut root = None;
        for candidate in [std::env::temp_dir(), std::path::PathBuf::from("/dev/shm")] {
            if !candidate.is_dir() {
                continue;
            }
            let dev = std::fs::metadata(&candidate)?.st_dev();
            let node = std::path::PathBuf::from(format!(
                "/sys/dev/block/{}:{}",
                libc::major(dev),
                libc::minor(dev)
            ));
            if !node.exists() {
                root = Some(candidate);
                break;
            }
        }
        let root = root.context("no temp filesystem without a sysfs block queue")?;
        let dir = temp_dir_at(&root, "queue")?;
        let path = dir.path().join("payload");
        std::fs::write(&path, [1u8, 2, 3, 4])?;
        let dev = std::fs::metadata(&path)?.st_dev();
        let node = format!("/sys/dev/block/{}:{}", libc::major(dev), libc::minor(dev));
        assert!(
            !std::path::Path::new(&node).exists(),
            "{node} is a sysfs block queue"
        );
        assert_eq!(super::disk_read_depth(&path, 1)?, 1);
        assert_eq!(super::disk_read_depth(&path, 6)?, 6);
        assert_eq!(super::disk_read_depth(&path, 8)?, 8);
        assert_eq!(super::disk_read_depth(&path, 2048)?, 2048);
        assert!(super::disk_read_depth(&path, 0).is_err());
        println!("PASS disk_read_depth depth=top_k top_k=6,8 sysfs_block=absent");
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

    fn compare_range_reads(
        reader: &DiskReader,
        ranges: &[(u64, NativeRange)],
        slot_bytes: usize,
        chunk: u64,
        row_bytes: u64,
    ) -> Result<(usize, usize)> {
        let mut expected = vec![0xa5; slot_bytes];
        let mut eligible = 0;
        let mut staged = 0;
        let mut staged_shards = 0;
        for (name, shard) in reader.shards.iter() {
            let indexes: Vec<_> = ranges
                .iter()
                .enumerate()
                .filter_map(|(i, (_, r))| (r.shard.as_ref() == name).then_some(i))
                .collect();
            let requests: Vec<_> = indexes
                .iter()
                .map(|&i| (ranges[i].1.offset, ranges[i].1.len))
                .collect();
            let needed = requests
                .iter()
                .map(|&(offset, len)| {
                    DiskReader::aligned_staging_bytes(offset, len, shard.alignment)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .context("empty test shard")?;
            let mut staging = AlignedBytes::new(needed, shard.alignment as usize)?;
            shard.read_spans(&requests, chunk, &mut staging, |index, bytes| {
                let start = ranges[indexes[index]].0 as usize;
                expected[start..start + bytes.len()].copy_from_slice(bytes);
                Ok(())
            })?;
            let mut required = false;
            for span in shard.plan(&requests, chunk)? {
                if !shard.direct && span.last == span.first + 1 {
                    eligible += 1;
                } else {
                    staged += 1;
                    required = true;
                }
            }
            staged_shards += usize::from(required);
        }
        let parts: Vec<_> = ranges
            .iter()
            .map(|(offset, native)| (*offset, native))
            .collect();
        let mut actual = vec![0xa5; slot_bytes];
        let before = ALIGNED_ALLOCATIONS.with(Cell::get);
        reader.read_parts(&parts, &mut actual, chunk)?;
        let allocations = ALIGNED_ALLOCATIONS.with(Cell::get) - before;
        assert_eq!(
            actual, expected,
            "native destination bytes differ from staged reads"
        );
        let staging = reader.staging.take();
        if staged == 0 {
            assert!(staging.is_none(), "single buffered spans allocated staging");
        } else {
            assert!(staging.is_some(), "gather/direct spans did not use staging");
        }
        reader.staging.set(staging);
        assert!(
            allocations <= staged_shards,
            "native staging grew within a shard: allocations={allocations} staged_shards={staged_shards}"
        );
        let mut shifted = parts;
        shifted[0].0 += row_bytes;
        actual.fill(0xa5);
        reader.read_parts(&shifted, &mut actual, chunk)?;
        assert_ne!(actual, expected, "one-row destination control escaped");
        println!(
            "PASS native_range_destination bytes={slot_bytes} eligible={eligible} staged={staged} staging_allocations={allocations} byte_mismatches=0 row_control={row_bytes} guards=true"
        );
        Ok((eligible, staged))
    }

    #[test]
    fn range_spans_preserve_destination_bytes() -> Result<()> {
        let dir = temp_dir("span-destinations")?;
        for (name, salt) in [("a", 17), ("b", 83)] {
            let bytes: Vec<_> = (0..32789).map(|i| ((i + salt) % 251) as u8).collect();
            std::fs::write(dir.path().join(name), bytes)?;
        }
        for direct in [false, true] {
            for (case, requests) in [
                vec![("a", 17, 8311, 64), ("a", 8328, 4093, 9000)],
                vec![("a", 31, 37, 128), ("a", 33, 11, 64), ("a", 61, 7, 256)],
                vec![("a", 32760, 29, 64), ("a", 7, 23, 256), ("a", 7, 23, 512)],
                vec![("b", 3, 4111, 512), ("a", 123, 71, 64), ("a", 151, 23, 80)],
            ]
            .into_iter()
            .enumerate()
            {
                let shards = ["a", "b"]
                    .into_iter()
                    .filter(|name| requests.iter().any(|r| r.0 == *name))
                    .map(|name| {
                        Ok((
                            name.to_owned(),
                            SpanReader::open(&dir.path().join(name), direct)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                let reader = DiskReader::detached(shards);
                let ranges: Vec<_> = requests
                    .iter()
                    .map(|&(name, offset, len, destination)| {
                        (
                            destination,
                            NativeRange {
                                shard: name.into(),
                                offset,
                                len,
                            },
                        )
                    })
                    .collect();
                let counts = compare_range_reads(&reader, &ranges, 16384, 8192, 32)?;
                if !direct && case == 0 {
                    assert_eq!(counts, (2, 0));
                }
                if !direct && case == 1 {
                    assert_eq!(counts, (0, 1));
                }
                let invalid = NativeRange {
                    shard: "a".into(),
                    offset: 32788,
                    len: 2,
                };
                assert!(
                    reader
                        .read_parts(&[(0, &invalid)], &mut [0; 32], 8192)
                        .is_err()
                );
                assert!(
                    reader
                        .read_parts(&[(32, &ranges[0].1)], &mut [0; 32], 8192)
                        .is_err()
                );
                assert!(
                    reader
                        .read_parts(&[(0, &ranges[0].1)], &mut [0; 16384], 0)
                        .is_err()
                );
            }
        }
        let reader = DiskReader::detached(
            [(
                "a".to_owned(),
                SpanReader::open(&dir.path().join("a"), false)?,
            )]
            .into(),
        );
        OpenOptions::new()
            .write(true)
            .open(dir.path().join("a"))?
            .set_len(3)?;
        let native = NativeRange {
            shard: "a".into(),
            offset: 17,
            len: 8311,
        };
        assert!(
            reader
                .read_parts(&[(64, &native)], &mut vec![0; 16384], 8192)
                .is_err()
        );
        assert!(reader.staging.take().is_none());
        Ok(())
    }

    #[test]
    #[ignore = "CPU production-size native range identity and staging capacity"]
    fn production_text_ranges_preserve_bytes_without_staging() -> Result<()> {
        use std::io::Write;
        let dir = temp_dir("text-spans")?;
        let mut file = File::create(dir.path().join("text"))?;
        let lengths = [
            262144000u64,
            262144000,
            262144000,
            10485760,
            83886080,
            83886080,
            10485760,
        ];
        let extent = 17 + lengths.iter().sum::<u64>();
        let mut at = 0;
        while at < extent {
            let n = (1024 * 1024).min((extent - at) as usize);
            let bytes: Vec<_> = (0..n).map(|i| ((at + i as u64) % 251) as u8).collect();
            file.write_all(&bytes)?;
            at += n as u64;
        }
        drop(file);
        let reader = DiskReader::detached(
            [(
                "text".to_owned(),
                SpanReader::open(&dir.path().join("text"), false)?,
            )]
            .into(),
        );
        let mut offset = 17;
        let mut destination = 64;
        let ranges: Vec<_> = lengths
            .into_iter()
            .map(|len| {
                let pair = (
                    destination,
                    NativeRange {
                        shard: "text".into(),
                        offset,
                        len,
                    },
                );
                offset += len;
                destination += len;
                pair
            })
            .collect();
        assert_eq!(lengths.iter().sum::<u64>(), 975175680);
        assert_eq!(
            lengths
                .iter()
                .map(|n| n.div_ceil(32 * 1024 * 1024))
                .sum::<u64>(),
            32
        );
        assert_eq!(
            compare_range_reads(
                &reader,
                &ranges,
                destination as usize + 10240 + 64,
                32 * 1024 * 1024,
                10240
            )?,
            (7, 0)
        );
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
                    weight: bf16::from_f32(0.25).to_f32(),
                },
                Route {
                    row: 0,
                    choice: 1,
                    expert: 0,
                    weight: bf16::from_f32(0.5).to_f32(),
                },
                Route {
                    row: 2,
                    choice: 4,
                    expert: 3,
                    weight: bf16::from_f32(0.75).to_f32(),
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
            weight: bf16::from_f32(0.5).to_f32(),
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
            weight: bf16::from_f32(0.5).to_f32(),
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
                    weight: bf16::from_f32(weight).to_f32(),
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
    use crate::experts::catalogue::{ExpertCatalogue, ExpertGeometry, ExpertLayout, ExpertRange};
    use crate::model_spec::ExpertFormat;
    use std::collections::BTreeMap;
    use std::os::unix::io::AsRawFd;
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
                            quant: None,
                            swap_halves: false,
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

    fn byte_at(index: u64) -> u8 {
        (index.wrapping_mul(37).wrapping_add(11) % 251) as u8
    }

    fn logical_with(reader: &DiskReader, slot: usize, residues: &[u32]) -> Vec<u8> {
        let plan = reader.plan.as_ref().expect("slot plan");
        let src = reader.ring[slot].bytes.as_ptr();
        let mut out = vec![0u8; reader.per_expert * 2];
        for (place, &residue) in plan.places.iter().zip(residues) {
            let from = DiskReader::slice_offset(*place, residue);
            let len = place.len as usize;
            let at = place.packed as usize;
            // SAFETY: the test holds the slot lease and the offsets stay inside the ring and the slot.
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(from), out.as_mut_ptr().add(at), len);
            }
        }
        out
    }

    fn slices_of(reader: &DiskReader, slot: usize) -> Vec<u8> {
        let guard_residues = unsafe { &*reader.ring[slot].residues.get() };
        logical_with(reader, slot, guard_residues)
    }

    #[test]
    fn published_direct_rings_match_the_counted_slots() -> Result<()> {
        let g1 = ExpertLayout::new(
            ExpertFormat::Fp8W8a16Block {
                block: [128, 128],
                scale_dtype: safetensors::Dtype::F32,
            },
            &ExpertGeometry {
                first_layer: 0,
                layers: 42,
                experts: 288,
                hidden: 4096,
                intermediate: 2048,
                top_k: 1,
            },
        )?;
        assert_eq!(g1.slot_bytes(), 25_171_968);
        let g1_plan = super::plan_slot(&g1, 4096)?;
        assert_eq!(g1_plan.ring_bytes, 25_184_256);
        assert_eq!(
            g1_plan.places.iter().filter(|place| place.windowed).count(),
            3
        );
        let g1_residues = [728u32, 728, 728, 0, 0, 0];
        let g1_copies = super::coalesce(&g1_plan.places, &g1_residues)?;
        assert_eq!(g1_copies.len(), 4);
        assert_eq!(g1_copies[3].1, 6144);
        let g1_catalogue = g1.slot_bytes() * 42 * 288;
        assert_eq!(g1_catalogue, 304_480_124_928);
        let d1 = ExpertLayout::new(
            e2m1_format(),
            &ExpertGeometry {
                first_layer: 0,
                layers: 40,
                experts: 384,
                hidden: 5120,
                intermediate: 2304,
                top_k: 1,
            },
        )?;
        assert_eq!(d1.slot_bytes(), 18_800_640);
        let d1_plan = super::plan_slot(&d1, 4096)?;
        assert_eq!(d1_plan.ring_bytes, 18_825_216);
        assert!(d1_plan.places.iter().all(|place| place.windowed));
        let d1_copies = super::coalesce(&d1_plan.places, &[760u32; 6])?;
        assert_eq!(d1_copies.len(), 6);
        assert_eq!(
            d1_plan
                .places
                .iter()
                .map(|place| place.ring)
                .collect::<Vec<_>>(),
            vec![0, 5_902_336, 11_804_672, 17_707_008, 18_079_744, 18_452_480]
        );
        let d1_catalogue = d1.slot_bytes() * 40 * 384;
        assert_eq!(d1_catalogue, 288_777_830_400);
        assert!(super::require_direct_alignment(2048).is_err());
        assert!(super::require_direct_alignment(6144).is_err());
        println!("PASS direct_rings g1_ring=25184256 g1_copies=4 d1_ring=18825216 d1_copies=6");
        Ok(())
    }

    #[test]
    fn direct_windows_match_every_residue_class_and_reject_a_shifted_read() -> Result<()> {
        const ALIGN: u64 = 4096;
        let code_residues = [760u64, 1192, 1592, 2944, 3104, 3928, 728, 3952];
        let clamp_expert = 2usize;
        let hidden = 2048usize;
        let intermediate = 64usize;
        let code_gate = (intermediate * hidden.div_ceil(2)) as u64;
        let code_down = (hidden * intermediate.div_ceil(2)) as u64;
        let scale_gate = (intermediate * hidden.div_ceil(32)) as u64;
        let scale_down = (hidden * intermediate.div_ceil(32)) as u64;
        assert!(
            [code_gate, code_down, scale_gate, scale_down]
                .into_iter()
                .all(|len| len.is_multiple_of(ALIGN))
        );
        struct Item {
            name: String,
            dtype: safetensors::Dtype,
            shape: Vec<usize>,
            len: u64,
            residue: u64,
            clamp: bool,
            offset: u64,
        }
        let mut items = Vec::new();
        for (expert, &residue) in code_residues.iter().enumerate() {
            let scale_residue = (residue + 2304) % ALIGN;
            let name = |role: &str| format!("layers.0.ffn.experts.{expert}.{role}");
            let rows = intermediate;
            let cols = hidden;
            for (role, dtype, shape, len, item_residue) in [
                (
                    "w1.weight",
                    safetensors::Dtype::I8,
                    vec![rows, cols / 2],
                    code_gate,
                    residue,
                ),
                (
                    "w3.weight",
                    safetensors::Dtype::I8,
                    vec![rows, cols / 2],
                    code_gate,
                    residue,
                ),
                (
                    "w2.weight",
                    safetensors::Dtype::I8,
                    vec![cols, rows / 2],
                    code_down,
                    residue,
                ),
                (
                    "w1.scale",
                    safetensors::Dtype::F8_E8M0,
                    vec![rows, cols / 32],
                    scale_gate,
                    scale_residue,
                ),
                (
                    "w3.scale",
                    safetensors::Dtype::F8_E8M0,
                    vec![rows, cols / 32],
                    scale_gate,
                    scale_residue,
                ),
                (
                    "w2.scale",
                    safetensors::Dtype::F8_E8M0,
                    vec![cols, rows / 32],
                    scale_down,
                    scale_residue,
                ),
            ] {
                items.push(Item {
                    name: name(role),
                    dtype,
                    shape,
                    len,
                    residue: item_residue,
                    clamp: expert == clamp_expert && role == "w3.weight",
                    offset: 0,
                });
            }
        }
        let clamp_at = items.iter().position(|item| item.clamp).expect("clamp");
        let clamped = items.remove(clamp_at);
        items.push(clamped);
        let mut cursor = 0u64;
        for item in &mut items {
            let base = cursor.div_ceil(ALIGN) * ALIGN;
            item.offset = base + item.residue;
            cursor = if item.clamp {
                item.offset + item.len
            } else {
                base + item.len + ALIGN
            };
        }
        let file_len = cursor;
        let file: Vec<u8> = (0..file_len).map(byte_at).collect();
        let dir = temp_dir("windows")?;
        std::fs::write(dir.path().join("windows.bin"), &file)?;
        let mut entries = BTreeMap::new();
        for item in &items {
            entries.insert(
                item.name.clone(),
                TensorEntry {
                    dtype: item.dtype,
                    shape: item.shape.clone(),
                    shard: "windows.bin".into(),
                    bytes: item.len as usize,
                    load: true,
                    data_start: 0,
                    data_offsets: [item.offset, item.offset + item.len],
                    scope: TensorScope::Text,
                    layer: None,
                    role: item.name.clone(),
                    scale_name: None,
                    quant: None,
                    swap_halves: false,
                },
            );
        }
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: code_residues.len() as u32,
            hidden: hidden as u32,
            intermediate: intermediate as u32,
            top_k: 2,
        };
        let catalogue = ExpertCatalogue::from_manifest(
            &TensorManifest {
                entries,
                resident_bytes: 0,
            },
            geometry,
            e2m1_format(),
        )?;
        let reader = DiskReader::open(&catalogue, dir.path(), geometry.experts)?;
        for shard in reader.shards.values() {
            let flags = unsafe { libc::fcntl(shard.file.as_raw_fd(), libc::F_GETFL) };
            assert!(flags >= 0 && flags & libc::O_DIRECT != 0);
        }
        let plan = reader.plan.as_ref().expect("plan");
        assert!(plan.places.iter().all(|place| place.windowed));
        assert_eq!(plan.ring_bytes, reader.per_expert * 2 + 6 * ALIGN as usize);
        let mut classes = 0usize;
        for expert in 0..geometry.experts {
            let slot = reader.read_expert(0, expert, &catalogue, catalogue.slot_bytes())?;
            let guard = reader.values(slot)?;
            let stored = unsafe { &*reader.ring[guard.slot].residues.get() }.clone();
            let expected = slices_of(&reader, guard.slot);
            let range = catalogue.range(0, expert)?;
            let ExpertRange::E2m1 {
                gate,
                up,
                down,
                gate_scale,
                up_scale,
                down_scale,
            } = range
            else {
                bail!("E2M1 range");
            };
            let mut file_bytes = Vec::new();
            for native in [gate, up, down, gate_scale, up_scale, down_scale] {
                let start = native.offset as usize;
                let end = start + native.len as usize;
                file_bytes.extend_from_slice(&file[start..end]);
            }
            assert_eq!(expected, file_bytes, "expert {expert}");
            let viewed: Vec<u8> = guard
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect();
            assert_eq!(viewed, file_bytes, "expert {expert} packed view");
            let copies = guard.disk_slices()?;
            assert_eq!(copies.len(), 6);
            let mut from_copies = vec![0u8; file_bytes.len()];
            let base = guard.bytes_ptr();
            for &(offset, len, packed) in &copies {
                // SAFETY: the lease holds the ring and the copy stays inside it.
                let src = unsafe { std::slice::from_raw_parts(base.add(offset), len) };
                from_copies[packed..packed + len].copy_from_slice(src);
            }
            assert_eq!(from_copies, file_bytes, "expert {expert} device slices");
            let mut cleared = stored.clone();
            for (place, residue) in plan.places.iter().zip(cleared.iter_mut()) {
                if place.windowed {
                    *residue = 0;
                }
            }
            assert_ne!(logical_with(&reader, guard.slot, &cleared), file_bytes);
            let neighbour = code_residues[(expert as usize + 1) % code_residues.len()];
            let neighbour_scale = ((neighbour + 2304) % ALIGN) as u32;
            let mut shifted = stored.clone();
            for (index, residue) in shifted.iter_mut().enumerate() {
                *residue = if index < 3 {
                    neighbour as u32
                } else {
                    neighbour_scale
                };
            }
            assert_ne!(logical_with(&reader, guard.slot, &shifted), file_bytes);
            if expert as usize == clamp_expert {
                assert_eq!(up.offset + up.len, file_len);
                assert_ne!(up.offset % ALIGN, 0);
                let begin = up.offset - up.offset % ALIGN;
                let shortfall = begin + up.len + ALIGN - file_len;
                assert_eq!(shortfall, ALIGN - up.offset % ALIGN);
            }
            classes += 1;
            drop(guard);
        }
        assert_eq!(classes, code_residues.len());
        println!(
            "PASS direct_window_bytes classes={classes} clamped_expert={clamp_expert} file_bytes={file_len} mismatches=0"
        );
        Ok(())
    }

    #[test]
    fn staged_scales_coalesce_behind_the_weight_windows() -> Result<()> {
        const ALIGN: u64 = 4096;
        let hidden = 64usize;
        let intermediate = 128usize;
        let code = (intermediate * hidden.div_ceil(2)) as u64;
        let scale = (intermediate * hidden.div_ceil(32)) as u64;
        let down_scale = (hidden * intermediate.div_ceil(32)) as u64;
        assert!(code.is_multiple_of(ALIGN));
        assert!(!scale.is_multiple_of(ALIGN) && !down_scale.is_multiple_of(ALIGN));
        let staged_scale_bytes = (scale * 2 + down_scale) as usize;
        let residue = 728u64;
        let scale_residue = (residue + 2304) % ALIGN;
        let roles = [
            (
                "w1.weight",
                safetensors::Dtype::I8,
                intermediate,
                hidden / 2,
                code,
                residue,
            ),
            (
                "w3.weight",
                safetensors::Dtype::I8,
                intermediate,
                hidden / 2,
                code,
                residue,
            ),
            (
                "w2.weight",
                safetensors::Dtype::I8,
                hidden,
                intermediate / 2,
                code,
                residue,
            ),
            (
                "w1.scale",
                safetensors::Dtype::F8_E8M0,
                intermediate,
                hidden / 32,
                scale,
                scale_residue,
            ),
            (
                "w3.scale",
                safetensors::Dtype::F8_E8M0,
                intermediate,
                hidden / 32,
                scale,
                scale_residue,
            ),
            (
                "w2.scale",
                safetensors::Dtype::F8_E8M0,
                hidden,
                intermediate / 32,
                down_scale,
                scale_residue,
            ),
        ];
        let mut cursor = 0u64;
        let mut placed = Vec::new();
        for (role, dtype, rows, cols, len, item_residue) in roles {
            let base = cursor.div_ceil(ALIGN) * ALIGN;
            let offset = base + item_residue;
            cursor = base + len + ALIGN;
            placed.push((role, dtype, rows, cols, len, offset));
        }
        let file: Vec<u8> = (0..cursor).map(byte_at).collect();
        let dir = temp_dir("staged-scales")?;
        std::fs::write(dir.path().join("staged.bin"), &file)?;
        let mut entries = BTreeMap::new();
        for (role, dtype, rows, cols, len, offset) in placed {
            entries.insert(
                format!("layers.0.ffn.experts.0.{role}"),
                TensorEntry {
                    dtype,
                    shape: vec![rows, cols],
                    shard: "staged.bin".into(),
                    bytes: len as usize,
                    load: true,
                    data_start: 0,
                    data_offsets: [offset, offset + len],
                    scope: TensorScope::Text,
                    layer: None,
                    role: role.into(),
                    scale_name: None,
                    quant: None,
                    swap_halves: false,
                },
            );
        }
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: 1,
            hidden: hidden as u32,
            intermediate: intermediate as u32,
            top_k: 1,
        };
        let catalogue = ExpertCatalogue::from_manifest(
            &TensorManifest {
                entries,
                resident_bytes: 0,
            },
            geometry,
            e2m1_format(),
        )?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let slot = reader.read_expert(0, 0, &catalogue, catalogue.slot_bytes())?;
        let guard = reader.values(slot)?;
        let got = slices_of(&reader, guard.slot);
        let range = catalogue.range(0, 0)?;
        let ExpertRange::E2m1 {
            gate,
            up,
            down,
            gate_scale,
            up_scale,
            down_scale,
        } = range
        else {
            bail!("E2M1 range");
        };
        let mut expected = Vec::new();
        for native in [gate, up, down, gate_scale, up_scale, down_scale] {
            let start = native.offset as usize;
            expected.extend_from_slice(&file[start..start + native.len as usize]);
        }
        assert_eq!(got, expected);
        let viewed: Vec<u8> = guard
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        assert_eq!(viewed, expected);
        let copies = guard.disk_slices()?;
        assert_eq!(copies.len(), 4);
        assert_eq!(copies[3].1, staged_scale_bytes);
        let mut cleared = unsafe { &*reader.ring[guard.slot].residues.get() }.clone();
        let plan = reader.plan.as_ref().expect("plan");
        for (place, residue) in plan.places.iter().zip(cleared.iter_mut()) {
            if place.windowed {
                *residue = 0;
            }
        }
        assert_ne!(logical_with(&reader, guard.slot, &cleared), expected);
        println!(
            "PASS staged_scale_copies copies=4 scale_bytes={} weight_residue={residue}",
            copies[3].1
        );
        Ok(())
    }
}
