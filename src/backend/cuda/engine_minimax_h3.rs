//! MiniMax-H3 FL2VA transformer execution, and the MiniMax-H3-Image transformer.
use crate::backend::cuda::weights::{
    DequantCopy, QuantKernels, dequantizes, device_slot_bytes, quant_source_finite,
};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::Flash,
    linear::LinearBiasMode,
    ops::{self, flat},
    verification::Shape,
};
use crate::backend::gpu::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use crate::checkpoint::{MappedCheckpoint, TensorEntry, TensorManifest};
use crate::convrot::ProductKernel;
use crate::experts::disk::{AlignedBytes, SpanReader};
use crate::gguf::GgmlType;
use crate::minimax_h3::{
    AdalnPlan, AdalnTimeEmbedder, H3Spec, ImageLayout, ImageSchedule, ImageSpec, LinearStorage,
    MODALITIES, adaln_input, image_row_plan, mixed_adaln_input, rotary_rows,
};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use safetensors::Dtype;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, OnceLock},
};

#[path = "engine_minimax_h3_audio.rs"]
mod audio;
#[path = "engine_minimax_h3_pipeline.rs"]
mod pipeline;
#[cfg(feature = "cuda")]
pub use pipeline::VideoSession;
pub use pipeline::{ImageSession, image_latents};
#[path = "engine_minimax_h3_vae.rs"]
mod vae;

pub(super) fn open_blas(
    device: &Device,
    ops: &ops::Ops,
    runtime: &Path,
    closure: Rc<Closure>,
) -> Result<Blas> {
    #[cfg(feature = "cuda")]
    {
        Blas::new(device, ops, runtime, closure)
    }
    #[cfg(feature = "rocm")]
    {
        let _ = (ops, runtime, closure);
        Blas::new(Arc::clone(&device.stream))
    }
}

fn equal_int8_bf16_rate(arch: &crate::backend::setup::DeviceArchitecture) -> bool {
    matches!(
        arch,
        crate::backend::setup::DeviceArchitecture::Hip(name) if name.starts_with("gfx11")
    )
}

fn bf16_preferred_for_device(
    arch: &crate::backend::setup::DeviceArchitecture,
    spec: &H3Spec,
    geometry: &crate::minimax_h3::Geometry,
    free_bytes: u64,
    weight_bytes: u64,
) -> Result<bool> {
    if !equal_int8_bf16_rate(arch) {
        return Ok(false);
    }
    let activation_bytes = crate::minimax_h3::unchunked_activation_bytes(spec, geometry)?;
    let sum = weight_bytes
        .checked_add(activation_bytes)
        .context("placement byte sum")?;
    Ok(sum <= free_bytes)
}

/// Selects `fl2va_bf16` over `fl2va_int8_convrot` when detected hardware has equal BF16/INT8 tensor-core rate
/// and the BF16 checkpoint plus activations fit in device memory.
pub(crate) fn prefer_hyperflow_bf16(
    ordinal: usize,
    spec: &H3Spec,
    component: &Path,
    geometry: &crate::minimax_h3::Geometry,
) -> Result<bool> {
    let device = Device::new(ordinal)?;
    if !equal_int8_bf16_rate(&device.info.arch) {
        return Ok(false);
    }
    let manifest = MappedCheckpoint::open_component(component, "diffusion_pytorch_model")?
        .minimax_h3_manifest(spec)?;
    let (free, _total) = device.ctx.mem_get_info()?;
    bf16_preferred_for_device(
        &device.info.arch,
        spec,
        geometry,
        u64::try_from(free)?,
        u64::try_from(manifest.resident_bytes)?,
    )
}

#[cfg(test)]
pub(super) fn capture_graph(
    stream: &Arc<CudaStream>,
    auto_free: bool,
    launch: impl FnOnce() -> Result<()>,
) -> Result<crate::backend::gpu::CudaGraph> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::{CUgraphInstantiate_flags as F, CUstreamCaptureMode as M};
        stream.begin_capture(M::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let launched = launch();
        let flags = if auto_free {
            F::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
        } else {
            F(0)
        };
        let captured = stream.end_capture(flags);
        launched?;
        captured?.context("stream capture produced no graph")
    }
    #[cfg(feature = "rocm")]
    {
        let _ = auto_free;
        let capture = stream.begin_capture()?;
        let launched = launch();
        let captured = capture.finish();
        launched?;
        captured
    }
}

/// Decode normalized F32 NCHW image latents to F32 RGB HWC before clamp and quantization.
pub fn image_decode(
    model: &Path,
    (ordinal, runtime): (usize, &Path),
    extent: [usize; 2],
    normalized: &[f32],
) -> Result<Vec<f32>> {
    ensure!(
        normalized.iter().all(|v| v.is_finite()),
        "image VAE input is nonfinite"
    );
    let device = Device::new(ordinal)?;
    let ops = ops::Ops::new(&device.ctx)?;
    device.finish_upload()?;
    let mut blas = open_blas(
        &device,
        &ops,
        runtime,
        Rc::new(Closure::new(true, "image VAE".into())),
    )?;
    let input = device.stream.clone_htod(normalized)?;
    let rgb = vae::decode_image_device(&device, &mut blas, model, &input, extent)?;
    let output = device.stream.clone_dtoh(&rgb)?;
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "image VAE output is nonfinite"
    );
    Ok(output)
}

/// F32 RGB HWC in [0,255] and caller posterior noise to normalized F32 NCHW latents and their spatial extent.
pub fn image_encode(
    model: &Path,
    (ordinal, runtime): (usize, &Path),
    extent: [usize; 2],
    rgb: &[f32],
    epsilon: &[f32],
) -> Result<(Vec<f32>, [usize; 2])> {
    ensure!(
        rgb.iter()
            .all(|v| v.is_finite() && (0.0..=255.0).contains(v)),
        "image VAE RGB outside [0,255]"
    );
    ensure!(
        epsilon.iter().all(|v| v.is_finite()),
        "image VAE posterior noise is nonfinite"
    );
    let device = Device::new(ordinal)?;
    let ops = ops::Ops::new(&device.ctx)?;
    device.finish_upload()?;
    let blas = open_blas(
        &device,
        &ops,
        runtime,
        Rc::new(Closure::new(true, "image VAE encoder".into())),
    )?;
    let pixels = device.stream.clone_htod(rgb)?;
    let noise = device.stream.clone_htod(epsilon)?;
    let (latents, shape) =
        vae::encode_image_device(&device, &blas, model, &pixels, extent, &noise)?;
    let output = device.stream.clone_dtoh(&latents)?;
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "image VAE encoder output is nonfinite"
    );
    Ok((output, shape))
}

/// Direct-IO read granularity.
const READ_CHUNK_BYTES: u64 = 32 << 20;

/// Published element types a tensor upload accepts.
trait Element: crate::backend::gpu::DeviceRepr {
    const DTYPE: Dtype;
    fn finite(bytes: &[u8]) -> bool;
}

impl Element for bf16 {
    const DTYPE: Dtype = Dtype::BF16;
    fn finite(bytes: &[u8]) -> bool {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| u16::from_le_bytes(*v) & 0x7f80 != 0x7f80)
    }
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn finite(bytes: &[u8]) -> bool {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_le_bytes(*v).is_finite())
    }
}

impl Element for i8 {
    const DTYPE: Dtype = Dtype::I8;
    /// Every int8 code is finite.
    fn finite(_bytes: &[u8]) -> bool {
        true
    }
}

/// O_DIRECT reads of single manifest tensors through one aligned pageable staging buffer.
struct EntryReader {
    dir: PathBuf,
    readers: BTreeMap<String, SpanReader>,
    staging: Option<AlignedBytes>,
    /// Set once from the device capability. sm below 80 rewrites BF16 weight bytes to FP16.
    fp16_weights: Option<bool>,
}

/// One batched direct-read destination: tensor identity plus a disjoint writable
/// window of the pinned arena at the tensor's packed offset, exactly its
/// `entry.bytes`; only the owning worker thread dereferences it.
struct DirectRead<'a> {
    name: &'a str,
    entry: &'a TensorEntry,
    dest: *mut u8,
    dest_len: usize,
    cast: FillCast,
}

fn align_min() -> usize {
    4096
}

/// The batch base pointer, Send by the disjoint-index discipline of `read_many_direct`.
struct DirectBatch(*mut u8);

impl Copy for DirectBatch {}
impl Clone for DirectBatch {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl Send for DirectBatch {}
unsafe impl Sync for DirectBatch {}

/// One owned direct-read destination: a tensor identity plus a disjoint writable window of a
/// pinned arena. `Raw` fills `entry.bytes`; `F32ToBf16` fills half of it with the narrowed
/// values. Ownership lets the batch cross into a fill thread.
#[derive(Clone)]
pub(crate) struct OwnedRead {
    name: String,
    entry: TensorEntry,
    dest: *mut u8,
    dest_len: usize,
    cast: FillCast,
}

/// Destination form of one fill read: the stored bytes, their BF16 narrowing, or BF16
/// weight bits rewritten as FP16. F32, I8, and U8 stay [`FillCast::Raw`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FillCast {
    Raw,
    F32ToBf16,
    Bf16ToFp16,
}

fn weight_fill_cast(fp16: bool, entry: &TensorEntry) -> FillCast {
    if fp16 && entry.quant.is_none() && entry.dtype == Dtype::BF16 {
        FillCast::Bf16ToFp16
    } else {
        FillCast::Raw
    }
}

/// The resident ring's raw windows keep the loader's finite check: unquantized F32 kept as
/// F32 is scanned before the window is published as a completed tensor.
pub(super) fn resident_raw_finite(
    name: &str,
    entry: &TensorEntry,
    cast: FillCast,
    window: &[u8],
) -> Result<()> {
    if cast == FillCast::Raw && entry.quant.is_none() && entry.dtype == Dtype::F32 {
        ensure!(<f32 as Element>::finite(window), "{name}: nonfinite value");
    }
    Ok(())
}

/// Round-robin claim order across the shard queues of `requests`: position-major over the
/// per-shard tensor queues, so concurrent claimers spread across shards.
pub(crate) fn claim_order<T>(requests: &[T], shard: impl Fn(&T) -> &str) -> Vec<usize> {
    let mut queues: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, request) in requests.iter().enumerate() {
        queues.entry(shard(request)).or_default().push(index);
    }
    let queues: Vec<Vec<usize>> = queues.into_values().collect();
    let total: usize = queues.iter().map(Vec::len).sum();
    let mut order = Vec::with_capacity(total);
    let deepest = queues.iter().map(Vec::len).max().unwrap_or(0);
    for position in 0..deepest {
        for queue in &queues {
            if let Some(&index) = queue.get(position) {
                order.push(index);
            }
        }
    }
    order
}

impl OwnedRead {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn entry(&self) -> &TensorEntry {
        &self.entry
    }

    pub(crate) fn dest_len(&self) -> usize {
        self.dest_len
    }

    pub(crate) fn cast(&self) -> FillCast {
        self.cast
    }

    /// One tensor into its `dest_len` window at `dest`.
    pub(crate) fn new(
        name: &str,
        entry: &TensorEntry,
        dest: *mut u8,
        dest_len: usize,
        cast: FillCast,
    ) -> Self {
        Self {
            name: name.to_owned(),
            entry: entry.clone(),
            dest,
            dest_len,
            cast,
        }
    }
}

// SAFETY: every destination window is disjoint; only the fill batch touches it before the join.
unsafe impl Send for OwnedRead {}
unsafe impl Sync for OwnedRead {}

/// Workers a fill batch runs across: twice the shard count, clamped to `[4, 8]`, bounded by the
/// tensor count.
pub(crate) fn fill_in_flight(shards: usize, tensors: usize) -> usize {
    (2 * shards).clamp(4, 8).min(tensors.max(1))
}

/// Runs one fill batch through a fresh reader on the calling thread.
fn run_fill_batch(dir: &Path, requests: &[OwnedRead]) -> Result<()> {
    let mut reader = EntryReader::new(dir);
    let mut batch: Vec<DirectRead<'_>> = requests
        .iter()
        .map(|request| DirectRead {
            name: &request.name,
            entry: &request.entry,
            dest: request.dest,
            dest_len: request.dest_len,
            cast: request.cast,
        })
        .collect();
    reader.read_many_direct(&mut batch)
}

/// A fill batch running on its own thread; a failed read surfaces at [`FillThread::join`] with
/// the reader's error text.
struct FillThread {
    handle: Option<std::thread::JoinHandle<Result<()>>>,
}

impl FillThread {
    fn spawn(dir: &Path, requests: Vec<OwnedRead>) -> Result<Self> {
        let dir = dir.to_path_buf();
        let handle = std::thread::Builder::new()
            .name("h3-fill".to_owned())
            .spawn(move || run_fill_batch(&dir, &requests))?;
        Ok(Self {
            handle: Some(handle),
        })
    }

    fn join(&mut self) -> Result<()> {
        let handle = self.handle.take().context("fill thread joined twice")?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("h3-fill worker panicked"))?
    }
}

impl Drop for FillThread {
    /// An abandoned fill still completes its reads; its result surfaces only through `join`.
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle
                .join()
                .map_err(|_| anyhow::anyhow!("h3-fill worker panicked"));
        }
    }
}

/// Bounce buffers reserved by live background fills; the upper bound is one largest-tensor
/// buffer per worker.
static LIVE_FILL_BOUNCE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The host reserve a dense tier keeps beside its pinned layer copies: `slot_count` device
/// slots, the largest single tensor, and 4 GiB.
fn pinned_host_reserve(slot_count: usize, slot_bytes: usize, largest: usize) -> Result<usize> {
    slot_count
        .checked_mul(slot_bytes)
        .and_then(|slots| slots.checked_add(largest))
        .and_then(|bytes| bytes.checked_add(4 << 30))
        .context("pinned host reserve overflow")
}

enum FillSchedule {
    Background(FillThread),
    Serial(Vec<OwnedRead>),
}

/// Derives whether a fill of `requests` runs beside the live fills within the sensed host
/// budget, and the bounce bound it reserves. `reserve` mirrors the caller's host-tier reserve
/// ([`pinned_host_reserve`]). Serial keeps the inline order; nothing is retried.
fn schedule_fill(
    component: &str,
    dir: &Path,
    requests: Vec<OwnedRead>,
    reserve: usize,
) -> Result<(FillSchedule, usize)> {
    let shards = requests
        .iter()
        .map(|request| request.entry.shard.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let in_flight = fill_in_flight(shards, requests.len());
    let largest = requests
        .iter()
        .map(|request| request.dest_len)
        .max()
        .unwrap_or(0);
    let bounce = in_flight
        .checked_mul(largest)
        .context("fill bounce bound overflow")?;
    let live = LIVE_FILL_BOUNCE_BYTES.fetch_add(bounce, std::sync::atomic::Ordering::SeqCst);
    let available = usize::try_from(crate::backend::cuda::calibrate::host_memory(Path::new(
        "/",
    ))?)?;
    let background = !requests.is_empty()
        && available
            .checked_sub(live)
            .and_then(|bytes| bytes.checked_sub(bounce))
            .is_some_and(|bytes| bytes >= reserve);
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_fill_schedule": {
                "component": component, "decision": if background { "background" } else { "serial" },
                "requests": requests.len(), "shards": shards, "in_flight": in_flight,
                "bounce_bound_bytes": bounce, "live_fill_bounce_bytes": live,
                "host_available_bytes": available, "host_reserve_bytes": reserve}})
        );
    }
    if !background {
        LIVE_FILL_BOUNCE_BYTES.fetch_sub(bounce, std::sync::atomic::Ordering::SeqCst);
        return Ok((FillSchedule::Serial(requests), 0));
    }
    Ok((
        FillSchedule::Background(FillThread::spawn(dir, requests)?),
        bounce,
    ))
}

/// One post-join finite check: `bytes` at arena offset `at` of `layer`.
impl FillCheck {
    /// The finite check for a filled window: unquantized `dtype` values, `bytes` at `at`.
    pub(crate) fn finite(
        layer: usize,
        tensor: &str,
        dtype: Dtype,
        at: usize,
        bytes: usize,
    ) -> Self {
        Self {
            layer,
            tensor: tensor.to_owned(),
            dtype,
            quant: None,
            at,
            bytes,
        }
    }
}

pub(crate) struct FillCheck {
    pub(crate) layer: usize,
    tensor: String,
    dtype: Dtype,
    quant: Option<GgmlType>,
    at: usize,
    bytes: usize,
}

/// Validates filled host-tier windows: unquantized BF16 and F32 tensors are finite.
fn check_filled_tier(
    dense: &mut crate::backend::cuda::weights::DenseStreaming,
    checks: &[FillCheck],
) -> Result<()> {
    for check in checks {
        if check.quant.is_some() {
            continue;
        }
        ensure!(
            matches!(check.dtype, Dtype::BF16 | Dtype::F32),
            "{}: {:?} is not a finite-checkable dtype",
            check.tensor,
            check.dtype
        );
        let host = dense
            .host_layer(check.layer)?
            .with_context(|| format!("{}: host tier vanished", check.tensor))?;
        let filled = &host[check.at..check.at + check.bytes];
        let finite = if check.dtype == Dtype::F32 {
            <f32 as Element>::finite(filled)
        } else {
            <bf16 as Element>::finite(filled)
        };
        ensure!(finite, "{}: nonfinite value", check.tensor);
    }
    Ok(())
}

/// A background fill plus the validation that runs once it joins. A failed read or check
/// stores its error text and replays it on every later [`PendingFill::finish`].
pub(crate) struct PendingFill {
    thread: Option<FillThread>,
    bounce: usize,
    checks: Vec<FillCheck>,
    failure: Option<String>,
}

impl PendingFill {
    /// Joins the fill thread; a failed read stores the reader's error text and replays it
    /// on every later call.
    fn join_fill(&mut self) -> Result<()> {
        if let Some(text) = &self.failure {
            bail!("{text}");
        }
        if let Some(thread) = self.thread.as_mut() {
            let joined = thread.join();
            LIVE_FILL_BOUNCE_BYTES.fetch_sub(self.bounce, std::sync::atomic::Ordering::SeqCst);
            self.thread = None;
            if let Err(error) = joined {
                let text = format!("{error:#}");
                self.failure = Some(text.clone());
                bail!("{text}");
            }
        }
        Ok(())
    }

    /// Joins the fill and validates the filled windows; a failed read or check propagates
    /// the reader's error text and stays failed.
    fn finish(&mut self, dense: &mut crate::backend::cuda::weights::DenseStreaming) -> Result<()> {
        self.join_fill()?;
        if let Err(error) = check_filled_tier(dense, &self.checks) {
            self.failure = Some(format!("{error:#}"));
            return Err(error);
        }
        self.checks.clear();
        Ok(())
    }
}

impl Drop for PendingFill {
    fn drop(&mut self) {
        if self.thread.is_some() {
            LIVE_FILL_BOUNCE_BYTES.fetch_sub(self.bounce, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Device slots the placement uses: the largest slot index plus one.
fn used_dense_slots(plan: &crate::prefill::DensePlacement) -> usize {
    plan.slots
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |index| index + 1)
}

/// The dense streaming tier and its background fill. `Drop` joins a pending fill before the
/// field drops free the pinned arena the fill writes into; `ready` hands out the tier only
/// after a pending fill joined and validated.
pub(crate) struct DenseTier {
    fill: Option<PendingFill>,
    dense: crate::backend::cuda::weights::DenseStreaming,
}

impl DenseTier {
    /// Fills `dense`'s host tier: in the background when the derived host budget holds,
    /// otherwise inline. Inline fills validate before returning; background fills defer
    /// their checks to the join before the tier's first consumer.
    pub(crate) fn schedule(
        component: &'static str,
        mut dense: crate::backend::cuda::weights::DenseStreaming,
        dir: &Path,
        requests: Vec<OwnedRead>,
        checks: Vec<FillCheck>,
        (slot_count, slot_bytes, largest): (usize, usize, usize),
    ) -> Result<Self> {
        let reserve = pinned_host_reserve(slot_count, slot_bytes, largest)?;
        let (schedule, bounce) = schedule_fill(component, dir, requests, reserve)?;
        let fill = match schedule {
            FillSchedule::Background(thread) => Some(PendingFill {
                thread: Some(thread),
                bounce,
                checks,
                failure: None,
            }),
            FillSchedule::Serial(requests) => {
                run_fill_batch(dir, &requests)?;
                check_filled_tier(&mut dense, &checks)?;
                None
            }
        };
        Ok(Self { fill, dense })
    }

    /// The streaming tier, joining and validating a pending fill first. A failed fill
    /// replays its error on every call.
    fn ready(&mut self) -> Result<&mut crate::backend::cuda::weights::DenseStreaming> {
        if self.fill.is_some() {
            let mut pending = self.fill.take().expect("fill presence checked");
            if let Err(error) = pending.finish(&mut self.dense) {
                self.fill = Some(pending);
                return Err(error);
            }
        }
        Ok(&mut self.dense)
    }
}

impl Drop for DenseTier {
    fn drop(&mut self) {
        // Joins a running fill before `dense` drops and frees the arena it writes into.
        self.fill.take();
    }
}

impl EntryReader {
    fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_owned(),
            readers: BTreeMap::new(),
            staging: None,
            fp16_weights: None,
        }
    }

    fn ensure_fp16(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        if self.fp16_weights.is_none() {
            self.fp16_weights = Some(cubin::capability(stream.context())? < 80);
        }
        Ok(())
    }

    fn store_bf16_as_fp16(&self, name: &str, bytes: &mut [u8]) -> Result<()> {
        if self.fp16_weights == Some(true) {
            crate::tensors::bf16_weight_to_fp16(name, bytes, true)?;
        }
        Ok(())
    }

    /// Copies a finite BF16 or F32 tensor into a buffer of its element type; the pageable source is consumed when the
    /// copy call returns.
    fn upload<T: Element>(
        &mut self,
        stream: &Arc<CudaStream>,
        name: &str,
        entry: &TensorEntry,
        destination: &mut CudaViewMut<'_, T>,
    ) -> Result<()> {
        ensure!(
            entry.dtype == T::DTYPE && entry.bytes == std::mem::size_of::<T>() * destination.len(),
            "{name}: {:?} {} B does not fill a {}-element {:?} buffer",
            entry.dtype,
            entry.bytes,
            destination.len(),
            T::DTYPE
        );
        self.ensure_fp16(stream)?;
        let convert = self.fp16_weights == Some(true) && entry.dtype == Dtype::BF16;
        let bytes = self.read(name, entry)?;
        ensure!(T::finite(bytes), "{name}: nonfinite value");
        let mut converted = Vec::new();
        let payload = if convert {
            converted.extend_from_slice(bytes);
            crate::tensors::bf16_weight_to_fp16(name, &mut converted, true)?;
            converted.as_slice()
        } else {
            bytes
        };
        let (pointer, _record) = destination.device_ptr_mut(stream);
        unsafe {
            super::copy::htod_async(pointer, payload, stream)?;
        }
        Ok(())
    }

    /// The tensor's source bytes in a device buffer, for a quantized GGUF entry.
    fn read_device(
        &mut self,
        stream: &Arc<CudaStream>,
        name: &str,
        entry: &TensorEntry,
    ) -> Result<CudaSlice<u8>> {
        let bytes = self.read(name, entry)?;
        quant_source_finite(name, entry, bytes)?;
        let mut buffer = stream.alloc_zeros::<u8>(entry.bytes)?;
        {
            let (pointer, _record) = buffer.device_ptr_mut(stream);
            unsafe {
                super::copy::htod_async(pointer, bytes, stream)?;
            }
        }
        Ok(buffer)
    }

    /// Copies a finite BF16 or F32 tensor into host memory of exactly its size.
    fn read_into(&mut self, name: &str, entry: &TensorEntry, destination: &mut [u8]) -> Result<()> {
        ensure!(
            matches!(entry.dtype, Dtype::BF16 | Dtype::F32) && entry.bytes == destination.len(),
            "{name}: {:?} {} B does not fill {} B",
            entry.dtype,
            entry.bytes,
            destination.len()
        );
        let bytes = self.read(name, entry)?;
        let finite = if entry.dtype == Dtype::F32 {
            <f32 as Element>::finite(bytes)
        } else {
            <bf16 as Element>::finite(bytes)
        };
        ensure!(finite, "{name}: nonfinite value");
        destination.copy_from_slice(bytes);
        if entry.dtype == Dtype::BF16 {
            self.store_bf16_as_fp16(name, destination)?;
        }
        Ok(())
    }

    /// The tensor's bytes, read with O_DIRECT into the reusable aligned staging buffer.
    fn read(&mut self, name: &str, entry: &TensorEntry) -> Result<&[u8]> {
        self.read_range(name, entry, 0, entry.bytes)
    }

    /// Fills the pinned windows with one aligned direct read each. Readers are opened
    /// first on this thread; workers then claim single tensors round-robin across the
    /// shard queues, so every destination is written by exactly one worker.
    /// Opens one O_DIRECT reader per shard named by `shards`, reusing open ones.
    fn open_shards<'i>(
        &mut self,
        shards: impl Iterator<Item = &'i str>,
    ) -> Result<&BTreeMap<String, SpanReader>> {
        for shard in shards {
            if !self.readers.contains_key(shard) {
                let reader = SpanReader::open(&self.dir.join(shard), true)?;
                self.readers.insert(shard.to_owned(), reader);
            }
        }
        Ok(&self.readers)
    }

    fn read_many_direct(&mut self, requests: &mut [DirectRead<'_>]) -> Result<()> {
        let mut spans: Vec<(*mut u8, usize)> = requests
            .iter()
            .map(|request| (request.dest, request.dest_len))
            .collect();
        spans.sort_unstable();
        for window in spans.windows(2) {
            ensure!(
                window[0].0.wrapping_add(window[0].1) <= window[1].0,
                "direct batch destination windows overlap"
            );
        }
        let readers =
            self.open_shards(requests.iter().map(|request| request.entry.shard.as_str()))?;
        let order = claim_order(requests, |request| request.entry.shard.as_str());
        let shards = requests
            .iter()
            .map(|request| request.entry.shard.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let in_flight = fill_in_flight(shards, order.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let batch = DirectBatch(requests.as_mut_ptr().cast::<u8>());
        let batch = &batch;
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = (0..in_flight)
                .map(|_| {
                    scope.spawn(|| -> Result<()> {
                        let mut bounce = AlignedBytes::new(align_min(), align_min())?;
                        loop {
                            let claimed = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            let Some(&index) = order.get(claimed) else {
                                return Ok(());
                            };
                            // SAFETY: each index is claimed by exactly one worker;
                            // the window is within the arena.
                            let request = unsafe { &mut *(batch.0 as *mut DirectRead).add(index) };
                            let destination = unsafe {
                                std::slice::from_raw_parts_mut(request.dest, request.dest_len)
                            };
                            Self::read_bounced(
                                readers,
                                request.name,
                                request.entry,
                                request.cast,
                                destination,
                                &mut bounce,
                            )?;
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("direct batch worker panicked"))??;
            }
            Ok(())
        })
    }

    /// One tensor through an aligned worker bounce into its packed destination.
    pub(super) fn read_bounced(
        readers: &BTreeMap<String, SpanReader>,
        name: &str,
        entry: &TensorEntry,
        cast: FillCast,
        destination: &mut [u8],
        bounce: &mut AlignedBytes,
    ) -> Result<()> {
        let reader = readers
            .get(&entry.shard)
            .with_context(|| format!("{name}: shard {} not open", entry.shard))?;
        let start = entry
            .data_start
            .checked_add(entry.data_offsets[0])
            .context("tensor offset overflow")?;
        let align = reader.alignment as usize;
        let head = start as usize % align;
        let begin = start - head as u64;
        let width = entry
            .bytes
            .checked_add(head)
            .and_then(|v| v.checked_add(align - 1))
            .context("pinned span overflow")?
            & !(align - 1);
        let width = width
            .min(usize::try_from(reader.file_len() - begin)?)
            .max(entry.bytes + head);
        let filled = match cast {
            FillCast::Raw => entry.bytes,
            FillCast::Bf16ToFp16 => {
                ensure!(
                    entry.dtype == Dtype::BF16
                        && entry.quant.is_none()
                        && entry.bytes.is_multiple_of(2),
                    "{name}: BF16 to FP16 fill source"
                );
                entry.bytes
            }
            FillCast::F32ToBf16 => {
                ensure!(
                    entry.dtype == Dtype::F32
                        && entry.quant.is_none()
                        && entry.bytes.is_multiple_of(4),
                    "{name}: F32 to BF16 fill source"
                );
                entry.bytes / 2
            }
        };
        ensure!(
            destination.len() >= filled,
            "{name}: dense slot holds {} of {}",
            destination.len(),
            filled
        );
        if bounce.len() < width + align {
            *bounce = AlignedBytes::new(width + align, align)?;
        }
        let span = bounce.as_mut();
        reader.read_direct(begin, width as u64, span)?;
        let source = &span[head..head + entry.bytes];
        match cast {
            FillCast::Raw => destination[..entry.bytes].copy_from_slice(source),
            FillCast::Bf16ToFp16 => {
                let output = &mut destination[..entry.bytes];
                output.copy_from_slice(source);
                crate::tensors::bf16_weight_to_fp16(name, output, true)?;
            }
            FillCast::F32ToBf16 => {
                let output = &mut destination[..filled];
                for (value, dst) in source
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(output.as_chunks_mut::<2>().0)
                {
                    let value = f32::from_le_bytes(*value);
                    ensure!(value.is_finite(), "{name}: nonfinite value");
                    let narrowed = bf16::from_f32(value);
                    ensure!(narrowed.is_finite(), "BF16 overflow for {name}");
                    *dst = narrowed.to_bits().to_le_bytes();
                }
            }
        }
        Ok(())
    }

    /// Bytes `start..start + bytes` of the tensor, read with O_DIRECT into the reusable aligned staging buffer.
    fn read_range(
        &mut self,
        name: &str,
        entry: &TensorEntry,
        start: usize,
        bytes: usize,
    ) -> Result<&[u8]> {
        ensure!(
            start
                .checked_add(bytes)
                .is_some_and(|end| end <= entry.bytes),
            "{name}: range {start}+{bytes} outside {} B",
            entry.bytes
        );
        if !self.readers.contains_key(&entry.shard) {
            let reader = SpanReader::open(&self.dir.join(&entry.shard), true)?;
            self.readers.insert(entry.shard.clone(), reader);
        }
        let reader = &self.readers[&entry.shard];
        let offset = entry
            .data_start
            .checked_add(entry.data_offsets[0])
            .and_then(|offset| offset.checked_add(start as u64))
            .context("tensor offset overflow")?;
        let span = reader
            .plan(&[(offset, bytes as u64)], READ_CHUNK_BYTES)?
            .pop()
            .with_context(|| format!("{name}: no read span"))?;
        let alignment = reader.alignment as usize;
        let needed = usize::try_from(span.end - span.begin)?.next_multiple_of(alignment);
        if self.staging.as_ref().is_none_or(|s| s.len() < needed) {
            self.staging = Some(AlignedBytes::new(needed, alignment)?);
        }
        let staging = self.staging.as_mut().context("staging buffer")?;
        let begin = usize::try_from(offset - span.begin)?;
        Ok(&reader.read_span(&span, READ_CHUNK_BYTES, staging)?[begin..begin + bytes])
    }
}

fn matrix<T: crate::backend::gpu::DeviceRepr + Copy>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    rows: usize,
    width: usize,
) -> Result<Tensor<T, 2>> {
    view(
        b,
        s,
        0,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}

fn vector<T: crate::backend::gpu::DeviceRepr + Copy>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
) -> Result<Tensor<T, 1>> {
    view(b, s, 0, [u32::try_from(b.len())?], [1])
}

/// Setup-time modulation tables, BF16: block rows `[layers, pairs, 6 * hidden]` in `AdalnPlan::pairs` order
/// (shift, scale, gate for attention, then for the FFN) and output-norm rows `[timesteps, 2 * hidden]` (shift, scale).
pub(crate) struct AdalnTable {
    pub blocks: CudaSlice<bf16>,
    pub norm_out: CudaSlice<bf16>,
}

/// Sorted distinct modality tags the plan's pairs address; one AdaLN weight block is read per tag.
fn adaln_tags(pairs: &[(u32, u8)]) -> Vec<u8> {
    let mut tags: Vec<u8> = pairs.iter().map(|&(_, tag)| tag).collect();
    tags.sort_unstable();
    tags.dedup();
    tags
}

/// Compact block index of `tag` among the plan's `tags`, erroring by name.
fn adaln_block(tags: &[u8], tag: u8) -> Result<usize> {
    tags.iter()
        .position(|&candidate| candidate == tag)
        .with_context(|| format!("adaln tag {tag} outside the plan's tag set"))
}

/// Flat element offset of pair `(t, tag)`'s `width` block in the compact table `[timesteps, tags.len() * width]`.
fn adaln_pair_offset(tags: &[u8], width: usize, t: u32, tag: u8) -> Result<usize> {
    Ok((t as usize * tags.len() + adaln_block(tags, tag)?) * width)
}

impl AdalnTable {
    /// Kept blocks: `[6 * hidden * tags, time_dim]` projection of the plan embedding over the plan's addressed
    /// modality tags (full 3-tag tables read every block), bias before output rounding, keeping the addressed
    /// `(timestep, tag)` slices. HyperFlow's embedding is SiLU of `mix_temb` on the two f32
    /// time embedders, rounded to BF16. A dropped block's modulation rows stay zero.
    pub(crate) fn build(
        d: &Device,
        blas: &mut Blas,
        transformer_dir: &Path,
        manifest: &TensorManifest,
        spec: &H3Spec,
        plan: &AdalnPlan,
    ) -> Result<Self> {
        let s = &d.stream;
        let timesteps = plan.timesteps.len();
        let width = 6 * spec.hidden;
        let modulation = width * MODALITIES as usize;
        let mut source = Source::new(transformer_dir, manifest);
        let hyperflow = spec.is_hyperflow()?;
        let time = AdalnTimeEmbedder {
            linear_1_weight: source.host_f32(
                s,
                "time_embedder.linear_1.weight",
                spec.time_hidden * spec.freq_dim,
            )?,
            linear_1_bias: source.host_f32(s, "time_embedder.linear_1.bias", spec.time_hidden)?,
            linear_2_weight: source.host_f32(
                s,
                "time_embedder.linear_2.weight",
                spec.time_dim * spec.time_hidden,
            )?,
            linear_2_bias: source.host_f32(s, "time_embedder.linear_2.bias", spec.time_dim)?,
        };
        let x_host = if hyperflow {
            ensure!(
                !plan.timesteps.is_empty() && plan.endpoints.len() == plan.timesteps.len(),
                "HyperFlow temb mix is not bound"
            );
            let gate = spec.hyperflow_gate.context("HyperFlow gate")?;
            let endpoint = AdalnTimeEmbedder {
                linear_1_weight: source.host_f32(
                    s,
                    "endpoint_time_embedder.linear_1.weight",
                    spec.time_hidden * spec.freq_dim,
                )?,
                linear_1_bias: source.host_f32(
                    s,
                    "endpoint_time_embedder.linear_1.bias",
                    spec.time_hidden,
                )?,
                linear_2_weight: source.host_f32(
                    s,
                    "endpoint_time_embedder.linear_2.weight",
                    spec.time_dim * spec.time_hidden,
                )?,
                linear_2_bias: source.host_f32(
                    s,
                    "endpoint_time_embedder.linear_2.bias",
                    spec.time_dim,
                )?,
            };
            mixed_adaln_input(
                spec,
                &time,
                &endpoint,
                &plan.timesteps,
                &plan.endpoints,
                gate,
            )?
        } else {
            adaln_input(spec, &time, &plan.timesteps)
        };
        let x = s.clone_htod(&x_host)?;
        let tags = adaln_tags(&plan.pairs);
        let compact = width * tags.len();
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_adaln_spans": {"tags": tags, "compact_modulation": compact,
                "full_modulation": modulation, "weight_block_bytes": width * spec.time_dim * 2,
                "bias_block_bytes": 2 * width}})
            );
        }
        let mut weight = s.alloc_zeros::<bf16>(compact * spec.time_dim)?;
        let mut bias = s.alloc_zeros::<bf16>(compact)?;
        let y = s.alloc_zeros::<bf16>(timesteps * compact)?;
        let mut blocks = s.alloc_zeros::<bf16>(spec.layers * plan.pairs.len() * width)?;
        let mode = LinearBiasMode::BeforeOutputRound;
        blas.prepare_biased(
            u32::try_from(timesteps)?,
            u32::try_from(compact)?,
            u32::try_from(spec.time_dim)?,
            vector(&bias, s)?,
            mode,
        )?;
        let dropped = spec.dropped_blocks.as_deref().unwrap_or(&[]);
        let weight_block = width * spec.time_dim * 2;
        for layer in 0..spec.layers {
            if dropped.contains(&layer) {
                continue;
            }
            let base = format!("transformer_blocks.{layer}.adaln_proj.linear");
            let weight_name = format!("{base}.weight");
            let quantized = manifest
                .entries
                .get(&weight_name)
                .with_context(|| format!("manifest lacks {weight_name}"))?
                .quant
                .is_some();
            if quantized {
                // GGUF adaln tensors are single quantized tensors; every quantized plan addresses all
                // modality blocks, so the compact buffers already span the whole tensor.
                ensure!(
                    tags.len() == MODALITIES as usize,
                    "{weight_name}: quantized tensor with a {}-tag plan",
                    tags.len()
                );
                source.fill(s, &weight_name, &mut weight.slice_mut(..))?;
                source.fill(s, &format!("{base}.bias"), &mut bias.slice_mut(..))?;
            } else {
                for (index, &tag) in tags.iter().enumerate() {
                    let first = usize::from(tag);
                    source.fill_span(
                        s,
                        &weight_name,
                        first * weight_block,
                        weight_block,
                        &mut weight.slice_mut(
                            index * width * spec.time_dim..(index + 1) * width * spec.time_dim,
                        ),
                    )?;
                    source.fill_span(
                        s,
                        &format!("{base}.bias"),
                        first * 2 * width,
                        2 * width,
                        &mut bias.slice_mut(index * width..(index + 1) * width),
                    )?;
                }
            }
            blas.linear_biased(
                matrix(&weight, s, compact, spec.time_dim)?,
                vector(&bias, s)?,
                matrix(&x, s, timesteps, spec.time_dim)?,
                &mut TensorMut(matrix(&y, s, timesteps, compact)?),
                mode,
            )?;
            for (pair, &(t, tag)) in plan.pairs.iter().enumerate() {
                let source = adaln_pair_offset(&tags, width, t, tag)?;
                let target = (layer * plan.pairs.len() + pair) * width;
                s.memcpy_dtod(
                    &y.slice(source..source + width),
                    &mut blocks.slice_mut(target..target + width),
                )?;
            }
        }
        let norm = 2 * spec.hidden;
        let mut norm_weight = s.alloc_zeros::<bf16>(norm * spec.time_dim)?;
        let mut norm_bias = s.alloc_zeros::<bf16>(norm)?;
        for (suffix, buffer) in [("weight", &mut norm_weight), ("bias", &mut norm_bias)] {
            let name = format!("norm_out.linear.{suffix}");
            source.fill(s, &name, &mut buffer.slice_mut(..))?;
        }
        let norm_out = s.alloc_zeros::<bf16>(timesteps * norm)?;
        blas.prepare_biased(
            u32::try_from(timesteps)?,
            u32::try_from(norm)?,
            u32::try_from(spec.time_dim)?,
            vector(&norm_bias, s)?,
            mode,
        )?;
        blas.linear_biased(
            matrix(&norm_weight, s, norm, spec.time_dim)?,
            vector(&norm_bias, s)?,
            matrix(&x, s, timesteps, spec.time_dim)?,
            &mut TensorMut(matrix(&norm_out, s, timesteps, norm)?),
            mode,
        )?;
        s.synchronize()?;
        for (name, table) in [("adaln.blocks", &blocks), ("adaln.norm_out", &norm_out)] {
            ensure!(
                !table.is_empty() && pointer(table, s) != 0,
                "{name} is not device-resident"
            );
        }
        Ok(Self { blocks, norm_out })
    }
}

/// Initial FFN chunk ceiling before weight placement.
const FFN_CHUNK_ROWS: usize = 1024;

/// Rows per FFN chunk: `rows` split evenly over `ceil(rows / FFN_CHUNK_ROWS)` chunks, so the tail chunk holds at
/// least `chunk - (chunks - 1)` rows.
fn ffn_chunk(rows: usize) -> usize {
    rows.div_ceil(rows.div_ceil(FFN_CHUNK_ROWS))
}

#[derive(Clone, Copy, Debug)]
struct FfnPlan {
    chunk: usize,
    borrowed: bool,
    allocation: usize,
    workspace: usize,
    tensor_cost: f64,
    launches: usize,
}

impl FfnPlan {
    fn alias_capacity(attention: (usize, usize), ffn: usize) -> Result<usize> {
        ensure!(ffn > 0, "zero FFN width");
        let hidden = ffn.checked_mul(2).context("FFN hidden width overflow")?;
        Ok((attention.0 / hidden).min(attention.1 / ffn))
    }

    fn allocation(chunk: usize, ffn: usize) -> Result<usize> {
        chunk
            .checked_mul(ffn)
            .and_then(|n| n.checked_mul(3 * std::mem::size_of::<bf16>()))
            .context("FFN workspace extent overflow")
    }

    fn select(choices: &[Self], budget: usize) -> Result<Self> {
        let mut best = None;
        for &plan in choices {
            let allocation = plan
                .allocation
                .checked_add(plan.workspace)
                .context("FFN allocation overflow")?;
            if allocation > budget {
                continue;
            }
            ensure!(
                plan.tensor_cost.is_finite() && plan.tensor_cost >= 0.0,
                "invalid FFN tensor cost"
            );
            let score = (plan.tensor_cost, plan.launches, allocation);
            if best.as_ref().is_none_or(|(_, previous)| score < *previous) {
                best = Some((plan, score));
            }
        }
        Ok(best
            .context("no FFN plan fits the measured workspace budget")?
            .0)
    }
}
/// Threads per hidden-width RMSNorm row: 672 eight-element vectors of a 5376 row fit one register pass.
const HIDDEN_NORM_THREADS: u32 = 768;
/// Threads per head-width RMSNorm row.
const HEAD_NORM_THREADS: u32 = 32;
/// Heads per CTA (`8` warps) and threads per CTA for fused DiT head RMSNorm + split-half RoPE.
const QK_ROPE_WARPS: usize = 8;
const QK_ROPE_THREADS: u32 = 256;

/// A vector view and its owned allocation or shared byte arena.
struct VectorWeight<T> {
    tensor: Tensor<T, 1>,
    _owned: Option<CudaSlice<T>>,
    _arena: Option<Arc<CudaSlice<u8>>>,
}

/// One base linear: dense BF16, int8 convrot (`qweight` `[out, in]`, `wscale` `[out]`), or `up(down(x))`.
enum BlockLinear {
    Dense {
        weight: Tensor<bf16, 2>,
        owned: Option<CudaSlice<bf16>>,
    },
    Int8 {
        qweight: Tensor<i8, 2>,
        wscale: VectorWeight<f32>,
        group: u32,
        owned: Option<CudaSlice<i8>>,
    },
    LowRank {
        down: Tensor<bf16, 2>,
        up: Tensor<bf16, 2>,
        owned: Vec<CudaSlice<bf16>>,
    },
}

/// One block's weights. `linears` is `LINEAR_ROLES` order. Dense MiniMax views point into `storage` or a streaming
/// slot. A dropped block's `linears` are 1×1 views of the vector arena and `dropped` is set.
pub(crate) struct BlockWeights {
    norm1: VectorWeight<bf16>,
    norm2: VectorWeight<bf16>,
    norm_q: VectorWeight<bf16>,
    norm_k: VectorWeight<bf16>,
    linears: [BlockLinear; 6],
    storage: Option<CudaSlice<bf16>>,
    dropped: bool,
    qkv_a: CudaSlice<bf16>,
    q_b: CudaSlice<bf16>,
    k_b: CudaSlice<bf16>,
    v_b: CudaSlice<bf16>,
    out_a: CudaSlice<bf16>,
    out_b: CudaSlice<bf16>,
    in_a: CudaSlice<bf16>,
    in_b: CudaSlice<bf16>,
    ffo_a: CudaSlice<bf16>,
    ffo_b: CudaSlice<bf16>,
    rank: usize,
}

/// A checkpoint directory read through its manifest.
pub(crate) struct Source<'a> {
    reader: EntryReader,
    manifest: &'a TensorManifest,
    quant: OnceLock<QuantKernels>,
    vectors: Option<(VectorArenaLayout, Arc<CudaSlice<u8>>)>,
    #[cfg(test)]
    vector_bytes: Option<Vec<u8>>,
}

impl<'a> Source<'a> {
    pub(crate) fn new(dir: &Path, manifest: &'a TensorManifest) -> Self {
        Self {
            reader: EntryReader::new(dir),
            manifest,
            quant: OnceLock::new(),
            vectors: None,
            #[cfg(test)]
            vector_bytes: None,
        }
    }

    fn quant_kernels(&self, s: &Arc<CudaStream>) -> Result<&QuantKernels> {
        if let Some(kernels) = self.quant.get() {
            return Ok(kernels);
        }
        let kernels = QuantKernels::new(s.context())?;
        let _ = self.quant.set(kernels);
        self.quant.get().context("quant kernels")
    }

    fn fill<T: Element>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        destination: &mut CudaViewMut<'_, T>,
    ) -> Result<()> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .with_context(|| format!("manifest lacks {name}"))?;
        let quant = match entry.quant {
            Some(quant) => quant,
            None => return self.reader.upload(s, name, entry, destination),
        };
        match quant {
            GgmlType::F32 | GgmlType::Bf16 => self.reader.upload(s, name, entry, destination),
            GgmlType::F16 | GgmlType::Q4_0 => {
                let src = self.reader.read_device(s, name, entry)?;
                let kernels = self.quant_kernels(s)?;
                let elements = destination.len();
                let (pointer, _record) = destination.device_ptr_mut(s);
                kernels.dequant(
                    s,
                    name,
                    entry,
                    DequantCopy {
                        source: src.device_ptr(s).0,
                        destination: pointer,
                        elements,
                        dtype: T::DTYPE,
                    },
                )?;
                Ok(())
            }
            other => bail!("{name}: unsupported GGUF type {other:?}"),
        }
    }

    /// `bytes` of an unquantized tensor at entry-relative byte `first`, read through the shared staging path
    /// (`EntryReader::read_range`) into `destination`; no second reader.
    fn fill_span<T: Element>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        first: usize,
        bytes: usize,
        destination: &mut CudaViewMut<'_, T>,
    ) -> Result<()> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .with_context(|| format!("manifest lacks {name}"))?;
        ensure!(
            entry.quant.is_none() && entry.dtype == T::DTYPE,
            "{name}: span fill needs an unquantized {:?} tensor, found {:?} quant {:?}",
            T::DTYPE,
            entry.dtype,
            entry.quant
        );
        ensure!(
            destination.len() * std::mem::size_of::<T>() == bytes,
            "{name}: span {bytes} B does not fill a {}-element {:?} buffer",
            destination.len(),
            T::DTYPE
        );
        self.reader.ensure_fp16(s)?;
        let convert = self.reader.fp16_weights == Some(true) && entry.dtype == Dtype::BF16;
        let slice = self.reader.read_range(name, entry, first, bytes)?;
        ensure!(T::finite(slice), "{name}: nonfinite value");
        let mut converted = Vec::new();
        let payload = if convert {
            converted.extend_from_slice(slice);
            crate::tensors::bf16_weight_to_fp16(name, &mut converted, true)?;
            converted.as_slice()
        } else {
            slice
        };
        let (pointer, _record) = destination.device_ptr_mut(s);
        unsafe {
            super::copy::htod_async(pointer, payload, s)?;
        }
        Ok(())
    }

    fn tensor<T: Element + crate::backend::gpu::ValidAsZeroBits>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        elements: usize,
    ) -> Result<CudaSlice<T>> {
        #[cfg(test)]
        let free_before = s.context().mem_get_info()?.0;
        let mut buffer = s.alloc_zeros::<T>(elements)?;
        self.fill(s, name, &mut buffer.slice_mut(..))?;
        #[cfg(test)]
        eprintln!(
            "{}",
            serde_json::json!({"h3_allocation_tensor": {
                "name": name, "dtype": format!("{:?}", T::DTYPE),
                "logical_bytes": elements * std::mem::size_of::<T>(),
                "free_before": free_before, "free_after": s.context().mem_get_info()?.0,
                "synchronized": false
            }})
        );
        Ok(buffer)
    }

    fn host_f32(&mut self, s: &Arc<CudaStream>, name: &str, elements: usize) -> Result<Vec<f32>> {
        let tensor = self.tensor::<f32>(s, name, elements)?;
        s.clone_dtoh(&tensor)
            .with_context(|| format!("copy {name} to host"))
    }

    fn vector<T: Element + crate::backend::gpu::ValidAsZeroBits + Copy>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        elements: usize,
    ) -> Result<VectorWeight<T>> {
        if let Some((layout, buffer)) = &self.vectors {
            let &(offset, bytes, dtype) = layout
                .offsets
                .get(name)
                .with_context(|| format!("vector arena lacks {name}"))?;
            ensure!(
                dtype == T::DTYPE && bytes == elements * std::mem::size_of::<T>(),
                "{name}: vector arena type or extent"
            );
            let ptr = pointer(buffer, s)
                .checked_add(u64::try_from(offset)?)
                .context("vector arena address overflow")?;
            return Ok(VectorWeight {
                tensor: unsafe {
                    Tensor::new(ptr, [u32::try_from(elements)?], [1], u64::try_from(bytes)?)?
                },
                _owned: None,
                _arena: Some(buffer.clone()),
            });
        }
        let owned = self.tensor::<T>(s, name, elements)?;
        let tensor = view(&owned, s, 0, [u32::try_from(elements)?], [1])?;
        Ok(VectorWeight {
            tensor,
            _owned: Some(owned),
            _arena: None,
        })
    }

    fn pack_vectors(&mut self, s: &Arc<CudaStream>, spec: &H3Spec) -> Result<()> {
        ensure!(self.vectors.is_none(), "vector arena already loaded");
        self.reader.ensure_fp16(s)?;
        let layout = VectorArenaLayout::new(self.manifest, spec)?;
        let dropped = spec
            .dropped_blocks
            .as_deref()
            .context("HyperFlow dropped blocks")?;
        let mut bytes = vec![0u8; layout.bytes];
        for (name, &(offset, len, dtype)) in &layout.offsets {
            if let Some(entry) = self.manifest.entries.get(name) {
                ensure!(
                    entry.load
                        && entry.dtype == dtype
                        && entry.bytes == len
                        && entry.quant.is_none(),
                    "{name}: vector arena manifest extent or type"
                );
                self.reader
                    .read_into(name, entry, &mut bytes[offset..offset + len])?;
            } else {
                ensure!(
                    dropped
                        .iter()
                        .any(|layer| name.starts_with(&format!("transformer_blocks.{layer}."))),
                    "manifest lacks vector arena tensor {name}"
                );
            }
        }
        let buffer = Arc::new(s.clone_htod(&bytes)?);
        #[cfg(test)]
        {
            self.vector_bytes = Some(bytes);
        }
        self.vectors = Some((layout, buffer));
        Ok(())
    }
}

/// Roles of `BlockWeights::linears`, in order.
const LINEAR_ROLES: [&str; 6] = [
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "ff.net.0.proj.weight",
    "ff.net.2.weight",
];
const Q: usize = 0;
const K: usize = 1;
const V: usize = 2;
const OUT: usize = 3;
const FF_IN: usize = 4;
const FF_OUT: usize = 5;

/// Element offset and shape of each loaded matrix of `block` in manifest name order, the order `DenseStreaming`
/// fills a layer slot in, and the block's total matrix elements.
type MatrixLayout = BTreeMap<String, (usize, [u32; 2])>;

fn matrix_layout(
    manifest: &TensorManifest,
    block: &str,
    roles: &[&str],
) -> Result<(MatrixLayout, usize)> {
    let prefix = format!("{block}.");
    let mut layout = BTreeMap::new();
    let mut elements = 0usize;
    for (name, entry) in &manifest.entries {
        let Some(role) = name.strip_prefix(&prefix) else {
            continue;
        };
        if !entry.load || entry.shape.len() != 2 {
            continue;
        }
        ensure!(entry.dtype == Dtype::BF16, "{name} is not BF16");
        layout.insert(
            role.to_owned(),
            (
                elements,
                [
                    u32::try_from(entry.shape[0])?,
                    u32::try_from(entry.shape[1])?,
                ],
            ),
        );
        elements += entry.shape[0] * entry.shape[1];
    }
    ensure!(
        layout.len() == roles.len() && roles.iter().all(|r| layout.contains_key(*r)),
        "{block}: matrix set {:?}",
        layout.keys().collect::<Vec<_>>()
    );
    Ok((layout, elements))
}

impl BlockWeights {
    /// Base linears read into one owned buffer in `matrix_layout` order.
    pub(crate) fn resident<const LORA: bool>(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        base: &mut Source<'_>,
        lora: Option<&mut Source<'_>>,
    ) -> Result<Self> {
        let (layout, elements) = matrix_layout(base.manifest, block, &LINEAR_ROLES)?;
        let mut storage = s.alloc_zeros::<bf16>(elements)?;
        for (role, (offset, shape)) in &layout {
            let count = shape[0] as usize * shape[1] as usize;
            base.fill(
                s,
                &format!("{block}.{role}"),
                &mut storage.slice_mut(*offset..offset + count),
            )?;
        }
        let at = pointer(&storage, s);
        let mut weights =
            Self::with_linears::<LORA>(s, spec, rank, block, (at, &layout), base, lora)?;
        weights.storage = Some(storage);
        Ok(weights)
    }

    /// Base linears read from the `DenseStreaming` slot at `slot`, valid between the layer's `begin` and `end`.
    pub(crate) fn streamed<const LORA: bool>(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        slot: u64,
        base: &mut Source<'_>,
        lora: Option<&mut Source<'_>>,
    ) -> Result<Self> {
        let (layout, _) = matrix_layout(base.manifest, block, &LINEAR_ROLES)?;
        Self::with_linears::<LORA>(s, spec, rank, block, (slot, &layout), base, lora)
    }

    fn with_linears<const LORA: bool>(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        (at, layout): (u64, &MatrixLayout),
        base: &mut Source<'_>,
        lora: Option<&mut Source<'_>>,
    ) -> Result<Self> {
        let mut linears = Vec::with_capacity(LINEAR_ROLES.len());
        for role in LINEAR_ROLES {
            let (offset, shape) = layout[role];
            let bytes = 2 * u64::from(shape[0]) * u64::from(shape[1]);
            linears.push(BlockLinear::Dense {
                weight: unsafe {
                    Tensor::new(
                        at + 2 * offset as u64,
                        shape,
                        [u64::from(shape[1]), 1],
                        bytes,
                    )?
                },
                owned: None,
            });
        }
        let linears: [BlockLinear; 6] = linears
            .try_into()
            .map_err(|_| anyhow::anyhow!("block linear count"))?;
        Self::with_linears_ready::<LORA>(s, spec, rank, block, linears, base, lora)
    }

    fn with_linears_ready<const LORA: bool>(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        linears: [BlockLinear; 6],
        base: &mut Source<'_>,
        lora: Option<&mut Source<'_>>,
    ) -> Result<Self> {
        ensure!(
            LORA == lora.is_some(),
            "{block}: LoRA source does not match the block mode"
        );
        let (h, w, f) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn);
        let mut n =
            |name: &str, count: usize| base.vector(s, &format!("{block}.{name}.weight"), count);
        let (norm1, norm2) = (n("norm1", h)?, n("norm2", h)?);
        let (norm_q, norm_k) = (
            n("attn.norm_q", spec.head_dim)?,
            n("attn.norm_k", spec.head_dim)?,
        );
        let (qkv_a, q_b, k_b, v_b, out_a, out_b, in_a, in_b, ffo_a, ffo_b, rank) = if LORA {
            let lora = lora.context("LoRA block without a source")?;
            let r = rank;
            let mut qkv_a = s.alloc_zeros::<bf16>(3 * r * h)?;
            for (i, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
                lora.fill(
                    s,
                    &format!("{block}.attn.{name}.lora_A.default.weight"),
                    &mut qkv_a.slice_mut(i * r * h..(i + 1) * r * h),
                )?;
            }
            let mut l = |name: &str, factor: &str, count: usize| {
                lora.tensor(
                    s,
                    &format!("{block}.{name}.lora_{factor}.default.weight"),
                    count,
                )
            };
            (
                qkv_a,
                l("attn.to_q", "B", w * r)?,
                l("attn.to_k", "B", w * r)?,
                l("attn.to_v", "B", w * r)?,
                l("attn.to_out.0", "A", r * w)?,
                l("attn.to_out.0", "B", h * r)?,
                l("ff.net.0.proj", "A", r * h)?,
                l("ff.net.0.proj", "B", 2 * f * r)?,
                l("ff.net.2", "A", r * f)?,
                l("ff.net.2", "B", h * r)?,
                r,
            )
        } else {
            ensure!(rank == 0, "{block}: rank {rank} without LoRA");
            let empty = || s.alloc_zeros::<bf16>(0);
            (
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                empty()?,
                0,
            )
        };
        for linear in &linears {
            linear.retain_resident(s)?;
        }
        Ok(Self {
            q_b,
            k_b,
            v_b,
            out_a,
            out_b,
            in_a,
            in_b,
            ffo_a,
            ffo_b,
            norm1,
            norm2,
            norm_q,
            norm_k,
            linears,
            storage: None,
            dropped: false,
            qkv_a,
            rank,
        })
    }

    /// No checkpoint tensors. `block_forward` returns the input.
    fn identity(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        base: &mut Source<'_>,
        layer: usize,
    ) -> Result<Self> {
        let block = format!("transformer_blocks.{layer}");
        let storage = base.vector::<bf16>(s, &format!("{block}.identity"), 1)?;
        let weight = unsafe { Tensor::new(storage.tensor.ptr, [1, 1], [1, 1], 2)? };
        let linears = std::array::from_fn(|_| BlockLinear::Dense {
            weight,
            owned: None,
        });
        let empty = || s.alloc_zeros::<bf16>(0);
        Ok(Self {
            norm1: base.vector(s, &format!("{block}.norm1.weight"), spec.hidden)?,
            norm2: base.vector(s, &format!("{block}.norm2.weight"), spec.hidden)?,
            norm_q: base.vector(s, &format!("{block}.attn.norm_q.weight"), spec.head_dim)?,
            norm_k: base.vector(s, &format!("{block}.attn.norm_k.weight"), spec.head_dim)?,
            linears,
            storage: None,
            dropped: true,
            qkv_a: empty()?,
            q_b: empty()?,
            k_b: empty()?,
            v_b: empty()?,
            out_a: empty()?,
            out_b: empty()?,
            in_a: empty()?,
            in_b: empty()?,
            ffo_a: empty()?,
            ffo_b: empty()?,
            rank: 0,
        })
    }

    /// Stored HyperFlow factors. `slot` is `(base, packed offsets, allocation bytes)`; resident factors are owned.
    fn stored(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        block: &str,
        base: &mut Source<'_>,
        slot: Option<FactorSlot<'_>>,
    ) -> Result<Self> {
        let mut linears = Vec::with_capacity(LINEAR_ROLES.len());
        for role in LINEAR_ROLES {
            let suffix = role.strip_suffix(".weight").context("linear role")?;
            let name = format!("{block}.{suffix}");
            linears.push(match spec.linear_storage(&name)? {
                LinearStorage::Dense => dense_factor(s, base, slot, &format!("{name}.weight"))?,
                LinearStorage::Int8 { group } => int8_factor(s, base, slot, &name, group)?,
                LinearStorage::LowRank { rank } => low_rank_factor(s, base, slot, &name, rank)?,
            });
        }
        let linears: [BlockLinear; 6] = linears
            .try_into()
            .map_err(|_| anyhow::anyhow!("block linear count"))?;
        Self::with_linears_ready::<false>(s, spec, 0, block, linears, base, None)
    }
}

type FactorSlot<'a> = (u64, &'a [(String, usize)], usize);

fn packed_offset(packed: &[(String, usize)], name: &str) -> Result<usize> {
    packed
        .iter()
        .find(|(entry, _)| entry == name)
        .map(|(_, offset)| *offset)
        .with_context(|| format!("{name} is outside the block slot"))
}

fn view_at<T: Copy>(base: u64, offset: usize, rows: usize, cols: usize) -> Result<Tensor<T, 2>> {
    let ptr = base
        .checked_add(u64::try_from(offset)?)
        .context("stored factor address")?;
    let span = u64::try_from(rows)?
        .checked_mul(u64::try_from(cols)?)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<T>() as u64))
        .context("stored factor bytes")?;
    ensure!(
        offset.is_multiple_of(16) && ptr.is_multiple_of(16),
        "stored factor alignment"
    );
    unsafe {
        Tensor::new(
            ptr,
            [u32::try_from(rows)?, u32::try_from(cols)?],
            [u64::try_from(cols)?, 1],
            span,
        )
    }
}

fn factor_shape(base: &Source<'_>, name: &str, dtype: Dtype) -> Result<(usize, usize)> {
    let entry = base
        .manifest
        .entries
        .get(name)
        .with_context(|| format!("manifest lacks {name}"))?;
    ensure!(
        entry.dtype == dtype && entry.shape.len() == 2 && entry.load,
        "{name} stored layout"
    );
    Ok((entry.shape[0], entry.shape[1]))
}

fn dense_factor(
    s: &Arc<CudaStream>,
    base: &mut Source<'_>,
    slot: Option<FactorSlot<'_>>,
    name: &str,
) -> Result<BlockLinear> {
    let (rows, cols) = factor_shape(base, name, Dtype::BF16)?;
    if let Some((ptr, packed, _)) = slot {
        Ok(BlockLinear::Dense {
            weight: view_at(ptr, packed_offset(packed, name)?, rows, cols)?,
            owned: None,
        })
    } else {
        let owned = base.tensor::<bf16>(s, name, rows * cols)?;
        let weight = matrix(&owned, s, rows, cols)?;
        Ok(BlockLinear::Dense {
            weight,
            owned: Some(owned),
        })
    }
}

fn int8_factor(
    s: &Arc<CudaStream>,
    base: &mut Source<'_>,
    slot: Option<FactorSlot<'_>>,
    name: &str,
    group: u32,
) -> Result<BlockLinear> {
    let qname = format!("{name}.qweight");
    let (rows, cols) = factor_shape(base, &qname, Dtype::I8)?;
    let group_usize = usize::try_from(group)?;
    ensure!(
        group_usize > 0 && cols.is_multiple_of(group_usize),
        "{name} convrot group {group}"
    );
    let optin = crate::backend::cuda::optin_shared_bytes(s.context())?;
    crate::convrot::ensure_group(group, optin).with_context(|| format!("{name} convrot group"))?;
    let wscale = base.vector::<f32>(s, &format!("{name}.wscale"), rows)?;
    let (qweight, owned) = if let Some((ptr, packed, _)) = slot {
        (
            view_at(ptr, packed_offset(packed, &qname)?, rows, cols)?,
            None,
        )
    } else {
        let owned = base.tensor::<i8>(s, &qname, rows * cols)?;
        let qweight = view(
            &owned,
            s,
            0,
            [u32::try_from(rows)?, u32::try_from(cols)?],
            [u64::try_from(cols)?, 1],
        )?;
        (qweight, Some(owned))
    };
    Ok(BlockLinear::Int8 {
        qweight,
        wscale,
        group,
        owned,
    })
}

fn low_rank_overread(name: &str, rank: usize, k: usize) -> Result<(usize, usize)> {
    let pitch = crate::minimax_h3::stored_low_rank_pitch(rank);
    let bytes = pitch
        .checked_mul(k)
        .and_then(|n| n.checked_mul(2))
        .with_context(|| format!("{name} low-rank down allocation"))?;
    Ok((pitch, bytes))
}

fn low_rank_factor(
    s: &Arc<CudaStream>,
    base: &mut Source<'_>,
    slot: Option<FactorSlot<'_>>,
    name: &str,
    rank: usize,
) -> Result<BlockLinear> {
    let down_name = format!("{name}.down.weight");
    let up_name = format!("{name}.up.weight");
    let (down_rows, down_cols) = factor_shape(base, &down_name, Dtype::BF16)?;
    let (up_rows, up_cols) = factor_shape(base, &up_name, Dtype::BF16)?;
    ensure!(
        down_rows == rank && up_cols == rank && rank > 0,
        "{name} low-rank {rank}"
    );
    let (pitch, bytes) = low_rank_overread(&down_name, rank, down_cols)?;
    if let Some((ptr, packed, allocation)) = slot {
        let offset = packed_offset(packed, &down_name)?;
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= allocation),
            "{down_name}: low-rank over-read exceeds the allocation"
        );
        Ok(BlockLinear::LowRank {
            down: view_at(ptr, offset, pitch, down_cols)?,
            up: view_at(ptr, packed_offset(packed, &up_name)?, up_rows, up_cols)?,
            owned: Vec::new(),
        })
    } else {
        let elements = pitch
            .checked_mul(down_cols)
            .with_context(|| format!("{down_name} low-rank down allocation"))?;
        let stored = rank
            .checked_mul(down_cols)
            .with_context(|| format!("{down_name} low-rank down allocation"))?;
        let mut down_owned = s.alloc_zeros::<bf16>(elements)?;
        base.fill(s, &down_name, &mut down_owned.slice_mut(..stored))?;
        let base_ptr = pointer(&down_owned, s);
        let span = u64::try_from(
            down_owned
                .len()
                .checked_mul(2)
                .context("resident factor bytes")?,
        )?;
        let end = base_ptr
            .checked_add(span)
            .context("resident factor bytes")?;
        ensure!(
            base_ptr
                .checked_add(u64::try_from(bytes)?)
                .is_some_and(|stop| stop <= end),
            "{down_name}: low-rank over-read exceeds the allocation"
        );
        let down = matrix(&down_owned, s, pitch, down_cols)?;
        let up_owned = base.tensor::<bf16>(s, &up_name, up_rows * up_cols)?;
        let up = matrix(&up_owned, s, up_rows, up_cols)?;
        Ok(BlockLinear::LowRank {
            down,
            up,
            owned: vec![down_owned, up_owned],
        })
    }
}

pub(crate) struct BlockKernels {
    norm: crate::backend::gpu::CudaFunction,
    norm_f32: crate::backend::gpu::CudaFunction,
    #[cfg(test)]
    norm_modulate: crate::backend::gpu::CudaFunction,
    norm_modulate_f32: crate::backend::gpu::CudaFunction,
    rotary: crate::backend::gpu::CudaFunction,
    rotary_q: crate::backend::gpu::CudaFunction,
    rotary_kv: crate::backend::gpu::CudaFunction,
    swiglu: crate::backend::gpu::CudaFunction,
    #[cfg(test)]
    modulate: crate::backend::gpu::CudaFunction,
    #[cfg(test)]
    serial_norm_modulate: std::cell::Cell<bool>,
    gated_add: crate::backend::gpu::CudaFunction,
    gated_add_rms: crate::backend::gpu::CudaFunction,
    #[cfg(test)]
    gated_add_rms_raw: crate::backend::gpu::CudaFunction,
    residual: crate::backend::gpu::CudaFunction,
    norm_module: Arc<CudaModule>,
    modulate_module: Arc<CudaModule>,
    act_module: Arc<CudaModule>,
    rope_module: Arc<CudaModule>,
    sm: u32,
    stored_launch: std::sync::OnceLock<StoredLaunch>,
}

impl BlockKernels {
    pub(crate) fn new(ctx: &Arc<crate::backend::gpu::CudaContext>) -> Result<Self> {
        let modulate_module = cubin::module(ctx, "modulate")?;
        let norm_module = cubin::module(ctx, "norm")?;
        let act_module = cubin::module(ctx, "act")?;
        let rope_module = cubin::module(ctx, "rope")?;
        let plain = norm_module.load_function("rms_norm_bf16")?;
        let norm_f32 = norm_module.load_function("rms_norm")?;
        #[cfg(test)]
        let norm_modulate = norm_module.load_function("rms_norm_modulate_bf16")?;
        let norm_modulate_f32 = norm_module.load_function("rms_norm_modulate")?;
        let active = |kernel: &crate::backend::gpu::CudaFunction| {
            kernel.occupancy_max_active_blocks_per_multiprocessor(HIDDEN_NORM_THREADS, 0, None)
        };
        let baseline = active(&plain)?;
        #[cfg(test)]
        let fused = active(&norm_modulate)?;
        let fused_f32 = active(&norm_modulate_f32)?;
        #[cfg(test)]
        ensure!(
            baseline > 0 && fused >= baseline,
            "rms_norm_modulate_bf16 occupancy {fused} below rms_norm_bf16 {baseline}"
        );
        ensure!(
            fused_f32 >= baseline,
            "rms_norm_modulate occupancy {fused_f32} below rms_norm_bf16 {baseline}"
        );
        let gated_add_rms = norm_module.load_function("gated_add_rms_norm_modulate")?;
        let gated_active = active(&gated_add_rms)?;
        ensure!(
            gated_active >= baseline,
            "gated_add_rms_norm_modulate occupancy {gated_active} below rms_norm_bf16 {baseline}"
        );
        Ok(Self {
            norm: plain,
            norm_f32,
            #[cfg(test)]
            norm_modulate,
            norm_modulate_f32,
            residual: norm_module.load_function("residual")?,
            rotary: rope_module.load_function("rotary_prefix_kv")?,
            rotary_q: norm_module.load_function("rmsnorm_rotary_q")?,
            rotary_kv: norm_module.load_function("rmsnorm_rotary_kv")?,
            swiglu: act_module.load_function("silu_mul")?,
            #[cfg(test)]
            modulate: modulate_module.load_function("row_modulate")?,
            #[cfg(test)]
            serial_norm_modulate: std::cell::Cell::new(false),
            gated_add: modulate_module.load_function("row_gated_add")?,
            gated_add_rms,
            #[cfg(test)]
            gated_add_rms_raw: norm_module.load_function("gated_add_rms_norm_modulate_raw_var")?,
            norm_module,
            modulate_module,
            act_module,
            rope_module,
            sm: cubin::capability(ctx)?,
            stored_launch: std::sync::OnceLock::new(),
        })
    }

    fn stored_launch(
        &self,
        ctx: &std::sync::Arc<crate::backend::gpu::CudaContext>,
    ) -> Result<&StoredLaunch> {
        if let Some(launch) = self.stored_launch.get() {
            return Ok(launch);
        }
        let launch = StoredLaunch {
            upcast: cubin::module(ctx, "vision_ops")?.load_function("vision_upcast_f32")?,
            convrot: crate::convrot::load_convrot(ctx)?,
        };
        let _ = self.stored_launch.set(launch);
        self.stored_launch.get().context("stored launch")
    }
}

struct StoredLaunch {
    upcast: crate::backend::gpu::CudaFunction,
    convrot: crate::convrot::ConvrotFns,
}

/// Activations of one block forward over `rows` sequence rows.
pub(crate) struct BlockWork {
    rows: usize,
    rank: usize,
    chunk: usize,
    fixed_lora: bool,
    product: [[ProductKernel; 2]; FF_OUT + 1],
    rotate_32: [[bool; 2]; FF_OUT + 1],
    qkv_product: ProductKernel,
    #[cfg(test)]
    repeat_ffn_input: bool,
    /// The previous live block already wrote this block's norm1 into `normed`.
    skip_norm1: bool,
    /// When set, `block_forward` copies the modulated norm2 rows out right
    /// after they are written (the FFN overwrites them otherwise).
    #[cfg(test)]
    keep_normed: bool,
    #[cfg(test)]
    normed_probe: Vec<bf16>,
    normed: CudaSlice<bf16>,
    q_raw: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    lora: CudaSlice<bf16>,
    hidden: Tensor<bf16, 2>,
    gated: Tensor<bf16, 2>,
    ffn_owned: Option<(CudaSlice<bf16>, CudaSlice<bf16>)>,
    stored: Option<StoredScratch>,
}

/// Convrot and low-rank scratch for one block forward. `codes` aliases `x`; `convrot` is absent when `k == 0`.
struct StoredScratch {
    convrot: Option<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)>,
    mid: CudaSlice<bf16>,
    k: usize,
    rank: usize,
    mid_pitch: usize,
}

#[derive(Clone, Copy)]
struct StoredPtrs {
    x: u64,
    rotated: u64,
    codes: u64,
    scales: u64,
    mid: u64,
    rows: usize,
    k: usize,
    rank: usize,
    product: ProductKernel,
    mid_pitch: usize,
    rotate_32: bool,
}

impl BlockWork {
    fn plan_rotation(&mut self, device: &Device, spec: &H3Spec) -> Result<()> {
        use crate::convrot::{group_dot_32_fills_wave, group_dot_32_wave_sizes};
        if !matches!(spec.int8_convrot.as_ref(), Some((256, _))) {
            return Ok(());
        }
        let [ordinary, fused] = group_dot_32_wave_sizes(device)?;
        let h = spec.hidden;
        let width = spec.heads * spec.head_dim;
        if (h, width) == (5376, 7168) {
            self.rotate_32[..OUT].fill([group_dot_32_fills_wave(self.rows, h, ordinary); 2]);
            self.rotate_32[OUT] = [group_dot_32_fills_wave(self.rows, width, ordinary); 2];
        }
        if (h, spec.ffn) == (5376, 14336) {
            let chunks = [self.chunk, self.rows % self.chunk];
            self.rotate_32[FF_IN] = chunks.map(|rows| group_dot_32_fills_wave(rows, h, ordinary));
            self.rotate_32[FF_OUT] =
                chunks.map(|rows| group_dot_32_fills_wave(rows, spec.ffn, fused));
        }
        Ok(())
    }

    /// Binds the fixed GEMM plans of every base and LoRA-A linear shape these rows run: q/k/v, out, the stacked
    /// q/k/v LoRA A and the out LoRA A over all rows; the two FFN projections and their LoRA A over full and tail
    /// chunks.
    pub(crate) fn prepare<const LORA: bool>(&self, blas: &Blas, spec: &H3Spec) -> Result<()> {
        let (h, width, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, self.rank);
        let mut shapes = vec![(self.rows, width, h, false), (self.rows, h, width, false)];
        if LORA {
            shapes.extend([(self.rows, 3 * r, h, false), (self.rows, r, width, false)]);
        }
        for rows in [self.chunk, self.rows % self.chunk]
            .into_iter()
            .filter(|&n| n > 0)
        {
            shapes.extend([(rows, 2 * f, h, false), (rows, h, f, false)]);
            if LORA {
                shapes.extend([(rows, r, h, self.fixed_lora), (rows, r, f, self.fixed_lora)]);
                if self.fixed_lora {
                    shapes.extend([(rows, 2 * f, r, true), (rows, h, r, true)]);
                }
            }
        }
        for (rows, output, input, family) in shapes {
            if family {
                blas.prepare_row_family(crate::backend::setup::LinearShape {
                    rows,
                    output,
                    input,
                })?;
            } else {
                blas.prepare_fixed(
                    u32::try_from(rows)?,
                    u32::try_from(output)?,
                    u32::try_from(input)?,
                )?;
            }
        }
        if spec.is_hyperflow()? {
            self.prepare_low_rank(blas, spec)?;
        }
        Ok(())
    }

    fn prepare_low_rank(&self, blas: &Blas, spec: &H3Spec) -> Result<()> {
        let ranks = spec.lowrank_ranks.as_ref().context("HyperFlow rank set")?;
        for (name, &rank) in ranks {
            let suffix = name
                .split_once('.')
                .and_then(|(_, rest)| rest.split_once('.').map(|(_, suffix)| suffix))
                .with_context(|| format!("{name} low-rank suffix"))?;
            let width = spec.heads * spec.head_dim;
            let (out, input, ffn) = match suffix {
                "attn.to_q" | "attn.to_k" | "attn.to_v" => (width, spec.hidden, false),
                "attn.to_out.0" => (spec.hidden, width, false),
                "ff.net.0.proj" => (2 * spec.ffn, spec.hidden, true),
                "ff.net.2" => (spec.hidden, spec.ffn, true),
                _ => bail!("{name} is not a block linear"),
            };
            ensure!(rank > 0, "{name} low-rank 0");
            let pitch = crate::minimax_h3::stored_low_rank_pitch(rank);
            let mut row_sets = vec![self.rows];
            if ffn {
                row_sets = [self.chunk, self.rows % self.chunk]
                    .into_iter()
                    .filter(|&n| n > 0)
                    .collect();
            }
            for rows in row_sets {
                blas.prepare_fixed(
                    u32::try_from(rows)?,
                    u32::try_from(pitch)?,
                    u32::try_from(input)?,
                )?;
                if pitch == rank {
                    blas.prepare_fixed(
                        u32::try_from(rows)?,
                        u32::try_from(out)?,
                        u32::try_from(rank)?,
                    )?;
                } else {
                    blas.prepare_fixed_b_ld(
                        u32::try_from(rows)?,
                        u32::try_from(out)?,
                        u32::try_from(rank)?,
                        u32::try_from(pitch)?,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn ffn_cost<const LORA: bool>(
        &self,
        blas: &Blas,
        spec: &H3Spec,
        chunk: usize,
        describe: bool,
    ) -> Result<FfnPlan> {
        let mut plan = FfnPlan {
            chunk,
            borrowed: false,
            allocation: if chunk == self.chunk {
                0
            } else {
                FfnPlan::allocation(chunk, spec.ffn)?
            },
            workspace: 0,
            tensor_cost: 0.0,
            launches: 0,
        };
        for (rows, count) in [(chunk, self.rows / chunk), (self.rows % chunk, 1)] {
            if rows == 0 || count == 0 {
                continue;
            }
            let mut launches = 1;
            let shapes = [
                (2 * spec.ffn, spec.hidden),
                (spec.hidden, spec.ffn),
                (self.rank, spec.hidden),
                (self.rank, spec.ffn),
                (2 * spec.ffn, self.rank),
                (spec.hidden, self.rank),
            ];
            let counted = if LORA { shapes.len() } else { 2 };
            for (index, &(output, input)) in shapes[..counted].iter().enumerate() {
                let (wave_cost, reduction) = blas.fixed_plan_cost(
                    crate::backend::setup::LinearShape {
                        rows,
                        output,
                        input,
                    },
                    describe,
                    index >= 2,
                )?;
                plan.tensor_cost +=
                    2.0 * rows as f64 * output as f64 * input as f64 * count as f64 * wave_cost;
                plan.workspace = plan.workspace.max(blas.fixed_workspace(
                    crate::backend::setup::LinearShape {
                        rows,
                        output,
                        input,
                    },
                    index >= 2,
                )?);
                launches += 1 + reduction;
            }
            plan.launches += count * launches;
        }
        Ok(plan)
    }

    fn ffn_choices<const LORA: bool>(
        &self,
        blas: &Blas,
        spec: &H3Spec,
        budget: usize,
    ) -> Result<Vec<FfnPlan>> {
        let mut choices = Vec::new();
        for chunks in 1..=self.rows.div_ceil(self.chunk) {
            let chunk = self.rows.div_ceil(chunks);
            let tail = self.rows % chunk;
            if tail > 0 && tail <= crate::linear_plan::MAX_SKINNY_ROWS {
                continue;
            }
            if chunk != self.chunk && FfnPlan::allocation(chunk, spec.ffn)? > budget {
                continue;
            }
            match self.ffn_cost::<LORA>(blas, spec, chunk, false) {
                Ok(plan) => choices.push(plan),
                Err(error) if Blas::is_not_supported(&error) => {
                    if crate::backend::diagnostics_enabled() {
                        eprintln!(
                            "FFN_PLAN_REJECT rows={} chunk={chunk} error={error:#}",
                            self.rows
                        );
                    }
                }
                Err(error) => {
                    return Err(error.context(format!("FFN plan rows={} chunk={chunk}", self.rows)));
                }
            }
        }
        ensure!(
            choices
                .iter()
                .any(|p| p.chunk == self.chunk && p.allocation == 0),
            "existing FFN plan missing from workspace choices"
        );
        Ok(choices)
    }

    fn alias_choices<const LORA: bool>(
        &self,
        blas: &Blas,
        spec: &H3Spec,
        owned: FfnPlan,
    ) -> Result<Vec<FfnPlan>> {
        let capacity = FfnPlan::alias_capacity((self.q_raw.len(), self.k_raw.len()), spec.ffn)?;
        let mut choices = vec![owned];
        for chunks in 1..=self.rows.div_ceil(self.chunk) {
            let chunk = self.rows.div_ceil(chunks);
            let tail = self.rows % chunk;
            if chunk > capacity || (tail > 0 && tail <= crate::linear_plan::MAX_SKINNY_ROWS) {
                continue;
            }
            let mut plan = match self.ffn_cost::<LORA>(blas, spec, chunk, false) {
                Ok(plan) => plan,
                Err(error) if Blas::is_not_supported(&error) => {
                    if crate::backend::diagnostics_enabled() {
                        eprintln!(
                            "FFN_ALIAS_REJECT rows={} chunk={chunk} error={error:#}",
                            self.rows
                        );
                    }
                    continue;
                }
                Err(error) => {
                    return Err(
                        error.context(format!("FFN alias rows={} chunk={chunk}", self.rows))
                    );
                }
            };
            if plan.tensor_cost <= owned.tensor_cost && plan.launches <= owned.launches {
                plan.borrowed = true;
                plan.allocation = 0;
                choices.push(plan);
            }
        }
        Ok(choices)
    }

    fn bind_ffn<const LORA: bool>(
        &mut self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        spec: &H3Spec,
        plan: FfnPlan,
        x: &CudaSlice<bf16>,
    ) -> Result<()> {
        let allocation = if plan.borrowed || plan.chunk == self.chunk {
            0
        } else {
            FfnPlan::allocation(plan.chunk, spec.ffn)?
        };
        ensure!(
            plan.chunk >= self.chunk && plan.chunk <= self.rows && plan.allocation == allocation,
            "FFN binding differs from the counted allocation"
        );
        if plan.borrowed {
            self.hidden = rows_view(&self.q_raw, s, 0, plan.chunk, 2 * spec.ffn, 2 * spec.ffn)?;
            self.gated = rows_view(&self.k_raw, s, 0, plan.chunk, spec.ffn, spec.ffn)?;
            self.ffn_owned = None;
        } else if plan.chunk != self.chunk {
            let hidden = s.alloc_zeros::<bf16>(plan.chunk * 2 * spec.ffn)?;
            let gated = s.alloc_zeros::<bf16>(plan.chunk * spec.ffn)?;
            self.hidden = matrix(&hidden, s, plan.chunk, 2 * spec.ffn)?;
            self.gated = matrix(&gated, s, plan.chunk, spec.ffn)?;
            self.ffn_owned = Some((hidden, gated));
        }
        crate::backend::cuda::contracts::no_overlap(
            &self.hidden,
            &self.gated,
            "FFN hidden",
            "FFN gated",
        )?;
        let hidden = &self.hidden;
        let gated = &self.gated;
        let overlap = |name: &str, buffer: &CudaSlice<bf16>| -> Result<()> {
            let protected = matrix(buffer, s, 1, buffer.len())?;
            crate::backend::cuda::contracts::no_overlap(hidden, &protected, "FFN hidden", name)?;
            crate::backend::cuda::contracts::no_overlap(gated, &protected, "FFN gated", name)
        };
        overlap("normed", &self.normed)?;
        if LORA {
            overlap("LoRA", &self.lora)?;
        }
        overlap("residual", x)?;
        self.chunk = plan.chunk;
        ensure!(
            self.hidden.ptr.is_multiple_of(256) && self.gated.ptr.is_multiple_of(256),
            "FFN view alignment"
        );
        self.fixed_lora = true;
        self.prepare::<LORA>(blas, spec)?;
        for rows in [self.chunk, self.rows % self.chunk]
            .into_iter()
            .filter(|&rows| rows > 0)
        {
            blas.bind_lower_wave(crate::backend::setup::LinearShape {
                rows,
                output: spec.hidden,
                input: spec.ffn,
            })?;
        }
        if self.rows > FFN_CHUNK_ROWS {
            self.ffn_cost::<LORA>(blas, spec, self.chunk, true)?;
        }
        Ok(())
    }

    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        rows: usize,
    ) -> Result<Self> {
        launch_extents(spec, rows)?;
        let width = spec.heads * spec.head_dim;
        let chunk = ffn_chunk(rows);
        let attention = || s.alloc_zeros::<bf16>(rows * width);
        let normed = s.alloc_zeros::<bf16>(rows * spec.hidden)?;
        let (q_raw, k_raw, v_raw, q, k) = (
            attention()?,
            attention()?,
            attention()?,
            attention()?,
            attention()?,
        );
        let lora = s.alloc_zeros::<bf16>(rows * 3 * rank)?;
        let hidden = s.alloc_zeros::<bf16>(chunk * 2 * spec.ffn)?;
        let gated = s.alloc_zeros::<bf16>(chunk * spec.ffn)?;
        let hidden_view = matrix(&hidden, s, chunk, 2 * spec.ffn)?;
        let gated_view = matrix(&gated, s, chunk, spec.ffn)?;
        let stored = if let Some((k, low_rank)) = crate::minimax_h3::stored_scratch_extents(spec)? {
            let pitch = crate::minimax_h3::stored_low_rank_pitch(low_rank);
            let convrot = if k > 0 {
                Some((
                    s.alloc_zeros::<f32>(rows * k)?,
                    s.alloc_zeros::<f32>(rows * k)?,
                    s.alloc_zeros::<f32>(rows)?,
                ))
            } else {
                None
            };
            Some(StoredScratch {
                convrot,
                mid: s.alloc_zeros::<bf16>(
                    rows.checked_mul(pitch).context("low-rank scratch rows")?,
                )?,
                k,
                rank: low_rank,
                mid_pitch: pitch,
            })
        } else {
            None
        };
        Ok(Self {
            rows,
            rank,
            chunk,
            #[cfg(test)]
            keep_normed: false,
            #[cfg(test)]
            normed_probe: Vec::new(),
            fixed_lora: false,
            product: [[ProductKernel::NFast; 2]; FF_OUT + 1],
            rotate_32: [[false; 2]; FF_OUT + 1],
            qkv_product: ProductKernel::NFast,
            #[cfg(test)]
            repeat_ffn_input: false,
            skip_norm1: false,
            normed,
            q_raw,
            k_raw,
            v_raw,
            q,
            k,
            lora,
            hidden: hidden_view,
            gated: gated_view,
            ffn_owned: Some((hidden, gated)),
            stored,
        })
    }
}

/// Transformer blocks read AdaLN rows and rotate the first `6 * rope_freq_dim` channels of q and k; token refiner
/// blocks do neither and add both residuals unmodulated.
pub(crate) enum Conditioning<'a> {
    Modulated {
        rotary: &'a CudaSlice<bf16>,
        modulation: &'a Modulation<'a>,
    },
    Plain,
}

/// Block modulation: `entries` rows of `6 * hidden` (shift, scale, gate of attention, then of the FFN) starting
/// `offset` elements into `table`, and the entry of every sequence row.
pub(crate) struct Modulation<'a> {
    pub table: &'a CudaSlice<bf16>,
    pub offset: usize,
    pub entries: u32,
    pub index: &'a CudaSlice<u32>,
}

fn pointer<T>(b: &CudaSlice<T>, s: &Arc<CudaStream>) -> u64 {
    b.device_ptr(s).0
}

fn rows_view<T: crate::backend::gpu::DeviceRepr + Copy>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    width: usize,
    pitch: usize,
) -> Result<Tensor<T, 2>> {
    view(
        b,
        s,
        offset,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [pitch as u64, 1],
    )
}

/// Stream, BLAS, kernels, geometry and LoRA scale shared by every block launch.
pub(crate) struct BlockContext<'a> {
    pub s: &'a Arc<CudaStream>,
    pub blas: &'a Blas,
    pub kernels: &'a BlockKernels,
    pub spec: &'a H3Spec,
    pub lora_scale: f32,
}

/// `y += lora_scale * (x * a^T) * b^T`; the intermediate row pitch is `rank`.
fn lora_pair(
    c: &BlockContext<'_>,
    x: Tensor<bf16, 2>,
    (a, b): (&CudaSlice<bf16>, &CudaSlice<bf16>),
    work: &mut CudaSlice<bf16>,
    y: Tensor<bf16, 2>,
    fixed: bool,
) -> Result<()> {
    let input = x.shape[1] as usize;
    let output = y.shape[1] as usize;
    let rows = x.shape[0] as usize;
    ensure!(a.len().is_multiple_of(input), "LoRA-A extent");
    let rank = a.len() / input;
    let aw = matrix(a, c.s, rank, input)?;
    let mut ay = TensorMut(rows_view(work, c.s, 0, rows, rank, rank)?);
    if fixed {
        c.blas.linear_row_family(aw, x, &mut ay, 1.0, 0.0)?;
    } else {
        c.blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(aw),
            x,
            &mut ay,
        )?;
    }
    let bw = matrix(b, c.s, output, rank)?;
    let bx = rows_view(work, c.s, 0, rows, rank, rank)?;
    let mut by = TensorMut(y);
    if fixed {
        c.blas.linear_row_family(bw, bx, &mut by, c.lora_scale, 1.0)
    } else {
        c.blas.linear_accumulate(bw, bx, &mut by, c.lora_scale)
    }
}

fn stored_ptrs(s: &Arc<CudaStream>, work: &BlockWork) -> Option<StoredPtrs> {
    let scratch = work.stored.as_ref()?;
    let (x, rotated, scales) = match scratch.convrot.as_ref() {
        Some((x, rotated, scales)) => (pointer(x, s), pointer(rotated, s), pointer(scales, s)),
        None => (0, 0, 0),
    };
    Some(StoredPtrs {
        x,
        rotated,
        codes: x,
        scales,
        mid: pointer(&scratch.mid, s),
        rows: work.rows,
        k: scratch.k,
        rank: scratch.rank,
        product: ProductKernel::NFast,
        mid_pitch: scratch.mid_pitch,
        rotate_32: false,
    })
}

fn resident_span<T>(owned: &CudaSlice<T>, s: &Arc<CudaStream>, view: &Tensor<T, 2>) -> Result<()> {
    let base = pointer(owned, s);
    let bytes = u64::try_from(owned.len())?
        .checked_mul(std::mem::size_of::<T>() as u64)
        .context("resident factor bytes")?;
    let end = base.checked_add(bytes).context("resident factor bytes")?;
    let stop = view
        .ptr
        .checked_add(view.span_bytes)
        .context("resident factor span")?;
    ensure!(
        view.ptr >= base && stop <= end,
        "resident factor leaves its allocation"
    );
    Ok(())
}

impl BlockLinear {
    /// A resident factor's view addresses the allocation in `owned`.
    fn retain_resident(&self, s: &Arc<CudaStream>) -> Result<()> {
        match self {
            BlockLinear::Dense {
                weight,
                owned: Some(slice),
            } => resident_span(slice, s, weight),
            BlockLinear::Dense { owned: None, .. } => Ok(()),
            BlockLinear::Int8 {
                qweight,
                owned: Some(slice),
                ..
            } => resident_span(slice, s, qweight),
            BlockLinear::Int8 { owned: None, .. } => Ok(()),
            BlockLinear::LowRank { down, up, owned } if owned.len() == 2 => {
                resident_span(&owned[0], s, down)?;
                resident_span(&owned[1], s, up)
            }
            BlockLinear::LowRank { owned, .. } => {
                ensure!(owned.is_empty(), "low-rank resident factors");
                Ok(())
            }
        }
    }
}

fn apply_linear(
    c: &BlockContext<'_>,
    linear: &BlockLinear,
    x: Tensor<bf16, 2>,
    y: Tensor<bf16, 2>,
    scratch: Option<StoredPtrs>,
) -> Result<()> {
    match linear {
        BlockLinear::Dense { weight, .. } => c.blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(*weight),
            x,
            &mut TensorMut(y),
        ),
        BlockLinear::Int8 {
            qweight,
            wscale,
            group,
            ..
        } => apply_int8(c, (*qweight, wscale.tensor, *group), x, y, scratch, false),
        BlockLinear::LowRank { down, up, .. } => apply_low_rank(c, *down, *up, x, y, scratch),
    }
}

fn apply_int8(
    c: &BlockContext<'_>,
    (qweight, wscale, group): (Tensor<i8, 2>, Tensor<f32, 1>, u32),
    x: Tensor<bf16, 2>,
    y: Tensor<bf16, 2>,
    scratch: Option<StoredPtrs>,
    fuse_silu: bool,
) -> Result<()> {
    let ptrs = scratch.context("stored linear without scratch")?;
    let rows = x.shape[0] as usize;
    let k = qweight.shape[1] as usize;
    let n = y.shape[1] as usize;
    ensure!(
        qweight.shape == [u32::try_from(n)?, u32::try_from(k)?]
            && wscale.shape == [u32::try_from(n)?]
            && x.shape[1] as usize == k * (1 + usize::from(fuse_silu))
            && y.shape[0] as usize == rows
            && rows <= ptrs.rows
            && k <= ptrs.k,
        "convrot shape"
    );
    if fuse_silu {
        ensure!(
            rows >= crate::convrot::TENSOR_CORE_ROWS
                && x.stride == [u64::from(x.shape[1]), 1]
                && k.is_multiple_of(8),
            "fused ConvRot input extent"
        );
        crate::backend::cuda::contracts::no_overlap(&x, &y, "gate/up", "ConvRot output")?;
    }
    let launch = c.kernels.stored_launch(c.s.context())?;
    // group_dot reads bf16. linear reads the upcast f32.
    let x_ptr = if rows >= crate::convrot::TENSOR_CORE_ROWS {
        x.ptr
    } else {
        let count = rows * k;
        unsafe {
            c.s.launch_builder(&launch.upcast)
                .arg(&x.ptr)
                .arg(&ptrs.x)
                .arg(&u32::try_from(count)?)
                .launch(flat(count))?;
        }
        ptrs.x
    };
    crate::convrot::launch_device(
        c.s,
        &launch.convrot,
        &crate::convrot::ConvrotArgs {
            x: x_ptr,
            qweight: qweight.ptr,
            wscale: wscale.ptr,
            y: y.ptr,
            rotated: ptrs.rotated,
            codes: ptrs.codes,
            scales: ptrs.scales,
            rows: i32::try_from(rows)?,
            k: i32::try_from(k)?,
            n: i32::try_from(n)?,
            group: i32::try_from(group)?,
            product: ptrs.product,
            fuse_silu,
            rotate_32: ptrs.rotate_32,
        },
    )
}

fn int8_linear(linear: &BlockLinear) -> Option<(Tensor<i8, 2>, Tensor<f32, 1>, u32)> {
    match linear {
        BlockLinear::Int8 {
            qweight,
            wscale,
            group,
            ..
        } => Some((*qweight, wscale.tensor, *group)),
        _ => None,
    }
}

/// Int8 Q, K, and V at or above [`crate::convrot::TENSOR_CORE_ROWS`] share one rotate-quant of the bf16 norm.
fn apply_shared_qkv(
    c: &BlockContext<'_>,
    w: &BlockWeights,
    work: &BlockWork,
    scratch: Option<StoredPtrs>,
) -> Result<bool> {
    let Some(q) = int8_linear(&w.linears[Q]) else {
        return Ok(false);
    };
    let Some(k) = int8_linear(&w.linears[K]) else {
        return Ok(false);
    };
    let Some(v) = int8_linear(&w.linears[V]) else {
        return Ok(false);
    };
    let rows = work.rows;
    if rows < crate::convrot::TENSOR_CORE_ROWS {
        return Ok(false);
    }
    ensure!(
        q.0.shape[1] == k.0.shape[1] && k.0.shape[1] == v.0.shape[1] && q.2 == k.2 && k.2 == v.2,
        "convrot shared activation"
    );
    let ptrs = scratch.context("stored linear without scratch")?;
    let h = c.spec.hidden;
    let width = c.spec.heads * c.spec.head_dim;
    let kdim = usize::try_from(q.0.shape[1])?;
    let n = usize::try_from(q.0.shape[0])?;
    ensure!(
        k.0.shape[0] == q.0.shape[0]
            && v.0.shape[0] == q.0.shape[0]
            && kdim == h
            && n == width
            && q.1.shape[0] == q.0.shape[0]
            && k.1.shape[0] == k.0.shape[0]
            && v.1.shape[0] == v.0.shape[0]
            && rows <= ptrs.rows
            && kdim <= ptrs.k,
        "convrot shape"
    );
    let launch = c.kernels.stored_launch(c.s.context())?;
    let x = rows_view(&work.normed, c.s, 0, rows, h, h)?;
    let group = i32::try_from(q.2)?;
    let rows_i = i32::try_from(rows)?;
    let k_i = i32::try_from(kdim)?;
    crate::convrot::launch_rotate_quant(
        c.s,
        &launch.convrot,
        &crate::convrot::ConvrotArgs {
            x: x.ptr,
            qweight: 0,
            wscale: 0,
            y: 0,
            rotated: ptrs.rotated,
            codes: ptrs.codes,
            scales: ptrs.scales,
            rows: rows_i,
            k: k_i,
            n: 0,
            group,
            product: ProductKernel::NFast,
            fuse_silu: false,
            rotate_32: work.rotate_32[Q][0],
        },
    )?;
    for ((qweight, wscale, _), out) in [(q.0, q.1, q.2), (k.0, k.1, k.2), (v.0, v.1, v.2)]
        .into_iter()
        .zip([&work.q_raw, &work.k_raw, &work.v_raw])
    {
        let y = rows_view(out, c.s, 0, rows, width, width)?;
        crate::convrot::launch_product(
            c.s,
            &launch.convrot,
            &crate::convrot::ConvrotArgs {
                x: ptrs.x,
                qweight: qweight.ptr,
                wscale: wscale.ptr,
                y: y.ptr,
                rotated: ptrs.rotated,
                codes: ptrs.codes,
                scales: ptrs.scales,
                rows: rows_i,
                k: k_i,
                n: i32::try_from(n)?,
                group,
                product: work.qkv_product,
                fuse_silu: false,
                rotate_32: false,
            },
        )?;
    }
    Ok(true)
}

fn apply_low_rank(
    c: &BlockContext<'_>,
    down: Tensor<bf16, 2>,
    up: Tensor<bf16, 2>,
    x: Tensor<bf16, 2>,
    y: Tensor<bf16, 2>,
    scratch: Option<StoredPtrs>,
) -> Result<()> {
    let ptrs = scratch.context("stored linear without scratch")?;
    let rows = x.shape[0] as usize;
    let k = x.shape[1] as usize;
    let n = y.shape[1] as usize;
    let rank = up.shape[1] as usize;
    let pitch = down.shape[0] as usize;
    ensure!(
        pitch == crate::minimax_h3::stored_low_rank_pitch(rank)
            && down.shape[1] as usize == k
            && up.shape[0] as usize == n
            && y.shape[0] as usize == rows
            && rows <= ptrs.rows
            && rank > 0
            && rank <= ptrs.rank
            && pitch <= ptrs.mid_pitch,
        "low-rank shape"
    );
    let span = |cols: usize| -> Result<u64> {
        u64::try_from(rows)?
            .checked_mul(u64::try_from(cols)?)
            .and_then(|elements| elements.checked_mul(2))
            .context("low-rank scratch bytes")
    };
    let mid = unsafe {
        Tensor::new(
            ptrs.mid,
            [u32::try_from(rows)?, u32::try_from(pitch)?],
            [u64::try_from(pitch)?, 1],
            span(pitch)?,
        )?
    };
    c.blas.linear_typed(
        crate::op_contract::ExecutionPhase::Prefill,
        crate::backend::cuda::linear::LinearWeightRef::Bf16(down),
        x,
        &mut TensorMut(mid),
    )?;
    let up_x = if pitch == rank {
        mid
    } else {
        unsafe {
            Tensor::new(
                ptrs.mid,
                [u32::try_from(rows)?, u32::try_from(rank)?],
                [u64::try_from(pitch)?, 1],
                span(pitch)?,
            )?
        }
    };
    if pitch == rank {
        c.blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(up),
            up_x,
            &mut TensorMut(y),
        )
    } else {
        c.blas.linear_bf16_b_ld(up, up_x, &mut TensorMut(y), pitch)
    }
}

/// One MiniMax-H3 transformer or token refiner block over `x [rows, hidden]` in place, BF16. The `LORA` monomorph
/// applies LoRA online at `c.lora_scale`; the other monomorph launches no LoRA GEMM. A modulated block's `rotary`
/// is `[rows, 6 * rope_freq_dim]` from `rotary_table`. A dropped block returns `x`.
pub(crate) fn block_forward<const LORA: bool>(
    c: &BlockContext<'_>,
    flash: &mut Flash,
    w: &BlockWeights,
    conditioning: &Conditioning<'_>,
    x: &mut CudaSlice<bf16>,
    work: &mut BlockWork,
) -> Result<()> {
    block_forward_tail::<LORA>(c, flash, w, conditioning, x, work, None)
}

fn block_forward_tail<const LORA: bool>(
    c: &BlockContext<'_>,
    flash: &mut Flash,
    w: &BlockWeights,
    conditioning: &Conditioning<'_>,
    x: &mut CudaSlice<bf16>,
    work: &mut BlockWork,
    tail: Option<NormTail>,
) -> Result<()> {
    let (s, blas, kernels, spec, lora_scale) = (c.s, c.blas, c.kernels, c.spec, c.lora_scale);
    let rows = work.rows;
    let (h, width, r) = (spec.hidden, spec.heads * spec.head_dim, w.rank);
    let (rotary_width, rotary_table, modulation) = match conditioning {
        Conditioning::Modulated { rotary, modulation } => {
            ensure!(
                rotary.len() == rows * 6 * spec.rope_freq_dim && modulation.index.len() == rows,
                "conditioning extents differ from the work rows"
            );
            ensure!(
                modulation.offset + modulation.entries as usize * 6 * h <= modulation.table.len(),
                "modulation table extent"
            );
            (6 * spec.rope_freq_dim, pointer(rotary, s), Some(modulation))
        }
        Conditioning::Plain => (0, pointer(&work.q, s), None),
    };
    ensure!(x.len() == rows * h, "block rows differ from the work rows");
    if w.dropped {
        return Ok(());
    }
    let skip_norm1 = std::mem::replace(&mut work.skip_norm1, false);
    #[cfg(test)]
    let serial_norm = kernels.serial_norm_modulate.get();
    #[cfg(not(test))]
    let serial_norm = false;
    let stride = 6 * h;
    let (table, index, entries) = match modulation {
        Some(m) => (
            pointer(m.table, s) + 2 * m.offset as u64,
            pointer(m.index, s),
            m.entries,
        ),
        None => (0, 0, 0),
    };
    let chunk = |c: usize| table + (2 * c * h) as u64;
    let rows_i = i32::try_from(rows)?;
    let row_grid = LaunchConfig {
        grid_dim: (u32::try_from(rows)?, u32::try_from(h.div_ceil(256))?, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let xp = pointer(x, s);
    let normed = pointer(&work.normed, s);
    let hidden_norm = |weight: &VectorWeight<bf16>, shift: usize, scale: usize| -> Result<()> {
        let n = NormRows {
            input: xp,
            weight: weight.tensor.ptr,
            output: normed,
            rows,
            dim: h,
            threads: HIDDEN_NORM_THREADS,
            eps: spec.norm_eps,
        };
        if modulation.is_some() {
            let m = ModulateRows {
                data: normed,
                shift: chunk(shift),
                scale: chunk(scale),
                index,
                rows,
                dim: h,
                stride,
                entries,
            };
            kernels.rms_norm_modulate_f32(s, n, m)
        } else {
            kernels.rms_norm_f32(s, n)
        }
    };
    let gated_add = |residual: u64, update: u64, gate: usize| -> Result<()> {
        kernels.prepare_fp16_store(s, &kernels.modulate_module)?;
        unsafe {
            s.launch_builder(&kernels.gated_add)
                .arg(&residual)
                .arg(&update)
                .arg(&chunk(gate))
                .arg(&index)
                .arg(&rows_i)
                .arg(&i32::try_from(h)?)
                .arg(&(stride as u64))
                .arg(&entries)
                .launch(row_grid)?;
        }
        kernels.finish_fp16_store(s, &kernels.modulate_module, "gated_add")
    };
    let residual = |residual: u64, update: u64| -> Result<()> {
        kernels.prepare_fp16_store(s, &kernels.norm_module)?;
        unsafe {
            s.launch_builder(&kernels.residual)
                .arg(&residual)
                .arg(&update)
                .arg(&i32::try_from(rows * h)?)
                .launch(flat(rows * h / 8))?;
        }
        kernels.finish_fp16_store(s, &kernels.norm_module, "residual")
    };
    let scratch = stored_ptrs(s, work);
    let base = |index: usize, x: Tensor<bf16, 2>, y: Tensor<bf16, 2>| -> Result<()> {
        let scratch = scratch.map(|mut ptrs| {
            ptrs.product = work.product[index][0];
            ptrs.rotate_32 = work.rotate_32[index][0];
            ptrs
        });
        apply_linear(c, &w.linears[index], x, y, scratch)
    };
    if skip_norm1 {
        ensure!(
            modulation.is_some(),
            "skipped norm1 on an unmodulated block"
        );
    } else {
        hidden_norm(&w.norm1, 0, 1)?;
    }
    if !apply_shared_qkv(c, w, work, scratch)? {
        for (index, out) in [(Q, &work.q_raw), (K, &work.k_raw), (V, &work.v_raw)] {
            base(
                index,
                rows_view(&work.normed, s, 0, rows, h, h)?,
                rows_view(out, s, 0, rows, width, width)?,
            )?;
        }
    }
    if LORA {
        blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(&w.qkv_a, s, 3 * r, h)?),
            rows_view(&work.normed, s, 0, rows, h, h)?,
            &mut TensorMut(rows_view(&work.lora, s, 0, rows, 3 * r, 3 * r)?),
        )?;
        for (i, (b, out)) in [
            (&w.q_b, &work.q_raw),
            (&w.k_b, &work.k_raw),
            (&w.v_b, &work.v_raw),
        ]
        .into_iter()
        .enumerate()
        {
            blas.linear_accumulate(
                matrix(b, s, width, r)?,
                rows_view(&work.lora, s, i * r, rows, r, 3 * r)?,
                &mut TensorMut(matrix(out, s, rows, width)?),
                lora_scale,
            )?;
        }
    }
    let head_rows = rows * spec.heads;
    let qk_rope_cfg = LaunchConfig {
        grid_dim: (u32::try_from(head_rows.div_ceil(QK_ROPE_WARPS))?, 1, 1),
        block_dim: (QK_ROPE_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    kernels.prepare_fp16_store(s, &kernels.norm_module)?;
    unsafe {
        s.launch_builder(&kernels.rotary_q)
            .arg(&pointer(&work.q_raw, s))
            .arg(&w.norm_q.tensor.ptr)
            .arg(&pointer(&work.q, s))
            .arg(&rotary_table)
            .arg(&0i32)
            .arg(&rows_i)
            .arg(&i32::try_from(spec.heads)?)
            .arg(&i32::try_from(spec.head_dim)?)
            .arg(&i32::try_from(rotary_width)?)
            .arg(&rows_i)
            .arg(&spec.qk_norm_eps)
            .launch(qk_rope_cfg)?;
    }
    kernels.finish_fp16_store(s, &kernels.norm_module, "q")?;
    kernels.prepare_fp16_store(s, &kernels.norm_module)?;
    unsafe {
        s.launch_builder(&kernels.rotary_kv)
            .arg(&pointer(&work.k_raw, s))
            .arg(&w.norm_k.tensor.ptr)
            .arg(&pointer(&work.v_raw, s))
            .arg(&pointer(&work.k, s))
            .arg(&pointer(&work.q_raw, s))
            .arg(&rotary_table)
            .arg(&0i32)
            .arg(&rows_i)
            .arg(&i32::try_from(spec.heads)?)
            .arg(&i32::try_from(spec.head_dim)?)
            .arg(&i32::try_from(rotary_width)?)
            .arg(&rows_i)
            .arg(&spec.qk_norm_eps)
            .launch(qk_rope_cfg)?;
    }
    kernels.finish_fp16_store(s, &kernels.norm_module, "k")?;
    flash.run(
        &work.q,
        width,
        &work.k,
        &work.q_raw,
        &mut work.v_raw,
        Shape {
            rows,
            start: 0,
            q_heads: spec.heads,
            kv_heads: spec.heads,
            capacity: rows,
            head_dim: spec.head_dim,
            causal: false,
            window_left: None,
        },
    )?;
    base(
        OUT,
        rows_view(&work.v_raw, s, 0, rows, width, width)?,
        rows_view(&work.normed, s, 0, rows, h, h)?,
    )?;
    if LORA {
        lora_pair(
            c,
            rows_view(&work.v_raw, s, 0, rows, width, width)?,
            (&w.out_a, &w.out_b),
            &mut work.lora,
            rows_view(&work.normed, s, 0, rows, h, h)?,
            false,
        )?;
    }
    let fuse_norm = |gate_chunk: usize, tail: NormTail| -> Result<()> {
        kernels.gated_add_rms(
            s,
            &GatedAddRms {
                x: xp,
                y: normed,
                gate: chunk(gate_chunk),
                gate_index: index,
                gate_stride: stride,
                gate_entries: entries,
                weight: tail.weight,
                output: normed,
                shift: tail.shift,
                scale: tail.scale,
                norm_index: tail.index,
                norm_stride: tail.stride,
                norm_entries: tail.entries,
                rows,
                dim: h,
                eps: tail.eps,
            },
        )
    };
    if modulation.is_some() && !serial_norm {
        fuse_norm(
            2,
            NormTail {
                weight: w.norm2.tensor.ptr,
                shift: chunk(3),
                scale: chunk(4),
                index,
                stride,
                entries,
                eps: spec.norm_eps,
                arms_norm1: false,
            },
        )?;
    } else if modulation.is_some() {
        gated_add(xp, normed, 2)?;
        hidden_norm(&w.norm2, 3, 4)?;
    } else {
        residual(xp, normed)?;
        hidden_norm(&w.norm2, 3, 4)?;
    }
    #[cfg(test)]
    if work.keep_normed {
        work.normed_probe = s.clone_dtoh(&work.normed)?;
    }
    ffn_forward::<false, LORA>(c, w, work, scratch)?;
    if let Some(tail) = tail.filter(|_| modulation.is_some() && !serial_norm) {
        let arms_norm1 = tail.arms_norm1;
        fuse_norm(5, tail)?;
        work.skip_norm1 = arms_norm1;
    } else if modulation.is_some() {
        gated_add(xp, normed, 5)?;
    } else {
        residual(xp, normed)?;
    }
    Ok(())
}

/// FFN rows overwrite their input after the gated projection; scratch is reused between chunks.
fn ffn_forward<const SHIFT_INPUT: bool, const LORA: bool>(
    c: &BlockContext<'_>,
    w: &BlockWeights,
    work: &mut BlockWork,
    scratch: Option<StoredPtrs>,
) -> Result<()> {
    let (s, h, f, rows) = (c.s, c.spec.hidden, c.spec.ffn, work.rows);
    let chunk = work.chunk;
    let products = &work.product;
    let rotate_32 = work.rotate_32;
    let fused_down = if !LORA && !SHIFT_INPUT {
        int8_linear(&w.linears[FF_OUT])
    } else {
        None
    };
    let base = |index: usize, x: Tensor<bf16, 2>, y: Tensor<bf16, 2>| -> Result<()> {
        let scratch = scratch.map(|mut ptrs| {
            let tail = usize::from(x.shape[0] as usize != chunk);
            ptrs.product = products[index][tail];
            ptrs.rotate_32 = index == FF_IN && rotate_32[FF_IN][tail];
            ptrs
        });
        apply_linear(c, &w.linears[index], x, y, scratch)
    };
    for start in (0..rows).step_by(chunk) {
        let n = chunk.min(rows - start);
        let input_start = start;
        #[cfg(test)]
        let input_start = if work.repeat_ffn_input {
            0
        } else {
            input_start
        };
        base(
            FF_IN,
            rows_view(&work.normed, s, input_start * h, n, h, h)?,
            work.hidden.prefix_rows(u32::try_from(n)?)?,
        )?;
        if LORA {
            lora_pair(
                c,
                rows_view(&work.normed, s, input_start * h, n, h, h)?,
                (&w.in_a, &w.in_b),
                &mut work.lora,
                work.hidden.prefix_rows(u32::try_from(n)?)?,
                work.fixed_lora,
            )?;
        }
        let output = rows_view(&work.normed, s, start * h, n, h, h)?;
        if let Some(weights) = fused_down
            && n >= crate::convrot::TENSOR_CORE_ROWS
        {
            let scratch = scratch.map(|mut ptrs| {
                let tail = usize::from(n != chunk);
                ptrs.product = products[FF_OUT][tail];
                ptrs.rotate_32 = rotate_32[FF_OUT][tail];
                ptrs
            });
            apply_int8(
                c,
                weights,
                work.hidden.prefix_rows(u32::try_from(n)?)?,
                output,
                scratch,
                true,
            )?;
            continue;
        }
        let hidden = work.hidden.ptr + if SHIFT_INPUT { 256 } else { 0 };
        let gated = work.gated.ptr;
        let cfg = ops::silu_mul_grid(n, f, hidden, gated)?;
        c.kernels.prepare_fp16_store(s, &c.kernels.act_module)?;
        unsafe {
            s.launch_builder(&c.kernels.swiglu)
                .arg(&hidden)
                .arg(&gated)
                .arg(&i32::try_from(n)?)
                .arg(&i32::try_from(f)?)
                .arg(&3i32)
                .launch(cfg)?;
        }
        c.kernels
            .finish_fp16_store(s, &c.kernels.act_module, "ffn")?;
        base(FF_OUT, work.gated.prefix_rows(u32::try_from(n)?)?, output)?;
        if LORA {
            lora_pair(
                c,
                work.gated.prefix_rows(u32::try_from(n)?)?,
                (&w.ffo_a, &w.ffo_b),
                &mut work.lora,
                rows_view(&work.normed, s, start * h, n, h, h)?,
                work.fixed_lora,
            )?;
        }
    }
    Ok(())
}

/// One RMSNorm launch: `rows` rows of `dim` elements, one block of `threads` per row.
struct NormRows {
    input: u64,
    weight: u64,
    output: u64,
    rows: usize,
    dim: usize,
    threads: u32,
    eps: f32,
}

/// Gated residual plus the RMSNorm that reads the stored BF16 row. `output` may alias `y`.
struct GatedAddRms {
    x: u64,
    y: u64,
    gate: u64,
    gate_index: u64,
    gate_stride: usize,
    gate_entries: u32,
    weight: u64,
    output: u64,
    shift: u64,
    scale: u64,
    norm_index: u64,
    norm_stride: usize,
    norm_entries: u32,
    rows: usize,
    dim: usize,
    eps: f32,
}

/// Norm applied to the gated residual: the next live block's norm1, or `norm_out` on the last live block.
#[derive(Clone, Copy)]
struct NormTail {
    weight: u64,
    shift: u64,
    scale: u64,
    index: u64,
    stride: usize,
    entries: u32,
    eps: f32,
    arms_norm1: bool,
}

struct Scale2Sources {
    hidden: usize,
    block_rows: usize,
    pairs: u32,
    timesteps: u32,
    norm_eps: f32,
    final_eps: f32,
    table: u64,
    block_table_len: usize,
    norm_out_weight: u64,
    norm_out_table: u64,
    norm_table_len: usize,
    row_pairs: u64,
    row_timesteps: u64,
}

/// One `row_modulate` launch over dense `[rows, dim]` data with table rows of `stride` elements.
struct ModulateRows {
    data: u64,
    shift: u64,
    scale: u64,
    index: u64,
    rows: usize,
    dim: usize,
    stride: usize,
    entries: u32,
}

impl BlockKernels {
    // sm >= 80 cubins do not define ff_fp16_store_fault.
    fn prepare_fp16_store(&self, s: &Arc<CudaStream>, module: &Arc<CudaModule>) -> Result<()> {
        if self.sm >= 80 {
            return Ok(());
        }
        let mut symbol = module.get_global("ff_fp16_store_fault", s)?;
        ensure!(
            symbol.len() == 4,
            "ff_fp16_store_fault is {} bytes",
            symbol.len()
        );
        s.memcpy_htod(&[0u8; 4], &mut symbol)?;
        Ok(())
    }

    fn finish_fp16_store(
        &self,
        s: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        tensor: &str,
    ) -> Result<()> {
        if self.sm >= 80 {
            return Ok(());
        }
        let mut symbol = module.get_global("ff_fp16_store_fault", s)?;
        ensure!(
            symbol.len() == 4,
            "ff_fp16_store_fault is {} bytes",
            symbol.len()
        );
        let bytes = s.clone_dtoh(&symbol)?;
        s.memcpy_htod(&[0u8; 4], &mut symbol)?;
        ensure!(
            bytes == [0u8; 4],
            "FP16 store of {tensor} is non-finite or above 65504"
        );
        Ok(())
    }

    #[cfg(test)]
    fn rms_norm_modulate(&self, s: &Arc<CudaStream>, n: NormRows, m: ModulateRows) -> Result<()> {
        self.prepare_norm_modulate(&n, &m)?;
        self.launch_norm_modulate(s, &self.norm_modulate, &n, &m)
    }

    /// Weight multiplies in f32. The modulate stores keep the four BF16 rounds.
    /// The test-only serial flag swaps the fused launch for the separate
    /// f32 norm and row modulation, so the identity gates compare distinct
    /// paths.
    fn rms_norm_modulate_f32(
        &self,
        s: &Arc<CudaStream>,
        n: NormRows,
        m: ModulateRows,
    ) -> Result<()> {
        self.prepare_norm_modulate(&n, &m)?;
        #[cfg(test)]
        if self.serial_norm_modulate.get() {
            self.rms_norm_f32(s, n)?;
            return self.row_modulate(s, m);
        }
        self.launch_norm_modulate(s, &self.norm_modulate_f32, &n, &m)
    }

    fn prepare_norm_modulate(&self, n: &NormRows, m: &ModulateRows) -> Result<()> {
        ensure!(
            n.rows == m.rows && n.dim == m.dim && n.output == m.data,
            "RMSNorm modulation output extent"
        );
        ensure!(
            n.rows > 0
                && n.dim > 0
                && n.dim.is_multiple_of(8)
                && n.threads == HIDDEN_NORM_THREADS
                && n.dim / 8 <= n.threads as usize
                && m.stride >= n.dim
                && m.entries > 0
                && m.index != 0
                && m.index.is_multiple_of(4),
            "RMSNorm modulation launch extent"
        );
        for p in [n.input, n.weight, n.output, m.shift, m.scale] {
            ensure!(
                p != 0 && p.is_multiple_of(16),
                "RMSNorm modulation alignment"
            );
        }
        n.rows
            .checked_mul(n.dim)
            .context("RMSNorm element extent overflow")?;
        m.stride
            .checked_mul(m.entries as usize)
            .context("modulation table extent overflow")?;
        Ok(())
    }

    fn launch_norm_modulate(
        &self,
        s: &Arc<CudaStream>,
        kernel: &crate::backend::gpu::CudaFunction,
        n: &NormRows,
        m: &ModulateRows,
    ) -> Result<()> {
        self.prepare_fp16_store(s, &self.norm_module)?;
        unsafe {
            s.launch_builder(kernel)
                .arg(&n.input)
                .arg(&n.weight)
                .arg(&n.output)
                .arg(&m.shift)
                .arg(&m.scale)
                .arg(&m.index)
                .arg(&i32::try_from(n.rows)?)
                .arg(&i32::try_from(n.dim)?)
                .arg(&n.eps)
                .arg(&(m.stride as u64))
                .arg(&m.entries)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(n.rows)?, 1, 1),
                    block_dim: (n.threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.finish_fp16_store(s, &self.norm_module, "rms_norm_modulate")
    }

    fn gated_add_rms(&self, s: &Arc<CudaStream>, g: &GatedAddRms) -> Result<()> {
        self.launch_gated_add_rms(s, &self.gated_add_rms, g)
    }

    #[cfg(test)]
    fn gated_add_rms_raw(&self, s: &Arc<CudaStream>, g: &GatedAddRms) -> Result<()> {
        self.launch_gated_add_rms(s, &self.gated_add_rms_raw, g)
    }

    fn launch_gated_add_rms(
        &self,
        s: &Arc<CudaStream>,
        kernel: &crate::backend::gpu::CudaFunction,
        g: &GatedAddRms,
    ) -> Result<()> {
        ensure!(
            g.rows > 0
                && g.dim > 0
                && g.dim.is_multiple_of(8)
                && g.dim / 8 <= HIDDEN_NORM_THREADS as usize
                && g.gate_stride >= g.dim
                && g.norm_stride >= g.dim
                && g.gate_stride.is_multiple_of(8)
                && g.norm_stride.is_multiple_of(8)
                && g.gate_entries > 0
                && g.norm_entries > 0
                && g.gate_index != 0
                && g.norm_index != 0
                && g.gate_index.is_multiple_of(4)
                && g.norm_index.is_multiple_of(4)
                && g.x != g.y
                && g.x != g.output,
            "gated RMSNorm launch extent"
        );
        for p in [g.x, g.y, g.gate, g.weight, g.output, g.shift, g.scale] {
            ensure!(p != 0 && p.is_multiple_of(16), "gated RMSNorm alignment");
        }
        g.rows
            .checked_mul(g.dim)
            .context("gated RMSNorm element extent overflow")?;
        g.gate_stride
            .checked_mul(g.gate_entries as usize)
            .context("gate table extent overflow")?;
        g.norm_stride
            .checked_mul(g.norm_entries as usize)
            .context("norm table extent overflow")?;
        let rows = i32::try_from(g.rows)?;
        let dim = i32::try_from(g.dim)?;
        self.prepare_fp16_store(s, &self.norm_module)?;
        unsafe {
            s.launch_builder(kernel)
                .arg(&g.x)
                .arg(&g.y)
                .arg(&g.gate)
                .arg(&g.gate_index)
                .arg(&(g.gate_stride as u64))
                .arg(&g.gate_entries)
                .arg(&g.weight)
                .arg(&g.output)
                .arg(&g.shift)
                .arg(&g.scale)
                .arg(&g.norm_index)
                .arg(&(g.norm_stride as u64))
                .arg(&g.norm_entries)
                .arg(&rows)
                .arg(&dim)
                .arg(&g.eps)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(g.rows)?, 1, 1),
                    block_dim: (HIDDEN_NORM_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.finish_fp16_store(s, &self.norm_module, "gated_residual")
    }

    fn rms_norm(&self, s: &Arc<CudaStream>, n: NormRows) -> Result<()> {
        self.launch_rms_norm(s, &self.norm, n)
    }

    /// Weight multiplies in f32 and the store rounds once.
    fn rms_norm_f32(&self, s: &Arc<CudaStream>, n: NormRows) -> Result<()> {
        self.launch_rms_norm(s, &self.norm_f32, n)
    }

    fn launch_rms_norm(
        &self,
        s: &Arc<CudaStream>,
        kernel: &crate::backend::gpu::CudaFunction,
        n: NormRows,
    ) -> Result<()> {
        self.prepare_fp16_store(s, &self.norm_module)?;
        unsafe {
            s.launch_builder(kernel)
                .arg(&n.input)
                .arg(&n.weight)
                .arg(&n.output)
                .arg(&i32::try_from(n.rows)?)
                .arg(&i32::try_from(n.dim)?)
                .arg(&n.eps)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(n.rows)?, 1, 1),
                    block_dim: (n.threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.finish_fp16_store(s, &self.norm_module, "rms_norm")
    }

    #[cfg(test)]
    fn row_modulate(&self, s: &Arc<CudaStream>, m: ModulateRows) -> Result<()> {
        unsafe {
            s.launch_builder(&self.modulate)
                .arg(&m.data)
                .arg(&m.shift)
                .arg(&m.scale)
                .arg(&m.index)
                .arg(&m.data)
                .arg(&i32::try_from(m.rows)?)
                .arg(&i32::try_from(m.dim)?)
                .arg(&(m.stride as u64))
                .arg(&m.entries)
                .launch(LaunchConfig {
                    grid_dim: (
                        u32::try_from(m.rows)?,
                        u32::try_from(m.dim.div_ceil(256))?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

/// `[text | conditioning video | audio | generated video]` row ranges of a packed layout.
#[derive(Clone, Copy)]
struct Ranges {
    text: usize,
    condition: usize,
    audio_start: usize,
    audio: usize,
    video_start: usize,
    rows: usize,
}

impl Ranges {
    fn of(layout: &crate::minimax_h3::PackedLayout) -> Result<Self> {
        let text = layout.text_indices.len();
        let condition = layout.condition_video_rows;
        let audio = layout.audio_indices.len();
        let audio_start = text + condition;
        let video_start = audio_start + audio;
        let rows = layout.rows();
        ensure!(
            layout
                .text_indices
                .iter()
                .enumerate()
                .all(|(i, &r)| r as usize == i)
                && layout
                    .audio_indices
                    .iter()
                    .enumerate()
                    .all(|(i, &r)| r as usize == audio_start + i)
                && layout.video_indices.iter().enumerate().all(|(i, &r)| {
                    r as usize
                        == if i < condition {
                            text + i
                        } else {
                            video_start + i - condition
                        }
                }),
            "packed layout is not [text | conditions | audio | video]"
        );
        Ok(Self {
            text,
            condition,
            audio_start,
            audio,
            video_start,
            rows,
        })
    }
}

/// Teacher-forced inputs of one evaluation: patchified video rows `[video rows, 96]` (conditioning first) and audio
/// rows `[audio rows, 32]` in F32, text states `[text rows, text_dim]` in BF16, and each sequence row's AdaLN pair and
/// timestep from `AdalnPlan`.
pub(crate) struct StepInputs<'a> {
    pub video: &'a CudaSlice<f32>,
    pub audio: &'a CudaSlice<f32>,
    pub text: &'a CudaSlice<bf16>,
    pub row_pairs: &'a CudaSlice<u32>,
    pub row_timesteps: &'a CudaSlice<u32>,
}

/// Transformer-block placement: `Derived` from the sensed VRAM and host budgets; the gates also force `Streamed` (no
/// block resident: two device slots, host tier from the sensed host budget, rest disk) and `StreamedDisk` (every block
/// from disk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    Derived,
    #[cfg(test)]
    Streamed,
    #[cfg(test)]
    StreamedDisk,
}

#[cfg(test)]
impl Placement {
    /// `FF_H3_PLACEMENT` = `derived`, `streamed` or `streamed-disk`.
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var("FF_H3_PLACEMENT").as_deref() {
            Ok("derived") => Ok(Self::Derived),
            Ok("streamed") => Ok(Self::Streamed),
            Ok("streamed-disk") => Ok(Self::StreamedDisk),
            other => anyhow::bail!(
                "FF_H3_PLACEMENT must be derived, streamed or streamed-disk: {other:?}"
            ),
        }
    }
}

/// `rows` sequence rows keep every launch of the block path inside its index type: `rotary_prefix_kv` indexes
/// `rows * 2 * heads * head_dim` elements and `residual` `rows * hidden` elements in i32, the q/k norm
/// (`rms_norm_bf16`, or `rms_norm` on a HyperFlow modulated block) runs `rows * heads` rows in i32, and
/// `vision_upcast_f32` / `vision_bias_f32` take u32 counts of `rows * hidden`.
fn launch_extents(spec: &H3Spec, rows: usize) -> Result<()> {
    let limit = i32::MAX as usize;
    for (kernel, per_row) in [
        ("rotary_prefix_kv", 2 * spec.heads * spec.head_dim),
        ("residual", spec.hidden),
        ("rms_norm_bf16", spec.heads),
    ] {
        let extent = rows
            .checked_mul(per_row)
            .with_context(|| format!("{kernel} extent of {rows} rows overflows"))?;
        ensure!(
            extent <= limit,
            "{rows} sequence rows give {kernel} {extent} indices, above its i32 limit {limit} (at most {} rows)",
            limit / per_row
        );
    }
    ensure!(
        rows * spec.hidden <= u32::MAX as usize,
        "{rows} sequence rows exceed the u32 counts of vision_upcast_f32 / vision_bias_f32"
    );
    Ok(())
}

/// `vision_affine_cast_bf16` of `count` F32 values at `source` to BF16 at `target`; `count == 0` (the conditioning
/// rows of a layout with no keyframe anchors, the `t2va` workflow) launches nothing.
fn cast_rows(
    s: &Arc<CudaStream>,
    kernel: &crate::backend::gpu::CudaFunction,
    source: u64,
    target: u64,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    unsafe {
        s.launch_builder(kernel)
            .arg(&source)
            .arg(&target)
            .arg(&(count as u64))
            .arg(&0f32)
            .arg(&1f32)
            .launch(flat(count))?;
    }
    Ok(())
}

/// `y [rows, n] = x [rows, k] * w^T + b` in F32 on rows starting at element `x_offset` of `x`.
fn projection(
    s: &Arc<CudaStream>,
    blas: &Blas,
    bias: &CudaFunction,
    [w, b]: &[CudaSlice<f32>; 2],
    (x, x_offset): (&CudaSlice<f32>, usize),
    (y, rows): (&CudaSlice<f32>, usize),
) -> Result<()> {
    let (n, k) = (b.len(), w.len() / b.len());
    blas.linear_f32(
        view(
            w,
            s,
            0,
            [u32::try_from(n)?, u32::try_from(k)?],
            [k as u64, 1],
        )?,
        view(
            x,
            s,
            x_offset,
            [u32::try_from(rows)?, u32::try_from(k)?],
            [k as u64, 1],
        )?,
        &mut TensorMut(view(
            y,
            s,
            0,
            [u32::try_from(rows)?, u32::try_from(n)?],
            [n as u64, 1],
        )?),
    )?;
    let out = pointer(y, s);
    unsafe {
        s.launch_builder(bias)
            .arg(&out)
            .arg(&pointer(b, s))
            .arg(&out)
            .arg(&u32::try_from(rows)?)
            .arg(&u32::try_from(n)?)
            .launch(flat(rows * n))?;
    }
    Ok(())
}

/// Rounds `count` F32 values to BF16 sequence rows starting at element `target` of the hidden state.
fn round_into(
    s: &Arc<CudaStream>,
    kernel: &CudaFunction,
    hidden: &CudaSlice<bf16>,
    source: u64,
    target: usize,
    count: usize,
) -> Result<()> {
    cast_rows(
        s,
        kernel,
        source,
        pointer(hidden, s) + 2 * target as u64,
        count,
    )
}

/// Widens `count` BF16 values starting at element `source` of `normed` into the F32 scratch.
fn upcast_from(
    s: &Arc<CudaStream>,
    kernel: &CudaFunction,
    normed: &CudaSlice<bf16>,
    scratch: &CudaSlice<f32>,
    source: usize,
    count: usize,
) -> Result<()> {
    unsafe {
        s.launch_builder(kernel)
            .arg(&(pointer(normed, s) + 2 * source as u64))
            .arg(&pointer(scratch, s))
            .arg(&u32::try_from(count)?)
            .launch(flat(count))?;
    }
    Ok(())
}

fn image_fill_rate(dir: &Path, manifest: &TensorManifest) -> Result<(usize, f64)> {
    let mut shards = BTreeMap::<&str, (&str, &TensorEntry)>::new();
    for (name, entry) in &manifest.entries {
        if entry.load && (4 << 20..=64 << 20).contains(&entry.bytes) {
            let sample = shards.entry(entry.shard.as_str()).or_insert((name, entry));
            if sample.1.bytes < entry.bytes {
                *sample = (name, entry);
            }
        }
    }
    let samples: Vec<_> = shards.into_values().take(4).collect();
    ensure!(
        !samples.is_empty(),
        "resident disk rate needs a published conditioner read sample"
    );
    let mut bufs: Vec<_> = samples
        .iter()
        .map(|(_, entry)| vec![0u8; entry.bytes])
        .collect();
    let bytes = bufs.iter().map(Vec::len).sum::<usize>();
    let mut requests = samples
        .iter()
        .zip(&mut bufs)
        .map(|(&(name, entry), buf)| DirectRead {
            name,
            entry,
            dest: buf.as_mut_ptr(),
            dest_len: buf.len(),
            cast: FillCast::Raw,
        })
        .collect::<Vec<_>>();
    let mut source = Source::new(dir, manifest);
    let began = std::time::Instant::now();
    source.reader.read_many_direct(&mut requests)?;
    let seconds = began.elapsed().as_secs_f64();
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "invalid resident I1 fill duration"
    );
    Ok((bytes, bytes as f64 / seconds))
}

/// manifest's dense linears.
/// Device and source bytes of every block, plus its staging and transfer workspace.
struct DenseSizing {
    layers: Vec<usize>,
    source_layers: Vec<usize>,
    largest: usize,
    staging: usize,
    transient: usize,
}

/// Sizing of a manifest's dense linears: a dequantized entry's device slot holds its BF16 tile, a pass-through
/// entry's holds its stored bytes, and a stored type with no device path is an error.
fn dense_sizing(manifest: &TensorManifest, layers: usize) -> Result<DenseSizing> {
    let mut device_layers = vec![0usize; layers];
    let mut source_layers = vec![0usize; layers];
    let mut dequant_layer = vec![false; layers];
    let mut largest = 0usize;
    for (name, entry) in &manifest.entries {
        if let Some(layer) = entry.layer
            && entry.load
            && entry.shape.len() == 2
        {
            let device_bytes = device_slot_bytes(name, entry)?;
            device_layers[layer as usize] += device_bytes;
            source_layers[layer as usize] += entry.bytes;
            largest = largest.max(device_bytes);
            if dequantizes(entry.quant) {
                dequant_layer[layer as usize] = true;
            }
        }
    }
    let staging = dequant_layer
        .iter()
        .zip(&source_layers)
        .filter_map(|(dequant, source)| dequant.then_some(*source))
        .max()
        .unwrap_or(0);
    let transient = manifest
        .entries
        .iter()
        .filter(|(_, entry)| entry.load && dequantizes(entry.quant))
        .map(|(_, entry)| entry.bytes)
        .max()
        .unwrap_or(0);
    Ok(DenseSizing {
        layers: device_layers,
        source_layers,
        largest,
        staging,
        transient,
    })
}

/// Bytes the dense placer read at setup: device free and total, the owned reserve, the largest matrix, and the host
/// budget after `slot_count * slot + largest + 4 GiB`.
#[derive(Clone, Copy)]
struct PlacementFacts {
    free_bytes: usize,
    total_bytes: usize,
    owned_bytes: usize,
    largest_bytes: usize,
    host_available_bytes: Option<usize>,
    device_budget: usize,
    host_budget: usize,
}

struct PlacedLayers {
    tier: Option<DenseTier>,
    #[cfg(test)]
    facts: PlacementFacts,
}

/// Device budget and host budget for `layers`, then the `DensePlacement` those budgets produce. `host_available` is
/// measured only when the layers do not all fit the device budget. Visit count 1 spends no host budget.
fn plan_dense_layers(
    (layers, source_layers, largest, owned, media): (&[usize], &[usize], usize, usize, usize),
    placement: Placement,
    (free, total, host_available): (usize, usize, Option<usize>),
    slot_count: usize,
    visits: usize,
) -> Result<(crate::prefill::DensePlacement, PlacementFacts)> {
    use crate::prefill::DensePlacement;
    DensePlacement::host_budget_for_visits(0, visits)?;
    let mut device_budget = DensePlacement::device_budget(free, total, owned, 0, media)?;
    if placement != Placement::Derived {
        let slots = 2 * layers.iter().copied().max().context("no streamed layers")?;
        ensure!(
            slots <= device_budget,
            "two block slots exceed the device budget"
        );
        device_budget = slots;
    }
    let probe = DensePlacement::with_allocations(
        layers,
        layers,
        Some(source_layers),
        device_budget,
        0,
        0,
        slot_count,
    )?;
    let (host_budget, host_available) = if probe.slot_bytes > 0 {
        let available = host_available.context("host tier without a measured host budget")?;
        let reserved = pinned_host_reserve(slot_count, probe.slot_bytes, largest)?;
        ensure!(
            available >= reserved,
            "dense pinned staging and host reserve exceed available RAM: need={reserved} available={available}"
        );
        let host_bytes = match placement {
            Placement::Derived => {
                DensePlacement::host_budget_for_visits(available - reserved, visits)?
            }
            #[cfg(test)]
            Placement::Streamed => available - reserved,
            #[cfg(test)]
            Placement::StreamedDisk => 0,
        };
        (host_bytes, Some(available))
    } else {
        (0, None)
    };
    let plan = DensePlacement::with_allocations(
        layers,
        layers,
        Some(source_layers),
        device_budget,
        host_budget,
        0,
        slot_count,
    )?;
    ensure!(
        placement == Placement::Derived || plan.resident_bytes == 0,
        "{placement:?} placement kept a resident block"
    );
    Ok((
        plan,
        PlacementFacts {
            free_bytes: free,
            total_bytes: total,
            owned_bytes: owned,
            largest_bytes: largest,
            host_available_bytes: host_available,
            device_budget,
            host_budget,
        },
    ))
}

/// Dense streaming of `layers` (per-layer device matrix bytes, `largest` device matrix) beside `owned` resident bytes:
/// placement derived from free device memory and host RAM, or forced streamed / disk-only. `source_layers` sizes the
/// host/disk tier; it equals `layers` for a native BF16 manifest and is the stored source size for a quantized one.
/// Host-tier copies are filled from `base` in `roles` order before the first `begin`.
fn place_layers(
    component: &'static str,
    d: &Device,
    base: &mut Source<'_>,
    (layers, source_layers, largest, owned, media): (&[usize], &[usize], usize, usize, usize),
    (placement, visits): (Placement, usize),
    name: impl Fn(usize) -> String,
    roles: &[&str],
) -> Result<PlacedLayers> {
    use crate::prefill::DensePlacement;
    let (free, total) = d.ctx.mem_get_info()?;
    let mut device_budget = DensePlacement::device_budget(free, total, owned, 0, media)?;
    if placement != Placement::Derived {
        let slots = 2 * layers.iter().copied().max().context("no streamed layers")?;
        ensure!(
            slots <= device_budget,
            "two block slots exceed the device budget"
        );
        device_budget = slots;
    }
    let extent = layers
        .iter()
        .try_fold(0usize, |sum, &bytes| sum.checked_add(bytes))
        .context("dense weight extent overflow")?;
    let host_available = if extent > device_budget {
        Some(usize::try_from(
            crate::backend::cuda::calibrate::host_memory(Path::new("/"))?,
        )?)
    } else {
        None
    };
    let (plan, facts) = plan_dense_layers(
        (layers, source_layers, largest, owned, media),
        placement,
        (free, total, host_available),
        2,
        visits,
    )?;
    let used_slots = plan
        .slots
        .iter()
        .flatten()
        .copied()
        .max()
        .map(|index| index + 1)
        .unwrap_or(0);
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_dense_placement": &plan, "mode": format!("{placement:?}"), "streaming_slots": used_slots, "free_bytes": facts.free_bytes, "total_bytes": facts.total_bytes, "owned_block_bytes": facts.owned_bytes, "largest_bytes": facts.largest_bytes, "device_budget": facts.device_budget, "host_budget": facts.host_budget, "host_available_bytes": facts.host_available_bytes})
        );
    }
    let tier = bind_layers(component, d, base, source_layers, plan, name, roles)?;
    #[cfg(test)]
    {
        Ok(PlacedLayers { tier, facts })
    }
    #[cfg(not(test))]
    {
        Ok(PlacedLayers { tier })
    }
}

fn bind_layers(
    component: &'static str,
    d: &Device,
    base: &mut Source<'_>,
    source_layers: &[usize],
    plan: crate::prefill::DensePlacement,
    name: impl Fn(usize) -> String,
    roles: &[&str],
) -> Result<Option<DenseTier>> {
    let tier = if plan.slot_bytes > 0 {
        let dir = base.reader.dir.clone();
        let host_tier = plan.host.clone();
        let fp16 = cubin::capability(&d.ctx)? < 80;
        let mut dense =
            crate::backend::cuda::weights::DenseStreaming::new(&dir, base.manifest, plan, d, None)?;
        let mut requests = Vec::new();
        let mut checks = Vec::new();
        let mut largest = 0usize;
        for layer in (0..source_layers.len()).filter(|&layer| host_tier[layer]) {
            let name = name(layer);
            let (layout, _) = matrix_layout(base.manifest, &name, roles)?;
            let host = dense
                .host_layer(layer)?
                .with_context(|| format!("{name}: host tier without a pinned copy"))?;
            ensure!(
                host.len() >= source_layers[layer],
                "{name}: pinned copy size"
            );
            let mut source_offset = 0usize;
            for (role, (offset, _)) in &layout {
                let tensor = format!("{name}.{role}");
                let entry = &base.manifest.entries[&tensor];

                let at = if entry.quant.is_some() {
                    source_offset
                } else {
                    2 * offset
                };
                requests.push(OwnedRead {
                    name: tensor.clone(),
                    entry: entry.clone(),
                    dest: unsafe { host.as_mut_ptr().add(at) },
                    dest_len: entry.bytes,
                    cast: weight_fill_cast(fp16, entry),
                });
                checks.push(FillCheck {
                    layer,
                    tensor,
                    dtype: entry.dtype,
                    quant: entry.quant,
                    at,
                    bytes: entry.bytes,
                });
                largest = largest.max(entry.bytes);
                source_offset += entry.bytes;
            }
        }
        let slot_count = used_dense_slots(&dense.plan);
        let slot_bytes = dense.plan.slot_bytes;
        Some(DenseTier::schedule(
            component,
            dense,
            &dir,
            requests,
            checks,
            (slot_count, slot_bytes, largest),
        )?)
    } else {
        None
    };
    Ok(tier)
}

type StoredOffsets = Vec<Vec<(String, usize)>>;

struct VectorArenaLayout {
    offsets: BTreeMap<String, (usize, usize, Dtype)>,
    bytes: usize,
}

impl VectorArenaLayout {
    fn new(manifest: &TensorManifest, spec: &H3Spec) -> Result<Self> {
        let dropped = spec
            .dropped_blocks
            .as_deref()
            .context("HyperFlow dropped blocks")?;
        let mut offsets = BTreeMap::new();
        let mut bytes = 0usize;
        let mut add = |name: String, len: usize, dtype: Dtype| -> Result<()> {
            ensure!(len > 0, "{name}: empty vector arena tensor");
            let offset = bytes;
            bytes = bytes
                .checked_add(stored_allocation_bytes(len, 256)?)
                .context("vector arena extent overflow")?;
            ensure!(
                offsets.insert(name.clone(), (offset, len, dtype)).is_none(),
                "duplicate vector arena tensor {name}"
            );
            Ok(())
        };
        for layer in 0..spec.layers {
            for (role, width) in [
                ("norm1", spec.hidden),
                ("norm2", spec.hidden),
                ("attn.norm_q", spec.head_dim),
                ("attn.norm_k", spec.head_dim),
            ] {
                add(
                    format!("transformer_blocks.{layer}.{role}.weight"),
                    width.checked_mul(2).context("norm extent overflow")?,
                    Dtype::BF16,
                )?;
            }
            if dropped.contains(&layer) {
                add(
                    format!("transformer_blocks.{layer}.identity"),
                    2,
                    Dtype::BF16,
                )?;
            }
        }
        for (name, entry) in &manifest.entries {
            if entry.load && entry.layer.is_some() && name.ends_with(".wscale") {
                ensure!(
                    entry.dtype == Dtype::F32 && entry.shape.len() == 1,
                    "{name}: vector arena scale layout"
                );
                add(name.clone(), entry.bytes, Dtype::F32)?;
            }
        }
        Ok(Self { offsets, bytes })
    }
}

fn stored_allocation_granularity(d: &Device) -> Result<usize> {
    #[cfg(feature = "cuda")]
    let granularity = {
        use cudarc::driver::sys::{self, CUmemAllocationGranularity_flags as G};
        let mut properties: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        properties.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.__bindgen_anon_1.id = d.ctx.cu_device();
        let mut granularity = 0;
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut granularity,
                &properties,
                G::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
            .result()?;
        }
        granularity
    };
    #[cfg(feature = "rocm")]
    let granularity = d.ctx.allocation_granularity()?;
    ensure!(granularity > 0, "zero stored allocation granularity");
    Ok(granularity)
}

fn stored_allocation_bytes(bytes: usize, granularity: usize) -> Result<usize> {
    ensure!(granularity > 0, "zero stored allocation granularity");
    bytes
        .div_ceil(granularity)
        .checked_mul(granularity)
        .context("stored allocation extent overflow")
}

fn stored_allocation_costs(
    manifest: &TensorManifest,
    spec: &H3Spec,
    granularity: usize,
    owned: usize,
) -> Result<(Vec<usize>, usize)> {
    let mut layers = vec![0usize; spec.layers];
    let arena = VectorArenaLayout::new(manifest, spec)?;
    let logical_metadata = arena
        .offsets
        .iter()
        .filter(|(name, _)| !name.ends_with(".identity"))
        .try_fold(0usize, |sum, (_, (_, bytes, _))| {
            sum.checked_add(*bytes).context("stored metadata extent")
        })?;
    for (name, entry) in &manifest.entries {
        let Some(layer) = entry.layer else { continue };
        if !entry.load {
            continue;
        }
        let layer = layer as usize;
        ensure!(layer < spec.layers, "{name}: stored allocation layer");
        if entry.shape.len() == 2 {
            layers[layer] = layers[layer]
                .checked_add(stored_allocation_bytes(entry.bytes, granularity)?)
                .context("stored layer allocation extent")?;
        }
    }
    let arena_bytes = stored_allocation_bytes(arena.bytes, granularity)?;
    let owned = owned
        .checked_sub(logical_metadata)
        .and_then(|n| n.checked_add(arena_bytes))
        .context("stored owned allocation extent")?;
    Ok((layers, owned))
}

#[cfg(test)]
fn trace_placement_pool(d: &Device, phase: &str) -> Result<()> {
    #[cfg(feature = "cuda")]
    let (reserved, used, event_tracking) = {
        use cudarc::driver::sys::{self, CUmemPool_attribute};
        let mut pool = std::ptr::null_mut();
        let (mut reserved, mut used) = (0u64, 0u64);
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, d.ctx.cu_device()).result()?;
            sys::cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                (&mut reserved as *mut u64).cast(),
            )
            .result()?;
            sys::cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                (&mut used as *mut u64).cast(),
            )
            .result()?;
        }
        (reserved, used, d.ctx.is_event_tracking())
    };
    #[cfg(feature = "rocm")]
    let (reserved, used, event_tracking) = {
        let (reserved, used) = d.ctx.default_mem_pool_usage()?;
        (reserved, used, false)
    };
    eprintln!(
        "{}",
        serde_json::json!({"h3_placement_pool": {
            "phase": phase, "reserved_bytes": reserved, "used_bytes": used,
            "free_bytes": d.ctx.mem_get_info()?.0,
            "event_tracking": event_tracking, "synchronized": false
        }})
    );
    Ok(())
}

/// Charged device bytes of one layer. A `.down.weight` charges `stored_low_rank_pitch(rank) * K * 2`, then the allocation granule.
fn stored_layer_charge(
    manifest: &TensorManifest,
    layer: usize,
    granularity: usize,
) -> Result<usize> {
    let layer_id = u32::try_from(layer).context("stored layer id")?;
    let mut sum = 0usize;
    for (name, entry) in &manifest.entries {
        if entry.layer != Some(layer_id) || !entry.load || entry.shape.len() != 2 {
            continue;
        }
        let bytes = if name.ends_with(".down.weight") {
            ensure!(
                entry.dtype == Dtype::BF16 && entry.shape.len() == 2,
                "{name} low-rank down layout"
            );
            let (rank, k) = (entry.shape[0], entry.shape[1]);
            let stored = rank
                .checked_mul(k)
                .and_then(|n| n.checked_mul(2))
                .with_context(|| format!("{name} low-rank down bytes"))?;
            ensure!(
                entry.bytes == stored && rank > 0,
                "{name} low-rank down bytes"
            );
            low_rank_overread(name, rank, k)?.1
        } else {
            entry.bytes
        };
        sum = sum
            .checked_add(stored_allocation_bytes(bytes, granularity)?)
            .with_context(|| format!("{name} stored layer charge"))?;
    }
    Ok(sum)
}

/// `(over-read of the last short down, largest past-end over-read)` in packed order, in bytes.
fn low_rank_layer_tail(
    packed: &[(String, usize)],
    manifest: &TensorManifest,
    packed_bytes: usize,
) -> Result<(usize, usize)> {
    let mut last_short = 0usize;
    let mut max_past_end = 0usize;
    for item in packed {
        let name = &item.0;
        let offset = item.1;
        if !name.ends_with(".down.weight") {
            continue;
        }
        let entry = manifest
            .entries
            .get(name)
            .with_context(|| format!("manifest lacks {name}"))?;
        ensure!(
            entry.dtype == Dtype::BF16 && entry.shape.len() == 2,
            "{name} low-rank down layout"
        );
        let (rank, k) = (entry.shape[0], entry.shape[1]);
        let stored = rank
            .checked_mul(k)
            .and_then(|n| n.checked_mul(2))
            .with_context(|| format!("{name} low-rank down bytes"))?;
        ensure!(
            entry.bytes == stored && rank > 0,
            "{name} low-rank down bytes"
        );
        let (pitch, pitched) = low_rank_overread(name, rank, k)?;
        if pitch == rank {
            continue;
        }
        let overread = pitched
            .checked_sub(stored)
            .with_context(|| format!("{name} low-rank over-read"))?;
        last_short = overread;
        let after = packed_bytes
            .checked_sub(offset)
            .and_then(|n| n.checked_sub(stored))
            .with_context(|| format!("{name} low-rank tail"))?;
        max_past_end = max_past_end.max(overread.saturating_sub(after));
    }
    Ok((last_short, max_past_end))
}

/// `(plan, device budget, host budget, available host bytes)` for one charged table.
fn stored_compact(
    (placement, visits): (Placement, usize),
    charged_device: &[usize],
    charged_slots: &[usize],
    kept_source: &[usize],
    device_cap: usize,
    largest: usize,
    granularity: usize,
) -> Result<(crate::prefill::DensePlacement, usize, usize, Option<usize>)> {
    use crate::prefill::DensePlacement;
    ensure!(
        charged_device.len() == charged_slots.len() && charged_device.len() == kept_source.len(),
        "stored placement extent"
    );
    DensePlacement::host_budget_for_visits(0, visits)?;
    let mut device_budget = device_cap;
    if placement != Placement::Derived {
        let slots = 2 * charged_slots
            .iter()
            .copied()
            .max()
            .context("no stored transformer block")?;
        ensure!(
            slots <= device_budget,
            "two block slots exceed the device budget"
        );
        device_budget = slots
            .checked_add(
                charged_device
                    .len()
                    .checked_mul(granularity)
                    .context("stored streaming setup extent")?,
            )
            .context("stored streaming setup extent")?;
        ensure!(
            device_budget <= device_cap,
            "stored streaming setup exceeds the device budget"
        );
    }
    let probe = DensePlacement::with_allocations(
        charged_device,
        charged_slots,
        Some(kept_source),
        device_budget,
        0,
        granularity,
        2,
    )?;
    let host_available = if probe.slot_bytes > 0 {
        Some(usize::try_from(
            crate::backend::cuda::calibrate::host_memory(Path::new("/"))?,
        )?)
    } else {
        None
    };
    let host_budget = if let Some(available) = host_available {
        let reserve = 2usize
            .checked_mul(probe.slot_bytes)
            .and_then(|bytes| bytes.checked_add(largest))
            .and_then(|bytes| bytes.checked_add(4 << 30))
            .context("stored host reserve overflow")?;
        let budget = available
            .checked_sub(reserve)
            .context("stored pinned staging and host reserve exceed available RAM")?;
        match placement {
            Placement::Derived => DensePlacement::host_budget_for_visits(budget, visits)?,
            #[cfg(test)]
            Placement::Streamed => budget,
            #[cfg(test)]
            Placement::StreamedDisk => 0,
        }
    } else {
        0
    };
    let compact = DensePlacement::with_allocations(
        charged_device,
        charged_slots,
        Some(kept_source),
        device_budget,
        host_budget,
        granularity,
        2,
    )?;
    Ok((compact, device_budget, host_budget, host_available))
}

/// `(resident, host tier, streamed)` among the kept layers.
fn placement_counts(compact: &crate::prefill::DensePlacement) -> (usize, usize, usize) {
    let slots = compact.slots.iter().filter(|slot| slot.is_some()).count();
    let host = compact.host.iter().filter(|flag| **flag).count();
    (compact.slots.len() - slots, host, slots)
}

/// HyperFlow transformer blocks. Dropped layers stay out of `DensePlacement` and come back as empty slots.
/// `packed` contains stored byte offsets; placement prices allocations at the reported granularity.
fn place_stored_layers(
    d: &Device,
    base: &mut Source<'_>,
    spec: &H3Spec,
    sizing: &DenseSizing,
    owned: usize,
    (placement, visits, selected): (Placement, usize, Option<crate::prefill::DensePlacement>),
) -> Result<(Option<DenseTier>, StoredOffsets)> {
    use crate::prefill::DensePlacement;
    let dropped = spec
        .dropped_blocks
        .as_deref()
        .context("HyperFlow dropped blocks")?;
    let mut order = vec![Vec::new(); spec.layers];
    let mut packed = vec![Vec::new(); spec.layers];
    let mut cursor = vec![0usize; spec.layers];
    for (name, entry) in &base.manifest.entries {
        let Some(layer) = entry.layer else {
            continue;
        };
        if !entry.load || entry.shape.len() != 2 {
            continue;
        }
        let layer = layer as usize;
        ensure!(
            layer < spec.layers,
            "stored linear {name} is outside {} layers",
            spec.layers
        );
        let bytes = device_slot_bytes(name, entry)?;
        ensure!(bytes == entry.bytes, "{name} is not pass-through");
        ensure!(cursor[layer].is_multiple_of(16), "stored factor alignment");
        match entry.dtype {
            Dtype::BF16 | Dtype::I8 => {}
            other => bail!("{name}: stored dtype {other:?}"),
        }
        packed[layer].push((name.clone(), cursor[layer]));
        order[layer].push(name.clone());
        cursor[layer] = cursor[layer]
            .checked_add(entry.bytes)
            .with_context(|| format!("{name} stored offset"))?;
    }
    for (layer, &bytes) in cursor.iter().enumerate() {
        ensure!(
            bytes == sizing.layers[layer] && bytes == sizing.source_layers[layer],
            "block {layer} stored bytes"
        );
    }
    let mut kept_device = Vec::new();
    let mut kept_source = Vec::new();
    let mut kept_index = Vec::new();
    for layer in 0..spec.layers {
        let (device, source) = (sizing.layers[layer], sizing.source_layers[layer]);
        if dropped.contains(&layer) {
            ensure!(
                device == 0 && source == 0,
                "dropped block {layer} stores a linear"
            );
            continue;
        }
        ensure!(device > 0, "block {layer} has no stored linear");
        ensure!(
            device == source,
            "block {layer} device bytes differ from the stored bytes"
        );
        kept_device.push(device);
        kept_source.push(source);
        kept_index.push(layer);
    }
    ensure!(!kept_device.is_empty(), "no stored transformer block");
    let granularity = stored_allocation_granularity(d)?;
    let mut last_short_overread = 0usize;
    let mut max_past_end = 0usize;
    let mut layer_tails = vec![0usize; spec.layers];
    for &layer in &kept_index {
        let (last, past) = low_rank_layer_tail(&packed[layer], base.manifest, cursor[layer])?;
        last_short_overread = last_short_overread.max(last);
        max_past_end = max_past_end.max(past);
        layer_tails[layer] = last.max(past);
    }
    let layer_tail_bytes = layer_tails.iter().copied().max().unwrap_or(0);
    let charged_device_after = kept_index
        .iter()
        .map(|&layer| stored_layer_charge(base.manifest, layer, granularity))
        .collect::<Result<Vec<_>>>()?;
    let charged_slots_after = kept_index
        .iter()
        .map(|&layer| {
            let bytes = cursor[layer]
                .checked_add(layer_tails[layer])
                .context("stored layer tail")?;
            stored_allocation_bytes(bytes, granularity)
        })
        .collect::<Result<Vec<_>>>()?;
    #[cfg(test)]
    trace_placement_pool(d, "before_placement")?;
    let (compact, cold) = if let Some(selected) = selected {
        ensure!(
            selected.slots.len() == kept_index.len()
                && selected
                    .slots
                    .iter()
                    .zip(&charged_slots_after)
                    .all(|(slot, &bytes)| slot.is_none() || bytes <= selected.slot_bytes),
            "resident HyperFlow plan cannot hold padded stored slots"
        );
        (selected, None)
    } else {
        let (charged_layers, owned) =
            stored_allocation_costs(base.manifest, spec, granularity, owned)?;
        let charged_device: Vec<_> = kept_index
            .iter()
            .map(|&layer| charged_layers[layer])
            .collect();
        let charged_slots: Vec<_> = kept_device
            .iter()
            .map(|&bytes| stored_allocation_bytes(bytes, granularity))
            .collect::<Result<_>>()?;
        let (free, total) = d.ctx.mem_get_info()?;
        let device_cap = DensePlacement::device_budget(free, total, owned, 0, 0)?;
        let (before, ..) = stored_compact(
            (placement, visits),
            &charged_device,
            &charged_slots,
            &kept_source,
            device_cap,
            sizing.largest,
            granularity,
        )?;
        let (compact, device_budget, host_budget, host_available) = stored_compact(
            (placement, visits),
            &charged_device_after,
            &charged_slots_after,
            &kept_source,
            device_cap,
            sizing.largest,
            granularity,
        )?;
        let facts = PlacementFacts {
            free_bytes: free,
            total_bytes: total,
            owned_bytes: owned,
            largest_bytes: sizing.largest,
            host_available_bytes: host_available,
            device_budget,
            host_budget,
        };
        (compact, Some((before, facts)))
    };
    ensure!(
        compact.slots.len() == kept_index.len() && compact.host.len() == kept_index.len(),
        "stored placement extent"
    );
    let mut slots = vec![None; spec.layers];
    let mut host_flags = vec![false; spec.layers];
    for (index, &layer) in kept_index.iter().enumerate() {
        slots[layer] = compact.slots[index];
        host_flags[layer] = compact.host[index];
    }
    let mut slot_bytes = 0usize;
    for (&layer, slot) in kept_index.iter().zip(&compact.slots) {
        if slot.is_some() {
            let bytes = cursor[layer]
                .checked_add(layer_tails[layer])
                .context("stored slot tail")?;
            slot_bytes = slot_bytes.max(bytes);
        }
    }
    let resident_bytes = kept_device
        .iter()
        .zip(&compact.slots)
        .filter_map(|(&bytes, slot)| slot.is_none().then_some(bytes))
        .sum();
    let plan = DensePlacement {
        slots,
        host: host_flags,
        slot_bytes,
        resident_bytes,
        host_bytes: compact.host_bytes,
        disk_bytes: compact.disk_bytes,
    };
    if let Some((before, facts)) = cold {
        let mut before_slot_bytes = 0usize;
        for (&layer, slot) in kept_index.iter().zip(&before.slots) {
            if slot.is_some() {
                before_slot_bytes = before_slot_bytes.max(cursor[layer]);
            }
        }
        let used_slots = compact
            .slots
            .iter()
            .flatten()
            .copied()
            .max()
            .map(|index| index + 1)
            .unwrap_or(0);
        let charged_slot_bytes = used_slots
            .checked_mul(compact.slot_bytes)
            .context("stored slot bytes")?;
        let predicted_free = facts
            .free_bytes
            .checked_sub(facts.owned_bytes)
            .and_then(|bytes| bytes.checked_sub(compact.resident_bytes))
            .and_then(|bytes| bytes.checked_sub(charged_slot_bytes))
            .and_then(|bytes| {
                bytes.checked_sub(
                    compact.slots.iter().filter(|slot| slot.is_some()).count() * granularity,
                )
            })
            .context("stored placement physical extent exceeds free VRAM")?;
        let (before_resident, before_host, before_slots) = placement_counts(&before);
        let (after_resident, after_host, after_slots) = placement_counts(&compact);
        eprintln!(
            "{}",
            serde_json::json!({"h3_dense_placement": &plan, "mode": format!("{placement:?}"), "streaming_slots": used_slots, "free_bytes": facts.free_bytes, "total_bytes": facts.total_bytes, "owned_block_bytes": facts.owned_bytes, "largest_bytes": facts.largest_bytes, "device_budget": facts.device_budget, "host_budget": facts.host_budget, "host_available_bytes": facts.host_available_bytes, "allocation_granularity": granularity, "charged_resident_bytes": compact.resident_bytes, "charged_slot_bytes": compact.slot_bytes, "predicted_min_driver_free_after_placement": predicted_free})
        );
        eprintln!(
            "{}",
            serde_json::json!({"h3_dense_placement_tail": {"before_resident": before_resident, "before_host": before_host, "before_slots": before_slots, "before_slot_bytes": before_slot_bytes, "after_resident": after_resident, "after_host": after_host, "after_slots": after_slots, "after_slot_bytes": slot_bytes, "last_short_overread_bytes": last_short_overread, "max_past_end_bytes": max_past_end, "layer_tail_bytes": layer_tail_bytes}})
        );
    }
    let mut tier = None;
    if plan.slot_bytes > 0 {
        let dir = base.reader.dir.clone();
        let fp16 = cubin::capability(&d.ctx)? < 80;
        let mut dense = crate::backend::cuda::weights::DenseStreaming::new(
            &dir,
            base.manifest,
            plan,
            d,
            Some(&order),
        )?;
        let mut requests = Vec::new();
        let mut checks = Vec::new();
        let mut largest = 0usize;
        for (layer, names) in order.iter().enumerate() {
            if !dense.plan.host[layer] {
                continue;
            }
            let host = dense.host_layer(layer)?.with_context(|| {
                format!("transformer_blocks.{layer}: host tier without a pinned copy")
            })?;
            ensure!(
                host.len() == sizing.source_layers[layer],
                "transformer_blocks.{layer}: pinned copy size"
            );
            let mut offset = 0usize;
            for name in names {
                let entry = base
                    .manifest
                    .entries
                    .get(name)
                    .with_context(|| format!("manifest lacks {name}"))?
                    .clone();
                ensure!(offset.is_multiple_of(16), "stored factor alignment");
                ensure!(
                    matches!(entry.dtype, Dtype::BF16 | Dtype::I8),
                    "{name}: stored dtype {:?}",
                    entry.dtype
                );
                let end = offset
                    .checked_add(entry.bytes)
                    .with_context(|| format!("{name} stored host offset"))?;
                requests.push(OwnedRead {
                    name: name.clone(),
                    entry: entry.clone(),
                    dest: unsafe { host.as_mut_ptr().add(offset) },
                    dest_len: entry.bytes,
                    cast: weight_fill_cast(fp16, &entry),
                });
                if entry.dtype == Dtype::BF16 {
                    checks.push(FillCheck {
                        layer,
                        tensor: name.clone(),
                        dtype: entry.dtype,
                        quant: entry.quant,
                        at: offset,
                        bytes: entry.bytes,
                    });
                }
                largest = largest.max(entry.bytes);
                offset = end;
            }
            ensure!(
                offset == host.len(),
                "transformer_blocks.{layer}: pinned copy size"
            );
        }
        let slot_count = used_dense_slots(&dense.plan);
        let slot_bytes = dense.plan.slot_bytes;
        tier = Some(DenseTier::schedule(
            "dit",
            dense,
            &dir,
            requests,
            checks,
            (slot_count, slot_bytes, largest),
        )?);
    }
    Ok((tier, packed))
}

fn require_resident_norms(
    s: &Arc<CudaStream>,
    blocks: &[BlockWeights],
    norm_out: &CudaSlice<bf16>,
    hidden: usize,
) -> Result<()> {
    ensure!(
        norm_out.len() == hidden && pointer(norm_out, s) != 0,
        "norm_out.norm.weight is not device-resident"
    );
    for (layer, block) in blocks.iter().enumerate() {
        for (name, weight) in [("norm1", &block.norm1), ("norm2", &block.norm2)] {
            let owner = weight
                ._owned
                .as_ref()
                .map(|v| (pointer(v, s), v.len() * std::mem::size_of::<bf16>()))
                .or_else(|| weight._arena.as_ref().map(|v| (pointer(v, s), v.len())));
            let bytes = hidden * std::mem::size_of::<bf16>();
            ensure!(
                weight.tensor.span_bytes == bytes as u64
                    && owner.is_some_and(|(base, len)| {
                        weight
                            .tensor
                            .ptr
                            .checked_sub(base)
                            .and_then(|offset| offset.checked_add(bytes as u64))
                            .is_some_and(|end| end <= len as u64)
                    }),
                "transformer_blocks.{layer}.{name}.weight is not device-resident"
            );
        }
    }
    Ok(())
}

fn fuse_cross_block(kernels: &BlockKernels) -> bool {
    #[cfg(test)]
    return !kernels.serial_norm_modulate.get();
    #[cfg(not(test))]
    {
        let _ = kernels;
        true
    }
}

fn scale2_tail(
    blocks: &[BlockWeights],
    layer: usize,
    src: &Scale2Sources,
) -> Result<Option<NormTail>> {
    if blocks[layer].dropped {
        return Ok(None);
    }
    if let Some(next) = (layer + 1..blocks.len()).find(|&i| !blocks[i].dropped) {
        let end = next
            .checked_add(1)
            .and_then(|n| n.checked_mul(src.block_rows))
            .context("modulation table extent")?;
        ensure!(end <= src.block_table_len, "modulation table extent");
        let base = src.table + 2 * (next * src.block_rows) as u64;
        Ok(Some(NormTail {
            weight: blocks[next].norm1.tensor.ptr,
            shift: base,
            scale: base + (2 * src.hidden) as u64,
            index: src.row_pairs,
            stride: 6 * src.hidden,
            entries: src.pairs,
            eps: src.norm_eps,
            arms_norm1: true,
        }))
    } else {
        ensure!(
            src.norm_table_len == src.timesteps as usize * 2 * src.hidden,
            "output-norm table extent"
        );
        Ok(Some(NormTail {
            weight: src.norm_out_weight,
            shift: src.norm_out_table,
            scale: src.norm_out_table + (2 * src.hidden) as u64,
            index: src.row_timesteps,
            stride: 2 * src.hidden,
            entries: src.timesteps,
            eps: src.final_eps,
            arms_norm1: false,
        }))
    }
}

fn forward_block<const LORA: bool>(
    c: &BlockContext<'_>,
    flash: &mut Flash,
    weights: &BlockWeights,
    conditioning: &Conditioning<'_>,
    x: &mut CudaSlice<bf16>,
    work: &mut BlockWork,
) -> Result<()> {
    block_forward::<LORA>(c, flash, weights, conditioning, x, work)
}

#[cfg(test)]
struct SampleTrace {
    video: Vec<f32>,
    audio: Vec<f32>,
    video_velocity: Vec<f32>,
    audio_velocity: Vec<f32>,
}

/// The transformer over one request layout: F32 patch and output projections, BF16 text embedder, norms, refiner
/// blocks and LoRA factors resident; the 50 transformer blocks' matrices placed by `DensePlacement` from the sensed
/// VRAM and host budgets (resident, or device slots fed from pinned host memory or the disk tier).
/// A HyperFlow kept block loads its stored int8, low-rank, or dense factor. A dropped block is identity.
pub(crate) struct Dit {
    ranges: Ranges,
    proj_in: [CudaSlice<f32>; 2],
    audio_proj_in: [CudaSlice<f32>; 2],
    proj_out: [CudaSlice<f32>; 2],
    audio_proj_out: [CudaSlice<f32>; 2],
    context: [CudaSlice<bf16>; 2],
    final_norm: CudaSlice<bf16>,
    norm_out: CudaSlice<bf16>,
    refiners: Vec<BlockWeights>,
    blocks: Vec<BlockWeights>,
    tier: Option<DenseTier>,
    rotary: CudaSlice<bf16>,
    x: CudaSlice<bf16>,
    work: BlockWork,
    flash: Flash,
    text: CudaSlice<bf16>,
    text_work: BlockWork,
    text_flash: Flash,
    scratch: CudaSlice<f32>,
    upcast: crate::backend::gpu::CudaFunction,
    bias: crate::backend::gpu::CudaFunction,
    round: crate::backend::gpu::CudaFunction,
    euler: crate::backend::gpu::CudaFunction,
    /// Velocity of the generated video rows, `[generated rows, 96]`.
    pub video: CudaSlice<f32>,
    /// Velocity of the audio rows, `[audio rows, 32]`.
    pub audio: CudaSlice<f32>,
    #[cfg(test)]
    trace: Option<Vec<SampleTrace>>,
    #[cfg(test)]
    trace_gate: Option<(
        &'static str,
        Rc<std::cell::RefCell<pipeline::CompositionGate>>,
    )>,
    #[cfg(test)]
    refiner_trace: Option<Vec<bf16>>,
}

fn bind_video_ffn(
    (d, blas): (&Device, &mut Blas),
    (spec, r): (&H3Spec, usize),
    (work, text_work, x): (&mut BlockWork, &BlockWork, &CudaSlice<bf16>),
    (transient, pending): (usize, usize),
) -> Result<()> {
    let s = &d.stream;
    let h = spec.hidden;
    let hyperflow = spec.is_hyperflow()?;
    let (free, total) = d.ctx.mem_get_info()?;
    let reserve = (total / 16)
        .checked_add(transient)
        .and_then(|bytes| bytes.checked_add(pending))
        .context("FFN live reserve overflow")?;
    if !hyperflow {
        for (output, input) in [(r, h), (r, spec.ffn), (2 * spec.ffn, r), (h, r)] {
            blas.bind_row_family(crate::backend::setup::LinearShape {
                rows: FFN_CHUNK_ROWS,
                output,
                input,
            })?;
        }
    }
    let budget = d.workspace_budget(free, free, reserve)?;
    let (choices, owned, chosen) = if hyperflow {
        let choices = work.ffn_choices::<false>(blas, spec, budget)?;
        let owned = FfnPlan::select(&choices, budget)?;
        let aliases = work.alias_choices::<false>(blas, spec, owned)?;
        let chosen = FfnPlan::select(&aliases, budget)?;
        (choices, owned, chosen)
    } else {
        let choices = work.ffn_choices::<true>(blas, spec, budget)?;
        let owned = FfnPlan::select(&choices, budget)?;
        let aliases = work.alias_choices::<true>(blas, spec, owned)?;
        let chosen = FfnPlan::select(&aliases, budget)?;
        (choices, owned, chosen)
    };
    let previous = *choices
        .iter()
        .find(|p| p.chunk == work.chunk)
        .context("main FFN baseline missing")?;
    if work.rows > FFN_CHUNK_ROWS {
        if hyperflow {
            work.ffn_cost::<false>(blas, spec, work.chunk, true)?;
        } else {
            work.ffn_cost::<true>(blas, spec, work.chunk, true)?;
        }
    }
    let description = |plan: FfnPlan| {
        serde_json::json!({
            "rows":work.rows,"previous_chunk":work.chunk,"chunk":plan.chunk,
            "storage":if plan.borrowed {"attention_scratch"} else {"owned"},
            "chunks":work.rows.div_ceil(plan.chunk),"tail":work.rows%plan.chunk,
            "new_allocation_bytes":plan.allocation,"estimated_tensor_cost":plan.tensor_cost,
            "estimated_launches_per_block":plan.launches,"previous_tensor_cost":previous.tensor_cost,
            "previous_launches_per_block":previous.launches})
    };
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_ffn_plan":{
                "workspace_budget":budget,"free_bytes":free,"live_reserve_bytes":reserve,
                "peak_additional_bytes":chosen.allocation+chosen.workspace,
                "lt_workspace_replacement_bytes":chosen.workspace,"main":description(chosen),
                "owned_alternative":description(owned),
                "alias_capacity_rows":FfnPlan::alias_capacity((work.q_raw.len(),work.k_raw.len()),spec.ffn)?,
                "refiner":{"rows":text_work.rows,"previous_chunk":text_work.chunk,"chunk":text_work.chunk,
                    "chunks":text_work.rows.div_ceil(text_work.chunk),"tail":text_work.rows%text_work.chunk,
                    "new_allocation_bytes":0,"ordinary_plans":true}}})
        );
    }
    blas.bind_workspace(chosen.workspace)?;
    if hyperflow {
        work.bind_ffn::<false>(s, blas, spec, chosen, x)?;
        if spec.int8_convrot.is_some() {
            work.product[FF_IN] = crate::convrot::gate_up_product_order(
                d,
                [work.chunk, work.rows % work.chunk],
                (h, 2 * spec.ffn),
            )?;
            let width = spec.heads * spec.head_dim;
            work.plan_rotation(d, spec)?;
            work.qkv_product = crate::convrot::n_fast_product(d, work.rows, (h, width))?;
            if d.info.arch == crate::backend::setup::DeviceArchitecture::Cuda(120) {
                work.product[OUT][0] = crate::convrot::n_fast_product(d, work.rows, (width, h))?;
                for (i, rows) in [work.chunk, work.rows % work.chunk].into_iter().enumerate() {
                    work.product[FF_OUT][i] =
                        crate::convrot::n_fast_product(d, rows, (spec.ffn, h))?;
                }
            }
            if crate::backend::diagnostics_enabled() {
                eprintln!(
                    "CONVROT_QKV_PRODUCT_SELECT rows={} k={h} n={width} l2_bytes={} kernel={}",
                    work.rows,
                    d.info.l2_bytes,
                    work.qkv_product.symbol_name()
                );
                eprintln!(
                    "CONVROT_N_FAST_PRODUCTS out={:?} down={:?}",
                    work.product[OUT][0], work.product[FF_OUT]
                );
            }
        }
    } else {
        work.bind_ffn::<true>(s, blas, spec, chosen, x)?;
    }
    s.synchronize()?;
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_ffn_storage":{
                "hidden_ptr":work.hidden.ptr,"hidden_shape":work.hidden.shape,"hidden_span_bytes":work.hidden.span_bytes,
                "gated_ptr":work.gated.ptr,"gated_shape":work.gated.shape,"gated_span_bytes":work.gated.span_bytes,
                "q_raw_ptr":pointer(&work.q_raw,s),"q_raw_bytes":2*work.q_raw.len(),
                "k_raw_ptr":pointer(&work.k_raw,s),"k_raw_bytes":2*work.k_raw.len(),
                "owned":work.ffn_owned.is_some(),"free_after_binding":d.ctx.mem_get_info()?.0}})
        );
    }
    Ok(())
}

type VideoPlacementChooser<'a> = dyn FnMut(
        &DenseSizing,
        &[usize],
        usize,
        &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<crate::prefill::DensePlacement>
    + 'a;

impl Dit {
    #[cfg(test)]
    pub(crate) fn new(
        d: &Device,
        blas: &mut Blas,
        spec: &H3Spec,
        layout: (usize, &crate::minimax_h3::PackedLayout),
        base: Source<'_>,
        lora: Option<Source<'_>>,
        (placement, visits): (Placement, usize),
    ) -> Result<Self> {
        Self::with_placement(d, blas, spec, layout, base, lora, (placement, visits, None))
    }

    fn with_placement(
        d: &Device,
        blas: &mut Blas,
        spec: &H3Spec,
        (rank, layout): (usize, &crate::minimax_h3::PackedLayout),
        mut base: Source<'_>,
        mut lora: Option<Source<'_>>,
        (placement, visits, choose): (Placement, usize, Option<&mut VideoPlacementChooser<'_>>),
    ) -> Result<Self> {
        let hyperflow = spec.is_hyperflow()?;
        ensure!(
            if hyperflow {
                rank == 0 && lora.is_none()
            } else {
                lora.is_some()
            },
            "HyperFlow requires no LoRA and rank0; MiniMax-H3 requires a LoRA source"
        );
        let s = &d.stream;
        let ranges = Ranges::of(layout)?;
        let (h, r) = (spec.hidden, rank);
        let patch = spec.video_row_width();
        let mut pair = |name: &str, n: usize, k: usize| -> Result<[CudaSlice<f32>; 2]> {
            Ok([
                base.tensor(s, &format!("{name}.weight"), n * k)?,
                base.tensor(s, &format!("{name}.bias"), n)?,
            ])
        };
        let proj_in = pair("proj_in", h, patch)?;
        let audio_proj_in = pair("audio_proj_in", h, spec.audio_in_channels)?;
        let proj_out = pair("proj_out", patch, h)?;
        let audio_proj_out = pair("audio_proj_out", spec.audio_in_channels, h)?;
        let context = [
            base.tensor(s, "context_embedder.weight", h * spec.text_dim)?,
            base.tensor(s, "context_embedder.bias", h)?,
        ];
        let final_norm = base.tensor(s, "token_refiner.final_norm.weight", h)?;
        let norm_out = base.tensor(s, "norm_out.norm.weight", h)?;
        let refiners = if hyperflow {
            (0..spec.refiner_layers)
                .map(|i| {
                    BlockWeights::resident::<false>(
                        s,
                        spec,
                        r,
                        &format!("token_refiner.refiner_blocks.{i}"),
                        &mut base,
                        None,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..spec.refiner_layers)
                .map(|i| {
                    BlockWeights::resident::<true>(
                        s,
                        spec,
                        r,
                        &format!("token_refiner.refiner_blocks.{i}"),
                        &mut base,
                        lora.as_mut(),
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };
        blas.prepare_biased(
            u32::try_from(ranges.text)?,
            u32::try_from(h)?,
            u32::try_from(spec.text_dim)?,
            vector(&context[1], s)?,
            LinearBiasMode::BeforeOutputRound,
        )?;
        let mut work = BlockWork::new(s, spec, r, ranges.rows)?;
        let text_work = BlockWork::new(s, spec, r, ranges.text)?;
        if hyperflow {
            work.prepare::<false>(blas, spec)?;
            text_work.prepare::<false>(blas, spec)?;
        } else {
            work.prepare::<true>(blas, spec)?;
            text_work.prepare::<true>(blas, spec)?;
        }
        let flash = Flash::new(
            s.clone(),
            ranges.rows,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?
        .with_full_single_probability()?;
        let text_flash = Flash::new(
            s.clone(),
            ranges.text,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?
        .with_full_single_probability()?;
        let generated = ranges.rows - ranges.video_start;
        let scratch_rows = (ranges.condition + generated).max(ranges.audio);
        let x = s.alloc_zeros::<bf16>(ranges.rows * h)?;
        let text = s.alloc_zeros::<bf16>(ranges.text * h)?;
        let scratch = s.alloc_zeros::<f32>(scratch_rows * h)?;
        let video = s.alloc_zeros::<f32>(generated * patch)?;
        let audio = s.alloc_zeros::<f32>(ranges.audio * spec.audio_in_channels)?;
        let rotary = s.clone_htod(&crate::minimax_h3::rotary_table(spec, layout))?;

        let sizing = dense_sizing(base.manifest, spec.layers)?;
        let lora_bytes = if hyperflow {
            0
        } else {
            lora.as_ref()
                .context("MiniMax-H3 LoRA source")?
                .manifest
                .entries
                .iter()
                .filter(|(name, _)| name.starts_with("transformer_blocks."))
                .map(|(_, entry)| entry.bytes)
                .sum::<usize>()
        };
        let mut owned = lora_bytes
            + spec.layers * 2 * (2 * h + 2 * spec.head_dim)
            + sizing.staging
            + sizing.transient;
        if hyperflow {
            let wscale = base
                .manifest
                .entries
                .iter()
                .filter(|(name, entry)| {
                    name.starts_with("transformer_blocks.")
                        && name.ends_with(".wscale")
                        && entry.load
                })
                .try_fold(0usize, |sum, (_, entry)| {
                    sum.checked_add(entry.bytes)
                        .context("HyperFlow wscale bytes")
                })?;
            owned = owned
                .checked_add(wscale)
                .context("HyperFlow wscale bytes")?;
        }
        let selected = if let Some(choose) = choose {
            let mut bind_work = |pending| {
                bind_video_ffn(
                    (d, blas),
                    (spec, r),
                    (&mut work, &text_work, &x),
                    (0, pending),
                )
            };
            Some(if hyperflow {
                let page = stored_allocation_granularity(d)?;
                let (_, owned) = stored_allocation_costs(base.manifest, spec, page, owned)?;
                let charged = (0..spec.layers)
                    .map(|layer| stored_layer_charge(base.manifest, layer, page))
                    .collect::<Result<Vec<_>>>()?;
                choose(&sizing, &charged, owned, &mut bind_work)?
            } else {
                choose(&sizing, &sizing.layers, owned, &mut bind_work)?
            })
        } else {
            None
        };
        let resident = selected.is_some();
        let (tier, blocks) = if hyperflow {
            let (tier, packed) = place_stored_layers(
                d,
                &mut base,
                spec,
                &sizing,
                owned,
                (placement, visits, selected),
            )?;
            #[cfg(test)]
            trace_placement_pool(d, "after_streaming_slots")?;
            base.pack_vectors(s, spec)?;
            let dropped = spec
                .dropped_blocks
                .as_deref()
                .context("HyperFlow dropped blocks")?;
            let mut blocks = Vec::with_capacity(spec.layers);
            for (layer, factors) in packed.iter().enumerate() {
                #[cfg(test)]
                let allocation_before = d.ctx.mem_get_info()?.0;
                if dropped.contains(&layer) {
                    blocks.push(BlockWeights::identity(s, spec, &mut base, layer)?);
                    #[cfg(test)]
                    eprintln!(
                        "{}",
                        serde_json::json!({"h3_allocation_block": {
                            "layer": layer, "dropped": true,
                            "free_before": allocation_before,
                            "free_after": d.ctx.mem_get_info()?.0,
                            "synchronized": false
                        }})
                    );
                    continue;
                }
                let name = format!("transformer_blocks.{layer}");
                let bytes = sizing.layers[layer];
                let slot = match tier.as_ref().map(|tier| &tier.dense) {
                    Some(streaming) if streaming.plan.slots[layer].is_some() => Some((
                        streaming.pointer(layer, 0, bytes)?,
                        factors.as_slice(),
                        streaming.plan.slot_bytes,
                    )),
                    _ => None,
                };
                blocks.push(BlockWeights::stored(s, spec, &name, &mut base, slot)?);
                #[cfg(test)]
                eprintln!(
                    "{}",
                    serde_json::json!({"h3_allocation_block": {
                        "layer": layer, "dropped": false,
                        "resident": slot.is_none(), "logical_matrix_bytes": bytes,
                        "free_before": allocation_before,
                        "free_after": d.ctx.mem_get_info()?.0,
                        "synchronized": false
                    }})
                );
            }
            #[cfg(test)]
            trace_placement_pool(d, "after_stored_blocks")?;
            (tier, blocks)
        } else {
            let tier = if let Some(selected) = selected {
                bind_layers(
                    "dit",
                    d,
                    &mut base,
                    &sizing.source_layers,
                    selected,
                    |layer| format!("transformer_blocks.{layer}"),
                    &LINEAR_ROLES,
                )?
            } else {
                place_layers(
                    "dit",
                    d,
                    &mut base,
                    (
                        &sizing.layers,
                        &sizing.source_layers,
                        sizing.largest,
                        owned,
                        0,
                    ),
                    (placement, visits),
                    |layer| format!("transformer_blocks.{layer}"),
                    &LINEAR_ROLES,
                )?
                .tier
            };
            let mut blocks = Vec::with_capacity(spec.layers);
            for (layer, &bytes) in sizing.layers.iter().enumerate() {
                let name = format!("transformer_blocks.{layer}");
                let slot = match tier.as_ref().map(|tier| &tier.dense) {
                    Some(streaming) if streaming.plan.slots[layer].is_some() => {
                        Some(streaming.pointer(layer, 0, bytes)?)
                    }
                    _ => None,
                };
                blocks.push(match slot {
                    Some(slot) => BlockWeights::streamed::<true>(
                        s,
                        spec,
                        r,
                        &name,
                        slot,
                        &mut base,
                        lora.as_mut(),
                    )?,
                    None => {
                        BlockWeights::resident::<true>(s, spec, r, &name, &mut base, lora.as_mut())?
                    }
                });
            }
            (tier, blocks)
        };
        if !resident {
            bind_video_ffn(
                (d, blas),
                (spec, r),
                (&mut work, &text_work, &x),
                (sizing.transient, 0),
            )?;
        }
        let vision = cubin::module(s.context(), "vision_ops")?;
        #[cfg(test)]
        if let Some((layout, buffer)) = &base.vectors {
            let expected = base
                .vector_bytes
                .as_ref()
                .context("vector arena upload bytes missing")?;
            ensure!(
                s.clone_dtoh(buffer.as_ref())?.as_slice() == expected.as_slice(),
                "RUNNER_FAIL vector arena upload changed a byte"
            );
            eprintln!(
                "{}",
                serde_json::json!({"h3_vector_arena": {
                    "tensors": layout.offsets.len(), "storage_bytes": layout.bytes,
                    "logical_bytes": layout.offsets.values().map(|(_, bytes, _)| bytes).sum::<usize>(),
                    "alignment": 256, "device_allocations": 1, "upload_byte_differences": 0
                }})
            );
        }
        require_resident_norms(s, &blocks, &norm_out, h)?;
        Ok(Self {
            ranges,
            proj_in,
            audio_proj_in,
            proj_out,
            audio_proj_out,
            context,
            final_norm,
            norm_out,
            refiners,
            blocks,
            tier,
            rotary,
            x,
            work,
            flash,
            text,
            text_work,
            text_flash,
            scratch,
            upcast: vision.load_function("vision_upcast_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            round: vision.load_function("vision_affine_cast_bf16")?,
            euler: cubin::module(s.context(), "flow")?.load_function("euler_blend_f32")?,
            video,
            audio,
            #[cfg(test)]
            trace: None,
            #[cfg(test)]
            trace_gate: None,
            #[cfg(test)]
            refiner_trace: None,
        })
    }

    /// `x = r * x + (1 - r) * (x + (1 - t) * v)` over `count` F32 latents at element `offset` of `x`, from grid
    /// points `k` and `k + 1`.
    fn euler(
        &self,
        s: &Arc<CudaStream>,
        (x, offset, count): (&CudaSlice<f32>, usize, usize),
        v: &CudaSlice<f32>,
        grid: &crate::minimax_h3::Schedule,
        k: usize,
    ) -> Result<()> {
        ensure!(
            v.len() == count && offset + count <= x.len(),
            "Euler extent"
        );
        let step = 1.0f32 - grid.timesteps[k];
        let ratio = grid.sigmas[k + 1] / grid.sigmas[k];
        unsafe {
            s.launch_builder(&self.euler)
                .arg(&(pointer(x, s) + 4 * offset as u64))
                .arg(&pointer(v, s))
                .arg(&(count as u64))
                .arg(&step)
                .arg(&ratio)
                .arg(&(1.0f32 - ratio))
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// The sampling chain in place on `[video rows, 96]` and `[audio rows, 32]` F32 latents: per evaluation of
    /// `plan`, the forward, then the Euler step of the generated video rows on the video grid and of every audio row
    /// on the audio grid; conditioning video rows stay fixed.
    pub(crate) fn sample(
        &mut self,
        c: &BlockContext<'_>,
        adaln: (&AdalnTable, &AdalnPlan),
        (video_grid, audio_grid): (&crate::minimax_h3::Schedule, &crate::minimax_h3::Schedule),
        (video, audio): (&CudaSlice<f32>, &CudaSlice<f32>),
        text: &CudaSlice<bf16>,
    ) -> Result<()> {
        let (table, plan) = adaln;
        let s = c.s;
        let patch = c.spec.video_row_width();
        let counts = (
            u32::try_from(plan.pairs.len())?,
            u32::try_from(plan.timesteps.len())?,
        );
        ensure!(
            plan.row_pairs.len() == video_grid.timesteps.len()
                && plan.row_pairs.len() == audio_grid.timesteps.len(),
            "plan and grids differ in evaluations"
        );
        let condition = self.ranges.condition;
        let generated = self.ranges.rows - self.ranges.video_start;
        for k in 0..plan.row_pairs.len() {
            #[cfg(test)]
            if let Some((case, _)) = &self.trace_gate {
                println!(
                    "HYPERFLOW_FORWARD case={case} phase=free step={k} state=START utc={}",
                    pipeline::composition_stamp()?
                );
            }
            let row_pairs = s.clone_htod(&plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&plan.row_timesteps[k])?;
            self.step(
                c,
                table,
                counts,
                &StepInputs {
                    video,
                    audio,
                    text,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            self.euler(
                s,
                (video, condition * patch, generated * patch),
                &self.video,
                video_grid,
                k,
            )?;
            self.euler(s, (audio, 0, audio.len()), &self.audio, audio_grid, k)?;
            #[cfg(test)]
            if let Some(trace) = &mut self.trace {
                trace.push(SampleTrace {
                    video: s.clone_dtoh(video)?,
                    audio: s.clone_dtoh(audio)?,
                    video_velocity: s.clone_dtoh(&self.video)?,
                    audio_velocity: s.clone_dtoh(&self.audio)?,
                });
                if let Some((case, gate)) = &self.trace_gate {
                    gate.borrow_mut().sample_step(
                        case,
                        k,
                        trace.last().context("Runner FAIL free trace missing")?,
                    )?;
                }
            }
        }
        s.synchronize()?;
        Ok(())
    }
    /// Joins the streaming fills and reports whether the placement keeps every layer off the
    /// disk tier (the hoist guard).
    pub(crate) fn hoist_ready(&mut self) -> Result<bool> {
        match self.tier.as_mut() {
            Some(tier) => {
                tier.ready()?;
                Ok(tier.dense.plan.disk_bytes == 0)
            }
            None => Ok(true),
        }
    }

    /// One teacher-forced evaluation; the velocities land in `self.video` and `self.audio`.
    pub(crate) fn step(
        &mut self,
        c: &BlockContext<'_>,
        adaln: &AdalnTable,
        (pairs, timesteps): (u32, u32),
        inputs: &StepInputs<'_>,
    ) -> Result<()> {
        if let Some(tier) = self.tier.as_mut() {
            tier.ready()?;
        }
        let (s, spec, r) = (c.s, c.spec, self.ranges);
        let h = spec.hidden;
        let video_rows = r.condition + (r.rows - r.video_start);
        let patch = spec.video_row_width();
        ensure!(
            inputs.video.len() == video_rows * patch
                && inputs.audio.len() == r.audio * spec.audio_in_channels
                && inputs.text.len() == r.text * spec.text_dim
                && inputs.row_pairs.len() == r.rows
                && inputs.row_timesteps.len() == r.rows,
            "evaluation inputs differ from the layout"
        );
        c.blas.linear_biased(
            matrix(&self.context[0], s, h, spec.text_dim)?,
            vector(&self.context[1], s)?,
            matrix(inputs.text, s, r.text, spec.text_dim)?,
            &mut TensorMut(matrix(&self.text, s, r.text, h)?),
            LinearBiasMode::BeforeOutputRound,
        )?;
        for refiner in &self.refiners {
            if spec.is_hyperflow()? {
                forward_block::<false>(
                    c,
                    &mut self.text_flash,
                    refiner,
                    &Conditioning::Plain,
                    &mut self.text,
                    &mut self.text_work,
                )?;
            } else {
                forward_block::<true>(
                    c,
                    &mut self.text_flash,
                    refiner,
                    &Conditioning::Plain,
                    &mut self.text,
                    &mut self.text_work,
                )?;
            }
        }
        c.kernels.rms_norm_f32(
            s,
            NormRows {
                input: pointer(&self.text, s),
                weight: pointer(&self.final_norm, s),
                output: pointer(&self.x, s),
                rows: r.text,
                dim: h,
                threads: HIDDEN_NORM_THREADS,
                eps: spec.final_norm_eps,
            },
        )?;
        // `[text, hidden]` refiner rows. The block stack then writes `x`.
        #[cfg(test)]
        {
            let count = r.text * h;
            s.memcpy_dtod(&self.x.slice(..count), &mut self.text.slice_mut(..count))?;
        }
        #[cfg(test)]
        if self.trace.as_ref().is_some_and(Vec::is_empty) && self.refiner_trace.is_none() {
            self.refiner_trace = Some(s.clone_dtoh(&self.x.slice(..r.text * h))?);
        }
        projection(
            c.s,
            c.blas,
            &self.bias,
            &self.proj_in,
            (inputs.video, 0),
            (&self.scratch, video_rows),
        )?;
        let scratch = pointer(&self.scratch, s);
        round_into(
            s,
            &self.round,
            &self.x,
            scratch,
            r.text * h,
            r.condition * h,
        )?;
        round_into(
            s,
            &self.round,
            &self.x,
            scratch + 4 * (r.condition * h) as u64,
            r.video_start * h,
            (video_rows - r.condition) * h,
        )?;
        projection(
            c.s,
            c.blas,
            &self.bias,
            &self.audio_proj_in,
            (inputs.audio, 0),
            (&self.scratch, r.audio),
        )?;
        round_into(
            s,
            &self.round,
            &self.x,
            scratch,
            r.audio_start * h,
            r.audio * h,
        )?;

        let block_rows = 6 * h * pairs as usize;
        let sources = Scale2Sources {
            hidden: h,
            block_rows,
            pairs,
            timesteps,
            norm_eps: spec.norm_eps,
            final_eps: spec.final_norm_eps,
            table: pointer(&adaln.blocks, s),
            block_table_len: adaln.blocks.len(),
            norm_out_weight: pointer(&self.norm_out, s),
            norm_out_table: pointer(&adaln.norm_out, s),
            norm_table_len: adaln.norm_out.len(),
            row_pairs: pointer(inputs.row_pairs, s),
            row_timesteps: pointer(inputs.row_timesteps, s),
        };
        let fuse = fuse_cross_block(c.kernels);
        let mut fused_norm_out = false;
        self.work.skip_norm1 = false;
        for layer in 0..self.blocks.len() {
            if let Some(tier) = self.tier.as_mut() {
                tier.ready()?.begin(layer, s)?;
            }
            let modulation = Modulation {
                table: &adaln.blocks,
                offset: layer * block_rows,
                entries: pairs,
                index: inputs.row_pairs,
            };
            let tail = if fuse {
                scale2_tail(&self.blocks, layer, &sources)?
            } else {
                None
            };
            fused_norm_out |= tail.as_ref().is_some_and(|tail| !tail.arms_norm1);
            let conditioning = Conditioning::Modulated {
                rotary: &self.rotary,
                modulation: &modulation,
            };
            if spec.is_hyperflow()? {
                block_forward_tail::<false>(
                    c,
                    &mut self.flash,
                    &self.blocks[layer],
                    &conditioning,
                    &mut self.x,
                    &mut self.work,
                    tail,
                )?;
            } else {
                block_forward_tail::<true>(
                    c,
                    &mut self.flash,
                    &self.blocks[layer],
                    &conditioning,
                    &mut self.x,
                    &mut self.work,
                    tail,
                )?;
            }
            if let Some(tier) = self.tier.as_mut() {
                tier.ready()?.end(layer, s)?;
            }
        }
        if !fused_norm_out {
            let normed = pointer(&self.work.normed, s);
            ensure!(
                adaln.norm_out.len() == timesteps as usize * 2 * h,
                "output-norm table extent"
            );
            let table = pointer(&adaln.norm_out, s);
            c.kernels.rms_norm_modulate_f32(
                s,
                NormRows {
                    input: pointer(&self.x, s),
                    weight: pointer(&self.norm_out, s),
                    output: normed,
                    rows: r.rows,
                    dim: h,
                    threads: HIDDEN_NORM_THREADS,
                    eps: spec.final_norm_eps,
                },
                ModulateRows {
                    data: normed,
                    shift: table,
                    scale: table + 2 * h as u64,
                    index: pointer(inputs.row_timesteps, s),
                    rows: r.rows,
                    dim: h,
                    stride: 2 * h,
                    entries: timesteps,
                },
            )?;
        }
        let generated = r.rows - r.video_start;
        upcast_from(
            s,
            &self.upcast,
            &self.work.normed,
            &self.scratch,
            r.video_start * h,
            generated * h,
        )?;
        projection(
            c.s,
            c.blas,
            &self.bias,
            &self.proj_out,
            (&self.scratch, 0),
            (&self.video, generated),
        )?;
        upcast_from(
            s,
            &self.upcast,
            &self.work.normed,
            &self.scratch,
            r.audio_start * h,
            r.audio * h,
        )?;
        projection(
            c.s,
            c.blas,
            &self.bias,
            &self.audio_proj_out,
            (&self.scratch, 0),
            (&self.audio, r.audio),
        )?;
        Ok(())
    }
}

/// Block geometry of an image transformer. `audio_in_channels` is 0.
fn image_block_spec(spec: &ImageSpec) -> H3Spec {
    H3Spec {
        hidden: spec.hidden,
        heads: spec.heads,
        head_dim: spec.head_dim,
        layers: spec.layers,
        refiner_layers: spec.refiner_layers,
        ffn: spec.ffn,
        in_channels: spec.in_channels,
        audio_in_channels: 0,
        patch: spec.patch,
        text_dim: spec.text_dim,
        freq_dim: spec.freq_dim,
        time_hidden: spec.time_hidden,
        time_dim: spec.time_dim,
        rope_freq_dim: spec.rope_freq_dim,
        rope_theta: spec.rope_theta,
        norm_eps: spec.norm_eps,
        qk_norm_eps: spec.qk_norm_eps,
        final_norm_eps: spec.final_norm_eps,
        hyperflow_gate: None,
        dropped_blocks: None,
        lowrank_ranks: None,
        int8_convrot: None,
    }
}

/// One image evaluation: patchified image rows `[image rows, 96]` in F32 (references first), text states
/// `[text rows, text_dim]` in BF16, and the AdaLN pair and timestep of every packed row.
pub(crate) struct ImageStepInputs<'a> {
    pub image: &'a CudaSlice<f32>,
    pub text: &'a CudaSlice<bf16>,
    pub row_pairs: &'a CudaSlice<u32>,
    pub row_timesteps: &'a CudaSlice<u32>,
}

/// Image transformer over one `[text | reference | generated]` layout. No audio projection and no LoRA. Six F32
/// `pdd_proj_out` heads; head `k` is evaluation `k`. The block matrices are placed by `DensePlacement`.
pub(crate) struct ImageDit {
    text_rows: usize,
    image_rows: usize,
    reference_rows: usize,
    row_width: usize,
    proj_in: [CudaSlice<f32>; 2],
    heads: Vec<[CudaSlice<f32>; 2]>,
    context: [CudaSlice<bf16>; 2],
    final_norm: CudaSlice<bf16>,
    norm_out: CudaSlice<bf16>,
    refiners: Vec<BlockWeights>,
    blocks: Vec<BlockWeights>,
    tier: Option<DenseTier>,
    #[cfg(test)]
    placement: PlacementFacts,
    rotary: CudaSlice<bf16>,
    x: CudaSlice<bf16>,
    work: BlockWork,
    flash: Flash,
    text: CudaSlice<bf16>,
    text_work: BlockWork,
    text_flash: Flash,
    scratch: CudaSlice<f32>,
    /// Velocity of every image row, `[image rows, 96]`, references first.
    pub velocity: CudaSlice<f32>,
    upcast: crate::backend::gpu::CudaFunction,
    bias: crate::backend::gpu::CudaFunction,
    round: crate::backend::gpu::CudaFunction,
    euler: crate::backend::gpu::CudaFunction,
}

type ImageCaptured<'a> = (Option<&'a mut Vec<Vec<f32>>>, Option<&'a [Vec<f32>]>);

type ImagePlacementChooser<'a> = dyn FnMut(
        &DenseSizing,
        usize,
        &mut dyn FnMut(usize) -> Result<FfnPlan>,
    ) -> Result<(crate::prefill::DensePlacement, usize, FfnPlan)>
    + 'a;

enum ImagePlacement<'a> {
    Single(Placement, usize),
    Resident(&'a mut ImagePlacementChooser<'a>),
}

impl ImageDit {
    pub(crate) fn new(
        d: &Device,
        blas: &mut Blas,
        (spec, image): (&H3Spec, &ImageSpec),
        layout: &ImageLayout,
        dir: &Path,
        manifest: &TensorManifest,
        (placement, visits): (Placement, usize),
    ) -> Result<Self> {
        Self::with_placement(
            d,
            blas,
            (spec, image),
            layout,
            dir,
            manifest,
            ImagePlacement::Single(placement, visits),
        )
    }

    fn resident(
        d: &Device,
        blas: &mut Blas,
        specs: (&H3Spec, &ImageSpec),
        layout: &ImageLayout,
        source: (&Path, &TensorManifest),
        choose: &mut ImagePlacementChooser<'_>,
    ) -> Result<Self> {
        Self::with_placement(
            d,
            blas,
            specs,
            layout,
            source.0,
            source.1,
            ImagePlacement::Resident(choose),
        )
    }

    fn with_placement(
        d: &Device,
        blas: &mut Blas,
        (spec, image): (&H3Spec, &ImageSpec),
        layout: &ImageLayout,
        dir: &Path,
        manifest: &TensorManifest,
        placement: ImagePlacement<'_>,
    ) -> Result<Self> {
        ensure!(
            image_block_spec(image) == *spec,
            "image block geometry differs from the image spec"
        );
        let s = &d.stream;
        let text_rows = layout.text_indices.len();
        let image_rows = layout.image_indices.len();
        let rows = layout.rows();
        let reference_rows = layout.reference_rows;
        ensure!(
            text_rows > 0
                && image_rows > reference_rows
                && text_rows + image_rows == rows
                && layout
                    .text_indices
                    .iter()
                    .enumerate()
                    .all(|(i, &row)| row as usize == i)
                && layout
                    .image_indices
                    .iter()
                    .enumerate()
                    .all(|(i, &row)| row as usize == text_rows + i),
            "image layout is not [text | reference | generated]"
        );
        let mut base = Source::new(dir, manifest);
        let (h, row_width) = (spec.hidden, image.row_width());
        let proj_in = [
            base.tensor(s, "proj_in.weight", h * row_width)?,
            base.tensor(s, "proj_in.bias", h)?,
        ];
        let mut heads = Vec::with_capacity(image.pdd_heads);
        for head in 0..image.pdd_heads {
            let name = format!("pdd_proj_out.{head}");
            heads.push([
                base.tensor(s, &format!("{name}.weight"), row_width * h)?,
                base.tensor(s, &format!("{name}.bias"), row_width)?,
            ]);
        }
        let context = [
            base.tensor(s, "context_embedder.weight", h * spec.text_dim)?,
            base.tensor(s, "context_embedder.bias", h)?,
        ];
        let final_norm = base.tensor(s, "token_refiner.final_norm.weight", h)?;
        let norm_out = base.tensor(s, "norm_out.norm.weight", h)?;
        let refiners = (0..spec.refiner_layers)
            .map(|i| {
                BlockWeights::resident::<false>(
                    s,
                    spec,
                    0,
                    &format!("token_refiner.refiner_blocks.{i}"),
                    &mut base,
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        blas.prepare_biased(
            u32::try_from(text_rows)?,
            u32::try_from(h)?,
            u32::try_from(spec.text_dim)?,
            vector(&context[1], s)?,
            LinearBiasMode::BeforeOutputRound,
        )?;
        let mut work = BlockWork::new(s, spec, 0, rows)?;
        work.prepare::<false>(blas, spec)?;
        let text_work = BlockWork::new(s, spec, 0, text_rows)?;
        text_work.prepare::<false>(blas, spec)?;
        let flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let text_flash = Flash::new(s.clone(), text_rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let x = s.alloc_zeros::<bf16>(rows * h)?;
        let text = s.alloc_zeros::<bf16>(text_rows * h)?;
        let scratch = s.alloc_zeros::<f32>(image_rows * h)?;
        let velocity = s.alloc_zeros::<f32>(image_rows * row_width)?;
        let rotary = s.clone_htod(&rotary_rows(
            spec.rope_freq_dim,
            spec.rope_theta,
            &layout.position_ids,
        ))?;
        let sizing = dense_sizing(manifest, spec.layers)?;
        let owned =
            spec.layers * 2 * (2 * h + 2 * spec.head_dim) + sizing.staging + sizing.transient;
        let name = |layer| format!("transformer_blocks.{layer}");
        let bind_ffn = |work: &mut BlockWork, blas: &mut Blas, budget| -> Result<FfnPlan> {
            let choices = work.ffn_choices::<false>(blas, spec, budget)?;
            let owned_plan = FfnPlan::select(&choices, budget)?;
            let aliases = work.alias_choices::<false>(blas, spec, owned_plan)?;
            let plan = FfnPlan::select(&aliases, budget)?;
            blas.bind_workspace(plan.workspace)?;
            work.bind_ffn::<false>(s, blas, spec, plan, &x)?;
            Ok(plan)
        };
        let (placed, device_after, prepared_ffn) = match placement {
            ImagePlacement::Single(mode, visits) => (
                place_layers(
                    "dit",
                    d,
                    &mut base,
                    (
                        &sizing.layers,
                        &sizing.source_layers,
                        sizing.largest,
                        owned,
                        0,
                    ),
                    (mode, visits),
                    name,
                    &LINEAR_ROLES,
                )?,
                0,
                None,
            ),
            ImagePlacement::Resident(choose) => {
                let (plan, device_after, chosen) = choose(&sizing, owned, &mut |later| {
                    let (free, total) = d.ctx.mem_get_info()?;
                    let reserved = (total / 16)
                        .checked_add(later)
                        .context("resident image FFN reserve overflow")?;
                    let budget = d.workspace_budget(free, free, reserved)?;
                    bind_ffn(&mut work, blas, budget)
                })?;
                #[cfg(test)]
                let facts = {
                    let (free, total) = d.ctx.mem_get_info()?;
                    PlacementFacts {
                        free_bytes: free,
                        total_bytes: total,
                        owned_bytes: owned,
                        largest_bytes: sizing.largest,
                        host_available_bytes: None,
                        device_budget: plan.resident_bytes + 2 * plan.slot_bytes,
                        host_budget: plan.host_bytes,
                    }
                };
                let tier = bind_layers(
                    "dit",
                    d,
                    &mut base,
                    &sizing.source_layers,
                    plan,
                    name,
                    &LINEAR_ROLES,
                )?;
                (
                    PlacedLayers {
                        tier,
                        #[cfg(test)]
                        facts,
                    },
                    device_after,
                    Some(chosen),
                )
            }
        };
        #[cfg(test)]
        let placement_facts = placed.facts;
        let tier = placed.tier;
        let mut blocks = Vec::with_capacity(spec.layers);
        for (layer, &bytes) in sizing.layers.iter().enumerate() {
            let name = format!("transformer_blocks.{layer}");
            let slot = match tier.as_ref().map(|tier| &tier.dense) {
                Some(streaming) if streaming.plan.slots[layer].is_some() => {
                    Some(streaming.pointer(layer, 0, bytes)?)
                }
                _ => None,
            };
            blocks.push(match slot {
                Some(slot) => {
                    BlockWeights::streamed::<false>(s, spec, 0, &name, slot, &mut base, None)?
                }
                None => BlockWeights::resident::<false>(s, spec, 0, &name, &mut base, None)?,
            });
        }
        let (free, total) = d.ctx.mem_get_info()?;
        let chosen = match prepared_ffn {
            Some(plan) => {
                let reserve = (total / 16)
                    .checked_add(sizing.transient)
                    .and_then(|n| n.checked_add(device_after))
                    .context("FFN live reserve overflow")?;
                d.workspace_budget(free, free, reserve)?;
                plan
            }
            None => bind_ffn(
                &mut work,
                blas,
                d.workspace_budget(free, free, sizing.transient)?,
            )?,
        };
        s.synchronize()?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({
                    "h3_image_ffn_plan": {
                        "rows": rows,
                        "chunk": chosen.chunk,
                        "allocation": chosen.allocation,
                        "workspace": chosen.workspace,
                    }
                })
            );
        }
        let vision = cubin::module(s.context(), "vision_ops")?;
        require_resident_norms(s, &blocks, &norm_out, h)?;
        Ok(Self {
            text_rows,
            image_rows,
            reference_rows,
            row_width,
            proj_in,
            heads,
            context,
            final_norm,
            norm_out,
            refiners,
            blocks,
            tier,
            #[cfg(test)]
            placement: placement_facts,
            rotary,
            x,
            work,
            flash,
            text,
            text_work,
            text_flash,
            scratch,
            velocity,
            upcast: vision.load_function("vision_upcast_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            round: vision.load_function("vision_affine_cast_bf16")?,
            euler: cubin::module(s.context(), "flow")?.load_function("euler_blend_f32")?,
        })
    }

    pub(crate) fn sample_host(
        &mut self,
        context: &BlockContext<'_>,
        modulation: (&AdalnTable, &AdalnPlan),
        schedule: &ImageSchedule,
        (rows, text): (&[f32], &[bf16]),
        captured: ImageCaptured<'_>,
    ) -> Result<Vec<f32>> {
        let mut image_rows = context.s.clone_htod(rows)?;
        let text = context.s.clone_htod(text)?;
        self.sample(
            context,
            modulation,
            schedule,
            &mut image_rows,
            &text,
            captured,
        )?;
        let out = context.s.clone_dtoh(&image_rows)?;
        ensure!(
            out.len() == rows.len() && out.iter().all(|v| v.is_finite()),
            "non-finite image sample"
        );
        Ok(out)
    }

    /// `x = r * x + (1 - r) * (x + sigma * v)` on the generated image rows. Reference rows stay put.
    fn euler(
        &self,
        s: &Arc<CudaStream>,
        image: &CudaSlice<f32>,
        schedule: &ImageSchedule,
        k: usize,
    ) -> Result<()> {
        let width = self.row_width;
        let offset = self.reference_rows * width;
        let count = (self.image_rows - self.reference_rows) * width;
        ensure!(
            k + 1 < schedule.sigmas.len()
                && offset + count <= image.len()
                && offset + count <= self.velocity.len(),
            "image Euler extent"
        );
        let sigma = schedule.sigmas[k];
        let sigma_next = schedule.sigmas[k + 1];
        ensure!(sigma > 0.0 && sigma_next >= 0.0, "image sigma {sigma}");
        let ratio = sigma_next / sigma;
        unsafe {
            s.launch_builder(&self.euler)
                .arg(&(pointer(image, s) + 4 * offset as u64))
                .arg(&(pointer(&self.velocity, s) + 4 * offset as u64))
                .arg(&(count as u64))
                .arg(&sigma)
                .arg(&ratio)
                .arg(&(1.0f32 - ratio))
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// One evaluation. Head `head` writes the velocity of every image row into `self.velocity`.
    pub(crate) fn step(
        &mut self,
        c: &BlockContext<'_>,
        adaln: &AdalnTable,
        (pairs, timesteps): (u32, u32),
        head: usize,
        inputs: &ImageStepInputs<'_>,
    ) -> Result<()> {
        if let Some(tier) = self.tier.as_mut() {
            tier.ready()?;
        }
        let (s, spec) = (c.s, c.spec);
        let h = spec.hidden;
        let (text_rows, image_rows, rows) = (
            self.text_rows,
            self.image_rows,
            self.text_rows + self.image_rows,
        );
        ensure!(
            head < self.heads.len()
                && inputs.image.len() == image_rows * self.row_width
                && inputs.text.len() == text_rows * spec.text_dim
                && inputs.row_pairs.len() == rows
                && inputs.row_timesteps.len() == rows,
            "image evaluation inputs differ from the layout"
        );
        c.blas.linear_biased(
            matrix(&self.context[0], s, h, spec.text_dim)?,
            vector(&self.context[1], s)?,
            matrix(inputs.text, s, text_rows, spec.text_dim)?,
            &mut TensorMut(matrix(&self.text, s, text_rows, h)?),
            LinearBiasMode::BeforeOutputRound,
        )?;
        for refiner in &self.refiners {
            block_forward::<false>(
                c,
                &mut self.text_flash,
                refiner,
                &Conditioning::Plain,
                &mut self.text,
                &mut self.text_work,
            )?;
        }
        c.kernels.rms_norm_f32(
            s,
            NormRows {
                input: pointer(&self.text, s),
                weight: pointer(&self.final_norm, s),
                output: pointer(&self.x, s),
                rows: text_rows,
                dim: h,
                threads: HIDDEN_NORM_THREADS,
                eps: spec.final_norm_eps,
            },
        )?;
        projection(
            c.s,
            c.blas,
            &self.bias,
            &self.proj_in,
            (inputs.image, 0),
            (&self.scratch, image_rows),
        )?;
        round_into(
            s,
            &self.round,
            &self.x,
            pointer(&self.scratch, s),
            text_rows * h,
            image_rows * h,
        )?;
        let block_rows = 6 * h * pairs as usize;
        let sources = Scale2Sources {
            hidden: h,
            block_rows,
            pairs,
            timesteps,
            norm_eps: spec.norm_eps,
            final_eps: spec.final_norm_eps,
            table: pointer(&adaln.blocks, s),
            block_table_len: adaln.blocks.len(),
            norm_out_weight: pointer(&self.norm_out, s),
            norm_out_table: pointer(&adaln.norm_out, s),
            norm_table_len: adaln.norm_out.len(),
            row_pairs: pointer(inputs.row_pairs, s),
            row_timesteps: pointer(inputs.row_timesteps, s),
        };
        let fuse = fuse_cross_block(c.kernels);
        let mut fused_norm_out = false;
        self.work.skip_norm1 = false;
        for layer in 0..self.blocks.len() {
            if let Some(tier) = self.tier.as_mut() {
                tier.ready()?.begin(layer, s)?;
            }
            let modulation = Modulation {
                table: &adaln.blocks,
                offset: layer * block_rows,
                entries: pairs,
                index: inputs.row_pairs,
            };
            let tail = if fuse {
                scale2_tail(&self.blocks, layer, &sources)?
            } else {
                None
            };
            fused_norm_out |= tail.as_ref().is_some_and(|tail| !tail.arms_norm1);
            block_forward_tail::<false>(
                c,
                &mut self.flash,
                &self.blocks[layer],
                &Conditioning::Modulated {
                    rotary: &self.rotary,
                    modulation: &modulation,
                },
                &mut self.x,
                &mut self.work,
                tail,
            )?;
            if let Some(tier) = self.tier.as_mut() {
                tier.ready()?.end(layer, s)?;
            }
        }
        if !fused_norm_out {
            let normed = pointer(&self.work.normed, s);
            ensure!(
                adaln.norm_out.len() == timesteps as usize * 2 * h,
                "output-norm table extent"
            );
            let table = pointer(&adaln.norm_out, s);
            c.kernels.rms_norm_modulate_f32(
                s,
                NormRows {
                    input: pointer(&self.x, s),
                    weight: pointer(&self.norm_out, s),
                    output: normed,
                    rows,
                    dim: h,
                    threads: HIDDEN_NORM_THREADS,
                    eps: spec.final_norm_eps,
                },
                ModulateRows {
                    data: normed,
                    shift: table,
                    scale: table + 2 * h as u64,
                    index: pointer(inputs.row_timesteps, s),
                    rows,
                    dim: h,
                    stride: 2 * h,
                    entries: timesteps,
                },
            )?;
        }
        upcast_from(
            s,
            &self.upcast,
            &self.work.normed,
            &self.scratch,
            text_rows * h,
            image_rows * h,
        )?;
        projection(
            c.s,
            c.blas,
            &self.bias,
            &self.heads[head],
            (&self.scratch, 0),
            (&self.velocity, image_rows),
        )?;
        Ok(())
    }

    /// Six published evaluations. `image` is updated in place on the generated rows; reference rows stay put.
    /// `captured.0`, when present, receives the generated rows after each Euler update.
    /// `captured.1`, when present, replaces `image` before evaluation k.
    pub(crate) fn sample(
        &mut self,
        c: &BlockContext<'_>,
        adaln: (&AdalnTable, &AdalnPlan),
        schedule: &ImageSchedule,
        image: &mut CudaSlice<f32>,
        text: &CudaSlice<bf16>,
        captured: ImageCaptured<'_>,
    ) -> Result<()> {
        let (mut trace, entering) = captured;
        let (table, plan) = adaln;
        let s = c.s;
        ensure!(
            plan.row_pairs.len() == schedule.timesteps.len()
                && schedule.sigmas.len() == schedule.timesteps.len() + 1,
            "image plan and schedule differ"
        );
        if let Some(steps) = trace.as_deref() {
            ensure!(steps.is_empty(), "image step trace is not empty");
        }
        if let Some(steps) = entering {
            ensure!(
                steps.len() == plan.row_pairs.len()
                    && steps.iter().all(|rows| {
                        rows.len() == image.len() && rows.iter().all(|value| value.is_finite())
                    }),
                "image entering rows"
            );
        }
        let counts = (
            u32::try_from(plan.pairs.len())?,
            u32::try_from(plan.timesteps.len())?,
        );
        for k in 0..plan.row_pairs.len() {
            if let Some(steps) = entering {
                s.memcpy_htod(&steps[k], image)?;
            }
            let row_pairs = s.clone_htod(&plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&plan.row_timesteps[k])?;
            self.step(
                c,
                table,
                counts,
                k,
                &ImageStepInputs {
                    image,
                    text,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            self.euler(s, image, schedule, k)?;
            if let Some(steps) = trace.as_deref_mut() {
                let width = self.row_width;
                let offset = self.reference_rows * width;
                let count = (self.image_rows - self.reference_rows) * width;
                let full = s.clone_dtoh(image)?;
                ensure!(offset + count <= full.len(), "image step trace extent");
                let rows = full[offset..offset + count].to_vec();
                ensure!(rows.iter().all(|v| v.is_finite()), "non-finite image step");
                steps.push(rows);
            }
        }
        s.synchronize()?;
        Ok(())
    }
}

/// Roles of a Qwen3-VL decoder layer's matrices in manifest name order; `mlp.gate_proj` and `mlp.up_proj` are
/// adjacent, so one `[2 * ffn, hidden]` view projects both.
const TEXT_ROLES: [&str; 7] = [
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
];

/// One Qwen3-VL decoder layer: matrices as views into `storage` or a dense streaming slot, norms owned.
pub(crate) struct TextLayer {
    input_norm: CudaSlice<bf16>,
    post_norm: CudaSlice<bf16>,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    q: Tensor<bf16, 2>,
    k: Tensor<bf16, 2>,
    v: Tensor<bf16, 2>,
    o: Tensor<bf16, 2>,
    gate_up: Tensor<bf16, 2>,
    down: Tensor<bf16, 2>,
    storage: Option<CudaSlice<bf16>>,
}

impl TextLayer {
    fn new(
        s: &Arc<CudaStream>,
        block: &str,
        (hidden, head_dim): (usize, usize),
        slot: Option<u64>,
        base: &mut Source<'_>,
    ) -> Result<Self> {
        let (layout, elements) = matrix_layout(base.manifest, block, &TEXT_ROLES)?;
        let mut storage = None;
        let at = match slot {
            Some(slot) => slot,
            None => {
                let mut owned = s.alloc_zeros::<bf16>(elements)?;
                for (role, (offset, shape)) in &layout {
                    let count = shape[0] as usize * shape[1] as usize;
                    base.fill(
                        s,
                        &format!("{block}.{role}"),
                        &mut owned.slice_mut(*offset..offset + count),
                    )?;
                }
                let at = pointer(&owned, s);
                storage = Some(owned);
                at
            }
        };
        let view = |offset: usize, shape: [u32; 2]| unsafe {
            Tensor::new(
                at + 2 * offset as u64,
                shape,
                [u64::from(shape[1]), 1],
                2 * u64::from(shape[0]) * u64::from(shape[1]),
            )
        };
        let role = |name: &str| -> Result<Tensor<bf16, 2>> {
            let (offset, shape) = layout[name];
            view(offset, shape)
        };
        let (gate, gate_shape) = layout["mlp.gate_proj.weight"];
        let (up, up_shape) = layout["mlp.up_proj.weight"];
        ensure!(
            gate_shape == up_shape && up == gate + gate_shape[0] as usize * gate_shape[1] as usize,
            "{block}: gate and up projections are not adjacent"
        );
        let norm = |base: &mut Source<'_>, name: &str, n: usize| {
            base.tensor::<bf16>(s, &format!("{block}.{name}.weight"), n)
        };
        let mut layer = Self {
            input_norm: norm(base, "input_layernorm", hidden)?,
            post_norm: norm(base, "post_attention_layernorm", hidden)?,
            q_norm: norm(base, "self_attn.q_norm", head_dim)?,
            k_norm: norm(base, "self_attn.k_norm", head_dim)?,
            q: role("self_attn.q_proj.weight")?,
            k: role("self_attn.k_proj.weight")?,
            v: role("self_attn.v_proj.weight")?,
            o: role("self_attn.o_proj.weight")?,
            gate_up: view(gate, [2 * gate_shape[0], gate_shape[1]])?,
            down: role("mlp.down_proj.weight")?,
            storage: None,
        };
        layer.storage = storage;
        Ok(layer)
    }
}

/// Conditioner input over `rows` prompt rows: the BF16 input embeddings `[rows, hidden]` with the image rows already
/// spliced, the deepstack features added to the image rows after decoder layers `0..deepstack.len()`
/// (`[image rows, hidden]` each, image runs in order), the image runs as `(first row, rows)`, and the BF16 mRoPE
/// table `[rows, head_dim]` (cosines, then sines).
pub(crate) struct ConditionerInput<'a> {
    pub embeds: &'a [bf16],
    pub deepstack: &'a [CudaSlice<bf16>],
    pub runs: &'a [(usize, usize)],
    pub rope: &'a [bf16],
}

/// The Qwen3-VL conditioner text decoder: layers `0..layers.len()` of the published checkpoint, placed through
/// dense streaming; the output is the residual after the last layer (`hidden_states[layers]`). Activations are
/// allocated before the placement is derived; `media` bytes stay unplaced for the vision request.
pub(crate) struct Conditioner {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    ffn: usize,
    eps: f32,
    capacity: usize,
    layers: Vec<TextLayer>,
    tier: Option<DenseTier>,
    kernels: BlockKernels,
    flash: Flash,
    x: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    keys: CudaSlice<bf16>,
    values: CudaSlice<bf16>,
    attention: CudaSlice<bf16>,
    gate_up: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    rope: CudaSlice<bf16>,
}

impl Conditioner {
    pub(super) fn layer_sizes(
        manifest: &TensorManifest,
        layers: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let mut bytes = vec![0usize; layers];
        let mut largest = 0usize;
        for entry in manifest.entries.values() {
            if let Some(layer) = entry.layer
                && entry.load
                && entry.shape.len() == 2
            {
                let layer = layer as usize;
                ensure!(layer < layers, "manifest loads decoder layer {layer}");
                bytes[layer] += entry.bytes;
                largest = largest.max(entry.bytes);
            }
        }
        Ok((bytes, largest))
    }

    pub(super) fn resident_fixed_bytes(
        device: &Device,
        spec: &crate::model_spec::ModelSpec,
        (layers, capacity): (usize, usize),
    ) -> Result<usize> {
        let (hidden, d, ffn) = (
            spec.hidden as usize,
            spec.head_dim as usize,
            spec.intermediate as usize,
        );
        let heads = spec.q_heads as usize;
        let kv = spec.kv_heads as usize;
        let widths = 2 * hidden + d + 2 * heads * d + 4 * kv * d + 3 * ffn;
        let norm = layers * 2 * (2 * hidden + 2 * d);
        let flash = Flash::workspace(&device.stream, capacity, heads, d, capacity)?;
        capacity
            .checked_mul(2 * widths)
            .and_then(|n| n.checked_add(norm))
            .and_then(|n| n.checked_add(flash))
            .context("conditioner fixed device extent overflow")
    }

    pub(crate) fn new(
        d: &Device,
        spec: &crate::model_spec::ModelSpec,
        (layers, capacity, media): (usize, usize, usize),
        base: Source<'_>,
        placement: Placement,
        requests: usize,
    ) -> Result<Self> {
        Self::with_plan(
            d,
            spec,
            (layers, capacity, media),
            base,
            (placement, None),
            requests,
        )
    }

    pub(super) fn with_plan(
        d: &Device,
        spec: &crate::model_spec::ModelSpec,
        (layers, capacity, media): (usize, usize, usize),
        mut base: Source<'_>,
        (placement, plan): (Placement, Option<crate::prefill::DensePlacement>),
        requests: usize,
    ) -> Result<Self> {
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Qwen3Vl
                && layers <= spec.layers.len(),
            "conditioner needs Qwen3-VL with at least {layers} decoder layers"
        );
        let s = &d.stream;
        let (hidden, head_dim, ffn) = (
            spec.hidden as usize,
            spec.head_dim as usize,
            spec.intermediate as usize,
        );
        let (q_heads, kv_heads) = (spec.q_heads as usize, spec.kv_heads as usize);
        let name = |layer: usize| format!("model.language_model.layers.{layer}");
        let (bytes, largest) = Self::layer_sizes(base.manifest, layers)?;
        let rows = |width: usize| s.alloc_zeros::<bf16>(capacity * width);
        let (x, normed, rope) = (rows(hidden)?, rows(hidden)?, rows(head_dim)?);
        let (q, attention) = (rows(q_heads * head_dim)?, rows(q_heads * head_dim)?);
        let (k, v) = (rows(kv_heads * head_dim)?, rows(kv_heads * head_dim)?);
        let (keys, values) = (rows(kv_heads * head_dim)?, rows(kv_heads * head_dim)?);
        let (gate_up, gated) = (rows(2 * ffn)?, rows(ffn)?);
        let flash = Flash::new(s.clone(), capacity, q_heads, kv_heads, head_dim)?;
        let owned = layers * 2 * (2 * hidden + 2 * head_dim);
        let tier = if let Some(plan) = plan {
            bind_layers("conditioner", d, &mut base, &bytes, plan, name, &TEXT_ROLES)?
        } else {
            let visits = crate::prefill::placement_visits(1, requests)?;
            place_layers(
                "conditioner",
                d,
                &mut base,
                (&bytes, &bytes, largest, owned, media),
                (placement, visits),
                name,
                &TEXT_ROLES,
            )?
            .tier
        };
        let mut text_layers = Vec::with_capacity(layers);
        for (layer, &layer_bytes) in bytes.iter().enumerate() {
            let slot = match tier.as_ref().map(|tier| &tier.dense) {
                Some(streaming) if streaming.plan.slots[layer].is_some() => {
                    Some(streaming.pointer(layer, 0, layer_bytes)?)
                }
                _ => None,
            };
            text_layers.push(TextLayer::new(
                s,
                &name(layer),
                (hidden, head_dim),
                slot,
                &mut base,
            )?);
        }
        Ok(Self {
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            ffn,
            eps: spec.rms_eps,
            capacity,
            layers: text_layers,
            tier,
            kernels: BlockKernels::new(s.context())?,
            flash,
            x,
            normed,
            q,
            k,
            v,
            keys,
            values,
            attention,
            gate_up,
            gated,
            rope,
        })
    }

    /// Binds the fixed GEMM plans of every decoder-layer shape at `rows`.
    pub(crate) fn prepare(&self, blas: &Blas, rows: usize) -> Result<()> {
        let (h, q, kv, f) = (
            self.hidden,
            self.q_heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.ffn,
        );
        for (output, input) in [(q, h), (kv, h), (h, q), (2 * f, h), (h, f)] {
            blas.prepare_fixed(
                u32::try_from(rows)?,
                u32::try_from(output)?,
                u32::try_from(input)?,
            )?;
        }
        Ok(())
    }

    /// Runs every decoder layer on `input` and returns the residual after layer `tap - 1` for each `tap`, in order.
    pub(crate) fn forward(
        &mut self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        input: &ConditionerInput<'_>,
        taps: &[usize],
    ) -> Result<Vec<Vec<bf16>>> {
        if let Some(tier) = self.tier.as_mut() {
            tier.ready()?;
        }
        let (h, d) = (self.hidden, self.head_dim);
        let (qw, kvw) = (self.q_heads * d, self.kv_heads * d);
        ensure!(
            !input.embeds.is_empty() && input.embeds.len().is_multiple_of(h),
            "embedding extent"
        );
        let rows = input.embeds.len() / h;
        ensure!(
            rows <= self.capacity,
            "{rows} rows exceed capacity {}",
            self.capacity
        );
        ensure!(input.rope.len() == rows * d, "mRoPE table extent");
        let image_rows: usize = input.runs.iter().map(|&(_, n)| n).sum();
        ensure!(
            input.runs.iter().all(|&(first, n)| first + n <= rows)
                && input.deepstack.len() <= self.layers.len()
                && input.deepstack.iter().all(|f| f.len() == image_rows * h),
            "image runs or deepstack features differ from the prompt"
        );
        ensure!(
            taps.iter()
                .all(|&tap| (1..=self.layers.len()).contains(&tap)),
            "taps outside 1..={}",
            self.layers.len()
        );
        s.memcpy_htod(input.embeds, &mut self.x.slice_mut(..rows * h))?;
        s.memcpy_htod(input.rope, &mut self.rope.slice_mut(..rows * d))?;
        let rows_i = i32::try_from(rows)?;
        let linear = |weight: Tensor<bf16, 2>,
                      x: &CudaSlice<bf16>,
                      k: usize,
                      y: &CudaSlice<bf16>,
                      n: usize|
         -> Result<()> {
            blas.linear_typed(
                crate::op_contract::ExecutionPhase::Prefill,
                crate::backend::cuda::linear::LinearWeightRef::Bf16(weight),
                rows_view(x, s, 0, rows, k, k)?,
                &mut TensorMut(rows_view(y, s, 0, rows, n, n)?),
            )
        };
        let kernels = &self.kernels;
        let norm = |input: &CudaSlice<bf16>,
                    weight: &CudaSlice<bf16>,
                    output: &CudaSlice<bf16>,
                    (rows, dim, threads): (usize, usize, u32)|
         -> Result<()> {
            kernels.rms_norm(
                s,
                NormRows {
                    input: pointer(input, s),
                    weight: pointer(weight, s),
                    output: pointer(output, s),
                    rows,
                    dim,
                    threads,
                    eps: self.eps,
                },
            )
        };
        let residual = |residual: u64, update: u64, count: usize| -> Result<()> {
            kernels.prepare_fp16_store(s, &kernels.norm_module)?;
            unsafe {
                s.launch_builder(&kernels.residual)
                    .arg(&residual)
                    .arg(&update)
                    .arg(&i32::try_from(count)?)
                    .launch(flat(count / 8))?;
            }
            kernels.finish_fp16_store(s, &kernels.norm_module, "residual")
        };
        let mut tapped = Vec::with_capacity(taps.len());
        let x = pointer(&self.x, s);
        let normed = pointer(&self.normed, s);
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(tier) = self.tier.as_mut() {
                tier.ready()?.begin(index, s)?;
            }
            norm(
                &self.x,
                &layer.input_norm,
                &self.normed,
                (rows, h, HIDDEN_NORM_THREADS),
            )?;
            linear(layer.q, &self.normed, h, &self.attention, qw)?;
            linear(layer.k, &self.normed, h, &self.keys, kvw)?;
            linear(layer.v, &self.normed, h, &self.v, kvw)?;
            norm(
                &self.attention,
                &layer.q_norm,
                &self.q,
                (rows * self.q_heads, d, HEAD_NORM_THREADS),
            )?;
            norm(
                &self.keys,
                &layer.k_norm,
                &self.k,
                (rows * self.kv_heads, d, HEAD_NORM_THREADS),
            )?;
            // q rotates in place; k and v land head-major [kv_heads, rows, head_dim] in keys and values.
            kernels.prepare_fp16_store(s, &kernels.rope_module)?;
            unsafe {
                s.launch_builder(&kernels.rotary)
                    .arg(&pointer(&self.q, s))
                    .arg(&pointer(&self.k, s))
                    .arg(&pointer(&self.v, s))
                    .arg(&pointer(&self.keys, s))
                    .arg(&pointer(&self.values, s))
                    .arg(&pointer(&self.rope, s))
                    .arg(&0i32)
                    .arg(&rows_i)
                    .arg(&i32::try_from(self.q_heads)?)
                    .arg(&i32::try_from(self.kv_heads)?)
                    .arg(&i32::try_from(d)?)
                    .arg(&i32::try_from(d)?)
                    .arg(&rows_i)
                    .launch(flat(rows * (self.q_heads + self.kv_heads) * d))?;
            }
            kernels.finish_fp16_store(s, &kernels.rope_module, "qk")?;
            self.flash.run(
                &self.q,
                qw,
                &self.keys,
                &self.values,
                &mut self.attention,
                Shape {
                    rows,
                    start: 0,
                    q_heads: self.q_heads,
                    kv_heads: self.kv_heads,
                    capacity: rows,
                    head_dim: d,
                    causal: true,
                    window_left: None,
                },
            )?;
            linear(layer.o, &self.attention, qw, &self.normed, h)?;
            residual(x, normed, rows * h)?;
            norm(
                &self.x,
                &layer.post_norm,
                &self.normed,
                (rows, h, HIDDEN_NORM_THREADS),
            )?;
            linear(layer.gate_up, &self.normed, h, &self.gate_up, 2 * self.ffn)?;
            let (gate_up, gated) = (pointer(&self.gate_up, s), pointer(&self.gated, s));
            let cfg = ops::silu_mul_grid(rows, self.ffn, gate_up, gated)?;
            kernels.prepare_fp16_store(s, &kernels.act_module)?;
            unsafe {
                s.launch_builder(&kernels.swiglu)
                    .arg(&gate_up)
                    .arg(&gated)
                    .arg(&rows_i)
                    .arg(&i32::try_from(self.ffn)?)
                    .arg(&0i32)
                    .launch(cfg)?;
            }
            kernels.finish_fp16_store(s, &kernels.act_module, "ffn")?;
            linear(layer.down, &self.gated, self.ffn, &self.normed, h)?;
            #[cfg(test)]
            if index == 43 && rows > 20 && h > 731 {
                let coordinate = s.clone_dtoh(&self.normed.slice(20 * h + 731..20 * h + 732))?[0];
                println!(
                    "H3_CONDITIONER_MLP layer=43 row=20 column=731 value={coordinate} finite={} utc={}",
                    coordinate.is_finite(),
                    pipeline::composition_stamp()?
                );
            }
            residual(x, normed, rows * h)?;
            if let Some(features) = input.deepstack.get(index) {
                let mut feature = 0usize;
                for &(first, n) in input.runs {
                    residual(
                        x + (2 * first * h) as u64,
                        pointer(features, s) + (2 * feature * h) as u64,
                        n * h,
                    )?;
                    feature += n;
                }
            }
            if let Some(tier) = self.tier.as_mut() {
                tier.ready()?.end(index, s)?;
            }
            if taps.contains(&(index + 1)) {
                tapped.push((index + 1, s.clone_dtoh(&self.x.slice(..rows * h))?));
            }
        }
        Ok(taps
            .iter()
            .map(|tap| {
                tapped
                    .iter()
                    .find(|(t, _)| t == tap)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default()
            })
            .collect())
    }
}

/// `[ids.len(), hidden]` BF16 rows of the published embedding table `embed`, read row by row.
pub(crate) fn embedding_rows(
    base: &mut Source<'_>,
    embed: &str,
    hidden: usize,
    ids: &[u32],
    fp16: bool,
) -> Result<Vec<bf16>> {
    let entry = base
        .manifest
        .entries
        .get(embed)
        .with_context(|| format!("manifest lacks {embed}"))?;
    ensure!(
        entry.dtype == Dtype::BF16 && entry.shape.len() == 2 && entry.shape[1] == hidden,
        "{embed}: {:?} {:?}",
        entry.dtype,
        entry.shape
    );
    let row = 2 * hidden;
    let mut out = Vec::with_capacity(ids.len() * hidden);
    let mut flushed = 0u64;
    for &id in ids {
        ensure!((id as usize) < entry.shape[0], "token {id} outside {embed}");
        let bytes = base
            .reader
            .read_range(embed, entry, id as usize * row, row)?;
        ensure!(
            <bf16 as Element>::finite(bytes),
            "{embed}: nonfinite row {id}"
        );
        let mut row_bytes = bytes.to_vec();
        if fp16 {
            flushed += crate::tensors::bf16_weight_to_fp16(embed, &mut row_bytes, false)?;
        }
        out.extend(
            row_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b)),
        );
    }
    if flushed > 0 {
        eprintln!("{embed}: BF16 to FP16 flushed {flushed} values to +0");
    }
    Ok(out)
}

/// Fills `out` with `count` standard normals from one documented `(seed, stream)` Philox counter range.
/// `block` selects the launch geometry; the counter-based output is identical for every geometry.
pub(crate) fn draw_normals(
    s: &Arc<CudaStream>,
    normal: &CudaFunction,
    (seed, stream): (u64, u64),
    count: usize,
    block: u32,
) -> Result<CudaSlice<f32>> {
    // The kernel writes every one of the `count` outputs.
    let out = unsafe { s.alloc::<f32>(count) }?;
    let threads = count.div_ceil(4);
    unsafe {
        s.launch_builder(normal)
            .arg(&seed)
            .arg(&stream)
            .arg(&out)
            .arg(&(count as i64))
            .launch(ops::grid(threads.div_ceil(block as usize), block))?;
    }
    Ok(out)
}

/// Runs one MiniMax-H3 i2va request on GPU `device` with the transformer LoRA at `lora` (cuBLAS calibration under
/// `runtime`) and returns its u8 frames and stereo PCM. The four noise draws are generated natively on device from
/// `request.seed`; there is no recorded-draw path. `gguf` replaces the transformer safetensors with one
/// community GGUF file.
pub fn generate(
    model: &Path,
    lora: &Path,
    request: &crate::minimax_h3::H3Request,
    (device, runtime): (usize, &Path),
    gguf: Option<&Path>,
) -> Result<crate::minimax_h3::H3Output> {
    Ok(pipeline::run(model, lora, request, (device, runtime), gguf)?.output)
}

pub(crate) fn generate_hyperflow(
    model: &Path,
    transformer: &Path,
    request: &crate::minimax_h3::H3Request,
    device: (usize, &Path),
) -> Result<crate::minimax_h3::H3Output> {
    Ok(pipeline::run_hyperflow((model, transformer), request, device)?.output)
}

/// Six published image evaluations. `rows` is `[image rows, 96]`, references first. The returned buffer is those
/// image rows after the six Euler updates. Reference rows stay at their input values. Placement is derived.
pub fn image_sample(
    model: &Path,
    layout: &ImageLayout,
    (device, runtime): (usize, &Path),
    rows: &[f32],
    text: &[bf16],
) -> Result<Vec<f32>> {
    image_sample_trace(model, layout, (device, runtime), rows, text, None, None)
}

/// Same evaluations as [`image_sample`]. `trace` receives the generated rows after each Euler update.
/// `entering`, when present, replaces the image rows before evaluation k.
pub(crate) fn image_sample_trace(
    model: &Path,
    layout: &ImageLayout,
    (device, runtime): (usize, &Path),
    rows: &[f32],
    text: &[bf16],
    trace: Option<&mut Vec<Vec<f32>>>,
    entering: Option<&[Vec<f32>]>,
) -> Result<Vec<f32>> {
    let image = ImageSpec::read(model)?;
    let spec = image_block_spec(&image);
    let schedule = ImageSchedule::published()?;
    let width = image.row_width();
    ensure!(
        rows.len() == layout.image_indices.len() * width
            && text.len() == layout.text_indices.len() * spec.text_dim
            && !rows.is_empty()
            && rows.iter().all(|v| v.is_finite())
            && text.iter().all(|v| v.to_f32().is_finite()),
        "image sample extents"
    );
    let dir = model.join("transformer_diffusers");
    let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
        .minimax_h3_image_manifest(&image)?;
    let plan = image_row_plan(layout, &schedule)?;
    let adaln = AdalnPlan::from_tags(&layout.token_tags, &plan)?;
    let device = Device::new(device)?;
    let ops = ops::Ops::new(&device.ctx)?;
    device.finish_upload()?;
    let mut blas = open_blas(
        &device,
        &ops,
        runtime,
        Rc::new(Closure::new(true, "MiniMax-H3-Image".into())),
    )?;
    let table = AdalnTable::build(&device, &mut blas, &dir, &manifest, &spec, &adaln)?;
    let mut dit = ImageDit::new(
        &device,
        &mut blas,
        (&spec, &image),
        layout,
        &dir,
        &manifest,
        (
            Placement::Derived,
            crate::prefill::placement_visits(plan.len(), 1)?,
        ),
    )?;
    let kernels = BlockKernels::new(&device.ctx)?;
    let s = &device.stream;
    let context = BlockContext {
        s,
        blas: &blas,
        kernels: &kernels,
        spec: &spec,
        lora_scale: 1.0,
    };
    dit.sample_host(
        &context,
        (&table, &adaln),
        &schedule,
        (rows, text),
        (trace, entering),
    )
}

/// One Qwen3-VL conditioner prompt: prompt tokens with their image grids and packed `pixel_values`.
pub(crate) struct ConditionerPrompt {
    pub tokens: Vec<u32>,
    pub modality: Vec<u8>,
    pub grids: Vec<crate::vision_input::ImageGrid>,
    pub pixels: Vec<f32>,
}

/// Image-token runs of a prompt as `(first row, rows)`.
pub(crate) fn image_runs(tokens: &[u32], image_token: u32) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (row, &id) in tokens.iter().enumerate() {
        if id != image_token {
            continue;
        }
        match runs.last_mut() {
            Some((first, n)) if *first + *n == row => *n += 1,
            _ => runs.push((row, 1)),
        }
    }
    runs
}

/// BF16 mRoPE table `[rows, head_dim]` of a conditioner cell from its validated position plan.
pub(crate) fn conditioner_rope(
    spec: &crate::model_spec::ModelSpec,
    cell: &ConditionerPrompt,
) -> Result<Vec<bf16>> {
    let vision = spec.vision.as_ref().context("vision missing")?;
    let section = vision.mrope_section.context("mRoPE section missing")?;
    let plan = crate::vision_input::MultimodalInput {
        input_ids: &cell.tokens,
        modality_ids: &cell.modality,
        pixel_values: &cell.pixels,
        grids: &cell.grids,
        audio: &[],
    }
    .validate(spec, cell.tokens.len())?;
    Ok(
        crate::minimax_h3::mrope_table(&plan, section, spec.rope_theta, spec.head_dim)?
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
    )
}

/// Conditioner input of one cell: table rows for text tokens, `merged` for the image rows in order.
pub(crate) fn conditioner_embeds(
    cell: &ConditionerPrompt,
    image_token: u32,
    table: &BTreeMap<u32, Vec<bf16>>,
    merged: &[bf16],
    hidden: usize,
) -> Result<Vec<bf16>> {
    let mut embeds = Vec::with_capacity(cell.tokens.len() * hidden);
    let mut feature = 0usize;
    for &id in &cell.tokens {
        if id == image_token {
            embeds.extend_from_slice(
                merged
                    .get(feature * hidden..(feature + 1) * hidden)
                    .context("fewer image features than image tokens")?,
            );
            feature += 1;
        } else {
            embeds.extend_from_slice(table.get(&id).context("embedding row missing")?);
        }
    }
    ensure!(
        feature * hidden == merged.len(),
        "more image features than image tokens"
    );
    Ok(embeds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::{closure::Closure, ops::Ops};
    use crate::checkpoint::MappedCheckpoint;
    use crate::minimax_h3::{
        AUDIO_TAG, Geometry, IMAGE_MIN_GENERATION_AREA, IMAGE_TAG, ImageLayout, ImageSchedule,
        ImageSpec, KEYFRAME_NOISE_AUG, KeyframeAnchor, PackedLayout, RowTimesteps, Schedule,
        TEXT_TAG, VIDEO_TAG, hyperflow_steps, image_canvas, image_latent_extent, image_layout,
        image_reference_latent_extent, image_vae_reference_size, noise_draws, patchify_latents,
        rotary_rows, row_timestep_plan, unpatchify_latents,
    };
    use anyhow::bail;
    use safetensors::SafeTensors;
    use std::{collections::BTreeMap, fs, rc::Rc};

    /// The H3-Image adaln span derivation: a plan addressing only IMAGE and TEXT reads two modality
    /// blocks of the published [96768, 2688] BF16 weight, never the audio rows; the counted extents
    /// match the safetensors headers of `transformer_diffusers`.
    #[test]
    fn adaln_spans_read_only_the_addressed_modality_blocks() {
        let pairs = [
            (0, TEXT_TAG),
            (0, IMAGE_TAG),
            (1, IMAGE_TAG),
            (1, TEXT_TAG),
            (5, TEXT_TAG),
        ];
        let tags = adaln_tags(&pairs);
        assert_eq!(tags, vec![IMAGE_TAG, TEXT_TAG]);
        let (rows, cols) = (96_768usize, 2_688usize);
        let width = 6 * 5_376usize;
        assert_eq!(width, rows / MODALITIES as usize);
        let weight_block = width * cols * 2;
        let bias_block = 2 * width;
        let weight_spans: Vec<(usize, usize)> = tags
            .iter()
            .map(|&tag| (usize::from(tag) * weight_block, weight_block))
            .collect();
        let bias_spans: Vec<(usize, usize)> = tags
            .iter()
            .map(|&tag| (usize::from(tag) * bias_block, bias_block))
            .collect();
        assert_eq!(
            weight_spans,
            vec![(0, 173_408_256), (173_408_256, 173_408_256)]
        );
        assert_eq!(bias_spans, vec![(0, 64_512), (64_512, 64_512)]);
        assert_eq!(
            rows * cols * 2 - weight_spans.iter().map(|&(_, n)| n).sum::<usize>(),
            173_408_256
        );
        assert_eq!(
            rows * 2 - bias_spans.iter().map(|&(_, n)| n).sum::<usize>(),
            64_512
        );
        assert_eq!(50usize * (173_408_256 + 64_512), 8_673_638_400usize);
    }

    /// The compact pair offsets pick the same tag blocks as the full 3-slot table, and a perturbed
    /// audio block changes no gathered pair.
    #[test]
    fn adaln_compact_pairs_match_the_full_table() {
        let (width, timesteps) = (4usize, 2usize);
        let full: Vec<u16> = (0..timesteps * MODALITIES as usize * width)
            .map(|i| i as u16)
            .collect();
        let tags = adaln_tags(&[(0, TEXT_TAG), (0, IMAGE_TAG), (1, IMAGE_TAG)]);
        assert_eq!(tags, vec![IMAGE_TAG, TEXT_TAG]);
        let gather = |table: &[u16]| -> Vec<u16> {
            (0..timesteps)
                .flat_map(|t| {
                    tags.iter().flat_map(move |&tag| {
                        table[t * MODALITIES as usize * width + usize::from(tag) * width..][..width]
                            .to_vec()
                    })
                })
                .collect()
        };
        let compact = gather(&full);
        for &(t, tag) in &[(0, IMAGE_TAG), (0, TEXT_TAG), (1, IMAGE_TAG)] {
            let at = adaln_pair_offset(&tags, width, t, tag).unwrap();
            let from = (t as usize * MODALITIES as usize + usize::from(tag)) * width;
            assert_eq!(&compact[at..at + width], &full[from..from + width]);
        }
        assert!(adaln_block(&tags, AUDIO_TAG).is_err());
        let mut perturbed = full.clone();
        for t in 0..timesteps {
            let at = (t * MODALITIES as usize + usize::from(AUDIO_TAG)) * width;
            for value in &mut perturbed[at..at + width] {
                *value = 999;
            }
        }
        assert_eq!(gather(&perturbed), compact);
    }

    #[test]
    fn stored_placement_preserves_the_physical_reserve() -> Result<()> {
        use crate::checkpoint::TensorScope;
        use crate::prefill::DensePlacement;
        let mut spec = H3Spec::parse(&serde_json::from_str(include_str!(
            "../../../tests/data/minimax-h3-transformer-config.json"
        ))?)?;
        spec.layers = 3;
        spec.hidden = 3;
        spec.head_dim = 2;
        spec.dropped_blocks = Some(vec![1]);
        let mut entries = BTreeMap::new();
        for (name, layer, bytes, shape) in [
            ("transformer_blocks.0.a.weight", 0, 33, vec![3, 11]),
            ("transformer_blocks.0.b.weight", 0, 17, vec![1, 17]),
            ("transformer_blocks.2.a.weight", 2, 33, vec![3, 11]),
            ("transformer_blocks.2.b.weight", 2, 17, vec![1, 17]),
            ("transformer_blocks.2.a.wscale", 2, 4, vec![1]),
        ] {
            entries.insert(
                name.to_owned(),
                TensorEntry {
                    dtype: if name.ends_with(".wscale") {
                        Dtype::F32
                    } else {
                        Dtype::I8
                    },
                    shape,
                    shard: String::new(),
                    bytes,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, bytes as u64],
                    scope: TensorScope::Text,
                    layer: Some(layer),
                    role: name.to_owned(),
                    scale_name: None,
                    quant: None,
                    swap_halves: false,
                },
            );
        }
        let manifest = TensorManifest {
            entries,
            resident_bytes: 104,
        };
        let (layers, owned) = stored_allocation_costs(&manifest, &spec, 16, 64)?;
        assert_eq!(layers, [80, 0, 80]);
        assert_eq!(owned, 3_584);
        let arena = VectorArenaLayout::new(&manifest, &spec)?;
        assert_eq!(arena.offsets.len(), 14);
        assert_eq!(arena.bytes, 3_584);
        let mut spans: Vec<_> = arena.offsets.values().copied().collect();
        spans.sort_unstable_by_key(|&(offset, _, _)| offset);
        for (index, &(offset, bytes, _)) in spans.iter().enumerate() {
            assert_eq!(offset % 256, 0);
            assert!(bytes > 0 && offset + bytes <= arena.bytes);
            if let Some(&(next, _, _)) = spans.get(index + 1) {
                assert!(offset + bytes <= next);
            }
        }
        let &(offset, bytes, _) = spans.last().context("arena is empty")?;
        assert_eq!(offset + stored_allocation_bytes(bytes, 256)?, arena.bytes);
        let total = 65_536;
        let reserve = total / 16;
        let exact = reserve + owned + 160;
        let budget = DensePlacement::device_budget(exact, total, owned, 0, 0)?;
        let plan = DensePlacement::new(&[layers[0], layers[2]], Some(&[50, 50]), budget, 0)?;
        assert_eq!(plan.resident_bytes, 160);
        assert_eq!(
            exact.checked_sub(reserve + owned + plan.resident_bytes),
            Some(0)
        );
        let short = DensePlacement::device_budget(exact - 1, total, owned, 0, 0)?;
        assert!(DensePlacement::new(&[layers[0], layers[2]], Some(&[50, 50]), short, 0).is_err());
        assert!(DensePlacement::new(&[50, 50], None, exact - 1 - reserve - 64, 0).is_ok());
        assert_eq!(
            DensePlacement::device_budget(reserve + owned, total, owned, 0, 0)?,
            0
        );
        assert!(DensePlacement::device_budget(reserve + owned - 1, total, owned, 0, 0).is_err());
        let streamed = DensePlacement::with_allocations(
            &[96, 80, 80, 80],
            &[64, 64, 64, 64],
            Some(&[50, 50, 50, 50]),
            256,
            150,
            16,
            2,
        )?;
        let streamed_count = streamed.slots.iter().filter(|slot| slot.is_some()).count();
        assert_eq!(
            streamed.resident_bytes + 2 * streamed.slot_bytes + streamed_count * 16,
            256
        );
        assert_eq!(streamed_count, 3);
        assert_eq!(streamed.host_bytes, 150);
        assert_eq!(streamed.disk_bytes, 0);
        assert!(
            DensePlacement::with_allocations(
                &[96, 80, 80, 80],
                &[64, 64, 64, 64],
                Some(&[50, 50, 50, 50]),
                191,
                150,
                16,
                2,
            )
            .is_err()
        );
        assert!(stored_allocation_bytes(usize::MAX, 2).is_err());
        assert!(stored_allocation_bytes(1, 0).is_err());
        println!(
            "PASS STORED_PHYSICAL_RESERVE arena_tensors=14 aligned_disjoint=true exact_headroom=0 one_byte_short=ERROR logical_only_control=FAIL_AS_REQUIRED"
        );
        Ok(())
    }

    /// Initial chunks avoid skinny tails; selection counts scratch and workspace replacements.
    #[test]
    fn ffn_chunks_leave_no_skinny_tail() -> Result<()> {
        for rows in [60_431, 40_045, 1_025, 2_049, 3_073, 1_024, 300] {
            let chunk = ffn_chunk(rows);
            let tail = rows % chunk;
            assert!(chunk <= FFN_CHUNK_ROWS, "rows={rows} chunk={chunk}");
            assert!(
                tail == 0 || tail > crate::linear_plan::MAX_SKINNY_ROWS,
                "rows={rows} chunk={chunk} tail={tail}"
            );
        }
        let baseline = FfnPlan {
            chunk: 1_002,
            borrowed: false,
            allocation: 0,
            workspace: 0,
            tensor_cost: 80.0,
            launches: 280,
        };
        let efficient = FfnPlan {
            chunk: 6_675,
            borrowed: false,
            allocation: FfnPlan::allocation(6_675, 14_336)?,
            workspace: 0,
            tensor_cost: 54.0,
            launches: 42,
        };
        let larger = FfnPlan {
            chunk: 8_009,
            borrowed: false,
            allocation: FfnPlan::allocation(8_009, 14_336)?,
            workspace: 0,
            tensor_cost: 55.0,
            launches: 35,
        };
        let choices = [baseline, efficient, larger];
        let full = efficient.allocation;
        assert_eq!(FfnPlan::select(&choices, full)?.chunk, 6_675);
        assert_eq!(FfnPlan::select(&choices, full - 1)?.chunk, 1_002);
        assert_eq!(FfnPlan::select(&choices, 0)?.chunk, 1_002);
        let with_workspace = FfnPlan {
            workspace: 4096,
            ..efficient
        };
        assert!(FfnPlan::select(&[with_workspace], full + 4095).is_err());
        assert_eq!(
            FfnPlan::select(&[with_workspace], full + 4096)?.chunk,
            6_675
        );
        assert!(FfnPlan::allocation(usize::MAX, 2).is_err());
        for (rows, expected) in [(40_045, 10_011), (60_431, 15_107)] {
            let elements = rows * 7_168;
            let capacity = FfnPlan::alias_capacity((elements, elements), 14_336)?;
            assert_eq!(capacity, expected);
            assert!(capacity * 28_672 <= elements);
            assert!((capacity + 1) * 28_672 > elements);
            assert_eq!(FfnPlan::alias_capacity((elements, 14_335), 14_336)?, 0);
        }
        assert!(FfnPlan::alias_capacity((usize::MAX, usize::MAX), usize::MAX).is_err());
        assert!(FfnPlan::alias_capacity((1, 1), 0).is_err());
        assert!(
            FfnPlan::select(
                &[FfnPlan {
                    allocation: usize::MAX,
                    workspace: 1,
                    ..baseline
                }],
                usize::MAX
            )
            .is_err()
        );
        Ok(())
    }

    /// At the logged free of 1_683_750_912 the selector takes chunk 2052. Subtracting `total / 16` again leaves
    /// 105_455_616 and takes chunk 1026.
    #[test]
    fn image_ffn_reserve_is_subtracted_once() -> Result<()> {
        let rows = 4_103usize;
        let initial = 821usize;
        let element = 86_016usize;
        let plan = |chunk: usize, tensor_cost: f64| FfnPlan {
            chunk,
            borrowed: false,
            allocation: if chunk == initial { 0 } else { chunk * element },
            workspace: 0,
            tensor_cost,
            launches: rows.div_ceil(chunk),
        };
        let choices = [
            plan(4_103, 256.0 / 231.0),
            plan(2_052, 128.0 / 119.0),
            plan(1_368, 256.0 / 231.0),
            plan(1_026, 256.0 / 189.0),
            plan(821, 256.0 / 147.0),
        ];
        let logged_free = 1_683_750_912usize;
        let total = 25_252_724_736usize;
        assert_eq!(FfnPlan::select(&choices, logged_free)?.chunk, 2_052);
        let doubled = logged_free - total / 16;
        assert_eq!(doubled, 105_455_616);
        assert_eq!(FfnPlan::select(&choices, doubled)?.chunk, 1_026);
        assert!(choices[1].allocation <= logged_free);
        assert!(choices[3].allocation <= doubled && choices[2].allocation > doubled);
        Ok(())
    }

    /// The small (40,045) and production (60,431) cell rows fit every launch's index type; the 1024x1792 canvas at
    /// 345 frames does not, and the rotary limit is named.
    #[test]
    fn launch_extents_reject_rows_beyond_the_rotary_index() -> Result<()> {
        let spec = H3Spec::parse(&serde_json::from_str(include_str!(
            "../../../tests/data/minimax-h3-transformer-config.json"
        ))?)?;
        for rows in [40_045, 60_431, 149_796] {
            launch_extents(&spec, rows)?;
        }
        let tags = vec![crate::minimax_h3::TEXT_TAG; 1327];
        let geometry = Geometry::resolve(&spec, 1024, 1792, 345)?;
        let layout = PackedLayout::build(&spec, &geometry, &tags, &[KeyframeAnchor::First])?;
        let error = launch_extents(&spec, layout.rows())
            .err()
            .context("1024x1792 at 345 frames passed the launch extents")?;
        ensure!(
            layout.rows() > 149_796
                && error.to_string().contains("rotary_prefix_kv")
                && error.to_string().contains("at most 149796 rows"),
            "rows={} error={error}",
            layout.rows()
        );
        ensure!(
            launch_extents(&spec, 149_797).is_err(),
            "149,797 rows passed"
        );
        Ok(())
    }

    /// Token-refiner block 0 (published weights, Turbo LoRA at its scale) on 1, 12, 16 and 17 text rows: the LoRA A
    /// projections of the skinny class (rows <= 16, input a multiple of 256) run on the bound fixed plans; every
    /// output is finite.
    #[test]
    #[ignore = "GPU; FF_MODEL and FF_H3_LORA"]
    fn refiner_block_runs_on_skinny_text_rows_gpu() -> Result<()> {
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let lora_path = path("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-refiner-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 refiner skinny-row test".into(),
            )),
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::resident::<true>(
            s,
            &spec,
            adapter.rank,
            "token_refiner.refiner_blocks.0",
            &mut Source::new(&transformer, &manifest),
            Some(&mut Source::new(&lora_dir, &lora_manifest)),
        )?;
        for rows in [1usize, 12, 16, 17] {
            let mut work = BlockWork::new(s, &spec, adapter.rank, rows)?;
            work.prepare::<true>(&blas, &spec)?;
            let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
                .with_full_single_probability()?;
            let input: Vec<bf16> = (0..rows * spec.hidden)
                .map(|i| bf16::from_f32(((i * 37 % 509) as f32 - 254.0) / 127.0))
                .collect();
            let mut x = s.clone_htod(&input)?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: adapter.scale,
            };
            block_forward::<true>(
                &context,
                &mut flash,
                &weights,
                &Conditioning::Plain,
                &mut x,
                &mut work,
            )?;
            let output = s.clone_dtoh(&x)?;
            ensure!(
                output.iter().all(|v| v.to_f32().is_finite()),
                "rows={rows}: non-finite refiner output"
            );
            println!("H3_REFINER_SKINNY rows={rows} finite=true");
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted"]
    fn decoded_rgb_is_byte_invariant_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        device.finish_upload()?;
        let kernel =
            cubin::module(&device.ctx, "vision_ops")?.load_function("planar_f32_to_rgb_u8")?;
        let (mean, std) = (
            crate::minimax_h3::VAE_PIXEL_MEAN.map(|v| v as f32),
            crate::minimax_h3::VAE_PIXEL_STD.map(|v| v as f32),
        );
        let convert = |input, output, shape, affine| {
            super::vae::planar_f32_to_rgb_u8(&device, &kernel, input, output, shape, affine)
        };
        let mut cells = 0;
        let mut defects = 0;
        let mut control_differences = 0;
        for (frames, allocated, pixels) in [
            (1usize, 1usize, 1usize),
            (2, 3, 17),
            (3, 5, 257),
            (1, 1, 65536),
            (124, 124, 768 * 1344),
            (192, 192, 768 * 1344),
        ] {
            let plane = allocated * pixels;
            let count = 3 * frames * pixels;
            let guard = -23.0f32;
            let mut host = vec![guard; 3 * plane + 6];
            for c in 0..3 {
                for i in 0..plane {
                    host[3 + c * plane + i] = if pixels == 65536 {
                        f32::from_bits((((i * 73 + c * 1009) % 65536) as u32) << 16)
                    } else {
                        ((i * 73 + i / pixels * 37 + c * 101) % 251) as f32 / 31.0 - 4.0
                    };
                }
            }
            let input = s.clone_htod(&host)?;
            let output = s.clone_htod(&vec![0xa5u8; count + 14])?;
            super::vae::planar_f32_to_rgb_u8(
                &device,
                &kernel,
                &input.slice(3..3 + 3 * plane),
                &output.slice(7..7 + count),
                (frames, allocated, pixels),
                (mean, std),
            )?;
            let observed = s.clone_dtoh(&output)?;
            ensure!(
                observed[..7]
                    .iter()
                    .chain(&observed[count + 7..])
                    .all(|&v| v == 0xa5),
                "RGB output guard changed"
            );
            ensure!(
                s.clone_dtoh(&input.slice(..3))? == [guard; 3]
                    && s.clone_dtoh(&input.slice(3 + 3 * plane..))? == [guard; 3],
                "RGB input guard changed"
            );
            let mut differences = 0usize;
            for (i, &actual) in observed[7..7 + count].iter().enumerate() {
                let c = i % 3;
                let v = (host[3 + c * plane + i / 3] * std[c] + mean[c]).clamp(0.0, 1.0) * 255.0;
                let expected = v.round_ties_even() as u8;
                differences += usize::from(actual != expected);
                if pixels == 65536 {
                    control_differences += usize::from(actual != v as u8);
                }
            }
            defects += usize::from(differences != 0);
            cells += 1;
            println!(
                "H3_RGB_IDENTITY frames={frames} allocated_frames={allocated} pixels={pixels} elements={count} byte_differences={differences} byte_limit=0 guards=true"
            );
        }
        let input = s.alloc_zeros::<f32>(9)?;
        let output = s.alloc_zeros::<u8>(9)?;
        convert(&input, &output, (1, 1, 1), (mean, std))?;
        let zero = s.clone_dtoh(&output.slice(..3))?;
        let zero_pass = zero == [124, 116, 104];
        defects += usize::from(!zero_pass);
        cells += 1;
        println!(
            "H3_RGB_IDENTITY decoded_zero={zero:?} expected=[124,116,104] byte_limit=0 {}",
            if zero_pass { "PASS" } else { "DEFECT" }
        );
        let invalid = [
            (0, 1, 1),
            (2, 1, 1),
            (1, 1, 0),
            (1, 1, 4),
            (1, usize::MAX, 2),
            (usize::MAX, usize::MAX, 1),
        ];
        for shape in invalid {
            ensure!(
                convert(&input, &output, shape, (mean, std)).is_err(),
                "RGB invalid extent accepted: {shape:?}"
            );
        }
        ensure!(
            super::vae::planar_f32_to_rgb_u8(
                &device,
                &kernel,
                &input,
                &output.slice(..2),
                (1, 1, 1),
                (mean, std)
            )
            .is_err(),
            "RGB short output accepted"
        );
        ensure!(
            convert(&input, &output, (1, 1, 1), ([f32::NAN; 3], std)).is_err(),
            "RGB nonfinite affine accepted"
        );
        ensure!(
            control_differences > 0,
            "RGB truncation control did not discriminate"
        );
        println!(
            "H3_RGB_CONTROL nearest_even_to_truncation byte_differences={control_differences} status=FAIL_AS_REQUIRED"
        );
        println!(
            "H3_RGB_GATE cells={cells} defects={defects} byte_limit=0 invalid_extents=8 exhaustive_patterns_per_channel=65536 channels=3"
        );
        ensure!(defects == 0, "H3_RGB_GATE DEFECT cells={defects}");
        Ok(())
    }

    /// Fused RMSNorm and indexed modulation preserve the five BF16 rounding boundaries.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted"]
    fn rms_norm_modulation_is_bitwise_invariant_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        device.finish_upload()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let plain_active = kernels
            .norm
            .occupancy_max_active_blocks_per_multiprocessor(HIDDEN_NORM_THREADS, 0, None)?;
        let fused_active = kernels
            .norm_modulate
            .occupancy_max_active_blocks_per_multiprocessor(HIDDEN_NORM_THREADS, 0, None)?;
        let gated_active = kernels
            .gated_add_rms
            .occupancy_max_active_blocks_per_multiprocessor(HIDDEN_NORM_THREADS, 0, None)?;
        println!(
            "H3_NORM_MODULATE_RESOURCE threads={HIDDEN_NORM_THREADS} plain_active={plain_active} fused_active={fused_active} gated_active={gated_active}"
        );
        ensure!(
            gated_active >= plain_active,
            "gated_add_rms_norm_modulate occupancy {gated_active} below rms_norm_bf16 {plain_active}"
        );
        let dim = 5376;
        let guard = bf16::from_f32(-23.0);
        let mut cells = 0;
        let mut defects = 0;
        let mut control_bits = 0;
        for rows in [1usize, 2, 33, 65, 40_045, 60_431] {
            for (stride, entries) in [(2 * dim, 1u32), (2 * dim, 7), (6 * dim, 1), (6 * dim, 7)] {
                let index: Vec<u32> = (0..rows)
                    .map(|r| ((r * 5 + r / 7) % entries as usize) as u32)
                    .collect();
                let indices = s.clone_htod(&index)?;
                let mut weight: Vec<bf16> = (0..dim)
                    .map(|d| bf16::from_f32(0.75 + (d * 31 % 257) as f32 / 512.0))
                    .collect();
                weight[0] = bf16::from_f32(129.0 / 128.0);
                let gamma = s.clone_htod(&weight)?;
                let mut table: Vec<bf16> = (0..stride * entries as usize)
                    .map(|i| bf16::from_f32(((i * 37 + i / dim * 13) % 251) as f32 / 256.0 - 0.5))
                    .collect();
                for entry in 0..entries as usize {
                    table[entry * stride] = bf16::from_f32(-130.0 / 128.0);
                    table[entry * stride + dim] = bf16::from_f32(1.0 / 128.0);
                }
                let params = s.clone_htod(&table)?;
                let mut input = s.alloc_zeros::<bf16>(rows * dim + 16)?;
                let mut serial = s.clone_htod(&vec![guard; rows * dim + 16])?;
                let mut fused = s.clone_htod(&vec![guard; rows * dim + 16])?;
                let input_at = pointer(&input, s) + 16;
                let serial_at = pointer(&serial, s) + 16;
                let fused_at = pointer(&fused, s) + 16;
                let norm = |output: u64| NormRows {
                    input: input_at,
                    weight: pointer(&gamma, s),
                    output,
                    rows,
                    dim,
                    threads: HIDDEN_NORM_THREADS,
                    eps: 1e-5,
                };
                let modulation = |output: u64| ModulateRows {
                    data: output,
                    shift: pointer(&params, s),
                    scale: pointer(&params, s) + 2 * dim as u64,
                    index: pointer(&indices, s),
                    rows,
                    dim,
                    stride,
                    entries,
                };
                let launch =
                    || kernels.rms_norm_modulate_f32(s, norm(fused_at), modulation(fused_at));
                s.synchronize()?;
                let graph = capture_graph(s, true, launch)?;
                for replay in 0..2 {
                    let host: Vec<bf16> = (0..rows * dim + 16)
                        .map(|i| {
                            if i < 8 || i >= rows * dim + 8 {
                                guard
                            } else if replay == 0 && i < dim + 8 {
                                bf16::ONE
                            } else {
                                bf16::from_f32(
                                    (((i - 8) * 43 + (i - 8) / dim * 17 + replay * 71) % 509)
                                        as f32
                                        / 127.0
                                        - 2.0,
                                )
                            }
                        })
                        .collect();
                    // The captured input pointer remains fixed across both replays.
                    let mut view = input.slice_mut(..);
                    s.memcpy_htod(&host, &mut view)?;
                    for captured in [false, true] {
                        s.memcpy_htod(&vec![guard; serial.len()], &mut serial)?;
                        s.memcpy_htod(&vec![guard; fused.len()], &mut fused)?;
                        kernels.rms_norm_f32(s, norm(serial_at))?;
                        kernels.row_modulate(s, modulation(serial_at))?;
                        if captured {
                            graph.launch()?;
                        } else {
                            launch()?;
                        }
                        let expected = s.clone_dtoh(&serial)?;
                        let actual = s.clone_dtoh(&fused)?;
                        ensure!(
                            expected.iter().chain(&actual).all(|v| v.is_finite()),
                            "RUNNER_FAIL nonfinite norm/modulation"
                        );
                        ensure!(
                            actual[..8]
                                .iter()
                                .chain(&actual[actual.len() - 8..])
                                .all(|v| *v == guard),
                            "RUNNER_FAIL norm/modulation output guard"
                        );
                        let bits = expected
                            .iter()
                            .zip(&actual)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        defects += usize::from(bits > 0);
                        cells += 1;
                        if replay == 0
                            && rows == 1
                            && entries == 1
                            && stride == 2 * dim
                            && !captured
                        {
                            let unrounded_product = (129.0f32 / 128.0) * (129.0 / 128.0);
                            let control = bf16::from_f32(unrounded_product - 130.0 / 128.0);
                            ensure!(
                                expected[8] == bf16::ZERO && control.to_f32() == 1.0 / 16384.0,
                                "RUNNER_FAIL scaled-product round control fixture"
                            );
                            control_bits = usize::from(actual[8].to_bits() != control.to_bits());
                            println!(
                                "H3_NORM_MODULATE_CONTROL omitted_scaled_product_round={} correct={} bit_differences={control_bits}",
                                control.to_f32(),
                                actual[8].to_f32()
                            );
                        }
                        println!(
                            "H3_NORM_MODULATE_CELL rows={rows} stride={stride} entries={entries} replay={replay} captured={captured} bit_differences={bits}"
                        );
                    }
                }
                for case in 0..6 {
                    let mut n = norm(fused_at);
                    let mut m = modulation(fused_at);
                    match case {
                        0 => {
                            n.output += 2;
                            m.data = n.output;
                        }
                        1 => {
                            n.dim += 1;
                            m.dim = n.dim;
                        }
                        2 => n.threads -= 32,
                        3 => m.index = 0,
                        4 => m.entries = 0,
                        _ => {
                            m.stride = usize::MAX;
                            m.entries = 2;
                        }
                    }
                    ensure!(
                        kernels.rms_norm_modulate_f32(s, n, m).is_err(),
                        "invalid norm/modulation case={case} accepted"
                    );
                }
            }
        }
        println!(
            "H3_NORM_MODULATE_GATE cells={cells} defects={defects} bit_limit=0 control_bit_differences={control_bits}"
        );
        ensure!(
            control_bits > 0,
            "RUNNER_FAIL scaled-product control undetected"
        );
        ensure!(
            defects == 0,
            "DEFECT norm/modulation bitwise cells={defects}"
        );
        let mut gated_cells = 0usize;
        let mut gated_defects = 0usize;
        for cell in [
            GatedRmsCell {
                rows: 4_103,
                gate_stride: 6 * dim,
                gate_entries: 7,
                norm_stride: 6 * dim,
                norm_entries: 7,
                alias_output: false,
            },
            GatedRmsCell {
                rows: 4_103,
                gate_stride: 6 * dim,
                gate_entries: 7,
                norm_stride: 2 * dim,
                norm_entries: 4,
                alias_output: true,
            },
            GatedRmsCell {
                rows: 39_741,
                gate_stride: 6 * dim,
                gate_entries: 4,
                norm_stride: 6 * dim,
                norm_entries: 4,
                alias_output: false,
            },
        ] {
            let bits = gated_rms_parent_bits(s, &kernels, dim, &cell)?;
            gated_defects += usize::from(bits > 0);
            gated_cells += 1;
            println!(
                "H3_GATED_RMS_CELL rows={} gate_stride={} norm_stride={} alias={} bit_differences={bits}",
                cell.rows, cell.gate_stride, cell.norm_stride, cell.alias_output
            );
        }
        let control_rows = gated_rms_raw_control(s, &kernels, dim)?;
        println!(
            "H3_GATED_RMS_GATE cells={gated_cells} defects={gated_defects} bit_limit=0 raw_variance_rows={control_rows}"
        );
        ensure!(
            control_rows == 1,
            "RUNNER_FAIL raw-variance control rows={control_rows}"
        );
        ensure!(
            gated_defects == 0,
            "DEFECT gated RMSNorm bitwise cells={gated_defects}"
        );
        Ok(())
    }

    struct GatedRmsCell {
        rows: usize,
        gate_stride: usize,
        gate_entries: u32,
        norm_stride: usize,
        norm_entries: u32,
        alias_output: bool,
    }

    fn gated_bit_diffs(name: &str, dim: usize, expected: &[bf16], actual: &[bf16]) -> usize {
        let mut n = 0usize;
        let mut shown = 0usize;
        for (i, (a, b)) in expected.iter().zip(actual).enumerate() {
            if a.to_bits() != b.to_bits() {
                if shown < 8 {
                    println!(
                        "H3_GATED_RMS_DIFF name={name} index={i} row={} col={} expected_bits={:#06x} actual_bits={:#06x}",
                        i / dim,
                        i % dim,
                        a.to_bits(),
                        b.to_bits()
                    );
                    shown += 1;
                }
                n += 1;
            }
        }
        if n > shown {
            println!("H3_GATED_RMS_DIFF name={name} further={n}");
        }
        n
    }

    fn parent_gated_rms(
        s: &Arc<CudaStream>,
        kernels: &BlockKernels,
        g: &GatedAddRms,
    ) -> Result<()> {
        let rows = i32::try_from(g.rows)?;
        let dim = i32::try_from(g.dim)?;
        unsafe {
            s.launch_builder(&kernels.gated_add)
                .arg(&g.x)
                .arg(&g.y)
                .arg(&g.gate)
                .arg(&g.gate_index)
                .arg(&rows)
                .arg(&dim)
                .arg(&(g.gate_stride as u64))
                .arg(&g.gate_entries)
                .launch(LaunchConfig {
                    grid_dim: (
                        u32::try_from(g.rows)?,
                        u32::try_from(g.dim.div_ceil(256))?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        kernels.rms_norm_modulate_f32(
            s,
            NormRows {
                input: g.x,
                weight: g.weight,
                output: g.output,
                rows: g.rows,
                dim: g.dim,
                threads: HIDDEN_NORM_THREADS,
                eps: g.eps,
            },
            ModulateRows {
                data: g.output,
                shift: g.shift,
                scale: g.scale,
                index: g.norm_index,
                rows: g.rows,
                dim: g.dim,
                stride: g.norm_stride,
                entries: g.norm_entries,
            },
        )
    }

    fn gated_rms_parent_bits(
        s: &Arc<CudaStream>,
        kernels: &BlockKernels,
        dim: usize,
        cell: &GatedRmsCell,
    ) -> Result<usize> {
        let rows = cell.rows;
        let x_host: Vec<bf16> = (0..rows * dim)
            .map(|i| bf16::from_f32(((i * 17 + 3) % 251) as f32 / 64.0 - 1.5))
            .collect();
        let y_host: Vec<bf16> = (0..rows * dim)
            .map(|i| bf16::from_f32(((i * 13 + 9) % 239) as f32 / 32.0 - 2.0))
            .collect();
        let gate_index: Vec<u32> = (0..rows)
            .map(|r| ((r * 3 + 1) % cell.gate_entries as usize) as u32)
            .collect();
        let norm_index: Vec<u32> = (0..rows)
            .map(|r| ((r * 5 + r / 7) % cell.norm_entries as usize) as u32)
            .collect();
        let weight: Vec<bf16> = (0..dim)
            .map(|d| bf16::from_f32(0.75 + (d * 31 % 257) as f32 / 512.0))
            .collect();
        let gate_table: Vec<bf16> = (0..cell.gate_stride * cell.gate_entries as usize)
            .map(|i| bf16::from_f32(((i * 19 + 5) % 191) as f32 / 96.0 - 0.4))
            .collect();
        let norm_table: Vec<bf16> = (0..cell.norm_stride * cell.norm_entries as usize)
            .map(|i| bf16::from_f32(((i * 23 + 11) % 181) as f32 / 80.0 - 0.7))
            .collect();
        let x_parent = s.clone_htod(&x_host)?;
        let x_fused = s.clone_htod(&x_host)?;
        let y_parent = s.clone_htod(&y_host)?;
        let y_fused = s.clone_htod(&y_host)?;
        let out_parent = s.alloc_zeros::<bf16>(rows * dim)?;
        let out_fused = s.alloc_zeros::<bf16>(rows * dim)?;
        let gate = s.clone_htod(&gate_table)?;
        let norm = s.clone_htod(&norm_table)?;
        let gamma = s.clone_htod(&weight)?;
        let gate_ids = s.clone_htod(&gate_index)?;
        let norm_ids = s.clone_htod(&norm_index)?;
        let spec = |x: u64, y: u64, separate: u64| GatedAddRms {
            x,
            y,
            gate: pointer(&gate, s),
            gate_index: pointer(&gate_ids, s),
            gate_stride: cell.gate_stride,
            gate_entries: cell.gate_entries,
            weight: pointer(&gamma, s),
            output: if cell.alias_output { y } else { separate },
            shift: pointer(&norm, s),
            scale: pointer(&norm, s) + 2 * dim as u64,
            norm_index: pointer(&norm_ids, s),
            norm_stride: cell.norm_stride,
            norm_entries: cell.norm_entries,
            rows,
            dim,
            eps: 1e-5,
        };
        let parent = spec(
            pointer(&x_parent, s),
            pointer(&y_parent, s),
            pointer(&out_parent, s),
        );
        let fused = spec(
            pointer(&x_fused, s),
            pointer(&y_fused, s),
            pointer(&out_fused, s),
        );
        parent_gated_rms(s, kernels, &parent)?;
        kernels.gated_add_rms(s, &fused)?;
        let x_bits = gated_bit_diffs(
            "x",
            dim,
            &s.clone_dtoh(&x_parent)?,
            &s.clone_dtoh(&x_fused)?,
        );
        let norm_expected = if cell.alias_output {
            s.clone_dtoh(&y_parent)?
        } else {
            s.clone_dtoh(&out_parent)?
        };
        let norm_actual = if cell.alias_output {
            s.clone_dtoh(&y_fused)?
        } else {
            s.clone_dtoh(&out_fused)?
        };
        ensure!(
            norm_expected
                .iter()
                .chain(&norm_actual)
                .all(|v| v.is_finite()),
            "RUNNER_FAIL nonfinite gated RMSNorm"
        );
        Ok(x_bits + gated_bit_diffs("norm", dim, &norm_expected, &norm_actual))
    }

    fn gated_rms_raw_control(
        s: &Arc<CudaStream>,
        kernels: &BlockKernels,
        dim: usize,
    ) -> Result<usize> {
        let rows = 2usize;
        let special = bf16::from_f32(1.01);
        let mut x_host = vec![bf16::ONE; rows * dim];
        let mut y_host = vec![bf16::ONE; rows * dim];
        let mut gate_host = vec![bf16::ONE; dim];
        x_host[0] = special;
        y_host[0] = special;
        gate_host[0] = special;
        let weight = vec![bf16::ONE; dim];
        let shift = vec![bf16::ZERO; dim];
        let scale = vec![bf16::ZERO; dim];
        let index = vec![0u32; rows];
        let x_rounded = s.clone_htod(&x_host)?;
        let x_raw = s.clone_htod(&x_host)?;
        let y = s.clone_htod(&y_host)?;
        let out_rounded = s.alloc_zeros::<bf16>(rows * dim)?;
        let out_raw = s.alloc_zeros::<bf16>(rows * dim)?;
        let gate = s.clone_htod(&gate_host)?;
        let gamma = s.clone_htod(&weight)?;
        let shift = s.clone_htod(&shift)?;
        let scale = s.clone_htod(&scale)?;
        let ids = s.clone_htod(&index)?;
        let spec = |x: u64, out: u64| GatedAddRms {
            x,
            y: pointer(&y, s),
            gate: pointer(&gate, s),
            gate_index: pointer(&ids, s),
            gate_stride: dim,
            gate_entries: 1,
            weight: pointer(&gamma, s),
            output: out,
            shift: pointer(&shift, s),
            scale: pointer(&scale, s),
            norm_index: pointer(&ids, s),
            norm_stride: dim,
            norm_entries: 1,
            rows,
            dim,
            eps: 1e-5,
        };
        let rounded = spec(pointer(&x_rounded, s), pointer(&out_rounded, s));
        let raw = spec(pointer(&x_raw, s), pointer(&out_raw, s));
        kernels.gated_add_rms(s, &rounded)?;
        kernels.gated_add_rms_raw(s, &raw)?;
        let x_bits = gated_bit_diffs(
            "control_x",
            dim,
            &s.clone_dtoh(&x_rounded)?,
            &s.clone_dtoh(&x_raw)?,
        );
        ensure!(
            x_bits == 0,
            "RUNNER_FAIL raw-variance control changed the stored residual"
        );
        let expected = s.clone_dtoh(&out_rounded)?;
        let actual = s.clone_dtoh(&out_raw)?;
        ensure!(
            expected.iter().chain(&actual).all(|v| v.is_finite()),
            "RUNNER_FAIL nonfinite raw-variance control"
        );
        let mut rows_differ = 0usize;
        for row in 0..rows {
            let start = row * dim;
            let differ = expected[start..start + dim]
                .iter()
                .zip(&actual[start..start + dim])
                .any(|(a, b)| a.to_bits() != b.to_bits());
            if differ {
                rows_differ += 1;
                let _ = gated_bit_diffs(
                    "control_norm",
                    dim,
                    &expected[start..start + dim],
                    &actual[start..start + dim],
                );
            }
        }
        Ok(rows_differ)
    }

    fn bf16_round(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    /// Host of `norm_scale`: `pre_weight` rounds `x * rstd` before the weight multiply.
    fn host_norm_rows(
        x: &[bf16],
        weight: &[bf16],
        rows: usize,
        dim: usize,
        eps: f32,
        pre_weight: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * dim];
        for row in 0..rows {
            let mut ssq = 0.0f32;
            for value in &x[row * dim..(row + 1) * dim] {
                let v = value.to_f32();
                ssq += v * v;
            }
            let rstd = 1.0 / (ssq / dim as f32 + eps).sqrt();
            for (d, value) in x[row * dim..(row + 1) * dim].iter().enumerate() {
                let normalized = value.to_f32() * rstd;
                let scaled = if pre_weight {
                    bf16_round(normalized) * weight[d].to_f32()
                } else {
                    normalized * weight[d].to_f32()
                };
                out[row * dim + d] = bf16_round(scaled);
            }
        }
        out
    }

    fn host_modulate(value: f32, shift: f32, scale: f32) -> f32 {
        let weighted = bf16_round(value);
        let factor = bf16_round(1.0 + scale);
        let scaled = bf16_round(weighted * factor);
        bf16_round(scaled + shift)
    }

    fn f32_abs_diff(actual: &[bf16], expected: &[f32]) -> (usize, f32) {
        let mut n = 0usize;
        let mut max = 0.0f32;
        for (got, want) in actual.iter().zip(expected) {
            let delta = (got.to_f32() - want).abs();
            if got.to_f32().to_bits() != want.to_bits() {
                n += 1;
            }
            max = max.max(delta);
        }
        (n, max)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn qk_rope_fusion_matches_parent_gpu() -> Result<()> {
        use cudarc::driver::sys::{CUdevice_attribute as DA, CUfunction_attribute as FA};
        let device = Device::new(0)?;
        let s = &device.stream;
        let kernels = BlockKernels::new(&device.ctx)?;
        let max_threads_sm = u32::try_from(
            device
                .ctx
                .attribute(DA::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        )?;
        for (label, func) in [
            ("rotary_q", &kernels.rotary_q),
            ("rotary_kv", &kernels.rotary_kv),
        ] {
            let local_bytes = func.get_attribute(FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
            let regs = func.get_attribute(FA::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
            let smem = func.get_attribute(FA::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
            let active_blocks =
                func.occupancy_max_active_blocks_per_multiprocessor(QK_ROPE_THREADS, 0, None)?;
            let active_warps = active_blocks * (QK_ROPE_THREADS / 32);
            println!(
                "H3_QK_ROPE_OCCUPANCY kernel={label} threads={QK_ROPE_THREADS} active_blocks={active_blocks} active_warps={active_warps} regs={regs} smem={smem} local={local_bytes}"
            );
            ensure!(
                local_bytes == 0,
                "{label} spills to local memory ({local_bytes} B)"
            );
            ensure!(
                active_blocks * QK_ROPE_THREADS >= max_threads_sm.min(1536),
                "{label} occupancy {active_warps} warps/SM below target on max_threads={max_threads_sm}"
            );
        }

        let cases = [
            (
                "prod", 1000usize, 56usize, 128usize, 96usize, 0i32, 1000usize,
            ),
            (
                "refiner", 14usize, 56usize, 128usize, 96usize, 0i32, 14usize,
            ),
            (
                "ragged_start",
                13usize,
                57usize,
                128usize,
                96usize,
                5i32,
                29usize,
            ),
            (
                "plain_no_rope",
                9usize,
                7usize,
                128usize,
                0usize,
                2i32,
                15usize,
            ),
        ];
        for (label, rows, heads, dim, rotary, start, capacity) in cases {
            let elements = rows * heads * dim;
            let kv_elements = heads * capacity * dim;
            let head_rows = rows * heads;
            let rows_i = i32::try_from(rows)?;
            let head_rows_i = i32::try_from(head_rows)?;
            let dim_i = i32::try_from(dim)?;
            let heads_i = i32::try_from(heads)?;
            let rotary_i = i32::try_from(rotary)?;
            let capacity_i = i32::try_from(capacity)?;
            let eps = 1.0e-5f32;
            let host = |offset: usize| -> Vec<bf16> {
                (0..elements)
                    .map(|i| bf16::from_f32((((i * 37 + offset) % 509) as f32 - 254.0) / 127.0))
                    .collect()
            };
            let q = host(0);
            let k = host(71);
            let v = host(143);
            let mut q_init = vec![bf16::ZERO; kv_elements];
            q_init[..elements].copy_from_slice(&q);
            let weight: Vec<bf16> = (0..dim)
                .map(|i| bf16::from_f32(0.5 + (i * 13 % 29) as f32 / 29.0))
                .collect();
            let table: Vec<bf16> = (0..=(capacity * rotary.max(1)))
                .map(|i| bf16::from_f32((i as f32 * 0.017).cos()))
                .collect();
            let weight = s.clone_htod(&weight)?;
            let table = s.clone_htod(&table)?;
            let mut q_parent = s.clone_htod(&q_init)?;
            let mut k_parent = s.alloc_zeros::<bf16>(kv_elements)?;
            let v_parent = s.clone_htod(&v)?;
            let k_fused = s.clone_htod(&k)?;
            let mut q_norm_parent = s.alloc_zeros::<bf16>(elements)?;
            let mut k_norm_parent = s.alloc_zeros::<bf16>(elements)?;
            let norm_grid = LaunchConfig {
                grid_dim: (u32::try_from(head_rows)?, 1, 1),
                block_dim: (HEAD_NORM_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let qk_rope_cfg = LaunchConfig {
                grid_dim: (u32::try_from(head_rows.div_ceil(QK_ROPE_WARPS))?, 1, 1),
                block_dim: (QK_ROPE_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                s.launch_builder(&kernels.norm_f32)
                    .arg(&q_parent)
                    .arg(&weight)
                    .arg(&mut q_norm_parent)
                    .arg(&head_rows_i)
                    .arg(&dim_i)
                    .arg(&eps)
                    .launch(norm_grid)?;
                s.launch_builder(&kernels.norm_f32)
                    .arg(&k_fused)
                    .arg(&weight)
                    .arg(&mut k_norm_parent)
                    .arg(&head_rows_i)
                    .arg(&dim_i)
                    .arg(&eps)
                    .launch(norm_grid)?;
                s.launch_builder(&kernels.rotary)
                    .arg(&mut q_norm_parent)
                    .arg(&k_norm_parent)
                    .arg(&v_parent)
                    .arg(&mut k_parent)
                    .arg(&mut q_parent)
                    .arg(&table)
                    .arg(&start)
                    .arg(&rows_i)
                    .arg(&heads_i)
                    .arg(&heads_i)
                    .arg(&dim_i)
                    .arg(&rotary_i)
                    .arg(&capacity_i)
                    .launch(flat(2 * elements))?;
            }
            let q_fused = s.clone_htod(&q_init)?;
            let v_fused = s.clone_htod(&v)?;
            let mut q_norm_fused = s.alloc_zeros::<bf16>(elements)?;
            let k_norm_fused = s.alloc_zeros::<bf16>(kv_elements)?;
            unsafe {
                s.launch_builder(&kernels.rotary_q)
                    .arg(&pointer(&q_fused, s))
                    .arg(&weight)
                    .arg(&mut q_norm_fused)
                    .arg(&table)
                    .arg(&start)
                    .arg(&rows_i)
                    .arg(&heads_i)
                    .arg(&dim_i)
                    .arg(&rotary_i)
                    .arg(&capacity_i)
                    .arg(&eps)
                    .launch(qk_rope_cfg)?;
                s.launch_builder(&kernels.rotary_kv)
                    .arg(&pointer(&k_fused, s))
                    .arg(&weight)
                    .arg(&v_fused)
                    .arg(&pointer(&k_norm_fused, s))
                    .arg(&pointer(&q_fused, s))
                    .arg(&table)
                    .arg(&start)
                    .arg(&rows_i)
                    .arg(&heads_i)
                    .arg(&dim_i)
                    .arg(&rotary_i)
                    .arg(&capacity_i)
                    .arg(&eps)
                    .launch(qk_rope_cfg)?;
            }
            let v_ref = s.clone_dtoh(&q_parent)?;
            let q_ref = s.clone_dtoh(&q_norm_parent)?;
            let k_ref = s.clone_dtoh(&k_parent)?;
            let v_got = s.clone_dtoh(&q_fused)?;
            let q_got = s.clone_dtoh(&q_norm_fused)?;
            let k_got = s.clone_dtoh(&k_norm_fused)?;
            let q_mismatch = q_ref
                .iter()
                .zip(&q_got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let k_mismatch = k_ref
                .iter()
                .zip(&k_got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let v_mismatch = v_ref
                .iter()
                .zip(&v_got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                q_mismatch == 0 && k_mismatch == 0 && v_mismatch == 0,
                "Q/K/V fusion mismatches {label} rows={rows}: {q_mismatch}/{k_mismatch}/{v_mismatch}"
            );
            let mut mutant_mismatch = 0usize;
            if rotary > 0 {
                let q_mutant = s.clone_htod(&q)?;
                let mut q_norm_mutant = s.alloc_zeros::<bf16>(elements)?;
                unsafe {
                    s.launch_builder(&kernels.rotary_q)
                        .arg(&q_mutant)
                        .arg(&weight)
                        .arg(&mut q_norm_mutant)
                        .arg(&(pointer(&table, s) + 2))
                        .arg(&start)
                        .arg(&rows_i)
                        .arg(&heads_i)
                        .arg(&dim_i)
                        .arg(&rotary_i)
                        .arg(&capacity_i)
                        .arg(&eps)
                        .launch(qk_rope_cfg)?;
                }
                let q_mutant = s.clone_dtoh(&q_norm_mutant)?;
                mutant_mismatch = q_ref
                    .iter()
                    .zip(&q_mutant)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                ensure!(
                    mutant_mismatch > 0,
                    "one-position RoPE mutant was not detected on {label}"
                );
            }
            if label == "refiner" {
                // Perturb one input element in [row=0, head=2] (inside CTA 0) and assert 0 bleed to sibling warps.
                let (p_row, p_head, p_col) = (0usize, 2usize, 5usize);
                let idx = (p_row * heads + p_head) * dim + p_col;
                let mut q_p_host = q_init.clone();
                let mut k_p_host = k.clone();
                let mut v_p_host = v.clone();
                q_p_host[idx] = bf16::from_f32(q_p_host[idx].to_f32() + 1.0);
                k_p_host[idx] = bf16::from_f32(k_p_host[idx].to_f32() + 1.0);
                v_p_host[idx] = bf16::from_f32(v_p_host[idx].to_f32() + 1.0);
                let q_p_dev = s.clone_htod(&q_p_host)?;
                let k_p_dev = s.clone_htod(&k_p_host)?;
                let v_p_dev = s.clone_htod(&v_p_host)?;
                let mut q_out_p = s.alloc_zeros::<bf16>(elements)?;
                let k_out_p = s.alloc_zeros::<bf16>(kv_elements)?;
                unsafe {
                    s.launch_builder(&kernels.rotary_q)
                        .arg(&pointer(&q_p_dev, s))
                        .arg(&weight)
                        .arg(&mut q_out_p)
                        .arg(&table)
                        .arg(&start)
                        .arg(&rows_i)
                        .arg(&heads_i)
                        .arg(&dim_i)
                        .arg(&rotary_i)
                        .arg(&capacity_i)
                        .arg(&eps)
                        .launch(qk_rope_cfg)?;
                    s.launch_builder(&kernels.rotary_kv)
                        .arg(&pointer(&k_p_dev, s))
                        .arg(&weight)
                        .arg(&v_p_dev)
                        .arg(&pointer(&k_out_p, s))
                        .arg(&pointer(&q_p_dev, s))
                        .arg(&table)
                        .arg(&start)
                        .arg(&rows_i)
                        .arg(&heads_i)
                        .arg(&dim_i)
                        .arg(&rotary_i)
                        .arg(&capacity_i)
                        .arg(&eps)
                        .launch(qk_rope_cfg)?;
                }
                let q_p_got = s.clone_dtoh(&q_out_p)?;
                let k_p_got = s.clone_dtoh(&k_out_p)?;
                let v_p_got = s.clone_dtoh(&q_p_dev)?;
                let q_target_start = (p_row * heads + p_head) * dim;
                let kv_target_start = (p_head * capacity + usize::try_from(start)? + p_row) * dim;
                let mut q_hit = 0usize;
                let mut q_bleed = 0usize;
                for (i, (a, b)) in q_ref.iter().zip(&q_p_got).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        if (q_target_start..q_target_start + dim).contains(&i) {
                            q_hit += 1;
                        } else {
                            q_bleed += 1;
                        }
                    }
                }
                let mut k_hit = 0usize;
                let mut k_bleed = 0usize;
                for (i, (a, b)) in k_ref.iter().zip(&k_p_got).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        if (kv_target_start..kv_target_start + dim).contains(&i) {
                            k_hit += 1;
                        } else {
                            k_bleed += 1;
                        }
                    }
                }
                let mut v_hit = 0usize;
                let mut v_bleed = 0usize;
                for (i, (a, b)) in v_ref.iter().zip(&v_p_got).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        if (kv_target_start..kv_target_start + dim).contains(&i) {
                            v_hit += 1;
                        } else {
                            v_bleed += 1;
                        }
                    }
                }
                ensure!(
                    q_hit > 0
                        && k_hit > 0
                        && v_hit == 1
                        && q_bleed == 0
                        && k_bleed == 0
                        && v_bleed == 0,
                    "CTA warp isolation failed: hit={q_hit}/{k_hit}/{v_hit} bleed={q_bleed}/{k_bleed}/{v_bleed}"
                );
                println!(
                    "H3_QK_ROPE_ISOLATION case={label} hit={q_hit}/{k_hit}/{v_hit} bleed={q_bleed}/{k_bleed}/{v_bleed}"
                );
            }
            println!(
                "H3_QK_ROPE_FUSION case={label} rows={rows} heads={heads} start={start} cap={capacity} rotary={rotary} q_mismatch={q_mismatch} k_mismatch={k_mismatch} v_mismatch={v_mismatch} mutant_mismatch={mutant_mismatch}"
            );
        }
        Ok(())
    }

    /// The f32-weight kernels match the post-weight cast and miss the pre-weight cast. The BF16 kernels do the opposite.
    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn hyperflow_norm_skips_the_pre_weight_round_gpu() -> Result<()> {
        let device = Device::new(0)?;
        let s = &device.stream;
        device.finish_upload()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let eps = 1.0e-5f32;
        let pattern = |len: usize, stride: usize, bias: usize| -> Vec<bf16> {
            (0..len)
                .map(|i| bf16::from_f32(((i * stride + bias) % 251) as f32 / 17.0 - 7.0))
                .collect()
        };
        let weight_of = |dim: usize| -> Vec<bf16> {
            (0..dim)
                .map(|d| bf16::from_f32(0.5 + (d * 13 % 64) as f32 / 32.0))
                .collect()
        };
        let score = |label: &str, actual: &[bf16], pre: &[f32], post: &[f32]| -> Result<()> {
            let (pre_n, pre_max) = f32_abs_diff(actual, pre);
            let (post_n, post_max) = f32_abs_diff(actual, post);
            let (formula_n, formula_max) = f32_abs_diff(
                &pre.iter().copied().map(bf16::from_f32).collect::<Vec<_>>(),
                post,
            );
            println!(
                "HYPERFLOW_NORM_CONTROL {label} pre_diff={pre_n} pre_max={pre_max:.8} post_diff={post_n} post_max={post_max:.8} formula_diff={formula_n} formula_max={formula_max:.8}"
            );
            ensure!(
                formula_n > 0 && formula_max >= 0.015625,
                "RUNNER_FAIL {label} formulas do not separate"
            );
            Ok(())
        };
        let separate = |label: &str,
                        closer_to_post: bool,
                        actual: &[bf16],
                        pre: &[f32],
                        post: &[f32]|
         -> Result<()> {
            score(label, actual, pre, post)?;
            let (pre_n, _) = f32_abs_diff(actual, pre);
            let (post_n, _) = f32_abs_diff(actual, post);
            if closer_to_post {
                ensure!(
                    post_n * 10 < pre_n,
                    "DEFECT {label} f32-weight kernel matches the pre-weight round pre_diff={pre_n} post_diff={post_n}"
                );
            } else {
                ensure!(
                    pre_n * 10 < post_n,
                    "DEFECT {label} bf16 kernel matches the post-weight round pre_diff={pre_n} post_diff={post_n}"
                );
            }
            Ok(())
        };

        let rows = 3usize;
        let dim = 128usize;
        let x = pattern(rows * dim, 17, 3);
        let w = weight_of(dim);
        let pre = host_norm_rows(&x, &w, rows, dim, eps, true);
        let post = host_norm_rows(&x, &w, rows, dim, eps, false);
        let input = s.clone_htod(&x)?;
        let gamma = s.clone_htod(&w)?;
        let y_f32 = s.alloc_zeros::<bf16>(rows * dim)?;
        let y_bf16 = s.alloc_zeros::<bf16>(rows * dim)?;
        let norm = |output: &CudaSlice<bf16>| NormRows {
            input: pointer(&input, s),
            weight: pointer(&gamma, s),
            output: pointer(output, s),
            rows,
            dim,
            threads: HEAD_NORM_THREADS,
            eps,
        };
        kernels.rms_norm_f32(s, norm(&y_f32))?;
        kernels.rms_norm(s, norm(&y_bf16))?;
        let got_f32 = s.clone_dtoh(&y_f32)?;
        let got_bf16 = s.clone_dtoh(&y_bf16)?;
        let device_diff = got_f32
            .iter()
            .zip(&got_bf16)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("HYPERFLOW_NORM_CONTROL qk device_diff={device_diff} rows={rows} dim={dim}");
        ensure!(device_diff > 0, "DEFECT qk kernels are bitwise identical");
        separate("qk-f32", true, &got_f32, &pre, &post)?;
        separate("qk-bf16", false, &got_bf16, &pre, &post)?;

        for (rows, dim) in [(3usize, 128usize), (1usize, 5376usize)] {
            let x = pattern(rows * dim, 17, 3);
            let w = weight_of(dim);
            let shift_h: Vec<bf16> = (0..dim)
                .map(|d| bf16::from_f32(((d * 37) % 251) as f32 / 256.0 - 0.5))
                .collect();
            let scale_h: Vec<bf16> = (0..dim)
                .map(|d| bf16::from_f32(((d * 19 + 5) % 251) as f32 / 256.0 - 0.5))
                .collect();
            let pre = modulate_rows(&x, &w, &shift_h, &scale_h, dim, eps, true);
            let post = modulate_rows(&x, &w, &shift_h, &scale_h, dim, eps, false);
            let input = s.clone_htod(&x)?;
            let gamma = s.clone_htod(&w)?;
            let shift = s.clone_htod(&shift_h)?;
            let scale = s.clone_htod(&scale_h)?;
            let index = s.clone_htod(&vec![0u32; rows])?;
            let y_f32 = s.alloc_zeros::<bf16>(rows * dim)?;
            let y_bf16 = s.alloc_zeros::<bf16>(rows * dim)?;
            let norm = |output: &CudaSlice<bf16>| NormRows {
                input: pointer(&input, s),
                weight: pointer(&gamma, s),
                output: pointer(output, s),
                rows,
                dim,
                threads: HIDDEN_NORM_THREADS,
                eps,
            };
            let modulation = |output: &CudaSlice<bf16>| ModulateRows {
                data: pointer(output, s),
                shift: pointer(&shift, s),
                scale: pointer(&scale, s),
                index: pointer(&index, s),
                rows,
                dim,
                stride: dim,
                entries: 1,
            };
            kernels.rms_norm_modulate_f32(s, norm(&y_f32), modulation(&y_f32))?;
            kernels.rms_norm_modulate(s, norm(&y_bf16), modulation(&y_bf16))?;
            let got_f32 = s.clone_dtoh(&y_f32)?;
            let got_bf16 = s.clone_dtoh(&y_bf16)?;
            let device_diff = got_f32
                .iter()
                .zip(&got_bf16)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "HYPERFLOW_NORM_CONTROL modulate device_diff={device_diff} rows={rows} dim={dim}"
            );
            ensure!(
                device_diff > 0,
                "DEFECT modulate kernels are bitwise identical rows={rows} dim={dim}"
            );
            separate(&format!("modulate-f32-{dim}"), true, &got_f32, &pre, &post)?;
            separate(
                &format!("modulate-bf16-{dim}"),
                false,
                &got_bf16,
                &pre,
                &post,
            )?;
        }
        Ok(())
    }

    /// MiniMax-H3 counterpart of the HyperFlow norm control: every norm site
    /// (modulated norm1/norm2 and norm_out, the plain refiner norms and
    /// final_norm, and the head-dim Q/K norms) runs the post-weight-round
    /// form on this path, matching the definition's single cast after the
    /// RMS weight.
    #[test]
    #[ignore = "GPU"]
    fn h3_norms_match_the_post_weight_round_gpu() -> Result<()> {
        let device = Device::new(0)?;
        let s = &device.stream;
        device.finish_upload()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let eps = 1.0e-5f32;
        let pattern = |len: usize, stride: usize, bias: usize| -> Vec<bf16> {
            (0..len)
                .map(|i| bf16::from_f32(((i * stride + bias) % 251) as f32 / 17.0 - 7.0))
                .collect()
        };
        let weight_of = |dim: usize| -> Vec<bf16> {
            (0..dim)
                .map(|d| bf16::from_f32(0.5 + (d * 13 % 64) as f32 / 32.0))
                .collect()
        };
        let score = |label: &str, actual: &[bf16], pre: &[f32], post: &[f32]| -> Result<()> {
            let (pre_n, pre_max) = f32_abs_diff(actual, pre);
            let (post_n, post_max) = f32_abs_diff(actual, post);
            println!(
                "H3_NORM_CONTROL {label} pre_diff={pre_n} pre_max={pre_max:.8} post_diff={post_n} post_max={post_max:.8}"
            );
            // Reduction order in the wide rows leaves a few one-ulp elements
            // against the host composition, so closeness is a ratio.
            ensure!(
                post_n * 10 < pre_n,
                "DEFECT {label} kernel matches the pre-weight round pre_diff={pre_n} post_diff={post_n}"
            );
            ensure!(
                pre_n > 0 && pre_max >= 0.015625,
                "DEFECT {label} pre-weight composition not separated"
            );
            Ok(())
        };

        // Q/K norm: head-dim rows (heads x rows rows of head_dim).
        for (rows, dim, threads) in [
            (8usize, 128usize, HEAD_NORM_THREADS),
            (3usize, 64usize, HEAD_NORM_THREADS),
        ] {
            let x = pattern(rows * dim, 17, 3);
            let w = weight_of(dim);
            let pre = host_norm_rows(&x, &w, rows, dim, eps, true);
            let post = host_norm_rows(&x, &w, rows, dim, eps, false);
            let input = s.clone_htod(&x)?;
            let gamma = s.clone_htod(&w)?;
            let y = s.alloc_zeros::<bf16>(rows * dim)?;
            kernels.rms_norm_f32(
                s,
                NormRows {
                    input: pointer(&input, s),
                    weight: pointer(&gamma, s),
                    output: pointer(&y, s),
                    rows,
                    dim,
                    threads,
                    eps,
                },
            )?;
            score(&format!("qk-{rows}x{dim}"), &s.clone_dtoh(&y)?, &pre, &post)?;
        }

        // Refiner norm1/norm2 and final_norm: plain hidden-dim rows.
        for (rows, dim) in [(3usize, 128usize), (1usize, 5376usize)] {
            let x = pattern(rows * dim, 17, 3);
            let w = weight_of(dim);
            let pre = host_norm_rows(&x, &w, rows, dim, eps, true);
            let post = host_norm_rows(&x, &w, rows, dim, eps, false);
            let input = s.clone_htod(&x)?;
            let gamma = s.clone_htod(&w)?;
            let y = s.alloc_zeros::<bf16>(rows * dim)?;
            kernels.rms_norm_f32(
                s,
                NormRows {
                    input: pointer(&input, s),
                    weight: pointer(&gamma, s),
                    output: pointer(&y, s),
                    rows,
                    dim,
                    threads: HIDDEN_NORM_THREADS,
                    eps,
                },
            )?;
            score(
                &format!("hidden-{rows}x{dim}"),
                &s.clone_dtoh(&y)?,
                &pre,
                &post,
            )?;
        }

        // Modulated norm1/norm2 and norm_out.
        for (rows, dim) in [(3usize, 128usize), (1usize, 5376usize)] {
            let x = pattern(rows * dim, 17, 3);
            let w = weight_of(dim);
            let shift_h: Vec<bf16> = (0..dim)
                .map(|d| bf16::from_f32(((d * 37) % 251) as f32 / 256.0 - 0.5))
                .collect();
            let scale_h: Vec<bf16> = (0..dim)
                .map(|d| bf16::from_f32(((d * 19 + 5) % 251) as f32 / 256.0 - 0.5))
                .collect();
            let pre = modulate_rows(&x, &w, &shift_h, &scale_h, dim, eps, true);
            let post = modulate_rows(&x, &w, &shift_h, &scale_h, dim, eps, false);
            let input = s.clone_htod(&x)?;
            let gamma = s.clone_htod(&w)?;
            let shift = s.clone_htod(&shift_h)?;
            let scale = s.clone_htod(&scale_h)?;
            let index = s.clone_htod(&vec![0u32; rows])?;
            let y = s.alloc_zeros::<bf16>(rows * dim)?;
            kernels.rms_norm_modulate_f32(
                s,
                NormRows {
                    input: pointer(&input, s),
                    weight: pointer(&gamma, s),
                    output: pointer(&y, s),
                    rows,
                    dim,
                    threads: HIDDEN_NORM_THREADS,
                    eps,
                },
                ModulateRows {
                    data: pointer(&y, s),
                    shift: pointer(&shift, s),
                    scale: pointer(&scale, s),
                    index: pointer(&index, s),
                    rows,
                    dim,
                    stride: dim,
                    entries: 1,
                },
            )?;
            score(
                &format!("modulate-{rows}x{dim}"),
                &s.clone_dtoh(&y)?,
                &pre,
                &post,
            )?;
        }
        println!("PASS h3_norms post_weight=true sites=qk,hidden,modulate");
        Ok(())
    }

    /// Engine-level norm control: `block_forward` on the MiniMax-H3
    /// (non-hyperflow) path with both modulation gates zero, so the residual
    /// passes attention and the FFN unchanged and `work.normed` holds exactly
    /// the modulated norm2 of the input rows. The definition rounds once,
    /// after the RMS weight; this path launched the pre-weight round before
    /// this branch.
    #[test]
    #[ignore = "GPU; FF_MODEL, FF_H3_LORA"]
    fn h3_block_norm_matches_the_post_weight_round_gpu() -> Result<()> {
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let lora_path = path("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        ensure!(!spec.is_hyperflow()?, "control needs the MiniMax-H3 path");
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-normctl-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 block norm control".into())),
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let mut base = Source::new(&transformer, &manifest);
        let mut lora = Source::new(&lora_dir, &lora_manifest);
        let weights = BlockWeights::resident::<true>(
            s,
            &spec,
            adapter.rank,
            "transformer_blocks.0",
            &mut base,
            Some(&mut lora),
        )?;
        let rows = 1usize;
        let mut work = BlockWork::new(s, &spec, adapter.rank, rows)?;
        work.prepare::<true>(&blas, &spec)?;
        work.keep_normed = true;
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let hidden = spec.hidden;
        let input: Vec<bf16> = (0..rows * hidden)
            .map(|i| bf16::from_f32(((i * 17 + 3) % 251) as f32 / 17.0 - 7.0))
            .collect();
        let pattern = |d: usize, m: usize| bf16::from_f32(((d * m + 5) % 251) as f32 / 256.0 - 0.5);
        let mut table = vec![bf16::from_f32(0.0); 6 * hidden];
        for d in 0..hidden {
            table[d] = pattern(d, 37);
            table[hidden + d] = pattern(d, 19);
            table[2 * hidden + d] = bf16::from_f32(0.0);
            table[3 * hidden + d] = pattern(d, 53);
            table[4 * hidden + d] = pattern(d, 29);
            table[5 * hidden + d] = bf16::from_f32(0.0);
        }
        let modulation_table = s.clone_htod(&table)?;
        let index = s.clone_htod(&vec![0u32; rows])?;
        let modulation = Modulation {
            table: &modulation_table,
            offset: 0,
            entries: 1,
            index: &index,
        };
        let rotary = s.alloc_zeros::<bf16>(rows * 6 * spec.rope_freq_dim)?;
        let mut x = s.clone_htod(&input)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: adapter.scale,
        };
        block_forward::<true>(
            &context,
            &mut flash,
            &weights,
            &Conditioning::Modulated {
                rotary: &rotary,
                modulation: &modulation,
            },
            &mut x,
            &mut work,
        )?;
        let output = s.clone_dtoh(&x)?;
        ensure!(
            output
                .iter()
                .zip(&input)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "zero gates must leave the residual untouched"
        );
        let normed = std::mem::take(&mut work.normed_probe);
        let norm2 = s.clone_dtoh(
            weights
                .norm2
                ._owned
                .as_ref()
                .context("resident norm2 allocation")?,
        )?;
        let shift_mlp = &table[3 * hidden..4 * hidden];
        let scale_mlp = &table[4 * hidden..5 * hidden];
        let pre = modulate_rows(
            &input,
            &norm2,
            shift_mlp,
            scale_mlp,
            hidden,
            spec.norm_eps,
            true,
        );
        let post = modulate_rows(
            &input,
            &norm2,
            shift_mlp,
            scale_mlp,
            hidden,
            spec.norm_eps,
            false,
        );
        let (pre_n, pre_max) = f32_abs_diff(&normed, &pre);
        let (post_n, post_max) = f32_abs_diff(&normed, &post);
        println!(
            "H3_BLOCK_NORM_CONTROL pre_diff={pre_n} pre_max={pre_max:.8} post_diff={post_n} post_max={post_max:.8}"
        );
        ensure!(
            post_n * 10 < pre_n,
            "block normed output matches the pre-weight round pre_diff={pre_n} post_diff={post_n}"
        );
        ensure!(
            pre_n > 0 && pre_max >= 0.015625,
            "pre-weight composition not separated"
        );
        fs::remove_dir_all(&runtime).ok();
        println!("PASS h3_block_norm post_weight=true path=block_forward");
        Ok(())
    }

    fn modulate_rows(
        x: &[bf16],
        weight: &[bf16],
        shift: &[bf16],
        scale: &[bf16],
        dim: usize,
        eps: f32,
        pre_weight: bool,
    ) -> Vec<f32> {
        let rows = x.len() / dim;
        let mut out = vec![0.0f32; rows * dim];
        for row in 0..rows {
            let mut ssq = 0.0f32;
            for value in &x[row * dim..(row + 1) * dim] {
                let v = value.to_f32();
                ssq += v * v;
            }
            let rstd = 1.0 / (ssq / dim as f32 + eps).sqrt();
            for d in 0..dim {
                let normalized = x[row * dim + d].to_f32() * rstd;
                let scaled = if pre_weight {
                    bf16_round(normalized) * weight[d].to_f32()
                } else {
                    normalized * weight[d].to_f32()
                };
                out[row * dim + d] = host_modulate(scaled, shift[d].to_f32(), scale[d].to_f32());
            }
        }
        out
    }

    /// `cast_rows` with `count == 0` launches nothing (the kernel traps on a zero count); a non-zero count rounds.
    #[test]
    #[ignore = "GPU test"]
    fn cast_rows_skips_an_empty_conditioning_range_gpu() -> Result<()> {
        let device = Device::new(0)?;
        let s = &device.stream;
        let kernel =
            cubin::module(s.context(), "vision_ops")?.load_function("vision_affine_cast_bf16")?;
        let values: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 / 3.0).collect();
        let source = s.clone_htod(&values)?;
        let target = s.alloc_zeros::<bf16>(16)?;
        cast_rows(s, &kernel, pointer(&source, s), pointer(&target, s), 0)?;
        s.synchronize()?;
        ensure!(
            s.clone_dtoh(&target)?.iter().all(|v| v.to_bits() == 0),
            "empty range wrote"
        );
        cast_rows(s, &kernel, pointer(&source, s), pointer(&target, s), 16)?;
        ensure!(
            s.clone_dtoh(&target)?
                .iter()
                .zip(&values)
                .all(|(got, &v)| got.to_bits() == bf16::from_f32(v).to_bits()),
            "cast differs from BF16 rounding"
        );
        Ok(())
    }

    /// Rows 4, k 16, n 8. `x` has 8 rows. Element `(r, c)` is `((r * 16 + c) * 3 + 1) % 17`.
    /// Weight `(i, j)` is `((i * 8 + j) * 5 + 1) % 13`. Bias `i` is `i + 1`.
    #[test]
    #[ignore = "GPU test"]
    fn projection_offset_matches_a_copied_window() -> Result<()> {
        let rows = 4usize;
        let k = 16usize;
        let n = 8usize;
        let mut x = vec![0.0f32; 8 * k];
        for r in 0..8 {
            for c in 0..k {
                x[r * k + c] = (((r * 16 + c) * 3 + 1) % 17) as f32;
            }
        }
        let mut weight = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                weight[i * k + j] = (((i * 8 + j) * 5 + 1) % 13) as f32;
            }
        }
        let bias_row: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let device = Device::new(0)?;
        let s = &device.stream;
        device.finish_upload()?;
        let ops = Ops::new(&device.ctx)?;
        let blas = open_blas(
            &device,
            &ops,
            &std::env::temp_dir(),
            Rc::new(Closure::new(true, "projection offset".into())),
        )?;
        let bias = cubin::module(s.context(), "vision_ops")?.load_function("vision_bias_f32")?;
        let x_d = s.clone_htod(&x)?;
        let stored = [s.clone_htod(&weight)?, s.clone_htod(&bias_row)?];
        let y0 = s.alloc_zeros::<f32>(rows * n)?;
        let y64 = s.alloc_zeros::<f32>(rows * n)?;
        projection(s, &blas, &bias, &stored, (&x_d, 0), (&y0, rows))?;
        projection(s, &blas, &bias, &stored, (&x_d, 64), (&y64, rows))?;
        s.synchronize()?;
        let at_zero = s.clone_dtoh(&y0)?;
        let at_sixty_four = s.clone_dtoh(&y64)?;
        let offset_diff = at_zero
            .iter()
            .zip(&at_sixty_four)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("PROJECTION_OFFSET differ={offset_diff}");
        ensure!(offset_diff > 0, "projection offset 0 matched offset 64");
        let copied = x[64..64 + rows * k].to_vec();
        let copied_d = s.clone_htod(&copied)?;
        let y_copy = s.alloc_zeros::<f32>(rows * n)?;
        projection(s, &blas, &bias, &stored, (&copied_d, 0), (&y_copy, rows))?;
        s.synchronize()?;
        let at_copy = s.clone_dtoh(&y_copy)?;
        let copy_diff = at_copy
            .iter()
            .zip(&at_sixty_four)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("PROJECTION_OFFSET_COPY differ={copy_diff}");
        ensure!(copy_diff == 0, "copied window differed in {copy_diff} bits");
        Ok(())
    }

    fn decoded(file: &SafeTensors<'_>, name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
        let view = file.tensor(name).with_context(|| format!("lacks {name}"))?;
        let values = match view.dtype() {
            Dtype::BF16 => view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f64::from(bf16::from_le_bytes(*b).to_f32()))
                .collect(),
            Dtype::F32 => view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f64::from(f32::from_le_bytes(*b)))
                .collect(),
            Dtype::I64 => view
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f64)
                .collect(),
            other => anyhow::bail!("{name}: unsupported dtype {other:?}"),
        };
        Ok((values, view.shape().to_vec()))
    }

    /// `[max_abs, rel_rms, cosine]` of `candidate` against `reference`.
    fn distance(candidate: &[f64], reference: &[f64]) -> Result<[f64; 3]> {
        let measured = crate::verify::metrics("h3.distance", candidate, reference)?;
        Ok([measured.max_abs, measured.rel_rmse, measured.cosine])
    }

    fn within(d: [f64; 3], threshold: &serde_json::Value) -> Result<bool> {
        let get = |key: &str| {
            threshold[key]
                .as_f64()
                .with_context(|| format!("threshold lacks {key}"))
        };
        Ok(d[0] <= get("max_abs")? && d[1] <= get("rel_rms")? && d[2] >= get("cosine_min")?)
    }

    fn n6_require_finite(number: f64, key: &str) -> Result<()> {
        ensure!(number.is_finite(), "Runner FAIL non-finite {key}");
        Ok(())
    }

    fn n6_finite(value: &serde_json::Value, key: &str) -> Result<f64> {
        let number = value
            .get(key)
            .with_context(|| format!("Runner FAIL missing {key}"))?
            .as_f64()
            .with_context(|| format!("Runner FAIL {key} not a number"))?;
        n6_require_finite(number, key)?;
        Ok(number)
    }

    fn n6_measured_arms(cell: &serde_json::Value, name: &str) -> Result<()> {
        let measured = cell
            .get("measured")
            .with_context(|| format!("Runner FAIL missing {name} measured"))?;
        for arm in ["bf16", "bf16strict"] {
            ensure!(
                measured.get(arm).is_some_and(|value| !value.is_null()),
                "Runner FAIL unqualified cell {name} missing {arm}"
            );
        }
        Ok(())
    }

    fn admit_n6_thresholds(frozen: &serde_json::Value) -> Result<()> {
        let thresholds = frozen
            .get("thresholds")
            .context("Runner FAIL missing thresholds")?;
        for cell_name in ["A", "D", "E"] {
            for output in ["merged", "deepstack0", "deepstack1", "deepstack2"] {
                let name = format!("{cell_name}.vision.{output}");
                let cell = thresholds
                    .get(&name)
                    .with_context(|| format!("Runner FAIL missing {name}"))?;
                for key in ["max_abs", "rel_rms", "cosine_min"] {
                    n6_finite(cell, key)?;
                }
                n6_measured_arms(cell, &name)?;
                ensure!(
                    cell.get("gating").and_then(|value| value.as_bool()) == Some(true),
                    "Runner FAIL unqualified cell {name}"
                );
            }
            for kind in ["text", "composition"] {
                for tap in TEXT_TAPS {
                    let name = format!("{cell_name}.{kind}.hidden_states_{tap}");
                    let cell = thresholds
                        .get(&name)
                        .with_context(|| format!("Runner FAIL missing {name}"))?;
                    n6_finite(cell, "median_norm_fp32")?;
                    n6_finite(cell, "sink_factor")?;
                    for key in ["I", "U"] {
                        ensure!(
                            cell.get(key).and_then(|value| value.as_array()).is_some(),
                            "Runner FAIL missing {name} {key}"
                        );
                    }
                    let aggregate = cell
                        .get("aggregate")
                        .with_context(|| format!("Runner FAIL missing {name} aggregate"))?;
                    for key in ["max_abs", "rel_rms", "cosine_min"] {
                        n6_finite(aggregate, key)?;
                    }
                    n6_measured_arms(cell, &name)?;
                    ensure!(
                        cell.get("gating").and_then(|value| value.as_bool()) == Some(tap == 50),
                        "Runner FAIL unqualified cell {name}"
                    );
                }
            }
        }
        Ok(())
    }

    fn admit_tensor_names(file: &SafeTensors<'_>, names: &[&str]) -> Result<()> {
        for name in names {
            file.tensor(name)
                .with_context(|| format!("Runner FAIL missing tensor {name}"))?;
        }
        Ok(())
    }

    fn admit_conditioner_tensors(path: &Path, names: &[&str]) -> Result<()> {
        let map = mapped(path)?;
        admit_tensor_names(&SafeTensors::deserialize(&map)?, names)
    }

    fn conditioner_named_f64(path: &Path, names: &[&str]) -> Result<Vec<Vec<f64>>> {
        admit_conditioner_tensors(path, names)?;
        let map = mapped(path)?;
        let file = SafeTensors::deserialize(&map)?;
        names
            .iter()
            .map(|name| Ok(decoded(&file, name)?.0))
            .collect()
    }

    fn conditioner_text_references(path: &Path) -> Result<Vec<Vec<f64>>> {
        let names: Vec<String> = TEXT_TAPS
            .iter()
            .map(|tap| format!("text.hidden_states_{tap}"))
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        conditioner_named_f64(path, &refs)
    }

    const VISION_TENSORS: [&str; 4] = [
        "vision.merged",
        "vision.deepstack0",
        "vision.deepstack1",
        "vision.deepstack2",
    ];

    fn n6_qualified_thresholds() -> serde_json::Value {
        let mut thresholds = serde_json::Map::new();
        let vision = serde_json::json!({
            "max_abs": 1.0,
            "rel_rms": 0.1,
            "cosine_min": 0.9,
            "measured": {"bf16": {}, "bf16strict": {}},
            "gating": true
        });
        for cell_name in ["A", "D", "E"] {
            for output in ["merged", "deepstack0", "deepstack1", "deepstack2"] {
                thresholds.insert(format!("{cell_name}.vision.{output}"), vision.clone());
            }
            for kind in ["text", "composition"] {
                for tap in TEXT_TAPS {
                    thresholds.insert(
                        format!("{cell_name}.{kind}.hidden_states_{tap}"),
                        serde_json::json!({
                            "median_norm_fp32": 1.0,
                            "sink_factor": 5.0,
                            "I": [],
                            "U": [],
                            "aggregate": {"max_abs": 1.0, "rel_rms": 0.1, "cosine_min": 0.9},
                            "measured": {"bf16": {}, "bf16strict": {}},
                            "gating": tap == 50
                        }),
                    );
                }
            }
        }
        serde_json::json!({"thresholds": thresholds})
    }

    #[test]
    fn h3_n6_admission_rejects_missing_threshold_field() -> Result<()> {
        admit_n6_thresholds(&n6_qualified_thresholds())?;
        let mut missing = n6_qualified_thresholds();
        missing["thresholds"]["A.vision.merged"]
            .as_object_mut()
            .context("vision cell")?
            .remove("max_abs");
        ensure!(admit_n6_thresholds(&missing).is_err());
        ensure!(n6_finite(&serde_json::json!({"max_abs": null}), "max_abs").is_err());
        Ok(())
    }

    #[test]
    fn h3_n6_admission_rejects_nonfinite_floor() -> Result<()> {
        ensure!(n6_finite(&serde_json::json!({"max_abs": 1.0}), "max_abs")? == 1.0);
        ensure!(n6_require_finite(f64::NAN, "max_abs").is_err());
        ensure!(n6_require_finite(f64::INFINITY, "max_abs").is_err());
        Ok(())
    }

    #[test]
    fn h3_n6_admission_rejects_missing_tensor() -> Result<()> {
        use safetensors::tensor::TensorView;
        let bytes = [0u8; 4];
        let present = TensorView::new(Dtype::F32, vec![1], &bytes)?;
        let buffer = safetensors::serialize([("present", present)], None)?;
        let file = SafeTensors::deserialize(&buffer)?;
        admit_tensor_names(&file, &["present"])?;
        ensure!(admit_tensor_names(&file, &["vision.merged"]).is_err());
        Ok(())
    }

    #[test]
    fn h3_n6_admission_rejects_unqualified_cell() -> Result<()> {
        let mut gating = n6_qualified_thresholds();
        gating["thresholds"]["A.vision.merged"]["gating"] = serde_json::json!(false);
        ensure!(admit_n6_thresholds(&gating).is_err());
        let mut tap = n6_qualified_thresholds();
        tap["thresholds"]["A.text.hidden_states_50"]["gating"] = serde_json::json!(false);
        ensure!(admit_n6_thresholds(&tap).is_err());
        let mut arm = n6_qualified_thresholds();
        arm["thresholds"]["A.text.hidden_states_50"]["measured"]
            .as_object_mut()
            .context("measured")?
            .remove("bf16");
        ensure!(admit_n6_thresholds(&arm).is_err());
        Ok(())
    }

    fn mapped(path: &Path) -> Result<memmap2::Mmap> {
        let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(unsafe { memmap2::Mmap::map(&file)? })
    }

    /// The engine's own layout for a Stage 0 run directory: geometry from `meta.json`, text tags from the capture.
    fn capture_layout(spec: &H3Spec, capture: &Path) -> Result<PackedLayout> {
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(capture.join("meta.json"))?)?;
        let size = |v: &serde_json::Value| v.as_u64().map(|n| n as usize).context("meta size");
        let geometry = Geometry::resolve(
            spec,
            size(&meta["resolved"]["height"])?,
            size(&meta["resolved"]["width"])?,
            size(&meta["cell"]["frames"])?,
        )?;
        let layout_map = mapped(&capture.join("layout.safetensors"))?;
        let layout_file = SafeTensors::deserialize(&layout_map)?;
        let (tags, _) = decoded(&layout_file, "token_tags")?;
        let (text, _) = decoded(&layout_file, "text_indices")?;
        let text_tags: Vec<u8> = text.iter().map(|&row| tags[row as usize] as u8).collect();
        let anchors: &[KeyframeAnchor] = if meta["cell"]["last_image"].is_null() {
            &[KeyframeAnchor::First]
        } else {
            &[KeyframeAnchor::First, KeyframeAnchor::Last]
        };
        PackedLayout::build(spec, &geometry, &text_tags, anchors)
    }

    /// AdaLN projections read once from disk, eager. Candidate rows of every block and step against the FP32 replay
    /// on the addressed rows, within the frozen `A.adaln.*` thresholds; control: the same rows scored against the
    /// replay's video/audio-swapped rows must leave a threshold.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_CAPTURE (BF16 run, --capture-adaln), FF_H3_REPLAY (its FP32 replay with the adaln stage), FF_H3_THRESHOLDS (frozen JSON)"]
    fn adaln_table_matches_the_fp32_replay() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let thresholds = &thresholds["thresholds"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_manifest(&spec)?;

        let layout = capture_layout(&spec, &capture)?;
        let plan: Vec<RowTimesteps> =
            row_timestep_plan(&layout, &Schedule::new(9, 6.0)?, &Schedule::new(9, 3.0)?)?;
        let adaln = AdalnPlan::build(&layout, &plan)?;
        let replay_map = mapped(&path("FF_H3_REPLAY")?.join("replay.safetensors"))?;
        let replay = SafeTensors::deserialize(&replay_map)?;
        let width = 6 * spec.hidden;
        let norm = 2 * spec.hidden;
        for (k, step) in plan.iter().enumerate() {
            for layer in 0..spec.layers {
                let name = format!("adaln.block{layer}.step{k}");
                let view = replay
                    .tensor(&name)
                    .with_context(|| format!("lacks {name}"))?;
                ensure!(
                    matches!(view.dtype(), Dtype::BF16 | Dtype::F32 | Dtype::I64)
                        && view.shape() == [step.timesteps.len() * 3, width],
                    "{name} shape {:?} {:?}",
                    view.dtype(),
                    view.shape()
                );
            }
            let name = format!("adaln.norm_out.step{k}");
            let view = replay
                .tensor(&name)
                .with_context(|| format!("lacks {name}"))?;
            ensure!(
                matches!(view.dtype(), Dtype::BF16 | Dtype::F32 | Dtype::I64),
                "{name}: unsupported dtype {:?}",
                view.dtype()
            );
        }

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-adaln-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 AdaLN gate".into())),
        )?;
        let table = AdalnTable::build(&device, &mut blas, &transformer, &manifest, &spec, &adaln)?;
        let blocks: Vec<f64> = device
            .stream
            .clone_dtoh(&table.blocks)?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect();
        let norm_out: Vec<f64> = device
            .stream
            .clone_dtoh(&table.norm_out)?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect();
        fs::remove_dir(&runtime)?;

        let global = |t: f32| adaln.timesteps.partition_point(|&v| v < t);
        let (mut checked, mut worst, mut control_failed) = (0usize, [0f64, 0f64, 1f64], false);
        for (k, step) in plan.iter().enumerate() {
            let mut rows: Vec<u32> = step
                .indices
                .iter()
                .zip(&layout.token_tags)
                .map(|(&t, &tag)| t * MODALITIES + u32::from(tag))
                .collect();
            rows.sort_unstable();
            rows.dedup();
            let pair = |row: u32| {
                let t = global(step.timesteps[(row / MODALITIES) as usize]) as u32;
                let key = (t, (row % MODALITIES) as u8);
                adaln.pairs.binary_search(&key).ok()
            };
            for layer in 0..spec.layers {
                let name = format!("adaln.block{layer}.step{k}");
                let (reference, _) = decoded(&replay, &name)?;
                let (mut candidate, mut expected, mut swapped) = (vec![], vec![], vec![]);
                for &row in &rows {
                    let p = pair(row).with_context(|| format!("{name} row {row} has no pair"))?;
                    let c = &blocks[(layer * adaln.pairs.len() + p) * width..][..width];
                    candidate.extend_from_slice(c);
                    expected.extend_from_slice(&reference[row as usize * width..][..width]);
                    let tag = row % MODALITIES;
                    if tag != 1 {
                        let other = row - tag + (2 - tag);
                        swapped.push((
                            c.to_vec(),
                            reference[other as usize * width..][..width].to_vec(),
                        ));
                    }
                }
                let threshold = &thresholds[format!("A.{name}")];
                let d = distance(&candidate, &expected)?;
                ensure!(within(d, threshold)?, "{name} {d:?} outside {threshold}");
                let (c, r): (Vec<f64>, Vec<f64>) = swapped
                    .iter()
                    .flat_map(|(c, r)| c.iter().copied().zip(r.iter().copied()))
                    .unzip();
                control_failed |= !within(distance(&c, &r)?, threshold)?;
                worst = [worst[0].max(d[0]), worst[1].max(d[1]), worst[2].min(d[2])];
                checked += 1;
            }
            let name = format!("adaln.norm_out.step{k}");
            let (reference, _) = decoded(&replay, &name)?;
            let mut used: Vec<u32> = step.indices.clone();
            used.sort_unstable();
            used.dedup();
            let (mut candidate, mut expected) = (vec![], vec![]);
            for &t in &used {
                let g = global(step.timesteps[t as usize]);
                candidate.extend_from_slice(&norm_out[g * norm..][..norm]);
                expected.extend_from_slice(&reference[t as usize * norm..][..norm]);
            }
            let d = distance(&candidate, &expected)?;
            let threshold = &thresholds[format!("A.{name}")];
            ensure!(within(d, threshold)?, "{name} {d:?} outside {threshold}");
            checked += 1;
        }
        ensure!(
            control_failed,
            "video/audio tag-swap control stayed within thresholds"
        );
        println!(
            "H3_ADALN_GATE PASS tables={checked} worst_max_abs={:.6e} worst_rel_rms={:.6e} worst_cosine={:.9} control=FAIL_AS_REQUIRED",
            worst[0], worst[1], worst[2]
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE"]
    fn ffn_lora_is_row_count_invariant() -> Result<()> {
        use crate::backend::setup::LinearShape;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let spec = H3Spec::read(&path("FF_MODEL")?)?;
        let file = path("FF_H3_LORA")?;
        let dir = file.parent().context("LoRA directory")?;
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .context("LoRA stem")?;
        let (manifest, adapter) =
            MappedCheckpoint::open_component(dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let captured = fs::read(path("FF_H3_CAPTURE")?.join("lora_a_up_input.bf16"))?;
        ensure!(
            captured.len() == 1511 * spec.hidden * 2,
            "captured input must be [1511,hidden] BF16"
        );
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let stream = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-row-gate-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let partitions: Vec<usize> = serde_json::from_slice(&fs::read(
            path("FF_H3_CAPTURE")?.join("row_partitions.json"),
        )?)?;
        ensure!(
            partitions.contains(&1008)
                && partitions.contains(&1511)
                && partitions.iter().all(|&n| n > 0 && n <= 60_431),
            "row partitions must cover the discriminator within production extents"
        );
        let total = *partitions.iter().max().context("row partitions empty")?;
        let mut source = Source::new(dir, &manifest);
        {
            let rows = 664;
            let weights = source.tensor::<bf16>(
                stream,
                "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_A.default.weight",
                adapter.rank * spec.hidden,
            )?;
            let bytes = fs::read(path("FF_H3_CAPTURE")?.join("refiner_a_up_input.bf16"))?;
            ensure!(
                bytes.len() == rows * spec.hidden * 2,
                "refiner common input must be [664,hidden] BF16"
            );
            let host: Vec<bf16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect();
            ensure!(
                host.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL nonfinite refiner input"
            );
            let input = stream.clone_htod(&host)?;
            let output = stream.alloc_zeros::<bf16>(rows * adapter.rank)?;
            let preservation = open_blas(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(true, "ordinary row preservation".into())),
            )?;
            preservation.prepare_fixed(rows as u32, adapter.rank as u32, spec.hidden as u32)?;
            let shape = LinearShape {
                rows,
                output: adapter.rank,
                input: spec.hidden,
            };
            let ordinary = || -> Result<Vec<bf16>> {
                preservation.linear_typed(
                    crate::op_contract::ExecutionPhase::Prefill,
                    crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(
                        &weights,
                        stream,
                        adapter.rank,
                        spec.hidden,
                    )?),
                    matrix(&input, stream, rows, spec.hidden)?,
                    &mut TensorMut(matrix(&output, stream, rows, adapter.rank)?),
                )?;
                stream
                    .clone_dtoh(&output)
                    .context("copy ordinary low-rank output to host")
            };
            let expected = ordinary()?;
            let before = preservation.bound_plan_receipt()?;
            preservation.bind_row_family(LinearShape {
                rows: FFN_CHUNK_ROWS,
                ..shape
            })?;
            preservation.bind_workspace(preservation.fixed_workspace(shape, true)?)?;
            for executed_family in [false, true] {
                if executed_family {
                    preservation.linear_row_family(
                        matrix(&weights, stream, adapter.rank, spec.hidden)?,
                        matrix(&input, stream, rows, spec.hidden)?,
                        &mut TensorMut(matrix(&output, stream, rows, adapter.rank)?),
                        1.0,
                        0.0,
                    )?;
                }
                let actual = ordinary()?;
                ensure!(
                    actual.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL nonfinite ordinary output"
                );
                let differences = expected
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!(
                    "H3_ORDINARY_PRESERVATION rows={rows} elements={} family_executed={executed_family} bit_differences={differences}",
                    expected.len()
                );
                ensure!(
                    differences == 0,
                    "DEFECT ordinary route changed after row-family binding"
                );
            }
            let after = preservation.bound_plan_receipt()?;
            let ordinary_after: Vec<_> = after["lt"]
                .as_array()
                .context("plan list missing")?
                .iter()
                .filter(|p| p["row_family"] == false)
                .cloned()
                .collect();
            ensure!(
                before["lt"] == serde_json::to_value(ordinary_after)?,
                "DEFECT ordinary plan identity changed"
            );
            println!(
                "H3_ORDINARY_PRESERVATION_GATE PASS before_after_binding=true after_family_execution=true bit_difference_limit=0"
            );
        }
        let mut cells = 0;
        for (name, output, input, accumulate) in [
            ("ff.net.0.proj.lora_A", adapter.rank, spec.hidden, false),
            ("ff.net.2.lora_A", adapter.rank, spec.ffn, false),
            ("ff.net.0.proj.lora_B", 2 * spec.ffn, adapter.rank, true),
            ("ff.net.2.lora_B", spec.hidden, adapter.rank, true),
        ] {
            let weight = source.tensor::<bf16>(
                stream,
                &format!("transformer_blocks.0.{name}.default.weight"),
                output * input,
            )?;
            let host: Vec<bf16> = if name == "ff.net.0.proj.lora_A" {
                captured
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .cycle()
                    .take(total * input)
                    .collect()
            } else {
                (0..total * input)
                    .map(|i| {
                        bf16::from_f32((((i * 37 + i / input * 11) % 251) as f32 - 125.0) / 127.0)
                    })
                    .collect()
            };
            let x = stream.clone_htod(&host)?;
            let initial: Vec<bf16> = (0..total * output)
                .map(|i| bf16::from_f32(((i * 19 % 257) as f32 - 128.0) / 131.0))
                .collect();
            let make_blas = || {
                open_blas(
                    &device,
                    &ops,
                    &runtime,
                    Rc::new(Closure::new(true, "FFN row invariance".into())),
                )
            };
            let fixed = make_blas()?;
            fixed.bind_row_family(LinearShape {
                rows: FFN_CHUNK_ROWS,
                output,
                input,
            })?;
            let heuristic = make_blas()?;
            let shape = |rows| LinearShape {
                rows,
                output,
                input,
            };
            let mut workspace = 0;
            for &chunk in &partitions {
                for rows in [chunk, total % chunk].into_iter().filter(|&r| r > 0) {
                    workspace = workspace.max(fixed.fixed_workspace(shape(rows), true)?);
                    heuristic.prepare_fixed(
                        u32::try_from(rows)?,
                        u32::try_from(output)?,
                        u32::try_from(input)?,
                    )?;
                }
            }
            let composition = if accumulate {
                let width = if output == 2 * spec.ffn {
                    spec.hidden
                } else {
                    spec.ffn
                };
                let key = name.replace("lora_B", "lora_A");
                let a = source.tensor::<bf16>(
                    stream,
                    &format!("transformer_blocks.0.{key}.default.weight"),
                    input * width,
                )?;
                let values: Vec<bf16> = (0..total * width)
                    .map(|i| {
                        bf16::from_f32((((i * 37 + i / width * 11) % 251) as f32 - 125.0) / 127.0)
                    })
                    .collect();
                let raw = stream.clone_htod(&values)?;
                fixed.bind_row_family(LinearShape {
                    rows: FFN_CHUNK_ROWS,
                    output: input,
                    input: width,
                })?;
                for &chunk in &partitions {
                    for rows in [chunk, total % chunk].into_iter().filter(|&r| r > 0) {
                        workspace = workspace.max(fixed.fixed_workspace(
                            LinearShape {
                                rows,
                                output: input,
                                input: width,
                            },
                            true,
                        )?);
                    }
                }
                Some((a, raw, width))
            } else {
                None
            };
            fixed.bind_workspace(workspace)?;
            let run = |blas: &Blas,
                       chunk: usize,
                       graph: bool,
                       fixed_b: bool,
                       composed: bool|
             -> Result<Vec<bf16>> {
                let y = stream.clone_htod(&initial)?;
                let intermediate = stream.alloc_zeros::<bf16>(total * input)?;
                let launch = || -> Result<()> {
                    for start in (0..total).step_by(chunk) {
                        let rows = chunk.min(total - start);
                        let w = matrix(&weight, stream, output, input)?;
                        let x = if composed {
                            let (a, raw, width) =
                                composition.as_ref().context("composed LoRA-A missing")?;
                            blas.linear_row_family(
                                matrix(a, stream, input, *width)?,
                                rows_view(raw, stream, start * width, rows, *width, *width)?,
                                &mut TensorMut(rows_view(
                                    &intermediate,
                                    stream,
                                    start * input,
                                    rows,
                                    input,
                                    input,
                                )?),
                                1.0,
                                0.0,
                            )?;
                            rows_view(&intermediate, stream, start * input, rows, input, input)?
                        } else {
                            rows_view(&x, stream, start * input, rows, input, input)?
                        };
                        let mut y =
                            TensorMut(rows_view(&y, stream, start * output, rows, output, output)?);
                        if accumulate {
                            if fixed_b {
                                blas.linear_row_family(w, x, &mut y, adapter.scale, 1.0)?;
                            } else {
                                blas.linear_accumulate(w, x, &mut y, adapter.scale)?;
                            }
                        } else if fixed_b {
                            blas.linear_row_family(w, x, &mut y, 1.0, 0.0)?;
                        } else {
                            blas.linear_typed(
                                crate::op_contract::ExecutionPhase::Prefill,
                                crate::backend::cuda::linear::LinearWeightRef::Bf16(w),
                                x,
                                &mut y,
                            )?;
                        }
                    }
                    Ok(())
                };
                if graph {
                    capture_graph(stream, false, launch)?.launch()?;
                } else {
                    launch()?;
                }
                let actual = stream.clone_dtoh(&y)?;
                ensure!(
                    actual.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL nonfinite LoRA row output"
                );
                Ok(actual)
            };
            if name == "ff.net.0.proj.lora_A" {
                let a = run(&heuristic, 1008, false, false, false)?;
                let b = run(&heuristic, 1511, false, false, false)?;
                let differences = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!("H3_ROW_CONTROL heuristic_1008_vs_1511 bit_differences={differences}");
                ensure!(
                    differences > 0,
                    "row-count discriminator did not detect old heuristic"
                );
            }
            for composed in [false, true].into_iter().filter(|&c| !c || accumulate) {
                let expected = run(&fixed, total, false, true, composed)?;
                for &chunk in &partitions {
                    for graph in [false, true] {
                        let actual = run(&fixed, chunk, graph, true, composed)?;
                        let differences = actual
                            .iter()
                            .zip(&expected)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        println!(
                            "H3_ROW_CELL operation={name} composed={composed} chunk={chunk} graph={graph} bit_differences={differences} elements={}",
                            actual.len()
                        );
                        ensure!(
                            differences == 0,
                            "DEFECT row-count invariance {name} chunk={chunk} graph={graph}"
                        );
                        cells += 1;
                    }
                }
            }
        }
        fs::remove_dir_all(runtime)?;
        println!("H3_ROW_GATE PASS cells={cells} bit_difference_limit=0 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_CAPTURE"]
    fn base_ffn_is_row_count_invariant() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let model = std::env::var_os("FF_MODEL")
            .map(PathBuf::from)
            .context("FF_MODEL missing")?;
        let capture = std::env::var_os("FF_H3_CAPTURE")
            .map(PathBuf::from)
            .context("FF_H3_CAPTURE missing")?;
        let partitions: Vec<usize> =
            serde_json::from_slice(&fs::read(capture.join("row_partitions.json"))?)?;
        ensure!(
            partitions.contains(&1008)
                && partitions.contains(&1511)
                && partitions.iter().all(|&n| n > 0 && n <= 60_431),
            "base row partitions outside production extents"
        );
        let spec = H3Spec::read(&model)?;
        let dir = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let stream = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-base-row-gate-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "base row invariance".into())),
        )?;
        let mut source = Source::new(&dir, &manifest);
        let mut defects = 0;
        let mut cells = 0;
        for layer in [0, spec.layers / 2] {
            for (name, output, input) in [
                ("ff.net.0.proj", 2 * spec.ffn, spec.hidden),
                ("ff.net.2", spec.hidden, spec.ffn),
            ] {
                let weight = source.tensor::<bf16>(
                    stream,
                    &format!("transformer_blocks.{layer}.{name}.weight"),
                    output * input,
                )?;
                for rows in [40_045usize, 60_431] {
                    for &other in &partitions {
                        for chunk in [1008, other] {
                            for n in [chunk, rows % chunk].into_iter().filter(|&n| n > 0) {
                                blas.prepare_fixed(
                                    u32::try_from(n)?,
                                    u32::try_from(output)?,
                                    u32::try_from(input)?,
                                )?;
                            }
                        }
                        let run = |start: usize, n: usize| -> Result<Vec<bf16>> {
                            let host: Vec<bf16> = (start * input..(start + n) * input)
                                .map(|i| {
                                    bf16::from_f32(
                                        (((i * 37 + i / input * 11) % 251) as f32 - 125.0) / 127.0,
                                    )
                                })
                                .collect();
                            let x = stream.clone_htod(&host)?;
                            let y = stream.alloc_zeros::<bf16>(n * output)?;
                            blas.linear_typed(
                                crate::op_contract::ExecutionPhase::Prefill,
                                crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(
                                    &weight, stream, output, input,
                                )?),
                                matrix(&x, stream, n, input)?,
                                &mut TensorMut(matrix(&y, stream, n, output)?),
                            )?;
                            let actual = stream.clone_dtoh(&y)?;
                            ensure!(
                                actual.iter().all(|v| v.is_finite()),
                                "RUNNER_FAIL nonfinite base row output"
                            );
                            Ok(actual)
                        };
                        let mut buffers = [Vec::new(), Vec::new()];
                        let mut begin = [0usize; 2];
                        let mut end = [0usize; 2];
                        let mut offset = 0;
                        let mut differences = 0;
                        while offset < rows {
                            for (arm, chunk) in [1008, other].into_iter().enumerate() {
                                if offset == end[arm] {
                                    begin[arm] = offset;
                                    end[arm] = (offset + chunk).min(rows);
                                    buffers[arm] = run(offset + 17, end[arm] - offset)?;
                                }
                            }
                            let stop = end[0].min(end[1]);
                            let a = &buffers[0]
                                [(offset - begin[0]) * output..(stop - begin[0]) * output];
                            let b = &buffers[1]
                                [(offset - begin[1]) * output..(stop - begin[1]) * output];
                            differences += a
                                .iter()
                                .zip(b)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            offset = stop;
                        }
                        println!(
                            "H3_BASE_ROW_CELL layer={layer} operation={name} rows={rows} partition=1008_vs_{other} global_row_offset=17 bit_differences={differences} elements={}",
                            rows * output
                        );
                        defects += usize::from(differences > 0);
                        cells += 1;
                    }
                }
            }
        }
        fs::remove_dir_all(runtime)?;
        println!("H3_BASE_ROW_GATE cells={cells} defects={defects} bit_difference_limit=0");
        ensure!(
            defects == 0,
            "DEFECT base FFN row invariance cells={defects}"
        );
        Ok(())
    }

    /// Owned and borrowed FFN storage use identical plans, including poisoned scratch and subsequent blocks.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE, FF_H3_REPLAY, FF_H3_THRESHOLDS"]
    fn ffn_alias_matches_owned_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let replay = path("FF_H3_REPLAY")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        ensure!(frozen["bit_difference_limit"] == 0, "alias gate bound");
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_path = path("FF_H3_LORA")?;
        let lora_dir = lora_path.parent().context("LoRA directory")?;
        let stem = lora_path
            .file_stem()
            .and_then(|v| v.to_str())
            .context("LoRA stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let bytes = fs::read(capture.join("lora_a_up_input.bf16"))?;
        ensure!(
            bytes.len() == 1511 * spec.hidden * 2,
            "common input [1511,hidden] BF16"
        );
        let common: Vec<bf16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        ensure!(
            common.iter().all(|v| v.is_finite()),
            "RUNNER_FAIL nonfinite common input"
        );
        let adaln_map = mapped(&replay.join("A-bf16/adaln.safetensors"))?;
        let adaln = SafeTensors::deserialize(&adaln_map)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights: Vec<BlockWeights> = (0..2)
            .map(|layer| {
                BlockWeights::resident::<true>(
                    s,
                    &spec,
                    adapter.rank,
                    &format!("transformer_blocks.{layer}"),
                    &mut Source::new(&transformer, &manifest),
                    Some(&mut Source::new(lora_dir, &lora_manifest)),
                )
            })
            .collect::<Result<_>>()?;
        let mut cells = 0;
        let mut defects = 0;
        let mut control_bits = 0;
        for case in ["A", "P"] {
            let layout = capture_layout(&spec, &replay.join(format!("{case}-bf16")))?;
            let rows = layout.rows();
            let chunk = frozen["runs"][format!("{case}-derived")]["plan"]["main"]["chunk"]
                .as_u64()
                .context("frozen main chunk")? as usize;
            let input: Vec<bf16> = (0..rows * spec.hidden)
                .map(|i| common[((i / spec.hidden + 17) % 1511) * spec.hidden + i % spec.hidden])
                .collect();
            let runtime =
                std::env::temp_dir().join(format!("ff-h3-alias-{}-{case}", std::process::id()));
            fs::create_dir(&runtime)?;
            let blas = open_blas(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(true, "H3 alias identity".into())),
            )?;
            let mut owned = BlockWork::new(s, &spec, adapter.rank, rows)?;
            let mut borrowed = BlockWork::new(s, &spec, adapter.rank, rows)?;
            owned.prepare::<true>(&blas, &spec)?;
            borrowed.prepare::<true>(&blas, &spec)?;
            for (output, input) in [
                (adapter.rank, spec.hidden),
                (adapter.rank, spec.ffn),
                (2 * spec.ffn, adapter.rank),
                (spec.hidden, adapter.rank),
            ] {
                blas.bind_row_family(crate::backend::setup::LinearShape {
                    rows: FFN_CHUNK_ROWS,
                    output,
                    input,
                })?;
            }
            let plan = owned.ffn_cost::<true>(&blas, &spec, chunk, false)?;
            blas.bind_workspace(plan.workspace)?;
            let mut x = s.clone_htod(&input)?;
            owned.bind_ffn::<true>(s, &blas, &spec, plan, &x)?;
            borrowed.bind_ffn::<true>(
                s,
                &blas,
                &spec,
                FfnPlan {
                    borrowed: true,
                    allocation: 0,
                    ..plan
                },
                &x,
            )?;
            ensure!(
                borrowed.hidden.ptr == pointer(&borrowed.q_raw, s)
                    && borrowed.gated.ptr == pointer(&borrowed.k_raw, s)
                    && borrowed.hidden.span_bytes + 256 <= (borrowed.q_raw.len() * 2) as u64,
                "borrowed address/control capacity"
            );
            let receipt = blas.bound_plan_receipt()?;
            let rotary = s.clone_htod(&crate::minimax_h3::rotary_table(&spec, &layout))?;
            let tables: Vec<_> = (0..2)
                .map(|layer| -> Result<_> {
                    let view = adaln.tensor(&format!("block{layer}.step0"))?;
                    ensure!(
                        view.dtype() == Dtype::BF16 && view.shape()[1] == 6 * spec.hidden,
                        "modulation layout"
                    );
                    let values: Vec<bf16> = view
                        .data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16::from_le_bytes(*b))
                        .collect();
                    Ok((s.clone_htod(&values)?, u32::try_from(view.shape()[0])?))
                })
                .collect::<Result<_>>()?;
            ensure!(
                tables[0].1 == tables[1].1 && tables[0].1 > 0,
                "modulation entries"
            );
            let index = s.clone_htod(
                &(0..rows)
                    .map(|r| (r % tables[0].1 as usize) as u32)
                    .collect::<Vec<_>>(),
            )?;
            let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
                .with_full_single_probability()?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: adapter.scale,
            };
            let mut run = |work: &mut BlockWork,
                           graph: bool,
                           full: bool,
                           shifted: bool|
             -> Result<Vec<bf16>> {
                s.memcpy_htod(&input, &mut x)?;
                s.memcpy_htod(&input, &mut work.normed)?;
                let poison = if work.ffn_owned.is_some() { 0x3f } else { 0xbf };
                unsafe {
                    crate::backend::cuda::copy::memset_bytes(
                        work.hidden.ptr,
                        poison,
                        usize::try_from(work.hidden.span_bytes)?,
                        s,
                    )?;
                    crate::backend::cuda::copy::memset_bytes(
                        work.gated.ptr,
                        poison,
                        usize::try_from(work.gated.span_bytes)?,
                        s,
                    )?;
                }
                s.synchronize()?;
                let mut launch = || -> Result<()> {
                    if full {
                        for (layer, weight) in weights.iter().enumerate() {
                            let modulation = Modulation {
                                table: &tables[layer].0,
                                offset: 0,
                                entries: tables[layer].1,
                                index: &index,
                            };
                            block_forward::<true>(
                                &context,
                                &mut flash,
                                weight,
                                &Conditioning::Modulated {
                                    rotary: &rotary,
                                    modulation: &modulation,
                                },
                                &mut x,
                                work,
                            )?;
                        }
                    } else if shifted {
                        ffn_forward::<true, true>(&context, &weights[0], work, None)?;
                    } else {
                        ffn_forward::<false, true>(&context, &weights[0], work, None)?;
                    }
                    Ok(())
                };
                if graph {
                    let captured = capture_graph(s, false, launch)?;
                    captured.launch()?;
                    captured.launch()?;
                } else {
                    launch()?;
                    launch()?;
                }
                let output = s.clone_dtoh(if full { &x } else { &work.normed })?;
                ensure!(
                    output.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL nonfinite alias output"
                );
                Ok(output)
            };
            for full in [false, true] {
                let reference = run(&mut owned, false, full, false)?;
                for (borrow, graph) in [(false, true), (true, false), (true, true)] {
                    let actual = run(
                        if borrow { &mut borrowed } else { &mut owned },
                        graph,
                        full,
                        false,
                    )?;
                    let bits = reference
                        .iter()
                        .zip(&actual)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    cells += 1;
                    defects += usize::from(bits > 0);
                    println!(
                        "H3_ALIAS_CELL case={case} rows={rows} chunk={chunk} tail={} borrowed={borrow} graph={graph} next_block={full} replays=2 bit_differences={bits} limit=0",
                        rows % chunk
                    );
                }
                if !full && case == "A" {
                    let control = run(&mut borrowed, false, false, true)?;
                    control_bits = reference
                        .iter()
                        .zip(&control)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    println!(
                        "H3_ALIAS_CONTROL consumer_offset_bytes=256 bit_differences={control_bits}"
                    );
                }
            }
            ensure!(
                receipt == blas.bound_plan_receipt()?,
                "alias test changed bound algorithms"
            );
            s.synchronize()?;
            drop(blas);
            fs::remove_dir_all(runtime)?;
        }
        println!(
            "H3_ALIAS_GATE cells={cells} defects={defects} bit_difference_limit=0 control_bit_differences={control_bits}"
        );
        ensure!(
            control_bits > 0,
            "RUNNER_FAIL alias-address control undetected"
        );
        ensure!(defects == 0, "DEFECT FFN storage alias cells={defects}");
        Ok(())
    }

    /// Resident block 0, eager. The native block on the captured block-0 input with the captured BF16 modulation
    /// rows, against the FP32 replay of block 0 within the frozen `A.block0.output` threshold; control: LoRA scale 0
    /// must leave it.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE (BF16 run, --capture-block-io --capture-adaln), FF_H3_REPLAY (FP32 replay with the block0 stage), FF_H3_THRESHOLDS"]
    fn block0_matches_the_fp32_replay() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let lora_path = path("FF_H3_LORA")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let threshold = &thresholds["thresholds"]["A.block0.output"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let layout = capture_layout(&spec, &capture)?;
        let rows = layout.rows();
        let rotary_host = crate::minimax_h3::rotary_table(&spec, &layout);

        let io_map = mapped(&capture.join("block_io.safetensors"))?;
        let io = SafeTensors::deserialize(&io_map)?;
        let hidden = io.tensor("block0.hidden_states")?;
        ensure!(
            hidden.dtype() == Dtype::BF16 && hidden.shape() == [1, rows, spec.hidden],
            "block0.hidden_states {:?} {:?}",
            hidden.dtype(),
            hidden.shape()
        );
        let input: Vec<bf16> = hidden
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let (indices, _) = decoded(&io, "block0.adaln_indices")?;
        let index: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let (cos, _) = decoded(&io, "block0.rotary_cos")?;
        let (sin, _) = decoded(&io, "block0.rotary_sin")?;
        let half = 3 * spec.rope_freq_dim;
        let rotary_width = 2 * half;
        let mut rotary_differs = 0usize;
        for row in 0..rows {
            for j in 0..half {
                let table = &rotary_host[row * rotary_width..][..rotary_width];
                let reference_cos = bf16::from_f64(cos[row * rotary_width + j]);
                let reference_sin = bf16::from_f64(sin[row * rotary_width + j]);
                rotary_differs += usize::from(table[j] != reference_cos)
                    + usize::from(table[half + j] != reference_sin);
            }
        }
        let adaln_map = mapped(&capture.join("adaln.safetensors"))?;
        let adaln = SafeTensors::deserialize(&adaln_map)?;
        let rows_table = adaln.tensor("block0.step0")?;
        ensure!(rows_table.dtype() == Dtype::BF16, "block0.step0 dtype");
        let entries = u32::try_from(rows_table.shape()[0])?;
        let table_host: Vec<bf16> = rows_table
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let (
            final_index_host,
            final_hidden_host,
            final_entries,
            final_table_host,
            reference,
            captured,
        ) = {
            let step_map = mapped(&capture.join("step0.safetensors"))?;
            let step = SafeTensors::deserialize(&step_map)?;
            let timestep_indices = step.tensor("timestep_indices")?;
            ensure!(
                timestep_indices.dtype() == Dtype::I64 && timestep_indices.shape() == [rows],
                "captured final norm index extent"
            );
            let final_index_host: Vec<u32> = timestep_indices
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| u32::try_from(i64::from_le_bytes(*v)))
                .collect::<std::result::Result<_, _>>()?;
            let block49 = io.tensor("block49.output")?;
            ensure!(
                block49.dtype() == Dtype::BF16 && block49.shape() == [1, rows, spec.hidden],
                "captured final norm input extent"
            );
            let final_hidden_host = bf16_of(&io, "block49.output")?;
            let norm_out = adaln.tensor("norm_out.step0")?;
            ensure!(
                norm_out.dtype() == Dtype::BF16
                    && norm_out.shape().len() == 2
                    && norm_out.shape()[1] == 2 * spec.hidden,
                "captured final modulation extent"
            );
            let final_entries = u32::try_from(norm_out.shape()[0])?;
            let final_table_host = bf16_of(&adaln, "norm_out.step0")?;
            ensure!(
                final_index_host.iter().all(|&v| v < final_entries),
                "captured final norm index bound"
            );
            let replay_map = mapped(&path("FF_H3_REPLAY")?.join("replay.safetensors"))?;
            let replay = SafeTensors::deserialize(&replay_map)?;
            let (reference, _) = decoded(&replay, "block0.output")?;
            let (captured, _) = decoded(&io, "block0.output")?;
            (
                final_index_host,
                final_hidden_host,
                final_entries,
                final_table_host,
                reference,
                captured,
            )
        };

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-block0-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 block gate".into())),
        )?;
        let r = adapter.rank;
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::resident::<true>(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(&transformer, &manifest),
            Some(&mut Source::new(&lora_dir, &lora_manifest)),
        )?;
        let mut work = BlockWork::new(s, &spec, r, rows)?;
        work.prepare::<true>(&blas, &spec)?;
        let rotary = s.clone_htod(&rotary_host)?;
        let table = s.clone_htod(&table_host)?;
        let index = s.clone_htod(&index)?;
        let mut captured_norm_defects = 0;
        {
            let hidden = s.clone_htod(&input)?;
            let final_hidden = s.clone_htod(&final_hidden_host)?;
            let final_weight = Source::new(&transformer, &manifest).tensor::<bf16>(
                s,
                "norm_out.norm.weight",
                spec.hidden,
            )?;
            let final_table = s.clone_htod(&final_table_host)?;
            let final_index = s.clone_htod(&final_index_host)?;
            let reference = s.alloc_zeros::<bf16>(rows * spec.hidden)?;
            for (name, x, weight, params, indices, stride, shift, scale, entries, eps) in [
                (
                    "norm1",
                    &hidden,
                    weights.norm1.tensor,
                    &table,
                    &index,
                    6 * spec.hidden,
                    0,
                    spec.hidden,
                    entries,
                    spec.norm_eps,
                ),
                (
                    "norm2",
                    &hidden,
                    weights.norm2.tensor,
                    &table,
                    &index,
                    6 * spec.hidden,
                    3 * spec.hidden,
                    4 * spec.hidden,
                    entries,
                    spec.norm_eps,
                ),
                (
                    "norm_out",
                    &final_hidden,
                    vector(&final_weight, s)?,
                    &final_table,
                    &final_index,
                    2 * spec.hidden,
                    0,
                    spec.hidden,
                    final_entries,
                    spec.final_norm_eps,
                ),
            ] {
                let n = NormRows {
                    input: pointer(x, s),
                    weight: weight.ptr,
                    output: pointer(&work.normed, s),
                    rows,
                    dim: spec.hidden,
                    threads: HIDDEN_NORM_THREADS,
                    eps,
                };
                let m = ModulateRows {
                    data: n.output,
                    shift: pointer(params, s) + 2 * shift as u64,
                    scale: pointer(params, s) + 2 * scale as u64,
                    index: pointer(indices, s),
                    rows,
                    dim: spec.hidden,
                    stride,
                    entries,
                };
                let serial_n = NormRows {
                    output: pointer(&reference, s),
                    ..n
                };
                let serial_m = ModulateRows {
                    data: serial_n.output,
                    ..m
                };
                kernels.rms_norm_modulate_f32(s, n, m)?;
                kernels.rms_norm_f32(s, serial_n)?;
                kernels.row_modulate(s, serial_m)?;
                let expected = s.clone_dtoh(&reference)?;
                let actual = s.clone_dtoh(&work.normed)?;
                ensure!(
                    expected.iter().chain(&actual).all(|v| v.is_finite()),
                    "RUNNER_FAIL captured norm output nonfinite"
                );
                let bits = expected
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                captured_norm_defects += usize::from(bits > 0);
                println!(
                    "H3_CAPTURED_NORM_MODULATION_CELL name={name} rows={rows} stride={stride} entries={entries} bit_differences={bits} limit=0"
                );
            }
        }
        let modulation = Modulation {
            table: &table,
            offset: 0,
            entries,
            index: &index,
        };
        let mut run = |lora_scale: f32| -> Result<Vec<f64>> {
            let mut x = s.clone_htod(&input)?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale,
            };
            block_forward::<true>(
                &context,
                &mut flash,
                &weights,
                &Conditioning::Modulated {
                    rotary: &rotary,
                    modulation: &modulation,
                },
                &mut x,
                &mut work,
            )?;
            Ok(s.clone_dtoh(&x)?
                .into_iter()
                .map(|v| f64::from(v.to_f32()))
                .collect())
        };
        let candidate = run(adapter.scale)?;
        kernels.serial_norm_modulate.set(true);
        let serial = run(adapter.scale)?;
        kernels.serial_norm_modulate.set(false);
        let bits = candidate
            .iter()
            .zip(&serial)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        drop(serial);
        println!("H3_BLOCK0_MODULATION_IDENTITY bit_differences={bits} limit=0");
        let control = run(0.0)?;
        fs::remove_dir(&runtime)?;

        ensure!(
            candidate.iter().all(|v| v.is_finite()),
            "non-finite block output"
        );
        let d = distance(&candidate, &reference)?;
        let d_capture = distance(&candidate, &captured)?;
        let d_control = distance(&control, &reference)?;
        println!(
            "H3_BLOCK0 candidate_vs_fp32={d:?} candidate_vs_bf16_capture={d_capture:?} lora0_vs_fp32={d_control:?} rotary_bf16_differs={rotary_differs}/{} threshold={threshold}",
            rows * rotary_width
        );
        ensure!(within(d, threshold)?, "block0 {d:?} outside {threshold}");
        ensure!(
            bits == 0 && captured_norm_defects == 0,
            "DEFECT block0 or captured norm/modulation identity"
        );
        ensure!(
            !within(d_control, threshold)?,
            "LoRA-scale-0 control stayed within {threshold}"
        );
        println!("H3_BLOCK0_GATE PASS control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// The GGUF layer gate: one H3 transformer block with the gguf-py-dequantized weights, run by the engine on the
    /// definition reference's dumped input and conditioning, compared with the definition's own output. The floor is
    /// the max over the reference's pairwise arm spreads and the engine's official-weight spread; the control
    /// re-runs with the fused gate+up halves unswapped.
    /// 2026-11-04: the `official` arm keeps the GGUF AdaLN and the GGUF fixed projections, and the floor adds the
    /// `bf16_strict` arm. The earlier floor was the FP32-vs-BF16 spread alone, and the `official` dump it scored
    /// against predated the harness revision that put the GGUF modulation in that arm. The thresholds below were
    /// re-derived from the regenerated arms before the candidate was rerun. The `bf16_strict` spread against
    /// `bf16` is exactly 0 because `allow_bf16_reduced_precision_reduction=False` is a no-op on the CPU
    /// reference, so the floor is the FP32-vs-BF16 spread in practice.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_GGUF_BLOCK_OUT, FF_H3_LORA"]
    fn gguf_block_matches_the_definition_reference() -> Result<()> {
        let var = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let out = var("FF_H3_GGUF_BLOCK_OUT")?;
        let lora_path = var("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        let transformer = model.join("transformer");
        let official = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;

        let input_map = mapped(&out.join("input.safetensors"))?;
        let input = SafeTensors::deserialize(&input_map)?;
        let hidden = input.tensor("hidden_states")?;
        ensure!(
            hidden.dtype() == Dtype::BF16 && hidden.shape().len() == 3,
            "hidden_states {:?} {:?}",
            hidden.dtype(),
            hidden.shape()
        );
        let rows = hidden.shape()[1];
        let input_host: Vec<bf16> = hidden
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let table = input.tensor("modulation")?;
        ensure!(
            table.dtype() == Dtype::BF16 && table.shape().len() == 2,
            "modulation {:?} {:?}",
            table.dtype(),
            table.shape()
        );
        let entries = u32::try_from(table.shape()[0])?;
        let table_host: Vec<bf16> = table
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let rotary = input.tensor("rotary_table")?;
        ensure!(
            rotary.dtype() == Dtype::BF16 && rotary.shape() == [rows, 6 * spec.rope_freq_dim],
            "rotary_table {:?} {:?}",
            rotary.dtype(),
            rotary.shape()
        );
        let rotary_host: Vec<bf16> = rotary
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let (indices, _) = decoded(&input, "adaln_indices")?;
        let index_host: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let output = |arm: &str| -> Result<Vec<f64>> {
            let map = mapped(&out.join(format!("output_{arm}.safetensors")))?;
            let file = SafeTensors::deserialize(&map)?;
            let tensor = file.tensor("output")?;
            ensure!(
                tensor.shape() == [1, rows, spec.hidden],
                "{arm} output {:?} {:?}",
                tensor.dtype(),
                tensor.shape()
            );
            Ok(match tensor.dtype() {
                Dtype::BF16 => tensor
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f64::from(bf16::from_le_bytes(*b).to_f32()))
                    .collect(),
                Dtype::F32 => tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f64::from(f32::from_le_bytes(*b)))
                    .collect(),
                other => bail!("{arm} output dtype {other:?}"),
            })
        };
        let reference_bf16 = output("bf16")?;
        let reference_fp32 = output("fp32")?;
        let reference_bf16_strict = output("bf16_strict")?;
        let reference_official = output("official")?;

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-gguf-block-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 GGUF block gate".into())),
        )?;
        let r = adapter.rank;
        let gguf_dir = gguf_path.parent().context("GGUF directory")?;
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::resident::<true>(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(gguf_dir, &manifest),
            Some(&mut Source::new(&lora_dir, &lora_manifest)),
        )?;
        let mut wrong = manifest.clone();
        wrong
            .entries
            .get_mut("transformer_blocks.0.ff.net.0.proj.weight")
            .context("fc1 entry")?
            .swap_halves = false;
        let wrong_weights = BlockWeights::resident::<true>(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(gguf_dir, &wrong),
            Some(&mut Source::new(&lora_dir, &lora_manifest)),
        )?;
        let official_weights = BlockWeights::resident::<true>(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(&transformer, &official),
            Some(&mut Source::new(&lora_dir, &lora_manifest)),
        )?;
        let mut work = BlockWork::new(s, &spec, r, rows)?;
        work.prepare::<true>(&blas, &spec)?;
        let rotary_device = s.clone_htod(&rotary_host)?;
        let table_device = s.clone_htod(&table_host)?;
        let index_device = s.clone_htod(&index_host)?;
        let modulation = Modulation {
            table: &table_device,
            offset: 0,
            entries,
            index: &index_device,
        };
        let conditioning = Conditioning::Modulated {
            rotary: &rotary_device,
            modulation: &modulation,
        };
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 0.0,
        };
        let mut run = |weights: &BlockWeights| -> Result<Vec<f64>> {
            let mut x = s.clone_htod(&input_host)?;
            block_forward::<true>(
                &context,
                &mut flash,
                weights,
                &conditioning,
                &mut x,
                &mut work,
            )?;
            Ok(s.clone_dtoh(&x)?
                .into_iter()
                .map(|v| f64::from(v.to_f32()))
                .collect())
        };
        let candidate = run(&weights)?;
        kernels.serial_norm_modulate.set(true);
        let serial = run(&weights)?;
        kernels.serial_norm_modulate.set(false);
        let bits = candidate
            .iter()
            .zip(&serial)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        drop(serial);
        println!("H3_GGUF_MODULATION_IDENTITY bit_differences={bits} limit=0");
        let control = run(&wrong_weights)?;
        let engine_official = run(&official_weights)?;

        // The floor on this gate's own input: the reference's max pairwise arm spread, and the engine's
        // block-vs-definition arithmetic spread measured here with the official BF16 weights on the same
        // conditioning (the `official` arm shares the GGUF temb/rotary and the GGUF AdaLN table).
        let arm_pairs = [
            ("bf16_vs_fp32", distance(&reference_bf16, &reference_fp32)?),
            (
                "bf16_vs_bf16_strict",
                distance(&reference_bf16, &reference_bf16_strict)?,
            ),
            (
                "fp32_vs_bf16_strict",
                distance(&reference_fp32, &reference_bf16_strict)?,
            ),
        ];
        let arm_floor = arm_pairs.iter().fold([0.0f64, 0.0, 2.0], |floor, (_, d)| {
            [floor[0].max(d[0]), floor[1].max(d[1]), floor[2].min(d[2])]
        });
        let engine_floor = distance(&engine_official, &reference_official)?;
        let floor = [
            arm_floor[0].max(engine_floor[0]),
            arm_floor[1].max(engine_floor[1]),
            arm_floor[2].min(engine_floor[2]),
        ];
        let d = distance(&candidate, &reference_bf16)?;
        let d_control = distance(&control, &reference_bf16)?;
        let abs_tau = (2.0 * floor[0]).max(1e-5);
        let rel_tau = (2.0 * floor[1]).max(1e-6);
        let cos_tau = 1.0 - 2.0 * (1.0 - floor[2]);
        let ok = d[0] <= abs_tau && d[1] <= rel_tau && d[2] >= cos_tau;
        let control_abs_ratio = d_control[0] / abs_tau;
        let control_rel_ratio = d_control[1] / rel_tau;
        let caught = control_abs_ratio > 10.0 || control_rel_ratio > 10.0;
        println!(
            "H3_GGUF_BLOCK candidate={d:?} arm_pairs={arm_pairs:?} arm_floor={arm_floor:?} engine_floor={engine_floor:?} floor={floor:?} control={d_control:?} control_abs_ratio={control_abs_ratio:.3} control_rel_ratio={control_rel_ratio:.3} abs_tau={abs_tau} rel_tau={rel_tau} within={ok}"
        );
        ensure!(
            candidate.iter().all(|v| v.is_finite()),
            "non-finite block output"
        );
        ensure!(ok, "GGUF block output {d:?} outside the floor {floor:?}");
        ensure!(
            caught,
            "unswapped-fc1 control stayed within 10x tau (abs {:.3}x, rel {:.3}x)",
            control_abs_ratio,
            control_rel_ratio
        );
        fs::remove_dir(&runtime)?;
        ensure!(bits == 0, "DEFECT GGUF norm/modulation identity");
        println!("H3_GGUF_BLOCK_GATE PASS control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// The streamed quantized path gate: the real target's GGUF manifest placed `Streamed` (host tier) and
    /// `StreamedDisk` (disk tier), every layer's device slot compared bit-exact against `BlockWeights::resident`
    /// for the same layer. The control shifts one entry's source offset by one Q4_0 block (18 B) and must fail.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_LORA"]
    fn gguf_streamed_slots_match_resident_weights() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let lora_path = var("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let gguf_dir = gguf_path.parent().context("GGUF directory")?.to_owned();

        let sizing = dense_sizing(&manifest, spec.layers)?;
        let owned = lora_manifest
            .entries
            .iter()
            .filter(|(name, _)| name.starts_with("transformer_blocks."))
            .map(|(_, entry)| entry.bytes)
            .sum::<usize>()
            + spec.layers * 2 * (2 * spec.hidden + 2 * spec.head_dim)
            + sizing.staging
            + sizing.transient;

        let device = Device::new(0)?;
        let stream = &device.stream;
        let mismatches =
            |placement: Placement, perturb: Option<(usize, &str, isize)>| -> Result<Vec<usize>> {
                let mut base = Source::new(&gguf_dir, &manifest);
                let placed = place_layers(
                    "dit",
                    &device,
                    &mut base,
                    (
                        &sizing.layers,
                        &sizing.source_layers,
                        sizing.largest,
                        owned,
                        0,
                    ),
                    (placement, crate::prefill::placement_visits(2, 1)?),
                    |layer| format!("transformer_blocks.{layer}"),
                    &LINEAR_ROLES,
                )?;
                let mut tier = placed.tier.context("streamed placement")?;
                tier.ready()?;
                if let Some((layer, name, delta)) = perturb {
                    tier.dense.shift_source_offset(layer, name, delta)?;
                }
                let mut mismatches = Vec::new();
                for (layer, &layer_bytes) in sizing.layers.iter().enumerate() {
                    let name = format!("transformer_blocks.{layer}");
                    let resident = BlockWeights::resident::<true>(
                        stream,
                        &spec,
                        adapter.rank,
                        &name,
                        &mut Source::new(&gguf_dir, &manifest),
                        Some(&mut Source::new(&lora_dir, &lora_manifest)),
                    )?;
                    let expected = stream
                        .clone_dtoh(resident.storage.as_ref().context("resident storage")?)?;
                    tier.dense.begin(layer, stream)?;
                    let pointer = tier.dense.pointer(layer, 0, layer_bytes)?;
                    let actual = stream.alloc_zeros::<bf16>(expected.len())?;
                    unsafe {
                        crate::backend::cuda::copy::dtod_raw(
                            actual.device_ptr(stream).0,
                            pointer,
                            layer_bytes,
                            stream,
                        )?;
                    }
                    tier.dense.end(layer, stream)?;
                    let actual = stream.clone_dtoh(&actual)?;
                    if actual
                        .iter()
                        .map(|v| v.to_bits())
                        .ne(expected.iter().map(|v| v.to_bits()))
                    {
                        mismatches.push(layer);
                    }
                }
                Ok(mismatches)
            };
        for placement in [Placement::Streamed, Placement::StreamedDisk] {
            let clean = mismatches(placement, None)?;
            ensure!(
                clean.is_empty(),
                "{placement:?}: streamed slots differ from resident in layers {clean:?}"
            );
            let control = mismatches(
                placement,
                Some((0, "transformer_blocks.0.attn.to_q.weight", 18)),
            )?;
            ensure!(
                control == vec![0],
                "{placement:?}: the shifted source offset control did not fail exactly layer 0: {control:?}"
            );
            println!(
                "H3_GGUF_STREAMED {placement:?} layers={} bit_exact_mismatches=0 control=FAIL_AS_REQUIRED",
                spec.layers
            );
        }
        println!("H3_GGUF_STREAMED_GATE PASS");
        Ok(())
    }

    /// The GGUF AdaLN gate: the 50 `adaln_proj` weights dequantized from the GGUF into the engine's AdaLN table,
    /// against the definition's per-block modulation for the same timesteps (the harness dump, layer-major). The
    /// floor is the reference's own FP32-vs-BF16 arm spread; the control swaps the video/audio tags.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_GGUF_ADALN_OUT"]
    fn gguf_adaln_matches_the_definition_modulation() -> Result<()> {
        let var = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let out = var("FF_H3_GGUF_ADALN_OUT")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;

        let adaln_map = mapped(&out.join("adaln.safetensors"))?;
        let adaln = SafeTensors::deserialize(&adaln_map)?;
        let timesteps_tensor = adaln.tensor("timesteps")?;
        ensure!(
            timesteps_tensor.dtype() == Dtype::F32 && timesteps_tensor.shape().len() == 1,
            "timesteps {:?} {:?}",
            timesteps_tensor.dtype(),
            timesteps_tensor.shape()
        );
        let timesteps: Vec<f32> = timesteps_tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let width = 6 * spec.hidden;
        let modulation = |arm: &str| -> Result<Vec<bf16>> {
            let tensor = adaln.tensor(&format!("modulation_{arm}"))?;
            ensure!(
                tensor.shape().len() == 2
                    && tensor.shape()[0] == spec.layers * timesteps.len() * MODALITIES as usize
                    && tensor.shape()[1] == width,
                "modulation_{arm} {:?} {:?} for {} layers x {} timesteps",
                tensor.dtype(),
                tensor.shape(),
                spec.layers,
                timesteps.len()
            );
            Ok(match tensor.dtype() {
                Dtype::BF16 => tensor
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .collect(),
                Dtype::F32 => tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| bf16::from_f32(f32::from_le_bytes(*b)))
                    .collect(),
                other => bail!("modulation_{arm} dtype {other:?}"),
            })
        };
        let to_f64 = |values: &[bf16]| -> Vec<f64> {
            values.iter().map(|&v| f64::from(v.to_f32())).collect()
        };
        let expected_bf16 = to_f64(&modulation("bf16")?);
        let expected_fp32 = to_f64(&modulation("fp32")?);
        let floor = distance(&expected_bf16, &expected_fp32)?;

        let pairs: Vec<(u32, u8)> = (0..timesteps.len() as u32)
            .flat_map(|t| (0..MODALITIES).map(move |tag| (t, tag as u8)))
            .collect();
        let plan = AdalnPlan {
            timesteps: timesteps.clone(),
            endpoints: Vec::new(),
            pairs: pairs.clone(),
            row_pairs: Vec::new(),
            row_timesteps: Vec::new(),
        };

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-gguf-adaln-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 GGUF AdaLN gate".into())),
        )?;
        let gguf_dir = gguf_path.parent().context("GGUF directory")?;
        let table = AdalnTable::build(&device, &mut blas, gguf_dir, &manifest, &spec, &plan)?;
        let blocks: Vec<f64> = s
            .clone_dtoh(&table.blocks)?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect();
        fs::remove_dir(&runtime)?;

        let mut candidate = Vec::new();
        let mut expected = Vec::new();
        let mut control_candidate = Vec::new();
        let mut control_expected = Vec::new();
        for layer in 0..spec.layers {
            for (pair, &(t, tag)) in pairs.iter().enumerate() {
                let row = layer * pairs.len() + pair;
                let c = &blocks[row * width..][..width];
                candidate.extend_from_slice(c);
                expected.extend_from_slice(&expected_bf16[row * width..][..width]);
                if tag != 1 {
                    control_candidate.extend_from_slice(c);
                    let other = layer * pairs.len()
                        + t as usize * MODALITIES as usize
                        + (2 - usize::from(tag));
                    control_expected.extend_from_slice(&expected_bf16[other * width..][..width]);
                }
            }
        }
        let d = distance(&candidate, &expected)?;
        let d_control = distance(&control_candidate, &control_expected)?;
        let abs_tau = (2.0 * floor[0]).max(1e-5);
        let rel_tau = (2.0 * floor[1]).max(1e-6);
        let cos_tau = 1.0 - 2.0 * (1.0 - floor[2]);
        let ok = d[0] <= abs_tau && d[1] <= rel_tau && d[2] >= cos_tau;
        let caught = d_control[0] > 10.0 * abs_tau || d_control[1] > 10.0 * rel_tau;
        println!(
            "H3_GGUF_ADALN timesteps={} pairs={} candidate={d:?} arm_floor={floor:?} control={d_control:?} abs_tau={abs_tau} rel_tau={rel_tau} within={ok}",
            timesteps.len(),
            pairs.len()
        );
        ensure!(
            candidate.iter().all(|v| v.is_finite()),
            "non-finite AdaLN table"
        );
        ensure!(ok, "GGUF AdaLN table {d:?} outside the arm floor {floor:?}");
        ensure!(caught, "tag-swap control stayed within 10x tau");
        println!("H3_GGUF_ADALN_GATE PASS control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// The quantized-manifest leg of the AdaLN build: the GGUF `adaln_proj` tensors (Q4_0, `quant` set on
    /// every entry) build the table through the whole-tensor fill. Fails on a span-only reader, which rejects
    /// every quantized entry on layer 0.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_CAPTURE, FF_Q38_GPU_WINDOW=granted"]
    fn gguf_adaln_table_builds_from_the_quantized_manifest() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let var = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let capture = var("FF_H3_CAPTURE")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        let weight = "transformer_blocks.0.adaln_proj.linear.weight";
        ensure!(
            manifest.entries[weight].quant.is_some(),
            "GGUF fixture {weight} is unquantized"
        );
        let layout = capture_layout(&spec, &capture)?;
        let rows: Vec<RowTimesteps> =
            row_timestep_plan(&layout, &Schedule::new(9, 6.0)?, &Schedule::new(9, 3.0)?)?;
        let adaln = AdalnPlan::build(&layout, &rows)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-gguf-adaln-build-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 GGUF AdaLN build gate".into(),
            )),
        )?;
        let table = AdalnTable::build(
            &device,
            &mut blas,
            gguf_path.parent().context("GGUF directory")?,
            &manifest,
            &spec,
            &adaln,
        )?;
        fs::remove_dir(&runtime)?;
        let width = 6 * spec.hidden;
        let blocks = s.clone_dtoh(&table.blocks)?;
        let norm_out = s.clone_dtoh(&table.norm_out)?;
        ensure!(
            blocks.len() == spec.layers * adaln.pairs.len() * width
                && blocks.iter().all(|v| v.to_f32().is_finite()),
            "GGUF AdaLN blocks extent {}",
            blocks.len()
        );
        ensure!(
            norm_out.len() == adaln.timesteps.len() * 2 * spec.hidden
                && norm_out.iter().all(|v| v.to_f32().is_finite()),
            "GGUF AdaLN norm_out extent {}",
            norm_out.len()
        );
        println!(
            "H3_GGUF_ADALN_BUILD PASS blocks={} norm_out={}",
            blocks.len(),
            norm_out.len()
        );
        Ok(())
    }

    /// Blocks placed per `FF_H3_PLACEMENT`, eager (the H3 path captures no graph), native AdaLN table. Keyframe
    /// conditioning rows are the captured FP32 VAE-encoder latents, the form under which the frozen keyframe cells
    /// are 0. Every step in `FF_H3_STEPS` (each held by `FF_H3_REPLAY`): the native velocity of the generated video
    /// rows and of the audio rows on the run's teacher-forced inputs, within the frozen `{FF_H3_CELL}.step{k}.*`
    /// thresholds; chunked controls repeat FFN input row 0, and single-chunk controls rotate text rows by one.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE (BF16 run), FF_H3_REPLAY (the replay the frozen step thresholds were measured against), FF_H3_THRESHOLDS, FF_H3_CELL, FF_H3_PLACEMENT, FF_H3_STEPS"]
    fn dit_steps_match_the_fp32_replay() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let cell = std::env::var("FF_H3_CELL").context("FF_H3_CELL missing")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let lora_path = path("FF_H3_LORA")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let thresholds = &thresholds["thresholds"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let layout = capture_layout(&spec, &capture)?;
        let plan = row_timestep_plan(&layout, &Schedule::new(9, 6.0)?, &Schedule::new(9, 3.0)?)?;
        let adaln_plan = AdalnPlan::build(&layout, &plan)?;
        let replay_map = mapped(&path("FF_H3_REPLAY")?.join("replay.safetensors"))?;
        let replay = SafeTensors::deserialize(&replay_map)?;
        let steps = std::env::var("FF_H3_STEPS")
            .context("FF_H3_STEPS missing")?
            .split(',')
            .map(|k| Ok(k.parse::<usize>()?))
            .collect::<Result<Vec<usize>>>()?;
        for &k in &steps {
            ensure!(
                k < plan.len() && replay.tensor(&format!("step{k}.noise_pred")).is_ok(),
                "replay holds no step {k} velocity"
            );
        }
        let text_map = mapped(&capture.join("dit_text_input.safetensors"))?;
        let text_file = SafeTensors::deserialize(&text_map)?;
        let text_view = text_file.tensor("encoder_hidden_states")?;
        ensure!(text_view.dtype() == Dtype::BF16, "text input dtype");
        let text_host: Vec<bf16> = text_view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let mut step_hosts = Vec::new();
        for &k in &steps {
            let step_map = mapped(&capture.join(format!("step{k}.safetensors")))?;
            let step_file = SafeTensors::deserialize(&step_map)?;
            let host = |name: &str| -> Result<Vec<f32>> {
                Ok(decoded(&step_file, name)?
                    .0
                    .into_iter()
                    .map(|v| v as f32)
                    .collect())
            };
            step_hosts.push((host("hidden_states")?, host("audio_hidden_states")?));
        }

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-dit-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 DiT gate".into())),
        )?;
        let table = AdalnTable::build(
            &device,
            &mut blas,
            &transformer,
            &manifest,
            &spec,
            &adaln_plan,
        )?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (adapter.rank, &layout),
            Source::new(&transformer, &manifest),
            Some(Source::new(&lora_dir, &lora_manifest)),
            (
                Placement::from_env()?,
                crate::prefill::placement_visits(plan.len(), 1)?,
            ),
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: adapter.scale,
        };
        let counts = (
            u32::try_from(adaln_plan.pairs.len())?,
            u32::try_from(adaln_plan.timesteps.len())?,
        );
        let text = s.clone_htod(&text_host)?;
        let rotated: Vec<bf16> = text_host[spec.text_dim..]
            .iter()
            .chain(&text_host[..spec.text_dim])
            .copied()
            .collect();
        let rotated = s.clone_htod(&rotated)?;
        let condition = layout.condition_video_rows;
        let patch = spec.video_row_width();
        let (mut control_failed, mut defects) = (false, vec![]);
        for (hosts, &k) in step_hosts.iter().zip(&steps) {
            let video = s.clone_htod(&hosts.0)?;
            let audio = s.clone_htod(&hosts.1)?;
            let row_pairs = s.clone_htod(&adaln_plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&adaln_plan.row_timesteps[k])?;
            let chunked_control = dit.work.chunk < dit.work.rows;
            let mut evaluate =
                |text: &CudaSlice<bf16>, repeat_ffn_input: bool| -> Result<[Vec<f64>; 2]> {
                    dit.work.repeat_ffn_input = repeat_ffn_input;
                    dit.step(
                        &context,
                        &table,
                        counts,
                        &StepInputs {
                            video: &video,
                            audio: &audio,
                            text,
                            row_pairs: &row_pairs,
                            row_timesteps: &row_timesteps,
                        },
                    )?;
                    let candidate: Vec<f64> = s
                        .clone_dtoh(&dit.video)?
                        .into_iter()
                        .map(f64::from)
                        .collect();
                    let audio_candidate: Vec<f64> = s
                        .clone_dtoh(&dit.audio)?
                        .into_iter()
                        .map(f64::from)
                        .collect();
                    ensure!(
                        candidate
                            .iter()
                            .chain(&audio_candidate)
                            .all(|v| v.is_finite()),
                        "step {k}: non-finite velocity"
                    );
                    Ok([candidate, audio_candidate])
                };
            let candidate = evaluate(&text, false)?;
            kernels.serial_norm_modulate.set(true);
            let serial = evaluate(&text, false)?;
            kernels.serial_norm_modulate.set(false);
            let bits: usize = candidate
                .iter()
                .zip(&serial)
                .map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count()
                })
                .sum();
            drop(serial);
            println!(
                "H3_STEP_MODULATION_IDENTITY cell={cell} step={k} bit_differences={bits} limit=0"
            );
            let (reference, _) = decoded(&replay, &format!("step{k}.noise_pred"))?;
            let (audio_reference, _) = decoded(&replay, &format!("step{k}.audio_noise_pred"))?;
            let d_video = distance(&candidate[0], &reference[condition * patch..])?;
            let d_audio = distance(&candidate[1], &audio_reference)?;
            let video_threshold = &thresholds[format!("{cell}.step{k}.noise_pred")];
            let audio_threshold = &thresholds[format!("{cell}.step{k}.audio_noise_pred")];
            let verdict = [
                within(d_video, video_threshold)?,
                within(d_audio, audio_threshold)?,
            ];
            println!(
                "H3_DIT step{k} video={d_video:?} audio={d_audio:?} thresholds={video_threshold} {audio_threshold} within={verdict:?}"
            );
            if verdict != [true, true] || bits > 0 {
                defects.push(k);
            }
            if k == steps[0] {
                let control_text = if chunked_control { &text } else { &rotated };
                let kind = if chunked_control {
                    "wrong-ffn-input-offset"
                } else {
                    "rotated-text"
                };
                let control = evaluate(control_text, chunked_control)?;
                let c_video = distance(&control[0], &reference[condition * patch..])?;
                let c_audio = distance(&control[1], &audio_reference)?;
                println!("H3_DIT step{k} {kind} control video={c_video:?} audio={c_audio:?}");
                control_failed =
                    !within(c_video, video_threshold)? || !within(c_audio, audio_threshold)?;
            }
        }
        fs::remove_dir(&runtime)?;
        ensure!(control_failed, "DiT control stayed within the thresholds");
        ensure!(
            defects.is_empty(),
            "H3_DIT_GATE DEFECT cell={cell} step identity or frozen thresholds: {defects:?} of {steps:?}"
        );
        println!("H3_DIT_GATE PASS cell={cell} steps={steps:?} control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Blocks placed per `FF_H3_PLACEMENT`, eager (the H3 path captures no graph), native AdaLN table and grids.
    /// Keyframe conditioning rows are the captured FP32 VAE-encoder latents, the form under which the frozen keyframe
    /// cells are 0. The full chain from the run's step-0 latents against the run's final latents (generated video
    /// rows, audio rows) within the frozen `{FF_H3_CELL}.final.*` thresholds; control: the video and audio grids
    /// swapped must leave a threshold.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE (BF16 run), FF_H3_THRESHOLDS, FF_H3_CELL, FF_H3_PLACEMENT"]
    fn dit_chain_matches_the_reference_final_latents() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let cell = std::env::var("FF_H3_CELL").context("FF_H3_CELL missing")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let lora_path = path("FF_H3_LORA")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let thresholds = &thresholds["thresholds"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let layout = capture_layout(&spec, &capture)?;
        let video_grid = Schedule::new(9, 6.0)?;
        let audio_grid = Schedule::new(9, 3.0)?;
        let plan = row_timestep_plan(&layout, &video_grid, &audio_grid)?;
        let adaln_plan = AdalnPlan::build(&layout, &plan)?;
        let read = |file: &str, name: &str| -> Result<(Vec<f64>, Vec<usize>)> {
            let map = mapped(&capture.join(file))?;
            decoded(&SafeTensors::deserialize(&map)?, name)
        };
        let (video0, _) = read("step0.safetensors", "hidden_states")?;
        let (audio0, _) = read("step0.safetensors", "audio_hidden_states")?;
        let (final_video, video_shape) = read("final.safetensors", "latents")?;
        let (final_audio, audio_shape) = read("final.safetensors", "audio_latents")?;
        let patch = spec.video_row_width();
        let geometry = (layout.rows()
            - layout.condition_video_rows
            - layout.audio_indices.len()
            - layout.text_indices.len())
            * patch;
        ensure!(
            video_shape.len() == 5
                && video_shape[0] == 1
                && video_shape[1] * spec.patch.iter().product::<usize>() == patch
                && final_video.len() == geometry
                && audio_shape.len() == 3
                && audio_shape[1] == spec.audio_in_channels
                && final_audio.len() == audio0.len(),
            "final latents {video_shape:?} {audio_shape:?} do not match the layout"
        );
        // The reference stores `[1, channels, frames, height, width]` video and `[audio_channels, channels, latents]`
        // audio; packed rows are frame-major then row-major with columns (channel, patch_h, patch_w), audio rows
        // channel-major.
        let (channels, frames, height, width) = (
            video_shape[1],
            video_shape[2],
            video_shape[3],
            video_shape[4],
        );
        let (ph, pw) = (spec.patch[1], spec.patch[2]);
        let mut final_video_rows = Vec::with_capacity(final_video.len());
        for frame in 0..frames {
            for i in 0..height / ph {
                for j in 0..width / pw {
                    for c in 0..channels {
                        for y in 0..ph {
                            for x in 0..pw {
                                final_video_rows.push(
                                    final_video[((c * frames + frame) * height + i * ph + y)
                                        * width
                                        + j * pw
                                        + x],
                                );
                            }
                        }
                    }
                }
            }
        }
        let (stereo, latents) = (audio_shape[0], audio_shape[2]);
        let mut final_audio_rows = Vec::with_capacity(final_audio.len());
        for ch in 0..stereo {
            for t in 0..latents {
                for c in 0..audio_shape[1] {
                    final_audio_rows.push(final_audio[(ch * audio_shape[1] + c) * latents + t]);
                }
            }
        }
        let text_map = mapped(&capture.join("dit_text_input.safetensors"))?;
        let text_file = SafeTensors::deserialize(&text_map)?;
        let text_host: Vec<bf16> = text_file
            .tensor("encoder_hidden_states")?
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-chain-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 chain gate".into())),
        )?;
        let table = AdalnTable::build(
            &device,
            &mut blas,
            &transformer,
            &manifest,
            &spec,
            &adaln_plan,
        )?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (adapter.rank, &layout),
            Source::new(&transformer, &manifest),
            Some(Source::new(&lora_dir, &lora_manifest)),
            (
                Placement::from_env()?,
                crate::prefill::placement_visits(plan.len(), 1)?,
            ),
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: adapter.scale,
        };
        let text = s.clone_htod(&text_host)?;
        let condition = layout.condition_video_rows * patch;
        let mut chain = |grids: (&Schedule, &Schedule)| -> Result<[Vec<f64>; 2]> {
            let video = s.clone_htod(&video0.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
            let audio = s.clone_htod(&audio0.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
            dit.sample(
                &context,
                (&table, &adaln_plan),
                grids,
                (&video, &audio),
                &text,
            )?;
            let video: Vec<f64> = s.clone_dtoh(&video)?.into_iter().map(f64::from).collect();
            let audio: Vec<f64> = s.clone_dtoh(&audio)?.into_iter().map(f64::from).collect();
            ensure!(
                video.iter().chain(&audio).all(|v| v.is_finite()),
                "non-finite latents"
            );
            ensure!(
                video[..condition] == video0[..condition],
                "conditioning rows moved"
            );
            Ok([video, audio])
        };
        kernels.serial_norm_modulate.set(true);
        let serial = chain((&video_grid, &audio_grid))?;
        kernels.serial_norm_modulate.set(false);
        let candidate = chain((&video_grid, &audio_grid))?;
        let bits: usize = candidate
            .iter()
            .zip(&serial)
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            })
            .sum();
        drop(serial);
        println!("H3_CHAIN_MODULATION_IDENTITY cell={cell} bit_differences={bits} limit=0");
        let d_video = distance(&candidate[0][condition..], &final_video_rows)?;
        let d_audio = distance(&candidate[1], &final_audio_rows)?;
        let video_threshold = &thresholds[format!("{cell}.final.latents")];
        let audio_threshold = &thresholds[format!("{cell}.final.audio_latents")];
        println!(
            "H3_CHAIN video={d_video:?} audio={d_audio:?} thresholds={video_threshold} {audio_threshold}"
        );
        let control = chain((&audio_grid, &video_grid))?;
        let c_video = distance(&control[0][condition..], &final_video_rows)?;
        let c_audio = distance(&control[1], &final_audio_rows)?;
        println!("H3_CHAIN swapped-grid control video={c_video:?} audio={c_audio:?}");
        fs::remove_dir(&runtime)?;
        ensure!(
            !within(c_video, video_threshold)? || !within(c_audio, audio_threshold)?,
            "swapped-grid control stayed within the thresholds"
        );
        ensure!(
            bits == 0 && within(d_video, video_threshold)? && within(d_audio, audio_threshold)?,
            "DEFECT chain identity or final latents outside the frozen thresholds"
        );
        println!("H3_CHAIN_GATE PASS cell={cell} control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Cells A, D and E of the N6 freeze: tokens and grids of `tokens_from`, pixel rows of `pixel_sources` in order.
    fn conditioner_cells(
        s0: &Path,
        image_token: u32,
    ) -> Result<Vec<(&'static str, ConditionerPrompt)>> {
        let mut cells = Vec::new();
        for (name, tokens_from, pixel_sources) in [
            ("A", "A-bf16", &["A-bf16"][..]),
            ("D", "D-bf16", &["D-bf16"][..]),
            ("E", "D-bf16", &["A-bf16", "B-bf16"][..]),
        ] {
            let map = mapped(&s0.join(tokens_from).join("text.safetensors"))?;
            let text = SafeTensors::deserialize(&map)?;
            let tokens: Vec<u32> = decoded(&text, "token_ids")?
                .0
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let (grid, _) = decoded(&text, "image_grid_thw")?;
            let grids = grid
                .as_chunks::<3>()
                .0
                .iter()
                .map(|g| crate::vision_input::ImageGrid {
                    t: g[0] as u32,
                    h: g[1] as u32,
                    w: g[2] as u32,
                })
                .collect();
            let mut pixels = Vec::new();
            for source in pixel_sources {
                let map = mapped(&s0.join(source).join("text.safetensors"))?;
                let file = SafeTensors::deserialize(&map)?;
                pixels.extend(
                    decoded(&file, "pixel_values")?
                        .0
                        .into_iter()
                        .map(|v| v as f32),
                );
            }
            let modality = tokens
                .iter()
                .map(|&id| u8::from(id == image_token))
                .collect();
            cells.push((
                name,
                ConditionerPrompt {
                    tokens,
                    modality,
                    grids,
                    pixels,
                },
            ));
        }
        Ok(cells)
    }

    /// N6b gate: the Qwen3-VL vision tower on the captured `pixel_values` of cells A, D and E; the merged features and
    /// the three deepstack outputs are scored against the fp32 composition legs at the frozen `thresholds-n6-r2.json`
    /// cells. Eager (the tower captures no graph), all tower weights resident. Control: deepstack taps shifted one
    /// block later (9/17/25) must leave every deepstack output of cell A outside its cell.
    #[test]
    #[ignore = "GPU gate; set FF_H3_TEXT_ENCODER, FF_H3_S0 (Stage 0 runs), FF_H3_N6_LEGS (N6 reference legs) and FF_H3_N6_THRESHOLDS"]
    fn conditioner_vision_matches_the_fp32_reference() -> Result<()> {
        use crate::backend::cuda::{ops::CommonOpsRef, weights::WeightCatalog};
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (encoder, s0, legs) = (
            var("FF_H3_TEXT_ENCODER")?,
            var("FF_H3_S0")?,
            var("FF_H3_N6_LEGS")?,
        );
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N6_THRESHOLDS")?)?)?;
        let spec = crate::model_spec::ModelSpec::read(&encoder)?;
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Qwen3Vl,
            "text encoder parsed as {:?}",
            spec.architecture
        );
        let image_token = spec.vision.as_ref().context("vision missing")?.image_token;
        admit_n6_thresholds(&frozen)?;
        let cells = conditioner_cells(&s0, image_token)?;
        let names = ["merged", "deepstack0", "deepstack1", "deepstack2"];
        let mut references = Vec::new();
        for (cell, _) in &cells {
            references.push(conditioner_named_f64(
                &legs
                    .join(format!("C-{cell}-fp32"))
                    .join("conditioner.safetensors"),
                &VISION_TENSORS,
            )?);
        }
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut catalog = WeightCatalog::load_with_policy(
            &encoder,
            &spec,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        ensure!(
            catalog.bytes == 1_190_533_600,
            "vision weight ledger {} B",
            catalog.bytes
        );
        let mut tower = super::super::vision_runtime::LayerNormTower::from_catalog(
            &spec,
            &mut catalog,
            &device,
            common,
        )?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        device.finish_upload()?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vision-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 conditioner vision gate".into(),
            )),
        )?;
        let mut run = |tower: &super::super::vision_runtime::LayerNormTower,
                       cell: &ConditionerPrompt|
         -> Result<Vec<Vec<f64>>> {
            let input = crate::vision_input::MultimodalInput {
                input_ids: &cell.tokens,
                modality_ids: &cell.modality,
                pixel_values: &cell.pixels,
                grids: &cell.grids,
                audio: &[],
            };
            let request = tower.prepare(&device, &mut blas, &input, cell.tokens.len(), 0)?;
            let encoded = tower.encode(&device.stream, &blas, request, None)?;
            std::iter::once(&encoded.features)
                .chain(&encoded.deepstack)
                .map(|f| {
                    Ok(device
                        .stream
                        .clone_dtoh(f)?
                        .into_iter()
                        .map(|v| f64::from(v.to_f32()))
                        .collect())
                })
                .collect()
        };
        let mut defects = Vec::new();
        let mut swapped_e: Option<Vec<Vec<f64>>> = None;
        for ((cell, inputs), expected) in cells.iter().zip(&references) {
            let outputs = run(&tower, inputs)?;
            ensure!(
                outputs.len() == names.len(),
                "cell {cell}: {} outputs",
                outputs.len()
            );
            for ((name, got), want) in names.iter().zip(&outputs).zip(expected.iter()) {
                ensure!(
                    got.len() == want.len(),
                    "cell {cell} {name}: {} vs {}",
                    got.len(),
                    want.len()
                );
                ensure!(
                    got.iter().all(|v| v.is_finite()),
                    "cell {cell} {name}: non-finite output"
                );
                let d = distance(got, want)?;
                let ok = within(d, &frozen["thresholds"][format!("{cell}.vision.{name}")])?;
                println!("H3_VISION cell={cell} {name}={d:?} within={ok}");
                if !ok {
                    defects.push(format!("{cell}.{name}"));
                }
            }
            if *cell == "E" {
                swapped_e = Some(outputs);
            }
        }
        let swapped = swapped_e.context("cell E missing")?;
        let d_expected = references.get(1).context("cell D missing")?;
        for (name, (got, want)) in names.iter().zip(swapped.iter().zip(d_expected.iter())) {
            let d = distance(got, want)?;
            println!("H3_VISION swapped-image control cell=E_vs_D {name}={d:?}");
            ensure!(
                !within(d, &frozen["thresholds"][format!("D.vision.{name}")])?,
                "swapped image passed D.vision.{name}"
            );
        }
        tower.shift_deepstack_taps()?;
        let (_, a) = cells.first().context("cell A missing")?;
        let shifted = run(&tower, a)?;
        let expected = references.first().context("cell A missing")?;
        for (k, name) in names.iter().enumerate().skip(1) {
            let d = distance(&shifted[k], &expected[k])?;
            println!("H3_VISION shifted-taps control cell=A {name}={d:?}");
            ensure!(
                !within(d, &frozen["thresholds"][format!("A.vision.{name}")])?,
                "shifted deepstack taps passed A.vision.{name}"
            );
        }
        if !defects.is_empty() {
            bail!("H3_VISION_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_VISION_GATE PASS cells=3 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Scores `got` against `want` (`[rows, width]`) under a frozen sink-rule cell: the candidate's sink rows `C`
    /// satisfy `I <= C <= U`, every row of `I` meets its row bound, and the rows outside `U` meet the aggregate bound.
    fn sink_rule(
        got: &[f64],
        want: &[f64],
        width: usize,
        cell: &serde_json::Value,
    ) -> Result<(bool, String)> {
        use std::collections::BTreeSet;
        let rows = want.len() / width;
        let number = |v: &serde_json::Value, key: &str| {
            v[key]
                .as_f64()
                .with_context(|| format!("sink cell lacks {key}"))
        };
        let (median, factor) = (
            number(cell, "median_norm_fp32")?,
            number(cell, "sink_factor")?,
        );
        let row = |v: &[f64], r: usize| v[r * width..(r + 1) * width].to_vec();
        let norm = |v: &[f64], r: usize| row(v, r).iter().map(|x| x * x).sum::<f64>().sqrt();
        let set = |key: &str| -> Result<BTreeSet<usize>> {
            cell[key]
                .as_array()
                .with_context(|| format!("sink cell lacks {key}"))?
                .iter()
                .map(|r| r.as_u64().map(|r| r as usize).context("sink row"))
                .collect()
        };
        let candidate: BTreeSet<usize> = (0..rows)
            .filter(|&r| norm(got, r) > factor * median)
            .collect();
        let (inter, union) = (set("I")?, set("U")?);
        let mut ok = inter.is_subset(&candidate) && candidate.is_subset(&union);
        let mut sink_rows = Vec::new();
        let mut mismatch_detail = String::new();
        for &r in &inter {
            let gr = row(got, r);
            let wr = row(want, r);
            let d = distance(&gr, &wr)?;
            let bound = &cell["rows"][r.to_string()];
            let within_row = d[1] <= number(bound, "rel_max")? && d[2] >= number(bound, "cos_min")?;
            if !within_row {
                let mut best = (0usize, 0.0_f64);
                let (mut sse, mut ref_ss) = (0.0_f64, 0.0_f64);
                for (c, (&gv, &wv)) in gr.iter().zip(&wr).enumerate() {
                    let ad = (gv - wv).abs();
                    if ad > best.1 {
                        best = (c, ad);
                    }
                }
                for (c, (&gv, &wv)) in gr.iter().zip(&wr).enumerate() {
                    if c != best.0 {
                        let df = gv - wv;
                        sse += df * df;
                        ref_ss += wv * wv;
                    }
                }
                let bulk_rel = if ref_ss > 0.0 {
                    (sse / ref_ss).sqrt()
                } else {
                    sse.sqrt()
                };
                mismatch_detail.push_str(&format!(
                    " row_fail[{r}:col={} got={} want={} bulk_rel={bulk_rel:.6}]",
                    best.0, gr[best.0], wr[best.0]
                ));
            }
            ok &= within_row;
            sink_rows.push((r, d[1], d[2], within_row));
        }
        let keep: Vec<usize> = (0..rows).filter(|r| !union.contains(r)).collect();
        let gather = |v: &[f64]| keep.iter().flat_map(|&r| row(v, r)).collect::<Vec<_>>();
        let aggregate = distance(&gather(got), &gather(want))?;
        ok &= within(aggregate, &cell["aggregate"])?;
        Ok((
            ok,
            format!(
                "sinks={candidate:?} I={inter:?} U={union:?} sink_rows={sink_rows:?} aggregate={aggregate:?}{mismatch_detail}"
            ),
        ))
    }

    /// BF16 copy of an F32 reference tensor.
    fn bf16_of(file: &SafeTensors<'_>, name: &str) -> Result<Vec<bf16>> {
        Ok(decoded(file, name)?
            .0
            .into_iter()
            .map(bf16::from_f64)
            .collect())
    }

    /// Conditioner text gate state shared by the N6c and N6d gates: the checkpoint manifest with `layers` decoder
    /// layers loading, the embedding rows of every cell's text tokens, and the frozen thresholds.
    struct TextGate {
        encoder: PathBuf,
        legs: PathBuf,
        spec: crate::model_spec::ModelSpec,
        frozen: serde_json::Value,
        cells: Vec<(&'static str, ConditionerPrompt)>,
        table: BTreeMap<u32, Vec<bf16>>,
        manifest: TensorManifest,
    }

    fn n6_cell_bound(
        cell: &str,
        label: &str,
        tap: usize,
        base: &serde_json::Value,
    ) -> serde_json::Value {
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (cell, label, tap);
            base.clone()
        }
        #[cfg(feature = "rocm")]
        {
            let mut value = base.clone();
            // 2026-11-20: ROCm PyTorch BF16 (default/hipblaslt/math/strict/eager) vs CUDA FP32 C-{A,D,E} floor: I={0,348}, U={0,95,103,174,262,328,348}, eager row[348] 2x rel / 4x cos.
            if label == "composition" && tap == 50 {
                value["I"] = serde_json::json!([0, 348]);
                value["U"] = serde_json::json!([0, 95, 103, 174, 262, 328, 348]);
                let (row348_rel, row348_cos) = match cell {
                    "A" => (
                        2.0 * 0.33944236792362403_f64,
                        1.0 - 4.0 * (1.0 - 0.9997210693941823_f64),
                    ),
                    "D" | "E" => (
                        2.0 * 0.29259424374665566_f64,
                        1.0 - 4.0 * (1.0 - 0.9998047682868612_f64),
                    ),
                    _ => (0.0_f64, 1.0_f64),
                };
                if let Some(rel) = value["rows"]["348"]["rel_max"].as_f64() {
                    value["rows"]["348"]["rel_max"] = serde_json::json!(rel.max(row348_rel));
                }
                if let Some(cos) = value["rows"]["348"]["cos_min"].as_f64() {
                    value["rows"]["348"]["cos_min"] = serde_json::json!(cos.min(row348_cos));
                }
            }
            // 2026-11-20: ROCm PyTorch BF16 causal-prefix T-A (math/strict) row[95] rel=0.005937088989201858 -> 2x=0.011874177978403716 (1 BF16 ULP at col 731; CUDA D/E bound unchanged).
            if label == "text"
                && tap == 50
                && matches!(cell, "D" | "E")
                && let Some(rel) = value["rows"]["95"]["rel_max"].as_f64()
            {
                value["rows"]["95"]["rel_max"] =
                    serde_json::json!(rel.max(2.0 * 0.005937088989201858_f64));
            }
            value
        }
    }

    impl TextGate {
        fn new(layers: usize) -> Result<Self> {
            let var = |key: &str| -> Result<PathBuf> {
                Ok(PathBuf::from(
                    std::env::var_os(key).with_context(|| format!("{key} missing"))?,
                ))
            };
            let (encoder, s0, legs) = (
                var("FF_H3_TEXT_ENCODER")?,
                var("FF_H3_S0")?,
                var("FF_H3_N6_LEGS")?,
            );
            let frozen = serde_json::from_slice(&fs::read(var("FF_H3_N6_THRESHOLDS")?)?)?;
            admit_n6_thresholds(&frozen)?;
            let spec = crate::model_spec::ModelSpec::read(&encoder)?;
            let image_token = spec.vision.as_ref().context("vision missing")?.image_token;
            let manifest = MappedCheckpoint::open_component(&encoder, "model")?
                .minimax_h3_conditioner_manifest(&spec, layers)?;
            let cells = conditioner_cells(&s0, image_token)?;
            let ids: std::collections::BTreeSet<u32> = cells
                .iter()
                .flat_map(|(_, c)| c.tokens.iter().copied())
                .filter(|&id| id != image_token)
                .collect();
            let ids: Vec<u32> = ids.into_iter().collect();
            let hidden = spec.hidden as usize;
            let rows = embedding_rows(
                &mut Source::new(&encoder, &manifest),
                "model.language_model.embed_tokens.weight",
                hidden,
                &ids,
                false,
            )?;
            let table = ids
                .iter()
                .zip(rows.chunks(hidden))
                .map(|(&id, row)| (id, row.to_vec()))
                .collect();
            println!(
                "H3_TEXT embedding_rows={} bytes={}",
                ids.len(),
                2 * rows.len()
            );
            Ok(Self {
                encoder,
                legs,
                spec,
                frozen,
                cells,
                table,
                manifest,
            })
        }

        fn image_token(&self) -> Result<u32> {
            Ok(self
                .spec
                .vision
                .as_ref()
                .context("vision missing")?
                .image_token)
        }

        /// Scores `taps` against `references` and the frozen `{cell}.{label}` cells. Returns whether hidden_states[50] passed.
        fn score(
            &self,
            cell: &str,
            label: &str,
            outputs: &[Vec<bf16>],
            taps: &[usize],
            references: &[Vec<f64>],
        ) -> Result<bool> {
            ensure!(
                outputs.len() == taps.len() && references.len() == taps.len(),
                "cell {cell} tap reference extent"
            );
            let mut gating = true;
            for ((output, &tap), want) in outputs.iter().zip(taps).zip(references) {
                let got: Vec<f64> = output.iter().map(|v| f64::from(v.to_f32())).collect();
                ensure!(
                    got.len() == want.len(),
                    "cell {cell} hidden_states[{tap}] extent"
                );
                ensure!(
                    got.iter().all(|v| v.is_finite()),
                    "cell {cell} hidden_states[{tap}] non-finite"
                );
                let cell_bound = n6_cell_bound(
                    cell,
                    label,
                    tap,
                    &self.frozen["thresholds"][format!("{cell}.{label}.hidden_states_{tap}")],
                );
                let (ok, summary) = sink_rule(&got, want, self.spec.hidden as usize, &cell_bound)?;
                println!("H3_TEXT {label} cell={cell} hidden_states_{tap} within={ok} {summary}");
                if tap == 50 {
                    gating = ok;
                }
            }
            Ok(gating)
        }
    }

    const TEXT_TAPS: [usize; 4] = [1, 4, 25, 50];

    /// N6c gate: Qwen3-VL decoder layers 0-49 on the fp32 reference vision features of cells A, D and E (image rows
    /// spliced as BF16, the three deepstack features added after layers 0-2); hidden_states[50] is scored under the
    /// frozen sink rule of `thresholds-n6-r2.json` against the `T-<cell>-fp32` legs, hidden_states[1/4/25] printed.
    /// Layers placed per `FF_H3_PLACEMENT`, eager (no graph capture). Layer 50 also loads for the control tap.
    /// Controls on cell A: hidden_states[51] scored as hidden_states[50], and deepstack off, must both fail.
    #[test]
    #[ignore = "GPU gate; set FF_H3_TEXT_ENCODER, FF_H3_S0, FF_H3_N6_LEGS, FF_H3_N6_THRESHOLDS and FF_H3_PLACEMENT"]
    fn conditioner_text_matches_the_fp32_reference() -> Result<()> {
        text_conditioner_gate(51)
    }

    #[test]
    #[ignore = "GPU; streamed-disk, frozen text fixtures, FF_H3_CAPTURE output and optional FF_H3_REPLAY parent taps"]
    fn conditioner_text_disk_matches_parent_bits() -> Result<()> {
        ensure!(
            std::env::var("FF_H3_PLACEMENT").as_deref() == Ok("streamed-disk"),
            "text bit gate requires streamed-disk"
        );
        ensure!(
            std::env::var_os("FF_H3_CAPTURE").is_some(),
            "text bit gate capture missing"
        );
        text_conditioner_gate(50)
    }

    fn text_conditioner_gate(layers: usize) -> Result<()> {
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-text-{}-{layers}", std::process::id()));
        fs::create_dir(&runtime)?;
        let outcome = (|| -> Result<()> {
            let gate = TextGate::new(layers)?;
            let mut text_refs = Vec::new();
            let mut vision_bf16 = Vec::new();
            for (cell, _) in &gate.cells {
                text_refs.push(conditioner_text_references(
                    &gate
                        .legs
                        .join(format!("T-{cell}-fp32"))
                        .join("conditioner.safetensors"),
                )?);
                vision_bf16.push(
                    conditioner_named_f64(
                        &gate
                            .legs
                            .join(format!("C-{cell}-fp32"))
                            .join("conditioner.safetensors"),
                        &VISION_TENSORS,
                    )?
                    .into_iter()
                    .map(|values| values.into_iter().map(bf16::from_f64).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                );
            }
            let tap50 = TEXT_TAPS
                .iter()
                .position(|&tap| tap == 50)
                .context("hidden_states_50")?;
            let (hidden, image_token) = (gate.spec.hidden as usize, gate.image_token()?);
            let device = Device::new(0)?;
            let ops = Ops::new(&device.ctx)?;
            device.finish_upload()?;
            let blas = open_blas(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(
                    true,
                    "MiniMax-H3 conditioner text gate".into(),
                )),
            )?;
            let capacity = gate
                .cells
                .iter()
                .map(|(_, c)| c.tokens.len())
                .max()
                .context("no cells")?;
            let mut conditioner = Conditioner::new(
                &device,
                &gate.spec,
                (layers, capacity, 0),
                Source::new(&gate.encoder, &gate.manifest),
                Placement::from_env()?,
                1,
            )?;
            let mut defects = Vec::new();
            for (((cell, inputs), text_ref), vision) in
                gate.cells.iter().zip(&text_refs).zip(&vision_bf16)
            {
                conditioner.prepare(&blas, inputs.tokens.len())?;
                let deepstack = (0..3)
                    .map(|k| {
                        device
                            .stream
                            .clone_htod(&vision[k + 1])
                            .context("upload deepstack tap")
                    })
                    .collect::<Result<Vec<_>>>()?;
                let embeds =
                    conditioner_embeds(inputs, image_token, &gate.table, &vision[0], hidden)?;
                let runs = image_runs(&inputs.tokens, image_token);
                let rope = conditioner_rope(&gate.spec, inputs)?;
                let input = ConditionerInput {
                    embeds: &embeds,
                    deepstack: &deepstack,
                    runs: &runs,
                    rope: &rope,
                };
                let mut taps = TEXT_TAPS.to_vec();
                if layers == 51 {
                    taps.push(51);
                }
                let outputs = conditioner.forward(&device.stream, &blas, &input, &taps)?;
                if !gate.score(cell, "text", &outputs[..4], &TEXT_TAPS, text_ref)? {
                    defects.push(*cell);
                }
                if let Some(root) = std::env::var_os("FF_H3_CAPTURE") {
                    let root = PathBuf::from(root);
                    fs::create_dir_all(&root)?;
                    let bytes: Vec<_> = outputs[3]
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect();
                    fs::write(root.join(format!("text-{cell}-tap50.bf16")), &bytes)?;
                    if let Some(parent) = std::env::var_os("FF_H3_REPLAY") {
                        let expected = fs::read(
                            PathBuf::from(parent).join(format!("text-{cell}-tap50.bf16")),
                        )?;
                        ensure!(
                            bytes.len() == expected.len(),
                            "text parent tap extent differs"
                        );
                        let differences = bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .zip(expected.as_chunks::<2>().0)
                            .filter(|(a, b)| a != b)
                            .count();
                        println!(
                            "H3_TEXT_PAIRED cell={cell} layers={layers} tap50_bit_differences={differences} bound=0"
                        );
                        if differences != 0 {
                            defects.push(*cell);
                        }
                    }
                }
                if *cell == "A" {
                    if layers == 51 {
                        ensure!(
                            !gate.score(
                                cell,
                                "text",
                                &outputs[4..],
                                &[50],
                                std::slice::from_ref(&text_ref[tap50])
                            )?,
                            "hidden_states[51] passed the hidden_states[50] cell"
                        );
                    }
                    let flat = ConditionerInput {
                        deepstack: &[],
                        ..input
                    };
                    let outputs = conditioner.forward(&device.stream, &blas, &flat, &[50])?;
                    ensure!(
                        !gate.score(
                            cell,
                            "text",
                            &outputs,
                            &[50],
                            std::slice::from_ref(&text_ref[tap50])
                        )?,
                        "deepstack-off control passed cell A"
                    );
                    println!(
                        "H3_TEXT controls cell=A layers={layers} tap51={} deepstack_off=FAIL as required",
                        if layers == 51 {
                            "FAIL"
                        } else {
                            "not_requested"
                        }
                    );
                }
            }
            if !defects.is_empty() {
                bail!("H3_TEXT_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
            }
            println!("H3_TEXT_GATE PASS cells=3 control=FAIL_AS_REQUIRED");
            Ok(())
        })();
        fs::remove_dir_all(&runtime)?;
        outcome
    }

    /// N6d gate: the native vision tower on the captured `pixel_values` feeds the native decoder layers 0-49 (merged
    /// features spliced, deepstack added) for cells A, D and E; hidden_states[50] is scored under the frozen sink
    /// rule against the `C-<cell>-fp32` composition legs. Tower eager with resident weights; layers placed per
    /// `FF_H3_PLACEMENT`, eager.
    #[test]
    #[ignore = "GPU gate; set FF_H3_TEXT_ENCODER, FF_H3_S0, FF_H3_N6_LEGS, FF_H3_N6_THRESHOLDS and FF_H3_PLACEMENT"]
    fn conditioner_composition_matches_the_fp32_reference() -> Result<()> {
        use crate::backend::cuda::{ops::CommonOpsRef, weights::WeightCatalog};
        let gate = TextGate::new(50)?;
        let mut text_refs = Vec::new();
        for (cell, _) in &gate.cells {
            text_refs.push(conditioner_text_references(
                &gate
                    .legs
                    .join(format!("C-{cell}-fp32"))
                    .join("conditioner.safetensors"),
            )?);
        }
        let (hidden, image_token) = (gate.spec.hidden as usize, gate.image_token()?);
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let mut catalog = WeightCatalog::load_with_policy(
            &gate.encoder,
            &gate.spec,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        let tower = super::super::vision_runtime::LayerNormTower::from_catalog(
            &gate.spec,
            &mut catalog,
            &device,
            CommonOpsRef::new(&device.ctx)?,
        )?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        device.finish_upload()?;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-composition-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 conditioner composition gate".into(),
            )),
        )?;
        let capacity = gate
            .cells
            .iter()
            .map(|(_, c)| c.tokens.len())
            .max()
            .context("no cells")?;
        let mut media = 0;
        for (_, c) in &gate.cells {
            let input = crate::vision_input::MultimodalInput {
                input_ids: &c.tokens,
                modality_ids: &c.modality,
                pixel_values: &c.pixels,
                grids: &c.grids,
                audio: &[],
            };
            media = media.max(super::super::vision_runtime::LayerNormTower::request_bytes(
                &gate.spec,
                &input,
                c.tokens.len(),
            )?);
        }
        let mut conditioner = Conditioner::new(
            &device,
            &gate.spec,
            (50, capacity, media),
            Source::new(&gate.encoder, &gate.manifest),
            Placement::from_env()?,
            1,
        )?;
        let mut defects = Vec::new();
        for ((cell, inputs), text_ref) in gate.cells.iter().zip(&text_refs) {
            let input = crate::vision_input::MultimodalInput {
                input_ids: &inputs.tokens,
                modality_ids: &inputs.modality,
                pixel_values: &inputs.pixels,
                grids: &inputs.grids,
                audio: &[],
            };
            let request = tower.prepare(&device, &mut blas, &input, inputs.tokens.len(), 0)?;
            let encoded = tower.encode(&device.stream, &blas, request, None)?;
            let merged = device.stream.clone_dtoh(&encoded.features)?;
            conditioner.prepare(&blas, inputs.tokens.len())?;
            let embeds = conditioner_embeds(inputs, image_token, &gate.table, &merged, hidden)?;
            let runs = image_runs(&inputs.tokens, image_token);
            let rope = conditioner_rope(&gate.spec, inputs)?;
            let outputs = conditioner.forward(
                &device.stream,
                &blas,
                &ConditionerInput {
                    embeds: &embeds,
                    deepstack: &encoded.deepstack,
                    runs: &runs,
                    rope: &rope,
                },
                &TEXT_TAPS,
            )?;
            if !gate.score(cell, "composition", &outputs, &TEXT_TAPS, text_ref)? {
                defects.push(*cell);
            }
        }
        drop((conditioner, tower, blas, ops, device));
        let model_root =
            if gate.encoder.file_name().and_then(|n| n.to_str()) == Some("text_encoder") {
                gate.encoder
                    .parent()
                    .context("text_encoder parent")?
                    .to_path_buf()
            } else {
                let root = runtime.join("model");
                fs::create_dir_all(&root)?;
                #[cfg(unix)]
                std::os::unix::fs::symlink(&gate.encoder, root.join("text_encoder"))?;
                root
            };
        let pipe_runtime = runtime.join("pipeline");
        fs::create_dir_all(&pipe_runtime)?;
        let pipe_a = super::pipeline::image_condition(
            &model_root,
            &crate::minimax_h3::ImageRunOptions {
                device: 0,
                runtime_dir: pipe_runtime,
            },
            &gate.cells[0].1,
        )?;
        if !gate.score(
            "A",
            "composition",
            &[pipe_a],
            &[50],
            &[text_refs[0][3].clone()],
        )? {
            defects.push("A.pipeline");
        }
        let _ = fs::remove_dir_all(&runtime);
        if !defects.is_empty() {
            bail!("H3_COMPOSITION_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_COMPOSITION_GATE PASS cells=3 pipeline=A");
        Ok(())
    }

    fn video_distance(
        candidate: &[f32],
        reference: &[u8],
        (channels, total, plane): (usize, usize, usize),
        frames: Option<&[usize]>,
    ) -> Result<[f64; 3]> {
        ensure!(
            candidate.len() == channels * total * plane && reference.len() == 4 * candidate.len(),
            "video comparison extent"
        );
        let all: Vec<_> = (0..total).collect();
        let mut got = Vec::new();
        let mut want = Vec::new();
        for c in 0..channels {
            for &f in frames.unwrap_or(&all) {
                let start = (c * total + f) * plane;
                for (&v, r) in candidate[start..start + plane]
                    .iter()
                    .zip(reference[4 * start..4 * (start + plane)].as_chunks::<4>().0)
                {
                    got.push(f64::from(v));
                    want.push(f64::from(f32::from_le_bytes(*r)));
                }
            }
        }
        let measured = crate::verify::metrics("h3.video", &got, &want)?;
        Ok([measured.max_abs, measured.rel_rmse, measured.cosine])
    }

    #[test]
    fn video_tile_batch_capacity() -> Result<()> {
        let spec = crate::minimax_h3::VaeSpec {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 24,
            block_out_channels: vec![128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_factors: vec![2, 2, 2, 2, 1, 1],
            temporal_factors: vec![1, 2, 2, 1, 1, 1],
            spatial_ratio: 16,
            temporal_ratio: 4,
            dim: 2048,
            layers: 36,
            heads: 32,
            head_dim: 64,
            registers: 4,
            ffn: 8192,
            rope_theta: 100.0,
            rope_dims: 48,
            norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            groups: 32,
            encoder_eps: 1e-6,
            latents_mean: vec![0.0; 24],
            latents_std: vec![1.0; 24],
        };
        let shape = (7, 16, 16);
        assert_eq!(
            vae::TileWork::batch_bytes(&spec, shape, 1, 28)?,
            vae::TileWork::bytes(&spec, shape)
        );
        assert_eq!(
            vae::TileWork::batch_bytes(&spec, shape, 3, 28)?,
            476_534_232
        );
        assert_eq!(
            vae::TileWork::batch_bytes(&spec, shape, 28, 28)?,
            4_442_630_708
        );
        assert!(vae::TileWork::batch_bytes(&spec, shape, 0, 28).is_err());
        assert!(vae::TileWork::batch_bytes(&spec, shape, 29, 28).is_err());
        for shape in [(0, 16, 16), (7, 0, 16), (7, 16, 0)] {
            assert!(vae::TileWork::batch_bytes(&spec, shape, 1, 28).is_err());
        }
        assert!(vae::TileWork::batch_bytes(&spec, (usize::MAX, 16, 16), 1, 28).is_err());
        let block = 134_295_552;
        let fixed = 13_020_336;
        let future = 7_192_309_556;
        let total = 16_000_000_000;
        let budget = crate::prefill::DensePlacement::device_budget(total, total, fixed, 0, future)?;
        let plan = crate::prefill::DensePlacement::new(&[block; 36], None, budget, 10_000_000_000)?;
        assert_eq!(plan.resident_bytes, 36 * block);
        assert_eq!(plan.slot_bytes, 0);
        assert_eq!(plan.host_bytes, 0);
        assert!(fixed + future + plan.resident_bytes + total / 16 <= total);
        let free = 12_000_000_000;
        let budget = crate::prefill::DensePlacement::device_budget(free, total, fixed, 0, future)?;
        let plan = crate::prefill::DensePlacement::new(&[block; 36], None, budget, 10_000_000_000)?;
        assert_eq!(plan.resident_bytes, 26 * block);
        assert_eq!(plan.slot_bytes, block);
        assert_eq!(plan.host_bytes, 10 * block);
        assert_eq!(plan.disk_bytes, 0);
        assert!(fixed + future + plan.resident_bytes + 2 * plan.slot_bytes + total / 16 <= free);
        let streamed =
            crate::prefill::DensePlacement::new(&[block; 36], None, 2 * block, 10_000_000_000)?;
        assert_eq!(streamed.resident_bytes, 0);
        assert_eq!(streamed.host_bytes, 36 * block);
        assert!(
            crate::prefill::DensePlacement::new(&[block; 36], None, 2 * block - 1, 10_000_000_000)
                .is_err()
        );
        Ok(())
    }

    /// Setup receipts precede the first video tile in standalone and released-transformer pipeline contexts.
    #[test]
    #[ignore = "GPU; FF_MODEL, FF_H3_LORA, FF_H3_S0, FF_H3_CELL, FF_H3_PLACEMENT"]
    fn video_decoder_batch_setup_gpu() -> Result<()> {
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let s0 = path("FF_H3_S0")?;
        let cell = std::env::var("FF_H3_CELL")?;
        ensure!(
            matches!(cell.as_str(), "A" | "P"),
            "unknown setup cell {cell}"
        );
        let placement = std::env::var("FF_H3_PLACEMENT")?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vae-setup-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            match placement.as_str() {
                "standalone" => {
                    let spec = crate::minimax_h3::VaeSpec::read(&model)?;
                    let dir = model.join("vae");
                    let manifest =
                        MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
                            .minimax_h3_vae_manifest(&spec)?;
                    let input_map = mapped(&s0.join(format!("{cell}-bf16/decode.safetensors")))?;
                    let file = SafeTensors::deserialize(&input_map)?;
                    let input = file.tensor("video_decoder_input")?;
                    let shape = input.shape();
                    ensure!(
                        input.dtype() == Dtype::F32
                            && shape.len() == 5
                            && shape[0] == 1
                            && shape[1] == spec.latent_channels,
                        "decoder setup input layout"
                    );
                    let device = Device::new(0)?;
                    let ops = Ops::new(&device.ctx)?;
                    device.finish_upload()?;
                    let mut blas = open_blas(
                        &device,
                        &ops,
                        &runtime,
                        Rc::new(Closure::new(true, "VAE setup gate".into())),
                    )?;
                    let decoder = vae::VideoDecoder::new(
                        &device,
                        &spec,
                        &dir,
                        &manifest,
                        (0, Placement::Derived),
                        None,
                    )?;
                    let tiles = crate::minimax_h3::VaeTiles::released(
                        &spec,
                        shape[3] * spec.spatial_ratio,
                        shape[4] * spec.spatial_ratio,
                    )?;
                    decoder.batch_setup(
                        &device,
                        &mut blas,
                        &tiles,
                        (shape[2], shape[3], shape[4]),
                    )?;
                }
                "pipeline" => {
                    let request = seeded_request(&s0, &cell, 42)?;
                    let result =
                        pipeline::decoder_setup(&model, &path("FF_H3_LORA")?, &request, &runtime);
                    match result {
                        Err(error) if error.to_string() == "H3_VAE_SETUP_COMPLETE" => (),
                        Err(error) => return Err(error),
                        Ok(_) => bail!("pipeline setup probe ran past the decoder boundary"),
                    }
                }
                _ => bail!("unknown setup placement {placement}"),
            }
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        println!(
            "H3_VAE_BATCH_SETUP_GATE PASS cell={cell} placement={placement} numerical_candidate_executed=false model_cells_scored=0"
        );
        Ok(())
    }

    /// N7b gate: the native keyframe encoder on the Stage 0 captured k0 pixels and posterior noise of cells A and P
    /// against the captured CUDA FP32 posterior mean, std and normalised latents, under the frozen thresholds-n7n8.json
    /// bounds; encoder tiling off must fail the latents bound.
    #[test]
    #[ignore = "requires FF_MODEL, FF_H3_S0, FF_H3_N7_THRESHOLDS and the GPU"]
    fn keyframe_encoder_matches_the_fp32_capture() -> Result<()> {
        use super::vae::{EncodeWork, KeyframeEncoder};
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, s0) = (var("FF_MODEL")?, var("FF_H3_S0")?);
        let capture = std::env::var_os("FF_H3_CAPTURE").map(PathBuf::from);
        if let Some(capture) = &capture {
            fs::create_dir_all(capture)?;
        }
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let spec = crate::minimax_h3::VaeSpec::read(&model)?;
        let vae_dir = model.join("vae");
        let checkpoint = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_vae_manifest(&spec)?;
        for cell in ["A", "P"] {
            let map = mapped(
                &s0.join(format!("{cell}-bf16"))
                    .join("keyframes.safetensors"),
            )?;
            let file = SafeTensors::deserialize(&map)?;
            let pixels = file.tensor("k0.pixels")?;
            let shape = pixels.shape();
            ensure!(
                pixels.dtype() == Dtype::U8
                    && shape.len() == 5
                    && shape[..3] == [1, spec.in_channels, 1],
                "keyframe pixels {:?} {shape:?}",
                pixels.dtype()
            );
            let eps = file.tensor("k0.posterior_eps")?;
            ensure!(
                eps.dtype() == Dtype::F32,
                "posterior noise dtype {:?}",
                eps.dtype()
            );
        }
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vae-enc-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 keyframe encoder gate".into(),
            )),
        )?;
        let encoder = KeyframeEncoder::new(&device, &spec, &vae_dir, &manifest)?;
        let f32s = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect()
        };
        let keys = ["posterior_mean", "posterior_std", "latents"];
        let mut defects = Vec::new();
        let mut tiling_off = None;
        for cell in ["A", "P"] {
            let map = mapped(
                &s0.join(format!("{cell}-bf16"))
                    .join("keyframes.safetensors"),
            )?;
            let file = SafeTensors::deserialize(&map)?;
            let pixels = file.tensor("k0.pixels")?;
            let shape = pixels.shape().to_vec();
            ensure!(
                pixels.dtype() == Dtype::U8
                    && shape.len() == 5
                    && shape[..3] == [1, spec.in_channels, 1],
                "keyframe pixels {:?} {shape:?}",
                pixels.dtype()
            );
            let (height, width) = (shape[3], shape[4]);
            let eps = file.tensor("k0.posterior_eps")?;
            ensure!(
                eps.dtype() == Dtype::F32,
                "posterior noise dtype {:?}",
                eps.dtype()
            );
            let frame = s.clone_htod(pixels.data())?;
            let eps = s.clone_htod(&f32s(eps.data()))?;
            let encode = |tiles: &crate::minimax_h3::VaeTiles| -> Result<Vec<Vec<f32>>> {
                let mut work = EncodeWork::new(s, &encoder, (tiles.height, tiles.width))?;
                let out =
                    encoder.encode((s, &blas), &mut work, tiles, (&frame, height, width), &eps)?;
                s.synchronize()?;
                out.iter()
                    .map(|b| s.clone_dtoh(b).context("copy encoded VAE band to host"))
                    .collect()
            };
            let tiles = crate::minimax_h3::VaeTiles::released(&spec, height, width)?;
            let native = encode(&tiles)?;
            if let Some(capture) = &capture {
                let taps: BTreeMap<String, Vec<f32>> = keys
                    .iter()
                    .zip(&native)
                    .map(|(name, values)| (format!("k0.{name}"), values.clone()))
                    .collect();
                super::vae::image_tests::save_taps(
                    &capture.join(format!("{cell}.safetensors")),
                    &taps,
                    &file,
                )?;
            }
            let as64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
            for (key, candidate) in keys.iter().zip(&native) {
                let reference = file.tensor(&format!("k0.{key}"))?;
                ensure!(
                    reference.dtype() == Dtype::F32
                        && reference.data().len() == 4 * candidate.len(),
                    "reference k0.{key} {:?} {:?}",
                    reference.dtype(),
                    reference.shape()
                );
                let d = distance(&as64(candidate), &as64(&f32s(reference.data())))?;
                let name = format!("{cell}.keyframe.k0.{key}");
                #[cfg(feature = "rocm")]
                let cell_bound = {
                    let mut value = frozen["cells"][&name].clone();
                    // 2026-11-20: ROCm FP32 AutoencoderKLMiniMaxH3 vs CUDA FP32 capture max_abs=2.2078e-4 (2x=4.4155e-4).
                    if *key == "posterior_mean"
                        && let Some(max_abs) = value["max_abs"].as_f64()
                    {
                        value["max_abs"] =
                            serde_json::json!(max_abs.max(2.0 * 0.00022077560424804688));
                    }
                    value
                };
                #[cfg(feature = "rocm")]
                let ok = within(d, &cell_bound)?;
                #[cfg(not(feature = "rocm"))]
                let ok = within(d, &frozen["cells"][&name])?;
                println!("H3_VAE_KEYFRAME cell={cell} {key}={d:?} within={ok}");
                if !ok {
                    defects.push(name);
                }
            }
            if cell == "A" {
                let untiled = encode(&crate::minimax_h3::VaeTiles::whole(height, width))?;
                let reference = f32s(file.tensor("k0.latents")?.data());
                let d = distance(&as64(&untiled[2]), &as64(&reference))?;
                let failed = !within(d, &frozen["cells"]["A.keyframe.k0.latents"])?;
                println!("H3_VAE_KEYFRAME tiling-off control cell=A latents={d:?} fails={failed}");
                tiling_off = Some(failed);
            }
        }
        ensure!(
            tiling_off == Some(true),
            "encoder tiling-off control passed the A latents bound"
        );
        if !defects.is_empty() {
            bail!("H3_VAE_KEYFRAME_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_VAE_KEYFRAME_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// N8a gate: the native audio VAE decoder on the Stage 0 captured audio decoder input of cells A and P against the
    /// CUDA FP32 replay under the frozen thresholds-n7n8.json bounds; activations without their FIR resampling must
    /// fail.
    #[test]
    #[ignore = "requires FF_MODEL, FF_H3_S0, FF_H3_N7_THRESHOLDS and the GPU"]
    fn audio_decoder_matches_the_fp32_replay() -> Result<()> {
        use super::audio::AudioDecoder;
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, s0) = (var("FF_MODEL")?, var("FF_H3_S0")?);
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let spec = crate::minimax_h3::AudioVaeSpec::read(&model)?;
        let audio_dir = model.join("audio_vae");
        let checkpoint = MappedCheckpoint::open_component(&audio_dir, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_audio_vae_manifest(&spec)?;
        for cell in ["A", "P"] {
            let map = mapped(&s0.join(format!("{cell}-bf16")).join("decode.safetensors"))?;
            let file = SafeTensors::deserialize(&map)?;
            let input = file.tensor("audio_decoder_input")?;
            let shape = input.shape();
            ensure!(
                input.dtype() == Dtype::F32 && shape.len() == 3 && shape[1] == spec.latent_channels,
                "audio decoder input {:?} {shape:?}",
                input.dtype()
            );
            let reference_map = mapped(
                &s0.join(format!("{cell}-fp32-replay"))
                    .join("replay.safetensors"),
            )?;
            let reference_file = SafeTensors::deserialize(&reference_map)?;
            let reference = reference_file.tensor("decode.audio_decoder_output")?;
            ensure!(
                reference.dtype() == Dtype::F32
                    && reference.shape() == [shape[0], 1, shape[2] * spec.hop()],
                "reference audio {:?} {:?}",
                reference.dtype(),
                reference.shape()
            );
        }
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-audio-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 audio decoder gate".into())),
        )?;
        let f32s = |bytes: &[u8]| -> Vec<f64> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f64::from(f32::from_le_bytes(*v)))
                .collect()
        };
        let decode = |decoder: &AudioDecoder, cell: &str| -> Result<(Vec<f64>, Vec<f64>)> {
            let map = mapped(&s0.join(format!("{cell}-bf16")).join("decode.safetensors"))?;
            let file = SafeTensors::deserialize(&map)?;
            let input = file.tensor("audio_decoder_input")?;
            let shape = input.shape().to_vec();
            ensure!(
                input.dtype() == Dtype::F32 && shape.len() == 3 && shape[1] == spec.latent_channels,
                "audio decoder input {:?} {shape:?}",
                input.dtype()
            );
            let latents = s.clone_htod(
                &input
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect::<Vec<_>>(),
            )?;
            let out = decoder.decode((s, &blas), &latents, (shape[0], shape[2]))?;
            s.synchronize()?;
            let candidate: Vec<f64> = s.clone_dtoh(&out)?.into_iter().map(f64::from).collect();
            let reference_map = mapped(
                &s0.join(format!("{cell}-fp32-replay"))
                    .join("replay.safetensors"),
            )?;
            let reference_file = SafeTensors::deserialize(&reference_map)?;
            let reference = reference_file.tensor("decode.audio_decoder_output")?;
            ensure!(
                reference.dtype() == Dtype::F32
                    && reference.shape() == [shape[0], 1, shape[2] * spec.hop()]
                    && reference.data().len() == 4 * candidate.len(),
                "reference audio {:?} {:?}",
                reference.dtype(),
                reference.shape()
            );
            Ok((candidate, f32s(reference.data())))
        };
        let decoder = AudioDecoder::new(&device, &spec, &audio_dir, &manifest)?;
        let mut defects = Vec::new();
        for cell in ["A", "P"] {
            let (candidate, reference) = decode(&decoder, cell)?;
            let d = distance(&candidate, &reference)?;
            let key = format!("{cell}.decode.audio_decoder_output");
            let ok = within(d, &frozen["cells"][&key])?;
            println!("H3_AUDIO cell={cell} pcm={d:?} within={ok}");
            if !ok {
                defects.push(key);
            }
        }
        let control = decoder.without_resampling();
        let (candidate, reference) = decode(&control, "A")?;
        let d = distance(&candidate, &reference)?;
        let failed = !within(d, &frozen["cells"]["A.decode.audio_decoder_output"])?;
        println!("H3_AUDIO fir-off control cell=A pcm={d:?} fails={failed}");
        ensure!(failed, "FIR-off control passed the A audio bound");
        if !defects.is_empty() {
            bail!("H3_AUDIO_DECODE_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_AUDIO_DECODE_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// The recorded i2va request of Stage 0 run `<cell>-bf16`: prompt tokens, image grid and `pixel_values`, the text
    /// tags of the run's layout, the keyframe u8 pixels, and the run's sampling (8 evaluations, shifts 6 / 3). The
    /// four noise draws are generated natively on device from `generator_seed`; no recorded draws are loaded.
    fn seeded_request(
        s0: &Path,
        cell: &str,
        generator_seed: u64,
    ) -> Result<crate::minimax_h3::H3Request> {
        let run = s0.join(format!("{cell}-bf16"));
        let meta: serde_json::Value = serde_json::from_slice(&fs::read(run.join("meta.json"))?)?;
        let text_map = mapped(&run.join("text.safetensors"))?;
        let text = SafeTensors::deserialize(&text_map)?;
        let token_ids: Vec<u32> = decoded(&text, "token_ids")?
            .0
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let grids = decoded(&text, "image_grid_thw")?
            .0
            .as_chunks::<3>()
            .0
            .iter()
            .map(|g| crate::vision_input::ImageGrid {
                t: g[0] as u32,
                h: g[1] as u32,
                w: g[2] as u32,
            })
            .collect();
        let f32s = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<_>>();
        let pixel_values = f32s(decoded(&text, "pixel_values")?.0);
        let layout_map = mapped(&run.join("layout.safetensors"))?;
        let layout = SafeTensors::deserialize(&layout_map)?;
        let (tags, _) = decoded(&layout, "token_tags")?;
        let (rows, _) = decoded(&layout, "text_indices")?;
        let text_tags = rows.iter().map(|&row| tags[row as usize] as u8).collect();
        let key_map = mapped(&run.join("keyframes.safetensors"))?;
        let keys = SafeTensors::deserialize(&key_map)?;
        let pixels = keys.tensor("k0.pixels")?;
        let shape = pixels.shape().to_vec();
        ensure!(
            pixels.dtype() == Dtype::U8 && shape.len() == 5 && shape[..3] == [1, 3, 1],
            "keyframe {shape:?}"
        );
        Ok(crate::minimax_h3::H3Request {
            token_ids,
            text_tags,
            grids,
            pixel_values,
            keyframes: vec![(
                crate::minimax_h3::KeyframeAnchor::First,
                pixels.data().to_vec(),
            )],
            height: shape[3],
            width: shape[4],
            frames: meta["cell"]["frames"].as_u64().context("meta frames")? as usize,
            seed: generator_seed,
            evaluations: 8,
            shifts: (6.0, 3.0),
        })
    }

    /// N9d run: `seeded_request`'s recorded presentation and keyframe with a request seed, through the production
    /// pipeline for cells A, P and B at generator seed 42, plus cell A at generator seed 43 for the seed control. B uses
    /// seed 42 so the A-vs-B media control varies only the keyframe; A at 43 isolates the seed. Writes
    /// `<FF_H3_E2E_OUT>/<cell>/native.safetensors` (denormalised decoder
    /// inputs, the BF16 video decoder output, u8 frames, PCM) for the splice gate. The media control requires A's and B's
    /// frames and PCM to differ; the seed control requires A at seeds 42 and 43 to differ. The envelope comparison is a
    /// measurement (N9-0: the x0.9 video-noise control is not detected), reported in the PR, not a gate.
    #[test]
    #[ignore = "GPU; FF_MODEL, FF_H3_LORA, FF_H3_S0, FF_H3_E2E_OUT"]
    fn end_to_end_generates_from_seeds() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, lora, s0, out) = (
            var("FF_MODEL")?,
            var("FF_H3_LORA")?,
            var("FF_H3_S0")?,
            var("FF_H3_E2E_OUT")?,
        );
        let runtime = std::env::temp_dir().join(format!("ff-h3-e2e-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut media = Vec::new();
        let mut seed43 = None;
        for (cell, seed) in [("A", 42), ("P", 42), ("B", 42), ("A", 43)] {
            let request = seeded_request(&s0, cell, seed)?;
            let generated = super::pipeline::run(&model, &lora, &request, (0, &runtime), None)?;
            let o = &generated.output;
            println!(
                "H3_E2E cell={cell} seed={seed} frames={} size={}x{} samples={} rate={} video_u8_mean={:.3}",
                o.frames,
                o.height,
                o.width,
                o.samples,
                o.sampling_rate,
                o.video.iter().map(|&v| f64::from(v)).sum::<f64>() / o.video.len() as f64
            );
            if seed == 43 && cell == "A" {
                seed43 = Some(generated.output.clone());
                continue;
            }
            let dir = out.join(cell);
            fs::create_dir_all(&dir)?;
            let mut t = crate::tensors::Tensors::default();
            t.f32(
                "video_latents".into(),
                generated.video_latents.0.clone(),
                &generated.video_latents.1,
            )?;
            t.f32(
                "audio_latents".into(),
                generated.audio_latents.0.clone(),
                &generated.audio_latents.1,
            )?;
            t.f32(
                "video_decoded".into(),
                generated.video_decoded.0.clone(),
                &generated.video_decoded.1,
            )?;
            t.f32("audio_decoded".into(), vec![2, 1, o.samples], &o.audio)?;
            t.u8(
                "video_u8".into(),
                vec![o.frames, o.height, o.width, 3],
                &o.video,
            )?;
            t.save_new(
                &dir.join("native.safetensors"),
                [("sampling_rate".to_owned(), o.sampling_rate.to_string())].into(),
            )?;
            media.push((cell, generated.output));
        }
        fs::remove_dir_all(&runtime)?;
        let (a, b) = (&media[0].1, &media[2].1);
        ensure!(
            a.video.len() == b.video.len() && a.audio.len() == b.audio.len(),
            "A and B extents differ"
        );
        let video_diff = a
            .video
            .iter()
            .zip(&b.video)
            .map(|(&x, &y)| f64::from(x.abs_diff(y)))
            .sum::<f64>()
            / a.video.len() as f64;
        let audio_rms = (a
            .audio
            .iter()
            .zip(&b.audio)
            .map(|(&x, &y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            / a.audio.len() as f64)
            .sqrt();
        let differ = a.video != b.video && a.audio != b.audio;
        println!(
            "H3_E2E media control A vs B video_mean_abs_u8={video_diff:.3} audio_rms={audio_rms:.5} differ={differ}"
        );
        ensure!(differ, "the second keyframe left the output unchanged");
        let alt = seed43.context("seed-43 run missing")?;
        let seed_differ = a.video != alt.video && a.audio != alt.audio;
        println!("H3_E2E seed control A seed42 vs seed43 differ={seed_differ}");
        ensure!(
            seed_differ,
            "generator seed 42 and 43 left the output unchanged"
        );
        println!(
            "H3_E2E_RUN PASS cells=3 media_control=DIFFER seed_control=DIFFER envelope=MEASUREMENT"
        );
        Ok(())
    }

    /// N9c: the frozen per-stream normal-distribution bounds and the planted biased-draw controls of the s9 freeze
    /// spec, applied to `philox_normal_f32` at each production count. Thresholds and planted offsets are read from
    /// `FF_H3_N9`; a candidate DEFECT is scored, not hidden. Two grid sizes and two launches are forced.
    #[test]
    #[ignore = "GPU; FF_H3_N9 (s9 freeze dir)"]
    fn native_noise_matches_the_frozen_distribution() -> Result<()> {
        fn norm_cdf(x: f64) -> f64 {
            let z = x / std::f64::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let mut p = 1.061405429f64;
            for &c in &[-1.453152027f64, 1.421413741, -0.284496736, 0.254829592] {
                p = c + t * p;
            }
            let e = 1.0 - p * t * (-z * z).exp();
            if z >= 0.0 {
                0.5 * (1.0 + e)
            } else {
                0.5 * (1.0 - e)
            }
        }
        fn f64s(x: &[f32]) -> Vec<f64> {
            x.iter().map(|&v| f64::from(v)).collect()
        }
        fn mean_var(a: &[f64]) -> (f64, f64) {
            let n = a.len() as f64;
            let m = a.iter().sum::<f64>() / n;
            (m, a.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n)
        }
        fn ks(a: &[f64]) -> f64 {
            let n = a.len() as f64;
            let mut s = a.to_vec();
            s.sort_by(f64::total_cmp);
            let mut d = 0.0f64;
            for (i, &v) in s.iter().enumerate() {
                let cdf = norm_cdf(v);
                d = d
                    .max((cdf - i as f64 / n).abs())
                    .max((cdf - (i as f64 + 1.0) / n).abs());
            }
            d
        }
        fn corr(a: &[f64], b: &[f64]) -> f64 {
            let (am, av) = mean_var(a);
            let (bm, bv) = mean_var(b);
            let cov = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x - am) * (y - bm))
                .sum::<f64>()
                / a.len() as f64;
            cov / (av * bv).sqrt()
        }

        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let n9 = var("FF_H3_N9")?;
        let add: serde_json::Value =
            serde_json::from_slice(&fs::read(n9.join("freeze-spec-n9-addendum1.json"))?)?;
        let spec: serde_json::Value =
            serde_json::from_slice(&fs::read(n9.join("freeze-spec-n9.json"))?)?;
        let rules = spec["N9c"]["gates_per_stream_on_production_count_N"]
            .as_array()
            .context("freeze-spec N9c gate rules missing")?;
        ensure!(
            rules.len() == 6,
            "Runner FAIL: freeze-spec N9c has {} gate rules, expected 6",
            rules.len()
        );
        let streams = add["streams"]
            .as_object()
            .context("addendum 1 streams missing")?;
        ensure!(
            streams.len() == 8,
            "addendum 1 declares 8 production streams, got {}",
            streams.len()
        );

        let device = Device::new(0)?;
        let s = &device.stream;
        let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
        let pairs = noise_draws(42);
        let site = |name: &str| -> Result<usize> {
            Ok(if name.ends_with("posterior_eps") {
                0
            } else if name.ends_with("anchor_noise") {
                1
            } else if name.ends_with("video_noise") {
                2
            } else if name.ends_with("audio_rows") {
                3
            } else {
                bail!("Runner FAIL: unknown draw site {name}");
            })
        };

        let mut defects = Vec::new();
        for (name, entry) in streams {
            let n = entry["N"].as_u64().context("N missing")? as usize;
            let mean_limit = entry["mean_limit"].as_f64().context("mean_limit missing")?;
            let ks_critical = entry["ks_critical"]
                .as_f64()
                .context("ks_critical missing")?;
            let delta = entry["planted_offset"]
                .as_f64()
                .context("planted_offset missing")?;
            let detected_by: Vec<&str> = entry["detected_by"]
                .as_array()
                .context("detected_by missing")?
                .iter()
                .map(|v| v.as_str().context("detected_by entry"))
                .collect::<Result<Vec<_>>>()?;
            let var_limit = 5.0 * (2.0 / n as f64).sqrt();
            let corr_limit = 5.0 / (n as f64).sqrt();
            let (seed, stream) = pairs[site(name)?];

            let d = super::draw_normals(s, &normal, (seed, stream), n, 256)?;
            let x = s.clone_dtoh(&d)?;
            ensure!(
                x.len() == n && x.iter().all(|v| v.is_finite()),
                "Runner FAIL: {name} produced {} values or non-finite draws",
                x.len()
            );
            let xf = f64s(&x);
            let (mean, vari) = mean_var(&xf);
            let ks_d = ks(&xf);
            let lag1 = corr(&xf[..n - 1], &xf[1..n]);

            // Determinism: two launches and two grid sizes bit-identical.
            let d2 = super::draw_normals(s, &normal, (seed, stream), n, 256)?;
            let d3 = super::draw_normals(s, &normal, (seed, stream), n, 1024)?;
            let (x2, x3) = (s.clone_dtoh(&d2)?, s.clone_dtoh(&d3)?);
            let deterministic = x.len() == x2.len()
                && x.len() == x3.len()
                && x.iter()
                    .zip(&x2)
                    .zip(&x3)
                    .all(|((a, b), c)| a.to_bits() == b.to_bits() && a.to_bits() == c.to_bits());

            // Seed control 42 vs 43 differ with |corr| <= 5/sqrt(N).
            let alt = super::draw_normals(s, &normal, (seed + 1, stream), n, 256)?;
            let af = f64s(&s.clone_dtoh(&alt)?);
            let seed_corr = corr(&xf, &af);
            let seed_differ = xf != af;

            // Cross-stream |corr| <= 5/sqrt(N) against a distinct counter range.
            let cross = super::draw_normals(s, &normal, (seed, stream + 4), n, 256)?;
            let cf = f64s(&s.clone_dtoh(&cross)?);
            let cross_corr = corr(&xf, &cf);

            // Planted biased-draw control: output x 0.999 + delta_s. Each frozen `detected_by` detector must
            // fire on the biased draws; a stream with an empty detected_by is "declared as having none".
            let biased: Vec<f64> = xf.iter().map(|&v| 0.999 * v + delta).collect();
            let (bm, bv) = mean_var(&biased);
            let bks = ks(&biased);
            let blag = corr(&biased[..n - 1], &biased[1..n]);
            let fires = |d: &str| match d {
                "mean" => bm.abs() > mean_limit,
                "var" => (bv - 1.0).abs() > var_limit,
                "ks" => bks > ks_critical,
                "corr" => blag.abs() > corr_limit,
                _ => false,
            };
            let detected = detected_by.is_empty() || detected_by.iter().all(|d| fires(d));

            let within = mean.abs() <= mean_limit
                && (vari - 1.0).abs() <= var_limit
                && ks_d <= ks_critical
                && lag1.abs() <= corr_limit
                && cross_corr.abs() <= corr_limit;
            let seed_ok = seed_corr.abs() <= corr_limit && seed_differ;
            let pass = within && deterministic && seed_ok && detected;
            println!(
                "H3_N9C {name} N={n} mean={mean:.6e} var={vari:.6} ks={ks_d:.6} lag1={lag1:.6} \
                 cross={cross_corr:.6} seed_corr={seed_corr:.6} deterministic={deterministic} \
                 control_mean={bm:.6e} control_ks={bks:.6} detected={detected} pass={pass}"
            );
            if !pass {
                defects.push(name.clone());
            }
        }
        if !defects.is_empty() {
            bail!("H3_N9C_GATE DEFECT streams outside frozen bounds: {defects:?}");
        }
        println!("H3_N9C_GATE PASS streams=8 controls_detected=8");
        Ok(())
    }

    /// GGUF measurement: one seeded cell A request run twice through `generate`, once with the official BF16
    /// transformer manifest and once with the community GGUF, and the frames/PCM distance printed. The two
    /// arms share the request, so only the transformer weights differ; the numbers are a measurement, not a gate.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_LORA, FF_H3_S0, FF_H3_E2E_OUT"]
    fn gguf_end_to_end_measurement_against_the_bf16_path() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, gguf, lora, s0, out) = (
            var("FF_MODEL")?,
            var("FF_H3_GGUF")?,
            var("FF_H3_LORA")?,
            var("FF_H3_S0")?,
            var("FF_H3_E2E_OUT")?,
        );
        let request = seeded_request(&s0, "A", 42)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-gguf-e2e-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let bf16 = super::generate(&model, &lora, &request, (0, &runtime), None)?;
        let quantized = super::generate(&model, &lora, &request, (0, &runtime), Some(&gguf))?;
        fs::remove_dir_all(&runtime)?;
        let (a, b) = (&bf16, &quantized);
        ensure!(
            a.frames == b.frames
                && a.height == b.height
                && a.width == b.width
                && a.video.len() == b.video.len()
                && a.samples == b.samples
                && a.audio.len() == b.audio.len(),
            "the two arms differ in extent"
        );
        ensure!(b.audio.iter().all(|v| v.is_finite()), "non-finite GGUF PCM");
        let video: Vec<f64> = a.video.iter().map(|&v| f64::from(v)).collect();
        let video_b: Vec<f64> = b.video.iter().map(|&v| f64::from(v)).collect();
        let pcm: Vec<f64> = a.audio.iter().map(|&v| f64::from(v)).collect();
        let pcm_b: Vec<f64> = b.audio.iter().map(|&v| f64::from(v)).collect();
        let video_d = distance(&video, &video_b)?;
        let pcm_d = distance(&pcm, &pcm_b)?;
        ensure!(
            a.video != b.video || a.audio != b.audio,
            "the GGUF arm left the output unchanged"
        );
        let dir = out.join("A-gguf");
        fs::create_dir_all(&dir)?;
        let mut t = crate::tensors::Tensors::default();
        t.u8(
            "video_u8".into(),
            vec![b.frames, b.height, b.width, 3],
            &b.video,
        )?;
        t.f32("audio_decoded".into(), vec![2, 1, b.samples], &b.audio)?;
        t.save_new(
            &dir.join("native.safetensors"),
            [("sampling_rate".to_owned(), b.sampling_rate.to_string())].into(),
        )?;
        println!(
            "H3_GGUF_E2E cell=A frames={} video_max_abs={:.6e} video_rel_rms={:.6e} video_cosine={:.9} pcm_max_abs={:.6e} pcm_rel_rms={:.6e} pcm_cosine={:.9}",
            b.frames, video_d[0], video_d[1], video_d[2], pcm_d[0], pcm_d[1], pcm_d[2]
        );
        println!(
            "H3_GGUF_E2E_MEASUREMENT frames={} video={video_d:?} pcm={pcm_d:?}",
            b.frames
        );
        Ok(())
    }

    /// N8b splice gate: for cells A and P, the native decoder outputs of the end-to-end run against the model
    /// definition's CUDA FP32 decoders on the same native latents (`reference.safetensors`, harness leg) under the
    /// frozen N7a / N8a bounds (video whole and cross-fade frames, audio); control: the native video with frame k
    /// replaced by frame k + 1 must leave the whole-video bound.
    #[test]
    #[ignore = "FF_H3_E2E_OUT (run and harness dumps), FF_H3_N7_THRESHOLDS"]
    fn end_to_end_splice_matches_the_definition_decoders() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let out = var("FF_H3_E2E_OUT")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let mut defects = Vec::new();
        for cell in ["A", "P"] {
            let native_map = mapped(&out.join(cell).join("native.safetensors"))?;
            let native = SafeTensors::deserialize(&native_map)?;
            let reference_map = mapped(&out.join(cell).join("reference.safetensors"))?;
            let reference = SafeTensors::deserialize(&reference_map)?;
            let video = native.tensor("video_decoded")?;
            let want = reference.tensor("video_decoder_output")?;
            ensure!(
                video.dtype() == Dtype::F32
                    && want.dtype() == Dtype::F32
                    && video.shape() == want.shape(),
                "splice video {:?} {:?} vs {:?} {:?}",
                video.dtype(),
                video.shape(),
                want.dtype(),
                want.shape()
            );
            let shape = video.shape().to_vec();
            let extents = (shape[1], shape[2], shape[3] * shape[4]);
            let candidate: Vec<f32> = video
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let key = format!("{cell}.decode.video_decoder_output");
            let whole = video_distance(&candidate, want.data(), extents, None)?;
            let ok = within(whole, &frozen["cells"][&key])?;
            println!("H3_SPLICE cell={cell} video={whole:?} within={ok}");
            if !ok {
                defects.push(key.clone());
            }
            let fade_key = format!("{key}.crossfade_frames");
            let fade: Vec<usize> = frozen["cells"][&fade_key]["frames"]
                .as_array()
                .context("crossfade frame list missing")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize).context("crossfade frame"))
                .collect::<Result<_>>()?;
            let faded = video_distance(&candidate, want.data(), extents, Some(&fade))?;
            let ok = within(faded, &frozen["cells"][&fade_key])?;
            println!("H3_SPLICE cell={cell} crossfade_frames={faded:?} within={ok}");
            if !ok {
                defects.push(fade_key);
            }
            let (channels, frames, plane) = extents;
            let mut shifted = candidate.clone();
            for c in 0..channels {
                for f in 0..frames {
                    let from = (c * frames + (f + 1).min(frames - 1)) * plane;
                    shifted[(c * frames + f) * plane..(c * frames + f + 1) * plane]
                        .copy_from_slice(&candidate[from..from + plane]);
                }
            }
            let mutant = video_distance(&shifted, want.data(), extents, None)?;
            let caught = !within(mutant, &frozen["cells"][&key])?;
            println!("H3_SPLICE frame-shift control cell={cell} video={mutant:?} fails={caught}");
            ensure!(caught, "frame-shift control passed the {cell} video bound");
            let pcm = native.tensor("audio_decoded")?;
            let want = reference.tensor("audio_decoder_output")?;
            ensure!(
                pcm.dtype() == Dtype::F32
                    && want.dtype() == Dtype::F32
                    && pcm.shape() == want.shape(),
                "splice audio {:?} vs {:?}",
                pcm.shape(),
                want.shape()
            );
            let as64 = |b: &[u8]| {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f64::from(f32::from_le_bytes(*v)))
                    .collect::<Vec<_>>()
            };
            let d = distance(&as64(pcm.data()), &as64(want.data()))?;
            let key = format!("{cell}.decode.audio_decoder_output");
            let ok = within(d, &frozen["cells"][&key])?;
            println!("H3_SPLICE cell={cell} pcm={d:?} within={ok}");
            if !ok {
                defects.push(key);
            }
        }
        if !defects.is_empty() {
            bail!("H3_SPLICE_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_SPLICE_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    struct ImageDitCell {
        name: &'static str,
        cross_max_abs: f64,
        cross_rmse: f64,
    }

    /// Cross-device pair of the 512 DiT cells. `tau = max(2 * pair, 1e-6)`.
    const IMAGE_DIT_512: &[ImageDitCell] = &[
        ImageDitCell {
            name: "dit_edit_step0",
            cross_max_abs: 4.986_862_897_872_925,
            cross_rmse: 1.7005365820597132e-1,
        },
        ImageDitCell {
            name: "dit_step0",
            cross_max_abs: 7.214_647_103_101_015e-1,
            cross_rmse: 5.7145887678887466e-2,
        },
        ImageDitCell {
            name: "dit_step5",
            cross_max_abs: 2.703_127_861_022_949e-1,
            cross_rmse: 2.1146607018288893e-2,
        },
        ImageDitCell {
            name: "euler_edit_step0",
            cross_max_abs: 2.859_330_177_307_129e-2,
            cross_rmse: 1.3387505749533405e-3,
        },
        ImageDitCell {
            name: "euler_step0",
            cross_max_abs: 2.343_052_625_656_128e-2,
            cross_rmse: 1.8558872180865057e-3,
        },
    ];

    /// Twice the 2026-11-19T10:35:51Z FP16-versus-BF16 floor. prod64 is the production cell.
    const FP16_BF16_FLOOR: &[(&str, f64, f64)] = &[
        ("tiny1", 0.015625, 0.0126576966),
        ("tiny7", 0.5, 0.123593084),
        ("full16", 1.0, 0.167570942),
        ("m1", 0.375, 0.142660596),
        ("m17", 0.5, 0.154278528),
        ("m64", 1.0, 0.2326228),
        ("prod64", 4.0, 0.73971736),
        ("tail12", 0.125, 0.037559314),
    ];

    fn fp16_dit_bound() -> (f64, f64) {
        (4.0, 0.73971736)
    }

    fn scored_image_tau(
        sm: u32,
        name: &str,
        published_max: f64,
        published_rmse: f64,
    ) -> (f64, f64) {
        if sm < 80 && name.starts_with("dit_") {
            fp16_dit_bound()
        } else {
            (published_max, published_rmse)
        }
    }

    #[test]
    fn fp16_dit_bound_matches_the_frozen_gemm_floor() {
        let frozen = [
            ("tiny1", 0.015625, 0.0126576966),
            ("tiny7", 0.5, 0.123593084),
            ("full16", 1.0, 0.167570942),
            ("m1", 0.375, 0.142660596),
            ("m17", 0.5, 0.154278528),
            ("m64", 1.0, 0.2326228),
            ("prod64", 4.0, 0.73971736),
            ("tail12", 0.125, 0.037559314),
        ];
        assert_eq!(FP16_BF16_FLOOR, &frozen[..]);
        assert_eq!(fp16_dit_bound(), (4.0, 0.73971736));
        assert_eq!(
            FP16_BF16_FLOOR
                .iter()
                .find(|cell| cell.0 == "prod64")
                .map(|cell| (cell.1, cell.2)),
            Some(fp16_dit_bound())
        );
        assert_eq!(
            scored_image_tau(75, "dit_step0", 1.0, 0.1),
            fp16_dit_bound()
        );
        assert_eq!(
            scored_image_tau(75, "dit_step5", 1.0, 0.1),
            fp16_dit_bound()
        );
        assert_eq!(
            scored_image_tau(75, "dit_edit_step0", 1.0, 0.1),
            fp16_dit_bound()
        );
        assert_eq!(scored_image_tau(75, "euler_step0", 1.0, 0.1), (1.0, 0.1));
        assert_eq!(
            scored_image_tau(75, "euler_edit_step0", 2.0, 0.2),
            (2.0, 0.2)
        );
        assert_eq!(scored_image_tau(89, "dit_step0", 1.0, 0.1), (1.0, 0.1));
        assert_eq!(scored_image_tau(80, "dit_step5", 3.0, 0.3), (3.0, 0.3));
    }

    #[test]
    fn t4_standard_pin_matches_the_counted_budget() {
        let slot = 770_703_360usize;
        let pin = 7_768_899_584usize;
        let host_pins = pin / slot;
        let disk_layers = 50 - host_pins;
        assert_eq!(host_pins, 10);
        assert_eq!(disk_layers, 40);
        assert_eq!(disk_layers * slot * 6, 184_968_806_400);
    }

    fn image_dit_tau(cross: f64) -> f64 {
        (2.0 * cross).max(1e-6)
    }

    fn image_f32(file: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
        let view = file.tensor(name).with_context(|| format!("lacks {name}"))?;
        ensure!(
            view.dtype() == Dtype::F32,
            "{name} dtype {:?}",
            view.dtype()
        );
        let (chunks, rest) = view.data().as_chunks::<4>();
        ensure!(rest.is_empty(), "{name} trailing bytes");
        Ok(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    fn image_tags(file: &SafeTensors<'_>, name: &str) -> Result<Vec<u8>> {
        let view = file.tensor(name).with_context(|| format!("lacks {name}"))?;
        ensure!(
            view.dtype() == Dtype::I64,
            "{name} dtype {:?}",
            view.dtype()
        );
        let (chunks, rest) = view.data().as_chunks::<8>();
        ensure!(rest.is_empty(), "{name} trailing bytes");
        chunks
            .iter()
            .map(|b| u8::try_from(i64::from_le_bytes(*b)).with_context(|| format!("{name} tag")))
            .collect()
    }

    fn image_bf16_exact(values: &[f32]) -> Result<Vec<bf16>> {
        let out: Vec<bf16> = values.iter().copied().map(bf16::from_f32).collect();
        ensure!(
            out.iter()
                .zip(values)
                .all(|(got, &want)| got.to_f32().to_bits() == want.to_bits()),
            "conditioner values are not bf16"
        );
        Ok(out)
    }

    fn image_max_rmse(candidate: &[f32], reference: &[f32]) -> Result<(f64, f64)> {
        crate::verify::max_abs_rmse("h3.image", candidate, reference)
    }

    /// Blocks placed per `FF_H3_PLACEMENT`, eager. Step 0 and step 5 are separate forwards on the same patchified
    /// rows. Euler step 0 writes generated rows only. `inputs.safetensors` supplies `t2i_rows` and `edit_rows`.
    /// Image row `i` takes the `(h, w)` of image row `i + 1`. That line is printed.
    /// Every image row's `(h, w)` is swapped to `(w, h)`.
    /// On sm >= 80 the control bar is `3 * tau` against the BF16 reference.
    /// On sm < 80 a `dit_*` cell uses the prod64 floor. Euler stays on `image_dit_tau`.
    /// On sm < 80 the transpose must differ from the unshifted candidate by more than that floor.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_H3_IMAGE, FF_H3_IMAGE_DEVICE0, FF_H3_PLACEMENT"]
    fn image_dit_512_matches_the_frozen_cells() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let model = path("FF_H3_IMAGE")?;
        let floor = path("FF_H3_IMAGE_DEVICE0")?;
        let placement = Placement::from_env()?;
        let image_spec = ImageSpec::read(&model)?;
        let spec = super::image_block_spec(&image_spec);
        let width = image_spec.row_width();
        let capture_map = mapped(&floor.join("a.safetensors"))?;
        let capture = SafeTensors::deserialize(&capture_map)?;
        let inputs_map = mapped(&floor.join("inputs.safetensors"))?;
        let inputs = SafeTensors::deserialize(&inputs_map)?;
        let t2i_tags = image_tags(&capture, "conditioner_t2i_tags")?;
        let edit_tags = image_tags(&capture, "conditioner_edit_tags")?;
        let encode = capture.tensor("vae_encode_one")?;
        ensure!(
            encode.shape().len() == 4
                && encode.shape()[0] == 1
                && encode.shape()[1] == image_spec.in_channels,
            "vae_encode_one shape {:?}",
            encode.shape()
        );
        let (ref_h, ref_w) = (encode.shape()[2], encode.shape()[3]);
        let step0_ref = image_f32(&capture, "dit_step0")?;
        ensure!(
            step0_ref.len().is_multiple_of(width)
                && capture.tensor("dit_step0")?.shape().last() == Some(&width),
            "dit_step0 width"
        );
        let image_rows = step0_ref.len() / width;
        let mut tokens = 1usize;
        while tokens * tokens < image_rows {
            tokens += 1;
        }
        ensure!(
            tokens * tokens == image_rows,
            "t2i image rows {image_rows} are not a square token grid"
        );
        let latent = tokens * image_spec.patch[1];
        let t2i_layout = image_layout(&image_spec, &t2i_tags, &[], latent, latent)?;
        let edit_layout = image_layout(&image_spec, &edit_tags, &[(ref_h, ref_w)], latent, latent)?;
        let t2i_text = image_bf16_exact(&image_f32(&capture, "conditioner_t2i")?)?;
        let edit_text = image_bf16_exact(&image_f32(&capture, "conditioner_edit")?)?;
        let t2i_rows = image_f32(&inputs, "t2i_rows")?;
        let edit_rows = image_f32(&inputs, "edit_rows")?;
        let step5_ref = image_f32(&capture, "dit_step5")?;
        let euler_ref = image_f32(&capture, "euler_step0")?;
        let edit_step_ref = image_f32(&capture, "dit_edit_step0")?;
        let edit_euler_ref = image_f32(&capture, "euler_edit_step0")?;
        ensure!(
            t2i_tags.len() >= 2
                && t2i_text.len() == t2i_tags.len() * image_spec.text_dim
                && edit_text.len() == edit_tags.len() * image_spec.text_dim
                && t2i_layout.image_indices.len() == image_rows
                && t2i_rows.len() == step0_ref.len()
                && step5_ref.len() == step0_ref.len()
                && euler_ref.len() == step0_ref.len()
                && edit_layout.image_indices.len() * width == edit_step_ref.len()
                && edit_rows.len() == edit_step_ref.len()
                && edit_euler_ref.len() == edit_step_ref.len()
                && t2i_rows.iter().any(|v| *v != 0.0)
                && edit_rows.iter().any(|v| *v != 0.0)
                && t2i_rows.iter().chain(&edit_rows).all(|v| v.is_finite()),
            "512 DiT fixture extents"
        );
        let source = t2i_layout.position_ids.clone();
        let image_at = &t2i_layout.image_indices;
        let n_image = image_at.len();
        let mut shifted_ids = source.clone();
        for i in 0..n_image {
            let to = image_at[i] as usize;
            let from = image_at[(i + 1) % n_image] as usize;
            shifted_ids[to][1] = source[from][1];
            shifted_ids[to][2] = source[from][2];
        }
        ensure!(
            n_image >= 2
                && shifted_ids != source
                && shifted_ids
                    .iter()
                    .zip(&source)
                    .all(|(got, want)| got[0] == want[0])
                && t2i_layout
                    .text_indices
                    .iter()
                    .all(|&row| shifted_ids[row as usize] == source[row as usize])
                && image_at.iter().enumerate().all(|(i, &row)| {
                    let from = image_at[(i + 1) % n_image] as usize;
                    let got = shifted_ids[row as usize];
                    got[1] == source[from][1] && got[2] == source[from][2]
                }),
            "rope shift is not a one-patch image-row rotation"
        );
        let mut transposed_ids = source.clone();
        for &row in image_at {
            let row = row as usize;
            transposed_ids[row][1] = source[row][2];
            transposed_ids[row][2] = source[row][1];
        }
        ensure!(
            transposed_ids != source
                && transposed_ids
                    .iter()
                    .zip(&source)
                    .all(|(got, want)| got[0] == want[0])
                && t2i_layout
                    .text_indices
                    .iter()
                    .all(|&row| transposed_ids[row as usize] == source[row as usize])
                && image_at.iter().all(|&row| {
                    let row = row as usize;
                    transposed_ids[row][1] == source[row][2]
                        && transposed_ids[row][2] == source[row][1]
                }),
            "spatial transpose is not (h, w) to (w, h)"
        );
        let schedule = ImageSchedule::published()?;
        let t2i_plan = AdalnPlan::from_tags(
            &t2i_layout.token_tags,
            &image_row_plan(&t2i_layout, &schedule)?,
        )?;
        let edit_plan = AdalnPlan::from_tags(
            &edit_layout.token_tags,
            &image_row_plan(&edit_layout, &schedule)?,
        )?;
        ensure!(
            t2i_plan.row_pairs.len() == 6 && edit_plan.row_pairs.len() == 6,
            "evaluation count"
        );
        let dir = model.join("transformer_diffusers");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_image_manifest(&image_spec)?;
        let device = Device::new(0)?;
        let sm = cubin::capability(&device.ctx)?;
        println!("H3_IMAGE_DIT sm={sm}");
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-image-dit-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3-Image DiT gate".into())),
        )?;
        let mut defects = Vec::new();
        let (
            shift_max,
            shift_rmse,
            shift_vs_max,
            shift_vs_rmse,
            control_max,
            control_rmse,
            versus_max,
            versus_rmse,
            sample_changed,
        ) = {
            let table = AdalnTable::build(&device, &mut blas, &dir, &manifest, &spec, &t2i_plan)?;
            let mut dit = ImageDit::new(
                &device,
                &mut blas,
                (&spec, &image_spec),
                &t2i_layout,
                &dir,
                &manifest,
                (
                    placement,
                    crate::prefill::placement_visits(schedule.timesteps.len(), 1)?,
                ),
            )?;
            let kernels = BlockKernels::new(&device.ctx)?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: 1.0,
            };
            let counts = (
                u32::try_from(t2i_plan.pairs.len())?,
                u32::try_from(t2i_plan.timesteps.len())?,
            );
            let rows_d = s.clone_htod(&t2i_rows)?;
            let text_d = s.clone_htod(&t2i_text)?;
            let velocity = |context: &BlockContext,
                            head: usize,
                            text: &CudaSlice<bf16>,
                            dit: &mut ImageDit|
             -> Result<Vec<f32>> {
                let row_pairs = s.clone_htod(&t2i_plan.row_pairs[head])?;
                let row_timesteps = s.clone_htod(&t2i_plan.row_timesteps[head])?;
                dit.step(
                    context,
                    &table,
                    counts,
                    head,
                    &ImageStepInputs {
                        image: &rows_d,
                        text,
                        row_pairs: &row_pairs,
                        row_timesteps: &row_timesteps,
                    },
                )?;
                s.clone_dtoh(&dit.velocity)
                    .context("copy DiT velocity to host")
            };
            let step0 = velocity(&context, 0, &text_d, &mut dit)?;
            let residual = s.clone_dtoh(&dit.x)?;
            let euler_rows = s.clone_htod(&t2i_rows)?;
            dit.euler(s, &euler_rows, &schedule, 0)?;
            let euler = s.clone_dtoh(&euler_rows)?;
            let step5 = velocity(&context, 5, &text_d, &mut dit)?;
            let mut fresh = s.clone_htod(&t2i_rows)?;
            dit.sample(
                &context,
                (&table, &t2i_plan),
                &schedule,
                &mut fresh,
                &text_d,
                (None, None),
            )?;
            let sampled = s.clone_dtoh(&fresh)?;
            let changed = sampled.len() == t2i_rows.len()
                && sampled.iter().all(|v| v.is_finite())
                && sampled
                    .iter()
                    .zip(&t2i_rows)
                    .any(|(got, want)| got.to_bits() != want.to_bits());
            for (name, candidate, reference) in [
                ("dit_step0", &step0, &step0_ref),
                ("dit_step5", &step5, &step5_ref),
                ("euler_step0", &euler, &euler_ref),
            ] {
                let cell = IMAGE_DIT_512
                    .iter()
                    .find(|cell| cell.name == name)
                    .context(name)?;
                let (max_abs, rmse) = image_max_rmse(candidate, reference)?;
                let (tau_max, tau_rmse) = scored_image_tau(
                    sm,
                    name,
                    image_dit_tau(cell.cross_max_abs),
                    image_dit_tau(cell.cross_rmse),
                );
                let ok = candidate.iter().all(|v| v.is_finite())
                    && max_abs <= tau_max
                    && rmse <= tau_rmse;
                println!(
                    "H3_IMAGE_DIT {name} max_abs={max_abs:.8e} rmse={rmse:.8e} tau_max={tau_max:.8e} tau_rmse={tau_rmse:.8e} within={ok}"
                );
                if !ok {
                    defects.push(name);
                }
            }
            let shifted_table = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &shifted_ids);
            let transposed_table =
                rotary_rows(spec.rope_freq_dim, spec.rope_theta, &transposed_ids);
            ensure!(
                shifted_table.len() == dit.rotary.len()
                    && transposed_table.len() == dit.rotary.len(),
                "rotated rotary length {} {} {}",
                shifted_table.len(),
                transposed_table.len(),
                dit.rotary.len()
            );
            dit.rotary = s.clone_htod(&shifted_table)?;
            let shift = velocity(&context, 0, &text_d, &mut dit)?;
            dit.rotary = s.clone_htod(&transposed_table)?;
            let control = velocity(&context, 0, &text_d, &mut dit)?;
            let (shift_max, shift_rmse) = image_max_rmse(&shift, &step0_ref)?;
            let (shift_vs_max, shift_vs_rmse) = image_max_rmse(&shift, &step0)?;
            let (control_max, control_rmse) = image_max_rmse(&control, &step0_ref)?;
            let (versus_max, versus_rmse) = image_max_rmse(&control, &step0)?;
            if placement == Placement::Derived {
                drop(dit);
                let mut streamed = ImageDit::new(
                    &device,
                    &mut blas,
                    (&spec, &image_spec),
                    &t2i_layout,
                    &dir,
                    &manifest,
                    (
                        Placement::Streamed,
                        crate::prefill::placement_visits(schedule.timesteps.len(), 1)?,
                    ),
                )?;
                let context = BlockContext {
                    s,
                    blas: &blas,
                    kernels: &kernels,
                    spec: &spec,
                    lora_scale: 1.0,
                };
                let clean = velocity(&context, 0, &text_d, &mut streamed)?;
                let bit_diffs = |actual: &[bf16]| -> Result<usize> {
                    ensure!(
                        actual.len() == residual.len()
                            && actual.iter().chain(&residual).all(|v| v.is_finite()),
                        "invalid full-block residual"
                    );
                    Ok(actual
                        .iter()
                        .zip(&residual)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count())
                };
                let clean_diff = bit_diffs(&s.clone_dtoh(&streamed.x)?)?;
                let velocity_diff = clean
                    .iter()
                    .zip(&step0)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                ensure!(
                    clean.len() == step0.len() && clean_diff == 0 && velocity_diff == 0,
                    "Streamed/Derived residual differences {clean_diff}, velocity differences {velocity_diff}"
                );
                let dense = streamed
                    .tier
                    .as_mut()
                    .context("forced streamed tier missing")?
                    .ready()?;
                ensure!(
                    dense.plan.slots[1].is_some() && dense.plan.slots[1] == dense.plan.slots[3],
                    "control requires layers 1 and 3 to reuse one slot"
                );
                // At the tail of layer 2 the layer-3 slot still holds layer 1.
                streamed.blocks[3].norm1.tensor.ptr = dense.pointer(3, 0, 2 * spec.hidden)?;
                let error =
                    require_resident_norms(s, &streamed.blocks, &streamed.norm_out, spec.hidden)
                        .unwrap_err();
                ensure!(
                    error
                        .to_string()
                        .contains("transformer_blocks.3.norm1.weight"),
                    "wrong residency error: {error}"
                );
                velocity(&context, 0, &text_d, &mut streamed)?;
                let control = bit_diffs(&s.clone_dtoh(&streamed.x)?)?;
                ensure!(
                    control > 0,
                    "full-block residual missed the next-slot norm read"
                );
                println!(
                    "H3_GATED_STREAMED_GATE PASS layers={} residual_elements={} residual_diff={clean_diff} velocity_diff={velocity_diff} next_slot_control={control}",
                    streamed.blocks.len(),
                    residual.len()
                );
            }
            (
                shift_max,
                shift_rmse,
                shift_vs_max,
                shift_vs_rmse,
                control_max,
                control_rmse,
                versus_max,
                versus_rmse,
                changed,
            )
        };
        {
            let table = AdalnTable::build(&device, &mut blas, &dir, &manifest, &spec, &edit_plan)?;
            let mut dit = ImageDit::new(
                &device,
                &mut blas,
                (&spec, &image_spec),
                &edit_layout,
                &dir,
                &manifest,
                (
                    placement,
                    crate::prefill::placement_visits(schedule.timesteps.len(), 1)?,
                ),
            )?;
            let kernels = BlockKernels::new(&device.ctx)?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: 1.0,
            };
            let counts = (
                u32::try_from(edit_plan.pairs.len())?,
                u32::try_from(edit_plan.timesteps.len())?,
            );
            let rows_d = s.clone_htod(&edit_rows)?;
            let text_d = s.clone_htod(&edit_text)?;
            let row_pairs = s.clone_htod(&edit_plan.row_pairs[0])?;
            let row_timesteps = s.clone_htod(&edit_plan.row_timesteps[0])?;
            dit.step(
                &context,
                &table,
                counts,
                0,
                &ImageStepInputs {
                    image: &rows_d,
                    text: &text_d,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            let step0 = s.clone_dtoh(&dit.velocity)?;
            dit.euler(s, &rows_d, &schedule, 0)?;
            let euler = s.clone_dtoh(&rows_d)?;
            let prefix = edit_layout.reference_rows * width;
            let reference_exact = euler[..prefix]
                .iter()
                .zip(&edit_rows[..prefix])
                .all(|(got, want)| got.to_bits() == want.to_bits());
            println!("H3_IMAGE_DIT euler_edit_reference_rows exact={reference_exact}");
            if !reference_exact {
                defects.push("euler_edit_reference_rows");
            }
            for (name, candidate, reference) in [
                ("dit_edit_step0", &step0, &edit_step_ref),
                ("euler_edit_step0", &euler, &edit_euler_ref),
            ] {
                let cell = IMAGE_DIT_512
                    .iter()
                    .find(|cell| cell.name == name)
                    .context(name)?;
                let (max_abs, rmse) = image_max_rmse(candidate, reference)?;
                let (tau_max, tau_rmse) = scored_image_tau(
                    sm,
                    name,
                    image_dit_tau(cell.cross_max_abs),
                    image_dit_tau(cell.cross_rmse),
                );
                let ok = candidate.iter().all(|v| v.is_finite())
                    && max_abs <= tau_max
                    && rmse <= tau_rmse;
                println!(
                    "H3_IMAGE_DIT {name} max_abs={max_abs:.8e} rmse={rmse:.8e} tau_max={tau_max:.8e} tau_rmse={tau_rmse:.8e} within={ok}"
                );
                if !ok {
                    defects.push(name);
                }
            }
        }
        fs::remove_dir_all(&runtime)?;
        let step0_cell = IMAGE_DIT_512
            .iter()
            .find(|cell| cell.name == "dit_step0")
            .context("dit_step0")?;
        let (bar_max, bar_rmse, shift_gt, caught) = if sm < 80 {
            let (bar_max, bar_rmse) = fp16_dit_bound();
            (
                bar_max,
                bar_rmse,
                shift_vs_max > bar_max && shift_vs_rmse > bar_rmse,
                versus_max > bar_max && versus_rmse > bar_rmse,
            )
        } else {
            let bar_max = 3.0 * image_dit_tau(step0_cell.cross_max_abs);
            let bar_rmse = 3.0 * image_dit_tau(step0_cell.cross_rmse);
            (
                bar_max,
                bar_rmse,
                shift_max > bar_max && shift_rmse > bar_rmse,
                control_max > bar_max && control_rmse > bar_rmse,
            )
        };
        println!(
            "H3_IMAGE_DIT rope_shift max_abs={shift_max:.8e} rmse={shift_rmse:.8e} vs_step0_max={shift_vs_max:.8e} vs_step0_rmse={shift_vs_rmse:.8e} bar_max={bar_max:.8e} bar_rmse={bar_rmse:.8e} gt={shift_gt} sample_changed={sample_changed}"
        );
        println!(
            "H3_IMAGE_DIT rope_transpose max_abs={control_max:.8e} rmse={control_rmse:.8e} vs_step0_max={versus_max:.8e} vs_step0_rmse={versus_rmse:.8e} bar_max={bar_max:.8e} bar_rmse={bar_rmse:.8e} gt={caught} sample_changed={sample_changed}"
        );
        ensure!(
            caught,
            "rope-transpose control stayed within {}",
            if sm < 80 { "the FP16 floor" } else { "3x tau" }
        );
        ensure!(sample_changed, "image sample left the input unchanged");
        ensure!(
            defects.is_empty(),
            "H3_IMAGE_DIT_GATE DEFECT cells outside the frozen taus: {defects:?}"
        );
        println!("H3_IMAGE_DIT_GATE PASS cells=5 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    struct Image4096Cell {
        name: &'static str,
        tau_max: f64,
        tau_rmse: f64,
    }

    /// 4096 taus, `max(2 * worst pair, 1e-6)` of the four arms. The frozen reference is the G4 sdpa B=1 arm.
    const IMAGE_DIT_4096: &[Image4096Cell] = &[
        Image4096Cell {
            name: "dit_step0",
            tau_max: 1.0555169582366943,
            tau_rmse: 0.10097579439649207,
        },
        Image4096Cell {
            name: "dit_step5",
            tau_max: 2.0533370971679688,
            tau_rmse: 0.10246208378706335,
        },
        Image4096Cell {
            name: "euler_step0",
            tau_max: 1.6550288200378418,
            tau_rmse: 0.3006873726360839,
        },
    ];

    fn image_f32_shaped(
        file: &SafeTensors<'_>,
        name: &str,
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let view = file.tensor(name).with_context(|| format!("lacks {name}"))?;
        let shape = view.shape();
        ensure!(
            shape == [rows, width] || shape == [1, rows, width],
            "{name} shape {shape:?}"
        );
        let values = image_f32(file, name)?;
        ensure!(
            values.len() == rows * width && values.iter().all(|v| v.is_finite()),
            "{name} length {} or a non-finite value",
            values.len()
        );
        Ok(values)
    }

    /// Canvas 4096, latent 256, 16384 image rows, S=16391. Placement is `Derived`.
    /// Step 0, Euler of a copy of those rows, then step 5. The reference is the G4 sdpa B=1 arm.
    /// Every image row swaps `(h, w)` to `(w, h)`.
    /// On sm >= 80 the control bar is `3 * tau` against the BF16 reference.
    /// On sm < 80 a `dit_*` cell uses the prod64 floor. Euler stays on the frozen 4096 tau.
    /// On sm < 80 the transpose must differ from the unshifted candidate by more than that floor.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_H3_IMAGE, FF_H3_IMAGE_4096"]
    fn image_dit_4096_matches_the_frozen_cells() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let model = path("FF_H3_IMAGE")?;
        let fixture = path("FF_H3_IMAGE_4096")?;
        let image_spec = ImageSpec::read(&model)?;
        let spec = super::image_block_spec(&image_spec);
        let width = image_spec.row_width();
        let tokens = 128usize;
        let latent = tokens * image_spec.patch[1];
        ensure!(
            width == 96 && latent == 256 && image_spec.patch == [1, 2, 2],
            "4096 image geometry width={width} latent={latent} patch={:?}",
            image_spec.patch
        );
        let image_rows = tokens * tokens;
        let capture_map = mapped(&fixture)?;
        let capture = SafeTensors::deserialize(&capture_map)?;
        let tags = image_tags(&capture, "conditioner_t2i_tags")?;
        let text_view = capture.tensor("conditioner_t2i")?;
        ensure!(
            tags.len() == 7
                && tags.iter().all(|&tag| tag == crate::minimax_h3::TEXT_TAG)
                && (text_view.shape() == [1, 7, image_spec.text_dim]
                    || text_view.shape() == [7, image_spec.text_dim]),
            "conditioner tags {:?} text {:?}",
            tags,
            text_view.shape()
        );
        let text = image_bf16_exact(&image_f32(&capture, "conditioner_t2i")?)?;
        let rows_host = image_f32_shaped(&capture, "t2i_rows", image_rows, width)?;
        let step0_ref = image_f32_shaped(&capture, "dit_step0", image_rows, width)?;
        let step5_ref = image_f32_shaped(&capture, "dit_step5", image_rows, width)?;
        let euler_ref = image_f32_shaped(&capture, "euler_step0", image_rows, width)?;
        ensure!(
            text.len() == tags.len() * image_spec.text_dim && rows_host.iter().any(|v| *v != 0.0),
            "4096 fixture extents"
        );
        let layout = image_layout(&image_spec, &tags, &[], latent, latent)?;
        ensure!(
            layout.rows() == tags.len() + image_rows
                && layout.image_indices.len() == image_rows
                && layout.text_indices.len() == tags.len()
                && layout.reference_rows == 0,
            "4096 layout extents"
        );
        ensure!(
            layout
                .text_indices
                .iter()
                .enumerate()
                .all(|(k, &row)| { layout.position_ids[row as usize] == [k as f64, 0.0, 0.0] }),
            "text position ids"
        );
        ensure!(
            (0..tokens).all(|h| {
                (0..tokens).all(|w| {
                    let id = layout.position_ids[tags.len() + h * tokens + w];
                    id[0] == tags.len() as f64
                        && id[1] == h as f64 * 0.25
                        && id[2] == w as f64 * 0.25
                })
            }),
            "spatial position ids"
        );
        let source = layout.position_ids.clone();
        let mut transposed_ids = source.clone();
        for &row in &layout.image_indices {
            let row = row as usize;
            transposed_ids[row][1] = source[row][2];
            transposed_ids[row][2] = source[row][1];
        }
        ensure!(
            transposed_ids != source
                && transposed_ids
                    .iter()
                    .zip(&source)
                    .all(|(got, want)| got[0] == want[0])
                && layout
                    .text_indices
                    .iter()
                    .all(|&row| transposed_ids[row as usize] == source[row as usize])
                && layout.image_indices.iter().all(|&row| {
                    let row = row as usize;
                    transposed_ids[row][1] == source[row][2]
                        && transposed_ids[row][2] == source[row][1]
                }),
            "spatial transpose is not (h, w) to (w, h)"
        );
        let schedule = ImageSchedule::published()?;
        let row_plan =
            AdalnPlan::from_tags(&layout.token_tags, &image_row_plan(&layout, &schedule)?)?;
        ensure!(row_plan.row_pairs.len() == 6, "evaluation count");
        let dir = model.join("transformer_diffusers");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_image_manifest(&image_spec)?;
        let sizing = super::dense_sizing(&manifest, spec.layers)?;
        let owned = spec.layers * 2 * (2 * spec.hidden + 2 * spec.head_dim)
            + sizing.staging
            + sizing.transient;
        let device = Device::new(0)?;
        let sm = cubin::capability(&device.ctx)?;
        println!("H3_IMAGE_4096 sm={sm}");
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-image-dit-4096-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3-Image 4096 DiT gate".into())),
        )?;
        let table = AdalnTable::build(&device, &mut blas, &dir, &manifest, &spec, &row_plan)?;
        let visits = crate::prefill::placement_visits(schedule.timesteps.len(), 1)?;
        let mut dit = ImageDit::new(
            &device,
            &mut blas,
            (&spec, &image_spec),
            &layout,
            &dir,
            &manifest,
            (Placement::Derived, visits),
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 1.0,
        };
        {
            let facts = dit.placement;
            let dense = dit
                .tier
                .as_ref()
                .map(|tier| &tier.dense)
                .context("4096 placement kept every block resident")?;
            ensure!(
                facts.largest_bytes == sizing.largest && facts.owned_bytes == owned,
                "placement facts differ from the image block sizes"
            );
            ensure!(
                facts.host_available_bytes.is_some() && dense.plan.slot_bytes > 0,
                "4096 placement kept every block resident"
            );
            let slot_count = dense
                .plan
                .slots
                .iter()
                .flatten()
                .copied()
                .max()
                .map(|index| index + 1)
                .context("4096 placement kept every block resident")?;
            let (plan, again) = super::plan_dense_layers(
                (
                    &sizing.layers,
                    &sizing.source_layers,
                    sizing.largest,
                    owned,
                    0,
                ),
                Placement::Derived,
                (
                    facts.free_bytes,
                    facts.total_bytes,
                    facts.host_available_bytes,
                ),
                slot_count,
                visits,
            )?;
            let resident = plan.slots.iter().filter(|slot| slot.is_none()).count();
            let host = plan.host.iter().filter(|flag| **flag).count();
            let disk = plan
                .slots
                .iter()
                .zip(&plan.host)
                .filter(|(slot, flag)| slot.is_some() && !**flag)
                .count();
            let streamed = sizing
                .source_layers
                .iter()
                .zip(&plan.slots)
                .filter(|(_, slot)| slot.is_some())
                .map(|(bytes, _)| *bytes)
                .sum::<usize>();
            let host_available = facts
                .host_available_bytes
                .context("4096 placement kept every block resident")?;
            println!(
                "H3_IMAGE_4096_PLACEMENT resident={resident} host={host} disk={disk} free_bytes={} total_bytes={} device_budget={} host_budget={} slot_bytes={} host_available_bytes={host_available}",
                facts.free_bytes,
                facts.total_bytes,
                facts.device_budget,
                facts.host_budget,
                plan.slot_bytes
            );
            ensure!(
                again.device_budget == facts.device_budget
                    && again.host_budget == facts.host_budget
                    && plan.slots == dense.plan.slots
                    && plan.host == dense.plan.host
                    && plan.slot_bytes == dense.plan.slot_bytes
                    && plan.resident_bytes == dense.plan.resident_bytes
                    && plan.host_bytes == dense.plan.host_bytes
                    && plan.disk_bytes == dense.plan.disk_bytes,
                "derived placement disagrees with the live free bytes"
            );
            if facts.host_budget >= streamed {
                ensure!(plan.disk_bytes == 0, "host budget holds the spill");
            }
        }
        let counts = (
            u32::try_from(row_plan.pairs.len())?,
            u32::try_from(row_plan.timesteps.len())?,
        );
        let rows_d = s.clone_htod(&rows_host)?;
        let text_d = s.clone_htod(&text)?;
        let mut defects = Vec::new();
        let mut score = |name: &'static str, candidate: &[f32], reference: &[f32]| -> Result<()> {
            ensure!(
                candidate.len() == image_rows * width && candidate.iter().all(|v| v.is_finite()),
                "{name} length {} or a non-finite value",
                candidate.len()
            );
            let cell = IMAGE_DIT_4096
                .iter()
                .find(|cell| cell.name == name)
                .context(name)?;
            let (max_abs, rmse) = image_max_rmse(candidate, reference)?;
            let (tau_max, tau_rmse) = scored_image_tau(sm, name, cell.tau_max, cell.tau_rmse);
            let ok = max_abs <= tau_max && rmse <= tau_rmse;
            println!(
                "H3_IMAGE_4096 {name} max_abs={max_abs:.8e} rmse={rmse:.8e} tau_max={tau_max:.8e} tau_rmse={tau_rmse:.8e} within={ok}"
            );
            if !ok {
                defects.push(name);
            }
            Ok(())
        };
        let velocity = |head: usize, dit: &mut ImageDit| -> Result<Vec<f32>> {
            let row_pairs = s.clone_htod(&row_plan.row_pairs[head])?;
            let row_timesteps = s.clone_htod(&row_plan.row_timesteps[head])?;
            dit.step(
                &context,
                &table,
                counts,
                head,
                &ImageStepInputs {
                    image: &rows_d,
                    text: &text_d,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            s.clone_dtoh(&dit.velocity)
                .context("copy DiT velocity to host")
        };
        let step0 = velocity(0, &mut dit)?;
        score("dit_step0", &step0, &step0_ref)?;
        let euler_rows = s.clone_htod(&rows_host)?;
        dit.euler(s, &euler_rows, &schedule, 0)?;
        let euler = s.clone_dtoh(&euler_rows)?;
        score("euler_step0", &euler, &euler_ref)?;
        let step5 = velocity(5, &mut dit)?;
        score("dit_step5", &step5, &step5_ref)?;
        let transposed = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &transposed_ids);
        ensure!(
            transposed.len() == dit.rotary.len(),
            "transposed rotary length {} {}",
            transposed.len(),
            dit.rotary.len()
        );
        dit.rotary = s.clone_htod(&transposed)?;
        let control = velocity(0, &mut dit)?;
        ensure!(
            control.len() == image_rows * width && control.iter().all(|v| v.is_finite()),
            "rope_transpose length {} or a non-finite value",
            control.len()
        );
        let (control_max, control_rmse) = image_max_rmse(&control, &step0_ref)?;
        let (versus_max, versus_rmse) = image_max_rmse(&control, &step0)?;
        drop(score);
        fs::remove_dir_all(&runtime)?;
        let step0_cell = IMAGE_DIT_4096
            .iter()
            .find(|cell| cell.name == "dit_step0")
            .context("dit_step0")?;
        let (bar_max, bar_rmse, caught) = if sm < 80 {
            let (bar_max, bar_rmse) = fp16_dit_bound();
            (
                bar_max,
                bar_rmse,
                versus_max > bar_max && versus_rmse > bar_rmse,
            )
        } else {
            let bar_max = 3.0 * step0_cell.tau_max;
            let bar_rmse = 3.0 * step0_cell.tau_rmse;
            (
                bar_max,
                bar_rmse,
                control_max > bar_max && control_rmse > bar_rmse,
            )
        };
        println!(
            "H3_IMAGE_4096 rope_transpose max_abs={control_max:.8e} rmse={control_rmse:.8e} vs_step0_max={versus_max:.8e} vs_step0_rmse={versus_rmse:.8e} bar_max={bar_max:.8e} bar_rmse={bar_rmse:.8e} gt={caught}"
        );
        ensure!(
            caught,
            "rope-transpose control stayed within {}",
            if sm < 80 { "the FP16 floor" } else { "3x tau" }
        );
        ensure!(
            defects.is_empty(),
            "H3_IMAGE_4096_GATE DEFECT cells outside the frozen taus: {defects:?}"
        );
        println!("H3_IMAGE_4096_GATE PASS cells=3 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Canvas 2048, latent 128, 4096 image rows. Placement is `Streamed`.
    /// One step of constant 0.1. The velocity is finite. Two device slots and no resident block.
    /// Host pins follow the sensed host budget. The counted re-read is `disk_layers * slot_bytes * 6`.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_H3_IMAGE"]
    fn image_dit_2048_streamed_admission() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let model = PathBuf::from(std::env::var_os("FF_H3_IMAGE").context("FF_H3_IMAGE missing")?);
        let image_spec = ImageSpec::read(&model)?;
        let spec = super::image_block_spec(&image_spec);
        let width = image_spec.row_width();
        let tokens = 64usize;
        let latent = tokens * image_spec.patch[1];
        ensure!(
            width == 96 && latent == 128 && image_spec.patch == [1, 2, 2] && spec.layers == 50,
            "2048 image geometry width={width} latent={latent} layers={} patch={:?}",
            spec.layers,
            image_spec.patch
        );
        let image_rows = tokens * tokens;
        let tags = vec![TEXT_TAG; 7];
        let text = vec![bf16::from_f32(0.1); tags.len() * image_spec.text_dim];
        let rows_host = vec![0.1f32; image_rows * width];
        let layout = image_layout(&image_spec, &tags, &[], latent, latent)?;
        ensure!(
            layout.image_indices.len() == image_rows
                && layout.text_indices.len() == tags.len()
                && layout.reference_rows == 0,
            "2048 layout extents"
        );
        let schedule = ImageSchedule::published()?;
        let row_plan =
            AdalnPlan::from_tags(&layout.token_tags, &image_row_plan(&layout, &schedule)?)?;
        let steps = row_plan.row_pairs.len();
        ensure!(steps == 6, "evaluation count");
        let dir = model.join("transformer_diffusers");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_image_manifest(&image_spec)?;
        let device = Device::new(0)?;
        let sm = cubin::capability(&device.ctx)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-image-2048-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3-Image 2048 admission".into())),
        )?;
        let visits = crate::prefill::placement_visits(schedule.timesteps.len(), 1)?;
        let table = AdalnTable::build(&device, &mut blas, &dir, &manifest, &spec, &row_plan)?;
        let mut dit = ImageDit::new(
            &device,
            &mut blas,
            (&spec, &image_spec),
            &layout,
            &dir,
            &manifest,
            (Placement::Streamed, visits),
        )?;
        let host_budget = dit.placement.host_budget;
        let plan = dit
            .tier
            .as_ref()
            .context("2048 streamed tier missing")?
            .dense
            .plan
            .clone();
        let host_pins = plan.host.iter().filter(|flag| **flag).count();
        let disk_layers = plan
            .slots
            .iter()
            .zip(&plan.host)
            .filter(|(slot, flag)| slot.is_some() && !**flag)
            .count();
        let resident = plan.slots.iter().filter(|slot| slot.is_none()).count();
        let slot_n = plan
            .slots
            .iter()
            .flatten()
            .copied()
            .max()
            .map(|index| index + 1)
            .context("2048 streamed slots missing")?;
        let reread = disk_layers
            .checked_mul(plan.slot_bytes)
            .and_then(|bytes| bytes.checked_mul(steps))
            .context("2048 re-read bytes overflow")?;
        println!(
            "H3_IMAGE_2048_PLACEMENT sm={sm} resident={resident} resident_bytes={} host_pins={host_pins} disk_layers={disk_layers} slot_bytes={} slot_count={slot_n} host_bytes={} disk_bytes={} host_budget={host_budget} visits={visits} reread_bytes={reread}",
            plan.resident_bytes, plan.slot_bytes, plan.host_bytes, plan.disk_bytes
        );
        ensure!(
            plan.slot_bytes == 770_703_360
                && plan.resident_bytes == 0
                && resident == 0
                && slot_n == 2
                && host_pins + disk_layers == spec.layers,
            "2048 streamed placement slot_bytes={} resident_bytes={} resident={resident} slots={slot_n} host_pins={host_pins} disk_layers={disk_layers}",
            plan.slot_bytes,
            plan.resident_bytes
        );
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 1.0,
        };
        let counts = (
            u32::try_from(row_plan.pairs.len())?,
            u32::try_from(row_plan.timesteps.len())?,
        );
        let rows_d = s.clone_htod(&rows_host)?;
        let text_d = s.clone_htod(&text)?;
        let row_pairs = s.clone_htod(&row_plan.row_pairs[0])?;
        let row_timesteps = s.clone_htod(&row_plan.row_timesteps[0])?;
        dit.step(
            &context,
            &table,
            counts,
            0,
            &ImageStepInputs {
                image: &rows_d,
                text: &text_d,
                row_pairs: &row_pairs,
                row_timesteps: &row_timesteps,
            },
        )?;
        let velocity = s.clone_dtoh(&dit.velocity)?;
        drop(dit);
        fs::remove_dir_all(&runtime)?;
        ensure!(
            velocity.len() == image_rows * width && velocity.iter().all(|value| value.is_finite()),
            "2048 velocity length {} or a non-finite value",
            velocity.len()
        );
        println!("H3_IMAGE_2048_ADMISSION finite=true rows={image_rows} width={width} sm={sm}");
        Ok(())
    }

    struct ImageE2eCell {
        name: &'static str,
        tau_max: f64,
        tau_rmse: f64,
        tau_rel: f64,
        gate_max: bool,
        gate_rmse: bool,
        gate_rel: bool,
        uncross_max: f64,
        uncross_rmse: f64,
    }

    /// Taus are `max(2 * pair, 1e-6)`. t2i_conditioner and t2i cell (b) are the 2026-11-13 isolated correction.
    /// Edit cells and cell (e) stay the 2026-11-12 recapture.
    /// t2i `uncross_*` is the isolated arm-A versus MATH pair. It is 0 when a cell has no uncross figure.
    /// `gate_max` is false where the max_abs tau exceeds rms(arm A).
    const IMAGE_E2E: &[ImageE2eCell] = &[
        ImageE2eCell {
            name: "t2i_conditioner",
            tau_max: 4.0,
            tau_rmse: 0.11145047448683923,
            tau_rel: 0.0013261567336856647,
            gate_max: true,
            gate_rmse: true,
            gate_rel: true,
            uncross_max: 0.0,
            uncross_rmse: 0.0,
        },
        ImageE2eCell {
            name: "edit_conditioner",
            tau_max: 1.408e3,
            tau_rmse: 1.2445085138788345e+00,
            tau_rel: 1.0412441562422288e-01,
            gate_max: false,
            gate_rmse: true,
            gate_rel: true,
            uncross_max: 0.0,
            uncross_rmse: 0.0,
        },
        ImageE2eCell {
            name: "t2i_step0",
            tau_max: 0.02681148052215576,
            tau_rmse: 0.0015735558732471566,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.011798381805419922,
            uncross_rmse: 0.0006590668803632128,
        },
        ImageE2eCell {
            name: "t2i_step1",
            tau_max: 0.12804865837097168,
            tau_rmse: 0.0056322554985020635,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.06402432918548584,
            uncross_rmse: 0.0028161277492510317,
        },
        ImageE2eCell {
            name: "t2i_step2",
            tau_max: 0.27604198455810547,
            tau_rmse: 0.007397312102626611,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.13802099227905273,
            uncross_rmse: 0.0036986560513133054,
        },
        ImageE2eCell {
            name: "t2i_step3",
            tau_max: 0.2156376838684082,
            tau_rmse: 0.005701600502433918,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.1078188419342041,
            uncross_rmse: 0.002850800251216959,
        },
        ImageE2eCell {
            name: "t2i_step4",
            tau_max: 0.3148576021194458,
            tau_rmse: 0.0060046395838299835,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.1574288010597229,
            uncross_rmse: 0.0030023197919149917,
        },
        ImageE2eCell {
            name: "t2i_step5",
            tau_max: 0.1341700553894043,
            tau_rmse: 0.00342396006666815,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.06708502769470215,
            uncross_rmse: 0.001711980033334075,
        },
        ImageE2eCell {
            name: "t2i_latent",
            tau_max: 0.1341700553894043,
            tau_rmse: 0.00342396006666815,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.06708502769470215,
            uncross_rmse: 0.001711980033334075,
        },
        ImageE2eCell {
            name: "edit_step0",
            tau_max: 4.474771022796631e-2,
            tau_rmse: 3.210507202025054e-3,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 6.309986114501953e-2,
            uncross_rmse: 3.511065910889702e-3,
        },
        ImageE2eCell {
            name: "edit_step1",
            tau_max: 4.0012475848197937e-01,
            tau_rmse: 1.505041278553131e-2,
            tau_rel: 0.0,
            gate_max: true,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 2.748512029647827e-1,
            uncross_rmse: 1.967127688184552e-2,
        },
        ImageE2eCell {
            name: "edit_step2",
            tau_max: 1.4161487221717834e+00,
            tau_rmse: 4.996969378738491e-2,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 7.28495180606842e-1,
            uncross_rmse: 4.290312446688843e-2,
        },
        ImageE2eCell {
            name: "edit_step3",
            tau_max: 3.1202070713043213e+00,
            tau_rmse: 1.1337469423903028e-01,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 1.487925112247467,
            uncross_rmse: 8.683005638543116e-2,
        },
        ImageE2eCell {
            name: "edit_step4",
            tau_max: 5.407879829406738,
            tau_rmse: 2.0802742376685993e-01,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 2.5422056913375854e+00,
            uncross_rmse: 1.5130835032512951e-01,
        },
        ImageE2eCell {
            name: "edit_step5",
            tau_max: 7.03294825553894,
            tau_rmse: 2.8209972577655884e-01,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 3.3823635578155518e+00,
            uncross_rmse: 2.022746474277021e-1,
        },
        ImageE2eCell {
            name: "edit_latent",
            tau_max: 7.03294825553894,
            tau_rmse: 2.820997257765589e-1,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 3.3823635578155518e+00,
            uncross_rmse: 2.022746474277021e-1,
        },
        ImageE2eCell {
            name: "t2i_rgb",
            tau_max: 0.0,
            tau_rmse: 1.465530558268238e-1,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.0,
            uncross_rmse: 0.0,
        },
        ImageE2eCell {
            name: "edit_rgb",
            tau_max: 0.0,
            tau_rmse: 1.406691362579064e-1,
            tau_rel: 0.0,
            gate_max: false,
            gate_rmse: true,
            gate_rel: false,
            uncross_max: 0.0,
            uncross_rmse: 0.0,
        },
    ];

    const IMAGE_E2E_T2I_NAMES: [&str; 7] = [
        "t2i_step0",
        "t2i_step1",
        "t2i_step2",
        "t2i_step3",
        "t2i_step4",
        "t2i_step5",
        "t2i_latent",
    ];
    const IMAGE_E2E_EDIT_NAMES: [&str; 7] = [
        "edit_step0",
        "edit_step1",
        "edit_step2",
        "edit_step3",
        "edit_step4",
        "edit_step5",
        "edit_latent",
    ];
    const IMAGE_E2E_PROMPT_T2I: &str = "a red cube on a white table";
    const IMAGE_E2E_PROMPT_EDIT: &str = "replace the left half of <Picture 1> with a blue square";
    const IMAGE_E2E_SEED: u64 = 20261112;
    const IMAGE_E2E_CONTROL_SEED: u64 = 20261113;

    fn image_e2e_cell(name: &'static str) -> Result<&'static ImageE2eCell> {
        IMAGE_E2E
            .iter()
            .find(|cell| cell.name == name)
            .context(name)
    }

    fn image_e2e_env(key: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(
            std::env::var_os(key).with_context(|| format!("{key} missing"))?,
        ))
    }

    fn image_e2e_npy(path: &Path, shape: &[usize]) -> Result<Vec<f32>> {
        let bytes = fs::read(path).with_context(|| format!("npy {}", path.display()))?;
        ensure!(
            bytes.len() >= 10 && bytes[..6] == *b"\x93NUMPY",
            "npy magic"
        );
        let (header_len, header_at) = match bytes[6] {
            1 => (
                u16::from_le_bytes(bytes[8..10].try_into().context("npy header")?) as usize,
                10,
            ),
            2 | 3 => (
                u32::from_le_bytes(bytes[8..12].try_into().context("npy header")?) as usize,
                12,
            ),
            major => bail!("npy version {major}"),
        };
        let header = std::str::from_utf8(
            bytes
                .get(header_at..header_at + header_len)
                .context("npy header length")?,
        )
        .context("npy header")?;
        ensure!(
            header.contains("'<f4'") && header.contains("'fortran_order': False"),
            "npy descr"
        );
        let shape_at = header.find("'shape':").context("npy shape")?;
        let rest = &header[shape_at..];
        let open = rest.find('(').context("npy shape")?;
        let close = rest.find(')').context("npy shape")?;
        let mut dims = Vec::new();
        for part in rest[open + 1..close].split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            dims.push(
                part.parse::<usize>()
                    .with_context(|| format!("npy dim {part}"))?,
            );
        }
        ensure!(dims.as_slice() == shape, "npy shape {dims:?}");
        let data = &bytes[header_at + header_len..];
        let (chunks, rest) = data.as_chunks::<4>();
        ensure!(rest.is_empty(), "npy trailing bytes");
        let values: Vec<f32> = chunks
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        let count: usize = shape.iter().product();
        ensure!(
            values.len() == count && values.iter().all(|value| value.is_finite()),
            "npy values"
        );
        Ok(values)
    }

    fn image_e2e_tags(path: &Path, len: usize) -> Result<Vec<u8>> {
        image_e2e_npy(path, &[len])?
            .into_iter()
            .map(|value| {
                ensure!(value == 0.0 || value == 1.0, "tag {value}");
                Ok(value as u8)
            })
            .collect()
    }

    fn image_e2e_steps(dir: &Path, case: &str) -> Result<[Vec<f32>; 6]> {
        let load = |index: usize| {
            image_e2e_npy(
                &dir.join(format!("{case}-latent-step{index}.npy")),
                &[4096, 96],
            )
        };
        Ok([load(0)?, load(1)?, load(2)?, load(3)?, load(4)?, load(5)?])
    }

    fn image_e2e_rms(values: &[f32]) -> Result<f64> {
        ensure!(
            !values.is_empty() && values.iter().all(|value| value.is_finite()),
            "rms"
        );
        let sum: f64 = values
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum();
        let rms = (sum / values.len() as f64).sqrt();
        ensure!(rms.is_finite() && rms > 0.0, "rms");
        Ok(rms)
    }

    fn image_e2e_judge(
        cell: &ImageE2eCell,
        label: &'static str,
        candidate: &[f32],
        reference: &[f32],
        rms_a: f64,
    ) -> Result<bool> {
        ensure!(
            candidate.len() == reference.len()
                && candidate.iter().all(|value| value.is_finite())
                && reference.iter().all(|value| value.is_finite()),
            "{label} length {} or a non-finite value",
            candidate.len()
        );
        let (max_abs, rmse) = image_max_rmse(candidate, reference)?;
        let rel = rmse / rms_a;
        ensure!(rel.is_finite(), "{label} relative rmse");
        let within = (!cell.gate_max || max_abs <= cell.tau_max)
            && (!cell.gate_rmse || rmse <= cell.tau_rmse)
            && (!cell.gate_rel || rel <= cell.tau_rel);
        if cell.uncross_max != 0.0 || cell.uncross_rmse != 0.0 {
            println!(
                "H3_IMAGE_E2E {label} max_abs={max_abs:.16e} rmse={rmse:.16e} rel={rel:.16e} tau_max={:.16e} tau_rmse={:.16e} tau_rel={:.16e} gate_max={} gate_rmse={} gate_rel={} uncross_max={:.16e} uncross_rmse={:.16e} within={within}",
                cell.tau_max,
                cell.tau_rmse,
                cell.tau_rel,
                cell.gate_max,
                cell.gate_rmse,
                cell.gate_rel,
                cell.uncross_max,
                cell.uncross_rmse,
            );
        } else {
            println!(
                "H3_IMAGE_E2E {label} max_abs={max_abs:.16e} rmse={rmse:.16e} rel={rel:.16e} tau_max={:.16e} tau_rmse={:.16e} tau_rel={:.16e} gate_max={} gate_rmse={} gate_rel={} within={within}",
                cell.tau_max,
                cell.tau_rmse,
                cell.tau_rel,
                cell.gate_max,
                cell.gate_rmse,
                cell.gate_rel,
            );
        }
        Ok(within)
    }

    fn image_e2e_keep(defects: &mut Vec<&'static str>, label: &'static str, within: bool) {
        if !within {
            defects.push(label);
        }
    }

    fn image_e2e_geometry(spec: &ImageSpec) -> Result<()> {
        let (canvas_h, canvas_w) = image_canvas(spec, 1024, 1024, IMAGE_MIN_GENERATION_AREA)?;
        let (latent_h, latent_w) = image_latent_extent(spec, canvas_h, canvas_w)?;
        let (ref_w, ref_h) = image_vae_reference_size(spec, 1536, 1024)?;
        let (crop_h, crop_w) = image_reference_latent_extent(spec, ref_h, ref_w)?;
        let reference_rows = (crop_h / spec.patch[1]) * (crop_w / spec.patch[2]);
        ensure!(
            canvas_h == 2048
                && canvas_w == 2048
                && latent_h == 128
                && latent_w == 128
                && spec.in_channels == 24
                && spec.patch == [1, 2, 2]
                && spec.row_width() == 96
                && spec.text_dim == 5120
                && ref_w == 1536
                && ref_h == 1024
                && crop_h == 64
                && crop_w == 96
                && reference_rows == 1536,
            "e2e geometry canvas={canvas_h}x{canvas_w} latent={latent_h}x{latent_w} ref={ref_w}x{ref_h} crop={crop_h}x{crop_w} rows={reference_rows} width={} patch={:?} text_dim={}",
            spec.row_width(),
            spec.patch,
            spec.text_dim
        );
        Ok(())
    }

    fn image_e2e_noise_equal(root: &Path) -> Result<()> {
        for (name, shape) in [
            ("t2i-generated-noise.npy", &[1usize, 24, 1, 128, 128][..]),
            ("edit-generated-noise.npy", &[1, 24, 1, 128, 128][..]),
            ("edit-posterior-eps.npy", &[1, 24, 64, 96][..]),
            ("edit-reference-noise.npy", &[1, 24, 64, 96][..]),
        ] {
            let left = image_e2e_npy(&root.join("arm-a").join(name), shape)?;
            let right = image_e2e_npy(&root.join("arm-b").join(name), shape)?;
            ensure!(left == right, "{name} arms differ");
        }
        Ok(())
    }

    fn image_e2e_patch(values: &[f32], shape: (usize, usize, usize, usize)) -> Result<Vec<f32>> {
        let (channels, frames, height, width) = shape;
        ensure!(
            values.len() == channels * frames * height * width,
            "patchify extent"
        );
        let rows = patchify_latents(values, shape, [1, 2, 2]);
        ensure!(rows.len() == values.len(), "patchify length");
        Ok(rows)
    }

    fn image_e2e_mix(latent: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        ensure!(latent.len() == noise.len(), "mix extent");
        let scale = KEYFRAME_NOISE_AUG as f32;
        let mixed: Vec<f32> = latent
            .iter()
            .zip(noise)
            .map(|(&value, &draw)| scale * value + (1.0 - scale) * draw)
            .collect();
        ensure!(
            mixed.iter().all(|value| value.is_finite()),
            "non-finite reference mix"
        );
        Ok(mixed)
    }

    fn image_e2e_reference() -> crate::serve::image::RgbImage {
        let (width, height) = (1536usize, 1024usize);
        let mut pixels = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((3 * x + y) % 256) as u8);
                pixels.push(((x + 5 * y) % 256) as u8);
                pixels.push(((7 * x + 11 * y) % 256) as u8);
            }
        }
        crate::serve::image::RgbImage {
            width,
            height,
            pixels,
        }
    }

    fn image_e2e_request(prompt: &str, seed: u64, edit: bool) -> crate::minimax_h3::ImageRequest {
        let mut request = crate::minimax_h3::ImageRequest::new(prompt, 1024, 1024, seed);
        if edit {
            request.references.push(crate::minimax_h3::ImageReference {
                image: image_e2e_reference(),
            });
        }
        request
    }

    fn image_e2e_runtime(label: &str) -> Result<PathBuf> {
        let dir =
            std::env::temp_dir().join(format!("ff-h3-image-e2e-{label}-{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir(&dir)?;
        Ok(dir)
    }

    fn image_e2e_nchw(rgb: &[u8], height: usize, width: usize) -> Result<Vec<f32>> {
        ensure!(rgb.len() == height * width * 3, "rgb extent");
        let plane = height * width;
        let mut out = vec![0f32; 3 * plane];
        for y in 0..height {
            for x in 0..width {
                let pixel = (y * width + x) * 3;
                for channel in 0..3 {
                    out[channel * plane + y * width + x] = f32::from(rgb[pixel + channel]) / 255.0;
                }
            }
        }
        Ok(out)
    }

    struct ImageE2eTeacher<'a> {
        model: &'a Path,
        spec: &'a ImageSpec,
        runtime: &'a Path,
        layout: &'a ImageLayout,
        rows: &'a [f32],
        text: &'a [bf16],
        steps: &'a [Vec<f32>; 6],
        latent: &'a [f32],
        names: &'a [&'static str; 7],
    }

    fn image_e2e_teacher(job: &ImageE2eTeacher<'_>, defects: &mut Vec<&'static str>) -> Result<()> {
        let width = job.spec.row_width();
        let mut trace = Vec::new();
        // Entering rows cover the image when reference_rows is 0.
        let entering = if job.layout.reference_rows == 0 {
            let mut prior = Vec::with_capacity(6);
            prior.push(job.rows.to_vec());
            for rows in &job.steps[..5] {
                ensure!(rows.len() == job.rows.len(), "t2i entering rows");
                prior.push(rows.clone());
            }
            Some(prior)
        } else {
            None
        };
        let sampled = super::image_sample_trace(
            job.model,
            job.layout,
            (0, job.runtime),
            job.rows,
            job.text,
            Some(&mut trace),
            entering.as_deref(),
        )?;
        let prefix = job.layout.reference_rows * width;
        let generated = job.layout.image_indices.len() - job.layout.reference_rows;
        ensure!(
            sampled.len() == job.rows.len() && sampled[..prefix] == job.rows[..prefix],
            "sample moved a reference row"
        );
        ensure!(
            trace.len() == 6 && trace.iter().all(|rows| rows.len() == generated * width),
            "image step trace"
        );
        ensure!(
            trace[5].as_slice() == &sampled[prefix..],
            "step 5 differs from the returned rows"
        );
        for (index, (step, reference)) in trace.iter().zip(job.steps.iter()).enumerate() {
            let name = job.names[index];
            let cell = image_e2e_cell(name)?;
            let rms = image_e2e_rms(reference)?;
            let within = image_e2e_judge(cell, name, step, reference, rms)?;
            image_e2e_keep(defects, name, within);
        }
        let values = unpatchify_latents(
            &sampled[prefix..],
            (job.spec.in_channels, 1, 128, 128),
            job.spec.patch,
        );
        let cell = image_e2e_cell(job.names[6])?;
        let rms = image_e2e_rms(job.latent)?;
        let within = image_e2e_judge(cell, job.names[6], &values, job.latent, rms)?;
        image_e2e_keep(defects, job.names[6], within);
        Ok(())
    }

    fn image_e2e_finish(runtime: &Path, outcome: Result<()>) -> Result<()> {
        fs::remove_dir_all(runtime)?;
        outcome
    }

    fn image_e2e_run_teacher(
        model: &Path,
        spec: &ImageSpec,
        arm: &Path,
        edit: bool,
        defects: &mut Vec<&'static str>,
    ) -> Result<()> {
        let case = if edit { "edit" } else { "t2i" };
        let tags = image_e2e_tags(
            &arm.join(format!("{case}-tags.npy")),
            if edit { 1557 } else { 7 },
        )?;
        if !edit {
            ensure!(tags.iter().all(|&tag| tag == TEXT_TAG), "t2i tags");
        }
        let references: &[(usize, usize)] = if edit { &[(64, 96)] } else { &[] };
        let layout = image_layout(spec, &tags, references, 128, 128)?;
        ensure!(
            layout.reference_rows == if edit { 1536 } else { 0 }
                && layout.image_indices.len() == layout.reference_rows + 4096,
            "{case} layout rows"
        );
        let conditioner = image_e2e_npy(
            &arm.join(format!("{case}-conditioner.npy")),
            if edit {
                &[1usize, 1557, 5120][..]
            } else {
                &[1, 7, 5120][..]
            },
        )?;
        ensure!(
            conditioner.len() == tags.len() * spec.text_dim,
            "{case} conditioner rows"
        );
        let text = image_bf16_exact(&conditioner)?;
        let generated = image_e2e_patch(
            &image_e2e_npy(
                &arm.join(format!("{case}-generated-noise.npy")),
                &[1, 24, 1, 128, 128],
            )?,
            (24, 1, 128, 128),
        )?;
        ensure!(
            generated.len() == 4096 * spec.row_width(),
            "{case} generated rows"
        );
        let rows = if edit {
            let mixed = image_e2e_mix(
                &image_e2e_npy(&arm.join("edit-reference-latent.npy"), &[1, 24, 64, 96])?,
                &image_e2e_npy(&arm.join("edit-reference-noise.npy"), &[1, 24, 64, 96])?,
            )?;
            let mut packed = image_e2e_patch(&mixed, (24, 1, 64, 96))?;
            ensure!(
                packed.len() == 1536 * spec.row_width(),
                "edit reference rows"
            );
            packed.extend(generated);
            packed
        } else {
            generated
        };
        ensure!(
            rows.len() == layout.image_indices.len() * spec.row_width(),
            "{case} sample rows"
        );
        let steps = image_e2e_steps(arm, case)?;
        let latent = image_e2e_npy(&arm.join(format!("{case}-latent.npy")), &[1, 24, 128, 128])?;
        let names = if edit {
            &IMAGE_E2E_EDIT_NAMES
        } else {
            &IMAGE_E2E_T2I_NAMES
        };
        let runtime = image_e2e_runtime(&format!("{case}-b"))?;
        let outcome = image_e2e_teacher(
            &ImageE2eTeacher {
                model,
                spec,
                runtime: &runtime,
                layout: &layout,
                rows: &rows,
                text: &text,
                steps: &steps,
                latent: &latent,
                names,
            },
            defects,
        );
        image_e2e_finish(&runtime, outcome)
    }

    fn image_e2e_noise(arm: &Path, edit: bool) -> Result<Vec<f32>> {
        let generated = image_e2e_patch(
            &image_e2e_npy(
                &arm.join(if edit {
                    "edit-generated-noise.npy"
                } else {
                    "t2i-generated-noise.npy"
                }),
                &[1, 24, 1, 128, 128],
            )?,
            (24, 1, 128, 128),
        )?;
        if !edit {
            return Ok(generated);
        }
        let epsilon = image_e2e_npy(&arm.join("edit-posterior-eps.npy"), &[1, 24, 64, 96])?;
        let mix = image_e2e_npy(&arm.join("edit-reference-noise.npy"), &[1, 24, 64, 96])?;
        ensure!(
            epsilon.len() == 24 * 64 * 96
                && mix.len() == epsilon.len()
                && generated.len() == 4096 * 96,
            "edit noise extent"
        );
        let mut noise = epsilon;
        noise.extend(generated);
        noise.extend(mix);
        Ok(noise)
    }

    fn image_e2e_rgb_values(rgb: &[u8], reference: &[f32]) -> Result<Vec<f32>> {
        ensure!(rgb.len() == 1024 * 1024 * 3, "rgb bytes");
        let values = image_e2e_nchw(rgb, 1024, 1024)?;
        ensure!(values.len() == reference.len(), "rgb extent");
        Ok(values)
    }

    fn image_e2e_run_pass(
        model: &Path,
        arm_a: &Path,
        arm_b: &Path,
        edit: bool,
        defects: &mut Vec<&'static str>,
    ) -> Result<bool> {
        let case = if edit { "edit" } else { "t2i" };
        let shape: &[usize] = if edit {
            &[1, 1557, 5120]
        } else {
            &[1, 7, 5120]
        };
        let left = image_e2e_npy(&arm_a.join(format!("{case}-conditioner.npy")), shape)?;
        let right = image_e2e_npy(&arm_b.join(format!("{case}-conditioner.npy")), shape)?;
        ensure!(left.len() == right.len(), "{case} conditioner pair");
        let rgb = image_e2e_npy(&arm_a.join(format!("{case}-rgb.npy")), &[1, 3, 1024, 1024])?;
        let noise = image_e2e_noise(arm_a, edit)?;
        let prompt = if edit {
            IMAGE_E2E_PROMPT_EDIT
        } else {
            IMAGE_E2E_PROMPT_T2I
        };
        let request = image_e2e_request(prompt, IMAGE_E2E_SEED, edit);
        let runtime = image_e2e_runtime(&format!("{case}-e"))?;
        let outcome = (|| {
            let options = crate::minimax_h3::ImageRunOptions {
                device: 0,
                runtime_dir: runtime.clone(),
            };
            let (latents, text) =
                super::pipeline::image_pass(model, &options, &request, Some(&noise))?;
            ensure!(
                latents.channels == 24 && latents.height == 128 && latents.width == 128,
                "{case} latent extent"
            );
            let got: Vec<f32> = text.iter().copied().map(|value| value.to_f32()).collect();
            let rms = image_e2e_rms(&left)?;
            let cell_name = if edit {
                "edit_conditioner"
            } else {
                "t2i_conditioner"
            };
            let cell = image_e2e_cell(cell_name)?;
            let (label_a, label_b) = if edit {
                ("edit_conditioner_a", "edit_conditioner_b")
            } else {
                ("t2i_conditioner_a", "t2i_conditioner_b")
            };
            image_e2e_keep(
                defects,
                label_a,
                image_e2e_judge(cell, label_a, &got, &left, rms)?,
            );
            image_e2e_keep(
                defects,
                label_b,
                image_e2e_judge(cell, label_b, &got, &right, rms)?,
            );
            let rgb_name = if edit { "edit_rgb" } else { "t2i_rgb" };
            let values = image_e2e_rgb_values(&latents.rgb, &rgb)?;
            image_e2e_judge(
                image_e2e_cell(rgb_name)?,
                rgb_name,
                &values,
                &rgb,
                image_e2e_rms(&rgb)?,
            )
        })();
        fs::remove_dir_all(&runtime)?;
        outcome
    }

    fn image_e2e_run_control(model: &Path, arm_a: &Path, edit: bool) -> Result<f64> {
        let case = if edit { "edit" } else { "t2i" };
        let rgb = image_e2e_npy(&arm_a.join(format!("{case}-rgb.npy")), &[1, 3, 1024, 1024])?;
        let prompt = if edit {
            IMAGE_E2E_PROMPT_EDIT
        } else {
            IMAGE_E2E_PROMPT_T2I
        };
        let request = image_e2e_request(prompt, IMAGE_E2E_CONTROL_SEED, edit);
        let runtime = image_e2e_runtime(&format!("{case}-c"))?;
        let outcome = (|| {
            let options = crate::minimax_h3::ImageRunOptions {
                device: 0,
                runtime_dir: runtime.clone(),
            };
            let latents = super::image_latents(model, &options, &request)?;
            ensure!(
                latents.channels == 24 && latents.height == 128 && latents.width == 128,
                "{case} control latent extent"
            );
            let values = image_e2e_rgb_values(&latents.rgb, &rgb)?;
            Ok(image_max_rmse(&values, &rgb)?.1)
        })();
        fs::remove_dir_all(&runtime)?;
        outcome
    }

    /// Control bar dated 2026-11-12: RGB rmse must exceed `3 * tau`.
    fn image_e2e_rgb_decision(
        defects: &mut Vec<&'static str>,
        label: &'static str,
        scored_within: bool,
        control_rmse: f64,
    ) -> Result<&'static str> {
        let cell = image_e2e_cell(label)?;
        let bar = 3.0 * cell.tau_rmse;
        let clears = control_rmse > bar;
        let decision = if clears { "gated" } else { "informational" };
        println!(
            "H3_IMAGE_E2E {label}_control rmse={control_rmse:.16e} bar={bar:.16e} clears={clears} decision={decision} scored_within={scored_within}"
        );
        if clears && !scored_within {
            defects.push(label);
        }
        Ok(decision)
    }

    #[test]
    fn resident_dense_placement_charges_the_next_stage() -> Result<()> {
        let layers = [8; 3];
        let total = 256;
        let fixed = 4;
        let later = 32;
        let free = total / 16 + fixed + later + 16;
        let host = Some((4 << 30) + 16 + 8 + 8);
        let (plan, facts) = super::plan_dense_layers(
            (&layers, &layers, 8, fixed, later),
            super::Placement::Derived,
            (free, total, host),
            2,
            2,
        )?;
        assert_eq!(facts.device_budget, 16);
        assert_eq!(plan.resident_bytes, 0);
        assert_eq!(plan.host_bytes, 8);
        assert_eq!(plan.disk_bytes, 16);
        assert!(
            super::plan_dense_layers(
                (&layers, &layers, 8, fixed, later),
                super::Placement::Derived,
                (free - 1, total, host),
                2,
                2,
            )
            .is_err()
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_H3_IMAGE"]
    fn resident_image_request_two_matches_fresh_and_bounds_allocations() -> Result<()> {
        use crate::minimax_h3::{ImageReference, ImageRequest, ImageRunOptions};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let model = image_e2e_env("FF_H3_IMAGE")?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-resident-{}", std::process::id()));
        std::fs::create_dir(&runtime)?;
        let options = ImageRunOptions {
            device: 0,
            runtime_dir: runtime.clone(),
        };
        let result = (|| -> Result<()> {
            for edit in [false, true] {
                let make_request = |right: bool| {
                    let prompt = match (edit, right) {
                        (false, false) => "a red cube on a white table",
                        (false, true) => "a blue cube on a white table",
                        (true, false) => "replace the left half of <Picture 1> with a blue square",
                        (true, true) => "replace the right half of <Picture 1> with a blue square",
                    };
                    let mut request =
                        ImageRequest::new(prompt, 1024, 1024, 20261112 + u64::from(right));
                    if edit {
                        let width = 1536;
                        let height = 1024;
                        let phase = usize::from(right) * 97;
                        let pixels = (0..width * height)
                            .flat_map(|i| {
                                let (x, y) = (i % width, i / width);
                                [
                                    (3 * x + y + phase) as u8,
                                    (x + 5 * y) as u8,
                                    (7 * x + 11 * y) as u8,
                                ]
                            })
                            .collect();
                        request.references.push(ImageReference {
                            image: crate::serve::image::RgbImage {
                                width,
                                height,
                                pixels,
                            },
                        });
                    }
                    request
                };
                let (a, b) = (make_request(false), make_request(true));
                let fresh = super::image_latents(&model, &options, &b)?;
                let workload = [Ok(make_request(false)), Ok(make_request(true))].into_iter();
                let mut resident = super::ImageSession::open(&model, &options, workload)?;
                let first = resident.render(&a)?;
                ensure!(
                    first.rgb != fresh.rgb,
                    "resident stale-request control matched the fresh output"
                );
                drop(first);
                let ctx = crate::backend::cuda::Device::new(0)?.ctx;
                use cudarc::driver::sys::{self, CUmemPool_attribute};
                let mut pool = std::ptr::null_mut();
                unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result()? };
                let media_control = edit.then(|| {
                    let mut alternate = make_request(true);
                    alternate.references = make_request(false).references;
                    alternate
                });
                let mut boundary_bytes = None;
                let mut latent_bits_exact = true;
                for request_number in 2..=8 {
                    let mut peak = 0u64;
                    unsafe {
                        sys::cuMemPoolSetAttribute(
                            pool,
                            CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                            (&mut peak as *mut u64).cast(),
                        )
                        .result()?;
                    }
                    let different_media = media_control.as_ref().filter(|_| request_number == 3);
                    let got = resident.render(different_media.unwrap_or(&b))?;
                    if different_media.is_some() {
                        ensure!(
                            got.rgb != fresh.rgb,
                            "resident edit ignored the changed reference"
                        );
                    } else {
                        ensure!(
                            got.rgb == fresh.rgb,
                            "resident request {request_number} RGB differs from the fresh request"
                        );
                        latent_bits_exact &= got
                            .values
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(fresh.values.iter().map(|v| v.to_bits()));
                    }
                    drop(got);
                    ctx.synchronize()?;
                    let (free, total) = ctx.mem_get_info()?;
                    let mut used = 0u64;
                    unsafe {
                        sys::cuMemPoolGetAttribute(
                            pool,
                            CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                            (&mut used as *mut u64).cast(),
                        )
                        .result()?;
                        sys::cuMemPoolGetAttribute(
                            pool,
                            CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                            (&mut peak as *mut u64).cast(),
                        )
                        .result()?;
                    }
                    ensure!(
                        free >= total / 16 && u64::try_from(total - total / 16)? >= peak,
                        "resident request {request_number} exhausted the device reserve: free={free}, pool_peak={peak}, total={total}"
                    );
                    ensure!(
                        boundary_bytes.is_none_or(|baseline| used == baseline),
                        "resident request {request_number} changed live allocation bytes: {used} vs {boundary_bytes:?}"
                    );
                    boundary_bytes = Some(used);
                    let host_remaining =
                        crate::backend::cuda::calibrate::host_memory(Path::new("/"))?;
                    ensure!(
                        host_remaining >= 4 << 30,
                        "resident request {request_number} exhausted host headroom: {host_remaining}"
                    );
                    println!(
                        "H3_IMAGE_RESIDENT_BOUNDARY edit={edit} request={request_number} free={free} used={used} pool_peak={peak} host_remaining={host_remaining}"
                    );
                }
                let mut outside = make_request(true);
                outside.width += 1;
                ensure!(
                    resident.render(&outside).is_err(),
                    "resident admitted a different output extent"
                );
                println!(
                    "H3_IMAGE_RESIDENT PASS edit={edit} requests=8 request2_rgb_exact=true latent_bits_exact={latent_bits_exact} control=distinct"
                );
            }
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(runtime);
        result?;
        cleanup?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_H3_VIDEO_MODEL, FF_H3_VIDEO_MODE, FF_H3_LORA or FF_HYPERFLOW_MODEL"]
    fn resident_video_requests_match_fresh_and_bound_allocations() -> Result<()> {
        use crate::minimax_h3::{H3Output, H3Request, KeyframeAnchor, prepare_video};
        use cudarc::driver::sys::{self, CUmemPool_attribute as A};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let base = image_e2e_env("FF_H3_VIDEO_MODEL")?;
        let mode = std::env::var("FF_H3_VIDEO_MODE")?;
        let (model, companion, lora, schedule, spec) = match mode.as_str() {
            "minimax" => (
                base.clone(),
                None,
                Some(image_e2e_env("FF_H3_LORA")?),
                (8, (6.0, 3.0)),
                H3Spec::read(&base)?,
            ),
            "hyperflow" => {
                let component = image_e2e_env("FF_HYPERFLOW_MODEL")?;
                let spec = H3Spec::parse(&serde_json::from_slice(&std::fs::read(
                    component.join("config.json"),
                )?)?)?;
                (
                    component
                        .parent()
                        .context("HyperFlow model root")?
                        .to_path_buf(),
                    Some(base.clone()),
                    None,
                    (6, (12.0, 3.0)),
                    spec,
                )
            }
            _ => anyhow::bail!("FF_H3_VIDEO_MODE must be minimax or hyperflow"),
        };
        let make = |phase: usize, seed| -> Result<H3Request> {
            let (width, height) = (1344, 768);
            let image = crate::serve::image::RgbImage {
                width,
                height,
                pixels: (0..width * height)
                    .flat_map(|i| {
                        let (x, y) = (i % width, i / width);
                        [
                            ((3 * x + y + phase) % 251) as u8,
                            ((x + 5 * y) % 253) as u8,
                            ((7 * x + 11 * y) % 255) as u8,
                        ]
                    })
                    .collect(),
            };
            prepare_video(
                &base,
                &spec,
                "a boat passing a stone pier",
                &[(KeyframeAnchor::First, &image)],
                ((124, seed), schedule),
            )
        };
        let request_a = make(0, 271)?;
        let request_b = make(47, 379)?;
        let media_control = make(0, 379)?;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-video-resident-{}", std::process::id()));
        std::fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            let options = crate::RunOptions {
                device: 0,
                runtime_dir: runtime.clone(),
                verbose: true,
            };
            let mut reference = if let Some(lora) = &lora {
                pipeline::run(&model, lora, &request_b, (0, &runtime), None)?
            } else {
                pipeline::run_hyperflow(
                    (&base, &model.join("fl2va_int8_convrot")),
                    &request_b,
                    (0, &runtime),
                )?
            };
            reference.video_decoded = Default::default();
            let fresh = &reference.output;
            ensure!(
                (fresh.frames, fresh.height, fresh.width) == (124, 768, 1344)
                    && fresh.video.len() == 124 * 768 * 1344 * 3
                    && fresh.audio.len() == 2 * fresh.samples
                    && fresh.audio.iter().all(|&sample| sample.is_finite()),
                "fresh video extent or PCM"
            );
            let mut session = VideoSession::open(
                (&model, companion.as_deref()),
                (lora.as_deref(), None),
                &options,
                [Ok(request_a.clone()), Ok(request_b.clone())],
            )?;
            ensure!(
                session.render(&request_a)?.video != fresh.video,
                "stale video request control"
            );
            let ctx = crate::backend::cuda::Device::new(0)?.ctx;
            let mut pool = std::ptr::null_mut();
            unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result()? };
            let mut bytes = None;
            for request in 2..=8 {
                let mut peak = 0u64;
                unsafe {
                    sys::cuMemPoolSetAttribute(
                        pool,
                        A::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                        (&mut peak as *mut u64).cast(),
                    )
                    .result()?;
                }
                let H3Output { video, audio, .. } = match request {
                    2 => session.render_with_reference(&request_b, &reference)?,
                    3 => session.render(&media_control)?,
                    _ => session.render(&request_b)?,
                };
                ensure!(
                    video.len() == fresh.video.len()
                        && audio.len() == fresh.audio.len()
                        && audio.iter().all(|&sample| sample.is_finite()),
                    "resident video request {request} extent or PCM"
                );
                if request == 3 {
                    ensure!(video != fresh.video, "video ignored changed reference");
                } else {
                    let rgb = video == fresh.video;
                    let pcm = pipeline::same_f32_bits(&audio, &fresh.audio);
                    if !rgb || !pcm {
                        let rgb_diff = video
                            .iter()
                            .zip(&fresh.video)
                            .filter(|(a, b)| a != b)
                            .count();
                        let pcm_diff = audio
                            .iter()
                            .zip(&fresh.audio)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        anyhow::bail!(
                            "resident video request {request} differs from fresh: RGB {rgb_diff}/{} PCM {pcm_diff}/{}",
                            video.len(),
                            audio.len()
                        );
                    }
                }
                drop((video, audio));
                ctx.synchronize()?;
                let (free, total) = ctx.mem_get_info()?;
                let mut used = 0u64;
                unsafe {
                    sys::cuMemPoolGetAttribute(
                        pool,
                        A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                        (&mut used as *mut u64).cast(),
                    )
                    .result()?;
                    sys::cuMemPoolGetAttribute(
                        pool,
                        A::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                        (&mut peak as *mut u64).cast(),
                    )
                    .result()?;
                }
                let host = crate::backend::cuda::calibrate::host_memory(Path::new("/"))?;
                ensure!(
                    free >= total / 16
                        && peak <= u64::try_from(total - total / 16)?
                        && bytes.is_none_or(|before| used == before)
                        && host >= 4 << 30,
                    "resident video request {request} memory boundary: free={free} used={used} peak={peak} host={host}"
                );
                bytes = Some(used);
                println!(
                    "H3_VIDEO_RESIDENT_BOUNDARY mode={mode} request={request} used={used} peak={peak} host={host}"
                );
                if request == 3 {
                    let mut different = request_b.clone();
                    different.frames += 17;
                    ensure!(
                        session.render(&different).is_err(),
                        "resident video admitted changed geometry"
                    );
                }
            }
            println!(
                "H3_VIDEO_RESIDENT PASS mode={mode} requests=8 rgb_exact=true pcm_bits_exact=true media_control=distinct"
            );
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(runtime);
        result?;
        cleanup?;
        Ok(())
    }

    /// Cells (a), (b) and (e). Cell (d) is the contract `image_rgb8` test. Cell (c) has no tau.
    /// Captured generated noise is patchified into rows. Posterior epsilon and the reference mix stay NCHW.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_H3_IMAGE, FF_H3_IMAGE_E2E"]
    fn image_e2e_matches_the_frozen_cells() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let model = image_e2e_env("FF_H3_IMAGE")?;
        let root = image_e2e_env("FF_H3_IMAGE_E2E")?;
        let spec = ImageSpec::read(&model)?;
        image_e2e_geometry(&spec)?;
        image_e2e_noise_equal(&root)?;
        let arm_a = root.join("arm-a");
        let arm_b = root.join("arm-b");
        let mut defects = Vec::new();
        image_e2e_run_teacher(&model, &spec, &arm_a, false, &mut defects)?;
        image_e2e_run_teacher(&model, &spec, &arm_a, true, &mut defects)?;
        let t2i_rgb = image_e2e_run_pass(&model, &arm_a, &arm_b, false, &mut defects)?;
        let edit_rgb = image_e2e_run_pass(&model, &arm_a, &arm_b, true, &mut defects)?;
        let t2i_control = image_e2e_run_control(&model, &arm_a, false)?;
        let edit_control = image_e2e_run_control(&model, &arm_a, true)?;
        let t2i_decision = image_e2e_rgb_decision(&mut defects, "t2i_rgb", t2i_rgb, t2i_control)?;
        let edit_decision =
            image_e2e_rgb_decision(&mut defects, "edit_rgb", edit_rgb, edit_control)?;
        ensure!(
            defects.is_empty(),
            "H3_IMAGE_E2E_GATE DEFECT cells outside the frozen taus: {defects:?}"
        );
        println!("H3_IMAGE_E2E_GATE PASS cells=a,b,e e_t2i={t2i_decision} e_edit={edit_decision}");
        Ok(())
    }

    /// Removes the BLAS runtime directory when the block-cell gate returns.
    struct BlockRuntime(PathBuf);

    impl Drop for BlockRuntime {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn env_dir(key: &str) -> Result<PathBuf> {
        std::env::var_os(key)
            .map(PathBuf::from)
            .with_context(|| format!("{key} missing"))
    }

    fn read_exact(path: &Path, bytes: usize) -> Result<Vec<u8>> {
        let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            data.len() == bytes,
            "{} is {} bytes, expected {bytes}",
            path.display(),
            data.len()
        );
        Ok(data)
    }

    fn read_bf16(path: &Path, elements: usize) -> Result<Vec<bf16>> {
        let data = read_exact(path, elements * 2)?;
        Ok(data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect())
    }

    fn read_f32(path: &Path, elements: usize) -> Result<Vec<f32>> {
        let data = read_exact(path, elements * 4)?;
        Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn read_u32(path: &Path, elements: usize) -> Result<Vec<u32>> {
        let data = read_exact(path, elements * 4)?;
        Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect())
    }

    /// `(max_abs, rmse)` in f64. Reference pairs were measured in f32; these taus are that measurement.
    fn max_rmse(candidate: &[f64], reference: &[f64]) -> Result<(f64, f64)> {
        crate::verify::max_abs_rmse("h3.max_rmse", candidate, reference)
    }

    fn f64_of_bf16(values: &[bf16]) -> Vec<f64> {
        values.iter().map(|v| f64::from(v.to_f32())).collect()
    }

    fn f64_of_f32(values: &[f32]) -> Vec<f64> {
        values.iter().copied().map(f64::from).collect()
    }

    /// Pair 0 rewritten to pair 2. Tag stays in `index % 3`.
    fn swap_video_pair(index: &[u32]) -> Vec<u32> {
        index
            .iter()
            .map(|&value| {
                let pair = value / MODALITIES;
                let tag = value % MODALITIES;
                let pair = if pair == 0 { 2 } else { pair };
                pair * MODALITIES + tag
            })
            .collect()
    }

    fn pair_columns(meta: &serde_json::Value) -> Result<(Vec<f32>, Vec<f32>)> {
        let pairs = meta["pair_values"]
            .as_array()
            .context("reference pair_values")?;
        let mut timesteps = Vec::with_capacity(pairs.len());
        let mut endpoints = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let row = pair.as_array().context("reference pair")?;
            ensure!(row.len() == 2, "reference pair width");
            let t = row[0].as_f64().context("reference pair t")? as f32;
            let r = row[1].as_f64().context("reference pair r")? as f32;
            ensure!(
                t.is_finite() && r.is_finite(),
                "reference pair is not finite"
            );
            timesteps.push(t);
            endpoints.push(r);
        }
        ensure!(!timesteps.is_empty(), "reference pairs are empty");
        Ok((timesteps, endpoints))
    }

    fn time_embedder(
        source: &mut Source<'_>,
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        prefix: &str,
    ) -> Result<AdalnTimeEmbedder> {
        Ok(AdalnTimeEmbedder {
            linear_1_weight: source.host_f32(
                s,
                &format!("{prefix}.linear_1.weight"),
                spec.time_hidden * spec.freq_dim,
            )?,
            linear_1_bias: source.host_f32(
                s,
                &format!("{prefix}.linear_1.bias"),
                spec.time_hidden,
            )?,
            linear_2_weight: source.host_f32(
                s,
                &format!("{prefix}.linear_2.weight"),
                spec.time_dim * spec.time_hidden,
            )?,
            linear_2_bias: source.host_f32(s, &format!("{prefix}.linear_2.bias"), spec.time_dim)?,
        })
    }

    /// max-abs, RMSE. `max(2 × worst pair, eps)` from floor-cpu, 4090-a, 4090-b, a4000-a, a4000-b on 2026-11-12.
    const BLOCK0_TAU: (f64, f64) = (1024.0, 5.925411224365234);
    const BLOCK1_TAU: (f64, f64) = (256.0, 3.944915533065796);
    const BLOCK6_TAU: (f64, f64) = (1e-6, 1e-7);
    const TEMB_TAU: (f64, f64) = (0.00738525390625, 0.0026896463241428137);
    /// 1000 rows. Remainders 8, 40 and 232 against tiles 16, 64 and 256.
    const RAGGED_ROWS: usize = 1000;
    /// max(2 × worst pair, eps). Worst pair is floor-cpu versus either GPU: max-abs 768, RMSE 2.6518330574035645.
    const RAGGED_BLOCK0_TAU: (f64, f64) = (1536.0, 5.303666114807129);
    /// Worst max-abs is 256. Worst RMSE is 4090 versus A4000, 1.9843404293060303.
    const RAGGED_BLOCK1_TAU: (f64, f64) = (512.0, 3.9686808586120605);

    fn record_cell(
        arm: &str,
        cell: &'static str,
        got: (f64, f64),
        tau: (f64, f64),
        defects: &mut Vec<&'static str>,
    ) {
        let pass = got.0 <= tau.0 && got.1 <= tau.1;
        println!(
            "HYPERFLOW_BLOCK arm={arm} cell={cell} max_abs={:.16e} rmse={:.16e} tau_abs={:.16e} tau_rmse={:.16e} {}",
            got.0,
            got.1,
            tau.0,
            tau.1,
            if pass { "PASS" } else { "DEFECT" }
        );
        if !pass {
            defects.push(cell);
        }
    }

    fn record_control(name: &str, got: (f64, f64), tau: (f64, f64), failed: &mut Vec<String>) {
        let abs_ratio = got.0 / tau.0;
        let rmse_ratio = got.1 / tau.1;
        let pass = abs_ratio > 10.0 && rmse_ratio > 10.0;
        println!(
            "HYPERFLOW_BLOCK control={name} max_abs={:.16e} rmse={:.16e} ratio_abs={abs_ratio:.6} ratio_rmse={rmse_ratio:.6} {}",
            got.0,
            got.1,
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            failed.push(name.to_owned());
        }
    }

    /// INT8 block 0, low-rank block 1, dropped block 6, the pre-SiLU f32 temb mix, and blocks 0 and 1 at 1000 rows.
    #[test]
    #[ignore = "requires a reserved GPU window; FF_HYPERFLOW_MODEL, FF_HYPERFLOW_BLOCK_REF, FF_HYPERFLOW_BLOCK_REF_RAGGED"]
    fn hyperflow_block_cells_match_the_definition() -> Result<()> {
        let model = env_dir("FF_HYPERFLOW_MODEL")?;
        let reference = env_dir("FF_HYPERFLOW_BLOCK_REF")?;
        let arm = reference
            .file_name()
            .and_then(|name| name.to_str())
            .context("reference arm name")?
            .to_owned();
        let spec = H3Spec::parse(&serde_json::from_slice(&fs::read(
            model.join("config.json"),
        )?)?)?;
        ensure!(spec.is_hyperflow()?, "transformer config is not HyperFlow");
        ensure!(
            matches!(
                spec.linear_storage("transformer_blocks.0.attn.to_q")?,
                LinearStorage::Int8 { .. }
            ),
            "block 0 to_q is not int8"
        );
        ensure!(
            matches!(
                spec.linear_storage("transformer_blocks.1.attn.to_q")?,
                LinearStorage::LowRank { .. }
            ),
            "block 1 to_q is not low-rank"
        );
        let dropped = spec.dropped_blocks.as_deref().context("dropped blocks")?;
        ensure!(
            dropped.contains(&6) && !dropped.contains(&0) && !dropped.contains(&1),
            "blocks 0, 1 and 6 roles {dropped:?}"
        );
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(reference.join("meta.json"))?)?;
        ensure!(
            meta["candidate_run"].as_bool() == Some(false),
            "reference arm is marked as a candidate run"
        );
        ensure!(meta["rows"].as_u64() == Some(20), "reference rows");
        let rows = 20usize;
        let hidden_count = rows * spec.hidden;
        let hidden = read_bf16(&reference.join("hidden.i16"), hidden_count)?;
        let positions = read_f32(&reference.join("positions.f32"), rows * 3)?;
        let adaln = read_u32(&reference.join("adaln.i32"), rows)?;
        let (timesteps, endpoints) = pair_columns(&meta)?;
        let entries = u32::try_from(timesteps.len())? * MODALITIES;
        ensure!(
            adaln.iter().all(|&index| index < entries),
            "adaln index is outside {entries} rows"
        );
        let temb_ref = read_f32(&reference.join("temb.f32"), timesteps.len() * spec.time_dim)?;
        let block_ref =
            |block: usize| read_bf16(&reference.join(format!("block{block}.i16")), hidden_count);
        let block0_ref = block_ref(0)?;
        let block1_ref = block_ref(1)?;
        let block6_ref = block_ref(6)?;
        ensure!(
            block6_ref
                .iter()
                .zip(&hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "dropped reference is not the input"
        );
        let control0 = read_bf16(&reference.join("control0.i16"), hidden_count)?;
        let control1 = read_bf16(&reference.join("control1.i16"), hidden_count)?;
        let mut control_fail = Vec::new();
        record_control(
            "reference-block0",
            max_rmse(&f64_of_bf16(&control0), &f64_of_bf16(&block0_ref))?,
            BLOCK0_TAU,
            &mut control_fail,
        );
        record_control(
            "reference-block1",
            max_rmse(&f64_of_bf16(&control1), &f64_of_bf16(&block1_ref))?,
            BLOCK1_TAU,
            &mut control_fail,
        );
        let mutant: Vec<bf16> = hidden
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() + v.to_f32()))
            .collect();
        record_control(
            "reference-block6-mutant",
            max_rmse(&f64_of_bf16(&mutant), &f64_of_bf16(&block6_ref))?,
            BLOCK6_TAU,
            &mut control_fail,
        );
        let temb_swapped = read_f32(
            &reference.join("temb_swapped.f32"),
            timesteps.len() * spec.time_dim,
        )?;
        record_control(
            "reference-temb-swapped",
            max_rmse(&f64_of_f32(&temb_swapped), &f64_of_f32(&temb_ref))?,
            TEMB_TAU,
            &mut control_fail,
        );
        let ragged = env_dir("FF_HYPERFLOW_BLOCK_REF_RAGGED")?;
        let ragged_meta: serde_json::Value =
            serde_json::from_slice(&fs::read(ragged.join("meta.json"))?)?;
        ensure!(
            ragged_meta["candidate_run"].as_bool() == Some(false),
            "ragged reference arm is marked as a candidate run"
        );
        ensure!(
            ragged_meta["rows"].as_u64() == Some(u64::try_from(RAGGED_ROWS)?),
            "ragged reference rows"
        );
        let (ragged_steps, ragged_ends) = pair_columns(&ragged_meta)?;
        ensure!(
            ragged_steps == timesteps && ragged_ends == endpoints,
            "ragged pairs differ from the 20-row arm"
        );
        let ragged_count = RAGGED_ROWS * spec.hidden;
        let ragged_hidden = read_bf16(&ragged.join("hidden.i16"), ragged_count)?;
        let ragged_positions = read_f32(&ragged.join("positions.f32"), RAGGED_ROWS * 3)?;
        let ragged_adaln = read_u32(&ragged.join("adaln.i32"), RAGGED_ROWS)?;
        ensure!(
            ragged_adaln.iter().all(|&index| index < entries),
            "ragged adaln index is outside {entries} rows"
        );
        let ragged_block0 = read_bf16(&ragged.join("block0.i16"), ragged_count)?;
        let ragged_block1 = read_bf16(&ragged.join("block1.i16"), ragged_count)?;
        let ragged_control0 = read_bf16(&ragged.join("control0.i16"), ragged_count)?;
        let ragged_control1 = read_bf16(&ragged.join("control1.i16"), ragged_count)?;
        record_control(
            "reference-ragged-block0",
            max_rmse(&f64_of_bf16(&ragged_control0), &f64_of_bf16(&ragged_block0))?,
            RAGGED_BLOCK0_TAU,
            &mut control_fail,
        );
        record_control(
            "reference-ragged-block1",
            max_rmse(&f64_of_bf16(&ragged_control1), &f64_of_bf16(&ragged_block1))?,
            RAGGED_BLOCK1_TAU,
            &mut control_fail,
        );
        ensure!(
            control_fail.is_empty(),
            "reference control stayed within 10x tau: {control_fail:?}"
        );

        let manifest = MappedCheckpoint::open_component(&model, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-hyperflow-block-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let _runtime = BlockRuntime(runtime.clone());
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "HyperFlow block cells".into())),
        )?;
        let mut source = Source::new(&model, &manifest);
        source.pack_vectors(s, &spec)?;
        let current = time_embedder(&mut source, s, &spec, "time_embedder")?;
        let endpoint = time_embedder(&mut source, s, &spec, "endpoint_time_embedder")?;
        let gate = spec.hyperflow_gate.context("HyperFlow gate")?;
        let temb = crate::minimax_h3::mixed_temb(
            &spec, &current, &endpoint, &timesteps, &endpoints, gate,
        )?;
        ensure!(
            temb.len() == temb_ref.len() && temb.iter().all(|v| v.is_finite()),
            "temb extent or a non-finite lane"
        );
        let mut defects = Vec::new();
        record_cell(
            &arm,
            "temb",
            max_rmse(&f64_of_f32(&temb), &f64_of_f32(&temb_ref))?,
            TEMB_TAU,
            &mut defects,
        );
        let swapped_temb = crate::minimax_h3::mixed_temb(
            &spec, &current, &endpoint, &endpoints, &timesteps, gate,
        )?;
        ensure!(
            swapped_temb.len() == temb_ref.len() && swapped_temb.iter().all(|v| v.is_finite()),
            "swapped temb extent or a non-finite lane"
        );
        record_control(
            "candidate-temb-swapped",
            max_rmse(&f64_of_f32(&swapped_temb), &f64_of_f32(&temb_ref))?,
            TEMB_TAU,
            &mut control_fail,
        );
        let x_host = mixed_adaln_input(&spec, &current, &endpoint, &timesteps, &endpoints, gate)?;
        ensure!(
            x_host.len() == timesteps.len() * spec.time_dim,
            "adaln input extent"
        );
        let coords: Vec<[f64; 3]> = positions
            .as_chunks::<3>()
            .0
            .iter()
            .map(|row| [f64::from(row[0]), f64::from(row[1]), f64::from(row[2])])
            .collect();
        ensure!(coords.len() == rows, "position rows");
        let rotary_host = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &coords);
        ensure!(
            rotary_host.len() == rows * 6 * spec.rope_freq_dim,
            "rotary length {}",
            rotary_host.len()
        );
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let mut work = BlockWork::new(s, &spec, 0, rows)?;
        work.plan_rotation(&device, &spec)?;
        work.prepare::<false>(&blas, &spec)?;
        let modulation_width = 6 * spec.hidden * MODALITIES as usize;
        let mut weight = s.alloc_zeros::<bf16>(modulation_width * spec.time_dim)?;
        let mut bias = s.alloc_zeros::<bf16>(modulation_width)?;
        let projected = s.alloc_zeros::<bf16>(timesteps.len() * modulation_width)?;
        let mode = LinearBiasMode::BeforeOutputRound;
        blas.prepare_biased(
            u32::try_from(timesteps.len())?,
            u32::try_from(modulation_width)?,
            u32::try_from(spec.time_dim)?,
            vector(&bias, s)?,
            mode,
        )?;
        let x = s.clone_htod(&x_host)?;
        let rotary = s.clone_htod(&rotary_host)?;
        let index = s.clone_htod(&adaln)?;
        let swapped = s.clone_htod(&swap_video_pair(&adaln))?;
        for (block, tau, reference_out) in [
            (0usize, BLOCK0_TAU, block0_ref),
            (1, BLOCK1_TAU, block1_ref),
        ] {
            let weights = BlockWeights::stored(
                s,
                &spec,
                &format!("transformer_blocks.{block}"),
                &mut source,
                None,
            )?;
            let name = format!("transformer_blocks.{block}.adaln_proj.linear");
            for (suffix, buffer) in [("weight", &mut weight), ("bias", &mut bias)] {
                source.fill(s, &format!("{name}.{suffix}"), &mut buffer.slice_mut(..))?;
            }
            blas.linear_biased(
                matrix(&weight, s, modulation_width, spec.time_dim)?,
                vector(&bias, s)?,
                matrix(&x, s, timesteps.len(), spec.time_dim)?,
                &mut TensorMut(matrix(&projected, s, timesteps.len(), modulation_width)?),
                mode,
            )?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: 0.0,
            };
            let mut launch = |index: &CudaSlice<u32>| -> Result<Vec<f64>> {
                let mut row = s.clone_htod(&hidden)?;
                let modulation = Modulation {
                    table: &projected,
                    offset: 0,
                    entries,
                    index,
                };
                block_forward::<false>(
                    &context,
                    &mut flash,
                    &weights,
                    &Conditioning::Modulated {
                        rotary: &rotary,
                        modulation: &modulation,
                    },
                    &mut row,
                    &mut work,
                )?;
                let host = s.clone_dtoh(&row)?;
                ensure!(
                    host.len() == hidden.len() && host.iter().all(|v| v.to_f32().is_finite()),
                    "block {block} output extent or a non-finite value"
                );
                Ok(host.into_iter().map(|v| f64::from(v.to_f32())).collect())
            };
            let output = launch(&index)?;
            record_cell(
                &arm,
                if block == 0 { "block0" } else { "block1" },
                max_rmse(&output, &f64_of_bf16(&reference_out))?,
                tau,
                &mut defects,
            );
            let swapped_out = launch(&swapped)?;
            record_control(
                &format!("candidate-block{block}"),
                max_rmse(&swapped_out, &output)?,
                tau,
                &mut control_fail,
            );
        }
        let identity = BlockWeights::identity(s, &spec, &mut source, 6)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 0.0,
        };
        let mut row = s.clone_htod(&hidden)?;
        let modulation = Modulation {
            table: &projected,
            offset: 0,
            entries,
            index: &index,
        };
        block_forward::<false>(
            &context,
            &mut flash,
            &identity,
            &Conditioning::Modulated {
                rotary: &rotary,
                modulation: &modulation,
            },
            &mut row,
            &mut work,
        )?;
        let dropped_out = s.clone_dtoh(&row)?;
        ensure!(
            dropped_out.len() == hidden.len() && dropped_out.iter().all(|v| v.to_f32().is_finite()),
            "dropped block output extent or a non-finite value"
        );
        record_cell(
            &arm,
            "block6",
            max_rmse(
                &dropped_out
                    .iter()
                    .map(|v| f64::from(v.to_f32()))
                    .collect::<Vec<_>>(),
                &f64_of_bf16(&block6_ref),
            )?,
            BLOCK6_TAU,
            &mut defects,
        );
        drop(flash);
        drop(work);
        let ragged_coords: Vec<[f64; 3]> = ragged_positions
            .as_chunks::<3>()
            .0
            .iter()
            .map(|row| [f64::from(row[0]), f64::from(row[1]), f64::from(row[2])])
            .collect();
        ensure!(ragged_coords.len() == RAGGED_ROWS, "ragged position rows");
        let ragged_rotary_host = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &ragged_coords);
        ensure!(
            ragged_rotary_host.len() == RAGGED_ROWS * 6 * spec.rope_freq_dim,
            "ragged rotary length {}",
            ragged_rotary_host.len()
        );
        let mut flash = Flash::new(
            s.clone(),
            RAGGED_ROWS,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?
        .with_full_single_probability()?;
        let mut work = BlockWork::new(s, &spec, 0, RAGGED_ROWS)?;
        work.plan_rotation(&device, &spec)?;
        work.prepare::<false>(&blas, &spec)?;
        let ragged_rotary = s.clone_htod(&ragged_rotary_host)?;
        let ragged_index = s.clone_htod(&ragged_adaln)?;
        let ragged_swapped = s.clone_htod(&swap_video_pair(&ragged_adaln))?;
        for (block, tau, reference_out) in [
            (0usize, RAGGED_BLOCK0_TAU, ragged_block0),
            (1, RAGGED_BLOCK1_TAU, ragged_block1),
        ] {
            let weights = BlockWeights::stored(
                s,
                &spec,
                &format!("transformer_blocks.{block}"),
                &mut source,
                None,
            )?;
            let name = format!("transformer_blocks.{block}.adaln_proj.linear");
            for (suffix, buffer) in [("weight", &mut weight), ("bias", &mut bias)] {
                source.fill(s, &format!("{name}.{suffix}"), &mut buffer.slice_mut(..))?;
            }
            blas.linear_biased(
                matrix(&weight, s, modulation_width, spec.time_dim)?,
                vector(&bias, s)?,
                matrix(&x, s, timesteps.len(), spec.time_dim)?,
                &mut TensorMut(matrix(&projected, s, timesteps.len(), modulation_width)?),
                mode,
            )?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: 0.0,
            };
            let mut launch = |index: &CudaSlice<u32>| -> Result<Vec<f64>> {
                let mut row = s.clone_htod(&ragged_hidden)?;
                let modulation = Modulation {
                    table: &projected,
                    offset: 0,
                    entries,
                    index,
                };
                block_forward::<false>(
                    &context,
                    &mut flash,
                    &weights,
                    &Conditioning::Modulated {
                        rotary: &ragged_rotary,
                        modulation: &modulation,
                    },
                    &mut row,
                    &mut work,
                )?;
                let host = s.clone_dtoh(&row)?;
                ensure!(
                    host.len() == ragged_hidden.len()
                        && host.iter().all(|v| v.to_f32().is_finite()),
                    "ragged block {block} output extent or a non-finite value"
                );
                Ok(host.into_iter().map(|v| f64::from(v.to_f32())).collect())
            };
            let output = launch(&ragged_index)?;
            record_cell(
                &arm,
                if block == 0 {
                    "ragged-block0"
                } else {
                    "ragged-block1"
                },
                max_rmse(&output, &f64_of_bf16(&reference_out))?,
                tau,
                &mut defects,
            );
            let swapped_out = launch(&ragged_swapped)?;
            record_control(
                &format!("candidate-ragged-block{block}"),
                max_rmse(&swapped_out, &output)?,
                tau,
                &mut control_fail,
            );
        }
        ensure!(
            control_fail.is_empty(),
            "control stayed within 10x tau: {control_fail:?}"
        );
        if defects.is_empty() {
            println!("HYPERFLOW_BLOCK_GATE PASS cells=6 controls=FAIL_AS_REQUIRED");
        } else {
            println!("HYPERFLOW_BLOCK_GATE DEFECT cells={defects:?} controls=FAIL_AS_REQUIRED");
        }
        Ok(())
    }

    /// 6-step raw table cast f64 to f32, then `shift * base / ((shift - 1) * base + 1)` in f32.
    fn step_grid(shift: f32) -> Schedule {
        let raw = [
            1.0f64, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0,
        ];
        let sigmas: Vec<f32> = raw
            .iter()
            .map(|&base| {
                let base = base as f32;
                shift * base / ((shift - 1.0) * base + 1.0)
            })
            .collect();
        let timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|sigma| 1.0 - sigma)
            .collect();
        Schedule { sigmas, timesteps }
    }

    fn assert_step_grids(video: &Schedule, audio: &Schedule) -> Result<()> {
        let flow = hyperflow_steps(6)?;
        ensure!(
            flow.len() == 6 && video.sigmas.len() == 7 && audio.sigmas.len() == 7,
            "step grid extent"
        );
        for (k, step) in flow.iter().enumerate() {
            ensure!(
                video.timesteps[k].to_bits() == step.video_t.to_bits()
                    && (1.0 - video.sigmas[k + 1]).to_bits() == step.video_r.to_bits(),
                "video grid step {k} differs from hyperflow_steps"
            );
            ensure!(
                audio.timesteps[k].to_bits() == step.audio_t.to_bits()
                    && (1.0 - audio.sigmas[k + 1]).to_bits() == step.audio_r.to_bits(),
                "audio grid step {k} differs from hyperflow_steps"
            );
            ensure!(
                video.sigmas[k] != 0.0 && audio.sigmas[k] != 0.0,
                "evaluation sigma {k} is zero"
            );
        }
        Ok(())
    }

    #[test]
    fn hyperflow_step_grid_matches_hyperflow_steps() -> Result<()> {
        assert_step_grids(&step_grid(12.0), &step_grid(3.0))
    }

    const STEP_TEXT: usize = 7;
    const STEP_KEY: usize = 17;
    const STEP_AUDIO: usize = 40;
    const STEP_GENERATED: usize = 936;
    const STEP_VIDEO: usize = STEP_KEY + STEP_GENERATED;
    const STEP_ROWS: usize = STEP_TEXT + STEP_KEY + STEP_AUDIO + STEP_GENERATED;
    const STEP_VIDEO_WIDTH: usize = 96;
    const STEP_AUDIO_WIDTH: usize = 32;

    fn step_layout() -> PackedLayout {
        let audio_start = STEP_TEXT + STEP_KEY;
        let video_start = audio_start + STEP_AUDIO;
        let mut position_ids = vec![[0.0f64; 3]; STEP_ROWS];
        for (row, position) in position_ids.iter_mut().enumerate() {
            for (axis, slot) in position.iter_mut().enumerate() {
                *slot = (((row * 3 + axis) * 5 + 1) % 17) as f64;
            }
        }
        let mut token_tags = vec![TEXT_TAG; STEP_TEXT];
        token_tags.extend(std::iter::repeat_n(VIDEO_TAG, STEP_KEY));
        token_tags.extend(std::iter::repeat_n(AUDIO_TAG, STEP_AUDIO));
        token_tags.extend(std::iter::repeat_n(VIDEO_TAG, STEP_GENERATED));
        let range = |start: usize, end: usize| (start as u32..end as u32).collect::<Vec<_>>();
        let mut video_indices = range(STEP_TEXT, STEP_TEXT + STEP_KEY);
        video_indices.extend(range(video_start, STEP_ROWS));
        PackedLayout {
            position_ids,
            token_tags,
            video_indices,
            audio_indices: range(audio_start, video_start),
            text_indices: range(0, STEP_TEXT),
            condition_video_rows: STEP_KEY,
        }
    }

    fn step_pattern(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 17 + 3) % 251 % 17) as f32 - 8.0)
            .collect()
    }

    struct StepTau {
        name: &'static str,
        abs: f64,
        rmse: f64,
        rel: f64,
        /// True when relative RMSE is the gate and max-abs is reported.
        relative: bool,
    }

    /// max(2 × worst pair, 1e-6) and max(2 × worst pair, 1e-7). The eps floor binds nothing.
    const STEP_TAUS: &[StepTau] = &[
        StepTau {
            name: "refiner.bf16",
            abs: 1.0,
            rmse: 0.02525967030888548,
            rel: 0.011894049114080173,
            relative: false,
        },
        StepTau {
            name: "vel-video-0.f32",
            abs: 1.0344287157058716,
            rmse: 0.08480887073585625,
            rel: 0.03737819870938896,
            relative: false,
        },
        StepTau {
            name: "vel-audio-0.f32",
            abs: 1.2973367869853973,
            rmse: 0.35147537383978655,
            rel: 0.12659836498390653,
            relative: false,
        },
        StepTau {
            name: "lat-video-0.f32",
            abs: 0.01794123649597168,
            rmse: 0.0014709306476793245,
            rel: 0.00030436084823948987,
            relative: false,
        },
        StepTau {
            name: "lat-audio-0.f32",
            abs: 0.08555293083190918,
            rmse: 0.023178044608518033,
            rel: 0.004942957125442557,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-0.f32",
            abs: 1.0344287157058716,
            rmse: 0.08480887073585625,
            rel: 0.03737819870938896,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-0.f32",
            abs: 1.2973367869853973,
            rmse: 0.35147537383978655,
            rel: 0.12659836498390653,
            relative: false,
        },
        StepTau {
            name: "vel-video-1.f32",
            abs: 0.6497125625610352,
            rmse: 0.06660809516296118,
            rel: 0.027951056617136708,
            relative: false,
        },
        StepTau {
            name: "vel-audio-1.f32",
            abs: 1.6679048538208008,
            rmse: 0.2512035030694407,
            rel: 0.09332653380557283,
            relative: false,
        },
        StepTau {
            name: "lat-video-1.f32",
            abs: 0.04277610778808594,
            rmse: 0.004835248490736819,
            rel: 0.0010299083327436733,
            relative: false,
        },
        StepTau {
            name: "lat-audio-1.f32",
            abs: 0.37093067169189453,
            rmse: 0.05664436512246173,
            rel: 0.013632521602094846,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-1.f32",
            abs: 0.8052752614021301,
            rmse: 0.06484987049676891,
            rel: 0.027213244838100666,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-1.f32",
            abs: 2.3954267501831055,
            rmse: 0.2548296150311912,
            rel: 0.09467369838109052,
            relative: false,
        },
        StepTau {
            name: "vel-video-2.f32",
            abs: 0.6330146789550781,
            rmse: 0.06821279829460544,
            rel: 0.026989072706701332,
            relative: false,
        },
        StepTau {
            name: "vel-audio-2.f32",
            abs: 1.1475920677185059,
            rmse: 0.216037983598139,
            rel: 0.08374482569409536,
            relative: false,
        },
        StepTau {
            name: "lat-video-2.f32",
            abs: 0.14624595642089844,
            rmse: 0.016806086510819676,
            rel: 0.003968390152038171,
            relative: false,
        },
        StepTau {
            name: "lat-audio-2.f32",
            abs: 0.5485353469848633,
            rmse: 0.08862821877285647,
            rel: 0.027361272251040947,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-2.f32",
            abs: 0.6307210922241211,
            rmse: 0.06718731531069264,
            rel: 0.026583330155387518,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-2.f32",
            abs: 0.8542909622192383,
            rmse: 0.185065455100368,
            rel: 0.07173865457015058,
            relative: false,
        },
        StepTau {
            name: "vel-video-3.f32",
            abs: 1.0802555084228516,
            rmse: 0.08557588311113899,
            rel: 0.03679224202385931,
            relative: false,
        },
        StepTau {
            name: "vel-audio-3.f32",
            abs: 1.540313720703125,
            rmse: 0.2872992311398981,
            rel: 0.1336442409171142,
            relative: false,
        },
        StepTau {
            name: "lat-video-3.f32",
            abs: 0.277801513671875,
            rmse: 0.028397494333519183,
            rel: 0.007408600318011723,
            relative: false,
        },
        StepTau {
            name: "lat-audio-3.f32",
            abs: 0.5863490104675293,
            rmse: 0.10182856943460111,
            rel: 0.03549959772912339,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-3.f32",
            abs: 0.9124784469604492,
            rmse: 0.08180765830905622,
            rel: 0.03517214260007093,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-3.f32",
            abs: 1.3214263319969177,
            rmse: 0.27508126771898883,
            rel: 0.12796075739207344,
            relative: false,
        },
        StepTau {
            name: "vel-video-4.f32",
            abs: 0.9622354507446289,
            rmse: 0.08843276977313971,
            rel: 0.04125339004308106,
            relative: false,
        },
        StepTau {
            name: "vel-audio-4.f32",
            abs: 1.509389877319336,
            rmse: 0.2899731406880125,
            rel: 0.14560669844538246,
            relative: false,
        },
        StepTau {
            name: "lat-video-4.f32",
            abs: 0.3898341655731201,
            rmse: 0.032307474598488914,
            rel: 0.008911228870711975,
            relative: false,
        },
        StepTau {
            name: "lat-audio-4.f32",
            abs: 0.5971055030822754,
            rmse: 0.10068899567672909,
            rel: 0.03682861818829619,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-4.f32",
            abs: 0.9593095779418945,
            rmse: 0.08518561480349118,
            rel: 0.03973861050109811,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-4.f32",
            abs: 1.9413082599639893,
            rmse: 0.3236822992442777,
            rel: 0.16253336714684885,
            relative: false,
        },
        StepTau {
            name: "vel-video-5.f32",
            abs: 1.1887826919555664,
            rmse: 0.0944259072454749,
            rel: 0.043959013882068826,
            relative: false,
        },
        StepTau {
            name: "vel-audio-5.f32",
            abs: 1.9153492450714111,
            rmse: 0.39024225987989564,
            rel: 0.19295143171868911,
            relative: false,
        },
        StepTau {
            name: "lat-video-5.f32",
            abs: 0.34001898765563965,
            rmse: 0.032920431940877466,
            rel: 0.00937076536622547,
            relative: false,
        },
        StepTau {
            name: "lat-audio-5.f32",
            abs: 0.5137481689453125,
            rmse: 0.09958590270065194,
            rel: 0.037359167926177196,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-5.f32",
            abs: 1.1486835479736328,
            rmse: 0.0967375876840172,
            rel: 0.04503519303091845,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-5.f32",
            abs: 2.3399438858032227,
            rmse: 0.41770544253584774,
            rel: 0.20653033118141956,
            relative: true,
        },
    ];

    /// 2 × measured `rocm-bf16-default` vs `rocm-bf16-math` floor on `fl2va_bf16` (gfx1100, 2026-11-20).
    const STEP_BF16_TAUS: &[StepTau] = &[
        StepTau {
            name: "refiner.bf16",
            abs: 0.5,
            rmse: 0.017668256077534167,
            rel: 0.00832016298328559,
            relative: false,
        },
        StepTau {
            name: "vel-video-0.f32",
            abs: 0.42182040214538574,
            rmse: 0.03774369037645171,
            rel: 0.016575089152730343,
            relative: false,
        },
        StepTau {
            name: "vel-audio-0.f32",
            abs: 0.7517321109771729,
            rmse: 0.1687024932763922,
            rel: 0.06024474022760859,
            relative: false,
        },
        StepTau {
            name: "lat-video-0.f32",
            abs: 0.007316112518310547,
            rmse: 0.0006546278958366227,
            rel: 0.000135457263608369,
            relative: false,
        },
        StepTau {
            name: "lat-audio-0.f32",
            abs: 0.04957318305969238,
            rmse: 0.011125090310132427,
            rel: 0.002373297814451077,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-0.f32",
            abs: 0.42182040214538574,
            rmse: 0.03774369037645171,
            rel: 0.016575089152730343,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-0.f32",
            abs: 0.7517321109771729,
            rmse: 0.1687024932763922,
            rel: 0.06024474022760859,
            relative: false,
        },
        StepTau {
            name: "vel-video-1.f32",
            abs: 0.2743256688117981,
            rmse: 0.03531438478702691,
            rel: 0.014842983183751128,
            relative: false,
        },
        StepTau {
            name: "vel-audio-1.f32",
            abs: 1.8866877555847168,
            rmse: 0.18829725965940894,
            rel: 0.06927082231579365,
            relative: false,
        },
        StepTau {
            name: "lat-video-1.f32",
            abs: 0.018365368247032166,
            rmse: 0.002501225302131081,
            rel: 0.0005327886361772385,
            relative: false,
        },
        StepTau {
            name: "lat-audio-1.f32",
            abs: 0.3870220184326172,
            rmse: 0.04005478176087406,
            rel: 0.009654786920627711,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-1.f32",
            abs: 0.2743256688117981,
            rmse: 0.03531438478702691,
            rel: 0.014842983183751128,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-1.f32",
            abs: 1.8866877555847168,
            rmse: 0.18829725965940894,
            rel: 0.06927082231579365,
            relative: false,
        },
        StepTau {
            name: "vel-video-2.f32",
            abs: 0.36568450927734375,
            rmse: 0.041601498633534004,
            rel: 0.01649239066771905,
            relative: false,
        },
        StepTau {
            name: "vel-audio-2.f32",
            abs: 0.5907998085021973,
            rmse: 0.16724169116595536,
            rel: 0.06546250094106436,
            relative: false,
        },
        StepTau {
            name: "lat-video-2.f32",
            abs: 0.09511947631835938,
            rmse: 0.0102221699456616,
            rel: 0.002413908253208725,
            relative: false,
        },
        StepTau {
            name: "lat-audio-2.f32",
            abs: 0.4305868148803711,
            rmse: 0.07353032736929174,
            rel: 0.02270540814631225,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-2.f32",
            abs: 0.36568450927734375,
            rmse: 0.041601498633534004,
            rel: 0.01649239066771905,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-2.f32",
            abs: 0.5907998085021973,
            rmse: 0.16724169116595536,
            rel: 0.06546250094106436,
            relative: false,
        },
        StepTau {
            name: "vel-video-3.f32",
            abs: 0.48323917388916016,
            rmse: 0.05016238961624561,
            rel: 0.021480094513262625,
            relative: false,
        },
        StepTau {
            name: "vel-audio-3.f32",
            abs: 1.0220494270324707,
            rmse: 0.20547608984167087,
            rel: 0.09529300634760446,
            relative: false,
        },
        StepTau {
            name: "lat-video-3.f32",
            abs: 0.14125442504882812,
            rmse: 0.01717454837784889,
            rel: 0.004483965147549544,
            relative: false,
        },
        StepTau {
            name: "lat-audio-3.f32",
            abs: 0.41321659088134766,
            rmse: 0.07934092694877547,
            rel: 0.027703147232461704,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-3.f32",
            abs: 0.48323917388916016,
            rmse: 0.05016238961624561,
            rel: 0.021480094513262625,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-3.f32",
            abs: 1.0220494270324707,
            rmse: 0.20547608984167087,
            rel: 0.09529300634760446,
            relative: false,
        },
        StepTau {
            name: "vel-video-4.f32",
            abs: 0.7903573513031006,
            rmse: 0.05641515715899659,
            rel: 0.026205407239694712,
            relative: false,
        },
        StepTau {
            name: "vel-audio-4.f32",
            abs: 1.454193115234375,
            rmse: 0.24726022475978954,
            rel: 0.1242327634348767,
            relative: false,
        },
        StepTau {
            name: "lat-video-4.f32",
            abs: 0.19100284576416016,
            rmse: 0.019096461467947093,
            rel: 0.0052730921097817116,
            relative: false,
        },
        StepTau {
            name: "lat-audio-4.f32",
            abs: 0.4078845977783203,
            rmse: 0.07943999013351834,
            rel: 0.029086524921537866,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-4.f32",
            abs: 0.7903573513031006,
            rmse: 0.05641515715899659,
            rel: 0.026205407239694712,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-4.f32",
            abs: 1.454193115234375,
            rmse: 0.24726022475978954,
            rel: 0.1242327634348767,
            relative: false,
        },
        StepTau {
            name: "vel-video-5.f32",
            abs: 0.5896859169006348,
            rmse: 0.05956043506078154,
            rel: 0.02773268014888072,
            relative: false,
        },
        StepTau {
            name: "vel-audio-5.f32",
            abs: 2.207589864730835,
            rmse: 0.32115949128539045,
            rel: 0.15805906485702587,
            relative: false,
        },
        StepTau {
            name: "lat-video-5.f32",
            abs: 0.1738595962524414,
            rmse: 0.01980567231133328,
            rel: 0.00564289666015423,
            relative: false,
        },
        StepTau {
            name: "lat-audio-5.f32",
            abs: 0.39391422271728516,
            rmse: 0.0792716483548706,
            rel: 0.029776176996678807,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-video-5.f32",
            abs: 0.5896859169006348,
            rmse: 0.05956043506078154,
            rel: 0.02773268014888072,
            relative: false,
        },
        StepTau {
            name: "teacher-vel-audio-5.f32",
            abs: 2.207589864730835,
            rmse: 0.32115949128539045,
            rel: 0.15805906485702587,
            relative: false,
        },
    ];

    fn step_tau(taus: &'static [StepTau], name: &str) -> Result<&'static StepTau> {
        taus.iter()
            .find(|tau| tau.name == name)
            .with_context(|| format!("no tau for {name}"))
    }

    type StepF32 = BTreeMap<String, Vec<f32>>;

    fn step_references(
        dir: &Path,
        hidden: usize,
        taus: &'static [StepTau],
    ) -> Result<(Vec<bf16>, StepF32)> {
        let mut saved = BTreeMap::new();
        let mut refiner = None;
        for tau in taus {
            if tau.name.ends_with(".bf16") {
                refiner = Some(read_bf16(&dir.join(tau.name), STEP_TEXT * hidden)?);
            } else {
                let elements = if tau.name.contains("video") {
                    STEP_GENERATED * STEP_VIDEO_WIDTH
                } else {
                    STEP_AUDIO * STEP_AUDIO_WIDTH
                };
                saved.insert(
                    tau.name.to_owned(),
                    read_f32(&dir.join(tau.name), elements)?,
                );
            }
        }
        Ok((refiner.context("refiner fixture")?, saved))
    }

    fn record_step(
        arm: &str,
        tau: &'static StepTau,
        candidate: &[f64],
        reference: &[f64],
        scored: &mut Vec<&'static str>,
        defects: &mut Vec<&'static str>,
    ) -> Result<()> {
        let (abs, rmse) = max_rmse(candidate, reference)?;
        let square: f64 = reference.iter().map(|value| value * value).sum();
        let rms = (square / reference.len() as f64).sqrt();
        ensure!(rms.is_finite() && rms > 0.0, "{} reference rms", tau.name);
        let rel = rmse / rms;
        let pass = if tau.relative {
            rel <= tau.rel
        } else {
            abs <= tau.abs && rmse <= tau.rmse
        };
        let gate = if tau.relative {
            "relative-RMSE"
        } else {
            "max-abs+RMSE"
        };
        println!(
            "HYPERFLOW_STEP arm={arm} cell={} max_abs={abs:.16e} rmse={rmse:.16e} rel={rel:.16e} tau_abs={:.16e} tau_rmse={:.16e} tau_rel={:.16e} gate={gate} {}",
            tau.name,
            tau.abs,
            tau.rmse,
            tau.rel,
            if pass { "PASS" } else { "DEFECT" }
        );
        scored.push(tau.name);
        if !pass {
            defects.push(tau.name);
        }
        Ok(())
    }

    fn score_f32(
        (arm, taus): (&str, &'static [StepTau]),
        name: &str,
        candidate: &[f32],
        saved: &StepF32,
        scored: &mut Vec<&'static str>,
        defects: &mut Vec<&'static str>,
    ) -> Result<()> {
        let tau = step_tau(taus, name)?;
        let reference = saved.get(name).with_context(|| format!("saved {name}"))?;
        record_step(
            arm,
            tau,
            &f64_of_f32(candidate),
            &f64_of_f32(reference),
            scored,
            defects,
        )
        .with_context(|| name.to_string())
    }

    fn note_refiner(seen: &mut Option<Vec<u16>>, now: &[bf16], label: &str) -> Result<()> {
        ensure!(
            now.iter().all(|value| value.to_f32().is_finite()),
            "{label} refiner is non-finite"
        );
        let bits: Vec<u16> = now.iter().map(|value| value.to_bits()).collect();
        if let Some(first) = seen {
            ensure!(bits == *first, "{label} refiner changed");
        } else {
            *seen = Some(bits);
        }
        Ok(())
    }

    /// Refiner rows and six free and teacher-forced steps at the frozen pair taus. A numeric miss returns Err.
    #[test]
    #[ignore = "requires a reserved GPU window; FF_HYPERFLOW_MODEL, FF_HYPERFLOW_STEP_REF"]
    fn hyperflow_step_cells_match_the_definition() -> Result<()> {
        let model = env_dir("FF_HYPERFLOW_MODEL")?;
        let reference = env_dir("FF_HYPERFLOW_STEP_REF")?;
        let arm = reference
            .file_name()
            .and_then(|name| name.to_str())
            .context("reference arm name")?
            .to_owned();
        let spec = H3Spec::parse(&serde_json::from_slice(&fs::read(
            model.join("config.json"),
        )?)?)?;
        ensure!(spec.is_hyperflow()?, "transformer config is not HyperFlow");
        ensure!(
            spec.hidden == 5376
                && spec.text_dim == 5120
                && spec.video_row_width() == STEP_VIDEO_WIDTH
                && spec.audio_in_channels == STEP_AUDIO_WIDTH,
            "config shape hidden {} text {} video width {} audio width {}",
            spec.hidden,
            spec.text_dim,
            spec.video_row_width(),
            spec.audio_in_channels
        );
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(reference.join("meta.json"))?)?;
        ensure!(
            meta["candidate_run"].as_bool() == Some(false),
            "reference arm is marked as a candidate run"
        );
        ensure!(
            meta["rows"].as_u64() == Some(u64::try_from(STEP_ROWS)?)
                && meta["video_rows"].as_u64() == Some(u64::try_from(STEP_VIDEO)?)
                && meta["teacher_from"].as_bool() == Some(true),
            "reference meta rows"
        );
        let layout = step_layout();
        ensure!(
            layout.rows() == STEP_ROWS
                && layout.video_indices.len() == STEP_VIDEO
                && layout.audio_indices.len() == STEP_AUDIO
                && layout.text_indices.len() == STEP_TEXT
                && layout.condition_video_rows == STEP_KEY,
            "pack layout"
        );
        let video_grid = step_grid(12.0);
        let audio_grid = step_grid(3.0);
        assert_step_grids(&video_grid, &audio_grid)?;
        let flow = hyperflow_steps(6)?;
        let row_plan: Vec<RowTimesteps> = flow
            .iter()
            .map(|step| RowTimesteps::hyperflow(&layout, step))
            .collect();
        let plan = AdalnPlan::build(&layout, &row_plan)?;
        ensure!(
            plan.row_pairs.len() == 6
                && plan.row_timesteps.len() == 6
                && plan
                    .row_pairs
                    .iter()
                    .chain(&plan.row_timesteps)
                    .all(|row| row.len() == STEP_ROWS),
            "adaln plan extent"
        );
        let taus = if spec.int8_convrot.is_some() {
            STEP_TAUS
        } else {
            STEP_BF16_TAUS
        };
        let (refiner_ref, saved) = step_references(&reference, spec.hidden, taus)?;
        let manifest = MappedCheckpoint::open_component(&model, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-hyperflow-step-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let _runtime = BlockRuntime(runtime.clone());
        let mut blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "HyperFlow step cells".into())),
        )?;
        let table = AdalnTable::build(&device, &mut blas, &model, &manifest, &spec, &plan)?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (0, &layout),
            Source::new(&model, &manifest),
            None,
            (
                Placement::Derived,
                crate::prefill::placement_visits(video_grid.timesteps.len(), 1)?,
            ),
        )?;
        ensure!(
            dit.work
                .stored
                .as_ref()
                .is_some_and(|s| s.convrot.is_some() == spec.int8_convrot.is_some()),
            "convrot scratch allocation disagrees with quantization presence"
        );
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 0.0,
        };
        let counts = (
            u32::try_from(plan.pairs.len())?,
            u32::try_from(plan.timesteps.len())?,
        );
        let video_count = STEP_VIDEO * STEP_VIDEO_WIDTH;
        let audio_count = STEP_AUDIO * STEP_AUDIO_WIDTH;
        let text_count = STEP_TEXT * spec.text_dim;
        let blob = step_pattern(video_count + audio_count + text_count);
        let video_host = blob[..video_count].to_vec();
        let audio_init = blob[video_count..video_count + audio_count].to_vec();
        let text_host: Vec<bf16> = blob[video_count + audio_count..]
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect();
        let key_elems = STEP_KEY * STEP_VIDEO_WIDTH;
        let generated_elems = STEP_GENERATED * STEP_VIDEO_WIDTH;
        let text = s.clone_htod(&text_host)?;
        let video = s.clone_htod(&video_host)?;
        let audio = s.clone_htod(&audio_init)?;
        let mut refiner_seen = None;
        let mut scored = Vec::new();
        let mut defects = Vec::new();
        for k in 0..6 {
            let row_pairs = s.clone_htod(&plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&plan.row_timesteps[k])?;
            dit.step(
                &context,
                &table,
                counts,
                &StepInputs {
                    video: &video,
                    audio: &audio,
                    text: &text,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            let refiner = s.clone_dtoh(&dit.text)?;
            ensure!(
                refiner.len() == STEP_TEXT * spec.hidden,
                "free step {k} refiner extent"
            );
            note_refiner(&mut refiner_seen, &refiner, &format!("free step {k}"))?;
            if k == 0 {
                record_step(
                    &arm,
                    step_tau(taus, "refiner.bf16")?,
                    &f64_of_bf16(&refiner),
                    &f64_of_bf16(&refiner_ref),
                    &mut scored,
                    &mut defects,
                )?;
            }
            let vel_video = s.clone_dtoh(&dit.video)?;
            let vel_audio = s.clone_dtoh(&dit.audio)?;
            score_f32(
                (&arm, taus),
                &format!("vel-video-{k}.f32"),
                &vel_video,
                &saved,
                &mut scored,
                &mut defects,
            )?;
            score_f32(
                (&arm, taus),
                &format!("vel-audio-{k}.f32"),
                &vel_audio,
                &saved,
                &mut scored,
                &mut defects,
            )?;
            dit.euler(
                s,
                (&video, key_elems, generated_elems),
                &dit.video,
                &video_grid,
                k,
            )?;
            dit.euler(s, (&audio, 0, audio_count), &dit.audio, &audio_grid, k)?;
            let video_now = s.clone_dtoh(&video)?;
            let audio_now = s.clone_dtoh(&audio)?;
            ensure!(
                video_now.len() == video_host.len()
                    && video_now[..key_elems]
                        .iter()
                        .zip(&video_host[..key_elems])
                        .all(|(got, want)| got.to_bits() == want.to_bits()),
                "free step {k} keyframe changed"
            );
            score_f32(
                (&arm, taus),
                &format!("lat-video-{k}.f32"),
                &video_now[key_elems..],
                &saved,
                &mut scored,
                &mut defects,
            )?;
            score_f32(
                (&arm, taus),
                &format!("lat-audio-{k}.f32"),
                &audio_now,
                &saved,
                &mut scored,
                &mut defects,
            )?;
        }
        for k in 0..6 {
            let generated = if k == 0 {
                video_host[key_elems..].to_vec()
            } else {
                saved
                    .get(&format!("lat-video-{}.f32", k - 1))
                    .with_context(|| format!("teacher video latent {}", k - 1))?
                    .clone()
            };
            let audio_in = if k == 0 {
                audio_init.clone()
            } else {
                saved
                    .get(&format!("lat-audio-{}.f32", k - 1))
                    .with_context(|| format!("teacher audio latent {}", k - 1))?
                    .clone()
            };
            let mut packed = video_host[..key_elems].to_vec();
            packed.extend(generated);
            let video_k = s.clone_htod(&packed)?;
            let audio_k = s.clone_htod(&audio_in)?;
            let row_pairs = s.clone_htod(&plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&plan.row_timesteps[k])?;
            dit.step(
                &context,
                &table,
                counts,
                &StepInputs {
                    video: &video_k,
                    audio: &audio_k,
                    text: &text,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            let refiner = s.clone_dtoh(&dit.text)?;
            note_refiner(&mut refiner_seen, &refiner, &format!("teacher step {k}"))?;
            let vel_video = s.clone_dtoh(&dit.video)?;
            let vel_audio = s.clone_dtoh(&dit.audio)?;
            score_f32(
                (&arm, taus),
                &format!("teacher-vel-video-{k}.f32"),
                &vel_video,
                &saved,
                &mut scored,
                &mut defects,
            )?;
            score_f32(
                (&arm, taus),
                &format!("teacher-vel-audio-{k}.f32"),
                &vel_audio,
                &saved,
                &mut scored,
                &mut defects,
            )?;
        }
        ensure!(
            scored.len() == taus.len() && taus.iter().all(|tau| scored.contains(&tau.name)),
            "scored {} of {}",
            scored.len(),
            taus.len()
        );
        if defects.is_empty() {
            println!("HYPERFLOW_STEP_GATE PASS cells={}", taus.len());
            Ok(())
        } else {
            println!("HYPERFLOW_STEP_GATE DEFECT cells={defects:?}");
            bail!("HYPERFLOW_STEP_GATE DEFECT cells={defects:?}")
        }
    }

    /// Block 2 resident downs. Rank 3137 pitches to 3144 and rank 2092 pitches to 2096, inside each allocation.
    #[test]
    #[ignore = "requires a reserved GPU window; FF_HYPERFLOW_MODEL, FF_Q38_GPU_WINDOW=granted"]
    fn hyperflow_low_rank_overread_stays_in_allocation() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let model = env_dir("FF_HYPERFLOW_MODEL")?;
        let spec = H3Spec::parse(&serde_json::from_slice(&fs::read(
            model.join("config.json"),
        )?)?)?;
        ensure!(spec.is_hyperflow()?, "transformer config is not HyperFlow");
        let rank_of = |name: &str| -> Result<usize> {
            match spec.linear_storage(name)? {
                LinearStorage::LowRank { rank } => Ok(rank),
                _ => bail!("{name} is not low-rank"),
            }
        };
        let down_rank = rank_of("transformer_blocks.2.ff.net.0.proj")?;
        let out_rank = rank_of("transformer_blocks.2.attn.to_out.0")?;
        ensure!(
            down_rank == 3137 && out_rank == 2092,
            "block 2 production ranks {down_rank} {out_rank}"
        );
        let manifest = MappedCheckpoint::open_component(&model, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-hyperflow-overread-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let _runtime = BlockRuntime(runtime.clone());
        let blas = open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "HyperFlow low-rank over-read".into())),
        )?;
        let mut source = Source::new(&model, &manifest);
        source.pack_vectors(s, &spec)?;
        let weights = BlockWeights::stored(s, &spec, "transformer_blocks.2", &mut source, None)?;
        let check = |linear: &BlockLinear, rank: usize| -> Result<()> {
            let BlockLinear::LowRank { down, owned, .. } = linear else {
                bail!("block 2 factor is not low-rank");
            };
            let pitch = rank.div_ceil(8) * 8;
            let k = down.shape[1] as usize;
            ensure!(
                down.shape[0] as usize == pitch
                    && owned.len() == 2
                    && owned[0].len() == pitch * k
                    && down.span_bytes == u64::try_from(pitch * k * 2)?,
                "block 2 down allocation rank={rank} pitch={pitch} k={k} owned={}",
                owned.len()
            );
            Ok(())
        };
        check(&weights.linears[FF_IN], down_rank)?;
        check(&weights.linears[OUT], out_rank)?;
        let rows = 20usize;
        let mut work = BlockWork::new(s, &spec, 0, rows)?;
        work.prepare::<false>(&blas, &spec)?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let hidden = s.clone_htod(&vec![bf16::from_f32(1.0); rows * spec.hidden])?;
        let rotary = s.alloc_zeros::<bf16>(rows * 6 * spec.rope_freq_dim)?;
        let entries = MODALITIES;
        let table = s.alloc_zeros::<bf16>(entries as usize * 6 * spec.hidden)?;
        let index = s.alloc_zeros::<u32>(rows)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 0.0,
        };
        let mut row = hidden;
        block_forward::<false>(
            &context,
            &mut flash,
            &weights,
            &Conditioning::Modulated {
                rotary: &rotary,
                modulation: &Modulation {
                    table: &table,
                    offset: 0,
                    entries,
                    index: &index,
                },
            },
            &mut row,
            &mut work,
        )?;
        let host = s.clone_dtoh(&row)?;
        ensure!(
            host.len() == rows * spec.hidden && host.iter().all(|v| v.to_f32().is_finite()),
            "block 2 output extent or a non-finite value"
        );
        println!(
            "HYPERFLOW_LOW_RANK_OVERREAD PASS block=2 down_rank={down_rank} down_pitch={} out_rank={out_rank} out_pitch={} rows={rows}",
            down_rank.div_ceil(8) * 8,
            out_rank.div_ceil(8) * 8
        );
        Ok(())
    }
}

#[cfg(test)]
mod h3_direct_batch_tests {
    use super::*;

    #[test]
    fn direct_batch_matches_the_serial_read() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-h3-direct-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let shard = dir.join("shard.bin");
        let align = 4096usize;
        let len = 64 * align;
        let file: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        std::fs::write(&shard, &file)?;
        let entry = |start: u64, bytes: usize| TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![bytes / 2],
            shard: "shard.bin".to_owned(),
            bytes,
            load: true,
            data_start: start,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let entries = [
            entry(align as u64 + 96, 5 * align / 2),
            entry(32 * align as u64 + 33, 3 * align),
        ];
        let mut reader = EntryReader::new(&dir);
        // Packed destinations at arbitrary offsets, as the dense fill produces.
        let mut offsets = [0usize; 2];
        let mut cursor = 0usize;
        for (index, entry) in entries.iter().enumerate() {
            offsets[index] = cursor + 123;
            cursor += entry.bytes + 123;
        }
        let total = cursor + 2 * align;
        let mut arena = AlignedBytes::new(total, align)?;
        let mut requests = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            requests.push(DirectRead {
                name: "t",
                entry,
                dest: unsafe { arena.as_mut().as_mut_ptr().add(offsets[index]) },
                dest_len: entry.bytes,
                cast: FillCast::Raw,
            });
        }
        reader.read_many_direct(&mut requests)?;
        for (index, entry) in entries.iter().enumerate() {
            let got = &arena.as_mut()[offsets[index]..offsets[index] + entry.bytes];
            let want = &file[entry.data_start as usize..entry.data_start as usize + entry.bytes];
            ensure!(got == want, "tensor {index}: batch bytes differ");
        }
        reader.read_many_direct(&mut [])?;
        let entry = &entries[0];
        let mut overlap = [
            DirectRead {
                name: "t",
                entry,
                dest: arena.as_mut().as_mut_ptr(),
                dest_len: total,
                cast: FillCast::Raw,
            },
            DirectRead {
                name: "t",
                entry,
                dest: unsafe { arena.as_mut().as_mut_ptr().add(offsets[0]) },
                dest_len: entry.bytes,
                cast: FillCast::Raw,
            },
        ];
        ensure!(
            reader.read_many_direct(&mut overlap).is_err(),
            "overlapping destination windows accepted"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    fn owned(entry: TensorEntry, arena: &mut AlignedBytes, at: usize) -> OwnedRead {
        OwnedRead::new(
            "t",
            &entry,
            unsafe { arena.as_mut().as_mut_ptr().add(at) },
            entry.bytes,
            FillCast::Raw,
        )
    }

    #[test]
    fn background_fill_failure_propagates_the_readers_text() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-h3-fillfail-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let align = 4096usize;
        let entry = TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![align / 2],
            shard: "missing.bin".to_owned(),
            bytes: align,
            load: true,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let mut arena = AlignedBytes::new(2 * align, align)?;
        let request = owned(entry, &mut arena, 0);
        let mut thread = FillThread::spawn(&dir, vec![request.clone()])?;
        let error = thread
            .join()
            .expect_err("fill of a missing shard succeeded");
        let text = format!("{error:#}");
        ensure!(
            text.contains("open native shard") && text.contains("missing.bin"),
            "background fill error lacks the reader's text: {text}"
        );
        let error = run_fill_batch(&dir, std::slice::from_ref(&request))
            .expect_err("serial fill of a missing shard succeeded");
        ensure!(
            format!("{error:#}").contains("open native shard"),
            "serial fill error lacks the reader's text: {error:#}"
        );
        // A failed pending fill stays failed: every later entry replays the reader's text.
        let mut pending = PendingFill {
            thread: Some(FillThread::spawn(&dir, vec![request.clone()])?),
            bounce: 0,
            checks: Vec::new(),
            failure: None,
        };
        let first = format!(
            "{:#}",
            pending.join_fill().expect_err("first entry succeeded")
        );
        let second = format!(
            "{:#}",
            pending.join_fill().expect_err("second entry succeeded")
        );
        ensure!(
            first == second && first.contains("open native shard"),
            "replayed fill error differs: first={first} second={second}"
        );
        // The resident-session form: after a successful join the pending is spent, so a
        // second entry finds the joined tier without rejoining or erroring.
        let mut arena_ok = AlignedBytes::new(2 * align, align)?;
        let good = TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![align],
            shard: "shard.bin".to_owned(),
            bytes: 2 * align,
            load: true,
            data_start: align as u64,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let file: Vec<u8> = (0..16 * align).map(|i| (i % 251) as u8).collect();
        std::fs::write(dir.join("shard.bin"), &file)?;
        let mut spent = PendingFill {
            thread: Some(FillThread::spawn(
                &dir,
                vec![owned(good, &mut arena_ok, 0)],
            )?),
            bounce: 0,
            checks: Vec::new(),
            failure: None,
        };
        spent.join_fill()?;
        ensure!(spent.thread.is_none(), "successful join kept the thread");
        spent.join_fill()?;
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// The arena a blocked fill writes into must not free before the fill joins. A FIFO
    /// shard blocks the worker in its O_DIRECT open until a writer appears, so the owner's
    /// drop is observed in two orders: the tier's (join first) and the freed-arena-first
    /// control that mirrors the reversed declaration order.
    #[test]
    fn dropping_a_tier_joins_its_fill_before_the_arena_frees() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-h3-filldrop-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let fifo = dir.join("blocked.bin");
        let path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes())
            .context("fifo path encoding")?;
        let rc = unsafe { libc::mkfifo(path.as_ptr(), 0o600) };
        ensure!(rc == 0, "mkfifo {fifo:?} failed: {rc}");
        let align = 4096usize;
        let entry = TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![align / 2],
            shard: "blocked.bin".to_owned(),
            bytes: align,
            load: true,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        struct ArenaFreed(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for ArenaFreed {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        /// Mirrors `DenseTier`: the pending fill joins before the arena drops.
        struct TierOrder {
            fill: Option<PendingFill>,
            arena: ArenaFreed,
        }
        impl Drop for TierOrder {
            fn drop(&mut self) {
                self.fill.take();
            }
        }
        /// The reversed order: the arena frees while the fill has not joined.
        struct TierFreedFirst {
            arena: ArenaFreed,
            fill: Option<PendingFill>,
        }
        let spawn_pending = |arena: &mut AlignedBytes| -> Result<PendingFill> {
            Ok(PendingFill {
                thread: Some(FillThread::spawn(
                    &dir,
                    vec![owned(entry.clone(), arena, 0)],
                )?),
                bounce: 0,
                checks: Vec::new(),
                failure: None,
            })
        };

        let mut arena = AlignedBytes::new(2 * align, align)?;
        let freed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tier = TierOrder {
            fill: Some(spawn_pending(&mut arena)?),
            arena: ArenaFreed(std::sync::Arc::clone(&freed)),
        };
        ensure!(
            tier.fill.is_some() && !tier.arena.0.load(std::sync::atomic::Ordering::SeqCst),
            "tier starts with a pending fill over a live arena"
        );
        let joined = std::thread::spawn(move || drop(tier));
        for _ in 0..150 {
            ensure!(
                !freed.load(std::sync::atomic::Ordering::SeqCst),
                "arena freed while the fill had not joined"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let mut control_arena = AlignedBytes::new(2 * align, align)?;
        let control_freed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let control = TierFreedFirst {
            arena: ArenaFreed(std::sync::Arc::clone(&control_freed)),
            fill: Some(spawn_pending(&mut control_arena)?),
        };
        ensure!(
            control.fill.is_some() && !control.arena.0.load(std::sync::atomic::Ordering::SeqCst),
            "control starts with a pending fill over a live arena"
        );
        let control_joined = std::thread::spawn(move || drop(control));
        let mut observed = false;
        for _ in 0..200 {
            if control_freed.load(std::sync::atomic::Ordering::SeqCst) {
                observed = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        ensure!(
            observed,
            "control order never freed the arena; the observation is blind"
        );

        std::fs::OpenOptions::new()
            .write(true)
            .open(&fifo)
            .context("open fifo writer")?;
        joined
            .join()
            .map_err(|_| anyhow::anyhow!("tier dropper panicked"))?;
        control_joined
            .join()
            .map_err(|_| anyhow::anyhow!("control dropper panicked"))?;
        ensure!(
            freed.load(std::sync::atomic::Ordering::SeqCst),
            "arena not freed after the fill joined"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn resident_raw_finite_rejects_nan_kept_as_f32() -> Result<()> {
        let entry = |dtype: Dtype| TensorEntry {
            dtype,
            shape: vec![8],
            shard: "s".to_owned(),
            bytes: 32,
            load: true,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let mut nan = vec![0u8; 32];
        nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        resident_raw_finite("t", &entry(Dtype::F32), FillCast::Raw, &nan)
            .expect_err("nonfinite raw F32 accepted");
        let finite = vec![0u8; 32];
        resident_raw_finite("t", &entry(Dtype::F32), FillCast::Raw, &finite)?;
        // The narrowed form validates inside the cast, not here.
        resident_raw_finite("t", &entry(Dtype::F32), FillCast::F32ToBf16, &nan)?;
        Ok(())
    }

    #[test]
    fn converted_fill_narrows_f32_to_bf16_and_rejects_overflow() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-h3-fillcast-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let align = 4096usize;
        let values: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.25 - 128.0).collect();
        let mut source = Vec::with_capacity(values.len() * 4);
        for value in &values {
            source.extend_from_slice(&value.to_le_bytes());
        }
        while source.len() < 2 * align {
            source.extend_from_slice(&0.0f32.to_le_bytes());
        }
        std::fs::write(dir.join("shard.bin"), &source)?;
        let entry = |shard: &str| TensorEntry {
            dtype: Dtype::F32,
            shape: vec![source.len() / 4],
            shard: shard.to_owned(),
            bytes: source.len(),
            load: true,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let mut arena = AlignedBytes::new(align, align)?;
        let request = OwnedRead::new(
            "narrow",
            &entry("shard.bin"),
            arena.as_mut().as_mut_ptr(),
            align,
            FillCast::F32ToBf16,
        );
        let dir2 = dir.clone();
        let mut thread = FillThread::spawn(&dir2, vec![request])?;
        thread.join()?;
        let bytes = arena.as_mut();
        for (index, value) in values.iter().enumerate() {
            let got = bf16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]);
            ensure!(got == bf16::from_f32(*value), "element {index}: {got}");
        }
        // The overflow arm: a value beyond BF16 range must fail with the loader's text.
        let mut hot = source.clone();
        hot[..4].copy_from_slice(&f32::MAX.to_le_bytes());
        std::fs::write(dir.join("shard.bin"), &hot)?;
        let mut overflow = AlignedBytes::new(align, align)?;
        let request = OwnedRead::new(
            "narrow",
            &entry("shard.bin"),
            overflow.as_mut().as_mut_ptr(),
            align,
            FillCast::F32ToBf16,
        );
        let mut thread = FillThread::spawn(&dir, vec![request])?;
        let error = thread.join().expect_err("overflowing F32 narrowed to BF16");
        ensure!(
            format!("{error:#}").contains("BF16 overflow for narrow"),
            "overflow error lacks the loader's text: {error:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn fill_schedule_keeps_serial_when_the_derived_host_budget_does_not_fit() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-h3-fillsched-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let align = 4096usize;
        let file: Vec<u8> = (0..16 * align).map(|i| (i % 251) as u8).collect();
        std::fs::write(dir.join("shard.bin"), &file)?;
        let entry = TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![align],
            shard: "shard.bin".to_owned(),
            bytes: 2 * align,
            load: true,
            data_start: align as u64,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let mut arena = AlignedBytes::new(4 * align, align)?;
        let request = owned(entry, &mut arena, 0);
        let (schedule, bounce) = schedule_fill("gate", &dir, Vec::new(), 0)?;
        ensure!(
            matches!(schedule, FillSchedule::Serial(_)) && bounce == 0,
            "empty batch scheduled in the background"
        );
        let (schedule, _) = schedule_fill("gate", &dir, vec![request.clone()], usize::MAX / 2)?;
        ensure!(
            matches!(schedule, FillSchedule::Serial(_)),
            "unusable host budget scheduled in the background"
        );
        let before = LIVE_FILL_BOUNCE_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        let (schedule, bounce) = schedule_fill("gate", &dir, vec![request], 0)?;
        match schedule {
            FillSchedule::Background(mut thread) => {
                ensure!(bounce > 0, "background fill reserved no bounce bound");
                ensure!(
                    LIVE_FILL_BOUNCE_BYTES.load(std::sync::atomic::Ordering::SeqCst)
                        == before + bounce,
                    "background fill did not reserve its bounce bound"
                );
                thread.join()?;
                LIVE_FILL_BOUNCE_BYTES.fetch_sub(bounce, std::sync::atomic::Ordering::SeqCst);
            }
            FillSchedule::Serial(_) => bail!("zero reserve stayed serial on a healthy host"),
        }
        ensure!(
            LIVE_FILL_BOUNCE_BYTES.load(std::sync::atomic::Ordering::SeqCst) == before,
            "fill bounce reservation leaked"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn prefer_hyperflow_bf16_follows_arch_and_memory_fit() -> Result<()> {
        let mut spec = H3Spec::parse(&serde_json::from_str(include_str!(
            "../../../tests/data/minimax-h3-transformer-config.json"
        ))?)?;
        spec.hyperflow_gate = Some(0.25);
        spec.dropped_blocks = Some(vec![6, 21, 22]);
        spec.lowrank_ranks = Some(std::collections::BTreeMap::from([(
            "transformer_blocks.1.attn.to_q".to_owned(),
            1024usize,
        )]));
        let geometry = crate::minimax_h3::Geometry::resolve(&spec, 768, 1280, 124)?;
        let weights = 27_433_807_488u64;
        let activations = crate::minimax_h3::unchunked_activation_bytes(&spec, &geometry)?;
        let need = weights + activations;
        let gfx11 = crate::backend::setup::DeviceArchitecture::Hip("gfx1100".into());
        let sm89 = crate::backend::setup::DeviceArchitecture::Cuda(89);
        assert!(bf16_preferred_for_device(
            &gfx11, &spec, &geometry, need, weights
        )?);
        assert!(!bf16_preferred_for_device(
            &gfx11,
            &spec,
            &geometry,
            need - 1,
            weights
        )?);
        assert!(!bf16_preferred_for_device(
            &sm89, &spec, &geometry, need, weights
        )?);
        Ok(())
    }

    #[test]
    fn bf16_to_fp16_fill_flushes_and_rejects_overflow() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-h3-fp16-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let align = 4096usize;
        let name = "model.language_model.layers.1.mlp.gate_proj.weight";
        let mut source = vec![0u8; 2 * align];
        let store = |source: &mut [u8], index: usize, bits: u16| {
            let bytes = bits.to_le_bytes();
            source[2 * index] = bytes[0];
            source[2 * index + 1] = bytes[1];
        };
        store(&mut source, 0, 142 << 7);
        store(&mut source, 1, (1 << 15) | (102 << 7) | 127);
        store(&mut source, 2, 103 << 7);
        store(&mut source, 3, 1 << 15);
        std::fs::write(dir.join("shard.bin"), &source)?;
        let entry = TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![source.len() / 2],
            shard: "shard.bin".into(),
            bytes: source.len(),
            load: true,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: String::new(),
            scale_name: None,
            quant: None,
            swap_halves: false,
        };
        let nbytes = source.len();
        let read = |arena: &mut AlignedBytes, cast: FillCast| {
            OwnedRead::new(name, &entry, arena.as_mut().as_mut_ptr(), nbytes, cast)
        };
        let mut raw = AlignedBytes::new(source.len(), align)?;
        let mut raw_thread = FillThread::spawn(&dir, vec![read(&mut raw, FillCast::Raw)])?;
        raw_thread.join()?;
        ensure!(
            u16::from_le_bytes([raw.as_mut()[2], raw.as_mut()[3]]) == (1 << 15) | (102 << 7) | 127,
            "raw fill rewrote BF16 bits"
        );
        let mut arena = AlignedBytes::new(source.len(), align)?;
        let mut thread = FillThread::spawn(&dir, vec![read(&mut arena, FillCast::Bf16ToFp16)])?;
        thread.join()?;
        let word = |bytes: &[u8], index: usize| {
            u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])
        };
        let got = arena.as_mut();
        ensure!(word(got, 0) == 0x7800, "exponent 142");
        ensure!(word(got, 1) == 0, "exponent 102");
        ensure!(word(got, 2) == 1, "exponent 103");
        ensure!(word(got, 3) == 0x8000, "negative zero");
        store(&mut source, 0, 143 << 7);
        std::fs::write(dir.join("shard.bin"), &source)?;
        let mut overflow = AlignedBytes::new(source.len(), align)?;
        let mut thread = FillThread::spawn(&dir, vec![read(&mut overflow, FillCast::Bf16ToFp16)])?;
        let error = thread.join().expect_err("BF16 exponent 143");
        ensure!(
            format!("{error:#}").contains(name),
            "overflow error lacks the tensor name: {error:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
