//! Two reusable pinned upload slots, ordered independently from model compute.
use super::cuda::WeightPool;
use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, HostSlice, LaunchConfig, PinnedHostSlice, PushKernelArg,
    SyncOnDrop, sys,
};
use candle_core::{DType, Storage, Tensor, op::BackpropOp};
use ff_core::weights::ModelWeights;
use std::sync::{Arc, Mutex};

/// Preserve cudarc's pinned-memory event guard when copying only a prefix.
struct Prefix<'a, T> {
    data: &'a mut PinnedHostSlice<T>,
    /// Payload offset within the aligned direct-read span.
    offset: usize,
    len: usize,
}

impl<T> HostSlice<T> for Prefix<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        let (data, guard) = unsafe { self.data.stream_synced_slice(stream) };
        (&data[self.offset..self.offset + self.len], guard)
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        let (data, guard) = unsafe { self.data.stream_synced_mut_slice(stream) };
        (&mut data[self.offset..self.offset + self.len], guard)
    }
}

struct Slot {
    host_weight: PinnedHostSlice<u8>,
    host_scales: PinnedHostSlice<f32>,
    weight: CudaSlice<u8>,
    scales: CudaSlice<f32>,
}

impl Slot {
    fn new(stream: &Arc<CudaStream>, weight_bytes: usize, scale_count: usize) -> Result<Self> {
        let context = stream.context();
        let mut host_weight = unsafe { context.alloc_pinned::<u8>(weight_bytes) }?;
        let mut host_scales = unsafe { context.alloc_pinned::<f32>(scale_count) }?;
        unsafe {
            host_weight.as_mut_ptr()?.write_bytes(0, weight_bytes);
            host_scales.as_mut_ptr()?.write_bytes(0, scale_count);
        }
        Ok(Self {
            host_weight,
            host_scales,
            weight: unsafe { stream.alloc(weight_bytes) }?,
            scales: unsafe { stream.alloc(scale_count) }?,
        })
    }
    fn bytes(&self) -> usize {
        self.weight.len() + self.scales.len() * 4
    }
}

pub(super) struct Staging {
    stream: Arc<CudaStream>,
    slots: [Option<Slot>; 2],
    next: usize,
    uploads: u64,
    uploaded_bytes: u64,
    slot_allocations: u64,
    read: (usize, Option<usize>),
}

impl Staging {
    pub(super) fn new(
        device: &candle_core::CudaDevice,
        read: (usize, Option<usize>),
    ) -> Result<Self> {
        ensure!(
            device.cuda_stream().context().is_event_tracking(),
            "pinned FP8 transfer requires CUDA event tracking"
        );
        let compute = device.cuda_stream();
        let stream = compute.context().new_stream()?;
        Ok(Self {
            stream,
            slots: [None, None],
            next: 0,
            uploads: 0,
            uploaded_bytes: 0,
            slot_allocations: 0,
            read,
        })
    }
    pub(super) fn bytes(&self) -> u64 {
        self.slots.iter().flatten().map(Slot::bytes).sum::<usize>() as u64
    }

    /// Record lane completion so the compute stream can wait before reading output.
    pub(super) fn ready_event(&self) -> Result<CudaEvent> {
        self.stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(Into::into)
    }

    pub(super) fn stats(&self) -> super::Fp8TransferStats {
        super::Fp8TransferStats {
            uploads: self.uploads,
            uploaded_bytes: self.uploaded_bytes,
            slot_allocations: self.slot_allocations,
            staging_bytes_per_tier: self.slots.iter().flatten().map(Slot::bytes).sum::<usize>()
                as u64,
        }
    }

    pub(super) fn load(
        &mut self,
        weights: &ModelWeights,
        name: &str,
        scale_name: &str,
        device: &candle_core::CudaDevice,
        dtype: DType,
        pool: &WeightPool,
    ) -> Result<Tensor> {
        super::validate_block_fp8_metadata(weights, name, scale_name, dtype)?;
        let meta = FillMeta::of(weights, name, scale_name)?;
        let stream = self.stream.clone();
        let index = self.next;
        self.ensure_slot(index, &meta)?;
        let slot = self.slots[index].as_mut().unwrap();
        let weight_delta = fill_prepared_into(
            weights,
            name,
            scale_name,
            &meta,
            &mut slot.host_weight,
            &mut slot.host_scales,
            self.read,
        )?;
        stream.memcpy_htod(
            &Prefix {
                data: &mut slot.host_weight,
                offset: weight_delta as usize,
                len: meta.count,
            },
            &mut slot.weight.slice_mut(..meta.count),
        )?;
        stream.memcpy_htod(
            &Prefix {
                data: &mut slot.host_scales,
                offset: 0,
                len: meta.scale_count,
            },
            &mut slot.scales.slice_mut(..meta.scale_count),
        )?;
        self.dequant_launched(device, dtype, pool, &meta)
    }

    /// Grow slot buffers to fit `meta`, including host padding for aligned direct reads.
    fn ensure_slot(&mut self, index: usize, meta: &FillMeta) -> Result<()> {
        if self.slots[index]
            .as_ref()
            .is_none_or(|s| s.weight.len() < meta.count || s.scales.len() < meta.scale_count)
        {
            self.slots[index] = None;
            self.slots[index] = Some(Slot::new(
                &self.stream,
                meta.count + DIRECT_FILL_SLACK,
                meta.scale_count,
            )?);
            self.slot_allocations += 1;
        }
        Ok(())
    }

    /// Queue lane dequantization and advance the slot and counters.
    fn dequant_launched(
        &mut self,
        device: &candle_core::CudaDevice,
        dtype: DType,
        pool: &WeightPool,
        meta: &FillMeta,
    ) -> Result<Tensor> {
        let stream = self.stream.clone();
        let index = self.next;
        let slot = self.slots[index].as_mut().unwrap();
        let count = meta.count;
        let rows = meta.rows;
        let cols = meta.cols;
        let name = match dtype {
            DType::BF16 => "glm_fp8_dequant_bf16",
            DType::F32 => "glm_fp8_dequant_f32",
            _ => unreachable!(),
        };
        let function = ff_core::cuda_kernel_assets::load_function(
            device,
            &crate::kernel_assets::GLM_FP8_DEQUANT,
            "glm_fp8_dequant_v1",
            name,
        )?;
        let raw = slot.weight.slice(..count);
        let scales = slot.scales.slice(..meta.scale_count);
        macro_rules! execute {
            ($ty:ty) => {{
                // Allocate on the compute stream so free/reuse follows the consuming matvec.
                // The lane waits for allocation before writing; the consumer separately waits
                // for lane completion before reading.
                let compute = device.cuda_stream();
                let mut allocated = unsafe { pool.allocate_on::<$ty>(&compute, count) }?;
                let block_ready =
                    compute.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                stream.wait(&block_ready)?;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&raw)
                    .arg(&scales)
                    .arg(&mut allocated)
                    .arg(&rows)
                    .arg(&cols);
                unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }?;
                candle_core::CudaStorage::wrap_cuda_slice(allocated, device.clone())
            }};
        }
        let output = if dtype == DType::BF16 {
            execute!(half::bf16)
        } else {
            execute!(f32)
        };
        self.next = (index + 1) % 2;
        self.uploads += 1;
        self.uploaded_bytes = self
            .uploaded_bytes
            .checked_add((count + meta.scale_count * 4) as u64)
            .context("FP8 transfer byte counter overflow")?;
        Ok(Tensor::from_storage(
            Storage::Cuda(output),
            meta.shape.clone(),
            BackpropOp::none(),
            false,
        ))
    }

    /// Upload and dequantize a staged projection. The returned H2D event must
    /// complete before `buffer` is refilled.
    pub(super) fn upload_prepared(
        &mut self,
        device: &candle_core::CudaDevice,
        dtype: DType,
        pool: &WeightPool,
        buffer: &mut FillBuffer,
        meta: &FillMeta,
    ) -> Result<(Tensor, CudaEvent)> {
        ensure!(
            buffer.host_weight.len() >= buffer.weight_delta as usize + meta.count
                && buffer.host_scales.len() >= meta.scale_count,
            "FP8 fill-ahead buffer shrank between fill and upload"
        );
        let stream = self.stream.clone();
        let index = self.next;
        self.ensure_slot(index, meta)?;
        let slot = self.slots[index].as_mut().unwrap();
        stream.memcpy_htod(
            &Prefix {
                data: &mut buffer.host_weight,
                offset: buffer.weight_delta as usize,
                len: meta.count,
            },
            &mut slot.weight.slice_mut(..meta.count),
        )?;
        stream.memcpy_htod(
            &Prefix {
                data: &mut buffer.host_scales,
                offset: 0,
                len: meta.scale_count,
            },
            &mut slot.scales.slice_mut(..meta.scale_count),
        )?;
        let drained = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let tensor = self.dequant_launched(device, dtype, pool, meta)?;
        Ok((tensor, drained))
    }
}

/// Validated projection dimensions shared by fill and upload.
pub(crate) struct FillMeta {
    pub(crate) count: usize,
    pub(crate) scale_count: usize,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) shape: Vec<usize>,
}

impl FillMeta {
    fn of(weights: &ModelWeights, name: &str, scale_name: &str) -> Result<Self> {
        let metadata = weights.raw_tensor_metadata(name)?;
        let scales = weights.raw_tensor_metadata(scale_name)?;
        let count = metadata.bytes;
        ensure!(
            count > 0 && count <= i32::MAX as usize && scales.bytes.is_multiple_of(4),
            "invalid FP8 staging dimensions"
        );
        Ok(Self {
            count,
            scale_count: scales.bytes / 4,
            rows: u32::try_from(metadata.shape[0])?,
            cols: u32::try_from(metadata.shape[1])?,
            shape: metadata.shape,
        })
    }
}

/// Pinned buffers and the last H2D event, which must complete before refill.
/// `weight_delta` locates the payload within an aligned direct-read span.
pub(crate) struct FillBuffer {
    pub(crate) host_weight: PinnedHostSlice<u8>,
    pub(crate) host_scales: PinnedHostSlice<f32>,
    pub(crate) weight_delta: u32,
    pub(crate) event: Option<CudaEvent>,
}

/// Pinned ring filled by host workers and uploaded in routing order.
pub(crate) struct FillRing {
    buffers: Vec<Mutex<FillBuffer>>,
    weight_bytes: usize,
    scale_count: usize,
}

impl FillRing {
    pub(crate) fn new(
        context: &Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
        depth: usize,
        weight_bytes: usize,
        scale_count: usize,
    ) -> Result<Self> {
        let mut buffers = Vec::with_capacity(depth.max(2));
        for _ in 0..depth.max(2) {
            let mut host_weight =
                unsafe { context.alloc_pinned::<u8>(weight_bytes + DIRECT_FILL_SLACK) }?;
            let mut host_scales = unsafe { context.alloc_pinned::<f32>(scale_count) }?;
            unsafe {
                host_weight
                    .as_mut_ptr()?
                    .write_bytes(0, weight_bytes + DIRECT_FILL_SLACK);
                host_scales.as_mut_ptr()?.write_bytes(0, scale_count);
            }
            buffers.push(Mutex::new(FillBuffer {
                host_weight,
                host_scales,
                weight_delta: 0,
                event: None,
            }));
        }
        Ok(Self {
            buffers,
            weight_bytes,
            scale_count,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.buffers.len()
    }

    pub(crate) fn buffer(&self, index: usize) -> &Mutex<FillBuffer> {
        &self.buffers[index]
    }

    /// Check both ring depth and per-buffer capacity.
    pub(crate) fn fits(&self, depth: usize, weight_bytes: usize, scale_count: usize) -> bool {
        self.buffers.len() >= depth.max(2)
            && self.weight_bytes >= weight_bytes
            && self.scale_count >= scale_count
    }
}

impl Drop for FillRing {
    fn drop(&mut self) {
        // Wait for H2D reads before freeing pinned pages. Ignore errors during teardown.
        for buffer in &self.buffers {
            if let Ok(mut guard) = buffer.lock()
                && let Some(event) = guard.event.take()
            {
                let _ = event.synchronize();
            }
        }
    }
}

/// Validate and fill a projection on a host worker without enqueueing GPU work.
/// The buffer mutex and release fence publish writes before upload.
pub(crate) fn fill_prepared(
    weights: &ModelWeights,
    name: &str,
    scale_name: &str,
    dtype: DType,
    buffer: &mut FillBuffer,
    read: (usize, Option<usize>),
) -> Result<FillMeta> {
    super::validate_block_fp8_metadata(weights, name, scale_name, dtype)?;
    let meta = FillMeta::of(weights, name, scale_name)?;
    ensure!(
        buffer.host_weight.len() >= meta.count && buffer.host_scales.len() >= meta.scale_count,
        "FP8 fill-ahead buffer undersized for {name}"
    );
    buffer.weight_delta = fill_prepared_into(
        weights,
        name,
        scale_name,
        &meta,
        &mut buffer.host_weight,
        &mut buffer.host_scales,
        read,
    )?;
    Ok(meta)
}

/// Read weights and convert little-endian scales into pinned buffers.
/// Returns the weight payload offset; aligned direct reads may leave leading padding.
/// The buffer mutex and trailing release fence publish writes before upload.
fn fill_prepared_into(
    weights: &ModelWeights,
    name: &str,
    scale_name: &str,
    meta: &FillMeta,
    host_weight: &mut PinnedHostSlice<u8>,
    host_scales: &mut PinnedHostSlice<f32>,
    read: (usize, Option<usize>),
) -> Result<u32> {
    // Platforms without positional pinned reads use the mmap view.
    let weight_delta =
        match fill_pinned_from_checkpoint(weights, name, meta.count, host_weight, read)? {
            Some(delta) => delta,
            None => {
                weights.with_tensor_bytes(name, |bytes| {
                    ensure!(
                        bytes.len() == meta.count,
                        "FP8 byte length changed after metadata inspection"
                    );
                    host_weight.as_mut_slice()?[..meta.count].copy_from_slice(bytes);
                    Ok(())
                })?;
                crate::io_trace::record_fill(crate::io_trace::FillKind::Mmap)?;
                0
            }
        };
    weights.with_tensor_bytes(scale_name, |bytes| {
        ensure!(
            bytes.len() == meta.scale_count * 4,
            "FP8 scale byte length changed"
        );
        for (dst, bytes) in host_scales.as_mut_slice()?[..meta.scale_count]
            .iter_mut()
            .zip(bytes.chunks_exact(4))
        {
            *dst = f32::from_le_bytes(bytes.try_into().unwrap());
        }
        Ok(())
    })?;
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    Ok(weight_delta)
}

/// Read a projection into pinned memory and return its payload offset.
/// Returns `None` when unavailable, leaving the caller to use the mmap view.
#[cfg(unix)]
fn fill_pinned_from_checkpoint(
    weights: &ModelWeights,
    name: &str,
    count: usize,
    host: &mut PinnedHostSlice<u8>,
    read: (usize, Option<usize>),
) -> Result<Option<u32>> {
    let metadata = weights.raw_tensor_metadata(name)?;
    ensure!(
        metadata.bytes == count,
        "FP8 checkpoint byte length changed for {name}"
    );
    let path = weights.root().join(&metadata.shard);
    let base = metadata.file_offset as u64;
    #[cfg(target_os = "linux")]
    if direct_fill_enabled()?
        && let Some(delta) = fill_direct(&path, base, count, host, read)?
    {
        crate::io_trace::record_fill(crate::io_trace::FillKind::Direct)?;
        return Ok(Some(delta));
    }
    crate::model::read_buffered(&path, base, &mut host.as_mut_slice()?[..count], read)
        .with_context(|| format!("buffered pinned fill failed for {name}"))?;
    crate::io_trace::record_fill(crate::io_trace::FillKind::Buffered)?;
    Ok(Some(0))
}

/// Read a system-aligned span with O_DIRECT, returning its payload offset.
/// An uncovered aligned span returns `Ok(None)` before reading.
#[cfg(target_os = "linux")]
fn fill_direct(
    path: &std::path::Path,
    base: u64,
    count: usize,
    host: &mut PinnedHostSlice<u8>,
    read: (usize, Option<usize>),
) -> Result<Option<u32>> {
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    let Some(alignment) = read.1 else {
        return Ok(None);
    };
    let alignment = u64::try_from(alignment)?;
    let aligned_base = base & !(alignment - 1);
    let delta = u32::try_from(base - aligned_base).context("direct fill offset exceeds u32")?;
    let mut span = (delta as usize + count).next_multiple_of(alignment as usize);
    let pointer = host.as_mut_ptr()? as usize;
    if !pointer.is_multiple_of(alignment as usize) || span > host.len() {
        return Ok(None);
    }
    // Use buffered reads if the payload extends beyond the aligned file end.
    let file_bytes = std::fs::metadata(path)
        .with_context(|| format!("failed to stat FP8 checkpoint shard {}", path.display()))?
        .len();
    let aligned_file_end = file_bytes & !(alignment - 1);
    if aligned_base + span as u64 > aligned_file_end {
        if base + count as u64 > aligned_file_end {
            return Ok(None);
        }
        span = usize::try_from(aligned_file_end - aligned_base)
            .context("direct fill span no longer covers its payload")?;
    }
    let chunk = crate::model::reader_chunk(span, read.0, read.1);
    let readers = span.div_ceil(chunk);
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for reader in 0..readers {
            let start = reader * chunk;
            if start >= span {
                break;
            }
            let end = (start + chunk).min(span);
            handles.push(scope.spawn(move || -> Result<()> {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)
                    .with_context(|| {
                        format!("failed to direct-open FP8 shard {}", path.display())
                    })?;
                let mut offset = start;
                while offset < end {
                    let destination = (pointer + offset) as *mut u8;
                    let wanted = (end - offset).min(4 << 20);
                    let slice = unsafe { std::slice::from_raw_parts_mut(destination, wanted) };
                    let read = file
                        .read_at(slice, aligned_base + offset as u64)
                        .context("direct positional read into pinned FP8 buffer failed")?;
                    ensure!(read > 0, "checkpoint shard ended mid-tensor");
                    // Keep the next read offset sector-aligned after a short read.
                    ensure!(
                        read.is_multiple_of(alignment as usize),
                        "direct read returned {read} unaligned bytes"
                    );
                    offset += read;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("direct FP8 fill reader panicked"))??;
        }
        Ok(())
    })?;
    Ok(Some(delta))
}

/// FF_GLM_DIRECT_FILL=1 enables direct reads for covered aligned spans.
#[cfg(target_os = "linux")]
fn direct_fill_enabled() -> Result<bool> {
    static ENABLED: std::sync::OnceLock<std::result::Result<usize, String>> =
        std::sync::OnceLock::new();
    Ok(ff_core::probe::cached_env_usize(&ENABLED, "FF_GLM_DIRECT_FILL", 0, 0, 1)? == 1)
}

/// Whether direct fills are enabled, allowing device page warming to be skipped.
#[cfg(target_os = "linux")]
pub(crate) fn direct_fill_active() -> Result<bool> {
    direct_fill_enabled()
}

/// Other platforms use buffered reads and benefit from page warming.
#[cfg(not(target_os = "linux"))]
pub(crate) fn direct_fill_active() -> Result<bool> {
    Ok(false)
}

/// Direct spans that exceed this reserved padding use buffered reads.
const DIRECT_FILL_SLACK: usize = super::PINNED_FILL_PADDING_BYTES;

#[cfg(not(unix))]
fn fill_pinned_from_checkpoint(
    _weights: &ModelWeights,
    _name: &str,
    _count: usize,
    _host: &mut PinnedHostSlice<u8>,
    _read: (usize, Option<usize>),
) -> Result<Option<u32>> {
    Ok(None)
}

impl Drop for Staging {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
