use crate::backend::gpu::{CudaFunction, CudaSlice, PushKernelArg};
use crate::backend::gpu::{
    CudaContext, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PinnedHostSlice, ValidAsZeroBits,
};
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use cudarc::driver::sys;
use half::bf16;
use std::sync::Arc;

pub fn pinned_zeros<T: DeviceRepr + ValidAsZeroBits>(
    context: &Arc<CudaContext>,
    len: usize,
) -> Result<PinnedHostSlice<T>> {
    // SAFETY: flags 0 allocates CPU-readable pinned storage, initialized before creating typed slices.
    let mut allocation = unsafe { context.alloc_pinned_with_flags(len, 0)? };
    unsafe {
        std::ptr::write_bytes(allocation.as_mut_ptr()?, 0, len);
    }
    Ok(allocation)
}

/// # Safety
/// Both spans stay live through completion; the caller orders all other accesses.
pub(crate) unsafe fn dtoh_async<T: DeviceRepr>(
    dst: &mut [T],
    src: u64,
    stream: &CudaStream,
) -> Result<()> {
    unsafe { dtoh_raw(dst.as_mut_ptr().cast(), src, size_of_val(dst), stream) }
}

/// # Safety
/// Both spans stay live through completion; the caller orders all other accesses.
pub(crate) unsafe fn htod_async<T: DeviceRepr>(
    dst: u64,
    src: &[T],
    stream: &CudaStream,
) -> Result<()> {
    unsafe { htod_raw(dst, src.as_ptr().cast(), size_of_val(src), stream) }
}

/// # Safety
/// The spans contain `bytes` valid bytes and stay live and ordered through completion.
pub(crate) unsafe fn dtoh_raw(
    dst: *mut std::ffi::c_void,
    src: u64,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemcpyDtoHAsync_v2(dst, src, bytes, stream.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        stream.copy_raw(dst as u64, src, bytes, 2)?;
    }
    Ok(())
}

/// # Safety
/// The spans contain `bytes` valid bytes and stay live and ordered through completion.
pub(crate) unsafe fn htod_raw(
    dst: u64,
    src: *const std::ffi::c_void,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemcpyHtoDAsync_v2(dst, src, bytes, stream.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        stream.copy_raw(dst, src as u64, bytes, 1)?;
    }
    Ok(())
}

/// # Safety
/// The device spans contain `bytes` valid bytes and stay live and ordered through completion.
pub(crate) unsafe fn dtod_raw(dst: u64, src: u64, bytes: usize, stream: &CudaStream) -> Result<()> {
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        stream.copy_raw(dst, src, bytes, 3)?;
    }
    Ok(())
}

/// # Safety
/// Both device spans stay live and all reads and writes are ordered through completion.
pub(crate) unsafe fn peer_raw(
    dst: u64,
    dst_context: &CudaContext,
    src: u64,
    src_context: &CudaContext,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    unsafe {
        sys::cuMemcpyPeerAsync(
            dst,
            dst_context.cu_ctx(),
            src,
            src_context.cu_ctx(),
            bytes,
            stream.cu_stream(),
        )
        .result()?;
        Ok(())
    }
    #[cfg(feature = "rocm")]
    anyhow::bail!(
        "HIP peer copy is not implemented: src_device={} dst_device={} stream_device={} src={src:#x} dst={dst:#x} bytes={bytes}",
        src_context.ordinal(),
        dst_context.ordinal(),
        stream.context().ordinal()
    )
}

/// # Safety
/// The destination contains `bytes` live bytes and prior readers are ordered on `stream`.
pub(crate) unsafe fn memset_bytes(
    dst: u64,
    value: u8,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemsetD8Async(dst, value, bytes, stream.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        stream.memset_bytes(dst, value, bytes)?;
    }
    Ok(())
}

/// # Safety
/// The aligned destination contains `count` live words and prior readers are ordered on `stream`.
pub(crate) unsafe fn memset_words(
    dst: u64,
    value: u32,
    count: usize,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemsetD32Async(dst, value, count, stream.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        stream.memset_words(dst, value, count)?;
    }
    Ok(())
}

pub fn bits(s: &Arc<CudaStream>, src: &impl DevicePtr<bf16>, dst: &mut Vec<u8>) -> Result<()> {
    for x in s.clone_dtoh(src)? {
        dst.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    Ok(())
}

pub struct Rect {
    pub rows: usize,
    pub width: usize,
    pub src_pitch: usize,
    pub src_col: usize,
    pub dst_pitch: usize,
    pub dst_col: usize,
}

pub fn columns(
    s: &Arc<CudaStream>,
    src: &impl DevicePtr<bf16>,
    dst: &mut impl DevicePtrMut<bf16>,
    r: Rect,
) -> Result<()> {
    ensure!(r.rows > 0, "invalid CUDA column copy rectangle");
    ensure!(
        (r.rows - 1) * r.src_pitch + r.src_col + r.width <= src.len()
            && (r.rows - 1) * r.dst_pitch + r.dst_col + r.width <= dst.len(),
        "CUDA column copy exceeds buffer"
    );
    s.context().bind_to_thread()?;
    let (src, _src) = src.device_ptr(s);
    columns_from(s, src, dst, r)
}

/// The same column copy from a raw device pointer; the caller owns the source
/// extent of `r.rows` rows of `r.src_pitch` BF16 values.
pub fn columns_from(
    s: &Arc<CudaStream>,
    src: u64,
    dst: &mut impl DevicePtrMut<bf16>,
    r: Rect,
) -> Result<()> {
    ensure!(
        r.rows > 0
            && r.width > 0
            && r.src_col + r.width <= r.src_pitch
            && r.dst_col + r.width <= r.dst_pitch,
        "invalid CUDA column copy rectangle"
    );
    ensure!(
        (r.rows - 1) * r.dst_pitch + r.dst_col + r.width <= dst.len(),
        "CUDA column copy exceeds buffer"
    );
    s.context().bind_to_thread()?;
    let (dst, _dst) = dst.device_ptr_mut(s);
    columns_raw(s, src, dst, r)
}

fn columns_raw(s: &Arc<CudaStream>, src: u64, dst: u64, r: Rect) -> Result<()> {
    ensure!(
        r.rows > 0
            && r.width > 0
            && r.src_col + r.width <= r.src_pitch
            && r.dst_col + r.width <= r.dst_pitch,
        "invalid CUDA column copy rectangle"
    );
    #[cfg(feature = "cuda")]
    let params = sys::CUDA_MEMCPY2D {
        srcXInBytes: r.src_col * 2,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src,
        srcArray: std::ptr::null_mut(),
        srcPitch: r.src_pitch * 2,
        dstXInBytes: r.dst_col * 2,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst,
        dstArray: std::ptr::null_mut(),
        dstPitch: r.dst_pitch * 2,
        WidthInBytes: r.width * 2,
        Height: r.rows,
    };
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemcpy2DAsync_v2(&params, s.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        s.copy_2d(
            dst + (r.dst_col * 2) as u64,
            src + (r.src_col * 2) as u64,
            (r.dst_pitch * 2, r.src_pitch * 2),
            (r.width * 2, r.rows),
        )?;
    }
    Ok(())
}

pub struct RingCopy<'a> {
    pub stream: &'a Arc<CudaStream>,
    pub source: &'a CudaSlice<bf16>,
    pub target: &'a mut CudaSlice<bf16>,
    pub source_capacity: usize,
    pub target_capacity: usize,
    pub heads: usize,
    pub head_dim: usize,
}

pub(crate) struct RingCopyCall {
    src: u64,
    dst: u64,
    source_start: usize,
    target_start: usize,
    rows: usize,
    wrap_source: bool,
    wrap_target: bool,
}

impl RingCopy<'_> {
    pub fn run(
        self,
        source_start: usize,
        target_start: usize,
        rows: usize,
        wrap_source: bool,
        wrap_target: bool,
    ) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        let lengths = [self.source.len(), self.target.len()];
        let (src, _src) = self.source.device_ptr(self.stream);
        let (dst, _dst) = self.target.device_ptr_mut(self.stream);
        RingCopyCall {
            src,
            dst,
            source_start,
            target_start,
            rows,
            wrap_source,
            wrap_target,
        }
        .run(
            self.stream,
            [self.source_capacity, self.target_capacity],
            lengths,
            [self.heads, self.head_dim],
        )
    }
    pub(crate) fn run_raw(self, call: RingCopyCall) -> Result<()> {
        let lengths = [self.source.len(), self.target.len()];
        call.run(
            self.stream,
            [self.source_capacity, self.target_capacity],
            lengths,
            [self.heads, self.head_dim],
        )
    }

    pub(crate) fn run_dynamic(
        self,
        function: &CudaFunction,
        position: &CudaSlice<i32>,
        rows: usize,
        to_ring: bool,
    ) -> Result<()> {
        ensure!(position.len() == 1, "dynamic ring position must be scalar");
        ensure!(rows > 0, "dynamic ring copy requires query rows");
        ensure!(
            self.heads > 0 && self.head_dim > 0,
            "dynamic ring copy requires nonzero head geometry"
        );
        for (capacity, len) in [
            (self.source_capacity, self.source.len()),
            (self.target_capacity, self.target.len()),
        ] {
            ensure!(
                capacity
                    .checked_mul(self.head_dim)
                    .and_then(|n| n.checked_mul(self.heads))
                    .is_some_and(|n| n <= len && n <= i32::MAX as usize),
                "dynamic ring copy geometry exceeds buffer"
            );
        }
        let ring = if to_ring {
            self.target_capacity
        } else {
            self.source_capacity
        };
        let (src, _src) = self.source.device_ptr(self.stream);
        let (dst, _dst) = self.target.device_ptr_mut(self.stream);
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&src)
                .arg(&dst)
                .arg(position)
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(self.source_capacity)?)
                .arg(&i32::try_from(self.target_capacity)?)
                .arg(&i32::try_from(self.heads)?)
                .arg(&i32::try_from(self.head_dim)?)
                .arg(&i32::from(to_ring))
                .launch(super::ops::flat(self.heads * ring * self.head_dim))?;
        }
        Ok(())
    }
}

impl RingCopyCall {
    fn run(
        self,
        stream: &Arc<CudaStream>,
        capacities: [usize; 2],
        lengths: [usize; 2],
        geometry: [usize; 2],
    ) -> Result<()> {
        let [source_capacity, target_capacity] = capacities;
        let [heads, head_dim] = geometry;
        let RingCopyCall {
            src,
            dst,
            source_start,
            target_start,
            rows,
            wrap_source,
            wrap_target,
        } = self;
        ensure!(
            rows > 0 && rows <= source_capacity && rows <= target_capacity,
            "ring row copy extent exceeds capacity"
        );
        ensure!(
            heads > 0 && head_dim > 0,
            "ring row copy requires nonzero head geometry"
        );
        for (capacity, len) in capacities.into_iter().zip(lengths) {
            ensure!(
                capacity
                    .checked_mul(head_dim)
                    .and_then(|n| n.checked_mul(heads))
                    .is_some_and(|n| n <= len && n <= usize::MAX / 2),
                "ring row copy geometry exceeds buffer"
            );
        }
        ensure!(
            source_start < source_capacity && target_start < target_capacity,
            "ring row copy start exceeds capacity"
        );
        ensure!(
            wrap_source || rows <= source_capacity - source_start,
            "ring row copy source range exceeds capacity"
        );
        ensure!(
            wrap_target || rows <= target_capacity - target_start,
            "ring row copy target range exceeds capacity"
        );
        let mut remaining = rows;
        let mut source_row = source_start;
        let mut target_row = target_start;
        while remaining > 0 {
            let count = remaining
                .min(source_capacity - source_row)
                .min(target_capacity - target_row);
            columns_raw(
                stream,
                src,
                dst,
                Rect {
                    rows: heads,
                    width: count * head_dim,
                    src_pitch: source_capacity * head_dim,
                    src_col: source_row * head_dim,
                    dst_pitch: target_capacity * head_dim,
                    dst_col: target_row * head_dim,
                },
            )?;
            remaining -= count;
            source_row = (source_row + count) % source_capacity;
            target_row = (target_row + count) % target_capacity;
        }
        Ok(())
    }
}

fn checked_byte_span<T>(
    elements: usize,
    byte_offset: usize,
    byte_count: usize,
) -> Result<(usize, usize)> {
    let size = std::mem::size_of::<T>();
    ensure!(byte_count > 0, "CUDA byte-span copy requires bytes");
    ensure!(
        byte_offset
            .checked_add(byte_count)
            .is_some_and(|end| end <= elements * size),
        "CUDA byte-span copy exceeds source"
    );
    Ok((byte_offset, byte_count))
}

pub fn dtoh_bytes<T: crate::backend::gpu::DeviceRepr>(
    s: &Arc<CudaStream>,
    src: &impl DevicePtr<T>,
    src_byte_offset: usize,
    dst: &mut PinnedHostSlice<u8>,
    dst_offset: usize,
    bytes: usize,
) -> Result<()> {
    let (src_offset, bytes) = checked_byte_span::<T>(src.len(), src_byte_offset, bytes)?;
    ensure!(
        dst_offset
            .checked_add(bytes)
            .is_some_and(|end| end <= dst.len()),
        "CUDA D2H byte span exceeds pinned destination"
    );
    s.context().bind_to_thread()?;
    let (src_ptr, _guard) = src.device_ptr(s);
    #[cfg(feature = "cuda")]
    let dst_ptr = dst.as_mut_ptr()?;
    #[cfg(feature = "rocm")]
    let (dst_slice, _write) = unsafe { dst.stream_synced_mut_slice(s) };
    #[cfg(feature = "rocm")]
    let dst_ptr = dst_slice.as_mut_ptr();
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemcpyDtoHAsync_v2(
            dst_ptr.add(dst_offset).cast(),
            src_ptr + src_offset as u64,
            bytes,
            s.cu_stream(),
        )
        .result()?;
        #[cfg(feature = "rocm")]
        s.copy_raw(
            dst_ptr.add(dst_offset) as u64,
            src_ptr + src_offset as u64,
            bytes,
            2,
        )?;
    }
    Ok(())
}

pub fn htod_bytes<T: crate::backend::gpu::DeviceRepr>(
    s: &Arc<CudaStream>,
    dst: &mut impl DevicePtrMut<T>,
    dst_byte_offset: usize,
    src: &PinnedHostSlice<u8>,
    src_offset: usize,
    bytes: usize,
) -> Result<()> {
    let (dst_offset, bytes) = checked_byte_span::<T>(dst.len(), dst_byte_offset, bytes)?;
    ensure!(
        src_offset
            .checked_add(bytes)
            .is_some_and(|end| end <= src.len()),
        "CUDA H2D byte span exceeds pinned source"
    );
    s.context().bind_to_thread()?;
    let (dst_ptr, _guard) = dst.device_ptr_mut(s);
    #[cfg(feature = "cuda")]
    let src_ptr = src.as_ptr()?;
    #[cfg(feature = "rocm")]
    let (src_slice, _read) = unsafe { src.stream_synced_slice(s) };
    #[cfg(feature = "rocm")]
    let src_ptr = src_slice.as_ptr();
    unsafe {
        #[cfg(feature = "cuda")]
        sys::cuMemcpyHtoDAsync_v2(
            dst_ptr + dst_offset as u64,
            src_ptr.add(src_offset).cast(),
            bytes,
            s.cu_stream(),
        )
        .result()?;
        #[cfg(feature = "rocm")]
        s.copy_raw(
            dst_ptr + dst_offset as u64,
            src_ptr.add(src_offset) as u64,
            bytes,
            1,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use anyhow::Context;

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU lease; ring-copy allocation event ordering"]
    fn ring_copy_cross_stream_order_gpu() -> Result<()> {
        use cudarc::driver::sys;
        let ctx = CudaContext::new(0)?;
        let write = ctx.new_stream()?;
        let read = ctx.new_stream()?;
        ensure!(
            ctx.is_event_tracking(),
            "allocation event tracking is disabled"
        );
        let head_dim = 128;
        let elements = 5 * head_dim;
        let host: Vec<_> = (0..elements)
            .map(|i| bf16::from_f32(((i * 13 + 7) % 113) as f32))
            .collect();
        let mut source = write.clone_htod(&host)?;
        let mut target = write.alloc_zeros::<bf16>(elements)?;
        let mut snapshot = read.alloc_zeros::<bf16>(elements)?;
        let target_ptr = target.device_ptr(&write).0;
        let snapshot_ptr = snapshot.device_ptr(&read).0;
        ctx.synchronize()?;
        write.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let scheduled = (|| -> Result<()> {
            unsafe {
                sys::cuMemsetD8Async(target_ptr, 0, elements * 2, write.cu_stream()).result()?
            };
            // All prior accesses completed before capture; allocation events start at this graph node.
            for buffer in [&mut source, &mut target, &mut snapshot] {
                unsafe { ctx.disable_event_tracking() };
                let (_, read_access) = buffer.device_ptr(&write);
                unsafe { ctx.enable_event_tracking() };
                drop(read_access);
                unsafe { ctx.disable_event_tracking() };
                let (_, write_access) = buffer.device_ptr_mut(&write);
                unsafe { ctx.enable_event_tracking() };
                drop(write_access);
            }
            ctx.check_err().context("ring allocation event priming")?;
            RingCopy {
                stream: &write,
                source: &source,
                target: &mut target,
                source_capacity: 5,
                target_capacity: 5,
                heads: 1,
                head_dim,
            }
            .run(1, 0, 3, false, false)
            .context("ring producer enqueue")?;
            ctx.check_err().context("ring producer allocation events")?;
            read.memcpy_dtod(&target, &mut snapshot)
                .context("ring consumer enqueue")?;
            ctx.check_err().context("ring consumer allocation events")?;
            let joined = read.record_event(None).context("ring join event")?;
            write.wait(&joined).context("ring join wait")?;
            Ok(())
        })();
        let captured = write.end_capture(sys::CUgraphInstantiate_flags(0));
        scheduled?;
        let graph = captured?.context("ring ordering capture is empty")?;
        let mut count = 0;
        unsafe {
            sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count).result()?
        };
        let mut nodes = vec![std::ptr::null_mut(); count];
        unsafe { sys::cuGraphGetNodes(graph.cu_graph(), nodes.as_mut_ptr(), &mut count).result()? };
        let mut copies = Vec::new();
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe { sys::cuGraphNodeGetType(node, &mut kind).result()? };
            if kind == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY {
                let mut params = std::mem::MaybeUninit::uninit();
                unsafe {
                    sys::cuGraphMemcpyNodeGetParams(node, params.as_mut_ptr()).result()?;
                    copies.push((node, params.assume_init().dstDevice));
                }
            }
        }
        let producer = copies
            .iter()
            .find(|(_, dst)| *dst == target_ptr)
            .context("ring producer copy absent")?
            .0;
        let consumer = copies
            .iter()
            .find(|(_, dst)| *dst == snapshot_ptr)
            .context("ring cross-stream consumer is outside the captured dependency")?
            .0;
        let mut count = 0;
        unsafe {
            sys::cuGraphGetEdges_v2(
                graph.cu_graph(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            )
            .result()?;
        }
        let mut from = vec![std::ptr::null_mut(); count];
        let mut to = vec![std::ptr::null_mut(); count];
        unsafe {
            sys::cuGraphGetEdges_v2(
                graph.cu_graph(),
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut count,
            )
            .result()?
        };
        let mut pending = vec![producer];
        let mut visited = Vec::new();
        while let Some(node) = pending.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.push(node);
            pending.extend(
                from.iter()
                    .zip(&to)
                    .filter_map(|(&a, &b)| (a == node).then_some(b)),
            );
        }
        ensure!(
            visited.contains(&consumer),
            "ring read has no dependency on the copy write"
        );
        graph.launch()?;
        write.synchronize()?;
        let actual = read.clone_dtoh(&snapshot)?;
        ensure!(
            actual[..3 * head_dim] == host[head_dim..4 * head_dim]
                && actual[3 * head_dim..].iter().all(|x| *x == bf16::ZERO),
            "ring cross-stream copy differs"
        );
        println!(
            "PASS ring_copy_cross_stream_order captured_dependency=true copied_elements=384 untouched_elements=256 event_tracking=true"
        );
        Ok(())
    }
    #[test]
    #[ignore = "GPU window required; head-major KV ring staging"]
    fn ring_copy_geometry_gpu() -> Result<()> {
        let device = crate::backend::cuda::Device::new(0)?;
        device.finish_upload()?;
        let s = &device.stream;
        let source_capacity = 513;
        let target_capacity = 1025;
        let rows = 511;
        let sentinel = bf16::from_bits(0x7fc1);
        for (heads, head_dim) in [(4, 128), (1, 256), (2, 256), (8, 256), (1, 512), (8, 512)] {
            let host: Vec<_> = (0..heads * source_capacity * head_dim)
                .map(|i| bf16::from_f32(((i * 37 + 11) % 1009) as f32 / 256.0 - 2.0))
                .collect();
            let source = s.clone_htod(&host)?;
            let empty = vec![sentinel; heads * target_capacity * head_dim + 32];
            for (source_start, target_start, wrap_source, wrap_target) in [
                (2, 17, false, false),
                (503, 0, true, false),
                (2, 1019, false, true),
                (503, 1019, true, true),
            ] {
                let mut expected = empty.clone();
                for head in 0..heads {
                    for row in 0..rows {
                        let src = (head * source_capacity + (source_start + row) % source_capacity)
                            * head_dim;
                        let dst = (head * target_capacity + (target_start + row) % target_capacity)
                            * head_dim;
                        expected[dst..dst + head_dim].copy_from_slice(&host[src..src + head_dim]);
                    }
                }
                for graph in [false, true] {
                    let mut target = s.clone_htod(&empty)?;
                    if graph {
                        s.synchronize()?;
                        #[cfg(feature = "cuda")]
                    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                        #[cfg(feature = "rocm")]
                        let capture = s.begin_capture()?;
                        let result = RingCopy {
                            stream: s,
                            source: &source,
                            target: &mut target,
                            source_capacity,
                            target_capacity,
                            heads,
                            head_dim,
                        }
                        .run(
                            source_start,
                            target_start,
                            rows,
                            wrap_source,
                            wrap_target,
                        );
                        #[cfg(feature = "cuda")]
                        let captured = s
                            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))
                            .map_err(anyhow::Error::from)
                            .and_then(|g| g.context("ring staging graph is empty"));
                        #[cfg(feature = "rocm")]
                        let captured = capture.finish();
                        result?;
                        let captured = captured?;
                        captured.launch()?;
                        captured.launch()?;
                    } else {
                        RingCopy {
                            stream: s,
                            source: &source,
                            target: &mut target,
                            source_capacity,
                            target_capacity,
                            heads,
                            head_dim,
                        }
                        .run(
                            source_start,
                            target_start,
                            rows,
                            wrap_source,
                            wrap_target,
                        )?;
                    }
                    let actual = s.clone_dtoh(&target)?;
                    ensure!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "ring staging differs heads={heads} dim={head_dim} source_start={source_start} target_start={target_start} graph={graph}"
                    );
                }
            }
            let mut wrong = s.clone_htod(&empty)?;
            RingCopy {
                stream: s,
                source: &source,
                target: &mut wrong,
                source_capacity,
                target_capacity,
                heads,
                head_dim,
            }
            .run(504, 0, rows, true, false)?;
            let wrong = s.clone_dtoh(&wrong)?;
            ensure!(
                wrong[..head_dim]
                    .iter()
                    .zip(&host[503 * head_dim..504 * head_dim])
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "shifted ring source control did not discriminate"
            );
            eprintln!(
                "PASS ring_copy_geometry heads={heads} dim={head_dim} rows={rows} source_capacity={source_capacity} target_capacity={target_capacity} graph_eager_exact=true shifted_source_rejected=true"
            );
        }
        Ok(())
    }
}
