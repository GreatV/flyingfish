#[cfg(feature = "cuda")]
use crate::backend::gpu::CudaSlice;
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

#[cfg(feature = "cuda")]
pub(crate) struct RingCopy<'a> {
    pub(crate) stream: &'a Arc<CudaStream>,
    pub(crate) source: &'a CudaSlice<bf16>,
    pub(crate) target: &'a mut CudaSlice<bf16>,
    pub(crate) source_capacity: usize,
    pub(crate) target_capacity: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
}

#[cfg(feature = "cuda")]
pub(crate) struct RingCopyCall {
    pub(crate) src: u64,
    pub(crate) dst: u64,
    pub(crate) source_start: usize,
    pub(crate) target_start: usize,
    pub(crate) rows: usize,
    pub(crate) wrap_source: bool,
    pub(crate) wrap_target: bool,
}

#[cfg(feature = "cuda")]
impl RingCopy<'_> {
    pub(crate) fn run(
        self,
        source_start: usize,
        target_start: usize,
        rows: usize,
        wrap_source: bool,
        wrap_target: bool,
    ) -> Result<()> {
        let (src, dst) = {
            let (src, _src) = self.source.device_ptr(self.stream);
            let (dst, _dst) = self.target.device_ptr_mut(self.stream);
            (src, dst)
        };
        self.run_raw(RingCopyCall {
            src,
            dst,
            source_start,
            target_start,
            rows,
            wrap_source,
            wrap_target,
        })
    }

    pub(crate) fn run_raw(self, call: RingCopyCall) -> Result<()> {
        let RingCopyCall {
            src,
            dst,
            source_start,
            target_start,
            rows,
            wrap_source,
            wrap_target,
        } = call;
        ensure!(
            rows > 0 && rows <= self.source_capacity && rows <= self.target_capacity,
            "ring row copy extent exceeds capacity"
        );
        ensure!(
            self.heads > 0 && self.head_dim > 0,
            "ring row copy requires nonzero head geometry"
        );
        for (capacity, len) in [
            (self.source_capacity, self.source.len()),
            (self.target_capacity, self.target.len()),
        ] {
            ensure!(
                capacity
                    .checked_mul(self.head_dim)
                    .and_then(|n| n.checked_mul(self.heads))
                    .is_some_and(|n| n <= len && n <= usize::MAX / 2),
                "ring row copy geometry exceeds buffer"
            );
        }
        ensure!(
            source_start < self.source_capacity && target_start < self.target_capacity,
            "ring row copy start exceeds capacity"
        );
        ensure!(
            wrap_source || rows <= self.source_capacity - source_start,
            "ring row copy source range exceeds capacity"
        );
        ensure!(
            wrap_target || rows <= self.target_capacity - target_start,
            "ring row copy target range exceeds capacity"
        );
        let mut remaining = rows;
        let mut source_row = source_start;
        let mut target_row = target_start;
        while remaining > 0 {
            let count = remaining
                .min(self.source_capacity - source_row)
                .min(self.target_capacity - target_row);
            columns_raw(
                self.stream,
                src,
                dst,
                Rect {
                    rows: self.heads,
                    width: count * self.head_dim,
                    src_pitch: self.source_capacity * self.head_dim,
                    src_col: source_row * self.head_dim,
                    dst_pitch: self.target_capacity * self.head_dim,
                    dst_col: target_row * self.head_dim,
                },
            )?;
            remaining -= count;
            source_row = (source_row + count) % self.source_capacity;
            target_row = (target_row + count) % self.target_capacity;
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
