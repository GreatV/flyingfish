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
        ensure!(rows > 0, "ring row copy requires rows");
        ensure!(
            source_start < self.source_capacity && target_start < self.target_capacity,
            "ring row copy start exceeds capacity"
        );
        if !wrap_source {
            ensure!(
                source_start + rows <= self.source_capacity,
                "ring row copy source range exceeds capacity"
            );
        }
        if !wrap_target {
            ensure!(
                target_start + rows <= self.target_capacity,
                "ring row copy target range exceeds capacity"
            );
        }
        let source_first = if wrap_source {
            rows.min(self.source_capacity - source_start)
        } else {
            rows
        };
        let target_first = if wrap_target {
            rows.min(self.target_capacity - target_start)
        } else {
            rows
        };
        let first = rows.min(source_first.min(target_first));
        for (offset, count) in [(0, first), (first, rows - first)] {
            if count == 0 {
                continue;
            }
            let source_row = if offset > 0 && wrap_source {
                0
            } else {
                source_start + offset
            };
            let target_row = if offset > 0 && wrap_target {
                0
            } else {
                target_start + offset
            };
            columns(
                self.stream,
                self.source,
                self.target,
                Rect {
                    rows: 4,
                    width: count * 128,
                    src_pitch: self.source_capacity * 128,
                    src_col: source_row * 128,
                    dst_pitch: self.target_capacity * 128,
                    dst_col: target_row * 128,
                },
            )?;
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
    let dst_ptr = dst.as_mut_ptr()?;
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
    let src_ptr = src.as_ptr()?;
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
