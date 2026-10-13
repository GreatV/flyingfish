use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, PinnedHostSlice, sys};
use half::bf16;
use std::sync::Arc;

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
    ensure!(
        r.rows > 0
            && r.width > 0
            && r.src_col + r.width <= r.src_pitch
            && r.dst_col + r.width <= r.dst_pitch,
        "invalid CUDA column copy rectangle"
    );
    ensure!(
        (r.rows - 1) * r.src_pitch + r.src_col + r.width <= src.len()
            && (r.rows - 1) * r.dst_pitch + r.dst_col + r.width <= dst.len(),
        "CUDA column copy exceeds buffer"
    );
    s.context().bind_to_thread()?;
    let (src, _src) = src.device_ptr(s);
    let (dst, _dst) = dst.device_ptr_mut(s);
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
        sys::cuMemcpy2DAsync_v2(&params, s.cu_stream()).result()?;
    }
    Ok(())
}

fn checked_byte_span<T>(
    elements: usize,
    element_offset: usize,
    byte_count: usize,
) -> Result<(usize, usize)> {
    let size = std::mem::size_of::<T>();
    let offset = element_offset
        .checked_mul(size)
        .context("CUDA byte-span offset overflow")?;
    ensure!(byte_count > 0, "CUDA byte-span copy requires bytes");
    ensure!(
        offset
            .checked_add(byte_count)
            .is_some_and(|end| end <= elements * size),
        "CUDA byte-span copy exceeds source"
    );
    Ok((offset, byte_count))
}

pub fn dtoh_bytes<T: cudarc::driver::DeviceRepr>(
    s: &Arc<CudaStream>,
    src: &impl DevicePtr<T>,
    src_element_offset: usize,
    dst: &mut PinnedHostSlice<u8>,
    dst_offset: usize,
    bytes: usize,
) -> Result<()> {
    let (src_offset, bytes) = checked_byte_span::<T>(src.len(), src_element_offset, bytes)?;
    ensure!(
        dst_offset
            .checked_add(bytes)
            .is_some_and(|end| end <= dst.len()),
        "CUDA D2H byte span exceeds pinned destination"
    );
    s.context().bind_to_thread()?;
    let (src_ptr, _guard) = src.device_ptr(s);
    let dst_ptr = dst.as_mut_ptr().map_err(anyhow::Error::from)?;
    unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            dst_ptr.add(dst_offset).cast(),
            src_ptr + src_offset as u64,
            bytes,
            s.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

pub fn htod_bytes<T: cudarc::driver::DeviceRepr>(
    s: &Arc<CudaStream>,
    dst: &mut impl DevicePtrMut<T>,
    dst_element_offset: usize,
    src: &PinnedHostSlice<u8>,
    src_offset: usize,
    bytes: usize,
) -> Result<()> {
    let (dst_offset, bytes) = checked_byte_span::<T>(dst.len(), dst_element_offset, bytes)?;
    ensure!(
        src_offset
            .checked_add(bytes)
            .is_some_and(|end| end <= src.len()),
        "CUDA H2D byte span exceeds pinned source"
    );
    s.context().bind_to_thread()?;
    let (dst_ptr, _guard) = dst.device_ptr_mut(s);
    let src_ptr = src.as_ptr().map_err(anyhow::Error::from)?;
    unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst_ptr + dst_offset as u64,
            src_ptr.add(src_offset).cast(),
            bytes,
            s.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}
