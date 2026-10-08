use anyhow::{Result, ensure};
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, sys};
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
