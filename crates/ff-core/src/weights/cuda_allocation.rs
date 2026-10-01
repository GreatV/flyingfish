//! Explicit experimental isolation of retained weights from activation pools.
use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, result,
};
use candle_core::{CudaDevice, DType, Storage, Tensor, op::BackpropOp};
use std::sync::Arc;

use super::device_cache::CudaWeightAllocator;

/// Allocate `count` uninitialized elements under the cache's allocator policy.
/// `Direct` allocations sit outside the stream pool that ordinary candle
/// tensors draw from, so only a `Direct` cache may ask for them.
unsafe fn allocate<T: DeviceRepr>(
    allocator: CudaWeightAllocator,
    count: usize,
    bytes: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<T>> {
    match allocator {
        CudaWeightAllocator::Direct => {
            Ok(unsafe { stream.upgrade_device_ptr::<T>(result::malloc_sync(bytes)?, count) })
        }
        CudaWeightAllocator::StreamPool => Ok(unsafe { stream.alloc::<T>(count) }?),
    }
}

pub(super) fn supports(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::BF16
            | DType::F16
            | DType::F32
            | DType::F64
            | DType::U8
            | DType::U32
            | DType::I16
            | DType::I64
    )
}

fn upload_slice<T: DeviceRepr>(
    data: &[u8],
    count: usize,
    stream: &Arc<CudaStream>,
    allocator: CudaWeightAllocator,
) -> Result<CudaSlice<T>> {
    ensure!(
        count.checked_mul(std::mem::size_of::<T>()) == Some(data.len()),
        "weight upload byte count mismatch"
    );
    stream.synchronize()?;
    stream.context().bind_to_thread()?;
    let mut output = unsafe { allocate::<T>(allocator, count, data.len(), stream) }?;
    let copied = {
        let (pointer, _written) = output.device_ptr_mut(stream);
        unsafe { result::memcpy_htod_async(pointer, data, stream.cu_stream()) }
    };
    let completed = stream.synchronize();
    copied.context("direct weight upload failed")?;
    completed.context("direct weight upload completion failed")?;
    Ok(output)
}

/// Fill pinned memory with parallel positional reads.
/// Host-side only; the caller issues the H2D copy on its own stream.
/// Returns `None` on allocation or read failure so the caller can use the mmap.
#[cfg(unix)]
pub fn fill_pinned_from_checkpoint(
    file: &std::fs::File,
    offset: usize,
    len: usize,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u8>> {
    let mut host = unsafe { stream.context().alloc_pinned::<u8>(len) }
        .map_err(|e| anyhow::anyhow!("pinned allocation of {len} bytes failed: {e}"))?;
    let readers = warm_threads()?.min(len.div_ceil(2 << 20).max(1));
    crate::storage::read_parallel_into(
        crate::storage::ParallelReadSource::Shared(file),
        offset as u64,
        host.as_mut_slice()
            .context("pinned buffer is not addressable")?,
        readers,
    )
    .context("parallel checkpoint read into the pinned buffer failed")?;
    Ok(host)
}

#[cfg(unix)]
fn warm_threads() -> Result<usize> {
    crate::probe::env_usize("FF_WEIGHT_WARM_THREADS", 8, 1, usize::MAX)
}

/// Upload file-backed bytes through a pinned buffer filled by parallel readers.
#[cfg(unix)]
fn upload_slice_from_source<T: DeviceRepr>(
    file: &std::fs::File,
    offset: usize,
    len: usize,
    count: usize,
    stream: &Arc<CudaStream>,
    allocator: CudaWeightAllocator,
) -> Result<Option<CudaSlice<T>>> {
    ensure!(
        count.checked_mul(std::mem::size_of::<T>()) == Some(len),
        "weight upload byte count mismatch"
    );
    let host = fill_pinned_from_checkpoint(file, offset, len, stream)?;
    let host_slice = host
        .as_slice()
        .context("direct weight upload: pinned buffer is not addressable")?;
    stream.synchronize()?;
    stream.context().bind_to_thread()?;
    let mut output = unsafe { allocate::<T>(allocator, count, len, stream) }?;
    let copied = {
        let (pointer, _written) = output.device_ptr_mut(stream);
        unsafe { result::memcpy_htod_async(pointer, host_slice, stream.cu_stream()) }
    };
    let completed = stream.synchronize();
    copied.context("direct weight upload failed")?;
    completed.context("direct weight upload completion failed")?;
    Ok(Some(output))
}

pub(super) fn upload(
    data: &[u8],
    source: Option<(&std::fs::File, usize, usize)>,
    shape: &[usize],
    dtype: DType,
    device: &CudaDevice,
    allocator: CudaWeightAllocator,
) -> Result<Tensor> {
    #[cfg(not(unix))]
    let _ = source;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &dimension| n.checked_mul(dimension))
        .context("weight shape overflow")?;
    ensure!(count > 0, "empty direct weight");
    let stream = device.cuda_stream();
    macro_rules! upload {
        ($ty:ty) => {{
            #[cfg(unix)]
            if let Some((file, offset, len)) = source
                && let Some(slice) =
                    upload_slice_from_source::<$ty>(file, offset, len, count, &stream, allocator)?
            {
                return Ok(Tensor::from_storage(
                    Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                        slice,
                        device.clone(),
                    )),
                    shape.to_vec(),
                    BackpropOp::none(),
                    false,
                ));
            }
            candle_core::CudaStorage::wrap_cuda_slice(
                upload_slice::<$ty>(data, count, &stream, allocator)?,
                device.clone(),
            )
        }};
    }
    let storage = match dtype {
        DType::BF16 => upload!(half::bf16),
        DType::F16 => upload!(half::f16),
        DType::F32 => upload!(f32),
        DType::F64 => upload!(f64),
        DType::U8 => upload!(u8),
        DType::U32 => upload!(u32),
        DType::I16 => upload!(i16),
        DType::I64 => upload!(i64),
        _ => anyhow::bail!("unsupported direct weight dtype {dtype:?}"),
    };
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        shape.to_vec(),
        BackpropOp::none(),
        false,
    ))
}
