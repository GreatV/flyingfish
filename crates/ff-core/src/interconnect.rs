//! Explicit tensor transport between CUDA devices, without a host tensor copy.
//! The initial transport synchronizes at each cut; asynchronous pipelining can
//! replace these barriers without changing the tensor contract.
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

pub fn copy_tensor(input: &Tensor, destination: &Device) -> Result<Tensor> {
    if input.device().same_device(destination) {
        return Ok(input.clone());
    }
    if !input.device().is_cuda() || !destination.is_cuda() {
        return Ok(input.to_device(destination)?);
    }
    #[cfg(not(feature = "cuda"))]
    bail!("CUDA tensor transport requires the cuda feature");
    #[cfg(feature = "cuda")]
    {
        use candle_core::{Storage, cuda_backend::CudaStorageSlice, op::BackpropOp};
        let input = input.contiguous()?;
        let target = destination.as_cuda_device()?.clone();
        let stream = target.cuda_stream();
        let (storage, layout) = input.storage_and_layout();
        let Storage::Cuda(source) = &*storage else {
            bail!("expected CUDA tensor storage")
        };
        source
            .device
            .cuda_stream()
            .synchronize()
            .context("source stream synchronization failed")?;
        let count = layout.shape().elem_count();
        let offset = layout.start_offset();
        macro_rules! copy {
            ($source:expr) => {{
                let copied = copy_slice($source, offset, count, &stream)?;
                candle_core::CudaStorage::wrap_cuda_slice(copied, target)
            }};
        }
        let copied = match &source.slice {
            CudaStorageSlice::BF16(x) => copy!(x),
            CudaStorageSlice::F16(x) => copy!(x),
            CudaStorageSlice::F32(x) => copy!(x),
            CudaStorageSlice::F64(x) => copy!(x),
            CudaStorageSlice::F8E4M3(x) => copy!(x),
            CudaStorageSlice::U8(x) => copy!(x),
            CudaStorageSlice::U32(x) => copy!(x),
            CudaStorageSlice::I64(x) => copy!(x),
            _ => bail!("unsupported CUDA peer tensor dtype {:?}", input.dtype()),
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(copied),
            input.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
fn copy_slice<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    source: &candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    offset: usize,
    count: usize,
    stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaSlice<T>> {
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, DeviceSlice, result};
    let view = source.slice(offset..offset + count);
    let mut output =
        unsafe { stream.alloc::<T>(count) }.context("peer destination allocation failed")?;
    {
        let source_stream = source.stream();
        let (src, _read) = view.device_ptr(source_stream);
        source_stream
            .synchronize()
            .context("source allocation synchronization failed")?;
        let (dst, _write) = output.device_ptr_mut(stream);
        stream.context().bind_to_thread()?;
        unsafe {
            if source_stream.context() == stream.context() {
                result::memcpy_dtod_async(dst, src, view.num_bytes(), stream.cu_stream())
            } else {
                result::memcpy_peer_async(
                    stream.context().cu_ctx(),
                    dst,
                    source_stream.context().cu_ctx(),
                    src,
                    view.num_bytes(),
                    stream.cu_stream(),
                )
            }
        }
        .context("CUDA device-to-device copy failed")?;
        stream
            .synchronize()
            .context("destination copy synchronization failed")?;
    }
    Ok(output)
}
