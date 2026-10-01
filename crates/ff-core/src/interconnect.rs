//! Explicit tensor transport between CUDA devices, without a host tensor copy.
//! The initial transport synchronizes at each cut; asynchronous pipelining can
//! replace these barriers without changing the tensor contract.
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::{Result, bail};
use candle_core::Tensor;

#[cfg(feature = "cuda")]
struct Region {
    ptr: candle_core::cuda_backend::cudarc::driver::sys::CUdeviceptr,
    bytes: usize,
    stream: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
fn regions(input: &Tensor, destination: &Tensor) -> Result<(Region, Region)> {
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceSlice};
    use candle_core::{Storage, cuda_backend::CudaStorageSlice};
    anyhow::ensure!(
        input.dtype() == destination.dtype() && input.dims() == destination.dims(),
        "boundary copy needs matching tensors: input {:?} {:?}, destination {:?} {:?}",
        input.dtype(),
        input.dims(),
        destination.dtype(),
        destination.dims()
    );
    anyhow::ensure!(
        input.is_contiguous() && destination.is_contiguous(),
        "boundary copy needs contiguous tensors"
    );
    let (input_storage, input_layout) = input.storage_and_layout();
    let (output_storage, output_layout) = destination.storage_and_layout();
    let (Storage::Cuda(source), Storage::Cuda(target)) = (&*input_storage, &*output_storage) else {
        bail!("boundary copy needs CUDA tensors")
    };
    let count = input_layout.shape().elem_count();
    let (from, to) = (input_layout.start_offset(), output_layout.start_offset());
    macro_rules! region {
        ($slice:expr, $offset:expr) => {{
            let slice = $slice;
            let view = slice.slice($offset..$offset + count);
            let (ptr, _guard) = view.device_ptr(slice.stream());
            Region {
                ptr,
                bytes: view.num_bytes(),
                stream: slice.stream().clone(),
            }
        }};
    }
    macro_rules! pair {
        ($($variant:ident),*) => {
            match (&source.slice, &target.slice) {
                $((CudaStorageSlice::$variant(a), CudaStorageSlice::$variant(b)) => {
                    (region!(a, from), region!(b, to))
                })*
                _ => bail!("unsupported CUDA boundary tensor dtype {:?}", input.dtype()),
            }
        };
    }
    Ok(pair!(BF16, F16, F32, F64, F8E4M3, U8, U32, I64))
}

/// Copy `input` into `destination` across CUDA devices and synchronize both streams.
pub fn copy_tensor_into(input: &Tensor, destination: &Tensor) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (input, destination);
        bail!("CUDA tensor transport requires the cuda feature");
    }
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::result;
        let (source, target) = regions(input, destination)?;
        source
            .stream
            .synchronize()
            .context("source stream synchronization failed")?;
        target.stream.context().bind_to_thread()?;
        // Peer access is not enabled here; setup validates the same context state.
        unsafe {
            if source.stream.context() == target.stream.context() {
                result::memcpy_dtod_async(
                    target.ptr,
                    source.ptr,
                    source.bytes,
                    target.stream.cu_stream(),
                )
            } else {
                result::memcpy_peer_async(
                    target.stream.context().cu_ctx(),
                    target.ptr,
                    source.stream.context().cu_ctx(),
                    source.ptr,
                    source.bytes,
                    target.stream.cu_stream(),
                )
            }
        }
        .context("CUDA device-to-device copy failed")?;
        target
            .stream
            .synchronize()
            .context("destination copy synchronization failed")?;
        Ok(())
    }
}

/// Copy `input` into `destination` through `host` and synchronize both streams.
pub fn stage_tensor_into(input: &Tensor, destination: &Tensor, host: &mut [u8]) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (input, destination, host);
        bail!("CUDA tensor transport requires the cuda feature");
    }
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::result;
        let (source, target) = regions(input, destination)?;
        anyhow::ensure!(
            host.len() >= source.bytes,
            "host staging buffer holds {} bytes, the boundary needs {}",
            host.len(),
            source.bytes
        );
        let host = &mut host[..source.bytes];
        source.stream.context().bind_to_thread()?;
        unsafe { result::memcpy_dtoh_async(host, source.ptr, source.stream.cu_stream()) }
            .context("CUDA device-to-host copy failed")?;
        source
            .stream
            .synchronize()
            .context("source stream synchronization failed")?;
        target.stream.context().bind_to_thread()?;
        unsafe { result::memcpy_htod_async(target.ptr, host, target.stream.cu_stream()) }
            .context("CUDA host-to-device copy failed")?;
        target
            .stream
            .synchronize()
            .context("destination copy synchronization failed")?;
        Ok(())
    }
}
