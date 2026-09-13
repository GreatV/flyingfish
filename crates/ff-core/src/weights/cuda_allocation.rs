//! Explicit experimental isolation of retained weights from activation pools.
use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, result,
};
use candle_core::{CudaDevice, DType, Storage, Tensor, op::BackpropOp};
use std::sync::Arc;

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
) -> Result<CudaSlice<T>> {
    ensure!(
        count.checked_mul(std::mem::size_of::<T>()) == Some(data.len()),
        "weight upload byte count mismatch"
    );
    stream.synchronize()?;
    stream.context().bind_to_thread()?;
    let mut output =
        unsafe { stream.upgrade_device_ptr::<T>(result::malloc_sync(data.len())?, count) };
    let copied = {
        let (pointer, _written) = output.device_ptr_mut(stream);
        unsafe { result::memcpy_htod_async(pointer, data, stream.cu_stream()) }
    };
    let completed = stream.synchronize();
    copied.context("direct weight upload failed")?;
    completed.context("direct weight upload completion failed")?;
    Ok(output)
}

pub(super) fn upload(
    data: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &CudaDevice,
) -> Result<Tensor> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &dimension| n.checked_mul(dimension))
        .context("weight shape overflow")?;
    ensure!(count > 0, "empty direct weight");
    let stream = device.cuda_stream();
    macro_rules! upload {
        ($ty:ty) => {
            candle_core::CudaStorage::wrap_cuda_slice(
                upload_slice::<$ty>(data, count, &stream)?,
                device.clone(),
            )
        };
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
