//! Shared helpers for the MoE GPU-gate tests.
use crate::backend::cuda::contracts::{Tensor, TensorMut};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub(crate) fn upload<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    values: Vec<T>,
) -> Result<CudaSlice<T>> {
    Ok(stream.clone_htod(&values)?)
}

pub(crate) fn zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    Ok(stream.alloc_zeros(len)?)
}

pub(crate) fn view<T: DeviceRepr + Copy, const R: usize>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    shape: [u32; R],
    stride: [u64; R],
) -> Result<Tensor<T, R>> {
    let (p, _guard) = b.device_ptr(s);
    unsafe {
        Tensor::new(
            p,
            shape,
            stride,
            (b.len() * std::mem::size_of::<T>()) as u64,
        )
    }
}

pub(crate) fn mut_view<T: DeviceRepr + Copy, const R: usize>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    shape: [u32; R],
    stride: [u64; R],
) -> Result<TensorMut<T, R>> {
    let (p, _guard) = b.device_ptr(s);
    unsafe {
        TensorMut::new(
            p,
            shape,
            stride,
            (b.len() * std::mem::size_of::<T>()) as u64,
        )
    }
}

/// Dense row-major `[rows, cols]` view.
pub(crate) fn dense<T: DeviceRepr + Copy>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    rows: u32,
    cols: u32,
) -> Result<Tensor<T, 2>> {
    view(b, s, [rows, cols], [u64::from(cols), 1])
}

pub(crate) fn dense_mut<T: DeviceRepr + Copy>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    rows: u32,
    cols: u32,
) -> Result<TensorMut<T, 2>> {
    mut_view(b, s, [rows, cols], [u64::from(cols), 1])
}

/// Device slot table: `[experts]` device addresses of each expert's weight
/// blocks, the per-launch valid mask (1 = resident and ready), and the slot
/// generation array checked against the launch's expected generation.
pub(crate) struct SlotTable {
    pub gate_up: CudaSlice<u64>,
    pub down: CudaSlice<u64>,
    pub valid: CudaSlice<u32>,
    pub generation: CudaSlice<u64>,
}

pub(crate) fn slots(
    stream: &Arc<CudaStream>,
    experts: usize,
    gate_up_base: u64,
    gate_up_stride: u64,
    down_base: u64,
    down_stride: u64,
) -> Result<SlotTable> {
    let gu = (0..experts)
        .map(|e| gate_up_base + e as u64 * gate_up_stride)
        .collect::<Vec<_>>();
    let dn = (0..experts)
        .map(|e| down_base + e as u64 * down_stride)
        .collect::<Vec<_>>();
    let valid = vec![1u32; experts];
    let generation = vec![1u64; experts];
    Ok(SlotTable {
        gate_up: upload(stream, gu)?,
        down: upload(stream, dn)?,
        valid: upload(stream, valid)?,
        generation: upload(stream, generation)?,
    })
}
