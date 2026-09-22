//! Stage-ahead weight prefetch on a second CUDA stream.
//!
//! The next execution stage's weight uploads overlap the current stage's
//! compute; `FF_H3_PREFETCH=0` restores synchronous loading. The prefetched
//! bytes are identical to what `ModelWeights::load` would upload; only their
//! transfer timing changes, and the compute stream waits on a recorded event
//! before reading them.

use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::{
    CudaEvent, CudaStream, DevicePtrMut, PinnedHostSlice, result, sys,
};

use candle_core::{CudaDevice, DType, Storage, Tensor, op::BackpropOp};
use ff_core::weights::ModelWeights;
use std::collections::BTreeMap;
use std::sync::Arc;

pub(crate) const PREFETCH_ENVIRONMENT_VARIABLE: &str = "FF_H3_PREFETCH";

pub(crate) fn requested() -> bool {
    std::env::var(PREFETCH_ENVIRONMENT_VARIABLE)
        .map(|value| value != "0")
        .unwrap_or(true)
}

pub(crate) struct StagePrefetcher {
    stream: Arc<CudaStream>,
    device: CudaDevice,
    slot: Option<Slot>,
    slab: Option<PinnedHostSlice<u8>>,
    slab_bytes: usize,
}

struct Slot {
    tensors: BTreeMap<String, Tensor>,
    ready: Option<CudaEvent>,
}

impl StagePrefetcher {
    pub(crate) fn new(device: &CudaDevice) -> Result<Self> {
        let compute = device.cuda_stream();
        ensure!(
            compute.context().is_event_tracking(),
            "H3 stage prefetch requires CUDA event tracking"
        );
        Ok(Self {
            stream: compute.context().new_stream()?,
            device: device.clone(),
            slot: None,
            slab: None,
            slab_bytes: 0,
        })
    }

    pub(crate) fn prefetch(&mut self, weights: &ModelWeights, names: &[&str]) -> Result<()> {
        let mut tensors = BTreeMap::new();
        self.stream
            .context()
            .bind_to_thread()
            .context("bind the prefetch stream")?;
        self.stream
            .synchronize()
            .context("drain the prefetch stream before slab reuse")?;
        let total: usize = names
            .iter()
            .map(|&name| {
                weights
                    .metadata(name)
                    .map(|metadata| metadata.bytes)
                    .unwrap_or(0)
            })
            .sum();
        self.ensure_slab(total)?;
        let mut offset = 0usize;
        for &name in names {
            if let Some((tensor, consumed)) = upload_async(
                weights,
                name,
                &self.stream,
                &self.device,
                &mut self.slab,
                offset,
            )? {
                offset += consumed;
                tensors.insert(name.to_owned(), tensor);
            }
        }
        let ready = (!tensors.is_empty())
            .then(|| {
                self.stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .context("record the prefetch event")
            })
            .transpose()?;
        self.slot = Some(Slot { tensors, ready });
        Ok(())
    }

    fn ensure_slab(&mut self, bytes: usize) -> Result<()> {
        if bytes > 0 && self.slab_bytes < bytes {
            self.slab = None;
            self.slab = Some(
                unsafe { self.stream.context().alloc_pinned::<u8>(bytes) }
                    .with_context(|| format!("pin the {bytes}-byte prefetch slab"))?,
            );
            self.slab_bytes = bytes;
        }
        Ok(())
    }

    pub(crate) fn take(&mut self, names: &[&str]) -> Result<BTreeMap<String, Tensor>> {
        let mut taken = BTreeMap::new();
        let Some(slot) = self.slot.as_mut() else {
            return Ok(taken);
        };
        if let Some(ready) = slot.ready.as_ref() {
            self.device
                .cuda_stream()
                .wait(ready)
                .context("wait for prefetched weights")?;
        }
        for &name in names {
            if let Some(tensor) = slot.tensors.remove(name) {
                taken.insert(name.to_owned(), tensor);
            }
        }
        Ok(taken)
    }
}

fn dtype_from_metadata(dtype: &str) -> Option<DType> {
    match dtype {
        "BF16" => Some(DType::BF16),
        "F16" => Some(DType::F16),
        "F32" => Some(DType::F32),
        _ => None,
    }
}

/// Copy big stages into the slab with bounded thread parallelism; one memcpy
/// thread streams roughly 2 GB/s, which cannot hide behind the stage compute.
fn fill_parallel(destination: &mut [u8], source: &[u8]) {
    const MIN_PARALLEL_BYTES: usize = 4 << 20;
    if source.len() < MIN_PARALLEL_BYTES {
        destination.copy_from_slice(source);
        return;
    }
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let chunk = source.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (source, destination) in source.chunks(chunk).zip(destination.chunks_mut(chunk)) {
            scope.spawn(move || destination.copy_from_slice(source));
        }
    });
}

/// Stage one tensor's bytes into the persistent slab at `offset` and enqueue
/// the device copy from there. Returns the tensor and the bytes consumed.
fn upload_async(
    weights: &ModelWeights,
    name: &str,
    stream: &Arc<CudaStream>,
    device: &CudaDevice,
    slab: &mut Option<PinnedHostSlice<u8>>,
    offset: usize,
) -> Result<Option<(Tensor, usize)>> {
    let metadata = weights.metadata(name)?;
    let Some(dtype) = dtype_from_metadata(&metadata.dtype) else {
        return Ok(None);
    };
    if metadata.bytes == 0 {
        return Ok(None);
    }
    let count = metadata
        .shape
        .iter()
        .try_fold(1usize, |n, &dimension| n.checked_mul(dimension))
        .with_context(|| format!("prefetch shape overflow for {name}"))?;
    ensure!(count > 0, "prefetch tensor {name} is empty");
    let bytes_len = metadata.bytes;
    let slab = slab
        .as_mut()
        .with_context(|| format!("prefetch slab is not allocated for {name}"))?;
    weights
        .with_tensor_bytes(name, move |bytes| {
            ensure!(
                bytes.len() == bytes_len,
                "prefetch byte count mismatch for {name}"
            );
            let staged = slab
                .as_mut_slice()
                .with_context(|| format!("lock the prefetch slab for {name}"))?;
            fill_parallel(&mut staged[offset..offset + bytes_len], bytes);
            let host = slab
                .as_slice()
                .with_context(|| format!("lock the prefetch slab for {name}"))?;
            let staged = &host[offset..offset + bytes_len];
            macro_rules! upload {
                ($element:ty) => {{
                    let bytes = count * std::mem::size_of::<$element>();
                    let mut slice = unsafe { stream.alloc::<$element>(count)? };
                    let pointer = slice.device_ptr_mut(stream).0;
                    unsafe {
                        result::memcpy_htod_async(pointer, &staged[..bytes], stream.cu_stream())
                    }
                    .with_context(|| format!("enqueue prefetch copy for {name}"))?;
                    Ok(Tensor::from_storage(
                        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                            slice,
                            device.clone(),
                        )),
                        metadata.shape.clone(),
                        BackpropOp::none(),
                        false,
                    ))
                }};
            }
            match dtype {
                DType::BF16 => upload!(half::bf16),
                DType::F16 => upload!(half::f16),
                DType::F32 => upload!(f32),
                _ => unreachable!("prefetch dtype filtered above"),
            }
        })
        .map(|tensor| Some((tensor, bytes_len)))
}
