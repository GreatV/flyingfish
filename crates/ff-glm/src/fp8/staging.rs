//! Two reusable pinned upload slots, ordered independently from model compute.
use super::cuda::WeightPool;
use anyhow::{Context, Result, ensure};
use candle_core::cuda_backend::cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, HostSlice, LaunchConfig, PinnedHostSlice, PushKernelArg,
    SyncOnDrop, sys,
};
use candle_core::{DType, Storage, Tensor, op::BackpropOp};
use ff_core::weights::ModelWeights;
use std::{collections::VecDeque, sync::Arc};

/// Preserve cudarc's pinned-memory event guard when copying only a prefix.
struct Prefix<'a, T> {
    data: &'a mut PinnedHostSlice<T>,
    len: usize,
}

impl<T> HostSlice<T> for Prefix<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        let (data, guard) = unsafe { self.data.stream_synced_slice(stream) };
        (&data[..self.len], guard)
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        let (data, guard) = unsafe { self.data.stream_synced_mut_slice(stream) };
        (&mut data[..self.len], guard)
    }
}

struct Slot {
    host_weight: PinnedHostSlice<u8>,
    host_scales: PinnedHostSlice<f32>,
    weight: CudaSlice<u8>,
    scales: CudaSlice<f32>,
}

impl Slot {
    fn new(stream: &Arc<CudaStream>, weight_bytes: usize, scale_count: usize) -> Result<Self> {
        let context = stream.context();
        let mut host_weight = unsafe { context.alloc_pinned::<u8>(weight_bytes) }?;
        let mut host_scales = unsafe { context.alloc_pinned::<f32>(scale_count) }?;
        unsafe {
            host_weight.as_mut_ptr()?.write_bytes(0, weight_bytes);
            host_scales.as_mut_ptr()?.write_bytes(0, scale_count);
        }
        Ok(Self {
            host_weight,
            host_scales,
            weight: unsafe { stream.alloc(weight_bytes) }?,
            scales: unsafe { stream.alloc(scale_count) }?,
        })
    }
    fn bytes(&self) -> usize {
        self.weight.len() + self.scales.len() * 4
    }
}

struct Trace {
    kind: &'static str,
    start: CudaEvent,
    end: CudaEvent,
}

pub(super) struct Staging {
    stream: Arc<CudaStream>,
    slots: [Option<Slot>; 2],
    next: usize,
    uploads: u64,
    uploaded_bytes: u64,
    slot_allocations: u64,
    epoch: Option<CudaEvent>,
    trace: VecDeque<Trace>,
}

impl Staging {
    pub(super) fn new(device: &candle_core::CudaDevice, trace: bool) -> Result<Self> {
        ensure!(
            device.cuda_stream().context().is_event_tracking(),
            "pinned FP8 transfer requires CUDA event tracking"
        );
        let compute = device.cuda_stream();
        let stream = compute.context().new_stream()?;
        let epoch = if trace {
            let epoch = device
                .cuda_stream()
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            stream.wait(&epoch)?;
            Some(epoch)
        } else {
            None
        };
        Ok(Self {
            stream,
            slots: [None, None],
            next: 0,
            uploads: 0,
            uploaded_bytes: 0,
            slot_allocations: 0,
            epoch,
            trace: VecDeque::new(),
        })
    }

    pub(super) fn begin(&self, stream: &CudaStream) -> Result<Option<CudaEvent>> {
        self.epoch
            .as_ref()
            .map(|_| stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))
            .transpose()
            .map_err(Into::into)
    }
    pub(super) fn end(
        &mut self,
        kind: &'static str,
        start: Option<CudaEvent>,
        stream: &CudaStream,
    ) -> Result<()> {
        if let Some(start) = start {
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            if self.trace.len() == 128 {
                self.trace.pop_front();
            }
            self.trace.push_back(Trace { kind, start, end });
        }
        Ok(())
    }
    pub(super) fn bytes(&self) -> u64 {
        self.slots.iter().flatten().map(Slot::bytes).sum::<usize>() as u64
    }

    pub(super) fn stats(&self) -> Result<super::Fp8TransferStats> {
        let mut result = super::Fp8TransferStats {
            uploads: self.uploads,
            uploaded_bytes: self.uploaded_bytes,
            slot_allocations: self.slot_allocations,
            staging_bytes_per_tier: self.slots.iter().flatten().map(Slot::bytes).sum::<usize>()
                as u64,
            trace: vec![],
        };
        if let Some(epoch) = &self.epoch {
            for event in &self.trace {
                result.trace.push(super::Fp8TransferInterval {
                    kind: event.kind.into(),
                    start_ms: epoch.elapsed_ms(&event.start)?,
                    end_ms: epoch.elapsed_ms(&event.end)?,
                });
            }
        }
        Ok(result)
    }

    pub(super) fn load(
        &mut self,
        weights: &ModelWeights,
        name: &str,
        scale_name: &str,
        device: &candle_core::CudaDevice,
        dtype: DType,
        pool: &WeightPool,
    ) -> Result<Tensor> {
        super::validate_block_fp8_metadata(weights, name, scale_name, dtype)?;
        let metadata = weights.raw_tensor_metadata(name)?;
        let scales = weights.raw_tensor_metadata(scale_name)?;
        let count = metadata.bytes;
        ensure!(
            count > 0 && count <= i32::MAX as usize && scales.bytes.is_multiple_of(4),
            "invalid FP8 staging dimensions"
        );
        let scale_count = scales.bytes / 4;
        let rows = u32::try_from(metadata.shape[0])?;
        let cols = u32::try_from(metadata.shape[1])?;
        let stream = self.stream.clone();
        let index = self.next;
        if self.slots[index]
            .as_ref()
            .is_none_or(|s| s.weight.len() < count || s.scales.len() < scale_count)
        {
            self.slots[index] = None;
            self.slots[index] = Some(Slot::new(&stream, count, scale_count)?);
            self.slot_allocations += 1;
        }
        let slot = self.slots[index].as_mut().unwrap();
        weights.with_tensor_bytes(name, |bytes| {
            ensure!(
                bytes.len() == count,
                "FP8 byte length changed after metadata inspection"
            );
            slot.host_weight.as_mut_slice()?[..count].copy_from_slice(bytes);
            Ok(())
        })?;
        weights.with_tensor_bytes(scale_name, |bytes| {
            ensure!(bytes.len() == scales.bytes, "FP8 scale byte length changed");
            for (dst, bytes) in slot.host_scales.as_mut_slice()?[..scale_count]
                .iter_mut()
                .zip(bytes.chunks_exact(4))
            {
                *dst = f32::from_le_bytes(bytes.try_into().unwrap());
            }
            Ok(())
        })?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let start = self
            .epoch
            .as_ref()
            .map(|_| stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)))
            .transpose()?;
        stream.memcpy_htod(
            &Prefix {
                data: &mut slot.host_weight,
                len: count,
            },
            &mut slot.weight.slice_mut(..count),
        )?;
        stream.memcpy_htod(
            &Prefix {
                data: &mut slot.host_scales,
                len: scale_count,
            },
            &mut slot.scales.slice_mut(..scale_count),
        )?;
        self.end("h2d", start, &stream)?;
        let slot = self.slots[index].as_mut().unwrap();
        let name = match dtype {
            DType::BF16 => "glm_fp8_dequant_bf16",
            DType::F32 => "glm_fp8_dequant_f32",
            _ => unreachable!(),
        };
        let kernel = device.get_or_load_custom_func(
            name,
            "glm_fp8_dequant_v1",
            include_str!(concat!(env!("OUT_DIR"), "/glm_fp8_dequant.ptx")),
        )?;
        let raw = slot.weight.slice(..count);
        let scales = slot.scales.slice(..scale_count);
        macro_rules! execute {
            ($ty:ty) => {{
                let mut allocated = unsafe { pool.allocate_on::<$ty>(&stream, count) }?;
                let mut launch = stream.launch_builder(&kernel);
                launch
                    .arg(&raw)
                    .arg(&scales)
                    .arg(&mut allocated)
                    .arg(&rows)
                    .arg(&cols);
                unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }?;
                candle_core::CudaStorage::wrap_cuda_slice(allocated, device.clone())
            }};
        }
        let output = if dtype == DType::BF16 {
            execute!(half::bf16)
        } else {
            execute!(f32)
        };
        self.next = (index + 1) % 2;
        self.uploads += 1;
        self.uploaded_bytes = self
            .uploaded_bytes
            .checked_add((count + scale_count * 4) as u64)
            .context("FP8 transfer byte counter overflow")?;
        #[cfg(test)]
        if std::env::var_os("FF_GLM_SYNC_FP8_UPLOAD").is_some() {
            self.stream.synchronize()?;
        }
        Ok(Tensor::from_storage(
            Storage::Cuda(output),
            metadata.shape,
            BackpropOp::none(),
            false,
        ))
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
