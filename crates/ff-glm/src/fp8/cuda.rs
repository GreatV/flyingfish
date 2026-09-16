use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{CudaContext, CudaSlice, DeviceRepr, result, sys};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, CustomOp2, DType, Layout, Shape, Tensor};
use std::sync::{Arc, Mutex};

/// Keep dequantized weights out of the pool used by short-lived activations
/// and FP8 upload buffers. Otherwise live cache entries pin fragmented pages.
pub(crate) struct WeightPool {
    pool: sys::CUmemoryPool,
    context: Arc<CudaContext>,
    /// Independent staging areas; FF_GLM_LOAD_LANES=1 uses a single lane.
    stagings: Vec<Mutex<Option<super::staging::Staging>>>,
    /// Pinned fill-ahead buffers, allocated lazily and grown as needed.
    fill_ring: Mutex<Option<Arc<super::staging::FillRing>>>,
    trace: bool,
}

unsafe impl Send for WeightPool {}
unsafe impl Sync for WeightPool {}

/// Upload lanes, each with its own buffers and stream
/// (`FF_GLM_LOAD_LANES`, default 1, clamped to 1..=8).
pub(crate) fn configured_lanes() -> usize {
    static LANES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LANES.get_or_init(|| {
        std::env::var("FF_GLM_LOAD_LANES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1)
            .clamp(1, 8)
    })
}

impl WeightPool {
    pub(crate) fn new(device: &candle_core::CudaDevice) -> anyhow::Result<Self> {
        let context = device.cuda_stream().context().clone();
        context.bind_to_thread()?;
        let mut properties: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        properties.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        properties.location = unsafe {
            std::mem::transmute::<[i32; 2], sys::CUmemLocation>([
                sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE as i32,
                i32::try_from(context.ordinal())?,
            ])
        };
        let pool = unsafe { result::mem_pool::create(&properties) }?;
        Ok(Self {
            pool,
            context,
            stagings: (0..configured_lanes()).map(|_| Mutex::new(None)).collect(),
            fill_ring: Mutex::new(None),
            trace: std::env::var_os("FF_GLM_TRACE_TRANSFERS").is_some_and(|v| v == "1"),
        })
    }

    /// The returned buffer is uninitialized and must be fully written before use.
    unsafe fn allocate<T: DeviceRepr>(
        &self,
        device: &candle_core::CudaDevice,
        count: usize,
    ) -> candle_core::Result<CudaSlice<T>> {
        unsafe { self.allocate_on(&device.cuda_stream(), count) }
    }

    pub(super) unsafe fn allocate_on<T: DeviceRepr>(
        &self,
        stream: &Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        count: usize,
    ) -> candle_core::Result<CudaSlice<T>> {
        if stream.context().as_ref() != self.context.as_ref() {
            candle_core::bail!("GLM weight pool belongs to a different CUDA context");
        }
        self.context.bind_to_thread().w()?;
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| candle_core::Error::Msg("GLM weight allocation overflow".into()))?;
        let mut pointer = 0;
        unsafe {
            sys::cuMemAllocFromPoolAsync(&mut pointer, bytes, self.pool, stream.cu_stream())
                .result()
                .w()?;
            Ok(stream.upgrade_device_ptr(pointer, count))
        }
    }

    pub(crate) fn load_pinned_on(
        &self,
        weights: &ff_core::weights::ModelWeights,
        name: &str,
        scale: &str,
        device: &candle_core::CudaDevice,
        dtype: DType,
        lane: usize,
    ) -> anyhow::Result<Tensor> {
        let lane = lane % self.stagings.len();
        let mut staging = self.stagings[lane]
            .lock()
            .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?;
        if staging.is_none() {
            *staging = Some(super::staging::Staging::new(device, self.trace)?);
        }
        staging
            .as_mut()
            .unwrap()
            .load(weights, name, scale, device, dtype, self)
    }

    pub(crate) fn lanes(&self) -> usize {
        self.stagings.len()
    }

    /// Allocate or grow the fill ring on the calling thread; workers only fill it.
    pub(crate) fn fill_ring(
        &self,
        depth: usize,
        weight_bytes: usize,
        scale_count: usize,
    ) -> anyhow::Result<Arc<super::staging::FillRing>> {
        let mut ring = self
            .fill_ring
            .lock()
            .map_err(|_| anyhow::anyhow!("FP8 fill ring lock poisoned"))?;
        if ring
            .as_ref()
            .is_none_or(|ring| !ring.fits(depth, weight_bytes, scale_count))
        {
            *ring = Some(Arc::new(super::staging::FillRing::new(
                &self.context,
                depth,
                weight_bytes,
                scale_count,
            )?));
        }
        Ok(ring.as_ref().unwrap().clone())
    }

    /// Upload and dequantize a staged projection on `lane`.
    /// The returned H2D event must complete before the buffer is refilled.
    pub(crate) fn upload_prepared_on(
        &self,
        lane: usize,
        device: &candle_core::CudaDevice,
        dtype: DType,
        buffer: &mut super::staging::FillBuffer,
        meta: &super::staging::FillMeta,
    ) -> anyhow::Result<(Tensor, candle_core::cuda_backend::cudarc::driver::CudaEvent)> {
        let lane = lane % self.stagings.len();
        let mut staging = self.stagings[lane]
            .lock()
            .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?;
        if staging.is_none() {
            *staging = Some(super::staging::Staging::new(device, self.trace)?);
        }
        staging
            .as_mut()
            .unwrap()
            .upload_prepared(device, dtype, self, buffer, meta)
    }

    /// Record completion of the lane's queued work for compute-stream consumers.
    /// May include later loads when callers share a lane.
    pub(crate) fn lane_ready(
        &self,
        lane: usize,
    ) -> anyhow::Result<Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>> {
        let lane = lane % self.stagings.len();
        self.stagings[lane]
            .lock()
            .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?
            .as_ref()
            .map(super::staging::Staging::ready_event)
            .transpose()
    }

    pub(crate) fn staging_bytes(&self) -> anyhow::Result<u64> {
        let mut total = 0;
        for staging in &self.stagings {
            total += staging
                .lock()
                .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?
                .as_ref()
                .map(super::staging::Staging::bytes)
                .unwrap_or(0);
        }
        Ok(total)
    }
    pub(crate) fn transfer_stats(&self) -> anyhow::Result<super::Fp8TransferStats> {
        let mut result = super::Fp8TransferStats::default();
        for staging in &self.stagings {
            let stats = staging
                .lock()
                .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?
                .as_ref()
                .map(super::staging::Staging::stats)
                .transpose()?;
            if let Some(stats) = stats {
                result.uploads += stats.uploads;
                result.uploaded_bytes += stats.uploaded_bytes;
                result.slot_allocations += stats.slot_allocations;
                result.staging_bytes_per_tier += stats.staging_bytes_per_tier;
                result.trace.extend(stats.trace);
            }
        }
        Ok(result)
    }
    pub(crate) fn tracing_enabled(&self) -> bool {
        self.trace
    }

    pub(crate) fn begin_compute(
        &self,
        device: &candle_core::CudaDevice,
    ) -> anyhow::Result<Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>> {
        if !self.trace {
            return Ok(None);
        }
        if self.stagings.len() > 1 {
            // Matvec tracing covers only lane 0; warn once when other lanes are active.
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "GLM FP8 compute trace brackets lane 0 only; {} lanes are active",
                    self.stagings.len()
                );
            }
        }
        self.stagings[0]
            .lock()
            .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?
            .as_ref()
            .map(|s| s.begin(&device.cuda_stream()))
            .transpose()
            .map(Option::flatten)
    }
    pub(crate) fn end_compute(
        &self,
        device: &candle_core::CudaDevice,
        start: Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>,
    ) -> anyhow::Result<()> {
        if let Some(start) = start
            && let Some(staging) = self.stagings[0]
                .lock()
                .map_err(|_| anyhow::anyhow!("FP8 staging lock poisoned"))?
                .as_mut()
        {
            staging.end("matmul", Some(start), &device.cuda_stream())?;
        }
        Ok(())
    }
}

impl Drop for WeightPool {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok() {
            let _ = unsafe { result::mem_pool::destroy(self.pool) };
        }
    }
}

struct Dequantize<'a> {
    dtype: DType,
    pool: Option<&'a WeightPool>,
}

impl CustomOp2 for Dequantize<'_> {
    fn name(&self) -> &'static str {
        "glm-block-fp8-f32-scale-v1"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("GLM CUDA FP8 dequantization requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        weight: &candle_core::CudaStorage,
        wl: &Layout,
        scale: &candle_core::CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        let CudaStorageSlice::F8E4M3(raw) = &weight.slice else {
            candle_core::bail!("GLM FP8 kernel requires E4M3 storage")
        };
        if !wl.is_contiguous() || !sl.is_contiguous() || scale.dtype() != DType::F32 {
            candle_core::bail!("GLM FP8 kernel requires contiguous weights and F32 scales")
        }
        let (rows, cols) = wl.shape().dims2()?;
        if rows == 0 || cols == 0 || sl.dims() != [rows.div_ceil(128), cols.div_ceil(128)] {
            candle_core::bail!("GLM FP8 kernel shape mismatch")
        }
        let count = i32::try_from(wl.shape().elem_count())
            .map_err(|_| candle_core::Error::Msg("GLM FP8 matrix exceeds i32 elements".into()))?
            as u32;
        let rows = u32::try_from(rows).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let cols = u32::try_from(cols).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let device = weight.device().clone();
        let raw = raw.slice(wl.start_offset()..wl.start_offset() + count as usize);
        let scales = scale.as_cuda_slice::<f32>()?;
        let scales = scales.slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count());
        let name = match self.dtype {
            DType::BF16 => "glm_fp8_dequant_bf16",
            DType::F32 => "glm_fp8_dequant_f32",
            _ => candle_core::bail!("GLM FP8 output must be BF16 or F32"),
        };
        let kernel = device.get_or_load_custom_func(
            name,
            "glm_fp8_dequant_v1",
            include_str!(concat!(env!("OUT_DIR"), "/glm_fp8_dequant.ptx")),
        )?;
        macro_rules! execute {
            ($t:ty) => {{
                let mut output = unsafe {
                    match self.pool {
                        Some(pool) => pool.allocate::<$t>(&device, count as usize),
                        None => device.alloc::<$t>(count as usize),
                    }
                }?;
                let mut launch = kernel.builder();
                launch
                    .arg(&raw)
                    .arg(&scales)
                    .arg(&mut output)
                    .arg(&rows)
                    .arg(&cols);
                unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }.w()?;
                candle_core::CudaStorage::wrap_cuda_slice(output, device)
            }};
        }
        let output = if self.dtype == DType::BF16 {
            execute!(half::bf16)
        } else {
            execute!(f32)
        };
        Ok((output, wl.shape().clone()))
    }
}

pub(super) fn dequantize(weight: &Tensor, scale: &Tensor, dtype: DType) -> anyhow::Result<Tensor> {
    Ok(weight
        .contiguous()?
        .apply_op2_no_bwd(&scale.contiguous()?, &Dequantize { dtype, pool: None })?)
}

pub(crate) fn load_weight(
    weights: &ff_core::weights::ModelWeights,
    name: &str,
    scale_name: &str,
    device: &candle_core::Device,
    dtype: DType,
    pool: &WeightPool,
) -> anyhow::Result<Tensor> {
    super::validate_block_fp8_metadata(weights, name, scale_name, dtype)?;
    let weight = weights.load(name, device)?;
    let scale = weights.load(scale_name, device)?;
    Ok(weight.apply_op2_no_bwd(
        &scale,
        &Dequantize {
            dtype,
            pool: Some(pool),
        },
    )?)
}
