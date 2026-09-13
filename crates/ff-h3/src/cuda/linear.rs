//! cuBLASLt BF16 linear+bias epilogue for official H3 parity.
//!
//! Only the three released BF16 biased H3 projection families and their
//! verified row ranges are accepted. There is no unfused fallback.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Layout, Shape, Tensor};
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

pub(crate) const MAX_CONTEXT_ROWS: usize = 16_384;
pub(crate) const MAX_TIMESTEP_ROWS: usize = 4;
const VERIFIED_CONTEXT_ROWS: std::ops::RangeInclusive<usize> = 1..=MAX_CONTEXT_ROWS;

pub(crate) const BACKEND: &str = "nvidia-rtx4090-sm89-sm128-driver595.84-cublaslt130401/bf16-compute32-bias-epilogue/workspace-4194304/heuristic-first/shape-profile:h3-transformer-context-time-v2";
pub(crate) const QWEN_BACKEND: &str = "nvidia-rtx4090-sm89-sm128-driver595.84-cublaslt130401/bf16-compute32-bias-epilogue/workspace-4194304/heuristic-first/shape-profile:qwen3vl-six-flinear-families-vision4032|28224-merger1008|7056-v2";

fn blas_for_device(
    device: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<Arc<CudaBlasLT>> {
    use candle_core::cuda_backend::DeviceId;

    crate::cuda::profile::validate_cuda_device(device)?;
    static HANDLES: OnceLock<Mutex<HashMap<DeviceId, Arc<CudaBlasLT>>>> = OnceLock::new();
    let handles = HANDLES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut handles = handles
        .lock()
        .map_err(|_| candle_core::Error::Msg("cuBLASLt handle cache lock is poisoned".into()))?;
    if let Some(handle) = handles.get(&device.id()) {
        return Ok(handle.clone());
    }
    let stream = device.cuda_stream();
    let handle = Arc::new(CudaBlasLT::new(stream).map_err(|error| {
        candle_core::Error::Msg(format!("failed to create cuBLASLt handle: {error:?}"))
    })?);
    handles.insert(device.id(), handle.clone());
    Ok(handle)
}

#[derive(Debug, Clone, Copy)]
enum BiasedLinearProfile {
    H3,
    Qwen,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceBiasedLinearBf16(BiasedLinearProfile);

impl CustomOp3 for ReferenceBiasedLinearBf16 {
    fn name(&self) -> &'static str {
        match self.0 {
            BiasedLinearProfile::H3 => BACKEND,
            BiasedLinearProfile::Qwen => QWEN_BACKEND,
        }
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _bias: &CpuStorage,
        _bias_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("official H3 cuBLASLt biased linear is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
        bias: &candle_core::CudaStorage,
        bias_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        if input.dtype() != candle_core::DType::BF16
            || weight.dtype() != candle_core::DType::BF16
            || bias.dtype() != candle_core::DType::BF16
        {
            candle_core::bail!("official H3 cuBLASLt biased linear requires BF16 tensors")
        }
        if !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !bias_layout.is_contiguous()
        {
            candle_core::bail!("official H3 cuBLASLt biased linear requires contiguous tensors")
        }
        if !input_layout.start_offset().is_multiple_of(128)
            || !weight_layout.start_offset().is_multiple_of(128)
            || !bias_layout.start_offset().is_multiple_of(128)
        {
            candle_core::bail!(
                "official H3 cuBLASLt biased linear requires 256-byte-aligned BF16 storage offsets"
            )
        }
        let Some(&input_width) = input_layout.dims().last() else {
            candle_core::bail!("official H3 cuBLASLt biased linear does not support scalar input")
        };
        if input_width == 0 || input_layout.shape().elem_count() == 0 {
            candle_core::bail!("official H3 cuBLASLt biased linear does not support empty input")
        }
        let [output_width, weight_input_width] = weight_layout.dims() else {
            candle_core::bail!("official H3 cuBLASLt biased linear weight must be rank two")
        };
        if *weight_input_width != input_width || bias_layout.dims() != [*output_width] {
            candle_core::bail!(
                "official H3 cuBLASLt biased linear shapes are incompatible: input={:?}, weight={:?}, bias={:?}",
                input_layout.dims(),
                weight_layout.dims(),
                bias_layout.dims()
            )
        }
        let rows = input_layout.shape().elem_count() / input_width;
        let supported_shape = match self.0 {
            BiasedLinearProfile::H3 => match (input_width, *output_width) {
                (5120, 5376) => VERIFIED_CONTEXT_ROWS.contains(&rows),
                (2688, 96768 | 10752) => (1..=MAX_TIMESTEP_ROWS).contains(&rows),
                _ => false,
            },
            BiasedLinearProfile::Qwen => match (input_width, *output_width) {
                (1152, 1152 | 3456 | 4304) | (4304, 1152) => {
                    matches!(rows, 4032 | 28224)
                }
                (4608, 4608 | 5120) => matches!(rows, 1008 | 7056),
                _ => false,
            },
        };
        if !supported_shape {
            match self.0 {
                BiasedLinearProfile::H3 => candle_core::bail!(
                    "unverified official H3 cuBLASLt biased-linear shape: rows={rows}, input_width={input_width}, output_width={output_width}; supported shapes are context M=1..={MAX_CONTEXT_ROWS} K=5120 N=5376 and timestep M=1..={MAX_TIMESTEP_ROWS} K=2688 N=96768|10752"
                ),
                BiasedLinearProfile::Qwen => candle_core::bail!(
                    "unverified official Qwen cuBLASLt biased-linear shape: rows={rows}, input_width={input_width}, output_width={output_width}"
                ),
            }
        }
        let device = input.device().clone();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(input_layout.start_offset()..);
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let weight = weight.slice(weight_layout.start_offset()..);
        let bias = bias.as_cuda_slice::<half::bf16>()?;
        let bias = bias.slice(bias_layout.start_offset()..);
        let mut output = unsafe { device.alloc::<half::bf16>(rows * output_width) }?;
        let blas = blas_for_device(&device)?;
        unsafe {
            blas.matmul(
                MatmulConfig {
                    transa: true,
                    transb: false,
                    transc: false,
                    m: *output_width as u64,
                    n: rows as u64,
                    k: input_width as u64,
                    alpha: 1.0,
                    lda: input_width as i64,
                    ldb: input_width as i64,
                    beta: 0.0,
                    ldc: *output_width as i64,
                    stride_a: None,
                    stride_b: None,
                    stride_c: None,
                    stride_bias: None,
                    batch_size: None,
                },
                &weight,
                &input,
                &mut output,
                Some(&bias),
                None,
            )
        }
        .map_err(|error| {
            candle_core::Error::Msg(format!("cuBLASLt biased matmul failed: {error:?}"))
        })?;
        let mut output_shape = input_layout.dims().to_vec();
        *output_shape
            .last_mut()
            .expect("non-scalar input shape has a final dimension") = *output_width;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            Shape::from_dims(&output_shape),
        ))
    }
}

pub(crate) fn linear(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> candle_core::Result<Tensor> {
    input.apply_op3_no_bwd(
        weight,
        bias,
        &ReferenceBiasedLinearBf16(BiasedLinearProfile::H3),
    )
}

pub(crate) fn qwen_linear(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> candle_core::Result<Tensor> {
    input.apply_op3_no_bwd(
        weight,
        bias,
        &ReferenceBiasedLinearBf16(BiasedLinearProfile::Qwen),
    )
}
