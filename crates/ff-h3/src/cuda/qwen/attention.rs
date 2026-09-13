//! Bounded QK/PV candidates for the released Qwen text-357 and vision grids.
//!
//! The official oracle executes one full query matrix per sequence/vision
//! segment. The current Flyingfish candidate uses configured 256-row query
//! chunks plus an explicit tail. This module keeps those contracts separate
//! and exists so empirical bit parity, rather than shape totals, decides which
//! implementation can be admitted.

use crate::core;
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Layout, Shape, Tensor};
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

const HEAD_MEAN_CUDA_MODULE: &str = "flyingfish_qwen_head_mean_pytorch_7269437";

pub(crate) const HEAD_RMS_NORM_BACKEND: &str = concat!(
    "pytorch-reduce-mean-f32-width128|5120@7269437d655783a26cba32aa88195b741ff496aa/",
    "source-sha256:ece80cefdf02a3ccc1dc5dc360d3065fa34324d1cb221a3ad887c329aa663280/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
    "/qwen-rsqrt:pytorch-unary-rsqrt-f32@7269437d655783a26cba32aa88195b741ff496aa",
    "/shape-profile:text357|1935|8620-q64|k8-head128-hidden5120-v4"
);

#[derive(Clone, Copy, Debug)]
struct PytorchMeanWidth128F32;

impl CustomOp1 for PytorchMeanWidth128F32 {
    fn name(&self) -> &'static str {
        HEAD_RMS_NORM_BACKEND
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("exact Qwen width-128 head mean is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        crate::cuda::device::require_tuned_kernel(input.device())?;
        if input.dtype() != DType::F32 || !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!(
                "exact Qwen head mean requires zero-offset contiguous F32 squared input"
            )
        }
        if !matches!(
            layout.dims(),
            [1, 357 | 1_935 | 8_620, 64, 128]
                | [1, 357 | 1_935 | 8_620, 8, 128]
                | [1, 357 | 1_935 | 8_620, 5120]
        ) {
            candle_core::bail!("unverified Qwen head-mean shape {:?}", layout.dims())
        }
        let elements = layout.shape().elem_count();
        let width = *layout.dims().last().unwrap();
        let rows = elements / width;
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| candle_core::Error::Msg("Qwen head-mean rows exceed i32".into()))?;
        let grid_x = u32::try_from(rows.div_ceil(16))
            .map_err(|_| candle_core::Error::Msg("Qwen head-mean grid exceeds u32".into()))?;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<f32>()?;
        if elements > input.len() {
            candle_core::bail!("Qwen head-mean input exceeds backing storage")
        }
        let mut output = unsafe { device.alloc::<f32>(rows) }?;
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::QWEN_HEAD_MEAN_F32,
            HEAD_MEAN_CUDA_MODULE,
            if width == 128 {
                "qwen_head_mean_width128_f32"
            } else {
                "qwen_hidden_mean_width5120_f32"
            },
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        candle_core::builder_arg!(builder, rows_i32);
        builder.arg(input);
        builder.arg(&mut output);
        let launch = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(launch) }.w()?;
        let mut output_dims = layout.dims().to_vec();
        *output_dims.last_mut().expect("rank-four shape is nonempty") = 1;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            Shape::from_dims(&output_dims),
        ))
    }
}

/// The row counts the two exact RMSNorm kernels were validated against.
///
/// Callers ask before dispatching so an off-profile shape reaches the portable
/// path rather than this module's refusal.
pub(crate) const EXACT_RMS_NORM_ROWS: [usize; 3] = [357, 1_935, 8_620];

/// Whether the exact hidden-width RMSNorm covers this input.
pub(crate) fn hidden_rms_norm_width5120_covers(input: &Tensor) -> bool {
    matches!(input.dims(), [1, rows, 5120] if EXACT_RMS_NORM_ROWS.contains(rows))
}

/// Whether the exact head-width RMSNorm covers this input.
pub(crate) fn head_rms_norm_width128_covers(input: &Tensor) -> bool {
    matches!(
        input.dims(),
        [1, rows, 64 | 8, 128] if EXACT_RMS_NORM_ROWS.contains(rows)
    )
}

pub(crate) fn hidden_rms_norm_width5120(
    input: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> candle_core::Result<Tensor> {
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().same_device(weight.device())
        || weight.dims() != [5120]
        || !matches!(input.dims(), [1, 357 | 1_935 | 8_620, 5120])
    {
        candle_core::bail!("unverified Qwen hidden RMSNorm shape/dtype")
    }
    let input_f32 = input.to_dtype(DType::F32)?;
    let variance = input_f32.sqr()?.apply_op1_no_bwd(&PytorchMeanWidth128F32)?;
    let inverse_root = crate::cuda::rms_norm::qwen_rsqrt(&(&variance + epsilon)?)?;
    weight.broadcast_mul(
        &input_f32
            .broadcast_mul(&inverse_root)?
            .to_dtype(DType::BF16)?,
    )
}

pub(crate) fn head_rms_norm_width128(
    input: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> candle_core::Result<Tensor> {
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || !input.is_contiguous()
        || input.layout().start_offset() != 0
        || !weight.is_contiguous()
        || weight.layout().start_offset() != 0
        || weight.dims() != [128]
        || !matches!(
            input.dims(),
            [1, 357 | 1_935 | 8_620, 64, 128] | [1, 357 | 1_935 | 8_620, 8, 128]
        )
    {
        candle_core::bail!(
            "exact Qwen head RMSNorm requires CUDA zero-offset contiguous BF16 text357|1935|8620 q64/k8 head128 inputs"
        )
    }
    let candle_core::Device::Cuda(cuda) = input.device() else {
        candle_core::bail!("exact Qwen head RMSNorm is CUDA-only")
    };
    crate::cuda::device::require_tuned_kernel(cuda)?;
    let input_f32 = input.to_dtype(DType::F32)?;
    let variance = input_f32.sqr()?.apply_op1_no_bwd(&PytorchMeanWidth128F32)?;
    let inverse_root = crate::cuda::rms_norm::qwen_rsqrt(&(&variance + epsilon)?)?;
    let pre_weight = input_f32
        .broadcast_mul(&inverse_root)?
        .to_dtype(DType::BF16)?;
    weight.broadcast_mul(&pre_weight)
}

pub(crate) const QK_PV_BACKEND: &str = "nvidia-rtx4090-sm89-sm128-driver595.84-cublaslt130401/bf16-compute32-no-bias/workspace-4194304/heuristic-first/official-strided-qk-pv-descriptors/shape-profile:text357-fullq357|text1935-query256-tail143|text8620-query256-tail172-operator-reference-q64-kv8-head128-v3";
const SCALE_CUDA_MODULE: &str = "flyingfish_qwen_attention_scale_pytorch_7269437";
pub(crate) const SCALE_BACKEND: &str = concat!(
    "pytorch-binary-mul-bf16-cuda@7269437d655783a26cba32aa88195b741ff496aa/",
    "source-sha256:9dcf411ce2d3dfb5aeb708788488206e6761c8bf6dd0d6a0dbd2c41506e9306d/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
    "/shape-profile:text357x357-query101|256|357|text1935x1935-query143|256|text8620x8620-query172|256-operator-reference|vision4032x4032-query192|256|4032-v3"
);

#[derive(Clone, Copy, Debug)]
struct PytorchBf16Scale {
    scale: f32,
}

impl candle_core::CustomOp1 for PytorchBf16Scale {
    fn name(&self) -> &'static str {
        SCALE_BACKEND
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("exact Qwen BF16 attention scale is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        crate::cuda::device::require_tuned_kernel(input.device())?;
        if input.dtype() != DType::BF16 || !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!("exact Qwen attention scale requires zero-offset contiguous BF16")
        }
        let [batch, heads, rows, width] = layout.dims() else {
            candle_core::bail!("exact Qwen attention scores must be rank four")
        };
        if *batch != 1
            || !matches!(
                (*heads, *rows, *width),
                (64, 101 | 256 | 357, 357)
                    | (64, 143 | 256, 1_935)
                    | (64, 172 | 256, 8_620)
                    | (16, 192 | 256 | 4_032, 4_032)
            )
        {
            candle_core::bail!("unverified Qwen attention-scale shape {:?}", layout.dims())
        }
        let elements = layout.shape().elem_count();
        let elements_i32 = i32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("Qwen scale elements exceed i32".into()))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("Qwen scale elements exceed u32".into()))?;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<half::bf16>()?;
        if elements > input.len() {
            candle_core::bail!("Qwen scale input exceeds backing storage")
        }
        let mut output = unsafe { device.alloc::<half::bf16>(elements) }?;
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::QWEN_ATTENTION_SCALE_BF16,
            SCALE_CUDA_MODULE,
            "qwen_attention_scale_bf16",
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        candle_core::builder_arg!(builder, elements_i32, self.scale);
        builder.arg(input);
        builder.arg(&mut output);
        unsafe { builder.launch(LaunchConfig::for_num_elems(elements_u32)) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn scale_scores(input: &Tensor, head_dim: usize) -> candle_core::Result<Tensor> {
    if !matches!(head_dim, 72 | 128) {
        candle_core::bail!("unverified Qwen attention head dimension {head_dim}")
    }
    let rank_three = input.rank() == 3;
    let input = if rank_three {
        input.unsqueeze(0)?
    } else {
        input.clone()
    };
    let output = input.apply_op1_no_bwd(&PytorchBf16Scale {
        scale: (1. / (head_dim as f64).sqrt()) as f32,
    })?;
    if rank_three {
        output.squeeze(0)
    } else {
        Ok(output)
    }
}

fn blas_for_device(
    device: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<Arc<CudaBlasLT>> {
    use candle_core::cuda_backend::DeviceId;

    static HANDLES: OnceLock<Mutex<HashMap<DeviceId, Arc<CudaBlasLT>>>> = OnceLock::new();
    let handles = HANDLES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut handles = handles.lock().map_err(|_| {
        candle_core::Error::Msg("Qwen attention cuBLASLt handle lock poisoned".into())
    })?;
    if let Some(handle) = handles.get(&device.id()) {
        return Ok(handle.clone());
    }
    crate::cuda::profile::validate_cuda_device(device)?;
    let handle = Arc::new(CudaBlasLT::new(device.cuda_stream()).map_err(|error| {
        candle_core::Error::Msg(format!(
            "failed to create Qwen attention cuBLASLt handle: {error:?}"
        ))
    })?);
    handles.insert(device.id(), handle.clone());
    Ok(handle)
}

#[derive(Clone, Copy, Debug)]
enum BatchedMatmulKind {
    Qk,
    Pv,
}

#[derive(Clone, Copy, Debug)]
struct StridedBatchedMatmul(BatchedMatmulKind);

fn last_referenced_element(layout: &Layout) -> candle_core::Result<usize> {
    let mut last = layout.start_offset();
    for (&dimension, &stride) in layout.dims().iter().zip(layout.stride()) {
        if dimension == 0 {
            candle_core::bail!("Qwen attention matmul does not support empty tensors")
        }
        last = last
            .checked_add((dimension - 1).checked_mul(stride).ok_or_else(|| {
                candle_core::Error::Msg("Qwen attention layout offset overflow".into())
            })?)
            .ok_or_else(|| {
                candle_core::Error::Msg("Qwen attention layout offset overflow".into())
            })?;
    }
    Ok(last)
}

impl CustomOp2 for StridedBatchedMatmul {
    fn name(&self) -> &'static str {
        QK_PV_BACKEND
    }

    fn cpu_fwd(
        &self,
        _left: &CpuStorage,
        _left_layout: &Layout,
        _right: &CpuStorage,
        _right_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("exact Qwen strided QK/PV matmul is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        left: &candle_core::CudaStorage,
        left_layout: &Layout,
        right: &candle_core::CudaStorage,
        right_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        if left.dtype() != DType::BF16 || right.dtype() != DType::BF16 {
            candle_core::bail!("exact Qwen strided QK/PV requires BF16 tensors")
        }
        if left.device().id() != right.device().id() {
            candle_core::bail!("Qwen QK/PV operands must share one CUDA device")
        }
        let [batch, rows, inner] = left_layout.dims() else {
            candle_core::bail!("Qwen QK/PV left operand must be rank three")
        };
        let [right_batch, right_rows, right_width] = right_layout.dims() else {
            candle_core::bail!("Qwen QK/PV right operand must be rank three")
        };
        if batch != right_batch {
            candle_core::bail!("Qwen QK/PV batch/head counts differ")
        }
        let (output_width, reduction, transa) = match self.0 {
            BatchedMatmulKind::Qk => {
                if inner != right_width {
                    candle_core::bail!("Qwen QK head dimensions differ")
                }
                (*right_rows, *inner, true)
            }
            BatchedMatmulKind::Pv => {
                if inner != right_rows {
                    candle_core::bail!("Qwen PV key dimensions differ")
                }
                (*right_width, *inner, false)
            }
        };
        let supported = match self.0 {
            BatchedMatmulKind::Qk => matches!(
                (*batch, *rows, reduction, output_width),
                (64, 101 | 256 | 357, 128, 357)
                    | (64, 143 | 256, 128, 1_935)
                    | (64, 172 | 256, 128, 8_620)
                    | (16, 192 | 256 | 4_032, 72, 4_032)
            ),
            BatchedMatmulKind::Pv => matches!(
                (*batch, *rows, reduction, output_width),
                (64, 101 | 256 | 357, 357, 128)
                    | (64, 143 | 256, 1_935, 128)
                    | (64, 172 | 256, 8_620, 128)
                    | (16, 192 | 256 | 4_032, 4_032, 72)
            ),
        };
        if !supported {
            candle_core::bail!(
                "unverified Qwen {:?} shape left={:?}, right={:?}",
                self.0,
                left_layout.dims(),
                right_layout.dims()
            )
        }
        if left_layout.stride()[2] != 1 || right_layout.stride()[2] != 1 {
            candle_core::bail!("Qwen QK/PV innermost dimensions must be contiguous")
        }
        let exact_layout = match self.0 {
            BatchedMatmulKind::Qk if *batch == 64 => {
                left_layout.stride() == [128, 8_192, 1]
                    && right_layout.stride() == [output_width * 128, 128, 1]
                    && left_layout.start_offset().is_multiple_of(8_192)
                    && right_layout.start_offset() == 0
            }
            BatchedMatmulKind::Qk if *batch == 16 => {
                left_layout.stride() == [72, 1_152, 1]
                    && right_layout.stride() == [72, 1_152, 1]
                    && left_layout.start_offset().is_multiple_of(1_152)
                    && right_layout.start_offset().is_multiple_of(1_152)
            }
            BatchedMatmulKind::Pv if *batch == 64 => {
                left_layout.stride() == [rows * inner, *inner, 1]
                    && right_layout.stride() == [reduction * 128, 128, 1]
                    && left_layout.start_offset() == 0
                    && right_layout.start_offset() == 0
            }
            BatchedMatmulKind::Pv if *batch == 16 => {
                left_layout.stride() == [rows * inner, *inner, 1]
                    && right_layout.stride() == [72, 3_456, 1]
                    && left_layout.start_offset() == 0
                    && right_layout.start_offset() % 3_456 == 2_304
            }
            _ => false,
        };
        if !exact_layout {
            candle_core::bail!(
                "unverified Qwen {:?} layout left strides/offset={:?}/{}, right={:?}/{}",
                self.0,
                left_layout.stride(),
                left_layout.start_offset(),
                right_layout.stride(),
                right_layout.start_offset()
            )
        }
        let left_storage = left.as_cuda_slice::<half::bf16>()?;
        let right_storage = right.as_cuda_slice::<half::bf16>()?;
        if last_referenced_element(left_layout)? >= left_storage.len()
            || last_referenced_element(right_layout)? >= right_storage.len()
        {
            candle_core::bail!("Qwen QK/PV layout exceeds backing storage")
        }
        let device = left.device().clone();
        crate::cuda::profile::validate_cuda_device(&device)?;
        let left = left_storage.slice(left_layout.start_offset()..);
        let right = right_storage.slice(right_layout.start_offset()..);
        let elements = batch
            .checked_mul(*rows)
            .and_then(|value| value.checked_mul(output_width))
            .ok_or_else(|| candle_core::Error::Msg("Qwen QK/PV output size overflow".into()))?;
        let mut output = unsafe { device.alloc::<half::bf16>(elements) }?;
        let blas = blas_for_device(&device)?;
        let (a, a_layout, b, b_layout) = match self.0 {
            BatchedMatmulKind::Qk => (&right, right_layout, &left, left_layout),
            BatchedMatmulKind::Pv => (&right, right_layout, &left, left_layout),
        };
        let config =
            MatmulConfig {
                transa,
                transb: false,
                transc: false,
                m: output_width as u64,
                n: *rows as u64,
                k: reduction as u64,
                alpha: 1.0,
                lda: i64::try_from(a_layout.stride()[1])
                    .map_err(|_| candle_core::Error::Msg("Qwen QK/PV lda exceeds i64".into()))?,
                ldb: i64::try_from(b_layout.stride()[1])
                    .map_err(|_| candle_core::Error::Msg("Qwen QK/PV ldb exceeds i64".into()))?,
                beta: 0.0,
                ldc: output_width as i64,
                stride_a: Some(i64::try_from(a_layout.stride()[0]).map_err(|_| {
                    candle_core::Error::Msg("Qwen QK/PV stride_a exceeds i64".into())
                })?),
                stride_b: Some(i64::try_from(b_layout.stride()[0]).map_err(|_| {
                    candle_core::Error::Msg("Qwen QK/PV stride_b exceeds i64".into())
                })?),
                stride_c: Some(i64::try_from(rows * output_width).map_err(|_| {
                    candle_core::Error::Msg("Qwen QK/PV stride_c exceeds i64".into())
                })?),
                stride_bias: None,
                batch_size: Some(
                    i32::try_from(*batch).map_err(|_| {
                        candle_core::Error::Msg("Qwen QK/PV batch exceeds i32".into())
                    })?,
                ),
            };
        unsafe { blas.matmul(config, a, b, &mut output, None, None) }.map_err(|error| {
            candle_core::Error::Msg(format!("Qwen strided cuBLASLt matmul failed: {error:?}"))
        })?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            Shape::from_dims(&[*batch, *rows, output_width]),
        ))
    }
}

pub(crate) fn qk_matmul(query: &Tensor, key: &Tensor) -> candle_core::Result<Tensor> {
    let rank_three = query.rank() == 3;
    let query = if rank_three {
        query.unsqueeze(0)?
    } else {
        query.clone()
    };
    let key = if rank_three {
        key.unsqueeze(0)?
    } else {
        key.clone()
    };
    let output = query
        .squeeze(0)?
        .apply_op2_no_bwd(
            &key.squeeze(0)?,
            &StridedBatchedMatmul(BatchedMatmulKind::Qk),
        )?
        .unsqueeze(0)?;
    if rank_three {
        output.squeeze(0)
    } else {
        Ok(output)
    }
}

pub(crate) fn pv_matmul(probabilities: &Tensor, value: &Tensor) -> candle_core::Result<Tensor> {
    let rank_three = probabilities.rank() == 3;
    let probabilities = if rank_three {
        probabilities.unsqueeze(0)?
    } else {
        probabilities.clone()
    };
    let value = if rank_three {
        value.unsqueeze(0)?
    } else {
        value.clone()
    };
    let output = probabilities
        .squeeze(0)?
        .apply_op2_no_bwd(
            &value.squeeze(0)?,
            &StridedBatchedMatmul(BatchedMatmulKind::Pv),
        )?
        .unsqueeze(0)?;
    if rank_three {
        output.squeeze(0)
    } else {
        Ok(output)
    }
}

fn require_bf16(input: &Tensor, label: &str, require_contiguous: bool) -> candle_core::Result<()> {
    if input.dtype() != DType::BF16
        || (require_contiguous && !input.is_contiguous())
        || (require_contiguous && input.layout().start_offset() != 0)
    {
        candle_core::bail!(
            "exact Qwen {label} must be zero-offset contiguous BF16, got dtype {:?}, shape {:?}, offset {}",
            input.dtype(),
            input.dims(),
            input.layout().start_offset()
        )
    }
    if !input.device().is_cuda() {
        candle_core::bail!("exact Qwen QK/PV is CUDA-only")
    }
    Ok(())
}

fn repeat_kv(input: &Tensor, groups: usize) -> candle_core::Result<Tensor> {
    let (batch, heads, sequence, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, heads, groups, sequence, head_dim))?
        .contiguous()?
        .reshape((batch, heads * groups, sequence, head_dim))
}

pub(crate) fn text_357_exact(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    causal_mask: &Tensor,
) -> candle_core::Result<Tensor> {
    for (tensor, label) in [
        (query, "text query"),
        (key, "text key"),
        (value, "text value"),
    ] {
        require_bf16(tensor, label, false)?;
    }
    require_bf16(causal_mask, "text causal mask", true)?;
    if query.dims() != [1, 64, 357, 128]
        || key.dims() != [1, 8, 357, 128]
        || value.dims() != [1, 8, 357, 128]
        || causal_mask.dims() != [1, 1, 357, 357]
    {
        candle_core::bail!("unverified exact Qwen text-357 QK/PV geometry")
    }
    if !query.device().same_device(key.device())
        || !query.device().same_device(value.device())
        || !query.device().same_device(causal_mask.device())
    {
        candle_core::bail!("Qwen text-357 QK/PV tensors must share one CUDA device")
    }
    crate::cuda::profile::validate_exact_profile(query.device())
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    let key = repeat_kv(key, 8)?.contiguous()?;
    let value = repeat_kv(value, 8)?.contiguous()?;
    let scores_raw = qk_matmul(query, &key)?;
    let scores = scale_scores(&scores_raw, 128)?.broadcast_add(causal_mask)?;
    let probabilities = core::qwen_softmax_last_dim(&scores)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    pv_matmul(&probabilities, &value)?
        .transpose(1, 2)?
        .contiguous()
}
