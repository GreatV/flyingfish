//! Exact CUDA BF16 LayerNorm for the released Qwen3-VL vision geometry.
//!
//! The CUDA kernel is restricted to the four real FL2VA/Ref2VA shapes whose
//! official PyTorch dispatch and result bits are covered by the oracle gate.
//! Other devices continue to use their existing implementation; this module
//! never substitutes a fallback for an unsupported CUDA shape.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Layout, Shape, Tensor};

const CUDA_MODULE: &str = "flyingfish_qwen_layer_norm_pytorch_7269437";
const EPSILON: f64 = 1e-6;

pub(crate) const BACKEND: &str = concat!(
    "pytorch-vectorized-layer-norm-bf16-cuda@7269437d655783a26cba32aa88195b741ff496aa/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
    "/shape-profile:1152x4032|28224,4608x1008|7056/eps1e-6-v1"
);

const SUPPORTED_SHAPES: &[(usize, usize)] = &[
    (4_032, 1_152),
    (28_224, 1_152),
    (1_008, 4_608),
    (7_056, 4_608),
];

#[derive(Debug, Clone, Copy)]
struct QwenLayerNormBf16 {
    epsilon: f32,
}

fn checked_storage_end(
    label: &str,
    start: usize,
    elements: usize,
    storage_elements: usize,
) -> candle_core::Result<usize> {
    let end = start.checked_add(elements).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "Qwen LayerNorm {label} storage range overflows usize"
        ))
    })?;
    if end > storage_elements {
        candle_core::bail!(
            "Qwen LayerNorm {label} range {start}..{end} exceeds storage length {storage_elements}"
        )
    }
    Ok(end)
}

impl CustomOp3 for QwenLayerNormBf16 {
    fn name(&self) -> &'static str {
        BACKEND
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
        candle_core::bail!("exact Qwen vision BF16 LayerNorm is CUDA-only")
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
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        crate::cuda::device::require_tuned_kernel(input.device())?;
        if input.device().id() != weight.device().id() || input.device().id() != bias.device().id()
        {
            candle_core::bail!("Qwen LayerNorm tensors must be on the same CUDA device")
        }
        if input.dtype() != candle_core::DType::BF16
            || weight.dtype() != candle_core::DType::BF16
            || bias.dtype() != candle_core::DType::BF16
        {
            candle_core::bail!("exact Qwen vision LayerNorm requires BF16 tensors")
        }
        if !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !bias_layout.is_contiguous()
        {
            candle_core::bail!("exact Qwen vision LayerNorm requires contiguous tensors")
        }
        let [rows, width] = input_layout.dims() else {
            candle_core::bail!("exact Qwen vision LayerNorm input must be rank two")
        };
        if !SUPPORTED_SHAPES.contains(&(*rows, *width)) {
            candle_core::bail!(
                "unverified Qwen vision LayerNorm shape {:?}; supported shapes are [4032,1152], [28224,1152], [1008,4608], and [7056,4608]",
                input_layout.dims()
            )
        }
        if weight_layout.dims() != [*width] || bias_layout.dims() != [*width] {
            candle_core::bail!(
                "Qwen vision LayerNorm affine shapes must both be [{width}], got weight={:?}, bias={:?}",
                weight_layout.dims(),
                bias_layout.dims()
            )
        }
        if input_layout.start_offset() != 0
            || weight_layout.start_offset() != 0
            || bias_layout.start_offset() != 0
        {
            candle_core::bail!(
                "Qwen vision LayerNorm only accepts the zero-offset vec4 BF16 layouts covered by the official oracle"
            )
        }

        let elements = rows.checked_mul(*width).ok_or_else(|| {
            candle_core::Error::Msg("Qwen LayerNorm element count overflows usize".into())
        })?;
        let rows_u32 = u32::try_from(*rows).map_err(|_| {
            candle_core::Error::Msg("Qwen LayerNorm rows exceed CUDA grid.x".into())
        })?;
        let width_i32 = i32::try_from(*width)
            .map_err(|_| candle_core::Error::Msg("Qwen LayerNorm width exceeds i32".into()))?;

        let device = input.device().clone();
        let input = input.as_cuda_slice::<half::bf16>()?;
        checked_storage_end("input", input_layout.start_offset(), elements, input.len())?;
        let input = input.slice(input_layout.start_offset()..);
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        checked_storage_end("weight", weight_layout.start_offset(), *width, weight.len())?;
        let weight = weight.slice(weight_layout.start_offset()..);
        let bias = bias.as_cuda_slice::<half::bf16>()?;
        checked_storage_end("bias", bias_layout.start_offset(), *width, bias.len())?;
        let bias = bias.slice(bias_layout.start_offset()..);

        let mut output = unsafe { device.alloc::<half::bf16>(elements) }?;
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::QWEN_LAYER_NORM_BF16,
            CUDA_MODULE,
            "qwen_layer_norm_bf16",
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        candle_core::builder_arg!(builder, width_i32, self.epsilon);
        builder.arg(&input);
        builder.arg(&weight);
        builder.arg(&bias);
        builder.arg(&mut output);
        let launch = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 4 * 3 / 2 * size_of::<f32>() as u32,
        };
        unsafe { builder.launch(launch) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            input_layout.shape().clone(),
        ))
    }
}

pub(crate) fn layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    epsilon: f64,
) -> candle_core::Result<Tensor> {
    if epsilon.to_bits() != EPSILON.to_bits() {
        candle_core::bail!(
            "exact Qwen vision LayerNorm only supports epsilon {EPSILON}, got {epsilon}"
        )
    }
    input.apply_op3_no_bwd(
        weight,
        bias,
        &QwenLayerNormBf16 {
            epsilon: EPSILON as f32,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn rejects_cpu_instead_of_falling_back() {
        let input = Tensor::zeros((4_032, 1_152), DType::BF16, &Device::Cpu).unwrap();
        let weight = Tensor::ones(1_152, DType::BF16, &Device::Cpu).unwrap();
        let bias = Tensor::zeros(1_152, DType::BF16, &Device::Cpu).unwrap();
        let error = layer_norm(&input, &weight, &bias, EPSILON)
            .unwrap_err()
            .to_string();
        assert!(error.contains("CUDA-only"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_a_different_epsilon_before_dispatch() {
        let input = Tensor::zeros((4_032, 1_152), DType::BF16, &Device::Cpu).unwrap();
        let weight = Tensor::ones(1_152, DType::BF16, &Device::Cpu).unwrap();
        let bias = Tensor::zeros(1_152, DType::BF16, &Device::Cpu).unwrap();
        let error = layer_norm(&input, &weight, &bias, 1e-5)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("only supports epsilon"),
            "unexpected error: {error}"
        );
    }
}
