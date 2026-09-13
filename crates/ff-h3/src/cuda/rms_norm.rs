//! CUDA BF16 RMSNorm matching the official H3 PyTorch reference.
//!
//! The kernel is intentionally restricted to the exact contiguous vec4 path
//! and the released H3 hidden/QK widths 5376/128 used by pinned PyTorch commit
//! 7269437d655783a26cba32aa88195b741ff496aa. Unsupported CUDA BF16 inputs fail
//! instead of silently selecting a different reduction tree.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Shape, Tensor};

const CUDA_MODULE: &str = "flyingfish_h3_rms_norm_pytorch_7269437";

pub(crate) const BACKEND: &str = concat!(
    "pytorch-vectorized-bf16-cuda@7269437d655783a26cba32aa88195b741ff496aa/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
    "/shape-profile:width128|5376-v2"
);
pub(crate) const QWEN_RSQRT_BACKEND: &str = concat!(
    "pytorch-qwen-unfused-rsqrt-f32@7269437d655783a26cba32aa88195b741ff496aa/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
);
pub(crate) const SUPPORTED_H3_WIDTHS: [usize; 2] = [128, 5_376];

fn validate_h3_width(width: usize) -> candle_core::Result<()> {
    if !SUPPORTED_H3_WIDTHS.contains(&width) {
        candle_core::bail!(
            "unverified official H3 CUDA RMSNorm width {width}; supported released widths are 128 and 5376"
        )
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct H3RmsNormBf16 {
    epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
struct QwenRsqrtF32;

impl CustomOp1 for QwenRsqrtF32 {
    fn name(&self) -> &'static str {
        QWEN_RSQRT_BACKEND
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("the official Qwen F32 rsqrt custom op is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        crate::cuda::device::require_tuned_kernel(input.device())?;
        if input.dtype() != candle_core::DType::F32 {
            candle_core::bail!("official Qwen rsqrt requires F32 input")
        }
        if !layout.is_contiguous() {
            candle_core::bail!("official Qwen rsqrt requires contiguous input")
        }
        let elements = layout.shape().elem_count();
        if elements == 0 {
            candle_core::bail!("official Qwen rsqrt does not support empty input")
        }
        let elements_i32 = i32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("Qwen rsqrt element count exceeds i32".into()))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("Qwen rsqrt element count exceeds u32".into()))?;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<f32>()?;
        let input = input.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(elements) }?;
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::H3_RMS_NORM_BF16,
            CUDA_MODULE,
            "qwen_rsqrt_f32",
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        candle_core::builder_arg!(builder, elements_i32);
        builder.arg(&input);
        builder.arg(&mut output);
        unsafe { builder.launch(LaunchConfig::for_num_elems(elements_u32)) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

impl CustomOp2 for H3RmsNormBf16 {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("the official H3 BF16 RMSNorm custom op is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        crate::cuda::device::require_tuned_kernel(input.device())?;
        if input.dtype() != candle_core::DType::BF16 || weight.dtype() != candle_core::DType::BF16 {
            candle_core::bail!("official H3 CUDA RMSNorm requires BF16 input and weight")
        }
        if !input_layout.is_contiguous() || !weight_layout.is_contiguous() {
            candle_core::bail!("official H3 CUDA RMSNorm requires contiguous input and weight")
        }
        let Some(&width) = input_layout.dims().last() else {
            candle_core::bail!("official H3 CUDA RMSNorm does not support scalar input")
        };
        if width == 0 || input_layout.shape().elem_count() == 0 {
            candle_core::bail!("official H3 CUDA RMSNorm does not support empty rows")
        }
        if weight_layout.dims() != [width] {
            candle_core::bail!(
                "official H3 CUDA RMSNorm weight shape {:?} does not match width {width}",
                weight_layout.dims()
            )
        }
        validate_h3_width(width)?;
        if width > (1usize << f32::MANTISSA_DIGITS) {
            candle_core::bail!(
                "official H3 CUDA RMSNorm width {width} exceeds the exact f32-count limit"
            )
        }
        if input_layout.shape().elem_count() > i32::MAX as usize {
            candle_core::bail!("official H3 CUDA RMSNorm limits one launch to i32::MAX elements")
        }
        if !input_layout.start_offset().is_multiple_of(4)
            || !weight_layout.start_offset().is_multiple_of(4)
        {
            candle_core::bail!(
                "official H3 CUDA RMSNorm requires vec4-aligned BF16 storage offsets"
            )
        }

        let rows = input_layout.shape().elem_count() / width;
        let rows = u32::try_from(rows).map_err(|_| {
            candle_core::Error::Msg("H3 RMSNorm row count exceeds CUDA grid.x".into())
        })?;
        let width_i32 = i32::try_from(width)
            .map_err(|_| candle_core::Error::Msg("H3 RMSNorm width exceeds i32".into()))?;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(input_layout.start_offset()..);
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let weight = weight.slice(weight_layout.start_offset()..);
        let mut output = unsafe { device.alloc::<half::bf16>(input_layout.shape().elem_count()) }?;
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::H3_RMS_NORM_BF16,
            CUDA_MODULE,
            "h3_rms_norm_bf16",
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        candle_core::builder_arg!(builder, width_i32, self.epsilon);
        builder.arg(&input);
        builder.arg(&weight);
        builder.arg(&mut output);
        let launch = LaunchConfig {
            grid_dim: (rows, 1, 1),
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

pub(crate) fn rms_norm(
    input: &Tensor,
    weight: &Tensor,
    epsilon: f64,
) -> candle_core::Result<Tensor> {
    input.apply_op2_no_bwd(
        weight,
        &H3RmsNormBf16 {
            epsilon: epsilon as f32,
        },
    )
}

pub(crate) fn qwen_rsqrt(input: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op1_no_bwd(&QwenRsqrtF32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_h3_width_contract_is_closed_to_hidden_and_qk_norms() {
        for width in SUPPORTED_H3_WIDTHS {
            validate_h3_width(width).unwrap();
        }
        for width in [0, 4, 64, 256, 5_120, 7_168] {
            assert!(validate_h3_width(width).is_err(), "accepted width {width}");
        }
    }
}
