//! Official native-math CUDA softmax probe.
//!
//! Both of upstream's kernels are transcribed: the persistent one covers
//! 1..=2048 and the regular register one covers 2049..=9216, which together are
//! the complete exact dispatch. H3's SDPA and Qwen's eager attention share
//! them, so neither range belongs to one caller. Wider full-softmax sequences
//! are rejected before a stage payload loads; they must use the separate
//! FlashAttention execution policy.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Tensor};

const CUDA_MODULE: &str = "flyingfish_h3_sdpa_softmax_pytorch_7269437";

pub(crate) const BACKEND: &str =
    "pytorch-persistent-softmax-f32-1..2048@7269437d655783a26cba32aa88195b741ff496aa/compute_80";

pub(crate) const REGULAR_BACKEND: &str = "pytorch-regular-register-softmax-f32-2049..9216@7269437d655783a26cba32aa88195b741ff496aa/compute_80";
pub(crate) const REGULAR_MIN_WIDTH: usize = 2049;
pub(crate) const REGULAR_MAX_WIDTH: usize = 9216;

#[derive(Debug, Clone, Copy)]
struct H3SdpaSoftmaxF32;

#[derive(Debug, Clone, Copy)]
struct RegularSoftmaxF32;

impl CustomOp1 for H3SdpaSoftmaxF32 {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("the official H3 SDPA persistent softmax probe is CUDA-only")
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
            candle_core::bail!("official H3 SDPA persistent softmax requires F32 input")
        }
        if !layout.is_contiguous() {
            candle_core::bail!("official H3 SDPA persistent softmax requires contiguous input")
        }
        let Some(&width) = layout.dims().last() else {
            candle_core::bail!("official H3 SDPA persistent softmax does not support scalars")
        };
        if !(1..=crate::core::CUDA_EXACT_FULL_SOFTMAX_MAX_KEY_ROWS).contains(&width) {
            candle_core::bail!(
                "official H3 SDPA persistent softmax requires width 1..={}, got {width}",
                crate::core::CUDA_EXACT_FULL_SOFTMAX_MAX_KEY_ROWS
            )
        }
        let elements = layout.shape().elem_count();
        if elements == 0 {
            candle_core::bail!("official H3 SDPA persistent softmax does not support empty input")
        }
        if elements > (1usize << 30) {
            candle_core::bail!(
                "official H3 SDPA persistent softmax limits one launch to 2^30 elements, got {elements}"
            )
        }
        let rows = elements / width;
        let rows_i32 = i32::try_from(rows)
            .map_err(|_| candle_core::Error::Msg("H3 SDPA row count exceeds i32".into()))?;
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| candle_core::Error::Msg("H3 SDPA row count exceeds u32".into()))?;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<f32>()?;
        let input = input.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(elements) }?;
        let log2 = usize::BITS - (width - 1).leading_zeros();
        let function_name = format!("h3_sdpa_softmax_f32_log2_{log2}");
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::H3_SDPA_SOFTMAX_F32,
            CUDA_MODULE,
            &function_name,
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        builder.arg(&input);
        builder.arg(&mut output);
        candle_core::builder_arg!(builder, rows_i32, width as i32);
        let next_power_of_two = 1u32 << log2;
        let warp_size = next_power_of_two.min(32);
        let batches_per_warp = if next_power_of_two <= 128 { 2 } else { 1 };
        let warps_per_block = 128 / warp_size;
        let rows_per_block = warps_per_block * batches_per_warp;
        let launch = LaunchConfig {
            grid_dim: (rows_u32.div_ceil(rows_per_block), 1, 1),
            block_dim: (warp_size, warps_per_block, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(launch) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

impl CustomOp1 for RegularSoftmaxF32 {
    fn name(&self) -> &'static str {
        REGULAR_BACKEND
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("the official Qwen regular softmax is CUDA-only")
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
            candle_core::bail!("official Qwen regular softmax requires F32 input")
        }
        if !layout.is_contiguous() {
            candle_core::bail!("official Qwen regular softmax requires contiguous input")
        }
        let Some(&width) = layout.dims().last() else {
            candle_core::bail!("official Qwen regular softmax does not support scalars")
        };
        if !(REGULAR_MIN_WIDTH..=REGULAR_MAX_WIDTH).contains(&width) {
            candle_core::bail!(
                "official Qwen regular softmax requires width {REGULAR_MIN_WIDTH}..={REGULAR_MAX_WIDTH}, got {width}"
            )
        }
        let elements = layout.shape().elem_count();
        if elements == 0 {
            candle_core::bail!("official Qwen regular softmax does not support empty input")
        }
        let rows = elements / width;
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| candle_core::Error::Msg("Qwen softmax row count exceeds u32".into()))?;
        let width_i64 = i64::try_from(width)
            .map_err(|_| candle_core::Error::Msg("Qwen softmax width exceeds i64".into()))?;
        let register_count = width.div_ceil(1024);
        if !(3..=9).contains(&register_count) {
            candle_core::bail!(
                "Qwen regular softmax register count must be 3..=9, got {register_count}"
            )
        }
        let device = input.device().clone();
        let input = input.as_cuda_slice::<f32>()?;
        let input = input.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(elements) }?;
        let function_name = format!("qwen_softmax_regular_f32_reg_{register_count}");
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::H3_SDPA_SOFTMAX_F32,
            CUDA_MODULE,
            &function_name,
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        builder.arg(&input);
        builder.arg(&mut output);
        candle_core::builder_arg!(builder, width_i64);
        let launch = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 32 * std::mem::size_of::<f32>() as u32,
        };
        unsafe { builder.launch(launch) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn softmax(input: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op1_no_bwd(&H3SdpaSoftmaxF32)
}

pub(crate) fn regular_softmax(input: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op1_no_bwd(&RegularSoftmaxF32)
}
