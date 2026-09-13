//! Exact CUDA BF16 GELU for the two distinct Qwen3-VL vision call families.
//!
//! Released vision blocks select `gelu_pytorch_tanh` from the model config;
//! patch mergers construct `nn.GELU()` and select the erf path. Unsupported
//! operation/shape pairs fail instead of sharing an approximate fallback.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Tensor};

const CUDA_MODULE: &str = "flyingfish_qwen_gelu_pytorch_7269437";
pub(crate) const TANH_BACKEND: &str = concat!(
    "pytorch-gelu-tanh-bf16-cuda@7269437d655783a26cba32aa88195b741ff496aa/",
    "source-sha256:d896d28a6f0762f2b0bc64f2aaef5cefa86182344ec28017deb3a0962d41e6cc/",
    "nvrtc-",
    env!("FLYINGFISH_QWEN_GELU_NVRTC_VERSION"),
    "/nvrtc-sha256:",
    env!("FLYINGFISH_QWEN_GELU_NVRTC_SHA256"),
    "/builtins-sha256:",
    env!("FLYINGFISH_QWEN_GELU_NVRTC_BUILTINS_SHA256"),
    "/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
    "/shape-profile:4032|28224x4304-v1"
);
pub(crate) const ERF_BACKEND: &str = concat!(
    "pytorch-gelu-erf-bf16-cuda@7269437d655783a26cba32aa88195b741ff496aa/",
    "source-sha256:d896d28a6f0762f2b0bc64f2aaef5cefa86182344ec28017deb3a0962d41e6cc/",
    "nvrtc-",
    env!("FLYINGFISH_QWEN_GELU_NVRTC_VERSION"),
    "/nvrtc-sha256:",
    env!("FLYINGFISH_QWEN_GELU_NVRTC_SHA256"),
    "/builtins-sha256:",
    env!("FLYINGFISH_QWEN_GELU_NVRTC_BUILTINS_SHA256"),
    "/",
    env!("FLYINGFISH_H3_RMS_NORM_PTX_ARCH"),
    "/shape-profile:1008|7056x4608-v1"
);

#[derive(Clone, Copy, Debug)]
enum GeluKind {
    Tanh,
    Erf,
}

#[derive(Clone, Copy, Debug)]
struct QwenGeluBf16 {
    kind: GeluKind,
}

impl CustomOp1 for QwenGeluBf16 {
    fn name(&self) -> &'static str {
        match self.kind {
            GeluKind::Tanh => TANH_BACKEND,
            GeluKind::Erf => ERF_BACKEND,
        }
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("exact Qwen vision GELU is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        crate::cuda::device::require_tuned_kernel(input.device())?;
        if input.dtype() != candle_core::DType::BF16 {
            candle_core::bail!("exact Qwen vision GELU requires BF16 input")
        }
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!("exact Qwen vision GELU requires a zero-offset contiguous input")
        }
        let [rows, width] = layout.dims() else {
            candle_core::bail!("exact Qwen vision GELU input must be rank two")
        };
        let supported = match self.kind {
            GeluKind::Tanh => matches!((*rows, *width), (4_032 | 28_224, 4_304)),
            GeluKind::Erf => matches!((*rows, *width), (1_008 | 7_056, 4_608)),
        };
        if !supported {
            candle_core::bail!(
                "unverified Qwen vision {:?} GELU shape {:?}",
                self.kind,
                layout.dims()
            )
        }
        let elements = rows.checked_mul(*width).ok_or_else(|| {
            candle_core::Error::Msg("Qwen GELU element count overflows usize".into())
        })?;
        let elements_i32 = i32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("Qwen GELU elements exceed i32".into()))?;
        let elements_u32 = u32::try_from(elements)
            .map_err(|_| candle_core::Error::Msg("Qwen GELU elements exceed u32".into()))?;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<half::bf16>()?;
        if elements > input.len() {
            candle_core::bail!(
                "Qwen GELU requires {elements} elements but storage has {}",
                input.len()
            )
        }
        let mut output = unsafe { device.alloc::<half::bf16>(elements) }?;
        let function_name = match self.kind {
            GeluKind::Tanh => "qwen_gelu_tanh_bf16",
            GeluKind::Erf => "qwen_gelu_erf_bf16",
        };
        let function = crate::cuda::kernel_assets::load_function(
            &device,
            &crate::cuda::kernel_assets::QWEN_GELU_BF16,
            CUDA_MODULE,
            function_name,
        )?;
        let stream = device.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        candle_core::builder_arg!(builder, elements_i32);
        builder.arg(input);
        builder.arg(&mut output);
        unsafe { builder.launch(LaunchConfig::for_num_elems(elements_u32)) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn vision_block_gelu_tanh(input: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op1_no_bwd(&QwenGeluBf16 {
        kind: GeluKind::Tanh,
    })
}

pub(crate) fn merger_gelu_erf(input: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op1_no_bwd(&QwenGeluBf16 {
        kind: GeluKind::Erf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn cuda_exact_api_never_falls_back_on_cpu() {
        let block = Tensor::zeros((4_032, 4_304), DType::BF16, &Device::Cpu).unwrap();
        let merger = Tensor::zeros((1_008, 4_608), DType::BF16, &Device::Cpu).unwrap();
        assert!(
            vision_block_gelu_tanh(&block)
                .unwrap_err()
                .to_string()
                .contains("CUDA-only")
        );
        assert!(
            merger_gelu_erf(&merger)
                .unwrap_err()
                .to_string()
                .contains("CUDA-only")
        );
    }
}
