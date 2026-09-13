//! Exact Qwen vision patch Conv3d for the verified real FL/Ref profiles.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, Layout, Shape, Tensor};
use cudarc::cudnn::sys::{cudnnConvolutionFwdAlgo_t, cudnnMathType_t};

pub(crate) const BACKEND: &str = "cudnn92101-legacy-implicit-precomp-gemm-tensor-op-math-bf16-conv3d-no-bias-then-bf16-add/empirically-bit-exact-pinned-pytorch-v8-operands/shape-profile:patch4032-ws42467344|28224-ws240648208-3x2x16x16-to1152-v1";
pub(crate) const FL_WORKSPACE_BYTES: usize = 42_467_344;
pub(crate) const REF_WORKSPACE_BYTES: usize = 240_648_208;
pub(crate) const FL_PATCH_ROWS: usize = 4_032;
pub(crate) const REF_PATCH_ROWS: usize = 28_224;

#[derive(Debug, Clone, Copy)]
struct QwenPatchConv3dBf16 {
    algorithm: cudnnConvolutionFwdAlgo_t,
    math_type: cudnnMathType_t,
    enforce_exact_contract: bool,
}

impl CustomOp2 for QwenPatchConv3dBf16 {
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
        candle_core::bail!("exact Qwen patch Conv3d is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use cudarc::cudnn::safe::{ConvForward, Cudnn};
        use cudarc::cudnn::sys::{cudnnConvolutionMode_t, cudnnTensorFormat_t};

        crate::cuda::profile::validate_cuda_device(input.device())?;
        crate::cuda::profile::validate_qwen_patch_cudnn_preflight()?;
        if input.dtype() != candle_core::DType::BF16 || weight.dtype() != candle_core::DType::BF16 {
            candle_core::bail!("exact Qwen patch Conv3d requires BF16 input and weight")
        }
        if !input_layout.is_contiguous() || !weight_layout.is_contiguous() {
            candle_core::bail!("exact Qwen patch Conv3d requires contiguous tensors")
        }
        if input_layout.start_offset() != 0 || weight_layout.start_offset() != 0 {
            candle_core::bail!("Qwen patch Conv3d requires zero storage offsets")
        }
        let [rows, width] = input_layout.dims() else {
            candle_core::bail!("Qwen patch Conv3d input must be [patches, 1536]")
        };
        if *width != 1536 || !matches!(*rows, FL_PATCH_ROWS | REF_PATCH_ROWS) {
            candle_core::bail!(
                "unverified Qwen patch Conv3d input shape {:?}; expected [4032|28224,1536]",
                input_layout.dims()
            )
        }
        if weight_layout.dims() != [1152, 3, 2, 16, 16] {
            candle_core::bail!(
                "unverified Qwen patch Conv3d weight shape {:?}",
                weight_layout.dims()
            )
        }
        let rows_i32 = i32::try_from(*rows)
            .map_err(|_| candle_core::Error::Msg("Qwen patch rows exceed i32".into()))?;
        let device = input.device().clone();
        let stream = device.cuda_stream();
        let cudnn = Cudnn::new(stream.clone()).map_err(|error| {
            candle_core::Error::Msg(format!("failed to create cuDNN: {error:?}"))
        })?;
        let input_desc = cudnn
            .create_nd_tensor::<half::bf16>(&[rows_i32, 3, 2, 16, 16], &[1536, 512, 256, 16, 1])
            .map_err(|error| {
                candle_core::Error::Msg(format!("cuDNN input descriptor: {error:?}"))
            })?;
        let weight_desc = cudnn
            .create_nd_filter::<half::bf16>(
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                &[1152, 3, 2, 16, 16],
            )
            .map_err(|error| {
                candle_core::Error::Msg(format!("cuDNN filter descriptor: {error:?}"))
            })?;
        let output_desc = cudnn
            .create_nd_tensor::<half::bf16>(&[rows_i32, 1152, 1, 1, 1], &[1152, 1, 1, 1, 1])
            .map_err(|error| {
                candle_core::Error::Msg(format!("cuDNN output descriptor: {error:?}"))
            })?;
        let mut conv = cudnn
            .create_convnd::<f32>(
                &[0, 0, 0],
                &[2, 16, 16],
                &[1, 1, 1],
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(|error| {
                candle_core::Error::Msg(format!("cuDNN convolution descriptor: {error:?}"))
            })?;
        conv.set_math_type(self.math_type)
            .map_err(|error| candle_core::Error::Msg(format!("cuDNN math type: {error:?}")))?;
        let operation = ConvForward {
            conv: &conv,
            x: &input_desc,
            w: &weight_desc,
            y: &output_desc,
        };
        let algorithm = self.algorithm;
        let workspace_bytes = operation.get_workspace_size(algorithm).map_err(|error| {
            candle_core::Error::Msg(format!("cuDNN workspace query: {error:?}"))
        })?;
        if self.enforce_exact_contract {
            let expected_workspace = match *rows {
                FL_PATCH_ROWS => FL_WORKSPACE_BYTES,
                REF_PATCH_ROWS => REF_WORKSPACE_BYTES,
                _ => unreachable!("shape guard fixed the supported row counts"),
            };
            if algorithm
                != cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                || self.math_type != cudnnMathType_t::CUDNN_TENSOR_OP_MATH
                || workspace_bytes != expected_workspace
            {
                candle_core::bail!(
                    "Qwen patch exact cuDNN tuple changed: algorithm={algorithm:?}, math={:?}, workspace={workspace_bytes}, expected workspace={expected_workspace}",
                    self.math_type
                )
            }
        } else if workspace_bytes > 4 * 1024 * 1024 * 1024usize {
            candle_core::bail!(
                "cuDNN diagnostic workspace {workspace_bytes} exceeds the 4 GiB bounded probe"
            )
        }
        if !self.enforce_exact_contract {
            eprintln!(
                "qwen_patch_candidate rows={rows} algorithm={algorithm:?} workspace_bytes={workspace_bytes}"
            );
        }
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(input_layout.start_offset()..);
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let weight = weight.slice(weight_layout.start_offset()..);
        let output_elements = rows.checked_mul(1152).ok_or_else(|| {
            candle_core::Error::Msg("Qwen patch output element count overflow".into())
        })?;
        let mut output = unsafe { device.alloc::<half::bf16>(output_elements) }?;
        let mut workspace = if workspace_bytes == 0 {
            None
        } else {
            Some(unsafe { device.alloc::<u8>(workspace_bytes) }?)
        };
        unsafe {
            operation.launch(
                algorithm,
                workspace.as_mut(),
                (half::bf16::from_f32(1.0), half::bf16::from_f32(0.0)),
                &input,
                &weight,
                &mut output,
            )
        }
        .map_err(|error| candle_core::Error::Msg(format!("cuDNN Conv3d launch: {error:?}")))?;
        crate::cuda::profile::validate_qwen_patch_loaded_cudnn()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            Shape::from_dims(&[*rows, 1152]),
        ))
    }
}

pub(crate) fn projection(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> candle_core::Result<Tensor> {
    if bias.dtype() != candle_core::DType::BF16 || bias.dims() != [1152] {
        candle_core::bail!("Qwen patch projection requires BF16 bias with shape [1152]")
    }
    if !bias.device().same_device(input.device()) {
        candle_core::bail!("Qwen patch projection bias is on a different device")
    }
    if !bias.is_contiguous() || bias.layout().start_offset() != 0 {
        candle_core::bail!("Qwen patch projection requires contiguous zero-offset BF16 bias")
    }
    let convolved = input.apply_op2_no_bwd(
        weight,
        &QwenPatchConv3dBf16 {
            algorithm: cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            math_type: cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
            enforce_exact_contract: true,
        },
    )?;
    convolved.broadcast_add(bias)
}
