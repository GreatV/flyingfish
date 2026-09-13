//! Numerical implementation descriptors.
//!
//! Each constant names one exact backend, library, PTX or crate the recorded
//! evidence covers. Changing a value here changes what the tree claims to have
//! verified, so they are kept together rather than spread across the contracts
//! that consume them.

pub(super) const VERIFIED_CUDNN_VERSION: u64 = 92_101;

pub(super) const VERIFIED_CUDNN_CUDART_VERSION: u64 = 13_020;

pub(super) const VERIFIED_CUDNN_DSO_COUNT: u64 = 8;

pub(super) const VERIFIED_QWEN_PATCH_FL_ROWS: u64 = 4_032;

pub(super) const VERIFIED_QWEN_PATCH_FL_WORKSPACE_BYTES: u64 = 42_467_344;

pub(super) const VERIFIED_QWEN_PATCH_REF_ROWS: u64 = 28_224;

pub(super) const VERIFIED_QWEN_PATCH_REF_WORKSPACE_BYTES: u64 = 240_648_208;

pub(super) const VERIFIED_QWEN_RMS_NORM_BACKEND: &str =
    "qwen3-vl-eager-composite-rmsnorm-f32-rsqrt-cast-input-dtype-weight-mul-v1";

pub(super) const VERIFIED_QWEN_ROPE_BACKEND: &str =
    "qwen3-vl-pinned-cpu-f32-invfreq-bits-device-f32-position-math-cos-sin-cast-v1";

pub(super) const VERIFIED_QWEN_ATTENTION_SCORE_BACKEND: &str =
    "qwen3-vl-eager-input-dtype-matmul-then-scale-v1";

pub(super) const VERIFIED_QWEN_ATTENTION_MASK_BACKEND: &str =
    "qwen3-vl-eager-causal-add-finfo-input-dtype-min-v1";

pub(super) const VERIFIED_QWEN_SILU_BACKEND: &str = "qwen3-vl-f32-silu-single-input-dtype-cast-v1";

pub(super) const VERIFIED_QWEN_SOFTMAX_BACKEND: &str =
    "qwen3-vl-eager-f32-persistent-1..2048+regular-register-2049..9216-v1";

#[cfg(feature = "cuda")]
pub(super) const VERIFIED_QWEN_VISION_POSITION_BACKEND: &str = "transformers838763bf-qwen3vl-host-stepwise-f32-bitexact-to-cuda-f32-bilinear-align-corners-taps/candle011-u32-gather-f32-weighted-sum4-bf16-cast/real-grid-profile:image1x48x84|video7x48x84-v1";

pub(super) const VERIFIED_H3_OUTPUT_HEAD_ORDER_BACKEND: &str =
    "both-heads-on-contiguous-full-packed-chunks-then-modality-select-v1";

pub(super) const VERIFIED_CANDLE_KERNELS_PTX_COUNT: u64 = 11;

#[cfg(feature = "cuda")]
pub(super) const VERIFIED_CUDA_NVCC_VERSION: &str = "13.2.86";

#[cfg(feature = "cuda")]
pub(super) const VERIFIED_CANDLE_KERNELS_BUILD_FLAGS: &str =
    "cudaforge-build-ptx:--expt-relaxed-constexpr,-std=c++17,-O3;ug-disabled";

pub(super) const VERIFIED_CUBLAS_LIBRARY_BASENAME: &str = "libcublas.so.13.4.1.3";

pub(super) const VERIFIED_CUBLAS_LIBRARY_BYTES: u64 = 54_198_912;

pub(super) const VERIFIED_CUBLAS_LT_LIBRARY_BASENAME: &str = "libcublasLt.so.13.4.1.3";

pub(super) const VERIFIED_CUBLAS_LT_LIBRARY_BYTES: u64 = 508_260_928;
