//! CUDA tensor-runtime and attention artifacts.
//!
//! Records algorithm, architecture, precision and library contracts. Legacy
//! digest strings remain readable annotations, not execution gates; new
//! contracts leave them empty. They are preserved when reading old metadata
//! so its data-integrity encoding can still round-trip.

use super::verified::*;
use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaPtxArchitectureContract {
    Compute80,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaCandleUgContract {
    Disabled,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaCandleKernelsContract {
    CandleKernels011CudaforgeNvcc13286O3Sm89PtxManifestV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaClassicCublasContract {
    Cublas130401,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3CudaTensorRuntimeArtifacts {
    pub candle_ug: CudaCandleUgContract,
    pub candle_kernels: CudaCandleKernelsContract,
    pub candle_kernels_ptx_count: u64,
    pub candle_gemm_reduced_precision_f32: bool,
    pub candle_gemm_reduced_precision_f16: bool,
    pub candle_gemm_reduced_precision_bf16: bool,
    pub classic_cublas: CudaClassicCublasContract,
    pub cublas_library_basename: String,
    pub cublas_library_bytes: u64,
    pub cublas_lt_library_basename: String,
    pub cublas_lt_library_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum H3CudaTensorRuntimeDifference {
    CandleUg,
    CandleKernels,
    CandleKernelsPtx,
    CandleGemmReducedPrecisionF32,
    CandleGemmReducedPrecisionF16,
    CandleGemmReducedPrecisionBf16,
    ClassicCublas,
    CublasLibrary,
    CublasLtLibrary,
}

impl H3CudaTensorRuntimeDifference {
    fn h3_path(self) -> &'static str {
        match self {
            Self::CandleUg => "numerics.cuda_artifacts.tensor_runtime.candle_ug",
            Self::CandleKernels => "numerics.cuda_artifacts.tensor_runtime.candle_kernels",
            Self::CandleKernelsPtx => {
                "numerics.cuda_artifacts.tensor_runtime.candle_kernels_ptx_count"
            }
            Self::CandleGemmReducedPrecisionF32 => {
                "numerics.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_f32"
            }
            Self::CandleGemmReducedPrecisionF16 => {
                "numerics.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_f16"
            }
            Self::CandleGemmReducedPrecisionBf16 => {
                "numerics.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_bf16"
            }
            Self::ClassicCublas => "numerics.cuda_artifacts.tensor_runtime.classic_cublas",
            Self::CublasLibrary => "numerics.cuda_artifacts.tensor_runtime.cublas_library",
            Self::CublasLtLibrary => "numerics.cuda_artifacts.tensor_runtime.cublas_lt_library",
        }
    }

    pub(super) fn qwen_path(self) -> &'static str {
        match self {
            Self::CandleUg => "qwen.cuda_artifacts.tensor_runtime.candle_ug",
            Self::CandleKernels => "qwen.cuda_artifacts.tensor_runtime.candle_kernels",
            Self::CandleKernelsPtx => "qwen.cuda_artifacts.tensor_runtime.candle_kernels_ptx_count",
            Self::CandleGemmReducedPrecisionF32 => {
                "qwen.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_f32"
            }
            Self::CandleGemmReducedPrecisionF16 => {
                "qwen.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_f16"
            }
            Self::CandleGemmReducedPrecisionBf16 => {
                "qwen.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_bf16"
            }
            Self::ClassicCublas => "qwen.cuda_artifacts.tensor_runtime.classic_cublas",
            Self::CublasLibrary => "qwen.cuda_artifacts.tensor_runtime.cublas_library",
            Self::CublasLtLibrary => "qwen.cuda_artifacts.tensor_runtime.cublas_lt_library",
        }
    }
}

impl H3CudaTensorRuntimeArtifacts {
    pub(super) fn verified() -> Self {
        Self {
            candle_ug: CudaCandleUgContract::Disabled,
            candle_kernels:
                CudaCandleKernelsContract::CandleKernels011CudaforgeNvcc13286O3Sm89PtxManifestV1,
            candle_kernels_ptx_count: VERIFIED_CANDLE_KERNELS_PTX_COUNT,
            candle_gemm_reduced_precision_f32: false,
            candle_gemm_reduced_precision_f16: false,
            candle_gemm_reduced_precision_bf16: false,
            classic_cublas: CudaClassicCublasContract::Cublas130401,
            cublas_library_basename: VERIFIED_CUBLAS_LIBRARY_BASENAME.to_owned(),
            cublas_library_bytes: VERIFIED_CUBLAS_LIBRARY_BYTES,
            cublas_lt_library_basename: VERIFIED_CUBLAS_LT_LIBRARY_BASENAME.to_owned(),
            cublas_lt_library_bytes: VERIFIED_CUBLAS_LT_LIBRARY_BYTES,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        for (label, basename, bytes) in [
            (
                "classic cuBLAS library",
                self.cublas_library_basename.as_str(),
                self.cublas_library_bytes,
            ),
            (
                "cuBLASLt library",
                self.cublas_lt_library_basename.as_str(),
                self.cublas_lt_library_bytes,
            ),
        ] {
            anyhow::ensure!(
                !basename.is_empty()
                    && basename.len() <= 128
                    && !basename.contains('/')
                    && !basename.contains('\\')
                    && basename.bytes().all(|byte| byte.is_ascii_graphic()),
                "{label} basename is invalid"
            );
            anyhow::ensure!(bytes > 0, "{label} byte size must be non-zero");
        }
        anyhow::ensure!(
            self.candle_kernels_ptx_count > 0,
            "Candle kernels embedded PTX count must be non-zero"
        );
        anyhow::ensure!(
            !self.candle_gemm_reduced_precision_f32
                && !self.candle_gemm_reduced_precision_f16
                && !self.candle_gemm_reduced_precision_bf16,
            "CUDA tensor runtime requires Candle GEMM reduced-precision f32/f16/bf16 flags to remain false"
        );
        Ok(())
    }

    pub(super) fn first_difference(&self, other: &Self) -> Option<H3CudaTensorRuntimeDifference> {
        if self.candle_ug != other.candle_ug {
            Some(H3CudaTensorRuntimeDifference::CandleUg)
        } else if self.candle_kernels != other.candle_kernels {
            Some(H3CudaTensorRuntimeDifference::CandleKernels)
        } else if self.candle_kernels_ptx_count != other.candle_kernels_ptx_count {
            Some(H3CudaTensorRuntimeDifference::CandleKernelsPtx)
        } else if self.candle_gemm_reduced_precision_f32 != other.candle_gemm_reduced_precision_f32
        {
            Some(H3CudaTensorRuntimeDifference::CandleGemmReducedPrecisionF32)
        } else if self.candle_gemm_reduced_precision_f16 != other.candle_gemm_reduced_precision_f16
        {
            Some(H3CudaTensorRuntimeDifference::CandleGemmReducedPrecisionF16)
        } else if self.candle_gemm_reduced_precision_bf16
            != other.candle_gemm_reduced_precision_bf16
        {
            Some(H3CudaTensorRuntimeDifference::CandleGemmReducedPrecisionBf16)
        } else if self.classic_cublas != other.classic_cublas {
            Some(H3CudaTensorRuntimeDifference::ClassicCublas)
        } else if (&self.cublas_library_basename, self.cublas_library_bytes)
            != (&other.cublas_library_basename, other.cublas_library_bytes)
        {
            Some(H3CudaTensorRuntimeDifference::CublasLibrary)
        } else if (
            &self.cublas_lt_library_basename,
            self.cublas_lt_library_bytes,
        ) != (
            &other.cublas_lt_library_basename,
            other.cublas_lt_library_bytes,
        ) {
            Some(H3CudaTensorRuntimeDifference::CublasLtLibrary)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "implementation", rename_all = "snake_case", deny_unknown_fields)]
pub enum H3CudaAttentionArtifacts {
    PytorchNativeMathPersistentSoftmax,
    OnlineSoftmaxWithPersistentTokenRefiner,
    CandleFlashAttention011,
}

/// Identities of the kernels compiled from this repository's own sources.
///
/// Host-independent: one `compute_80` PTX per kernel serves every device, so
/// these digests are the same wherever the kernels run. What differs per device
/// is only whether that PTX was translated ahead of time by `ptxas` or at load
/// time by the driver, which does not change the instructions.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3TunedKernelArtifacts {
    pub ptx_architecture: CudaPtxArchitectureContract,
}

/// Identities that only mean something on a host carrying the reference
/// cuBLASLt and cuDNN builds, plus the operators that call into them.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3ReferenceLibraryArtifacts {
    pub tensor_runtime: H3CudaTensorRuntimeArtifacts,
}

/// Build artifacts of a CUDA H3 policy, one section per capability axis.
///
/// Reference-library and tuned-kernel identities are optional independently.
/// Attention identities also remain present when FlashAttention runs without
/// either of those capabilities.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3CudaNumericalArtifacts {
    #[serde(deserialize_with = "crate::required_option")]
    pub tuned_kernels: Option<H3TunedKernelArtifacts>,
    #[serde(deserialize_with = "crate::required_option")]
    pub reference_libraries: Option<H3ReferenceLibraryArtifacts>,
    /// FlashAttention is compiled independently of the reference vendor libraries.
    /// Portable full/online attention uses Candle and has no extra artifact here.
    #[serde(deserialize_with = "crate::required_option")]
    pub attention: Option<H3CudaAttentionArtifacts>,
}

impl H3CudaNumericalArtifacts {
    pub(super) fn validate(&self) -> Result<()> {
        if let Some(reference) = &self.reference_libraries {
            reference.tensor_runtime.validate()?;
        }
        Ok(())
    }

    pub(super) fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self
            .tuned_kernels
            .as_ref()
            .map(|value| value.ptx_architecture)
            != other
                .tuned_kernels
                .as_ref()
                .map(|value| value.ptx_architecture)
        {
            return Some("numerics.cuda_artifacts.tuned_kernels.ptx_architecture");
        }
        match (&self.reference_libraries, &other.reference_libraries) {
            (Some(left), Some(right)) => {
                if let Some(field) = left.tensor_runtime.first_difference(&right.tensor_runtime) {
                    return Some(field.h3_path());
                }
            }
            (None, None) => {}
            _ => return Some("numerics.cuda_artifacts.reference_libraries"),
        }
        match (&self.attention, &other.attention) {
            (None, None) => None,
            (Some(left), Some(right))
                if std::mem::discriminant(left) == std::mem::discriminant(right) =>
            {
                None
            }
            (Some(_), Some(_)) => Some("numerics.cuda_artifacts.attention.implementation"),
            _ => Some("numerics.cuda_artifacts.attention"),
        }
    }
}

/// The artifacts a CUDA run records, given what its device afforded.
pub(super) fn cuda_artifacts_for(
    capabilities: CudaCapabilities,
    attention_backend: AttentionBackendPolicy,
) -> H3CudaNumericalArtifacts {
    let attention = match attention_backend {
        AttentionBackendPolicy::FullSoftmax => {
            H3CudaAttentionArtifacts::PytorchNativeMathPersistentSoftmax
        }
        AttentionBackendPolicy::OnlineSoftmax => {
            H3CudaAttentionArtifacts::OnlineSoftmaxWithPersistentTokenRefiner
        }
        AttentionBackendPolicy::FlashAttention => H3CudaAttentionArtifacts::CandleFlashAttention011,
    };
    H3CudaNumericalArtifacts {
        tuned_kernels: capabilities
            .tuned_kernels
            .then_some(H3TunedKernelArtifacts {
                ptx_architecture: CudaPtxArchitectureContract::Compute80,
            }),
        reference_libraries: capabilities.reference_libraries.then(|| {
            H3ReferenceLibraryArtifacts {
                tensor_runtime: H3CudaTensorRuntimeArtifacts::verified(),
            }
        }),
        attention: (capabilities.reference_libraries
            || attention_backend == AttentionBackendPolicy::FlashAttention)
            .then_some(attention),
    }
}

#[cfg(feature = "cuda")]
pub(super) fn validate_compiled_cuda_tensor_runtime() -> Result<()> {
    for (field, actual, expected) in [
        (
            "nvcc",
            crate::cuda::profile::NVCC_VERSION,
            VERIFIED_CUDA_NVCC_VERSION,
        ),
        (
            "candle_kernels_build_flags",
            crate::cuda::profile::CANDLE_KERNELS_BUILD_FLAGS,
            VERIFIED_CANDLE_KERNELS_BUILD_FLAGS,
        ),
        (
            "cublas_library_basename",
            crate::cuda::profile::CUBLAS_LIBRARY_BASENAME,
            VERIFIED_CUBLAS_LIBRARY_BASENAME,
        ),
        (
            "cublas_lt_library_basename",
            crate::cuda::profile::CUBLASLT_LIBRARY_BASENAME,
            VERIFIED_CUBLAS_LT_LIBRARY_BASENAME,
        ),
    ] {
        anyhow::ensure!(
            actual == expected,
            "compiled CUDA tensor runtime mismatch at tensor_runtime.{field}"
        );
    }
    anyhow::ensure!(
        crate::cuda::profile::CUBLAS_VERSION == 130_401
            && crate::cuda::profile::CUBLASLT_VERSION == 130_401
            && crate::cuda::profile::CUBLAS_LIBRARY_BYTES == VERIFIED_CUBLAS_LIBRARY_BYTES
            && crate::cuda::profile::CUBLASLT_LIBRARY_BYTES == VERIFIED_CUBLAS_LT_LIBRARY_BYTES,
        "compiled CUDA tensor runtime version or library-size contract changed"
    );
    anyhow::ensure!(
        !crate::cuda::profile::CANDLE_GEMM_REDUCED_PRECISION_F32
            && !crate::cuda::profile::CANDLE_GEMM_REDUCED_PRECISION_F16
            && !crate::cuda::profile::CANDLE_GEMM_REDUCED_PRECISION_BF16,
        "compiled CUDA tensor runtime Candle GEMM reduced-precision contract changed"
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(super) fn validate_compiled_cuda_contract(
    _capabilities: CudaCapabilities,
    _attention_backend: AttentionBackendPolicy,
) -> Result<()> {
    bail!(
        "cannot execute the verified H3 CUDA numerical contract: this binary was not compiled with the cuda feature"
    )
}

#[cfg(feature = "cuda")]
pub(super) fn validate_compiled_cuda_contract(
    capabilities: CudaCapabilities,
    attention_backend: AttentionBackendPolicy,
) -> Result<()> {
    if capabilities.reference_libraries {
        validate_compiled_cuda_tensor_runtime()?;
    }
    if attention_backend == AttentionBackendPolicy::FlashAttention {
        #[cfg(not(feature = "flash-attn"))]
        bail!(
            "cannot execute the H3 CUDA FlashAttention numerical contract: this binary was not compiled with the flash-attn feature"
        );
    }
    Ok(())
}

impl PartialEq for H3CudaTensorRuntimeArtifacts {
    fn eq(&self, other: &Self) -> bool {
        self.first_difference(other).is_none()
    }
}
impl Eq for H3CudaTensorRuntimeArtifacts {}

impl PartialEq for H3CudaNumericalArtifacts {
    fn eq(&self, other: &Self) -> bool {
        self.first_difference(other).is_none()
    }
}
impl Eq for H3CudaNumericalArtifacts {}
