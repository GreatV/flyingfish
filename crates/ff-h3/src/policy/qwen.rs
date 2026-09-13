//! Qwen3-VL conditioning contracts.
//!
//! The text encoder has its own operator-level contracts, admitted vision-grid
//! profiles and compiled-artifact checks, all separate from the H3 denoiser's.

use super::cuda_artifacts::*;
use super::verified::*;
use super::*;

pub const H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION: u32 = 2;

pub const MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES: usize = 64 * 1024;

pub const H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR: &str = "ff_qwen_numerical_contract_json";

pub const H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR: &str = "ff_qwen_numerical_contract_schema";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenRmsNormContract {
    F32NormalizeThenInputDtypeCastAndInputDtypeWeightMultiplyV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenRotaryContract {
    PinnedCpuF32InvFreqBitsDeviceF32PositionMathCosSinCastV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenAttentionScoresContract {
    EagerInputDtypeMatmulThenScaleV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenAttentionMaskContract {
    EagerCausalAddFiniteInputDtypeMinimumV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenAttentionMatmulContract {
    CandleChunkedQkAndPvV1,
    CublasLt130401PytorchScaleText357FullQueryV1,
    CublasLt130401PytorchScaleFl1935Query256Tail143V1,
    CublasLt130401PytorchScaleRef8620Query256Tail172OperatorReferenceV1,
    /// Portable Candle chunked QK/PV on the CUDA runtime.
    ///
    /// Selected when the request is outside every pinned operator profile. The
    /// arithmetic is Candle's, not the transcribed cuBLASLt path, so a result
    /// produced under this contract carries no operator-level parity evidence —
    /// but it executes, and the enclosing policy hash says which path ran.
    CandleChunkedQkAndPvCudaV1,
    UnverifiedQkAndPvBackend,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionLayerNormContract {
    NotApplicableNoVision,
    Pytorch7269437VectorizedBf16CudaRealFlRefV1,
    /// Candle's own composition: F32 promotion, mean, centring, variance,
    /// reciprocal square root, weight and bias in F32, one cast back.
    ///
    /// This is what CPU, Metal and portable CUDA actually evaluate. It carries
    /// no operator-level parity evidence, which is a statement about what has
    /// been measured, not a reason to refuse the request.
    CandleF32CenterVarianceThenInputDtypeCastV1,
    Unverified,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionPositionInterpolationContract {
    NotApplicableNoVision,
    Transformers838763bfCudaF32BilinearRealFlRefV1,
    /// Host-computed bilinear taps and weights, then a Candle gather and F32
    /// weighted sum. Used for every grid the pinned kernel does not cover.
    HostF32BilinearTapsThenCandleGatherWeightedSumV1,
    Unverified,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenSoftmaxContract {
    CandleF32CompositeV1,
    Pytorch7269437Persistent1To2048AndRegular2049To9216CudaV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenSiluContract {
    F32PromoteThenInputDtypeCastV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionBlockGeluContract {
    NotApplicableNoVision,
    CandleTanhV1,
    Pytorch7269437Nvrtc130Bf16TanhRealFlRefV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionMergerGeluContract {
    NotApplicableNoVision,
    CandleF32ErfThenInputDtypeSingleCastV1,
    Pytorch7269437Nvrtc130Bf16ErfRealFlRefV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenBiasedLinearContract {
    CandleLinearV1,
    CublasLt130401Sm89Sm128Driver59584Bf16Compute32BiasEpilogueSixRealFlRefVisionLinearFamiliesV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionPatchProjectionContract {
    NotApplicableNoVision,
    CandleFlattenLinearV1,
    Cudnn92101LegacyImplicitPrecompGemmTensorOpMathEmpiricalV8ParityV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionGridModality {
    Image,
    Video,
}

impl H3QwenVisionGridModality {}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3QwenVisionGridEncodingContract {
    ModalityU8CountU32ThwU64LittleEndianV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenOrderedVisionGrid {
    pub ordinal: u64,
    pub modality: H3QwenVisionGridModality,
    pub temporal: u64,
    pub height: u64,
    pub width: u64,
    pub patch_rows: u64,
    pub merger_rows: u64,
    pub attention_segment_rows: u64,
    pub attention_segment_count: u64,
    pub query_full_chunks: u64,
    pub query_tail_rows: u64,
}

impl H3QwenOrderedVisionGrid {
    fn from_canonical(
        ordinal: usize,
        modality: H3QwenVisionGridModality,
        temporal: usize,
        height: usize,
        width: usize,
        configured_query_rows: NonZeroUsize,
    ) -> Result<Self> {
        anyhow::ensure!(
            temporal > 0 && height > 0 && width > 0,
            "Qwen ordered vision-grid dimensions must be non-zero"
        );
        anyhow::ensure!(
            height.is_multiple_of(2) && width.is_multiple_of(2),
            "Qwen ordered vision grids must be aligned to the released 2x2 merge"
        );
        let segment_rows = height
            .checked_mul(width)
            .context("Qwen vision attention segment rows overflow usize")?;
        let patch_rows = temporal
            .checked_mul(segment_rows)
            .context("Qwen vision patch rows overflow usize")?;
        Ok(Self {
            ordinal: u64::try_from(ordinal).context("Qwen vision-grid ordinal exceeds u64")?,
            modality,
            temporal: u64::try_from(temporal).context("Qwen vision temporal size exceeds u64")?,
            height: u64::try_from(height).context("Qwen vision height exceeds u64")?,
            width: u64::try_from(width).context("Qwen vision width exceeds u64")?,
            patch_rows: u64::try_from(patch_rows).context("Qwen vision patch rows exceed u64")?,
            merger_rows: u64::try_from(patch_rows / 4)
                .context("Qwen vision merger rows exceed u64")?,
            attention_segment_rows: u64::try_from(segment_rows)
                .context("Qwen vision attention segment rows exceed u64")?,
            attention_segment_count: u64::try_from(temporal)
                .context("Qwen vision segment count exceeds u64")?,
            query_full_chunks: u64::try_from(segment_rows / configured_query_rows.get())
                .context("Qwen vision query full-chunk count exceeds u64")?,
            query_tail_rows: u64::try_from(segment_rows % configured_query_rows.get())
                .context("Qwen vision query tail rows exceed u64")?,
        })
    }

    fn validate_derived(&self, ordinal: usize, configured_query_rows: u64) -> Result<()> {
        anyhow::ensure!(
            self.ordinal == u64::try_from(ordinal).context("Qwen grid ordinal exceeds u64")?,
            "Qwen ordered vision-grid ordinals must be contiguous from zero"
        );
        anyhow::ensure!(
            self.temporal > 0 && self.height > 0 && self.width > 0,
            "Qwen ordered vision-grid dimensions must be non-zero"
        );
        anyhow::ensure!(
            self.height.is_multiple_of(2) && self.width.is_multiple_of(2),
            "Qwen ordered vision grids must be aligned to the released 2x2 merge"
        );
        let segment_rows = self
            .height
            .checked_mul(self.width)
            .context("Qwen vision attention segment rows overflow u64")?;
        let patch_rows = self
            .temporal
            .checked_mul(segment_rows)
            .context("Qwen vision patch rows overflow u64")?;
        anyhow::ensure!(
            self.patch_rows == patch_rows
                && self.merger_rows == patch_rows / 4
                && self.attention_segment_rows == segment_rows
                && self.attention_segment_count == self.temporal
                && self.query_full_chunks == segment_rows / configured_query_rows
                && self.query_tail_rows == segment_rows % configured_query_rows,
            "Qwen ordered vision-grid derived geometry is inconsistent"
        );
        for value in [
            self.temporal,
            self.height,
            self.width,
            self.patch_rows,
            self.merger_rows,
            self.attention_segment_rows,
            self.attention_segment_count,
            self.query_full_chunks,
            self.query_tail_rows,
        ] {
            usize::try_from(value).context("Qwen ordered vision-grid value exceeds usize")?;
        }
        Ok(())
    }

    fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.ordinal != other.ordinal {
            Some("qwen.ordered_vision_grids.ordinal")
        } else if self.modality != other.modality {
            Some("qwen.ordered_vision_grids.modality")
        } else if self.temporal != other.temporal {
            Some("qwen.ordered_vision_grids.temporal")
        } else if self.height != other.height {
            Some("qwen.ordered_vision_grids.height")
        } else if self.width != other.width {
            Some("qwen.ordered_vision_grids.width")
        } else if self.patch_rows != other.patch_rows {
            Some("qwen.ordered_vision_grids.patch_rows")
        } else if self.merger_rows != other.merger_rows {
            Some("qwen.ordered_vision_grids.merger_rows")
        } else if self.attention_segment_rows != other.attention_segment_rows {
            Some("qwen.ordered_vision_grids.attention_segment_rows")
        } else if self.attention_segment_count != other.attention_segment_count {
            Some("qwen.ordered_vision_grids.attention_segment_count")
        } else if self.query_full_chunks != other.query_full_chunks {
            Some("qwen.ordered_vision_grids.query_full_chunks")
        } else if self.query_tail_rows != other.query_tail_rows {
            Some("qwen.ordered_vision_grids.query_tail_rows")
        } else {
            None
        }
    }
}

pub(super) const MAX_QWEN_ORDERED_VISION_GRIDS: usize = 12;

/// Whether a `(temporal, height, width)` grid list is one the pinned position
/// kernel was validated against.
///
/// The dispatch in `multimodal_text_encoder` calls this too, so the kernel that
/// runs and the contract that is recorded can never disagree about which grids
/// the exact path covers.
pub fn position_profile_is_verified_for_dimensions(grids: &[[usize; 3]]) -> bool {
    let images = grids.iter().filter(|grid| grid[0] == 1).count();
    let videos = grids.iter().filter(|grid| grid[0] == 7).count();
    !grids.is_empty()
        && images <= 1
        && videos <= 1
        && images + videos == grids.len()
        && grids
            .iter()
            .all(|grid| *grid == [1, 48, 84] || *grid == [7, 48, 84])
}

pub(super) fn qwen_position_profile_is_verified(grids: &[H3QwenOrderedVisionGrid]) -> bool {
    let mut images = grids
        .iter()
        .filter(|grid| grid.modality == H3QwenVisionGridModality::Image);
    let mut videos = grids
        .iter()
        .filter(|grid| grid.modality == H3QwenVisionGridModality::Video);
    let image = images.next();
    let image_extra = images.next();
    let video = videos.next();
    let video_extra = videos.next();
    let valid_image = image_extra.is_none()
        && image.is_none_or(|grid| (grid.temporal, grid.height, grid.width) == (1, 48, 84));
    let valid_video = video_extra.is_none()
        && video.is_none_or(|grid| (grid.temporal, grid.height, grid.width) == (7, 48, 84));
    !grids.is_empty() && valid_image && valid_video
}

pub(super) fn qwen_fl_attention_profile_is_verified(grids: &[H3QwenOrderedVisionGrid]) -> bool {
    matches!(
        grids,
        [H3QwenOrderedVisionGrid {
            modality: H3QwenVisionGridModality::Image,
            temporal: 1,
            height: 48,
            width: 84,
            ..
        }]
    )
}

pub(super) fn qwen_ref_chunk_operator_profile_is_verified(
    grids: &[H3QwenOrderedVisionGrid],
) -> bool {
    matches!(
        grids,
        [H3QwenOrderedVisionGrid {
            modality: H3QwenVisionGridModality::Video,
            temporal: 7,
            height: 48,
            width: 84,
            ..
        }]
    )
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenCudaPatchArtifacts {
    pub cudnn_version: u64,
    pub cudnn_cudart_version: u64,
    pub cudnn_dso_count: u64,
    pub fl_patch_rows: u64,
    pub fl_workspace_bytes: u64,
    pub ref_patch_rows: u64,
    pub ref_workspace_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenVisionLinearGeometry {
    pub image_total_patch_rows: u64,
    pub image_merger_rows: u64,
    pub video_total_patch_rows: u64,
    pub video_merger_rows: u64,
}

impl H3QwenVisionLinearGeometry {
    pub fn from_patch_rows(
        image_total_patch_rows: usize,
        video_total_patch_rows: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            image_total_patch_rows.is_multiple_of(4) && video_total_patch_rows.is_multiple_of(4),
            "Qwen vision patch rows must be divisible by the released 2x2 merger area"
        );
        let geometry = Self {
            image_total_patch_rows: u64::try_from(image_total_patch_rows)
                .context("Qwen image patch rows exceed u64")?,
            image_merger_rows: u64::try_from(image_total_patch_rows / 4)
                .context("Qwen image merger rows exceed u64")?,
            video_total_patch_rows: u64::try_from(video_total_patch_rows)
                .context("Qwen video patch rows exceed u64")?,
            video_merger_rows: u64::try_from(video_total_patch_rows / 4)
                .context("Qwen video merger rows exceed u64")?,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.image_total_patch_rows
                == self
                    .image_merger_rows
                    .checked_mul(4)
                    .context("Qwen image merger-to-patch row derivation overflows u64")?
                && self.video_total_patch_rows
                    == self
                        .video_merger_rows
                        .checked_mul(4)
                        .context("Qwen video merger-to-patch row derivation overflows u64")?,
            "Qwen vision merger rows must be exactly one quarter of patch rows"
        );
        for rows in [
            self.image_total_patch_rows,
            self.image_merger_rows,
            self.video_total_patch_rows,
            self.video_merger_rows,
        ] {
            usize::try_from(rows).context("Qwen vision linear geometry exceeds usize")?;
        }
        Ok(())
    }

    fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.image_total_patch_rows != other.image_total_patch_rows {
            Some("qwen.vision_linear_geometry.image_total_patch_rows")
        } else if self.image_merger_rows != other.image_merger_rows {
            Some("qwen.vision_linear_geometry.image_merger_rows")
        } else if self.video_total_patch_rows != other.video_total_patch_rows {
            Some("qwen.vision_linear_geometry.video_total_patch_rows")
        } else if self.video_merger_rows != other.video_merger_rows {
            Some("qwen.vision_linear_geometry.video_merger_rows")
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenCudaArtifacts {
    pub tensor_runtime: H3CudaTensorRuntimeArtifacts,
    #[serde(deserialize_with = "crate::required_option")]
    pub patch: Option<H3QwenCudaPatchArtifacts>,
    pub persistent_min_width: u64,
    pub persistent_max_width: u64,
    pub regular_min_width: u64,
    pub regular_max_width: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3QwenNumericalContract {
    pub schema_version: u32,
    pub execution_backend: ExecutionBackendPolicy,
    /// Present exactly when the backend is CUDA.
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_capabilities: Option<CudaCapabilities>,
    pub tensor_backend: H3TensorBackendContract,
    pub rms_norm: H3QwenRmsNormContract,
    pub rotary: H3QwenRotaryContract,
    pub attention_scores: H3QwenAttentionScoresContract,
    pub attention_mask: H3QwenAttentionMaskContract,
    pub attention_matmul: H3QwenAttentionMatmulContract,
    pub softmax: H3QwenSoftmaxContract,
    pub silu: H3QwenSiluContract,
    pub vision_block_gelu: H3QwenVisionBlockGeluContract,
    pub vision_merger_gelu: H3QwenVisionMergerGeluContract,
    pub biased_linear: H3QwenBiasedLinearContract,
    pub vision_patch_projection: H3QwenVisionPatchProjectionContract,
    pub vision_layer_norm: H3QwenVisionLayerNormContract,
    pub vision_position_interpolation: H3QwenVisionPositionInterpolationContract,
    pub vision_grid_encoding: H3QwenVisionGridEncodingContract,
    pub ordered_vision_grids: Vec<H3QwenOrderedVisionGrid>,
    pub configured_query_rows: u64,
    pub language_rows: u64,
    pub effective_language_query_rows: u64,
    pub max_vision_segment_rows: u64,
    pub effective_vision_query_rows: u64,
    pub max_attention_width: u64,
    pub vision_linear_geometry: H3QwenVisionLinearGeometry,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_artifacts: Option<H3QwenCudaArtifacts>,
}

impl H3QwenCudaPatchArtifacts {
    fn verified() -> Self {
        Self {
            cudnn_version: VERIFIED_CUDNN_VERSION,
            cudnn_cudart_version: VERIFIED_CUDNN_CUDART_VERSION,
            cudnn_dso_count: VERIFIED_CUDNN_DSO_COUNT,
            fl_patch_rows: VERIFIED_QWEN_PATCH_FL_ROWS,
            fl_workspace_bytes: VERIFIED_QWEN_PATCH_FL_WORKSPACE_BYTES,
            ref_patch_rows: VERIFIED_QWEN_PATCH_REF_ROWS,
            ref_workspace_bytes: VERIFIED_QWEN_PATCH_REF_WORKSPACE_BYTES,
        }
    }

    fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("cuDNN version", self.cudnn_version),
            ("cuDNN CUDART version", self.cudnn_cudart_version),
            ("cuDNN DSO count", self.cudnn_dso_count),
            ("FL patch rows", self.fl_patch_rows),
            ("FL workspace bytes", self.fl_workspace_bytes),
            ("Ref patch rows", self.ref_patch_rows),
            ("Ref workspace bytes", self.ref_workspace_bytes),
        ] {
            anyhow::ensure!(value > 0, "Qwen patch {label} must be non-zero");
        }
        Ok(())
    }

    fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.cudnn_version != other.cudnn_version {
            return Some("qwen.cuda_artifacts.patch.cudnn_version");
        }
        if self.cudnn_cudart_version != other.cudnn_cudart_version {
            return Some("qwen.cuda_artifacts.patch.cudnn_cudart_version");
        }
        if self.cudnn_dso_count != other.cudnn_dso_count {
            return Some("qwen.cuda_artifacts.patch.cudnn_dso_count");
        }
        if self.fl_patch_rows != other.fl_patch_rows {
            return Some("qwen.cuda_artifacts.patch.fl_patch_rows");
        }
        if self.fl_workspace_bytes != other.fl_workspace_bytes {
            return Some("qwen.cuda_artifacts.patch.fl_workspace_bytes");
        }
        if self.ref_patch_rows != other.ref_patch_rows {
            return Some("qwen.cuda_artifacts.patch.ref_patch_rows");
        }
        if self.ref_workspace_bytes != other.ref_workspace_bytes {
            return Some("qwen.cuda_artifacts.patch.ref_workspace_bytes");
        }
        None
    }
}

impl H3QwenCudaArtifacts {
    fn verified(has_vision: bool) -> Self {
        Self {
            tensor_runtime: H3CudaTensorRuntimeArtifacts::verified(),
            patch: has_vision.then(H3QwenCudaPatchArtifacts::verified),
            persistent_min_width: 1,
            persistent_max_width: 2048,
            regular_min_width: 2049,
            regular_max_width: 9216,
        }
    }

    fn validate(&self) -> Result<()> {
        self.tensor_runtime.validate()?;
        if let Some(patch) = &self.patch {
            patch.validate()?;
        }
        anyhow::ensure!(
            (self.persistent_min_width, self.persistent_max_width) == (1, 2048)
                && (self.regular_min_width, self.regular_max_width) == (2049, 9216),
            "CUDA Qwen softmax ranges must be persistent 1..=2048 and regular 2049..=9216"
        );
        Ok(())
    }

    fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if let Some(field) = self.tensor_runtime.first_difference(&other.tensor_runtime) {
            return Some(field.qwen_path());
        }
        match (&self.patch, &other.patch) {
            (Some(left), Some(right)) => {
                if let Some(field) = left.first_difference(right) {
                    return Some(field);
                }
            }
            (None, None) => {}
            _ => return Some("qwen.cuda_artifacts.patch"),
        }
        if self.persistent_min_width != other.persistent_min_width {
            return Some("qwen.cuda_artifacts.persistent_min_width");
        }
        if self.persistent_max_width != other.persistent_max_width {
            return Some("qwen.cuda_artifacts.persistent_max_width");
        }
        if self.regular_min_width != other.regular_min_width {
            return Some("qwen.cuda_artifacts.regular_min_width");
        }
        if self.regular_max_width != other.regular_max_width {
            return Some("qwen.cuda_artifacts.regular_max_width");
        }
        None
    }
}

impl H3QwenNumericalContract {
    pub fn for_verified_target(
        execution_backend: ExecutionBackendPolicy,
        configured_query_rows: NonZeroUsize,
        language_rows: NonZeroUsize,
        max_vision_segment_rows: usize,
        vision_linear_geometry: H3QwenVisionLinearGeometry,
    ) -> Result<Self> {
        anyhow::ensure!(
            max_vision_segment_rows == 0
                && vision_linear_geometry.image_total_patch_rows == 0
                && vision_linear_geometry.video_total_patch_rows == 0,
            "Qwen vision contracts require canonical ordered modality/T/H/W grids; totals alone cannot mint a verified contract"
        );
        Self::for_verified_target_with_grids(
            execution_backend,
            configured_query_rows,
            language_rows,
            max_vision_segment_rows,
            vision_linear_geometry,
            &[],
        )
    }

    /// As [`Self::for_target_with_grids`], modelling a CUDA device that affords
    /// the reference capabilities.
    pub fn for_verified_target_with_grids(
        execution_backend: ExecutionBackendPolicy,
        configured_query_rows: NonZeroUsize,
        language_rows: NonZeroUsize,
        max_vision_segment_rows: usize,
        vision_linear_geometry: H3QwenVisionLinearGeometry,
        canonical_grids: &[(H3QwenVisionGridModality, usize, usize, usize)],
    ) -> Result<Self> {
        Self::for_target_with_grids(
            execution_backend,
            execution_backend
                .is_cuda()
                .then_some(CudaCapabilities::REFERENCE),
            configured_query_rows,
            language_rows,
            max_vision_segment_rows,
            vision_linear_geometry,
            canonical_grids,
        )
    }

    /// The Qwen contract for a target with the capabilities its device affords.
    #[allow(clippy::too_many_arguments)]
    pub fn for_target_with_grids(
        execution_backend: ExecutionBackendPolicy,
        capabilities: Option<CudaCapabilities>,
        configured_query_rows: NonZeroUsize,
        language_rows: NonZeroUsize,
        max_vision_segment_rows: usize,
        vision_linear_geometry: H3QwenVisionLinearGeometry,
        canonical_grids: &[(H3QwenVisionGridModality, usize, usize, usize)],
    ) -> Result<Self> {
        let ordered_vision_grids = canonical_grids
            .iter()
            .enumerate()
            .map(|(ordinal, &(modality, temporal, height, width))| {
                H3QwenOrderedVisionGrid::from_canonical(
                    ordinal,
                    modality,
                    temporal,
                    height,
                    width,
                    configured_query_rows,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (mut image_patch_rows, mut video_patch_rows, mut derived_max_segment) =
            (0_u64, 0_u64, 0_u64);
        for grid in &ordered_vision_grids {
            let total = match grid.modality {
                H3QwenVisionGridModality::Image => &mut image_patch_rows,
                H3QwenVisionGridModality::Video => &mut video_patch_rows,
            };
            *total = total
                .checked_add(grid.patch_rows)
                .context("Qwen ordered vision patch-row total overflows u64")?;
            derived_max_segment = derived_max_segment.max(grid.attention_segment_rows);
        }
        anyhow::ensure!(
            image_patch_rows == vision_linear_geometry.image_total_patch_rows
                && video_patch_rows == vision_linear_geometry.video_total_patch_rows
                && derived_max_segment
                    == u64::try_from(max_vision_segment_rows)
                        .context("Qwen maximum vision segment rows exceed u64")?,
            "Qwen canonical ordered grids disagree with the supplied vision totals/max segment"
        );
        let has_vision = vision_linear_geometry.image_total_patch_rows > 0
            || vision_linear_geometry.video_total_patch_rows > 0;
        anyhow::ensure!(
            has_vision != ordered_vision_grids.is_empty(),
            "Qwen canonical ordered grids must be present exactly when vision rows are present"
        );
        let position_profile_verified = qwen_position_profile_is_verified(&ordered_vision_grids);
        let tuned = capabilities.is_some_and(|capabilities| capabilities.tuned_kernels);
        let reference = capabilities.is_some_and(|capabilities| capabilities.reference_libraries);
        let cuda_artifacts = reference.then(|| H3QwenCudaArtifacts::verified(has_vision));
        let softmax = if tuned {
            H3QwenSoftmaxContract::Pytorch7269437Persistent1To2048AndRegular2049To9216CudaV1
        } else {
            H3QwenSoftmaxContract::CandleF32CompositeV1
        };
        let attention_matmul = match execution_backend {
            ExecutionBackendPolicy::Cpu
            | ExecutionBackendPolicy::Metal => {
                H3QwenAttentionMatmulContract::CandleChunkedQkAndPvV1
            }
            ExecutionBackendPolicy::Cuda
                if reference
                    && !has_vision
                    && language_rows.get() == 357
                    && configured_query_rows.get() == 357 =>
            {
                H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleText357FullQueryV1
            }
            ExecutionBackendPolicy::Cuda
                if reference
                    && language_rows.get() == 1_935
                    && configured_query_rows.get() == 256
                    && qwen_fl_attention_profile_is_verified(&ordered_vision_grids) =>
            {
                H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleFl1935Query256Tail143V1
            }
            ExecutionBackendPolicy::Cuda
                if reference
                    && language_rows.get() == 8_620
                    && configured_query_rows.get() == 256
                    && qwen_ref_chunk_operator_profile_is_verified(&ordered_vision_grids) =>
            {
                H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleRef8620Query256Tail172OperatorReferenceV1
            }
            ExecutionBackendPolicy::Cuda => {
                H3QwenAttentionMatmulContract::CandleChunkedQkAndPvCudaV1
            }
        };
        let contract = Self {
            schema_version: H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION,
            execution_backend,
            cuda_capabilities: capabilities,
            tensor_backend: match execution_backend {
                ExecutionBackendPolicy::Cpu => H3TensorBackendContract::CandleCpu011V1,
                ExecutionBackendPolicy::Cuda => H3TensorBackendContract::CandleCuda011V1,
                ExecutionBackendPolicy::Metal => H3TensorBackendContract::CandleMetal011V1,
            },
            rms_norm:
                H3QwenRmsNormContract::F32NormalizeThenInputDtypeCastAndInputDtypeWeightMultiplyV1,
            rotary: H3QwenRotaryContract::PinnedCpuF32InvFreqBitsDeviceF32PositionMathCosSinCastV1,
            attention_scores: H3QwenAttentionScoresContract::EagerInputDtypeMatmulThenScaleV1,
            attention_mask: H3QwenAttentionMaskContract::EagerCausalAddFiniteInputDtypeMinimumV1,
            attention_matmul,
            softmax,
            silu: H3QwenSiluContract::F32PromoteThenInputDtypeCastV1,
            vision_block_gelu: if !has_vision {
                H3QwenVisionBlockGeluContract::NotApplicableNoVision
            } else {
                if tuned {
                    H3QwenVisionBlockGeluContract::Pytorch7269437Nvrtc130Bf16TanhRealFlRefV1
                } else {
                    H3QwenVisionBlockGeluContract::CandleTanhV1
                }
            },
            vision_merger_gelu: if !has_vision {
                H3QwenVisionMergerGeluContract::NotApplicableNoVision
            } else {
                if tuned {
                    H3QwenVisionMergerGeluContract::Pytorch7269437Nvrtc130Bf16ErfRealFlRefV1
                } else {
                    H3QwenVisionMergerGeluContract::CandleF32ErfThenInputDtypeSingleCastV1
                }
            },
            biased_linear: if reference {
                H3QwenBiasedLinearContract::CublasLt130401Sm89Sm128Driver59584Bf16Compute32BiasEpilogueSixRealFlRefVisionLinearFamiliesV1
            } else {
                H3QwenBiasedLinearContract::CandleLinearV1
            },
            vision_patch_projection: if !has_vision {
                H3QwenVisionPatchProjectionContract::NotApplicableNoVision
            } else {
                if reference {
                    H3QwenVisionPatchProjectionContract::Cudnn92101LegacyImplicitPrecompGemmTensorOpMathEmpiricalV8ParityV1
                } else {
                    H3QwenVisionPatchProjectionContract::CandleFlattenLinearV1
                }
            },
            vision_layer_norm: if has_vision {
                if tuned {
                    H3QwenVisionLayerNormContract::Pytorch7269437VectorizedBf16CudaRealFlRefV1
                } else {
                    H3QwenVisionLayerNormContract::CandleF32CenterVarianceThenInputDtypeCastV1
                }
            } else {
                H3QwenVisionLayerNormContract::NotApplicableNoVision
            },
            vision_position_interpolation: if has_vision {
                if tuned && position_profile_verified {
                    H3QwenVisionPositionInterpolationContract::Transformers838763bfCudaF32BilinearRealFlRefV1
                } else {
                    H3QwenVisionPositionInterpolationContract::HostF32BilinearTapsThenCandleGatherWeightedSumV1
                }
            } else {
                H3QwenVisionPositionInterpolationContract::NotApplicableNoVision
            },
            vision_grid_encoding:
                H3QwenVisionGridEncodingContract::ModalityU8CountU32ThwU64LittleEndianV1,
            ordered_vision_grids,
            configured_query_rows: u64::try_from(configured_query_rows.get())
                .context("configured Qwen query rows exceed u64")?,
            language_rows: u64::try_from(language_rows.get())
                .context("Qwen language rows exceed u64")?,
            effective_language_query_rows: u64::try_from(
                configured_query_rows.get().min(language_rows.get()),
            )
            .context("effective Qwen language query rows exceed u64")?,
            max_vision_segment_rows: u64::try_from(max_vision_segment_rows)
                .context("maximum Qwen vision segment rows exceed u64")?,
            effective_vision_query_rows: u64::try_from(if max_vision_segment_rows == 0 {
                0
            } else {
                configured_query_rows.get().min(max_vision_segment_rows)
            })
            .context("effective Qwen vision query rows exceed u64")?,
            max_attention_width: u64::try_from(language_rows.get().max(max_vision_segment_rows))
                .context("maximum Qwen attention width exceeds u64")?,
            vision_linear_geometry,
            cuda_artifacts,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION,
            "unsupported H3 Qwen numerical-contract schema {}; this build supports schema {}",
            self.schema_version,
            H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.cuda_capabilities.is_some() == self.execution_backend.is_cuda(),
            "Qwen CUDA capabilities must be present exactly for a CUDA backend"
        );
        let tuned = self
            .cuda_capabilities
            .is_some_and(|caps| caps.tuned_kernels);
        let reference = self
            .cuda_capabilities
            .is_some_and(|caps| caps.reference_libraries);
        anyhow::ensure!(
            self.rms_norm
                == H3QwenRmsNormContract::F32NormalizeThenInputDtypeCastAndInputDtypeWeightMultiplyV1
                && self.rotary
                    == H3QwenRotaryContract::PinnedCpuF32InvFreqBitsDeviceF32PositionMathCosSinCastV1
                && self.attention_scores
                    == H3QwenAttentionScoresContract::EagerInputDtypeMatmulThenScaleV1
                && self.attention_mask
                    == H3QwenAttentionMaskContract::EagerCausalAddFiniteInputDtypeMinimumV1
                && self.silu == H3QwenSiluContract::F32PromoteThenInputDtypeCastV1
                && self.tensor_backend
                    == match self.execution_backend {
                        ExecutionBackendPolicy::Cpu => H3TensorBackendContract::CandleCpu011V1,
                        ExecutionBackendPolicy::Cuda => H3TensorBackendContract::CandleCuda011V1,
                        ExecutionBackendPolicy::Metal => {
                            H3TensorBackendContract::CandleMetal011V1
                        }
                    }
                && self.biased_linear
                    == if reference {
                        H3QwenBiasedLinearContract::CublasLt130401Sm89Sm128Driver59584Bf16Compute32BiasEpilogueSixRealFlRefVisionLinearFamiliesV1
                    } else {
                        H3QwenBiasedLinearContract::CandleLinearV1
                    },
            "H3 Qwen numerical contract is incompatible with its recorded capabilities"
        );
        for (name, rows) in [
            ("configured_query_rows", self.configured_query_rows),
            ("language_rows", self.language_rows),
        ] {
            anyhow::ensure!(rows > 0, "Qwen contract {name} must be non-zero");
            usize::try_from(rows).with_context(|| format!("Qwen contract {name} exceeds usize"))?;
        }
        usize::try_from(self.max_vision_segment_rows)
            .context("Qwen contract max_vision_segment_rows exceeds usize")?;
        anyhow::ensure!(
            self.vision_grid_encoding
                == H3QwenVisionGridEncodingContract::ModalityU8CountU32ThwU64LittleEndianV1,
            "unsupported Qwen ordered vision-grid canonical encoding"
        );
        let (mut image_patch_rows, mut video_patch_rows, mut grid_max_segment) =
            (0_u64, 0_u64, 0_u64);
        for (ordinal, grid) in self.ordered_vision_grids.iter().enumerate() {
            grid.validate_derived(ordinal, self.configured_query_rows)?;
            let total = match grid.modality {
                H3QwenVisionGridModality::Image => &mut image_patch_rows,
                H3QwenVisionGridModality::Video => &mut video_patch_rows,
            };
            *total = (*total)
                .checked_add(grid.patch_rows)
                .context("Qwen ordered vision patch-row total overflows u64")?;
            grid_max_segment = grid_max_segment.max(grid.attention_segment_rows);
        }
        anyhow::ensure!(
            self.ordered_vision_grids.len() <= MAX_QWEN_ORDERED_VISION_GRIDS,
            "Qwen ordered vision-grid count exceeds {MAX_QWEN_ORDERED_VISION_GRIDS}"
        );
        self.vision_linear_geometry.validate()?;
        let has_vision = self.vision_linear_geometry.image_total_patch_rows > 0
            || self.vision_linear_geometry.video_total_patch_rows > 0;
        anyhow::ensure!(
            has_vision != self.ordered_vision_grids.is_empty()
                && image_patch_rows == self.vision_linear_geometry.image_total_patch_rows
                && video_patch_rows == self.vision_linear_geometry.video_total_patch_rows
                && grid_max_segment == self.max_vision_segment_rows,
            "Qwen ordered vision grids disagree with bound vision geometry/max segment"
        );
        let expected_attention_matmul = match self.execution_backend {
            ExecutionBackendPolicy::Cpu
            | ExecutionBackendPolicy::Metal => {
                H3QwenAttentionMatmulContract::CandleChunkedQkAndPvV1
            }
            ExecutionBackendPolicy::Cuda
                if reference && !has_vision
                    && self.language_rows == 357
                    && self.configured_query_rows == 357 =>
            {
                H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleText357FullQueryV1
            }
            ExecutionBackendPolicy::Cuda
                if reference && self.language_rows == 1_935
                    && self.configured_query_rows == 256
                    && qwen_fl_attention_profile_is_verified(&self.ordered_vision_grids) =>
            {
                H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleFl1935Query256Tail143V1
            }
            ExecutionBackendPolicy::Cuda
                if reference && self.language_rows == 8_620
                    && self.configured_query_rows == 256
                    && qwen_ref_chunk_operator_profile_is_verified(&self.ordered_vision_grids) =>
            {
                H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleRef8620Query256Tail172OperatorReferenceV1
            }
            ExecutionBackendPolicy::Cuda => {
                H3QwenAttentionMatmulContract::CandleChunkedQkAndPvCudaV1
            }
        };
        anyhow::ensure!(
            self.attention_matmul == expected_attention_matmul,
            "Qwen attention-matmul contract is incompatible with the backend/language/vision geometry"
        );
        let position_profile_verified =
            qwen_position_profile_is_verified(&self.ordered_vision_grids);
        let expected_block_gelu = if !has_vision {
            H3QwenVisionBlockGeluContract::NotApplicableNoVision
        } else {
            match self.execution_backend {
                ExecutionBackendPolicy::Cuda if tuned => {
                    H3QwenVisionBlockGeluContract::Pytorch7269437Nvrtc130Bf16TanhRealFlRefV1
                }
                ExecutionBackendPolicy::Cuda
                | ExecutionBackendPolicy::Cpu
                | ExecutionBackendPolicy::Metal => H3QwenVisionBlockGeluContract::CandleTanhV1,
            }
        };
        let expected_merger_gelu = if !has_vision {
            H3QwenVisionMergerGeluContract::NotApplicableNoVision
        } else {
            match self.execution_backend {
                ExecutionBackendPolicy::Cuda if tuned => {
                    H3QwenVisionMergerGeluContract::Pytorch7269437Nvrtc130Bf16ErfRealFlRefV1
                }
                ExecutionBackendPolicy::Cuda
                | ExecutionBackendPolicy::Cpu
                | ExecutionBackendPolicy::Metal => {
                    H3QwenVisionMergerGeluContract::CandleF32ErfThenInputDtypeSingleCastV1
                }
            }
        };
        anyhow::ensure!(
            self.vision_block_gelu == expected_block_gelu
                && self.vision_merger_gelu == expected_merger_gelu,
            "Qwen vision block/merger GELU contracts are incompatible with the backend/vision geometry"
        );
        let expected_patch = if !has_vision {
            H3QwenVisionPatchProjectionContract::NotApplicableNoVision
        } else {
            match self.execution_backend {
                ExecutionBackendPolicy::Cuda if reference => {
                    H3QwenVisionPatchProjectionContract::Cudnn92101LegacyImplicitPrecompGemmTensorOpMathEmpiricalV8ParityV1
                }
                ExecutionBackendPolicy::Cuda | ExecutionBackendPolicy::Cpu
            | ExecutionBackendPolicy::Metal => {
                    H3QwenVisionPatchProjectionContract::CandleFlattenLinearV1
                }
            }
        };
        anyhow::ensure!(
            self.vision_patch_projection == expected_patch,
            "Qwen vision patch-projection contract is incompatible with the backend/vision geometry"
        );
        anyhow::ensure!(
            self.vision_layer_norm
                == if has_vision {
                    match self.execution_backend {
                        ExecutionBackendPolicy::Cuda if tuned => {
                            H3QwenVisionLayerNormContract::Pytorch7269437VectorizedBf16CudaRealFlRefV1
                        }
                        ExecutionBackendPolicy::Cuda | ExecutionBackendPolicy::Cpu
            | ExecutionBackendPolicy::Metal => {
                            H3QwenVisionLayerNormContract::CandleF32CenterVarianceThenInputDtypeCastV1
                        }
                    }
                } else {
                    H3QwenVisionLayerNormContract::NotApplicableNoVision
                }
                && self.vision_position_interpolation
                    == if has_vision {
                        match self.execution_backend {
                            ExecutionBackendPolicy::Cuda if tuned && position_profile_verified => {
                                H3QwenVisionPositionInterpolationContract::Transformers838763bfCudaF32BilinearRealFlRefV1
                            }
                            ExecutionBackendPolicy::Cpu
                            | ExecutionBackendPolicy::Metal
                            | ExecutionBackendPolicy::Cuda => {
                                H3QwenVisionPositionInterpolationContract::HostF32BilinearTapsThenCandleGatherWeightedSumV1
                            }
                        }
                    } else {
                        H3QwenVisionPositionInterpolationContract::NotApplicableNoVision
                    },
            "Qwen vision LayerNorm/position contracts are incompatible with the vision geometry"
        );
        let expected_language = self.configured_query_rows.min(self.language_rows);
        anyhow::ensure!(
            self.effective_language_query_rows == expected_language,
            "conditioning effective_language_query_rows {} must equal min(configured {}, language {}) = {expected_language}",
            self.effective_language_query_rows,
            self.configured_query_rows,
            self.language_rows
        );
        let expected_vision = if self.max_vision_segment_rows == 0 {
            0
        } else {
            self.configured_query_rows.min(self.max_vision_segment_rows)
        };
        anyhow::ensure!(
            self.effective_vision_query_rows == expected_vision,
            "conditioning effective_vision_query_rows {} must equal {expected_vision}",
            self.effective_vision_query_rows
        );
        anyhow::ensure!(
            self.max_attention_width == self.language_rows.max(self.max_vision_segment_rows),
            "conditioning max_attention_width {} must equal max(language {}, vision {})",
            self.max_attention_width,
            self.language_rows,
            self.max_vision_segment_rows
        );
        let expected_softmax = if tuned {
            H3QwenSoftmaxContract::Pytorch7269437Persistent1To2048AndRegular2049To9216CudaV1
        } else {
            H3QwenSoftmaxContract::CandleF32CompositeV1
        };
        anyhow::ensure!(
            self.softmax == expected_softmax && self.cuda_artifacts.is_some() == reference,
            "H3 Qwen softmax/artifact contract is incompatible with its recorded capabilities"
        );
        if tuned {
            anyhow::ensure!(
                self.language_rows <= 9216 && self.max_vision_segment_rows <= 9216,
                "CUDA Qwen attention width exceeds the verified 1..=9216 range"
            );
        }
        if let Some(artifacts) = &self.cuda_artifacts {
            for (label, patch_rows, merger_rows) in [
                (
                    "image",
                    self.vision_linear_geometry.image_total_patch_rows,
                    self.vision_linear_geometry.image_merger_rows,
                ),
                (
                    "video",
                    self.vision_linear_geometry.video_total_patch_rows,
                    self.vision_linear_geometry.video_merger_rows,
                ),
            ] {
                anyhow::ensure!(
                    (patch_rows, merger_rows) == (0, 0)
                        || matches!((patch_rows, merger_rows), (4032, 1008) | (28224, 7056)),
                    "CUDA Qwen {label} biased-linear geometry ({patch_rows}, {merger_rows}) is outside the verified FL/Ref shape profile"
                );
            }
            anyhow::ensure!(
                artifacts.patch.is_some() == has_vision,
                "CUDA Qwen patch artifacts must be present exactly when vision patch rows are present"
            );
            artifacts.validate()?;
        }
        let bytes =
            serde_json::to_vec(self).context("failed to measure Qwen numerical contract JSON")?;
        anyhow::ensure!(
            bytes.len() <= MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES,
            "Qwen numerical contract JSON exceeds {MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES} bytes"
        );
        Ok(())
    }

    pub fn validate_for(
        &self,
        device: &Device,
        configured_query_rows: NonZeroUsize,
        language_rows: NonZeroUsize,
        max_vision_segment_rows: usize,
        vision_linear_geometry: H3QwenVisionLinearGeometry,
    ) -> Result<()> {
        self.validate()?;
        let canonical_grids = self
            .ordered_vision_grids
            .iter()
            .map(|grid| {
                Ok((
                    grid.modality,
                    usize::try_from(grid.temporal).context("Qwen grid temporal exceeds usize")?,
                    usize::try_from(grid.height).context("Qwen grid height exceeds usize")?,
                    usize::try_from(grid.width).context("Qwen grid width exceeds usize")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        self.validate_for_with_grids(
            device,
            configured_query_rows,
            language_rows,
            max_vision_segment_rows,
            vision_linear_geometry,
            &canonical_grids,
        )
    }

    pub fn validate_for_with_grids(
        &self,
        device: &Device,
        configured_query_rows: NonZeroUsize,
        language_rows: NonZeroUsize,
        max_vision_segment_rows: usize,
        vision_linear_geometry: H3QwenVisionLinearGeometry,
        canonical_grids: &[(H3QwenVisionGridModality, usize, usize, usize)],
    ) -> Result<()> {
        self.validate()?;
        let has_vision = !canonical_grids.is_empty();
        let actual_backend = ExecutionBackendPolicy::from_device(device);
        let actual_capabilities = actual_backend
            .is_cuda()
            .then(|| CudaCapabilities::from_device(device));
        let expected = Self::for_target_with_grids(
            actual_backend,
            actual_capabilities,
            configured_query_rows,
            language_rows,
            max_vision_segment_rows,
            vision_linear_geometry,
            canonical_grids,
        )?;
        if let Some(field) = self.first_difference(&expected) {
            bail!("Qwen numerical contract differs from the selected runtime/request at {field}");
        }
        anyhow::ensure!(
            self.attention_matmul != H3QwenAttentionMatmulContract::UnverifiedQkAndPvBackend,
            "Qwen QK/PV attention matmul backend is unverified and cannot execute"
        );

        anyhow::ensure!(
            self.vision_layer_norm != H3QwenVisionLayerNormContract::Unverified
                && self.vision_position_interpolation
                    != H3QwenVisionPositionInterpolationContract::Unverified,
            "Qwen vision LayerNorm/position interpolation is unverified and cannot execute"
        );
        if actual_capabilities.is_some_and(|capabilities| capabilities.reference_libraries) {
            validate_compiled_qwen_contract(actual_backend, has_vision)?;
            #[cfg(feature = "cuda")]
            if has_vision {
                crate::cuda::profile::validate_qwen_patch_cudnn_preflight()
                    .map_err(anyhow::Error::from)?;
            }
            crate::cuda::profile::validate_exact_profile(device)
                .context("selected CUDA device does not satisfy the Qwen conditioning profile")?;
        }
        Ok(())
    }

    pub fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.schema_version != other.schema_version {
            Some("qwen.schema_version")
        } else if self.execution_backend != other.execution_backend {
            Some("qwen.execution_backend")
        } else if self.cuda_capabilities != other.cuda_capabilities {
            Some("qwen.cuda_capabilities")
        } else if self.tensor_backend != other.tensor_backend {
            Some("qwen.tensor_backend")
        } else if self.rms_norm != other.rms_norm {
            Some("qwen.rms_norm")
        } else if self.rotary != other.rotary {
            Some("qwen.rotary")
        } else if self.attention_scores != other.attention_scores {
            Some("qwen.attention_scores")
        } else if self.attention_mask != other.attention_mask {
            Some("qwen.attention_mask")
        } else if self.softmax != other.softmax {
            Some("qwen.softmax")
        } else if self.silu != other.silu {
            Some("qwen.silu")
        } else if self.vision_block_gelu != other.vision_block_gelu {
            Some("qwen.vision_block_gelu")
        } else if self.vision_merger_gelu != other.vision_merger_gelu {
            Some("qwen.vision_merger_gelu")
        } else if self.biased_linear != other.biased_linear {
            Some("qwen.biased_linear")
        } else if self.vision_patch_projection != other.vision_patch_projection {
            Some("qwen.vision_patch_projection")
        } else if self.vision_layer_norm != other.vision_layer_norm {
            Some("qwen.vision_layer_norm")
        } else if self.configured_query_rows != other.configured_query_rows {
            Some("qwen.configured_query_rows")
        } else if self.vision_grid_encoding != other.vision_grid_encoding {
            Some("qwen.vision_grid_encoding")
        } else if self.ordered_vision_grids.len() != other.ordered_vision_grids.len() {
            Some("qwen.ordered_vision_grids.length")
        } else if let Some(field) = self
            .ordered_vision_grids
            .iter()
            .zip(&other.ordered_vision_grids)
            .find_map(|(left, right)| left.first_difference(right))
        {
            Some(field)
        } else if self.vision_position_interpolation != other.vision_position_interpolation {
            Some("qwen.vision_position_interpolation")
        } else if self.language_rows != other.language_rows {
            Some("qwen.language_rows")
        } else if self.effective_language_query_rows != other.effective_language_query_rows {
            Some("qwen.effective_language_query_rows")
        } else if self.max_vision_segment_rows != other.max_vision_segment_rows {
            Some("qwen.max_vision_segment_rows")
        } else if self.effective_vision_query_rows != other.effective_vision_query_rows {
            Some("qwen.effective_vision_query_rows")
        } else if self.max_attention_width != other.max_attention_width {
            Some("qwen.max_attention_width")
        } else if let Some(field) = self
            .vision_linear_geometry
            .first_difference(&other.vision_linear_geometry)
        {
            Some(field)
        } else if self.attention_matmul != other.attention_matmul {
            Some("qwen.attention_matmul")
        } else {
            match (&self.cuda_artifacts, &other.cuda_artifacts) {
                (Some(left), Some(right)) => left.first_difference(right),
                (None, None) => None,
                _ => Some("qwen.cuda_artifacts"),
            }
        }
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES,
            "Qwen numerical-contract JSON must contain 1..={MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES} bytes"
        );
        let contract: Self =
            serde_json::from_slice(bytes).context("invalid H3 Qwen numerical-contract JSON")?;
        contract.validate()?;
        Ok(contract)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).context("failed to serialize H3 Qwen numerical contract")
    }

    pub fn insert_artifact_tensors(
        &self,
        tensors: &mut HashMap<&'static str, Tensor>,
        device: &Device,
    ) -> Result<()> {
        self.validate()?;
        for name in [
            H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR,
            H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR,
        ] {
            anyhow::ensure!(
                !tensors.contains_key(name),
                "artifact already contains Qwen numerical-contract tensor {name}"
            );
        }
        let json = self.canonical_json()?;
        tensors.insert(
            H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR,
            Tensor::from_vec(json.clone(), json.len(), device)?,
        );
        tensors.insert(
            H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR,
            Tensor::new(H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION, device)?,
        );
        Ok(())
    }

    pub fn take_artifact_tensors(tensors: &mut HashMap<String, Tensor>) -> Result<Option<Self>> {
        let present = [
            H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR,
            H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR,
        ]
        .map(|name| tensors.contains_key(name));
        if present.iter().all(|present| !present) {
            return Ok(None);
        }
        anyhow::ensure!(
            present.iter().all(|present| *present),
            "artifact has incomplete Qwen numerical-contract metadata"
        );
        let json_tensor = tensors
            .remove(H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR)
            .context("artifact is missing Qwen numerical-contract JSON")?;
        anyhow::ensure!(
            json_tensor.dtype() == DType::U8
                && json_tensor.rank() == 1
                && (1..=MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES)
                    .contains(&json_tensor.elem_count()),
            "Qwen numerical-contract JSON tensor must be a bounded non-empty U8 vector"
        );
        let json = json_tensor
            .to_vec1::<u8>()
            .context("Qwen numerical-contract JSON tensor is not a U8 vector")?;
        let schema_tensor = tensors
            .remove(H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR)
            .context("artifact is missing Qwen numerical-contract schema")?;
        anyhow::ensure!(
            schema_tensor.dtype() == DType::U32 && schema_tensor.rank() == 0,
            "Qwen numerical-contract schema tensor must be a U32 scalar"
        );
        let schema = schema_tensor
            .to_scalar::<u32>()
            .context("Qwen numerical-contract schema tensor is not a U32 scalar")?;
        anyhow::ensure!(
            schema == H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION,
            "unsupported artifact Qwen numerical-contract schema {schema}; this build supports schema {H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_VERSION}"
        );
        let contract = Self::from_json(&json)?;
        anyhow::ensure!(
            contract.schema_version == schema,
            "Qwen numerical-contract JSON schema {} disagrees with artifact schema {schema}",
            contract.schema_version
        );
        anyhow::ensure!(
            json == contract.canonical_json()?,
            "artifact Qwen numerical-contract JSON is not canonical"
        );
        Ok(Some(contract))
    }
}

pub(super) fn validate_compiled_qwen_contract(
    execution_backend: ExecutionBackendPolicy,
    has_vision: bool,
) -> Result<()> {
    for (field, actual, expected) in [
        (
            "rms_norm",
            crate::core::QWEN_RMS_NORM_BACKEND,
            VERIFIED_QWEN_RMS_NORM_BACKEND,
        ),
        (
            "rotary",
            crate::core::QWEN_ROPE_BACKEND,
            VERIFIED_QWEN_ROPE_BACKEND,
        ),
        (
            "attention_scores",
            crate::core::QWEN_ATTENTION_SCORE_BACKEND,
            VERIFIED_QWEN_ATTENTION_SCORE_BACKEND,
        ),
        (
            "attention_mask",
            crate::core::QWEN_ATTENTION_MASK_BACKEND,
            VERIFIED_QWEN_ATTENTION_MASK_BACKEND,
        ),
        (
            "silu",
            crate::core::QWEN_SILU_BACKEND,
            VERIFIED_QWEN_SILU_BACKEND,
        ),
    ] {
        anyhow::ensure!(
            actual == expected,
            "compiled H3 Qwen numerical contract mismatch at qwen.{field}"
        );
    }
    if execution_backend == ExecutionBackendPolicy::Cuda {
        anyhow::ensure!(
            crate::core::QWEN_SOFTMAX_BACKEND == VERIFIED_QWEN_SOFTMAX_BACKEND,
            "compiled H3 Qwen numerical contract mismatch at qwen.softmax"
        );
        validate_compiled_qwen_cuda_artifacts(has_vision)?;
        #[cfg(feature = "cuda")]
        if has_vision {
            anyhow::ensure!(
                crate::cuda::qwen::position::BACKEND == VERIFIED_QWEN_VISION_POSITION_BACKEND,
                "compiled H3 Qwen numerical contract mismatch at qwen.vision_position_interpolation"
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(super) fn validate_compiled_qwen_cuda_artifacts(_has_vision: bool) -> Result<()> {
    bail!(
        "cannot execute the verified H3 Qwen CUDA numerical contract: this binary was not compiled with the cuda feature"
    )
}

#[cfg(feature = "cuda")]
pub(super) fn validate_compiled_qwen_cuda_artifacts(has_vision: bool) -> Result<()> {
    validate_compiled_cuda_tensor_runtime()?;
    anyhow::ensure!(
        crate::core::CUDA_EXACT_FULL_SOFTMAX_MAX_KEY_ROWS == 2048
            && crate::cuda::sdpa_softmax::REGULAR_MIN_WIDTH == 2049
            && crate::cuda::sdpa_softmax::REGULAR_MAX_WIDTH == 9216
            && crate::core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS == 9216,
        "compiled H3 Qwen softmax ranges differ from persistent 1..=2048 and regular 2049..=9216"
    );

    if has_vision {
        anyhow::ensure!(
            crate::cuda::profile::CUDNN_VERSION as u64 == VERIFIED_CUDNN_VERSION
                && crate::cuda::profile::CUDNN_CUDART_VERSION as u64
                    == VERIFIED_CUDNN_CUDART_VERSION
                && crate::cuda::profile::QWEN_PATCH_CUDNN_DSO_COUNT as u64
                    == VERIFIED_CUDNN_DSO_COUNT,
            "compiled H3 Qwen numerical contract mismatch at qwen.cuda_artifacts.patch cuDNN identity"
        );
        anyhow::ensure!(
            crate::cuda::qwen::patch::FL_WORKSPACE_BYTES as u64
                == VERIFIED_QWEN_PATCH_FL_WORKSPACE_BYTES
                && crate::cuda::qwen::patch::REF_WORKSPACE_BYTES as u64
                    == VERIFIED_QWEN_PATCH_REF_WORKSPACE_BYTES,
            "compiled H3 Qwen numerical contract mismatch at qwen.cuda_artifacts.patch workspace"
        );
    }
    Ok(())
}

impl PartialEq for H3QwenCudaPatchArtifacts {
    fn eq(&self, other: &Self) -> bool {
        self.first_difference(other).is_none()
    }
}
impl Eq for H3QwenCudaPatchArtifacts {}

impl PartialEq for H3QwenCudaArtifacts {
    fn eq(&self, other: &Self) -> bool {
        self.first_difference(other).is_none()
    }
}
impl Eq for H3QwenCudaArtifacts {}
