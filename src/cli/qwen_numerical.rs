use anyhow::{Context, Result};
use candle_core::Device;
use flyingfish::h3::policy::{
    CudaCapabilities, ExecutionBackendPolicy, H3QwenNumericalContract, H3QwenVisionLinearGeometry,
};
use std::num::NonZeroUsize;

pub(crate) fn build_qwen_numerical_contract(
    device: &Device,
    configured_query_rows: usize,
    language_rows: usize,
    image_total_patch_rows: usize,
    video_total_patch_rows: usize,
    max_vision_segment_rows: usize,
) -> Result<H3QwenNumericalContract> {
    let configured_query_rows = NonZeroUsize::new(configured_query_rows)
        .context("Qwen configured query rows must be non-zero")?;
    let language_rows =
        NonZeroUsize::new(language_rows).context("Qwen language rows must be non-zero")?;
    let geometry = H3QwenVisionLinearGeometry::from_patch_rows(
        image_total_patch_rows,
        video_total_patch_rows,
    )?;
    let backend = ExecutionBackendPolicy::from_device(device);
    let contract = H3QwenNumericalContract::for_target_with_grids(
        backend,
        backend
            .is_cuda()
            .then(|| CudaCapabilities::from_device(device)),
        configured_query_rows,
        language_rows,
        max_vision_segment_rows,
        geometry,
        &[],
    )?;
    contract.validate_for(
        device,
        configured_query_rows,
        language_rows,
        max_vision_segment_rows,
        geometry,
    )?;
    Ok(contract)
}

pub(crate) fn validate_qwen_numerical_contract(
    contract: &H3QwenNumericalContract,
    device: &Device,
    language_rows: usize,
    image_total_patch_rows: usize,
    video_total_patch_rows: usize,
    max_vision_segment_rows: usize,
) -> Result<()> {
    contract.validate_for(
        device,
        NonZeroUsize::new(
            usize::try_from(contract.configured_query_rows)
                .context("recorded Qwen configured query rows exceed usize")?,
        )
        .context("recorded Qwen configured query rows must be non-zero")?,
        NonZeroUsize::new(language_rows).context("Qwen language rows must be non-zero")?,
        max_vision_segment_rows,
        H3QwenVisionLinearGeometry::from_patch_rows(
            image_total_patch_rows,
            video_total_patch_rows,
        )?,
    )
}
