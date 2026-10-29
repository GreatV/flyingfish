use super::{
    contracts::{Tensor, TensorMut, no_overlap},
    cubin,
};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct DsaTopKInput {
    pub(crate) scores: Tensor<f32, 2>,
    pub(crate) ids: Tensor<i32, 2>,
}
#[derive(Debug)]
pub struct DsaTopKOutput {
    pub(crate) scores: TensorMut<f32, 2>,
    pub(crate) ids: TensorMut<i32, 2>,
    pub(crate) margins: TensorMut<f32, 1>,
}
pub struct DsaSelectionPlan {
    rows: u32,
    candidates: u32,
    keep: u32,
    kernel: CudaFunction,
    grid: LaunchConfig,
}

impl DsaSelectionPlan {
    pub fn new(ctx: &Arc<CudaContext>, rows: u32, candidates: u32, keep: u32) -> Result<Self> {
        ensure!(
            rows > 0
                && rows <= i32::MAX as u32
                && candidates > 0
                && candidates <= i32::MAX as u32
                && keep > 0
                && keep <= 2048,
            "bounded selection extents"
        );
        let module = cubin::module(ctx, "topk")?;
        Ok(Self {
            rows,
            candidates,
            keep,
            kernel: module.load_function(if keep <= 512 {
                "dsa_bounded_topk512"
            } else {
                "dsa_bounded_topk2048"
            })?,
            grid: LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }
    pub fn select_scores(
        &self,
        input: &DsaTopKInput,
        output: &DsaTopKOutput,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        input.scores.validate()?;
        input.ids.validate()?;
        output.scores.read().validate()?;
        output.ids.read().validate()?;
        output.margins.read().validate()?;
        let candidates = self.candidates;
        let keep = self.keep;
        ensure!(
            input.scores.shape == [self.rows, candidates]
                && input.scores.stride == [u64::from(candidates), 1]
                && input.ids.shape == input.scores.shape
                && input.ids.stride == input.scores.stride,
            "DSA candidates must be contiguous [rows,candidates]"
        );
        ensure!(
            output.scores.read().shape == [self.rows, keep]
                && output.scores.read().stride == [u64::from(keep), 1]
                && output.ids.read().shape == output.scores.read().shape
                && output.ids.read().stride == output.scores.read().stride
                && output.margins.read().shape == [self.rows]
                && output.margins.read().stride == [1],
            "DSA selection output geometry"
        );
        no_overlap(
            &input.scores,
            &output.scores.read(),
            "DSA scores",
            "DSA selected scores",
        )?;
        no_overlap(
            &input.ids,
            &output.ids.read(),
            "DSA IDs",
            "DSA selected IDs",
        )?;
        no_overlap(&input.scores, &output.ids.read(), "scores", "selected IDs")?;
        no_overlap(&input.scores, &output.margins.read(), "scores", "margins")?;
        no_overlap(&input.ids, &output.scores.read(), "IDs", "selected scores")?;
        no_overlap(&input.ids, &output.margins.read(), "IDs", "margins")?;
        no_overlap(
            &output.scores.read(),
            &output.ids.read(),
            "selected scores",
            "selected IDs",
        )?;
        no_overlap(
            &output.scores.read(),
            &output.margins.read(),
            "selected scores",
            "margins",
        )?;
        no_overlap(
            &output.ids.read(),
            &output.margins.read(),
            "selected IDs",
            "margins",
        )?;
        let candidates = candidates as i32;
        let keep = keep as i32;
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&input.scores.ptr)
                .arg(&input.ids.ptr)
                .arg(&candidates)
                .arg(&keep)
                .arg(&output.ids.0.ptr)
                .arg(&output.scores.0.ptr)
                .arg(&output.margins.0.ptr)
                .launch(self.grid)?;
        }
        Ok(())
    }
}
