//! Single deterministic MoE combine over per-(row, choice) contributions.
use crate::backend::cuda::contracts::{Tensor, TensorMut};
use crate::backend::cuda::cubin;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, PushKernelArg};
use half::bf16;
use std::sync::Arc;

use crate::backend::cuda::ops::grid;

pub struct CombineOps {
    shared_scale: CudaFunction,
    combine: CudaFunction,
}

pub struct MoeCombine {
    /// `[rows, top_k]` router expert ids (dense).
    pub ids: Tensor<i32, 2>,
    /// `[rows, top_k]` the final active set this combine completes (dense).
    pub final_active: Tensor<u32, 2>,
    /// `[rows, top_k]` written by the contribute stages (dense).
    pub done: Tensor<u32, 2>,
    /// `[rows, top_k, hidden]` weighted BF16 contributions (dense).
    pub contributions: Tensor<bf16, 3>,
    /// `[rows, hidden]` shared-expert output (dense).
    pub shared_out: Tensor<bf16, 2>,
    /// `[rows, hidden]` token rows (dense).
    pub x: Tensor<bf16, 2>,
    /// `[hidden]` shared-expert sigmoid gate vector.
    pub shared_expert_gate: Tensor<bf16, 1>,
    /// `[rows]` sigmoid scales (written by shared_scale).
    pub scale: TensorMut<bf16, 2>,
    /// `[rows, hidden]` output.
    pub y: TensorMut<bf16, 2>,
}

impl MoeCombine {
    fn validate(&self) -> Result<(u32, u32, u32)> {
        let (rows, top_k) = (self.ids.shape[0], self.ids.shape[1]);
        self.ids.validate()?;
        ensure!(
            self.ids.shape == [rows, top_k] && self.ids.stride == [u64::from(top_k), 1],
            "moe combine ids must be dense [rows, top_k] (got {}x{})",
            self.ids.shape[0],
            self.ids.shape[1]
        );
        for t in [&self.final_active, &self.done] {
            t.validate()?;
            ensure!(
                t.shape == [rows, top_k] && t.stride == [u64::from(top_k), 1],
                "moe combine final_active/done must be dense [rows, top_k] (got {}x{})",
                t.shape[0],
                t.shape[1]
            );
        }
        let hidden = self.y.read().shape[1];
        self.y.read().validate()?;
        ensure!(
            self.y.read().shape == [rows, hidden] && self.y.read().stride == [u64::from(hidden), 1],
            "moe combine y must be dense [rows, hidden] (got {}x{})",
            self.y.read().shape[0],
            self.y.read().shape[1]
        );
        self.contributions.validate()?;
        ensure!(
            self.contributions.shape == [rows, top_k, hidden]
                && self.contributions.stride[2] == 1
                && self.contributions.stride[1] == u64::from(hidden)
                && self.contributions.stride[0] == u64::from(top_k) * u64::from(hidden),
            "moe combine contributions must be dense [rows, top_k, hidden]"
        );
        for (t, name) in [(&self.shared_out, "shared_out"), (&self.x, "x")] {
            t.validate()?;
            ensure!(
                t.shape == [rows, hidden] && t.stride == [u64::from(hidden), 1],
                "moe combine {name} must be dense [rows, hidden]"
            );
        }
        self.scale.read().validate()?;
        ensure!(
            self.scale.read().shape == [rows, 1] && self.scale.read().stride == [1, 1],
            "moe combine scale must be a dense [rows, 1] buffer"
        );
        self.shared_expert_gate.validate()?;
        ensure!(
            self.shared_expert_gate.shape == [hidden] && self.shared_expert_gate.stride == [1],
            "moe combine shared gate must be dense [hidden]"
        );
        ensure!(
            (1..=32).contains(&top_k),
            "moe combine top_k {top_k} outside [1,32]"
        );
        Ok((rows, top_k, hidden))
    }
}

impl CombineOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_combine")?;
        Ok(Self {
            shared_scale: module.load_function("moe_shared_scale")?,
            combine: module.load_function("moe_combine")?,
        })
    }

    pub fn shared_scale(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        shared_expert_gate: Tensor<bf16, 1>,
        scale: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let rows = x.shape[0];
        let hidden = x.shape[1];
        x.validate()?;
        shared_expert_gate.validate()?;
        scale.read().validate()?;
        ensure!(
            x.stride[1] == 1
                && shared_expert_gate.shape == [hidden]
                && shared_expert_gate.stride == [1]
                && scale.read().shape == [rows, 1]
                && scale.read().stride == [1, 1],
            "moe combine shared_scale layout mismatch (rows {rows}, hidden {hidden})"
        );
        // SAFETY: the validated row, gate and scale views cover every launched row and hidden value.
        unsafe {
            stream
                .launch_builder(&self.shared_scale)
                .arg(&x.ptr)
                .arg(&shared_expert_gate.ptr)
                .arg(&scale.0.ptr)
                .arg(&(rows as i32))
                .arg(&(hidden as i64))
                .arg(&(x.stride[0] as i64))
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }

    pub fn combine(&self, stream: &Arc<CudaStream>, input: &MoeCombine) -> Result<()> {
        let (rows, top_k, hidden) = input.validate()?;
        // SAFETY: all validated views cover rows*top_k choices and rows*hidden output values.
        unsafe {
            stream
                .launch_builder(&self.combine)
                .arg(&input.contributions.ptr)
                .arg(&input.shared_out.ptr)
                .arg(&input.scale.read().ptr)
                .arg(&input.ids.ptr)
                .arg(&input.done.ptr)
                .arg(&input.final_active.ptr)
                .arg(&input.y.0.ptr)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i64))
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }
}
