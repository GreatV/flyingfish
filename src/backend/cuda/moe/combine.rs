//! Single deterministic MoE combine over per-(row, choice) contributions.
use crate::backend::cuda::contracts::{Tensor, TensorMut, no_overlap};
use crate::backend::cuda::cubin;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, PushKernelArg};
use half::bf16;
use std::sync::Arc;

use crate::backend::cuda::ops::grid;

pub struct CombineOps {
    shared_scale: CudaFunction,
    combine: CudaFunction,
    combine_ungated: CudaFunction,
    combine_routed: CudaFunction,
    combine_routed_scalar: CudaFunction,
    join_routed: CudaFunction,
    combine_residual: CudaFunction,
    combine_residual_scalar: CudaFunction,
    join_routed_residual: CudaFunction,
    join_routed_residual_scalar: CudaFunction,
}

pub struct MoeCombineRouted {
    pub ids: Tensor<i32, 2>,
    pub final_active: Tensor<u32, 2>,
    pub done: Tensor<u32, 2>,
    pub contributions: Tensor<bf16, 3>,
    pub routed_out: TensorMut<bf16, 2>,
}

pub struct MoeJoinRouted {
    pub routed_in: Tensor<bf16, 2>,
    pub shared_out: Tensor<bf16, 2>,
    pub scale: Option<Tensor<bf16, 2>>,
    pub y: TensorMut<bf16, 2>,
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
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    /// `[rows]` sigmoid scales (written by shared_scale).
    pub scale: Option<TensorMut<bf16, 2>>,
    /// `[rows, hidden]` output.
    pub y: TensorMut<bf16, 2>,
}

/// The fused combine: the routed sum, the shared-expert sigmoid gate and the
/// residual add in one launch. The residual is read and written in place.
pub struct MoeCombineResidual {
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
    /// `[rows, hidden]` rows the sigmoid gate dot product reads (dense).
    pub x: Tensor<bf16, 2>,
    /// `[hidden]` shared-expert sigmoid gate vector; absent is the ungated form.
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    /// `[rows, hidden]` residual stream, updated in place.
    pub residual: TensorMut<bf16, 2>,
}

/// The fused peer join: the routed sum comes from the peer GPU, the
/// shared-expert sigmoid gate and the residual add are local. The residual is
/// read and written in place.
pub struct MoeJoinRoutedResidual {
    /// `[rows, hidden]` routed sum produced by the peer (dense).
    pub routed_in: Tensor<bf16, 2>,
    /// `[rows, hidden]` shared-expert output (dense).
    pub shared_out: Tensor<bf16, 2>,
    /// `[rows, hidden]` rows the sigmoid gate dot product reads (dense).
    pub x: Tensor<bf16, 2>,
    /// `[hidden]` shared-expert sigmoid gate vector; absent is the ungated form.
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    /// `[rows, hidden]` residual stream, updated in place.
    pub residual: TensorMut<bf16, 2>,
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
        match (&self.shared_expert_gate, &self.scale) {
            (Some(gate), Some(scale)) => {
                scale.read().validate()?;
                ensure!(
                    scale.read().shape == [rows, 1] && scale.read().stride == [1, 1],
                    "moe combine scale must be a dense [rows, 1] buffer"
                );
                gate.validate()?;
                ensure!(
                    gate.shape == [hidden] && gate.stride == [1],
                    "moe combine shared gate must be dense [hidden]"
                );
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!("moe combine has partial shared gate/scale inputs")
            }
        }
        ensure!(
            (1..=32).contains(&top_k),
            "moe combine top_k {top_k} outside [1,32]"
        );
        Ok((rows, top_k, hidden))
    }
}

impl MoeCombineResidual {
    fn validate(&self) -> Result<(u32, u32, u32)> {
        let (rows, top_k) = (self.ids.shape[0], self.ids.shape[1]);
        self.ids.validate()?;
        ensure!(
            self.ids.shape == [rows, top_k] && self.ids.stride == [u64::from(top_k), 1],
            "moe combine_residual ids must be dense [rows, top_k] (got {}x{})",
            self.ids.shape[0],
            self.ids.shape[1]
        );
        for t in [&self.final_active, &self.done] {
            t.validate()?;
            ensure!(
                t.shape == [rows, top_k] && t.stride == [u64::from(top_k), 1],
                "moe combine_residual final_active/done must be dense [rows, top_k] (got {}x{})",
                t.shape[0],
                t.shape[1]
            );
        }
        let hidden = self.residual.read().shape[1];
        self.residual.read().validate()?;
        ensure!(
            self.residual.read().shape == [rows, hidden]
                && self.residual.read().stride == [u64::from(hidden), 1],
            "moe combine_residual residual must be dense [rows, hidden] (got {}x{})",
            self.residual.read().shape[0],
            self.residual.read().shape[1]
        );
        self.contributions.validate()?;
        ensure!(
            self.contributions.shape == [rows, top_k, hidden]
                && self.contributions.stride[2] == 1
                && self.contributions.stride[1] == u64::from(hidden)
                && self.contributions.stride[0] == u64::from(top_k) * u64::from(hidden),
            "moe combine_residual contributions must be dense [rows, top_k, hidden]"
        );
        for (t, name) in [(&self.shared_out, "shared_out"), (&self.x, "x")] {
            t.validate()?;
            ensure!(
                t.shape == [rows, hidden] && t.stride == [u64::from(hidden), 1],
                "moe combine_residual {name} must be dense [rows, hidden]"
            );
        }
        if let Some(gate) = &self.shared_expert_gate {
            gate.validate()?;
            ensure!(
                gate.shape == [hidden] && gate.stride == [1],
                "moe combine_residual shared gate must be dense [hidden]"
            );
        }
        ensure!(
            (1..=32).contains(&top_k),
            "moe combine_residual top_k {top_k} outside [1,32]"
        );
        no_overlap(
            &self.residual.read(),
            &self.contributions,
            "residual",
            "contributions",
        )?;
        no_overlap(
            &self.residual.read(),
            &self.shared_out,
            "residual",
            "shared_out",
        )?;
        no_overlap(&self.residual.read(), &self.x, "residual", "x")?;
        Ok((rows, top_k, hidden))
    }
}

impl MoeJoinRoutedResidual {
    fn validate(&self) -> Result<(u32, u32)> {
        let (rows, hidden) = (self.residual.read().shape[0], self.residual.read().shape[1]);
        self.residual.read().validate()?;
        ensure!(
            self.residual.read().shape == [rows, hidden]
                && self.residual.read().stride == [u64::from(hidden), 1],
            "moe join_routed_residual residual must be dense [rows, hidden] (got {}x{})",
            self.residual.read().shape[0],
            self.residual.read().shape[1]
        );
        for (t, name) in [
            (&self.routed_in, "routed_in"),
            (&self.shared_out, "shared_out"),
            (&self.x, "x"),
        ] {
            t.validate()?;
            ensure!(
                t.shape == [rows, hidden] && t.stride == [u64::from(hidden), 1],
                "moe join_routed_residual {name} must be dense [rows, hidden] (got {}x{})",
                t.shape[0],
                t.shape[1]
            );
        }
        if let Some(gate) = &self.shared_expert_gate {
            gate.validate()?;
            ensure!(
                gate.shape == [hidden] && gate.stride == [1],
                "moe join_routed_residual shared gate must be dense [hidden]"
            );
        }
        no_overlap(
            &self.residual.read(),
            &self.routed_in,
            "residual",
            "routed_in",
        )?;
        no_overlap(
            &self.residual.read(),
            &self.shared_out,
            "residual",
            "shared_out",
        )?;
        no_overlap(&self.residual.read(), &self.x, "residual", "x")?;
        Ok((rows, hidden))
    }
}

impl MoeCombineRouted {
    fn validate(&self) -> Result<(u32, u32, u32)> {
        let (rows, top_k) = (self.ids.shape[0], self.ids.shape[1]);
        self.ids.validate()?;
        ensure!(
            self.ids.shape == [rows, top_k] && self.ids.stride == [u64::from(top_k), 1],
            "moe combine_routed ids must be dense [rows, top_k] (got {}x{})",
            self.ids.shape[0],
            self.ids.shape[1]
        );
        for t in [&self.final_active, &self.done] {
            t.validate()?;
            ensure!(
                t.shape == [rows, top_k] && t.stride == [u64::from(top_k), 1],
                "moe combine_routed final_active/done must be dense [rows, top_k] (got {}x{})",
                t.shape[0],
                t.shape[1]
            );
        }
        let hidden = self.routed_out.read().shape[1];
        self.routed_out.read().validate()?;
        ensure!(
            self.routed_out.read().shape == [rows, hidden]
                && self.routed_out.read().stride == [u64::from(hidden), 1],
            "moe combine_routed routed_out must be dense [rows, hidden] (got {}x{})",
            self.routed_out.read().shape[0],
            self.routed_out.read().shape[1]
        );
        self.contributions.validate()?;
        ensure!(
            self.contributions.shape == [rows, top_k, hidden]
                && self.contributions.stride[2] == 1
                && self.contributions.stride[1] == u64::from(hidden)
                && self.contributions.stride[0] == u64::from(top_k) * u64::from(hidden),
            "moe combine_routed contributions must be dense [rows, top_k, hidden]"
        );
        ensure!(
            (1..=32).contains(&top_k),
            "moe combine_routed top_k {top_k} outside [1,32]"
        );
        Ok((rows, top_k, hidden))
    }
}

impl MoeJoinRouted {
    fn validate(&self) -> Result<(u32, u32)> {
        let (rows, hidden) = (self.routed_in.shape[0], self.routed_in.shape[1]);
        for (t, name) in [
            (&self.routed_in, "routed_in"),
            (&self.shared_out, "shared_out"),
            (&self.y.read(), "y"),
        ] {
            t.validate()?;
            ensure!(
                t.shape == [rows, hidden] && t.stride == [u64::from(hidden), 1],
                "moe join_routed {name} must be dense [rows, hidden] (got {}x{})",
                t.shape[0],
                t.shape[1]
            );
        }
        if let Some(scale) = &self.scale {
            scale.validate()?;
            ensure!(
                scale.shape == [rows, 1] && scale.stride == [1, 1],
                "moe join_routed scale must be a dense [rows, 1] buffer"
            );
        }
        Ok((rows, hidden))
    }
}

impl CombineOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_combine")?;
        Ok(Self {
            shared_scale: module.load_function("moe_shared_scale")?,
            combine: module.load_function("moe_combine")?,
            combine_ungated: module.load_function("moe_combine_ungated")?,
            combine_routed: module.load_function("moe_combine_routed")?,
            combine_routed_scalar: module.load_function("moe_combine_routed_scalar")?,
            join_routed: module.load_function("moe_join_routed")?,
            combine_residual: module.load_function("moe_combine_residual")?,
            combine_residual_scalar: module.load_function("moe_combine_residual_scalar")?,
            join_routed_residual: module.load_function("moe_join_routed_residual")?,
            join_routed_residual_scalar: module.load_function("moe_join_routed_residual_scalar")?,
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
        if let (Some(_), Some(scale)) = (&input.shared_expert_gate, &input.scale) {
            unsafe {
                stream
                    .launch_builder(&self.combine)
                    .arg(&input.contributions.ptr)
                    .arg(&input.shared_out.ptr)
                    .arg(&scale.read().ptr)
                    .arg(&input.ids.ptr)
                    .arg(&input.done.ptr)
                    .arg(&input.final_active.ptr)
                    .arg(&input.y.0.ptr)
                    .arg(&(rows as i32))
                    .arg(&(top_k as i32))
                    .arg(&(hidden as i64))
                    .launch(grid(rows as usize, 256))?;
            }
        } else {
            ensure!(
                input.shared_expert_gate.is_none() && input.scale.is_none(),
                "moe ungated combine has partial gate/scale inputs"
            );
            unsafe {
                stream
                    .launch_builder(&self.combine_ungated)
                    .arg(&input.contributions.ptr)
                    .arg(&input.shared_out.ptr)
                    .arg(&input.ids.ptr)
                    .arg(&input.done.ptr)
                    .arg(&input.final_active.ptr)
                    .arg(&input.y.0.ptr)
                    .arg(&(rows as i32))
                    .arg(&(top_k as i32))
                    .arg(&(hidden as i64))
                    .launch(grid(rows as usize, 256))?;
            }
        }
        Ok(())
    }

    pub fn combine_routed(&self, stream: &Arc<CudaStream>, input: &MoeCombineRouted) -> Result<()> {
        let (rows, top_k, hidden) = input.validate()?;
        // The 16-byte group path needs hidden % 8 == 0; the scalar entry point
        // covers every other extent.
        let kernel = if hidden % 8 == 0 {
            &self.combine_routed
        } else {
            &self.combine_routed_scalar
        };
        // SAFETY: all validated views cover rows*top_k choices and rows*hidden output values.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&input.contributions.ptr)
                .arg(&input.ids.ptr)
                .arg(&input.done.ptr)
                .arg(&input.final_active.ptr)
                .arg(&input.routed_out.0.ptr)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i64))
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }

    /// The fused combine: the routed sum, the shared-expert sigmoid gate and
    /// the residual add in one launch. The residual is updated in place.
    pub fn combine_residual(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeCombineResidual,
    ) -> Result<()> {
        let (rows, top_k, hidden) = input.validate()?;
        // The 16-byte group path needs hidden % 8 == 0; the scalar entry point
        // covers every other extent.
        let kernel = if hidden % 8 == 0 {
            &self.combine_residual
        } else {
            &self.combine_residual_scalar
        };
        let seg = input
            .shared_expert_gate
            .as_ref()
            .map_or(0u64, |gate| gate.ptr);
        let x_stride = input.x.stride[0];
        // SAFETY: the validated views cover rows*top_k choices, rows*hidden shared
        // and residual values and the hidden gate vector.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&input.contributions.ptr)
                .arg(&input.shared_out.ptr)
                .arg(&seg)
                .arg(&input.x.ptr)
                .arg(&(x_stride as i64))
                .arg(&input.ids.ptr)
                .arg(&input.done.ptr)
                .arg(&input.final_active.ptr)
                .arg(&input.residual.0.ptr)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i64))
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }

    /// The fused peer join: the routed sum comes from the peer GPU, the
    /// shared-expert sigmoid gate and the residual add are local. The residual
    /// is updated in place.
    pub fn join_routed_residual(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeJoinRoutedResidual,
    ) -> Result<()> {
        let (rows, hidden) = input.validate()?;
        // The 16-byte group path needs hidden % 8 == 0; the scalar entry point
        // covers every other extent.
        let kernel = if hidden % 8 == 0 {
            &self.join_routed_residual
        } else {
            &self.join_routed_residual_scalar
        };
        let seg = input
            .shared_expert_gate
            .as_ref()
            .map_or(0u64, |gate| gate.ptr);
        let x_stride = input.x.stride[0];
        // SAFETY: the validated views cover rows*hidden routed, shared and
        // residual values and the hidden gate vector.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&input.routed_in.ptr)
                .arg(&input.shared_out.ptr)
                .arg(&seg)
                .arg(&input.x.ptr)
                .arg(&(x_stride as i64))
                .arg(&input.residual.0.ptr)
                .arg(&(rows as i32))
                .arg(&(hidden as i64))
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }

    pub fn join_routed(&self, stream: &Arc<CudaStream>, input: &MoeJoinRouted) -> Result<()> {
        let (rows, hidden) = input.validate()?;
        let scale_ptr = input.scale.as_ref().map_or(0u64, |s| s.ptr);
        // SAFETY: all validated views cover rows*hidden values and optional rows scale values.
        unsafe {
            stream
                .launch_builder(&self.join_routed)
                .arg(&input.routed_in.ptr)
                .arg(&input.shared_out.ptr)
                .arg(&scale_ptr)
                .arg(&input.y.0.ptr)
                .arg(&(rows as i32))
                .arg(&(hidden as i64))
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-byte aligned stand-in view; the validation never dereferences it.
    fn fake<T: Copy, const R: usize>(
        ptr: u64,
        shape: [u32; R],
        stride: [u64; R],
        bytes: u64,
    ) -> Result<Tensor<T, R>> {
        // SAFETY: the fake pointer is aligned and the extent covers the shape.
        unsafe { Tensor::new(ptr, shape, stride, bytes) }
    }

    fn residual_input(rows: u32, top_k: u32, hidden: u32) -> Result<MoeCombineResidual> {
        let choices = u64::from(rows) * u64::from(top_k);
        Ok(MoeCombineResidual {
            ids: fake::<i32, 2>(0x1000, [rows, top_k], [u64::from(top_k), 1], choices * 4)?,
            final_active: fake::<u32, 2>(
                0x2000,
                [rows, top_k],
                [u64::from(top_k), 1],
                choices * 4,
            )?,
            done: fake::<u32, 2>(0x3000, [rows, top_k], [u64::from(top_k), 1], choices * 4)?,
            contributions: fake::<bf16, 3>(
                0x4000,
                [rows, top_k, hidden],
                [u64::from(top_k) * u64::from(hidden), u64::from(hidden), 1],
                choices * u64::from(hidden) * 2,
            )?,
            shared_out: fake::<bf16, 2>(
                0x5000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?,
            x: fake::<bf16, 2>(
                0x6000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?,
            shared_expert_gate: Some(fake::<bf16, 1>(
                0x7000,
                [hidden],
                [1],
                u64::from(hidden) * 2,
            )?),
            residual: TensorMut(fake::<bf16, 2>(
                0x8000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?),
        })
    }

    fn join_input(rows: u32, hidden: u32) -> Result<MoeJoinRoutedResidual> {
        Ok(MoeJoinRoutedResidual {
            routed_in: fake::<bf16, 2>(
                0x9000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?,
            shared_out: fake::<bf16, 2>(
                0xa000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?,
            x: fake::<bf16, 2>(
                0xb000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?,
            shared_expert_gate: Some(fake::<bf16, 1>(
                0xc000,
                [hidden],
                [1],
                u64::from(hidden) * 2,
            )?),
            residual: TensorMut(fake::<bf16, 2>(
                0xd000,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?),
        })
    }

    #[test]
    fn combine_residual_validation_names_the_failing_view() {
        let input = residual_input(4, 8, 16).unwrap();
        assert_eq!(input.validate().unwrap(), (4, 8, 16));

        let mut sparse_residual = residual_input(4, 8, 16).unwrap();
        sparse_residual.residual =
            TensorMut(fake::<bf16, 2>(0x8000, [4, 16], [32, 1], 4 * 32 * 2).unwrap());
        let error = sparse_residual.validate().unwrap_err().to_string();
        assert!(error.contains("residual must be dense"), "{error}");

        let mut wrong_contributions = residual_input(4, 8, 16).unwrap();
        wrong_contributions.contributions =
            fake::<bf16, 3>(0x4000, [4, 8, 8], [64, 8, 1], 4 * 8 * 8 * 2).unwrap();
        let error = wrong_contributions.validate().unwrap_err().to_string();
        assert!(error.contains("contributions must be dense"), "{error}");

        let mut wide_gate = residual_input(4, 8, 16).unwrap();
        wide_gate.shared_expert_gate = Some(fake::<bf16, 1>(0x7000, [8], [1], 16).unwrap());
        let error = wide_gate.validate().unwrap_err().to_string();
        assert!(error.contains("shared gate must be dense"), "{error}");

        let mut aliased = residual_input(4, 8, 16).unwrap();
        aliased.residual =
            TensorMut(fake::<bf16, 2>(0x4000, [4, 16], [16, 1], 4 * 8 * 16 * 2).unwrap());
        let error = aliased.validate().unwrap_err().to_string();
        assert!(error.contains("tensors overlap"), "{error}");
    }

    #[test]
    fn join_routed_residual_validation_names_the_failing_view() {
        let input = join_input(4, 16).unwrap();
        assert_eq!(input.validate().unwrap(), (4, 16));

        let mut aliased = join_input(4, 16).unwrap();
        aliased.residual =
            TensorMut(fake::<bf16, 2>(0x9000, [4, 16], [16, 1], 4 * 16 * 2).unwrap());
        let error = aliased.validate().unwrap_err().to_string();
        assert!(error.contains("tensors overlap"), "{error}");

        let mut sparse = join_input(4, 16).unwrap();
        sparse.shared_out = fake::<bf16, 2>(0xa000, [4, 16], [32, 1], 4 * 32 * 2).unwrap();
        let error = sparse.validate().unwrap_err().to_string();
        assert!(error.contains("shared_out must be dense"), "{error}");
    }

    fn ascending_order(ids: &[i32]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&choice| ids[choice]);
        order
    }

    fn routed_sum(
        contributions: &[bf16],
        row: usize,
        top_k: usize,
        hidden: usize,
        order: &[usize],
        lanes: usize,
    ) -> Vec<f32> {
        let mut acc = vec![0.0f32; hidden];
        for &choice in order {
            let src = &contributions[(row * top_k + choice) * hidden..][..hidden];
            let mut d0 = 0;
            while d0 < hidden {
                let width = lanes.min(hidden - d0);
                let mut group = [0.0f32; 8];
                for j in 0..width {
                    group[j] = src[d0 + j].to_f32();
                }
                for j in 0..width {
                    acc[d0 + j] += group[j];
                }
                d0 += width;
            }
        }
        acc
    }

    #[test]
    fn scalar_and_group_routed_sums_match_the_fp64_oracle() {
        // hidden = 10 is not a multiple of eight: the group path runs the
        // first eight elements and the scalar path the remaining two.
        let (rows, top_k, hidden) = (2usize, 4usize, 10usize);
        let mut contributions = vec![bf16::ZERO; rows * top_k * hidden];
        for (index, value) in contributions.iter_mut().enumerate() {
            // Powers of two keep the FP32 and FP64 sums exact.
            *value = bf16::from_f32(0.5f32.powi((index % 5) as i32 - 2));
        }
        let ids: Vec<i32> = vec![5, 2, 7, 1, 9, 0, 4, 6];
        for row in 0..rows {
            let order = ascending_order(&ids[row * top_k..(row + 1) * top_k]);
            let scalar = routed_sum(&contributions, row, top_k, hidden, &order, 1);
            let grouped = routed_sum(&contributions, row, top_k, hidden, &order, 8);
            let mut oracle = vec![0.0f64; hidden];
            for &choice in &order {
                let src = &contributions[(row * top_k + choice) * hidden..][..hidden];
                for (value, addend) in oracle.iter_mut().zip(src) {
                    *value += addend.to_f64();
                }
            }
            for (d, ((scalar, grouped), oracle)) in
                scalar.iter().zip(&grouped).zip(&oracle).enumerate()
            {
                assert_eq!(
                    scalar.to_bits(),
                    grouped.to_bits(),
                    "row {row} element {d} differs between the scalar and group paths"
                );
                assert_eq!(
                    bf16::from_f32(*scalar).to_bits(),
                    bf16::from_f32(*oracle as f32).to_bits(),
                    "row {row} element {d} differs from the FP64 oracle"
                );
            }
        }
    }
}
