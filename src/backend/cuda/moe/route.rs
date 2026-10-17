//! MoE router op: x·Wᵀ logits, FP32 softmax over BF16 logits, top-k ids and
//! renormalised BF16 weights (kernels/moe_route.cu).
use crate::backend::cuda::contracts::{Tensor, TensorMut};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, PushKernelArg};
use half::bf16;
use std::sync::Arc;

pub(crate) struct RouteOps {
    route: CudaFunction,
}

impl RouteOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_route")?;
        Ok(Self {
            route: module.load_function("moe_route_bf16")?,
        })
    }

    pub fn route(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let plan = validate(&x, &router, &ids.0, &weights.0)?;
        unsafe {
            stream
                .launch_builder(&self.route)
                .arg(&x.ptr)
                .arg(&router.ptr)
                .arg(&ids.0.ptr)
                .arg(&weights.0.ptr)
                .arg(&plan.rows)
                .arg(&plan.experts)
                .arg(&plan.top_k)
                .arg(&plan.hidden)
                .launch(grid(plan.rows as usize, plan.experts.div_ceil(32) * 32))?;
        }
        Ok(())
    }
}

pub(crate) struct RoutePlan {
    pub rows: u32,
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
}

pub(crate) fn validate(
    x: &Tensor<bf16, 2>,
    router: &Tensor<bf16, 2>,
    ids: &Tensor<i32, 2>,
    weights: &Tensor<bf16, 2>,
) -> Result<RoutePlan> {
    x.validate()?;
    router.validate()?;
    ids.validate()?;
    weights.validate()?;
    let (rows, hidden) = (x.shape[0], x.shape[1]);
    let (experts, top_k) = (router.shape[0], ids.shape[1]);
    ensure!(
        router.shape[1] == hidden,
        "route hidden width mismatch: x {:?}, router {:?}",
        x.shape,
        router.shape
    );
    ensure!(
        ids.shape == [rows, top_k] && weights.shape == [rows, top_k],
        "route ids/weights shape mismatch: ids {:?}, weights {:?}, expected [{rows}, {top_k}]",
        ids.shape,
        weights.shape
    );
    ensure!(
        x.stride == [u64::from(hidden), 1] && router.stride == [u64::from(hidden), 1],
        "route requires contiguous x and router rows: x stride {:?}, router stride {:?}, expected [{hidden}, 1]",
        x.stride,
        router.stride
    );
    ensure!(
        ids.stride == [u64::from(top_k), 1] && weights.stride == [u64::from(top_k), 1],
        "route requires contiguous ids and weights rows: ids stride {:?}, weights stride {:?}, expected [{top_k}, 1]",
        ids.stride,
        weights.stride
    );
    ensure!(
        hidden >= 8 && hidden.is_multiple_of(8),
        "route hidden width {hidden} must be a multiple of 8"
    );
    ensure!(experts <= 1024, "route expert count {experts} above 1024");
    ensure!(
        (1..=32).contains(&top_k),
        "route top_k {top_k} outside [1,32]"
    );
    ensure!(
        top_k < experts,
        "route top_k {top_k} must be below the expert count {experts}"
    );
    Ok(RoutePlan {
        rows,
        experts,
        top_k,
        hidden,
    })
}
