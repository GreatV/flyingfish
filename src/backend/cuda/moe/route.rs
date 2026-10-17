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
        probs: Option<&mut TensorMut<f32, 2>>,
    ) -> Result<()> {
        let plan = validate(&x, &router, &ids.0, &weights.0, probs.map(|p| &p.0))?;
        unsafe {
            stream
                .launch_builder(&self.route)
                .arg(&x.ptr)
                .arg(&router.ptr)
                .arg(&ids.0.ptr)
                .arg(&weights.0.ptr)
                .arg(&plan.probs_ptr)
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
    pub probs_ptr: u64,
}

pub(crate) fn validate(
    x: &Tensor<bf16, 2>,
    router: &Tensor<bf16, 2>,
    ids: &Tensor<i32, 2>,
    weights: &Tensor<bf16, 2>,
    probs: Option<&Tensor<f32, 2>>,
) -> Result<RoutePlan> {
    x.validate()?;
    router.validate()?;
    ids.validate()?;
    weights.validate()?;
    let (rows, hidden) = (x.shape[0], x.shape[1]);
    let (experts, top_k) = (router.shape[0], ids.shape[1]);
    ensure!(router.shape[1] == hidden, "route hidden width mismatch");
    ensure!(
        ids.shape == [rows, top_k] && weights.shape == [rows, top_k],
        "route ids/weights shape mismatch"
    );
    ensure!(
        x.stride == [u64::from(hidden), 1] && router.stride == [u64::from(hidden), 1],
        "route requires contiguous x and router rows"
    );
    ensure!(
        ids.stride == [u64::from(top_k), 1] && weights.stride == [u64::from(top_k), 1],
        "route requires contiguous ids and weights rows"
    );
    ensure!(
        hidden >= 8 && hidden.is_multiple_of(8),
        "route hidden width must be a multiple of 8"
    );
    ensure!(experts <= 1024, "route expert count above 1024");
    ensure!((1..=32).contains(&top_k), "route top_k outside [1,32]");
    ensure!(
        top_k < experts,
        "route top_k must be below the expert count"
    );
    let probs_ptr = match probs {
        Some(p) => {
            p.validate()?;
            ensure!(
                p.shape == [rows, experts] && p.stride == [u64::from(experts), 1],
                "route probs must be a contiguous [rows, experts] view"
            );
            p.ptr
        }
        None => 0,
    };
    Ok(RoutePlan {
        rows,
        experts,
        top_k,
        hidden,
        probs_ptr,
    })
}

/// Router outputs feed decode/prefill: torch.topk returns K distinct
/// experts per row, so a duplicate id within a row is an invalid input.
pub(crate) fn ensure_distinct_row_experts(ids: &[i32], top_k: usize) -> Result<()> {
    ensure!(
        !ids.is_empty() && ids.len().is_multiple_of(top_k),
        "ids length {} is not a multiple of top_k {top_k}",
        ids.len()
    );
    for row in ids.chunks(top_k) {
        for c0 in 0..row.len() {
            for c1 in c0 + 1..row.len() {
                ensure!(
                    row[c0] != row[c1],
                    "duplicate expert id {} within a row (top-k must be distinct)",
                    row[c0]
                );
            }
        }
    }
    Ok(())
}
