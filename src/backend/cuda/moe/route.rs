//! MoE router op: x·Wᵀ logits, FP32 softmax over BF16 logits, top-k ids and
//! renormalised BF16 weights (kernels/moe_route.cu). `route` computes the
//! logits with `moe_route_logits_bf16` (one FP32 chain per expert, decode row
//! counts); `route_prefill` computes them with one cuBLAS BF16 GEMM (FP32
//! accumulation). Both apply `moe_route_topk` to the dense logits buffer.
use crate::backend::cuda::contracts::{Tensor, TensorMut, no_overlap};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
use anyhow::{Result, ensure};
use cudarc::cublas::{CudaBlas, sys as bs};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

/// Experts per `moe_route_logits_bf16` block and its block size.
const LOGITS_EXPERTS: u32 = 32;
const LOGITS_THREADS: u32 = 128;

pub(crate) struct RouteOps {
    logits: CudaFunction,
    topk: CudaFunction,
    blas: CudaBlas,
    stream: Arc<CudaStream>,
}

impl RouteOps {
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_route")?;
        let blas = CudaBlas::new(stream.clone())?;
        unsafe {
            bs::cublasSetMathMode(*blas.handle(), bs::cublasMath_t::CUBLAS_DEFAULT_MATH)
                .result()?;
        }
        Ok(Self {
            logits: module.load_function("moe_route_logits_bf16")?,
            topk: module.load_function("moe_route_topk")?,
            blas,
            stream: stream.clone(),
        })
    }

    /// Decode routing: `moe_route_logits_bf16` into `logits` ([rows, experts]
    /// dense), then `moe_route_topk`.
    pub fn route(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
        logits: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "MoE route stream differs from its construction stream"
        );
        let plan = validate(&x, &router, &ids.0, &weights.0)?;
        check_logits(&plan, &x, &router, &ids.0, &weights.0, &logits.0)?;
        unsafe {
            stream
                .launch_builder(&self.logits)
                .arg(&x.ptr)
                .arg(&router.ptr)
                .arg(&logits.0.ptr)
                .arg(&plan.rows)
                .arg(&plan.experts)
                .arg(&plan.hidden)
                .launch(LaunchConfig {
                    grid_dim: (plan.experts.div_ceil(LOGITS_EXPERTS), plan.rows, 1),
                    block_dim: (LOGITS_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.topk(stream, &plan, logits, ids, weights)
    }

    /// Prefill routing: logits = x·routerᵀ via one cuBLAS GEMM, then
    /// `moe_route_topk`. `logits` is the GEMM output buffer, [rows, experts]
    /// dense; it is not read after this call returns.
    pub fn route_prefill(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
        logits: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe route prefill stream is not the RouteOps construction stream (cuBLAS handle bound at construction)"
        );
        let plan = validate(&x, &router, &ids.0, &weights.0)?;
        check_logits(&plan, &x, &router, &ids.0, &weights.0, &logits.0)?;
        let (rows, experts) = (plan.rows, plan.experts);
        ensure!(
            experts.is_multiple_of(2),
            "route prefill expert count {experts} must be even (cuBLAS BF16 ldc*2 must be a multiple of 4 bytes)"
        );
        ensure!(
            rows <= i32::MAX as u32 && experts <= i32::MAX as u32 && plan.hidden <= i32::MAX as u32,
            "route prefill GEMM dimensions must fit i32: rows {rows}, experts {experts}, hidden {}",
            plan.hidden
        );

        // opA=T/opB=N: router[H,E] * x[H,rows] -> logits[E,rows], column-major.
        // CUBLAS_COMPUTE_32F/CUBLAS_GEMM_DEFAULT; lda=ldb=H, ldc=E.
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            bs::cublasGemmEx(
                *self.blas.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                experts as i32,
                rows as i32,
                plan.hidden as i32,
                (&alpha as *const f32).cast(),
                router.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                plan.hidden as i32,
                x.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                plan.hidden as i32,
                (&beta as *const f32).cast(),
                logits.0.ptr as *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                experts as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        self.topk(stream, &plan, logits, ids, weights)
    }

    fn topk(
        &self,
        stream: &Arc<CudaStream>,
        plan: &RoutePlan,
        logits: &TensorMut<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.topk)
                .arg(&logits.0.ptr)
                .arg(&ids.0.ptr)
                .arg(&weights.0.ptr)
                .arg(&plan.rows)
                .arg(&plan.experts)
                .arg(&plan.top_k)
                .launch(grid(plan.rows as usize, plan.experts.div_ceil(32) * 32))?;
        }
        Ok(())
    }
}

fn check_logits(
    plan: &RoutePlan,
    x: &Tensor<bf16, 2>,
    router: &Tensor<bf16, 2>,
    ids: &Tensor<i32, 2>,
    weights: &Tensor<bf16, 2>,
    logits: &Tensor<bf16, 2>,
) -> Result<()> {
    logits.validate()?;
    let (rows, experts) = (plan.rows, plan.experts);
    ensure!(
        logits.shape == [rows, experts] && logits.stride == [u64::from(experts), 1],
        "route logits must be a dense [rows, experts] view: shape {:?}, stride {:?}, expected [{rows}, {experts}] with stride [{experts}, 1]",
        logits.shape,
        logits.stride
    );
    no_overlap(x, logits, "x", "logits")?;
    no_overlap(router, logits, "router", "logits")?;
    no_overlap(ids, logits, "ids", "logits")?;
    no_overlap(weights, logits, "weights", "logits")
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
