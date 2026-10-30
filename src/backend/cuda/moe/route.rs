//! BF16 router logits and deterministic top-k. Decode uses one launch;
//! prefill uses cuBLAS GEMM followed by selection over dense BF16 logits.
use crate::backend::cuda::contracts::{Tensor, TensorMut, no_overlap};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
#[cfg(feature = "rocm")]
use crate::backend::gpu::blas::Blas as CudaBlas;
use crate::backend::gpu::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use anyhow::{Context, Result, ensure};
#[cfg(feature = "cuda")]
use cudarc::cublas::{CudaBlas, sys as bs};
use half::bf16;
use std::{any::TypeId, sync::Arc};

pub(crate) struct RouteOps {
    decode: CudaFunction,
    launch: Vec<(u32, LaunchConfig)>,
    completed: CudaSlice<u32>,
    experts: u32,
    topk: CudaFunction,
    sigmoid_topk: CudaFunction,
    noaux_topk: CudaFunction,
    noaux_topk_f32: CudaFunction,
    blas: CudaBlas,
    #[cfg(feature = "cuda")]
    logits_gemm: Option<super::super::blas::RouteLogits>,
    stream: Arc<CudaStream>,
}

impl RouteOps {
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        experts: u32,
        max_rows: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        use cudarc::driver::sys::CUdevice_attribute as A;
        ensure!(
            (1..=1024).contains(&experts),
            "invalid route experts={experts}"
        );
        ensure!(
            max_rows > 0 && max_rows <= u32::MAX as usize,
            "invalid route capacity={max_rows}"
        );
        let module = cubin::module(ctx, "moe_route")?;
        let decode = module.load_function("moe_route_decode_bf16")?;
        #[cfg(feature = "cuda")]
        let (sms, grid_y, max_threads, shared_limit) = (
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
        );
        #[cfg(feature = "rocm")]
        let (sms, grid_y, max_threads, shared_limit) = (
            ctx.properties.compute_units,
            ctx.properties.max_grid_y,
            ctx.properties.max_threads,
            ctx.properties.shared_bytes,
        );
        let sms = u32::try_from(sms)?;
        let threads = experts.div_ceil(32) * 32;
        ensure!(sms > 0, "route device has no SMs");
        ensure!(
            max_rows <= usize::try_from(grid_y)?,
            "route capacity exceeds device grid row limit"
        );
        ensure!(
            threads <= u32::try_from(max_threads)?
                && threads <= u32::try_from(decode.max_threads_per_block()?)?,
            "route block exceeds device or function thread limit"
        );
        let shared_limit = u32::try_from(shared_limit)?;
        let static_shared = u32::try_from(decode.shared_size_bytes()?)?;
        for owners in [1, 2, 4, 8, 16, 32] {
            let shared = 16 * (owners * 65 + 64);
            ensure!(
                shared + static_shared <= shared_limit,
                "route shared memory exceeds device limit"
            );
            ensure!(
                decode.occupancy_max_active_blocks_per_multiprocessor(
                    threads,
                    shared as usize,
                    None
                )? > 0,
                "route has zero occupancy: threads={threads} owners={owners} shared={shared}"
            );
        }
        let launch = (1..=max_rows as u32)
            .map(|rows| {
                let target = sms.div_ceil(rows);
                let mut owners = 1;
                for q in [2, 4, 8, 16, 32] {
                    if experts.div_ceil(q) >= target {
                        owners = q;
                    }
                }
                (
                    owners,
                    LaunchConfig {
                        grid_dim: (experts.div_ceil(owners), rows, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 16 * (owners * 65 + 64),
                    },
                )
            })
            .collect();
        let blas = CudaBlas::new(stream.clone())?;
        #[cfg(feature = "cuda")]
        unsafe {
            bs::cublasSetMathMode(*blas.handle(), bs::cublasMath_t::CUBLAS_DEFAULT_MATH)
                .result()?;
        }
        Ok(Self {
            decode,
            launch,
            completed: stream.alloc_zeros(max_rows)?,
            experts,
            topk: module.load_function("moe_route_topk")?,
            sigmoid_topk: module.load_function("moe_route_topk_sigmoid")?,
            noaux_topk: module.load_function("moe_route_topk_noaux")?,
            noaux_topk_f32: module.load_function("moe_route_topk_noaux_f32")?,
            blas,
            #[cfg(feature = "cuda")]
            logits_gemm: None,
            stream: stream.clone(),
        })
    }

    /// One decode launch with row-local completion counters on the bound stream.
    pub fn route(
        &self,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
        logits: &mut TensorMut<bf16, 2>,
        score: crate::model_spec::RouterScore,
    ) -> Result<()> {
        let stream = &self.stream;
        let plan = validate(&x, &router, &ids.0, &weights.0)?;
        check_logits(&plan, &x, &router, &ids.0, &weights.0, &logits.0)?;
        ensure!(
            plan.experts == self.experts && plan.rows > 0,
            "route shape differs from construction: experts={} expected={} rows={}",
            plan.experts,
            self.experts,
            plan.rows
        );
        let &(owners, launch) = self
            .launch
            .get(plan.rows as usize - 1)
            .context("route rows exceed admitted capacity")?;
        let score_mode = match score {
            crate::model_spec::RouterScore::Softmax => 0i32,
            crate::model_spec::RouterScore::Sigmoid => 1,
            crate::model_spec::RouterScore::SqrtSoftplus => {
                anyhow::bail!("sqrt-softplus requires the FP32 noaux entry")
            }
        };
        let (completed, _guard) = self.completed.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.decode)
                .arg(&x.ptr)
                .arg(&router.ptr)
                .arg(&logits.0.ptr)
                .arg(&ids.0.ptr)
                .arg(&weights.0.ptr)
                .arg(&completed)
                .arg(&plan.rows)
                .arg(&plan.experts)
                .arg(&plan.hidden)
                .arg(&plan.top_k)
                .arg(&owners)
                .arg(&score_mode)
                .launch(launch)?;
        }
        Ok(())
    }

    pub fn workspace_bytes(&self) -> usize {
        self.completed.len() * size_of::<u32>()
    }

    /// Router logits with an FP32 output on the RouteOps cuBLAS handle:
    /// one cublasGemmEx with opA=T/opB=N, `router[H,E] * x[H,rows] ->
    /// logits[E,rows]` column-major (row-major `[rows, experts]`),
    /// CUBLAS_COMPUTE_32F with FP32 C (model definition Gate.forward
    /// linear(x.float(), weight.float())).
    pub fn logits_f32(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        logits: &mut TensorMut<f32, 2>,
    ) -> Result<()> {
        let (rows, experts, hidden) = self.validate_logits_f32(stream, x, router, logits)?;
        #[cfg(feature = "rocm")]
        unsafe {
            self.blas.bf16_gemm(
                crate::backend::setup::LinearShape {
                    rows: rows as usize,
                    output: experts as usize,
                    input: hidden as usize,
                },
                [router.ptr, x.ptr, logits.0.ptr],
                [hidden as usize, hidden as usize, experts as usize],
                None,
                true,
            )?;
        }
        #[cfg(feature = "cuda")]
        {
            let alpha = 1f32;
            let beta = 0f32;
            unsafe {
                bs::cublasGemmEx(
                    *self.blas.handle(),
                    bs::cublasOperation_t::CUBLAS_OP_T,
                    bs::cublasOperation_t::CUBLAS_OP_N,
                    experts as i32,
                    rows as i32,
                    hidden as i32,
                    (&alpha as *const f32).cast(),
                    router.ptr as *const core::ffi::c_void,
                    bs::cudaDataType_t::CUDA_R_16BF,
                    hidden as i32,
                    x.ptr as *const core::ffi::c_void,
                    bs::cudaDataType_t::CUDA_R_16BF,
                    hidden as i32,
                    (&beta as *const f32).cast(),
                    logits.0.ptr as *mut core::ffi::c_void,
                    bs::cudaDataType_t::CUDA_R_32F,
                    experts as i32,
                    bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .result()?;
            }
        }
        Ok(())
    }

    /// Prefill routing: logits = x·routerᵀ via one cuBLAS GEMM, then
    /// `moe_route_topk`. `logits` is the GEMM output buffer, [rows, experts]
    /// dense; it is not read after this call returns.
    pub fn route_prefill(
        &self,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
        logits: &mut TensorMut<bf16, 2>,
        score: crate::model_spec::RouterScore,
    ) -> Result<()> {
        let stream = &self.stream;
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
        #[cfg(feature = "cuda")]
        unsafe {
            let alpha = 1f32;
            let beta = 0f32;
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
        #[cfg(feature = "rocm")]
        unsafe {
            self.blas.bf16_gemm(
                crate::backend::setup::LinearShape {
                    rows: rows as usize,
                    output: experts as usize,
                    input: plan.hidden as usize,
                },
                [router.ptr, x.ptr, logits.0.ptr],
                [plan.hidden as usize, plan.hidden as usize, experts as usize],
                None,
                false,
            )?;
        }
        self.topk(stream, &plan, logits, ids, weights, score)
    }

    fn validate_logits_f32(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        logits: &TensorMut<f32, 2>,
    ) -> Result<(u32, u32, u32)> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe route FP32 logits stream is not the RouteOps construction stream (cuBLAS handle bound at construction)"
        );
        x.validate()?;
        router.validate()?;
        logits.0.validate()?;
        let (rows, hidden) = (x.shape[0], x.shape[1]);
        let experts = router.shape[0];
        ensure!(
            x.stride == [u64::from(hidden), 1] && router.stride == [u64::from(hidden), 1],
            "FP32 route logits require contiguous x and router rows: x {:?}/{:?}, router {:?}/{:?}",
            x.shape,
            x.stride,
            router.shape,
            router.stride
        );
        ensure!(
            router.shape[1] == hidden,
            "FP32 route logits hidden width mismatch: x {:?}, router {:?}",
            x.shape,
            router.shape
        );
        ensure!(
            logits.0.shape == [rows, experts] && logits.0.stride == [u64::from(experts), 1],
            "FP32 route logits must be a dense [rows, experts] view: shape {:?}, stride {:?}, expected [{rows}, {experts}]",
            logits.0.shape,
            logits.0.stride
        );
        no_overlap(&x, &logits.0, "x", "logits")?;
        no_overlap(&router, &logits.0, "router", "logits")?;
        ensure!(
            rows <= i32::MAX as u32 && experts <= i32::MAX as u32 && hidden <= i32::MAX as u32,
            "FP32 route logits GEMM dimensions must fit i32: rows {rows}, experts {experts}, hidden {hidden}"
        );
        Ok((rows, experts, hidden))
    }

    #[cfg(feature = "cuda")]
    pub fn prepare_logits_f32(&mut self, rows: u32, hidden: u32) -> Result<()> {
        if self.logits_gemm.is_none() {
            self.logits_gemm = Some(super::super::blas::RouteLogits::new(&self.stream)?);
        }
        self.logits_gemm
            .as_ref()
            .context("Lt route handle missing")?
            .plan_for(rows as usize, self.experts as usize, hidden as usize)
    }

    #[cfg(feature = "cuda")]
    pub fn logits_f32_lt(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        router: Tensor<bf16, 2>,
        logits: &mut TensorMut<f32, 2>,
    ) -> Result<()> {
        self.validate_logits_f32(stream, x, router, logits)?;
        self.logits_gemm
            .as_ref()
            .context("Lt route logits were not prepared")?
            .matmul_f32(&router, &x, &logits.0)
    }

    pub fn topk_noaux<T: DeviceRepr + Copy + 'static>(
        &self,
        logits: Tensor<T, 2>,
        bias: Option<Tensor<f32, 2>>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
        score: crate::model_spec::RouterScore,
        routed_scaling: f32,
    ) -> Result<()> {
        self.noaux(
            logits,
            bias,
            ids,
            weights,
            (&self.noaux_topk, score, routed_scaling),
        )
    }

    /// The same noaux selection writing FP32 routed weights (model definition
    /// Gate.forward FP32 scores applied by Expert.forward in FP32).
    pub fn topk_noaux_f32<T: DeviceRepr + Copy + 'static>(
        &self,
        logits: Tensor<T, 2>,
        bias: Option<Tensor<f32, 2>>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<f32, 2>,
        score: crate::model_spec::RouterScore,
        routed_scaling: f32,
    ) -> Result<()> {
        self.noaux(
            logits,
            bias,
            ids,
            weights,
            (&self.noaux_topk_f32, score, routed_scaling),
        )
    }

    fn noaux<T, W>(
        &self,
        logits: Tensor<T, 2>,
        bias: Option<Tensor<f32, 2>>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<W, 2>,
        launch: (&CudaFunction, crate::model_spec::RouterScore, f32),
    ) -> Result<()>
    where
        T: DeviceRepr + Copy + 'static,
        W: DeviceRepr + Copy + 'static,
    {
        let (kernel, score, routed_scaling) = launch;
        let (rows, experts) = (logits.shape[0], logits.shape[1]);
        let top_k = ids.0.shape[1];
        if TypeId::of::<T>() == TypeId::of::<bf16>() {
            ensure!(
                experts > 0
                    && experts <= 1024
                    && top_k > 0
                    && top_k < experts
                    && top_k <= 32
                    && bias.is_none()
                    && score == crate::model_spec::RouterScore::Sigmoid
                    && routed_scaling == 1.0
                    && logits.ptr.is_multiple_of(16),
                "BF16 sigmoid requires zero bias, unit scaling and top_k below experts"
            );
            unsafe {
                self.stream
                    .launch_builder(&self.sigmoid_topk)
                    .arg(&logits.ptr)
                    .arg(&ids.0.ptr)
                    .arg(&weights.0.ptr)
                    .arg(&rows)
                    .arg(&experts)
                    .arg(&top_k)
                    .launch(grid(rows as usize, experts.div_ceil(32) * 32))?;
            }
            return Ok(());
        }
        logits.validate()?;
        ids.0.validate()?;
        weights.0.validate()?;
        ensure!(
            experts > 0 && experts <= 1024 && top_k > 0 && top_k <= 32 && top_k <= experts,
            "invalid noaux extents rows={rows} experts={experts} top_k={top_k}"
        );
        ensure!(
            ids.0.shape == [rows, top_k] && weights.0.shape == [rows, top_k],
            "noaux input/output shape mismatch"
        );
        ensure!(
            logits.stride == [u64::from(experts), 1]
                && ids.0.stride == [u64::from(top_k), 1]
                && weights.0.stride == ids.0.stride,
            "noaux requires dense logits and contiguous outputs"
        );
        ensure!(
            routed_scaling.is_finite() && routed_scaling > 0.0,
            "invalid routed scaling"
        );
        no_overlap(&logits, &ids.0, "logits", "ids")?;
        no_overlap(&logits, &weights.0, "logits", "weights")?;
        no_overlap(&ids.0, &weights.0, "ids", "weights")?;
        if let Some(bias) = bias {
            bias.validate()?;
            ensure!(
                bias.shape == [rows, experts]
                    && bias.stride == logits.stride
                    && bias.ptr.is_multiple_of(4),
                "noaux bias must be an aligned dense [rows, experts] FP32 view"
            );
            no_overlap(&bias, &ids.0, "bias", "ids")?;
            no_overlap(&bias, &weights.0, "bias", "weights")?;
            no_overlap(&bias, &logits, "bias", "logits")?;
        }
        ensure!(
            TypeId::of::<T>() == TypeId::of::<f32>(),
            "noaux logits must be BF16 sigmoid or FP32"
        );
        let bias = bias.context("FP32 noaux requires an explicit bias input")?;
        let score_mode = match score {
            crate::model_spec::RouterScore::Sigmoid => 0,
            crate::model_spec::RouterScore::SqrtSoftplus => 1,
            crate::model_spec::RouterScore::Softmax => {
                anyhow::bail!("noaux_tc requires a pointwise score")
            }
        };
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&logits.ptr)
                .arg(&bias.ptr)
                .arg(&ids.0.ptr)
                .arg(&weights.0.ptr)
                .arg(&rows)
                .arg(&experts)
                .arg(&top_k)
                .arg(&score_mode)
                .arg(&routed_scaling)
                .launch(grid(rows as usize, experts.div_ceil(32) * 32))?;
        }
        Ok(())
    }

    fn topk(
        &self,
        stream: &Arc<CudaStream>,
        plan: &RoutePlan,
        logits: &TensorMut<bf16, 2>,
        ids: &mut TensorMut<i32, 2>,
        weights: &mut TensorMut<bf16, 2>,
        score: crate::model_spec::RouterScore,
    ) -> Result<()> {
        if score == crate::model_spec::RouterScore::Sigmoid {
            return self.topk_noaux(logits.0, None, ids, weights, score, 1.0);
        }
        let function = match score {
            crate::model_spec::RouterScore::Softmax => &self.topk,
            crate::model_spec::RouterScore::SqrtSoftplus => {
                anyhow::bail!("sqrt-softplus requires the FP32 noaux entry")
            }
            crate::model_spec::RouterScore::Sigmoid => unreachable!(),
        };
        unsafe {
            stream
                .launch_builder(function)
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
