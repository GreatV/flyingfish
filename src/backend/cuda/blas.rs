use super::{Device, cubin, linear_calibrate, ops::Ops};

/// cuBLASLt heuristic candidates trialled per linear shape, alongside Skinny.
pub const LT_CANDIDATES: u32 = 2;

unsafe fn heuristic_algos(
    handle: sys::cublasLtHandle_t,
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    pref: sys::cublasLtMatmulPreference_t,
) -> Result<Vec<sys::cublasLtMatmulHeuristicResult_t>> {
    let mut results = vec![
        std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::uninit();
        LT_CANDIDATES as usize
    ];
    let mut count = 0i32;
    unsafe {
        sys::cublasLtMatmulAlgoGetHeuristic(
            handle,
            desc,
            a,
            b,
            c,
            c,
            pref,
            LT_CANDIDATES as i32,
            results.as_mut_ptr().cast(),
            &mut count,
        )
        .result()?;
    }
    ensure!(count >= 1, "cuBLASLt returned no algorithm");
    results.truncate(count as usize);
    let ok: Vec<sys::cublasLtMatmulHeuristicResult_t> = results
        .into_iter()
        .filter_map(|r| {
            let r = unsafe { r.assume_init() };
            r.state.result().is_ok().then_some(r)
        })
        .collect();
    ensure!(!ok.is_empty(), "cuBLASLt returned no successful algorithm");
    Ok(ok)
}
use crate::backend::cuda::moe::route::ROUTE_LOGITS_WORKSPACE_BYTES;
use crate::backend::setup::{LinearChoice, LinearImpl, LinearShape, LtAlgorithm};
use anyhow::{Context, Result, ensure};
use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, sys as bs},
    cublaslt::{result as lt, sys},
    driver::{
        CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut,
        LaunchConfig, PushKernelArg,
    },
};
use half::bf16;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    mem::size_of,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

/// Algorithm selection for one plan: a heuristic rank or a fixed algorithm.
enum PlanAlgo {
    Heuristic(u32),
    Fixed(sys::cublasLtMatmulAlgo_t),
}

/// Config identity of a raw Lt algorithm struct.
fn algo_config(algo: &sys::cublasLtMatmulAlgo_t) -> Result<LtAlgorithm> {
    use sys::cublasLtMatmulAlgoConfigAttributes_t as A;
    let read = |attr: sys::cublasLtMatmulAlgoConfigAttributes_t, width: usize| -> Result<i64> {
        let mut value = 0i64;
        let mut written = 0usize;
        unsafe {
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                algo,
                attr,
                (&mut value as *mut i64).cast(),
                width,
                &mut written,
            )
            .result()?;
        }
        ensure!(
            written == width,
            "Lt attribute {attr:?} returned {written} bytes"
        );
        Ok(value)
    };
    Ok(LtAlgorithm {
        id: read(A::CUBLASLT_ALGO_CONFIG_ID, size_of::<i32>())? as i32,
        tile: read(A::CUBLASLT_ALGO_CONFIG_TILE_ID, size_of::<i32>())? as i32,
        split_k: read(A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, size_of::<i32>())? as i32,
        reduction: read(A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, size_of::<i32>())? as i32,
        swizzle: read(A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, size_of::<i32>())? as i32,
        custom: read(A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, size_of::<i32>())? as i32,
        stages: read(A::CUBLASLT_ALGO_CONFIG_STAGES_ID, size_of::<i32>())? as i32,
        inner: read(A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, size_of::<u16>())? as u16,
        cluster: read(A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, size_of::<u16>())? as u16,
    })
}

struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    pref: sys::cublasLtMatmulPreference_t,
    algo: Option<sys::cublasLtMatmulAlgo_t>,
    algo_index: u32,
    algo_count: usize,
    waves: f32,
    workspace: usize,
}

impl Plan {
    fn identity(&self) -> Result<LtAlgorithm> {
        algo_config(self.algo.as_ref().context("Lt algorithm missing")?)
    }

    fn attr(&self, attr: sys::cublasLtMatmulAlgoConfigAttributes_t) -> Result<i32> {
        let mut value = 0i32;
        let mut written = 0usize;
        unsafe {
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                self.algo.as_ref().context("Lt algorithm missing")?,
                attr,
                (&mut value as *mut i32).cast(),
                size_of::<i32>(),
                &mut written,
            )
            .result()?;
        }
        ensure!(
            written == size_of::<i32>(),
            "Lt attribute {attr:?} returned {written} bytes"
        );
        Ok(value)
    }

    fn describe(&self, rows: usize, output: usize, input: usize) -> Result<()> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as A;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"cublaslt_plan":{"rows":rows,"output":output,"input":input,
                "id":self.attr(A::CUBLASLT_ALGO_CONFIG_ID)?,"tile":self.attr(A::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
                "split_k":self.attr(A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,"reduction_scheme":self.attr(A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
                "estimated_waves":self.waves,"workspace_bytes":self.workspace}})
            );
        }
        Ok(())
    }

    fn new(
        handle: sys::cublasLtHandle_t,
        rows: usize,
        output: usize,
        input: usize,
        workspace: usize,
        algo_index: u32,
        ab: sys::cudaDataType_t,
    ) -> Result<Self> {
        Self::build(
            handle,
            (rows, output, input, input),
            workspace,
            None,
            ab,
            ab,
            PlanAlgo::Heuristic(algo_index),
        )
    }

    /// F16-input plan bound to the heuristic first result, the reference
    /// library's pick for plain F16 linears.
    fn new_f16(
        handle: sys::cublasLtHandle_t,
        rows: usize,
        output: usize,
        input: usize,
        workspace: usize,
        algo_index: u32,
    ) -> Result<Self> {
        Self::build(
            handle,
            (rows, output, input, input),
            workspace,
            None,
            sys::cudaDataType_t::CUDA_R_16F,
            sys::cudaDataType_t::CUDA_R_16F,
            PlanAlgo::Heuristic(algo_index),
        )
    }

    /// F32-output variant of the plain plan: same geometry and heuristic
    /// contract, C/D stored as CUDA_R_32F.
    fn new_f32_out(
        handle: sys::cublasLtHandle_t,
        rows: usize,
        output: usize,
        input: usize,
        workspace: usize,
        algo_index: u32,
    ) -> Result<Self> {
        Self::build(
            handle,
            (rows, output, input, input),
            workspace,
            None,
            sys::cudaDataType_t::CUDA_R_16BF,
            sys::cudaDataType_t::CUDA_R_32F,
            PlanAlgo::Heuristic(algo_index),
        )
    }

    /// Geometry tuple is `(rows, output, input, b_ld)`.
    fn build(
        handle: sys::cublasLtHandle_t,
        geometry: (usize, usize, usize, usize),
        workspace: usize,
        bias: Option<u64>,
        ab_dtype: sys::cudaDataType_t,
        c_dtype: sys::cudaDataType_t,
        select: PlanAlgo,
    ) -> Result<Self> {
        let (rows, output, input, b_ld) = geometry;
        let b_ld = i64::try_from(b_ld).context("Lt B leading dimension")?;
        ensure!(
            b_ld >= i64::try_from(input).context("Lt K")?,
            "Lt B leading dimension {b_ld} is below K {input}"
        );
        let mut p = Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            pref: std::ptr::null_mut(),
            algo: None,
            algo_index: 0,
            algo_count: 0,
            waves: 0.0,
            workspace: 0,
        };
        p.desc = lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cuda_lt_dtype(crate::backend::setup::lt_compute_operand()),
        )?;
        let transpose = 1i32;
        unsafe {
            lt::set_matmul_desc_attribute(
                p.desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&transpose as *const i32).cast(),
                size_of::<i32>(),
            )?;
        }
        let dtype = ab_dtype;
        if let Some(ptr) = bias {
            let epilogue = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            unsafe {
                lt::set_matmul_desc_attribute(
                    p.desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    (&epilogue as *const sys::cublasLtEpilogue_t).cast(),
                    size_of::<sys::cublasLtEpilogue_t>(),
                )?;
                lt::set_matmul_desc_attribute(
                    p.desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&ptr as *const u64).cast(),
                    size_of::<u64>(),
                )?;
                lt::set_matmul_desc_attribute(
                    p.desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    (&dtype as *const sys::cudaDataType_t).cast(),
                    size_of::<sys::cudaDataType_t>(),
                )?;
            }
        }
        p.a = lt::create_matrix_layout(dtype, input as u64, output as u64, input as i64)?;
        p.b = lt::create_matrix_layout(dtype, input as u64, rows as u64, b_ld)?;
        p.c = lt::create_matrix_layout(c_dtype, output as u64, rows as u64, output as i64)?;
        p.pref = lt::create_matmul_pref()?;
        unsafe {
            // Biased plans are single-pass; split-K bias epilogues round before bias.
            let reduction_mask = if bias.is_some() {
                sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as u32
            } else {
                sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32
            };
            lt::set_matmul_pref_attribute(p.pref, sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                (&reduction_mask as *const u32).cast(), size_of::<u32>())?;
            lt::set_matmul_pref_attribute(
                p.pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&workspace as *const usize).cast(),
                size_of::<usize>(),
            )?;
            let checked = match select {
                PlanAlgo::Heuristic(index) => {
                    let algos = heuristic_algos(handle, p.desc, p.a, p.b, p.c, p.pref)?;
                    ensure!(
                        (index as usize) < algos.len(),
                        "Lt heuristic returned {} algos; index {index} unavailable",
                        algos.len()
                    );
                    p.algo_count = algos.len();
                    let heuristic = &algos[index as usize];
                    heuristic.state.result()?;
                    p.waves = heuristic.wavesCount;
                    p.workspace = heuristic.workspaceSize;
                    p.algo = Some(heuristic.algo);
                    p.algo_index = index;
                    None
                }
                PlanAlgo::Fixed(algo) => Some(algo),
            };
            if let Some(algo) = checked {
                let mut result = std::mem::MaybeUninit::uninit();
                sys::cublasLtMatmulAlgoCheck(
                    handle,
                    p.desc,
                    p.a,
                    p.b,
                    p.c,
                    p.c,
                    &algo,
                    result.as_mut_ptr(),
                )
                .result()
                .with_context(|| {
                    format!("fixed Lt AlgoCheck rows={rows} output={output} input={input}")
                })?;
                let result = result.assume_init();
                result.state.result()?;
                p.algo = Some(algo);
                p.waves = result.wavesCount;
                p.workspace = result.workspaceSize;
                p.algo_count = 1;
            }
        }
        let split_k =
            p.attr(sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?;
        let reduction = p.attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        )?;
        if bias.is_some() {
            ensure!(
                split_k == 1
                    && reduction
                        == sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as i32,
                "biased Lt plan must add bias before output rounding: split_k={split_k}, reduction={reduction}"
            );
        }
        ensure!(
            split_k <= 1
                || reduction
                    == sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
                        as i32,
            "Lt selected non-FP32 split-K reduction {reduction}"
        );
        Ok(p)
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            for (name, x) in [
                ("destroy Lt layout A", self.a),
                ("destroy Lt layout B", self.b),
                ("destroy Lt layout C", self.c),
            ] {
                if !x.is_null() {
                    drop_result(name, lt::destroy_matrix_layout(x));
                }
            }
            if !self.desc.is_null() {
                drop_result("destroy Lt descriptor", lt::destroy_matmul_desc(self.desc));
            }
            if !self.pref.is_null() {
                drop_result("destroy Lt preference", lt::destroy_matmul_pref(self.pref));
            }
        }
    }
}

/// Lt GEMM plans for the MoE route logits: BF16 operands, CUBLAS_COMPUTE_32F,
/// BF16 or F32 output. Plans are bound on first use per geometry; the
/// workspace is preallocated so captured launches perform no allocation.
pub(crate) struct RouteLogits {
    handle: sys::cublasLtHandle_t,
    stream: Arc<CudaStream>,
    workspace: CudaSlice<u8>,
    workspace_ptr: u64,
    plans: RouteLogitsPlans,
}

/// Keyed by (rows, experts, hidden): (BF16-out prefill plan, F32-out plan).
/// Keyed by (rows, experts, hidden).
type RouteLogitsPlans = RefCell<BTreeMap<(usize, usize, usize), Plan>>;

impl RouteLogits {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let handle = lt::create_handle()?;
        let workspace = stream.alloc_zeros::<u8>(ROUTE_LOGITS_WORKSPACE_BYTES)?;
        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _guard) = workspace.device_ptr(stream);
            ptr
        };
        Ok(Self {
            handle,
            stream: stream.clone(),
            workspace,
            workspace_ptr,
            plans: RefCell::new(BTreeMap::new()),
        })
    }

    pub(crate) fn plan_for(&self, rows: usize, experts: usize, hidden: usize) -> Result<()> {
        if self.plans.borrow().contains_key(&(rows, experts, hidden)) {
            return Ok(());
        }
        let workspace = self.workspace.len();
        let plan = Plan::new_f32_out(self.handle, rows, experts, hidden, workspace, 0)?;
        self.plans
            .borrow_mut()
            .insert((rows, experts, hidden), plan);
        Ok(())
    }

    pub(crate) fn matmul_f32(
        &self,
        weight: &crate::backend::cuda::contracts::Tensor<bf16, 2>,
        x: &crate::backend::cuda::contracts::Tensor<bf16, 2>,
        y: &crate::backend::cuda::contracts::Tensor<f32, 2>,
    ) -> Result<()> {
        let plans = self.plans.borrow();
        let p = plans
            .get(&(
                x.shape[0] as usize,
                weight.shape[0] as usize,
                weight.shape[1] as usize,
            ))
            .expect("route logits F32 plan bound before run");
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            lt::matmul(
                self.handle,
                p.desc,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                weight.ptr as _,
                p.a,
                x.ptr as _,
                p.b,
                y.ptr as _,
                p.c,
                y.ptr as _,
                p.c,
                p.algo.as_ref().context("route logits F32 algo missing")?,
                self.workspace_ptr as _,
                self.workspace.len(),
                self.stream.cu_stream() as _,
            )?;
        }
        Ok(())
    }
}

impl Drop for RouteLogits {
    fn drop(&mut self) {
        unsafe {
            drop_result(
                "destroy route logits Lt handle",
                lt::destroy_handle(self.handle),
            );
        }
    }
}

type LinearPlans = RefCell<BTreeMap<(usize, usize, usize, u32, bool), Plan>>;

fn biased_group_dtypes(half: sys::cudaDataType_t) -> (sys::cudaDataType_t, sys::cudaDataType_t) {
    (half, half)
}

fn cuda_lt_dtype(operand: crate::backend::setup::LtOperand) -> sys::cudaDataType_t {
    use crate::backend::setup::LtOperand;
    match operand {
        LtOperand::F32 => sys::cudaDataType_t::CUDA_R_32F,
        LtOperand::F16 => sys::cudaDataType_t::CUDA_R_16F,
        LtOperand::Bf16 => sys::cudaDataType_t::CUDA_R_16BF,
    }
}

pub struct Blas {
    choices: RefCell<BTreeMap<LinearShape, (LinearChoice, u32)>>,
    coverage: Option<(usize, std::collections::BTreeSet<(usize, usize)>)>,
    calibration: linear_calibrate::Environment,
    fp8: Option<super::linear::Fp8Ops>,
    nvfp4: Option<super::linear::NvFp4Ops>,
    skinny: CudaFunction,
    skinny_device: (usize, usize),
    skinny_plans: RefCell<BTreeMap<(usize, usize), (LaunchConfig, bool)>>,
    invariant_rows: usize,
    plans: LinearPlans,
    pitched_b: RefCell<BTreeMap<(usize, usize, usize, usize), Plan>>,
    row_families: RefCell<BTreeMap<(usize, usize), (sys::cublasLtMatmulAlgo_t, usize)>>,
    biased_plans: RefCell<BTreeMap<(LinearShape, u64, super::linear::LinearBiasMode), Plan>>,
    f16_reference: bool,
    f16_plans: RefCell<BTreeMap<LinearShape, Plan>>,
    bias_add: Option<CudaFunction>,
    bias_fill_f16: Option<CudaFunction>,
    handle: sys::cublasLtHandle_t,
    half_dtype: sys::cudaDataType_t,
    decode: CudaBlas,
    workspace: RefCell<CudaSlice<u8>>,
    stream: Arc<CudaStream>,
    closure: Rc<super::closure::Closure>,
}

/// Bounds the i32-cast paths (rows == 1 cuBLAS and skinny); the cuBLASLt path passes u64/i64 layouts.
fn checked_dims(rows: usize, output: usize, input: usize) -> Result<()> {
    ensure!(
        rows > 0
            && output > 0
            && input > 0
            && rows <= i32::MAX as usize
            && output <= i32::MAX as usize
            && input <= i32::MAX as usize,
        "linear dimensions must fit i32: rows={rows} output={output} input={input}"
    );
    Ok(())
}

fn algorithm_index(candidates: &[LinearImpl], implementation: &LinearImpl) -> Result<u32> {
    if *implementation == LinearImpl::Skinny {
        return Ok(0);
    }
    let matches: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| *candidate == implementation)
        .map(|(index, _)| index)
        .collect();
    ensure!(
        matches.len() == 1,
        "frozen algorithm {implementation:?} has {} matches in current Lt candidates {candidates:?}",
        matches.len()
    );
    Ok(matches[0] as u32)
}

impl Blas {
    #[cfg(test)]
    pub(super) fn calibration_identity_gate(
        device: &Device,
        ops: &Ops,
        runtime: &Path,
    ) -> Result<()> {
        use std::collections::BTreeSet;
        let stream = &device.stream;
        let mut discriminating = 0;
        for (output, input) in [(2048, 2048), (12288, 2048), (2048, 6144)] {
            let shape = LinearShape {
                rows: 8,
                output,
                input,
            };
            let mut blas = Self::new(
                device,
                ops,
                runtime,
                Rc::new(super::closure::Closure::new(
                    true,
                    "Lt identity gate".into(),
                )),
            )?;
            let w = stream.clone_htod(&super::random(output * input, 13))?;
            let x = stream.clone_htod(&super::random(8 * input, 17))?;
            blas.calibrate_shape(&w, &x.slice(..), shape)?;
            let candidates = blas.candidates(shape)?;
            let Some(other) = candidates
                .iter()
                .position(|candidate| *candidate != candidates[0])
            else {
                println!("Lt identity cell {shape:?} has one distinct candidate");
                continue;
            };
            discriminating += 1;
            let path = blas.choice_path(shape);
            let cold = std::fs::read(&path)?;
            let mut cache: serde_json::Value = serde_json::from_slice(&cold)?;
            let trials = cache["trials"]
                .as_array_mut()
                .context("Lt trials missing")?;
            for (i, trial) in trials.iter_mut().enumerate() {
                let median = (i + 1) as f64;
                trial["median_us"] = median.into();
                trial["p10_us"] = median.into();
                trial["p90_us"] = median.into();
            }
            let implementation = serde_json::to_value(&candidates[0])?;
            cache["choice"]["implementation"] = implementation.clone();
            cache["choice"]["median_us"] = 1.0.into();
            let frozen = serde_json::to_vec_pretty(&cache)?;
            std::fs::write(&path, &frozen)?;
            let mut y = stream.alloc_zeros::<bf16>(8 * output)?;
            blas.cublas(&w, &x.slice(..), &mut y.slice_mut(..), shape, 0)?;
            let expected = stream.clone_dtoh(&y)?;
            {
                let mut plans = blas.plans.borrow_mut();
                let first = plans
                    .remove(&(8, output, input, 0, false))
                    .context("first Lt plan missing")?;
                let second = plans
                    .remove(&(8, output, input, other as u32, false))
                    .context("second Lt plan missing")?;
                plans.insert((8, output, input, 0, false), second);
                plans.insert((8, output, input, other as u32, false), first);
            }
            blas.choices.borrow_mut().clear();
            blas.load_choices(&BTreeSet::from([shape]))?;
            let index = blas
                .choices
                .borrow()
                .get(&shape)
                .context("bound choice missing")?
                .1;
            let actual_identity = blas
                .plans
                .borrow()
                .get(&(8, output, input, index, false))
                .context("bound Lt plan missing")?
                .identity()?;
            ensure!(
                LinearImpl::CublasLt(actual_identity) == candidates[0],
                "DEFECT Lt identity binding after reorder: expected={:?} actual={actual_identity:?} index={index}",
                candidates[0]
            );
            for graph in [false, true] {
                if graph {
                    stream.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                }
                blas.linear(&w, &x.slice(..), &mut y.slice_mut(..), 8, output, input)?;
                if graph {
                    let captured = stream
                        .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                        .context("Lt gate captured no graph")?;
                    captured.launch()?;
                    stream.synchronize()?;
                }
                let actual = stream.clone_dtoh(&y)?;
                ensure!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "DEFECT Lt identity output changed after reuse {shape:?} graph={graph}"
                );
            }
            ensure!(
                std::fs::read(&path)? == frozen,
                "Lt cache changed during reuse"
            );
            let wrong_tile = i32::MAX;
            cache["choice"]["implementation"]["CublasLt"]["tile"] = wrong_tile.into();
            cache["trials"][0]["implementation"]["CublasLt"]["tile"] = wrong_tile.into();
            std::fs::write(&path, serde_json::to_vec(&cache)?)?;
            blas.choices.borrow_mut().clear();
            ensure!(
                blas.load_choices(&BTreeSet::from([shape])).is_err(),
                "invalid Lt tile was accepted"
            );
            std::fs::write(&path, &frozen)?;
            println!(
                "PASS calibration_lt_identity shape={shape:?} id={} tile={} reordered_index={index} eager_graph_bits=0 missing_tile=rejected fixture=planted_frozen_choice",
                actual_identity.id, actual_identity.tile
            );
        }
        ensure!(
            discriminating > 0,
            "no native Lt shape distinguishes reordered algorithm identities"
        );
        Ok(())
    }

    pub(crate) fn new(
        device: &Device,
        ops: &Ops,
        runtime: &Path,
        closure: Rc<super::closure::Closure>,
    ) -> Result<Self> {
        device.info.require_cuda_tensor_ops("gemm_skinny")?;
        let half_dtype = cuda_lt_dtype(device.info.arch.published_bf16_lt());
        ensure!(
            cuda_lt_dtype(device.info.arch.published_f32_lt()) == sys::cudaDataType_t::CUDA_R_32F,
            "published F32 Lt operand is CUDA_R_32F"
        );
        let stream = device.stream.clone();
        let calibration =
            linear_calibrate::Environment::new(device, ops, runtime, closure.clone())?;
        let decode = CudaBlas::new(stream.clone())?;
        unsafe {
            bs::cublasSetMathMode(
                *decode.handle(),
                bs::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let workspace = stream.alloc_zeros::<u8>(4 * 1024 * 1024)?;
        let handle = lt::create_handle()?;
        Ok(Self {
            choices: RefCell::new(BTreeMap::new()),
            coverage: None,
            calibration,
            fp8: None,
            nvfp4: None,
            skinny: cubin::module(&device.ctx, "gemm_skinny")?.load_function("gemm_skinny_bf16")?,
            skinny_device: (device.info.sms, device.info.l2_bytes),
            skinny_plans: RefCell::new(BTreeMap::new()),
            invariant_rows: 0,
            plans: RefCell::new(BTreeMap::new()),
            pitched_b: RefCell::new(BTreeMap::new()),
            row_families: RefCell::new(BTreeMap::new()),
            biased_plans: RefCell::new(BTreeMap::new()),
            f16_reference: false,
            f16_plans: RefCell::new(BTreeMap::new()),
            bias_add: None,
            bias_fill_f16: None,
            handle,
            half_dtype,
            decode,
            workspace: RefCell::new(workspace),
            stream,
            closure,
        })
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn declare(&mut self, rows: usize, pairs: &std::collections::BTreeSet<(usize, usize)>) {
        self.coverage = Some((rows, pairs.clone()));
    }
    pub fn choice_path(&self, shape: LinearShape) -> PathBuf {
        self.calibration.path(shape)
    }
    pub fn load_choices(&mut self, shapes: &std::collections::BTreeSet<LinearShape>) -> Result<()> {
        for &shape in shapes {
            let choice = self.calibration.read(shape)?;
            let index = algorithm_index(&self.candidates(shape)?, &choice.implementation)
                .with_context(|| format!("bind frozen Lt choice for {shape:?}"))?;
            if choice.implementation == LinearImpl::Skinny {
                self.prepare_skinny(shape)?;
            }
            self.choices.borrow_mut().insert(shape, (choice, index));
        }
        Ok(())
    }
    pub fn prepare(&mut self, rows: usize, output: usize, input: usize) -> Result<()> {
        self.prepare_with(rows, output, input, 0)
    }

    pub fn prepare_with(
        &self,
        rows: usize,
        output: usize,
        input: usize,
        algo_index: u32,
    ) -> Result<()> {
        if (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows)
            && (input.is_multiple_of(256) || rows <= self.invariant_rows)
        {
            self.prepare_skinny(LinearShape {
                rows,
                output,
                input,
            })?;
        }
        if rows > 1
            && !self
                .plans
                .borrow()
                .contains_key(&(rows, output, input, algo_index, false))
        {
            let p = Plan::new(
                self.handle,
                rows,
                output,
                input,
                self.workspace.borrow().len(),
                algo_index,
                self.half_dtype,
            )
            .with_context(|| format!("Lt plan rows={rows} output={output} input={input}"))?;
            p.describe(rows, output, input)?;
            self.plans
                .borrow_mut()
                .insert((rows, output, input, algo_index, false), p);
        }
        Ok(())
    }

    pub fn lt_algo_count(&self, rows: usize, output: usize, input: usize) -> Result<u32> {
        if rows <= 1 {
            return Ok(1);
        }
        self.prepare_with(rows, output, input, 0)?;
        let count = self
            .plans
            .borrow()
            .get(&(rows, output, input, 0, false))
            .context("Lt plan missing")?
            .algo_count;
        Ok(count as u32)
    }

    fn candidates(&self, shape: LinearShape) -> Result<Vec<LinearImpl>> {
        let LinearShape {
            rows,
            output,
            input,
        } = shape;
        if rows == 1 {
            return Ok(vec![LinearImpl::Cublas]);
        }
        let count = self.lt_algo_count(rows, output, input)?;
        (0..count)
            .map(|index| {
                self.prepare_with(rows, output, input, index)?;
                let plans = self.plans.borrow();
                let plan = plans
                    .get(&(rows, output, input, index, false))
                    .context("Lt candidate missing")?;
                Ok(LinearImpl::CublasLt(plan.identity()?))
            })
            .collect()
    }

    pub fn linear(
        &self,
        w: &impl DevicePtr<bf16>,
        x: &CudaView<'_, bf16>,
        y: &mut CudaViewMut<'_, bf16>,
        rows: usize,
        output: usize,
        input: usize,
    ) -> Result<()> {
        let shape = LinearShape {
            rows,
            output,
            input,
        };
        ensure!(
            !linear_calibrate::supports(shape) || self.choices.borrow().contains_key(&shape),
            "missing frozen linear choice: {shape:?}; run calibrate for this deployment"
        );
        if !linear_calibrate::supports(shape) {
            let (limit, pairs) = self
                .coverage
                .as_ref()
                .context("linear coverage was not declared")?;
            ensure!(
                shape.rows > 0
                    && shape.rows <= *limit
                    && pairs.contains(&(shape.output, shape.input)),
                "linear shape outside cuBLAS-only coverage: {shape:?}"
            );
        }
        let mut algo = 0u32;
        if let Some((choice, index)) = self.choices.borrow().get(&shape) {
            match &choice.implementation {
                LinearImpl::Cublas | LinearImpl::CublasLt(_) => algo = *index,
                LinearImpl::Skinny => return self.skinny(w, x, y, shape),
            }
        }
        self.prepare_with(rows, output, input, algo)?;
        self.cublas(w, x, y, shape, algo)
    }

    pub fn calibrate_shape(
        &mut self,
        w: &impl DevicePtr<bf16>,
        x: &CudaView<'_, bf16>,
        shape: LinearShape,
    ) -> Result<()> {
        linear_calibrate::check_extent(shape, w.len(), x.len())?;
        self.prepare_skinny(shape)?;
        let candidates = self.candidates(shape)?;
        let (rows, output) = (shape.rows, shape.output);
        {
            let choice = self
                .calibration
                .load(shape, &candidates, |implementation, out| {
                    let mut y = out.slice_mut(..rows * output);
                    match implementation {
                        LinearImpl::Cublas | LinearImpl::CublasLt(_) => {
                            let index = algorithm_index(&candidates, implementation)?;
                            self.cublas(w, x, &mut y, shape, index)
                        }
                        LinearImpl::Skinny => self.skinny(w, x, &mut y, shape),
                    }
                })?;
            let index = algorithm_index(&candidates, &choice.implementation)?;
            self.choices.borrow_mut().insert(shape, (choice, index));
        }
        Ok(())
    }

    fn skinny(
        &self,
        w: &impl DevicePtr<bf16>,
        x: &CudaView<'_, bf16>,
        y: &mut CudaViewMut<'_, bf16>,
        shape: LinearShape,
    ) -> Result<()> {
        let (wp, _w) = w.device_ptr(&self.stream);
        let (xp, _x) = x.device_ptr(&self.stream);
        let (yp, _y) = y.device_ptr_mut(&self.stream);
        self.skinny_raw(wp, xp, yp, shape)
    }
    fn skinny_raw(&self, wp: u64, xp: u64, yp: u64, shape: LinearShape) -> Result<()> {
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&shape.rows),
            "skinny rows exceed the bound: {shape:?}"
        );
        let &(mut launch, tiny) = self
            .skinny_plans
            .borrow()
            .get(&(shape.output, shape.input))
            .context("skinny launch missing; bind before run/capture")?;
        if tiny {
            launch.grid_dim.1 = shape.rows as u32;
        }
        unsafe {
            self.stream
                .launch_builder(&self.skinny)
                .arg(&wp)
                .arg(&xp)
                .arg(&yp)
                .arg(&(shape.output as i32))
                .arg(&(shape.input as i32))
                .arg(&(shape.input as i64))
                .arg(&(shape.rows as i32))
                .arg(&i32::from(tiny))
                .launch(launch)?;
        }
        Ok(())
    }
    pub(super) fn prepare_skinny(&self, shape: LinearShape) -> Result<()> {
        let mut plans = self.skinny_plans.borrow_mut();
        if let std::collections::btree_map::Entry::Vacant(entry) =
            plans.entry((shape.output, shape.input))
        {
            let bytes = shape
                .output
                .checked_mul(shape.input)
                .and_then(|n| n.checked_mul(size_of::<bf16>()))
                .context("skinny weight byte extent overflow")?;
            let tiny = shape.output <= 16 * 4
                && shape.output.div_ceil(16) < self.skinny_device.0
                && bytes <= self.skinny_device.1;
            entry.insert((Self::skinny_launch(shape, tiny)?, tiny));
        }
        Ok(())
    }
    pub(super) fn skinny_launch(shape: LinearShape, tiny: bool) -> Result<LaunchConfig> {
        checked_dims(shape.rows, shape.output, shape.input)?;
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&shape.rows)
                && shape.input.is_multiple_of(if tiny { 2 } else { 256 }),
            "unsupported skinny GEMM shape: {shape:?}"
        );
        Ok(LaunchConfig {
            grid_dim: (
                if tiny {
                    shape.output
                } else {
                    shape.output.div_ceil(16)
                } as u32,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: if tiny {
                4 * size_of::<f32>() as u32
            } else {
                40960
            },
        })
    }

    pub fn decode_impl(&self) -> &'static str {
        let choices = self.choices.borrow();
        if choices.values().any(|(choice, _)| {
            choice.shape.rows == 1 && choice.implementation == LinearImpl::Skinny
        }) {
            "calibrated cuBLAS/skinny BF16, FP32 accumulation"
        } else {
            "calibrated cuBLAS BF16, FP32 accumulation"
        }
    }

    fn cublas(
        &self,
        w: &impl DevicePtr<bf16>,
        x: &CudaView<'_, bf16>,
        y: &mut CudaViewMut<'_, bf16>,
        shape: LinearShape,
        algo_index: u32,
    ) -> Result<()> {
        let LinearShape {
            rows,
            output,
            input,
        } = shape;
        if rows == 1 {
            // This branch casts the dimensions to i32.
            checked_dims(rows, output, input)?;
            let cfg = GemmConfig {
                transa: bs::cublasOperation_t::CUBLAS_OP_T,
                transb: bs::cublasOperation_t::CUBLAS_OP_N,
                m: output as i32,
                n: 1,
                k: input as i32,
                alpha: bf16::ONE,
                lda: input as i32,
                ldb: input as i32,
                beta: bf16::ZERO,
                ldc: output as i32,
            };
            unsafe {
                self.decode.gemm(cfg, w, x, y)?;
            }
        } else {
            let plans = self.plans.borrow();
            let p = plans
                .get(&(rows, output, input, algo_index, false))
                .context("Lt shape was not prepared")?;
            let (a, _ra) = w.device_ptr(&self.stream);
            let (b, _rb) = x.device_ptr(&self.stream);
            let (c, _rc) = y.device_ptr_mut(&self.stream);
            let workspace_owner = self.workspace.borrow();
            let (work, _rw) = workspace_owner.device_ptr(&self.stream);
            let alpha = 1f32;
            let beta = 0f32;
            unsafe {
                lt::matmul(
                    self.handle,
                    p.desc,
                    (&alpha as *const f32).cast(),
                    (&beta as *const f32).cast(),
                    a as _,
                    p.a,
                    b as _,
                    p.b,
                    c as _,
                    p.c,
                    c as _,
                    p.c,
                    p.algo.as_ref().context("Lt algorithm missing")?,
                    work as _,
                    self.workspace.borrow().len(),
                    self.stream.cu_stream() as _,
                )?;
            }
        }
        Ok(())
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        drop_result("synchronize BLAS stream", self.stream.synchronize());
        self.plans.borrow_mut().clear();
        self.biased_plans.borrow_mut().clear();
        self.pitched_b.borrow_mut().clear();
        unsafe {
            drop_result("destroy Lt handle", lt::destroy_handle(self.handle));
        }
    }
}

fn drop_result(name: &str, result: std::result::Result<(), impl std::fmt::Debug>) {
    if let Err(error) = result {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr().lock(),
            "{name} failed during drop: {error:?}"
        );
    }
}

impl Blas {
    pub(crate) fn fp8_direct_rows(invariant_rows: usize) -> std::ops::RangeInclusive<usize> {
        let rows = match invariant_rows {
            0 => crate::mtp::MAX_VERIFY_ROWS,
            rows => rows,
        };
        1..=rows
    }

    pub(crate) fn direct_fp8_rows(&self) -> std::ops::RangeInclusive<usize> {
        Self::fp8_direct_rows(self.invariant_rows)
    }

    pub(crate) fn direct_quant_rows(&self) -> std::ops::RangeInclusive<usize> {
        if let Some(nvfp4) = &self.nvfp4 {
            1..=nvfp4.direct_max_rows()
        } else {
            self.direct_fp8_rows()
        }
    }

    pub(crate) fn bind_invariant(
        &mut self,
        ctx: &Arc<cudarc::driver::CudaContext>,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows),
            "invariant linear row bound={rows} exceeds {}",
            crate::linear_plan::MAX_SKINNY_ROWS
        );
        if self.fp8.is_some() {
            let fp8 = super::linear::Fp8Ops::for_rows(ctx, rows)?;
            fp8.require_rows(rows)?;
            self.fp8 = Some(fp8);
        }
        if let Some(nvfp4) = &self.nvfp4 {
            nvfp4.require_rows(rows)?;
        }
        self.invariant_rows = rows;
        Ok(())
    }
    pub(crate) fn enable_fp8(&mut self, ctx: &Arc<cudarc::driver::CudaContext>) -> Result<()> {
        ensure!(self.fp8.is_none(), "FP8 operators already bound");
        let fp8 = match self.invariant_rows {
            0 => crate::backend::gpu::linear::Fp8Ops::new(ctx)?,
            rows => super::linear::Fp8Ops::for_rows(ctx, rows)?,
        };
        fp8.require_rows(*self.direct_fp8_rows().end())?;
        self.fp8 = Some(fp8);
        Ok(())
    }
    pub(crate) fn enable_nvfp4(&mut self, ctx: &Arc<cudarc::driver::CudaContext>) -> Result<()> {
        ensure!(self.nvfp4.is_none(), "NVFP4 operators already bound");
        let nvfp4 = super::linear::NvFp4Ops::new(ctx)?;
        nvfp4.require_rows(crate::linear_plan::MAX_SKINNY_ROWS)?;
        self.nvfp4 = Some(nvfp4);
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn bound_plan_receipt(&self) -> Result<serde_json::Value> {
        let mut plans = Vec::new();
        for (&(rows, output, input, index, row_family), p) in self.plans.borrow().iter() {
            let id = p.identity()?;
            plans.push(serde_json::json!({
                "rows":rows,"output":output,"input":input,"algo_index":index,"row_family":row_family,
                "id":id.id,
                "tile":id.tile,
                "split_k":id.split_k,
                "reduction_scheme":id.reduction,
                "workspace_bytes":p.workspace,
                "cta_swizzling":id.swizzle,
                "custom_option":id.custom,
                "stages_id":id.stages,
                "inner_shape_id":id.inner,
                "cluster_shape_id":id.cluster,
                "opaque_algorithm":p.algo.as_ref().context("fixed Lt algorithm missing")?.data
            }));
        }
        ensure!(!plans.is_empty(), "fixed Lt plans missing");
        Ok(
            serde_json::json!({"lt":plans,"fp8_small_bound":self.fp8.is_some(),"biased_plans":self.biased_plans.borrow().len(),"measured_choices":self.choices.borrow().len()}),
        )
    }
    pub(crate) fn fp8_ops(&self) -> Result<&super::linear::Fp8Ops> {
        self.fp8.as_ref().context("FP8 operators not bound")
    }
    pub(crate) fn nvfp4_ops(&self) -> Result<&super::linear::NvFp4Ops> {
        self.nvfp4.as_ref().context("NVFP4 operators not bound")
    }
    // Setup/eager binding for the fixed BF16 Lt path.
    pub(crate) fn prepare_fixed(&self, rows: u32, output: u32, input: u32) -> Result<()> {
        let key = (rows as usize, output as usize, input as usize, 0, false);
        if !self.plans.borrow().contains_key(&key) {
            self.closure.bind_fixed()?;
            self.prepare_request_fixed(rows, output, input)?;
        }
        Ok(())
    }

    /// B leading dimension `b_ld` with logical K = `input`. The plan is not the packed `(rows, output, input)` plan.
    pub(crate) fn prepare_fixed_b_ld(
        &self,
        rows: u32,
        output: u32,
        input: u32,
        b_ld: u32,
    ) -> Result<()> {
        ensure!(
            [rows, output, input, b_ld]
                .iter()
                .all(|&n| n > 0 && n <= i32::MAX as u32)
                && b_ld >= input,
            "invalid pitched linear dimensions"
        );
        let key = (
            rows as usize,
            output as usize,
            input as usize,
            b_ld as usize,
        );
        if !self.pitched_b.borrow().contains_key(&key) {
            self.closure.bind_fixed()?;
            let plan = Plan::build(
                self.handle,
                (key.0, key.1, key.2, key.3),
                self.workspace.borrow().len(),
                None,
                self.half_dtype,
                self.half_dtype,
                PlanAlgo::Heuristic(0),
            )?;
            self.pitched_b.borrow_mut().insert(key, plan);
        }
        Ok(())
    }

    /// One opaque algorithm per (output,input) family, selected at the canonical row count.
    pub(crate) fn bind_row_family(&self, shape: LinearShape) -> Result<()> {
        let key = (shape.output, shape.input);
        ensure!(
            !self.row_families.borrow().contains_key(&key),
            "row family already bound {key:?}"
        );
        let plan = Plan::new(
            self.handle,
            shape.rows,
            shape.output,
            shape.input,
            self.workspace.borrow().len(),
            0,
            self.half_dtype,
        )?;
        let algo = plan.algo.context("canonical Lt algorithm missing")?;
        self.row_families
            .borrow_mut()
            .insert(key, (algo, self.workspace.borrow().len()));
        Ok(())
    }

    pub(crate) fn prepare_row_family(&self, shape: LinearShape) -> Result<()> {
        checked_dims(shape.rows, shape.output, shape.input)?;
        let key = (shape.rows, shape.output, shape.input, 0, true);
        if !self.plans.borrow().contains_key(&key) {
            self.closure.bind_fixed()?;
            let algo = self
                .row_families
                .borrow()
                .get(&(shape.output, shape.input))
                .map(|&(algo, _)| algo)
                .context("row family was not bound")?;
            let plan = Plan::build(
                self.handle,
                (shape.rows, shape.output, shape.input, shape.input),
                self.workspace.borrow().len(),
                None,
                self.half_dtype,
                self.half_dtype,
                PlanAlgo::Fixed(algo),
            )?;
            self.plans.borrow_mut().insert(key, plan);
        }
        Ok(())
    }

    pub(crate) fn fixed_workspace(&self, shape: LinearShape, family: bool) -> Result<usize> {
        if family {
            self.prepare_row_family(shape)?;
        } else {
            self.prepare_fixed(
                u32::try_from(shape.rows)?,
                u32::try_from(shape.output)?,
                u32::try_from(shape.input)?,
            )?;
        }
        let plans = self.plans.borrow();
        let plan = plans
            .get(&(shape.rows, shape.output, shape.input, 0, family))
            .context("fixed workspace plan missing")?;
        Ok(if plan.workspace > self.workspace.borrow().len() {
            plan.workspace
        } else {
            0
        })
    }

    /// Full replacement allocation while the old workspace remains live.
    pub(crate) fn bind_workspace(&self, bytes: usize) -> Result<()> {
        if bytes > self.workspace.borrow().len() {
            let workspace = self.stream.alloc_zeros::<u8>(bytes)?;
            *self.workspace.borrow_mut() = workspace;
        }
        Ok(())
    }

    pub(crate) fn is_not_supported(error: &anyhow::Error) -> bool {
        error
            .downcast_ref::<cudarc::cublaslt::result::CublasError>()
            .is_some_and(|e| {
                e.0 == cudarc::cublaslt::sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
            })
    }
}

pub(crate) use super::linear::{lower_wave_index, wave_multiplier};

impl Blas {
    /// Stores the lower whole-wave heuristic of this down shape at plan index 0. A tie keeps 0. `algo_count < 2` keeps 0.
    pub(crate) fn bind_lower_wave(&self, shape: LinearShape) -> Result<()> {
        let key = (shape.rows, shape.output, shape.input, 0u32, false);
        let (waves, algo_count) = {
            let plans = self.plans.borrow();
            let plan = plans.get(&key).context("fixed linear cost plan missing")?;
            (f64::from(plan.waves), plan.algo_count)
        };
        if algo_count < 2 {
            return Ok(());
        }
        let alternative = Plan::new(
            self.handle,
            shape.rows,
            shape.output,
            shape.input,
            self.workspace.borrow().len(),
            1,
            self.half_dtype,
        )?;
        let index = lower_wave_index(&[waves, f64::from(alternative.waves)])?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_down_wave":{
                    "rows": shape.rows, "output": shape.output, "input": shape.input,
                    "waves0": waves, "waves1": alternative.waves, "index": index
                }})
            );
        }
        if index == 0 {
            return Ok(());
        }
        self.bind_workspace(alternative.workspace)?;
        self.plans.borrow_mut().insert(key, alternative);
        Ok(())
    }

    /// Fixed Lt plan's whole-wave work multiplier and separate split-K reduction launch count.
    pub(crate) fn fixed_plan_cost(
        &self,
        shape: LinearShape,
        describe: bool,
        family: bool,
    ) -> Result<(f64, usize)> {
        if family {
            self.prepare_row_family(shape)?;
        } else {
            self.prepare_fixed(
                u32::try_from(shape.rows)?,
                u32::try_from(shape.output)?,
                u32::try_from(shape.input)?,
            )?;
        }
        let plans = self.plans.borrow();
        let plan = plans
            .get(&(shape.rows, shape.output, shape.input, 0, family))
            .context("fixed linear cost plan missing")?;
        let waves = f64::from(plan.waves);
        if describe {
            plan.describe(shape.rows, shape.output, shape.input)?;
            if let Some(&(_, original_workspace)) = self
                .row_families
                .borrow()
                .get(&(shape.output, shape.input))
                .filter(|_| family)
            {
                let previous = Plan::new(
                    self.handle,
                    shape.rows,
                    shape.output,
                    shape.input,
                    original_workspace,
                    0,
                    self.half_dtype,
                )?;
                if crate::backend::diagnostics_enabled() {
                    eprintln!(
                        "{}",
                        serde_json::json!({"row_family_comparison":{
                            "rows":shape.rows,"output":shape.output,"input":shape.input,
                            "bound":plan.identity()?,"heuristic":previous.identity()?,
                            "bound_workspace":plan.workspace,"heuristic_workspace":previous.workspace,
                            "bound_waves":plan.waves,"heuristic_waves":previous.waves
                        }})
                    );
                }
            }
        }
        let split =
            plan.attr(sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?;
        Ok((wave_multiplier(waves)?, usize::from(split > 1)))
    }

    /// Request-bound BF16 Lt plans have no measured choices or calibration files.
    pub(crate) fn prepare_request_fixed(&self, rows: u32, output: u32, input: u32) -> Result<()> {
        ensure!(
            [rows, output, input]
                .iter()
                .all(|&n| n > 0 && n <= i32::MAX as u32),
            "invalid fixed linear dimensions"
        );
        let key = (rows as usize, output as usize, input as usize, 0, false);
        if (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&key.0)
            && (input.is_multiple_of(256) || key.0 <= self.invariant_rows)
        {
            self.prepare_skinny(LinearShape {
                rows: key.0,
                output: key.1,
                input: key.2,
            })?;
        }
        if !self.plans.borrow().contains_key(&key) {
            let p = Plan::new(
                self.handle,
                key.0,
                key.1,
                key.2,
                self.workspace.borrow().len(),
                0,
                self.half_dtype,
            )?;
            self.plans.borrow_mut().insert(key, p);
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn fixed_plan_count(&self) -> usize {
        self.plans.borrow().len()
    }
    #[cfg(test)]
    pub(super) fn has_fixed_plan(&self, rows: u32) -> bool {
        self.plans.borrow().keys().any(|key| key.0 == rows as usize)
    }
    #[inline(always)]
    pub(crate) fn linear_f32(
        &self,
        weight: super::contracts::Tensor<f32, 2>,
        x: super::contracts::Tensor<f32, 2>,
        y: &mut super::contracts::TensorMut<f32, 2>,
    ) -> Result<()> {
        linear_f32_on(&self.decode, weight, x, y)
    }

    pub(crate) fn linear_f16(
        &self,
        weight: super::contracts::Tensor<half::f16, 2>,
        x: super::contracts::Tensor<half::f16, 2>,
        y: &mut super::contracts::TensorMut<half::f16, 2>,
        accumulate: bool,
    ) -> Result<()> {
        weight.validate()?;
        x.validate()?;
        y.0.validate()?;
        let rows = x.shape[0] as usize;
        let input = x.shape[1] as usize;
        let output = weight.shape[0] as usize;
        checked_dims(rows, output, input)?;
        ensure!(
            weight.shape == [output as u32, input as u32]
                && weight.stride == [input as u64, 1]
                && x.stride == [input as u64, 1]
                && y.0.shape == [rows as u32, output as u32]
                && y.0.stride == [output as u64, 1],
            "FP16 linear requires dense [output,input] x [rows,input] -> [rows,output]"
        );
        if self.f16_reference {
            return self.linear_f16_reference(weight, x, y, accumulate);
        }
        let alpha = 1f32;
        let beta = if accumulate { 1f32 } else { 0f32 };
        unsafe {
            bs::cublasGemmEx(
                *self.decode.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                output as i32,
                rows as i32,
                input as i32,
                (&alpha as *const f32).cast(),
                weight.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16F,
                input as i32,
                x.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16F,
                input as i32,
                (&beta as *const f32).cast(),
                y.0.ptr as *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16F,
                output as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        Ok(())
    }

    /// Binds subsequent F16 linears to the heuristic first result, the
    /// reference library's algorithm pick, instead of the legacy GEMM entry.
    pub(crate) fn enable_f16_reference(&mut self) {
        self.f16_reference = true;
    }

    fn linear_f16_reference(
        &self,
        weight: super::contracts::Tensor<half::f16, 2>,
        x: super::contracts::Tensor<half::f16, 2>,
        y: &mut super::contracts::TensorMut<half::f16, 2>,
        accumulate: bool,
    ) -> Result<()> {
        let shape = LinearShape {
            rows: x.shape[0] as usize,
            output: weight.shape[0] as usize,
            input: weight.shape[1] as usize,
        };
        {
            let mut plans = self.f16_plans.borrow_mut();
            if let std::collections::btree_map::Entry::Vacant(entry) = plans.entry(shape) {
                let plan = Plan::new_f16(
                    self.handle,
                    shape.rows,
                    shape.output,
                    shape.input,
                    1 << 20,
                    0,
                )?;
                entry.insert(plan);
            }
        }
        let plans = self.f16_plans.borrow();
        let p = plans.get(&shape).context("F16 reference plan missing")?;
        let workspace_owner = self.workspace.borrow();
        let (workspace, _read) = workspace_owner.device_ptr(&self.stream);
        let alpha = 1f32;
        let beta = if accumulate { 1f32 } else { 0f32 };
        unsafe {
            lt::matmul(
                self.handle,
                p.desc,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                weight.ptr as _,
                p.a,
                x.ptr as _,
                p.b,
                y.0.ptr as _,
                p.c,
                y.0.ptr as _,
                p.c,
                p.algo.as_ref().context("F16 reference algo missing")?,
                workspace as _,
                workspace_owner.len(),
                self.stream.cu_stream() as _,
            )?;
        }
        Ok(())
    }

    pub(crate) fn prepare_f16_biased(&mut self) -> Result<()> {
        if self.bias_fill_f16.is_none() {
            self.bias_fill_f16 = Some(
                cubin::module(self.stream.context(), "q38_common")?
                    .load_function("q38_bias_fill_f16")?,
            );
        }
        Ok(())
    }

    pub(crate) fn linear_f16_biased(
        &self,
        weight: super::contracts::Tensor<half::f16, 2>,
        bias: super::contracts::Tensor<half::f16, 1>,
        x: super::contracts::Tensor<half::f16, 2>,
        y: &mut super::contracts::TensorMut<half::f16, 2>,
    ) -> Result<()> {
        Self::validate_f16_bias(weight, bias, x, y)?;
        let kernel = self
            .bias_fill_f16
            .as_ref()
            .context("F16 bias fill not prepared")?;
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&y.0.ptr)
                .arg(&bias.ptr)
                .arg(&(u64::from(y.0.shape[0]) * u64::from(y.0.shape[1])))
                .arg(&y.0.shape[1])
                .launch(LaunchConfig {
                    grid_dim: (
                        (u64::from(y.0.shape[0]) * u64::from(y.0.shape[1]))
                            .div_ceil(256)
                            .min(65535) as u32,
                        1,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.linear_f16(weight, x, y, true)
    }

    fn validate_f16_bias(
        weight: super::contracts::Tensor<half::f16, 2>,
        bias: super::contracts::Tensor<half::f16, 1>,
        x: super::contracts::Tensor<half::f16, 2>,
        y: &super::contracts::TensorMut<half::f16, 2>,
    ) -> Result<()> {
        bias.validate()?;
        super::linear::validate_call(weight.shape, x, &y.0)?;
        ensure!(
            bias.shape == [weight.shape[0]] && bias.stride == [1],
            "F16 bias extent"
        );
        super::linear::disjoint(&bias, &y.0)?;
        super::linear::disjoint(&weight, &y.0)
    }

    pub(crate) fn linear_f16_prefilled(
        &self,
        weight: super::contracts::Tensor<half::f16, 2>,
        bias: super::contracts::Tensor<half::f16, 1>,
        x: super::contracts::Tensor<half::f16, 2>,
        y: &mut super::contracts::TensorMut<half::f16, 2>,
    ) -> Result<()> {
        Self::validate_f16_bias(weight, bias, x, y)?;
        self.linear_f16(weight, x, y, true)
    }

    /// `y = bf16(y + alpha * x * weight^T)` with BF16 operands and FP32 compute. `weight` is dense
    /// `[output, input]`; `x [rows, input]` and `y [rows, output]` may have row pitches above their widths.
    pub(crate) fn linear_accumulate(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
        alpha: f32,
    ) -> Result<()> {
        self.accumulate_bound(weight, x, y, alpha, 1.0, false)
    }

    pub(crate) fn linear_row_family(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.accumulate_bound(weight, x, y, alpha, beta, true)
    }

    fn accumulate_bound(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
        alpha: f32,
        beta: f32,
        fixed: bool,
    ) -> Result<()> {
        weight.validate()?;
        x.validate()?;
        y.0.validate()?;
        let rows = x.shape[0] as usize;
        let input = x.shape[1] as usize;
        let output = weight.shape[0] as usize;
        checked_dims(rows, output, input)?;
        let (x_pitch, y_pitch) = (x.stride[0], y.0.stride[0]);
        ensure!(
            weight.shape == [output as u32, input as u32]
                && weight.stride == [input as u64, 1]
                && x.stride[1] == 1
                && x_pitch >= input as u64
                && y.0.shape == [rows as u32, output as u32]
                && y.0.stride[1] == 1
                && y_pitch >= output as u64
                && x_pitch <= i32::MAX as u64
                && y_pitch <= i32::MAX as u64,
            "accumulating linear requires dense [output,input] weights and unit-stride rows"
        );
        ensure!(alpha.is_finite(), "accumulating linear alpha {alpha}");
        if fixed {
            ensure!(
                beta == 0.0 || beta == 1.0,
                "linear beta must be zero or one"
            );
            super::linear::disjoint(&weight, &y.0)?;
            super::linear::disjoint(&x, &y.0)?;
            ensure!(
                [weight.ptr, x.ptr, y.0.ptr]
                    .iter()
                    .all(|p| p.is_multiple_of(256)),
                "fixed family pointer alignment"
            );
            ensure!(
                self.row_families.borrow().contains_key(&(output, input)),
                "accumulating row family not bound"
            );
            ensure!(
                x_pitch == input as u64 && y_pitch == output as u64,
                "fixed accumulating family requires dense rows"
            );
            let plans = self.plans.borrow();
            let p = plans
                .get(&(rows, output, input, 0, true))
                .context("accumulating family was not prepared")?;
            let owner = self.workspace.borrow();
            ensure!(
                p.workspace <= owner.len(),
                "fixed accumulating workspace was not bound"
            );
            let (workspace, _guard) = owner.device_ptr(&self.stream);
            unsafe {
                lt::matmul(
                    self.handle,
                    p.desc,
                    (&alpha as *const f32).cast(),
                    (&beta as *const f32).cast(),
                    weight.ptr as _,
                    p.a,
                    x.ptr as _,
                    p.b,
                    y.0.ptr as _,
                    p.c,
                    y.0.ptr as _,
                    p.c,
                    p.algo.as_ref().context("accumulating algorithm missing")?,
                    workspace as _,
                    owner.len(),
                    self.stream.cu_stream() as _,
                )?;
            }
            return Ok(());
        }
        unsafe {
            bs::cublasGemmEx(
                *self.decode.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                output as i32,
                rows as i32,
                input as i32,
                (&alpha as *const f32).cast(),
                weight.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                input as i32,
                x.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                x_pitch as i32,
                (&beta as *const f32).cast(),
                y.0.ptr as *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                y_pitch as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        Ok(())
    }

    pub(crate) fn linear_bf16(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
    ) -> Result<()> {
        weight.validate()?;
        x.validate()?;
        y.0.validate()?;
        let rows = x.shape[0] as usize;
        let input = x.shape[1] as usize;
        let output = weight.shape[0] as usize;
        checked_dims(rows, output, input)?;
        let (x_pitch, y_pitch) = (x.stride[0], y.0.stride[0]);
        ensure!(
            weight.shape == [output as u32, input as u32]
                && weight.stride == [input as u64, 1]
                && x.stride[1] == 1
                && x_pitch >= input as u64
                && y.0.shape == [rows as u32, output as u32]
                && y.0.stride[1] == 1
                && y_pitch >= output as u64
                && x_pitch <= i32::MAX as u64
                && y_pitch <= i32::MAX as u64,
            "linear_bf16 requires dense weights and unit-stride rows"
        );
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            bs::cublasGemmEx(
                *self.decode.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                output as i32,
                rows as i32,
                input as i32,
                (&alpha as *const f32).cast(),
                weight.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                input as i32,
                x.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                x_pitch as i32,
                (&beta as *const f32).cast(),
                y.0.ptr as *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                y_pitch as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        Ok(())
    }

    pub(crate) fn linear_bf16_to_f32(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<f32, 2>,
    ) -> Result<()> {
        weight.validate()?;
        x.validate()?;
        y.0.validate()?;
        let rows = x.shape[0] as usize;
        let input = x.shape[1] as usize;
        let output = weight.shape[0] as usize;
        checked_dims(rows, output, input)?;
        let (x_pitch, y_pitch) = (x.stride[0], y.0.stride[0]);
        ensure!(
            weight.shape == [output as u32, input as u32]
                && weight.stride == [input as u64, 1]
                && x.stride[1] == 1
                && x_pitch >= input as u64
                && y.0.shape == [rows as u32, output as u32]
                && y.0.stride[1] == 1
                && y_pitch >= output as u64
                && x_pitch <= i32::MAX as u64
                && y_pitch <= i32::MAX as u64,
            "linear_bf16_to_f32 requires dense weights and unit-stride rows"
        );
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            bs::cublasGemmEx(
                *self.decode.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                output as i32,
                rows as i32,
                input as i32,
                (&alpha as *const f32).cast(),
                weight.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                input as i32,
                x.ptr as *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                x_pitch as i32,
                (&beta as *const f32).cast(),
                y.0.ptr as *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_32F,
                y_pitch as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        Ok(())
    }

    pub(crate) fn linear_typed(
        &self,
        phase: crate::op_contract::ExecutionPhase,
        weight: super::linear::LinearWeightRef,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
    ) -> Result<()> {
        self.linear_typed_bound(phase, weight, x, y, 0)
    }
    pub(crate) fn linear_typed_bound(
        &self,
        phase: crate::op_contract::ExecutionPhase,
        weight: super::linear::LinearWeightRef,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
        invariant_rows: usize,
    ) -> Result<()> {
        use super::linear::LinearWeightRef;
        if phase == crate::op_contract::ExecutionPhase::Decode {
            ensure!(x.shape[0] == 1, "Decode requires M1");
        }
        match weight {
            LinearWeightRef::Fp8 {
                codes,
                scale,
                layout,
            } => {
                let direct_rows = self.direct_fp8_rows();
                ensure!(
                    direct_rows.contains(&(x.shape[0] as usize)),
                    "direct FP8 rows={} exceeds the bound={} and requires materialized BF16 weight",
                    x.shape[0],
                    direct_rows.end()
                );
                self.fp8_ops()?
                    .small_gemm(&self.stream, codes, scale, layout, x, y)
            }
            LinearWeightRef::NvFp4 {
                codes,
                scale,
                global_scale,
            } => {
                let nvfp4 = self.nvfp4_ops()?;
                let direct_rows = 1..=nvfp4.direct_max_rows();
                ensure!(
                    direct_rows.contains(&(x.shape[0] as usize)),
                    "direct NVFP4 rows={} exceeds the bound={} and requires materialized BF16 weight",
                    x.shape[0],
                    direct_rows.end()
                );
                nvfp4.gemm(&self.stream, codes, scale, global_scale, x, y)
            }
            LinearWeightRef::Bf16(w) => {
                w.validate()?;
                super::linear::validate_call(w.shape, x, &y.0)?;
                super::linear::disjoint(&w, &y.0)?;
                ensure!(
                    w.stride == [u64::from(w.shape[1]), 1],
                    "BF16 linear requires row-major weight"
                );
                if (1..=self.invariant_rows.max(invariant_rows)).contains(&(x.shape[0] as usize)) {
                    #[cfg(test)]
                    if std::env::var_os("FF_ORACLE_ROWS").is_some() {
                        eprintln!(
                            "GEMMA4_LINEAR_LAUNCH backend=skinny phase={phase:?} rows={} output={} input={}",
                            x.shape[0], w.shape[0], w.shape[1]
                        );
                    }
                    return self.skinny_raw(
                        w.ptr,
                        x.ptr,
                        y.0.ptr,
                        LinearShape {
                            rows: x.shape[0] as usize,
                            output: w.shape[0] as usize,
                            input: w.shape[1] as usize,
                        },
                    );
                }
                let plans = self.plans.borrow();
                let p = plans
                    .get(&(
                        x.shape[0] as usize,
                        w.shape[0] as usize,
                        w.shape[1] as usize,
                        0,
                        false,
                    ))
                    .with_context(|| {
                        format!(
                            "fixed Lt plan missing; bind before run/capture rows={} output={} input={}",
                            x.shape[0], w.shape[0], w.shape[1]
                        )
                    })?;
                #[cfg(test)]
                if std::env::var_os("FF_ORACLE_ROWS").is_some() {
                    eprintln!(
                        "GEMMA4_LINEAR_LAUNCH backend=lt phase={phase:?} rows={} output={} input={}",
                        x.shape[0], w.shape[0], w.shape[1]
                    );
                }
                let alpha = 1f32;
                let beta = 0f32;
                let workspace_owner = self.workspace.borrow();
                ensure!(
                    p.workspace <= workspace_owner.len(),
                    "fixed Lt workspace not bound"
                );
                let (workspace, _guard) = workspace_owner.device_ptr(&self.stream);
                unsafe {
                    lt::matmul(
                        self.handle,
                        p.desc,
                        (&alpha as *const f32).cast(),
                        (&beta as *const f32).cast(),
                        w.ptr as _,
                        p.a,
                        x.ptr as _,
                        p.b,
                        y.0.ptr as _,
                        p.c,
                        y.0.ptr as _,
                        p.c,
                        p.algo.as_ref().context("fixed Lt algo missing")?,
                        workspace as _,
                        self.workspace.borrow().len(),
                        self.stream.cu_stream() as _,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// BF16 Lt GEMM whose B leading dimension is `b_ld` and whose K stays `weight.shape[1]`.
    pub(crate) fn linear_bf16_b_ld(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
        b_ld: usize,
    ) -> Result<()> {
        weight.validate()?;
        x.validate()?;
        y.0.validate()?;
        ensure!(
            weight.stride == [u64::from(weight.shape[1]), 1],
            "BF16 linear requires row-major weight"
        );
        ensure!(
            b_ld >= weight.shape[1] as usize
                && x.shape[1] == weight.shape[1]
                && y.0.shape == [x.shape[0], weight.shape[0]]
                && x.stride == [u64::try_from(b_ld)?, 1]
                && y.0.stride == [u64::from(weight.shape[0]), 1],
            "pitched linear shape rows={} output={} input={} b_ld={b_ld}",
            x.shape[0],
            weight.shape[0],
            weight.shape[1]
        );
        super::linear::disjoint(&weight, &y.0)?;
        super::linear::disjoint(&x, &y.0)?;
        let plans = self.pitched_b.borrow();
        let p = plans
            .get(&(
                x.shape[0] as usize,
                weight.shape[0] as usize,
                weight.shape[1] as usize,
                b_ld,
            ))
            .with_context(|| {
                format!(
                    "pitched Lt plan missing; bind before run rows={} output={} input={} b_ld={b_ld}",
                    x.shape[0], weight.shape[0], weight.shape[1]
                )
            })?;
        let alpha = 1f32;
        let beta = 0f32;
        let workspace_owner = self.workspace.borrow();
        ensure!(
            p.workspace <= workspace_owner.len(),
            "pitched Lt workspace not bound"
        );
        let (workspace, _guard) = workspace_owner.device_ptr(&self.stream);
        unsafe {
            lt::matmul(
                self.handle,
                p.desc,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                weight.ptr as _,
                p.a,
                x.ptr as _,
                p.b,
                y.0.ptr as _,
                p.c,
                y.0.ptr as _,
                p.c,
                p.algo.as_ref().context("pitched Lt algo missing")?,
                workspace as _,
                self.workspace.borrow().len(),
                self.stream.cu_stream() as _,
            )?;
        }
        Ok(())
    }
}

impl Blas {
    #[cfg(test)]
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.workspace.borrow().len()
    }

    /// Request-bound BF16 Lt plans are keyed by geometry, bias pointer and rounding mode.
    pub(crate) fn prepare_biased(
        &mut self,
        rows: u32,
        output: u32,
        input: u32,
        bias: super::contracts::Tensor<bf16, 1>,
        mode: super::linear::LinearBiasMode,
    ) -> Result<()> {
        ensure!(
            [rows, output, input]
                .iter()
                .all(|&n| n > 0 && n <= i32::MAX as u32),
            "invalid biased linear dimensions"
        );
        bias.validate()?;
        ensure!(
            bias.shape == [output] && bias.stride == [1],
            "linear bias shape mismatch"
        );
        let shape = LinearShape {
            rows: rows as usize,
            output: output as usize,
            input: input as usize,
        };
        let key = (shape, bias.ptr, mode);
        let mut plans = self.biased_plans.borrow_mut();
        if let std::collections::btree_map::Entry::Vacant(entry) = plans.entry(key) {
            let plan = Plan::build(
                self.handle,
                (shape.rows, shape.output, shape.input, shape.input),
                self.workspace.borrow().len(),
                (mode == super::linear::LinearBiasMode::BeforeOutputRound).then_some(bias.ptr),
                self.half_dtype,
                self.half_dtype,
                PlanAlgo::Heuristic(0),
            )?;
            plan.describe(shape.rows, shape.output, shape.input)?;
            entry.insert(plan);
        }
        if mode == super::linear::LinearBiasMode::AfterOutputRound && self.bias_add.is_none() {
            self.bias_add = Some(
                cubin::module(self.stream.context(), "q38_common")?
                    .load_function("q38_bias_add")?,
            );
        }
        Ok(())
    }

    pub(crate) fn prepare_biased_group(
        &mut self,
        (rows, canonical_rows): (usize, usize),
        [output, input]: [u32; 2],
        bias: super::contracts::Tensor<bf16, 1>,
    ) -> Result<()> {
        let mode = super::linear::LinearBiasMode::BeforeOutputRound;
        self.prepare_biased(u32::try_from(canonical_rows)?, output, input, bias, mode)?;
        let shape = LinearShape {
            rows,
            output: output as usize,
            input: input as usize,
        };
        checked_dims(rows, shape.output, shape.input)?;
        let canonical = LinearShape {
            rows: canonical_rows,
            ..shape
        };
        let mut plans = self.biased_plans.borrow_mut();
        let algo = plans[&(canonical, bias.ptr, mode)]
            .algo
            .context("canonical biased Lt algorithm missing")?;
        if let std::collections::btree_map::Entry::Vacant(entry) =
            plans.entry((shape, bias.ptr, mode))
        {
            let workspace = self.workspace.borrow().len();
            let (ab, c) = biased_group_dtypes(self.half_dtype);
            let plan = Plan::build(
                self.handle,
                (rows, shape.output, shape.input, shape.input),
                workspace,
                Some(bias.ptr),
                ab,
                c,
                PlanAlgo::Fixed(algo),
            )?;
            ensure!(
                plan.workspace <= workspace,
                "grouped biased Lt plan exceeds workspace"
            );
            plan.describe(rows, shape.output, shape.input)?;
            entry.insert(plan);
        }
        Ok(())
    }

    /// Biased Lt plan bound at heuristic rank 0 under a 1 MiB workspace
    /// preference.
    pub(crate) fn prepare_biased_reference(
        &mut self,
        rows: u32,
        output: u32,
        input: u32,
        bias: super::contracts::Tensor<bf16, 1>,
        mode: super::linear::LinearBiasMode,
    ) -> Result<()> {
        ensure!(
            [rows, output, input]
                .iter()
                .all(|&n| n > 0 && n <= i32::MAX as u32),
            "invalid biased linear dimensions"
        );
        bias.validate()?;
        ensure!(
            bias.shape == [output] && bias.stride == [1],
            "linear bias shape mismatch"
        );
        let shape = LinearShape {
            rows: rows as usize,
            output: output as usize,
            input: input as usize,
        };
        let key = (shape, bias.ptr, mode);
        let mut plans = self.biased_plans.borrow_mut();
        if let std::collections::btree_map::Entry::Vacant(entry) = plans.entry(key) {
            let plan = Plan::build(
                self.handle,
                (shape.rows, shape.output, shape.input, shape.input),
                1 << 20,
                (mode == super::linear::LinearBiasMode::BeforeOutputRound).then_some(bias.ptr),
                self.half_dtype,
                self.half_dtype,
                PlanAlgo::Heuristic(0),
            )?;
            plan.describe(shape.rows, shape.output, shape.input)?;
            entry.insert(plan);
        }
        if mode == super::linear::LinearBiasMode::AfterOutputRound && self.bias_add.is_none() {
            self.bias_add = Some(
                cubin::module(self.stream.context(), "q38_common")?
                    .load_function("q38_bias_add")?,
            );
        }
        Ok(())
    }

    /// BF16 inputs/weights/bias and FP32 accumulation. The explicit mode selects
    /// bias before the output conversion or a subsequent BF16 bias add.
    pub(crate) fn linear_biased(
        &self,
        weight: super::contracts::Tensor<bf16, 2>,
        bias: super::contracts::Tensor<bf16, 1>,
        x: super::contracts::Tensor<bf16, 2>,
        y: &mut super::contracts::TensorMut<bf16, 2>,
        mode: super::linear::LinearBiasMode,
    ) -> Result<()> {
        weight.validate()?;
        bias.validate()?;
        super::linear::validate_call(weight.shape, x, &y.0)?;
        ensure!(
            weight.stride == [u64::from(weight.shape[1]), 1]
                && bias.shape == [weight.shape[0]]
                && bias.stride == [1],
            "biased linear layout mismatch"
        );
        super::linear::disjoint(&weight, &y.0)?;
        super::linear::disjoint(&bias, &y.0)?;
        let add = match mode {
            super::linear::LinearBiasMode::BeforeOutputRound => None,
            super::linear::LinearBiasMode::AfterOutputRound => Some(
                self.bias_add
                    .as_ref()
                    .context("bias add kernel not prepared")?,
            ),
        };
        let shape = LinearShape {
            rows: x.shape[0] as usize,
            output: weight.shape[0] as usize,
            input: weight.shape[1] as usize,
        };
        let plans = self.biased_plans.borrow();
        let p = plans
            .get(&(shape, bias.ptr, mode))
            .with_context(|| format!(
                "biased Lt plan missing rows={} output={} input={} mode={mode:?}; bind before run/capture",
                x.shape[0], weight.shape[0], weight.shape[1],
            ))?;
        let workspace_owner = self.workspace.borrow();
        let (workspace, _read) = workspace_owner.device_ptr(&self.stream);
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            lt::matmul(
                self.handle,
                p.desc,
                (&alpha as *const f32).cast(),
                (&beta as *const f32).cast(),
                weight.ptr as _,
                p.a,
                x.ptr as _,
                p.b,
                y.0.ptr as _,
                p.c,
                y.0.ptr as _,
                p.c,
                p.algo.as_ref().context("biased Lt algo missing")?,
                workspace as _,
                self.workspace.borrow().len(),
                self.stream.cu_stream() as _,
            )?;
        }
        if let Some(add) = add {
            let count = u64::from(y.0.shape[0]) * u64::from(y.0.shape[1]);
            unsafe {
                self.stream
                    .launch_builder(add)
                    .arg(&y.0.ptr)
                    .arg(&bias.ptr)
                    .arg(&count)
                    .arg(&y.0.shape[1])
                    .launch(super::ops::grid(
                        count.div_ceil(256).min(65535) as usize,
                        256,
                    ))?;
            }
        }
        Ok(())
    }
}

pub(crate) fn linear_f32_on(
    blas: &CudaBlas,
    weight: super::contracts::Tensor<f32, 2>,
    x: super::contracts::Tensor<f32, 2>,
    y: &mut super::contracts::TensorMut<f32, 2>,
) -> Result<()> {
    weight.validate()?;
    x.validate()?;
    y.0.validate()?;
    let rows = x.shape[0] as usize;
    let input = x.shape[1] as usize;
    let output = weight.shape[0] as usize;
    checked_dims(rows, output, input)?;
    let column_major = y.0.stride[0] == 1 && y.0.stride[1] >= rows as u64;
    ensure!(
        weight.shape == [output as u32, input as u32]
            && weight.stride == [input as u64, 1]
            && x.stride == [input as u64, 1]
            && y.0.shape == [rows as u32, output as u32]
            && (y.0.stride == [output as u64, 1] || column_major),
        "FP32 linear requires dense operands and row/column-major output"
    );
    let (m, n, a, b, ld) = if column_major {
        (
            rows,
            output,
            x.ptr,
            weight.ptr,
            usize::try_from(y.0.stride[1])?,
        )
    } else {
        (output, rows, weight.ptr, x.ptr, output)
    };
    ensure!(ld <= i32::MAX as usize, "FP32 linear output pitch overflow");
    let alpha = 1f32;
    let beta = 0f32;
    unsafe {
        bs::cublasGemmEx(
            *blas.handle(),
            bs::cublasOperation_t::CUBLAS_OP_T,
            bs::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            n as i32,
            input as i32,
            (&alpha as *const f32).cast(),
            a as *const core::ffi::c_void,
            bs::cudaDataType_t::CUDA_R_32F,
            input as i32,
            b as *const core::ffi::c_void,
            bs::cudaDataType_t::CUDA_R_32F,
            input as i32,
            (&beta as *const f32).cast(),
            y.0.ptr as *mut core::ffi::c_void,
            bs::cudaDataType_t::CUDA_R_32F,
            ld as i32,
            bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .result()?;
    }
    Ok(())
}

pub(crate) fn batched_f32_on(
    blas: &CudaBlas,
    left: super::contracts::Tensor<f32, 3>,
    right: super::contracts::Tensor<f32, 3>,
    output: &mut super::contracts::TensorMut<f32, 3>,
    transpose: [bool; 2],
    alpha: f32,
) -> Result<()> {
    use super::contracts::no_overlap;
    for t in [&left, &right, &output.0] {
        t.validate()?;
        ensure!(
            t.stride[2] == 1
                && t.stride[1] >= u64::from(t.shape[2])
                && (t.stride[0] >= u64::from(t.shape[1]) * t.stride[1]
                    || (t.stride[0] == u64::from(t.shape[2])
                        && t.stride[1] >= u64::from(t.shape[0]) * t.stride[0])),
            "F32 batched GEMM row/head stride"
        );
    }
    no_overlap(&left, &output.0, "left", "output")?;
    no_overlap(&right, &output.0, "right", "output")?;
    let [m, k] = if transpose[0] {
        [left.shape[2], left.shape[1]]
    } else {
        [left.shape[1], left.shape[2]]
    };
    let [rk, n] = if transpose[1] {
        [right.shape[2], right.shape[1]]
    } else {
        [right.shape[1], right.shape[2]]
    };
    let batches = left.shape[0];
    ensure!(
        k == rk
            && right.shape[0] == batches
            && output.0.shape == [batches, m, n]
            && alpha.is_finite(),
        "F32 batched GEMM shape/scale mismatch"
    );
    let trans = |value: bool| {
        if value {
            bs::cublasOperation_t::CUBLAS_OP_T
        } else {
            bs::cublasOperation_t::CUBLAS_OP_N
        }
    };
    let beta = 0f32;
    unsafe {
        bs::cublasGemmStridedBatchedEx(
            *blas.handle(),
            trans(transpose[1]),
            trans(transpose[0]),
            i32::try_from(n)?,
            i32::try_from(m)?,
            i32::try_from(k)?,
            (&alpha as *const f32).cast(),
            right.ptr as *const core::ffi::c_void,
            bs::cudaDataType_t::CUDA_R_32F,
            i32::try_from(right.stride[1])?,
            i64::try_from(right.stride[0])?,
            left.ptr as *const core::ffi::c_void,
            bs::cudaDataType_t::CUDA_R_32F,
            i32::try_from(left.stride[1])?,
            i64::try_from(left.stride[0])?,
            (&beta as *const f32).cast(),
            output.0.ptr as *mut core::ffi::c_void,
            bs::cudaDataType_t::CUDA_R_32F,
            i32::try_from(output.0.stride[1])?,
            i64::try_from(output.0.stride[0])?,
            i32::try_from(batches)?,
            bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .result()?;
    }
    Ok(())
}

impl Blas {
    pub(crate) fn batched_f32(
        &self,
        left: super::contracts::Tensor<f32, 3>,
        right: super::contracts::Tensor<f32, 3>,
        output: &mut super::contracts::TensorMut<f32, 3>,
        transpose: [bool; 2],
        alpha: f32,
    ) -> Result<()> {
        batched_f32_on(&self.decode, left, right, output, transpose, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn lt_operand_maps_to_cublas_data_type() {
        use crate::backend::setup::{LtOperand, lt_compute_operand};
        assert_eq!(
            cuda_lt_dtype(LtOperand::F16),
            sys::cudaDataType_t::CUDA_R_16F
        );
        assert_eq!(
            cuda_lt_dtype(LtOperand::Bf16),
            sys::cudaDataType_t::CUDA_R_16BF
        );
        assert_eq!(
            cuda_lt_dtype(LtOperand::F32),
            sys::cudaDataType_t::CUDA_R_32F
        );
        assert_eq!(
            cuda_lt_dtype(lt_compute_operand()),
            sys::cudaDataType_t::CUDA_R_32F
        );
    }

    #[test]
    fn grouped_biased_plan_keeps_the_fp16_operand() {
        let fp16 = sys::cudaDataType_t::CUDA_R_16F;
        assert_eq!(biased_group_dtypes(fp16), (fp16, fp16));
        let bf16 = sys::cudaDataType_t::CUDA_R_16BF;
        assert_eq!(biased_group_dtypes(bf16), (bf16, bf16));
    }

    #[test]
    #[ignore = "exclusive CUDA window and an empty FLYINGFISH_RUNTIME_DIR"]
    fn m1_calibration_binds_skinny_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
        );
        ensure!(
            std::fs::read_dir(&runtime)?.next().is_none(),
            "runtime must be empty"
        );
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let stream = &device.stream;
        for output in [32, 2048] {
            let shape = LinearShape {
                rows: 1,
                output,
                input: 2048,
            };
            let mut cold = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(super::super::closure::Closure::new(
                    true,
                    "cold M1 gate".into(),
                )),
            )?;
            cold.prepare(128, output, shape.input)?;
            assert!(cold.skinny_plans.borrow().is_empty());
            let w = stream.clone_htod(&super::super::random(output * shape.input, 13))?;
            let x = stream.clone_htod(&super::super::random(shape.input, 17))?;
            cold.calibrate_shape(&w, &x.slice(..), shape)?;
            let mut y = stream.alloc_zeros::<bf16>(output)?;
            cold.skinny(&w, &x.slice(..), &mut y.slice_mut(..), shape)?;
            let expected = stream.clone_dtoh(&y)?;
            ensure!(
                expected.iter().all(|v| v.is_finite()),
                "nonfinite M1 reference"
            );
            let path = cold.choice_path(shape);
            let mut cache: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            let skinny = serde_json::to_value(LinearImpl::Skinny)?;
            for trial in cache["trials"]
                .as_array_mut()
                .context("linear trials missing")?
            {
                let median = if trial["implementation"] == skinny {
                    1.0
                } else {
                    2.0
                };
                for key in ["median_us", "p10_us", "p90_us"] {
                    trial[key] = median.into();
                }
            }
            cache["choice"]["implementation"] = skinny;
            cache["choice"]["median_us"] = 1.0.into();
            let frozen = serde_json::to_vec(&cache)?;
            std::fs::write(&path, &frozen)?;
            for cached_calibration in [false, true] {
                let mut loaded = Blas::new(
                    &device,
                    &ops,
                    &runtime,
                    Rc::new(super::super::closure::Closure::new(
                        false,
                        "frozen M1 gate".into(),
                    )),
                )?;
                assert!(loaded.skinny_plans.borrow().is_empty());
                if cached_calibration {
                    loaded.calibrate_shape(&w, &x.slice(..), shape)?;
                } else {
                    loaded.load_choices(&std::collections::BTreeSet::from([shape]))?;
                }
                for graph in [false, true] {
                    if graph {
                        stream.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    }
                    let launch = loaded.linear(
                        &w,
                        &x.slice(..),
                        &mut y.slice_mut(..),
                        1,
                        output,
                        shape.input,
                    );
                    if graph {
                        let captured =
                            stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                        launch?;
                        captured?.context("M1 gate captured no graph")?.launch()?;
                    } else {
                        launch?;
                    }
                    let actual = stream.clone_dtoh(&y)?;
                    ensure!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "DEFECT M1 frozen Skinny output changed: {shape:?} cached_calibration={cached_calibration} graph={graph}"
                    );
                }
                ensure!(std::fs::read(&path)? == frozen, "frozen M1 cache changed");
            }
            println!(
                "PASS M1_SKINNY_BINDING weight=[{output},2048] cold=true cached_calibration=true frozen_load=true eager_graph_bits=0 fixture=planted_frozen_skinny"
            );
        }
        Ok(())
    }

    #[test]
    fn algorithm_identity_survives_heuristic_reordering() -> Result<()> {
        let a = LtAlgorithm {
            id: 7,
            tile: 20,
            split_k: 1,
            reduction: 0,
            swizzle: 0,
            custom: 0,
            stages: 16,
            inner: 0,
            cluster: 0,
        };
        let b = LtAlgorithm { tile: 21, ..a };
        let original = [LinearImpl::CublasLt(a), LinearImpl::CublasLt(b)];
        let reordered = [original[1].clone(), original[0].clone()];
        assert_eq!(algorithm_index(&original, &original[0])?, 0);
        assert_eq!(algorithm_index(&reordered, &original[0])?, 1);
        for wrong in [
            LtAlgorithm { id: 8, ..a },
            LtAlgorithm { tile: 22, ..a },
            LtAlgorithm { stages: 17, ..a },
        ] {
            assert!(algorithm_index(&original, &LinearImpl::CublasLt(wrong)).is_err());
        }
        assert!(
            algorithm_index(&[original[0].clone(), original[0].clone()], &original[0]).is_err()
        );
        Ok(())
    }

    /// Only the i32-cast paths (rows == 1 cuBLAS and skinny) are bounded.
    #[test]
    fn linear_dimensions_must_fit_i32() {
        checked_dims(1, 5120, 17408).expect("Q38 M1 shape");
        checked_dims(16, 17408, 5120).expect("skinny-bounded rows");
        checked_dims(2048, 17408, 5120).expect("multi-row Lt shape passes the pure check");
        assert!(checked_dims(1, (i32::MAX as usize) + 1, 17408).is_err());
        assert!(checked_dims(1, 5120, (i32::MAX as usize) + 1).is_err());
        assert!(checked_dims(0, 5120, 17408).is_err());
        for (output, input, tiny) in [
            (32, 2, true),
            (32, 128, true),
            (32, 130, true),
            (64, 130, true),
            (32, 2048, true),
            (48, 5120, true),
            (65, 256, false),
        ] {
            for rows in [1, 16] {
                let launch = Blas::skinny_launch(
                    LinearShape {
                        rows,
                        output,
                        input,
                    },
                    tiny,
                )
                .expect("supported skinny shape");
                assert_eq!(
                    launch.grid_dim.0 as usize,
                    if tiny { output } else { output.div_ceil(16) }
                );
                assert_eq!(launch.shared_mem_bytes, if tiny { 16 } else { 40960 });
            }
        }
        for (rows, output, input, tiny) in [
            (1, 32, 129, true),
            (1, 65, 128, false),
            (1, 32, 128, false),
            (0, 32, 128, true),
            (17, 32, 128, true),
            (1, 0, 128, true),
            (1, 32, 0, true),
        ] {
            assert!(
                Blas::skinny_launch(
                    LinearShape {
                        rows,
                        output,
                        input
                    },
                    tiny
                )
                .is_err()
            );
        }
    }

    struct Floor {
        name: &'static str,
        m: i32,
        out: i32,
        k: i32,
        tiny: i32,
        pad_k: i32,
        tile: bool,
        max_abs: f64,
        rmse: f64,
    }

    // Twice the 2026-11-19T10:35:51Z floor. eps_min 1e-6 / 1e-7 does not bind.
    const FLOOR: &[Floor] = &[
        Floor {
            name: "tiny1",
            m: 1,
            out: 8,
            k: 32,
            tiny: 1,
            pad_k: 0,
            tile: false,
            max_abs: 0.015625,
            rmse: 0.0126576966,
        },
        Floor {
            name: "tiny7",
            m: 7,
            out: 16,
            k: 128,
            tiny: 1,
            pad_k: 0,
            tile: false,
            max_abs: 0.5,
            rmse: 0.123593084,
        },
        Floor {
            name: "full16",
            m: 16,
            out: 48,
            k: 256,
            tiny: 0,
            pad_k: 0,
            tile: false,
            max_abs: 1.0,
            rmse: 0.167570942,
        },
        Floor {
            name: "m1",
            m: 1,
            out: 48,
            k: 256,
            tiny: 0,
            pad_k: 0,
            tile: false,
            max_abs: 0.375,
            rmse: 0.142660596,
        },
        Floor {
            name: "m17",
            m: 17,
            out: 32,
            k: 256,
            tiny: 0,
            pad_k: 0,
            tile: true,
            max_abs: 0.5,
            rmse: 0.154278528,
        },
        Floor {
            name: "m64",
            m: 64,
            out: 64,
            k: 512,
            tiny: 0,
            pad_k: 0,
            tile: true,
            max_abs: 1.0,
            rmse: 0.2326228,
        },
        Floor {
            name: "prod64",
            m: 64,
            out: 5376,
            k: 5376,
            tiny: 0,
            pad_k: 0,
            tile: true,
            max_abs: 4.0,
            rmse: 0.73971736,
        },
        Floor {
            name: "tail12",
            m: 8,
            out: 32,
            k: 12,
            tiny: 0,
            pad_k: 256,
            tile: true,
            max_abs: 0.125,
            rmse: 0.037559314,
        },
    ];

    struct Draw(u32);

    impl Draw {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as f32 * (1.0 / 16777216.0) * 6.0 - 3.0
        }
    }

    struct Row {
        w: u64,
        x: u64,
        y: u64,
        out: i32,
        k: i32,
        m: i32,
        tiny: i32,
    }

    fn launch_row(stream: &Arc<CudaStream>, function: &CudaFunction, row: Row) -> Result<()> {
        let (grid_x, grid_y, smem) = if row.tiny != 0 {
            (row.out as u32, row.m as u32, 16)
        } else {
            ((row.out as u32).div_ceil(16), 1, 40960)
        };
        unsafe {
            stream
                .launch_builder(function)
                .arg(&row.w)
                .arg(&row.x)
                .arg(&row.y)
                .arg(&row.out)
                .arg(&row.k)
                .arg(&(i64::from(row.k)))
                .arg(&row.m)
                .arg(&row.tiny)
                .launch(LaunchConfig {
                    grid_dim: (grid_x, grid_y, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: smem,
                })?;
        }
        Ok(())
    }

    fn rounded_pair(draw: &mut Draw, rows: i32, cols: i32, live: i32) -> (Vec<bf16>, Vec<f16>) {
        let mut b = Vec::with_capacity((rows as usize) * (cols as usize));
        let mut f = Vec::with_capacity((rows as usize) * (cols as usize));
        for _row in 0..rows {
            for col in 0..cols {
                let v = if col < live { draw.next() } else { 0.0 };
                let bits = bf16::from_f32(v);
                b.push(bits);
                f.push(f16::from_f32(bits.to_f32()));
            }
        }
        (b, f)
    }

    fn distance(fp: &[f16], bf: &[bf16]) -> (f64, f64) {
        let mut max_abs = 0.0f64;
        let mut sse = 0.0f64;
        for (a, b) in fp.iter().zip(bf) {
            let d = f64::from(a.to_f32()) - f64::from(b.to_f32());
            max_abs = max_abs.max(d.abs());
            sse += d * d;
        }
        (max_abs, (sse / fp.len() as f64).sqrt())
    }

    fn score_cell(
        stream: &Arc<CudaStream>,
        bf16_fn: &CudaFunction,
        fp16_fn: &CudaFunction,
        draw: &mut Draw,
        cell: &Floor,
    ) -> Result<bool> {
        let bf_k = if cell.pad_k > 0 { cell.pad_k } else { cell.k };
        let y_n = (cell.m as usize)
            .checked_mul(cell.out as usize)
            .context("output extent")?;
        let (hw, fw) = rounded_pair(draw, cell.out, bf_k, cell.k);
        let (hx, fx) = rounded_pair(draw, cell.m, bf_k, cell.k);
        let dw = stream.clone_htod(&hw)?;
        let dx = stream.clone_htod(&hx)?;
        let mut dy = stream.alloc_zeros::<bf16>(y_n)?;
        {
            let (wp, _wg) = dw.device_ptr(stream);
            let (xp, _xg) = dx.device_ptr(stream);
            let (yp, _yg) = dy.device_ptr_mut(stream);
            if cell.tile {
                let mut m0 = 0i32;
                while m0 < cell.m {
                    let mt = (cell.m - m0).min(16);
                    launch_row(
                        stream,
                        bf16_fn,
                        Row {
                            w: wp,
                            x: xp + (m0 as u64) * (bf_k as u64) * 2,
                            y: yp + (m0 as u64) * (cell.out as u64) * 2,
                            out: cell.out,
                            k: bf_k,
                            m: mt,
                            tiny: 0,
                        },
                    )?;
                    m0 += 16;
                }
            } else {
                launch_row(
                    stream,
                    bf16_fn,
                    Row {
                        w: wp,
                        x: xp,
                        y: yp,
                        out: cell.out,
                        k: bf_k,
                        m: cell.m,
                        tiny: cell.tiny,
                    },
                )?;
            }
        }
        let (ew, ex, fp_k) = if cell.pad_k > 0 {
            let mut pw = Vec::new();
            let mut px = Vec::new();
            for row in 0..cell.out as usize {
                let start = row * bf_k as usize;
                pw.extend_from_slice(&fw[start..start + cell.k as usize]);
            }
            for row in 0..cell.m as usize {
                let start = row * bf_k as usize;
                px.extend_from_slice(&fx[start..start + cell.k as usize]);
            }
            (stream.clone_htod(&pw)?, stream.clone_htod(&px)?, cell.k)
        } else {
            (stream.clone_htod(&fw)?, stream.clone_htod(&fx)?, bf_k)
        };
        let mut ey = stream.alloc_zeros::<f16>(y_n)?;
        {
            let (ewp, _ewg) = ew.device_ptr(stream);
            let (exp, _exg) = ex.device_ptr(stream);
            let (eyp, _eyg) = ey.device_ptr_mut(stream);
            launch_row(
                stream,
                fp16_fn,
                Row {
                    w: ewp,
                    x: exp,
                    y: eyp,
                    out: cell.out,
                    k: fp_k,
                    m: cell.m,
                    tiny: cell.tiny,
                },
            )?;
        }
        let hy = stream.clone_dtoh(&dy)?;
        let fy = stream.clone_dtoh(&ey)?;
        ensure!(
            hy.iter().all(|v| v.is_finite()) && fy.iter().all(|v| v.is_finite()),
            "non-finite output {}",
            cell.name
        );
        let (max_abs, rmse) = distance(&fy, &hy);
        let pass = max_abs <= cell.max_abs && rmse <= cell.rmse;
        println!(
            "CASE {} m={} out={} in={} tiny={} max_abs={max_abs:.8} rmse={rmse:.8} n={y_n} bound_abs={} bound_rmse={} {}",
            cell.name,
            cell.m,
            cell.out,
            cell.k,
            cell.tiny,
            cell.max_abs,
            cell.rmse,
            if pass { "PASS" } else { "DEFECT" }
        );
        Ok(pass)
    }

    #[test]
    #[ignore = "4090 FP16 skinny GEMM against the production BF16 kernel"]
    fn gemm_skinny_fp16_matches_bf16_floor_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = Device::new(0)?;
        let sm = cubin::capability(&device.ctx)?;
        let path = PathBuf::from(env!("OUT_DIR")).join(format!("gemm_skinny_fp16_sm{sm}.cubin"));
        let bytes = std::fs::read(&path).with_context(|| format!("missing fp16 cubin sm_{sm}"))?;
        let fp16_module = device
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(bytes))?;
        let fp16 = fp16_module.load_function("gemm_skinny_bf16")?;
        let bf16_module = cubin::module(&device.ctx, "gemm_skinny")?;
        let bf16_fn = bf16_module.load_function("gemm_skinny_bf16")?;
        let stream = &device.stream;
        println!("GPU sm_{sm}");
        let mut draw = Draw(1);
        let mut defects = 0usize;
        for cell in FLOOR {
            if !score_cell(stream, &bf16_fn, &fp16, &mut draw, cell)? {
                defects += 1;
            }
        }
        let (m, out, k) = (16i32, 32i32, 256i32);
        let mut flip = Draw(99);
        let mut fw = Vec::new();
        for _ in 0..(out as usize) * (k as usize) {
            fw.push(f16::from_f32(flip.next()));
        }
        let mut fx = Vec::new();
        for _ in 0..(m as usize) * (k as usize) {
            fx.push(f16::from_f32(flip.next()));
        }
        let mut ew = stream.clone_htod(&fw)?;
        let ex = stream.clone_htod(&fx)?;
        let y_n = (m as usize) * (out as usize);
        let mut ey = stream.alloc_zeros::<f16>(y_n)?;
        {
            let (ewp, _ewg) = ew.device_ptr(stream);
            let (exp, _exg) = ex.device_ptr(stream);
            let (eyp, _eyg) = ey.device_ptr_mut(stream);
            launch_row(
                stream,
                &fp16,
                Row {
                    w: ewp,
                    x: exp,
                    y: eyp,
                    out,
                    k,
                    m,
                    tiny: 0,
                },
            )?;
        }
        let fy0 = stream.clone_dtoh(&ey)?;
        let before = fw[3].to_bits();
        fw[3] = f16::from_f32(fw[3].to_f32() + 1.0);
        ensure!(
            fw[3].to_bits() != before,
            "bit flip did not change the input"
        );
        stream.memcpy_htod(&fw, &mut ew)?;
        {
            let (ewp, _ewg) = ew.device_ptr(stream);
            let (exp, _exg) = ex.device_ptr(stream);
            let (eyp, _eyg) = ey.device_ptr_mut(stream);
            launch_row(
                stream,
                &fp16,
                Row {
                    w: ewp,
                    x: exp,
                    y: eyp,
                    out,
                    k,
                    m,
                    tiny: 0,
                },
            )?;
        }
        let fy1 = stream.clone_dtoh(&ey)?;
        let changed = fy0
            .iter()
            .zip(&fy1)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("FLIP full16 changed={changed} of {y_n}");
        ensure!(changed > 0, "bit flip left the output unchanged");
        ensure!(defects == 0, "FP16_GEMM defects={defects}");
        println!("FP16_GEMM cells={} defects={defects}", FLOOR.len());
        Ok(())
    }
}
