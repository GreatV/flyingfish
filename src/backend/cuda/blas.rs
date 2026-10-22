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
    fn attr16(&self, attr: sys::cublasLtMatmulAlgoConfigAttributes_t) -> Result<u16> {
        let mut value = 0u16;
        let mut written = 0usize;
        unsafe {
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                self.algo.as_ref().context("Lt algorithm missing")?,
                attr,
                (&mut value as *mut u16).cast(),
                size_of::<u16>(),
                &mut written,
            )
            .result()?;
        }
        ensure!(
            written == size_of::<u16>(),
            "Lt attribute {attr:?} returned {written} bytes"
        );
        Ok(value)
    }

    fn identity(&self) -> Result<LtAlgorithm> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as A;
        Ok(LtAlgorithm {
            id: self.attr(A::CUBLASLT_ALGO_CONFIG_ID)?,
            tile: self.attr(A::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
            split_k: self.attr(A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
            reduction: self.attr(A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
            swizzle: self.attr(A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
            custom: self.attr(A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
            stages: self.attr(A::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
            inner: self.attr16(A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?,
            cluster: self.attr16(A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?,
        })
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
        eprintln!(
            "{}",
            serde_json::json!({"cublaslt_plan":{"rows":rows,"output":output,"input":input,
            "id":self.attr(A::CUBLASLT_ALGO_CONFIG_ID)?,"tile":self.attr(A::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
            "split_k":self.attr(A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,"reduction_scheme":self.attr(A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
            "estimated_waves":self.waves,"workspace_bytes":self.workspace}})
        );
        Ok(())
    }

    fn new(
        handle: sys::cublasLtHandle_t,
        rows: usize,
        output: usize,
        input: usize,
        workspace: usize,
        algo_index: u32,
    ) -> Result<Self> {
        Self::new_with_bias(handle, rows, output, input, workspace, algo_index, None)
    }

    fn new_with_bias(
        handle: sys::cublasLtHandle_t,
        rows: usize,
        output: usize,
        input: usize,
        workspace: usize,
        algo_index: u32,
        bias: Option<u64>,
    ) -> Result<Self> {
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
            sys::cudaDataType_t::CUDA_R_32F,
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
        let dtype = sys::cudaDataType_t::CUDA_R_16BF;
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
        p.b = lt::create_matrix_layout(dtype, input as u64, rows as u64, input as i64)?;
        p.c = lt::create_matrix_layout(dtype, output as u64, rows as u64, output as i64)?;
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
            let algos = heuristic_algos(handle, p.desc, p.a, p.b, p.c, p.pref)?;
            ensure!(
                (algo_index as usize) < algos.len(),
                "Lt heuristic returned {} algos; index {algo_index} unavailable",
                algos.len()
            );
            p.algo_count = algos.len();
            let heuristic = &algos[algo_index as usize];
            heuristic.state.result()?;
            p.waves = heuristic.wavesCount;
            p.workspace = heuristic.workspaceSize;
            p.algo = Some(heuristic.algo);
            p.algo_index = algo_index;
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

pub struct Blas {
    choices: RefCell<BTreeMap<LinearShape, (LinearChoice, u32)>>,
    coverage: Option<(usize, std::collections::BTreeSet<(usize, usize)>)>,
    calibration: linear_calibrate::Environment,
    fp8: Option<super::linear::Fp8Ops>,
    skinny: CudaFunction,
    skinny_device: (usize, usize),
    skinny_plans: RefCell<BTreeMap<(usize, usize), (LaunchConfig, bool)>>,
    invariant_rows: usize,
    plans: RefCell<BTreeMap<(usize, usize, usize, u32), Plan>>,
    biased_plans: RefCell<BTreeMap<(LinearShape, u64, super::linear::LinearBiasMode), Plan>>,
    bias_add: Option<CudaFunction>,
    handle: sys::cublasLtHandle_t,
    decode: CudaBlas,
    workspace: CudaSlice<u8>,
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
            let w = stream.clone_htod(&super::calibrate::random(output * input, 13))?;
            let x = stream.clone_htod(&super::calibrate::random(8 * input, 17))?;
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
                    .remove(&(8, output, input, 0))
                    .context("first Lt plan missing")?;
                let second = plans
                    .remove(&(8, output, input, other as u32))
                    .context("second Lt plan missing")?;
                plans.insert((8, output, input, 0), second);
                plans.insert((8, output, input, other as u32), first);
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
                .get(&(8, output, input, index))
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

    pub fn new(
        device: &Device,
        ops: &Ops,
        runtime: &Path,
        closure: Rc<super::closure::Closure>,
    ) -> Result<Self> {
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
            skinny: cubin::module(&device.ctx, "gemm_skinny")?.load_function("gemm_skinny_bf16")?,
            skinny_device: (device.info.sms, device.info.l2_bytes),
            skinny_plans: RefCell::new(BTreeMap::new()),
            invariant_rows: 0,
            plans: RefCell::new(BTreeMap::new()),
            biased_plans: RefCell::new(BTreeMap::new()),
            bias_add: None,
            handle,
            decode,
            workspace,
            stream,
            closure,
        })
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
        if (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows) && input.is_multiple_of(256) {
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
                .contains_key(&(rows, output, input, algo_index))
        {
            let p = Plan::new(
                self.handle,
                rows,
                output,
                input,
                self.workspace.len(),
                algo_index,
            )
            .with_context(|| format!("Lt plan rows={rows} output={output} input={input}"))?;
            p.describe(rows, output, input)?;
            self.plans
                .borrow_mut()
                .insert((rows, output, input, algo_index), p);
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
            .get(&(rows, output, input, 0))
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
                    .get(&(rows, output, input, index))
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
            let mut launch = Self::skinny_launch(shape)?;
            let bytes = shape
                .output
                .checked_mul(shape.input)
                .and_then(|n| n.checked_mul(size_of::<bf16>()))
                .context("skinny weight byte extent overflow")?;
            let tiny = shape.output <= 16 * 4
                && (launch.grid_dim.0 as usize) < self.skinny_device.0
                && bytes <= self.skinny_device.1;
            if tiny {
                launch.grid_dim = (shape.output as u32, 1, 1);
                launch.shared_mem_bytes = 4 * size_of::<f32>() as u32;
            }
            entry.insert((launch, tiny));
        }
        Ok(())
    }
    pub(super) fn skinny_launch(shape: LinearShape) -> Result<LaunchConfig> {
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&shape.rows)
                && shape.input.is_multiple_of(256),
            "unsupported skinny GEMM shape: {shape:?}"
        );
        checked_dims(shape.rows, shape.output, shape.input)?;
        Ok(LaunchConfig {
            grid_dim: (shape.output.div_ceil(16) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 40960,
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
                .get(&(rows, output, input, algo_index))
                .context("Lt shape was not prepared")?;
            let (a, _ra) = w.device_ptr(&self.stream);
            let (b, _rb) = x.device_ptr(&self.stream);
            let (c, _rc) = y.device_ptr_mut(&self.stream);
            let (work, _rw) = self.workspace.device_ptr(&self.stream);
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
                    self.workspace.len(),
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
    pub(crate) fn direct_fp8_rows(&self) -> std::ops::RangeInclusive<usize> {
        let rows = match self.invariant_rows {
            0 => crate::mtp::MAX_VERIFY_ROWS,
            rows => rows,
        };
        1..=rows
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
        self.invariant_rows = rows;
        Ok(())
    }
    pub(crate) fn enable_fp8(&mut self, ctx: &Arc<cudarc::driver::CudaContext>) -> Result<()> {
        ensure!(self.fp8.is_none(), "FP8 operators already bound");
        let fp8 = match self.invariant_rows {
            0 => super::linear::Fp8Ops::new(ctx)?,
            rows => super::linear::Fp8Ops::for_rows(ctx, rows)?,
        };
        fp8.require_rows(*self.direct_fp8_rows().end())?;
        self.fp8 = Some(fp8);
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn bound_plan_receipt(&self) -> Result<serde_json::Value> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as A;
        let mut plans = Vec::new();
        for (&(rows, output, input, index), p) in self.plans.borrow().iter() {
            plans.push(serde_json::json!({
                "rows":rows,"output":output,"input":input,"algo_index":index,
                "id":p.attr(A::CUBLASLT_ALGO_CONFIG_ID)?,
                "tile":p.attr(A::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
                "split_k":p.attr(A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
                "reduction_scheme":p.attr(A::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
                "workspace_bytes":p.workspace,
                "cta_swizzling":p.attr(A::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
                "custom_option":p.attr(A::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
                "stages_id":p.attr(A::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
                "inner_shape_id":p.attr16(A::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?,
                "cluster_shape_id":p.attr16(A::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?,
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
    // Setup/eager binding for the fixed BF16 Lt path.
    pub(crate) fn prepare_fixed(&self, rows: u32, output: u32, input: u32) -> Result<()> {
        let key = (rows as usize, output as usize, input as usize, 0);
        if !self.plans.borrow().contains_key(&key) {
            self.closure.bind_fixed()?;
            self.prepare_request_fixed(rows, output, input)?;
        }
        Ok(())
    }

    /// Request-bound BF16 Lt plans have no measured choices or calibration files.
    pub(crate) fn prepare_request_fixed(&self, rows: u32, output: u32, input: u32) -> Result<()> {
        ensure!(
            [rows, output, input]
                .iter()
                .all(|&n| n > 0 && n <= i32::MAX as u32),
            "invalid fixed linear dimensions"
        );
        let key = (rows as usize, output as usize, input as usize, 0);
        if (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&key.0) && input.is_multiple_of(256) {
            self.prepare_skinny(LinearShape {
                rows: key.0,
                output: key.1,
                input: key.2,
            })?;
        }
        if !self.plans.borrow().contains_key(&key) {
            let p = Plan::new(self.handle, key.0, key.1, key.2, self.workspace.len(), 0)?;
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
            LinearWeightRef::Bf16(w) => {
                w.validate()?;
                super::linear::validate_call(w.shape, x, &y.0)?;
                super::linear::disjoint(&w, &y.0)?;
                ensure!(
                    w.stride == [u64::from(w.shape[1]), 1],
                    "BF16 linear requires row-major weight"
                );
                if (1..=self.invariant_rows.max(invariant_rows)).contains(&(x.shape[0] as usize)) {
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
                    ))
                    .context("fixed Lt plan missing; bind before run/capture")?;
                let (workspace, _r) = self.workspace.device_ptr(&self.stream);
                let alpha = 1f32;
                let beta = 0f32;
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
                        self.workspace.len(),
                        self.stream.cu_stream() as _,
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl Blas {
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
            let plan = Plan::new_with_bias(
                self.handle,
                shape.rows,
                shape.output,
                shape.input,
                self.workspace.len(),
                0,
                (mode == super::linear::LinearBiasMode::BeforeOutputRound).then_some(bias.ptr),
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
        let (workspace, _read) = self.workspace.device_ptr(&self.stream);
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
                self.workspace.len(),
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

#[cfg(test)]
mod tests {
    use super::*;

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
            let w =
                stream.clone_htod(&super::super::calibrate::random(output * shape.input, 13))?;
            let x = stream.clone_htod(&super::super::calibrate::random(shape.input, 17))?;
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
    }
}
