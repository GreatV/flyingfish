use super::{Device, cubin, linear_calibrate, ops::Ops};

/// Uniform number of cuBLASLt heuristic candidates per linear shape: every
/// shape trials CublasLt(0..LT_CANDIDATES) plus Skinny; calibration picks
/// per shape. LT-AUTOTUNE (2026-10-09): K=2 captures all measured wins.
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
    ensure!(count >= 1, "cuBLASLt returned no algorithm");
    results.truncate(count as usize);
    Ok(unsafe {
        std::mem::transmute::<
            Vec<std::mem::MaybeUninit<sys::cublasLtMatmulHeuristicResult_t>>,
            Vec<sys::cublasLtMatmulHeuristicResult_t>,
        >(results)
    })
}
use crate::backend::setup::{LinearChoice, LinearImpl, LinearShape};
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
    waves: f32,
    workspace: usize,
}

impl Plan {
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

    fn describe(&self, rows: usize, output: usize, input: usize, algo_index: u32) -> Result<()> {
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
        let mut p = Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            pref: std::ptr::null_mut(),
            algo: None,
            algo_index: 0,
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
        p.a = lt::create_matrix_layout(dtype, input as u64, output as u64, input as i64)?;
        p.b = lt::create_matrix_layout(dtype, input as u64, rows as u64, input as i64)?;
        p.c = lt::create_matrix_layout(dtype, output as u64, rows as u64, output as i64)?;
        p.pref = lt::create_matmul_pref()?;
        unsafe {
            let reduction_mask =
                sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
            lt::set_matmul_pref_attribute(p.pref, sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                (&reduction_mask as *const u32).cast(), size_of::<u32>())?;
            lt::set_matmul_pref_attribute(
                p.pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&workspace as *const usize).cast(),
                size_of::<usize>(),
            )?;
            let algos = unsafe { heuristic_algos(handle, p.desc, p.a, p.b, p.c, p.pref)? };
            ensure!(
                (algo_index as usize) < algos.len(),
                "Lt heuristic returned {} algos; index {algo_index} unavailable",
                algos.len()
            );
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
    choices: RefCell<BTreeMap<LinearShape, LinearChoice>>,
    coverage: Option<(usize, std::collections::BTreeSet<(usize, usize)>)>,
    calibration: linear_calibrate::Environment,
    skinny: CudaFunction,
    plans: RefCell<BTreeMap<(usize, usize, usize, u32), Plan>>,
    handle: sys::cublasLtHandle_t,
    decode: CudaBlas,
    workspace: CudaSlice<u8>,
    stream: Arc<CudaStream>,
}

impl Blas {
    pub fn new(
        device: &Device,
        ops: &Ops,
        runtime: &Path,
        closure: Rc<super::closure::Closure>,
    ) -> Result<Self> {
        let stream = device.stream.clone();
        let calibration = linear_calibrate::Environment::new(device, ops, runtime, closure)?;
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
            skinny: cubin::module(&device.ctx, "gemm_skinny")?.load_function("gemm_skinny_bf16")?,
            plans: RefCell::new(BTreeMap::new()),
            handle,
            decode,
            workspace,
            stream,
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
            self.choices.borrow_mut().insert(shape, choice);
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
            p.describe(rows, output, input, algo_index)?;
            self.plans
                .borrow_mut()
                .insert((rows, output, input, algo_index), p);
        }
        Ok(())
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
                rows > 0 && rows <= *limit && pairs.contains(&(output, input)),
                "linear shape outside cuBLAS-only coverage: {shape:?}"
            );
        }
        let mut algo = 0u32;
        if let Some(choice) = self.choices.borrow().get(&shape) {
            match &choice.implementation {
                LinearImpl::CublasLt(index) => algo = *index,
                LinearImpl::Skinny => return self.skinny(w, x, y, shape),
                LinearImpl::Candidate(name) => {
                    anyhow::bail!("linear candidate {name} has not been registered for {shape:?}")
                }
            }
        }
        self.prepare_with(rows, output, input, algo)?;
        self.cublas(w, x, y, rows, output, input, algo)
    }

    pub fn calibrate_shape(
        &mut self,
        w: &impl DevicePtr<bf16>,
        x: &CudaView<'_, bf16>,
        shape: LinearShape,
    ) -> Result<()> {
        linear_calibrate::check_extent(shape, w.len(), x.len())?;
        self.prepare(shape.rows, shape.output, shape.input)?;
        let (rows, output, input) = (shape.rows, shape.output, shape.input);
        {
            let choice = self.calibration.load(shape, |implementation, out| {
                let mut y = out.slice_mut(..rows * output);
                match implementation {
                    LinearImpl::CublasLt(algo) => {
                        self.prepare_with(rows, output, input, *algo)?;
                        self.cublas(w, x, &mut y, rows, output, input, *algo)
                    }
                    LinearImpl::Skinny => self.skinny(w, x, &mut y, shape),
                    LinearImpl::Candidate(name) => {
                        anyhow::bail!("unknown linear candidate {name}")
                    }
                }
            })?;
            self.choices.borrow_mut().insert(shape, choice);
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
        ensure!(
            shape.rows > 0 && shape.rows <= 16 && shape.input.is_multiple_of(256),
            "unsupported skinny GEMM shape: {shape:?}"
        );
        let (wp, _w) = w.device_ptr(&self.stream);
        let (xp, _x) = x.device_ptr(&self.stream);
        let (yp, _y) = y.device_ptr_mut(&self.stream);
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
                .launch(LaunchConfig {
                    grid_dim: (shape.output.div_ceil(16) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 40960,
                })?;
        }
        Ok(())
    }

    pub fn decode_impl(&self) -> &'static str {
        let choices = self.choices.borrow();
        let rows: Vec<_> = choices.values().filter(|c| c.shape.rows == 1).collect();
        if rows
            .iter()
            .any(|c| matches!(c.implementation, LinearImpl::Skinny))
        {
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
        rows: usize,
        output: usize,
        input: usize,
        algo_index: u32,
    ) -> Result<()> {
        if rows == 1 {
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
