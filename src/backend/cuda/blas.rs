use super::{Device, cubin, linear_calibrate, ops::Ops};
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
use std::{cell::RefCell, collections::BTreeMap, mem::size_of, path::Path, sync::Arc};

struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    pref: sys::cublasLtMatmulPreference_t,
    algo: Option<sys::cublasLtMatmulAlgo_t>,
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
    ) -> Result<Self> {
        let mut p = Self {
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            pref: std::ptr::null_mut(),
            algo: None,
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
            let heuristic =
                lt::get_matmul_algo_heuristic(handle, p.desc, p.a, p.b, p.c, p.c, p.pref)?;
            heuristic.state.result()?;
            p.waves = heuristic.wavesCount;
            p.workspace = heuristic.workspaceSize;
            p.algo = Some(heuristic.algo);
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
            for x in [self.a, self.b, self.c] {
                if !x.is_null() {
                    lt::destroy_matrix_layout(x).expect("destroy Lt layout");
                }
            }
            if !self.desc.is_null() {
                lt::destroy_matmul_desc(self.desc).expect("destroy Lt descriptor");
            }
            if !self.pref.is_null() {
                lt::destroy_matmul_pref(self.pref).expect("destroy Lt preference");
            }
        }
    }
}

pub struct Blas {
    choices: RefCell<BTreeMap<LinearShape, LinearChoice>>,
    calibration: linear_calibrate::Environment,
    skinny: CudaFunction,
    plans: BTreeMap<(usize, usize, usize), Plan>,
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
        measured: Vec<LinearChoice>,
    ) -> Result<Self> {
        let stream = device.stream.clone();
        let calibration = linear_calibrate::Environment::new(device, ops, runtime)?;
        let mut choices = BTreeMap::new();
        for choice in measured {
            ensure!(
                choice.key == calibration.key,
                "measured linear cache identity mismatch"
            );
            ensure!(
                choice.shape.rows > 0
                    && choice.shape.output > 0
                    && choice.shape.input > 0
                    && choice.median_us.is_finite()
                    && choice.median_us > 0.0,
                "invalid measured linear choice"
            );
            ensure!(
                choices.insert(choice.shape, choice).is_none(),
                "duplicate measured linear shape/M choice"
            );
        }
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
            choices: RefCell::new(choices),
            calibration,
            skinny: cubin::module(&device.ctx, "gemm_skinny")?.load_function("gemm_skinny_bf16")?,
            plans: BTreeMap::new(),
            handle,
            decode,
            workspace,
            stream,
        })
    }

    pub fn prepare(&mut self, rows: usize, output: usize, input: usize) -> Result<()> {
        if rows > 1 && !self.plans.contains_key(&(rows, output, input)) {
            let p = Plan::new(self.handle, rows, output, input, self.workspace.len())
                .with_context(|| format!("Lt plan rows={rows} output={output} input={input}"))?;
            p.describe(rows, output, input)?;
            self.plans.insert((rows, output, input), p);
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
        if rows <= 16 && input.is_multiple_of(256) && !self.choices.borrow().contains_key(&shape) {
            let choice = self.calibration.load(shape, |implementation, out| {
                let mut y = out.slice_mut(..rows * output);
                match implementation {
                    LinearImpl::Cublas => self.cublas(w, x, &mut y, rows, output, input),
                    LinearImpl::Skinny => self.skinny(w, x, &mut y, shape),
                    LinearImpl::Candidate(name) => {
                        anyhow::bail!("unknown linear candidate {name}")
                    }
                }
            })?;
            self.choices.borrow_mut().insert(shape, choice);
        }
        if let Some(choice) = self.choices.borrow().get(&shape) {
            match &choice.implementation {
                LinearImpl::Cublas => {}
                LinearImpl::Skinny => return self.skinny(w, x, y, shape),
                LinearImpl::Candidate(name) => {
                    anyhow::bail!("linear candidate {name} has not been registered for {shape:?}")
                }
            }
        }
        self.cublas(w, x, y, rows, output, input)
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
            let p = self
                .plans
                .get(&(rows, output, input))
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
        self.stream.synchronize().expect("synchronize BLAS stream");
        self.plans.clear();
        unsafe {
            lt::destroy_handle(self.handle).expect("destroy Lt handle");
        }
    }
}
