//! BF16 HIP library plans are bound by library algorithm index, not heuristic order.
use crate::backend::gpu::linear::LinearBiasMode;
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg,
};
use crate::backend::{hip::drop_result, setup::LinearShape};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::{CStr, c_char, c_void},
    hash::Hasher,
    io::Read,
    sync::Arc,
};

type Handle = *mut c_void;
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Algorithm {
    data: [u8; 16],
    max_workspace_bytes: usize,
}
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Heuristic {
    algorithm: Algorithm,
    workspace: usize,
    status: i32,
    waves: f32,
    reserved: [i32; 4],
}
const _: () = assert!(size_of::<Algorithm>() == 24 && align_of::<Algorithm>() == 8);
const _: () = assert!(size_of::<Heuristic>() == 56 && align_of::<Heuristic>() == 8);

unsafe extern "C" {
    fn hipblasCreate(handle: *mut Handle) -> i32;
    fn hipblasDestroy(handle: Handle) -> i32;
    fn hipblasSetStream(handle: Handle, stream: Handle) -> i32;
    fn hipblasGemmEx(
        handle: Handle,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        at: i32,
        lda: i32,
        b: *const c_void,
        bt: i32,
        ldb: i32,
        beta: *const c_void,
        c: Handle,
        ct: i32,
        ldc: i32,
        compute: i32,
        algo: i32,
    ) -> i32;
    fn hipblasGemmBatchedEx(
        handle: Handle,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const *const c_void,
        at: i32,
        lda: i32,
        b: *const *const c_void,
        bt: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut Handle,
        ct: i32,
        ldc: i32,
        batch: i32,
        compute: i32,
        algo: i32,
    ) -> i32;
    fn hipblasLtCreate(handle: *mut Handle) -> i32;
    fn hipblasLtDestroy(handle: Handle) -> i32;
    fn hipblasLtGetVersion(handle: Handle, version: *mut i32) -> i32;
    fn hipblasLtGetGitRevision(handle: Handle, revision: *mut c_char) -> i32;
    fn hipblasLtMatmulDescCreate(desc: *mut Handle, compute: i32, scale: i32) -> i32;
    fn hipblasLtMatmulDescDestroy(desc: Handle) -> i32;
    fn hipblasLtMatmulDescSetAttribute(
        desc: Handle,
        attr: i32,
        value: *const c_void,
        bytes: usize,
    ) -> i32;
    fn hipblasLtMatrixLayoutCreate(
        layout: *mut Handle,
        dtype: i32,
        rows: u64,
        columns: u64,
        ld: i64,
    ) -> i32;
    fn hipblasLtMatrixLayoutDestroy(layout: Handle) -> i32;
    fn hipblasLtMatmulPreferenceCreate(preference: *mut Handle) -> i32;
    fn hipblasLtMatmulPreferenceDestroy(preference: Handle) -> i32;
    fn hipblasLtMatmulPreferenceSetAttribute(
        pref: Handle,
        attr: i32,
        value: *const c_void,
        bytes: usize,
    ) -> i32;
    fn hipblasLtMatmulAlgoGetHeuristic(
        handle: Handle,
        desc: Handle,
        a: Handle,
        b: Handle,
        c: Handle,
        d: Handle,
        pref: Handle,
        count: i32,
        output: *mut Heuristic,
        returned: *mut i32,
    ) -> i32;
    fn hipblasLtMatmul(
        handle: Handle,
        desc: Handle,
        alpha: *const c_void,
        a: *const c_void,
        alayout: Handle,
        b: *const c_void,
        blayout: Handle,
        beta: *const c_void,
        c: *const c_void,
        clayout: Handle,
        d: Handle,
        dlayout: Handle,
        algorithm: *const Algorithm,
        workspace: Handle,
        workspace_bytes: usize,
        stream: Handle,
    ) -> i32;
    fn ff_hip_lt_index(algorithm: *mut Algorithm) -> i32;
    fn ff_hip_lt_supported(
        handle: Handle,
        desc: Handle,
        a: Handle,
        b: Handle,
        c: Handle,
        algorithm: *mut Algorithm,
        workspace: *mut usize,
    ) -> i32;
}
fn check(status: i32, name: &str) -> Result<()> {
    ensure!(status == 0, "{name} failed: hipBLAS status {status}");
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub backend: String,
    pub architecture: String,
    pub device_uuid: String,
    pub driver: i32,
    pub runtime: i32,
    pub library: i32,
    pub revision: String,
    pub binary: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Implementation {
    Library,
    Wmma,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub identity: Identity,
    pub shape: LinearShape,
    pub bias: bool,
    pub algorithm: Option<i32>,
    pub workspace: usize,
    pub implementation: Implementation,
    pub setup_median_ms: Option<[f64; 2]>,
}

struct Plan {
    context: Arc<crate::backend::hip::Context>,
    desc: Handle,
    a: Handle,
    b: Handle,
    c: Handle,
    preference: Handle,
    algorithm: Algorithm,
    choice: Choice,
    _bias_owner: Option<Arc<CudaSlice<bf16>>>,
    bias_binding: Option<u64>,
    output_f32: bool,
}
// Published descriptors are immutable; mutation requires exclusive Arc ownership.
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}
impl Drop for Plan {
    fn drop(&mut self) {
        drop_result(self.context.bind_to_thread());
        unsafe {
            for layout in [self.a, self.b, self.c] {
                if !layout.is_null() {
                    drop_result(check(
                        hipblasLtMatrixLayoutDestroy(layout),
                        "hipblasLtMatrixLayoutDestroy",
                    ));
                }
            }
            if !self.desc.is_null() {
                drop_result(check(
                    hipblasLtMatmulDescDestroy(self.desc),
                    "hipblasLtMatmulDescDestroy",
                ));
            }
            if !self.preference.is_null() {
                drop_result(check(
                    hipblasLtMatmulPreferenceDestroy(self.preference),
                    "hipblasLtMatmulPreferenceDestroy",
                ));
            }
        }
    }
}

struct Handles {
    handle: Handle,
    lt: Handle,
    stream: Arc<CudaStream>,
}
// Blas alone submits library calls; graph references retain ownership for destruction.
unsafe impl Send for Handles {}
unsafe impl Sync for Handles {}
impl Drop for Handles {
    fn drop(&mut self) {
        drop_result(self.stream.synchronize());
        unsafe {
            if !self.lt.is_null() {
                drop_result(check(hipblasLtDestroy(self.lt), "hipblasLtDestroy"));
            }
            if !self.handle.is_null() {
                drop_result(check(hipblasDestroy(self.handle), "hipblasDestroy"));
            }
        }
    }
}

type Binding = (LinearShape, Option<u64>, LinearBiasMode);

pub struct Blas {
    stream: Arc<CudaStream>,
    handles: Arc<Handles>,
    identity: Identity,
    workspace: CudaSlice<u8>,
    skinny: Option<CudaFunction>,
    bias_add: Option<CudaFunction>,
    rounded_output: RefCell<Option<CudaSlice<f32>>>,
    fp8: Option<crate::backend::gpu::linear::Fp8Ops>,
    invariant_rows: usize,
    plans: RefCell<BTreeMap<Binding, Arc<Plan>>>,
}
impl Blas {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        let p = &stream.context().properties;
        let identity = Identity {
            backend: "hip".into(),
            architecture: stream.context().architecture().into(),
            device_uuid: p.uuid.iter().map(|b| format!("{b:02x}")).collect(),
            driver: p.driver_version,
            runtime: p.runtime_version,
            library: 0,
            revision: String::new(),
            binary: binary_key()?,
        };
        let workspace = stream.alloc_zeros(4 * 1024 * 1024)?;
        let skinny = if stream.context().supports_wmma() {
            let function = crate::backend::gpu::cubin::module(stream.context(), "gemm_skinny")?
                .load_function("gemm_skinny_bf16")?;
            ensure!(
                function.max_threads_per_block()? >= 128
                    && p.shared_bytes >= 8192
                    && function.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)? > 0,
                "HIP BF16 WMMA requires 128 threads and 8192 bytes resident LDS"
            );
            Some(function)
        } else {
            None
        };
        let mut result = Self {
            handles: Arc::new(Handles {
                stream: stream.clone(),
                handle: std::ptr::null_mut(),
                lt: std::ptr::null_mut(),
            }),
            stream,
            identity,
            workspace,
            skinny,
            bias_add: None,
            rounded_output: RefCell::new(None),
            fp8: None,
            invariant_rows: 0,
            plans: RefCell::new(BTreeMap::new()),
        };
        let handles =
            Arc::get_mut(&mut result.handles).context("HIP library handles already shared")?;
        unsafe {
            check(hipblasCreate(&mut handles.handle), "hipblasCreate")?;
            check(
                hipblasSetStream(handles.handle, result.stream.raw()),
                "hipblasSetStream",
            )?;
            check(hipblasLtCreate(&mut handles.lt), "hipblasLtCreate")?;
            check(
                hipblasLtGetVersion(handles.lt, &mut result.identity.library),
                "hipblasLtGetVersion",
            )?;
            let mut revision = [0i8; 128];
            check(
                hipblasLtGetGitRevision(handles.lt, revision.as_mut_ptr()),
                "hipblasLtGetGitRevision",
            )?;
            result.identity.revision = CStr::from_ptr(revision.as_ptr()).to_str()?.into();
        }
        ensure!(result.identity.library > 0, "hipBLASLt version is invalid");
        result.stream.register(&result.handles);
        Ok(result)
    }

    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn enable_fp8(
        &mut self,
        context: &Arc<crate::backend::gpu::CudaContext>,
    ) -> Result<()> {
        ensure!(self.fp8.is_none(), "FP8 operators already bound");
        ensure!(
            Arc::ptr_eq(context, self.stream.context()),
            "FP8 context differs from HIP BLAS"
        );
        self.fp8 = Some(crate::backend::gpu::linear::Fp8Ops::new(context)?);
        Ok(())
    }

    pub(crate) fn fp8_ops(&self) -> Result<&crate::backend::gpu::linear::Fp8Ops> {
        self.fp8.as_ref().context("FP8 operators not bound")
    }

    pub fn prepare_choice(
        &self,
        shape: LinearShape,
        bias: Option<Arc<CudaSlice<bf16>>>,
        frozen: Option<&Choice>,
    ) -> Result<Choice> {
        if let Some(bias) = &bias {
            bias.check_stream(&self.stream)?;
            ensure!(bias.len() == shape.output, "HIP BF16 bias width mismatch");
        }
        let pointer = bias.as_ref().map(|b| b.device_ptr(&self.stream).0);
        self.prepare_binding(
            shape,
            pointer,
            bias,
            LinearBiasMode::BeforeOutputRound,
            frozen,
        )
    }

    fn prepare_binding(
        &self,
        shape: LinearShape,
        bias: Option<u64>,
        owner: Option<Arc<CudaSlice<bf16>>>,
        rounding: LinearBiasMode,
        frozen: Option<&Choice>,
    ) -> Result<Choice> {
        self.stream.require_eager("library plan preparation")?;
        self.stream.context().bind_to_thread()?;
        ensure!(
            shape.rows > 0
                && shape.output > 0
                && shape.input > 0
                && [shape.rows, shape.output, shape.input]
                    .iter()
                    .all(|&n| n <= i32::MAX as usize),
            "HIP linear dimensions exceed i32: {shape:?}"
        );
        if let Some(frozen) = frozen {
            ensure!(
                frozen.identity == self.identity
                    && frozen.shape == shape
                    && frozen.bias == bias.is_some(),
                "HIP frozen algorithm identity/shape/epilogue mismatch"
            );
            self.validate_choice(frozen)?;
        }
        let key = (shape, bias, rounding);
        if let Some(plan) = self.plans.borrow().get(&key) {
            if let Some(frozen) = frozen {
                ensure!(
                    &plan.choice == frozen,
                    "HIP frozen choice differs from prepared plan"
                );
            }
            return Ok(plan.choice.clone());
        }
        let mut plan = Plan {
            context: self.stream.context().clone(),
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
            algorithm: Algorithm::default(),
            choice: Choice {
                identity: self.identity.clone(),
                shape,
                bias: bias.is_some(),
                algorithm: None,
                workspace: 0,
                implementation: frozen.map_or(Implementation::Library, |c| c.implementation),
                setup_median_ms: frozen.and_then(|c| c.setup_median_ms),
            },
            _bias_owner: owner,
            bias_binding: bias.filter(|_| rounding == LinearBiasMode::BeforeOutputRound),
            output_f32: bias.is_some() && rounding == LinearBiasMode::AfterOutputRound,
        };
        if !plan.output_f32 && (shape.rows > 1 || bias.is_some()) {
            self.prepare_lt(&mut plan, frozen)?;
        }
        if let Some(frozen) = frozen {
            ensure!(
                &plan.choice == frozen,
                "HIP rebound algorithm/workspace differs from frozen choice"
            );
        }
        let choice = plan.choice.clone();
        self.plans.borrow_mut().insert(key, Arc::new(plan));
        Ok(choice)
    }

    fn prepare_lt(&self, plan: &mut Plan, frozen: Option<&Choice>) -> Result<()> {
        let s = plan.choice.shape;
        unsafe {
            check(
                hipblasLtMatmulDescCreate(&mut plan.desc, 2, 0),
                "hipblasLtMatmulDescCreate",
            )?;
            let transpose = 112i32;
            desc_attribute(plan.desc, 0, &transpose)?;
            if let Some(bias) = plan.bias_binding {
                desc_attribute(plan.desc, 2, &4u32)?;
                desc_attribute(plan.desc, 3, &bias)?;
                desc_attribute(plan.desc, 4, &14i32)?;
            }
            check(
                hipblasLtMatrixLayoutCreate(
                    &mut plan.a,
                    14,
                    s.input as u64,
                    s.output as u64,
                    s.input as i64,
                ),
                "hipblasLtMatrixLayoutCreate(A)",
            )?;
            check(
                hipblasLtMatrixLayoutCreate(
                    &mut plan.b,
                    14,
                    s.input as u64,
                    s.rows as u64,
                    s.input as i64,
                ),
                "hipblasLtMatrixLayoutCreate(B)",
            )?;
            check(
                hipblasLtMatrixLayoutCreate(
                    &mut plan.c,
                    14,
                    s.output as u64,
                    s.rows as u64,
                    s.output as i64,
                ),
                "hipblasLtMatrixLayoutCreate(C)",
            )?;
            check(
                hipblasLtMatmulPreferenceCreate(&mut plan.preference),
                "hipblasLtMatmulPreferenceCreate",
            )?;
            let workspace = self.workspace.len() as u64;
            check(
                hipblasLtMatmulPreferenceSetAttribute(
                    plan.preference,
                    1,
                    (&workspace as *const u64).cast(),
                    8,
                ),
                "hipblasLtMatmulPreferenceSetAttribute(workspace)",
            )?;
            let mut candidates = [Heuristic::default(); 64];
            let mut count = 0;
            check(
                hipblasLtMatmulAlgoGetHeuristic(
                    self.handles.lt,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    plan.c,
                    plan.preference,
                    candidates.len() as i32,
                    candidates.as_mut_ptr(),
                    &mut count,
                ),
                "hipblasLtMatmulAlgoGetHeuristic",
            )?;
            ensure!(
                (1..=candidates.len() as i32).contains(&count),
                "hipBLASLt heuristic returned invalid candidate count {count}"
            );
            let mut choices = Vec::new();
            for candidate in &mut candidates[..count as usize] {
                if candidate.status != 0 {
                    continue;
                }
                let index = ff_hip_lt_index(&mut candidate.algorithm);
                ensure!(
                    index >= 0,
                    "hipBLASLt heuristic algorithm has no stable index"
                );
                choices.push((index, *candidate));
            }
            let (index, mut selected) = if let Some(frozen) = frozen {
                let index = frozen
                    .algorithm
                    .context("HIP multirow frozen choice lacks algorithm index")?;
                let matches: Vec<_> = choices.iter().filter(|(id, _)| *id == index).collect();
                ensure!(
                    matches.len() == 1,
                    "HIP frozen algorithm index {index} has {} heuristic matches",
                    matches.len()
                );
                *matches[0]
            } else {
                *choices
                    .first()
                    .context("hipBLASLt has no valid heuristic algorithm")?
            };
            let mut required = 0;
            check(
                ff_hip_lt_supported(
                    self.handles.lt,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    &mut selected.algorithm,
                    &mut required,
                ),
                "hipBLASLt algorithm support query",
            )?;
            ensure!(
                required <= self.workspace.len(),
                "HIP algorithm {index} workspace {required} exceeds allocation {}",
                self.workspace.len()
            );
            plan.algorithm = selected.algorithm;
            plan.choice.algorithm = Some(index);
            plan.choice.workspace = required;
        }
        Ok(())
    }

    pub fn linear_with_bias(
        &self,
        shape: LinearShape,
        bias: Option<&CudaSlice<bf16>>,
        weight: &impl DevicePtr<bf16>,
        input: &impl DevicePtr<bf16>,
        output: &mut impl DevicePtrMut<bf16>,
    ) -> Result<()> {
        if let Some(bias) = bias {
            bias.check_stream(&self.stream)?;
        }
        let binding = LinearShape {
            rows: if bias.is_none() && (1..=self.invariant_rows).contains(&shape.rows) {
                1
            } else {
                shape.rows
            },
            ..shape
        };
        let key = (
            binding,
            bias.map(|b| b.device_ptr(&self.stream).0),
            LinearBiasMode::BeforeOutputRound,
        );
        let plans = self.plans.borrow();
        let plan = plans
            .get(&key)
            .context("HIP linear shape/bias binding not prepared")?;
        if plan.choice.implementation == Implementation::Wmma {
            return self.wmma((shape, shape.input), bias, weight, input, output);
        }
        ensure!(
            weight.len()
                == shape
                    .output
                    .checked_mul(shape.input)
                    .context("HIP weight extent overflow")?
                && input.len()
                    == shape
                        .rows
                        .checked_mul(shape.input)
                        .context("HIP input extent overflow")?
                && output.len()
                    == shape
                        .rows
                        .checked_mul(shape.output)
                        .context("HIP output extent overflow")?,
            "HIP linear storage shape mismatch"
        );
        self.stream.context().bind_to_thread()?;
        weight.check_stream(&self.stream)?;
        input.check_stream(&self.stream)?;
        output.check_stream(&self.stream)?;
        let (a, _a) = weight.device_ptr(&self.stream);
        let (b, _b) = input.device_ptr(&self.stream);
        let (c, _c) = output.device_ptr_mut(&self.stream);
        self.library_raw(plan, shape, [a, b, c])
    }

    fn library_raw(&self, plan: &Arc<Plan>, shape: LinearShape, pointers: [u64; 3]) -> Result<()> {
        if !plan.choice.bias && (2..=self.invariant_rows).contains(&shape.rows) {
            for row in 0..shape.rows {
                self.library_raw(
                    plan,
                    LinearShape { rows: 1, ..shape },
                    [
                        pointers[0],
                        pointers[1] + (row * shape.input * size_of::<bf16>()) as u64,
                        pointers[2] + (row * shape.output * size_of::<bf16>()) as u64,
                    ],
                )?;
            }
            return Ok(());
        }
        self.stream.context().bind_to_thread()?;
        self.stream.retain(plan);
        let [a, b, c] = pointers;
        let (alpha, beta) = (1f32, 0f32);
        unsafe {
            if plan.output_f32 || (shape.rows == 1 && !plan.choice.bias) {
                self.bf16_gemm(
                    shape,
                    [a, b, c],
                    [shape.input, shape.input, shape.output],
                    None,
                    plan.output_f32,
                )?;
            } else {
                let (workspace, _work) = self.workspace.device_ptr(&self.stream);
                check(
                    hipblasLtMatmul(
                        self.handles.lt,
                        plan.desc,
                        (&alpha as *const f32).cast(),
                        a as *const c_void,
                        plan.a,
                        b as *const c_void,
                        plan.b,
                        (&beta as *const f32).cast(),
                        c as *const c_void,
                        plan.c,
                        c as Handle,
                        plan.c,
                        &plan.algorithm,
                        workspace as Handle,
                        self.workspace.len(),
                        self.stream.raw(),
                    ),
                    "hipblasLtMatmul",
                )?;
            }
        }
        Ok(())
    }

    /// BF16 operands, BF16 or F32 output, FP32 accumulation, W transpose, alpha=1 and beta=0.
    /// # Safety
    /// Pointers cover the strided matrices, or `batch` device pointer arrays.
    /// Their allocations and ordering remain valid until the stream completes.
    pub(crate) unsafe fn bf16_gemm(
        &self,
        shape: LinearShape,
        pointers: [u64; 3],
        strides: [usize; 3],
        batch: Option<usize>,
        output_f32: bool,
    ) -> Result<()> {
        ensure!(
            [shape.rows, shape.output, shape.input]
                .into_iter()
                .chain(strides)
                .all(|n| n > 0 && n <= i32::MAX as usize)
                && strides[0] >= shape.input
                && strides[1] >= shape.input
                && strides[2] >= shape.output
                && pointers
                    .iter()
                    .all(|p| *p != 0 && p.is_multiple_of(if batch.is_some() { 8 } else { 2 }))
                && batch.is_none_or(|n| n > 0 && n <= i32::MAX as usize),
            "HIP BF16 GEMM shape, stride, pointer or batch extent is invalid"
        );
        ensure!(
            !output_f32 || pointers[2].is_multiple_of(4),
            "HIP F32 GEMM output alignment is invalid"
        );
        self.stream.context().bind_to_thread()?;
        self.stream.retain(&self.handles);
        let (alpha, beta) = (1f32, 0f32);
        let [a, b, c] = pointers;
        let [lda, ldb, ldc] = strides.map(|n| n as i32);
        unsafe {
            if let Some(batch) = batch {
                check(
                    hipblasGemmBatchedEx(
                        self.handles.handle,
                        112,
                        111,
                        shape.output as i32,
                        shape.rows as i32,
                        shape.input as i32,
                        (&alpha as *const f32).cast(),
                        a as *const *const c_void,
                        14,
                        lda,
                        b as *const *const c_void,
                        14,
                        ldb,
                        (&beta as *const f32).cast(),
                        c as *mut Handle,
                        if output_f32 { 0 } else { 14 },
                        ldc,
                        batch as i32,
                        2,
                        160,
                    ),
                    "hipblasGemmBatchedEx",
                )?;
            } else {
                check(
                    hipblasGemmEx(
                        self.handles.handle,
                        112,
                        111,
                        shape.output as i32,
                        shape.rows as i32,
                        shape.input as i32,
                        (&alpha as *const f32).cast(),
                        a as *const c_void,
                        14,
                        lda,
                        b as *const c_void,
                        14,
                        ldb,
                        (&beta as *const f32).cast(),
                        c as Handle,
                        if output_f32 { 0 } else { 14 },
                        ldc,
                        2,
                        160,
                    ),
                    "hipblasGemmEx",
                )?;
            }
        }
        Ok(())
    }

    fn validate_choice(&self, choice: &Choice) -> Result<()> {
        if choice.implementation == Implementation::Wmma {
            self.stream.context().require_wmma()?;
            ensure!(
                self.wmma_candidate(choice.shape) && choice.setup_median_ms.is_some(),
                "HIP WMMA choice lacks supported shape or setup trials"
            );
        }
        if let Some(times) = choice.setup_median_ms {
            ensure!(
                self.wmma_candidate(choice.shape)
                    && times.iter().all(|t| t.is_finite() && *t > 0.0)
                    && choice.implementation
                        == if times[1] < times[0] {
                            Implementation::Wmma
                        } else {
                            Implementation::Library
                        },
                "HIP linear setup candidate times/winner mismatch"
            );
        }
        Ok(())
    }

    pub fn wmma_candidate(&self, shape: LinearShape) -> bool {
        self.skinny.is_some()
            && (1..=16).contains(&shape.rows)
            && shape.input > 0
            && shape.output > 0
            && shape.input <= i32::MAX as usize - 127
            && shape.output <= i32::MAX as usize - 15
    }

    pub fn wmma(
        &self,
        layout: (LinearShape, usize),
        bias: Option<&CudaSlice<bf16>>,
        weight: &impl DevicePtr<bf16>,
        input: &impl DevicePtr<bf16>,
        output: &mut impl DevicePtrMut<bf16>,
    ) -> Result<()> {
        self.stream.context().require_wmma()?;
        let (shape, stride) = layout;
        ensure!(
            self.wmma_candidate(shape) && stride >= shape.input,
            "HIP WMMA shape/stride unsupported"
        );
        let required = (shape.output - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(shape.input))
            .context("HIP WMMA weight extent overflow")?;
        ensure!(
            weight.len() >= required
                && input.len() == shape.rows * shape.input
                && output.len() == shape.rows * shape.output,
            "HIP WMMA storage extent mismatch"
        );
        weight.check_stream(&self.stream)?;
        input.check_stream(&self.stream)?;
        output.check_stream(&self.stream)?;
        let bias_ptr = if let Some(bias) = bias {
            bias.check_stream(&self.stream)?;
            ensure!(bias.len() == shape.output, "HIP WMMA bias width mismatch");
            bias.device_ptr(&self.stream).0
        } else {
            0
        };
        let (w, _w) = weight.device_ptr(&self.stream);
        let (x, _x) = input.device_ptr(&self.stream);
        let (y, _y) = output.device_ptr_mut(&self.stream);
        self.wmma_raw(shape, stride, [w, x, y, bias_ptr])
    }

    fn wmma_raw(&self, shape: LinearShape, stride: usize, pointers: [u64; 4]) -> Result<()> {
        let function = self
            .skinny
            .as_ref()
            .context("HIP WMMA function not bound")?;
        let [w, x, y, bias_ptr] = pointers;
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&w)
                .arg(&x)
                .arg(&y)
                .arg(&bias_ptr)
                .arg(&(shape.output as i32))
                .arg(&(shape.input as i32))
                .arg(&(stride as u64))
                .arg(&(shape.rows as i32))
                .launch(crate::backend::gpu::ops::grid(
                    shape.output.div_ceil(16),
                    128,
                ))?;
        }
        Ok(())
    }

    pub(crate) fn prepare_fixed(&self, rows: u32, output: u32, input: u32) -> Result<()> {
        self.prepare_choice(
            LinearShape {
                rows: if (1..=self.invariant_rows).contains(&(rows as usize)) {
                    1
                } else {
                    rows as usize
                },
                output: output as usize,
                input: input as usize,
            },
            None,
            None,
        )?;
        Ok(())
    }

    pub(crate) fn prepare_request_fixed(&self, rows: u32, output: u32, input: u32) -> Result<()> {
        self.prepare_fixed(rows, output, input)
    }

    pub(crate) fn prepare_biased(
        &mut self,
        rows: u32,
        output: u32,
        input: u32,
        bias: crate::backend::gpu::contracts::Tensor<bf16, 1>,
        mode: LinearBiasMode,
    ) -> Result<()> {
        bias.validate()?;
        ensure!(
            bias.shape == [output] && bias.stride == [1],
            "linear bias shape mismatch"
        );
        self.stream.require_eager("biased projection preparation")?;
        if mode == LinearBiasMode::AfterOutputRound {
            let count = (rows as usize)
                .checked_mul(output as usize)
                .context("HIP rounded output extent overflow")?;
            let buffer = self.rounded_output.get_mut();
            if buffer.as_ref().is_none_or(|b| b.len() < count) {
                *buffer = Some(self.stream.alloc_zeros(count)?);
            }
        }
        self.prepare_binding(
            LinearShape {
                rows: rows as usize,
                output: output as usize,
                input: input as usize,
            },
            Some(bias.ptr),
            None,
            mode,
            None,
        )?;
        if mode == LinearBiasMode::AfterOutputRound && self.bias_add.is_none() {
            self.bias_add = Some(
                crate::backend::gpu::cubin::module(self.stream.context(), "q38_common")?
                    .load_function("q38_bias_add")?,
            );
        }
        Ok(())
    }

    pub(crate) fn linear_biased(
        &self,
        weight: crate::backend::gpu::contracts::Tensor<bf16, 2>,
        bias: crate::backend::gpu::contracts::Tensor<bf16, 1>,
        input: crate::backend::gpu::contracts::Tensor<bf16, 2>,
        output: &mut crate::backend::gpu::contracts::TensorMut<bf16, 2>,
        mode: LinearBiasMode,
    ) -> Result<()> {
        use crate::backend::gpu::linear;
        weight.validate()?;
        bias.validate()?;
        linear::validate_call(weight.shape, input, &output.0)?;
        ensure!(
            weight.stride == [u64::from(weight.shape[1]), 1]
                && bias.shape == [weight.shape[0]]
                && bias.stride == [1],
            "biased linear layout mismatch"
        );
        linear::disjoint(&weight, &output.0)?;
        linear::disjoint(&bias, &output.0)?;
        let shape = LinearShape {
            rows: input.shape[0] as usize,
            output: weight.shape[0] as usize,
            input: weight.shape[1] as usize,
        };
        let plans = self.plans.borrow();
        let plan = plans
            .get(&(shape, Some(bias.ptr), mode))
            .context("HIP biased linear shape/binding/rounding not prepared")?;
        if mode == LinearBiasMode::AfterOutputRound {
            let mut buffer = self
                .rounded_output
                .try_borrow_mut()
                .context("HIP rounded projection buffer already borrowed")?;
            let storage = buffer
                .as_mut()
                .context("HIP rounded projection buffer not prepared")?;
            ensure!(
                storage.len() >= shape.rows * shape.output,
                "HIP rounded projection buffer capacity exceeded"
            );
            let (unrounded, _write) = storage.device_ptr_mut(&self.stream);
            self.library_raw(plan, shape, [weight.ptr, input.ptr, unrounded])?;
            let add = self
                .bias_add
                .as_ref()
                .context("HIP bias add kernel not prepared")?;
            let count = u64::from(output.0.shape[0]) * u64::from(output.0.shape[1]);
            unsafe {
                self.stream
                    .launch_builder(add)
                    .arg(&unrounded)
                    .arg(&output.0.ptr)
                    .arg(&bias.ptr)
                    .arg(&count)
                    .arg(&output.0.shape[1])
                    .launch(crate::backend::gpu::ops::grid(
                        count.div_ceil(256).min(65535) as usize,
                        256,
                    ))?;
            }
        } else {
            self.library_raw(plan, shape, [weight.ptr, input.ptr, output.0.ptr])?;
        }
        Ok(())
    }

    pub(crate) fn prepare_skinny(&self, shape: LinearShape) -> Result<()> {
        self.stream
            .require_eager("invariant WMMA plan preparation")?;
        ensure!(
            self.wmma_candidate(shape),
            "HIP invariant WMMA shape unsupported: {shape:?}"
        );
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn projection_rounding_probe(&self, count: usize) -> Result<Vec<f32>> {
        let buffer = self.rounded_output.borrow();
        let values = buffer.as_ref().context("rounded projection missing")?;
        self.stream.clone_dtoh(&values.slice(..count))
    }

    pub(crate) fn linear_typed(
        &self,
        phase: crate::op_contract::ExecutionPhase,
        weight: crate::backend::gpu::linear::LinearWeightRef,
        x: crate::backend::gpu::contracts::Tensor<bf16, 2>,
        y: &mut crate::backend::gpu::contracts::TensorMut<bf16, 2>,
    ) -> Result<()> {
        self.linear_typed_bound(phase, weight, x, y, 0)
    }

    pub(crate) fn linear_typed_bound(
        &self,
        phase: crate::op_contract::ExecutionPhase,
        weight: crate::backend::gpu::linear::LinearWeightRef,
        x: crate::backend::gpu::contracts::Tensor<bf16, 2>,
        y: &mut crate::backend::gpu::contracts::TensorMut<bf16, 2>,
        invariant_rows: usize,
    ) -> Result<()> {
        use crate::backend::gpu::linear;
        let linear::LinearWeightRef::Bf16(w) = weight else {
            anyhow::bail!("HIP FP8 projection requires a setup-bound channel call");
        };
        if phase == crate::op_contract::ExecutionPhase::Decode {
            ensure!(x.shape[0] == 1, "Decode requires M1");
        }
        w.validate()?;
        linear::validate_call(w.shape, x, &y.0)?;
        linear::disjoint(&w, &y.0)?;
        ensure!(
            w.stride == [u64::from(w.shape[1]), 1],
            "HIP BF16 weight requires row-major layout"
        );
        let shape = LinearShape {
            rows: x.shape[0] as usize,
            output: w.shape[0] as usize,
            input: w.shape[1] as usize,
        };
        if (1..=invariant_rows).contains(&shape.rows) {
            ensure!(
                self.wmma_candidate(shape),
                "HIP invariant WMMA shape unsupported: {shape:?}"
            );
            return self.wmma_raw(shape, shape.input, [w.ptr, x.ptr, y.0.ptr, 0]);
        }
        let plans = self.plans.borrow();
        let binding = LinearShape {
            rows: if (1..=self.invariant_rows).contains(&shape.rows) {
                1
            } else {
                shape.rows
            },
            ..shape
        };
        let plan = plans
            .get(&(binding, None, LinearBiasMode::BeforeOutputRound))
            .context("HIP fixed linear shape not prepared")?;
        if plan.choice.implementation == Implementation::Wmma {
            return self.wmma_raw(shape, shape.input, [w.ptr, x.ptr, y.0.ptr, 0]);
        }
        self.library_raw(plan, shape, [w.ptr, x.ptr, y.0.ptr])
    }

    pub fn calibrate_shape(
        &self,
        shape: LinearShape,
        weight: &impl DevicePtr<bf16>,
        input: &impl DevicePtr<bf16>,
    ) -> Result<Choice> {
        let mut choice = self.prepare_choice(shape, None, None)?;
        if !self.wmma_candidate(shape) {
            return Ok(choice);
        }
        ensure!(
            choice.setup_median_ms.is_none(),
            "HIP linear shape already calibrated"
        );
        let mut output = self.stream.alloc_zeros::<bf16>(shape.rows * shape.output)?;
        self.linear_with_bias(shape, None, weight, input, &mut output)?;
        let reference = self.stream.clone_dtoh(&output)?;
        self.wmma((shape, shape.input), None, weight, input, &mut output)?;
        let actual = self.stream.clone_dtoh(&output)?;
        ensure!(
            reference.iter().chain(&actual).all(|v| v.is_finite()),
            "HIP linear setup produced nonfinite output"
        );
        let signal: f64 = reference
            .iter()
            .map(|v| f64::from(v.to_f32()).powi(2))
            .sum();
        ensure!(signal > 0.0, "HIP linear setup has zero reference signal");
        let error: f64 = reference
            .iter()
            .zip(&actual)
            .map(|(a, b)| (f64::from(a.to_f32()) - f64::from(b.to_f32())).powi(2))
            .sum();
        let bound = crate::backend::cuda::bf16_chain_bound(1, &[shape.input]);
        ensure!(
            (error / signal).sqrt() <= bound,
            "HIP WMMA setup exceeds BF16 numerical bound {bound}"
        );
        let begin = self.stream.context().new_event(None)?;
        let end = self.stream.context().new_event(None)?;
        let mut times = [0.0; 2];
        for (candidate, median) in times.iter_mut().enumerate() {
            let mut samples = Vec::with_capacity(12);
            for sample in 0..14 {
                begin.record(&self.stream)?;
                if candidate == 0 {
                    self.linear_with_bias(shape, None, weight, input, &mut output)?;
                } else {
                    self.wmma((shape, shape.input), None, weight, input, &mut output)?;
                }
                end.record(&self.stream)?;
                end.synchronize()?;
                if sample >= 2 {
                    samples.push(f64::from(begin.elapsed_ms(&end)?));
                }
            }
            samples.sort_by(f64::total_cmp);
            *median = (samples[5] + samples[6]) * 0.5;
        }
        choice.setup_median_ms = Some(times);
        choice.implementation = if times[1] < times[0] {
            Implementation::Wmma
        } else {
            Implementation::Library
        };
        self.validate_choice(&choice)?;
        let mut plans = self.plans.borrow_mut();
        let plan = plans
            .get_mut(&(shape, None, LinearBiasMode::BeforeOutputRound))
            .context("HIP calibrated plan missing")?;
        Arc::get_mut(plan)
            .context("HIP captured plan cannot be recalibrated")?
            .choice = choice.clone();
        eprintln!(
            "{}",
            serde_json::json!({"hip_linear_setup":{"choice":choice,"samples":12,"untuned":true,"relative_rmse":(error/signal).sqrt()}})
        );
        Ok(choice)
    }
}

unsafe fn desc_attribute<T>(desc: Handle, attribute: i32, value: &T) -> Result<()> {
    unsafe {
        check(
            hipblasLtMatmulDescSetAttribute(
                desc,
                attribute,
                (value as *const T).cast(),
                size_of::<T>(),
            ),
            "hipblasLtMatmulDescSetAttribute",
        )
    }
}
fn binary_key() -> Result<String> {
    let mut file = std::fs::File::open(std::env::current_exe()?)?;
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    let mut bytes = [0u8; 65536];
    loop {
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        hash.write(&bytes[..count]);
    }
    Ok(format!("hash64-{:016x}", hash.finish()))
}

impl Blas {
    pub(crate) fn bind_invariant(
        &mut self,
        ctx: &Arc<crate::backend::gpu::CudaContext>,
        rows: usize,
    ) -> Result<()> {
        ctx.require_wmma()?;
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows),
            "HIP invariant row bound exceeds skinny capacity: {rows}"
        );
        self.invariant_rows = rows;
        Ok(())
    }

    pub fn prepare(&self, rows: usize, output: usize, input: usize) -> Result<()> {
        let shape = LinearShape {
            rows: if (1..=self.invariant_rows).contains(&rows) {
                1
            } else {
                rows
            },
            output,
            input,
        };
        ensure!(
            self.plans
                .borrow()
                .contains_key(&(shape, None, LinearBiasMode::BeforeOutputRound)),
            "HIP linear shape is outside the frozen setup closure: {shape:?}"
        );
        Ok(())
    }
    pub fn linear(
        &self,
        weight: &impl DevicePtr<bf16>,
        x: &impl DevicePtr<bf16>,
        y: &mut impl DevicePtrMut<bf16>,
        rows: usize,
        output: usize,
        input: usize,
    ) -> Result<()> {
        self.linear_with_bias(
            LinearShape {
                rows,
                output,
                input,
            },
            None,
            weight,
            x,
            y,
        )
    }
    pub fn decode_impl(&self) -> &'static str {
        "HIP BLAS BF16 with FP32 accumulation"
    }
}
