//! BF16 HIP library plans are bound by library algorithm index, not heuristic order.
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
    bias: Option<Arc<CudaSlice<bf16>>>,
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

pub struct Blas {
    stream: Arc<CudaStream>,
    handles: Arc<Handles>,
    identity: Identity,
    workspace: CudaSlice<u8>,
    skinny: Option<CudaFunction>,
    // Every plan accumulates in FP32 and adds bias before the BF16 output rounding.
    plans: RefCell<BTreeMap<(LinearShape, Option<u64>), Arc<Plan>>>,
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

    pub fn prepare_choice(
        &self,
        shape: LinearShape,
        bias: Option<Arc<CudaSlice<bf16>>>,
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
        if let Some(bias) = &bias {
            bias.check_stream(&self.stream)?;
            ensure!(
                shape.rows > 1 && bias.len() == shape.output,
                "HIP BF16 bias requires multirow output-width vector"
            );
        }
        if let Some(frozen) = frozen {
            ensure!(
                frozen.identity == self.identity
                    && frozen.shape == shape
                    && frozen.bias == bias.is_some(),
                "HIP frozen algorithm identity/shape/epilogue mismatch"
            );
            self.validate_choice(frozen)?;
        }
        let key = (shape, bias.as_ref().map(|b| b.device_ptr(&self.stream).0));
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
            bias,
        };
        if shape.rows > 1 {
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
            if let Some(bias) = &plan.bias {
                desc_attribute(plan.desc, 2, &4u32)?;
                desc_attribute(plan.desc, 3, &bias.device_ptr(&self.stream).0)?;
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
        let key = (shape, bias.map(|b| b.device_ptr(&self.stream).0));
        let plans = self.plans.borrow();
        let plan = plans
            .get(&key)
            .context("HIP linear shape/bias binding not prepared")?;
        self.stream.retain(plan);
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
        let (alpha, beta) = (1f32, 0f32);
        unsafe {
            if shape.rows == 1 {
                check(
                    hipblasGemmEx(
                        self.handles.handle,
                        112,
                        111,
                        shape.output as i32,
                        1,
                        shape.input as i32,
                        (&alpha as *const f32).cast(),
                        a as *const c_void,
                        14,
                        shape.input as i32,
                        b as *const c_void,
                        14,
                        shape.input as i32,
                        (&beta as *const f32).cast(),
                        c as Handle,
                        14,
                        shape.output as i32,
                        2,
                        160,
                    ),
                    "hipblasGemmEx",
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
        let function = self
            .skinny
            .as_ref()
            .context("HIP WMMA function not bound")?;
        let (w, _w) = weight.device_ptr(&self.stream);
        let (x, _x) = input.device_ptr(&self.stream);
        let (y, _y) = output.device_ptr_mut(&self.stream);
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
            .get_mut(&(shape, None))
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
    pub fn prepare(&self, rows: usize, output: usize, input: usize) -> Result<()> {
        let shape = LinearShape {
            rows,
            output,
            input,
        };
        ensure!(
            self.plans.borrow().contains_key(&(shape, None)),
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
