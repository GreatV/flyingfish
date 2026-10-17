//! Host-side routed-expert FFN for decode misses; model-independent.
use crate::experts::arena::{BlockLease, HostBlocks};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use std::ops::Range;
use std::sync::{
    Arc, Condvar, Mutex, RwLock,
    atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicUsize, Ordering},
};
use std::thread::{self, JoinHandle};

/// Partial-sum lanes shared by every arm; the scalar arm defines this order.
const LANES: usize = 16;

/// Spin rounds a parked worker and the calling thread run before they block.
const SPIN_ROUNDS: u32 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub hidden: usize,
    pub intermediate: usize,
}

impl Shape {
    pub fn new(hidden: usize, intermediate: usize) -> Result<Self> {
        ensure!(hidden > 0 && intermediate > 0, "zero expert dimension");
        ensure!(
            intermediate.is_multiple_of(8) && hidden.is_multiple_of(4),
            "expert dimensions are not multiple-of-lane shaped"
        );
        Ok(Self {
            hidden,
            intermediate,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512Bf16,
}

impl Isa {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::is_x86_feature_detected as has;
            if has!("avx512bf16") && has!("avx512f") {
                return Self::Avx512Bf16;
            }
            if has!("avx2") && has!("fma") {
                return Self::Avx2;
            }
        }
        Self::Scalar
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2+fma",
            Self::Avx512Bf16 => "avx512bf16",
        }
    }

    fn available(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                use std::arch::is_x86_feature_detected as has;
                has!("avx2") && has!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Bf16 => {
                use std::arch::is_x86_feature_detected as has;
                has!("avx512bf16") && has!("avx512f")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

/// Native `[2I, H]` gate-up and `[H, I]` down slices addressed by element ranges
/// into an arena, one row stride each; rows are contiguous and gate rows precede up rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertWeights {
    pub gate_up: Range<usize>,
    pub gate_up_stride: usize,
    pub down: Range<usize>,
    pub down_stride: usize,
}

impl ExpertWeights {
    fn validate(&self, shape: &Shape, arena: usize) -> Result<()> {
        let (hidden, intermediate) = (shape.hidden, shape.intermediate);
        ensure!(
            self.gate_up_stride >= hidden && self.down_stride >= intermediate,
            "expert stride below the row width"
        );
        let gate_up_end = self
            .gate_up
            .start
            .checked_add((2 * intermediate - 1) * self.gate_up_stride + hidden)
            .context("gate-up range overflow")?;
        ensure!(
            gate_up_end <= self.gate_up.end && self.gate_up.end <= arena,
            "gate-up range exceeds the arena"
        );
        let down_end = self
            .down
            .start
            .checked_add((hidden - 1) * self.down_stride + intermediate)
            .context("down range overflow")?;
        ensure!(
            down_end <= self.down.end && self.down.end <= arena,
            "down range exceeds the arena"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assignment {
    pub row: u32,
    pub choice: u32,
    pub expert: u32,
    pub weight: bf16,
}

pub struct Scratch {
    hidden: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gate_bf: Vec<bf16>,
    up_bf: Vec<bf16>,
    product: Vec<f32>,
    activated: Vec<bf16>,
    activated_f: Vec<f32>,
    down: Vec<f32>,
    down_bf: Vec<bf16>,
    weighted: Vec<f32>,
}

impl Scratch {
    pub fn new(shape: &Shape) -> Self {
        Self {
            hidden: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            gate_bf: vec![bf16::ZERO; shape.intermediate],
            up_bf: vec![bf16::ZERO; shape.intermediate],
            product: vec![0.0; shape.intermediate],
            activated: vec![bf16::ZERO; shape.intermediate],
            activated_f: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
            down_bf: vec![bf16::ZERO; shape.hidden],
            weighted: vec![0.0; shape.hidden],
        }
    }

    fn check(&self, shape: &Shape) -> Result<()> {
        ensure!(
            self.hidden.len() == shape.hidden
                && self.gate.len() == shape.intermediate
                && self.up.len() == shape.intermediate
                && self.gate_bf.len() == shape.intermediate
                && self.up_bf.len() == shape.intermediate
                && self.product.len() == shape.intermediate
                && self.activated.len() == shape.intermediate
                && self.activated_f.len() == shape.intermediate
                && self.down.len() == shape.hidden
                && self.down_bf.len() == shape.hidden
                && self.weighted.len() == shape.hidden,
            "scratch dimensions differ from the expert shape"
        );
        Ok(())
    }
}

type DotFn = unsafe fn(&[f32], &[bf16]) -> f32;
type RoundFn = unsafe fn(&[f32], &mut [bf16]);

struct Arm {
    dot: DotFn,
    round: RoundFn,
}

const SCALAR_ARM: Arm = Arm {
    dot: scalar_dot,
    round: scalar_round,
};

fn arm_for(isa: Isa) -> Result<&'static Arm> {
    match isa {
        Isa::Scalar => Ok(&SCALAR_ARM),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => {
            ensure!(isa.available(), "AVX2+FMA not available");
            Ok(&AVX2_ARM)
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Bf16 => {
            ensure!(isa.available(), "AVX512_BF16 not available");
            Ok(&AVX512_ARM)
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => bail!("unsupported ISA arm {}", isa.name()),
    }
}

/// One routed-expert contribution input.
pub struct Contribution<'a> {
    pub shape: &'a Shape,
    pub arena: &'a [bf16],
    pub weights: &'a ExpertWeights,
    pub hidden: &'a [bf16],
    pub routing: bf16,
}

/// One `(row, choice)` contribution with the model-definition BF16 boundaries.
pub fn contribution(
    isa: Isa,
    input: Contribution<'_>,
    out: &mut [bf16],
    scratch: &mut Scratch,
) -> Result<()> {
    ensure!(
        input.hidden.len() == input.shape.hidden && out.len() == input.shape.hidden,
        "hidden/out width differs from the expert shape"
    );
    input.weights.validate(input.shape, input.arena.len())?;
    scratch.check(input.shape)?;
    compute(arm_for(isa)?, input, out, scratch);
    Ok(())
}

fn compute(arm: &Arm, input: Contribution<'_>, out: &mut [bf16], scratch: &mut Scratch) {
    let Contribution {
        shape,
        arena,
        weights,
        hidden,
        routing,
    } = input;
    let (hidden_width, intermediate) = (shape.hidden, shape.intermediate);
    for (value, source) in scratch.hidden.iter_mut().zip(hidden) {
        *value = f32::from(*source);
    }
    let gate_up = &arena[weights.gate_up.clone()];
    for j in 0..intermediate {
        let gate = &gate_up[j * weights.gate_up_stride..][..hidden_width];
        let up = &gate_up[(intermediate + j) * weights.gate_up_stride..][..hidden_width];
        // SAFETY: the arm was selected by `Isa::detect` or checked by `Isa::available`.
        scratch.gate[j] = unsafe { (arm.dot)(&scratch.hidden, gate) };
        scratch.up[j] = unsafe { (arm.dot)(&scratch.hidden, up) };
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.gate, &mut scratch.gate_bf) };
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.up, &mut scratch.up_bf) };
    for j in 0..intermediate {
        let gate = f32::from(scratch.gate_bf[j]);
        let silu = gate * (1.0 / (1.0 + (-gate).exp()));
        scratch.product[j] = silu * f32::from(scratch.up_bf[j]);
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.product, &mut scratch.activated) };
    for (value, source) in scratch.activated_f.iter_mut().zip(&scratch.activated) {
        *value = f32::from(*source);
    }
    let down = &arena[weights.down.clone()];
    for k in 0..hidden_width {
        let row = &down[k * weights.down_stride..][..intermediate];
        // SAFETY: as above.
        scratch.down[k] = unsafe { (arm.dot)(&scratch.activated_f, row) };
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.down, &mut scratch.down_bf) };
    let weight = f32::from(routing);
    for (value, source) in scratch.weighted.iter_mut().zip(&scratch.down_bf) {
        *value = f32::from(*source) * weight;
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.weighted, out) };
}

fn pairwise(lanes: &[f32; LANES]) -> f32 {
    let low = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let high = ((lanes[8] + lanes[9]) + (lanes[10] + lanes[11]))
        + ((lanes[12] + lanes[13]) + (lanes[14] + lanes[15]));
    low + high
}

fn scalar_dot(x: &[f32], weights: &[bf16]) -> f32 {
    assert_eq!(x.len(), weights.len(), "dot operand width mismatch");
    let mut lanes = [0f32; LANES];
    let mut k = 0;
    while k + LANES <= x.len() {
        for lane in 0..LANES {
            lanes[lane] = x[k + lane].mul_add(f32::from(weights[k + lane]), lanes[lane]);
        }
        k += LANES;
    }
    while k < x.len() {
        let lane = k % LANES;
        lanes[lane] = x[k].mul_add(f32::from(weights[k]), lanes[lane]);
        k += 1;
    }
    pairwise(&lanes)
}

fn scalar_round(values: &[f32], out: &mut [bf16]) {
    assert_eq!(values.len(), out.len(), "round operand width mismatch");
    for (value, target) in values.iter().zip(out.iter_mut()) {
        *target = bf16::from_f32(*value);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Arm, LANES, bf16, pairwise};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(x: &[f32], weights: &[bf16]) -> f32 {
        assert_eq!(x.len(), weights.len(), "dot operand width mismatch");
        let mut low = _mm256_setzero_ps();
        let mut high = _mm256_setzero_ps();
        let mut k = 0;
        while k + LANES <= x.len() {
            unsafe {
                let xl = _mm256_loadu_ps(x.as_ptr().add(k));
                let xh = _mm256_loadu_ps(x.as_ptr().add(k + 8));
                low = _mm256_fmadd_ps(xl, widen8(weights.as_ptr().add(k)), low);
                high = _mm256_fmadd_ps(xh, widen8(weights.as_ptr().add(k + 8)), high);
            }
            k += LANES;
        }
        let mut lanes = [0f32; LANES];
        unsafe {
            _mm256_storeu_ps(lanes.as_mut_ptr(), low);
            _mm256_storeu_ps(lanes.as_mut_ptr().add(8), high);
        }
        while k < x.len() {
            let lane = k % LANES;
            lanes[lane] = x[k].mul_add(f32::from(weights[k]), lanes[lane]);
            k += 1;
        }
        pairwise(&lanes)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn widen8(values: *const bf16) -> __m256 {
        unsafe {
            let packed = _mm_loadu_si128(values as *const __m128i);
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16))
        }
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn dot_512(x: &[f32], weights: &[bf16]) -> f32 {
        assert_eq!(x.len(), weights.len(), "dot operand width mismatch");
        let mut acc = _mm512_setzero_ps();
        let mut k = 0;
        while k + LANES <= x.len() {
            unsafe {
                let xv = _mm512_loadu_ps(x.as_ptr().add(k));
                acc = _mm512_fmadd_ps(xv, widen16(weights.as_ptr().add(k)), acc);
            }
            k += LANES;
        }
        let mut lanes = [0f32; LANES];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
        while k < x.len() {
            let lane = k % LANES;
            lanes[lane] = x[k].mul_add(f32::from(weights[k]), lanes[lane]);
            k += 1;
        }
        pairwise(&lanes)
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn widen16(values: *const bf16) -> __m512 {
        unsafe {
            let packed = _mm256_loadu_si256(values as *const __m256i);
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16))
        }
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn round_512(values: &[f32], out: &mut [bf16]) {
        assert_eq!(values.len(), out.len(), "round operand width mismatch");
        let mut k = 0;
        while k + 32 <= values.len() {
            let low = unsafe { _mm512_loadu_ps(values.as_ptr().add(k)) };
            let high = unsafe { _mm512_loadu_ps(values.as_ptr().add(k + 16)) };
            let packed = _mm512_cvtne2ps_pbh(high, low);
            let raw: __m512i = unsafe { std::mem::transmute(packed) };
            unsafe { _mm512_storeu_si512(out.as_mut_ptr().add(k) as *mut __m512i, raw) };
            k += 32;
        }
        for (value, target) in values[k..].iter().zip(out[k..].iter_mut()) {
            *target = bf16::from_f32(*value);
        }
    }

    pub(super) const AVX2: Arm = Arm {
        dot,
        round: super::scalar_round,
    };

    pub(super) const AVX512: Arm = Arm {
        dot: dot_512,
        round: round_512,
    };
}

#[cfg(target_arch = "x86_64")]
use x86::{AVX2 as AVX2_ARM, AVX512 as AVX512_ARM};

/// A fixed worker pool over native expert slices with per-job block read leases.
pub struct ExpertPool {
    shape: Shape,
    workers: usize,
    max_rows: usize,
    max_assignments: usize,
    isa: Isa,
    shared: Arc<PoolShared>,
    handles: Vec<JoinHandle<()>>,
}

pub struct Job<'a> {
    pub arena: Arc<HostBlocks>,
    pub hidden: &'a [bf16],
    pub rows: usize,
    pub experts: &'a [ExpertWeights],
    pub assignments: &'a [Assignment],
    pub out: &'a mut [bf16],
}

struct PoolShared {
    job: RwLock<JobSlot>,
    park: Mutex<()>,
    wake: Condvar,
    error: Mutex<Option<String>>,
    workers: usize,
    generation: AtomicU64,
    shutdown: AtomicBool,
    phase1: AtomicUsize,
    finished: AtomicUsize,
    failed: AtomicBool,
    activated: Box<[AtomicU16]>,
    out: Box<[AtomicU16]>,
}

struct JobSlot {
    rows: usize,
    arena: Option<Arc<BlockLease>>,
    hidden: Vec<bf16>,
    experts: Vec<ExpertWeights>,
    assignments: Vec<Assignment>,
}

impl JobSlot {
    fn new(shape: &Shape, max_rows: usize, max_assignments: usize) -> Self {
        Self {
            rows: 0,
            arena: None,
            hidden: vec![bf16::ZERO; max_rows * shape.hidden],
            experts: Vec::with_capacity(max_assignments),
            assignments: Vec::with_capacity(max_assignments),
        }
    }
}

struct Local {
    hidden_bf: Vec<bf16>,
    hidden_f: Vec<f32>,
    hidden_rows: Vec<bool>,
    experts: Vec<ExpertWeights>,
    items: Vec<Assignment>,
    activated_bf: Vec<bf16>,
    act_f: Vec<f32>,
    block: Vec<bf16>,
}

impl Local {
    fn new(shape: &Shape, workers: usize, max_rows: usize, max_assignments: usize) -> Self {
        let block = (max_assignments * shape.hidden).div_ceil(workers);
        Self {
            hidden_bf: vec![bf16::ZERO; max_rows * shape.hidden],
            hidden_f: vec![0.0; max_rows * shape.hidden],
            hidden_rows: vec![false; max_rows],
            experts: Vec::with_capacity(max_assignments),
            items: Vec::with_capacity(max_assignments),
            activated_bf: vec![bf16::ZERO; max_assignments * shape.intermediate],
            act_f: vec![0.0; shape.intermediate],
            block: vec![bf16::ZERO; block],
        }
    }
}

/// Host block of every assignment expert; each expert's slices must stay inside one block.
fn assignment_blocks(
    arena: &HostBlocks,
    experts: &[ExpertWeights],
    assignments: &[Assignment],
) -> Result<Vec<u32>> {
    let per_block = arena.block_elements();
    let mut blocks = Vec::with_capacity(assignments.len());
    for (index, assignment) in assignments.iter().enumerate() {
        let weights = experts
            .get(assignment.expert as usize)
            .with_context(|| format!("assignment {index} expert {}", assignment.expert))?;
        let block = weights.gate_up.start / per_block;
        let end = (block + 1) * per_block;
        ensure!(
            weights.gate_up.end <= end
                && weights.down.start >= block * per_block
                && weights.down.end <= end,
            "expert {} weights span host blocks",
            assignment.expert
        );
        blocks.push(u32::try_from(block).context("host block index exceeds u32")?);
    }
    Ok(blocks)
}

fn block(index: usize, workers: usize, total: usize) -> std::ops::Range<usize> {
    let chunk = total.div_ceil(workers);
    let start = (index * chunk).min(total);
    let end = ((index + 1) * chunk).min(total);
    start..end
}

impl ExpertPool {
    pub fn new(
        workers: usize,
        shape: Shape,
        max_rows: usize,
        max_assignments: usize,
    ) -> Result<Self> {
        ensure!(
            workers > 0 && workers <= 256,
            "invalid expert worker count {workers}"
        );
        ensure!(max_rows > 0, "zero expert row capacity");
        ensure!(
            max_assignments > 0 && max_assignments <= 1 << 16,
            "invalid expert assignment capacity {max_assignments}"
        );
        let isa = Isa::detect();
        let shared = Arc::new(PoolShared {
            job: RwLock::new(JobSlot::new(&shape, max_rows, max_assignments)),
            park: Mutex::new(()),
            wake: Condvar::new(),
            error: Mutex::new(None),
            workers,
            generation: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            phase1: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            activated: (0..max_assignments * shape.intermediate)
                .map(|_| AtomicU16::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            out: (0..max_assignments * shape.hidden)
                .map(|_| AtomicU16::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        });
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let worker_shared = shared.clone();
            let spawned = thread::Builder::new()
                .name(format!("expert-ffn-{index}"))
                .spawn(move || worker(index, worker_shared, shape, isa, max_rows, max_assignments));
            match spawned {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    {
                        let _guard = shared.park.lock().unwrap_or_else(|e| e.into_inner());
                        shared.shutdown.store(true, Ordering::Release);
                        shared.wake.notify_all();
                    }
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(error).context("spawn expert worker");
                }
            }
        }
        Ok(Self {
            shape,
            workers,
            max_rows,
            max_assignments,
            isa,
            shared,
            handles,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    pub fn run(&self, job: Job<'_>) -> Result<()> {
        let Job {
            arena,
            hidden,
            rows,
            experts,
            assignments,
            out,
        } = job;
        self.validate(
            arena.elements(),
            hidden.len(),
            rows,
            experts,
            assignments,
            out.len(),
        )?;
        let blocks = assignment_blocks(&arena, experts, assignments)?;
        let lease = Arc::new(arena.lease(&blocks)?);
        {
            let mut slot = self.shared.job.write().unwrap_or_else(|e| e.into_inner());
            slot.rows = rows;
            slot.arena = Some(lease.clone());
            slot.hidden[..hidden.len()].copy_from_slice(hidden);
            slot.experts.clear();
            slot.experts.extend_from_slice(experts);
            slot.assignments.clear();
            slot.assignments.extend_from_slice(assignments);
        }
        *self.shared.error.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.shared.phase1.store(0, Ordering::Release);
        self.shared.finished.store(0, Ordering::Release);
        self.shared.failed.store(false, Ordering::Release);
        {
            let _guard = self.shared.park.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.generation.fetch_add(1, Ordering::Release);
            self.shared.wake.notify_all();
        }
        let mut spins = 0u32;
        loop {
            if self.shared.finished.load(Ordering::Acquire) >= self.workers {
                break;
            }
            if spins < SPIN_ROUNDS {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let mut guard = self.shared.park.lock().unwrap_or_else(|e| e.into_inner());
            while self.shared.finished.load(Ordering::Acquire) < self.workers {
                guard = self
                    .shared
                    .wake
                    .wait(guard)
                    .unwrap_or_else(|e| e.into_inner());
            }
            break;
        }
        if let Some(error) = self
            .shared
            .error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            bail!("{error}");
        }
        let len = out.len();
        for (target, source) in out.iter_mut().zip(&self.shared.out[..len]) {
            *target = bf16::from_bits(source.load(Ordering::Acquire));
        }
        {
            let mut slot = self.shared.job.write().unwrap_or_else(|e| e.into_inner());
            slot.arena = None;
        }
        drop(lease);
        Ok(())
    }

    fn validate(
        &self,
        arena_len: usize,
        hidden_len: usize,
        rows: usize,
        experts: &[ExpertWeights],
        assignments: &[Assignment],
        out_len: usize,
    ) -> Result<()> {
        let hidden = self.shape.hidden;
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "rows {rows} exceed the pool's {}",
            self.max_rows
        );
        ensure!(
            hidden_len == rows * hidden,
            "hidden extent differs from rows * hidden"
        );
        ensure!(
            assignments.len() <= self.max_assignments,
            "assignments exceed the pool's {}",
            self.max_assignments
        );
        ensure!(
            experts.len() <= self.max_assignments,
            "experts exceed the pool's {}",
            self.max_assignments
        );
        ensure!(
            out_len.is_multiple_of(hidden) && out_len >= hidden,
            "out extent is not a whole number of rows"
        );
        let out_rows = out_len / hidden;
        ensure!(
            out_rows <= self.max_assignments,
            "out extent exceeds the pool's {} slots",
            self.max_assignments
        );
        for (index, weights) in experts.iter().enumerate() {
            weights
                .validate(&self.shape, arena_len)
                .with_context(|| format!("expert {index} weights"))?;
        }
        for (index, assignment) in assignments.iter().enumerate() {
            ensure!(
                (assignment.row as usize) < rows,
                "assignment {index} row {} outside {rows}",
                assignment.row
            );
            ensure!(
                (assignment.expert as usize) < experts.len(),
                "assignment {index} expert {} outside {}",
                assignment.expert,
                experts.len()
            );
            ensure!(
                (assignment.choice as usize) < out_rows,
                "assignment {index} choice {} outside {out_rows}",
                assignment.choice
            );
            for other in &assignments[..index] {
                ensure!(
                    other.choice != assignment.choice,
                    "duplicate choice slot {}",
                    assignment.choice
                );
            }
        }
        Ok(())
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        {
            let _guard = self.shared.park.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.shutdown.store(true, Ordering::Release);
            self.shared.wake.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn worker(
    index: usize,
    shared: Arc<PoolShared>,
    shape: Shape,
    isa: Isa,
    max_rows: usize,
    max_assignments: usize,
) {
    let mut local = Local::new(&shape, shared.workers, max_rows, max_assignments);
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let current = shared.generation.load(Ordering::Acquire);
            if current != seen {
                seen = current;
                break;
            }
            if spins < SPIN_ROUNDS {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let mut guard = shared.park.lock().unwrap_or_else(|e| e.into_inner());
            while shared.generation.load(Ordering::Acquire) == seen
                && !shared.shutdown.load(Ordering::Acquire)
            {
                guard = shared.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
            drop(guard);
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            phases(index, &shared, &shape, isa, &mut local)
        }));
        let message = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error.to_string()),
            Err(_) => Some("expert worker panicked".to_string()),
        };
        if let Some(message) = message {
            let mut error = shared.error.lock().unwrap_or_else(|e| e.into_inner());
            if error.is_none() {
                *error = Some(message);
            }
            shared.failed.store(true, Ordering::Release);
        }
        let done = shared.finished.fetch_add(1, Ordering::AcqRel) + 1;
        if done == shared.workers {
            let _guard = shared.park.lock().unwrap_or_else(|e| e.into_inner());
            shared.wake.notify_all();
        }
    }
}

fn phases(
    index: usize,
    shared: &PoolShared,
    shape: &Shape,
    isa: Isa,
    local: &mut Local,
) -> Result<()> {
    let arm = arm_for(isa)?;
    let (hidden, intermediate) = (shape.hidden, shape.intermediate);
    let workers = shared.workers;
    let (lease, items_len, range1, rows) = {
        let slot = shared.job.read().unwrap_or_else(|e| e.into_inner());
        local.experts.clear();
        local.experts.extend_from_slice(&slot.experts);
        local.items.clear();
        local.items.extend_from_slice(&slot.assignments);
        let items_len = slot.assignments.len();
        let range1 = block(index, workers, items_len * intermediate);
        local.hidden_rows[..slot.rows].fill(false);
        if range1.start < range1.end {
            let first = range1.start / intermediate;
            let last = (range1.end - 1) / intermediate;
            for item in &slot.assignments[first..=last] {
                let row = item.row as usize;
                if !local.hidden_rows[row] {
                    local.hidden_bf[row * hidden..][..hidden]
                        .copy_from_slice(&slot.hidden[row * hidden..][..hidden]);
                    local.hidden_rows[row] = true;
                }
            }
        }
        let lease = slot.arena.clone().context("job lease missing")?;
        (lease, items_len, range1, slot.rows)
    };
    for row in 0..rows {
        if local.hidden_rows[row] {
            for (value, source) in local.hidden_f[row * hidden..][..hidden]
                .iter_mut()
                .zip(&local.hidden_bf[row * hidden..][..hidden])
            {
                *value = f32::from(*source);
            }
        }
    }
    for unit in range1.clone() {
        let item = &local.items[unit / intermediate];
        let j = unit % intermediate;
        let expert = &local.experts[item.expert as usize];
        let block_elements = lease.block_elements();
        let block = expert.gate_up.start / block_elements;
        let data = lease.read(u32::try_from(block)?)?;
        let gate_up = &data[expert.gate_up.start % block_elements..]
            [..expert.gate_up.end - expert.gate_up.start];
        let x = &local.hidden_f[item.row as usize * hidden..][..hidden];
        let gate = unsafe { (arm.dot)(x, &gate_up[j * expert.gate_up_stride..][..hidden]) };
        let up = unsafe {
            (arm.dot)(
                x,
                &gate_up[(intermediate + j) * expert.gate_up_stride..][..hidden],
            )
        };
        let gate = bf16::from_f32(gate);
        let up = bf16::from_f32(up);
        let gate = f32::from(gate);
        let silu = gate * (1.0 / (1.0 + (-gate).exp()));
        local.block[unit - range1.start] = bf16::from_f32(silu * f32::from(up));
    }
    if shared.failed.load(Ordering::Acquire) {
        return Ok(());
    }
    for (offset, value) in local.block[..range1.len()].iter().enumerate() {
        shared.activated[range1.start + offset].store(value.to_bits(), Ordering::Relaxed);
    }
    shared.phase1.fetch_add(1, Ordering::Release);
    let mut spins = 0u32;
    while shared.phase1.load(Ordering::Acquire) < workers && !shared.failed.load(Ordering::Acquire)
    {
        if spins < 4096 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
    if shared.failed.load(Ordering::Acquire) {
        return Ok(());
    }
    let total2 = items_len * hidden;
    let range2 = block(index, workers, total2);
    if range2.start < range2.end {
        let first_item = range2.start / hidden;
        let last_item = (range2.end - 1) / hidden;
        let start = first_item * intermediate;
        let end = (last_item + 1) * intermediate;
        local.activated_bf.clear();
        for slot in &shared.activated[start..end] {
            local
                .activated_bf
                .push(bf16::from_bits(slot.load(Ordering::Relaxed)));
        }
        let mut cached = usize::MAX;
        for unit in range2.clone() {
            let item_index = unit / hidden;
            if cached != item_index {
                let start = (item_index - first_item) * intermediate;
                for (value, source) in local
                    .act_f
                    .iter_mut()
                    .zip(&local.activated_bf[start..start + intermediate])
                {
                    *value = f32::from(*source);
                }
                cached = item_index;
            }
            let item = &local.items[item_index];
            let expert = &local.experts[item.expert as usize];
            let block = expert.down.start / lease.block_elements();
            let data = lease.read(u32::try_from(block)?)?;
            let down = &data[expert.down.start % lease.block_elements()..]
                [..expert.down.end - expert.down.start];
            let k = unit % hidden;
            let value = unsafe {
                (arm.dot)(
                    &local.act_f,
                    &down[k * expert.down_stride..][..intermediate],
                )
            };
            let linear = bf16::from_f32(value);
            local.block[unit - range2.start] =
                bf16::from_f32(f32::from(linear) * f32::from(item.weight));
        }
    }
    if shared.failed.load(Ordering::Acquire) {
        return Ok(());
    }
    for (offset, unit) in range2.clone().enumerate() {
        let item = &local.items[unit / hidden];
        let k = unit % hidden;
        shared.out[item.choice as usize * hidden + k]
            .store(local.block[offset].to_bits(), Ordering::Relaxed);
    }
    Ok(())
}

pub fn measure_service_table(
    shape: Shape,
    ring_experts: u32,
    workers: &[u32],
    rows: &[u32],
    missed: &[u32],
) -> Result<Vec<crate::experts::plan::ServiceCell>> {
    Shape::new(shape.hidden, shape.intermediate)?;
    let rows_max = *rows.iter().max().context("empty service row lattice")? as usize;
    let missed_max = *missed.iter().max().context("empty service miss lattice")? as usize;
    ensure!(
        !workers.is_empty()
            && workers.iter().all(|&n| n > 0 && n <= 256)
            && rows.iter().all(|&n| n > 0)
            && missed.iter().all(|&n| n > 0),
        "invalid expert service lattice: workers={workers:?} rows={rows:?} missed={missed:?}"
    );
    ensure!(
        ring_experts as usize >= missed_max,
        "expert service ring {ring_experts} cannot hold {missed_max} misses"
    );
    let per = shape
        .hidden
        .checked_mul(shape.intermediate)
        .and_then(|n| n.checked_mul(3))
        .context("service expert extent overflow")?;
    let max_assignments = rows_max
        .checked_mul(missed_max)
        .context("service assignment extent overflow")?;
    ensure!(
        max_assignments <= 1 << 16,
        "service assignments {max_assignments} exceed pool capacity {}",
        1 << 16
    );
    let arena = HostBlocks::new(ring_experts, per)?;
    let mut values = vec![bf16::ZERO; per];
    for (index, slot) in values.iter_mut().enumerate() {
        *slot = bf16::from_f32((((index % 251) as f32) - 125.0) / 128.0);
    }
    for expert in 0..ring_experts {
        arena.claim(expert)?.slice()?.copy_from_slice(&values);
    }
    let hidden: Vec<bf16> = (0..rows_max * shape.hidden)
        .map(|k| bf16::from_f32(((k as f64 * 0.6180339887498949 + 0.9).sin() * 0.75) as f32))
        .collect();
    let mut cells = Vec::new();
    let mut cursor = 0usize;
    for &workers in workers {
        let pool = ExpertPool::new(workers as usize, shape, rows_max, max_assignments)?;
        for &rows in rows {
            for &missed in missed {
                let assignments: Vec<Assignment> = (0..rows * missed)
                    .map(|slot| Assignment {
                        row: slot / missed,
                        choice: slot,
                        expert: slot % missed,
                        weight: bf16::from_f32(0.125 * ((slot % missed) as f32 + 1.0)),
                    })
                    .collect();
                let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
                let mut samples = [0.0; 5];
                for trial in 0..6 {
                    cursor = (cursor + missed as usize) % ring_experts as usize;
                    let selected: Vec<ExpertWeights> = (0..missed as usize)
                        .map(|choice| {
                            let start = ((cursor + choice) % ring_experts as usize) * per;
                            let down = start + 2 * shape.intermediate * shape.hidden;
                            ExpertWeights {
                                gate_up: start..down,
                                gate_up_stride: shape.hidden,
                                down: down..start + per,
                                down_stride: shape.intermediate,
                            }
                        })
                        .collect();
                    let started = std::time::Instant::now();
                    pool.run(Job {
                        arena: arena.clone(),
                        hidden: &hidden[..rows as usize * shape.hidden],
                        rows: rows as usize,
                        experts: &selected,
                        assignments: &assignments,
                        out: &mut out,
                    })?;
                    let seconds = started.elapsed().as_secs_f64();
                    ensure!(
                        out.iter().all(|x| x.to_f32().is_finite()),
                        "non-finite CPU service output: workers={workers} rows={rows} missed={missed}"
                    );
                    if trial > 0 {
                        samples[trial - 1] = seconds;
                    }
                }
                samples.sort_by(f64::total_cmp);
                cells.push(crate::experts::plan::ServiceCell {
                    workers,
                    rows,
                    experts: missed,
                    seconds: samples[2],
                });
            }
        }
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 2048;
    const I: usize = 512;

    fn production_shape() -> Shape {
        Shape::new(H, I).unwrap()
    }

    fn pattern(seed: usize, phase: f64) -> bf16 {
        bf16::from_f32(((seed as f64 * 0.6180339887498949 + phase).sin() * 0.75) as f32)
    }

    fn per_expert(shape: &Shape) -> usize {
        2 * shape.intermediate * shape.hidden + shape.hidden * shape.intermediate
    }

    fn build_arena(shape: &Shape, experts: usize) -> Arc<HostBlocks> {
        let per = per_expert(shape);
        let arena = HostBlocks::new(u32::try_from(experts).unwrap(), per).unwrap();
        let mut values = vec![bf16::ZERO; per];
        for expert in 0..experts {
            for j in 0..2 * shape.intermediate {
                for k in 0..shape.hidden {
                    values[j * shape.hidden + k] =
                        pattern(expert * 7919 + j * shape.hidden + k, 0.11);
                }
            }
            let down = 2 * shape.intermediate * shape.hidden;
            for j in 0..shape.hidden {
                for k in 0..shape.intermediate {
                    values[down + j * shape.intermediate + k] =
                        pattern(expert * 104729 + j * shape.intermediate + k, 1.7);
                }
            }
            write_block(&arena, u32::try_from(expert).unwrap(), &values);
        }
        arena
    }

    fn write_block(arena: &Arc<HostBlocks>, block: u32, values: &[bf16]) {
        let mut claim = arena.claim(block).unwrap();
        claim.slice().unwrap().copy_from_slice(values);
    }

    fn arena_flat(arena: &Arc<HostBlocks>) -> Vec<bf16> {
        let blocks: Vec<u32> = (0..arena.blocks()).collect();
        let lease = arena.lease(&blocks).unwrap();
        let mut flat = Vec::with_capacity(arena.elements());
        for block in blocks {
            flat.extend_from_slice(lease.read(block).unwrap());
        }
        flat
    }

    fn weights(shape: &Shape, expert: usize) -> ExpertWeights {
        let per = per_expert(shape);
        let base = expert * per;
        let gate_up_len = 2 * shape.intermediate * shape.hidden;
        ExpertWeights {
            gate_up: base..base + gate_up_len,
            gate_up_stride: shape.hidden,
            down: base + gate_up_len..base + per,
            down_stride: shape.intermediate,
        }
    }

    fn hidden_row(shape: &Shape, row: usize) -> Vec<bf16> {
        (0..shape.hidden)
            .map(|k| pattern(row * 4099 + k, 2.3))
            .collect()
    }

    /// FP64 accumulation with the model's BF16 rounding points (f64 → f32 → bf16).
    fn reference(
        shape: &Shape,
        arena: &[bf16],
        weights: &ExpertWeights,
        hidden: &[bf16],
        routing: bf16,
    ) -> Vec<bf16> {
        let (h, i) = (shape.hidden, shape.intermediate);
        let gate_up = &arena[weights.gate_up.clone()];
        let mut gate = vec![0.0f64; i];
        let mut up = vec![0.0f64; i];
        for j in 0..i {
            for k in 0..h {
                gate[j] +=
                    f64::from(hidden[k]) * f64::from(gate_up[j * weights.gate_up_stride + k]);
                up[j] +=
                    f64::from(hidden[k]) * f64::from(gate_up[(i + j) * weights.gate_up_stride + k]);
            }
        }
        let mut activated = vec![0.0f64; i];
        for j in 0..i {
            let gate = f64::from(bf16::from_f32(gate[j] as f32));
            let up = f64::from(bf16::from_f32(up[j] as f32));
            let silu = gate * (1.0 / (1.0 + (-gate).exp()));
            activated[j] = f64::from(bf16::from_f32((silu * up) as f32));
        }
        let down = &arena[weights.down.clone()];
        let mut out = Vec::with_capacity(h);
        for k in 0..h {
            let mut acc = 0.0f64;
            for j in 0..i {
                acc += activated[j] * f64::from(down[k * weights.down_stride + j]);
            }
            let linear = bf16::from_f32(acc as f32);
            out.push(bf16::from_f32(f32::from(linear) * f32::from(routing)));
        }
        out
    }

    fn order_key(value: bf16) -> u32 {
        let bits = u32::from(value.to_bits());
        if bits & 0x8000 != 0 {
            (!bits) & 0xffff
        } else {
            bits | 0x8000
        }
    }

    #[test]
    fn scalar_matches_the_fp64_reference_at_the_bf16_boundaries() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 4);
        let mut scratch = Scratch::new(&shape);
        let mut elements = 0usize;
        let mut differing = 0usize;
        let mut worst = 0u32;
        for (expert, routing) in [(0usize, 0.3125f32), (1, -0.5), (2, 0.0625), (3, 1.0)] {
            let hidden = hidden_row(&shape, expert);
            let routing = bf16::from_f32(routing);
            let mut out = vec![bf16::ZERO; shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape: &shape,
                    arena: &arena_flat(&arena),
                    weights: &weights(&shape, expert),
                    hidden: &hidden,
                    routing,
                },
                &mut out,
                &mut scratch,
            )?;
            let expected = reference(
                &shape,
                &arena_flat(&arena),
                &weights(&shape, expert),
                &hidden,
                routing,
            );
            for (actual, expected) in out.iter().zip(&expected) {
                let distance = order_key(*actual).abs_diff(order_key(*expected));
                elements += 1;
                differing += usize::from(distance != 0);
                worst = worst.max(distance);
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "scalar_vs_fp64": {
                    "elements": elements,
                    "differing": differing,
                    "max_bf16_steps": worst,
                }
            })
        );
        assert!(
            worst <= 1,
            "FP32 accumulation moved a result by more than one BF16 step"
        );
        Ok(())
    }

    #[test]
    fn detected_arms_match_the_scalar_arm_bitwise() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 4);
        let mut scratch = Scratch::new(&shape);
        for (expert, routing) in [(0usize, 0.3125f32), (1, -0.5), (2, 0.0625), (3, 1.0)] {
            let hidden = hidden_row(&shape, expert);
            let weights0 = weights(&shape, expert);
            let routing = bf16::from_f32(routing);
            let mut scalar = vec![bf16::ZERO; shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape: &shape,
                    arena: &arena_flat(&arena),
                    weights: &weights0,
                    hidden: &hidden,
                    routing,
                },
                &mut scalar,
                &mut scratch,
            )?;
            for isa in [Isa::Avx2, Isa::Avx512Bf16] {
                if !isa.available() {
                    println!("arm {} unavailable", isa.name());
                    continue;
                }
                let mut out = vec![bf16::ZERO; shape.hidden];
                contribution(
                    isa,
                    Contribution {
                        shape: &shape,
                        arena: &arena_flat(&arena),
                        weights: &weights0,
                        hidden: &hidden,
                        routing,
                    },
                    &mut out,
                    &mut scratch,
                )?;
                assert!(
                    out.iter()
                        .zip(&scalar)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{} differs from the scalar arm",
                    isa.name()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn gate_up_split_and_expert_pointer_controls_change_the_result() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 2);
        let hidden = hidden_row(&shape, 0);
        let routing = bf16::from_f32(0.25);
        let mut scratch = Scratch::new(&shape);
        let mut correct = vec![bf16::ZERO; shape.hidden];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &arena_flat(&arena),
                weights: &weights(&shape, 0),
                hidden: &hidden,
                routing,
            },
            &mut correct,
            &mut scratch,
        )?;
        let mut wrong_expert = vec![bf16::ZERO; shape.hidden];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &arena_flat(&arena),
                weights: &weights(&shape, 1),
                hidden: &hidden,
                routing,
            },
            &mut wrong_expert,
            &mut scratch,
        )?;
        assert!(
            correct != wrong_expert,
            "wrong expert pointer produced identical output"
        );
        let (h, i) = (shape.hidden, shape.intermediate);
        let flat = arena_flat(&arena);
        let mut shifted: Vec<bf16> = flat.clone();
        let source = &flat[..2 * i * h];
        for j in 0..i - 1 {
            shifted[(i + j) * h..][..h].copy_from_slice(&source[(i + j + 1) * h..][..h]);
        }
        shifted[(2 * i - 1) * h..][..h].copy_from_slice(&source[i * h..][..h]);
        let shifted_weights = ExpertWeights {
            gate_up: 0..2 * i * h,
            gate_up_stride: h,
            down: 2 * i * h..per_expert(&shape),
            down_stride: i,
        };
        let mut split = vec![bf16::ZERO; shape.hidden];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &shifted,
                weights: &shifted_weights,
                hidden: &hidden,
                routing,
            },
            &mut split,
            &mut scratch,
        )?;
        assert!(
            correct != split,
            "gate/up split shifted by one row produced identical output"
        );
        Ok(())
    }

    #[test]
    fn pool_matches_single_threaded_contributions() -> Result<()> {
        let shape = Shape::new(128, 64)?;
        let arena = build_arena(&shape, 8);
        let mut scratch = Scratch::new(&shape);
        let rows = 8usize;
        let hidden: Vec<bf16> = (0..rows * shape.hidden).map(|k| pattern(k, 0.7)).collect();
        for experts_used in [1usize, 2, 5, 8] {
            let mut assignments = Vec::new();
            for row in 0..rows {
                for choice in 0..8 {
                    assignments.push(Assignment {
                        row: row as u32,
                        choice: (row * 8 + choice) as u32,
                        expert: ((row * 3 + choice) % experts_used) as u32,
                        weight: bf16::from_f32(0.125 * (choice as f32 + 1.0)),
                    });
                }
            }
            let experts: Vec<_> = (0..experts_used).map(|e| weights(&shape, e)).collect();
            let mut expected = vec![bf16::ZERO; assignments.len() * shape.hidden];
            for assignment in &assignments {
                let row = &hidden[assignment.row as usize * shape.hidden..][..shape.hidden];
                let out =
                    &mut expected[assignment.choice as usize * shape.hidden..][..shape.hidden];
                contribution(
                    Isa::Scalar,
                    Contribution {
                        shape: &shape,
                        arena: &arena_flat(&arena),
                        weights: &experts[assignment.expert as usize],
                        hidden: row,
                        routing: assignment.weight,
                    },
                    out,
                    &mut scratch,
                )?;
            }
            for workers in [1usize, 2, 4, 8] {
                let pool = ExpertPool::new(workers, shape, rows, assignments.len())?;
                let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
                pool.run(Job {
                    arena: arena.clone(),
                    hidden: &hidden,
                    rows,
                    experts: &experts,
                    assignments: &assignments,
                    out: &mut out,
                })?;
                assert_eq!(out, expected, "workers {workers}, experts {experts_used}");
                // Assignment order must not move results: `out` is indexed by choice slot.
                let mut shuffled = assignments.clone();
                shuffled.reverse();
                let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
                pool.run(Job {
                    arena: arena.clone(),
                    hidden: &hidden,
                    rows,
                    experts: &experts,
                    assignments: &shuffled,
                    out: &mut out,
                })?;
                assert_eq!(
                    out, expected,
                    "reversed assignment order, workers {workers}, experts {experts_used}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn pool_production_shape_matches_scalar() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 4);
        let hidden = hidden_row(&shape, 0);
        let experts: Vec<_> = (0..4).map(|e| weights(&shape, e)).collect();
        let assignments: Vec<_> = (0..8)
            .map(|choice| Assignment {
                row: 0,
                choice,
                expert: choice % 4,
                weight: bf16::from_f32(0.125),
            })
            .collect();
        let mut scratch = Scratch::new(&shape);
        let mut expected = vec![bf16::ZERO; assignments.len() * shape.hidden];
        for assignment in &assignments {
            let out = &mut expected[assignment.choice as usize * shape.hidden..][..shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape: &shape,
                    arena: &arena_flat(&arena),
                    weights: &experts[assignment.expert as usize],
                    hidden: &hidden,
                    routing: assignment.weight,
                },
                out,
                &mut scratch,
            )?;
        }
        let pool = ExpertPool::new(4, shape, 1, assignments.len())?;
        let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
        pool.run(Job {
            arena,
            hidden: &hidden,
            rows: 1,
            experts: &experts,
            assignments: &assignments,
            out: &mut out,
        })?;
        assert_eq!(out, expected);
        Ok(())
    }

    #[test]
    fn pool_rejects_invalid_jobs() -> Result<()> {
        let shape = Shape::new(128, 64)?;
        let arena = build_arena(&shape, 1);
        let experts = [weights(&shape, 0)];
        let hidden = vec![bf16::ZERO; shape.hidden];
        let pool = ExpertPool::new(1, shape, 1, 2)?;
        let good = Assignment {
            row: 0,
            choice: 0,
            expert: 0,
            weight: bf16::from_f32(0.5),
        };
        let mut out = vec![bf16::ZERO; shape.hidden];
        pool.run(Job {
            arena: arena.clone(),
            hidden: &hidden,
            rows: 1,
            experts: &experts,
            assignments: &[good],
            out: &mut out,
        })?;
        assert!(ExpertPool::new(0, shape, 1, 1).is_err());
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 2,
                experts: &experts,
                assignments: &[good],
                out: &mut out,
            })
            .is_err()
        );
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden[..shape.hidden - 1],
                rows: 1,
                experts: &experts,
                assignments: &[good],
                out: &mut out,
            })
            .is_err()
        );
        let bad_row = Assignment { row: 1, ..good };
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[bad_row],
                out: &mut out,
            })
            .is_err()
        );
        let bad_expert = Assignment { expert: 1, ..good };
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[bad_expert],
                out: &mut out,
            })
            .is_err()
        );
        let duplicate = Assignment { choice: 0, ..good };
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[good, duplicate],
                out: &mut out,
            })
            .is_err()
        );
        let bad_choice = Assignment { choice: 1, ..good };
        assert!(
            pool.run(Job {
                arena,
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[good, bad_choice],
                out: &mut out,
            })
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_T4_FIXTURE with the torch-CPU model-definition contribution"]
    fn model_definition_matches_torch_cpu() -> Result<()> {
        let path = std::env::var("FF_T4_FIXTURE").context("FF_T4_FIXTURE missing")?;
        let bytes = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let read = |name: &str| -> Result<Vec<bf16>> {
            let tensor = tensors.tensor(name)?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::BF16,
                "{name} must be BF16"
            );
            Ok(tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
                .collect())
        };
        let shape = production_shape();
        let hidden = read("x")?;
        let gate_up = read("gate_up")?;
        let down = read("down")?;
        let weight = read("weight")?;
        let expected = read("contribution")?;
        ensure!(
            hidden.len() == shape.hidden
                && gate_up.len() == 2 * shape.intermediate * shape.hidden
                && down.len() == shape.hidden * shape.intermediate
                && weight.len() == 1
                && expected.len() == shape.hidden,
            "torch fixture shape"
        );
        let mut values = gate_up;
        values.extend_from_slice(&down);
        let arena: Arc<[bf16]> = Arc::from(values.into_boxed_slice());
        let weights = ExpertWeights {
            gate_up: 0..2 * shape.intermediate * shape.hidden,
            gate_up_stride: shape.hidden,
            down: 2 * shape.intermediate * shape.hidden..per_expert(&shape),
            down_stride: shape.intermediate,
        };
        let mut scratch = Scratch::new(&shape);
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                continue;
            }
            let mut out = vec![bf16::ZERO; shape.hidden];
            contribution(
                isa,
                Contribution {
                    shape: &shape,
                    arena: &arena,
                    weights: &weights,
                    hidden: &hidden,
                    routing: weight[0],
                },
                &mut out,
                &mut scratch,
            )?;
            let mut different = 0usize;
            let mut worst = 0u32;
            for (actual, reference) in out.iter().zip(&expected) {
                let distance = order_key(*actual).abs_diff(order_key(*reference));
                different += usize::from(distance != 0);
                worst = worst.max(distance);
            }
            println!(
                "{}",
                serde_json::json!({
                    "torch_parity": {
                        "isa": isa.name(),
                        "elements": out.len(),
                        "different": different,
                        "max_bf16_steps": worst,
                    }
                })
            );
            // Torch's own FP32 accumulation deviates from FP64 by up to 2 BF16 steps on this fixture.
            assert!(
                worst <= 2,
                "{} differs from the torch contribution beyond the reference noise",
                isa.name()
            );
        }
        Ok(())
    }

    /// 12.5x the 64 MiB LLC of the Ryzen 7950X; 128 x 6,291,456 B = 805 MB.
    const RING_EXPERTS: usize = 128;

    #[test]
    #[ignore = "descriptive CPU timing table"]
    fn service_time_table() -> Result<()> {
        let shape = production_shape();
        let per = per_expert(&shape);
        println!(
            "{}",
            serde_json::json!({
                "table": "service_time",
                "isa": Isa::detect().name(),
                "ring_experts": RING_EXPERTS,
                "ring_bytes": RING_EXPERTS * per * 2,
                "expert_bytes": per * 2,
            })
        );
        let cells = measure_service_table(
            shape,
            RING_EXPERTS as u32,
            &[1, 2, 4, 8, 16, 24],
            &[1, 2, 5, 8],
            &[1, 2, 4, 8],
        )?;
        for cell in cells {
            let cold_bytes = f64::from(cell.experts) * per as f64 * 2.0;
            let flops = f64::from(cell.rows * cell.experts)
                * 6.0
                * shape.hidden as f64
                * shape.intermediate as f64;
            println!(
                "{}",
                serde_json::json!({
                    "workers": cell.workers,
                    "rows": cell.rows,
                    "missed": cell.experts,
                    "seconds": cell.seconds,
                    "GFLOPs": flops / cell.seconds / 1e9,
                    "cold_GB_s": cold_bytes / cell.seconds / 1e9,
                })
            );
        }
        Ok(())
    }
}
