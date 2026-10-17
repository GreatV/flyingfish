//! MoE prefill op: per-expert count/scan/permute, two cuBLAS grouped GEMMs
//! (cublasGemmGroupedBatchedEx), SwiGLU and the weighted combine with the
//! shared expert (kernels/moe_prefill.cu).
//!
//! Grouped-GEMM column-major mapping of the row-major buffers:
//!   GEMM1 per expert e with m_e routed rows:
//!     gemm(opA=T, opB=N, m=2I, n=m_e, k=H, lda=H, ldb=H, ldc=2I)
//!     A = gate_up_e row-major [2I,H] = column-major [H,2I]; op(A)=[2I,H].
//!     B = X_perm_e row-major [m_e,H] = column-major [H,m_e].
//!     C = H1_e row-major [m_e,2I] = column-major [2I,m_e];
//!     C = op(A)·B = H1_e^T, i.e. H1_e = X_perm_e·gate_up_e^T.
//!   GEMM2 per expert e:
//!     gemm(opA=T, opB=N, m=H, n=m_e, k=I, lda=I, ldb=I, ldc=H)
//!     A = down_e row-major [H,I]; B = act_e [m_e,I]; C = Y_e [m_e,H] =
//!     act_e·down_e^T.
//!   Shared expert runs as extra problems over all rows: gate and up as one
//!   m=I, n=rows, k=H group of size 2 writing the two halves of H1s
//!   [rows,2I] (ldc=2I), down as one m=H, n=rows, k=I problem (FP32
//!   accumulate, BF16 out, alpha=1, beta=0). It shares the grouped call
//!   instead of the Blas cuBLASLt path: the Lt path requires setup-time
//!   plans, calibration state and a runtime directory, and the numerics
//!   (CUBLAS_COMPUTE_32F in, BF16 out) are identical here.
//!
//! Count sync: the m/n/k/ld arrays of this cuBLAS API are host-side, so
//! after the permute launch the op copies counts[E], the active-assignment
//! total, the scan total and the two slot tables to the host and
//! synchronizes the stream once per call; per-expert problem sizes and the
//! A/B/C device pointer arrays are built from those host values and uploaded
//! back. The A pointers of routed experts come from the device slot table
//! values, so weights may live in non-contiguous staging buffers.
//!
//! Active cohort and slot generation (T3 revision 2026-10-17): `active
//! [rows,top_k]` selects the assignments this launch computes; inactive
//! assignments are never dereferenced and trip no guard. An active
//! assignment on a slot with slot_valid == 0 or slot_generation !=
//! assignment_gen (the per-assignment snapshot at routing time) traps. Permute stores the (row, choice) provenance
//! pair per slot (src_row/src_choice) plus the inverse map inv_pos[a] = pos;
//! the combine applies exactly the active (row, choice) pairs once and
//! checks each join against the stored pair. The scan total is asserted
//! equal to the count kernel's active-assignment total.
//!
//! Zero-count experts are excluded from the problem list on the host; the
//! cuBLAS 13.4 documentation of cublasGemmGroupedBatchedEx only rejects
//! negative m/n/k/group_size (CUBLAS_STATUS_INVALID_VALUE) and states no
//! semantics for zero-sized problems, so none are passed.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
use anyhow::{Context, Result, ensure};
use cudarc::{
    cublas::{CudaBlas, sys as bs},
    driver::{
        CudaContext, CudaEvent, CudaFunction, CudaStream, LaunchConfig, PushKernelArg, sys as ds,
    },
};
use half::bf16;
use std::sync::Arc;

const SCAN_THREADS: u32 = 1024;
const BLOCK_THREADS: u32 = 256;
const MAX_EXPERTS: usize = 1024;
const MAX_TOP_K: usize = 32;
/// Region alignment inside the scratch buffer.
const REGION: u64 = 256;
/// cuBLAS BF16 pointer rule: (k*2)%16==0 needs 16B, (k*2)%4==0 needs 4B.
const ALIGN_BYTES: u64 = 16;
const ALIGN_MIN_BYTES: u64 = 4;
/// grid.y limit of the SwiGLU launch.
const MAX_GRID_Y: u64 = 65535;

#[derive(Debug)]
pub(crate) struct MoePrefill {
    pub x: Tensor<bf16, 2>,
    pub ids: Tensor<i32, 2>,
    pub active: Tensor<u32, 2>,
    pub weights: Tensor<bf16, 2>,
    pub slots_gate_up: Tensor<u64, 1>,
    pub slot_generation: Tensor<u64, 1>,
    pub assignment_gen: Tensor<u64, 2>,
    pub slots_down: Tensor<u64, 1>,
    pub slot_valid: Tensor<u32, 1>,
    pub shared_gate: Tensor<bf16, 2>,
    pub shared_up: Tensor<bf16, 2>,
    pub shared_down: Tensor<bf16, 2>,
    pub shared_expert_gate: Tensor<bf16, 1>,
    pub y: TensorMut<bf16, 2>,
}

#[derive(Debug)]
pub(crate) struct PrefillPlan {
    rows: u32,
    top_k: u32,
    experts: u32,
    hidden: u32,
    inter: u32,
    x_stride: u64,
    assignments: u64,
    layout: PrefillLayout,
}

impl MoePrefill {
    pub(crate) fn validate(&self) -> Result<PrefillPlan> {
        self.x.validate()?;
        self.y.read().validate()?;
        let (rows, hidden) = (self.x.shape[0], self.x.shape[1]);
        ensure!(
            self.x.stride[1] == 1 && self.x.stride[0] >= u64::from(hidden),
            "moe prefill requires unit inner stride on x"
        );
        ensure!(
            self.y.read().shape == [rows, hidden] && self.y.read().stride == [u64::from(hidden), 1],
            "moe prefill y must be a dense [rows, hidden] view"
        );
        self.ids.validate()?;
        self.active.validate()?;
        self.weights.validate()?;
        self.assignment_gen.validate()?;
        let top_k = self.ids.shape[1];
        ensure!(
            self.ids.shape == [rows, top_k]
                && self.ids.stride == [u64::from(top_k), 1]
                && self.active.shape == [rows, top_k]
                && self.active.stride == [u64::from(top_k), 1]
                && self.weights.shape == [rows, top_k]
                && self.weights.stride == [u64::from(top_k), 1]
                && self.assignment_gen.shape == [rows, top_k]
                && self.assignment_gen.stride == [u64::from(top_k), 1],
            "moe prefill ids/active/weights/assignment_gen must be dense [rows, top_k] views"
        );
        let experts = self.slots_gate_up.shape[0];
        self.slots_gate_up.validate()?;
        self.slots_down.validate()?;
        self.slot_valid.validate()?;
        self.slot_generation.validate()?;
        ensure!(
            self.slots_gate_up.shape == [experts]
                && self.slots_gate_up.stride == [1]
                && self.slots_down.shape == [experts]
                && self.slots_down.stride == [1]
                && self.slot_valid.shape == [experts]
                && self.slot_valid.stride == [1]
                && self.slot_generation.shape == [experts]
                && self.slot_generation.stride == [1],
            "moe prefill slot tables must be dense [experts] views"
        );
        let inter = self.shared_gate.shape[0];
        self.shared_gate.validate()?;
        self.shared_up.validate()?;
        self.shared_down.validate()?;
        self.shared_expert_gate.validate()?;
        ensure!(
            self.shared_gate.shape == [inter, hidden]
                && self.shared_gate.stride == [u64::from(hidden), 1]
                && self.shared_up.shape == [inter, hidden]
                && self.shared_up.stride == [u64::from(hidden), 1]
                && self.shared_down.shape == [hidden, inter]
                && self.shared_down.stride == [u64::from(inter), 1]
                && self.shared_expert_gate.shape == [hidden]
                && self.shared_expert_gate.stride == [1],
            "moe prefill shared expert layout mismatch"
        );
        ensure!(
            experts as usize <= MAX_EXPERTS,
            "moe prefill expert count above 1024"
        );
        ensure!(
            top_k >= 1 && top_k as usize <= MAX_TOP_K,
            "moe prefill top_k outside [1,32]"
        );
        ensure!(
            hidden.is_multiple_of(2) && inter.is_multiple_of(2),
            "moe prefill hidden/inter must be even for cuBLAS BF16 alignment"
        );
        ensure!(
            u64::from(hidden) <= i32::MAX as u64
                && u64::from(inter) <= i32::MAX as u64
                && self.x.stride[0] <= i32::MAX as u64,
            "moe prefill dimensions exceed the cuBLAS grouped i32 leading dims"
        );
        let assignments = u64::from(rows) * u64::from(top_k);
        ensure!(
            assignments <= i32::MAX as u64,
            "moe prefill assignment count exceeds i32"
        );
        ensure!(
            assignments <= MAX_GRID_Y && u64::from(rows) <= MAX_GRID_Y,
            "moe prefill rows exceed the SwiGLU grid"
        );
        no_overlap(&self.x, &self.y.read(), "x", "y")?;
        no_overlap(&self.ids, &self.y.read(), "ids", "y")?;
        no_overlap(&self.weights, &self.y.read(), "weights", "y")?;
        no_overlap(&self.shared_gate, &self.y.read(), "shared_gate", "y")?;
        no_overlap(&self.shared_up, &self.y.read(), "shared_up", "y")?;
        no_overlap(&self.shared_down, &self.y.read(), "shared_down", "y")?;
        no_overlap(
            &self.shared_expert_gate,
            &self.y.read(),
            "shared_expert_gate",
            "y",
        )?;
        let layout = prefill_layout(
            rows as usize,
            top_k as usize,
            experts as usize,
            hidden as usize,
            inter as usize,
        )?;
        Ok(PrefillPlan {
            rows,
            top_k,
            experts,
            hidden,
            inter,
            x_stride: self.x.stride[0],
            assignments,
            layout,
        })
    }
}

fn no_overlap<T: Copy, U: Copy, const R: usize, const S: usize>(
    a: &Tensor<T, R>,
    b: &Tensor<U, S>,
    aname: &str,
    bname: &str,
) -> Result<()> {
    ensure!(
        a.ptr + a.span_bytes <= b.ptr || b.ptr + b.span_bytes <= a.ptr,
        "moe prefill tensors overlap: {aname} vs {bname}"
    );
    Ok(())
}

/// Scratch regions, byte offsets from the scratch base, each 256-aligned.
#[derive(Debug)]
pub(crate) struct PrefillLayout {
    counts: u64,
    active_total: u64,
    offsets: u64,
    cursor: u64,
    src_row: u64,
    src_choice: u64,
    inv_pos: u64,
    w_row: u64,
    x_perm: u64,
    h1: u64,
    act: u64,
    y_perm: u64,
    h1s: u64,
    act_s: u64,
    y_s: u64,
    ptra1: u64,
    ptrb1: u64,
    ptrc1: u64,
    ptra2: u64,
    ptrb2: u64,
    ptrc2: u64,
    bytes: u64,
}

pub(crate) fn prefill_scratch_bytes(
    rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
) -> u64 {
    prefill_layout(rows, top_k, experts, hidden, inter)
        .expect("moe prefill scratch size overflow")
        .bytes
}

fn region(at: &mut u64, elements: usize, element_bytes: u64) -> Result<u64> {
    let len = (elements as u64)
        .checked_mul(element_bytes)
        .context("moe prefill scratch size overflow")?;
    let off = *at;
    *at = at
        .checked_add(len)
        .and_then(|v| v.checked_add(REGION - 1))
        .context("moe prefill scratch size overflow")?
        & !(REGION - 1);
    Ok(off)
}

fn prefill_layout(
    rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<PrefillLayout> {
    let assignments = rows
        .checked_mul(top_k)
        .context("moe prefill assignment overflow")?;
    let mut at = 0u64;
    let counts = region(&mut at, experts, 4)?;
    let active_total = region(&mut at, 1, 4)?;
    let offsets = region(&mut at, experts + 1, 4)?;
    let cursor = region(&mut at, experts, 4)?;
    let src_row = region(&mut at, assignments, 4)?;
    let src_choice = region(&mut at, assignments, 4)?;
    let inv_pos = region(&mut at, assignments, 4)?;
    let w_row = region(&mut at, assignments, 2)?;
    let x_perm = region(&mut at, assignments * hidden, 2)?;
    let h1 = region(&mut at, assignments * 2 * inter, 2)?;
    let act = region(&mut at, assignments * inter, 2)?;
    let y_perm = region(&mut at, assignments * hidden, 2)?;
    let h1s = region(&mut at, rows * 2 * inter, 2)?;
    let act_s = region(&mut at, rows * inter, 2)?;
    let y_s = region(&mut at, rows * hidden, 2)?;
    let ptra1 = region(&mut at, experts + 2, 8)?;
    let ptrb1 = region(&mut at, experts + 2, 8)?;
    let ptrc1 = region(&mut at, experts + 2, 8)?;
    let ptra2 = region(&mut at, experts + 2, 8)?;
    let ptrb2 = region(&mut at, experts + 2, 8)?;
    let ptrc2 = region(&mut at, experts + 2, 8)?;
    Ok(PrefillLayout {
        counts,
        active_total,
        offsets,
        cursor,
        src_row,
        src_choice,
        inv_pos,
        w_row,
        x_perm,
        h1,
        act,
        y_perm,
        h1s,
        act_s,
        y_s,
        ptra1,
        ptrb1,
        ptrc1,
        ptra2,
        ptrb2,
        ptrc2,
        bytes: at,
    })
}

/// Active experts (count > 0) in problem order: count ascending, id ascending.
pub(crate) fn problem_order(counts: &[i32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..counts.len() as u32)
        .filter(|&e| counts[e as usize] > 0)
        .collect();
    order.sort_by_key(|&e| (counts[e as usize], e));
    order
}

/// Consecutive same-count runs of the problem order as (count, run length).
pub(crate) fn group_runs(order: &[u32], counts: &[i32]) -> Vec<(u32, u32)> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &e in order {
        let count = counts[e as usize] as u32;
        match runs.last_mut() {
            Some(last) if last.0 == count => last.1 += 1,
            _ => runs.push((count, 1)),
        }
    }
    runs
}

struct GemmGroup {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    size: i32,
}

/// Pipeline stages; marks[i] is recorded before stage i starts, marks[i+1]
/// after it. The count-sync host gap lands inside the gemm1 stage, so
/// StageTiming::sync_us reports it separately.
pub(crate) const STAGES: [&str; 7] = [
    "count", "scan", "permute", "gemm1", "swiglu", "gemm2", "combine",
];

pub(crate) struct StageTiming {
    marks: Vec<CudaEvent>,
    /// Host wall time of the counts D2H copies plus the stream synchronize.
    pub(crate) sync_us: f64,
}

impl StageTiming {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let ctx = stream.context();
        let marks = (0..=STAGES.len())
            .map(|_| ctx.new_event(Some(ds::CUevent_flags::CU_EVENT_DEFAULT)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            marks,
            sync_us: 0.0,
        })
    }
    fn mark(&mut self, stream: &Arc<CudaStream>, stage: usize) -> Result<()> {
        self.marks[stage].record(stream)?;
        Ok(())
    }
    /// Stream time of one stage in microseconds.
    pub(crate) fn stage_us(&self, stage: usize) -> Result<f32> {
        Ok(self.marks[stage].elapsed_ms(&self.marks[stage + 1])? * 1000.0)
    }
    pub(crate) fn total_us(&self) -> Result<f32> {
        Ok(self.marks[0].elapsed_ms(&self.marks[STAGES.len()])? * 1000.0)
    }
}

pub(crate) struct PrefillOps {
    count: CudaFunction,
    scan: CudaFunction,
    permute: CudaFunction,
    swiglu: CudaFunction,
    combine: CudaFunction,
    blas: CudaBlas,
}

impl PrefillOps {
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_prefill")?;
        let blas = CudaBlas::new(stream.clone())?;
        unsafe {
            bs::cublasSetMathMode(
                *blas.handle(),
                bs::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        Ok(Self {
            count: module.load_function("moe_count")?,
            scan: module.load_function("moe_scan")?,
            permute: module.load_function("moe_permute")?,
            swiglu: module.load_function("moe_swiglu")?,
            combine: module.load_function("moe_combine")?,
            blas,
        })
    }

    pub fn prefill(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefill,
        scratch: &Scratch,
    ) -> Result<()> {
        self.run(stream, input, scratch, None)
    }

    pub(crate) fn run(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefill,
        scratch: &Scratch,
        mut timing: Option<&mut StageTiming>,
    ) -> Result<()> {
        let plan = input.validate()?;
        scratch.validate(plan.layout.bytes)?;
        let experts = plan.experts as usize;
        let (rows, top_k, hidden, inter) = (plan.rows, plan.top_k, plan.hidden, plan.inter);
        let x_stride = plan.x_stride as i64;
        let layout = &plan.layout;
        let counts_at = scratch.ptr + layout.counts;
        let active_total_at = scratch.ptr + layout.active_total;
        let offsets_at = scratch.ptr + layout.offsets;
        let cursor_at = scratch.ptr + layout.cursor;
        let src_row_at = scratch.ptr + layout.src_row;
        let src_choice_at = scratch.ptr + layout.src_choice;
        let inv_pos_at = scratch.ptr + layout.inv_pos;
        let w_row_at = scratch.ptr + layout.w_row;
        let x_perm_at = scratch.ptr + layout.x_perm;
        let h1_at = scratch.ptr + layout.h1;
        let act_at = scratch.ptr + layout.act;
        let y_perm_at = scratch.ptr + layout.y_perm;
        let h1s_at = scratch.ptr + layout.h1s;
        let act_s_at = scratch.ptr + layout.act_s;
        let y_s_at = scratch.ptr + layout.y_s;

        if let Some(t) = timing.as_mut() {
            t.mark(stream, 0)?;
        }
        unsafe {
            stream
                .launch_builder(&self.count)
                .arg(&input.ids.ptr)
                .arg(&input.active.ptr)
                .arg(&counts_at)
                .arg(&active_total_at)
                .arg(&(plan.assignments as i64))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .launch(grid(1, SCAN_THREADS))?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, 1)?;
        }
        unsafe {
            stream
                .launch_builder(&self.scan)
                .arg(&counts_at)
                .arg(&offsets_at)
                .arg(&cursor_at)
                .arg(&(experts as i32))
                .launch(grid(1, SCAN_THREADS))?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, 2)?;
        }
        unsafe {
            stream
                .launch_builder(&self.permute)
                .arg(&input.ids.ptr)
                .arg(&input.weights.ptr)
                .arg(&input.x.ptr)
                .arg(&offsets_at)
                .arg(&cursor_at)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&input.assignment_gen.ptr)
                .arg(&input.active.ptr)
                .arg(&src_row_at)
                .arg(&src_choice_at)
                .arg(&w_row_at)
                .arg(&inv_pos_at)
                .arg(&x_perm_at)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(experts as i32))
                .arg(&(hidden as i64))
                .arg(&x_stride)
                .launch(grid(plan.assignments as usize, BLOCK_THREADS))?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, 3)?;
        }

        // Count sync: the grouped-GEMM m/n/k/ld arrays are host-side in
        // cuBLAS, so counts, the scan total and the slot tables come to the
        // host here and the stream synchronizes once per prefill call.
        let sync_start = std::time::Instant::now();
        let mut counts = vec![0i32; experts];
        let mut slots_gu = vec![0u64; experts];
        let mut slots_dn = vec![0u64; experts];
        let mut total = [0i32; 1];
        let mut active_total = [0i32; 1];
        unsafe {
            dtoh(stream, counts_at, &mut counts)?;
            dtoh(stream, active_total_at, &mut active_total)?;
            dtoh(stream, input.slots_gate_up.ptr, &mut slots_gu)?;
            dtoh(stream, input.slots_down.ptr, &mut slots_dn)?;
            dtoh(stream, offsets_at + (experts as u64) * 4, &mut total)?;
        }
        stream.synchronize()?;
        ensure!(
            total[0] == active_total[0],
            "moe prefill scan total {} != {} active assignments",
            total[0],
            active_total[0]
        );
        if let Some(t) = timing.as_mut() {
            t.sync_us = sync_start.elapsed().as_secs_f64() * 1e6;
        }

        let order = problem_order(&counts);
        let runs = group_runs(&order, &counts);
        let mut offsets = vec![0i32; experts];
        let mut acc = 0i32;
        for e in 0..experts {
            offsets[e] = acc;
            acc += counts[e];
        }
        let (gu_align, dn_align) = if hidden.is_multiple_of(8) && inter.is_multiple_of(8) {
            (ALIGN_BYTES, ALIGN_BYTES)
        } else {
            (ALIGN_MIN_BYTES, ALIGN_MIN_BYTES)
        };
        for &e in &order {
            let e = e as usize;
            ensure!(
                slots_gu[e] != 0 && slots_dn[e] != 0,
                "null expert slot pointer"
            );
            ensure!(
                slots_gu[e].is_multiple_of(gu_align),
                "gate_up slot pointer of expert {e} below cuBLAS alignment"
            );
            ensure!(
                slots_dn[e].is_multiple_of(dn_align),
                "down slot pointer of expert {e} below cuBLAS alignment"
            );
        }
        let mut a1 = Vec::with_capacity(order.len() + 2);
        let mut b1 = Vec::with_capacity(order.len() + 2);
        let mut c1 = Vec::with_capacity(order.len() + 2);
        let mut a2 = Vec::with_capacity(order.len() + 1);
        let mut b2 = Vec::with_capacity(order.len() + 1);
        let mut c2 = Vec::with_capacity(order.len() + 1);
        let mut groups1 = Vec::with_capacity(runs.len() + 1);
        let mut groups2 = Vec::with_capacity(runs.len() + 1);
        for &(count, size) in &runs {
            groups1.push(GemmGroup {
                m: (2 * inter) as i32,
                n: count as i32,
                k: hidden as i32,
                lda: hidden as i32,
                ldb: hidden as i32,
                ldc: (2 * inter) as i32,
                size: size as i32,
            });
            groups2.push(GemmGroup {
                m: hidden as i32,
                n: count as i32,
                k: inter as i32,
                lda: inter as i32,
                ldb: inter as i32,
                ldc: hidden as i32,
                size: size as i32,
            });
        }
        for &e in &order {
            let off = offsets[e as usize] as u64;
            a1.push(slots_gu[e as usize]);
            b1.push(x_perm_at + elem(off * u64::from(hidden)));
            c1.push(h1_at + elem(off * u64::from(2 * inter)));
            a2.push(slots_dn[e as usize]);
            b2.push(act_at + elem(off * u64::from(inter)));
            c2.push(y_perm_at + elem(off * u64::from(hidden)));
        }
        a1.push(input.shared_gate.ptr);
        b1.push(input.x.ptr);
        c1.push(h1s_at);
        a1.push(input.shared_up.ptr);
        b1.push(input.x.ptr);
        c1.push(h1s_at + elem(u64::from(inter)));
        groups1.push(GemmGroup {
            m: inter as i32,
            n: rows as i32,
            k: hidden as i32,
            lda: hidden as i32,
            ldb: plan.x_stride as i32,
            ldc: (2 * inter) as i32,
            size: 2,
        });
        a2.push(input.shared_down.ptr);
        b2.push(act_s_at);
        c2.push(y_s_at);
        groups2.push(GemmGroup {
            m: hidden as i32,
            n: rows as i32,
            k: inter as i32,
            lda: inter as i32,
            ldb: inter as i32,
            ldc: hidden as i32,
            size: 1,
        });

        unsafe {
            htod(stream, scratch.ptr + layout.ptra1, &a1)?;
            htod(stream, scratch.ptr + layout.ptrb1, &b1)?;
            htod(stream, scratch.ptr + layout.ptrc1, &c1)?;
            self.grouped_gemm(
                scratch.ptr + layout.ptra1,
                scratch.ptr + layout.ptrb1,
                scratch.ptr + layout.ptrc1,
                &groups1,
            )
            .context("moe prefill grouped GEMM 1")?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, 4)?;
        }

        let swiglu_grid = |n: u32, rows_y: u32| LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK_THREADS), rows_y, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.swiglu)
                .arg(&h1_at)
                .arg(&act_at)
                .arg(&(active_total[0] as i64))
                .arg(&(inter as i32))
                .launch(swiglu_grid(inter, active_total[0] as u32))?;
            stream
                .launch_builder(&self.swiglu)
                .arg(&h1s_at)
                .arg(&act_s_at)
                .arg(&(u64::from(rows) as i64))
                .arg(&(inter as i32))
                .launch(swiglu_grid(inter, rows))?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, 5)?;
        }

        unsafe {
            htod(stream, scratch.ptr + layout.ptra2, &a2)?;
            htod(stream, scratch.ptr + layout.ptrb2, &b2)?;
            htod(stream, scratch.ptr + layout.ptrc2, &c2)?;
            self.grouped_gemm(
                scratch.ptr + layout.ptra2,
                scratch.ptr + layout.ptrb2,
                scratch.ptr + layout.ptrc2,
                &groups2,
            )
            .context("moe prefill grouped GEMM 2")?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, 6)?;
        }

        unsafe {
            stream
                .launch_builder(&self.combine)
                .arg(&y_perm_at)
                .arg(&w_row_at)
                .arg(&inv_pos_at)
                .arg(&src_row_at)
                .arg(&src_choice_at)
                .arg(&input.active.ptr)
                .arg(&input.x.ptr)
                .arg(&input.shared_expert_gate.ptr)
                .arg(&y_s_at)
                .arg(&input.y.0.ptr)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i64))
                .arg(&x_stride)
                .launch(grid(rows as usize, BLOCK_THREADS))?;
        }
        if let Some(t) = timing.as_mut() {
            t.mark(stream, STAGES.len())?;
        }
        Ok(())
    }

    /// cublasGemmGroupedBatchedEx with BF16 A/B/C, CUBLAS_COMPUTE_32F,
    /// alpha=1, beta=0, opA=T, opB=N for every group. A/B/C are device
    /// addresses of the per-problem pointer arrays (problem_count entries).
    ///
    /// # Safety
    /// A/B/C must address problem_count device pointers whose targets match
    /// the group shapes and are kept alive until the stream completes.
    unsafe fn grouped_gemm(&self, a: u64, b: u64, c: u64, groups: &[GemmGroup]) -> Result<()> {
        let n = groups.len();
        let transa = vec![bs::cublasOperation_t::CUBLAS_OP_T; n];
        let transb = vec![bs::cublasOperation_t::CUBLAS_OP_N; n];
        let m: Vec<i32> = groups.iter().map(|g| g.m).collect();
        let cols: Vec<i32> = groups.iter().map(|g| g.n).collect();
        let k: Vec<i32> = groups.iter().map(|g| g.k).collect();
        let lda: Vec<i32> = groups.iter().map(|g| g.lda).collect();
        let ldb: Vec<i32> = groups.iter().map(|g| g.ldb).collect();
        let ldc: Vec<i32> = groups.iter().map(|g| g.ldc).collect();
        let alpha = vec![1f32; n];
        let beta = vec![0f32; n];
        let size: Vec<i32> = groups.iter().map(|g| g.size).collect();
        unsafe {
            bs::cublasGemmGroupedBatchedEx(
                *self.blas.handle(),
                transa.as_ptr(),
                transb.as_ptr(),
                m.as_ptr(),
                cols.as_ptr(),
                k.as_ptr(),
                alpha.as_ptr().cast(),
                a as *const *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                lda.as_ptr(),
                b as *const *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                ldb.as_ptr(),
                beta.as_ptr().cast(),
                c as *mut *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                ldc.as_ptr(),
                n as i32,
                size.as_ptr(),
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            )
            .result()?;
        }
        Ok(())
    }
}

fn elem(count: u64) -> u64 {
    count * 2
}

/// # Safety
/// `src` must address `dst.len()` elements of device memory ordered before
/// this call on the stream.
unsafe fn dtoh<T>(stream: &Arc<CudaStream>, src: u64, dst: &mut [T]) -> Result<()> {
    stream.context().bind_to_thread()?;
    unsafe {
        ds::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().cast(),
            src,
            std::mem::size_of_val(dst),
            stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// # Safety
/// `dst` must address `src.len()` elements of writable device memory that no
/// in-flight work on the stream touches.
unsafe fn htod<T>(stream: &Arc<CudaStream>, dst: u64, src: &[T]) -> Result<()> {
    stream.context().bind_to_thread()?;
    unsafe {
        ds::cuMemcpyHtoDAsync_v2(
            dst,
            src.as_ptr().cast(),
            std::mem::size_of_val(src),
            stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}
