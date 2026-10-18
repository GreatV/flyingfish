//! MoE prefill op: per-expert count/scan/permute, two cuBLAS grouped GEMM
//! stages (cublasGemmGroupedBatchedEx, one problem per expert), silu_mul
//! activation (kernels/act.cu), the per-assignment scatter (moe_scatter) and
//! the shared combine (kernels/moe_combine.cu).
//!
//! Grouped-GEMM column-major mapping of the row-major buffers:
//!   GEMM1 per expert e with m_e routed rows:
//!     gemm(opA=T, opB=N, m=2I, n=m_e, k=H, lda=H, ldb=H, ldc=2I)
//!     A = gate_up_e row-major [2I,H] = column-major [H,2I]; op(A)=[2I,H].
//!     B = X_perm_e row-major [m_e,H] = column-major [H,m_e].
//!     C = H1_e row-major [m_e,2I] = column-major [2I,m_e];
//!     C = op(A)·B = H1_e^T, i.e. H1_e = X_perm_e·gate_up_e^T.
//!   GEMM2 per expert e over the expert's whole segment:
//!     gemm(opA=T, opB=N, m=H, n=m_e, k=I, lda=I, ldb=I, ldc=H)
//!     A = down_e row-major [H,I]; B = act_e [m_e,I]; C = Y_perm_e [m_e,H]
//!     at segment offset offsets[e], so Y_perm is one contiguous
//!     [active_assignments, H] buffer in expert-segment order.
//!   moe_scatter computes contributions[a] = BF16(Y_perm[pos_of[a]] *
//!   weights[a]) on the active assignments (pos_of[a] = the segment position
//!   moe_permute claimed for assignment a), giving the combine its
//!   per-(row, choice) contribution at slot a = row*top_k + choice.
//!   Shared expert runs as extra problems over all rows: gate and up as one
//!   m=I, n=rows, k=H group of size 2 writing the two halves of H1s
//!   [rows,2I] (ldc=2I), down as one m=H, n=rows, k=I problem (FP32
//!   accumulate, BF16 out, alpha=1, beta=0) writing shared_out [rows,H].
//!
//! Count sync and the persistent stage: the m/n/k/ld arrays of this cuBLAS
//! API are host-side, so after the permute launch one pinned D2H copy moves
//! the scratch span [counts, offsets end] (counts[E], the active-assignment
//! total and the scan total) to the host and the stream synchronizes once per
//! call. Per-expert problem sizes and the routed GEMMs' A/B/C device pointer
//! arrays are rebuilt in pinned host mirrors and copied back with one pinned
//! H2D per GEMM call. The stage (pinned D2H window, pinned pointer mirrors,
//! device pointer blocks, reused host Vecs) is created once in
//! `PrefillOps::new` with capacity for `MAX_EXPERTS` experts; expert counts
//! above the capacity error by name in validation, and the copy helpers
//! bound-check every transfer against the stage capacity. The per-expert
//! weight pointers come from the caller's host slot tables; the tables and
//! every input tensor are immutable until the returned event.
//!
//! Contribute/wrapper split: `prefill_contribute` runs
//! count→scan→permute→GEMM1→silu→GEMM2→scatter for one cohort's active
//! assignments into CALLER-OWNED contributions [rows,top_k,hidden] and done
//! [rows,top_k] buffers — no zero/reset of either, no shared FFN, no sigmoid
//! gate, no combine — and returns the completion event of the cohort's last
//! launch. Consecutive cohorts write disjoint (row, choice) sets into the
//! same buffers; the permute done-claim is exactly-once across cohorts. The
//! `prefill` wrapper zeroes the scratch-backed done buffer, contributes the
//! all-active cohort, then runs the shared FFN (one grouped gate/up call,
//! silu, one grouped down call over a single 9-pointer block), the sigmoid
//! gate and the combine, returning the combine event. Per call the
//! contribute path issues one pinned D2H and at most two pinned H2D; the
//! wrapper adds one more 9-pointer H2D.
//!
//! Active cohort and slot generation: `active [rows,top_k]` selects the
//! assignments this launch computes; inactive assignments are never
//! dereferenced and trip no guard. An active assignment on a slot with
//! slot_valid == 0 or slot_generation != assignment_gen (the per-assignment
//! snapshot at routing time) traps. Permute records pos_of[a] and sets
//! done[a] = 1 exactly once (a pre-set done traps); the scan total is
//! asserted equal to the count kernel's active-assignment total. When that
//! total is zero the routed problems (grouped GEMMs, silu, gemm2, scatter)
//! are skipped and the combine writes the shared-only output.
//!
//! Zero-count experts are excluded from the problem list.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
use anyhow::{Context, Result, ensure};
use cudarc::{
    cublas::{CudaBlas, sys as bs},
    driver::{
        CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice,
        PushKernelArg, sys as ds,
    },
};
use half::bf16;
use std::sync::{Arc, Mutex};

use super::combine::{CombineOps, MoeCombine};

const SCAN_THREADS: u32 = 1024;
const BLOCK_THREADS: u32 = 256;
pub(crate) const MAX_EXPERTS: usize = 1024;
const MAX_TOP_K: usize = 32;
/// Region alignment inside the scratch buffer.
const REGION: u64 = 256;
/// cuBLAS BF16 pointer rule: (k*2)%16==0 needs 16B, (k*2)%4==0 needs 4B.
const ALIGN_BYTES: u64 = 16;
const ALIGN_MIN_BYTES: u64 = 4;
/// i32 words of the pinned D2H window at full capacity: counts[E], the
/// active-assignment total and offsets[E+1] with the scratch region paddings.
const D2H_WINDOW_WORDS: usize = 2113;
/// A/B/C pointer block capacities in device pointers: one problem per active
/// expert plus the shared entries.
pub(crate) const GEMM1_PTR_WORDS: usize = 3 * (MAX_EXPERTS + 2);
pub(crate) const GEMM2_PTR_WORDS: usize = 3 * (MAX_EXPERTS + 1);
/// Upper bound of per-GEMM groups: one per same-count run plus the shared
/// group.
const MAX_GROUPS: usize = MAX_EXPERTS + 2;

/// One cohort's contribute input. `contributions` and `done` are
/// CALLER-OWNED and survive across cohorts; every other tensor and both slot
/// tables are immutable until the event returned by
/// `PrefillOps::prefill_contribute`.
#[derive(Debug)]
pub(crate) struct MoePrefillContribute<'a> {
    pub x: Tensor<bf16, 2>,
    pub ids: Tensor<i32, 2>,
    pub weights: Tensor<bf16, 2>,
    /// `[rows, top_k]` THIS cohort's assignment mask.
    pub active: Tensor<u32, 2>,
    pub assignment_gen: Tensor<u64, 2>,
    /// `[experts]` device addresses of each expert's fused gate/up blocks.
    pub slots_gate_up: Tensor<u64, 1>,
    /// `[experts]` device addresses of each expert's down blocks.
    pub slots_down: Tensor<u64, 1>,
    pub slot_valid: Tensor<u32, 1>,
    pub slot_generation: Tensor<u64, 1>,
    /// `[experts]` host snapshot of the slot addresses for the grouped calls.
    pub slots_gate_up_host: &'a [u64],
    pub slots_down_host: &'a [u64],
    pub shared_gate: Tensor<bf16, 2>,
    pub shared_up: Tensor<bf16, 2>,
    pub shared_down: Tensor<bf16, 2>,
    pub shared_expert_gate: Tensor<bf16, 1>,
    /// CALLER-OWNED `[rows, top_k, hidden]` weighted BF16 contributions.
    pub contributions: Tensor<bf16, 3>,
    /// CALLER-OWNED `[rows, top_k]` done bits (zeroed by the first cohort).
    pub done: TensorMut<u32, 2>,
}

/// The wrapper input: the cohort fields plus the output `y`; the wrapper
/// owns `contributions`/`done` in scratch.
#[derive(Debug)]
#[cfg(test)]
pub(crate) struct MoePrefill<'a> {
    pub x: Tensor<bf16, 2>,
    pub ids: Tensor<i32, 2>,
    pub active: Tensor<u32, 2>,
    pub weights: Tensor<bf16, 2>,
    pub slots_gate_up: Tensor<u64, 1>,
    pub slots_down: Tensor<u64, 1>,
    pub slot_valid: Tensor<u32, 1>,
    pub slot_generation: Tensor<u64, 1>,
    pub assignment_gen: Tensor<u64, 2>,
    /// `[experts]` device addresses of each expert's fused gate/up blocks.
    /// Immutable until the event returned by `PrefillOps::prefill`.
    pub slots_gate_up_host: &'a [u64],
    /// `[experts]` device addresses of each expert's down blocks.
    /// Immutable until the event returned by `PrefillOps::prefill`.
    pub slots_down_host: &'a [u64],
    pub shared_gate: Tensor<bf16, 2>,
    pub shared_up: Tensor<bf16, 2>,
    pub shared_down: Tensor<bf16, 2>,
    pub shared_expert_gate: Tensor<bf16, 1>,
    pub y: TensorMut<bf16, 2>,
}

/// The join input: caller-owned contributions and done bits accumulated by
/// any mix of contribute cohorts (GPU, streamed, CPU) plus the final-active
/// union; `prefill_finish` runs the shared FFN, the sigmoid gate and the
/// single combine. No routed arithmetic runs here and `done` is never
/// written.
#[derive(Debug)]
pub(crate) struct MoePrefillFinish {
    pub x: Tensor<bf16, 2>,
    pub ids: Tensor<i32, 2>,
    /// `[rows, top_k]` the completed assignment union the combine checks
    /// `done` against.
    pub final_active: Tensor<u32, 2>,
    /// CALLER-OWNED `[rows, top_k, hidden]` weighted BF16 contributions.
    pub contributions: Tensor<bf16, 3>,
    /// CALLER-OWNED `[rows, top_k]` done bits, one per completed choice.
    pub done: Tensor<u32, 2>,
    /// Slot-table length the caller's buffers were built against; sets the
    /// scratch region offsets.
    pub experts: u32,
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
    assignments: u64,
    layout: PrefillLayout,
}

impl MoePrefillFinish {
    pub(crate) fn validate(&self) -> Result<PrefillPlan> {
        self.x.validate()?;
        let (rows, hidden) = (self.x.shape[0], self.x.shape[1]);
        ensure!(
            self.x.stride == [u64::from(hidden), 1],
            "moe prefill finish x must be a dense [rows, hidden] view (shape {:?})",
            self.x.shape
        );
        self.ids.validate()?;
        let top_k = self.ids.shape[1];
        ensure!(
            self.ids.shape == [rows, top_k] && self.ids.stride == [u64::from(top_k), 1],
            "moe prefill finish ids must be dense [rows, top_k] (shape {:?})",
            self.ids.shape
        );
        for (t, name) in [(&self.final_active, "final_active"), (&self.done, "done")] {
            t.validate()?;
            ensure!(
                t.shape == [rows, top_k] && t.stride == [u64::from(top_k), 1],
                "moe prefill finish {name} must be dense [rows, top_k] (shape {:?})",
                t.shape
            );
        }
        self.contributions.validate()?;
        ensure!(
            self.contributions.shape == [rows, top_k, hidden]
                && self.contributions.stride[2] == 1
                && self.contributions.stride[1] == u64::from(hidden)
                && self.contributions.stride[0] == u64::from(top_k) * u64::from(hidden),
            "moe prefill finish contributions must be dense [rows, top_k, hidden] (shape {:?})",
            self.contributions.shape
        );
        let inter = self.shared_gate.shape[0];
        for (t, name) in [
            (&self.shared_gate, "shared_gate"),
            (&self.shared_up, "shared_up"),
        ] {
            t.validate()?;
            ensure!(
                t.shape == [inter, hidden] && t.stride == [u64::from(hidden), 1],
                "moe prefill finish {name} must be dense [inter, hidden] (shape {:?})",
                t.shape
            );
        }
        self.shared_down.validate()?;
        ensure!(
            self.shared_down.shape == [hidden, inter]
                && self.shared_down.stride == [u64::from(inter), 1],
            "moe prefill finish shared_down must be dense [hidden, inter] (shape {:?})",
            self.shared_down.shape
        );
        self.shared_expert_gate.validate()?;
        ensure!(
            self.shared_expert_gate.shape == [hidden] && self.shared_expert_gate.stride == [1],
            "moe prefill finish shared gate must be dense [hidden]"
        );
        self.y.read().validate()?;
        ensure!(
            self.y.read().shape == [rows, hidden] && self.y.read().stride == [u64::from(hidden), 1],
            "moe prefill finish y must be a dense [rows, hidden] view (shape {:?})",
            self.y.read().shape
        );
        no_overlap(&self.x, &self.y.read(), "x", "y")?;
        no_overlap(&self.ids, &self.y.read(), "ids", "y")?;
        no_overlap(&self.contributions, &self.y.read(), "contributions", "y")?;
        no_overlap(&self.shared_gate, &self.y.read(), "shared_gate", "y")?;
        no_overlap(&self.shared_up, &self.y.read(), "shared_up", "y")?;
        no_overlap(&self.shared_down, &self.y.read(), "shared_down", "y")?;
        no_overlap(
            &self.shared_expert_gate,
            &self.y.read(),
            "shared_expert_gate",
            "y",
        )?;
        ensure!(
            (1..=MAX_EXPERTS).contains(&(self.experts as usize)),
            "moe prefill finish experts {} outside [1, {}]",
            self.experts,
            MAX_EXPERTS
        );
        let layout = prefill_layout(
            rows as usize,
            top_k as usize,
            self.experts as usize,
            hidden as usize,
            inter as usize,
        )?;
        Ok(PrefillPlan {
            rows,
            top_k,
            experts: self.experts,
            hidden,
            inter,
            assignments: u64::from(rows) * u64::from(top_k),
            layout,
        })
    }
}

impl MoePrefillContribute<'_> {
    pub(crate) fn validate(&self) -> Result<PrefillPlan> {
        self.x.validate()?;
        let (rows, hidden) = (self.x.shape[0], self.x.shape[1]);
        ensure!(
            self.x.stride == [u64::from(hidden), 1],
            "moe prefill x must be dense [rows, hidden] (stride {:?}, hidden {hidden})",
            self.x.stride
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
            "moe prefill ids/active/weights/assignment_gen must be dense [rows, top_k] views (top_k {top_k})"
        );
        let experts = self.slot_valid.shape[0];
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
            "moe prefill slot tables must be dense [experts] views (experts {experts})"
        );
        ensure!(
            self.slots_gate_up_host.len() == experts as usize
                && self.slots_down_host.len() == experts as usize,
            "moe prefill host slot tables must hold one pointer per expert (gate_up {}, down {}, experts {experts})",
            self.slots_gate_up_host.len(),
            self.slots_down_host.len()
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
            "moe prefill shared expert layout mismatch (inter {inter}, hidden {hidden})"
        );
        self.contributions.validate()?;
        ensure!(
            self.contributions.shape == [rows, top_k, hidden]
                && self.contributions.stride[2] == 1
                && self.contributions.stride[1] == u64::from(hidden)
                && self.contributions.stride[0] == u64::from(top_k) * u64::from(hidden),
            "moe prefill contributions must be dense [rows, top_k, hidden] (shape {:?}, top_k {top_k}, hidden {hidden})",
            self.contributions.shape
        );
        self.done.read().validate()?;
        ensure!(
            self.done.read().shape == [rows, top_k]
                && self.done.read().stride == [u64::from(top_k), 1],
            "moe prefill done must be a dense [rows, top_k] view (shape {:?}, top_k {top_k})",
            self.done.read().shape
        );
        ensure!(
            experts as usize <= MAX_EXPERTS,
            "moe prefill expert count {experts} above the stage capacity 1024"
        );
        ensure!(
            top_k >= 1 && top_k as usize <= MAX_TOP_K,
            "moe prefill top_k {top_k} outside [1,32]"
        );
        ensure!(
            hidden.is_multiple_of(2) && inter.is_multiple_of(2),
            "moe prefill hidden {hidden} and inter {inter} must be even for cuBLAS BF16 alignment"
        );
        ensure!(
            u64::from(hidden) <= i32::MAX as u64 && u64::from(inter) <= i32::MAX as u64,
            "moe prefill hidden {hidden} and inter {inter} exceed the cuBLAS grouped i32 leading dims"
        );
        let assignments = u64::from(rows) * u64::from(top_k);
        ensure!(
            assignments <= i32::MAX as u64,
            "moe prefill assignment count {assignments} exceeds i32"
        );
        let silu_routed = assignments * u64::from(inter);
        let silu_shared = u64::from(rows) * u64::from(inter);
        ensure!(
            silu_routed <= i32::MAX as u64 && silu_shared <= i32::MAX as u64,
            "moe prefill activation elements assignments*inter {silu_routed} and rows*inter {silu_shared} exceed the silu_mul flat-grid index"
        );
        no_overlap(&self.x, &self.contributions, "x", "contributions")?;
        no_overlap(&self.ids, &self.contributions, "ids", "contributions")?;
        no_overlap(
            &self.weights,
            &self.contributions,
            "weights",
            "contributions",
        )?;
        no_overlap(
            &self.shared_gate,
            &self.contributions,
            "shared_gate",
            "contributions",
        )?;
        no_overlap(
            &self.shared_up,
            &self.contributions,
            "shared_up",
            "contributions",
        )?;
        no_overlap(
            &self.shared_down,
            &self.contributions,
            "shared_down",
            "contributions",
        )?;
        no_overlap(
            &self.shared_expert_gate,
            &self.contributions,
            "shared_expert_gate",
            "contributions",
        )?;
        no_overlap(&self.ids, &self.done.read(), "ids", "done")?;
        no_overlap(&self.weights, &self.done.read(), "weights", "done")?;
        no_overlap(&self.active, &self.done.read(), "active", "done")?;
        no_overlap(
            &self.assignment_gen,
            &self.done.read(),
            "assignment_gen",
            "done",
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
            assignments,
            layout,
        })
    }
}

#[cfg(test)]
impl MoePrefill<'_> {
    /// The wrapper-only checks: `y` density and its separation from every
    /// input; every cohort-level check runs in
    /// `MoePrefillContribute::validate`.
    fn validate_y(&self, plan: &PrefillPlan) -> Result<()> {
        let (rows, hidden) = (plan.rows, plan.hidden);
        self.y.read().validate()?;
        ensure!(
            self.y.read().shape == [rows, hidden] && self.y.read().stride == [u64::from(hidden), 1],
            "moe prefill y must be a dense [rows, hidden] view (shape {:?}, hidden {hidden})",
            self.y.read().shape
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
        Ok(())
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
/// `pos_of`, `y_perm` and `shared_out` are read back by the gate's scatter
/// and join cells.
#[derive(Debug)]
pub(crate) struct PrefillLayout {
    counts: u64,
    active_total: u64,
    offsets: u64,
    cursor: u64,
    pub(crate) pos_of: u64,
    #[cfg(test)]
    done: u64,
    x_perm: u64,
    h1: u64,
    act: u64,
    pub(crate) y_perm: u64,
    #[cfg(test)]
    contributions: u64,
    h1s: u64,
    act_s: u64,
    pub(crate) shared_out: u64,
    scale: u64,
    bytes: u64,
}

/// Persistent pinned host bytes reserved by PrefillOps::new: one D2H
/// window plus the two GEMM pointer mirrors.
#[cfg(test)]
pub fn stage_host_pinned_bytes() -> u64 {
    (D2H_WINDOW_WORDS * 4 + (GEMM1_PTR_WORDS + GEMM2_PTR_WORDS) * 8) as u64
}

/// Persistent device bytes reserved by PrefillOps::new: the two GEMM
/// pointer blocks.
#[cfg(test)]
pub fn stage_device_bytes() -> u64 {
    ((GEMM1_PTR_WORDS + GEMM2_PTR_WORDS) * 8) as u64
}

pub(crate) fn prefill_scratch_bytes(
    rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<u64> {
    prefill_layout(rows, top_k, experts, hidden, inter).map(|layout| layout.bytes)
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

pub(crate) fn prefill_layout(
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
    let pos_of = region(&mut at, assignments, 4)?;
    #[cfg(test)]
    let done = at;
    region(&mut at, assignments, 4)?;
    let x_perm = region(&mut at, assignments * hidden, 2)?;
    let h1 = region(&mut at, assignments * 2 * inter, 2)?;
    let act = region(&mut at, assignments * inter, 2)?;
    let y_perm = region(&mut at, assignments * hidden, 2)?;
    #[cfg(test)]
    let contributions = at;
    region(&mut at, assignments * hidden, 2)?;
    let h1s = region(&mut at, rows * 2 * inter, 2)?;
    let act_s = region(&mut at, rows * inter, 2)?;
    let shared_out = region(&mut at, rows * hidden, 2)?;
    let scale = region(&mut at, rows, 2)?;
    Ok(PrefillLayout {
        counts,
        active_total,
        offsets,
        cursor,
        pos_of,
        #[cfg(test)]
        done,
        x_perm,
        h1,
        act,
        y_perm,
        #[cfg(test)]
        contributions,
        h1s,
        act_s,
        shared_out,
        scale,
        bytes: at,
    })
}

/// Word offset of the active-assignment total inside the D2H window: the
/// counts region is padded to the 256-byte scratch alignment.
pub(crate) fn d2h_active_total_word(experts: usize) -> usize {
    (REGION as usize / 4) * ((experts * 4).div_ceil(REGION as usize))
}

/// i32 words of the D2H window for an experts count: counts[E], the
/// region-padded active total and offsets[E+1]. The last word is the scan
/// total offsets[E].
pub(crate) fn d2h_window_words(experts: usize) -> usize {
    d2h_active_total_word(experts) + REGION as usize / 4 + experts + 1
}

/// Active experts (count > 0) in problem order: count ascending, id ascending.
pub(crate) fn problem_order(counts: &[i32], order: &mut Vec<u32>) {
    order.clear();
    order.extend((0..counts.len() as u32).filter(|&e| counts[e as usize] > 0));
    order.sort_by_key(|&e| counts[e as usize]);
}

/// Consecutive same-count runs of the problem order as (count, run length).
pub(crate) fn group_runs(order: &[u32], counts: &[i32], runs: &mut Vec<(u32, u32)>) {
    runs.clear();
    for &e in order {
        let count = counts[e as usize] as u32;
        match runs.last_mut() {
            Some(last) if last.0 == count => last.1 += 1,
            _ => runs.push((count, 1)),
        }
    }
}

/// One grouped-GEMM group's problem shape.
struct GemmShape {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    size: u32,
}

/// Per-group host arrays of one grouped-GEMM call.
struct GemmArrays {
    m: Vec<i32>,
    n: Vec<i32>,
    k: Vec<i32>,
    lda: Vec<i32>,
    ldb: Vec<i32>,
    ldc: Vec<i32>,
    size: Vec<i32>,
}

impl GemmArrays {
    fn push(&mut self, g: GemmShape) {
        let GemmShape {
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            size,
        } = g;
        self.m.push(m as i32);
        self.n.push(n as i32);
        self.k.push(k as i32);
        self.lda.push(lda as i32);
        self.ldb.push(ldb as i32);
        self.ldc.push(ldc as i32);
        self.size.push(size as i32);
    }

    fn clear(&mut self) {
        self.m.clear();
        self.n.clear();
        self.k.clear();
        self.lda.clear();
        self.ldb.clear();
        self.ldc.clear();
        self.size.clear();
    }
}

/// Host staging reused across prefill calls: the pinned D2H window, the
/// pinned [A|B|C] pointer mirrors with their device blocks, and the host
/// vectors the problem sizes and pointers are rebuilt into. Every capacity
/// is bounded by `MAX_EXPERTS`; `MoePrefill::validate` rejects larger expert
/// counts by name and the copy helpers bound-check each transfer.
struct Stage {
    d2h: PinnedHostSlice<i32>,
    gemm1_host: PinnedHostSlice<u64>,
    gemm1_dev: CudaSlice<u64>,
    gemm2_host: PinnedHostSlice<u64>,
    gemm2_dev: CudaSlice<u64>,
    gemm1_copy_done: CudaEvent,
    gemm2_copy_done: CudaEvent,
    order: Vec<u32>,
    runs: Vec<(u32, u32)>,
    offsets: Vec<i32>,
    arrays: GemmArrays,
    transa: Vec<bs::cublasOperation_t>,
    transb: Vec<bs::cublasOperation_t>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
}

impl Stage {
    fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        // SAFETY: the stage owns its pinned mirrors and initializes every transmitted word before upload.
        let stage = unsafe {
            Stage {
                d2h: ctx.alloc_pinned_with_flags::<i32>(D2H_WINDOW_WORDS, 0)?,
                gemm1_host: ctx.alloc_pinned_with_flags::<u64>(GEMM1_PTR_WORDS, 0)?,
                gemm1_dev: stream.alloc_zeros::<u64>(GEMM1_PTR_WORDS)?,
                gemm2_host: ctx.alloc_pinned_with_flags::<u64>(GEMM2_PTR_WORDS, 0)?,
                gemm2_dev: stream.alloc_zeros::<u64>(GEMM2_PTR_WORDS)?,
                gemm1_copy_done: ctx.new_event(Some(ds::CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
                gemm2_copy_done: ctx.new_event(Some(ds::CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
                order: Vec::with_capacity(MAX_EXPERTS),
                runs: Vec::with_capacity(MAX_EXPERTS),
                offsets: Vec::with_capacity(MAX_EXPERTS),
                arrays: GemmArrays {
                    m: Vec::with_capacity(MAX_GROUPS),
                    n: Vec::with_capacity(MAX_GROUPS),
                    k: Vec::with_capacity(MAX_GROUPS),
                    lda: Vec::with_capacity(MAX_GROUPS),
                    ldb: Vec::with_capacity(MAX_GROUPS),
                    ldc: Vec::with_capacity(MAX_GROUPS),
                    size: Vec::with_capacity(MAX_GROUPS),
                },
                transa: vec![bs::cublasOperation_t::CUBLAS_OP_T; MAX_GROUPS],
                transb: vec![bs::cublasOperation_t::CUBLAS_OP_N; MAX_GROUPS],
                alpha: vec![1f32; MAX_GROUPS],
                beta: vec![0f32; MAX_GROUPS],
            }
        };
        Ok(stage)
    }
}

pub(crate) struct PrefillOps {
    count: CudaFunction,
    scan: CudaFunction,
    permute: CudaFunction,
    scatter: CudaFunction,
    silu: CudaFunction,
    combine: CombineOps,
    blas: CudaBlas,
    stream: Arc<CudaStream>,
    stage: Mutex<Stage>,
}

impl PrefillOps {
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_prefill")?;
        let act = cubin::module(ctx, "act")?;
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
            scatter: module.load_function("moe_scatter")?,
            silu: act.load_function("silu_mul")?,
            combine: CombineOps::new(ctx)?,
            blas,
            stream: stream.clone(),
            stage: Mutex::new(Stage::new(ctx, stream)?),
        })
    }

    /// Contribute one cohort: count→scan→permute→GEMM1→silu→GEMM2→scatter
    /// over THIS cohort's active assignments into the caller-owned
    /// contributions/done buffers. No zero/reset of either buffer, no shared
    /// FFN, no sigmoid gate, no combine; a cohort with no active assignments
    /// runs no routed problem. Every input tensor and both slot tables are
    /// immutable until this stream completes.
    pub fn prefill_contribute(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillContribute<'_>,
        scratch: &Scratch,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe prefill stream is not the PrefillOps construction stream (cuBLAS handle bound at construction)"
        );
        let plan = input.validate()?;
        scratch.validate(plan.layout.bytes)?;
        self.contribute(stream, &input, &plan, scratch)?;
        Ok(())
    }

    fn contribute(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoePrefillContribute<'_>,
        plan: &PrefillPlan,
        scratch: &Scratch,
    ) -> Result<()> {
        let experts = plan.experts as usize;
        let (rows, top_k, hidden, inter) = (plan.rows, plan.top_k, plan.hidden, plan.inter);
        let layout = &plan.layout;
        let counts_at = scratch.ptr + layout.counts;
        let active_total_at = scratch.ptr + layout.active_total;
        let offsets_at = scratch.ptr + layout.offsets;
        let cursor_at = scratch.ptr + layout.cursor;
        let pos_of_at = scratch.ptr + layout.pos_of;
        let x_perm_at = scratch.ptr + layout.x_perm;
        let h1_at = scratch.ptr + layout.h1;
        let act_at = scratch.ptr + layout.act;
        let y_perm_at = scratch.ptr + layout.y_perm;

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
        unsafe {
            stream
                .launch_builder(&self.scan)
                .arg(&counts_at)
                .arg(&offsets_at)
                .arg(&cursor_at)
                .arg(&(experts as i32))
                .launch(grid(1, SCAN_THREADS))?;
        }
        unsafe {
            stream
                .launch_builder(&self.permute)
                .arg(&input.ids.ptr)
                .arg(&input.x.ptr)
                .arg(&offsets_at)
                .arg(&cursor_at)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&input.assignment_gen.ptr)
                .arg(&input.active.ptr)
                .arg(&input.done.0.ptr)
                .arg(&pos_of_at)
                .arg(&x_perm_at)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(experts as i32))
                .arg(&(hidden as i64))
                .launch(grid(plan.assignments as usize, BLOCK_THREADS))?;
        }

        // Count sync: the grouped-GEMM m/n/k/ld arrays are host-side in
        // cuBLAS, so one pinned D2H moves counts, the active total and the
        // scan total to the host and the stream synchronizes once per call.
        let mut guard = self
            .stage
            .lock()
            .map_err(|_| anyhow::anyhow!("moe prefill stage mutex is poisoned"))?;
        let stage: &mut Stage = &mut guard;
        let d2h_words = d2h_window_words(experts);
        unsafe { dtoh_window(stream, counts_at, &mut stage.d2h, d2h_words)? };
        stream.synchronize()?;
        let window = stage.d2h.as_slice()?;
        let active_total_word = d2h_active_total_word(experts);
        let counts = &window[..experts];
        let active_total = window[active_total_word] as u64;
        ensure!(
            window[d2h_words - 1] as u64 == active_total,
            "moe prefill scan total {} != {active_total} active assignments",
            window[d2h_words - 1]
        );
        if active_total == 0 {
            return Ok(());
        }
        problem_order(counts, &mut stage.order);
        group_runs(&stage.order, counts, &mut stage.runs);
        stage.offsets.clear();
        let mut acc = 0i32;
        for &count in counts {
            stage.offsets.push(acc);
            acc += count;
        }
        let (gu_align, dn_align) = if hidden.is_multiple_of(8) && inter.is_multiple_of(8) {
            (ALIGN_BYTES, ALIGN_BYTES)
        } else {
            (ALIGN_MIN_BYTES, ALIGN_MIN_BYTES)
        };

        // GEMM1: one problem per active expert, B/C at the expert's segment
        // offsets.
        let len1 = stage.order.len();
        {
            let mirror = stage.gemm1_host.as_mut_slice()?;
            for (j, &e) in stage.order.iter().enumerate() {
                let e = e as usize;
                let gu = input.slots_gate_up_host[e];
                let dn = input.slots_down_host[e];
                ensure!(
                    gu != 0,
                    "moe prefill gate_up slot pointer of expert {e} is null"
                );
                ensure!(
                    dn != 0,
                    "moe prefill down slot pointer of expert {e} is null"
                );
                ensure!(
                    gu.is_multiple_of(gu_align),
                    "moe prefill gate_up slot pointer of expert {e} (0x{gu:x}) below cuBLAS alignment {gu_align}"
                );
                ensure!(
                    dn.is_multiple_of(dn_align),
                    "moe prefill down slot pointer of expert {e} (0x{dn:x}) below cuBLAS alignment {dn_align}"
                );
                let off = stage.offsets[e] as u64;
                mirror[j] = gu;
                mirror[len1 + j] = x_perm_at + elem(off * u64::from(hidden));
                mirror[2 * len1 + j] = h1_at + elem(off * u64::from(2 * inter));
            }
        }
        stage.arrays.clear();
        for &(count, size) in &stage.runs {
            stage.arrays.push(GemmShape {
                m: 2 * inter,
                n: count,
                k: hidden,
                lda: hidden,
                ldb: hidden,
                ldc: 2 * inter,
                size,
            });
        }
        let (gemm1_base, _g1) = stage.gemm1_dev.device_ptr(stream);
        unsafe {
            htod_ptrs(
                stream,
                gemm1_base,
                &stage.gemm1_host,
                3 * len1,
                &mut stage.gemm1_copy_done,
            )?;
            self.grouped_gemm(stage, gemm1_base, len1)
                .context("moe prefill grouped GEMM 1")?;
        }

        let silu_grid = |elems: usize| grid(elems.div_ceil(BLOCK_THREADS as usize), BLOCK_THREADS);
        unsafe {
            stream
                .launch_builder(&self.silu)
                .arg(&h1_at)
                .arg(&act_at)
                .arg(&(active_total as i32))
                .arg(&(inter as i32))
                .launch(silu_grid(active_total as usize * inter as usize))?;
        }

        // GEMM2: one problem per active expert over the expert's whole
        // segment, C rebuilt from the offsets into the contiguous Y_perm.
        let len2 = stage.order.len();
        {
            let mirror = stage.gemm2_host.as_mut_slice()?;
            for (j, &e) in stage.order.iter().enumerate() {
                let e = e as usize;
                let off = stage.offsets[e] as u64;
                mirror[j] = input.slots_down_host[e];
                mirror[len2 + j] = act_at + elem(off * u64::from(inter));
                mirror[2 * len2 + j] = y_perm_at + elem(off * u64::from(hidden));
            }
        }
        stage.arrays.clear();
        for &(count, size) in &stage.runs {
            stage.arrays.push(GemmShape {
                m: hidden,
                n: count,
                k: inter,
                lda: inter,
                ldb: inter,
                ldc: hidden,
                size,
            });
        }
        let (gemm2_base, _g2) = stage.gemm2_dev.device_ptr(stream);
        unsafe {
            htod_ptrs(
                stream,
                gemm2_base,
                &stage.gemm2_host,
                3 * len2,
                &mut stage.gemm2_copy_done,
            )?;
            self.grouped_gemm(stage, gemm2_base, len2)
                .context("moe prefill grouped GEMM 2")?;
        }
        unsafe {
            stream
                .launch_builder(&self.scatter)
                .arg(&y_perm_at)
                .arg(&input.weights.ptr)
                .arg(&pos_of_at)
                .arg(&input.active.ptr)
                .arg(&input.contributions.ptr)
                .arg(&(plan.assignments as i64))
                .arg(&(hidden as i64))
                .launch(grid(plan.assignments as usize, BLOCK_THREADS))?;
        }
        Ok(())
    }

    /// Contribute + combine over the active assignments and the shared
    /// expert: the all-active cohort into scratch-backed buffers, then the
    /// shared FFN, the sigmoid gate and the combine. The host slot tables
    /// and every input tensor are immutable until this stream completes.
    #[cfg(test)]
    pub fn prefill(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefill<'_>,
        scratch: &Scratch,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe prefill stream is not the PrefillOps construction stream (cuBLAS handle bound at construction)"
        );
        let (rows, top_k, hidden) = (input.x.shape[0], input.ids.shape[1], input.x.shape[1]);
        let layout = prefill_layout(
            rows as usize,
            top_k as usize,
            input.slot_valid.shape[0] as usize,
            hidden as usize,
            input.shared_gate.shape[0] as usize,
        )?;
        let assignments = u64::from(rows) * u64::from(top_k);
        let cohort = MoePrefillContribute {
            contributions: unsafe {
                Tensor::<bf16, 3>::new(
                    scratch.ptr + layout.contributions,
                    [rows, top_k, hidden],
                    [u64::from(top_k) * u64::from(hidden), u64::from(hidden), 1],
                    assignments * u64::from(hidden) * 2,
                )?
            },
            done: unsafe {
                TensorMut::<u32, 2>::new(
                    scratch.ptr + layout.done,
                    [rows, top_k],
                    [u64::from(top_k), 1],
                    assignments * 4,
                )?
            },
            x: input.x,
            ids: input.ids,
            weights: input.weights,
            active: input.active,
            assignment_gen: input.assignment_gen,
            slots_gate_up: input.slots_gate_up,
            slots_down: input.slots_down,
            slot_valid: input.slot_valid,
            slot_generation: input.slot_generation,
            slots_gate_up_host: input.slots_gate_up_host,
            slots_down_host: input.slots_down_host,
            shared_gate: input.shared_gate,
            shared_up: input.shared_up,
            shared_down: input.shared_down,
            shared_expert_gate: input.shared_expert_gate,
        };
        let plan = cohort.validate()?;
        input.validate_y(&plan)?;
        scratch.validate(plan.layout.bytes)?;
        unsafe {
            zero_words(
                stream,
                scratch.ptr + plan.layout.done,
                plan.assignments as usize,
            )?;
        }
        self.contribute(stream, &cohort, &plan, scratch)?;
        let y = input.y;
        let fin = MoePrefillFinish {
            x: cohort.x,
            ids: cohort.ids,
            final_active: input.active,
            contributions: cohort.contributions,
            done: cohort.done.read(),
            experts: plan.experts,
            shared_gate: cohort.shared_gate,
            shared_up: cohort.shared_up,
            shared_down: cohort.shared_down,
            shared_expert_gate: cohort.shared_expert_gate,
            y,
        };
        self.finish_body(stream, fin, &plan, scratch)?;
        Ok(())
    }

    /// The join over caller-owned contributions: shared FFN, sigmoid gate,
    /// one combine. `done` is read, never written; the combine traps unless
    /// every final-active choice carries exactly one done bit. The caller
    /// orders buffer reuse after this stream completes.
    pub fn prefill_finish(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillFinish,
        scratch: &Scratch,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe prefill stream is not the PrefillOps construction stream (cuBLAS handle bound at construction)"
        );
        let plan = input.validate()?;
        scratch.validate(plan.layout.bytes)?;
        self.finish_body(stream, input, &plan, scratch)?;
        Ok(())
    }

    fn finish_body(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillFinish,
        plan: &PrefillPlan,
        scratch: &Scratch,
    ) -> Result<()> {
        // Shared FFN: the gate/up group (2 problems) and the down group (1
        // problem) share one 9-pointer block [A|B|C] per group, uploaded
        // with a single H2D.
        let (rows, hidden, inter) = (plan.rows, plan.hidden, plan.inter);
        let h1s_at = scratch.ptr + plan.layout.h1s;
        let act_s_at = scratch.ptr + plan.layout.act_s;
        let shared_out_at = scratch.ptr + plan.layout.shared_out;
        {
            let mut guard = self
                .stage
                .lock()
                .map_err(|_| anyhow::anyhow!("moe prefill stage mutex is poisoned"))?;
            let stage: &mut Stage = &mut guard;
            {
                let mirror = stage.gemm1_host.as_mut_slice()?;
                mirror[0] = input.shared_gate.ptr;
                mirror[1] = input.shared_up.ptr;
                mirror[2] = input.x.ptr;
                mirror[3] = input.x.ptr;
                mirror[4] = h1s_at;
                mirror[5] = h1s_at + elem(u64::from(inter));
                mirror[6] = input.shared_down.ptr;
                mirror[7] = act_s_at;
                mirror[8] = shared_out_at;
            }
            stage.arrays.clear();
            stage.arrays.push(GemmShape {
                m: inter,
                n: rows,
                k: hidden,
                lda: hidden,
                ldb: hidden,
                ldc: 2 * inter,
                size: 2,
            });
            let (base, _gs) = stage.gemm1_dev.device_ptr(stream);
            unsafe {
                htod_ptrs(
                    stream,
                    base,
                    &stage.gemm1_host,
                    9,
                    &mut stage.gemm1_copy_done,
                )?;
                self.grouped_gemm(stage, base, 2)
                    .context("moe prefill shared gate/up GEMM")?;
            }
            unsafe {
                stream
                    .launch_builder(&self.silu)
                    .arg(&h1s_at)
                    .arg(&act_s_at)
                    .arg(&(rows as i32))
                    .arg(&(inter as i32))
                    .launch(grid(
                        (rows as usize * inter as usize).div_ceil(BLOCK_THREADS as usize),
                        BLOCK_THREADS,
                    ))?;
            }
            stage.arrays.clear();
            stage.arrays.push(GemmShape {
                m: hidden,
                n: rows,
                k: inter,
                lda: inter,
                ldb: inter,
                ldc: hidden,
                size: 1,
            });
            unsafe {
                self.grouped_gemm(stage, base + 6 * 8, 1)
                    .context("moe prefill shared down GEMM")?;
            }
        }

        // moe_combine's scale column over the same [rows] words; the nonzero
        // stride keeps Tensor::validate satisfied.
        let scale_at = scratch.ptr + plan.layout.scale;
        let mut scale =
            unsafe { TensorMut::<bf16, 2>::new(scale_at, [rows, 1], [1, 1], u64::from(rows) * 2)? };
        self.combine
            .shared_scale(stream, input.x, input.shared_expert_gate, &mut scale)?;
        let combine_input = MoeCombine {
            ids: input.ids,
            final_active: input.final_active,
            done: input.done,
            contributions: input.contributions,
            shared_out: unsafe {
                Tensor::<bf16, 2>::new(
                    shared_out_at,
                    [rows, hidden],
                    [u64::from(hidden), 1],
                    u64::from(rows) * u64::from(hidden) * 2,
                )?
            },
            x: input.x,
            shared_expert_gate: input.shared_expert_gate,
            scale,
            y: input.y,
        };
        self.combine.combine(stream, &combine_input)?;
        Ok(())
    }

    /// cublasGemmGroupedBatchedEx with BF16 A/B/C, CUBLAS_COMPUTE_32F,
    /// alpha=1, beta=0, opA=T, opB=N for every group. `block` is the device
    /// address of the pointer block holding `problems` A pointers, then
    /// `problems` B pointers, then `problems` C pointers; the per-group
    /// m/n/k/ld/size arrays come from the stage.
    ///
    /// # Safety
    /// `block` must address 3*`problems` device pointers whose targets match
    /// the group shapes and are kept alive until the stream completes.
    unsafe fn grouped_gemm(&self, stage: &Stage, block: u64, problems: usize) -> Result<()> {
        let arrays = &stage.arrays;
        let n = arrays.m.len();
        unsafe {
            bs::cublasGemmGroupedBatchedEx(
                *self.blas.handle(),
                stage.transa.as_ptr(),
                stage.transb.as_ptr(),
                arrays.m.as_ptr(),
                arrays.n.as_ptr(),
                arrays.k.as_ptr(),
                stage.alpha.as_ptr().cast(),
                block as *const *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                arrays.lda.as_ptr(),
                (block + (problems as u64) * 8) as *const *const core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                arrays.ldb.as_ptr(),
                stage.beta.as_ptr().cast(),
                (block + 2 * (problems as u64) * 8) as *mut *mut core::ffi::c_void,
                bs::cudaDataType_t::CUDA_R_16BF,
                arrays.ldc.as_ptr(),
                n as i32,
                arrays.size.as_ptr(),
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
/// `dst` must address `words` writable device words that no in-flight work
/// on the stream touches.
#[cfg(test)]
unsafe fn zero_words(stream: &Arc<CudaStream>, dst: u64, words: usize) -> Result<()> {
    stream.context().bind_to_thread()?;
    unsafe {
        ds::cuMemsetD32Async(dst, 0, words, stream.cu_stream()).result()?;
    }
    Ok(())
}

/// # Safety
/// `src` must address `words` i32 device words ordered before this call on
/// the stream.
unsafe fn dtoh_window(
    stream: &Arc<CudaStream>,
    src: u64,
    window: &mut PinnedHostSlice<i32>,
    words: usize,
) -> Result<()> {
    ensure!(
        words <= window.len(),
        "moe prefill D2H window {words} words exceeds the stage capacity {}",
        window.len()
    );
    stream.context().bind_to_thread()?;
    let dst = window.as_mut_ptr()?;
    unsafe {
        ds::cuMemcpyDtoHAsync_v2(
            dst.cast(),
            src,
            words * std::mem::size_of::<i32>(),
            stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// # Safety
/// `dst` must address `words` u64 words of writable device memory that no
/// in-flight work on the stream touches.
unsafe fn htod_ptrs(
    stream: &Arc<CudaStream>,
    dst: u64,
    src: &PinnedHostSlice<u64>,
    words: usize,
    done: &mut CudaEvent,
) -> Result<()> {
    ensure!(
        words <= src.len(),
        "moe prefill pointer mirror {words} words exceeds the stage capacity {}",
        src.len()
    );
    stream.context().bind_to_thread()?;
    // The pinned mirror is rewritten on the next call; the previous copy
    // must have completed first.
    stream.wait(done)?;
    let host = src.as_ptr()?;
    unsafe {
        ds::cuMemcpyHtoDAsync_v2(
            dst,
            host.cast(),
            words * std::mem::size_of::<u64>(),
            stream.cu_stream(),
        )
        .result()?;
    }
    done.record(stream)?;
    Ok(())
}
