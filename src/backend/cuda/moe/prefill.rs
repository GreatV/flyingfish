//! Cohorts write caller-owned weighted BF16 contributions and done bits.
//! Grouped GEMMs use opA=T/opB=N over native [2I, H] and [H, I] expert weights.
//! Count/scan/permute stage one host count read and two pointer-table uploads per cohort.
//! Finish runs the shared FFN and one combine after every contribution is complete.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut, no_overlap};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
#[cfg(feature = "rocm")]
use crate::backend::gpu::blas::Blas as CudaBlas;
use crate::backend::gpu::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice,
    PushKernelArg,
};
use crate::experts::catalogue::{ExpertGeometry, ExpertLayout, Fp8Slot};
use crate::model_spec::ExpertFormat;
use anyhow::{Context, Result, ensure};
#[cfg(feature = "cuda")]
use cudarc::{
    cublas::{CudaBlas, sys as bs},
    driver::sys as ds,
};
use half::bf16;
use std::sync::{Arc, Mutex};

use super::combine::{
    CombineOps, MoeCombine, MoeCombineResidual, MoeCombineRouted, MoeJoinRouted,
    MoeJoinRoutedResidual,
};

const BLOCK_THREADS: u32 = 256;
/// One 256-thread tile contains sixteen independent sixteen-lane dots.
pub(crate) const LANE16_ROWS: u32 = BLOCK_THREADS / 16;
/// Native W8A16 grouped GEMM tile extent along packed rows.
pub(crate) const W8A16_TILE_ROWS: u32 = 64;
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
/// A/B/C pointers and row counts, one problem per active expert plus shared entries.
pub(crate) const GEMM1_PTR_WORDS: usize = 4 * (MAX_EXPERTS + 2);
pub(crate) const GEMM2_PTR_WORDS: usize = 4 * (MAX_EXPERTS + 1);
/// Upper bound of per-GEMM groups: one per same-count run plus the shared
/// group.
const MAX_GROUPS: usize = MAX_EXPERTS + 2;

/// Caller-owned contribution/done buffers survive cohorts; inputs stay immutable until stream completion.
#[derive(Debug)]
pub(crate) struct MoePrefillContribute<'a> {
    pub mask_from_slots: bool,
    pub expert_inter: u32,
    /// Host-packed `(cohort, base, active)` wave expectation checked at the count sync.
    pub wave: Option<(usize, u32, u32)>,
    pub wait_copy: Option<&'a CudaEvent>,
    pub reader_event: Option<&'a CudaEvent>,
    /// Whole-span counts at most LANE16_ROWS use the CPU arithmetic form in every wave.
    pub cpu_arithmetic: &'a [bool],
    /// Native FP8 tables carry the six-slice slot base in both pointer arrays.
    pub fp8_slot: Option<&'a Fp8Slot>,
    pub x: Tensor<bf16, 2>,
    pub ids: Tensor<i32, 2>,
    pub weights: Tensor<bf16, 2>,
    /// `[rows, top_k]` THIS cohort's assignment mask.
    pub active: Tensor<u32, 2>,
    pub assignment_gen: Option<Tensor<u64, 2>>,
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
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    /// CALLER-OWNED `[rows, top_k, hidden]` weighted BF16 contributions.
    pub contributions: Tensor<bf16, 3>,
    /// CALLER-OWNED `[rows, top_k]` done bits (zeroed by the first cohort).
    pub done: TensorMut<u32, 2>,
}

/// The wrapper input: the cohort fields plus the output `y`.
#[derive(Debug)]
#[cfg(test)]
pub(crate) struct MoePrefill<'a> {
    pub expert_inter: u32,
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
    /// Immutable until the stream completes.
    pub slots_gate_up_host: &'a [u64],
    /// `[experts]` device addresses of each expert's down blocks.
    /// Immutable until the stream completes.
    pub slots_down_host: &'a [u64],
    pub shared_gate: Tensor<bf16, 2>,
    pub shared_up: Tensor<bf16, 2>,
    pub shared_down: Tensor<bf16, 2>,
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    pub contributions: Tensor<bf16, 3>,
    pub done: TensorMut<u32, 2>,
    pub y: TensorMut<bf16, 2>,
}

/// The join input: caller-owned contributions and done bits accumulated by
/// any mix of contribute cohorts (GPU, streamed, CPU) plus the final-active
/// union; `prefill_finish` runs the shared FFN, the sigmoid gate and the
/// single combine. No routed arithmetic runs here and `done` is never
/// written.
#[derive(Debug)]
pub(crate) struct MoePrefillFinish {
    pub expert_inter: u32,
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
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    pub y: TensorMut<bf16, 2>,
}

#[derive(Debug)]
pub(crate) struct PrefillPlan {
    rows: u32,
    top_k: u32,
    experts: u32,
    hidden: u32,
    inter: u32,
    shared_inter: u32,
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
        let shared_inter = self.shared_gate.shape[0];
        for (t, name) in [
            (&self.shared_gate, "shared_gate"),
            (&self.shared_up, "shared_up"),
        ] {
            t.validate()?;
            ensure!(
                t.shape == [shared_inter, hidden] && t.stride == [u64::from(hidden), 1],
                "moe prefill finish {name} must be dense [shared_inter, hidden] (shape {:?})",
                t.shape
            );
        }
        self.shared_down.validate()?;
        ensure!(
            self.shared_down.shape == [hidden, shared_inter]
                && self.shared_down.stride == [u64::from(shared_inter), 1],
            "moe prefill finish shared_down must be dense [hidden, shared_inter] (shape {:?})",
            self.shared_down.shape
        );
        if let Some(gate) = &self.shared_expert_gate {
            gate.validate()?;
            ensure!(
                gate.shape == [hidden] && gate.stride == [1],
                "moe prefill finish shared gate must be dense [hidden]"
            );
        }
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
        if let Some(gate) = &self.shared_expert_gate {
            no_overlap(gate, &self.y.read(), "shared_expert_gate", "y")?;
        }
        ensure!(
            (1..=MAX_EXPERTS).contains(&(self.experts as usize)),
            "moe prefill finish experts {} outside [1, {}]",
            self.experts,
            MAX_EXPERTS
        );
        let layout = prefill_layout_mixed(
            rows as usize,
            rows as usize,
            top_k as usize,
            self.experts as usize,
            hidden as usize,
            (self.expert_inter as usize, shared_inter as usize),
            (None, crate::model_spec::RoutedWeightBoundary::AfterDownBf16),
        )?;
        Ok(PrefillPlan {
            rows,
            top_k,
            experts: self.experts,
            hidden,
            inter: shared_inter,
            shared_inter,
            assignments: u64::from(rows) * u64::from(top_k),
            layout,
        })
    }
}

impl MoePrefillContribute<'_> {
    pub(crate) fn validate(
        &self,
        routed_weight: crate::model_spec::RoutedWeightBoundary,
    ) -> Result<PrefillPlan> {
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
        ensure!(
            self.mask_from_slots == self.assignment_gen.is_none(),
            "moe prefill assignment generation must be present exactly for explicit masks"
        );
        if let Some(generation) = &self.assignment_gen {
            generation.validate()?;
        }
        let top_k = self.ids.shape[1];
        ensure!(
            self.ids.shape == [rows, top_k]
                && self.ids.stride == [u64::from(top_k), 1]
                && self.active.shape == [rows, top_k]
                && self.active.stride == [u64::from(top_k), 1]
                && self.weights.shape == [rows, top_k]
                && self.weights.stride == [u64::from(top_k), 1]
                && self
                    .assignment_gen
                    .as_ref()
                    .is_none_or(|generation| generation.shape == [rows, top_k]
                        && generation.stride == [u64::from(top_k), 1]),
            "moe prefill ids/active/weights/assignment_gen must be dense [rows, top_k] views (top_k {top_k})"
        );
        let experts = self.slot_valid.shape[0];
        ensure!(
            self.cpu_arithmetic.len() == experts as usize,
            "moe prefill CPU arithmetic mask length={} experts={experts}",
            self.cpu_arithmetic.len()
        );
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
        let shared_inter = self.shared_gate.shape[0];
        let inter = self.expert_inter;
        self.shared_gate.validate()?;
        self.shared_up.validate()?;
        self.shared_down.validate()?;
        if let Some(gate) = &self.shared_expert_gate {
            gate.validate()?;
        }
        ensure!(
            self.shared_gate.shape == [shared_inter, hidden]
                && self.shared_gate.stride == [u64::from(hidden), 1]
                && self.shared_up.shape == [shared_inter, hidden]
                && self.shared_up.stride == [u64::from(hidden), 1]
                && self.shared_down.shape == [hidden, shared_inter]
                && self.shared_down.stride == [u64::from(shared_inter), 1]
                && self
                    .shared_expert_gate
                    .as_ref()
                    .is_none_or(|gate| gate.shape == [hidden] && gate.stride == [1]),
            "moe prefill shared expert layout mismatch (inter {shared_inter}, hidden {hidden})"
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
            hidden.is_multiple_of(2) && inter.is_multiple_of(2) && shared_inter.is_multiple_of(2),
            "moe prefill hidden {hidden}, inter {inter}, and shared inter {shared_inter} must be even for cuBLAS BF16 alignment"
        );
        ensure!(
            u64::from(hidden) <= i32::MAX as u64
                && u64::from(inter) <= i32::MAX as u64
                && u64::from(shared_inter) <= i32::MAX as u64,
            "moe prefill hidden {hidden}, inter {inter}, or shared inter {shared_inter} exceeds the cuBLAS grouped i32 leading dims"
        );
        let assignments = u64::from(rows) * u64::from(top_k);
        ensure!(
            assignments <= i32::MAX as u64,
            "moe prefill assignment count {assignments} exceeds i32"
        );
        let silu_routed = assignments * u64::from(inter);
        let silu_shared = u64::from(rows) * u64::from(shared_inter);
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
        if let Some(gate) = &self.shared_expert_gate {
            no_overlap(
                gate,
                &self.contributions,
                "shared_expert_gate",
                "contributions",
            )?;
        }
        no_overlap(&self.ids, &self.done.read(), "ids", "done")?;
        no_overlap(&self.weights, &self.done.read(), "weights", "done")?;
        no_overlap(&self.active, &self.done.read(), "active", "done")?;
        if let Some(generation) = &self.assignment_gen {
            no_overlap(generation, &self.done.read(), "assignment_gen", "done")?;
        }
        let active_rows = match self.wave {
            Some((cohort, base, expected_active)) => {
                ensure!(
                    u64::from(expected_active) <= assignments,
                    "moe prefill cohort={cohort} base={base} rows={rows} host-packed active={expected_active} exceeds wave assignments={assignments}"
                );
                (expected_active as usize).div_ceil(top_k as usize)
            }
            None => rows as usize,
        };
        let layout = prefill_layout_mixed(
            active_rows,
            rows as usize,
            top_k as usize,
            experts as usize,
            hidden as usize,
            (inter as usize, shared_inter as usize),
            (self.fp8_slot, routed_weight),
        )?;
        Ok(PrefillPlan {
            rows,
            top_k,
            experts,
            hidden,
            inter,
            shared_inter,
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
        if let Some(gate) = &self.shared_expert_gate {
            no_overlap(gate, &self.y.read(), "shared_expert_gate", "y")?;
        }
        Ok(())
    }
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
    pub(crate) compact_active: u64,
    compact_capacity: usize,
    x_perm: u64,
    pub(crate) h1: u64,
    pub(crate) act: u64,
    pub(crate) y_perm: u64,
    h1s: u64,
    act_s: u64,
    pub(crate) shared_out: u64,
    scale: u64,
    bytes: u64,
}

/// Persistent pinned host bytes reserved by PrefillOps::new: one D2H
/// window plus the two GEMM pointer mirrors.
pub fn stage_host_pinned_bytes() -> u64 {
    (D2H_WINDOW_WORDS * 4 + (GEMM1_PTR_WORDS + GEMM2_PTR_WORDS) * 8) as u64
}

/// Persistent device bytes for GEMM pointer blocks and count/retirement state.
pub fn stage_device_bytes() -> u64 {
    ((GEMM1_PTR_WORDS + GEMM2_PTR_WORDS) * 8 + (MAX_EXPERTS + 2) * 4) as u64
}

#[cfg(test)]
pub(crate) fn prefill_scratch_bytes(
    rows: usize,
    contribute_rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
    fp8_slot: Option<&Fp8Slot>,
) -> Result<u64> {
    prefill_scratch_bytes_mixed(
        rows,
        contribute_rows,
        top_k,
        experts,
        hidden,
        (inter, inter),
        fp8_slot,
    )
}

#[cfg(test)]
pub(crate) fn prefill_scratch_bytes_mixed(
    rows: usize,
    contribute_rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    intermediates: (usize, usize),
    fp8_slot: Option<&Fp8Slot>,
) -> Result<u64> {
    prefill_scratch_bytes_mixed_with_boundary(
        rows,
        contribute_rows,
        top_k,
        experts,
        hidden,
        intermediates,
        (
            fp8_slot,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        ),
    )
}

pub(crate) fn prefill_scratch_bytes_mixed_with_boundary(
    rows: usize,
    contribute_rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    intermediates: (usize, usize),
    arithmetic: (Option<&Fp8Slot>, crate::model_spec::RoutedWeightBoundary),
) -> Result<u64> {
    prefill_layout_mixed(
        rows,
        contribute_rows,
        top_k,
        experts,
        hidden,
        intermediates,
        arithmetic,
    )
    .map(|layout| layout.bytes)
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

#[cfg(test)]
pub(crate) fn prefill_layout(
    rows: usize,
    contribute_rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
    fp8_slot: Option<&Fp8Slot>,
) -> Result<PrefillLayout> {
    prefill_layout_mixed(
        rows,
        contribute_rows,
        top_k,
        experts,
        hidden,
        (inter, inter),
        (
            fp8_slot,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        ),
    )
}

pub(crate) fn prefill_layout_mixed(
    rows: usize,
    contribute_rows: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    intermediates: (usize, usize),
    arithmetic: (Option<&Fp8Slot>, crate::model_spec::RoutedWeightBoundary),
) -> Result<PrefillLayout> {
    let (fp8_slot, routed_weight) = arithmetic;
    let (inter, shared_inter) = intermediates;
    ensure!(
        contribute_rows >= rows,
        "moe prefill contribute_rows={contribute_rows} below wave rows={rows}"
    );
    let assignments = rows
        .checked_mul(top_k)
        .context("moe prefill assignment overflow")?;
    let contribute_assignments = contribute_rows
        .checked_mul(top_k)
        .context("moe prefill contribute assignment overflow")?;
    let routed_hidden = assignments
        .checked_mul(hidden)
        .context("moe prefill hidden extent overflow")?;
    let routed_inter = assignments
        .checked_mul(inter)
        .context("moe prefill intermediate extent overflow")?;
    let routed_gate_up = routed_inter
        .checked_mul(2)
        .context("moe prefill gate-up extent overflow")?;
    let shared_inter_extent = rows
        .checked_mul(shared_inter)
        .context("moe prefill shared intermediate overflow")?;
    let shared_gate_up = shared_inter_extent
        .checked_mul(2)
        .context("moe prefill shared gate-up overflow")?;
    let shared_hidden = rows
        .checked_mul(hidden)
        .context("moe prefill shared hidden overflow")?;
    let mut at = 0u64;
    let counts = region(&mut at, experts, 4)?;
    let active_total = region(&mut at, 1, 4)?;
    let offsets = region(
        &mut at,
        experts
            .checked_add(1)
            .context("moe prefill expert extent overflow")?,
        4,
    )?;
    let cursor = region(&mut at, experts, 4)?;
    let pos_of = region(&mut at, contribute_assignments, 4)?;
    let compact_active = region(&mut at, assignments, 4)?;
    let x_perm = region(&mut at, routed_hidden, 2)?;
    let h1 = region(&mut at, routed_gate_up, 2)?;
    let act = region(&mut at, routed_inter, 2)?;
    let y_perm_bytes = if fp8_slot.is_none()
        && routed_weight == crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16
    {
        4
    } else {
        2
    };
    let y_perm = region(&mut at, routed_hidden, y_perm_bytes)?;
    let h1s = region(&mut at, shared_gate_up, 2)?;
    let act_s = region(&mut at, shared_inter_extent, 2)?;
    let shared_out = region(&mut at, shared_hidden, 2)?;
    let scale = region(&mut at, rows, 2)?;
    if let Some(slot) = fp8_slot {
        ensure!(
            hidden.is_multiple_of(128) && inter.is_multiple_of(64),
            "unsupported native expert GEMM shape hidden={hidden} intermediate={inter}: output channels must be multiples of 128 and K multiples of 64"
        );
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: u32::try_from(experts)?,
            hidden: u32::try_from(hidden)?,
            intermediate: u32::try_from(inter)?,
            top_k: u32::try_from(top_k)?,
        };
        ensure!(
            ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?
                == ExpertLayout::Fp8W8a16(slot.clone()),
            "FP8 prefill slot layout differs from geometry"
        );
    }
    Ok(PrefillLayout {
        counts,
        active_total,
        offsets,
        cursor,
        pos_of,
        compact_active,
        compact_capacity: assignments,
        x_perm,
        h1,
        act,
        y_perm,
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

/// Tensor-core problems precede CPU-form problems, each ordered by count then id.
pub(crate) fn problem_order(counts: &[i32], cpu_arithmetic: &[bool], order: &mut Vec<u32>) {
    order.clear();
    order.extend((0..counts.len() as u32).filter(|&e| counts[e as usize] > 0));
    order.sort_unstable_by_key(|&e| (cpu_arithmetic[e as usize], counts[e as usize], e));
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
    #[cfg(feature = "cuda")]
    transa: Vec<bs::cublasOperation_t>,
    #[cfg(feature = "cuda")]
    transb: Vec<bs::cublasOperation_t>,
    #[cfg(feature = "cuda")]
    alpha: Vec<f32>,
    #[cfg(feature = "cuda")]
    beta: Vec<f32>,
}

impl Stage {
    fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let event_flags = ds::CUevent_flags::CU_EVENT_DISABLE_TIMING;
        #[cfg(feature = "rocm")]
        let event_flags = 2;
        // SAFETY: the stage owns its pinned mirrors and initializes every transmitted word before upload.
        let stage = unsafe {
            Stage {
                d2h: ctx.alloc_pinned_with_flags::<i32>(D2H_WINDOW_WORDS, 0)?,
                gemm1_host: ctx.alloc_pinned_with_flags::<u64>(GEMM1_PTR_WORDS, 0)?,
                gemm1_dev: stream.alloc_zeros::<u64>(GEMM1_PTR_WORDS)?,
                gemm2_host: ctx.alloc_pinned_with_flags::<u64>(GEMM2_PTR_WORDS, 0)?,
                gemm2_dev: stream.alloc_zeros::<u64>(GEMM2_PTR_WORDS)?,
                gemm1_copy_done: ctx.new_event(Some(event_flags))?,
                gemm2_copy_done: ctx.new_event(Some(event_flags))?,
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
                #[cfg(feature = "cuda")]
                transa: vec![bs::cublasOperation_t::CUBLAS_OP_T; MAX_GROUPS],
                #[cfg(feature = "cuda")]
                transb: vec![bs::cublasOperation_t::CUBLAS_OP_N; MAX_GROUPS],
                #[cfg(feature = "cuda")]
                alpha: vec![1f32; MAX_GROUPS],
                #[cfg(feature = "cuda")]
                beta: vec![0f32; MAX_GROUPS],
            }
        };
        Ok(stage)
    }
}

pub(crate) struct PrefillOps {
    fp8: Option<(CudaFunction, CudaFunction)>,
    count_scan: CudaFunction,
    count_state: CudaSlice<u32>,
    count_blocks: usize,
    permute: CudaFunction,
    scatter: CudaFunction,
    scatter_f32: CudaFunction,
    lane16: CudaFunction,
    lane16_f32: CudaFunction,
    silu: CudaFunction,
    swiglu_rounding: crate::model_spec::SwigluRounding,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
    combine: CombineOps,
    blas: CudaBlas,
    stream: Arc<CudaStream>,
    stage: Mutex<Stage>,
}

pub(crate) fn arithmetic_mask(counts: &[u32], mask: &mut [bool]) -> Result<()> {
    ensure!(
        counts.len() == mask.len(),
        "prefill arithmetic counts={} mask={}",
        counts.len(),
        mask.len()
    );
    for (&rows, form) in counts.iter().zip(mask) {
        *form = rows <= LANE16_ROWS;
    }
    Ok(())
}

impl PrefillOps {
    #[cfg(test)]
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        format: ExpertFormat,
    ) -> Result<Self> {
        Self::new_with_boundaries(
            ctx,
            stream,
            format,
            crate::model_spec::SwigluRounding::Twice,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        )
    }

    pub fn new_with_boundaries(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        format: ExpertFormat,
        swiglu_rounding: crate::model_spec::SwigluRounding,
        routed_weight: crate::model_spec::RoutedWeightBoundary,
    ) -> Result<Self> {
        #[cfg(feature = "rocm")]
        if format == ExpertFormat::Fp8W8a16 {
            ctx.require_wmma()?;
        }
        let module = cubin::module(ctx, "moe_prefill")?;
        let act = cubin::module(ctx, "act")?;
        let blas = CudaBlas::new(stream.clone())?;
        #[cfg(feature = "cuda")]
        unsafe {
            bs::cublasSetMathMode(
                *blas.handle(),
                bs::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let count_scan = module.load_function("moe_count_scan")?;
        #[cfg(feature = "cuda")]
        let sms = usize::try_from(
            ctx.attribute(ds::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?;
        #[cfg(feature = "rocm")]
        let sms = usize::try_from(ctx.properties.compute_units)?;
        let blocks_per_sm =
            count_scan.occupancy_max_active_blocks_per_multiprocessor(BLOCK_THREADS, 0, None)?
                as usize;
        let count_blocks = sms
            .checked_mul(blocks_per_sm)
            .context("moe count CTA capacity overflow")?;
        ensure!(
            count_blocks > 0,
            "moe count kernel has no resident CTA capacity"
        );
        Ok(Self {
            fp8: (format == ExpertFormat::Fp8W8a16)
                .then(|| -> Result<_> {
                    Ok((
                        module.load_function("moe_gemm_w8a16")?,
                        module.load_function("moe_gemm_lane16_w8a16")?,
                    ))
                })
                .transpose()?,
            count_scan,
            count_state: stream.alloc_zeros(MAX_EXPERTS + 2)?,
            count_blocks,
            permute: module.load_function("moe_permute")?,
            scatter: module.load_function("moe_scatter")?,
            scatter_f32: module.load_function("moe_scatter_f32")?,
            lane16: module.load_function("moe_gemm_lane16")?,
            lane16_f32: module.load_function("moe_gemm_lane16_f32")?,
            silu: act.load_function("silu_mul")?,
            swiglu_rounding,
            routed_weight,
            combine: CombineOps::new(ctx)?,
            blas,
            stream: stream.clone(),
            stage: Mutex::new(Stage::new(ctx, stream)?),
        })
    }

    /// Count→scan→permute→GEMM1→silu→GEMM2→scatter into caller-owned contribution/done buffers.
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
        let plan = input.validate(self.routed_weight)?;
        ensure!(
            input.fp8_slot.is_none() || self.fp8.is_some(),
            "FP8 prefill operators not bound at setup"
        );
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
        let output_fp32 = input.fp8_slot.is_none()
            && self.routed_weight
                == crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16;
        let y_elem = if output_fp32 { 4 } else { 2 };
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

        let compact_at = scratch.ptr + layout.compact_active;
        let expected_active = input.wave.map(|(_, _, count)| count as i32).unwrap_or(-1);
        let mask_from_slots = i32::from(input.mask_from_slots);
        let assignment_gen = input
            .assignment_gen
            .as_ref()
            .map_or(0, |generation| generation.ptr);
        let count_blocks = (plan.assignments as usize)
            .div_ceil(2048)
            .max(1)
            .min(self.count_blocks);
        let permute_blocks = input
            .wave
            .map_or(plan.assignments as usize, |(_, _, count)| count as usize)
            .max(1);
        let (count_state, _state) = self.count_state.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.count_scan)
                .arg(&input.ids.ptr)
                .arg(&input.active.ptr)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&assignment_gen)
                .arg(&input.done.0.ptr)
                .arg(&count_state)
                .arg(&compact_at)
                .arg(&counts_at)
                .arg(&active_total_at)
                .arg(&offsets_at)
                .arg(&cursor_at)
                .arg(&(plan.assignments as i64))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&(layout.compact_capacity as i32))
                .arg(&expected_active)
                .arg(&mask_from_slots)
                .launch(grid(count_blocks, BLOCK_THREADS))?;
            stream
                .launch_builder(&self.permute)
                .arg(&input.ids.ptr)
                .arg(&input.x.ptr)
                .arg(&offsets_at)
                .arg(&cursor_at)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&assignment_gen)
                .arg(&compact_at)
                .arg(&active_total_at)
                .arg(&input.done.0.ptr)
                .arg(&pos_of_at)
                .arg(&x_perm_at)
                .arg(&(rows as i32))
                .arg(&(top_k as i32))
                .arg(&(experts as i32))
                .arg(&(hidden as i64))
                .arg(&mask_from_slots)
                .launch(grid(permute_blocks, BLOCK_THREADS))?;
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
        if let Some((cohort, base, expected_active)) = input.wave {
            ensure!(
                active_total == u64::from(expected_active),
                "moe prefill wave count mismatch cohort={cohort} base={base} rows={rows}: device active_total={active_total} != host={expected_active}"
            );
        }
        if active_total == 0 {
            if let Some(wait) = input.wait_copy {
                stream.wait(wait)?;
            }
            if let Some(reader) = input.reader_event {
                reader.record(stream)?;
            }
            return Ok(());
        }
        problem_order(counts, input.cpu_arithmetic, &mut stage.order);
        // FP32 routed contributions use lane16; grouped cuBLAS has no BF16-input/FP32-C form on sm86.
        let tensor_problems = if output_fp32 {
            0
        } else {
            stage
                .order
                .partition_point(|&expert| !input.cpu_arithmetic[expert as usize])
        };
        let small_rows = stage.order[tensor_problems..]
            .iter()
            .map(|&expert| counts[expert as usize] as u32)
            .max()
            .unwrap_or(0);
        group_runs(&stage.order[..tensor_problems], counts, &mut stage.runs);
        ensure!(
            stage.order.len() <= experts.min(plan.assignments as usize),
            "prefill active experts exceed the assignment bound"
        );
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
            stage.gemm1_copy_done.synchronize()?;
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
                if input.fp8_slot.is_some() {
                    ensure!(
                        gu == dn,
                        "FP8 prefill expert {e} needs identical native slot bases"
                    );
                }
                mirror[j] = gu;
                mirror[len1 + j] = x_perm_at + elem(off * u64::from(hidden));
                mirror[2 * len1 + j] = h1_at + elem(off * u64::from(2 * inter));
                mirror[3 * len1 + j] = counts[e] as u64;
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
        if let Some(wait) = input.wait_copy {
            stream.wait(wait)?;
        }
        let (gemm1_base, _g1) = stage.gemm1_dev.device_ptr(stream);
        unsafe {
            htod_ptrs(
                stream,
                gemm1_base,
                &stage.gemm1_host,
                4 * len1,
                &mut stage.gemm1_copy_done,
            )?;
        }
        if let Some(slot) = input.fp8_slot {
            self.native_gemm(
                stage,
                gemm1_base,
                len1,
                (tensor_problems, small_rows, false),
                (2 * inter, hidden),
                slot.gate_scale.start,
            )?;
        } else {
            unsafe {
                self.grouped_gemm(stage, gemm1_base, len1)
                    .context("moe prefill grouped GEMM 1")?;
            }
            self.small_gemm(
                gemm1_base,
                len1,
                (tensor_problems, small_rows, false),
                (2 * inter, hidden),
                None,
            )?;
        }

        let silu_grid = |elems: usize| grid(elems.div_ceil(BLOCK_THREADS as usize), BLOCK_THREADS);
        unsafe {
            stream
                .launch_builder(&self.silu)
                .arg(&h1_at)
                .arg(&act_at)
                .arg(&(active_total as i32))
                .arg(&(inter as i32))
                .arg(&(self.swiglu_rounding as i32))
                .launch(silu_grid(active_total as usize * inter as usize))?;
        }

        // GEMM2: one problem per active expert over the expert's whole
        // segment, C rebuilt from the offsets into the contiguous Y_perm.
        let len2 = stage.order.len();
        {
            stage.gemm2_copy_done.synchronize()?;
            let mirror = stage.gemm2_host.as_mut_slice()?;
            for (j, &e) in stage.order.iter().enumerate() {
                let e = e as usize;
                let off = stage.offsets[e] as u64;
                mirror[j] = if let Some(slot) = input.fp8_slot {
                    input.slots_down_host[e] + slot.down.start
                } else {
                    input.slots_down_host[e]
                };
                mirror[len2 + j] = act_at + elem(off * u64::from(inter));
                mirror[2 * len2 + j] = y_perm_at + off * u64::from(hidden) * y_elem;
                mirror[3 * len2 + j] = counts[e] as u64;
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
                4 * len2,
                &mut stage.gemm2_copy_done,
            )?;
        }
        if let Some(slot) = input.fp8_slot {
            self.native_gemm(
                stage,
                gemm2_base,
                len2,
                (tensor_problems, small_rows, false),
                (hidden, inter),
                slot.down_scale.start - slot.down.start,
            )?;
        } else {
            unsafe {
                if !output_fp32 {
                    self.grouped_gemm(stage, gemm2_base, len2)
                        .context("moe prefill grouped GEMM 2")?;
                }
            }
            self.small_gemm(
                gemm2_base,
                len2,
                (tensor_problems, small_rows, output_fp32),
                (hidden, inter),
                None,
            )?;
        }
        if let Some(reader) = input.reader_event {
            reader.record(stream)?;
        }
        unsafe {
            if output_fp32 {
                stream
                    .launch_builder(&self.scatter_f32)
                    .arg(&y_perm_at)
                    .arg(&input.weights.ptr)
                    .arg(&pos_of_at)
                    .arg(&compact_at)
                    .arg(&input.contributions.ptr)
                    .arg(&(plan.assignments as i64))
                    .arg(&(hidden as i64))
                    .arg(&(active_total as i32))
                    .arg(&1i32)
                    .launch(grid(active_total as usize, BLOCK_THREADS))?;
            } else {
                stream
                    .launch_builder(&self.scatter)
                    .arg(&y_perm_at)
                    .arg(&input.weights.ptr)
                    .arg(&pos_of_at)
                    .arg(&compact_at)
                    .arg(&input.contributions.ptr)
                    .arg(&(plan.assignments as i64))
                    .arg(&(hidden as i64))
                    .arg(&(active_total as i32))
                    .launch(grid(active_total as usize, BLOCK_THREADS))?;
            }
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
        let cohort = MoePrefillContribute {
            mask_from_slots: false,
            expert_inter: input.expert_inter,
            cpu_arithmetic: &[false; MAX_EXPERTS][..input.slot_valid.shape[0] as usize],
            fp8_slot: None,
            wave: None,
            wait_copy: None,
            reader_event: None,
            contributions: input.contributions,
            done: TensorMut(input.done.0),
            x: input.x,
            ids: input.ids,
            weights: input.weights,
            active: input.active,
            assignment_gen: Some(input.assignment_gen),
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
        let plan = cohort.validate(self.routed_weight)?;
        input.validate_y(&plan)?;
        scratch.validate(plan.layout.bytes)?;
        unsafe {
            zero_words(stream, cohort.done.0.ptr, plan.assignments as usize)?;
        }
        self.contribute(stream, &cohort, &plan, scratch)?;
        let y = input.y;
        let fin = MoePrefillFinish {
            expert_inter: input.expert_inter,
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
        self.finish_body(stream, fin, None, &plan, scratch)?;
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
        self.finish_body(stream, input, None, &plan, scratch)?;
        Ok(())
    }

    pub fn prefill_finish_routed(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillFinish,
        routed_peer: Tensor<bf16, 2>,
        scratch: &Scratch,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe prefill stream is not the PrefillOps construction stream (cuBLAS handle bound at construction)"
        );
        let plan = input.validate()?;
        routed_peer.validate()?;
        ensure!(
            routed_peer.shape == [plan.rows, plan.hidden]
                && routed_peer.stride == [u64::from(plan.hidden), 1],
            "moe prefill finish routed_peer must be dense [rows, hidden] (shape {:?})",
            routed_peer.shape
        );
        no_overlap(&routed_peer, &input.y.read(), "routed_peer", "y")?;
        scratch.validate(plan.layout.bytes)?;
        self.finish_body(stream, input, Some(routed_peer), &plan, scratch)?;
        Ok(())
    }

    /// The fused finish over caller-owned contributions: the shared FFN, then
    /// the routed sum, the shared sigmoid gate and the residual add in one
    /// launch. `input.y` is the residual buffer, read and written in place.
    pub fn prefill_finish_residual(
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
        self.finish_body_residual(stream, input, None, &plan, scratch)?;
        Ok(())
    }

    /// The fused peer finish: the routed sum comes from the peer GPU, the
    /// shared sigmoid gate and the residual add are local. `input.y` is the
    /// residual buffer, read and written in place.
    pub fn prefill_finish_routed_residual(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillFinish,
        routed_peer: Tensor<bf16, 2>,
        scratch: &Scratch,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(stream, &self.stream),
            "moe prefill stream is not the PrefillOps construction stream (cuBLAS handle bound at construction)"
        );
        let plan = input.validate()?;
        routed_peer.validate()?;
        ensure!(
            routed_peer.shape == [plan.rows, plan.hidden]
                && routed_peer.stride == [u64::from(plan.hidden), 1],
            "moe prefill finish routed_peer must be dense [rows, hidden] (shape {:?})",
            routed_peer.shape
        );
        no_overlap(&routed_peer, &input.y.read(), "routed_peer", "y")?;
        scratch.validate(plan.layout.bytes)?;
        self.finish_body_residual(stream, input, Some(routed_peer), &plan, scratch)?;
        Ok(())
    }

    pub fn combine_routed(&self, stream: &Arc<CudaStream>, input: &MoeCombineRouted) -> Result<()> {
        self.combine.combine_routed_prefill(stream, input)
    }

    fn finish_body(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillFinish,
        routed_peer: Option<Tensor<bf16, 2>>,
        plan: &PrefillPlan,
        scratch: &Scratch,
    ) -> Result<()> {
        let shared_out = self.shared_ffn(stream, &input, plan, scratch)?;
        // moe_combine's scale column over the same [rows] words; the nonzero
        // stride keeps Tensor::validate satisfied.
        let rows = plan.rows;
        let scale_at = scratch.ptr + plan.layout.scale;
        let mut gate = None;
        let mut scale = None;
        if let Some(shared_gate) = input.shared_expert_gate {
            let mut written = unsafe {
                TensorMut::<bf16, 2>::new(scale_at, [rows, 1], [1, 1], u64::from(rows) * 2)?
            };
            self.combine
                .shared_scale(stream, input.x, shared_gate, &mut written)?;
            gate = Some(shared_gate);
            scale = Some(written);
        }
        if let Some(routed_in) = routed_peer {
            self.combine.join_routed(
                stream,
                &MoeJoinRouted {
                    routed_in,
                    shared_out,
                    scale: scale.map(|s| s.read()),
                    y: input.y,
                },
            )?;
        } else {
            let combine_input = MoeCombine {
                ids: input.ids,
                final_active: input.final_active,
                done: input.done,
                contributions: input.contributions,
                shared_out,
                x: input.x,
                shared_expert_gate: gate,
                scale,
                y: input.y,
            };
            self.combine.combine(stream, &combine_input)?;
        }
        Ok(())
    }

    /// The shared FFN of one wave: the gate/up group (2 problems) and the down
    /// group (1 problem) share one 9-pointer block [A|B|C] per group, uploaded
    /// with a single H2D. The result is the scratch-backed shared output.
    fn shared_ffn(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoePrefillFinish,
        plan: &PrefillPlan,
        scratch: &Scratch,
    ) -> Result<Tensor<bf16, 2>> {
        let (rows, hidden, inter) = (plan.rows, plan.hidden, plan.shared_inter);
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
                stage.gemm1_copy_done.synchronize()?;
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
                    .arg(&(self.swiglu_rounding as i32))
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
        Ok(unsafe {
            Tensor::<bf16, 2>::new(
                shared_out_at,
                [rows, hidden],
                [u64::from(hidden), 1],
                u64::from(rows) * u64::from(hidden) * 2,
            )?
        })
    }

    /// The fused finish: the shared FFN, then the routed sum, the shared
    /// sigmoid gate and the residual add in one launch. `input.y` is the
    /// residual buffer, read and written in place.
    fn finish_body_residual(
        &self,
        stream: &Arc<CudaStream>,
        input: MoePrefillFinish,
        routed_peer: Option<Tensor<bf16, 2>>,
        plan: &PrefillPlan,
        scratch: &Scratch,
    ) -> Result<()> {
        let shared_out = self.shared_ffn(stream, &input, plan, scratch)?;
        if let Some(routed_in) = routed_peer {
            self.combine.join_routed_residual(
                stream,
                &MoeJoinRoutedResidual {
                    routed_in,
                    shared_out,
                    x: input.x,
                    shared_expert_gate: input.shared_expert_gate,
                    residual: Some(input.y),
                    y_out: None,
                },
                1,
            )?;
        } else {
            self.combine.combine_residual(
                stream,
                &MoeCombineResidual {
                    ids: input.ids,
                    final_active: input.final_active,
                    done: input.done,
                    contributions: input.contributions,
                    shared_out,
                    x: input.x,
                    shared_expert_gate: input.shared_expert_gate,
                    residual: Some(input.y),
                    y_out: None,
                },
                1,
            )?;
        }
        Ok(())
    }

    fn native_gemm(
        &self,
        stage: &Stage,
        block: u64,
        problems: usize,
        small: (usize, u32, bool),
        shape: (u32, u32),
        scale_offset: u64,
    ) -> Result<()> {
        let (tensor_problems, _, _) = small;
        if tensor_problems > 0 {
            let kernel = &self.fp8.as_ref().context("native expert GEMM missing")?.0;
            let rows = stage
                .runs
                .last()
                .context("native tensor-core groups missing")?
                .0;
            // SAFETY: both native matrices and BF16 input/output views cover the validated shape.
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&block)
                    .arg(&(problems as i32))
                    .arg(&(tensor_problems as i32))
                    .arg(&(shape.0 as i32))
                    .arg(&(shape.1 as i32))
                    .arg(&scale_offset)
                    .launch(crate::backend::gpu::LaunchConfig {
                        grid_dim: (
                            shape.0.div_ceil(128),
                            rows.div_ceil(W8A16_TILE_ROWS),
                            tensor_problems as u32,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        self.small_gemm(block, problems, small, shape, Some(scale_offset))
    }

    fn small_gemm(
        &self,
        block: u64,
        problems: usize,
        small: (usize, u32, bool),
        shape: (u32, u32),
        scale_offset: Option<u64>,
    ) -> Result<()> {
        let (first, rows, output_fp32) = small;
        if first == problems {
            return Ok(());
        }
        let kernel = match (scale_offset.is_some(), output_fp32) {
            (true, _) => &self.fp8.as_ref().context("FP8 lane16 operator missing")?.1,
            (false, true) => &self.lane16_f32,
            (false, false) => &self.lane16,
        };
        let offset = scale_offset.unwrap_or(0);
        let (problem_count, first_problem, columns, inner) = (
            problems as i32,
            first as i32,
            shape.0 as i32,
            shape.1 as i32,
        );
        // SAFETY: the stream-ordered A/B/C/rows mirror covers every selected problem and matrix.
        unsafe {
            let mut launch = self.stream.launch_builder(kernel);
            launch
                .arg(&block)
                .arg(&problem_count)
                .arg(&first_problem)
                .arg(&columns)
                .arg(&inner);
            if scale_offset.is_some() {
                launch.arg(&offset);
            }
            launch.launch(crate::backend::gpu::LaunchConfig {
                grid_dim: (
                    shape.0.div_ceil(LANE16_ROWS),
                    rows,
                    (problems - first) as u32,
                ),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }

    /// BF16 A/B/C, FP32 accumulation, alpha=1, beta=0, opA=T, opB=N.
    /// # Safety
    /// `block` must address 3*`problems` device pointers whose targets match
    /// the group shapes and are kept alive until the stream completes.
    unsafe fn grouped_gemm(&self, stage: &Stage, block: u64, problems: usize) -> Result<()> {
        let arrays = &stage.arrays;
        let n = arrays.m.len();
        if n == 0 {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "rocm")]
        {
            let mut offset = 0usize;
            for group in 0..n {
                let size = usize::try_from(arrays.size[group])?;
                let end = offset
                    .checked_add(size)
                    .context("HIP grouped GEMM extent overflow")?;
                ensure!(end <= problems, "HIP grouped GEMM pointer extent exceeded");
                unsafe {
                    self.blas.bf16_gemm(
                        crate::backend::setup::LinearShape {
                            rows: usize::try_from(arrays.n[group])?,
                            output: usize::try_from(arrays.m[group])?,
                            input: usize::try_from(arrays.k[group])?,
                        },
                        [
                            block + offset as u64 * 8,
                            block + (problems + offset) as u64 * 8,
                            block + (2 * problems + offset) as u64 * 8,
                        ],
                        [
                            usize::try_from(arrays.lda[group])?,
                            usize::try_from(arrays.ldb[group])?,
                            usize::try_from(arrays.ldc[group])?,
                        ],
                        Some(size),
                        false,
                    )?;
                }
                offset = end;
            }
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
        #[cfg(feature = "cuda")]
        ds::cuMemsetD32Async(dst, 0, words, stream.cu_stream()).result()?;
        #[cfg(feature = "rocm")]
        stream.memset_words(dst, 0, words)?;
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
    #[cfg(feature = "cuda")]
    let dst = window.as_mut_ptr()?;
    #[cfg(feature = "rocm")]
    let (span, _write) = unsafe { window.stream_synced_mut_slice(stream) };
    #[cfg(feature = "rocm")]
    let dst = span.as_mut_ptr();
    unsafe {
        #[cfg(feature = "rocm")]
        stream.copy_raw(dst as u64, src, words * size_of::<i32>(), 2)?;
        #[cfg(feature = "cuda")]
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
/// `dst` covers `words` u64 words ordered on `stream`. The host mirror stays
/// immutable until the recorded `done` event completes.
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
    #[cfg(feature = "cuda")]
    let host = src.as_ptr()?;
    #[cfg(feature = "rocm")]
    let (span, _read) = unsafe { src.stream_synced_slice(stream) };
    #[cfg(feature = "rocm")]
    let host = span.as_ptr();
    unsafe {
        #[cfg(feature = "rocm")]
        stream.copy_raw(dst, host as u64, words * size_of::<u64>(), 1)?;
        #[cfg(feature = "cuda")]
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
