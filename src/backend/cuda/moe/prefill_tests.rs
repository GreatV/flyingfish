//! CPU gates for the MoE prefill op (scratch sizing, problem grouping) and
//! the frozen GPU gate. The GPU test is ignored by default and runs only
//! under an acknowledged window with --include-ignored --test-threads=1.
use super::combine::{CombineOps, MoeCombine};
use super::prefill::{
    GEMM1_PTR_WORDS, GEMM2_PTR_WORDS, MAX_EXPERTS, MoePrefill, MoePrefillContribute, PrefillOps,
    d2h_active_total_word, d2h_window_words, group_runs, prefill_layout, prefill_scratch_bytes,
    problem_order,
};
use super::reference::{self, Combine, MoeDims, MoeFixture};
use super::testing;
use crate::backend::cuda::contracts::{Scratch, TensorMut};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::sync::Arc;

fn up256(n: u64) -> u64 {
    n.div_ceil(256) * 256
}

#[test]
fn scratch_sizing_matches_region_math() {
    for (rows, top_k, experts, hidden, inter) in [
        (2usize, 2usize, 4usize, 8usize, 4usize),
        (37, 8, 256, 2048, 512),
        (4096, 8, 256, 2048, 512),
    ] {
        let assignments = rows * top_k;
        let regions: [(usize, u64); 15] = [
            (experts, 4),
            (1, 4),
            (experts + 1, 4),
            (experts, 4),
            (assignments, 4),
            (assignments, 4),
            (assignments * hidden, 2),
            (assignments * 2 * inter, 2),
            (assignments * inter, 2),
            (assignments * hidden, 2),
            (assignments * hidden, 2),
            (rows * 2 * inter, 2),
            (rows * inter, 2),
            (rows * hidden, 2),
            (rows, 2),
        ];
        let expected: u64 = regions
            .iter()
            .map(|&(n, size)| up256((n as u64) * size))
            .sum();
        let got = prefill_scratch_bytes(rows, top_k, experts, hidden, inter).unwrap();
        assert_eq!(got, expected, "rows={rows} top_k={top_k}");
        assert_eq!(got % 256, 0);
        assert!(
            got > regions.iter().map(|&(n, s)| (n as u64) * s).sum::<u64>(),
            "scratch must pad at least one region"
        );
    }
    // Production prefill chunk: raw regions 532,950,024 B plus the region
    // paddings; y_perm duplicates the contributions extent for the
    // per-expert grouped GEMM2.
    assert_eq!(
        prefill_scratch_bytes(4096, 8, 256, 2048, 512).unwrap(),
        532_950_528
    );
    assert!(
        prefill_scratch_bytes(4096, 8, 256, 2048, 512).unwrap()
            > prefill_scratch_bytes(512, 8, 256, 2048, 512).unwrap()
    );
}

/// Stage capacities: the D2H window holds counts[E], the 256-byte-aligned
/// active total and offsets[E+1] for every experts count up to MAX_EXPERTS,
/// and the pointer blocks hold one A/B/C pointer per active expert plus the
/// shared entries.
#[test]
fn stage_capacity_math() {
    for experts in [1usize, 2, 8, 255, 256, 257, 1023, MAX_EXPERTS] {
        let padded_counts = up256(experts as u64 * 4) as usize;
        assert_eq!(d2h_active_total_word(experts), padded_counts / 4);
        assert_eq!(
            d2h_window_words(experts),
            padded_counts / 4 + up256(4) as usize / 4 + experts + 1,
            "experts={experts}"
        );
    }
    for experts in 1..=MAX_EXPERTS {
        assert!(
            d2h_window_words(experts) <= d2h_window_words(MAX_EXPERTS),
            "D2H window exceeds the stage capacity at experts={experts}"
        );
    }
    assert_eq!(d2h_window_words(MAX_EXPERTS), 2113);
    assert_eq!(GEMM1_PTR_WORDS, 3 * (MAX_EXPERTS + 2));
    assert_eq!(GEMM2_PTR_WORDS, 3 * (MAX_EXPERTS + 1));
}

#[test]
fn grouping_orders_and_runs() {
    let mut order = Vec::new();
    let mut runs = Vec::new();
    let counts = [0i32, 3, 0, 1, 3, 2];
    problem_order(&counts, &mut order);
    assert_eq!(order, vec![3, 5, 1, 4]);
    group_runs(&order, &counts, &mut runs);
    assert_eq!(runs, vec![(1, 1), (2, 1), (3, 2)]);
    problem_order(&[0, 0, 0], &mut order);
    assert!(order.is_empty());
    group_runs(&order, &[0, 0, 0], &mut runs);
    assert!(runs.is_empty());
    let flat = [2, 2, 2, 2];
    problem_order(&flat, &mut order);
    assert_eq!(order, vec![0, 1, 2, 3]);
    group_runs(&order, &flat, &mut runs);
    assert_eq!(runs, vec![(2, 4)]);
}

/// The eviction-arm fixture: hot-expert weights scaled by 1.25 (bf16 round);
/// the hot expert occupies choice 0, the other choices are distinct experts
/// (router outputs are distinct per row).
fn eviction_fixture(fx: &MoeFixture, rows: usize) -> (MoeFixture, Vec<i32>, Vec<bf16>) {
    let (h, i) = (fx.dims.hidden, fx.dims.inter);
    let mut dims = fx.dims;
    dims.rows = rows;
    let mut fx2 = reference::fixture(dims, 0);
    let gu_base = HOT_EXPERT * 2 * i * h;
    for (dst, src) in fx2.gate_up[gu_base..gu_base + 2 * i * h]
        .iter_mut()
        .zip(fx.gate_up[gu_base..gu_base + 2 * i * h].iter())
    {
        *dst = bf16::from_f32(src.to_f32() * 1.25);
    }
    let dn_base = HOT_EXPERT * h * i;
    for (dst, src) in fx2.down[dn_base..dn_base + h * i]
        .iter_mut()
        .zip(fx.down[dn_base..dn_base + h * i].iter())
    {
        *dst = bf16::from_f32(src.to_f32() * 1.25);
    }
    let top_k = fx.dims.top_k;
    let mut ids = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        ids.push(HOT_EXPERT as i32);
        for c in 0..top_k - 1 {
            let mut e = c;
            if e == HOT_EXPERT {
                e = top_k;
            }
            ids.push(e as i32);
        }
    }
    let weights: Vec<bf16> = (0..rows * top_k)
        .map(|a| bf16::from_f32(0.1 + 0.0125 * (a % 5) as f32))
        .collect();
    (fx2, ids, weights)
}

/// Combine-order spread of the reference on the eviction weight structure.
fn eviction_combine_noise(fx2: &MoeFixture, ids: &[i32], weights: &[bf16], rows: &[usize]) -> f32 {
    let top_k = fx2.dims.top_k;
    let all: Vec<usize> = (0..top_k).collect();
    let mut noise = 0.0f32;
    for &row in rows {
        let ids_row = &ids[row * top_k..(row + 1) * top_k];
        let weights_row = &weights[row * top_k..(row + 1) * top_k];
        let asc =
            reference::moe_row_ids(fx2, row, ids_row, weights_row, &all, Combine::AscendingBf16);
        let desc = reference::moe_row_ids(
            fx2,
            row,
            ids_row,
            weights_row,
            &all,
            Combine::DescendingBf16,
        );
        let f32c = reference::moe_row_ids(fx2, row, ids_row, weights_row, &all, Combine::Fp32);
        for d in 0..fx2.dims.hidden {
            noise = noise
                .max((asc[d].to_f32() - desc[d].to_f32()).abs())
                .max((asc[d].to_f32() - f32c[d].to_f32()).abs());
        }
    }
    noise
}

/// Freeze receipt for the eviction-arm noise cell: the reference
/// combine-order spread on the eviction weight structure.
#[test]
fn eviction_noise_cell_is_pinned() {
    let fx = reference::fixture(MoeDims::minicpm47(6), 0);
    let (fx2, ids, weights) = eviction_fixture(&fx, 6);
    let noise = eviction_combine_noise(&fx2, &ids, &weights, &[0, 1, 2, 3, 4]);
    println!(
        "MOE_EVICT_NOISE_CELL max_spread={noise:.9} tol_evict={:.9}",
        2.0 * noise
    );
    // Pinned to the 2026-10-17 CPU-measured value on the distinct-experts
    // eviction fixture within 1e-6; the exact f32 bits are libm-dependent
    // through f32::exp.
    assert!(
        (noise - 0.001_953_125).abs() < 1e-6,
        "cell drifted: {noise}"
    );
}

const SKEW_SUBSET: [usize; 16] = [
    3, 17, 42, 77, 100, 137, 150, 171, 190, 201, 220, 231, 240, 247, 250, 255,
];

/// Hot-expert id of the skew and eviction cases.
const HOT_EXPERT: usize = 7;

fn planted_uniform(rows: usize, experts: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    for r in 0..rows {
        for kk in 0..top_k {
            ids[r * top_k + kk] = ((r * top_k + kk) % experts) as i32;
            weights[r * top_k + kk] = bf16::from_f32(0.1 + 0.0125 * ((r + 3 * kk) % 5) as f32);
        }
    }
    (ids, weights)
}

fn planted_skewed(rows: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    for r in 0..rows {
        for kk in 0..top_k {
            ids[r * top_k + kk] = SKEW_SUBSET[(r + kk) % 16] as i32;
            weights[r * top_k + kk] = bf16::from_f32(0.1 + 0.0125 * ((r + 5 * kk) % 5) as f32);
        }
    }
    (ids, weights)
}

/// One expert takes ~1/3 of all assignments: the hot expert occupies choice 0
/// of every row (distinct top-k caps a single expert at rows assignments);
/// other choices are distinct experts.
fn planted_hot(rows: usize, experts: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    for r in 0..rows {
        ids[r * top_k] = HOT_EXPERT as i32;
        for c in 1..top_k {
            let mut e = (r + c - 1) % (experts - 1);
            if e == HOT_EXPERT {
                e = experts - 1;
            }
            ids[r * top_k + c] = e as i32;
        }
        for c in 0..top_k {
            weights[r * top_k + c] = bf16::from_f32(0.1 + 0.0125 * ((r + 3 * c) % 5) as f32);
        }
    }
    (ids, weights)
}

/// 16 rows at an odd stride, coprime with the power-of-two counts under test.
fn sample_rows(rows: usize) -> Vec<usize> {
    (0..16).map(|j| (403 * j) % rows).collect()
}

fn all_active(rows: usize, top_k: usize) -> Vec<u32> {
    vec![1u32; rows * top_k]
}

struct CaseBufs {
    x: CudaSlice<bf16>,
    ids: CudaSlice<i32>,
    active: CudaSlice<u32>,
    weights: CudaSlice<bf16>,
    gens: CudaSlice<u64>,
    y: CudaSlice<bf16>,
}

fn case_bufs(
    s: &Arc<CudaStream>,
    fx: &MoeFixture,
    rows: usize,
    ids: &[i32],
    active: &[u32],
    weights: &[bf16],
    gens: &[u64],
) -> Result<CaseBufs> {
    let hidden = fx.dims.hidden;
    Ok(CaseBufs {
        x: s.clone_htod(&fx.x[..rows * hidden])?,
        ids: s.clone_htod(ids)?,
        active: s.clone_htod(active)?,
        weights: s.clone_htod(weights)?,
        gens: s.clone_htod(gens)?,
        y: s.alloc_zeros(rows * hidden)?,
    })
}

struct SharedBufs {
    sg: CudaSlice<bf16>,
    su: CudaSlice<bf16>,
    sd: CudaSlice<bf16>,
    seg: CudaSlice<bf16>,
}

/// Host slot tables plus the device tables (pointer arrays, valid mask,
/// generation) they were published with.
struct DeviceSlots {
    gu_host: Vec<u64>,
    dn_host: Vec<u64>,
    gu_dev: CudaSlice<u64>,
    dn_dev: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    generation: CudaSlice<u64>,
}

fn device_slots(
    s: &Arc<CudaStream>,
    gu_host: Vec<u64>,
    dn_host: Vec<u64>,
    valid: Vec<u32>,
    generation: Vec<u64>,
) -> Result<DeviceSlots> {
    Ok(DeviceSlots {
        gu_dev: s.clone_htod(&gu_host)?,
        dn_dev: s.clone_htod(&dn_host)?,
        valid: s.clone_htod(&valid)?,
        generation: s.clone_htod(&generation)?,
        gu_host,
        dn_host,
    })
}

fn moe_input<'a>(
    s: &Arc<CudaStream>,
    fx: &MoeFixture,
    rows: usize,
    bufs: &CaseBufs,
    slots: &'a DeviceSlots,
    shared: &SharedBufs,
    y: TensorMut<bf16, 2>,
) -> Result<MoePrefill<'a>> {
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    Ok(MoePrefill {
        x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
        active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&slots.gu_dev, s, [experts as u32], [1])?,
        slots_down: testing::view(&slots.dn_dev, s, [experts as u32], [1])?,
        slot_valid: testing::view(&slots.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&slots.generation, s, [experts as u32], [1])?,
        assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
        slots_gate_up_host: &slots.gu_host,
        slots_down_host: &slots.dn_host,
        shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: testing::view(&shared.seg, s, [hidden as u32], [1])?,
        y,
    })
}

/// Caller-owned contribute targets: the contributions/done pair a cohort
/// writes into.
struct CohortBufs {
    contributions: CudaSlice<bf16>,
    done: CudaSlice<u32>,
}

/// A contribute input over caller-owned contributions/done buffers.
fn contribute_input<'a>(
    s: &Arc<CudaStream>,
    fx: &MoeFixture,
    rows: usize,
    bufs: &CaseBufs,
    slots: &'a DeviceSlots,
    shared: &SharedBufs,
    cohort: &CohortBufs,
) -> Result<MoePrefillContribute<'a>> {
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    Ok(MoePrefillContribute {
        x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
        active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&slots.gu_dev, s, [experts as u32], [1])?,
        slots_down: testing::view(&slots.dn_dev, s, [experts as u32], [1])?,
        slot_valid: testing::view(&slots.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&slots.generation, s, [experts as u32], [1])?,
        assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
        slots_gate_up_host: &slots.gu_host,
        slots_down_host: &slots.dn_host,
        shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: testing::view(&shared.seg, s, [hidden as u32], [1])?,
        contributions: testing::view(
            &cohort.contributions,
            s,
            [rows as u32, top_k as u32, hidden as u32],
            [(top_k * hidden) as u64, hidden as u64, 1],
        )?,
        done: testing::mut_view(
            &cohort.done,
            s,
            [rows as u32, top_k as u32],
            [top_k as u64, 1],
        )?,
    })
}

/// Raw device-address copies for the mutant/join cells.
fn raw_dtoh<T>(s: &Arc<CudaStream>, src: u64, dst: &mut [T]) -> Result<()> {
    s.context().bind_to_thread()?;
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().cast(),
            src,
            std::mem::size_of_val(dst),
            s.cu_stream(),
        )
        .result()?;
    }
    s.synchronize()?;
    Ok(())
}

fn raw_htod<T>(s: &Arc<CudaStream>, dst: u64, src: &[T]) -> Result<()> {
    s.context().bind_to_thread()?;
    unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            dst,
            src.as_ptr().cast(),
            std::mem::size_of_val(src),
            s.cu_stream(),
        )
        .result()?;
    }
    s.synchronize()?;
    Ok(())
}

struct Arm<'a> {
    label: &'a str,
    rows: usize,
    ids: &'a [i32],
    active: &'a [u32],
    weights: &'a [bf16],
    check: &'a [usize],
    gens: &'a [u64],
    slots: &'a DeviceSlots,
}

struct GpuCase<'a> {
    s: &'a Arc<CudaStream>,
    ops: &'a PrefillOps,
    fx: &'a MoeFixture,
    scratch: &'a Scratch,
    shared: &'a SharedBufs,
    tol: f32,
}

impl GpuCase<'_> {
    fn bufs(
        &self,
        rows: usize,
        ids: &[i32],
        active: &[u32],
        weights: &[bf16],
        gens: &[u64],
    ) -> Result<CaseBufs> {
        case_bufs(self.s, self.fx, rows, ids, active, weights, gens)
    }

    /// Runs one arm, compares the active choices of every checked row against
    /// the reference and records tolerance misses into `defects`.
    fn run(&self, arm: Arm<'_>, defects: &mut Vec<String>) -> Result<()> {
        let Arm {
            label,
            rows,
            ids,
            active,
            weights,
            check,
            gens,
            slots,
        } = arm;
        let s = self.s;
        let (top_k, hidden) = (self.fx.dims.top_k, self.fx.dims.hidden);
        let case_bufs = self.bufs(rows, ids, active, weights, gens)?;
        let y_view = testing::dense_mut(&case_bufs.y, s, rows as u32, hidden as u32)?;
        let input = moe_input(s, self.fx, rows, &case_bufs, slots, self.shared, y_view)?;
        self.ops.prefill(s, input, self.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&case_bufs.y)?;
        let mut max_diff = 0.0f32;
        for &row in check {
            let keep_row: Vec<usize> = (0..top_k)
                .filter(|&c| active[row * top_k + c] == 1)
                .collect();
            let ids_row = &ids[row * top_k..(row + 1) * top_k];
            let weights_row = &weights[row * top_k..(row + 1) * top_k];
            // Zero router weights make every contribution an exact zero, so
            // the expected row of an all-inactive arm is the gated shared
            // expert alone.
            let expected = if keep_row.is_empty() {
                let zeros = vec![bf16::ZERO; top_k];
                let all: Vec<usize> = (0..top_k).collect();
                reference::moe_row_ids(self.fx, row, ids_row, &zeros, &all, Combine::Fp32)
            } else {
                reference::moe_row_ids(self.fx, row, ids_row, weights_row, &keep_row, Combine::Fp32)
            };
            for c in 0..hidden {
                let got = y[row * hidden + c];
                anyhow::ensure!(
                    got.to_f32().is_finite(),
                    "moe_prefill {label} row {row} column {c} is non-finite"
                );
                max_diff = max_diff.max((got.to_f32() - expected[c].to_f32()).abs());
            }
        }
        eprintln!(
            "moe_prefill {label} rows={rows} checked={} max_diff={max_diff:.6} tol={:.6}",
            check.len(),
            self.tol
        );
        if max_diff > self.tol {
            defects.push(format!(
                "{label} rows={rows} max_diff {max_diff:.6} > tol {:.6}",
                self.tol
            ));
        }
        if rows >= 4096 {
            let norms: Vec<f64> = (0..rows)
                .map(|r| {
                    (0..hidden)
                        .map(|c| {
                            let v = y[r * hidden + c].to_f32() as f64;
                            v * v
                        })
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            let mean_all: f64 = norms.iter().sum::<f64>() / rows as f64;
            let mean_checked: f64 =
                check.iter().map(|&r| norms[r]).sum::<f64>() / check.len() as f64;
            if !(0.5..=2.0).contains(&(mean_all / mean_checked)) {
                defects.push(format!(
                    "{label} rows={rows} aggregate row-norm control: {mean_all:.4} vs {mean_checked:.4}"
                ));
            }
        }
        Ok(())
    }
}

/// Shared context of the K2/K4 join cells.
struct GateCtx<'a> {
    s: &'a Arc<CudaStream>,
    ops: &'a PrefillOps,
    combine: &'a CombineOps,
    fx: &'a MoeFixture,
    slots: &'a DeviceSlots,
    shared: &'a SharedBufs,
    scratch: &'a Scratch,
}

/// Real shared-FFN bits for the join cells: one wrapper run on this input,
/// then the shared_out region read back from scratch.
fn capture_shared_out(
    ctx: &GateCtx<'_>,
    rows: usize,
    ids: &[i32],
    active: &[u32],
    weights: &[bf16],
    gens: &[u64],
) -> Result<CudaSlice<bf16>> {
    let fx = ctx.fx;
    let (top_k, hidden, experts, inter) = (
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.experts,
        fx.dims.inter,
    );
    let case_bufs = case_bufs(ctx.s, fx, rows, ids, active, weights, gens)?;
    let y: CudaSlice<bf16> = ctx.s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, ctx.s, rows as u32, hidden as u32)?;
    let input = moe_input(ctx.s, fx, rows, &case_bufs, ctx.slots, ctx.shared, y_view)?;
    ctx.ops.prefill(ctx.s, input, ctx.scratch)?;
    ctx.s.synchronize()?;
    let layout = prefill_layout(rows, top_k, experts, hidden, inter)?;
    let mut host = vec![bf16::ZERO; rows * hidden];
    raw_dtoh(ctx.s, ctx.scratch.ptr + layout.shared_out, &mut host)?;
    Ok(ctx.s.clone_htod(&host)?)
}

/// One wrapper-style join over caller-owned buffers: sigmoid gate once per
/// row, then the combine with `bufs.active` as the final union mask;
/// returns the y bits.
fn join_combine(
    ctx: &GateCtx<'_>,
    rows: usize,
    bufs: &CaseBufs,
    shared_out: &CudaSlice<bf16>,
    contributions: &CudaSlice<bf16>,
    done: &CudaSlice<u32>,
) -> Result<Vec<u16>> {
    let (s, fx, shared) = (ctx.s, ctx.fx, ctx.shared);
    let (top_k, hidden) = (fx.dims.top_k, fx.dims.hidden);
    let scale: CudaSlice<bf16> = s.alloc_zeros(rows)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let mut scale_view = testing::dense_mut(&scale, s, rows as u32, 1)?;
    ctx.combine.shared_scale(
        s,
        testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        testing::view(&shared.seg, s, [hidden as u32], [1])?,
        &mut scale_view,
    )?;
    let input = MoeCombine {
        ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
        final_active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
        done: testing::view(done, s, [rows as u32, top_k as u32], [top_k as u64, 1])?,
        contributions: testing::view(
            contributions,
            s,
            [rows as u32, top_k as u32, hidden as u32],
            [(top_k * hidden) as u64, hidden as u64, 1],
        )?,
        shared_out: testing::dense(shared_out, s, rows as u32, hidden as u32)?,
        x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        shared_expert_gate: testing::view(&shared.seg, s, [hidden as u32], [1])?,
        scale: scale_view,
        y: testing::dense_mut(&y, s, rows as u32, hidden as u32)?,
    };
    ctx.combine.combine(s, &input)?;
    s.synchronize()?;
    Ok(s.clone_dtoh(&y)?.iter().map(|v| v.to_bits()).collect())
}

/// Contribute `cohorts` sequentially into one caller-owned
/// contributions/done pair, then one join; returns the y bits.
fn join_bits(
    ctx: &GateCtx<'_>,
    rows: usize,
    ids: &[i32],
    weights: &[bf16],
    gens: &[u64],
    shared_out: &CudaSlice<bf16>,
    cohorts: &[Vec<u32>],
) -> Result<Vec<u16>> {
    let (s, fx) = (ctx.s, ctx.fx);
    let (top_k, hidden) = (fx.dims.top_k, fx.dims.hidden);
    let cohort = CohortBufs {
        contributions: s.alloc_zeros(rows * top_k * hidden)?,
        done: s.alloc_zeros(rows * top_k)?,
    };
    for active in cohorts {
        let bufs = case_bufs(s, fx, rows, ids, active, weights, gens)?;
        let input = contribute_input(s, fx, rows, &bufs, ctx.slots, ctx.shared, &cohort)?;
        ctx.ops.prefill_contribute(s, input, ctx.scratch)?;
        s.synchronize()?;
    }
    let union = case_bufs(s, fx, rows, ids, &all_active(rows, top_k), weights, gens)?;
    join_combine(
        ctx,
        rows,
        &union,
        shared_out,
        &cohort.contributions,
        &cohort.done,
    )
}

/// Runs one control input and returns its max diff against the reference on
/// the original ids/weights.
fn control_diff(
    case: &GpuCase<'_>,
    rows: usize,
    ids: &[i32],
    active: &[u32],
    weights: &[bf16],
    gens: &[u64],
    slots: &DeviceSlots,
) -> Result<f32> {
    let s = case.s;
    let (top_k, hidden) = (case.fx.dims.top_k, case.fx.dims.hidden);
    let case_bufs = case.bufs(rows, ids, active, weights, gens)?;
    let y_view = testing::dense_mut(&case_bufs.y, s, rows as u32, hidden as u32)?;
    let input = moe_input(s, case.fx, rows, &case_bufs, slots, case.shared, y_view)?;
    case.ops.prefill(s, input, case.scratch)?;
    s.synchronize()?;
    let y = s.clone_dtoh(&case_bufs.y)?;
    let all: Vec<usize> = (0..top_k).collect();
    let mut max_diff = 0.0f32;
    for row in 0..rows {
        let expected = reference::moe_row_ids(
            case.fx,
            row,
            &ids[row * top_k..(row + 1) * top_k],
            &weights[row * top_k..(row + 1) * top_k],
            &all,
            Combine::Fp32,
        );
        for c in 0..hidden {
            max_diff = max_diff.max((y[row * hidden + c].to_f32() - expected[c].to_f32()).abs());
        }
    }
    Ok(max_diff)
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn moe_prefill_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    match std::env::var("FF_MOE_CHILD").as_deref() {
        Ok("stale") => return stale_generation_child(),
        Ok("dup") => return duplicate_ids_child(),
        Ok("oobpos") => return pos_of_oob_child(),
        _ => {}
    }
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let mut defects: Vec<String> = Vec::new();
    let ops = PrefillOps::new(&d.ctx, s)?;
    // The 16384-row arm slices the same fixture, so its weights are the ones
    // every smaller-rows arm already scores against.
    let fx = reference::fixture(MoeDims::minicpm47(16384), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let gu_buf: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn_buf: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _g1) = gu_buf.device_ptr(s);
    let (dn_ptr, _g2) = dn_buf.device_ptr(s);
    let gu_stride = (2 * inter * hidden * 2) as u64;
    let dn_stride = (hidden * inter * 2) as u64;
    let main_gu: Vec<u64> = (0..experts)
        .map(|e| gu_ptr + e as u64 * gu_stride)
        .collect();
    let main_dn: Vec<u64> = (0..experts)
        .map(|e| dn_ptr + e as u64 * dn_stride)
        .collect();
    let main_slots = device_slots(
        s,
        main_gu.clone(),
        main_dn.clone(),
        vec![1u32; experts],
        vec![1u64; experts],
    )?;
    let scratch_bytes = prefill_scratch_bytes(16384, top_k, experts, hidden, inter)?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    let tol = reference::tolerance(&fx, &(0..8).collect::<Vec<_>>());
    let case = GpuCase {
        s,
        ops: &ops,
        fx: &fx,
        scratch: &scratch,
        shared: &shared,
        tol,
    };

    // The per-expert GEMM2 shape (n = m_e per expert, restored from the
    // per-assignment n=1 problems) is confirmed on device by every arm
    // below: they score the full path against the unchanged oracle under
    // the frozen tolerance.
    for rows in [1usize, 37, 512, 4096] {
        // Full-row reference compares run only at rows <= 37; above that the
        // debug-build host reference (about 0.1 GFLOP per row) exceeds the
        // GPU window, so checks sample rows and every arm still runs.
        let check: Vec<usize> = if rows <= 37 {
            (0..rows).collect()
        } else {
            sample_rows(rows)
        };
        let active = all_active(rows, top_k);
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "uniform",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        let (ids, weights) = planted_skewed(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "skewed",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        if rows == 4096 {
            let (ids, weights) = planted_hot(rows, experts, top_k);
            let gens = vec![1u64; rows * top_k];
            case.run(
                Arm {
                    label: "hot",
                    rows,
                    ids: &ids,
                    active: &active,
                    weights: &weights,
                    check: &check,
                    gens: &gens,
                    slots: &main_slots,
                },
                &mut defects,
            )?;
        }
        // Fixture-driven end-to-end: route_row ids/weights for every row at
        // rows <= 512, and for the coprime-stride sample at 4096.
        let sample = sample_rows(rows);
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            if rows <= 512 || sample.contains(&r) {
                let routed = reference::route_row(&fx, r);
                ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
                weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
            } else {
                for kk in 0..top_k {
                    ids[r * top_k + kk] = ((r * top_k + kk) % experts) as i32;
                    weights[r * top_k + kk] = bf16::from_f32(0.1);
                }
            }
        }
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "routed",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // rows=16384: 131072 assignments drive the permute/scatter grids, the
    // count loop and the largest per-expert GEMM2 segments; 16-row coprime
    // sample checks like the 512/4096 arms.
    {
        let rows = 16384usize;
        let check = sample_rows(rows);
        let active = all_active(rows, top_k);
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "uniform",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        let (ids, weights) = planted_skewed(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "skewed",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // Partial-active cohort at rows=37: every 5th assignment inactive; the
    // reference computes exactly the kept choices of each row.
    {
        let rows = 37usize;
        let check: Vec<usize> = (0..rows).collect();
        let mut active = all_active(rows, top_k);
        for (a, v) in active.iter_mut().enumerate() {
            if a % 5 == 0 {
                *v = 0;
            }
        }
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "partial-active",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // All-inactive cohort at rows=2: active_total == 0 skips the routed
    // problems and the combine writes the gated shared expert only.
    {
        let rows = 2usize;
        let check: Vec<usize> = (0..rows).collect();
        let active = vec![0u32; rows * top_k];
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "all-inactive",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // Negative controls at rows=37 (uniform arm covers every expert, so
    // experts 3 and 17 receive rows).
    {
        let rows = 37usize;
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let mut swapped_gu = main_gu.clone();
        let mut swapped_dn = main_dn.clone();
        swapped_gu.swap(3, 17);
        swapped_dn.swap(3, 17);
        let swapped = device_slots(
            s,
            swapped_gu,
            swapped_dn,
            vec![1u32; experts],
            vec![1u64; experts],
        )?;
        let diff = control_diff(&case, rows, &ids, &active, &weights, &gens, &swapped)?;
        anyhow::ensure!(
            diff > tol,
            "swapped slot-table control was not detected: {diff:.6} <= {tol:.6}"
        );

        // Gate/up split off-by-one: every expert block of gate_up rolled by
        // one row.
        let mut rolled = fx.gate_up.clone();
        for e in 0..experts {
            let base = e * 2 * inter * hidden;
            for j in 0..2 * inter {
                let src = base + ((j + 1) % (2 * inter)) * hidden;
                let dst = base + j * hidden;
                rolled.copy_within(src..src + hidden, dst);
            }
        }
        let rolled_buf: CudaSlice<bf16> = s.clone_htod(rolled.as_slice())?;
        let (rolled_ptr, _gr) = rolled_buf.device_ptr(s);
        let rolled_slots = device_slots(
            s,
            (0..experts)
                .map(|e| rolled_ptr + e as u64 * gu_stride)
                .collect(),
            main_dn.clone(),
            vec![1u32; experts],
            vec![1u64; experts],
        )?;
        let diff = control_diff(&case, rows, &ids, &active, &weights, &gens, &rolled_slots)?;
        anyhow::ensure!(
            diff > tol,
            "rolled gate/up split control was not detected: {diff:.6} <= {tol:.6}"
        );
    }

    // Eviction-reuse control: all assignments route to the hot expert, whose
    // weights move to a new device buffer scaled by 1.25 (bf16 round) with
    // the slot generation bumped to 2; the relaunch must read the replaced
    // weights. The tolerance is twice the reference combine-order spread on
    // this weight structure.
    {
        let rows = 37usize;
        let (fx2, ids, weights) = eviction_fixture(&fx, rows);
        let tol_evict = 2.0 * eviction_combine_noise(&fx2, &ids, &weights, &[0, 1, 2, 3, 4]);
        let check: Vec<usize> = (0..rows).collect();
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let ev_case = GpuCase {
            s: case.s,
            ops: case.ops,
            fx: case.fx,
            scratch: case.scratch,
            shared: case.shared,
            tol: tol_evict,
        };
        ev_case.run(
            Arm {
                label: "eviction-before",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        let scale_block = |src: &[bf16], e: usize, rows_n: usize, width: usize| -> Vec<bf16> {
            let mut out = src[e * rows_n * width..(e + 1) * rows_n * width].to_vec();
            for v in &mut out {
                *v = bf16::from_f32(v.to_f32() * 1.25);
            }
            out
        };
        let hot_gu = scale_block(&fx.gate_up, HOT_EXPERT, 2 * inter, hidden);
        let hot_dn = scale_block(&fx.down, HOT_EXPERT, hidden, inter);
        let hot_gu_buf: CudaSlice<bf16> = s.clone_htod(&hot_gu)?;
        let hot_dn_buf: CudaSlice<bf16> = s.clone_htod(&hot_dn)?;
        let (hot_gu_ptr, _h1) = hot_gu_buf.device_ptr(s);
        let (hot_dn_ptr, _h2) = hot_dn_buf.device_ptr(s);
        let mut ev_gu = main_gu.clone();
        let mut ev_dn = main_dn.clone();
        ev_gu[HOT_EXPERT] = hot_gu_ptr;
        ev_dn[HOT_EXPERT] = hot_dn_ptr;
        // Only the replaced slot advances to generation 2; the snapshot each
        // assignment carries comes from the table observed at routing time.
        let mut ev_gen = vec![1u64; experts];
        ev_gen[HOT_EXPERT] = 2;
        let mut snapshot = vec![1u64; rows * top_k];
        for r in 0..rows {
            snapshot[r * top_k] = 2;
        }
        let evicted = device_slots(s, ev_gu, ev_dn, vec![1u32; experts], ev_gen)?;
        let case_bufs = case.bufs(rows, &ids, &active, &weights, &snapshot)?;
        let y_view = testing::dense_mut(&case_bufs.y, s, rows as u32, hidden as u32)?;
        let input = moe_input(s, case.fx, rows, &case_bufs, &evicted, case.shared, y_view)?;
        case.ops.prefill(s, input, case.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&case_bufs.y)?;
        let all: Vec<usize> = (0..top_k).collect();
        let (mut new_diff, mut old_diff) = (0.0f32, 0.0f32);
        for row in 0..rows {
            let ids_row = &ids[row * top_k..(row + 1) * top_k];
            let weights_row = &weights[row * top_k..(row + 1) * top_k];
            let expected_new =
                reference::moe_row_ids(&fx2, row, ids_row, weights_row, &all, Combine::Fp32);
            let expected_old =
                reference::moe_row_ids(&fx, row, ids_row, weights_row, &all, Combine::Fp32);
            for c in 0..hidden {
                let got = y[row * hidden + c].to_f32();
                new_diff = new_diff.max((got - expected_new[c].to_f32()).abs());
                old_diff = old_diff.max((got - expected_old[c].to_f32()).abs());
            }
        }
        eprintln!(
            "moe_prefill eviction-reuse rows={rows} expert={HOT_EXPERT} generation=2 new-weight max_diff={new_diff:.6} old-weight max_diff={old_diff:.6} tol={tol_evict:.6}"
        );
        if new_diff > tol_evict {
            defects.push(format!(
                "eviction-reuse new-weight {new_diff:.6} > tol {tol_evict:.6}"
            ));
        }
        anyhow::ensure!(
            old_diff > tol_evict,
            "eviction-reuse matched the old weights: new {new_diff:.6} old {old_diff:.6} tol {tol_evict:.6}"
        );
    }

    // K2 duplicate-pos_of mutant: after a clean all-active contribute, the
    // real moe_scatter is re-launched over a planted pos_of where choice 1
    // of row 0 points at choice 0's Y_perm row; the joined output must move
    // beyond the frozen tolerance (two active assignments reading one
    // Y_perm row).
    {
        let rows = 37usize;
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let combine_ops = CombineOps::new(&d.ctx)?;
        let ctx = GateCtx {
            s,
            ops: &ops,
            combine: &combine_ops,
            fx: &fx,
            slots: &main_slots,
            shared: &shared,
            scratch: &scratch,
        };
        let shared_out = capture_shared_out(&ctx, rows, &ids, &active, &weights, &gens)?;
        let cohort = CohortBufs {
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            done: s.alloc_zeros(rows * top_k)?,
        };
        let bufs = case_bufs(s, &fx, rows, &ids, &active, &weights, &gens)?;
        let input = contribute_input(s, &fx, rows, &bufs, &main_slots, &shared, &cohort)?;
        ops.prefill_contribute(s, input, &scratch)?;
        s.synchronize()?;
        let layout = prefill_layout(rows, top_k, experts, hidden, inter)?;
        let mut pos_of = vec![0i32; rows * top_k];
        raw_dtoh(s, scratch.ptr + layout.pos_of, &mut pos_of)?;
        let planted = (pos_of[0], pos_of[1]);
        pos_of[1] = pos_of[0];
        raw_htod(s, scratch.ptr + layout.pos_of + 4, &pos_of[1..2])?;
        let module = cubin::module(&d.ctx, "moe_prefill")?;
        let scatter = module.load_function("moe_scatter")?;
        let (contributions_ptr, _gc) = cohort.contributions.device_ptr(s);
        let (weights_ptr, _gw) = bufs.weights.device_ptr(s);
        let (active_ptr, _ga) = bufs.active.device_ptr(s);
        unsafe {
            s.launch_builder(&scatter)
                .arg(&(scratch.ptr + layout.y_perm))
                .arg(&weights_ptr)
                .arg(&(scratch.ptr + layout.pos_of))
                .arg(&active_ptr)
                .arg(&contributions_ptr)
                .arg(&((rows * top_k) as i64))
                .arg(&(hidden as i64))
                .launch(grid(rows * top_k, 256))?;
        }
        s.synchronize()?;
        let bits = join_combine(
            &ctx,
            rows,
            &bufs,
            &shared_out,
            &cohort.contributions,
            &cohort.done,
        )?;
        let mut row0_diff = 0.0f32;
        let mut max_diff = 0.0f32;
        for row in 0..rows {
            let expected = reference::moe_row_ids(
                &fx,
                row,
                &ids[row * top_k..(row + 1) * top_k],
                &weights[row * top_k..(row + 1) * top_k],
                &(0..top_k).collect::<Vec<_>>(),
                Combine::Fp32,
            );
            for c in 0..hidden {
                let got = bf16::from_bits(bits[row * hidden + c]).to_f32();
                let diff = (got - expected[c].to_f32()).abs();
                if row == 0 {
                    row0_diff = row0_diff.max(diff);
                }
                max_diff = max_diff.max(diff);
            }
        }
        eprintln!(
            "moe_prefill K2 duplicate-pos_of rows={rows} planted pos_of[1]={} (was {}) row0 max_diff={row0_diff:.6} max_diff={max_diff:.6} tol={:.6}",
            planted.0, planted.1, tol
        );
        anyhow::ensure!(
            row0_diff > tol && max_diff > tol,
            "duplicate pos_of was not detected: row0 {row0_diff:.6} max {max_diff:.6} <= tol {tol:.6}"
        );
    }

    // K4 two-cohort join, two partitions of the routed rows=37 input into
    // ONE caller-owned contributions/done pair plus one wrapper-style join
    // with the union mask. The FAIL predicate is the production cohort
    // form: split by expert id parity (every expert's rows wholly in one
    // cohort). The choice-parity split is reported only (first differing
    // word if any).
    {
        let rows = 37usize;
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        let combine_ops = CombineOps::new(&d.ctx)?;
        let ctx = GateCtx {
            s,
            ops: &ops,
            combine: &combine_ops,
            fx: &fx,
            slots: &main_slots,
            shared: &shared,
            scratch: &scratch,
        };
        let shared_out =
            capture_shared_out(&ctx, rows, &ids, &all_active(rows, top_k), &weights, &gens)?;
        let single = join_bits(
            &ctx,
            rows,
            &ids,
            &weights,
            &gens,
            &shared_out,
            &[all_active(rows, top_k)],
        )?;
        let mut by_expert = (vec![0u32; rows * top_k], vec![0u32; rows * top_k]);
        for (a, &id) in ids.iter().enumerate() {
            if id % 2 == 0 {
                by_expert.0[a] = 1;
            } else {
                by_expert.1[a] = 1;
            }
        }
        let expert_split = join_bits(
            &ctx,
            rows,
            &ids,
            &weights,
            &gens,
            &shared_out,
            &[by_expert.0, by_expert.1],
        )?;
        let expert_bad = single.iter().zip(&expert_split).position(|(a, b)| a != b);
        eprintln!(
            "moe_prefill K4 two-cohort join rows={rows} cohorts=expert-parity words={} first_differing_word={:?} (FAIL predicate)",
            single.len(),
            expert_bad
        );
        anyhow::ensure!(
            expert_bad.is_none(),
            "expert-parity two-cohort join is not bitwise equal to the single cohort: first differing word {expert_bad:?} of {}",
            single.len()
        );
        let mut even = vec![0u32; rows * top_k];
        let mut odd = vec![0u32; rows * top_k];
        for a in 0..rows * top_k {
            if a % 2 == 0 {
                even[a] = 1;
            } else {
                odd[a] = 1;
            }
        }
        let split = join_bits(&ctx, rows, &ids, &weights, &gens, &shared_out, &[even, odd])?;
        let first_bad = single.iter().zip(&split).position(|(a, b)| a != b);
        eprintln!(
            "moe_prefill K4 two-cohort join rows={rows} cohorts=choice-parity words={} first_differing_word={:?} (reported result, not a FAIL predicate)",
            single.len(),
            first_bad
        );
    }

    anyhow::ensure!(defects.is_empty(), "recorded defects: {defects:?}");
    Ok(())
}

/// True when the error chain reports the device-side trap's launch failure.
fn trap_launched(err: &anyhow::Error) -> bool {
    const MARKERS: [&str; 4] = [
        "illegal instruction",
        "launch failed",
        "unspecified launch failure",
        "assert",
    ];
    err.chain().any(|cause| {
        let text = cause.to_string().to_lowercase();
        MARKERS.iter().any(|marker| text.contains(marker))
    })
}

/// Child shared setup: fixture-sized slot tables, shared expert buffers and
/// scratch. Returns the weight buffers so their device storage outlives the
/// launch.
struct ChildSetup {
    slots: DeviceSlots,
    shared: SharedBufs,
    scratch: Scratch,
    _weights: (CudaSlice<bf16>, CudaSlice<bf16>),
    _scratch_buf: CudaSlice<u8>,
}

fn child_setup(s: &Arc<CudaStream>, fx: &MoeFixture, generation: Vec<u64>) -> Result<ChildSetup> {
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let gu_ptr = {
        let (ptr, guard) = gu.device_ptr(s);
        drop(guard);
        ptr
    };
    let dn_ptr = {
        let (ptr, guard) = dn.device_ptr(s);
        drop(guard);
        ptr
    };
    let slots = device_slots(
        s,
        (0..experts)
            .map(|e| gu_ptr + e as u64 * (2 * inter * hidden * 2) as u64)
            .collect(),
        (0..experts)
            .map(|e| dn_ptr + e as u64 * (hidden * inter * 2) as u64)
            .collect(),
        vec![1u32; experts],
        generation,
    )?;
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    let scratch_bytes = prefill_scratch_bytes(fx.dims.rows, top_k, experts, hidden, inter)?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let scratch_ptr = {
        let (ptr, guard) = scratch_buf.device_ptr(s);
        drop(guard);
        ptr
    };
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    Ok(ChildSetup {
        slots,
        shared,
        scratch,
        _weights: (gu, dn),
        _scratch_buf: scratch_buf,
    })
}

/// A small fixture whose replaced slot (generation 2) mismatches assignment
/// 0's routing-time snapshot (generation 1); the permute guard must trap.
/// Exits 3 only on the trap's launch failure; every other outcome fails the
/// test (harness exit 101).
fn stale_generation_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s)?;
    let fx = reference::fixture(MoeDims::new(8, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden) = (fx.dims.experts, fx.dims.top_k, fx.dims.hidden);
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let mut table_gen = vec![1u64; experts];
    table_gen[ids[0] as usize] = 2;
    let mut snapshot = vec![1u64; rows * top_k];
    for r in 0..rows {
        for c in 0..top_k {
            snapshot[r * top_k + c] = table_gen[ids[r * top_k + c] as usize];
        }
    }
    snapshot[0] = 1;
    let setup = child_setup(s, &fx, table_gen)?;
    let active = all_active(rows, top_k);
    let case_bufs = case_bufs(s, &fx, rows, &ids, &active, &weights, &snapshot)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = moe_input(
        s,
        &fx,
        rows,
        &case_bufs,
        &setup.slots,
        &setup.shared,
        y_view,
    )?;
    match ops.prefill(s, input, &setup.scratch) {
        Ok(_event) => anyhow::bail!("stale generation did not trap"),
        Err(err) if trap_launched(&err) => std::process::exit(3),
        Err(err) => anyhow::bail!("prefill failed without the trap: {err:#}"),
    }
}

/// Duplicate expert ids within a row's active choices must trip the permute
/// guard. Exits 3 only on the trap's launch failure; every other outcome
/// fails the test (harness exit 101).
fn duplicate_ids_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s)?;
    let fx = reference::fixture(MoeDims::new(2, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden) = (fx.dims.experts, fx.dims.top_k, fx.dims.hidden);
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let mut dup = ids.clone();
    dup[1] = dup[0];
    let active = all_active(rows, top_k);
    let gens = vec![1u64; rows * top_k];
    let setup = child_setup(s, &fx, vec![1u64; experts])?;
    let case_bufs = case_bufs(s, &fx, rows, &dup, &active, &weights, &gens)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = moe_input(
        s,
        &fx,
        rows,
        &case_bufs,
        &setup.slots,
        &setup.shared,
        y_view,
    )?;
    match ops.prefill(s, input, &setup.scratch) {
        Ok(_event) => anyhow::bail!("duplicate ids did not trap"),
        Err(err) if trap_launched(&err) => std::process::exit(3),
        Err(err) => anyhow::bail!("prefill failed without the trap: {err:#}"),
    }
}

/// pos_of[0] planted to `assignments` (out of range) before a direct
/// moe_scatter relaunch over the cohort's scratch; the scatter's position
/// guard must trap. Exits 3 only on the trap's launch failure; every other
/// outcome fails the test (harness exit 101).
fn pos_of_oob_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s)?;
    let fx = reference::fixture(MoeDims::new(8, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let active = all_active(rows, top_k);
    let gens = vec![1u64; rows * top_k];
    let setup = child_setup(s, &fx, vec![1u64; experts])?;
    let cohort = CohortBufs {
        contributions: s.alloc_zeros(rows * top_k * hidden)?,
        done: s.alloc_zeros(rows * top_k)?,
    };
    let bufs = case_bufs(s, &fx, rows, &ids, &active, &weights, &gens)?;
    let input = contribute_input(s, &fx, rows, &bufs, &setup.slots, &setup.shared, &cohort)?;
    ops.prefill_contribute(s, input, &setup.scratch)?;
    s.synchronize()?;
    let layout = prefill_layout(rows, top_k, experts, hidden, inter)?;
    let planted = [(rows * top_k) as i32];
    raw_htod(s, setup.scratch.ptr + layout.pos_of, &planted)?;
    let module = cubin::module(&d.ctx, "moe_prefill")?;
    let scatter = module.load_function("moe_scatter")?;
    let (contributions_ptr, _gc) = cohort.contributions.device_ptr(s);
    let (weights_ptr, _gw) = bufs.weights.device_ptr(s);
    let (active_ptr, _ga) = bufs.active.device_ptr(s);
    let launched = unsafe {
        s.launch_builder(&scatter)
            .arg(&(setup.scratch.ptr + layout.y_perm))
            .arg(&weights_ptr)
            .arg(&(setup.scratch.ptr + layout.pos_of))
            .arg(&active_ptr)
            .arg(&contributions_ptr)
            .arg(&((rows * top_k) as i64))
            .arg(&(hidden as i64))
            .launch(grid(rows * top_k, 256))
    };
    let outcome = launched
        .and_then(|_| s.synchronize())
        .map_err(anyhow::Error::from);
    eprintln!(
        "moe_prefill K2 out-of-range pos_of rows={rows} planted pos_of[0]={} assignments={}",
        planted[0],
        rows * top_k
    );
    match outcome {
        Ok(()) => anyhow::bail!("out-of-range pos_of did not trap"),
        Err(err) if trap_launched(&err) => std::process::exit(3),
        Err(err) => anyhow::bail!("scatter relaunch failed without the trap: {err:#}"),
    }
}

// Same isolation as the gate children: the parent makes no CUDA call.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_stale_generation_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "stale")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "stale-generation child exit {:?} != 3 (the permute trap was not proven)",
        status.code()
    );
    Ok(())
}

// Parent for the duplicate-ids child; no CUDA call in the parent.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_duplicate_ids_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "dup")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "duplicate-ids child exit {:?} != 3 (the permute trap was not proven)",
        status.code()
    );
    Ok(())
}

// Parent for the out-of-range pos_of child; no CUDA call in the parent.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_pos_of_bounds_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "oobpos")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "out-of-range pos_of child exit {:?} != 3 (the scatter trap was not proven)",
        status.code()
    );
    Ok(())
}

#[test]
fn stage_accounting_is_pinned() {
    assert_eq!(super::prefill::stage_host_pinned_bytes(), 57_676);
    assert_eq!(super::prefill::stage_device_bytes(), 49_224);
}
