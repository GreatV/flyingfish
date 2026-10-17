//! CPU gates for the MoE prefill op (scratch sizing, validation, grouping)
//! and the frozen GPU gate. The GPU test is ignored by default and runs only
//! under an acknowledged window with --include-ignored --test-threads=1.
use super::decode::{DecodeOps, MoeDecode, decode_scratch_bytes};
use super::prefill::{
    MoePrefill, PrefillOps, STAGES, StageTiming, group_runs, prefill_scratch_bytes, problem_order,
};
use super::reference::{self, MoeDims, MoeFixture};
use super::testing;
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
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
        let regions: [(usize, u64); 21] = [
            (experts, 4),
            (1, 4),
            (experts + 1, 4),
            (experts, 4),
            (assignments, 4),
            (assignments, 4),
            (assignments, 4),
            (assignments, 2),
            (assignments * hidden, 2),
            (assignments * 2 * inter, 2),
            (assignments * inter, 2),
            (assignments * hidden, 2),
            (rows * 2 * inter, 2),
            (rows * inter, 2),
            (rows * hidden, 2),
            (experts + 2, 8),
            (experts + 2, 8),
            (experts + 2, 8),
            (experts + 2, 8),
            (experts + 2, 8),
            (experts + 2, 8),
        ];
        let expected: u64 = regions
            .iter()
            .map(|&(n, size)| up256((n as u64) * size))
            .sum();
        let got = prefill_scratch_bytes(rows, top_k, experts, hidden, inter);
        assert_eq!(got, expected, "rows={rows} top_k={top_k}");
        assert_eq!(got % 256, 0);
        assert!(
            got > regions.iter().map(|&(n, s)| (n as u64) * s).sum::<u64>(),
            "scratch must pad at least one region"
        );
    }
    // Production prefill chunk: raw regions 398_801_288 B plus the offsets
    // padding (252 B) and six pointer-array paddings (240 B each).
    assert_eq!(prefill_scratch_bytes(4096, 8, 256, 2048, 512), 398_935_040);
    assert!(
        prefill_scratch_bytes(4096, 8, 256, 2048, 512)
            > prefill_scratch_bytes(512, 8, 256, 2048, 512)
    );
}

#[test]
fn grouping_orders_and_runs() {
    let counts = [0i32, 3, 0, 1, 3, 2];
    assert_eq!(problem_order(&counts), vec![3, 5, 1, 4]);
    assert_eq!(
        group_runs(&problem_order(&counts), &counts),
        vec![(1, 1), (2, 1), (3, 2)]
    );
    assert!(problem_order(&[0, 0, 0]).is_empty());
    assert!(group_runs(&[], &[]).is_empty());
    let flat = [2, 2, 2, 2];
    assert_eq!(problem_order(&flat), vec![0, 1, 2, 3]);
    assert_eq!(group_runs(&problem_order(&flat), &flat), vec![(2, 4)]);
}

const ROWS: u32 = 4;
const EXPERTS: u32 = 16;
const TOP_K: u32 = 4;
const HIDDEN: u32 = 32;
const INTER: u32 = 16;

fn tensor<T: Copy>(
    at: &mut u64,
    shape: [u32; 2],
    stride: [u64; 2],
    element_bytes: u64,
) -> Result<Tensor<T, 2>> {
    let span = stride[0] * u64::from(shape[0] - 1) + u64::from(shape[1]);
    let t = unsafe { Tensor::<T, 2>::new(*at, shape, stride, span * element_bytes)? };
    *at += 0x10_0000;
    Ok(t)
}

/// Synthetic-address input for CPU validation checks (no Device needed).
fn input_at(hidden: u32, experts: u32, top_k: u32) -> Result<MoePrefill> {
    let mut at = 0x10_0000u64;
    let x = tensor::<bf16>(&mut at, [ROWS, hidden], [u64::from(hidden), 1], 2)?;
    let ids = tensor::<i32>(&mut at, [ROWS, top_k], [u64::from(top_k), 1], 4)?;
    let active = tensor::<u32>(&mut at, [ROWS, top_k], [u64::from(top_k), 1], 4)?;
    let weights = tensor::<bf16>(&mut at, [ROWS, top_k], [u64::from(top_k), 1], 2)?;
    let assignment_gen = tensor::<u64>(&mut at, [ROWS, top_k], [u64::from(top_k), 1], 8)?;
    let slots_gu = unsafe { Tensor::<u64, 1>::new(at, [experts], [1], u64::from(experts) * 8)? };
    at += 0x10_0000;
    let slots_dn = unsafe { Tensor::<u64, 1>::new(at, [experts], [1], u64::from(experts) * 8)? };
    at += 0x10_0000;
    let valid = unsafe { Tensor::<u32, 1>::new(at, [experts], [1], u64::from(experts) * 4)? };
    at += 0x10_0000;
    let generation = unsafe { Tensor::<u64, 1>::new(at, [experts], [1], u64::from(experts) * 8)? };
    at += 0x10_0000;
    let shared_gate = tensor::<bf16>(&mut at, [INTER, hidden], [u64::from(hidden), 1], 2)?;
    let shared_up = tensor::<bf16>(&mut at, [INTER, hidden], [u64::from(hidden), 1], 2)?;
    let shared_down = tensor::<bf16>(&mut at, [hidden, INTER], [u64::from(INTER), 1], 2)?;
    let shared_expert_gate =
        unsafe { Tensor::<bf16, 1>::new(at, [hidden], [1], u64::from(hidden) * 2)? };
    at += 0x10_0000;
    let y = unsafe {
        TensorMut::<bf16, 2>::new(
            at,
            [ROWS, hidden],
            [u64::from(hidden), 1],
            u64::from(ROWS * hidden) * 2,
        )?
    };
    Ok(MoePrefill {
        x,
        ids,
        active,
        weights,
        assignment_gen,
        slots_gate_up: slots_gu,
        slots_down: slots_dn,
        slot_valid: valid,
        slot_generation: generation,
        shared_gate,
        shared_up,
        shared_down,
        shared_expert_gate,
        y,
    })
}

#[test]
fn validation_errors_are_named() -> Result<()> {
    input_at(HIDDEN, EXPERTS, TOP_K)?.validate()?;
    let mut m = input_at(HIDDEN, EXPERTS, TOP_K)?;
    m.y = unsafe {
        TensorMut::<bf16, 2>::new(
            0x900_0000,
            [ROWS + 1, HIDDEN],
            [u64::from(HIDDEN), 1],
            1 << 20,
        )?
    };
    assert!(m.validate().unwrap_err().to_string().contains("dense"));
    let mut m = input_at(HIDDEN, EXPERTS, TOP_K)?;
    m.ids = tensor::<i32>(
        &mut 0xA000_0000u64,
        [ROWS, TOP_K],
        [u64::from(TOP_K) + 2, 1],
        4,
    )?;
    assert!(
        m.validate()
            .unwrap_err()
            .to_string()
            .contains("ids/active/weights")
    );
    let mut m = input_at(HIDDEN, EXPERTS, TOP_K)?;
    m.active = tensor::<u32>(
        &mut 0xA800_0000u64,
        [ROWS, TOP_K],
        [u64::from(TOP_K) + 2, 1],
        4,
    )?;
    assert!(
        m.validate()
            .unwrap_err()
            .to_string()
            .contains("ids/active/weights")
    );
    let mut m = input_at(HIDDEN, EXPERTS, TOP_K)?;
    m.slots_down = unsafe { Tensor::<u64, 1>::new(0xB000_0000, [EXPERTS - 1], [1], 128)? };
    assert!(
        m.validate()
            .unwrap_err()
            .to_string()
            .contains("slot tables")
    );
    let mut m = input_at(HIDDEN, EXPERTS, TOP_K)?;
    m.slot_generation = unsafe { Tensor::<u64, 1>::new(0xB800_0000, [EXPERTS - 1], [1], 120)? };
    assert!(
        m.validate()
            .unwrap_err()
            .to_string()
            .contains("slot tables")
    );
    assert!(
        input_at(33, EXPERTS, TOP_K)?
            .validate()
            .unwrap_err()
            .to_string()
            .contains("even")
    );
    assert!(
        input_at(HIDDEN, 1025, TOP_K)?
            .validate()
            .unwrap_err()
            .to_string()
            .contains("above 1024")
    );
    assert!(
        input_at(HIDDEN, EXPERTS, 33)?
            .validate()
            .unwrap_err()
            .to_string()
            .contains("top_k outside")
    );
    let mut m = input_at(HIDDEN, EXPERTS, TOP_K)?;
    m.y = unsafe {
        TensorMut::<bf16, 2>::new(
            m.x.ptr + 16,
            [ROWS, HIDDEN],
            [u64::from(HIDDEN), 1],
            1 << 20,
        )?
    };
    assert!(m.validate().unwrap_err().to_string().contains("overlap"));
    Ok(())
}

// Oracle mirroring reference::moe_row_choices numerics for planted
// ids/weights over a choice subset: per-expert BF16 boundaries, ascending-id
// BF16 accumulation, shared expert gated by sigmoid(bf16(dot)). Pinned
// against reference::moe_row below.
#[derive(Clone, Copy, Debug)]
enum OracleCombine {
    AscendingBf16,
    DescendingBf16,
    Fp32,
}

fn oracle_row(
    fx: &MoeFixture,
    row: usize,
    ids: &[i32],
    weights: &[bf16],
    keep: &[usize],
) -> Vec<bf16> {
    oracle_row_with(fx, row, ids, weights, keep, OracleCombine::AscendingBf16)
}

/// One expert's BF16-boundary weighted contribution on a row.
fn oracle_contribution(fx: &MoeFixture, e: usize, xr: &[bf16], w: bf16) -> Vec<bf16> {
    let (h, i) = (fx.dims.hidden, fx.dims.inter);
    let dot = |a: &[bf16], b: &[bf16]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x.to_f32() * y.to_f32())
            .sum()
    };
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut prod = Vec::with_capacity(i);
    for j in 0..i {
        let g = bf16::from_f32(dot(xr, fx.gate_row(e, j)));
        let u = bf16::from_f32(dot(xr, fx.up_row(e, j)));
        let s = bf16::from_f32(silu(g.to_f32()));
        prod.push(bf16::from_f32(s.to_f32() * u.to_f32()));
    }
    let mut out = Vec::with_capacity(h);
    for drow in fx.down.chunks(i).skip(e * h).take(h) {
        let lin = bf16::from_f32(dot(&prod, drow));
        out.push(bf16::from_f32(lin.to_f32() * w.to_f32()));
    }
    out
}

/// The sigmoid-gated shared expert term of a row.
fn oracle_shared(fx: &MoeFixture, xr: &[bf16]) -> Vec<bf16> {
    let (h, i) = (fx.dims.hidden, fx.dims.inter);
    let dot = |a: &[bf16], b: &[bf16]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x.to_f32() * y.to_f32())
            .sum()
    };
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut prod = Vec::with_capacity(i);
    for j in 0..i {
        let g = bf16::from_f32(dot(xr, &fx.shared_gate[j * h..(j + 1) * h]));
        let u = bf16::from_f32(dot(xr, &fx.shared_up[j * h..(j + 1) * h]));
        let s = bf16::from_f32(silu(g.to_f32()));
        prod.push(bf16::from_f32(s.to_f32() * u.to_f32()));
    }
    let scale = bf16::from_f32(
        1.0 / (1.0 + (-bf16::from_f32(dot(xr, &fx.shared_expert_gate)).to_f32()).exp()),
    );
    let mut gated = Vec::with_capacity(h);
    for drow in fx.shared_down.chunks(i).take(h) {
        let shared = bf16::from_f32(dot(&prod, drow));
        gated.push(bf16::from_f32(scale.to_f32() * shared.to_f32()));
    }
    gated
}

fn oracle_row_with(
    fx: &MoeFixture,
    row: usize,
    ids: &[i32],
    weights: &[bf16],
    keep: &[usize],
    combine: OracleCombine,
) -> Vec<bf16> {
    let h = fx.dims.hidden;
    let xr = &fx.x[row * h..(row + 1) * h];
    let mut slots: Vec<(usize, bf16)> = keep
        .iter()
        .map(|&c| (ids[c] as usize, weights[c]))
        .collect();
    slots.sort_by_key(|&(id, _)| id);
    let contributions: Vec<Vec<bf16>> = slots
        .iter()
        .map(|&(e, w)| oracle_contribution(fx, e, xr, w))
        .collect();
    let gated = oracle_shared(fx, xr);
    let mut y = vec![bf16::from_f32(0.0); h];
    let mut yf = vec![0.0f32; h];
    match combine {
        OracleCombine::AscendingBf16 | OracleCombine::DescendingBf16 => {
            let order: Box<dyn Iterator<Item = &Vec<bf16>>> = match combine {
                OracleCombine::AscendingBf16 => Box::new(contributions.iter()),
                OracleCombine::DescendingBf16 => Box::new(contributions.iter().rev()),
                OracleCombine::Fp32 => unreachable!(),
            };
            for c in order {
                for d in 0..h {
                    y[d] = bf16::from_f32(y[d].to_f32() + c[d].to_f32());
                }
            }
            for d in 0..h {
                y[d] = bf16::from_f32(y[d].to_f32() + gated[d].to_f32());
            }
        }
        OracleCombine::Fp32 => {
            for c in &contributions {
                for d in 0..h {
                    yf[d] += c[d].to_f32();
                }
            }
            for d in 0..h {
                y[d] = bf16::from_f32(yf[d] + gated[d].to_f32());
            }
        }
    }
    y
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

/// Reference-only noise cell on the eviction-arm weight structure: max over
/// the sampled rows of |asc-desc| and |asc-fp32| of the oracle combines.
fn eviction_combine_noise(fx2: &MoeFixture, ids: &[i32], weights: &[bf16], rows: &[usize]) -> f32 {
    let all: Vec<usize> = (0..fx2.dims.top_k).collect();
    let mut noise = 0.0f32;
    for &row in rows {
        let asc = oracle_row_with(fx2, row, ids, weights, &all, OracleCombine::AscendingBf16);
        let desc = oracle_row_with(fx2, row, ids, weights, &all, OracleCombine::DescendingBf16);
        let f32c = oracle_row_with(fx2, row, ids, weights, &all, OracleCombine::Fp32);
        for d in 0..fx2.dims.hidden {
            noise = noise
                .max((asc[d].to_f32() - desc[d].to_f32()).abs())
                .max((asc[d].to_f32() - f32c[d].to_f32()).abs());
        }
    }
    noise
}

#[test]
fn oracle_matches_frozen_reference_on_routed_rows() -> Result<()> {
    let fx = reference::fixture(MoeDims::new(4, 16, 4, 32, 16), 0);
    for row in 0..2 {
        let routed = reference::route_row(&fx, row);
        let all: Vec<usize> = (0..fx.dims.top_k).collect();
        let planted = oracle_row(&fx, row, &routed.ids, &routed.weights, &all);
        let frozen = reference::moe_row(&fx, row);
        assert_eq!(planted, frozen, "oracle must equal reference on row {row}");
        let keep: Vec<usize> = (0..fx.dims.top_k).filter(|&c| c != 1).collect();
        let sub = oracle_row(&fx, row, &routed.ids, &routed.weights, &keep);
        let frozen_sub = reference::moe_row_choices(&fx, row, &keep);
        assert_eq!(
            sub, frozen_sub,
            "oracle subset must equal reference subset on row {row}"
        );
    }
    Ok(())
}

const SKEW_SUBSET: [usize; 16] = [
    3, 17, 42, 77, 100, 137, 150, 171, 190, 201, 220, 231, 240, 247, 250, 255,
];

/// Hot-expert id of the revision-(e) skew and eviction cases.
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

/// One expert takes ~1/3 of all assignments; returns (ids, weights, m_max).
/// Hot-skew at the valid maximum: the hot expert occupies choice 0 of every
/// row (distinct top-k caps a single expert at rows assignments = 1/K of
/// all); other choices are distinct experts.
fn planted_hot(rows: usize, experts: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>, usize) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    let mut counts = vec![0usize; experts];
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
            counts[ids[r * top_k + c] as usize] += 1;
        }
    }
    let m_max = counts.into_iter().max().unwrap();
    (ids, weights, m_max)
}

/// 16 rows at an odd stride, coprime with the power-of-two counts under test.
fn sample_rows(rows: usize) -> Vec<usize> {
    (0..16).map(|j| (403 * j) % rows).collect()
}

fn all_active(rows: usize, top_k: usize) -> Vec<u32> {
    vec![1u32; rows * top_k]
}

struct CaseBufs {
    gens: CudaSlice<u64>,
    x: CudaSlice<bf16>,
    ids: CudaSlice<i32>,
    active: CudaSlice<u32>,
    weights: CudaSlice<bf16>,
    y: CudaSlice<bf16>,
}

struct Arm<'a> {
    label: &'a str,
    rows: usize,
    ids: &'a [i32],
    active: &'a [u32],
    weights: &'a [bf16],
    check: &'a [usize],
    gens: &'a [u64],
    frozen_reference: bool,
}

struct GpuCase<'a> {
    s: &'a Arc<CudaStream>,
    ops: &'a PrefillOps,
    fx: &'a MoeFixture,
    scratch: &'a Scratch,
    table: &'a testing::SlotTable,
    sg: &'a CudaSlice<bf16>,
    su: &'a CudaSlice<bf16>,
    sd: &'a CudaSlice<bf16>,
    seg: &'a CudaSlice<bf16>,
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
        let s = self.s;
        let hidden = self.fx.dims.hidden;
        Ok(CaseBufs {
            x: s.clone_htod(&self.fx.x[..rows * hidden])?,
            ids: s.clone_htod(ids)?,
            active: s.clone_htod(active)?,
            weights: s.clone_htod(weights)?,
            gens: s.clone_htod(gens)?,
            y: s.alloc_zeros(rows * hidden)?,
        })
    }

    fn views(
        &self,
        rows: usize,
        bufs: &CaseBufs,
        table: &testing::SlotTable,
        y: TensorMut<bf16, 2>,
    ) -> Result<MoePrefill> {
        let s = self.s;
        let (experts, top_k, hidden, inter) = (
            self.fx.dims.experts,
            self.fx.dims.top_k,
            self.fx.dims.hidden,
            self.fx.dims.inter,
        );
        Ok(MoePrefill {
            x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
            ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
            active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
            weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
            assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
            slots_gate_up: testing::view(&table.gate_up, s, [experts as u32], [1])?,
            slots_down: testing::view(&table.down, s, [experts as u32], [1])?,
            slot_valid: testing::view(&table.valid, s, [experts as u32], [1])?,
            slot_generation: testing::view(&table.generation, s, [experts as u32], [1])?,
            shared_gate: testing::dense(self.sg, s, inter as u32, hidden as u32)?,
            shared_up: testing::dense(self.su, s, inter as u32, hidden as u32)?,
            shared_down: testing::dense(self.sd, s, hidden as u32, inter as u32)?,
            shared_expert_gate: testing::view(self.seg, s, [hidden as u32], [1])?,
            y,
        })
    }

    /// Runs one arm and compares the active choices of every checked row.
    /// `frozen_reference` uses reference::moe_row_choices (valid only for
    /// route_row ids), otherwise the pinned oracle.
    fn run(&self, arm: Arm<'_>) -> Result<f32> {
        let Arm {
            label,
            rows,
            ids,
            active,
            weights,
            check,
            gens,
            frozen_reference,
        } = arm;
        let s = self.s;
        let (top_k, hidden) = (self.fx.dims.top_k, self.fx.dims.hidden);
        let bufs = self.bufs(rows, ids, active, weights, gens)?;
        let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
        let input = self.views(rows, &bufs, self.table, y_view)?;
        self.ops.prefill(s, input, self.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&bufs.y)?;
        let mut max_diff = 0.0f32;
        for &row in check {
            let keep_row: Vec<usize> = (0..top_k)
                .filter(|&c| active[row * top_k + c] == 1)
                .collect();
            let expected = if frozen_reference {
                reference::moe_row_choices(self.fx, row, &keep_row)
            } else {
                oracle_row(
                    self.fx,
                    row,
                    &ids[row * top_k..(row + 1) * top_k],
                    &weights[row * top_k..(row + 1) * top_k],
                    &keep_row,
                )
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
        let mean_checked: f64 = check.iter().map(|&r| norms[r]).sum::<f64>() / check.len() as f64;
        eprintln!(
            "moe_prefill {label} rows={rows} checked={} max_diff={max_diff:.6} tol={:.6} mean_row_norm={mean_all:.4} mean_checked_norm={mean_checked:.4}",
            check.len(),
            self.tol
        );
        anyhow::ensure!(
            max_diff <= self.tol,
            "moe_prefill {label} exceeded tolerance: {max_diff} > {}",
            self.tol
        );
        anyhow::ensure!(
            rows < 4096 || (0.5..=2.0).contains(&(mean_all / mean_checked)),
            "moe_prefill {label} aggregate row-norm control failed: {mean_all} vs {mean_checked}"
        );
        Ok(max_diff)
    }
}

#[test]
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn moe_prefill_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    if std::env::var("FF_MOE_TRAP_CHILD").as_deref() == Ok("1") {
        return stale_generation_child();
    }
    if std::env::var("FF_MOE_DUP_CHILD").as_deref() == Ok("1") {
        return duplicate_ids_child();
    }
    let mut defects: Vec<String> = Vec::new();
    let ops = PrefillOps::new(&d.ctx, s)?;
    let fx = reference::fixture(MoeDims::minicpm47(4096), 0);
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
    let table = testing::slots(s, experts, gu_ptr, gu_stride, dn_ptr, dn_stride)?;
    let scratch_bytes = prefill_scratch_bytes(4096, top_k, experts, hidden, inter);
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let sg: CudaSlice<bf16> = s.clone_htod(fx.shared_gate.as_slice())?;
    let su: CudaSlice<bf16> = s.clone_htod(fx.shared_up.as_slice())?;
    let sd: CudaSlice<bf16> = s.clone_htod(fx.shared_down.as_slice())?;
    let seg: CudaSlice<bf16> = s.clone_htod(fx.shared_expert_gate.as_slice())?;
    let tolerance_rows: Vec<usize> = (0..8).collect();
    let tol = reference::tolerance(&fx, &tolerance_rows);
    eprintln!(
        "moe_prefill tolerance={tol:.6} over rows {tolerance_rows:?} (2x combine noise; cell-1 pre-GPU value)"
    );
    let case = GpuCase {
        s,
        ops: &ops,
        fx: &fx,
        scratch: &scratch,
        table: &table,
        sg: &sg,
        su: &su,
        sd: &sd,
        seg: &seg,
        tol,
    };

    for rows in [1usize, 37, 512, 4096] {
        // Full-row oracle compares run only at rows <= 37; above that the
        // debug-build host oracle (about 0.1 GFLOP per row) exceeds the GPU
        // window, so checks sample rows and every arm still runs.
        let check: Vec<usize> = if rows <= 37 {
            (0..rows).collect()
        } else {
            sample_rows(rows)
        };
        let active = all_active(rows, top_k);
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        super::route::ensure_distinct_row_experts(&ids, top_k)?;
        let gens = vec![1u64; rows * top_k];
        case.run(Arm {
            label: "uniform",
            rows,
            ids: &ids,
            active: &active,
            weights: &weights,
            check: &check,
            gens: &gens,
            frozen_reference: false,
        })?;
        let (ids, weights) = planted_skewed(rows, top_k);
        super::route::ensure_distinct_row_experts(&ids, top_k)?;
        let gens = vec![1u64; rows * top_k];
        case.run(Arm {
            label: "skewed",
            rows,
            ids: &ids,
            active: &active,
            weights: &weights,
            check: &check,
            gens: &gens,
            frozen_reference: false,
        })?;
        if rows == 4096 {
            let (ids, weights, m_max) = planted_hot(rows, experts, top_k);
            super::route::ensure_distinct_row_experts(&ids, top_k)?;
            eprintln!("moe_prefill hot-skew rows={rows} m_max={m_max}");
            let gens = vec![1u64; rows * top_k];
            case.run(Arm {
                label: "hot",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                frozen_reference: false,
            })?;
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
        super::route::ensure_distinct_row_experts(&ids, top_k)?;
        let gens = vec![1u64; rows * top_k];
        case.run(Arm {
            label: "routed",
            rows,
            ids: &ids,
            active: &active,
            weights: &weights,
            check: &check,
            gens: &gens,
            frozen_reference: rows <= 512,
        })?;
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
        super::route::ensure_distinct_row_experts(&ids, top_k)?;
        let gens = vec![1u64; rows * top_k];
        case.run(Arm {
            label: "partial-active",
            rows,
            ids: &ids,
            active: &active,
            weights: &weights,
            check: &check,
            gens: &gens,
            frozen_reference: false,
        })?;
    }

    // Negative controls at rows=37 (uniform arm covers every expert, so
    // experts 3 and 17 receive rows).
    let (ids, weights) = planted_uniform(37, experts, top_k);
    super::route::ensure_distinct_row_experts(&ids, top_k)?;
    let active = all_active(37, top_k);
    let mut swapped_gu: Vec<u64> = (0..experts as u64)
        .map(|e| gu_ptr + e * gu_stride)
        .collect();
    let mut swapped_dn: Vec<u64> = (0..experts as u64)
        .map(|e| dn_ptr + e * dn_stride)
        .collect();
    swapped_gu.swap(3, 17);
    swapped_dn.swap(3, 17);
    let swapped = testing::SlotTable {
        gate_up: s.clone_htod(swapped_gu.as_slice())?,
        down: s.clone_htod(swapped_dn.as_slice())?,
        valid: s.clone_htod(vec![1u32; experts].as_slice())?,
        generation: s.clone_htod(vec![1u64; experts].as_slice())?,
    };
    let bad = |table: &testing::SlotTable| -> Result<f32> {
        let rows = 37usize;
        let gens = vec![1u64; rows * top_k];
        let bufs = case.bufs(rows, &ids, &active, &weights, &gens)?;
        let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
        let input = case.views(rows, &bufs, table, y_view)?;
        case.ops.prefill(s, input, case.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&bufs.y)?;
        let all: Vec<usize> = (0..case.fx.dims.top_k).collect();
        let mut max_diff = 0.0f32;
        for row in 0..rows {
            let expected = oracle_row(
                case.fx,
                row,
                &ids[row * top_k..(row + 1) * top_k],
                &weights[row * top_k..(row + 1) * top_k],
                &all,
            );
            for c in 0..hidden {
                max_diff =
                    max_diff.max((y[row * hidden + c].to_f32() - expected[c].to_f32()).abs());
            }
        }
        Ok(max_diff)
    };
    let diff = bad(&swapped)?;
    anyhow::ensure!(
        diff > tol,
        "swapped slot-table control was not detected: {diff} <= {tol}"
    );
    eprintln!(
        "moe_prefill negative-control swapped-slots max_diff={diff:.6} tol={tol:.6} DETECTED"
    );

    // Gate/up split off-by-one: every expert block of gate_up rolled by one row.
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
    let rolled = testing::slots(s, experts, rolled_ptr, gu_stride, dn_ptr, dn_stride)?;
    let diff = bad(&rolled)?;
    anyhow::ensure!(
        diff > tol,
        "rolled gate/up split control was not detected: {diff} <= {tol}"
    );
    eprintln!(
        "moe_prefill negative-control rolled-gate-up max_diff={diff:.6} tol={tol:.6} DETECTED"
    );

    // Noise cell-2 (revision (d), reference-only, unscored): GEMV decode vs
    // grouped-GEMM prefill on the same fixture rows, full-active generation 1.
    {
        let rows = 8usize;
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let bufs = case.bufs(rows, &ids, &active, &weights, &gens)?;
        let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
        let prefill_input = case.views(rows, &bufs, &table, y_view)?;
        case.ops.prefill(s, prefill_input, case.scratch)?;
        let dec_y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
        let mut dec_y_view = testing::dense_mut(&dec_y, s, rows as u32, hidden as u32)?;
        let decode_ops = DecodeOps::new(&d.ctx)?;
        let decode = MoeDecode {
            x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
            ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
            active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
            weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
            assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
            slots_gate_up: testing::view(&table.gate_up, s, [experts as u32], [1])?,
            slots_down: testing::view(&table.down, s, [experts as u32], [1])?,
            slot_valid: testing::view(&table.valid, s, [experts as u32], [1])?,
            slot_generation: testing::view(&table.generation, s, [experts as u32], [1])?,
            shared_gate: testing::dense(&sg, s, inter as u32, hidden as u32)?,
            shared_up: testing::dense(&su, s, inter as u32, hidden as u32)?,
            shared_down: testing::dense(&sd, s, hidden as u32, inter as u32)?,
            shared_expert_gate: testing::view(&seg, s, [hidden as u32], [1])?,
            y: &mut dec_y_view,
        };
        anyhow::ensure!(
            scratch.bytes >= decode_scratch_bytes(rows, top_k, inter),
            "prefill scratch below decode cell-2 requirement"
        );
        decode_ops.decode(s, decode, &scratch)?;
        s.synchronize()?;
        let y_pre = s.clone_dtoh(&bufs.y)?;
        let y_dec = s.clone_dtoh(&dec_y)?;
        let mut spread = 0.0f32;
        for v in y_pre.iter().zip(&y_dec) {
            spread = spread.max((v.0.to_f32() - v.1.to_f32()).abs());
        }
        eprintln!(
            "moe_prefill noise cell-2 gemv-vs-grouped-gemm rows={rows} max_spread={spread:.6} (reference-only, unscored)"
        );
    }

    // Informational shape cell (Correction 4 residual): decode-op vs
    // prefill-op outputs on the eviction weight structure, unscored; feeds
    // any Correction 5, never this run's verdict.
    {
        let rows = 8usize;
        let (fx2, ids, weights) = eviction_fixture(&fx, rows);
        super::route::ensure_distinct_row_experts(&ids, top_k)?;
        let (h, i) = (fx.dims.hidden, fx.dims.inter);
        let gu_base = HOT_EXPERT * 2 * i * h;
        let hot_gu: CudaSlice<bf16> = s.clone_htod(&fx2.gate_up[gu_base..gu_base + 2 * i * h])?;
        let dn_base = HOT_EXPERT * h * i;
        let hot_dn: CudaSlice<bf16> = s.clone_htod(&fx2.down[dn_base..dn_base + h * i])?;
        let (hot_gu_ptr, _cg) = hot_gu.device_ptr(s);
        let (hot_dn_ptr, _cd) = hot_dn.device_ptr(s);
        let ev1_gu: Vec<u64> = (0..experts as u64)
            .map(|e| gu_ptr + e * gu_stride)
            .collect::<Vec<_>>()
            .into_iter()
            .enumerate()
            .map(|(e, p)| if e == HOT_EXPERT { hot_gu_ptr } else { p })
            .collect();
        let ev1_dn: Vec<u64> = (0..experts as u64)
            .map(|e| dn_ptr + e * dn_stride)
            .collect::<Vec<_>>()
            .into_iter()
            .enumerate()
            .map(|(e, p)| if e == HOT_EXPERT { hot_dn_ptr } else { p })
            .collect();
        let shape_table = testing::SlotTable {
            gate_up: s.clone_htod(ev1_gu.as_slice())?,
            down: s.clone_htod(ev1_dn.as_slice())?,
            valid: s.clone_htod(vec![1u32; experts].as_slice())?,
            generation: s.clone_htod(vec![1u64; experts].as_slice())?,
        };
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let bufs = case.bufs(rows, &ids, &active, &weights, &gens)?;
        let y_view = testing::dense_mut(&bufs.y, s, rows as u32, h as u32)?;
        let prefill_input = case.views(rows, &bufs, &shape_table, y_view)?;
        case.ops.prefill(s, prefill_input, case.scratch)?;
        let dec_y: CudaSlice<bf16> = s.alloc_zeros(rows * h)?;
        let mut dec_y_view = testing::dense_mut(&dec_y, s, rows as u32, h as u32)?;
        let decode_ops = DecodeOps::new(&d.ctx)?;
        let decode = MoeDecode {
            x: testing::dense(&bufs.x, s, rows as u32, h as u32)?,
            ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
            active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
            weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
            assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
            slots_gate_up: testing::view(&shape_table.gate_up, s, [experts as u32], [1])?,
            slots_down: testing::view(&shape_table.down, s, [experts as u32], [1])?,
            slot_valid: testing::view(&shape_table.valid, s, [experts as u32], [1])?,
            slot_generation: testing::view(&shape_table.generation, s, [experts as u32], [1])?,
            shared_gate: testing::dense(&sg, s, i as u32, h as u32)?,
            shared_up: testing::dense(&su, s, i as u32, h as u32)?,
            shared_down: testing::dense(&sd, s, h as u32, i as u32)?,
            shared_expert_gate: testing::view(&seg, s, [h as u32], [1])?,
            y: &mut dec_y_view,
        };
        anyhow::ensure!(
            scratch.bytes >= decode_scratch_bytes(rows, top_k, i),
            "prefill scratch below decode shape-cell requirement"
        );
        decode_ops.decode(s, decode, &scratch)?;
        s.synchronize()?;
        let y_pre = s.clone_dtoh(&bufs.y)?;
        let y_dec = s.clone_dtoh(&dec_y)?;
        let mut spread = 0.0f32;
        for v in y_pre.iter().zip(&y_dec) {
            spread = spread.max((v.0.to_f32() - v.1.to_f32()).abs());
        }
        eprintln!(
            "moe_prefill shape cell eviction gemv-vs-grouped-gemm rows={rows} max_spread={spread:.6} (reference-only, unscored)"
        );
    }

    // Eviction-reuse control: all assignments route to the hot expert, whose
    // weights move to a new device buffer scaled by 1.25 (bf16 round) with
    // the slot generation bumped to 2; the relaunch at expected_generation=2
    // must read the replaced weights. Eight same-expert contributions put
    // |y| near 10.7; the tolerance is twice the reference-only noise cell
    // measured on this weight structure (oracle asc/desc/fp32 combines).
    {
        let rows = 37usize;
        let (fx2, ids, weights) = eviction_fixture(&fx, rows);
        super::route::ensure_distinct_row_experts(&ids, top_k)?;
        let tol_evict = 2.0 * eviction_combine_noise(&fx2, &ids, &weights, &[0, 1, 2, 3, 4]);
        eprintln!(
            "moe_prefill eviction noise cell rows=[0,1]: max_spread={} tol_evict={tol_evict:.6}",
            tol_evict / 2.0
        );
        let case = GpuCase {
            s: case.s,
            ops: case.ops,
            fx: case.fx,
            scratch: case.scratch,
            table: case.table,
            sg: case.sg,
            su: case.su,
            sd: case.sd,
            seg: case.seg,
            tol: tol_evict,
        };
        let check: Vec<usize> = (0..rows).collect();
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(Arm {
            label: "eviction-before",
            rows,
            ids: &ids,
            active: &active,
            weights: &weights,
            check: &check,
            gens: &gens,
            frozen_reference: false,
        })?;
        let scale_block = |src: &[bf16], e: usize, rows_n: usize, width: usize| -> Vec<bf16> {
            let mut out = src[e * rows_n * width..(e + 1) * rows_n * width].to_vec();
            for v in &mut out {
                *v = bf16::from_f32(v.to_f32() * 1.25);
            }
            out
        };
        let hot_gu = scale_block(&fx.gate_up, HOT_EXPERT, 2 * inter, hidden);
        let hot_dn = scale_block(&fx.down, HOT_EXPERT, hidden, inter);
        let hot_gu_buf: CudaSlice<bf16> = s.clone_htod(hot_gu.as_slice())?;
        let hot_dn_buf: CudaSlice<bf16> = s.clone_htod(hot_dn.as_slice())?;
        let (hot_gu_ptr, _h1) = hot_gu_buf.device_ptr(s);
        let (hot_dn_ptr, _h2) = hot_dn_buf.device_ptr(s);
        let mut ev_gu: Vec<u64> = (0..experts as u64)
            .map(|e| gu_ptr + e * gu_stride)
            .collect();
        let mut ev_dn: Vec<u64> = (0..experts as u64)
            .map(|e| dn_ptr + e * dn_stride)
            .collect();
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
        let evicted = testing::SlotTable {
            gate_up: s.clone_htod(ev_gu.as_slice())?,
            down: s.clone_htod(ev_dn.as_slice())?,
            valid: s.clone_htod(vec![1u32; experts].as_slice())?,
            generation: s.clone_htod(ev_gen.as_slice())?,
        };
        // The eviction fixture differs from the original only in the hot
        // expert's blocks; routing and x are identical by construction
        // (same variant, prefix rows).
        let mut max_diff = 0.0f32;
        let mut old_diff = 0.0f32;
        let bufs = case.bufs(rows, &ids, &active, &weights, &snapshot)?;
        let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
        let input = case.views(rows, &bufs, &evicted, y_view)?;
        case.ops.prefill(s, input, case.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&bufs.y)?;
        let all: Vec<usize> = (0..top_k).collect();
        let mut row_max = vec![0.0f32; rows];
        let mut worst = (0usize, 0usize);
        for row in 0..rows {
            let expected = oracle_row(
                &fx2,
                row,
                &ids[row * top_k..(row + 1) * top_k],
                &weights[row * top_k..(row + 1) * top_k],
                &all,
            );
            let expected_old = oracle_row(
                &fx,
                row,
                &ids[row * top_k..(row + 1) * top_k],
                &weights[row * top_k..(row + 1) * top_k],
                &all,
            );
            for c in 0..hidden {
                let d = (y[row * hidden + c].to_f32() - expected[c].to_f32()).abs();
                if d > row_max[row] {
                    row_max[row] = d;
                    if d > max_diff {
                        max_diff = d;
                        worst = (row, c);
                    }
                }
                old_diff =
                    old_diff.max((y[row * hidden + c].to_f32() - expected_old[c].to_f32()).abs());
            }
        }
        let (wr, wc) = worst;
        let expected_w = oracle_row(
            &fx2,
            wr,
            &ids[wr * top_k..(wr + 1) * top_k],
            &weights[wr * top_k..(wr + 1) * top_k],
            &all,
        );
        eprintln!(
            "moe_prefill eviction-slice rows={rows} row_max[min,mean,max]={:.6},{:.6},{:.6} worst=({wr},{wc}) cand={:.6} oracle_new={:.6}",
            row_max.iter().copied().fold(f32::INFINITY, f32::min),
            row_max.iter().sum::<f32>() / rows as f32,
            row_max.iter().copied().fold(0.0f32, f32::max),
            y[wr * hidden + wc].to_f32(),
            expected_w[wc].to_f32()
        );
        // Stale-slot discriminator: a candidate matching the unscaled
        // reference reads old weights (P1); matching neither is a tolerance
        // question.
        eprintln!(
            "moe_prefill eviction-reuse new-weight max_diff={max_diff:.6} old-weight max_diff={old_diff:.6} tol={tol_evict:.6} verdict={}",
            if old_diff <= tol_evict && max_diff > tol_evict {
                "MATCHES_OLD_WEIGHTS_STALE_SLOT_P1"
            } else if max_diff <= tol_evict {
                "MATCHES_NEW_WEIGHTS"
            } else {
                "MATCHES_NEITHER"
            }
        );
        eprintln!(
            "moe_prefill eviction-reuse rows={rows} expert={HOT_EXPERT} generation=2 max_diff={max_diff:.6} tol={tol_evict:.6}"
        );
        if max_diff > tol_evict {
            // A DEFECT is a recorded result; the run continues so the
            // reference-only cells and timing sections still execute.
            eprintln!(
                "moe_prefill DEFECT eviction-reuse exceeded tolerance: {max_diff} > {tol_evict}"
            );
            defects.push(format!(
                "eviction-reuse {max_diff:.6} > tol_evict {tol_evict:.6}"
            ));
        }
    }

    // DEFECT-5 discriminator: per-choice isolation on the eviction-reuse
    // inputs. Each launch activates exactly one choice of every row; the
    // output minus the shared term is that choice's contribution.
    {
        let rows = 37usize;
        let (fx2, ids, weights) = eviction_fixture(&fx, rows);
        let mut ev_gen = vec![1u64; experts];
        ev_gen[HOT_EXPERT] = 2;
        let mut snapshot = vec![1u64; rows * top_k];
        for r in 0..rows {
            snapshot[r * top_k] = 2;
        }
        let gu_base2 = HOT_EXPERT * 2 * inter * hidden;
        let hot_gu2 = fx2.gate_up[gu_base2..gu_base2 + 2 * inter * hidden].to_vec();
        let dn_base2 = HOT_EXPERT * hidden * inter;
        let hot_dn2 = fx2.down[dn_base2..dn_base2 + hidden * inter].to_vec();
        let hot_gu2_buf: CudaSlice<bf16> = s.clone_htod(&hot_gu2)?;
        let hot_dn2_buf: CudaSlice<bf16> = s.clone_htod(&hot_dn2)?;
        let (hot_gu2_ptr, _h1) = hot_gu2_buf.device_ptr(s);
        let (hot_dn2_ptr, _h2) = hot_dn2_buf.device_ptr(s);
        let mut iso_gu: Vec<u64> = (0..experts as u64)
            .map(|e| gu_ptr + e * gu_stride)
            .collect();
        let mut iso_dn: Vec<u64> = (0..experts as u64)
            .map(|e| dn_ptr + e * dn_stride)
            .collect();
        iso_gu[HOT_EXPERT] = hot_gu2_ptr;
        iso_dn[HOT_EXPERT] = hot_dn2_ptr;
        let evicted = testing::SlotTable {
            gate_up: s.clone_htod(iso_gu.as_slice())?,
            down: s.clone_htod(iso_dn.as_slice())?,
            valid: s.clone_htod(vec![1u32; experts].as_slice())?,
            generation: s.clone_htod(ev_gen.as_slice())?,
        };
        for c in 0..top_k {
            let mut active = vec![0u32; rows * top_k];
            for r in 0..rows {
                active[r * top_k + c] = 1;
            }
            let bufs = case.bufs(rows, &ids, &active, &weights, &snapshot)?;
            let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
            let input = case.views(rows, &bufs, &evicted, y_view)?;
            case.ops.prefill(s, input, case.scratch)?;
            s.synchronize()?;
            let y = s.clone_dtoh(&bufs.y)?;
            let mut max_diff = 0.0f32;
            let mut worst = (0usize, 0usize);
            for r in 0..rows {
                let e = ids[r * top_k + c] as usize;
                let w = weights[r * top_k + c];
                let xr = &fx2.x[r * hidden..(r + 1) * hidden];
                let contribution = oracle_contribution(&fx2, e, xr, w);
                let shared = oracle_shared(&fx2, xr);
                for d in 0..hidden {
                    let diff = (y[r * hidden + d].to_f32()
                        - (contribution[d].to_f32() + shared[d].to_f32()))
                    .abs();
                    if diff > max_diff {
                        max_diff = diff;
                        worst = (r, d);
                    }
                }
            }
            eprintln!(
                "moe_prefill evict-choice c={c} expert={} max_diff={max_diff:.6} worst_row={} worst_col={}",
                ids[c], worst.0, worst.1
            );
        }
    }

    if std::env::var("FF_MOE_TIMING").as_deref() == Ok("1") {
        timing(&case)?;
    }
    anyhow::ensure!(defects.is_empty(), "recorded defects: {defects:?}");
    Ok(())
}

/// Stale-generation arm of the re-exec'd child (FF_MOE_TRAP_CHILD=1): a small
/// fixture whose only slot generation (1) mismatches the launch's
/// expected_generation (3) on an active assignment. Exits non-zero whether
/// the trap fires (exit 3) or is missed (test failure).
fn stale_generation_child() -> Result<()> {
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
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _a) = gu.device_ptr(s);
    let (dn_ptr, _b) = dn.device_ptr(s);
    let table = testing::slots(
        s,
        experts,
        gu_ptr,
        (2 * inter * hidden * 2) as u64,
        dn_ptr,
        (hidden * inter * 2) as u64,
    )?;
    let sg: CudaSlice<bf16> = s.clone_htod(fx.shared_gate.as_slice())?;
    let su: CudaSlice<bf16> = s.clone_htod(fx.shared_up.as_slice())?;
    let sd: CudaSlice<bf16> = s.clone_htod(fx.shared_down.as_slice())?;
    let seg: CudaSlice<bf16> = s.clone_htod(fx.shared_expert_gate.as_slice())?;
    let scratch_bytes = prefill_scratch_bytes(8, top_k, experts, hidden, inter);
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _c) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    super::route::ensure_distinct_row_experts(&ids, top_k)?;
    let active = all_active(rows, top_k);
    let x: CudaSlice<bf16> = s.clone_htod(fx.x.as_slice())?;
    let ids_buf: CudaSlice<i32> = s.clone_htod(&ids)?;
    let active_buf: CudaSlice<u32> = s.clone_htod(&active)?;
    let weights_buf: CudaSlice<bf16> = s.clone_htod(&weights)?;
    // Replaced slot advances to generation 2; every assignment carries its
    // slot's snapshot except assignment 0, which still holds the OLD
    // generation of the replaced slot and must trip the guard.
    let mut table_gen = vec![1u64; experts];
    table_gen[ids[0] as usize] = 2;
    let mut snapshot = vec![1u64; rows * top_k];
    for r in 0..rows {
        for c in 0..top_k {
            snapshot[r * top_k + c] = table_gen[ids[r * top_k + c] as usize];
        }
    }
    snapshot[0] = 1;
    let table_gen_buf: CudaSlice<u64> = s.clone_htod(&table_gen)?;
    let gens_buf: CudaSlice<u64> = s.clone_htod(&snapshot)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = MoePrefill {
        x: testing::dense(&x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&ids_buf, s, rows as u32, top_k as u32)?,
        active: testing::dense(&active_buf, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&weights_buf, s, rows as u32, top_k as u32)?,
        assignment_gen: testing::dense(&gens_buf, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&table.gate_up, s, [experts as u32], [1])?,
        slots_down: testing::view(&table.down, s, [experts as u32], [1])?,
        slot_valid: testing::view(&table.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&table_gen_buf, s, [experts as u32], [1])?,
        shared_gate: testing::dense(&sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: testing::view(&seg, s, [hidden as u32], [1])?,
        y: y_view,
    };
    let outcome = ops.prefill(s, input, &scratch);
    eprintln!("moe_prefill stale-generation child outcome: {outcome:?}");
    match outcome {
        Ok(()) => anyhow::bail!("stale generation did not trap"),
        Err(_) => std::process::exit(3),
    }
}

/// Duplicate expert ids within a row are unreachable router outputs; the
/// permute guard must trap. Runs in a child process (FF_MOE_DUP_CHILD).
fn duplicate_ids_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s)?;
    let fx = reference::fixture(MoeDims::new(2, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _a) = gu.device_ptr(s);
    let (dn_ptr, _b) = dn.device_ptr(s);
    let table = testing::slots(
        s,
        experts,
        gu_ptr,
        (2 * inter * hidden * 2) as u64,
        dn_ptr,
        (hidden * inter * 2) as u64,
    )?;
    let sg: CudaSlice<bf16> = s.clone_htod(fx.shared_gate.as_slice())?;
    let su: CudaSlice<bf16> = s.clone_htod(fx.shared_up.as_slice())?;
    let sd: CudaSlice<bf16> = s.clone_htod(fx.shared_down.as_slice())?;
    let seg: CudaSlice<bf16> = s.clone_htod(fx.shared_expert_gate.as_slice())?;
    let scratch_bytes = prefill_scratch_bytes(2, top_k, experts, hidden, inter);
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _c) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let mut dup = ids.clone();
    dup[1] = dup[0];
    let active = all_active(rows, top_k);
    let x: CudaSlice<bf16> = s.clone_htod(fx.x.as_slice())?;
    let ids_buf: CudaSlice<i32> = s.clone_htod(&dup)?;
    let active_buf: CudaSlice<u32> = s.clone_htod(&active)?;
    let weights_buf: CudaSlice<bf16> = s.clone_htod(&weights)?;
    let gens = vec![1u64; rows * top_k];
    let gens_buf: CudaSlice<u64> = s.clone_htod(&gens)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = MoePrefill {
        x: testing::dense(&x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&ids_buf, s, rows as u32, top_k as u32)?,
        active: testing::dense(&active_buf, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&weights_buf, s, rows as u32, top_k as u32)?,
        assignment_gen: testing::dense(&gens_buf, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&table.gate_up, s, [experts as u32], [1])?,
        slots_down: testing::view(&table.down, s, [experts as u32], [1])?,
        slot_valid: testing::view(&table.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&table.generation, s, [experts as u32], [1])?,
        shared_gate: testing::dense(&sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: testing::view(&seg, s, [hidden as u32], [1])?,
        y: y_view,
    };
    let outcome = ops.prefill(s, input, &scratch);
    eprintln!("moe_prefill duplicate-ids child outcome: {outcome:?}");
    match outcome {
        Ok(()) => anyhow::bail!("duplicate ids did not trap"),
        Err(_) => std::process::exit(3),
    }
}

fn timing(case: &GpuCase<'_>) -> Result<()> {
    let s = case.s;
    let (experts, top_k, hidden) = (
        case.fx.dims.experts,
        case.fx.dims.top_k,
        case.fx.dims.hidden,
    );
    const WARMUP: usize = 3;
    const ITERS: usize = 20;
    for rows in [512usize, 4096] {
        let active = all_active(rows, top_k);
        let (u_ids, u_w) = planted_uniform(rows, experts, top_k);
        let (s_ids, s_w) = planted_skewed(rows, top_k);
        let (h_ids, h_w, m_max) = planted_hot(rows, experts, top_k);
        let arms: Vec<(&str, Vec<i32>, Vec<bf16>, usize)> = vec![
            ("uniform", u_ids, u_w, 0),
            ("skewed", s_ids, s_w, 0),
            ("hot", h_ids, h_w, m_max),
        ];
        for (label, ids, weights, m_max) in arms {
            let gens = vec![1u64; rows * top_k];
            let bufs = case.bufs(rows, &ids, &active, &weights, &gens)?;
            for _ in 0..WARMUP {
                let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
                let input = case.views(rows, &bufs, case.table, y_view)?;
                case.ops.prefill(s, input, case.scratch)?;
            }
            let mut stage_timing = StageTiming::new(s)?;
            let mut stage_sum = vec![0f64; STAGES.len()];
            let (mut total, mut sync) = (0f64, 0f64);
            for _ in 0..ITERS {
                let y_view = testing::dense_mut(&bufs.y, s, rows as u32, hidden as u32)?;
                let input = case.views(rows, &bufs, case.table, y_view)?;
                case.ops
                    .run(s, input, case.scratch, Some(&mut stage_timing))?;
                for (i, sum) in stage_sum.iter_mut().enumerate() {
                    *sum += stage_timing.stage_us(i)? as f64;
                }
                total += stage_timing.total_us()? as f64;
                sync += stage_timing.sync_us;
            }
            let flop = rows as f64
                * top_k as f64
                * 2.0
                * (2.0 * case.fx.dims.inter as f64 * hidden as f64
                    + hidden as f64 * case.fx.dims.inter as f64)
                + rows as f64 * 6.0 * case.fx.dims.inter as f64 * hidden as f64;
            let tflops = flop / (total / ITERS as f64 * 1e-6) / 1e12;
            eprintln!(
                "moe_prefill timing {label} rows={rows} m_max={m_max} iters={ITERS} stages_us={:?} count_sync_us={:.1} e2e_us={:.1} tflops={tflops:.1}",
                (0..STAGES.len())
                    .map(|i| (STAGES[i], stage_sum[i] / ITERS as f64))
                    .collect::<Vec<_>>(),
                sync / ITERS as f64,
                total / ITERS as f64
            );
        }
    }
    Ok(())
}

// Same isolation as decode_stale_generation_gpu: parent makes no CUDA call.
#[test]
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn prefill_stale_generation_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_MOE_GPU_WINDOW", "granted")
        .env("FF_MOE_TRAP_CHILD", "1")
        .env_remove("FF_MOE_TIMING")
        .status()?;
    anyhow::ensure!(
        !status.success(),
        "stale-generation control child exited successfully"
    );
    println!(
        "moe_prefill negative-control stale-generation child exit={:?} DETECTED",
        status.code()
    );
    Ok(())
}

/// Freeze receipt for the eviction-arm noise cell (correction 4): the
/// reference-only combine-order spread on the eviction weight structure.
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

// Parent for duplicate_ids_child; no CUDA call in the parent.
#[test]
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn prefill_duplicate_ids_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_MOE_GPU_WINDOW", "granted")
        .env("FF_MOE_DUP_CHILD", "1")
        .env_remove("FF_MOE_TIMING")
        .status()?;
    anyhow::ensure!(!status.success(), "duplicate-ids child exited successfully");
    println!(
        "moe_prefill duplicate-ids child exit={:?} TRAPPED",
        status.code()
    );
    Ok(())
}
