//! Route-op gates: reference layout checks and the GPU window gate for
//! kernels/moe_route.cu. Default cargo test never initializes a GPU here.
use super::reference::{self, MoeDims, RouteRow};
use super::route::RouteOps;
use super::testing;
use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::TensorMut;
use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use std::collections::HashMap;

/// Exact-id compare gate. Rows at or below the threshold skip the exact-id
/// compare. Variant 0 (E=256/K=8/H=2048) margins span an exact cut tie on
/// row 0 and 2.52e-5..1.19e-3 elsewhere. BF16 products are exact in FP32 and
/// the kernel accumulates sequentially in index order, so logits reproduce
/// the reference bitwise and prob noise is expf/sum rounding (<= ~1e-6
/// relative at p_cut <= 1/8).
const CUT_MARGIN_THRESHOLD: f32 = 1e-5;

const ROW_CASES: [u32; 7] = [1, 2, 5, 8, 37, 512, 4096];
/// Rows compared against the reference per case; every case also checks
/// fixture row 0 (exact-tie row) and row 1 (cut-race row) when in range.
fn checked_rows(n: u32) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..n.min(8)).map(|i| i as usize).collect();
    for &extra in &[36u32, 255, 256, 511, 1024, 2047, 3000, 4095] {
        if extra < n {
            rows.push(extra as usize);
        }
    }
    rows
}

#[test]
fn route_row_layout_matches_reference() -> Result<()> {
    let dims = MoeDims::new(4, 16, 4, 128, 64);
    let fx = reference::fixture(dims, 0);
    let (e, k) = (dims.experts, dims.top_k);
    for row in 0..dims.rows {
        let routed = reference::route_row(&fx, row);
        ensure!(
            routed.ids.len() == k && routed.weights.len() == k && routed.probs.len() == e,
            "route_row output extents mismatch"
        );
        for j in 1..k {
            let (a, b) = (routed.ids[j - 1] as usize, routed.ids[j] as usize);
            ensure!(
                routed.probs[a] > routed.probs[b] || (routed.probs[a] == routed.probs[b] && a < b),
                "route_row order must descend with exact ties by lower id"
            );
        }
        let sum: f32 = routed.weights.iter().map(|w| w.to_f32()).sum();
        ensure!(
            (sum - 1.0).abs() <= k as f32 * 2.0f32.powi(-8),
            "renormalised weights must sum to 1 within BF16 rounding"
        );
    }
    let tie = reference::route_row(&fx, fx.exact_tie_row);
    let (a, b) = fx.exact_tie_experts;
    ensure!(
        tie.probs[a].to_bits() == tie.probs[b].to_bits(),
        "exact-tie pair must produce bitwise-equal probabilities"
    );
    ensure!(
        tie.ids[0] as usize == a && tie.ids[1] as usize == b,
        "exact-tie pair must rank inside the top-k with the lower id first"
    );
    Ok(())
}

struct CaseOutputs<'a> {
    dims: MoeDims,
    ids: &'a [i32],
    weights: &'a [bf16],
}

fn compare_case(
    case: u32,
    refs: &HashMap<usize, RouteRow>,
    rows: &[usize],
    out: &CaseOutputs<'_>,
    tol: f32,
) -> Result<()> {
    let k = out.dims.top_k;
    let mut strict = 0;
    let mut gated = 0;
    for &row in rows {
        let want = &refs[&row];
        let ids = &out.ids[row * k..(row + 1) * k];
        let weights = &out.weights[row * k..(row + 1) * k];
        ensure!(
            ids.iter().all(|&id| id >= 0) && weights.iter().all(|w| w.to_f32() != -17.0),
            "route case {case} row {row} left sentinel outputs"
        );
        let cut = want.probs[want.ids[k - 1] as usize];
        if want.margin > CUT_MARGIN_THRESHOLD {
            strict += 1;
            ensure!(
                ids == &want.ids[..],
                "route case {case} row {row} ids {:?} != reference {:?}",
                ids,
                &want.ids[..]
            );
        } else {
            gated += 1;
        }
        eprintln!(
            "MOE_ROUTE_MARGIN case={case} row={row} margin={:e} p_cut={:e} strict={}",
            want.margin,
            cut,
            want.margin > CUT_MARGIN_THRESHOLD
        );
        for (j, (got, &expected)) in weights.iter().zip(&want.weights).enumerate() {
            let (got, expected) = (got.to_f32(), expected.to_f32());
            ensure!(
                (got - expected).abs() <= tol,
                "route case {case} row {row} weight {j} {got:e} vs {expected:e} (tol {tol:e})"
            );
        }
    }
    eprintln!(
        "MOE_ROUTE_CASE rows={case} checked={} strict_ids={strict} gated_ids={gated} PASS",
        rows.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn moe_route_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    let s = &d.stream;
    let ops = RouteOps::new(&d.ctx)?;
    let dims = MoeDims::minicpm47(4096);
    let (e, k, h) = (dims.experts, dims.top_k, dims.hidden);
    let fx = reference::fixture(dims, 0);
    let refs: HashMap<usize, RouteRow> = checked_rows(4096)
        .into_iter()
        .map(|row| (row, reference::route_row(&fx, row)))
        .collect();
    let tol = reference::tolerance(&fx, &[0, 1, 37]);

    let xb = testing::upload(&d.upload, fx.x.clone())?;
    let mut rb = testing::upload(&d.upload, fx.router_w.clone())?;
    d.finish_upload()?;
    let mut idsb = testing::zeros::<i32>(s, dims.rows * k)?;
    let mut wb = testing::zeros::<bf16>(s, dims.rows * k)?;

    let run = |rows: u32,
               xb: &CudaSlice<bf16>,
               rb: &CudaSlice<bf16>,
               idsb: &mut CudaSlice<i32>,
               wb: &mut CudaSlice<bf16>|
     -> Result<()> {
        let x = testing::dense(xb, s, dims.rows as u32, h as u32)?.prefix_rows(rows)?;
        let router = testing::dense(rb, s, e as u32, h as u32)?;
        let mut ids = TensorMut(
            testing::dense_mut(idsb, s, dims.rows as u32, k as u32)?
                .0
                .prefix_rows(rows)?,
        );
        let mut weights = TensorMut(
            testing::dense_mut(wb, s, dims.rows as u32, k as u32)?
                .0
                .prefix_rows(rows)?,
        );
        ops.route(s, x, router, &mut ids, &mut weights)
    };

    for &case in &ROW_CASES {
        s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
        s.memcpy_htod(&vec![bf16::from_f32(-17.0); dims.rows * k], &mut wb)?;
        run(case, &xb, &rb, &mut idsb, &mut wb)?;
        s.synchronize()?;
        let rows = checked_rows(case);
        let ids = s.clone_dtoh(&idsb)?;
        let weights = s.clone_dtoh(&wb)?;
        let (a, b) = fx.exact_tie_experts;
        let first = &ids[0..k];
        ensure!(
            first[0] as usize == a && first[1] as usize == b,
            "exact-tie row must return both tie experts with the lower id first: {first:?} vs ({a},{b})"
        );
        compare_case(
            case,
            &refs,
            &rows,
            &CaseOutputs {
                dims,
                ids: &ids,
                weights: &weights,
            },
            tol,
        )?;
    }

    // Negative control: perturbing one router row must change routed ids.
    let mut tampered = fx.router_w.clone();
    for (slot, &value) in tampered[..h].iter_mut().zip(&fx.x) {
        *slot = bf16::from_f32(value.to_f32() * 0.9);
    }
    s.memcpy_htod(&tampered, &mut rb)?;
    s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
    run(1, &xb, &rb, &mut idsb, &mut wb)?;
    s.synchronize()?;
    let ids = s.clone_dtoh(&idsb)?;
    let baseline = reference::route_row(&fx, 0);
    ensure!(
        ids[0] == 0,
        "perturbed router row 0 must take rank 0 of the tie row"
    );
    let changed = (0..k).filter(|&j| ids[j] != baseline.ids[j]).count();
    ensure!(
        changed > 0,
        "router perturbation control detected no id change"
    );
    eprintln!(
        "MOE_ROUTE_CONTROL perturbed_ids={:?} changed={changed} PASS",
        &ids[..k]
    );
    Ok(())
}
