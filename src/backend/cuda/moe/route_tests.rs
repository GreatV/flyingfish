//! Route-op gates: CPU view validation, reference layout checks and the GPU
//! window gate for kernels/moe_route.cu. Default cargo test never initializes
//! a GPU here.
use super::reference::{self, MoeDims, RouteRow};
use super::route::{self, RouteOps};
use super::testing;
use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::{Tensor, TensorMut};
use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use half::bf16;
use std::collections::HashMap;
use std::mem::size_of;

/// Exact-id compare gate. Frozen on CPU before the GPU run from the fixture
/// margins of the checked rows (variant 0, E=256/K=8/H=2048): margins span an
/// exact cut tie on row 0 and 2.52e-5..1.19e-3 elsewhere. BF16 products are
/// exact in FP32 and the kernel accumulates sequentially in index order, so
/// logits reproduce the reference bitwise and prob noise is expf/sum rounding
/// (<= ~1e-6 relative at p_cut <= 1/8). Rows at or below the threshold skip
/// the exact-id compare.
const CUT_MARGIN_THRESHOLD: f32 = 1e-5;
/// |gpu - cpu| <= PROBS_REL_TOL * cpu + PROBS_ABS_FLOOR; the floor covers
/// expf underflow-to-zero differences.
const PROBS_REL_TOL: f32 = 1e-5;
const PROBS_ABS_FLOOR: f32 = 1e-30;

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

fn view<T: Copy>(ptr: u64, shape: [u32; 2], stride: [u64; 2]) -> Result<Tensor<T, 2>> {
    let span = ((u64::from(shape[0] - 1)) * stride[0] + (u64::from(shape[1] - 1)) * stride[1] + 1)
        * size_of::<T>() as u64;
    unsafe { Tensor::new(ptr, shape, stride, span) }
}

#[test]
fn route_validation_rejects_bad_views() -> Result<()> {
    let base = 0x10000u64;
    let ids = view::<i32>(base + 0x2000, [5, 4], [4, 1])?;
    let weights = view::<bf16>(base + 0x3000, [5, 4], [4, 1])?;
    let probs = view::<f32>(base + 0x4000, [5, 16], [16, 1])?;
    let good = |x: &Tensor<bf16, 2>,
                router: &Tensor<bf16, 2>,
                ids: &Tensor<i32, 2>,
                weights: &Tensor<bf16, 2>,
                probs: Option<&Tensor<f32, 2>>|
     -> Result<()> { route::validate(x, router, ids, weights, probs).map(|_| ()) };
    let x = |hidden: u32| view::<bf16>(base, [5, hidden], [u64::from(hidden), 1]);
    let router = |experts: u32, hidden: u32| {
        view::<bf16>(base + 0x1000, [experts, hidden], [u64::from(hidden), 1])
    };

    good(&x(128)?, &router(16, 128)?, &ids, &weights, Some(&probs))?;

    let cases = [
        (
            good(&x(128)?, &router(16, 64)?, &ids, &weights, Some(&probs)),
            "route hidden width mismatch",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &view::<i32>(base + 0x2000, [5, 3], [3, 1])?,
                &weights,
                Some(&probs),
            ),
            "route ids/weights shape mismatch",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &view::<i32>(base + 0x2000, [4, 4], [4, 1])?,
                &weights,
                Some(&probs),
            ),
            "route ids/weights shape mismatch",
        ),
        (
            good(
                &view::<bf16>(base, [5, 128], [256, 1])?,
                &router(16, 128)?,
                &ids,
                &weights,
                Some(&probs),
            ),
            "route requires contiguous x and router rows",
        ),
        (
            good(
                &x(128)?,
                &view::<bf16>(base + 0x1000, [16, 128], [257, 1])?,
                &ids,
                &weights,
                Some(&probs),
            ),
            "route requires contiguous x and router rows",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &view::<i32>(base + 0x2000, [5, 4], [5, 1])?,
                &weights,
                Some(&probs),
            ),
            "route requires contiguous ids and weights rows",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &ids,
                &view::<bf16>(base + 0x3000, [5, 4], [5, 1])?,
                Some(&probs),
            ),
            "route requires contiguous ids and weights rows",
        ),
        (
            good(&x(12)?, &router(16, 12)?, &ids, &weights, Some(&probs)),
            "route hidden width must be a multiple of 8",
        ),
        (
            good(
                &x(128)?,
                &router(1040, 128)?,
                &view::<i32>(base + 0x2000, [5, 4], [4, 1])?,
                &weights,
                Some(&view::<f32>(base + 0x4000, [5, 1040], [1040, 1])?),
            ),
            "route expert count above 1024",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &view::<i32>(base + 0x2000, [5, 33], [33, 1])?,
                &view::<bf16>(base + 0x3000, [5, 33], [33, 1])?,
                Some(&view::<f32>(base + 0x4000, [5, 16], [16, 1])?),
            ),
            "route top_k outside [1,32]",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &view::<i32>(base + 0x2000, [5, 16], [16, 1])?,
                &view::<bf16>(base + 0x3000, [5, 16], [16, 1])?,
                Some(&probs),
            ),
            "route top_k must be below the expert count",
        ),
        (
            good(
                &x(128)?,
                &router(16, 128)?,
                &ids,
                &weights,
                Some(&view::<f32>(base + 0x4000, [5, 15], [15, 1])?),
            ),
            "route probs must be a contiguous [rows, experts] view",
        ),
    ];
    for (case, needle) in cases {
        let error = case.expect_err("route validation must reject the view");
        ensure!(
            error.to_string().contains(needle),
            "route validation error {error} lacks {needle}"
        );
    }

    let plan = route::validate(&x(128)?, &router(16, 128)?, &ids, &weights, None)?;
    ensure!(
        plan.probs_ptr == 0,
        "route without probs must pass a null pointer"
    );
    ensure!(
        plan.rows == 5 && plan.experts == 16 && plan.top_k == 4 && plan.hidden == 128,
        "route plan dims mismatch"
    );
    Ok(())
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
    probs: &'a [f32],
}

fn compare_case(
    case: u32,
    refs: &HashMap<usize, RouteRow>,
    rows: &[usize],
    out: &CaseOutputs<'_>,
    tol: f32,
) -> Result<()> {
    let (e, k) = (out.dims.experts, out.dims.top_k);
    let mut strict = 0;
    let mut gated = 0;
    for &row in rows {
        let want = &refs[&row];
        let ids = &out.ids[row * k..(row + 1) * k];
        let weights = &out.weights[row * k..(row + 1) * k];
        let probs = &out.probs[row * e..(row + 1) * e];
        ensure!(
            ids.iter().all(|&id| id >= 0) && weights.iter().all(|w| w.to_f32() != -17.0),
            "route case {case} row {row} left sentinel outputs"
        );
        ensure!(
            probs.iter().all(|p| *p >= 0.0 && p.is_finite()),
            "route case {case} row {row} wrote invalid probabilities"
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
        for (expert, (&got, &want_p)) in probs.iter().zip(&want.probs).enumerate() {
            ensure!(
                (got - want_p).abs() <= PROBS_REL_TOL * want_p + PROBS_ABS_FLOOR,
                "route case {case} row {row} prob {expert} {got:e} vs {want_p:e}"
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
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn moe_route_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
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
    let mut pb = testing::zeros::<f32>(s, dims.rows * e)?;

    let run = |rows: u32,
               xb: &CudaSlice<bf16>,
               rb: &CudaSlice<bf16>,
               idsb: &mut CudaSlice<i32>,
               wb: &mut CudaSlice<bf16>,
               pb: &mut CudaSlice<f32>|
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
        let mut probs = TensorMut(
            testing::dense_mut(pb, s, dims.rows as u32, e as u32)?
                .0
                .prefix_rows(rows)?,
        );
        ops.route(s, x, router, &mut ids, &mut weights, Some(&mut probs))
    };

    for &case in &ROW_CASES {
        s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
        s.memcpy_htod(&vec![bf16::from_f32(-17.0); dims.rows * k], &mut wb)?;
        s.memcpy_htod(&vec![-1.0f32; dims.rows * e], &mut pb)?;
        run(case, &xb, &rb, &mut idsb, &mut wb, &mut pb)?;
        s.synchronize()?;
        let rows = checked_rows(case);
        let ids = s.clone_dtoh(&idsb)?;
        let weights = s.clone_dtoh(&wb)?;
        let probs = s.clone_dtoh(&pb)?;
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
                probs: &probs,
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
    run(1, &xb, &rb, &mut idsb, &mut wb, &mut pb)?;
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

    if std::env::var("FF_MOE_TIMING").as_deref() == Ok("1") {
        use cudarc::driver::sys::CUevent_flags;
        let (begin, end) = (
            d.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
            d.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        );
        const WARMUP: usize = 20;
        const ITERS: usize = 100;
        let router_bytes = (e * h * 2) as f64;
        for &case in &[1u32, dims.rows as u32] {
            for _ in 0..WARMUP {
                run(case, &xb, &rb, &mut idsb, &mut wb, &mut pb)?;
            }
            s.synchronize()?;
            begin.record(s)?;
            for _ in 0..ITERS {
                run(case, &xb, &rb, &mut idsb, &mut wb, &mut pb)?;
            }
            end.record(s)?;
            s.synchronize()?;
            let per_launch_us = f64::from(begin.elapsed_ms(&end)?) * 1000.0 / ITERS as f64;
            let gb_s = case as f64 * router_bytes / (per_launch_us * 1e-6) / 1e9;
            eprintln!(
                "MOE_ROUTE_TIMING rows={case} iters={ITERS} per_launch_us={per_launch_us:.3} router_GB_s={gb_s:.1}"
            );
        }
    }
    Ok(())
}

#[test]
fn distinct_row_experts_validation_rejects_duplicates() -> Result<()> {
    let distinct: Vec<i32> = vec![5, 0, 9, 200, 7, 3, 2, 1];
    super::route::ensure_distinct_row_experts(&distinct, 8)?;
    let duplicate: Vec<i32> = vec![5, 0, 9, 5, 7, 3, 2, 1];
    let err = super::route::ensure_distinct_row_experts(&duplicate, 8)
        .expect_err("duplicate within a row must be rejected");
    assert!(
        err.to_string().contains("duplicate expert id"),
        "unexpected error: {err}"
    );
    let bad_len: Vec<i32> = vec![1, 2, 3];
    assert!(super::route::ensure_distinct_row_experts(&bad_len, 8).is_err());
    Ok(())
}
