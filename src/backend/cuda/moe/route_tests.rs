//! Route-op gates: reference layout checks and the GPU window gate for
//! kernels/moe_route.cu. Default cargo test never initializes a GPU here.
use super::reference::{self, MoeDims, RouteRow};
use super::route::RouteOps;
use super::testing;
use crate::backend::cuda::contracts::TensorMut;
use crate::backend::cuda::{Device, cubin};
use crate::backend::gpu::{CudaSlice, LaunchConfig, PushKernelArg};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::collections::HashMap;

fn strict_ids(row: &RouteRow) -> bool {
    let cut = row.probs[*row.ids.last().expect("nonempty top-k") as usize];
    row.margin > row.probs.len() as f32 * (f32::EPSILON / 2.0) * cut
}

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
) -> Result<()> {
    let k = out.dims.top_k;
    let mut strict = 0;
    let mut gated = 0;
    let mut max_weight_dev = 0.0f32;
    for &row in rows {
        let want = &refs[&row];
        let ids = &out.ids[row * k..(row + 1) * k];
        let weights = &out.weights[row * k..(row + 1) * k];
        ensure!(
            ids.iter().all(|&id| id >= 0) && weights.iter().all(|w| w.to_f32() != -17.0),
            "route case {case} row {row} left sentinel outputs"
        );
        let cut = want.probs[want.ids[k - 1] as usize];
        if strict_ids(want) {
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
            strict_ids(want)
        );
        for (j, (got, &expected)) in weights.iter().zip(&want.weights).enumerate() {
            let (got, expected) = (got.to_f32(), expected.to_f32());
            let dev = (got - expected).abs();
            let rounding = crate::backend::cuda::gamma(k + 1, f64::from(f32::EPSILON) / 2.0) as f32;
            let tol = reference::bf16_ulp(expected) + expected.abs() * rounding;
            max_weight_dev = max_weight_dev.max(dev);
            ensure!(
                dev <= tol,
                "route case {case} row {row} weight {j} {got:e} vs {expected:e} (tol {tol:e})"
            );
        }
    }
    eprintln!(
        "MOE_ROUTE_CASE rows={case} checked={} strict_ids={strict} gated_ids={gated} max_weight_dev={max_weight_dev:e} PASS",
        rows.len()
    );
    Ok(())
}

fn invariant_route(device: &Device) -> Result<()> {
    use crate::backend::cuda::contracts::view;
    let rows = crate::linear_plan::MAX_SKINNY_ROWS;
    let (e, k, h) = (256usize, 8usize, 2048usize);
    let s = &device.stream;
    let ops = RouteOps::new(&device.ctx, s, e as u32, rows)?;
    let values: Vec<_> = (0..rows * h)
        .map(|i| bf16::from_f32(((i / h * 71 + i % h * 13) % 509) as f32 / 256.0 - 254.0 / 256.0))
        .collect();
    let mut router: Vec<_> = (0..e * h)
        .map(|i| bf16::from_f32(((i / h * 37 + i % h * 17) % 257) as f32 / 4096.0 - 128.0 / 4096.0))
        .collect();
    let tie = router[3 * h..4 * h].to_vec();
    router[7 * h..8 * h].copy_from_slice(&tie);
    let x = s.clone_htod(&values)?;
    let router = s.clone_htod(&router)?;
    let w = view(&router, s, 0, [e as u32, h as u32], [h as u64, 1])?;
    let expected_ids = s.alloc_zeros::<i32>(rows * k)?;
    let expected_weights = s.alloc_zeros::<bf16>(rows * k)?;
    let logits = s.alloc_zeros::<bf16>(rows * e)?;
    let logits_view = |m: u32| -> Result<_> {
        Ok(TensorMut(view(
            &logits,
            s,
            0,
            [m, e as u32],
            [e as u64, 1],
        )?))
    };
    device.finish_upload()?;
    for r in 0..rows {
        ops.route(
            view(&x, s, r * h, [1, h as u32], [h as u64, 1])?,
            w,
            &mut TensorMut(view(&expected_ids, s, r * k, [1, k as u32], [k as u64, 1])?),
            &mut TensorMut(view(
                &expected_weights,
                s,
                r * k,
                [1, k as u32],
                [k as u64, 1],
            )?),
            &mut logits_view(1)?,
            crate::model_spec::RouterScore::Softmax,
        )?;
    }
    let expected_ids = s.clone_dtoh(&expected_ids)?;
    let expected_weights = s.clone_dtoh(&expected_weights)?;
    let canary = bf16::from_bits(0x7fc1);
    let ids = s.clone_htod(&vec![-33i32; (rows + 1) * k])?;
    let weights = s.clone_htod(&vec![canary; (rows + 1) * k])?;
    let mut defects = 0;
    for m in 1..=rows as u32 {
        let input = view(&x, s, 0, [m, h as u32], [h as u64, 1])?;
        let mut id_view = TensorMut(view(&ids, s, 0, [m, k as u32], [k as u64, 1])?);
        let mut weight_view = TensorMut(view(&weights, s, 0, [m, k as u32], [k as u64, 1])?);
        let mut logit_view = logits_view(m)?;
        let mut execute = || {
            ops.route(
                input,
                w,
                &mut id_view,
                &mut weight_view,
                &mut logit_view,
                crate::model_spec::RouterScore::Softmax,
            )
        };
        for graph in [false, true] {
            if graph {
                s.synchronize()?;
                testing::capture(s, &mut execute)?.launch()?;
            } else {
                execute()?;
            }
            let got_ids = s.clone_dtoh(&ids)?;
            let got_weights = s.clone_dtoh(&weights)?;
            let choices = m as usize * k;
            ensure!(
                got_ids[choices..].iter().all(|&v| v == -33)
                    && got_weights[choices..]
                        .iter()
                        .all(|v| v.to_bits() == canary.to_bits()),
                "router wrote a dead row"
            );
            ensure!(
                got_weights[..choices].iter().all(|v| v.is_finite()),
                "nonfinite invariant router boundary"
            );
            let mismatches = got_ids[..choices]
                .iter()
                .zip(&expected_ids)
                .filter(|(a, b)| a != b)
                .count()
                + got_weights[..choices]
                    .iter()
                    .zip(&expected_weights)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            defects += usize::from(mismatches != 0);
            println!(
                "{} INVARIANT_ROUTE x=[{m},{h}] router=[{e},{h}] top_k={k} graph={graph} ids_weights_bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
        }
    }
    let rx = crate::backend::cuda::random(rows * h, 11);
    let rw = crate::backend::cuda::random(e * h, 12);
    let (dx, dw) = (s.clone_htod(&rx)?, s.clone_htod(&rw)?);
    ops.route(
        view(&dx, s, 0, [rows as u32, h as u32], [h as u64, 1])?,
        view(&dw, s, 0, [e as u32, h as u32], [h as u64, 1])?,
        &mut TensorMut(view(&ids, s, 0, [rows as u32, k as u32], [k as u64, 1])?),
        &mut TensorMut(view(
            &weights,
            s,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?),
        &mut logits_view(rows as u32)?,
        crate::model_spec::RouterScore::Softmax,
    )?;
    let got_logits = s.clone_dtoh(&logits)?;
    let chain = |r: usize, j: usize, order: &mut dyn Iterator<Item = usize>| {
        bf16::from_f32(order.fold(0f32, |acc, i| {
            rx[r * h + i].to_f32().mul_add(rw[j * h + i].to_f32(), acc)
        }))
    };
    let (mut logit_mismatches, mut reversed_differs) = (0usize, 0usize);
    for r in 0..rows {
        for j in 0..e {
            let got = got_logits[r * e + j].to_bits();
            logit_mismatches += usize::from(got != chain(r, j, &mut (0..h)).to_bits());
            reversed_differs += usize::from(got != chain(r, j, &mut (0..h).rev()).to_bits());
        }
    }
    ensure!(
        reversed_differs > 0,
        "router logit bit control escaped: reversed accumulation matches every logit"
    );
    println!(
        "{} ROUTE_LOGITS_BITWISE x=[{rows},{h}] router=[{e},{h}] fmaf_ascending_mismatches={logit_mismatches} reversed_control_differs={reversed_differs}",
        if logit_mismatches == 0 {
            "PASS"
        } else {
            "DEFECT"
        }
    );
    defects += usize::from(logit_mismatches != 0);
    let mut wrong = values.clone();
    for r in 1..rows {
        wrong[r * h..(r + 1) * h].copy_from_slice(&values[..h]);
    }
    let wrong = s.clone_htod(&wrong)?;
    ops.route(
        view(&wrong, s, 0, [rows as u32, h as u32], [h as u64, 1])?,
        w,
        &mut TensorMut(view(&ids, s, 0, [rows as u32, k as u32], [k as u64, 1])?),
        &mut TensorMut(view(
            &weights,
            s,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?),
        &mut logits_view(rows as u32)?,
        crate::model_spec::RouterScore::Softmax,
    )?;
    let mismatches = s
        .clone_dtoh(&ids)?
        .iter()
        .zip(&expected_ids)
        .filter(|(a, b)| a != b)
        .count()
        + s.clone_dtoh(&weights)?
            .iter()
            .zip(&expected_weights)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    ensure!(mismatches > 0, "router row0-reuse control escaped");
    println!("PASS INVARIANT_ROUTE_CONTROL row0_reuse=true bit_mismatches={mismatches}");
    println!(
        "INVARIANT_ROUTE_SUMMARY rows=1..16 eager_graph=true ids_weights=true defects={defects}"
    );
    ensure!(defects == 0, "invariant router DEFECT cells={defects}");
    Ok(())
}

fn sigmoid_route(device: &Device, ops: &RouteOps) -> Result<()> {
    let dims = MoeDims::new(4096, 256, 6, 2048, 512);
    let (e, k, h) = (dims.experts, dims.top_k, dims.hidden);
    let fx = reference::fixture(dims, 3);
    let s = &device.stream;
    let xb = testing::upload_device(device, fx.x.clone())?;
    let rb = testing::upload_device(device, fx.router_w.clone())?;
    device.finish_upload()?;
    let mut idsb = testing::zeros::<i32>(s, dims.rows * k)?;
    let mut wb = testing::zeros::<bf16>(s, dims.rows * k)?;
    let lb = testing::zeros::<bf16>(s, dims.rows * e)?;

    let run = |row: usize, idsb: &mut CudaSlice<i32>, wb: &mut CudaSlice<bf16>| -> Result<()> {
        let rows = u32::try_from(row + 1)?;
        let x = testing::dense(&xb, s, dims.rows as u32, h as u32)?.prefix_rows(rows)?;
        let router = testing::dense(&rb, s, e as u32, h as u32)?;
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
        let mut logits = TensorMut(
            testing::dense_mut(&lb, s, dims.rows as u32, e as u32)?
                .0
                .prefix_rows(rows)?,
        );
        ops.route(
            x,
            router,
            &mut ids,
            &mut weights,
            &mut logits,
            crate::model_spec::RouterScore::Sigmoid,
        )
    };

    let mut strict = 0;
    let mut skipped = 0;
    for row in [0usize, 1, 3, 7, 15, 63, 1023, 2047] {
        let want = reference::sigmoid_route_row(&fx, row);
        ensure!(
            want.probs.iter().all(|value| value.is_finite()),
            "non-finite sigmoid reference row={row}"
        );
        if want.margin <= f32::EPSILON {
            skipped += 1;
            continue;
        }
        s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
        s.memcpy_htod(&vec![bf16::from_f32(-17.0); dims.rows * k], &mut wb)?;
        run(row, &mut idsb, &mut wb)?;
        s.synchronize()?;
        let ids = s.clone_dtoh(&idsb)?;
        let weights = s.clone_dtoh(&wb)?;
        ensure!(
            ids[row * k..(row + 1) * k] == want.ids[..],
            "sigmoid route row={row} ids {:?} != {:?}",
            &ids[row * k..(row + 1) * k],
            &want.ids[..]
        );
        for (got, expected) in weights[row * k..(row + 1) * k].iter().zip(&want.weights) {
            let got = got.to_f32();
            let expected = expected.to_f32();
            let tolerance = reference::bf16_ulp(expected) + f32::EPSILON;
            ensure!(
                (got - expected).abs() <= tolerance,
                "sigmoid route row={row} weight {got:e} != {expected:e}"
            );
        }
        strict += 1;
    }
    ensure!(strict > 0, "sigmoid route gate had no strict rows");

    s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
    run(fx.exact_tie_row, &mut idsb, &mut wb)?;
    s.synchronize()?;
    let first = (
        s.clone_dtoh(&idsb)?[..k].to_vec(),
        s.clone_dtoh(&wb)?[..k].to_vec(),
    );
    s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
    run(fx.exact_tie_row, &mut idsb, &mut wb)?;
    s.synchronize()?;
    let second = (
        s.clone_dtoh(&idsb)?[..k].to_vec(),
        s.clone_dtoh(&wb)?[..k].to_vec(),
    );
    ensure!(
        first == second,
        "sigmoid exact-tie kernel is not deterministic"
    );
    let tie = reference::sigmoid_route_row(&fx, fx.exact_tie_row);
    let (lower, higher) = fx.exact_tie_experts;
    let lower_rank = tie.ids.iter().position(|id| *id as usize == lower);
    let higher_rank = tie.ids.iter().position(|id| *id as usize == higher);
    ensure!(
        lower_rank.is_some() && higher_rank.is_some() && lower_rank < higher_rank,
        "sigmoid exact tie did not order lower expert id first"
    );
    let mut neighbor_logits = vec![bf16::from_f32(-1.0); e];
    for logit in &mut neighbor_logits[..5] {
        *logit = bf16::from_f32(10.0);
    }
    neighbor_logits[5] = bf16::ONE;
    neighbor_logits[250] = bf16::ONE;
    let mut neighbor_logits_dev = s.clone_htod(&neighbor_logits)?;
    let mut neighbor_ids = s.alloc_zeros::<i32>(k)?;
    let neighbor_weights = s.alloc_zeros::<bf16>(k)?;
    let sigmoid_topk =
        cubin::module(&device.ctx, "moe_route")?.load_function("moe_route_topk_sigmoid")?;
    unsafe {
        s.launch_builder(&sigmoid_topk)
            .arg(&neighbor_logits_dev)
            .arg(&neighbor_ids)
            .arg(&neighbor_weights)
            .arg(&1i32)
            .arg(&(e as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    s.synchronize()?;
    let tie_ids = s.clone_dtoh(&neighbor_ids)?;
    ensure!(
        tie_ids.contains(&(5i32)) && !tie_ids.contains(&250i32),
        "sigmoid tie control did not select lower expert id"
    );
    neighbor_logits[250] = bf16::from_bits(bf16::ONE.to_bits() + 1);
    s.memcpy_htod(&neighbor_logits, &mut neighbor_logits_dev)?;
    s.memcpy_htod(&vec![-1i32; k], &mut neighbor_ids)?;
    unsafe {
        s.launch_builder(&sigmoid_topk)
            .arg(&neighbor_logits_dev)
            .arg(&neighbor_ids)
            .arg(&neighbor_weights)
            .arg(&1i32)
            .arg(&(e as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    s.synchronize()?;
    let neighbor_ids = s.clone_dtoh(&neighbor_ids)?;
    ensure!(
        neighbor_ids.contains(&250i32) && !neighbor_ids.contains(&5i32),
        "sigmoid one-ULP neighbor did not change selection"
    );
    eprintln!(
        "MOE_ROUTE_SIGMOID strict_rows={strict} tie_rows_skipped={skipped} deterministic=true tie_lower_id=true one_ulp_neighbor_changes_selection=true PASS"
    );
    Ok(())
}

fn noaux_margin(
    logits: &[f32],
    bias: &[f32],
    ids: &[i32],
    score: crate::model_spec::RouterScore,
) -> f32 {
    let raw = |id: usize| match score {
        crate::model_spec::RouterScore::Sigmoid => 1.0 / (1.0 + (-logits[id]).exp()),
        crate::model_spec::RouterScore::SqrtSoftplus => {
            let softplus = logits[id].max(0.0) + (-logits[id].abs()).exp().ln_1p();
            softplus.sqrt()
        }
        crate::model_spec::RouterScore::Softmax => unreachable!(),
    };
    let last = ids[ids.len() - 1] as usize;
    let cut = raw(last) + bias[last];
    let next = (0..logits.len())
        .filter(|id| !ids.contains(&(*id as i32)))
        .map(|id| raw(id) + bias[id])
        .fold(f32::NEG_INFINITY, f32::max);
    cut - next
}

#[test]
#[ignore = "requires a GPU window; FF_Q38_GPU_WINDOW=granted"]
fn moe_route_decode_bitwise_gpu() -> Result<()> {
    use crate::backend::cuda::contracts::view;
    use crate::model_spec::RouterScore;
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    let s = &device.stream;
    let module = cubin::module(&device.ctx, "moe_route")?;
    let baseline_logits = module.load_function("moe_route_logits_bf16")?;
    let tie_control = module.load_function("moe_route_decode_reverse_ties")?;
    let mut defects = 0;
    for (e, h, k, score) in [
        (256usize, 2048usize, 8usize, RouterScore::Softmax),
        (256, 2048, 6, RouterScore::Sigmoid),
        (33, 520, 8, RouterScore::Softmax),
        (33, 520, 6, RouterScore::Sigmoid),
        (1024, 8, 32, RouterScore::Softmax),
    ] {
        let rows = 16usize;
        let ops = RouteOps::new(&device.ctx, s, e as u32, rows)?;
        let topk = module.load_function(if score == RouterScore::Softmax {
            "moe_route_topk"
        } else {
            "moe_route_topk_sigmoid"
        })?;
        let ties: Vec<_> = (0..=k).map(|i| i * (e - 1) / k).collect();
        let mut values: Vec<_> = (0..rows * h)
            .map(|i| {
                bf16::from_f32(((i / h * 71 + i % h * 13) % 509) as f32 / 256.0 - 254.0 / 256.0)
            })
            .collect();
        values[..3 * h].fill(bf16::ZERO);
        values[h] = bf16::ONE;
        values[2 * h] = bf16::from_f32(1024.0);
        let mut router: Vec<_> = (0..e * h)
            .map(|i| {
                bf16::from_f32(((i / h * 37 + i % h * 17) % 257) as f32 / 4096.0 - 128.0 / 4096.0)
            })
            .collect();
        for expert in 0..e {
            router[expert * h] = bf16::from_f32(if ties.contains(&expert) { 1.0 } else { -1.0 });
        }
        let x = s.clone_htod(&values)?;
        let w = s.clone_htod(&router)?;
        device.finish_upload()?;
        let expected_logits = s.alloc_zeros::<bf16>(rows * e)?;
        let expected_ids = s.alloc_zeros::<i32>(rows * k)?;
        let expected_weights = s.alloc_zeros::<bf16>(rows * k)?;
        let xv = view(&x, s, 0, [rows as u32, h as u32], [h as u64, 1])?;
        let wv = view(&w, s, 0, [e as u32, h as u32], [h as u64, 1])?;
        let lv = view(
            &expected_logits,
            s,
            0,
            [rows as u32, e as u32],
            [e as u64, 1],
        )?;
        let iv = view(&expected_ids, s, 0, [rows as u32, k as u32], [k as u64, 1])?;
        let wtv = view(
            &expected_weights,
            s,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        unsafe {
            s.launch_builder(&baseline_logits)
                .arg(&xv.ptr)
                .arg(&wv.ptr)
                .arg(&lv.ptr)
                .arg(&(rows as u32))
                .arg(&(e as u32))
                .arg(&(h as u32))
                .launch(LaunchConfig {
                    grid_dim: ((e as u32).div_ceil(32), rows as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            s.launch_builder(&topk)
                .arg(&lv.ptr)
                .arg(&iv.ptr)
                .arg(&wtv.ptr)
                .arg(&(rows as u32))
                .arg(&(e as u32))
                .arg(&(k as u32))
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: ((e as u32).div_ceil(32) * 32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let expected_logits = s.clone_dtoh(&expected_logits)?;
        let expected_ids = s.clone_dtoh(&expected_ids)?;
        let expected_weights = s.clone_dtoh(&expected_weights)?;
        ensure!(
            expected_logits
                .iter()
                .chain(&expected_weights)
                .all(|v| v.is_finite()),
            "FAIL nonfinite baseline route boundary"
        );
        ensure!(
            expected_ids[..k].iter().copied().eq(0..k as i32)
                && expected_ids[k..2 * k]
                    .iter()
                    .copied()
                    .eq(ties[..k].iter().map(|&v| v as i32))
                && expected_ids[2 * k..3 * k] == expected_ids[k..2 * k],
            "FAIL baseline exact-tie fixture validity"
        );
        for r in 0..rows {
            let selected = &expected_ids[r * k..(r + 1) * k];
            let cut = expected_logits[r * e + selected[k - 1] as usize].to_f32();
            let outside = (0..e)
                .filter(|j| !selected.contains(&(*j as i32)))
                .map(|j| expected_logits[r * e + j].to_f32())
                .fold(f32::NEG_INFINITY, f32::max);
            println!(
                "ROUTE_DECODE_MARGIN score={score:?} row={r} bf16_logit_cut_margin={:e}",
                cut - outside
            );
        }
        if e == 256 {
            let logits = s.alloc_zeros::<bf16>(e)?;
            let ids = s.alloc_zeros::<i32>(k)?;
            let weights = s.alloc_zeros::<bf16>(k)?;
            let completed = s.alloc_zeros::<u32>(1)?;
            let mode = if score == RouterScore::Softmax {
                0i32
            } else {
                1
            };
            unsafe {
                s.launch_builder(&tie_control)
                    .arg(&xv.ptr)
                    .arg(&wv.ptr)
                    .arg(&logits)
                    .arg(&ids)
                    .arg(&weights)
                    .arg(&completed)
                    .arg(&1u32)
                    .arg(&(e as u32))
                    .arg(&(h as u32))
                    .arg(&(k as u32))
                    .arg(&2u32)
                    .arg(&mode)
                    .launch(LaunchConfig {
                        grid_dim: (128, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 16 * (2 * 65 + 64),
                    })?;
            }
            let logits = s.clone_dtoh(&logits)?;
            let ids = s.clone_dtoh(&ids)?;
            let weights = s.clone_dtoh(&weights)?;
            let id_diff = ids
                .iter()
                .zip(&expected_ids)
                .filter(|(a, b)| a != b)
                .count();
            ensure!(
                logits
                    .iter()
                    .zip(&expected_logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && weights
                        .iter()
                        .zip(&expected_weights)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                    && id_diff == k
                    && ids.iter().copied().eq((e - k..e).rev().map(|v| v as i32)),
                "FAIL reversed-tie control escaped or perturbed arithmetic"
            );
            println!(
                "PASS ROUTE_DECODE_CONTROL score={score:?} reverse_global_id_ties=true logits=0 weights=0 id_mismatches={id_diff}"
            );
        }
        for m in 1..=rows {
            let canary = bf16::from_bits(0x7fc1);
            let logits = s.clone_htod(&vec![canary; (m + 1) * e])?;
            let ids = s.clone_htod(&vec![-33i32; (m + 1) * k])?;
            let weights = s.clone_htod(&vec![canary; (m + 1) * k])?;
            let mut l = TensorMut(view(&logits, s, 0, [m as u32, e as u32], [e as u64, 1])?);
            let mut i = TensorMut(view(&ids, s, 0, [m as u32, k as u32], [k as u64, 1])?);
            let mut wt = TensorMut(view(&weights, s, 0, [m as u32, k as u32], [k as u64, 1])?);
            let mut run = || {
                ops.route(
                    xv.prefix_rows(m as u32)?,
                    wv,
                    &mut i,
                    &mut wt,
                    &mut l,
                    score,
                )
            };
            for graph in [false, true] {
                if graph {
                    s.synchronize()?;
                    let captured = testing::capture(s, &mut run)?;
                    for _ in 0..3 {
                        captured.launch()?;
                    }
                } else {
                    for _ in 0..3 {
                        run()?;
                    }
                }
                let actual_logits = s.clone_dtoh(&logits)?;
                let actual_ids = s.clone_dtoh(&ids)?;
                let actual_weights = s.clone_dtoh(&weights)?;
                ensure!(
                    actual_logits[..m * e]
                        .iter()
                        .chain(&actual_weights[..m * k])
                        .all(|v| v.is_finite()),
                    "FAIL nonfinite fused route boundary"
                );
                let bounds = actual_logits[m * e..]
                    .iter()
                    .chain(&actual_weights[m * k..])
                    .all(|v| v.to_bits() == canary.to_bits())
                    && actual_ids[m * k..].iter().all(|&v| v == -33);
                let logit_diff = actual_logits[..m * e]
                    .iter()
                    .zip(&expected_logits)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let id_diff = actual_ids[..m * k]
                    .iter()
                    .zip(&expected_ids)
                    .filter(|(a, b)| a != b)
                    .count();
                let weight_diff = actual_weights[..m * k]
                    .iter()
                    .zip(&expected_weights)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let pass = bounds && logit_diff == 0 && id_diff == 0 && weight_diff == 0;
                defects += usize::from(!pass);
                println!(
                    "{} ROUTE_DECODE_BITWISE sm={} x=[{m},{h}] router=[{e},{h}] k={k} score={score:?} graph={graph} repeats=3 logits={logit_diff} ids={id_diff} weights={weight_diff} bounds={bounds}",
                    if pass { "PASS" } else { "DEFECT" },
                    device.info.arch
                );
            }
        }
        let mut instances = Vec::new();
        for stream in [s.clone(), device.ctx.new_stream()?] {
            let independent = RouteOps::new(&device.ctx, &stream, e as u32, rows)?;
            let logits = stream.alloc_zeros::<bf16>(rows * e)?;
            let ids = stream.alloc_zeros::<i32>(rows * k)?;
            let weights = stream.alloc_zeros::<bf16>(rows * k)?;
            let mut l = testing::dense_mut(&logits, &stream, rows as u32, e as u32)?;
            let mut i = testing::dense_mut(&ids, &stream, rows as u32, k as u32)?;
            let mut wt = testing::dense_mut(&weights, &stream, rows as u32, k as u32)?;
            for _ in 0..3 {
                independent.route(xv, wv, &mut i, &mut wt, &mut l, score)?;
            }
            instances.push((independent, stream, logits, ids, weights));
        }
        for (_ops, stream, logits, ids, weights) in instances {
            let actual_logits = stream.clone_dtoh(&logits)?;
            let actual_ids = stream.clone_dtoh(&ids)?;
            let actual_weights = stream.clone_dtoh(&weights)?;
            ensure!(
                actual_logits
                    .iter()
                    .chain(&actual_weights)
                    .all(|v| v.is_finite()),
                "FAIL nonfinite independent-stream boundary"
            );
            let pass = actual_logits
                .iter()
                .zip(&expected_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && actual_ids == expected_ids
                && actual_weights
                    .iter()
                    .zip(&expected_weights)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            defects += usize::from(!pass);
            println!(
                "{} ROUTE_DECODE_STREAMS x=[{rows},{h}] router=[{e},{h}] score={score:?} repeats=3",
                if pass { "PASS" } else { "DEFECT" }
            );
        }
    }
    println!("ROUTE_DECODE_BITWISE_SUMMARY defects={defects}");
    ensure!(defects == 0, "fused router DEFECT cells={defects}");
    Ok(())
}

fn noaux_route(device: &Device, ops: &RouteOps) -> Result<()> {
    let (rows, experts) = (2usize, 8usize);
    let base = [0.25f32, -0.5, 1.0, 0.75, -1.25, 0.125, 0.5, -0.75];
    let mut host_logits = base.repeat(rows);
    host_logits[experts + 3] = host_logits[experts + 2];
    let host_bias: Vec<f32> = (0..rows)
        .flat_map(|_| [0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.4])
        .collect();
    let s = &device.stream;
    let lb = testing::upload_device(device, host_logits.clone())?;
    let bb = testing::upload_device(device, host_bias.clone())?;
    device.finish_upload()?;
    let modes = [
        ("sigmoid", crate::model_spec::RouterScore::Sigmoid, 2.5),
        (
            "sqrtsoftplus",
            crate::model_spec::RouterScore::SqrtSoftplus,
            1.5,
        ),
    ];
    for (mode, score, scaling) in modes {
        for top_k in [4usize, 8] {
            let ids = testing::zeros::<i32>(s, rows * top_k)?;
            let weights = testing::zeros::<bf16>(s, rows * top_k)?;
            let logits = testing::dense(&lb, s, rows as u32, experts as u32)?;
            let bias = testing::dense(&bb, s, rows as u32, experts as u32)?;
            let mut ids_view = testing::dense_mut(&ids, s, rows as u32, top_k as u32)?;
            let mut weights_view = testing::dense_mut(&weights, s, rows as u32, top_k as u32)?;
            ops.topk_noaux(
                logits,
                Some(bias),
                &mut ids_view,
                &mut weights_view,
                score,
                scaling,
            )?;
            s.synchronize()?;
            let actual_ids = s.clone_dtoh(&ids)?;
            let actual_weights = s.clone_dtoh(&weights)?;
            for row in 0..rows {
                let row_logits = &host_logits[row * experts..(row + 1) * experts];
                let (expected_ids, expected_weights) = reference::noaux_tc_route(
                    row_logits,
                    &host_bias[row * experts..(row + 1) * experts],
                    top_k,
                    score,
                    scaling,
                )?;
                ensure!(
                    actual_ids[row * top_k..(row + 1) * top_k] == expected_ids,
                    "noaux {mode} top_k={top_k} row={row} ids mismatch"
                );
                let margin = noaux_margin(
                    row_logits,
                    &host_bias[row * experts..(row + 1) * experts],
                    &expected_ids,
                    score,
                );
                if top_k < experts {
                    ensure!(
                        margin > 2.0 * f32::EPSILON,
                        "noaux {mode} top_k={top_k} row={row} cut margin {margin:e} is below the ID detection floor"
                    );
                }
                eprintln!(
                    "MOE_ROUTE_NOAUX_MARGIN mode={mode} top_k={top_k} row={row} margin={margin:e}"
                );
                if row == 1 {
                    ensure!(
                        row_logits[2] == row_logits[3]
                            && host_bias[experts + 2] == host_bias[experts + 3],
                        "noaux exact-tie fixture is not tied"
                    );
                    let row_ids = &actual_ids[row * top_k..(row + 1) * top_k];
                    let lower = row_ids.iter().position(|&id| id == 2);
                    let higher = row_ids.iter().position(|&id| id == 3);
                    ensure!(
                        lower.is_some() && higher.is_some() && lower < higher,
                        "noaux exact tie did not select the lower expert ID first"
                    );
                }
                for (choice, (actual, expected)) in actual_weights[row * top_k..(row + 1) * top_k]
                    .iter()
                    .zip(&expected_weights)
                    .enumerate()
                {
                    let actual = actual.to_f32();
                    let expected = *expected;
                    let tolerance = reference::bf16_ulp(expected) + 2.0 * f32::EPSILON;
                    ensure!(
                        (actual - expected).abs() <= tolerance,
                        "noaux {mode} top_k={top_k} row={row} choice={choice} actual={actual} expected={expected} tolerance={tolerance}"
                    );
                }
                if top_k == 4 && row == 0 {
                    let (zero_bias_ids, _) = reference::noaux_tc_route(
                        row_logits,
                        &vec![0.0; experts],
                        top_k,
                        score,
                        scaling,
                    )?;
                    ensure!(
                        zero_bias_ids != expected_ids,
                        "noaux {mode} bias control does not change selection"
                    );
                }
            }
            eprintln!("MOE_ROUTE_NOAUX mode={mode} top_k={top_k} scaling={scaling} PASS");
        }
    }
    let experts = 288usize;
    let launch = crate::backend::cuda::ops::grid(1, experts as u32);
    ensure!(
        launch.block_dim.0 as usize >= experts,
        "noaux launch blockDim {} is below experts={experts}",
        launch.block_dim.0
    );
    let host_logits: Vec<f32> = (0..experts)
        .map(|index| {
            let x = index as f32;
            (x * 0.37).sin() + 0.25 * (x * 0.013).cos() + (index % 17) as f32 * 1e-6
        })
        .collect();
    let host_bias: Vec<f32> = (0..experts)
        .map(|index| 0.05 * (index as f32 * 0.19).cos())
        .collect();
    let lb = testing::upload_device(device, host_logits.clone())?;
    let bb = testing::upload_device(device, host_bias.clone())?;
    device.finish_upload()?;
    for (mode, score, scaling) in modes {
        let top_k = 8usize;
        let ids = testing::zeros::<i32>(s, top_k)?;
        let weights = testing::zeros::<bf16>(s, top_k)?;
        let logits = testing::dense(&lb, s, 1, experts as u32)?;
        let bias = testing::dense(&bb, s, 1, experts as u32)?;
        let mut ids_view = testing::dense_mut(&ids, s, 1, top_k as u32)?;
        let mut weights_view = testing::dense_mut(&weights, s, 1, top_k as u32)?;
        ops.topk_noaux(
            logits,
            Some(bias),
            &mut ids_view,
            &mut weights_view,
            score,
            scaling,
        )?;
        s.synchronize()?;
        let actual_ids = s.clone_dtoh(&ids)?;
        let actual_weights = s.clone_dtoh(&weights)?;
        let (expected_ids, expected_weights) =
            reference::noaux_tc_route(&host_logits, &host_bias, top_k, score, scaling)?;
        ensure!(
            actual_ids == expected_ids,
            "noaux {mode} experts={experts} ids mismatch"
        );
        let margin = noaux_margin(&host_logits, &host_bias, &expected_ids, score);
        ensure!(
            margin > 2.0 * f32::EPSILON,
            "noaux {mode} experts={experts} cut margin {margin:e} is below the ID detection floor"
        );
        for (choice, (actual, expected)) in actual_weights.iter().zip(&expected_weights).enumerate()
        {
            let actual = actual.to_f32();
            let expected = *expected;
            let tolerance = reference::bf16_ulp(expected) + 2.0 * f32::EPSILON;
            ensure!(
                (actual - expected).abs() <= tolerance,
                "noaux {mode} experts={experts} choice={choice} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
        eprintln!(
            "MOE_ROUTE_NOAUX_LARGE mode={mode} experts={experts} top_k={top_k} blockDim={} margin={margin:e} PASS",
            launch.block_dim.0
        );
        let captured = testing::capture(s, || {
            ops.topk_noaux(
                logits,
                Some(bias),
                &mut ids_view,
                &mut weights_view,
                score,
                scaling,
            )
        })?;
        for _ in 0..3 {
            captured.launch()?;
        }
        let graph_ids = s.clone_dtoh(&ids)?;
        let graph_weights = s.clone_dtoh(&weights)?;
        let unchanged_logits = s.clone_dtoh(&lb)?;
        let unchanged_bias = s.clone_dtoh(&bb)?;
        ensure!(
            graph_ids == actual_ids
                && graph_weights
                    .iter()
                    .zip(&actual_weights)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && unchanged_logits
                    .iter()
                    .zip(&host_logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && unchanged_bias
                    .iter()
                    .zip(&host_bias)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "noaux graph/input bitwise mismatch"
        );
        println!(
            "PASS ROUTE_NOAUX_BITWISE mode={mode} logits=[1,{experts}] top_k={top_k} eager_graph=true repeats=3"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn moe_route_noaux_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    let ops = RouteOps::new(&device.ctx, &device.stream, 288, 2)?;
    noaux_route(&device, &ops)
}

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn moe_route_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    invariant_route(&d)?;
    let s = &d.stream;
    let ops = RouteOps::new(&d.ctx, s, 256, 4096)?;
    sigmoid_route(&d, &ops)?;
    let dims = MoeDims::minicpm47(4096);
    let (e, k, h) = (dims.experts, dims.top_k, dims.hidden);
    let fx = reference::fixture(dims, 0);
    let refs: HashMap<usize, RouteRow> = checked_rows(4096)
        .into_iter()
        .map(|row| (row, reference::route_row(&fx, row)))
        .collect();

    let xb = testing::upload_device(&d, fx.x.clone())?;
    let mut rb = testing::upload_device(&d, fx.router_w.clone())?;
    d.finish_upload()?;
    let mut idsb = testing::zeros::<i32>(s, dims.rows * k)?;
    let mut wb = testing::zeros::<bf16>(s, dims.rows * k)?;
    let lb = testing::zeros::<bf16>(s, dims.rows * e)?;

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
        let mut logits = TensorMut(
            testing::dense_mut(&lb, s, dims.rows as u32, e as u32)?
                .0
                .prefix_rows(rows)?,
        );
        ops.route(
            x,
            router,
            &mut ids,
            &mut weights,
            &mut logits,
            crate::model_spec::RouterScore::Softmax,
        )
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

/// Prefill logits match the official BF16 GEMM bitwise; softmax/top-k consume those logits.
const PREFILL_ROW_CASES: [u32; 4] = [8, 37, 512, 4096];

fn official_router_logits(d: &Device, ops: &RouteOps) -> Result<()> {
    use safetensors::{Dtype, SafeTensors};
    use std::path::PathBuf;

    let reference = PathBuf::from(std::env::var("FF_MINICPM_REFERENCE_DIR")?);
    let raw =
        std::fs::read(reference.join(format!("router_logits_sm{}.safetensors", d.info.arch)))?;
    let fixture = SafeTensors::deserialize(&raw)?;
    let layer = fixture.tensor("layer")?;
    ensure!(
        layer.dtype() == Dtype::I32 && layer.shape() == [1],
        "router fixture layer layout"
    );
    let layer = u32::from_le_bytes(layer.data().try_into()?);
    let model = PathBuf::from(std::env::var("FF_MODEL")?);
    let spec = crate::model_spec::ModelSpec::minicpm(
        &serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?,
        &serde_json::from_slice(&std::fs::read(model.join("generation_config.json"))?)?,
    )?;
    let moe = spec
        .moe
        .context("router fixture model has no MoE geometry")?;
    ensure!(
        (layer as usize) < spec.layers.len(),
        "router fixture layer={layer} exceeds model layers={}",
        spec.layers.len()
    );
    let bf16_values = |data: &[u8]| {
        data.as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
            .collect::<Vec<_>>()
    };
    let x = fixture.tensor("x")?;
    ensure!(
        x.dtype() == Dtype::BF16 && x.shape() == [4096, spec.hidden as usize],
        "router fixture x layout {:?}, hidden={}",
        x.shape(),
        spec.hidden
    );
    let checkpoint = crate::checkpoint::MappedCheckpoint::open(&model)?;
    let views = checkpoint.views()?;
    let name = format!("model.language_model.layers.{layer}.mlp.gate.weight");
    let shard = checkpoint
        .index
        .weight_map
        .get(&name)
        .context("router weight absent")?;
    let w = views[shard].tensor(&name)?;
    ensure!(
        w.dtype() == Dtype::BF16 && w.shape() == [moe.experts as usize, spec.hidden as usize],
        "router weight {name} layout {:?}",
        w.shape()
    );
    let s = &d.stream;
    let xb = s.clone_htod(&bf16_values(x.data()))?;
    let wb = s.clone_htod(&bf16_values(w.data()))?;
    let ids = s.alloc_zeros::<i32>(4096 * moe.experts_per_tok as usize)?;
    let weights = s.alloc_zeros::<bf16>(ids.len())?;
    let logits = s.alloc_zeros::<bf16>(4096 * moe.experts as usize)?;
    let mut total = 0;
    for rows in [1u32, 94, 481, 1024, 4096] {
        ops.route_prefill(
            testing::dense(&xb, s, 4096, spec.hidden)?.prefix_rows(rows)?,
            testing::dense(&wb, s, moe.experts, spec.hidden)?,
            &mut TensorMut(testing::dense(&ids, s, 4096, moe.experts_per_tok)?.prefix_rows(rows)?),
            &mut TensorMut(
                testing::dense(&weights, s, 4096, moe.experts_per_tok)?.prefix_rows(rows)?,
            ),
            &mut TensorMut(testing::dense(&logits, s, 4096, moe.experts)?.prefix_rows(rows)?),
            crate::model_spec::RouterScore::Softmax,
        )?;
        let actual = s.clone_dtoh(&logits)?;
        let expected = fixture.tensor(&format!("logits_{rows}"))?;
        ensure!(
            expected.dtype() == Dtype::BF16
                && expected.shape() == [rows as usize, moe.experts as usize],
            "router fixture logits rows={rows} layout {:?}",
            expected.shape()
        );
        let expected = bf16_values(expected.data());
        ensure!(
            actual[..expected.len()]
                .iter()
                .chain(&expected)
                .all(|v| v.is_finite()),
            "router logits rows={rows} are nonfinite"
        );
        let mismatches = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        total += mismatches;
        eprintln!(
            "MOE_ROUTE_OFFICIAL sm={} rows={rows} bit_mismatches={mismatches} {}",
            d.info.arch,
            if mismatches == 0 { "PASS" } else { "DEFECT" }
        );
    }
    ensure!(total == 0, "official router logits bit_mismatches={total}");
    Ok(())
}

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn moe_route_prefill_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    let s = &d.stream;
    let ops = RouteOps::new(&d.ctx, s, 256, 4096)?;
    official_router_logits(&d, &ops)?;
    let dims = MoeDims::minicpm47(4096);
    let (e, k, h) = (dims.experts, dims.top_k, dims.hidden);
    let fx = reference::fixture(dims, 0);
    let fixtures = std::path::PathBuf::from(std::env::var("FF_MINICPM_REFERENCE_DIR")?);
    let raw =
        std::fs::read(fixtures.join(format!("router_synthetic_sm{}.safetensors", d.info.arch)))?;
    let stored = safetensors::SafeTensors::deserialize(&raw)?;
    for (name, values, shape) in [
        ("x", &fx.x, [dims.rows, h]),
        ("router", &fx.router_w, [e, h]),
    ] {
        let tensor = stored.tensor(name)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::BF16 && tensor.shape() == shape,
            "synthetic router {name} fixture layout {:?}",
            tensor.shape()
        );
        ensure!(
            tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .zip(values)
                .all(|(a, b)| u16::from_le_bytes(*a) == b.to_bits()),
            "synthetic router {name} fixture input differs"
        );
    }

    let xb = testing::upload_device(&d, fx.x.clone())?;
    let rb = testing::upload_device(&d, fx.router_w.clone())?;
    d.finish_upload()?;
    let mut idsb = testing::zeros::<i32>(s, dims.rows * k)?;
    let mut wb = testing::zeros::<bf16>(s, dims.rows * k)?;
    let lb = testing::zeros::<bf16>(s, dims.rows * e)?;

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
        let mut logits = TensorMut(
            testing::dense_mut(&lb, s, dims.rows as u32, e as u32)?
                .0
                .prefix_rows(rows)?,
        );
        ops.route_prefill(
            x,
            router,
            &mut ids,
            &mut weights,
            &mut logits,
            crate::model_spec::RouterScore::Softmax,
        )
    };

    for &case in &PREFILL_ROW_CASES {
        s.memcpy_htod(&vec![-1i32; dims.rows * k], &mut idsb)?;
        s.memcpy_htod(&vec![bf16::from_f32(-17.0); dims.rows * k], &mut wb)?;
        run(case, &xb, &rb, &mut idsb, &mut wb)?;
        s.synchronize()?;
        let rows = checked_rows(case);
        let ids = s.clone_dtoh(&idsb)?;
        let weights = s.clone_dtoh(&wb)?;
        let actual = s.clone_dtoh(&lb)?;
        let expected = stored.tensor(&format!("logits_{case}"))?;
        ensure!(
            expected.dtype() == safetensors::Dtype::BF16 && expected.shape() == [case as usize, e],
            "synthetic logits rows={case} fixture layout {:?}",
            expected.shape()
        );
        let logits = expected
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
            .collect::<Vec<_>>();
        let mismatches = actual
            .iter()
            .zip(&logits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "MOE_ROUTE_SYNTHETIC sm={} rows={case} bit_mismatches={mismatches}",
            d.info.arch
        );
        ensure!(
            mismatches == 0,
            "synthetic official logits rows={case} bit_mismatches={mismatches}"
        );
        let refs = rows
            .iter()
            .map(|&row| {
                (
                    row,
                    reference::route_logits(&logits[row * e..(row + 1) * e], k),
                )
            })
            .collect::<HashMap<_, _>>();
        let (a, b) = fx.exact_tie_experts;
        let first = &ids[0..k];
        ensure!(
            first[0] as usize == a && first[1] as usize == b,
            "prefill exact-tie row must return both tie experts with the lower id first: {first:?} vs ({a},{b})"
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
        )?;
    }

    Ok(())
}
