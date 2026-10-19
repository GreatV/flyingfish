//! Route-op gates: reference layout checks and the GPU window gate for
//! kernels/moe_route.cu. Default cargo test never initializes a GPU here.
use super::reference::{self, MoeDims, RouteRow};
use super::route::RouteOps;
use super::testing;
use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::TensorMut;
use anyhow::{Context, Result, ensure};
use cudarc::driver::CudaSlice;
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

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn moe_route_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    let s = &d.stream;
    let ops = RouteOps::new(&d.ctx, s)?;
    let dims = MoeDims::minicpm47(4096);
    let (e, k, h) = (dims.experts, dims.top_k, dims.hidden);
    let fx = reference::fixture(dims, 0);
    let refs: HashMap<usize, RouteRow> = checked_rows(4096)
        .into_iter()
        .map(|row| (row, reference::route_row(&fx, row)))
        .collect();

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
            s,
            testing::dense(&xb, s, 4096, spec.hidden)?.prefix_rows(rows)?,
            testing::dense(&wb, s, moe.experts, spec.hidden)?,
            &mut TensorMut(testing::dense(&ids, s, 4096, moe.experts_per_tok)?.prefix_rows(rows)?),
            &mut TensorMut(
                testing::dense(&weights, s, 4096, moe.experts_per_tok)?.prefix_rows(rows)?,
            ),
            &mut TensorMut(testing::dense(&logits, s, 4096, moe.experts)?.prefix_rows(rows)?),
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
    let ops = RouteOps::new(&d.ctx, s)?;
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

    let xb = testing::upload(&d.upload, fx.x.clone())?;
    let rb = testing::upload(&d.upload, fx.router_w.clone())?;
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
        ops.route_prefill(s, x, router, &mut ids, &mut weights, &mut logits)
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
