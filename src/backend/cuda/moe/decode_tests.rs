//! CPU validation gates and the GPU correctness/control/timing gate for the
//! decode expert FFN op.
use super::decode::{self, DecodeOps, MoeDecode, decode_scratch_bytes};
use super::reference::{self, MoeDims};
use super::testing;
use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use anyhow::Result;
use cudarc::driver::DevicePtr;
use half::bf16;

#[test]
fn decode_scratch_bytes_sizes_h_and_scale() {
    assert_eq!(decode_scratch_bytes(1, 8, 512), 9 * 512 * 2 + 4);
    assert_eq!(decode_scratch_bytes(8, 8, 512), 8 * 9 * 512 * 2 + 8 * 4);
    assert_eq!(decode_scratch_bytes(3, 4, 64), 3 * 5 * 64 * 2 + 3 * 4);
}

fn cpu_input(y: &mut TensorMut<bf16, 2>) -> Result<MoeDecode<'_>> {
    let (rows, k, experts, h, i) = (8u32, 8u32, 256u32, 2048u32, 512u32);
    unsafe {
        Ok(MoeDecode {
            x: Tensor::new(
                0x0100_0000,
                [rows, h],
                [u64::from(h), 1],
                u64::from(rows) * u64::from(h) * 2,
            )?,
            assignment_gen: Tensor::new(
                0x0f00_0000,
                [rows, k],
                [u64::from(k), 1],
                u64::from(rows) * u64::from(k) * 8,
            )?,
            ids: Tensor::new(
                0x0200_0000,
                [rows, k],
                [u64::from(k), 1],
                u64::from(rows) * u64::from(k) * 4,
            )?,
            active: Tensor::new(
                0x0c00_0000,
                [rows, k],
                [u64::from(k), 1],
                u64::from(rows) * u64::from(k) * 4,
            )?,
            weights: Tensor::new(
                0x0300_0000,
                [rows, k],
                [u64::from(k), 1],
                u64::from(rows) * u64::from(k) * 2,
            )?,
            slots_gate_up: Tensor::new(0x0400_0000, [experts], [1], u64::from(experts) * 8)?,
            slots_down: Tensor::new(0x0500_0000, [experts], [1], u64::from(experts) * 8)?,
            slot_valid: Tensor::new(0x0600_0000, [experts], [1], u64::from(experts) * 4)?,
            slot_generation: Tensor::new(0x0d00_0000, [experts], [1], u64::from(experts) * 8)?,
            shared_gate: Tensor::new(
                0x0700_0000,
                [i, h],
                [u64::from(h), 1],
                u64::from(i) * u64::from(h) * 2,
            )?,
            shared_up: Tensor::new(
                0x0800_0000,
                [i, h],
                [u64::from(h), 1],
                u64::from(i) * u64::from(h) * 2,
            )?,
            shared_down: Tensor::new(
                0x0a00_0000,
                [h, i],
                [u64::from(i), 1],
                u64::from(h) * u64::from(i) * 2,
            )?,
            shared_expert_gate: Tensor::new(0x0b00_0000, [h], [1], u64::from(h) * 2)?,
            y,
        })
    }
}

fn named_error<T>(result: anyhow::Result<T>, needle: &str) {
    let message = match result {
        Ok(_) => panic!("validation must fail for {needle:?}"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains(needle),
        "validation error {message:?} does not name {needle:?}"
    );
}

#[test]
fn decode_validation_names_rejected_bindings() -> Result<()> {
    let mut y =
        unsafe { TensorMut::<bf16, 2>::new(0x0900_0000, [8, 2048], [2048, 1], 8 * 2048 * 2)? };
    let scratch = Scratch {
        ptr: 0x2000_0000,
        bytes: decode_scratch_bytes(8, 8, 512),
    };
    {
        let input = cpu_input(&mut y)?;
        assert!(decode::validate(&input, &scratch).is_ok());
    }

    let mut bad = cpu_input(&mut y)?;
    bad.x = unsafe { Tensor::<bf16, 2>::new(0x0100_0000, [8, 2048], [4096, 1], 8 * 4096 * 2)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode x must be row-contiguous",
    );

    bad = cpu_input(&mut y)?;
    bad.x = unsafe { Tensor::<bf16, 2>::new(0x0100_0000, [9, 2048], [2048, 1], 9 * 2048 * 2)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode rows outside [1,8]",
    );

    bad = cpu_input(&mut y)?;
    bad.ids = unsafe { Tensor::<i32, 2>::new(0x0200_0000, [7, 8], [8, 1], 7 * 8 * 4)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode ids must be row-contiguous",
    );

    bad = cpu_input(&mut y)?;
    bad.active = unsafe { Tensor::<u32, 2>::new(0x0c00_0000, [7, 8], [8, 1], 7 * 8 * 4)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode active must be row-contiguous [rows, top_k]",
    );

    bad = cpu_input(&mut y)?;
    bad.slot_generation = unsafe { Tensor::<u64, 1>::new(0x0d00_0000, [255], [1], 255 * 8)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode slot tables must be contiguous [experts]",
    );

    bad = cpu_input(&mut y)?;
    bad.weights = unsafe { Tensor::<bf16, 2>::new(0x0300_0000, [8, 8], [16, 1], 8 * 16 * 2)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode router weights must be row-contiguous",
    );

    bad = cpu_input(&mut y)?;
    bad.slots_down = unsafe { Tensor::<u64, 1>::new(0x0500_0000, [255], [1], 255 * 8)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode slot tables must be contiguous [experts]",
    );

    bad = cpu_input(&mut y)?;
    bad.slot_valid = unsafe { Tensor::<u32, 1>::new(0x0600_0000, [256], [2], 2048 * 4)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode slot tables must be contiguous [experts]",
    );

    bad = cpu_input(&mut y)?;
    bad.shared_down =
        unsafe { Tensor::<bf16, 2>::new(0x0a00_0000, [512, 2048], [2048, 1], 512 * 2048 * 2)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode shared_down must be row-contiguous [hidden, inter]",
    );

    bad = cpu_input(&mut y)?;
    bad.shared_expert_gate = unsafe { Tensor::<bf16, 1>::new(0x0b00_0000, [2048], [2], 4096 * 2)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode shared expert gate must be a contiguous [hidden] vector",
    );

    bad = cpu_input(&mut y)?;
    bad.shared_gate =
        unsafe { Tensor::<bf16, 2>::new(0x0700_0000, [513, 2048], [2048, 1], 513 * 2048 * 2)? };
    bad.shared_up =
        unsafe { Tensor::<bf16, 2>::new(0x0800_0000, [513, 2048], [2048, 1], 513 * 2048 * 2)? };
    bad.shared_down =
        unsafe { Tensor::<bf16, 2>::new(0x0a00_0000, [2048, 513], [513, 1], 2048 * 513 * 2)? };
    named_error(
        decode::validate(&bad, &scratch),
        "decode inter must be a multiple of 8",
    );

    let short = Scratch {
        ptr: 0x2000_0000,
        bytes: decode_scratch_bytes(8, 8, 512) - 1,
    };
    bad = cpu_input(&mut y)?;
    named_error(
        decode::validate(&bad, &short),
        "scratch alignment/capacity mismatch",
    );

    let mut aliased_y =
        unsafe { TensorMut::<bf16, 2>::new(0x0100_0000, [8, 2048], [2048, 1], 8 * 2048 * 2)? };
    let aliased = cpu_input(&mut aliased_y)?;
    named_error(
        decode::validate(&aliased, &scratch),
        "decode output overlaps x",
    );
    Ok(())
}

fn max_row_diff(got: &[bf16], want: &[bf16]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g.to_f32() - w.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

/// Host mirror of the moe_ffn boundary order: fp32 dot products, BF16
/// rounding at gate/up, silu, the gate·up product, down, each weighted
/// contribution and the sigmoid-gated shared term, routed terms and the
/// shared term combined in FP32.
fn plan_row(fx: &reference::MoeFixture, row: usize) -> Vec<bf16> {
    let (h, i, k) = (fx.dims.hidden, fx.dims.inter, fx.dims.top_k);
    let bf = bf16::from_f32;
    let dot = |a: &[bf16], b: &[bf16]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&p, &q)| p.to_f32() * q.to_f32())
            .sum()
    };
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let xr = &fx.x[row * h..(row + 1) * h];
    let routed = reference::route_row(fx, row);
    let mut total = vec![0.0f32; h];
    for ki in 0..k {
        let e = routed.ids[ki] as usize;
        let mut p = vec![bf16::from_f32(0.0); i];
        for (j, out) in p.iter_mut().enumerate() {
            let g = bf(dot(xr, fx.gate_row(e, j)));
            let u = bf(dot(xr, fx.up_row(e, j)));
            let s = bf(silu(g.to_f32()));
            *out = bf(s.to_f32() * u.to_f32());
        }
        for (d, acc) in total.iter_mut().enumerate() {
            let lin = bf(dot(&p, fx.down_row(e, d)));
            let contribution = bf(lin.to_f32() * routed.weights[ki].to_f32());
            *acc += contribution.to_f32();
        }
    }
    let mut ps = vec![bf16::from_f32(0.0); i];
    for (j, out) in ps.iter_mut().enumerate() {
        let g = bf(dot(xr, &fx.shared_gate[j * h..(j + 1) * h]));
        let u = bf(dot(xr, &fx.shared_up[j * h..(j + 1) * h]));
        let s = bf(silu(g.to_f32()));
        *out = bf(s.to_f32() * u.to_f32());
    }
    let gated_input = bf(dot(xr, &fx.shared_expert_gate)).to_f32();
    let scale = bf(1.0 / (1.0 + (-gated_input).exp()));
    for (d, acc) in total.iter_mut().enumerate() {
        let so = bf(dot(&ps, &fx.shared_down[d * i..(d + 1) * i]));
        *acc += bf(scale.to_f32() * so.to_f32()).to_f32();
    }
    total.iter().map(|&v| bf(v)).collect()
}

#[test]
fn decode_numeric_plan_lands_within_reference_tolerance() {
    let fx = reference::fixture(reference::MoeDims::new(4, 16, 4, 128, 64), 1);
    let tol = reference::tolerance(&fx, &[0, 1, 2, 3]);
    for row in 0..4 {
        let want = reference::moe_row(&fx, row);
        let diff = max_row_diff(&plan_row(&fx, row), &want);
        assert!(
            diff <= tol,
            "row {row}: plan diff {diff:e} exceeds tolerance {tol:e}"
        );
    }
}

#[test]
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn decode_expert_ffn_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    let ops = DecodeOps::new(&device.ctx)?;
    let s = &device.stream;
    let fx = reference::fixture(MoeDims::minicpm47(8), 0);
    let (e, k, h, i) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let (eu, ku, hu, iu) = (e as u32, k as u32, h as u32, i as u32);

    let x_dev = testing::upload(&device.upload, fx.x.clone())?;
    let gu_dev = testing::upload(&device.upload, fx.gate_up.clone())?;
    let dn_dev = testing::upload(&device.upload, fx.down.clone())?;
    let sg_dev = testing::upload(&device.upload, fx.shared_gate.clone())?;
    let su_dev = testing::upload(&device.upload, fx.shared_up.clone())?;
    let sd_dev = testing::upload(&device.upload, fx.shared_down.clone())?;
    let se_dev = testing::upload(&device.upload, fx.shared_expert_gate.clone())?;
    device.finish_upload()?;

    let mut ids = Vec::with_capacity(8 * k);
    let mut weights = Vec::with_capacity(8 * k);
    for r in 0..8 {
        let routed = reference::route_row(&fx, r);
        ids.extend_from_slice(&routed.ids);
        weights.extend_from_slice(&routed.weights);
    }
    super::route::ensure_distinct_row_experts(&ids, k)?;
    let ids_dev = testing::upload(s, ids)?;
    let weights_dev = testing::upload(s, weights)?;
    let y_dev = testing::zeros::<bf16>(s, 8 * h)?;

    let (gu_base, dn_base) = (gu_dev.device_ptr(s).0, dn_dev.device_ptr(s).0);
    let (gu_stride, dn_stride) = ((2 * i * h * 2) as u64, ((h * i * 2) as u64));
    let table = testing::slots(s, e, gu_base, gu_stride, dn_base, dn_stride)?;
    let gu_view = testing::view(&table.gate_up, s, [eu], [1])?;
    let dn_view = testing::view(&table.down, s, [eu], [1])?;
    let valid_view = testing::view(&table.valid, s, [eu], [1])?;
    let x_full = testing::dense(&x_dev, s, 8, hu)?;
    let ids_full = testing::dense(&ids_dev, s, 8, ku)?;
    let weights_full = testing::dense(&weights_dev, s, 8, ku)?;
    let active_dev = testing::upload(s, vec![1u32; 8 * k])?;
    let active_full = testing::dense(&active_dev, s, 8, ku)?;
    let gen_view = testing::view(&table.generation, s, [eu], [1])?;
    let shared_gate_view = testing::dense(&sg_dev, s, iu, hu)?;
    let shared_up_view = testing::dense(&su_dev, s, iu, hu)?;
    let shared_down_view = testing::dense(&sd_dev, s, hu, iu)?;
    let seg_view = testing::view(&se_dev, s, [hu], [1])?;
    let scratch_mem = testing::zeros::<u8>(s, decode_scratch_bytes(8, k, i) as usize)?;
    let scratch = Scratch {
        ptr: scratch_mem.device_ptr(s).0,
        bytes: decode_scratch_bytes(8, k, i),
    };

    let agen1_dev = testing::upload(s, vec![1u64; 8 * ku as usize])?;
    let agen_full = testing::dense(&agen1_dev, s, 8, ku)?;
    let launch = |rows: u32,
                  gu: Tensor<u64, 1>,
                  dn: Tensor<u64, 1>,
                  slot_gen: Tensor<u64, 1>,
                  active: Tensor<u32, 2>,
                  agen: Tensor<u64, 2>,
                  y: &mut TensorMut<bf16, 2>|
     -> Result<()> {
        ops.decode(
            s,
            MoeDecode {
                x: x_full.prefix_rows(rows)?,
                ids: ids_full.prefix_rows(rows)?,
                active: active.prefix_rows(rows)?,
                weights: weights_full.prefix_rows(rows)?,
                slots_gate_up: gu,
                slots_down: dn,
                slot_valid: valid_view,
                slot_generation: slot_gen,
                assignment_gen: agen.prefix_rows(rows)?,
                shared_gate: shared_gate_view,
                shared_up: shared_up_view,
                shared_down: shared_down_view,
                shared_expert_gate: seg_view,
                y,
            },
            &scratch,
        )
    };

    if std::env::var("FF_MOE_TRAP_CHILD").as_deref() == Ok("1") {
        let mut y = testing::mut_view(&y_dev, s, [1, hu], [u64::from(hu), 1])?;
        let agen3 = testing::upload(s, vec![3u64; 8 * ku as usize])?;
        let agen3_view = testing::dense(&agen3, s, 8, ku)?;
        launch(
            1,
            gu_view,
            dn_view,
            gen_view,
            active_full,
            agen3_view,
            &mut y,
        )?;
        s.synchronize()?;
        return Ok(());
    }
    if std::env::var("FF_MOE_DUP_CHILD").as_deref() == Ok("1") {
        let mut dup = Vec::with_capacity(8 * ku as usize);
        for r in 0..8u32 {
            let routed = reference::route_row(&fx, r as usize);
            dup.extend_from_slice(&routed.ids);
        }
        dup[1] = dup[0];
        let dup_dev = testing::upload(s, dup)?;
        let dup_view = testing::dense(&dup_dev, s, 8, ku)?;
        let mut y = testing::mut_view(&y_dev, s, [1, hu], [u64::from(hu), 1])?;
        ops.decode(
            s,
            MoeDecode {
                x: x_full.prefix_rows(1)?,
                ids: dup_view.prefix_rows(1)?,
                active: active_full.prefix_rows(1)?,
                weights: weights_full.prefix_rows(1)?,
                slots_gate_up: gu_view,
                slots_down: dn_view,
                slot_valid: valid_view,
                slot_generation: gen_view,
                assignment_gen: agen_full.prefix_rows(1)?,
                shared_gate: shared_gate_view,
                shared_up: shared_up_view,
                shared_down: shared_down_view,
                shared_expert_gate: seg_view,
                y: &mut y,
            },
            &scratch,
        )?;
        s.synchronize()?;
        return Ok(());
    }

    for rows in [1u32, 2, 5, 8] {
        let mut y = testing::mut_view(&y_dev, s, [rows, hu], [u64::from(hu), 1])?;
        launch(
            rows,
            gu_view,
            dn_view,
            gen_view,
            active_full,
            agen_full,
            &mut y,
        )?;
        let got = s.clone_dtoh(&y_dev)?;
        let sample: Vec<usize> = (0..rows as usize).collect();
        let tol = reference::tolerance(&fx, &sample);
        let mut worst = 0.0f32;
        for r in 0..rows as usize {
            let want = reference::moe_row(&fx, r);
            let diff = max_row_diff(&got[r * h..(r + 1) * h], &want);
            worst = worst.max(diff);
            anyhow::ensure!(
                diff <= tol,
                "decode rows={rows} row={r} max diff {diff:e} exceeds tolerance {tol:e}"
            );
            println!("MOE_DECODE_SPREAD rows={rows} row={r} max_diff={diff:e} tol={tol:e}");
        }
        println!("MOE_DECODE rows={rows} worst_diff={worst:e} tol={tol:e} PASS");
    }

    let tol = reference::tolerance(&fx, &[0]);
    {
        let dropped = [2usize, 5];
        let keep: Vec<usize> = (0..k).filter(|c| !dropped.contains(c)).collect();
        let mut partial = vec![1u32; 8 * k];
        for &c in &dropped {
            partial[c] = 0;
        }
        let partial_dev = testing::upload(s, partial)?;
        let partial_view = testing::dense(&partial_dev, s, 8, ku)?;
        let mut y = testing::mut_view(&y_dev, s, [1, hu], [u64::from(hu), 1])?;
        launch(
            1,
            gu_view,
            dn_view,
            gen_view,
            partial_view,
            agen_full,
            &mut y,
        )?;
        let got = s.clone_dtoh(&y_dev)?;
        let want = reference::moe_row_choices(&fx, 0, &keep);
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff <= tol,
            "partial-active row 0 max diff {diff:e} exceeds tolerance {tol:e}"
        );
        println!("MOE_DECODE_PARTIAL row=0 dropped={dropped:?} max_diff={diff:e} tol={tol:e} PASS");
    }
    {
        // Swap the two active choices with the largest weight spread; a
        // same-weight pair leaves the weighted sum invariant under swap.
        let routed = reference::route_row(&fx, 0);
        let mut pair = (0usize, 1usize);
        let mut spread = 0.0f32;
        for c0 in 0..routed.weights.len() {
            for c1 in c0 + 1..routed.weights.len() {
                let d = (routed.weights[c0].to_f32() - routed.weights[c1].to_f32()).abs();
                if d > spread {
                    spread = d;
                    pair = (c0, c1);
                }
            }
        }
        anyhow::ensure!(spread > 0.0, "swap control requires distinct weights");
        let (a, b) = (routed.ids[pair.0] as usize, routed.ids[pair.1] as usize);
        let mut gu_tab: Vec<u64> = (0..e).map(|ex| gu_base + ex as u64 * gu_stride).collect();
        let mut dn_tab: Vec<u64> = (0..e).map(|ex| dn_base + ex as u64 * dn_stride).collect();
        gu_tab.swap(a, b);
        dn_tab.swap(a, b);
        let gu_sw = testing::upload(s, gu_tab)?;
        let dn_sw = testing::upload(s, dn_tab)?;
        let gu_sw_view = testing::view(&gu_sw, s, [eu], [1])?;
        let dn_sw_view = testing::view(&dn_sw, s, [eu], [1])?;
        let mut y = testing::mut_view(&y_dev, s, [1, hu], [u64::from(hu), 1])?;
        launch(
            1,
            gu_sw_view,
            dn_sw_view,
            gen_view,
            active_full,
            agen_full,
            &mut y,
        )?;
        let got = s.clone_dtoh(&y_dev)?;
        let want = reference::moe_row(&fx, 0);
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff > tol,
            "swapped slot-table control undetected: diff {diff:e} within tolerance {tol:e}"
        );
        println!("MOE_DECODE_CONTROL swapped_slots diff={diff:e} tol={tol:e} DETECTED");
    }
    {
        let mut rolled = fx.gate_up.clone();
        let block = 2 * i * h;
        for ex in 0..e {
            let base = ex * block;
            let first = rolled[base..base + h].to_vec();
            rolled.copy_within(base + h..base + block, base);
            rolled[base + block - h..base + block].copy_from_slice(&first);
        }
        let rolled_dev = testing::upload(&device.upload, rolled)?;
        device.finish_upload()?;
        let rolled_base = rolled_dev.device_ptr(s).0;
        let rolled_table = testing::slots(s, e, rolled_base, gu_stride, dn_base, dn_stride)?;
        let gu_rolled = testing::view(&rolled_table.gate_up, s, [eu], [1])?;
        let dn_rolled = testing::view(&rolled_table.down, s, [eu], [1])?;
        let gen_rolled = testing::view(&rolled_table.generation, s, [eu], [1])?;
        let mut y = testing::mut_view(&y_dev, s, [1, hu], [u64::from(hu), 1])?;
        launch(
            1,
            gu_rolled,
            dn_rolled,
            gen_rolled,
            active_full,
            agen_full,
            &mut y,
        )?;
        let got = s.clone_dtoh(&y_dev)?;
        let want = reference::moe_row(&fx, 0);
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff > tol,
            "gate/up off-by-one control undetected: diff {diff:e} within tolerance {tol:e}"
        );
        println!("MOE_DECODE_CONTROL gate_up_rolled diff={diff:e} tol={tol:e} DETECTED");
    }
    {
        let victim = reference::route_row(&fx, 0).ids[0] as usize;
        let mut fx2 = reference::fixture(MoeDims::minicpm47(8), 0);
        let scaled = |v: bf16| bf16::from_f32(v.to_f32() * 1.25);
        for v in fx2.gate_up[victim * 2 * i * h..(victim + 1) * 2 * i * h].iter_mut() {
            *v = scaled(*v);
        }
        for v in fx2.down[victim * h * i..(victim + 1) * h * i].iter_mut() {
            *v = scaled(*v);
        }
        let gu_block = fx2.gate_up[victim * 2 * i * h..(victim + 1) * 2 * i * h].to_vec();
        let dn_block = fx2.down[victim * h * i..(victim + 1) * h * i].to_vec();
        let gu_ev = testing::upload(s, gu_block)?;
        let dn_ev = testing::upload(s, dn_block)?;
        let mut gu_tab: Vec<u64> = (0..e).map(|ex| gu_base + ex as u64 * gu_stride).collect();
        let mut dn_tab: Vec<u64> = (0..e).map(|ex| dn_base + ex as u64 * dn_stride).collect();
        gu_tab[victim] = gu_ev.device_ptr(s).0;
        dn_tab[victim] = dn_ev.device_ptr(s).0;
        let gen2 = testing::upload(s, vec![2u64; e])?;
        let gu_ev_tab = testing::upload(s, gu_tab)?;
        let dn_ev_tab = testing::upload(s, dn_tab)?;
        let gu_ev_view = testing::view(&gu_ev_tab, s, [eu], [1])?;
        let dn_ev_view = testing::view(&dn_ev_tab, s, [eu], [1])?;
        let gen2_view = testing::view(&gen2, s, [eu], [1])?;
        let mut y = testing::mut_view(&y_dev, s, [1, hu], [u64::from(hu), 1])?;
        let agen2 = testing::upload(s, vec![2u64; 8 * ku as usize])?;
        let agen2_view = testing::dense(&agen2, s, 8, ku)?;
        launch(
            1,
            gu_ev_view,
            dn_ev_view,
            gen2_view,
            active_full,
            agen2_view,
            &mut y,
        )?;
        let got = s.clone_dtoh(&y_dev)?;
        let want = reference::moe_row(&fx2, 0);
        let diff = max_row_diff(&got[..h], &want);
        let tol2 = reference::tolerance(&fx2, &[0]);
        anyhow::ensure!(
            diff <= tol2,
            "eviction-reuse row 0 max diff {diff:e} exceeds tolerance {tol2:e}"
        );
        println!(
            "MOE_DECODE_EVICT victim={victim} generation=2 max_diff={diff:e} tol={tol2:e} PASS"
        );
    }
    println!("MOE_DECODE_GPU rows=1,2,5,8 partial=1 evict=1 controls=2/2 PASS");

    if std::env::var("FF_MOE_TIMING").as_deref() == Ok("1") {
        use cudarc::driver::sys::CUevent_flags;
        let begin = device
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = device
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let per_row = (k * (2 * i * h + h * i) + 2 * i * h + h * i) as u64 * 2;
        let copy_src = testing::zeros::<u8>(s, (8 * per_row) as usize)?;
        let copy_dst = testing::zeros::<u8>(s, (8 * per_row) as usize)?;
        let (src_ptr, dst_ptr) = (copy_src.device_ptr(s).0, copy_dst.device_ptr(s).0);
        let iters = 100u32;
        let reps = 20u32;
        for rows in 1..=8u32 {
            let mut y = testing::mut_view(&y_dev, s, [rows, hu], [u64::from(hu), 1])?;
            for _ in 0..10 {
                launch(
                    rows,
                    gu_view,
                    dn_view,
                    gen_view,
                    active_full,
                    agen_full,
                    &mut y,
                )?;
            }
            s.synchronize()?;
            begin.record(s)?;
            for _ in 0..iters {
                launch(
                    rows,
                    gu_view,
                    dn_view,
                    gen_view,
                    active_full,
                    agen_full,
                    &mut y,
                )?;
            }
            end.record(s)?;
            s.synchronize()?;
            let launch_us = f64::from(begin.elapsed_ms(&end)?) * 1000.0 / f64::from(iters);
            let bytes = u64::from(rows) * per_row;
            let gbps = bytes as f64 / (launch_us * 1e-6) / 1e9;
            begin.record(s)?;
            for _ in 0..reps {
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst_ptr,
                        src_ptr,
                        bytes as usize,
                        s.cu_stream(),
                    )?;
                }
            }
            end.record(s)?;
            s.synchronize()?;
            let copy_us = f64::from(begin.elapsed_ms(&end)?) * 1000.0 / f64::from(reps);
            let copy_gbps = bytes as f64 / (copy_us * 1e-6) / 1e9;
            println!(
                "MOE_DECODE_TIMING rows={rows} per_launch_us={launch_us:.2} bytes={bytes} achieved_GBps={gbps:.0} d2d_copy_us={copy_us:.2} d2d_GBps={copy_gbps:.0}"
            );
        }
    }
    Ok(())
}

// The device-side trap poisons concurrent contexts on this driver, so the
// stale-generation child runs from this dedicated test (no CUDA launches in
// the parent) and the gate invocation skips it via --skip stale_generation.
#[test]
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn decode_stale_generation_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(["decode_expert_ffn_gpu", "--ignored", "--test-threads=1"])
        .env("FF_MOE_TRAP_CHILD", "1")
        .env("FF_MOE_GPU_WINDOW", "granted")
        .status()?;
    anyhow::ensure!(
        !status.success(),
        "stale-generation child exited successfully"
    );
    println!("MOE_DECODE_CONTROL stale_generation child={status} TRAPPED");
    Ok(())
}

// Parent for the FF_MOE_DUP_CHILD branch of decode_expert_ffn_gpu; no CUDA
// call in the parent.
#[test]
#[ignore = "requires GPU window, FF_MOE_GPU_WINDOW=granted"]
fn decode_duplicate_ids_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_MOE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args(["decode_expert_ffn_gpu", "--ignored", "--test-threads=1"])
        .env("FF_MOE_GPU_WINDOW", "granted")
        .env("FF_MOE_DUP_CHILD", "1")
        .status()?;
    anyhow::ensure!(!status.success(), "duplicate-ids child exited successfully");
    println!("MOE_DECODE_CONTROL duplicate_ids child={status} TRAPPED");
    Ok(())
}
