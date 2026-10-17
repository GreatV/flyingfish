//! GPU correctness, control and trap gates for the decode contribute and
//! combine ops.
use super::combine::{CombineOps, MoeCombine};
use super::decode::{DecodeOps, MoeDecode, decode_scratch_bytes};
use super::reference::{self, Combine, MoeDims};
use super::testing;
use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use anyhow::Result;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DriverError};
use half::bf16;
use std::process::Command;
use std::sync::Arc;

#[test]
fn decode_scratch_bytes_sizes_h_shared_and_scale() {
    assert_eq!(decode_scratch_bytes(1, 8, 512), 9 * 512 * 2 + 2);
    assert_eq!(decode_scratch_bytes(8, 8, 512), 8 * 9 * 512 * 2 + 8 * 2);
    assert_eq!(decode_scratch_bytes(3, 4, 64), 3 * 5 * 64 * 2 + 3 * 2);
}

fn max_row_diff(got: &[bf16], want: &[bf16]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g.to_f32() - w.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

/// Per-launch output buffers: contributions [rows, top_k, hidden],
/// shared_out [rows, hidden], done [rows, top_k] (zeroed) and y [rows, hidden].
struct DecodeIo {
    contributions: Tensor<bf16, 3>,
    shared_out: TensorMut<bf16, 2>,
    done: TensorMut<u32, 2>,
    y: TensorMut<bf16, 2>,
    _contrib_dev: CudaSlice<bf16>,
    _shared_out_dev: CudaSlice<bf16>,
    _done_dev: CudaSlice<u32>,
    y_dev: CudaSlice<bf16>,
}

fn decode_io(s: &Arc<CudaStream>, rows: u32, top_k: u32, hidden: u32) -> Result<DecodeIo> {
    let contrib_dev = testing::zeros::<bf16>(s, rows as usize * top_k as usize * hidden as usize)?;
    let contributions = testing::view(
        &contrib_dev,
        s,
        [rows, top_k, hidden],
        [u64::from(top_k) * u64::from(hidden), u64::from(hidden), 1],
    )?;
    let shared_out_dev = testing::zeros::<bf16>(s, rows as usize * hidden as usize)?;
    let shared_out = testing::mut_view(&shared_out_dev, s, [rows, hidden], [u64::from(hidden), 1])?;
    let done_dev = testing::zeros::<u32>(s, rows as usize * top_k as usize)?;
    let done = testing::mut_view(&done_dev, s, [rows, top_k], [u64::from(top_k), 1])?;
    let y_dev = testing::zeros::<bf16>(s, rows as usize * hidden as usize)?;
    let y = testing::mut_view(&y_dev, s, [rows, hidden], [u64::from(hidden), 1])?;
    Ok(DecodeIo {
        contributions,
        shared_out,
        done,
        y,
        _contrib_dev: contrib_dev,
        _shared_out_dev: shared_out_dev,
        _done_dev: done_dev,
        y_dev,
    })
}

/// Per-arm routed bindings: ids/active/weights/assignment_gen views (rows
/// narrowed per launch) and the two slot-table/generation views.
struct Routing {
    ids: Tensor<i32, 2>,
    active: Tensor<u32, 2>,
    weights: Tensor<bf16, 2>,
    agen: Tensor<u64, 2>,
    gu: Tensor<u64, 1>,
    dn: Tensor<u64, 1>,
    slot_gen: Tensor<u64, 1>,
}

struct Gate<'a> {
    ops: &'a DecodeOps,
    combine: &'a CombineOps,
    s: &'a Arc<CudaStream>,
    x: Tensor<bf16, 2>,
    shared_gate: Tensor<bf16, 2>,
    shared_up: Tensor<bf16, 2>,
    shared_down: Tensor<bf16, 2>,
    seg: Tensor<bf16, 1>,
    valid: Tensor<u32, 1>,
    scratch: Scratch,
    top_k: u32,
    hidden: u32,
    inter: u32,
}

impl Gate<'_> {
    fn input<'a>(
        &self,
        rows: u32,
        rt: &Routing,
        contributions: Tensor<bf16, 3>,
        shared_out: &'a mut TensorMut<bf16, 2>,
        done: &'a mut TensorMut<u32, 2>,
    ) -> Result<MoeDecode<'a>> {
        Ok(MoeDecode {
            x: self.x.prefix_rows(rows)?,
            ids: rt.ids.prefix_rows(rows)?,
            active: rt.active.prefix_rows(rows)?,
            weights: rt.weights.prefix_rows(rows)?,
            slots_gate_up: rt.gu,
            slots_down: rt.dn,
            slot_valid: self.valid,
            slot_generation: rt.slot_gen,
            assignment_gen: rt.agen.prefix_rows(rows)?,
            shared_gate: self.shared_gate,
            shared_up: self.shared_up,
            shared_down: self.shared_down,
            shared_expert_gate: self.seg,
            contributions,
            shared_out,
            done,
        })
    }

    /// contribute + shared_scale + combine through DecodeOps::decode.
    fn run(&self, rows: u32, rt: &Routing) -> Result<Vec<bf16>> {
        let mut io = decode_io(self.s, rows, self.top_k, self.hidden)?;
        {
            let input = self.input(rows, rt, io.contributions, &mut io.shared_out, &mut io.done)?;
            self.ops.decode(self.s, &input, &mut io.y, &self.scratch)?;
        }
        self.s.synchronize()?;
        Ok(self.s.clone_dtoh(&io.y_dev)?)
    }

    /// contribute, then shared_scale and combine called directly.
    fn run_split(&self, rows: u32, rt: &Routing) -> Result<Vec<bf16>> {
        let mut io = decode_io(self.s, rows, self.top_k, self.hidden)?;
        {
            let input = self.input(rows, rt, io.contributions, &mut io.shared_out, &mut io.done)?;
            self.ops.contribute(self.s, &input, &self.scratch)?;
            let scale_ptr = self.scratch.ptr
                + u64::from(rows) * u64::from(self.top_k + 1) * u64::from(self.inter) * 2;
            let mut scale = unsafe {
                TensorMut::<bf16, 2>::new(scale_ptr, [rows, 1], [1, 1], u64::from(rows) * 2)?
            };
            self.combine
                .shared_scale(self.s, input.x, input.shared_expert_gate, &mut scale)?;
            let combine_input = MoeCombine {
                ids: input.ids,
                final_active: input.active,
                done: input.done.read(),
                contributions: input.contributions,
                shared_out: input.shared_out.read(),
                x: input.x,
                shared_expert_gate: input.shared_expert_gate,
                scale,
                y: TensorMut(io.y.read()),
            };
            self.combine.combine(self.s, &combine_input)?;
        }
        self.s.synchronize()?;
        Ok(self.s.clone_dtoh(&io.y_dev)?)
    }

    /// contribute only, synchronized; trap children discriminate the error.
    fn contribute_sync(&self, rows: u32, rt: &Routing) -> Result<()> {
        let mut io = decode_io(self.s, rows, self.top_k, self.hidden)?;
        let input = self.input(rows, rt, io.contributions, &mut io.shared_out, &mut io.done)?;
        self.ops.contribute(self.s, &input, &self.scratch)?;
        Ok(self.s.synchronize()?)
    }
}

/// A device-side trap aborts the launch and surfaces as
/// CUDA_ERROR_LAUNCH_FAILED; any other outcome (including success) fails the
/// child with exit 101 so the parent rejects it.
fn trap_exit(outcome: Result<()>) -> ! {
    match outcome {
        Ok(()) => {
            eprintln!("FF_MOE_CHILD: launch completed; trap did not fire");
            std::process::exit(101);
        }
        Err(error) => match error.downcast_ref::<DriverError>() {
            Some(&DriverError(CUresult::CUDA_ERROR_LAUNCH_FAILED)) => std::process::exit(3),
            _ => {
                eprintln!("FF_MOE_CHILD: {error:?}");
                std::process::exit(101);
            }
        },
    }
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn decode_expert_ffn_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    let ops = DecodeOps::new(&device.ctx)?;
    let combine_ops = CombineOps::new(&device.ctx)?;
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

    let mut ids_host = Vec::with_capacity(8 * k);
    let mut weights_host = Vec::with_capacity(8 * k);
    for r in 0..8 {
        let routed = reference::route_row(&fx, r);
        ids_host.extend_from_slice(&routed.ids);
        weights_host.extend_from_slice(&routed.weights);
    }
    let ids_dev = testing::upload(s, ids_host.clone())?;
    let weights_dev = testing::upload(s, weights_host.clone())?;
    let keep_all: Vec<usize> = (0..k).collect();

    let (gu_base, dn_base) = (gu_dev.device_ptr(s).0, dn_dev.device_ptr(s).0);
    let (gu_stride, dn_stride) = ((2 * i * h * 2) as u64, ((h * i * 2) as u64));
    let table = testing::slots(s, e, gu_base, gu_stride, dn_base, dn_stride)?;
    let gu_view = testing::view(&table.gate_up, s, [eu], [1])?;
    let dn_view = testing::view(&table.down, s, [eu], [1])?;
    let valid_view = testing::view(&table.valid, s, [eu], [1])?;
    let gen_view = testing::view(&table.generation, s, [eu], [1])?;
    let x_full = testing::dense(&x_dev, s, 8, hu)?;
    let ids_full = testing::dense(&ids_dev, s, 8, ku)?;
    let weights_full = testing::dense(&weights_dev, s, 8, ku)?;
    let active_dev = testing::upload(s, vec![1u32; 8 * k])?;
    let active_full = testing::dense(&active_dev, s, 8, ku)?;
    let agen1_dev = testing::upload(s, vec![1u64; 8 * k])?;
    let agen1_full = testing::dense(&agen1_dev, s, 8, ku)?;
    let shared_gate_view = testing::dense(&sg_dev, s, iu, hu)?;
    let shared_up_view = testing::dense(&su_dev, s, iu, hu)?;
    let shared_down_view = testing::dense(&sd_dev, s, hu, iu)?;
    let seg_view = testing::view(&se_dev, s, [hu], [1])?;
    let scratch_mem = testing::zeros::<u8>(s, decode_scratch_bytes(8, k, i) as usize)?;
    let scratch = Scratch {
        ptr: scratch_mem.device_ptr(s).0,
        bytes: decode_scratch_bytes(8, k, i),
    };
    let gate = Gate {
        ops: &ops,
        combine: &combine_ops,
        s,
        x: x_full,
        shared_gate: shared_gate_view,
        shared_up: shared_up_view,
        shared_down: shared_down_view,
        seg: seg_view,
        valid: valid_view,
        scratch,
        top_k: ku,
        hidden: hu,
        inter: iu,
    };

    match std::env::var("FF_MOE_CHILD").as_deref() {
        Ok("stale") => {
            let victim = ids_host[0] as usize;
            let mut gens = vec![1u64; e];
            gens[victim] = 2;
            let gens_dev = testing::upload(s, gens)?;
            let gens_view = testing::view(&gens_dev, s, [eu], [1])?;
            let rt = Routing {
                ids: ids_full.prefix_rows(1)?,
                active: active_full.prefix_rows(1)?,
                weights: weights_full.prefix_rows(1)?,
                agen: agen1_full.prefix_rows(1)?,
                gu: gu_view,
                dn: dn_view,
                slot_gen: gens_view,
            };
            trap_exit(gate.contribute_sync(1, &rt));
        }
        Ok("dup") => {
            let mut dup_ids = ids_host[..k].to_vec();
            dup_ids[1] = dup_ids[0];
            let dup_dev = testing::upload(s, dup_ids)?;
            let dup_view = testing::dense(&dup_dev, s, 1, ku)?;
            let rt = Routing {
                ids: dup_view,
                active: active_full.prefix_rows(1)?,
                weights: weights_full.prefix_rows(1)?,
                agen: agen1_full.prefix_rows(1)?,
                gu: gu_view,
                dn: dn_view,
                slot_gen: gen_view,
            };
            trap_exit(gate.contribute_sync(1, &rt));
        }
        Ok(other) => {
            eprintln!("FF_MOE_CHILD={other}: unknown trap arm");
            std::process::exit(101);
        }
        Err(_) => {}
    }

    for rows in [1u32, 2, 5, 8] {
        let rt = Routing {
            ids: ids_full.prefix_rows(rows)?,
            active: active_full.prefix_rows(rows)?,
            weights: weights_full.prefix_rows(rows)?,
            agen: agen1_full.prefix_rows(rows)?,
            gu: gu_view,
            dn: dn_view,
            slot_gen: gen_view,
        };
        let got = gate.run(rows, &rt)?;
        let sample: Vec<usize> = (0..rows as usize).collect();
        let tol = reference::tolerance(&fx, &sample);
        let mut worst = 0.0f32;
        for r in 0..rows as usize {
            let want = reference::moe_row_ids(
                &fx,
                r,
                &ids_host[r * k..(r + 1) * k],
                &weights_host[r * k..(r + 1) * k],
                &keep_all,
                Combine::Fp32,
            );
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

    {
        let dropped = [2usize, 5];
        let keep: Vec<usize> = (0..k).filter(|c| !dropped.contains(c)).collect();
        let mut partial = vec![1u32; 8 * k];
        for &c in &dropped {
            partial[c] = 0;
        }
        let partial_dev = testing::upload(s, partial)?;
        let partial_view = testing::dense(&partial_dev, s, 8, ku)?;
        let rt = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: partial_view.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen1_full.prefix_rows(1)?,
            gu: gu_view,
            dn: dn_view,
            slot_gen: gen_view,
        };
        let got = gate.run(1, &rt)?;
        let want = reference::moe_row_ids(
            &fx,
            0,
            &ids_host[..k],
            &weights_host[..k],
            &keep,
            Combine::Fp32,
        );
        let tol = reference::tolerance(&fx, &[0]);
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff <= tol,
            "partial-active row 0 (dropped {dropped:?}) max diff {diff:e} exceeds tolerance {tol:e}"
        );
        println!("MOE_DECODE_PARTIAL row=0 dropped={dropped:?} max_diff={diff:e} tol={tol:e} PASS");
    }

    let tol0 = reference::tolerance(&fx, &[0]);
    {
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
        let rt = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: active_full.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen1_full.prefix_rows(1)?,
            gu: gu_sw_view,
            dn: dn_sw_view,
            slot_gen: gen_view,
        };
        let got = gate.run(1, &rt)?;
        let want = reference::moe_row_ids(
            &fx,
            0,
            &ids_host[..k],
            &weights_host[..k],
            &keep_all,
            Combine::Fp32,
        );
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff > tol0,
            "swapped slot-table control undetected: diff {diff:e} within tolerance {tol0:e}"
        );
        println!("MOE_DECODE_CONTROL swapped_slots diff={diff:e} tol={tol0:e} DETECTED");
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
        let rt = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: active_full.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen1_full.prefix_rows(1)?,
            gu: gu_rolled,
            dn: dn_rolled,
            slot_gen: gen_rolled,
        };
        let got = gate.run(1, &rt)?;
        let want = reference::moe_row_ids(
            &fx,
            0,
            &ids_host[..k],
            &weights_host[..k],
            &keep_all,
            Combine::Fp32,
        );
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff > tol0,
            "gate/up off-by-one control undetected: diff {diff:e} within tolerance {tol0:e}"
        );
        println!("MOE_DECODE_CONTROL gate_up_rolled diff={diff:e} tol={tol0:e} DETECTED");
    }
    {
        let victim = ids_host[0] as usize;
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
        let gens: Vec<u64> = (0..e).map(|ex| if ex == victim { 2 } else { 1 }).collect();
        let mut agen: Vec<u64> = vec![1; 8 * k];
        for c in 0..k {
            if ids_host[c] as usize == victim {
                agen[c] = 2;
            }
        }
        let gu_ev_tab = testing::upload(s, gu_tab)?;
        let dn_ev_tab = testing::upload(s, dn_tab)?;
        let gens_dev = testing::upload(s, gens)?;
        let agen_dev = testing::upload(s, agen)?;
        let gu_ev_view = testing::view(&gu_ev_tab, s, [eu], [1])?;
        let dn_ev_view = testing::view(&dn_ev_tab, s, [eu], [1])?;
        let gens_view = testing::view(&gens_dev, s, [eu], [1])?;
        let agen_view = testing::dense(&agen_dev, s, 8, ku)?;
        let rt = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: active_full.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen_view.prefix_rows(1)?,
            gu: gu_ev_view,
            dn: dn_ev_view,
            slot_gen: gens_view,
        };
        let got = gate.run_split(1, &rt)?;
        let want = reference::moe_row_ids(
            &fx2,
            0,
            &ids_host[..k],
            &weights_host[..k],
            &keep_all,
            Combine::Fp32,
        );
        let tol2 = reference::tolerance(&fx2, &[0]);
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff <= tol2,
            "eviction-reuse row 0 (victim {victim}, generation 2) max diff {diff:e} exceeds tolerance {tol2:e}"
        );
        println!(
            "MOE_DECODE_EVICT victim={victim} generation=2 max_diff={diff:e} tol={tol2:e} PASS"
        );
    }
    println!("MOE_DECODE_GPU rows=1,2,5,8 partial=1 evict=1 controls=2/2 PASS");
    Ok(())
}

/// A device-side trap poisons the CUDA context, so each trap arm runs in a
/// re-exec'd child; the parent accepts exit 3 only.
fn trap_parent(arm: &str) -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = Command::new(std::env::current_exe()?)
        .args(["decode_expert_ffn_gpu", "--ignored", "--test-threads=1"])
        .env("FF_MOE_CHILD", arm)
        .env("FF_Q38_GPU_WINDOW", "granted")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "FF_MOE_CHILD={arm} child exit {status:?}; expected exit 3 (verified trap launch failure)"
    );
    println!("MOE_DECODE_CONTROL {arm} child_exit=3 TRAPPED");
    Ok(())
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn decode_stale_generation_gpu() -> Result<()> {
    trap_parent("stale")
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn decode_duplicate_ids_gpu() -> Result<()> {
    trap_parent("dup")
}
