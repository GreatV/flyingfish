//! GPU correctness, control and trap gates for the decode contribute and
//! combine ops.
use super::combine::{CombineOps, MoeCombine};
use super::decode::{DecodeOps, MoeDecode, decode_scratch_bytes};
use super::reference::{self, Combine, MoeDims};
use super::testing;
use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use crate::{
    experts::catalogue::{ExpertGeometry, ExpertLayout},
    model_spec::ExpertFormat,
};
use anyhow::{Context, Result, ensure};
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

fn pow2i(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Round-to-nearest-even f32 onto the E4M3 grid (bias 7, largest finite
/// magnitude 448); magnitudes above 448 clamp to 448.
fn e4m3_encode(value: f32) -> u8 {
    let sign = u8::from(value.is_sign_negative()) << 7;
    let mag = value.abs().min(448.0);
    let (field, mantissa) = if mag < pow2i(-6) {
        (0u8, (mag / pow2i(-9)).round_ties_even() as u8)
    } else {
        let e = ((mag.to_bits() >> 23) & 0xff) as i32 - 127;
        let mut q = (mag / pow2i(e - 3)).round_ties_even() as u32;
        let mut field = (e + 7) as u32;
        if q == 16 {
            q = 8;
            field += 1;
        }
        (field as u8, (q - 8) as u8)
    };
    sign | (field << 3) | mantissa
}

#[test]
fn e4m3_encode_round_trips_nearest_grid_point() {
    let cases: [(f32, u8); 6] = [
        (0.0, 0x00),
        (448.0, 0x7e),
        (449.0, 0x7e),
        (1.5 * pow2i(-9), 0x02),
        (2.5 * pow2i(-9), 0x02),
        (1.1875, 0x3a),
    ];
    for (value, code) in cases {
        assert_eq!(e4m3_encode(value), code, "value {value:e}");
    }
    assert_eq!(e4m3_encode(-0.0), 0x80);
    assert_eq!(e4m3_encode(-448.0), 0xfe);
    let mut state = 0x243f_6a88u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 8_388_608.0 - 1.0) * 440.0
    };
    for _ in 0..4096 {
        let v = next();
        let code = e4m3_encode(v);
        assert_ne!(code & 0x7f, 0x7f, "reserved code from {v:e}");
        let d = crate::fp8::decode_e4m3(code);
        assert!(
            (v - d).abs() <= pow2i(-9) + d.abs() * pow2i(-4),
            "value {v:e} decoded to {d:e}"
        );
    }
}

/// Per-output-channel W8A16 quantization of one weight row:
/// s = amax(|w|)/448 in f32 (scale stored BF16), codes = e4m3(w/s).
fn quantize_row(row: &[bf16]) -> (Vec<u8>, bf16) {
    let amax = row.iter().map(|w| w.to_f32().abs()).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return (vec![0u8; row.len()], bf16::from_f32(0.0));
    }
    let codes = row
        .iter()
        .map(|&w| e4m3_encode(w.to_f32() / (amax / 448.0)))
        .collect();
    (codes, bf16::from_f32(amax / 448.0))
}

/// Per-expert W8A16 slots in the checkpoint in-slot order plus the
/// materialized effective BF16 weights bf16(f32(code) * f32(scale)).
struct W8a16World {
    slots: Vec<u8>,
    effective_gate_up: Vec<bf16>,
    effective_down: Vec<bf16>,
    slot_bytes: u64,
}

fn w8a16_world(fx: &reference::MoeFixture) -> Result<W8a16World> {
    let (e, h, i) = (fx.dims.experts, fx.dims.hidden, fx.dims.inter);
    let layout = ExpertLayout::new(
        ExpertFormat::Fp8W8a16,
        &ExpertGeometry {
            layers: 1,
            experts: e as u32,
            hidden: h as u32,
            intermediate: i as u32,
            top_k: fx.dims.top_k as u32,
        },
    )?;
    let slot_bytes = layout.slot_bytes();
    let weight = i * h;
    let mut slots = vec![0u8; e * slot_bytes as usize];
    let mut effective_gate_up = vec![bf16::from_f32(0.0); e * 2 * i * h];
    let mut effective_down = vec![bf16::from_f32(0.0); e * h * i];
    let mut quantize_into =
        |row: &[bf16], codes_at: usize, scale_at: usize, effective: &mut [bf16]| {
            let (codes, scale) = quantize_row(row);
            assert_eq!(codes.len(), effective.len());
            slots[codes_at..codes_at + codes.len()].copy_from_slice(&codes);
            slots[scale_at..scale_at + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
            for (dst, &code) in effective.iter_mut().zip(&codes) {
                *dst = crate::fp8::effective_weight(code, scale);
            }
        };
    for ex in 0..e {
        let base = ex * slot_bytes as usize;
        for j in 0..i {
            quantize_into(
                fx.gate_row(ex, j),
                base + j * h,
                base + 3 * weight + 2 * j,
                &mut effective_gate_up[(ex * 2 * i + j) * h..][..h],
            );
            quantize_into(
                fx.up_row(ex, j),
                base + weight + j * h,
                base + 3 * weight + 2 * (i + j),
                &mut effective_gate_up[(ex * 2 * i + i + j) * h..][..h],
            );
        }
        for d in 0..h {
            quantize_into(
                fx.down_row(ex, d),
                base + 2 * weight + d * i,
                base + 3 * weight + 4 * i + 2 * d,
                &mut effective_down[(ex * h + d) * i..][..i],
            );
        }
    }
    assert!(
        !slots.iter().any(|&b| (b & 0x7f) == 0x7f),
        "fixture quantization produced the reserved e4m3 code"
    );
    Ok(W8a16World {
        slots,
        effective_gate_up,
        effective_down,
        slot_bytes,
    })
}

fn u16_words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

fn bf16_bits(words: &[bf16]) -> Vec<u16> {
    words.iter().map(|w| w.to_bits()).collect()
}

fn ensure_bitwise(name: &str, got: &[u16], want: &[u16]) -> Result<()> {
    anyhow::ensure!(
        got.len() == want.len(),
        "{name} word count {} != {}",
        got.len(),
        want.len()
    );
    for (idx, (&g, &w)) in got.iter().zip(want).enumerate() {
        anyhow::ensure!(g == w, "{name} word {idx}: {g:#06x} != {w:#06x}");
    }
    Ok(())
}

fn max_row_diff(got: &[bf16], want: &[bf16]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g.to_f32() - w.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn row<T: Copy>(tensor: Tensor<T, 2>, index: usize) -> Result<Tensor<T, 2>> {
    ensure!(index < tensor.shape[0] as usize, "test row outside tensor");
    let offset = index as u64 * tensor.stride[0] * size_of::<T>() as u64;
    // SAFETY: the original view owns the complete row and outlives every test launch.
    unsafe {
        Tensor::new(
            tensor.ptr + offset,
            [1, tensor.shape[1]],
            tensor.stride,
            tensor.span_bytes - offset,
        )
    }
}

fn invariant_decode(device: &Device) -> Result<()> {
    use crate::cpu::expert_ffn as cpu;
    let max_rows = crate::linear_plan::MAX_SKINNY_ROWS;
    let fx = reference::fixture(
        MoeDims {
            experts: 17,
            ..MoeDims::minicpm47(max_rows)
        },
        3,
    );
    let (e, k, h, i) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let world = w8a16_world(&fx)?;
    let mut arena = fx.gate_up.clone();
    arena.extend_from_slice(&fx.down);
    let s = &device.stream;
    let x = s.clone_htod(&fx.x)?;
    let gu = s.clone_htod(&fx.gate_up)?;
    let dn = s.clone_htod(&fx.down)?;
    let fp8 = s.clone_htod(&world.slots)?;
    let shared_gate = s.clone_htod(&fx.shared_gate)?;
    let shared_up = s.clone_htod(&fx.shared_up)?;
    let shared_down = s.clone_htod(&fx.shared_down)?;
    let shared_expert_gate = s.clone_htod(&fx.shared_expert_gate)?;
    let ids: Vec<_> = (0..max_rows * k)
        .map(|a| ((a / k * 5 + a % k * 3) % e) as i32)
        .collect();
    let weights: Vec<_> = (0..max_rows * k)
        .map(|a| bf16::from_f32(1.0 / (2 + (a * 7) % 19) as f32))
        .collect();
    let ids_device = s.clone_htod(&ids)?;
    let weights_device = s.clone_htod(&weights)?;
    let active = s.clone_htod(&vec![1u32; max_rows * k])?;
    let assigned = s.clone_htod(&vec![1u64; max_rows * k])?;
    let scratch = s.alloc_zeros::<u8>(decode_scratch_bytes(max_rows, k, i) as usize)?;
    let ops = DecodeOps::new_invariant(&device.ctx)?;
    let legacy = DecodeOps::new(&device.ctx)?;
    let combine = CombineOps::new(&device.ctx)?;
    let geometry = ExpertGeometry {
        layers: 1,
        experts: e as u32,
        top_k: k as u32,
        hidden: h as u32,
        intermediate: i as u32,
    };
    let layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
    let shape = cpu::Shape::new(h, i)?;
    let mut cpu_scratch = cpu::Scratch::new(&shape);
    let mut defects = 0;
    for format in [ExpertFormat::Bf16, ExpertFormat::Fp8W8a16] {
        let slot = (format == ExpertFormat::Fp8W8a16)
            .then(|| layout.fp8_slot())
            .flatten();
        let table = if slot.is_some() {
            let base = fp8.device_ptr(s).0;
            testing::slots(s, e, base, world.slot_bytes, base, world.slot_bytes)?
        } else {
            testing::slots(
                s,
                e,
                gu.device_ptr(s).0,
                (2 * i * h * 2) as u64,
                dn.device_ptr(s).0,
                (h * i * 2) as u64,
            )?
        };
        let routing = Routing {
            ids: testing::dense(&ids_device, s, max_rows as u32, k as u32)?,
            weights: testing::dense(&weights_device, s, max_rows as u32, k as u32)?,
            active: testing::dense(&active, s, max_rows as u32, k as u32)?,
            agen: testing::dense(&assigned, s, max_rows as u32, k as u32)?,
            gu: testing::view(&table.gate_up, s, [e as u32], [1])?,
            dn: testing::view(&table.down, s, [e as u32], [1])?,
            slot_gen: testing::view(&table.generation, s, [e as u32], [1])?,
        };
        let gate = Gate {
            ops: &ops,
            combine: &combine,
            s,
            x: testing::dense(&x, s, max_rows as u32, h as u32)?,
            shared_gate: testing::dense(&shared_gate, s, i as u32, h as u32)?,
            shared_up: testing::dense(&shared_up, s, i as u32, h as u32)?,
            shared_down: testing::dense(&shared_down, s, h as u32, i as u32)?,
            seg: testing::view(&shared_expert_gate, s, [h as u32], [1])?,
            valid: testing::view(&table.valid, s, [e as u32], [1])?,
            scratch: Scratch {
                ptr: scratch.device_ptr(s).0,
                bytes: scratch.len() as u64,
            },
            top_k: k as u32,
            hidden: h as u32,
            inter: i as u32,
        };
        let mut serial: [Vec<bf16>; 3] = std::array::from_fn(|_| Vec::new());
        for r in 0..max_rows {
            let one = Gate {
                x: row(gate.x, r)?,
                ..gate
            };
            let route = Routing {
                ids: row(routing.ids, r)?,
                weights: row(routing.weights, r)?,
                active: row(routing.active, r)?,
                agen: row(routing.agen, r)?,
                ..routing
            };
            for (all, values) in serial.iter_mut().zip(one.stages(1, &route, slot, false)?) {
                all.extend(values);
            }
        }
        for rows in 1..=max_rows {
            for graph in [false, true] {
                let got = gate.stages(rows as u32, &routing, slot, graph)?;
                for ((stage, values), reference) in ["contribution", "shared", "combine"]
                    .into_iter()
                    .zip(got)
                    .zip(&serial)
                {
                    let mismatches = values
                        .iter()
                        .zip(reference)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    defects += usize::from(mismatches != 0);
                    println!(
                        "{} INVARIANT_MOE format={format:?} x=[{rows},{h}] top_k={k} inter={i} stage={stage} graph={graph} bit_mismatches={mismatches}",
                        if mismatches == 0 { "PASS" } else { "DEFECT" }
                    );
                }
            }
        }
        let mut cpu_output = vec![bf16::ZERO; max_rows * k * h];
        for r in 0..max_rows {
            for c in 0..k {
                let a = r * k + c;
                let expert = ids[a] as usize;
                let output = &mut cpu_output[a * h..(a + 1) * h];
                if let Some(slot_layout) = slot {
                    let start = expert * world.slot_bytes as usize;
                    cpu::fp8_contribution(
                        cpu::Isa::detect(),
                        cpu::Fp8Contribution {
                            shape: &shape,
                            slot: &world.slots[start..start + world.slot_bytes as usize],
                            slot_layout,
                            hidden: &fx.x[r * h..(r + 1) * h],
                            routing: weights[a],
                        },
                        output,
                        &mut cpu_scratch,
                    )?;
                } else {
                    cpu::contribution(
                        cpu::Isa::detect(),
                        cpu::Contribution {
                            shape: &shape,
                            arena: &arena,
                            weights: &cpu::ExpertWeights {
                                gate_up: expert * 2 * i * h..(expert + 1) * 2 * i * h,
                                gate_up_stride: h,
                                down: fx.gate_up.len() + expert * h * i
                                    ..fx.gate_up.len() + (expert + 1) * h * i,
                                down_stride: i,
                            },
                            hidden: &fx.x[r * h..(r + 1) * h],
                            routing: weights[a],
                        },
                        output,
                        &mut cpu_scratch,
                    )?;
                }
            }
        }
        let mismatches = cpu_output
            .iter()
            .zip(&serial[0])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        defects += usize::from(mismatches != 0);
        println!(
            "{} INVARIANT_MOE_CPU format={format:?} rows={max_rows} isa={} weighted=true bit_mismatches={mismatches}",
            if mismatches == 0 { "PASS" } else { "DEFECT" },
            cpu::Isa::detect().name()
        );
        let old = Gate {
            ops: &legacy,
            ..gate
        }
        .stages(1, &routing, slot, false)?;
        let mismatches = old[0]
            .iter()
            .zip(&cpu_output)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            mismatches > 0,
            "legacy lane8 control did not distinguish lane16"
        );
        println!(
            "PASS INVARIANT_MOE_CONTROL format={format:?} legacy_lane8=true bit_mismatches={mismatches}"
        );
    }
    println!(
        "INVARIANT_MOE_SUMMARY formats=2 rows=1..16 stages=3 eager_graph=true weighted_CPU=true defects={defects}"
    );
    ensure!(defects == 0, "invariant MoE DEFECT cells={defects}");
    Ok(())
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
    fn stages(
        &self,
        rows: u32,
        routing: &Routing,
        layout: Option<&crate::experts::catalogue::Fp8Slot>,
        graph: bool,
    ) -> Result<[Vec<bf16>; 3]> {
        let mut io = decode_io(self.s, rows + 1, self.top_k, self.hidden)?;
        let words = rows as usize * self.hidden as usize;
        let choices = rows as usize * self.top_k as usize;
        let marker = bf16::from_bits(0x7fc1);
        self.s
            .memcpy_htod(&vec![marker; io._contrib_dev.len()], &mut io._contrib_dev)?;
        self.s.memcpy_htod(
            &vec![marker; io._shared_out_dev.len()],
            &mut io._shared_out_dev,
        )?;
        self.s
            .memcpy_htod(&vec![marker; io.y_dev.len()], &mut io.y_dev)?;
        self.s
            .memcpy_htod(&vec![0xfeedbeefu32; io._done_dev.len()], &mut io._done_dev)?;
        io.contributions = io.contributions.prefix_rows(rows)?;
        io.shared_out = TensorMut(io.shared_out.read().prefix_rows(rows)?);
        io.done = TensorMut(io.done.read().prefix_rows(rows)?);
        io.y = TensorMut(io.y.read().prefix_rows(rows)?);
        let input = self.input(
            rows,
            routing,
            io.contributions,
            &mut io.shared_out,
            &mut io.done,
        )?;
        let mut done = io._done_dev.slice_mut(..choices);
        let mut execute = || -> Result<()> {
            self.s.memset_zeros(&mut done)?;
            self.ops
                .decode(self.s, &input, &mut io.y, &self.scratch, layout)
        };
        execute()?;
        if graph {
            self.s.synchronize()?;
            self.s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            let launched = execute();
            let captured = self
                .s
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
            launched?;
            captured?.context("empty invariant FFN graph")?.launch()?;
        }
        self.s.synchronize()?;
        let completed = self.s.clone_dtoh(&io._done_dev)?;
        ensure!(
            completed[..choices].iter().all(|&v| v == 1)
                && completed[choices..].iter().all(|&v| v == 0xfeedbeef),
            "invariant FFN completion/tail rows={rows} graph={graph}"
        );
        let mut stages = [
            self.s.clone_dtoh(&io._contrib_dev)?,
            self.s.clone_dtoh(&io._shared_out_dev)?,
            self.s.clone_dtoh(&io.y_dev)?,
        ];
        for (stage, count) in stages
            .iter_mut()
            .zip([words * self.top_k as usize, words, words])
        {
            ensure!(
                stage[..count].iter().all(|v| v.is_finite())
                    && stage[count..]
                        .iter()
                        .all(|v| v.to_bits() == marker.to_bits()),
                "invariant FFN finite/tail rows={rows} graph={graph}"
            );
            stage.truncate(count);
        }
        Ok(stages)
    }

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
            self.ops
                .decode(self.s, &input, &mut io.y, &self.scratch, None)?;
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
    if std::env::var_os("FF_MOE_CHILD").is_none() {
        invariant_decode(&device)?;
    }
    let ops = DecodeOps::new(&device.ctx)?;
    let combine_ops = CombineOps::new(&device.ctx)?;
    let s = &device.stream;
    {
        let dims = MoeDims::minicpm47(2);
        let (rows, k, h) = (dims.rows, dims.top_k, dims.hidden);
        let mut io = decode_io(s, rows as u32, k as u32, h as u32)?;
        let half_ulp = 2f32.powi(-8);
        let mut contributions = vec![bf16::ZERO; rows * k * h];
        let mut shared = vec![bf16::ZERO; rows * h];
        for row in 0..rows {
            let sign = if row == 0 { 1.0 } else { -1.0 };
            contributions[row * k * h..][..h].fill(bf16::from_f32(sign));
            contributions[(row * k + 1) * h..][..h].fill(bf16::from_f32(sign * half_ulp));
            shared[row * h..][..h].fill(bf16::from_f32(sign * half_ulp));
        }
        s.memcpy_htod(&contributions, &mut io._contrib_dev)?;
        s.memcpy_htod(&shared, &mut io._shared_out_dev)?;
        let ids = testing::upload(s, (0..rows * k).map(|c| (c % k) as i32).collect())?;
        let active = testing::upload(s, vec![1u32; rows * k])?;
        let scale = testing::upload(s, vec![bf16::ONE; rows])?;
        let active_view = testing::dense(&active, s, rows as u32, k as u32)?;
        combine_ops.combine(
            s,
            &MoeCombine {
                ids: testing::dense(&ids, s, rows as u32, k as u32)?,
                final_active: active_view,
                done: active_view,
                contributions: io.contributions,
                shared_out: io.shared_out.read(),
                x: io.shared_out.read(),
                shared_expert_gate: testing::view(&io._shared_out_dev, s, [h as u32], [1])?,
                scale: testing::dense_mut(&scale, s, rows as u32, 1)?,
                y: TensorMut(io.y.read()),
            },
        )?;
        s.synchronize()?;
        let output = s.clone_dtoh(&io.y_dev)?;
        let mismatches = output
            .iter()
            .enumerate()
            .filter(|(word, value)| {
                let expected = bf16::from_f32(if *word < h { 1.0 } else { -1.0 });
                value.to_bits() != expected.to_bits()
            })
            .count();
        println!(
            "{} moe_combine_bf16_boundary rows={rows} top_k={k} hidden={h} bit_mismatches={mismatches}",
            if mismatches == 0 { "PASS" } else { "DEFECT" }
        );
        anyhow::ensure!(mismatches == 0, "DEFECT routed/shared BF16 boundary");
    }
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

/// W8A16 arm of the decode gate: the shared-expert BF16 tensors and the
/// validation surface of Gate, with Routing's gu/dn views carrying slot
/// base pointers.
struct W8Gate<'a> {
    ops: &'a DecodeOps,
    s: &'a Arc<CudaStream>,
    x: Tensor<bf16, 2>,
    shared_gate: Tensor<bf16, 2>,
    shared_up: Tensor<bf16, 2>,
    shared_down: Tensor<bf16, 2>,
    seg: Tensor<bf16, 1>,
    valid: Tensor<u32, 1>,
    slot_bytes: u64,
    top_k: u32,
    hidden: u32,
}

impl W8Gate<'_> {
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

    /// The native slot layout selects the routed weight loads.
    fn run(&self, rows: u32, rt: &Routing, scratch: &Scratch) -> Result<Vec<bf16>> {
        let mut io = decode_io(self.s, rows, self.top_k, self.hidden)?;
        {
            let input = self.input(rows, rt, io.contributions, &mut io.shared_out, &mut io.done)?;
            let geometry = ExpertGeometry {
                layers: 1,
                experts: self.valid.shape[0],
                hidden: self.hidden,
                intermediate: self.shared_gate.shape[0],
                top_k: self.top_k,
            };
            let layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
            ensure!(
                layout.slot_bytes() == self.slot_bytes,
                "W8A16 gate slot bytes"
            );
            self.ops
                .decode(self.s, &input, &mut io.y, scratch, layout.fp8_slot())?;
        }
        self.s.synchronize()?;
        Ok(self.s.clone_dtoh(&io.y_dev)?)
    }
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn decode_expert_ffn_w8a16_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    let ops = DecodeOps::new(&device.ctx)?;
    let s = &device.stream;
    let fx = reference::fixture(MoeDims::new(4, 16, 4, 128, 64), 0);
    let (e, k, h, i) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let (eu, ku, hu, iu) = (e as u32, k as u32, h as u32, i as u32);
    let world = w8a16_world(&fx)?;

    // Oracle fixture: routing inputs unchanged, routed experts replaced by
    // the materialized effective weights.
    let mut fxq = reference::fixture(MoeDims::new(4, 16, 4, 128, 64), 0);
    fxq.gate_up = world.effective_gate_up.clone();
    fxq.down = world.effective_down.clone();

    let x_dev = testing::upload(&device.upload, fx.x.clone())?;
    let sg_dev = testing::upload(&device.upload, fx.shared_gate.clone())?;
    let su_dev = testing::upload(&device.upload, fx.shared_up.clone())?;
    let sd_dev = testing::upload(&device.upload, fx.shared_down.clone())?;
    let se_dev = testing::upload(&device.upload, fx.shared_expert_gate.clone())?;
    let slots_dev = testing::upload(&device.upload, world.slots.clone())?;
    let gu_eff_dev = testing::upload(&device.upload, world.effective_gate_up.clone())?;
    let dn_eff_dev = testing::upload(&device.upload, world.effective_down.clone())?;
    device.finish_upload()?;

    let mut ids_host = Vec::with_capacity(4 * k);
    let mut weights_host = Vec::with_capacity(4 * k);
    for r in 0..4 {
        let routed = reference::route_row(&fx, r);
        ids_host.extend_from_slice(&routed.ids);
        weights_host.extend_from_slice(&routed.weights);
    }
    let ids_dev = testing::upload(s, ids_host.clone())?;
    let weights_dev = testing::upload(s, weights_host.clone())?;
    let keep_all: Vec<usize> = (0..k).collect();

    let x_full = testing::dense(&x_dev, s, 4, hu)?;
    let ids_full = testing::dense(&ids_dev, s, 4, ku)?;
    let weights_full = testing::dense(&weights_dev, s, 4, ku)?;
    let active_dev = testing::upload(s, vec![1u32; 4 * k])?;
    let active_full = testing::dense(&active_dev, s, 4, ku)?;
    let agen_dev = testing::upload(s, vec![1u64; 4 * k])?;
    let agen_full = testing::dense(&agen_dev, s, 4, ku)?;
    let shared_gate_view = testing::dense(&sg_dev, s, iu, hu)?;
    let shared_up_view = testing::dense(&su_dev, s, iu, hu)?;
    let shared_down_view = testing::dense(&sd_dev, s, hu, iu)?;
    let seg_view = testing::view(&se_dev, s, [hu], [1])?;

    let slot_base = slots_dev.device_ptr(s).0;
    let w8_table = testing::slots(
        s,
        e,
        slot_base,
        world.slot_bytes,
        slot_base,
        world.slot_bytes,
    )?;
    let w8_gu = testing::view(&w8_table.gate_up, s, [eu], [1])?;
    let w8_dn = testing::view(&w8_table.down, s, [eu], [1])?;
    let w8_valid = testing::view(&w8_table.valid, s, [eu], [1])?;
    let w8_gen = testing::view(&w8_table.generation, s, [eu], [1])?;

    let bf16_table = testing::slots(
        s,
        e,
        gu_eff_dev.device_ptr(s).0,
        (2 * i * h * 2) as u64,
        dn_eff_dev.device_ptr(s).0,
        (h * i * 2) as u64,
    )?;
    let bf16_gu = testing::view(&bf16_table.gate_up, s, [eu], [1])?;
    let bf16_dn = testing::view(&bf16_table.down, s, [eu], [1])?;
    let bf16_valid = testing::view(&bf16_table.valid, s, [eu], [1])?;
    let bf16_gen = testing::view(&bf16_table.generation, s, [eu], [1])?;

    let scratch_mem = testing::zeros::<u8>(s, decode_scratch_bytes(4, k, i) as usize)?;
    let scratch = Scratch {
        ptr: scratch_mem.device_ptr(s).0,
        bytes: decode_scratch_bytes(4, k, i),
    };
    let w8 = W8Gate {
        ops: &ops,
        s,
        x: x_full,
        shared_gate: shared_gate_view,
        shared_up: shared_up_view,
        shared_down: shared_down_view,
        seg: seg_view,
        valid: w8_valid,
        slot_bytes: world.slot_bytes,
        top_k: ku,
        hidden: hu,
    };

    // K1 bitwise cell: the W8A16 kernels against the BF16 kernels running on
    // the host-side materialized effective weights, one row's expert set.
    {
        let rt_w8 = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: active_full.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen_full.prefix_rows(1)?,
            gu: w8_gu,
            dn: w8_dn,
            slot_gen: w8_gen,
        };
        let mut io_w8 = decode_io(s, 1, ku, hu)?;
        {
            let input = w8.input(
                1,
                &rt_w8,
                io_w8.contributions,
                &mut io_w8.shared_out,
                &mut io_w8.done,
            )?;
            let layout = ExpertLayout::new(
                ExpertFormat::Fp8W8a16,
                &ExpertGeometry {
                    layers: 1,
                    experts: eu,
                    hidden: hu,
                    intermediate: iu,
                    top_k: ku,
                },
            )?;
            ops.contribute_w8a16(s, &input, layout.fp8_slot().unwrap(), &scratch)?;
        }
        s.synchronize()?;
        let scratch_w8 = s.clone_dtoh(&scratch_mem)?;
        let contrib_w8 = s.clone_dtoh(&io_w8._contrib_dev)?;
        let shared_out_w8 = s.clone_dtoh(&io_w8._shared_out_dev)?;

        let rt_bf16 = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: active_full.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen_full.prefix_rows(1)?,
            gu: bf16_gu,
            dn: bf16_dn,
            slot_gen: bf16_gen,
        };
        let mut io_bf16 = decode_io(s, 1, ku, hu)?;
        {
            let input = MoeDecode {
                x: w8.x.prefix_rows(1)?,
                ids: rt_bf16.ids,
                active: rt_bf16.active,
                weights: rt_bf16.weights,
                slots_gate_up: rt_bf16.gu,
                slots_down: rt_bf16.dn,
                slot_valid: bf16_valid,
                slot_generation: rt_bf16.slot_gen,
                assignment_gen: rt_bf16.agen,
                shared_gate: w8.shared_gate,
                shared_up: w8.shared_up,
                shared_down: w8.shared_down,
                shared_expert_gate: w8.seg,
                contributions: io_bf16.contributions,
                shared_out: &mut io_bf16.shared_out,
                done: &mut io_bf16.done,
            };
            ops.contribute(s, &input, &scratch)?;
        }
        s.synchronize()?;
        let scratch_bf16 = s.clone_dtoh(&scratch_mem)?;
        let contrib_bf16 = s.clone_dtoh(&io_bf16._contrib_dev)?;
        let shared_out_bf16 = s.clone_dtoh(&io_bf16._shared_out_dev)?;

        let (h_bytes, hs_bytes) = (k * i * 2, i * 2);
        ensure_bitwise(
            "h",
            &u16_words(&scratch_w8[..h_bytes]),
            &u16_words(&scratch_bf16[..h_bytes]),
        )?;
        ensure_bitwise(
            "h_shared",
            &u16_words(&scratch_w8[h_bytes..h_bytes + hs_bytes]),
            &u16_words(&scratch_bf16[h_bytes..h_bytes + hs_bytes]),
        )?;
        ensure_bitwise(
            "contributions",
            &bf16_bits(&contrib_w8),
            &bf16_bits(&contrib_bf16),
        )?;
        ensure_bitwise(
            "shared_out",
            &bf16_bits(&shared_out_w8),
            &bf16_bits(&shared_out_bf16),
        )?;
        println!(
            "MOE_DECODE_W8A16_K1_BITWISE h={h_bytes}B h_shared={hs_bytes}B contributions={}w shared_out={}w PASS",
            k * h,
            h
        );
    }

    for rows in [1u32, 2, 3, 4] {
        let rt = Routing {
            ids: ids_full.prefix_rows(rows)?,
            active: active_full.prefix_rows(rows)?,
            weights: weights_full.prefix_rows(rows)?,
            agen: agen_full.prefix_rows(rows)?,
            gu: w8_gu,
            dn: w8_dn,
            slot_gen: w8_gen,
        };
        let got = w8.run(rows, &rt, &scratch)?;
        let sample: Vec<usize> = (0..rows as usize).collect();
        let tol = reference::tolerance(&fxq, &sample);
        let mut worst = 0.0f32;
        for r in 0..rows as usize {
            let want = reference::moe_row_ids(
                &fxq,
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
                "w8a16 decode rows={rows} row={r} max diff {diff:e} exceeds tolerance {tol:e}"
            );
            println!("MOE_DECODE_W8A16_SPREAD rows={rows} row={r} max_diff={diff:e} tol={tol:e}");
        }
        println!("MOE_DECODE_W8A16 rows={rows} worst_diff={worst:e} tol={tol:e} PASS");
    }

    {
        let victim = ids_host[0] as usize;
        let vs = victim * world.slot_bytes as usize;
        let gate_scale_at = |j: usize| {
            let at = vs + 3 * i * h + 2 * j;
            bf16::from_bits(u16::from_le_bytes([world.slots[at], world.slots[at + 1]]))
        };
        let (mut j0, mut j1, mut spread) = (0usize, 1usize, 0.0f32);
        for a in 0..i {
            for b in a + 1..i {
                let d = (gate_scale_at(a).to_f32() - gate_scale_at(b).to_f32()).abs();
                if d > spread {
                    spread = d;
                    (j0, j1) = (a, b);
                }
            }
        }
        anyhow::ensure!(
            spread > 0.0,
            "swapped-scale control requires distinct gate scales"
        );
        let mut corrupted = world.slots.clone();
        let (a0, a1) = (vs + 3 * i * h + 2 * j0, vs + 3 * i * h + 2 * j1);
        for t in 0..2 {
            corrupted.swap(a0 + t, a1 + t);
        }
        let corrupt_dev = testing::upload(s, corrupted)?;
        let corrupt_base = corrupt_dev.device_ptr(s).0;
        let mut gu_tab: Vec<u64> = (0..e)
            .map(|ex| slot_base + ex as u64 * world.slot_bytes)
            .collect();
        let dn_tab = gu_tab.clone();
        gu_tab[victim] = corrupt_base;
        let gu_cor = testing::upload(s, gu_tab)?;
        let dn_cor = testing::upload(s, dn_tab)?;
        let gu_cor_view = testing::view(&gu_cor, s, [eu], [1])?;
        let dn_cor_view = testing::view(&dn_cor, s, [eu], [1])?;
        let rt = Routing {
            ids: ids_full.prefix_rows(1)?,
            active: active_full.prefix_rows(1)?,
            weights: weights_full.prefix_rows(1)?,
            agen: agen_full.prefix_rows(1)?,
            gu: gu_cor_view,
            dn: dn_cor_view,
            slot_gen: w8_gen,
        };
        let got = w8.run(1, &rt, &scratch)?;
        let want = reference::moe_row_ids(
            &fxq,
            0,
            &ids_host[..k],
            &weights_host[..k],
            &keep_all,
            Combine::Fp32,
        );
        let tol0 = reference::tolerance(&fxq, &[0]);
        let diff = max_row_diff(&got[..h], &want);
        anyhow::ensure!(
            diff > tol0,
            "swapped gate-scale control undetected: diff {diff:e} within tolerance {tol0:e}"
        );
        println!(
            "MOE_DECODE_W8A16_CONTROL swapped_gate_scales victim={victim} rows=({j0},{j1}) diff={diff:e} tol={tol0:e} DETECTED"
        );
    }
    println!("MOE_DECODE_W8A16_GPU k1_bitwise=1 rows=1,2,3,4 control=1 PASS");
    Ok(())
}
