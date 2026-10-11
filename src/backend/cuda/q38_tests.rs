//! Explicit-window GPU probes. Default cargo test never initializes a GPU here.
use super::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{RowDomain, SeqRef, Tensor, TensorMut},
    linear::{Fp8Ops, LinearWeightRef},
    ops::{CommonOpsRef, GatedRmsInput, GdnGateBindings, Ops, RmsMode, RmsOptions},
};
use crate::{
    fp8::effective_weight,
    op_contract::{ExecutionPhase, SequenceMeta},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;
use std::{rc::Rc, sync::Arc};

fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    shape: [u32; R],
    stride: [u64; R],
) -> Result<Tensor<T, R>> {
    let (p, _r) = b.device_ptr(s);
    unsafe {
        Tensor::new(
            p,
            shape,
            stride,
            (b.len() * std::mem::size_of::<T>()) as u64,
        )
    }
}
fn meta(
    s: &Arc<CudaStream>,
    prefix: i32,
    rows: i32,
    capacity: i32,
) -> Result<(CudaSlice<i32>, SeqRef)> {
    let m = SequenceMeta {
        prefix,
        valid_rows: rows,
        key_count: prefix + rows,
        capacity,
        causal: 1,
        reserved: [0; 3],
    };
    m.validate(capacity as u32, rows as u32)?;
    let b = s.clone_htod(&[
        m.prefix,
        m.valid_rows,
        m.key_count,
        m.capacity,
        m.causal,
        0,
        0,
        0,
    ])?;
    let (p, _r) = b.device_ptr(s);
    drop(_r);
    Ok((b, SeqRef { ptr: p }))
}
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted and FF_Q38_PROBE_DIR"]
fn q38_primitives_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    let s = &d.stream;
    let fixture_dir = std::env::temp_dir().join(format!("ff-q38-upload-{}", std::process::id()));
    std::fs::create_dir(&fixture_dir)?;
    let uploaded = (|| -> Result<()> {
        use safetensors::{
            Dtype,
            tensor::{TensorView, serialize_to_file},
        };
        use std::collections::BTreeMap;
        let codes = vec![0x38u8; 129 * 257];
        let scales: Vec<u8> = [0u16, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x3f81]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        serialize_to_file(
            [
                (
                    "w",
                    TensorView::new(Dtype::F8_E4M3, vec![129, 257], &codes)?,
                ),
                ("s", TensorView::new(Dtype::BF16, vec![2, 3], &scales)?),
            ],
            None,
            &fixture_dir.join("model.safetensors"),
        )?;
        let mapped = crate::checkpoint::MappedCheckpoint::open(&fixture_dir)?;
        let manifest = crate::checkpoint::TensorManifest {
            entries: BTreeMap::new(),
            resident_bytes: codes.len() + scales.len(),
        };
        super::weights::checkpoint_mapped(&mapped, &d, Some(&manifest), |loader| {
            let w = loader.load_fp8("w", &[129, 257])?;
            let scale = loader.load(&[("s".into(), vec![2, 3])])?;
            ensure!(
                d.upload.clone_dtoh(&w)? == codes,
                "uploaded FP8 payload mismatch"
            );
            let bits: Vec<u8> = d
                .upload
                .clone_dtoh(&scale)?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            ensure!(bits == scales, "uploaded BF16 payload mismatch");
            ensure!(
                loader.bytes == manifest.resident_bytes,
                "upload accounting mismatch"
            );
            Ok(())
        })
    })();
    std::fs::remove_dir_all(&fixture_dir)?;
    uploaded?;
    let fp8 = Fp8Ops::new(&d.ctx)?;
    // All codes, including the two NaNs, plus non-integral 128-block scale tails.
    for (n, k) in [(1u32, 256u32), (129, 257)] {
        let codes: Vec<u8> = (0..n * k).map(|i| i as u8).collect();
        let scales: Vec<bf16> = (0..n.div_ceil(128) * k.div_ceil(128))
            .map(|i| bf16::from_f32([1.0078125, 0.03125, 3.140625][i as usize % 3]))
            .collect();
        let wb = s.clone_htod(&codes)?;
        let sb = s.clone_htod(&scales)?;
        let y = s.alloc_zeros::<bf16>((n * k) as usize)?;
        let w = view(&wb, s, [n, k], [u64::from(k), 1])?;
        let sc = view(
            &sb,
            s,
            [n.div_ceil(128), k.div_ceil(128)],
            [u64::from(k.div_ceil(128)), 1],
        )?;
        let mut out = TensorMut(view(&y, s, [n, k], [u64::from(k), 1])?);
        fp8.materialize(s, w, sc, &mut out)?;
        let got = s.clone_dtoh(&y)?;
        for row in 0..n {
            for col in 0..k {
                let i = (row * k + col) as usize;
                let want = effective_weight(
                    codes[i],
                    scales[((row / 128) * k.div_ceil(128) + col / 128) as usize],
                );
                ensure!(
                    if want.to_f32().is_nan() {
                        got[i].to_f32().is_nan()
                    } else {
                        got[i].to_bits() == want.to_bits()
                    },
                    "dequant mismatch at {n}x{k} ({row},{col})"
                );
            }
        }
    }
    let ops = Ops::new(&d.ctx)?;
    let closure = Rc::new(Closure::new(false, "GPU primitive probe".into()));
    let runtime = std::path::PathBuf::from(
        std::env::var_os("FF_Q38_PROBE_DIR").context("FF_Q38_PROBE_DIR is required")?,
    );
    std::fs::create_dir_all(&runtime)?;
    let mut blas = Blas::new(&d, &ops, &runtime, closure)?;
    blas.enable_fp8(&d.ctx)?;
    // Exact sums isolate wiring, phase, tail and storage errors from backend rounding noise.
    for (n, k) in [
        (129u32, 257u32),
        (12288, 5120),
        (17408, 5120),
        (5120, 17408),
        (10240, 5120),
        (6144, 5120),
        (5120, 6144),
    ] {
        let codes: Vec<u8> = (0..n * k)
            .map(|i| if (i / k) % 2 == 0 { 0x38 } else { 0xb8 })
            .collect();
        let scales = vec![bf16::ONE; (n.div_ceil(128) * k.div_ceil(128)) as usize];
        let wb = s.clone_htod(&codes)?;
        let sb = s.clone_htod(&scales)?;
        let w = view(&wb, s, [n, k], [u64::from(k), 1])?;
        let sc = view(
            &sb,
            s,
            [n.div_ceil(128), k.div_ceil(128)],
            [u64::from(k.div_ceil(128)), 1],
        )?;
        let materialized = s.alloc_zeros::<bf16>((n * k) as usize)?;
        let mut mat = TensorMut(view(&materialized, s, [n, k], [u64::from(k), 1])?);
        fp8.materialize(s, w, sc, &mut mat)?;
        for m in [1u32, 2, 63, 64, 65] {
            let xb = s.clone_htod(&vec![bf16::ONE; (m * k) as usize])?;
            let yb = s.alloc_zeros::<bf16>((m * n) as usize)?;
            let x = view(&xb, s, [m, k], [u64::from(k), 1])?;
            let mut y = TensorMut(view(&yb, s, [m, n], [u64::from(n), 1])?);
            blas.prepare_fixed(m, n, k)?;
            blas.linear_typed(
                ExecutionPhase::Prefill,
                LinearWeightRef::Bf16(mat.read()),
                x,
                &mut y,
            )?;
            let got = s.clone_dtoh(&yb)?;
            for (i, value) in got.iter().enumerate() {
                let want = bf16::from_f32(if (i % n as usize).is_multiple_of(2) {
                    k as f32
                } else {
                    -(k as f32)
                });
                ensure!(
                    value.to_bits() == want.to_bits(),
                    "Lt exact-sum mismatch {m}x{n}x{k} at {i}"
                );
            }
            if m == 1 {
                blas.linear_typed(
                    ExecutionPhase::Decode,
                    LinearWeightRef::BlockFp8 {
                        codes: w,
                        scale: sc,
                    },
                    x,
                    &mut y,
                )?;
                let decoded = s.clone_dtoh(&yb)?;
                ensure!(
                    got.iter()
                        .zip(decoded)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "direct GEMV exact-sum mismatch {n}x{k}"
                );
            }
        }
    }
    // Sparse inputs across scale blocks expose skipped effective-BF16 rounding.
    {
        let (n, k) = (129u32, 257u32);
        let codes: Vec<u8> = (0..n * k)
            .map(|i| [0x2d, 0x35, 0x41][i as usize % 3])
            .collect();
        let scales: Vec<bf16> = (0..6)
            .map(|i| bf16::from_f32([1.0078125, 0.03125, 3.140625][i % 3]))
            .collect();
        let mut inputs = vec![bf16::ZERO; k as usize];
        inputs[127] = bf16::ONE;
        inputs[128] = bf16::ONE;
        inputs[256] = bf16::from_f32(-1.0);
        let wb = s.clone_htod(&codes)?;
        let sb = s.clone_htod(&scales)?;
        let xb = s.clone_htod(&inputs)?;
        let yb = s.alloc_zeros::<bf16>(n as usize)?;
        let codes_view = view(&wb, s, [n, k], [u64::from(k), 1])?;
        let scales_view = view(&sb, s, [2, 3], [3, 1])?;
        let x = view(&xb, s, [1, k], [u64::from(k), 1])?;
        let mut y = TensorMut(view(&yb, s, [1, n], [u64::from(n), 1])?);
        blas.linear_typed(
            ExecutionPhase::Decode,
            LinearWeightRef::BlockFp8 {
                codes: codes_view,
                scale: scales_view,
            },
            x,
            &mut y,
        )?;
        let got = s.clone_dtoh(&yb)?;
        for row in 0..n as usize {
            let sum = [127usize, 128, 256]
                .iter()
                .map(|&col| {
                    effective_weight(
                        codes[row * k as usize + col],
                        scales[(row / 128) * 3 + col / 128],
                    )
                    .to_f32()
                        * inputs[col].to_f32()
                })
                .sum::<f32>();
            ensure!(
                got[row].to_bits() == bf16::from_f32(sum).to_bits(),
                "GEMV effective-weight rounding mismatch at row {row}"
            );
        }
    }
    if let Ok(path) = std::env::var("FF_Q38_OFFICIAL_SAMPLE") {
        let path = std::path::Path::new(&path);
        let dir = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("official fixture has no parent"))?;
        // Fixture uses the same safetensors basename and shared Loader as a checkpoint.
        let mapped = crate::checkpoint::MappedCheckpoint::open(dir)?;
        let views = mapped.views()?;
        let tensors = views
            .get("model.safetensors")
            .ok_or_else(|| anyhow::anyhow!("official fixture shard missing"))?;
        let manifest = crate::checkpoint::TensorManifest {
            entries: std::collections::BTreeMap::new(),
            resident_bytes: 204497280,
        };
        let uploaded = super::weights::checkpoint_mapped(&mapped, &d, Some(&manifest), |loader| {
            let mut values = Vec::new();
            for (label, n, k) in [
                ("q", 12288u32, 5120u32),
                ("gdn", 10240, 5120),
                ("mlp", 17408, 5120),
            ] {
                let codes =
                    loader.load_fp8(&format!("{label}.weight"), &[n as usize, k as usize])?;
                let scales = loader.load(&[(
                    format!("{label}.weight_scale_inv"),
                    vec![n.div_ceil(128) as usize, k.div_ceil(128) as usize],
                )])?;
                values.push((label, n, k, codes, scales));
            }
            Ok(values)
        })?;
        for (label, n, k, codes, scales) in uploaded {
            let source = tensors.tensor(&format!("{label}.weight"))?;
            let scale_source = tensors.tensor(&format!("{label}.weight_scale_inv"))?;
            let expected = |row: u32, col: u32| {
                let code = source.data()[(u64::from(row) * u64::from(k) + u64::from(col)) as usize];
                let index = ((row / 128) * k.div_ceil(128) + col / 128) as usize * 2;
                let scale = bf16::from_bits(u16::from_le_bytes([
                    scale_source.data()[index],
                    scale_source.data()[index + 1],
                ]));
                effective_weight(code, scale)
            };
            let w = view(&codes, s, [n, k], [u64::from(k), 1])?;
            let sc = view(
                &scales,
                s,
                [n.div_ceil(128), k.div_ceil(128)],
                [u64::from(k.div_ceil(128)), 1],
            )?;
            let materialized = s.alloc_zeros::<bf16>((u64::from(n) * u64::from(k)) as usize)?;
            let mut mat = TensorMut(view(&materialized, s, [n, k], [u64::from(k), 1])?);
            fp8.materialize(s, w, sc, &mut mat)?;
            for row in [0u32, 127, 128, n - 1] {
                let base = row as usize * k as usize;
                let got = s.clone_dtoh(&materialized.slice(base..base + k as usize))?;
                for (col, value) in got.iter().enumerate() {
                    ensure!(
                        value.to_bits() == expected(row, col as u32).to_bits(),
                        "official {label} dequant mismatch ({row},{col})"
                    );
                }
            }
            let mut xb = s.alloc_zeros::<bf16>(k as usize)?;
            let yb = s.alloc_zeros::<bf16>(n as usize)?;
            let x = view(&xb, s, [1, k], [u64::from(k), 1])?;
            let mut y = TensorMut(view(&yb, s, [1, n], [u64::from(n), 1])?);
            blas.prepare_fixed(1, n, k)?;
            for col in [127u32, 128, k - 1] {
                let mut onehot = vec![bf16::ZERO; k as usize];
                onehot[col as usize] = bf16::ONE;
                s.memcpy_htod(&onehot, &mut xb)?;
                for (phase, weight) in [
                    (ExecutionPhase::Prefill, LinearWeightRef::Bf16(mat.read())),
                    (
                        ExecutionPhase::Decode,
                        LinearWeightRef::BlockFp8 {
                            codes: w,
                            scale: sc,
                        },
                    ),
                ] {
                    blas.linear_typed(phase, weight, x, &mut y)?;
                    let got = s.clone_dtoh(&yb)?;
                    for (row, value) in got.iter().enumerate() {
                        ensure!(
                            value.to_f32() == expected(row as u32, col).to_f32(),
                            "official {label} {phase:?} onehot mismatch ({row},{col})"
                        );
                    }
                }
            }
            eprintln!(
                "Q38_OFFICIAL_MATRIX {label} N={n} K={k} raw_upload/dequant_samples/Lt_onehot/direct_GEMV_onehot PASS"
            );
        }
    } else {
        eprintln!("Q38_OFFICIAL_MATRIX SKIP: FF_Q38_OFFICIAL_SAMPLE unset");
    }
    let common = CommonOpsRef::new(&d.ctx)?;
    let (mut metadata, mref) = meta(s, 7, 2, 64)?;
    let rows = RowDomain {
        meta: mref,
        row_base: 0,
        row_capacity: 3,
    };
    for width in [128u32, 256, 5120] {
        let x = s.clone_htod(&vec![bf16::ONE; (3 * width) as usize])?;
        let weight = s.clone_htod(&vec![bf16::ZERO; width as usize])?;
        let y = s.clone_htod(&vec![bf16::from_f32(-17.0); (3 * width) as usize])?;
        let xv = view(
            &x,
            s,
            [3, 1, width],
            [u64::from(width), u64::from(width), 1],
        )?;
        let wv = view(&weight, s, [width], [1])?;
        let mut yv = TensorMut(view(&y, s, xv.shape, xv.stride)?);
        common.norm.rms_with(
            s,
            rows,
            xv,
            wv,
            &mut yv,
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: 1e-6,
            },
        )?;
        let got = s.clone_dtoh(&y)?;
        ensure!(
            got[..(2 * width) as usize]
                .iter()
                .all(|v| v.to_bits() == bf16::ONE.to_bits()),
            "zero-centered w=0 counterexample failed"
        );
        ensure!(
            got[(2 * width) as usize..]
                .iter()
                .all(|v| v.to_bits() == bf16::from_f32(-17.0).to_bits()),
            "inactive norm row was written"
        );
        common.norm.rms_with(
            s,
            rows,
            xv,
            wv,
            &mut yv,
            RmsOptions {
                mode: RmsMode::Standard,
                eps: 1e-6,
            },
        )?;
        ensure!(
            s.clone_dtoh(&y)?[..(2 * width) as usize]
                .iter()
                .all(|v| v.to_f32() == 0.0),
            "Standard norm did not use w directly"
        );
        common.norm.rms_gated(
            s,
            rows,
            GatedRmsInput {
                x: xv,
                weight: wv,
                gate: xv,
            },
            &mut yv,
            1e-6,
        )?;
        ensure!(
            s.clone_dtoh(&y)?[..(2 * width) as usize]
                .iter()
                .all(|v| v.to_f32() == 0.0),
            "gated Standard norm mismatch"
        );
    }
    let width = 128u32;
    let mut data = vec![bf16::ZERO; 3 * 128];
    data[0] = bf16::ONE;
    data[128] = bf16::from_f32(2.0);
    let x = s.clone_htod(&data)?;
    let y = s.alloc_zeros::<bf16>(3 * 128)?;
    let xv = view(&x, s, [3, 1, width], [128, 128, 1])?;
    let mut yv = TensorMut(view(&y, s, xv.shape, xv.stride)?);
    common.norm.l2_heads(s, rows, xv, &mut yv, 1e-6)?;
    let got = s.clone_dtoh(&y)?;
    for (i, v) in got.iter().enumerate() {
        ensure!(
            v.to_bits()
                == if i == 0 || i == 128 {
                    bf16::ONE.to_bits()
                } else {
                    0
                },
            "BF16 l2 one-hot mismatch"
        );
    }
    // This dyadic fixture distinguishes the mandated BF16 chain from FP32-only L2.
    // Rounded BF16 squares sum exactly to 87; the two algorithms differ at 85 lanes.
    let chain_data: Vec<bf16> = (0..3 * 128)
        .map(|i| bf16::from_f32(if i % 128 % 3 == 0 { 0.125 } else { 1.0078125 }))
        .collect();
    let chain_x = s.clone_htod(&chain_data)?;
    let chain_view = view(&chain_x, s, [3, 1, 128], [128, 128, 1])?;
    common.norm.l2_heads(s, rows, chain_view, &mut yv, 1e-6)?;
    let chain_y = s.clone_dtoh(&y)?;
    for (i, value) in chain_y[..256].iter().enumerate() {
        let want = bf16::from_f32(if i % 128 % 3 == 0 {
            55.0 / 4096.0
        } else {
            111.0 / 1024.0
        });
        ensure!(
            value.to_bits() == want.to_bits(),
            "L2 visible BF16 chain mismatch at {i}; FP32 shortcut would produce 0.10791015625 on large lanes"
        );
    }
    let zero = s.alloc_zeros::<bf16>(3 * 128)?;
    let zv = view(&zero, s, xv.shape, xv.stride)?;
    common.gate.sigmoid_mul(s, rows, xv, zv, &mut yv)?;
    let got = s.clone_dtoh(&y)?;
    ensure!(
        got[0].to_f32() == 0.5 && got[128].to_f32() == 1.0,
        "sigmoid gate boundary mismatch"
    );
    common.gate.silu(s, rows, zv, &mut yv)?;
    ensure!(
        s.clone_dtoh(&y)?.iter().all(|v| v.to_f32() == 0.0),
        "SiLU zero mismatch"
    );
    let a = s.alloc_zeros::<bf16>(3 * 48)?;
    let log = s.alloc_zeros::<bf16>(48)?;
    let beta = s.alloc_zeros::<bf16>(3 * 48)?;
    let g = s.alloc_zeros::<f32>(3 * 48)?;
    let av = view(&a, s, [3, 48], [48, 1])?;
    let lv = view(&log, s, [48], [1])?;
    let mut bv = TensorMut(view(&beta, s, [3, 48], [48, 1])?);
    let mut gv = TensorMut(view(&g, s, [3, 48], [48, 1])?);
    common.gate.gdn_gates_with(
        s,
        rows,
        GdnGateBindings {
            a: av,
            b: av,
            a_log: lv,
            dt_bias: lv,
            beta: &mut bv,
            g: &mut gv,
        },
    )?;
    ensure!(
        s.clone_dtoh(&beta)?[..96].iter().all(|v| v.to_f32() == 0.5),
        "beta zero-input mismatch"
    );
    let values = s.clone_dtoh(&g)?;
    let max_err = values[..96]
        .iter()
        .map(|v| (v + std::f32::consts::LN_2).abs())
        .fold(0f32, f32::max);
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite g for finite zero input"
    );
    // Captured common ops reread device valid_rows on every replay.
    s.synchronize()?;
    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let capture_result = common.norm.l2_heads(s, rows, chain_view, &mut yv, 1e-6);
    let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
    capture_result?;
    let graph = captured?.ok_or_else(|| anyhow::anyhow!("empty common-op graph"))?;
    let marker = vec![bf16::from_f32(-17.0); 3 * 128];
    let mut y_store = y;
    s.memcpy_htod(&marker, &mut y_store)?;
    s.memcpy_htod(&[9i32, 1, 10, 64, 1, 0, 0, 0], &mut metadata)?;
    graph.launch()?;
    let replayed = s.clone_dtoh(&y_store)?;
    ensure!(
        replayed[..128]
            .iter()
            .zip(&chain_y[..128])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Graph/eager L2 mismatch"
    );
    ensure!(
        replayed[128..].iter().all(|v| v.to_f32() == -17.0),
        "Graph captured stale valid_rows"
    );
    drop(graph);
    eprintln!(
        "Q38_PRIMITIVE_PROBE exact_dequant=true exact_linear_shapes=true visible_norm_boundaries=true g_zero_max_abs={max_err:e}; full G2/G3 acceptance requires reference fixtures"
    );
    Ok(())
}

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn q38_state_checkpoint_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    let s = &d.stream;
    let root: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?;
    let spec = crate::model_spec::ModelSpec::qwen(&root, &serde_json::Value::Null)?;
    let mut state = super::state::StateStore::new(&spec, 32, s)?;
    ensure!(
        state.gdn.len() == 48 && state.kv.len() == 16,
        "state mapping mismatch"
    );
    ensure!(
        state.bytes() == 48 * (3145728 + 122880) + 16 * 2 * 4 * 32 * 256 * 2 + 32,
        "state byte accounting mismatch"
    );
    state.seq_ref(s)?.validate()?;
    state.set_sequence(s, 5, 1, 1)?;
    for (i, st) in state.gdn.iter_mut().enumerate() {
        s.memcpy_htod(&vec![i as f32 + 1.0; st.s.len()], &mut st.s)?;
        s.memcpy_htod(&vec![i as f32 + 11.0; st.history.len()], &mut st.history)?;
    }
    for kv in &mut state.kv {
        s.memcpy_htod(&vec![bf16::from_f32(-2.0); kv.k.len()], &mut kv.k)?;
        s.memcpy_htod(&vec![bf16::from_f32(3.0); kv.v.len()], &mut kv.v)?;
    }
    let mut token = s.clone_htod(&[42u32])?;
    let mut checkpoint = super::state::StateCheckpoint::new(&state, &spec, 1, s)?;
    ensure!(
        checkpoint.bytes() == 48 * (3145728 + 122880) + 16 * 2 * 4 * 256 * 2 + 36,
        "checkpoint byte accounting mismatch"
    );
    ensure!(
        checkpoint.restore(&mut state, &mut token, s).is_err(),
        "empty snapshot restored"
    );
    ensure!(
        checkpoint.save(&state, &token, 4, s).is_err(),
        "stale host prefix accepted"
    );
    checkpoint.save(&state, &token, 5, s)?;
    state.reset(s)?;
    for kv in &mut state.kv {
        s.memset_zeros(&mut kv.k)?;
        s.memset_zeros(&mut kv.v)?;
    }
    s.memcpy_htod(&[99u32], &mut token)?;
    checkpoint.restore(&mut state, &mut token, s)?;
    for (i, st) in state.gdn.iter().enumerate() {
        ensure!(
            s.clone_dtoh(&st.s)?.iter().all(|&v| v == i as f32 + 1.0),
            "S state mixed across layers"
        );
        ensure!(
            s.clone_dtoh(&st.history)?
                .iter()
                .all(|&v| v == i as f32 + 11.0),
            "history state mixed across layers"
        );
    }
    for kv in &state.kv {
        for (buf, value) in [(&kv.k, -2.0f32), (&kv.v, 3.0)] {
            let got = s.clone_dtoh(buf)?;
            for (i, v) in got.iter().enumerate() {
                let position = (i / 256) % 32;
                ensure!(
                    v.to_f32() == if position == 5 { value } else { 0.0 },
                    "checkpoint wrote outside saved KV window"
                );
            }
        }
    }
    ensure!(
        s.clone_dtoh(state.sequence.as_ref().unwrap())? == [5, 1, 6, 32, 1, 0, 0, 0],
        "sequence metadata not restored"
    );
    ensure!(s.clone_dtoh(&token)? == [42], "token not restored");
    state.reset(s)?;
    for st in &state.gdn {
        ensure!(
            s.clone_dtoh(&st.s)?
                .iter()
                .chain(s.clone_dtoh(&st.history)?.iter())
                .all(|&v| v == 0.0),
            "reset retained GDN state"
        );
    }
    ensure!(
        s.clone_dtoh(state.sequence.as_ref().unwrap())? == [0, 1, 1, 32, 1, 0, 0, 0],
        "reset metadata mismatch"
    );
    // Decode metadata advances on the device, including across repeated replays.
    state.set_sequence(s, 5, 1, 1)?;
    let advance = super::cubin::module(&d.ctx, "q38_common")?.load_function("q38_advance")?;
    let seq = state.seq_ref(s)?;
    use cudarc::driver::PushKernelArg;
    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launched = unsafe {
        s.launch_builder(&advance)
            .arg(&seq.ptr)
            .launch(super::ops::grid(1, 1))
    };
    let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
    launched?;
    let graph = captured?.ok_or_else(|| anyhow::anyhow!("empty metadata graph"))?;
    for prefix in [6, 7, 8] {
        graph.launch()?;
        ensure!(
            s.clone_dtoh(state.sequence.as_ref().unwrap())?
                == [prefix, 1, prefix + 1, 32, 1, 0, 0, 0],
            "graph reused stale sequence prefix"
        );
    }
    checkpoint.restore(&mut state, &mut token, s)?;
    ensure!(
        s.clone_dtoh(state.sequence.as_ref().unwrap())? == [5, 1, 6, 32, 1, 0, 0, 0],
        "checkpoint did not restore graph-mutated metadata"
    );
    eprintln!(
        "Q38_STATE_PROBE mapping/byte_accounting/nonzero_prefix/S/history/KV_window/token/reset/repeated_metadata_graph PASS"
    );
    Ok(())
}

#[test]
#[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
fn q38_vision_common_gpu() -> Result<()> {
    use super::{linear::LinearBiasMode, ops::GeluMode};
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    let s = &device.stream;
    let common = CommonOpsRef::new(&device.ctx)?;
    let ops = Ops::new(&device.ctx)?;
    let closure = Rc::new(Closure::new(false, "explicit common-op probe".into()));
    let mut blas = Blas::new(&device, &ops, &std::env::temp_dir(), closure)?;
    // A nonzero mean distinguishes LayerNorm from RMS; constant row must equal bias.
    let width = 1152usize;
    let host: Vec<_> = (0..2 * width)
        .map(|i| {
            bf16::from_f32(if i < width {
                if i % 2 == 0 { 1.0 } else { 5.0 }
            } else {
                7.0
            })
        })
        .collect();
    let x = s.clone_htod(&host)?;
    let weight = s.clone_htod(&vec![bf16::from_f32(0.5); width])?;
    let bias = s.clone_htod(&vec![bf16::from_f32(0.25); width])?;
    let out = s.clone_htod(&vec![bf16::NAN; 2 * width])?;
    let xv = view(&x, s, [2, 1, width as u32], [width as u64, width as u64, 1])?;
    let yv = view(
        &out,
        s,
        [2, 1, width as u32],
        [width as u64, width as u64, 1],
    )?;
    let wv = view(&weight, s, [width as u32], [1])?;
    let bv = view(&bias, s, [width as u32], [1])?;
    common
        .norm
        .layer_norm(s, xv, wv, bv, &mut TensorMut(yv), 1e-6)?;
    let expected: Vec<_> = (0..2 * width)
        .map(|i| {
            bf16::from_f32(if i >= width {
                0.25
            } else if i % 2 == 0 {
                -0.25
            } else {
                0.75
            })
        })
        .collect();
    ensure!(
        s.clone_dtoh(&out)? == expected,
        "LayerNorm affine/centered variance mismatch"
    );
    common
        .norm
        .layer_norm(s, xv, wv, bv, &mut TensorMut(xv), 1e-6)?;
    ensure!(s.clone_dtoh(&x)? == expected, "in-place LayerNorm mismatch");
    // erf gold values use the mathematical CDF, not the device tanh approximation.
    let inputs = [-3f32, -2., -1., 0., 1., 2., 3., 0.5];
    let exact = [
        -0.00404969409489031,
        -0.04550026389635842,
        -0.15865525393145707,
        0.,
        0.8413447460685429,
        1.9544997361036416,
        2.99595030590511,
        0.34573123063700656,
    ];
    let gx = s.clone_htod(&inputs.map(bf16::from_f32))?;
    let gy = s.clone_htod(&[bf16::NAN; 8])?;
    let gv = view(&gx, s, [1, 1, 8], [8, 8, 1])?;
    let gyv = view(&gy, s, [1, 1, 8], [8, 8, 1])?;
    common
        .gate
        .gelu(s, gv, &mut TensorMut(gyv), GeluMode::Exact)?;
    let got_exact = s.clone_dtoh(&gy)?;
    ensure!(
        got_exact == exact.map(bf16::from_f64),
        "exact GELU mismatch"
    );
    common
        .gate
        .gelu(s, gv, &mut TensorMut(gyv), GeluMode::Tanh)?;
    let got_tanh = s.clone_dtoh(&gy)?;
    let tanh = inputs.map(|v| {
        let v = f64::from(v);
        bf16::from_f64(
            0.5 * v
                * (1. + ((2. / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh()),
        )
    });
    ensure!(
        got_tanh == tanh && got_exact[0] != got_tanh[0],
        "GELU modes not distinguished"
    );
    // 1.0078125^2 - 1.015625 = 2^-14 before BF16 rounding; after-round add is zero.
    let mut wh = vec![bf16::ZERO; 32 * 128];
    for row in 0..32 {
        wh[row * 128] = bf16::from_f32(1.0078125);
    }
    let mut xh = vec![bf16::ZERO; 2 * 128];
    xh[0] = bf16::from_f32(1.0078125);
    xh[128] = xh[0];
    let lw = s.clone_htod(&wh)?;
    let lx = s.clone_htod(&xh)?;
    let lb = s.clone_htod(&[bf16::from_f32(-1.015625); 32])?;
    let ly = s.clone_htod(&[bf16::NAN; 64])?;
    let lwv = view(&lw, s, [32, 128], [128, 1])?;
    let lxv = view(&lx, s, [2, 128], [128, 1])?;
    let lbv = view(&lb, s, [32], [1])?;
    let lyv = view(&ly, s, [2, 32], [32, 1])?;
    for (mode, expected) in [
        (
            LinearBiasMode::BeforeOutputRound,
            bf16::from_f32(2f32.powi(-14)),
        ),
        (LinearBiasMode::AfterOutputRound, bf16::ZERO),
    ] {
        blas.prepare_biased(2, 32, 128, lbv, mode)?;
        blas.linear_biased(lwv, lbv, lxv, &mut TensorMut(lyv), mode)?;
        ensure!(
            s.clone_dtoh(&ly)?.iter().all(|v| *v == expected),
            "bias rounding mode {mode:?} mismatch"
        );
    }
    // Prepared descriptors and common ops must capture without setup or allocation.
    device.finish_upload()?;
    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launches = (|| -> Result<()> {
        common
            .gate
            .gelu(s, gv, &mut TensorMut(gyv), GeluMode::Exact)?;
        blas.linear_biased(
            lwv,
            lbv,
            lxv,
            &mut TensorMut(lyv),
            LinearBiasMode::BeforeOutputRound,
        )
    })();
    let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
    launches?;
    let graph = captured?.ok_or_else(|| anyhow::anyhow!("empty common-op graph"))?;
    graph.launch()?;
    ensure!(s.clone_dtoh(&gy)? == got_exact, "GELU graph/eager mismatch");
    ensure!(
        s.clone_dtoh(&ly)?
            .iter()
            .all(|v| *v == bf16::from_f32(2f32.powi(-14))),
        "bias graph/eager mismatch"
    );
    if let Some(dir) = std::env::var_os("FF_Q38_OPREF_DIR") {
        vision_opref_diagnostics(&device, &common, &mut blas, &std::path::PathBuf::from(dir))?;
    }
    eprintln!(
        "Q38_VISION_COMMON_PROBE LayerNorm/alias/GELU_modes/bias_rounding/graph PASS; reference qualification requires per-case limits"
    );
    Ok(())
}

fn vision_opref_diagnostics(
    d: &Device,
    common: &CommonOpsRef,
    blas: &mut Blas,
    dir: &std::path::Path,
) -> Result<()> {
    use super::{linear::LinearBiasMode, ops::GeluMode};
    use safetensors::tensor::TensorView;
    let s = &d.stream;
    let dump = std::env::var_os("FF_Q38_OPREF_DUMP").map(std::path::PathBuf::from);
    let selected = std::env::var("FF_Q38_OPREF_KINDS").unwrap_or_else(|_| "linear,ln,gelu".into());
    ensure!(
        !selected.is_empty()
            && selected
                .split(',')
                .all(|k| matches!(k, "linear" | "ln" | "gelu")),
        "invalid OpRef operator selection"
    );
    if let Some(path) = &dump {
        std::fs::create_dir_all(path)?;
    }
    for (kind, expected_cases) in [("linear", 8), ("ln", 10), ("gelu", 6)] {
        if !selected.split(',').any(|k| k == kind) {
            continue;
        }
        let mut outputs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let file = std::fs::File::open(dir.join(format!("opref-vision-{kind}.sm89.safetensors")))?;
        let mapped = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&mapped)?;
        let load = |name: &str| -> Result<Vec<bf16>> {
            let t = tensors.tensor(name)?;
            ensure!(
                t.dtype() == safetensors::Dtype::BF16,
                "OpRef tensor is not BF16: {name}"
            );
            Ok(t.data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
                .collect())
        };
        let mut cases: Vec<_> = tensors
            .names()
            .iter()
            .filter_map(|n| n.strip_suffix(".x").map(str::to_owned))
            .collect();
        cases.sort();
        ensure!(
            cases.len() == expected_cases,
            "partial OpRef fixture {kind}: {} cases",
            cases.len()
        );
        for case in cases {
            let shape = tensors.tensor(&format!("{case}.x"))?.shape().to_vec();
            ensure!(
                shape.len() == 2 && shape.iter().all(|&v| v > 0),
                "OpRef must contain nonempty row matrices"
            );
            let (rows, width) = (shape[0] as u32, shape[1] as u32);
            let x = s.clone_htod(&load(&format!("{case}.x"))?)?;
            let gold = load(&format!(
                "{case}.{}",
                if kind == "linear" { "out_runtime" } else { "y" }
            ))?;
            let output_width = u32::try_from(gold.len() / rows as usize)?;
            ensure!(
                gold.len() == rows as usize * output_width as usize,
                "OpRef output shape mismatch"
            );
            let out = s.clone_htod(&vec![bf16::NAN; gold.len()])?;
            if kind == "linear" {
                let w = s.clone_htod(&load(&format!("{case}.w"))?)?;
                let b = s.clone_htod(&load(&format!("{case}.b"))?)?;
                let bias = view(&b, s, [output_width], [1])?;
                blas.prepare_biased(
                    rows,
                    output_width,
                    width,
                    bias,
                    LinearBiasMode::BeforeOutputRound,
                )?;
                blas.linear_biased(
                    view(&w, s, [output_width, width], [u64::from(width), 1])?,
                    bias,
                    view(&x, s, [rows, width], [u64::from(width), 1])?,
                    &mut TensorMut(view(
                        &out,
                        s,
                        [rows, output_width],
                        [u64::from(output_width), 1],
                    )?),
                    LinearBiasMode::BeforeOutputRound,
                )?;
                s.synchronize()?;
            } else {
                ensure!(
                    output_width == width,
                    "OpRef pointwise output width mismatch"
                );
                let xv = view(
                    &x,
                    s,
                    [rows, 1, width],
                    [u64::from(width), u64::from(width), 1],
                )?;
                let mut y = TensorMut(view(
                    &out,
                    s,
                    [rows, 1, width],
                    [u64::from(width), u64::from(width), 1],
                )?);
                if kind == "ln" {
                    let w = s.clone_htod(&load(&format!("{case}.weight"))?)?;
                    let b = s.clone_htod(&load(&format!("{case}.bias"))?)?;
                    common.norm.layer_norm(
                        s,
                        xv,
                        view(&w, s, [width], [1])?,
                        view(&b, s, [width], [1])?,
                        &mut y,
                        1e-6,
                    )?;
                    s.synchronize()?;
                } else {
                    let mode = if case.contains("gelu_exact") {
                        GeluMode::Exact
                    } else {
                        ensure!(case.contains("gelu_tanh"), "unknown GELU OpRef mode");
                        GeluMode::Tanh
                    };
                    common.gate.gelu(s, xv, &mut y, mode)?;
                }
            }
            let got = s.clone_dtoh(&out)?;
            ensure!(
                got.iter().all(|v| v.is_finite()) && gold.iter().all(|v| v.is_finite()),
                "OpRef output is unwritten/nonfinite"
            );
            let equal = got
                .iter()
                .zip(&gold)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            let max_abs = got
                .iter()
                .zip(&gold)
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0f32, f32::max);
            if kind == "gelu" && equal != gold.len() {
                let input = s.clone_dtoh(&x)?;
                let mut seen = std::collections::BTreeSet::new();
                for ((a, b), v) in got.iter().zip(&gold).zip(input) {
                    if a.to_bits() != b.to_bits() && seen.insert(v.to_bits()) {
                        eprintln!(
                            "Q38_GELU_DIFF case={case} x={} got={} ref={} got_bits={} ref_bits={}",
                            v.to_f32(),
                            a.to_f32(),
                            b.to_f32(),
                            a.to_bits(),
                            b.to_bits()
                        );
                    }
                }
            }
            eprintln!(
                "Q38_VISION_OPREF DIAGNOSTIC kind={kind} case={case} equal={equal}/{} max_abs={max_abs:e}; acceptance requires per-case reference limits",
                gold.len()
            );
            if dump.is_some() {
                outputs.push((
                    format!("{case}.candidate"),
                    vec![rows as usize, output_width as usize],
                    got.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
                ));
                if kind == "linear" {
                    let w = s.clone_htod(&load(&format!("{case}.w"))?)?;
                    let b = s.clone_htod(&load(&format!("{case}.b"))?)?;
                    let bias = view(&b, s, [output_width], [1])?;
                    blas.prepare_biased(
                        rows,
                        output_width,
                        width,
                        bias,
                        LinearBiasMode::AfterOutputRound,
                    )?;
                    blas.linear_biased(
                        view(&w, s, [output_width, width], [u64::from(width), 1])?,
                        bias,
                        view(&x, s, [rows, width], [u64::from(width), 1])?,
                        &mut TensorMut(view(
                            &out,
                            s,
                            [rows, output_width],
                            [u64::from(output_width), 1],
                        )?),
                        LinearBiasMode::AfterOutputRound,
                    )?;
                    s.synchronize()?;
                } else {
                    let xv = view(
                        &x,
                        s,
                        [rows, 1, width],
                        [u64::from(width), u64::from(width), 1],
                    )?;
                    let mut y = TensorMut(view(
                        &out,
                        s,
                        [rows, 1, width],
                        [u64::from(width), u64::from(width), 1],
                    )?);
                    if kind == "ln" {
                        let w = s.clone_htod(&load(&format!("{case}.weight"))?)?;
                        let b = s.clone_htod(&load(&format!("{case}.bias"))?)?;
                        common.norm.layer_norm(
                            s,
                            xv,
                            view(&w, s, [width], [1])?,
                            view(&b, s, [width], [1])?,
                            &mut y,
                            1.0,
                        )?;
                        s.synchronize()?;
                    } else {
                        let wrong = if case.contains("gelu_exact") {
                            GeluMode::Tanh
                        } else {
                            GeluMode::Exact
                        };
                        common.gate.gelu(s, xv, &mut y, wrong)?;
                    }
                }
                let control = s.clone_dtoh(&out)?;
                ensure!(
                    control.iter().all(|v| v.is_finite()),
                    "negative control failed to produce finite outputs"
                );
                outputs.push((
                    format!("{case}.control"),
                    vec![rows as usize, output_width as usize],
                    control
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect(),
                ));
            }
        }
        if let Some(path) = &dump {
            let tensors = outputs
                .iter()
                .map(|(name, shape, data)| {
                    Ok((
                        name.as_str(),
                        TensorView::new(safetensors::Dtype::BF16, shape.clone(), data)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            safetensors::serialize_to_file(
                tensors,
                None,
                &path.join(format!("{kind}.safetensors")),
            )?;
            eprintln!(
                "Q38_OPREF_DUMP kind={kind} tensors={} candidate+control; numerical verdict pending frozen rules",
                outputs.len()
            );
        }
    }
    Ok(())
}
