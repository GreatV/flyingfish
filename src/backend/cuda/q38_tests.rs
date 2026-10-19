//! GPU probes. Default cargo test never initializes a GPU here.
use super::{
    Device,
    attention_d256::{AttentionBindings, D, FullAttentionOps, PositionBindings},
    blas::Blas,
    closure::Closure,
    contracts::{RowDomain, Scratch, SeqRef, Tensor, TensorMut},
    cubin,
    linear::{Fp8Ops, LinearWeightRef},
    ops::{
        CommonOpsRef, GatedRmsInput, GdnGateBindings, Ops, ResidualRmsInput, RmsMode, RmsOptions,
        grid,
    },
};
use crate::{
    fp8::{Fp8ScaleLayout, effective_weight},
    op_contract::{
        AttentionLimits, AttentionWorkspace, ExecutionPhase, FullAttentionSpec, SequenceMeta,
    },
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::{rc::Rc, sync::Arc};

pub(super) fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
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
        super::weights::checkpoint_mapped::<()>(&mapped, &d, Some(&manifest), false, |loader| {
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
        .map(|(value, _)| value)
    })();
    std::fs::remove_dir_all(&fixture_dir)?;
    uploaded?;
    let fp8 = Fp8Ops::new(&d.ctx)?;
    // All codes, including the two NaNs, plus non-integral 128-block scale tails.
    for layout in [Fp8ScaleLayout::Block128, Fp8ScaleLayout::PerChannel] {
        for (n, k) in [(1u32, 256u32), (129, 257)] {
            let codes: Vec<u8> = (0..n * k).map(|i| i as u8).collect();
            let shape = layout.scale_shape(n, k).map(|v| v as u32);
            let scales: Vec<bf16> = (0..shape[0] * shape[1])
                .map(|i| bf16::from_f32([1.0078125, 0.03125, 3.140625][i as usize % 3]))
                .collect();
            let wb = s.clone_htod(&codes)?;
            let mut sb = s.clone_htod(&scales)?;
            let y = s.alloc_zeros::<bf16>((n * k) as usize)?;
            let w = view(&wb, s, [n, k], [u64::from(k), 1])?;
            let sc = view(&sb, s, shape, [u64::from(shape[1]), 1])?;
            let mut out = TensorMut(view(&y, s, [n, k], [u64::from(k), 1])?);
            fp8.materialize(s, w, sc, layout, &mut out)?;
            let got = s.clone_dtoh(&y)?;
            for row in 0..n {
                for col in 0..k {
                    let i = (row * k + col) as usize;
                    let want = effective_weight(
                        codes[i],
                        scales[match layout {
                            Fp8ScaleLayout::Block128 => {
                                ((row / 128) * k.div_ceil(128) + col / 128) as usize
                            }
                            Fp8ScaleLayout::PerChannel => row as usize,
                        }],
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
            if layout == Fp8ScaleLayout::PerChannel && n > 1 {
                let mut permuted = scales.clone();
                permuted.rotate_left(1);
                s.memcpy_htod(&permuted, &mut sb)?;
                fp8.materialize(s, w, sc, layout, &mut out)?;
                let wrong = s.clone_dtoh(&y)?;
                let differences = got
                    .iter()
                    .zip(&wrong)
                    .filter(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() != b.to_bits())
                    .count();
                ensure!(differences > 0, "FP8 swapped-row-scale control undetected");
                eprintln!(
                    "PASS FP8_SWAPPED_ROW_SCALE stored_weight=[{n},{k}] layout={layout:?} mismatches={differences}"
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
        fp8.materialize(s, w, sc, Fp8ScaleLayout::Block128, &mut mat)?;
        for m in [1u32, 2, 3, 4, 5, 63, 64, 65] {
            let row_values = [1.0f32, -1.0, 2.0, -2.0, 4.0];
            let inputs: Vec<_> = (0..m * k)
                .map(|i| bf16::from_f32(row_values[(i / k) as usize % 5]))
                .collect();
            let xb = s.clone_htod(&inputs)?;
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
            ensure!(
                got.iter().all(|v| v.is_finite()),
                "Lt reference non-finite M={m}"
            );
            for (i, value) in got.iter().enumerate() {
                let want = bf16::from_f32(if (i % n as usize).is_multiple_of(2) {
                    k as f32 * row_values[(i / n as usize) % 5]
                } else {
                    -(k as f32) * row_values[(i / n as usize) % 5]
                });
                ensure!(
                    value.to_bits() == want.to_bits(),
                    "Lt exact-sum mismatch {m}x{n}x{k} at {i}"
                );
            }
            if m <= 5 {
                blas.linear_typed(
                    if m == 1 {
                        ExecutionPhase::Decode
                    } else {
                        ExecutionPhase::Prefill
                    },
                    LinearWeightRef::Fp8 {
                        layout: Fp8ScaleLayout::Block128,
                        codes: w,
                        scale: sc,
                    },
                    x,
                    &mut y,
                )?;
                let decoded = s.clone_dtoh(&yb)?;
                ensure!(
                    decoded.iter().all(|v| v.is_finite()),
                    "direct FP8 non-finite M={m}"
                );
                ensure!(
                    got.iter()
                        .zip(decoded)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "direct FP8 exact-sum mismatch {m}x{n}x{k}"
                );
            }
        }
    }
    // Sparse inputs across scale blocks expose skipped effective-BF16 rounding.
    for layout in [Fp8ScaleLayout::Block128, Fp8ScaleLayout::PerChannel] {
        for (n, k) in [(129u32, 257u32), (6144, 5120), (8192, 2048)] {
            let columns: &[usize] = if k == 257 {
                &[127, 128, 256]
            } else if k == 2048 {
                &[0, 127, 128, 511, 512, 1023, 1024, 2047]
            } else {
                &[0, 127, 128, 511, 512, 1023, 1024, 5119]
            };
            let shape = layout.scale_shape(n, k).map(|v| v as u32);
            let scale_cols = shape[1];
            let codes: Vec<u8> = (0..n * k)
                .map(|i| [0x2d, 0x35, 0x41][i as usize % 131 % 3])
                .collect();
            let scales: Vec<bf16> = (0..shape[0] * scale_cols)
                .map(|i| bf16::from_f32([1.0078125, 0.03125, 3.140625][i as usize % 137 % 3]))
                .collect();
            let wb = s.clone_htod(&codes)?;
            let sb = s.clone_htod(&scales)?;
            let codes_view = view(&wb, s, [n, k], [u64::from(k), 1])?;
            let scales_view = view(&sb, s, shape, [u64::from(scale_cols), 1])?;
            for m in 1u32..=5 {
                let mut inputs = vec![bf16::ZERO; (m * k) as usize];
                for row in 0..m as usize {
                    for (i, &col) in columns.iter().enumerate() {
                        let coefficients = [1.0f32, 1.0, -1.0, 2.0, -2.0, 1.0, 1.0, -1.0];
                        let row_values = [1.0f32, -1.0, 2.0, -2.0, 4.0];
                        inputs[row * k as usize + col] =
                            bf16::from_f32(coefficients[i] * row_values[row]);
                    }
                }
                let mut xb = s.clone_htod(&inputs)?;
                let yb = s.alloc_zeros::<bf16>((m * n) as usize)?;
                let x = view(&xb, s, [m, k], [u64::from(k), 1])?;
                let mut y = TensorMut(view(&yb, s, [m, n], [u64::from(n), 1])?);
                blas.linear_typed(
                    if m == 1 {
                        ExecutionPhase::Decode
                    } else {
                        ExecutionPhase::Prefill
                    },
                    LinearWeightRef::Fp8 {
                        layout,
                        codes: codes_view,
                        scale: scales_view,
                    },
                    x,
                    &mut y,
                )?;
                let got = s.clone_dtoh(&yb)?;
                ensure!(
                    got.iter().all(|v| v.is_finite()),
                    "direct FP8 sparse non-finite M={m}"
                );
                let expected: Vec<_> = (0..(m * n) as usize)
                    .map(|index| {
                        let row = index / n as usize;
                        let out = index % n as usize;
                        bf16::from_f32(
                            columns
                                .iter()
                                .map(|&col| {
                                    effective_weight(
                                        codes[out * k as usize + col],
                                        scales[match layout {
                                            Fp8ScaleLayout::Block128 => {
                                                (out / 128) * scale_cols as usize + col / 128
                                            }
                                            Fp8ScaleLayout::PerChannel => out,
                                        }],
                                    )
                                    .to_f32()
                                        * inputs[row * k as usize + col].to_f32()
                                })
                                .sum::<f32>(),
                        )
                    })
                    .collect();
                ensure!(
                    got.iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "direct FP8 effective-weight rounding mismatch M={m}"
                );
                if m > 1 {
                    let rotated: Vec<_> = inputs[k as usize..]
                        .iter()
                        .chain(&inputs[..k as usize])
                        .copied()
                        .collect();
                    s.memcpy_htod(&rotated, &mut xb)?;
                    blas.linear_typed(
                        ExecutionPhase::Prefill,
                        LinearWeightRef::Fp8 {
                            layout,
                            codes: codes_view,
                            scale: scales_view,
                        },
                        x,
                        &mut y,
                    )?;
                    let mutant = s.clone_dtoh(&yb)?;
                    ensure!(
                        mutant.iter().all(|v| v.is_finite()),
                        "direct FP8 mutant non-finite M={m}"
                    );
                    ensure!(
                        mutant.iter().enumerate().all(|(i, v)| v.is_finite()
                            && v.to_bits()
                                == expected[((i / n as usize + 1) % m as usize) * n as usize
                                    + i % n as usize]
                                    .to_bits()),
                        "FP8 rotated-input reference mismatch M={m}"
                    );
                    let mutant_mismatches = mutant
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    ensure!(
                        mutant_mismatches > 0,
                        "FP8 wrong-row mutant undetected M={m}"
                    );
                    eprintln!("Q38_FP8_ROW_MUTANT M={m} mismatches={mutant_mismatches} PASS");
                }
                eprintln!(
                    "Q38_FP8_DIRECT layout={layout:?} M={m} stored_weight=[{n},{k}] scale=[{},{}] exact=PASS",
                    shape[0], scale_cols
                );
            }
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
        let (uploaded, _) =
            super::weights::checkpoint_mapped(&mapped, &d, Some(&manifest), false, |loader| {
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
            fp8.materialize(s, w, sc, Fp8ScaleLayout::Block128, &mut mat)?;
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
                        LinearWeightRef::Fp8 {
                            layout: Fp8ScaleLayout::Block128,
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
    for width in [128u32, 17_408] {
        let count = 3 * width as usize;
        let swiglu_gate = (0..count)
            .map(|i| bf16::from_f32(((i % 17) as f32) * 0.125 - 1.0))
            .collect::<Vec<_>>();
        let swiglu_up = (0..count)
            .map(|i| bf16::from_f32(1.0 + ((i % 5) as f32) * 0.25))
            .collect::<Vec<_>>();
        let gate_x = s.clone_htod(&swiglu_gate)?;
        let up_x = s.clone_htod(&swiglu_up)?;
        let baseline = s.alloc_zeros::<bf16>(count)?;
        let fused = s.clone_htod(&vec![bf16::from_f32(-17.0); count])?;
        let gate_view = view(
            &gate_x,
            s,
            [3, 1, width],
            [u64::from(width), u64::from(width), 1],
        )?;
        let up_view = view(&up_x, s, gate_view.shape, gate_view.stride)?;
        let mut baseline_view = TensorMut(view(&baseline, s, gate_view.shape, gate_view.stride)?);
        let mut fused_view = TensorMut(view(&fused, s, gate_view.shape, gate_view.stride)?);
        common.gate.silu(s, rows, gate_view, &mut baseline_view)?;
        common
            .gate
            .swiglu(s, rows, gate_view, up_view, &mut fused_view)?;
        let activated = s.clone_dtoh(&baseline)?;
        let multiplied = s.clone_dtoh(&fused)?;
        ensure!(
            activated.iter().chain(&multiplied).all(|v| v.is_finite()),
            "nonfinite SwiGLU component output"
        );
        let mismatches = (0..2 * width as usize)
            .filter(|&i| {
                multiplied[i].to_bits()
                    != bf16::from_f32(activated[i].to_f32() * swiglu_up[i].to_f32()).to_bits()
            })
            .count();
        let inactive = multiplied[2 * width as usize..]
            .iter()
            .filter(|v| v.to_bits() != bf16::from_f32(-17.0).to_bits())
            .count();
        eprintln!(
            "Q38_SWIGLU width={width} finite=true active_mismatches={mismatches} inactive_mismatches={inactive}"
        );
        for i in 0..2 * width as usize {
            let want = bf16::from_f32(activated[i].to_f32() * swiglu_up[i].to_f32());
            ensure!(
                multiplied[i].to_bits() == want.to_bits(),
                "fused SwiGLU bit mismatch at {i}"
            );
        }
        ensure!(
            multiplied[2 * width as usize..]
                .iter()
                .all(|v| v.to_bits() == bf16::from_f32(-17.0).to_bits()),
            "inactive fused SwiGLU row was written"
        );
    }
    let a = s.alloc_zeros::<bf16>(3 * 48)?;
    let log = s.alloc_zeros::<bf16>(48)?;
    let beta = s.alloc_zeros::<bf16>(3 * 48)?;
    let g = s.alloc_zeros::<f32>(3 * 48)?;
    let av = view(&a, s, [3, 48], [48, 1])?;
    let lv = view(&log, s, [48], [1])?;
    let mut bv = TensorMut(view(&beta, s, [3, 48], [48, 1])?);
    let mut gv = TensorMut(view(&g, s, [3, 48], [48, 1])?);
    super::ops::GateOps::gdn_gates_with(
        &cubin::module(&d.ctx, "q38_common")?
            .load_function(crate::op_contract::GdnSpec::q38().gate_kernel()?)?,
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

const VERIFY_M4_ROWS: usize = 4;
const VERIFY_CAUSAL_MASK_LEN: usize = 64;

fn verify_bf16(
    stage: &str,
    case: u32,
    row: usize,
    a: &[bf16],
    b: &[bf16],
    coordinate: impl Fn(usize) -> String,
) -> Result<Option<String>> {
    ensure!(
        a.len() == b.len(),
        "VERIFY_M4 extent mismatch stage={stage} case={case}"
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "VERIFY_M4 non-finite stage={stage} case={case} row={row}"
    );
    let mut first = None;
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            first = Some(format!(
                "VERIFY_M4 first_diff stage={stage} case={case} row={row} at={} batch={:#04x} serial={:#04x}",
                coordinate(index),
                x.to_bits(),
                y.to_bits()
            ));
            break;
        }
    }
    Ok(first)
}

fn verify_f32(
    stage: &str,
    case: u32,
    row: usize,
    a: &[f32],
    b: &[f32],
    coordinate: impl Fn(usize) -> String,
) -> Result<Option<String>> {
    ensure!(
        a.len() == b.len(),
        "VERIFY_M4 extent mismatch stage={stage} case={case}"
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "VERIFY_M4 non-finite stage={stage} case={case} row={row}"
    );
    let mut first = None;
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            first = Some(format!(
                "VERIFY_M4 first_diff stage={stage} case={case} row={row} at={} batch={:#010x} serial={:#010x}",
                coordinate(index),
                x.to_bits(),
                y.to_bits()
            ));
            break;
        }
    }
    Ok(first)
}

fn verify_record(
    first: &mut Option<String>,
    defects: &mut usize,
    words: &mut usize,
    count: usize,
    diff: Option<String>,
) {
    *words += count;
    if diff.is_some() {
        *defects += 1;
        if first.is_none() {
            *first = diff;
        }
    }
}

fn verify_projection_m4(
    device: &Device,
    blas: &Blas,
    output: u32,
    input: u32,
    state: &mut u64,
) -> Result<(usize, Option<String>)> {
    let code_set = [0x20u8, 0x35, 0x48, 0x59, 0x68, 0x80, 0x99, 0xb0, 0xc0, 0xd8];
    let codes = (0..u64::from(output) * u64::from(input))
        .map(|i| code_set[(i % 17) as usize % code_set.len()])
        .collect::<Vec<_>>();
    let scales = (0..output.div_ceil(128) * input.div_ceil(128))
        .map(|i| {
            [
                bf16::from_f32(0.25),
                bf16::ONE,
                bf16::from_f32(1.0078125),
                bf16::from_f32(2.0),
                bf16::from_f32(0.03125),
            ][(i % 13) as usize % 5]
        })
        .collect::<Vec<_>>();
    let rows = (0..VERIFY_M4_ROWS * input as usize)
        .map(|_| bf16::from_f32(next_f32_unit(state) * 3.0 - 1.5))
        .collect::<Vec<_>>();

    let s = &device.stream;
    let code_buffer = s.clone_htod(&codes)?;
    let scale_buffer = s.clone_htod(&scales)?;
    let row_buffer = s.clone_htod(&rows)?;
    let batch_out = s.clone_htod(&vec![bf16::ZERO; VERIFY_M4_ROWS * output as usize])?;
    let scale_cols = input.div_ceil(128);
    let codes_view = view(&code_buffer, s, [output, input], [u64::from(input), 1])?;
    let scales_view = view(
        &scale_buffer,
        s,
        [output.div_ceil(128), scale_cols],
        [u64::from(scale_cols), 1],
    )?;
    let weight = LinearWeightRef::Fp8 {
        layout: Fp8ScaleLayout::Block128,
        codes: codes_view,
        scale: scales_view,
    };
    let batch_x = view(
        &row_buffer,
        s,
        [VERIFY_M4_ROWS as u32, input],
        [u64::from(input), 1],
    )?;
    let mut batch_y = TensorMut(view(
        &batch_out,
        s,
        [VERIFY_M4_ROWS as u32, output],
        [u64::from(output), 1],
    )?);
    blas.linear_typed(ExecutionPhase::Prefill, weight, batch_x, &mut batch_y)?;

    let mut serial = Vec::with_capacity(VERIFY_M4_ROWS * output as usize);
    for row in 0..VERIFY_M4_ROWS {
        let values = rows[row * input as usize..(row + 1) * input as usize].to_vec();
        let x_buffer = s.clone_htod(&values)?;
        let y_buffer = s.clone_htod(&vec![bf16::ZERO; output as usize])?;
        let x = view(&x_buffer, s, [1, input], [u64::from(input), 1])?;
        let mut y = TensorMut(view(&y_buffer, s, [1, output], [u64::from(output), 1])?);
        blas.linear_typed(ExecutionPhase::Decode, weight, x, &mut y)?;
        serial.extend(s.clone_dtoh(&y_buffer)?);
    }
    let batch = s.clone_dtoh(&batch_out)?;
    let diff = verify_bf16("projection", output, usize::MAX, &batch, &serial, |index| {
        format!(
            "row={} col={}",
            index / output as usize,
            index % (output as usize)
        )
    })?;
    Ok((batch.len(), diff))
}

struct VerifyAttentionFixture {
    prefix: u32,
    capacity: u32,
    q_gate: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    q_norm: Vec<bf16>,
    k_norm: Vec<bf16>,
    prefix_k: Vec<bf16>,
    prefix_v: Vec<bf16>,
}

impl VerifyAttentionFixture {
    fn new(prefix: u32, capacity: u32, state: &mut u64) -> Self {
        let qwen = FullAttentionSpec::q38();
        let value = |state: &mut u64| bf16::from_f32(next_f32_unit(state) * 3.0 - 1.5);
        let q_gate = (0..VERIFY_M4_ROWS * qwen.hq as usize * 2 * D as usize)
            .map(|_| value(state))
            .collect::<Vec<_>>();
        let k = (0..VERIFY_M4_ROWS * qwen.hkv as usize * D as usize)
            .map(|_| value(state))
            .collect::<Vec<_>>();
        let v = (0..VERIFY_M4_ROWS * qwen.hkv as usize * D as usize)
            .map(|_| value(state))
            .collect::<Vec<_>>();
        let q_norm = (0..D as usize)
            .map(|_| bf16::from_f32(next_f32_unit(state) * 0.25))
            .collect::<Vec<_>>();
        let k_norm = (0..D as usize)
            .map(|_| bf16::from_f32(next_f32_unit(state) * 0.25))
            .collect::<Vec<_>>();
        let prefix_k = (0..prefix as usize * qwen.hkv as usize * D as usize)
            .map(|_| value(state))
            .collect::<Vec<_>>();
        let prefix_v = (0..prefix as usize * qwen.hkv as usize * D as usize)
            .map(|_| value(state))
            .collect::<Vec<_>>();
        Self {
            prefix,
            capacity,
            q_gate,
            k,
            v,
            q_norm,
            k_norm,
            prefix_k,
            prefix_v,
        }
    }

    fn caches(&self) -> (Vec<bf16>, Vec<bf16>) {
        let qwen = FullAttentionSpec::q38();
        let span = self.capacity as usize * D as usize;
        let nan = bf16::from_bits(0x7fc0);
        let mut k = vec![nan; qwen.hkv as usize * span];
        let mut v = vec![nan; qwen.hkv as usize * span];
        for head in 0..qwen.hkv as usize {
            for position in 0..self.prefix as usize {
                let source = (head * self.prefix as usize + position) * D as usize;
                let target = head * span + position * D as usize;
                k[target..target + D as usize]
                    .copy_from_slice(&self.prefix_k[source..source + D as usize]);
                v[target..target + D as usize]
                    .copy_from_slice(&self.prefix_v[source..source + D as usize]);
            }
        }
        (k, v)
    }
}

struct VerifyRunDump {
    gate: Vec<bf16>,
    q: Vec<bf16>,
    core: Vec<bf16>,
    k_cache: Vec<bf16>,
    v_cache: Vec<bf16>,
    scratch: Vec<u8>,
    workspace: AttentionWorkspace,
    max_parts: u32,
}

fn verify_f32_at(bytes: &[u8], offset: u64) -> Result<f32> {
    let start = usize::try_from(offset).context("scratch offset overflow")?;
    ensure!(
        start.checked_add(4).is_some_and(|end| end <= bytes.len()),
        "scratch read outside workspace"
    );
    Ok(f32::from_le_bytes(bytes[start..start + 4].try_into()?))
}

struct VerifyPartials {
    max: Vec<f32>,
    sum: Vec<f32>,
    out: Vec<f32>,
}

fn verify_run_batch(
    device: &Device,
    common: &CommonOpsRef,
    fixture: &VerifyAttentionFixture,
    rows: u32,
) -> Result<VerifyRunDump> {
    let qwen = FullAttentionSpec::q38();
    let s = &device.stream;
    let limits = AttentionLimits {
        max_rows: rows,
        capacity: fixture.capacity,
        key_chunk: 256,
        causal: 1,
    };
    let request = FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits)?;
    let q_gate = s.clone_htod(&fixture.q_gate)?;
    let k = s.clone_htod(&fixture.k)?;
    let v = s.clone_htod(&fixture.v)?;
    let q = s.alloc_zeros::<bf16>(rows as usize * qwen.hq as usize * D as usize)?;
    let gate = s.alloc_zeros::<bf16>(rows as usize * qwen.hq as usize * D as usize)?;
    let core = s.clone_htod(&vec![
        bf16::from_bits(0x7fc0);
        rows as usize * qwen.hq as usize * D as usize
    ])?;
    let (host_k, host_v) = fixture.caches();
    let k_cache = s.clone_htod(&host_k)?;
    let v_cache = s.clone_htod(&host_v)?;
    let q_norm = s.clone_htod(&fixture.q_norm)?;
    let k_norm = s.clone_htod(&fixture.k_norm)?;
    let mask = s.clone_htod(&verify_causal_mask())?;
    let (metadata, meta_ref) = meta(
        s,
        fixture.prefix as i32,
        rows as i32,
        fixture.capacity as i32,
    )?;
    let scratch = s.alloc_zeros::<u8>(usize::try_from(request.bytes)?)?;
    let (scratch_ptr, _scratch_hold) = scratch.device_ptr(s);
    let bindings = AttentionBindings {
        compact: None,
        q_gate: view(
            &q_gate,
            s,
            [rows, qwen.hq, 2 * D],
            [(qwen.hq * 2 * D).into(), (2 * D).into(), 1],
        )?,
        k: view(
            &k,
            s,
            [rows, qwen.hkv, D],
            [(qwen.hkv * D).into(), D.into(), 1],
        )?,
        v: view(
            &v,
            s,
            [rows, qwen.hkv, D],
            [(qwen.hkv * D).into(), D.into(), 1],
        )?,
        q: TensorMut(view(
            &q,
            s,
            [rows, qwen.hq, D],
            [(qwen.hq * D).into(), D.into(), 1],
        )?),
        gate: Some(TensorMut(view(
            &gate,
            s,
            [rows, qwen.hq, D],
            [(qwen.hq * D).into(), D.into(), 1],
        )?)),
        norm_mode: super::ops::RmsMode::ZeroCentered,
        k_cache: TensorMut(view(
            &k_cache,
            s,
            [qwen.hkv, fixture.capacity, D],
            [u64::from(fixture.capacity) * u64::from(D), D.into(), 1],
        )?),
        v_cache: TensorMut(view(
            &v_cache,
            s,
            [qwen.hkv, fixture.capacity, D],
            [u64::from(fixture.capacity) * u64::from(D), D.into(), 1],
        )?),
        core_out: TensorMut(view(
            &core,
            s,
            [rows, qwen.hq, D],
            [(qwen.hq * D).into(), D.into(), 1],
        )?),
        q_norm: view(&q_norm, s, [D], [1])?,
        k_norm: view(&k_norm, s, [D], [1])?,
        causal_mask: view(&mask, s, [VERIFY_CAUSAL_MASK_LEN as u32], [1])?,
        positions: PositionBindings::Linear,
    };
    let plan = FullAttentionOps::prepare(
        &device.ctx,
        &FullAttentionSpec::q38(),
        limits,
        bindings,
        common.clone(),
    )?;
    let max_parts = plan.max_parts();
    FullAttentionOps::run(
        &plan,
        s,
        meta_ref,
        Scratch {
            ptr: scratch_ptr,
            bytes: request.bytes,
        },
    )?;
    s.synchronize()?;
    drop(metadata);
    Ok(VerifyRunDump {
        gate: s.clone_dtoh(&gate)?,
        q: s.clone_dtoh(&q)?,
        core: s.clone_dtoh(&core)?,
        k_cache: s.clone_dtoh(&k_cache)?,
        v_cache: s.clone_dtoh(&v_cache)?,
        scratch: s.clone_dtoh(&scratch)?,
        workspace: plan.workspace(),
        max_parts,
    })
}

fn verify_causal_mask() -> Vec<u64> {
    (0..VERIFY_CAUSAL_MASK_LEN)
        .map(|row| {
            if row + 1 == VERIFY_CAUSAL_MASK_LEN {
                u64::MAX
            } else {
                (1u64 << (row + 1)) - 1
            }
        })
        .collect()
}

struct VerifySerialDump {
    gate: Vec<bf16>,
    q: Vec<bf16>,
    core: Vec<bf16>,
    k_cache: Vec<bf16>,
    v_cache: Vec<bf16>,
    partials: Vec<VerifyPartials>,
}

fn verify_run_serial(
    device: &Device,
    common: &CommonOpsRef,
    fixture: &VerifyAttentionFixture,
    rows: u32,
) -> Result<VerifySerialDump> {
    let qwen = FullAttentionSpec::q38();
    let s = &device.stream;
    let limits = AttentionLimits {
        max_rows: 1,
        capacity: fixture.capacity,
        key_chunk: 256,
        causal: 1,
    };
    let request = FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits)?;
    let (host_k, host_v) = fixture.caches();
    let k_cache = s.clone_htod(&host_k)?;
    let v_cache = s.clone_htod(&host_v)?;
    let q_norm = s.clone_htod(&fixture.q_norm)?;
    let k_norm = s.clone_htod(&fixture.k_norm)?;
    let mask = s.clone_htod(&verify_causal_mask())?;
    let mut q_gate = s.alloc_zeros::<bf16>(qwen.hq as usize * 2 * D as usize)?;
    let mut k = s.alloc_zeros::<bf16>(qwen.hkv as usize * D as usize)?;
    let mut v = s.alloc_zeros::<bf16>(qwen.hkv as usize * D as usize)?;
    let q = s.alloc_zeros::<bf16>(qwen.hq as usize * D as usize)?;
    let gate = s.alloc_zeros::<bf16>(qwen.hq as usize * D as usize)?;
    let mut core = s.clone_htod(&vec![
        bf16::from_bits(0x7fc0);
        qwen.hq as usize * D as usize
    ])?;
    let scratch = s.alloc_zeros::<u8>(usize::try_from(request.bytes)?)?;
    let (scratch_ptr, _scratch_hold) = scratch.device_ptr(s);
    let mut gates = Vec::new();
    let mut queries = Vec::new();
    let mut outputs = Vec::new();
    let mut partials = Vec::new();

    for row in 0..rows as usize {
        let q_gate_row = fixture.q_gate[row * qwen.hq as usize * 2 * D as usize
            ..(row + 1) * qwen.hq as usize * 2 * D as usize]
            .to_vec();
        let k_row = fixture.k
            [row * qwen.hkv as usize * D as usize..(row + 1) * qwen.hkv as usize * D as usize]
            .to_vec();
        let v_row = fixture.v
            [row * qwen.hkv as usize * D as usize..(row + 1) * qwen.hkv as usize * D as usize]
            .to_vec();
        s.memcpy_htod(&q_gate_row, &mut q_gate.slice_mut(..))?;
        s.memcpy_htod(&k_row, &mut k.slice_mut(..))?;
        s.memcpy_htod(&v_row, &mut v.slice_mut(..))?;
        s.memcpy_htod(
            &vec![bf16::from_bits(0x7fc0); qwen.hq as usize * D as usize],
            &mut core.slice_mut(..),
        )?;
        let prefix = fixture.prefix + row as u32;
        let (metadata, meta_ref) = meta(s, prefix as i32, 1, fixture.capacity as i32)?;
        let bindings = AttentionBindings {
            compact: None,
            q_gate: view(
                &q_gate,
                s,
                [1, qwen.hq, 2 * D],
                [(qwen.hq * 2 * D).into(), (2 * D).into(), 1],
            )?,
            k: view(
                &k,
                s,
                [1, qwen.hkv, D],
                [(qwen.hkv * D).into(), D.into(), 1],
            )?,
            v: view(
                &v,
                s,
                [1, qwen.hkv, D],
                [(qwen.hkv * D).into(), D.into(), 1],
            )?,
            q: TensorMut(view(
                &q,
                s,
                [1, qwen.hq, D],
                [(qwen.hq * D).into(), D.into(), 1],
            )?),
            gate: Some(TensorMut(view(
                &gate,
                s,
                [1, qwen.hq, D],
                [(qwen.hq * D).into(), D.into(), 1],
            )?)),
            norm_mode: super::ops::RmsMode::ZeroCentered,
            k_cache: TensorMut(view(
                &k_cache,
                s,
                [qwen.hkv, fixture.capacity, D],
                [u64::from(fixture.capacity) * u64::from(D), D.into(), 1],
            )?),
            v_cache: TensorMut(view(
                &v_cache,
                s,
                [qwen.hkv, fixture.capacity, D],
                [u64::from(fixture.capacity) * u64::from(D), D.into(), 1],
            )?),
            core_out: TensorMut(view(
                &core,
                s,
                [1, qwen.hq, D],
                [(qwen.hq * D).into(), D.into(), 1],
            )?),
            q_norm: view(&q_norm, s, [D], [1])?,
            k_norm: view(&k_norm, s, [D], [1])?,
            causal_mask: view(&mask, s, [VERIFY_CAUSAL_MASK_LEN as u32], [1])?,
            positions: PositionBindings::Linear,
        };
        let plan = FullAttentionOps::prepare(
            &device.ctx,
            &FullAttentionSpec::q38(),
            limits,
            bindings,
            common.clone(),
        )?;
        FullAttentionOps::run(
            &plan,
            s,
            meta_ref,
            Scratch {
                ptr: scratch_ptr,
                bytes: request.bytes,
            },
        )?;
        s.synchronize()?;
        drop(metadata);
        let run = VerifyRunDump {
            gate: s.clone_dtoh(&gate)?,
            q: s.clone_dtoh(&q)?,
            core: s.clone_dtoh(&core)?,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            scratch: s.clone_dtoh(&scratch)?,
            workspace: plan.workspace(),
            max_parts: plan.max_parts(),
        };
        let row_partials = verify_row_partials(&run, 0, prefix + 1, limits.key_chunk)?;
        gates.extend(run.gate);
        queries.extend(run.q);
        outputs.extend(run.core);
        partials.push(row_partials);
    }

    Ok(VerifySerialDump {
        gate: gates,
        q: queries,
        core: outputs,
        k_cache: s.clone_dtoh(&k_cache)?,
        v_cache: s.clone_dtoh(&v_cache)?,
        partials,
    })
}

fn verify_row_partials(
    run: &VerifyRunDump,
    row: usize,
    key_count: u32,
    key_chunk: u32,
) -> Result<VerifyPartials> {
    let qwen = FullAttentionSpec::q38();
    let active = key_count.div_ceil(key_chunk).max(1);
    let segment_rows = run.workspace.segment_rows as usize;
    let mut max = Vec::with_capacity(active as usize * qwen.hq as usize);
    let mut sum = Vec::with_capacity(active as usize * qwen.hq as usize);
    let mut out = Vec::with_capacity(active as usize * qwen.hq as usize * D as usize);
    for head in 0..qwen.hq as usize {
        let kv = head / (qwen.hq / qwen.hkv) as usize;
        let group = head % (qwen.hq / qwen.hkv) as usize;
        for part in 0..active as usize {
            let index = (kv * run.max_parts as usize + part)
                * ((qwen.hq / qwen.hkv) as usize * segment_rows)
                + group * segment_rows
                + row;
            max.push(verify_f32_at(
                &run.scratch,
                run.workspace.p_max + u64::try_from(index)? * 4,
            )?);
            sum.push(verify_f32_at(
                &run.scratch,
                run.workspace.p_sum + u64::try_from(index)? * 4,
            )?);
            for dimension in 0..D as usize {
                out.push(verify_f32_at(
                    &run.scratch,
                    run.workspace.p_out
                        + u64::try_from(index)? * u64::from(D) * 4
                        + u64::try_from(dimension)? * 4,
                )?);
            }
        }
    }
    ensure!(
        max.iter().chain(&sum).chain(&out).all(|v| v.is_finite()),
        "VERIFY_M4 non-finite partials row={row}"
    );
    Ok(VerifyPartials { max, sum, out })
}

fn verify_partial_sum(
    batch: &VerifyPartials,
    serial: &VerifyPartials,
    case: u32,
    row: usize,
    active: usize,
) -> Result<Option<String>> {
    verify_f32("partial_sum", case, row, &batch.sum, &serial.sum, |index| {
        format!("head={} part={}", index / active, index % active)
    })
}

#[test]
fn verify_partial_sum_rejects_max_substitution() -> Result<()> {
    let batch = VerifyPartials {
        max: vec![2.0],
        sum: vec![1.0],
        out: vec![3.0],
    };
    let serial = VerifyPartials {
        max: vec![1.0],
        sum: vec![2.0],
        out: vec![3.0],
    };
    ensure!(
        verify_partial_sum(&batch, &serial, 256, 0, 1)?.is_some(),
        "distinct partial sums compared equal"
    );
    Ok(())
}

#[test]
#[ignore = "requires coordinator-granted local 4090 window; FF_Q38_GPU_WINDOW=granted"]
fn verify_attention_m4_serial_bits_gpu() -> Result<()> {
    let qwen = FullAttentionSpec::q38();
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    ensure!(
        device.info.arch == 89,
        "VERIFY_M4 requires the authorized sm_89 device"
    );
    let ops = Ops::new(&device.ctx)?;
    let common = CommonOpsRef::new(&device.ctx)?;
    let runtime = std::env::temp_dir().join(format!("ff-verify-attn-{}", std::process::id()));
    std::fs::create_dir_all(&runtime)?;
    let mut first_diff = None;
    let mut defects = 0;
    let projection = (|| -> Result<(usize, usize)> {
        let closure = Rc::new(Closure::new(false, "VERIFY_M4 projections".into()));
        let mut blas = Blas::new(&device, &ops, &runtime, closure)?;
        blas.enable_fp8(&device.ctx)?;
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut words = 0;
        let mut projection_defects = 0;
        for (output, input) in [(12288u32, 5120u32), (1024, 5120), (5120, 6144)] {
            let (count, diff) = verify_projection_m4(&device, &blas, output, input, &mut state)?;
            let status = if diff.is_none() { "PASS" } else { "DEFECT" };
            verify_record(
                &mut first_diff,
                &mut projection_defects,
                &mut words,
                count,
                diff,
            );
            println!("VERIFY_M4_PROJECTION output={output} rows=4 serial=BITWISE {status}");
        }
        Ok((words, projection_defects))
    })();
    std::fs::remove_dir_all(&runtime)?;
    let (projection_words, projection_defects) = projection?;
    defects += projection_defects;

    let cases = [
        (251u32, 1024u32),
        (252, 1024),
        (253, 1024),
        (283, 1024),
        (284, 1024),
        (285, 1024),
        (507, 1024),
        (508, 1024),
        (509, 1024),
        (16379, 16832),
        (16380, 16832),
        (16381, 16832),
    ];
    let mut attention_words = 0;
    let mut partial_words = 0;
    let mut rng = 0xf00d_face_1234_5678u64;
    for rows in [1_u32, 2, 3, 4] {
        for (case_index, &(prefix, capacity)) in cases.iter().enumerate() {
            let defects_before = defects;
            let fixture = VerifyAttentionFixture::new(prefix, capacity, &mut rng);
            let batch = verify_run_batch(&device, &common, &fixture, rows)?;
            let serial = verify_run_serial(&device, &common, &fixture, rows)?;
            let row_span = qwen.hq as usize * D as usize;
            let cache_span = capacity as usize * D as usize;
            for row in 0..rows as usize {
                let base = row * row_span;
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut attention_words,
                    row_span,
                    verify_bf16(
                        "gate",
                        prefix,
                        row,
                        &batch.gate[base..base + row_span],
                        &serial.gate[base..base + row_span],
                        |index| format!("head={} dim={}", index / D as usize, index % D as usize),
                    )?,
                );
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut attention_words,
                    row_span,
                    verify_bf16(
                        "q_rope",
                        prefix,
                        row,
                        &batch.q[base..base + row_span],
                        &serial.q[base..base + row_span],
                        |index| format!("head={} dim={}", index / D as usize, index % D as usize),
                    )?,
                );
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut attention_words,
                    row_span,
                    verify_bf16(
                        "core_out",
                        prefix,
                        row,
                        &batch.core[base..base + row_span],
                        &serial.core[base..base + row_span],
                        |index| format!("head={} dim={}", index / D as usize, index % D as usize),
                    )?,
                );

                for head in 0..qwen.hkv as usize {
                    let cache_base = head * cache_span + (prefix as usize + row) * D as usize;
                    let batch_k = batch.k_cache[cache_base..cache_base + D as usize].to_vec();
                    let serial_k = serial.k_cache[cache_base..cache_base + D as usize].to_vec();
                    verify_record(
                        &mut first_diff,
                        &mut defects,
                        &mut attention_words,
                        D as usize,
                        verify_bf16("k_cache", prefix, row, &batch_k, &serial_k, |index| {
                            format!("head={head} dim={index}")
                        })?,
                    );
                    let batch_v = batch.v_cache[cache_base..cache_base + D as usize].to_vec();
                    let serial_v = serial.v_cache[cache_base..cache_base + D as usize].to_vec();
                    verify_record(
                        &mut first_diff,
                        &mut defects,
                        &mut attention_words,
                        D as usize,
                        verify_bf16("v_cache", prefix, row, &batch_v, &serial_v, |index| {
                            format!("head={head} dim={index}")
                        })?,
                    );
                }

                let key_count = prefix + row as u32 + 1;
                let batch_partials = verify_row_partials(&batch, row, key_count, 256)?;
                let serial_partials = &serial.partials[row];
                let active = key_count.div_ceil(256) as usize;
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut partial_words,
                    batch_partials.max.len(),
                    verify_f32(
                        "partial_max",
                        prefix,
                        row,
                        &batch_partials.max,
                        &serial_partials.max,
                        |index| format!("head={} part={}", index / active, index % active),
                    )?,
                );
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut partial_words,
                    batch_partials.sum.len(),
                    verify_partial_sum(&batch_partials, serial_partials, prefix, row, active)?,
                );
                let out_stride = active * D as usize;
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut partial_words,
                    batch_partials.out.len(),
                    verify_f32(
                        "partial_out",
                        prefix,
                        row,
                        &batch_partials.out,
                        &serial_partials.out,
                        |index| {
                            format!(
                                "head={} part={} dim={}",
                                index / out_stride,
                                index % out_stride / D as usize,
                                index % D as usize
                            )
                        },
                    )?,
                );
            }

            for (stage, batch_cache, serial_cache) in [
                ("k_cache", &batch.k_cache, &serial.k_cache),
                ("v_cache", &batch.v_cache, &serial.v_cache),
            ] {
                let diff = batch_cache
                .iter()
                .zip(serial_cache)
                .enumerate()
                .find_map(|(index, (x, y))| {
                    (x.to_bits() != y.to_bits()).then(|| {
                        format!(
                            "VERIFY_M4 first_diff stage=full_{stage} case={prefix} at={index} batch={:#04x} serial={:#04x}",
                            x.to_bits(),
                            y.to_bits()
                        )
                    })
                });
                verify_record(
                    &mut first_diff,
                    &mut defects,
                    &mut attention_words,
                    batch_cache.len(),
                    diff,
                );
            }
            let status = if defects == defects_before {
                "PASS"
            } else {
                "DEFECT"
            };
            println!(
                "VERIFY_M4_CASE index={case_index} rows={rows} prefix={prefix} key_counts={}..{} capacity={capacity} projection_through_cache=BITWISE {status}",
                prefix + 1,
                prefix + rows
            );
        }
    }

    if defects == 0 {
        println!(
            "VERIFY_M4_KERNEL_PASS cases={} rows=4 projections=3 projection_words={projection_words} attention_words={attention_words} partial_words={partial_words} arch={}",
            cases.len(),
            device.info.arch
        );
        return Ok(());
    }
    println!(
        "VERIFY_M4_KERNEL_DEFECT cases={} rows=4 defects={defects} projections=3 projection_words={projection_words} attention_words={attention_words} partial_words={partial_words} arch={}",
        cases.len(),
        device.info.arch
    );
    anyhow::bail!(
        "{}",
        first_diff.context("VERIFY_M4 defect without a location")?
    )
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
        checkpoint.save(&state, &token, 4, s, None).is_err(),
        "stale host prefix accepted"
    );
    checkpoint.save(&state, &token, 5, s, None)?;
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
    let common = super::cubin::module(&d.ctx, "q38_common")?;
    let check = common.load_function("q38_sequence_check")?;
    let stage = common.load_function("q38_stage_inputs")?;
    let mut flag = s.alloc_zeros::<u32>(8)?;
    let flag_view = view(&flag, s, [8], [1])?;
    let flag_ptr = flag_view.element_ptr(7)?;
    checkpoint.save(&state, &token, 4, s, Some((&check, flag_view)))?;
    ensure!(
        s.clone_dtoh(&flag.slice(7..8))? == [0x8000_0000],
        "device checkpoint accepted a stale prefix"
    );
    s.memset_zeros(&mut flag)?;
    checkpoint.save(&state, &token, 5, s, Some((&check, flag_view)))?;
    ensure!(
        s.clone_dtoh(&flag.slice(7..8))? == [0],
        "valid device checkpoint rejected"
    );
    let mut damaged_controls = 1;
    for field in 0..8 {
        let mut values = [5i32, 1, 6, 32, 1, 0, 0, 0];
        values[field] = match field {
            0 => 4,
            1 => 0,
            2 => 7,
            3 => 31,
            4 => 0,
            _ => 1,
        };
        s.memcpy_htod(
            &values,
            state.sequence.as_mut().context("sequence missing")?,
        )?;
        s.memset_zeros(&mut flag)?;
        unsafe {
            s.launch_builder(&check)
                .arg(&seq.ptr)
                .arg(&5i32)
                .arg(&32i32)
                .arg(&1i32)
                .arg(&flag_ptr)
                .launch(super::ops::grid(1, 1))?;
        }
        ensure!(
            s.clone_dtoh(&flag.slice(7..8))? == [0x8000_0000],
            "device sequence field {field} escaped validation"
        );
        damaged_controls += 1;
    }
    let mut staged = s.clone_htod(&[u32::MAX; 8])?;
    let mut axes = s.clone_htod(&[-9i32; 24])?;
    for rows in 1..=4usize {
        let values: Vec<_> = (0..rows).map(|i| (i * 97 + 13) as u32).collect();
        let mut source_values = vec![u32::MAX];
        source_values.extend_from_slice(&values);
        let source = s.clone_htod(&source_values)?;
        let source_ptr = view(&source, s, [(rows + 1) as u32], [1])?.element_ptr(1)?;
        s.memcpy_htod(&[u32::MAX; 8], &mut staged)?;
        s.memcpy_htod(&[-9i32; 24], &mut axes)?;
        s.memset_zeros(&mut flag)?;
        unsafe {
            s.launch_builder(&stage)
                .arg(&source_ptr)
                .arg(&mut staged)
                .arg(&mut axes)
                .arg(&seq.ptr)
                .arg(&(rows as u32))
                .arg(&5i32)
                .arg(&32i32)
                .arg(&248320u32)
                .arg(&flag_ptr)
                .launch(super::ops::flat(rows))?;
        }
        let ids = s.clone_dtoh(&staged)?;
        ensure!(
            ids[..rows] == values && ids[rows..].iter().all(|&t| t == u32::MAX),
            "staged ID extent/tail mismatch"
        );
        let got = s.clone_dtoh(&axes)?;
        ensure!(
            got[..3 * rows]
                == (0..rows)
                    .flat_map(|i| [6 + i as i32; 3])
                    .collect::<Vec<_>>()
                && got[3 * rows..].iter().all(|&v| v == -9),
            "staged RoPE axes/tail mismatch"
        );
        ensure!(
            s.clone_dtoh(state.sequence.as_ref().context("sequence missing")?)?
                == [5, rows as i32, 5 + rows as i32, 32, 1, 0, 0, 0],
            "staged sequence mismatch"
        );
        ensure!(
            s.clone_dtoh(&flag.slice(7..8))? == [0],
            "valid staged input rejected"
        );
        let bad = s.clone_htod(&[248320u32])?;
        unsafe {
            s.launch_builder(&stage)
                .arg(&bad)
                .arg(&mut staged)
                .arg(&mut axes)
                .arg(&seq.ptr)
                .arg(&1u32)
                .arg(&5i32)
                .arg(&32i32)
                .arg(&248320u32)
                .arg(&flag_ptr)
                .launch(super::ops::flat(1))?;
        }
        ensure!(
            s.clone_dtoh(&flag.slice(7..8))? == [0x4000_0000] && s.clone_dtoh(&staged)?[0] == 0,
            "invalid device ID was not quarantined"
        );
        unsafe {
            s.launch_builder(&stage)
                .arg(&source_ptr)
                .arg(&mut staged)
                .arg(&mut axes)
                .arg(&seq.ptr)
                .arg(&(rows as u32))
                .arg(&5i32)
                .arg(&32i32)
                .arg(&248320u32)
                .arg(&flag_ptr)
                .launch(super::ops::flat(rows))?;
        }
        ensure!(
            s.clone_dtoh(&flag.slice(7..8))? == [0x4000_0000],
            "device input error flag was cleared"
        );
        damaged_controls += 1;
    }
    let count = 3 * spec.hidden as usize;
    let mut values: Vec<_> = (0..count + 16)
        .map(|i| {
            if i < count {
                bf16::from_f32((i % 131) as f32 * 0.03125)
            } else {
                bf16::NAN
            }
        })
        .collect();
    let mut hidden = s.clone_htod(&values)?;
    let ops = Ops::new(&d.ctx)?;
    let hidden_view = view(&hidden, s, [3, spec.hidden], [u64::from(spec.hidden), 1])?;
    s.memset_zeros(&mut flag)?;
    ops.finite(s, hidden_view, flag_view)?;
    ensure!(
        s.clone_dtoh(&flag.slice(7..8))? == [0],
        "finite check read inactive hidden tail"
    );
    values[count - 1] = bf16::NAN;
    s.memcpy_htod(&values, &mut hidden)?;
    ops.finite(s, hidden_view, flag_view)?;
    ensure!(
        s.clone_dtoh(&flag.slice(7..8))? == [1],
        "finite check missed the last active hidden element"
    );
    damaged_controls += 1;
    let short_flag = s.clone_htod(&[0x1357u32, 0])?;
    let short_view = view(&short_flag, s, [2], [1])?;
    values[count - 1] = bf16::from_f32(0.5);
    s.memcpy_htod(&values, &mut hidden)?;
    ops.finite(s, hidden_view, short_view)?;
    ensure!(
        s.clone_dtoh(&short_flag)? == [0x1357, 0],
        "finite check wrote outside bootstrap status word"
    );
    values[count - 1] = bf16::NAN;
    s.memcpy_htod(&values, &mut hidden)?;
    ops.finite(s, hidden_view, short_view)?;
    ensure!(
        s.clone_dtoh(&short_flag)? == [0x1357, 1],
        "finite check missed bootstrap status word"
    );
    damaged_controls += 1;
    eprintln!(
        "Q38_MTP_DEVICE_BOUNDARY_PASS staged_shapes=4 hidden_shape=[3,{}] damaged_controls={damaged_controls} sticky_status=true source_word_offset=1 status_word_offsets=[1,7]",
        spec.hidden
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

#[derive(Clone, Copy, Debug)]
struct NormCase {
    valid_rows: u32,
    row_capacity: u32,
    heads: u32,
    width: u32,
    mode: i32,
    row_pad: u32,
    head_pad: u32,
}

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545f4914f6cdd1d)
}

fn next_f32_unit(state: &mut u64) -> f32 {
    ((next_u64(state) >> 40) as f32) * (1.0 / 16777216.0)
}

fn build_norm_fixture(case: NormCase, order_sensitive: bool) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let hs = (case.width + case.head_pad) as usize;
    let rs = (case.heads as usize) * hs + (case.row_pad as usize);
    let total = (case.row_capacity as usize) * rs;
    let w_len = case.width as usize;
    if !order_sensitive {
        let x = (0..total)
            .map(|i| {
                bf16::from_f32(
                    (((i % 131) as f32) - 65.0) * 0.03125
                        + (((i % 257) as f32) - 128.0) * 0.0078125,
                )
            })
            .collect();
        let z = (0..total)
            .map(|i| bf16::from_f32((((i % 137) as f32) - 68.0) * 0.0234375))
            .collect();
        let w = (0..w_len)
            .map(|d| bf16::from_f32((((d % 139) as f32) - 69.0) * 0.015625))
            .collect();
        return (x, w, z);
    }
    let mut seed = u64::from(case.width).wrapping_mul(0x9e3779b97f4a7c15) ^ 0xd1b54a32d192ed03;
    let sample_spread = |s: &mut u64, idx: usize| -> bf16 {
        let col = idx % w_len;
        if (idx / w_len).is_multiple_of(2) {
            if col == 0 || col == 64 {
                return bf16::from_f32(64.0);
            }
            let jitter = 0.9375f32 + 0.125f32 * next_f32_unit(s);
            return bf16::from_f32(0.022097087f32 * jitter);
        }
        let mut g = -3.0f32;
        for _ in 0..6 {
            g += next_f32_unit(s);
        }
        let exp_u = -8.0f32 + 16.0f32 * next_f32_unit(s);
        bf16::from_f32(g * exp_u.exp2())
    };
    let x = (0..total).map(|i| sample_spread(&mut seed, i)).collect();
    let z = (0..total)
        .map(|i| bf16::from_f32(0.5 + 0.5 * next_f32_unit(&mut seed) + ((i % 7) as f32) * 0.03125))
        .collect();
    let w = (0..w_len)
        .map(|d| bf16::from_f32(0.75 + 0.5 * next_f32_unit(&mut seed) + ((d % 5) as f32) * 0.0625))
        .collect();
    (x, w, z)
}

struct BaselineNormArgs<'a> {
    rows: RowDomain,
    x: Tensor<bf16, 3>,
    wp: u64,
    zp: u64,
    zs: [u64; 3],
    y: &'a mut TensorMut<bf16, 3>,
    mode: i32,
    eps: f32,
}

fn launch_baseline_norm(
    func: &CudaFunction,
    s: &Arc<CudaStream>,
    args: BaselineNormArgs<'_>,
) -> Result<()> {
    let blocks = args.rows.row_capacity * args.x.shape[1];
    unsafe {
        s.launch_builder(func)
            .arg(&args.x.ptr)
            .arg(&args.wp)
            .arg(&args.zp)
            .arg(&args.y.0.ptr)
            .arg(&args.rows.meta.ptr)
            .arg(&args.rows.row_base)
            .arg(&args.rows.row_capacity)
            .arg(&args.x.shape[1])
            .arg(&args.x.shape[2])
            .arg(&args.x.stride[0])
            .arg(&args.x.stride[1])
            .arg(&args.x.stride[2])
            .arg(&args.y.0.stride[0])
            .arg(&args.y.0.stride[1])
            .arg(&args.y.0.stride[2])
            .arg(&args.zs[0])
            .arg(&args.zs[1])
            .arg(&args.zs[2])
            .arg(&args.mode)
            .arg(&args.eps)
            .launch(grid(blocks as usize, 256))?;
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted and FF_Q38_NORM_ORACLE_DIR"]
fn q38_norm_stage_ab_gpu() -> Result<()> {
    let oracle_dir = std::path::PathBuf::from(
        std::env::var_os("FF_Q38_NORM_ORACLE_DIR").context("FF_Q38_NORM_ORACLE_DIR is required")?,
    );
    ensure!(oracle_dir.is_dir(), "norm oracle directory is missing");
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = Device::new(0)?;
    let ops = Ops::new(&d.ctx)?;
    let common = CommonOpsRef::new(&d.ctx)?;
    let s = &d.stream;
    let sm = cubin::capability(&d.ctx)?;
    let baseline = oracle_dir.join(format!("norm-baseline-sm{sm}.cubin"));
    let base_mod = d.ctx.load_module(cudarc::nvrtc::Ptx::from_binary(
        std::fs::read(&baseline)
            .with_context(|| format!("norm baseline missing: {}", baseline.display()))?,
    ))?;
    let base_norm = base_mod.load_function("q38_norm_baseline_4bf2cfc")?;
    let mutant_norm = base_mod.load_function("q38_norm_mutant_tree128")?;
    let eps = 1e-6f32;
    let sentinel = bf16::from_f32(-17.0);
    let cases = [
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 1,
            width: 5120,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 1,
            width: 5120,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 16,
            row_capacity: 16,
            heads: 1,
            width: 5120,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 17,
            row_capacity: 19,
            heads: 1,
            width: 5120,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2038,
            row_capacity: 2048,
            heads: 1,
            width: 5120,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 1,
            width: 5120,
            mode: 4,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 1,
            width: 5120,
            mode: 4,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 16,
            row_capacity: 16,
            heads: 1,
            width: 5120,
            mode: 4,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 17,
            row_capacity: 19,
            heads: 1,
            width: 5120,
            mode: 4,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2038,
            row_capacity: 2048,
            heads: 1,
            width: 5120,
            mode: 4,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 16,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 16,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 17,
            row_capacity: 19,
            heads: 16,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2038,
            row_capacity: 2048,
            heads: 16,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 24,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 24,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2038,
            row_capacity: 2048,
            heads: 24,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 4,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 4,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 17,
            row_capacity: 19,
            heads: 4,
            width: 256,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 48,
            width: 128,
            mode: 3,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 48,
            width: 128,
            mode: 3,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 17,
            row_capacity: 19,
            heads: 48,
            width: 128,
            mode: 3,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2038,
            row_capacity: 2048,
            heads: 48,
            width: 128,
            mode: 3,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 1,
            row_capacity: 1,
            heads: 16,
            width: 128,
            mode: 2,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 3,
            row_capacity: 4,
            heads: 16,
            width: 128,
            mode: 2,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 17,
            row_capacity: 19,
            heads: 16,
            width: 128,
            mode: 2,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2038,
            row_capacity: 2048,
            heads: 16,
            width: 128,
            mode: 2,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2,
            row_capacity: 3,
            heads: 3,
            width: 128,
            mode: 0,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2,
            row_capacity: 3,
            heads: 2,
            width: 130,
            mode: 1,
            row_pad: 0,
            head_pad: 0,
        },
        NormCase {
            valid_rows: 2,
            row_capacity: 3,
            heads: 2,
            width: 128,
            mode: 1,
            row_pad: 2,
            head_pad: 2,
        },
    ];
    let mut cases = cases.to_vec();
    for (heads, width, mode) in [
        (1, 5120, 1),
        (1, 5120, 4),
        (16, 256, 1),
        (24, 256, 1),
        (4, 256, 1),
        (48, 128, 3),
        (16, 128, 2),
    ] {
        for rows in [4, 2048] {
            cases.push(NormCase {
                valid_rows: rows,
                row_capacity: rows,
                heads,
                width,
                mode,
                row_pad: 0,
                head_pad: 0,
            });
        }
    }
    let mut executed_cases = 0;
    let mut defects = 0;
    for order_sensitive in [false, true] {
        for mut case in cases.iter().copied() {
            if case.mode == 2 {
                case.row_pad = 8192;
            }
            let variants: &[(usize, bool)] = if case.mode == 2 {
                &[(0, false), (2048, false)]
            } else if case.mode == 3 || (case.mode == 1 && case.heads == 24) {
                &[(0, false), (0, true)]
            } else {
                &[(0, false)]
            };
            for &(input_offset, in_place) in variants {
                for graph in [false, true] {
                    let (mut x_host, w_host, mut z_host) =
                        build_norm_fixture(case, order_sensitive);
                    let head_stride = (case.width + case.head_pad) as usize;
                    let row_stride = case.heads as usize * head_stride + case.row_pad as usize;
                    for index in 0..x_host.len() {
                        let row = index / row_stride;
                        let offset = (index % row_stride).checked_sub(input_offset);
                        let active = row < case.valid_rows as usize
                            && offset.is_some_and(|v| {
                                v < case.heads as usize * head_stride
                                    && v % head_stride < case.width as usize
                            });
                        if !active {
                            x_host[index] = bf16::NAN;
                            z_host[index] = bf16::NAN;
                        }
                    }
                    let hs = u64::from(case.width + case.head_pad);
                    let rs = u64::from(case.heads) * hs + u64::from(case.row_pad);
                    let shape = [case.row_capacity, case.heads, case.width];
                    let stride = [rs, hs, 1];
                    let output_stride = if case.mode == 2 {
                        [u64::from(case.heads * case.width), u64::from(case.width), 1]
                    } else {
                        stride
                    };
                    let output_len = (u64::from(case.row_capacity) * output_stride[0]) as usize;
                    let (_meta_buf, mref) =
                        meta(s, 0, case.valid_rows as i32, case.row_capacity as i32)?;
                    let rows = RowDomain {
                        meta: mref,
                        row_base: 0,
                        row_capacity: case.row_capacity,
                    };
                    let mut xb_base = s.clone_htod(&x_host)?;
                    let xb_cand = s.clone_htod(&x_host)?;
                    let wb = s.clone_htod(&w_host)?;
                    let zb = s.clone_htod(&z_host)?;
                    let yb_base = s.clone_htod(&vec![sentinel; output_len])?;
                    let yb_cand = s.clone_htod(&vec![sentinel; output_len])?;
                    let wv = view(&wb, s, [case.width], [1])?;
                    let mut zv = view(&zb, s, shape, stride)?;
                    zv.ptr += (input_offset * 2) as u64;
                    zv.span_bytes -= (input_offset * 2) as u64;
                    zv.validate()?;
                    if case.mode == 4 {
                        let active_elems = (case.valid_rows * case.heads * case.width) as i32;
                        unsafe {
                            s.launch_builder(&ops.residual)
                                .arg(&mut xb_base)
                                .arg(&zb)
                                .arg(&active_elems)
                                .launch(super::ops::flat(active_elems as usize))?;
                        }
                    }
                    let mut xv_base = view(&xb_base, s, shape, stride)?;
                    let mut xv_cand = view(&xb_cand, s, shape, stride)?;
                    for x in [&mut xv_base, &mut xv_cand] {
                        x.ptr += (input_offset * 2) as u64;
                        x.span_bytes -= (input_offset * 2) as u64;
                        x.validate()?;
                    }
                    let mut yv_base = TensorMut(if in_place {
                        xv_base
                    } else {
                        view(&yb_base, s, shape, output_stride)?
                    });
                    launch_baseline_norm(
                        &base_norm,
                        s,
                        BaselineNormArgs {
                            rows,
                            x: xv_base,
                            wp: if case.mode == 2 { 0 } else { wv.ptr },
                            zp: if case.mode == 3 { zv.ptr } else { 0 },
                            zs: if case.mode == 3 { zv.stride } else { [0; 3] },
                            y: &mut yv_base,
                            mode: if case.mode == 4 { 1 } else { case.mode },
                            eps,
                        },
                    )?;
                    let mut yv_cand = TensorMut(if in_place {
                        xv_cand
                    } else {
                        view(&yb_cand, s, shape, output_stride)?
                    });
                    if graph {
                        s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    }
                    match case.mode {
                        0 => common.norm.rms_with(
                            s,
                            rows,
                            xv_cand,
                            wv,
                            &mut yv_cand,
                            RmsOptions {
                                mode: RmsMode::Standard,
                                eps,
                            },
                        )?,
                        1 => common.norm.rms_with(
                            s,
                            rows,
                            xv_cand,
                            wv,
                            &mut yv_cand,
                            RmsOptions {
                                mode: RmsMode::ZeroCentered,
                                eps,
                            },
                        )?,
                        2 => common.norm.l2_heads(s, rows, xv_cand, &mut yv_cand, eps)?,
                        3 => common.norm.rms_gated(
                            s,
                            rows,
                            GatedRmsInput {
                                x: xv_cand,
                                weight: wv,
                                gate: zv,
                            },
                            &mut yv_cand,
                            eps,
                        )?,
                        4 => {
                            let mut xv_cand = TensorMut(xv_cand);
                            common.norm.rms_add_with(
                                s,
                                rows,
                                ResidualRmsInput {
                                    x: &mut xv_cand,
                                    residual: zv,
                                    weight: wv,
                                },
                                &mut yv_cand,
                                RmsOptions {
                                    mode: RmsMode::ZeroCentered,
                                    eps,
                                },
                            )?;
                        }
                        _ => unreachable!(),
                    }
                    if graph {
                        s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                            .context("norm capture returned no graph")?
                            .launch()?;
                    }
                    let got_y_base = s.clone_dtoh(if in_place { &xb_base } else { &yb_base })?;
                    let got_y_cand = s.clone_dtoh(if in_place { &xb_cand } else { &yb_cand })?;
                    for row in 0..case.valid_rows {
                        for head in 0..case.heads {
                            for col in 0..case.width {
                                let index = (u64::from(row) * output_stride[0]
                                    + u64::from(head) * output_stride[1]
                                    + u64::from(col))
                                    as usize;
                                ensure!(
                                    got_y_base[index].is_finite() && got_y_cand[index].is_finite(),
                                    "non-finite norm output on {case:?} index={index}"
                                );
                            }
                        }
                    }
                    if !got_y_base
                        .iter()
                        .zip(&got_y_cand)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                    {
                        defects += 1;
                        eprintln!(
                            "Q38_NORM_DEFECT y {case:?} order_sensitive={order_sensitive} graph={graph} input_offset={input_offset} in_place={in_place}"
                        );
                    }
                    executed_cases += 1;
                    if case.mode == 4 {
                        let got_x_base = s.clone_dtoh(&xb_base)?;
                        let got_x_cand = s.clone_dtoh(&xb_cand)?;
                        if !got_x_base
                            .iter()
                            .zip(&got_x_cand)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                        {
                            defects += 1;
                            eprintln!(
                                "Q38_NORM_DEFECT x {case:?} order_sensitive={order_sensitive} graph={graph} input_offset={input_offset} in_place={in_place}"
                            );
                        }
                    }
                }
            }
        }
    }
    let witness_raw = std::fs::read(oracle_dir.join("norm-order-sensitive.bf16"))?;
    ensure!(
        witness_raw.len() == 3 * 64 * 128 * 2,
        "norm witness extent mismatch"
    );
    let witnesses: Vec<_> = witness_raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
        .collect();
    ensure!(
        witnesses.iter().all(|v| v.is_finite() && *v != bf16::ZERO),
        "norm witness values invalid"
    );
    for (group, (heads, width, mode)) in [(48u32, 128u32, 3i32), (16, 256, 1), (1, 5120, 1)]
        .into_iter()
        .enumerate()
    {
        let case = NormCase {
            valid_rows: 64,
            row_capacity: 64,
            heads,
            width,
            mode,
            row_pad: 0,
            head_pad: 0,
        };
        let mut x_host = vec![bf16::ZERO; (case.row_capacity * heads * width) as usize];
        for slice in 0..(case.row_capacity * heads) as usize {
            let source = (group * 64 + slice % 64) * 128;
            let target = slice * width as usize;
            x_host[target..target + 128].copy_from_slice(&witnesses[source..source + 128]);
        }
        let w_host = vec![bf16::ONE; width as usize];
        let z_host = vec![bf16::from_f32(32.0); x_host.len()];
        let shape = [case.row_capacity, case.heads, case.width];
        let stride = [u64::from(case.heads * case.width), u64::from(case.width), 1];
        let (_meta_buf, mref) = meta(s, 0, case.valid_rows as i32, case.row_capacity as i32)?;
        let rows = RowDomain {
            meta: mref,
            row_base: 0,
            row_capacity: case.row_capacity,
        };
        let xb = s.clone_htod(&x_host)?;
        let wb = s.clone_htod(&w_host)?;
        let zb = s.clone_htod(&z_host)?;
        let yb_base = s.clone_htod(&vec![sentinel; x_host.len()])?;
        let yb_mut = s.clone_htod(&vec![sentinel; x_host.len()])?;
        let yb_cand = s.clone_htod(&vec![sentinel; x_host.len()])?;
        let xv = view(&xb, s, shape, stride)?;
        let wv = view(&wb, s, [case.width], [1])?;
        let zv = view(&zb, s, shape, stride)?;
        let mut yv_base = TensorMut(view(&yb_base, s, shape, stride)?);
        let mut yv_mut = TensorMut(view(&yb_mut, s, shape, stride)?);
        let mut yv_cand = TensorMut(view(&yb_cand, s, shape, stride)?);
        launch_baseline_norm(
            &base_norm,
            s,
            BaselineNormArgs {
                rows,
                x: xv,
                wp: wv.ptr,
                zp: if mode == 3 { zv.ptr } else { 0 },
                zs: if mode == 3 { zv.stride } else { [0; 3] },
                y: &mut yv_base,
                mode,
                eps,
            },
        )?;
        launch_baseline_norm(
            &mutant_norm,
            s,
            BaselineNormArgs {
                rows,
                x: xv,
                wp: wv.ptr,
                zp: if mode == 3 { zv.ptr } else { 0 },
                zs: if mode == 3 { zv.stride } else { [0; 3] },
                y: &mut yv_mut,
                mode,
                eps,
            },
        )?;
        if mode == 3 {
            common.norm.rms_gated(
                s,
                rows,
                GatedRmsInput {
                    x: xv,
                    weight: wv,
                    gate: zv,
                },
                &mut yv_cand,
                eps,
            )?;
        } else {
            common.norm.rms_with(
                s,
                rows,
                xv,
                wv,
                &mut yv_cand,
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps,
                },
            )?;
        }
        let got_base = s.clone_dtoh(&yb_base)?;
        let got_cand = s.clone_dtoh(&yb_cand)?;
        ensure!(
            got_base.iter().chain(&got_cand).all(|v| v.is_finite()),
            "non-finite norm witness output width={width}"
        );
        if !got_base
            .iter()
            .zip(&got_cand)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            defects += 1;
            eprintln!("Q38_NORM_DEFECT witness width={width}");
        }
        let got_mut = s.clone_dtoh(&yb_mut)?;
        let mismatches = got_base
            .iter()
            .zip(&got_mut)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            mismatches > 0,
            "planted tree128 mutant was not detected for width={width}"
        );
        eprintln!(
            "Q38_NORM_MUTANT_CHECK width={width} heads={heads} mode={mode} mismatches={mismatches} PASS"
        );
    }
    eprintln!(
        "Q38_NORM_STAGE_AB cases={} gpu_vs_gpu={} graph_eager=EXECUTED inactive_NaN=CHECKED mutant_detected=PASS defects={defects}",
        executed_cases,
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "norm candidate has {defects} defects");
    Ok(())
}

fn invariant_gated_norm(device: &Device) -> Result<()> {
    let rows = crate::linear_plan::MAX_SKINNY_ROWS;
    let (heads, width) = (32usize, 128usize);
    let row_words = heads * width;
    let s = &device.stream;
    let common = CommonOpsRef::new(&device.ctx)?;
    let x: Vec<_> = (0..rows * row_words)
        .map(|i| {
            bf16::from_f32(((i * 17 + i / row_words * 71) % 509) as f32 / 128.0 - 254.0 / 128.0)
        })
        .collect();
    let z: Vec<_> = (0..rows * row_words)
        .map(|i| {
            bf16::from_f32(((i * 13 + i / row_words * 37) % 257) as f32 / 128.0 - 128.0 / 128.0)
        })
        .collect();
    let weight: Vec<_> = (0..width)
        .map(|i| bf16::from_f32((i % 127) as f32 / 128.0 + 0.5))
        .collect();
    let xb = s.clone_htod(&x)?;
    let zb = s.clone_htod(&z)?;
    let wb = s.clone_htod(&weight)?;
    let expected = s.alloc_zeros::<bf16>(rows * row_words)?;
    let output = s.clone_htod(&vec![bf16::from_bits(0x7fc1); (rows + 1) * row_words])?;
    let tensor = |buffer: &CudaSlice<bf16>, offset, n| {
        super::contracts::view(
            buffer,
            s,
            offset,
            [n, heads as u32, width as u32],
            [row_words as u64, width as u64, 1],
        )
    };
    let weight = view(&wb, s, [width as u32], [1])?;
    let (_one, one_meta) = meta(s, 0, 1, 32)?;
    let one = RowDomain {
        meta: one_meta,
        row_base: 0,
        row_capacity: 1,
    };
    device.finish_upload()?;
    for r in 0..rows {
        common.norm.rms_gated(
            s,
            one,
            GatedRmsInput {
                x: tensor(&xb, r * row_words, 1)?,
                weight,
                gate: tensor(&zb, r * row_words, 1)?,
            },
            &mut TensorMut(tensor(&expected, r * row_words, 1)?),
            1e-6,
        )?;
    }
    let expected = s.clone_dtoh(&expected)?;
    let mut defects = 0;
    for m in 1..=rows as u32 {
        let (_metadata, seq) = meta(s, 0, m as i32, 32)?;
        let domain = RowDomain {
            meta: seq,
            row_base: 0,
            row_capacity: m,
        };
        let x = tensor(&xb, 0, m)?;
        let gate = tensor(&zb, 0, m)?;
        let mut y = TensorMut(tensor(&output, 0, m)?);
        let mut execute = || {
            common
                .norm
                .rms_gated(s, domain, GatedRmsInput { x, weight, gate }, &mut y, 1e-6)
        };
        for graph in [false, true] {
            if graph {
                s.synchronize()?;
                s.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                let launched = execute();
                let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                launched?;
                captured?
                    .context("empty invariant gated norm graph")?
                    .launch()?;
            } else {
                execute()?;
            }
            let got = s.clone_dtoh(&output)?;
            let count = m as usize * row_words;
            ensure!(
                got[..count].iter().all(|v| v.is_finite())
                    && got[count..].iter().all(|v| v.to_bits() == 0x7fc1),
                "invariant gated norm finite/tail"
            );
            let mismatches = got[..count]
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            defects += usize::from(mismatches != 0);
            println!(
                "{} INVARIANT_GATED_NORM x=[{m},{heads},{width}] graph={graph} bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
        }
    }
    let mut wrong = z.clone();
    for r in 1..rows {
        wrong[r * row_words..(r + 1) * row_words].copy_from_slice(&z[..row_words]);
    }
    let wrong = s.clone_htod(&wrong)?;
    let (_metadata, seq) = meta(s, 0, rows as i32, 32)?;
    common.norm.rms_gated(
        s,
        RowDomain {
            meta: seq,
            row_base: 0,
            row_capacity: rows as u32,
        },
        GatedRmsInput {
            x: tensor(&xb, 0, rows as u32)?,
            weight,
            gate: tensor(&wrong, 0, rows as u32)?,
        },
        &mut TensorMut(tensor(&output, 0, rows as u32)?),
        1e-6,
    )?;
    let mismatches = s
        .clone_dtoh(&output)?
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    ensure!(mismatches > 0, "gated norm row0-reuse control escaped");
    println!("PASS INVARIANT_GATED_NORM_CONTROL row0_reuse=true bit_mismatches={mismatches}");
    println!("INVARIANT_GATED_NORM_SUMMARY rows=1..16 eager_graph=true defects={defects}");
    ensure!(defects == 0, "invariant gated norm DEFECT cells={defects}");
    Ok(())
}

#[test]
#[ignore = "requires GPU window and FF_Q38_GPU_WINDOW=granted"]
fn residual_rms_capture_gpu() -> Result<()> {
    use super::ops::ResidualRmsCapture;
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let device = Device::new(0)?;
    invariant_gated_norm(&device)?;
    let stream = &device.stream;
    let ops = Ops::new(&device.ctx)?;
    let common = CommonOpsRef::new(&device.ctx)?;
    let sentinel = bf16::from_f32(-17.0);
    let mut cells = 0;
    let mut controls = 0;
    let mut defects = 0;
    for hidden in [2048usize, 5120] {
        for rows in (1usize..=17).chain([256, 4096]) {
            for mode in [RmsMode::Standard, RmsMode::ZeroCentered] {
                let total = rows + 2;
                let active = if rows == 1 { 1 } else { rows - 1 };
                let (_metadata, sequence) =
                    meta(stream, 7, (active + 1) as i32, (total + 8) as i32)?;
                let domain = RowDomain {
                    meta: sequence,
                    row_base: 1,
                    row_capacity: rows as u32,
                };
                let pattern = |i: usize, layer: usize| {
                    bf16::from_f32((((i + layer * 17) % 257) as f32 - 128.0) / 128.0)
                };
                let addend_host: Vec<_> = (0..total * hidden).map(|i| pattern(i, 3)).collect();
                let residual_host: Vec<_> = (0..total * hidden)
                    .map(|i| bf16::from_f32(((i % 263) as f32 - 131.0) / 256.0))
                    .collect();
                let weight_host: Vec<_> = (0..hidden)
                    .map(|i| bf16::from_f32((i % 251) as f32 / 256.0 + 0.5))
                    .collect();
                let addend = stream.clone_htod(&addend_host)?;
                let weights = stream.clone_htod(&weight_host)?;
                let mut residual = stream.clone_htod(&residual_host)?;
                let baseline = stream.clone_htod(&residual_host)?;
                let mut normalized = stream.clone_htod(&vec![sentinel; total * hidden])?;
                let baseline_normalized = stream.clone_htod(&vec![sentinel; total * hidden])?;
                let mut capture = stream.clone_htod(&vec![sentinel; total * 5 * hidden])?;
                let shape = [total as u32, 1, hidden as u32];
                let strides = [hidden as u64, hidden as u64, 1];
                let addend_view = view(&addend, stream, shape, strides)?;
                let weight_view = view(&weights, stream, [hidden as u32], [1])?;
                let mut residual_view = TensorMut(view(&residual, stream, shape, strides)?);
                let mut normalized_view = TensorMut(view(&normalized, stream, shape, strides)?);
                let mut capture_view = TensorMut(view(
                    &capture,
                    stream,
                    [total as u32, 5, hidden as u32],
                    [(5 * hidden) as u64, hidden as u64, 1],
                )?);
                let baseline_view = TensorMut(view(&baseline, stream, shape, strides)?);
                let mut baseline_output =
                    TensorMut(view(&baseline_normalized, stream, shape, strides)?);
                let options = RmsOptions { mode, eps: 1e-6 };
                let (_one_metadata, one_sequence) = meta(stream, 0, 1, 32)?;
                let one_domain = RowDomain {
                    meta: one_sequence,
                    row_base: 0,
                    row_capacity: 1,
                };
                match mode {
                    RmsMode::ZeroCentered => {
                        let offset = (hidden * std::mem::size_of::<bf16>()) as u64;
                        let rp = baseline_view.0.ptr + offset;
                        let xp = addend_view.ptr + offset;
                        unsafe {
                            stream
                                .launch_builder(&ops.residual)
                                .arg(&rp)
                                .arg(&xp)
                                .arg(&((rows * hidden) as i32))
                                .launch(super::ops::flat(rows * hidden))?;
                        }
                        for row in 1..=active {
                            let offset = (row * hidden * size_of::<bf16>()) as u64;
                            // SAFETY: each retained row is inside the original dense allocation.
                            let (input, mut output) = unsafe {
                                (
                                    Tensor::new(
                                        baseline_view.0.ptr + offset,
                                        [1, 1, hidden as u32],
                                        strides,
                                        (hidden * 2) as u64,
                                    )?,
                                    TensorMut(Tensor::new(
                                        baseline_output.0.ptr + offset,
                                        [1, 1, hidden as u32],
                                        strides,
                                        (hidden * 2) as u64,
                                    )?),
                                )
                            };
                            common.norm.rms_with(
                                stream,
                                one_domain,
                                input,
                                weight_view,
                                &mut output,
                                options,
                            )?;
                        }
                    }
                    RmsMode::Standard => {
                        for row in 1..=active {
                            let offset = (row * hidden * size_of::<bf16>()) as u64;
                            let rp = baseline_view.0.ptr + offset;
                            let xp = addend_view.ptr + offset;
                            let yp = baseline_output.0.ptr + offset;
                            // SAFETY: pointers cover this one row in the preallocated baseline buffers.
                            unsafe {
                                stream
                                    .launch_builder(&ops.add_norm)
                                    .arg(&xp)
                                    .arg(&rp)
                                    .arg(&weight_view.ptr)
                                    .arg(&yp)
                                    .arg(&rp)
                                    .arg(&1i32)
                                    .arg(&(hidden as i32))
                                    .arg(&options.eps)
                                    .launch(grid(1, 256))?;
                            }
                        }
                    }
                }
                let mut launch = || {
                    ops.add_rmsnorm_capture(
                        stream,
                        Some(domain),
                        ResidualRmsCapture {
                            input: ResidualRmsInput {
                                x: &mut residual_view,
                                residual: addend_view,
                                weight: weight_view,
                            },
                            capture: &mut capture_view,
                            slot: 2,
                        },
                        &mut normalized_view,
                        options,
                    )
                };
                launch()?;
                stream.synchronize()?;
                let expected_residual = stream.clone_dtoh(&baseline)?;
                let expected_normalized = stream.clone_dtoh(&baseline_normalized)?;
                let actual_residual = stream.clone_dtoh(&residual)?;
                let actual_normalized = stream.clone_dtoh(&normalized)?;
                let actual_capture = stream.clone_dtoh(&capture)?;
                let unequal = |a: &[bf16], b: &[bf16]| {
                    a.iter()
                        .zip(b)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                };
                let check = |name: &str, actual: &[bf16], expected: &[bf16]| -> Result<usize> {
                    ensure!(
                        actual.len() == expected.len(),
                        "capture comparison shape {name}"
                    );
                    ensure!(
                        actual
                            .iter()
                            .chain(expected)
                            .all(|x| x.to_f32().is_finite()),
                        "non-finite capture comparison {name} hidden={hidden} rows={rows} mode={mode:?}"
                    );
                    let mismatch = unequal(actual, expected);
                    if mismatch != 0 {
                        println!(
                            "DEFECT CAPTURE_OP hidden={hidden} rows={rows} mode={mode:?} value={name} bit_mismatches={mismatch} bound=0"
                        );
                    }
                    Ok(usize::from(mismatch != 0))
                };
                let cell_before = defects;
                if matches!(mode, RmsMode::ZeroCentered) {
                    common.norm.rms_with(
                        stream,
                        domain,
                        baseline_view.0,
                        weight_view,
                        &mut baseline_output,
                        options,
                    )?;
                    defects += check(
                        "batch_rms_vs_M1",
                        &stream.clone_dtoh(&baseline_normalized)?,
                        &expected_normalized,
                    )?;
                }
                defects += check("residual", &actual_residual, &expected_residual)?;
                defects += check("normalization", &actual_normalized, &expected_normalized)?;
                let mut expected_capture = vec![sentinel; total * 5 * hidden];
                for row in 1..=active {
                    let offset = row * 5 * hidden + 2 * hidden;
                    expected_capture[offset..offset + hidden]
                        .copy_from_slice(&expected_residual[row * hidden..(row + 1) * hidden]);
                }
                defects += check("capture_layout", &actual_capture, &expected_capture)?;
                stream.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                let captured_launch = launch();
                let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?;
                captured_launch?;
                let graph = graph.context("capture graph missing")?;
                for _ in 0..2 {
                    stream.memcpy_htod(&residual_host, &mut residual)?;
                    stream.memcpy_htod(&vec![sentinel; total * hidden], &mut normalized)?;
                    stream.memcpy_htod(&vec![sentinel; total * 5 * hidden], &mut capture)?;
                    graph.launch()?;
                    stream.synchronize()?;
                    defects += check(
                        "graph_residual",
                        &stream.clone_dtoh(&residual)?,
                        &expected_residual,
                    )?;
                    defects += check(
                        "graph_normalization",
                        &stream.clone_dtoh(&normalized)?,
                        &expected_normalized,
                    )?;
                    defects += check(
                        "graph_capture",
                        &stream.clone_dtoh(&capture)?,
                        &expected_capture,
                    )?;
                }
                if rows == 17 {
                    let wrong: Vec<_> = (0..total * hidden).map(|i| pattern(i, 4)).collect();
                    let wrong = stream.clone_htod(&wrong)?;
                    stream.memcpy_htod(&residual_host, &mut residual)?;
                    let wrong_view = view(&wrong, stream, shape, strides)?;
                    ops.add_rmsnorm_capture(
                        stream,
                        Some(domain),
                        ResidualRmsCapture {
                            input: ResidualRmsInput {
                                x: &mut residual_view,
                                residual: wrong_view,
                                weight: weight_view,
                            },
                            capture: &mut capture_view,
                            slot: 2,
                        },
                        &mut normalized_view,
                        options,
                    )?;
                    stream.synchronize()?;
                    let wrong_capture = stream.clone_dtoh(&capture)?;
                    ensure!(
                        wrong_capture.iter().all(|x| x.to_f32().is_finite()),
                        "non-finite wrong-layer capture control"
                    );
                    let mismatch = unequal(&wrong_capture, &expected_capture);
                    defects += usize::from(mismatch == 0);
                    println!(
                        "CAPTURE_WRONG_LAYER hidden={hidden} rows={rows} mode={mode:?} bit_mismatches={mismatch} {}",
                        if mismatch > 0 { "PASS" } else { "DEFECT" }
                    );
                    controls += 1;
                }
                println!(
                    "CAPTURE_OP hidden={hidden} rows={rows} active={active} taps=5 mode={mode:?} residual/norm/capture/graph_bits {}",
                    if defects == cell_before {
                        "PASS"
                    } else {
                        "DEFECT"
                    }
                );
                cells += 1;
            }
        }
    }
    println!(
        "RESIDUAL_RMS_CAPTURE cells={cells} wrong_layer_controls={controls} graph_replays={} defects={defects} {}",
        cells * 2,
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "capture candidate has {defects} defects");
    Ok(())
}
