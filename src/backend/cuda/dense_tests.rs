use super::*;
use crate::{
    backend::setup::{AttentionPlan, MultiImpl, MultiPlan, MultiShape},
    config::Config,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

#[derive(Default)]
struct Gate {
    cells: usize,
    defects: usize,
}
impl Gate {
    fn check(&mut self, name: &str, pass: bool) {
        self.cells += 1;
        if !pass {
            self.defects += 1;
        }
        println!(
            "DENSE_HIP_CELL name={name} result={}",
            if pass { "PASS" } else { "DEFECT" }
        );
    }
    fn compare(
        &mut self,
        name: &str,
        actual: &[bf16],
        expected: &[bf16],
        roundings: u32,
        reduction: usize,
    ) -> Result<bool> {
        ensure!(
            actual.len() == expected.len() && !actual.is_empty(),
            "RUNNER_FAIL {name}: reference extent"
        );
        let error = actual.iter().zip(expected).try_fold(0f64, |max, (a, b)| {
            let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
            ensure!(
                a.is_finite() && b.is_finite(),
                "RUNNER_FAIL {name}: nonfinite value"
            );
            Ok::<_, anyhow::Error>(max.max((a - b).abs() / (1.0 + b.abs())))
        })?;
        let nu = reduction as f64 * 2f64.powi(-24);
        let bound = 2.0 * (f64::from(roundings) * 2f64.powi(-8) + nu / (1.0 - nu)) + 1e-5;
        println!(
            "DENSE_HIP_ERROR name={name} normalized_max={error:e} bound={bound:e} roundings={roundings} reduction={reduction}"
        );
        let root = std::path::PathBuf::from(
            std::env::var_os("FF_DENSE_EVIDENCE").context("FF_DENSE_EVIDENCE required")?,
        );
        let bytes: Vec<_> = actual
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        std::fs::write(root.join(format!("{name}.bf16")), bytes)?;
        self.check(name, error <= bound);
        Ok(error <= bound)
    }
}

fn pattern(n: usize, seed: usize) -> Vec<bf16> {
    (0..n)
        .map(|i| {
            bf16::from_f32(
                (((i * 17 + seed * 31) % 257) as f32 - 128.0) / 256.0
                    + (((i * 29 + seed * 13) % 263) as f32 - 131.0) / 512.0,
            )
        })
        .collect()
}
fn config() -> Result<Config> {
    Ok(serde_json::from_value(serde_json::json!({
        "model_type":"llama", "hidden_size":2048,"intermediate_size":6144,
        "num_hidden_layers":42,"num_attention_heads":16,"num_key_value_heads":2,
        "head_dim":128,"vocab_size":130560,"max_position_embeddings":131072,
        "rms_norm_eps":1e-6,"rope_theta":5000000.0,"hidden_act":"silu",
        "tie_word_embeddings":false,"torch_dtype":"bfloat16","eos_token_id":[1,130073]
    }))?)
}

#[test]
#[ignore = "requires GPU lease and FF_DENSE_GPU_WINDOW=granted"]
fn dense_runtime_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let ctx = CudaContext::new(0)?;
    let s = ctx.new_stream()?;
    let ops = ops::Ops::new(&ctx)?;
    let mut gate = Gate::default();
    let c = config()?;
    for stride in [2048, 2560, 6144, 12288, 130560, 129 * 128, 4096 * 128] {
        for prime in [257, 263] {
            ensure!(
                stride % prime != 0,
                "RUNNER_FAIL fixture period divides stride"
            );
        }
    }
    println!(
        "DENSE_HIP_GEOMETRY hidden=[rows,2048] qkv=[rows,2560] kv=[2,capacity,128] periods=257,263 model_layers=42"
    );
    copy_gate(&ctx, &s, &mut gate)?;
    norm_gate(&ctx, &s, &ops, &mut gate)?;
    rope_gate(&s, &ops, &mut gate)?;
    activation_sample(&s, &ops, &mut gate)?;
    attention_gate(&ctx, &s, &c, &mut gate)?;
    conversion_gate(&ctx, &s, &mut gate)?;
    #[cfg(feature = "rocm")]
    blas_gate(&s, &mut gate)?;
    s.synchronize()?;
    println!(
        "DENSE_RUNTIME_RESULT cells={} defects={} result={}",
        gate.cells,
        gate.defects,
        if gate.defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(gate.defects == 0, "dense runtime defects={}", gate.defects);
    Ok(())
}

fn copy_gate(ctx: &Arc<CudaContext>, s: &Arc<CudaStream>, gate: &mut Gate) -> Result<()> {
    let upload = ctx.new_stream()?;
    let host = pattern(3 * 272, 7);
    let mut source = upload.alloc_zeros::<bf16>(host.len())?;
    #[cfg(feature = "rocm")]
    {
        gate.check(
            "copy.unfenced_read_rejected",
            s.clone_dtoh(&source).is_err(),
        );
        gate.check(
            "copy.unfenced_write_rejected",
            s.memset_zeros(&mut source).is_err(),
        );
        gate.check(
            "copy.unfenced_view_read_rejected",
            s.clone_dtoh(&source.slice(8..40)).is_err(),
        );
        gate.check(
            "copy.unfenced_view_write_rejected",
            s.memset_zeros(&mut source.slice_mut(8..40)).is_err(),
        );
        let mut destination = s.alloc_zeros::<bf16>(host.len())?;
        gate.check(
            "copy.unfenced_dtod_source_rejected",
            s.memcpy_dtod(&source, &mut destination).is_err(),
        );
        gate.check(
            "copy.unfenced_dtod_destination_rejected",
            s.memcpy_dtod(&destination, &mut source).is_err(),
        );
    }
    let mut staging = copy::pinned_zeros::<u8>(ctx, host.len() * 2)?;
    for (bytes, value) in staging
        .as_mut_slice()?
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(&host)
    {
        bytes.copy_from_slice(&value.to_bits().to_le_bytes());
    }
    copy::htod_bytes(&upload, &mut source, 0, &staging, 0, host.len() * 2)?;
    s.wait(&upload.record_event(None)?)?;
    #[cfg(feature = "rocm")]
    source.handoff(s)?;
    gate.check(
        "copy.typed_subview_after_upload_fence",
        s.clone_dtoh(&source.slice(8..40))? == host[8..40],
    );
    let mut target = s.clone_htod(&vec![bf16::from_f32(-9.0); 3 * 288])?;
    copy::columns(
        s,
        &source,
        &mut target,
        copy::Rect {
            rows: 3,
            width: 256,
            src_pitch: 272,
            src_col: 8,
            dst_pitch: 288,
            dst_col: 16,
        },
    )?;
    let got = s.clone_dtoh(&target)?;
    let mut expected = vec![bf16::from_f32(-9.0); 3 * 288];
    for row in 0..3 {
        expected[row * 288 + 16..row * 288 + 272]
            .copy_from_slice(&host[row * 272 + 8..row * 272 + 264]);
    }
    gate.check("copy.pitched_offsets_guards", got == expected);
    #[cfg(feature = "rocm")]
    {
        gate.check(
            "copy.previous_owner_rejected",
            upload.clone_dtoh(&source).is_err(),
        );
        source.handoff(&upload)?;
        upload.memset_zeros(&mut source)?;
        source.handoff(s)?;
        gate.check(
            "copy.handoff_write_order",
            s.clone_dtoh(&source)?.iter().all(|x| x.to_bits() == 0),
        );
        s.memcpy_htod(&host, &mut source)?;
        let ops = ops::Ops::new(ctx)?;
        let mut word = s.alloc_zeros::<i32>(1)?;
        let mut pending = s.launch_builder(&ops.advance);
        // A retained allocation cannot be handed off while a launch owns it.
        pending.arg(&mut word);
        gate.check(
            "copy.handoff_retained_launch_rejected",
            word.handoff(&upload).is_err(),
        );
        drop(pending);
        word.handoff(&upload)?;
        gate.check("copy.handoff_owner", Arc::ptr_eq(word.stream(), &upload));
    }
    let mut pinned = copy::pinned_zeros::<u8>(ctx, 96)?;
    copy::dtoh_bytes(s, &source, 16, &mut pinned, 8, 64)?;
    s.synchronize()?;
    let bytes = pinned.as_slice()?;
    let expected_bytes: Vec<_> = host[8..40]
        .iter()
        .flat_map(|x| x.to_bits().to_ne_bytes())
        .collect();
    gate.check(
        "copy.pinned_d2h",
        bytes[8..72] == expected_bytes && bytes[..8].iter().chain(&bytes[72..]).all(|&b| b == 0),
    );
    copy::htod_bytes(s, &mut target, 32, &pinned, 8, 64)?;
    s.synchronize()?;
    let got = s.clone_dtoh(&target)?;
    gate.check("copy.pinned_h2d", got[16..48] == host[8..40]);
    s.memcpy_htod(&host[40..72], &mut target.slice_mut(16..48))?;
    expected[16..48].copy_from_slice(&host[40..72]);
    gate.check(
        "copy.typed_mut_subview_guards",
        s.clone_dtoh(&target)? == expected,
    );
    gate.check(
        "copy.bounds",
        copy::columns(
            s,
            &source,
            &mut target,
            copy::Rect {
                rows: 4,
                width: 256,
                src_pitch: 272,
                src_col: 8,
                dst_pitch: 288,
                dst_col: 16,
            },
        )
        .is_err(),
    );
    let model = crate::model_spec::ModelSpec::from_llama(&config()?)?;
    let mut state = state::StateStore::new(&model, 129, s)?;
    gate.check(
        "state.bf16_capacity",
        state.kv.len() == 42
            && state
                .kv
                .iter()
                .all(|kv| kv.k.len() == 2 * 129 * 128 && kv.v.len() == kv.k.len()),
    );
    state.reset(s)?;
    #[cfg(feature = "rocm")]
    {
        let info = ctx.device_info()?;
        for name in [
            "gemm_skinny",
            "tcmqa",
            "markov",
            "moe_prefill",
            "vision_attn",
            "gdn_chunkwise",
        ] {
            gate.check(name, info.require_cuda_tensor_ops(name).is_err());
        }
        gate.check(
            "state.compact_unsupported",
            state::StateStore::new_with_format(&model, 129, crate::kv::KvFormat::Affine8, s)
                .is_err(),
        );
    }
    Ok(())
}

fn rms(input: &[f32], weights: &[bf16]) -> Vec<bf16> {
    let variance = input.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / input.len() as f64;
    let inv = (variance + 1e-6).sqrt().recip();
    input
        .iter()
        .zip(weights)
        .map(|(&x, w)| bf16::from_f32((f64::from(x) * inv * f64::from(w.to_f32())) as f32))
        .collect()
}
fn norm_gate(
    ctx: &Arc<CudaContext>,
    s: &Arc<CudaStream>,
    ops: &ops::Ops,
    gate: &mut Gate,
) -> Result<()> {
    let norm = cubin::module(ctx, "norm")?.load_function("rms_norm")?;
    for dim in [2053usize, 2048, 6144] {
        for rows in [1usize, 3, 16, 17, 33] {
            let x = pattern(rows * dim, 3);
            let z = pattern(rows * dim, 8);
            let w = pattern(dim, 11);
            let dx = s.clone_htod(&x)?;
            let dz = s.clone_htod(&z)?;
            let dw = s.clone_htod(&w)?;
            let mut y = s.alloc_zeros::<bf16>(rows * dim)?;
            let mut residual = s.alloc_zeros::<bf16>(rows * dim)?;
            unsafe {
                s.launch_builder(&norm)
                    .arg(&dx)
                    .arg(&dw)
                    .arg(&mut y)
                    .arg(&(rows as i32))
                    .arg(&(dim as i32))
                    .arg(&1e-6f32)
                    .launch(ops::grid(rows, 256))?;
            }
            let expected: Vec<_> = x
                .chunks_exact(dim)
                .flat_map(|row| rms(&row.iter().map(|x| x.to_f32()).collect::<Vec<_>>(), &w))
                .collect();
            gate.compare(
                &format!("norm.{rows}.{dim}"),
                &s.clone_dtoh(&y)?,
                &expected,
                2,
                dim,
            )?;
            for fp32 in [false, true] {
                if fp32 && dim > 2048 {
                    continue;
                }
                let f = if fp32 {
                    &ops.add_norm_fp32
                } else {
                    &ops.add_norm
                };
                unsafe {
                    s.launch_builder(f)
                        .arg(&dx)
                        .arg(&dz)
                        .arg(&dw)
                        .arg(&mut y)
                        .arg(&mut residual)
                        .arg(&(rows as i32))
                        .arg(&(dim as i32))
                        .arg(&1e-6f32)
                        .launch(ops::grid(rows, 256))?;
                }
                let sums: Vec<_> = x
                    .iter()
                    .zip(&z)
                    .map(|(a, b)| a.to_f32() + b.to_f32())
                    .collect();
                let rounded: Vec<_> = sums.iter().map(|&v| bf16::from_f32(v)).collect();
                let consumed: Vec<_> = if fp32 {
                    sums
                } else {
                    rounded.iter().map(|v| v.to_f32()).collect()
                };
                let expected: Vec<_> = consumed
                    .chunks_exact(dim)
                    .flat_map(|row| rms(row, &w))
                    .collect();
                gate.check("norm.residual_bits", s.clone_dtoh(&residual)? == rounded);
                gate.compare(
                    &format!("add_norm.{rows}.{dim}.fp32{fp32}"),
                    &s.clone_dtoh(&y)?,
                    &expected,
                    3,
                    dim,
                )?;
            }
        }
    }
    let rows = 3usize;
    let dim = 2048usize;
    let embed = pattern(37 * dim, 4);
    let weights = pattern(dim, 9);
    let ids = [3u32, 17, 36];
    let e = s.clone_htod(&embed)?;
    let w = s.clone_htod(&weights)?;
    let ids_gpu = s.clone_htod(&ids)?;
    let mut x = s.alloc_zeros::<bf16>(rows * dim)?;
    let mut y = s.alloc_zeros::<bf16>(rows * dim)?;
    unsafe {
        s.launch_builder(&ops.embed_rmsnorm)
            .arg(&e)
            .arg(&ids_gpu)
            .arg(&mut x)
            .arg(&w)
            .arg(&mut y)
            .arg(&(rows as i32))
            .arg(&(dim as i32))
            .arg(&1e-6f32)
            .launch(ops::grid(rows, 256))?;
    }
    let gathered: Vec<_> = ids
        .iter()
        .flat_map(|&id| {
            embed[id as usize * dim..(id as usize + 1) * dim]
                .iter()
                .copied()
        })
        .collect();
    gate.check("embed.gather", s.clone_dtoh(&x)? == gathered);
    let expected: Vec<_> = gathered
        .chunks_exact(dim)
        .flat_map(|row| {
            rms(
                &row.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
                &weights,
            )
        })
        .collect();
    gate.compare("embed.norm", &s.clone_dtoh(&y)?, &expected, 2, dim)?;
    Ok(())
}

fn rope_gate(s: &Arc<CudaStream>, ops: &ops::Ops, gate: &mut Gate) -> Result<()> {
    for capacity in [129usize, 4096] {
        for rows in [1usize, 3, 16, 17, 33] {
            for start in [0usize, 1, 127, capacity - rows] {
                if start + rows > capacity {
                    continue;
                }
                let input = pattern(rows * 2560, 19);
                let mut expected = input.clone();
                let sentinel = bf16::from_f32(-7.0);
                let mut ek = vec![sentinel; 2 * capacity * 128];
                let mut ev = ek.clone();
                for row in 0..rows {
                    for head in 0..20 {
                        for dim in 0..64 {
                            let off = row * 2560 + head * 128 + dim;
                            if head < 18 {
                                let angle = (start + row) as f32
                                    / 5000000f32.powf((2 * dim) as f32 / 128.0);
                                let c = bf16::from_f32(angle.cos()).to_f32();
                                let t = bf16::from_f32(angle.sin()).to_f32();
                                let a = input[off].to_f32();
                                let b = input[off + 64].to_f32();
                                let x = bf16::from_f32(
                                    bf16::from_f32(a * c).to_f32() - bf16::from_f32(b * t).to_f32(),
                                );
                                let y = bf16::from_f32(
                                    bf16::from_f32(b * c).to_f32() + bf16::from_f32(a * t).to_f32(),
                                );
                                expected[off] = x;
                                expected[off + 64] = y;
                                if head >= 16 {
                                    let dst = ((head - 16) * capacity + start + row) * 128 + dim;
                                    ek[dst] = x;
                                    ek[dst + 64] = y;
                                }
                            } else {
                                let dst = ((head - 18) * capacity + start + row) * 128 + dim;
                                ev[dst] = input[off];
                                ev[dst + 64] = input[off + 64];
                            }
                        }
                    }
                }
                let mut q = s.clone_htod(&input)?;
                let mut k = s.clone_htod(&vec![sentinel; ek.len()])?;
                let mut v = s.clone_htod(&vec![sentinel; ev.len()])?;
                let pos = s.clone_htod(&[start as i32])?;
                unsafe {
                    s.launch_builder(&ops.rope_kv)
                        .arg(&mut q)
                        .arg(&mut k)
                        .arg(&mut v)
                        .arg(&pos)
                        .arg(&(rows as i32))
                        .arg(&16i32)
                        .arg(&2i32)
                        .arg(&128i32)
                        .arg(&5000000f32)
                        .arg(&(capacity as i32))
                        .launch(ops::flat(rows * 20 * 64))?;
                }
                gate.compare(
                    &format!("rope.qkv.{capacity}.{rows}.{start}"),
                    &s.clone_dtoh(&q)?,
                    &expected,
                    6,
                    128,
                )?;
                let actual_k = s.clone_dtoh(&k)?;
                gate.compare(
                    &format!("rope.k.{capacity}.{rows}.{start}"),
                    &actual_k,
                    &ek,
                    6,
                    128,
                )?;
                gate.check("rope.v_exact", s.clone_dtoh(&v)? == ev);
                gate.check(
                    "rope.untouched_slots",
                    (0..2 * capacity)
                        .filter(|i| !(start..start + rows).contains(&(i % capacity)))
                        .all(|i| {
                            actual_k[i * 128..(i + 1) * 128]
                                .iter()
                                .all(|&x| x == sentinel)
                        }),
                );
            }
        }
    }
    Ok(())
}

fn activation_sample(s: &Arc<CudaStream>, ops: &ops::Ops, gate: &mut Gate) -> Result<()> {
    let dim = 6144usize;
    for rows in [1usize, 3, 16, 17, 33] {
        let input = pattern(rows * 2 * dim, 5);
        let x = s.clone_htod(&input)?;
        let mut y = s.alloc_zeros::<bf16>(rows * dim)?;
        unsafe {
            s.launch_builder(&ops.act)
                .arg(&x)
                .arg(&mut y)
                .arg(&(rows as i32))
                .arg(&(dim as i32))
                .arg(&0i32)
                .launch(ops::flat(rows * dim))?;
        }
        let expected: Vec<_> = (0..rows * dim)
            .map(|i| {
                let r = i / dim;
                let d = i % dim;
                let g = input[r * 2 * dim + d].to_f32();
                bf16::from_f32(
                    bf16::from_f32(g / (1.0 + (-g).exp())).to_f32()
                        * input[r * 2 * dim + dim + d].to_f32(),
                )
            })
            .collect();
        gate.compare(
            &format!("activation.twice.{rows}"),
            &s.clone_dtoh(&y)?,
            &expected,
            3,
            1,
        )?;
    }
    let mut values = pattern(130560, 31);
    values[130007] = bf16::from_f32(8.0);
    values[299] = bf16::from_f32(4.0);
    let logits = s.clone_htod(&values)?;
    let mut token = s.alloc_zeros::<u32>(1)?;
    unsafe {
        s.launch_builder(&ops.argmax)
            .arg(&logits)
            .arg(&mut token)
            .arg(&130560i32)
            .launch(ops::grid(1, 1024))?;
    }
    gate.check("sample.argmax_margin4", s.clone_dtoh(&token)? == [130007]);
    let mut position = s.clone_htod(&[127i32])?;
    unsafe {
        s.launch_builder(&ops.advance)
            .arg(&mut position)
            .launch(ops::grid(1, 1))?;
    }
    gate.check("sample.advance", s.clone_dtoh(&position)? == [128]);
    Ok(())
}

fn attention_reference(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    rows: usize,
    start: usize,
    capacity: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * 2048];
    for row in 0..rows {
        for head in 0..16 {
            let kh = head / 8;
            let len = start + row + 1;
            let scores: Vec<_> = (0..len)
                .map(|pos| {
                    (0..128)
                        .map(|d| {
                            f64::from(q[row * 2560 + head * 128 + d].to_f32())
                                * f64::from(k[(kh * capacity + pos) * 128 + d].to_f32())
                        })
                        .sum::<f64>()
                        / 128f64.sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probabilities: Vec<_> = scores.iter().map(|&v| (v - max).exp()).collect();
            let sum = probabilities.iter().sum::<f64>();
            for d in 0..128 {
                let value = probabilities
                    .iter()
                    .enumerate()
                    .map(|(pos, &p)| p * f64::from(v[(kh * capacity + pos) * 128 + d].to_f32()))
                    .sum::<f64>()
                    / sum;
                out[row * 2048 + head * 128 + d] = bf16::from_f32(value as f32);
            }
        }
    }
    out
}

fn attention_gate(
    ctx: &Arc<CudaContext>,
    s: &Arc<CudaStream>,
    c: &Config,
    gate: &mut Gate,
) -> Result<()> {
    let plan = AttentionPlan {
        chunk: 128,
        qpack: 4,
        threads: 128,
        merge_threads: 256,
    };
    for capacity in [129usize, 4096] {
        for rows in [1usize, 3, 16, 17, 33] {
            for start in [0usize, 1, 127, capacity - rows] {
                if start + rows > capacity {
                    continue;
                }
                let mut q = pattern(rows * 2560, 7);
                let mut k = pattern(2 * capacity * 128, 9);
                let mut v = pattern(2 * capacity * 128, 11);
                for row in 0..rows {
                    q[row * 2560] = bf16::from_f32(4.0);
                }
                k[0] = bf16::from_f32(8.0);
                k[128] = bf16::from_f32(-8.0);
                for d in 0..128 {
                    v[d] = bf16::from_f32(2.0);
                    v[128 + d] = bf16::from_f32(-2.0);
                }
                let expected = attention_reference(&q, &k, &v, rows, start, capacity);
                let dq = s.clone_htod(&q)?;
                let dk = s.clone_htod(&k)?;
                let mut dv = s.clone_htod(&v)?;
                let prefix = s.clone_htod(&[start as i32])?;
                let guard = bf16::from_f32(13.0);
                let mut output = s.clone_htod(&vec![guard; (rows + 1) * 2048])?;
                let shape = verification::Shape {
                    rows,
                    start,
                    q_heads: 16,
                    kv_heads: 2,
                    capacity,
                    head_dim: 128,
                    causal: true,
                    window_left: None,
                };
                let mut mq = verification::Verification::with_plan(
                    ctx,
                    s,
                    c,
                    MultiShape {
                        rows: rows.min(16),
                        capacity,
                        causal: true,
                    },
                    MultiPlan {
                        implementation: MultiImpl::V1,
                        launch: plan,
                    },
                )?;
                let positions: Vec<i32> = (0..capacity as i32).collect();
                let positions = s.clone_htod(&positions)?;
                mq.run_prefill(verification::Query {
                    qkv: &dq,
                    k: &dk,
                    v: &dv,
                    out: &mut output,
                    prefix: &positions,
                    shape,
                })?;
                let actual = s.clone_dtoh(&output.slice(..rows * 2048))?;
                gate.compare(
                    &format!("attention.prefill.{capacity}.{rows}.{start}"),
                    &actual,
                    &expected,
                    2,
                    capacity + 128,
                )?;
                gate.check(
                    "attention.prefill.output_guard",
                    s.clone_dtoh(&output.slice(rows * 2048..))?
                        .iter()
                        .all(|x| *x == guard),
                );
                gate.check(
                    "attention.prefill.positions_unchanged",
                    s.clone_dtoh(&positions)?
                        .iter()
                        .copied()
                        .eq(0..capacity as i32),
                );
                if rows <= 16 {
                    mq.run(&dq, &dk, &dv, &mut output, &prefix, shape)?;
                    gate.check(
                        "attention.prefill.single_chunk_identity",
                        s.clone_dtoh(&output.slice(..rows * 2048))? == actual,
                    );
                }
                if rows == 1 {
                    let mut single = decode::Attention::new(ctx, s, c, capacity, plan)?;
                    let length = s.clone_htod(&[(start + 1) as i32])?;
                    single.run(s, &dq, &dk, &dv, &mut output, &length)?;
                    gate.compare(
                        &format!("attention.decode.{capacity}.{start}"),
                        &s.clone_dtoh(&output.slice(..rows * 2048))?,
                        &expected,
                        2,
                        capacity + 128,
                    )?;
                }
                if capacity == 129 && rows == 1 && start == 127 {
                    for d in 0..128 {
                        v.swap(d, 128 + d);
                    }
                    s.memcpy_htod(&v, &mut dv)?;
                    mq.run(&dq, &dk, &dv, &mut output, &prefix, shape)?;
                    let mut control = Gate::default();
                    let incorrectly_passed = control.compare(
                        "EXPECTED_MUTANT.V_slots0_1",
                        &s.clone_dtoh(&output.slice(..rows * 2048))?,
                        &expected,
                        2,
                        capacity + 128,
                    )?;
                    gate.check("mutant.V_slots0_1_rejected", !incorrectly_passed);
                }
            }
        }
    }
    for rows in [17usize, 33] {
        gate.check(
            "attention.rows_capacity",
            verification::Verification::with_plan(
                ctx,
                s,
                c,
                MultiShape {
                    rows,
                    capacity: 129,
                    causal: true,
                },
                MultiPlan {
                    implementation: MultiImpl::V1,
                    launch: plan,
                },
            )
            .is_err(),
        );
    }
    Ok(())
}

fn conversion_gate(ctx: &Arc<CudaContext>, s: &Arc<CudaStream>, gate: &mut Gate) -> Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var_os("FF_DENSE_EVIDENCE").context("FF_DENSE_EVIDENCE required")?,
    );
    std::fs::create_dir_all(&root)?;
    let source = root.join("conversion.cu");
    let binary = root.join("conversion.bin");
    let code = r#"#include "gpu_compat.cuh"
extern "C" __global__ void conversion(const float* x, __nv_bfloat16* scalar, __nv_bfloat162* packed, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) scalar[i] = __float2bfloat16_rn(x[i]);
    if (2 * i + 1 < n) packed[i] = __floats2bfloat162_rn(x[2 * i], x[2 * i + 1]);
}
"#;
    std::fs::write(&source, code)?;
    let mut command = conversion_command(ctx)?;
    command
        .arg("-I")
        .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"))
        .arg(&source)
        .arg("-o")
        .arg(&binary);
    std::fs::write(root.join("conversion.COMMAND"), format!("{command:?}\n"))?;
    let result = command
        .output()
        .context("execute conversion gate compiler")?;
    std::fs::write(
        root.join("conversion.log"),
        [result.stdout, result.stderr].concat(),
    )?;
    std::fs::write(
        root.join("conversion.rc"),
        format!("{:?}\n", result.status.code()),
    )?;
    ensure!(
        result.status.success(),
        "RUNNER_FAIL conversion compiler failed"
    );
    let bytes = std::fs::read(&binary)?;
    #[cfg(feature = "cuda")]
    let module = ctx.load_module(cudarc::nvrtc::Ptx::from_binary(bytes))?;
    #[cfg(feature = "rocm")]
    let module = ctx.load_binary(&bytes)?;
    std::fs::remove_file(binary)?;
    let function = module.load_function("conversion")?;
    let bits = [
        0x3f808000u32,
        0x3f818000,
        0xbf808000,
        0xbf818000,
        0x3f807fff,
        0x3f808001,
        0xbf807fff,
        0xbf808001,
        0,
        0x80000000,
        0x7f800000,
        0xff800000,
        0x7fc12345,
        0xffa12345,
        1,
        0x80000001,
        0x7f7fffff,
        0xff7fffff,
    ];
    let values: Vec<_> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let input = s.clone_htod(&values)?;
    let mut scalar = s.alloc_zeros::<bf16>(bits.len())?;
    let mut packed = s.alloc_zeros::<bf16>(bits.len())?;
    unsafe {
        s.launch_builder(&function)
            .arg(&input)
            .arg(&mut scalar)
            .arg(&mut packed)
            .arg(&(bits.len() as i32))
            .launch(ops::flat(bits.len()))?;
    }
    for (name, actual) in [
        ("scalar", s.clone_dtoh(&scalar)?),
        ("packed", s.clone_dtoh(&packed)?),
    ] {
        for (i, (&raw, &value)) in bits.iter().zip(&actual).enumerate() {
            let nan = raw & 0x7f800000 == 0x7f800000 && raw & 0x007fffff != 0;
            let expected = (raw.wrapping_add(0x7fff + ((raw >> 16) & 1)) >> 16) as u16;
            gate.check(
                &format!("bf16.{name}.{i}.input_{raw:08x}"),
                if nan {
                    value.is_nan()
                } else {
                    value.to_bits() == expected
                },
            );
        }
    }
    Ok(())
}

fn conversion_command(ctx: &Arc<CudaContext>) -> Result<std::process::Command> {
    #[cfg(feature = "rocm")]
    {
        let compiler = std::env::var_os("HIP_HOME")
            .or_else(|| std::env::var_os("ROCM_PATH"))
            .map(|home| std::path::PathBuf::from(home).join("bin/hipcc"))
            .unwrap_or_else(|| "hipcc".into());
        let mut command = std::process::Command::new(compiler);
        command
            .args(["--genco", "-O3", "--std=c++17", "-mno-wavefrontsize64"])
            .arg(format!("--offload-arch={}", ctx.architecture()));
        Ok(command)
    }
    #[cfg(feature = "cuda")]
    {
        let compiler = std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map(|home| std::path::PathBuf::from(home).join("bin/nvcc"))
            .unwrap_or_else(|| "nvcc".into());
        let mut command = std::process::Command::new(compiler);
        command
            .args(["--cubin", "-O3", "--std=c++17"])
            .arg(format!("-arch=sm_{}", cubin::capability(ctx)?));
        Ok(command)
    }
}

#[cfg(feature = "rocm")]
fn blas_gate(s: &Arc<CudaStream>, gate: &mut Gate) -> Result<()> {
    use crate::backend::setup::LinearShape;
    for (output, input, bias) in [(2560usize, 2048usize, false), (264, 272, true)] {
        for rows in [1usize, 3, 16] {
            if rows == 1 && bias {
                continue;
            }
            let shape = LinearShape {
                rows,
                output,
                input,
            };
            let w = pattern(output * input, 33);
            let x = pattern(rows * input, 47);
            let b = pattern(output, 53);
            let dw = s.clone_htod(&w)?;
            let dx = s.clone_htod(&x)?;
            let mut dy = s.alloc_zeros::<bf16>(rows * output)?;
            let db = if bias {
                Some(Arc::new(s.clone_htod(&b)?))
            } else {
                None
            };
            let library = blas::Blas::new(s.clone())?;
            let choice = library.prepare(shape, db.clone(), None)?;
            let frozen: blas::Choice = serde_json::from_slice(&serde_json::to_vec(&choice)?)?;
            let rebound = blas::Blas::new(s.clone())?;
            gate.check(
                "blas.identity_rebind",
                rebound.prepare(shape, db.clone(), Some(&frozen))? == choice,
            );
            println!(
                "HIP_BLAS_CHOICE {} cached=true measurements=0 tuned=false",
                serde_json::to_string(&choice)?
            );
            rebound.linear(shape, db.as_deref(), &dw, &dx, &mut dy)?;
            let reference = |bias: Option<&[bf16]>| -> Vec<bf16> {
                (0..rows * output)
                    .map(|i| {
                        let row = i / output;
                        let o = i % output;
                        let sum = (0..input)
                            .map(|k| {
                                f64::from(w[o * input + k].to_f32())
                                    * f64::from(x[row * input + k].to_f32())
                            })
                            .sum::<f64>();
                        bf16::from_f32(
                            (sum + bias.map_or(0.0, |b| f64::from(b[o].to_f32()))) as f32,
                        )
                    })
                    .collect()
            };
            let expected = reference(bias.then_some(&b[..]));
            gate.compare(
                &format!("blas.rows{rows}.weight[{output},{input}].bias{bias}"),
                &s.clone_dtoh(&dy)?,
                &expected,
                2,
                input,
            )?;
            if bias {
                let second_bias: Vec<_> = pattern(output, 71)
                    .into_iter()
                    .map(|b| bf16::from_f32(b.to_f32() + 1.0))
                    .collect();
                let second = Arc::new(s.clone_htod(&second_bias)?);
                let prepared = rebound.prepare(shape, Some(second.clone()), Some(&frozen));
                gate.check("blas.second_bias_same_shape_prepared", prepared.is_ok());
                let unbiased = rebound.prepare(shape, None, None);
                gate.check("blas.unbiased_same_shape_prepared", unbiased.is_ok());
                if prepared.is_ok() && unbiased.is_ok() {
                    for (label, binding, values) in [
                        ("second", Some(second.as_ref()), Some(&second_bias[..])),
                        ("unbiased", None, None),
                        ("first", db.as_deref(), Some(&b[..])),
                    ] {
                        rebound.linear(shape, binding, &dw, &dx, &mut dy)?;
                        gate.compare(
                            &format!("blas.binding.{label}.rows{rows}.weight[{output},{input}]"),
                            &s.clone_dtoh(&dy)?,
                            &reference(values),
                            2,
                            input,
                        )?;
                    }
                }
                let unprepared = s.clone_htod(&b)?;
                gate.check(
                    "blas.unprepared_bias_rejected",
                    rebound
                        .linear(shape, Some(&unprepared), &dw, &dx, &mut dy)
                        .is_err(),
                );
            }
        }
    }
    Ok(())
}
