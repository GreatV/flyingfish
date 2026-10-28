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

#[cfg(feature = "rocm")]
fn gdn_core_reference(
    rows: usize,
    heads: usize,
    initial: &[f32],
    prepared: [&[f64]; 5],
) -> (Vec<bf16>, Vec<f64>) {
    let [q, k, v, beta, g] = prepared;
    let mut state: Vec<_> = initial.iter().map(|x| f64::from(*x)).collect();
    let mut out = vec![bf16::ZERO; rows * heads * 128];
    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * 128;
            let s = &mut state[head * 128 * 128..][..128 * 128];
            let decay = g[row * heads + head].exp();
            for value in s.iter_mut() {
                *value *= decay;
            }
            for column in 0..128 {
                let dot: f64 = (0..128).map(|d| k[base + d] * s[d * 128 + column]).sum();
                let delta = (v[base + column] - dot) * beta[row * heads + head];
                for d in 0..128 {
                    s[d * 128 + column] += k[base + d] * delta;
                }
                let value: f64 = (0..128).map(|d| q[base + d] * s[d * 128 + column]).sum();
                out[base + column] = bf16::from_f64(value);
            }
        }
    }
    (out, state)
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires HIP GPU lease and FF_DENSE_GPU_WINDOW=granted"]
fn hip_gdn_runtime_gpu() -> Result<()> {
    let minicpm = crate::model_spec::ModelSpec::minicpm(
        &serde_json::from_str(include_str!("../../../tests/data/minicpm-config.json"))?,
        &serde_json::json!({"eos_token_id":[248044,248046]}),
    )?;
    let qwen = crate::model_spec::ModelSpec::qwen(
        &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
        &serde_json::json!({}),
    )?;
    for model in [minicpm, qwen] {
        hip_gdn_case(model)?;
    }
    Ok(())
}

#[cfg(feature = "rocm")]
fn hip_gdn_case(model: crate::model_spec::ModelSpec) -> Result<()> {
    use crate::backend::cuda::gdn::{GdnBindings, GdnOps};
    use crate::backend::cuda::moe::testing::{capture, view};
    use crate::gdn_plan::GdnWorkspace;
    use crate::op_contract::{ExecutionPhase, GdnSpec};
    use contracts::{Scratch, SeqRef, TensorMut};
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = crate::backend::cuda::Device::new(0)?;
    let stream = &device.stream;
    let spec = GdnSpec::from_model(&model)?;
    let (heads, cd) = (spec.hv as usize, spec.conv_dim()? as usize);
    ensure!(
        matches!((heads, cd, spec.hk), (32, 8192, 16) | (48, 10240, 16)),
        "GDN fixture geometry"
    );
    let mut gate = Gate::default();
    {
        let mut store = state::StateStore::new(&model, 129, stream)?;
        gate.check(
            "gdn.store_layout",
            store.gdn.len() == if heads == 32 { 30 } else { 48 }
                && store.kv.len() == if heads == 32 { 10 } else { 16 }
                && store
                    .gdn
                    .iter()
                    .all(|s| s.s.len() == heads * 128 * 128 && s.history.len() == cd * 3),
        );
        for state in &mut store.gdn {
            stream.memcpy_htod(&vec![0.25f32; state.s.len()], &mut state.s)?;
            stream.memcpy_htod(&vec![-0.5f32; state.history.len()], &mut state.history)?;
        }
        store.set_sequence(stream, 23, 17, 17)?;
        gate.check(
            "gdn.store_sequence",
            stream.clone_dtoh(store.sequence.as_ref().context("GDN sequence missing")?)?
                == [23, 17, 40, 129, 1, 0, 0, 0],
        );
        store.reset(stream)?;
        let mut cleared = true;
        for state in &store.gdn {
            cleared &= stream
                .clone_dtoh(&state.s)?
                .iter()
                .all(|v| v.to_bits() == 0)
                && stream
                    .clone_dtoh(&state.history)?
                    .iter()
                    .all(|v| v.to_bits() == 0);
        }
        gate.check("gdn.store_reset", cleared);
        gate.check(
            "gdn.store_compact_rejected",
            state::StateStore::new_with_format(&model, 129, crate::kv::KvFormat::Affine8, stream)
                .is_err(),
        );
    }
    for (rows, valid) in [
        (1u32, 1u32),
        (3, 3),
        (17, 17),
        (37, 33),
        (129, 127),
        (257, 257),
    ] {
        let n = valid as usize;
        let mut qkv = pattern(rows as usize * cd, 71);
        qkv[n * cd..].fill(bf16::NAN);
        let conv = pattern(cd * 4, 73);
        let a = pattern(rows as usize * heads, 79);
        let b = pattern(rows as usize * heads, 83);
        let a_log: Vec<_> = (0..heads)
            .map(|h| bf16::from_f32(-3.0 + h as f32 / 128.0))
            .collect();
        let dt = pattern(heads, 89);
        let history: Vec<_> = pattern(cd * 3, 97).iter().map(|v| v.to_f32()).collect();
        let initial: Vec<_> = (0..heads * 128 * 128)
            .map(|i| ((i * 19 % 251) as f32 - 125.0) / 1024.0)
            .collect();
        let dqkv = stream.clone_htod(&qkv)?;
        let dc = stream.clone_htod(&conv)?;
        let da = stream.clone_htod(&a)?;
        let db = stream.clone_htod(&b)?;
        let dal = stream.clone_htod(&a_log)?;
        let ddt = stream.clone_htod(&dt)?;
        let mut dh = stream.clone_htod(&history)?;
        let mut ds = stream.clone_htod(&initial)?;
        let dout =
            stream.clone_htod(&vec![bf16::from_bits(0x7fc5); rows as usize * heads * 128])?;
        let metadata =
            stream.clone_htod(&[23i32, valid as i32, 23 + valid as i32, 8192, 1, 0, 0, 0])?;
        let meta = SeqRef {
            ptr: metadata.device_ptr(stream).0,
        };
        let ws = GdnWorkspace::new(&spec, rows, false)?;
        let scratch = stream.clone_htod(&vec![0xa5u8; ws.bytes as usize + 256])?;
        let plan = GdnOps::prepare(
            &device.ctx,
            &spec,
            rows,
            GdnBindings {
                qkv: view(&dqkv, stream, [rows, cd as u32], [cd as u64, 1])?,
                a: view(&da, stream, [rows, spec.hv], [heads as u64, 1])?,
                b: view(&db, stream, [rows, spec.hv], [heads as u64, 1])?,
                conv_weight: view(&dc, stream, [cd as u32, 4], [4, 1])?,
                a_log: view(&dal, stream, [spec.hv], [1])?,
                dt_bias: view(&ddt, stream, [spec.hv], [1])?,
                core_out: TensorMut(view(
                    &dout,
                    stream,
                    [rows, spec.hv, 128],
                    [heads as u64 * 128, 128, 1],
                )?),
                s: TensorMut(view(&ds, stream, [spec.hv, 128, 128], [128 * 128, 128, 1])?),
                history: TensorMut(view(&dh, stream, [cd as u32, 3], [3, 1])?),
            },
            false,
        )?;
        let phase = if rows == 1 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let execute = || {
            GdnOps::run(
                &plan,
                stream,
                meta,
                phase,
                Scratch {
                    ptr: scratch.device_ptr(stream).0,
                    bytes: ws.bytes,
                },
            )
        };
        execute()?;
        let actual = stream.clone_dtoh(&dout)?;
        let state = stream.clone_dtoh(&ds)?;
        let hist = stream.clone_dtoh(&dh)?;
        let bytes = stream.clone_dtoh(&scratch)?;
        gate.check(
            "gdn.output_guard",
            actual[n * heads * 128..]
                .iter()
                .all(|v| v.to_bits() == 0x7fc5),
        );
        gate.check(
            "gdn.scratch_guard",
            bytes[ws.bytes as usize..].iter().all(|v| *v == 0xa5),
        );
        let read = |offset: u64, count: usize| -> Result<Vec<f64>> {
            let start = offset as usize;
            let result: Vec<_> = bytes[start..start + count * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f64::from(f32::from_le_bytes(*v)))
                .collect();
            ensure!(
                result.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL nonfinite GDN workspace"
            );
            Ok(result)
        };
        let prepared = [
            read(ws.q_fp32.offset, n * heads * 128)?,
            read(ws.k_fp32.offset, n * heads * 128)?,
            read(ws.v_fp32.offset, n * heads * 128)?,
            read(ws.beta_fp32.offset, n * heads)?,
            read(ws.g.offset, n * heads)?,
        ];
        let conv_actual: Vec<_> = bytes[ws.conv_out.offset as usize..][..n * cd * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect();
        let mut conv_ref = Vec::with_capacity(n * cd);
        for row in 0..n {
            for ch in 0..cd {
                let value: f64 = (0..4)
                    .map(|tap| {
                        let index = row + tap;
                        let x = if index < 3 {
                            history[ch * 3 + index]
                        } else {
                            qkv[(index - 3) * cd + ch].to_f32()
                        };
                        f64::from(x) * f64::from(conv[ch * 4 + tap].to_f32())
                    })
                    .sum();
                let rounded = bf16::from_f64(value).to_f64();
                conv_ref.push(bf16::from_f64(rounded / (1.0 + (-rounded).exp())));
            }
        }
        gate.compare(&format!("gdn.conv.{rows}"), &conv_actual, &conv_ref, 2, 4)?;
        let mut norm_ratio = 0f64;
        let norm_bound = crate::backend::cuda::gamma(136, 2f64.powi(-24));
        for row in 0..n {
            for head in 0..heads {
                for (side, prepared_side) in prepared[..2].iter().enumerate() {
                    let key_heads = spec.hk as usize;
                    let base = row * cd + (side * key_heads + head / (heads / key_heads)) * 128;
                    let source = &conv_actual[base..base + 128];
                    let scale = (source.iter().map(|v| v.to_f64().powi(2)).sum::<f64>()
                        + f64::from(spec.l2_eps))
                    .sqrt()
                    .recip()
                        / if side == 0 { 128f64.sqrt() } else { 1.0 };
                    for (d, value) in source.iter().enumerate() {
                        let expected = value.to_f64() * scale;
                        let error =
                            (prepared_side[(row * heads + head) * 128 + d] - expected).abs();
                        norm_ratio = norm_ratio.max(if expected == 0.0 {
                            if error == 0.0 { 0.0 } else { f64::INFINITY }
                        } else {
                            error / (expected.abs() * norm_bound)
                        });
                    }
                }
            }
        }
        gate.check("gdn.qk_norm", norm_ratio <= 1.0);
        let actual_beta: Vec<_> = prepared[3].iter().map(|v| bf16::from_f64(*v)).collect();
        let expected_beta: Vec<_> = b[..n * heads]
            .iter()
            .map(|v| bf16::from_f64(1.0 / (1.0 + (-v.to_f64()).exp())))
            .collect();
        gate.compare(
            &format!("gdn.beta.{rows}"),
            &actual_beta,
            &expected_beta,
            1,
            1,
        )?;
        let gate_bound = crate::backend::cuda::gamma(8, 2f64.powi(-24));
        let gate_error = prepared[4]
            .iter()
            .enumerate()
            .map(|(i, actual)| {
                let h = i % heads;
                let t = f64::from(a[i].to_f32() + dt[h].to_f32());
                let expected = -a_log[h].to_f64().exp() * t.exp().ln_1p();
                (actual - expected).abs() / (1.0 + expected.abs())
            })
            .fold(0f64, f64::max);
        gate.check("gdn.decay_gate", gate_error <= gate_bound);
        gate.check(
            "gdn.value_widen",
            prepared[2].iter().enumerate().all(|(i, v)| {
                *v == conv_actual[(i / (heads * 128)) * cd + 4096 + i % (heads * 128)].to_f64()
            }),
        );
        let (expected, expected_state) =
            gdn_core_reference(n, heads, &initial, prepared.each_ref().map(Vec::as_slice));
        gate.compare(
            &format!("gdn.core.{rows}"),
            &actual[..n * heads * 128],
            &expected,
            1,
            n * 512,
        )?;
        let bound = crate::backend::cuda::gamma(n * 512 + 128, 2f64.powi(-24));
        let error = state
            .iter()
            .zip(&expected_state)
            .try_fold(0f64, |e, (a, b)| {
                ensure!(
                    a.is_finite() && b.is_finite(),
                    "RUNNER_FAIL GDN state nonfinite"
                );
                Ok::<_, anyhow::Error>(e.max((f64::from(*a) - b).abs() / (1.0 + b.abs())))
            })?;
        gate.check("gdn.state", error <= bound);
        let history_bad = hist
            .iter()
            .enumerate()
            .filter(|(i, v)| {
                let (ch, tap) = (i / 3, i % 3);
                let index = n + tap;
                let expected = if index < 3 {
                    history[ch * 3 + index]
                } else {
                    qkv[(index - 3) * cd + ch].to_f32()
                };
                v.to_bits() != expected.to_bits()
            })
            .count();
        gate.check("gdn.raw_history", history_bad == 0);
        stream.memcpy_htod(&initial, &mut ds)?;
        stream.memcpy_htod(&history, &mut dh)?;
        let graph = capture(stream, execute)?;
        for replay in 0..2 {
            if replay != 0 {
                stream.memcpy_htod(&initial, &mut ds)?;
                stream.memcpy_htod(&history, &mut dh)?;
            }
            graph.launch()?;
            let observed = stream.clone_dtoh(&dout)?;
            let observed_state = stream.clone_dtoh(&ds)?;
            let observed_hist = stream.clone_dtoh(&dh)?;
            gate.check(
                "gdn.graph_bits",
                observed
                    .iter()
                    .zip(&actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && observed_state
                        .iter()
                        .zip(&state)
                        .chain(observed_hist.iter().zip(&hist))
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
            );
        }
        if rows == 3 {
            let differences = state[..128 * 128]
                .iter()
                .zip(&state[128 * 128..2 * 128 * 128])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(differences > 0, "GDN head0-for-head1 control escaped");
            println!("PASS HIP_GDN_CONTROL wrong_state_head differences={differences}");
        }
        if rows == 1 && heads == 32 {
            let norm = pattern(128, 101);
            let z = pattern(heads * 128, 103);
            let dn = stream.clone_htod(&norm)?;
            let dz = stream.clone_htod(&z)?;
            let gated = ops::GatedRmsInput {
                x: view(
                    &dout,
                    stream,
                    [1, spec.hv, 128],
                    [heads as u64 * 128, 128, 1],
                )?,
                weight: view(&dn, stream, [128], [1])?,
                gate: view(&dz, stream, [1, spec.hv, 128], [heads as u64 * 128, 128, 1])?,
            };
            for with_norm in [false, true] {
                stream.memcpy_htod(&initial, &mut ds)?;
                stream.memcpy_htod(&history, &mut dh)?;
                let execute_fused = || -> Result<()> {
                    ensure!(
                        GdnOps::run_step(
                            &plan,
                            stream,
                            meta,
                            phase,
                            Scratch {
                                ptr: scratch.device_ptr(stream).0,
                                bytes: ws.bytes,
                            },
                            false,
                            with_norm.then_some(&gated)
                        )? == with_norm,
                        "fused GDN gated output contract"
                    );
                    Ok(())
                };
                execute_fused()?;
                let fused = stream.clone_dtoh(&dout)?;
                let fused_state = stream.clone_dtoh(&ds)?;
                let fused_hist = stream.clone_dtoh(&dh)?;
                let mut expected = expected.clone();
                if with_norm {
                    for (head, values) in expected.as_chunks_mut::<128>().0.iter_mut().enumerate() {
                        let variance =
                            values.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / 128.0;
                        let inv = 1.0 / (variance + f64::from(spec.rms_eps)).sqrt();
                        for (d, value) in values.iter_mut().enumerate() {
                            let normalized = bf16::from_f64(value.to_f64() * inv);
                            let scaled = bf16::from_f64(normalized.to_f64() * norm[d].to_f64());
                            let gate = z[head * 128 + d].to_f64();
                            *value = bf16::from_f64(scaled.to_f64() * gate / (1.0 + (-gate).exp()));
                        }
                    }
                }
                gate.compare(
                    &format!("gdn.fused.norm{with_norm}"),
                    &fused,
                    &expected,
                    if with_norm { 4 } else { 1 },
                    640,
                )?;
                gate.check(
                    "gdn.fused_state",
                    fused_state.iter().zip(&expected_state).all(|(a, b)| {
                        a.is_finite() && (f64::from(*a) - b).abs() / (1.0 + b.abs()) <= bound
                    }),
                );
                gate.check("gdn.fused_history", fused_hist == hist);
                stream.memcpy_htod(&initial, &mut ds)?;
                stream.memcpy_htod(&history, &mut dh)?;
                capture(stream, execute_fused)?.launch()?;
                gate.check(
                    "gdn.fused_graph_bits",
                    stream
                        .clone_dtoh(&dout)?
                        .iter()
                        .zip(&fused)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                        && stream.clone_dtoh(&ds)? == fused_state
                        && stream.clone_dtoh(&dh)? == fused_hist,
                );
            }
        }
        println!(
            "HIP_GDN heads={heads} rows={rows} valid={valid} norm_ratio={norm_ratio:e} state_error={error:e} state_bound={bound:e}"
        );
    }
    println!(
        "HIP_GDN_SUMMARY heads={heads} cells={} defects={}",
        gate.cells, gate.defects
    );
    ensure!(gate.defects == 0, "GDN defects={}", gate.defects);
    Ok(())
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
    #[cfg(feature = "rocm")]
    graph_gate(&ctx, &s, &mut gate)?;
    #[cfg(feature = "rocm")]
    {
        let supports_wmma = ctx.supports_wmma();
        let info = ctx.device_info()?;
        ensure!(
            wmma_cells(&info) == supports_wmma,
            "RUNNER_FAIL gate WMMA selection differs from runtime capability"
        );
        if wmma_cells(&info) {
            wmma_gate(&s, &mut gate)?;
            fp8_encoding_gate(&ctx, &s, &mut gate)?;
            for layout in [
                crate::fp8::Fp8ScaleLayout::PerChannel,
                crate::fp8::Fp8ScaleLayout::Block(crate::fp8::Fp8BlockScale::new(
                    [128, 128],
                    safetensors::Dtype::BF16,
                )?),
            ] {
                fp8_projection_gate(&ctx, &s, &mut gate, layout)?;
            }
            tcmqa_gate(&ctx, &s, &c, &mut gate)?;
            markov_gate(&ctx, &s, &mut gate)?;
        } else {
            println!("WMMA cells NOT_COVERED arch={}", info.arch);
        }
    }
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

#[cfg(feature = "rocm")]
fn fp8_projection_gate(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gate: &mut Gate,
    layout: crate::fp8::Fp8ScaleLayout,
) -> Result<()> {
    let module = cubin::module(ctx, "fp8_linear")?;
    let mut shapes = vec![(263usize, 257usize), (512, 2048), (2048, 512)];
    let kind = if layout == crate::fp8::Fp8ScaleLayout::PerChannel {
        "channel"
    } else {
        shapes.push((17408, 5120));
        "block"
    };
    for (n, k) in shapes {
        let codes: Vec<u8> = (0..n * k)
            .map(|i| ((i * 17 % 257) % 127) as u8 | (u8::from(i * 29 % 263 > 131) << 7))
            .collect();
        let count = layout.scale_shape(n as u32, k as u32).iter().product();
        let scales: Vec<_> = (0..count)
            .map(|i| bf16::from_f32((1 + i * 31 % 263) as f32 / 65536.0))
            .collect();
        let max_rows = if n > 2048 { 3 } else { 33 };
        let input = pattern(max_rows * k, 19);
        let reference = fp8_projection_reference(&codes, &scales, &input, k, layout)?;
        let weights = stream.clone_htod(&codes)?;
        let scale = stream.clone_htod(&scales)?;
        let x = stream.clone_htod(&input)?;
        let wrong: Vec<_> = scales.iter().cycle().skip(1).take(count).copied().collect();
        let wrong_scale = stream.clone_htod(&wrong)?;
        let sentinel = bf16::from_bits(0x7fc1);
        for mode in ["simt", "wmma"] {
            let name = format!("fp8_{kind}_bf16_{mode}");
            let function = module.load_function(&name)?;
            let tiled = name.ends_with("wmma");
            let mut longest = None;
            for rows in [33usize, 17, 16, 9, 8, 3, 1] {
                if rows > max_rows {
                    continue;
                }
                let mut y = stream.clone_htod(&vec![sentinel; rows * n + 16])?;
                let launch = |y: &mut CudaSlice<bf16>, scales: &CudaSlice<bf16>| unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&weights)
                        .arg(scales)
                        .arg(&x)
                        .arg(y)
                        .arg(&(rows as i32))
                        .arg(&(n as i32))
                        .arg(&(k as i32))
                        .launch(LaunchConfig {
                            grid_dim: if tiled {
                                (n.div_ceil(64) as u32, rows.div_ceil(16) as u32, 1)
                            } else {
                                (n.div_ceil(4) as u32, rows as u32, 1)
                            },
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })
                };
                launch(&mut y, &scale)?;
                let eager = stream.clone_dtoh(&y)?;
                let capture = stream.begin_capture()?;
                launch(&mut y, &scale)?;
                let graph = capture.finish()?;
                graph.launch()?;
                let replay = stream.clone_dtoh(&y)?;
                let label = format!("fp8.{kind}.{name}.M{rows}.W[{n},{k}]");
                ensure!(
                    eager[..rows * n].iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL {label}: nonfinite output"
                );
                let errors = eager[..rows * n]
                    .iter()
                    .zip(&reference)
                    .filter(|(a, (value, bound))| (f64::from(a.to_f32()) - value).abs() > *bound)
                    .count();
                println!(
                    "FP8_CHANNEL_REFERENCE name={label} elements={} defects={errors} fp64=true",
                    rows * n
                );
                gate.check(&label, errors == 0);
                gate.check(
                    &format!("{label}.graph_identity_guard"),
                    eager
                        .iter()
                        .zip(&replay)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                        && eager[rows * n..]
                            .iter()
                            .all(|v| v.to_bits() == sentinel.to_bits()),
                );
                if let Some(previous) = &longest {
                    let previous: &Vec<bf16> = previous;
                    gate.check(
                        &format!("{label}.batch_position_identity"),
                        eager[..rows * n]
                            .iter()
                            .zip(previous)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    );
                } else {
                    longest = Some(eager[..rows * n].to_vec());
                }
                if rows == 1 {
                    launch(&mut y, &wrong_scale)?;
                    let mutant = stream.clone_dtoh(&y)?;
                    gate.check(
                        &format!("{label}.scale_index_mutant_rejected"),
                        mutant[..n]
                            .iter()
                            .zip(&reference)
                            .any(|(a, (value, bound))| {
                                (f64::from(a.to_f32()) - value).abs() > *bound
                            }),
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "rocm")]
fn fp8_projection_reference(
    codes: &[u8],
    scales: &[bf16],
    input: &[bf16],
    k: usize,
    layout: crate::fp8::Fp8ScaleLayout,
) -> Result<Vec<(f64, f64)>> {
    ensure!(
        k > 0 && codes.len().is_multiple_of(k) && input.len().is_multiple_of(k),
        "RUNNER_FAIL FP8 reference extents"
    );
    let n = codes.len() / k;
    let shape = layout.scale_shape(n as u32, k as u32);
    ensure!(scales.len() == shape[0] * shape[1], "FP8 reference scales");
    let mut reference = Vec::with_capacity(input.len() / k * n);
    let nu = (k + 6) as f64 * 2f64.powi(-24);
    for x in input.chunks_exact(k) {
        for (i, row) in codes.chunks_exact(k).enumerate() {
            let (value, absolute) =
                row.iter()
                    .zip(x)
                    .enumerate()
                    .fold((0.0f64, 0.0f64), |a, (j, (&w, x))| {
                        let index = match layout {
                            crate::fp8::Fp8ScaleLayout::PerChannel => i,
                            crate::fp8::Fp8ScaleLayout::Block(block) => {
                                let [bn, bk] = block.block();
                                i / bn as usize * shape[1] + j / bk as usize
                            }
                        };
                        let scale = scales[index];
                        let product = f64::from(crate::fp8::effective_weight(w, scale).to_f32())
                            * f64::from(x.to_f32());
                        (a.0 + product, a.1 + product.abs())
                    });
            let bound = value.abs() / 256.0 + (nu / (1.0 - nu)) * absolute + 2f64.powi(-134);
            reference.push((value, bound));
        }
    }
    Ok(reference)
}

#[cfg(feature = "rocm")]
fn fp8_channel_launch(
    s: &Arc<CudaStream>,
    function: (&CudaFunction, bool),
    weights: (&CudaSlice<u8>, &CudaSlice<bf16>),
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    shape: (usize, usize, usize),
) -> Result<()> {
    let (rows, n, k) = shape;
    unsafe {
        s.launch_builder(function.0)
            .arg(weights.0)
            .arg(weights.1)
            .arg(input)
            .arg(output)
            .arg(&(rows as i32))
            .arg(&(n as i32))
            .arg(&(k as i32))
            .launch(LaunchConfig {
                grid_dim: if function.1 {
                    (n.div_ceil(64) as u32, rows.div_ceil(16) as u32, 1)
                } else {
                    (n.div_ceil(4) as u32, rows as u32, 1)
                },
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

#[cfg(feature = "rocm")]
fn fp8_quantize(
    s: &Arc<CudaStream>,
    function: &CudaFunction,
    input: &mut CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    unsafe {
        s.launch_builder(function)
            .arg(input)
            .arg(&(rows as i32))
            .arg(&(width as i32))
            .arg(&0u64)
            .arg(&0u32)
            .arg(&(rows as u32))
            .launch(ops::grid(rows, 256))?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires GPU lease, FF_MODEL published per-channel FP8 checkpoint and FF_DENSE_EVIDENCE"]
fn fp8_published_expert_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let path = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL required")?);
    let spec = crate::model_spec::ModelSpec::read(&path)?;
    let moe = spec.moe.context("RUNNER_FAIL expert geometry missing")?;
    ensure!(
        spec.architecture == crate::model_spec::Architecture::MiniCpmV47
            && spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Channel
            && spec.hidden == 2048
            && moe.expert_intermediate == 512
            && moe.experts == 256
            && moe.experts_per_tok == 8,
        "RUNNER_FAIL published MiniCPM per-channel geometry"
    );
    let mapped = crate::checkpoint::MappedCheckpoint::open(&path)?;
    let views = mapped.views()?;
    let ctx = CudaContext::new(0)?;
    ctx.require_wmma()?;
    let s = ctx.new_stream()?;
    let module = cubin::module(&ctx, "fp8_linear")?;
    let quantize = module.load_function("e4m3_fake_quant_bf16")?;
    let act = cubin::module(&ctx, "act")?.load_function("silu_mul_split")?;
    let mut weights = Vec::new();
    for (role, n, k) in [
        ("gate", 512usize, 2048usize),
        ("up", 512, 2048),
        ("down", 2048, 512),
    ] {
        let name = format!("model.language_model.layers.0.mlp.experts.0.{role}_proj.weight");
        let scale_name = format!("{name}_scale");
        let w = views[&mapped.index.weight_map[&name]].tensor(&name)?;
        let scale = views[&mapped.index.weight_map[&scale_name]].tensor(&scale_name)?;
        ensure!(
            w.dtype() == safetensors::Dtype::F8_E4M3
                && w.shape() == [n, k]
                && scale.dtype() == safetensors::Dtype::BF16
                && scale.shape() == [n, 1],
            "RUNNER_FAIL native FP8 projection layout {name}"
        );
        let host_scale = scale
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| bf16::from_bits(u16::from_le_bytes(b)))
            .collect::<Vec<_>>();
        ensure!(
            w.data().iter().all(|v| v & 127 != 127)
                && host_scale.iter().all(|v| v.is_finite() && *v > bf16::ZERO),
            "RUNNER_FAIL nonfinite published projection {name}"
        );
        weights.push((
            w.data(),
            host_scale.clone(),
            s.clone_htod(w.data())?,
            s.clone_htod(&host_scale)?,
        ));
    }
    let device_weight_bytes: usize = weights.iter().map(|w| w.2.len() + 2 * w.3.len()).sum();
    ensure!(
        device_weight_bytes == 3 * 512 * 2048 + 2 * (512 + 512 + 2048),
        "RUNNER_FAIL native storage footprint"
    );
    let channel_ops = linear::Fp8Ops::new(&ctx)?;
    let mut gate = Gate::default();
    let sentinel = bf16::from_bits(0x7fc1);
    for name in [
        "fp8_channel_bf16_simt",
        "fp8_channel_bf16_wmma",
        "bound_channel",
        "bound_channel_invariant",
    ] {
        let function = if name.starts_with("bound_channel") {
            None
        } else {
            Some(module.load_function(name)?)
        };
        let tiled = name.ends_with("wmma");
        let mut longest = None;
        for rows in [33usize, 17, 16, 9, 8, 3, 1] {
            let host_input = pattern(rows * 2048, 23);
            let original = s.clone_htod(&host_input)?;
            let mut input = s.alloc_zeros::<bf16>(rows * 2048)?;
            let mut g = s.clone_htod(&vec![sentinel; rows * 512 + 16])?;
            let mut u = s.clone_htod(&vec![sentinel; rows * 512 + 16])?;
            let mut hidden = s.clone_htod(&vec![sentinel; rows * 512 + 16])?;
            let mut hidden_before_quantize = s.alloc_zeros::<bf16>(rows * 512)?;
            let mut output = s.clone_htod(&vec![sentinel; rows * 2048 + 16])?;
            let input_view = contracts::view(&input, &s, 0, [rows as u32, 2048], [2048, 1])?;
            let hidden_view = contracts::view(&hidden, &s, 0, [rows as u32, 512], [512, 1])?;
            let bound = if name.starts_with("bound_channel") {
                let bind = |stage: usize, x, y, n: u32, k: u32| {
                    channel_ops.bind_native(
                        linear::LinearWeightRef::Fp8 {
                            codes: contracts::view(
                                &weights[stage].2,
                                &s,
                                0,
                                [n, k],
                                [u64::from(k), 1],
                            )?,
                            scale: contracts::view(&weights[stage].3, &s, 0, [n, 1], [1, 1])?,
                            layout: crate::fp8::Fp8ScaleLayout::PerChannel,
                        },
                        x,
                        &contracts::TensorMut(contracts::view(
                            y,
                            &s,
                            0,
                            [rows as u32, n],
                            [u64::from(n), 1],
                        )?),
                        name == "bound_channel_invariant",
                    )
                };
                Some([
                    bind(0, input_view, &g, 512, 2048)?,
                    bind(1, input_view, &u, 512, 2048)?,
                    bind(2, hidden_view, &output, 2048, 512)?,
                ])
            } else {
                None
            };
            let mut run = || -> Result<()> {
                s.memcpy_dtod(&original, &mut input)?;
                if bound.is_some() {
                    channel_ops.fake_quant(&s, &mut contracts::TensorMut(input_view), None)?;
                } else {
                    fp8_quantize(&s, &quantize, &mut input, rows, 2048)?;
                }
                if let Some(bound) = &bound {
                    bound[0].run(&s)
                } else {
                    fp8_channel_launch(
                        &s,
                        (
                            function
                                .as_ref()
                                .context("direct projection function missing")?,
                            tiled,
                        ),
                        (&weights[0].2, &weights[0].3),
                        &input,
                        &mut g,
                        (rows, 512, 2048),
                    )
                }?;
                if let Some(bound) = &bound {
                    bound[1].run(&s)
                } else {
                    fp8_channel_launch(
                        &s,
                        (
                            function
                                .as_ref()
                                .context("direct projection function missing")?,
                            tiled,
                        ),
                        (&weights[1].2, &weights[1].3),
                        &input,
                        &mut u,
                        (rows, 512, 2048),
                    )
                }?;
                unsafe {
                    s.launch_builder(&act)
                        .arg(&g)
                        .arg(&u)
                        .arg(&mut hidden)
                        .arg(&((rows * 512) as i32))
                        .arg(&0i32)
                        .launch(ops::flat(rows * 512))?;
                }
                s.memcpy_dtod(&hidden.slice(..rows * 512), &mut hidden_before_quantize)?;
                if bound.is_some() {
                    channel_ops.fake_quant(&s, &mut contracts::TensorMut(hidden_view), None)?;
                } else {
                    fp8_quantize(&s, &quantize, &mut hidden, rows, 512)?;
                }
                if let Some(bound) = &bound {
                    bound[2].run(&s)
                } else {
                    fp8_channel_launch(
                        &s,
                        (
                            function
                                .as_ref()
                                .context("direct projection function missing")?,
                            tiled,
                        ),
                        (&weights[2].2, &weights[2].3),
                        &hidden,
                        &mut output,
                        (rows, 2048, 512),
                    )
                }
            };
            run()?;
            let capture = s.begin_capture()?;
            run()?;
            let graph = capture.finish()?;
            let eager = [&input, &g, &u, &hidden, &output, &hidden_before_quantize]
                .map(|v| s.clone_dtoh(v))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let label = format!("fp8.expert0.{name}.M{rows}");
            for repeat in 0..2 {
                graph.launch()?;
                for (buffer, expected) in
                    [&input, &g, &u, &hidden, &output, &hidden_before_quantize]
                        .into_iter()
                        .zip(&eager)
                {
                    let actual = s.clone_dtoh(buffer)?;
                    gate.check(
                        &format!("{label}.graph{repeat}"),
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    );
                }
            }
            for (stage, input_index, output_index, k) in
                [(0, 0, 1, 2048), (1, 0, 2, 2048), (2, 3, 4, 512)]
            {
                let reference = fp8_projection_reference(
                    weights[stage].0,
                    &weights[stage].1,
                    &eager[input_index][..rows * k],
                    k,
                    crate::fp8::Fp8ScaleLayout::PerChannel,
                )?;
                let actual = &eager[output_index][..reference.len()];
                ensure!(
                    actual.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL {label} nonfinite stage{stage}"
                );
                let errors = actual
                    .iter()
                    .zip(&reference)
                    .filter(|(a, (value, bound))| (f64::from(a.to_f32()) - value).abs() > *bound)
                    .count();
                let first = actual
                    .iter()
                    .zip(&reference)
                    .enumerate()
                    .find(|(_, (a, (v, b)))| (f64::from(a.to_f32()) - v).abs() > *b);
                if let Some((index, (value, (expected, bound)))) = first {
                    println!(
                        "FP8_EXPERT_ERROR name={label}.stage{stage} count={errors} first={index} actual={} expected={expected:e} bound={bound:e}",
                        value.to_f32()
                    );
                }
                gate.check(&format!("{label}.stage{stage}.fp64"), errors == 0);
            }
            let activated = eager[1][..rows * 512]
                .iter()
                .zip(&eager[2])
                .map(|(g, u)| {
                    let v = g.to_f32();
                    let silu = bf16::from_f32(v / (1.0 + (-v).exp()));
                    bf16::from_f32(silu.to_f32() * u.to_f32())
                })
                .collect::<Vec<_>>();
            gate.compare(&format!("{label}.swiglu"), &eager[5], &activated, 2, 1)?;
            for (which, before, after, width) in [
                ("input", host_input, &eager[0], 2048usize),
                ("down", eager[5].clone(), &eager[3], 512),
            ] {
                let reference = before
                    .chunks_exact(width)
                    .map(crate::fp8::fake_quant_reference)
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flat_map(|(_, _, v)| v)
                    .collect::<Vec<_>>();
                gate.check(
                    &format!("{label}.quantize.{which}.bits"),
                    reference.len() == rows * width
                        && reference
                            .iter()
                            .zip(after)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                );
            }
            gate.check(
                &format!("{label}.guards"),
                [(1, 512), (2, 512), (3, 512), (4, 2048)]
                    .iter()
                    .all(|&(i, width)| {
                        eager[i][rows * width..]
                            .iter()
                            .all(|v| v.to_bits() == sentinel.to_bits())
                    }),
            );
            if let Some(previous) = &longest {
                let previous: &Vec<bf16> = previous;
                gate.check(
                    &format!("{label}.batch_identity"),
                    eager[4][..rows * 2048]
                        .iter()
                        .zip(previous)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                );
            } else {
                longest = Some(eager[4][..rows * 2048].to_vec());
            }
        }
    }
    println!(
        "FP8_PUBLISHED_EXPERT cells={} defects={} native_weight_bytes={device_weight_bytes} gpu_weight_materialization_bytes=0",
        gate.cells, gate.defects
    );
    ensure!(
        gate.defects == 0,
        "published expert defects={}",
        gate.defects
    );
    Ok(())
}

#[cfg(feature = "rocm")]
fn fp8_encoding_gate(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gate: &mut Gate,
) -> Result<()> {
    let module = cubin::module(ctx, "fp8_linear")?;
    let decode = module.load_function("fp8_dequant_bf16_per_channel")?;
    let codes = stream.clone_htod(&(0..=255u8).collect::<Vec<_>>())?;
    let scales = stream.clone_htod(&vec![bf16::ONE; 256])?;
    let mut decoded = stream.alloc_zeros::<bf16>(256)?;
    unsafe {
        stream
            .launch_builder(&decode)
            .arg(&codes)
            .arg(&scales)
            .arg(&mut decoded)
            .arg(&256i32)
            .arg(&1i32)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let actual = stream.clone_dtoh(&decoded)?;
    for (code, value) in actual.into_iter().enumerate() {
        let expected = bf16::from_f32(crate::fp8::decode_e4m3(code as u8));
        let equal = if expected.is_nan() {
            value.is_nan() && value.is_sign_negative() == expected.is_sign_negative()
        } else {
            value.to_bits() == expected.to_bits()
        };
        gate.check(&format!("fp8.decode.code{code:02x}"), equal);
    }

    let width = 269usize;
    let mut values = vec![bf16::ZERO; 4 * width];
    for code in 0..126u8 {
        let midpoint = 0.5 * (crate::fp8::decode_e4m3(code) + crate::fp8::decode_e4m3(code + 1));
        values[2 * usize::from(code)] = bf16::from_f32(midpoint);
        values[2 * usize::from(code) + 1] = bf16::from_f32(-midpoint);
    }
    values[252] = bf16::from_f32(448.0);
    values[253] = bf16::from_f32(-448.0);
    values[254] = bf16::NEG_ZERO;
    for i in 0..width {
        values[2 * width + i] = bf16::from_bits(1 + (i % 127) as u16);
        values[3 * width + i] = bf16::from_f32(
            (((i * 17) % 257) as f32 - 128.0) / 256.0 + (((i * 29) % 263) as f32 - 131.0) / 512.0,
        );
    }
    let expected = values
        .chunks_exact(width)
        .map(crate::fp8::fake_quant_reference)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flat_map(|(_, _, row)| row)
        .collect::<Vec<_>>();
    let mut input = stream.clone_htod(&values)?;
    let quantize = module.load_function("e4m3_fake_quant_bf16")?;
    unsafe {
        stream
            .launch_builder(&quantize)
            .arg(&mut input)
            .arg(&4i32)
            .arg(&(width as i32))
            .arg(&0u64)
            .arg(&0u32)
            .arg(&4u32)
            .launch(LaunchConfig {
                grid_dim: (4, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let actual = stream.clone_dtoh(&input)?;
    for row in 0..4 {
        let range = row * width..(row + 1) * width;
        gate.check(
            &format!("fp8.quantize.row{row}.rne_zero_subnormal"),
            actual[range.clone()]
                .iter()
                .zip(&expected[range])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        );
    }
    Ok(())
}

#[cfg(feature = "rocm")]
fn graph_gate(ctx: &Arc<CudaContext>, s: &Arc<CudaStream>, gate: &mut Gate) -> Result<()> {
    use crate::backend::setup::LinearShape;
    let shape = LinearShape {
        rows: 3,
        input: 259,
        output: 32,
    };
    let blas = blas::Blas::new(s.clone())?;
    let bias = Arc::new(s.clone_htod(&pattern(32, 17))?);
    blas.prepare_choice(shape, Some(bias.clone()), None)?;
    let weight = s.clone_htod(&pattern(32 * 259, 3))?;
    let input = s.clone_htod(&pattern(3 * 259, 7))?;
    let norm_weight = s.clone_htod(&pattern(32, 11))?;
    let mut linear = s.alloc_zeros::<bf16>(96)?;
    let mut output = s.alloc_zeros::<bf16>(96)?;
    let norm = cubin::module(ctx, "norm")?.load_function("rms_norm")?;
    let mut run = || -> Result<()> {
        blas.linear_with_bias(shape, Some(&bias), &weight, &input, &mut linear)?;
        unsafe {
            s.launch_builder(&norm)
                .arg(&linear)
                .arg(&norm_weight)
                .arg(&mut output)
                .arg(&3i32)
                .arg(&32i32)
                .arg(&1e-6f32)
                .launch(ops::grid(3, 256))?;
        }
        Ok(())
    };
    run()?;
    let capture = s.begin_capture()?;
    run()?;
    let graph = capture.finish()?;
    let expected = s.clone_dtoh(&output)?;
    ensure!(
        expected.iter().all(|v| v.is_finite()) && expected.iter().any(|v| v.to_f32() != 0.0),
        "RUNNER_FAIL graph reference"
    );
    drop((blas, bias, weight, input, norm_weight, linear, norm));
    graph.upload()?;
    for repeat in 0..3 {
        s.memset_zeros(&mut output)?;
        gate.check(
            "graph.missing_launch_control_rejected",
            s.clone_dtoh(&output)? != expected,
        );
        graph.launch()?;
        gate.check(
            &format!("graph.retained_library_module_allocations.{repeat}"),
            s.clone_dtoh(&output)? == expected,
        );
    }
    let other = ctx.new_stream()?;
    gate.check(
        "graph.retained_allocation_handoff_rejected",
        output.handoff(&other).is_err(),
    );
    drop(graph);
    output.handoff(&other)?;
    gate.check(
        "graph.destroy_releases_allocation",
        other.clone_dtoh(&output)? == expected,
    );
    output.handoff(s)?;
    let capture = s.begin_capture()?;
    gate.check(
        "graph.capture_allocation_rejected",
        s.alloc_zeros::<u32>(1).is_err(),
    );
    gate.check(
        "graph.capture_htod_rejected",
        s.memcpy_htod(&expected, &mut output).is_err(),
    );
    gate.check(
        "graph.capture_dtoh_rejected",
        s.clone_dtoh(&output).is_err(),
    );
    s.memset_zeros(&mut output)?;
    drop(capture);
    gate.check(
        "graph.aborted_capture_not_executed",
        s.clone_dtoh(&output)? == expected,
    );
    let capture = s.begin_capture()?;
    s.memset_zeros(&mut output)?;
    let graph = capture.finish()?;
    graph.launch()?;
    gate.check(
        "graph.capture_after_abort",
        s.clone_dtoh(&output)?.iter().all(|v| v.to_bits() == 0),
    );
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
        ctx.record_err(Err(anyhow::anyhow!("planted resource failure")));
        gate.check(
            "context.deferred_error_propagates",
            ctx.check_err().is_err(),
        );
        gate.check("context.deferred_error_consumed", ctx.check_err().is_ok());
        let table: Vec<i32> = (0..129).flat_map(|p| [p; 3]).collect();
        let table = s.clone_htod(&table)?;
        let metadata = s.clone_htod(&[23i32, 33, 56, 129, 1, 0, 0, 0])?;
        let mut active = s.clone_htod(&[-7i32; 37 * 3 + 7])?;
        let gather = cubin::module(ctx, "q38_vision_io")?.load_function("q38_text_positions")?;
        unsafe {
            s.launch_builder(&gather)
                .arg(&table)
                .arg(&mut active)
                .arg(&metadata)
                .arg(&37u32)
                .launch(ops::grid(1, 256))?;
        }
        let positions = s.clone_dtoh(&active)?;
        gate.check(
            "positions.text_prefix",
            positions[..33 * 3]
                .iter()
                .copied()
                .eq((23..56).flat_map(|p| [p; 3])),
        );
        gate.check(
            "positions.text_tail_guards",
            positions[33 * 3..].iter().all(|&p| p == -7),
        );
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
    let mut raw_input = copy::pinned_zeros::<u32>(ctx, 263)?;
    for (i, value) in raw_input.as_mut_slice()?.iter_mut().enumerate() {
        *value = ((i * 17 % 251) as u32) << 16 | (i * 29 % 257) as u32;
    }
    let mut raw_output = copy::pinned_zeros::<u32>(ctx, 263)?;
    let raw_source = s.clone_htod(&[0xa5a5_a5a5u32; 269])?;
    let raw_target = s.clone_htod(&[0x5a5a_5a5au32; 269])?;
    let (src, _src) = raw_source.device_ptr(s);
    let (dst, _dst) = raw_target.device_ptr(s);
    upload.wait(&s.record_event(None)?)?;
    {
        #[cfg(feature = "cuda")]
        use cudarc::driver::HostSlice;
        let (input, _read) = unsafe { raw_input.stream_synced_slice(&upload) };
        // Both allocations are ordered after their initialization and retained through the final fence.
        unsafe { copy::htod_async(src + 12, input, upload.as_ref())? };
    }
    s.wait(&upload.record_event(None)?)?;
    unsafe {
        copy::dtod_raw(dst + 12, src + 12, 263 * 4, s.as_ref())?;
        copy::memset_bytes(dst + 12 + 7 * 4, 0xab, 7, s.as_ref())?;
        copy::memset_words(dst + 12 + 19 * 4, 0x1122_3344, 5, s.as_ref())?;
    }
    let completed = s.record_event(None)?;
    upload.wait(&completed)?;
    {
        #[cfg(feature = "cuda")]
        use cudarc::driver::HostSlice;
        let (output, _write) = unsafe { raw_output.stream_synced_mut_slice(&upload) };
        // The read follows the compute event and the pinned output is unread until synchronization.
        unsafe { copy::dtoh_async(output, dst + 12, upload.as_ref())? };
    }
    #[cfg(feature = "cuda")]
    upload.synchronize()?;
    #[cfg(feature = "rocm")]
    let syncs_before = ctx
        .device_synchronizations
        .load(std::sync::atomic::Ordering::Relaxed);
    let mut raw_expected = raw_input
        .as_slice()?
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    raw_expected[28..35].fill(0xab);
    raw_expected[76..96].copy_from_slice(&0x1122_3344u32.to_le_bytes().repeat(5));
    let raw_actual = raw_output
        .as_slice()?
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    gate.check("copy.raw_fenced_mailbox", raw_actual == raw_expected);
    gate.check("copy.raw_event_complete", completed.try_is_complete()?);
    #[cfg(feature = "rocm")]
    {
        raw_output.as_mut_slice()?;
        gate.check(
            "copy.pinned_access_buffer_event_only",
            ctx.device_synchronizations
                .load(std::sync::atomic::Ordering::Relaxed)
                == syncs_before,
        );
    }
    raw_expected[76] ^= 1;
    gate.check("copy.raw_wrong_word_control", raw_actual != raw_expected);
    let guarded = s.clone_dtoh(&raw_target)?;
    gate.check(
        "copy.raw_guards",
        guarded[..3]
            .iter()
            .chain(&guarded[266..])
            .all(|&v| v == 0x5a5a_5a5a),
    );
    #[cfg(feature = "rocm")]
    gate.check(
        "copy.raw_safe_api_still_rejects_foreign_owner",
        upload.clone_dtoh(&raw_source).is_err(),
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
    let masks: Vec<_> = (0..rows)
        .map(|row| u64::MAX >> (63 - row.min(63)))
        .collect();
    masked_attention_reference(q, k, v, start, capacity, &masks)
}

fn masked_attention_reference(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    start: usize,
    capacity: usize,
    masks: &[u64],
) -> Vec<bf16> {
    let rows = masks.len();
    let mut out = vec![bf16::ZERO; rows * 2048];
    for row in 0..rows {
        for head in 0..16 {
            let kh = head / 8;
            let positions: Vec<_> = (0..start + rows)
                .filter(|&pos| pos < start || masks[row] & (1u64 << (pos - start)) != 0)
                .collect();
            let scores: Vec<_> = positions
                .iter()
                .copied()
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
                    .map(|(j, &p)| {
                        p * f64::from(v[(kh * capacity + positions[j]) * 128 + d].to_f32())
                    })
                    .sum::<f64>()
                    / sum;
                out[row * 2048 + head * 128 + d] = bf16::from_f32(value as f32);
            }
        }
    }
    out
}

#[cfg(feature = "rocm")]
fn tcmqa_gate(
    ctx: &Arc<CudaContext>,
    s: &Arc<CudaStream>,
    c: &Config,
    gate: &mut Gate,
) -> Result<()> {
    for (capacity, rows, start) in [
        (129usize, 1usize, 127usize),
        (129, 3, 63),
        (129, 7, 1),
        (129, 8, 64),
        (129, 16, 113),
        (129, 33, 63),
        (129, 64, 65),
        (4096, 8, 4088),
    ] {
        let q = pattern(rows * 2560, 7);
        let k = pattern(2 * capacity * 128, 9);
        let v = pattern(2 * capacity * 128, 11);
        let mut dq = s.clone_htod(&q)?;
        let dk = s.clone_htod(&k)?;
        let dv = s.clone_htod(&v)?;
        let prefix = s.clone_htod(&[start as i32])?;
        let guard = bf16::from_f32(13.0);
        let mut output = s.clone_htod(&vec![guard; (rows + 1) * 2048])?;
        for causal in [true, false] {
            let masks: Vec<_> = (0..rows)
                .map(|row| {
                    if causal {
                        u64::MAX >> (63 - row)
                    } else {
                        1 | (1u64 << row) | (1u64 << (row / 2))
                    }
                })
                .collect();
            let expected = masked_attention_reference(&q, &k, &v, start, capacity, &masks);
            let shape = verification::Shape {
                rows,
                start,
                q_heads: 16,
                kv_heads: 2,
                capacity,
                head_dim: 128,
                causal,
                window_left: None,
            };
            let mut mq = verification::Verification::with_plan(
                ctx,
                s,
                c,
                MultiShape {
                    rows,
                    capacity,
                    causal,
                },
                MultiPlan {
                    implementation: MultiImpl::Tcmqa,
                    launch: AttentionPlan {
                        chunk: 64,
                        qpack: 8,
                        threads: 128,
                        merge_threads: 256,
                    },
                },
            )?;
            let mut ancestors = s.clone_htod(&masks)?;
            if causal {
                mq.run(&dq, &dk, &dv, &mut output, &prefix, shape)?;
            } else {
                mq.run_tree(
                    verification::Query {
                        qkv: &dq,
                        k: &dk,
                        v: &dv,
                        out: &mut output,
                        prefix: &prefix,
                        shape,
                    },
                    &ancestors,
                )?;
            }
            gate.compare(
                &format!("tcmqa.{capacity}.{rows}.{start}.causal{causal}"),
                &s.clone_dtoh(&output.slice(..rows * 2048))?,
                &expected,
                2,
                capacity + 128,
            )?;
            gate.check(
                "tcmqa.output_guard",
                s.clone_dtoh(&output.slice(rows * 2048..))?
                    .iter()
                    .all(|&x| x == guard),
            );
            if !causal {
                let same_query: Vec<_> = q[..2560]
                    .iter()
                    .copied()
                    .cycle()
                    .take(rows * 2560)
                    .collect();
                s.memcpy_htod(&same_query, &mut dq)?;
                s.memcpy_htod(&vec![1u64; rows], &mut ancestors)?;
                mq.run_tree(
                    verification::Query {
                        qkv: &dq,
                        k: &dk,
                        v: &dv,
                        out: &mut output,
                        prefix: &prefix,
                        shape,
                    },
                    &ancestors,
                )?;
                let actual = s.clone_dtoh(&output.slice(..rows * 2048))?;
                gate.check(
                    "tcmqa.batch_position_identity",
                    actual
                        .as_chunks::<2048>()
                        .0
                        .iter()
                        .all(|row| row == &actual[..2048]),
                );
                s.memcpy_htod(&q, &mut dq)?;
                if rows == 7 {
                    let mut mutant_v = v.clone();
                    for head in 0..2 {
                        mutant_v
                            [(head * capacity + start) * 128..(head * capacity + start + 1) * 128]
                            .fill(bf16::from_f32(16.0));
                    }
                    let mutant_v = s.clone_htod(&mutant_v)?;
                    let reference = masked_attention_reference(
                        &q,
                        &k,
                        &s.clone_dtoh(&mutant_v)?,
                        start,
                        capacity,
                        &vec![1; rows],
                    );
                    s.memcpy_htod(&vec![2u64; rows], &mut ancestors)?;
                    mq.run_tree(
                        verification::Query {
                            qkv: &dq,
                            k: &dk,
                            v: &mutant_v,
                            out: &mut output,
                            prefix: &prefix,
                            shape,
                        },
                        &ancestors,
                    )?;
                    let actual = s.clone_dtoh(&output.slice(..rows * 2048))?;
                    gate.check(
                        "tcmqa.ancestor_index_mutant_rejected",
                        actual
                            .iter()
                            .zip(reference)
                            .any(|(a, b)| (a.to_f32() - b.to_f32()).abs() > 1.0),
                    );
                }
            }
        }
    }
    Ok(())
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
            let choice = library.prepare_choice(shape, db.clone(), None)?;
            let frozen: blas::Choice = serde_json::from_slice(&serde_json::to_vec(&choice)?)?;
            let rebound = blas::Blas::new(s.clone())?;
            gate.check(
                "blas.identity_rebind",
                rebound.prepare_choice(shape, db.clone(), Some(&frozen))? == choice,
            );
            println!(
                "HIP_BLAS_CHOICE {} cached=true measurements=0 tuned=false",
                serde_json::to_string(&choice)?
            );
            rebound.linear_with_bias(shape, db.as_deref(), &dw, &dx, &mut dy)?;
            if !bias {
                let original = s.clone_dtoh(&dy)?;
                let weight =
                    contracts::view(&dw, s, 0, [output as u32, input as u32], [input as u64, 1])?;
                let x = contracts::view(&dx, s, 0, [rows as u32, input as u32], [input as u64, 1])?;
                let mut y = contracts::TensorMut(contracts::view(
                    &dy,
                    s,
                    0,
                    [rows as u32, output as u32],
                    [output as u64, 1],
                )?);
                rebound.linear_typed_bound(
                    crate::op_contract::ExecutionPhase::Prefill,
                    linear::LinearWeightRef::Bf16(weight),
                    x,
                    &mut y,
                    0,
                )?;
                gate.check(
                    "blas.typed_library_identity",
                    original == s.clone_dtoh(&dy)?,
                );
            }
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
                let prepared = rebound.prepare_choice(shape, Some(second.clone()), Some(&frozen));
                gate.check("blas.second_bias_same_shape_prepared", prepared.is_ok());
                let unbiased = rebound.prepare_choice(shape, None, None);
                gate.check("blas.unbiased_same_shape_prepared", unbiased.is_ok());
                if prepared.is_ok() && unbiased.is_ok() {
                    for (label, binding, values) in [
                        ("second", Some(second.as_ref()), Some(&second_bias[..])),
                        ("unbiased", None, None),
                        ("first", db.as_deref(), Some(&b[..])),
                    ] {
                        rebound.linear_with_bias(shape, binding, &dw, &dx, &mut dy)?;
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
                        .linear_with_bias(shape, Some(&unprepared), &dw, &dx, &mut dy)
                        .is_err(),
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "rocm")]
fn wmma_cells(info: &crate::backend::setup::DeviceInfo) -> bool {
    use crate::backend::setup::DeviceArchitecture;
    info.warp == 32
        && matches!(&info.arch, DeviceArchitecture::Hip(gfx) if gfx.starts_with("gfx11"))
}

#[cfg(feature = "rocm")]
#[test]
fn dense_cells_exclude_unsupported_wmma() {
    use crate::backend::setup::{DeviceArchitecture, DeviceInfo};
    let mut info = DeviceInfo {
        name: "gate fixture".into(),
        arch: DeviceArchitecture::Hip("gfx1100".into()),
        sms: 48,
        l2_bytes: 6 * 1024 * 1024,
        warp: 32,
        max_threads: 1024,
        shared_bytes: 65536,
        sm_shared_bytes: 65536,
        cooperative: true,
        optin_shared_bytes: 65536,
        memory_bytes_per_second: 1.0,
    };
    assert!(wmma_cells(&info));
    for gfx in ["gfx1030", "gfx1200"] {
        info.arch = DeviceArchitecture::Hip(gfx.into());
        assert!(
            !wmma_cells(&info),
            "unsupported WMMA cells selected for {gfx}"
        );
    }
    info.arch = DeviceArchitecture::Hip("gfx1100".into());
    info.warp = 64;
    assert!(!wmma_cells(&info));
}

#[cfg(feature = "rocm")]
fn wmma_gate(s: &Arc<CudaStream>, gate: &mut Gate) -> Result<()> {
    use crate::backend::setup::LinearShape;
    let blas = blas::Blas::new(s.clone())?;
    s.context().require_wmma()?;
    for (output, input, padding) in [
        (19usize, 259usize, 0usize),
        (23, 269, 8),
        (2560, 2048, 0),
        (2048, 2048, 0),
        (12288, 2048, 0),
        (2048, 6144, 0),
        (130560, 2048, 0),
    ] {
        let stride = input + padding;
        for period in [257, 263] {
            ensure!(
                input % period != 0 && stride % period != 0 && output % period != 0,
                "RUNNER_FAIL WMMA fixture period divides a matrix stride"
            );
        }
        let weights = pattern(output * stride, 137);
        let dw = s.clone_htod(&weights)?;
        for rows in [1usize, 3, 7, 8, 15, 16] {
            let shape = LinearShape {
                rows,
                output,
                input,
            };
            let mut values = pattern(rows * input, 173);
            let sparse = output >= 2048;
            if sparse {
                for (i, value) in values.iter_mut().enumerate() {
                    let row = i / input;
                    let k = i % input;
                    if !(0..7).any(|j| (row * 263 + j * 257 + 17) % input == k) {
                        *value = bf16::ZERO;
                    }
                }
            }
            let dx = s.clone_htod(&values)?;
            let bias = pattern(output, 181);
            let db = s.clone_htod(&bias)?;
            let expected: Vec<_> = (0..rows)
                .flat_map(|row| {
                    let terms: Vec<_> = if sparse {
                        (0..7).map(|j| (row * 263 + j * 257 + 17) % input).collect()
                    } else {
                        (0..input).collect()
                    };
                    let w = &weights;
                    let x = &values;
                    let b = &bias;
                    (0..output).map(move |o| {
                        let sum: f64 = terms
                            .iter()
                            .map(|&k| {
                                f64::from(w[o * stride + k].to_f32())
                                    * f64::from(x[row * input + k].to_f32())
                            })
                            .sum();
                        bf16::from_f32((sum + f64::from(b[o].to_f32())) as f32)
                    })
                })
                .collect();
            let canary = bf16::from_bits(0x42a5);
            let mut dy = s.clone_htod(&vec![canary; rows * output + 34])?;
            blas.wmma(
                (shape, stride),
                Some(&db),
                &dw,
                &dx,
                &mut dy.slice_mut(17..17 + rows * output),
            )?;
            let actual = s.clone_dtoh(&dy)?;
            gate.check(
                "wmma.output_canaries",
                actual[..17]
                    .iter()
                    .chain(&actual[17 + rows * output..])
                    .all(|v| v.to_bits() == canary.to_bits()),
            );
            gate.compare(
                &format!("wmma.rows{rows}.W[{output},{stride}].K{input}"),
                &actual[17..17 + rows * output],
                &expected,
                2,
                input,
            )?;
            if padding > 0 && rows == 7 {
                blas.wmma(
                    (shape, stride - 1),
                    Some(&db),
                    &dw,
                    &dx,
                    &mut dy.slice_mut(17..17 + rows * output),
                )?;
                let mutant = s.clone_dtoh(&dy)?;
                let error = mutant[17..17 + rows * output]
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| {
                        f64::from((a.to_f32() - b.to_f32()).abs())
                            / (1.0 + f64::from(b.to_f32().abs()))
                    })
                    .fold(0.0f64, f64::max);
                gate.check(
                    "mutant.wmma_weight_stride_rejected",
                    error > crate::backend::cuda::bf16_chain_bound(2, &[input]) + 1e-5,
                );
            }
        }
    }
    let shape = LinearShape {
        rows: 3,
        output: 5,
        input: 128,
    };
    let mut weights = vec![bf16::ZERO; shape.output * shape.input];
    for (o, first, second) in [
        (0, 1.0, 0.00390625),
        (1, 1.0, 0.01171875),
        (2, f32::INFINITY, 0.0),
        (3, f32::NAN, 0.0),
        (4, f32::NEG_INFINITY, 0.0),
    ] {
        weights[o * shape.input] = bf16::from_f32(first);
        weights[o * shape.input + 1] = bf16::from_f32(second);
    }
    let mut input = vec![bf16::ZERO; shape.rows * shape.input];
    for row in 0..shape.rows {
        input[row * shape.input..row * shape.input + 2].fill(bf16::ONE);
    }
    let dw = s.clone_htod(&weights)?;
    let dx = s.clone_htod(&input)?;
    let db = s.clone_htod(&vec![bf16::from_f32(0.00390625); shape.output])?;
    let mut dy = s.alloc_zeros::<bf16>(shape.rows * shape.output)?;
    for bias in [None, Some(&db)] {
        blas.wmma((shape, shape.input), bias, &dw, &dx, &mut dy)?;
        for (i, actual) in s.clone_dtoh(&dy)?.iter().enumerate() {
            let o = i % shape.output;
            let expected = bf16::from_f32(
                weights[o * shape.input].to_f32()
                    + weights[o * shape.input + 1].to_f32()
                    + if bias.is_some() { 0.00390625 } else { 0.0 },
            );
            gate.check(
                &format!("wmma.tie_nan_inf.bias{}.element{i}", bias.is_some()),
                if expected.is_nan() {
                    actual.is_nan()
                } else {
                    actual.to_bits() == expected.to_bits()
                },
            );
        }
    }
    Ok(())
}

#[cfg(feature = "rocm")]
fn markov_gate(ctx: &Arc<CudaContext>, s: &Arc<CudaStream>, gate: &mut Gate) -> Result<()> {
    let module = cubin::module(ctx, "markov")?;
    let first = [1, 8, 64].map(|p| module.load_function(&format!("markov2_top4_phase1_p{p}")));
    let first = first.into_iter().collect::<Result<Vec<_>>>()?;
    let second = module.load_function("markov2_top4_phase2")?;
    let coefficient = |index: usize, period: usize| {
        let v = index * 31 % period;
        (if v.is_multiple_of(2) { 1.0 } else { -1.0 }) * 2f32.powi((v % 5) as i32 - 4)
    };
    let e: Vec<_> = (0..256)
        .map(|k| ((k * 17 % 257) as f32 - 128.0) / 256.0)
        .collect();
    let g: Vec<_> = (0..256)
        .map(|k| ((k * 29 % 263) as f32 - 131.0) / 256.0)
        .collect();
    let dot: f64 = e
        .iter()
        .zip(&g)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    for vocab in [263usize, 130560] {
        let w1: Vec<_> = (0..vocab * 256)
            .map(|i| bf16::from_f32(coefficient(i / 256, 257) * e[i % 256]))
            .collect();
        let w2: Vec<_> = (0..vocab * 256)
            .map(|i| bf16::from_f32(coefficient(i / 256, 263) * g[i % 256]))
            .collect();
        let base: Vec<_> = (0..7 * vocab)
            .map(|i| {
                bf16::from_f32(
                    ((i % vocab * 47 + i / vocab * 31) % 269) as f32 / 32.0 - 134.0 / 32.0,
                )
            })
            .collect();
        let dw1 = s.clone_htod(&w1)?;
        let dw2 = s.clone_htod(&w2)?;
        let db = s.clone_htod(&base)?;
        let mut identities = std::collections::BTreeMap::new();
        for rounding in [0i32, 1] {
            for p in [1usize, 3, 8, 17, 64] {
                let requests: Vec<_> = (0..p)
                    .map(|j| {
                        let id = (j * 3 + p) % 7;
                        ((id * 5 % 7) as i32, ((id * 1009 + 257) % vocab) as u32)
                    })
                    .collect();
                let tokens = s.clone_htod(&requests.iter().map(|r| r.1).collect::<Vec<_>>())?;
                let rows = s.clone_htod(&requests.iter().map(|r| r.0).collect::<Vec<_>>())?;
                let (index, pt) = if p == 1 {
                    (0, 1)
                } else if p <= 8 {
                    (1, 8)
                } else {
                    (2, 64)
                };
                let mut partial_top = s.alloc_zeros::<f32>(vocab.div_ceil(256) * pt * 8)?;
                let mut partial_lse = s.alloc_zeros::<f32>(vocab.div_ceil(256) * pt * 2)?;
                let mut top = s.alloc_zeros::<u32>(p * 4)?;
                let mut logp = s.alloc_zeros::<f32>(p * 4)?;
                let mut lse = s.alloc_zeros::<f32>(p)?;
                unsafe {
                    s.launch_builder(&first[index])
                        .arg(&db)
                        .arg(&dw1)
                        .arg(&dw2)
                        .arg(&tokens)
                        .arg(&rows)
                        .arg(&mut partial_top)
                        .arg(&mut partial_lse)
                        .arg(&(vocab as i32))
                        .arg(&256i32)
                        .arg(&(p as i32))
                        .arg(&rounding)
                        .launch(ops::grid(vocab.div_ceil(256), 256))?;
                    s.launch_builder(&second)
                        .arg(&partial_top)
                        .arg(&partial_lse)
                        .arg(&mut top)
                        .arg(&mut logp)
                        .arg(&mut lse)
                        .arg(&(vocab as i32))
                        .arg(&(p as i32))
                        .arg(&(pt as i32))
                        .launch(ops::grid(p, 256))?;
                }
                let top = s.clone_dtoh(&top)?;
                let logp = s.clone_dtoh(&logp)?;
                let lse = s.clone_dtoh(&lse)?;
                ensure!(
                    logp.iter().chain(&lse).all(|v| v.is_finite()),
                    "RUNNER_FAIL Markov nonfinite output"
                );
                for (j, &(row, token)) in requests.iter().enumerate() {
                    let scores: Vec<_> = (0..vocab)
                        .map(|v| {
                            let bias = bf16::from_f32(
                                (dot * f64::from(coefficient(token as usize, 257))
                                    * f64::from(coefficient(v, 263)))
                                    as f32,
                            )
                            .to_f32();
                            let score = bias + base[row as usize * vocab + v].to_f32();
                            if rounding == 0 {
                                bf16::from_f32(score).to_f32()
                            } else {
                                score
                            }
                        })
                        .collect();
                    let mut order: Vec<_> = (0..vocab as u32).collect();
                    order.sort_unstable_by(|&a, &b| {
                        scores[b as usize]
                            .total_cmp(&scores[a as usize])
                            .then(a.cmp(&b))
                    });
                    let mx = f64::from(scores[order[0] as usize]);
                    let expected_lse = mx
                        + scores
                            .iter()
                            .map(|&x| (f64::from(x) - mx).exp())
                            .sum::<f64>()
                            .ln();
                    let error = (f64::from(lse[j]) - expected_lse).abs().max(
                        (0..4)
                            .map(|k| {
                                (f64::from(logp[j * 4 + k])
                                    - (f64::from(scores[order[k] as usize]) - expected_lse))
                                    .abs()
                            })
                            .fold(0.0, f64::max),
                    );
                    let name = format!("markov.v{vocab}.p{p}.round{rounding}.row{j}");
                    println!(
                        "MARKOV_ERROR name={name} lse_logp_max={error:e} bound=1e-4 margin4_5={}",
                        scores[order[3] as usize] - scores[order[4] as usize]
                    );
                    gate.check(&name, top[j * 4..j * 4 + 4] == order[..4] && error <= 1e-4);
                    let bits: Vec<_> = top[j * 4..j * 4 + 4]
                        .iter()
                        .copied()
                        .chain(logp[j * 4..j * 4 + 4].iter().map(|x| x.to_bits()))
                        .chain([lse[j].to_bits()])
                        .collect();
                    let key = (rounding, row, token);
                    if let Some(previous) = identities.insert(key, bits.clone()) {
                        gate.check("markov.batch_position_identity", previous == bits);
                    }
                }
                if vocab == 263 && p == 3 && rounding == 0 {
                    let wrong_rows =
                        s.clone_htod(&requests.iter().map(|r| (r.0 + 1) % 7).collect::<Vec<_>>())?;
                    let mut mutant_top = s.alloc_zeros::<u32>(p * 4)?;
                    let mut mutant_logp = s.alloc_zeros::<f32>(p * 4)?;
                    let mut mutant_lse = s.alloc_zeros::<f32>(p)?;
                    unsafe {
                        s.launch_builder(&first[index])
                            .arg(&db)
                            .arg(&dw1)
                            .arg(&dw2)
                            .arg(&tokens)
                            .arg(&wrong_rows)
                            .arg(&mut partial_top)
                            .arg(&mut partial_lse)
                            .arg(&(vocab as i32))
                            .arg(&256i32)
                            .arg(&(p as i32))
                            .arg(&rounding)
                            .launch(ops::grid(vocab.div_ceil(256), 256))?;
                        s.launch_builder(&second)
                            .arg(&partial_top)
                            .arg(&partial_lse)
                            .arg(&mut mutant_top)
                            .arg(&mut mutant_logp)
                            .arg(&mut mutant_lse)
                            .arg(&(vocab as i32))
                            .arg(&(p as i32))
                            .arg(&(pt as i32))
                            .launch(ops::grid(p, 256))?;
                    }
                    gate.check(
                        "markov.request_row_mutant_rejected",
                        s.clone_dtoh(&mutant_top)? != top
                            || s.clone_dtoh(&mutant_lse)?
                                .iter()
                                .zip(&lse)
                                .any(|(a, b)| (a - b).abs() > 1e-4),
                    );
                }
            }
        }
    }
    Ok(())
}
