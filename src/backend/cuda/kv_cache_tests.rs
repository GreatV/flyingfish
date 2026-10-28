use super::state::{CacheView, StateCheckpoint, StateStore};
use super::{
    Device, cubin,
    decode::{Attention, GqaDecodeInput, SlidingWindow},
    flash::{Flash, Shape},
    ops::flat,
};
use crate::{
    backend::setup::AttentionPlan, kv::KvFormat, op_contract::AttentionWorkspace,
    prefix_state::sha256,
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use half::bf16;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

struct CommittedKv {
    prefix: usize,
    layout: crate::kv::KvLayout,
    live: [Vec<bf16>; 2],
    codes: [Vec<u8>; 2],
    stats: [Vec<f32>; 2],
}

#[test]
#[ignore = "GPU window required; head-major KV ring staging"]
fn ring_copy_geometry_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let source_capacity = 513;
    let target_capacity = 1025;
    let rows = 511;
    let sentinel = bf16::from_bits(0x7fc1);
    for (heads, head_dim) in [(4, 128), (1, 256), (2, 256), (8, 256), (1, 512), (8, 512)] {
        let host: Vec<_> = (0..heads * source_capacity * head_dim)
            .map(|i| bf16::from_f32(((i * 37 + 11) % 1009) as f32 / 256.0 - 2.0))
            .collect();
        let source = s.clone_htod(&host)?;
        let empty = vec![sentinel; heads * target_capacity * head_dim + 32];
        for (source_start, target_start, wrap_source, wrap_target) in [
            (2, 17, false, false),
            (503, 0, true, false),
            (2, 1019, false, true),
            (503, 1019, true, true),
        ] {
            let mut expected = empty.clone();
            for head in 0..heads {
                for row in 0..rows {
                    let src = (head * source_capacity + (source_start + row) % source_capacity)
                        * head_dim;
                    let dst = (head * target_capacity + (target_start + row) % target_capacity)
                        * head_dim;
                    expected[dst..dst + head_dim].copy_from_slice(&host[src..src + head_dim]);
                }
            }
            for graph in [false, true] {
                let mut target = s.clone_htod(&empty)?;
                if graph {
                    let (src, dst) = {
                        let (src, _src) = source.device_ptr(s);
                        let (dst, _dst) = target.device_ptr(s);
                        (src, dst)
                    };
                    s.synchronize()?;
                    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let result = super::copy::RingCopy {
                        stream: s,
                        source: &source,
                        target: &mut target,
                        source_capacity,
                        target_capacity,
                        heads,
                        head_dim,
                    }
                    .run_raw(super::copy::RingCopyCall {
                        src,
                        dst,
                        source_start,
                        target_start,
                        rows,
                        wrap_source,
                        wrap_target,
                    });
                    let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                    result?;
                    let captured = captured?.context("ring staging graph is empty")?;
                    captured.launch()?;
                    captured.launch()?;
                } else {
                    super::copy::RingCopy {
                        stream: s,
                        source: &source,
                        target: &mut target,
                        source_capacity,
                        target_capacity,
                        heads,
                        head_dim,
                    }
                    .run(
                        source_start,
                        target_start,
                        rows,
                        wrap_source,
                        wrap_target,
                    )?;
                }
                let actual = s.clone_dtoh(&target)?;
                ensure!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "ring staging differs heads={heads} dim={head_dim} source_start={source_start} target_start={target_start} graph={graph}"
                );
            }
        }
        let mut wrong = s.clone_htod(&empty)?;
        super::copy::RingCopy {
            stream: s,
            source: &source,
            target: &mut wrong,
            source_capacity,
            target_capacity,
            heads,
            head_dim,
        }
        .run(504, 0, rows, true, false)?;
        let wrong = s.clone_dtoh(&wrong)?;
        ensure!(
            wrong[..head_dim]
                .iter()
                .zip(&host[503 * head_dim..504 * head_dim])
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "shifted ring source control did not discriminate"
        );
        eprintln!(
            "PASS ring_copy_geometry heads={heads} dim={head_dim} rows={rows} source_capacity={source_capacity} target_capacity={target_capacity} graph_eager_exact=true shifted_source_rejected=true"
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU window required; D128 bidirectional and prefix-LM masks"]
fn prefill_sequence_masks_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let same_bits = |a: &[bf16], b: &[bf16]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    for (hq, hkv, rows, prefix) in [(16, 8, 2048, None), (12, 12, 130, Some(65))] {
        let width = hq * 128;
        let pitch = width + 512;
        let capacity = rows + 64;
        let mut host_q = vec![bf16::NAN; rows * pitch];
        for row in 0..rows {
            for col in 0..width {
                host_q[row * pitch + col] =
                    bf16::from_f32(((row * 37 + col * 53 + 11) % 1009) as f32 / 512.0 - 1.0);
            }
        }
        let mut host_k = vec![bf16::NAN; hkv * capacity * 128];
        let mut host_v = host_k.clone();
        for head in 0..hkv {
            for row in 0..rows {
                for col in 0..128 {
                    let i = (head * capacity + row) * 128 + col;
                    host_k[i] = bf16::from_f32(((i * 17 + 23) % 1013) as f32 / 512.0 - 1.0);
                    host_v[i] = bf16::from_f32(((i * 29 + 7) % 1009) as f32 / 512.0 - 1.0);
                }
            }
        }
        let q = s.clone_htod(&host_q)?;
        let k = s.clone_htod(&host_k)?;
        let mut v = s.clone_htod(&host_v)?;
        let mut output = s.alloc_zeros::<bf16>(rows * width)?;
        let mut flash = Flash::new(s.clone(), rows, hq, hkv, 128)?;
        let shape = Shape {
            rows,
            start: 0,
            q_heads: hq,
            kv_heads: hkv,
            capacity,
            head_dim: 128,
            causal: false,
            window_left: None,
        };
        let mut execute = |values: &CudaSlice<bf16>, causal: bool| -> Result<Vec<bf16>> {
            flash.run_masked(super::flash::Query {
                q: &q,
                q_row_stride: pitch,
                k: &k,
                v: values,
                out: &mut output,
                shape: Shape { causal, ..shape },
                keys: rows,
                prefix: if causal { None } else { prefix },
            })?;
            let values = s.clone_dtoh(&output)?;
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite masked prefill"
            );
            Ok(values)
        };
        let baseline = execute(&v, false)?;
        let causal_baseline = execute(&v, true)?;
        let last = prefix.unwrap_or(rows) - 1;
        let mut changed = host_v.clone();
        for head in 0..hkv {
            changed[(head * capacity + last) * 128..(head * capacity + last + 1) * 128]
                .fill(bf16::from_f32(64.0));
        }
        s.memcpy_htod(&changed, &mut v)?;
        let legal = execute(&v, false)?;
        let causal_control = execute(&v, true)?;
        ensure!(
            !same_bits(&legal[..width], &baseline[..width]),
            "legal future prefix key omitted"
        );
        ensure!(
            same_bits(&causal_control[..width], &causal_baseline[..width]),
            "causal control read future key"
        );
        if let Some(p) = prefix {
            let mut changed = host_v.clone();
            for head in 0..hkv {
                changed[(head * capacity + p + 1) * 128..(head * capacity + p + 2) * 128]
                    .fill(bf16::from_f32(64.0));
            }
            s.memcpy_htod(&changed, &mut v)?;
            let future = execute(&v, false)?;
            ensure!(
                same_bits(&future[..(p + 1) * width], &baseline[..(p + 1) * width]),
                "prefix/causal suffix read future key"
            );
            ensure!(
                !same_bits(
                    &future[(p + 1) * width..(p + 2) * width],
                    &baseline[(p + 1) * width..(p + 2) * width]
                ),
                "suffix omitted current key"
            );
        }
        let copied_q = s.clone_dtoh(&q)?;
        ensure!(
            copied_q
                .iter()
                .zip(&host_q)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "pitched query or padding modified"
        );
        eprintln!(
            "PASS prefill_sequence_masks hq={hq} hkv={hkv} rows={rows} prefix={prefix:?} causal_control_rejected=true"
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU window required; HD128 Flint attention gate"]
fn gqa32_attention_window_sink_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let capacity = 1024;
    let ring = 513;
    let (hq, hkv, dim) = (32usize, 4usize, 128usize);
    let q: Vec<_> = (0..hq * dim)
        .map(|i| bf16::from_f32((((i * 71 + 13) % 251) as f32) / 64.0 - 2.0))
        .collect();
    let logical_k = |p: usize| -> Vec<bf16> {
        (0..hkv * dim)
            .map(|i| {
                let h = i / dim;
                let d = i % dim;
                bf16::from_f32((((p * 29 + h * 113 + d * 17) % 503) as f32) / 64.0 - 4.0)
            })
            .collect()
    };
    let logical_v = |p: usize| -> Vec<bf16> {
        (0..hkv * dim)
            .map(|i| {
                let h = i / dim;
                let d = i % dim;
                bf16::from_f32((((p * 43 + h * 89 + d * 31) % 503) as f32) / 64.0 - 4.0)
            })
            .collect()
    };
    let mut full_k = vec![bf16::ZERO; hkv * capacity * dim];
    let mut full_v = vec![bf16::ZERO; hkv * capacity * dim];
    for p in 0..capacity {
        for (src, dst) in [(&logical_k(p), &mut full_k), (&logical_v(p), &mut full_v)] {
            for h in 0..hkv {
                let src = &src[h * dim..(h + 1) * dim];
                let dst = &mut dst[(h * capacity + p) * dim..(h * capacity + p + 1) * dim];
                dst.copy_from_slice(src);
            }
        }
    }
    let first = 512;
    let mut window_k = vec![bf16::ZERO; hkv * ring * dim];
    let mut window_v = vec![bf16::ZERO; hkv * ring * dim];
    for logical in 0..ring {
        let p = 512 + logical;
        let physical = if p >= ring { p - ring } else { p };
        for (src, dst) in [
            (&logical_k(p), &mut window_k),
            (&logical_v(p), &mut window_v),
        ] {
            for h in 0..hkv {
                let src = &src[h * dim..(h + 1) * dim];
                let dst = &mut dst[(h * ring + physical) * dim..(h * ring + physical + 1) * dim];
                dst.copy_from_slice(src);
            }
        }
    }
    let expected_window_k = logical_cache(&logical_k, first, ring);
    let expected_window_v = logical_cache(&logical_v, first, ring);
    let sinks: Vec<_> = (0..hq)
        .map(|h| bf16::from_f32(((h * 37 + 5) % 64) as f32 / 8.0 - 4.0))
        .collect();
    let negligible: Vec<_> = vec![bf16::from_f32(-100.0); hq];
    let q_dev = s.clone_htod(&q)?;
    let full_k_dev = s.clone_htod(&full_k)?;
    let full_v_dev = s.clone_htod(&full_v)?;
    let window_k_dev = s.clone_htod(&window_k)?;
    let window_v_dev = s.clone_htod(&window_v)?;
    let sinks_dev = s.clone_htod(&sinks)?;
    let negligible_dev = s.clone_htod(&negligible)?;
    let length = s.clone_htod(&[capacity as i32])?;
    let window_length = s.clone_htod(&[ring as i32])?;
    let window_position = s.clone_htod(&[1024i32])?;
    device.finish_upload()?;
    let plan = AttentionPlan {
        chunk: 512,
        qpack: 4,
        threads: 256,
        merge_threads: 256,
    };
    let config = crate::config::Config {
        model_type: "flint".into(),
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 1,
        num_attention_heads: hq,
        num_key_value_heads: hkv,
        head_dim: dim,
        vocab_size: 150144,
        max_position_embeddings: capacity,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_scaling: None,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        torch_dtype: "bfloat16".into(),
        eos_token_id: vec![150001],
        attention_bias: false,
        mlp_bias: false,
    };
    let mut attention = Attention::new(&device.ctx, s, &config, capacity, plan)?;

    let expected_attention = |k: &[bf16], v: &[bf16], cache_len: usize, sinks: &[bf16]| {
        (0..hq * dim)
            .map(|i| {
                let h = i / dim;
                let d = i % dim;
                let kv = h / (hq / hkv);
                let mut scores = vec![0f32; cache_len];
                for (p, score) in scores.iter_mut().enumerate() {
                    *score = (0..dim)
                        .map(|j| {
                            q[h * dim + j].to_f32() * k[(kv * cache_len + p) * dim + j].to_f32()
                        })
                        .sum::<f32>()
                        * (dim as f32).sqrt().recip();
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sink = sinks[h].to_f32();
                let shift = max.max(sink);
                let mut denominator = (sink - shift).exp();
                for score in &scores {
                    denominator += (score - shift).exp();
                }
                let mut numerator = 0f32;
                for p in 0..cache_len {
                    let probability = ((scores[p] - shift).exp() / denominator) as f32;
                    numerator += probability * v[(kv * cache_len + p) * dim + d].to_f32();
                }
                bf16::from_f32(numerator).to_f32()
            })
            .collect::<Vec<_>>()
    };

    let mut normal = s.alloc_zeros::<bf16>(hq * dim)?;
    let mut sink_output = s.alloc_zeros::<bf16>(hq * dim)?;
    let mut negligible_output = s.alloc_zeros::<bf16>(hq * dim)?;
    let mut window_output = s.alloc_zeros::<bf16>(hq * dim)?;
    attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &full_k_dev,
            v: &full_v_dev,
            output: &mut normal,
            output_offset: 0,
            length: &length,
            sinks: None,
            window: None,
        },
    )?;
    attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &full_k_dev,
            v: &full_v_dev,
            output: &mut negligible_output,
            output_offset: 0,
            length: &length,
            sinks: Some(&negligible_dev),
            window: None,
        },
    )?;
    attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &full_k_dev,
            v: &full_v_dev,
            output: &mut sink_output,
            output_offset: 0,
            length: &length,
            sinks: Some(&sinks_dev),
            window: None,
        },
    )?;
    attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &window_k_dev,
            v: &window_v_dev,
            output: &mut window_output,
            output_offset: 0,
            length: &window_length,
            sinks: Some(&sinks_dev),
            window: Some(SlidingWindow {
                position: &window_position,
                head_stride: (ring * 128) as i32,
            }),
        },
    )?;
    s.synchronize()?;

    let normal = s.clone_dtoh(&normal)?;
    let negligible = s.clone_dtoh(&negligible_output)?;
    let sink = s.clone_dtoh(&sink_output)?;
    let window = s.clone_dtoh(&window_output)?;
    ensure!(
        normal
            .iter()
            .zip(&negligible)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "negligible sink changed ordinary attention output"
    );
    let full_expected = expected_attention(&full_k, &full_v, capacity, &sinks);
    let window_expected = expected_attention(&expected_window_k, &expected_window_v, ring, &sinks);
    let bound = super::bf16_chain_bound(3, &[dim, ring + 1]);
    for (name, actual, expected) in [
        ("full", &sink, &full_expected),
        ("window", &window, &window_expected),
    ] {
        let magnitude = expected
            .iter()
            .map(|value| value.abs())
            .fold(0f32, f32::max);
        let max = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a.to_f32() - b).abs())
            .fold(0f32, f32::max);
        if let Some((index, value, wanted)) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (a, b))| (a.to_f32() - *b).abs() > bound as f32 * magnitude)
            .map(|(index, (a, b))| (index, a.to_f32(), *b))
        {
            eprintln!(
                "{name} first_bad index={index} value={value:e} expected={wanted:e} ordinary={} head={} dim={}",
                normal[index].to_f32(),
                index / dim,
                index % dim
            );
        }
        ensure!(
            max <= bound as f32 * magnitude,
            "{name} attention max={max:e} bound={}",
            bound as f32 * magnitude
        );
        eprintln!(
            "GQA32_ATTENTION mode={name} max_abs={max:e} relative_bound={} PASS",
            bound as f32 * magnitude
        );
    }

    let mut r5_attention = Attention::new(
        &device.ctx,
        s,
        &config,
        capacity,
        AttentionPlan {
            chunk: 128,
            qpack: 2,
            threads: 256,
            merge_threads: 256,
        },
    )?;
    r5_attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &full_k_dev,
            v: &full_v_dev,
            output: &mut sink_output,
            output_offset: 0,
            length: &length,
            sinks: Some(&sinks_dev),
            window: None,
        },
    )?;
    r5_attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &window_k_dev,
            v: &window_v_dev,
            output: &mut window_output,
            output_offset: 0,
            length: &window_length,
            sinks: Some(&sinks_dev),
            window: Some(SlidingWindow {
                position: &window_position,
                head_stride: (ring * 128) as i32,
            }),
        },
    )?;
    s.synchronize()?;
    let r5_full = s.clone_dtoh(&sink_output)?;
    let r5_window = s.clone_dtoh(&window_output)?;
    let mut graph_length = s.clone_htod(&[capacity as i32])?;
    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    r5_attention.set_length(s, capacity as i32, &mut graph_length)?;
    r5_attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &full_k_dev,
            v: &full_v_dev,
            output: &mut sink_output,
            output_offset: 0,
            length: &graph_length,
            sinks: Some(&sinks_dev),
            window: None,
        },
    )?;
    r5_attention.set_length(s, ring as i32, &mut graph_length)?;
    r5_attention.run_with(
        s,
        GqaDecodeInput {
            q: &q_dev,
            q_offset: 0,
            k: &window_k_dev,
            v: &window_v_dev,
            output: &mut window_output,
            output_offset: 0,
            length: &graph_length,
            sinks: Some(&sinks_dev),
            window: Some(SlidingWindow {
                position: &window_position,
                head_stride: (ring * 128) as i32,
            }),
        },
    )?;
    let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
    let graph = captured?.context("GQA32 sink graph capture")?;
    graph.upload()?;
    graph.launch()?;
    s.synchronize()?;
    let graph_full = s.clone_dtoh(&sink_output)?;
    let graph_window = s.clone_dtoh(&window_output)?;
    ensure!(
        graph_full
            .iter()
            .zip(&r5_full)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
            && graph_window
                .iter()
                .zip(&r5_window)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "GQA32 chunk-128 qpack-2 sink graph differs from eager attention"
    );
    eprintln!("GQA32_SINK_GRAPH chunk=128 qpack=2 chunks=8 bitwise=true PASS");

    let boundary_sinks = vec![bf16::ZERO; hq];
    let boundary_sinks_dev = s.clone_htod(&boundary_sinks)?;
    device.finish_upload()?;
    for position in [0usize, 1, 511, 512, 513, 1024, 1025, 262_143] {
        let first = position.saturating_sub(512);
        let tokens = position - first + 1;
        let q = vec![bf16::ZERO; hq * dim];
        let k = vec![bf16::ZERO; hkv * ring * dim];
        let mut v = vec![bf16::ZERO; hkv * ring * dim];
        for logical in [first.saturating_sub(1), first, position] {
            if logical < first || logical > position {
                continue;
            }
            let physical = logical % ring;
            for head in 0..hkv {
                let index = (head * ring + physical) * dim;
                if logical + 1 == first {
                    v[index + 1] = bf16::ONE;
                } else if logical == first && position >= 512 {
                    v[index] = bf16::ONE;
                } else if logical == position {
                    v[index + 2] = bf16::ONE;
                }
            }
        }
        let q_dev = s.clone_htod(&q)?;
        let k_dev = s.clone_htod(&k)?;
        let mut v_dev = s.alloc_zeros::<bf16>(hkv * ring * dim)?;
        s.memcpy_htod(&v, &mut v_dev)?;
        let length_dev = s.clone_htod(&[tokens as i32])?;
        let position_dev = s.clone_htod(&[position as i32])?;
        let mut output = s.alloc_zeros::<bf16>(hq * dim)?;
        device.finish_upload()?;
        attention.run_with(
            s,
            GqaDecodeInput {
                q: &q_dev,
                q_offset: 0,
                k: &k_dev,
                v: &v_dev,
                output: &mut output,
                output_offset: 0,
                length: &length_dev,
                sinks: Some(&boundary_sinks_dev),
                window: Some(SlidingWindow {
                    position: &position_dev,
                    head_stride: (ring * dim) as i32,
                }),
            },
        )?;
        s.synchronize()?;
        let output = s.clone_dtoh(&output)?;
        let expected = bf16::from_f32(1.0 / (tokens as f32 + 1.0)).to_f32();
        let tolerance = 4.0 * bf16::EPSILON.to_f32() * expected;
        for head in 0..hq {
            let base = head * dim;
            ensure!(
                output[base + 1].to_bits() == bf16::ZERO.to_bits(),
                "decode ring p={position} included excluded key"
            );
            for (d, wanted) in [
                (0usize, if position >= 512 { expected } else { 0.0 }),
                (2, expected),
            ] {
                let error = (output[base + d].to_f32() - wanted).abs();
                ensure!(
                    error <= tolerance,
                    "decode ring p={position} d={d} output={:.9e} expected={wanted:e}",
                    output[base + d].to_f32(),
                );
            }
        }
        eprintln!(
            "GQA32_RING position={position} first={first} tokens={tokens} p_minus_512=visible p_minus_513=excluded PASS"
        );
    }
    Ok(())
}

fn logical_cache(logical: &dyn Fn(usize) -> Vec<bf16>, first: usize, ring: usize) -> Vec<bf16> {
    let (hkv, dim) = (4usize, 128usize);
    let mut values = vec![bf16::ZERO; hkv * ring * dim];
    for logical_pos in 0..ring {
        let p = first + logical_pos;
        let row = logical(p);
        for h in 0..hkv {
            let src = &row[h * dim..(h + 1) * dim];
            let dst =
                &mut values[(h * ring + logical_pos) * dim..(h * ring + logical_pos + 1) * dim];
            dst.copy_from_slice(src);
        }
    }
    values
}

#[derive(Clone, Copy)]
struct AttentionMetrics {
    max_abs: f32,
    p99_abs: f32,
    mean_abs: f32,
    cosine: f32,
    max_reference: f32,
}

fn attention_metrics(actual: &[bf16], expected: &[f32]) -> AttentionMetrics {
    assert_eq!(actual.len(), expected.len());
    let mut errors: Vec<_> = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual.to_f32() - expected).abs())
        .collect();
    errors.sort_by(f32::total_cmp);
    let max_abs = errors.last().copied().unwrap_or_default();
    let p99_abs = errors[(expected.len() * 99).div_ceil(100) - 1];
    let mean_abs = errors.iter().sum::<f32>() / errors.len() as f32;
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.to_f32() * expected)
        .sum::<f32>();
    let actual_norm = actual
        .iter()
        .map(|value| value.to_f32() * value.to_f32())
        .sum::<f32>()
        .sqrt();
    let expected_norm = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let cosine = if actual_norm == 0.0 && expected_norm == 0.0 {
        1.0
    } else {
        dot / (actual_norm * expected_norm)
    };
    AttentionMetrics {
        max_abs,
        p99_abs,
        mean_abs,
        cosine,
        max_reference: expected
            .iter()
            .map(|value| value.abs())
            .fold(0f32, f32::max),
    }
}

#[test]
#[ignore = "GPU window required; Flint QK norm/RoPE/NoPE gate"]
fn gqa32_qknorm_rope_nope_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let (max_rows, hq, hkv, dim, capacity, start) =
        (2048usize, 32usize, 4usize, 128usize, 2055usize, 7usize);
    let heads = hq + 2 * hkv;
    let input: Vec<_> = (0..max_rows * heads * dim)
        .map(|i| bf16::from_f32((((i * 73 + 31) % 503) as f32) / 64.0 - 4.0))
        .collect();
    let q_norm: Vec<_> = (0..dim)
        .map(|i| bf16::from_f32((((i * 37 + 3) % 127) as f32) / 64.0 - 1.0))
        .collect();
    let k_norm: Vec<_> = (0..dim)
        .map(|i| bf16::from_f32((((i * 53 + 11) % 127) as f32) / 64.0 - 1.0))
        .collect();
    let expected = |rows: usize, rope: bool| -> Vec<Vec<bf16>> {
        let mut output = input.clone();
        let mut k = vec![bf16::ZERO; hkv * capacity * dim];
        let mut v = vec![bf16::ZERO; hkv * capacity * dim];
        for row in 0..rows {
            let position = start + row;
            for head in 0..heads {
                let base = (row * heads + head) * dim;
                let weight = if head < hq { &q_norm } else { &k_norm };
                if head < hq + hkv {
                    let mut values = [0f32; 128];
                    for d in 0..dim {
                        values[d] = input[base + d].to_f32();
                    }
                    let mean = values.iter().map(|value| value * value).sum::<f32>() / dim as f32;
                    let rstd = (mean + 1e-6).sqrt().recip();
                    for d in 0..dim {
                        output[base + d] = bf16::from_f32(values[d] * rstd * weight[d].to_f32());
                    }
                    if rope {
                        for pair in 0..dim / 2 {
                            let frequency = pair as f32 * 2.0 / dim as f32;
                            let angle = position as f32 * 10_000f32.powf(-frequency);
                            let cos = bf16::from_f32(angle.cos()).to_f32();
                            let sin = bf16::from_f32(angle.sin()).to_f32();
                            let a = output[base + pair].to_f32();
                            let b = output[base + pair + dim / 2].to_f32();
                            output[base + pair] = bf16::from_f32(a * cos - b * sin);
                            output[base + pair + dim / 2] = bf16::from_f32(b * cos + a * sin);
                        }
                    }
                    if head >= hq {
                        let kv_head = head - hq;
                        let dst = (kv_head * capacity + position) * dim;
                        k[dst..dst + dim].copy_from_slice(&output[base..base + dim]);
                    }
                } else {
                    let kv_head = head - hq - hkv;
                    let dst = (kv_head * capacity + position) * dim;
                    v[dst..dst + dim].copy_from_slice(&input[base..base + dim]);
                }
            }
        }
        vec![output, k, v]
    };

    let start_dev = s.clone_htod(&[start as i32])?;
    let q_norm_dev = s.clone_htod(&q_norm)?;
    let k_norm_dev = s.clone_htod(&k_norm)?;
    let module = cubin::module(&device.ctx, "rope")?;
    let rope = module.load_function("draft_qknorm_rope_kv")?;
    let nope = module.load_function("qknorm_kv_nope")?;
    let flint_rope = module.load_function("flint_qknorm_rope_kv")?;
    let flint_nope = module.load_function("flint_qknorm_kv_nope")?;
    for (function, use_rope, name) in [(&rope, true, "rope"), (&nope, false, "nope")] {
        let reference = expected(max_rows, use_rope);
        for rows in [1usize, 2, 3, 64, 512, 513, 1024, 2048] {
            let input_dev = s.clone_htod(&input[..rows * heads * dim])?;
            let k_dev = s.alloc_zeros::<bf16>(hkv * capacity * dim)?;
            let v_dev = s.alloc_zeros::<bf16>(hkv * capacity * dim)?;
            unsafe {
                s.launch_builder(function)
                    .arg(&input_dev)
                    .arg(&k_dev)
                    .arg(&v_dev)
                    .arg(&q_norm_dev)
                    .arg(&k_norm_dev)
                    .arg(&start_dev)
                    .arg(&(rows as i32))
                    .arg(&(hq as i32))
                    .arg(&(hkv as i32))
                    .arg(&(dim as i32))
                    .arg(&(capacity as i32))
                    .arg(&0i32)
                    .arg(&10_000.0f32)
                    .arg(&1e-6f32)
                    .launch(LaunchConfig {
                        grid_dim: ((rows * heads) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            s.synchronize()?;
            let actual_output = s.clone_dtoh(&input_dev)?;
            let actual_k = s.clone_dtoh(&k_dev)?;
            let actual_v = s.clone_dtoh(&v_dev)?;
            let expected_output = &reference[0][..rows * heads * dim];
            let mut live_k = Vec::with_capacity(hkv * rows * dim);
            let mut live_v = Vec::with_capacity(hkv * rows * dim);
            let mut expected_k = Vec::with_capacity(hkv * rows * dim);
            let mut expected_v = Vec::with_capacity(hkv * rows * dim);
            for head in 0..hkv {
                for position in start..start + rows {
                    let range =
                        (head * capacity + position) * dim..(head * capacity + position + 1) * dim;
                    live_k.extend_from_slice(&actual_k[range.clone()]);
                    live_v.extend_from_slice(&actual_v[range.clone()]);
                    expected_k.extend_from_slice(&reference[1][range.clone()]);
                    expected_v.extend_from_slice(&reference[2][range.clone()]);
                }
            }
            ensure!(
                live_v
                    .iter()
                    .zip(&expected_v)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                "{name} rows={rows} V writes are not bitwise"
            );
            let output_metrics = attention_metrics(
                &actual_output,
                &expected_output
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
            );
            let k_metrics = attention_metrics(
                &live_k,
                &expected_k
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
            );
            let bound = super::bf16_chain_bound(3, &[dim]);
            let output_bound = bound as f32 * output_metrics.max_reference;
            let k_bound = bound as f32 * k_metrics.max_reference;
            for (label, metrics, max_bound) in [
                ("output", output_metrics, output_bound),
                ("k", k_metrics, k_bound),
            ] {
                ensure!(
                    metrics.max_abs <= max_bound,
                    "{name} rows={rows} {label} max={:.9e} bound={max_bound:e}",
                    metrics.max_abs
                );
                ensure!(
                    metrics.p99_abs <= max_bound / 2.0,
                    "{name} rows={rows} {label} p99={:.9e} bound={}",
                    metrics.p99_abs,
                    max_bound / 2.0
                );
                ensure!(
                    metrics.mean_abs <= max_bound / 4.0,
                    "{name} rows={rows} {label} mean={:.9e} bound={}",
                    metrics.mean_abs,
                    max_bound / 4.0
                );
                ensure!(
                    metrics.cosine >= 1.0 - bound as f32,
                    "{name} rows={rows} {label} cosine={:.9e} bound={}",
                    metrics.cosine,
                    1.0 - bound as f32
                );
            }
            eprintln!(
                "GQA32_QKNORM mode={name} rows={rows} output_max={:.9e} output_p99={:.9e} output_mean={:.9e} output_cosine={:.9e} k_max={:.9e} PASS",
                output_metrics.max_abs,
                output_metrics.p99_abs,
                output_metrics.mean_abs,
                output_metrics.cosine,
                k_metrics.max_abs
            );
        }
    }
    for (function, use_rope, name) in [
        (&flint_rope, true, "planar-rope"),
        (&flint_nope, false, "planar-nope"),
    ] {
        let rows = 3usize;
        let q_width = hq * dim;
        let kv_width = hkv * dim;
        let mut planar = Vec::with_capacity(rows * (q_width + 2 * kv_width));
        for row in 0..rows {
            let base = row * heads * dim;
            planar.extend_from_slice(&input[base..base + q_width]);
        }
        for row in 0..rows {
            let base = row * heads * dim + q_width;
            planar.extend_from_slice(&input[base..base + kv_width]);
        }
        for row in 0..rows {
            let base = row * heads * dim + q_width + kv_width;
            planar.extend_from_slice(&input[base..base + kv_width]);
        }
        let planar_dev = s.clone_htod(&planar)?;
        let k_dev = s.alloc_zeros::<bf16>(hkv * capacity * dim)?;
        let v_dev = s.alloc_zeros::<bf16>(hkv * capacity * dim)?;
        let q_stride = q_width as u64;
        let kv_stride = kv_width as u64;
        let k_offset = (rows * q_width) as u64;
        let v_offset = (rows * (q_width + kv_width)) as u64;
        unsafe {
            s.launch_builder(function)
                .arg(&planar_dev)
                .arg(&k_dev)
                .arg(&v_dev)
                .arg(&q_norm_dev)
                .arg(&k_norm_dev)
                .arg(&start_dev)
                .arg(&-1i32)
                .arg(&(rows as i32))
                .arg(&(hq as i32))
                .arg(&(hkv as i32))
                .arg(&(dim as i32))
                .arg(&(capacity as i32))
                .arg(&0i32)
                .arg(&10_000.0f32)
                .arg(&1e-6f32)
                .arg(&q_stride)
                .arg(&k_offset)
                .arg(&kv_stride)
                .arg(&v_offset)
                .arg(&kv_stride)
                .launch(LaunchConfig {
                    grid_dim: ((rows * heads) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        s.synchronize()?;
        let actual = s.clone_dtoh(&planar_dev)?;
        let reference = expected(rows, use_rope);
        let mut expected_q = Vec::with_capacity(rows * q_width);
        let mut expected_k = Vec::with_capacity(rows * kv_width);
        for row in 0..rows {
            let base = row * heads * dim;
            expected_q.extend_from_slice(&reference[0][base..base + q_width]);
            expected_k.extend_from_slice(&reference[0][base + q_width..base + q_width + kv_width]);
        }
        let expected_v = &planar[v_offset as usize..];
        let q_metrics = attention_metrics(
            &actual[..rows * q_width],
            &expected_q
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
        );
        let k_metrics = attention_metrics(
            &actual[k_offset as usize..k_offset as usize + rows * kv_width],
            &expected_k
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
        );
        let mismatch = actual[v_offset as usize..]
            .iter()
            .zip(expected_v)
            .enumerate()
            .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits());
        if let Some((index, (actual, expected))) = mismatch {
            ensure!(
                false,
                "{name} V planar writes are not bitwise at index {index}: actual={actual:?} expected={expected:?}"
            );
        }
        let bound = super::bf16_chain_bound(3, &[dim]) as f32;
        for (label, metrics) in [("q", q_metrics), ("k", k_metrics)] {
            let max_bound = bound * metrics.max_reference;
            ensure!(
                metrics.max_abs <= max_bound
                    && metrics.p99_abs <= max_bound / 2.0
                    && metrics.mean_abs <= max_bound / 4.0
                    && metrics.cosine >= 1.0 - bound,
                "{name} {label} max={:.9e} p99={:.9e} mean={:.9e} cosine={:.9e} bound={max_bound:e}",
                metrics.max_abs,
                metrics.p99_abs,
                metrics.mean_abs,
                metrics.cosine
            );
        }
        eprintln!(
            "GQA32_QKNORM mode={name} rows={rows} q_max={:.9e} k_max={:.9e} v_bitwise=true PASS",
            q_metrics.max_abs, k_metrics.max_abs
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU window required; Flint sliding prefill ring-overwrite gate"]
fn flint_sliding_ring_prefill_interleaved_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let (rows, ring, hq, hkv, dim) = (520usize, 513usize, 32usize, 4usize, 128usize);
    let heads = hq + 2 * hkv;
    let q_width = hq * dim;
    let kv_width = hkv * dim;
    let mut q_source = Vec::with_capacity(rows * q_width);
    let mut k_source = Vec::with_capacity(rows * kv_width);
    let mut v_source = Vec::with_capacity(rows * kv_width);
    for _ in 0..rows {
        let row = q_source.len() / q_width;
        q_source.extend(
            (0..q_width)
                .map(|i| bf16::from_f32((((i * 83 + 17 + row * 29) % 503) as f32) / 64.0 - 4.0)),
        );
        k_source.extend(
            (0..kv_width)
                .map(|i| bf16::from_f32((((i * 61 + 29 + row * 37) % 503) as f32) / 64.0 - 4.0)),
        );
        v_source.extend(
            (0..kv_width)
                .map(|i| bf16::from_f32((((i * 47 + 41 + row * 43) % 503) as f32) / 64.0 - 4.0)),
        );
    }
    let mut planar = q_source;
    planar.extend(k_source);
    planar.extend(v_source);
    let q_norm: Vec<_> = (0..dim)
        .map(|i| bf16::from_f32(((i * 37 + 3) % 127) as f32 / 64.0 - 1.0))
        .collect();
    let k_norm: Vec<_> = (0..dim)
        .map(|i| bf16::from_f32(((i * 53 + 11) % 127) as f32 / 64.0 - 1.0))
        .collect();
    let source = planar.clone();
    let norm_rope = |row: usize, is_q: bool| -> Vec<bf16> {
        let width = if is_q { q_width } else { kv_width };
        let offset = if is_q {
            row * q_width
        } else {
            rows * q_width + row * kv_width
        };
        let head_count = if is_q { hq } else { hkv };
        let weight = if is_q { &q_norm } else { &k_norm };
        let mut values = source[offset..offset + width].to_vec();
        for head in 0..head_count {
            let base = head * dim;
            let squared: f32 = (0..dim).map(|d| values[base + d].to_f32().powi(2)).sum();
            let rstd = (squared / dim as f32 + 1e-6).sqrt().recip();
            for d in 0..dim {
                values[base + d] =
                    bf16::from_f32(values[base + d].to_f32() * rstd * weight[d].to_f32());
            }
            for pair in 0..dim / 2 {
                let angle = row as f32 * 10_000f32.powf(-(pair as f32 * 2.0 / dim as f32));
                let cos = bf16::from_f32(angle.cos()).to_f32();
                let sin = bf16::from_f32(angle.sin()).to_f32();
                let a = values[base + pair].to_f32();
                let b = values[base + pair + dim / 2].to_f32();
                values[base + pair] = bf16::from_f32(a * cos - b * sin);
                values[base + pair + dim / 2] = bf16::from_f32(b * cos + a * sin);
            }
        }
        values
    };
    let normed_q: Vec<_> = (0..rows).map(|row| norm_rope(row, true)).collect();
    let normed_k: Vec<_> = (0..rows).map(|row| norm_rope(row, false)).collect();
    let mut expected = vec![bf16::ZERO; rows * q_width];
    for row in 0..rows {
        let first = row.saturating_sub(ring - 1);
        let q = &normed_q[row];
        for head in 0..hq {
            let kv_head = head / (hq / hkv);
            let mut scores = Vec::with_capacity(row - first + 1);
            for k in &normed_k[first..=row] {
                let score: f32 = (0..dim)
                    .map(|d| q[head * dim + d].to_f32() * k[kv_head * dim + d].to_f32())
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|score| (score - max).exp()).sum();
            for d in 0..dim {
                let value_offset = rows * (q_width + kv_width) + kv_head * dim;
                let numerator: f32 = scores
                    .iter()
                    .enumerate()
                    .map(|(index, score)| {
                        let position = first + index;
                        let value = source[value_offset + position * kv_width + d].to_f32();
                        (score - max).exp() * value
                    })
                    .sum();
                expected[row * q_width + head * dim + d] = bf16::from_f32(numerator / sum);
            }
        }
    }

    let config = crate::config::Config {
        model_type: "flint".into(),
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 1,
        num_attention_heads: hq,
        num_key_value_heads: hkv,
        head_dim: dim,
        vocab_size: 150144,
        max_position_embeddings: ring,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_scaling: None,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        torch_dtype: "bfloat16".into(),
        eos_token_id: vec![150001],
        attention_bias: false,
        mlp_bias: false,
    };
    let module = cubin::module(&device.ctx, "rope")?;
    let rope = module.load_function("flint_qknorm_rope_kv")?;
    let q_norm_dev = s.clone_htod(&q_norm)?;
    let k_norm_dev = s.clone_htod(&k_norm)?;
    let mut attention = Attention::new(
        &device.ctx,
        s,
        &config,
        ring,
        AttentionPlan {
            chunk: 128,
            qpack: 2,
            threads: 256,
            merge_threads: 256,
        },
    )?;
    let mut position = s.clone_htod(&[0i32])?;
    let mut length = s.clone_htod(&[1i32])?;
    let mut run_case = |interleaved: bool| -> Result<Vec<bf16>> {
        let mut source_dev = s.clone_htod(&source)?;
        let k = s.alloc_zeros::<bf16>(hkv * ring * dim)?;
        let v = s.alloc_zeros::<bf16>(hkv * ring * dim)?;
        let mut output = s.alloc_zeros::<bf16>(rows * q_width)?;
        let k_offset = rows * q_width;
        let v_offset = rows * (q_width + kv_width);
        if !interleaved {
            unsafe {
                s.launch_builder(&rope)
                    .arg(&mut source_dev)
                    .arg(&k)
                    .arg(&v)
                    .arg(&q_norm_dev)
                    .arg(&k_norm_dev)
                    .arg(&position)
                    .arg(&-1i32)
                    .arg(&(rows as i32))
                    .arg(&(hq as i32))
                    .arg(&(hkv as i32))
                    .arg(&(dim as i32))
                    .arg(&(ring as i32))
                    .arg(&(ring as i32))
                    .arg(&10_000.0f32)
                    .arg(&1e-6f32)
                    .arg(&(q_width as u64))
                    .arg(&(k_offset as u64))
                    .arg(&(kv_width as u64))
                    .arg(&(v_offset as u64))
                    .arg(&(kv_width as u64))
                    .launch(LaunchConfig {
                        grid_dim: ((rows * heads) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        for row in 0..rows {
            let tokens = (row + 1).min(ring);
            s.memcpy_htod(&[row as i32], &mut position)?;
            s.memcpy_htod(&[tokens as i32], &mut length)?;
            if interleaved {
                let q_base = row * q_width;
                let row_k = k_offset + row * kv_width;
                let row_v = v_offset + row * kv_width;
                unsafe {
                    s.launch_builder(&rope)
                        .arg(&mut source_dev.slice_mut(q_base..source_dev.len()))
                        .arg(&k)
                        .arg(&v)
                        .arg(&q_norm_dev)
                        .arg(&k_norm_dev)
                        .arg(&position)
                        .arg(&-1i32)
                        .arg(&1i32)
                        .arg(&(hq as i32))
                        .arg(&(hkv as i32))
                        .arg(&(dim as i32))
                        .arg(&(ring as i32))
                        .arg(&(ring as i32))
                        .arg(&10_000.0f32)
                        .arg(&1e-6f32)
                        .arg(&(q_width as u64))
                        .arg(&((row_k - q_base) as u64))
                        .arg(&(kv_width as u64))
                        .arg(&((row_v - q_base) as u64))
                        .arg(&(kv_width as u64))
                        .launch(LaunchConfig {
                            grid_dim: (heads as u32, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            }
            attention.run_with(
                s,
                GqaDecodeInput {
                    q: &source_dev,
                    q_offset: row * q_width,
                    k: &k,
                    v: &v,
                    output: &mut output,
                    output_offset: row * q_width,
                    length: &length,
                    sinks: None,
                    window: Some(SlidingWindow {
                        position: &position,
                        head_stride: (ring * dim) as i32,
                    }),
                },
            )?;
        }
        s.synchronize()?;
        Ok(s.clone_dtoh(&output)?)
    };
    let interleaved = run_case(true)?;
    let metrics = attention_metrics(
        &interleaved,
        &expected
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>(),
    );
    let bound = super::bf16_chain_bound(3, &[dim, ring + 1]) as f32;
    let max_bound = bound * metrics.max_reference;
    ensure!(
        metrics.max_abs <= max_bound
            && metrics.p99_abs <= max_bound / 2.0
            && metrics.mean_abs <= max_bound / 4.0
            && metrics.cosine >= 1.0 - bound,
        "interleaved ring prefill max={:.9e} p99={:.9e} mean={:.9e} cosine={:.9e} bound={max_bound:e}",
        metrics.max_abs,
        metrics.p99_abs,
        metrics.mean_abs,
        metrics.cosine
    );
    let legacy = run_case(false)?;
    let legacy_metrics = attention_metrics(
        &legacy,
        &expected
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>(),
    );
    let legacy_bound = bound * legacy_metrics.max_reference;
    ensure!(
        legacy_metrics.max_abs > legacy_bound || legacy_metrics.cosine < 1.0 - bound,
        "full-write negative control passed the interleaved bound: max={:.9e} cosine={:.9e} bound={legacy_bound:e}",
        legacy_metrics.max_abs,
        legacy_metrics.cosine
    );
    let chunk = 520usize;
    let logical_k: &dyn Fn(usize) -> Vec<bf16> = &|p: usize| -> Vec<bf16> {
        (0..kv_width)
            .map(|i| bf16::from_f32((((i * 61 + 29 + p * 37) % 503) as f32) / 64.0 - 4.0))
            .collect()
    };
    let logical_v: &dyn Fn(usize) -> Vec<bf16> = &|p: usize| -> Vec<bf16> {
        (0..kv_width)
            .map(|i| bf16::from_f32((((i * 47 + 41 + p * 43) % 503) as f32) / 64.0 - 4.0))
            .collect()
    };
    let staged_capacity = 512 + chunk;
    let window = ring - 1;
    let mut flash = Flash::new(s.clone(), chunk, hq, hkv, dim)?;
    let mut initial_k = vec![bf16::ZERO; hkv * ring * dim];
    let mut initial_v = vec![bf16::ZERO; hkv * ring * dim];
    for absolute in 0..=512 {
        let source_k = logical_k(absolute);
        let source_v = logical_v(absolute);
        for head in 0..hkv {
            initial_k
                [(head * ring + absolute % ring) * dim..(head * ring + absolute % ring + 1) * dim]
                .copy_from_slice(&source_k[head * dim..(head + 1) * dim]);
            initial_v
                [(head * ring + absolute % ring) * dim..(head * ring + absolute % ring + 1) * dim]
                .copy_from_slice(&source_v[head * dim..(head + 1) * dim]);
        }
    }
    let mut ring_k = s.clone_htod(&initial_k)?;
    let mut ring_v = s.clone_htod(&initial_v)?;
    let mut stage_k = vec![bf16::ZERO; hkv * staged_capacity * dim];
    let mut stage_v = vec![bf16::ZERO; hkv * staged_capacity * dim];
    for absolute in 513..=1032 {
        let source_k = logical_k(absolute);
        let source_v = logical_v(absolute);
        let storage = absolute - 1;
        for head in 0..hkv {
            stage_k[(head * staged_capacity + storage) * dim
                ..(head * staged_capacity + storage + 1) * dim]
                .copy_from_slice(&source_k[head * dim..(head + 1) * dim]);
            stage_v[(head * staged_capacity + storage) * dim
                ..(head * staged_capacity + storage + 1) * dim]
                .copy_from_slice(&source_v[head * dim..(head + 1) * dim]);
        }
    }
    let mut stage_k = s.clone_htod(&stage_k)?;
    let mut stage_v = s.clone_htod(&stage_v)?;
    for (source, target) in [(&ring_k, &mut stage_k), (&ring_v, &mut stage_v)] {
        super::copy::RingCopy {
            stream: s,
            source,
            target,
            source_capacity: ring,
            target_capacity: staged_capacity,
            heads: hkv,
            head_dim: dim,
        }
        .run(1, 0, 512, true, false)?;
    }
    for (actual, logical) in [
        (s.clone_dtoh(&stage_k)?, &logical_k),
        (s.clone_dtoh(&stage_v)?, &logical_v),
    ] {
        for head in 0..hkv {
            for position in 0..staged_capacity {
                let expected = logical(position + 1);
                assert_eq!(
                    &actual[(head * staged_capacity + position) * dim
                        ..(head * staged_capacity + position + 1) * dim],
                    &expected[head * dim..(head + 1) * dim],
                    "staged K/V mismatch at head={head} position={position}"
                );
            }
        }
    }
    let staged_q = s.alloc_zeros::<bf16>(chunk * hq * dim)?;
    let no_sinks = s.clone_htod(&vec![bf16::NEG_INFINITY; hq])?;
    let mut staged_output = s.alloc_zeros::<bf16>(chunk * hq * dim)?;
    flash.run(
        &staged_q,
        hq * dim,
        &stage_k,
        &stage_v,
        &mut staged_output,
        Shape {
            rows: chunk,
            start: 512,
            q_heads: hq,
            kv_heads: hkv,
            capacity: staged_capacity,
            head_dim: dim,
            causal: true,
            window_left: Some(window),
        },
    )?;
    flash.apply_sinks(&no_sinks, &mut staged_output, chunk, hq)?;
    for (source, target) in [(&stage_k, &mut ring_k), (&stage_v, &mut ring_v)] {
        super::copy::RingCopy {
            stream: s,
            source,
            target,
            source_capacity: staged_capacity,
            target_capacity: ring,
            heads: hkv,
            head_dim: dim,
        }
        .run(519, 520 % ring, 513, false, true)?;
    }
    for (actual, logical) in [
        (s.clone_dtoh(&ring_k)?, &logical_k),
        (s.clone_dtoh(&ring_v)?, &logical_v),
    ] {
        for head in 0..hkv {
            for position in 0..ring {
                let expected = logical(520 + (position + ring - 520 % ring) % ring);
                assert_eq!(
                    &actual[(head * ring + position) * dim..(head * ring + position + 1) * dim],
                    &expected[head * dim..(head + 1) * dim],
                    "committed K/V mismatch at head={head} position={position}"
                );
            }
        }
    }
    let mut prefix = vec![0.0f32; hkv * dim * 1033];
    for head in 0..hkv {
        for d in 0..dim {
            for absolute in 1..=1032 {
                let value = logical_v(absolute)[head * dim + d].to_f32();
                prefix[(head * dim + d) * 1033 + absolute] =
                    prefix[(head * dim + d) * 1033 + absolute - 1] + value;
            }
        }
    }
    let staged_actual = s.clone_dtoh(&staged_output)?;
    let mut expected = vec![0.0f32; chunk * hq * dim];
    for row in 0..chunk {
        let absolute = 513 + row;
        let first = absolute.saturating_sub(window);
        let tokens = absolute - first + 1;
        for head in 0..hq {
            let kv_head = head / (hq / hkv);
            for d in 0..dim {
                let index = (kv_head * dim + d) * 1033;
                let before = if first == 0 {
                    0.0
                } else {
                    prefix[index + first - 1]
                };
                expected[(row * hq + head) * dim + d] =
                    (prefix[index + absolute] - before) / tokens as f32;
            }
        }
    }
    let metrics = attention_metrics(&staged_actual, &expected);
    let bound = super::bf16_chain_bound(3, &[dim, window + 2]) as f32 * metrics.max_reference;
    ensure!(
        metrics.max_abs <= bound,
        "staged FA2 max={} bound={bound}",
        metrics.max_abs
    );
    ensure!(metrics.p99_abs <= bound / 2.0, "staged FA2 p99 bound");
    ensure!(metrics.mean_abs <= bound / 4.0, "staged FA2 mean bound");
    ensure!(
        metrics.cosine >= 1.0 - bound as f32,
        "staged FA2 cosine bound"
    );
    for (source, target) in [(&ring_k, &mut stage_k), (&ring_v, &mut stage_v)] {
        super::copy::RingCopy {
            stream: s,
            source,
            target,
            source_capacity: ring,
            target_capacity: staged_capacity,
            heads: hkv,
            head_dim: dim,
        }
        .run(1031 % ring, 0, 1, false, false)?;
    }
    let mut overwrite_output = s.alloc_zeros::<bf16>(chunk * hq * dim)?;
    flash.run(
        &staged_q,
        hq * dim,
        &stage_k,
        &stage_v,
        &mut overwrite_output,
        Shape {
            rows: chunk,
            start: 512,
            q_heads: hq,
            kv_heads: hkv,
            capacity: staged_capacity,
            head_dim: dim,
            causal: true,
            window_left: Some(window),
        },
    )?;
    flash.apply_sinks(&no_sinks, &mut overwrite_output, chunk, hq)?;
    let overwrite_output = s.clone_dtoh(&overwrite_output)?;
    ensure!(
        staged_actual
            .iter()
            .zip(&overwrite_output)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "early-key overwrite control does not discriminate"
    );
    eprintln!(
        "FLINT_STAGED_RING chunk={chunk} staged_capacity={staged_capacity} max_abs={:.9e} p99_abs={:.9e} mean_abs={:.9e} cosine={:.9e} relative_bound={bound:e} overwrite_control=PASS",
        metrics.max_abs, metrics.p99_abs, metrics.mean_abs, metrics.cosine
    );
    eprintln!(
        "FLINT_RING_PREFILL rows={rows} ring={ring} interleaved_max={:.9e} interleaved_cosine={:.9e} legacy_max={:.9e} legacy_cosine={:.9e} PASS",
        metrics.max_abs, metrics.cosine, legacy_metrics.max_abs, legacy_metrics.cosine
    );
    Ok(())
}

#[test]
#[ignore = "GPU window required; Flint FA2 learned-sink gate"]
fn fa2_gqa32_learned_sink_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let (rows, start, hq, hkv, dim, capacity) =
        (128usize, 64usize, 32usize, 4usize, 128usize, 256usize);
    let slide_rows = 2048usize;
    let q: Vec<_> = (0..rows * hq * dim)
        .map(|i| bf16::from_f32((((i * 83 + 17) % 503) as f32) / 64.0 - 4.0))
        .collect();
    let k: Vec<_> = (0..hkv * capacity * dim)
        .map(|i| bf16::from_f32((((i * 61 + 29) % 503) as f32) / 64.0 - 4.0))
        .collect();
    let v: Vec<_> = (0..hkv * capacity * dim)
        .map(|i| bf16::from_f32((((i * 47 + 41) % 503) as f32) / 64.0 - 4.0))
        .collect();
    let sinks: Vec<_> = (0..hq)
        .map(|h| bf16::from_f32((((h * 41 + 7) % 64) as f32) / 8.0 - 4.0))
        .collect();
    let negligible = vec![bf16::from_f32(-100.0); hq];
    let q_dev = s.clone_htod(&q)?;
    let k_dev = s.clone_htod(&k)?;
    let v_dev = s.clone_htod(&v)?;
    let sinks_dev = s.clone_htod(&sinks)?;
    let negligible_dev = s.clone_htod(&negligible)?;
    let mut ordinary = s.alloc_zeros::<bf16>(rows * hq * dim)?;
    let mut sink_output = s.alloc_zeros::<bf16>(rows * hq * dim)?;
    let mut negligible_output = s.alloc_zeros::<bf16>(rows * hq * dim)?;
    let mut flash = Flash::new(s.clone(), rows.max(slide_rows), hq, hkv, dim)?;
    let shape = Shape {
        rows,
        start,
        q_heads: hq,
        kv_heads: hkv,
        capacity,
        head_dim: dim,
        causal: true,
        window_left: None,
    };
    flash.run(&q_dev, hq * dim, &k_dev, &v_dev, &mut ordinary, shape)?;
    s.memcpy_dtod(&ordinary, &mut sink_output)?;
    flash.apply_sinks(&sinks_dev, &mut sink_output, rows, hq)?;
    s.memcpy_dtod(&ordinary, &mut negligible_output)?;
    flash.apply_sinks(&negligible_dev, &mut negligible_output, rows, hq)?;
    s.synchronize()?;
    let ordinary = s.clone_dtoh(&ordinary)?;
    let negligible_output = s.clone_dtoh(&negligible_output)?;
    let sink_output = s.clone_dtoh(&sink_output)?;
    ensure!(
        ordinary
            .iter()
            .zip(&negligible_output)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "negligible FA2 sink changed ordinary output"
    );
    ensure!(
        ordinary
            .iter()
            .zip(&sink_output)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "meaningful FA2 sink did not change ordinary output"
    );
    let negligible_max = ordinary
        .iter()
        .zip(&negligible_output)
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);

    let mut expected = vec![0f32; rows * hq * dim];
    for row in 0..rows {
        let keys = start + row + 1;
        for (head, sink) in sinks.iter().enumerate() {
            let kv_head = head / (hq / hkv);
            let mut scores = vec![0f32; keys];
            for (position, score) in scores.iter_mut().enumerate() {
                let q_base = (row * hq + head) * dim;
                let k_base = (kv_head * capacity + position) * dim;
                *score = (0..dim)
                    .map(|d| q[q_base + d].to_f32() * k[k_base + d].to_f32())
                    .sum::<f32>()
                    * (dim as f32).sqrt().recip();
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|score| (score - max_score).exp()).sum();
            let expected_lse = max_score + sum.ln();
            let denominator = 1.0 + (sink.to_f32() - expected_lse).exp();
            for d in 0..dim {
                let index = (row * hq + head) * dim + d;
                let mut numerator = 0f32;
                for (position, score) in scores.iter().enumerate() {
                    let probability = ((score - max_score).exp() / sum) as f32;
                    let v_base = (kv_head * capacity + position) * dim;
                    numerator += probability * v[v_base + d].to_f32();
                }
                expected[index] = bf16::from_f32(numerator / denominator).to_f32();
            }
        }
    }
    let metrics = attention_metrics(&sink_output, &expected);
    let bound = super::bf16_chain_bound(3, &[dim, capacity + 1]);
    let max_bound = bound as f32 * metrics.max_reference;
    ensure!(
        metrics.max_abs <= max_bound,
        "FA2 sink max={:.9e} bound={max_bound:e}",
        metrics.max_abs
    );
    ensure!(
        metrics.p99_abs <= max_bound / 2.0,
        "FA2 sink p99={:.9e} bound={}",
        metrics.p99_abs,
        max_bound / 2.0
    );
    ensure!(
        metrics.mean_abs <= max_bound / 4.0,
        "FA2 sink mean={:.9e} bound={}",
        metrics.mean_abs,
        max_bound / 4.0
    );
    ensure!(
        metrics.cosine >= 1.0 - bound as f32,
        "FA2 sink cosine={:.9e} bound={}",
        metrics.cosine,
        1.0 - bound as f32
    );
    eprintln!(
        "FA2_GQA32_SINK rows={rows} negligible_max={negligible_max:e} max_abs={:.9e} p99_abs={:.9e} mean_abs={:.9e} cosine={:.9e} max_reference={:.9e} relative_bound={max_bound:e} PASS",
        metrics.max_abs, metrics.p99_abs, metrics.mean_abs, metrics.cosine, metrics.max_reference
    );

    let (slide_capacity, window) = (2048usize, 512usize);
    let slide_q = vec![bf16::ZERO; slide_rows * hq * dim];
    let slide_k = vec![bf16::ZERO; hkv * slide_capacity * dim];
    let mut slide_v = vec![bf16::ZERO; hkv * slide_capacity * dim];
    let slide_pattern = [1.0f32, -1.0, 0.5, -0.5, 0.25];
    let mut slide_prefix = vec![0f32; slide_capacity + 1];
    for position in 0..slide_capacity {
        slide_prefix[position + 1] = slide_prefix[position] + slide_pattern[position % 5];
        let value = bf16::from_f32(slide_pattern[position % 5]);
        for head in 0..hkv {
            slide_v[(head * slide_capacity + position) * dim] = value;
        }
    }
    let slide_q_dev = s.clone_htod(&slide_q)?;
    let slide_k_dev = s.clone_htod(&slide_k)?;
    let slide_v_dev = s.clone_htod(&slide_v)?;
    let zero_sinks = vec![bf16::ZERO; hq];
    let zero_sinks_dev = s.clone_htod(&zero_sinks)?;
    let mut slide_ordinary = s.alloc_zeros::<bf16>(slide_rows * hq * dim)?;
    let mut slide_sink = s.alloc_zeros::<bf16>(slide_rows * hq * dim)?;
    let slide_shape = Shape {
        rows: slide_rows,
        start: 0,
        q_heads: hq,
        kv_heads: hkv,
        capacity: slide_capacity,
        head_dim: dim,
        causal: true,
        window_left: Some(window),
    };
    flash.run(
        &slide_q_dev,
        hq * dim,
        &slide_k_dev,
        &slide_v_dev,
        &mut slide_ordinary,
        slide_shape,
    )?;
    s.memcpy_dtod(&slide_ordinary, &mut slide_sink)?;
    flash.apply_sinks(&zero_sinks_dev, &mut slide_sink, slide_rows, hq)?;
    s.synchronize()?;
    let slide_ordinary = s.clone_dtoh(&slide_ordinary)?;
    let slide_sink = s.clone_dtoh(&slide_sink)?;
    ensure!(
        slide_ordinary
            .iter()
            .zip(&slide_sink)
            .any(|(ordinary, sink)| ordinary.to_bits() != sink.to_bits()),
        "zero FA2 sink did not rescale sliding prefill"
    );
    let mut slide_expected = vec![0f32; slide_rows * hq * dim];
    for row in 0..slide_rows {
        let first = row.saturating_sub(window);
        let tokens = row - first + 1;
        let window_sum = slide_prefix[row + 1] - slide_prefix[first];
        for head in 0..hq {
            slide_expected[(row * hq + head) * dim] = window_sum / (tokens as f32 + 1.0);
        }
    }
    let slide_metrics = attention_metrics(&slide_sink, &slide_expected);
    let slide_bound = super::bf16_chain_bound(3, &[dim, window + 2]);
    let slide_max_bound = slide_bound as f32 * slide_metrics.max_reference;
    eprintln!(
        "FA2_SLIDING_METRICS max_abs={:.9e} p99_abs={:.9e} mean_abs={:.9e} cosine={:.9e} max_reference={:.9e}",
        slide_metrics.max_abs,
        slide_metrics.p99_abs,
        slide_metrics.mean_abs,
        slide_metrics.cosine,
        slide_metrics.max_reference
    );
    ensure!(
        slide_metrics.max_abs <= slide_max_bound,
        "sliding FA2 max={:.9e} bound={slide_max_bound:e}",
        slide_metrics.max_abs
    );
    ensure!(
        slide_metrics.p99_abs <= slide_max_bound / 2.0,
        "sliding FA2 p99={:.9e} bound={}",
        slide_metrics.p99_abs,
        slide_max_bound / 2.0
    );
    ensure!(
        slide_metrics.mean_abs <= slide_max_bound / 4.0,
        "sliding FA2 mean={:.9e} bound={}",
        slide_metrics.mean_abs,
        slide_max_bound / 4.0
    );
    ensure!(
        slide_metrics.cosine >= 1.0 - slide_bound as f32,
        "sliding FA2 cosine={:.9e} bound={}",
        slide_metrics.cosine,
        1.0 - slide_bound as f32
    );

    let one_capacity = 1025usize;
    let one_start = 1024usize;
    let one_q = vec![bf16::ZERO; hq * dim];
    let one_k = vec![bf16::ZERO; hkv * one_capacity * dim];
    let mut one_v = vec![bf16::ZERO; hkv * one_capacity * dim];
    for head in 0..hkv {
        one_v[(head * one_capacity + one_start - window) * dim] = bf16::ONE;
        one_v[(head * one_capacity + one_start - window - 1) * dim + 1] = bf16::ONE;
        one_v[(head * one_capacity + one_start) * dim + 2] = bf16::ONE;
    }
    let one_q_dev = s.clone_htod(&one_q)?;
    let one_k_dev = s.clone_htod(&one_k)?;
    let one_v_dev = s.clone_htod(&one_v)?;
    let mut one_output = s.alloc_zeros::<bf16>(hq * dim)?;
    flash.run(
        &one_q_dev,
        hq * dim,
        &one_k_dev,
        &one_v_dev,
        &mut one_output,
        Shape {
            rows: 1,
            start: one_start,
            q_heads: hq,
            kv_heads: hkv,
            capacity: one_capacity,
            head_dim: dim,
            causal: true,
            window_left: Some(window),
        },
    )?;
    flash.apply_sinks(&sinks_dev, &mut one_output, 1, hq)?;
    s.synchronize()?;
    let one_output = s.clone_dtoh(&one_output)?;
    for head in 0..hq {
        let base = head * dim;
        ensure!(
            one_output[base + 1].to_bits() == bf16::ZERO.to_bits(),
            "sliding FA2 included excluded key p-513"
        );
        ensure!(
            one_output[base].to_f32() > 0.0 && one_output[base + 2].to_f32() > 0.0,
            "sliding FA2 omitted p-512 or p"
        );
    }
    eprintln!(
        "FA2_GQA32_SLIDING rows={slide_rows} window={window} max_abs={:.9e} p99_abs={:.9e} mean_abs={:.9e} cosine={:.9e} relative_bound={slide_max_bound:e} one_hot_boundary=PASS",
        slide_metrics.max_abs, slide_metrics.p99_abs, slide_metrics.mean_abs, slide_metrics.cosine
    );

    Ok(())
}

fn affine_codes(values: &[bf16]) -> (Vec<u8>, [f32; 2]) {
    let lo = values.iter().map(|v| v.to_f32()).fold(f32::MAX, f32::min);
    let hi = values.iter().map(|v| v.to_f32()).fold(f32::MIN, f32::max);
    let scale = if hi == lo { 1.0 } else { (hi - lo) / 255.0 };
    let codes = values
        .iter()
        .map(|v| {
            ((v.to_f32() - lo) / scale)
                .clamp(0.0, 255.0)
                .round_ties_even() as u8
        })
        .collect();
    (codes, [lo, scale])
}

impl CommittedKv {
    fn capture(kv: &super::engine::Kv, prefix: usize, s: &Arc<CudaStream>) -> Result<Self> {
        let c = kv.compact.as_ref().context("compact snapshot missing")?;
        ensure!(
            s.clone_dtoh(&c.cursor)?
                == [prefix as i32, crate::kv::sealed_until(prefix) as i32, 0, 0],
            "compact snapshot is not committed at prefix {prefix}"
        );
        let live = [s.clone_dtoh(&kv.k)?, s.clone_dtoh(&kv.v)?];
        let stats = [s.clone_dtoh(&c.k_stats)?, s.clone_dtoh(&c.v_stats)?];
        ensure!(
            live.iter().flatten().all(|v| v.is_finite())
                && stats.iter().flatten().all(|v| v.is_finite()),
            "nonfinite compact snapshot"
        );
        Ok(Self {
            prefix,
            layout: c.layout,
            live,
            codes: [s.clone_dtoh(&c.k_code)?, s.clone_dtoh(&c.v_code)?],
            stats,
        })
    }

    fn differences(&self, kv: &super::engine::Kv, s: &Arc<CudaStream>) -> Result<usize> {
        let c = kv.compact.as_ref().context("compact comparison missing")?;
        let cursor = s.clone_dtoh(&c.cursor)?;
        let sealed = usize::try_from(cursor[1])?;
        let previous = crate::kv::sealed_until(self.prefix);
        ensure!(
            cursor[0] >= self.prefix as i32
                && cursor[2..] == [0, 0]
                && sealed >= previous
                && sealed <= self.prefix
                && sealed.is_multiple_of(crate::kv::GROUP_TOKENS),
            "compact comparison cursor {cursor:?} prefix={}",
            self.prefix
        );
        let mut codes = self.codes.clone();
        let mut stats = self.stats.clone();
        for h in 0..4 {
            for start in (previous..sealed).step_by(crate::kv::GROUP_TOKENS) {
                for d in 0..256 {
                    let values = (start..start + crate::kv::GROUP_TOKENS)
                        .map(|p| {
                            Ok(self.live[0][(h * self.layout.ring
                                + crate::kv::ring_slot(p, self.layout.ring)?)
                                * 256
                                + d])
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (packed, scale) = affine_codes(&values);
                    let stat = (h * self.layout.capacity.div_ceil(crate::kv::GROUP_TOKENS)
                        + start / crate::kv::GROUP_TOKENS)
                        * 256
                        + d;
                    stats[0][2 * stat..2 * stat + 2].copy_from_slice(&scale);
                    for (r, code) in packed.into_iter().enumerate() {
                        codes[0][(h * self.layout.capacity + start + r) * 256 + d] = code;
                    }
                }
                for p in start..start + crate::kv::GROUP_TOKENS {
                    let lo =
                        (h * self.layout.ring + crate::kv::ring_slot(p, self.layout.ring)?) * 256;
                    let (packed, scale) = affine_codes(&self.live[1][lo..lo + 256]);
                    let dest = (h * self.layout.capacity + p) * 256;
                    codes[1][dest..dest + 256].copy_from_slice(&packed);
                    let stat = h * self.layout.capacity + p;
                    stats[1][2 * stat..2 * stat + 2].copy_from_slice(&scale);
                }
            }
        }
        let mut different = 0;
        for (which, tensor) in [&kv.k, &kv.v].into_iter().enumerate() {
            let actual = s.clone_dtoh(tensor)?;
            for h in 0..4 {
                for p in (0..self.prefix).filter(|&p| p < crate::kv::SINK_TOKENS || p >= sealed) {
                    let lo =
                        (h * self.layout.ring + crate::kv::ring_slot(p, self.layout.ring)?) * 256;
                    different += actual[lo..lo + 256]
                        .iter()
                        .zip(&self.live[which][lo..lo + 256])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
            }
        }
        for (which, tensor) in [&c.k_code, &c.v_code].into_iter().enumerate() {
            different += s
                .clone_dtoh(tensor)?
                .iter()
                .zip(&codes[which])
                .filter(|(a, b)| a != b)
                .count();
        }
        for (which, tensor) in [&c.k_stats, &c.v_stats].into_iter().enumerate() {
            let actual = s.clone_dtoh(tensor)?;
            ensure!(
                actual.iter().all(|v| v.is_finite()),
                "nonfinite KV statistics"
            );
            different += actual
                .iter()
                .zip(&stats[which])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
        Ok(different)
    }
}

#[test]
#[ignore = "local GPU window required"]
fn kv_cache_five_row_wrap_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let cap = 16640;
    let prefix = 4095;
    let rows = crate::mtp::MAX_VERIFY_ROWS;
    ensure!(rows == 5, "five-row regression geometry changed");
    let spec = crate::model_spec::ModelSpec::qwen(
        &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
        &Value::Null,
    )?;
    ensure!(
        spec.kv_heads == 4 && spec.head_dim == 256,
        "Qwen KV geometry"
    );
    let spec = crate::mtp::component_spec(&spec)?;
    let values = |shift: usize| {
        (0..4 * cap * 256)
            .map(|i| {
                let d = i % 256;
                let p = i / 256 % cap;
                let h = i / (256 * cap);
                bf16::from_f32(((p * 17 + d * 29 + h * 43 + shift) % 251) as f32 / 64.0 - 2.0)
            })
            .collect::<Vec<_>>()
    };
    let k_src = s.clone_htod(&values(3))?;
    let v_src = s.clone_htod(&values(19))?;
    let mut state = StateStore::new_with_format(&spec, cap, KvFormat::Affine8, s)?;
    state.set_sequence(s, 0, prefix as u32, prefix as u32)?;
    let kv = &state.kv[0];
    let c = kv.compact.as_ref().context("compact wrap cache missing")?;
    let cache = c.view(&kv.k, &kv.v, s);
    c.finish(
        cache,
        state.seq_ref(s)?,
        Some((&k_src, &v_src, prefix)),
        true,
        s,
    )?;
    let before = CommittedKv::capture(kv, prefix, s)?;
    let k_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
    let v_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
    state.set_sequence(s, prefix as u32, rows as u32, rows as u32)?;
    let kv = &state.kv[0];
    let c = kv.compact.as_ref().context("compact wrap cache missing")?;
    c.dequant_prefix(cache, state.seq_ref(s)?, &k_full, &v_full, prefix, s)?;
    let projection = |shift: usize| {
        (0..rows * 1024)
            .map(|i| bf16::from_f32(((i * 31 + shift) % 257) as f32 / 128.0 - 1.0))
            .collect::<Vec<_>>()
    };
    let k = s.clone_htod(&projection(11))?;
    let v = s.clone_htod(&projection(23))?;
    let module = cubin::module(&device.ctx, "tcmqa_d256")?;
    let writer = module.load_function("tcmqa_d256_write_ring")?;
    let scatter_k = module.load_function("tcmqa_d256_rope_k_scatter")?;
    let scatter_v = module.load_function("tcmqa_d256_v_scatter")?;
    let meta = state.seq_ref(s)?.ptr;
    let (kp, vp) = (k.device_ptr(s).0, v.device_ptr(s).0);
    let (kf, vf) = (k_full.device_ptr(s).0, v_full.device_ptr(s).0);
    let (stride, head, dim, axes) = (1024i32, (cap * 256) as i32, 256i32, 0u64);
    let write_rows = || -> Result<()> {
        unsafe {
            s.launch_builder(&writer)
                .arg(&kp)
                .arg(&vp)
                .arg(&meta)
                .arg(&axes)
                .arg(&stride)
                .arg(&stride)
                .arg(&cache)
                .launch(flat(rows * 1024))?;
            s.launch_builder(&scatter_k)
                .arg(&kp)
                .arg(&kf)
                .arg(&meta)
                .arg(&stride)
                .arg(&head)
                .arg(&dim)
                .launch(flat(rows * 1024))?;
            s.launch_builder(&scatter_v)
                .arg(&vp)
                .arg(&vf)
                .arg(&meta)
                .arg(&stride)
                .arg(&head)
                .arg(&dim)
                .launch(flat(rows * 1024))?;
        }
        Ok(())
    };
    write_rows()?;
    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launches = write_rows();
    let graph = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
    launches?;
    let graph = graph?.context("empty five-row writer graph")?;
    graph.launch()?;
    let q = s.clone_htod(
        &(0..rows * 6144)
            .map(|i| bf16::from_f32(((i * 7) % 251) as f32 / 512.0 - 0.25))
            .collect::<Vec<_>>(),
    )?;
    let core = Core {
        device: &device,
        q: &q,
        k: &kv.k,
        v: &kv.v,
        compact: Some(cache),
    };
    let eager = core.run(prefix, rows as u32, 0, false)?;
    let actual = core.run(prefix, rows as u32, 0, true)?;
    let expected = Core {
        device: &device,
        q: &q,
        k: &k_full,
        v: &v_full,
        compact: None,
    }
    .run(prefix, rows as u32, 0, true)?;
    let attention_differences = actual
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    state.set_sequence(s, (prefix + rows) as u32, 1, rows as u32)?;
    state.compact_commit(s, true)?;
    let committed_differences = before.differences(&state.kv[0], s)?;
    let graph_differences = eager
        .iter()
        .zip(&actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let different = attention_differences + committed_differences + graph_differences;
    let verdict = if different == 0 { "PASS" } else { "DEFECT" };
    println!(
        "T24_KV_FIVE_ROW_WRAP {verdict} prefix={prefix} rows={rows} attention_differences={attention_differences} committed_differences={committed_differences} graph_differences={graph_differences}"
    );
    ensure!(different == 0, "five-row KV wrap DEFECT");
    Ok(())
}

fn input() -> Result<(PathBuf, PathBuf, Value)> {
    let root = PathBuf::from(
        std::env::var_os("FF_Q38_MTP_REF_ROOT").context("FF_Q38_MTP_REF_ROOT missing")?,
    );
    let out =
        PathBuf::from(std::env::var_os("FF_Q38_MTP_OUTPUT").context("FF_Q38_MTP_OUTPUT missing")?);
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("FIXTURES.json"))?)?;
    ensure!(
        manifest["capacity"] == 16640 && manifest["shape"] == json!([4, 16640, 256]),
        "KV fixture geometry"
    );
    std::fs::create_dir_all(&out)?;
    Ok((root, out, manifest))
}
#[test]
fn compact_binding_rejects_wrong_geometry_and_alignment() -> Result<()> {
    assert_eq!(std::mem::size_of::<CacheView>(), 64);
    let view = CacheView {
        k_code: 256,
        v_code: 512,
        k_stats: 768,
        v_stats: 1024,
        k_ring: 1280,
        v_ring: 1536,
        cursor: 1792,
        capacity: 16640,
        ring: crate::kv::RING_TOKENS as i32,
    };
    for rows in 1..=crate::mtp::MAX_VERIFY_ROWS as u32 {
        view.validate(16640, rows)?;
    }
    assert!(
        CacheView {
            ring: view.ring - 1,
            ..view
        }
        .validate(16640, 4)
        .is_err()
    );
    assert!(
        CacheView {
            k_stats: 769,
            ..view
        }
        .validate(16640, 4)
        .is_err()
    );
    assert!(view.validate(16640, 0).is_err());
    assert!(
        view.validate(16640, crate::mtp::MAX_VERIFY_ROWS as u32 + 1)
            .is_err()
    );
    Ok(())
}
fn bf16_file(path: PathBuf, elements: usize) -> Result<Vec<bf16>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() == elements * 2, "KV fixture extent");
    let data = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])))
        .collect::<Vec<_>>();
    ensure!(data.iter().all(|x| x.is_finite()), "nonfinite KV fixture");
    Ok(data)
}
fn digest<T: DeviceRepr>(device: &Device, tensor: &CudaSlice<T>) -> Result<String> {
    let host = device.stream.clone_dtoh(tensor)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            host.as_ptr().cast::<u8>(),
            std::mem::size_of_val(host.as_slice()),
        )
    };
    Ok(sha256(bytes))
}
fn host_digest(values: &[bf16]) -> String {
    sha256(
        &values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn cache_hashes(device: &Device, state: &StateStore) -> Result<Value> {
    let c = state.kv[0]
        .compact
        .as_ref()
        .context("compact cache missing")?;
    Ok(
        json!({"k_code":digest(device,&c.k_code)?,"v_code":digest(device,&c.v_code)?,"k_stats":digest(device,&c.k_stats)?,"v_stats":digest(device,&c.v_stats)?}),
    )
}
struct Core<'a> {
    device: &'a Device,
    q: &'a CudaSlice<bf16>,
    k: &'a CudaSlice<bf16>,
    v: &'a CudaSlice<bf16>,
    compact: Option<CacheView>,
}
impl Core<'_> {
    fn run(&self, prefix: usize, rows: u32, qrow: usize, graph: bool) -> Result<Vec<bf16>> {
        let cap = 16640u32;
        let ws = AttentionWorkspace::new(
            &crate::op_contract::FullAttentionSpec::q38(),
            rows,
            cap,
            256,
        )?;
        let s = &self.device.stream;
        let meta = s.clone_htod(&[
            prefix as i32,
            rows as i32,
            prefix as i32 + rows as i32,
            cap as i32,
            1,
            0,
            0,
            0,
        ])?;
        let scratch = s.alloc_zeros::<u8>(ws.bytes as usize)?;
        let output = s.alloc_zeros::<bf16>(rows as usize * 6144)?;
        let mask = s.clone_htod(
            &(0..64)
                .map(|i| {
                    if i == 63 {
                        u64::MAX
                    } else {
                        (1u64 << (i + 1)) - 1
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        let module = cubin::module(&self.device.ctx, "tcmqa_d256")?;
        let p1 = module.load_function(if self.compact.is_some() {
            "tcmqa_d256_phase1_compact"
        } else {
            "tcmqa_d256_phase1"
        })?;
        p1.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,68096)?;
        let p2 = module.load_function("tcmqa_d256_phase2")?;
        let q = self.q.device_ptr(s).0 + (qrow * 6144 * 2) as u64;
        let (k, v) = (self.k.device_ptr(s).0, self.v.device_ptr(s).0);
        let base = scratch.device_ptr(s).0;
        let (max, sum, out) = (base + ws.p_max, base + ws.p_sum, base + ws.p_out);
        let (meta, mask, output_ptr) = (
            meta.device_ptr(s).0,
            mask.device_ptr(s).0,
            output.device_ptr(s).0,
        );
        let (qt, head, pos, chunk, parts, seg, row0) = (
            6144i32,
            cap as i32 * 256,
            256i32,
            256i32,
            cap.div_ceil(256) as i32,
            rows as i32,
            0i32,
        );
        let label = if self.compact.is_some() {
            "compact TCMQA"
        } else {
            "BF16 TCMQA"
        };
        if graph {
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .with_context(|| format!("{label} graph begin"))?;
        }
        let launches = (|| -> Result<()> {
            unsafe {
                let mut launch = s.launch_builder(&p1);
                launch
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&max)
                    .arg(&sum)
                    .arg(&out)
                    .arg(&mask)
                    .arg(&meta)
                    .arg(&qt)
                    .arg(&head)
                    .arg(&pos)
                    .arg(&head)
                    .arg(&pos)
                    .arg(&chunk)
                    .arg(&parts)
                    .arg(&seg)
                    .arg(&row0);
                if let Some(cache) = self.compact.as_ref() {
                    launch.arg(cache);
                }
                launch.launch(LaunchConfig {
                    grid_dim: (cap.div_ceil(256), 4, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 68096,
                })?;
                s.launch_builder(&p2)
                    .arg(&max)
                    .arg(&sum)
                    .arg(&out)
                    .arg(&output_ptr)
                    .arg(&meta)
                    .arg(&qt)
                    .arg(&chunk)
                    .arg(&parts)
                    .arg(&seg)
                    .arg(&row0)
                    .launch(LaunchConfig {
                        grid_dim: (rows * 24, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: (2 * cap.div_ceil(256) + 256) * 4,
                    })?;
            }
            Ok(())
        })();
        let captured = if graph {
            Some(s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0)))
        } else {
            None
        };
        launches.with_context(|| format!("{label} eager/graph launches"))?;
        let graph = captured
            .map(|g| {
                g.with_context(|| format!("{label} graph end"))?
                    .context("empty TCMQA graph")
            })
            .transpose()?;
        if let Some(g) = graph.as_ref() {
            g.launch()?;
        }
        s.synchronize()?;
        let result = s.clone_dtoh(&output)?;
        ensure!(
            result.iter().all(|x| x.is_finite()),
            "nonfinite attention output"
        );
        Ok(result)
    }
}
fn same(a: &[bf16], b: &[bf16], label: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{label}: extent");
    if let Some(i) = a
        .iter()
        .zip(b)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        anyhow::bail!(
            "{label}: word{i} got{} expected{}",
            a[i].to_bits(),
            b[i].to_bits()
        );
    }
    Ok(())
}

#[test]
#[ignore]
fn kv_cache_kernel_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "coordinator GPU window required"
    );
    let (root, out, manifest) = input()?;
    let device = Device::new(0)?;
    device.finish_upload()?;
    ensure!(
        !device.ctx.is_event_tracking(),
        "compact gate requires event tracking disabled before allocation"
    );
    let s = &device.stream;
    let cap = 16640;
    let spec = crate::model_spec::ModelSpec::qwen(
        &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
        &Value::Null,
    )?;
    let spec = crate::mtp::component_spec(&spec)?;
    let smoke = match manifest["run_mode"].as_str() {
        Some("smoke") => true,
        Some("formal") => false,
        _ => anyhow::bail!("fixture run_mode must be smoke/formal"),
    };
    let mut reports = Vec::new();
    let mut defects = Vec::new();
    macro_rules! score {($condition:expr,$label:expr)=>{if !$condition {defects.push(json!({"stage":$label,"class":"DEFECT"}));}};}
    macro_rules! bits {($a:expr,$b:expr,$label:expr)=>{if let Err(error)=same($a,$b,$label) {defects.push(json!({"stage":$label,"class":"DEFECT","difference":error.to_string()}));}};}
    for family in if smoke {
        vec!["synthetic"]
    } else {
        vec!["synthetic", "real"]
    } {
        let k_src = s.clone_htod(&bf16_file(
            root.join(format!("{family}-k.bf16")),
            4 * cap * 256,
        )?)?;
        let v_src = s.clone_htod(&bf16_file(
            root.join(format!("{family}-v.bf16")),
            4 * cap * 256,
        )?)?;
        let q = s.clone_htod(&bf16_file(
            root.join(format!("{family}-q.bf16")),
            2048 * 6144,
        )?)?;
        let k_proj = s.clone_htod(&bf16_file(root.join("writer-k.bf16"), 4 * 1024)?)?;
        let v_proj = s.clone_htod(&bf16_file(root.join("writer-v.bf16"), 4 * 1024)?)?;
        let format = KvFormat::Affine8;
        let codec = "affine8";
        for prefix in if smoke {
            vec![2176usize]
        } else {
            vec![
                2047, 2048, 2049, 2175, 2176, 2177, 2242, 2243, 2244, 4095, 4096, 16320,
            ]
        } {
            let defect_start = defects.len();
            let mut state = StateStore::new_with_format(&spec, cap, format, s)?;
            state.set_sequence(s, 0, prefix as u32, prefix as u32)?;
            let kv = &state.kv[0];
            let c = kv.compact.as_ref().context("compact missing")?;
            c.finish(
                c.view(&kv.k, &kv.v, s),
                state.seq_ref(s)?,
                Some((&k_src, &v_src, prefix)),
                true,
                s,
            )?;
            s.synchronize()?;
            let expected = &manifest["expected"][family][codec][prefix.to_string()];
            let cache_initial = cache_hashes(&device, &state)?;
            score!(
                cache_initial == expected["cache"],
                format!("CPU codec oracle {family}/{codec}/{prefix}")
            );
            state.set_sequence(s, prefix as u32, 4, 4)?;
            let k_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
            let v_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
            let kv = &state.kv[0];
            let c = kv.compact.as_ref().context("compact missing")?;
            c.dequant_prefix(
                c.view(&kv.k, &kv.v, s),
                state.seq_ref(s)?,
                &k_full,
                &v_full,
                prefix,
                s,
            )?;
            let dequant_k = digest(&device, &k_full)?;
            let dequant_v = digest(&device, &v_full)?;
            score!(
                dequant_k == expected["dequant_k"] && dequant_v == expected["dequant_v"],
                format!("CPU dequant oracle {family}/{codec}/{prefix}")
            );
            let mut token = s.alloc_zeros::<u32>(1)?;
            let mut checkpoint = StateCheckpoint::new(&state, &spec, 4, s)?;
            let before = [digest(&device, &kv.k)?, digest(&device, &kv.v)?];
            checkpoint.save(&state, &token, prefix, s, None)?;
            let meta = state.seq_ref(s)?.ptr;
            let cache = c.view(&kv.k, &kv.v, s);
            let module = cubin::module(&device.ctx, "tcmqa_d256")?;
            let write = module.load_function("tcmqa_d256_write_ring")?;
            let k_scatter = module.load_function("tcmqa_d256_rope_k_scatter")?;
            let v_scatter = module.load_function("tcmqa_d256_v_scatter")?;
            let (kp, vp) = (k_proj.device_ptr(s).0, v_proj.device_ptr(s).0);
            let (kf, vf) = (k_full.device_ptr(s).0, v_full.device_ptr(s).0);
            let (stride, head, dim, axes) = (1024i32, (cap * 256) as i32, 256i32, 0u64);
            unsafe {
                s.launch_builder(&write)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&meta)
                    .arg(&axes)
                    .arg(&stride)
                    .arg(&stride)
                    .arg(&cache)
                    .launch(flat(4 * 1024))?;
                s.launch_builder(&k_scatter)
                    .arg(&kp)
                    .arg(&kf)
                    .arg(&meta)
                    .arg(&stride)
                    .arg(&head)
                    .arg(&dim)
                    .launch(flat(4 * 1024))?;
                s.launch_builder(&v_scatter)
                    .arg(&vp)
                    .arg(&vf)
                    .arg(&meta)
                    .arg(&stride)
                    .arg(&head)
                    .arg(&dim)
                    .launch(flat(4 * 1024))?;
            }
            for h in 0..4 {
                for r in 0..4 {
                    let src = (h * cap + prefix + r) * 256;
                    let dst = (h * cache.ring as usize
                        + crate::kv::ring_slot(prefix + r, cache.ring as usize)?)
                        * 256;
                    bits!(
                        &s.clone_dtoh(&kv.k.slice(dst..dst + 256))?,
                        &s.clone_dtoh(&k_full.slice(src..src + 256))?,
                        "K ring writer"
                    );
                    bits!(
                        &s.clone_dtoh(&kv.v.slice(dst..dst + 256))?,
                        &s.clone_dtoh(&v_full.slice(src..src + 256))?,
                        "V ring writer"
                    );
                }
            }
            let linear_k = digest(&device, &k_full)?;
            let equal_axes = s.clone_htod(
                &(0..4)
                    .flat_map(|r| [prefix as i32 + r; 3])
                    .collect::<Vec<_>>(),
            )?;
            let distinct_axes = s.clone_htod(
                &(0..4)
                    .flat_map(|r| {
                        [
                            prefix as i32 + r + 3,
                            prefix as i32 + r + 7,
                            prefix as i32 + r + 13,
                        ]
                    })
                    .collect::<Vec<_>>(),
            )?;
            let graph_axes = equal_axes.device_ptr(s).0;
            let scatter_axes = module.load_function("tcmqa_d256_rope_k_scatter_axes")?;
            for (coordinates, equal) in [(&equal_axes, true), (&distinct_axes, false)] {
                let axes_ptr = coordinates.device_ptr(s).0;
                unsafe {
                    s.launch_builder(&write)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&meta)
                        .arg(&axes_ptr)
                        .arg(&stride)
                        .arg(&stride)
                        .arg(&cache)
                        .launch(flat(4 * 1024))?;
                    s.launch_builder(&scatter_axes)
                        .arg(&kp)
                        .arg(&kf)
                        .arg(&meta)
                        .arg(&axes_ptr)
                        .arg(&stride)
                        .arg(&head)
                        .arg(&dim)
                        .launch(flat(4 * 1024))?;
                }
                ensure!(
                    (digest(&device, &k_full)? == linear_k) == equal,
                    "axes reference negative control invalid"
                );
                for h in 0..4 {
                    for r in 0..4 {
                        let src = (h * cap + prefix + r) * 256;
                        let dst = (h * cache.ring as usize
                            + crate::kv::ring_slot(prefix + r, cache.ring as usize)?)
                            * 256;
                        bits!(
                            &s.clone_dtoh(&kv.k.slice(dst..dst + 256))?,
                            &s.clone_dtoh(&k_full.slice(src..src + 256))?,
                            "axes K ring writer"
                        );
                        bits!(
                            &s.clone_dtoh(&kv.v.slice(dst..dst + 256))?,
                            &s.clone_dtoh(&v_full.slice(src..src + 256))?,
                            "axes V ring writer"
                        );
                    }
                }
            }
            unsafe {
                s.launch_builder(&write)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&meta)
                    .arg(&axes)
                    .arg(&stride)
                    .arg(&stride)
                    .arg(&cache)
                    .launch(flat(4 * 1024))?;
                s.launch_builder(&k_scatter)
                    .arg(&kp)
                    .arg(&kf)
                    .arg(&meta)
                    .arg(&stride)
                    .arg(&head)
                    .arg(&dim)
                    .launch(flat(4 * 1024))?;
            }
            let mut row_checks = Vec::new();
            for rows in 1..=4 {
                let compact = Core {
                    device: &device,
                    q: &q,
                    k: &kv.k,
                    v: &kv.v,
                    compact: Some(cache),
                }
                .run(prefix, rows, 0, false)?;
                let reference = Core {
                    device: &device,
                    q: &q,
                    k: &k_full,
                    v: &v_full,
                    compact: None,
                }
                .run(prefix, rows, 0, false)?;
                bits!(&compact, &reference, "TCMQA dequant vs reconstructed BF16");
                row_checks.push(json!({"rows":rows,"candidate":host_digest(&compact),"reference":host_digest(&reference)}));
                for row in 0..rows as usize {
                    let one = Core {
                        device: &device,
                        q: &q,
                        k: &kv.k,
                        v: &kv.v,
                        compact: Some(cache),
                    }
                    .run(prefix + row, 1, row, false)?;
                    bits!(
                        &compact[row * 6144..(row + 1) * 6144],
                        &one,
                        "TCMQA row invariance"
                    );
                    let control = Core {
                        device: &device,
                        q: &q,
                        k: &k_full,
                        v: &v_full,
                        compact: None,
                    }
                    .run(prefix + row, 1, row, false)?;
                    same(
                        &reference[row * 6144..(row + 1) * 6144],
                        &control,
                        "BF16 oracle row invariance validity",
                    )?;
                }
            }
            let cg = Core {
                device: &device,
                q: &q,
                k: &kv.k,
                v: &kv.v,
                compact: Some(cache),
            }
            .run(prefix, 4, 0, true)?;
            let rg = Core {
                device: &device,
                q: &q,
                k: &k_full,
                v: &v_full,
                compact: None,
            }
            .run(prefix, 4, 0, true)?;
            bits!(&cg, &rg, "TCMQA graph bits");
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .context("KV writer/deferred commit graph begin")?;
            let launches = (|| -> Result<()> {
                unsafe {
                    s.launch_builder(&write)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&meta)
                        .arg(&graph_axes)
                        .arg(&stride)
                        .arg(&stride)
                        .arg(&cache)
                        .launch(flat(4 * 1024))?;
                }
                state.compact_commit(s, false)
            })();
            let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
            launches.context("KV writer/deferred commit graph launches")?;
            let graph = captured
                .context("KV writer/deferred commit graph end")?
                .context("empty KV writer/deferred commit graph")?;
            graph.launch()?;
            s.synchronize()?;
            state.set_sequence(s, prefix as u32 + 4, 1, 4)?;
            state.compact_commit(s, false)?;
            score!(
                s.clone_dtoh(
                    &state.kv[0]
                        .compact
                        .as_ref()
                        .context("compact missing")?
                        .cursor
                )? == [prefix as i32, crate::kv::sealed_until(prefix) as i32, 1, 0],
                "tentative rows sealed"
            );
            checkpoint.restore(&mut state, &mut token, s)?;
            score!(
                before
                    == [
                        digest(&device, &state.kv[0].k)?,
                        digest(&device, &state.kv[0].v)?
                    ],
                "rollback ring bits"
            );
            unsafe {
                s.launch_builder(&write)
                    .arg(&kp)
                    .arg(&vp)
                    .arg(&meta)
                    .arg(&axes)
                    .arg(&stride)
                    .arg(&stride)
                    .arg(&cache)
                    .launch(flat(4 * 1024))?;
            }
            let mut committed_hashes = Vec::new();
            for accepted in 0..=4 {
                state.set_sequence(s, prefix as u32 + accepted, 1, 4)?;
                state.compact_commit(s, true)?;
                let expected = &manifest["expected"][family][codec]
                    [(prefix + accepted as usize).to_string()]["cache"];
                let actual = cache_hashes(&device, &state)?;
                score!(actual == *expected, "committed sealing oracle");
                committed_hashes.push(json!({"prefix":prefix+accepted as usize,"cache":actual}));
            }
            reports.push(json!({"family":family,"codec":codec,"prefix":prefix,"rows":[1,2,3,4],"writer_coordinates":["linear","axes_equal","axes_distinct"],"graph_writer_coordinates":"axes_equal","cache_initial":cache_initial,"dequant_k":dequant_k,"dequant_v":dequant_v,"committed":committed_hashes,"row_checks":row_checks,"graph_candidate":host_digest(&cg),"graph_reference":host_digest(&rg),"checks":["writer_bits","TCMQA_bits","CPU_dequant_bits","commit0..4","rollback_bits"],"defect_indices":(defect_start..defects.len()).collect::<Vec<_>>()}));
        }
        let defect_start = defects.len();
        let prefix = 14336usize;
        let rows = 2048usize;
        let end = prefix + rows;
        let mut state = StateStore::new_with_format(&spec, cap, format, s)?;
        state.set_sequence(s, 0, prefix as u32, prefix as u32)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact missing")?;
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((&k_src, &v_src, prefix)),
            true,
            s,
        )?;
        state.set_sequence(s, prefix as u32, rows as u32, rows as u32)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact missing")?;
        let mut k_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
        let mut v_full = s.alloc_zeros::<bf16>(4 * cap * 256)?;
        c.dequant_prefix(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            &k_full,
            &v_full,
            prefix,
            s,
        )?;
        for h in 0..4 {
            let start = (h * cap + prefix) * 256;
            let stop = (h * cap + end) * 256;
            s.memcpy_dtod(
                &k_src.slice(start..stop),
                &mut k_full.slice_mut(start..stop),
            )?;
            s.memcpy_dtod(
                &v_src.slice(start..stop),
                &mut v_full.slice_mut(start..stop),
            )?;
        }
        let oracle_k = s.clone_htod(&bf16_file(
            root.join(format!("{family}-{codec}-prefill-k.bf16")),
            4 * cap * 256,
        )?)?;
        let oracle_v = s.clone_htod(&bf16_file(
            root.join(format!("{family}-{codec}-prefill-v.bf16")),
            4 * cap * 256,
        )?)?;
        let input_k = digest(&device, &k_full)?;
        let input_v = digest(&device, &v_full)?;
        let reference_k = digest(&device, &oracle_k)?;
        let reference_v = digest(&device, &oracle_v)?;
        ensure!(
            reference_k == manifest["files"][format!("{family}-{codec}-prefill-k.bf16")]["sha256"]
                && reference_v
                    == manifest["files"][format!("{family}-{codec}-prefill-v.bf16")]["sha256"],
            "FA2 uploaded oracle invalid"
        );
        score!(
            input_k == reference_k && input_v == reference_v,
            "FA2 input oracle"
        );
        let mut flash = Flash::new(s.clone(), rows, 24, 4, 256)?;
        let mut result = s.alloc_zeros::<bf16>(rows * 6144)?;
        let mut reference = s.alloc_zeros::<bf16>(rows * 6144)?;
        let shape = Shape {
            rows,
            start: prefix,
            q_heads: 24,
            kv_heads: 4,
            capacity: cap,
            head_dim: 256,
            causal: true,
            window_left: None,
        };
        flash.run(&q, 6144, &k_full, &v_full, &mut result, shape)?;
        flash.run(&q, 6144, &oracle_k, &oracle_v, &mut reference, shape)?;
        let fa_candidate = s.clone_dtoh(&result)?;
        let fa_reference = s.clone_dtoh(&reference)?;
        ensure!(
            fa_candidate
                .iter()
                .chain(&fa_reference)
                .all(|v| v.is_finite()),
            "nonfinite FA2 output"
        );
        bits!(&fa_candidate, &fa_reference, "FA2 reconstructed prefix");
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((&k_full, &v_full, rows)),
            true,
            s,
        )?;
        let cache_final = cache_hashes(&device, &state)?;
        score!(
            cache_final == manifest["expected"][family][codec][end.to_string()]["cache"],
            "prefill sealing oracle"
        );
        reports.push(json!({"family":family,"codec":codec,"prefill_rows":rows,"prefix":prefix,"input_k":input_k,"input_v":input_v,"reference_k":reference_k,"reference_v":reference_v,"candidate":host_digest(&fa_candidate),"reference":host_digest(&fa_reference),"cache_final":cache_final,"checks":["FA2_bits","CPU_sealed_once_bits"],"defect_indices":(defect_start..defects.len()).collect::<Vec<_>>()}));
    }
    ensure!(
        reports.len() == if smoke { 2 } else { 26 },
        "kernel case count"
    );
    std::fs::write(
        out.join("KERNEL.json"),
        serde_json::to_vec_pretty(
            &json!({"status":"KERNEL_VALIDITY_PASS","run_mode":manifest["run_mode"],"cases":reports,"defects":defects,"candidate_pass":defects.is_empty(),"GPU_executed":true}),
        )?,
    )?;
    println!(
        "KV8_KERNEL_VALIDITY_PASS cases={} defects={}",
        reports.len(),
        defects.len()
    );
    Ok(())
}

#[test]
#[ignore = "authorized component GPU window required"]
fn kv_cache_prefix_request_history_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "coordinator GPU window required"
    );
    let (root, out, _) = input()?;
    let device = Device::new(0)?;
    device.finish_upload()?;
    ensure!(
        !device.ctx.is_event_tracking(),
        "prefix history test requires event tracking disabled"
    );
    let s = &device.stream;
    let spec = crate::model_spec::ModelSpec::qwen(
        &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
        &Value::Null,
    )?;
    let spec = crate::mtp::component_spec(&spec)?;
    let cap = 16640;
    let k = s.clone_htod(&bf16_file(root.join("synthetic-k.bf16"), 4 * cap * 256)?)?;
    let v = s.clone_htod(&bf16_file(root.join("synthetic-v.bf16"), 4 * cap * 256)?)?;
    let mut ring = unsafe { device.ctx.alloc_pinned_with_flags::<u8>(1 << 20, 0)? };
    let mut receipts = Vec::new();
    let format = KvFormat::Affine8;
    let mut stores = [
        StateStore::new_with_format(&spec, cap, format, s)?,
        StateStore::new_with_format(&spec, cap, format, s)?,
    ];
    let mut history = Vec::new();
    let mut payloads = Vec::new();
    for (index, state) in stores.iter_mut().enumerate() {
        state.set_sequence(s, 0, 4096, 4096)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact cache missing")?;
        let (old_k, old_v) = if index == 0 { (&k, &v) } else { (&v, &k) };
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((old_k, old_v, 4096)),
            true,
            s,
        )?;
        history.push(cache_hashes(&device, state)?);
        state.reset(s)?;
        state.set_sequence(s, 0, 2177, 2177)?;
        let kv = &state.kv[0];
        let c = kv.compact.as_ref().context("compact cache missing")?;
        c.finish(
            c.view(&kv.k, &kv.v, s),
            state.seq_ref(s)?,
            Some((&k, &v, 2177)),
            true,
            s,
        )?;
        let mut bytes = Vec::new();
        for start in (0..2177).step_by(256) {
            bytes.extend(state.kv[0].compact_prefix_block(
                start..(start + 256).min(2177),
                2177,
                s,
                &mut ring,
                None,
            )?);
        }
        payloads.push(bytes);
    }
    ensure!(
        history[0] != history[1],
        "request-history negative control did not change prior cache"
    );
    let identical = payloads[0] == payloads[1];
    receipts.push(json!({"codec":format.tag(),"prefix":2177,"previous_prefix":4096,"history_control_distinct":true,"payload_sha256":sha256(&payloads[0]),"payload_bytes":payloads[0].len(),"two_state_stores_identical":identical}));
    std::fs::write(
        out.join("PREFIX_HISTORY.json"),
        serde_json::to_vec_pretty(
            &json!({"status":"PREFIX_HISTORY_VALIDITY_PASS","candidate_pass":receipts.iter().all(|r|r["two_state_stores_identical"]==true),"GPU_executed":true,"cases":receipts}),
        )?,
    )?;
    println!(
        "PREFIX_HISTORY_VALIDITY_PASS: affine8; two engine-owned state stores; different previous requests; prefix payload comparison recorded"
    );
    Ok(())
}
