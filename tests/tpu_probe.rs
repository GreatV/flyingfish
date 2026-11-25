#![cfg(feature = "tpu")]

use anyhow::{Context, Result, ensure};
use flyingfish::backend::tpu::{
    TpuDevice,
    hlo::{emit_bf16_attention, emit_bf16_matmul},
    pjrt::{CompiledMemoryStats, HostSemantics},
};
use half::bf16;
use serde::Serialize;
use std::{env, fs, path::PathBuf, thread, time::Instant};

// Frozen pre-run tolerances from 2026-11-25T15:41:36Z PLAN (reports/12.md).
const MATMUL_MAX_ABS_TOL: f64 = 1.6e-2;
const MATMUL_RMSE_TOL: f64 = 5.0e-3;
const MATMUL_REL_L2_TOL: f64 = 4.0e-3;
const MATMUL_NEG_RMSE_FLOOR: f64 = 0.10;

const ATTN_MAX_ABS_TOL: f64 = 1.0e-2;
const ATTN_RMSE_TOL: f64 = 2.0e-3;
const ATTN_REL_L2_TOL: f64 = 1.0e-2;
const ATTN_NEG_UNSCALED_RMSE_FLOOR: f64 = 0.20;
const ATTN_NEG_SHIFT_RMSE_FLOOR: f64 = 0.05;

// Real MiniMax-H3-Image transformer_diffusers/config.json dimensions.
const H3_HIDDEN: usize = 5376;
const H3_HEADS: usize = 56;
const H3_HEAD_DIM: usize = 128;
const H3_ATTN_WIDTH: usize = H3_HEADS * H3_HEAD_DIM; // 7168
const H3_ROWS_1024: usize = 1031; // 32*32 image + 7 text
const H3_ROWS_2048: usize = 4103; // 64*64 image + 7 text

#[derive(Clone, Debug, Serialize)]
struct ErrorMetrics {
    checked_elements: usize,
    all_finite: bool,
    max_abs: f64,
    rmse: f64,
    rel_l2: f64,
}

#[derive(Clone, Debug, Serialize)]
struct OpReport {
    name: String,
    shape: String,
    compile_ms: f64,
    execute_ms: f64,
    compiled_memory_stats: Option<CompiledMemoryStats>,
    metrics: ErrorMetrics,
    passed: bool,
    negative_controls: Vec<NegControlReport>,
}

#[derive(Clone, Debug, Serialize)]
struct NegControlReport {
    name: String,
    rmse: f64,
    min_expected_rmse: f64,
    failed_as_required: bool,
}

#[derive(Clone, Debug, Serialize)]
struct BandwidthRow {
    mode: String,
    size_mib: usize,
    size_bytes: usize,
    min_gib_s: f64,
    median_gib_s: f64,
    max_gib_s: f64,
    min_gb_s: f64,
    median_gb_s: f64,
    max_gb_s: f64,
    samples_ms: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct ProbeSummary {
    device: flyingfish::backend::tpu::DeviceAttributes,
    host_mem_total_kib: Option<u64>,
    host_mem_available_kib: Option<u64>,
    host_cpus: usize,
    op1_matmul: OpReport,
    op2a_attention_1031: OpReport,
    op2b_attention_4103: OpReport,
    roundtrip_64mib_mismatches: usize,
    dma_map_status: String,
    pinned_host_memory_status: String,
    h2d_bandwidth: Vec<BandwidthRow>,
}

fn matmul_x(m: usize, k: usize) -> bf16 {
    let v = ((((m * 37 + k * 53) % 103) as i32 - 51) as f32) / 256.0 + 0.015625;
    bf16::from_f32(v)
}

fn matmul_w(n: usize, k: usize) -> bf16 {
    let v = ((((n * 41 + k * 59) % 109) as i32 - 54) as f32) / 256.0 + 0.015625;
    bf16::from_f32(v)
}

fn attn_q(r: usize, h: usize, d: usize) -> bf16 {
    let ph = (r * 3 + h * 7) % 97;
    bf16::from_f32(((((ph + d * 13) % 97) as i32 - 48) as f32) / 48.0)
}

fn attn_k(k: usize, h: usize, d: usize) -> bf16 {
    let ph = (k * 5 + h * 11) % 97;
    bf16::from_f32(((((ph + d * 13) % 97) as i32 - 48) as f32) / 48.0)
}

fn attn_v(k: usize, h: usize, d: usize) -> bf16 {
    let ph = (k * 5 + h * 11) % 97;
    bf16::from_f32(((((ph * 7 + h * 23 + d * 19) % 103) as i32 - 51) as f32) / 51.0)
}

fn read_meminfo_kib(key: &str) -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            return rest.split_whitespace().next()?.parse::<u64>().ok();
        }
    }
    None
}

fn eval_matmul_op1(dev: &TpuDevice) -> Result<OpReport> {
    let (m, k, n) = (H3_ROWS_2048, H3_HIDDEN, H3_ATTN_WIDTH);
    let mut x = vec![bf16::ZERO; m * k];
    for row in 0..m {
        for col in 0..k {
            x[row * k + col] = matmul_x(row, col);
        }
    }
    let mut w = vec![bf16::ZERO; n * k];
    for row in 0..n {
        for col in 0..k {
            w[row * k + col] = matmul_w(row, col);
        }
    }

    let mlir = emit_bf16_matmul(m, k, n)?;
    let t_compile = Instant::now();
    let exec = dev.compile_mlir(&mlir)?;
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    let x_buf = dev.upload_bf16(
        &x,
        &[m as i64, k as i64],
        HostSemantics::ImmutableUntilTransferCompletes,
    )?;
    let w_buf = dev.upload_bf16(
        &w,
        &[n as i64, k as i64],
        HostSemantics::ImmutableUntilTransferCompletes,
    )?;

    let t_exec = Instant::now();
    let outs = exec.execute(&[&x_buf, &w_buf], &[m * n * 2])?;
    let execute_ms = t_exec.elapsed().as_secs_f64() * 1e3;
    let got = outs[0].download_bf16()?;
    ensure!(
        got.len() == m * n,
        "op1 output len {} != {}",
        got.len(),
        m * n
    );

    let x_f64: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
    let w_f64: Vec<f64> = w.iter().map(|v| v.to_f64()).collect();
    let workers = thread::available_parallelism().map_or(1, usize::from);
    let chunk = m.div_ceil(workers);

    let (all_finite, max_abs, sq_err, sq_ref, neg_sq_err) = thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..m).step_by(chunk) {
            let end = (start + chunk).min(m);
            let x_ref = &x_f64;
            let w_ref = &w_f64;
            let got_ref = &got;
            handles.push(scope.spawn(move || {
                let mut finite = true;
                let mut m_abs = 0.0f64;
                let mut se = 0.0f64;
                let mut sr = 0.0f64;
                let mut neg_se = 0.0f64;
                for row in start..end {
                    let xr = &x_ref[row * k..(row + 1) * k];
                    let gr = &got_ref[row * n..(row + 1) * n];
                    for col in 0..n {
                        let wr = &w_ref[col * k..(col + 1) * k];
                        let mut acc = 0.0f64;
                        let mut acc_shift = 0.0f64;
                        for kk in 0..k - 1 {
                            let xv = xr[kk];
                            acc += xv * wr[kk];
                            acc_shift += xv * wr[kk + 1];
                        }
                        acc += xr[k - 1] * wr[k - 1];
                        acc_shift += xr[k - 1] * wr[0];
                        let g = gr[col].to_f64();
                        if !g.is_finite() {
                            finite = false;
                        }
                        let err = (g - acc).abs();
                        if err > m_abs {
                            m_abs = err;
                        }
                        se += err * err;
                        sr += acc * acc;
                        let nerr = g - acc_shift;
                        neg_se += nerr * nerr;
                    }
                }
                (finite, m_abs, se, sr, neg_se)
            }));
        }
        let mut fin = true;
        let mut m_abs = 0.0f64;
        let mut se = 0.0f64;
        let mut sr = 0.0f64;
        let mut neg_se = 0.0f64;
        for h in handles {
            let (f, ma, e, r, ne) = h.join().expect("op1 worker panicked");
            fin &= f;
            m_abs = m_abs.max(ma);
            se += e;
            sr += r;
            neg_se += ne;
        }
        (fin, m_abs, se, sr, neg_se)
    });

    let count = m * n;
    let metrics = ErrorMetrics {
        checked_elements: count,
        all_finite,
        max_abs,
        rmse: (sq_err / count as f64).sqrt(),
        rel_l2: (sq_err / sq_ref).sqrt(),
    };
    let neg_rmse = (neg_sq_err / count as f64).sqrt();
    let neg = NegControlReport {
        name: "w_k_shift_plus_1".into(),
        rmse: neg_rmse,
        min_expected_rmse: MATMUL_NEG_RMSE_FLOOR,
        failed_as_required: neg_rmse > MATMUL_NEG_RMSE_FLOOR,
    };
    let passed = metrics.all_finite
        && metrics.max_abs <= MATMUL_MAX_ABS_TOL
        && metrics.rmse <= MATMUL_RMSE_TOL
        && metrics.rel_l2 <= MATMUL_REL_L2_TOL
        && neg.failed_as_required;
    println!(
        "TPU_PROBE_OP1 shape=[{m},{k}]x[{n},{k}]^T compile_ms={compile_ms:.2} exec_ms={execute_ms:.2} checked={count} finite={} max_abs={:.6e} rmse={:.6e} rel_l2={:.6e} neg_rmse={:.6e} pass={passed}",
        metrics.all_finite, metrics.max_abs, metrics.rmse, metrics.rel_l2, neg.rmse
    );
    Ok(OpReport {
        name: "h3_image_to_q_bf16_matmul".into(),
        shape: format!("[{m}, {k}] x [{n}, {k}]^T -> [{m}, {n}]"),
        compile_ms,
        execute_ms,
        compiled_memory_stats: exec.compiled_memory_stats.clone(),
        metrics,
        passed,
        negative_controls: vec![neg],
    })
}

fn eval_attention_op2(
    dev: &TpuDevice,
    name: &str,
    rows: usize,
    sampled_rows: Option<Vec<usize>>,
    run_neg_controls: bool,
) -> Result<OpReport> {
    let (heads, d) = (H3_HEADS, H3_HEAD_DIM);
    let total = rows * heads * d;
    let mut q = vec![bf16::ZERO; total];
    let mut k = vec![bf16::ZERO; total];
    let mut v = vec![bf16::ZERO; total];
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * d;
            for dim in 0..d {
                q[base + dim] = attn_q(r, h, dim);
                k[base + dim] = attn_k(r, h, dim);
                v[base + dim] = attn_v(r, h, dim);
            }
        }
    }

    let scale = 1.0f32 / (d as f32).sqrt();
    let mlir = emit_bf16_attention(rows, heads, d, scale)?;
    let t_compile = Instant::now();
    let exec = dev.compile_mlir(&mlir)?;
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    let dims = [rows as i64, heads as i64, d as i64];
    let q_buf = dev.upload_bf16(&q, &dims, HostSemantics::ImmutableUntilTransferCompletes)?;
    let k_buf = dev.upload_bf16(&k, &dims, HostSemantics::ImmutableUntilTransferCompletes)?;
    let v_buf = dev.upload_bf16(&v, &dims, HostSemantics::ImmutableUntilTransferCompletes)?;

    let t_exec = Instant::now();
    let outs = exec.execute(&[&q_buf, &k_buf, &v_buf], &[total * 2])?;
    let execute_ms = t_exec.elapsed().as_secs_f64() * 1e3;
    let got = outs[0].download_bf16()?;
    ensure!(
        got.len() == total,
        "{name} output len {} != {total}",
        got.len()
    );

    let all_finite = got.iter().all(|x| x.is_finite());
    let check_rows: Vec<usize> = sampled_rows.unwrap_or_else(|| (0..rows).collect());
    let q_f64: Vec<f64> = q.iter().map(|x| x.to_f64()).collect();
    let k_f64: Vec<f64> = k.iter().map(|x| x.to_f64()).collect();
    let v_f64: Vec<f64> = v.iter().map(|x| x.to_f64()).collect();
    let scale_f64 = 1.0f64 / (d as f64).sqrt();

    let (max_abs, sq_err, sq_ref, neg_unscaled_se, neg_head_se, neg_vshift_se) =
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for h in 0..heads {
                let q_ref = &q_f64;
                let k_ref = &k_f64;
                let v_ref = &v_f64;
                let got_ref = &got;
                let rows_ref = &check_rows;
                handles.push(scope.spawn(move || {
                    let mut m_abs = 0.0f64;
                    let mut se = 0.0f64;
                    let mut sr = 0.0f64;
                    let mut nu_se = 0.0f64;
                    let mut nh_se = 0.0f64;
                    let mut nv_se = 0.0f64;
                    let mut scores = vec![0.0f64; rows];
                    let mut probs = vec![0.0f64; rows];
                    let mut probs_unscaled = if run_neg_controls {
                        vec![0.0f64; rows]
                    } else {
                        Vec::new()
                    };
                    let mut probs_head = if run_neg_controls {
                        vec![0.0f64; rows]
                    } else {
                        Vec::new()
                    };
                    let h_alt = (h + 1) % heads;

                    for &rq in rows_ref {
                        let q_off = (rq * heads + h) * d;
                        let mut peak = f64::NEG_INFINITY;
                        let mut peak_un = f64::NEG_INFINITY;
                        for (rk, score) in scores.iter_mut().enumerate() {
                            let k_off = (rk * heads + h) * d;
                            let mut dot = 0.0f64;
                            for dim in 0..d {
                                dot += q_ref[q_off + dim] * k_ref[k_off + dim];
                            }
                            let s = dot * scale_f64;
                            *score = s;
                            if s > peak {
                                peak = s;
                            }
                            if run_neg_controls && dot > peak_un {
                                peak_un = dot;
                            }
                        }
                        let mut sum = 0.0f64;
                        for (p, &s) in probs.iter_mut().zip(&scores) {
                            let e = (s - peak).exp();
                            *p = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        for p in &mut probs {
                            *p *= inv;
                        }

                        if run_neg_controls {
                            let mut sum_un = 0.0f64;
                            for (p, &s) in probs_unscaled.iter_mut().zip(&scores) {
                                let e = (s / scale_f64 - peak_un).exp();
                                *p = e;
                                sum_un += e;
                            }
                            for p in &mut probs_unscaled {
                                *p /= sum_un;
                            }

                            let mut peak_h = f64::NEG_INFINITY;
                            for (rk, p) in probs_head.iter_mut().enumerate() {
                                let k_off = (rk * heads + h_alt) * d;
                                let mut dot = 0.0f64;
                                for dim in 0..d {
                                    dot += q_ref[q_off + dim] * k_ref[k_off + dim];
                                }
                                let s = dot * scale_f64;
                                *p = s;
                                if s > peak_h {
                                    peak_h = s;
                                }
                            }
                            let mut sum_h = 0.0f64;
                            for p in &mut probs_head {
                                *p = (*p - peak_h).exp();
                                sum_h += *p;
                            }
                            for p in &mut probs_head {
                                *p /= sum_h;
                            }
                        }

                        let mut out_ref = [0.0f64; H3_HEAD_DIM];
                        let mut out_un = [0.0f64; H3_HEAD_DIM];
                        let mut out_hd = [0.0f64; H3_HEAD_DIM];
                        let mut out_vs = [0.0f64; H3_HEAD_DIM];
                        for (rk, &p) in probs.iter().enumerate() {
                            let v_off = (rk * heads + h) * d;
                            let vs_rk = if rk + 1 == rows { 0 } else { rk + 1 };
                            let vs_off = (vs_rk * heads + h) * d;
                            for dim in 0..d {
                                let vv = v_ref[v_off + dim];
                                out_ref[dim] += p * vv;
                                if run_neg_controls {
                                    out_un[dim] += probs_unscaled[rk] * vv;
                                    out_hd[dim] += probs_head[rk] * vv;
                                    out_vs[dim] += p * v_ref[vs_off + dim];
                                }
                            }
                        }

                        for dim in 0..d {
                            let g = got_ref[q_off + dim].to_f64();
                            let r_val = out_ref[dim];
                            let err = (g - r_val).abs();
                            if err > m_abs {
                                m_abs = err;
                            }
                            se += err * err;
                            sr += r_val * r_val;
                            if run_neg_controls {
                                let du = g - out_un[dim];
                                let dh = g - out_hd[dim];
                                let dv = g - out_vs[dim];
                                nu_se += du * du;
                                nh_se += dh * dh;
                                nv_se += dv * dv;
                            }
                        }
                    }
                    (m_abs, se, sr, nu_se, nh_se, nv_se)
                }));
            }
            let mut m_abs = 0.0f64;
            let mut se = 0.0f64;
            let mut sr = 0.0f64;
            let mut nu = 0.0f64;
            let mut nh = 0.0f64;
            let mut nv = 0.0f64;
            for h in handles {
                let (ma, e, r, u, hd, vs) = h.join().expect("attention worker panicked");
                m_abs = m_abs.max(ma);
                se += e;
                sr += r;
                nu += u;
                nh += hd;
                nv += vs;
            }
            (m_abs, se, sr, nu, nh, nv)
        });

    let checked = check_rows.len() * heads * d;
    let metrics = ErrorMetrics {
        checked_elements: checked,
        all_finite,
        max_abs,
        rmse: (sq_err / checked as f64).sqrt(),
        rel_l2: (sq_err / sq_ref).sqrt(),
    };
    let mut negs = Vec::new();
    if run_neg_controls {
        let u_rmse = (neg_unscaled_se / checked as f64).sqrt();
        let h_rmse = (neg_head_se / checked as f64).sqrt();
        let v_rmse = (neg_vshift_se / checked as f64).sqrt();
        negs.push(NegControlReport {
            name: "missing_inv_sqrt_d_scale".into(),
            rmse: u_rmse,
            min_expected_rmse: ATTN_NEG_UNSCALED_RMSE_FLOOR,
            failed_as_required: u_rmse > ATTN_NEG_UNSCALED_RMSE_FLOOR,
        });
        negs.push(NegControlReport {
            name: "k_head_shift_plus_1".into(),
            rmse: h_rmse,
            min_expected_rmse: ATTN_NEG_SHIFT_RMSE_FLOOR,
            failed_as_required: h_rmse > ATTN_NEG_SHIFT_RMSE_FLOOR,
        });
        negs.push(NegControlReport {
            name: "v_key_shift_plus_1".into(),
            rmse: v_rmse,
            min_expected_rmse: ATTN_NEG_SHIFT_RMSE_FLOOR,
            failed_as_required: v_rmse > ATTN_NEG_SHIFT_RMSE_FLOOR,
        });
    }
    let passed = metrics.all_finite
        && metrics.max_abs <= ATTN_MAX_ABS_TOL
        && metrics.rmse <= ATTN_RMSE_TOL
        && metrics.rel_l2 <= ATTN_REL_L2_TOL
        && negs.iter().all(|n| n.failed_as_required);
    println!(
        "TPU_PROBE_OP2 name={name} shape=[{rows},{heads},{d}] compile_ms={compile_ms:.2} exec_ms={execute_ms:.2} checked={checked}/{total} finite={} max_abs={:.6e} rmse={:.6e} rel_l2={:.6e} negs_ok={} pass={passed}",
        metrics.all_finite,
        metrics.max_abs,
        metrics.rmse,
        metrics.rel_l2,
        negs.iter().all(|n| n.failed_as_required)
    );
    Ok(OpReport {
        name: name.into(),
        shape: format!("[{rows}, {heads}, {d}]"),
        compile_ms,
        execute_ms,
        compiled_memory_stats: exec.compiled_memory_stats.clone(),
        metrics,
        passed,
        negative_controls: negs,
    })
}

fn coprime_sample_rows(total_rows: usize, count: usize) -> Vec<usize> {
    let mut rows = Vec::with_capacity(count);
    rows.push(0);
    rows.push(total_rows - 1);
    let mut cur = 17usize;
    while rows.len() < count {
        if !rows.contains(&cur) {
            rows.push(cur);
        }
        cur = (cur + 37) % total_rows;
    }
    rows.sort_unstable();
    rows
}

fn summarize_bw(mode: &str, size_mib: usize, samples_s: Vec<f64>) -> BandwidthRow {
    let size_bytes = size_mib * 1024 * 1024;
    let gib = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gb = size_bytes as f64 / 1e9;
    let mut rates_gib: Vec<f64> = samples_s.iter().map(|&s| gib / s).collect();
    let mut rates_gb: Vec<f64> = samples_s.iter().map(|&s| gb / s).collect();
    rates_gib.sort_by(|a, b| a.total_cmp(b));
    rates_gb.sort_by(|a, b| a.total_cmp(b));
    let mid = rates_gib.len() / 2;
    let row = BandwidthRow {
        mode: mode.into(),
        size_mib,
        size_bytes,
        min_gib_s: rates_gib[0],
        median_gib_s: rates_gib[mid],
        max_gib_s: *rates_gib.last().expect("non-empty bw samples"),
        min_gb_s: rates_gb[0],
        median_gb_s: rates_gb[mid],
        max_gb_s: *rates_gb.last().expect("non-empty bw samples"),
        samples_ms: samples_s.iter().map(|&s| s * 1e3).collect(),
    };
    println!(
        "TPU_PROBE_H2D mode={mode} size_mib={size_mib} min_gib_s={:.3} median_gib_s={:.3} max_gib_s={:.3} min_gb_s={:.3} median_gb_s={:.3} max_gb_s={:.3}",
        row.min_gib_s, row.median_gib_s, row.max_gib_s, row.min_gb_s, row.median_gb_s, row.max_gb_s
    );
    row
}

fn measure_h2d_bandwidth(dev: &TpuDevice) -> Result<(usize, String, String, Vec<BandwidthRow>)> {
    let rt_bytes = 64 * 1024 * 1024;
    let mut rt_src = vec![0u8; rt_bytes];
    for (i, b) in rt_src.iter_mut().enumerate() {
        *b = ((i * 131 + 17) & 0xFF) as u8;
    }
    let rt_buf = dev.upload_u8(
        &rt_src,
        HostSemantics::ImmutableUntilTransferCompletes,
        None,
    )?;
    let _on_device_bytes = rt_buf.on_device_size_in_bytes()?;
    let rt_back = rt_buf.download_bytes()?;
    drop(rt_buf);
    let mismatches = rt_src
        .iter()
        .zip(rt_back.iter())
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        mismatches == 0,
        "64 MiB H2D->D2H round-trip had {mismatches} mismatches"
    );

    let sizes_mib = [64usize, 128, 256, 512, 1024];
    let mut rows = Vec::new();
    let max_bytes = 1024 * 1024 * 1024;
    let mut pageable = vec![0u8; max_bytes];
    for i in (0..max_bytes).step_by(4096) {
        pageable[i] = ((i >> 12) & 0xFF) as u8;
    }

    for (mode, sem) in [
        (
            "pageable_immutable_during_call",
            HostSemantics::ImmutableOnlyDuringCall,
        ),
        (
            "pageable_immutable_until_complete",
            HostSemantics::ImmutableUntilTransferCompletes,
        ),
    ] {
        for &mib in &sizes_mib {
            let slice = &pageable[..mib * 1024 * 1024];
            let warm = dev.upload_u8(slice, sem, None)?;
            drop(warm);
            let mut times = Vec::with_capacity(5);
            for _ in 0..5 {
                let t0 = Instant::now();
                let buf = dev.upload_u8(slice, sem, None)?;
                times.push(t0.elapsed().as_secs_f64());
                drop(buf);
            }
            rows.push(summarize_bw(mode, mib, times));
        }
    }

    let mut dma_notes = Vec::new();
    for &mib in &sizes_mib {
        let size = mib * 1024 * 1024;
        match dev.alloc_dma_mapped(size) {
            Ok(mut dma) => {
                if dma_notes.is_empty() {
                    dma_notes.push(format!(
                        "ok mlocked={} mlock_errno={}",
                        dma.mlocked, dma.mlock_errno
                    ));
                }
                for i in (0..size).step_by(4096) {
                    dma.as_mut_slice()[i] = ((i >> 12) & 0xFF) as u8;
                }
                let slice = dma.as_slice();
                let warm =
                    dev.upload_u8(slice, HostSemantics::ImmutableUntilTransferCompletes, None)?;
                drop(warm);
                let mut times = Vec::with_capacity(5);
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let buf =
                        dev.upload_u8(slice, HostSemantics::ImmutableUntilTransferCompletes, None)?;
                    times.push(t0.elapsed().as_secs_f64());
                    drop(buf);
                }
                rows.push(summarize_bw("dma_mapped_pinned", mib, times));
            }
            Err(err) => {
                let msg = format!("error_at_{mib}mib: {err:#}");
                println!("TPU_PROBE_DMA_MAP {msg}");
                dma_notes.push(msg);
                break;
            }
        }
    }
    let dma_map_status = dma_notes.join("; ");

    let pinned_mem = dev
        .attributes
        .addressable_memories
        .iter()
        .find(|m| m.kind == "pinned_host");
    let pinned_host_memory_status = match pinned_mem {
        Some(mem_info) => {
            let mut status = "ok".to_string();
            for &mib in &sizes_mib {
                let slice = &pageable[..mib * 1024 * 1024];
                match dev.upload_u8(
                    slice,
                    HostSemantics::ImmutableUntilTransferCompletes,
                    Some(mem_info),
                ) {
                    Ok(host_buf) => {
                        let warm = dev.copy_buffer_to_default_hbm(&host_buf)?;
                        drop(warm);
                        let mut times = Vec::with_capacity(5);
                        for _ in 0..5 {
                            let t0 = Instant::now();
                            let dev_buf = dev.copy_buffer_to_default_hbm(&host_buf)?;
                            times.push(t0.elapsed().as_secs_f64());
                            drop(dev_buf);
                        }
                        rows.push(summarize_bw("pinned_host_memory_space", mib, times));
                    }
                    Err(err) => {
                        status = format!("error_at_{mib}mib: {err:#}");
                        println!("TPU_PROBE_PINNED_HOST {status}");
                        break;
                    }
                }
            }
            status
        }
        None => {
            let kinds: Vec<&str> = dev
                .attributes
                .addressable_memories
                .iter()
                .map(|m| m.kind.as_str())
                .collect();
            let msg = format!("absent (addressable_memories={kinds:?})");
            println!("TPU_PROBE_PINNED_HOST {msg}");
            msg
        }
    };

    Ok((mismatches, dma_map_status, pinned_host_memory_status, rows))
}

#[test]
#[ignore = "TPU; requires FF_PJRT_LIB pointing to libtpu.so"]
fn tpu_probe_v5e1() -> Result<()> {
    let lib_path = PathBuf::from(env::var("FF_PJRT_LIB").context("FF_PJRT_LIB env var missing")?);
    let dev = TpuDevice::open(&lib_path, 0)?;
    let _arch = dev.architecture();
    let _live_stats = dev.memory_stats()?;
    let host_mem_total_kib = read_meminfo_kib("MemTotal:");
    let host_mem_available_kib = read_meminfo_kib("MemAvailable:");
    let host_cpus = thread::available_parallelism().map_or(1, usize::from);

    println!(
        "TPU_PROBE_DEVICE pjrt={}.{} struct_size={} platform={} chip={:?} total_devices={} addressable={} hbm_limit={:?} hbm_reservable={:?} host_ram_kib={:?} host_cpus={}",
        dev.attributes.pjrt_major,
        dev.attributes.pjrt_minor,
        dev.attributes.pjrt_api_struct_size,
        dev.attributes.platform_name,
        dev.attributes.chip_kind,
        dev.attributes.total_devices,
        dev.attributes.addressable_devices,
        dev.attributes.memory_stats.bytes_limit,
        dev.attributes.memory_stats.bytes_reservable_limit,
        host_mem_total_kib,
        host_cpus
    );

    let op1 = eval_matmul_op1(&dev)?;
    let op2a = eval_attention_op2(&dev, "h3_image_attn_1031", H3_ROWS_1024, None, true)?;
    let sampled_4103 = coprime_sample_rows(H3_ROWS_2048, 256);
    let op2b = eval_attention_op2(
        &dev,
        "h3_image_attn_4103",
        H3_ROWS_2048,
        Some(sampled_4103),
        false,
    )?;
    let (roundtrip_64mib_mismatches, dma_map_status, pinned_host_memory_status, h2d_bandwidth) =
        measure_h2d_bandwidth(&dev)?;

    let summary = ProbeSummary {
        device: dev.attributes.clone(),
        host_mem_total_kib,
        host_mem_available_kib,
        host_cpus,
        op1_matmul: op1.clone(),
        op2a_attention_1031: op2a.clone(),
        op2b_attention_4103: op2b.clone(),
        roundtrip_64mib_mismatches,
        dma_map_status,
        pinned_host_memory_status,
        h2d_bandwidth,
    };

    if let Ok(out_path) = env::var("FF_TPU_PROBE_OUT") {
        let json = serde_json::to_string_pretty(&summary)?;
        fs::write(&out_path, json).with_context(|| format!("write {out_path}"))?;
    }

    ensure!(
        op1.passed,
        "Op1 BF16 matmul failed frozen tolerances or negative control"
    );
    ensure!(
        op2a.passed,
        "Op2a BF16 attention (R=1031) failed frozen tolerances or negative controls"
    );
    ensure!(
        op2b.passed,
        "Op2b BF16 attention (R=4103) failed frozen tolerances"
    );
    println!(
        "TPU_PROBE_PASS op1=PASS op2a=PASS op2b=PASS roundtrip_mismatches={roundtrip_64mib_mismatches}"
    );
    Ok(())
}
