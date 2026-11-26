#![cfg(feature = "tpu")]

use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::tpu::{
        TpuDevice,
        dit_block::{
            BlockRunInputs, StreamedDitBlock, TpuAdalnCache, compile_h3_dit_block,
            compute_block_adaln_table, load_block_active_weights_from_shard,
            load_block_adaln_proj_from_shard, load_time_embedder_from_shard,
        },
        hlo::{emit_bf16_attention, emit_bf16_matmul},
        pjrt::{CompiledMemoryStats, HostSemantics},
        plan::{BlockPlacement, ModelFootprint, TpuPlacementPlan, TpuTopology},
    },
    minimax_h3::{
        AdalnPlan, ImageSchedule, ImageSpec, TEXT_TAG, image_latent_extent, image_layout,
        image_row_plan, rotary_rows,
    },
};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

// Frozen op-level tolerances (Step 1).
const MATMUL_MAX_ABS_TOL: f64 = 1.6e-2;
const MATMUL_RMSE_TOL: f64 = 5.0e-3;
const MATMUL_REL_L2_TOL: f64 = 4.0e-3;
const MATMUL_NEG_RMSE_FLOOR: f64 = 0.10;

const ATTN_MAX_ABS_TOL: f64 = 1.0e-2;
const ATTN_RMSE_TOL: f64 = 2.0e-3;
const ATTN_REL_L2_TOL: f64 = 1.0e-2;
const ATTN_NEG_UNSCALED_RMSE_FLOOR: f64 = 0.20;
const ATTN_NEG_SHIFT_RMSE_FLOOR: f64 = 0.05;

// Frozen from local CPU model-definition ref_bf16 vs ref_f32 floor (R=1031: rel_l2=5.134699e-3,
// rel_max_abs=4.553578e-3; R=4103: rel_l2=5.607429e-3, rel_max_abs=6.719102e-3).
// Multiples of max floor: 3.03x for rel_l2 (1.70e-2), 4.02x for rel_max_abs (2.70e-2).
const DIT_REL_L2_TOL: f64 = 1.70e-2;
const DIT_REL_MAX_ABS_TOL: f64 = 2.70e-2;
const ROPE_COS_SIN_CONTROL_MULT: f64 = 4.0;
const ROPE_HW_CONTROL_MULT: f64 = 1.0;
const ADALN_CONTROL_MULT: f64 = 6.0;
const SWIGLU_CONTROL_MULT: f64 = 6.0;

const DIT_HEAD_CHUNK: usize = 2;
const H3_HIDDEN: usize = 5376;
const H3_HEADS: usize = 56;
const H3_HEAD_DIM: usize = 128;
const H3_ATTN_WIDTH: usize = H3_HEADS * H3_HEAD_DIM;
const H3_ROWS_1024: usize = 1031;
const H3_ROWS_2048: usize = 4103;
const HBM_RELEASE_SLACK_BYTES: i64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
struct BlockMetrics {
    elements: usize,
    all_finite: bool,
    max_abs: f64,
    rmse: f64,
    ref_rms: f64,
    ref_max_abs: f64,
    rel_l2: f64,
    rel_max_abs: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ControlReport {
    name: String,
    rel_l2: f64,
    rel_max_abs: f64,
    min_expected_rel_l2: f64,
    failed_as_required: bool,
}

#[derive(Clone, Debug, Serialize)]
struct BlockCaseReport {
    canvas: usize,
    rows: usize,
    head_chunk: usize,
    adaln_pairs: usize,
    compile_ms: f64,
    h2d_ms: f64,
    execute_ms: f64,
    active_weight_bytes: u64,
    used_pinned_host_copy: bool,
    compiled_memory_stats: Option<CompiledMemoryStats>,
    hbm_before_bytes: i64,
    hbm_after_drop_bytes: i64,
    hbm_peak_bytes: Option<i64>,
    metrics: BlockMetrics,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Step2Summary {
    device: flyingfish::backend::tpu::DeviceAttributes,
    tolerances: ToleranceSummary,
    adaln_first_recomputed: bool,
    adaln_second_recomputed: bool,
    cases: Vec<BlockCaseReport>,
    controls_1031: Vec<ControlReport>,
    live_topology: TpuTopology,
    live_plan: TpuPlacementPlan,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ToleranceSummary {
    rel_l2_tol: f64,
    rel_max_abs_tol: f64,
    rope_control_mult: f64,
    adaln_control_mult: f64,
    swiglu_control_mult: f64,
}

struct CaseFixture {
    x: Vec<bf16>,
    ref_bf16: Vec<bf16>,
    position_ids: Vec<[f64; 3]>,
    token_tags: Vec<u8>,
    timesteps: Vec<f32>,
    step_index: usize,
}

fn env_path(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("{name} env var is required"))
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

fn compute_metrics(cand: &[bf16], reference: &[bf16]) -> Result<BlockMetrics> {
    ensure!(
        cand.len() == reference.len() && !cand.is_empty(),
        "metric slice length mismatch {} vs {}",
        cand.len(),
        reference.len()
    );
    let mut all_finite = true;
    let mut max_abs = 0.0f64;
    let mut ref_max_abs = 0.0f64;
    let mut sq_err = 0.0f64;
    let mut sq_ref = 0.0f64;
    for (&c, &r) in cand.iter().zip(reference) {
        let cf = c.to_f64();
        let rf = r.to_f64();
        if !cf.is_finite() || !rf.is_finite() {
            all_finite = false;
        }
        let d = (cf - rf).abs();
        if d > max_abs {
            max_abs = d;
        }
        let ra = rf.abs();
        if ra > ref_max_abs {
            ref_max_abs = ra;
        }
        sq_err += d * d;
        sq_ref += rf * rf;
    }
    ensure!(ref_max_abs > 0.0 && sq_ref > 0.0, "zero reference tensor");
    let n = cand.len() as f64;
    Ok(BlockMetrics {
        elements: cand.len(),
        all_finite,
        max_abs,
        rmse: (sq_err / n).sqrt(),
        ref_rms: (sq_ref / n).sqrt(),
        ref_max_abs,
        rel_l2: (sq_err / sq_ref).sqrt(),
        rel_max_abs: max_abs / ref_max_abs,
    })
}

fn load_case_fixture(path: &Path, expected_rows: usize, hidden: usize) -> Result<CaseFixture> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: read-only memory map of a frozen test fixture file.
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse safetensors {}", path.display()))?;

    let read_bf16 = |name: &str, shape: &[usize]| -> Result<Vec<bf16>> {
        let v = st.tensor(name).with_context(|| format!("missing {name}"))?;
        ensure!(
            v.dtype() == Dtype::BF16 && v.shape() == shape,
            "{name} dtype/shape mismatch"
        );
        Ok(v.data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16::from_le_bytes(c))
            .collect())
    };
    let x = read_bf16("x", &[expected_rows, hidden])?;
    let ref_bf16 = read_bf16("ref_bf16", &[expected_rows, hidden])?;

    let pos_view = st.tensor("position_ids").context("missing position_ids")?;
    ensure!(
        pos_view.dtype() == Dtype::F64 && pos_view.shape() == [expected_rows, 3],
        "position_ids dtype/shape mismatch"
    );
    let pos_f64: Vec<f64> = pos_view
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| f64::from_le_bytes(c))
        .collect();
    let position_ids: Vec<[f64; 3]> = pos_f64.as_chunks::<3>().0.to_vec();

    let tags_view = st.tensor("token_tags").context("missing token_tags")?;
    ensure!(
        tags_view.dtype() == Dtype::U8 && tags_view.shape() == [expected_rows],
        "token_tags dtype/shape mismatch"
    );
    let token_tags = tags_view.data().to_vec();

    let ts_view = st.tensor("timesteps").context("missing timesteps")?;
    ensure!(
        ts_view.dtype() == Dtype::F32 && ts_view.shape().len() == 1,
        "timesteps dtype/shape mismatch"
    );
    let timesteps: Vec<f32> = ts_view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();

    let step_view = st.tensor("step_index").context("missing step_index")?;
    ensure!(
        step_view.dtype() == Dtype::I32 && step_view.shape() == [1],
        "step_index dtype/shape mismatch"
    );
    let step_i32 = i32::from_le_bytes(
        step_view.data()[..4]
            .try_into()
            .context("step_index bytes")?,
    );
    let step_index = usize::try_from(step_i32).context("negative step_index")?;

    Ok(CaseFixture {
        x,
        ref_bf16,
        position_ids,
        token_tags,
        timesteps,
        step_index,
    })
}

fn check_live_plan(
    topology: &TpuTopology,
    plan: &TpuPlacementPlan,
    spec: &ImageSpec,
    active_block_bytes: u64,
) -> Result<()> {
    ensure!(
        plan.blocks.len() == spec.layers,
        "live plan layers {} != {}",
        plan.blocks.len(),
        spec.layers
    );
    ensure!(
        plan.hbm_peak_accounted_bytes() <= plan.usable_hbm_bytes,
        "live plan HBM {} > usable {}",
        plan.hbm_peak_accounted_bytes(),
        plan.usable_hbm_bytes
    );
    ensure!(
        plan.host_staged_bytes <= plan.usable_host_bytes,
        "live plan host {} > usable {}",
        plan.host_staged_bytes,
        plan.usable_host_bytes
    );
    let resident = plan.count(BlockPlacement::HbmResident);
    let pinned = plan.count(BlockPlacement::PinnedHostStreamed);
    let pageable = plan.count(BlockPlacement::PageableHostStreamed);
    let disk = plan.count(BlockPlacement::DiskStreamed);
    ensure!(
        resident + pinned + pageable + disk == spec.layers,
        "live plan tier sum mismatch"
    );
    if topology.has_pinned_host {
        ensure!(
            pageable == 0,
            "live device exposes pinned_host yet assigned {pageable} pageable blocks"
        );
    }
    let host_tier = if topology.has_pinned_host {
        BlockPlacement::PinnedHostStreamed
    } else {
        BlockPlacement::PageableHostStreamed
    };
    let host_count = pinned + pageable;
    ensure!(
        plan.blocks[..resident]
            .iter()
            .all(|&p| p == BlockPlacement::HbmResident)
            && plan.blocks[resident..resident + host_count]
                .iter()
                .all(|&p| p == host_tier)
            && plan.blocks[resident + host_count..]
                .iter()
                .all(|&p| p == BlockPlacement::DiskStreamed),
        "live plan block tiers are not monotonic"
    );
    if resident < spec.layers {
        ensure!(
            plan.hbm_resident_bytes + active_block_bytes > plan.hbm_weight_budget_bytes,
            "live plan left unused HBM capacity for another resident block"
        );
    } else {
        ensure!(
            plan.prefetch_depth == 0,
            "all-resident plan needs zero staging slots"
        );
    }
    if disk > 0 {
        ensure!(
            plan.host_staged_bytes + active_block_bytes > plan.usable_host_bytes,
            "live plan spilled to disk while host RAM still fit another block"
        );
    }
    Ok(())
}

#[test]
#[ignore = "TPU; FF_TPU_LIB path to libtpu.so"]
fn tpu_op_matmul_and_attention_gate() -> Result<()> {
    let lib_path = env_path("FF_TPU_LIB")?;
    let dev = TpuDevice::open(&lib_path, 0)?;

    // Op 1: H3-Image to_q matmul [4103, 5376] x [7168, 5376]^T.
    let (m, k, n) = (H3_ROWS_2048, H3_HIDDEN, H3_ATTN_WIDTH);
    let mut x = vec![bf16::ZERO; m * k];
    for (row, chunk) in x.chunks_exact_mut(k).enumerate() {
        for (col, slot) in chunk.iter_mut().enumerate() {
            *slot = matmul_x(row, col);
        }
    }
    let mut w = vec![bf16::ZERO; n * k];
    for (row, chunk) in w.chunks_exact_mut(k).enumerate() {
        for (col, slot) in chunk.iter_mut().enumerate() {
            *slot = matmul_w(row, col);
        }
    }
    let exec1 = dev.compile_mlir(&emit_bf16_matmul(m, k, n)?)?;
    let sem = HostSemantics::ImmutableUntilTransferCompletes;
    let x_buf = dev.upload_bf16(&x, &[m as i64, k as i64], sem)?;
    let w_buf = dev.upload_bf16(&w, &[n as i64, k as i64], sem)?;
    let outs1 = exec1.execute(&[&x_buf, &w_buf], &[m * n * 2])?;
    let got1 = outs1[0].download_bf16()?;
    drop((outs1, x_buf, w_buf, exec1));

    let x_f64: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
    let w_f64: Vec<f64> = w.iter().map(|v| v.to_f64()).collect();
    let workers = thread::available_parallelism().map_or(1, usize::from);
    let chunk = m.div_ceil(workers);
    let (fin1, max1, se1, sr1, neg_se1) = thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..m).step_by(chunk) {
            let end = (start + chunk).min(m);
            let (xr_all, wr_all, gr_all) = (&x_f64, &w_f64, &got1);
            handles.push(scope.spawn(move || {
                let (mut fin, mut ma, mut se, mut sr, mut nse) =
                    (true, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for row in start..end {
                    let xr = &xr_all[row * k..(row + 1) * k];
                    let gr = &gr_all[row * n..(row + 1) * n];
                    for col in 0..n {
                        let wr = &wr_all[col * k..(col + 1) * k];
                        let mut acc = 0.0f64;
                        let mut acc_shift = 0.0f64;
                        for kk in 0..k - 1 {
                            acc += xr[kk] * wr[kk];
                            acc_shift += xr[kk] * wr[kk + 1];
                        }
                        acc += xr[k - 1] * wr[k - 1];
                        acc_shift += xr[k - 1] * wr[0];
                        let g = gr[col].to_f64();
                        fin &= g.is_finite();
                        let e = (g - acc).abs();
                        ma = ma.max(e);
                        se += e * e;
                        sr += acc * acc;
                        let ne = g - acc_shift;
                        nse += ne * ne;
                    }
                }
                (fin, ma, se, sr, nse)
            }));
        }
        let (mut fin, mut ma, mut se, mut sr, mut nse) = (true, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for h in handles {
            let (f, a, e, r, ne) = h.join().expect("op1 worker panicked");
            fin &= f;
            ma = ma.max(a);
            se += e;
            sr += r;
            nse += ne;
        }
        (fin, ma, se, sr, nse)
    });
    let rmse1 = (se1 / (m * n) as f64).sqrt();
    let rel1 = (se1 / sr1).sqrt();
    let neg1 = (neg_se1 / (m * n) as f64).sqrt();
    println!(
        "TPU_OP_GATE_MATMUL finite={fin1} max_abs={max1:.6e} rmse={rmse1:.6e} rel_l2={rel1:.6e} neg_rmse={neg1:.6e}"
    );
    ensure!(
        fin1 && max1 <= MATMUL_MAX_ABS_TOL
            && rmse1 <= MATMUL_RMSE_TOL
            && rel1 <= MATMUL_REL_L2_TOL
            && neg1 > MATMUL_NEG_RMSE_FLOOR,
        "op1 matmul gate failed"
    );

    // Op 2: H3-Image bidirectional attention [1031, 56, 128].
    let (rows, heads, d) = (H3_ROWS_1024, H3_HEADS, H3_HEAD_DIM);
    let total = rows * heads * d;
    let mut q = vec![bf16::ZERO; total];
    let mut k_in = vec![bf16::ZERO; total];
    let mut v_in = vec![bf16::ZERO; total];
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * d;
            for dim in 0..d {
                q[base + dim] = attn_q(r, h, dim);
                k_in[base + dim] = attn_k(r, h, dim);
                v_in[base + dim] = attn_v(r, h, dim);
            }
        }
    }
    let scale = 1.0f32 / (d as f32).sqrt();
    let exec2 = dev.compile_mlir(&emit_bf16_attention(rows, heads, d, scale)?)?;
    let dims = [rows as i64, heads as i64, d as i64];
    let q_buf = dev.upload_bf16(&q, &dims, sem)?;
    let k_buf = dev.upload_bf16(&k_in, &dims, sem)?;
    let v_buf = dev.upload_bf16(&v_in, &dims, sem)?;
    let outs2 = exec2.execute(&[&q_buf, &k_buf, &v_buf], &[total * 2])?;
    let got2 = outs2[0].download_bf16()?;
    let fin2 = got2.iter().all(|x| x.is_finite());
    let q_f64: Vec<f64> = q.iter().map(|x| x.to_f64()).collect();
    let k_f64: Vec<f64> = k_in.iter().map(|x| x.to_f64()).collect();
    let v_f64: Vec<f64> = v_in.iter().map(|x| x.to_f64()).collect();
    let scale_f64 = 1.0f64 / (d as f64).sqrt();
    let (max2, se2, sr2, nu_se, nh_se, nv_se) = thread::scope(|scope| {
        let mut handles = Vec::new();
        for h in 0..heads {
            let (qr, kr, vr, gr) = (&q_f64, &k_f64, &v_f64, &got2);
            handles.push(scope.spawn(move || {
                let (mut ma, mut se, mut sr, mut nu, mut nh, mut nv) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let mut scores = vec![0.0f64; rows];
                let mut probs = vec![0.0f64; rows];
                let mut probs_un = vec![0.0f64; rows];
                let mut probs_h = vec![0.0f64; rows];
                let h_alt = (h + 1) % heads;
                for rq in 0..rows {
                    let q_off = (rq * heads + h) * d;
                    let (mut peak, mut peak_un, mut peak_h) =
                        (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                    for rk in 0..rows {
                        let k_off = (rk * heads + h) * d;
                        let kh_off = (rk * heads + h_alt) * d;
                        let (mut dot, mut dot_h) = (0.0f64, 0.0f64);
                        for dim in 0..d {
                            dot += qr[q_off + dim] * kr[k_off + dim];
                            dot_h += qr[q_off + dim] * kr[kh_off + dim];
                        }
                        scores[rk] = dot * scale_f64;
                        probs_h[rk] = dot_h * scale_f64;
                        peak = peak.max(scores[rk]);
                        peak_un = peak_un.max(dot);
                        peak_h = peak_h.max(probs_h[rk]);
                    }
                    let (mut s0, mut su, mut sh) = (0.0f64, 0.0f64, 0.0f64);
                    for (((p0, pu), ph), &sc) in probs
                        .iter_mut()
                        .zip(&mut probs_un)
                        .zip(&mut probs_h)
                        .zip(&scores)
                    {
                        let e0 = (sc - peak).exp();
                        let eu = (sc / scale_f64 - peak_un).exp();
                        let eh = (*ph - peak_h).exp();
                        *p0 = e0;
                        *pu = eu;
                        *ph = eh;
                        s0 += e0;
                        su += eu;
                        sh += eh;
                    }
                    for p in &mut probs {
                        *p /= s0;
                    }
                    for p in &mut probs_un {
                        *p /= su;
                    }
                    for p in &mut probs_h {
                        *p /= sh;
                    }
                    for dim in 0..d {
                        let (mut a0, mut au, mut ah, mut av) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                        for rk in 0..rows {
                            let vv = vr[(rk * heads + h) * d + dim];
                            let vshift = vr[(((rk + 1) % rows) * heads + h) * d + dim];
                            a0 += probs[rk] * vv;
                            au += probs_un[rk] * vv;
                            ah += probs_h[rk] * vv;
                            av += probs[rk] * vshift;
                        }
                        let g = gr[q_off + dim].to_f64();
                        let e = (g - a0).abs();
                        ma = ma.max(e);
                        se += e * e;
                        sr += a0 * a0;
                        nu += (g - au) * (g - au);
                        nh += (g - ah) * (g - ah);
                        nv += (g - av) * (g - av);
                    }
                }
                (ma, se, sr, nu, nh, nv)
            }));
        }
        let (mut ma, mut se, mut sr, mut nu, mut nh, mut nv) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for h in handles {
            let (a, e, r, u, hd_e, ve) = h.join().expect("op2 worker panicked");
            ma = ma.max(a);
            se += e;
            sr += r;
            nu += u;
            nh += hd_e;
            nv += ve;
        }
        (ma, se, sr, nu, nh, nv)
    });
    let rmse2 = (se2 / total as f64).sqrt();
    let rel2 = (se2 / sr2).sqrt();
    let nu_r = (nu_se / total as f64).sqrt();
    let nh_r = (nh_se / total as f64).sqrt();
    let nv_r = (nv_se / total as f64).sqrt();
    println!(
        "TPU_OP_GATE_ATTN finite={fin2} max_abs={max2:.6e} rmse={rmse2:.6e} rel_l2={rel2:.6e} neg_unscaled={nu_r:.6e} neg_head={nh_r:.6e} neg_vshift={nv_r:.6e}"
    );
    ensure!(
        fin2 && max2 <= ATTN_MAX_ABS_TOL
            && rmse2 <= ATTN_RMSE_TOL
            && rel2 <= ATTN_REL_L2_TOL
            && nu_r > ATTN_NEG_UNSCALED_RMSE_FLOOR
            && nh_r > ATTN_NEG_SHIFT_RMSE_FLOOR
            && nv_r > ATTN_NEG_SHIFT_RMSE_FLOOR,
        "op2 attention gate failed"
    );
    Ok(())
}

#[test]
#[ignore = "TPU; FF_TPU_LIB, FF_H3_IMAGE_CONFIG, FF_H3_IMAGE_SHARD0, FF_TPU_STEP2_FIXTURES"]
fn tpu_h3_image_dit_block0_gate() -> Result<()> {
    let lib_path = env_path("FF_TPU_LIB")?;
    let cfg_path = env_path("FF_H3_IMAGE_CONFIG")?;
    let shard0_path = env_path("FF_H3_IMAGE_SHARD0")?;
    let fixtures_dir = env_path("FF_TPU_STEP2_FIXTURES")?;
    let out_dir = env::var_os("FF_TPU_GATE_OUT").map(PathBuf::from);

    let cfg_json: serde_json::Value = serde_json::from_slice(
        &fs::read(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?,
    )?;
    let spec = ImageSpec::parse(&cfg_json)?;
    let dev = TpuDevice::open(&lib_path, 0)?;

    let time_embedder = load_time_embedder_from_shard(&shard0_path, &spec)?;
    let adaln_proj = load_block_adaln_proj_from_shard(&shard0_path, &spec, 0)?;
    let active_weights = load_block_active_weights_from_shard(&shard0_path, &spec, 0)?;
    let active_bytes = active_weights.active_bytes();
    ensure!(
        active_bytes == 770_725_376,
        "unexpected H3-Image block 0 active bytes {active_bytes}"
    );

    // Stage active block 0 weights in pinned_host (when exposed) before running cases.
    let streamed = StreamedDitBlock::stage(&dev, &spec, active_weights)?;
    ensure!(
        streamed.staged_in_pinned_host == dev.pinned_host_memory().is_some(),
        "pinned_host staging mismatch"
    );

    let schedule = ImageSchedule::published()?;
    let mut adaln_cache = TpuAdalnCache::default();
    let mut adaln_first_recomputed = false;
    let mut adaln_second_recomputed = true;
    let mut cases = Vec::new();
    let mut controls = Vec::new();
    let mut last_compiled_peak_4103 = 0u64;
    let mut last_h2d_bps_4103 = None;
    let mut last_adaln_pairs = 0usize;

    let mut adaln_proj_opt = Some(adaln_proj);
    for (case_idx, &(canvas, expected_rows)) in
        [(1024usize, H3_ROWS_1024), (2048usize, H3_ROWS_2048)]
            .iter()
            .enumerate()
    {
        let fix = load_case_fixture(
            &fixtures_dir.join(format!("case_{expected_rows}.safetensors")),
            expected_rows,
            spec.hidden,
        )?;
        ensure!(
            fix.timesteps == schedule.timesteps,
            "fixture timesteps {:?} != published {:?}",
            fix.timesteps,
            schedule.timesteps
        );
        let text_len = fix.token_tags.iter().filter(|&&t| t == TEXT_TAG).count();
        let text_tags = vec![TEXT_TAG; text_len];
        let (lh, lw) = image_latent_extent(&spec, canvas, canvas)?;
        let layout = image_layout(&spec, &text_tags, &[], lh, lw)?;
        ensure!(layout.rows() == expected_rows, "layout rows mismatch");
        ensure!(
            layout.token_tags == fix.token_tags,
            "layout token_tags mismatch"
        );
        ensure!(
            layout.position_ids == fix.position_ids,
            "layout position_ids mismatch"
        );

        let row_plan = image_row_plan(&layout, &schedule)?;
        let adaln_plan = AdalnPlan::from_tags(&layout.token_tags, &row_plan)?;
        let recomputed = adaln_cache.ensure_for_plan(&adaln_plan, |plan| {
            let proj = adaln_proj_opt
                .take()
                .context("adaln_proj consumed unexpectedly before recompute")?;
            let tbl = compute_block_adaln_table(&dev, &spec, &time_embedder, &proj, plan)?;
            Ok(vec![tbl])
        })?;
        if case_idx == 0 {
            adaln_first_recomputed = recomputed;
            ensure!(recomputed, "first AdaLN table build must compute");
        } else {
            adaln_second_recomputed = recomputed;
            ensure!(
                !recomputed,
                "identical schedule/tags must hit TpuAdalnCache"
            );
        }
        let adaln_table = adaln_cache
            .layer(0)
            .context("missing cached layer 0 AdaLN")?;
        let pairs = adaln_plan.pairs.len();
        last_adaln_pairs = pairs;

        let rotary = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &layout.position_ids);
        let row_pairs_i32: Vec<i32> = adaln_plan.row_pairs[fix.step_index]
            .iter()
            .map(|&v| i32::try_from(v).context("row_pair overflows i32"))
            .collect::<Result<Vec<_>>>()?;

        let t_compile = Instant::now();
        let exec = compile_h3_dit_block(&dev, &spec, expected_rows, pairs, DIT_HEAD_CHUNK)?;
        let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

        let sem = HostSemantics::ImmutableUntilTransferCompletes;
        let x_buf = dev.upload_bf16(&fix.x, &[expected_rows as i64, spec.hidden as i64], sem)?;
        let adaln_buf =
            dev.upload_bf16(adaln_table, &[pairs as i64, 6, spec.hidden as i64], sem)?;
        let rp_buf = dev.upload_i32(&row_pairs_i32, &[expected_rows as i64], sem)?;
        let rot_buf = dev.upload_bf16(
            &rotary,
            &[expected_rows as i64, (6 * spec.rope_freq_dim) as i64],
            sem,
        )?;

        let hbm_before = dev.memory_stats()?.bytes_in_use;
        let inputs = BlockRunInputs {
            x: &x_buf,
            adaln: &adaln_buf,
            row_pairs: &rp_buf,
            rotary: &rot_buf,
            rows: expected_rows,
        };
        let (out_buf, timing) = streamed.execute_streamed(&dev, &exec, &inputs)?;
        let got = out_buf.download_bf16()?;
        drop(out_buf);
        let stats_after = dev.memory_stats()?;
        let hbm_after_drop = stats_after.bytes_in_use;
        let hbm_peak = stats_after.peak_bytes_in_use;

        let metrics = compute_metrics(&got, &fix.ref_bf16)?;
        let hbm_released = hbm_after_drop <= hbm_before + HBM_RELEASE_SLACK_BYTES;
        let passed = metrics.all_finite
            && metrics.rel_l2 <= DIT_REL_L2_TOL
            && metrics.rel_max_abs <= DIT_REL_MAX_ABS_TOL
            && hbm_released;
        println!(
            "TPU_STEP2_BLOCK canvas={canvas} rows={expected_rows} head_chunk={DIT_HEAD_CHUNK} compile_ms={compile_ms:.2} h2d_ms={:.2} exec_ms={:.2} pinned_copy={} finite={} rel_l2={:.6e} rel_max_abs={:.6e} max_abs={:.6e} rmse={:.6e} hbm_before={hbm_before} hbm_after_drop={hbm_after_drop} hbm_peak={hbm_peak:?} pass={passed}",
            timing.h2d_ms,
            timing.execute_ms,
            timing.used_pinned_host_copy,
            metrics.all_finite,
            metrics.rel_l2,
            metrics.rel_max_abs,
            metrics.max_abs,
            metrics.rmse
        );

        if expected_rows == H3_ROWS_1024 {
            // Control 1a: 3D RoPE cos/sin transpose (swap cos and sin halves of rotary table).
            let half_rot = 3 * spec.rope_freq_dim;
            let rot_dim = 2 * half_rot;
            let mut rot_cs_bad = Vec::with_capacity(rotary.len());
            for row in rotary.chunks_exact(rot_dim) {
                rot_cs_bad.extend_from_slice(&row[half_rot..rot_dim]);
                rot_cs_bad.extend_from_slice(&row[..half_rot]);
            }
            let rot_cs_buf =
                dev.upload_bf16(&rot_cs_bad, &[expected_rows as i64, rot_dim as i64], sem)?;
            let (out_c1a, _) = streamed.execute_streamed(
                &dev,
                &exec,
                &BlockRunInputs {
                    rotary: &rot_cs_buf,
                    ..inputs
                },
            )?;
            let m_c1a = compute_metrics(&out_c1a.download_bf16()?, &fix.ref_bf16)?;
            let min_c1a = ROPE_COS_SIN_CONTROL_MULT * DIT_REL_L2_TOL;
            let c1a = ControlReport {
                name: "rope_cos_sin_swap".into(),
                rel_l2: m_c1a.rel_l2,
                rel_max_abs: m_c1a.rel_max_abs,
                min_expected_rel_l2: min_c1a,
                failed_as_required: m_c1a.rel_l2 > min_c1a,
            };
            println!(
                "TPU_STEP2_CONTROL name={} rel_l2={:.6e} rel_max_abs={:.6e} min_expected={:.6e} failed_as_required={}",
                c1a.name,
                c1a.rel_l2,
                c1a.rel_max_abs,
                c1a.min_expected_rel_l2,
                c1a.failed_as_required
            );
            controls.push(c1a);
            drop((out_c1a, rot_cs_buf));

            // Control 1b: 3D RoPE spatial (h, w) axis transpose through the same compiled executable.
            let pos_swapped: Vec<[f64; 3]> = layout
                .position_ids
                .iter()
                .map(|&[t, h, w]| [t, w, h])
                .collect();
            let rot_hw_bad = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &pos_swapped);
            let rot_hw_buf =
                dev.upload_bf16(&rot_hw_bad, &[expected_rows as i64, rot_dim as i64], sem)?;
            let (out_c1b, _) = streamed.execute_streamed(
                &dev,
                &exec,
                &BlockRunInputs {
                    rotary: &rot_hw_buf,
                    ..inputs
                },
            )?;
            let m_c1b = compute_metrics(&out_c1b.download_bf16()?, &fix.ref_bf16)?;
            let min_c1b = ROPE_HW_CONTROL_MULT * DIT_REL_L2_TOL;
            let c1b = ControlReport {
                name: "rope_hw_transpose".into(),
                rel_l2: m_c1b.rel_l2,
                rel_max_abs: m_c1b.rel_max_abs,
                min_expected_rel_l2: min_c1b,
                failed_as_required: m_c1b.rel_l2 > min_c1b,
            };
            println!(
                "TPU_STEP2_CONTROL name={} rel_l2={:.6e} rel_max_abs={:.6e} min_expected={:.6e} failed_as_required={}",
                c1b.name,
                c1b.rel_l2,
                c1b.rel_max_abs,
                c1b.min_expected_rel_l2,
                c1b.failed_as_required
            );
            controls.push(c1b);
            drop((out_c1b, rot_hw_buf));

            // Control 2: AdaLN gate negation (negate gate_msa slot 2 and gate_mlp slot 5).
            let mut adaln_bad = adaln_table.to_vec();
            let hidden = spec.hidden;
            for p_idx in 0..pairs {
                for slot in [2usize, 5usize] {
                    let off = (p_idx * 6 + slot) * hidden;
                    for val in &mut adaln_bad[off..off + hidden] {
                        *val = bf16::from_f32(-val.to_f32());
                    }
                }
            }
            let adaln_bad_buf =
                dev.upload_bf16(&adaln_bad, &[pairs as i64, 6, spec.hidden as i64], sem)?;
            let (out_c2, _) = streamed.execute_streamed(
                &dev,
                &exec,
                &BlockRunInputs {
                    adaln: &adaln_bad_buf,
                    ..inputs
                },
            )?;
            let m_c2 = compute_metrics(&out_c2.download_bf16()?, &fix.ref_bf16)?;
            let min_c2 = ADALN_CONTROL_MULT * DIT_REL_L2_TOL;
            let c2 = ControlReport {
                name: "adaln_gate_negate".into(),
                rel_l2: m_c2.rel_l2,
                rel_max_abs: m_c2.rel_max_abs,
                min_expected_rel_l2: min_c2,
                failed_as_required: m_c2.rel_l2 > min_c2,
            };
            println!(
                "TPU_STEP2_CONTROL name={} rel_l2={:.6e} rel_max_abs={:.6e} min_expected={:.6e} failed_as_required={}",
                c2.name, c2.rel_l2, c2.rel_max_abs, c2.min_expected_rel_l2, c2.failed_as_required
            );
            controls.push(c2);
            drop((out_c2, adaln_bad_buf));

            // Control 3: SwiGLU half-swap (swap ff.net.0.proj lower [0..ffn] and upper [ffn..2*ffn] rows).
            let mut swapped_weights = load_block_active_weights_from_shard(&shard0_path, &spec, 0)?;
            let half_elems = spec.ffn * spec.hidden;
            let (lo, hi) = swapped_weights.ff_up.split_at_mut(half_elems);
            lo.swap_with_slice(hi);
            let swapped_block = StreamedDitBlock::stage(&dev, &spec, swapped_weights)?;
            let (out_c3, _) = swapped_block.execute_streamed(&dev, &exec, &inputs)?;
            let m_c3 = compute_metrics(&out_c3.download_bf16()?, &fix.ref_bf16)?;
            let min_c3 = SWIGLU_CONTROL_MULT * DIT_REL_L2_TOL;
            let c3 = ControlReport {
                name: "swiglu_half_swap".into(),
                rel_l2: m_c3.rel_l2,
                rel_max_abs: m_c3.rel_max_abs,
                min_expected_rel_l2: min_c3,
                failed_as_required: m_c3.rel_l2 > min_c3,
            };
            println!(
                "TPU_STEP2_CONTROL name={} rel_l2={:.6e} rel_max_abs={:.6e} min_expected={:.6e} failed_as_required={}",
                c3.name, c3.rel_l2, c3.rel_max_abs, c3.min_expected_rel_l2, c3.failed_as_required
            );
            controls.push(c3);
            drop((out_c3, swapped_block));
        } else {
            if let Some(cs) = exec.compiled_memory_stats.as_ref() {
                last_compiled_peak_4103 = u64::try_from(cs.peak_memory_in_bytes.max(0))?;
            }
            if timing.h2d_ms > 0.0 {
                last_h2d_bps_4103 = Some(active_bytes as f64 / (timing.h2d_ms * 1e-3));
            }
        }

        cases.push(BlockCaseReport {
            canvas,
            rows: expected_rows,
            head_chunk: DIT_HEAD_CHUNK,
            adaln_pairs: pairs,
            compile_ms,
            h2d_ms: timing.h2d_ms,
            execute_ms: timing.execute_ms,
            active_weight_bytes: active_bytes,
            used_pinned_host_copy: timing.used_pinned_host_copy,
            compiled_memory_stats: exec.compiled_memory_stats.clone(),
            hbm_before_bytes: hbm_before,
            hbm_after_drop_bytes: hbm_after_drop,
            hbm_peak_bytes: hbm_peak,
            metrics,
            passed,
        });
        drop((x_buf, adaln_buf, rp_buf, rot_buf, exec));
    }

    // Derive the live hardware-adaptive placement plan from sensed device/host metrics and compiled R=4103 stats.
    let live_topology = TpuTopology::detect(&dev, last_h2d_bps_4103)?;
    let persistent_hbm_bytes =
        (spec.layers as u64) * (last_adaln_pairs as u64) * 6 * (spec.hidden as u64) * 2
            + 128 * 1024 * 1024;
    let compiled_peak = last_compiled_peak_4103.max(active_bytes);
    let footprint = ModelFootprint {
        persistent_hbm_bytes,
        compiled_block_peak_bytes: compiled_peak,
        compiled_block_weight_arg_bytes: active_bytes,
        block_weight_bytes: vec![active_bytes; spec.layers],
    };
    let live_plan = TpuPlacementPlan::derive(&live_topology, &footprint)?;
    check_live_plan(&live_topology, &live_plan, &spec, active_bytes)?;
    println!(
        "TPU_STEP2_PLAN chip=\"{}\" hbm_limit={} host_avail={} has_pinned={} compiled_peak_4103={} resident={} pinned_host={} pageable={} disk={} prefetch_depth={} est_stream_s={:?}",
        live_topology.chip_kind,
        live_topology.hbm_limit_bytes,
        live_topology.host_available_bytes,
        live_topology.has_pinned_host,
        compiled_peak,
        live_plan.count(BlockPlacement::HbmResident),
        live_plan.count(BlockPlacement::PinnedHostStreamed),
        live_plan.count(BlockPlacement::PageableHostStreamed),
        live_plan.count(BlockPlacement::DiskStreamed),
        live_plan.prefetch_depth,
        live_plan.estimated_stream_s_per_step
    );

    let all_cases_pass = cases.iter().all(|c| c.passed);
    let all_controls_pass = controls.len() == 4 && controls.iter().all(|c| c.failed_as_required);
    let overall_pass = all_cases_pass && all_controls_pass;
    let summary = Step2Summary {
        device: dev.attributes.clone(),
        tolerances: ToleranceSummary {
            rel_l2_tol: DIT_REL_L2_TOL,
            rel_max_abs_tol: DIT_REL_MAX_ABS_TOL,
            rope_control_mult: ROPE_COS_SIN_CONTROL_MULT,
            adaln_control_mult: ADALN_CONTROL_MULT,
            swiglu_control_mult: SWIGLU_CONTROL_MULT,
        },
        adaln_first_recomputed,
        adaln_second_recomputed,
        cases,
        controls_1031: controls,
        live_topology,
        live_plan,
        passed: overall_pass,
    };
    if let Some(dir) = out_dir {
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join("tpu-step2-summary.json"),
            serde_json::to_string_pretty(&summary)?,
        )?;
    }
    println!(
        "TPU_STEP2_RESULT overall_pass={overall_pass} cases_pass={all_cases_pass} controls_pass={all_controls_pass}"
    );
    ensure!(overall_pass, "TPU Step 2 H3-Image DiT block 0 gate failed");
    Ok(())
}
