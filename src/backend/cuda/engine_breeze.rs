//! Breeze-TTS-2 M0 prefill and M1 greedy decode.
//! GEMM row count is 17 when `supports` is true for the logical row count.
//! Text full-attention layers are indices where `(index + 1)` is divisible by 6.
//! Depth head `k` reads post-norm position `k + 1`.
//! Rope table row: `dim/2` cos values, then `dim/2` sin values.

use super::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{self, RowDomain, SeqRef, Tensor, TensorMut},
    cubin,
    flash::{self, Flash, Score},
    ops::{self, CommonOpsRef, Ops, RmsMode, RmsOptions},
    verification::Shape,
};
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, PushKernelArg,
};
use crate::backend::setup::LinearShape;
use crate::checkpoint::MappedCheckpoint;
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::Dtype;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

const TEXT_LAYERS: usize = 26;
const TEXT_H: usize = 1152;
const TEXT_I: usize = 6912;
const TEXT_HEADS: usize = 4;
const TEXT_KV: usize = 1;
const TEXT_DIM: usize = 256;
const TEXT_Q: usize = TEXT_HEADS * TEXT_DIM;
const TEXT_KVW: usize = TEXT_KV * TEXT_DIM;
const TEXT_MAX: usize = 652;
const TEXT_VOCAB: u32 = 262_158;
const TEXT_EOI: u32 = 256_000;
const TEXT_EPS: f32 = 1e-6;
const TEXT_OUT: usize = 2048;
const TEXT_POSITIONS: usize = 653;

const BB_LAYERS: usize = 28;
const BB_H: usize = 2048;
const BB_I: usize = 6144;
const BB_HEADS: usize = 16;
const BB_KV: usize = 8;
const BB_DIM: usize = 128;
const BB_Q: usize = BB_HEADS * BB_DIM;
const BB_KVW: usize = BB_KV * BB_DIM;
const BB_MAX: usize = 716;
const BB_EPS: f32 = 1e-6;
const BB_VOCAB: usize = 2052;

const DP_LAYERS: usize = 12;
const DP_H: usize = 1024;
const DP_I: usize = 8192;
const DP_HEADS: usize = 8;
const DP_KV: usize = 2;
const DP_DIM: usize = 128;
const DP_Q: usize = DP_HEADS * DP_DIM;
const DP_KVW: usize = DP_KV * DP_DIM;
const DP_ROWS: usize = 16;
const DP_EPS: f32 = 1e-5;
const DP_FRAMES: usize = 64;
const BOOKS: usize = 16;
const CODEBOOK: i32 = 2051;
const HEAD_N: usize = 15;
#[cfg(test)]
const AUDIO: i32 = 262_144;
#[cfg(test)]
const AUDIO_EOS: i32 = 262_145;

const H_ELEMS: usize = BB_MAX * BB_H;
const D_ELEMS: usize = BB_MAX * BB_VOCAB;
const MLP_ELEMS: usize = TEXT_MAX * TEXT_I;
const QKV_ELEMS: usize = BB_MAX * (BB_Q + BB_KVW + BB_KVW);
const KV_ELEMS: usize = BB_KV * BB_MAX * BB_DIM;
const ATTN_ELEMS: usize = H_ELEMS;
const AUDIO_ELEMS: usize = DP_FRAMES * BB_H;
const STACK_ELEMS: usize = DP_FRAMES * DP_ROWS * DP_H;
const HEAD_IN_ELEMS: usize = DP_FRAMES * DP_H;
const HEAD_OUT_ELEMS: usize = DP_FRAMES * (CODEBOOK as usize);
const EMBED_WS: usize = 17 * BB_H;
const IDS_ELEMS: usize = DP_FRAMES * BOOKS;

pub fn gemm_rows(logical: usize, input: usize) -> usize {
    if super::linear_calibrate::supports(LinearShape {
        rows: logical,
        output: 1,
        input,
    }) {
        17
    } else {
        logical
    }
}

enum RopeLaw {
    Plain { theta: f32, divisor: f32 },
    Llama3,
}

fn inv_freqs(dim: usize, law: RopeLaw) -> Result<Vec<f32>> {
    ensure!(dim == 128 || dim == 256, "rope dim {dim}");
    let n = dim / 2;
    let mut out = Vec::with_capacity(n);
    match law {
        RopeLaw::Plain { theta, divisor } => {
            for i in 0..n {
                let inv = 1.0 / theta.powf((2 * i) as f32 / dim as f32) / divisor;
                ensure!(inv.is_finite(), "non-finite inverse frequency");
                out.push(inv);
            }
        }
        RopeLaw::Llama3 => {
            let theta = 500_000f32;
            let factor = 32f32;
            let low_freq = 0.001953125f32;
            let high_freq = 0.0078125f32;
            let old = 16f32;
            let low_w = old / low_freq;
            let high_w = old / high_freq;
            let two_pi = (2.0f64 * std::f64::consts::PI) as f32;
            for i in 0..n {
                let inv = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
                let wavelen = two_pi / inv;
                let mut scaled = if wavelen > low_w { inv / factor } else { inv };
                if wavelen >= high_w && wavelen <= low_w {
                    let smooth = (old / wavelen - low_freq) / (high_freq - low_freq);
                    scaled = (1.0 - smooth) * inv / factor + smooth * inv;
                }
                ensure!(scaled.is_finite(), "non-finite llama3 inverse frequency");
                out.push(scaled);
            }
        }
    }
    Ok(out)
}

fn rope_table(positions: usize, dim: usize, law: RopeLaw) -> Result<Vec<bf16>> {
    ensure!(positions > 0, "empty rope table");
    let inv = inv_freqs(dim, law)?;
    let mut table = vec![bf16::ZERO; positions * dim];
    for p in 0..positions {
        for (d, &freq) in inv.iter().enumerate() {
            let angle = p as f32 * freq;
            table[p * dim + d] = bf16::from_f32(angle.cos());
            table[p * dim + dim / 2 + d] = bf16::from_f32(angle.sin());
        }
    }
    Ok(table)
}

fn text_rope_law(full: bool) -> RopeLaw {
    RopeLaw::Plain {
        theta: if full { 1_000_000.0 } else { 10_000.0 },
        divisor: if full { 8.0 } else { 1.0 },
    }
}

fn identity_rope_table(positions: usize, dim: usize) -> Result<Vec<bf16>> {
    ensure!(
        positions > 0 && dim.is_multiple_of(2),
        "identity rope table"
    );
    let mut table = vec![bf16::ZERO; positions * dim];
    for position in 0..positions {
        for pair in 0..dim / 2 {
            table[position * dim + pair] = bf16::ONE;
        }
    }
    Ok(table)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextPlant {
    Nominal,
    RopeOmitted,
    WindowIgnored,
}

pub struct Positions {
    pub text: Vec<usize>,
    pub audio: Vec<usize>,
    pub eos: Vec<usize>,
    pub seq: usize,
}

struct TextLayer {
    pre_attn: CudaSlice<bf16>,
    post_attn: CudaSlice<bf16>,
    pre_ffn: CudaSlice<bf16>,
    post_ffn: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    qn: CudaSlice<bf16>,
    kn: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
    full: bool,
}

struct DecoderLayer {
    pre: CudaSlice<bf16>,
    post: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    qn: CudaSlice<bf16>,
    kn: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
}

struct Weights {
    text_embed: CudaSlice<bf16>,
    text: Vec<TextLayer>,
    text_norm: CudaSlice<bf16>,
    text_proj: CudaSlice<bf16>,
    backbone: Vec<DecoderLayer>,
    backbone_norm: CudaSlice<bf16>,
    lm: CudaSlice<bf16>,
    audio: CudaSlice<bf16>,
    projector: CudaSlice<bf16>,
    depth: Vec<DecoderLayer>,
    depth_norm: CudaSlice<bf16>,
    heads: Vec<CudaSlice<bf16>>,
}

struct Kern {
    stream: Arc<CudaStream>,
    ops: Ops,
    common: CommonOpsRef,
    rope_one_plus: CudaFunction,
    rope_cast: CudaFunction,
    rope_skip: CudaFunction,
    embed_rows_sum: CudaFunction,
    residual_scale: CudaFunction,
    table_slide: CudaSlice<bf16>,
    table_full: CudaSlice<bf16>,
    table_identity: CudaSlice<bf16>,
    table_backbone: CudaSlice<bf16>,
    table_depth: CudaSlice<bf16>,
    drains: Cell<u64>,
    h2d: Cell<u64>,
    d2h: Cell<u64>,
    d2d: Cell<u64>,
}

struct Buf {
    h: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    d: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    lk: Vec<CudaSlice<bf16>>,
    lv: Vec<CudaSlice<bf16>>,
    dk: CudaSlice<bf16>,
    dv: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    audio: CudaSlice<bf16>,
    stack: CudaSlice<bf16>,
    head_in: CudaSlice<bf16>,
    head_out: CudaSlice<bf16>,
    embed_ws: CudaSlice<bf16>,
    meta: CudaSlice<i32>,
    start: CudaSlice<i32>,
    tok: CudaSlice<u32>,
    ids: CudaSlice<i32>,
}

struct Flashes {
    text_full: Flash,
    text_slide: Flash,
    backbone: Flash,
    depth: Flash,
    slide_left: usize,
}

pub struct Breeze {
    device: Device,
    w: Weights,
    k: Kern,
    flash: Flashes,
    buf: Buf,
    blas: Blas,
    pub weight_bytes: usize,
}

#[derive(Clone, Copy)]
enum RopeKind {
    OnePlus,
    Cast,
    Skip,
}

#[derive(Clone, Copy)]
enum KvSlot {
    Shared,
    Depth,
    Layer(usize),
}

#[derive(Clone, Copy)]
struct RopeGeom {
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    capacity: usize,
    q_row_stride: usize,
    k_offset: usize,
    k_row_stride: usize,
    v_offset: usize,
    v_row_stride: usize,
    eps: f32,
    kind: RopeKind,
    slot: KvSlot,
}

struct Gemm<'a> {
    w: &'a CudaSlice<bf16>,
    x: &'a mut CudaSlice<bf16>,
    y: &'a mut CudaSlice<bf16>,
    x_off: usize,
    y_off: usize,
    logical: usize,
    output: usize,
    input: usize,
}

struct Rms<'a> {
    x: &'a CudaSlice<bf16>,
    y: &'a mut CudaSlice<bf16>,
    w: &'a CudaSlice<bf16>,
    rows: usize,
    width: usize,
    mode: RmsMode,
    eps: f32,
}

#[derive(Clone, Copy)]
struct TextStep {
    rows: usize,
    plant: TextPlant,
    window_left: Option<usize>,
}

#[derive(Clone, Copy)]
struct DecGeom {
    rows: usize,
    hidden: usize,
    intermediate: usize,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    eps: f32,
}

struct DecPass<'a> {
    layer: &'a DecoderLayer,
    table: &'a CudaSlice<bf16>,
    flash: &'a mut Flash,
    kind: RopeKind,
    geom: DecGeom,
    origin: i32,
    capacity: usize,
    slot: KvSlot,
}

#[derive(Clone, Copy)]
struct DecLoad {
    hidden: usize,
    intermediate: usize,
    q: usize,
    kv: usize,
    dim: usize,
    qk_norm: bool,
}

fn copy_h2d<T: DeviceRepr>(k: &Kern, host: &[T], device: &mut impl DevicePtrMut<T>) -> Result<()> {
    k.stream.memcpy_htod(host, device)?;
    k.stream.synchronize()?;
    k.h2d.set(k.h2d.get() + std::mem::size_of_val(host) as u64);
    k.drains.set(k.drains.get() + 1);
    Ok(())
}

fn copy_d2d<T: DeviceRepr>(
    k: &Kern,
    src: &impl DevicePtr<T>,
    dst: &mut impl DevicePtrMut<T>,
) -> Result<()> {
    k.stream.memcpy_dtod(src, dst)?;
    k.d2d.set(k.d2d.get() + src.num_bytes() as u64);
    Ok(())
}

fn upload_origin(k: &Kern, start: &mut CudaSlice<i32>, origin: i32) -> Result<()> {
    copy_h2d(k, &[origin], start)
}

fn rank3(
    buf: &CudaSlice<bf16>,
    stream: &Arc<CudaStream>,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 3>> {
    contracts::view(
        buf,
        stream,
        0,
        [u32::try_from(rows)?, 1, u32::try_from(width)?],
        [u64::try_from(width)?, u64::try_from(width)?, 1],
    )
}

fn weight1(buf: &CudaSlice<bf16>, stream: &Arc<CudaStream>, n: usize) -> Result<Tensor<bf16, 1>> {
    contracts::view(buf, stream, 0, [u32::try_from(n)?], [1])
}

fn domain(meta: &CudaSlice<i32>, stream: &Arc<CudaStream>, rows: usize) -> Result<RowDomain> {
    let (ptr, _guard) = meta.device_ptr(stream);
    Ok(RowDomain {
        meta: SeqRef { ptr },
        row_base: 0,
        row_capacity: u32::try_from(rows)?,
    })
}

fn set_meta(k: &Kern, meta: &mut CudaSlice<i32>, rows: usize) -> Result<()> {
    let rows = i32::try_from(rows)?;
    copy_h2d(k, &[0, rows, rows, rows, 1, 0, 0, 0], meta)
}

fn qkv_layout(
    logical: usize,
    input: usize,
    q_width: usize,
    kv_width: usize,
) -> Result<(usize, usize)> {
    let m = gemm_rows(logical, input);
    let k_off = m.checked_mul(q_width).context("qkv offset")?;
    let v_off = k_off
        .checked_add(m.checked_mul(kv_width).context("qkv offset")?)
        .context("qkv offset")?;
    Ok((k_off, v_off))
}

fn gemm(stream: &Arc<CudaStream>, blas: &Blas, g: Gemm<'_>) -> Result<()> {
    let rows = gemm_rows(g.logical, g.input);
    let x_need = g
        .x_off
        .checked_add(rows.checked_mul(g.input).context("gemm extent")?)
        .context("gemm extent")?;
    let y_need = g
        .y_off
        .checked_add(rows.checked_mul(g.output).context("gemm extent")?)
        .context("gemm extent")?;
    ensure!(x_need <= g.x.len() && y_need <= g.y.len(), "gemm buffer");
    if rows > g.logical {
        let start = g.x_off + g.logical * g.input;
        stream.memset_zeros(&mut g.x.slice_mut(start..x_need))?;
    }
    blas.linear(
        g.w,
        &g.x.slice(g.x_off..x_need),
        &mut g.y.slice_mut(g.y_off..y_need),
        rows,
        g.output,
        g.input,
    )
}

fn rms(k: &Kern, meta: &CudaSlice<i32>, call: Rms<'_>) -> Result<()> {
    let rows = domain(meta, &k.stream, call.rows)?;
    let x = rank3(call.x, &k.stream, call.rows, call.width)?;
    let mut y = TensorMut(rank3(call.y, &k.stream, call.rows, call.width)?);
    let weight = weight1(call.w, &k.stream, call.width)?;
    k.common.norm.rms_with(
        &k.stream,
        rows,
        x,
        weight,
        &mut y,
        RmsOptions {
            mode: call.mode,
            eps: call.eps,
        },
    )
}

fn gelu(
    k: &Kern,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    act: &mut CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    let x = rank3(gate, &k.stream, rows, width)?;
    let up = rank3(up, &k.stream, rows, width)?;
    let mut y = TensorMut(rank3(act, &k.stream, rows, width)?);
    k.common.gate.gelu_tanh_mul(&k.stream, x, up, &mut y)
}

fn swiglu(
    k: &Kern,
    meta: &CudaSlice<i32>,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    act: &mut CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    let rows_domain = domain(meta, &k.stream, rows)?;
    let x = rank3(gate, &k.stream, rows, width)?;
    let up = rank3(up, &k.stream, rows, width)?;
    let mut y = TensorMut(rank3(act, &k.stream, rows, width)?);
    k.common.gate.swiglu(&k.stream, rows_domain, x, up, &mut y)
}

fn residual(k: &Kern, a: &mut CudaSlice<bf16>, b: &CudaSlice<bf16>, n: usize) -> Result<()> {
    let count = i64::try_from(n)?;
    let (ap, _ag) = a.device_ptr_mut(&k.stream);
    let (bp, _bg) = b.device_ptr(&k.stream);
    unsafe {
        k.stream
            .launch_builder(&k.residual_scale)
            .arg(&ap)
            .arg(&bp)
            .arg(&1.0f32)
            .arg(&ap)
            .arg(&count)
            .launch(ops::flat(n))?;
    }
    Ok(())
}

struct RopeBind<'a> {
    qkv: &'a mut CudaSlice<bf16>,
    cache_k: &'a mut CudaSlice<bf16>,
    cache_v: &'a mut CudaSlice<bf16>,
    start: &'a CudaSlice<i32>,
    qn: &'a CudaSlice<bf16>,
    kn: &'a CudaSlice<bf16>,
    table: &'a CudaSlice<bf16>,
}

fn rope(
    k: &Kern,
    buf: &mut Buf,
    qn: &CudaSlice<bf16>,
    kn: &CudaSlice<bf16>,
    table: &CudaSlice<bf16>,
    g: RopeGeom,
) -> Result<()> {
    match g.slot {
        KvSlot::Shared => rope_launch(
            k,
            RopeBind {
                qkv: &mut buf.qkv,
                cache_k: &mut buf.k,
                cache_v: &mut buf.v,
                start: &buf.start,
                qn,
                kn,
                table,
            },
            g,
        ),
        KvSlot::Depth => rope_launch(
            k,
            RopeBind {
                qkv: &mut buf.qkv,
                cache_k: &mut buf.dk,
                cache_v: &mut buf.dv,
                start: &buf.start,
                qn,
                kn,
                table,
            },
            g,
        ),
        KvSlot::Layer(index) => {
            let Buf {
                qkv, lk, lv, start, ..
            } = buf;
            ensure!(index < lk.len() && index < lv.len(), "layer cache");
            rope_launch(
                k,
                RopeBind {
                    qkv,
                    cache_k: &mut lk[index],
                    cache_v: &mut lv[index],
                    start,
                    qn,
                    kn,
                    table,
                },
                g,
            )
        }
    }
}

fn rope_launch(k: &Kern, bind: RopeBind<'_>, g: RopeGeom) -> Result<()> {
    let function = match g.kind {
        RopeKind::OnePlus => &k.rope_one_plus,
        RopeKind::Cast => &k.rope_cast,
        RopeKind::Skip => &k.rope_skip,
    };
    let rows = i32::try_from(g.rows)?;
    let q_heads = i32::try_from(g.q_heads)?;
    let kv_heads = i32::try_from(g.kv_heads)?;
    let dim = i32::try_from(g.dim)?;
    let capacity = i32::try_from(g.capacity)?;
    let q_stride = u64::try_from(g.q_row_stride)?;
    let k_offset = u64::try_from(g.k_offset)?;
    let k_stride = u64::try_from(g.k_row_stride)?;
    let v_offset = u64::try_from(g.v_offset)?;
    let v_stride = u64::try_from(g.v_row_stride)?;
    let ring = 0i32;
    let cache_start = -1i32; // cache_start < 0 writes KV at *start + row.
    let grid = g.rows * (g.q_heads + 2 * g.kv_heads);
    unsafe {
        k.stream
            .launch_builder(function)
            .arg(bind.qkv)
            .arg(bind.cache_k)
            .arg(bind.cache_v)
            .arg(bind.qn)
            .arg(bind.kn)
            .arg(bind.table)
            .arg(bind.start)
            .arg(&cache_start)
            .arg(&rows)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&dim)
            .arg(&capacity)
            .arg(&ring)
            .arg(&g.eps)
            .arg(&q_stride)
            .arg(&k_offset)
            .arg(&k_stride)
            .arg(&v_offset)
            .arg(&v_stride)
            .launch(ops::grid(grid, 128))?;
    }
    Ok(())
}

struct Embed<'a> {
    weight: &'a CudaSlice<bf16>,
    ids: &'a CudaSlice<u32>,
    y: &'a mut CudaSlice<bf16>,
    id_off: usize,
    y_off: usize,
    rows: usize,
    dim: usize,
}

fn embed_tokens(k: &Kern, call: Embed<'_>) -> Result<()> {
    let Embed {
        weight,
        ids,
        y,
        id_off,
        y_off,
        rows,
        dim,
    } = call;
    let id_end = id_off.checked_add(rows).context("embed ids")?;
    let y_elems = rows.checked_mul(dim).context("embed")?;
    let y_end = y_off.checked_add(y_elems).context("embed")?;
    ensure!(id_end <= ids.len() && y_end <= y.len(), "embed");
    let rows_i = i32::try_from(rows)?;
    let dim_i = i32::try_from(dim)?;
    let ids = ids.slice(id_off..id_end);
    let mut y = y.slice_mut(y_off..y_end);
    unsafe {
        k.stream
            .launch_builder(&k.ops.embed)
            .arg(weight)
            .arg(&ids)
            .arg(&mut y)
            .arg(&rows_i)
            .arg(&dim_i)
            .launch(ops::flat(y_elems))?;
    }
    Ok(())
}

fn download_elems(k: &Kern, src: &CudaSlice<bf16>, start: usize, n: usize) -> Result<Vec<f32>> {
    let end = start.checked_add(n).context("score offset")?;
    ensure!(end <= src.len(), "score row");
    let raw = k.stream.clone_dtoh(&src.slice(start..end))?;
    k.stream.synchronize()?;
    k.d2h.set(k.d2h.get() + (n * 2) as u64);
    k.drains.set(k.drains.get() + 1);
    let out: Vec<f32> = raw.iter().map(|value| value.to_f32()).collect();
    ensure!(
        out.iter().all(|value| value.is_finite()),
        "Runner FAIL non-finite candidate"
    );
    Ok(out)
}

fn download_f32(k: &Kern, src: &CudaSlice<bf16>, n: usize) -> Result<Vec<f32>> {
    download_elems(k, src, 0, n)
}

fn download_row(k: &Kern, src: &CudaSlice<bf16>, row: usize, width: usize) -> Result<Vec<f32>> {
    let start = row.checked_mul(width).context("score offset")?;
    download_elems(k, src, start, width)
}

fn depth_d2d_bytes() -> u64 {
    (DP_ROWS * DP_H * 2 + HEAD_N * DP_H * 2 + 2 * HEAD_N * (CODEBOOK as usize) * 2) as u64
}

fn traffic(text_rows: usize, audio_steps: usize, depth_frames: usize) -> (u64, u64, u64, u64) {
    let passes = audio_steps + 1;
    let h2d = (text_rows * BB_H * 2) as u64
        + (passes * 36) as u64
        + (audio_steps * BOOKS * 4) as u64
        + (depth_frames * (32 + 64 + BB_H * 2 + 4)) as u64;
    let d2h =
        (passes * BB_VOCAB * 2) as u64 + (depth_frames * HEAD_N * (CODEBOOK as usize) * 2) as u64;
    let d2d = (depth_frames as u64) * depth_d2d_bytes() + (audio_steps * BB_H * 2) as u64;
    let drains = (passes * 3 + 1 + audio_steps + depth_frames * 5) as u64;
    (drains, h2d, d2h, d2d)
}

fn coverage() -> BTreeSet<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    for pair in [
        (TEXT_Q, TEXT_H),
        (TEXT_KVW, TEXT_H),
        (TEXT_H, TEXT_Q),
        (TEXT_I, TEXT_H),
        (TEXT_H, TEXT_I),
        (TEXT_OUT, TEXT_H),
        (BB_Q, BB_H),
        (BB_KVW, BB_H),
        (BB_H, BB_Q),
        (BB_I, BB_H),
        (BB_H, BB_I),
        (BB_VOCAB, BB_H),
        (DP_Q, DP_H),
        (DP_KVW, DP_H),
        (DP_H, DP_Q),
        (DP_I, DP_H),
        (DP_H, DP_I),
        (DP_H, BB_H),
        (CODEBOOK as usize, DP_H),
    ] {
        pairs.insert(pair);
    }
    pairs
}

fn read_bf16(ckpt: &MappedCheckpoint, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
    let view = ckpt
        .tensor(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        view.dtype() == Dtype::BF16,
        "{name} dtype {:?}",
        view.dtype()
    );
    ensure!(
        view.shape() == shape,
        "{name} shape {:?} != {shape:?}",
        view.shape()
    );
    let mut bytes = vec![0u8; view.data().len()];
    let shard = ckpt
        .index
        .weight_map
        .get(name)
        .with_context(|| format!("index lacks {name}"))?;
    ckpt.read_mapped_range(shard, view.data(), &mut bytes)?;
    let elems = shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim).context("shape"))?;
    ensure!(bytes.len() == elems * 2, "{name} byte length");
    let mut host = Vec::with_capacity(elems);
    for chunk in bytes.as_chunks::<2>().0 {
        host.push(bf16::from_bits(u16::from_le_bytes(*chunk)));
    }
    Ok(host)
}

fn upload(stream: &Arc<CudaStream>, host: &[bf16], bytes: &mut usize) -> Result<CudaSlice<bf16>> {
    let device = stream.clone_htod(host)?;
    stream.synchronize()?;
    *bytes += host.len() * 2;
    Ok(device)
}

fn load_vec(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    len: usize,
    bytes: &mut usize,
) -> Result<CudaSlice<bf16>> {
    let host = read_bf16(ckpt, name, &[len])?;
    upload(stream, &host, bytes)
}

fn load_matrix(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    output: usize,
    input: usize,
    bytes: &mut usize,
) -> Result<CudaSlice<bf16>> {
    let host = read_bf16(ckpt, name, &[output, input])?;
    upload(stream, &host, bytes)
}

fn alloc_buf(stream: &Arc<CudaStream>) -> Result<Buf> {
    let mut lk = Vec::with_capacity(BB_LAYERS);
    let mut lv = Vec::with_capacity(BB_LAYERS);
    for _ in 0..BB_LAYERS {
        lk.push(stream.alloc_zeros(KV_ELEMS)?);
        lv.push(stream.alloc_zeros(KV_ELEMS)?);
    }
    Ok(Buf {
        h: stream.alloc_zeros(H_ELEMS)?,
        n: stream.alloc_zeros(H_ELEMS)?,
        d: stream.alloc_zeros(D_ELEMS)?,
        gate: stream.alloc_zeros(MLP_ELEMS)?,
        up: stream.alloc_zeros(MLP_ELEMS)?,
        act: stream.alloc_zeros(MLP_ELEMS)?,
        qkv: stream.alloc_zeros(QKV_ELEMS)?,
        k: stream.alloc_zeros(KV_ELEMS)?,
        v: stream.alloc_zeros(KV_ELEMS)?,
        lk,
        lv,
        dk: stream.alloc_zeros(DP_KV * DP_ROWS * DP_DIM)?,
        dv: stream.alloc_zeros(DP_KV * DP_ROWS * DP_DIM)?,
        attn: stream.alloc_zeros(ATTN_ELEMS)?,
        audio: stream.alloc_zeros(AUDIO_ELEMS)?,
        stack: stream.alloc_zeros(STACK_ELEMS)?,
        head_in: stream.alloc_zeros(HEAD_IN_ELEMS)?,
        head_out: stream.alloc_zeros(HEAD_OUT_ELEMS)?,
        embed_ws: stream.alloc_zeros(EMBED_WS)?,
        meta: stream.alloc_zeros(8)?,
        start: stream.alloc_zeros(1)?,
        tok: stream.alloc_zeros(TEXT_MAX)?,
        ids: stream.alloc_zeros(IDS_ELEMS)?,
    })
}

fn upload_table(
    stream: &Arc<CudaStream>,
    positions: usize,
    dim: usize,
    law: RopeLaw,
) -> Result<CudaSlice<bf16>> {
    let host = rope_table(positions, dim, law)?;
    let device = stream.clone_htod(&host)?;
    stream.synchronize()?;
    Ok(device)
}

struct Attn<'a> {
    q: &'a CudaSlice<bf16>,
    k: &'a CudaSlice<bf16>,
    v: &'a CudaSlice<bf16>,
    o: &'a CudaSlice<bf16>,
    qn: &'a CudaSlice<bf16>,
    kn: &'a CudaSlice<bf16>,
    table: &'a CudaSlice<bf16>,
    flash: &'a mut Flash,
    rows: usize,
    hidden: usize,
    q_width: usize,
    kv_width: usize,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    eps: f32,
    kind: RopeKind,
    start: i32,
    capacity: usize,
    slot: KvSlot,
    causal: bool,
    window_left: Option<usize>,
}

struct Mlp<'a> {
    gate: &'a CudaSlice<bf16>,
    up: &'a CudaSlice<bf16>,
    down: &'a CudaSlice<bf16>,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    gelu: bool,
}

struct DepthIn<'a> {
    layers: &'a [DecoderLayer],
    norm: &'a CudaSlice<bf16>,
    projector: &'a CudaSlice<bf16>,
    embed: &'a CudaSlice<bf16>,
    table: &'a CudaSlice<bf16>,
    flash: &'a mut Flash,
    cond: &'a [f32],
    ids: &'a [u32],
    frame: usize,
    slot: KvSlot,
}

fn tensor_name(prefix: &str, index: usize, leaf: &str) -> String {
    format!("{prefix}.{index}.{leaf}")
}

fn load_text_embed(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    bytes: &mut usize,
) -> Result<CudaSlice<bf16>> {
    let mut host = read_bf16(
        ckpt,
        "text_encoder.embed_tokens.weight",
        &[TEXT_VOCAB as usize, TEXT_H],
    )?;
    let scale = bf16::from_f32((TEXT_H as f64).sqrt() as f32).to_f32();
    for value in &mut host {
        *value = bf16::from_f32(value.to_f32() * scale);
    }
    upload(stream, &host, bytes)
}

fn load_text_layer(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    index: usize,
    bytes: &mut usize,
) -> Result<TextLayer> {
    let prefix = "text_encoder.layers";
    Ok(TextLayer {
        pre_attn: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "pre_self_attn_layernorm.weight"),
            TEXT_H,
            bytes,
        )?,
        post_attn: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "post_self_attn_layernorm.weight"),
            TEXT_H,
            bytes,
        )?,
        pre_ffn: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "pre_feedforward_layernorm.weight"),
            TEXT_H,
            bytes,
        )?,
        post_ffn: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "post_feedforward_layernorm.weight"),
            TEXT_H,
            bytes,
        )?,
        q: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.q_proj.weight"),
            TEXT_Q,
            TEXT_H,
            bytes,
        )?,
        k: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.k_proj.weight"),
            TEXT_KVW,
            TEXT_H,
            bytes,
        )?,
        v: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.v_proj.weight"),
            TEXT_KVW,
            TEXT_H,
            bytes,
        )?,
        o: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.o_proj.weight"),
            TEXT_H,
            TEXT_Q,
            bytes,
        )?,
        qn: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.q_norm.weight"),
            TEXT_DIM,
            bytes,
        )?,
        kn: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.k_norm.weight"),
            TEXT_DIM,
            bytes,
        )?,
        gate: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "mlp.gate_proj.weight"),
            TEXT_I,
            TEXT_H,
            bytes,
        )?,
        up: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "mlp.up_proj.weight"),
            TEXT_I,
            TEXT_H,
            bytes,
        )?,
        down: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "mlp.down_proj.weight"),
            TEXT_H,
            TEXT_I,
            bytes,
        )?,
        full: (index + 1).is_multiple_of(6),
    })
}

fn load_decoder_layer(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    prefix: &str,
    index: usize,
    spec: DecLoad,
    bytes: &mut usize,
) -> Result<DecoderLayer> {
    let qn = if spec.qk_norm {
        load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.q_norm.weight"),
            spec.dim,
            bytes,
        )?
    } else {
        stream.alloc_zeros(spec.dim)?
    };
    let kn = if spec.qk_norm {
        load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.k_norm.weight"),
            spec.dim,
            bytes,
        )?
    } else {
        stream.alloc_zeros(spec.dim)?
    };
    Ok(DecoderLayer {
        pre: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "input_layernorm.weight"),
            spec.hidden,
            bytes,
        )?,
        post: load_vec(
            ckpt,
            stream,
            &tensor_name(prefix, index, "post_attention_layernorm.weight"),
            spec.hidden,
            bytes,
        )?,
        q: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.q_proj.weight"),
            spec.q,
            spec.hidden,
            bytes,
        )?,
        k: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.k_proj.weight"),
            spec.kv,
            spec.hidden,
            bytes,
        )?,
        v: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.v_proj.weight"),
            spec.kv,
            spec.hidden,
            bytes,
        )?,
        o: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "self_attn.o_proj.weight"),
            spec.hidden,
            spec.q,
            bytes,
        )?,
        qn,
        kn,
        gate: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "mlp.gate_proj.weight"),
            spec.intermediate,
            spec.hidden,
            bytes,
        )?,
        up: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "mlp.up_proj.weight"),
            spec.intermediate,
            spec.hidden,
            bytes,
        )?,
        down: load_matrix(
            ckpt,
            stream,
            &tensor_name(prefix, index, "mlp.down_proj.weight"),
            spec.hidden,
            spec.intermediate,
            bytes,
        )?,
    })
}

// Stored head is [1024, 2051]. The GEMM weight is [2051, 1024].
fn load_heads(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    bytes: &mut usize,
) -> Result<Vec<CudaSlice<bf16>>> {
    let vocab = CODEBOOK as usize;
    let host = read_bf16(
        ckpt,
        "depth_decoder.codebooks_head.weight",
        &[HEAD_N, DP_H, vocab],
    )?;
    let mut heads = Vec::with_capacity(HEAD_N);
    for head in 0..HEAD_N {
        let mut transposed = vec![bf16::ZERO; vocab * DP_H];
        for row in 0..DP_H {
            for col in 0..vocab {
                let src = (head * DP_H + row) * vocab + col;
                transposed[col * DP_H + row] = host[src];
            }
        }
        heads.push(upload(stream, &transposed, bytes)?);
    }
    Ok(heads)
}

fn load_weights(
    ckpt: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    bytes: &mut usize,
) -> Result<Weights> {
    let text_embed = load_text_embed(ckpt, stream, bytes)?;
    let mut text = Vec::with_capacity(TEXT_LAYERS);
    for index in 0..TEXT_LAYERS {
        text.push(load_text_layer(ckpt, stream, index, bytes)?);
    }
    let text_norm = load_vec(ckpt, stream, "text_encoder.norm.weight", TEXT_H, bytes)?;
    let text_proj = load_matrix(
        ckpt,
        stream,
        "text_encoder_proj.weight",
        TEXT_OUT,
        TEXT_H,
        bytes,
    )?;
    let backbone_spec = DecLoad {
        hidden: BB_H,
        intermediate: BB_I,
        q: BB_Q,
        kv: BB_KVW,
        dim: BB_DIM,
        qk_norm: true,
    };
    let mut backbone = Vec::with_capacity(BB_LAYERS);
    for index in 0..BB_LAYERS {
        backbone.push(load_decoder_layer(
            ckpt,
            stream,
            "backbone_model.layers",
            index,
            backbone_spec,
            bytes,
        )?);
    }
    let backbone_norm = load_vec(ckpt, stream, "backbone_model.norm.weight", BB_H, bytes)?;
    let lm = load_matrix(ckpt, stream, "lm_head.weight", BB_VOCAB, BB_H, bytes)?;
    let audio = load_matrix(
        ckpt,
        stream,
        "depth_decoder.model.embed_tokens.weight",
        BOOKS * CODEBOOK as usize,
        BB_H,
        bytes,
    )?;
    let projector = load_matrix(
        ckpt,
        stream,
        "depth_decoder.model.inputs_embeds_projector.weight",
        DP_H,
        BB_H,
        bytes,
    )?;
    let depth_spec = DecLoad {
        hidden: DP_H,
        intermediate: DP_I,
        q: DP_Q,
        kv: DP_KVW,
        dim: DP_DIM,
        qk_norm: false,
    };
    let mut depth = Vec::with_capacity(DP_LAYERS);
    for index in 0..DP_LAYERS {
        depth.push(load_decoder_layer(
            ckpt,
            stream,
            "depth_decoder.model.layers",
            index,
            depth_spec,
            bytes,
        )?);
    }
    let depth_norm = load_vec(ckpt, stream, "depth_decoder.model.norm.weight", DP_H, bytes)?;
    let heads = load_heads(ckpt, stream, bytes)?;
    ensure!(
        text.len() == TEXT_LAYERS && backbone.len() == BB_LAYERS && depth.len() == DP_LAYERS,
        "layer count"
    );
    ensure!(heads.len() == HEAD_N, "head count");
    ensure!(!text[0].full && text[5].full, "text layer 0 is sliding");
    Ok(Weights {
        text_embed,
        text,
        text_norm,
        text_proj,
        backbone,
        backbone_norm,
        lm,
        audio,
        projector,
        depth,
        depth_norm,
        heads,
    })
}

fn attend(k: &Kern, buf: &mut Buf, blas: &Blas, call: Attn<'_>) -> Result<()> {
    ensure!(call.q_width == call.q_heads * call.dim, "q width");
    ensure!(call.kv_width == call.kv_heads * call.dim, "kv width");
    let rows = call.rows;
    let q_width = call.q_width;
    let kv_width = call.kv_width;
    let (k_off, v_off) = qkv_layout(rows, call.hidden, q_width, kv_width)?;
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.q,
            x: &mut buf.n,
            y: &mut buf.qkv,
            x_off: 0,
            y_off: 0,
            logical: rows,
            output: q_width,
            input: call.hidden,
        },
    )?;
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.k,
            x: &mut buf.n,
            y: &mut buf.qkv,
            x_off: 0,
            y_off: k_off,
            logical: rows,
            output: kv_width,
            input: call.hidden,
        },
    )?;
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.v,
            x: &mut buf.n,
            y: &mut buf.qkv,
            x_off: 0,
            y_off: v_off,
            logical: rows,
            output: kv_width,
            input: call.hidden,
        },
    )?;
    ensure!(call.start >= 0, "attention origin");
    rope(
        k,
        buf,
        call.qn,
        call.kn,
        call.table,
        RopeGeom {
            rows,
            q_heads: call.q_heads,
            kv_heads: call.kv_heads,
            dim: call.dim,
            capacity: call.capacity,
            q_row_stride: q_width,
            k_offset: k_off,
            k_row_stride: kv_width,
            v_offset: v_off,
            v_row_stride: kv_width,
            eps: call.eps,
            kind: call.kind,
            slot: call.slot,
        },
    )?;
    let shape = Shape {
        rows,
        start: usize::try_from(call.start)?,
        q_heads: call.q_heads,
        kv_heads: call.kv_heads,
        capacity: call.capacity,
        head_dim: call.dim,
        causal: call.causal,
        window_left: call.window_left,
    };
    match call.slot {
        KvSlot::Shared => {
            call.flash
                .run(&buf.qkv, q_width, &buf.k, &buf.v, &mut buf.attn, shape)?;
        }
        KvSlot::Depth => {
            call.flash
                .run(&buf.qkv, q_width, &buf.dk, &buf.dv, &mut buf.attn, shape)?;
        }
        KvSlot::Layer(index) => {
            let Buf {
                qkv, lk, lv, attn, ..
            } = buf;
            ensure!(index < lk.len() && index < lv.len(), "layer cache");
            call.flash
                .run(qkv, q_width, &lk[index], &lv[index], attn, shape)?;
        }
    }
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.o,
            x: &mut buf.attn,
            y: &mut buf.n,
            x_off: 0,
            y_off: 0,
            logical: rows,
            output: call.hidden,
            input: q_width,
        },
    )
}

fn mlp(k: &Kern, buf: &mut Buf, blas: &Blas, call: Mlp<'_>) -> Result<()> {
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.gate,
            x: &mut buf.n,
            y: &mut buf.gate,
            x_off: 0,
            y_off: 0,
            logical: call.rows,
            output: call.intermediate,
            input: call.hidden,
        },
    )?;
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.up,
            x: &mut buf.n,
            y: &mut buf.up,
            x_off: 0,
            y_off: 0,
            logical: call.rows,
            output: call.intermediate,
            input: call.hidden,
        },
    )?;
    if call.gelu {
        gelu(
            k,
            &buf.gate,
            &buf.up,
            &mut buf.act,
            call.rows,
            call.intermediate,
        )?;
    } else {
        swiglu(
            k,
            &buf.meta,
            &buf.gate,
            &buf.up,
            &mut buf.act,
            call.rows,
            call.intermediate,
        )?;
    }
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: call.down,
            x: &mut buf.act,
            y: &mut buf.n,
            x_off: 0,
            y_off: 0,
            logical: call.rows,
            output: call.hidden,
            input: call.intermediate,
        },
    )
}

fn text_layer(
    layer: &TextLayer,
    k: &Kern,
    flash: &mut Flash,
    buf: &mut Buf,
    blas: &Blas,
    step: TextStep,
) -> Result<()> {
    let rows = step.rows;
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.h,
            y: &mut buf.n,
            w: &layer.pre_attn,
            rows,
            width: TEXT_H,
            mode: RmsMode::ZeroCentered,
            eps: TEXT_EPS,
        },
    )?;
    let table = if step.plant == TextPlant::RopeOmitted {
        &k.table_identity
    } else if layer.full {
        &k.table_full
    } else {
        &k.table_slide
    };
    let start = 0;
    attend(
        k,
        buf,
        blas,
        Attn {
            q: &layer.q,
            k: &layer.k,
            v: &layer.v,
            o: &layer.o,
            qn: &layer.qn,
            kn: &layer.kn,
            table,
            flash,
            rows,
            hidden: TEXT_H,
            q_width: TEXT_Q,
            kv_width: TEXT_KVW,
            q_heads: TEXT_HEADS,
            kv_heads: TEXT_KV,
            dim: TEXT_DIM,
            eps: TEXT_EPS,
            kind: RopeKind::OnePlus,
            start,
            capacity: rows,
            slot: KvSlot::Shared,
            causal: false,
            window_left: step.window_left,
        },
    )?;
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.n,
            y: &mut buf.attn,
            w: &layer.post_attn,
            rows,
            width: TEXT_H,
            mode: RmsMode::ZeroCentered,
            eps: TEXT_EPS,
        },
    )?;
    residual(k, &mut buf.h, &buf.attn, rows * TEXT_H)?;
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.h,
            y: &mut buf.n,
            w: &layer.pre_ffn,
            rows,
            width: TEXT_H,
            mode: RmsMode::ZeroCentered,
            eps: TEXT_EPS,
        },
    )?;
    mlp(
        k,
        buf,
        blas,
        Mlp {
            gate: &layer.gate,
            up: &layer.up,
            down: &layer.down,
            rows,
            hidden: TEXT_H,
            intermediate: TEXT_I,
            gelu: true,
        },
    )?;
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.n,
            y: &mut buf.attn,
            w: &layer.post_ffn,
            rows,
            width: TEXT_H,
            mode: RmsMode::ZeroCentered,
            eps: TEXT_EPS,
        },
    )?;
    residual(k, &mut buf.h, &buf.attn, rows * TEXT_H)
}

fn decoder_layer(k: &Kern, buf: &mut Buf, blas: &Blas, pass: DecPass<'_>) -> Result<()> {
    let rows = pass.geom.rows;
    let hidden = pass.geom.hidden;
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.h,
            y: &mut buf.n,
            w: &pass.layer.pre,
            rows,
            width: hidden,
            mode: RmsMode::Standard,
            eps: pass.geom.eps,
        },
    )?;
    attend(
        k,
        buf,
        blas,
        Attn {
            q: &pass.layer.q,
            k: &pass.layer.k,
            v: &pass.layer.v,
            o: &pass.layer.o,
            qn: &pass.layer.qn,
            kn: &pass.layer.kn,
            table: pass.table,
            flash: pass.flash,
            rows,
            hidden,
            q_width: pass.geom.q_heads * pass.geom.dim,
            kv_width: pass.geom.kv_heads * pass.geom.dim,
            q_heads: pass.geom.q_heads,
            kv_heads: pass.geom.kv_heads,
            dim: pass.geom.dim,
            eps: pass.geom.eps,
            kind: pass.kind,
            start: pass.origin,
            capacity: pass.capacity,
            slot: pass.slot,
            causal: true,
            window_left: None,
        },
    )?;
    residual(k, &mut buf.h, &buf.n, rows * hidden)?;
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.h,
            y: &mut buf.n,
            w: &pass.layer.post,
            rows,
            width: hidden,
            mode: RmsMode::Standard,
            eps: pass.geom.eps,
        },
    )?;
    mlp(
        k,
        buf,
        blas,
        Mlp {
            gate: &pass.layer.gate,
            up: &pass.layer.up,
            down: &pass.layer.down,
            rows,
            hidden,
            intermediate: pass.geom.intermediate,
            gelu: false,
        },
    )?;
    residual(k, &mut buf.h, &buf.n, rows * hidden)
}

fn launch_row_sum(
    k: &Kern,
    table: &CudaSlice<bf16>,
    ids: &CudaSlice<i32>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    dim: usize,
) -> Result<()> {
    let rows_i = i32::try_from(rows)?;
    let books_i = i32::try_from(BOOKS)?;
    let dim_i = i32::try_from(dim)?;
    ensure!(
        rows * BOOKS <= ids.len() && rows * dim <= out.len(),
        "audio sum"
    );
    unsafe {
        k.stream
            .launch_builder(&k.embed_rows_sum)
            .arg(table)
            .arg(ids)
            .arg(out)
            .arg(&rows_i)
            .arg(&books_i)
            .arg(&CODEBOOK)
            .arg(&dim_i)
            .launch(ops::flat(rows * dim))?;
    }
    Ok(())
}

fn audio_source_book(book: usize) -> usize {
    (book + 1) % BOOKS
}

fn rotate_audio_table(k: &Kern, table: &CudaSlice<bf16>) -> Result<CudaSlice<bf16>> {
    let block = (CODEBOOK as usize) * BB_H;
    let total = BOOKS * block;
    ensure!(table.len() == total, "audio table");
    let mut scratch = k.stream.alloc_zeros::<bf16>(total)?;
    for book in 0..BOOKS {
        let src_book = audio_source_book(book);
        let src = src_book * block;
        let dst = book * block;
        copy_d2d(
            k,
            &table.slice(src..src + block),
            &mut scratch.slice_mut(dst..dst + block),
        )?;
    }
    Ok(scratch)
}

fn sum_audio(
    k: &Kern,
    table: &CudaSlice<bf16>,
    buf: &mut Buf,
    codes: &[i32],
    rows: usize,
) -> Result<Vec<bf16>> {
    ensure!(codes.len() == rows * BOOKS, "audio code length");
    ensure!(rows > 0 && rows <= DP_FRAMES, "audio rows {rows}");
    for &id in codes {
        ensure!((0..CODEBOOK).contains(&id), "audio code {id}");
    }
    copy_h2d(k, codes, &mut buf.ids.slice_mut(..codes.len()))?;
    launch_row_sum(k, table, &buf.ids, &mut buf.audio, rows, BB_H)?;
    let wide = download_f32(k, &buf.audio, rows * BB_H)?;
    Ok(wide.into_iter().map(bf16::from_f32).collect())
}

fn check_positions(positions: &Positions) -> Result<()> {
    let Positions {
        text,
        audio,
        eos,
        seq,
    } = positions;
    ensure!((1..=BB_MAX).contains(seq), "seq {seq}");
    ensure!(
        text.len() <= TEXT_MAX && audio.len() <= DP_FRAMES,
        "position counts"
    );
    let mut seen = vec![0u8; *seq];
    for &index in text {
        ensure!(index < *seq && seen[index] == 0, "text position {index}");
        seen[index] = 1;
    }
    for &index in audio {
        ensure!(index > 0 && index < *seq, "audio position {index}");
        ensure!(seen[index] == 0, "duplicate position {index}");
        seen[index] = 2;
    }
    for &index in eos {
        ensure!(index < *seq && seen[index] == 0, "eos position {index}");
        seen[index] = 3;
    }
    ensure!(seen.iter().all(|&slot| slot != 0), "unassigned position");
    Ok(())
}

fn assemble_hidden(
    projected: &[f32],
    positions: &Positions,
    audio: &[bf16],
    eos: Option<&[bf16]>,
) -> Result<Vec<bf16>> {
    ensure!(
        projected.len() == positions.text.len() * BB_H,
        "projected length"
    );
    ensure!(
        audio.len() == positions.audio.len() * BB_H,
        "audio sum length"
    );
    if positions.eos.is_empty() {
        ensure!(eos.is_none(), "eos embedding");
    } else {
        ensure!(eos.is_some_and(|row| row.len() == BB_H), "eos embedding");
    }
    let mut hidden = vec![bf16::ZERO; positions.seq * BB_H];
    for (row, &pos) in positions.text.iter().enumerate() {
        for col in 0..BB_H {
            hidden[pos * BB_H + col] = bf16::from_f32(projected[row * BB_H + col]);
        }
    }
    for (frame, &pos) in positions.audio.iter().enumerate() {
        let dst = pos * BB_H;
        let src = frame * BB_H;
        hidden[dst..dst + BB_H].copy_from_slice(&audio[src..src + BB_H]);
    }
    if let Some(row) = eos {
        for &pos in &positions.eos {
            let dst = pos * BB_H;
            hidden[dst..dst + BB_H].copy_from_slice(row);
        }
    }
    Ok(hidden)
}

fn depth_ids(codes: &[i32]) -> Result<[u32; DP_ROWS]> {
    ensure!(codes.len() == DP_ROWS, "depth frame codes");
    ensure!(codes[0] == 0, "depth pad");
    let mut ids = [0u32; DP_ROWS];
    let vocab = CODEBOOK as usize;
    for book in 0..HEAD_N {
        let code = codes[book + 1];
        ensure!((0..CODEBOOK).contains(&code), "depth code {code}");
        let index = book * vocab + usize::try_from(code)?;
        ensure!(index < BOOKS * vocab, "depth embed {index}");
        ids[book + 1] = u32::try_from(index)?;
    }
    Ok(ids)
}

fn depth_frame(k: &Kern, buf: &mut Buf, blas: &Blas, input: DepthIn<'_>) -> Result<()> {
    let DepthIn {
        layers,
        norm,
        projector,
        embed,
        table,
        flash,
        cond,
        ids,
        frame,
        slot,
    } = input;
    ensure!(cond.len() == BB_H && ids.len() == DP_ROWS, "depth frame");
    stage_depth_ids(k, buf, ids)?;
    embed_tokens(
        k,
        Embed {
            weight: embed,
            ids: &buf.tok,
            y: &mut buf.embed_ws,
            id_off: 0,
            y_off: 0,
            rows: DP_ROWS,
            dim: BB_H,
        },
    )?;
    stage_depth_cond(k, buf, cond)?;
    upload_origin(k, &mut buf.start, 0)?;
    depth_after(
        k,
        buf,
        blas,
        DepthRun {
            layers,
            norm,
            projector,
            table,
            flash,
            frame,
            slot,
        },
    )
}

fn stage_depth_ids(k: &Kern, buf: &mut Buf, ids: &[u32]) -> Result<()> {
    ensure!(ids.len() == DP_ROWS, "depth frame");
    k.stream.memset_zeros(&mut buf.embed_ws)?;
    copy_h2d(k, ids, &mut buf.tok.slice_mut(..DP_ROWS))
}

fn stage_depth_cond(k: &Kern, buf: &mut Buf, cond: &[f32]) -> Result<()> {
    ensure!(cond.len() == BB_H, "depth frame");
    let mut row = vec![bf16::ZERO; BB_H];
    for (dst, &value) in row.iter_mut().zip(cond.iter()) {
        *dst = bf16::from_f32(value);
    }
    copy_h2d(k, &row, &mut buf.embed_ws.slice_mut(..BB_H))
}

struct DepthRun<'a> {
    layers: &'a [DecoderLayer],
    norm: &'a CudaSlice<bf16>,
    projector: &'a CudaSlice<bf16>,
    table: &'a CudaSlice<bf16>,
    flash: &'a mut Flash,
    frame: usize,
    slot: KvSlot,
}

fn depth_after(k: &Kern, buf: &mut Buf, blas: &Blas, run: DepthRun<'_>) -> Result<()> {
    let DepthRun {
        layers,
        norm,
        projector,
        table,
        flash,
        frame,
        slot,
    } = run;
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: projector,
            x: &mut buf.embed_ws,
            y: &mut buf.h,
            x_off: 0,
            y_off: 0,
            logical: DP_ROWS,
            output: DP_H,
            input: BB_H,
        },
    )?;
    let geom = DecGeom {
        rows: DP_ROWS,
        hidden: DP_H,
        intermediate: DP_I,
        q_heads: DP_HEADS,
        kv_heads: DP_KV,
        dim: DP_DIM,
        eps: DP_EPS,
    };
    for layer in layers {
        decoder_layer(
            k,
            buf,
            blas,
            DecPass {
                layer,
                table,
                flash: &mut *flash,
                kind: RopeKind::Skip,
                geom,
                origin: 0,
                capacity: DP_ROWS,
                slot,
            },
        )?;
    }
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.h,
            y: &mut buf.n,
            w: norm,
            rows: DP_ROWS,
            width: DP_H,
            mode: RmsMode::Standard,
            eps: DP_EPS,
        },
    )?;
    let n = DP_ROWS * DP_H;
    let dst = frame * n;
    copy_d2d(k, &buf.n.slice(..n), &mut buf.stack.slice_mut(dst..dst + n))
}

fn depth_heads(
    k: &Kern,
    buf: &mut Buf,
    blas: &Blas,
    heads: &[CudaSlice<bf16>],
    frames: usize,
    plant: bool,
) -> Result<Vec<f32>> {
    ensure!(heads.len() == HEAD_N, "head count");
    let vocab = CODEBOOK as usize;
    let mut out = vec![0f32; frames * HEAD_N * vocab];
    for book in 0..HEAD_N {
        for frame in 0..frames {
            let src = (frame * DP_ROWS + book + 1) * DP_H;
            let dst = frame * DP_H;
            copy_d2d(
                k,
                &buf.stack.slice(src..src + DP_H),
                &mut buf.head_in.slice_mut(dst..dst + DP_H),
            )?;
        }
        let which = if plant && book == 0 { 1 } else { book };
        gemm(
            &k.stream,
            blas,
            Gemm {
                w: &heads[which],
                x: &mut buf.head_in,
                y: &mut buf.head_out,
                x_off: 0,
                y_off: 0,
                logical: frames,
                output: vocab,
                input: DP_H,
            },
        )?;
        let part = download_f32(k, &buf.head_out, frames * vocab)?;
        for frame in 0..frames {
            let dst = (frame * HEAD_N + book) * vocab;
            let src = frame * vocab;
            out[dst..dst + vocab].copy_from_slice(&part[src..src + vocab]);
        }
    }
    Ok(out)
}

struct ScorePass<'a> {
    layers: &'a [DecoderLayer],
    norm: &'a CudaSlice<bf16>,
    lm: &'a CudaSlice<bf16>,
    table: &'a CudaSlice<bf16>,
    flash: &'a mut Flash,
    rows: usize,
    origin: i32,
    score_row: usize,
}

struct DepthStep<'a> {
    w: &'a Weights,
    flash: &'a mut Flash,
    cond: &'a [f32],
    codes: &'a [i32],
    frame: usize,
    plant: bool,
}

fn upload_hidden(k: &Kern, buf: &mut Buf, src: &[f32]) -> Result<()> {
    ensure!(
        !src.is_empty() && src.len().is_multiple_of(BB_H),
        "hidden rows"
    );
    ensure!(
        src.iter().all(|value| value.is_finite()),
        "non-finite hidden"
    );
    let host: Vec<bf16> = src.iter().copied().map(bf16::from_f32).collect();
    copy_h2d(k, &host, &mut buf.h.slice_mut(..host.len()))
}

fn place_audio_row(k: &Kern, table: &CudaSlice<bf16>, buf: &mut Buf, codes: &[i32]) -> Result<()> {
    ensure!(codes.len() == BOOKS, "audio row");
    for &id in codes {
        ensure!((0..CODEBOOK).contains(&id), "audio code {id}");
    }
    copy_h2d(k, codes, &mut buf.ids.slice_mut(..BOOKS))?;
    launch_row_sum(k, table, &buf.ids, &mut buf.audio, 1, BB_H)?;
    copy_d2d(k, &buf.audio.slice(..BB_H), &mut buf.h.slice_mut(..BB_H))
}

fn backbone_rows(k: &Kern, buf: &mut Buf, blas: &Blas, pass: ScorePass<'_>) -> Result<Vec<f32>> {
    ensure!(pass.rows > 0 && pass.score_row < pass.rows, "score row");
    ensure!(pass.origin >= 0, "score origin");
    let end = usize::try_from(pass.origin)?
        .checked_add(pass.rows)
        .context("backbone cache")?;
    ensure!(end <= BB_MAX, "backbone cache");
    ensure!(pass.layers.len() == BB_LAYERS, "backbone layers");
    set_meta(k, &mut buf.meta, pass.rows)?;
    upload_origin(k, &mut buf.start, pass.origin)?;
    let geom = DecGeom {
        rows: pass.rows,
        hidden: BB_H,
        intermediate: BB_I,
        q_heads: BB_HEADS,
        kv_heads: BB_KV,
        dim: BB_DIM,
        eps: BB_EPS,
    };
    for (index, layer) in pass.layers.iter().enumerate() {
        decoder_layer(
            k,
            buf,
            blas,
            DecPass {
                layer,
                table: pass.table,
                flash: &mut *pass.flash,
                kind: RopeKind::Cast,
                geom,
                origin: pass.origin,
                capacity: BB_MAX,
                slot: KvSlot::Layer(index),
            },
        )?;
    }
    rms(
        k,
        &buf.meta,
        Rms {
            x: &buf.h,
            y: &mut buf.n,
            w: pass.norm,
            rows: pass.rows,
            width: BB_H,
            mode: RmsMode::Standard,
            eps: BB_EPS,
        },
    )?;
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: pass.lm,
            x: &mut buf.n,
            y: &mut buf.d,
            x_off: 0,
            y_off: 0,
            logical: pass.rows,
            output: BB_VOCAB,
            input: BB_H,
        },
    )?;
    download_row(k, &buf.d, pass.score_row, BB_VOCAB)
}

fn depth_input(codes: &[i32]) -> Result<[u32; DP_ROWS]> {
    ensure!(codes.len() == BOOKS, "depth frame codes");
    let mut row = [0i32; DP_ROWS];
    for (book, &code) in codes.iter().take(HEAD_N).enumerate() {
        row[book + 1] = code;
    }
    depth_ids(&row)
}

fn depth_head_row(
    k: &Kern,
    buf: &mut Buf,
    blas: &Blas,
    heads: &[CudaSlice<bf16>],
    frame: usize,
    book: usize,
    plant: bool,
) -> Result<()> {
    let vocab = CODEBOOK as usize;
    let src = (frame * DP_ROWS + book + 1) * DP_H;
    copy_d2d(
        k,
        &buf.stack.slice(src..src + DP_H),
        &mut buf.head_in.slice_mut(..DP_H),
    )?;
    let which = if plant && book == 0 { 1 } else { book };
    gemm(
        &k.stream,
        blas,
        Gemm {
            w: &heads[which],
            x: &mut buf.head_in,
            y: &mut buf.head_out,
            x_off: 0,
            y_off: 0,
            logical: 1,
            output: vocab,
            input: DP_H,
        },
    )
}

fn depth_pack(
    k: &Kern,
    buf: &mut Buf,
    blas: &Blas,
    heads: &[CudaSlice<bf16>],
    frame: usize,
    plant: bool,
) -> Result<()> {
    let vocab = CODEBOOK as usize;
    let pack = gemm_rows(1, DP_H) * vocab;
    let packed = HEAD_N * vocab;
    ensure!(
        pack + packed <= buf.head_out.len() && vocab <= buf.head_in.len(),
        "head pack"
    );
    for book in 0..HEAD_N {
        depth_head_row(k, buf, blas, heads, frame, book, plant)?;
        copy_d2d(
            k,
            &buf.head_out.slice(..vocab),
            &mut buf.head_in.slice_mut(..vocab),
        )?;
        let dst = pack + book * vocab;
        copy_d2d(
            k,
            &buf.head_in.slice(..vocab),
            &mut buf.head_out.slice_mut(dst..dst + vocab),
        )?;
    }
    Ok(())
}

fn clear_depth_kv(k: &Kern, buf: &mut Buf) -> Result<()> {
    k.stream.memset_zeros(&mut buf.dk)?;
    k.stream.memset_zeros(&mut buf.dv)?;
    Ok(())
}

fn depth_step(k: &Kern, buf: &mut Buf, blas: &Blas, step: DepthStep<'_>) -> Result<Vec<f32>> {
    set_meta(k, &mut buf.meta, DP_ROWS)?;
    let ids = depth_input(step.codes)?;
    let vocab = CODEBOOK as usize;
    let pack = gemm_rows(1, DP_H) * vocab;
    let packed = HEAD_N * vocab;
    depth_frame(
        k,
        buf,
        blas,
        DepthIn {
            layers: &step.w.depth,
            norm: &step.w.depth_norm,
            projector: &step.w.projector,
            embed: &step.w.audio,
            table: &k.table_depth,
            flash: step.flash,
            cond: step.cond,
            ids: &ids,
            frame: step.frame,
            slot: KvSlot::Depth,
        },
    )?;
    clear_depth_kv(k, buf)?;
    depth_pack(k, buf, blas, &step.w.heads, step.frame, step.plant)?;
    download_elems(k, &buf.head_out, pack, packed)
}

fn decode_shape(
    projected: &[f32],
    cond: &[f32],
    text_rows: usize,
    fed: &[i32],
    frames: &[i32],
) -> Result<(usize, usize)> {
    ensure!(fed.len().is_multiple_of(BOOKS), "fed codes");
    ensure!(frames.len().is_multiple_of(BOOKS), "depth codes");
    ensure!(cond.len().is_multiple_of(BB_H), "cond");
    let audio_steps = fed.len() / BOOKS;
    let depth_frames = frames.len() / BOOKS;
    ensure!(
        (1..=DP_FRAMES).contains(&depth_frames),
        "frames {depth_frames}"
    );
    ensure!(cond.len() == depth_frames * BB_H, "cond length");
    ensure!((1..=TEXT_MAX).contains(&text_rows), "text rows {text_rows}");
    ensure!(
        audio_steps >= 1 && depth_frames >= audio_steps,
        "decode steps"
    );
    ensure!(text_rows + audio_steps <= BB_MAX, "decode length");
    ensure!(projected.len() == text_rows * BB_H, "projected length");
    ensure!(
        projected.iter().all(|value| value.is_finite()),
        "non-finite projected"
    );
    ensure!(
        cond.iter().all(|value| value.is_finite()),
        "non-finite cond"
    );
    Ok((audio_steps, depth_frames))
}

impl Breeze {
    fn bound_device(&self) -> Result<()> {
        ensure!(self.device.ctx.ordinal() == 0, "device ordinal");
        Ok(())
    }

    pub fn load(weights: &Path, runtime: &Path) -> Result<Self> {
        ensure!(runtime.is_dir(), "runtime directory");
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let rope_module = cubin::module(&device.ctx, "rope")?;
        let embed_module = cubin::module(&device.ctx, "embed")?;
        let act_module = cubin::module(&device.ctx, "act")?;
        let rope_one_plus = rope_module.load_function("qknorm_rope_table_kv_one_plus")?;
        let rope_cast = rope_module.load_function("qknorm_rope_table_kv_cast")?;
        let rope_skip = rope_module.load_function("qknorm_rope_table_kv_skip")?;
        let embed_rows_sum = embed_module.load_function("embed_rows_sum")?;
        let residual_scale = act_module.load_function("residual_scale")?;
        let stream = device.stream.clone();
        let table_slide = upload_table(&stream, TEXT_POSITIONS, TEXT_DIM, text_rope_law(false))?;
        let table_full = upload_table(&stream, TEXT_POSITIONS, TEXT_DIM, text_rope_law(true))?;
        let identity = identity_rope_table(TEXT_POSITIONS, TEXT_DIM)?;
        let table_identity = stream.clone_htod(&identity)?;
        stream.synchronize()?;
        let table_backbone = upload_table(
            &stream,
            BB_MAX,
            BB_DIM,
            RopeLaw::Plain {
                theta: 1_000_000.0,
                divisor: 1.0,
            },
        )?;
        let table_depth = upload_table(&stream, DP_ROWS, DP_DIM, RopeLaw::Llama3)?;
        let (slide_left, slide_right) = flash::window_radii(512);
        ensure!(slide_left == 255 && slide_right == 256, "window radii");
        let text_full = Flash::new(stream.clone(), TEXT_MAX, TEXT_HEADS, TEXT_KV, TEXT_DIM)
            .context("text full attention")?;
        let text_slide = Flash::new_with_score(
            stream.clone(),
            TEXT_MAX,
            TEXT_HEADS,
            TEXT_KV,
            TEXT_DIM,
            Score {
                scale: 0.0625,
                window_left: Some(slide_left),
                window_right: Some(slide_right),
            },
            TEXT_MAX,
        )
        .context("text sliding attention")?;
        let mut backbone_flash = Flash::new(stream.clone(), BB_MAX, BB_HEADS, BB_KV, BB_DIM)
            .context("backbone attention")?;
        let mut depth_flash = Flash::new(stream.clone(), DP_ROWS, DP_HEADS, DP_KV, DP_DIM)
            .context("depth attention")?;
        let buf = alloc_buf(&stream)?;
        backbone_flash.bind_position(&buf.start, BB_MAX - 1)?;
        depth_flash.bind_position(&buf.start, DP_ROWS - 1)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(false, "breeze-m0".to_string())),
        )?;
        blas.declare(1024, &coverage());
        let ckpt = MappedCheckpoint::open(weights).context("open breeze weights")?;
        let mut weight_bytes = 0usize;
        let w = load_weights(&ckpt, &stream, &mut weight_bytes)?;
        drop(ckpt);
        eprintln!("weight_bytes={weight_bytes}");
        Ok(Self {
            device,
            w,
            k: Kern {
                stream,
                ops,
                common,
                rope_one_plus,
                rope_cast,
                rope_skip,
                embed_rows_sum,
                residual_scale,
                table_slide,
                table_full,
                table_identity,
                table_backbone,
                table_depth,
                drains: Cell::new(0),
                h2d: Cell::new(0),
                d2h: Cell::new(0),
                d2d: Cell::new(0),
            },
            flash: Flashes {
                text_full,
                text_slide,
                backbone: backbone_flash,
                depth: depth_flash,
                slide_left,
            },
            buf,
            blas,
            weight_bytes,
        })
    }

    pub fn text(&mut self, token_ids: &[u32], plant: TextPlant) -> Result<Vec<f32>> {
        self.bound_device()?;
        let rows = token_ids.len();
        ensure!((1..=TEXT_MAX).contains(&rows), "text rows {rows}");
        ensure!(
            token_ids
                .iter()
                .all(|&id| id != TEXT_EOI && id < TEXT_VOCAB),
            "text token id"
        );
        let Breeze {
            ref w,
            ref k,
            ref mut flash,
            ref mut buf,
            ref blas,
            ..
        } = *self;
        let slide_left = flash.slide_left;
        ensure!(slide_left > 0, "sliding window left");
        set_meta(k, &mut buf.meta, rows)?;
        copy_h2d(k, token_ids, &mut buf.tok.slice_mut(..rows))?;
        upload_origin(k, &mut buf.start, 0)?;
        embed_tokens(
            k,
            Embed {
                weight: &w.text_embed,
                ids: &buf.tok,
                y: &mut buf.h,
                id_off: 0,
                y_off: 0,
                rows,
                dim: TEXT_H,
            },
        )?;
        for index in 0..w.text.len() {
            let unwindowed = w.text[index].full || plant == TextPlant::WindowIgnored;
            let window_left = if unwindowed { None } else { Some(slide_left) };
            let step = TextStep {
                rows,
                plant,
                window_left,
            };
            if unwindowed {
                text_layer(&w.text[index], k, &mut flash.text_full, buf, blas, step)?;
            } else {
                text_layer(&w.text[index], k, &mut flash.text_slide, buf, blas, step)?;
            }
        }
        rms(
            k,
            &buf.meta,
            Rms {
                x: &buf.h,
                y: &mut buf.n,
                w: &w.text_norm,
                rows,
                width: TEXT_H,
                mode: RmsMode::ZeroCentered,
                eps: TEXT_EPS,
            },
        )?;
        gemm(
            &k.stream,
            blas,
            Gemm {
                w: &w.text_proj,
                x: &mut buf.n,
                y: &mut buf.d,
                x_off: 0,
                y_off: 0,
                logical: rows,
                output: TEXT_OUT,
                input: TEXT_H,
            },
        )?;
        download_f32(k, &buf.d, rows * TEXT_OUT)
    }

    pub fn sliding_layer(&mut self, hidden: &[f32], plant: TextPlant) -> Result<Vec<f32>> {
        self.bound_device()?;
        ensure!(
            matches!(plant, TextPlant::Nominal | TextPlant::WindowIgnored),
            "sliding plant"
        );
        ensure!(hidden.len().is_multiple_of(TEXT_H), "sliding input");
        let rows = hidden.len() / TEXT_H;
        ensure!((1..=TEXT_MAX).contains(&rows), "sliding rows {rows}");
        ensure!(
            hidden.iter().all(|value| value.is_finite()),
            "non-finite sliding input"
        );
        let host: Vec<bf16> = hidden.iter().copied().map(bf16::from_f32).collect();
        let Breeze {
            ref w,
            ref k,
            ref mut flash,
            ref mut buf,
            ref blas,
            ..
        } = *self;
        ensure!(!w.text[0].full, "layer 0 is full");
        copy_h2d(k, &host, &mut buf.h.slice_mut(..host.len()))?;
        set_meta(k, &mut buf.meta, rows)?;
        upload_origin(k, &mut buf.start, 0)?;
        let slide_left = flash.slide_left;
        ensure!(slide_left > 0, "sliding window left");
        let ignored = plant == TextPlant::WindowIgnored;
        let window_left = if ignored { None } else { Some(slide_left) };
        let step = TextStep {
            rows,
            plant,
            window_left,
        };
        if ignored {
            text_layer(&w.text[0], k, &mut flash.text_full, buf, blas, step)?;
        } else {
            text_layer(&w.text[0], k, &mut flash.text_slide, buf, blas, step)?;
        }
        download_f32(k, &buf.h, rows * TEXT_H)
    }

    pub fn backbone(
        &mut self,
        projected: &[f32],
        positions: &Positions,
        codes: &[i32],
        plant_audio: bool,
    ) -> Result<Vec<f32>> {
        self.bound_device()?;
        check_positions(positions)?;
        ensure!(
            projected.len() == positions.text.len() * BB_H,
            "projected length"
        );
        ensure!(
            codes.len() == positions.audio.len() * BOOKS,
            "audio code length"
        );
        ensure!(
            projected.iter().all(|value| value.is_finite()),
            "non-finite projected"
        );
        for &id in codes {
            ensure!((0..CODEBOOK).contains(&id), "audio code {id}");
        }
        if plant_audio {
            ensure!(!positions.audio.is_empty(), "audio plant with no audio");
        }
        let rows = positions.seq;
        let Breeze {
            ref w,
            ref k,
            ref mut flash,
            ref mut buf,
            ref blas,
            ..
        } = *self;
        let audio = if positions.audio.is_empty() {
            Vec::new()
        } else if plant_audio {
            let rotated = rotate_audio_table(k, &w.audio)?;
            sum_audio(k, &rotated, buf, codes, positions.audio.len())?
        } else {
            sum_audio(k, &w.audio, buf, codes, positions.audio.len())?
        };
        let eos = if positions.eos.is_empty() {
            None
        } else {
            let zeros = [0i32; BOOKS];
            Some(sum_audio(k, &w.audio, buf, &zeros, 1)?)
        };
        let hidden = assemble_hidden(projected, positions, &audio, eos.as_deref())?;
        copy_h2d(k, &hidden, &mut buf.h.slice_mut(..hidden.len()))?;
        set_meta(k, &mut buf.meta, rows)?;
        upload_origin(k, &mut buf.start, 0)?;
        let geom = DecGeom {
            rows,
            hidden: BB_H,
            intermediate: BB_I,
            q_heads: BB_HEADS,
            kv_heads: BB_KV,
            dim: BB_DIM,
            eps: BB_EPS,
        };
        for index in 0..w.backbone.len() {
            decoder_layer(
                k,
                buf,
                blas,
                DecPass {
                    layer: &w.backbone[index],
                    table: &k.table_backbone,
                    flash: &mut flash.backbone,
                    kind: RopeKind::Cast,
                    geom,
                    origin: 0,
                    capacity: rows,
                    slot: KvSlot::Shared,
                },
            )?;
        }
        rms(
            k,
            &buf.meta,
            Rms {
                x: &buf.h,
                y: &mut buf.n,
                w: &w.backbone_norm,
                rows,
                width: BB_H,
                mode: RmsMode::Standard,
                eps: BB_EPS,
            },
        )?;
        gemm(
            &k.stream,
            blas,
            Gemm {
                w: &w.lm,
                x: &mut buf.n,
                y: &mut buf.d,
                x_off: 0,
                y_off: 0,
                logical: rows,
                output: BB_VOCAB,
                input: BB_H,
            },
        )?;
        download_f32(k, &buf.d, rows * BB_VOCAB)
    }

    pub fn depth(
        &mut self,
        cond: &[f32],
        codes: &[i32],
        frames: usize,
        plant_head: bool,
    ) -> Result<Vec<f32>> {
        self.bound_device()?;
        ensure!((1..=DP_FRAMES).contains(&frames), "frames {frames}");
        ensure!(cond.len() == frames * BB_H, "cond length");
        ensure!(codes.len() == frames * DP_ROWS, "depth code length");
        ensure!(
            cond.iter().all(|value| value.is_finite()),
            "non-finite cond"
        );
        let mut id_rows = Vec::with_capacity(frames);
        for frame in 0..frames {
            let start = frame * DP_ROWS;
            id_rows.push(depth_ids(&codes[start..start + DP_ROWS])?);
        }
        let Breeze {
            ref w,
            ref k,
            ref mut flash,
            ref mut buf,
            ref blas,
            ..
        } = *self;
        set_meta(k, &mut buf.meta, DP_ROWS)?;
        for frame in 0..frames {
            let cond_row = &cond[frame * BB_H..(frame + 1) * BB_H];
            depth_frame(
                k,
                buf,
                blas,
                DepthIn {
                    layers: &w.depth,
                    norm: &w.depth_norm,
                    projector: &w.projector,
                    embed: &w.audio,
                    table: &k.table_depth,
                    flash: &mut flash.depth,
                    cond: cond_row,
                    ids: &id_rows[frame],
                    frame,
                    slot: KvSlot::Shared,
                },
            )?;
        }
        depth_heads(k, buf, blas, &w.heads, frames, plant_head)
    }

    /// Backbone KV is one BB_MAX cache per layer. Each depth frame uses DP_ROWS and is zeroed after the frame.
    pub fn greedy_decode(
        &mut self,
        projected: &[f32],
        cond: &[f32],
        text_rows: usize,
        fed: &[i32],
        frames: &[i32],
        plant_head: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (audio_steps, depth_frames) = decode_shape(projected, cond, text_rows, fed, frames)?;
        self.bound_device()?;
        let Breeze {
            ref w,
            ref k,
            ref mut flash,
            ref mut buf,
            ref blas,
            ..
        } = *self;
        k.drains.set(0);
        k.h2d.set(0);
        k.d2h.set(0);
        k.d2d.set(0);
        upload_hidden(k, buf, projected)?;
        let mut backbone = Vec::with_capacity((audio_steps + 1) * BB_VOCAB);
        let mut depth = Vec::with_capacity(depth_frames * HEAD_N * (CODEBOOK as usize));
        backbone.extend(backbone_rows(
            k,
            buf,
            blas,
            ScorePass {
                layers: &w.backbone,
                norm: &w.backbone_norm,
                lm: &w.lm,
                table: &k.table_backbone,
                flash: &mut flash.backbone,
                rows: text_rows,
                origin: 0,
                score_row: text_rows - 1,
            },
        )?);
        let mut fed_rows = fed.chunks(BOOKS);
        let mut cond_rows = cond.chunks(BB_H);
        for (frame, codes) in frames.chunks(BOOKS).enumerate() {
            let row = cond_rows.next().context("cond row")?;
            depth.extend(depth_step(
                k,
                buf,
                blas,
                DepthStep {
                    w,
                    flash: &mut flash.depth,
                    cond: row,
                    codes,
                    frame,
                    plant: plant_head,
                },
            )?);
            if frame < audio_steps {
                let fed_row = fed_rows.next().context("fed row")?;
                place_audio_row(k, &w.audio, buf, fed_row)?;
                let origin = i32::try_from(text_rows + frame)?;
                backbone.extend(backbone_rows(
                    k,
                    buf,
                    blas,
                    ScorePass {
                        layers: &w.backbone,
                        norm: &w.backbone_norm,
                        lm: &w.lm,
                        table: &k.table_backbone,
                        flash: &mut flash.backbone,
                        rows: 1,
                        origin,
                        score_row: 0,
                    },
                )?);
            }
        }
        ensure!(
            backbone.len() == (audio_steps + 1) * BB_VOCAB,
            "backbone scores"
        );
        ensure!(
            depth.len() == depth_frames * HEAD_N * (CODEBOOK as usize),
            "depth scores"
        );
        // per_layer: 2 RMS, 4 QKV/O GEMMs, 1 RoPE, 1 attention, 2 residuals, 3 MLP GEMMs, 1 SwiGLU.
        let per_layer = 14;
        let prefill = BB_LAYERS * per_layer + 2;
        let step = prefill + 1;
        let frame_launches = 2 + DP_LAYERS * per_layer + 1 + HEAD_N;
        eprintln!(
            "LAUNCH backbone_prefill={prefill} backbone_step={step} depth_frame={frame_launches}"
        );
        eprintln!(
            "COPY backbone_row_bytes={} backbone_copies={} depth_row_bytes={} depth_copies={} depth_kv_clears={}",
            BB_VOCAB * 4,
            audio_steps + 1,
            CODEBOOK as usize * 4,
            depth_frames * HEAD_N,
            depth_frames
        );
        let (drains, h2d, d2h, d2d) = traffic(text_rows, audio_steps, depth_frames);
        eprintln!(
            "TRAFFIC drains={drains} h2d={h2d} d2h={d2h} d2d={d2d} got_drains={} got_h2d={} got_d2h={} got_d2d={}",
            k.drains.get(),
            k.h2d.get(),
            k.d2h.get(),
            k.d2d.get()
        );
        ensure!(
            k.drains.get() == drains
                && k.h2d.get() == h2d
                && k.d2h.get() == d2h
                && k.d2d.get() == d2d,
            "traffic"
        );
        Ok((backbone, depth))
    }
}

#[cfg(test)]
mod tests {
    use super::bf16;
    use anyhow::{Context, Result, bail, ensure};
    use safetensors::{Dtype, SafeTensors};
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    const CODE_EXPR: &str = "(frame * 17 + book * 13 + 3) % 2047";
    const SHORT_MAX: f64 = 0.5;
    const SHORT_RMSE: f64 = 0.011573881842195988;
    const LONG_MAX: f64 = 10.3125;
    const LONG_RMSE: f64 = 0.05359283462166786;
    const FORCED_BB_MAX: f64 = 0.8125;
    const FORCED_BB_RMSE: f64 = 0.025961503386497498;
    const FORCED_DP_MAX: f64 = 0.875;
    const FORCED_DP_RMSE: f64 = 0.058841850608587265;
    const SLIDE_MAX: f64 = 16.0;
    const SLIDE_RMSE: f64 = 0.11044254899024963;
    const SLIDE_RMSE_TEXT: &str = "0.11044254899024963";
    const SIGNAL_MAX: f64 = 3.0;
    const SIGNAL_RMSE: f64 = 0.08392622321844101;
    const SIGNAL_RMSE_TEXT: &str = "0.08392622321844101";
    const M1_BB_MAX: f64 = 0.25;
    const M1_BB_RMSE: f64 = 0.039339788258075714;
    const M1_BB_RMSE_TEXT: &str = "0.039339788258075714";
    const M1_DP_MAX: f64 = 0.375;
    const M1_DP_RMSE: f64 = 0.050264500081539154;
    const M1_DP_RMSE_TEXT: &str = "0.050264500081539154";
    const M1_TEXT: usize = 7;
    const M1_STEPS: usize = 4;
    const M1_AUDIO: usize = 3;
    const M1_IDS: [i32; M1_STEPS] = [404, 1014, 1248, 1119];
    const RESERVED_START: usize = 2048;
    const RESERVED_END: usize = 2051;

    struct Sequence {
        ids: Vec<i32>,
        text: Vec<usize>,
        audio: Vec<usize>,
        eos: Vec<usize>,
        codes: Vec<i32>,
    }

    struct GateBar {
        tensor: &'static str,
        plant: &'static str,
        source: &'static str,
        tau_max: f64,
        tau_rmse: f64,
        nominal: bool,
        rmse_detect: bool,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum GateStatus {
        Pass,
        Defect,
        ProtocolFail,
    }

    impl GateStatus {
        fn label(self) -> &'static str {
            match self {
                Self::Pass => "PASS",
                Self::Defect => "DEFECT",
                Self::ProtocolFail => "PROTOCOL_FAIL",
            }
        }
    }

    const SHORT_NOMINAL: GateBar = GateBar {
        tensor: "projected_text",
        plant: "nominal",
        source: "short_floors.projected_text",
        tau_max: SHORT_MAX,
        tau_rmse: SHORT_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const SHORT_ROPE: GateBar = GateBar {
        tensor: "projected_text",
        plant: "text_encoder_rope_omitted",
        source: "short_floors.projected_text",
        tau_max: SHORT_MAX,
        tau_rmse: SHORT_RMSE,
        nominal: false,
        rmse_detect: false,
    };
    const LONG_NOMINAL: GateBar = GateBar {
        tensor: "projected_text",
        plant: "nominal",
        source: "long_floors.projected_text",
        tau_max: LONG_MAX,
        tau_rmse: LONG_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const SLIDE_NOMINAL: GateBar = GateBar {
        tensor: "sliding_hidden",
        plant: "nominal",
        source: "slide_floors.sliding_hidden",
        tau_max: SLIDE_MAX,
        tau_rmse: SLIDE_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const SIGNAL_NOMINAL: GateBar = GateBar {
        tensor: "sliding_hidden",
        plant: "nominal",
        source: "slide_signal_floors.sliding_hidden",
        tau_max: SIGNAL_MAX,
        tau_rmse: SIGNAL_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const SIGNAL_IGNORED: GateBar = GateBar {
        tensor: "sliding_hidden",
        plant: "text_encoder_window_ignored",
        source: "slide_signal_floors.sliding_hidden",
        tau_max: SIGNAL_MAX,
        tau_rmse: SIGNAL_RMSE,
        nominal: false,
        rmse_detect: true,
    };
    const FORCED_BB_NOMINAL: GateBar = GateBar {
        tensor: "backbone_logits",
        plant: "nominal",
        source: "forced_floors.backbone_logits",
        tau_max: FORCED_BB_MAX,
        tau_rmse: FORCED_BB_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const FORCED_BB_AUDIO: GateBar = GateBar {
        tensor: "backbone_logits",
        plant: "audio_codebook_index",
        source: "forced_floors.backbone_logits",
        tau_max: FORCED_BB_MAX,
        tau_rmse: FORCED_BB_RMSE,
        nominal: false,
        rmse_detect: false,
    };
    const FORCED_DP_NOMINAL: GateBar = GateBar {
        tensor: "depth_logits",
        plant: "nominal",
        source: "forced_floors.depth_logits",
        tau_max: FORCED_DP_MAX,
        tau_rmse: FORCED_DP_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const FORCED_DP_HEAD: GateBar = GateBar {
        tensor: "depth_logits",
        plant: "depth_head_codebook",
        source: "forced_floors.depth_logits",
        tau_max: FORCED_DP_MAX,
        tau_rmse: FORCED_DP_RMSE,
        nominal: false,
        rmse_detect: false,
    };
    const M1_BB_NOMINAL: GateBar = GateBar {
        tensor: "backbone_scores",
        plant: "nominal",
        source: "floors-forced.backbone_scores",
        tau_max: M1_BB_MAX,
        tau_rmse: M1_BB_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const M1_BB_AUDIO: GateBar = GateBar {
        tensor: "backbone_scores",
        plant: "audio_embedding_row",
        source: "floors-forced.backbone_scores",
        tau_max: M1_BB_MAX,
        tau_rmse: M1_BB_RMSE,
        nominal: false,
        rmse_detect: false,
    };
    const M1_DP_NOMINAL: GateBar = GateBar {
        tensor: "depth_scores",
        plant: "nominal",
        source: "floors-forced.depth_scores",
        tau_max: M1_DP_MAX,
        tau_rmse: M1_DP_RMSE,
        nominal: true,
        rmse_detect: false,
    };
    const M1_DP_HEAD: GateBar = GateBar {
        tensor: "depth_scores",
        plant: "depth_head_codebook",
        source: "floors-forced.depth_scores",
        tau_max: M1_DP_MAX,
        tau_rmse: M1_DP_RMSE,
        nominal: false,
        rmse_detect: false,
    };

    struct RuntimeDir(PathBuf);

    impl RuntimeDir {
        fn create() -> Result<Self> {
            Self::with_prefix("n8-m0")
        }

        fn with_prefix(prefix: &str) -> Result<Self> {
            let path = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
            if path.exists() {
                std::fs::remove_dir_all(&path)?;
            }
            std::fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for RuntimeDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn env_path(name: &str, dir: bool) -> Result<PathBuf> {
        let value = std::env::var(name).with_context(|| format!("missing {name}"))?;
        let path = PathBuf::from(value);
        ensure!(
            if dir { path.is_dir() } else { path.is_file() },
            "{name} is not a {}",
            if dir { "directory" } else { "file" }
        );
        Ok(path)
    }

    fn lock_tau_literals() -> Result<()> {
        ensure!(
            SHORT_MAX.to_bits() == 0x3fe0000000000000,
            "short max_abs bits"
        );
        ensure!(
            SHORT_RMSE.to_bits() == 0x3f87b40c20000000,
            "short rmse bits"
        );
        ensure!(
            LONG_MAX.to_bits() == 0x4024a00000000000,
            "long max_abs bits"
        );
        ensure!(LONG_RMSE.to_bits() == 0x3fab708520000000, "long rmse bits");
        ensure!(
            FORCED_BB_MAX.to_bits() == 0x3fea000000000000,
            "forced backbone max_abs bits"
        );
        ensure!(
            FORCED_BB_RMSE.to_bits() == 0x3f9a95a700000000,
            "forced backbone rmse bits"
        );
        ensure!(
            FORCED_DP_MAX.to_bits() == 0x3fec000000000000,
            "forced depth max_abs bits"
        );
        ensure!(
            FORCED_DP_RMSE.to_bits() == 0x3fae2084e0000000,
            "forced depth rmse bits"
        );
        ensure!(
            SLIDE_MAX.to_bits() == 0x4030000000000000,
            "slide max_abs bits"
        );
        ensure!(
            SLIDE_RMSE.to_bits() == 0x3fbc45f680000000,
            "slide rmse bits"
        );
        ensure!(
            SIGNAL_MAX.to_bits() == 0x4008000000000000,
            "signal max_abs bits"
        );
        ensure!(
            SIGNAL_RMSE.to_bits() == 0x3fb57c3060000000,
            "signal rmse bits"
        );
        Ok(())
    }

    fn check_header(doc: &Value, schema: &str) -> Result<()> {
        ensure!(doc["schema"].as_str() == Some(schema), "schema {schema}");
        ensure!(
            doc["candidate_run"].as_bool() == Some(false),
            "candidate_run"
        );
        ensure!(
            doc["fixture"]["code"].as_str() == Some(CODE_EXPR),
            "fixture code"
        );
        Ok(())
    }

    fn check_shapes(doc: &Value, shapes: &[(&str, &[u64])]) -> Result<()> {
        for &(name, shape) in shapes {
            let got = doc["tensors"][name]["shape"]
                .as_array()
                .with_context(|| format!("{name} shape"))?;
            ensure!(got.len() == shape.len(), "{name} rank");
            for (item, want) in got.iter().zip(shape.iter()) {
                ensure!(item.as_u64() == Some(*want), "{name} shape");
            }
        }
        Ok(())
    }

    fn check_tau(
        text: &str,
        doc: &Value,
        tensor: &str,
        max_abs: f64,
        rmse: f64,
        rmse_text: &str,
    ) -> Result<()> {
        ensure!(text.contains(rmse_text), "{tensor} rmse decimal missing");
        let tau = &doc["tensors"][tensor]["tau"];
        let got_max = tau["max_abs"].as_f64().context("tau max_abs")?;
        let got_rmse = tau["rmse"].as_f64().context("tau rmse")?;
        ensure!(
            got_max.to_bits() == max_abs.to_bits(),
            "{tensor} tau max_abs"
        );
        ensure!(got_rmse.to_bits() == rmse.to_bits(), "{tensor} tau rmse");
        let binds = &doc["tensors"][tensor]["eps_min_binds"];
        ensure!(
            binds["max_abs"].as_bool() == Some(false),
            "{tensor} eps_min_binds"
        );
        ensure!(
            binds["rmse"].as_bool() == Some(false),
            "{tensor} eps_min_binds"
        );
        Ok(())
    }

    fn check_controls(doc: &Value, pairs: &[(&str, &str)]) -> Result<()> {
        let controls = doc["controls"].as_array().context("controls")?;
        for &(tensor, name) in pairs {
            let found = controls.iter().any(|item| {
                item["tensor"].as_str() == Some(tensor)
                    && item["name"].as_str() == Some(name)
                    && item["measured"].as_bool() == Some(false)
            });
            ensure!(found, "control {tensor} {name}");
        }
        Ok(())
    }

    fn floor_doc(name: &str) -> Result<(String, Value)> {
        let text = std::fs::read_to_string(env_path(name, false)?)?;
        let doc = serde_json::from_str(&text)?;
        Ok((text, doc))
    }

    fn assert_frozen_floors() -> Result<()> {
        lock_tau_literals()?;
        let (short, short_doc) = floor_doc("N8_FLOORS")?;
        let (long, long_doc) = floor_doc("N8_FLOORS_LONG")?;
        let (forced, forced_doc) = floor_doc("N8_FLOORS_LONG_FORCED")?;
        check_header(&short_doc, "n8-m0-v1")?;
        check_header(&long_doc, "n8-m0-long-v1")?;
        check_header(&forced_doc, "n8-m0-long-forced-v1")?;
        check_shapes(
            &short_doc,
            &[
                ("projected_text", &[1, 6, 2048]),
                ("backbone_logits", &[1, 8, 2052]),
                ("depth_logits", &[2, 15, 2051]),
            ],
        )?;
        check_shapes(
            &long_doc,
            &[
                ("projected_text", &[1, 652, 2048]),
                ("backbone_logits", &[1, 716, 2052]),
                ("depth_logits", &[64, 15, 2051]),
            ],
        )?;
        check_shapes(
            &forced_doc,
            &[
                ("backbone_logits", &[1, 716, 2052]),
                ("depth_logits", &[64, 15, 2051]),
            ],
        )?;
        check_tau(
            &short,
            &short_doc,
            "projected_text",
            SHORT_MAX,
            SHORT_RMSE,
            "0.011573881842195988",
        )?;
        check_tau(
            &long,
            &long_doc,
            "projected_text",
            LONG_MAX,
            LONG_RMSE,
            "0.05359283462166786",
        )?;
        check_tau(
            &forced,
            &forced_doc,
            "backbone_logits",
            FORCED_BB_MAX,
            FORCED_BB_RMSE,
            "0.025961503386497498",
        )?;
        check_tau(
            &forced,
            &forced_doc,
            "depth_logits",
            FORCED_DP_MAX,
            FORCED_DP_RMSE,
            "0.058841850608587265",
        )?;
        check_controls(
            &short_doc,
            &[
                ("projected_text", "text_encoder_rope_position"),
                ("backbone_logits", "audio_embedding_row"),
                ("depth_logits", "depth_head_codebook"),
            ],
        )?;
        check_controls(
            &long_doc,
            &[
                ("projected_text", "text_encoder_sliding_window"),
                ("backbone_logits", "audio_embedding_row"),
                ("depth_logits", "depth_head_codebook"),
            ],
        )?;
        check_controls(
            &forced_doc,
            &[
                ("backbone_logits", "audio_embedding_row"),
                ("depth_logits", "depth_head_codebook"),
            ],
        )?;
        Ok(())
    }

    fn assert_slide_floors() -> Result<()> {
        ensure!(
            SLIDE_MAX.to_bits() != 0 && SLIDE_RMSE.to_bits() != 0,
            "slide tau unset"
        );
        let (text, doc) = floor_doc("N8_FLOORS_SLIDE")?;
        check_header(&doc, "n8-m0-long-slide-v1")?;
        check_shapes(&doc, &[("sliding_hidden", &[1, 652, 1152])])?;
        check_tau(
            &text,
            &doc,
            "sliding_hidden",
            SLIDE_MAX,
            SLIDE_RMSE,
            SLIDE_RMSE_TEXT,
        )?;
        check_controls(&doc, &[("sliding_hidden", "text_encoder_sliding_window")])?;
        Ok(())
    }

    fn assert_signal_floors() -> Result<()> {
        let (text, doc) = floor_doc("N8_FLOORS_SLIDE_SIGNAL")?;
        ensure!(
            doc["schema"].as_str() == Some("n8-m0-long-slide-signal-v1"),
            "signal schema"
        );
        ensure!(
            doc["candidate_run"].as_bool() == Some(false),
            "signal candidate_run"
        );
        ensure!(doc["fixture"]["far_rows"].as_u64() == Some(16), "far rows");
        ensure!(
            doc["fixture"]["sliding_window"].as_u64() == Some(512),
            "signal window"
        );
        ensure!(doc["fixture"]["layer"].as_u64() == Some(0), "signal layer");
        check_shapes(&doc, &[("sliding_hidden", &[1, 652, 1152])])?;
        check_tau(
            &text,
            &doc,
            "sliding_hidden",
            SIGNAL_MAX,
            SIGNAL_RMSE,
            SIGNAL_RMSE_TEXT,
        )?;
        let name = ["text_encoder_window_", "ignored"].concat();
        check_controls(&doc, &[("sliding_hidden", name.as_str())])?;
        Ok(())
    }

    #[test]
    fn rope_omitted_table_is_identity() -> Result<()> {
        let identity = super::identity_rope_table(2, super::TEXT_DIM)?;
        let dim = super::TEXT_DIM;
        for position in 0..2 {
            let row = &identity[position * dim..(position + 1) * dim];
            for pair in 0..dim / 2 {
                ensure!(row[pair] == bf16::ONE, "identity cos");
                ensure!(row[dim / 2 + pair] == bf16::ZERO, "identity sin");
            }
        }
        let slide = super::rope_table(2, dim, super::text_rope_law(false))?;
        position_zero(&slide, dim)?;
        ensure!(slide[..dim] == identity[..dim], "position 0 sliding table");
        ensure!(
            slide[dim..] != identity[dim..],
            "position 1 sliding table is identity"
        );
        let kernel = include_str!("../../../kernels/rope.cu");
        ensure!(
            !kernel.contains("KEY_CACHE_TABLE"),
            "rope kernel indexes keys by a cache table"
        );
        ensure!(
            kernel
                .matches("qknorm_rope_kv_impl<true, true, true, false, false, true, false, true>(")
                .count()
                == 1,
            "one_plus instantiation"
        );
        let engine = include_str!("engine_breeze.rs");
        let rope_metric = ["text_encoder_rope_", "omitted"].concat();
        let window_metric = ["text_encoder_window_", "ignored"].concat();
        let detect = ["rmse_detect: ", "true"].concat();
        let old_theta = ["planted_rope_", "theta"].concat();
        let old_shift = ["text_slide_", "shift"].concat();
        let old_base = ["TextPlant::", "RopeBase"].concat();
        let old_variant = ["TextPlant::", "SlidingWindow"].concat();
        let old_metric = ["text_encoder_rope_", "base"].concat();
        ensure!(
            engine.matches(&rope_metric).count() == 1,
            "rope omitted metric"
        );
        ensure!(
            engine.matches(&window_metric).count() == 1,
            "window ignored metric"
        );
        ensure!(
            engine.matches(&detect).count() == 1,
            "one rmse detection bar"
        );
        ensure!(!engine.contains(&old_theta), "old rope base function");
        ensure!(!engine.contains(&old_shift), "one-key window flash");
        ensure!(!engine.contains(&old_base), "rope base plant");
        ensure!(!engine.contains(&old_variant), "one-key window plant");
        ensure!(!engine.contains(&old_metric), "rope base metric");
        Ok(())
    }

    #[test]
    fn audio_plant_wraps_every_codebook() {
        let mut seen = [false; super::BOOKS];
        for book in 0..super::BOOKS {
            let source = super::audio_source_book(book);
            assert!(source < super::BOOKS && !seen[source]);
            seen[source] = true;
            assert_eq!(source, (book + 1) % super::BOOKS);
        }
        assert!(seen.iter().all(|hit| *hit));
        assert_eq!(super::audio_source_book(super::BOOKS - 1), 0);
        let kernel = include_str!("../../../kernels/embed.cu");
        assert!(kernel.contains("(size_t)book * (unsigned)vocab"));
        let engine = include_str!("engine_breeze.rs");
        assert!(engine.contains("audio_source_book(book)"));
        let old = ["audio_codes", "[0]"].concat();
        assert!(!engine.contains(&old));
    }

    #[test]
    fn decode_shape_accepts_64_frames_and_rejects_65() -> Result<()> {
        let text_rows = 7;
        let audio_steps = 4;
        let projected = vec![0f32; text_rows * super::BB_H];
        let fed = vec![0i32; audio_steps * super::BOOKS];
        let frames_64 = vec![0i32; super::DP_FRAMES * super::BOOKS];
        let cond_64 = vec![0f32; super::DP_FRAMES * super::BB_H];
        let (steps, depth) =
            super::decode_shape(&projected, &cond_64, text_rows, &fed, &frames_64)?;
        ensure!(
            steps == audio_steps && depth == super::DP_FRAMES,
            "64 frames"
        );
        let frames_65 = vec![0i32; (super::DP_FRAMES + 1) * super::BOOKS];
        let cond_65 = vec![0f32; (super::DP_FRAMES + 1) * super::BB_H];
        ensure!(
            super::decode_shape(&projected, &cond_65, text_rows, &fed, &frames_65).is_err(),
            "65 frames"
        );
        Ok(())
    }

    fn position_zero(table: &[bf16], dim: usize) -> Result<()> {
        ensure!(table.len() >= dim, "rope table");
        for d in 0..dim / 2 {
            ensure!(table[d] == bf16::ONE, "rope cos");
            ensure!(table[dim / 2 + d] == bf16::ZERO, "rope sin");
        }
        Ok(())
    }

    fn audio_code(frame: usize, book: usize) -> Result<i32> {
        let value = frame
            .checked_mul(17)
            .and_then(|n| n.checked_add(book.checked_mul(13)?))
            .and_then(|n| n.checked_add(3))
            .context("code")?;
        i32::try_from(value % 2047).context("code")
    }

    fn synthesize(frames: usize) -> Result<Vec<i32>> {
        let mut codes = Vec::with_capacity(frames * super::BOOKS);
        for frame in 0..frames {
            for book in 0..super::BOOKS {
                codes.push(audio_code(frame, book)?);
            }
        }
        Ok(codes)
    }

    fn i32_tensor(tensors: &SafeTensors<'_>, name: &str) -> Result<(Vec<usize>, Vec<i32>)> {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("Runner FAIL missing {name}"))?;
        ensure!(view.dtype() == Dtype::I32, "Runner FAIL {name} dtype");
        let count = view.shape().iter().product::<usize>();
        ensure!(view.data().len() == count * 4, "Runner FAIL {name} bytes");
        let mut values = Vec::with_capacity(count);
        for chunk in view.data().as_chunks::<4>().0 {
            values.push(i32::from_le_bytes(*chunk));
        }
        Ok((view.shape().to_vec(), values))
    }

    fn maybe_i32(tensors: &SafeTensors<'_>, name: &str) -> Result<Option<(Vec<usize>, Vec<i32>)>> {
        if tensors.names().contains(&name) {
            Ok(Some(i32_tensor(tensors, name)?))
        } else {
            Ok(None)
        }
    }

    fn shape_ok(shape: &[usize], seq: usize) -> bool {
        (shape.len() == 1 && shape[0] == seq)
            || (shape.len() == 2 && shape[0] == 1 && shape[1] == seq)
    }

    fn load_sequence(path: &Path) -> Result<Sequence> {
        let bytes = std::fs::read(path).context("Runner FAIL inputs")?;
        let tensors = SafeTensors::deserialize(&bytes).context("Runner FAIL inputs")?;
        let (id_shape, ids) = i32_tensor(&tensors, "input_ids")?;
        ensure!(
            id_shape.len() == 2 && id_shape[0] == 1,
            "Runner FAIL input_ids shape"
        );
        let seq = id_shape[1];
        ensure!(ids.len() == seq && seq > 0, "Runner FAIL input_ids length");
        if let Some((mask_shape, mask)) = maybe_i32(&tensors, "attention_mask")? {
            ensure!(
                shape_ok(&mask_shape, seq) && mask.len() == seq,
                "Runner FAIL attention_mask"
            );
            ensure!(
                mask.iter().all(|value| *value != 0),
                "Runner FAIL attention mask zero"
            );
        }
        let mut text = Vec::new();
        let mut audio = Vec::new();
        let mut eos = Vec::new();
        if let Some((mask_shape, mask)) = maybe_i32(&tensors, "text_ids_mask")? {
            ensure!(
                shape_ok(&mask_shape, seq) && mask.len() == seq,
                "Runner FAIL text_ids_mask"
            );
            for (index, (&flag, &id)) in mask.iter().zip(&ids).enumerate() {
                if flag != 0 {
                    ensure!(
                        id != super::AUDIO && id != super::AUDIO_EOS,
                        "Runner FAIL text id {id}"
                    );
                    text.push(index);
                } else if id == super::AUDIO {
                    ensure!(index > 0, "Runner FAIL audio at 0");
                    audio.push(index);
                } else if id == super::AUDIO_EOS {
                    eos.push(index);
                } else {
                    bail!("Runner FAIL position {index} id {id}");
                }
            }
        } else {
            for (index, &id) in ids.iter().enumerate() {
                if id == super::AUDIO {
                    ensure!(index > 0, "Runner FAIL audio at 0");
                    audio.push(index);
                } else if id == super::AUDIO_EOS {
                    eos.push(index);
                } else {
                    text.push(index);
                }
            }
        }
        let (len_shape, len_values) = i32_tensor(&tensors, "text_ids_len")?;
        ensure!(
            len_shape.is_empty() || len_shape.len() == 1,
            "Runner FAIL text_ids_len"
        );
        let len_sum: i64 = len_values.iter().map(|value| i64::from(*value)).sum();
        ensure!(
            len_sum == i64::try_from(text.len())?,
            "Runner FAIL text_ids_len sum"
        );
        let codes = if let Some((value_shape, values)) = maybe_i32(&tensors, "input_values")? {
            ensure!(
                value_shape.len() == 3
                    && value_shape[0] == 1
                    && value_shape[1] == audio.len()
                    && value_shape[2] == super::BOOKS,
                "Runner FAIL input_values shape"
            );
            ensure!(
                values.len() == audio.len() * super::BOOKS,
                "Runner FAIL input_values length"
            );
            for frame in 0..audio.len() {
                for book in 0..super::BOOKS {
                    ensure!(
                        values[frame * super::BOOKS + book] == audio_code(frame, book)?,
                        "Runner FAIL input_values code"
                    );
                }
            }
            values
        } else {
            synthesize(audio.len())?
        };
        Ok(Sequence {
            ids,
            text,
            audio,
            eos,
            codes,
        })
    }

    fn expect_layout(seq: &Sequence, text: usize, frames: usize) -> Result<()> {
        ensure!(seq.ids.len() == text + frames, "Runner FAIL seq");
        ensure!(
            seq.text == (0..text).collect::<Vec<_>>(),
            "Runner FAIL text positions"
        );
        ensure!(
            seq.audio == (text..text + frames).collect::<Vec<_>>(),
            "Runner FAIL audio positions"
        );
        ensure!(seq.eos.is_empty(), "Runner FAIL eos");
        ensure!(
            seq.audio.iter().all(|&pos| seq.ids[pos] == super::AUDIO),
            "Runner FAIL audio id"
        );
        ensure!(
            seq.codes.len() == frames * super::BOOKS,
            "Runner FAIL codes"
        );
        Ok(())
    }

    fn text_ids(seq: &Sequence) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(seq.text.len());
        for &pos in &seq.text {
            let id = seq.ids[pos];
            ensure!(id >= 0, "Runner FAIL text id {id}");
            out.push(u32::try_from(id)?);
        }
        Ok(out)
    }

    fn depth_codes(codes: &[i32], frames: usize) -> Result<Vec<i32>> {
        ensure!(
            codes.len() == frames * super::BOOKS,
            "Runner FAIL code length"
        );
        let mut out = vec![0i32; frames * super::DP_ROWS];
        for frame in 0..frames {
            for book in 0..super::HEAD_N {
                out[frame * super::DP_ROWS + book + 1] = codes[frame * super::BOOKS + book];
            }
        }
        Ok(out)
    }

    fn cond_rows(hidden: &[f32], audio: &[usize]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(audio.len() * super::BB_H);
        for &pos in audio {
            ensure!(pos > 0, "Runner FAIL audio at 0");
            let start = (pos - 1) * super::BB_H;
            let end = start + super::BB_H;
            ensure!(end <= hidden.len(), "Runner FAIL cond row");
            out.extend_from_slice(&hidden[start..end]);
        }
        Ok(out)
    }

    fn tensor_f32(path: &Path, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        tensor_values(path, name, shape, false)
    }

    fn tensor_values(path: &Path, name: &str, shape: &[usize], neginf: bool) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).with_context(|| format!("Runner FAIL read {name}"))?;
        let tensors = SafeTensors::deserialize(&bytes).context("Runner FAIL safetensors")?;
        let view = tensors
            .tensor(name)
            .with_context(|| format!("Runner FAIL missing {name}"))?;
        ensure!(view.dtype() == Dtype::F32, "Runner FAIL {name} dtype");
        ensure!(view.shape() == shape, "Runner FAIL {name} shape");
        let count = shape.iter().product::<usize>();
        ensure!(view.data().len() == count * 4, "Runner FAIL {name} bytes");
        let mut values = Vec::with_capacity(count);
        for chunk in view.data().as_chunks::<4>().0 {
            values.push(f32::from_le_bytes(*chunk));
        }
        let valid = if neginf {
            values.iter().all(|value| score_ok(*value))
        } else {
            values.iter().all(|value| value.is_finite())
        };
        ensure!(valid, "Runner FAIL non-finite reference {name}");
        Ok(values)
    }

    fn difference(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
        ensure!(!actual.is_empty(), "Runner FAIL empty tensor");
        ensure!(actual.len() == expected.len(), "Runner FAIL element count");
        ensure!(
            actual.iter().all(|value| value.is_finite()),
            "Runner FAIL non-finite candidate"
        );
        ensure!(
            expected.iter().all(|value| value.is_finite()),
            "Runner FAIL non-finite reference"
        );
        let mut max_abs = 0.0f64;
        let mut squares = 0.0f64;
        for (&got, &want) in actual.iter().zip(expected) {
            let error = (f64::from(got) - f64::from(want)).abs();
            max_abs = max_abs.max(error);
            squares += error * error;
        }
        let rmse = (squares / actual.len() as f64).sqrt();
        ensure!(
            max_abs.is_finite() && rmse.is_finite(),
            "Runner FAIL non-finite metric"
        );
        Ok((max_abs, rmse))
    }

    fn report(bar: &GateBar, max_abs: f64, rmse: f64) -> GateStatus {
        let bar_max = 10.0 * bar.tau_max;
        let bar_rmse = 10.0 * bar.tau_rmse;
        let detect = if bar.nominal {
            "nominal"
        } else if bar.rmse_detect {
            "rmse"
        } else {
            "both"
        };
        let status = if bar.nominal {
            if max_abs <= bar.tau_max && rmse <= bar.tau_rmse {
                GateStatus::Pass
            } else {
                GateStatus::Defect
            }
        } else if bar.rmse_detect {
            if rmse >= bar_rmse {
                GateStatus::Pass
            } else {
                GateStatus::ProtocolFail
            }
        } else if max_abs >= bar_max && rmse >= bar_rmse {
            GateStatus::Pass
        } else {
            GateStatus::ProtocolFail
        };
        println!(
            "METRIC tensor={} plant={} bar_source={} max_abs={max_abs:?} rmse={rmse:?} tau_max_abs={:?} tau_rmse={:?} bar_max_abs={bar_max:?} bar_rmse={bar_rmse:?} detect={detect} status={}",
            bar.tensor,
            bar.plant,
            bar.source,
            bar.tau_max,
            bar.tau_rmse,
            status.label()
        );
        status
    }

    fn arm(actual: &[f32], expected: &[f32], bar: &GateBar) -> Result<GateStatus> {
        let (max_abs, rmse) = difference(actual, expected)?;
        Ok(report(bar, max_abs, rmse))
    }

    fn record(status: GateStatus, defect: &mut bool, protocol: &mut bool) {
        match status {
            GateStatus::Pass => {}
            GateStatus::Defect => *defect = true,
            GateStatus::ProtocolFail => *protocol = true,
        }
    }

    fn finish_gate(defect: bool, protocol: bool) -> Result<()> {
        match (defect, protocol) {
            (false, false) => Ok(()),
            (true, false) => bail!("candidate DEFECT"),
            (false, true) => bail!("protocol FAIL"),
            (true, true) => bail!("candidate DEFECT\nprotocol FAIL"),
        }
    }

    #[test]
    fn nominal_defect_fails_the_candidate() -> Result<()> {
        let row = [1.0f32];
        let zero = [0.0f32];
        let over = GateBar {
            tensor: "probe",
            plant: "nominal",
            source: "probe",
            tau_max: 0.0,
            tau_rmse: 0.0,
            nominal: true,
            rmse_detect: false,
        };
        let mut defect = false;
        let mut protocol = false;
        record(arm(&row, &zero, &over)?, &mut defect, &mut protocol);
        match finish_gate(defect, protocol) {
            Ok(()) => bail!("nominal defect"),
            Err(err) => ensure!(err.to_string() == "candidate DEFECT", "candidate DEFECT"),
        }
        let inside = GateBar {
            tensor: "probe",
            plant: "nominal",
            source: "probe",
            tau_max: 1.0,
            tau_rmse: 1.0,
            nominal: true,
            rmse_detect: false,
        };
        defect = false;
        protocol = false;
        record(arm(&row, &zero, &inside)?, &mut defect, &mut protocol);
        finish_gate(defect, protocol)?;
        let miss = GateBar {
            tensor: "probe",
            plant: "probe",
            source: "probe",
            tau_max: 100.0,
            tau_rmse: 100.0,
            nominal: false,
            rmse_detect: false,
        };
        defect = false;
        protocol = false;
        record(arm(&row, &zero, &miss)?, &mut defect, &mut protocol);
        match finish_gate(defect, protocol) {
            Ok(()) => bail!("control miss"),
            Err(err) => ensure!(err.to_string() == "protocol FAIL", "protocol FAIL"),
        }
        Ok(())
    }

    #[test]
    fn n8_gemm_rows_and_rope_table() -> Result<()> {
        ensure!(super::gemm_rows(6, 1024) == 17, "gemm rows 6 1024");
        ensure!(super::gemm_rows(6, 1152) == 6, "gemm rows 6 1152");
        ensure!(super::gemm_rows(652, 1024) == 652, "gemm rows 652 1024");
        ensure!(super::gemm_rows(16, 1024) == 17, "gemm rows 16 1024");
        ensure!(super::gemm_rows(64, 1024) == 64, "gemm rows 64 1024");
        let slide = super::rope_table(
            2,
            super::TEXT_DIM,
            super::RopeLaw::Plain {
                theta: 10_000.0,
                divisor: 1.0,
            },
        )?;
        let full = super::rope_table(
            2,
            super::TEXT_DIM,
            super::RopeLaw::Plain {
                theta: 1_000_000.0,
                divisor: 8.0,
            },
        )?;
        let backbone = super::rope_table(
            2,
            super::BB_DIM,
            super::RopeLaw::Plain {
                theta: 1_000_000.0,
                divisor: 1.0,
            },
        )?;
        let depth = super::rope_table(2, super::DP_DIM, super::RopeLaw::Llama3)?;
        position_zero(&slide, super::TEXT_DIM)?;
        position_zero(&full, super::TEXT_DIM)?;
        position_zero(&backbone, super::BB_DIM)?;
        position_zero(&depth, super::DP_DIM)?;
        ensure!(slide != full, "sliding and full rope tables");
        ensure!(
            super::super::flash::window_radii(512) == (255, 256),
            "window radii"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease, frozen floors, and external fixtures"]
    fn n8_m0_candidate() -> Result<()> {
        let weights = env_path("N8_WEIGHTS", true)?;
        let short_inputs = env_path("N8_SHORT_INPUTS", false)?;
        let short_reference = env_path("N8_SHORT_REFERENCE", false)?;
        let long_inputs = env_path("N8_LONG_INPUTS", false)?;
        let long_reference = env_path("N8_LONG_REFERENCE", false)?;
        let forced_projected_path = env_path("N8_FORCED_PROJECTED", false)?;
        let forced_hidden_path = env_path("N8_FORCED_HIDDEN", false)?;
        let forced_reference = env_path("N8_FORCED_REFERENCE", false)?;
        let slide_input_path = env_path("N8_SLIDE_INPUT", false)?;
        let slide_reference_path = env_path("N8_SLIDE_REFERENCE", false)?;
        let signal_input_path = env_path("N8_SLIDE_SIGNAL_INPUT", false)?;
        let signal_reference_path = env_path("N8_SLIDE_SIGNAL_REFERENCE", false)?;
        assert_frozen_floors()?;
        assert_slide_floors()?;
        assert_signal_floors()?;
        let short = load_sequence(&short_inputs)?;
        let long = load_sequence(&long_inputs)?;
        expect_layout(&short, 6, 2)?;
        expect_layout(&long, 652, 64)?;
        let short_projected = tensor_f32(&short_reference, "projected_text", &[1, 6, 2048])?;
        let long_projected = tensor_f32(&long_reference, "projected_text", &[1, 652, 2048])?;
        let forced_projected =
            tensor_f32(&forced_projected_path, "projected_text", &[1, 652, 2048])?;
        let forced_hidden = tensor_f32(&forced_hidden_path, "backbone_hidden", &[1, 716, 2048])?;
        let forced_backbone = tensor_f32(&forced_reference, "backbone_logits", &[1, 716, 2052])?;
        let forced_depth = tensor_f32(&forced_reference, "depth_logits", &[64, 15, 2051])?;
        let slide_input = tensor_f32(&slide_input_path, "sliding_input", &[1, 652, 1152])?;
        let slide_hidden = tensor_f32(&slide_reference_path, "sliding_hidden", &[1, 652, 1152])?;
        let signal_input = tensor_f32(&signal_input_path, "sliding_input", &[1, 652, 1152])?;
        let signal_hidden = tensor_f32(&signal_reference_path, "sliding_hidden", &[1, 652, 1152])?;
        let short_text = text_ids(&short)?;
        let long_text = text_ids(&long)?;
        let positions = super::Positions {
            text: long.text.clone(),
            audio: long.audio.clone(),
            eos: long.eos.clone(),
            seq: long.ids.len(),
        };
        let cond = cond_rows(&forced_hidden, &long.audio)?;
        let depth_ids = depth_codes(&long.codes, long.audio.len())?;
        let runtime = RuntimeDir::create()?;
        let mut model = super::Breeze::load(&weights, &runtime.0)?;
        println!("weight_bytes={}", model.weight_bytes);
        let mut defect = false;
        let mut protocol = false;
        record(
            arm(
                &model.text(&short_text, super::TextPlant::Nominal)?,
                &short_projected,
                &SHORT_NOMINAL,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.text(&short_text, super::TextPlant::RopeOmitted)?,
                &short_projected,
                &SHORT_ROPE,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.text(&long_text, super::TextPlant::Nominal)?,
                &long_projected,
                &LONG_NOMINAL,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.sliding_layer(&slide_input, super::TextPlant::Nominal)?,
                &slide_hidden,
                &SLIDE_NOMINAL,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.sliding_layer(&signal_input, super::TextPlant::Nominal)?,
                &signal_hidden,
                &SIGNAL_NOMINAL,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.sliding_layer(&signal_input, super::TextPlant::WindowIgnored)?,
                &signal_hidden,
                &SIGNAL_IGNORED,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.backbone(&forced_projected, &positions, &long.codes, false)?,
                &forced_backbone,
                &FORCED_BB_NOMINAL,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.backbone(&forced_projected, &positions, &long.codes, true)?,
                &forced_backbone,
                &FORCED_BB_AUDIO,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.depth(&cond, &depth_ids, long.audio.len(), false)?,
                &forced_depth,
                &FORCED_DP_NOMINAL,
            )?,
            &mut defect,
            &mut protocol,
        );
        record(
            arm(
                &model.depth(&cond, &depth_ids, long.audio.len(), true)?,
                &forced_depth,
                &FORCED_DP_HEAD,
            )?,
            &mut defect,
            &mut protocol,
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
        finish_gate(defect, protocol)?;
        Ok(())
    }

    fn score_ok(value: f32) -> bool {
        !value.is_nan() && !(value.is_infinite() && value.is_sign_positive())
    }

    fn json_i32(value: &Value) -> Result<i32> {
        let number = value.as_i64().context("Runner FAIL fixture int")?;
        i32::try_from(number).context("Runner FAIL fixture int")
    }

    fn m1_prompt() -> [u32; M1_TEXT] {
        [2, 262_146, 9_259, 699, 506, 20_015, 236_761]
    }

    fn selected<'a>(doc: &'a Value, name: &str) -> Result<&'a Value> {
        let controls = doc["controls"].as_array().context("Runner FAIL controls")?;
        controls
            .iter()
            .find(|item| {
                item["name"].as_str() == Some(name) && item["measured"].as_bool() == Some(true)
            })
            .with_context(|| format!("Runner FAIL control {name}"))
    }

    fn forced_codes(doc: &Value) -> Result<Vec<i32>> {
        let frames = doc["fixture"]["forced_frames"]
            .as_array()
            .context("Runner FAIL frames")?;
        ensure!(frames.len() == M1_STEPS, "Runner FAIL frames");
        let mut codes = Vec::with_capacity(M1_STEPS * super::BOOKS);
        for (step, frame) in frames.iter().enumerate() {
            let row = frame.as_array().context("Runner FAIL frame")?;
            ensure!(row.len() == super::BOOKS, "Runner FAIL frame");
            for item in row {
                let code = json_i32(item)?;
                ensure!(
                    (0..super::CODEBOOK).contains(&code),
                    "Runner FAIL code {code}"
                );
                codes.push(code);
            }
            ensure!(
                codes[step * super::BOOKS] == M1_IDS[step],
                "Runner FAIL code0"
            );
        }
        ensure!(codes[2 * super::BOOKS + 1] == 621, "Runner FAIL plant code");
        Ok(codes)
    }

    fn assert_m1_ids(doc: &Value) -> Result<()> {
        let rows = doc["id_gate"]["backbone"]
            .as_array()
            .context("Runner FAIL backbone ids")?;
        ensure!(rows.len() == M1_STEPS, "Runner FAIL backbone ids");
        for (step, row) in rows.iter().enumerate() {
            ensure!(
                row["step"].as_u64() == Some(step as u64),
                "Runner FAIL step"
            );
            ensure!(
                row["assert_id"].as_bool() == Some(true),
                "Runner FAIL assert"
            );
            for arm in ["4090_a", "4090_b", "a4000", "cpu"] {
                ensure!(
                    row["ids"][arm].as_i64() == Some(i64::from(M1_IDS[step])),
                    "Runner FAIL backbone id"
                );
            }
        }
        ensure!(
            doc["id_gate"]["depth_assert_count"].as_u64() == Some(24),
            "Runner FAIL depth assert count"
        );
        ensure!(
            doc["id_gate"]["depth_positions"].as_u64() == Some(60),
            "Runner FAIL depth positions"
        );
        Ok(())
    }

    fn lock_m1_bits() -> Result<()> {
        ensure!(
            M1_BB_MAX.to_bits() == 0x3fd0_0000_0000_0000,
            "m1 backbone max_abs bits"
        );
        ensure!(
            M1_BB_RMSE.to_bits() == 0x3fa4_2458_4000_0000,
            "m1 backbone rmse bits"
        );
        ensure!(
            M1_DP_MAX.to_bits() == 0x3fd8_0000_0000_0000,
            "m1 depth max_abs bits"
        );
        ensure!(
            M1_DP_RMSE.to_bits() == 0x3fa9_bc44_c000_0000,
            "m1 depth rmse bits"
        );
        Ok(())
    }

    fn assert_m1_fixture(doc: &Value) -> Result<()> {
        ensure!(
            doc["schema"].as_str() == Some("n8-m1-forced-v1"),
            "Runner FAIL schema"
        );
        ensure!(
            doc["candidate_run"].as_bool() == Some(false),
            "Runner FAIL candidate_run"
        );
        ensure!(
            doc["supersedes"].as_str()
                == Some("floors.json n8-m1-greedy-v1 rejected 2026-11-09T04:25:37Z"),
            "Runner FAIL supersedes"
        );
        let prompt = doc["fixture"]["prompt_ids"]
            .as_array()
            .context("Runner FAIL prompt")?;
        ensure!(prompt.len() == M1_TEXT, "Runner FAIL prompt");
        for (got, want) in prompt.iter().zip(m1_prompt()) {
            ensure!(got.as_u64() == Some(u64::from(want)), "Runner FAIL prompt");
        }
        ensure!(
            doc["fixture"]["text_len"].as_u64() == Some(M1_TEXT as u64),
            "Runner FAIL text_len"
        );
        ensure!(
            doc["fixture"]["max_new_tokens"].as_u64() == Some(M1_STEPS as u64),
            "Runner FAIL steps"
        );
        ensure!(
            doc["fixture"]["do_sample"].as_bool() == Some(false),
            "Runner FAIL do_sample"
        );
        ensure!(
            doc["fixture"]["depth_decoder_do_sample"].as_bool() == Some(false),
            "Runner FAIL depth sample"
        );
        ensure!(
            doc["fixture"]["output_audio"].as_bool() == Some(false),
            "Runner FAIL output_audio"
        );
        ensure!(
            doc["fixture"]["cfg_scale"]
                .as_f64()
                .is_some_and(|value| value.to_bits() == 1.0f64.to_bits()),
            "Runner FAIL cfg_scale"
        );
        Ok(())
    }

    fn assert_m1_counts(doc: &Value) -> Result<()> {
        ensure!(
            doc["tensors"]["backbone_scores"]["neginf_count"].as_u64() == Some(12),
            "Runner FAIL neginf"
        );
        ensure!(
            doc["tensors"]["depth_scores"]["neginf_count"].as_u64() == Some(180),
            "Runner FAIL neginf"
        );
        ensure!(
            doc["tensors"]["backbone_scores"]["finite_count"].as_u64() == Some(8196),
            "Runner FAIL finite"
        );
        ensure!(
            doc["tensors"]["depth_scores"]["finite_count"].as_u64() == Some(122_880),
            "Runner FAIL finite"
        );
        Ok(())
    }

    fn assert_m1_controls(doc: &Value) -> Result<()> {
        let audio = selected(doc, "audio_embedding_row")?;
        ensure!(
            audio["tensor"].as_str() == Some("backbone_scores"),
            "Runner FAIL audio control"
        );
        ensure!(
            audio["selection"].as_str() == Some(AUDIO_SCAN),
            "Runner FAIL audio scan"
        );
        ensure!(
            audio["selected"]["frame"].as_u64() == Some(2),
            "Runner FAIL plant frame"
        );
        ensure!(
            audio["selected"]["book"].as_u64() == Some(1),
            "Runner FAIL plant book"
        );
        ensure!(
            audio["selected"]["code"].as_u64() == Some(621),
            "Runner FAIL plant code"
        );
        ensure!(
            audio["selected"]["planted_code"].as_u64() == Some(622),
            "Runner FAIL planted code"
        );
        ensure!(
            audio["selected"]["affected_step"].as_u64() == Some(3),
            "Runner FAIL affected step"
        );
        ensure!(
            audio["selected"]["exceeds_10x"].as_bool() == Some(true),
            "Runner FAIL audio 10x"
        );
        let depth = selected(doc, "depth_head_codebook")?;
        ensure!(
            depth["tensor"].as_str() == Some("depth_scores"),
            "Runner FAIL depth control"
        );
        ensure!(depth["selection"].is_null(), "Runner FAIL depth selection");
        ensure!(
            depth["selected"]["step"].as_u64() == Some(0),
            "Runner FAIL depth step"
        );
        ensure!(
            depth["selected"]["reads_head"].as_u64() == Some(1),
            "Runner FAIL depth head"
        );
        ensure!(
            depth["selected"]["exceeds_10x"].as_bool() == Some(true),
            "Runner FAIL depth 10x"
        );
        Ok(())
    }

    fn assert_m1_floors(text: &str, doc: &Value) -> Result<Vec<i32>> {
        lock_m1_bits()?;
        assert_m1_fixture(doc)?;
        check_shapes(
            doc,
            &[
                (
                    "backbone_scores",
                    &[M1_STEPS as u64, super::BB_VOCAB as u64],
                ),
                (
                    "depth_scores",
                    &[
                        M1_STEPS as u64,
                        super::HEAD_N as u64,
                        super::CODEBOOK as u64,
                    ],
                ),
            ],
        )?;
        check_tau(
            text,
            doc,
            "backbone_scores",
            M1_BB_MAX,
            M1_BB_RMSE,
            M1_BB_RMSE_TEXT,
        )?;
        check_tau(
            text,
            doc,
            "depth_scores",
            M1_DP_MAX,
            M1_DP_RMSE,
            M1_DP_RMSE_TEXT,
        )?;
        assert_m1_counts(doc)?;
        assert_m1_controls(doc)?;
        assert_m1_ids(doc)?;
        ensure!(
            doc["id_gate"]["free_run"]["role"].as_str() == Some("measurement only; not a tau"),
            "Runner FAIL free run"
        );
        ensure!(
            doc["id_gate"]["free_run"]["near_tie"].as_str() == Some(NEAR_TIE),
            "Runner FAIL near tie"
        );
        ensure!(
            doc["id_gate"]["free_run"]["candidate_rule"].as_str() == Some(FREE_RULE),
            "Runner FAIL free-run rule"
        );
        forced_codes(doc)
    }

    fn mask_reserved(scores: &mut [f32], vocab: usize) -> Result<()> {
        ensure!(
            vocab >= RESERVED_END && scores.len().is_multiple_of(vocab),
            "Runner FAIL vocab"
        );
        let rows = scores.len() / vocab;
        for row in 0..rows {
            for class in RESERVED_START..RESERVED_END {
                scores[row * vocab + class] = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn require_mask(scores: &[f32], vocab: usize, neginf: usize) -> Result<()> {
        ensure!(scores.len().is_multiple_of(vocab), "Runner FAIL vocab");
        let mut seen = 0usize;
        for (index, &value) in scores.iter().enumerate() {
            ensure!(score_ok(value), "Runner FAIL non-finite reference");
            let reserved = (RESERVED_START..RESERVED_END).contains(&(index % vocab));
            if reserved {
                ensure!(value.is_infinite(), "Runner FAIL reference mask");
                seen += 1;
            } else {
                ensure!(value.is_finite(), "Runner FAIL reference mask");
            }
        }
        ensure!(seen == neginf, "Runner FAIL neginf count");
        Ok(())
    }

    fn finite_cells(actual: &[f32], expected: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(actual.len() == expected.len(), "Runner FAIL element count");
        let mut got = Vec::new();
        let mut want = Vec::new();
        for (&left, &right) in actual.iter().zip(expected) {
            ensure!(score_ok(left), "Runner FAIL non-finite candidate");
            ensure!(score_ok(right), "Runner FAIL non-finite reference");
            ensure!(
                left.is_infinite() == right.is_infinite(),
                "Runner FAIL mask"
            );
            if left.is_finite() {
                got.push(left);
                want.push(right);
            }
        }
        ensure!(!got.is_empty(), "Runner FAIL empty tensor");
        Ok((got, want))
    }

    fn score_masked(
        actual: &[f32],
        expected: &[f32],
        bar: &GateBar,
        finite: usize,
    ) -> Result<GateStatus> {
        let (got, want) = finite_cells(actual, expected)?;
        ensure!(got.len() == finite, "Runner FAIL finite count");
        arm(&got, &want, bar)
    }

    fn score_row(scores: &[f32], index: usize, vocab: usize) -> Result<&[f32]> {
        let start = index * vocab;
        let end = start + vocab;
        ensure!(end <= scores.len(), "Runner FAIL score row");
        Ok(&scores[start..end])
    }

    fn top2(row: &[f32]) -> Result<(usize, f64)> {
        let mut best_at = None;
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for (index, &value) in row.iter().enumerate() {
            ensure!(score_ok(value), "Runner FAIL non-finite score");
            if !value.is_finite() {
                continue;
            }
            if best_at.is_none() || value > best {
                second = best;
                best = value;
                best_at = Some(index);
            } else if value > second {
                second = value;
            }
        }
        let id = best_at.context("Runner FAIL finite scores")?;
        ensure!(second.is_finite(), "Runner FAIL finite scores");
        Ok((id, f64::from(best) - f64::from(second)))
    }

    fn agreed_id(rows: &[&[f32]], tau: f64) -> Result<Option<i32>> {
        ensure!(rows.len() == 4, "Runner FAIL arms");
        let mut ids = [0i32; 4];
        let mut min_margin = f64::INFINITY;
        for (slot, row) in rows.iter().enumerate() {
            let (id, margin) = top2(row)?;
            ids[slot] = i32::try_from(id)?;
            min_margin = min_margin.min(margin);
        }
        if ids.iter().any(|id| *id != ids[0]) {
            ensure!(min_margin <= tau, "Runner FAIL id disagree");
            return Ok(None);
        }
        if min_margin > tau {
            Ok(Some(ids[0]))
        } else {
            Ok(None)
        }
    }

    struct ArmScores {
        backbone: Vec<f32>,
        depth: Vec<f32>,
    }

    fn load_arm(path: &Path, backbone: &[usize], depth: &[usize]) -> Result<ArmScores> {
        Ok(ArmScores {
            backbone: tensor_values(path, "backbone_scores", backbone, true)?,
            depth: tensor_values(path, "depth_scores", depth, true)?,
        })
    }

    fn backbone_ids(arms: &[ArmScores; 4]) -> Result<[i32; M1_STEPS]> {
        let mut ids = [0i32; M1_STEPS];
        for step in 0..M1_STEPS {
            let rows = [
                score_row(&arms[0].backbone, step, super::BB_VOCAB)?,
                score_row(&arms[1].backbone, step, super::BB_VOCAB)?,
                score_row(&arms[2].backbone, step, super::BB_VOCAB)?,
                score_row(&arms[3].backbone, step, super::BB_VOCAB)?,
            ];
            let id = agreed_id(&rows, M1_BB_MAX)?.context("Runner FAIL backbone margin")?;
            ensure!(id == M1_IDS[step], "Runner FAIL backbone id");
            ids[step] = id;
        }
        Ok(ids)
    }

    fn depth_asserted(arms: &[ArmScores; 4]) -> Result<Vec<(usize, i32)>> {
        let vocab = super::CODEBOOK as usize;
        let mut out = Vec::new();
        for index in 0..M1_STEPS * super::HEAD_N {
            let rows = [
                score_row(&arms[0].depth, index, vocab)?,
                score_row(&arms[1].depth, index, vocab)?,
                score_row(&arms[2].depth, index, vocab)?,
                score_row(&arms[3].depth, index, vocab)?,
            ];
            if let Some(id) = agreed_id(&rows, M1_DP_MAX)? {
                out.push((index, id));
            }
        }
        ensure!(out.len() == 24, "Runner FAIL depth assert count");
        Ok(out)
    }

    fn next_scores(logits: &[f32], seq: usize) -> Result<Vec<f32>> {
        let vocab = super::BB_VOCAB;
        ensure!(logits.len() == seq * vocab, "Runner FAIL logits");
        let start = (M1_TEXT - 1) * vocab;
        let end = seq * vocab;
        ensure!(end - start == M1_STEPS * vocab, "Runner FAIL score rows");
        let mut scores = logits[start..end].to_vec();
        mask_reserved(&mut scores, vocab)?;
        Ok(scores)
    }

    fn backbone_scores(
        model: &mut super::Breeze,
        projected: &[f32],
        codes: &[i32],
    ) -> Result<Vec<f32>> {
        let seq = M1_TEXT + M1_AUDIO;
        let positions = super::Positions {
            text: (0..M1_TEXT).collect(),
            audio: (M1_TEXT..seq).collect(),
            eos: Vec::new(),
            seq,
        };
        ensure!(
            codes.len() == M1_AUDIO * super::BOOKS,
            "Runner FAIL audio codes"
        );
        let logits = model.backbone(projected, &positions, codes, false)?;
        next_scores(&logits, seq)
    }

    fn plant_audio(codes: &[i32]) -> Result<Vec<i32>> {
        let index = 2 * super::BOOKS + 1;
        ensure!(index < codes.len(), "Runner FAIL plant");
        ensure!(codes[index] == 621, "Runner FAIL plant code");
        let mut planted = codes.to_vec();
        planted[index] = 622;
        Ok(planted)
    }

    fn depth_scores(
        model: &mut super::Breeze,
        hidden: &[f32],
        frames: &[i32],
        plant: bool,
    ) -> Result<Vec<f32>> {
        let ids = depth_codes(frames, M1_STEPS)?;
        let mut scores = model.depth(hidden, &ids, M1_STEPS, plant)?;
        mask_reserved(&mut scores, super::CODEBOOK as usize)?;
        Ok(scores)
    }

    fn greedy_scores(
        model: &mut super::Breeze,
        projected: &[f32],
        hidden: &[f32],
        frames: &[i32],
        plant_audio_row: bool,
        plant_head: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let fed_src = &frames[..M1_AUDIO * super::BOOKS];
        let fed = if plant_audio_row {
            plant_audio(fed_src)?
        } else {
            fed_src.to_vec()
        };
        let (mut backbone, mut depth) =
            model.greedy_decode(projected, hidden, M1_TEXT, &fed, frames, plant_head)?;
        mask_reserved(&mut backbone, super::BB_VOCAB)?;
        mask_reserved(&mut depth, super::CODEBOOK as usize)?;
        Ok((backbone, depth))
    }

    fn note_equiv(name: &str, decode: &[f32], prefill: &[f32]) -> Result<()> {
        let (got, want) = finite_cells(decode, prefill)?;
        let (max_abs, rmse) = difference(&got, &want)?;
        println!(
            "EQUIV tensor={name} cells={} max_abs={max_abs} rmse={rmse}",
            got.len()
        );
        Ok(())
    }

    fn note_id(
        tensor: &str,
        place: &str,
        row: &[f32],
        expected: i32,
        defect: &mut bool,
    ) -> Result<()> {
        let (id, margin) = top2(row)?;
        let got = i32::try_from(id)?;
        let status = if got == expected {
            GateStatus::Pass
        } else {
            *defect = true;
            GateStatus::Defect
        };
        println!(
            "ID tensor={tensor} {place} id={got} margin={margin:?} expected={expected} status={}",
            status.label()
        );
        Ok(())
    }

    const AUDIO_SCAN: &str = "frames 1, 2, 0; within a frame books 0..15. First plant strictly above 10x on both metrics whose affected-step argmax changes. If none changes the argmax, the first plant strictly above 10x on both metrics.";
    const NEAR_TIE: &str = "backbone step 2, 4090 margin 0, A4000 id 508, other arms id 1248";
    const FREE_RULE: &str =
        "a free-run id comparison applies only where the reference margin exceeds 2x tau";

    #[test]
    #[ignore = "requires GPU lease, frozen floors, and external fixtures"]
    fn n8_m1_candidate() -> Result<()> {
        ensure!(
            super::traffic(M1_TEXT, M1_AUDIO, M1_STEPS) == (36, 45_792, 262_536, 758_480),
            "m1 score traffic"
        );
        ensure!(
            super::traffic(203, 63, 64) == (576, 1_106_368, 4_200_576, 12_197_120),
            "production score traffic"
        );
        let weights = env_path("N8_WEIGHTS", true)?;
        let floors = env_path("N8_M1_FLOORS", false)?;
        let cpu_path = env_path("N8_M1_CPU", false)?;
        let repeat_a = env_path("N8_M1_4090_A", false)?;
        let repeat_b = env_path("N8_M1_4090_B", false)?;
        let a4000 = env_path("N8_M1_A4000", false)?;
        let text = std::fs::read_to_string(&floors).context("Runner FAIL floors")?;
        let doc: Value = serde_json::from_str(&text).context("Runner FAIL floors")?;
        let frames = assert_m1_floors(&text, &doc)?;
        let backbone_shape = [M1_STEPS, super::BB_VOCAB];
        let depth_shape = [M1_STEPS, super::HEAD_N, super::CODEBOOK as usize];
        let cpu_backbone = tensor_values(&cpu_path, "backbone_scores", &backbone_shape, true)?;
        let cpu_depth = tensor_values(&cpu_path, "depth_scores", &depth_shape, true)?;
        let hidden = tensor_f32(&cpu_path, "backbone_hidden", &[M1_STEPS, super::BB_H])?;
        let depth_vocab = super::CODEBOOK as usize;
        require_mask(&cpu_backbone, super::BB_VOCAB, 12)?;
        require_mask(&cpu_depth, depth_vocab, 180)?;
        let arms = [
            load_arm(&repeat_a, &backbone_shape, &depth_shape)?,
            load_arm(&repeat_b, &backbone_shape, &depth_shape)?,
            load_arm(&a4000, &backbone_shape, &depth_shape)?,
            ArmScores {
                backbone: cpu_backbone.clone(),
                depth: cpu_depth.clone(),
            },
        ];
        for arm in &arms {
            require_mask(&arm.backbone, super::BB_VOCAB, 12)?;
            require_mask(&arm.depth, depth_vocab, 180)?;
        }
        let backbone_expected = backbone_ids(&arms)?;
        let depth_expected = depth_asserted(&arms)?;
        let prompt = m1_prompt();
        let runtime = RuntimeDir::with_prefix("n8-m1")?;
        let mut model = super::Breeze::load(&weights, &runtime.0)?;
        println!("weight_bytes={}", model.weight_bytes);
        let projected = model.text(&prompt, super::TextPlant::Nominal)?;
        let fed = &frames[..M1_AUDIO * super::BOOKS];
        let (nominal_backbone, nominal_depth) =
            greedy_scores(&mut model, &projected, &hidden, &frames, false, false)?;
        let prefill_backbone = backbone_scores(&mut model, &projected, fed)?;
        note_equiv("backbone_scores", &nominal_backbone, &prefill_backbone)?;
        let prefill_depth = depth_scores(&mut model, &hidden, &frames, false)?;
        note_equiv("depth_scores", &nominal_depth, &prefill_depth)?;
        let (audio_backbone, _) =
            greedy_scores(&mut model, &projected, &hidden, &frames, true, false)?;
        let (_, head_depth) = greedy_scores(&mut model, &projected, &hidden, &frames, false, true)?;
        let mut defect = false;
        let mut protocol = false;
        record(
            score_masked(&nominal_backbone, &cpu_backbone, &M1_BB_NOMINAL, 8196)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_masked(&audio_backbone, &cpu_backbone, &M1_BB_AUDIO, 8196)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_masked(&nominal_depth, &cpu_depth, &M1_DP_NOMINAL, 122_880)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_masked(&head_depth, &cpu_depth, &M1_DP_HEAD, 122_880)?,
            &mut defect,
            &mut protocol,
        );
        for (step, expected) in backbone_expected.iter().enumerate() {
            let row = score_row(&nominal_backbone, step, super::BB_VOCAB)?;
            note_id(
                "backbone_scores",
                &format!("step={step}"),
                row,
                *expected,
                &mut defect,
            )?;
        }
        for (index, id) in depth_expected {
            let frame = index / super::HEAD_N;
            let step = index % super::HEAD_N;
            let row = score_row(&nominal_depth, index, depth_vocab)?;
            note_id(
                "depth_scores",
                &format!("frame={frame} step={step}"),
                row,
                id,
                &mut defect,
            )?;
        }
        println!("FREE_RUN role=measurement near_tie={NEAR_TIE}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        finish_gate(defect, protocol)
    }
}
