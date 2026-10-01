//! GPU decode for dense Qwen3.8-27B: the edge0 closed loop minus MoE.
//! One sync per token (the final token id read). All per-token-varying
//! values live in device buffers (position counter, next_token) so the
//! step is graph-capturable later. Multiple ordinals partition the layer
//! stack across devices; hidden and x1 hop between devices through host
//! staging (one blocking copy per hop). A device whose range exceeds free
//! memory streams its int4 projections through a two-slot ring on a
//! prefetch stream instead.

use crate::config::{LayerKind, Qwen35Config, TEXT_PREFIX, TextConfig};
use crate::gemv16::{Gemv16Kernels, Seg16};
use crate::prefill::Mode;
use crate::weights::{HostProj16, HostProjection, QuantFormat, Qwen35Weights};
use crate::wide::{DownBuffers, GroupPair, WideGeom};
use anyhow::{Context, Result, ensure};
use cudarc::driver::safe::{
    CudaEvent, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, PushKernelArg,
};
use ff_core::residency::{Placement, place, split_layers_by_bytes};
use ff_edge0::gpu::{
    AttnGeom, GdnUpload, GpuContext, GpuGdn, GpuQuant, GroupSeg, KvCache, MropeGeom, QkOutputs,
    QkvNorm, ScoreBuffers,
};
use ff_edge0::int4::GROUP_SIZE;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

type CudaSliceF = CudaSlice<f32>;
type CudaSliceI = CudaSlice<i32>;

/// On-device bytes for the projections, KV cache, and scratch one layer
/// range needs.
pub fn layer_device_bytes(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer: usize,
) -> Result<u64> {
    let text = &config.text_config;
    let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
    let mut total = weights.tensor_device_bytes(&format!("{prefix}.input_layernorm"))?
        + weights.tensor_device_bytes(&format!("{prefix}.post_attention_layernorm"))?;
    total += streamed_layer_bytes(weights, config, layer)?;
    match text.layer_kind(layer) {
        LayerKind::LinearAttention => {
            for p in ["conv1d", "A_log", "dt_bias", "norm"] {
                total += weights.tensor_device_bytes(&format!("{prefix}.linear_attn.{p}"))?;
            }
            // GpuGdn::upload's conv, recurrent, and output buffers.
            let conv_dim = text.conv_dim();
            let buffers = conv_dim * (text.linear_conv_kernel_dim - 1)
                + text.linear_num_value_heads
                    * text.linear_key_head_dim
                    * text.linear_value_head_dim
                + conv_dim
                + text.linear_num_value_heads * text.linear_value_head_dim;
            total += (buffers * std::mem::size_of::<f32>()) as u64;
        }
        LayerKind::FullAttention => {
            for p in ["q_norm", "k_norm"] {
                total += weights.tensor_device_bytes(&format!("{prefix}.self_attn.{p}"))?;
            }
        }
    }
    Ok(total)
}

/// On-device bytes for the device-0 extras: embeddings, lm_head, final
/// norm, and the token/position counters. 16-bit checkpoints gather
/// embeddings on the host, so the table costs no device memory.
pub fn static_device_bytes(weights: &Qwen35Weights, _config: &Qwen35Config) -> Result<u64> {
    let counters = 6 * std::mem::size_of::<i32>() as u64;
    let embed = if weights.format().is_16bit() {
        0
    } else {
        weights.tensor_device_bytes(&format!("{TEXT_PREFIX}.embed_tokens"))?
    };
    Ok(embed
        + weights.tensor_device_bytes("lm_head")?
        + weights.tensor_device_bytes(&format!("{TEXT_PREFIX}.norm"))?
        + counters)
}

fn stack_scratch_bytes(text: &TextConfig) -> u64 {
    let attn = text.num_attention_heads * text.head_dim;
    let words = 3 * attn + 2 * text.intermediate_size + 4 * text.hidden_size + 2 * text.hidden_size;
    (words * std::mem::size_of::<f32>()) as u64
}

/// On-device bytes one layer range needs: per-stack scratch, the layers'
/// projections, and the KV planes of its full-attention layers.
pub fn range_device_bytes(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layers: Range<usize>,
    max_ctx: usize,
) -> Result<u64> {
    let text = &config.text_config;
    let mut required = stack_scratch_bytes(text)
        + prefill_bytes(text, Mode::Gemv, PREFILL_BLOCK, weights.format())?;
    for layer in layers.clone() {
        required += layer_device_bytes(weights, config, layer)?;
    }
    let kv_pair = 2 * max_ctx * (text.num_key_value_heads * text.head_dim) * size_of::<f32>();
    let full = layers
        .filter(|&layer| text.layer_kind(layer) == LayerKind::FullAttention)
        .count();
    Ok(required + (kv_pair * full) as u64)
}

/// The byte-balanced contiguous layer ranges `parts` devices would host.
pub fn partition_layers(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    parts: usize,
) -> Result<Vec<(usize, usize)>> {
    anyhow::ensure!(parts > 0, "at least one device required");
    anyhow::ensure!(
        parts <= config.text_config.num_hidden_layers,
        "{parts} devices for {} layers",
        config.text_config.num_hidden_layers
    );
    let layer_bytes: Vec<u64> = (0..config.text_config.num_hidden_layers)
        .map(|layer| layer_device_bytes(weights, config, layer))
        .collect::<Result<_>>()?;
    Ok(split_layers_by_bytes(&layer_bytes, parts))
}

/// Env var that drops the resident prefix and streams every layer.
pub const FORCE_STREAM_ENV: &str = "QWEN35_FORCE_STREAM";

/// The env override the CLI and examples pass into the constructors.
pub fn force_stream_requested() -> bool {
    std::env::var_os(FORCE_STREAM_ENV).is_some()
}

/// Env var that replaces every device's free-memory reading with the
/// given MiB count inside the residency plan.
pub const FREE_OVERRIDE_MIB_ENV: &str = "QWEN35_FREE_OVERRIDE_MIB";

pub(crate) fn graphs_enabled() -> Result<bool> {
    let value = ff_core::probe::env_value::<u8>("QWEN35_GRAPH")?.unwrap_or(1);
    ensure!(
        value <= 1,
        "QWEN35_GRAPH must be 0 or 1; supply a valid value and rerun ff"
    );
    Ok(value == 1)
}

fn free_override_mib() -> Result<Option<u64>> {
    ff_core::probe::env_value::<u64>(FREE_OVERRIDE_MIB_ENV)?
        .map(|mib| {
            mib.checked_mul(1024 * 1024).context(
                "QWEN35_FREE_OVERRIDE_MIB overflows bytes; supply a valid value and rerun ff",
            )
        })
        .transpose()
}

fn override_free(free: &[u64], override_bytes: Option<u64>) -> Vec<u64> {
    match override_bytes {
        Some(bytes) => vec![bytes; free.len()],
        None => free.to_vec(),
    }
}

/// Tokens per batched-prefill block.
const PREFILL_BLOCK: usize = 32;

fn prefill_bytes(text: &TextConfig, mode: Mode, block: usize, format: QuantFormat) -> Result<u64> {
    let geom = crate::prefill::BatchGeom::of(text, block);
    let workspace = crate::prefill::workspace_bytes(mode, &geom, format)?;
    // The decode-side GDN calibration allocates one op per device at the
    // largest feasible chunk; the prefill charge is that workspace.
    let gdn = crate::prefill::GdnPrefill::bytes(&geom, 64)?;
    Ok(u64::try_from(
        workspace
            .checked_add(gdn)
            .context("prefill bytes overflow")?,
    )?)
}

/// The int4 projections the streaming path re-uploads per layer, as
/// layer-relative suffixes.
fn streaming_names(kind: LayerKind) -> &'static [&'static str] {
    match kind {
        LayerKind::LinearAttention => &[
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        LayerKind::FullAttention => &[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    }
}

/// On-device bytes of one layer's streaming projections, y buffers included.
fn streamed_layer_bytes(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    layer: usize,
) -> Result<u64> {
    let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
    let mut total = 0u64;
    for suffix in streaming_names(config.text_config.layer_kind(layer)) {
        let name = format!("{prefix}.{suffix}");
        let (out_dim, _) = weights.projection_shape(&name)?;
        total += weights.tensor_device_bytes(&name)? + 4 * out_dim as u64;
    }
    Ok(total)
}

/// One streaming slot: every layer-kind projection name's buffers, sized
/// to the name's shape (a slot serves any layer of either kind).
fn slot_union_bytes(weights: &Qwen35Weights, config: &Qwen35Config) -> Result<u64> {
    let text = &config.text_config;
    let mut total = 0u64;
    let mut seen: Vec<&str> = Vec::new();
    for layer in 0..text.num_hidden_layers {
        for &suffix in streaming_names(text.layer_kind(layer)) {
            if seen.contains(&suffix) {
                continue;
            }
            seen.push(suffix);
            let name = format!("{TEXT_PREFIX}.layers.{layer}.{suffix}");
            let (out_dim, _) = weights.projection_shape(&name)?;
            total += weights.tensor_device_bytes(&name)? + 4 * out_dim as u64;
        }
    }
    Ok(total)
}

pub use ff_core::residency::Residency;

/// One device's residency decision with the bytes behind it.
#[derive(Clone, Debug)]
pub struct DevicePlan {
    /// Per-engine (mode, admitted T): each engine's largest block whose
    /// workspace fits. The probe picks one after the weights are up.
    pub prefill_candidates: Vec<(Mode, usize)>,
    pub residency: Residency,
    /// Layers below this layer number (global numbering) stay resident;
    /// the range's suffix above it streams through the slot ring.
    pub resident_through: usize,
    pub resident_bytes: u64,
    /// The chosen plan's footprint (full residency when Resident).
    pub streaming_bytes: u64,
    pub free: u64,
}

/// Resident weight sources the prefill probe times.
#[derive(Clone, Copy)]
enum ProbeSrc<'a> {
    Int4(&'a GpuQuant, &'a GpuQuant),
    P16(&'a GpuProj16, &'a GpuProj16),
    None,
}

struct ProbeArgs<'a> {
    candidates: &'a [(Mode, usize)],
    streaming: bool,
    streamed_bytes_per_layer: u64,
    h2d_gbps: Option<f64>,
    src: ProbeSrc<'a>,
    format: QuantFormat,
    text: &'a crate::config::TextConfig,
    source: &'a str,
    override_env: Option<&'a str>,
}

/// Time one gate/up projection per candidate at its own admitted T and
/// pick the fastest by non-overlapping per-token ranges; a tie keeps the
/// bitwise-exact path (packed for int4, gemv for 16-bit). Runs once at
/// setup on a resident layer's weights; `h2d_gbps` is the measured slot
/// upload bandwidth when the plan streams. Returns the choice and the
/// provenance line.
fn probe_prefill_choice(
    ctx: &GpuContext,
    ordinal: usize,
    args: ProbeArgs<'_>,
) -> Result<(Mode, usize, String)> {
    let ProbeArgs {
        candidates,
        streaming,
        streamed_bytes_per_layer,
        h2d_gbps,
        src,
        format,
        text,
        source,
        override_env,
    } = args;
    let started = std::time::Instant::now();
    ensure!(
        !candidates.is_empty(),
        "prefill probe cuda:{ordinal}: no admitted candidates"
    );
    if matches!(src, ProbeSrc::None) {
        // Fully-streamed plan: no resident weights to probe; pick the
        // smallest-workspace candidate.
        let mut best = candidates[0];
        for c in candidates.iter().skip(1) {
            let a = prefill_bytes(text, c.0, c.1, format)?;
            let b = prefill_bytes(text, best.0, best.1, format)?;
            if a < b {
                best = *c;
            }
        }
        let line = format!(
            "prefill probe: mode {} block {} via fully-streamed plan [{}] UNPROBED — no resident weights to time; derived smallest workspace",
            best.0.as_str(),
            best.1,
            override_env
                .map(|v| format!("QWEN35_PREFILL={v} override"))
                .unwrap_or_else(|| "derived".to_string()),
        );
        return Ok((best.0, best.1, line));
    }
    let mut per_token: Vec<(Mode, usize, f64, f64, f64)> = Vec::new();
    for (mode, t) in candidates {
        let range;
        match (mode, src) {
            (Mode::Gemv, ProbeSrc::Int4(gate, up)) => {
                let wide = crate::wide::WideKernels::load(ctx)?;
                wide.capture_body(Some(0))?;
                let x = ctx.stream.alloc_zeros::<f32>(*t * gate.in_dim)?;
                let mut input = ctx.stream.alloc_zeros::<f32>(gate.in_dim)?;
                let mut y = ctx.stream.alloc_zeros::<f32>(*t * gate.out_dim)?;
                let mut y2 = ctx.stream.alloc_zeros::<f32>(*t * up.out_dim)?;
                range = ff_core::probe::probe(
                    1,
                    |_| Ok(()),
                    |_| {
                        for (p, out) in [(gate, &mut y), (up, &mut y2)] {
                            let segs = [
                                p.group_seg(),
                                p.empty_seg_like(),
                                p.empty_seg_like(),
                                p.empty_seg_like(),
                            ];
                            gemv_rows(
                                ctx,
                                &x,
                                out,
                                &mut input,
                                p.y_ref(),
                                (*t, p.out_dim, p.in_dim),
                                |input| {
                                    wide.group(
                                        ctx,
                                        &segs,
                                        [p.y_ref(); 4],
                                        &GroupPair {
                                            x: input,
                                            xb: input,
                                        },
                                        p.in_dim,
                                        1,
                                    )
                                },
                            )?;
                        }
                        Ok(())
                    },
                    || ctx.stream.synchronize().context("prefill probe fence"),
                    |_, d| Ok(d.as_secs_f64() * 1000.0),
                )
                .with_context(|| format!("prefill probe cuda:{ordinal} {}@T{t}", mode.as_str()))?
                    [0];
            }
            (_, ProbeSrc::Int4(gate, up)) => {
                let mut batch = crate::prefill::Batch::new(
                    ctx,
                    ordinal,
                    *mode,
                    &crate::prefill::BatchGeom::of(text, *t),
                    format,
                )?;
                let x = ctx.stream.alloc_zeros::<f32>(*t * gate.in_dim)?;
                let mut y = ctx.stream.alloc_zeros::<f32>(*t * gate.out_dim)?;
                let mut y2 = ctx.stream.alloc_zeros::<f32>(*t * up.out_dim)?;
                range = ff_core::probe::probe(
                    1,
                    |_| Ok(()),
                    |_| {
                        let (gp, gs, gb) = gate.tensors();
                        let gm = crate::prefill::Matrix {
                            packed: gp,
                            scales: gs,
                            biases: gb,
                            rows: gate.out_dim,
                            cols: gate.in_dim,
                        };
                        let (upp, us, ub) = up.tensors();
                        let um = crate::prefill::Matrix {
                            packed: upp,
                            scales: us,
                            biases: ub,
                            rows: up.out_dim,
                            cols: up.in_dim,
                        };
                        batch.projector.project(gm, &x, &mut y, *t, 1)?;
                        batch.projector.project(um, &x, &mut y2, *t, 1)?;
                        Ok(())
                    },
                    || ctx.stream.synchronize().context("prefill probe fence"),
                    |_, d| Ok(d.as_secs_f64() * 1000.0),
                )
                .with_context(|| format!("prefill probe cuda:{ordinal} {}@T{t}", mode.as_str()))?
                    [0];
            }
            (_, ProbeSrc::P16(gate, up)) => {
                let x = ctx.stream.alloc_zeros::<f32>(*t * gate.in_dim)?;
                let mut y = ctx.stream.alloc_zeros::<f32>(*t * gate.out_dim)?;
                let mut y2 = ctx.stream.alloc_zeros::<f32>(*t * up.out_dim)?;
                if *mode == Mode::Gemv {
                    let g16 = crate::gemv16::Gemv16Kernels::load(ctx)?;
                    let mut input = ctx.stream.alloc_zeros::<f32>(gate.in_dim)?;
                    range = ff_core::probe::probe(
                        1,
                        |_| Ok(()),
                        |_| {
                            for (p, out) in [(gate, &mut y), (up, &mut y2)] {
                                let seg = p.seg();
                                let empty = seg.empty_like();
                                gemv_rows(
                                    ctx,
                                    &x,
                                    out,
                                    &mut input,
                                    &p.y,
                                    (*t, p.out_dim, p.in_dim),
                                    |input| {
                                        g16.group(
                                            ctx,
                                            format,
                                            [&seg, &empty, &empty, &empty],
                                            input,
                                            p.in_dim,
                                        )
                                    },
                                )?;
                            }
                            Ok(())
                        },
                        || ctx.stream.synchronize().context("prefill probe fence"),
                        |_, d| Ok(d.as_secs_f64() * 1000.0),
                    )
                    .with_context(|| {
                        format!("prefill probe cuda:{ordinal} {}@T{t}", mode.as_str())
                    })?[0];
                } else {
                    let mut batch = crate::prefill::Batch::new(
                        ctx,
                        ordinal,
                        *mode,
                        &crate::prefill::BatchGeom::of(text, *t),
                        format,
                    )?;
                    range = ff_core::probe::probe(
                        1,
                        |_| Ok(()),
                        |_| {
                            batch.projector.project16(
                                &gate.w,
                                &x,
                                &mut y,
                                gate.out_dim,
                                gate.in_dim,
                                *t,
                            )?;
                            batch
                                .projector
                                .project16(&up.w, &x, &mut y2, up.out_dim, up.in_dim, *t)?;
                            Ok(())
                        },
                        || ctx.stream.synchronize().context("prefill probe fence"),
                        |_, d| Ok(d.as_secs_f64() * 1000.0),
                    )
                    .with_context(|| {
                        format!("prefill probe cuda:{ordinal} {}@T{t}", mode.as_str())
                    })?[0];
                }
            }
            _ => anyhow::bail!("prefill probe: no weight source for {mode:?}"),
        }
        per_token.push((
            *mode,
            *t,
            range.min / *t as f64,
            range.median / *t as f64,
            range.max / *t as f64,
        ));
    }
    if streaming && let Some(bw) = h2d_gbps {
        for (_, t, mn, md, mx) in per_token.iter_mut() {
            let term = streamed_bytes_per_layer as f64 / (*t as f64 * bw * 1e9);
            *mn += term;
            *md += term;
            *mx += term;
        }
    }
    per_token.sort_by(|a, b| a.3.total_cmp(&b.3));
    let exact = if format.is_16bit() {
        Mode::Gemv
    } else {
        Mode::Packed
    };
    let preferred = per_token.iter().position(|(m, _, _, _, _)| *m == exact);
    let ranges: Vec<_> = per_token
        .iter()
        .map(|(_, _, min, median, max)| ff_core::probe::ProbeRange {
            min: *min,
            median: *median,
            max: *max,
        })
        .collect();
    let (selected, separated) = ff_core::probe::probe_choice(&ranges, preferred)?;
    let (mode, block, _, _, _) = per_token[selected];
    let reason = if per_token.len() == 1 {
        "single candidate"
    } else if separated {
        "non-overlapping fastest range"
    } else if preferred.is_some() {
        "ranges overlap; bitwise-exact path"
    } else {
        "ranges overlap; exact path not a candidate"
    };
    let mut line = format!(
        "prefill probe: mode {} block {} via {} [{}]; {}",
        mode.as_str(),
        block,
        source,
        override_env
            .map(|v| format!("QWEN35_PREFILL={v} override"))
            .unwrap_or_else(|| "derived".to_string()),
        reason,
    );
    for (m, t, mn, md, mx) in &per_token {
        line.push_str(&format!(
            " | {}@T{} min {:.4} med {:.4} max {:.4} ms/tok",
            m.as_str(),
            t,
            mn,
            md,
            mx
        ));
    }
    line.push_str(&format!(
        " | probe {} ms",
        started.elapsed().as_secs_f64() * 1000.0
    ));
    Ok((mode, block, line))
}

/// Residency plan per device; `free` carries one entry per range. Each
/// device keeps a resident prefix of its range and streams the suffix:
/// hybrid(k) = range_device_bytes − Σ streamed_layer_bytes +
/// Σ_{l<k} streamed_layer_bytes(l) + 2×slot_union (+ device-0 statics in
/// every plan). Resident is the k=end case without the slot ring.
/// QWEN35_FREE_OVERRIDE_MIB replaces the free readings; QWEN35_FORCE_STREAM
/// drops the prefix entirely.
/// The stream-ordered pool's allocation granularity, queried once per
/// process.
fn pool_granularity(ordinal: usize) -> Result<usize> {
    use cudarc::driver::sys::{
        CUmemAllocationGranularity_flags, CUmemAllocationHandleType, CUmemAllocationProp,
        CUmemAllocationType, CUmemLocation, CUmemLocationType, cuMemGetAllocationGranularity,
    };
    fn query(ordinal: usize) -> Result<usize> {
        let location = CUmemLocation {
            type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            __bindgen_anon_1: cudarc::driver::sys::CUmemLocation_st__bindgen_ty_1 {
                id: ordinal as std::ffi::c_int,
            },
        };
        let prop = CUmemAllocationProp {
            type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location,
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: unsafe { std::mem::zeroed() },
        };
        let mut granularity = 0usize;
        unsafe {
            cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .result()?;
        }
        Ok(granularity)
    }
    query(ordinal)
}

fn round_up_to(value: u64, granularity: usize) -> u64 {
    value.div_ceil(granularity as u64) * granularity as u64
}

/// The stream-ordered pool's allocation granularity for one device.
pub fn pool_granularity_on(ordinal: usize) -> Result<usize> {
    pool_granularity(ordinal)
}

/// The scalar inputs of one residency plan: the ranges with their KV
/// context, the per-device free readings, the ring depth, and the stream
/// policy. Weights and config are passed separately.
pub struct PlanParams<'a> {
    pub ranges: &'a [(usize, usize)],
    pub max_ctx: usize,
    pub free: &'a [u64],
    pub ring_slots: usize,
    pub force_stream: bool,
    /// The stream-ordered pool's allocation granularity, queried once per
    /// process.
    pub granularity: usize,
}

pub fn plan_residency(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    params: PlanParams<'_>,
) -> Result<Vec<DevicePlan>> {
    let PlanParams {
        ranges,
        max_ctx,
        free,
        ring_slots,
        force_stream,
        granularity,
    } = params;
    ensure!(
        ranges.len() == free.len(),
        "{} ranges for {} free-memory values",
        ranges.len(),
        free.len()
    );
    let override_env = ff_core::probe::env_value::<String>("QWEN35_PREFILL")?;
    let is16 = weights.format().is_16bit();
    let requested = match override_env.as_deref() {
        Some("gemv") => Some(Mode::Gemv),
        Some("packed") => Some(Mode::Packed),
        Some("mma") => Some(Mode::Mma),
        Some(other) => anyhow::bail!("QWEN35_PREFILL must be gemv, packed or mma, got {other}"),
        None => None,
    };
    let requested = match requested {
        Some(Mode::Packed) if !crate::prefill::supports(&config.text_config) => Some(Mode::Gemv),
        other => other,
    };
    let baseline_prefill = prefill_bytes(
        &config.text_config,
        Mode::Gemv,
        PREFILL_BLOCK,
        weights.format(),
    )?;
    let free = override_free(free, free_override_mib()?);
    let budget =
        |i: usize| free[i].saturating_sub(ff_core::probe::device_admission_reserve_bytes());
    let slots = ring_slots as u64 * slot_union_bytes(weights, config)?;
    let mut plans = Vec::with_capacity(ranges.len());
    for (i, &(start, end)) in ranges.iter().enumerate() {
        let resident = range_device_bytes(weights, config, start..end, max_ctx)?;
        let mut projs = Vec::with_capacity(end - start);
        let mut streamed = 0u64;
        for layer in start..end {
            let proj = streamed_layer_bytes(weights, config, layer)?;
            projs.push(proj);
            streamed += proj;
        }
        let statics = if i == 0 {
            // Batch::new's persistent q/g slabs, the mode's xb buffer, and
            // int4_group_sums' xs row, at the plan's largest candidate T.
            let batch_geom =
                crate::prefill::BatchGeom::of(&config.text_config, crate::prefill::BLOCK);
            let slab = crate::prefill::BLOCK * (batch_geom.q / 2) * 4 * 2;
            let mode_buf = match requested.unwrap_or(Mode::Mma) {
                Mode::Gemv => {
                    config
                        .text_config
                        .hidden_size
                        .max(config.text_config.conv_dim())
                        * 4
                }
                Mode::Packed => {
                    crate::prefill::BLOCK
                        * (4 * config.text_config.hidden_size)
                            .max(config.text_config.conv_dim())
                            .max(config.text_config.intermediate_size)
                        * 4
                }
                Mode::Mma => {
                    crate::prefill::BLOCK
                        * config
                            .text_config
                            .intermediate_size
                            .max(config.text_config.hidden_size)
                        * 2
                        + crate::prefill::BLOCK * (config.text_config.intermediate_size / 64) * 4
                }
            };
            static_device_bytes(weights, config)?
                + round_up_to((slab + mode_buf) as u64, granularity)
        } else {
            0
        };
        // Every pool-backed term is rounded up to the driver's allocation
        // granularity.
        let round = |v: u64| round_up_to(v, granularity);
        let plan = |extra| {
            place(
                round(resident + statics + extra),
                round(resident - streamed + slots + statics + extra),
                &projs,
                start,
                budget(i),
                force_stream,
            )
        };
        // Per-engine candidates: each engine's largest T whose workspace
        // keeps the plan's residency. The probe picks one after the
        // weights are up; the GEMV planes (extra 0) are always present.
        let plain = plan(0);
        let mut prefill_candidates: Vec<(Mode, usize)> = Vec::new();
        let mut engines: Vec<Mode> = Vec::new();
        match requested {
            Some(engine) => engines.push(engine),
            None => {
                engines.push(Mode::Mma);
                if !is16 {
                    engines.push(Mode::Packed);
                }
            }
        }
        'outer: for engine in engines {
            for t in [crate::prefill::BLOCK, 512, 256, 128, 64, PREFILL_BLOCK] {
                let extra = prefill_bytes(&config.text_config, engine, t, weights.format())?
                    .checked_sub(baseline_prefill)
                    .context("prefill workspace shrank below GEMV planes")?;
                let (mode, _) = select_prefill(engine, plain, plan(extra));
                if mode != Mode::Gemv {
                    prefill_candidates.push((mode, t));
                    continue 'outer;
                }
            }
        }
        if is16 {
            prefill_candidates.push((Mode::Gemv, PREFILL_BLOCK));
        }
        if prefill_candidates.is_empty() {
            prefill_candidates.push((Mode::Gemv, PREFILL_BLOCK));
        }
        // The committed placement covers the largest candidate's
        // workspace, so the probe's later choice always fits inside it.
        let mut max_extra = 0u64;
        let mut selected = plain;
        for (engine, t) in &prefill_candidates {
            let extra = prefill_bytes(&config.text_config, *engine, *t, weights.format())?
                .checked_sub(baseline_prefill)
                .context("prefill workspace shrank below GEMV planes")?;
            let (_, candidate) = select_prefill(*engine, plain, plan(extra));
            if extra >= max_extra {
                max_extra = extra;
                selected = candidate;
            }
        }
        let extra = max_extra;
        plans.push(DevicePlan {
            prefill_candidates,
            residency: selected.residency,
            resident_through: selected.through,
            resident_bytes: round(resident + statics + extra),
            streaming_bytes: selected.bytes,
            free: free[i],
        });
    }
    Ok(plans)
}

fn select_prefill(mode: Mode, plain: Placement, candidate: Placement) -> (Mode, Placement) {
    if mode != Mode::Gemv && !candidate.keeps(&plain) {
        (Mode::Gemv, plain)
    } else {
        (mode, candidate)
    }
}

fn upload_boundary(
    ctx: &GpuContext,
    weights: &Qwen35Weights,
    first_layer: usize,
) -> Result<CudaSliceF> {
    let name = format!("{TEXT_PREFIX}.layers.{first_layer}.input_layernorm.weight");
    ctx.upload_f32(&weights.f32_named(&name)?)
}

struct LayerUploads {
    proj: HashMap<String, GpuQuant>,
    proj16: HashMap<String, GpuProj16>,
    ln: Vec<[CudaSliceF; 2]>,
    attn_norms: Vec<[CudaSliceF; 2]>,
    gdn: Vec<GpuGdn>,
    kv_keys: Vec<CudaSliceF>,
    kv_values: Vec<CudaSliceF>,
}

/// One 16-bit projection's raw bytes on the device with its f32 output.
fn upload_proj16(ctx: &GpuContext, weights: &Qwen35Weights, name: &str) -> Result<GpuProj16> {
    let projection = weights.host_proj16(name)?;
    Ok(GpuProj16 {
        w: ctx
            .stream
            .clone_htod(projection.bytes())
            .with_context(|| format!("upload {name}"))?,
        y: ctx.stream.alloc_zeros::<f32>(projection.out_dim)?,
        out_dim: projection.out_dim,
        in_dim: projection.in_dim,
    })
}

fn upload_stack(
    ctx: &GpuContext,
    weights: &Qwen35Weights,
    text: &TextConfig,
    layers: Range<usize>,
    max_ctx: usize,
    resident_through: usize,
) -> Result<LayerUploads> {
    let kv_stride = text.num_key_value_heads * text.head_dim;
    let conv_dim = text.conv_dim();
    let eps = text.rms_norm_eps as f32;
    let format16 = weights.format().is_16bit();
    let mut proj = HashMap::new();
    let mut proj16 = HashMap::new();
    let mut ln = Vec::new();
    let mut attn_norms = Vec::new();
    let mut gdn = Vec::new();
    let mut kv_keys = Vec::new();
    let mut kv_values = Vec::new();
    for layer in layers {
        let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
        ln.push([
            ctx.upload_f32(&weights.f32_named(&format!("{prefix}.input_layernorm.weight"))?)?,
            ctx.upload_f32(
                &weights.f32_named(&format!("{prefix}.post_attention_layernorm.weight"))?,
            )?,
        ]);
        match text.layer_kind(layer) {
            LayerKind::LinearAttention => {
                if layer < resident_through {
                    for p in [
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_b",
                        "in_proj_a",
                        "out_proj",
                    ] {
                        let name = format!("{prefix}.linear_attn.{p}");
                        if format16 {
                            proj16.insert(name.clone(), upload_proj16(ctx, weights, &name)?);
                        } else {
                            let q = weights.quant_projection(&name)?;
                            proj.insert(name, ctx.upload(&q, None)?);
                        }
                    }
                }
                gdn.push(GpuGdn::upload(
                    ctx,
                    GdnUpload {
                        conv1d: &weights
                            .f32_named(&format!("{prefix}.linear_attn.conv1d.weight"))?,
                        a_log: &weights.f32_named(&format!("{prefix}.linear_attn.A_log"))?,
                        dt_bias: &weights.f32_named(&format!("{prefix}.linear_attn.dt_bias"))?,
                        norm: &weights.f32_named(&format!("{prefix}.linear_attn.norm.weight"))?,
                        conv_dim,
                        kernel: text.linear_conv_kernel_dim,
                        num_v: text.linear_num_value_heads,
                        num_k: text.linear_num_key_heads,
                        dk: text.linear_key_head_dim,
                        dv: text.linear_value_head_dim,
                        eps,
                    },
                )?);
            }
            LayerKind::FullAttention => {
                if layer < resident_through {
                    for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                        let name = format!("{prefix}.self_attn.{p}");
                        if format16 {
                            proj16.insert(name.clone(), upload_proj16(ctx, weights, &name)?);
                        } else {
                            let q = weights.quant_projection(&name)?;
                            proj.insert(name, ctx.upload(&q, None)?);
                        }
                    }
                }
                attn_norms.push([
                    ctx.upload_f32(
                        &weights.f32_named(&format!("{prefix}.self_attn.q_norm.weight"))?,
                    )?,
                    ctx.upload_f32(
                        &weights.f32_named(&format!("{prefix}.self_attn.k_norm.weight"))?,
                    )?,
                ]);
                kv_keys.push(
                    ctx.stream
                        .alloc_zeros::<f32>(max_ctx * kv_stride)
                        .context("kv k")?,
                );
                kv_values.push(
                    ctx.stream
                        .alloc_zeros::<f32>(max_ctx * kv_stride)
                        .context("kv v")?,
                );
            }
        }
        if layer < resident_through {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                let name = format!("{prefix}.mlp.{p}");
                if format16 {
                    proj16.insert(name.clone(), upload_proj16(ctx, weights, &name)?);
                } else {
                    let q = weights.quant_projection(&name)?;
                    proj.insert(name, ctx.upload(&q, None)?);
                }
            }
        }
    }
    Ok(LayerUploads {
        proj,
        proj16,
        ln,
        attn_norms,
        gdn,
        kv_keys,
        kv_values,
    })
}

/// One projection name's device buffers inside a streaming slot.
struct SlotBuf {
    packed: CudaSlice<u32>,
    scales: CudaSlice<u16>,
    biases: CudaSlice<u16>,
    y: CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
}

/// Per-device streaming state: a two-slot ring refilled on a prefetch
/// stream, ordered against the compute stream by fill/free events. A
/// layer's slot is its index parity, so consecutive tokens reuse the
/// same two slots.
pub(crate) struct StreamState {
    prefetch: Arc<CudaStream>,
    slots: [HashMap<String, SlotBuf>; 2],
    filled: [CudaEvent; 2],
    freed: [CudaEvent; 2],
    host: Vec<HashMap<String, HostProjection>>,
    range: (usize, usize),
}

impl StreamState {
    fn slot_of(layer: usize) -> usize {
        layer & 1
    }

    fn build(
        ctx: &GpuContext,
        weights: &Qwen35Weights,
        text: &TextConfig,
        range: (usize, usize),
    ) -> Result<Self> {
        let prefetch = ctx.context.new_stream().context("prefetch stream")?;
        let mut host = Vec::with_capacity(range.1 - range.0);
        let mut dims: HashMap<String, (usize, usize)> = HashMap::new();
        for layer in range.0..range.1 {
            let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
            let mut per_layer = HashMap::new();
            for &suffix in streaming_names(text.layer_kind(layer)) {
                let projection = weights.host_projection(&format!("{prefix}.{suffix}"))?;
                let shape = (projection.out_dim, projection.in_dim);
                match dims.get(suffix) {
                    Some(seen) => ensure!(
                        *seen == shape,
                        "{prefix}.{suffix} shape {shape:?} differs from earlier layers"
                    ),
                    None => {
                        dims.insert(suffix.to_string(), shape);
                    }
                }
                per_layer.insert(suffix.to_string(), projection);
            }
            host.push(per_layer);
        }
        let mut slots = [HashMap::new(), HashMap::new()];
        for (suffix, (out_dim, in_dim)) in &dims {
            let words = out_dim * in_dim / 8;
            let groups = out_dim * in_dim / GROUP_SIZE;
            for map in slots.iter_mut() {
                map.insert(
                    suffix.clone(),
                    SlotBuf {
                        packed: ctx
                            .stream
                            .alloc_zeros::<u32>(words)
                            .context("slot packed")?,
                        scales: ctx
                            .stream
                            .alloc_zeros::<u16>(groups)
                            .context("slot scales")?,
                        biases: ctx
                            .stream
                            .alloc_zeros::<u16>(groups)
                            .context("slot biases")?,
                        y: ctx.stream.alloc_zeros::<f32>(*out_dim).context("slot y")?,
                        out_dim: *out_dim,
                        in_dim: *in_dim,
                    },
                );
            }
        }
        let filled = [
            ctx.context.new_event(None).context("fill event")?,
            ctx.context.new_event(None).context("fill event")?,
        ];
        let freed = [
            ctx.context.new_event(None).context("free event")?,
            ctx.context.new_event(None).context("free event")?,
        ];
        for event in &freed {
            event.record(&ctx.stream).context("initial free record")?;
        }
        ctx.stream.synchronize().context("slot alloc sync")?;
        Ok(Self {
            prefetch,
            slots,
            filled,
            freed,
            host,
            range,
        })
    }

    fn slot_buf(&self, layer: usize, suffix: &str) -> Result<&SlotBuf> {
        self.slots[Self::slot_of(layer)]
            .get(suffix)
            .with_context(|| format!("{suffix} missing from slot"))
    }

    /// Queue one layer's projections into its slot on the prefetch
    /// stream; the slot's previous reader frees it first.
    fn fetch(&mut self, layer: usize) -> Result<()> {
        let slot = Self::slot_of(layer);
        self.prefetch
            .wait(&self.freed[slot])
            .context("prefetch wait")?;
        let entries = &self.host[layer - self.range.0];
        let bufs = &mut self.slots[slot];
        for (suffix, projection) in entries {
            let buf = bufs
                .get_mut(suffix)
                .with_context(|| format!("{suffix} slot missing"))?;
            self.prefetch
                .memcpy_htod(projection.packed.as_slice(), &mut buf.packed)?;
            self.prefetch
                .memcpy_htod(&projection.scales, &mut buf.scales)?;
            self.prefetch
                .memcpy_htod(&projection.biases, &mut buf.biases)?;
        }
        self.filled[slot].record(&self.prefetch)?;
        Ok(())
    }

    /// Fill the ring's first slot ahead of the range's first layer; the
    /// layer+1 fetch rides the compute-wait of `begin_layer`, so priming the
    /// second layer here would be uploaded twice.
    fn prime(&mut self) -> Result<()> {
        self.fetch(self.range.0)
    }

    /// The compute stream waits the layer's slot; layer+1 prefetches into
    /// the slot the previous layer freed.
    fn begin_layer(&mut self, ctx: &GpuContext, layer: usize) -> Result<()> {
        ctx.stream.wait(&self.filled[Self::slot_of(layer)])?;
        if layer + 1 < self.range.1 {
            self.fetch(layer + 1)?;
        }
        Ok(())
    }

    /// The slot is reusable once the layer's last projection reader — the
    /// closing residual norm — is enqueued.
    fn end_layer(&mut self, ctx: &GpuContext, layer: usize) -> Result<()> {
        self.freed[Self::slot_of(layer)].record(&ctx.stream)?;
        Ok(())
    }
}

/// A resident 16-bit projection: raw dtype-native weights + f32 output.
pub(crate) struct GpuProj16 {
    w: CudaSlice<u8>,
    y: CudaSliceF,
    out_dim: usize,
    in_dim: usize,
}

impl GpuProj16 {
    fn seg(&self) -> Seg16<'_> {
        Seg16 {
            w: &self.w,
            y: &self.y,
            rows: self.out_dim,
        }
    }
}

/// One projection name's device buffers inside a 16-bit streaming slot.
struct SlotBuf16 {
    w: CudaSlice<u8>,
    y: CudaSliceF,
    out_dim: usize,
    in_dim: usize,
}

/// Raw write handle into one pinned staging buffer. The owning
/// PinnedHostSlice lives in StreamState16; only the fill worker writes.
struct PinnedPtr {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for PinnedPtr {}

impl PinnedPtr {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Per-slot fill publication: the layer the pinned buffers currently hold.
type FillGate = Arc<(std::sync::Mutex<Option<usize>>, std::sync::Condvar)>;

/// Streaming pipeline counters: where one streamed token's time goes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stream16Stats {
    /// Worker time copying the mapping into pinned staging.
    pub fill_ms: f64,
    /// Driver time blocked on the fill gate.
    pub gate_wait_ms: f64,
    /// Event-timed H2D on the prefetch stream.
    pub h2d_ms: f64,
    pub h2d_bytes: u64,
}

/// Streaming-ring geometry derived once at setup from measured rates and
/// the host RAM budget. depth grows by one slot when the fill side is the
/// slower stage, capped so the pinned ring stays within a quarter of
/// available RAM.
#[derive(Clone, Copy)]
pub struct RingGeom {
    pub depth: usize,
    pub threads: usize,
}

fn mem_available_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemAvailable:"))
                .and_then(|line| {
                    line.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<u64>().ok())
                })
        })
        .map(|kib| kib * 1024)
        .unwrap_or(u64::MAX)
}

/// Probe the device's pinned-H2D rate and a single fill thread's copy rate,
/// then derive the ring geometry. Two slots of pipeline minimum; one more
/// when the fill side is slower than H2D, bounded by the RAM budget.
pub fn ring_geom(
    context: &Arc<cudarc::driver::CudaContext>,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
) -> Result<RingGeom> {
    let slot_bytes = slot_union_bytes(weights, config)?;
    const PROBE: usize = 32 << 20;
    let mut pinned = unsafe { context.alloc_pinned::<u8>(PROBE) }.context("probe pinned")?;
    unsafe {
        pinned
            .as_mut_ptr()
            .context("probe pinned ptr")?
            .write_bytes(1, PROBE)
    };
    let prefetch = context.new_stream().context("probe stream")?;
    let mut dev = prefetch.alloc_zeros::<u8>(PROBE).context("probe device")?;
    prefetch.memcpy_htod(&pinned, &mut dev)?;
    prefetch.synchronize()?;
    let start = std::time::Instant::now();
    for _ in 0..3 {
        prefetch.memcpy_htod(&pinned, &mut dev)?;
    }
    prefetch.synchronize()?;
    let h2d_rate = 3.0 * PROBE as f64 / start.elapsed().as_secs_f64();

    let src = vec![2u8; PROBE];
    let t = std::time::Instant::now();
    for _ in 0..3 {
        copy_pinned(
            pinned.as_mut_slice().context("probe pinned slice")?,
            &src,
            1,
        );
    }
    let thread_rate = 3.0 * PROBE as f64 / t.elapsed().as_secs_f64();

    let derived = (h2d_rate / thread_rate).ceil() as usize;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let derived = derived.clamp(1, (cores / 2).max(1));
    let threads = derived;
    let fill_rate = threads as f64 * thread_rate;
    let target = 2 + usize::from(fill_rate < h2d_rate);
    let ram_cap = ((mem_available_bytes() / 4 / slot_bytes.max(1)) as usize).clamp(2, 8);
    Ok(RingGeom {
        depth: target.min(ram_cap),
        threads,
    })
}

/// Parallel memcpy from the checkpoint mapping into a pinned buffer.
fn copy_pinned(dst: &mut [u8], src: &[u8], threads: usize) {
    let chunk = src.len().div_ceil(threads).max(2 << 20);
    std::thread::scope(|s| {
        for (d, c) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            s.spawn(move || d.copy_from_slice(c));
        }
    });
}

/// 16-bit streaming state: a device ring refilled on a prefetch stream
/// from pinned staging, which a fill worker copies from the checkpoint
/// mapping ahead of use. Ring depth and fill-thread count come from
/// ring_geom's setup probes, not constants. The worker never runs more
/// than one layer behind the last fetch request; per-slot gates publish
/// which layer the pinned buffers hold, so stale fills from an earlier
/// request cannot be consumed.
pub(crate) struct StreamState16 {
    prefetch: Arc<CudaStream>,
    slots: Vec<HashMap<String, SlotBuf16>>,
    pinned: Vec<HashMap<String, PinnedHostSlice<u8>>>,
    filled: Vec<Arc<CudaEvent>>,
    freed: Vec<CudaEvent>,
    gates: Vec<FillGate>,
    queue: Option<std::sync::mpsc::Sender<usize>>,
    /// Set by the exit watcher when the fill worker dies, with its error.
    worker_exit: Arc<std::sync::Mutex<Option<String>>>,
    last_queued: Option<usize>,
    range: (usize, usize),
    /// Each layer's own projection suffixes; a fetch transfers only these.
    suffixes: Vec<&'static [&'static str]>,
    stats: Arc<std::sync::Mutex<Stream16Stats>>,
    h2d_start: Vec<Option<CudaEvent>>,
    context: Arc<cudarc::driver::CudaContext>,
}

impl StreamState16 {
    fn slot_of(&self, layer: usize) -> usize {
        layer % self.slots.len()
    }

    fn build(
        ctx: &GpuContext,
        weights: &Qwen35Weights,
        text: &TextConfig,
        range: (usize, usize),
        geom: &RingGeom,
    ) -> Result<Self> {
        let prefetch = ctx.context.new_stream().context("prefetch stream")?;
        let mut host = Vec::with_capacity(range.1 - range.0);
        let mut dims: HashMap<String, (usize, usize)> = HashMap::new();
        for layer in range.0..range.1 {
            let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
            let mut per_layer = HashMap::new();
            for &suffix in streaming_names(text.layer_kind(layer)) {
                let projection = weights.host_proj16(&format!("{prefix}.{suffix}"))?;
                let shape = (projection.out_dim, projection.in_dim);
                match dims.get(suffix) {
                    Some(seen) => ensure!(
                        *seen == shape,
                        "{prefix}.{suffix} shape {shape:?} differs from earlier layers"
                    ),
                    None => {
                        dims.insert(suffix.to_string(), shape);
                    }
                }
                per_layer.insert(suffix.to_string(), projection);
            }
            host.push(per_layer);
        }
        let depth = geom.depth;
        let mut slots: Vec<HashMap<String, SlotBuf16>> =
            (0..depth).map(|_| HashMap::new()).collect();
        let mut pinned: Vec<HashMap<String, PinnedHostSlice<u8>>> =
            (0..depth).map(|_| HashMap::new()).collect();
        for (suffix, &(out_dim, in_dim)) in &dims {
            let bytes = out_dim * in_dim * 2;
            for (map, pmap) in slots.iter_mut().zip(pinned.iter_mut()) {
                map.insert(
                    suffix.clone(),
                    SlotBuf16 {
                        w: ctx.stream.alloc_zeros::<u8>(bytes).context("slot w")?,
                        y: ctx.stream.alloc_zeros::<f32>(out_dim).context("slot y")?,
                        out_dim,
                        in_dim,
                    },
                );
                pmap.insert(
                    suffix.clone(),
                    unsafe { ctx.context.alloc_pinned::<u8>(bytes) }.context("pinned slot")?,
                );
            }
        }
        // Timing-enabled: the H2D elapsed measurement reads these.
        let mut filled = Vec::with_capacity(depth);
        for _ in 0..depth {
            filled.push(Arc::new(
                ctx.context
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .context("fill event")?,
            ));
        }
        let mut freed = Vec::with_capacity(depth);
        for _ in 0..depth {
            freed.push(ctx.context.new_event(None).context("free event")?);
        }
        for event in &freed {
            event.record(&ctx.stream).context("initial free record")?;
        }
        // The worker synchronizes a slot's fill event before rewriting its
        // pinned buffers; pre-record so the first fill has a completed event.
        for event in &filled {
            event.record(&ctx.stream).context("initial fill record")?;
        }
        ctx.stream.synchronize().context("slot alloc sync")?;
        let gates: Vec<FillGate> = (0..depth)
            .map(|_| Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new())))
            .collect();
        let (tx, rx) = std::sync::mpsc::channel::<usize>();
        let stats = Arc::new(std::sync::Mutex::new(Stream16Stats::default()));
        let worker_exit: Arc<std::sync::Mutex<Option<String>>> =
            Arc::new(std::sync::Mutex::new(None));
        {
            let mut pinned_ptrs: Vec<HashMap<String, PinnedPtr>> =
                (0..depth).map(|_| HashMap::new()).collect();
            for (slot, map) in pinned.iter_mut().enumerate() {
                for (suffix, buf) in map.iter_mut() {
                    pinned_ptrs[slot].insert(
                        suffix.clone(),
                        PinnedPtr {
                            ptr: buf.as_mut_ptr().context("pinned pointer")?,
                            len: buf.len(),
                        },
                    );
                }
            }
            let (gates, filled) = (gates.clone(), filled.clone());
            let stats = stats.clone();
            let threads = geom.threads;
            let worker_exit = worker_exit.clone();
            std::thread::Builder::new()
                .name("qwen35-fill".to_owned())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        fill_worker(FillArgs {
                            rx,
                            host,
                            pinned: pinned_ptrs,
                            gates: gates.clone(),
                            filled,
                            first: range.0,
                            stats,
                            depth,
                            threads,
                        })
                    }));
                    let message = match result {
                        Err(payload) => format!("fill worker panicked: {payload:?}"),
                        Ok(Err(e)) => format!("fill worker failed: {e:#}"),
                        Ok(Ok(())) => "fill worker exited before any fill completed".to_owned(),
                    };
                    *worker_exit.lock().expect("worker exit slot poisoned") = Some(message);
                    for gate in &gates {
                        let (lock, cvar) = &**gate;
                        *lock.lock().expect("fill gate poisoned") = Some(usize::MAX);
                        cvar.notify_all();
                    }
                })
                .context("spawn fill worker")?
        };
        Ok(Self {
            prefetch,
            slots,
            pinned,
            filled,
            freed,
            gates,
            queue: Some(tx),
            worker_exit,
            last_queued: None,
            range,
            suffixes: (range.0..range.1)
                .map(|layer| streaming_names(text.layer_kind(layer)))
                .collect(),
            stats,
            h2d_start: (0..depth).map(|_| None).collect(),
            context: ctx.context.clone(),
        })
    }

    fn slot_buf(&self, layer: usize, suffix: &str) -> Result<&SlotBuf16> {
        self.slots[self.slot_of(layer)]
            .get(suffix)
            .with_context(|| format!("{suffix} missing from slot"))
    }

    /// Queue one layer's projections into its slot on the prefetch stream,
    /// after the fill worker published them; the next layer goes to the
    /// worker first so it stays ahead.
    fn fetch(&mut self, layer: usize) -> Result<()> {
        let slot = self.slot_of(layer);
        if layer + 1 < self.range.1 && self.last_queued != Some(layer + 1) {
            if let Some(queue) = &self.queue {
                queue
                    .send(layer + 1)
                    .context("fill worker gone while queueing")?;
            }
            self.last_queued = Some(layer + 1);
        }
        let (lock, cvar) = &*self.gates[slot];
        let mut ready = lock.lock().expect("fill gate poisoned");
        let gate_wait = std::time::Instant::now();
        while *ready != Some(layer) {
            if let Some(message) = self
                .worker_exit
                .lock()
                .expect("worker exit slot poisoned")
                .as_ref()
            {
                anyhow::bail!(
                    "streamed 16-bit fill for layer {layer} (slot {slot}) never completed: {message}"
                );
            }
            ready = cvar.wait(ready).expect("fill gate poisoned");
        }
        drop(ready);
        // The gate implies the worker drained this slot's previous H2D, so
        // its event pair is readable now.
        let gate_wait = gate_wait.elapsed();
        let mut h2d_ms = 0.0f64;
        if let Some(start) = self.h2d_start[slot].take() {
            h2d_ms = start
                .elapsed_ms(&self.filled[slot])
                .context("h2d elapsed")? as f64;
        }
        self.prefetch
            .wait(&self.freed[slot])
            .context("prefetch wait")?;
        let h2d_start = self.ctx_event()?;
        h2d_start
            .record(&self.prefetch)
            .context("h2d start record")?;
        let mut bytes = 0u64;
        for &suffix in self.suffixes[layer - self.range.0].iter() {
            let buf = self.slots[slot]
                .get_mut(suffix)
                .with_context(|| format!("{suffix} slot missing"))?;
            let pinned = &self.pinned[slot][suffix];
            bytes += pinned.len() as u64;
            self.prefetch
                .memcpy_htod(pinned, &mut buf.w)
                .context("slot h2d")?;
        }
        self.filled[slot]
            .record(&self.prefetch)
            .context("fill event record")?;
        self.h2d_start[slot] = Some(h2d_start);
        {
            let mut stats = self.stats.lock().expect("stream stats poisoned");
            stats.gate_wait_ms += gate_wait.as_secs_f64() * 1000.0;
            stats.h2d_bytes += bytes;
            stats.h2d_ms += h2d_ms;
        }
        Ok(())
    }

    fn ctx_event(&self) -> Result<CudaEvent> {
        self.context
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .context("h2d start event")
    }

    /// Fill the ring's first slot ahead of the range's first layer.
    fn prime(&mut self) -> Result<()> {
        if self.last_queued != Some(self.range.0) {
            if let Some(queue) = &self.queue {
                queue
                    .send(self.range.0)
                    .context("fill worker gone while priming")?;
            }
            self.last_queued = Some(self.range.0);
        }
        self.fetch(self.range.0)
    }

    /// The compute stream waits the layer's slot; layer+1 prefetches into
    /// the slot the previous layer freed.
    fn begin_layer(&mut self, ctx: &GpuContext, layer: usize) -> Result<()> {
        ctx.stream.wait(&self.filled[self.slot_of(layer)])?;
        if layer + 1 < self.range.1 {
            self.fetch(layer + 1)?;
        }
        Ok(())
    }

    /// The slot is reusable once the layer's last projection reader — the
    /// closing residual norm — is enqueued.
    fn end_layer(&mut self, ctx: &GpuContext, layer: usize) -> Result<()> {
        self.freed[self.slot_of(layer)].record(&ctx.stream)?;
        Ok(())
    }
}

impl Drop for StreamState16 {
    fn drop(&mut self) {
        self.queue.take();
    }
}

/// Everything the fill worker owns: the layer queue, the host projections
/// and pinned targets, and the derived geometry.
struct FillArgs {
    rx: std::sync::mpsc::Receiver<usize>,
    host: Vec<HashMap<String, HostProj16>>,
    pinned: Vec<HashMap<String, PinnedPtr>>,
    gates: Vec<FillGate>,
    filled: Vec<Arc<CudaEvent>>,
    first: usize,
    stats: Arc<std::sync::Mutex<Stream16Stats>>,
    depth: usize,
    threads: usize,
}

/// Fill one requested layer per message: wait the slot's last H2D, copy
/// the layer's projections from the mapping into pinned staging with a
/// thread gang, then publish the layer number.
fn fill_worker(args: FillArgs) -> Result<()> {
    let FillArgs {
        rx,
        host,
        mut pinned,
        gates,
        filled,
        first,
        stats,
        depth,
        threads,
    } = args;
    while let Ok(layer) = rx.recv() {
        let slot = layer % depth;
        filled[slot].synchronize().context("slot H2D drain")?;
        let fill_start = std::time::Instant::now();
        let pinned = &mut pinned[slot];
        for (suffix, projection) in &host[layer - first] {
            let dst = pinned
                .get_mut(suffix)
                .with_context(|| format!("{suffix} pinned slot missing"))?;
            copy_pinned(dst.as_mut_slice(), projection.bytes(), threads);
        }
        let fill_ms = fill_start.elapsed().as_secs_f64() * 1000.0;
        {
            let mut stats = stats.lock().expect("stream stats poisoned");
            stats.fill_ms += fill_ms;
        }
        let (lock, cvar) = &*gates[slot];
        *lock.lock().expect("fill gate poisoned") = Some(layer);
        cvar.notify_one();
    }
    Ok(())
}

pub(crate) struct PeerGpu {
    prefill: Mode,
    attn: Option<AttnPrefillGpu>,
    batch: crate::prefill::Batch,
    ctx: GpuContext,
    wide: crate::wide::WideKernels,
    g16: Gemv16Kernels,
    proj: HashMap<String, GpuQuant>,
    proj16: HashMap<String, GpuProj16>,
    ln: Vec<[CudaSliceF; 2]>,
    attn_norms: Vec<[CudaSliceF; 2]>,
    gdn: Vec<GpuGdn>,
    kv_keys: Vec<CudaSliceF>,
    kv_values: Vec<CudaSliceF>,
    q_out: CudaSliceF,
    gate_out: CudaSliceF,
    attn_out: CudaSliceF,
    inner: CudaSliceF,
    hidden: CudaSliceF,
    x1: CudaSliceF,
    pos: CudaSliceI,
    rope_pos: CudaSliceI,
    /// Per-block position planes for prefill (step-(i) hoisted uploads).
    pos_block: CudaSliceI,
    rope_pos_block: CudaSliceI,
    boundary_norm: CudaSliceF,
    streaming: Option<StreamState>,
    streaming16: Option<StreamState16>,
}

pub struct QwenGpu {
    prefill: Mode,
    prefill_block: usize,
    attn: Option<AttnPrefillGpu>,
    batch: crate::prefill::Batch,
    pub(crate) config: Qwen35Config,
    format: QuantFormat,
    /// Spec-mode access (the verify driver pokes these directly).
    pub ctx: GpuContext,
    pub(crate) proj: HashMap<String, GpuQuant>,
    pub(crate) proj16: HashMap<String, GpuProj16>,
    pub(crate) ln: Vec<[CudaSliceF; 2]>,
    pub(crate) attn_norms: Vec<[CudaSliceF; 2]>,
    pub(crate) final_norm_w: CudaSliceF,
    pub(crate) gdn: Vec<GpuGdn>,
    pub(crate) kv_keys: Vec<CudaSliceF>,
    pub(crate) kv_values: Vec<CudaSliceF>,
    pub(crate) q_out: CudaSliceF,
    pub(crate) gate_out: CudaSliceF,
    pub(crate) attn_out: CudaSliceF,
    pub(crate) inner: CudaSliceF,
    /// Spec-mode access (the verify driver pokes these directly).
    pub hidden: CudaSliceF,
    pub(crate) x1: CudaSliceF,
    /// Spec-mode access (the verify driver pokes these directly).
    pub pos: CudaSliceI,
    /// mrope position [t,h,w], device-side; prefill writes per token,
    /// decode advances via inc3 (graph-captured). Bit-identical at [p,p,p].
    pub rope_pos: CudaSliceI,
    /// Per-block position planes for prefill (step-(i) hoisted uploads).
    pos_block: CudaSliceI,
    rope_pos_block: CudaSliceI,
    /// Spec-mode access (the verify driver pokes these directly).
    pub next_token: CudaSliceI,
    /// Captured decode graph (QWEN35_GRAPH=0 disables; pos/next_token are
    /// device-read so the step replays without host input).
    /// Per-device captured resident-span graphs; index 0 is device 0 and
    /// i+1 its peers. None until first replay.
    span_graphs: Vec<Option<ff_edge0::gpu::DecodeGraph>>,
    /// Spare for the borrow swap at finalize (argmax needs &mut next_token
    /// while lm_head's y borrows the projection map).
    nt_spare: CudaSliceI,
    /// Host mirror of the position counter (prefill loop bookkeeping).
    pub position: usize,
    prefill_packed_tokens: usize,
    /// KV-cache capacity; step() refuses to write past it.
    max_ctx: usize,
    /// Wide split-K GEMVs for dense MLP projections.
    pub(crate) wide: crate::wide::WideKernels,
    /// Grouped 16-bit GEMVs for raw bf16/f16 checkpoints (gemv16.rs).
    pub(crate) g16: Gemv16Kernels,
    /// Host-side embedding table for 16-bit checkpoints (row gather + a
    /// small H2D per token); None on int4.
    embed_host: Option<HostProj16>,
    /// The last token read back to the host; the 16-bit embed gathers
    /// its row. Set by decode() entry and read_token().
    next_host_token: Option<u32>,
    /// Layer-stack hosts for ordinals[1..]; empty on a single device.
    pub(crate) peers: Vec<PeerGpu>,
    /// dev0's boundary norm when peers exist; the final norm otherwise.
    pub(crate) boundary_norms: Vec<CudaSliceF>,
    /// Layer range per device (peers included), bytes-balanced.
    pub(crate) ranges: Vec<(usize, usize)>,
    /// Host buffer for the hidden/x1 device hops.
    staging: Vec<f32>,
    /// dev0's streaming weights; None runs its range resident.
    pub(crate) streaming: Option<StreamState>,
    /// dev0's 16-bit streaming weights; None on int4 or full residency.
    pub(crate) streaming16: Option<StreamState16>,
}

pub(crate) struct ProbeState {
    hidden: Vec<f32>,
    x1: Vec<f32>,
    logits: Vec<f32>,
    token: Vec<i32>,
    spare: Vec<i32>,
    pos: Vec<i32>,
    rope: Vec<i32>,
    gdn: Vec<(Vec<f32>, Vec<f32>)>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    position: usize,
    host_token: Option<u32>,
    packed_tokens: usize,
}

impl ProbeState {
    pub(crate) fn bytes(&self) -> usize {
        4 * (self.hidden.len()
            + self.x1.len()
            + self.logits.len()
            + self.token.len()
            + self.spare.len()
            + self.pos.len()
            + self.rope.len()
            + self
                .gdn
                .iter()
                .chain(&self.kv)
                .map(|(a, b)| a.len() + b.len())
                .sum::<usize>())
    }
}

impl QwenGpu {
    pub fn new(
        ordinals: &[usize],
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        force_stream: bool,
    ) -> Result<Self> {
        Self::with_max_ctx(ordinals, weights, config, 4096, force_stream)
    }

    pub fn with_max_ctx(
        ordinals: &[usize],
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        max_ctx: usize,
        force_stream: bool,
    ) -> Result<Self> {
        ensure!(!ordinals.is_empty(), "at least one device ordinal required");
        let format = weights.format();
        ensure!(
            !format.is_16bit()
                || !matches!(std::env::var("QWEN35_PREFILL").as_deref(), Ok("packed")),
            "QWEN35_PREFILL=packed requires the int4 checkpoint"
        );
        for (i, &a) in ordinals.iter().enumerate() {
            for &b in &ordinals[i + 1..] {
                ensure!(a != b, "device ordinal {a} listed twice");
            }
        }
        let ranges = partition_layers(weights, config, ordinals.len())?;
        let contexts: Vec<GpuContext> = ordinals
            .iter()
            .map(|&ordinal| GpuContext::new(ordinal))
            .collect::<Result<_>>()?;
        let mut attention = contexts
            .iter()
            .zip(&ranges)
            .enumerate()
            .map(|(i, (ctx, &(start, end)))| {
                if (start..end)
                    .any(|layer| config.text_config.layer_kind(layer) == LayerKind::FullAttention)
                {
                    ensure!(
                        config.text_config.head_dim == D_ATTN
                            && config.text_config.num_key_value_heads > 0
                            && config.text_config.num_attention_heads > 0
                            && config.text_config.num_attention_heads.is_multiple_of(config.text_config.num_key_value_heads),
                        "tiled attention requires head dimension 256 and an integral positive query/KV head ratio"
                    );
                    Ok(Some(AttnPrefillGpu::new(ctx, max_ctx, &config.text_config).with_context(
                        || format!("tiled attention setup on device {}", ordinals[i]),
                    )?))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let free: Vec<u64> = contexts
            .iter()
            .map(|ctx| Ok(ctx.context.mem_get_info().context("mem_get_info")?.0 as u64))
            .collect::<Result<_>>()?;
        let geom = if format.is_16bit() {
            ring_geom(&contexts[0].context, weights, config)?
        } else {
            RingGeom {
                depth: 2,
                threads: 1,
            }
        };
        let granularity = pool_granularity(contexts[0].context.ordinal())?;
        let plans = plan_residency(
            weights,
            config,
            PlanParams {
                ranges: &ranges,
                max_ctx,
                free: &free,
                ring_slots: geom.depth,
                force_stream,
                granularity,
            },
        )?;
        let mut shortage = String::new();
        for (i, plan) in plans.iter().enumerate() {
            if plan.residency == Residency::Insufficient {
                shortage.push_str(&format!(
                    "device {}: layers {}..{} resident {} B / streaming {} B, {} B free\n",
                    ordinals[i],
                    ranges[i].0,
                    ranges[i].1,
                    plan.resident_bytes,
                    plan.streaming_bytes,
                    plan.free
                ));
            }
        }
        ensure!(
            shortage.is_empty(),
            "device memory short of resident and streaming plans:\n{shortage}pass more device ordinals or free device memory"
        );
        for (i, plan) in plans.iter().enumerate() {
            eprintln!(
                "qwen35: residency: cuda:{} {} / {} layers resident, {} streamed",
                ordinals[i],
                plan.resident_through - ranges[i].0,
                ranges[i].1 - ranges[i].0,
                ranges[i].1 - plan.resident_through,
            );
            if plan.prefill_candidates.len() == 1 && plan.prefill_candidates[0].0 == Mode::Gemv {
                eprintln!(
                    "qwen35: device {}: packed prefill not admitted; using GEMV",
                    ordinals[i]
                );
            }
            if plan.residency == Residency::Streaming {
                eprintln!(
                    "qwen35: device {}: {} layers resident, {} streamed (full residency needs {} B, {} B free)",
                    ordinals[i],
                    plan.resident_through - ranges[i].0,
                    ranges[i].1 - plan.resident_through,
                    plan.resident_bytes,
                    plan.free
                );
            }
        }
        let text = &config.text_config;
        let mut contexts = contexts.into_iter();
        let ctx = contexts.next().expect("ordinals checked non-empty");
        let wide = crate::wide::WideKernels::load(&ctx)?;
        let g16 = Gemv16Kernels::load(&ctx)?;
        let hidden_size = text.hidden_size;

        let attn = attention.next().expect("device count checked");

        let mut stack = upload_stack(
            &ctx,
            weights,
            text,
            ranges[0].0..ranges[0].1,
            max_ctx,
            plans[0].resident_through,
        )?;

        // The prefill mode derives here: the weights for a resident layer
        // are on device, and every Batch below allocates for the choice.
        let streamed_layer = ranges[0].0;
        // The streamed-class extras (the H2D measurement, the probe's
        // streamed term) are int4-only: the 16-bit ring builds its own
        // state below.
        let streaming0 = plans[0].residency == Residency::Streaming && !format.is_16bit();
        let mut h2d_gbps = None;
        if streaming0 {
            // Upload the complete first streamed layer, including packed weights, for the H2D measurement.
            let prefix = format!("{TEXT_PREFIX}.layers.{streamed_layer}");
            let mut host_bytes = 0u64;
            let mut uploads: Vec<CudaSlice<u16>> = Vec::new();
            let mut packed_uploads: Vec<CudaSlice<u32>> = Vec::new();
            let t0 = std::time::Instant::now();
            for &suffix in crate::gpu::streaming_names(text.layer_kind(streamed_layer)) {
                let hp = weights.host_projection(&format!("{prefix}.{suffix}"))?;
                host_bytes += (hp.packed.as_slice().len() * 4
                    + hp.scales.len() * 2
                    + hp.biases.len() * 2) as u64;
                packed_uploads.push(ctx.stream.clone_htod(hp.packed.as_slice())?);
                uploads.push(ctx.stream.clone_htod(&hp.scales)?);
                uploads.push(ctx.stream.clone_htod(&hp.biases)?);
            }
            ctx.stream.synchronize().context("probe h2d sync")?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms > 0.0 && host_bytes > 0 {
                h2d_gbps = Some(host_bytes as f64 / (ms / 1000.0) / 1e9);
            }
            drop(packed_uploads);
            drop(uploads);
        }
        let src = if !stack.proj.is_empty() {
            let g = stack
                .proj
                .iter()
                .find(|(k, _)| k.contains("mlp.gate_proj"))
                .map(|(_, v)| v)
                .context("probe: no resident gate projection")?;
            let u = stack
                .proj
                .iter()
                .find(|(k, _)| k.contains("mlp.up_proj"))
                .map(|(_, v)| v)
                .context("probe: no resident up projection")?;
            ProbeSrc::Int4(g, u)
        } else {
            match (
                stack
                    .proj16
                    .iter()
                    .find(|(k, _)| k.contains("mlp.gate_proj")),
                stack.proj16.iter().find(|(k, _)| k.contains("mlp.up_proj")),
            ) {
                (Some((_, g)), Some((_, u))) => ProbeSrc::P16(g, u),
                _ => ProbeSrc::None,
            }
        };
        let (chosen_mode, chosen_block, probe_line) = probe_prefill_choice(
            &ctx,
            ordinals[0],
            ProbeArgs {
                candidates: &plans[0].prefill_candidates,
                streaming: streaming0,
                streamed_bytes_per_layer: 0,
                h2d_gbps,
                src,
                format: weights.format(),
                text,
                source: if stack.proj.is_empty() {
                    "streamed slot"
                } else {
                    "resident layer"
                },
                override_env: ff_core::probe::env_value::<String>("QWEN35_PREFILL")?.as_deref(),
            },
        )?;
        eprintln!("qwen35: {probe_line}");
        let _ = streamed_layer;

        let streaming = (plans[0].residency == Residency::Streaming && !format.is_16bit())
            .then(|| {
                StreamState::build(
                    &ctx,
                    weights,
                    text,
                    (plans[0].resident_through, ranges[0].1),
                )
            })
            .transpose()?;
        let streaming16 = (plans[0].residency == Residency::Streaming && format.is_16bit())
            .then(|| {
                StreamState16::build(
                    &ctx,
                    weights,
                    text,
                    (plans[0].resident_through, ranges[0].1),
                    &geom,
                )
            })
            .transpose()?;
        let embed_host = if format.is_16bit() {
            let name = format!("{TEXT_PREFIX}.embed_tokens");
            stack.proj16.insert(
                "lm_head".to_string(),
                upload_proj16(&ctx, weights, "lm_head")?,
            );
            Some(weights.host_proj16(&name)?)
        } else {
            let embed = weights.quant_projection(&format!("{TEXT_PREFIX}.embed_tokens"))?;
            let embed = ctx.upload(&embed, None)?;
            let lm = weights.quant_projection("lm_head")?;
            let lm = ctx.upload(&lm, None)?;
            stack
                .proj
                .insert(format!("{TEXT_PREFIX}.embed_tokens"), embed);
            stack.proj.insert("lm_head".to_string(), lm);
            None
        };

        // split=1 scratch for gate/up [intermediate, hidden]; split=4 for
        // down [hidden, intermediate] (2176 words/row, wpt<=3).

        let mut peers = Vec::new();
        let mut boundary_norms = Vec::new();
        if ranges.len() > 1 {
            boundary_norms.push(upload_boundary(&ctx, weights, ranges[1].0)?);
        }
        for (i, pctx) in contexts.enumerate() {
            let pwide = crate::wide::WideKernels::load(&pctx)?;
            let pg16 = Gemv16Kernels::load(&pctx)?;
            let pstack = upload_stack(
                &pctx,
                weights,
                text,
                ranges[i + 1].0..ranges[i + 1].1,
                max_ctx,
                plans[i + 1].resident_through,
            )?;
            let pstreaming = (plans[i + 1].residency == Residency::Streaming && !format.is_16bit())
                .then(|| {
                    StreamState::build(
                        &pctx,
                        weights,
                        text,
                        (plans[i + 1].resident_through, ranges[i + 1].1),
                    )
                })
                .transpose()?;
            let pstreaming16 = (plans[i + 1].residency == Residency::Streaming
                && format.is_16bit())
            .then(|| {
                StreamState16::build(
                    &pctx,
                    weights,
                    text,
                    (plans[i + 1].resident_through, ranges[i + 1].1),
                    &geom,
                )
            })
            .transpose()?;
            let boundary_norm = if i + 2 < ranges.len() {
                upload_boundary(&pctx, weights, ranges[i + 2].0)?
            } else {
                pctx.upload_f32(&weights.f32_named(&format!("{TEXT_PREFIX}.norm.weight"))?)?
            };
            let pbatch = crate::prefill::Batch::new(
                &pctx,
                ordinals[i + 1],
                chosen_mode,
                &crate::prefill::BatchGeom::of(text, chosen_block),
                format,
            )?;

            peers.push(PeerGpu {
                prefill: chosen_mode,
                attn: attention.next().expect("device count checked"),
                batch: pbatch,
                q_out: pctx
                    .stream
                    .alloc_zeros::<f32>(text.num_attention_heads * text.head_dim)
                    .context("q_out")?,
                gate_out: pctx
                    .stream
                    .alloc_zeros::<f32>(text.num_attention_heads * text.head_dim)
                    .context("gate_out")?,
                attn_out: pctx
                    .stream
                    .alloc_zeros::<f32>(text.num_attention_heads * text.head_dim)
                    .context("attn_out")?,
                inner: pctx
                    .stream
                    .alloc_zeros::<f32>(text.intermediate_size)
                    .context("inner")?,
                hidden: pctx
                    .stream
                    .alloc_zeros::<f32>(hidden_size)
                    .context("hidden")?,
                x1: pctx.stream.alloc_zeros::<f32>(hidden_size).context("x1")?,
                pos: pctx.upload_i32(&[0])?,
                rope_pos: pctx.upload_i32(&[0, 0, 0])?,
                pos_block: pctx.stream.alloc_zeros::<i32>(crate::prefill::BLOCK)?,
                rope_pos_block: pctx.stream.alloc_zeros::<i32>(3 * crate::prefill::BLOCK)?,
                boundary_norm,
                streaming: pstreaming,
                streaming16: pstreaming16,
                ctx: pctx,
                wide: pwide,
                g16: pg16,
                proj: pstack.proj,
                proj16: pstack.proj16,
                ln: pstack.ln,
                attn_norms: pstack.attn_norms,
                gdn: pstack.gdn,
                kv_keys: pstack.kv_keys,
                kv_values: pstack.kv_values,
            });
        }

        let batch = crate::prefill::Batch::new(
            &ctx,
            ordinals[0],
            chosen_mode,
            &crate::prefill::BatchGeom::of(text, chosen_block),
            format,
        )?;

        if ctx.context.has_async_alloc() {
            use cudarc::driver::result;
            unsafe {
                result::mem_pool::trim_to(
                    result::device::get_mem_pool(ctx.context.ordinal() as _)?,
                    0,
                )?;
            }
        }

        let mut gpu = Self {
            prefill: chosen_mode,
            prefill_block: chosen_block,
            attn,
            batch,
            final_norm_w: ctx
                .upload_f32(&weights.f32_named(&format!("{TEXT_PREFIX}.norm.weight"))?)?,
            q_out: ctx
                .stream
                .alloc_zeros::<f32>(text.num_attention_heads * text.head_dim)?,
            gate_out: ctx
                .stream
                .alloc_zeros::<f32>(text.num_attention_heads * text.head_dim)?,
            attn_out: ctx
                .stream
                .alloc_zeros::<f32>(text.num_attention_heads * text.head_dim)?,
            inner: ctx.stream.alloc_zeros::<f32>(text.intermediate_size)?,
            hidden: ctx.stream.alloc_zeros::<f32>(hidden_size)?,
            x1: ctx.stream.alloc_zeros::<f32>(hidden_size)?,

            pos: ctx.upload_i32(&[0])?,
            rope_pos: ctx.upload_i32(&[0, 0, 0])?,
            pos_block: ctx.stream.alloc_zeros::<i32>(crate::prefill::BLOCK)?,
            rope_pos_block: ctx.stream.alloc_zeros::<i32>(3 * crate::prefill::BLOCK)?,
            next_token: ctx.upload_i32(&[0])?,
            nt_spare: ctx.upload_i32(&[0])?,
            span_graphs: (0..ranges.len()).map(|_| None).collect(),
            ctx,
            proj: stack.proj,
            proj16: stack.proj16,
            ln: stack.ln,
            attn_norms: stack.attn_norms,
            gdn: stack.gdn,
            kv_keys: stack.kv_keys,
            kv_values: stack.kv_values,
            position: 0,
            prefill_packed_tokens: 0,
            max_ctx,
            config: config.clone(),
            format,
            wide,
            g16,
            embed_host,
            next_host_token: None,
            peers,
            boundary_norms,
            ranges,
            staging: vec![0f32; 2 * hidden_size],
            streaming,
            streaming16,
        };
        if let Some(g) = gpu.gdn.first_mut() {
            let batch = &mut gpu.batch;
            batch.gdn = Some(crate::prefill::GdnPrefill::select(
                &gpu.ctx,
                ordinals[0],
                &crate::prefill::BatchGeom::of(text, gpu.prefill_block),
                g,
                batch,
                text.rms_norm_eps as f32,
                &[],
            )?);
        }
        for i in 0..gpu.peers.len() {
            let (before, remaining) = gpu.peers.split_at_mut(i);
            let previous: Vec<_> = gpu
                .batch
                .gdn
                .iter()
                .chain(before.iter().filter_map(|peer| peer.batch.gdn.as_ref()))
                .collect();
            let peer = &mut remaining[0];
            if let Some(g) = peer.gdn.first_mut() {
                let batch = &mut peer.batch;
                batch.gdn = Some(crate::prefill::GdnPrefill::select(
                    &peer.ctx,
                    ordinals[i + 1],
                    &crate::prefill::BatchGeom::of(text, gpu.prefill_block),
                    g,
                    batch,
                    text.rms_norm_eps as f32,
                    &previous,
                )?);
            }
        }
        Ok(gpu)
    }

    pub(crate) fn probe_state(&mut self, rows: usize) -> Result<ProbeState> {
        ensure!(
            self.peers.is_empty(),
            "speculative calibration state requires one device"
        );
        ensure!(
            rows <= self.max_ctx,
            "probe state needs {rows} KV rows, capacity {}",
            self.max_ctx
        );
        self.ctx.stream.synchronize()?;
        let mut gdn = Vec::new();
        for g in &mut self.gdn {
            let (conv, recurrent) = self.ctx.gdn_state_mut(g);
            gdn.push((
                self.ctx.stream.clone_dtoh(&*conv)?,
                self.ctx.stream.clone_dtoh(&*recurrent)?,
            ));
        }
        let stride = self.config.text_config.num_key_value_heads * self.config.text_config.head_dim;
        let mut kv = Vec::new();
        for (k, v) in self.kv_keys.iter().zip(&self.kv_values) {
            kv.push((
                self.ctx.stream.clone_dtoh(&k.slice(..rows * stride))?,
                self.ctx.stream.clone_dtoh(&v.slice(..rows * stride))?,
            ));
        }
        Ok(ProbeState {
            hidden: self.ctx.stream.clone_dtoh(&self.hidden)?,
            x1: self.ctx.stream.clone_dtoh(&self.x1)?,
            logits: self.ctx.stream.clone_dtoh(self.get("lm_head")?.y_ref())?,
            token: self.ctx.stream.clone_dtoh(&self.next_token)?,
            spare: self.ctx.stream.clone_dtoh(&self.nt_spare)?,
            pos: self.ctx.stream.clone_dtoh(&self.pos)?,
            rope: self.ctx.stream.clone_dtoh(&self.rope_pos)?,
            gdn,
            kv,
            position: self.position,
            host_token: self.next_host_token,
            packed_tokens: self.prefill_packed_tokens,
        })
    }

    pub(crate) fn restore_probe_state(&mut self, state: &ProbeState) -> Result<()> {
        self.ctx.stream.synchronize()?;
        self.ctx
            .stream
            .memcpy_htod(&state.hidden, &mut self.hidden)?;
        self.ctx.stream.memcpy_htod(&state.x1, &mut self.x1)?;
        self.ctx.stream.memcpy_htod(
            &state.logits,
            self.proj
                .get_mut("lm_head")
                .context("probe lm_head missing")?
                .y_mut(),
        )?;
        self.ctx
            .stream
            .memcpy_htod(&state.token, &mut self.next_token)?;
        self.ctx
            .stream
            .memcpy_htod(&state.spare, &mut self.nt_spare)?;
        self.ctx.stream.memcpy_htod(&state.pos, &mut self.pos)?;
        self.ctx
            .stream
            .memcpy_htod(&state.rope, &mut self.rope_pos)?;
        for (g, (conv, recurrent)) in self.gdn.iter_mut().zip(&state.gdn) {
            let (dst_conv, dst_recurrent) = self.ctx.gdn_state_mut(g);
            self.ctx.stream.memcpy_htod(conv, dst_conv)?;
            self.ctx.stream.memcpy_htod(recurrent, dst_recurrent)?;
        }
        for ((k, v), (src_k, src_v)) in self
            .kv_keys
            .iter_mut()
            .zip(&mut self.kv_values)
            .zip(&state.kv)
        {
            self.ctx
                .stream
                .memcpy_htod(src_k, &mut k.slice_mut(..src_k.len()))?;
            self.ctx
                .stream
                .memcpy_htod(src_v, &mut v.slice_mut(..src_v.len()))?;
        }
        self.position = state.position;
        self.next_host_token = state.host_token;
        self.prefill_packed_tokens = state.packed_tokens;
        Ok(())
    }

    pub(crate) fn probe_round_capacity(&self, rounds: usize) -> Result<usize> {
        let remaining = self
            .max_ctx
            .checked_sub(self.position + 2)
            .context("column-2 probe has no KV prefix capacity")?;
        let capacity = (remaining / 2).min(rounds);
        ensure!(
            capacity > 0,
            "column-2 probe has zero replay capacity at position {} / {}",
            self.position,
            self.max_ctx
        );
        Ok(capacity)
    }

    fn capture_group_bodies(&self, device: usize, body: usize, selected: &[usize]) -> Result<()> {
        self.wide
            .capture_body(Some(if device == 0 { body } else { selected[0] }))?;
        for (i, peer) in self.peers.iter().enumerate() {
            peer.wide.capture_body(Some(if device == i + 1 {
                body
            } else {
                selected[i + 1]
            }))?;
        }
        Ok(())
    }

    fn reset_probe_state(&mut self) -> Result<()> {
        self.reset()?;
        self.ctx.stream.memset_zeros(&mut self.hidden)?;
        self.ctx.stream.memset_zeros(&mut self.x1)?;
        for value in self.kv_keys.iter_mut().chain(&mut self.kv_values) {
            self.ctx.stream.memset_zeros(value)?;
        }
        for peer in &mut self.peers {
            peer.ctx.stream.memset_zeros(&mut peer.hidden)?;
            peer.ctx.stream.memset_zeros(&mut peer.x1)?;
            for value in peer.kv_keys.iter_mut().chain(&mut peer.kv_values) {
                peer.ctx.stream.memset_zeros(value)?;
            }
        }
        Ok(())
    }

    pub fn calibrate_groups(&mut self) -> Result<()> {
        let devices = self.peers.len() + 1;
        let mut selected = vec![0; devices];
        for device in 0..devices {
            let ordinal = if device == 0 {
                self.ctx.context.ordinal()
            } else {
                self.peers[device - 1].ctx.context.ordinal()
            };
            let stream = self.ctx.stream.clone();
            let capacity = self.max_ctx;
            let started = std::time::Instant::now();
            let mut programs = [Vec::new(), Vec::new()];
            for (body, program) in programs.iter_mut().enumerate() {
                self.capture_group_bodies(device, body, &selected)?;
                self.reset_probe_state()?;
                self.span_graphs = (0..devices).map(|_| None).collect();
                self.step_with_graphs(true)?;
                self.ctx
                    .stream
                    .synchronize()
                    .context("decode candidate capture fence")?;
                *program =
                    std::mem::replace(&mut self.span_graphs, (0..devices).map(|_| None).collect());
            }
            let capture_ms = started.elapsed().as_secs_f64() * 1000.0;
            let programs = std::cell::RefCell::new(programs);
            let current = std::cell::Cell::new(usize::MAX);
            let topology = format!(
                "whole decode; device ranges {:?}; streaming {}",
                self.ranges,
                self.is_streaming()
            );
            let model = std::cell::RefCell::new(&mut *self);
            let choice = ff_edge0::gpu::probe_decode(
                &stream,
                ff_edge0::gpu::DecodeProgram {
                    device: ordinal,
                    cols: 1,
                    capacity,
                    capture_ms,
                    state_bytes: 0,
                    seed_token: 0,
                    topology,
                },
                |body| {
                    let mut model = model.borrow_mut();
                    let old = current.get();
                    if old < 2 {
                        programs.borrow_mut()[old] = std::mem::take(&mut model.span_graphs);
                    }
                    model.span_graphs = std::mem::take(&mut programs.borrow_mut()[body]);
                    current.set(body);
                    model.capture_group_bodies(device, body, &selected)?;
                    model.reset_probe_state()
                },
                |_, repeats| {
                    let mut model = model.borrow_mut();
                    for _ in 0..repeats {
                        model.step_with_graphs(true)?;
                    }
                    Ok(())
                },
            )?;
            let mut model = model.borrow_mut();
            let body = if device == 0 {
                model.wide.bodies.bind(choice)
            } else {
                model.peers[device - 1].wide.bodies.bind(choice)
            };
            selected[device] = body;
            let old = current.get();
            if old != body {
                programs.borrow_mut()[old] = std::mem::take(&mut model.span_graphs);
                model.span_graphs = std::mem::take(&mut programs.borrow_mut()[body]);
            }
            model.reset_probe_state()?;
            model.wide.capture_body(None)?;
            for peer in &model.peers {
                peer.wide.capture_body(None)?;
            }
        }
        self.span_graphs = (0..devices).map(|_| None).collect();
        Ok(())
    }

    pub fn group_key(
        &self,
        device: usize,
        cols: u32,
        binary: &ff_core::identity::BinaryIdentity,
        rounds: Option<usize>,
    ) -> Result<serde_json::Value> {
        let (ctx, wide, stream, range) = if device == 0 {
            (
                &self.ctx,
                &self.wide,
                self.streaming.as_ref(),
                self.ranges[0],
            )
        } else {
            let peer = self
                .peers
                .get(device - 1)
                .context("group4 device missing")?;
            (
                &peer.ctx,
                &peer.wide,
                peer.streaming.as_ref(),
                self.ranges[device],
            )
        };
        let class = match stream {
            None => "resident_graph",
            Some(value) if value.range.0 == range.0 => "fully_streamed",
            Some(_) => "streamed",
        };
        let cuda = candle_core::Device::new_cuda(ctx.context.ordinal())?;
        let fingerprint = ff_core::probe::HardwareFingerprint::collect(&cuda)?;
        let text = &self.config.text_config;
        let geometry = serde_json::json!({"layers":text.num_hidden_layers,"hidden":text.hidden_size,"intermediate":text.intermediate_size,"heads":text.num_attention_heads,"kv_heads":text.num_key_value_heads,"head_dim":text.head_dim,"vocab":text.vocab_size,"gdn":[text.linear_num_key_heads,text.linear_num_value_heads,text.linear_key_head_dim,text.linear_value_head_dim,text.linear_conv_kernel_dim],"layers_kind":(0..text.num_hidden_layers).map(|l| text.layer_kind(l)==crate::config::LayerKind::LinearAttention).collect::<Vec<_>>()});
        Ok(
            serde_json::json!({"adapter":"qwen35","device":ctx.context.ordinal(),"cols":cols,"geometry":geometry,"class":class,"settings":{"speculative":cols==2,"max_context":self.max_ctx,"rounds":rounds,"ring_slots":stream.map(|s|s.slots.len()),"graphs":graphs_enabled()?,"devices":std::iter::once(self.ctx.context.ordinal()).chain(self.peers.iter().map(|p|p.ctx.context.ordinal())).collect::<Vec<_>>()},"fingerprint":fingerprint,"binary":{"schema_version":binary.schema_version,"package_name":binary.package_name,"package_version":binary.package_version,"compiled_features":binary.compiled_features},"image":wide.bodies.image()}),
        )
    }

    fn group_split(&self) -> Vec<(usize, usize, usize, usize)> {
        self.ranges
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                (
                    if i == 0 {
                        self.ctx.context.ordinal()
                    } else {
                        self.peers[i - 1].ctx.context.ordinal()
                    },
                    a,
                    if i == 0 {
                        self.streaming.as_ref().map_or(b, |s| s.range.0)
                    } else {
                        self.peers[i - 1]
                            .streaming
                            .as_ref()
                            .map_or(b, |s| s.range.0)
                    },
                    b,
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn bind_groups(
        &mut self,
        records: &[ff_core::probe::DecodeChoice],
        binary: &ff_core::identity::BinaryIdentity,
    ) -> Result<()> {
        if self.format.is_16bit() {
            return Ok(());
        }
        for device in 0..self.peers.len() + 1 {
            let key = self.group_key(device, 1, binary, None)?;
            eprintln!(
                "group4 runtime split (device, start, resident end, end): {:?}",
                self.group_split()
            );
            let bodies = if device == 0 {
                &mut self.wide.bodies
            } else {
                &mut self.peers[device - 1].wide.bodies
            };
            bodies.apply(&key, records)?;
        }
        Ok(())
    }

    pub fn bind_round(
        &mut self,
        records: &[ff_core::probe::DecodeChoice],
        binary: &ff_core::identity::BinaryIdentity,
        rounds: usize,
    ) -> Result<()> {
        let key = self.group_key(0, 2, binary, Some(rounds))?;
        self.wide.bodies.apply(&key, records)
    }

    pub fn calibration_records(
        &self,
        binary: &ff_core::identity::BinaryIdentity,
        rounds: Option<usize>,
    ) -> Result<Vec<ff_core::probe::DecodeChoice>> {
        let mut records = self.group_choices();
        let split = self.group_split();
        for record in &mut records {
            let device = if record.device == self.ctx.context.ordinal() {
                0
            } else {
                self.peers
                    .iter()
                    .position(|p| p.ctx.context.ordinal() == record.device)
                    .context("calibration peer missing")?
                    + 1
            };
            record.set_key(
                &self.group_key(
                    device,
                    record.cols,
                    binary,
                    if record.cols == 2 { rounds } else { None },
                )?,
                split.clone(),
            )?;
        }
        Ok(records)
    }

    pub fn forced_group(&self) -> bool {
        self.wide.bodies.forced()
    }

    pub fn group_choices(&self) -> Vec<ff_core::probe::DecodeChoice> {
        self.wide
            .choices()
            .into_iter()
            .chain(self.peers.iter().flat_map(|peer| peer.wide.choices()))
            .collect()
    }

    /// The layer range each device ordinal hosts.
    pub fn layer_ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    /// True when any device streams part of its layer range.
    pub fn is_streaming(&self) -> bool {
        self.streaming.is_some() || self.peers.iter().any(|peer| peer.streaming.is_some())
    }

    fn get(&self, name: &str) -> Result<&GpuQuant> {
        self.proj
            .get(name)
            .with_context(|| format!("{name} not resident"))
    }

    /// One decode/prefill step for the token in `next_token` (device-side).
    /// No syncs; the only per-token host read is `read_token`. Norms use
    /// the fused add+norm pairing: each residual write also produces the
    /// next normed input (the edge0 pairing; loop-top norm only for
    /// layer 0).
    pub fn step(&mut self) -> Result<()> {
        let graphs = graphs_enabled()?;
        self.step_with_graphs(graphs)
    }

    fn step_with_graphs(&mut self, graphs: bool) -> Result<()> {
        anyhow::ensure!(
            self.position < self.max_ctx,
            "position {} reached max_ctx {} (KV cache capacity)",
            self.position,
            self.max_ctx
        );
        // Graph capture is derived, not opted in: every device's resident
        // span is captured (device 0 as one whole step when nothing
        // streams), the streamed tails and peer hops stay eager.
        // QWEN35_GRAPH=0 disables.
        if !graphs {
            self.step_inner()?;
            return Ok(());
        }
        // The embed runs eager and is never captured: the 16-bit host
        // gather produces a local row whose captured memcpy would read
        // freed memory on replay.
        self.embed_from_next_token()?;
        if self.streaming.is_none() && self.streaming16.is_none() && self.peers.is_empty() {
            if let Some(g) = &self.span_graphs[0] {
                g.0.launch().context("decode graph replay")?;
                self.position += 1;
                return Ok(());
            }
            let stream = self.ctx.stream.clone();
            ff_edge0::gpu::begin_decode_capture(&stream).context("decode capture begin")?;
            if let Err(e) = self.step_layers() {
                let _ = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                return Err(e.context("decode capture enqueue"));
            }
            let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .context("decode capture end")?.context("decode capture produced no graph")?;
            graph.launch().context("decode captured step launch")?;
            self.span_graphs[0] = Some(ff_edge0::gpu::DecodeGraph(graph));
            return Ok(());
        }
        self.step_inner_spans()
    }

    /// One decode step through per-device resident-span graphs; the
    /// streamed tails and peer boundary hops run eager.
    fn step_inner_spans(&mut self) -> Result<()> {
        let text = self.config.text_config.clone();
        let eps = text.rms_norm_eps as f32;
        let n = text.hidden_size;
        self.ctx
            .glue_rmsnorm_zc(&self.hidden, &self.ln[0][0], &self.x1, n, eps)?;
        self.run_device_stack(0, &text)?;
        for i in 0..self.peers.len() {
            if i == 0 {
                self.ctx
                    .stream
                    .memcpy_dtoh(&self.hidden, &mut self.staging[..n])?;
                self.ctx
                    .stream
                    .memcpy_dtoh(&self.x1, &mut self.staging[n..])?;
            } else {
                let prev = &self.peers[i - 1];
                prev.ctx
                    .stream
                    .memcpy_dtoh(&prev.hidden, &mut self.staging[..n])?;
                prev.ctx
                    .stream
                    .memcpy_dtoh(&prev.x1, &mut self.staging[n..])?;
            }
            {
                let peer = &mut self.peers[i];
                peer.ctx
                    .stream
                    .memcpy_htod(&self.staging[..n], &mut peer.hidden)?;
                peer.ctx
                    .stream
                    .memcpy_htod(&self.staging[n..], &mut peer.x1)?;
                self.run_device_stack(i + 1, &text)?;
            }
        }
        if let Some(last) = self.peers.last() {
            last.ctx
                .stream
                .memcpy_dtoh(&last.x1, &mut self.staging[..n])?;
            self.ctx
                .stream
                .memcpy_htod(&self.staging[..n], &mut self.x1)?;
        }
        let lm_y: &CudaSliceF;
        let lm_out: usize;
        if self.format.is_16bit() {
            let lm = self.proj16.get("lm_head").context("lm_head resident")?;
            let seg = lm.seg();
            let e = [seg.empty_like(), seg.empty_like(), seg.empty_like()];
            self.g16.group(
                &self.ctx,
                self.format,
                [&seg, &e[0], &e[1], &e[2]],
                &self.x1,
                lm.in_dim,
            )?;
            lm_y = &lm.y;
            lm_out = lm.out_dim;
        } else {
            let lm = self.proj.get("lm_head").context("lm_head resident")?;
            let segs = [
                lm.group_seg(),
                lm.empty_seg_like(),
                lm.empty_seg_like(),
                lm.empty_seg_like(),
            ];
            self.wide.group(
                &self.ctx,
                &segs,
                [lm.y_ref(); 4],
                &GroupPair {
                    x: &self.x1,
                    xb: &self.x1,
                },
                lm.in_dim,
                1,
            )?;
            lm_y = lm.y_ref();
            lm_out = lm.out_dim;
        }
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        self.ctx.glue_argmax(lm_y, &mut self.nt_spare, lm_out)?;
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        self.ctx.glue_inc(&mut self.pos)?;
        self.ctx.glue_inc3(&mut self.rope_pos)?;
        for peer in &mut self.peers {
            peer.ctx.glue_inc(&mut peer.pos)?;
            peer.ctx.glue_inc3(&mut peer.rope_pos)?;
        }
        self.position += 1;
        Ok(())
    }

    /// Run one device's layer stack: the resident prefix as a captured
    /// graph, the streamed tail eager.
    fn run_device_stack(&mut self, device: usize, text: &TextConfig) -> Result<()> {
        let (start, end) = if device == 0 {
            self.ranges[0]
        } else {
            self.ranges[device]
        };
        let (through4, through16) = if device == 0 {
            (
                self.streaming.as_ref().map_or(end, |s| s.range.0),
                self.streaming16.as_ref().map_or(end, |s| s.range.0),
            )
        } else {
            let peer = &self.peers[device - 1];
            (
                peer.streaming.as_ref().map_or(end, |s| s.range.0),
                peer.streaming16.as_ref().map_or(end, |s| s.range.0),
            )
        };
        let graph_end = through4.min(through16).min(end);
        let gdn_done = (start..graph_end)
            .filter(|&l| text.layer_kind(l) == crate::config::LayerKind::LinearAttention)
            .count();
        let kv_done = (graph_end - start) - gdn_done;
        let prefix_origin = SpanOrigin {
            local_base: start,
            range_end: end,
            gdn_index_start: 0,
            kv_index_start: 0,
        };
        let tail_origin = SpanOrigin {
            local_base: start,
            range_end: end,
            gdn_index_start: gdn_done,
            kv_index_start: kv_done,
        };
        if graph_end > start {
            let graph = self.span_graphs[device].take();
            match graph {
                Some(g) => {
                    g.0.launch().context("resident span replay")?;
                    self.span_graphs[device] = Some(g);
                }
                None => {
                    let stream = if device == 0 {
                        self.ctx.stream.clone()
                    } else {
                        self.peers[device - 1].ctx.stream.clone()
                    };
                    ff_edge0::gpu::begin_decode_capture(&stream)
                        .with_context(|| format!("device {device} resident span capture begin"))?;
                    if let Err(e) =
                        self.run_stack_range(device, start..graph_end, text, prefix_origin)
                    {
                        let _ = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                        return Err(e.context(format!("device {device} resident span enqueue")));
                    }
                    let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                        .with_context(||format!("device {device} resident span capture end"))?
                        .with_context(||format!("device {device} resident span produced no graph"))?;
                    graph
                        .launch()
                        .with_context(|| format!("device {device} resident span launch"))?;
                    self.span_graphs[device] = Some(ff_edge0::gpu::DecodeGraph(graph));
                }
            }
        }
        if graph_end < end {
            self.run_stack_range(device, graph_end..end, text, tail_origin)?;
        }
        Ok(())
    }

    /// Build one device's StackView and run a layer range of it.
    fn run_stack_range(
        &mut self,
        device: usize,
        layers: std::ops::Range<usize>,
        text: &TextConfig,
        origin: SpanOrigin,
    ) -> Result<()> {
        if device == 0 {
            run_stack(
                &mut StackView {
                    attn: self.attn.as_ref(),
                    ctx: &self.ctx,
                    wide: &self.wide,
                    g16: &self.g16,
                    format: self.format,
                    proj: &self.proj,
                    proj16: &self.proj16,
                    ln: &self.ln,
                    attn_norms: &self.attn_norms,
                    gdn: &self.gdn,
                    kv_keys: &self.kv_keys,
                    kv_values: &self.kv_values,
                    q_out: &self.q_out,
                    gate_out: &self.gate_out,
                    attn_out: &self.attn_out,
                    inner: &self.inner,
                    pos: &mut self.pos,
                    rope_pos: &mut self.rope_pos,
                    hidden: &self.hidden,
                    x1: &self.x1,
                    boundary_norm: self.boundary_norms.first().unwrap_or(&self.final_norm_w),
                    stream: self.streaming.as_mut(),
                    stream16: self.streaming16.as_mut(),
                },
                layers,
                text,
                origin,
            )
        } else {
            let peer = &mut self.peers[device - 1];
            run_stack(
                &mut StackView {
                    attn: peer.attn.as_ref(),
                    ctx: &peer.ctx,
                    wide: &peer.wide,
                    g16: &peer.g16,
                    format: self.format,
                    proj: &peer.proj,
                    proj16: &peer.proj16,
                    ln: &peer.ln,
                    attn_norms: &peer.attn_norms,
                    gdn: &peer.gdn,
                    kv_keys: &peer.kv_keys,
                    kv_values: &peer.kv_values,
                    q_out: &peer.q_out,
                    gate_out: &peer.gate_out,
                    attn_out: &peer.attn_out,
                    inner: &peer.inner,
                    pos: &mut peer.pos,
                    rope_pos: &mut peer.rope_pos,
                    hidden: &peer.hidden,
                    x1: &peer.x1,
                    boundary_norm: &peer.boundary_norm,
                    stream: peer.streaming.as_mut(),
                    stream16: peer.streaming16.as_mut(),
                },
                layers,
                text,
                origin,
            )
        }
    }

    fn step_inner(&mut self) -> Result<()> {
        self.embed_from_next_token()?;
        self.step_layers()
    }

    fn embed_from_next_token(&mut self) -> Result<()> {
        if let Some(embed) = &self.embed_host {
            let token = self
                .next_host_token
                .context("16-bit embed gathered before any token was read")?;
            let row = embed.row_f32(self.format, token as usize)?;
            self.ctx.stream.memcpy_htod(&row, &mut self.hidden)?;
            return Ok(());
        }
        self.ctx.glue_embed_row(
            self.get(&format!("{TEXT_PREFIX}.embed_tokens"))?,
            &self.next_token,
            &self.hidden,
        )
    }

    /// Layer stack + lm_head + argmax + counter bumps; reads `hidden`
    /// (embed output or spliced vision row) and `rope_pos`.
    fn step_layers(&mut self) -> Result<()> {
        let text = self.config.text_config.clone();
        let eps = text.rms_norm_eps as f32;
        let n = text.hidden_size;
        self.ctx
            .glue_rmsnorm_zc(&self.hidden, &self.ln[0][0], &self.x1, n, eps)?;
        run_stack(
            &mut StackView {
                attn: self.attn.as_ref(),
                ctx: &self.ctx,
                wide: &self.wide,
                g16: &self.g16,
                format: self.format,
                proj: &self.proj,
                proj16: &self.proj16,
                ln: &self.ln,
                attn_norms: &self.attn_norms,
                gdn: &self.gdn,
                kv_keys: &self.kv_keys,
                kv_values: &self.kv_values,
                q_out: &self.q_out,
                gate_out: &self.gate_out,
                attn_out: &self.attn_out,
                inner: &self.inner,
                pos: &mut self.pos,
                rope_pos: &mut self.rope_pos,
                hidden: &self.hidden,
                x1: &self.x1,
                boundary_norm: self.boundary_norms.first().unwrap_or(&self.final_norm_w),
                stream: self.streaming.as_mut(),
                stream16: self.streaming16.as_mut(),
            },
            self.ranges[0].0..self.ranges[0].1,
            &text,
            SpanOrigin {
                local_base: self.ranges[0].0,
                range_end: self.ranges[0].1,
                gdn_index_start: 0,
                kv_index_start: 0,
            },
        )?;
        for i in 0..self.peers.len() {
            if i == 0 {
                self.ctx
                    .stream
                    .memcpy_dtoh(&self.hidden, &mut self.staging[..n])?;
                self.ctx
                    .stream
                    .memcpy_dtoh(&self.x1, &mut self.staging[n..])?;
            } else {
                let prev = &self.peers[i - 1];
                prev.ctx
                    .stream
                    .memcpy_dtoh(&prev.hidden, &mut self.staging[..n])?;
                prev.ctx
                    .stream
                    .memcpy_dtoh(&prev.x1, &mut self.staging[n..])?;
            }
            let (start, end) = self.ranges[i + 1];
            {
                let peer = &mut self.peers[i];
                peer.ctx
                    .stream
                    .memcpy_htod(&self.staging[..n], &mut peer.hidden)?;
                peer.ctx
                    .stream
                    .memcpy_htod(&self.staging[n..], &mut peer.x1)?;
                run_stack(
                    &mut StackView {
                        attn: peer.attn.as_ref(),
                        ctx: &peer.ctx,
                        wide: &peer.wide,
                        g16: &peer.g16,
                        format: self.format,
                        proj: &peer.proj,
                        proj16: &peer.proj16,
                        ln: &peer.ln,
                        attn_norms: &peer.attn_norms,
                        gdn: &peer.gdn,
                        kv_keys: &peer.kv_keys,
                        kv_values: &peer.kv_values,
                        q_out: &peer.q_out,
                        gate_out: &peer.gate_out,
                        attn_out: &peer.attn_out,
                        inner: &peer.inner,
                        pos: &mut peer.pos,
                        rope_pos: &mut peer.rope_pos,
                        hidden: &peer.hidden,
                        x1: &peer.x1,
                        boundary_norm: &peer.boundary_norm,
                        stream: peer.streaming.as_mut(),
                        stream16: peer.streaming16.as_mut(),
                    },
                    start..end,
                    &text,
                    SpanOrigin {
                        local_base: start,
                        range_end: end,
                        gdn_index_start: 0,
                        kv_index_start: 0,
                    },
                )?;
            }
        }
        if let Some(last) = self.peers.last() {
            last.ctx
                .stream
                .memcpy_dtoh(&last.x1, &mut self.staging[..n])?;
            self.ctx
                .stream
                .memcpy_htod(&self.staging[..n], &mut self.x1)?;
        }
        // x1 already holds the final-normed hidden.
        // x1 already holds the final-normed hidden.
        let lm_y: &CudaSliceF;
        let lm_out: usize;
        if self.format.is_16bit() {
            let lm = self.proj16.get("lm_head").context("lm_head resident")?;
            let seg = lm.seg();
            let e = [seg.empty_like(), seg.empty_like(), seg.empty_like()];
            self.g16.group(
                &self.ctx,
                self.format,
                [&seg, &e[0], &e[1], &e[2]],
                &self.x1,
                lm.in_dim,
            )?;
            lm_y = &lm.y;
            lm_out = lm.out_dim;
        } else {
            let lm = self.proj.get("lm_head").context("lm_head resident")?;
            let segs = [
                lm.group_seg(),
                lm.empty_seg_like(),
                lm.empty_seg_like(),
                lm.empty_seg_like(),
            ];
            self.wide.group(
                &self.ctx,
                &segs,
                [lm.y_ref(); 4],
                &GroupPair {
                    x: &self.x1,
                    xb: &self.x1,
                },
                lm.in_dim,
                1,
            )?;
            lm_y = lm.y_ref();
            lm_out = lm.out_dim;
        }
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        self.ctx.glue_argmax(lm_y, &mut self.nt_spare, lm_out)?;
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        self.ctx.glue_inc(&mut self.pos)?;
        self.ctx.glue_inc3(&mut self.rope_pos)?;
        for peer in &mut self.peers {
            peer.ctx.glue_inc(&mut peer.pos)?;
            peer.ctx.glue_inc3(&mut peer.rope_pos)?;
        }
        self.position += 1;
        Ok(())
    }

    /// Feed a prompt token (text-only: rope position = KV index).
    pub fn push_token(&mut self, token: u32) -> Result<()> {
        let p = self.position as i32;
        self.push_token_at(token, [p, p, p])
    }

    /// Prefill one text token at an explicit mrope position.
    pub fn push_token_at(&mut self, token: u32, pos3: [i32; 3]) -> Result<()> {
        self.next_host_token = Some(token);
        self.ctx
            .stream
            .memcpy_htod(&[token as i32], &mut self.next_token)?;
        self.ctx.stream.memcpy_htod(&pos3, &mut self.rope_pos)?;
        for peer in &mut self.peers {
            peer.ctx.stream.memcpy_htod(&pos3, &mut peer.rope_pos)?;
        }
        self.step()
    }

    /// Text-only batched prefill at consecutive text positions.
    pub fn push_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        let base = self.position;
        let pos3: Vec<[i32; 3]> = (base..base + tokens.len())
            .map(|p| [p as i32, p as i32, p as i32])
            .collect();
        self.push_tokens_at(tokens, &pos3)
    }

    /// Batched prefill: blocks of tokens cross the stack layer-major, so
    /// a streamed layer's weights flow once per block, not once per
    /// token. Text only; vision rows keep the per-token path.
    pub fn push_tokens_at(&mut self, tokens: &[u32], pos3: &[[i32; 3]]) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prefill block");
        ensure!(
            pos3.len() == tokens.len(),
            "{} positions for {} tokens",
            pos3.len(),
            tokens.len()
        );
        ensure!(
            self.position + tokens.len() <= self.max_ctx,
            "position {} + {} tokens exceeds max_ctx {} (KV cache capacity)",
            self.position,
            tokens.len(),
            self.max_ctx
        );
        let block = self.prefill_block;
        for (chunk, positions) in tokens.chunks(block).zip(pos3.chunks(block)) {
            self.push_block(chunk, positions)?;
        }
        Ok(())
    }

    /// One block through the whole stack, layer-major on every device;
    /// the tail mirrors step_layers (lm_head, argmax, counter bumps).
    fn push_block(&mut self, tokens: &[u32], pos3: &[[i32; 3]]) -> Result<()> {
        let text = self.config.text_config.clone();
        let n = text.hidden_size;
        let eps = text.rms_norm_eps as f32;
        let base = self.position;
        ensure!(
            tokens.len() <= self.pos_block.len(),
            "prefill block {} exceeds the {}-token position planes",
            tokens.len(),
            self.pos_block.len()
        );
        let pos_vec: Vec<i32> = (base..base + tokens.len()).map(|p| p as i32).collect();
        let rope_flat: Vec<i32> = pos3.iter().flatten().copied().collect();
        self.ctx
            .stream
            .memcpy_htod(&pos_vec, &mut self.pos_block.slice_mut(..tokens.len()))?;
        self.ctx.stream.memcpy_htod(
            &rope_flat,
            &mut self.rope_pos_block.slice_mut(..3 * tokens.len()),
        )?;
        let embed_name = format!("{TEXT_PREFIX}.embed_tokens");
        {
            let batch = &mut self.batch;
            for (t, &token) in tokens.iter().enumerate() {
                if let Some(embed) = &self.embed_host {
                    let row = embed.row_f32(self.format, token as usize)?;
                    self.ctx
                        .stream
                        .memcpy_htod(&row, &mut batch.hidden.slice_mut(t * n..(t + 1) * n))?;
                } else {
                    let embed = self
                        .proj
                        .get(&embed_name)
                        .context("embed_tokens resident")?;
                    self.ctx
                        .stream
                        .memcpy_htod(&[token as i32], &mut self.next_token)?;
                    self.ctx.glue_embed_row(
                        embed,
                        &self.next_token,
                        &batch.hidden.slice(t * n..(t + 1) * n),
                    )?;
                }
            }
            let batch = &self.batch;
            batch.rows.rmsnorm(
                &self.ctx,
                crate::prefill::NormRows {
                    x: &batch.hidden,
                    w: &self.ln[0][0],
                    out: &batch.x1,
                    tokens: tokens.len(),
                    n,
                    eps,
                },
            )?;
        }
        let (start0, end0) = self.ranges[0];
        run_block(
            &mut StackView {
                attn: self.attn.as_ref(),
                ctx: &self.ctx,
                wide: &self.wide,
                g16: &self.g16,
                format: self.format,
                proj: &self.proj,
                proj16: &self.proj16,
                ln: &self.ln,
                attn_norms: &self.attn_norms,
                gdn: &self.gdn,
                kv_keys: &self.kv_keys,
                kv_values: &self.kv_values,
                q_out: &self.q_out,
                gate_out: &self.gate_out,
                attn_out: &self.attn_out,
                inner: &self.inner,
                pos: &mut self.pos_block,
                rope_pos: &mut self.rope_pos_block,
                hidden: &self.hidden,
                x1: &self.x1,
                boundary_norm: self.boundary_norms.first().unwrap_or(&self.final_norm_w),
                stream: self.streaming.as_mut(),
                stream16: self.streaming16.as_mut(),
            },
            &text,
            BlockArgs {
                layers: start0..end0,
                tokens: tokens.len(),
                base,
                batch: &mut self.batch,
            },
        )?;
        for i in 0..self.peers.len() {
            let (start, end) = self.ranges[i + 1];
            let mut hop = vec![0f32; 2 * n * tokens.len()];

            {
                let (src, slab_h, slab_x) = if i == 0 {
                    let batch = &self.batch;
                    (&self.ctx, &batch.hidden, &batch.x1)
                } else {
                    let prev = &self.peers[i - 1];
                    let batch = &prev.batch;
                    (&prev.ctx, &batch.hidden, &batch.x1)
                };
                src.stream.memcpy_dtoh(
                    &slab_h.slice(..n * tokens.len()),
                    &mut hop[..n * tokens.len()],
                )?;
                src.stream.memcpy_dtoh(
                    &slab_x.slice(..n * tokens.len()),
                    &mut hop[n * tokens.len()..],
                )?;
            }
            let peer = &mut self.peers[i];
            peer.ctx
                .stream
                .memcpy_htod(&pos_vec, &mut peer.pos_block.slice_mut(..tokens.len()))?;
            peer.ctx.stream.memcpy_htod(
                &rope_flat,
                &mut peer.rope_pos_block.slice_mut(..3 * tokens.len()),
            )?;
            let pbatch = &mut peer.batch;
            peer.ctx
                .stream
                .memcpy_htod(&hop[..n * tokens.len()], &mut pbatch.hidden)?;
            peer.ctx
                .stream
                .memcpy_htod(&hop[n * tokens.len()..], &mut pbatch.x1)?;
            run_block(
                &mut StackView {
                    attn: peer.attn.as_ref(),
                    ctx: &peer.ctx,
                    wide: &peer.wide,
                    g16: &peer.g16,
                    format: self.format,
                    proj: &peer.proj,
                    proj16: &peer.proj16,
                    ln: &peer.ln,
                    attn_norms: &peer.attn_norms,
                    gdn: &peer.gdn,
                    kv_keys: &peer.kv_keys,
                    kv_values: &peer.kv_values,
                    q_out: &peer.q_out,
                    gate_out: &peer.gate_out,
                    attn_out: &peer.attn_out,
                    inner: &peer.inner,
                    pos: &mut peer.pos_block,
                    rope_pos: &mut peer.rope_pos_block,
                    hidden: &peer.hidden,
                    x1: &peer.x1,
                    boundary_norm: &peer.boundary_norm,
                    stream: peer.streaming.as_mut(),
                    stream16: peer.streaming16.as_mut(),
                },
                &text,
                BlockArgs {
                    layers: start..end,
                    tokens: tokens.len(),
                    base,
                    batch: &mut peer.batch,
                },
            )?;
        }

        let (last_ctx, last_hidden, last_x1) = if self.peers.is_empty() {
            {
                let batch = &self.batch;
                (
                    &self.ctx,
                    batch.hidden.slice((tokens.len() - 1) * n..tokens.len() * n),
                    batch.x1.slice((tokens.len() - 1) * n..tokens.len() * n),
                )
            }
        } else {
            let peer = self.peers.last().expect("checked non-empty");
            {
                let batch = &peer.batch;
                (
                    &peer.ctx,
                    batch.hidden.slice((tokens.len() - 1) * n..tokens.len() * n),
                    batch.x1.slice((tokens.len() - 1) * n..tokens.len() * n),
                )
            }
        };
        // The final token's hidden state feeds the next step's and the
        // speculative draft's reads of `self.hidden`; prefill is the only
        // writer, so hop it here the same way x1 hops.
        last_ctx
            .stream
            .memcpy_dtoh(&last_hidden, &mut self.staging[..n])?;
        self.ctx
            .stream
            .memcpy_htod(&self.staging[..n], &mut self.hidden)?;
        last_ctx
            .stream
            .memcpy_dtoh(&last_x1, &mut self.staging[..n])?;
        self.ctx
            .stream
            .memcpy_htod(&self.staging[..n], &mut self.x1)?;
        // x1 already holds the final-normed hidden.
        let lm_y: &CudaSliceF;
        let lm_out: usize;
        if self.format.is_16bit() {
            let lm = self.proj16.get("lm_head").context("lm_head resident")?;
            let seg = lm.seg();
            let e = [seg.empty_like(), seg.empty_like(), seg.empty_like()];
            self.g16.group(
                &self.ctx,
                self.format,
                [&seg, &e[0], &e[1], &e[2]],
                &self.x1,
                lm.in_dim,
            )?;
            lm_y = &lm.y;
            lm_out = lm.out_dim;
        } else {
            let lm = self.proj.get("lm_head").context("lm_head resident")?;
            let segs = [
                lm.group_seg(),
                lm.empty_seg_like(),
                lm.empty_seg_like(),
                lm.empty_seg_like(),
            ];
            self.wide.group(
                &self.ctx,
                &segs,
                [lm.y_ref(); 4],
                &GroupPair {
                    x: &self.x1,
                    xb: &self.x1,
                },
                lm.in_dim,
                1,
            )?;
            lm_y = lm.y_ref();
            lm_out = lm.out_dim;
        }
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        self.ctx.glue_argmax(lm_y, &mut self.nt_spare, lm_out)?;
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        // The position planes carry per-token values during the block; the
        // scalar counters must land past the block's end for the next step.
        let next_pos = (base + tokens.len()) as i32;
        let mut next_rope = *pos3.last().expect("checked non-empty");
        for component in &mut next_rope {
            *component += 1;
        }
        self.ctx.stream.memcpy_htod(&[next_pos], &mut self.pos)?;
        self.ctx
            .stream
            .memcpy_htod(&next_rope, &mut self.rope_pos)?;
        for peer in &mut self.peers {
            peer.ctx.stream.memcpy_htod(&[next_pos], &mut peer.pos)?;
            peer.ctx
                .stream
                .memcpy_htod(&next_rope, &mut peer.rope_pos)?;
        }
        self.position += tokens.len();

        let modes = || std::iter::once(self.prefill).chain(self.peers.iter().map(|p| p.prefill));
        if modes().any(|m| m != Mode::Gemv) {
            self.prefill_packed_tokens += tokens.len();
        }

        Ok(())
    }

    /// Prefill one merged vision row into `hidden`. Eager — never
    /// graph-captured (the decode graph is splice-free).
    pub fn push_vision_row(&mut self, row: &[f32], pos3: [i32; 3]) -> Result<()> {
        anyhow::ensure!(
            row.len() == self.hidden.len(),
            "vision row {} != hidden {}",
            row.len(),
            self.hidden.len()
        );
        anyhow::ensure!(
            self.position < self.max_ctx,
            "position {} reached max_ctx {} (KV cache capacity)",
            self.position,
            self.max_ctx
        );
        self.ctx.stream.memcpy_htod(row, &mut self.hidden)?;
        self.ctx.stream.memcpy_htod(&pos3, &mut self.rope_pos)?;
        for peer in &mut self.peers {
            peer.ctx.stream.memcpy_htod(&pos3, &mut peer.rope_pos)?;
        }
        self.step_layers()
    }

    /// Reset target state; recreate any speculative decoder afterward.
    pub fn reset(&mut self) -> Result<()> {
        reset_stack(
            &self.ctx,
            &mut self.gdn,
            &mut self.pos,
            &mut self.rope_pos,
            self.streaming.as_ref(),
            self.streaming16.as_ref(),
        )?;
        for peer in &mut self.peers {
            reset_stack(
                &peer.ctx,
                &mut peer.gdn,
                &mut peer.pos,
                &mut peer.rope_pos,
                peer.streaming.as_ref(),
                peer.streaming16.as_ref(),
            )?;
        }
        self.ctx.stream.memset_zeros(&mut self.next_token)?;
        self.ctx.stream.memset_zeros(&mut self.nt_spare)?;
        for state in self
            .streaming16
            .iter()
            .chain(self.peers.iter().filter_map(|p| p.streaming16.as_ref()))
        {
            *state.stats.lock().expect("stream stats poisoned") = Stream16Stats::default();
        }
        self.position = 0;
        self.prefill_packed_tokens = 0;
        Ok(())
    }

    /// Decode from the first token returned by prefill.
    pub fn decode(&mut self, first: u32, max_tokens: usize) -> Result<Vec<u32>> {
        ensure!(max_tokens > 0, "decode budget must be positive");
        self.next_host_token = Some(first);
        let mut tokens = vec![first];
        while tokens.len() < max_tokens
            && !self
                .config
                .text_config
                .eos_token_id
                .contains(tokens.last().unwrap())
        {
            self.step()?;
            tokens.push(self.read_token()?);
        }
        Ok(tokens)
    }

    /// The generated token id (the one sync per token).
    pub fn read_token(&mut self) -> Result<u32> {
        let mut id = [0i32];
        self.ctx.stream.memcpy_dtoh(&self.next_token, &mut id)?;
        self.ctx.counted_sync()?;
        let id = id[0] as u32;
        self.next_host_token = Some(id);
        Ok(id)
    }

    pub fn read_logits(&self) -> Result<Vec<f32>> {
        if self.format.is_16bit() {
            let lm = self.proj16.get("lm_head").context("lm_head resident")?;
            return self.ctx.dtoh(&lm.y);
        }
        let lm = self.proj.get("lm_head").context("lm_head resident")?;
        self.ctx.dtoh(lm.y_ref())
    }

    pub fn prefill_modes(&self) -> Vec<&'static str> {
        std::iter::once(self.prefill)
            .chain(self.peers.iter().map(|p| p.prefill))
            .map(Mode::as_str)
            .collect()
    }

    pub fn prefill_packed_tokens(&self) -> usize {
        self.prefill_packed_tokens
    }

    /// The admission-derived prefill block size.
    pub fn prefill_block(&self) -> usize {
        self.prefill_block
    }

    /// Logits of every row of the most recent prefill block, [tokens,
    /// vocab] on the host. Single-device only; the block's x1 rows are
    /// already final-normed. The lm_head readout is the decode one (one
    /// row per launch), so every arm scores with identical numerics.
    pub fn block_logits(&mut self, tokens: usize) -> Result<Vec<f32>> {
        ensure!(self.peers.is_empty(), "block_logits is single-device");
        let text = self.config.text_config.clone();
        let n = text.hidden_size;
        let slab = &self.batch.x1;
        let mut lm_x = self.ctx.stream.alloc_zeros::<f32>(n)?;
        let mut out = Vec::with_capacity(tokens * text.vocab_size);
        for t in 0..tokens {
            self.ctx
                .stream
                .memcpy_dtod(&slab.slice(t * n..(t + 1) * n), &mut lm_x)?;
            if self.format.is_16bit() {
                let lm = self.proj16.get("lm_head").context("lm_head resident")?;
                let seg = lm.seg();
                let e = [seg.empty_like(), seg.empty_like(), seg.empty_like()];
                self.g16.group(
                    &self.ctx,
                    self.format,
                    [&seg, &e[0], &e[1], &e[2]],
                    &lm_x,
                    lm.in_dim,
                )?;
                out.extend(self.ctx.dtoh(&lm.y)?);
            } else {
                let lm = self.proj.get("lm_head").context("lm_head resident")?;
                let segs = [
                    lm.group_seg(),
                    lm.empty_seg_like(),
                    lm.empty_seg_like(),
                    lm.empty_seg_like(),
                ];
                self.wide.group(
                    &self.ctx,
                    &segs,
                    [lm.y_ref(); 4],
                    &GroupPair {
                        x: &lm_x,
                        xb: &lm_x,
                    },
                    lm.in_dim,
                    1,
                )?;
                out.extend(self.ctx.dtoh(lm.y_ref())?);
            }
        }
        Ok(out)
    }

    /// Streaming pipeline counters across all 16-bit streamed ranges.
    pub fn stream16_stats(&self) -> Option<Stream16Stats> {
        let mut total = None;
        for state in self
            .streaming16
            .iter()
            .chain(self.peers.iter().filter_map(|p| p.streaming16.as_ref()))
        {
            let s = *state.stats.lock().expect("stream stats poisoned");
            let acc = total.get_or_insert(Stream16Stats::default());
            acc.fill_ms += s.fill_ms;
            acc.gate_wait_ms += s.gate_wait_ms;
            acc.h2d_ms += s.h2d_ms;
            acc.h2d_bytes += s.h2d_bytes;
        }
        total
    }
}

fn reset_stack(
    ctx: &GpuContext,
    gdn: &mut [GpuGdn],
    pos: &mut CudaSliceI,
    rope_pos: &mut CudaSliceI,
    streaming: Option<&StreamState>,
    streaming16: Option<&StreamState16>,
) -> Result<()> {
    ctx.stream.synchronize()?;
    if let Some(state) = streaming {
        state.prefetch.synchronize()?;
    }
    if let Some(state) = streaming16 {
        state.prefetch.synchronize()?;
    }
    for g in gdn {
        let (conv, recurrent) = ctx.gdn_state_mut(g);
        ctx.stream.memset_zeros(conv)?;
        ctx.stream.memset_zeros(recurrent)?;
    }
    ctx.stream.memset_zeros(pos)?;
    ctx.stream.memset_zeros(rope_pos)?;
    Ok(())
}

struct StackView<'a> {
    ctx: &'a GpuContext,
    attn: Option<&'a AttnPrefillGpu>,
    wide: &'a crate::wide::WideKernels,
    g16: &'a Gemv16Kernels,
    format: QuantFormat,
    proj: &'a HashMap<String, GpuQuant>,
    proj16: &'a HashMap<String, GpuProj16>,
    ln: &'a [[CudaSliceF; 2]],
    attn_norms: &'a [[CudaSliceF; 2]],
    gdn: &'a [GpuGdn],
    kv_keys: &'a [CudaSliceF],
    kv_values: &'a [CudaSliceF],
    q_out: &'a CudaSliceF,
    gate_out: &'a CudaSliceF,
    attn_out: &'a CudaSliceF,
    inner: &'a CudaSliceF,
    pos: &'a mut CudaSliceI,
    rope_pos: &'a mut CudaSliceI,
    hidden: &'a CudaSliceF,
    x1: &'a CudaSliceF,
    boundary_norm: &'a CudaSliceF,
    /// Streaming weights for this range; None runs fully resident.
    stream: Option<&'a mut StreamState>,
    /// 16-bit streaming weights; set instead of `stream` on raw checkpoints.
    stream16: Option<&'a mut StreamState16>,
}

/// The two streaming rings a range may drive; at most one is set.
#[derive(Clone, Copy)]
struct Streams<'a> {
    i4: Option<&'a StreamState>,
    w16: Option<&'a StreamState16>,
}

/// A projection either resident (GpuQuant) or in this layer's slot.
enum ProjHandle<'a> {
    Resident(&'a GpuQuant),
    Slot(&'a SlotBuf),
}

impl ProjHandle<'_> {
    fn matrix(&self) -> crate::prefill::Matrix<'_> {
        let (packed, scales, biases) = self.tensors();
        crate::prefill::Matrix {
            packed,
            scales,
            biases,
            rows: self.out_dim(),
            cols: self.in_dim(),
        }
    }

    fn y_ref(&self) -> &CudaSlice<f32> {
        match self {
            Self::Resident(q) => q.y_ref(),
            Self::Slot(b) => &b.y,
        }
    }

    fn tensors(&self) -> (&CudaSlice<u32>, &CudaSlice<u16>, &CudaSlice<u16>) {
        match self {
            Self::Resident(q) => q.tensors(),
            Self::Slot(b) => (&b.packed, &b.scales, &b.biases),
        }
    }

    fn group_seg(&self) -> GroupSeg<'_> {
        match self {
            Self::Resident(q) => q.group_seg(),
            Self::Slot(b) => GroupSeg {
                packed: &b.packed,
                scales: &b.scales,
                biases: &b.biases,
                y: &b.y,
                rows: b.out_dim,
                lora: None,
            },
        }
    }

    /// A zero-row segment reusing the projection's pointers.
    fn empty_seg_like(&self) -> GroupSeg<'_> {
        match self {
            Self::Resident(q) => q.empty_seg_like(),
            Self::Slot(b) => GroupSeg {
                packed: &b.packed,
                scales: &b.scales,
                biases: &b.biases,
                y: &b.y,
                rows: 0,
                lora: None,
            },
        }
    }

    fn out_dim(&self) -> usize {
        match self {
            Self::Resident(q) => q.out_dim,
            Self::Slot(b) => b.out_dim,
        }
    }

    fn in_dim(&self) -> usize {
        match self {
            Self::Resident(q) => q.in_dim,
            Self::Slot(b) => b.in_dim,
        }
    }
}

fn layer_proj<'a>(
    view: &'a StackView,
    stream: Option<&'a StreamState>,
    layer: usize,
    suffix: &str,
) -> Result<ProjHandle<'a>> {
    match stream {
        Some(state) if layer >= state.range.0 => {
            Ok(ProjHandle::Slot(state.slot_buf(layer, suffix)?))
        }
        _ => {
            let name = format!("{TEXT_PREFIX}.layers.{layer}.{suffix}");
            view.proj
                .get(&name)
                .map(ProjHandle::Resident)
                .with_context(|| format!("{name} not resident"))
        }
    }
}

/// A 16-bit projection either resident (GpuProj16) or in this layer's slot.
enum Proj16<'a> {
    Resident(&'a GpuProj16),
    Slot(&'a SlotBuf16),
}

impl Proj16<'_> {
    fn seg(&self) -> Seg16<'_> {
        match self {
            Self::Resident(p) => Seg16 {
                w: &p.w,
                y: &p.y,
                rows: p.out_dim,
            },
            Self::Slot(b) => Seg16 {
                w: &b.w,
                y: &b.y,
                rows: b.out_dim,
            },
        }
    }

    fn y_ref(&self) -> &CudaSlice<f32> {
        match self {
            Self::Resident(p) => &p.y,
            Self::Slot(b) => &b.y,
        }
    }

    fn in_dim(&self) -> usize {
        match self {
            Self::Resident(p) => p.in_dim,
            Self::Slot(b) => b.in_dim,
        }
    }

    fn w(&self) -> &CudaSlice<u8> {
        match self {
            Self::Resident(p) => &p.w,
            Self::Slot(b) => &b.w,
        }
    }

    fn out_dim(&self) -> usize {
        match self {
            Self::Resident(p) => p.out_dim,
            Self::Slot(b) => b.out_dim,
        }
    }
}

fn layer_proj16<'a>(
    view: &'a StackView,
    stream: Option<&'a StreamState16>,
    layer: usize,
    suffix: &str,
) -> Result<Proj16<'a>> {
    match stream {
        Some(state) if layer >= state.range.0 => Ok(Proj16::Slot(state.slot_buf(layer, suffix)?)),
        _ => {
            let name = format!("{TEXT_PREFIX}.layers.{layer}.{suffix}");
            view.proj16
                .get(&name)
                .map(Proj16::Resident)
                .with_context(|| format!("{name} not resident"))
        }
    }
}

/// Attention + MLP for one device's contiguous layer range. `x1` enters
/// normed for the range's first layer and leaves normed by the range's
/// boundary norm; `hidden` carries the residual across. A streaming view
/// waits each layer's slot fill, refills the ring one layer ahead, and
/// frees the slot once the layer's last projection reader is enqueued.
/// Where a run_stack call sits inside its device's layer range: a span may
/// cover the whole range or only the resident prefix of a split step.
struct SpanOrigin {
    local_base: usize,
    range_end: usize,
    gdn_index_start: usize,
    kv_index_start: usize,
}

fn run_stack(
    view: &mut StackView,
    layers: Range<usize>,
    text: &TextConfig,
    origin: SpanOrigin,
) -> Result<()> {
    let mut gdn_index = origin.gdn_index_start;
    let mut kv_index = origin.kv_index_start;
    let end = origin.range_end;
    let mut stream = view.stream.take();
    let mut stream16 = view.stream16.take();
    if let Some(state) = stream.as_mut() {
        state.prime()?;
    }
    if let Some(state) = stream16.as_mut() {
        state.prime()?;
    }
    // Each ring's range.0 is its resident prefix boundary; layers below it
    // run from the resident projection map and skip the slot ring entirely.
    let through4 = stream.as_ref().map_or(end, |state| state.range.0);
    let through16 = stream16.as_ref().map_or(end, |state| state.range.0);
    for layer in layers {
        let local = layer - origin.local_base;
        if layer >= through4
            && let Some(state) = stream.as_mut()
        {
            state.begin_layer(view.ctx, layer)?;
        }
        if layer >= through16
            && let Some(state) = stream16.as_mut()
        {
            state.begin_layer(view.ctx, layer)?;
        }
        run_layer(
            view,
            text,
            LayerSpan {
                layer,
                local,
                range_end: end,
                gdn_index,
                kv_index,
                hidden: view.hidden,
                x1: view.x1,
                pos_off: 0,
            },
            Streams {
                i4: stream.as_deref(),
                w16: stream16.as_deref(),
            },
        )?;
        if text.layer_kind(layer) == LayerKind::LinearAttention {
            gdn_index += 1;
        } else {
            kv_index += 1;
        }
        if layer >= through4
            && let Some(state) = stream.as_mut()
        {
            state.end_layer(view.ctx, layer)?;
        }
        if layer >= through16
            && let Some(state) = stream16.as_mut()
        {
            state.end_layer(view.ctx, layer)?;
        }
    }
    Ok(())
}

struct BlockArgs<'a> {
    layers: Range<usize>,
    tokens: usize,
    base: usize,
    batch: &'a mut crate::prefill::Batch,
}

struct LayerBlock {
    layer: usize,
    local: usize,
    end: usize,
    gdn_index: usize,
    kv_index: usize,
    tokens: usize,
    base: usize,
}

/// One batched projection's geometry: layer, suffix, block length, and
/// the packed engine's split-K (4 on the wide down shape, 1 elsewhere).
struct ProjArgs<'a> {
    layer: usize,
    suffix: &'a str,
    tokens: usize,
    split: usize,
}

fn gemv_rows(
    ctx: &GpuContext,
    x: &CudaSliceF,
    y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
    input: &mut CudaSliceF,
    output: &CudaSliceF,
    shape: (usize, usize, usize),
    mut launch: impl FnMut(&CudaSliceF) -> Result<()>,
) -> Result<()> {
    let (tokens, rows, cols) = shape;
    ensure!(
        x.len() >= tokens * cols && y.len() >= tokens * rows && input.len() >= cols,
        "GEMV block buffer is too small"
    );
    for t in 0..tokens {
        ctx.stream.memcpy_dtod(
            &x.slice(t * cols..(t + 1) * cols),
            &mut input.slice_mut(..cols),
        )?;
        launch(input)?;
        let (source, _source_guard) = output.device_ptr(&ctx.stream);
        let (target, _target_guard) = y.device_ptr_mut(&ctx.stream);
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                target + (t * rows * size_of::<f32>()) as u64,
                source,
                rows * size_of::<f32>(),
                ctx.stream.cu_stream(),
            )?;
        }
    }
    Ok(())
}

struct AttnPrefillGpu {
    qk: cudarc::driver::CudaFunction,
    rope: MropeGeom,
    kern: cudarc::driver::CudaFunction,
    combine: cudarc::driver::CudaFunction,
    partials: CudaSliceF,
    pml: CudaSliceF,
    smem_bytes: u32,
    split: usize,
    blocks_per_sm: u32,
    sm_count: usize,
    groups: usize,
    kv_heads: usize,
}

impl AttnPrefillGpu {
    fn new(ctx: &GpuContext, max_ctx: usize, text: &TextConfig) -> Result<Self> {
        let sm_count: usize = ctx
            .context
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?
            .try_into()
            .context("invalid CUDA SM count")?;
        ensure!(
            max_ctx > 0 && sm_count > 0,
            "invalid attention context or SM count"
        );
        let module = ff_edge0::kernel_assets::load_module(
            &ctx.context,
            &crate::kernel_assets::ATTN_PREFILL,
        )?;
        let kern = module.load_function("attn_prefill")?;
        let combine = module.load_function("attn_prefill_combine")?;
        let qk_module = ff_edge0::kernel_assets::load_module(
            &ctx.context,
            &crate::kernel_assets::QK_NORM_ROPE_ROWS,
        )
        .context("qk_norm_rope_rows module load failed")?;
        let qk = qk_module
            .load_function("qk_norm_rope_rows")
            .context("qk_norm_rope_rows missing")?;
        let rope = MropeGeom {
            heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rotary_dim: (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize,
            theta: text.rope.rope_theta,
            sec_h: text.rope.mrope_section[1],
            sec_w: text.rope.mrope_section[2],
        };
        let smem_bytes = AttnPrefillPlan::smem_bytes();
        kern.set_attribute(cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes as i32)
            .with_context(|| format!("tiled attention dynamic SMEM opt-in refused for {smem_bytes} B"))?;
        let blocks_per_sm = kern.occupancy_max_active_blocks_per_multiprocessor(
            TPB_ATTN as u32,
            smem_bytes as usize,
            None,
        )?;
        ensure!(blocks_per_sm > 0, "tiled attention occupancy is zero");
        let kv_heads = text.num_key_value_heads;
        let groups = kv_heads * (text.num_attention_heads / kv_heads).div_ceil(BM_ATTN / BT_ATTN);
        let rows = AttnPrefillPlan::capacity(max_ctx, sm_count, blocks_per_sm, groups)?;
        let bytes = rows * (D_ATTN + 2) * size_of::<f32>();
        let free = ctx.context.mem_get_info()?.0;
        ensure!(
            bytes <= free,
            "tiled attention partials require {bytes} B, only {free} B free"
        );
        let partials = ctx.stream.alloc_zeros::<f32>(rows * D_ATTN)?;
        let pml = ctx.stream.alloc_zeros::<f32>(rows * 2)?;
        eprintln!(
            "qwen35: tiled attention: {smem_bytes} B shared, {blocks_per_sm} blocks/SM, {} SMs, {bytes} B device scratch",
            sm_count
        );
        Ok(Self {
            qk,
            rope,
            kern,
            combine,
            partials,
            pml,
            smem_bytes,
            split: MAX_S_ATTN,
            blocks_per_sm,
            sm_count,
            groups,
            kv_heads,
        })
    }

    fn qk_rows(
        &self,
        view: &StackView,
        batch: &crate::prefill::Batch,
        tokens: usize,
        kv_index: usize,
    ) -> Result<()> {
        let geom = &self.rope;
        let kv_dim = geom.kv_heads * geom.head_dim;
        let [qn, kn] = &view.attn_norms[kv_index];
        let tokens = u32::try_from(tokens).context("qk_norm_rope_rows tokens")?;
        ensure!(
            tokens > 0 && tokens <= 65535,
            "qk_norm_rope_rows token grid exceeds 1..=65535"
        );
        unsafe {
            view.ctx
                .stream
                .launch_builder(&self.qk)
                .arg(&batch.gate)
                .arg(qn)
                .arg(&batch.up)
                .arg(kn)
                .arg(
                    &batch
                        .up
                        .slice(tokens as usize * kv_dim..tokens as usize * 2 * kv_dim),
                )
                .arg(&batch.qslab)
                .arg(&batch.gslab)
                .arg(&view.kv_keys[kv_index])
                .arg(&view.kv_values[kv_index])
                .arg(&*view.pos)
                .arg(&*view.rope_pos)
                .arg(&(kv_dim as i32))
                .arg(&(geom.heads as i32))
                .arg(&(geom.kv_heads as i32))
                .arg(&(geom.head_dim as i32))
                .arg(&(geom.rotary_dim as i32))
                .arg(&geom.theta)
                .arg(&(geom.sec_h as i32))
                .arg(&(geom.sec_w as i32))
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: ((geom.heads + 2 * geom.kv_heads) as u32, tokens, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("qk_norm_rope_rows launch failed")?;
        }
        Ok(())
    }
}

struct AttnLaunchArgs {
    kv_index: usize,
    kv_stride: usize,
    heads: usize,
    tokens: usize,
    base: usize,
    scale: f32,
}

struct AttnPrefillPlan {
    qtiles: usize,
    groups: usize,
    split: usize,
    chunk: usize,
    smem_bytes: u32,
}

impl AttnPrefillPlan {
    fn capacity(
        max_ctx: usize,
        sm_count: usize,
        blocks_per_sm: u32,
        groups: usize,
    ) -> Result<usize> {
        (1..=crate::prefill::BLOCK.min(max_ctx))
            .map(|tokens| {
                Self::compute(
                    tokens,
                    max_ctx - tokens,
                    MAX_S_ATTN,
                    Self::smem_bytes(),
                    sm_count,
                    blocks_per_sm,
                    groups,
                )
                .rows()
            })
            .max()
            .context("no attention block fits the context")
    }

    fn rows(&self) -> usize {
        self.qtiles * self.groups * self.split * BM_ATTN
    }

    const fn smem_bytes() -> u32 {
        ((BM_ATTN * D_ATTN + 2 * BN_ATTN * D_ATTN + BM_ATTN * BN_ATTN) * 2
            + BM_ATTN * 2 * size_of::<f32>()) as u32
    }

    fn compute(
        tokens: usize,
        base: usize,
        split_cap: usize,
        smem_bytes: u32,
        sm_count: usize,
        blocks_per_sm: u32,
        groups: usize,
    ) -> Self {
        let qtiles = tokens.div_ceil(BT_ATTN);
        let per = qtiles * groups;
        let want = sm_count * blocks_per_sm as usize * 2;
        let cap = (base + tokens).div_ceil(BN_ATTN);
        let split = want.div_ceil(per).min(cap).min(split_cap).max(1);
        let causal_max = base + tokens;
        let chunk = causal_max.div_ceil(split);
        Self {
            qtiles,
            groups,
            split,
            chunk,
            smem_bytes,
        }
    }
}

fn attention_prefill_launch(
    view: &StackView,
    q: &CudaSliceF,
    gate: &CudaSliceF,
    out: &CudaSliceF,
    args: AttnLaunchArgs,
) -> Result<()> {
    let attn = view
        .attn
        .context("missing device-local attention prefill state")?;
    let plan = AttnPrefillPlan::compute(
        args.tokens,
        args.base,
        attn.split,
        attn.smem_bytes,
        attn.sm_count,
        attn.blocks_per_sm,
        attn.groups,
    );
    ensure!(
        plan.rows() * D_ATTN <= attn.partials.len() && plan.rows() * 2 <= attn.pml.len(),
        "attention plan exceeds admitted partial storage"
    );
    let t_i = args.tokens as i32;
    let base_i = args.base as i32;
    unsafe {
        view.ctx
            .stream
            .launch_builder(&attn.kern)
            .arg(q)
            .arg(gate)
            .arg(&view.kv_keys[args.kv_index])
            .arg(&view.kv_values[args.kv_index])
            .arg(&attn.partials)
            .arg(&attn.pml)
            .arg(&t_i)
            .arg(&base_i)
            .arg(&(attn.kv_heads as i32))
            .arg(&(args.kv_stride as i32))
            .arg(&(args.heads as i32))
            .arg(&args.scale)
            .arg(&(plan.chunk as i32))
            .launch(cudarc::driver::LaunchConfig {
                grid_dim: (plan.qtiles as u32, plan.groups as u32, plan.split as u32),
                block_dim: (TPB_ATTN as u32, 1, 1),
                shared_mem_bytes: plan.smem_bytes,
            })
            .context(format!(
                "tiled attention prefill launch failed: qtiles {}, split {}, chunk {}, smem {} B",
                plan.qtiles, plan.split, plan.chunk, plan.smem_bytes
            ))?;
    }
    unsafe {
        view.ctx
            .stream
            .launch_builder(&attn.combine)
            .arg(&attn.partials)
            .arg(&attn.pml)
            .arg(gate)
            .arg(out)
            .arg(&t_i)
            .arg(&base_i)
            .arg(&(attn.kv_heads as i32))
            .arg(&(args.heads as i32))
            .arg(&(plan.split as i32))
            .launch(cudarc::driver::LaunchConfig {
                grid_dim: (plan.qtiles as u32, plan.groups as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .context("tiled attention combine launch failed")?;
    }
    Ok(())
}

const BM_ATTN: usize = 48;
const D_ATTN: usize = 256;
const MAX_S_ATTN: usize = 64;
const BT_ATTN: usize = 8;
const BN_ATTN: usize = 32;
const TPB_ATTN: usize = 192;

/// One batched projection for the layer, routed by checkpoint format.
fn proj_batch(
    view: &StackView,
    streams: Streams,
    projector: &mut crate::prefill::Projector,
    args: ProjArgs<'_>,
    x: &CudaSliceF,
    y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
) -> Result<()> {
    if let crate::prefill::Projector::Gemv { input } = projector {
        let p16 = if view.format.is_16bit() {
            Some(layer_proj16(view, streams.w16, args.layer, args.suffix)?)
        } else {
            None
        };
        let p4 = if view.format.is_16bit() {
            None
        } else {
            Some(layer_proj(view, streams.i4, args.layer, args.suffix)?)
        };
        let (rows, cols, output) = if let Some(p) = &p16 {
            (p.out_dim(), p.in_dim(), p.y_ref())
        } else {
            let p = p4.as_ref().context("missing int4 projection")?;
            (p.out_dim(), p.in_dim(), p.y_ref())
        };
        return gemv_rows(
            view.ctx,
            x,
            y,
            input,
            output,
            (args.tokens, rows, cols),
            |input| {
                if let Some(p) = &p16 {
                    let seg = p.seg();
                    let empty = seg.empty_like();
                    view.g16.group(
                        view.ctx,
                        view.format,
                        [&seg, &empty, &empty, &empty],
                        input,
                        cols,
                    )?;
                } else {
                    let p = p4.as_ref().context("missing int4 projection")?;
                    if args.split > 1 {
                        let (packed, scales, biases) = p.tensors();
                        view.wide.down(
                            view.ctx,
                            &DownBuffers {
                                packed,
                                scales,
                                biases,
                                x: input,
                                xb: input,
                                y: output,
                                yb: output,
                            },
                            &WideGeom {
                                rows,
                                in_dim: cols,
                                cols: 1,
                            },
                        )?;
                    } else {
                        let segs = [
                            p.group_seg(),
                            p.empty_seg_like(),
                            p.empty_seg_like(),
                            p.empty_seg_like(),
                        ];
                        view.wide.group(
                            view.ctx,
                            &segs,
                            [output; 4],
                            &GroupPair {
                                x: input,
                                xb: input,
                            },
                            cols,
                            1,
                        )?;
                    }
                }
                Ok(())
            },
        );
    }
    if view.format.is_16bit() {
        let p = layer_proj16(view, streams.w16, args.layer, args.suffix)?;
        let (rows, in_dim) = (p.out_dim(), p.in_dim());
        projector.project16(p.w(), x, y, rows, in_dim, args.tokens)
    } else {
        let p = layer_proj(view, streams.i4, args.layer, args.suffix)?;
        projector.project(p.matrix(), x, y, args.tokens, args.split)
    }
}

/// One layer over a prefill block.
fn run_block_packed(
    view: &StackView,
    text: &TextConfig,
    batch: &mut crate::prefill::Batch,
    span: LayerBlock,
    streams: Streams,
) -> Result<()> {
    let LayerBlock {
        layer,
        local,
        end,
        gdn_index,
        kv_index,
        tokens,
        base,
    } = span;
    let n = text.hidden_size;
    let eps = text.rms_norm_eps as f32;
    match text.layer_kind(layer) {
        LayerKind::LinearAttention => {
            let conv = text.conv_dim();
            let z_dim = text.linear_num_value_heads * text.linear_value_head_dim;
            let ba = text.linear_num_value_heads;

            // Mixer slabs alias the MLP buffers: the mixer phase's live
            // range ends before the MLP writes them, and gate/up/inner are
            // dead until the MLP, so one arena carries both uses.
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "linear_attn.in_proj_qkv",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch.gate.slice_mut(..tokens * conv),
            )?;
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "linear_attn.in_proj_z",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch.up.slice_mut(..tokens * z_dim),
            )?;
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "linear_attn.in_proj_b",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch.up.slice_mut(tokens * z_dim..tokens * (z_dim + ba)),
            )?;
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "linear_attn.in_proj_a",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch
                    .up
                    .slice_mut(tokens * (z_dim + ba)..tokens * (z_dim + 2 * ba)),
            )?;
            let g = &view.gdn[gdn_index];
            batch
                .gdn
                .as_ref()
                .context("GDN prefill was not calibrated")?
                .run(view.ctx, g, batch, tokens, eps)?;
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "linear_attn.out_proj",
                    tokens,
                    split: 1,
                },
                &batch.inner,
                &mut batch.out,
            )?;
        }
        LayerKind::FullAttention => {
            let q_dim = 2 * text.num_attention_heads * text.head_dim;
            let kv_dim = text.num_key_value_heads * text.head_dim;

            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "self_attn.q_proj",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch.gate.slice_mut(..tokens * q_dim),
            )?;
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "self_attn.k_proj",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch.up.slice_mut(..tokens * kv_dim),
            )?;
            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "self_attn.v_proj",
                    tokens,
                    split: 1,
                },
                &batch.x1,
                &mut batch.up.slice_mut(tokens * kv_dim..tokens * 2 * kv_dim),
            )?;
            view.attn
                .context("missing device-local attention prefill state")?
                .qk_rows(view, batch, tokens, kv_index)?;
            let scale = 1.0 / (text.head_dim as f32).sqrt();
            {
                attention_prefill_launch(
                    view,
                    &batch.qslab,
                    &batch.gslab,
                    &batch.inner,
                    AttnLaunchArgs {
                        kv_index,
                        kv_stride: text.num_key_value_heads * text.head_dim,
                        heads: text.num_attention_heads,
                        tokens,
                        base,
                        scale,
                    },
                )?;
            }

            proj_batch(
                view,
                streams,
                &mut batch.projector,
                ProjArgs {
                    layer,
                    suffix: "self_attn.o_proj",
                    tokens,
                    split: 1,
                },
                &batch.inner,
                &mut batch.out,
            )?;
        }
    }
    batch.rows.add_rmsnorm(
        view.ctx,
        crate::prefill::AddNormRows {
            acc: &batch.hidden,
            delta: &batch.out,
            w: &view.ln[local][1],
            out: &batch.x1,
            tokens,
            n,
            eps,
        },
    )?;
    proj_batch(
        view,
        streams,
        &mut batch.projector,
        ProjArgs {
            layer,
            suffix: "mlp.gate_proj",
            tokens,
            split: 1,
        },
        &batch.x1,
        &mut batch.gate,
    )?;
    proj_batch(
        view,
        streams,
        &mut batch.projector,
        ProjArgs {
            layer,
            suffix: "mlp.up_proj",
            tokens,
            split: 1,
        },
        &batch.x1,
        &mut batch.up,
    )?;
    view.ctx.silu_mul(
        &batch.gate,
        &batch.up,
        &batch.inner,
        tokens * text.intermediate_size,
    )?;
    proj_batch(
        view,
        streams,
        &mut batch.projector,
        ProjArgs {
            layer,
            suffix: "mlp.down_proj",
            tokens,
            split: 4,
        },
        &batch.inner,
        &mut batch.out,
    )?;
    let next_w: &CudaSliceF = if layer + 1 < end {
        &view.ln[local + 1][0]
    } else {
        view.boundary_norm
    };
    batch.rows.add_rmsnorm(
        view.ctx,
        crate::prefill::AddNormRows {
            acc: &batch.hidden,
            delta: &batch.out,
            w: next_w,
            out: &batch.x1,
            tokens,
            n,
            eps,
        },
    )?;
    Ok(())
}

/// Advance each layer over the prefill block.
fn run_block(view: &mut StackView, text: &TextConfig, args: BlockArgs<'_>) -> Result<()> {
    let BlockArgs {
        layers,
        tokens,
        base,
        batch,
    } = args;
    let mut gdn_index = 0usize;
    let mut kv_index = 0usize;
    let start = layers.start;
    let end = layers.end;
    let mut stream = view.stream.take();
    let mut stream16 = view.stream16.take();
    if let Some(state) = stream.as_mut() {
        state.prime()?;
    }
    if let Some(state) = stream16.as_mut() {
        state.prime()?;
    }
    let through4 = stream.as_ref().map_or(end, |state| state.range.0);
    let through16 = stream16.as_ref().map_or(end, |state| state.range.0);
    for layer in layers {
        let local = layer - start;
        if layer >= through4
            && let Some(state) = stream.as_mut()
        {
            state.begin_layer(view.ctx, layer)?;
        }
        if layer >= through16
            && let Some(state) = stream16.as_mut()
        {
            state.begin_layer(view.ctx, layer)?;
        }

        run_block_packed(
            view,
            text,
            batch,
            LayerBlock {
                layer,
                local,
                end,
                gdn_index,
                kv_index,
                tokens,
                base,
            },
            Streams {
                i4: stream.as_deref(),
                w16: stream16.as_deref(),
            },
        )?;

        if text.layer_kind(layer) == LayerKind::LinearAttention {
            gdn_index += 1;
        } else {
            kv_index += 1;
        }
        if layer >= through4
            && let Some(state) = stream.as_mut()
        {
            state.end_layer(view.ctx, layer)?;
        }
        if layer >= through16
            && let Some(state) = stream16.as_mut()
        {
            state.end_layer(view.ctx, layer)?;
        }
    }
    Ok(())
}

/// One layer of one token: attention (GDN or full) on `x1`, the dense
/// MLP, then the fused residual add and next-layer norm. `pos_off` is the
/// token's index into the block position buffers (0 in decode).
#[derive(Clone, Copy)]
struct LayerSpan<'a> {
    layer: usize,
    local: usize,
    range_end: usize,
    gdn_index: usize,
    kv_index: usize,
    hidden: &'a CudaSliceF,
    x1: &'a CudaSliceF,
    pos_off: usize,
}

fn run_layer(
    view: &StackView,
    text: &TextConfig,
    span: LayerSpan<'_>,
    streams: Streams,
) -> Result<()> {
    run_mixer(view, text, span, streams)?;
    run_mlp(view, text, span, streams)
}

fn run_mixer(
    view: &StackView,
    text: &TextConfig,
    span: LayerSpan<'_>,
    streams: Streams,
) -> Result<()> {
    let eps = text.rms_norm_eps as f32;
    let n = text.hidden_size;
    let LayerSpan {
        layer,
        local,
        gdn_index,
        kv_index,
        hidden,
        x1,
        ..
    } = span;
    // x1 enters holding rmsnorm_zc(hidden, ln_in).
    if text.layer_kind(layer) == LayerKind::LinearAttention {
        if view.format.is_16bit() {
            let qkv = layer_proj16(view, streams.w16, layer, "linear_attn.in_proj_qkv")?;
            let z = layer_proj16(view, streams.w16, layer, "linear_attn.in_proj_z")?;
            let b = layer_proj16(view, streams.w16, layer, "linear_attn.in_proj_b")?;
            let a = layer_proj16(view, streams.w16, layer, "linear_attn.in_proj_a")?;
            let out_proj = layer_proj16(view, streams.w16, layer, "linear_attn.out_proj")?;
            let [qs, zs, bs, gas] = [qkv.seg(), z.seg(), b.seg(), a.seg()];
            view.g16.group(
                view.ctx,
                view.format,
                [&qs, &zs, &bs, &gas],
                x1,
                qkv.in_dim(),
            )?;
            let g = &view.gdn[gdn_index];
            view.ctx.gdn_conv_heads(
                g,
                qkv.y_ref(),
                z.y_ref(),
                b.y_ref(),
                a.y_ref(),
                view.ctx.gdn_out_buf(g),
            )?;
            let gout = view.ctx.gdn_out_buf(g);
            let os = out_proj.seg();
            let e = [os.empty_like(), os.empty_like(), os.empty_like()];
            view.g16.group(
                view.ctx,
                view.format,
                [&os, &e[0], &e[1], &e[2]],
                gout,
                out_proj.in_dim(),
            )?;
            view.ctx.glue_add_rmsnorm_zc(
                hidden,
                out_proj.y_ref(),
                &view.ln[local][1],
                x1,
                n,
                eps,
            )?;
            return Ok(());
        }
        let projs = streams.i4;
        let qkv = layer_proj(view, projs, layer, "linear_attn.in_proj_qkv")?;
        let z = layer_proj(view, projs, layer, "linear_attn.in_proj_z")?;
        let b = layer_proj(view, projs, layer, "linear_attn.in_proj_b")?;
        let a = layer_proj(view, projs, layer, "linear_attn.in_proj_a")?;
        let out_proj = layer_proj(view, projs, layer, "linear_attn.out_proj")?;
        let segs = [qkv.group_seg(), z.group_seg(), b.group_seg(), a.group_seg()];
        view.wide.group(
            view.ctx,
            &segs,
            [qkv.y_ref(), z.y_ref(), b.y_ref(), a.y_ref()],
            &GroupPair { x: x1, xb: x1 },
            qkv.in_dim(),
            1,
        )?;
        let g = &view.gdn[gdn_index];
        view.ctx.gdn_conv_heads(
            g,
            qkv.y_ref(),
            z.y_ref(),
            b.y_ref(),
            a.y_ref(),
            view.ctx.gdn_out_buf(g),
        )?;
        let gout = view.ctx.gdn_out_buf(g);
        let segs = [
            out_proj.group_seg(),
            out_proj.empty_seg_like(),
            out_proj.empty_seg_like(),
            out_proj.empty_seg_like(),
        ];
        view.wide.group(
            view.ctx,
            &segs,
            [out_proj.y_ref(); 4],
            &GroupPair { x: gout, xb: gout },
            out_proj.in_dim(),
            1,
        )?;
        view.ctx
            .glue_add_rmsnorm_zc(hidden, out_proj.y_ref(), &view.ln[local][1], x1, n, eps)?;
    } else {
        if view.format.is_16bit() {
            let q = layer_proj16(view, streams.w16, layer, "self_attn.q_proj")?;
            let k = layer_proj16(view, streams.w16, layer, "self_attn.k_proj")?;
            let v = layer_proj16(view, streams.w16, layer, "self_attn.v_proj")?;
            let o = layer_proj16(view, streams.w16, layer, "self_attn.o_proj")?;
            let [qs, ks, vs] = [q.seg(), k.seg(), v.seg()];
            let ke = ks.empty_like();
            view.g16
                .group(view.ctx, view.format, [&qs, &ks, &vs, &ke], x1, q.in_dim())?;
            let [qn, kn] = &view.attn_norms[kv_index];
            let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
            let pos_v = view.pos.slice(span.pos_off..);
            let rope_v = view.rope_pos.slice(3 * span.pos_off..);
            view.ctx.glue_attn_qk_zc_mrope(
                &QkvNorm {
                    q_raw: q.y_ref().as_view(),
                    q_norm_w: qn,
                    k_raw: k.y_ref().as_view(),
                    k_norm_w: kn,
                    v_raw: v.y_ref().as_view(),
                },
                &QkOutputs {
                    q_out: view.q_out.as_view(),
                    gate_out: view.gate_out.as_view(),
                },
                &KvCache {
                    keys: &view.kv_keys[kv_index],
                    values: &view.kv_values[kv_index],
                    stride: text.num_key_value_heads * text.head_dim,
                },
                &pos_v,
                &rope_v,
                &MropeGeom {
                    heads: text.num_attention_heads,
                    kv_heads: text.num_key_value_heads,
                    head_dim: text.head_dim,
                    rotary_dim,
                    theta: text.rope.rope_theta,
                    sec_h: text.rope.mrope_section[1],
                    sec_w: text.rope.mrope_section[2],
                },
            )?;
            let scale = 1.0 / (text.head_dim as f32).sqrt();
            view.ctx.glue_attn_scores_raw(
                &ScoreBuffers {
                    q: view.q_out.as_view(),
                    gate: view.gate_out.as_view(),
                    out: view.attn_out.as_view(),
                },
                &KvCache {
                    keys: &view.kv_keys[kv_index],
                    values: &view.kv_values[kv_index],
                    stride: text.num_key_value_heads * text.head_dim,
                },
                &pos_v,
                &AttnGeom {
                    heads: text.num_attention_heads,
                    kv_heads: text.num_key_value_heads,
                    head_dim: text.head_dim,
                },
                scale,
            )?;
            let os = o.seg();
            let e = [os.empty_like(), os.empty_like(), os.empty_like()];
            view.g16.group(
                view.ctx,
                view.format,
                [&os, &e[0], &e[1], &e[2]],
                view.attn_out,
                o.in_dim(),
            )?;
            view.ctx
                .glue_add_rmsnorm_zc(hidden, o.y_ref(), &view.ln[local][1], x1, n, eps)?;
            return Ok(());
        }
        let projs = streams.i4;
        let q = layer_proj(view, projs, layer, "self_attn.q_proj")?;
        let k = layer_proj(view, projs, layer, "self_attn.k_proj")?;
        let v = layer_proj(view, projs, layer, "self_attn.v_proj")?;
        let o = layer_proj(view, projs, layer, "self_attn.o_proj")?;
        let segs = [
            q.group_seg(),
            k.group_seg(),
            v.group_seg(),
            q.empty_seg_like(),
        ];
        view.wide.group(
            view.ctx,
            &segs,
            [q.y_ref(), k.y_ref(), v.y_ref(), q.y_ref()],
            &GroupPair { x: x1, xb: x1 },
            q.in_dim(),
            1,
        )?;
        let [qn, kn] = &view.attn_norms[kv_index];
        let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let pos_v = view.pos.slice(span.pos_off..);
        let rope_v = view.rope_pos.slice(3 * span.pos_off..);
        view.ctx.glue_attn_qk_zc_mrope(
            &QkvNorm {
                q_raw: q.y_ref().as_view(),
                q_norm_w: qn,
                k_raw: k.y_ref().as_view(),
                k_norm_w: kn,
                v_raw: v.y_ref().as_view(),
            },
            &QkOutputs {
                q_out: view.q_out.as_view(),
                gate_out: view.gate_out.as_view(),
            },
            &KvCache {
                keys: &view.kv_keys[kv_index],
                values: &view.kv_values[kv_index],
                stride: text.num_key_value_heads * text.head_dim,
            },
            &pos_v,
            &rope_v,
            &MropeGeom {
                heads: text.num_attention_heads,
                kv_heads: text.num_key_value_heads,
                head_dim: text.head_dim,
                rotary_dim,
                theta: text.rope.rope_theta,
                sec_h: text.rope.mrope_section[1],
                sec_w: text.rope.mrope_section[2],
            },
        )?;
        let scale = 1.0 / (text.head_dim as f32).sqrt();
        view.ctx.glue_attn_scores_raw(
            &ScoreBuffers {
                q: view.q_out.as_view(),
                gate: view.gate_out.as_view(),
                out: view.attn_out.as_view(),
            },
            &KvCache {
                keys: &view.kv_keys[kv_index],
                values: &view.kv_values[kv_index],
                stride: text.num_key_value_heads * text.head_dim,
            },
            &pos_v,
            &AttnGeom {
                heads: text.num_attention_heads,
                kv_heads: text.num_key_value_heads,
                head_dim: text.head_dim,
            },
            scale,
        )?;
        let segs = [
            o.group_seg(),
            o.empty_seg_like(),
            o.empty_seg_like(),
            o.empty_seg_like(),
        ];
        view.wide.group(
            view.ctx,
            &segs,
            [o.y_ref(); 4],
            &GroupPair {
                x: view.attn_out,
                xb: view.attn_out,
            },
            o.in_dim(),
            1,
        )?;
        view.ctx
            .glue_add_rmsnorm_zc(hidden, o.y_ref(), &view.ln[local][1], x1, n, eps)?;
    }
    Ok(())
}

fn run_mlp(
    view: &StackView,
    text: &TextConfig,
    span: LayerSpan<'_>,
    streams: Streams,
) -> Result<()> {
    let LayerSpan { layer, x1, .. } = span;
    // Dense MLP on x1, then the fused residual + next-layer's norm
    // (the boundary norm after the range's last layer).
    if view.format.is_16bit() {
        let gate = layer_proj16(view, streams.w16, layer, "mlp.gate_proj")?;
        let up = layer_proj16(view, streams.w16, layer, "mlp.up_proj")?;
        let down = layer_proj16(view, streams.w16, layer, "mlp.down_proj")?;
        let gs = gate.seg();
        let ge = [gs.empty_like(), gs.empty_like(), gs.empty_like()];
        view.g16.group(
            view.ctx,
            view.format,
            [&gs, &ge[0], &ge[1], &ge[2]],
            x1,
            gate.in_dim(),
        )?;
        let us = up.seg();
        let ue = [us.empty_like(), us.empty_like(), us.empty_like()];
        view.g16.group(
            view.ctx,
            view.format,
            [&us, &ue[0], &ue[1], &ue[2]],
            x1,
            up.in_dim(),
        )?;
        view.ctx
            .silu_mul(gate.y_ref(), up.y_ref(), view.inner, text.intermediate_size)?;
        let ds = down.seg();
        let de = [ds.empty_like(), ds.empty_like(), ds.empty_like()];
        view.g16.group(
            view.ctx,
            view.format,
            [&ds, &de[0], &de[1], &de[2]],
            view.inner,
            down.in_dim(),
        )?;
        return finish_layer(view, text, span, down.y_ref());
    }
    let projs = streams.i4;
    let gate = layer_proj(view, projs, layer, "mlp.gate_proj")?;
    let up = layer_proj(view, projs, layer, "mlp.up_proj")?;
    let down = layer_proj(view, projs, layer, "mlp.down_proj")?;
    view.wide.group(
        view.ctx,
        &[
            gate.group_seg(),
            up.group_seg(),
            gate.empty_seg_like(),
            gate.empty_seg_like(),
        ],
        [gate.y_ref(), up.y_ref(), gate.y_ref(), up.y_ref()],
        &GroupPair { x: x1, xb: x1 },
        gate.in_dim(),
        1,
    )?;
    view.ctx
        .silu_mul(gate.y_ref(), up.y_ref(), view.inner, text.intermediate_size)?;
    let (dp, ds, db) = down.tensors();
    view.wide.down(
        view.ctx,
        &DownBuffers {
            packed: dp,
            scales: ds,
            biases: db,
            x: view.inner,
            xb: view.inner,
            y: down.y_ref(),
            yb: down.y_ref(),
        },
        &WideGeom {
            rows: down.out_dim(),
            in_dim: down.in_dim(),
            cols: 1,
        },
    )?;
    finish_layer(view, text, span, down.y_ref())
}

fn finish_layer(
    view: &StackView,
    text: &TextConfig,
    span: LayerSpan<'_>,
    delta: &CudaSliceF,
) -> Result<()> {
    let LayerSpan {
        layer,
        local,
        range_end: end,
        hidden,
        x1,
        ..
    } = span;
    let n = text.hidden_size;
    let eps = text.rms_norm_eps as f32;
    let next_w: &CudaSliceF = if layer + 1 < end {
        &view.ln[local + 1][0]
    } else {
        view.boundary_norm
    };
    view.ctx
        .glue_add_rmsnorm_zc(hidden, delta, next_w, x1, n, eps)?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn grouped_names(kind: LayerKind, prefill: Mode) -> Vec<Vec<&'static str>> {
    let names = streaming_names(kind);
    let mixer = if kind == LayerKind::LinearAttention {
        4
    } else {
        3
    };
    let mut groups = vec![
        names[..mixer].to_vec(),
        vec![names[mixer]],
        names[mixer + 1..names.len() - 1].to_vec(),
    ];
    if prefill == Mode::Gemv {
        groups.extend(names[..names.len() - 1].iter().map(|name| vec![*name]));
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::PlanParams;
    use super::{
        GpuContext, LayerKind, Mode, QuantFormat, Qwen35Weights, QwenGpu, Residency, StreamState,
        StreamState16, TEXT_PREFIX, layer_device_bytes, override_free, plan_residency,
        prefill_bytes, ring_geom, split_layers_by_bytes,
    };
    use anyhow::Result;
    use ff_core::paths::checkpoint_dir;

    /// The bf16 cost model against a small budget: the plan's
    /// streaming_bytes never exceed the budget.
    #[test]
    fn bf16_cost_model_keeps_the_plan_within_a_small_budget() -> Result<()> {
        let Some(dir) = ff_core::paths::checkpoint_dir("Qwen/Qwen3.8-27B") else {
            eprintln!("FF_MODELS_DIR unset; skipping the cost-model test");
            return Ok(());
        };
        let weights = Qwen35Weights::open(&dir)?;
        let config = crate::config::Qwen35Config::from_model_dir(&dir)?;
        let granularity = 2 << 20;
        let mut prev = u64::MAX;
        for free in [
            15_470_428_160u64,
            15_264_630_464,
            15_064_630_464,
            14_064_630_464,
        ] {
            let plans = plan_residency(
                &weights,
                &config,
                PlanParams {
                    ranges: &[(0, 64)],
                    max_ctx: 4096,
                    free: &[free],
                    ring_slots: 2,
                    force_stream: false,
                    granularity,
                },
            )?;
            let p = &plans[0];
            assert!(
                p.streaming_bytes <= free - ff_core::probe::device_admission_reserve_bytes(),
                "free {free}: streaming_bytes {} exceeds the budget",
                p.streaming_bytes
            );
            assert!(
                p.resident_through as u64 <= prev,
                "free {free}: the prefix grew as free shrank"
            );
            prev = p.resident_through as u64;
        }
        Ok(())
    }

    /// prefill_bytes carries the GDN workspace at chunk 64: it rises by
    /// exactly bytes(chunk 64) − bytes(chunk 16) over a chunk-16 charge.
    #[test]
    fn prefill_bytes_charges_the_gdn_workspace_at_the_largest_chunk() {
        let Some(dir) = ff_core::paths::checkpoint_dir("Qwen/Qwen3.8-27B") else {
            eprintln!("FF_MODELS_DIR unset; skipping the gdn-charge test");
            return;
        };
        let weights = Qwen35Weights::open(&dir).unwrap();
        let config = crate::config::Qwen35Config::from_model_dir(&dir).unwrap();
        let format = weights.format();
        let geom = crate::prefill::BatchGeom::of(&config.text_config, 512);
        let charged = prefill_bytes(&config.text_config, Mode::Mma, 512, format).unwrap();
        let workspace = crate::prefill::workspace_bytes(Mode::Mma, &geom, format).unwrap();
        let gdn64 = crate::prefill::GdnPrefill::bytes(&geom, 64).unwrap();
        let gdn16 = crate::prefill::GdnPrefill::bytes(&geom, 16).unwrap();
        assert_eq!(
            charged as usize,
            workspace + gdn64,
            "prefill_bytes must charge the GDN workspace at chunk 64"
        );
        assert_eq!(
            charged as usize - (workspace + gdn16),
            gdn64 - gdn16,
            "the rise must be exactly bytes(64) − bytes(16)"
        );
    }

    #[test]
    fn attention_capacity_covers_larger_tail_grids() {
        use super::AttnPrefillPlan;
        let shared = AttnPrefillPlan::smem_bytes();
        let full = AttnPrefillPlan::compute(128, 4096 - 128, 64, shared, 128, 1, 4);
        let tail = AttnPrefillPlan::compute(120, 4096 - 120, 64, shared, 128, 1, 4);
        assert!(tail.rows() > full.rows());
        for (sms, active) in [(48, 1), (76, 1), (128, 1), (170, 2)] {
            for groups in [1, 4, 6] {
                let capacity = AttnPrefillPlan::capacity(8192, sms, active, groups).unwrap();
                for tokens in 1..=crate::prefill::BLOCK {
                    for base in [0, 1, 127, 4096, 8192 - tokens] {
                        let plan =
                            AttnPrefillPlan::compute(tokens, base, 64, shared, sms, active, groups);
                        assert!(plan.rows() <= capacity);
                        assert!(plan.chunk * plan.split >= base + tokens);
                        assert!(plan.qtiles * 8 >= tokens);
                    }
                }
            }
        }
    }

    #[test]
    fn packed_prefill_preserves_residency_across_budgets() {
        use super::{Mode, Residency, place, select_prefill};
        let projections = [100, 200, 300];
        for forced in [false, true] {
            for budget in 0..1000 {
                let plain = place(700, 250, &projections, 4, budget, forced);
                let candidate = place(750, 300, &projections, 4, budget, forced);
                let (mode, chosen) = select_prefill(Mode::Packed, plain, candidate);
                if plain.residency != Residency::Insufficient {
                    assert!(chosen.through >= plain.through);
                    assert!(chosen.bytes <= budget);
                }
                if mode == Mode::Packed {
                    assert_eq!(chosen.through, plain.through);
                }
            }
        }
        let plain = place(700, 250, &projections, 4, 720, false);
        let candidate = place(750, 300, &projections, 4, 720, false);
        let (mode, chosen) = select_prefill(Mode::Packed, plain, candidate);
        assert_eq!(mode, Mode::Gemv);
        assert_eq!(chosen.residency, Residency::Resident);
    }

    #[test]
    fn free_override_replaces_every_device() {
        assert_eq!(override_free(&[10, 20], None), vec![10, 20]);
        assert_eq!(override_free(&[10, 20], Some(7)), vec![7, 7]);
        assert!(override_free(&[], Some(7)).is_empty());
    }

    #[test]
    fn slot_ring_alternates_and_wraps_across_tokens() {
        for layer in 0..128usize {
            assert_eq!(StreamState::slot_of(layer), layer % 2);
        }
        assert_eq!(StreamState::slot_of(62), 0);
        assert_eq!(StreamState::slot_of(63), 1);
    }

    #[test]
    fn streamed_plan_constructs_and_prefills() {
        let Ok(ctx) = GpuContext::new(0) else {
            eprintln!("no CUDA device; skipping streamed-plan test");
            return;
        };
        drop(ctx);
        let Some(checkpoint) = checkpoint_dir("Qwen/Qwen3.8-27B-int4-rtn") else {
            eprintln!("FF_MODELS_DIR unset; skipping streamed-plan test");
            return;
        };
        if !checkpoint.is_dir() {
            eprintln!("no Qwen checkpoint; skipping streamed-plan test");
            return;
        }
        let weights = Qwen35Weights::open(&checkpoint).unwrap();
        let config = crate::config::Qwen35Config::from_model_dir(&checkpoint).unwrap();
        let streamed = QwenGpu::new(&[0], &weights, &config, true);
        let mut gpu = match streamed {
            Ok(gpu) => gpu,
            Err(e) => {
                let text = format!("{e:#}");
                assert!(
                    text.contains("short of"),
                    "streamed construction failed: {text}"
                );
                eprintln!("streamed plan does not fit; skipping streamed-plan test");
                return;
            }
        };
        let vocab = config.text_config.vocab_size as u32;
        let ids: Vec<u32> = (0..8u32).map(|i| 1000 + i * 997 % (vocab - 1000)).collect();
        gpu.push_tokens(&ids).unwrap();
        let token = gpu.read_token().unwrap();
        assert!(token < vocab);
    }

    #[test]
    fn split_balances_the_real_checkpoint_layers() {
        let Some(dir) = checkpoint_dir("Qwen/Qwen3.8-27B-int4-rtn") else {
            return;
        };
        let dir = &dir;
        if !dir.exists() {
            return;
        }
        let weights = crate::weights::Qwen35Weights::open(dir).unwrap();
        let config = crate::config::Qwen35Config::from_model_dir(dir).unwrap();
        let bytes: Vec<u64> = (0..config.text_config.num_hidden_layers)
            .map(|layer| layer_device_bytes(&weights, &config, layer).unwrap())
            .collect();
        let total: u64 = bytes.iter().sum();
        let ideal = total.div_ceil(2);
        let widest = *bytes.iter().max().unwrap();
        for parts in 2..=4usize {
            let ranges = split_layers_by_bytes(&bytes, parts);
            assert_eq!(ranges.len(), parts);
            assert_eq!(ranges[0].0, 0);
            assert_eq!(ranges.last().unwrap().1, bytes.len());
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].1, pair[1].0);
            }
            for &(start, end) in &ranges {
                assert!(end > start);
                assert!(bytes[start..end].iter().sum::<u64>() <= ideal + widest);
            }
        }
    }

    #[test]
    fn plan_residency_picks_a_hybrid_prefix_on_real_bytes() {
        let Some(dir) = checkpoint_dir("Qwen/Qwen3.8-27B-int4-rtn") else {
            return;
        };
        let dir = &dir;
        if !dir.exists() {
            return;
        }
        let weights = crate::weights::Qwen35Weights::open(dir).unwrap();
        let config = crate::config::Qwen35Config::from_model_dir(dir).unwrap();
        let ranges = super::partition_layers(&weights, &config, 1).unwrap();
        let plans = plan_residency(
            &weights,
            &config,
            PlanParams {
                ranges: &ranges,
                max_ctx: 4096,
                free: &[u64::MAX],
                ring_slots: 2,
                force_stream: false,
                granularity: 2 * 1024 * 1024,
            },
        )
        .unwrap();
        assert_eq!(plans[0].residency, Residency::Resident);
        assert_eq!(plans[0].resident_through, ranges[0].1);
        assert_eq!(plans[0].streaming_bytes, plans[0].resident_bytes);
        // free = 0 names the minimum footprint (nothing resident); the plan
        // also needs the derived admission reserve on top of it.
        let base = plan_residency(
            &weights,
            &config,
            PlanParams {
                ranges: &ranges,
                max_ctx: 4096,
                free: &[0],
                ring_slots: 2,
                force_stream: false,
                granularity: 2 * 1024 * 1024,
            },
        )
        .unwrap()[0]
            .streaming_bytes;
        let floor = base + ff_core::probe::device_admission_reserve_bytes();
        let plans = plan_residency(
            &weights,
            &config,
            PlanParams {
                ranges: &ranges,
                max_ctx: 4096,
                free: &[floor],
                ring_slots: 2,
                force_stream: false,
                granularity: 2 * 1024 * 1024,
            },
        )
        .unwrap();
        assert_eq!(plans[0].residency, Residency::Streaming);
        assert_eq!(plans[0].resident_through, ranges[0].0);
        assert!(plans[0].streaming_bytes <= floor);
        let plans = plan_residency(
            &weights,
            &config,
            PlanParams {
                ranges: &ranges,
                max_ctx: 4096,
                free: &[floor - 1],
                ring_slots: 2,
                force_stream: false,
                granularity: 2 * 1024 * 1024,
            },
        )
        .unwrap();
        assert_eq!(plans[0].residency, Residency::Insufficient);
        // The prefix grows with free memory and the plan always fits.
        let mut last = ranges[0].0;
        for free in [floor, floor + (1u64 << 30), floor + (4u64 << 30), u64::MAX] {
            let plan = &plan_residency(
                &weights,
                &config,
                PlanParams {
                    ranges: &ranges,
                    max_ctx: 4096,
                    free: &[free],
                    ring_slots: 2,
                    force_stream: false,
                    granularity: 2 * 1024 * 1024,
                },
            )
            .unwrap()[0];
            assert!(plan.resident_through >= last);
            assert!(plan.streaming_bytes <= free);
            last = plan.resident_through;
        }
        assert_eq!(last, ranges[0].1);
    }

    /// One deterministic 16-bit value per element; norms get +1 so the
    /// tiny model stays in a sane numerical range.
    fn tiny_value(index: usize, seed: u32, norm: bool) -> f32 {
        let v = ((index as f32) * 0.007 + seed as f32).sin() * 0.05;
        if norm { v + 1.0 } else { v }
    }

    /// A two-layer (one GDN, one full-attention) 16-bit checkpoint in a
    /// tempdir: config.json plus one safetensors shard. No CWD or process
    /// env changes — tests in one target share process state.
    fn write_tiny_checkpoint(dir: &std::path::Path, format: QuantFormat) -> Result<()> {
        let (dtype, encode): (safetensors::Dtype, fn(f32) -> [u8; 2]) = match format {
            QuantFormat::Bf16 => (safetensors::Dtype::BF16, |v| {
                half::bf16::from_f32(v).to_le_bytes()
            }),
            QuantFormat::F16 => (safetensors::Dtype::F16, |v| {
                half::f16::from_f32(v).to_le_bytes()
            }),
            QuantFormat::GroupAffine { .. } | QuantFormat::BlockFp8 { .. } => {
                anyhow::bail!("tiny checkpoint is 16-bit only")
            }
        };
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut put = |name: &str, shape: &[usize], seed: u32, norm: bool| {
            let count: usize = shape.iter().product();
            let mut bytes = Vec::with_capacity(count * 2);
            for i in 0..count {
                bytes.extend_from_slice(&encode(tiny_value(i, seed, norm)));
            }
            tensors.push((name.to_string(), shape.to_vec(), bytes));
        };
        let m = TEXT_PREFIX;
        put(&format!("{m}.embed_tokens.weight"), &[256, 256], 1, false);
        put("lm_head.weight", &[256, 256], 2, false);
        put(&format!("{m}.norm.weight"), &[256], 3, true);
        for (layer, gdn) in [(0usize, true), (1usize, false)] {
            let p = format!("{m}.layers.{layer}");
            let s = (layer * 10) as u32;
            put(&format!("{p}.input_layernorm.weight"), &[256], s + 1, true);
            put(
                &format!("{p}.post_attention_layernorm.weight"),
                &[256],
                s + 2,
                true,
            );
            put(
                &format!("{p}.mlp.gate_proj.weight"),
                &[256, 256],
                s + 3,
                false,
            );
            put(
                &format!("{p}.mlp.up_proj.weight"),
                &[256, 256],
                s + 4,
                false,
            );
            put(
                &format!("{p}.mlp.down_proj.weight"),
                &[256, 256],
                s + 5,
                false,
            );
            if gdn {
                put(
                    &format!("{p}.linear_attn.in_proj_qkv.weight"),
                    &[384, 256],
                    s + 6,
                    false,
                );
                put(
                    &format!("{p}.linear_attn.in_proj_z.weight"),
                    &[128, 256],
                    s + 7,
                    false,
                );
                put(
                    &format!("{p}.linear_attn.in_proj_b.weight"),
                    &[1, 256],
                    s + 8,
                    false,
                );
                put(
                    &format!("{p}.linear_attn.in_proj_a.weight"),
                    &[1, 256],
                    s + 9,
                    false,
                );
                put(
                    &format!("{p}.linear_attn.out_proj.weight"),
                    &[256, 128],
                    s + 10,
                    false,
                );
                put(
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &[384, 1, 4],
                    s + 11,
                    false,
                );
                put(&format!("{p}.linear_attn.A_log"), &[1], s + 12, false);
                put(&format!("{p}.linear_attn.dt_bias"), &[1], s + 13, false);
                put(
                    &format!("{p}.linear_attn.norm.weight"),
                    &[128],
                    s + 14,
                    true,
                );
            } else {
                put(
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[512, 256],
                    s + 6,
                    false,
                );
                put(
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[256, 256],
                    s + 7,
                    false,
                );
                put(
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[256, 256],
                    s + 8,
                    false,
                );
                put(
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[256, 256],
                    s + 9,
                    false,
                );
                put(
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[256],
                    s + 10,
                    true,
                );
                put(
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[256],
                    s + 11,
                    true,
                );
            }
        }
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                Ok((
                    name.clone(),
                    safetensors::tensor::TensorView::new(dtype, shape.clone(), bytes)?,
                ))
            })
            .collect::<Result<_, safetensors::SafeTensorError>>()?;
        safetensors::serialize_to_file(views, None, &dir.join("model-00001-of-00001.safetensors"))?;
        let weight_map: serde_json::Map<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, ..)| (name.clone(), "model-00001-of-00001.safetensors".into()))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 0},
                "weight_map": weight_map,
            }))?,
        )?;
        let config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5_text",
            "text_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "full_attention_interval": 2,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "intermediate_size": 256,
                "vocab_size": 256,
                "eos_token_id": [255],
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 4096
            }
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config)?)?;
        Ok(())
    }

    /// GPU decode/prefill of a tiny 16-bit checkpoint against the CPU host
    /// path: greedy token ids must match exactly, teacher-forced logits
    /// must agree on top-1 with a small max_abs.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn synthetic_16bit_checkpoint_gpu_matches_cpu() -> Result<()> {
        for format in [QuantFormat::Bf16, QuantFormat::F16] {
            let dir = tempfile::tempdir()?;
            write_tiny_checkpoint(dir.path(), format)?;
            let weights = Qwen35Weights::open(dir.path())?;
            assert_eq!(weights.format(), format);
            let config = crate::config::Qwen35Config::from_model_dir(dir.path())?;
            let prompt = [3u32, 11, 7];
            let mut cpu = crate::model::Qwen35Text::load(dir.path(), config.clone())?;
            let mut cpu_logits = Vec::new();
            let mut teacher_logits = Vec::new();
            for &token in &prompt {
                let (_, normed) = cpu.forward_raw(token)?;
                cpu_logits = cpu.logits(&normed)?;
                teacher_logits.push(cpu_logits.clone());
            }
            let mut cpu_ids = prompt.to_vec();
            for _ in 0..8 {
                let next = ff_core::math::argmax(&cpu_logits)?;
                cpu_ids.push(next);
                let (_, normed) = cpu.forward_raw(next)?;
                cpu_logits = cpu.logits(&normed)?;
                teacher_logits.push(cpu_logits.clone());
            }
            let mut gpu = QwenGpu::with_max_ctx(&[0], &weights, &config, 512, false)?;
            gpu.push_tokens(&prompt)?;
            let first = gpu.read_token()?;
            let decoded = gpu.decode(first, 8)?;
            let mut gpu_ids = prompt.to_vec();
            gpu_ids.extend_from_slice(&decoded);
            assert_eq!(gpu_ids, cpu_ids, "{format:?} greedy ids diverge");
            gpu.reset()?;
            let mut max_abs = 0f32;
            for (step, &token) in cpu_ids[..cpu_ids.len() - 1].iter().enumerate() {
                gpu.push_token(token)?;
                let got = gpu.read_logits()?;
                let expect = &teacher_logits[step];
                assert_eq!(
                    ff_core::math::argmax(&got)?,
                    ff_core::math::argmax(expect)?,
                    "{format:?} step {step} top-1 disagrees"
                );
                for (g, e) in got.iter().zip(expect) {
                    max_abs = max_abs.max((g - e).abs());
                }
            }
            assert!(
                max_abs <= 0.01,
                "{format:?} teacher-forced logits max_abs {max_abs}"
            );
        }
        Ok(())
    }

    /// Teacher-forced logits on the real bf16 checkpoint: a few steps of
    /// GPU read_logits against the CPU host path (max_abs, top-1).
    #[test]
    #[ignore = "requires a CUDA device and the real bf16 checkpoint"]
    fn real_bf16_teacher_forced_logits_match() -> Result<()> {
        let Some(dir) = checkpoint_dir("Qwen/Qwen3.8-27B") else {
            return Ok(());
        };
        let dir = &dir;
        if !dir.exists() {
            return Ok(());
        }
        let weights = Qwen35Weights::open(dir)?;
        assert_eq!(weights.format(), QuantFormat::Bf16);
        let config = crate::config::Qwen35Config::from_model_dir(dir)?;
        let ids = [1596u32, 1144, 310, 4087];
        let mut cpu = crate::model::Qwen35Text::load(dir, config.clone())?;
        let mut gpu = QwenGpu::with_max_ctx(&[0], &weights, &config, 4096, false)?;
        let mut max_abs = 0f32;
        for &token in &ids {
            let (_, normed) = cpu.forward_raw(token)?;
            let expect = cpu.logits(&normed)?;
            gpu.push_token(token)?;
            let got = gpu.read_logits()?;
            let (top_g, top_e) = (
                ff_core::math::argmax(&got)?,
                ff_core::math::argmax(&expect)?,
            );
            assert_eq!(top_g, top_e, "top-1 disagrees after token {token}");
            let step_max = got
                .iter()
                .zip(&expect)
                .map(|(g, e)| (g - e).abs())
                .fold(0f32, f32::max);
            max_abs = max_abs.max(step_max);
        }
        assert!(max_abs <= 0.05, "teacher-forced max_abs {max_abs}");
        Ok(())
    }

    /// The 16-bit slot ring end to end: prime and fetch publish each
    /// layer's exact bytes through the pinned staging and the H2D copy.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn streaming16_ring_delivers_exact_bytes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_tiny_checkpoint(dir.path(), QuantFormat::Bf16)?;
        let weights = Qwen35Weights::open(dir.path())?;
        let config = crate::config::Qwen35Config::from_model_dir(dir.path())?;
        let text = &config.text_config;
        let ctx = GpuContext::new(0)?;
        let geom = ring_geom(&ctx.context, &weights, &config)?;
        let mut state = StreamState16::build(&ctx, &weights, text, (0, 2), &geom)?;
        state.prime()?;
        state.prefetch.synchronize()?;
        for &suffix in super::streaming_names(LayerKind::LinearAttention) {
            let name = format!("{TEXT_PREFIX}.layers.0.{suffix}");
            let expect = weights.host_proj16(&name)?;
            let got = ctx.stream.clone_dtoh(&state.slot_buf(0, suffix)?.w)?;
            assert_eq!(
                got,
                expect.bytes(),
                "layer 0 slot bytes differ for {suffix}"
            );
        }
        state.begin_layer(&ctx, 0)?;
        state.end_layer(&ctx, 0)?;
        state.begin_layer(&ctx, 1)?;
        state.prefetch.synchronize()?;
        for &suffix in super::streaming_names(LayerKind::FullAttention) {
            let name = format!("{TEXT_PREFIX}.layers.1.{suffix}");
            let expect = weights.host_proj16(&name)?;
            let got = ctx.stream.clone_dtoh(&state.slot_buf(1, suffix)?.w)?;
            assert_eq!(
                got,
                expect.bytes(),
                "layer 1 slot bytes differ for {suffix}"
            );
        }
        state.end_layer(&ctx, 1)?;
        Ok(())
    }
}
