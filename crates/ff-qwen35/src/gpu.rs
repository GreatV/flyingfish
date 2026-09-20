//! GPU decode for dense Qwen3.8-27B: the edge0 closed loop minus MoE.
//! One sync per token (the final token id read). All per-token-varying
//! values live in device buffers (position counter, next_token) so the
//! step is graph-capturable later. Multiple ordinals partition the layer
//! stack across devices; hidden and x1 hop between devices through host
//! staging (one blocking copy per hop).

use crate::config::{LayerKind, Qwen35Config, TEXT_PREFIX, TextConfig};
use crate::weights::Qwen35Weights;
use anyhow::{Context, Result, ensure};
use cudarc::driver::safe::CudaSlice;
use ff_edge0::gpu::{GpuContext, GpuGdn, GpuQuant};
use std::collections::HashMap;
use std::ops::Range;

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
    match text.layer_kind(layer) {
        LayerKind::LinearAttention => {
            for p in [
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ] {
                total += weights.tensor_device_bytes(&format!("{prefix}.linear_attn.{p}"))?;
            }
            for p in ["conv1d", "A_log", "dt_bias", "norm"] {
                total += weights.tensor_device_bytes(&format!("{prefix}.linear_attn.{p}"))?;
            }
        }
        LayerKind::FullAttention => {
            for p in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                total += weights.tensor_device_bytes(&format!("{prefix}.self_attn.{p}"))?;
            }
        }
    }
    for p in ["gate_proj", "up_proj", "down_proj"] {
        total += weights.tensor_device_bytes(&format!("{prefix}.mlp.{p}"))?;
    }
    Ok(total)
}

/// On-device bytes for the device-0 extras: embeddings, lm_head, final
/// norm, and the token/position counters.
pub fn static_device_bytes(weights: &Qwen35Weights, _config: &Qwen35Config) -> Result<u64> {
    let counters = 6 * std::mem::size_of::<i32>() as u64;
    Ok(
        weights.tensor_device_bytes(&format!("{TEXT_PREFIX}.embed_tokens"))?
            + weights.tensor_device_bytes("lm_head")?
            + weights.tensor_device_bytes(&format!("{TEXT_PREFIX}.norm"))?
            + counters,
    )
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
    let mut required = stack_scratch_bytes(text);
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

/// Contiguous layer ranges, one per device, boundaries placed closest to
/// each device's share of the total bytes.
pub(crate) fn split_layers_by_bytes(layer_bytes: &[u64], parts: usize) -> Vec<(usize, usize)> {
    let total: u64 = layer_bytes.iter().sum();
    let mut ranges = Vec::with_capacity(parts);
    let mut start = 0usize;
    for d in 1..parts {
        let target = total * d as u64 / parts as u64;
        let last_end = layer_bytes.len() - (parts - d);
        let mut cum: u64 = layer_bytes[..start + 1].iter().sum();
        let mut best = start + 1;
        let mut best_err = cum.abs_diff(target);
        let mut end = start + 2;
        while end <= last_end {
            cum += layer_bytes[end - 1];
            let err = cum.abs_diff(target);
            if err >= best_err {
                break;
            }
            best_err = err;
            best = end;
            end += 1;
        }
        ranges.push((start, best));
        start = best;
    }
    ranges.push((start, layer_bytes.len()));
    ranges
}

fn upload_boundary(
    ctx: &GpuContext,
    weights: &Qwen35Weights,
    first_layer: usize,
) -> Result<CudaSliceF> {
    let name = format!("{TEXT_PREFIX}.layers.{first_layer}.input_layernorm.weight");
    ctx.upload_f32(&weights.f32_named(&name)?)
}

fn preflight(
    contexts: &[GpuContext],
    ordinals: &[usize],
    ranges: &[(usize, usize)],
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    max_ctx: usize,
) -> Result<()> {
    let mut report = String::new();
    let mut fits = true;
    for (i, ctx) in contexts.iter().enumerate() {
        let (start, end) = ranges[i];
        let mut required = range_device_bytes(weights, config, start..end, max_ctx)?;
        if i == 0 {
            required += static_device_bytes(weights, config)?;
        }
        let free = ctx.context.mem_get_info().context("mem_get_info")?.0;
        if required > free as u64 {
            fits = false;
        }
        report.push_str(&format!(
            "device {}: layers {start}..{end} need {required} B, {free} B free\n",
            ordinals[i]
        ));
    }
    ensure!(
        fits,
        "device memory short of the partitioned plan:\n{report}pass more device ordinals to spread the layer stack"
    );
    Ok(())
}

struct LayerUploads {
    proj: HashMap<String, GpuQuant>,
    ln: Vec<[CudaSliceF; 2]>,
    attn_norms: Vec<[CudaSliceF; 2]>,
    gdn: Vec<GpuGdn>,
    kv_keys: Vec<CudaSliceF>,
    kv_values: Vec<CudaSliceF>,
}

fn upload_stack(
    ctx: &GpuContext,
    weights: &Qwen35Weights,
    text: &TextConfig,
    layers: Range<usize>,
    max_ctx: usize,
) -> Result<LayerUploads> {
    let kv_stride = text.num_key_value_heads * text.head_dim;
    let conv_dim = text.conv_dim();
    let eps = text.rms_norm_eps as f32;
    let mut proj = HashMap::new();
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
                for p in [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ] {
                    let name = format!("{prefix}.linear_attn.{p}");
                    let q = weights.quant_projection(&name)?;
                    proj.insert(name, ctx.upload(&q, None)?);
                }
                gdn.push(GpuGdn::upload(
                    ctx,
                    &weights.f32_named(&format!("{prefix}.linear_attn.conv1d.weight"))?,
                    &weights.f32_named(&format!("{prefix}.linear_attn.A_log"))?,
                    &weights.f32_named(&format!("{prefix}.linear_attn.dt_bias"))?,
                    &weights.f32_named(&format!("{prefix}.linear_attn.norm.weight"))?,
                    conv_dim,
                    text.linear_conv_kernel_dim,
                    text.linear_num_value_heads,
                    text.linear_num_key_heads,
                    text.linear_key_head_dim,
                    text.linear_value_head_dim,
                    eps,
                )?);
            }
            LayerKind::FullAttention => {
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    let name = format!("{prefix}.self_attn.{p}");
                    let q = weights.quant_projection(&name)?;
                    proj.insert(name, ctx.upload(&q, None)?);
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
        for p in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("{prefix}.mlp.{p}");
            let q = weights.quant_projection(&name)?;
            proj.insert(name, ctx.upload(&q, None)?);
        }
    }
    Ok(LayerUploads {
        proj,
        ln,
        attn_norms,
        gdn,
        kv_keys,
        kv_values,
    })
}

pub(crate) struct PeerGpu {
    ctx: GpuContext,
    wide: crate::wide::WideKernels,
    proj: HashMap<String, GpuQuant>,
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
    scr_gu: CudaSliceF,
    scr_down: CudaSliceF,
    pos: CudaSliceI,
    rope_pos: CudaSliceI,
    boundary_norm: CudaSliceF,
}

pub struct QwenGpu {
    pub(crate) config: Qwen35Config,
    /// Spec-mode access (the verify driver pokes these directly).
    pub ctx: GpuContext,
    pub(crate) proj: HashMap<String, GpuQuant>,
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
    /// Spec-mode access (the verify driver pokes these directly).
    pub next_token: CudaSliceI,
    /// Captured decode-step graph (QWEN35_GRAPH=1; pos/next_token are
    /// device-read so the step replays without host input).
    decode_graph: Option<ff_edge0::gpu::DecodeGraph>,
    /// Spare for the borrow swap at finalize (argmax needs &mut next_token
    /// while lm_head's y borrows the projection map).
    nt_spare: CudaSliceI,
    /// Host mirror of the position counter (prefill loop bookkeeping).
    pub position: usize,
    /// KV-cache capacity; step() refuses to write past it.
    max_ctx: usize,
    /// Wide split-K GEMVs for the dense-MLP whale shapes (wide.rs): the
    /// in-4096 chunk primitive is occupancy-starved there (cold 161 GB/s
    /// on down vs 426 split-K — wide_bw).
    pub(crate) wide: crate::wide::WideKernels,
    scr_gu: CudaSliceF,
    scr_down: CudaSliceF,
    /// Layer-stack hosts for ordinals[1..]; empty on a single device.
    pub(crate) peers: Vec<PeerGpu>,
    /// dev0's boundary norm when peers exist; the final norm otherwise.
    pub(crate) boundary_norms: Vec<CudaSliceF>,
    /// Layer range per device (peers included), bytes-balanced.
    pub(crate) ranges: Vec<(usize, usize)>,
    /// Host buffer for the hidden/x1 device hops.
    staging: Vec<f32>,
}

impl QwenGpu {
    pub fn new(ordinals: &[usize], weights: &Qwen35Weights, config: &Qwen35Config) -> Result<Self> {
        Self::with_max_ctx(ordinals, weights, config, 4096)
    }

    pub fn with_max_ctx(
        ordinals: &[usize],
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        max_ctx: usize,
    ) -> Result<Self> {
        ensure!(!ordinals.is_empty(), "at least one device ordinal required");
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
        preflight(&contexts, ordinals, &ranges, weights, config, max_ctx)?;
        let text = &config.text_config;
        let mut contexts = contexts.into_iter();
        let ctx = contexts.next().expect("ordinals checked non-empty");
        let wide = crate::wide::WideKernels::load(&ctx)?;
        let hidden_size = text.hidden_size;

        let mut stack = upload_stack(&ctx, weights, text, ranges[0].0..ranges[0].1, max_ctx)?;
        let embed = weights.quant_projection(&format!("{TEXT_PREFIX}.embed_tokens"))?;
        let embed = ctx.upload(&embed, None)?;
        let lm = weights.quant_projection("lm_head")?;
        let lm = ctx.upload(&lm, None)?;
        stack
            .proj
            .insert(format!("{TEXT_PREFIX}.embed_tokens"), embed);
        stack.proj.insert("lm_head".to_string(), lm);

        // split=1 scratch for gate/up [intermediate, hidden]; split=4 for
        // down [hidden, intermediate] (2176 words/row, wpt<=3).
        let scr_gu = ctx.stream.alloc_zeros::<f32>(text.intermediate_size)?;
        let scr_down = ctx.stream.alloc_zeros::<f32>(4 * text.hidden_size)?;

        let mut peers = Vec::new();
        let mut boundary_norms = Vec::new();
        if ranges.len() > 1 {
            boundary_norms.push(upload_boundary(&ctx, weights, ranges[1].0)?);
        }
        for (i, pctx) in contexts.enumerate() {
            let pwide = crate::wide::WideKernels::load(&pctx)?;
            let pstack = upload_stack(
                &pctx,
                weights,
                text,
                ranges[i + 1].0..ranges[i + 1].1,
                max_ctx,
            )?;
            let boundary_norm = if i + 2 < ranges.len() {
                upload_boundary(&pctx, weights, ranges[i + 2].0)?
            } else {
                pctx.upload_f32(&weights.f32_named(&format!("{TEXT_PREFIX}.norm.weight"))?)?
            };
            peers.push(PeerGpu {
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
                scr_gu: pctx
                    .stream
                    .alloc_zeros::<f32>(text.intermediate_size)
                    .context("scr_gu")?,
                scr_down: pctx
                    .stream
                    .alloc_zeros::<f32>(4 * text.hidden_size)
                    .context("scr_down")?,
                hidden: pctx
                    .stream
                    .alloc_zeros::<f32>(hidden_size)
                    .context("hidden")?,
                x1: pctx.stream.alloc_zeros::<f32>(hidden_size).context("x1")?,
                pos: pctx.upload_i32(&[0])?,
                rope_pos: pctx.upload_i32(&[0, 0, 0])?,
                boundary_norm,
                ctx: pctx,
                wide: pwide,
                proj: pstack.proj,
                ln: pstack.ln,
                attn_norms: pstack.attn_norms,
                gdn: pstack.gdn,
                kv_keys: pstack.kv_keys,
                kv_values: pstack.kv_values,
            });
        }

        Ok(Self {
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
            next_token: ctx.upload_i32(&[0])?,
            nt_spare: ctx.upload_i32(&[0])?,
            decode_graph: None,
            ctx,
            proj: stack.proj,
            ln: stack.ln,
            attn_norms: stack.attn_norms,
            gdn: stack.gdn,
            kv_keys: stack.kv_keys,
            kv_values: stack.kv_values,
            position: 0,
            max_ctx,
            config: config.clone(),
            wide,
            scr_gu,
            scr_down,
            peers,
            boundary_norms,
            ranges,
            staging: vec![0f32; 2 * hidden_size],
        })
    }

    /// The layer range each device ordinal hosts.
    pub fn layer_ranges(&self) -> &[(usize, usize)] {
        &self.ranges
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
        anyhow::ensure!(
            self.position < self.max_ctx,
            "position {} reached max_ctx {} (KV cache capacity)",
            self.position,
            self.max_ctx
        );
        // QWEN35_GRAPH=1: capture the step once, replay after. pos and
        // next_token are device-read, so the step needs no host input.
        // Host staging in the multi-device path cannot be captured.
        if self.peers.is_empty() && std::env::var_os("QWEN35_GRAPH").is_some() {
            if let Some(g) = &self.decode_graph {
                g.0.launch().context("decode graph replay")?;
                self.position += 1;
                return Ok(());
            }
            self.ctx
                .stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL,
                )
                .context("capture begin")?;
            let enq = self.step_inner();
            match enq {
                Ok(()) => {
                    let g = self
                        .ctx
                        .stream
                        .end_capture(
                            cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                        )
                        .context("capture end")?
                        .context("capture produced no graph")?;
                    let g = ff_edge0::gpu::DecodeGraph(g);
                    // Capture only records; the captured step has not run.
                    // Launch once so this step actually executes (device pos
                    // and next_token advance). Host position was already
                    // incremented by step_inner — do not increment again.
                    g.0.launch().context("decode graph first launch")?;
                    self.decode_graph = Some(g);
                }
                Err(e) => {
                    let _ = self.ctx.stream.end_capture(
                        cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    );
                    return Err(e);
                }
            }
            return Ok(());
        }
        self.step_inner()
    }

    fn step_inner(&mut self) -> Result<()> {
        self.embed_from_next_token()?;
        self.step_layers()
    }

    fn embed_from_next_token(&mut self) -> Result<()> {
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
            &StackView {
                ctx: &self.ctx,
                wide: &self.wide,
                proj: &self.proj,
                ln: &self.ln,
                attn_norms: &self.attn_norms,
                gdn: &self.gdn,
                kv_keys: &self.kv_keys,
                kv_values: &self.kv_values,
                q_out: &self.q_out,
                gate_out: &self.gate_out,
                attn_out: &self.attn_out,
                inner: &self.inner,
                scr_gu: &self.scr_gu,
                scr_down: &self.scr_down,
                pos: &self.pos,
                rope_pos: &self.rope_pos,
                hidden: &self.hidden,
                x1: &self.x1,
                boundary_norm: self.boundary_norms.first().unwrap_or(&self.final_norm_w),
            },
            self.ranges[0].0..self.ranges[0].1,
            &text,
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
            {
                let peer = &mut self.peers[i];
                peer.ctx
                    .stream
                    .memcpy_htod(&self.staging[..n], &mut peer.hidden)?;
                peer.ctx
                    .stream
                    .memcpy_htod(&self.staging[n..], &mut peer.x1)?;
            }
            let (start, end) = self.ranges[i + 1];
            let peer = &self.peers[i];
            run_stack(
                &StackView {
                    ctx: &peer.ctx,
                    wide: &peer.wide,
                    proj: &peer.proj,
                    ln: &peer.ln,
                    attn_norms: &peer.attn_norms,
                    gdn: &peer.gdn,
                    kv_keys: &peer.kv_keys,
                    kv_values: &peer.kv_values,
                    q_out: &peer.q_out,
                    gate_out: &peer.gate_out,
                    attn_out: &peer.attn_out,
                    inner: &peer.inner,
                    scr_gu: &peer.scr_gu,
                    scr_down: &peer.scr_down,
                    pos: &peer.pos,
                    rope_pos: &peer.rope_pos,
                    hidden: &peer.hidden,
                    x1: &peer.x1,
                    boundary_norm: &peer.boundary_norm,
                },
                start..end,
                &text,
            )?;
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
            &self.x1,
            &self.x1,
            lm.in_dim,
            1,
        )?;
        let lm_out = lm.out_dim;
        std::mem::swap(&mut self.next_token, &mut self.nt_spare);
        self.ctx
            .glue_argmax(lm.y_ref(), &mut self.nt_spare, lm_out)?;
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
        self.ctx
            .stream
            .memcpy_htod(&[token as i32], &mut self.next_token)?;
        self.ctx.stream.memcpy_htod(&pos3, &mut self.rope_pos)?;
        for peer in &mut self.peers {
            peer.ctx.stream.memcpy_htod(&pos3, &mut peer.rope_pos)?;
        }
        self.step()
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

    /// The generated token id (the one sync per token).
    pub fn read_token(&self) -> Result<u32> {
        let mut id = [0i32];
        self.ctx.stream.memcpy_dtoh(&self.next_token, &mut id)?;
        self.ctx.counted_sync()?;
        Ok(id[0] as u32)
    }
}

struct StackView<'a> {
    ctx: &'a GpuContext,
    wide: &'a crate::wide::WideKernels,
    proj: &'a HashMap<String, GpuQuant>,
    ln: &'a [[CudaSliceF; 2]],
    attn_norms: &'a [[CudaSliceF; 2]],
    gdn: &'a [GpuGdn],
    kv_keys: &'a [CudaSliceF],
    kv_values: &'a [CudaSliceF],
    q_out: &'a CudaSliceF,
    gate_out: &'a CudaSliceF,
    attn_out: &'a CudaSliceF,
    inner: &'a CudaSliceF,
    scr_gu: &'a CudaSliceF,
    scr_down: &'a CudaSliceF,
    pos: &'a CudaSliceI,
    rope_pos: &'a CudaSliceI,
    hidden: &'a CudaSliceF,
    x1: &'a CudaSliceF,
    boundary_norm: &'a CudaSliceF,
}

fn stack_get<'a>(proj: &'a HashMap<String, GpuQuant>, name: &str) -> Result<&'a GpuQuant> {
    proj.get(name)
        .with_context(|| format!("{name} not resident"))
}

/// Attention + MLP for one device's contiguous layer range. `x1` enters
/// normed for the range's first layer and leaves normed by the range's
/// boundary norm; `hidden` carries the residual across.
fn run_stack(view: &StackView, layers: Range<usize>, text: &TextConfig) -> Result<()> {
    let eps = text.rms_norm_eps as f32;
    let n = text.hidden_size;
    let mut gdn_index = 0usize;
    let mut kv_index = 0usize;
    let start = layers.start;
    let end = layers.end;
    for layer in layers {
        let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
        let local = layer - start;
        // x1 enters holding rmsnorm_zc(hidden, ln_in).
        if text.layer_kind(layer) == LayerKind::LinearAttention {
            let qkv = stack_get(view.proj, &format!("{prefix}.linear_attn.in_proj_qkv"))?;
            let z = stack_get(view.proj, &format!("{prefix}.linear_attn.in_proj_z"))?;
            let b = stack_get(view.proj, &format!("{prefix}.linear_attn.in_proj_b"))?;
            let a = stack_get(view.proj, &format!("{prefix}.linear_attn.in_proj_a"))?;
            let out_proj = stack_get(view.proj, &format!("{prefix}.linear_attn.out_proj"))?;
            let segs = [qkv.group_seg(), z.group_seg(), b.group_seg(), a.group_seg()];
            view.wide.group(
                view.ctx,
                &segs,
                [qkv.y_ref(), z.y_ref(), b.y_ref(), a.y_ref()],
                view.x1,
                view.x1,
                qkv.in_dim,
                1,
            )?;
            let g = &view.gdn[gdn_index];
            view.ctx
                .gdn_conv_heads(g, qkv.y_ref(), z.y_ref(), b.y_ref(), a.y_ref())?;
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
                gout,
                gout,
                out_proj.in_dim,
                1,
            )?;
            view.ctx.glue_add_rmsnorm_zc(
                view.hidden,
                out_proj.y_ref(),
                &view.ln[local][1],
                view.x1,
                n,
                eps,
            )?;
            gdn_index += 1;
        } else {
            let q = stack_get(view.proj, &format!("{prefix}.self_attn.q_proj"))?;
            let k = stack_get(view.proj, &format!("{prefix}.self_attn.k_proj"))?;
            let v = stack_get(view.proj, &format!("{prefix}.self_attn.v_proj"))?;
            let o = stack_get(view.proj, &format!("{prefix}.self_attn.o_proj"))?;
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
                view.x1,
                view.x1,
                q.in_dim,
                1,
            )?;
            let [qn, kn] = &view.attn_norms[kv_index];
            let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
            view.ctx.glue_attn_qk_zc_mrope(
                q.y_ref(),
                qn,
                k.y_ref(),
                kn,
                v.y_ref(),
                view.q_out,
                view.gate_out,
                &view.kv_keys[kv_index],
                &view.kv_values[kv_index],
                view.pos,
                view.rope_pos,
                text.num_key_value_heads * text.head_dim,
                text.num_attention_heads,
                text.num_key_value_heads,
                text.head_dim,
                rotary_dim,
                text.rope.rope_theta,
                text.rope.mrope_section[1],
                text.rope.mrope_section[2],
            )?;
            let scale = 1.0 / (text.head_dim as f32).sqrt();
            view.ctx.glue_attn_scores_raw(
                view.q_out,
                view.gate_out,
                &view.kv_keys[kv_index],
                &view.kv_values[kv_index],
                view.attn_out,
                view.pos,
                text.num_key_value_heads * text.head_dim,
                text.num_attention_heads,
                text.num_key_value_heads,
                text.head_dim,
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
                view.attn_out,
                view.attn_out,
                o.in_dim,
                1,
            )?;
            view.ctx.glue_add_rmsnorm_zc(
                view.hidden,
                o.y_ref(),
                &view.ln[local][1],
                view.x1,
                n,
                eps,
            )?;
            kv_index += 1;
        }
        // Dense MLP on x1, then the fused residual + next-layer's norm
        // (the boundary norm after the range's last layer).
        let gate = stack_get(view.proj, &format!("{prefix}.mlp.gate_proj"))?;
        let up = stack_get(view.proj, &format!("{prefix}.mlp.up_proj"))?;
        let down = stack_get(view.proj, &format!("{prefix}.mlp.down_proj"))?;
        let (gp, gs, gb) = gate.tensors();
        let (up_, us, ub) = up.tensors();
        view.wide.down(
            view.ctx,
            gp,
            gs,
            gb,
            view.x1,
            view.x1,
            gate.y_ref(),
            gate.y_ref(),
            view.scr_gu,
            view.scr_gu,
            gate.out_dim,
            gate.in_dim,
            1,
            1,
        )?;
        view.wide.down(
            view.ctx,
            up_,
            us,
            ub,
            view.x1,
            view.x1,
            up.y_ref(),
            up.y_ref(),
            view.scr_gu,
            view.scr_gu,
            up.out_dim,
            up.in_dim,
            1,
            1,
        )?;
        view.ctx
            .silu_mul(gate.y_ref(), up.y_ref(), view.inner, text.intermediate_size)?;
        let (dp, ds, db) = down.tensors();
        view.wide.down(
            view.ctx,
            dp,
            ds,
            db,
            view.inner,
            view.inner,
            down.y_ref(),
            down.y_ref(),
            view.scr_down,
            view.scr_down,
            down.out_dim,
            down.in_dim,
            4,
            1,
        )?;
        let next_w: &CudaSliceF = if layer + 1 < end {
            &view.ln[local + 1][0]
        } else {
            view.boundary_norm
        };
        view.ctx
            .glue_add_rmsnorm_zc(view.hidden, down.y_ref(), next_w, view.x1, n, eps)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{layer_device_bytes, split_layers_by_bytes};

    #[test]
    fn split_covers_all_layers_contiguously() {
        let uniform: Vec<u64> = vec![100; 64];
        for parts in 1..=4usize {
            let ranges = split_layers_by_bytes(&uniform, parts);
            assert_eq!(ranges.len(), parts);
            assert_eq!(ranges[0].0, 0);
            assert_eq!(ranges.last().unwrap().1, 64);
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].1, pair[1].0);
            }
            for &(start, end) in &ranges {
                assert!(end > start);
            }
            if 64u64.is_multiple_of(parts as u64) {
                for &(start, end) in &ranges {
                    assert_eq!(end - start, 64 / parts);
                }
            }
        }
    }

    #[test]
    fn split_balances_lumpy_layers() {
        let lumpy = [10u64, 10, 10, 10, 10, 100];
        let ranges = split_layers_by_bytes(&lumpy, 2);
        assert_eq!(ranges, vec![(0, 5), (5, 6)]);
        let total: u64 = lumpy.iter().sum();
        let ideal = total.div_ceil(2);
        let widest = *lumpy.iter().max().unwrap();
        for &(start, end) in &ranges {
            assert!(lumpy[start..end].iter().sum::<u64>() <= ideal + widest);
        }

        let head = [100u64, 1, 1, 1, 1];
        assert_eq!(split_layers_by_bytes(&head, 2), vec![(0, 1), (1, 5)]);
    }

    #[test]
    fn split_balances_the_real_checkpoint_layers() {
        let dir = std::path::Path::new("../../models/Qwen/Qwen3.8-27B-int4");
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
}
