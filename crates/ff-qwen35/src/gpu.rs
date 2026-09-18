//! GPU decode for dense Qwen3.8-27B: the edge0 closed loop minus MoE.
//! One sync per token (the final token id read). All per-token-varying
//! values live in device buffers (position counter, next_token) so the
//! step is graph-capturable later.

use crate::config::{LayerKind, Qwen35Config, TEXT_PREFIX};
use crate::weights::Qwen35Weights;
use anyhow::{Context, Result};
use cudarc::driver::safe::CudaSlice;
use ff_edge0::gpu::{GpuContext, GpuGdn, GpuQuant};
use std::collections::HashMap;

type CudaSliceF = CudaSlice<f32>;
type CudaSliceI = CudaSlice<i32>;

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
}

impl QwenGpu {
    pub fn new(weights: &Qwen35Weights, config: &Qwen35Config) -> Result<Self> {
        Self::with_max_ctx(weights, config, 4096)
    }

    pub fn with_max_ctx(
        weights: &Qwen35Weights,
        config: &Qwen35Config,
        max_ctx: usize,
    ) -> Result<Self> {
        let ctx = GpuContext::new()?;
        let wide = crate::wide::WideKernels::load(&ctx)?;
        let text = &config.text_config;
        let hidden_size = text.hidden_size;
        let kv_stride = text.num_key_value_heads * text.head_dim;
        let conv_dim = text.conv_dim();
        let eps = text.rms_norm_eps as f32;

        let mut proj = HashMap::new();
        let mut ln = Vec::new();
        let mut attn_norms = Vec::new();
        let mut gdn = Vec::new();
        let mut kv_keys = Vec::new();
        let mut kv_values = Vec::new();
        for layer in 0..text.num_hidden_layers {
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
                        &ctx,
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
        let embed = weights.quant_projection(&format!("{TEXT_PREFIX}.embed_tokens"))?;
        let embed = ctx.upload(&embed, None)?;
        let lm = weights.quant_projection("lm_head")?;
        let lm = ctx.upload(&lm, None)?;
        proj.insert(format!("{TEXT_PREFIX}.embed_tokens"), embed);
        proj.insert("lm_head".to_string(), lm);

        // split=1 scratch for gate/up [intermediate, hidden]; split=4 for
        // down [hidden, intermediate] (2176 words/row, wpt<=3).
        let scr_gu = ctx.stream.alloc_zeros::<f32>(text.intermediate_size)?;
        let scr_down = ctx.stream.alloc_zeros::<f32>(4 * text.hidden_size)?;
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
            proj,
            ln,
            attn_norms,
            gdn,
            kv_keys,
            kv_values,
            position: 0,
            max_ctx,
            config: config.clone(),
            wide,
            scr_gu,
            scr_down,
        })
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
        if std::env::var_os("QWEN35_GRAPH").is_some() {
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
        let mut gdn_index = 0usize;
        let mut kv_index = 0usize;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
            // x1 enters holding rmsnorm_zc(hidden, ln_in).
            if text.layer_kind(layer) == LayerKind::LinearAttention {
                let qkv = self.get(&format!("{prefix}.linear_attn.in_proj_qkv"))?;
                let z = self.get(&format!("{prefix}.linear_attn.in_proj_z"))?;
                let b = self.get(&format!("{prefix}.linear_attn.in_proj_b"))?;
                let a = self.get(&format!("{prefix}.linear_attn.in_proj_a"))?;
                let out_proj = self.get(&format!("{prefix}.linear_attn.out_proj"))?;
                let segs = [qkv.group_seg(), z.group_seg(), b.group_seg(), a.group_seg()];
                self.wide.group(
                    &self.ctx,
                    &segs,
                    [qkv.y_ref(), z.y_ref(), b.y_ref(), a.y_ref()],
                    &self.x1,
                    &self.x1,
                    qkv.in_dim,
                    1,
                )?;
                let g = &self.gdn[gdn_index];
                self.ctx
                    .gdn_conv_heads(g, qkv.y_ref(), z.y_ref(), b.y_ref(), a.y_ref())?;
                let gout = self.ctx.gdn_out_buf(g);
                let segs = [
                    out_proj.group_seg(),
                    out_proj.empty_seg_like(),
                    out_proj.empty_seg_like(),
                    out_proj.empty_seg_like(),
                ];
                self.wide.group(
                    &self.ctx,
                    &segs,
                    [out_proj.y_ref(); 4],
                    gout,
                    gout,
                    out_proj.in_dim,
                    1,
                )?;
                self.ctx.glue_add_rmsnorm_zc(
                    &self.hidden,
                    out_proj.y_ref(),
                    &self.ln[layer][1],
                    &self.x1,
                    n,
                    eps,
                )?;
                gdn_index += 1;
            } else {
                let q = self.get(&format!("{prefix}.self_attn.q_proj"))?;
                let k = self.get(&format!("{prefix}.self_attn.k_proj"))?;
                let v = self.get(&format!("{prefix}.self_attn.v_proj"))?;
                let o = self.get(&format!("{prefix}.self_attn.o_proj"))?;
                let segs = [
                    q.group_seg(),
                    k.group_seg(),
                    v.group_seg(),
                    q.empty_seg_like(),
                ];
                self.wide.group(
                    &self.ctx,
                    &segs,
                    [q.y_ref(), k.y_ref(), v.y_ref(), q.y_ref()],
                    &self.x1,
                    &self.x1,
                    q.in_dim,
                    1,
                )?;
                let [qn, kn] = &self.attn_norms[kv_index];
                let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
                self.ctx.glue_attn_qk_zc_mrope(
                    q.y_ref(),
                    qn,
                    k.y_ref(),
                    kn,
                    v.y_ref(),
                    &self.q_out,
                    &self.gate_out,
                    &self.kv_keys[kv_index],
                    &self.kv_values[kv_index],
                    &self.pos,
                    &self.rope_pos,
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
                self.ctx.glue_attn_scores_raw(
                    &self.q_out,
                    &self.gate_out,
                    &self.kv_keys[kv_index],
                    &self.kv_values[kv_index],
                    &self.attn_out,
                    &self.pos,
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
                self.wide.group(
                    &self.ctx,
                    &segs,
                    [o.y_ref(); 4],
                    &self.attn_out,
                    &self.attn_out,
                    o.in_dim,
                    1,
                )?;
                self.ctx.glue_add_rmsnorm_zc(
                    &self.hidden,
                    o.y_ref(),
                    &self.ln[layer][1],
                    &self.x1,
                    n,
                    eps,
                )?;
                kv_index += 1;
            }
            // Dense MLP on x1, then the fused residual + next-layer's norm
            // (final_norm after the last layer).
            let gate = self.get(&format!("{prefix}.mlp.gate_proj"))?;
            let up = self.get(&format!("{prefix}.mlp.up_proj"))?;
            let down = self.get(&format!("{prefix}.mlp.down_proj"))?;
            let (gp, gs, gb) = gate.tensors();
            let (up_, us, ub) = up.tensors();
            self.wide.down(
                &self.ctx,
                gp,
                gs,
                gb,
                &self.x1,
                &self.x1,
                gate.y_ref(),
                gate.y_ref(),
                &self.scr_gu,
                &self.scr_gu,
                gate.out_dim,
                gate.in_dim,
                1,
                1,
            )?;
            self.wide.down(
                &self.ctx,
                up_,
                us,
                ub,
                &self.x1,
                &self.x1,
                up.y_ref(),
                up.y_ref(),
                &self.scr_gu,
                &self.scr_gu,
                up.out_dim,
                up.in_dim,
                1,
                1,
            )?;
            self.ctx.silu_mul(
                gate.y_ref(),
                up.y_ref(),
                &self.inner,
                text.intermediate_size,
            )?;
            let (dp, ds, db) = down.tensors();
            self.wide.down(
                &self.ctx,
                dp,
                ds,
                db,
                &self.inner,
                &self.inner,
                down.y_ref(),
                down.y_ref(),
                &self.scr_down,
                &self.scr_down,
                down.out_dim,
                down.in_dim,
                4,
                1,
            )?;
            let next_w = if layer + 1 < text.num_hidden_layers {
                &self.ln[layer + 1][0]
            } else {
                &self.final_norm_w
            };
            self.ctx
                .glue_add_rmsnorm_zc(&self.hidden, down.y_ref(), next_w, &self.x1, n, eps)?;
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
