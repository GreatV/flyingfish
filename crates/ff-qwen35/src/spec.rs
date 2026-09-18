//! MTP speculative decode: batch-2 verify rounds over the batch2.cu
//! kernels (weights read once per token pair). Reject restores GDN state
//! from the post-A scratch via dtod — no recompute; KV needs nothing
//! (write slots derive from the position counters).
//!
//! Timelines: the main model uses `gpu.pos`; the draft position rides
//! `pos_b` (= pos + 1, uploaded per round). The MTP layer keeps its OWN
//! 0-based KV timeline (`mtp_pos`): rope angles shift by a constant
//! against the main timeline, which preserves relative angles — the
//! acceptance measurement arbitrates draft quality anyway.

use crate::config::{LayerKind, TEXT_PREFIX};
use crate::gpu::QwenGpu;
use anyhow::{Context, Result};
use cudarc::driver::safe::{CudaSlice, LaunchConfig, PushKernelArg};
use ff_edge0::gpu::{GpuQuant, GroupSeg};

pub struct QwenSpec {
    conv2: cudarc::driver::safe::CudaFunction,
    heads2: cudarc::driver::safe::CudaFunction,
    // B-side (draft token) buffers
    pub hidden_b: CudaSlice<f32>,
    x1_b: CudaSlice<f32>,
    q_out_b: CudaSlice<f32>,
    gate_out_b: CudaSlice<f32>,
    attn_out_b: CudaSlice<f32>,
    inner_b: CudaSlice<f32>,
    /// Per-projection second output buffer (the shared GpuQuant y is A's).
    y_b: std::collections::HashMap<String, CudaSlice<f32>>,
    /// Shared split-K scratch (sequential same-stream calls may share).
    scr: CudaSlice<f32>,
    /// cols=2 needs a SEPARATE B scratch (concurrent columns race on one).
    scr_b: CudaSlice<f32>,
    pos_b: CudaSlice<i32>,
    out_a: CudaSlice<i32>,
    out_b: CudaSlice<i32>,
    // GDN post-A scratch (reject restore source)
    conv_scratch: Vec<CudaSlice<f32>>,
    rec_scratch: Vec<CudaSlice<f32>>,
    gdn_conv_out_a: Vec<CudaSlice<f32>>,
    gdn_out_a: Vec<CudaSlice<f32>>,
    gdn_conv_out_b: Vec<CudaSlice<f32>>,
    gdn_out_b: Vec<CudaSlice<f32>>,
    // MTP draft side (own 0-based KV timeline)
    mtp_q_out: CudaSlice<f32>,
    mtp_gate_out: CudaSlice<f32>,
    mtp_attn_out: CudaSlice<f32>,
    mtp_inner: CudaSlice<f32>,
    mtp_x1: CudaSlice<f32>,
    mtp_cat: CudaSlice<f32>,
    /// Persistent draft-layer residual stream (fc writes straight in).
    mtp_res: CudaSlice<f32>,
    mtp_embed: CudaSlice<f32>,
    mtp_kv_keys: CudaSlice<f32>,
    mtp_kv_values: CudaSlice<f32>,
    mtp_pos: CudaSlice<i32>,
    mtp_tok: CudaSlice<i32>,
    pub nt_draft: CudaSlice<i32>,
    mtp_proj: std::collections::HashMap<String, GpuQuant>,
    mtp_pre_embed_w: CudaSlice<f32>,
    mtp_pre_hidden_w: CudaSlice<f32>,
    mtp_in_norm: CudaSlice<f32>,
    mtp_post_norm: CudaSlice<f32>,
    mtp_qnorm: CudaSlice<f32>,
    mtp_knorm: CudaSlice<f32>,
    mtp_final_norm: CudaSlice<f32>,
    concat: cudarc::driver::safe::CudaFunction,
    /// Host mirror of the draft id written to `nt_draft`.
    pub draft_id: u32,
}

impl QwenSpec {
    pub fn new(gpu: &QwenGpu, weights: &crate::weights::Qwen35Weights) -> Result<Self> {
        let ctx = &gpu.ctx;
        let text = &gpu.config.text_config;
        let n = text.hidden_size;
        let q_dim = text.num_attention_heads * text.head_dim;
        let conv_dim = text.conv_dim();
        let kv_stride = text.num_key_value_heads * text.head_dim;
        let max_ctx = gpu
            .kv_keys
            .first()
            .map(|k| k.len() / kv_stride)
            .unwrap_or(4096);

        let ptx = cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
            env!("OUT_DIR"),
            "/qwen_batch2.ptx"
        )));
        let module = ctx
            .context
            .load_module(ptx)
            .map_err(|e| anyhow::anyhow!("batch2 module: {e:?}"))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))
        };
        let conv2 = load("qwen_gdn_conv2")?;
        let heads2 = load("qwen_gdn_heads2")?;

        let mut y_b = std::collections::HashMap::new();
        for (name, q) in &gpu.proj {
            y_b.insert(
                name.clone(),
                ctx.stream.alloc_zeros::<f32>(q.out_dim).context("y_b")?,
            );
        }
        let gdn_layers = (0..text.num_hidden_layers)
            .filter(|&l| text.layer_kind(l) == LayerKind::LinearAttention)
            .count();
        let mk = |len: usize| {
            (0..gdn_layers)
                .map(|_| {
                    ctx.stream
                        .alloc_zeros::<f32>(len)
                        .map_err(anyhow::Error::from)
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let rec_len =
            text.linear_num_value_heads * text.linear_key_head_dim * text.linear_value_head_dim;
        let mut mtp_proj = std::collections::HashMap::new();
        for name in [
            "mtp.fc".to_string(),
            "mtp.layers.0.self_attn.q_proj".into(),
            "mtp.layers.0.self_attn.k_proj".into(),
            "mtp.layers.0.self_attn.v_proj".into(),
            "mtp.layers.0.self_attn.o_proj".into(),
            "mtp.layers.0.mlp.gate_proj".into(),
            "mtp.layers.0.mlp.up_proj".into(),
            "mtp.layers.0.mlp.down_proj".into(),
        ] {
            let q = weights.quant_projection(&name)?;
            mtp_proj.insert(name, ctx.upload(&q, None)?);
        }
        let concat = load("qwen_concat")?;
        Ok(Self {
            conv2,
            heads2,
            hidden_b: ctx.stream.alloc_zeros::<f32>(n)?,
            x1_b: ctx.stream.alloc_zeros::<f32>(n)?,
            q_out_b: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            gate_out_b: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            attn_out_b: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            inner_b: ctx.stream.alloc_zeros::<f32>(text.intermediate_size)?,
            y_b,
            scr: ctx
                .stream
                .alloc_zeros::<f32>(text.intermediate_size.max(4 * text.hidden_size))
                .context("splitk scratch")?,
            scr_b: ctx
                .stream
                .alloc_zeros::<f32>(text.intermediate_size.max(4 * text.hidden_size))
                .context("splitk scratch b")?,
            pos_b: ctx.upload_i32(&[0])?,
            out_a: ctx.upload_i32(&[0])?,
            out_b: ctx.upload_i32(&[0])?,
            conv_scratch: mk((text.linear_conv_kernel_dim - 1) * conv_dim)?,
            rec_scratch: mk(rec_len)?,
            gdn_conv_out_a: mk(conv_dim)?,
            gdn_out_a: mk(text.linear_num_value_heads * text.linear_value_head_dim)?,
            gdn_conv_out_b: mk(conv_dim)?,
            gdn_out_b: mk(text.linear_num_value_heads * text.linear_value_head_dim)?,
            mtp_q_out: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            mtp_gate_out: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            mtp_attn_out: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            mtp_inner: ctx.stream.alloc_zeros::<f32>(text.intermediate_size)?,
            mtp_x1: ctx.stream.alloc_zeros::<f32>(n)?,
            mtp_cat: ctx.stream.alloc_zeros::<f32>(2 * n)?,
            mtp_res: ctx.stream.alloc_zeros::<f32>(n)?,
            mtp_embed: ctx.stream.alloc_zeros::<f32>(n)?,
            mtp_kv_keys: ctx.stream.alloc_zeros::<f32>(max_ctx * kv_stride)?,
            mtp_kv_values: ctx.stream.alloc_zeros::<f32>(max_ctx * kv_stride)?,
            mtp_pos: ctx.upload_i32(&[0])?,
            mtp_tok: ctx.upload_i32(&[0])?,
            nt_draft: ctx.upload_i32(&[0])?,
            draft_id: 0,
            mtp_proj,
            mtp_pre_embed_w: ctx
                .upload_f32(&weights.f32_named("mtp.pre_fc_norm_embedding.weight")?)?,
            mtp_pre_hidden_w: ctx
                .upload_f32(&weights.f32_named("mtp.pre_fc_norm_hidden.weight")?)?,
            mtp_in_norm: ctx
                .upload_f32(&weights.f32_named("mtp.layers.0.input_layernorm.weight")?)?,
            mtp_post_norm: ctx
                .upload_f32(&weights.f32_named("mtp.layers.0.post_attention_layernorm.weight")?)?,
            mtp_qnorm: ctx
                .upload_f32(&weights.f32_named("mtp.layers.0.self_attn.q_norm.weight")?)?,
            mtp_knorm: ctx
                .upload_f32(&weights.f32_named("mtp.layers.0.self_attn.k_norm.weight")?)?,
            mtp_final_norm: ctx.upload_f32(&weights.f32_named("mtp.norm.weight")?)?,
            concat,
        })
    }

    /// cols=2 (concurrent columns, gridDim.y=2): wins over sequential
    /// when one call's tiles exceed L2. Each slot is (segment, its column-B
    /// output). Unused slots must be empty_seg_like() (rows=0) — a repeated
    /// real projection is re-read and recomputed for identical output.
    fn group2_wide(
        &self,
        gpu: &QwenGpu,
        slots: [(GroupSeg, &CudaSlice<f32>); 4],
        xa: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
        in_dim: usize,
    ) -> Result<()> {
        let [(s0, y0), (s1, y1), (s2, y2), (s3, y3)] = slots;
        gpu.wide
            .group(
                &gpu.ctx,
                &[s0, s1, s2, s3],
                [y0, y1, y2, y3],
                xa,
                xb,
                in_dim,
                2,
            )
            .context("group2_wide")
    }

    #[allow(clippy::too_many_arguments)]
    fn down2_wide(
        &self,
        gpu: &QwenGpu,
        q: &GpuQuant,
        yb: &CudaSlice<f32>,
        xa: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
        split: usize,
    ) -> Result<()> {
        let (p, s, b) = q.gemv_parts();
        gpu.wide
            .down(
                &gpu.ctx,
                p,
                s,
                b,
                xa,
                xb,
                q.y_ref(),
                yb,
                &self.scr,
                &self.scr_b,
                q.out_dim,
                q.in_dim,
                split,
                2,
            )
            .context("down2_wide")
    }

    /// One verify round: run the pair [tok_a (real), draft] through the
    /// model with one weight pass. Returns (token after A, token after B).
    /// tok_a rides `gpu.next_token`, the draft `self.nt_draft` (both
    /// device-read) — parameters here forced per-round CudaSlice clones.
    pub fn verify_round(&mut self, gpu: &mut QwenGpu) -> Result<(u32, u32)> {
        let text = gpu.config.text_config.clone();
        let eps = text.rms_norm_eps as f32;
        let n = text.hidden_size;
        let kv_stride = text.num_key_value_heads * text.head_dim;
        let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let scale = 1.0 / (text.head_dim as f32).sqrt();
        gpu.ctx
            .stream
            .memcpy_htod(&[gpu.position as i32 + 1], &mut self.pos_b)?;
        let embed = gpu
            .proj
            .get(&format!("{TEXT_PREFIX}.embed_tokens"))
            .unwrap();
        gpu.ctx
            .glue_embed_row(embed, &gpu.next_token, &gpu.hidden)?;
        gpu.ctx
            .glue_embed_row(embed, &self.nt_draft, &self.hidden_b)?;
        gpu.ctx
            .glue_rmsnorm_zc(&gpu.hidden, &gpu.ln[0][0], &gpu.x1, n, eps)?;
        gpu.ctx
            .glue_rmsnorm_zc(&self.hidden_b, &gpu.ln[0][0], &self.x1_b, n, eps)?;

        let mut gdn_index = 0usize;
        let mut kv_index = 0usize;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
            if text.layer_kind(layer) == LayerKind::LinearAttention {
                let names = [
                    format!("{prefix}.linear_attn.in_proj_qkv"),
                    format!("{prefix}.linear_attn.in_proj_z"),
                    format!("{prefix}.linear_attn.in_proj_b"),
                    format!("{prefix}.linear_attn.in_proj_a"),
                ];
                let qkv = gpu.proj.get(&names[0]).unwrap();
                let z = gpu.proj.get(&names[1]).unwrap();
                let b = gpu.proj.get(&names[2]).unwrap();
                let a = gpu.proj.get(&names[3]).unwrap();
                self.group2_wide(
                    gpu,
                    [
                        (qkv.group_seg(), &self.y_b[&names[0]]),
                        (z.group_seg(), &self.y_b[&names[1]]),
                        (b.group_seg(), &self.y_b[&names[2]]),
                        (a.group_seg(), &self.y_b[&names[3]]),
                    ],
                    &gpu.x1,
                    &self.x1_b,
                    qkv.in_dim,
                )?;
                let g = &gpu.gdn[gdn_index];
                let (conv_w, a_log, dt_bias, norm_w, conv_state, recurrent) = gpu.ctx.gdn_parts(g);
                let kernel_i = text.linear_conv_kernel_dim as i32;
                let conv_dim_i = text.conv_dim() as i32;
                unsafe {
                    gpu.ctx
                        .stream
                        .launch_builder(&self.conv2)
                        .arg(conv_w)
                        .arg(conv_state)
                        .arg(&self.conv_scratch[gdn_index])
                        .arg(qkv.y_ref())
                        .arg(&self.y_b[&names[0]])
                        .arg(&self.gdn_conv_out_a[gdn_index])
                        .arg(&self.gdn_conv_out_b[gdn_index])
                        .arg(&conv_dim_i)
                        .arg(&kernel_i)
                        .launch(LaunchConfig {
                            grid_dim: ((text.conv_dim() as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map(|_| ())
                        .map_err(|e| anyhow::anyhow!("conv2: {e}"))?;
                    let num_v = text.linear_num_value_heads as i32;
                    let num_k = text.linear_num_key_heads as i32;
                    let dk = text.linear_key_head_dim as i32;
                    let dv = text.linear_value_head_dim as i32;
                    let sc = 1.0f32 / (text.linear_key_head_dim as f32).sqrt();
                    gpu.ctx
                        .stream
                        .launch_builder(&self.heads2)
                        .arg(&self.gdn_conv_out_a[gdn_index])
                        .arg(&self.gdn_conv_out_b[gdn_index])
                        .arg(z.y_ref())
                        .arg(&self.y_b[&names[1]])
                        .arg(b.y_ref())
                        .arg(&self.y_b[&names[2]])
                        .arg(a.y_ref())
                        .arg(&self.y_b[&names[3]])
                        .arg(a_log)
                        .arg(dt_bias)
                        .arg(norm_w)
                        .arg(recurrent)
                        .arg(&self.rec_scratch[gdn_index])
                        .arg(recurrent)
                        .arg(&self.gdn_out_a[gdn_index])
                        .arg(&self.gdn_out_b[gdn_index])
                        .arg(&num_v)
                        .arg(&num_k)
                        .arg(&dk)
                        .arg(&dv)
                        .arg(&sc)
                        .arg(&eps)
                        .launch(LaunchConfig {
                            grid_dim: (text.linear_num_value_heads as u32, 1, 1),
                            block_dim: (1024, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map(|_| ())
                        .map_err(|e| anyhow::anyhow!("heads2: {e}"))?;
                }
                let out_proj = gpu
                    .proj
                    .get(&format!("{prefix}.linear_attn.out_proj"))
                    .unwrap();
                let out_yb = &self.y_b[&format!("{prefix}.linear_attn.out_proj")];
                // One real slot — a repeated projection is pure re-read.
                self.group2_wide(
                    gpu,
                    [
                        (out_proj.group_seg(), out_yb),
                        (out_proj.empty_seg_like(), out_yb),
                        (out_proj.empty_seg_like(), out_yb),
                        (out_proj.empty_seg_like(), out_yb),
                    ],
                    &self.gdn_out_a[gdn_index],
                    &self.gdn_out_b[gdn_index],
                    out_proj.in_dim,
                )?;
                gpu.ctx.glue_add_rmsnorm_zc(
                    &gpu.hidden,
                    out_proj.y_ref(),
                    &gpu.ln[layer][1],
                    &gpu.x1,
                    n,
                    eps,
                )?;
                gpu.ctx.glue_add_rmsnorm_zc(
                    &self.hidden_b,
                    &self.y_b[&format!("{prefix}.linear_attn.out_proj")],
                    &gpu.ln[layer][1],
                    &self.x1_b,
                    n,
                    eps,
                )?;
                gdn_index += 1;
            } else {
                let qname = format!("{prefix}.self_attn.q_proj");
                let kname = format!("{prefix}.self_attn.k_proj");
                let vname = format!("{prefix}.self_attn.v_proj");
                let oname = format!("{prefix}.self_attn.o_proj");
                let q = gpu.proj.get(&qname).unwrap();
                let k = gpu.proj.get(&kname).unwrap();
                let v = gpu.proj.get(&vname).unwrap();
                let o = gpu.proj.get(&oname).unwrap();
                self.group2_wide(
                    gpu,
                    [
                        (q.group_seg(), &self.y_b[&qname]),
                        (k.group_seg(), &self.y_b[&kname]),
                        (v.group_seg(), &self.y_b[&vname]),
                        (q.empty_seg_like(), &self.y_b[&qname]),
                    ],
                    &gpu.x1,
                    &self.x1_b,
                    q.in_dim,
                )?;
                let [qn, kn] = &gpu.attn_norms[kv_index];
                gpu.ctx.glue_attn_qk_raw(
                    q.y_ref(),
                    qn,
                    k.y_ref(),
                    kn,
                    v.y_ref(),
                    &gpu.q_out,
                    &gpu.gate_out,
                    &gpu.kv_keys[kv_index],
                    &gpu.kv_values[kv_index],
                    &gpu.pos,
                    kv_stride,
                    text.num_attention_heads,
                    text.num_key_value_heads,
                    text.head_dim,
                    rotary_dim,
                    text.rope.rope_theta,
                    true,
                )?;
                gpu.ctx.glue_attn_qk_raw(
                    &self.y_b[&qname],
                    qn,
                    &self.y_b[&kname],
                    kn,
                    &self.y_b[&vname],
                    &self.q_out_b,
                    &self.gate_out_b,
                    &gpu.kv_keys[kv_index],
                    &gpu.kv_values[kv_index],
                    &self.pos_b,
                    kv_stride,
                    text.num_attention_heads,
                    text.num_key_value_heads,
                    text.head_dim,
                    rotary_dim,
                    text.rope.rope_theta,
                    true,
                )?;
                gpu.ctx.glue_attn_scores_raw(
                    &gpu.q_out,
                    &gpu.gate_out,
                    &gpu.kv_keys[kv_index],
                    &gpu.kv_values[kv_index],
                    &gpu.attn_out,
                    &gpu.pos,
                    kv_stride,
                    text.num_attention_heads,
                    text.num_key_value_heads,
                    text.head_dim,
                    scale,
                )?;
                gpu.ctx.glue_attn_scores_raw(
                    &self.q_out_b,
                    &self.gate_out_b,
                    &gpu.kv_keys[kv_index],
                    &gpu.kv_values[kv_index],
                    &self.attn_out_b,
                    &self.pos_b,
                    kv_stride,
                    text.num_attention_heads,
                    text.num_key_value_heads,
                    text.head_dim,
                    scale,
                )?;
                self.group2_wide(
                    gpu,
                    [
                        (o.group_seg(), &self.y_b[&oname]),
                        (o.empty_seg_like(), &self.y_b[&oname]),
                        (o.empty_seg_like(), &self.y_b[&oname]),
                        (o.empty_seg_like(), &self.y_b[&oname]),
                    ],
                    &gpu.attn_out,
                    &self.attn_out_b,
                    o.in_dim,
                )?;
                gpu.ctx.glue_add_rmsnorm_zc(
                    &gpu.hidden,
                    o.y_ref(),
                    &gpu.ln[layer][1],
                    &gpu.x1,
                    n,
                    eps,
                )?;
                gpu.ctx.glue_add_rmsnorm_zc(
                    &self.hidden_b,
                    &self.y_b[&oname],
                    &gpu.ln[layer][1],
                    &self.x1_b,
                    n,
                    eps,
                )?;
                kv_index += 1;
            }
            // Dense MLP, both columns.
            let gname = format!("{prefix}.mlp.gate_proj");
            let uname = format!("{prefix}.mlp.up_proj");
            let dname = format!("{prefix}.mlp.down_proj");
            let gate = gpu.proj.get(&gname).unwrap();
            let up = gpu.proj.get(&uname).unwrap();
            let down = gpu.proj.get(&dname).unwrap();
            self.group2_wide(
                gpu,
                [
                    (gate.group_seg(), &self.y_b[&gname]),
                    (up.group_seg(), &self.y_b[&uname]),
                    (gate.empty_seg_like(), &self.y_b[&gname]),
                    (gate.empty_seg_like(), &self.y_b[&gname]),
                ],
                &gpu.x1,
                &self.x1_b,
                gate.in_dim,
            )?;
            gpu.ctx
                .silu_mul(gate.y_ref(), up.y_ref(), &gpu.inner, text.intermediate_size)?;
            gpu.ctx.silu_mul(
                &self.y_b[&gname],
                &self.y_b[&uname],
                &self.inner_b,
                text.intermediate_size,
            )?;
            self.down2_wide(gpu, down, &self.y_b[&dname], &gpu.inner, &self.inner_b, 4)?;
            let next_w = if layer + 1 < text.num_hidden_layers {
                &gpu.ln[layer + 1][0]
            } else {
                &gpu.final_norm_w
            };
            gpu.ctx
                .glue_add_rmsnorm_zc(&gpu.hidden, down.y_ref(), next_w, &gpu.x1, n, eps)?;
            gpu.ctx.glue_add_rmsnorm_zc(
                &self.hidden_b,
                &self.y_b[&dname],
                next_w,
                &self.x1_b,
                n,
                eps,
            )?;
        }
        // Final norm is folded into the last layer's MLP add; lm_head + argmax x2.
        let lm = gpu.proj.get("lm_head").unwrap();
        let lsegs: [GroupSeg; 4] = [
            lm.group_seg(),
            lm.empty_seg_like(),
            lm.empty_seg_like(),
            lm.empty_seg_like(),
        ];
        let lyb = [
            &self.y_b["lm_head"],
            &self.y_b["lm_head"],
            &self.y_b["lm_head"],
            &self.y_b["lm_head"],
        ];
        gpu.wide
            .group(&gpu.ctx, &lsegs, lyb, &gpu.x1, &self.x1_b, lm.in_dim, 2)
            .context("lm cols=2")?;
        gpu.ctx
            .glue_argmax(lm.y_ref(), &mut self.out_a, lm.out_dim)?;
        gpu.ctx
            .glue_argmax(&self.y_b["lm_head"], &mut self.out_b, lm.out_dim)?;
        let mut ids = [0i32, 0i32];
        gpu.ctx.stream.memcpy_dtoh(&self.out_a, &mut ids[..1])?;
        gpu.ctx.stream.memcpy_dtoh(&self.out_b, &mut ids[1..])?;
        gpu.ctx.counted_sync()?;
        Ok((ids[0] as u32, ids[1] as u32))
    }

    /// GPU MTP draft: mtp.fc(cat([norm_embed(embed(tok)), norm_hidden(h)]))
    /// -> one decoder layer (own KV timeline) -> mtp.norm -> shared
    /// lm_head -> nt_draft. Fusion order arbitrated empirically (87% vs
    /// 0% reversed, CPU measurement). The residual stream rides fc's y
    /// buffer for the layer's duration.
    /// `h`: hidden to draft from; `None` = `self.hidden_b` (accept branch,
    /// not passable as a reference under &mut self).
    pub fn draft(&mut self, gpu: &QwenGpu, h: Option<&CudaSlice<f32>>, tok: u32) -> Result<()> {
        let h = match h {
            Some(h) => h,
            None => &self.hidden_b,
        };
        let text = &gpu.config.text_config;
        let eps = text.rms_norm_eps as f32;
        let n = text.hidden_size;
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let head_dim = text.head_dim;
        let kv_stride = kv_heads * head_dim;
        let rotary_dim = (head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        gpu.ctx
            .stream
            .memcpy_htod(&[tok as i32], &mut self.mtp_tok)?;
        let embed = gpu
            .proj
            .get(&format!("{TEXT_PREFIX}.embed_tokens"))
            .context("embed resident")?;
        gpu.ctx
            .glue_embed_row(embed, &self.mtp_tok, &self.mtp_embed)?;
        gpu.ctx
            .glue_rmsnorm_zc(&self.mtp_embed, &self.mtp_pre_embed_w, &self.mtp_x1, n, eps)?;
        gpu.ctx
            .glue_rmsnorm_zc(h, &self.mtp_pre_hidden_w, &self.mtp_embed, n, eps)?;
        let n_i = n as i32;
        unsafe {
            gpu.ctx
                .stream
                .launch_builder(&self.concat)
                .arg(&self.mtp_cat)
                .arg(&self.mtp_x1)
                .arg(&self.mtp_embed)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("concat: {e}"))?;
        }
        let fc = self.mtp_proj.get("mtp.fc").unwrap();
        fc.launch(&gpu.ctx, &self.mtp_cat, &self.mtp_res)?;
        let xbuf = &self.mtp_res;
        // Decoder layer: norm -> attn -> +res -> norm -> mlp -> +res.
        gpu.ctx
            .glue_rmsnorm_zc(xbuf, &self.mtp_in_norm, &self.mtp_x1, n, eps)?;
        let (q, k, v) = (
            self.mtp_proj.get("mtp.layers.0.self_attn.q_proj").unwrap(),
            self.mtp_proj.get("mtp.layers.0.self_attn.k_proj").unwrap(),
            self.mtp_proj.get("mtp.layers.0.self_attn.v_proj").unwrap(),
        );
        let segs: [GroupSeg; 4] = [
            q.group_seg(),
            k.group_seg(),
            v.group_seg(),
            q.empty_seg_like(),
        ];
        gpu.ctx.glue_group4(&segs, &self.mtp_x1, q.in_dim, 0)?;
        gpu.ctx.glue_attn_qk_raw(
            q.y_ref(),
            &self.mtp_qnorm,
            k.y_ref(),
            &self.mtp_knorm,
            v.y_ref(),
            &self.mtp_q_out,
            &self.mtp_gate_out,
            &self.mtp_kv_keys,
            &self.mtp_kv_values,
            &self.mtp_pos,
            kv_stride,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            text.rope.rope_theta,
            true,
        )?;
        gpu.ctx.glue_attn_scores_raw(
            &self.mtp_q_out,
            &self.mtp_gate_out,
            &self.mtp_kv_keys,
            &self.mtp_kv_values,
            &self.mtp_attn_out,
            &self.mtp_pos,
            kv_stride,
            heads,
            kv_heads,
            head_dim,
            scale,
        )?;
        let o = self.mtp_proj.get("mtp.layers.0.self_attn.o_proj").unwrap();
        let segs: [GroupSeg; 4] = [
            o.group_seg(),
            o.empty_seg_like(),
            o.empty_seg_like(),
            o.empty_seg_like(),
        ];
        gpu.ctx
            .glue_group4(&segs, &self.mtp_attn_out, o.in_dim, 0)?;
        gpu.ctx.glue_add_inplace(xbuf, o.y_ref(), n)?;
        gpu.ctx
            .glue_rmsnorm_zc(xbuf, &self.mtp_post_norm, &self.mtp_x1, n, eps)?;
        let (gate, up, down) = (
            self.mtp_proj.get("mtp.layers.0.mlp.gate_proj").unwrap(),
            self.mtp_proj.get("mtp.layers.0.mlp.up_proj").unwrap(),
            self.mtp_proj.get("mtp.layers.0.mlp.down_proj").unwrap(),
        );
        let segs: [GroupSeg; 4] = [
            gate.group_seg(),
            up.group_seg(),
            gate.empty_seg_like(),
            gate.empty_seg_like(),
        ];
        gpu.ctx.glue_group4(&segs, &self.mtp_x1, gate.in_dim, 0)?;
        gpu.ctx.silu_mul(
            gate.y_ref(),
            up.y_ref(),
            &self.mtp_inner,
            text.intermediate_size,
        )?;
        down.launch(&gpu.ctx, &self.mtp_inner, down.y_ref())?;
        gpu.ctx.glue_add_inplace(xbuf, down.y_ref(), n)?;
        gpu.ctx
            .glue_rmsnorm_zc(xbuf, &self.mtp_final_norm, &self.mtp_x1, n, eps)?;
        let lm = gpu.proj.get("lm_head").unwrap();
        lm.launch(&gpu.ctx, &self.mtp_x1, lm.y_ref())?;
        gpu.ctx
            .glue_argmax(lm.y_ref(), &mut self.nt_draft, lm.out_dim)?;
        gpu.ctx.glue_inc(&mut self.mtp_pos)?;
        let mut id = [0i32];
        gpu.ctx.stream.memcpy_dtoh(&self.nt_draft, &mut id)?;
        gpu.ctx.counted_sync()?;
        self.draft_id = id[0] as u32;
        Ok(())
    }

    /// Restore GDN state to post-A after a reject (dtod from scratch).
    pub fn reject_restore(&mut self, gpu: &mut QwenGpu) -> Result<()> {
        for i in 0..self.rec_scratch.len() {
            let g = &mut gpu.gdn[i];
            let (conv_state, recurrent) = gpu.ctx.gdn_state_mut(g);
            gpu.ctx
                .stream
                .memcpy_dtod(&self.conv_scratch[i], conv_state)?;
            gpu.ctx
                .stream
                .memcpy_dtod(&self.rec_scratch[i], recurrent)?;
        }
        Ok(())
    }
}
