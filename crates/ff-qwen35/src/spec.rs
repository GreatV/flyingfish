//! MTP speculative decode: batch-2 verify rounds over the batch2.cu
//! kernels (weights read once per token pair). A round is fully
//! device-driven — the accept kernel advances the round counters, a
//! predicated kernel reloads GDN state on reject — so the whole round
//! captures as one CUDA graph; eager rounds run the identical sequence.
//! KV needs nothing special: write slots derive from the position
//! counters.
//!
//! Timelines: the main model uses `gpu.pos`; the draft position rides
//! `pos_b` (= pos + 1, maintained by the accept kernel). The MTP layer
//! keeps its OWN 0-based KV timeline (`mtp_pos`): rope angles shift by a
//! constant against the main timeline, which preserves relative angles —
//! the acceptance measurement arbitrates draft quality anyway.

use crate::config::{LayerKind, TEXT_PREFIX};
use crate::gpu::QwenGpu;
use crate::wide::{DownBuffers, GroupPair, WideGeom};
use anyhow::{Context, Result};
use cudarc::driver::safe::{CudaSlice, LaunchConfig, PushKernelArg};
use ff_edge0::gpu::{
    AttnGeom, GpuQuant, GroupSeg, KvCache, QkGeom, QkOutputs, QkvNorm, ScoreBuffers,
};

/// What one speculative round did, read back from the device ring.
pub struct RoundRecord {
    pub accepted: bool,
    pub pending: u32,
    pub drafted: u32,
    pub a1: u32,
}

pub struct QwenSpec {
    conv2: cudarc::driver::safe::CudaFunction,
    heads2: cudarc::driver::safe::CudaFunction,
    restore: cudarc::driver::safe::CudaFunction,
    // B-side (draft token) buffers
    pub hidden_b: CudaSlice<f32>,
    x1_b: CudaSlice<f32>,
    q_out_b: CudaSlice<f32>,
    gate_out_b: CudaSlice<f32>,
    attn_out_b: CudaSlice<f32>,
    inner_b: CudaSlice<f32>,
    /// Per-projection second output buffer (the shared GpuQuant y is A's).
    y_b: std::collections::HashMap<String, CudaSlice<f32>>,
    pos_b: CudaSlice<i32>,

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
    accept_ring: CudaSlice<f32>,
    round_idx: CudaSlice<i32>,
    accept_flag: CudaSlice<f32>,
    host_ring: Vec<f32>,
    max_rounds: usize,
    round_graph: Option<ff_edge0::gpu::DecodeGraph>,
    hidden_sel: CudaSlice<f32>,
}

/// Device bytes the speculative path adds on top of a resident plan: the
/// B-column activation set, the per-layer A/B-seam GDN scratch, the MTP
/// draft layer with its own KV timeline and weights, and the round ring.
/// Admission charges this instead of a blanket reserve.
pub fn workspace_bytes(
    weights: &crate::weights::Qwen35Weights,
    text: &crate::config::TextConfig,
    max_ctx: usize,
    max_rounds: usize,
) -> u64 {
    let n = text.hidden_size;
    let q_dim = text.num_attention_heads * text.head_dim;
    let conv_dim = text.conv_dim();
    let kv_stride = text.num_key_value_heads * text.head_dim;
    let rec_len =
        text.linear_num_value_heads * text.linear_key_head_dim * text.linear_value_head_dim;
    let f32_bytes = |count: usize| 4 * count as u64;
    let mut bytes = 0u64;
    let proj_out = |name: &str| -> u64 {
        weights
            .projection_shape(name)
            .map(|(out, _)| f32_bytes(out))
            .unwrap_or(0)
    };
    let gdn_layers = (0..text.num_hidden_layers)
        .filter(|&l| text.layer_kind(l) == LayerKind::LinearAttention)
        .count();
    for layer in 0..text.num_hidden_layers {
        let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
        match text.layer_kind(layer) {
            LayerKind::LinearAttention => {
                for p in [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ] {
                    bytes += proj_out(&format!("{prefix}.linear_attn.{p}"));
                }
            }
            LayerKind::FullAttention => {
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    bytes += proj_out(&format!("{prefix}.self_attn.{p}"));
                }
            }
        }
        for p in ["gate_proj", "up_proj", "down_proj"] {
            bytes += proj_out(&format!("{prefix}.mlp.{p}"));
        }
    }
    bytes += proj_out(&format!("{TEXT_PREFIX}.embed_tokens"));
    bytes += proj_out("lm_head");
    bytes += f32_bytes(2 * text.intermediate_size.max(4 * n));
    bytes += f32_bytes(2 * n + 3 * q_dim + text.intermediate_size);
    bytes += f32_bytes(
        gdn_layers
            * ((text.linear_conv_kernel_dim - 1) * conv_dim
                + rec_len
                + 2 * conv_dim
                + 2 * text.linear_num_value_heads * text.linear_value_head_dim),
    );
    bytes += f32_bytes(4 * n + 3 * q_dim + text.intermediate_size + 2 * max_ctx * kv_stride);
    for name in [
        "mtp.fc",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.k_proj",
        "mtp.layers.0.self_attn.v_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
        "mtp.layers.0.mlp.down_proj",
    ] {
        bytes += weights.tensor_device_bytes(name).unwrap_or(0);
    }
    bytes += f32_bytes(7 * n);
    bytes += f32_bytes(7 * max_rounds + 2);
    bytes
}

impl QwenSpec {
    pub fn new(
        gpu: &mut QwenGpu,
        weights: &crate::weights::Qwen35Weights,
        max_rounds: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            gpu.peers.is_empty(),
            "speculative decode drives a single device"
        );
        anyhow::ensure!(
            !gpu.is_streaming(),
            "speculative decode requires resident weights"
        );
        anyhow::ensure!(
            weights.format() == crate::weights::INT4,
            "speculative decode on 16-bit checkpoints lands with phase 4"
        );
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

        let module =
            ff_edge0::kernel_assets::load_module(&ctx.context, &crate::kernel_assets::QWEN_BATCH2)?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| anyhow::anyhow!("{name}: {e:?}"))
        };
        let conv2 = load("qwen_gdn_conv2")?;
        let heads2 = load("qwen_gdn_heads2")?;
        let restore = load("qwen_gdn_restore")?;

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
        let spec = Self {
            conv2,
            heads2,
            restore,
            hidden_b: ctx.stream.alloc_zeros::<f32>(n)?,
            x1_b: ctx.stream.alloc_zeros::<f32>(n)?,
            q_out_b: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            gate_out_b: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            attn_out_b: ctx.stream.alloc_zeros::<f32>(q_dim)?,
            inner_b: ctx.stream.alloc_zeros::<f32>(text.intermediate_size)?,
            y_b,
            pos_b: ctx.upload_i32(&[0])?,

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
            accept_ring: ctx.stream.alloc_zeros::<f32>(7 * max_rounds)?,
            round_idx: ctx.upload_i32(&[0])?,
            accept_flag: ctx.upload_f32(&[0.0])?,
            host_ring: vec![0.0; 7 * max_rounds],
            max_rounds,
            round_graph: None,
            hidden_sel: ctx.stream.alloc_zeros::<f32>(n)?,
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
        };
        Ok(spec)
    }

    fn clear_probe_state(&mut self, gpu: &QwenGpu) -> Result<()> {
        for value in [
            &mut self.hidden_b,
            &mut self.x1_b,
            &mut self.q_out_b,
            &mut self.gate_out_b,
            &mut self.attn_out_b,
            &mut self.inner_b,
            &mut self.mtp_q_out,
            &mut self.mtp_gate_out,
            &mut self.mtp_attn_out,
            &mut self.mtp_inner,
            &mut self.mtp_x1,
            &mut self.mtp_cat,
            &mut self.mtp_res,
            &mut self.mtp_embed,
            &mut self.mtp_kv_keys,
            &mut self.mtp_kv_values,
            &mut self.hidden_sel,
            &mut self.accept_ring,
            &mut self.accept_flag,
        ] {
            gpu.ctx.stream.memset_zeros(value)?;
        }
        for value in [
            &mut self.pos_b,
            &mut self.mtp_pos,
            &mut self.mtp_tok,
            &mut self.nt_draft,
            &mut self.round_idx,
        ] {
            gpu.ctx.stream.memset_zeros(value)?;
        }
        for value in self
            .y_b
            .values_mut()
            .chain(&mut self.conv_scratch)
            .chain(&mut self.rec_scratch)
            .chain(&mut self.gdn_conv_out_a)
            .chain(&mut self.gdn_out_a)
            .chain(&mut self.gdn_conv_out_b)
            .chain(&mut self.gdn_out_b)
        {
            gpu.ctx.stream.memset_zeros(value)?;
        }
        self.host_ring.fill(0.0);
        Ok(())
    }

    pub fn calibrate_groups(&mut self, gpu: &mut QwenGpu) -> Result<()> {
        let capacity = gpu.probe_round_capacity(self.max_rounds)?;
        let rows = gpu
            .position
            .checked_add(
                capacity
                    .checked_mul(2)
                    .context("column-2 probe prefix overflow")?,
            )
            .and_then(|n| n.checked_add(2))
            .context("column-2 probe prefix overflow")?;
        let saved = gpu.probe_state(rows)?;
        let pending = gpu.read_token()?;
        let body1 = gpu.wide.bodies.chosen(1)?;
        let stream = gpu.ctx.stream.clone();
        let ordinal = gpu.ctx.context.ordinal();
        let started = std::time::Instant::now();
        let mut graphs = [None, None];
        for (body, graph) in graphs.iter_mut().enumerate() {
            gpu.restore_probe_state(&saved)?;
            self.clear_probe_state(gpu)?;
            gpu.wide.capture_body(Some(body1))?;
            self.bootstrap(gpu, pending)?;
            gpu.wide.capture_body(Some(body))?;
            ff_edge0::gpu::begin_decode_capture(&stream).context("column-2 probe capture begin")?;
            if let Err(e) = self.enqueue_round(gpu) {
                let _ = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                return Err(e.context("column-2 probe enqueue"));
            }
            *graph = Some(stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH).context("column-2 probe capture end")?.context("column-2 probe graph missing")?);
        }
        let capture_ms = started.elapsed().as_secs_f64() * 1000.0;
        let state = std::cell::RefCell::new((self, gpu));
        let choice = ff_edge0::gpu::probe_decode(
            &stream,
            ff_edge0::gpu::DecodeProgram {
                device: ordinal,
                cols: 2,
                capacity,
                capture_ms,
                state_bytes: saved.bytes(),
                seed_token: pending,
                topology: "single-device resident speculative round".into(),
            },
            |body| {
                let mut state = state.borrow_mut();
                let (spec, gpu) = &mut *state;
                gpu.restore_probe_state(&saved)?;
                spec.clear_probe_state(gpu)?;
                gpu.wide.capture_body(Some(body1))?;
                spec.bootstrap(gpu, pending)?;
                gpu.wide.capture_body(Some(body))?;
                Ok(())
            },
            |body, repeats| {
                let graph = graphs[body]
                    .as_ref()
                    .context("column-2 candidate graph missing")?;
                for _ in 0..repeats {
                    graph.launch().context("column-2 candidate replay")?;
                }
                Ok(())
            },
        )?;
        let (spec, gpu) = state.into_inner();
        gpu.wide.bodies.bind(choice);
        gpu.restore_probe_state(&saved)?;
        spec.clear_probe_state(gpu)?;
        gpu.wide.capture_body(None)?;
        spec.round_graph = None;
        Ok(())
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
                &GroupPair { x: xa, xb },
                in_dim,
                2,
            )
            .context("group2_wide")
    }

    fn down2_wide(
        &self,
        gpu: &QwenGpu,
        q: &GpuQuant,
        yb: &CudaSlice<f32>,
        xa: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
    ) -> Result<()> {
        let (p, s, b) = q.tensors();
        gpu.wide
            .down(
                &gpu.ctx,
                &DownBuffers {
                    packed: p,
                    scales: s,
                    biases: b,
                    x: xa,
                    xb,
                    y: q.y_ref(),
                    yb,
                },
                &WideGeom {
                    rows: q.out_dim,
                    in_dim: q.in_dim,
                    cols: 2,
                },
            )
            .context("down2_wide")
    }

    /// One device-driven round: the verify pass over the pair [tok_a
    /// (real), draft], the accept record, the predicated GDN restore and
    /// the next draft. tok_a rides `gpu.next_token`, the draft
    /// `self.nt_draft` — both device-maintained, as are `pos_b` and the
    /// round counters (all advanced inside the accept kernel). No syncs
    /// and no host transfers, so the sequence is graph-capturable.
    fn enqueue_round(&mut self, gpu: &mut QwenGpu) -> Result<()> {
        let text = gpu.config.text_config.clone();
        let eps = text.rms_norm_eps as f32;
        let n = text.hidden_size;
        let kv_stride = text.num_key_value_heads * text.head_dim;
        let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let scale = 1.0 / (text.head_dim as f32).sqrt();
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
                    &QkvNorm {
                        q_raw: q.y_ref().as_view(),
                        q_norm_w: qn,
                        k_raw: k.y_ref().as_view(),
                        k_norm_w: kn,
                        v_raw: v.y_ref().as_view(),
                    },
                    &QkOutputs {
                        q_out: gpu.q_out.as_view(),
                        gate_out: gpu.gate_out.as_view(),
                    },
                    &KvCache {
                        keys: &gpu.kv_keys[kv_index],
                        values: &gpu.kv_values[kv_index],
                        stride: kv_stride,
                    },
                    &gpu.pos,
                    &QkGeom {
                        heads: text.num_attention_heads,
                        kv_heads: text.num_key_value_heads,
                        head_dim: text.head_dim,
                        rotary_dim,
                        theta: text.rope.rope_theta,
                    },
                    true,
                )?;
                gpu.ctx.glue_attn_qk_raw(
                    &QkvNorm {
                        q_raw: self.y_b[&qname].as_view(),
                        q_norm_w: qn,
                        k_raw: self.y_b[&kname].as_view(),
                        k_norm_w: kn,
                        v_raw: self.y_b[&vname].as_view(),
                    },
                    &QkOutputs {
                        q_out: self.q_out_b.as_view(),
                        gate_out: self.gate_out_b.as_view(),
                    },
                    &KvCache {
                        keys: &gpu.kv_keys[kv_index],
                        values: &gpu.kv_values[kv_index],
                        stride: kv_stride,
                    },
                    &self.pos_b,
                    &QkGeom {
                        heads: text.num_attention_heads,
                        kv_heads: text.num_key_value_heads,
                        head_dim: text.head_dim,
                        rotary_dim,
                        theta: text.rope.rope_theta,
                    },
                    true,
                )?;
                gpu.ctx.glue_attn_scores_raw(
                    &ScoreBuffers {
                        q: gpu.q_out.as_view(),
                        gate: gpu.gate_out.as_view(),
                        out: gpu.attn_out.as_view(),
                    },
                    &KvCache {
                        keys: &gpu.kv_keys[kv_index],
                        values: &gpu.kv_values[kv_index],
                        stride: kv_stride,
                    },
                    &gpu.pos,
                    &AttnGeom {
                        heads: text.num_attention_heads,
                        kv_heads: text.num_key_value_heads,
                        head_dim: text.head_dim,
                    },
                    scale,
                )?;
                gpu.ctx.glue_attn_scores_raw(
                    &ScoreBuffers {
                        q: self.q_out_b.as_view(),
                        gate: self.gate_out_b.as_view(),
                        out: self.attn_out_b.as_view(),
                    },
                    &KvCache {
                        keys: &gpu.kv_keys[kv_index],
                        values: &gpu.kv_values[kv_index],
                        stride: kv_stride,
                    },
                    &self.pos_b,
                    &AttnGeom {
                        heads: text.num_attention_heads,
                        kv_heads: text.num_key_value_heads,
                        head_dim: text.head_dim,
                    },
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
            self.down2_wide(gpu, down, &self.y_b[&dname], &gpu.inner, &self.inner_b)?;
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
            .group(
                &gpu.ctx,
                &lsegs,
                lyb,
                &GroupPair {
                    x: &gpu.x1,
                    xb: &self.x1_b,
                },
                lm.in_dim,
                2,
            )
            .context("lm cols=2")?;
        gpu.ctx.glue_spec_accept(
            &ff_edge0::gpu::SpecAcceptBuffers {
                logits_a: lm.y_ref(),
                logits_b: &self.y_b["lm_head"],
                draft_id: &self.nt_draft,
                hidden_accept: &self.hidden_b,
                hidden_reject: &gpu.hidden,
                next_token: &gpu.next_token,
                pos: &gpu.pos,
                rope_pos: &gpu.rope_pos,
                pos_b: &self.pos_b,
                flag: &self.accept_flag,
            },
            &mut ff_edge0::gpu::SpecAcceptOutputs {
                hidden_sel: &mut self.hidden_sel,
                mtp_tok: &mut self.mtp_tok,
                record: &mut self.accept_ring,
                round_idx: &mut self.round_idx,
            },
            lm.out_dim,
            n,
        )?;
        self.enqueue_gdn_restore(gpu)?;
        self.draft_body(gpu)
    }

    /// Reload each GDN layer's conv/recurrent state from the post-A seam
    /// scratch; the kernel no-ops when the round accepted.
    fn enqueue_gdn_restore(&mut self, gpu: &QwenGpu) -> Result<()> {
        for i in 0..self.rec_scratch.len() {
            let (_, _, _, _, conv_state, recurrent) = gpu.ctx.gdn_parts(&gpu.gdn[i]);
            let (conv_n, rec_n) = (conv_state.len() as i32, recurrent.len() as i32);
            let total = conv_n + rec_n;
            let grid = ((total as u32).div_ceil(256)).clamp(1, 1024);
            unsafe {
                gpu.ctx
                    .stream
                    .launch_builder(&self.restore)
                    .arg(conv_state)
                    .arg(&self.conv_scratch[i])
                    .arg(recurrent)
                    .arg(&self.rec_scratch[i])
                    .arg(&self.accept_flag)
                    .arg(&conv_n)
                    .arg(&rec_n)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("gdn restore: {e}"))?;
            }
        }
        Ok(())
    }

    /// Seed the first round from the prefill: pending token into the
    /// A-column slot, the prefill hidden as the first draft's input, the
    /// B-column timeline, and the first draft itself. Device writes only;
    /// the first round can follow without a sync.
    pub fn bootstrap(&mut self, gpu: &mut QwenGpu, pending: u32) -> Result<()> {
        gpu.ctx
            .stream
            .memcpy_htod(&[pending as i32], &mut gpu.next_token)?;
        gpu.ctx
            .stream
            .memcpy_htod(&[pending as i32], &mut self.mtp_tok)?;
        gpu.ctx
            .stream
            .memcpy_htod(&[gpu.position as i32 + 1], &mut self.pos_b)?;
        gpu.ctx
            .stream
            .memcpy_dtod(&gpu.hidden, &mut self.hidden_sel)?;
        self.draft_body(gpu)
    }

    /// Capture the selected program after its production bootstrap.
    pub fn capture_round_graph(&mut self, gpu: &mut QwenGpu) -> Result<()> {
        if self.round_graph.is_some() || !crate::gpu::graphs_enabled()? {
            return Ok(());
        }
        let body = gpu.wide.bodies.chosen(2)?;
        gpu.wide.capture_body(Some(body))?;
        ff_edge0::gpu::begin_decode_capture(&gpu.ctx.stream)
            .context("column-2 production capture begin")?;
        let result = self.enqueue_round(gpu);
        let captured = gpu.ctx.stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        gpu.wide.capture_body(None)?;
        result.context("column-2 production capture enqueue")?;
        let graph = captured
            .context("column-2 production capture end")?
            .context("column-2 production graph missing")?;
        self.round_graph = Some(ff_edge0::gpu::DecodeGraph(graph));
        Ok(())
    }

    pub fn round_step(&mut self, gpu: &mut QwenGpu) -> Result<()> {
        if let Some(g) = &self.round_graph {
            return g.0.launch().context("round graph replay");
        }
        let body = gpu.wide.bodies.chosen(2)?;
        gpu.wide.capture_body(Some(body))?;
        let result = self.enqueue_round(gpu);
        gpu.wide.capture_body(None)?;
        result
    }

    /// Sync and read back the ring slots of the rounds just executed.
    /// The ring index is cumulative across stretches; `first_round` is
    /// where this stretch starts.
    pub fn records(
        &mut self,
        gpu: &mut QwenGpu,
        first_round: usize,
        rounds: usize,
    ) -> Result<Vec<RoundRecord>> {
        anyhow::ensure!(
            first_round + rounds <= self.max_rounds,
            "round ring overflow: {} + {} > {}",
            first_round,
            rounds,
            self.max_rounds
        );
        let lo = 7 * first_round;
        let hi = 7 * (first_round + rounds);
        gpu.ctx
            .stream
            .memcpy_dtoh(&self.accept_ring.slice(lo..hi), &mut self.host_ring[lo..hi])?;
        gpu.ctx.counted_sync()?;
        let mut out = Vec::with_capacity(rounds);
        for r in (lo..hi).step_by(7) {
            let s = &self.host_ring[r..r + 7];
            out.push(RoundRecord {
                accepted: s[0] != 0.0,
                pending: s[1] as i32 as u32,
                drafted: s[6] as i32 as u32,
                a1: s[4] as i32 as u32,
            });
        }
        Ok(out)
    }

    /// GPU MTP draft: mtp.fc(cat([norm_embed(embed(tok)), norm_hidden(h)]))
    /// -> one decoder layer (own KV timeline) -> mtp.norm -> shared
    /// lm_head -> nt_draft. Fusion order arbitrated empirically (87% vs
    /// 0% reversed, CPU measurement). The residual stream rides fc's y
    /// buffer for the layer's duration.
    /// `h`: hidden to draft from; `None` = `self.hidden_b` (accept branch,
    /// not passable as a reference under &mut self).
    /// The draft layer + wide head + argmax, device-driven: the embed token
    /// comes from mtp_tok and the result lands in nt_draft, both device-side.
    fn draft_body(&mut self, gpu: &QwenGpu) -> Result<()> {
        let h = &self.hidden_sel;
        let text = &gpu.config.text_config;
        let eps = text.rms_norm_eps as f32;
        let n = text.hidden_size;
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let head_dim = text.head_dim;
        let kv_stride = kv_heads * head_dim;
        let rotary_dim = (head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
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
            &QkvNorm {
                q_raw: q.y_ref().as_view(),
                q_norm_w: &self.mtp_qnorm,
                k_raw: k.y_ref().as_view(),
                k_norm_w: &self.mtp_knorm,
                v_raw: v.y_ref().as_view(),
            },
            &QkOutputs {
                q_out: self.mtp_q_out.as_view(),
                gate_out: self.mtp_gate_out.as_view(),
            },
            &KvCache {
                keys: &self.mtp_kv_keys,
                values: &self.mtp_kv_values,
                stride: kv_stride,
            },
            &self.mtp_pos,
            &QkGeom {
                heads,
                kv_heads,
                head_dim,
                rotary_dim,
                theta: text.rope.rope_theta,
            },
            true,
        )?;
        gpu.ctx.glue_attn_scores_raw(
            &ScoreBuffers {
                q: self.mtp_q_out.as_view(),
                gate: self.mtp_gate_out.as_view(),
                out: self.mtp_attn_out.as_view(),
            },
            &KvCache {
                keys: &self.mtp_kv_keys,
                values: &self.mtp_kv_values,
                stride: kv_stride,
            },
            &self.mtp_pos,
            &AttnGeom {
                heads,
                kv_heads,
                head_dim,
            },
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
        let segs = [
            lm.group_seg(),
            lm.empty_seg_like(),
            lm.empty_seg_like(),
            lm.empty_seg_like(),
        ];
        gpu.wide.group(
            &gpu.ctx,
            &segs,
            [lm.y_ref(); 4],
            &GroupPair {
                x: &self.mtp_x1,
                xb: &self.mtp_x1,
            },
            lm.in_dim,
            1,
        )?;
        gpu.ctx
            .glue_argmax(lm.y_ref(), &mut self.nt_draft, lm.out_dim)?;
        gpu.ctx.glue_inc(&mut self.mtp_pos)?;
        Ok(())
    }
}

/// The speculative round economics the enablement rule runs on: the mode
/// that produced the output, the running acceptance counter, and the
/// round's cost overhead over a plain step (round_ms/plain_ms − 1, the
/// break-even acceptance).
pub struct SpecStats {
    pub mode: &'static str,
    pub r: f64,
    pub c: f64,
}

/// Greedy decode with one MTP-drafted token verified per round. A draft is
/// only emitted after the round's own greedy token confirmed it. The
/// round's batch-2 kernels are not bit-identical to the single-token step,
/// so ids diverge from the plain path only where an argmax near-tie breaks
/// the other way (first observed at a 0.025 logit gap).
///
/// Round economics gate the path: `c` (round_ms/plain_ms − 1) is
/// calibrated at setup over three timed round reps against two timed
/// plain steps; a round pays when `(1 + r) * plain_ms > round_ms`, i.e.
/// `r > c`. The setup verdict is `r0 > c` from the calibration rounds; the
/// running counter re-evaluates it at stretch boundaries (~32 rounds),
/// disabling at `r < 0.95c` and re-enabling at `r > 1.05c`. A disabled
/// verdict runs plain steps to the end of the request.
pub fn decode(
    gpu: &mut QwenGpu,
    weights: &crate::weights::Qwen35Weights,
    eos: &impl Fn(&u32) -> bool,
    max_new_tokens: usize,
) -> Result<(Vec<u32>, SpecStats)> {
    use std::time::Instant;

    let mut spec = QwenSpec::new(gpu, weights, max_new_tokens + 2)?;
    let mut stats = SpecStats {
        mode: "spec",
        r: 0.0,
        c: 0.0,
    };
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut pending = gpu.read_token()?;
    if eos(&pending) || max_new_tokens < 4 {
        plain_tail(gpu, pending, eos, max_new_tokens, &mut generated)?;
        return Ok((generated, stats));
    }
    spec.bootstrap(gpu, pending)?;

    // Setup calibration: three timed round reps price the round; the same
    // rounds emit real tokens and give the setup acceptance sample.
    let reps = 3usize;
    let per = ((max_new_tokens / 2).min(8) / reps).max(1);
    let mut rounds_total = 0usize;
    let (mut accepts, mut rounds_seen) = (0usize, 0usize);
    let mut rep_ms: Vec<f64> = Vec::with_capacity(reps);
    let mut eos_hit = false;
    for _ in 0..reps {
        let t = Instant::now();
        let recs = run_rounds(&mut spec, gpu, rounds_total, per)?;
        rep_ms.push(t.elapsed().as_secs_f64() * 1000.0 / per as f64);
        rounds_total += per;
        let walk = walk_records(pending, eos, &recs);
        generated.extend(walk.ids);
        gpu.position += walk.pos_delta;
        accepts += walk.accepts;
        rounds_seen += recs.len();
        pending = walk.pending;
        if walk.eos_hit {
            eos_hit = true;
            break;
        }
    }
    let r0 = accepts as f64 / rounds_seen as f64;
    stats.r = r0;
    let mut sorted = rep_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let round_ms = sorted[sorted.len() / 2];

    // Two timed plain steps price the token the rounds compete against.
    let mut step_ms = None;
    if !eos_hit && max_new_tokens - generated.len() >= 3 && !eos(&pending) {
        let t = Instant::now();
        for _ in 0..2 {
            generated.push(pending);
            gpu.step()?;
            pending = gpu.read_token()?;
        }
        step_ms = Some(t.elapsed().as_secs_f64() * 1000.0 / 2.0);
    }
    let c = match step_ms {
        Some(ms) => round_ms / ms - 1.0,
        None => f64::INFINITY,
    };
    stats.c = c;
    eprintln!(
        "spec calibration: r0 = {r0:.2} over {rounds_seen} rounds, round {round_ms:.1} ms \
         (spread {:.1}-{:.1} ms), plain {:.1} ms, c = {c:.2}",
        rep_ms.iter().cloned().fold(f64::MAX, f64::min),
        rep_ms.iter().cloned().fold(f64::MIN, f64::max),
        step_ms.unwrap_or(f64::NAN),
    );
    let mut enabled = match step_ms {
        Some(_) => r0 > c,
        None => true,
    };
    let mut spec_ran = false;
    if !enabled {
        eprintln!("spec disabled at setup: r0 = {r0:.2} <= c = {c:.2}");
    } else {
        // The graph pays capture once, only when spec actually runs; the
        // calibration rounds above went eager either way.
        spec.capture_round_graph(gpu)?;
    }

    loop {
        let remaining = max_new_tokens - generated.len();
        if remaining == 0 {
            break;
        }
        if !enabled || remaining < 4 || eos(&pending) {
            plain_tail(gpu, pending, eos, max_new_tokens, &mut generated)?;
            break;
        }
        let count = (remaining / 2).min(32);
        let recs = run_rounds(&mut spec, gpu, rounds_total, count)?;
        rounds_total += count;
        spec_ran = true;
        let walk = walk_records(pending, eos, &recs);
        generated.extend(walk.ids);
        gpu.position += walk.pos_delta;
        accepts += walk.accepts;
        rounds_seen += recs.len();
        pending = walk.pending;
        stats.r = accepts as f64 / rounds_seen as f64;
        if walk.eos_hit {
            break;
        }
        if enabled && stats.r < 0.95 * c {
            enabled = false;
            eprintln!(
                "spec disabled at stretch boundary: r = {:.2} < 0.95 c = {:.2}",
                stats.r, c
            );
        } else if !enabled && stats.r > 1.05 * c {
            enabled = true;
            eprintln!(
                "spec re-enabled at stretch boundary: r = {:.2} > 1.05 c = {:.2}",
                stats.r, c
            );
        }
    }
    generated.truncate(max_new_tokens);
    stats.mode = if spec_ran && enabled { "spec" } else { "plain" };
    Ok((generated, stats))
}

fn run_rounds(
    spec: &mut QwenSpec,
    gpu: &mut QwenGpu,
    first_round: usize,
    rounds: usize,
) -> Result<Vec<RoundRecord>> {
    for _ in 0..rounds {
        spec.round_step(gpu)?;
    }
    spec.records(gpu, first_round, rounds)
}

/// Fill the generation with plain steps. `pending` is a model-confirmed
/// token not yet emitted; EOS ends the generation after being pushed.
fn plain_tail(
    gpu: &mut QwenGpu,
    mut pending: u32,
    eos: &impl Fn(&u32) -> bool,
    want: usize,
    generated: &mut Vec<u32>,
) -> Result<()> {
    while generated.len() < want {
        generated.push(pending);
        if generated.len() >= want || eos(&pending) {
            break;
        }
        gpu.step()?;
        pending = gpu.read_token()?;
    }
    Ok(())
}

/// Replay the ring records into output ids: a round appends its pending
/// and, on accept, the drafted token; EOS anywhere ends the output while
/// every executed round still counts toward the position. Device position
/// truth lives in the records, so `pos_delta` covers all of them.
fn walk_records(pending: u32, eos: &impl Fn(&u32) -> bool, records: &[RoundRecord]) -> Walk {
    let mut ids = Vec::new();
    let mut pending = pending;
    let mut pos_delta = 0usize;
    let mut accepts = 0usize;
    let mut eos_hit = false;
    for rec in records {
        pos_delta += if rec.accepted { 2 } else { 1 };
        if eos_hit {
            continue;
        }
        if eos(&pending) {
            ids.push(pending);
            eos_hit = true;
            continue;
        }
        ids.push(pending);
        if rec.accepted {
            accepts += 1;
            if eos(&rec.drafted) {
                ids.push(rec.drafted);
                eos_hit = true;
                continue;
            }
            ids.push(rec.drafted);
        }
        pending = rec.pending;
    }
    Walk {
        ids,
        pending,
        pos_delta,
        accepts,
        eos_hit,
    }
}

struct Walk {
    ids: Vec<u32>,
    pending: u32,
    pos_delta: usize,
    accepts: usize,
    eos_hit: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(accepted: bool, pending: u32, drafted: u32, a1: u32) -> RoundRecord {
        RoundRecord {
            accepted,
            pending,
            drafted,
            a1,
        }
    }

    fn eos(t: &u32) -> bool {
        *t == 99
    }

    #[test]
    fn accepts_chain_two_tokens_per_round() {
        let w = walk_records(10, &eos, &[rec(true, 12, 11, 11), rec(true, 14, 13, 13)]);
        assert_eq!(w.ids, vec![10, 11, 12, 13]);
        assert_eq!(w.pending, 14);
        assert_eq!(w.pos_delta, 4);
        assert_eq!(w.accepts, 2);
        assert!(!w.eos_hit);
    }

    #[test]
    fn reject_emits_only_the_pending() {
        // a1 becomes the next round's input; the plain tail emits it.
        let w = walk_records(10, &eos, &[rec(false, 11, 7, 11)]);
        assert_eq!(w.ids, vec![10]);
        assert_eq!(w.pending, 11);
        assert_eq!(w.pos_delta, 1);
        assert_eq!(w.accepts, 0);
    }

    #[test]
    fn eos_on_the_draft_ends_output_but_counts_positions() {
        let w = walk_records(10, &eos, &[rec(true, 12, 99, 99), rec(false, 20, 5, 20)]);
        assert_eq!(w.ids, vec![10, 99]);
        assert_eq!(w.pos_delta, 3);
        assert!(w.eos_hit);
    }

    #[test]
    fn eos_pending_ends_before_any_round() {
        let w = walk_records(99, &eos, &[rec(true, 12, 11, 11)]);
        assert_eq!(w.ids, vec![99]);
        assert!(w.eos_hit);
    }
}
