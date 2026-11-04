use super::Device;
use super::contracts::{Tensor, TensorMut};
use super::weights::Music3SynWeights;
use crate::backend::gpu::{CudaFunction, DevicePtr, DevicePtrMut, PushKernelArg};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use std::sync::Arc;

/// Native MiniMax-Music3 synthesis engine: condition encoder, F32 flow
/// transformer, FlowMatchEuler scheduler and vocoder, fed the condition
/// stream produced by the autoregressive stage. Linears run on the shared
/// `Blas::linear_f32` (CUBLAS_COMPUTE_32F, no TF32); attention uses a
/// dedicated handle pinned to CUBLAS_DEFAULT_MATH.
pub struct Music3Syn {
    pub spec: crate::music3::SynSpec,
    pub weights: Music3SynWeights,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub blas: super::blas::Blas,
    concat: CudaFunction,
    tail: CudaFunction,
    layernorm: CudaFunction,
    fourier: CudaFunction,
    softmax: CudaFunction,
    rotary: CudaFunction,
    silu: CudaFunction,
    silu_mul: CudaFunction,
    bias: CudaFunction,
    add_scale: CudaFunction,
    cond_sum: CudaFunction,
    pad_rows_n: CudaFunction,
    nearest_rows: CudaFunction,
    snake_rows: CudaFunction,
    tanh_fn: CudaFunction,
    transpose: CudaFunction,
    im2col: CudaFunction,
    conv_transpose: CudaFunction,
    guided_euler: CudaFunction,
    cond_weights: CudaSlice<f32>,
    cond_scale: f32,
    guidance_scale: f32,
    vocoder_folds: VocoderFolds,
    max_length: usize,
}

/// One conv1d call site on the published folded [out, in*k] layout.
pub struct Conv1dSite<'a> {
    pub w: &'a CudaSlice<f32>,
    pub x: &'a CudaSlice<f32>,
    pub rows: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub out: &'a mut CudaSlice<f32>,
    pub bias: &'a CudaSlice<u8>,
    pub dilation: usize,
    pub padded: &'a mut CudaSlice<f32>,
    pub cols: &'a mut CudaSlice<f32>,
}

/// k=3 condition conv call site on the published [out, in*3] bytes.
struct K3Site<'a> {
    w: &'a CudaSlice<u8>,
    x: &'a CudaSlice<f32>,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    out: &'a mut CudaSlice<f32>,
    padded: &'a mut CudaSlice<f32>,
    cols: &'a mut CudaSlice<f32>,
}

/// Preallocated activation buffers for the synthesis path, sized at load
/// for `max_length` latents (pair rows = 2 * (max_length + 1) tokens).
pub struct Workspace {
    pub hidden: CudaSlice<f32>,
    pub pre: CudaSlice<f32>,
    pub wide: CudaSlice<f32>,
    pub norm: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    pub scores: CudaSlice<f32>,
    pub ff_buf: CudaSlice<f32>,
    pub ff_gated: CudaSlice<f32>,
    pub ff_out: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    pub post: CudaSlice<f32>,
    pub fourier: CudaSlice<f32>,
    pub t1: CudaSlice<f32>,
    pub temb: CudaSlice<f32>,
    pub x: CudaSlice<f32>,
    pub pair_v_c: CudaSlice<f32>,
    pub pair_v_u: CudaSlice<f32>,
    pub cond: CudaSlice<f32>,
    pub zeros_cond: CudaSlice<f32>,
    pub cond_fused: CudaSlice<f32>,
    pub cond_conv: CudaSlice<f32>,
    pub cond_pad: CudaSlice<f32>,
    pub cond_cols: CudaSlice<f32>,
    pub rotary_cos: CudaSlice<f32>,
    pub rotary_sin: CudaSlice<f32>,
    pub voc_x: CudaSlice<f32>,
    pub voc_dec: CudaSlice<f32>,
    pub voc_cur: CudaSlice<f32>,
    pub voc_next: CudaSlice<f32>,
    pub voc_sn1: CudaSlice<f32>,
    pub voc_conv: CudaSlice<f32>,
    pub voc_sn2: CudaSlice<f32>,
    pub voc_gated: CudaSlice<f32>,
    pub voc_pad: CudaSlice<f32>,
    pub voc_cols: CudaSlice<f32>,
    pub voc_out: CudaSlice<f32>,
}

fn f32_matrix(
    bytes: &CudaSlice<u8>,
    s: &Arc<cudarc::driver::CudaStream>,
    rows: usize,
    cols: usize,
) -> Result<Tensor<f32, 2>> {
    ensure!(
        bytes.len() == rows * cols * 4,
        "F32 weight byte extent mismatch"
    );
    unsafe {
        Tensor::new(
            bytes.device_ptr(s).0,
            [u32::try_from(rows)?, u32::try_from(cols)?],
            [cols as u64, 1],
            (rows * cols * 4) as u64,
        )
    }
}

impl Music3Syn {
    pub fn load(dir: &std::path::Path, d: &Device, max_length: usize) -> Result<Self> {
        let spec = crate::music3::SynSpec::read(dir)?;
        let weights = Music3SynWeights::load(dir, &spec, d)?;
        let ops = super::ops::Ops::new(&d.ctx)?;
        let closure = std::rc::Rc::new(super::closure::Closure::new(
            false,
            "Music3Syn setup".into(),
        ));
        let blas = super::blas::Blas::new(d, &ops, std::path::Path::new(""), closure)?;
        // The attention GEMMs share the Blas handle: CUBLAS_COMPUTE_32F
        // (no TF32) is the compute type of every F32 GEMM in this engine.
        let common = super::cubin::module(&d.ctx, "q38_common")?;
        let act = super::cubin::module(&d.ctx, "act")?;
        let rope = super::cubin::module(&d.ctx, "rope")?;
        let kernel = |m: &std::sync::Arc<cudarc::driver::CudaModule>, name: &str| {
            m.load_function(name)
                .with_context(|| format!("loading kernel {name}"))
        };
        let logits_bytes = d.stream.clone_dtoh(&weights.cond.layer_weight_logits)?;
        let logits: Vec<f32> = logits_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let exp_sum: f32 = logits.iter().map(|l| l.exp()).sum();
        let softmax: Vec<f32> = logits.iter().map(|l| l.exp() / exp_sum).collect();
        let cond_weights = d.stream.clone_htod(&softmax)?;
        let scale_bytes = d.stream.clone_dtoh(&weights.cond.layer_scale)?;
        let cond_scale = f32::from_le_bytes(scale_bytes[0..4].try_into()?);
        let wn_fold = kernel(&common, "weight_norm_fold_f32")?;
        let s = &d.stream;
        let fold = |g: &CudaSlice<u8>, v: &CudaSlice<u8>, channels: usize, rest: usize| {
            fold_weight_norm(s, &wn_fold, g, v, channels, rest)
        };
        let vc = &weights.vocoder;
        let hidden = spec.decoder_hidden as usize;
        let input_dim = spec.decoder_input as usize;
        // The encoder conv is not weight-normed: the published [out, in*k]
        // bytes are the folded layout already.
        let conv_in_w = fold(&vc.conv_in_g, &vc.conv_in_v, hidden, input_dim * 7)?;
        let mut blocks = Vec::with_capacity(vc.blocks.len());
        let mut stage_in = hidden;
        for vb in vc.blocks.iter().zip(&spec.upsampling) {
            let (block, ratio) = vb;
            let stage_out = stage_in / 2;
            let stride = *ratio as usize;
            let conv_t_w = fold(
                &block.conv_t_g,
                &block.conv_t_v,
                stage_in,
                stage_out * 2 * stride,
            )?;
            let mut units = Vec::with_capacity(block.units.len());
            for u in &block.units {
                let c1_w = fold(&u.c1_g, &u.c1_v, stage_out, stage_out * 7)?;
                let c2_w = fold(&u.c2_g, &u.c2_v, stage_out, stage_out)?;
                units.push(VocoderUnitFold { c1_w, c2_w });
            }
            blocks.push(VocoderBlockFold { conv_t_w, units });
            stage_in = stage_out;
        }
        let tail_dim = stage_in;
        let conv_out_w = fold(&vc.conv_out_g, &vc.conv_out_v, 1, tail_dim * 7)?;
        let vocoder_folds = VocoderFolds {
            conv_in_w,
            blocks,
            conv_out_w,
        };
        let vision = super::cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            spec,
            weights,
            stream: d.stream.clone(),
            blas,
            concat: kernel(&common, "concat_channels_f32")?,
            tail: kernel(&common, "transpose_residual_f32")?,
            layernorm: kernel(&common, "q38_layer_norm_f32")?,
            fourier: kernel(&common, "fourier_time_f32")?,
            softmax: kernel(&common, "softmax_rows_f32")?,
            rotary: kernel(&rope, "partial_rotary_f32")?,
            silu: kernel(&act, "silu_f32")?,
            silu_mul: kernel(&act, "silu_mul_f32")?,
            bias: kernel(&vision, "vision_bias_f32")?,
            add_scale: kernel(&common, "add_scale_f32")?,
            cond_sum: kernel(&common, "weighted_layer_sum_f32")?,
            pad_rows_n: kernel(&common, "pad_rows_n_f32")?,
            nearest_rows: kernel(&common, "nearest_rows_f32")?,
            snake_rows: kernel(&common, "snake_rows_f32")?,
            tanh_fn: kernel(&common, "tanh_f32")?,
            transpose: kernel(&common, "transpose_f32")?,
            im2col: kernel(&common, "conv_im2col_f32")?,
            conv_transpose: kernel(&common, "conv_transpose_rows_f32")?,
            guided_euler: kernel(&common, "guided_euler_f32")?,
            cond_weights,
            cond_scale,
            // Guidance 1.7 and 30 steps are the model definition's
            // denoise defaults (denoise.py), not published config keys.
            guidance_scale: 1.7,
            vocoder_folds,
            max_length,
        })
    }

    fn linear(
        &self,
        weight: &CudaSlice<u8>,
        out_rows: usize,
        in_cols: usize,
        input: &CudaView<f32>,
        rows: usize,
        out: &mut CudaViewMut<f32>,
    ) -> Result<()> {
        let mut view = TensorMut(unsafe {
            Tensor::new(
                out.device_ptr_mut(&self.stream).0,
                [rows as u32, out_rows as u32],
                [out_rows as u64, 1],
                (rows * out_rows * 4) as u64,
            )?
        });
        self.blas.linear_f32(
            f32_matrix(weight, &self.stream, out_rows, in_cols)?,
            unsafe {
                Tensor::new(
                    input.device_ptr(&self.stream).0,
                    [rows as u32, in_cols as u32],
                    [in_cols as u64, 1],
                    (rows * in_cols * 4) as u64,
                )?
            },
            &mut view,
        )
    }

    /// One flow-transformer call: `x_t` and `velocity` are channel-major
    /// [in_channels, length]; `cond` is row-major [length, condition_dim];
    /// `rotary` holds (cos, sin) tables for the window positions, each
    /// [1 + length, rotary_dim] with the temb token at position 0.
    pub fn flow_velocity(
        &self,
        ws: &mut Workspace,
        x_t: &CudaSlice<f32>,
        t: f32,
        cond: &CudaSlice<f32>,
        length: usize,
        velocity: &mut CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(length <= self.max_length, "flow length exceeds workspace");
        let s = &self.stream;
        let dim = self.spec.flow_dim() as usize;
        let concat_width = self.spec.concat_channels() as usize;
        let ff = self.spec.flow_ff_inner as usize;
        let channels = self.spec.in_channels as usize;
        let heads = self.spec.flow_heads as usize;
        let head_dim = self.spec.flow_head_dim as usize;
        let rotary_dim = self.spec.rotary_dim as usize;
        let fourier_dim = self.spec.fourier_dim as usize;
        let tokens = length + 1;
        ensure!(
            x_t.len() == channels * length
                && cond.len() == length * dim
                && velocity.len() == channels * length
                && ws.rotary_cos.len() >= tokens * rotary_dim
                && ws.rotary_sin.len() >= tokens * rotary_dim,
            "flow_velocity extent mismatch"
        );

        // Channel concat, preprocess conv, residual.
        let mut hidden = ws.hidden.slice_mut(0..ws.hidden.len());
        let mut pre = ws.pre.slice_mut(0..ws.pre.len());
        unsafe {
            s.launch_builder(&self.concat)
                .arg(x_t)
                .arg(cond)
                .arg(&mut hidden)
                .arg(&(length as i64))
                .arg(&(channels as i32))
                .arg(&(dim as i32))
                .launch(super::ops::flat(concat_width * length))?;
        }
        self.linear(
            &self.weights.flow.preprocess,
            concat_width,
            concat_width,
            &hidden.slice(0..concat_width * length),
            length,
            &mut pre.slice_mut(0..concat_width * length),
        )?;
        unsafe {
            s.launch_builder(&self.add_scale)
                .arg(&mut pre)
                .arg(&mut hidden)
                .arg(&((concat_width * length) as i64))
                .arg(&1.0f32)
                .launch(super::ops::flat(concat_width * length))?;
        }

        // proj_in writes rows 1..=length of the token buffer; row 0 is temb.
        let mut wide = ws.wide.slice_mut(0..ws.wide.len());
        self.linear(
            &self.weights.flow.proj_in,
            dim,
            concat_width,
            &pre.slice(0..concat_width * length),
            length,
            &mut wide.slice_mut(dim..dim + length * dim),
        )?;
        let temb_ptr = ws.temb.device_ptr(s).0;
        let l2b_ptr = self.weights.flow.time_l2_b.device_ptr(s).0;
        let mut temb = ws.temb.slice_mut(0..ws.temb.len());
        let mut fourier = ws.fourier.slice_mut(0..ws.fourier.len());
        unsafe {
            s.launch_builder(&self.fourier)
                .arg(&t)
                .arg(&self.weights.flow.time_proj)
                .arg(&mut fourier)
                .arg(&((fourier_dim / 2) as u32))
                .launch(super::ops::flat(fourier_dim / 2))?;
        }
        let t1_ptr = ws.t1.device_ptr(s).0;
        let l1b_ptr = self.weights.flow.time_l1_b.device_ptr(s).0;
        let mut t1 = ws.t1.slice_mut(0..ws.t1.len());
        self.linear(
            &self.weights.flow.time_l1_w,
            dim,
            fourier_dim,
            &fourier.slice(0..fourier_dim),
            1,
            &mut t1.slice_mut(0..dim),
        )?;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&t1_ptr)
                .arg(&l1b_ptr)
                .arg(&t1_ptr)
                .arg(&1u32)
                .arg(&(dim as u32))
                .launch(super::ops::flat(dim))?;
            s.launch_builder(&self.silu)
                .arg(&mut t1)
                .arg(&(dim as i64))
                .launch(super::ops::flat(dim))?;
        }
        self.linear(
            &self.weights.flow.time_l2_w,
            dim,
            dim,
            &t1.slice(0..dim),
            1,
            &mut temb.slice_mut(0..dim),
        )?;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&temb_ptr)
                .arg(&l2b_ptr)
                .arg(&temb_ptr)
                .arg(&1u32)
                .arg(&(dim as u32))
                .launch(super::ops::flat(dim))?;
        }
        s.memcpy_dtod(&temb, &mut wide)?;

        let mut norm = ws.norm.slice_mut(0..ws.norm.len());
        let mut q = ws.q.slice_mut(0..ws.q.len());
        let mut k = ws.k.slice_mut(0..ws.k.len());
        let mut v = ws.v.slice_mut(0..ws.v.len());
        let mut attn = ws.attn.slice_mut(0..ws.attn.len());
        let mut scores = ws.scores.slice_mut(0..ws.scores.len());
        let ffb_ptr = ws.ff_buf.device_ptr(s).0;
        let mut ff_buf = ws.ff_buf.slice_mut(0..ws.ff_buf.len());
        let mut ff_gated = ws.ff_gated.slice_mut(0..ws.ff_gated.len());
        let ffo_ptr = ws.ff_out.device_ptr(s).0;
        let mut ff_out = ws.ff_out.slice_mut(0..ws.ff_out.len());
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for block in &self.weights.flow.blocks {
            self.layernorm(
                wide.device_ptr(s).0,
                &block.norm1_w,
                &block.norm1_b,
                tokens,
                dim,
                norm.device_ptr_mut(s).0,
            )?;
            self.linear(
                &block.to_q,
                dim,
                dim,
                &norm.slice(0..dim * tokens),
                tokens,
                &mut q.slice_mut(0..dim * tokens),
            )?;
            self.linear(
                &block.to_k,
                dim,
                dim,
                &norm.slice(0..dim * tokens),
                tokens,
                &mut k.slice_mut(0..dim * tokens),
            )?;
            self.linear(
                &block.to_v,
                dim,
                dim,
                &norm.slice(0..dim * tokens),
                tokens,
                &mut v.slice_mut(0..dim * tokens),
            )?;
            for buffer in [&mut q, &mut k] {
                unsafe {
                    s.launch_builder(&self.rotary)
                        .arg(buffer)
                        .arg(&ws.rotary_cos)
                        .arg(&ws.rotary_sin)
                        .arg(&(tokens as i32))
                        .arg(&(heads as i32))
                        .arg(&(head_dim as i32))
                        .arg(&(rotary_dim as i32))
                        .launch(super::ops::flat(tokens * heads * rotary_dim))?;
                }
            }
            let q_ptr = q.device_ptr(s).0;
            let k_ptr = k.device_ptr(s).0;
            let v_ptr = v.device_ptr(s).0;
            let attn_ptr = attn.device_ptr_mut(s).0;
            let scores_ptr = scores.device_ptr_mut(s).0;
            for head in 0..heads {
                let off = 4 * (head * head_dim) as u64;
                let op_t = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
                let op_n = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
                unsafe {
                    self.blas.gemm_f32(&super::blas::GemmF32Raw {
                        transa: op_t,
                        transb: op_n,
                        m: tokens as i32,
                        n: tokens as i32,
                        k: head_dim as i32,
                        alpha: scale,
                        a: k_ptr + off,
                        lda: dim as i32,
                        b: q_ptr + off,
                        ldb: dim as i32,
                        c: scores_ptr,
                        ldc: tokens as i32,
                    })?;
                }
                unsafe {
                    s.launch_builder(&self.softmax)
                        .arg(&mut scores)
                        .arg(&(tokens as u32))
                        .arg(&(tokens as u32))
                        .arg(&(tokens as u64))
                        .launch(super::ops::grid(tokens, 256))?;
                }
                unsafe {
                    self.blas.gemm_f32(&super::blas::GemmF32Raw {
                        transa: op_n,
                        transb: op_n,
                        m: head_dim as i32,
                        n: tokens as i32,
                        k: tokens as i32,
                        alpha: 1.0,
                        a: v_ptr + off,
                        lda: dim as i32,
                        b: scores_ptr,
                        ldb: tokens as i32,
                        c: attn_ptr + off,
                        ldc: dim as i32,
                    })?;
                }
            }
            self.linear(
                &block.to_out,
                dim,
                dim,
                &attn.slice(0..dim * tokens),
                tokens,
                &mut ff_out.slice_mut(0..dim * tokens),
            )?;
            unsafe {
                s.launch_builder(&self.add_scale)
                    .arg(&mut wide)
                    .arg(&mut ff_out)
                    .arg(&((dim * tokens) as i64))
                    .arg(&1.0f32)
                    .launch(super::ops::flat(dim * tokens))?;
            }

            self.layernorm(
                wide.device_ptr(s).0,
                &block.norm2_w,
                &block.norm2_b,
                tokens,
                dim,
                norm.device_ptr_mut(s).0,
            )?;
            self.linear(
                &block.ff_in_w,
                2 * ff,
                dim,
                &norm.slice(0..dim * tokens),
                tokens,
                &mut ff_buf.slice_mut(0..2 * ff * tokens),
            )?;
            unsafe {
                let ffinb_ptr = block.ff_in_b.device_ptr(s).0;
                s.launch_builder(&self.bias)
                    .arg(&ffb_ptr)
                    .arg(&ffinb_ptr)
                    .arg(&ffb_ptr)
                    .arg(&(tokens as u32))
                    .arg(&((2 * ff) as u32))
                    .launch(super::ops::flat(2 * ff * tokens))?;
                s.launch_builder(&self.silu_mul)
                    .arg(&mut ff_buf)
                    .arg(&mut ff_gated)
                    .arg(&(tokens as i64))
                    .arg(&(ff as i32))
                    .launch(super::ops::flat(ff * tokens))?;
            }
            self.linear(
                &block.ff_out_w,
                dim,
                ff,
                &ff_gated.slice(0..ff * tokens),
                tokens,
                &mut ff_out.slice_mut(0..dim * tokens),
            )?;
            unsafe {
                let ffob_ptr = block.ff_out_b.device_ptr(s).0;
                s.launch_builder(&self.bias)
                    .arg(&ffo_ptr)
                    .arg(&ffob_ptr)
                    .arg(&ffo_ptr)
                    .arg(&(tokens as u32))
                    .arg(&(dim as u32))
                    .launch(super::ops::flat(dim * tokens))?;
                s.launch_builder(&self.add_scale)
                    .arg(&mut wide)
                    .arg(&mut ff_out)
                    .arg(&((dim * tokens) as i64))
                    .arg(&1.0f32)
                    .launch(super::ops::flat(dim * tokens))?;
            }
        }

        // proj_out over rows 1.., postprocess conv, transpose + residual.
        let mut proj = ws.proj.slice_mut(0..ws.proj.len());
        self.linear(
            &self.weights.flow.proj_out,
            channels,
            dim,
            &wide.slice(dim..dim + length * dim),
            length,
            &mut proj.slice_mut(0..channels * length),
        )?;
        let mut post = ws.post.slice_mut(0..ws.post.len());
        self.linear(
            &self.weights.flow.postprocess,
            channels,
            channels,
            &proj.slice(0..channels * length),
            length,
            &mut post.slice_mut(0..channels * length),
        )?;
        unsafe {
            s.launch_builder(&self.tail)
                .arg(&mut post)
                .arg(&mut proj)
                .arg(velocity)
                .arg(&(length as i64))
                .arg(&(channels as i32))
                .launch(super::ops::flat(channels * length))?;
        }
        Ok(())
    }

    /// Flow sigma schedule for `steps` Euler steps: numpy-semantics
    /// linspace(steps ascending 1/steps..1 in f64, cast f32) with the
    /// terminal 0 first, 31 entries for 30 steps.
    pub fn sigma_schedule(steps: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; steps + 1];
        let start = 1.0f64 / steps as f64;
        let step = (1.0 - start) / (steps as f64 - 1.0);
        for i in 0..steps {
            out[i + 1] = (start + i as f64 * step) as f32;
        }
        out[steps] = 1.0;
        out
    }

    /// One batched CFG flow call: branch 0 conditions on `cond`, branch 1 on
    /// zeros; both share `x_t` (one weight read per pair). Writes branch
    /// velocities channel-major [in_channels, length] each.
    pub fn flow_velocity_pair(&self, ws: &mut Workspace, t: f32, length: usize) -> Result<()> {
        let s = &self.stream;
        let dim = self.spec.flow_dim() as usize;
        let concat_width = self.spec.concat_channels() as usize;
        let channels = self.spec.in_channels as usize;
        let ff = self.spec.flow_ff_inner as usize;
        let fourier_dim = self.spec.fourier_dim as usize;
        let heads = self.spec.flow_heads as usize;
        let head_dim = self.spec.flow_head_dim as usize;
        let rotary_dim = self.spec.rotary_dim as usize;
        let tokens = length + 1;
        let pair_tokens = 2 * tokens;
        ensure!(length <= self.max_length, "flow_velocity_pair length");
        let cond_branch: &CudaSlice<f32> = &ws.cond;
        let zeros_branch: &CudaSlice<f32> = &ws.zeros_cond;

        // Branch concat halves into one [2*length, concat_width] buffer.
        // Branch 0 carries ws.cond, branch 1 the zero condition.
        for branch in 0..2 {
            let x_t: &CudaSlice<f32> = &ws.x;
            let cond_in = if branch == 0 {
                cond_branch
            } else {
                zeros_branch
            };
            let mut hidden = ws.hidden.slice_mut(0..2 * length * concat_width);
            let base = branch * length * concat_width;
            let mut half = hidden.slice_mut(base..base + length * concat_width);
            unsafe {
                s.launch_builder(&self.concat)
                    .arg(x_t)
                    .arg(cond_in)
                    .arg(&mut half)
                    .arg(&(length as i64))
                    .arg(&(channels as i32))
                    .arg(&(dim as i32))
                    .launch(super::ops::flat(length * concat_width))?;
            }
        }
        let hidden_ro = ws.hidden.slice(0..2 * length * concat_width);
        let mut pre = ws.pre.slice_mut(0..2 * length * concat_width);
        self.linear(
            &self.weights.flow.preprocess,
            concat_width,
            concat_width,
            &hidden_ro,
            2 * length,
            &mut pre,
        )?;
        let count_pre = pre.len();
        let hidden_ptr = ws.hidden.device_ptr(s).0;
        let pre_ptr = ws.pre.device_ptr(s).0;
        unsafe {
            s.launch_builder(&self.add_scale)
                .arg(&pre_ptr)
                .arg(&hidden_ptr)
                .arg(&((count_pre) as i64))
                .arg(&1.0f32)
                .launch(super::ops::flat(count_pre))?;
        }

        // proj_in writes rows 1..=length of each branch token window;
        // row tokens of window 1 is its temb row, not body row 0.
        for branch in 0..2 {
            let pre_branch = ws
                .pre
                .slice(branch * length * concat_width..(branch + 1) * length * concat_width);
            let base = (branch * tokens + 1) * dim;
            let mut wide = ws.wide.slice_mut(base..base + length * dim);
            self.linear(
                &self.weights.flow.proj_in,
                dim,
                concat_width,
                &pre_branch,
                length,
                &mut wide,
            )?;
        }
        let mut fourier = ws.fourier.slice_mut(0..fourier_dim);
        unsafe {
            s.launch_builder(&self.fourier)
                .arg(&t)
                .arg(&self.weights.flow.time_proj)
                .arg(&mut fourier)
                .arg(&((fourier_dim / 2) as u32))
                .launch(super::ops::flat(fourier_dim / 2))?;
        }
        let fourier_ro = ws.fourier.slice(0..fourier_dim);
        let mut t1 = ws.t1.slice_mut(0..dim);
        self.linear(
            &self.weights.flow.time_l1_w,
            dim,
            fourier_dim,
            &fourier_ro,
            1,
            &mut t1,
        )?;
        let t1_ptr = ws.t1.device_ptr(s).0;
        let l1b_ptr = self.weights.flow.time_l1_b.device_ptr(s).0;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&t1_ptr)
                .arg(&l1b_ptr)
                .arg(&t1_ptr)
                .arg(&1u32)
                .arg(&(dim as u32))
                .launch(super::ops::flat(dim))?;
            s.launch_builder(&self.silu)
                .arg(&t1_ptr)
                .arg(&(dim as i64))
                .launch(super::ops::flat(dim))?;
        }
        let t1_ro = ws.t1.slice(0..dim);
        let mut temb_out = ws.temb.slice_mut(0..dim);
        self.linear(
            &self.weights.flow.time_l2_w,
            dim,
            dim,
            &t1_ro,
            1,
            &mut temb_out,
        )?;
        let temb_ptr = ws.temb.device_ptr(s).0;
        let l2b_ptr = self.weights.flow.time_l2_b.device_ptr(s).0;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&temb_ptr)
                .arg(&l2b_ptr)
                .arg(&temb_ptr)
                .arg(&1u32)
                .arg(&(dim as u32))
                .launch(super::ops::flat(dim))?;
        }
        let wide_ptr = ws.wide.device_ptr(s).0;
        let temb_ro = ws.temb.slice(0..dim);
        for branch in 0..2 {
            let base = branch * tokens * dim;
            let mut dst = ws.wide.slice_mut(base..base + dim);
            s.memcpy_dtod(&temb_ro, &mut dst)?;
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for block in &self.weights.flow.blocks {
            self.layernorm(
                wide_ptr,
                &block.norm1_w,
                &block.norm1_b,
                pair_tokens,
                dim,
                ws.norm.device_ptr_mut(s).0,
            )?;
            let norm_ro = ws.norm.slice(0..pair_tokens * dim);
            let mut q = ws.q.slice_mut(0..pair_tokens * dim);
            let mut k = ws.k.slice_mut(0..pair_tokens * dim);
            let mut v = ws.v.slice_mut(0..pair_tokens * dim);
            self.linear(&block.to_q, dim, dim, &norm_ro, pair_tokens, &mut q)?;
            self.linear(&block.to_k, dim, dim, &norm_ro, pair_tokens, &mut k)?;
            self.linear(&block.to_v, dim, dim, &norm_ro, pair_tokens, &mut v)?;
            let q_ptr = ws.q.device_ptr(s).0;
            let k_ptr = ws.k.device_ptr(s).0;
            let v_ptr = ws.v.device_ptr(s).0;
            let attn_ptr = ws.attn.device_ptr_mut(s).0;
            let scores_ptr = ws.scores.device_ptr_mut(s).0;
            let cos_ptr = ws.rotary_cos.device_ptr(s).0;
            let sin_ptr = ws.rotary_sin.device_ptr(s).0;
            // Each branch window rotates its own rows 0..tokens against
            // the shared single-branch table.
            for buffer in [q_ptr, k_ptr] {
                for branch in 0..2 {
                    let base = buffer + 4 * (branch * tokens * dim) as u64;
                    unsafe {
                        s.launch_builder(&self.rotary)
                            .arg(&base)
                            .arg(&cos_ptr)
                            .arg(&sin_ptr)
                            .arg(&(tokens as i32))
                            .arg(&(heads as i32))
                            .arg(&(head_dim as i32))
                            .arg(&(rotary_dim as i32))
                            .launch(super::ops::flat(tokens * heads * rotary_dim))?;
                    }
                }
            }
            for head in 0..heads {
                use cudarc::cublas::sys::cublasOperation_t as Op;
                let op_t = Op::CUBLAS_OP_T;
                let op_n = Op::CUBLAS_OP_N;
                let off = 4 * (head * head_dim) as u64;
                for branch in 0..2 {
                    let base = 4 * (branch * tokens * dim) as u64;
                    unsafe {
                        self.blas.gemm_f32(&super::blas::GemmF32Raw {
                            transa: op_t,
                            transb: op_n,
                            m: tokens as i32,
                            n: tokens as i32,
                            k: head_dim as i32,
                            alpha: scale,
                            a: k_ptr + off + base,
                            lda: dim as i32,
                            b: q_ptr + off + base,
                            ldb: dim as i32,
                            c: scores_ptr + 4 * (branch * tokens * tokens) as u64,
                            ldc: tokens as i32,
                        })?;
                    }
                }
                unsafe {
                    s.launch_builder(&self.softmax)
                        .arg(&scores_ptr)
                        .arg(&(2 * tokens as u32))
                        .arg(&(tokens as u32))
                        .arg(&(tokens as u64))
                        .launch(super::ops::grid(2 * tokens, 256))?;
                }
                for branch in 0..2 {
                    let base = 4 * (branch * tokens * dim) as u64;
                    unsafe {
                        self.blas.gemm_f32(&super::blas::GemmF32Raw {
                            transa: op_n,
                            transb: op_n,
                            m: head_dim as i32,
                            n: tokens as i32,
                            k: tokens as i32,
                            alpha: 1.0,
                            a: v_ptr + off + base,
                            lda: dim as i32,
                            b: scores_ptr + 4 * (branch * tokens * tokens) as u64,
                            ldb: tokens as i32,
                            c: attn_ptr + off + base,
                            ldc: dim as i32,
                        })?;
                    }
                }
            }
            let attn_ro = ws.attn.slice(0..pair_tokens * dim);
            let mut ff_out = ws.ff_out.slice_mut(0..pair_tokens * dim);
            self.linear(&block.to_out, dim, dim, &attn_ro, pair_tokens, &mut ff_out)?;
            let count = pair_tokens * dim;
            let wide2 = ws.wide.device_ptr(s).0;
            let ffo = ws.ff_out.device_ptr(s).0;
            unsafe {
                s.launch_builder(&self.add_scale)
                    .arg(&wide2)
                    .arg(&ffo)
                    .arg(&((count) as i64))
                    .arg(&1.0f32)
                    .launch(super::ops::flat(count))?;
            }
            self.layernorm(
                wide_ptr,
                &block.norm2_w,
                &block.norm2_b,
                pair_tokens,
                dim,
                ws.norm.device_ptr_mut(s).0,
            )?;
            let norm_ro = ws.norm.slice(0..pair_tokens * dim);
            let mut ff_buf = ws.ff_buf.slice_mut(0..pair_tokens * 2 * ff);
            self.linear(
                &block.ff_in_w,
                2 * ff,
                dim,
                &norm_ro,
                pair_tokens,
                &mut ff_buf,
            )?;
            let count_ff = pair_tokens * 2 * ff;
            let ffb_ptr = ws.ff_buf.device_ptr(s).0;
            let ffinb_ptr = block.ff_in_b.device_ptr(s).0;
            unsafe {
                s.launch_builder(&self.bias)
                    .arg(&ffb_ptr)
                    .arg(&ffinb_ptr)
                    .arg(&ffb_ptr)
                    .arg(&(pair_tokens as u32))
                    .arg(&((2 * ff) as u32))
                    .launch(super::ops::flat(count_ff))?;
                let gated_ptr = ws.ff_gated.device_ptr_mut(s).0;
                s.launch_builder(&self.silu_mul)
                    .arg(&ffb_ptr)
                    .arg(&gated_ptr)
                    .arg(&(pair_tokens as i64))
                    .arg(&(ff as i32))
                    .launch(super::ops::flat(pair_tokens * ff))?;
            }
            let gated_ro = ws.ff_gated.slice(0..pair_tokens * ff);
            let mut ff_out2 = ws.ff_out.slice_mut(0..pair_tokens * dim);
            self.linear(
                &block.ff_out_w,
                dim,
                ff,
                &gated_ro,
                pair_tokens,
                &mut ff_out2,
            )?;
            let count2 = pair_tokens * dim;
            let ffo2 = ws.ff_out.device_ptr(s).0;
            let ffob_ptr = block.ff_out_b.device_ptr(s).0;
            unsafe {
                s.launch_builder(&self.bias)
                    .arg(&ffo2)
                    .arg(&ffob_ptr)
                    .arg(&ffo2)
                    .arg(&(pair_tokens as u32))
                    .arg(&(dim as u32))
                    .launch(super::ops::flat(count2))?;
                s.launch_builder(&self.add_scale)
                    .arg(&wide2)
                    .arg(&ffo2)
                    .arg(&((count2) as i64))
                    .arg(&1.0f32)
                    .launch(super::ops::flat(count2))?;
            }
        }

        // proj_out over rows 1.. of each branch, postprocess, transpose tail.
        for branch in 0..2 {
            let base = 4 * (branch * tokens * dim + dim) as u64;
            let (wide_body, mut proj_view) = unsafe {
                (
                    Tensor::<f32, 2>::new(
                        ws.wide.device_ptr(s).0 + base,
                        [length as u32, dim as u32],
                        [dim as u64, 1],
                        (length * dim * 4) as u64,
                    )?,
                    TensorMut(Tensor::<f32, 2>::new(
                        ws.proj.device_ptr_mut(s).0 + 4 * (branch * length * channels) as u64,
                        [length as u32, channels as u32],
                        [channels as u64, 1],
                        (length * channels * 4) as u64,
                    )?),
                )
            };
            let weight_t = f32_matrix(&self.weights.flow.proj_out, s, channels, dim)?;
            self.blas.linear_f32(weight_t, wide_body, &mut proj_view)?;
        }
        let proj_ro = ws.proj.slice(0..2 * length * channels);
        let mut post = ws.post.slice_mut(0..2 * length * channels);
        self.linear(
            &self.weights.flow.postprocess,
            channels,
            channels,
            &proj_ro,
            2 * length,
            &mut post,
        )?;
        let post_ptr = ws.post.device_ptr(s).0;
        let proj_ptr = ws.proj.device_ptr(s).0;
        let v0 = ws.pair_v_c.device_ptr_mut(s).0;
        let v1 = ws.pair_v_u.device_ptr_mut(s).0;
        unsafe {
            s.launch_builder(&self.tail)
                .arg(&post_ptr)
                .arg(&proj_ptr)
                .arg(&v0)
                .arg(&(length as i64))
                .arg(&(channels as i32))
                .launch(super::ops::flat(channels * length))?;
            s.launch_builder(&self.tail)
                .arg(&(post_ptr + 4 * (length * channels) as u64))
                .arg(&(proj_ptr + 4 * (length * channels) as u64))
                .arg(&v1)
                .arg(&(length as i64))
                .arg(&(channels as i32))
                .launch(super::ops::flat(channels * length))?;
        }
        Ok(())
    }

    /// Full synthesis: condition encode, 30 Euler steps with the CFG pair
    /// (one batched flow call per step), guided update, then the vocoder.
    /// `initial` is the initial latent [in_channels, length] channel-major;
    /// `audio` receives [2, length * product(upsampling)].
    pub fn synthesize(
        &self,
        ws: &mut Workspace,
        condition_input: &CudaSlice<f32>,
        frames: usize,
        initial: &CudaSlice<f32>,
        audio: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let s = &self.stream;
        let length = self.spec.latent_length(frames);
        ensure!(length <= self.max_length, "length exceeds workspace");
        self.condition_encode(ws, condition_input, frames)?;
        let channels = self.spec.in_channels as usize;
        ensure!(
            initial.len() == channels * length,
            "initial latent extent: got {} elements, in_channels * latent_length = {}",
            initial.len(),
            channels * length
        );
        let sigmas = Self::sigma_schedule(30);
        s.memcpy_dtod(initial, &mut ws.x.slice_mut(0..channels * length))?;
        // Pair k runs at t = sigmas[k] and applies x +=
        // (sigmas[k+1] - sigmas[k]) * guided(v_pair_k), k = 0..30.
        // The transition-29 output is the vocoder latent; no call input
        // of the reference loop observes it.
        for step in 0..30 {
            let t = sigmas[step];
            self.flow_velocity_pair(ws, t, length)?;
            let dt = sigmas[step + 1] - sigmas[step];
            self.guided_euler_update(&mut ws.x, &ws.pair_v_c, &ws.pair_v_u, dt, channels * length)?;
        }
        self.vocoder(ws, length, audio)?;
        Ok(())
    }

    /// Fused guided Euler update over the first `count` elements; the
    /// launch form synthesize and the euler gate share.
    pub(crate) fn guided_euler_update(
        &self,
        x: &mut CudaSlice<f32>,
        c: &CudaSlice<f32>,
        u: &CudaSlice<f32>,
        dt: f32,
        count: usize,
    ) -> Result<()> {
        let s = &self.stream;
        let x_ptr = x.device_ptr_mut(s).0;
        let c_ptr = c.device_ptr(s).0;
        let u_ptr = u.device_ptr(s).0;
        unsafe {
            s.launch_builder(&self.guided_euler)
                .arg(&x_ptr)
                .arg(&c_ptr)
                .arg(&u_ptr)
                .arg(&self.guidance_scale)
                .arg(&dt)
                .arg(&(count as i64))
                .arg(&0i32)
                .arg(&0u64)
                .launch(super::ops::flat(count))?;
        }
        Ok(())
    }

    /// Latent length for a frame count: int(frames * out_rate * in_hop /
    /// (in_rate * out_hop)), at least 1.
    /// Condition encoder: fused layer softmax-sum, k=3 conv and nearest
    /// upsample. `input` is [frames, layers*cond_hidden], `out`
    /// [latent_length, cond_out].
    pub fn condition_encode(
        &self,
        ws: &mut Workspace,
        input: &CudaSlice<f32>,
        frames: usize,
    ) -> Result<()> {
        let s = &self.stream;
        let cond_hidden = self.spec.cond_hidden as usize;
        let cond_out = self.spec.cond_out as usize;
        let layers = self.spec.cond_layers as usize;
        let length = self.spec.latent_length(frames);
        ensure!(
            input.len() == frames * layers * cond_hidden
                && ws.cond.len() >= length * cond_out
                && ws.cond_fused.len() >= frames * cond_hidden
                && ws.cond_conv.len() >= frames * cond_out,
            "condition_encode extent mismatch"
        );
        unsafe {
            s.launch_builder(&self.cond_sum)
                .arg(input)
                .arg(&self.cond_weights)
                .arg(&self.cond_scale)
                .arg(&mut ws.cond_fused.slice_mut(0..frames * cond_hidden))
                .arg(&(frames as i64))
                .arg(&(layers as i32))
                .arg(&(cond_hidden as i32))
                .launch(super::ops::flat(frames * cond_hidden))?;
        }
        self.conv1d_k3(K3Site {
            w: &self.weights.cond.proj_w,
            x: &ws.cond_fused,
            rows: frames,
            in_dim: cond_hidden,
            out_dim: cond_out,
            out: &mut ws.cond_conv,
            padded: &mut ws.cond_pad,
            cols: &mut ws.cond_cols,
        })?;
        let conv_ptr = ws.cond_conv.device_ptr(s).0;
        let conv_bias = self.weights.cond.proj_b.device_ptr(s).0;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&conv_ptr)
                .arg(&conv_bias)
                .arg(&conv_ptr)
                .arg(&(frames as u32))
                .arg(&(cond_out as u32))
                .launch(super::ops::flat(frames * cond_out))?;
        }
        let map: Vec<u32> = (0..length)
            .map(|j| u32::try_from(j * frames / length).context("map index overflow"))
            .collect::<Result<_>>()?;
        let d_map = s.clone_htod(&map)?;
        unsafe {
            s.launch_builder(&self.nearest_rows)
                .arg(&ws.cond_conv.slice(0..frames * cond_out))
                .arg(&d_map)
                .arg(&mut ws.cond.slice_mut(0..length * cond_out))
                .arg(&(length as i64))
                .arg(&(cond_out as i64))
                .launch(super::ops::flat(length * cond_out))?;
        }
        Ok(())
    }

    /// Conv1d on the published folded [out, in*k] layout: zero-pad,
    /// im2col activations to [rows, in*k], one GEMM, bias.
    fn conv1d(&self, site: Conv1dSite<'_>) -> Result<()> {
        let Conv1dSite {
            w,
            x,
            rows,
            in_dim,
            out_dim,
            out,
            bias,
            dilation,
            padded,
            cols,
        } = site;
        let s = &self.stream;
        let kernel = 7usize;
        let pad = 3 * dilation;
        ensure!(w.len() == out_dim * in_dim * kernel, "conv1d weight extent");
        ensure!(
            padded.len() >= (rows + 2 * pad) * in_dim && cols.len() >= rows * in_dim * kernel,
            "conv1d scratch extent"
        );
        {
            let mut padded = padded.slice_mut(0..(rows + 2 * pad) * in_dim);
            unsafe {
                s.launch_builder(&self.pad_rows_n)
                    .arg(x)
                    .arg(&mut padded)
                    .arg(&(rows as i64))
                    .arg(&(pad as i64))
                    .arg(&(in_dim as i64))
                    .launch(super::ops::flat((rows + 2 * pad) * in_dim))?;
            }
        }
        {
            let mut cols = cols.slice_mut(0..rows * in_dim * kernel);
            unsafe {
                s.launch_builder(&self.im2col)
                    .arg(&padded.slice(0..(rows + 2 * pad) * in_dim))
                    .arg(&mut cols)
                    .arg(&(rows as i64))
                    .arg(&(in_dim as i32))
                    .arg(&(kernel as i32))
                    .arg(&(dilation as i32))
                    .launch(super::ops::flat(rows * in_dim * kernel))?;
            }
            self.linear_bytes(
                w,
                out_dim,
                in_dim * kernel,
                &cols.slice(0..rows * in_dim * kernel),
                rows,
                out,
            )?;
        }
        let count = rows * out_dim;
        let out_ptr = out.device_ptr(&self.stream).0;
        let bias_ptr = bias.device_ptr(&self.stream).0;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&out_ptr)
                .arg(&bias_ptr)
                .arg(&out_ptr)
                .arg(&(rows as u32))
                .arg(&(out_dim as u32))
                .launch(super::ops::flat(count))?;
        }
        Ok(())
    }

    /// k=3 conv on the published folded [out, in*3] layout (condition
    /// encoder): zero-pad one row each side, im2col, one GEMM.
    fn conv1d_k3(&self, site: K3Site<'_>) -> Result<()> {
        let K3Site {
            w: w_bytes,
            x,
            rows,
            in_dim,
            out_dim,
            out,
            padded,
            cols,
        } = site;
        let s = &self.stream;
        ensure!(
            w_bytes.len() == out_dim * in_dim * 3 * 4,
            "k3 weight extent"
        );
        let w_tensor = f32_matrix(w_bytes, s, out_dim, in_dim * 3)?;
        ensure!(
            padded.len() >= (rows + 2) * in_dim && cols.len() >= rows * in_dim * 3,
            "k3 scratch extent"
        );
        {
            let mut padded = padded.slice_mut(0..(rows + 2) * in_dim);
            unsafe {
                s.launch_builder(&self.pad_rows_n)
                    .arg(x)
                    .arg(&mut padded)
                    .arg(&(rows as i64))
                    .arg(&1i64)
                    .arg(&(in_dim as i64))
                    .launch(super::ops::flat((rows + 2) * in_dim))?;
            }
        }
        {
            let mut cols = cols.slice_mut(0..rows * in_dim * 3);
            unsafe {
                s.launch_builder(&self.im2col)
                    .arg(&padded.slice(0..(rows + 2) * in_dim))
                    .arg(&mut cols)
                    .arg(&(rows as i64))
                    .arg(&(in_dim as i32))
                    .arg(&3i32)
                    .arg(&1i32)
                    .launch(super::ops::flat(rows * in_dim * 3))?;
            }
        }
        let mut view = TensorMut(unsafe {
            Tensor::new(
                out.device_ptr_mut(s).0,
                [rows as u32, out_dim as u32],
                [out_dim as u64, 1],
                (rows * out_dim * 4) as u64,
            )?
        });
        let x_tensor = unsafe {
            Tensor::<f32, 2>::new(
                cols.device_ptr(s).0,
                [rows as u32, (in_dim * 3) as u32],
                [(in_dim * 3) as u64, 1],
                (rows * in_dim * 3 * 4) as u64,
            )?
        };
        self.blas.linear_f32(w_tensor, x_tensor, &mut view)
    }

    /// One vocoder stream over workspace scratch buffers; the residual
    /// units read the stage input through fresh snake buffers, never the
    /// snake outputs.
    fn vocoder_stream(&self, ws: &mut Workspace, half: usize, length: usize) -> Result<()> {
        let s = &self.stream;
        let streams = self.spec.vocoder_stream_channels() as usize;
        let input_dim = self.spec.decoder_input as usize;
        let hidden = self.spec.decoder_hidden as usize;
        let latent =
            ws.x.slice(half * streams * length..(half + 1) * streams * length);
        unsafe {
            s.launch_builder(&self.transpose)
                .arg(&latent)
                .arg(&mut ws.voc_x.slice_mut(0..length * streams))
                .arg(&(streams as i64))
                .arg(&(length as i64))
                .launch(super::ops::flat(streams * length))?;
        }
        self.linear(
            &self.weights.vocoder.dec_in_proj_w,
            input_dim,
            streams,
            &ws.voc_x.slice(0..length * streams),
            length,
            &mut ws.voc_dec.slice_mut(0..length * input_dim),
        )?;
        let dec_ptr = ws.voc_dec.device_ptr(s).0;
        let dec_bias = self.weights.vocoder.dec_in_proj_b.device_ptr(s).0;
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&dec_ptr)
                .arg(&dec_bias)
                .arg(&dec_ptr)
                .arg(&(length as u32))
                .arg(&(input_dim as u32))
                .launch(super::ops::flat(length * input_dim))?;
        }
        self.conv1d(Conv1dSite {
            w: &self.vocoder_folds.conv_in_w,
            x: &ws.voc_dec,
            rows: length,
            in_dim: input_dim,
            out_dim: hidden,
            out: &mut ws.voc_cur,
            bias: &self.weights.vocoder.conv_in_b,
            dilation: 1,
            padded: &mut ws.voc_pad,
            cols: &mut ws.voc_cols,
        })?;
        let mut rows = length;
        let mut dim = hidden;
        for (block_index, vb) in self.vocoder_folds.blocks.iter().enumerate() {
            let stride = self.spec.upsampling[block_index] as usize;
            let out_dim = dim / 2;
            let count = rows * dim;
            let cur_ptr = ws.voc_cur.device_ptr_mut(s).0;

            unsafe {
                s.launch_builder(&self.snake_rows)
                    .arg(&cur_ptr)
                    .arg(&self.weights.vocoder.blocks[block_index].snake1_alpha)
                    .arg(&cur_ptr)
                    .arg(&(count as i64))
                    .arg(&(dim as i32))
                    .launch(super::ops::flat(count))?;
            }
            unsafe {
                s.launch_builder(&self.conv_transpose)
                    .arg(&ws.voc_cur.slice(0..count))
                    .arg(&vb.conv_t_w)
                    .arg(&self.weights.vocoder.blocks[block_index].conv_t_b)
                    .arg(&mut ws.voc_next.slice_mut(0..rows * stride * out_dim))
                    .arg(&(rows as i32))
                    .arg(&(dim as i32))
                    .arg(&(out_dim as i32))
                    .arg(&(2 * stride as i32))
                    .arg(&(stride as i32))
                    .arg(&(stride.div_ceil(2) as i32))
                    .launch(super::ops::grid(rows * stride, 256))?;
            }
            std::mem::swap(&mut ws.voc_cur, &mut ws.voc_next);
            rows *= stride;
            dim = out_dim;
            for (unit_index, unit) in vb.units.iter().enumerate() {
                let dilation = [1usize, 3, 9][unit_index];
                let count = rows * dim;
                let unit_weights = &self.weights.vocoder.blocks[block_index].units[unit_index];
                let sn1_in = ws.voc_cur.device_ptr(s).0;
                let sn1_out = ws.voc_sn1.device_ptr_mut(s).0;
                unsafe {
                    s.launch_builder(&self.snake_rows)
                        .arg(&sn1_in)
                        .arg(&unit_weights.snake1_alpha)
                        .arg(&sn1_out)
                        .arg(&(count as i64))
                        .arg(&(dim as i32))
                        .launch(super::ops::flat(count))?;
                }
                self.conv1d(Conv1dSite {
                    w: &unit.c1_w,
                    x: &ws.voc_sn1,
                    rows,
                    in_dim: dim,
                    out_dim: dim,
                    out: &mut ws.voc_conv,
                    bias: &unit_weights.c1_b,
                    dilation,
                    padded: &mut ws.voc_pad,
                    cols: &mut ws.voc_cols,
                })?;
                let conv_ptr = ws.voc_conv.device_ptr(s).0;
                let sn2_out = ws.voc_sn2.device_ptr_mut(s).0;
                unsafe {
                    s.launch_builder(&self.snake_rows)
                        .arg(&conv_ptr)
                        .arg(&unit_weights.snake2_alpha)
                        .arg(&sn2_out)
                        .arg(&(count as i64))
                        .arg(&(dim as i32))
                        .launch(super::ops::flat(count))?;
                }
                self.linear_bytes(
                    &unit.c2_w,
                    dim,
                    dim,
                    &ws.voc_sn2.slice(0..count),
                    rows,
                    &mut ws.voc_gated,
                )?;
                let gated_ptr = ws.voc_gated.device_ptr(s).0;
                let gated_bias = unit_weights.c2_b.device_ptr(s).0;
                unsafe {
                    s.launch_builder(&self.bias)
                        .arg(&gated_ptr)
                        .arg(&gated_bias)
                        .arg(&gated_ptr)
                        .arg(&(rows as u32))
                        .arg(&(dim as u32))
                        .launch(super::ops::flat(count))?;
                    let cur_ptr = ws.voc_cur.device_ptr_mut(s).0;
                    s.launch_builder(&self.add_scale)
                        .arg(&cur_ptr)
                        .arg(&gated_ptr)
                        .arg(&(count as i64))
                        .arg(&1.0f32)
                        .launch(super::ops::flat(count))?;
                }
            }
        }
        let total = rows;
        let tail = dim;
        ensure!(ws.voc_out.len() >= total, "vocoder output extent");
        let count = total * tail;
        let cur_ptr = ws.voc_cur.device_ptr_mut(s).0;

        unsafe {
            s.launch_builder(&self.snake_rows)
                .arg(&cur_ptr)
                .arg(&self.weights.vocoder.snake_out_alpha)
                .arg(&cur_ptr)
                .arg(&(count as i64))
                .arg(&(tail as i32))
                .launch(super::ops::flat(count))?;
        }
        self.conv1d(Conv1dSite {
            w: &self.vocoder_folds.conv_out_w,
            x: &ws.voc_cur,
            rows: total,
            in_dim: tail,
            out_dim: 1,
            out: &mut ws.voc_out,
            bias: &self.weights.vocoder.conv_out_b,
            dilation: 1,
            padded: &mut ws.voc_pad,
            cols: &mut ws.voc_cols,
        })?;
        unsafe {
            s.launch_builder(&self.tanh_fn)
                .arg(&mut ws.voc_out.slice_mut(0..total))
                .arg(&(total as i64))
                .launch(super::ops::flat(total))?;
        }
        Ok(())
    }

    /// Decode both latent halves of ws.x into `out` [2, length * total].
    pub fn vocoder(
        &self,
        ws: &mut Workspace,
        length: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let streams = self.spec.vocoder_stream_channels() as usize;
        let total: usize = self.spec.upsampling.iter().map(|r| *r as usize).product();
        let channels = self.spec.in_channels as usize;
        ensure!(
            channels == 2 * streams && out.len() == 2 * length * total,
            "vocoder extent mismatch"
        );
        let s = &self.stream;
        for half in 0..2 {
            self.vocoder_stream(ws, half, length)?;
            s.memcpy_dtod(
                &ws.voc_out.slice(0..length * total),
                &mut out.slice_mut(half * length * total..(half + 1) * length * total),
            )?;
        }
        Ok(())
    }

    /// `linear` for a dense F32 weight already on device.
    fn linear_bytes(
        &self,
        weight: &CudaSlice<f32>,
        out_rows: usize,
        in_cols: usize,
        input: &CudaView<f32>,
        rows: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let mut view = TensorMut(unsafe {
            Tensor::new(
                out.device_ptr_mut(&self.stream).0,
                [rows as u32, out_rows as u32],
                [out_rows as u64, 1],
                (rows * out_rows * 4) as u64,
            )?
        });
        let w = unsafe {
            Tensor::<f32, 2>::new(
                weight.device_ptr(&self.stream).0,
                [out_rows as u32, in_cols as u32],
                [in_cols as u64, 1],
                (out_rows * in_cols * 4) as u64,
            )?
        };
        self.blas.linear_f32(
            w,
            unsafe {
                Tensor::new(
                    input.device_ptr(&self.stream).0,
                    [rows as u32, in_cols as u32],
                    [in_cols as u64, 1],
                    (rows * in_cols * 4) as u64,
                )?
            },
            &mut view,
        )
    }

    fn layernorm(
        &self,
        input: u64,
        weight: &CudaSlice<u8>,
        bias: &CudaSlice<u8>,
        rows: usize,
        width: usize,
        out: u64,
    ) -> Result<()> {
        let stride = width as u64;
        unsafe {
            self.stream
                .launch_builder(&self.layernorm)
                .arg(&input)
                .arg(weight)
                .arg(bias)
                .arg(&out)
                .arg(&(rows as u32))
                .arg(&1u32)
                .arg(&(width as u32))
                .arg(&stride)
                .arg(&0u64)
                .arg(&1u64)
                .arg(&stride)
                .arg(&0u64)
                .arg(&1u64)
                .arg(&1.0e-5f32)
                .launch(super::ops::grid(rows, 256))?;
        }
        Ok(())
    }
}

/// Folded vocoder weights on the published layouts: conv weights
/// [out, in*k], transposed conv weights [in, out*k].
pub struct VocoderFolds {
    pub conv_in_w: CudaSlice<f32>,
    pub blocks: Vec<VocoderBlockFold>,
    pub conv_out_w: CudaSlice<f32>,
}

pub struct VocoderBlockFold {
    pub conv_t_w: CudaSlice<f32>,
    pub units: Vec<VocoderUnitFold>,
}

pub struct VocoderUnitFold {
    pub c1_w: CudaSlice<f32>,
    pub c2_w: CudaSlice<f32>,
}

impl Music3Syn {
    pub fn workspace(
        spec: &crate::music3::SynSpec,
        d: &Device,
        max_length: usize,
    ) -> Result<Workspace> {
        build_workspace(spec, d, max_length)
    }
}

fn build_workspace(
    spec: &crate::music3::SynSpec,
    d: &Device,
    max_length: usize,
) -> Result<Workspace> {
    let s = &d.stream;
    let alloc = |n: usize| -> Result<CudaSlice<f32>> { Ok(unsafe { s.alloc(n.max(1))? }) };
    let dim = spec.flow_dim() as usize;
    let concat_width = spec.concat_channels() as usize;
    let channels = spec.in_channels as usize;
    let cond_dim = spec.condition_dim as usize;
    let cond_hidden = spec.cond_hidden as usize;
    let ff = spec.flow_ff_inner as usize;
    let tokens = max_length + 1;
    let pair = 2 * tokens;
    let rotary_dim = spec.rotary_dim as usize;
    let streams = spec.vocoder_stream_channels() as usize;
    let decoder_input = spec.decoder_input as usize;
    // frames that produce max_length latent rows (inverse of latent_length)
    let cond_frames = (max_length as u64 * spec.input_rate as u64 * spec.output_hop as u64)
        .div_ceil(spec.output_rate as u64 * spec.input_hop as u64) as usize;
    let decoder_hidden = spec.decoder_hidden as usize;
    let voc_total: usize = spec.upsampling.iter().map(|r| *r as usize).product();
    // Largest vocoder stage extent over all blocks and both endpoints,
    // and the conv1d scratch extents (kernel 7, dilations 1/3/9).
    let mut voc_stage = max_length * decoder_hidden.max(decoder_input);
    let mut voc_pad = 0usize;
    let mut voc_cols = 0usize;
    {
        let mut note = |rows: usize, in_dim: usize, dilation: usize| {
            voc_pad = voc_pad.max((rows + 6 * dilation) * in_dim);
            voc_cols = voc_cols.max(rows * in_dim * 7);
        };
        note(max_length, decoder_input, 1);
        note(
            max_length * voc_total,
            1.max(spec.decoder_hidden as usize >> spec.upsampling.len()),
            1,
        );
        let (mut rows, mut dim) = (max_length, decoder_hidden);
        for r in spec.upsampling.iter() {
            rows *= *r as usize;
            dim /= 2;
            voc_stage = voc_stage.max(rows * dim);
            for dilation in [1usize, 3, 9] {
                note(rows, dim, dilation);
            }
        }
    }
    let half = rotary_dim / 2;
    let mut cos = Vec::with_capacity(tokens * rotary_dim);
    let mut sin = Vec::with_capacity(tokens * rotary_dim);
    for pos in 0..tokens {
        let freqs: Vec<f32> = (0..half)
            .map(|i| pos as f32 / 10000.0f32.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        for _ in 0..2 {
            for f in &freqs {
                cos.push(f.cos());
                sin.push(f.sin());
            }
        }
    }
    Ok(Workspace {
        hidden: alloc(2 * max_length * concat_width)?,
        pre: alloc(2 * max_length * concat_width)?,
        wide: alloc(pair * dim)?,
        norm: alloc(pair * dim)?,
        q: alloc(pair * dim)?,
        k: alloc(pair * dim)?,
        v: alloc(pair * dim)?,
        attn: alloc(pair * dim)?,
        scores: alloc(pair * tokens)?,
        ff_buf: alloc(pair * 2 * ff)?,
        ff_gated: alloc(pair * ff)?,
        ff_out: alloc(pair * dim)?,
        proj: alloc(2 * max_length * channels)?,
        post: alloc(2 * max_length * channels)?,
        fourier: alloc(spec.fourier_dim as usize)?,
        t1: alloc(dim)?,
        temb: alloc(dim)?,
        x: alloc(channels * max_length)?,
        pair_v_c: alloc(channels * max_length)?,
        pair_v_u: alloc(channels * max_length)?,
        cond: alloc(max_length * cond_dim)?,
        zeros_cond: s.alloc_zeros(max_length * cond_dim)?,
        cond_fused: alloc(cond_frames * cond_hidden)?,
        cond_conv: alloc(cond_frames * cond_dim)?,
        cond_pad: alloc((cond_frames + 2) * cond_hidden)?,
        cond_cols: alloc(cond_frames * cond_hidden * 3)?,
        rotary_cos: s.clone_htod(&cos)?,
        rotary_sin: s.clone_htod(&sin)?,
        voc_x: alloc(max_length * streams)?,
        voc_dec: alloc(max_length * decoder_input)?,
        voc_cur: alloc(voc_stage)?,
        voc_next: alloc(voc_stage)?,
        voc_sn1: alloc(voc_stage)?,
        voc_conv: alloc(voc_stage)?,
        voc_sn2: alloc(voc_stage)?,
        voc_gated: alloc(voc_stage)?,
        voc_pad: alloc(voc_pad)?,
        voc_cols: alloc(voc_cols)?,
        voc_out: alloc(max_length * voc_total)?,
    })
}

fn fold_weight_norm(
    s: &Arc<cudarc::driver::CudaStream>,
    fold: &CudaFunction,
    g: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    channels: usize,
    rest: usize,
) -> Result<CudaSlice<f32>> {
    // g is [C, 1, 1] bytes: C contiguous floats, read by the kernel directly.
    let mut folded: CudaSlice<f32> = unsafe { s.alloc(channels * rest)? };
    unsafe {
        s.launch_builder(fold)
            .arg(g)
            .arg(v)
            .arg(&mut folded)
            .arg(&(channels as i64))
            .arg(&(rest as i64))
            .launch(super::ops::grid(channels, 256))?;
    }
    Ok(folded)
}
