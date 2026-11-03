use super::Device;
use super::contracts::{Tensor, TensorMut};
use super::weights::Music3SynWeights;
use crate::backend::gpu::{CudaFunction, DevicePtr, DevicePtrMut, PushKernelArg};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use std::sync::Arc;

/// Native MiniMax-Music3 synthesis engine: condition encoder, F32 flow
/// transformer, FlowMatchEuler scheduler and vocoder, fed the condition
/// stream produced by the autoregressive stage. Linears run on the shared
/// `Blas::linear_f32` (CUBLAS_COMPUTE_32F, no TF32); attention uses a
/// dedicated handle pinned to CUBLAS_DEFAULT_MATH.
pub struct Music3Syn {
    pub spec: crate::model_spec::Music3SynSpec,
    pub weights: Music3SynWeights,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub blas: super::blas::Blas,
    attention: CudaBlas,
    concat: CudaFunction,
    tail: CudaFunction,
    layernorm: CudaFunction,
    fourier: CudaFunction,
    softmax: CudaFunction,
    rotary: CudaFunction,
    silu: CudaFunction,
    silu_mul: CudaFunction,
    bias_add: CudaFunction,
    add_inplace: CudaFunction,
    cond_sum: CudaFunction,
    pad_rows: CudaFunction,
    nearest_rows: CudaFunction,
    cond_weights: CudaSlice<f32>,
    cond_scale: f32,
    proj_planes: [CudaSlice<f32>; 3],
    vocoder: VocoderSetup,
    pad_rows_n: CudaFunction,
    snake_rows: CudaFunction,
    scatter_add: CudaFunction,
    tanh_fn: CudaFunction,
    transpose_fn: CudaFunction,
    euler_fn: CudaFunction,
    #[cfg(test)]
    pub(crate) vocoder_taps: std::cell::RefCell<Vec<(&'static str, Vec<f32>)>>,
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
    pub fn load(dir: &std::path::Path, d: &Device) -> Result<Self> {
        let spec = crate::model_spec::Music3SynSpec::read(dir)?;
        let weights = Music3SynWeights::load(dir, &spec, d)?;
        let ops = super::ops::Ops::new(&d.ctx)?;
        let closure = std::rc::Rc::new(super::closure::Closure::new(
            false,
            "Music3Syn setup".into(),
        ));
        let blas = super::blas::Blas::new(d, &ops, std::path::Path::new(""), closure)?;
        let attention = CudaBlas::new(d.stream.clone())?;
        let mode = cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(*attention.handle(), mode)
                .result()
                .context("cublasSetMathMode failed")?;
            let mut readback = cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
            cudarc::cublas::sys::cublasGetMathMode(*attention.handle(), &mut readback)
                .result()
                .context("cublasGetMathMode failed")?;
            ensure!(
                readback == mode,
                "attention GEMM math mode is not CUBLAS_DEFAULT_MATH"
            );
        }
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
        let cond_out = spec.cond_out as usize;
        let cond_hidden = spec.cond_hidden as usize;
        let mut proj_planes = Vec::with_capacity(3);
        for k in 0..3 {
            let mut plane: CudaSlice<f32> = unsafe { d.stream.alloc(cond_out * cond_hidden)? };
            unsafe {
                d.stream
                    .launch_builder(&kernel(&common, "extract_plane_f32")?)
                    .arg(&weights.cond.proj_w)
                    .arg(&k)
                    .arg(&mut plane)
                    .arg(&((cond_out * cond_hidden) as i64))
                    .arg(&(cond_hidden as i32))
                    .arg(&3i32)
                    .launch(super::ops::flat(cond_out * cond_hidden))?;
            }
            proj_planes.push(plane);
        }
        let [p0, p1, p2] = proj_planes.try_into().unwrap();
        let wn_fold = kernel(&common, "weight_norm_fold_f32")?;
        let extract = kernel(&common, "extract_plane_f32")?;
        let extract_t = kernel(&common, "extract_plane_transposed_f32")?;
        let s = &d.stream;
        let plane = |w: &CudaSlice<f32>,
                     k: usize,
                     rows: usize,
                     cols: usize,
                     kernel: i32|
         -> Result<CudaSlice<f32>> {
            let mut out: CudaSlice<f32> = unsafe { s.alloc(rows * cols)? };
            unsafe {
                s.launch_builder(&extract)
                    .arg(w)
                    .arg(&(k as i32))
                    .arg(&mut out)
                    .arg(&((rows * cols) as i64))
                    .arg(&(cols as i32))
                    .arg(&kernel)
                    .launch(super::ops::flat(rows * cols))?;
            }
            Ok(out)
        };
        let _ = &extract_t;
        let vc = &weights.vocoder;
        let hidden = spec.decoder_hidden as usize;
        let input_dim = spec.decoder_input as usize;
        let conv_in_folded = fold_weight_norm(
            s,
            &wn_fold,
            &vc.conv_in_g,
            &vc.conv_in_v,
            hidden,
            input_dim * 7,
        )?;
        let conv_in_planes = (0..7)
            .map(|k| plane(&conv_in_folded, k, hidden, input_dim, 7))
            .collect::<Result<Vec<_>>>()?;
        let mut blocks = Vec::with_capacity(vc.blocks.len());
        let mut stage_in = hidden;
        for (vb, ratio) in vc.blocks.iter().zip(&spec.upsampling) {
            let stage_out = stage_in / 2;
            let stride = *ratio as usize;
            let conv_t_folded = fold_weight_norm(
                s,
                &wn_fold,
                &vb.conv_t_g,
                &vb.conv_t_v,
                stage_in,
                stage_out * 2 * stride,
            )?;
            let conv_t_planes = (0..2 * stride)
                .map(|k| {
                    let mut out: CudaSlice<f32> = unsafe { s.alloc(stage_out * stage_in)? };
                    unsafe {
                        s.launch_builder(&extract_t)
                            .arg(&conv_t_folded)
                            .arg(&(k as i32))
                            .arg(&mut out)
                            .arg(&((stage_out * stage_in) as i64))
                            .arg(&(stage_in as i32))
                            .arg(&(stage_out as i32))
                            .arg(&((2 * stride) as i32))
                            .launch(super::ops::flat(stage_out * stage_in))?;
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut units = Vec::with_capacity(vb.units.len());
            for u in &vb.units {
                let c1_folded =
                    fold_weight_norm(s, &wn_fold, &u.c1_g, &u.c1_v, stage_out, stage_out * 7)?;
                let c1_planes = (0..7)
                    .map(|k| plane(&c1_folded, k, stage_out, stage_out, 7))
                    .collect::<Result<Vec<_>>>()?;
                let c2_w = fold_weight_norm(s, &wn_fold, &u.c2_g, &u.c2_v, stage_out, stage_out)?;
                units.push(VocoderUnitSetup { c1_planes, c2_w });
            }
            blocks.push(VocoderBlockSetup {
                conv_t_planes,
                units,
            });
            stage_in = stage_out;
        }
        let tail_dim = stage_in;
        let conv_out_folded =
            fold_weight_norm(s, &wn_fold, &vc.conv_out_g, &vc.conv_out_v, 1, tail_dim * 7)?;
        let conv_out_planes = (0..7)
            .map(|k| plane(&conv_out_folded, k, 1, tail_dim, 7))
            .collect::<Result<Vec<_>>>()?;
        #[cfg(test)]
        #[cfg(test)]
        let fold_tap = s.clone_dtoh(&conv_in_planes[0])?;
        let vocoder = VocoderSetup {
            conv_in_planes,
            blocks,
            conv_out_planes,
        };
        let built = Self {
            spec,
            weights,
            stream: d.stream.clone(),
            blas,
            attention,
            concat: kernel(&common, "flow_concat_f32")?,
            tail: kernel(&common, "flow_tail_f32")?,
            layernorm: kernel(&common, "q38_layer_norm_f32")?,
            fourier: kernel(&common, "fourier_time_f32")?,
            softmax: kernel(&common, "softmax_rows_f32")?,
            rotary: kernel(&rope, "partial_rotary_f32")?,
            silu: kernel(&act, "silu_f32")?,
            silu_mul: kernel(&act, "silu_mul_f32")?,
            bias_add: kernel(&common, "bias_add_f32")?,
            add_inplace: kernel(&common, "add_inplace_f32")?,
            cond_sum: kernel(&common, "cond_layer_sum_f32")?,
            pad_rows: kernel(&common, "pad_rows_f32")?,
            nearest_rows: kernel(&common, "nearest_rows_f32")?,
            cond_weights,
            cond_scale,
            proj_planes: [p0, p1, p2],
            vocoder,
            pad_rows_n: kernel(&common, "pad_rows_n_f32")?,
            snake_rows: kernel(&common, "snake_rows_f32")?,
            scatter_add: kernel(&common, "scatter_strided_add_f32")?,
            tanh_fn: kernel(&common, "tanh_f32")?,
            transpose_fn: kernel(&common, "transpose_f32")?,
            euler_fn: kernel(&common, "euler_update_f32")?,
            #[cfg(test)]
            vocoder_taps: std::cell::RefCell::new(Vec::new()),
        };
        #[cfg(test)]
        built.vocoder_taps.borrow_mut().push(("fold_p0", fold_tap));
        #[allow(clippy::needless_return)]
        Ok(built)
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
    #[allow(clippy::too_many_arguments)]
    pub fn flow_velocity(
        &self,
        x_t: &CudaSlice<f32>,
        t: f32,
        cond: &CudaSlice<f32>,
        rotary: (&CudaSlice<f32>, &CudaSlice<f32>),
        length: usize,
        velocity: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.flow_velocity_probed(x_t, t, cond, rotary, length, velocity, None)
    }

    /// Stage taps for the numerics gates: (name, buffer contents in the
    /// buffer's own layout).
    #[allow(clippy::too_many_arguments)]
    pub fn flow_velocity_probed(
        &self,
        x_t: &CudaSlice<f32>,
        t: f32,
        cond: &CudaSlice<f32>,
        rotary: (&CudaSlice<f32>, &CudaSlice<f32>),
        length: usize,
        velocity: &mut CudaSlice<f32>,
        mut probe: Option<&mut Vec<(&'static str, Vec<f32>)>>,
    ) -> Result<()> {
        let mut tap = |name: &'static str, buffer: &CudaSlice<f32>| -> Result<()> {
            if let Some(probe) = probe.as_deref_mut() {
                probe.push((name, self.stream.clone_dtoh(buffer)?));
            }
            Ok(())
        };
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
                && rotary.0.len() == tokens * rotary_dim
                && rotary.1.len() == tokens * rotary_dim,
            "flow_velocity extent mismatch"
        );

        // Channel concat, preprocess conv, residual.
        let mut hidden: CudaSlice<f32> = unsafe { s.alloc(concat_width * length)? };
        let mut pre: CudaSlice<f32> = unsafe { s.alloc(concat_width * length)? };
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
        tap("preprocess", &pre)?;
        unsafe {
            s.launch_builder(&self.add_inplace)
                .arg(&pre)
                .arg(&hidden)
                .arg(&((concat_width * length) as i64))
                .launch(super::ops::flat(concat_width * length))?;
        }

        // proj_in writes rows 1..=length of the token buffer; row 0 is temb.
        let mut wide: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        self.linear(
            &self.weights.flow.proj_in,
            dim,
            concat_width,
            &pre.slice(0..concat_width * length),
            length,
            &mut wide.slice_mut(dim..dim + length * dim),
        )?;
        let mut temb: CudaSlice<f32> = unsafe { s.alloc(dim)? };
        let mut fourier: CudaSlice<f32> = unsafe { s.alloc(fourier_dim)? };
        let t_dev = s.clone_htod(&[t])?;
        unsafe {
            s.launch_builder(&self.fourier)
                .arg(&t_dev)
                .arg(&self.weights.flow.time_proj)
                .arg(&mut fourier)
                .arg(&((fourier_dim / 2) as u32))
                .launch(super::ops::flat(fourier_dim / 2))?;
        }
        tap("fourier", &fourier)?;
        let mut t1: CudaSlice<f32> = unsafe { s.alloc(dim)? };
        self.linear(
            &self.weights.flow.time_l1_w,
            dim,
            fourier_dim,
            &fourier.slice(0..fourier_dim),
            1,
            &mut t1.slice_mut(0..dim),
        )?;
        unsafe {
            s.launch_builder(&self.bias_add)
                .arg(&t1)
                .arg(&self.weights.flow.time_l1_b)
                .arg(&(dim as i64))
                .arg(&(dim as i32))
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
            s.launch_builder(&self.bias_add)
                .arg(&temb)
                .arg(&self.weights.flow.time_l2_b)
                .arg(&(dim as i64))
                .arg(&(dim as i32))
                .launch(super::ops::flat(dim))?;
        }
        s.memcpy_dtod(&temb, &mut wide)?;
        tap("temb", &temb)?;
        tap("wide_after_proj_in", &wide)?;

        let mut norm: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        let mut q: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        let mut k: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        let mut v: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        let mut attn: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        let mut scores: CudaSlice<f32> = unsafe { s.alloc(tokens * tokens)? };
        let mut ff_buf: CudaSlice<f32> = unsafe { s.alloc(2 * ff * tokens)? };
        let mut ff_gated: CudaSlice<f32> = unsafe { s.alloc(ff * tokens)? };
        let mut ff_out: CudaSlice<f32> = unsafe { s.alloc(dim * tokens)? };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        for (block_index, block) in self.weights.flow.blocks.iter().enumerate() {
            self.layernorm(
                &wide,
                &block.norm1_w,
                &block.norm1_b,
                tokens,
                dim,
                &mut norm,
            )?;
            if block_index == 0 {
                tap("block0_norm1", &norm)?;
            }
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
            for buffer in [&q, &k] {
                unsafe {
                    s.launch_builder(&self.rotary)
                        .arg(buffer)
                        .arg(rotary.0)
                        .arg(rotary.1)
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
                    gemm_f32(
                        &self.attention,
                        &GemmF32 {
                            transa: op_t,
                            transb: op_n,
                            m: tokens as i32,
                            n: tokens as i32,
                            k: head_dim as i32,
                            alpha: scale,
                            lda: dim as i32,
                            ldb: dim as i32,
                            ldc: tokens as i32,
                        },
                        k_ptr + off,
                        q_ptr + off,
                        scores_ptr,
                    )?;
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
                    gemm_f32(
                        &self.attention,
                        &GemmF32 {
                            transa: op_n,
                            transb: op_n,
                            m: head_dim as i32,
                            n: tokens as i32,
                            k: tokens as i32,
                            alpha: 1.0,
                            lda: dim as i32,
                            ldb: tokens as i32,
                            ldc: dim as i32,
                        },
                        v_ptr + off,
                        scores_ptr,
                        attn_ptr + off,
                    )?;
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
            if block_index == 0 {
                tap("block0_attn", &ff_out)?;
            }
            unsafe {
                s.launch_builder(&self.add_inplace)
                    .arg(&wide)
                    .arg(&ff_out)
                    .arg(&((dim * tokens) as i64))
                    .launch(super::ops::flat(dim * tokens))?;
            }

            self.layernorm(
                &wide,
                &block.norm2_w,
                &block.norm2_b,
                tokens,
                dim,
                &mut norm,
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
                s.launch_builder(&self.bias_add)
                    .arg(&ff_buf)
                    .arg(&block.ff_in_b)
                    .arg(&((2 * ff * tokens) as i64))
                    .arg(&((2 * ff) as i32))
                    .launch(super::ops::flat(2 * ff * tokens))?;
                s.launch_builder(&self.silu_mul)
                    .arg(&ff_buf)
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
                s.launch_builder(&self.bias_add)
                    .arg(&ff_out)
                    .arg(&block.ff_out_b)
                    .arg(&((dim * tokens) as i64))
                    .arg(&(dim as i32))
                    .launch(super::ops::flat(dim * tokens))?;
                s.launch_builder(&self.add_inplace)
                    .arg(&wide)
                    .arg(&ff_out)
                    .arg(&((dim * tokens) as i64))
                    .launch(super::ops::flat(dim * tokens))?;
            }
            if block_index == 0 {
                tap("block0_out", &wide)?;
            }
        }

        tap("block_last_out", &wide)?;
        // proj_out over rows 1.., postprocess conv, transpose + residual.
        let mut proj: CudaSlice<f32> = unsafe { s.alloc(channels * length)? };
        self.linear(
            &self.weights.flow.proj_out,
            channels,
            dim,
            &wide.slice(dim..dim + length * dim),
            length,
            &mut proj.slice_mut(0..channels * length),
        )?;
        tap("proj_out", &proj)?;
        let mut post: CudaSlice<f32> = unsafe { s.alloc(channels * length)? };
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
                .arg(&post)
                .arg(&proj)
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

    /// One Euler update: x += (sigma_next - sigma) * velocity.
    pub fn euler_step(
        &self,
        x: &mut CudaSlice<f32>,
        velocity: &CudaSlice<f32>,
        sigma: f32,
        sigma_next: f32,
    ) -> Result<()> {
        ensure!(x.len() == velocity.len(), "euler extent mismatch");
        let count = x.len();
        unsafe {
            self.stream
                .launch_builder(&self.euler_fn)
                .arg(&mut *x)
                .arg(velocity)
                .arg(&(sigma_next - sigma))
                .arg(&(count as i64))
                .launch(super::ops::flat(count))?;
        }
        Ok(())
    }

    /// Latent length for a frame count: int(frames * out_rate * in_hop /
    /// (in_rate * out_hop)), at least 1.
    pub fn latent_length(&self, frames: usize) -> usize {
        let numerator = self.spec.output_rate as u64 * self.spec.input_hop as u64 * frames as u64;
        let denominator = self.spec.input_rate as u64 * self.spec.output_hop as u64;
        usize::max(1, (numerator / denominator) as usize)
    }

    /// Condition encoder: fused layer softmax-sum, k=3 conv (padded, three
    /// shifted GEMMs over repacked planes) and nearest upsample. `input` is
    /// [frames, layers*cond_hidden], `out` [latent_length, cond_out].
    pub fn condition_encode(
        &self,
        input: &CudaSlice<f32>,
        frames: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let s = &self.stream;
        let cond_hidden = self.spec.cond_hidden as usize;
        let cond_out = self.spec.cond_out as usize;
        let layers = self.spec.cond_layers as usize;
        let length = self.latent_length(frames);
        ensure!(
            input.len() == frames * layers * cond_hidden && out.len() == length * cond_out,
            "condition_encode extent mismatch"
        );
        let mut fused: CudaSlice<f32> = unsafe { s.alloc(frames * cond_hidden)? };
        unsafe {
            s.launch_builder(&self.cond_sum)
                .arg(input)
                .arg(&self.cond_weights)
                .arg(&self.cond_scale)
                .arg(&mut fused)
                .arg(&(frames as i64))
                .arg(&(layers as i32))
                .arg(&(cond_hidden as i32))
                .launch(super::ops::flat(frames * cond_hidden))?;
        }
        let mut padded: CudaSlice<f32> = unsafe { s.alloc((frames + 2) * cond_hidden)? };
        unsafe {
            s.launch_builder(&self.pad_rows)
                .arg(&fused)
                .arg(&mut padded)
                .arg(&(frames as i64))
                .arg(&(cond_hidden as i64))
                .launch(super::ops::flat((frames + 2) * cond_hidden))?;
        }
        let mut conv: CudaSlice<f32> = unsafe { s.alloc(frames * cond_out)? };
        let mut shifted: CudaSlice<f32> = unsafe { s.alloc(frames * cond_out)? };
        for (k, plane) in self.proj_planes.iter().enumerate() {
            let window = padded.slice(k * cond_hidden..k * cond_hidden + frames * cond_hidden);
            if k == 0 {
                self.linear_bytes(plane, cond_out, cond_hidden, &window, frames, &mut conv)?;
            } else {
                self.linear_bytes(plane, cond_out, cond_hidden, &window, frames, &mut shifted)?;
                unsafe {
                    s.launch_builder(&self.add_inplace)
                        .arg(&conv)
                        .arg(&shifted)
                        .arg(&((frames * cond_out) as i64))
                        .launch(super::ops::flat(frames * cond_out))?;
                }
            }
        }
        unsafe {
            s.launch_builder(&self.bias_add)
                .arg(&conv)
                .arg(&self.weights.cond.proj_b)
                .arg(&((frames * cond_out) as i64))
                .arg(&(cond_out as i32))
                .launch(super::ops::flat(frames * cond_out))?;
        }
        let map: Vec<u32> = (0..length)
            .map(|j| u32::try_from(j * frames / length).context("map index overflow"))
            .collect::<Result<_>>()?;
        let d_map = s.clone_htod(&map)?;
        unsafe {
            s.launch_builder(&self.nearest_rows)
                .arg(&conv)
                .arg(&d_map)
                .arg(out)
                .arg(&(length as i64))
                .arg(&(cond_out as i64))
                .launch(super::ops::flat(length * cond_out))?;
        }
        Ok(())
    }

    /// Shifted-view conv over a zero-padded buffer: out rows `rows`, one
    /// GEMM per kernel tap with dilated offsets, bias at the end.
    #[allow(clippy::too_many_arguments)]
    fn conv_taps(
        &self,
        planes: &[CudaSlice<f32>],
        padded: &CudaSlice<f32>,
        in_dim: usize,
        rows: usize,
        dilation: usize,
        out: &mut CudaSlice<f32>,
        bias: &CudaSlice<u8>,
    ) -> Result<()> {
        let s = &self.stream;
        let out_dim = out.len() / rows;
        let mut scratch: CudaSlice<f32> = unsafe { s.alloc(rows * out_dim)? };
        for (k, plane) in planes.iter().enumerate() {
            let offset = k * dilation * in_dim;
            let window = padded.slice(offset..offset + rows * in_dim);
            if k == 0 {
                self.linear_bytes(plane, out_dim, in_dim, &window, rows, &mut *out)?;
            } else {
                self.linear_bytes(plane, out_dim, in_dim, &window, rows, &mut scratch)?;
                unsafe {
                    s.launch_builder(&self.add_inplace)
                        .arg(&mut *out)
                        .arg(&scratch)
                        .arg(&((rows * out_dim) as i64))
                        .launch(super::ops::flat(rows * out_dim))?;
                }
            }
        }
        unsafe {
            let count = out.len();
            s.launch_builder(&self.bias_add)
                .arg(&mut *out)
                .arg(bias)
                .arg(&(count as i64))
                .arg(&((count / rows) as i32))
                .launch(super::ops::flat(count))?;
        }
        Ok(())
    }

    /// One folded stereo stream: latent half [channels/2, length]
    /// channel-major -> waveform [total_samples].
    fn vocoder_stream(
        &self,
        latent: &cudarc::driver::CudaView<f32>,
        length: usize,
        waveform: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let s = &self.stream;
        let streams = self.spec.vocoder_stream_channels() as usize;
        let input_dim = self.spec.decoder_input as usize;
        let hidden = self.spec.decoder_hidden as usize;
        ensure!(latent.len() == streams * length, "vocoder stream extent");
        let mut x: CudaSlice<f32> = unsafe { s.alloc(length * streams)? };
        unsafe {
            s.launch_builder(&self.transpose_fn)
                .arg(latent)
                .arg(&mut x)
                .arg(&(streams as i64))
                .arg(&(length as i64))
                .launch(super::ops::flat(streams * length))?;
        }
        let mut buf: CudaSlice<f32> = unsafe { s.alloc(length * input_dim)? };
        self.linear(
            &self.weights.vocoder.dec_in_proj_w,
            input_dim,
            streams,
            &x.slice(0..length * streams),
            length,
            &mut buf.slice_mut(0..length * input_dim),
        )?;
        unsafe {
            s.launch_builder(&self.bias_add)
                .arg(&buf)
                .arg(&self.weights.vocoder.dec_in_proj_b)
                .arg(&((length * input_dim) as i64))
                .arg(&(input_dim as i32))
                .launch(super::ops::flat(length * input_dim))?;
        }
        #[cfg(test)]
        self.vocoder_taps
            .borrow_mut()
            .push(("dec_in", s.clone_dtoh(&buf)?));
        // conv_in k=7 pad=3.
        let mut padded: CudaSlice<f32> = unsafe { s.alloc((length + 6) * hidden)? };
        let mut wide: CudaSlice<f32> = unsafe { s.alloc(length * hidden)? };

        unsafe {
            s.launch_builder(&self.pad_rows_n)
                .arg(&buf.slice(0..length * input_dim))
                .arg(&mut padded)
                .arg(&(length as i64))
                .arg(&3i64)
                .arg(&(input_dim as i64))
                .launch(super::ops::flat((length + 6) * input_dim))?;
        }
        let mut wide_buf: CudaSlice<f32> = unsafe { s.alloc(length * hidden)? };
        self.conv_taps(
            &self.vocoder.conv_in_planes,
            &padded,
            input_dim,
            length,
            1,
            &mut wide_buf,
            &self.weights.vocoder.conv_in_b,
        )?;
        std::mem::swap(&mut wide, &mut wide_buf);
        #[cfg(test)]
        self.vocoder_taps
            .borrow_mut()
            .push(("conv_in", s.clone_dtoh(&wide)?));
        let mut rows = length;
        let mut dim = hidden;
        for (block_index, vb) in self.vocoder.blocks.iter().enumerate() {
            let stride = self.spec.upsampling[block_index] as usize;
            let out_dim = dim / 2;
            let pad = stride.div_ceil(2);
            unsafe {
                s.launch_builder(&self.snake_rows)
                    .arg(&wide)
                    .arg(&self.weights.vocoder.blocks[block_index].snake1_alpha)
                    .arg(&wide)
                    .arg(&((rows * dim) as i64))
                    .arg(&(dim as i32))
                    .launch(super::ops::flat(rows * dim))?;
            }
            let mut up: CudaSlice<f32> = s.alloc_zeros(rows * stride * out_dim)?;
            let mut gemm_out: CudaSlice<f32> = unsafe { s.alloc(rows * out_dim)? };
            for (k, plane) in vb.conv_t_planes.iter().enumerate() {
                self.linear_bytes(
                    plane,
                    out_dim,
                    dim,
                    &wide.slice(0..rows * dim),
                    rows,
                    &mut gemm_out,
                )?;
                unsafe {
                    s.launch_builder(&self.scatter_add)
                        .arg(&gemm_out)
                        .arg(&mut up)
                        .arg(&(rows as i64))
                        .arg(&(out_dim as i32))
                        .arg(&(stride as i32))
                        .arg(&(k as i32))
                        .arg(&(pad as i32))
                        .launch(super::ops::flat(rows * out_dim))?;
                }
            }
            unsafe {
                s.launch_builder(&self.bias_add)
                    .arg(&up)
                    .arg(&self.weights.vocoder.blocks[block_index].conv_t_b)
                    .arg(&((rows * stride * out_dim) as i64))
                    .arg(&(out_dim as i32))
                    .launch(super::ops::flat(rows * stride * out_dim))?;
            }
            rows *= stride;
            dim = out_dim;
            wide = up;
            #[cfg(test)]
            if block_index == 0 {
                self.vocoder_taps
                    .borrow_mut()
                    .push(("convt0", s.clone_dtoh(&wide)?));
            }
            #[cfg(test)]
            self.vocoder_taps
                .borrow_mut()
                .push(("block", s.clone_dtoh(&wide)?));
            for (unit_index, unit) in vb.units.iter().enumerate() {
                let dilation = [1usize, 3, 9][unit_index];
                let mut snaked: CudaSlice<f32> = unsafe { s.alloc(rows * dim)? };
                unsafe {
                    s.launch_builder(&self.snake_rows)
                        .arg(&wide)
                        .arg(
                            &self.weights.vocoder.blocks[block_index].units[unit_index]
                                .snake1_alpha,
                        )
                        .arg(&mut snaked)
                        .arg(&((rows * dim) as i64))
                        .arg(&(dim as i32))
                        .launch(super::ops::flat(rows * dim))?;
                }
                let pad_rows = 3 * dilation;
                let mut padded: CudaSlice<f32> = unsafe { s.alloc((rows + 2 * pad_rows) * dim)? };
                unsafe {
                    s.launch_builder(&self.pad_rows_n)
                        .arg(&snaked)
                        .arg(&mut padded)
                        .arg(&(rows as i64))
                        .arg(&(pad_rows as i64))
                        .arg(&(dim as i64))
                        .launch(super::ops::flat((rows + 2 * pad_rows) * dim))?;
                }
                let mut conv: CudaSlice<f32> = unsafe { s.alloc(rows * dim)? };
                self.conv_taps(
                    &unit.c1_planes,
                    &padded,
                    dim,
                    rows,
                    dilation,
                    &mut conv,
                    &self.weights.vocoder.blocks[block_index].units[unit_index].c1_b,
                )?;
                unsafe {
                    s.launch_builder(&self.snake_rows)
                        .arg(&conv)
                        .arg(
                            &self.weights.vocoder.blocks[block_index].units[unit_index]
                                .snake2_alpha,
                        )
                        .arg(&conv)
                        .arg(&((rows * dim) as i64))
                        .arg(&(dim as i32))
                        .launch(super::ops::flat(rows * dim))?;
                }
                let mut gated: CudaSlice<f32> = unsafe { s.alloc(rows * dim)? };
                self.linear_bytes(
                    &unit.c2_w,
                    dim,
                    dim,
                    &conv.slice(0..rows * dim),
                    rows,
                    &mut gated,
                )?;
                unsafe {
                    s.launch_builder(&self.bias_add)
                        .arg(&gated)
                        .arg(&self.weights.vocoder.blocks[block_index].units[unit_index].c2_b)
                        .arg(&((rows * dim) as i64))
                        .arg(&(dim as i32))
                        .launch(super::ops::flat(rows * dim))?;
                    s.launch_builder(&self.add_inplace)
                        .arg(&wide)
                        .arg(&gated)
                        .arg(&((rows * dim) as i64))
                        .launch(super::ops::flat(rows * dim))?;
                    #[cfg(test)]
                    if block_index == 0 && unit_index == 0 {
                        self.vocoder_taps
                            .borrow_mut()
                            .push(("u1_0", s.clone_dtoh(&wide)?));
                    }
                }
            }
        }
        let total = rows;
        let tail = dim;
        ensure!(waveform.len() == total, "waveform extent");
        unsafe {
            s.launch_builder(&self.snake_rows)
                .arg(&wide)
                .arg(&self.weights.vocoder.snake_out_alpha)
                .arg(&wide)
                .arg(&((total * tail) as i64))
                .arg(&(tail as i32))
                .launch(super::ops::flat(total * tail))?;
        }
        let pad_rows = 3;
        let mut padded: CudaSlice<f32> = unsafe { s.alloc((total + 2 * pad_rows) * tail)? };
        unsafe {
            s.launch_builder(&self.pad_rows_n)
                .arg(&wide)
                .arg(&mut padded)
                .arg(&(total as i64))
                .arg(&(pad_rows as i64))
                .arg(&(tail as i64))
                .launch(super::ops::flat((total + 2 * pad_rows) * tail))?;
        }
        let mut conv: CudaSlice<f32> = unsafe { s.alloc(total)? };
        self.conv_taps(
            &self.vocoder.conv_out_planes,
            &padded,
            tail,
            total,
            1,
            &mut conv,
            &self.weights.vocoder.conv_out_b,
        )?;
        unsafe {
            s.launch_builder(&self.tanh_fn)
                .arg(&mut conv)
                .arg(&(total as i64))
                .launch(super::ops::flat(total))?;
        }
        s.memcpy_dtod(&conv, waveform)?;
        Ok(())
    }

    /// Full vocoder: latents [channels, length] channel-major -> stereo
    /// [2, total] (channel-major rows per stream, concatenated).
    pub fn vocoder(
        &self,
        latents: &CudaSlice<f32>,
        length: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let streams = self.spec.vocoder_stream_channels() as usize;
        let total: usize = self.spec.upsampling.iter().map(|r| *r as usize).product();
        ensure!(
            latents.len() == 2 * streams * length && out.len() == 2 * length * total,
            "vocoder extent mismatch"
        );
        let s = &self.stream;
        let mut wave: CudaSlice<f32> = unsafe { s.alloc(length * total)? };
        for half in 0..2 {
            let latent = latents.slice(half * streams * length..(half + 1) * streams * length);
            self.vocoder_stream(&latent, length, &mut wave)?;
            s.memcpy_dtod(
                &wave,
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
        input: &CudaSlice<f32>,
        weight: &CudaSlice<u8>,
        bias: &CudaSlice<u8>,
        rows: usize,
        width: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(input.len() == rows * width && out.len() == rows * width);
        let stride = width as u64;
        unsafe {
            self.stream
                .launch_builder(&self.layernorm)
                .arg(input)
                .arg(weight)
                .arg(bias)
                .arg(out)
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

/// C = alpha * op(A) * op(B) in CUBLAS_COMPUTE_32F (no TF32) on raw device
/// pointers; the same call form as `Blas::linear_f32`.
struct GemmF32 {
    transa: cudarc::cublas::sys::cublasOperation_t,
    transb: cudarc::cublas::sys::cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    lda: i32,
    ldb: i32,
    ldc: i32,
}

unsafe fn gemm_f32(handle: &CudaBlas, cfg: &GemmF32, a: u64, b: u64, c: u64) -> Result<()> {
    let GemmF32 {
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        lda,
        ldb,
        ldc,
    } = *cfg;
    let beta = 0f32;
    unsafe {
        cudarc::cublas::sys::cublasGemmEx(
            *handle.handle(),
            transa,
            transb,
            m,
            n,
            k,
            (&alpha as *const f32).cast(),
            a as *const core::ffi::c_void,
            cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
            lda,
            b as *const core::ffi::c_void,
            cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
            ldb,
            (&beta as *const f32).cast(),
            c as *mut core::ffi::c_void,
            cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F,
            ldc,
            cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .result()
        .context("cublasGemmEx failed")?;
    }
    Ok(())
}

/// Folded vocoder weights: weight-norm convs folded once at setup
/// (g * v / ||v||, in memory) and their kernel planes repacked dense.
pub struct VocoderSetup {
    pub conv_in_planes: Vec<CudaSlice<f32>>,
    pub blocks: Vec<VocoderBlockSetup>,
    pub conv_out_planes: Vec<CudaSlice<f32>>,
}

pub struct VocoderBlockSetup {
    pub conv_t_planes: Vec<CudaSlice<f32>>,
    pub units: Vec<VocoderUnitSetup>,
}

pub struct VocoderUnitSetup {
    pub c1_planes: Vec<CudaSlice<f32>>,
    pub c2_w: CudaSlice<f32>,
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
