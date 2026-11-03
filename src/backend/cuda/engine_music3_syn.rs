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
        Ok(Self {
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
