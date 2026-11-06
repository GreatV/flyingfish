use super::*;
use crate::backend::cuda::Device;
use crate::backend::cuda::blas::Blas;
use crate::backend::cuda::cubin;
use crate::backend::gpu::DevicePtr;
use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaGraph, CudaSlice, CudaStream};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

pub(crate) fn fp8_dequant_block(
    codes: &[u8],
    scale_inv: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<bf16> {
    let scale_cols = cols.div_ceil(128);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let sr = r / 128;
        for c in 0..cols {
            let sc = c / 128;
            let scale = scale_inv[sr * scale_cols + sc];
            let v = crate::fp8::decode_e4m3(codes[r * cols + c]);
            out.push(bf16::from_f32(v * scale));
        }
    }
    out
}

pub(crate) fn run_linear_bf16(
    stream: &Arc<cudarc::driver::CudaStream>,
    blas: &Blas,
    weight: &CudaSlice<bf16>,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    shape: (usize, usize, usize),
) -> Result<()> {
    let (rows, output_dim, input_dim) = shape;
    let (w_ptr, _w_drop) = weight.device_ptr(stream);
    let (x_ptr, _x_drop) = x.device_ptr(stream);
    let (y_ptr, _y_drop) = y.device_ptr(stream);
    let w_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            w_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim * 2) as u64,
        )?
    };
    let x_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            x_ptr,
            [rows as u32, input_dim as u32],
            [input_dim as u64, 1],
            (rows * input_dim * 2) as u64,
        )?
    };
    let mut y_tensor = unsafe {
        crate::backend::cuda::contracts::TensorMut(crate::backend::cuda::contracts::Tensor::new(
            y_ptr,
            [rows as u32, output_dim as u32],
            [output_dim as u64, 1],
            (rows * output_dim * 2) as u64,
        )?)
    };
    blas.linear_bf16(w_tensor, x_tensor, &mut y_tensor)
}

pub(crate) fn run_linear_bf16_to_f32(
    stream: &Arc<cudarc::driver::CudaStream>,
    blas: &Blas,
    weight: &CudaSlice<bf16>,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<f32>,
    shape: (usize, usize, usize),
) -> Result<()> {
    let (rows, output_dim, input_dim) = shape;
    let (w_ptr, _w_drop) = weight.device_ptr(stream);
    let (x_ptr, _x_drop) = x.device_ptr(stream);
    let (y_ptr, _y_drop) = y.device_ptr(stream);
    let w_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            w_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim * 2) as u64,
        )?
    };
    let x_tensor = unsafe {
        crate::backend::cuda::contracts::Tensor::new(
            x_ptr,
            [rows as u32, input_dim as u32],
            [input_dim as u64, 1],
            (rows * input_dim * 2) as u64,
        )?
    };
    let mut y_tensor = unsafe {
        crate::backend::cuda::contracts::TensorMut(crate::backend::cuda::contracts::Tensor::new(
            y_ptr,
            [rows as u32, output_dim as u32],
            [output_dim as u64, 1],
            (rows * output_dim * 4) as u64,
        )?)
    };
    blas.linear_bf16_to_f32(w_tensor, x_tensor, &mut y_tensor)
}

pub(crate) fn run_rms_norm(
    stream: &cudarc::driver::CudaStream,
    norm_fn: &CudaFunction,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    dims: (i32, i32),
    eps: f32,
) -> Result<()> {
    let (rows, dim) = dims;
    unsafe {
        stream
            .launch_builder(norm_fn)
            .arg(x)
            .arg(w)
            .arg(y)
            .arg(&rows)
            .arg(&dim)
            .arg(&eps)
            .launch(crate::backend::cuda::ops::grid(rows as usize, 256))?;
    }
    Ok(())
}

pub(crate) fn run_silu_mul_split(
    stream: &cudarc::driver::CudaStream,
    silu_fn: &CudaFunction,
    gate: &CudaSlice<bf16>,
    up: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    count: i32,
) -> Result<()> {
    let mode = 1i32;
    unsafe {
        stream
            .launch_builder(silu_fn)
            .arg(gate)
            .arg(up)
            .arg(y)
            .arg(&count)
            .arg(&mode)
            .launch(crate::backend::cuda::ops::grid(
                (count as usize).div_ceil(8),
                256,
            ))?;
    }
    Ok(())
}

pub(crate) fn run_concat_eh(
    stream: &cudarc::driver::CudaStream,
    concat_fn: &CudaFunction,
    enorm: &CudaSlice<bf16>,
    hnorm: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    dim: i32,
    swap: bool,
) -> Result<()> {
    let swap_i32 = if swap { 1i32 } else { 0i32 };
    unsafe {
        stream
            .launch_builder(concat_fn)
            .arg(enorm)
            .arg(hnorm)
            .arg(out)
            .arg(&dim)
            .arg(&swap_i32)
            .launch(crate::backend::cuda::ops::grid(
                (dim as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

pub(crate) fn run_accum_scaled(
    stream: &cudarc::driver::CudaStream,
    accum_fn: &CudaFunction,
    accum: &mut CudaSlice<bf16>,
    val: &CudaSlice<bf16>,
    scale: f32,
    count: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(accum_fn)
            .arg(accum)
            .arg(val)
            .arg(&scale)
            .arg(&count)
            .launch(crate::backend::cuda::ops::grid(
                (count as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

pub(crate) fn run_add(
    stream: &cudarc::driver::CudaStream,
    add_fn: &CudaFunction,
    x: &mut CudaSlice<bf16>,
    res: &CudaSlice<bf16>,
    count: i32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(add_fn)
            .arg(x)
            .arg(res)
            .arg(&count)
            .launch(crate::backend::cuda::ops::grid(
                (count as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

pub(crate) fn run_extract_dsa_value(
    stream: &cudarc::driver::CudaStream,
    extract_fn: &CudaFunction,
    kv_expanded: &CudaSlice<bf16>,
    attn_out: &mut CudaSlice<bf16>,
    heads: i32,
    head_dim: i32,
) -> Result<()> {
    let total = heads * head_dim;
    unsafe {
        stream
            .launch_builder(extract_fn)
            .arg(kv_expanded)
            .arg(attn_out)
            .arg(&heads)
            .arg(&head_dim)
            .launch(crate::backend::cuda::ops::grid(
                (total as usize).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct GlmMtpExpertHost {
    pub(crate) gate_codes: Vec<u8>,
    pub(crate) gate_scales: Vec<f32>,
    pub(crate) up_codes: Vec<u8>,
    pub(crate) up_scales: Vec<f32>,
    pub(crate) down_codes: Vec<u8>,
    pub(crate) down_scales: Vec<f32>,
}

pub(crate) struct GlmMtpRuntime {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) silu_mul_split: CudaFunction,
    pub(crate) concat_eh: CudaFunction,
    pub(crate) accum_scaled: CudaFunction,
    pub(crate) add: CudaFunction,
    pub(crate) extract_dsa: CudaFunction,

    pub(crate) embed_tokens: Vec<bf16>,
    enorm_w: CudaSlice<bf16>,
    hnorm_w: CudaSlice<bf16>,
    eh_proj_w: CudaSlice<bf16>,
    input_norm_w: CudaSlice<bf16>,
    q_a_w: CudaSlice<bf16>,
    q_a_norm_w: CudaSlice<bf16>,
    q_b_w: CudaSlice<bf16>,
    kv_a_w: CudaSlice<bf16>,
    kv_a_norm_w: CudaSlice<bf16>,
    kv_b_w: CudaSlice<bf16>,
    o_w: CudaSlice<bf16>,
    post_norm_w: CudaSlice<bf16>,
    sh_gate_w: CudaSlice<bf16>,
    sh_up_w: CudaSlice<bf16>,
    sh_down_w: CudaSlice<bf16>,
    router_w: CudaSlice<bf16>,
    router_b: Vec<f32>,
    experts: Vec<GlmMtpExpertHost>,
    slot_gate: Vec<CudaSlice<bf16>>,
    slot_up: Vec<CudaSlice<bf16>>,
    slot_down: Vec<CudaSlice<bf16>>,
    head_norm_w: CudaSlice<bf16>,
    lm_head_w: CudaSlice<bf16>,

    pub(crate) token_embed_buf: CudaSlice<bf16>,
    pub(crate) enorm_buf: CudaSlice<bf16>,
    pub(crate) target_hidden_buf: CudaSlice<bf16>,
    pub(crate) hnorm_buf: CudaSlice<bf16>,
    pub(crate) eh_buf: CudaSlice<bf16>,
    pub(crate) x_buf: CudaSlice<bf16>,
    pub(crate) x_norm_buf: CudaSlice<bf16>,
    pub(crate) q_a_buf: CudaSlice<bf16>,
    pub(crate) q_a_norm_buf: CudaSlice<bf16>,
    pub(crate) q_b_buf: CudaSlice<bf16>,
    pub(crate) kv_a_buf: CudaSlice<bf16>,
    pub(crate) kv_a_norm_buf: CudaSlice<bf16>,
    pub(crate) kv_b_buf: CudaSlice<bf16>,
    pub(crate) attn_out_buf: CudaSlice<bf16>,
    pub(crate) o_buf: CudaSlice<bf16>,
    pub(crate) sh_gate_buf: CudaSlice<bf16>,
    pub(crate) sh_up_buf: CudaSlice<bf16>,
    pub(crate) sh_act_buf: CudaSlice<bf16>,
    pub(crate) sh_out_buf: CudaSlice<bf16>,
    pub(crate) router_logits_buf: CudaSlice<bf16>,
    pub(crate) exp_gate_buf: CudaSlice<bf16>,
    pub(crate) exp_up_buf: CudaSlice<bf16>,
    pub(crate) exp_act_buf: CudaSlice<bf16>,
    pub(crate) exp_out_buf: CudaSlice<bf16>,
    pub(crate) routed_accum_buf: CudaSlice<bf16>,
    pub(crate) head_norm_buf: CudaSlice<bf16>,
    pub(crate) logits_buf: CudaSlice<f32>,
    staged_weights: [f32; 8],

    graph: Option<CudaGraph>,
}

impl GlmMtpRuntime {
    pub(crate) fn load(model_dir: &Path, device: &Device, blas: &Blas) -> Result<Self> {
        unsafe {
            device.ctx.disable_event_tracking();
        }
        let stream = &device.stream;

        let norm_mod = cubin::module(&device.ctx, "norm")?;
        let rms_norm = norm_mod.load_function("rms_norm")?;

        let act_mod = cubin::module(&device.ctx, "act")?;
        let silu_mul_split = act_mod.load_function("silu_mul_split")?;

        let mtp_mod = cubin::module(&device.ctx, "glm_mtp")?;
        let concat_eh = mtp_mod.load_function("glm_mtp_concat_eh")?;
        let accum_scaled = mtp_mod.load_function("glm_mtp_accum_scaled")?;
        let add = mtp_mod.load_function("glm_mtp_add")?;
        let extract_dsa = mtp_mod.load_function("glm_mtp_extract_dsa_value")?;

        let shard1_file = std::fs::File::open(model_dir.join("model-00001-of-00062.safetensors"))?;
        let mmap1 = unsafe { memmap2::MmapOptions::new().map(&shard1_file)? };
        let st1 = safetensors::SafeTensors::deserialize(&mmap1)?;

        let shard2_file = std::fs::File::open(model_dir.join("model-00002-of-00062.safetensors"))?;
        let mmap2 = unsafe { memmap2::MmapOptions::new().map(&shard2_file)? };
        let st2 = safetensors::SafeTensors::deserialize(&mmap2)?;

        let get_st_bf16 = |name: &str| -> Result<Vec<bf16>> {
            if let Ok(t) = st1.tensor(name) {
                let data = t.data();
                return Ok(data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                    .collect());
            }
            let t = st2.tensor(name)?;
            let data = t.data();
            Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        };

        let get_st_f32 = |name: &str| -> Result<Vec<f32>> {
            if let Ok(t) = st1.tensor(name) {
                let data = t.data();
                return Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect());
            }
            let t = st2.tensor(name)?;
            let data = t.data();
            Ok(data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };

        let get_st_u8 = |name: &str| -> Result<Vec<u8>> {
            if let Ok(t) = st1.tensor(name) {
                return Ok(t.data().to_vec());
            }
            let t = st2.tensor(name)?;
            Ok(t.data().to_vec())
        };

        let get_st_fp8_linear = |role: &str, rows: usize, cols: usize| -> Result<Vec<bf16>> {
            let code_name = format!("{role}.weight");
            let scale_name = format!("{role}.weight_scale_inv");
            let codes = get_st_u8(&code_name)?;
            let scales = get_st_f32(&scale_name)?;
            Ok(fp8_dequant_block(&codes, &scales, rows, cols))
        };

        let embed_tokens = get_st_bf16("model.language_model.embed_tokens.weight")?;
        let lm_head_w = stream.clone_htod(&get_st_bf16("lm_head.weight")?)?;

        let enorm_w =
            stream.clone_htod(&get_st_bf16("model.language_model.layers.45.enorm.weight")?)?;
        let hnorm_w =
            stream.clone_htod(&get_st_bf16("model.language_model.layers.45.hnorm.weight")?)?;
        let eh_proj_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.eh_proj.weight",
        )?)?;

        let input_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.input_layernorm.weight",
        )?)?;
        let q_a_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.q_a_proj.weight",
        )?)?;
        let q_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.q_a_layernorm.weight",
        )?)?;
        let q_b_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.q_b_proj.weight",
        )?)?;
        let kv_a_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight",
        )?)?;
        let kv_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_a_layernorm.weight",
        )?)?;
        let kv_b_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_b_proj.weight",
        )?)?;
        let o_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.o_proj.weight",
        )?)?;
        let post_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.post_attention_layernorm.weight",
        )?)?;

        let sh_gate_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.mlp.shared_experts.gate_proj",
            2048,
            4096,
        )?)?;
        let sh_up_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.mlp.shared_experts.up_proj",
            2048,
            4096,
        )?)?;
        let sh_down_w = stream.clone_htod(&get_st_fp8_linear(
            "model.language_model.layers.45.mlp.shared_experts.down_proj",
            4096,
            2048,
        )?)?;
        let router_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.mlp.gate.weight",
        )?)?;
        let router_b =
            get_st_f32("model.language_model.layers.45.mlp.gate.e_score_correction_bias")?;

        let head_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.shared_head.norm.weight",
        )?)?;

        let mut experts = Vec::with_capacity(288);
        for e in 0..288 {
            let pfx = format!("model.language_model.layers.45.mlp.experts.{e}");
            let gate_codes = get_st_u8(&format!("{pfx}.gate_proj.weight"))?;
            let gate_scales = get_st_f32(&format!("{pfx}.gate_proj.weight_scale_inv"))?;
            let up_codes = get_st_u8(&format!("{pfx}.up_proj.weight"))?;
            let up_scales = get_st_f32(&format!("{pfx}.up_proj.weight_scale_inv"))?;
            let down_codes = get_st_u8(&format!("{pfx}.down_proj.weight"))?;
            let down_scales = get_st_f32(&format!("{pfx}.down_proj.weight_scale_inv"))?;
            experts.push(GlmMtpExpertHost {
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
                down_codes,
                down_scales,
            });
        }

        let mut slot_gate = Vec::with_capacity(8);
        let mut slot_up = Vec::with_capacity(8);
        let mut slot_down = Vec::with_capacity(8);
        for _ in 0..8 {
            slot_gate.push(stream.alloc_zeros::<bf16>(2048 * 4096)?);
            slot_up.push(stream.alloc_zeros::<bf16>(2048 * 4096)?);
            slot_down.push(stream.alloc_zeros::<bf16>(4096 * 2048)?);
        }

        let mut runtime = Self {
            stream: stream.clone(),
            rms_norm,
            silu_mul_split,
            concat_eh,
            accum_scaled,
            add,
            extract_dsa,
            embed_tokens,
            enorm_w,
            hnorm_w,
            eh_proj_w,
            input_norm_w,
            q_a_w,
            q_a_norm_w,
            q_b_w,
            kv_a_w,
            kv_a_norm_w,
            kv_b_w,
            o_w,
            post_norm_w,
            sh_gate_w,
            sh_up_w,
            sh_down_w,
            router_w,
            router_b,
            experts,
            slot_gate,
            slot_up,
            slot_down,
            head_norm_w,
            lm_head_w,
            token_embed_buf: stream.alloc_zeros::<bf16>(4096)?,
            enorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            target_hidden_buf: stream.alloc_zeros::<bf16>(4096)?,
            hnorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            eh_buf: stream.alloc_zeros::<bf16>(8192)?,
            x_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
            q_a_buf: stream.alloc_zeros::<bf16>(1536)?,
            q_a_norm_buf: stream.alloc_zeros::<bf16>(1536)?,
            q_b_buf: stream.alloc_zeros::<bf16>(16384)?,
            kv_a_buf: stream.alloc_zeros::<bf16>(512)?,
            kv_a_norm_buf: stream.alloc_zeros::<bf16>(512)?,
            kv_b_buf: stream.alloc_zeros::<bf16>(32768)?,
            attn_out_buf: stream.alloc_zeros::<bf16>(16384)?,
            o_buf: stream.alloc_zeros::<bf16>(4096)?,
            sh_gate_buf: stream.alloc_zeros::<bf16>(2048)?,
            sh_up_buf: stream.alloc_zeros::<bf16>(2048)?,
            sh_act_buf: stream.alloc_zeros::<bf16>(2048)?,
            sh_out_buf: stream.alloc_zeros::<bf16>(4096)?,
            router_logits_buf: stream.alloc_zeros::<bf16>(288)?,
            exp_gate_buf: stream.alloc_zeros::<bf16>(2048)?,
            exp_up_buf: stream.alloc_zeros::<bf16>(2048)?,
            exp_act_buf: stream.alloc_zeros::<bf16>(2048)?,
            exp_out_buf: stream.alloc_zeros::<bf16>(4096)?,
            routed_accum_buf: stream.alloc_zeros::<bf16>(4096)?,
            head_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
            logits_buf: stream.alloc_zeros::<f32>(154880)?,
            staged_weights: [0.0; 8],
            graph: None,
        };

        // Capture graph for 1-row execution
        runtime.capture_graph(blas)?;
        Ok(runtime)
    }

    pub(crate) fn forward_staged_kernels(&mut self, swap_eh: bool, blas: &Blas) -> Result<()> {
        let stream = self.stream.clone();

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.token_embed_buf,
            &self.enorm_w,
            &mut self.enorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.target_hidden_buf,
            &self.hnorm_w,
            &mut self.hnorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_concat_eh(
            &stream,
            &self.concat_eh,
            &self.enorm_buf,
            &self.hnorm_buf,
            &mut self.eh_buf,
            4096,
            swap_eh,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.eh_proj_w,
            &self.eh_buf,
            &mut self.x_buf,
            (1, 4096, 8192),
        )?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.q_a_w,
            &self.x_norm_buf,
            &mut self.q_a_buf,
            (1, 1536, 4096),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.q_a_buf,
            &self.q_a_norm_w,
            &mut self.q_a_norm_buf,
            (1, 1536),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.q_b_w,
            &self.q_a_norm_buf,
            &mut self.q_b_buf,
            (1, 16384, 1536),
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.kv_a_w,
            &self.x_norm_buf,
            &mut self.kv_a_buf,
            (1, 512, 4096),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.kv_a_buf,
            &self.kv_a_norm_w,
            &mut self.kv_a_norm_buf,
            (1, 512),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.kv_b_w,
            &self.kv_a_norm_buf,
            &mut self.kv_b_buf,
            (1, 32768, 512),
        )?;
        run_extract_dsa_value(
            &stream,
            &self.extract_dsa,
            &self.kv_b_buf,
            &mut self.attn_out_buf,
            64,
            256,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.o_w,
            &self.attn_out_buf,
            &mut self.o_buf,
            (1, 4096, 16384),
        )?;
        run_add(&stream, &self.add, &mut self.x_buf, &self.o_buf, 4096)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.post_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.sh_gate_w,
            &self.x_norm_buf,
            &mut self.sh_gate_buf,
            (1, 2048, 4096),
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.sh_up_w,
            &self.x_norm_buf,
            &mut self.sh_up_buf,
            (1, 2048, 4096),
        )?;
        run_silu_mul_split(
            &stream,
            &self.silu_mul_split,
            &self.sh_gate_buf,
            &self.sh_up_buf,
            &mut self.sh_act_buf,
            2048,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.sh_down_w,
            &self.sh_act_buf,
            &mut self.sh_out_buf,
            (1, 4096, 2048),
        )?;

        stream.memset_zeros(&mut self.routed_accum_buf)?;

        for k in 0..8 {
            let w = self.staged_weights[k];
            run_linear_bf16(
                &stream,
                blas,
                &self.slot_gate[k],
                &self.x_norm_buf,
                &mut self.exp_gate_buf,
                (1, 2048, 4096),
            )?;
            run_linear_bf16(
                &stream,
                blas,
                &self.slot_up[k],
                &self.x_norm_buf,
                &mut self.exp_up_buf,
                (1, 2048, 4096),
            )?;
            run_silu_mul_split(
                &stream,
                &self.silu_mul_split,
                &self.exp_gate_buf,
                &self.exp_up_buf,
                &mut self.exp_act_buf,
                2048,
            )?;
            run_linear_bf16(
                &stream,
                blas,
                &self.slot_down[k],
                &self.exp_act_buf,
                &mut self.exp_out_buf,
                (1, 4096, 2048),
            )?;
            run_accum_scaled(
                &stream,
                &self.accum_scaled,
                &mut self.routed_accum_buf,
                &self.exp_out_buf,
                w,
                4096,
            )?;
        }

        run_add(&stream, &self.add, &mut self.x_buf, &self.sh_out_buf, 4096)?;
        run_add(
            &stream,
            &self.add,
            &mut self.x_buf,
            &self.routed_accum_buf,
            4096,
        )?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.head_norm_w,
            &mut self.head_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16_to_f32(
            &stream,
            blas,
            &self.lm_head_w,
            &self.head_norm_buf,
            &mut self.logits_buf,
            (1, 154880, 4096),
        )?;

        Ok(())
    }

    pub(crate) fn stage_top8_experts(&mut self) -> Result<()> {
        let stream = &self.stream;
        let raw_logits = stream.clone_dtoh(&self.router_logits_buf)?;

        let mut scores: Vec<(usize, f32)> = raw_logits
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let logit = l.to_f32() + self.router_b[i];
                let s = 1.0f32 / (1.0f32 + (-logit).exp());
                (i, s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top8 = &scores[..8];
        let sum: f32 = top8.iter().map(|(_, s)| s).sum();

        for (k, &(exp_id, score)) in top8.iter().enumerate() {
            let w = (score / sum) * 2.5f32;
            self.staged_weights[k] = w;
            let exp = &self.experts[exp_id];
            let gate_bf16 = fp8_dequant_block(&exp.gate_codes, &exp.gate_scales, 2048, 4096);
            let up_bf16 = fp8_dequant_block(&exp.up_codes, &exp.up_scales, 2048, 4096);
            let down_bf16 = fp8_dequant_block(&exp.down_codes, &exp.down_scales, 4096, 2048);

            stream.memcpy_htod(&gate_bf16, &mut self.slot_gate[k])?;
            stream.memcpy_htod(&up_bf16, &mut self.slot_up[k])?;
            stream.memcpy_htod(&down_bf16, &mut self.slot_down[k])?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn forward(
        &mut self,
        token_id: u32,
        target_hidden: &[bf16],
        swap_eh: bool,
        blas: &Blas,
    ) -> Result<Vec<f32>> {
        let stream = self.stream.clone();

        let token_offset = (token_id as usize) * 4096;
        let token_slice = &self.embed_tokens[token_offset..token_offset + 4096];
        stream.memcpy_htod(token_slice, &mut self.token_embed_buf)?;
        stream.memcpy_htod(target_hidden, &mut self.target_hidden_buf)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.token_embed_buf,
            &self.enorm_w,
            &mut self.enorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.target_hidden_buf,
            &self.hnorm_w,
            &mut self.hnorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_concat_eh(
            &stream,
            &self.concat_eh,
            &self.enorm_buf,
            &self.hnorm_buf,
            &mut self.eh_buf,
            4096,
            swap_eh,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.eh_proj_w,
            &self.eh_buf,
            &mut self.x_buf,
            (1, 4096, 8192),
        )?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.router_w,
            &self.x_norm_buf,
            &mut self.router_logits_buf,
            (1, 288, 4096),
        )?;

        self.stage_top8_experts()?;
        self.forward_staged_kernels(swap_eh, blas)?;

        Ok(stream.clone_dtoh(&self.logits_buf)?)
    }

    pub(crate) fn capture_graph(&mut self, blas: &Blas) -> Result<()> {
        let stream = self.stream.clone();
        stream.memset_zeros(&mut self.logits_buf)?;
        stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        self.forward_staged_kernels(false, blas)?;
        let captured = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?;
        let graph = captured.context("Runner FAIL empty draft graph")?;
        graph.upload()?;
        self.graph = Some(graph);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn forward_graph(
        &mut self,
        token_id: u32,
        target_hidden: &[bf16],
        swap_eh: bool,
        blas: &Blas,
    ) -> Result<Vec<f32>> {
        let stream = self.stream.clone();
        let token_offset = (token_id as usize) * 4096;
        let token_slice = &self.embed_tokens[token_offset..token_offset + 4096];
        stream.memcpy_htod(token_slice, &mut self.token_embed_buf)?;
        stream.memcpy_htod(target_hidden, &mut self.target_hidden_buf)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.token_embed_buf,
            &self.enorm_w,
            &mut self.enorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.target_hidden_buf,
            &self.hnorm_w,
            &mut self.hnorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_concat_eh(
            &stream,
            &self.concat_eh,
            &self.enorm_buf,
            &self.hnorm_buf,
            &mut self.eh_buf,
            4096,
            swap_eh,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.eh_proj_w,
            &self.eh_buf,
            &mut self.x_buf,
            (1, 4096, 8192),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.router_w,
            &self.x_norm_buf,
            &mut self.router_logits_buf,
            (1, 288, 4096),
        )?;
        self.stage_top8_experts()?;

        let g = self.graph.as_ref().context("draft graph missing")?;
        g.launch()?;
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&self.logits_buf)?)
    }

    pub(crate) fn step_draft_device(
        &mut self,
        token_id: u32,
        target_hidden: &CudaSlice<bf16>,
        force_eager: bool,
        blas: &Blas,
    ) -> Result<(u32, Vec<f32>)> {
        let stream = self.stream.clone();
        let token_offset = (token_id as usize) * 4096;
        let token_slice = &self.embed_tokens[token_offset..token_offset + 4096];
        stream.memcpy_htod(token_slice, &mut self.token_embed_buf)?;
        stream.memcpy_dtod(&target_hidden.slice(..4096), &mut self.target_hidden_buf)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.token_embed_buf,
            &self.enorm_w,
            &mut self.enorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.target_hidden_buf,
            &self.hnorm_w,
            &mut self.hnorm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_concat_eh(
            &stream,
            &self.concat_eh,
            &self.enorm_buf,
            &self.hnorm_buf,
            &mut self.eh_buf,
            4096,
            false,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.eh_proj_w,
            &self.eh_buf,
            &mut self.x_buf,
            (1, 4096, 8192),
        )?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.router_w,
            &self.x_norm_buf,
            &mut self.router_logits_buf,
            (1, 288, 4096),
        )?;
        self.stage_top8_experts()?;

        if force_eager || self.graph.is_none() {
            self.forward_staged_kernels(false, blas)?;
        } else {
            let g = self.graph.as_ref().context("draft graph missing")?;
            g.launch()?;
            stream.synchronize()?;
        }

        let logits = stream.clone_dtoh(&self.logits_buf)?;
        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &val) in logits.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_id = i as u32;
            }
        }
        Ok((best_id, logits))
    }
}
