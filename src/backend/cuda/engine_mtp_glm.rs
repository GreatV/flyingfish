use super::super::qwen::{GlmDsaOutputs, heads, matrix};
use super::*;
use crate::backend::cuda::Device;
use crate::backend::cuda::blas::Blas;
use crate::backend::cuda::contracts::{Tensor, TensorMut, view};
use crate::backend::cuda::cubin;
use crate::backend::cuda::dsa::{
    DsaAppendInput, DsaAttentionInput, DsaIndexInput, DsaIndexOutput, DsaPlan, DsaState,
};
use crate::backend::cuda::linear::{Fp8ScaleRef, LinearWeightRef};
use crate::backend::cuda::moe::decode::{NativeMoe, NativeMoeGeometry, NativeMoeWeights};
use crate::backend::cuda::ops::{CommonOpsRef, NormOps};
use crate::backend::gpu::DevicePtr;
use crate::fp8::{Fp8BlockScale, Fp8ScaleLayout};
use crate::op_contract::{DsaMeta, DsaSpec};
use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

pub(crate) fn run_linear_bf16(
    stream: &Arc<CudaStream>,
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
        Tensor::new(
            w_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim * 2) as u64,
        )?
    };
    let x_tensor = unsafe {
        Tensor::new(
            x_ptr,
            [rows as u32, input_dim as u32],
            [input_dim as u64, 1],
            (rows * input_dim * 2) as u64,
        )?
    };
    let mut y_tensor = unsafe {
        TensorMut(Tensor::new(
            y_ptr,
            [rows as u32, output_dim as u32],
            [output_dim as u64, 1],
            (rows * output_dim * 2) as u64,
        )?)
    };
    blas.linear_bf16(w_tensor, x_tensor, &mut y_tensor)
}

pub(crate) fn run_linear_bf16_to_f32(
    stream: &Arc<CudaStream>,
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
        Tensor::new(
            w_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim * 2) as u64,
        )?
    };
    let x_tensor = unsafe {
        Tensor::new(
            x_ptr,
            [rows as u32, input_dim as u32],
            [input_dim as u64, 1],
            (rows * input_dim * 2) as u64,
        )?
    };
    let mut y_tensor = unsafe {
        TensorMut(Tensor::new(
            y_ptr,
            [rows as u32, output_dim as u32],
            [output_dim as u64, 1],
            (rows * output_dim * 4) as u64,
        )?)
    };
    blas.linear_bf16_to_f32(w_tensor, x_tensor, &mut y_tensor)
}

pub(crate) fn run_rms_norm(
    stream: &CudaStream,
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

pub(crate) fn run_concat_eh(
    stream: &CudaStream,
    concat_fn: &CudaFunction,
    enorm: &CudaSlice<bf16>,
    hnorm: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    dim: i32,
    swap: bool,
) -> Result<()> {
    let mode: i32 = if swap { 1 } else { 0 };
    unsafe {
        stream
            .launch_builder(concat_fn)
            .arg(enorm)
            .arg(hnorm)
            .arg(out)
            .arg(&dim)
            .arg(&mode)
            .launch(crate::backend::cuda::ops::grid(
                (dim as usize * 2).div_ceil(256),
                256,
            ))?;
    }
    Ok(())
}

pub(crate) fn run_add(
    stream: &CudaStream,
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

fn dequant_fp8_block_f32_host(codes: &[u8], scales: &[f32], rows: usize, cols: usize) -> Vec<bf16> {
    let scale_cols = cols.div_ceil(128);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let sr = r / 128;
        let row_offset = r * cols;
        let scale_row_offset = sr * scale_cols;
        for c in 0..cols {
            let sc = c / 128;
            let scale = scales[scale_row_offset + sc];
            let code = codes[row_offset + c];
            let val = crate::fp8::decode_e4m3(code) * scale;
            out.push(bf16::from_f32(val));
        }
    }
    out
}

#[derive(Clone, Copy)]
pub(crate) struct GlmMtpSliceInfo {
    pub(crate) address: usize,
    pub(crate) len: usize,
}

#[derive(Clone)]
pub(crate) struct GlmMtpExpertHost {
    pub(crate) gate_codes: GlmMtpSliceInfo,
    pub(crate) gate_scales: Vec<f32>,
    pub(crate) up_codes: GlmMtpSliceInfo,
    pub(crate) up_scales: Vec<f32>,
    pub(crate) down_codes: GlmMtpSliceInfo,
    pub(crate) down_scales: Vec<f32>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct GlmMtpStageTaps {
    pub enorm: Vec<bf16>,
    pub hnorm: Vec<bf16>,
    pub eh_in: Vec<bf16>,
    pub x_init: Vec<bf16>,
    pub x_norm1: Vec<bf16>,
    pub kv_latent: Vec<bf16>,
    pub q_a: Vec<bf16>,
    pub q_out: Vec<bf16>,
    pub attn_out: Vec<bf16>,
    pub o_out: Vec<bf16>,
    pub x_curr: Vec<bf16>,
    pub x_norm2: Vec<bf16>,
    pub router_logits: Vec<f32>,
    pub routing_ids: Vec<i32>,
    pub routing_weights: Vec<f32>,
    pub moe_out: Vec<bf16>,
}

#[derive(Clone, Copy)]
pub(crate) struct GlmMtpShared<'a> {
    pub(crate) blas: &'a Blas,
    pub(crate) embedding: &'a CudaSlice<bf16>,
    pub(crate) head: &'a CudaSlice<bf16>,
}

pub(crate) struct GlmMtpRuntime {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) concat_eh: CudaFunction,
    pub(crate) add: CudaFunction,

    pub(crate) dsa_plan: DsaPlan,
    pub(crate) native_moe: NativeMoe,

    enorm_w: CudaSlice<bf16>,
    hnorm_w: CudaSlice<bf16>,
    eh_proj_w: CudaSlice<bf16>,
    input_norm_w: CudaSlice<bf16>,

    kv_a_w: CudaSlice<bf16>,
    kv_a_norm_w: CudaSlice<bf16>,
    kv_b_w: CudaSlice<bf16>,

    q_a_w: CudaSlice<bf16>,
    q_a_norm_w: CudaSlice<bf16>,
    q_b_w: CudaSlice<bf16>,

    o_w: CudaSlice<bf16>,
    post_norm_w: CudaSlice<bf16>,

    _router_w: CudaSlice<bf16>,
    _router_b: CudaSlice<f32>,
    _sh_gate_codes: CudaSlice<u8>,
    _sh_gate_scales: CudaSlice<f32>,
    _sh_up_codes: CudaSlice<u8>,
    _sh_up_scales: CudaSlice<f32>,
    _sh_down_codes: CudaSlice<u8>,
    _sh_down_scales: CudaSlice<f32>,

    slots_buf: CudaSlice<u64>,
    valid_buf: CudaSlice<u32>,
    slots_host: Vec<u64>,
    valid_host: Vec<u32>,

    slot_gate_codes: Vec<CudaSlice<u8>>,
    slot_gate_scales: Vec<CudaSlice<f32>>,
    slot_up_codes: Vec<CudaSlice<u8>>,
    slot_up_scales: Vec<CudaSlice<f32>>,
    slot_down_codes: Vec<CudaSlice<u8>>,
    slot_down_scales: Vec<CudaSlice<f32>>,
    staged_slots: [Option<usize>; 8],
    experts: Vec<GlmMtpExpertHost>,
    _checkpoint: crate::checkpoint::MappedCheckpoint,

    head_norm_w: CudaSlice<bf16>,

    pub(crate) _standalone_embed: Option<CudaSlice<bf16>>,
    pub(crate) standalone_head: Option<CudaSlice<bf16>>,

    pub(crate) token_embed_buf: CudaSlice<bf16>,
    pub(crate) enorm_buf: CudaSlice<bf16>,
    pub(crate) target_hidden_buf: CudaSlice<bf16>,
    pub(crate) hnorm_buf: CudaSlice<bf16>,
    pub(crate) eh_buf: CudaSlice<bf16>,
    pub(crate) x_buf: CudaSlice<bf16>,
    pub(crate) x_init_buf: CudaSlice<bf16>,
    pub(crate) x_norm_buf: CudaSlice<bf16>,
    pub(crate) x_norm1_buf: CudaSlice<bf16>,
    pub(crate) x_curr_buf: CudaSlice<bf16>,
    pub(crate) kv_a_buf: CudaSlice<bf16>,
    pub(crate) kv_latent_buf: CudaSlice<bf16>,
    pub(crate) q_a_buf: CudaSlice<bf16>,
    pub(crate) q_a_norm_buf: CudaSlice<bf16>,
    pub(crate) q_b_buf: CudaSlice<bf16>,
    pub(crate) attn_out_buf: CudaSlice<bf16>,
    pub(crate) o_buf: CudaSlice<bf16>,
    pub(crate) moe_out_buf: CudaSlice<bf16>,
    pub(crate) head_norm_buf: CudaSlice<bf16>,
    pub(crate) logits_buf: CudaSlice<f32>,
    index_norm: Arc<NormOps>,
    index_wq: CudaSlice<bf16>,
    index_wk: CudaSlice<bf16>,
    index_gate: CudaSlice<bf16>,
    index_weights: CudaSlice<bf16>,
    index_norm_w: CudaSlice<bf16>,
    index_norm_b: CudaSlice<bf16>,
    index_ape: CudaSlice<bf16>,
    index_q: CudaSlice<bf16>,
    index_k: CudaSlice<bf16>,
    index_g: CudaSlice<bf16>,
    index_head_weights: CudaSlice<bf16>,
    index_outputs: GlmDsaOutputs,

    pub(crate) last_route: Option<(Vec<i32>, Vec<f32>)>,
    pub(crate) last_staged_experts: usize,

    pub(crate) _standalone_dsa: Option<DsaState>,
}

impl GlmMtpRuntime {
    pub(crate) fn load(
        model_dir: &Path,
        device: &Device,
        _blas: &Blas,
        standalone_weights: bool,
        capacity: usize,
    ) -> Result<Self> {
        unsafe {
            device.ctx.disable_event_tracking();
        }
        let stream = &device.stream;

        let norm_mod = cubin::module(&device.ctx, "norm")?;
        let rms_norm = norm_mod.load_function("rms_norm")?;

        let mtp_mod = cubin::module(&device.ctx, "glm_mtp")?;
        let concat_eh = mtp_mod.load_function("glm_mtp_concat_eh")?;
        let add = mtp_mod.load_function("glm_mtp_add")?;

        let checkpoint = crate::checkpoint::MappedCheckpoint::open_component(model_dir, "model")?;
        let views = checkpoint.views()?;
        let tensor = |name: &str| -> Result<safetensors::tensor::TensorView<'_>> {
            let shard = checkpoint
                .index
                .weight_map
                .get(name)
                .with_context(|| format!("GLM MTP tensor missing from index: {name}"))?;
            Ok(views
                .get(shard)
                .context("indexed shard missing")?
                .tensor(name)?)
        };
        let get_st_bf16 = |name: &str| -> Result<Vec<bf16>> {
            let t = tensor(name)?;
            ensure!(
                t.dtype() == safetensors::Dtype::BF16,
                "{name}: expected BF16"
            );
            Ok(t.data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
                .collect())
        };
        let get_st_f32 = |name: &str| -> Result<Vec<f32>> {
            let t = tensor(name)?;
            ensure!(t.dtype() == safetensors::Dtype::F32, "{name}: expected F32");
            Ok(t.data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect())
        };
        let get_st_slice_info = |name: &str| -> Result<GlmMtpSliceInfo> {
            let t = tensor(name)?;
            Ok(GlmMtpSliceInfo {
                address: t.data().as_ptr() as usize,
                len: t.data().len(),
            })
        };
        let slice_u8_init = |info: GlmMtpSliceInfo| -> &[u8] {
            // The immutable mapped range is owned by checkpoint.
            unsafe { std::slice::from_raw_parts(info.address as *const u8, info.len) }
        };

        let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let config_val: serde_json::Value = serde_json::from_str(&config_str)?;
        let routed_scaling_factor = config_val
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(2.5) as f32;
        let swiglu_limit = config_val
            .get("swiglu_limit")
            .and_then(|v| v.as_f64())
            .unwrap_or(10.0) as f32;

        let dsa_spec = DsaSpec::glm53();
        let dsa_plan = DsaPlan::new(&device.ctx, &device.info, dsa_spec, capacity as u32, 1)?;
        let _standalone_dsa = if standalone_weights {
            Some(DsaState::new(dsa_spec, capacity as u32, stream)?)
        } else {
            None
        };

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

        let q_a_w = stream.clone_htod(&dequant_fp8_block_f32_host(
            slice_u8_init(get_st_slice_info(
                "model.language_model.layers.45.self_attn.q_a_proj.weight",
            )?),
            &get_st_f32("model.language_model.layers.45.self_attn.q_a_proj.weight_scale_inv")?,
            1536,
            4096,
        ))?;
        let q_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.q_a_layernorm.weight",
        )?)?;
        let q_b_w = stream.clone_htod(&dequant_fp8_block_f32_host(
            slice_u8_init(get_st_slice_info(
                "model.language_model.layers.45.self_attn.q_b_proj.weight",
            )?),
            &get_st_f32("model.language_model.layers.45.self_attn.q_b_proj.weight_scale_inv")?,
            16384,
            1536,
        ))?;

        let kv_a_w = stream.clone_htod(&dequant_fp8_block_f32_host(
            slice_u8_init(get_st_slice_info(
                "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight",
            )?),
            &get_st_f32(
                "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            )?,
            512,
            4096,
        ))?;
        let kv_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_a_layernorm.weight",
        )?)?;
        let kv_b_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_b_proj.weight",
        )?)?;

        let o_w = stream.clone_htod(&dequant_fp8_block_f32_host(
            slice_u8_init(get_st_slice_info(
                "model.language_model.layers.45.self_attn.o_proj.weight",
            )?),
            &get_st_f32("model.language_model.layers.45.self_attn.o_proj.weight_scale_inv")?,
            4096,
            16384,
        ))?;
        let post_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.post_attention_layernorm.weight",
        )?)?;

        let router_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.mlp.gate.weight",
        )?)?;
        let router_b = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.gate.e_score_correction_bias",
        )?)?;

        let sh_gate_codes = stream.clone_htod(slice_u8_init(get_st_slice_info(
            "model.language_model.layers.45.mlp.shared_experts.gate_proj.weight",
        )?))?;
        let sh_gate_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.shared_experts.gate_proj.weight_scale_inv",
        )?)?;
        let sh_up_codes = stream.clone_htod(slice_u8_init(get_st_slice_info(
            "model.language_model.layers.45.mlp.shared_experts.up_proj.weight",
        )?))?;
        let sh_up_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.shared_experts.up_proj.weight_scale_inv",
        )?)?;
        let sh_down_codes = stream.clone_htod(slice_u8_init(get_st_slice_info(
            "model.language_model.layers.45.mlp.shared_experts.down_proj.weight",
        )?))?;
        let sh_down_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.shared_experts.down_proj.weight_scale_inv",
        )?)?;

        let mut experts = Vec::with_capacity(288);
        for i in 0..288 {
            let gate_codes = get_st_slice_info(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.gate_proj.weight"
            ))?;
            let gate_scales = get_st_f32(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.gate_proj.weight_scale_inv"
            ))?;
            let up_codes = get_st_slice_info(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.up_proj.weight"
            ))?;
            let up_scales = get_st_f32(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.up_proj.weight_scale_inv"
            ))?;
            let down_codes = get_st_slice_info(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.down_proj.weight"
            ))?;
            let down_scales = get_st_f32(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.down_proj.weight_scale_inv"
            ))?;
            experts.push(GlmMtpExpertHost {
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
                down_codes,
                down_scales,
            });
        }

        let mut slot_gate_codes = Vec::with_capacity(8);
        let mut slot_gate_scales = Vec::with_capacity(8);
        let mut slot_up_codes = Vec::with_capacity(8);
        let mut slot_up_scales = Vec::with_capacity(8);
        let mut slot_down_codes = Vec::with_capacity(8);
        let mut slot_down_scales = Vec::with_capacity(8);
        for _ in 0..8 {
            slot_gate_codes.push(stream.alloc_zeros::<u8>(2048 * 4096)?);
            slot_gate_scales.push(stream.alloc_zeros::<f32>(16 * 32)?);
            slot_up_codes.push(stream.alloc_zeros::<u8>(2048 * 4096)?);
            slot_up_scales.push(stream.alloc_zeros::<f32>(16 * 32)?);
            slot_down_codes.push(stream.alloc_zeros::<u8>(4096 * 2048)?);
            slot_down_scales.push(stream.alloc_zeros::<f32>(32 * 16)?);
        }

        let mut slots_host = vec![0u64; 288 * 6];
        let mut valid_host = vec![0u32; 288];

        for k in 0..8 {
            let exp = &experts[k];
            stream.memcpy_htod(slice_u8_init(exp.gate_codes), &mut slot_gate_codes[k])?;
            stream.memcpy_htod(&exp.gate_scales, &mut slot_gate_scales[k])?;
            stream.memcpy_htod(slice_u8_init(exp.up_codes), &mut slot_up_codes[k])?;
            stream.memcpy_htod(&exp.up_scales, &mut slot_up_scales[k])?;
            stream.memcpy_htod(slice_u8_init(exp.down_codes), &mut slot_down_codes[k])?;
            stream.memcpy_htod(&exp.down_scales, &mut slot_down_scales[k])?;

            let (gc_ptr, _) = slot_gate_codes[k].device_ptr(stream);
            let (gs_ptr, _) = slot_gate_scales[k].device_ptr(stream);
            let (uc_ptr, _) = slot_up_codes[k].device_ptr(stream);
            let (us_ptr, _) = slot_up_scales[k].device_ptr(stream);
            let (dc_ptr, _) = slot_down_codes[k].device_ptr(stream);
            let (ds_ptr, _) = slot_down_scales[k].device_ptr(stream);

            slots_host[k * 6] = gc_ptr;
            slots_host[k * 6 + 1] = gs_ptr;
            slots_host[k * 6 + 2] = uc_ptr;
            slots_host[k * 6 + 3] = us_ptr;
            slots_host[k * 6 + 4] = dc_ptr;
            slots_host[k * 6 + 5] = ds_ptr;
            valid_host[k] = 1;
        }

        let slots_buf = stream.clone_htod(&slots_host)?;
        let valid_buf = stream.clone_htod(&valid_host)?;

        let router_tensor = unsafe {
            Tensor::new(
                router_w.device_ptr(stream).0,
                [288, 4096],
                [4096, 1],
                288 * 4096 * 2,
            )?
        };
        let bias_tensor =
            unsafe { Tensor::new(router_b.device_ptr(stream).0, [1, 288], [288, 1], 288 * 4)? };
        let slots_tensor = unsafe {
            Tensor::new(
                slots_buf.device_ptr(stream).0,
                [288, 6],
                [6, 1],
                288 * 6 * 8,
            )?
        };
        let valid_tensor =
            unsafe { Tensor::new(valid_buf.device_ptr(stream).0, [288], [1], 288 * 4)? };

        let shared_gate = LinearWeightRef::Fp8 {
            codes: unsafe {
                Tensor::new(
                    sh_gate_codes.device_ptr(stream).0,
                    [2048, 4096],
                    [4096, 1],
                    2048 * 4096,
                )?
            },
            scale: Fp8ScaleRef::F32(unsafe {
                Tensor::new(
                    sh_gate_scales.device_ptr(stream).0,
                    [16, 32],
                    [32, 1],
                    16 * 32 * 4,
                )?
            }),
            layout: Fp8ScaleLayout::Block(
                Fp8BlockScale::new([128, 128], safetensors::Dtype::F32).unwrap(),
            ),
        };
        let shared_up = LinearWeightRef::Fp8 {
            codes: unsafe {
                Tensor::new(
                    sh_up_codes.device_ptr(stream).0,
                    [2048, 4096],
                    [4096, 1],
                    2048 * 4096,
                )?
            },
            scale: Fp8ScaleRef::F32(unsafe {
                Tensor::new(
                    sh_up_scales.device_ptr(stream).0,
                    [16, 32],
                    [32, 1],
                    16 * 32 * 4,
                )?
            }),
            layout: Fp8ScaleLayout::Block(
                Fp8BlockScale::new([128, 128], safetensors::Dtype::F32).unwrap(),
            ),
        };
        let shared_down = LinearWeightRef::Fp8 {
            codes: unsafe {
                Tensor::new(
                    sh_down_codes.device_ptr(stream).0,
                    [4096, 2048],
                    [2048, 1],
                    4096 * 2048,
                )?
            },
            scale: Fp8ScaleRef::F32(unsafe {
                Tensor::new(
                    sh_down_scales.device_ptr(stream).0,
                    [32, 16],
                    [16, 1],
                    32 * 16 * 4,
                )?
            }),
            layout: Fp8ScaleLayout::Block(
                Fp8BlockScale::new([128, 128], safetensors::Dtype::F32).unwrap(),
            ),
        };

        let geometry = NativeMoeGeometry {
            hidden: 4096,
            intermediate: 2048,
            experts: 288,
            top_k: 8,
            scaling: routed_scaling_factor,
            limit: swiglu_limit,
        };
        let weights = NativeMoeWeights {
            router: router_tensor,
            bias: bias_tensor,
            slots: slots_tensor,
            valid: valid_tensor,
            shared: [shared_gate, shared_up, shared_down],
        };
        let native_moe = NativeMoe::new(&device.ctx, stream, geometry, weights, 1)?;

        let head_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.shared_head.norm.weight",
        )?)?;

        let _standalone_embed = if standalone_weights {
            match get_st_bf16("model.language_model.embed_tokens.weight") {
                Ok(v) => Some(stream.clone_htod(&v)?),
                Err(_) => None,
            }
        } else {
            None
        };
        let standalone_head = if standalone_weights {
            match get_st_bf16("lm_head.weight") {
                Ok(v) => Some(stream.clone_htod(&v)?),
                Err(_) => None,
            }
        } else {
            None
        };

        let load_index = |suffix: &str| -> Result<CudaSlice<bf16>> {
            Ok(stream.clone_htod(&get_st_bf16(&format!(
                "model.language_model.layers.45.self_attn.indexer.{suffix}"
            ))?)?)
        };
        let index_wq = load_index("wq_b.weight")?;
        let index_wk = load_index("wk.weight")?;
        let index_gate = load_index("index_kpool_compress_gate")?;
        let index_weights = load_index("weights_proj.weight")?;
        let index_norm_w = load_index("k_norm.weight")?;
        let index_norm_b = load_index("k_norm.bias")?;
        let index_ape = load_index("index_kpool_compress_ape")?;
        drop(views);
        Ok(Self {
            stream: stream.clone(),
            rms_norm,
            concat_eh,
            add,
            dsa_plan,
            native_moe,
            enorm_w,
            hnorm_w,
            eh_proj_w,
            input_norm_w,
            kv_a_w,
            kv_a_norm_w,
            kv_b_w,
            q_a_w,
            q_a_norm_w,
            q_b_w,
            o_w,
            post_norm_w,
            _router_w: router_w,
            _router_b: router_b,
            _sh_gate_codes: sh_gate_codes,
            _sh_gate_scales: sh_gate_scales,
            _sh_up_codes: sh_up_codes,
            _sh_up_scales: sh_up_scales,
            _sh_down_codes: sh_down_codes,
            _sh_down_scales: sh_down_scales,
            slots_buf,
            valid_buf,
            slots_host,
            valid_host,
            slot_gate_codes,
            slot_gate_scales,
            slot_up_codes,
            slot_up_scales,
            slot_down_codes,
            slot_down_scales,
            staged_slots: [
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
            ],
            experts,
            _checkpoint: checkpoint,
            head_norm_w,
            _standalone_embed,
            standalone_head,
            token_embed_buf: stream.alloc_zeros::<bf16>(4096)?,
            enorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            target_hidden_buf: stream.alloc_zeros::<bf16>(4096)?,
            hnorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            eh_buf: stream.alloc_zeros::<bf16>(8192)?,
            x_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_init_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_norm1_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_curr_buf: stream.alloc_zeros::<bf16>(4096)?,
            kv_a_buf: stream.alloc_zeros::<bf16>(512)?,
            kv_latent_buf: stream.alloc_zeros::<bf16>(512)?,
            q_a_buf: stream.alloc_zeros::<bf16>(1536)?,
            q_a_norm_buf: stream.alloc_zeros::<bf16>(1536)?,
            q_b_buf: stream.alloc_zeros::<bf16>(16384)?,
            attn_out_buf: stream.alloc_zeros::<bf16>(16384)?,
            o_buf: stream.alloc_zeros::<bf16>(4096)?,
            moe_out_buf: stream.alloc_zeros::<bf16>(4096)?,
            head_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
            logits_buf: stream.alloc_zeros::<f32>(154880)?,
            index_norm: CommonOpsRef::new(&device.ctx)?.norm,
            index_wq,
            index_wk,
            index_gate,
            index_weights,
            index_norm_w,
            index_norm_b,
            index_ape,
            index_q: stream.alloc_zeros(32 * 128)?,
            index_k: stream.alloc_zeros(128)?,
            index_g: stream.alloc_zeros(128)?,
            index_head_weights: stream.alloc_zeros(32)?,
            index_outputs: GlmDsaOutputs::new(1, stream)?,
            last_route: None,
            last_staged_experts: 0,
            _standalone_dsa,
        })
    }

    pub(crate) fn allocation_bytes(&self) -> usize {
        let bf16_elems = self.enorm_w.len()
            + self.hnorm_w.len()
            + self.eh_proj_w.len()
            + self.input_norm_w.len()
            + self.kv_a_w.len()
            + self.kv_a_norm_w.len()
            + self.kv_b_w.len()
            + self.q_a_w.len()
            + self.q_a_norm_w.len()
            + self.q_b_w.len()
            + self.o_w.len()
            + self.post_norm_w.len()
            + self._router_w.len()
            + self.head_norm_w.len()
            + self._standalone_embed.as_ref().map_or(0, CudaSlice::len)
            + self.standalone_head.as_ref().map_or(0, CudaSlice::len)
            + self.token_embed_buf.len()
            + self.enorm_buf.len()
            + self.target_hidden_buf.len()
            + self.hnorm_buf.len()
            + self.eh_buf.len()
            + self.x_buf.len()
            + self.x_init_buf.len()
            + self.x_norm_buf.len()
            + self.x_norm1_buf.len()
            + self.x_curr_buf.len()
            + self.kv_a_buf.len()
            + self.kv_latent_buf.len()
            + self.q_a_buf.len()
            + self.q_a_norm_buf.len()
            + self.q_b_buf.len()
            + self.attn_out_buf.len()
            + self.o_buf.len()
            + self.moe_out_buf.len()
            + self.head_norm_buf.len()
            + self.index_wq.len()
            + self.index_wk.len()
            + self.index_gate.len()
            + self.index_weights.len()
            + self.index_norm_w.len()
            + self.index_norm_b.len()
            + self.index_ape.len()
            + self.index_q.len()
            + self.index_k.len()
            + self.index_g.len()
            + self.index_head_weights.len()
            + self.index_outputs.packed.len();
        let u8_elems = self._sh_gate_codes.len()
            + self._sh_up_codes.len()
            + self._sh_down_codes.len()
            + self
                .slot_gate_codes
                .iter()
                .map(CudaSlice::len)
                .sum::<usize>()
            + self.slot_up_codes.iter().map(CudaSlice::len).sum::<usize>()
            + self
                .slot_down_codes
                .iter()
                .map(CudaSlice::len)
                .sum::<usize>();
        let f32_elems = self._router_b.len()
            + self._sh_gate_scales.len()
            + self._sh_up_scales.len()
            + self._sh_down_scales.len()
            + self
                .slot_gate_scales
                .iter()
                .map(CudaSlice::len)
                .sum::<usize>()
            + self
                .slot_up_scales
                .iter()
                .map(CudaSlice::len)
                .sum::<usize>()
            + self
                .slot_down_scales
                .iter()
                .map(CudaSlice::len)
                .sum::<usize>()
            + self.logits_buf.len();
        self.native_moe.allocation_bytes()
            + bf16_elems * std::mem::size_of::<bf16>()
            + u8_elems
            + f32_elems * std::mem::size_of::<f32>()
            + self.slots_buf.len() * std::mem::size_of::<u64>()
            + self.valid_buf.len() * std::mem::size_of::<u32>()
            + (self.index_outputs.indices.len()
                + self.index_outputs.selected.len()
                + self.index_outputs.scores.len()
                + self.index_outputs.margins.len())
                * 4
            + self._standalone_dsa.as_ref().map_or(0, DsaState::bytes)
    }

    fn stage_routed_experts(&mut self, ids: &[i32]) -> Result<usize> {
        let stream = self.stream.clone();
        let slice_u8 = |info: GlmMtpSliceInfo| -> &[u8] {
            // The immutable mapped range is owned by self._checkpoint.
            unsafe { std::slice::from_raw_parts(info.address as *const u8, info.len) }
        };
        let target_experts: Vec<usize> = ids.iter().take(8).map(|&id| id as usize).collect();

        let mut needed_experts = Vec::new();
        for &exp_id in &target_experts {
            if !self.staged_slots.contains(&Some(exp_id)) {
                needed_experts.push(exp_id);
            }
        }

        let mut updated = false;
        if !needed_experts.is_empty() {
            let mut victim_slots = Vec::new();
            for (k, slot) in self.staged_slots.iter().enumerate() {
                if let Some(old_id) = *slot {
                    if !target_experts.contains(&old_id) {
                        victim_slots.push(k);
                    }
                } else {
                    victim_slots.push(k);
                }
            }

            for (&exp_id, &k) in needed_experts.iter().zip(&victim_slots) {
                let exp = &self.experts[exp_id];
                stream.memcpy_htod(slice_u8(exp.gate_codes), &mut self.slot_gate_codes[k])?;
                stream.memcpy_htod(&exp.gate_scales, &mut self.slot_gate_scales[k])?;
                stream.memcpy_htod(slice_u8(exp.up_codes), &mut self.slot_up_codes[k])?;
                stream.memcpy_htod(&exp.up_scales, &mut self.slot_up_scales[k])?;
                stream.memcpy_htod(slice_u8(exp.down_codes), &mut self.slot_down_codes[k])?;
                stream.memcpy_htod(&exp.down_scales, &mut self.slot_down_scales[k])?;

                let (gc_ptr, _) = self.slot_gate_codes[k].device_ptr(&stream);
                let (gs_ptr, _) = self.slot_gate_scales[k].device_ptr(&stream);
                let (uc_ptr, _) = self.slot_up_codes[k].device_ptr(&stream);
                let (us_ptr, _) = self.slot_up_scales[k].device_ptr(&stream);
                let (dc_ptr, _) = self.slot_down_codes[k].device_ptr(&stream);
                let (ds_ptr, _) = self.slot_down_scales[k].device_ptr(&stream);

                self.slots_host[exp_id * 6] = gc_ptr;
                self.slots_host[exp_id * 6 + 1] = gs_ptr;
                self.slots_host[exp_id * 6 + 2] = uc_ptr;
                self.slots_host[exp_id * 6 + 3] = us_ptr;
                self.slots_host[exp_id * 6 + 4] = dc_ptr;
                self.slots_host[exp_id * 6 + 5] = ds_ptr;

                self.staged_slots[k] = Some(exp_id);
                updated = true;
            }
        }

        let mut valid_changed = false;
        for (i, val) in self.valid_host.iter_mut().enumerate() {
            let expected = if target_experts.contains(&i) { 1 } else { 0 };
            if *val != expected {
                *val = expected;
                valid_changed = true;
            }
        }

        if updated {
            stream.memcpy_htod(&self.slots_host, &mut self.slots_buf)?;
        }
        if updated || valid_changed {
            stream.memcpy_htod(&self.valid_host, &mut self.valid_buf)?;
        }
        Ok(needed_experts.len())
    }

    fn index_input(&self) -> Result<DsaIndexInput> {
        Ok(DsaIndexInput {
            q: heads(&self.index_q, &self.stream, 0, 1, 32, 128)?,
            weights: matrix(&self.index_head_weights, &self.stream, 0, 1, 32)?,
            ape: matrix(&self.index_ape, &self.stream, 0, 4, 128)?,
        })
    }

    fn write_kv_entry(
        &mut self,
        dsa_state: &mut DsaState,
        pos: usize,
        swap_eh: bool,
        blas: &Blas,
    ) -> Result<()> {
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
        stream.memcpy_dtod(&self.x_buf, &mut self.x_init_buf)?;
        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.input_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;
        stream.memcpy_dtod(&self.x_norm_buf, &mut self.x_norm1_buf)?;
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
            &mut self.kv_latent_buf,
            (1, 512),
            1e-5,
        )?;

        let cap = dsa_state.token_ids.len();
        dsa_state.set_meta(
            &stream,
            DsaMeta {
                prefix: pos as i32,
                valid_rows: 1,
                key_count: (pos + 1) as i32,
                capacity: cap as i32,
                first_key: 0,
                reserved: [0, 0, 0],
            },
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.index_wk,
            &self.x_norm_buf,
            &mut self.index_k,
            (1, 128, 4096),
        )?;
        run_linear_bf16(
            &stream,
            blas,
            &self.index_gate,
            &self.x_norm_buf,
            &mut self.index_g,
            (1, 128, 4096),
        )?;
        self.index_norm.layer_norm(
            &stream,
            heads(&self.index_k, &stream, 0, 1, 1, 128)?,
            view(&self.index_norm_w, &stream, 0, [128], [1])?,
            view(&self.index_norm_b, &stream, 0, [128], [1])?,
            &mut TensorMut(heads(&self.index_k, &stream, 0, 1, 1, 128)?),
            1e-6,
        )?;
        stream.memcpy_dtod(
            &self.index_k,
            &mut self.index_outputs.packed.slice_mut(..128),
        )?;
        stream.memcpy_dtod(
            &self.index_g,
            &mut self.index_outputs.packed.slice_mut(128..256),
        )?;
        self.dsa_plan.append(
            dsa_state,
            &DsaAppendInput {
                latent: matrix(&self.kv_latent_buf, &stream, 0, 1, 512)?,
                packed: matrix(&self.index_outputs.packed, &stream, 0, 1, 257)?,
            },
            &stream,
        )?;
        self.dsa_plan
            .pool(dsa_state, &self.index_input()?, &stream)?;
        Ok(())
    }

    fn forward_layer45_core(
        &mut self,
        dsa_state: &mut DsaState,
        pos: usize,
        swap_eh: bool,
        _router_bias_control: bool,
        blas: &Blas,
        head_w: Option<&CudaSlice<bf16>>,
    ) -> Result<()> {
        let stream = self.stream.clone();
        self.write_kv_entry(dsa_state, pos, swap_eh, blas)?;

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

        if let Some(indices) = crate::mtp::glm_mtp_dense_indices(pos) {
            stream.memcpy_htod(&indices, &mut self.index_outputs.indices)?;
        } else {
            run_linear_bf16(
                &stream,
                blas,
                &self.index_wq,
                &self.q_a_norm_buf,
                &mut self.index_q,
                (1, 4096, 1536),
            )?;
            run_linear_bf16(
                &stream,
                blas,
                &self.index_weights,
                &self.x_norm_buf,
                &mut self.index_head_weights,
                (1, 32, 4096),
            )?;
            self.dsa_plan.index(
                dsa_state,
                &self.index_input()?,
                &DsaIndexOutput {
                    selected: TensorMut(view(
                        &self.index_outputs.selected,
                        &stream,
                        0,
                        [1, 512],
                        [512, 1],
                    )?),
                    scores: TensorMut(view(
                        &self.index_outputs.scores,
                        &stream,
                        0,
                        [1, 512],
                        [512, 1],
                    )?),
                    margins: TensorMut(view(&self.index_outputs.margins, &stream, 0, [1], [1])?),
                    indices: TensorMut(view(
                        &self.index_outputs.indices,
                        &stream,
                        0,
                        [1, 2051],
                        [2051, 1],
                    )?),
                },
                &stream,
            )?;
        }

        let (q_ptr, _) = self.q_b_buf.device_ptr(&stream);
        let (kv_b_ptr, _) = self.kv_b_w.device_ptr(&stream);
        let (idx_ptr, _) = self.index_outputs.indices.device_ptr(&stream);
        let (out_ptr, _) = self.attn_out_buf.device_ptr(&stream);

        let q_tensor = unsafe { Tensor::new(q_ptr, [1, 64, 256], [16384, 256, 1], 16384 * 2)? };
        let kv_b_tensor =
            unsafe { Tensor::new(kv_b_ptr, [32768, 512], [512, 1], 32768 * 512 * 2)? };
        let idx_tensor = unsafe { Tensor::new(idx_ptr, [1, 2051], [2051, 1], 2051 * 4)? };
        let mut out_tensor = unsafe {
            TensorMut(Tensor::new(
                out_ptr,
                [1, 64, 256],
                [16384, 256, 1],
                16384 * 2,
            )?)
        };

        self.dsa_plan.attention(
            dsa_state,
            &DsaAttentionInput {
                q: q_tensor,
                kv_b: Some(kv_b_tensor),
                indices: idx_tensor,
                sinks: None,
                out: &mut out_tensor,
            },
            &stream,
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
        stream.memcpy_dtod(&self.x_buf, &mut self.x_curr_buf)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.post_norm_w,
            &mut self.x_norm_buf,
            (1, 4096),
            1e-5,
        )?;

        let (x_norm_ptr, _) = self.x_norm_buf.device_ptr(&stream);
        let (moe_out_ptr, _) = self.moe_out_buf.device_ptr(&stream);
        let x_norm_tensor = unsafe { Tensor::new(x_norm_ptr, [1, 4096], [4096, 1], 4096 * 2)? };
        let moe_out_tensor =
            unsafe { TensorMut(Tensor::new(moe_out_ptr, [1, 4096], [4096, 1], 4096 * 2)?) };

        self.native_moe.enqueue_route(&stream, x_norm_tensor)?;
        let (ids, weights) = self.native_moe.read_route(&stream)?;
        self.last_route = Some((ids.clone(), weights));
        self.last_staged_experts = self.stage_routed_experts(&ids)?;
        self.native_moe
            .run(&stream, x_norm_tensor, &moe_out_tensor)?;
        run_add(&stream, &self.add, &mut self.x_buf, &self.moe_out_buf, 4096)?;

        run_rms_norm(
            &stream,
            &self.rms_norm,
            &self.x_buf,
            &self.head_norm_w,
            &mut self.head_norm_buf,
            (1, 4096),
            1e-5,
        )?;

        let target_head = match head_w {
            Some(h) => h,
            None => self
                .standalone_head
                .as_ref()
                .context("standalone head missing")?,
        };
        run_linear_bf16_to_f32(
            &stream,
            blas,
            target_head,
            &self.head_norm_buf,
            &mut self.logits_buf,
            (1, 154880, 4096),
        )?;

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn check_long_index(&mut self, blas: &Blas) -> Result<()> {
        let stream = self.stream.clone();
        let mut state = self
            ._standalone_dsa
            .take()
            .context("standalone DSA missing")?;
        let pos = 2052usize;
        ensure!(state.token_ids.len() > pos, "long-index fixture capacity");
        state.clear(&stream)?;
        let token_ids: Vec<i32> = (0..pos as i32).collect();
        let pool_ids: Vec<i32> = (0..(pos / 4) as i32).collect();
        stream.memcpy_htod(&token_ids, &mut state.token_ids.slice_mut(..pos))?;
        stream.memcpy_htod(&pool_ids, &mut state.pool_ids.slice_mut(..pool_ids.len()))?;
        let mut packed = vec![bf16::ZERO; pos * 257];
        for row in packed.as_chunks_mut::<257>().0 {
            row[256] = bf16::ONE;
        }
        stream.memcpy_htod(&packed, &mut state.packed.slice_mut(..packed.len()))?;
        let keys: Vec<bf16> = (0..pool_ids.len() * 128)
            .map(|i| bf16::from_f32(((i * 37 + 13) % 1009) as f32 / 512.0 - 1.0))
            .collect();
        stream.memcpy_htod(&keys, &mut state.pool_keys.slice_mut(..keys.len()))?;
        self.forward_layer45_core(&mut state, pos, false, false, blas, None)?;
        let indices = stream.clone_dtoh(&self.index_outputs.indices)?;
        let selected = stream.clone_dtoh(&self.index_outputs.selected)?;
        let mut expected: Vec<i32> = selected
            .iter()
            .flat_map(|pool| (0..4).map(move |i| pool * 4 + i))
            .collect();
        expected.push(pos as i32);
        expected.resize(2051, -1);
        ensure!(indices == expected, "DEFECT MTP top-k plus tail expansion");
        ensure!(
            indices.contains(&(pos as i32)),
            "DEFECT MTP current tail excluded"
        );
        let unique: std::collections::BTreeSet<_> =
            indices.iter().copied().filter(|i| *i >= 0).collect();
        ensure!(
            unique.len() == 2049 && unique.iter().all(|i| *i <= pos as i32),
            "DEFECT MTP selected index bounds or duplicates"
        );
        let truncated: Vec<i32> = (0..2051).collect();
        ensure!(
            indices != truncated && !truncated.contains(&(pos as i32)),
            "Runner FAIL truncated-index control"
        );
        println!(
            "PASS GLM_MTP_LONG_INDEX position={pos} selected_pools={} unique_keys={} current_tail=true truncated_control=true",
            selected.len(),
            unique.len()
        );
        self._standalone_dsa = Some(state);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn populate_history(
        &mut self,
        history_tokens: &[u32],
        history_hidden: &[bf16],
        dsa_state: &mut DsaState,
        blas: &Blas,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let p = history_tokens.len();
        anyhow::ensure!(
            history_hidden.len() == p * 4096,
            "history hidden length mismatch: expected {}, got {}",
            p * 4096,
            history_hidden.len()
        );

        dsa_state.clear(&stream)?;

        for t in 0..p.saturating_sub(1) {
            let tok = history_tokens[t] as usize;
            let tok_slice = self
                ._standalone_embed
                .as_ref()
                .context("standalone embed missing")?
                .slice(tok * 4096..(tok + 1) * 4096);
            stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;

            let h_slice = &history_hidden[t * 4096..(t + 1) * 4096];
            stream.memcpy_htod(h_slice, &mut self.target_hidden_buf)?;
            self.write_kv_entry(dsa_state, t, false, blas)?;
        }

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn forward_with_history(
        &mut self,
        history_tokens: &[u32],
        history_hidden: &[bf16],
        swap_eh: bool,
        router_bias_control: bool,
        blas: &Blas,
    ) -> Result<Vec<f32>> {
        let p = history_tokens.len();
        let last_token = history_tokens[p - 1] as usize;
        let last_hidden = &history_hidden[(p - 1) * 4096..p * 4096];

        let stream = self.stream.clone();
        let mut dsa = self
            ._standalone_dsa
            .take()
            .context("standalone DSA state missing")?;
        self.populate_history(history_tokens, history_hidden, &mut dsa, blas)?;

        let tok_slice = self
            ._standalone_embed
            .as_ref()
            .context("standalone embed missing")?
            .slice(last_token * 4096..(last_token + 1) * 4096);
        stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;
        stream.memcpy_htod(last_hidden, &mut self.target_hidden_buf)?;

        self.forward_layer45_core(&mut dsa, p - 1, swap_eh, router_bias_control, blas, None)?;

        self._standalone_dsa = Some(dsa);
        Ok(self.stream.clone_dtoh(&self.logits_buf)?)
    }

    #[cfg(test)]
    pub(crate) fn last_route(&self) -> Option<&(Vec<i32>, Vec<f32>)> {
        self.last_route.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn forward_graph(
        &mut self,
        history_tokens: &[u32],
        history_hidden: &[bf16],
        swap_eh: bool,
        blas: &Blas,
    ) -> Result<Vec<f32>> {
        self.forward_with_history(history_tokens, history_hidden, swap_eh, false, blas)
    }

    #[cfg(test)]
    pub(crate) fn forward(
        &mut self,
        history_tokens: &[u32],
        history_hidden: &[bf16],
        swap_eh: bool,
        blas: &Blas,
    ) -> Result<Vec<f32>> {
        self.forward_with_history(history_tokens, history_hidden, swap_eh, false, blas)
    }

    #[cfg(test)]
    pub(crate) fn read_stage_taps(&self) -> Result<GlmMtpStageTaps> {
        let stream = &self.stream;
        let (ids, weights) = self.last_route.as_ref().cloned().unwrap_or_default();
        let router_logits = self.native_moe.read_logits(stream)?;
        Ok(GlmMtpStageTaps {
            enorm: stream.clone_dtoh(&self.enorm_buf)?,
            hnorm: stream.clone_dtoh(&self.hnorm_buf)?,
            eh_in: stream.clone_dtoh(&self.eh_buf)?,
            x_init: stream.clone_dtoh(&self.x_init_buf)?,
            x_norm1: stream.clone_dtoh(&self.x_norm1_buf)?,
            kv_latent: stream.clone_dtoh(&self.kv_latent_buf)?,
            q_a: stream.clone_dtoh(&self.q_a_norm_buf)?,
            q_out: stream.clone_dtoh(&self.q_b_buf)?,
            attn_out: stream.clone_dtoh(&self.attn_out_buf)?,
            o_out: stream.clone_dtoh(&self.o_buf)?,
            x_curr: stream.clone_dtoh(&self.x_curr_buf)?,
            x_norm2: stream.clone_dtoh(&self.x_norm_buf)?,
            router_logits,
            routing_ids: ids,
            routing_weights: weights,
            moe_out: stream.clone_dtoh(&self.moe_out_buf)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn test_route_custom_norm(
        &mut self,
        norm: &[bf16],
    ) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>)> {
        let stream = self.stream.clone();
        stream.memcpy_htod(norm, &mut self.x_norm_buf)?;
        let (x_norm_ptr, _) = self.x_norm_buf.device_ptr(&stream);
        let x_norm_tensor = unsafe { Tensor::new(x_norm_ptr, [1, 4096], [4096, 1], 4096 * 2)? };
        self.native_moe.enqueue_route(&stream, x_norm_tensor)?;
        let logits = self.native_moe.read_logits(&stream)?;
        let (ids, weights) = self.native_moe.read_route(&stream)?;
        Ok((logits, ids, weights))
    }

    // Commits one validated (token, post_norm_hidden) pair at 0-based MTP cache index `pos` (model definition).
    pub(crate) fn commit_kv_device(
        &mut self,
        token: u32,
        target_hidden: &CudaSlice<bf16>,
        hidden_row: usize,
        shared: GlmMtpShared<'_>,
        dsa_state: &mut DsaState,
        pos: usize,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let tok = token as usize;
        let tok_slice = shared.embedding.slice(tok * 4096..(tok + 1) * 4096);
        stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;
        let hidden_slice = target_hidden.slice(hidden_row * 4096..(hidden_row + 1) * 4096);
        stream.memcpy_dtod(&hidden_slice, &mut self.target_hidden_buf)?;
        self.write_kv_entry(dsa_state, pos, false, shared.blas)
    }

    pub(crate) fn step_draft_device(
        &mut self,
        anchor_token: u32,
        target_hidden: &CudaSlice<bf16>,
        hidden_row: usize,
        shared: GlmMtpShared<'_>,
        dsa_state: &mut DsaState,
        draft_pos: usize,
    ) -> Result<(u32, Vec<f32>, usize)> {
        let stream = self.stream.clone();
        let tok = anchor_token as usize;
        let tok_slice = shared.embedding.slice(tok * 4096..(tok + 1) * 4096);
        stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;
        let hidden_slice = target_hidden.slice(hidden_row * 4096..(hidden_row + 1) * 4096);
        stream.memcpy_dtod(&hidden_slice, &mut self.target_hidden_buf)?;

        self.forward_layer45_core(
            dsa_state,
            draft_pos,
            false,
            false,
            shared.blas,
            Some(shared.head),
        )?;

        let logits: Vec<f32> = stream.clone_dtoh(&self.logits_buf)?;
        let mut best_id = 0u32;
        let mut best_score = -f32::INFINITY;
        for (i, &s) in logits.iter().enumerate() {
            if s > best_score {
                best_score = s;
                best_id = i as u32;
            }
        }
        Ok((best_id, logits, self.last_staged_experts))
    }
}
