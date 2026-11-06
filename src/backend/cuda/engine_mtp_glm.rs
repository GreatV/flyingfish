use super::*;
use crate::backend::cuda::Device;
use crate::backend::cuda::blas::Blas;
use crate::backend::cuda::contracts::{Tensor, TensorMut};
use crate::backend::cuda::cubin;
use crate::backend::cuda::dsa::{DsaAttentionInput, DsaPlan, DsaState};
use crate::backend::cuda::linear::{Fp8Ops, Fp8ScaleRef, LinearWeightRef};
use crate::backend::cuda::moe::decode::{NativeMoe, NativeMoeGeometry, NativeMoeWeights};
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
            .arg(y)
            .arg(w)
            .arg(&eps)
            .arg(&dim)
            .arg(&rows)
            .launch(crate::backend::cuda::ops::grid(
                (rows as usize).div_ceil(32) * 32,
                dim as u32,
            ))?;
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

pub(crate) fn run_fp8_linear(
    stream: &Arc<CudaStream>,
    fp8: &Fp8Ops,
    codes: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    shape: (usize, usize, usize),
) -> Result<()> {
    let (rows, output_dim, input_dim) = shape;
    let (c_ptr, _) = codes.device_ptr(stream);
    let (s_ptr, _) = scales.device_ptr(stream);
    let (x_ptr, _) = x.device_ptr(stream);
    let (y_ptr, _) = y.device_ptr(stream);

    let codes_tensor = unsafe {
        Tensor::new(
            c_ptr,
            [output_dim as u32, input_dim as u32],
            [input_dim as u64, 1],
            (output_dim * input_dim) as u64,
        )?
    };
    let scale_tensor = unsafe {
        Tensor::new(
            s_ptr,
            [
                output_dim.div_ceil(128) as u32,
                input_dim.div_ceil(128) as u32,
            ],
            [input_dim.div_ceil(128) as u64, 1],
            (output_dim.div_ceil(128) * input_dim.div_ceil(128) * 4) as u64,
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

    fp8.small_gemm(
        stream,
        codes_tensor,
        Fp8ScaleRef::F32(scale_tensor),
        Fp8ScaleLayout::Block(Fp8BlockScale::new([128, 128], safetensors::Dtype::F32).unwrap()),
        x_tensor,
        &mut y_tensor,
    )
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
    pub(crate) concat_eh: CudaFunction,
    pub(crate) add: CudaFunction,

    pub(crate) fp8: Fp8Ops,
    pub(crate) dsa_plan: DsaPlan,
    pub(crate) native_moe: NativeMoe,

    enorm_w: CudaSlice<bf16>,
    hnorm_w: CudaSlice<bf16>,
    eh_proj_w: CudaSlice<bf16>,
    input_norm_w: CudaSlice<bf16>,

    q_a_codes: CudaSlice<u8>,
    q_a_scales: CudaSlice<f32>,
    q_a_norm_w: CudaSlice<bf16>,
    q_b_codes: CudaSlice<u8>,
    q_b_scales: CudaSlice<f32>,

    kv_a_codes: CudaSlice<u8>,
    kv_a_scales: CudaSlice<f32>,
    kv_a_norm_w: CudaSlice<bf16>,
    kv_b_w: CudaSlice<bf16>,

    o_codes: CudaSlice<u8>,
    o_scales: CudaSlice<f32>,
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

    head_norm_w: CudaSlice<bf16>,

    #[cfg(test)]
    pub(crate) standalone_embed: Option<CudaSlice<bf16>>,
    pub(crate) standalone_head: Option<CudaSlice<bf16>>,

    pub(crate) token_embed_buf: CudaSlice<bf16>,
    pub(crate) enorm_buf: CudaSlice<bf16>,
    pub(crate) target_hidden_buf: CudaSlice<bf16>,
    pub(crate) hnorm_buf: CudaSlice<bf16>,
    pub(crate) eh_buf: CudaSlice<bf16>,
    pub(crate) x_buf: CudaSlice<bf16>,
    pub(crate) x_norm_buf: CudaSlice<bf16>,
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
    pub(crate) dsa_indices_buf: CudaSlice<i32>,

    pub(crate) last_route: Option<(Vec<i32>, Vec<f32>)>,

    #[cfg(test)]
    pub(crate) standalone_dsa: Option<DsaState>,
}

impl GlmMtpRuntime {
    pub(crate) fn load(model_dir: &Path, device: &Device, _blas: &Blas) -> Result<Self> {
        unsafe {
            device.ctx.disable_event_tracking();
        }
        let stream = &device.stream;

        let norm_mod = cubin::module(&device.ctx, "norm")?;
        let rms_norm = norm_mod.load_function("rms_norm")?;

        let mtp_mod = cubin::module(&device.ctx, "glm_mtp")?;
        let concat_eh = mtp_mod.load_function("glm_mtp_concat_eh")?;
        let add = mtp_mod.load_function("glm_mtp_add")?;

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

        let fp8 = Fp8Ops::new(&device.ctx)?;
        let dsa_spec = DsaSpec::glm53();
        let dsa_plan = DsaPlan::new(&device.ctx, &device.info, dsa_spec, 2048, 1)?;
        #[cfg(test)]
        let standalone_dsa = Some(DsaState::new(dsa_spec, 2048, stream)?);

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

        let q_a_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.self_attn.q_a_proj.weight",
        )?)?;
        let q_a_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.self_attn.q_a_proj.weight_scale_inv",
        )?)?;
        let q_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.q_a_layernorm.weight",
        )?)?;
        let q_b_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.self_attn.q_b_proj.weight",
        )?)?;
        let q_b_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.self_attn.q_b_proj.weight_scale_inv",
        )?)?;

        let kv_a_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight",
        )?)?;
        let kv_a_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
        )?)?;
        let kv_a_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_a_layernorm.weight",
        )?)?;
        let kv_b_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.self_attn.kv_b_proj.weight",
        )?)?;

        let o_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.self_attn.o_proj.weight",
        )?)?;
        let o_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.self_attn.o_proj.weight_scale_inv",
        )?)?;
        let post_norm_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.post_attention_layernorm.weight",
        )?)?;

        let router_w = stream.clone_htod(&get_st_bf16(
            "model.language_model.layers.45.mlp.gate.weight",
        )?)?;
        let router_b = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.gate.e_score_correction_bias",
        )?)?;

        let sh_gate_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.mlp.shared_experts.gate_proj.weight",
        )?)?;
        let sh_gate_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.shared_experts.gate_proj.weight_scale_inv",
        )?)?;
        let sh_up_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.mlp.shared_experts.up_proj.weight",
        )?)?;
        let sh_up_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.shared_experts.up_proj.weight_scale_inv",
        )?)?;
        let sh_down_codes = stream.clone_htod(&get_st_u8(
            "model.language_model.layers.45.mlp.shared_experts.down_proj.weight",
        )?)?;
        let sh_down_scales = stream.clone_htod(&get_st_f32(
            "model.language_model.layers.45.mlp.shared_experts.down_proj.weight_scale_inv",
        )?)?;

        let mut experts = Vec::with_capacity(288);
        for i in 0..288 {
            let gate_codes = get_st_u8(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.gate_proj.weight"
            ))?;
            let gate_scales = get_st_f32(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.gate_proj.weight_scale_inv"
            ))?;
            let up_codes = get_st_u8(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.up_proj.weight"
            ))?;
            let up_scales = get_st_f32(&format!(
                "model.language_model.layers.45.mlp.experts.{i}.up_proj.weight_scale_inv"
            ))?;
            let down_codes = get_st_u8(&format!(
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
            stream.memcpy_htod(&exp.gate_codes, &mut slot_gate_codes[k])?;
            stream.memcpy_htod(&exp.gate_scales, &mut slot_gate_scales[k])?;
            stream.memcpy_htod(&exp.up_codes, &mut slot_up_codes[k])?;
            stream.memcpy_htod(&exp.up_scales, &mut slot_up_scales[k])?;
            stream.memcpy_htod(&exp.down_codes, &mut slot_down_codes[k])?;
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

        #[cfg(test)]
        let standalone_embed = match get_st_bf16("model.language_model.embed_tokens.weight") {
            Ok(v) => Some(stream.clone_htod(&v)?),
            Err(_) => None,
        };
        let standalone_head = match get_st_bf16("lm_head.weight") {
            Ok(v) => Some(stream.clone_htod(&v)?),
            Err(_) => None,
        };

        Ok(Self {
            stream: stream.clone(),
            rms_norm,
            concat_eh,
            add,
            fp8,
            dsa_plan,
            native_moe,
            enorm_w,
            hnorm_w,
            eh_proj_w,
            input_norm_w,
            q_a_codes,
            q_a_scales,
            q_a_norm_w,
            q_b_codes,
            q_b_scales,
            kv_a_codes,
            kv_a_scales,
            kv_a_norm_w,
            kv_b_w,
            o_codes,
            o_scales,
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
            head_norm_w,
            #[cfg(test)]
            standalone_embed,
            standalone_head,
            token_embed_buf: stream.alloc_zeros::<bf16>(4096)?,
            enorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            target_hidden_buf: stream.alloc_zeros::<bf16>(4096)?,
            hnorm_buf: stream.alloc_zeros::<bf16>(4096)?,
            eh_buf: stream.alloc_zeros::<bf16>(8192)?,
            x_buf: stream.alloc_zeros::<bf16>(4096)?,
            x_norm_buf: stream.alloc_zeros::<bf16>(4096)?,
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
            dsa_indices_buf: stream.alloc_zeros::<i32>(2051)?,
            last_route: None,
            #[cfg(test)]
            standalone_dsa,
        })
    }

    fn stage_routed_experts(&mut self, ids: &[i32]) -> Result<()> {
        let stream = self.stream.clone();
        let mut updated = false;
        for (k, &expert_id_i32) in ids.iter().take(8).enumerate() {
            let expert_id = expert_id_i32 as usize;
            if self.staged_slots[k] != Some(expert_id) {
                if let Some(old_id) = self.staged_slots[k] {
                    self.valid_host[old_id] = 0;
                }
                let exp = &self.experts[expert_id];
                stream.memcpy_htod(&exp.gate_codes, &mut self.slot_gate_codes[k])?;
                stream.memcpy_htod(&exp.gate_scales, &mut self.slot_gate_scales[k])?;
                stream.memcpy_htod(&exp.up_codes, &mut self.slot_up_codes[k])?;
                stream.memcpy_htod(&exp.up_scales, &mut self.slot_up_scales[k])?;
                stream.memcpy_htod(&exp.down_codes, &mut self.slot_down_codes[k])?;
                stream.memcpy_htod(&exp.down_scales, &mut self.slot_down_scales[k])?;

                let (gc_ptr, _) = self.slot_gate_codes[k].device_ptr(&stream);
                let (gs_ptr, _) = self.slot_gate_scales[k].device_ptr(&stream);
                let (uc_ptr, _) = self.slot_up_codes[k].device_ptr(&stream);
                let (us_ptr, _) = self.slot_up_scales[k].device_ptr(&stream);
                let (dc_ptr, _) = self.slot_down_codes[k].device_ptr(&stream);
                let (ds_ptr, _) = self.slot_down_scales[k].device_ptr(&stream);

                self.slots_host[expert_id * 6] = gc_ptr;
                self.slots_host[expert_id * 6 + 1] = gs_ptr;
                self.slots_host[expert_id * 6 + 2] = uc_ptr;
                self.slots_host[expert_id * 6 + 3] = us_ptr;
                self.slots_host[expert_id * 6 + 4] = dc_ptr;
                self.slots_host[expert_id * 6 + 5] = ds_ptr;
                self.valid_host[expert_id] = 1;

                self.staged_slots[k] = Some(expert_id);
                updated = true;
            }
        }
        if updated {
            stream.memcpy_htod(&self.slots_host, &mut self.slots_buf)?;
            stream.memcpy_htod(&self.valid_host, &mut self.valid_buf)?;
        }
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

        run_fp8_linear(
            &stream,
            &self.fp8,
            &self.kv_a_codes,
            &self.kv_a_scales,
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
        let physical = pos % cap;
        stream.memcpy_dtod(
            &self.kv_latent_buf,
            &mut dsa_state
                .latent
                .slice_mut(physical * 512..(physical + 1) * 512),
        )?;
        stream.memcpy_htod(
            &[pos as i32],
            &mut dsa_state.token_ids.slice_mut(physical..physical + 1),
        )?;
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

        run_fp8_linear(
            &stream,
            &self.fp8,
            &self.q_a_codes,
            &self.q_a_scales,
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
        run_fp8_linear(
            &stream,
            &self.fp8,
            &self.q_b_codes,
            &self.q_b_scales,
            &self.q_a_norm_buf,
            &mut self.q_b_buf,
            (1, 16384, 1536),
        )?;

        let mut indices = vec![-1i32; 2051];
        for (i, item) in indices.iter_mut().take(pos.min(2050) + 1).enumerate() {
            *item = i as i32;
        }
        stream.memcpy_htod(&indices, &mut self.dsa_indices_buf)?;

        let (q_ptr, _) = self.q_b_buf.device_ptr(&stream);
        let (kv_b_ptr, _) = self.kv_b_w.device_ptr(&stream);
        let (idx_ptr, _) = self.dsa_indices_buf.device_ptr(&stream);
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

        run_fp8_linear(
            &stream,
            &self.fp8,
            &self.o_codes,
            &self.o_scales,
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

        let (x_norm_ptr, _) = self.x_norm_buf.device_ptr(&stream);
        let (moe_out_ptr, _) = self.moe_out_buf.device_ptr(&stream);
        let x_norm_tensor = unsafe { Tensor::new(x_norm_ptr, [1, 4096], [4096, 1], 4096 * 2)? };
        let moe_out_tensor =
            unsafe { TensorMut(Tensor::new(moe_out_ptr, [1, 4096], [4096, 1], 4096 * 2)?) };

        self.native_moe.enqueue_route(&stream, x_norm_tensor)?;
        let (ids, weights) = self.native_moe.read_route(&stream)?;
        self.last_route = Some((ids.clone(), weights));
        self.stage_routed_experts(&ids)?;
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

        dsa_state.reset(&stream)?;

        for t in 0..p.saturating_sub(1) {
            let tok = history_tokens[t] as usize;
            let tok_slice = self
                .standalone_embed
                .as_ref()
                .context("standalone embed missing")?
                .slice(tok * 4096..(tok + 1) * 4096);
            stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;

            let h_slice = &history_hidden[t * 4096..(t + 1) * 4096];
            stream.memcpy_htod(h_slice, &mut self.target_hidden_buf)?;

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
            run_fp8_linear(
                &stream,
                &self.fp8,
                &self.kv_a_codes,
                &self.kv_a_scales,
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
            let physical = t % cap;
            stream.memcpy_dtod(
                &self.kv_latent_buf,
                &mut dsa_state
                    .latent
                    .slice_mut(physical * 512..(physical + 1) * 512),
            )?;
            stream.memcpy_htod(
                &[t as i32],
                &mut dsa_state.token_ids.slice_mut(physical..physical + 1),
            )?;
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
            .standalone_dsa
            .take()
            .context("standalone DSA state missing")?;
        self.populate_history(history_tokens, history_hidden, &mut dsa, blas)?;

        let tok_slice = self
            .standalone_embed
            .as_ref()
            .context("standalone embed missing")?
            .slice(last_token * 4096..(last_token + 1) * 4096);
        stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;
        stream.memcpy_htod(last_hidden, &mut self.target_hidden_buf)?;

        self.forward_layer45_core(&mut dsa, p - 1, swap_eh, router_bias_control, blas, None)?;

        self.standalone_dsa = Some(dsa);
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_draft_device(
        &mut self,
        anchor_token: u32,
        target_hidden: &CudaSlice<bf16>,
        _force_eager_draft: bool,
        blas: &Blas,
        embedding: &CudaSlice<bf16>,
        head: &CudaSlice<bf16>,
        dsa_state: &mut DsaState,
        start_pos: usize,
    ) -> Result<(u32, Vec<f32>)> {
        let stream = self.stream.clone();
        let tok = anchor_token as usize;
        let tok_slice = embedding.slice(tok * 4096..(tok + 1) * 4096);
        stream.memcpy_dtod(&tok_slice, &mut self.token_embed_buf)?;
        stream.memcpy_dtod(target_hidden, &mut self.target_hidden_buf)?;

        self.forward_layer45_core(dsa_state, start_pos, false, false, blas, Some(head))?;

        let logits: Vec<f32> = stream.clone_dtoh(&self.logits_buf)?;
        let mut best_id = 0u32;
        let mut best_score = -f32::INFINITY;
        for (i, &s) in logits.iter().enumerate() {
            if s > best_score {
                best_score = s;
                best_id = i as u32;
            }
        }
        Ok((best_id, logits))
    }
}
