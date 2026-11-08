//! Native joint schema head weights for clef-flash: the published BF16
//! tensors of joint_head.safetensors as context views, dims derived from
//! joint_head_config.json. Forward stages compose shared ops.

use crate::backend::cuda::contracts::Tensor;
use crate::backend::gpu::{CudaStream, DevicePtr};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

pub struct RoutingWeights {
    pub query_norm_w: Tensor<bf16, 1>,
    pub query_norm_b: Tensor<bf16, 1>,
    pub memory_norm_w: Tensor<bf16, 1>,
    pub memory_norm_b: Tensor<bf16, 1>,
    pub attn_in_proj: Tensor<bf16, 2>,
    pub attn_in_proj_b: Tensor<bf16, 1>,
    pub attn_out_proj: Tensor<bf16, 2>,
    pub attn_out_proj_b: Tensor<bf16, 1>,
    pub ff0_w: Tensor<bf16, 2>,
    pub ff0_b: Tensor<bf16, 1>,
    pub ff3_w: Tensor<bf16, 2>,
    pub ff3_b: Tensor<bf16, 1>,
    pub ff_norm_w: Tensor<bf16, 1>,
    pub ff_norm_b: Tensor<bf16, 1>,
}

pub struct DecoderWeights {
    pub self_in_proj: Tensor<bf16, 2>,
    pub self_in_proj_b: Tensor<bf16, 1>,
    pub self_out_proj: Tensor<bf16, 2>,
    pub self_out_proj_b: Tensor<bf16, 1>,
    pub cross_in_proj: Tensor<bf16, 2>,
    pub cross_in_proj_b: Tensor<bf16, 1>,
    pub cross_out_proj: Tensor<bf16, 2>,
    pub cross_out_proj_b: Tensor<bf16, 1>,
    pub norm1_w: Tensor<bf16, 1>,
    pub norm1_b: Tensor<bf16, 1>,
    pub norm2_w: Tensor<bf16, 1>,
    pub norm2_b: Tensor<bf16, 1>,
    pub norm3_w: Tensor<bf16, 1>,
    pub norm3_b: Tensor<bf16, 1>,
    pub linear1_w: Tensor<bf16, 2>,
    pub linear1_b: Tensor<bf16, 1>,
    pub linear2_w: Tensor<bf16, 2>,
    pub linear2_b: Tensor<bf16, 1>,
}

pub struct JointWeights {
    pub hidden_norm_w: Tensor<bf16, 1>,
    pub hidden_norm_b: Tensor<bf16, 1>,
    pub memory_proj: Tensor<bf16, 2>,
    pub question_proj: Tensor<bf16, 2>,
    pub option_question_proj: Tensor<bf16, 2>,
    pub global_proj: Tensor<bf16, 2>,
    pub option_context_proj: Tensor<bf16, 2>,
    pub option_lexical_proj: Tensor<bf16, 2>,
    pub type_embedding: Tensor<bf16, 2>,
    pub field_norm_w: Tensor<bf16, 1>,
    pub field_norm_b: Tensor<bf16, 1>,
    pub option_norm_w: Tensor<bf16, 1>,
    pub option_norm_b: Tensor<bf16, 1>,
    pub option_summary_norm_w: Tensor<bf16, 1>,
    pub option_summary_norm_b: Tensor<bf16, 1>,
    pub scorer0_w: Tensor<bf16, 2>,
    pub scorer0_b: Tensor<bf16, 1>,
    pub scorer3_w: Tensor<bf16, 2>,
    pub scorer3_b: Tensor<bf16, 1>,
    pub prior_logit_scale: f32,
    pub joint_logit_scale: f32,
    pub residual_gate: f32,
    pub routing: Vec<RoutingWeights>,
    pub decoder: Vec<DecoderWeights>,
}

/// Uploads one BF16 [rows, cols] tensor from the published file.
pub fn upload2(
    view: &safetensors::SafeTensors,
    name: &str,
    shape: [usize; 2],
    stream: &Arc<CudaStream>,
) -> Result<Tensor<bf16, 2>> {
    let t = view.tensor(name)?;
    ensure!(
        t.dtype() == safetensors::Dtype::BF16 && t.shape() == shape,
        "joint tensor {name} dtype/shape mismatch: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let host: Vec<bf16> = t
        .data()
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut dev = stream.alloc_zeros::<bf16>(shape[0] * shape[1])?;
    stream.memcpy_htod(&host, &mut dev)?;
    let (ptr, _) = dev.device_ptr(stream);
    unsafe {
        Tensor::new(
            ptr,
            [shape[0] as u32, shape[1] as u32],
            [shape[1] as u64, 1],
            (shape[0] * shape[1]) as u64,
        )
    }
}

/// Uploads one BF16 [n] tensor from the published file.
pub fn upload1(
    view: &safetensors::SafeTensors,
    name: &str,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<Tensor<bf16, 1>> {
    let t = view.tensor(name)?;
    ensure!(
        t.dtype() == safetensors::Dtype::BF16 && t.shape() == [n],
        "joint tensor {name} dtype/shape mismatch: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let host: Vec<bf16> = t
        .data()
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut dev = stream.alloc_zeros::<bf16>(n)?;
    stream.memcpy_htod(&host, &mut dev)?;
    let (ptr, _) = dev.device_ptr(stream);
    unsafe { Tensor::new(ptr, [n as u32], [1], n as u64) }
}

/// Reads one BF16 scalar from the published file.
pub fn scalar(view: &safetensors::SafeTensors, name: &str) -> Result<f32> {
    let t = view.tensor(name)?;
    let d = t.data();
    Ok(bf16::from_bits(u16::from_le_bytes([d[0], d[1]])).to_f32())
}

/// Loads the published joint-head weights as context views on `stream`.
pub fn load_joint_weights(
    model_dir: &std::path::Path,
    stream: &Arc<CudaStream>,
) -> Result<JointWeights> {
    let bytes = std::fs::read(model_dir.join("joint_head.safetensors"))
        .context("open joint_head.safetensors")?;
    let view = safetensors::SafeTensors::deserialize(&bytes)?;
    let routing = (0..2)
        .map(|i| {
            Ok(RoutingWeights {
                query_norm_w: upload1(
                    &view,
                    &format!("evidence_layers.{i}.query_norm.weight"),
                    1024,
                    stream,
                )?,
                query_norm_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.query_norm.bias"),
                    1024,
                    stream,
                )?,
                memory_norm_w: upload1(
                    &view,
                    &format!("evidence_layers.{i}.memory_norm.weight"),
                    1024,
                    stream,
                )?,
                memory_norm_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.memory_norm.bias"),
                    1024,
                    stream,
                )?,
                attn_in_proj: upload2(
                    &view,
                    &format!("evidence_layers.{i}.attention.in_proj_weight"),
                    [3072, 1024],
                    stream,
                )?,
                attn_in_proj_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.attention.in_proj_bias"),
                    3072,
                    stream,
                )?,
                attn_out_proj: upload2(
                    &view,
                    &format!("evidence_layers.{i}.attention.out_proj.weight"),
                    [1024, 1024],
                    stream,
                )?,
                attn_out_proj_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.attention.out_proj.bias"),
                    1024,
                    stream,
                )?,
                ff0_w: upload2(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.0.weight"),
                    [4096, 1024],
                    stream,
                )?,
                ff0_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.0.bias"),
                    4096,
                    stream,
                )?,
                ff3_w: upload2(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.3.weight"),
                    [1024, 4096],
                    stream,
                )?,
                ff3_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.3.bias"),
                    1024,
                    stream,
                )?,
                ff_norm_w: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward_norm.weight"),
                    1024,
                    stream,
                )?,
                ff_norm_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward_norm.bias"),
                    1024,
                    stream,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let decoder = (0..4)
        .map(|i| {
            Ok(DecoderWeights {
                self_in_proj: upload2(
                    &view,
                    &format!("layers.{i}.self_attn.in_proj_weight"),
                    [3072, 1024],
                    stream,
                )?,
                self_in_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.self_attn.in_proj_bias"),
                    3072,
                    stream,
                )?,
                self_out_proj: upload2(
                    &view,
                    &format!("layers.{i}.self_attn.out_proj.weight"),
                    [1024, 1024],
                    stream,
                )?,
                self_out_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.self_attn.out_proj.bias"),
                    1024,
                    stream,
                )?,
                cross_in_proj: upload2(
                    &view,
                    &format!("layers.{i}.multihead_attn.in_proj_weight"),
                    [3072, 1024],
                    stream,
                )?,
                cross_in_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.multihead_attn.in_proj_bias"),
                    3072,
                    stream,
                )?,
                cross_out_proj: upload2(
                    &view,
                    &format!("layers.{i}.multihead_attn.out_proj.weight"),
                    [1024, 1024],
                    stream,
                )?,
                cross_out_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.multihead_attn.out_proj.bias"),
                    1024,
                    stream,
                )?,
                norm1_w: upload1(&view, &format!("layers.{i}.norm1.weight"), 1024, stream)?,
                norm1_b: upload1(&view, &format!("layers.{i}.norm1.bias"), 1024, stream)?,
                norm2_w: upload1(&view, &format!("layers.{i}.norm2.weight"), 1024, stream)?,
                norm2_b: upload1(&view, &format!("layers.{i}.norm2.bias"), 1024, stream)?,
                norm3_w: upload1(&view, &format!("layers.{i}.norm3.weight"), 1024, stream)?,
                norm3_b: upload1(&view, &format!("layers.{i}.norm3.bias"), 1024, stream)?,
                linear1_w: upload2(
                    &view,
                    &format!("layers.{i}.linear1.weight"),
                    [4096, 1024],
                    stream,
                )?,
                linear1_b: upload1(&view, &format!("layers.{i}.linear1.bias"), 4096, stream)?,
                linear2_w: upload2(
                    &view,
                    &format!("layers.{i}.linear2.weight"),
                    [1024, 4096],
                    stream,
                )?,
                linear2_b: upload1(&view, &format!("layers.{i}.linear2.bias"), 1024, stream)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(JointWeights {
        hidden_norm_w: upload1(&view, "hidden_norm.weight", 4096, stream)?,
        hidden_norm_b: upload1(&view, "hidden_norm.bias", 4096, stream)?,
        memory_proj: upload2(&view, "memory_projection.weight", [1024, 4096], stream)?,
        question_proj: upload2(&view, "question_projection.weight", [1024, 4096], stream)?,
        option_question_proj: upload2(
            &view,
            "option_question_projection.weight",
            [1024, 4096],
            stream,
        )?,
        global_proj: upload2(&view, "global_projection.weight", [1024, 4096], stream)?,
        option_context_proj: upload2(
            &view,
            "option_context_projection.weight",
            [1024, 4096],
            stream,
        )?,
        option_lexical_proj: upload2(
            &view,
            "option_lexical_projection.weight",
            [1024, 4096],
            stream,
        )?,
        type_embedding: upload2(&view, "type_embedding.weight", [3, 1024], stream)?,
        field_norm_w: upload1(&view, "field_norm.weight", 1024, stream)?,
        field_norm_b: upload1(&view, "field_norm.bias", 1024, stream)?,
        option_norm_w: upload1(&view, "option_norm.weight", 1024, stream)?,
        option_norm_b: upload1(&view, "option_norm.bias", 1024, stream)?,
        option_summary_norm_w: upload1(&view, "option_summary_norm.weight", 1024, stream)?,
        option_summary_norm_b: upload1(&view, "option_summary_norm.bias", 1024, stream)?,
        scorer0_w: upload2(&view, "residual_scorer.0.weight", [1024, 4096], stream)?,
        scorer0_b: upload1(&view, "residual_scorer.0.bias", 1024, stream)?,
        scorer3_w: upload2(&view, "residual_scorer.3.weight", [1, 1024], stream)?,
        scorer3_b: upload1(&view, "residual_scorer.3.bias", 1, stream)?,
        prior_logit_scale: scalar(&view, "prior_logit_scale")?,
        joint_logit_scale: scalar(&view, "joint_logit_scale")?,
        residual_gate: scalar(&view, "residual_gate")?,
        routing,
        decoder,
    })
}
