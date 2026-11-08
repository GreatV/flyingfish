//! Native joint schema head weights for clef-flash: the published BF16
//! tensors of joint_head.safetensors as context views, dims derived from
//! joint_head_config.json. Forward stages compose shared ops.

use crate::backend::gpu::CudaSlice;
use crate::backend::gpu::CudaStream;
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

pub struct RoutingWeights {
    pub query_norm_w: Arc<CudaSlice<bf16>>,
    pub query_norm_b: Arc<CudaSlice<bf16>>,
    pub memory_norm_w: Arc<CudaSlice<bf16>>,
    pub memory_norm_b: Arc<CudaSlice<bf16>>,
    pub attn_in_proj: Arc<CudaSlice<bf16>>,
    pub attn_in_proj_b: Arc<CudaSlice<bf16>>,
    pub attn_out_proj: Arc<CudaSlice<bf16>>,
    pub attn_out_proj_b: Arc<CudaSlice<bf16>>,
    pub ff0_w: Arc<CudaSlice<bf16>>,
    pub ff0_b: Arc<CudaSlice<bf16>>,
    pub ff3_w: Arc<CudaSlice<bf16>>,
    pub ff3_b: Arc<CudaSlice<bf16>>,
    pub ff_norm_w: Arc<CudaSlice<bf16>>,
    pub ff_norm_b: Arc<CudaSlice<bf16>>,
}

pub struct DecoderWeights {
    pub self_in_proj: Arc<CudaSlice<bf16>>,
    pub self_in_proj_b: Arc<CudaSlice<bf16>>,
    pub self_out_proj: Arc<CudaSlice<bf16>>,
    pub self_out_proj_b: Arc<CudaSlice<bf16>>,
    pub cross_in_proj: Arc<CudaSlice<bf16>>,
    pub cross_in_proj_b: Arc<CudaSlice<bf16>>,
    pub cross_out_proj: Arc<CudaSlice<bf16>>,
    pub cross_out_proj_b: Arc<CudaSlice<bf16>>,
    pub norm1_w: Arc<CudaSlice<bf16>>,
    pub norm1_b: Arc<CudaSlice<bf16>>,
    pub norm2_w: Arc<CudaSlice<bf16>>,
    pub norm2_b: Arc<CudaSlice<bf16>>,
    pub norm3_w: Arc<CudaSlice<bf16>>,
    pub norm3_b: Arc<CudaSlice<bf16>>,
    pub linear1_w: Arc<CudaSlice<bf16>>,
    pub linear1_b: Arc<CudaSlice<bf16>>,
    pub linear2_w: Arc<CudaSlice<bf16>>,
    pub linear2_b: Arc<CudaSlice<bf16>>,
}

pub struct JointWeights {
    pub hidden_norm_w: Arc<CudaSlice<bf16>>,
    pub hidden_norm_b: Arc<CudaSlice<bf16>>,
    pub memory_proj: Arc<CudaSlice<bf16>>,
    pub question_proj: Arc<CudaSlice<bf16>>,
    pub option_question_proj: Arc<CudaSlice<bf16>>,
    pub global_proj: Arc<CudaSlice<bf16>>,
    pub option_context_proj: Arc<CudaSlice<bf16>>,
    pub option_lexical_proj: Arc<CudaSlice<bf16>>,
    pub type_embedding: Arc<CudaSlice<bf16>>,
    pub field_norm_w: Arc<CudaSlice<bf16>>,
    pub field_norm_b: Arc<CudaSlice<bf16>>,
    pub option_norm_w: Arc<CudaSlice<bf16>>,
    pub option_norm_b: Arc<CudaSlice<bf16>>,
    pub option_summary_norm_w: Arc<CudaSlice<bf16>>,
    pub option_summary_norm_b: Arc<CudaSlice<bf16>>,
    pub scorer0_w: Arc<CudaSlice<bf16>>,
    pub scorer0_b: Arc<CudaSlice<bf16>>,
    pub scorer3_w: Arc<CudaSlice<bf16>>,
    pub scorer3_b: Arc<CudaSlice<bf16>>,
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
) -> Result<Arc<CudaSlice<bf16>>> {
    let t = view.tensor(name)?;
    ensure!(
        t.dtype() == safetensors::Dtype::BF16 && t.shape() == shape,
        "joint tensor {name} dtype/shape mismatch: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let host: Vec<bf16> = t
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut dev = stream.alloc_zeros::<bf16>(shape[0] * shape[1])?;
    stream.memcpy_htod(&host, &mut dev)?;
    Ok(Arc::new(dev))
}

/// Uploads one BF16 [n] tensor from the published file.
pub fn upload1(
    view: &safetensors::SafeTensors,
    name: &str,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<Arc<CudaSlice<bf16>>> {
    let t = view.tensor(name)?;
    ensure!(
        t.dtype() == safetensors::Dtype::BF16 && t.shape() == [n],
        "joint tensor {name} dtype/shape mismatch: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let host: Vec<bf16> = t
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut dev = stream.alloc_zeros::<bf16>(n)?;
    stream.memcpy_htod(&host, &mut dev)?;
    Ok(Arc::new(dev))
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

/// Span-mean operands: the source rows, the device span bounds and the
/// destination, with the derived widths.
pub struct SpanMean<'a> {
    pub x: &'a CudaSlice<bf16>,
    pub begins: &'a CudaSlice<i32>,
    pub ends: &'a CudaSlice<i32>,
    pub out: &'a mut CudaSlice<bf16>,
    pub spans: usize,
    pub width: usize,
}

/// Per-span row means of a BF16 [rows, width] slice: F32 accumulation over
/// each [begin, end) row range, BF16 result rows. Spans are device i32 pairs.
pub fn mean_span_rows(
    stream: &Arc<CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    m: SpanMean<'_>,
) -> Result<()> {
    use crate::backend::gpu::{DevicePtr, LaunchConfig, PushKernelArg};
    let SpanMean {
        x,
        begins,
        ends,
        out,
        spans,
        width,
    } = m;
    ensure!(
        x.len() >= begins.len() * width
            && out.len() == spans * width
            && begins.len() == spans
            && ends.len() == spans,
        "mean_span_rows extent mismatch: x={} spans={spans} width={width} out={}",
        x.len(),
        out.len()
    );
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&x.device_ptr(stream).0)
            .arg(&begins.device_ptr(stream).0)
            .arg(&ends.device_ptr(stream).0)
            .arg(&out.device_ptr(stream).0)
            .arg(&(width as i32))
            .launch(LaunchConfig {
                grid_dim: (spans as u32, width.div_ceil(256) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    /// Loads the published joint-head weights and asserts the full census:
    /// every tensor's element count and the three scalar gates.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL"]
    fn joint_weights_load_census() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir =
            std::path::PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let dev = crate::backend::cuda::Device::new(0)?;
        let w = load_joint_weights(&dir, &dev.stream)?;
        let ln1 = |t: &CudaSlice<bf16>| t.len() == 1024;
        let ln4096 = |t: &CudaSlice<bf16>| t.len() == 4096;
        let p2 = |t: &CudaSlice<bf16>| t.len() == 1024 * 4096;
        ensure!(
            ln4096(&w.hidden_norm_w) && ln4096(&w.hidden_norm_b),
            "hidden_norm"
        );
        ensure!(p2(&w.memory_proj), "memory_proj");
        ensure!(
            p2(&w.question_proj) && p2(&w.option_question_proj),
            "q/oq proj"
        );
        ensure!(
            p2(&w.global_proj) && p2(&w.option_context_proj),
            "g/ctx proj"
        );
        ensure!(p2(&w.option_lexical_proj), "lex proj");
        ensure!(w.type_embedding.len() == 3 * 1024, "type_embedding");
        ensure!(ln1(&w.field_norm_w) && ln1(&w.field_norm_b), "field_norm");
        ensure!(
            ln1(&w.option_norm_w) && ln1(&w.option_norm_b),
            "option_norm"
        );
        ensure!(
            ln1(&w.option_summary_norm_w) && ln1(&w.option_summary_norm_b),
            "option_summary_norm"
        );
        ensure!(p2(&w.scorer0_w) && w.scorer0_b.len() == 1024, "scorer0");
        ensure!(
            w.scorer3_w.len() == 1024 && w.scorer3_b.len() == 1,
            "scorer3"
        );
        for r in &w.routing {
            ensure!(
                ln1(&r.query_norm_w) && ln1(&r.query_norm_b),
                "routing qnorm"
            );
            ensure!(
                ln1(&r.memory_norm_w) && ln1(&r.memory_norm_b),
                "routing mnorm"
            );
            ensure!(r.attn_in_proj.len() == 3072 * 1024, "routing in_proj");
            ensure!(r.attn_in_proj_b.len() == 3072, "routing in_proj bias");
            ensure!(r.attn_out_proj.len() == 1024 * 1024, "routing out_proj");
            ensure!(r.attn_out_proj_b.len() == 1024, "routing out bias");
            ensure!(
                r.ff0_w.len() == 4096 * 1024 && r.ff0_b.len() == 4096,
                "routing ff0"
            );
            ensure!(
                r.ff3_w.len() == 1024 * 4096 && r.ff3_b.len() == 1024,
                "routing ff3"
            );
            ensure!(ln1(&r.ff_norm_w) && ln1(&r.ff_norm_b), "routing ffnorm");
        }
        for d in &w.decoder {
            ensure!(d.self_in_proj.len() == 3072 * 1024, "dec self in_proj");
            ensure!(d.self_in_proj_b.len() == 3072, "dec self bias");
            ensure!(d.self_out_proj.len() == 1024 * 1024, "dec self out");
            ensure!(d.self_out_proj_b.len() == 1024, "dec self out bias");
            ensure!(d.cross_in_proj.len() == 3072 * 1024, "dec cross in");
            ensure!(d.cross_in_proj_b.len() == 3072, "dec cross bias");
            ensure!(d.cross_out_proj.len() == 1024 * 1024, "dec cross out");
            ensure!(d.cross_out_proj_b.len() == 1024, "dec cross out bias");
            ensure!(ln1(&d.norm1_w) && ln1(&d.norm1_b), "dec norm1");
            ensure!(ln1(&d.norm2_w) && ln1(&d.norm2_b), "dec norm2");
            ensure!(ln1(&d.norm3_w) && ln1(&d.norm3_b), "dec norm3");
            ensure!(d.linear1_w.len() == 4096 * 1024, "dec lin1");
            ensure!(d.linear1_b.len() == 4096, "dec lin1 bias");
            ensure!(d.linear2_w.len() == 1024 * 4096, "dec lin2");
            ensure!(d.linear2_b.len() == 1024, "dec lin2 bias");
        }
        ensure!(w.routing.len() == 2 && w.decoder.len() == 4, "layer counts");
        println!(
            "JOINT_CENSUS routing={} decoder={} scales p={:.4} j={:.4} g={:.4}",
            w.routing.len(),
            w.decoder.len(),
            w.prior_logit_scale,
            w.joint_logit_scale,
            w.residual_gate
        );
        Ok(())
    }

    /// mean_span_rows against a host reference on synthetic spans.
    #[test]
    #[ignore = "requires coordinator-granted GPU window"]
    fn mean_span_rows_matches_host() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dev = crate::backend::cuda::Device::new(0)?;
        let stream = &dev.stream;
        let rows = 12usize;
        let width = 320usize;
        let host: Vec<bf16> = (0..rows * width)
            .map(|i| bf16::from_f32(((i * 37) % 199) as f32 / 199.0))
            .collect();
        let begins = vec![0i32, 5, 11];
        let ends = vec![5i32, 11, 12];
        let spans = begins.len();
        let mut x = stream.alloc_zeros::<bf16>(rows * width)?;
        stream.memcpy_htod(&host, &mut x)?;
        let mut b = stream.alloc_zeros::<i32>(spans)?;
        stream.memcpy_htod(&begins, &mut b)?;
        let mut e = stream.alloc_zeros::<i32>(spans)?;
        stream.memcpy_htod(&ends, &mut e)?;
        let mut out = stream.alloc_zeros::<bf16>(spans * width)?;
        let module = crate::backend::cuda::cubin::module(&dev.ctx, "q38_common")?;
        let kernel = module.load_function("mean_span_rows")?;
        mean_span_rows(
            stream,
            &kernel,
            SpanMean {
                x: &x,
                begins: &b,
                ends: &e,
                out: &mut out,
                spans,
                width,
            },
        )?;
        let got = stream.clone_dtoh(&out)?;
        for (s, (bs, en)) in begins.iter().zip(ends.iter()).enumerate() {
            for c in 0..width {
                let sum: f32 = (*bs as usize..*en as usize)
                    .map(|r| host[r * width + c].to_f32())
                    .sum();
                let want = bf16::from_f32(sum / (en - bs) as f32);
                ensure!(
                    got[s * width + c].to_bits() == want.to_bits(),
                    "span {s} col {c} mismatch"
                );
            }
        }
        println!("MEAN_SPAN_ROWS ok spans={spans} width={width}");
        let _ = (&mut x, &mut b, &mut e);
        Ok(())
    }
}
