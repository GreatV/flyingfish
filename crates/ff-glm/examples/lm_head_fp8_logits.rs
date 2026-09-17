//! Logit-level FP8 harness for GLM's lm_head (D2 acceptance measurement).
//!
//! Inputs: the checkpoint's lm_head plus one or more capture-parity
//! safetensors, each holding a real final hidden state (the lm_head input,
//! [4096]) and the pipeline's reference next-token logits. Recomputes logits
//! through a per-row e4m3-quantized lm_head and reports per capture: top-1
//! flip vs the BF16 recompute, top-10 overlap, max absolute logit deviation,
//! winner-margin shift, and a sanity deviation between the BF16 recompute and
//! the captured reference.
//!
//! Usage: cargo run --release -p ff-glm --example lm_head_fp8_logits --
//!   <checkpoint-dir> <capture.safetensors> [more captures...]

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};

fn quantize_per_row_fp8(w: &Tensor) -> Result<Tensor> {
    let (rows, cols) = w.dims2()?;
    let am = w.abs()?.max_keepdim(1)?.clamp(1e-12, f64::MAX)?;
    let scale = (am / 448.0)?;
    let scale_bc = scale.broadcast_as((rows, cols))?.contiguous()?;
    let q = w.div(&scale_bc)?.to_dtype(DType::F8E4M3)?;
    Ok(q.to_dtype(DType::F32)?.mul(&scale_bc)?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .context("usage: <checkpoint-dir> <capture>...")?;
    let captures = &args[2..];
    ensure!(
        !captures.is_empty(),
        "at least one capture file is required"
    );
    let device = Device::Cpu;
    let lm_head =
        candle_core::safetensors::load(format!("{dir}/model-00001-of-00062.safetensors"), &device)?
            .get("lm_head.weight")
            .context("lm_head.weight missing")?
            .to_dtype(DType::F32)?;
    let lm_head_fp8 = quantize_per_row_fp8(&lm_head)?;
    let vocab = lm_head.dims()[0];
    let lm_head_t = lm_head.t()?;
    let lm_head_fp8_t = lm_head_fp8.t()?;

    let mut total_flips = 0usize;
    for path in captures {
        let capture = candle_core::safetensors::load(path, &device)?;
        let hidden = capture
            .get("final_hidden_state")
            .context("final_hidden_state missing")?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?; // [1, 4096]
        let reference = capture
            .get("next_token_logits")
            .context("next_token_logits missing")?
            .to_dtype(DType::F32)?
            .flatten_all()?;
        let reference_id = capture
            .get("next_token_id")
            .and_then(|t| t.to_vec0::<u32>().ok())
            .context("next_token_id missing")?;

        let logits_bf16 = hidden.matmul(&lm_head_t)?.squeeze(0)?;
        let logits_fp8 = hidden.matmul(&lm_head_fp8_t)?.squeeze(0)?;
        let max_dev = (&logits_fp8 - &logits_bf16)?
            .abs()?
            .max(0)?
            .to_vec0::<f32>()?;
        let sanity_dev = (&logits_bf16 - &reference)?
            .abs()?
            .max(0)?
            .to_vec0::<f32>()?;
        let top_bf16 = logits_bf16.argmax(0)?.to_vec0::<u32>()?;
        let top_fp8 = logits_fp8.argmax(0)?.to_vec0::<u32>()?;
        let flip = top_bf16 != top_fp8;
        total_flips += flip as usize;

        let bf16_v = logits_bf16.to_vec1::<f32>()?;
        let fp8_v = logits_fp8.to_vec1::<f32>()?;
        let mut bf16_idx: Vec<u32> = (0..vocab as u32).collect();
        let mut fp8_idx = bf16_idx.clone();
        bf16_idx.sort_by(|a, b| bf16_v[*b as usize].total_cmp(&bf16_v[*a as usize]));
        fp8_idx.sort_by(|a, b| fp8_v[*b as usize].total_cmp(&fp8_v[*a as usize]));
        let overlap = bf16_idx[..10]
            .iter()
            .filter(|i| fp8_idx[..10].contains(i))
            .count();
        let margin_bf16 = bf16_v[bf16_idx[0] as usize] - bf16_v[bf16_idx[1] as usize];
        let margin_fp8 = fp8_v[fp8_idx[0] as usize] - fp8_v[fp8_idx[1] as usize];

        println!(
            "{path}: flip={flip} top10_overlap={overlap}/10 max|Δlogit|={max_dev:.4} \
             sanity|recompute−reference|={sanity_dev:.5} margin {margin_bf16:.3}->{margin_fp8:.3} \
             pipeline_next={reference_id} bf16_top={top_bf16} fp8_top={top_fp8}"
        );
    }
    println!("total top-1 flips: {total_flips}/{}", captures.len());
    Ok(())
}
