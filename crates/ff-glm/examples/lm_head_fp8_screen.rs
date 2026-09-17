//! Weight-domain FP8 screen for GLM's lm_head (D2's first measurement).
//!
//! Quantizes the real BF16 lm_head to e4m3 at two scale granularities
//! (per-row, and the 128x128 block layout the expert path uses), dequantizes
//! back, and reports the error distribution per vocabulary row. This answers
//! "would FP8 residency flip argmax" at the weight level before any kernel
//! work; the logit-level check against captured hidden states comes after.
//!
//! Usage: cargo run --release -p ff-glm --example lm_head_fp8_screen -- <checkpoint-dir>

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

fn fp8_roundtrip(weights: &Tensor, scales: &Tensor, rb: usize, cb: usize) -> Result<Tensor> {
    let (rows, cols) = weights.dims2().context("lm_head must be rank-2")?;
    let nrow = rows.div_ceil(rb);
    let ncol = cols.div_ceil(cb);
    let blocked = weights
        .reshape((nrow, rb, ncol, cb))?
        .transpose(1, 2)?
        .flatten(2, 3)?; // [nrow, ncol, rb*cb]
    let scales_bc = scales
        .reshape((nrow, ncol, 1))?
        .broadcast_as((nrow, ncol, rb * cb))?
        .contiguous()?;
    // `scales` is absmax/448, so quantization divides by it — that maps each
    // block's largest magnitude onto e4m3's 448 and uses the format's range.
    // Multiplying instead drives every weight below e4m3's smallest subnormal
    // (2^-9): a block with absmax 0.5 becomes 5.6e-4, flushes to zero, and the
    // screen reports ~100% relative error whether or not FP8 is usable here.
    let q = (blocked.clone() * &scales_bc.recip()?)?.to_dtype(DType::F8E4M3)?;
    let back = (q.to_dtype(DType::F32)? * &scales_bc)?;
    Ok(back
        .reshape((nrow, ncol, rb, cb))?
        .transpose(1, 2)?
        .reshape((rows, cols))?)
}

fn screen(lm_head: &Tensor, name: &str, rb: usize, cb: usize) -> Result<()> {
    let (rows, cols) = lm_head.dims2()?;
    let nrow = rows.div_ceil(rb);
    let ncol = cols.div_ceil(cb);
    let blocked = lm_head
        .reshape((nrow, rb, ncol, cb))?
        .transpose(1, 2)?
        .flatten(2, 3)?;
    let absmax = blocked.abs()?.max(2)?.clamp(1e-12, f64::MAX)?;
    let scales = (absmax / 448.0)?; // e4m3 max magnitude
    let roundtrip = fp8_roundtrip(lm_head, &scales, rb, cb)?;
    let err = (&roundtrip - lm_head)?.abs()?;
    let rel = (&err / &lm_head.abs()?.clamp(1e-3, f64::MAX)?)?;
    let mut rel_sorted = rel.flatten_all()?.to_vec1::<f32>()?;
    rel_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = rel_sorted.len();
    println!(
        "{name}: relative |err| median {:.4} p95 {:.4} p99 {:.4} max {:.4}",
        rel_sorted[n / 2],
        rel_sorted[n * 95 / 100],
        rel_sorted[n * 99 / 100],
        rel_sorted[n - 1]
    );
    let mut abs_sorted = err.flatten_all()?.to_vec1::<f32>()?;
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{name}: absolute |err| median {:.6} p99 {:.6} max {:.6}",
        abs_sorted[n / 2],
        abs_sorted[n * 99 / 100],
        abs_sorted[n - 1]
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .context("usage: lm_head_fp8_screen <checkpoint-dir>")?;
    let shard = format!("{dir}/model-00001-of-00062.safetensors");
    let tensors = candle_core::safetensors::load(&shard, &Device::Cpu)?;
    let lm_head = tensors
        .get("lm_head.weight")
        .context("lm_head.weight not in shard 1")?
        .to_dtype(DType::F32)?;
    let (rows, cols) = lm_head.dims2()?;
    println!("lm_head [{}x{}] BF16 loaded", rows, cols);
    screen(&lm_head, "per-row (scale per vocab row)", 1, cols)?;
    screen(&lm_head, "block 128x128 (expert-path layout)", 128, 128)?;
    Ok(())
}
