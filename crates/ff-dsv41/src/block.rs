//! One transformer block: hyper-connection residual plumbing around the
//! attention and FFN sublayers, with the optional engram insertion.
//!
//! Reference: `Block` in `inference/model.py`. The residual stream is
//! `hc_mult` parallel copies; each sublayer's coefficients are computed by
//! its own `hc_mixes` and consumed by the *next* one.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

use crate::math::{hc_split_sinkhorn, rms_norm};

/// Coefficient sets one `hc_mixes` projection yields.
pub struct HcMixes {
    pub pre: Tensor,
    pub post: Tensor,
    pub comb: Tensor,
}

/// Project the flattened stream, normalized over the whole `hc * dim` width,
/// and split the result through the Sinkhorn mixer.
#[allow(clippy::too_many_arguments)]
pub fn hc_mixes(
    x: &Tensor,
    hc_fn: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    hc_mult: usize,
    sinkhorn_iters: usize,
    hc_eps: f64,
    norm_eps: f64,
) -> Result<HcMixes> {
    let dims = x.dims();
    ensure!(
        dims.len() == 4 && dims[2] == hc_mult,
        "hc mixes need [batch, seq, {hc_mult}, dim], found {dims:?}"
    );
    let hidden = dims[3];
    let mix_width = (2 + hc_mult) * hc_mult;
    ensure!(
        hc_fn.dims() == [mix_width, hc_mult * hidden],
        "hc_fn must be [{mix_width}, {}]",
        hc_mult * hidden
    );
    let flat = x
        .to_dtype(DType::F32)?
        .flatten_from(2)?
        .flatten_to(1)?
        .contiguous()?;
    let tokens = flat.dims()[0];
    let rows = flat
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read hc stream")?;
    let (hc_fn_guard, _) = crate::math::resident_f32(hc_fn)?;
    let weights = crate::math::resident_f32_slice(&hc_fn_guard)?;
    let mut mixes = Vec::with_capacity(tokens * mix_width);
    for token in 0..tokens {
        let base = token * hc_mult * hidden;
        for row in 0..mix_width {
            let mut sum = 0.0f32;
            for column in 0..hc_mult * hidden {
                sum += weights[row * hc_mult * hidden + column] * rows[base + column];
            }
            mixes.push(sum);
        }
    }
    let row_width = hc_mult * hidden;
    for token in 0..tokens {
        let squares = rows[token * row_width..(token + 1) * row_width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / row_width as f32;
        let rstd = 1.0 / (squares + norm_eps as f32).sqrt();
        for value in &mut mixes[token * mix_width..(token + 1) * mix_width] {
            *value *= rstd;
        }
    }
    let mixes =
        Tensor::from_vec(mixes, (tokens, mix_width), x.device()).map_err(anyhow::Error::from)?;
    let (pre, post, comb) =
        hc_split_sinkhorn(&mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)?;
    Ok(HcMixes { pre, post, comb })
}

/// Collapse the hc copies into one sublayer input, weighted by `pre_mix`.
pub fn hc_pre(x: &Tensor, pre_mix: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    ensure!(dims.len() == 4, "hc_pre needs [batch, seq, hc, dim]");
    let values = x
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read hc stream")?;
    let mixes = pre_mix
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read pre mix")?;
    let [batch, seq, hc, hidden] = [dims[0], dims[1], dims[2], dims[3]];
    let mut output = vec![0.0f32; batch * seq * hidden];
    for b in 0..batch {
        for s in 0..seq {
            for d in 0..hidden {
                let mut sum = 0.0;
                for copy in 0..hc {
                    sum += mixes[(b * seq + s) * hc + copy]
                        * values[((b * seq + s) * hc + copy) * hidden + d];
                }
                output[(b * seq + s) * hidden + d] = sum;
            }
        }
    }
    Tensor::from_vec(output, (batch, seq, hidden), x.device()).map_err(anyhow::Error::from)
}

/// Expand a sublayer output back to hc copies and mix the residual in
/// through `comb`.
pub fn hc_post(x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    ensure!(dims.len() == 3, "hc_post input must be [batch, seq, dim]");
    let residual_dims = residual.dims();
    ensure!(
        residual_dims.len() == 4,
        "hc_post residual must be [batch, seq, hc, dim]"
    );
    let [batch, seq, hidden] = [dims[0], dims[1], dims[2]];
    let hc = residual_dims[2];
    let sublayer = x
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read sublayer output")?;
    let stream = residual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read residual stream")?;
    let post_values = post
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read post mix")?;
    let comb_values = comb
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read comb")?;
    let mut output = vec![0.0f32; batch * seq * hc * hidden];
    for b in 0..batch {
        for s in 0..seq {
            let token = b * seq + s;
            for copy in 0..hc {
                let scale = post_values[token * hc + copy];
                for d in 0..hidden {
                    let mut value = scale * sublayer[token * hidden + d];
                    for source in 0..hc {
                        // comb's first axis is the source stream and the
                        // second the destination, matching the reference mHC
                        // contraction (and ff-glm's explicit transpose).
                        value += comb_values[(token * hc + source) * hc + copy]
                            * stream[(token * hc + source) * hidden + d];
                    }
                    output[(token * hc + copy) * hidden + d] = value;
                }
            }
        }
    }
    Tensor::from_vec(output, (batch, seq, hc, hidden), x.device()).map_err(anyhow::Error::from)
}

/// The block tail: normalize the collapsed stream. Kept here so callers share
/// one RMSNorm entry point with the model.
pub fn block_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    rms_norm(x, weight, eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn hc_mixes_normalize_each_token_by_its_own_row() {
        let device = candle_core::Device::Cpu;
        let hc = 1usize;
        let hidden = 4usize;
        // Selector rows: mix row j reads channel j of the token's flat row.
        let mut selector = vec![0.0f32; 3 * hidden];
        for row in 0..3 {
            selector[row * hidden + row] = 1.0;
        }
        let hc_fn = Tensor::from_vec(selector, (3, hidden), &device).unwrap();
        let x = Tensor::from_vec(
            vec![
                1.0f32, 1.0, 1.0, 1.0, // token 0: unit scale
                100.0f32, 100.0, 100.0, 100.0, // token 1: 100x scale
            ],
            (1, 2, hc, hidden),
            &device,
        )
        .unwrap();
        let scale = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (3,), &device).unwrap();
        let base = Tensor::zeros(3, candle_core::DType::F32, &device).unwrap();
        let mixes = hc_mixes(&x, &hc_fn, &scale, &base, hc, 4, 1e-6, 1e-20).unwrap();
        // Per-token stats make the scaled mixes scale-invariant: both tokens
        // land at 1/sqrt(hidden), so their pre coefficients agree. A global
        // statistic would keep them 100x apart.
        let pre = mixes.pre.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // rstd = 1/sqrt(mean of squares over the row); a unit channel scaled
        // by it lands at exactly 1.0.
        let expected = 1.0 / (1.0 + (-1.0f32).exp()) + 1e-6;
        assert!((pre[0] - expected).abs() < 1e-4, "{} vs {expected}", pre[0]);
        assert!((pre[1] - pre[0]).abs() < 1e-4, "{} vs {}", pre[1], pre[0]);
    }

    #[test]
    fn pre_collapse_and_post_expand_round_trip_a_copy() {
        let device = Device::Cpu;
        let hc = 2;
        let stream = Tensor::from_vec(
            (0..3 * hc * 4).map(|index| index as f32 * 0.25).collect(),
            (1, 3, hc, 4),
            &device,
        )
        .unwrap();
        let pre =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], (1, 3, hc), &device).unwrap();
        let collapsed = hc_pre(&stream, &pre).unwrap();
        assert_eq!(collapsed.dims(), [1, 3, 4]);
        let values = collapsed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Token 0 mixes (1,0): just copy 0's channels 0..4.
        assert_eq!(values[0], 0.0);
        assert_eq!(values[3], 0.75);
        // Token 2 mixes (1,1): sum of both copies.
        assert_eq!(values[8], (16 + 20) as f32 * 0.25);

        let post =
            Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (1, 3, hc), &device).unwrap();
        let comb = Tensor::zeros((1, 3, hc, hc), DType::F32, &device).unwrap();
        let expanded = hc_post(&collapsed, &stream, &post, &comb).unwrap();
        assert_eq!(expanded.dims(), [1, 3, hc, 4]);
        // With zero comb and post 1, every copy of the output equals the
        // collapsed sublayer value itself.
        let out = expanded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out[0], values[0]);
        assert_eq!(out[4], values[0]);
        assert_eq!(out[8], values[4]);
    }

    #[test]
    fn hc_post_contracts_comb_by_source_rows_not_destination_rows() {
        let device = candle_core::Device::Cpu;
        // One token, hc=2, hidden=1, non-symmetric comb: source 0 feeds only
        // destination 1. The reference contraction reads comb[source][dest],
        // so destination 0 receives nothing and destination 1 receives
        // comb[0][1] * stream[0]. Reading comb[dest][source] instead (the old
        // behaviour) moves the value to destination 0.
        // comb[0][1] * stream[0] = 1.0 * 2.0 = 2.0 lands at destination 1;
        // destination 0 receives nothing. The transposed (old) contraction
        // would produce [3.0, 0.0] instead.
        let comb = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 0.0], (1, 1, 2, 2), &device).unwrap();
        let residual = Tensor::from_vec(vec![2.0f32, 3.0], (1, 1, 2, 1), &device).unwrap();
        let sublayer = Tensor::from_vec(vec![9.0f32], (1, 1, 1), &device).unwrap();
        let post = Tensor::zeros((1, 1, 2), DType::F32, &device).unwrap();
        let out = hc_post(&sublayer, &residual, &post, &comb).unwrap();
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, [0.0, 2.0]);
    }
}
