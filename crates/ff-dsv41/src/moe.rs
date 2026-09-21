//! Top-k routed experts plus one shared expert.
//!
//! Reference: `inference/model.py` (`Gate`, `Expert`, `MoE`). The correction
//! bias steers expert selection only; routing weights come from the unbiased
//! sqrt-softplus scores. Image-span tokens use a separate bias.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct Gate {
    pub weight: Tensor,
    pub bias: Tensor,
    pub bias_vl: Option<Tensor>,
    pub topk: usize,
    pub gate_temp: f64,
    pub norm_topk_prob: bool,
    pub route_scale: f64,
}

impl Gate {
    /// Returns the routing weights and selected expert ids per token.
    pub fn forward(&self, x: &Tensor, image_mask: Option<&[bool]>) -> Result<(Vec<f32>, Vec<u32>)> {
        let dims = x.dims();
        ensure!(dims.len() == 2, "gate input must be [tokens, hidden]");
        let [tokens, hidden] = [dims[0], dims[1]];
        let experts = self.weight.dims()[0];
        ensure!(
            self.weight.dims() == [experts, hidden],
            "gate weight must be [experts, hidden]"
        );
        let x_values = x
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read gate input")?;
        let weight = self
            .weight
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read gate weight")?;
        let bias = self
            .bias
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read gate bias")?;
        let bias_vl = match &self.bias_vl {
            Some(tensor) => Some(
                tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .context("read gate image bias")?,
            ),
            None => None,
        };
        let topk = self.topk;
        let mut weights = vec![0.0f32; tokens * topk];
        let mut indices = vec![0u32; tokens * topk];
        for token in 0..tokens {
            let base = token * hidden;
            let mut scores = Vec::with_capacity(experts);
            for expert in 0..experts {
                let mut dot = 0.0f32;
                for column in 0..hidden {
                    dot += weight[expert * hidden + column] * x_values[base + column];
                }
                scores.push(softplus(dot / self.gate_temp as f32).sqrt());
            }
            let selection_bias: &Vec<f32> = match (bias_vl.as_ref(), image_mask) {
                (Some(bias_vl), Some(mask)) if mask[token] => bias_vl,
                _ => &bias,
            };
            let mut order = (0..experts).collect::<Vec<_>>();
            order.sort_by(|a, b| {
                (scores[*b] + selection_bias[*b])
                    .partial_cmp(&(scores[*a] + selection_bias[*a]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut total = 0.0f32;
            for (slot, expert) in order.iter().take(topk).enumerate() {
                indices[token * topk + slot] = *expert as u32;
                weights[token * topk + slot] = scores[*expert];
                total += scores[*expert];
            }
            if self.norm_topk_prob && topk > 1 {
                for weight in &mut weights[token * topk..(token + 1) * topk] {
                    *weight /= total + 1e-20;
                }
            }
            for weight in &mut weights[token * topk..(token + 1) * topk] {
                *weight *= self.route_scale as f32;
            }
        }
        Ok((weights, indices))
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        // ln(1 + e^x) loses small negative scores to F32 spacing; ln_1p
        // evaluates them accurately instead of rounding to zero.
        value.exp().ln_1p()
    }
}

/// One SwiGLU FFN. The clamps come straight from training: the up branch is
/// clamped on both sides, the gate branch only from above.
pub struct Expert {
    pub w1: Tensor,
    pub w2: Tensor,
    pub w3: Tensor,
    pub swiglu_limit: f64,
}

impl Expert {
    pub fn forward(&self, x: &Tensor, weight: f32) -> Result<Tensor> {
        let tokens = x.dims()[0];
        self.forward_weighted(x, &vec![weight; tokens])
    }

    pub fn forward_weighted(&self, x: &Tensor, weights: &[f32]) -> Result<Tensor> {
        let dims = x.dims();
        ensure!(dims.len() == 2, "expert input must be [tokens, hidden]");
        let [tokens, hidden] = [dims[0], dims[1]];
        ensure!(
            weights.len() == tokens,
            "expert weights {} do not cover {tokens} rows",
            weights.len()
        );
        let inter = self.w1.dims()[0];
        let x_values = x
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read expert input")?;
        let (w1_guard, _) = crate::math::resident_f32(&self.w1)?;
        let w1 = crate::math::resident_f32_slice(&w1_guard)?;
        let (w2_guard, _) = crate::math::resident_f32(&self.w2)?;
        let w2 = crate::math::resident_f32_slice(&w2_guard)?;
        let (w3_guard, _) = crate::math::resident_f32(&self.w3)?;
        let w3 = crate::math::resident_f32_slice(&w3_guard)?;
        let limit = self.swiglu_limit as f32;
        let mut output = vec![0.0f32; tokens * hidden];
        for (token, &weight) in weights.iter().enumerate() {
            let base = token * hidden;
            let mut inner = vec![0.0f32; inter];
            for row in 0..inter {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for column in 0..hidden {
                    let value = x_values[base + column];
                    gate += w1[row * hidden + column] * value;
                    up += w3[row * hidden + column] * value;
                }
                let gate = if limit > 0.0 { gate.min(limit) } else { gate };
                let up = if limit > 0.0 {
                    up.clamp(-limit, limit)
                } else {
                    up
                };
                inner[row] = silu(gate) * up * weight;
            }
            for column in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..inter {
                    sum += w2[column * inter + row] * inner[row];
                }
                output[base + column] = sum;
            }
        }
        Tensor::from_vec(output, (tokens, hidden), x.device()).map_err(anyhow::Error::from)
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn sqrt_softplus_preserves_small_negative_scores() {
        // (1 + e^x).ln() rounds to zero below about -17 in F32; ln_1p keeps
        // the positive score the reference softplus produces.
        let small = softplus(-20.0);
        assert!(small > 0.0, "softplus(-20) collapsed to zero");
        assert!((small - (-20.0f32).exp()).abs() < 1e-9, "{small}");
        assert_eq!(softplus(25.0), 25.0);
    }

    fn two_expert_gate() -> Gate {
        let device = Device::Cpu;
        // Expert 0 outranks expert 1 on the raw scores; the bias flips it.
        let weight = Tensor::from_vec(vec![2.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let bias = Tensor::from_vec(vec![0.0f32, 5.0], (2,), &device).unwrap();
        Gate {
            weight,
            bias,
            bias_vl: None,
            topk: 2,
            gate_temp: 1.0,
            norm_topk_prob: true,
            route_scale: 1.5,
        }
    }

    #[test]
    fn gate_selects_by_bias_but_weights_by_raw_score() {
        let device = Device::Cpu;
        let gate = two_expert_gate();
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        let (weights, indices) = gate.forward(&x, None).unwrap();
        // softplus(2)^2 vs softplus(1)^2... scores are sqrt(softplus(dot)):
        // expert0 score = sqrt(softplus(2)), expert1 = sqrt(softplus(1)); the
        // +5 bias puts expert 1 first in selection order.
        assert_eq!(indices[0], 1);
        assert_eq!(indices[1], 0);
        let score0 = softplus(2.0).sqrt();
        let score1 = softplus(1.0).sqrt();
        let total = score0 + score1;
        assert!((weights[0] - score1 / (total + 1e-20) * 1.5).abs() < 1e-5);
        assert!((weights[1] - score0 / (total + 1e-20) * 1.5).abs() < 1e-5);
    }

    #[test]
    fn image_tokens_use_the_image_bias() {
        let device = Device::Cpu;
        let mut gate = two_expert_gate();
        gate.bias = Tensor::from_vec(vec![5.0f32, 0.0], (2,), &device).unwrap();
        gate.bias_vl = Some(Tensor::from_vec(vec![0.0f32, 5.0], (2,), &device).unwrap());
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device).unwrap();
        let (_, plain) = gate.forward(&x, Some(&[false])).unwrap();
        let (_, image) = gate.forward(&x, Some(&[true])).unwrap();
        assert_eq!(plain[0], 0);
        assert_eq!(image[0], 1);
    }

    #[test]
    fn expert_clamps_then_blends() {
        let device = Device::Cpu;
        // w1/w3 rows pick single input channels; limit 1.0 clamps both.
        let w1 = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();
        let w3 = Tensor::from_vec(vec![0.0f32, 1.0], (1, 2), &device).unwrap();
        let w2 = Tensor::from_vec(vec![3.0f32, 3.0], (2, 1), &device).unwrap();
        let expert = Expert {
            w1,
            w2,
            w3,
            swiglu_limit: 1.0,
        };
        let x = Tensor::from_vec(vec![4.0f32, 0.5], (1, 2), &device).unwrap();
        let output = expert.forward(&x, 2.0).unwrap();
        // gate = min(4, 1) = 1, up = clamp(0.5) = 0.5; silu(1) * 0.5 * 2 * 3.
        let expected = silu(1.0) * 0.5 * 2.0 * 3.0;
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - expected).abs() < 1e-5);
        assert!((values[1] - expected).abs() < 1e-5);
    }
}
