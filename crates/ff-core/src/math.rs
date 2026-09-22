//! Scalar and tensor math primitives shared by the model adapters.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};
use rand::Rng;
use rand::rngs::StdRng;

/// SwiGLU gate activation, `x / (1 + e^-x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`. Above the threshold the F32 evaluation rounds to `x`.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// PyTorch evaluates half-precision SiLU in F32 and casts only the result;
/// adapters match that boundary by promoting BF16/F16 inputs once.
pub fn silu_with_reference_rounding(input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    match dtype {
        DType::BF16 | DType::F16 => input
            .to_dtype(DType::F32)?
            .silu()?
            .to_dtype(dtype)
            .map_err(Into::into),
        _ => input.silu().map_err(Into::into),
    }
}

/// RMS normalization over a slice: `x * rsqrt(mean(x^2) + eps) * weight`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm activation of {} does not match the {}-wide weight",
        x.len(),
        weight.len()
    );
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight).map(|(&v, &w)| v * inv * w).collect()
}

/// L2 normalization with the HF kernel's 1e-6 epsilon.
pub fn l2norm(x: &[f32]) -> Vec<f32> {
    let norm = (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    x.iter().map(|v| v / norm).collect()
}

/// First-maximum argmax over logits, breaking ties toward the lower index
/// like `torch.argmax`.
pub fn argmax(values: &[f32]) -> Result<u32> {
    ensure!(!values.is_empty(), "logits are empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "logits contain a non-finite value"
    );
    let mut index = 0usize;
    for (candidate, value) in values.iter().enumerate().skip(1) {
        if *value > values[index] {
            index = candidate;
        }
    }
    u32::try_from(index).context("token id exceeds u32")
}

/// Nucleus sampling: F64 softmax over the vocabulary, keep the smallest
/// prefix reaching `top_p`, draw by inverse CDF. Zero temperature is argmax.
pub fn nucleus_sample(
    values: &[f32],
    temperature: f64,
    top_p: f64,
    rng: &mut StdRng,
) -> Result<u32> {
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "sampling temperature must be finite and non-negative"
    );
    ensure!(
        top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
        "top-p must lie in (0, 1]"
    );
    if temperature == 0.0 {
        return argmax(values);
    }
    ensure!(
        temperature.recip().is_finite(),
        "sampling temperature must be safely invertible"
    );
    ensure!(!values.is_empty(), "logits are empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "logits contain a non-finite value"
    );
    let inverse = 1.0 / temperature;
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    let maximum = values[order[0]] as f64 * inverse;
    let mut probabilities = order
        .iter()
        .map(|&index| (values[index] as f64 * inverse - maximum).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    ensure!(
        total.is_finite() && total > 0.0,
        "softmax normalization is invalid"
    );
    for probability in &mut probabilities {
        *probability /= total;
    }
    let mut retained = 0usize;
    let mut cumulative = 0.0;
    for probability in &probabilities {
        cumulative += *probability;
        retained += 1;
        if cumulative >= top_p {
            break;
        }
    }
    let retained_total = probabilities[..retained].iter().sum::<f64>();
    let target = rng.random::<f64>() * retained_total;
    let mut cumulative = 0.0;
    for (rank, probability) in probabilities[..retained].iter().enumerate() {
        cumulative += *probability;
        if target <= cumulative {
            return u32::try_from(order[rank]).context("token id exceeds u32");
        }
    }
    u32::try_from(order[retained - 1]).context("token id exceeds u32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::SeedableRng;

    #[test]
    fn silu_matches_sigmoid_definition() {
        assert_eq!(silu(0.0), 0.0);
        let x = 1.25f32;
        let expected = x / (1.0 + (-x).exp());
        assert!((silu(x) - expected).abs() < 1e-7);
    }

    #[test]
    fn softplus_keeps_small_negative_scores() {
        assert_eq!(softplus(30.0), 30.0);
        let tiny = (-20.0f32).exp();
        assert!((softplus(-20.0) - tiny).abs() < 1e-12);
    }

    #[test]
    fn half_precision_silu_promotes_and_casts_once() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(vec![1.0f32, -2.0, 0.5], (3,), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let actual = silu_with_reference_rounding(&input).unwrap();
        assert_eq!(actual.dtype(), DType::BF16);
        let expected = input
            .to_dtype(DType::F32)
            .unwrap()
            .silu()
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(
            actual
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn argmax_breaks_ties_toward_the_first_maximum() {
        for (values, expected) in [
            (vec![3f32, 3., 2.], 0u32),
            (vec![2f32, 3., 3.], 1),
            (vec![1f32, 1., 1.], 0),
            (vec![-0f32, 0., -1.], 0),
            (vec![f32::MIN, f32::MIN], 0),
        ] {
            assert_eq!(argmax(&values).unwrap(), expected, "ties in {values:?}");
        }
    }

    #[test]
    fn argmax_rejects_empty_and_non_finite_logits() {
        assert!(argmax(&[]).is_err());
        assert!(argmax(&[1.0, f32::NAN]).is_err());
    }

    #[test]
    fn zero_temperature_sampling_is_greedy() {
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(
            nucleus_sample(&[-1.0, 3., 2.], 0.0, 0.95, &mut rng).unwrap(),
            1
        );
    }

    #[test]
    fn sampling_rejects_out_of_range_top_p_and_temperature() {
        let mut rng = StdRng::seed_from_u64(0);
        for (temperature, top_p) in [
            (1.0f64, 0.0f64),
            (1.0, -1.0),
            (1.0, f64::NAN),
            (1.0, 1.5),
            (-1.0, 0.95),
            (f64::INFINITY, 0.95),
        ] {
            let error = nucleus_sample(&[1.0, 2.0, 3.0], temperature, top_p, &mut rng)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("top-p") || error.contains("temperature"),
                "({temperature}, {top_p}): {error}"
            );
        }
        nucleus_sample(&[1.0, 2.0, 3.0], 1.0, 1.0, &mut rng).unwrap();
    }

    #[test]
    fn positive_temperature_must_not_underflow_into_greedy_sampling() {
        let mut rng = StdRng::seed_from_u64(7);
        let error = nucleus_sample(&[-1.0, 3., 2.], f64::from_bits(1), 0.95, &mut rng)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safely invertible"), "{error}");
    }

    #[test]
    fn nucleus_draws_from_the_retained_prefix() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..16 {
            let token = nucleus_sample(&[0.0, 10.0, 9.0, -50.0], 1.0, 0.5, &mut rng).unwrap();
            assert!(token == 1 || token == 2, "sampled {token}");
        }
    }
}
