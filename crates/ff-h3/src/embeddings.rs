use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

pub fn timestep_projection(timesteps: &Tensor, embedding_dim: usize) -> Result<Tensor> {
    anyhow::ensure!(timesteps.rank() == 1, "timesteps must be one-dimensional");
    let half = embedding_dim / 2;
    anyhow::ensure!(half > 0, "timestep embedding dimension is too small");
    let frequencies = (0..half)
        .map(|index| (-10_000f32.ln() * index as f32 / half as f32).exp())
        .collect::<Vec<_>>();
    let frequencies = Tensor::from_vec(frequencies, half, timesteps.device())?;
    let phases = timesteps
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&frequencies.unsqueeze(0)?)?;
    let mut result = Tensor::cat(&[&phases.cos()?, &phases.sin()?], 1)?;
    if embedding_dim % 2 == 1 {
        let padding = Tensor::zeros((timesteps.dim(0)?, 1), DType::F32, timesteps.device())?;
        result = Tensor::cat(&[&result, &padding], 1)?;
    }
    Ok(result)
}

pub fn rotary(
    position_ids: &Tensor,
    rope_freq_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let (sequence, axes) = position_ids
        .dims2()
        .context("position_ids must be [sequence, 3]")?;
    anyhow::ensure!(axes == 3, "position_ids must have exactly three axes");
    anyhow::ensure!(
        rope_freq_dim > 0,
        "rope frequency dimension must be non-zero"
    );
    let inverse_frequencies = (0..rope_freq_dim)
        .map(|index| {
            let exponent = (2 * index) as f32 / (2 * rope_freq_dim) as f32;
            1f32 / (rope_theta as f32).powf(exponent)
        })
        .collect::<Vec<_>>();
    let inverse_frequencies = Tensor::from_vec(
        inverse_frequencies,
        (1, 1, rope_freq_dim),
        position_ids.device(),
    )?;
    let frequencies = position_ids
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .broadcast_mul(&inverse_frequencies)?;
    let mut axes = Vec::with_capacity(3);
    for axis in 0..3 {
        axes.push(
            frequencies
                .narrow(1, axis, 1)?
                .reshape((sequence, rope_freq_dim))?,
        );
    }
    let axis_refs = axes.iter().collect::<Vec<_>>();
    let frequencies = Tensor::cat(&axis_refs, 1)?;
    let frequencies = Tensor::cat(&[&frequencies, &frequencies], 1)?;
    Ok((frequencies.cos()?, frequencies.sin()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn timestep_zero_is_cosine_then_sine() {
        let timestep = Tensor::new(&[0f32], &Device::Cpu).unwrap();
        let value = timestep_projection(&timestep, 5)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(value, vec![vec![1., 1., 0., 0., 0.]]);
    }

    #[test]
    fn zero_positions_have_identity_rotation() {
        let positions = Tensor::zeros((2, 3), DType::I64, &Device::Cpu).unwrap();
        let (cos, sin) = rotary(&positions, 2, 10_000.).unwrap();
        assert_eq!(cos.dims(), &[2, 12]);
        assert!(
            cos.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 1.)
        );
        assert!(
            sin.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 0.)
        );
    }

    #[test]
    fn rotary_rounds_power_and_reciprocal_in_f32() {
        let positions = Tensor::new(&[[357f32, 0., 0.]], &Device::Cpu).unwrap();
        let (cos, sin) = rotary(&positions, 16, 10_000.).unwrap();
        let cos = cos.to_vec2::<f32>().unwrap();
        let sin = sin.to_vec2::<f32>().unwrap();
        let mut differs_from_f64 = false;
        for index in 0..16 {
            let inverse = 1f32 / 10_000f32.powf(index as f32 / 16.);
            let angle = 357f32 * inverse;
            assert_eq!(cos[0][index].to_bits(), angle.cos().to_bits());
            assert_eq!(sin[0][index].to_bits(), angle.sin().to_bits());
            let old_inverse = (1f64 / 10_000f64.powf(index as f64 / 16.)) as f32;
            differs_from_f64 |= inverse.to_bits() != old_inverse.to_bits();
        }
        assert!(
            differs_from_f64,
            "regression must distinguish the old FP64 construction"
        );
    }
}
