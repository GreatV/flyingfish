use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
struct SchedulerConfig {
    shift: f32,
}

#[derive(Clone, Debug)]
pub struct H3Scheduler {
    shift: f32,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    step_index: Option<usize>,
}

impl H3Scheduler {
    pub fn new(shift: f32) -> Result<Self> {
        anyhow::ensure!(
            shift.is_finite() && shift > 0.,
            "scheduler shift must be finite and positive"
        );
        Ok(Self {
            shift,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: None,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read scheduler config {}", path.display()))?;
        let config: SchedulerConfig = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid scheduler config {}", path.display()))?;
        Self::new(config.shift)
    }

    pub fn shift(&self) -> f32 {
        self.shift
    }

    pub fn set_shift(&mut self, shift: f32) -> Result<()> {
        anyhow::ensure!(
            shift.is_finite() && shift > 0.,
            "scheduler shift must be finite and positive"
        );
        self.shift = shift;
        self.sigmas.clear();
        self.timesteps.clear();
        self.step_index = None;
        Ok(())
    }

    pub fn set_timesteps(&mut self, points: usize) -> Result<&[f32]> {
        anyhow::ensure!(points >= 2, "scheduler requires at least two sigma points");
        let denominator = (points - 1) as f32;
        let mut sigmas = Vec::with_capacity(points);
        for index in 0..points {
            let base = 1. - index as f32 / denominator;
            let shifted = self.shift * base / (1. + (self.shift - 1.) * base);
            if sigmas.last().copied() != Some(shifted) {
                sigmas.push(shifted);
            }
        }
        self.install_sigmas(sigmas)?;
        Ok(&self.timesteps)
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn step_index(&self) -> Option<usize> {
        self.step_index
    }

    pub fn seek(&mut self, evaluation_index: usize) -> Result<()> {
        anyhow::ensure!(!self.timesteps.is_empty(), "scheduler has no timesteps");
        anyhow::ensure!(
            evaluation_index < self.timesteps.len(),
            "scheduler evaluation index {evaluation_index} is out of range for {} timesteps",
            self.timesteps.len()
        );
        self.step_index = Some(evaluation_index);
        Ok(())
    }

    pub fn scale_noise(&self, sample: &Tensor, timestep: f32, noise: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(sample.dims() == noise.dims(), "sample/noise shapes differ");
        anyhow::ensure!(
            sample.dtype() == noise.dtype(),
            "sample/noise dtypes differ"
        );
        sample
            .affine(timestep as f64, 0.)?
            .add(&noise.affine((1. - timestep) as f64, 0.)?)
            .map_err(Into::into)
    }

    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
    ) -> Result<Tensor> {
        anyhow::ensure!(!self.timesteps.is_empty(), "scheduler has no timesteps");
        anyhow::ensure!(
            model_output.dims() == sample.dims(),
            "model output/sample shapes differ"
        );
        anyhow::ensure!(
            model_output.dtype() == sample.dtype(),
            "model output/sample dtypes differ"
        );
        let index = match self.step_index {
            Some(index) => {
                anyhow::ensure!(
                    index < self.timesteps.len(),
                    "scheduler is already complete"
                );
                anyhow::ensure!(
                    self.timesteps[index] == timestep,
                    "timestep {timestep} does not match scheduler evaluation index {index} ({})",
                    self.timesteps[index]
                );
                index
            }
            None => self
                .timesteps
                .iter()
                .position(|&candidate| candidate == timestep)
                .context("timestep is not present in the scheduler")?,
        };
        anyhow::ensure!(
            index + 1 < self.sigmas.len(),
            "scheduler is already complete"
        );

        let sigma_from_timestep = 1. - timestep;
        let denoised = sample.add(&model_output.affine(sigma_from_timestep as f64, 0.)?)?;
        let ratio = self.sigmas[index + 1] / self.sigmas[index];
        let original_dtype = sample.dtype();
        let compute_dtype = match original_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            dtype => dtype,
        };
        let sample = sample.to_dtype(compute_dtype)?;
        let denoised = denoised.to_dtype(compute_dtype)?;
        let previous = sample
            .affine(ratio as f64, 0.)?
            .add(&denoised.affine((1. - ratio) as f64, 0.)?)?
            .to_dtype(original_dtype)?;
        self.step_index = Some(index + 1);
        Ok(previous)
    }

    fn install_sigmas(&mut self, sigmas: Vec<f32>) -> Result<()> {
        anyhow::ensure!(
            sigmas.len() >= 2,
            "sigma schedule needs at least two values"
        );
        anyhow::ensure!(
            sigmas.windows(2).all(|pair| pair[1] < pair[0]),
            "sigmas must be strictly decreasing"
        );
        anyhow::ensure!(
            sigmas.last() == Some(&0.),
            "sigma schedule must end at zero"
        );
        anyhow::ensure!(
            sigmas.iter().all(|value| value.is_finite()),
            "sigmas must be finite"
        );
        self.timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|sigma| 1. - sigma)
            .collect();
        self.sigmas = sigmas;
        self.step_index = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn builds_shifted_video_schedule_with_terminal_zero() {
        let mut scheduler = H3Scheduler::new(12.).unwrap();
        scheduler.set_timesteps(3).unwrap();
        assert_eq!(scheduler.sigmas()[0], 1.);
        assert!((scheduler.sigmas()[1] - 12. / 13.).abs() < 1e-6);
        assert_eq!(scheduler.sigmas()[2], 0.);
        assert_eq!(scheduler.timesteps().len(), 2);
    }

    #[test]
    fn rejects_nonfinite_shifts() {
        assert!(H3Scheduler::new(f32::INFINITY).is_err());
        assert!(H3Scheduler::new(f32::NAN).is_err());
        let mut scheduler = H3Scheduler::new(1.0).unwrap();
        assert!(scheduler.set_shift(f32::INFINITY).is_err());
        assert!(scheduler.set_shift(f32::NAN).is_err());
    }

    #[test]
    fn data_ward_velocity_moves_sample_toward_denoised() {
        let mut scheduler = H3Scheduler::new(1.).unwrap();
        let timesteps = scheduler.set_timesteps(3).unwrap().to_vec();
        let sample = Tensor::new(&[2f32], &Device::Cpu).unwrap();
        let velocity = Tensor::new(&[4f32], &Device::Cpu).unwrap();
        let first = scheduler
            .step(&velocity, timesteps[0], &sample)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(first, vec![4.]);
        let second_sample = Tensor::new(first.as_slice(), &Device::Cpu).unwrap();
        let second = scheduler
            .step(&velocity, timesteps[1], &second_sample)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(second, vec![6.]);
    }

    #[test]
    fn seek_positions_the_next_evaluation() {
        let mut scheduler = H3Scheduler::new(1.).unwrap();
        let timesteps = scheduler.set_timesteps(5).unwrap().to_vec();
        scheduler.seek(2).unwrap();
        assert_eq!(scheduler.step_index(), Some(2));

        let sample = Tensor::new(&[2f32], &Device::Cpu).unwrap();
        let velocity = Tensor::new(&[4f32], &Device::Cpu).unwrap();
        scheduler.step(&velocity, timesteps[2], &sample).unwrap();
        assert_eq!(scheduler.step_index(), Some(3));
    }

    #[test]
    fn seek_rejects_missing_schedule_and_non_evaluation_indices() {
        let mut scheduler = H3Scheduler::new(1.).unwrap();
        let error = scheduler.seek(0).unwrap_err().to_string();
        assert!(error.contains("no timesteps"));

        scheduler.set_timesteps(4).unwrap();
        let evaluation_count = scheduler.timesteps().len();
        let error = scheduler.seek(evaluation_count).unwrap_err().to_string();
        assert!(error.contains("out of range"));
        assert_eq!(scheduler.step_index(), None);
    }

    #[test]
    fn positioned_scheduler_rejects_a_mismatched_timestep() {
        let mut scheduler = H3Scheduler::new(1.).unwrap();
        let timesteps = scheduler.set_timesteps(5).unwrap().to_vec();
        scheduler.seek(2).unwrap();
        let sample = Tensor::new(&[2f32], &Device::Cpu).unwrap();
        let velocity = Tensor::new(&[4f32], &Device::Cpu).unwrap();

        let error = scheduler
            .step(&velocity, timesteps[1], &sample)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not match scheduler evaluation index 2"));
        assert_eq!(scheduler.step_index(), Some(2));
    }

    #[test]
    fn first_step_still_auto_locates_its_timestep() {
        let mut scheduler = H3Scheduler::new(1.).unwrap();
        let timesteps = scheduler.set_timesteps(5).unwrap().to_vec();
        let sample = Tensor::new(&[2f32], &Device::Cpu).unwrap();
        let velocity = Tensor::new(&[4f32], &Device::Cpu).unwrap();

        scheduler.step(&velocity, timesteps[2], &sample).unwrap();
        assert_eq!(scheduler.step_index(), Some(3));
    }

    #[test]
    fn scale_noise_uses_clean_at_t_one() {
        let scheduler = H3Scheduler::new(3.).unwrap();
        let sample = Tensor::new(&[2f32], &Device::Cpu).unwrap();
        let noise = Tensor::new(&[10f32], &Device::Cpu).unwrap();
        assert_eq!(
            scheduler
                .scale_noise(&sample, 1., &noise)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.]
        );
    }
}
