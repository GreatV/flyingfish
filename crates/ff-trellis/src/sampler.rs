//! Flow-matching Euler sampling with classifier-free guidance over an interval.
//!
//! Transcribed from `microsoft/TRELLIS`,
//! `trellis/pipelines/samplers/flow_euler.py` and the two guidance mixins it
//! composes. Three details decide the result and are easy to get wrong, so they
//! are spelled out here:
//!
//! - the timestep sequence is `linspace(1, 0, steps + 1)` *rescaled* by
//!   `rescale_t * t / (1 + (rescale_t - 1) * t)`, and the guidance interval is
//!   tested against the rescaled value, not the raw one;
//! - the model is called with `1000 * t`, not `t`;
//! - the Euler update is `x - (t - t_prev) * v`, taken on the guided velocity.

use anyhow::Result;
use candle_core::{Device, Tensor};

/// A model that predicts a velocity field for the sampler to integrate.
pub trait VelocityModel {
    /// `x` is the current latent, `timesteps` is one value per batch element,
    /// and `cond` is the conditioning the model cross-attends to.
    fn velocity(&self, x: &Tensor, timesteps: &Tensor, cond: &Tensor) -> Result<Tensor>;
}

/// What a `pipeline.json` sampler block specifies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParameters {
    pub steps: usize,
    pub cfg_strength: f64,
    /// Guidance is applied only while the rescaled timestep lies within.
    pub cfg_interval: (f64, f64),
    pub rescale_t: f64,
}

impl SamplerParameters {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.steps > 0, "a sampler needs at least one step");
        anyhow::ensure!(
            self.rescale_t.is_finite() && self.rescale_t > 0.0,
            "rescale_t must be positive, got {}",
            self.rescale_t
        );
        anyhow::ensure!(
            self.cfg_interval.0.is_finite()
                && self.cfg_interval.1.is_finite()
                && self.cfg_interval.0 <= self.cfg_interval.1,
            "guidance interval {:?} is empty",
            self.cfg_interval
        );
        anyhow::ensure!(
            self.cfg_strength.is_finite(),
            "guidance strength must be finite"
        );
        Ok(())
    }

    /// The rescaled timestep sequence, longest first, with `steps + 1` entries.
    ///
    /// `rescale_t = 1` leaves the linear sequence untouched; larger values bend
    /// it towards zero, spending more steps near the clean end.
    pub fn timesteps(&self) -> Vec<f64> {
        (0..=self.steps)
            .map(|index| {
                let linear = 1.0 - index as f64 / self.steps as f64;
                self.rescale_t * linear / (1.0 + (self.rescale_t - 1.0) * linear)
            })
            .collect()
    }

    fn guides_at(&self, t: f64) -> bool {
        self.cfg_interval.0 <= t && t <= self.cfg_interval.1
    }
}

/// `FlowEulerGuidanceIntervalSampler`.
///
/// `sigma_min` is carried because the published pipelines specify it, but it
/// takes part only in the auxiliary `pred_x_0` the reference also returns and
/// the pipelines do not consume; the sampled trajectory does not depend on it.
#[derive(Clone, Copy, Debug)]
pub struct FlowEulerGuidanceIntervalSampler {
    pub sigma_min: f64,
}

impl FlowEulerGuidanceIntervalSampler {
    pub fn new(sigma_min: f64) -> Self {
        Self { sigma_min }
    }

    /// Integrate `noise` down to a sample.
    pub fn sample<M: VelocityModel>(
        &self,
        model: &M,
        noise: &Tensor,
        cond: &Tensor,
        negative_cond: &Tensor,
        parameters: SamplerParameters,
        device: &Device,
    ) -> Result<Tensor> {
        parameters.validate()?;
        let schedule = parameters.timesteps();
        let batch = noise.dim(0)?;
        let mut sample = noise.clone();
        for window in schedule.windows(2) {
            let (t, previous) = (window[0], window[1]);
            let scaled = Tensor::from_vec(vec![(1000.0 * t) as f32; batch], batch, device)?;
            let velocity = if parameters.guides_at(t) {
                let positive = model.velocity(&sample, &scaled, cond)?;
                let negative = model.velocity(&sample, &scaled, negative_cond)?;
                ((positive * (1.0 + parameters.cfg_strength))?
                    - (negative * parameters.cfg_strength)?)?
            } else {
                model.velocity(&sample, &scaled, cond)?
            };
            sample = (sample - (velocity * (t - previous))?)?;
        }
        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule the published text pipelines actually run.
    fn published() -> SamplerParameters {
        SamplerParameters {
            steps: 25,
            cfg_strength: 7.5,
            cfg_interval: (0.5, 0.95),
            rescale_t: 3.0,
        }
    }

    #[test]
    fn the_schedule_starts_at_one_ends_at_zero_and_decreases() {
        let schedule = published().timesteps();
        assert_eq!(schedule.len(), 26);
        assert!((schedule[0] - 1.0).abs() < 1e-12);
        assert!(schedule[25].abs() < 1e-12);
        for window in schedule.windows(2) {
            assert!(window[0] > window[1], "schedule is not decreasing");
        }
    }

    /// `rescale_t = 1` is the identity, which is what makes it a rescale.
    #[test]
    fn a_unit_rescale_leaves_the_linear_schedule_alone() {
        let parameters = SamplerParameters {
            rescale_t: 1.0,
            ..published()
        };
        for (index, t) in parameters.timesteps().iter().enumerate() {
            let linear = 1.0 - index as f64 / 25.0;
            assert!((t - linear).abs() < 1e-12, "step {index}: {t} vs {linear}");
        }
    }

    /// Rescaling above one bends the schedule towards zero, spending more of
    /// the budget near the clean end.
    #[test]
    fn rescaling_moves_the_schedule_towards_the_clean_end() {
        let schedule = published().timesteps();
        for (index, t) in schedule.iter().enumerate().take(25).skip(1) {
            let linear = 1.0 - index as f64 / 25.0;
            assert!(*t > linear, "step {index}: {t} should exceed {linear}");
        }
    }

    #[test]
    fn guidance_applies_only_inside_the_interval() {
        let parameters = published();
        assert!(!parameters.guides_at(0.99));
        assert!(parameters.guides_at(0.95));
        assert!(parameters.guides_at(0.7));
        assert!(parameters.guides_at(0.5));
        assert!(!parameters.guides_at(0.49));
    }

    #[test]
    fn an_empty_interval_or_a_stepless_schedule_is_refused() {
        assert!(
            SamplerParameters {
                steps: 0,
                ..published()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplerParameters {
                cfg_interval: (0.9, 0.1),
                ..published()
            }
            .validate()
            .is_err()
        );
    }

    /// With a velocity that is constant in `x`, Euler integration is exact and
    /// the sample is the noise less the integral, so the guided combination and
    /// the step arithmetic can be checked against a closed form.
    #[test]
    fn euler_integration_of_a_constant_field_matches_its_closed_form() {
        struct Constant {
            positive: f64,
            negative: f64,
        }
        impl VelocityModel for Constant {
            fn velocity(&self, x: &Tensor, _t: &Tensor, cond: &Tensor) -> Result<Tensor> {
                let marker = cond.flatten_all()?.to_vec1::<f32>()?[0];
                let value = if marker > 0.0 {
                    self.positive
                } else {
                    self.negative
                };
                Ok((x.ones_like()? * value)?)
            }
        }

        let device = Device::Cpu;
        let parameters = SamplerParameters {
            steps: 4,
            cfg_strength: 2.0,
            cfg_interval: (0.0, 1.0),
            rescale_t: 1.0,
        };
        let model = Constant {
            positive: 3.0,
            negative: 1.0,
        };
        let noise = Tensor::zeros((1, 2), candle_core::DType::F32, &device).unwrap();
        let cond = Tensor::new(&[[1.0f32]], &device).unwrap();
        let negative = Tensor::new(&[[-1.0f32]], &device).unwrap();
        let sample = FlowEulerGuidanceIntervalSampler::new(1e-5)
            .sample(&model, &noise, &cond, &negative, parameters, &device)
            .unwrap();

        let expected = -7.0f32;
        for value in sample.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!((value - expected).abs() < 1e-5, "{value} != {expected}");
        }
    }
}
