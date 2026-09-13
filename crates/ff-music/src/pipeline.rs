use crate::{Component, acoustic, autoregressive, prompt};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::residency::{PhaseResidencyDemand, WeightPhase};
use ff_core::weights::{CachePolicy, DeviceCache, WeightSource};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::path::Path;
use tokenizers::Tokenizer;

mod memory;
pub use memory::{AcousticMemoryEstimate, PhaseMemoryRelease, RequestMemoryEstimate};

const ACOUSTIC_WINDOW_FRAMES: usize = 200;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub duration_seconds: f64,
    pub steps: usize,
    pub seed: u64,
    pub attention_query_chunk: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            duration_seconds: 8.,
            steps: 30,
            seed: 0,
            attention_query_chunk: 32,
        }
    }
}

impl Options {
    pub fn frames(&self) -> Result<usize> {
        anyhow::ensure!(
            self.duration_seconds.is_finite() && (0.04..=360.).contains(&self.duration_seconds),
            "Music3 duration must be between 0.04 and 360 seconds"
        );
        anyhow::ensure!(
            self.steps > 0 && self.attention_query_chunk > 0,
            "steps and attention chunk must be positive"
        );
        Ok((self.duration_seconds * 25.) as usize)
    }
}

pub struct Music3 {
    language: Component,
    depth: Component,
    condition: Component,
    transformer: Component,
    vocoder: Component,
    tokenizer: Tokenizer,
    residency: DeviceCache,
    memory_releases: Vec<PhaseMemoryRelease>,
}

pub struct Audio {
    /// `[1, 2, samples]`, F32 on the CPU, at the vocoder's native sample rate.
    pub waveform: Tensor,
    pub sample_rate: u32,
    pub generated_frames: usize,
}

impl Music3 {
    fn prepare_request(
        &self,
        caption: &str,
        lyrics: &str,
        options: &Options,
    ) -> Result<(Vec<u32>, usize, usize)> {
        let frames = options.frames()?;
        let (ids, prompt_length) = prompt::tokenize(&self.tokenizer, caption, lyrics)?;
        let maximum = self.language.n("max_position_embeddings")?;
        anyhow::ensure!(
            prompt_length
                .checked_add(frames)
                .is_some_and(|length| length <= maximum),
            "prompt plus music frames exceeds language-model context"
        );
        Ok((ids, prompt_length, frames))
    }

    pub fn open(
        root: &Path,
        device: &Device,
        source: WeightSource,
        cache: CachePolicy,
        residency: DeviceCache,
    ) -> Result<Self> {
        let load = |name, class| {
            Component::open(
                &root.join(name),
                class,
                device,
                source,
                cache,
                residency.clone(),
            )
        };
        let language = load("language_model", "Qwen3ForCausalLM")?;
        let depth = load("rvq_depth_decoder", "MiniMaxMusic3RVQDepthDecoder")?;
        let condition = load("condition_encoder", "MiniMaxMusic3ConditionEncoder")?;
        let transformer = load("transformer", "MiniMaxMusic3Transformer1DModel")?;
        let vocoder = load("vocoder", "MiniMaxMusic3Vocoder")?;
        let scheduler: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join("scheduler/scheduler_config.json"),
        )?)?;
        for (name, expected) in [
            ("num_train_timesteps", serde_json::json!(1)),
            ("invert_sigmas", true.into()),
            ("shift", serde_json::json!(1.0)),
            ("stochastic_sampling", false.into()),
            ("use_dynamic_shifting", false.into()),
            ("use_beta_sigmas", false.into()),
            ("use_exponential_sigmas", false.into()),
            ("use_karras_sigmas", false.into()),
        ] {
            anyhow::ensure!(
                scheduler[name] == expected,
                "unsupported Music3 scheduler {name}"
            );
        }
        anyhow::ensure!(
            scheduler["shift_terminal"].is_null(),
            "unsupported shifted scheduler terminal"
        );
        for (key, value) in [
            ("input_sampling_rate", 24000),
            ("input_hop_length", 960),
            ("output_sampling_rate", 44100),
            ("output_hop_length", 512),
        ] {
            condition.expect(key, serde_json::json!(value))?;
        }
        vocoder.expect("sampling_rate", serde_json::json!(44100))?;
        vocoder.expect("upsampling_ratios", serde_json::json!([8, 8, 4, 2]))?;
        anyhow::ensure!(
            condition.n("out_dim")? == transformer.n("condition_dim")?
                && transformer.n("in_channels")? == vocoder.n("latent_channels")?
                && condition.n("condition_hidden_dim")? == language.n("hidden_size")?
                && condition.n("num_condition_layers")? == depth.n("num_codebooks")?,
            "Music3 component geometry mismatch"
        );
        let tokenizer = Tokenizer::from_file(root.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load Music3 tokenizer: {e}"))?;
        let mut model = Self {
            language,
            depth,
            condition,
            transformer,
            vocoder,
            tokenizer,
            residency: residency.clone(),
            memory_releases: Vec::new(),
        };
        model.configure_device_cache(residency)?;
        Ok(model)
    }

    /// The residency phases of one request:
    /// maximal intervals over which the set of tensors read does not change.
    /// `frames` is the request's audio frame count (`duration_seconds * 25`)
    /// and `steps` its denoise step count; the reuse counts below follow from
    /// the loops in `generate`.
    pub fn weight_phases(&self, frames: usize, steps: usize) -> Vec<WeightPhase> {
        let chunks = chunk_starts(frames).len() as u64;
        vec![
            self.condition.weight_phase("music3.condition", chunks),
            self.language
                .weight_phase("music3.autoregressive.language", frames as u64),
            self.depth.weight_phase(
                "music3.autoregressive.depth",
                (frames as u64).saturating_mul(7),
            ),
            self.transformer
                .weight_phase("music3.denoise", steps as u64 * chunks),
            self.vocoder.weight_phase("music3.vocode", chunks),
        ]
    }

    /// The same phases paired with the store that can charge them, which is
    /// what `PhaseResidencyDemand::from_phase` needs: every name in a phase
    /// must resolve against one checkpoint.
    fn phases_with_stores(&self, frames: usize, steps: usize) -> Vec<(&Component, WeightPhase)> {
        let chunks = chunk_starts(frames).len() as u64;
        vec![
            (
                &self.condition,
                self.condition.weight_phase("music3.condition", chunks),
            ),
            (
                &self.language,
                self.language
                    .weight_phase("music3.autoregressive.language", frames as u64),
            ),
            (
                &self.depth,
                self.depth.weight_phase(
                    "music3.autoregressive.depth",
                    (frames as u64).saturating_mul(7),
                ),
            ),
            (
                &self.transformer,
                self.transformer
                    .weight_phase("music3.denoise", steps as u64 * chunks),
            ),
            (
                &self.vocoder,
                self.vocoder.weight_phase("music3.vocode", chunks),
            ),
        ]
    }

    /// What one request would claim from the device tier, charged from
    /// metadata only. No payload is read, so
    /// this may precede [`Music3::configure_device_cache`].
    pub fn residency_demands(
        &self,
        frames: usize,
        steps: usize,
        device: &Device,
    ) -> Result<Vec<PhaseResidencyDemand>> {
        self.phases_with_stores(frames, steps)
            .into_iter()
            .flat_map(|(component, phase)| {
                component
                    .device_weight_phases(&phase.name, phase.reuse_count)
                    .into_iter()
                    .map(move |phase| {
                        PhaseResidencyDemand::from_phase(&phase, &component.weights, device)
                    })
            })
            .collect()
    }

    /// Attach every component to one shared budget. The ceiling belongs to the
    /// request, not to each of the five checkpoints, so they share one cache.
    /// Must precede payload access.
    pub fn configure_device_cache(&mut self, cache: DeviceCache) -> Result<()> {
        for (component, phase) in [
            (&mut self.condition, "music3.condition"),
            (&mut self.language, "music3.autoregressive.language"),
            (&mut self.depth, "music3.autoregressive.depth"),
            (&mut self.transformer, "music3.denoise"),
            (&mut self.vocoder, "music3.vocode"),
        ] {
            let prioritized = component
                .device_weight_phases(phase, 1)
                .into_iter()
                .filter(|group| cache.phase_selected(phase) || cache.phase_selected(&group.name))
                .flat_map(|group| group.tensors)
                .collect::<Vec<_>>();
            component
                .weights
                .configure_device_cache_with_priority(cache.clone(), prioritized)?;
        }
        self.residency = cache;
        Ok(())
    }

    pub fn device_cache_stats(&self) -> ff_core::weights::DeviceCacheStats {
        self.residency.stats()
    }

    /// Reuse weight stores and their bounded cache; request KV and RNG are fresh.
    pub fn generate(
        &mut self,
        caption: &str,
        lyrics: &str,
        options: &Options,
        mut progress: impl FnMut(&str, usize, usize),
    ) -> Result<Audio> {
        self.generate_cancellable(caption, lyrics, options, |stage, done, total| {
            progress(stage, done, total);
            Ok(())
        })
    }

    /// A callback error cancels at the next stage boundary. Request-local
    /// state is dropped on every exit, so the next request starts cleanly.
    pub fn generate_cancellable(
        &mut self,
        caption: &str,
        lyrics: &str,
        options: &Options,
        progress: impl FnMut(&str, usize, usize) -> Result<()>,
    ) -> Result<Audio> {
        self.memory_releases.clear();
        let result = self.generate_inner(caption, lyrics, options, progress);
        self.activate_weight_stage(None);
        result
    }

    pub fn memory_releases(&self) -> &[PhaseMemoryRelease] {
        &self.memory_releases
    }

    fn activate_weight_stage(&self, stage: Option<&str>) {
        for (component, phase) in [
            (&self.language, "autoregressive"),
            (&self.depth, "autoregressive"),
            (&self.condition, "condition"),
            (&self.transformer, "denoise"),
            (&self.vocoder, "vocoder"),
        ] {
            component
                .weights
                .set_device_priority_active(stage == Some(phase));
        }
    }

    fn generate_inner(
        &mut self,
        caption: &str,
        lyrics: &str,
        options: &Options,
        mut progress: impl FnMut(&str, usize, usize) -> Result<()>,
    ) -> Result<Audio> {
        progress("start", 0, 1)?;
        let (ids, _, frames) = self.prepare_request(caption, lyrics, options)?;
        let mut rng = StdRng::seed_from_u64(options.seed);
        self.residency
            .set_capacity_bytes(self.residency.policy().max_bytes);
        self.activate_weight_stage(Some("autoregressive"));
        let frame_hiddens = autoregressive::generate(
            &self.language,
            &self.depth,
            &ids,
            frames,
            options.attention_query_chunk,
            &mut rng,
            &mut progress,
        )?;
        let generated_frames = frame_hiddens.dim(1)?;
        let starts = chunk_starts(generated_frames);
        let device = self.transformer.device.clone();
        let channels = self.transformer.n("in_channels")?;
        let mut carry: Option<(Tensor, Tensor)> = None;
        let mut waveforms = Vec::new();
        let schedule = timesteps(options.steps);
        for (index, &start) in starts.iter().enumerate() {
            let count = ACOUSTIC_WINDOW_FRAMES.min(generated_frames - start);
            let memory = self.acoustic_memory(count, options.attention_query_chunk)?;
            self.prepare_acoustic_stage("condition", memory.condition_device_bytes)?;
            let mut condition =
                acoustic::condition(&self.condition, &frame_hiddens.narrow(1, start, count)?)?;
            let length = condition.dim(1)?;
            let noise = (0..channels * length)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect::<Vec<f32>>();
            let mut latent = Tensor::from_vec(noise, (1, channels, length), &device)?;
            let overlap = carry
                .as_ref()
                .map_or(0, |(latent, _)| latent.dims()[2].min(length));
            let noise_prompt = latent.narrow(2, 0, overlap)?.contiguous()?;
            if let Some((_, previous_condition)) = &carry {
                condition =
                    replace_prefix(&condition, &previous_condition.narrow(1, 0, overlap)?, 1)?;
            }
            let pair_condition = Tensor::cat(&[&condition, &condition.zeros_like()?], 0)?;
            self.prepare_acoustic_stage("denoise", memory.denoise_device_bytes)?;
            for step in 0..options.steps {
                let time = schedule[step];
                if let Some((previous, _)) = &carry {
                    let blend = ((&noise_prompt * (1. - (1. - 1e-6) * f64::from(time)))?
                        + (previous.narrow(2, 0, overlap)? * f64::from(time))?)?;
                    latent = replace_prefix(&latent, &blend, 2)?;
                }
                let velocity = acoustic::velocity(
                    &self.transformer,
                    &Tensor::cat(&[&latent, &latent], 0)?,
                    &pair_condition,
                    time,
                    options.attention_query_chunk,
                )?;
                let unconditional = velocity.narrow(0, 1, 1)?;
                let guided =
                    (&unconditional + ((velocity.narrow(0, 0, 1)? - &unconditional)? * 1.7)?)?;
                let next = schedule[step + 1];
                latent = (&latent + (guided * f64::from(next - time))?)?;
                progress(
                    "denoise",
                    index * options.steps + step + 1,
                    starts.len() * options.steps,
                )?;
            }
            if let Some((previous, _)) = &carry {
                latent = replace_prefix(&latent, &previous.narrow(2, 0, overlap)?, 2)?;
            }
            let left = length.saturating_sub(344);
            let right = length.saturating_sub(172).max(left);
            carry = Some((
                latent.narrow(2, left, right - left)?.contiguous()?,
                condition.narrow(1, left, right - left)?.contiguous()?,
            ));
            self.prepare_acoustic_stage("vocoder", memory.vocoder_device_bytes)?;
            let waveform =
                self.vocode_with_memory_recovery(&latent, memory.vocoder_device_bytes)?;
            let crop_left = if index == 0 { 0 } else { 86 * 512 };
            let crop_right = if index + 1 == starts.len() {
                0
            } else {
                258 * 512
            };
            let samples = waveform
                .dim(2)?
                .checked_sub(crop_left + crop_right)
                .context("Music3 window is shorter than its overlap crop")?;
            waveforms.push(
                waveform
                    .narrow(2, crop_left, samples)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?,
            );
            progress("vocoder", index + 1, starts.len())?;
        }
        Ok(Audio {
            waveform: Tensor::cat(&waveforms, 2)?,
            // The component's own rate. `open` has already refused a vocoder
            // that does not declare the verified one, so this reads the value
            // from the checkpoint rather than repeating it here.
            sample_rate: u32::try_from(self.vocoder.n("sampling_rate")?)
                .context("Music3 vocoder sample rate exceeds u32")?,
            generated_frames,
        })
    }
}

fn replace_prefix(x: &Tensor, prefix: &Tensor, axis: usize) -> Result<Tensor> {
    let n = prefix.dim(axis)?;
    let tail = x
        .dim(axis)?
        .checked_sub(n)
        .context("overlap prefix exceeds window")?;
    Ok(Tensor::cat(&[prefix, &x.narrow(axis, n, tail)?], axis)?)
}

fn chunk_starts(frames: usize) -> Vec<usize> {
    if frames <= ACOUSTIC_WINDOW_FRAMES {
        vec![0]
    } else {
        (0..frames - 100).step_by(100).collect()
    }
}

fn timesteps(steps: usize) -> Vec<f32> {
    (0..steps)
        .map(|i| 1f32 - (1. - i as f64 / steps as f64) as f32)
        .chain([1.])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_windows_do_not_create_short_trailing_chunks() {
        assert_eq!(chunk_starts(200), [0]);
        assert_eq!(chunk_starts(201), [0, 100]);
        assert_eq!(chunk_starts(300), [0, 100]);
        assert_eq!(chunk_starts(301), [0, 100, 200]);
    }

    #[test]
    fn schedule_preserves_the_reference_f32_inversion_order() {
        assert_eq!(timesteps(1), [0., 1.]);
        let schedule = timesteps(3);
        assert_eq!(schedule, [0., 1f32 - 2f32 / 3., 1f32 - 1f32 / 3., 1.]);
        assert!(schedule.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn overlap_prefix_preserves_the_new_window_tail() {
        let x = Tensor::new(&[[[1f32, 2., 3., 4.]]], &Device::Cpu).unwrap();
        let prefix = Tensor::new(&[[[9f32, 8.]]], &Device::Cpu).unwrap();
        let result = replace_prefix(&x, &prefix, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(result, [9., 8., 3., 4.]);
        assert!(replace_prefix(&prefix, &x, 2).is_err());
    }
}
