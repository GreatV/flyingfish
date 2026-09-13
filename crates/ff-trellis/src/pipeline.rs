//! Text prompt to occupied voxels.
//!
//! This is the first half of `TrellisTextTo3DPipeline`: encode the prompt with
//! CLIP, sample the sparse-structure latent with the flow model, decode it to
//! occupancy logits, and take the coordinates where they are positive. The
//! reference spells that last step `torch.argwhere(decoder(z_s) > 0)`, keeping
//! the batch index and the three spatial axes.
//!
//! Structured-latent sampling and Gaussian output are composed in
//! [`crate::generation`]. This module retains the standalone first stage.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::{CachePolicy, DeviceCache, WeightSource};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::path::Path;

use crate::checkpoint::TrellisCheckpoint;
use crate::clip_text::{ClipTextConfig, ClipTextModel, ClipTokenizer};
use crate::config::{ComponentConfig, ConditionerKind};
use crate::sampler::{FlowEulerGuidanceIntervalSampler, SamplerParameters};
use crate::sparse_structure_decoder::SparseStructureDecoder;
use crate::sparse_structure_flow::SparseStructureFlow;

/// One voxel the structure decoder marked occupied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Voxel {
    /// Which sample of the batch this voxel belongs to.
    pub sample: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// What one run produced, and what it ran.
#[derive(Clone, Debug)]
pub struct StructureOutcome {
    /// Side length of the decoded occupancy grid.
    pub resolution: usize,
    /// Occupied voxels, ordered by sample then by x, y, z.
    pub voxels: Vec<Voxel>,
    /// Voxels per sample, in sample order.
    pub occupancy_per_sample: Vec<usize>,
    /// The schedule that was integrated.
    pub parameters: SamplerParameters,
}

impl StructureOutcome {
    /// Occupied fraction of one sample's grid.
    pub fn density(&self, sample: usize) -> f64 {
        let volume = self.resolution.pow(3) as f64;
        self.occupancy_per_sample
            .get(sample)
            .map_or(0.0, |count| *count as f64 / volume)
    }
}

/// The loaded text-to-structure half of a TRELLIS pipeline.
pub struct TextToStructure {
    tokenizer: ClipTokenizer,
    text: ClipTextModel,
    flow: SparseStructureFlow,
    decoder: SparseStructureDecoder,
    sampler: FlowEulerGuidanceIntervalSampler,
    parameters: SamplerParameters,
    device: Device,
}

impl TextToStructure {
    /// Load every component a text pipeline needs.
    ///
    /// `checkpoint` is a published TRELLIS text checkpoint; `conditioner` is
    /// the CLIP checkpoint its `pipeline.json` names, which the TRELLIS
    /// checkpoint does not contain.
    pub fn load(
        checkpoint: &Path,
        conditioner: &Path,
        device: &Device,
        models_root: Option<&Path>,
    ) -> Result<Self> {
        let checkpoint = TrellisCheckpoint::open(checkpoint, models_root)?;
        let sampler_spec = checkpoint.pipeline.sparse_structure_sampler()?;
        let (steps, cfg_strength, cfg_interval, rescale_t) = sampler_spec.parameters()?;
        let parameters = SamplerParameters {
            steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        };
        parameters.validate()?;

        let expected = checkpoint
            .pipeline
            .conditioner()
            .context("pipeline names no conditioning model")?;
        anyhow::ensure!(
            expected.kind == ConditionerKind::Text,
            "{} is not a text pipeline: it conditions on {}",
            checkpoint.root.display(),
            expected.model
        );
        anyhow::ensure!(
            expected.model == "openai/clip-vit-large-patch14",
            "unsupported TRELLIS text conditioner {}",
            expected.model
        );
        validate_clip_conditioner(conditioner).with_context(|| {
            format!(
                "this pipeline conditions on {} but {} is not a compatible CLIP checkpoint",
                expected.model,
                conditioner.display()
            )
        })?;

        let flow = load_component::<SparseStructureFlow>(
            &checkpoint,
            "sparse_structure_flow_model",
            device,
        )?;
        let decoder = load_decoder(&checkpoint, device)?;
        Ok(Self {
            tokenizer: ClipTokenizer::open(
                conditioner,
                ClipTextConfig::VIT_LARGE_PATCH14.max_position_embeddings,
            )?,
            text: ClipTextModel::open(conditioner, ClipTextConfig::VIT_LARGE_PATCH14, device)?,
            flow,
            decoder,
            sampler: FlowEulerGuidanceIntervalSampler::new(sampler_spec.sigma_min()?),
            parameters,
            device: device.clone(),
        })
    }

    /// The schedule this pipeline will integrate.
    pub fn parameters(&self) -> SamplerParameters {
        self.parameters
    }

    /// Generate occupied voxels for one prompt.
    ///
    /// The negative conditioning is the empty prompt, as
    /// `_init_text_cond_model` computes it once and reuses it.
    pub fn generate(
        &self,
        prompt: &str,
        samples: usize,
        seed: u64,
        parameters: Option<SamplerParameters>,
    ) -> Result<StructureOutcome> {
        anyhow::ensure!(samples > 0, "a run needs at least one sample");
        let parameters = match parameters {
            Some(parameters) => {
                parameters.validate()?;
                parameters
            }
            None => self.parameters,
        };

        let cond = self.encode(prompt, samples)?;
        let negative = self.encode("", samples)?;
        self.generate_conditioned(&cond, &negative, samples, seed, parameters)
    }

    /// Reuse the same encoded conditions for structure and SLat sampling.
    pub(crate) fn generate_conditioned(
        &self,
        cond: &Tensor,
        negative: &Tensor,
        samples: usize,
        seed: u64,
        parameters: SamplerParameters,
    ) -> Result<StructureOutcome> {
        parameters.validate()?;
        anyhow::ensure!(
            samples > 0 && cond.dim(0)? == samples && negative.dims() == cond.dims(),
            "conditioning batch does not match structure samples"
        );
        let args = self.flow.args();
        let resolution = args.resolution;
        let shape = [
            samples,
            args.in_channels,
            resolution,
            resolution,
            resolution,
        ];
        let noise = if self.device.is_cpu() {
            cpu_noise(&shape, seed, 0)?
        } else {
            self.device.set_seed(seed)?;
            Tensor::randn(0f32, 1f32, shape.as_slice(), &self.device)?
        };
        let latent =
            self.sampler
                .sample(&self.flow, &noise, cond, negative, parameters, &self.device)?;
        let logits = self.decoder.forward(&latent)?;
        let voxels = occupied_voxels(&logits)?;
        let mut occupancy_per_sample = vec![0usize; samples];
        for voxel in &voxels {
            if let Some(count) = occupancy_per_sample.get_mut(voxel.sample as usize) {
                *count += 1;
            }
        }
        Ok(StructureOutcome {
            resolution: logits.dim(2)?,
            voxels,
            occupancy_per_sample,
            parameters,
        })
    }

    /// CLIP's `last_hidden_state`, repeated to the batch the sampler needs.
    pub(crate) fn encode(&self, prompt: &str, samples: usize) -> Result<Tensor> {
        let tokens = self.tokenizer.encode(&[prompt], &self.device)?;
        let encoded = self.text.forward(&tokens)?;
        if samples == 1 {
            return Ok(encoded);
        }
        Tensor::cat(&vec![&encoded; samples], 0).map_err(Into::into)
    }

    pub(crate) fn noise_elements(&self) -> usize {
        let args = self.flow.args();
        args.in_channels * args.resolution.pow(3)
    }
}

pub(crate) fn cpu_noise(shape: &[usize], seed: u64, skip: usize) -> Result<Tensor> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .context("noise shape overflow")?;
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..skip {
        let _: f32 = StandardNormal.sample(&mut rng);
    }
    let values = (0..count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
}

fn validate_clip_conditioner(directory: &Path) -> Result<()> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let text = match config["architectures"][0].as_str() {
        Some("CLIPModel") => &config["text_config"],
        Some("CLIPTextModel") => &config,
        _ => anyhow::bail!("conditioner must declare CLIPModel or CLIPTextModel"),
    };
    let expected = ClipTextConfig::VIT_LARGE_PATCH14;
    for (name, value) in [
        ("hidden_size", expected.hidden_size),
        ("intermediate_size", expected.intermediate_size),
        ("num_hidden_layers", expected.num_hidden_layers),
        ("num_attention_heads", expected.num_attention_heads),
        ("max_position_embeddings", expected.max_position_embeddings),
        ("vocab_size", expected.vocab_size),
    ] {
        anyhow::ensure!(
            text[name].as_u64() == Some(value as u64),
            "incompatible CLIP {name}"
        );
    }
    anyhow::ensure!(
        text["hidden_act"] == "quick_gelu" && text["layer_norm_eps"] == 1e-5,
        "incompatible CLIP activation or normalization"
    );
    Ok(())
}

/// Coordinates where the decoder's logits are positive.
pub(crate) fn occupied_voxels(logits: &Tensor) -> Result<Vec<Voxel>> {
    let (batch, channels, depth, height, width) = logits.dims5()?;
    anyhow::ensure!(
        channels == 1,
        "occupancy logits should have one channel, got {channels}"
    );
    let values = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut voxels = Vec::new();
    for (index, value) in values.iter().enumerate() {
        anyhow::ensure!(
            value.is_finite(),
            "occupancy logits contain non-finite values"
        );
        if *value <= 0.0 {
            continue;
        }
        let within = index % (depth * height * width);
        let sample = index / (depth * height * width);
        voxels.push(Voxel {
            sample: sample as u32,
            x: (within / (height * width)) as u32,
            y: (within / width % height) as u32,
            z: (within % width) as u32,
        });
    }
    debug_assert!(batch * depth * height * width == values.len());
    Ok(voxels)
}

/// Build an eagerly loaded component: `from_component` copies every tensor it
/// needs, so the store is opened without a residency budget and dropped here.
fn load_component<T: FromComponent>(
    checkpoint: &TrellisCheckpoint,
    role: &str,
    device: &Device,
) -> Result<T> {
    let component = checkpoint
        .components
        .iter()
        .find(|component| component.role == role)
        .with_context(|| format!("pipeline names no {role}"))?;
    let weights = component.open_weights(
        WeightSource::Mmap,
        CachePolicy::new(1),
        DeviceCache::disabled(),
    )?;
    T::from_component(&component.config, &weights, device)
        .with_context(|| format!("failed to load {role} from {}", component.reference))
}

/// The decoder is copied into owned layers, so its store is opened without a
/// residency budget and dropped here.
pub(crate) fn load_decoder(
    checkpoint: &TrellisCheckpoint,
    device: &Device,
) -> Result<SparseStructureDecoder> {
    let component = checkpoint
        .components
        .iter()
        .find(|component| component.role == "sparse_structure_decoder")
        .context("pipeline names no sparse_structure_decoder")?;
    let ComponentConfig::SparseStructureDecoder(args) = &component.config else {
        anyhow::bail!(
            "{} is a {}, not a SparseStructureDecoder",
            component.reference,
            component.config.name()
        )
    };
    let weights = component.open_weights(
        WeightSource::Mmap,
        CachePolicy::new(1),
        DeviceCache::disabled(),
    )?;
    SparseStructureDecoder::load(args, &weights, device)
        .with_context(|| format!("failed to load the decoder from {}", component.reference))
}

/// Build a component from the configuration its sidecar carries.
trait FromComponent: Sized {
    fn from_component(
        config: &ComponentConfig,
        weights: &ff_core::weights::ModelWeights,
        device: &Device,
    ) -> Result<Self>;
}

impl FromComponent for SparseStructureFlow {
    fn from_component(
        config: &ComponentConfig,
        weights: &ff_core::weights::ModelWeights,
        device: &Device,
    ) -> Result<Self> {
        let ComponentConfig::SparseStructureFlowModel(args) = config else {
            anyhow::bail!(
                "expected a SparseStructureFlowModel, got a {}",
                config.name()
            )
        };
        Self::load(args, weights, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioner_identity_does_not_depend_on_directory_name() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("my encoder");
        std::fs::create_dir(&directory).unwrap();
        let mut config = serde_json::json!({"architectures":["CLIPModel"],"text_config":{
            "hidden_size":768,"intermediate_size":3072,"num_hidden_layers":12,"num_attention_heads":12,
            "max_position_embeddings":77,"vocab_size":49408,"hidden_act":"quick_gelu","layer_norm_eps":1e-5}});
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        assert!(validate_clip_conditioner(&directory).is_ok());
        config["text_config"]["hidden_size"] = serde_json::json!(512);
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        assert!(validate_clip_conditioner(&directory).is_err());
    }

    #[test]
    fn cpu_noise_is_seeded_and_can_continue_the_same_stream() {
        let full = cpu_noise(&[9], 42, 0).unwrap().to_vec1::<f32>().unwrap();
        let first = cpu_noise(&[4], 42, 0).unwrap().to_vec1::<f32>().unwrap();
        let second = cpu_noise(&[5], 42, 4).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(full, [first, second].concat());
        assert_ne!(
            full,
            cpu_noise(&[9], 43, 0).unwrap().to_vec1::<f32>().unwrap()
        );
        let bad = Tensor::from_vec(vec![f32::NAN], (1, 1, 1, 1, 1), &Device::Cpu).unwrap();
        assert!(occupied_voxels(&bad).is_err());
    }

    /// Index of one voxel in a `side`-cubed grid, laid out as the tensor is:
    /// sample-major, then x, then y, then z.
    fn index(side: usize, voxel: Voxel) -> usize {
        let within = (voxel.x as usize * side + voxel.y as usize) * side + voxel.z as usize;
        voxel.sample as usize * side.pow(3) + within
    }

    /// The occupancy scan reads the grid in the reference's axis order, which
    /// is what makes a coordinate mean the same thing to both.
    #[test]
    fn occupancy_reads_the_grid_in_axis_order() {
        let device = Device::Cpu;
        let occupied = Voxel {
            sample: 0,
            x: 1,
            y: 0,
            z: 1,
        };
        let mut values = vec![-1.0f32; 8];
        values[index(2, occupied)] = 1.0;
        let logits = Tensor::from_vec(values, (1, 1, 2, 2, 2), &device).unwrap();
        assert_eq!(occupied_voxels(&logits).unwrap(), vec![occupied]);
    }

    /// Voxels carry the sample they came from, so a batched run stays separable.
    #[test]
    fn occupancy_separates_the_batch() {
        let device = Device::Cpu;
        let first = Voxel {
            sample: 0,
            x: 0,
            y: 0,
            z: 0,
        };
        let second = Voxel {
            sample: 1,
            x: 1,
            y: 1,
            z: 1,
        };
        let mut values = vec![-1.0f32; 16];
        values[index(2, first)] = 1.0;
        values[index(2, second)] = 1.0;
        let logits = Tensor::from_vec(values, (2, 1, 2, 2, 2), &device).unwrap();
        assert_eq!(occupied_voxels(&logits).unwrap(), vec![first, second]);
    }

    /// A grid with no positive logit produces nothing rather than failing: an
    /// empty structure is a real outcome for a prompt the model cannot place.
    #[test]
    fn an_empty_grid_produces_no_voxels() {
        let device = Device::Cpu;
        let logits = Tensor::full(-1.0f32, (1, 1, 4, 4, 4), &device).unwrap();
        assert!(occupied_voxels(&logits).unwrap().is_empty());
    }

    #[test]
    fn density_is_the_occupied_fraction_of_the_grid() {
        let outcome = StructureOutcome {
            resolution: 4,
            voxels: Vec::new(),
            occupancy_per_sample: vec![8, 0],
            parameters: SamplerParameters {
                steps: 1,
                cfg_strength: 0.0,
                cfg_interval: (0.0, 1.0),
                rescale_t: 1.0,
            },
        };
        assert!((outcome.density(0) - 8.0 / 64.0).abs() < 1e-12);
        assert_eq!(outcome.density(1), 0.0);
        assert_eq!(outcome.density(7), 0.0);
    }
}
