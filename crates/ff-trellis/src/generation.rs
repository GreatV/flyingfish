//! Complete TRELLIS-1 text -> sparse structure -> SLat -> Gaussian pipeline.
use crate::{
    checkpoint::TrellisCheckpoint,
    config::{ComponentConfig, SamplerSpec},
    gaussian::{GaussianCloud, GaussianDecoder},
    pipeline::TextToStructure,
    sampler::SamplerParameters,
    slat_flow::SLatFlow,
    sparse::Grid,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::residency::PhaseResidencyDemand;
use ff_core::weights::{CachePolicy, DeviceCache, WeightSource};
use std::path::Path;
pub(crate) mod memory;

pub struct GenerationOptions {
    pub seed: u64,
    pub structure_steps: Option<usize>,
    pub slat_steps: Option<usize>,
    pub query_chunk: usize,
    pub voxel_chunk: usize,
    /// One device residency budget shared by every store this run opens that
    /// streams weights per use. Eagerly loaded
    /// components do not attach to it. The default streams every read.
    pub device_cache: DeviceCache,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            structure_steps: None,
            slat_steps: None,
            query_chunk: 128,
            voxel_chunk: 256,
            device_cache: DeviceCache::disabled(),
        }
    }
}
pub struct GeneratedGaussians {
    pub grid: Grid,
    /// Denormalized structured latents, ready for any published SLat decoder.
    pub latents: Tensor,
    pub cloud: GaussianCloud,
    pub resolution: usize,
}

/// What one request would claim from the device tier, charged from metadata
/// only.
///
/// Only components that keep their weight store are declared. The structure
/// flow, its decoder, the CLIP text encoder and the DINO conditioners copy
/// every tensor into owned layers at load time and never read the store
/// again, so they have no per-use reads for residency to save.
///
/// A flow model's reuse count is its sampler step count. Classifier-free
/// guidance calls the model twice inside the guidance interval, so the true
/// figure is higher; the step count is a consistent lower bound rather than a
/// per-schedule calculation. Decoders run once per request.
pub fn residency_demands(
    model: &Path,
    models_root: Option<&Path>,
    device: &Device,
    options: &GenerationOptions,
) -> Result<Vec<PhaseResidencyDemand>> {
    let checkpoint = TrellisCheckpoint::open(model, models_root)?;
    let steps = |key: &str, override_steps: Option<usize>| -> Result<u64> {
        residency_sampler_steps(
            &checkpoint.pipeline.name,
            checkpoint
                .pipeline
                .args
                .rest
                .get(key)
                .with_context(|| format!("missing {key}"))?,
            override_steps,
        )
    };
    let roles: Vec<(&str, u64)> = if checkpoint.pipeline.name == "Trellis2ImageTo3DPipeline" {
        vec![
            (
                "sparse_structure_flow_model",
                steps("sparse_structure_sampler", options.structure_steps)?,
            ),
            (
                "shape_slat_flow_model_512",
                steps("shape_slat_sampler", options.slat_steps)?,
            ),
            (
                "tex_slat_flow_model_512",
                steps("tex_slat_sampler", options.slat_steps)?,
            ),
            ("shape_slat_decoder", 1),
            ("tex_slat_decoder", 1),
        ]
    } else {
        vec![
            (
                "slat_flow_model",
                steps("slat_sampler", options.slat_steps)?,
            ),
            ("slat_decoder_gs", 1),
        ]
    };
    let mut demands = Vec::new();
    for (index, (role, reuse)) in roles.into_iter().enumerate() {
        let Some(component) = checkpoint.components.iter().find(|c| c.role == role) else {
            continue;
        };
        let weights = component.open_weights(
            WeightSource::Mmap,
            CachePolicy::new(1),
            DeviceCache::disabled(),
        )?;
        let phase = component.weight_phase(&weights, reuse);
        let mut demand = PhaseResidencyDemand::from_phase(&phase, &weights, device)?;
        let start = u64::try_from(index)?;
        demand.lifetime = Some(start..start + 1);
        demands.push(demand);
    }
    Ok(demands)
}

fn residency_sampler_steps(
    pipeline: &str,
    value: &serde_json::Value,
    override_steps: Option<usize>,
) -> Result<u64> {
    let steps = if pipeline == "Trellis2ImageTo3DPipeline" {
        crate::trellis2_flow::Sampling2::read(value, override_steps)?.steps
    } else {
        let spec: SamplerSpec = serde_json::from_value(value.clone())?;
        override_steps.unwrap_or(spec.parameters()?.0)
    };
    anyhow::ensure!(steps > 0, "residency sampler steps must be positive");
    Ok(u64::try_from(steps)?)
}

pub fn generate_text(
    model: &Path,
    conditioner: &Path,
    models_root: Option<&Path>,
    device: &Device,
    prompt: &str,
    options: &GenerationOptions,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<GeneratedGaussians> {
    let checkpoint = TrellisCheckpoint::open(model, models_root)?;
    anyhow::ensure!(
        checkpoint.pipeline.name == "TrellisTextTo3DPipeline",
        "this command requires a TRELLIS-1 text pipeline"
    );
    anyhow::ensure!(
        options.query_chunk > 0 && options.voxel_chunk > 0,
        "inference chunks must be positive"
    );
    let structure = TextToStructure::load(model, conditioner, device, models_root)?;
    let structure_parameters = SamplerParameters {
        steps: options
            .structure_steps
            .unwrap_or(structure.parameters().steps),
        ..structure.parameters()
    };
    structure_parameters.validate()?;
    progress("structure", 0, structure_parameters.steps);
    let cond = structure.encode(prompt, 1)?;
    let negative = structure.encode("", 1)?;
    let outcome =
        structure.generate_conditioned(&cond, &negative, 1, options.seed, structure_parameters)?;
    progress(
        "structure",
        structure_parameters.steps,
        structure_parameters.steps,
    );
    let consumed_noise = structure.noise_elements();
    drop(structure);
    let grid = Grid::new(
        outcome
            .voxels
            .iter()
            .map(|v| [v.sample, v.x, v.y, v.z])
            .collect(),
    )?;
    finish_slat(
        &checkpoint,
        device,
        options,
        StructuredInput {
            grid,
            resolution: outcome.resolution,
            cond,
            negative,
            consumed_noise,
        },
        &mut progress,
    )
}

struct StructuredInput {
    grid: Grid,
    resolution: usize,
    cond: Tensor,
    negative: Tensor,
    consumed_noise: usize,
}

fn finish_slat(
    checkpoint: &TrellisCheckpoint,
    device: &Device,
    options: &GenerationOptions,
    input: StructuredInput,
    progress: &mut impl FnMut(&str, usize, usize),
) -> Result<GeneratedGaussians> {
    let StructuredInput {
        grid,
        resolution,
        cond,
        negative,
        consumed_noise,
    } = input;
    let component = checkpoint
        .components
        .iter()
        .find(|c| c.role == "slat_flow_model")
        .context("pipeline has no SLat flow")?;
    let ComponentConfig::SLatFlowModel(args) = &component.config else {
        anyhow::bail!("slat_flow_model has the wrong component class")
    };
    let spec: SamplerSpec = serde_json::from_value(
        checkpoint
            .pipeline
            .args
            .rest
            .get("slat_sampler")
            .context("missing SLat sampler")?
            .clone(),
    )?;
    let (steps, cfg_strength, cfg_interval, rescale_t) = spec.parameters()?;
    let parameters = SamplerParameters {
        steps: options.slat_steps.unwrap_or(steps),
        cfg_strength,
        cfg_interval,
        rescale_t,
    };
    parameters.validate()?;
    let normalization = checkpoint
        .pipeline
        .args
        .rest
        .get("slat_normalization")
        .context("missing SLat normalization")?;
    let vector = |name: &str| -> Result<Vec<f32>> {
        let items = normalization
            .get(name)
            .and_then(|v| v.as_array())
            .context("invalid SLat normalization array")?;
        anyhow::ensure!(
            items.len() == args.in_channels,
            "SLat normalization width mismatch"
        );
        items
            .iter()
            .map(|v| {
                let value = v.as_f64().context("normalization value is not a number")? as f32;
                anyhow::ensure!(
                    value.is_finite() && (name != "std" || value > 0.),
                    "invalid SLat normalization value"
                );
                Ok(value)
            })
            .collect()
    };
    let mean = Tensor::from_vec(vector("mean")?, (1, args.in_channels), device)?;
    let std = Tensor::from_vec(vector("std")?, (1, args.in_channels), device)?;
    anyhow::ensure!(
        resolution == args.resolution,
        "structure/SLat resolutions differ"
    );
    let weights = component.open_weights(
        WeightSource::Mmap,
        CachePolicy::new(1),
        options.device_cache.clone(),
    )?;
    if device.is_cuda() && options.device_cache.policy().is_enabled() {
        let reserve = memory::flow(args, &grid, cond.dim(1)?, &weights, device, options)?;
        memory::prepare(&options.device_cache, device, "slat", reserve)?;
    }
    let flow = SLatFlow::new(
        args.clone(),
        weights,
        grid.clone(),
        device,
        options.query_chunk,
        options.voxel_chunk,
    )?;
    let noise = if device.is_cpu() {
        crate::pipeline::cpu_noise(
            &[grid.coords.len(), args.in_channels],
            options.seed,
            consumed_noise,
        )?
    } else {
        Tensor::randn(0f32, 1f32, (grid.coords.len(), args.in_channels), device)?
    };
    let latents = flow
        .sample(&noise, &cond, &negative, parameters, |done, total| {
            progress("slat", done, total)
        })?
        .broadcast_mul(&std)?
        .broadcast_add(&mean)?;
    drop(flow);
    progress("gaussians", 0, 1);
    let cloud = decode_gaussians(
        checkpoint,
        device,
        &grid,
        &latents,
        options.query_chunk,
        options.device_cache.clone(),
    )?;
    progress("gaussians", 1, 1);
    Ok(GeneratedGaussians {
        grid,
        latents: latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
        cloud,
        resolution: args.resolution,
    })
}

pub fn generate_image(
    model: &Path,
    conditioner: &Path,
    models_root: Option<&Path>,
    device: &Device,
    rgb: &Tensor,
    options: &GenerationOptions,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<GeneratedGaussians> {
    let checkpoint = TrellisCheckpoint::open(model, models_root)?;
    anyhow::ensure!(
        checkpoint.pipeline.name == "TrellisImageTo3DPipeline",
        "image generation requires a TRELLIS-1 image pipeline"
    );
    anyhow::ensure!(
        checkpoint
            .pipeline
            .conditioner()
            .is_some_and(|c| c.model == "dinov2_vitl14_reg"),
        "unsupported TRELLIS image conditioner"
    );
    anyhow::ensure!(
        options.query_chunk > 0 && options.voxel_chunk > 0,
        "inference chunks must be positive"
    );
    progress("image_encoder", 0, 1);
    let encoder = crate::dinov2::DinoV2::open(conditioner, device, options.query_chunk)?;
    let cond = encoder.encode(rgb)?;
    let negative = cond.zeros_like()?;
    drop(encoder);
    progress("image_encoder", 1, 1);
    let component = checkpoint
        .components
        .iter()
        .find(|c| c.role == "sparse_structure_flow_model")
        .context("missing structure flow")?;
    let ComponentConfig::SparseStructureFlowModel(args) = &component.config else {
        anyhow::bail!("wrong structure flow class")
    };
    anyhow::ensure!(
        cond.dim(2)? == args.cond_channels,
        "image encoder/structure flow width mismatch"
    );
    let spec = checkpoint.pipeline.sparse_structure_sampler()?;
    let (steps, cfg_strength, cfg_interval, rescale_t) = spec.parameters()?;
    let parameters = SamplerParameters {
        steps: options.structure_steps.unwrap_or(steps),
        cfg_strength,
        cfg_interval,
        rescale_t,
    };
    parameters.validate()?;
    let weights = component.open_weights(
        WeightSource::Mmap,
        CachePolicy::new(1),
        DeviceCache::disabled(),
    )?;
    let flow = crate::sparse_structure_flow::SparseStructureFlow::load(args, &weights, device)?;
    let shape = [
        1,
        args.in_channels,
        args.resolution,
        args.resolution,
        args.resolution,
    ];
    let consumed_noise = shape.iter().product();
    let noise = if device.is_cpu() {
        crate::pipeline::cpu_noise(&shape, options.seed, 0)?
    } else {
        device.set_seed(options.seed)?;
        Tensor::randn(0f32, 1f32, shape.as_slice(), device)?
    };
    progress("structure", 0, parameters.steps);
    let latent = crate::sampler::FlowEulerGuidanceIntervalSampler::new(spec.sigma_min()?)
        .sample(&flow, &noise, &cond, &negative, parameters, device)?;
    drop(flow);
    let decoder = crate::pipeline::load_decoder(&checkpoint, device)?;
    let logits = decoder.forward(&latent)?;
    let resolution = logits.dim(2)?;
    let coords = crate::pipeline::occupied_voxels(&logits)?
        .into_iter()
        .map(|v| [v.sample, v.x, v.y, v.z])
        .collect();
    drop(decoder);
    progress("structure", parameters.steps, parameters.steps);
    finish_slat(
        &checkpoint,
        device,
        options,
        StructuredInput {
            grid: Grid::new(coords)?,
            resolution,
            cond,
            negative,
            consumed_noise,
        },
        &mut progress,
    )
}

pub fn decode_gaussians(
    checkpoint: &TrellisCheckpoint,
    device: &Device,
    grid: &Grid,
    latents: &Tensor,
    query_chunk: usize,
    residency: DeviceCache,
) -> Result<GaussianCloud> {
    let component = checkpoint
        .components
        .iter()
        .find(|c| c.role == "slat_decoder_gs")
        .context("pipeline has no Gaussian decoder")?;
    let ComponentConfig::SLatGaussianDecoder(args) = &component.config else {
        anyhow::bail!("slat_decoder_gs has the wrong component class")
    };
    let weights =
        component.open_weights(WeightSource::Mmap, CachePolicy::new(1), residency.clone())?;
    if device.is_cuda() && residency.policy().is_enabled() {
        let reserve = memory::gaussian(args, grid, &weights, device, query_chunk)?;
        memory::prepare(&residency, device, "gaussians", reserve)?;
    }
    GaussianDecoder::new(args.clone(), weights, device, query_chunk)?
        .decode(grid, &latents.to_device(device)?)
}

#[cfg(test)]
mod residency_tests {
    use super::*;

    #[test]
    fn trellis2_residency_uses_execution_sampler_schema() {
        let sampler = serde_json::json!({
            "name":"FlowEulerGuidanceIntervalSampler", "args":{"sigma_min":1e-5},
            "params":{"steps":12,"guidance_strength":7.5,"guidance_rescale":0.7,
                "guidance_interval":[0.6,1.0],"rescale_t":5.0}
        });
        assert_eq!(
            residency_sampler_steps("Trellis2ImageTo3DPipeline", &sampler, None).unwrap(),
            12
        );
        assert_eq!(
            residency_sampler_steps("Trellis2ImageTo3DPipeline", &sampler, Some(3)).unwrap(),
            3
        );
        assert!(residency_sampler_steps("Trellis2ImageTo3DPipeline", &sampler, Some(0)).is_err());
        assert!(residency_sampler_steps("TrellisImageTo3DPipeline", &sampler, None).is_err());
    }
}
