//! TRELLIS.2's published direct-512 shape/material inference profile.
use crate::{
    checkpoint::{ResolvedComponent, TrellisCheckpoint},
    dinov3::DinoV3,
    generation::GenerationOptions,
    mesh::Mesh,
    sparse::Grid,
    trellis2_flow::{Flow2, Sampling2},
    trellis2_vae::Vae2,
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use ff_core::weights::{CachePolicy, WeightSource};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::Value;
use std::path::Path;

fn component<'a>(checkpoint: &'a TrellisCheckpoint, name: &str) -> Result<&'a ResolvedComponent> {
    checkpoint
        .components
        .iter()
        .find(|c| c.role == name)
        .with_context(|| format!("missing TRELLIS.2 component {name}"))
}
fn config(component: &ResolvedComponent) -> Result<Value> {
    Ok(serde_json::to_value(&component.config)?["args"].clone())
}
fn noise(rows: usize, channels: usize, rng: &mut StdRng, device: &Device) -> Result<Tensor> {
    let n = rows.checked_mul(channels).context("noise size overflow")?;
    Ok(Tensor::from_vec(
        (0..n)
            .map(|_| StandardNormal.sample(rng))
            .collect::<Vec<f32>>(),
        (rows, channels),
        device,
    )?)
}
fn normalization(
    checkpoint: &TrellisCheckpoint,
    name: &str,
    width: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let values = checkpoint
        .pipeline
        .args
        .rest
        .get(name)
        .context("missing TRELLIS.2 normalization")?;
    let read = |key: &str| -> Result<Tensor> {
        let array = values[key]
            .as_array()
            .context("invalid normalization array")?;
        anyhow::ensure!(array.len() == width, "normalization width mismatch");
        let data = array
            .iter()
            .map(|v| {
                let v = v.as_f64().context("invalid normalization value")? as f32;
                anyhow::ensure!(
                    v.is_finite() && (key != "std" || v > 0.),
                    "non-finite or nonpositive normalization value"
                );
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::from_vec(data, (1, width), device)?)
    };
    Ok((read("mean")?, read("std")?))
}

pub fn generate_512(
    checkpoint: &TrellisCheckpoint,
    conditioner: &Path,
    rgb: &Tensor,
    device: &Device,
    options: &GenerationOptions,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<Mesh> {
    anyhow::ensure!(
        checkpoint.pipeline.name == "Trellis2ImageTo3DPipeline" && rgb.dims() == [1, 3, 512, 512],
        "TRELLIS.2 direct-512 requires a prepared 512x512 RGB image"
    );
    anyhow::ensure!(
        options.query_chunk > 0 && options.voxel_chunk > 0,
        "inference chunks must be positive"
    );
    let sampler = |name: &str, steps| {
        Sampling2::read(
            checkpoint
                .pipeline
                .args
                .rest
                .get(name)
                .context("missing sampler")?,
            steps,
        )
    };
    let structure_sampler = sampler("sparse_structure_sampler", options.structure_steps)?;
    let shape_sampler = sampler("shape_slat_sampler", options.slat_steps)?;
    let texture_sampler = sampler("tex_slat_sampler", options.slat_steps)?;
    let mut rng = StdRng::seed_from_u64(options.seed);
    progress("image_encoder", 0, 1);
    let dino = DinoV3::open(conditioner, device, options.query_chunk)?;
    let condition = dino.encode(rgb)?;
    drop(dino);
    anyhow::ensure!(
        condition.dims() == [1, 1029, 1024],
        "TRELLIS.2 needs DINOv3 ViT-L/16 with four register tokens"
    );
    progress("image_encoder", 1, 1);
    let load_flow = |component: &ResolvedComponent, config: &Value, grid: &Grid| -> Result<Flow2> {
        let weights = component.open_weights(
            WeightSource::Mmap,
            CachePolicy::new(1),
            options.device_cache.clone(),
        )?;
        if device.is_cuda() && options.device_cache.policy().is_enabled() {
            let reserve = crate::generation::memory::flow2(
                config,
                grid.coords.len(),
                condition.dim(1)?,
                &weights,
                device,
                options.query_chunk,
            )?;
            crate::generation::memory::prepare(
                &options.device_cache,
                device,
                &component.role,
                reserve,
            )?;
        }
        Flow2::new(config, weights, grid, device, options.query_chunk)
    };
    let ss = component(checkpoint, "sparse_structure_flow_model")?;
    let ss_config = config(ss)?;
    anyhow::ensure!(
        ss_config["resolution"] == 16 && ss_config["in_channels"] == 8,
        "unsupported TRELLIS.2 structure profile"
    );
    let dense_grid = Grid::new(
        (0..16)
            .flat_map(|x| (0..16).flat_map(move |y| (0..16).map(move |z| [0, x, y, z])))
            .collect(),
    )?;
    let flow = load_flow(ss, &ss_config, &dense_grid)?;
    let structure = structure_sampler.sample(
        &flow,
        &noise(4096, 8, &mut rng, device)?,
        &condition,
        None,
        false,
        |done, total| progress("structure_flow", done, total),
    )?;
    drop(flow);
    let latent = structure.t()?.contiguous()?.reshape((1, 8, 16, 16, 16))?;
    let decoder = crate::pipeline::load_decoder(checkpoint, device)?;
    let logits = decoder.forward(&latent)?;
    drop(decoder);
    anyhow::ensure!(
        logits.dims() == [1, 1, 64, 64, 64],
        "unexpected structure decoder output"
    );
    let coords = crate::pipeline::occupied_voxels(&logits)?
        .into_iter()
        .map(|v| [v.sample, v.x, v.y, v.z])
        .collect();
    let grid = Grid::new(coords)?.downsample()?.coarse;
    let shape = component(checkpoint, "shape_slat_flow_model_512")?;
    let shape_config = config(shape)?;
    anyhow::ensure!(
        shape_config["resolution"] == 32
            && shape_config["in_channels"] == 32
            && shape_config["out_channels"] == 32,
        "unsupported 512 shape profile"
    );
    let (shape_mean, shape_std) =
        normalization(checkpoint, "shape_slat_normalization", 32, device)?;
    let flow = load_flow(shape, &shape_config, &grid)?;
    let shape_latent = shape_sampler
        .sample(
            &flow,
            &noise(grid.coords.len(), 32, &mut rng, device)?,
            &condition,
            None,
            true,
            |done, total| progress("shape_flow", done, total),
        )?
        .broadcast_mul(&shape_std)?
        .broadcast_add(&shape_mean)?;
    drop(flow);
    let texture = component(checkpoint, "tex_slat_flow_model_512")?;
    let texture_config = config(texture)?;
    anyhow::ensure!(
        texture_config["in_channels"] == 64 && texture_config["out_channels"] == 32,
        "unsupported texture flow profile"
    );
    let (tex_mean, tex_std) = normalization(checkpoint, "tex_slat_normalization", 32, device)?;
    let normalized_shape = shape_latent
        .broadcast_sub(&shape_mean)?
        .broadcast_div(&shape_std)?;
    let flow = load_flow(texture, &texture_config, &grid)?;
    let texture_latent = texture_sampler
        .sample(
            &flow,
            &noise(grid.coords.len(), 32, &mut rng, device)?,
            &condition,
            Some(&normalized_shape),
            true,
            |done, total| progress("texture_flow", done, total),
        )?
        .broadcast_mul(&tex_std)?
        .broadcast_add(&tex_mean)?;
    drop(flow);
    let shape_decoder = component(checkpoint, "shape_slat_decoder")?;
    let decoder = Vae2::new(
        &config(shape_decoder)?,
        shape_decoder.open_weights(
            WeightSource::Mmap,
            CachePolicy::new(1),
            options.device_cache.clone(),
        )?,
        device,
        options.voxel_chunk,
        7,
    )?;
    let shape_output = decoder.decode(&grid, &shape_latent, None, |done, total| {
        progress("shape_decoder", done, total)
    })?;
    drop(decoder);
    let texture_decoder = component(checkpoint, "tex_slat_decoder")?;
    let decoder = Vae2::new(
        &config(texture_decoder)?,
        texture_decoder.open_weights(
            WeightSource::Mmap,
            CachePolicy::new(1),
            options.device_cache.clone(),
        )?,
        device,
        options.voxel_chunk,
        6,
    )?;
    let texture_output = decoder.decode(
        &grid,
        &texture_latent,
        Some(&shape_output.subdivisions),
        |done, total| progress("texture_decoder", done, total),
    )?;
    anyhow::ensure!(
        shape_output.grid.coords == texture_output.grid.coords,
        "shape and texture voxel coordinates differ"
    );
    let texture = ((texture_output.features * 0.5)? + 0.5)?;
    Mesh::from_dual_grid(&shape_output.grid, &shape_output.features, &texture, 512)
}
