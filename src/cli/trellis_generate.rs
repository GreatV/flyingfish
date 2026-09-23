use super::DeviceCacheArgs;
use super::device_parse::parse_device_single;
use super::output_hygiene::{ensure_new_output, resolve_output_outside_model};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, safetensors};
use flyingfish::runtime::artifact::ArtifactStaging;
use flyingfish::runtime::weights::DeviceCache;
use flyingfish::trellis::checkpoint::TrellisCheckpoint;
use flyingfish::trellis::generation::{self, GenerationOptions};
use flyingfish::trellis::sparse::Grid;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, clap::Args)]
pub(super) struct GenerateArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    conditioner: PathBuf,
    #[arg(long)]
    models_root: Option<PathBuf>,
    #[arg(long, conflicts_with = "image", required_unless_present = "image")]
    prompt: Option<String>,
    #[arg(
        long,
        conflicts_with = "prompt",
        help = "Prepared PNG: 518x518 for TRELLIS-1, 512x512 for TRELLIS.2"
    )]
    image: Option<PathBuf>,
    #[arg(
        long,
        help = "TRELLIS.2 direct output resolution; currently requires 512"
    )]
    resolution: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    structure_steps: Option<usize>,
    #[arg(long)]
    slat_steps: Option<usize>,
    #[arg(long, default_value_t = 128)]
    attention_query_chunk_size: usize,
    #[arg(long, default_value_t = 256)]
    voxel_chunk_size: usize,
    #[arg(long, default_value = "auto")]
    device: String,
    #[command(flatten)]
    device_cache: DeviceCacheArgs,
    #[arg(long)]
    output: PathBuf,
    #[arg(
        long,
        help = "Also save denormalized SLat features and coordinates for later decoding"
    )]
    latents_output: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(super) struct DecodeArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    models_root: Option<PathBuf>,
    #[arg(long)]
    inputs: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "auto")]
    device: String,
    #[command(flatten)]
    device_cache: DeviceCacheArgs,
    #[arg(long, default_value_t = 128)]
    attention_query_chunk_size: usize,
}

fn output_path(path: &Path, checkpoint: &TrellisCheckpoint) -> Result<PathBuf> {
    let path = resolve_output_outside_model(path, &checkpoint.root)?;
    for component in &checkpoint.components {
        resolve_output_outside_model(&path, &component.directory)?;
    }
    ensure_new_output(&path, "TRELLIS output")?;
    Ok(path)
}

fn read_prepared_image(path: &Path, size: usize) -> Result<Tensor> {
    let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    anyhow::ensure!(
        reader.info().width as usize == size && reader.info().height as usize == size,
        "image input must be prepared as a {size}x{size} PNG"
    );
    let mut bytes = vec![0u8; reader.output_buffer_size().context("PNG size overflow")?];
    let info = reader.next_frame(&mut bytes)?;
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => bail!("TRELLIS image must be RGB or RGBA"),
    };
    let mut values = vec![0f32; 3 * size * size];
    for pixel in 0..size * size {
        let alpha = if channels == 4 {
            bytes[pixel * channels + 3] as f32 / 255.
        } else {
            1.
        };
        for channel in 0..3 {
            values[channel * size * size + pixel] =
                bytes[pixel * channels + channel] as f32 / 255. * alpha;
        }
    }
    Ok(Tensor::from_vec(values, (1, 3, size, size), &Device::Cpu)?)
}

fn publish_cloud(
    cloud: &flyingfish::trellis::gaussian::GaussianCloud,
    staging: ArtifactStaging,
) -> Result<()> {
    let mut writer = BufWriter::new(std::fs::File::create(staging.producer_path())?);
    cloud.write_ply(&mut writer)?;
    writer.flush()?;
    drop(writer);
    staging.publish()?;
    Ok(())
}

pub(super) fn generate(args: GenerateArgs) -> Result<()> {
    let checkpoint = TrellisCheckpoint::open(&args.model, args.models_root.as_deref())?;
    let output = output_path(&args.output, &checkpoint)?;
    resolve_output_outside_model(&output, &args.conditioner)?;
    let latents_output = args
        .latents_output
        .as_ref()
        .map(|p| output_path(p, &checkpoint))
        .transpose()?;
    if let Some(path) = &latents_output {
        resolve_output_outside_model(path, &args.conditioner)?;
        anyhow::ensure!(path != &output, "PLY and latent output paths must differ");
    }
    let staging = ArtifactStaging::new_for_path_producer(&output)?;
    let latent_staging = latents_output
        .as_ref()
        .map(ArtifactStaging::new_for_path_producer)
        .transpose()?;
    let started = Instant::now();
    let device = parse_device_single(&args.device)?;
    let mut options = GenerationOptions {
        seed: args.seed,
        structure_steps: args.structure_steps,
        slat_steps: args.slat_steps,
        query_chunk: args.attention_query_chunk_size,
        voxel_chunk: args.voxel_chunk_size,
        device_cache: DeviceCache::disabled(),
    };
    let demands =
        generation::residency_demands(&args.model, args.models_root.as_deref(), &device, &options)?;
    options.device_cache = super::resource::decide_auto_residency_with_required_memory(
        &demands,
        &device,
        args.device_cache,
        0,
        0,
        // discrete-only adapters share nothing.
        1,
    )?;
    let progress = |stage: &str, done: usize, total: usize| {
        eprintln!(
            "{stage}: {done}/{total} ({:.1}s)",
            started.elapsed().as_secs_f64()
        )
    };
    if checkpoint.pipeline.name == "Trellis2ImageTo3DPipeline" {
        anyhow::ensure!(
            args.resolution == Some(512),
            "TRELLIS.2 currently requires explicit --resolution 512"
        );
        anyhow::ensure!(
            args.latents_output.is_none(),
            "intermediate latent export is only available for TRELLIS-1"
        );
        let image = args.image.as_ref().context("TRELLIS.2 requires --image")?;
        let mesh = flyingfish::trellis::trellis2::generate_512(
            &checkpoint,
            &args.conditioner,
            &read_prepared_image(image, 512)?,
            &device,
            &options,
            progress,
        )?;
        let mut writer = BufWriter::new(std::fs::File::create(staging.producer_path())?);
        mesh.write_ply(&mut writer)?;
        writer.flush()?;
        drop(writer);
        staging.publish()?;
        println!(
            "saved {}: {} vertices, {} triangles, {:.1}s",
            output.display(),
            mesh.vertices.len(),
            mesh.triangles.len(),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    anyhow::ensure!(
        args.resolution.is_none(),
        "--resolution only applies to TRELLIS.2"
    );
    let result = match (&args.prompt, &args.image) {
        (Some(prompt), None) => generation::generate_text(
            &args.model,
            &args.conditioner,
            args.models_root.as_deref(),
            &device,
            prompt,
            &options,
            progress,
        )?,
        (None, Some(image)) => generation::generate_image(
            &args.model,
            &args.conditioner,
            args.models_root.as_deref(),
            &device,
            &read_prepared_image(image, 518)?,
            &options,
            progress,
        )?,
        _ => bail!("provide either --prompt or --image"),
    };
    if let Some(staging) = latent_staging {
        let rows = result.grid.coords.len();
        let coords = result
            .grid
            .coords
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let tensors = HashMap::from([
            (
                "voxel_coords".to_string(),
                Tensor::from_vec(coords, (rows, 4), &Device::Cpu)?,
            ),
            ("slat_latents".to_string(), result.latents.clone()),
            (
                "trellis_slat_schema".to_string(),
                Tensor::new(&[1u32], &Device::Cpu)?,
            ),
            (
                "trellis_resolution".to_string(),
                Tensor::new(&[u32::try_from(result.resolution)?], &Device::Cpu)?,
            ),
        ]);
        safetensors::save(&tensors, staging.producer_path())?;
        staging.publish()?;
    }
    publish_cloud(&result.cloud, staging)?;
    println!(
        "saved {}: {} occupied voxels, {} Gaussians, {:.1}s",
        output.display(),
        result.grid.coords.len(),
        result.cloud.vertices.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

pub(super) fn decode(args: DecodeArgs) -> Result<()> {
    let checkpoint = TrellisCheckpoint::open(&args.model, args.models_root.as_deref())?;
    let output = output_path(&args.output, &checkpoint)?;
    let staging = ArtifactStaging::new_for_path_producer(&output)?;
    let mut values = safetensors::load(&args.inputs, &Device::Cpu)?;
    let schema = values
        .remove("trellis_slat_schema")
        .context("missing TRELLIS latent schema")?
        .to_vec1::<u32>()?;
    anyhow::ensure!(schema == [1], "unsupported TRELLIS latent schema");
    let resolution = values
        .remove("trellis_resolution")
        .context("missing TRELLIS resolution")?
        .to_vec1::<u32>()?;
    let component = checkpoint
        .components
        .iter()
        .find(|c| c.role == "slat_decoder_gs")
        .context("missing Gaussian decoder")?;
    let flyingfish::trellis::config::ComponentConfig::SLatGaussianDecoder(config) =
        &component.config
    else {
        bail!("wrong Gaussian decoder class")
    };
    anyhow::ensure!(
        resolution == [u32::try_from(config.resolution)?],
        "latent/decoder resolution mismatch"
    );
    let coords = values
        .remove("voxel_coords")
        .context("missing voxel coordinates")?
        .to_vec2::<u32>()?;
    let coords = coords
        .into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|_| anyhow::anyhow!("coordinates must have four columns"))
        })
        .collect::<Result<Vec<[u32; 4]>>>()?;
    let grid = Grid::new(coords)?;
    let latents = values
        .remove("slat_latents")
        .context("missing structured latents")?;
    let device = parse_device_single(&args.device)?;
    let demands = generation::residency_demands(
        &args.model,
        args.models_root.as_deref(),
        &device,
        &GenerationOptions::default(),
    )?;
    let cloud = generation::decode_gaussians(
        &checkpoint,
        &device,
        &grid,
        &latents,
        args.attention_query_chunk_size,
        super::resource::decide_auto_residency_with_required_memory(
            &demands,
            &device,
            args.device_cache,
            0,
            0,
            // discrete-only adapters share nothing.
            1,
        )?,
    )?;
    publish_cloud(&cloud, staging)?;
    println!(
        "saved {}: {} Gaussians",
        output.display(),
        cloud.vertices.len()
    );
    Ok(())
}
