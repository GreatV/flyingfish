//! MiniMax-H3 image/video VAEs: tiled ViT decoders and the causal CNN keyframe encoder.
use super::{
    DenseTier, Element, EntryReader, FillCast, FillCheck, OwnedRead, Source, pointer, rows_view,
};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::{Flash, Score},
    linear::LinearBiasMode,
    ops::{self, flat},
    verification::Shape,
    weights::DenseStreaming,
};
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use crate::checkpoint::TensorManifest;
use crate::minimax_h3::{DecodePlan, VAE_PIXEL_MEAN, VAE_PIXEL_STD, VAE_TILE, VaeSpec, VaeTiles};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{path::Path, sync::Arc};

pub(super) struct ImageWork {
    core: TileWork,
    tiles: VaeTiles,
    canvas: [usize; 2],
    grid: CudaSlice<u32>,
    raw: CudaSlice<bf16>,
    planar: CudaSlice<bf16>,
    rgb: CudaSlice<f32>,
}

impl ImageWork {
    pub(super) fn visits(&self) -> usize {
        self.tiles.count().div_ceil(self.core.batch)
    }
}

#[cfg(test)]
enum DecoderRows<'a> {
    Bf16(&'a CudaSlice<bf16>),
    F32(&'a CudaSlice<f32>),
}

#[cfg(test)]
type ImageRows<'a> = (DecoderRows<'a>, usize, usize, usize);

struct ImageDecoder {
    core: VideoDecoder,
    latent_affine: (CudaSlice<f32>, CudaSlice<f32>),
    pixel_affine: (CudaSlice<f32>, CudaSlice<f32>),
    gather: CudaFunction,
    pixels: CudaFunction,
}

impl<T: DeviceRepr + Copy> Linear<T> {
    fn load(
        load: &mut impl FnMut(&str, usize) -> Result<Tensor<T, 1>>,
        base: &str,
        out: usize,
        input: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: {
                let data = load(&format!("{base}.weight"), out * input)?;
                unsafe {
                    Tensor::new(
                        data.ptr,
                        [out as u32, input as u32],
                        [input as u64, 1],
                        data.span_bytes,
                    )?
                }
            },
            bias: load(&format!("{base}.bias"), out)?,
        })
    }
}

impl ImageDecoder {
    fn new(
        device: &Device,
        spec: &VaeSpec,
        component: &Path,
        manifest: &TensorManifest,
    ) -> Result<Self> {
        let (free, total) = device.ctx.mem_get_info()?;
        let bytes = decoder_weight_bytes(manifest);
        ensure!(
            bytes <= device.workspace_budget(free, free, total / 16)?,
            "image VAE decoder weights need {bytes} bytes"
        );
        let mut loader = DecoderWeightLoad::new(component, manifest);
        loader.preload_resident(device, "image-vae")?;
        Self::from_weights(device, spec, loader)
    }

    fn from_weights(
        device: &Device,
        spec: &VaeSpec,
        loader: DecoderWeightLoad<'_>,
    ) -> Result<Self> {
        ensure!(
            spec.out_channels == VAE_PIXEL_STD.len(),
            "image VAE requires three channels"
        );
        let s = &device.stream;
        let vision = cubin::module(s.context(), "vision_ops")?;
        let bytes = decoder_weight_bytes(loader.manifest);
        let tensors = loader
            .manifest
            .entries
            .keys()
            .filter(|name| name.starts_with("decoder.") || name.starts_with("post_quant_conv."))
            .count();
        let core = VideoDecoder::from_weights(device, spec, loader)?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"image_vae_decoder_load":{"tensors":tensors,"weight_bytes":bytes,"dtype":"BF16/F32"}})
            );
        }
        Ok(Self {
            core,
            latent_affine: (
                s.clone_htod(&spec.latents_std)?,
                s.clone_htod(&spec.latents_mean)?,
            ),
            pixel_affine: (
                s.clone_htod(&VAE_PIXEL_STD.map(|x| x as f32))?,
                s.clone_htod(&VAE_PIXEL_MEAN.map(|x| x as f32))?,
            ),
            gather: vision.load_function("tile_rows_f32_to_bf16")?,
            pixels: vision.load_function("planar_bf16_affine_rows_f32")?,
        })
    }

    fn work_size(
        spec: &VaeSpec,
        tiles: &VaeTiles,
        canvas: [usize; 2],
        batch: usize,
    ) -> Result<usize> {
        let shape = (
            1,
            tiles.height / spec.spatial_ratio,
            tiles.width / spec.spatial_ratio,
        );
        let grid_words = 2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2;
        let fixed = 2u128
            * tiles.count() as u128
            * spec.out_channels as u128
            * tiles.height as u128
            * tiles.width as u128
            + 6u128 * canvas[0] as u128 * canvas[1] as u128 * spec.out_channels as u128
            + 4u128 * grid_words as u128;
        Ok(usize::try_from(
            fixed + TileWork::batch_bytes(spec, shape, batch, tiles.count())? as u128,
        )?)
    }

    fn batch(
        spec: &VaeSpec,
        tiles: &VaeTiles,
        canvas: [usize; 2],
        budget: usize,
        grid: usize,
    ) -> Result<(usize, usize)> {
        let tokens = tiles.height / spec.spatial_ratio * (tiles.width / spec.spatial_ratio)
            + spec.registers
            + 1;
        let cap = (i32::MAX as usize / tokens / (2 * spec.ffn))
            .min(grid / tokens / (2 * spec.heads))
            .min(grid.saturating_mul(2048) / tokens / spec.ffn)
            .min(tiles.count());
        for batch in (1..=cap).rev() {
            let bytes = Self::work_size(spec, tiles, canvas, batch)?;
            if bytes <= budget {
                return Ok((batch, bytes));
            }
        }
        anyhow::bail!(
            "image VAE serial tile requires {} bytes and a valid grid; workspace budget {budget}, grid limit {grid}",
            Self::work_size(spec, tiles, canvas, 1)?
        );
    }

    fn work(&self, device: &Device, canvas: [usize; 2], budget: usize) -> Result<ImageWork> {
        Self::work_for(device, &self.core.spec, canvas, budget)
    }

    fn work_for(
        device: &Device,
        spec: &VaeSpec,
        canvas: [usize; 2],
        budget: usize,
    ) -> Result<ImageWork> {
        ensure!(
            canvas
                .into_iter()
                .all(|n| n > 0 && n <= u32::MAX as usize && n.is_multiple_of(spec.spatial_ratio)),
            "image VAE invalid canvas {canvas:?}"
        );
        ensure!(
            canvas[0] as u128 * canvas[1] as u128 * spec.out_channels as u128 <= u32::MAX as u128,
            "image VAE canvas exceeds the layout index range"
        );
        let tiles = VaeTiles::released(spec, canvas[0], canvas[1])?;
        let grid_limit = usize::try_from(device.ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        )?)?;
        let (batch, bytes) = Self::batch(spec, &tiles, canvas, budget, grid_limit)?;
        let s = &device.stream;
        let grid: Vec<u32> = tiles
            .rows
            .0
            .iter()
            .chain(&tiles.rows.1)
            .chain(&tiles.cols.0)
            .chain(&tiles.cols.1)
            .map(|&v| u32::try_from(v))
            .collect::<std::result::Result<_, _>>()?;
        let pixels = spec.out_channels * canvas[0] * canvas[1];
        let raw = tiles.count() * spec.out_channels * tiles.height * tiles.width;
        let tile = (
            1,
            tiles.height / spec.spatial_ratio,
            tiles.width / spec.spatial_ratio,
        );
        let work = ImageWork {
            core: TileWork::with_rope_axes(s, spec, tile, (batch, tiles.count()), 2)?,
            grid: s.clone_htod(&grid)?,
            raw: unsafe { s.alloc(raw)? },
            planar: unsafe { s.alloc(pixels)? },
            rgb: unsafe { s.alloc(pixels)? },
            tiles,
            canvas,
        };
        let actual = work.core.allocated_bytes()
            + 4 * (work.grid.len() + work.rgb.len())
            + 2 * (work.raw.len() + work.planar.len());
        ensure!(
            actual == bytes,
            "image VAE workspace allocation {actual}/{bytes}"
        );
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"image_vae_plan":{
                    "canvas":canvas,"tiles":work.tiles.count(),"tokens":work.core.tokens,
                    "linear_batch":batch,"attention_batch":batch,"bytes":bytes,"workspace_budget":budget
                }})
            );
        }
        Ok(work)
    }

    fn run(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        mut w: ImageWork,
        input: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        self.launch(
            s,
            blas,
            &mut w,
            input,
            #[cfg(test)]
            |_, _, _, _| Ok(()),
        )?;
        s.synchronize()?;
        Ok(w.rgb)
    }

    fn launch(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut ImageWork,
        input: &CudaSlice<f32>,
        #[cfg(test)] mut observe: impl FnMut(&str, usize, usize, ImageRows<'_>) -> Result<()>,
    ) -> Result<()> {
        let spec = &self.core.spec;
        let [height, width] = w.canvas;
        let latent = [height / spec.spatial_ratio, width / spec.spatial_ratio];
        ensure!(
            input.len() == latent[0] * latent[1] * spec.latent_channels,
            "image VAE latent allocation differs from its canvas"
        );
        let tiles = w.tiles.count();
        for first in (0..tiles).step_by(w.core.batch) {
            let active = (tiles - first).min(w.core.batch);
            unsafe {
                s.launch_builder(&self.gather)
                    .arg(&pointer(input, s))
                    .arg(&pointer(&self.latent_affine.0, s))
                    .arg(&pointer(&self.latent_affine.1, s))
                    .arg(&pointer(&w.grid, s))
                    .arg(&pointer(&w.core.latent, s))
                    .arg(&(spec.latent_channels as u32))
                    .arg(&(latent[0] as u32))
                    .arg(&(latent[1] as u32))
                    .arg(&(w.tiles.rows.0.len() as u32))
                    .arg(&(w.tiles.cols.0.len() as u32))
                    .arg(&(spec.spatial_ratio as u32))
                    .arg(&(w.core.height as u32))
                    .arg(&(w.core.width as u32))
                    .arg(&(first as u32))
                    .arg(&(active as u32))
                    .launch(flat(active * w.core.patches * spec.latent_channels))?;
            }
            self.core.project_tiles(
                (s, blas),
                &mut w.core,
                active,
                (&w.raw, first),
                #[cfg(test)]
                |role, rows| observe(role, first, active, rows),
            )?;
        }
        unsafe {
            s.launch_builder(&self.core.kernels.stitch)
                .arg(&pointer(&w.raw, s))
                .arg(&pointer(&w.planar, s))
                .arg(&pointer(&w.grid, s))
                .arg(&(w.tiles.rows.0.len() as u32))
                .arg(&(w.tiles.cols.0.len() as u32))
                .arg(&(spec.out_channels as u32))
                .arg(&(w.tiles.height as u32))
                .arg(&(w.tiles.width as u32))
                .arg(&(height as u32))
                .arg(&(width as u32))
                .launch(flat(w.planar.len()))?;
            s.launch_builder(&self.pixels)
                .arg(&pointer(&w.planar, s))
                .arg(&pointer(&self.pixel_affine.0, s))
                .arg(&pointer(&self.pixel_affine.1, s))
                .arg(&pointer(&w.rgb, s))
                .arg(&(spec.out_channels as u32))
                .arg(&((height * width) as u64))
                .launch(flat(w.rgb.len()))?;
        }
        Ok(())
    }
}

// Kept fixed outside streamed block slots so fused scale2 -> next.norm1 reads blocks[L+1].norm1 before begin(L+1).
const DECODER_RESIDENT_BLOCK_WEIGHT: &str = "norm1.weight";

fn image_decoder_layer_names(manifest: &TensorManifest, depth: usize) -> Vec<Vec<String>> {
    (0..depth)
        .map(|layer| {
            let prefix = format!("decoder.transformer_blocks.{layer}.");
            manifest
                .entries
                .keys()
                .filter(|name| {
                    name.strip_prefix(&prefix)
                        .is_some_and(|field| field != DECODER_RESIDENT_BLOCK_WEIGHT)
                })
                .cloned()
                .collect()
        })
        .collect()
}

pub(super) fn image_decoder_sizes(
    manifest: &TensorManifest,
    depth: usize,
) -> ([Vec<usize>; 2], usize, usize) {
    let order = image_decoder_layer_names(manifest, depth);
    let device: Vec<_> = order
        .iter()
        .map(|names| {
            names
                .iter()
                .map(|name| decoder_tensor_bytes(manifest, name))
                .sum()
        })
        .collect();
    let source = order
        .iter()
        .map(|names| names.iter().map(|name| manifest.entries[name].bytes).sum())
        .collect();
    let fixed = decoder_weight_bytes(manifest) - device.iter().sum::<usize>();
    let largest = order
        .iter()
        .flatten()
        .map(|name| manifest.entries[name].bytes)
        .max()
        .unwrap_or(0);
    ([device, source], fixed, largest)
}

pub(super) struct ImageDecodeStage {
    decoder: ImageDecoder,
    work: ImageWork,
}

impl ImageDecodeStage {
    pub(super) fn minimum_work(spec: &VaeSpec, canvas: [usize; 2]) -> Result<usize> {
        let tiles = VaeTiles::released(spec, canvas[0], canvas[1])?;
        ImageDecoder::work_size(spec, &tiles, canvas, 1)
    }

    pub(super) fn prepare_work(
        device: &Device,
        spec: &VaeSpec,
        canvas: [usize; 2],
        later: usize,
    ) -> Result<ImageWork> {
        let (free, total) = device.ctx.mem_get_info()?;
        let budget = device.workspace_budget(free, free, total / 16)?;
        ImageDecoder::work_for(
            device,
            spec,
            canvas,
            budget
                .checked_sub(later)
                .context("image VAE workspace tier reserve")?,
        )
    }

    pub(super) fn bind(
        device: &Device,
        blas: &mut Blas,
        (dir, spec, manifest): (&Path, &VaeSpec, &TensorManifest),
        work: ImageWork,
        plan: crate::prefill::DensePlacement,
    ) -> Result<Self> {
        let mut loader = DecoderWeightLoad::new(dir, manifest);
        if plan.slot_bytes > 0 {
            let order = image_decoder_layer_names(manifest, spec.layers);
            let slots = plan.slots.clone();
            let mut dense =
                DenseStreaming::with_f32_cast(dir, manifest, plan, device, &order, |name| {
                    decoder_weight_width(name) == 2
                })?;
            let mut requests = Vec::new();
            let mut checks = Vec::new();
            let mut largest = 0usize;
            for (layer, names) in order
                .iter()
                .enumerate()
                .filter(|(layer, _)| slots[*layer].is_some())
            {
                let mut pinned = dense.host_layer(layer)?;
                let (mut source, mut runtime) = (0, 0);
                for name in names {
                    let entry = &manifest.entries[name.as_str()];
                    let end = source + entry.bytes;
                    if let Some(bytes) = &mut pinned {
                        requests.push(OwnedRead::new(
                            name,
                            entry,
                            unsafe { bytes.as_mut_ptr().add(source) },
                            entry.bytes,
                            FillCast::Raw,
                        ));
                        checks.push(FillCheck::finite(
                            layer,
                            name,
                            entry.dtype,
                            source,
                            entry.bytes,
                        ));
                        largest = largest.max(entry.bytes);
                    }
                    loader.offsets.insert(name.clone(), (layer, runtime));
                    source = end;
                    runtime += decoder_tensor_bytes(manifest, name);
                }
                ensure!(
                    pinned.is_none_or(|pinned| pinned.len() == source),
                    "image VAE pinned layer byte count"
                );
            }
            let slot_bytes = dense.plan.slot_bytes;
            let slot_count = slots.iter().flatten().copied().max().map_or(0, |i| i + 1);
            loader.dense = Some(DenseTier::schedule(
                "image-vae",
                dense,
                dir,
                requests,
                checks,
                (slot_count, slot_bytes, largest),
            )?);
        }
        loader.preload_resident(device, "image-vae")?;
        let decoder = ImageDecoder::from_weights(device, spec, loader)?;
        decoder.core.bind(blas, &work.core)?;
        Ok(Self { decoder, work })
    }

    pub(super) fn run(&mut self, device: &Device, blas: &Blas, latent: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            latent.iter().all(|v| v.is_finite()),
            "image VAE input is nonfinite"
        );
        let s = &device.stream;
        let input = s.clone_htod(latent)?;
        self.decoder.launch(
            s,
            blas,
            &mut self.work,
            &input,
            #[cfg(test)]
            |_, _, _, _| Ok(()),
        )?;
        s.synchronize()?;
        let output = s.clone_dtoh(&self.work.rgb)?;
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "image VAE output is nonfinite"
        );
        Ok(output)
    }
}

pub(super) fn image_vae_manifest(
    model: &Path,
) -> Result<(
    std::path::PathBuf,
    crate::minimax_h3::ImageVaeSpec,
    TensorManifest,
)> {
    let component = model.join("vae_diffusers");
    let config = component.join("config.json");
    let spec = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
        &std::fs::read(&config).with_context(|| format!("read {}", config.display()))?,
    )?)?;
    let checkpoint =
        crate::checkpoint::MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
    let manifest = checkpoint.minimax_h3_image_vae_manifest(&spec)?;
    Ok((component, spec, manifest))
}

pub(super) struct ImageEncodeStage {
    encoder: KeyframeEncoder,
    work: EncodeWork,
    tiles: VaeTiles,
    extent: [usize; 2],
}

pub(super) struct ImageEncodeSetup {
    encoder: KeyframeEncoder,
    tiles: VaeTiles,
    extent: [usize; 2],
    sizing: (usize, [usize; 2]),
    pub(super) transient_bytes: usize,
}

impl ImageEncodeSetup {
    pub(super) fn new(
        device: &Device,
        (dir, spec, manifest): (&Path, &VaeSpec, &TensorManifest),
        extent: [usize; 2],
    ) -> Result<Self> {
        let [height, width] = extent;
        let pixels = height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(spec.in_channels))
            .context("resident image encoder input extent overflow")?;
        let noise = (height / spec.spatial_ratio)
            .checked_mul(width / spec.spatial_ratio)
            .and_then(|n| n.checked_mul(spec.latent_channels))
            .context("resident image encoder noise extent overflow")?;
        let (frame, output) = image_encode_shape(spec, extent, pixels, noise)?;
        let encoder = KeyframeEncoder::load(device, spec, dir, manifest, true)?;
        let tiles = VaeTiles::released(spec, frame[0], frame[1])?;
        let future = image_encode_future(spec, &tiles, frame, output)?;
        let transient_bytes = pixels
            .checked_add(noise)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(future))
            .context("resident image encoder transient extent overflow")?;
        let sizing = EncodeWork::sizing(&encoder, (tiles.height, tiles.width))?;
        Ok(Self {
            encoder,
            tiles,
            extent,
            sizing,
            transient_bytes,
        })
    }

    pub(super) fn minimum_work(&self) -> usize {
        let (activations, [minimum_columns, _]) = self.sizing;
        4 * (5 * activations + minimum_columns)
    }

    pub(super) fn bind(self, device: &Device, later: usize) -> Result<ImageEncodeStage> {
        let (free, total) = device.ctx.mem_get_info()?;
        let reserved = (total / 16)
            .checked_add(later)
            .context("resident image encoder reserve overflow")?;
        let budget = device.workspace_budget(free, free, reserved)?;
        let work = EncodeWork::allocate(
            &device.stream,
            (self.tiles.height, self.tiles.width),
            self.sizing,
            budget,
        )?;
        Ok(ImageEncodeStage {
            encoder: self.encoder,
            work,
            tiles: self.tiles,
            extent: self.extent,
        })
    }
}

impl ImageEncodeStage {
    pub(super) fn run(
        &mut self,
        device: &Device,
        blas: &Blas,
        extent: [usize; 2],
        rgb: &[f32],
        noise: &[f32],
    ) -> Result<(Vec<f32>, [usize; 2])> {
        ensure!(
            extent == self.extent,
            "reference extent {extent:?} differs from admitted {:?}",
            self.extent
        );
        ensure!(
            rgb.iter()
                .all(|v| v.is_finite() && (0.0..=255.0).contains(v)),
            "image VAE RGB outside [0,255]"
        );
        ensure!(
            noise.iter().all(|v| v.is_finite()),
            "image VAE posterior noise is nonfinite"
        );
        let shape = image_encode_shape(&self.encoder.spec, extent, rgb.len(), noise.len())?;
        let s = &device.stream;
        let pixels = s.clone_htod(rgb)?;
        let noise = s.clone_htod(noise)?;
        let [_, _, latents] = self.encoder.encode_frame(
            (s, blas),
            &mut self.work,
            &self.tiles,
            (&pixels, extent[0], extent[1]),
            &noise,
            shape.0,
        )?;
        let output = s.clone_dtoh(&latents)?;
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "image VAE encoder output is nonfinite"
        );
        Ok((output, shape.1))
    }
}

/// Normalized F32 `[24, latent_h, latent_w]` to F32 `[height, width, 3]`, before clamp and quantization.
pub(super) fn decode_image_device(
    device: &Device,
    blas: &mut Blas,
    model: &Path,
    latent: &CudaSlice<f32>,
    extent: [usize; 2],
) -> Result<CudaSlice<f32>> {
    let component = model.join("vae_diffusers");
    let config = component.join("config.json");
    let spec = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
        &std::fs::read(&config).with_context(|| format!("read {}", config.display()))?,
    )?)?;
    let checkpoint =
        crate::checkpoint::MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
    let manifest = checkpoint.minimax_h3_image_vae_manifest(&spec)?;
    let canvas = [
        extent[0]
            .checked_mul(spec.spatial_ratio)
            .context("image VAE canvas overflow")?,
        extent[1]
            .checked_mul(spec.spatial_ratio)
            .context("image VAE canvas overflow")?,
    ];
    let elements = extent[0]
        .checked_mul(extent[1])
        .and_then(|n| n.checked_mul(spec.latent_channels))
        .context("image VAE input extent overflow")?;
    ensure!(
        extent.into_iter().all(|n| n > 0) && latent.len() == elements,
        "image VAE normalized latent extent mismatch"
    );
    let decoder = ImageDecoder::new(device, &spec.single_frame(), &component, &manifest)?;
    let (free, total) = device.ctx.mem_get_info()?;
    let budget = device.workspace_budget(free, free, total / 16)?;
    let work = decoder.work(device, canvas, budget)?;
    decoder.core.bind(blas, &work.core)?;
    decoder.run(&device.stream, blas, work, latent)
}

/// F32 RGB `[height,width,3]` and F32 posterior noise `[1,24,height/16,width/16]` to cropped normalized latents.
pub(super) fn encode_image_device(
    device: &Device,
    blas: &Blas,
    model: &Path,
    rgb: &CudaSlice<f32>,
    extent: [usize; 2],
    epsilon: &CudaSlice<f32>,
) -> Result<(CudaSlice<f32>, [usize; 2])> {
    let component = model.join("vae_diffusers");
    let published = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
        &std::fs::read(component.join("config.json"))?,
    )?)?;
    let spec = published.single_frame();
    let shape = image_encode_shape(&spec, extent, rgb.len(), epsilon.len())?;
    let checkpoint =
        crate::checkpoint::MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
    let manifest = checkpoint.minimax_h3_image_vae_manifest(&published)?;
    let encoder = KeyframeEncoder::load(device, &spec, &component, &manifest, true)?;
    let tiles = VaeTiles::released(&spec, shape.0[0], shape.0[1])?;
    let (free, total) = device.ctx.mem_get_info()?;
    let budget = device.workspace_budget(free, free, total / 16)?;
    let future = image_encode_future(&spec, &tiles, shape.0, shape.1)?;
    let mut work = EncodeWork::with_budget(
        &device.stream,
        &encoder,
        (tiles.height, tiles.width),
        budget
            .checked_sub(future)
            .context("image encoder budget holds no frame buffers")?,
    )?;
    let [_, _, latents] = encoder.encode_frame(
        (&device.stream, blas),
        &mut work,
        &tiles,
        (rgb, extent[0], extent[1]),
        epsilon,
        shape.0,
    )?;
    Ok((latents, shape.1))
}

fn image_encode_shape(
    spec: &VaeSpec,
    extent: [usize; 2],
    rgb: usize,
    noise: usize,
) -> Result<([usize; 2], [usize; 2])> {
    let r = spec.spatial_ratio;
    ensure!(
        extent.into_iter().all(|n| n <= u32::MAX as usize),
        "image encoder extent exceeds kernel indices"
    );
    let input = extent[0]
        .checked_mul(extent[1])
        .and_then(|n| n.checked_mul(spec.in_channels))
        .context("image encoder RGB extent overflow")?;
    let latent = [extent[0] / r, extent[1] / r];
    let expected = latent[0]
        .checked_mul(latent[1])
        .and_then(|n| n.checked_mul(spec.latent_channels))
        .context("image encoder posterior extent overflow")?;
    let output = latent.map(|n| n / 2 * 2);
    ensure!(
        output.into_iter().all(|n| n > 0) && input == rgb && expected == noise,
        "image encoder RGB/posterior shapes differ: {extent:?}, RGB{rgb}/{input}, epsilon{noise}/{expected}"
    );
    Ok((latent.map(|n| n * r), output))
}

fn image_encode_future(
    spec: &VaeSpec,
    tiles: &VaeTiles,
    canvas: [usize; 2],
    output: [usize; 2],
) -> Result<usize> {
    let c = spec.latent_channels as u128;
    let plane = (canvas[0] / spec.spatial_ratio) as u128 * (canvas[1] / spec.spatial_ratio) as u128;
    let tile_plane =
        (tiles.height / spec.spatial_ratio) as u128 * (tiles.width / spec.spatial_ratio) as u128;
    let grid = (2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2) as u128;
    Ok(usize::try_from(
        4 * (grid
            + tiles.count() as u128 * 2 * c * tile_plane
            + 4 * c * plane
            + c * output[0] as u128 * output[1] as u128),
    )?)
}

pub(crate) fn planar_f32_to_rgb_u8<T: DeviceRepr>(
    device: &Device,
    kernel: &CudaFunction,
    input: &impl DevicePtr<T>,
    output: &impl DevicePtr<u8>,
    (frames, allocated_frames, pixels): (usize, usize, usize),
    (mean, std): ([f32; 3], [f32; 3]),
) -> Result<()> {
    ensure!(
        frames > 0 && frames <= allocated_frames && pixels > 0,
        "RGB extent frames={frames} allocated_frames={allocated_frames} pixels={pixels}"
    );
    let plane = allocated_frames
        .checked_mul(pixels)
        .context("RGB plane overflow")?;
    let elements = frames
        .checked_mul(pixels)
        .context("RGB pixel extent overflow")?;
    let input_elements = plane.checked_mul(3).context("RGB input extent overflow")?;
    let output_elements = elements
        .checked_mul(3)
        .context("RGB output extent overflow")?;
    let input_bytes = input_elements
        .checked_mul(size_of::<T>())
        .context("RGB input byte extent overflow")?;
    ensure!(
        input.len() >= input_elements && output.len() >= output_elements,
        "RGB needs input {input_elements}/{} and output {output_elements}/{} elements",
        input.len(),
        output.len()
    );
    ensure!(
        mean.into_iter().chain(std).all(f32::is_finite),
        "RGB affine parameters are nonfinite"
    );
    use cudarc::driver::sys::CUdevice_attribute as A;
    let active = kernel.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    let sms = u32::try_from(
        device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
    )?;
    let grid_limit = u32::try_from(
        device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
    )?;
    let resident = active
        .checked_mul(sms)
        .context("RGB resident grid overflow")?;
    ensure!(resident > 0 && grid_limit > 0, "RGB has no resident blocks");
    let grid = u32::try_from(
        elements
            .div_ceil(256)
            .min(resident.min(grid_limit) as usize),
    )?;
    let s = &device.stream;
    let source = input.device_ptr(s).0;
    let target = output.device_ptr(s).0;
    let source_end = source
        .checked_add(input_bytes as u64)
        .context("RGB source address overflow")?;
    let target_end = target
        .checked_add(output_elements as u64)
        .context("RGB target address overflow")?;
    ensure!(
        source_end <= target || target_end <= source,
        "RGB input/output overlap"
    );
    unsafe {
        s.launch_builder(kernel)
            .arg(&source)
            .arg(&target)
            .arg(&(allocated_frames as u64))
            .arg(&(frames as u64))
            .arg(&(pixels as u64))
            .arg(&mean[0])
            .arg(&mean[1])
            .arg(&mean[2])
            .arg(&std[0])
            .arg(&std[1])
            .arg(&std[2])
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_rgb_launch":{"frames":frames,"allocated_frames":allocated_frames,
            "pixels":pixels,"input_bytes":input_bytes,"output_bytes":output_elements,
            "grid":grid,"threads":256,"active_blocks_per_sm":active,"sms":sms}})
        );
    }
    Ok(())
}

struct Linear<T> {
    weight: Tensor<T, 2>,
    bias: Tensor<T, 1>,
}

struct DecoderBlock<T> {
    q: Linear<T>,
    k: Linear<T>,
    v: Linear<T>,
    out: Linear<T>,
    ff_in: Linear<T>,
    ff_out: Linear<T>,
    norm1: Tensor<f32, 1>,
    norm2: Tensor<f32, 1>,
    scale1: Tensor<f32, 1>,
    scale2: Tensor<f32, 1>,
}

struct VaeKernels {
    frame_blend: CudaFunction,
    latent_rows: CudaFunction,
    rgb: CudaFunction,
    stitch: CudaFunction,
    append: CudaFunction,
    rms: CudaFunction,
    head_pack: CudaFunction,
    norm_out: CudaFunction,
    residual: CudaFunction,
    residual_rms: CudaFunction,
    patches: CudaFunction,
    swiglu: CudaFunction,
}

pub(crate) struct VideoDecoder {
    spec: VaeSpec,
    post_quant: Linear<bf16>,
    proj_in: Linear<bf16>,
    registers: Tensor<bf16, 1>,
    norm_out: (Tensor<f32, 1>, Tensor<f32, 1>),
    proj_out: Linear<bf16>,
    blocks: Vec<DecoderBlock<bf16>>,
    tier: Option<std::cell::RefCell<DenseTier>>,
    _weights: Vec<CudaSlice<u8>>,
    kernels: VaeKernels,
    #[cfg(test)]
    prefix_only: bool,
}

impl Linear<bf16> {
    fn bind(&self, blas: &mut Blas, rows: usize) -> Result<()> {
        blas.prepare_biased(
            u32::try_from(rows)?,
            self.weight.shape[0],
            self.weight.shape[1],
            self.bias,
            LinearBiasMode::BeforeOutputRound,
        )
    }

    fn apply(&self, blas: &Blas, x: Tensor<bf16, 2>, y: Tensor<bf16, 2>) -> Result<()> {
        blas.linear_biased(
            self.weight,
            self.bias,
            x,
            &mut TensorMut(y),
            LinearBiasMode::BeforeOutputRound,
        )
    }
}

fn decoder_weight_width(name: &str) -> usize {
    if name
        .split('.')
        .any(|part| matches!(part, "norm1" | "norm2" | "norm_out" | "scale1" | "scale2"))
    {
        4
    } else {
        2
    }
}

fn decoder_tensor_bytes(manifest: &TensorManifest, name: &str) -> usize {
    manifest.entries[name].bytes / 4 * decoder_weight_width(name)
}

pub(super) fn decoder_weight_bytes(manifest: &TensorManifest) -> usize {
    manifest
        .entries
        .keys()
        .filter(|name| name.starts_with("decoder.") || name.starts_with("post_quant_conv."))
        .map(|name| decoder_tensor_bytes(manifest, name))
        .sum()
}

fn read_decoder_weight_chunk(
    source: &mut EntryReader,
    manifest: &TensorManifest,
    name: &str,
    destination_offset: usize,
    dst: &mut [u8],
) -> Result<()> {
    let width = decoder_weight_width(name);
    ensure!(
        dst.len().is_multiple_of(width),
        "video weight {name} chunk alignment"
    );
    let src = source.read_range(
        name,
        &manifest.entries[name],
        destination_offset / width * 4,
        dst.len() / width * 4,
    )?;
    ensure!(
        <f32 as Element>::finite(src),
        "nonfinite video weight {name}"
    );
    if width == 4 {
        dst.copy_from_slice(src);
    } else {
        for (src, dst) in src
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<2>().0)
        {
            let x = bf16::from_f32(f32::from_le_bytes(*src));
            ensure!(x.is_finite(), "BF16 overflow for video weight {name}");
            *dst = x.to_bits().to_le_bytes();
        }
    }
    Ok(())
}

struct DecoderWeightLoad<'a> {
    manifest: &'a TensorManifest,
    reader: EntryReader,
    dense: Option<DenseTier>,
    offsets: std::collections::BTreeMap<String, (usize, usize)>,
    consumed: std::collections::BTreeSet<String>,
    storage: Vec<CudaSlice<u8>>,
    staging: Vec<u8>,
    resident: std::collections::BTreeMap<String, u64>,
}

impl<'a> DecoderWeightLoad<'a> {
    fn new(dir: &Path, manifest: &'a TensorManifest) -> Self {
        Self {
            manifest,
            reader: EntryReader::new(dir),
            dense: None,
            offsets: Default::default(),
            consumed: Default::default(),
            storage: Vec::new(),
            staging: vec![0; 16 << 20],
            resident: Default::default(),
        }
    }

    /// Preloads every resident decoder tensor (no pinned offset) through a bounded pinned
    /// ring: fill workers read and narrow a tensor into a free slot, this thread uploads
    /// the slot on the upload stream and returns it after its event. The ring holds
    /// `in_flight` slots of the largest tensor; a host budget that denies it keeps the
    /// serial chunked staging in [`DecoderWeightLoad::tensor`]. Nothing is retried.
    fn preload_resident(&mut self, d: &Device, component: &'static str) -> Result<()> {
        let names: Vec<(String, usize)> = self
            .manifest
            .entries
            .iter()
            .filter(|(name, _)| {
                (name.starts_with("decoder.") || name.starts_with("post_quant_conv."))
                    && !self.offsets.contains_key(*name)
            })
            .map(|(name, _)| (name.clone(), decoder_tensor_bytes(self.manifest, name)))
            .collect();
        if names.is_empty() {
            return Ok(());
        }
        let dir = self.reader.dir.clone();
        let requests: Vec<super::OwnedRead> = names
            .iter()
            .map(|(name, bytes)| {
                let entry = &self.manifest.entries[name.as_str()];
                let cast = if decoder_weight_width(name) == 2 {
                    super::FillCast::F32ToBf16
                } else {
                    super::FillCast::Raw
                };
                // The device buffer is allocated now; the ring fills it below.
                super::OwnedRead::new(name, entry, std::ptr::null_mut(), *bytes, cast)
            })
            .collect();
        let largest = names.iter().map(|(_, bytes)| *bytes).max().unwrap_or(0);
        let shards = requests
            .iter()
            .map(|request| request.entry().shard.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let in_flight = super::fill_in_flight(shards, requests.len());
        let ring_bytes = in_flight
            .checked_mul(largest)
            .context("resident ring extent overflow")?;
        let available = usize::try_from(crate::backend::cuda::calibrate::host_memory(Path::new(
            "/",
        ))?)?;
        let reserve = 4 << 30;
        let ring = available >= ring_bytes + reserve;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_fill_schedule": {
                    "component": component, "decision": if ring { "ring" } else { "serial" },
                    "requests": requests.len(), "shards": shards, "in_flight": in_flight,
                    "ring_bytes": ring_bytes, "slot_bytes": largest,
                    "host_available_bytes": available, "host_reserve_bytes": reserve}})
            );
        }
        if !ring {
            return Ok(());
        }
        let mut pin = unsafe { d.ctx.alloc_pinned_with_flags::<u8>(ring_bytes, 0)? };
        /// The ring base, Send by the free-slot discipline: a worker writes only its claimed slot.
        struct RingBase(*mut u8);
        unsafe impl Send for RingBase {}
        unsafe impl Sync for RingBase {}
        let base = RingBase(pin.as_mut_ptr()?);
        let events = (0..in_flight)
            .map(|_| d.ctx.new_event(None))
            .collect::<Result<Vec<_>, _>>()?;
        let mut destinations = Vec::with_capacity(requests.len());
        for (name, bytes) in &names {
            let buffer = unsafe { d.upload.alloc::<u8>(*bytes)? };
            let ptr = buffer.device_ptr(&d.upload).0;
            self.resident.insert(name.clone(), ptr);
            destinations.push(ptr);
            self.storage.push(buffer);
        }
        let free = std::sync::Arc::new((
            std::sync::Mutex::new((0..in_flight).collect::<Vec<_>>()),
            std::sync::Condvar::new(),
        ));
        let abort = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(usize, usize)>>();
        let slot_bytes = largest;
        let producer_base = base;
        let consumer_base = pin.as_mut_ptr()?;
        let producer_dir = dir.clone();
        let producer_requests = requests.clone();
        let producer_free = std::sync::Arc::clone(&free);
        let producer_abort = std::sync::Arc::clone(&abort);
        let producer = std::thread::Builder::new()
            .name("vae-ring".to_owned())
            .spawn(move || -> Result<()> {
                let base = &producer_base;
                let mut reader = EntryReader::new(&producer_dir);
                let readers = reader.open_shards(
                    producer_requests
                        .iter()
                        .map(|request| request.entry().shard.as_str()),
                )?;
                let order = super::claim_order(&producer_requests, |request| {
                    request.entry().shard.as_str()
                });
                let next = std::sync::atomic::AtomicUsize::new(0);
                std::thread::scope(|scope| -> Result<()> {
                    let handles: Vec<_> = (0..in_flight)
                        .map(|_| {
                            let readers = &readers;
                            let requests = &producer_requests;
                            let order = &order;
                            let next = &next;
                            let free = &producer_free;
                            let abort = &producer_abort;
                            let done = &done_tx;
                            scope.spawn(move || -> Result<()> {
                                let mut bounce =
                                    crate::experts::disk::AlignedBytes::new(4096, 4096)?;
                                loop {
                                    let claimed =
                                        next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    let Some(&index) = order.get(claimed) else {
                                        return Ok(());
                                    };
                                    if abort.load(std::sync::atomic::Ordering::SeqCst) {
                                        return Ok(());
                                    }
                                    let slot = {
                                        let mut guard = free.0.lock().unwrap();
                                        loop {
                                            if let Some(slot) = guard.pop() {
                                                break slot;
                                            }
                                            if abort.load(std::sync::atomic::Ordering::SeqCst) {
                                                return Ok(());
                                            }
                                            guard = free
                                                .1
                                                .wait_timeout(
                                                    guard,
                                                    std::time::Duration::from_millis(100),
                                                )
                                                .unwrap()
                                                .0;
                                        }
                                    };
                                    let request = &requests[index];
                                    let window = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            base.0.add(slot * slot_bytes),
                                            request.dest_len(),
                                        )
                                    };
                                    let result = EntryReader::read_bounced(
                                        readers,
                                        request.name(),
                                        request.entry(),
                                        request.cast(),
                                        window,
                                        &mut bounce,
                                    );
                                    if let Err(error) = result {
                                        abort.store(true, std::sync::atomic::Ordering::SeqCst);
                                        let _ = done.send(Err(error));
                                        return Ok(());
                                    }
                                    done.send(Ok((index, slot)))?;
                                }
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle
                            .join()
                            .map_err(|_| anyhow::anyhow!("vae-ring worker panicked"))??;
                    }
                    Ok(())
                })
            })?;
        let mut received = 0usize;
        let mut inflight: Vec<usize> = Vec::new();
        let error = loop {
            // Return slots whose upload completed.
            let mut still = Vec::new();
            for &slot in &inflight {
                if events[slot].try_is_complete()? {
                    free.0.lock().unwrap().push(slot);
                    free.1.notify_all();
                } else {
                    still.push(slot);
                }
            }
            inflight = still;
            match done_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(Ok((index, slot))) => {
                    let window = unsafe {
                        std::slice::from_raw_parts(
                            consumer_base.add(slot * slot_bytes),
                            names[index].1,
                        )
                    };
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async(
                            destinations[index],
                            window,
                            d.upload.cu_stream(),
                        )?;
                    }
                    events[slot].record(&d.upload)?;
                    inflight.push(slot);
                    received += 1;
                    if received == names.len() {
                        break Ok(());
                    }
                }
                Ok(Err(error)) => break Err(error),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(anyhow::anyhow!("vae-ring producer ended early"));
                }
            }
        };
        abort.store(true, std::sync::atomic::Ordering::SeqCst);
        free.1.notify_all();
        let joined = producer
            .join()
            .map_err(|_| anyhow::anyhow!("vae-ring producer panicked"));
        d.upload.synchronize()?;
        drop(pin);
        joined??;
        error?;
        Ok(())
    }

    fn tensor<T: Element + Copy>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        count: usize,
    ) -> Result<Tensor<T, 1>> {
        use safetensors::Dtype;
        let entry = self
            .manifest
            .entries
            .get(name)
            .with_context(|| format!("video weight {name} missing"))?;
        let width = decoder_weight_width(name);
        ensure!(
            self.consumed.insert(name.to_owned())
                && entry.dtype == Dtype::F32
                && entry.quant.is_none()
                && entry.bytes == 4 * count
                && T::DTYPE == if width == 4 { Dtype::F32 } else { Dtype::BF16 },
            "invalid source or runtime dtype for video weight {name}"
        );
        let bytes = count * width;
        let ptr = if let Some(&(layer, offset)) = self.offsets.get(name) {
            self.dense
                .as_ref()
                .context("video pinned tensor lacks stream binding")?
                .dense
                .pointer(layer, offset, bytes)?
        } else if let Some(ptr) = self.resident.get(name) {
            *ptr
        } else {
            let buffer = unsafe { s.alloc::<u8>(bytes)? };
            let (ptr, record) = buffer.device_ptr(s);
            let chunk = self.staging.len();
            for offset in (0..bytes).step_by(chunk) {
                let end = (offset + chunk).min(bytes);
                let dst = &mut self.staging[..end - offset];
                read_decoder_weight_chunk(&mut self.reader, self.manifest, name, offset, dst)?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        ptr + offset as u64,
                        dst,
                        s.cu_stream(),
                    )?;
                }
            }
            drop(record);
            self.storage.push(buffer);
            ptr
        };
        ensure!(
            ptr.is_multiple_of(16),
            "video weight {name} pointer alignment"
        );
        unsafe { Tensor::new(ptr, [u32::try_from(count)?], [1], bytes as u64) }
    }

    fn linear(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        out: usize,
        input: usize,
    ) -> Result<Linear<bf16>> {
        Linear::load(
            &mut |name, count| self.tensor(s, name, count),
            name,
            out,
            input,
        )
    }
}

fn video_weight_load<'a>(
    d: &Device,
    spec: &VaeSpec,
    dir: &Path,
    manifest: &'a TensorManifest,
    (future, mode): (usize, super::Placement),
) -> Result<DecoderWeightLoad<'a>> {
    let mut order = vec![Vec::new(); spec.layers];
    let mut sizes = vec![0; spec.layers];
    let mut fixed_norm1 = 0usize;
    for (name, entry) in manifest
        .entries
        .iter()
        .filter(|(name, _)| name.starts_with("decoder.") || name.starts_with("post_quant_conv."))
    {
        ensure!(
            entry.dtype == safetensors::Dtype::F32
                && entry.quant.is_none()
                && entry.bytes.is_multiple_of(4),
            "video decoder weight {name} is not published F32"
        );
        if let Some(rest) = name.strip_prefix("decoder.transformer_blocks.") {
            let (layer, field) = rest.split_once('.').context("video block role")?;
            let layer = layer.parse::<usize>()?;
            let dst = order.get_mut(layer).context("video block index")?;
            if field == DECODER_RESIDENT_BLOCK_WEIGHT {
                fixed_norm1 += 1;
                continue;
            }
            dst.push(name);
            sizes[layer] += decoder_tensor_bytes(manifest, name);
        }
    }
    ensure!(
        fixed_norm1 == spec.layers && order.iter().all(|names| names.len() == 15),
        "video block weight inventory"
    );
    let weights = decoder_weight_bytes(manifest);
    let fixed = weights - sizes.iter().sum::<usize>();
    let (free, total) = d.ctx.mem_get_info()?;
    let mut budget = crate::prefill::DensePlacement::device_budget(free, total, fixed, 0, future)?;
    if mode != super::Placement::Derived {
        let slots = 2 * sizes.iter().max().context("video VAE has no blocks")?;
        ensure!(
            slots <= budget,
            "video BF16 stream slots exceed device budget"
        );
        budget = slots;
    }
    let host = if sizes.iter().sum::<usize>() > budget {
        usize::try_from(crate::backend::cuda::calibrate::host_memory(Path::new(
            "/",
        ))?)?
        .checked_sub(4 << 30)
        .context("video host reserve")?
    } else {
        0
    };
    let plan = crate::prefill::DensePlacement::new(&sizes, Some(&sizes), budget, host)?;
    ensure!(
        plan.disk_bytes == 0,
        "video BF16 pinned layer plan exceeds host budget"
    );
    ensure!(
        mode == super::Placement::Derived || plan.resident_bytes == 0,
        "forced streamed video decoder retained blocks"
    );
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"video_vae_precision_placement": {
                "weight_bytes":weights,"fixed_bytes":fixed,"resident_block_bytes":plan.resident_bytes,
                "slot_bytes":2*plan.slot_bytes,"pinned_bytes":plan.host_bytes,"future_bytes":future,
                "free_bytes":free,"total_bytes":total,"reserve_bytes":total/16,"dtype":"BF16/F32"
            }})
        );
    }
    let mut loader = DecoderWeightLoad::new(dir, manifest);
    if plan.slot_bytes > 0 {
        let slot_count = plan
            .slots
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |index| index + 1);
        let mut dense = DenseStreaming::new_preloaded(dir, plan, d, &sizes)?;
        let mut requests = Vec::new();
        let mut checks = Vec::new();
        let mut largest = 0usize;
        for (layer, names) in order.iter().enumerate() {
            if let Some(pinned) = dense.host_layer(layer)? {
                let mut offset = 0;
                for name in names {
                    let width = decoder_weight_width(name);
                    let bytes = decoder_tensor_bytes(manifest, name);
                    let entry = &manifest.entries[name.as_str()];
                    ensure!(offset % 16 == 0, "video pinned weight alignment");
                    loader.offsets.insert((*name).clone(), (layer, offset));
                    let cast = if width == 2 {
                        FillCast::F32ToBf16
                    } else {
                        FillCast::Raw
                    };
                    requests.push(OwnedRead::new(
                        name,
                        entry,
                        unsafe { pinned.as_mut_ptr().add(offset) },
                        bytes,
                        cast,
                    ));
                    checks.push(FillCheck::finite(
                        layer,
                        name,
                        if width == 2 {
                            safetensors::Dtype::BF16
                        } else {
                            safetensors::Dtype::F32
                        },
                        offset,
                        bytes,
                    ));
                    largest = largest.max(bytes);
                    offset += bytes;
                }
                ensure!(offset == pinned.len(), "video pinned layer byte count");
            }
        }
        let slot_bytes = dense.plan.slot_bytes;
        loader.dense = Some(DenseTier::schedule(
            "video-vae",
            dense,
            dir,
            requests,
            checks,
            (slot_count, slot_bytes, largest),
        )?);
    }
    loader.preload_resident(d, "video-vae")?;
    Ok(loader)
}

impl VideoDecoder {
    pub(crate) fn new(
        d: &Device,
        spec: &VaeSpec,
        dir: &Path,
        manifest: &TensorManifest,
        placement: (usize, super::Placement),
    ) -> Result<Self> {
        Self::from_weights(
            d,
            spec,
            video_weight_load(d, spec, dir, manifest, placement)?,
        )
    }

    fn from_weights(d: &Device, spec: &VaeSpec, mut load: DecoderWeightLoad<'_>) -> Result<Self> {
        let vision = cubin::module(&d.ctx, "vision_ops")?;
        let common = cubin::module(&d.ctx, "q38_common")?;
        let s = &d.stream;
        let manifest = load.manifest;
        ensure!(
            spec.dim <= 2048,
            "VAE decoder dim {} exceeds residual RMSNorm limit",
            spec.dim
        );
        let (dim, latent, ffn) = (spec.dim, spec.latent_channels, spec.ffn);
        let post_quant = load.linear(s, "post_quant_conv", latent, latent)?;
        let proj_in = load.linear(s, "decoder.proj_in", dim, latent)?;
        let registers = load.tensor(s, "decoder.register_tokens", spec.registers * dim)?;
        let norm_out = (
            load.tensor(s, "decoder.norm_out.weight", dim)?,
            load.tensor(s, "decoder.norm_out.bias", dim)?,
        );
        let proj_out = load.linear(s, "decoder.proj_out", spec.patch_width(), dim)?;
        let mut blocks = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            let b = format!("decoder.transformer_blocks.{layer}");
            blocks.push(DecoderBlock {
                q: load.linear(s, &format!("{b}.attn.to_q"), dim, dim)?,
                k: load.linear(s, &format!("{b}.attn.to_k"), dim, dim)?,
                v: load.linear(s, &format!("{b}.attn.to_v"), dim, dim)?,
                out: load.linear(s, &format!("{b}.attn.to_out.0"), dim, dim)?,
                ff_in: load.linear(s, &format!("{b}.ff.net.0.proj"), 2 * ffn, dim)?,
                ff_out: load.linear(s, &format!("{b}.ff.net.2"), dim, ffn)?,
                norm1: load.tensor(s, &format!("{b}.norm1.weight"), dim)?,
                norm2: load.tensor(s, &format!("{b}.norm2.weight"), dim)?,
                scale1: load.tensor(s, &format!("{b}.scale1"), dim)?,
                scale2: load.tensor(s, &format!("{b}.scale2"), dim)?,
            });
        }
        ensure!(
            load.consumed.len()
                == manifest
                    .entries
                    .keys()
                    .filter(|n| { n.starts_with("decoder.") || n.starts_with("post_quant_conv.") })
                    .count(),
            "VAE decoder tensor inventory mismatch"
        );
        Ok(Self {
            spec: spec.clone(),
            post_quant,
            proj_in,
            registers,
            norm_out,
            proj_out,
            blocks,
            tier: load.dense.map(std::cell::RefCell::new),
            _weights: load.storage,
            kernels: VaeKernels {
                frame_blend: vision.load_function("frame_blend_bf16")?,
                latent_rows: vision.load_function("latent_rows_bf16")?,
                rgb: vision.load_function("planar_bf16_to_rgb_u8")?,
                stitch: vision.load_function("tile_stitch_bf16")?,
                append: vision.load_function("append_register_rows_bf16_to_f32")?,
                rms: vision.load_function("vision_rms_scale_f32")?,
                head_pack: vision.load_function("rms_rotary_pack_qkv_bf16")?,
                norm_out: common.load_function("layer_norm_f32_to_bf16")?,
                residual: vision.load_function("scaled_residual_f32")?,
                residual_rms: vision.load_function("scaled_residual_rms_f32")?,
                patches: vision.load_function("unpatchify_bf16")?,
                swiglu: cubin::module(&d.ctx, "act")?.load_function("silu_mul")?,
            },
            #[cfg(test)]
            prefix_only: false,
        })
    }

    pub(crate) fn rgb(
        &self,
        d: &Device,
        decoded: &CudaSlice<bf16>,
        shape: (usize, usize, usize),
    ) -> Result<CudaSlice<u8>> {
        let count = shape
            .0
            .checked_mul(shape.2)
            .and_then(|n| n.checked_mul(3))
            .context("RGB output overflow")?;
        let (free, total) = d.ctx.mem_get_info()?;
        ensure!(
            count <= d.workspace_budget(free, free, total / 16)?,
            "RGB output exceeds memory budget"
        );
        let output = unsafe { d.stream.alloc::<u8>(count)? };
        planar_f32_to_rgb_u8(
            d,
            &self.kernels.rgb,
            decoded,
            &output,
            shape,
            (
                VAE_PIXEL_MEAN.map(|v| v as f32),
                VAE_PIXEL_STD.map(|v| v as f32),
            ),
        )?;
        Ok(output)
    }

    pub(crate) fn work(
        &self,
        d: &Device,
        tiles: &VaeTiles,
        shape: (usize, usize, usize),
    ) -> Result<TileWork> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let spec = &self.spec;
        let tile = (
            spec.chunk_latents() + spec.overlap_latents(),
            tiles.height / spec.spatial_ratio,
            tiles.width / spec.spatial_ratio,
        );
        let plan = DecodePlan::new(spec, shape.0)?;
        let total_frames = plan.chunks
            * (spec.chunk_latents() * spec.temporal_ratio - spec.frame_pre_padding())
            + tile.0 * spec.temporal_ratio
            - spec.chunk_latents() * spec.temporal_ratio
            - spec.frame_pre_padding();
        let canvas = (shape.1 * spec.spatial_ratio, shape.2 * spec.spatial_ratio);
        let mandatory = decode_bytes(spec, tiles, canvas)
            + 2 * spec.out_channels * total_frames * canvas.0 * canvas.1
            + 2 * spec.latent_channels * shape.0 * shape.1 * shape.2;
        let (free, total) = d.ctx.mem_get_info()?;
        let budget = crate::prefill::DensePlacement::device_budget(free, total, mandatory, 0, 0)?;
        let tokens = tile.0 * tile.1 * tile.2 + spec.registers + 1;
        let grid_x = usize::try_from(d.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?)?;
        let grid_z = usize::try_from(d.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?)?;
        let cap = tiles
            .count()
            .min(grid_z)
            .min(grid_x / (2 * tokens * spec.heads))
            .min(i32::MAX as usize / (tokens * spec.ffn));
        let mut work = None;
        for batch in (1..=cap).rev() {
            let bytes = TileWork::batch_bytes(spec, tile, batch, tiles.count())?;
            if bytes <= budget {
                work = Some((batch, bytes));
                break;
            }
        }
        let (batch, bytes) = work.context("video BF16 decoder cannot fit one tile")?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"video_vae_bf16_plan":{
            "linear_batch":batch,"attention_batch":batch,"work_bytes":bytes,"future_bytes":mandatory,"budget":budget}})
            );
        }
        TileWork::new(&d.stream, spec, tile, (batch, tiles.count()))
    }

    pub(crate) fn bind(&self, blas: &mut Blas, work: &TileWork) -> Result<()> {
        for &(active, _) in &work.attention {
            self.post_quant.bind(blas, active * work.patches)?;
            self.proj_in.bind(blas, active * work.patches)?;
            let rows = active * work.tokens;
            self.proj_out.bind(blas, rows)?;
            for b in &self.blocks {
                for linear in [&b.q, &b.k, &b.v, &b.out, &b.ff_in, &b.ff_out] {
                    linear.bind(blas, rows)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn batch_setup(
        &self,
        d: &Device,
        blas: &mut Blas,
        tiles: &VaeTiles,
        shape: (usize, usize, usize),
    ) -> Result<()> {
        let work = self.work(d, tiles, shape)?;
        self.bind(blas, &work)?;
        println!(
            "PASS VIDEO_BF16_SETUP linear={} attention={} bytes={}",
            work.batch,
            work.batch,
            work.allocated_bytes()
        );
        Ok(())
    }

    fn block(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        (layer, active): (usize, usize),
        #[cfg(test)] observe: &mut impl FnMut(&str, ImageRows<'_>) -> Result<()>,
    ) -> Result<()> {
        let b = &self.blocks[layer];
        let spec = &self.spec;
        let rows = active * w.tokens;
        let dim = spec.dim;
        let n = rows * dim;
        let mat = |buffer| rows_view(buffer, s, 0, rows, dim, dim);
        let norm = |weight: Tensor<f32, 1>| -> Result<()> {
            unsafe {
                s.launch_builder(&self.kernels.rms)
                    .arg(&pointer(&w.x, s))
                    .arg(&weight.ptr)
                    .arg(&1u32)
                    .arg(&0u64)
                    .arg(&pointer(&w.norm, s))
                    .arg(&(dim as u32))
                    .arg(&spec.norm_eps)
                    .launch(ops::grid(rows, 256))?;
            }
            Ok(())
        };
        let residual = |scale: Tensor<f32, 1>| -> Result<()> {
            unsafe {
                s.launch_builder(&self.kernels.residual)
                    .arg(&pointer(&w.x, s))
                    .arg(&pointer(&w.o, s))
                    .arg(&scale.ptr)
                    .arg(&(n as u64))
                    .arg(&(dim as u32))
                    .launch(flat(n))?;
            }
            Ok(())
        };
        let residual_norm = |res_scale: Tensor<f32, 1>, norm_scale: Tensor<f32, 1>| -> Result<()> {
            unsafe {
                s.launch_builder(&self.kernels.residual_rms)
                    .arg(&pointer(&w.x, s))
                    .arg(&pointer(&w.o, s))
                    .arg(&res_scale.ptr)
                    .arg(&norm_scale.ptr)
                    .arg(&0u64)
                    .arg(&pointer(&w.norm, s))
                    .arg(&(dim as u32))
                    .arg(&spec.norm_eps)
                    .launch(ops::grid(rows, 256))?;
            }
            Ok(())
        };
        if layer == 0 {
            norm(b.norm1)?;
        }
        for (j, linear) in [&b.q, &b.k, &b.v].into_iter().enumerate() {
            linear.apply(
                blas,
                mat(&w.norm)?,
                rows_view(&w.ff, s, j * n, rows, dim, dim)?,
            )?;
        }
        #[cfg(test)]
        if layer == 0 {
            observe("q0", (DecoderRows::Bf16(&w.ff), 0, w.tokens, dim))?;
            observe("k0", (DecoderRows::Bf16(&w.ff), n, w.tokens, dim))?;
        }
        unsafe {
            s.launch_builder(&self.kernels.head_pack)
                .arg(&pointer(&w.ff, s))
                .arg(&pointer(&w.rope, s))
                .arg(&pointer(&w.q, s))
                .arg(&pointer(&w.k, s))
                .arg(&pointer(&w.v, s))
                .arg(&(rows as u32))
                .arg(&(w.tokens as u32))
                .arg(&(spec.heads as u32))
                .arg(&(spec.head_dim as u32))
                .arg(&(spec.rope_dims as u32))
                .arg(&spec.norm_eps)
                .launch(ops::grid((rows * spec.heads).div_ceil(8), 256))?;
        }
        let (_, flash) = w
            .attention
            .iter_mut()
            .find(|(tiles, _)| *tiles == active)
            .context("video attention batch is outside setup")?;
        flash.run(
            &w.q,
            dim,
            &w.k,
            &w.v,
            &mut w.attn,
            Shape {
                rows,
                start: 0,
                q_heads: spec.heads,
                kv_heads: spec.heads,
                head_dim: spec.head_dim,
                capacity: rows,
                causal: false,
                window_left: None,
            },
        )?;
        b.out.apply(blas, mat(&w.attn)?, mat(&w.o)?)?;
        residual_norm(b.scale1, b.norm2)?;
        let f = spec.ffn;
        b.ff_in.apply(
            blas,
            mat(&w.norm)?,
            rows_view(&w.ff, s, 0, rows, 2 * f, 2 * f)?,
        )?;
        let (gu, y) = (pointer(&w.ff, s), pointer(&w.gated, s));
        unsafe {
            s.launch_builder(&self.kernels.swiglu)
                .arg(&gu)
                .arg(&y)
                .arg(&(rows as i32))
                .arg(&(f as i32))
                .arg(&3i32)
                .launch(ops::silu_mul_grid(rows, f, gu, y)?)?;
        }
        b.ff_out
            .apply(blas, rows_view(&w.gated, s, 0, rows, f, f)?, mat(&w.o)?)?;
        if let Some(next) = self.blocks.get(layer + 1) {
            residual_norm(b.scale2, next.norm1)
        } else {
            residual(b.scale2)
        }
    }

    fn tile(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        z: &LatentVideo<'_>,
        (origins, first_frame): (&[(usize, usize)], usize),
        (raw, slot): (&CudaSlice<bf16>, usize),
    ) -> Result<()> {
        let spec = &self.spec;
        let active = origins.len();
        ensure!(active > 0 && active <= w.batch, "video tile batch capacity");
        ensure!(
            (slot + active) * w.pixels() <= raw.len(),
            "video raw tile capacity"
        );
        let p = w.patches;
        for (tile, &(row, col)) in origins.iter().enumerate() {
            ensure!(
                first_frame + w.frames <= z.frames
                    && row + w.height <= z.height
                    && col + w.width <= z.width,
                "video tile input capacity"
            );
            let target = pointer(&w.latent, s) + (2 * tile * p * spec.latent_channels) as u64;
            unsafe {
                s.launch_builder(&self.kernels.latent_rows)
                    .arg(&pointer(z.data, s))
                    .arg(&target)
                    .arg(&(spec.latent_channels as u32))
                    .arg(&(z.frames as u32))
                    .arg(&(z.height as u32))
                    .arg(&(z.width as u32))
                    .arg(&(first_frame as u32))
                    .arg(&(row as u32))
                    .arg(&(col as u32))
                    .arg(&(w.frames as u32))
                    .arg(&(w.height as u32))
                    .arg(&(w.width as u32))
                    .launch(flat(p * spec.latent_channels))?;
            }
        }
        self.project_tiles(
            (s, blas),
            w,
            active,
            (raw, slot),
            #[cfg(test)]
            |_, _| Ok(()),
        )
    }

    fn project_tiles(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut TileWork,
        active: usize,
        (raw, slot): (&CudaSlice<bf16>, usize),
        #[cfg(test)] mut observe: impl FnMut(&str, ImageRows<'_>) -> Result<()>,
    ) -> Result<()> {
        let spec = &self.spec;
        let p = w.patches;
        let latent = |data| {
            rows_view(
                data,
                s,
                0,
                active * p,
                spec.latent_channels,
                spec.latent_channels,
            )
        };
        self.post_quant
            .apply(blas, latent(&w.latent)?, latent(&w.latent_q)?)?;
        self.proj_in.apply(
            blas,
            latent(&w.latent_q)?,
            rows_view(&w.o, s, 0, active * p, spec.dim, spec.dim)?,
        )?;
        #[cfg(test)]
        observe("proj_in", (DecoderRows::Bf16(&w.o), 0, p, spec.dim))?;
        #[cfg(test)]
        if self.prefix_only {
            return Ok(());
        }
        let rows = active * w.tokens;
        unsafe {
            s.launch_builder(&self.kernels.append)
                .arg(&pointer(&w.o, s))
                .arg(&self.registers.ptr)
                .arg(&pointer(&w.x, s))
                .arg(&(active as u32))
                .arg(&(p as u32))
                .arg(&(spec.registers as u32))
                .arg(&(spec.dim as u32))
                .launch(flat(rows * spec.dim))?;
        }
        for layer in 0..self.blocks.len() {
            if let Some(tier) = &self.tier {
                tier.borrow_mut().ready()?.begin(layer, s)?;
            }
            self.block(
                s,
                blas,
                w,
                (layer, active),
                #[cfg(test)]
                &mut observe,
            )?;
            #[cfg(test)]
            if layer == 0 || layer + 1 == spec.layers {
                let role = if layer == 0 { "block0" } else { "block_last" };
                observe(role, (DecoderRows::F32(&w.x), 0, w.tokens, spec.dim))?;
            }
            if let Some(tier) = &self.tier {
                tier.borrow_mut().ready()?.end(layer, s)?;
            }
        }
        unsafe {
            s.launch_builder(&self.kernels.norm_out)
                .arg(&pointer(&w.x, s))
                .arg(&self.norm_out.0.ptr)
                .arg(&self.norm_out.1.ptr)
                .arg(&pointer(&w.norm, s))
                .arg(&(rows as u32))
                .arg(&(spec.dim as u32))
                .arg(&spec.norm_eps)
                .launch(ops::grid(rows, 256))?;
        }
        let width = spec.patch_width();
        self.proj_out.apply(
            blas,
            rows_view(&w.norm, s, 0, rows, spec.dim, spec.dim)?,
            rows_view(&w.patch_rows, s, 0, rows, width, width)?,
        )?;
        for tile in 0..active {
            let source = pointer(&w.patch_rows, s) + (2 * tile * w.tokens * width) as u64;
            let target = pointer(raw, s) + (2 * (slot + tile) * w.pixels()) as u64;
            unsafe {
                s.launch_builder(&self.kernels.patches)
                    .arg(&source)
                    .arg(&target)
                    .arg(&(w.frames as u32))
                    .arg(&(w.height as u32))
                    .arg(&(w.width as u32))
                    .arg(&(spec.out_channels as u32))
                    .arg(&(spec.temporal_ratio as u32))
                    .arg(&(spec.spatial_ratio as u32))
                    .launch(flat(w.pixels()))?;
            }
        }
        Ok(())
    }

    pub(crate) fn decode(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        tiles: &VaeTiles,
        z: LatentVideo<'_>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<DecodePlan> {
        self.decode_observed((s, blas), w, tiles, z, out, |_, _, _| Ok(()))
    }

    fn decode_observed(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut TileWork,
        tiles: &VaeTiles,
        z: LatentVideo<'_>,
        out: &mut CudaSlice<bf16>,
        mut observe: impl FnMut(&str, usize, &CudaSlice<bf16>) -> Result<()>,
    ) -> Result<DecodePlan> {
        let spec = &self.spec;
        let r = spec.spatial_ratio;
        let plan = DecodePlan::new(spec, z.frames)?;
        ensure!(
            plan.pad == 0,
            "padded latent lengths ({} pad frames) are not bound",
            plan.pad
        );
        let (chunk, overlap) = (spec.chunk_latents(), spec.overlap_latents());
        ensure!(
            w.frames == chunk + overlap
                && w.height * r == tiles.height
                && w.width * r == tiles.width,
            "tile work {}x{}x{} does not match the tile grid {}x{} px",
            w.frames,
            w.height,
            w.width,
            tiles.height,
            tiles.width
        );
        let (height, width) = (z.height * r, z.width * r);
        let plane = height * width;
        let clip_frames = w.frames * spec.temporal_ratio;
        let (pre, fade) = (spec.frame_pre_padding(), spec.frame_overlap());
        let lead = chunk * spec.temporal_ratio;
        let kept = lead - pre;
        let tail = clip_frames - lead - pre;
        let total = plan.chunks * kept + if spec.token_drop > 0 { tail } else { 0 };
        let channels = spec.out_channels;
        ensure!(
            out.len() == channels * total * plane,
            "decode output holds {} elements, not {}",
            out.len(),
            channels * total * plane
        );
        let (rows, cols) = (&tiles.rows, &tiles.cols);
        let grid: Vec<u32> = rows
            .0
            .iter()
            .chain(&rows.1)
            .chain(&cols.0)
            .chain(&cols.1)
            .map(|&v| u32::try_from(v))
            .collect::<Result<_, _>>()?;
        let grid = s.clone_htod(&grid)?;
        let slots = tiles.count();
        let raw = s.alloc_zeros::<bf16>(slots * w.pixels())?;
        let canvas = s.alloc_zeros::<bf16>(channels * clip_frames * plane)?;
        let mut faded = s.alloc_zeros::<bf16>(channels * tail * plane)?;
        ensure!(
            2 * (raw.len() + canvas.len() + faded.len())
                == decode_bytes(spec, tiles, (height, width)),
            "decode buffers differ from decode_bytes"
        );
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_decode": {"chunks": plan.chunks, "tiles": slots, "tokens_per_tile": w.tokens,
                    "frames": plan.frames, "raw_tile_bytes": 2 * raw.len(), "canvas_bytes": 2 * canvas.len(),
                    "output_bytes": 2 * out.len(), "free_bytes": s.context().mem_get_info()?.0}})
            );
        }
        let copy_frames = |s: &Arc<CudaStream>,
                           from: &CudaSlice<bf16>,
                           from_frames: usize,
                           from_first: usize,
                           to: &mut CudaSlice<bf16>,
                           to_frames: usize,
                           to_first: usize,
                           count: usize|
         -> Result<()> {
            for c in 0..channels {
                let a = (c * from_frames + from_first) * plane;
                let b = (c * to_frames + to_first) * plane;
                s.memcpy_dtod(
                    &from.slice(a..a + count * plane),
                    &mut to.slice_mut(b..b + count * plane),
                )?;
            }
            Ok(())
        };
        let origins: Vec<_> = rows
            .0
            .iter()
            .flat_map(|&row| cols.0.iter().map(move |&col| (row / r, col / r)))
            .collect();
        for i in 0..plan.chunks {
            for (group, origins) in origins.chunks(w.batch).enumerate() {
                self.tile(
                    s,
                    blas,
                    w,
                    &z,
                    (origins, i * chunk),
                    (&raw, group * w.batch),
                )?;
            }
            observe("raw", i, &raw)?;
            let planes = channels * clip_frames;
            unsafe {
                s.launch_builder(&self.kernels.stitch)
                    .arg(&pointer(&raw, s))
                    .arg(&pointer(&canvas, s))
                    .arg(&pointer(&grid, s))
                    .arg(&(rows.0.len() as u32))
                    .arg(&(cols.0.len() as u32))
                    .arg(&(planes as u32))
                    .arg(&(tiles.height as u32))
                    .arg(&(tiles.width as u32))
                    .arg(&(height as u32))
                    .arg(&(width as u32))
                    .launch(flat(planes * plane))?;
            }
            observe("canvas", i, &canvas)?;
            let dst = i * kept;
            if i > 0 && fade > 0 {
                unsafe {
                    s.launch_builder(&self.kernels.frame_blend)
                        .arg(&pointer(&faded, s))
                        .arg(&pointer(&canvas, s))
                        .arg(&pointer(out, s))
                        .arg(&(channels as u32))
                        .arg(&(fade as u32))
                        .arg(&(plane as u64))
                        .arg(&(tail as u32))
                        .arg(&0u32)
                        .arg(&(clip_frames as u32))
                        .arg(&(pre as u32))
                        .arg(&(total as u32))
                        .arg(&(dst as u32))
                        .launch(flat(channels * fade * plane))?;
                }
                copy_frames(
                    s,
                    &canvas,
                    clip_frames,
                    pre + fade,
                    out,
                    total,
                    dst + fade,
                    kept - fade,
                )?;
            } else {
                copy_frames(s, &canvas, clip_frames, pre, out, total, dst, kept)?;
            }
            observe("emitted", i, out)?;
            if spec.token_drop > 0 {
                copy_frames(
                    s,
                    &canvas,
                    clip_frames,
                    lead + pre,
                    &mut faded,
                    tail,
                    0,
                    tail,
                )?;
            }
            observe("tail", i, &faded)?;
        }
        if spec.token_drop > 0 {
            copy_frames(s, &faded, tail, 0, out, total, plan.chunks * kept, tail)?;
        }
        observe("final", plan.chunks, out)?;
        Ok(plan)
    }
}

/// Device bytes `VideoDecoder::decode` allocates besides its output for a `height x width` px video through `tiles`:
/// raw tiles of one clip, the stitched clip canvas and the cross-faded tail, BF16.
pub(crate) fn decode_bytes(
    spec: &VaeSpec,
    tiles: &VaeTiles,
    (height, width): (usize, usize),
) -> usize {
    let clip_frames = (spec.chunk_latents() + spec.overlap_latents()) * spec.temporal_ratio;
    let lead = spec.chunk_latents() * spec.temporal_ratio;
    let tail = clip_frames - lead - spec.frame_pre_padding();
    let channels = spec.out_channels;
    let tile = channels * clip_frames * tiles.height * tiles.width;
    2 * (tiles.count() * tile + channels * (clip_frames + tail) * height * width)
}

/// A device latent video `[latent_channels, frames, height, width]` BF16.
pub(crate) struct LatentVideo<'a> {
    pub data: &'a CudaSlice<bf16>,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

pub(crate) struct TileWork {
    rope: CudaSlice<f32>,
    x: CudaSlice<f32>,
    norm: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    ff: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    latent: CudaSlice<bf16>,
    latent_q: CudaSlice<bf16>,
    patch_rows: CudaSlice<bf16>,
    attention: Vec<(usize, Flash)>,
    frames: usize,
    height: usize,
    width: usize,
    patches: usize,
    tokens: usize,
    batch: usize,
    spec: VaeSpec,
}

impl TileWork {
    pub(crate) fn batch_bytes(
        spec: &VaeSpec,
        shape: (usize, usize, usize),
        batch: usize,
        slots: usize,
    ) -> Result<usize> {
        ensure!(batch > 0 && batch <= slots, "invalid video tile batch");
        let (f, h, w) = shape;
        ensure!(f > 0 && h > 0 && w > 0, "empty video tile");
        let p = (f as u128)
            .checked_mul(h as u128)
            .and_then(|n| n.checked_mul(w as u128))
            .context("video tile overflow")?;
        ensure!(
            p <= u32::MAX as u128,
            "video patch count exceeds layout capacity"
        );
        let t = p + spec.registers as u128 + 1;
        let per = 2 * t * (8 * spec.dim + 3 * spec.ffn + spec.patch_width()) as u128
            + 4 * p * spec.latent_channels as u128;
        let tail = slots % batch;
        let attention = [batch, tail]
            .into_iter()
            .filter(|&n| n != 0)
            .map(|n| 4 * (n as u128 * t * spec.heads as u128 + n as u128 + 1))
            .sum::<u128>();
        Ok(usize::try_from(
            batch as u128 * per + attention + 4 * (t * spec.rope_dims as u128),
        )?)
    }

    pub(crate) fn bytes(spec: &VaeSpec, shape: (usize, usize, usize)) -> usize {
        Self::batch_bytes(spec, shape, 1, 1).expect("validated video tile extent")
    }

    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &VaeSpec,
        shape: (usize, usize, usize),
        (batch, slots): (usize, usize),
    ) -> Result<Self> {
        Self::with_rope_axes(s, spec, shape, (batch, slots), 3)
    }

    fn with_rope_axes(
        s: &Arc<CudaStream>,
        spec: &VaeSpec,
        shape: (usize, usize, usize),
        (batch, slots): (usize, usize),
        axes: usize,
    ) -> Result<Self> {
        let bytes = Self::batch_bytes(spec, shape, batch, slots)?;
        let (frames, height, width) = shape;
        let patches = frames * height * width;
        let tokens = patches + spec.registers + 1;
        let rows = batch * tokens;
        let n = rows * spec.dim;
        let latent = batch * patches * spec.latent_channels;
        let alloc = |elements| unsafe { s.alloc::<bf16>(elements) };
        let mut attention = Vec::with_capacity(2);
        for active in [batch, slots % batch].into_iter().filter(|&n| n > 0) {
            let rows = active * tokens;
            let score = Score::from_tokens(1.0 / (spec.head_dim as f32).sqrt(), None)?;
            let mut flash = Flash::new_with_score(
                s.clone(),
                rows,
                spec.heads,
                spec.heads,
                spec.head_dim,
                score,
                rows,
            )?;
            let offsets = (0..=active)
                .map(|tile| u32::try_from(tile * tokens))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            flash.bind_bf16_windows(&offsets)?;
            attention.push((active, flash));
        }
        let work = Self {
            rope: s.clone_htod(&rotary_table(spec, shape, axes, |x| x))?,
            x: unsafe { s.alloc::<f32>(n)? },
            norm: alloc(n)?,
            o: alloc(n)?,
            q: alloc(n)?,
            k: alloc(n)?,
            v: alloc(n)?,
            attn: alloc(n)?,
            ff: alloc(rows * 2 * spec.ffn)?,
            gated: alloc(rows * spec.ffn)?,
            latent: alloc(latent)?,
            latent_q: alloc(latent)?,
            patch_rows: alloc(rows * spec.patch_width())?,
            attention,
            frames,
            height,
            width,
            patches,
            tokens,
            batch,
            spec: spec.clone(),
        };
        ensure!(
            work.allocated_bytes() == bytes,
            "video work differs from counted plan"
        );
        Ok(work)
    }

    fn allocated_bytes(&self) -> usize {
        4 * (self.rope.len() + self.x.len())
            + 2 * [
                &self.norm,
                &self.o,
                &self.q,
                &self.k,
                &self.v,
                &self.attn,
                &self.ff,
                &self.gated,
                &self.latent,
                &self.latent_q,
                &self.patch_rows,
            ]
            .iter()
            .map(|v| v.len())
            .sum::<usize>()
            + self
                .attention
                .iter()
                .map(|(n, _)| 4 * (n * self.tokens * self.spec.heads + n + 1))
                .sum::<usize>()
    }

    fn pixels(&self) -> usize {
        self.spec.out_channels
            * self.frames
            * self.spec.temporal_ratio
            * self.height
            * self.spec.spatial_ratio
            * self.width
            * self.spec.spatial_ratio
    }
}

fn rotary_table<T: Clone>(
    spec: &VaeSpec,
    (frames, height, width): (usize, usize, usize),
    axes: usize,
    cast: impl Fn(f32) -> T,
) -> Vec<T> {
    let half = spec.rope_dims / 2;
    let freqs = half / axes;
    let step = (2 * axes) as f32 / spec.rope_dims as f32;
    let inv: Vec<f32> = (0..freqs)
        .map(|k| 1.0f32 / spec.rope_theta.powf(k as f32 * step))
        .collect();
    let coordinate = |i: usize, size: usize| 2.0f32 * ((i as f32 + 0.5) / size as f32) - 1.0;
    let turn = (2.0 * std::f64::consts::PI) as f32;
    let tokens = frames * height * width + spec.registers + 1;
    let mut table = vec![cast(0.0); tokens * spec.rope_dims];
    for (row, chunk) in table.chunks_mut(spec.rope_dims).enumerate() {
        if row >= frames * height * width {
            chunk[..half].fill(cast(1.0));
            continue;
        }
        let position = [row / (height * width), row / width % height, row % width];
        let sizes = [frames, height, width];
        for axis in 0..axes {
            let source = 3 - axes + axis;
            let x = coordinate(position[source], sizes[source]);
            for (k, inv) in inv.iter().enumerate() {
                let angle = turn * x * inv;
                chunk[axis * freqs + k] = cast(angle.cos());
                chunk[half + axis * freqs + k] = cast(angle.sin());
            }
        }
    }
    table
}

/// A causal 3D convolution `[out, input, k, k, k]` with bias, published F32, applied to one frame.
struct Conv {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    out: usize,
    input: usize,
    kernel: usize,
}

/// Uploads one published F32 tensor of the F32 set by name and element count.
type LoadF32<'a> = dyn FnMut(&str, usize) -> Result<CudaSlice<f32>> + 'a;

impl Conv {
    fn load(
        load: &mut LoadF32<'_>,
        base: &str,
        (out, input, kernel): (usize, usize, usize),
    ) -> Result<Self> {
        Ok(Self {
            weight: load(&format!("{base}.weight"), out * input * kernel.pow(3))?,
            bias: load(&format!("{base}.bias"), out)?,
            out,
            input,
            kernel,
        })
    }

    /// GEMM reduction width: `input x k^3` im2col columns (the input rows themselves for `k = 1`).
    fn columns(&self) -> usize {
        self.input * self.kernel.pow(3)
    }

    /// Output extent of `size` input pixels: `size` at stride 1 (pad `k / 2` each side); at stride 2 the input carries
    /// one reflected row/column after it and no pad before, `(size + 1 - k) / 2 + 1`.
    fn extent(&self, size: usize, stride: usize) -> usize {
        if stride == 1 {
            size
        } else {
            (size + 1 - self.kernel) / stride + 1
        }
    }
}

struct GroupNorm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

impl GroupNorm {
    fn load(load: &mut LoadF32<'_>, base: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: load(&format!("{base}.weight"), channels)?,
            bias: load(&format!("{base}.bias"), channels)?,
        })
    }
}

struct Resnet {
    norm1: GroupNorm,
    conv1: Conv,
    norm2: GroupNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// One encoder level: its resnets, then the stride-2 downsampling convolution when the level downsamples.
struct EncoderLevel {
    resnets: Vec<Resnet>,
    down: Option<Conv>,
}

struct EncoderKernels {
    im2col: CudaFunction,
    group_norm: CudaFunction,
    bias_residual: CudaFunction,
    pixels: CudaFunction,
    stitch: CudaFunction,
    posterior: CudaFunction,
}

/// Image/video encoder and quant_conv weights, F32 and resident, for single frames.
pub(crate) struct KeyframeEncoder {
    spec: VaeSpec,
    image: bool,
    conv_in: Conv,
    levels: Vec<EncoderLevel>,
    norm_out: GroupNorm,
    conv_out: Conv,
    quant: Conv,
    /// `[pixel mean | pixel std]` and `[latents_mean | latents_std]`, F32.
    pixel_table: CudaSlice<f32>,
    latent_table: CudaSlice<f32>,
    kernels: EncoderKernels,
}

/// F32 mean/std at the posterior extent and normalized latents at the output extent, channel-major.
pub(crate) type KeyframeLatents = [CudaSlice<f32>; 3];

impl KeyframeEncoder {
    pub(crate) fn new(
        d: &Device,
        spec: &VaeSpec,
        vae_dir: &Path,
        manifest: &TensorManifest,
    ) -> Result<Self> {
        Self::load(d, spec, vae_dir, manifest, false)
    }

    fn load(
        d: &Device,
        spec: &VaeSpec,
        vae_dir: &Path,
        manifest: &TensorManifest,
        image: bool,
    ) -> Result<Self> {
        let s = &d.stream;
        let (free, total) = d.ctx.mem_get_info()?;
        let expected: std::collections::BTreeSet<_> = manifest
            .entries
            .keys()
            .filter(|name| name.starts_with("encoder.") || name.starts_with("quant_conv."))
            .cloned()
            .collect();
        if image {
            let required: usize = expected
                .iter()
                .map(|name| manifest.entries[name].bytes)
                .sum();
            ensure!(
                required <= d.workspace_budget(free, free, total / 16)?,
                "image VAE encoder weights need {required} bytes"
            );
        }
        ensure!(
            spec.in_channels == VAE_PIXEL_MEAN.len(),
            "{} input channels, pixel normalisation has {}",
            spec.in_channels,
            VAE_PIXEL_MEAN.len()
        );
        let vision = cubin::module(s.context(), "vision_ops")?;
        let kernels = EncoderKernels {
            im2col: vision.load_function("causal_frame_im2col_f32")?,
            group_norm: vision.load_function("group_norm_silu_f32")?,
            bias_residual: vision.load_function("bias_residual_f32")?,
            pixels: vision.load_function(if image {
                "normalize_hwc_window_f32"
            } else {
                "normalize_u8_window_f32"
            })?,
            stitch: vision.load_function("tile_stitch_f32")?,
            posterior: vision.load_function(if image {
                "gaussian_posterior_crop_f32"
            } else {
                "gaussian_posterior_f32"
            })?,
        };
        let mut source = Source::new(vae_dir, manifest);
        let mut bytes = 0usize;
        let mut consumed = std::collections::BTreeSet::new();
        let load: &mut LoadF32<'_> = &mut |name: &str, elements: usize| {
            ensure!(
                consumed.insert(name.to_owned()),
                "duplicate encoder tensor {name}"
            );
            bytes += 4 * elements;
            source.tensor::<f32>(s, name, elements)
        };
        let first = spec.block_out_channels[0];
        let conv_in = Conv::load(load, "encoder.conv_in", (first, spec.in_channels, 3))?;
        let mut levels = Vec::with_capacity(spec.block_out_channels.len());
        let mut input = first;
        for (i, &out) in spec.block_out_channels.iter().enumerate() {
            let mut resnets = Vec::with_capacity(spec.layers_per_block);
            for j in 0..spec.layers_per_block {
                let base = format!("encoder.down_blocks.{i}.resnets.{j}");
                let channels = if j == 0 { input } else { out };
                resnets.push(Resnet {
                    norm1: GroupNorm::load(load, &format!("{base}.norm1"), channels)?,
                    conv1: Conv::load(load, &format!("{base}.conv1"), (out, channels, 3))?,
                    norm2: GroupNorm::load(load, &format!("{base}.norm2"), out)?,
                    conv2: Conv::load(load, &format!("{base}.conv2"), (out, out, 3))?,
                    shortcut: if channels == out {
                        None
                    } else {
                        Some(Conv::load(
                            load,
                            &format!("{base}.conv_shortcut"),
                            (out, channels, 1),
                        )?)
                    },
                });
            }
            let down = if spec.spatial_factors[i] * spec.temporal_factors[i] > 1 {
                let base = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
                Some(Conv::load(load, &base, (out, out, 3))?)
            } else {
                None
            };
            levels.push(EncoderLevel { resnets, down });
            input = out;
        }
        let latent2 = 2 * spec.latent_channels;
        let norm_out = GroupNorm::load(load, "encoder.norm_out", input)?;
        let conv_out = Conv::load(load, "encoder.conv_out", (latent2, input, 3))?;
        let quant = Conv::load(load, "quant_conv", (latent2, latent2, 1))?;
        ensure!(
            consumed == expected,
            "encoder partial loader tensor set differs"
        );
        let table =
            |a: &[f64], b: &[f64]| -> Vec<f32> { a.iter().chain(b).map(|&v| v as f32).collect() };
        let pixel_table = s.clone_htod(&table(&VAE_PIXEL_MEAN, &VAE_PIXEL_STD))?;
        let latent_table = s.clone_htod(
            &spec
                .latents_mean
                .iter()
                .chain(&spec.latents_std)
                .copied()
                .collect::<Vec<f32>>(),
        )?;
        s.synchronize()?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_encoder_placement": {"mode": "resident", "weight_bytes": bytes,
                    "tensors":consumed.len(),"image":image,
                    "free_before_bytes": free, "free_after_bytes": d.ctx.mem_get_info()?.0, "total_bytes": total}})
            );
        }
        Ok(Self {
            spec: spec.clone(),
            image,
            conv_in,
            levels,
            norm_out,
            conv_out,
            quant,
            pixel_table,
            latent_table,
            kernels,
        })
    }

    /// Every convolution with its stride, in execution order, with the input extent it sees for a `height x width`
    /// tile.
    fn convolutions(&self, (height, width): (usize, usize)) -> Vec<(&Conv, usize, usize, usize)> {
        let (mut h, mut w) = (height, width);
        let mut out = vec![(&self.conv_in, 1, h, w)];
        for level in &self.levels {
            for r in &level.resnets {
                out.push((&r.conv1, 1, h, w));
                out.push((&r.conv2, 1, h, w));
            }
            if let Some(down) = &level.down {
                out.push((down, 2, h, w));
                (h, w) = (down.extent(h, 2), down.extent(w, 2));
            }
        }
        out.push((&self.conv_out, 1, h, w));
        out
    }

    fn group_norm(
        &self,
        s: &Arc<CudaStream>,
        norm: &GroupNorm,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        (pixels, channels): (usize, usize),
    ) -> Result<()> {
        let groups = u32::try_from(self.spec.groups)?;
        unsafe {
            s.launch_builder(&self.kernels.group_norm)
                .arg(&pointer(x, s))
                .arg(&pointer(&norm.weight, s))
                .arg(&pointer(&norm.bias, s))
                .arg(&pointer(y, s))
                .arg(&(pixels as u64))
                .arg(&u32::try_from(channels)?)
                .arg(&groups)
                .arg(&self.spec.encoder_eps)
                .arg(&0u32)
                .arg(&0u32)
                .launch(LaunchConfig {
                    grid_dim: (groups, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn bias_residual(
        &self,
        s: &Arc<CudaStream>,
        (y, offset, count): (&CudaSlice<f32>, usize, usize),
        bias: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
        (channels, inner): (usize, usize),
    ) -> Result<()> {
        let target = pointer(y, s) + (offset * std::mem::size_of::<f32>()) as u64;
        let residual = residual.map_or(0, |r| pointer(r, s));
        unsafe {
            s.launch_builder(&self.kernels.bias_residual)
                .arg(&target)
                .arg(&pointer(bias, s))
                .arg(&residual)
                .arg(&(count as u64))
                .arg(&u32::try_from(channels)?)
                .arg(&(inner as u64))
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// `y [out_h * out_w, out] = conv(x [h * w, input]) + bias (+ residual)`: im2col row chunks that fit `columns`,
    /// each one F32 GEMM against the published `[out, input * k^3]` weight. Returns the output extent.
    fn conv(
        &self,
        (s, blas, columns): (&Arc<CudaStream>, &Blas, &CudaSlice<f32>),
        c: &Conv,
        (x, height, width, stride): (&CudaSlice<f32>, usize, usize, usize),
        y: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
    ) -> Result<(usize, usize)> {
        let (out_h, out_w) = (c.extent(height, stride), c.extent(width, stride));
        let pixels = out_h * out_w;
        let k = c.columns();
        ensure!(
            pixels * c.out <= y.len() && height * width * c.input <= x.len(),
            "convolution {height}x{width}x{} -> {out_h}x{out_w}x{} exceeds its buffers",
            c.input,
            c.out
        );
        let weight = view(
            &c.weight,
            s,
            0,
            [u32::try_from(c.out)?, u32::try_from(k)?],
            [k as u64, 1],
        )?;
        if c.kernel == 1 {
            ensure!(stride == 1, "strided 1x1x1 convolution");
            blas.linear_f32(
                weight,
                view(
                    x,
                    s,
                    0,
                    [u32::try_from(pixels)?, u32::try_from(k)?],
                    [k as u64, 1],
                )?,
                &mut TensorMut(view(
                    y,
                    s,
                    0,
                    [u32::try_from(pixels)?, u32::try_from(c.out)?],
                    [c.out as u64, 1],
                )?),
            )?;
        } else {
            let chunk = (columns.len() / k) / 4 * 4;
            ensure!(chunk > 0, "im2col workspace holds no row of {k} columns");
            let pad = if stride == 1 { c.kernel / 2 } else { 0 };
            for first in (0..pixels).step_by(chunk) {
                let rows = chunk.min(pixels - first);
                unsafe {
                    s.launch_builder(&self.kernels.im2col)
                        .arg(&pointer(x, s))
                        .arg(&pointer(columns, s))
                        .arg(&u32::try_from(c.input)?)
                        .arg(&u32::try_from(height)?)
                        .arg(&u32::try_from(width)?)
                        .arg(&u32::try_from(out_w)?)
                        .arg(&u32::try_from(c.kernel)?)
                        .arg(&u32::try_from(c.kernel)?)
                        .arg(&u32::try_from(stride)?)
                        .arg(&u32::try_from(pad)?)
                        .arg(&(first as u64))
                        .arg(&u32::try_from(rows)?)
                        .arg(&0u32)
                        .launch(flat(rows * k))?;
                }
                blas.linear_f32(
                    weight,
                    view(
                        columns,
                        s,
                        0,
                        [u32::try_from(rows)?, u32::try_from(k)?],
                        [k as u64, 1],
                    )?,
                    &mut TensorMut(view(
                        y,
                        s,
                        first * c.out,
                        [u32::try_from(rows)?, u32::try_from(c.out)?],
                        [c.out as u64, 1],
                    )?),
                )?;
            }
        }
        self.bias_residual(s, (y, 0, pixels * c.out), &c.bias, residual, (c.out, 1))?;
        Ok((out_h, out_w))
    }

    /// Encodes the tile at pixel `origin` of `frame` (`[in_channels, height, width]` u8) into moments slot `slot` of
    /// `moments` (`[slots, 2 * latent_channels, tile_h / r, tile_w / r]`, channel-major).
    fn tile<T: DeviceRepr>(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut EncodeWork,
        (frame, height, width): (&CudaSlice<T>, usize, usize),
        origin: (usize, usize),
        (moments, slot): (&CudaSlice<f32>, usize),
    ) -> Result<()> {
        let channels = self.spec.in_channels;
        let (mut h, mut wd) = (w.height, w.width);
        unsafe {
            s.launch_builder(&self.kernels.pixels)
                .arg(&pointer(frame, s))
                .arg(&pointer(&self.pixel_table, s))
                .arg(&pointer(&w.x, s))
                .arg(&u32::try_from(channels)?)
                .arg(&u32::try_from(height)?)
                .arg(&u32::try_from(width)?)
                .arg(&u32::try_from(origin.0)?)
                .arg(&u32::try_from(origin.1)?)
                .arg(&u32::try_from(h)?)
                .arg(&u32::try_from(wd)?)
                .launch(flat(h * wd * channels))?;
        }
        let ctx = (s, blas, &w.columns);
        self.conv(ctx, &self.conv_in, (&w.x, h, wd, 1), &w.y, None)?;
        std::mem::swap(&mut w.x, &mut w.y);
        for level in &self.levels {
            for r in &level.resnets {
                let px = h * wd;
                self.group_norm(s, &r.norm1, &w.x, &w.normed, (px, r.conv1.input))?;
                let ctx = (s, blas, &w.columns);
                self.conv(ctx, &r.conv1, (&w.normed, h, wd, 1), &w.hidden, None)?;
                self.group_norm(s, &r.norm2, &w.hidden, &w.normed, (px, r.conv1.out))?;
                let residual = match &r.shortcut {
                    Some(shortcut) => {
                        self.conv(ctx, shortcut, (&w.x, h, wd, 1), &w.shortcut, None)?;
                        &w.shortcut
                    }
                    None => &w.x,
                };
                self.conv(ctx, &r.conv2, (&w.normed, h, wd, 1), &w.y, Some(residual))?;
                std::mem::swap(&mut w.x, &mut w.y);
            }
            if let Some(down) = &level.down {
                let ctx = (s, blas, &w.columns);
                (h, wd) = self.conv(ctx, down, (&w.x, h, wd, 2), &w.y, None)?;
                std::mem::swap(&mut w.x, &mut w.y);
            }
        }
        let px = h * wd;
        self.group_norm(
            s,
            &self.norm_out,
            &w.x,
            &w.normed,
            (px, self.conv_out.input),
        )?;
        let ctx = (s, blas, &w.columns);
        self.conv(ctx, &self.conv_out, (&w.normed, h, wd, 1), &w.y, None)?;
        let q = &self.quant;
        let size = q.out * px;
        ensure!(
            (slot + 1) * size <= moments.len(),
            "moments slot {slot} out of range"
        );
        blas.linear_f32(
            view(
                &w.y,
                s,
                0,
                [u32::try_from(px)?, u32::try_from(q.input)?],
                [q.input as u64, 1],
            )?,
            view(
                &q.weight,
                s,
                0,
                [u32::try_from(q.out)?, u32::try_from(q.input)?],
                [q.input as u64, 1],
            )?,
            &mut TensorMut(view(
                moments,
                s,
                slot * size,
                [u32::try_from(q.out)?, u32::try_from(px)?],
                [px as u64, 1],
            )?),
        )?;
        self.bias_residual(s, (moments, slot * size, size), &q.bias, None, (q.out, px))
    }

    /// Encodes one u8 frame `[in_channels, height, width]` through `tiles`: posterior moments of every tile, stitched
    /// in latent space, then mean, std and normalised latents of the sample `mean + std * eps` rounded to F16, with
    /// `eps` `[latent_channels, height / r, width / r]` F32.
    pub(crate) fn encode(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut EncodeWork,
        tiles: &VaeTiles,
        (frame, height, width): (&CudaSlice<u8>, usize, usize),
        eps: &CudaSlice<f32>,
    ) -> Result<KeyframeLatents> {
        self.encode_frame(
            (s, blas),
            w,
            tiles,
            (frame, height, width),
            eps,
            [height, width],
        )
    }

    fn encode_frame<T: DeviceRepr>(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut EncodeWork,
        tiles: &VaeTiles,
        (frame, source_h, source_w): (&CudaSlice<T>, usize, usize),
        eps: &CudaSlice<f32>,
        [height, width]: [usize; 2],
    ) -> Result<KeyframeLatents> {
        let spec = &self.spec;
        let r = spec.spatial_ratio;
        ensure!(
            w.height == tiles.height
                && w.width == tiles.width
                && height.is_multiple_of(r)
                && width.is_multiple_of(r)
                && height <= source_h
                && width <= source_w
                && frame.len() == spec.in_channels * source_h * source_w,
            "encode work {}x{} / frame {height}x{width} do not match the tile grid {}x{}",
            w.height,
            w.width,
            tiles.height,
            tiles.width
        );
        let (lh, lw) = (height / r, width / r);
        let plane = lh * lw;
        let latent = spec.latent_channels;
        ensure!(
            eps.len() == latent * plane,
            "posterior noise holds {} values, not {}",
            eps.len(),
            latent * plane
        );
        let (rows, cols) = (&tiles.rows, &tiles.cols);
        let grid: Vec<u32> = rows
            .0
            .iter()
            .chain(&rows.1)
            .chain(&cols.0)
            .chain(&cols.1)
            .map(|&v| u32::try_from(v / r))
            .collect::<Result<_, _>>()?;
        let grid = s.clone_htod(&grid)?;
        let (th, tw) = (tiles.height / r, tiles.width / r);
        let moments = s.alloc_zeros::<f32>(tiles.count() * 2 * latent * th * tw)?;
        for (ti, &row) in rows.0.iter().enumerate() {
            for (tj, &col) in cols.0.iter().enumerate() {
                self.tile(
                    (s, blas),
                    w,
                    (frame, source_h, source_w),
                    (row, col),
                    (&moments, ti * cols.0.len() + tj),
                )?;
            }
        }
        let canvas = s.alloc_zeros::<f32>(2 * latent * plane)?;
        unsafe {
            s.launch_builder(&self.kernels.stitch)
                .arg(&pointer(&moments, s))
                .arg(&pointer(&canvas, s))
                .arg(&pointer(&grid, s))
                .arg(&(rows.0.len() as u32))
                .arg(&(cols.0.len() as u32))
                .arg(&u32::try_from(2 * latent)?)
                .arg(&u32::try_from(th)?)
                .arg(&u32::try_from(tw)?)
                .arg(&u32::try_from(lh)?)
                .arg(&u32::try_from(lw)?)
                .launch(flat(2 * latent * plane))?;
        }
        let out = [
            s.alloc_zeros::<f32>(latent * plane)?,
            s.alloc_zeros::<f32>(latent * plane)?,
            s.alloc_zeros::<f32>(
                latent
                    * if self.image {
                        (lh / 2 * 2) * (lw / 2 * 2)
                    } else {
                        plane
                    },
            )?,
        ];
        unsafe {
            let pointers = [
                pointer(&canvas, s),
                pointer(eps, s),
                pointer(&self.latent_table, s),
                pointer(&out[0], s),
                pointer(&out[1], s),
                pointer(&out[2], s),
            ];
            let channels = u32::try_from(latent)?;
            let dimensions = [
                u32::try_from(lh)?,
                u32::try_from(lw)?,
                u32::try_from(lh / 2 * 2)?,
                u32::try_from(lw / 2 * 2)?,
            ];
            let plane = plane as u64;
            let mut launch = s.launch_builder(&self.kernels.posterior);
            for p in &pointers {
                launch.arg(p);
            }
            launch.arg(&channels);
            if self.image {
                for d in &dimensions {
                    launch.arg(d);
                }
            } else {
                launch.arg(&plane);
            }
            launch.launch(flat(latent * plane as usize))?;
        }
        Ok(out)
    }
}

/// Activations of one encoder tile forward over `height x width` pixels, pixel-major `[pixels, channels]` F32, and
/// the im2col workspace: the largest convolution of a released `VAE_TILE` tile in one chunk, larger frames in row
/// chunks.
pub(crate) struct EncodeWork {
    height: usize,
    width: usize,
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    shortcut: CudaSlice<f32>,
    columns: CudaSlice<f32>,
}

impl EncodeWork {
    pub(crate) fn new(
        s: &Arc<CudaStream>,
        encoder: &KeyframeEncoder,
        (height, width): (usize, usize),
    ) -> Result<Self> {
        Self::with_budget(s, encoder, (height, width), usize::MAX)
    }

    fn with_budget(
        s: &Arc<CudaStream>,
        encoder: &KeyframeEncoder,
        tile: (usize, usize),
        budget: usize,
    ) -> Result<Self> {
        Self::allocate(s, tile, Self::sizing(encoder, tile)?, budget)
    }

    fn sizing(
        encoder: &KeyframeEncoder,
        (height, width): (usize, usize),
    ) -> Result<(usize, [usize; 2])> {
        let activations = encoder
            .convolutions((height, width))
            .iter()
            .map(|(c, stride, h, w)| {
                (h * w * c.input).max(c.extent(*h, *stride) * c.extent(*w, *stride) * c.out)
            })
            .max()
            .context("encoder without convolutions")?;
        let columns = |(h, w): (usize, usize)| -> Result<usize> {
            encoder
                .convolutions((h, w))
                .iter()
                .filter(|(c, ..)| c.kernel > 1)
                .map(|(c, stride, h, w)| {
                    c.extent(*h, *stride) * c.extent(*w, *stride) * c.columns()
                })
                .max()
                .context("encoder without spatial convolutions")
        };
        let minimum = encoder
            .convolutions((height, width))
            .iter()
            .filter(|(c, ..)| c.kernel > 1)
            .map(|(c, ..)| 4 * c.columns())
            .max()
            .context("encoder without spatial convolutions")?;
        let maximum = columns((height, width))?.min(columns((VAE_TILE, VAE_TILE))?);
        Ok((activations, [minimum, maximum]))
    }

    fn allocate(
        s: &Arc<CudaStream>,
        (height, width): (usize, usize),
        (activations, [minimum, maximum]): (usize, [usize; 2]),
        budget: usize,
    ) -> Result<Self> {
        let fixed = 5 * activations * 4;
        let available = budget
            .checked_sub(fixed)
            .context("encoder budget holds no activation set")?;
        let workspace = maximum.min(available / 4);
        ensure!(
            workspace >= minimum,
            "encoder im2col budget needs at least {} bytes",
            4 * minimum
        );
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"vae_encoder_work":{
                "activation_bytes":fixed,"column_bytes":4 * workspace,"minimum_column_bytes":4 * minimum,
                "budget":budget,"tile":[height,width]}})
            );
        }
        let buffer = || s.alloc_zeros::<f32>(activations);
        Ok(Self {
            height,
            width,
            x: buffer()?,
            y: buffer()?,
            normed: buffer()?,
            hidden: buffer()?,
            shortcut: buffer()?,
            columns: s.alloc_zeros::<f32>(workspace)?,
        })
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use crate::backend::cuda::{closure::Closure, ops::Ops};
    use crate::checkpoint::MappedCheckpoint;
    use safetensors::{Dtype, SafeTensors};
    use std::{collections::BTreeMap, fs, path::PathBuf, rc::Rc};

    fn geometry() -> Result<VaeSpec> {
        Ok(
            crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_str(include_str!(
                "../../../tests/data/minimax-h3-image-vae-config.json"
            ))?)?
            .single_frame(),
        )
    }

    #[test]
    fn image_encoder_shapes_keep_noise_before_patch_crop() -> Result<()> {
        let spec = geometry()?;
        assert_eq!(
            image_encode_shape(&spec, [270, 286], 270 * 286 * 3, 24 * 16 * 17)?,
            ([256, 272], [16, 16])
        );
        assert!(image_encode_shape(&spec, [270, 286], 270 * 286 * 3, 24 * 16 * 16).is_err());
        assert_eq!(
            image_encode_shape(&spec, [32, 32], 32 * 32 * 3, 24 * 2 * 2)?,
            ([32, 32], [2, 2])
        );
        assert!(image_encode_shape(&spec, [31, 32], 31 * 32 * 3, 24 * 2).is_err());
        assert!(image_encode_shape(&spec, [usize::MAX, 32], 0, 0).is_err());
        Ok(())
    }

    #[test]
    fn image_workspace_respects_memory_and_launch_limits() -> Result<()> {
        let spec = geometry()?;
        let tiles = VaeTiles::released(&spec, 4096, 4096)?;
        let canvas = [4096, 4096];
        let full = ImageDecoder::work_size(&spec, &tiles, canvas, tiles.count())?;
        assert_eq!(tiles.count(), 441);
        assert_eq!(
            ImageDecoder::batch(&spec, &tiles, canvas, full, i32::MAX as usize)?,
            (441, full)
        );
        let serial = ImageDecoder::work_size(&spec, &tiles, canvas, 1)?;
        assert!(ImageDecoder::batch(&spec, &tiles, canvas, serial - 1, i32::MAX as usize).is_err());
        assert_eq!(
            ImageDecoder::batch(&spec, &tiles, canvas, serial, i32::MAX as usize)?,
            (1, serial)
        );
        assert!(ImageDecoder::batch(&spec, &tiles, canvas, full, 1).is_err());
        let (bounded, _) = ImageDecoder::batch(&spec, &tiles, canvas, full, 65535)?;
        assert_eq!(bounded, 7);
        assert!(ImageDecoder::batch(&spec, &tiles, canvas, full, 8351).is_err());
        assert_eq!(
            ImageDecoder::batch(&spec, &tiles, canvas, full, 8352)?,
            (1, serial)
        );

        Ok(())
    }

    fn tensor(file: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
        let value = file.tensor(name)?;
        ensure!(
            value.dtype() == Dtype::F32,
            "RUNNER_FAIL {name} fixture dtype"
        );
        Ok(value
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn score(
        label: &str,
        actual: &[f32],
        reference: &[f32],
        threshold: &serde_json::Value,
    ) -> Result<bool> {
        ensure!(
            !actual.is_empty() && actual.len() == reference.len(),
            "RUNNER_FAIL {label} lengths"
        );
        let (max_abs, rmse) = crate::verify::max_abs_rmse(label, actual, reference)?;
        let max_tau = threshold["max_abs"]
            .as_f64()
            .context("RUNNER_FAIL missing max_abs tau")?;
        let rmse_tau = threshold["rmse"]
            .as_f64()
            .context("RUNNER_FAIL missing rmse tau")?;
        ensure!(
            max_tau.is_finite() && rmse_tau.is_finite() && max_tau > 0.0 && rmse_tau > 0.0,
            "RUNNER_FAIL invalid taus"
        );
        let defect = max_abs > max_tau || rmse > rmse_tau;
        println!(
            "IMAGE_VAE_CELL {label} max_abs={max_abs:.10e} rmse={rmse:.10e} tau_max={max_tau:.10e} tau_rmse={rmse_tau:.10e} status={}",
            if defect { "DEFECT" } else { "PASS" }
        );
        Ok(defect)
    }

    fn path(key: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(
            std::env::var_os(key).with_context(|| format!("{key} missing"))?,
        ))
    }

    #[test]
    #[ignore = "GPU; FF_MODEL, longest video capacity and forced BF16 streaming"]
    fn video_decoder_longest_capacity_gpu() -> Result<()> {
        let model = path("FF_MODEL")?;
        let spec = VaeSpec::read(&model)?;
        let component = model.join("vae");
        let manifest = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?
            .minimax_h3_vae_manifest(&spec)?;
        let runtime =
            std::env::temp_dir().join(format!("ff-video-capacity-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            let d = Device::new(0)?;
            let s = &d.stream;
            let ops = Ops::new(&d.ctx)?;
            let mut blas = Blas::new(
                &d,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "BF16 video capacity".into())),
            )?;
            let (height, width) = (576, 1856);
            let r = spec.spatial_ratio;
            let tiles = VaeTiles::released(&spec, height, width)?;
            let tile = (
                spec.chunk_latents() + spec.overlap_latents(),
                tiles.height / r,
                tiles.width / r,
            );
            for (frames, latent_frames, caller) in
                [(345, 102, "MiniMax-H3"), (362, 107, "HyperFlow")]
            {
                let plan = DecodePlan::new(&spec, latent_frames)?;
                ensure!(
                    plan.frames == frames && plan.pad == 0,
                    "RUNNER_FAIL longest frame layout"
                );
                let total = plan.chunks
                    * (spec.chunk_latents() * spec.temporal_ratio - spec.frame_pre_padding())
                    + tile.0 * spec.temporal_ratio
                    - spec.chunk_latents() * spec.temporal_ratio
                    - spec.frame_pre_padding();
                let count = spec.latent_channels * latent_frames * (height / r) * (width / r);
                let output_count = spec.out_channels * total * height * width;
                let scratch = decode_bytes(&spec, &tiles, (height, width));
                let future =
                    TileWork::bytes(&spec, tile) + scratch + 2 * (count + output_count) + (1 << 30);
                for placement in [
                    super::super::Placement::Derived,
                    super::super::Placement::Streamed,
                ] {
                    let mut decoder =
                        VideoDecoder::new(&d, &spec, &component, &manifest, (future, placement))?;
                    decoder.prefix_only = true;
                    let mut work =
                        decoder.work(&d, &tiles, (latent_frames, height / r, width / r))?;
                    decoder.post_quant.bind(&mut blas, work.patches)?;
                    decoder.proj_in.bind(&mut blas, work.patches)?;
                    ensure!(
                        work.allocated_bytes()
                            == TileWork::batch_bytes(&spec, tile, work.batch, tiles.count())?,
                        "RUNNER_FAIL capacity workspace count"
                    );
                    let input: Vec<_> = (0..count)
                        .map(|i| bf16::from_f32(((i * 73 % 65521) as f32 - 32760.0) / 8192.0))
                        .collect();
                    let input_device = s.clone_htod(&input)?;
                    let output = s.alloc_zeros::<bf16>(output_count)?;
                    let raw = s.alloc_zeros::<bf16>(tiles.count() * work.pixels())?;
                    let canvas = s.alloc_zeros::<bf16>(
                        spec.out_channels * tile.0 * spec.temporal_ratio * height * width,
                    )?;
                    let tail = tile.0 * spec.temporal_ratio
                        - spec.chunk_latents() * spec.temporal_ratio
                        - spec.frame_pre_padding();
                    let faded = s.alloc_zeros::<bf16>(spec.out_channels * tail * height * width)?;
                    ensure!(
                        2 * (raw.len() + canvas.len() + faded.len()) == scratch,
                        "RUNNER_FAIL longest stitch count"
                    );
                    let first_frame = (plan.chunks - 1) * spec.chunk_latents();
                    let row = *tiles.rows.0.last().context("no tile rows")? / r;
                    let col = *tiles.cols.0.last().context("no tile columns")? / r;
                    decoder.tile(
                        s,
                        &blas,
                        &mut work,
                        &LatentVideo {
                            data: &input_device,
                            frames: latent_frames,
                            height: height / r,
                            width: width / r,
                        },
                        (&[(row, col)], first_frame),
                        (&raw, tiles.count() - 1),
                    )?;
                    let actual =
                        s.clone_dtoh(&work.latent.slice(..work.patches * spec.latent_channels))?;
                    let mut differing = 0;
                    for (i, value) in actual.iter().enumerate() {
                        let c = i % spec.latent_channels;
                        let token = i / spec.latent_channels;
                        let x = token % tile.2;
                        let y = token / tile.2 % tile.1;
                        let t = token / (tile.1 * tile.2);
                        let index =
                            ((c * latent_frames + first_frame + t) * (height / r) + row + y)
                                * (width / r)
                                + col
                                + x;
                        differing += usize::from(value.to_bits() != input[index].to_bits());
                    }
                    ensure!(
                        differing == 0,
                        "DEFECT longest last-tile BF16 gather {differing}"
                    );
                    let active = 2
                        * (input_device.len()
                            + output.len()
                            + raw.len()
                            + canvas.len()
                            + faded.len())
                        + work.allocated_bytes();
                    println!(
                        "PASS VIDEO_BF16_CAPACITY caller={caller} frames={frames} latent=[1,24,{latent_frames},{},{}] canvas=[{height},{width}] placement={placement:?} linear={} attention={} activation_bytes={active} last_tile_differences={differing} transformer_forwards=0",
                        height / r,
                        width / r,
                        work.batch,
                        work.batch
                    );
                    drop((faded, canvas, raw, output, input_device, work, decoder));
                    s.synchronize()?;
                }
            }
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_MODEL, publisher BF16 prefix floors and FF_H3_CAPTURE"]
    fn video_decoder_prefix_capture_gpu() -> Result<()> {
        let (model, root, capture) = (
            path("FF_MODEL")?,
            path("FF_HYPERFLOW_COMPOSITION")?,
            path("FF_H3_CAPTURE")?,
        );
        let case = std::env::var("FF_HYPERFLOW_CASE")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("BF16_VIDEO_REPLAY_FLOORS.json"))?)?;
        ensure!(
            matches!(case.as_str(), "text" | "first" | "last" | "both")
                && frozen["precision"] == "publisher-bf16"
                && frozen["formula"] == "max(2 * worst_valid_reference_pair, 1e-6)"
                && frozen["native_arrays_read"] == false
                && frozen["controls"]["replay"][&case] == true,
            "RUNNER_FAIL unqualified BF16 prefix floor"
        );
        let spec = VaeSpec::read(&model)?;
        let shape = (
            spec.chunk_latents() + spec.overlap_latents(),
            VAE_TILE / spec.spatial_ratio,
            VAE_TILE / spec.spatial_ratio,
        );
        ensure!(
            shape == (7, 16, 16) && spec.latent_channels == 24 && spec.dim == 2048,
            "RUNNER_FAIL prefix production shape"
        );
        let bytes = fs::read(root.join(format!(
            "arm-a/replay-a/{case}/decoder-video-input.safetensors"
        )))?;
        let input = SafeTensors::deserialize(&bytes)?;
        ensure!(
            input.tensor("value")?.shape() == [1, 24, 37, 48, 84],
            "RUNNER_FAIL prefix input shape"
        );
        let values = tensor(&input, "value")?;
        let runtime = std::env::temp_dir().join(format!("ff-video-prefix-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        fs::create_dir_all(&capture)?;
        let result = (|| -> Result<()> {
            let device = Device::new(0)?;
            let s = &device.stream;
            let ops = Ops::new(&device.ctx)?;
            let mut blas = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "video decoder prefix capture".into())),
            )?;
            let component = model.join("vae");
            let manifest = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?
                .minimax_h3_vae_manifest(&spec)?;
            let mut decoder = VideoDecoder::new(
                &device,
                &spec,
                &component,
                &manifest,
                (
                    TileWork::bytes(&spec, shape),
                    super::super::Placement::Derived,
                ),
            )?;
            decoder.prefix_only = true;
            let mut work = TileWork::new(s, &spec, shape, (1, 1))?;
            decoder.bind(&mut blas, &work)?;
            let rounded: Vec<_> = values.iter().copied().map(bf16::from_f32).collect();
            let input = s.clone_htod(&rounded)?;
            let raw = s.alloc_zeros::<bf16>(work.pixels())?;
            decoder.tile(
                s,
                &blas,
                &mut work,
                &LatentVideo {
                    data: &input,
                    frames: 37,
                    height: 48,
                    width: 84,
                },
                (&[(0, 0)], 0),
                (&raw, 0),
            )?;
            let mut defects = 0;
            for (name, shape, actual) in [
                ("packed", [work.patches, 24], s.clone_dtoh(&work.latent)?),
                (
                    "post_quant",
                    [work.patches, 24],
                    s.clone_dtoh(&work.latent_q)?,
                ),
                (
                    "proj_in",
                    [16, 2048],
                    s.clone_dtoh(&work.o.slice(..16 * 2048))?,
                ),
            ] {
                let actual: Vec<_> = actual.into_iter().map(bf16::to_f32).collect();
                let floor = &frozen["prefix"][format!("replay/{case}/{name}")];
                let tau_abs = floor["tau_abs"]
                    .as_f64()
                    .context("RUNNER_FAIL prefix abs floor")?;
                let tau_rmse = floor["tau_rmse"]
                    .as_f64()
                    .context("RUNNER_FAIL prefix RMSE floor")?;
                ensure!(
                    [tau_abs, tau_rmse]
                        .iter()
                        .all(|x| x.is_finite() && *x >= 1e-6),
                    "RUNNER_FAIL invalid taus for BF16 prefix {name}"
                );
                for arm in ["arm-a", "arm-b", "arm-c"] {
                    let bytes = fs::read(root.join(format!(
                        "{arm}/bf16-video/replay-a/{case}/{name}.safetensors"
                    )))?;
                    let expected = SafeTensors::deserialize(&bytes)?;
                    ensure!(
                        expected.tensor("value")?.shape() == shape,
                        "RUNNER_FAIL BF16 prefix {name} shape"
                    );
                    let expected = tensor(&expected, "value")?;
                    let (max_abs, rmse) = crate::verify::max_abs_rmse(name, &actual, &expected)?;
                    let pass = if name == "packed" {
                        actual
                            .iter()
                            .map(|x| x.to_bits())
                            .eq(expected.iter().map(|x| x.to_bits()))
                    } else {
                        max_abs <= tau_abs && rmse <= tau_rmse
                    };
                    defects += usize::from(!pass);
                    println!(
                        "VIDEO_PREFIX_CELL {} boundary={name} ref={arm} max_abs={max_abs} rmse={rmse} tau_abs={tau_abs} tau_rmse={tau_rmse}",
                        if pass { "PASS" } else { "DEFECT" }
                    );
                }
                let file = capture.join(name).with_extension("safetensors");
                ensure!(!file.exists(), "RUNNER_FAIL prefix capture exists");
                let mut trace = crate::trace::Trace::default();
                trace.set_device(&device.info.name)?;
                trace.add("value".into(), shape.to_vec(), actual)?;
                trace.save(&file)?;
            }
            println!(
                "VIDEO_PREFIX_CAPTURE boundaries=3 comparisons=9 defects={defects} latent_channel_control=true transformer_forwards=0 input_bytes={} work_bytes={} raw_bytes={} blas_workspace_bytes={}",
                2 * values.len(),
                work.allocated_bytes(),
                2 * work.pixels(),
                blas.workspace_bytes()
            );
            ensure!(defects == 0, "DEFECT video BF16 prefix boundaries");
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
    }

    fn input(root: &Path, name: &str, shape: [usize; 2]) -> Result<Vec<f32>> {
        let bytes = fs::read(root.join(format!("inputs/decode-{name}.safetensors")))?;
        let file = SafeTensors::deserialize(&bytes)?;
        ensure!(
            file.tensor("normalized_latent")?.shape() == [1, 24, shape[0] / 16, shape[1] / 16],
            "RUNNER_FAIL {name} input shape"
        );
        tensor(&file, "normalized_latent")
    }

    fn save_taps(
        path: &Path,
        taps: &BTreeMap<String, Vec<f32>>,
        reference: &SafeTensors<'_>,
    ) -> Result<()> {
        let bytes: BTreeMap<_, Vec<u8>> = taps
            .iter()
            .filter(|(name, _)| name.as_str() != "rgb")
            .map(|(name, values)| {
                (
                    name.as_str(),
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                )
            })
            .collect();
        let views: BTreeMap<_, _> = bytes
            .iter()
            .map(|(name, data)| {
                Ok((
                    *name,
                    safetensors::tensor::TensorView::new(
                        Dtype::F32,
                        reference.tensor(name)?.shape().to_vec(),
                        data,
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        safetensors::tensor::serialize_to_file(views, None, path)?;
        Ok(())
    }

    fn reference_gate(root: &Path, model: &Path, runtime: &Path, capture: &Path) -> Result<()> {
        let spec = geometry()?;
        let component = model.join("vae_diffusers");
        let published = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
            &fs::read(component.join("config.json"))?,
        )?)?;
        ensure!(
            published.single_frame() == spec,
            "RUNNER_FAIL published VAE differs from the frozen geometry"
        );
        let checkpoint = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_image_vae_manifest(&published)?;
        ensure!(
            manifest.entries.len() == 511 && decoder_weight_bytes(&manifest) == 807_981_744,
            "RUNNER_FAIL VAE tensor census or decoder dtype inventory"
        );
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("FROZEN_DECODER_FLOORS.json"))?)?;
        ensure!(
            frozen["formula"] == "max(2*worst_valid_pair,1e-6)"
                && frozen["dtype"] == "BF16/F32"
                && frozen["precision_control"]["bf16_off_grid"] == 0
                && frozen["precision_control"]["old_f32_off_grid"]
                    .as_u64()
                    .is_some_and(|n| n != 0),
            "RUNNER_FAIL decoder floor formula"
        );
        let device = Device::new(0)?;
        let sm = cubin::capability(&device.ctx)?;
        ensure!(sm == 89, "RUNNER_FAIL VAE fixture has no sm{sm} arm");
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let mut blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "image VAE gate".into())),
        )?;
        let decoder = ImageDecoder::new(&device, &spec, &component, &manifest)?;
        let mut defects = 0;
        let mut cells = 0;
        for (name, canvas) in [
            ("small", [128, 192]),
            ("overlap", [272, 304]),
            ("2048", [2048, 2048]),
            ("4096", [4096, 4096]),
        ] {
            let z = s.clone_htod(&input(root, name, canvas)?)?;
            let expected_bytes = fs::read(root.join(format!(
                "reference-sm{sm}/output/decode-{name}-canonical0.safetensors"
            )))?;
            let expected = SafeTensors::deserialize(&expected_bytes)?;
            for low in [false, true] {
                let (free, total) = device.ctx.mem_get_info()?;
                let live = device.workspace_budget(free, free, total / 16)?;
                let budget = if low {
                    let tiles = VaeTiles::released(&spec, canvas[0], canvas[1])?;
                    live.min(ImageDecoder::work_size(&spec, &tiles, canvas, 1)?)
                } else {
                    live
                };
                let mut work = decoder.work(&device, canvas, budget)?;
                decoder.core.bind(&mut blas, &work.core)?;
                if low {
                    ensure!(
                        work.core.batch == 1,
                        "RUNNER_FAIL serial budget did not force serial tiles"
                    );
                }
                let selected: std::collections::BTreeSet<_> =
                    [0, work.tiles.count() / 2, work.tiles.count() - 1]
                        .into_iter()
                        .collect();
                let mut taps = BTreeMap::new();
                decoder.launch(
                    s,
                    &blas,
                    &mut work,
                    &z,
                    |role, first, active, (data, offset, rows, width)| {
                        let role = if role == "block_last" {
                            "block23"
                        } else {
                            role
                        };
                        for &tile in &selected {
                            if (first..first + active).contains(&tile) {
                                let start = offset + (tile - first) * rows * width;
                                let end = start + rows * width;
                                let value: Vec<f32> = match data {
                                    DecoderRows::Bf16(data) => {
                                        let mut values = s.clone_dtoh(&data.slice(start..end))?;
                                        if matches!(role, "q0" | "k0") {
                                            for head in values.chunks_exact_mut(spec.head_dim) {
                                                let rstd = cpu_rms_rstd(head, spec.norm_eps);
                                                for value in head {
                                                    *value = bf16::from_f32(value.to_f32() * rstd);
                                                }
                                            }
                                        }
                                        values.into_iter().map(bf16::to_f32).collect()
                                    }
                                    DecoderRows::F32(data) => {
                                        s.clone_dtoh(&data.slice(start..end))?
                                    }
                                };
                                if role == "proj_in" {
                                    ensure!(
                                        value.iter().all(|&x| x.is_finite()
                                            && bf16::from_f32(x).to_f32().to_bits() == x.to_bits()),
                                        "image decoder ingress precision control failed"
                                    );
                                }
                                ensure!(
                                    taps.insert(format!("{role}.tile{tile}"), value).is_none(),
                                    "RUNNER_FAIL duplicate tap"
                                );
                            }
                        }
                        Ok(())
                    },
                )?;
                taps.insert("rgb".into(), s.clone_dtoh(&work.rgb)?);
                ensure!(
                    taps.len()
                        == expected
                            .names()
                            .iter()
                            .filter(|key| !key.starts_with("norm_out."))
                            .count(),
                    "RUNNER_FAIL missing boundary tap"
                );
                for (key, actual) in &taps {
                    defects += usize::from(score(
                        &format!("case={name} low={low} boundary={key}"),
                        actual,
                        &tensor(&expected, key)?,
                        &frozen["cases"][name][key],
                    )?);
                    cells += 1;
                }
                save_taps(
                    &capture.join(format!("{name}-{low}-{}.safetensors", std::process::id())),
                    &taps,
                    &expected,
                )?;
                if name == "2048" && !low {
                    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let launched = decoder.launch(s, &blas, &mut work, &z, |_, _, _, _| Ok(()));
                    let graph = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                    launched?;
                    graph?
                        .context("RUNNER_FAIL image VAE graph empty")?
                        .launch()?;
                    let actual = s.clone_dtoh(&work.rgb)?;
                    let bits = actual
                        .iter()
                        .zip(&taps["rgb"])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    ensure!(
                        actual.iter().all(|v| v.is_finite()),
                        "RUNNER_FAIL image VAE graph nonfinite"
                    );
                    println!(
                        "IMAGE_VAE_GRAPH case=2048 linear={} attention={} tail={} bit_differences={bits} status={}",
                        work.core.batch,
                        work.core.batch,
                        work.tiles.count() % work.core.batch,
                        if bits == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(bits != 0);
                }
            }
        }
        let canvas = [272, 304];
        let latent = input(root, "overlap", canvas)?;
        let z = s.clone_htod(&latent)?;
        let mut work = decoder.work(&device, canvas, usize::MAX)?;
        decoder.core.bind(&mut blas, &work.core)?;
        decoder.launch(s, &blas, &mut work, &z, |_, _, _, _| Ok(()))?;
        let resident = s.clone_dtoh(&work.rgb)?;
        drop(work);
        drop(decoder);
        let ([device_layers, source_layers], _, _) = image_decoder_sizes(&manifest, spec.layers);
        let slots = 2 * device_layers
            .iter()
            .max()
            .context("RUNNER_FAIL no decoder layers")?;
        let source_bytes: usize = source_layers.iter().sum();
        for (name, host_bytes) in [("pinned", source_bytes), ("disk", 0)] {
            let plan = crate::prefill::DensePlacement::new(
                &device_layers,
                Some(&source_layers),
                slots,
                host_bytes,
            )?;
            ensure!(
                plan.resident_bytes == 0
                    && plan.host_bytes == host_bytes
                    && plan.disk_bytes == source_bytes - host_bytes,
                "RUNNER_FAIL image VAE {name} tier not forced"
            );
            let work = ImageDecodeStage::prepare_work(&device, &spec, canvas, 0)?;
            let mut stage = ImageDecodeStage::bind(
                &device,
                &mut blas,
                (&component, &spec, &manifest),
                work,
                plan,
            )?;
            for request in 1..=2 {
                let output = stage.run(&device, &blas, &latent)?;
                ensure!(
                    output.len() == resident.len(),
                    "RUNNER_FAIL image VAE {name} output shape"
                );
                let diff = output
                    .iter()
                    .zip(&resident)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!(
                    "IMAGE_VAE_TIER tier={name} request={request} bit_differences={diff} source_bytes={source_bytes}"
                );
                defects += usize::from(diff != 0);
            }
        }
        println!(
            "{} IMAGE_VAE_DECODER cells={cells} graph_cells=1 tier_cells=4 control=F32_OFF_GRID defects={defects}",
            if defects == 0 { "PASS" } else { "FAIL" }
        );
        ensure!(
            defects == 0,
            "image VAE decoder has {defects} finite numerical defects"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_S0 frozen references, FF_H3_CAPTURE output"]
    fn image_decoder_matches_the_frozen_reference() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = path("FF_H3_S0")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        fs::create_dir_all(&capture)?;
        let runtime = std::env::temp_dir().join(format!("ff-image-vae-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = reference_gate(&root, &model, &runtime, &capture);
        fs::remove_dir_all(&runtime)?;
        result
    }

    #[test]
    #[ignore = "GPU; production and ragged im2col indexing"]
    fn causal_frame_im2col_matches_cpu_indexing_gpu() -> Result<()> {
        struct Case {
            name: &'static str,
            channels: usize,
            height: usize,
            width: usize,
            out_w: usize,
            kt: usize,
            k: usize,
            stride: usize,
            pad: usize,
            first: usize,
            rows: usize,
            layout: u32,
        }
        fn expected(case: &Case, x: &[f32]) -> Vec<f32> {
            let taps = case.k * case.k;
            let frame_cols = case.kt * taps;
            let cols = case.channels * frame_cols;
            let reflect = |i: isize, n: usize| {
                let n = n as isize;
                if i < 0 {
                    -i
                } else if i >= n {
                    2 * (n - 1) - i
                } else {
                    i
                }
            };
            let mut out = vec![0.0f32; case.rows * cols];
            for (i, value) in out.iter_mut().enumerate() {
                let r = i / cols;
                let col = i - r * cols;
                let c = col / frame_cols;
                let rem = col - c * frame_cols;
                if rem / taps + 1 < case.kt {
                    continue;
                }
                let j = rem % taps;
                let p = case.first + r;
                let (oy, ox) = (p / case.out_w, p % case.out_w);
                let iy = reflect(
                    (oy * case.stride + j / case.k) as isize - case.pad as isize,
                    case.height,
                ) as usize;
                let ix = reflect(
                    (ox * case.stride + j % case.k) as isize - case.pad as isize,
                    case.width,
                ) as usize;
                let pixel = iy * case.width + ix;
                *value = x[if case.layout == 1 {
                    c * case.height * case.width + pixel
                } else {
                    pixel * case.channels + c
                }];
            }
            out
        }
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let kernel =
            cubin::module(&device.ctx, "vision_ops")?.load_function("causal_frame_im2col_f32")?;
        let run = |case: &Case, negative: bool| -> Result<(Vec<f32>, Vec<f32>)> {
            let taps = case.k * case.k;
            let cols = case.channels * case.kt * taps;
            let count = case.rows * cols;
            let mut x: Vec<f32> = (0..case.channels * case.height * case.width)
                .map(|i| ((i * 37 % 251) as f32 - 125.0) / 17.0)
                .collect();
            if negative {
                let center = 2 * case.channels;
                x[center] += 1.0;
            }
            let want = expected(case, &x);
            let input = s.clone_htod(&x)?;
            let mut output = s.clone_htod(&vec![-12345.5f32; count + 16])?;
            unsafe {
                s.launch_builder(&kernel)
                    .arg(&input)
                    .arg(&mut output)
                    .arg(&(case.channels as u32))
                    .arg(&(case.height as u32))
                    .arg(&(case.width as u32))
                    .arg(&(case.out_w as u32))
                    .arg(&(case.kt as u32))
                    .arg(&(case.k as u32))
                    .arg(&(case.stride as u32))
                    .arg(&(case.pad as u32))
                    .arg(&(case.first as u64))
                    .arg(&(case.rows as u32))
                    .arg(&case.layout)
                    .launch(flat(count))?;
            }
            let got = s.clone_dtoh(&output)?;
            ensure!(
                got[count..].iter().all(|&v| v == -12345.5),
                "guard overwrite"
            );
            let got = got[..count].to_vec();
            if got != want {
                let differences: Vec<usize> = got
                    .iter()
                    .zip(&want)
                    .enumerate()
                    .filter_map(|(i, (a, b))| (a.to_bits() != b.to_bits()).then_some(i))
                    .take(8)
                    .collect();
                anyhow::bail!(
                    "{} mismatches in {} at {differences:?}",
                    got.len(),
                    case.name
                );
            }
            Ok((got, want))
        };
        for (name, layout) in [("production", 0), ("production-layout1", 1)] {
            run(
                &Case {
                    name,
                    channels: 512,
                    height: 256,
                    width: 256,
                    out_w: 256,
                    kt: 3,
                    k: 3,
                    stride: 1,
                    pad: 1,
                    first: 0,
                    rows: 1024,
                    layout,
                },
                false,
            )?;
        }
        for layout in [0, 1] {
            let ragged = Case {
                name: "ragged",
                channels: 3,
                height: 5,
                width: 7,
                out_w: 7,
                kt: 3,
                k: 3,
                stride: 1,
                pad: 1,
                first: 2,
                rows: 5,
                layout,
            };
            let (base, _) = run(&ragged, false)?;
            let (negative, _) = run(&ragged, true)?;
            ensure!(
                base != negative,
                "one-pixel control did not change the output"
            );
        }
        for layout in [0, 1] {
            run(
                &Case {
                    name: "stride2",
                    channels: 5,
                    height: 6,
                    width: 8,
                    out_w: 4,
                    kt: 3,
                    k: 3,
                    stride: 2,
                    pad: 0,
                    first: 0,
                    rows: 12,
                    layout,
                },
                false,
            )?;
        }
        run(
            &Case {
                name: "pointwise",
                channels: 7,
                height: 5,
                width: 7,
                out_w: 7,
                kt: 1,
                k: 1,
                stride: 1,
                pad: 0,
                first: 0,
                rows: 5,
                layout: 0,
            },
            false,
        )?;
        println!(
            "CAUSAL_FRAME_IM2COL_GATE PASS production=2 layouts=4 edge_cases=5 control=DIFFERS"
        );
        Ok(())
    }

    fn encoder_layouts(device: &Device, encoder: &KeyframeEncoder) -> Result<usize> {
        let s = &device.stream;
        let rgb: Vec<f32> = (0..60).map(|i| ((i * 37) % 251) as f32 / 8.0).collect();
        let pixels = s.clone_htod(&rgb)?;
        let norm = s.clone_htod(&vec![-12345.5f32; 18 + 64])?;
        let plane = 16 * 17;
        let count = 24 * plane;
        let moments = s.clone_htod(&vec![0f32; 2 * count])?;
        let noise: Vec<f32> = (0..count)
            .map(|i| ((i * 37) % 251) as f32 / 137.0 - 1.0)
            .collect();
        let eps = s.clone_htod(&noise)?;
        let mean = s.clone_htod(&vec![-12345.5f32; count + 64])?;
        let std = s.clone_htod(&vec![-12345.5f32; count + 64])?;
        let latent = s.clone_htod(&vec![-12345.5f32; 24 * 256 + 64])?;
        let launch = || -> Result<()> {
            unsafe {
                s.launch_builder(&encoder.kernels.pixels)
                    .arg(&pointer(&pixels, s))
                    .arg(&pointer(&encoder.pixel_table, s))
                    .arg(&(pointer(&norm, s) + 128))
                    .arg(&3u32)
                    .arg(&4u32)
                    .arg(&5u32)
                    .arg(&1u32)
                    .arg(&1u32)
                    .arg(&2u32)
                    .arg(&3u32)
                    .launch(flat(18))?;
                s.launch_builder(&encoder.kernels.posterior)
                    .arg(&pointer(&moments, s))
                    .arg(&pointer(&eps, s))
                    .arg(&pointer(&encoder.latent_table, s))
                    .arg(&(pointer(&mean, s) + 128))
                    .arg(&(pointer(&std, s) + 128))
                    .arg(&(pointer(&latent, s) + 128))
                    .arg(&24u32)
                    .arg(&16u32)
                    .arg(&17u32)
                    .arg(&16u32)
                    .arg(&16u32)
                    .launch(flat(count))?;
            }
            Ok(())
        };
        launch()?;
        let eager = [&norm, &mean, &std, &latent]
            .into_iter()
            .map(|b| s.clone_dtoh(b))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for values in &eager {
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL encoder layout nonfinite"
            );
            ensure!(
                values[..32]
                    .iter()
                    .chain(&values[values.len() - 32..])
                    .all(|v| *v == -12345.5),
                "RUNNER_FAIL encoder layout guard overwritten"
            );
        }
        let mut fractional_bits = 0usize;
        for y in 0..2 {
            for x in 0..3 {
                for c in 0..3 {
                    let v = rgb[((y + 1) * 5 + x + 1) * 3 + c] * (1f32 / 255.0);
                    let expected = (v - VAE_PIXEL_MEAN[c] as f32) / VAE_PIXEL_STD[c] as f32;
                    fractional_bits += (eager[0][32 + (y * 3 + x) * 3 + c].to_bits()
                        ^ expected.to_bits())
                    .count_ones() as usize;
                }
            }
        }
        let moment_errors = eager[1][32..32 + count]
            .iter()
            .filter(|v| **v != 0.0)
            .count()
            + eager[2][32..32 + count]
                .iter()
                .filter(|v| **v != 1.0)
                .count();
        let mut crop_bits = 0usize;
        for c in 0..24 {
            for y in 0..16 {
                for x in 0..16 {
                    let rounded = half::f16::from_f32(noise[(c * 16 + y) * 17 + x]).to_f32();
                    let expected =
                        (rounded - encoder.spec.latents_mean[c]) / encoder.spec.latents_std[c];
                    crop_bits += (eager[3][32 + (c * 16 + y) * 16 + x].to_bits()
                        ^ expected.to_bits())
                    .count_ones() as usize;
                }
            }
        }
        s.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        let launched = launch();
        let graph = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
        launched?;
        graph?
            .context("RUNNER_FAIL encoder layout graph empty")?
            .launch()?;
        let mut graph_bits = 0usize;
        for (buffer, before) in [&norm, &mean, &std, &latent].into_iter().zip(eager) {
            let after = s.clone_dtoh(buffer)?;
            ensure!(
                after.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL encoder layout graph nonfinite"
            );
            graph_bits += after
                .iter()
                .zip(before)
                .map(|(a, b)| (a.to_bits() ^ b.to_bits()).count_ones() as usize)
                .sum::<usize>();
        }
        let defects = [fractional_bits, moment_errors, crop_bits, graph_bits]
            .into_iter()
            .filter(|v| *v != 0)
            .count();
        println!(
            "{} IMAGE_VAE_ENCODER_LAYOUTS exports=2 fractional_rgb_bits={fractional_bits} mean_std_bad_values={moment_errors} posterior_crop_bits={crop_bits} graph_eager_bits={graph_bits} guards=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        Ok(defects)
    }

    fn encoder_reference_gate(
        root: &Path,
        model: &Path,
        runtime: &Path,
        capture: &Path,
    ) -> Result<()> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let spec = geometry()?;
        let component = model.join("vae_diffusers");
        let published = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
            &fs::read(component.join("config.json"))?,
        )?)?;
        ensure!(
            published.single_frame() == spec,
            "RUNNER_FAIL encoder published geometry differs"
        );
        let checkpoint = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_image_vae_manifest(&published)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "image VAE encoder gate".into())),
        )?;
        let encoder = KeyframeEncoder::load(&device, &spec, &component, &manifest, true)?;
        let s = &device.stream;
        let sm = device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
            * 10
            + device
                .ctx
                .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        ensure!(
            [86, 89].contains(&sm),
            "RUNNER_FAIL encoder reference missing for sm{sm}"
        );
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("FROZEN_ENCODER_FLOORS.json"))?)?;
        let mut defects = encoder_layouts(&device, &encoder)?;
        let mut cells = 0;
        for (name, extent) in [
            ("edit0", [1536, 1024]),
            ("edit1", [1536, 1024]),
            ("crop", [270, 286]),
            ("landscape0", [1024, 1536]),
            ("landscape1", [1024, 1536]),
        ] {
            let input_bytes =
                fs::read(root.join(format!("encoder-inputs/encode-{name}.safetensors")))?;
            let input = SafeTensors::deserialize(&input_bytes)?;
            ensure!(
                input.tensor("rgb")?.shape() == [extent[0], extent[1], 3],
                "RUNNER_FAIL encoder RGB fixture shape"
            );
            let rgb = tensor(&input, "rgb")?;
            let noise = tensor(&input, "epsilon")?;
            let shape = image_encode_shape(&spec, extent, rgb.len(), noise.len())?;
            let pixels = s.clone_htod(&rgb)?;
            let eps = s.clone_htod(&noise)?;
            let tiles = VaeTiles::released(&spec, shape.0[0], shape.0[1])?;
            let expected_bytes = fs::read(root.join(format!(
                "reference-sm{sm}-cudnn/output/encode-{name}-canonical0.safetensors"
            )))?;
            let expected = SafeTensors::deserialize(&expected_bytes)?;
            for low in [false, true] {
                let (free, total) = device.ctx.mem_get_info()?;
                let available = device
                    .workspace_budget(free, free, total / 16)?
                    .checked_sub(image_encode_future(&spec, &tiles, shape.0, shape.1)?)
                    .context("RUNNER_FAIL encoder future memory budget")?;
                let budget = if low {
                    available.min(167_772_160 + (64 << 20))
                } else {
                    available
                };
                let mut work =
                    EncodeWork::with_budget(s, &encoder, (tiles.height, tiles.width), budget)?;
                ensure!(
                    !low || work.columns.len() * 4 <= 64 << 20,
                    "RUNNER_FAIL low encoder budget not forced"
                );
                ensure!(
                    EncodeWork::with_budget(
                        s,
                        &encoder,
                        (tiles.height, tiles.width),
                        5 * work.x.len() * 4 - 1
                    )
                    .is_err(),
                    "RUNNER_FAIL encoder accepted insufficient activation budget"
                );
                let outputs = encoder.encode_frame(
                    (s, &blas),
                    &mut work,
                    &tiles,
                    (&pixels, extent[0], extent[1]),
                    &eps,
                    shape.0,
                )?;
                let mut taps = BTreeMap::new();
                for (role, buffer) in ["mean", "std", "normalized_latent"]
                    .into_iter()
                    .zip(&outputs)
                {
                    let actual = s.clone_dtoh(buffer)?;
                    defects += usize::from(score(
                        &format!("encoder case={name} low={low} boundary={role}"),
                        &actual,
                        &tensor(&expected, role)?,
                        &frozen["cases"][name][role],
                    )?);
                    cells += 1;
                    taps.insert(role.to_owned(), actual);
                }
                save_taps(
                    &capture.join(format!(
                        "encoder-{name}-{low}-{}.safetensors",
                        std::process::id()
                    )),
                    &taps,
                    &expected,
                )?;
                println!(
                    "IMAGE_VAE_ENCODER_PLAN case={name} low={low} tiles={} column_bytes={} output={:?}",
                    tiles.count(),
                    4 * work.columns.len(),
                    shape.1
                );
            }
            if name == "edit0" {
                let eps = s.clone_htod(&vec![0f32; noise.len()])?;
                let (free, total) = device.ctx.mem_get_info()?;
                let budget = device
                    .workspace_budget(free, free, total / 16)?
                    .checked_sub(image_encode_future(&spec, &tiles, shape.0, shape.1)?)
                    .context("RUNNER_FAIL control budget")?;
                let mut work =
                    EncodeWork::with_budget(s, &encoder, (tiles.height, tiles.width), budget)?;
                let [_, _, out] = encoder.encode_frame(
                    (s, &blas),
                    &mut work,
                    &tiles,
                    (&pixels, extent[0], extent[1]),
                    &eps,
                    shape.0,
                )?;
                let detected = score(
                    "encoder control=posterior_mode",
                    &s.clone_dtoh(&out)?,
                    &tensor(&expected, "normalized_latent")?,
                    &frozen["cases"][name]["normalized_latent"],
                )?;
                ensure!(detected, "RUNNER_FAIL posterior-mode control not detected");
                println!("IMAGE_VAE_ENCODER_CONTROL posterior_mode_detected=true status=PASS");
            }
        }
        println!(
            "{} IMAGE_VAE_ENCODER cells={cells} modes=2 control=FAIL_AS_REQUIRED defects={defects}",
            if defects == 0 { "PASS" } else { "FAIL" }
        );
        ensure!(
            defects == 0,
            "image VAE encoder has {defects} finite numerical defects"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_S0 encoder references, FF_H3_CAPTURE output"]
    fn image_encoder_matches_the_frozen_reference() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = path("FF_H3_S0")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        fs::create_dir_all(&capture)?;
        let runtime =
            std::env::temp_dir().join(format!("ff-image-vae-encoder-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = encoder_reference_gate(&root, &model, &runtime, &capture);
        fs::remove_dir_all(&runtime)?;
        result
    }

    fn cpu_rms_rstd(row: &[bf16], eps: f32) -> f32 {
        let dim = row.len();
        let active_warps = dim.div_ceil(32).min(8);
        let mut partials = [0.0f32; 8];
        for (w, partial) in partials.iter_mut().enumerate().take(active_warps) {
            let mut warp = [0.0f32; 32];
            for (lane, slot) in warp.iter_mut().enumerate() {
                let idx = w * 32 + lane;
                if idx < dim {
                    let x = row[idx].to_f32();
                    *slot = x * x;
                }
            }
            for step in [16usize, 8, 4, 2, 1] {
                for lane in 0..32 - step {
                    warp[lane] += warp[lane + step];
                }
            }
            *partial = warp[0];
        }
        let e0 = (partials[0] + partials[4]) + (partials[2] + partials[6]);
        let e1 = (partials[1] + partials[5]) + (partials[3] + partials[7]);
        1.0f32 / ((e0 + e1) / (dim as f32) + eps).sqrt()
    }

    #[test]
    #[ignore = "GPU; CPU-reference parity (<=1 BF16 ULP Q/K, 0 ULP V) and negative control for rms_rotary_pack_qkv_bf16"]
    fn video_head_pack_fusion_matches_cpu_reference_bf16_gpu() -> Result<()> {
        struct Case {
            label: &'static str,
            rows: usize,
            tokens: usize,
            heads: usize,
            dim: usize,
            rotary: usize,
            eps: f32,
        }
        fn make_qkv(elements: usize) -> Vec<bf16> {
            (0..3 * elements)
                .map(|i| {
                    let sign = if (i.wrapping_mul(29) ^ (i >> 4)) & 1 == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    let mag = 1.0f32 + ((i.wrapping_mul(37).wrapping_add(11)) % 127) as f32 / 512.0;
                    bf16::from_f32(sign * mag)
                })
                .collect()
        }
        fn make_table(tokens: usize, rotary: usize) -> Vec<f32> {
            let half = rotary / 2;
            (0..tokens * rotary)
                .map(|i| {
                    let p = i % half;
                    let t = i / rotary;
                    let sign = if (p + t).is_multiple_of(3) {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    let angle = sign * ((((p * 17 + t * 13) % 37) + 1) as f32) * 0.005;
                    if (i / half).is_multiple_of(2) {
                        angle.cos()
                    } else {
                        angle.sin()
                    }
                })
                .collect()
        }
        fn cpu_head_pack_reference(qkv: &[bf16], table: &[f32], c: &Case) -> [Vec<bf16>; 3] {
            let elements = c.rows * c.heads * c.dim;
            let half = c.rotary / 2;
            let mut q_ref = vec![bf16::ZERO; elements];
            let mut k_ref = vec![bf16::ZERO; elements];
            let mut v_ref = vec![bf16::ZERO; elements];
            let mut norm = vec![bf16::ZERO; c.dim];
            for row in 0..c.rows {
                let t_row = &table[(row % c.tokens) * c.rotary..];
                for head in 0..c.heads {
                    let src = (row * c.heads + head) * c.dim;
                    let kv = (head * c.rows + row) * c.dim;
                    v_ref[kv..kv + c.dim]
                        .copy_from_slice(&qkv[2 * elements + src..2 * elements + src + c.dim]);
                    for (in_off, out, out_off) in
                        [(src, &mut q_ref, src), (elements + src, &mut k_ref, kv)]
                    {
                        let rstd = cpu_rms_rstd(&qkv[in_off..in_off + c.dim], c.eps);
                        for (dst, &src_v) in norm.iter_mut().zip(&qkv[in_off..in_off + c.dim]) {
                            *dst = bf16::from_f32(src_v.to_f32() * rstd);
                        }
                        for pair in 0..half {
                            let a = norm[pair].to_f32();
                            let b = norm[pair + half].to_f32();
                            let c_v = bf16::from_f32(t_row[pair]).to_f32();
                            let s_v = bf16::from_f32(t_row[pair + half]).to_f32();
                            let ac = bf16::from_f32(a * c_v).to_f32();
                            let bs = bf16::from_f32(b * s_v).to_f32();
                            let bc = bf16::from_f32(b * c_v).to_f32();
                            let a_s = bf16::from_f32(a * s_v).to_f32();
                            out[out_off + pair] = bf16::from_f32(ac - bs);
                            out[out_off + pair + half] = bf16::from_f32(bc + a_s);
                        }
                        out[out_off + c.rotary..out_off + c.dim]
                            .copy_from_slice(&norm[c.rotary..c.dim]);
                    }
                }
            }
            [q_ref, k_ref, v_ref]
        }
        fn bf16_ulp_diff(a: bf16, b: bf16) -> Result<u16> {
            ensure!(
                a.is_finite() && b.is_finite(),
                "C1 parity FAIL non-finite bf16: a={a} b={b}"
            );
            let map = |x: bf16| -> i32 {
                let bits = x.to_bits() as i32;
                if bits & 0x8000 != 0 {
                    0x8000 - (bits & 0x7fff)
                } else {
                    0x8000 + bits
                }
            };
            Ok((map(a) - map(b)).unsigned_abs() as u16)
        }
        fn run_fused(
            s: &Arc<CudaStream>,
            head_pack: &CudaFunction,
            c: &Case,
            qkv_host: &[bf16],
            table_host: &[f32],
        ) -> Result<[Vec<bf16>; 3]> {
            let elements = c.rows * c.heads * c.dim;
            let table_dev = s.clone_htod(table_host)?;
            let qkv_dev = s.clone_htod(qkv_host)?;
            let q_dev = s.alloc_zeros::<bf16>(elements)?;
            let k_dev = s.alloc_zeros::<bf16>(elements)?;
            let v_dev = s.alloc_zeros::<bf16>(elements)?;
            unsafe {
                s.launch_builder(head_pack)
                    .arg(&pointer(&qkv_dev, s))
                    .arg(&pointer(&table_dev, s))
                    .arg(&pointer(&q_dev, s))
                    .arg(&pointer(&k_dev, s))
                    .arg(&pointer(&v_dev, s))
                    .arg(&(c.rows as u32))
                    .arg(&(c.tokens as u32))
                    .arg(&(c.heads as u32))
                    .arg(&(c.dim as u32))
                    .arg(&(c.rotary as u32))
                    .arg(&c.eps)
                    .launch(ops::grid((c.rows * c.heads).div_ceil(8), 256))?;
            }
            Ok([
                s.clone_dtoh(&q_dev)?,
                s.clone_dtoh(&k_dev)?,
                s.clone_dtoh(&v_dev)?,
            ])
        }

        let device = Device::new(0)?;
        let s = &device.stream;
        let head_pack =
            cubin::module(&device.ctx, "vision_ops")?.load_function("rms_rotary_pack_qkv_bf16")?;
        let cases = [
            Case {
                label: "prod",
                rows: 50_316,
                tokens: 1_797,
                heads: 32,
                dim: 64,
                rotary: 48,
                eps: 1.0e-5,
            },
            Case {
                label: "ragged_tail_96",
                rows: 91,
                tokens: 13,
                heads: 7,
                dim: 96,
                rotary: 64,
                eps: 1.0e-6,
            },
        ];
        let names = ["q_out", "k_out", "v_out"];
        let max_ulps = [1u16, 1u16, 0u16];
        for c in &cases {
            let elements = c.rows * c.heads * c.dim;
            let qkv_host = make_qkv(elements);
            let table_host = make_table(c.tokens, c.rotary);
            let expected = cpu_head_pack_reference(&qkv_host, &table_host, c);
            let got = run_fused(s, &head_pack, c, &qkv_host, &table_host)?;
            let mut worst = [0u16; 3];
            let mut nz = [0usize; 3];
            for (ti, ((tensor_name, a), b)) in names
                .iter()
                .zip(got.iter())
                .zip(expected.iter())
                .enumerate()
            {
                let mut bad = Vec::new();
                for (i, (&ga, &eb)) in a.iter().zip(b.iter()).enumerate() {
                    let ulp = if ti == 2 {
                        u16::from(ga.to_bits() != eb.to_bits())
                    } else {
                        bf16_ulp_diff(ga, eb)?
                    };
                    if ulp > 0 {
                        nz[ti] += 1;
                        worst[ti] = worst[ti].max(ulp);
                    }
                    if ulp > max_ulps[ti] && bad.len() < 8 {
                        bad.push((i, ulp, ga.to_bits(), eb.to_bits()));
                    }
                }
                ensure!(
                    bad.is_empty(),
                    "C1 CPU-ref FAIL case={} tensor={} max_ulp={} limit={} first={:?}",
                    c.label,
                    tensor_name,
                    worst[ti],
                    max_ulps[ti],
                    bad
                );
            }
            println!(
                "PASS C1_CPU_REF case={} rows={} tokens={} heads={} dim={} rotary={} max_ulp={:?} nonzero_ulp={:?}",
                c.label, c.rows, c.tokens, c.heads, c.dim, c.rotary, worst, nz
            );
        }
        let prod = &cases[0];
        let elements = prod.rows * prod.heads * prod.dim;
        let qkv_host = make_qkv(elements);
        let table_host = make_table(prod.tokens, prod.rotary);
        let expected = cpu_head_pack_reference(&qkv_host, &table_host, prod);
        let mut perturbed = qkv_host;
        let (bad_row, bad_head, bad_d) = (17usize, 5usize, 3usize);
        let src_off = (bad_row * prod.heads + bad_head) * prod.dim + bad_d;
        for j in 0..3 {
            let idx = j * elements + src_off;
            perturbed[idx] = bf16::from_f32(perturbed[idx].to_f32() + 1.0);
        }
        let got = run_fused(s, &head_pack, prod, &perturbed, &table_host)?;
        let q_range = (bad_row * prod.heads + bad_head) * prod.dim
            ..(bad_row * prod.heads + bad_head + 1) * prod.dim;
        let kv_range = (bad_head * prod.rows + bad_row) * prod.dim
            ..(bad_head * prod.rows + bad_row + 1) * prod.dim;
        let ranges = [&q_range, &kv_range, &kv_range];
        let mut bad_counts = [0usize; 3];
        for (ti, ((tensor_name, a), b)) in names
            .iter()
            .zip(got.iter())
            .zip(expected.iter())
            .enumerate()
        {
            for (i, (&ga, &eb)) in a.iter().zip(b.iter()).enumerate() {
                let ulp = if ti == 2 {
                    u16::from(ga.to_bits() != eb.to_bits())
                } else {
                    bf16_ulp_diff(ga, eb)?
                };
                if ulp > max_ulps[ti] {
                    ensure!(
                        ranges[ti].contains(&i),
                        "C1 negative control FAIL tensor={} spilled outside [{bad_row}, {bad_head}] at idx={i} ulp={ulp}",
                        tensor_name
                    );
                    bad_counts[ti] += 1;
                }
            }
            ensure!(
                bad_counts[ti] > 0,
                "C1 negative control FAIL tensor={} missed perturbation",
                tensor_name
            );
        }
        println!(
            "PASS C1_NEGATIVE_CONTROL row={} head={} q_exceed={} k_exceed={} v_diff={}",
            bad_row, bad_head, bad_counts[0], bad_counts[1], bad_counts[2]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a reserved GPU window"]
    fn vae_residual_rms_fusion_matches_parent_gpu() -> Result<()> {
        use cudarc::driver::sys::{CUdevice_attribute as DA, CUfunction_attribute as FA};
        let d = Device::new(0)?;
        let s = &d.stream;
        let vision = cubin::module(&d.ctx, "vision_ops")?;
        let residual = vision.load_function("scaled_residual_f32")?;
        let rms = vision.load_function("vision_rms_scale_f32")?;
        let fused = vision.load_function("scaled_residual_rms_f32")?;

        let max_threads_sm = u32::try_from(
            d.ctx
                .attribute(DA::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?,
        )?;
        let local_bytes = fused.get_attribute(FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
        let regs = fused.get_attribute(FA::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
        let smem = fused.get_attribute(FA::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
        let active_blocks = fused.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
        let active_warps = active_blocks * 8;
        println!(
            "H3_VAE_RESIDUAL_RMS_OCCUPANCY threads=256 active_blocks={active_blocks} active_warps={active_warps} regs={regs} smem={smem} local={local_bytes}"
        );
        ensure!(
            local_bytes == 0,
            "scaled_residual_rms_f32 spills to local memory ({local_bytes} B)"
        );
        ensure!(
            active_blocks * 256 >= max_threads_sm.min(1536),
            "scaled_residual_rms_f32 occupancy {active_warps} warps/SM below target on max_threads={max_threads_sm}"
        );

        let cases = [
            ("video_prod", 50316usize, 2048usize, false),
            ("video_ragged", 17usize, 768usize, false),
            ("image_prod_f32out", 31581usize, 1024usize, true),
            ("image_ragged_f32out", 19usize, 512usize, true),
        ];
        for (label, rows, width, out_f32) in cases {
            let count = rows * width;
            let eps = 1.0e-6f32;
            let x_host: Vec<f32> = (0..count)
                .map(|i| (((i * 29 + 7) % 509) as f32 - 254.0) / 64.0)
                .collect();
            let delta_host: Vec<bf16> = (0..count)
                .map(|i| bf16::from_f32((((i * 43 + 11) % 389) as f32 - 194.0) / 128.0))
                .collect();
            let res_scale_host: Vec<f32> = (0..width)
                .map(|i| 0.25 + ((i * 17) % 61) as f32 / 64.0)
                .collect();
            let norm_scale_host: Vec<f32> = (0..width)
                .map(|i| 0.5 + ((i * 13 + 3) % 47) as f32 / 32.0)
                .collect();

            let x_ref = s.clone_htod(&x_host)?;
            let x_got = s.clone_htod(&x_host)?;
            let delta = s.clone_htod(&delta_host)?;
            let res_scale = s.clone_htod(&res_scale_host)?;
            let norm_scale = s.clone_htod(&norm_scale_host)?;
            let y_f32_ref = s.alloc_zeros::<f32>(if out_f32 { count } else { 1 })?;
            let y_bf16_ref = s.alloc_zeros::<bf16>(if out_f32 { 1 } else { count })?;
            let y_f32_got = s.alloc_zeros::<f32>(if out_f32 { count } else { 1 })?;
            let y_bf16_got = s.alloc_zeros::<bf16>(if out_f32 { 1 } else { count })?;

            let out_f32_ref_ptr = if out_f32 { pointer(&y_f32_ref, s) } else { 0 };
            let out_bf16_ref_ptr = if out_f32 { 0 } else { pointer(&y_bf16_ref, s) };
            let out_f32_got_ptr = if out_f32 { pointer(&y_f32_got, s) } else { 0 };
            let out_bf16_got_ptr = if out_f32 { 0 } else { pointer(&y_bf16_got, s) };
            let width_u = u32::try_from(width)?;
            let count_u = u64::try_from(count)?;
            let row_grid = ops::grid(rows, 256);

            unsafe {
                s.launch_builder(&residual)
                    .arg(&pointer(&x_ref, s))
                    .arg(&delta)
                    .arg(&res_scale)
                    .arg(&count_u)
                    .arg(&width_u)
                    .launch(flat(count))?;
                s.launch_builder(&rms)
                    .arg(&pointer(&x_ref, s))
                    .arg(&norm_scale)
                    .arg(&1u32)
                    .arg(&out_f32_ref_ptr)
                    .arg(&out_bf16_ref_ptr)
                    .arg(&width_u)
                    .arg(&eps)
                    .launch(row_grid)?;
                s.launch_builder(&fused)
                    .arg(&pointer(&x_got, s))
                    .arg(&delta)
                    .arg(&res_scale)
                    .arg(&norm_scale)
                    .arg(&out_f32_got_ptr)
                    .arg(&out_bf16_got_ptr)
                    .arg(&width_u)
                    .arg(&eps)
                    .launch(row_grid)?;
            }

            let res_ref_h = s.clone_dtoh(&x_ref)?;
            let res_got_h = s.clone_dtoh(&x_got)?;
            let res_diff = res_ref_h
                .iter()
                .zip(&res_got_h)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let out_diff = if out_f32 {
                let a = s.clone_dtoh(&y_f32_ref)?;
                let b = s.clone_dtoh(&y_f32_got)?;
                a.iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            } else {
                let a = s.clone_dtoh(&y_bf16_ref)?;
                let b = s.clone_dtoh(&y_bf16_got)?;
                a.iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            };
            ensure!(
                res_diff == 0 && out_diff == 0,
                "scaled_residual_rms_f32 mismatch on {label}: res_diff={res_diff} out_diff={out_diff}"
            );

            if label == "video_ragged" {
                let norm_ref_h = s.clone_dtoh(&y_bf16_ref)?;
                // Control A: perturb delta[row=7, col=13]; assert zero cross-row bleed.
                let (p_row, p_col) = (7usize, 13usize);
                let idx = p_row * width + p_col;
                let mut delta_mut_h = delta_host.clone();
                delta_mut_h[idx] = bf16::from_f32(delta_mut_h[idx].to_f32() + 1.0);
                let delta_mut = s.clone_htod(&delta_mut_h)?;
                let x_mut = s.clone_htod(&x_host)?;
                let y_mut = s.alloc_zeros::<bf16>(count)?;
                unsafe {
                    s.launch_builder(&fused)
                        .arg(&pointer(&x_mut, s))
                        .arg(&delta_mut)
                        .arg(&res_scale)
                        .arg(&norm_scale)
                        .arg(&0u64)
                        .arg(&pointer(&y_mut, s))
                        .arg(&width_u)
                        .arg(&eps)
                        .launch(row_grid)?;
                }
                let x_mut_h = s.clone_dtoh(&x_mut)?;
                let y_mut_h = s.clone_dtoh(&y_mut)?;
                let mut res_hit = 0usize;
                let mut res_bleed = 0usize;
                for (i, (a, b)) in res_ref_h.iter().zip(&x_mut_h).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        if i == idx {
                            res_hit += 1;
                        } else {
                            res_bleed += 1;
                        }
                    }
                }
                let row_range = p_row * width..(p_row + 1) * width;
                let mut norm_hit = 0usize;
                let mut norm_bleed = 0usize;
                for (i, (a, b)) in norm_ref_h.iter().zip(&y_mut_h).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        if row_range.contains(&i) {
                            norm_hit += 1;
                        } else {
                            norm_bleed += 1;
                        }
                    }
                }
                ensure!(
                    res_hit == 1 && res_bleed == 0 && norm_hit > 0 && norm_bleed == 0,
                    "delta control failed: res=({res_hit},{res_bleed}) norm=({norm_hit},{norm_bleed})"
                );

                // Control B: perturb norm_scale[col=13]; assert residual unchanged and norm differs only at col=13.
                let mut ns_mut_h = norm_scale_host.clone();
                ns_mut_h[p_col] += 0.5;
                let ns_mut = s.clone_htod(&ns_mut_h)?;
                let x_ns = s.clone_htod(&x_host)?;
                let y_ns = s.alloc_zeros::<bf16>(count)?;
                unsafe {
                    s.launch_builder(&fused)
                        .arg(&pointer(&x_ns, s))
                        .arg(&delta)
                        .arg(&res_scale)
                        .arg(&ns_mut)
                        .arg(&0u64)
                        .arg(&pointer(&y_ns, s))
                        .arg(&width_u)
                        .arg(&eps)
                        .launch(row_grid)?;
                }
                let x_ns_h = s.clone_dtoh(&x_ns)?;
                let y_ns_h = s.clone_dtoh(&y_ns)?;
                let ns_res_diff = res_ref_h
                    .iter()
                    .zip(&x_ns_h)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let mut col_hit = 0usize;
                let mut col_bleed = 0usize;
                for (i, (a, b)) in norm_ref_h.iter().zip(&y_ns_h).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        if i % width == p_col {
                            col_hit += 1;
                        } else {
                            col_bleed += 1;
                        }
                    }
                }
                ensure!(
                    ns_res_diff == 0 && col_hit == rows && col_bleed == 0,
                    "norm_scale control failed: res_diff={ns_res_diff} col=({col_hit},{col_bleed})"
                );
                println!(
                    "H3_VAE_RESIDUAL_RMS_CONTROLS res_hit={res_hit}/{res_bleed} norm_row_hit={norm_hit}/{norm_bleed} col_hit={col_hit}/{col_bleed}"
                );
            }
            println!(
                "H3_VAE_RESIDUAL_RMS_PARITY case={label} rows={rows} width={width} out_f32={out_f32} res_diff={res_diff} out_diff={out_diff}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_MODEL, Derived vs Streamed full video decoder block parity"]
    fn video_decoder_streamed_matches_derived_gpu() -> Result<()> {
        let model = path("FF_MODEL")?;
        let spec = VaeSpec::read(&model)?;
        let component = model.join("vae");
        let manifest = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?
            .minimax_h3_vae_manifest(&spec)?;
        let shape = (
            spec.chunk_latents() + spec.overlap_latents(),
            VAE_TILE / spec.spatial_ratio,
            VAE_TILE / spec.spatial_ratio,
        );
        let count = spec.latent_channels * shape.0 * shape.1 * shape.2;
        let input_host: Vec<_> = (0..count)
            .map(|i| bf16::from_f32((((i * 73 + 17) % 65521) as f32 - 32760.0) / 8192.0))
            .collect();
        let runtime =
            std::env::temp_dir().join(format!("ff-video-streamed-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            let d = Device::new(0)?;
            let s = &d.stream;
            let ops = Ops::new(&d.ctx)?;
            let mut blas = Blas::new(
                &d,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "BF16 video full tile".into())),
            )?;
            let run = |blas: &mut Blas, placement| -> Result<Vec<bf16>> {
                let future = TileWork::bytes(&spec, shape) + (1 << 30);
                let decoder =
                    VideoDecoder::new(&d, &spec, &component, &manifest, (future, placement))?;
                let mut work = TileWork::new(s, &spec, shape, (1, 1))?;
                decoder.bind(blas, &work)?;
                let input = s.clone_htod(&input_host)?;
                let raw = s.alloc_zeros::<bf16>(work.pixels())?;
                decoder.tile(
                    s,
                    blas,
                    &mut work,
                    &LatentVideo {
                        data: &input,
                        frames: shape.0,
                        height: shape.1,
                        width: shape.2,
                    },
                    (&[(0, 0)], 0),
                    (&raw, 0),
                )?;
                let out = s.clone_dtoh(&raw)?;
                drop((raw, input, work, decoder));
                s.synchronize()?;
                Ok(out)
            };
            let derived = run(&mut blas, super::super::Placement::Derived)?;
            let streamed = run(&mut blas, super::super::Placement::Streamed)?;
            let differing = derived
                .iter()
                .zip(&streamed)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "VIDEO_DERIVED_VS_STREAMED_FULL_TILE len={} differing={differing}",
                derived.len()
            );
            ensure!(
                differing == 0,
                "DEFECT Derived vs Streamed full-block parity mismatch: {differing} differing bits"
            );
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
    }
}
