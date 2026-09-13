use crate::vae_tiling::split_tiles;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, ops};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::BufReader,
    path::Path,
};

/// MiniMax-H3 resets a fresh posterior generator to this seed for every visual condition.
pub const CONDITION_ENCODE_SEED: u64 = 42;

const TILE_SIZE: usize = 256;
const TILE_OVERLAP: usize = 64;
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The encoder-relevant fields in the released `vae/config.json`.
///
/// Decoder fields are intentionally ignored by serde. The encoder and decoder share the same
/// root VAE config and safetensors index; opening a separate or synthesized encoder checkpoint
/// would silently violate the released MiniMax-H3 conditioning recipe.
#[derive(Clone, Debug, Deserialize)]
pub struct VideoVaeEncoderConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub in_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub spatial_downsample_factors: Vec<usize>,
    pub temporal_downsample_factors: Vec<usize>,
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub spatial_padding_mode: String,
    pub clip_length: usize,
    pub token_drop: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl VideoVaeEncoderConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read visual VAE config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid visual VAE config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn spatial_ratio(&self) -> Result<usize> {
        checked_product(
            &self.spatial_downsample_factors,
            "spatial compression ratio",
        )
    }

    pub fn temporal_ratio(&self) -> Result<usize> {
        checked_product(
            &self.temporal_downsample_factors,
            "temporal compression ratio",
        )
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.class_name == "AutoencoderKLMiniMaxH3",
            "unsupported visual VAE class {}",
            self.class_name
        );
        anyhow::ensure!(
            self.in_channels == 3,
            "MiniMax-H3 visual VAE encoder requires three RGB input channels"
        );
        anyhow::ensure!(
            self.latent_channels > 0,
            "VAE latent channels must be non-zero"
        );
        anyhow::ensure!(
            !self.block_out_channels.is_empty(),
            "VAE encoder must contain at least one down block"
        );
        anyhow::ensure!(
            self.layers_per_block > 0,
            "VAE encoder layers_per_block must be non-zero"
        );
        anyhow::ensure!(
            self.block_out_channels.len() == self.spatial_downsample_factors.len()
                && self.block_out_channels.len() == self.temporal_downsample_factors.len(),
            "VAE encoder channel and downsample-factor lists must have equal lengths"
        );
        anyhow::ensure!(
            self.norm_num_groups > 0,
            "VAE encoder norm_num_groups must be non-zero"
        );
        anyhow::ensure!(
            self.norm_eps.is_finite() && self.norm_eps > 0.,
            "VAE encoder normalization epsilon must be finite and positive"
        );
        anyhow::ensure!(
            self.block_out_channels
                .iter()
                .all(|channels| *channels > 0 && channels.is_multiple_of(self.norm_num_groups)),
            "every VAE encoder block width must be non-zero and divisible by norm_num_groups"
        );
        anyhow::ensure!(
            self.spatial_downsample_factors
                .iter()
                .all(|factor| matches!(factor, 1 | 2)),
            "VAE encoder only supports spatial downsample factors 1 and 2"
        );
        anyhow::ensure!(
            self.temporal_downsample_factors
                .iter()
                .all(|factor| matches!(factor, 1 | 2)),
            "VAE encoder only supports temporal downsample factors 1 and 2"
        );
        anyhow::ensure!(
            self.spatial_padding_mode == "reflect",
            "unsupported VAE encoder spatial padding mode {}",
            self.spatial_padding_mode
        );
        anyhow::ensure!(
            self.clip_length > 0,
            "VAE encoder clip_length must be non-zero"
        );
        anyhow::ensure!(
            self.latents_mean.len() == self.latent_channels
                && self.latents_std.len() == self.latent_channels,
            "visual latent statistics must have one value per latent channel"
        );
        anyhow::ensure!(
            self.latents_mean.iter().all(|value| value.is_finite())
                && self
                    .latents_std
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.),
            "visual latent statistics must be finite and standard deviations positive"
        );
        let spatial_ratio = self.spatial_ratio()?;
        let temporal_ratio = self.temporal_ratio()?;
        let tokens_per_chunk = self.clip_length.div_ceil(temporal_ratio);
        anyhow::ensure!(
            self.token_drop < tokens_per_chunk,
            "VAE token_drop {} leaves no latent frames in a {}-frame clip",
            self.token_drop,
            self.clip_length
        );
        anyhow::ensure!(
            TILE_SIZE.is_multiple_of(spatial_ratio) && TILE_OVERLAP.is_multiple_of(spatial_ratio),
            "released 256/64 VAE tile geometry must be latent-aligned"
        );
        Ok(())
    }
}

/// F32, layer-streamed implementation of the released MiniMax-H3 visual VAE encoder.
///
/// `ModelWeights` only maps/reads the official root VAE shards. Every convolution or norm loads
/// just its own tensors onto `device`, computes in F32, and releases them before the next stage.
/// Spatial tiles are encoded one at a time and moved back to CPU before stitching.
pub struct StreamedVideoVaeEncoder {
    weights: ModelWeights,
    config: VideoVaeEncoderConfig,
    device: Device,
    tile_size: usize,
    tile_overlap: usize,
}

impl StreamedVideoVaeEncoder {
    /// Opens the released `MiniMax-H3/vae` component directly.
    pub fn open(
        vae_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
    ) -> Result<Self> {
        let vae_dir = vae_dir.as_ref();
        let config = VideoVaeEncoderConfig::from_file(vae_dir.join("config.json"))?;
        let weights = ModelWeights::open(vae_dir, source, cache_policy)?;
        validate_weight_inventory(&weights, &config)?;
        Ok(Self {
            weights,
            config,
            device,
            tile_size: TILE_SIZE,
            tile_overlap: TILE_OVERLAP,
        })
    }

    /// Convenience entry point when the caller holds the root `MiniMax-H3` directory.
    pub fn open_from_model_root(
        model_root: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
    ) -> Result<Self> {
        Self::open(
            model_root.as_ref().join("vae"),
            source,
            cache_policy,
            device,
        )
    }

    pub fn config(&self) -> &VideoVaeEncoderConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reads one PNG without modifying it and returns its released-recipe conditioning latent.
    pub fn encode_png_keyframe(&self, path: impl AsRef<Path>) -> Result<Tensor> {
        let path = path.as_ref();
        let pixels = read_png_rgb8(path)?;
        self.encode_keyframe_pixels(&pixels)
            .with_context(|| format!("failed to encode PNG keyframe {}", path.display()))
    }

    /// Alias whose word order matches pipeline call sites.
    pub fn encode_keyframe_png(&self, path: impl AsRef<Path>) -> Result<Tensor> {
        self.encode_png_keyframe(path)
    }

    /// Encodes a single `[1, 3, 1, H, W]` 0..255 tensor as a keyframe condition.
    pub fn encode_keyframe_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, _, _) = pixels.dims5()?;
        anyhow::ensure!(
            batch == 1 && channels == self.config.in_channels && frames == 1,
            "a keyframe must have shape [1, {}, 1, H, W]",
            self.config.in_channels
        );
        self.encode_video_pixels(pixels)
    }

    /// Tensor API for image or multi-frame Ref2VA conditions.
    ///
    /// Input is `[B, 3, T, H, W]` in the 0..255 pixel range. Output is an F32 CPU tensor after
    /// ImageNet normalization, causal/tiled VAE encoding, posterior sampling with a fresh seed 42,
    /// F16-to-F32 round-trip, and per-channel latent normalization.
    pub fn encode_video_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let (_, channels, frames, height, width) = pixels.dims5()?;
        anyhow::ensure!(
            channels == self.config.in_channels,
            "visual condition has {channels} channels, expected {}",
            self.config.in_channels
        );
        anyhow::ensure!(
            frames > 0 && height > 0 && width > 0,
            "visual condition dimensions must be non-zero"
        );
        self.validate_spatial_input(height, width)?;
        let pixels = imagenet_normalize(pixels, self.config.in_channels)?;
        let moments = self.encode_moments(&pixels)?;
        self.sample_and_normalize(&moments)
    }

    /// Encodes already ImageNet-normalized F32 pixels to posterior mean/log-variance moments.
    ///
    /// This mirrors diffusers `_encode`: a lone frame takes the direct spatial path; multiple
    /// frames are padded by repeating the final frame, encoded in `clip_length` chunks, concatenated,
    /// and finally have `token_drop` latent frames removed once from the complete sequence.
    pub fn encode_moments(&self, imagenet_pixels: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, height, width) = imagenet_pixels.dims5()?;
        anyhow::ensure!(
            batch > 0 && channels == self.config.in_channels,
            "normalized visual input must have shape [B, {}, T, H, W]",
            self.config.in_channels
        );
        anyhow::ensure!(
            frames > 0 && height > 0 && width > 0,
            "normalized visual input dimensions must be non-zero"
        );
        anyhow::ensure!(
            imagenet_pixels.dtype() == DType::F32,
            "already-normalized visual VAE input must be F32"
        );
        self.validate_spatial_input(height, width)?;
        let pixels = imagenet_pixels.clone();
        if frames == 1 {
            return self.encode_clip_tiled(&pixels);
        }

        let clip_length = self.config.clip_length;
        let pad_frames = (clip_length - frames % clip_length) % clip_length;
        let pixels = if pad_frames == 0 {
            pixels
        } else {
            let last = pixels.narrow(2, frames - 1, 1)?;
            let repeated = last.repeat((1, 1, pad_frames, 1, 1))?;
            Tensor::cat(&[&pixels, &repeated], 2)?
        };
        let padded_frames = pixels.dim(2)?;
        let mut clips = Vec::with_capacity(padded_frames / clip_length);
        for start in (0..padded_frames).step_by(clip_length) {
            let clip = pixels.narrow(2, start, clip_length)?;
            clips.push(self.encode_clip_tiled(&clip)?);
        }
        let refs = clips.iter().collect::<Vec<_>>();
        let moments = Tensor::cat(&refs, 2)?;
        if self.config.token_drop == 0 {
            return Ok(moments);
        }
        let latent_frames = moments.dim(2)?;
        anyhow::ensure!(
            latent_frames > self.config.token_drop,
            "encoded sequence has {latent_frames} latent frames, cannot drop {}",
            self.config.token_drop
        );
        moments
            .narrow(2, 0, latent_frames - self.config.token_drop)
            .map_err(Into::into)
    }

    /// Applies the released DiagonalGaussian/sample/F16/normalization tail to moments.
    pub fn sample_and_normalize(&self, moments: &Tensor) -> Result<Tensor> {
        let (_, moment_channels, frames, height, width) = moments.dims5()?;
        let expected_channels = self
            .config
            .latent_channels
            .checked_mul(2)
            .context("visual posterior channel count overflow")?;
        anyhow::ensure!(
            moment_channels == expected_channels,
            "visual posterior has {moment_channels} channels, expected {expected_channels}"
        );
        anyhow::ensure!(
            frames > 0 && height > 0 && width > 0,
            "visual posterior dimensions must be non-zero"
        );

        anyhow::ensure!(
            moments.dtype() == DType::F32,
            "visual posterior moments must be F32"
        );
        let mean = moments.narrow(1, 0, self.config.latent_channels)?;
        let logvar = moments
            .narrow(1, self.config.latent_channels, self.config.latent_channels)?
            .clamp(-30f32, 20f32)?;
        let std = logvar.affine(0.5, 0.)?.exp()?;

        let mut rng = StdRng::seed_from_u64(CONDITION_ENCODE_SEED);
        let noise = (0..mean.elem_count())
            .map(|_| StandardNormal.sample(&mut rng))
            .collect::<Vec<f32>>();
        let noise = Tensor::from_vec(noise, mean.shape().clone(), mean.device())?;
        let sample = mean.add(&std.mul(&noise)?)?;
        let sample = sample
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;

        let latent_mean = Tensor::from_vec(
            self.config.latents_mean.clone(),
            (1, self.config.latent_channels, 1, 1, 1),
            &Device::Cpu,
        )?;
        let latent_std = Tensor::from_vec(
            self.config.latents_std.clone(),
            (1, self.config.latent_channels, 1, 1, 1),
            &Device::Cpu,
        )?;
        sample
            .broadcast_sub(&latent_mean)?
            .broadcast_div(&latent_std)
            .map_err(Into::into)
    }

    fn encode_clip_tiled(&self, pixels: &Tensor) -> Result<Tensor> {
        let height = pixels.dim(3)?;
        let width = pixels.dim(4)?;
        let ratio = self.config.spatial_ratio()?;
        let (ys, y_lengths, y_overlaps) =
            split_tiles(height, self.tile_size, self.tile_overlap, ratio)?;
        let (xs, x_lengths, x_overlaps) =
            split_tiles(width, self.tile_size, self.tile_overlap, ratio)?;

        let mut tiles = Vec::with_capacity(ys.len());
        for (&y, &tile_height) in ys.iter().zip(&y_lengths) {
            anyhow::ensure!(
                y.checked_add(tile_height).is_some_and(|end| end <= height),
                "vertical VAE tile exceeds the validated input canvas"
            );
            let mut row = Vec::with_capacity(xs.len());
            for (&x, &tile_width) in xs.iter().zip(&x_lengths) {
                anyhow::ensure!(
                    x.checked_add(tile_width).is_some_and(|end| end <= width),
                    "horizontal VAE tile exceeds the validated input canvas"
                );
                let tile = pixels
                    .narrow(3, y, tile_height)?
                    .narrow(4, x, tile_width)?
                    .to_device(&self.device)?;
                row.push(self.encode_tile(&tile)?.to_device(&Device::Cpu)?);
            }
            tiles.push(row);
        }

        let latent_y_overlaps = y_overlaps
            .iter()
            .map(|overlap| overlap / ratio)
            .collect::<Vec<_>>();
        let latent_x_overlaps = x_overlaps
            .iter()
            .map(|overlap| overlap / ratio)
            .collect::<Vec<_>>();
        stitch_tiles(tiles, &latent_y_overlaps, &latent_x_overlaps)
    }

    fn validate_spatial_input(&self, height: usize, width: usize) -> Result<()> {
        let ratio = self.config.spatial_ratio()?;
        let minimum = ratio
            .checked_mul(2)
            .context("visual VAE minimum spatial size overflow")?;
        anyhow::ensure!(
            height >= minimum
                && width >= minimum
                && height.is_multiple_of(ratio)
                && width.is_multiple_of(ratio),
            "visual VAE input must be at least {minimum}x{minimum} and divisible by its {ratio}x spatial compression ratio; got {height}x{width}"
        );
        Ok(())
    }

    fn encode_tile(&self, pixels: &Tensor) -> Result<Tensor> {
        let mut hidden = self.conv3d("encoder.conv_in", pixels, 1, 1, 2, 1)?;
        for block in 0..self.config.block_out_channels.len() {
            let in_channels = if block == 0 {
                self.config.block_out_channels[0]
            } else {
                self.config.block_out_channels[block - 1]
            };
            let out_channels = self.config.block_out_channels[block];
            for layer in 0..self.config.layers_per_block {
                hidden = self.resnet_block(hidden, block, layer, in_channels, out_channels)?;
            }
            let temporal_stride = self.config.temporal_downsample_factors[block];
            let spatial_stride = self.config.spatial_downsample_factors[block];
            if temporal_stride * spatial_stride > 1 {
                if spatial_stride == 2 {
                    hidden = reflect_pad_5d(&hidden, 0, 1, 0, 1)?;
                }
                hidden = self.conv3d(
                    &format!("encoder.down_blocks.{block}.downsamplers.0.conv"),
                    &hidden,
                    temporal_stride,
                    spatial_stride,
                    2,
                    0,
                )?;
            }
        }

        hidden = self.group_norm("encoder.norm_out", &hidden)?;
        hidden = ops::silu(&hidden)?;
        hidden = self.conv3d("encoder.conv_out", &hidden, 1, 1, 2, 1)?;
        self.conv3d("quant_conv", &hidden, 1, 1, 0, 0)
    }

    fn resnet_block(
        &self,
        hidden: Tensor,
        block: usize,
        layer: usize,
        block_in_channels: usize,
        out_channels: usize,
    ) -> Result<Tensor> {
        let prefix = format!("encoder.down_blocks.{block}.resnets.{layer}");
        let input_channels = if layer == 0 {
            block_in_channels
        } else {
            out_channels
        };
        anyhow::ensure!(
            hidden.dim(1)? == input_channels,
            "{prefix} received {} channels, expected {input_channels}",
            hidden.dim(1)?
        );

        let residual = hidden.clone();
        let mut output = self.group_norm(&format!("{prefix}.norm1"), &hidden)?;
        output = ops::silu(&output)?;
        output = self.conv3d(&format!("{prefix}.conv1"), &output, 1, 1, 2, 1)?;
        output = self.group_norm(&format!("{prefix}.norm2"), &output)?;
        output = ops::silu(&output)?;
        output = self.conv3d(&format!("{prefix}.conv2"), &output, 1, 1, 2, 1)?;

        let residual = if input_channels == out_channels {
            residual
        } else {
            self.conv3d(&format!("{prefix}.conv_shortcut"), &residual, 1, 1, 0, 0)?
        };
        residual.add(&output).map_err(Into::into)
    }

    fn group_norm(&self, prefix: &str, input: &Tensor) -> Result<Tensor> {
        let names = [format!("{prefix}.weight"), format!("{prefix}.bias")];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            isolated_group_norm(
                input,
                required(weights, &names[0])?,
                required(weights, &names[1])?,
                self.config.norm_num_groups,
                self.config.norm_eps,
            )
            .with_context(|| prefix.to_owned())
        })
    }

    fn conv3d(
        &self,
        prefix: &str,
        input: &Tensor,
        temporal_stride: usize,
        spatial_stride: usize,
        temporal_padding: usize,
        spatial_padding: usize,
    ) -> Result<Tensor> {
        let names = [format!("{prefix}.weight"), format!("{prefix}.bias")];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            let weight = required(weights, &names[0])?;
            let bias = required(weights, &names[1])?;
            causal_conv3d(
                input,
                weight,
                bias,
                temporal_stride,
                spatial_stride,
                temporal_padding,
                spatial_padding,
            )
            .with_context(|| prefix.to_owned())
        })
    }
}

fn validate_weight_inventory(weights: &ModelWeights, config: &VideoVaeEncoderConfig) -> Result<()> {
    let first_channels = config.block_out_channels[0];
    let last_channels = *config
        .block_out_channels
        .last()
        .context("visual VAE encoder has no channel blocks")?;
    let moment_channels = config
        .latent_channels
        .checked_mul(2)
        .context("visual VAE posterior channel count overflow")?;
    let mut expected = vec![
        (
            "encoder.conv_in.weight".to_owned(),
            vec![first_channels, config.in_channels, 3, 3, 3],
        ),
        ("encoder.conv_in.bias".to_owned(), vec![first_channels]),
        ("encoder.norm_out.weight".to_owned(), vec![last_channels]),
        ("encoder.norm_out.bias".to_owned(), vec![last_channels]),
        (
            "encoder.conv_out.weight".to_owned(),
            vec![moment_channels, last_channels, 3, 3, 3],
        ),
        ("encoder.conv_out.bias".to_owned(), vec![moment_channels]),
        (
            "quant_conv.weight".to_owned(),
            vec![moment_channels, moment_channels, 1, 1, 1],
        ),
        ("quant_conv.bias".to_owned(), vec![moment_channels]),
    ];
    for block in 0..config.block_out_channels.len() {
        let block_input_channels = if block == 0 {
            config.block_out_channels[0]
        } else {
            config.block_out_channels[block - 1]
        };
        let output_channels = config.block_out_channels[block];
        for layer in 0..config.layers_per_block {
            let prefix = format!("encoder.down_blocks.{block}.resnets.{layer}");
            let input_channels = if layer == 0 {
                block_input_channels
            } else {
                output_channels
            };
            for suffix in ["norm1.weight", "norm1.bias"] {
                expected.push((format!("{prefix}.{suffix}"), vec![input_channels]));
            }
            expected.push((
                format!("{prefix}.conv1.weight"),
                vec![output_channels, input_channels, 3, 3, 3],
            ));
            expected.push((format!("{prefix}.conv1.bias"), vec![output_channels]));
            for suffix in ["norm2.weight", "norm2.bias"] {
                expected.push((format!("{prefix}.{suffix}"), vec![output_channels]));
            }
            expected.push((
                format!("{prefix}.conv2.weight"),
                vec![output_channels, output_channels, 3, 3, 3],
            ));
            expected.push((format!("{prefix}.conv2.bias"), vec![output_channels]));
            if layer == 0 && input_channels != output_channels {
                expected.push((
                    format!("{prefix}.conv_shortcut.weight"),
                    vec![output_channels, input_channels, 1, 1, 1],
                ));
                expected.push((
                    format!("{prefix}.conv_shortcut.bias"),
                    vec![output_channels],
                ));
            }
        }
        if config.spatial_downsample_factors[block] * config.temporal_downsample_factors[block] > 1
        {
            let prefix = format!("encoder.down_blocks.{block}.downsamplers.0.conv");
            expected.push((
                format!("{prefix}.weight"),
                vec![output_channels, output_channels, 3, 3, 3],
            ));
            expected.push((format!("{prefix}.bias"), vec![output_channels]));
        }
    }
    for (name, shape) in expected {
        let metadata = weights
            .metadata(&name)
            .with_context(|| format!("official root VAE is missing encoder tensor {name}"))?;
        anyhow::ensure!(
            metadata.dtype == "F32",
            "visual VAE encoder tensor {name} has dtype {}, expected F32",
            metadata.dtype
        );
        anyhow::ensure!(
            metadata.shape == shape,
            "visual VAE encoder tensor {name} has shape {:?}, expected {shape:?}",
            metadata.shape
        );
    }
    Ok(())
}

/// Cross-correlation equivalent to the released causal Conv3d.
///
/// Candle has no general Conv3d primitive, so each temporal kernel plane is evaluated as one
/// batched Conv2d. Temporal frames are gathered at the requested stride, summed, and biased once.
fn causal_conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    temporal_stride: usize,
    spatial_stride: usize,
    temporal_padding: usize,
    spatial_padding: usize,
) -> Result<Tensor> {
    anyhow::ensure!(
        temporal_stride > 0 && spatial_stride > 0,
        "Conv3d strides must be non-zero"
    );
    let (batch, in_channels, _, _, _) = input.dims5()?;
    let (out_channels, weight_in_channels, kernel_t, kernel_h, kernel_w) = weight.dims5()?;
    anyhow::ensure!(
        in_channels == weight_in_channels,
        "Conv3d input has {in_channels} channels but weight expects {weight_in_channels}"
    );
    anyhow::ensure!(
        kernel_h == kernel_w && kernel_h > 0 && kernel_t > 0,
        "Conv3d kernels must be non-empty and spatially square"
    );
    anyhow::ensure!(
        bias.dims1()? == out_channels,
        "Conv3d bias width does not match output channels"
    );
    anyhow::ensure!(
        input.dtype() == DType::F32 && weight.dtype() == DType::F32 && bias.dtype() == DType::F32,
        "streamed VAE Conv3d requires F32 input and parameters"
    );

    let input = if spatial_padding == 0 {
        input.clone()
    } else {
        reflect_pad_5d(
            input,
            spatial_padding,
            spatial_padding,
            spatial_padding,
            spatial_padding,
        )?
    };
    let input = input.pad_with_zeros(2, temporal_padding, 0)?.contiguous()?;
    let padded_frames = input.dim(2)?;
    anyhow::ensure!(
        padded_frames >= kernel_t,
        "causal Conv3d kernel is longer than its padded input"
    );
    let output_frames = (padded_frames - kernel_t) / temporal_stride + 1;
    let config = Conv2dConfig {
        stride: spatial_stride,
        ..Default::default()
    };
    let mut output: Option<Tensor> = None;
    for temporal_kernel in 0..kernel_t {
        let indices = (0..output_frames)
            .map(|frame| {
                u32::try_from(temporal_kernel + frame * temporal_stride)
                    .context("Conv3d temporal index exceeds u32")
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = Tensor::from_vec(indices, output_frames, input.device())?;
        let frames = input
            .index_select(&indices, 2)?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        let (_, _, _, height, width) = frames.dims5()?;
        let frames = frames.reshape((batch * output_frames, in_channels, height, width))?;
        let kernel = weight
            .narrow(2, temporal_kernel, 1)?
            .squeeze(2)?
            .contiguous()?;
        let convolved = Conv2d::new(kernel, None, config).forward(&frames)?;
        let (_, _, output_height, output_width) = convolved.dims4()?;
        let convolved = convolved
            .reshape((
                batch,
                output_frames,
                out_channels,
                output_height,
                output_width,
            ))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?;
        output = Some(match output {
            Some(accumulated) => accumulated.add(&convolved)?,
            None => convolved,
        });
    }
    let output = output.context("Conv3d has no temporal kernel planes")?;
    output
        .broadcast_add(&bias.reshape((1, out_channels, 1, 1, 1))?)
        .map_err(Into::into)
}

/// GroupNorm with time folded into batch, exactly matching `MiniMaxH3VideoGroupNorm`.
fn isolated_group_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    groups: usize,
    eps: f64,
) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = input.dims5()?;
    anyhow::ensure!(groups > 0, "GroupNorm group count must be non-zero");
    anyhow::ensure!(
        channels.is_multiple_of(groups),
        "GroupNorm channels must be divisible by groups"
    );
    anyhow::ensure!(
        weight.dims1()? == channels && bias.dims1()? == channels,
        "GroupNorm affine parameter width does not match input channels"
    );
    anyhow::ensure!(
        input.dtype() == DType::F32 && weight.dtype() == DType::F32 && bias.dtype() == DType::F32,
        "streamed VAE GroupNorm requires F32 input and parameters"
    );
    let group_width = channels / groups * height * width;
    let per_frame = input.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((
        batch * frames,
        groups,
        group_width,
    ))?;
    let mean = per_frame.mean_keepdim(2)?;
    let centered = per_frame.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(2)?;
    let normalized = centered.broadcast_div(&(&variance + eps)?.sqrt()?)?;
    normalized
        .reshape((batch, frames, channels, height, width))?
        .broadcast_mul(&weight.reshape((1, 1, channels, 1, 1))?)?
        .broadcast_add(&bias.reshape((1, 1, channels, 1, 1))?)?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()
        .map_err(Into::into)
}

fn reflect_pad_5d(
    input: &Tensor,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<Tensor> {
    let input = reflect_pad_dim(input, 4, left, right)?;
    reflect_pad_dim(&input, 3, top, bottom)
}

fn reflect_pad_dim(input: &Tensor, dim: usize, before: usize, after: usize) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(input.clone());
    }
    let length = input.dim(dim)?;
    anyhow::ensure!(
        before < length && after < length,
        "reflection padding ({before}, {after}) must be smaller than dimension {length}"
    );
    let before_values = if before == 0 {
        None
    } else {
        Some(input.narrow(dim, 1, before)?.contiguous()?.flip(&[dim])?)
    };
    let after_values = if after == 0 {
        None
    } else {
        Some(
            input
                .narrow(dim, length - after - 1, after)?
                .contiguous()?
                .flip(&[dim])?,
        )
    };
    match (&before_values, &after_values) {
        (Some(before), Some(after)) => Tensor::cat(&[before, input, after], dim),
        (Some(before), None) => Tensor::cat(&[before, input], dim),
        (None, Some(after)) => Tensor::cat(&[input, after], dim),
        (None, None) => Ok(input.clone()),
    }
    .map_err(Into::into)
}

/// Tile stitching follows the pinned diffusers implementation, including use of the unmodified
/// upper/left neighbour for each blend. That detail affects the doubly-overlapped corners.
fn stitch_tiles(
    tiles: Vec<Vec<Tensor>>,
    height_overlaps: &[usize],
    width_overlaps: &[usize],
) -> Result<Tensor> {
    let rows = tiles.len();
    let columns = tiles.first().context("no encoded VAE tiles")?.len();
    anyhow::ensure!(columns > 0, "encoded VAE tile rows must not be empty");
    anyhow::ensure!(
        tiles.iter().all(|row| row.len() == columns),
        "encoded VAE tile grid must be rectangular"
    );
    anyhow::ensure!(
        height_overlaps.len() == rows - 1 && width_overlaps.len() == columns - 1,
        "encoded VAE overlap counts do not match the tile grid"
    );

    let mut result_rows = Vec::with_capacity(rows);
    for row_index in 0..rows {
        let mut result_row = Vec::with_capacity(columns);
        for column_index in 0..columns {
            let mut tile = tiles[row_index][column_index].clone();
            if row_index > 0 {
                tile = blend(
                    &tiles[row_index - 1][column_index],
                    &tile,
                    height_overlaps[row_index - 1],
                    3,
                )?;
            }
            if column_index > 0 {
                tile = blend(
                    &tiles[row_index][column_index - 1],
                    &tile,
                    width_overlaps[column_index - 1],
                    4,
                )?;
            }
            if row_index + 1 < rows {
                let keep = tile
                    .dim(3)?
                    .checked_sub(height_overlaps[row_index])
                    .context("vertical latent overlap exceeds encoded tile")?;
                tile = tile.narrow(3, 0, keep)?;
            }
            if column_index + 1 < columns {
                let keep = tile
                    .dim(4)?
                    .checked_sub(width_overlaps[column_index])
                    .context("horizontal latent overlap exceeds encoded tile")?;
                tile = tile.narrow(4, 0, keep)?;
            }
            result_row.push(tile);
        }
        let refs = result_row.iter().collect::<Vec<_>>();
        result_rows.push(Tensor::cat(&refs, 4)?);
    }
    let refs = result_rows.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 3).map_err(Into::into)
}

fn blend(a: &Tensor, b: &Tensor, requested_extent: usize, dim: usize) -> Result<Tensor> {
    if requested_extent == 0 {
        return Ok(b.clone());
    }
    anyhow::ensure!(
        requested_extent <= a.dim(dim)? && requested_extent <= b.dim(dim)?,
        "requested tile overlap {requested_extent} exceeds a source tile along dimension {dim}"
    );
    let extent = requested_extent;
    let mut shape = vec![1; b.rank()];
    shape[dim] = extent;
    let weight_b = Tensor::arange(0f32, extent as f32, b.device())?
        .affine(1. / extent as f64, 0.)?
        .reshape(Shape::from_dims(&shape))?;
    let weight_a = weight_b.affine(-1., 1.)?;
    let a_tail = a.narrow(dim, a.dim(dim)? - extent, extent)?;
    let b_head = b.narrow(dim, 0, extent)?;
    let blended = a_tail
        .broadcast_mul(&weight_a)?
        .add(&b_head.broadcast_mul(&weight_b)?)?;
    if extent == b.dim(dim)? {
        Ok(blended)
    } else {
        let rest = b.narrow(dim, extent, b.dim(dim)? - extent)?;
        Tensor::cat(&[&blended, &rest], dim).map_err(Into::into)
    }
}

fn imagenet_normalize(pixels: &Tensor, channels: usize) -> Result<Tensor> {
    anyhow::ensure!(
        channels == PIXEL_MEAN.len(),
        "released MiniMax-H3 pixel normalization requires three RGB channels"
    );
    let pixels = pixels.to_dtype(DType::F32)?.affine(1. / 255., 0.)?;
    let mean = Tensor::from_vec(PIXEL_MEAN.to_vec(), (1, channels, 1, 1, 1), pixels.device())?;
    let std = Tensor::from_vec(PIXEL_STD.to_vec(), (1, channels, 1, 1, 1), pixels.device())?;
    pixels
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
        .map_err(Into::into)
}

fn read_png_rgb8(path: &Path) -> Result<Tensor> {
    let file = File::open(path)
        .with_context(|| format!("failed to open PNG keyframe {}", path.display()))?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .with_context(|| format!("failed to read PNG header {}", path.display()))?;
    let header = reader.info();
    anyhow::ensure!(
        header.animation_control.is_none()
            && header.frame_control.is_none()
            && header.color_type == png::ColorType::Rgb
            && header.bit_depth == png::BitDepth::Eight,
        "visual VAE keyframe PNG must be one non-animated RGB8 image"
    );
    let png_width = header.width;
    let png_height = header.height;
    let output_size = reader
        .output_buffer_size()
        .context("PNG output buffer size overflow")?;
    let mut bytes = vec![0u8; output_size];
    let info = reader
        .next_frame(&mut bytes)
        .with_context(|| format!("failed to decode PNG keyframe {}", path.display()))?;
    anyhow::ensure!(
        info.width == png_width
            && info.height == png_height
            && info.color_type == png::ColorType::Rgb
            && info.bit_depth == png::BitDepth::Eight,
        "decoded visual VAE keyframe does not match its RGB8 PNG header"
    );
    let source = &bytes[..info.buffer_size()];
    let pixels = source.to_vec();
    let height = usize::try_from(info.height).context("PNG height exceeds usize")?;
    let width = usize::try_from(info.width).context("PNG width exceeds usize")?;
    let expected = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("decoded PNG RGB size overflow")?;
    anyhow::ensure!(
        pixels.len() == expected,
        "decoded PNG RGB payload has {} bytes, expected {expected}",
        pixels.len()
    );
    reader
        .finish()
        .with_context(|| format!("failed to finish PNG keyframe {}", path.display()))?;
    Tensor::from_vec(pixels, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .unsqueeze(2)
        .map_err(Into::into)
}

fn checked_product(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        anyhow::ensure!(*value > 0, "{name} contains zero");
        product
            .checked_mul(*value)
            .with_context(|| format!("{name} overflow"))
    })
}

fn required<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    weights
        .get(name)
        .with_context(|| format!("missing visual VAE encoder tensor {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors;
    use serde_json::json;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn tensor(values: Vec<f32>, shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    fn full(value: f32, shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        tensor(vec![value; shape.elem_count()], shape)
    }

    fn ones(shape: impl Into<Shape>) -> Tensor {
        full(1., shape)
    }

    fn zeros(shape: impl Into<Shape>) -> Tensor {
        full(0., shape)
    }

    fn assert_close(actual: &Tensor, expected: &[f32], tolerance: f32) {
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn causal_conv3d_never_reads_a_future_frame() {
        let input = tensor(vec![1., 2., 3.], (1, 1, 3, 1, 1));
        let weight = tensor(vec![1., 10., 100.], (1, 1, 3, 1, 1));
        let bias = tensor(vec![0.5], 1);
        let output = causal_conv3d(&input, &weight, &bias, 1, 1, 2, 0).unwrap();
        assert_eq!(output.dims(), &[1, 1, 3, 1, 1]);
        assert_close(&output, &[100.5, 210.5, 321.5], 1e-6);

        let changed_future = tensor(vec![1., 2., 30_000.], (1, 1, 3, 1, 1));
        let changed = causal_conv3d(&changed_future, &weight, &bias, 1, 1, 2, 0).unwrap();
        assert_close(&changed.narrow(2, 0, 2).unwrap(), &[100.5, 210.5], 1e-6);
    }

    #[test]
    fn group_norm_statistics_are_isolated_per_frame() {
        let input = tensor(vec![1., 100., 3., 104.], (1, 2, 2, 1, 1));
        let output = isolated_group_norm(&input, &ones(2), &zeros(2), 1, 1e-6).unwrap();
        assert_close(
            &output,
            &[-0.999_999_5, -0.999_999_9, 0.999_999_5, 0.999_999_9],
            1e-5,
        );
    }

    #[test]
    fn reflection_padding_matches_torch_order() {
        let input = tensor(vec![1., 2., 3., 4.], (1, 1, 1, 1, 4));
        let padded = reflect_pad_dim(&input, 4, 2, 1).unwrap();
        assert_close(&padded, &[3., 2., 1., 2., 3., 4., 3.], 0.);
    }

    #[test]
    fn tiny_encoder_has_official_temporal_and_spatial_shapes() {
        let directory = TempDir::new().unwrap();
        let encoder = open_tiny_encoder(directory.path());
        let values = (0..3 * 5 * 8 * 8)
            .map(|index| (index % 256) as f32)
            .collect::<Vec<_>>();
        let pixels = tensor(values, (1, 3, 5, 8, 8));
        let latents = encoder.encode_video_pixels(&pixels).unwrap();
        assert_eq!(latents.dims(), &[1, 1, 3, 2, 2]);
        assert_eq!(latents.dtype(), DType::F32);
        assert!(latents.device().is_cpu());

        let png_path = directory.path().join("keyframe.png");
        let file = File::create(&png_path).unwrap();
        let mut png_encoder = png::Encoder::new(file, 8, 8);
        png_encoder.set_color(png::ColorType::Rgb);
        png_encoder.set_depth(png::BitDepth::Eight);
        let mut writer = png_encoder.write_header().unwrap();
        writer.write_image_data(&[127u8; 8 * 8 * 3]).unwrap();
        writer.finish().unwrap();
        let keyframe = encoder.encode_png_keyframe(&png_path).unwrap();
        assert_eq!(keyframe.dims(), &[1, 1, 1, 2, 2]);

        let rgba_path = directory.path().join("rgba.png");
        let file = File::create(&rgba_path).unwrap();
        let mut png_encoder = png::Encoder::new(file, 8, 8);
        png_encoder.set_color(png::ColorType::Rgba);
        png_encoder.set_depth(png::BitDepth::Eight);
        let mut writer = png_encoder.write_header().unwrap();
        writer.write_image_data(&[127u8; 8 * 8 * 4]).unwrap();
        writer.finish().unwrap();
        let result = read_png_rgb8(&rgba_path);
        let Err(error) = result else {
            panic!("RGBA must not be silently converted to RGB");
        };
        assert!(error.to_string().contains("RGB8"));

        let moments = zeros((1, 2, 1, 1, 2));
        let first = encoder.sample_and_normalize(&moments).unwrap();
        let second = encoder.sample_and_normalize(&moments).unwrap();
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn invalid_config_and_spatial_geometry_fail_before_weight_materialization() {
        let directory = TempDir::new().unwrap();
        let encoder = open_tiny_encoder(directory.path());

        let mut non_rgb = encoder.config.clone();
        non_rgb.in_channels = 4;
        assert!(non_rgb.validate().is_err());
        let mut impossible_drop = encoder.config.clone();
        impossible_drop.token_drop = 2;
        assert!(impossible_drop.validate().is_err());

        let before = encoder.weights.access_stats();
        let unaligned = zeros((1, 3, 1, 8, 7));
        let error = encoder.encode_video_pixels(&unaligned).unwrap_err();
        assert!(error.to_string().contains("divisible"));
        assert_eq!(encoder.weights.access_stats(), before);
        let half_normalized = Tensor::zeros((1, 3, 1, 8, 8), DType::F16, &Device::Cpu).unwrap();
        let result = encoder.encode_moments(&half_normalized);
        let Err(error) = result else {
            panic!("half-precision normalized input must be rejected");
        };
        assert!(error.to_string().contains("must be F32"));
        assert_eq!(encoder.weights.access_stats(), before);

        let mut wrong_shape_contract = encoder.config.clone();
        wrong_shape_contract.latent_channels = 2;
        wrong_shape_contract.latents_mean = vec![0.; 2];
        wrong_shape_contract.latents_std = vec![1.; 2];
        let error = validate_weight_inventory(&encoder.weights, &wrong_shape_contract).unwrap_err();
        assert!(error.to_string().contains("shape"));
    }

    fn open_tiny_encoder(directory: &Path) -> StreamedVideoVaeEncoder {
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "AutoencoderKLMiniMaxH3",
                "in_channels": 3,
                "latent_channels": 1,
                "block_out_channels": [2, 2],
                "layers_per_block": 1,
                "spatial_downsample_factors": [2, 2],
                "temporal_downsample_factors": [1, 2],
                "norm_num_groups": 1,
                "norm_eps": 1e-6,
                "spatial_padding_mode": "reflect",
                "clip_length": 3,
                "token_drop": 1,
                "latents_mean": [0.0],
                "latents_std": [1.0]
            }))
            .unwrap(),
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "encoder.conv_in.weight".to_owned(),
            full(0.01, (2, 3, 3, 3, 3)),
        );
        weights.insert("encoder.conv_in.bias".to_owned(), zeros(2));
        for block in 0..2 {
            let prefix = format!("encoder.down_blocks.{block}.resnets.0");
            for norm in ["norm1", "norm2"] {
                weights.insert(format!("{prefix}.{norm}.weight"), ones(2));
                weights.insert(format!("{prefix}.{norm}.bias"), zeros(2));
            }
            for conv in ["conv1", "conv2"] {
                weights.insert(
                    format!("{prefix}.{conv}.weight"),
                    full(0.01, (2, 2, 3, 3, 3)),
                );
                weights.insert(format!("{prefix}.{conv}.bias"), zeros(2));
            }
            let downsample = format!("encoder.down_blocks.{block}.downsamplers.0.conv");
            weights.insert(format!("{downsample}.weight"), full(0.01, (2, 2, 3, 3, 3)));
            weights.insert(format!("{downsample}.bias"), zeros(2));
        }
        weights.insert("encoder.norm_out.weight".to_owned(), ones(2));
        weights.insert("encoder.norm_out.bias".to_owned(), zeros(2));
        weights.insert(
            "encoder.conv_out.weight".to_owned(),
            full(0.01, (2, 2, 3, 3, 3)),
        );
        weights.insert("encoder.conv_out.bias".to_owned(), zeros(2));
        weights.insert("quant_conv.weight".to_owned(), full(0.01, (2, 2, 1, 1, 1)));
        weights.insert("quant_conv.bias".to_owned(), zeros(2));
        safetensors::save(
            &weights,
            directory.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        StreamedVideoVaeEncoder::open(
            directory,
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
        )
        .unwrap()
    }
}
