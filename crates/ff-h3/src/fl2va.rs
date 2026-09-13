//! End-to-end preparation of MiniMax-H3 first/last-frame conditioning.
//!
//! It resolves and prepares the canvas, runs the multimodal Qwen conditioner, encodes keyframes
//! with the visual VAE, applies request-RNG conditioning noise, patchifies the frozen rows, and
//! builds [`ConditionedLayout`]. Target noise is drawn from the same request RNG in the released
//! order: condition draws first, then target video, then target audio. The prepared value can run
//! the shared conditioned denoiser; video/audio decoding remains a separate output concern.

use crate::{
    config::TransformerConfig,
    h3_conditioning::{
        CONDITION_VIDEO_TIMESTEP, ConditionedLayout, KeyframeAnchor,
        denoise_conditioned_with_observer,
    },
    layout,
    model::StreamedTransformer,
    multimodal_text_encoder::{RgbImage, StreamedMultimodalTextEncoder, resolve_fl2va_canvas_size},
    pipeline::{DenoiseObserver, T2vaExecutionOptions, T2vaLatents, T2vaSchedule},
    scheduler::H3Scheduler,
    text_encoder::PromptEncoding,
    video_vae_encoder::StreamedVideoVaeEncoder,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

pub const MINIMAX_H3_FPS: usize = 24;
pub const MINIMAX_H3_AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const MINIMAX_H3_AUDIO_CHANNELS: usize = 2;
pub const MINIMAX_H3_AUDIO_LATENT_CHANNELS: usize = 32;
/// The released output-duration range, from MiniMax-H3's own specification:
/// "Output duration | 4-15 seconds".
///
/// The bound applies to the *requested* duration. Frame alignment then rounds
/// up to the next representable count, which is what the released assets show:
/// a 10-second request is published as 243 frames (10.125 s), an 8-second one
/// as 192 (8.000 s), a 5-second one as 124 (5.167 s).
pub const MINIMAX_H3_MIN_SECONDS: usize = 4;
pub const MINIMAX_H3_MAX_SECONDS: usize = 15;

/// The aligned frame counts the released duration range can produce.
///
/// Alignment rounds a request up to the next `17 * n + 5`, so a stored bundle
/// carries one of these rather than the requested count.
/// `released_duration_range_matches_the_published_assets` keeps them equal to
/// what the range actually aligns to.
pub const MINIMAX_H3_MIN_ALIGNED_FRAMES: usize = 107;
pub const MINIMAX_H3_MAX_ALIGNED_FRAMES: usize = 362;
pub const MINIMAX_H3_PATCH_SIZE: [usize; 3] = [1, 2, 2];
pub const MINIMAX_H3_MAX_PROMPT_CHARS: usize = 7_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fl2vaCanvas {
    pub height: usize,
    pub width: usize,
}

impl Fl2vaCanvas {
    pub fn new(height: usize, width: usize) -> Result<Self> {
        anyhow::ensure!(
            height > 0 && width > 0,
            "FL2VA canvas dimensions must be non-zero"
        );
        Ok(Self { height, width })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fl2vaOptions {
    /// Requested pixel-frame count. It is rounded up to the next VAE-aligned count.
    pub requested_num_frames: usize,
    /// An explicit generated canvas, or `None` to derive it from the first keyframe.
    pub canvas: Option<Fl2vaCanvas>,
    pub patch_size: [usize; 3],
    pub audio_channels: usize,
    /// Width of an empty FL2VA audio-conditioning row.
    pub audio_latent_channels: usize,
}

impl Fl2vaOptions {
    pub const fn official(requested_num_frames: usize) -> Self {
        Self {
            requested_num_frames,
            canvas: None,
            patch_size: MINIMAX_H3_PATCH_SIZE,
            audio_channels: MINIMAX_H3_AUDIO_CHANNELS,
            audio_latent_channels: MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        }
    }

    pub const fn with_canvas(mut self, canvas: Fl2vaCanvas) -> Self {
        self.canvas = Some(canvas);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fl2vaFrameGeometry {
    pub num_frames: usize,
    pub num_latent_frames: usize,
    pub num_audio_latents: usize,
}

pub struct PreparedFl2vaKeyframes {
    pub canvas: Fl2vaCanvas,
    pub images: Vec<RgbImage>,
    pub anchors: Vec<KeyframeAnchor>,
}

/// Everything the conditioned denoising boundary needs from FL2VA preparation.
pub struct PreparedFl2va {
    pub canvas: Fl2vaCanvas,
    /// VAE-aligned pixel-frame count, which may be greater than the requested count.
    pub num_frames: usize,
    pub num_latent_frames: usize,
    pub num_audio_latents: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub keyframes: Vec<RgbImage>,
    pub anchors: Vec<KeyframeAnchor>,
    pub prompt: PromptEncoding,
    /// Sampled, F16-rounded, normalized keyframe latents on CPU, one tensor per anchor.
    pub condition_latents: Vec<Tensor>,
    /// Noised and patchified frozen video rows on the same device as `prompt.embeddings`.
    pub condition_video_rows: Tensor,
    /// Empty `[0, audio_latent_channels]` tensor on the same device as `prompt.embeddings`.
    pub condition_audio_rows: Tensor,
    /// Initial generated-video noise, drawn after every condition from the request RNG.
    pub initial_target_video_latents: Tensor,
    /// Initial generated-audio noise, drawn after target-video noise from the request RNG.
    pub initial_target_audio_latents: Tensor,
    pub layout: ConditionedLayout,
    pub patch_size: [usize; 3],
    pub audio_channels: usize,
}

impl PreparedFl2va {
    /// Validate the prepared VAE/Qwen tensors against a concrete denoiser before any layer runs.
    pub fn validate_transformer(&self, transformer: &StreamedTransformer) -> Result<()> {
        self.validate_transformer_contract(transformer.config(), transformer.device())
    }

    fn validate_transformer_contract(
        &self,
        config: &TransformerConfig,
        device: &Device,
    ) -> Result<()> {
        anyhow::ensure!(
            config.patch_size == self.patch_size,
            "FL2VA was prepared for patch {:?}, transformer requires {:?}",
            self.patch_size,
            config.patch_size
        );
        let (prompt_batch, _, prompt_width) = self.prompt.embeddings.dims3()?;
        anyhow::ensure!(
            prompt_batch == 1 && prompt_width == config.text_dim,
            "FL2VA Qwen output must be [1, rows, {}], got {:?}",
            config.text_dim,
            self.prompt.embeddings.dims()
        );
        let (video_batch, video_channels, video_frames, video_height, video_width) =
            self.initial_target_video_latents.dims5()?;
        anyhow::ensure!(
            (
                video_batch,
                video_channels,
                video_frames,
                video_height,
                video_width
            ) == (
                1,
                config.in_channels,
                self.num_latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            "FL2VA target video shape {:?} disagrees with its prepared geometry and transformer",
            self.initial_target_video_latents.dims()
        );
        let (audio_channels, audio_latent_channels, audio_frames) =
            self.initial_target_audio_latents.dims3()?;
        anyhow::ensure!(
            audio_channels == self.audio_channels
                && audio_latent_channels == config.audio_in_channels
                && audio_frames == self.num_audio_latents,
            "FL2VA target audio shape {:?} does not match {} channels and transformer width {}",
            self.initial_target_audio_latents.dims(),
            self.audio_channels,
            config.audio_in_channels
        );
        anyhow::ensure!(
            self.keyframes.len() == self.anchors.len()
                && self.condition_latents.len() == self.anchors.len()
                && (1..=2).contains(&self.anchors.len())
                && self.keyframes.iter().all(|image| {
                    image.height() == self.canvas.height && image.width() == self.canvas.width
                })
                && self.condition_latents.iter().all(|latent| {
                    latent.dims()
                        == [
                            1,
                            config.in_channels,
                            1,
                            self.latent_height,
                            self.latent_width,
                        ]
                }),
            "FL2VA keyframes, anchors, and condition latents disagree with the prepared canvas"
        );
        let patch_volume = self.patch_size.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .context("FL2VA patch volume overflow")
        })?;
        let video_row_width = config
            .in_channels
            .checked_mul(patch_volume)
            .context("FL2VA video row width overflow")?;
        anyhow::ensure!(
            self.condition_video_rows.dims()
                == [self.layout.condition_video_rows(), video_row_width],
            "FL2VA condition video rows do not match transformer input width"
        );
        anyhow::ensure!(
            self.condition_audio_rows.dims() == [0, config.audio_in_channels],
            "FL2VA condition audio rows do not match transformer input width"
        );
        anyhow::ensure!(
            [
                &self.condition_video_rows,
                &self.condition_audio_rows,
                &self.initial_target_video_latents,
                &self.initial_target_audio_latents,
            ]
            .iter()
            .all(|tensor| tensor.dtype() == DType::F32),
            "FL2VA condition rows and target latents must be F32"
        );
        for (name, tensor) in [
            ("Qwen embeddings", &self.prompt.embeddings),
            ("condition video rows", &self.condition_video_rows),
            ("condition audio rows", &self.condition_audio_rows),
            ("target video noise", &self.initial_target_video_latents),
            ("target audio noise", &self.initial_target_audio_latents),
        ] {
            anyhow::ensure!(
                tensor.device().same_device(device),
                "{name} and the FL2VA transformer are on different devices"
            );
        }
        Ok(())
    }

    /// Run the shared conditioned H3 denoiser over frozen keyframe rows and target noise.
    pub fn denoise(
        &self,
        transformer: &StreamedTransformer,
        schedule: T2vaSchedule,
        options: T2vaExecutionOptions,
        observer: &mut dyn DenoiseObserver,
    ) -> Result<T2vaLatents> {
        self.validate_transformer(transformer)?;
        denoise_conditioned_with_observer(
            transformer,
            &self.prompt.embeddings,
            &self.prompt.text_token_tags,
            &self.condition_video_rows,
            &self.condition_audio_rows,
            &self.initial_target_video_latents,
            &self.initial_target_audio_latents,
            &self.layout,
            schedule,
            options,
            observer,
        )
    }
}

/// Prepare a real MiniMax-H3 FL2VA request up to the denoising boundary.
///
/// `request_rng` is intentionally borrowed rather than constructed here. The function consumes
/// keyframe-condition draws in packed order, then target-video noise, then target-audio noise.
/// VAE posterior sampling remains independently fixed at seed 42 inside
/// [`StreamedVideoVaeEncoder`].
#[allow(clippy::too_many_arguments)]
pub fn prepare_fl2va(
    text_encoder: &StreamedMultimodalTextEncoder,
    video_vae: &StreamedVideoVaeEncoder,
    tokenizer: &Tokenizer,
    prompt: &str,
    first_image: Option<&RgbImage>,
    last_image: Option<&RgbImage>,
    options: Fl2vaOptions,
    request_rng: &mut StdRng,
) -> Result<PreparedFl2va> {
    prepare_fl2va_with(
        text_encoder,
        video_vae,
        tokenizer,
        prompt,
        first_image,
        last_image,
        options,
        request_rng,
    )
}

/// Resolve the canvas and apply the released keyframe placement rules.
///
/// The first frame is the geometry anchor and is stretched to the canvas with Lanczos resampling.
/// The optional last frame is resized to cover the canvas, then centre-cropped with the exact
/// `round(source * scale)` and `(resized - canvas) / 2` integer arithmetic used by diffusers.
pub fn prepare_fl2va_keyframes(
    first_image: &RgbImage,
    last_image: Option<&RgbImage>,
    explicit_canvas: Option<Fl2vaCanvas>,
) -> Result<PreparedFl2vaKeyframes> {
    prepare_optional_fl2va_keyframes(Some(first_image), last_image, explicit_canvas)
}

/// General keyframe placement API supporting first-only, last-only, and first+last FL2VA.
///
/// When only `last_image` is present it is both the geometry anchor and the first packed image, so
/// it is stretched to the resolved canvas while retaining [`KeyframeAnchor::Last`].
pub fn prepare_optional_fl2va_keyframes(
    first_image: Option<&RgbImage>,
    last_image: Option<&RgbImage>,
    explicit_canvas: Option<Fl2vaCanvas>,
) -> Result<PreparedFl2vaKeyframes> {
    anyhow::ensure!(
        first_image.is_some() || last_image.is_some(),
        "FL2VA requires a first image, a last image, or both"
    );
    let ordered = [
        first_image.map(|image| (KeyframeAnchor::First, image)),
        last_image.map(|image| (KeyframeAnchor::Last, image)),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let geometry_anchor = ordered[0].1;
    let canvas = match explicit_canvas {
        Some(canvas) => Fl2vaCanvas::new(canvas.height, canvas.width)?,
        None => {
            let (height, width) =
                resolve_fl2va_canvas_size(geometry_anchor.width(), geometry_anchor.height())?;
            Fl2vaCanvas { height, width }
        }
    };
    let mut images = Vec::with_capacity(ordered.len());
    let mut anchors = Vec::with_capacity(ordered.len());
    for (index, (anchor, image)) in ordered.into_iter().enumerate() {
        let image = if image.width() == canvas.width && image.height() == canvas.height {
            image.clone()
        } else if index == 0 {
            resize_stretched_to_canvas(image, canvas)?
        } else {
            resize_cover_crop_to_canvas(image, canvas)?
        };
        images.push(image);
        anchors.push(anchor);
    }
    Ok(PreparedFl2vaKeyframes {
        canvas,
        images,
        anchors,
    })
}

/// Stretch an RGB image directly to the target canvas using Lanczos-3 antialiasing.
pub fn resize_stretched_to_canvas(image: &RgbImage, canvas: Fl2vaCanvas) -> Result<RgbImage> {
    let canvas = Fl2vaCanvas::new(canvas.height, canvas.width)?;
    resize_lanczos3(image, canvas.width, canvas.height)
}

/// Cover-resize and centre-crop a follower keyframe onto the target canvas.
pub fn resize_cover_crop_to_canvas(image: &RgbImage, canvas: Fl2vaCanvas) -> Result<RgbImage> {
    let canvas = Fl2vaCanvas::new(canvas.height, canvas.width)?;
    if image.width() == canvas.width && image.height() == canvas.height {
        return Ok(image.clone());
    }
    let (width, height, left, top) = cover_crop_geometry(image, canvas)?;
    let resized = resize_lanczos3(image, width, height)?;
    crop_rgb(&resized, left, top, canvas.width, canvas.height)
}

/// Resolve the `17 * n + 5` video geometry and the matching 40 Hz audio grid.
pub fn resolve_fl2va_frame_geometry(
    requested_num_frames: usize,
    clip_length: usize,
    temporal_compression_ratio: usize,
    token_drop: usize,
) -> Result<Fl2vaFrameGeometry> {
    anyhow::ensure!(requested_num_frames > 0, "num_frames must be positive");
    anyhow::ensure!(
        clip_length > 0 && temporal_compression_ratio > 0,
        "VAE temporal geometry must be non-zero"
    );
    let tokens_per_chunk = clip_length.div_ceil(temporal_compression_ratio);
    anyhow::ensure!(
        tokens_per_chunk < clip_length && token_drop < tokens_per_chunk,
        "VAE clip/token geometry cannot represent FL2VA frame alignment"
    );
    let minimum = MINIMAX_H3_MIN_SECONDS * MINIMAX_H3_FPS;
    let maximum = MINIMAX_H3_MAX_SECONDS * MINIMAX_H3_FPS;
    anyhow::ensure!(
        (minimum..=maximum).contains(&requested_num_frames),
        "MiniMax-H3 requires {MINIMAX_H3_MIN_SECONDS} through {MINIMAX_H3_MAX_SECONDS} seconds at {MINIMAX_H3_FPS} fps, which is {minimum} through {maximum} frames; requested {requested_num_frames}"
    );
    let remainder = requested_num_frames % clip_length;
    let alignment = (tokens_per_chunk + clip_length - remainder) % clip_length;
    let num_frames = requested_num_frames
        .checked_add(alignment)
        .context("aligned FL2VA frame count overflow")?;
    let completed_clips = (num_frames - tokens_per_chunk) / clip_length;
    let tail_tokens = tokens_per_chunk - token_drop;
    let num_latent_frames = completed_clips
        .checked_mul(tokens_per_chunk)
        .and_then(|value| value.checked_add(tail_tokens))
        .context("FL2VA latent frame count overflow")?;
    let audio_numerator = num_frames
        .checked_mul(MINIMAX_H3_AUDIO_LATENTS_PER_SECOND)
        .context("FL2VA audio latent count overflow")?;
    let num_audio_latents = round_ratio_ties_even(audio_numerator, MINIMAX_H3_FPS)?;
    Ok(Fl2vaFrameGeometry {
        num_frames,
        num_latent_frames,
        num_audio_latents,
    })
}

/// Apply the released `t = 0.999` augmentation and patchify keyframes in packed order.
pub fn prepare_condition_video_rows(
    condition_latents: &[Tensor],
    patch_size: [usize; 3],
    request_rng: &mut StdRng,
    device: &Device,
) -> Result<Tensor> {
    anyhow::ensure!(
        (1..=2).contains(&condition_latents.len()),
        "FL2VA requires one or two keyframe latents"
    );
    anyhow::ensure!(
        patch_size == MINIMAX_H3_PATCH_SIZE,
        "FL2VA condition rows require patch {MINIMAX_H3_PATCH_SIZE:?}"
    );
    let mut latent_shape = None;
    for (index, condition) in condition_latents.iter().enumerate() {
        let (batch, channels, frames, height, width) = condition
            .dims5()
            .with_context(|| format!("FL2VA keyframe latent {index} must be rank five"))?;
        anyhow::ensure!(
            batch == 1 && channels > 0 && frames == 1 && height > 0 && width > 0,
            "FL2VA keyframe latent {index} must have shape [1, channels, 1, H, W]"
        );
        anyhow::ensure!(
            condition.dtype() == DType::F32,
            "FL2VA keyframe latent {index} must be F32"
        );
        let shape = [channels, height, width];
        if let Some(expected) = latent_shape {
            anyhow::ensure!(
                shape == expected,
                "FL2VA keyframe latent {index} has channel/spatial shape {shape:?}, expected {expected:?}"
            );
        } else {
            latent_shape = Some(shape);
        }
    }

    let scheduler = H3Scheduler::new(12.)?;
    let mut packed = Vec::with_capacity(condition_latents.len());
    let mut row_width = None;
    for condition in condition_latents {
        let condition = condition.to_device(device)?;
        let noise = (0..condition.elem_count())
            .map(|_| StandardNormal.sample(request_rng))
            .collect::<Vec<f32>>();
        let noise = Tensor::from_vec(noise, condition.shape().clone(), device)?;
        let noised = scheduler.scale_noise(&condition, CONDITION_VIDEO_TIMESTEP, &noise)?;
        let rows = layout::patchify_video(&noised, patch_size)?;
        let width = rows.dim(1)?;
        if let Some(expected) = row_width {
            anyhow::ensure!(
                width == expected,
                "FL2VA keyframe patch widths differ: {width} and {expected}"
            );
        } else {
            row_width = Some(width);
        }
        packed.push(rows);
    }
    let refs = packed.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}

trait Fl2vaPromptBackend {
    fn encode_fl2va(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        keyframes: &[RgbImage],
        canvas: Fl2vaCanvas,
    ) -> Result<PromptEncoding>;
}

impl Fl2vaPromptBackend for StreamedMultimodalTextEncoder {
    fn encode_fl2va(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        keyframes: &[RgbImage],
        canvas: Fl2vaCanvas,
    ) -> Result<PromptEncoding> {
        self.encode_fl2va_prompt(tokenizer, prompt, keyframes, canvas.height, canvas.width)
    }
}

#[derive(Clone, Copy)]
struct VaeGeometry {
    latent_channels: usize,
    spatial_ratio: usize,
    temporal_ratio: usize,
    clip_length: usize,
    token_drop: usize,
}

trait Fl2vaVaeBackend {
    fn geometry(&self) -> Result<VaeGeometry>;
    fn encode_keyframe(&self, pixels: &Tensor) -> Result<Tensor>;
}

impl Fl2vaVaeBackend for StreamedVideoVaeEncoder {
    fn geometry(&self) -> Result<VaeGeometry> {
        Ok(VaeGeometry {
            latent_channels: self.config().latent_channels,
            spatial_ratio: self.config().spatial_ratio()?,
            temporal_ratio: self.config().temporal_ratio()?,
            clip_length: self.config().clip_length,
            token_drop: self.config().token_drop,
        })
    }

    fn encode_keyframe(&self, pixels: &Tensor) -> Result<Tensor> {
        self.encode_keyframe_pixels(pixels)
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_fl2va_with(
    text_encoder: &impl Fl2vaPromptBackend,
    video_vae: &impl Fl2vaVaeBackend,
    tokenizer: &Tokenizer,
    prompt_text: &str,
    first_image: Option<&RgbImage>,
    last_image: Option<&RgbImage>,
    options: Fl2vaOptions,
    request_rng: &mut StdRng,
) -> Result<PreparedFl2va> {
    anyhow::ensure!(
        first_image.is_some() || last_image.is_some(),
        "FL2VA requires a first image, a last image, or both"
    );
    anyhow::ensure!(!prompt_text.is_empty(), "FL2VA prompt must not be empty");
    anyhow::ensure!(
        prompt_text.chars().count() <= MINIMAX_H3_MAX_PROMPT_CHARS,
        "FL2VA prompt exceeds the {MINIMAX_H3_MAX_PROMPT_CHARS}-character limit"
    );
    anyhow::ensure!(
        options.audio_channels == MINIMAX_H3_AUDIO_CHANNELS,
        "MiniMax-H3 FL2VA requires stereo audio rows"
    );
    anyhow::ensure!(
        options.audio_latent_channels == MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        "MiniMax-H3 FL2VA requires {} audio latent channels",
        MINIMAX_H3_AUDIO_LATENT_CHANNELS
    );
    anyhow::ensure!(
        options.patch_size == MINIMAX_H3_PATCH_SIZE,
        "MiniMax-H3 FL2VA requires transformer patch {MINIMAX_H3_PATCH_SIZE:?}"
    );
    let vae = video_vae.geometry()?;
    anyhow::ensure!(
        vae.latent_channels == 24
            && vae.spatial_ratio == 16
            && vae.temporal_ratio == 4
            && vae.clip_length == 17
            && vae.token_drop == 3,
        "FL2VA requires the released visual VAE geometry (24 channels, 16x spatial, 4x temporal, clip 17, token_drop 3)"
    );
    if let Some(canvas) = options.canvas {
        let latent_height = canvas.height / vae.spatial_ratio;
        let latent_width = canvas.width / vae.spatial_ratio;
        anyhow::ensure!(
            canvas.height.is_multiple_of(vae.spatial_ratio)
                && canvas.width.is_multiple_of(vae.spatial_ratio)
                && latent_height.is_multiple_of(options.patch_size[1])
                && latent_width.is_multiple_of(options.patch_size[2]),
            "explicit FL2VA canvas {}x{} is not aligned to the VAE and transformer patches",
            canvas.height,
            canvas.width
        );
    }
    let frame_geometry = resolve_fl2va_frame_geometry(
        options.requested_num_frames,
        vae.clip_length,
        vae.temporal_ratio,
        vae.token_drop,
    )?;
    let prepared_keyframes =
        prepare_optional_fl2va_keyframes(first_image, last_image, options.canvas)?;
    let canvas = prepared_keyframes.canvas;
    let latent_height = canvas.height / vae.spatial_ratio;
    let latent_width = canvas.width / vae.spatial_ratio;
    anyhow::ensure!(
        canvas.height.is_multiple_of(vae.spatial_ratio)
            && canvas.width.is_multiple_of(vae.spatial_ratio)
            && latent_height.is_multiple_of(options.patch_size[1])
            && latent_width.is_multiple_of(options.patch_size[2]),
        "FL2VA canvas {}x{} is not aligned to the VAE and transformer patches",
        canvas.height,
        canvas.width
    );

    let prompt =
        text_encoder.encode_fl2va(tokenizer, prompt_text, &prepared_keyframes.images, canvas)?;
    let execution_device = prompt.embeddings.device().clone();
    let mut condition_latents = Vec::with_capacity(prepared_keyframes.images.len());
    for (index, image) in prepared_keyframes.images.iter().enumerate() {
        let pixels = rgb_image_tensor(image)?;
        let latent = video_vae
            .encode_keyframe(&pixels)
            .with_context(|| format!("failed to encode FL2VA keyframe {index}"))?;
        anyhow::ensure!(
            latent.dims() == [1, vae.latent_channels, 1, latent_height, latent_width,],
            "FL2VA keyframe {index} encoded to {:?}, expected [1, {}, 1, {latent_height}, {latent_width}]",
            latent.dims(),
            vae.latent_channels
        );
        condition_latents.push(latent);
    }

    let layout = ConditionedLayout::fl2va(
        &prompt.text_token_tags,
        frame_geometry.num_latent_frames,
        latent_height,
        latent_width,
        frame_geometry.num_audio_latents,
        options.patch_size,
        options.audio_channels,
        &prepared_keyframes.anchors,
        &execution_device,
    )?;
    let condition_video_rows = prepare_condition_video_rows(
        &condition_latents,
        options.patch_size,
        request_rng,
        &execution_device,
    )?;
    anyhow::ensure!(
        condition_video_rows.dim(0)? == layout.condition_video_rows(),
        "layout reserved {} condition rows but keyframes packed into {}",
        layout.condition_video_rows(),
        condition_video_rows.dim(0)?
    );
    anyhow::ensure!(
        condition_video_rows
            .device()
            .same_device(prompt.embeddings.device()),
        "condition rows and Qwen embeddings are on different devices"
    );
    let condition_audio_rows = Tensor::zeros(
        (0, options.audio_latent_channels),
        DType::F32,
        &execution_device,
    )?;
    let patch_volume = options
        .patch_size
        .iter()
        .try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .context("FL2VA patch volume overflow")
        })?;
    let expected_video_row_width = vae
        .latent_channels
        .checked_mul(patch_volume)
        .context("FL2VA condition row width overflow")?;
    anyhow::ensure!(
        condition_video_rows.dim(1)? == expected_video_row_width,
        "VAE keyframe rows have width {}, expected {expected_video_row_width} from {} channels and patch {:?}",
        condition_video_rows.dim(1)?,
        vae.latent_channels,
        options.patch_size
    );

    let initial_target_video_latents = random_normal(
        Shape::from_dims(&[
            1,
            vae.latent_channels,
            frame_geometry.num_latent_frames,
            latent_height,
            latent_width,
        ]),
        request_rng,
        &execution_device,
    )?;
    let target_audio_rows = options
        .audio_channels
        .checked_mul(frame_geometry.num_audio_latents)
        .context("FL2VA target audio row count overflow")?;
    let initial_target_audio_rows = random_normal(
        Shape::from_dims(&[target_audio_rows, options.audio_latent_channels]),
        request_rng,
        &execution_device,
    )?;
    let initial_target_audio_latents = layout::unpack_audio(
        &initial_target_audio_rows,
        options.audio_channels,
        frame_geometry.num_audio_latents,
    )?;

    Ok(PreparedFl2va {
        canvas,
        num_frames: frame_geometry.num_frames,
        num_latent_frames: frame_geometry.num_latent_frames,
        num_audio_latents: frame_geometry.num_audio_latents,
        latent_height,
        latent_width,
        keyframes: prepared_keyframes.images,
        anchors: prepared_keyframes.anchors,
        prompt,
        condition_latents,
        condition_video_rows,
        condition_audio_rows,
        initial_target_video_latents,
        initial_target_audio_latents,
        layout,
        patch_size: options.patch_size,
        audio_channels: options.audio_channels,
    })
}

fn rgb_image_tensor(image: &RgbImage) -> Result<Tensor> {
    Tensor::from_vec(
        image.pixels().to_vec(),
        (image.height(), image.width(), 3),
        &Device::Cpu,
    )?
    .permute((2, 0, 1))?
    .unsqueeze(0)?
    .unsqueeze(2)
    .map_err(Into::into)
}

fn random_normal(shape: Shape, rng: &mut StdRng, device: &Device) -> Result<Tensor> {
    let values = (0..shape.elem_count())
        .map(|_| StandardNormal.sample(rng))
        .collect::<Vec<f32>>();
    Tensor::from_vec(values, shape, device).map_err(Into::into)
}

fn cover_crop_geometry(
    image: &RgbImage,
    canvas: Fl2vaCanvas,
) -> Result<(usize, usize, usize, usize)> {
    let scale = (canvas.width as f64 / image.width() as f64)
        .max(canvas.height as f64 / image.height() as f64);
    anyhow::ensure!(
        scale.is_finite() && scale > 0.,
        "invalid cover-resize scale"
    );
    let width = canvas
        .width
        .max(float_to_usize_ties_even(image.width() as f64 * scale)?);
    let height = canvas
        .height
        .max(float_to_usize_ties_even(image.height() as f64 * scale)?);
    Ok((
        width,
        height,
        (width - canvas.width) / 2,
        (height - canvas.height) / 2,
    ))
}

fn crop_rgb(
    image: &RgbImage,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
) -> Result<RgbImage> {
    let right = left.checked_add(width).context("RGB crop width overflow")?;
    let bottom = top
        .checked_add(height)
        .context("RGB crop height overflow")?;
    anyhow::ensure!(
        right <= image.width() && bottom <= image.height(),
        "RGB crop exceeds resized keyframe"
    );
    let capacity = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("RGB crop buffer overflow")?;
    let mut pixels = Vec::with_capacity(capacity);
    for row in top..bottom {
        let start = (row * image.width() + left) * 3;
        pixels.extend_from_slice(&image.pixels()[start..start + width * 3]);
    }
    RgbImage::new(width, height, pixels)
}

#[derive(Clone)]
struct AxisCoefficients {
    first: usize,
    weights: Vec<f64>,
}

fn resize_lanczos3(image: &RgbImage, width: usize, height: usize) -> Result<RgbImage> {
    anyhow::ensure!(width > 0 && height > 0, "resize target must be non-zero");
    if image.width() == width && image.height() == height {
        return Ok(image.clone());
    }
    let horizontal = lanczos_axis_coefficients(image.width(), width);
    let vertical = lanczos_axis_coefficients(image.height(), height);
    let intermediate_len = image
        .height()
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("horizontal Lanczos buffer overflow")?;
    let mut intermediate = vec![0u8; intermediate_len];
    for row in 0..image.height() {
        for (column, coefficients) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let value = coefficients
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(offset, weight)| {
                        let source = coefficients.first + offset;
                        image.pixels()[(row * image.width() + source) * 3 + channel] as f64 * weight
                    })
                    .sum::<f64>();
                intermediate[(row * width + column) * 3 + channel] = round_u8(value);
            }
        }
    }

    let output_len = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("vertical Lanczos buffer overflow")?;
    let mut pixels = vec![0u8; output_len];
    for (row, coefficients) in vertical.iter().enumerate() {
        for column in 0..width {
            for channel in 0..3 {
                let value = coefficients
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(offset, weight)| {
                        let source = coefficients.first + offset;
                        intermediate[(source * width + column) * 3 + channel] as f64 * weight
                    })
                    .sum::<f64>();
                pixels[(row * width + column) * 3 + channel] = round_u8(value);
            }
        }
    }
    RgbImage::new(width, height, pixels)
}

/// Pillow-compatible filter geometry: half-pixel centres, a widened filter while reducing, and
/// coefficient renormalization after clipping the support to the source image.
fn lanczos_axis_coefficients(source: usize, target: usize) -> Vec<AxisCoefficients> {
    let scale = source as f64 / target as f64;
    let filter_scale = scale.max(1.);
    let support = 3. * filter_scale;
    (0..target)
        .map(|output| {
            let center = (output as f64 + 0.5) * scale;
            let first = ((center - support + 0.5).floor() as isize).max(0) as usize;
            let last = ((center + support + 0.5).floor() as usize).min(source);
            let mut weights = (first..last)
                .map(|input| lanczos3((input as f64 - center + 0.5) / filter_scale))
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            if total != 0. {
                for weight in &mut weights {
                    *weight /= total;
                }
            }
            AxisCoefficients { first, weights }
        })
        .collect()
}

fn lanczos3(value: f64) -> f64 {
    let value = value.abs();
    if value == 0. {
        1.
    } else if value < 3. {
        let pi_value = std::f64::consts::PI * value;
        (pi_value.sin() / pi_value) * ((pi_value / 3.).sin() / (pi_value / 3.))
    } else {
        0.
    }
}

fn round_u8(value: f64) -> u8 {
    value.round().clamp(0., 255.) as u8
}

fn float_to_usize_ties_even(value: f64) -> Result<usize> {
    anyhow::ensure!(
        value.is_finite() && value >= 0. && value <= usize::MAX as f64,
        "rounded image dimension exceeds usize"
    );
    Ok(round_ties_even(value) as usize)
}

fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.
    } else if (floor as u64).is_multiple_of(2) {
        floor
    } else {
        floor + 1.
    }
}

fn round_ratio_ties_even(numerator: usize, denominator: usize) -> Result<usize> {
    anyhow::ensure!(denominator > 0, "rounding denominator must be non-zero");
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled = remainder
        .checked_mul(2)
        .context("rounding remainder overflow")?;
    if doubled < denominator || (doubled == denominator && quotient.is_multiple_of(2)) {
        Ok(quotient)
    } else {
        quotient.checked_add(1).context("rounded ratio overflow")
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn released_duration_range_matches_the_published_assets() {
        for (seconds, expected_frames) in [(5usize, 124usize), (8, 192), (10, 243)] {
            assert_eq!(
                resolve_fl2va_frame_geometry(seconds * MINIMAX_H3_FPS, 17, 4, 3)
                    .unwrap()
                    .num_frames,
                expected_frames,
                "{seconds}s must reproduce the published frame count"
            );
        }
        assert_eq!(
            resolve_fl2va_frame_geometry(MINIMAX_H3_MIN_SECONDS * MINIMAX_H3_FPS, 17, 4, 3)
                .unwrap()
                .num_frames,
            MINIMAX_H3_MIN_ALIGNED_FRAMES
        );
        assert_eq!(
            resolve_fl2va_frame_geometry(MINIMAX_H3_MAX_SECONDS * MINIMAX_H3_FPS, 17, 4, 3)
                .unwrap()
                .num_frames,
            MINIMAX_H3_MAX_ALIGNED_FRAMES
        );
        for outside in [
            MINIMAX_H3_MIN_SECONDS * MINIMAX_H3_FPS - 1,
            MINIMAX_H3_MAX_SECONDS * MINIMAX_H3_FPS + 1,
        ] {
            assert!(
                resolve_fl2va_frame_geometry(outside, 17, 4, 3).is_err(),
                "{outside} frames"
            );
        }
    }
    use super::*;
    use crate::layout::{TEXT_TAG, VIDEO_TAG};
    use rand::SeedableRng;
    use std::cell::RefCell;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn frame_geometry_matches_released_ten_second_alignment() {
        let geometry = resolve_fl2va_frame_geometry(240, 17, 4, 3).unwrap();
        assert_eq!(
            geometry,
            Fl2vaFrameGeometry {
                num_frames: 243,
                num_latent_frames: 72,
                num_audio_latents: 405,
            }
        );
        assert_eq!(
            resolve_fl2va_frame_geometry(96, 17, 4, 3)
                .unwrap()
                .num_frames,
            107
        );
        assert_eq!(
            resolve_fl2va_frame_geometry(346, 17, 4, 3)
                .unwrap()
                .num_frames,
            362
        );
        assert!(resolve_fl2va_frame_geometry(95, 17, 4, 3).is_err());
        assert!(resolve_fl2va_frame_geometry(361, 17, 4, 3).is_err());
    }

    #[test]
    fn first_stretches_and_follower_uses_released_cover_arithmetic() {
        let first = solid_image(3, 2, [17, 31, 47]);
        let last = solid_image(3, 2, [101, 109, 127]);
        let canvas = Fl2vaCanvas::new(3, 4).unwrap();
        let (width, height, left, top) = cover_crop_geometry(&last, canvas).unwrap();
        assert_eq!((width, height, left, top), (4, 3, 0, 0));

        let prepared = prepare_fl2va_keyframes(&first, Some(&last), Some(canvas)).unwrap();
        assert_eq!(prepared.canvas, canvas);
        assert_eq!(
            prepared.anchors,
            [KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        assert!(
            prepared
                .images
                .iter()
                .all(|image| image.width() == 4 && image.height() == 3)
        );
        assert!(
            prepared.images[0]
                .pixels()
                .chunks_exact(3)
                .all(|p| p == [17, 31, 47])
        );
        assert!(
            prepared.images[1]
                .pixels()
                .chunks_exact(3)
                .all(|p| p == [101, 109, 127])
        );
    }

    #[test]
    fn last_only_is_the_geometry_anchor_but_keeps_the_last_anchor_tag() {
        let last = solid_image(1, 1, [23, 47, 89]);
        let prepared = prepare_optional_fl2va_keyframes(None, Some(&last), None).unwrap();
        assert_eq!(
            prepared.canvas,
            Fl2vaCanvas {
                height: 768,
                width: 768
            }
        );
        assert_eq!(prepared.anchors, [KeyframeAnchor::Last]);
        assert_eq!(prepared.images.len(), 1);
        assert_eq!(
            (prepared.images[0].height(), prepared.images[0].width()),
            (768, 768)
        );
        assert!(
            prepared.images[0]
                .pixels()
                .chunks_exact(3)
                .all(|pixel| pixel == [23, 47, 89])
        );
        assert!(prepare_optional_fl2va_keyframes(None, None, None).is_err());
    }

    #[test]
    fn lanczos_pixels_match_pillow_reference() {
        let pixels = (0..5)
            .flat_map(|row| {
                (0..7).flat_map(move |column| {
                    (0..3).map(move |channel| (row * 41 + column * 17 + channel * 67) as u8)
                })
            })
            .collect::<Vec<_>>();
        let image = RgbImage::new(7, 5, pixels).unwrap();
        for (width, height, expected) in [
            (
                11,
                9,
                &[
                    0, 63, 130, 2, 69, 135, 16, 83, 150, 27, 94, 161, 37, 104, 168, 48, 115, 173,
                    59, 126, 180, 69, 136, 203, 80, 147, 241, 94, 161, 255, 100, 167, 255, 8, 75,
                    139, 14, 81, 145, 28, 95, 158, 39, 105, 169, 49, 116, 186, 60, 127, 212, 71,
                    136, 231, 81, 145, 213, 92, 156, 162, 106, 170, 165, 112, 176, 179, 34, 101,
                    164, 40, 107, 170, 54, 121, 183, 65, 132, 194, 75, 143, 207, 86, 153, 235, 97,
                    161, 255, 107, 169, 200, 118, 180, 39, 132, 195, 21, 138, 201, 50, 61, 128,
                    222, 67, 134, 226, 81, 147, 245, 92, 156, 255, 102, 160, 213, 113, 174, 133,
                    124, 207, 144, 134, 229, 118, 145, 240, 0, 156, 253, 0, 159, 255, 11, 81, 148,
                    215, 87, 154, 217, 101, 168, 240, 112, 179, 255, 122, 189, 170, 133, 200, 11,
                    144, 211, 0, 154, 221, 40, 165, 232, 43, 179, 246, 57, 185, 252, 63, 101, 169,
                    93, 107, 171, 97, 121, 191, 116, 132, 223, 129, 142, 255, 87, 153, 255, 4, 165,
                    149, 1, 171, 95, 37, 184, 107, 66, 222, 127, 85, 245, 130, 88, 128, 194, 0,
                    134, 203, 0, 148, 213, 11, 159, 198, 21, 169, 229, 39, 180, 220, 62, 190, 70,
                    75, 203, 0, 82, 213, 3, 90, 210, 26, 104, 204, 28, 110, 154, 216, 14, 160, 241,
                    20, 174, 229, 33, 185, 105, 44, 195, 72, 62, 206, 87, 87, 212, 42, 100, 241,
                    18, 107, 244, 29, 117, 140, 46, 130, 64, 51, 137, 166, 224, 53, 172, 255, 59,
                    186, 234, 74, 197, 44, 85, 207, 0, 91, 218, 0, 96, 220, 41, 107, 255, 60, 118,
                    255, 71, 128, 92, 84, 142, 0, 90, 148,
                ][..],
            ),
            (
                4,
                3,
                &[
                    19, 87, 166, 48, 110, 201, 80, 156, 208, 107, 191, 135, 88, 153, 178, 117, 204,
                    160, 148, 183, 57, 188, 187, 39, 157, 226, 21, 183, 136, 49, 228, 41, 85, 163,
                    37, 130,
                ][..],
            ),
            (
                14,
                10,
                &[
                    0, 61, 128, 0, 65, 132, 7, 74, 141, 18, 85, 153, 26, 93, 161, 34, 101, 166, 43,
                    110, 171, 51, 118, 172, 60, 127, 181, 68, 135, 202, 76, 143, 234, 87, 154, 255,
                    96, 163, 255, 100, 167, 255, 5, 71, 136, 8, 75, 140, 17, 84, 150, 28, 95, 159,
                    36, 103, 168, 44, 111, 179, 53, 120, 196, 61, 127, 214, 70, 135, 223, 78, 143,
                    212, 86, 151, 184, 97, 162, 180, 106, 171, 188, 110, 175, 197, 26, 93, 152, 30,
                    97, 157, 39, 106, 166, 50, 117, 174, 58, 125, 183, 66, 133, 198, 75, 142, 220,
                    83, 148, 255, 92, 153, 255, 100, 159, 213, 108, 167, 97, 119, 179, 54, 128,
                    187, 59, 132, 191, 83, 51, 118, 201, 55, 122, 205, 64, 131, 212, 75, 142, 230,
                    83, 149, 236, 91, 155, 224, 100, 164, 186, 108, 177, 196, 117, 197, 206, 125,
                    209, 157, 133, 217, 27, 144, 227, 0, 152, 236, 0, 155, 240, 8, 71, 138, 232,
                    75, 143, 235, 84, 152, 239, 95, 161, 255, 103, 165, 255, 111, 168, 223, 120,
                    177, 97, 128, 195, 49, 137, 224, 59, 146, 240, 68, 154, 249, 31, 162, 255, 22,
                    167, 255, 30, 169, 255, 40, 89, 156, 171, 93, 159, 173, 102, 168, 177, 113,
                    183, 209, 121, 200, 210, 129, 218, 162, 138, 228, 39, 146, 215, 0, 155, 184, 0,
                    161, 176, 27, 170, 181, 51, 186, 198, 69, 205, 206, 78, 213, 209, 79, 109, 177,
                    43, 113, 178, 46, 122, 187, 53, 133, 205, 71, 141, 229, 77, 149, 255, 67, 158,
                    255, 35, 166, 217, 22, 176, 91, 31, 180, 44, 50, 189, 44, 67, 211, 75, 82, 239,
                    78, 92, 252, 81, 94, 134, 199, 0, 138, 209, 0, 147, 218, 1, 158, 210, 9, 166,
                    176, 18, 174, 185, 34, 183, 196, 62, 191, 149, 79, 198, 32, 88, 213, 0, 91,
                    219, 0, 96, 210, 19, 105, 183, 23, 114, 172, 26, 119, 156, 216, 20, 160, 238,
                    24, 169, 247, 34, 180, 202, 43, 188, 89, 52, 196, 49, 63, 205, 57, 81, 212, 66,
                    94, 216, 34, 103, 244, 26, 109, 247, 32, 116, 196, 48, 127, 90, 55, 136, 45,
                    59, 140, 166, 223, 53, 170, 253, 58, 179, 255, 66, 190, 195, 78, 198, 36, 86,
                    206, 0, 91, 215, 0, 95, 221, 19, 100, 223, 46, 109, 255, 62, 118, 255, 71, 126,
                    184, 78, 137, 33, 88, 146, 0, 93, 150,
                ][..],
            ),
            (
                3,
                2,
                &[
                    44, 114, 188, 83, 163, 177, 130, 195, 105, 141, 202, 85, 187, 128, 74, 186, 85,
                    82,
                ][..],
            ),
        ] {
            let resized = resize_lanczos3(&image, width, height).unwrap();
            assert_eq!(
                resized.pixels(),
                expected,
                "Pillow mismatch at {width}x{height}"
            );
        }
    }

    #[test]
    fn condition_noise_is_reproducible_and_advances_the_request_rng() {
        let conditions = [
            Tensor::zeros((1, 1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap(),
            Tensor::ones((1, 1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap(),
        ];
        let mut first_rng = StdRng::seed_from_u64(7);
        let first = prepare_condition_video_rows(
            &conditions,
            MINIMAX_H3_PATCH_SIZE,
            &mut first_rng,
            &Device::Cpu,
        )
        .unwrap();
        let next_after_conditions: f32 = StandardNormal.sample(&mut first_rng);

        let mut second_rng = StdRng::seed_from_u64(7);
        let second = prepare_condition_video_rows(
            &conditions,
            MINIMAX_H3_PATCH_SIZE,
            &mut second_rng,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(first.dims(), &[2, 4]);
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let next_again: f32 = StandardNormal.sample(&mut second_rng);
        assert_eq!(next_after_conditions.to_bits(), next_again.to_bits());
    }

    #[test]
    fn invalid_condition_shapes_fail_before_consuming_request_rng() {
        let conditions = [
            Tensor::zeros((1, 1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((1, 1, 1, 2, 4), DType::F32, &Device::Cpu).unwrap(),
        ];
        let mut actual_rng = StdRng::seed_from_u64(23);
        let mut untouched_rng = actual_rng.clone();
        let result = prepare_condition_video_rows(
            &conditions,
            MINIMAX_H3_PATCH_SIZE,
            &mut actual_rng,
            &Device::Cpu,
        );
        let Err(error) = result else {
            panic!("mismatched condition geometry must fail");
        };
        assert!(error.to_string().contains("expected"));
        let actual: f32 = StandardNormal.sample(&mut actual_rng);
        let untouched: f32 = StandardNormal.sample(&mut untouched_rng);
        assert_eq!(actual.to_bits(), untouched.to_bits());
    }

    #[test]
    fn tiny_orchestration_prepares_qwen_vae_and_layout_boundary() {
        let text = MockTextEncoder::default();
        let vae = MockVideoVae::default();
        let tokenizer = Tokenizer::new(WordLevel::default());
        let first = solid_image(16, 9, [13, 29, 61]);
        let last = solid_image(9, 16, [71, 83, 97]);
        let options = Fl2vaOptions::official(120).with_canvas(Fl2vaCanvas::new(32, 32).unwrap());
        let mut rng = StdRng::seed_from_u64(11);
        let prepared = prepare_fl2va_with(
            &text,
            &vae,
            &tokenizer,
            "a paper boat on a stream",
            Some(&first),
            Some(&last),
            options,
            &mut rng,
        )
        .unwrap();

        assert_eq!(
            prepared.canvas,
            Fl2vaCanvas {
                height: 32,
                width: 32
            }
        );
        assert_eq!(prepared.num_frames, 124);
        assert_eq!(prepared.num_latent_frames, 37);
        assert_eq!(prepared.num_audio_latents, 207);
        assert_eq!((prepared.latent_height, prepared.latent_width), (2, 2));
        assert_eq!(
            prepared.anchors,
            [KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        assert_eq!(prepared.condition_latents.len(), 2);
        assert_eq!(prepared.condition_video_rows.dims(), &[2, 96]);
        assert_eq!(prepared.condition_audio_rows.dims(), &[0, 32]);
        assert_eq!(
            prepared.initial_target_video_latents.dims(),
            &[1, 24, 37, 2, 2]
        );
        assert_eq!(prepared.initial_target_audio_latents.dims(), &[2, 32, 207]);
        assert_eq!(prepared.layout.condition_video_rows(), 2);
        assert_eq!(prepared.layout.condition_audio_rows(), 0);
        assert_eq!(prepared.prompt.embeddings.dims(), &[1, 3, 8]);
        assert!(
            prepared
                .condition_video_rows
                .device()
                .same_device(prepared.prompt.embeddings.device())
        );
        assert_eq!(
            text.seen.borrow().as_slice(),
            &[("a paper boat on a stream".to_owned(), 2, 32, 32)]
        );
        assert_eq!(vae.encoded.borrow().as_slice(), &[(32, 32), (32, 32)]);

        let mut expected_rng = StdRng::seed_from_u64(11);
        for _ in 0..2 * 24 * 2 * 2 {
            let _: f32 = StandardNormal.sample(&mut expected_rng);
        }
        let expected_video = (0..prepared.initial_target_video_latents.elem_count())
            .map(|_| StandardNormal.sample(&mut expected_rng))
            .collect::<Vec<f32>>();
        let actual_video = prepared
            .initial_target_video_latents
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual_video, expected_video);
        let expected_audio_rows = (0..prepared.initial_target_audio_latents.elem_count())
            .map(|_| StandardNormal.sample(&mut expected_rng))
            .collect::<Vec<f32>>();
        let actual_audio_rows = layout::pack_audio(&prepared.initial_target_audio_latents)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual_audio_rows, expected_audio_rows);
        let expected_next: f32 = StandardNormal.sample(&mut expected_rng);
        let actual_next: f32 = StandardNormal.sample(&mut rng);
        assert_eq!(actual_next.to_bits(), expected_next.to_bits());

        let config = mock_transformer_config();
        prepared
            .validate_transformer_contract(&config, &Device::Cpu)
            .unwrap();
        let mut wrong_video_channels = config.clone();
        wrong_video_channels.in_channels = 25;
        assert!(
            prepared
                .validate_transformer_contract(&wrong_video_channels, &Device::Cpu)
                .is_err()
        );
        let mut wrong_patch = config;
        wrong_patch.patch_size = [1, 1, 2];
        assert!(
            prepared
                .validate_transformer_contract(&wrong_patch, &Device::Cpu)
                .is_err()
        );
    }

    #[test]
    fn tiny_last_only_orchestration_is_packed_at_the_last_anchor() {
        let text = MockTextEncoder::default();
        let vae = MockVideoVae::default();
        let tokenizer = Tokenizer::new(WordLevel::default());
        let last = solid_image(9, 16, [71, 83, 97]);
        let options = Fl2vaOptions::official(120).with_canvas(Fl2vaCanvas::new(32, 32).unwrap());
        let mut rng = StdRng::seed_from_u64(19);
        let prepared = prepare_fl2va_with(
            &text,
            &vae,
            &tokenizer,
            "arrive at this frame",
            None,
            Some(&last),
            options,
            &mut rng,
        )
        .unwrap();
        assert_eq!(prepared.anchors, [KeyframeAnchor::Last]);
        assert_eq!(prepared.condition_video_rows.dims(), &[1, 96]);
        assert_eq!(prepared.layout.condition_video_rows(), 1);
        assert_eq!(
            text.seen.borrow().as_slice(),
            &[("arrive at this frame".to_owned(), 1, 32, 32)]
        );
        assert_eq!(vae.encoded.borrow().as_slice(), &[(32, 32)]);
    }

    #[test]
    fn invalid_fl2va_contracts_fail_before_conditioner_execution() {
        let text = MockTextEncoder::default();
        let vae = MockVideoVae::default();
        let tokenizer = Tokenizer::new(WordLevel::default());
        let first = solid_image(16, 9, [13, 29, 61]);

        let mut rng = StdRng::seed_from_u64(31);
        assert!(
            prepare_fl2va_with(
                &text,
                &vae,
                &tokenizer,
                "prompt",
                None,
                None,
                Fl2vaOptions::official(120),
                &mut rng,
            )
            .is_err()
        );
        assert!(
            prepare_fl2va_with(
                &text,
                &vae,
                &tokenizer,
                "",
                Some(&first),
                None,
                Fl2vaOptions::official(120),
                &mut rng,
            )
            .is_err()
        );
        let mut wrong_patch = Fl2vaOptions::official(120);
        wrong_patch.patch_size = [1, 1, 2];
        assert!(
            prepare_fl2va_with(
                &text,
                &vae,
                &tokenizer,
                "prompt",
                Some(&first),
                None,
                wrong_patch,
                &mut rng,
            )
            .is_err()
        );
        let unaligned = Fl2vaOptions::official(120).with_canvas(Fl2vaCanvas::new(33, 32).unwrap());
        assert!(
            prepare_fl2va_with(
                &text,
                &vae,
                &tokenizer,
                "prompt",
                Some(&first),
                None,
                unaligned,
                &mut rng,
            )
            .is_err()
        );
        assert!(text.seen.borrow().is_empty());
        assert!(vae.encoded.borrow().is_empty());
    }

    fn solid_image(width: usize, height: usize, color: [u8; 3]) -> RgbImage {
        let pixels = (0..width * height).flat_map(|_| color).collect::<Vec<_>>();
        RgbImage::new(width, height, pixels).unwrap()
    }

    fn mock_transformer_config() -> TransformerConfig {
        TransformerConfig {
            class_name: "MiniMaxH3Transformer3DModel".to_owned(),
            num_attention_heads: 1,
            attention_head_dim: 8,
            hidden_size: 8,
            num_layers: 1,
            num_refiner_layers: 1,
            ffn_dim: 8,
            in_channels: 24,
            audio_in_channels: 32,
            patch_size: MINIMAX_H3_PATCH_SIZE,
            text_dim: 8,
            freq_dim: 8,
            time_embed_hidden_dim: 8,
            time_embed_dim: 4,
            rope_freq_dim: 2,
            rope_theta: 10_000.,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    #[derive(Default)]
    struct MockTextEncoder {
        seen: RefCell<Vec<(String, usize, usize, usize)>>,
    }

    impl Fl2vaPromptBackend for MockTextEncoder {
        fn encode_fl2va(
            &self,
            _tokenizer: &Tokenizer,
            prompt: &str,
            keyframes: &[RgbImage],
            canvas: Fl2vaCanvas,
        ) -> Result<PromptEncoding> {
            anyhow::ensure!(
                keyframes.iter().all(|image| {
                    image.height() == canvas.height && image.width() == canvas.width
                }),
                "mock received an unprepared keyframe"
            );
            self.seen.borrow_mut().push((
                prompt.to_owned(),
                keyframes.len(),
                canvas.height,
                canvas.width,
            ));
            Ok(PromptEncoding {
                embeddings: Tensor::zeros((1, 3, 8), DType::F32, &Device::Cpu)?,
                text_token_tags: vec![TEXT_TAG, VIDEO_TAG, TEXT_TAG],
                token_ids: vec![1, 2, 3],
                numerical_contract: crate::text_encoder::test_qwen_numerical_contract(3),
            })
        }
    }

    #[derive(Default)]
    struct MockVideoVae {
        encoded: RefCell<Vec<(usize, usize)>>,
    }

    impl Fl2vaVaeBackend for MockVideoVae {
        fn geometry(&self) -> Result<VaeGeometry> {
            Ok(VaeGeometry {
                latent_channels: 24,
                spatial_ratio: 16,
                temporal_ratio: 4,
                clip_length: 17,
                token_drop: 3,
            })
        }

        fn encode_keyframe(&self, pixels: &Tensor) -> Result<Tensor> {
            let (batch, channels, frames, height, width) = pixels.dims5()?;
            anyhow::ensure!(
                (batch, channels, frames) == (1, 3, 1),
                "mock received invalid pixels"
            );
            self.encoded.borrow_mut().push((height, width));
            Tensor::zeros(
                (1, 24, 1, height / 16, width / 16),
                DType::F32,
                &Device::Cpu,
            )
            .map_err(Into::into)
        }
    }
}
