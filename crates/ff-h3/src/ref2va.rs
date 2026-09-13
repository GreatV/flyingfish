//! End-to-end preparation and denoising orchestration for MiniMax-H3 Ref2VA.
//!
//! This module starts after media decoding. Image/video inputs are owned RGB8
//! frames, video frames already carry MiniMax-H3's fixed 24 fps, and audio is
//! mono or stereo PCM already resampled to 32 kHz. Keeping that boundary
//! explicit prevents a lost frame/sample rate from silently changing the
//! conditioning. The caller remains responsible for the released reference
//! resize policy (2048-pixel image short edge; 768-pixel video short edge).
//! A production caller must open the denoiser from root-level
//! `transformer_ref/` and the visual encoder from root-level `vae/`; selecting
//! checkpoint component paths is deliberately outside this media/Tensor API.

use crate::policy::H3QwenNumericalContract;
use crate::{
    audio_vae_encoder::StreamedAudioVaeEncoder,
    h3_conditioning::{
        CONDITION_VIDEO_TIMESTEP, ConditionedLayout, ReferenceBlock,
        denoise_conditioned_with_observer,
    },
    layout,
    model::StreamedTransformer,
    multimodal_text_encoder::{PreprocessedVision, RgbImage, StreamedMultimodalTextEncoder},
    pipeline::{DenoiseObserver, T2vaExecutionOptions, T2vaLatents, T2vaSchedule},
    scheduler::H3Scheduler,
    text_encoder::PromptEncoding,
    video_vae_encoder::{StreamedVideoVaeEncoder, VideoVaeEncoderConfig},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

pub const REF2VA_FPS: usize = 24;
pub const REF2VA_QWEN_SAMPLE_FPS: usize = 2;
pub const REF2VA_QWEN_SPATIAL_MERGE: usize = 2;
pub const REF2VA_MIN_DURATION_SECONDS: usize = 5;
pub const REF2VA_MAX_DURATION_SECONDS: usize = 15;

const MAX_IMAGES: usize = 9;
const MAX_VIDEOS: usize = 3;
const MAX_AUDIOS: usize = 3;
const MAX_REFERENCES: usize = 12;

/// One decoded and rate-normalized Ref2VA input.
///
/// Video frames must be consecutive 24 fps RGB8 frames of one equal geometry.
/// A soundtrack and a standalone audio clip are `[channels, samples]` floating
/// PCM at 32 kHz; one channel is duplicated to stereo by the audio VAE encoder.
#[derive(Clone)]
pub enum Ref2vaReference {
    Image(RgbImage),
    Video {
        frames: Vec<RgbImage>,
        soundtrack: Option<Tensor>,
    },
    Audio(Tensor),
}

/// Generated-video geometry at the public pixel/frame boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ref2vaTarget {
    /// Requested frames; preparation rounds this up to the next VAE-aligned count.
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
}

/// All immutable conditioning and initial target noise needed by the existing
/// conditioned denoising loop.
pub struct PreparedRef2va {
    pub prompt_embeddings: Tensor,
    pub text_token_tags: Vec<u32>,
    pub token_ids: Vec<u32>,
    pub reference_blocks: Vec<ReferenceBlock>,
    pub condition_video_rows: Tensor,
    pub condition_audio_rows: Tensor,
    pub initial_target_video_latents: Tensor,
    pub initial_target_audio_latents: Tensor,
    pub layout: ConditionedLayout,
    pub target: Ref2vaTarget,
    pub qwen_numerical_contract: H3QwenNumericalContract,
}

impl PreparedRef2va {
    /// Denoise with the root-level `transformer_ref/` checkpoint partition.
    /// Its config is intentionally identical to `transformer/`, so callers
    /// must select the reference-trained component by path when opening it.
    pub fn denoise(
        &self,
        transformer: &StreamedTransformer,
        schedule: T2vaSchedule,
        options: T2vaExecutionOptions,
        observer: &mut dyn DenoiseObserver,
    ) -> Result<T2vaLatents> {
        denoise_conditioned_with_observer(
            transformer,
            &self.prompt_embeddings,
            &self.text_token_tags,
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

/// Borrowed, read-only Ref2VA component set.
pub struct Ref2vaPipeline<'a> {
    tokenizer: &'a Tokenizer,
    text_encoder: &'a StreamedMultimodalTextEncoder,
    video_encoder: &'a StreamedVideoVaeEncoder,
    audio_encoder: &'a StreamedAudioVaeEncoder,
    transformer_ref: &'a StreamedTransformer,
}

impl<'a> Ref2vaPipeline<'a> {
    /// Bind the shared encoders to the reference-trained transformer. The last
    /// argument must have been opened from root-level `transformer_ref/`.
    pub fn new(
        tokenizer: &'a Tokenizer,
        text_encoder: &'a StreamedMultimodalTextEncoder,
        video_encoder: &'a StreamedVideoVaeEncoder,
        audio_encoder: &'a StreamedAudioVaeEncoder,
        transformer_ref: &'a StreamedTransformer,
    ) -> Result<Self> {
        text_encoder.validate_tokenizer(tokenizer)?;
        anyhow::ensure!(
            transformer_ref.config().in_channels == video_encoder.config().latent_channels,
            "Ref2VA transformer/video-VAE latent channel counts differ"
        );
        anyhow::ensure!(
            transformer_ref.config().audio_in_channels == audio_encoder.config().latent_channels,
            "Ref2VA transformer/audio-VAE latent channel counts differ"
        );
        anyhow::ensure!(
            audio_encoder.config().sampling_rate == 32_000,
            "released Ref2VA requires a 32 kHz audio VAE"
        );
        Ok(Self {
            tokenizer,
            text_encoder,
            video_encoder,
            audio_encoder,
            transformer_ref,
        })
    }

    /// Encode the official labelled Qwen3-VL presentation, independently of
    /// the visual/audio VAE pass. This is useful when staging long runs.
    pub fn encode_prompt(
        &self,
        prompt: &str,
        references: &[Ref2vaReference],
        target_num_frames: usize,
    ) -> Result<PromptEncoding> {
        validate_reference_counts(references)?;
        let mut images = Vec::new();
        let mut sampled_videos = Vec::new();
        let mut video_timestamps = Vec::new();
        for reference in references {
            match reference {
                Ref2vaReference::Image(image) => {
                    validate_visual_geometry(std::slice::from_ref(image), "reference image")?;
                    let ratio = image.width() as f64 / image.height() as f64;
                    anyhow::ensure!(
                        (0.25..=4.).contains(&ratio),
                        "a Ref2VA image must have an aspect ratio between 1:4 and 4:1"
                    );
                    images.push(image.clone());
                }
                Ref2vaReference::Video { frames, .. } => {
                    let frames = truncate_video(frames, target_num_frames)?;
                    validate_visual_geometry(frames, "reference video")?;
                    let (sampled, timestamps) = sample_video_for_qwen(frames)?;
                    sampled_videos.push(sampled);
                    video_timestamps.push(timestamps);
                }
                Ref2vaReference::Audio(_) => {}
            }
        }

        let image_vision = (!images.is_empty())
            .then(|| self.text_encoder.preprocess_images(&images))
            .transpose()?;
        let video_vision = (!sampled_videos.is_empty())
            .then(|| self.text_encoder.preprocess_videos(&sampled_videos))
            .transpose()?;
        let image_counts = image_vision
            .as_ref()
            .map(|vision| vision.merged_token_counts(REF2VA_QWEN_SPATIAL_MERGE))
            .transpose()?
            .unwrap_or_default();
        let video_presentations =
            build_video_presentations(video_vision.as_ref(), &video_timestamps)?;
        let specials = PresentationSpecialTokens::from_tokenizer(self.tokenizer)?;
        let (token_ids, text_token_tags) = build_presentation_with(
            prompt,
            references,
            &image_counts,
            &video_presentations,
            specials,
            |text| tokenize(self.tokenizer, text),
        )?;
        self.text_encoder.encode_preprocessed_presentation(
            &token_ids,
            &text_token_tags,
            image_vision.as_ref(),
            video_vision.as_ref(),
        )
    }

    /// Run every conditioning encoder, build the interleaved Ref2VA layout,
    /// and draw condition/video/audio noise in the checkpoint-defined order.
    pub fn prepare(
        &self,
        prompt: &str,
        references: &[Ref2vaReference],
        target: Ref2vaTarget,
        seed: u64,
    ) -> Result<PreparedRef2va> {
        validate_reference_counts(references)?;
        let resolved = resolve_target(
            target,
            self.video_encoder.config(),
            self.audio_encoder.config().hop_length()?,
            self.audio_encoder.config().sampling_rate,
        )?;
        let prompt = self.encode_prompt(prompt, references, resolved.target.num_frames)?;
        let (_, _, prompt_width) = prompt.embeddings.dims3()?;
        anyhow::ensure!(
            prompt_width == self.transformer_ref.config().text_dim,
            "Qwen conditioning width is {prompt_width}, expected {}",
            self.transformer_ref.config().text_dim
        );

        let maximum_audio_samples = resolved
            .target
            .num_frames
            .checked_mul(self.audio_encoder.config().sampling_rate as usize)
            .context("reference audio duration overflow")?
            / REF2VA_FPS;
        let mut encoded = Vec::with_capacity(references.len());
        for reference in references {
            match reference {
                Ref2vaReference::Image(image) => {
                    let pixels = rgb_frames_tensor(
                        std::slice::from_ref(image),
                        self.video_encoder.device(),
                    )?;
                    let visual = self.video_encoder.encode_keyframe_pixels(&pixels)?;
                    let (_, _, latent_frames, latent_height, latent_width) = visual.dims5()?;
                    anyhow::ensure!(
                        latent_frames == 1,
                        "an image reference must encode to one latent frame"
                    );
                    encoded.push(EncodedReference {
                        block: ReferenceBlock::Image {
                            latent_frames,
                            latent_height,
                            latent_width,
                        },
                        visual: Some(visual),
                        audio_rows: None,
                    });
                }
                Ref2vaReference::Video { frames, soundtrack } => {
                    let frames = truncate_video(frames, resolved.target.num_frames)?;
                    let vae_frames = select_video_vae_frames(frames, self.video_encoder.config())?;
                    let pixels = rgb_frames_tensor(vae_frames, self.video_encoder.device())?;
                    let visual = self.video_encoder.encode_video_pixels(&pixels)?;
                    let (_, _, latent_frames, latent_height, latent_width) = visual.dims5()?;
                    let audio_rows = soundtrack
                        .as_ref()
                        .map(|audio| self.encode_audio(audio, maximum_audio_samples))
                        .transpose()?;
                    let audio_latents = audio_rows
                        .as_ref()
                        .map(|rows| {
                            let count = rows.dim(0)?;
                            anyhow::ensure!(
                                count.is_multiple_of(2),
                                "video soundtrack rows are not stereo channel-major"
                            );
                            Ok(count / 2)
                        })
                        .transpose()?
                        .unwrap_or(0);
                    encoded.push(EncodedReference {
                        block: ReferenceBlock::Video {
                            latent_frames,
                            latent_height,
                            latent_width,
                            audio_latents,
                        },
                        visual: Some(visual),
                        audio_rows,
                    });
                }
                Ref2vaReference::Audio(audio) => {
                    let audio_rows = self.encode_audio(audio, maximum_audio_samples)?;
                    let audio_row_count = audio_rows.dim(0)?;
                    anyhow::ensure!(
                        audio_row_count.is_multiple_of(2),
                        "reference audio rows are not stereo channel-major"
                    );
                    let audio_latents = audio_row_count / 2;
                    encoded.push(EncodedReference {
                        block: ReferenceBlock::Audio { audio_latents },
                        visual: None,
                        audio_rows: Some(audio_rows),
                    });
                }
            }
        }

        assemble_encoded(
            prompt,
            encoded,
            resolved,
            self.transformer_ref.config().patch_size,
            self.video_encoder.config().latent_channels,
            self.audio_encoder.config().latent_channels,
            seed,
            self.transformer_ref.device(),
        )
    }

    /// Convenience end-to-end path through conditioning and the existing H3
    /// denoiser. Decoding remains shared with the T2VA/FL2VA output modules.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        references: &[Ref2vaReference],
        target: Ref2vaTarget,
        seed: u64,
        schedule: T2vaSchedule,
        options: T2vaExecutionOptions,
        observer: &mut dyn DenoiseObserver,
    ) -> Result<T2vaLatents> {
        self.prepare(prompt, references, target, seed)?.denoise(
            self.transformer_ref,
            schedule,
            options,
            observer,
        )
    }

    fn encode_audio(&self, audio: &Tensor, maximum_samples: usize) -> Result<Tensor> {
        let (_, samples) = audio
            .dims2()
            .context("reference PCM must be [channels, samples]")?;
        anyhow::ensure!(samples > 0, "reference PCM must contain samples");
        let length = samples.min(maximum_samples);
        anyhow::ensure!(
            length > 0,
            "target duration leaves no reference PCM samples"
        );
        let audio = audio
            .narrow(1, 0, length)?
            .to_device(self.audio_encoder.device())?;
        self.audio_encoder
            .encode_condition_rows(&audio, self.audio_encoder.config().sampling_rate)
    }
}

struct ResolvedTarget {
    target: Ref2vaTarget,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_latents: usize,
}

struct EncodedReference {
    block: ReferenceBlock,
    visual: Option<Tensor>,
    audio_rows: Option<Tensor>,
}

#[derive(Clone, Copy)]
struct PresentationSpecialTokens {
    vision_start: u32,
    vision_end: u32,
    image_pad: u32,
    video_pad: u32,
}

impl PresentationSpecialTokens {
    fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        Ok(Self {
            vision_start: required_token_id(tokenizer, "<|vision_start|>")?,
            vision_end: required_token_id(tokenizer, "<|vision_end|>")?,
            image_pad: required_token_id(tokenizer, "<|image_pad|>")?,
            video_pad: required_token_id(tokenizer, "<|video_pad|>")?,
        })
    }
}

struct VideoPresentation {
    tokens_per_block: usize,
    timestamps: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn assemble_encoded(
    prompt: PromptEncoding,
    encoded: Vec<EncodedReference>,
    target: ResolvedTarget,
    patch_size: [usize; 3],
    video_latent_channels: usize,
    audio_latent_channels: usize,
    seed: u64,
    device: &Device,
) -> Result<PreparedRef2va> {
    let reference_blocks = encoded.iter().map(|entry| entry.block).collect::<Vec<_>>();
    let layout = ConditionedLayout::ref2va(
        &prompt.text_token_tags,
        &reference_blocks,
        target.latent_frames,
        target.latent_height,
        target.latent_width,
        target.audio_latents,
        patch_size,
        2,
        device,
    )?;

    let mut rng = StdRng::seed_from_u64(seed);
    let scheduler = H3Scheduler::new(12.)?;
    let mut condition_video = Vec::new();
    let mut condition_audio = Vec::new();
    for entry in encoded {
        if let Some(clean) = entry.visual {
            let clean = clean.to_device(device)?.to_dtype(DType::F32)?;
            anyhow::ensure!(
                clean.dim(1)? == video_latent_channels,
                "reference visual latent channel count differs from the video VAE"
            );
            let noise = random_normal(clean.shape().clone(), &mut rng, device)?;
            let noised = scheduler.scale_noise(&clean, CONDITION_VIDEO_TIMESTEP, &noise)?;
            condition_video.push(layout::patchify_video(&noised, patch_size)?);
        }
        if let Some(rows) = entry.audio_rows {
            let rows = rows.to_device(device)?.to_dtype(DType::F32)?;
            anyhow::ensure!(
                rows.dim(1)? == audio_latent_channels,
                "reference audio latent channel count differs from the audio VAE"
            );
            condition_audio.push(rows);
        }
    }
    anyhow::ensure!(
        !condition_video.is_empty(),
        "Ref2VA needs a visual reference"
    );
    let condition_video_refs = condition_video.iter().collect::<Vec<_>>();
    let condition_video_rows = Tensor::cat(&condition_video_refs, 0)?;
    let condition_audio_rows = if condition_audio.is_empty() {
        Tensor::zeros((0, audio_latent_channels), DType::F32, device)?
    } else {
        let refs = condition_audio.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 0)?
    };

    anyhow::ensure!(
        condition_video_rows.dim(0)? == layout.condition_video_rows(),
        "encoded visual rows disagree with the Ref2VA layout"
    );
    anyhow::ensure!(
        condition_audio_rows.dim(0)? == layout.condition_audio_rows(),
        "encoded audio rows disagree with the Ref2VA layout"
    );
    let initial_target_video_latents = random_normal(
        Shape::from_dims(&[
            1,
            video_latent_channels,
            target.latent_frames,
            target.latent_height,
            target.latent_width,
        ]),
        &mut rng,
        device,
    )?;
    let initial_target_audio_rows = random_normal(
        Shape::from_dims(&[2 * target.audio_latents, audio_latent_channels]),
        &mut rng,
        device,
    )?;
    let initial_target_audio_latents =
        layout::unpack_audio(&initial_target_audio_rows, 2, target.audio_latents)?;
    let (prompt_batch, _, _) = prompt.embeddings.dims3()?;
    anyhow::ensure!(
        prompt_batch == 1,
        "Ref2VA prompt embeddings must have batch size one"
    );

    Ok(PreparedRef2va {
        prompt_embeddings: prompt.embeddings.to_device(device)?,
        text_token_tags: prompt.text_token_tags,
        token_ids: prompt.token_ids,
        reference_blocks,
        condition_video_rows,
        condition_audio_rows,
        initial_target_video_latents,
        initial_target_audio_latents,
        layout,
        target: target.target,
        qwen_numerical_contract: prompt.numerical_contract,
    })
}

fn resolve_target(
    target: Ref2vaTarget,
    video: &VideoVaeEncoderConfig,
    audio_hop: usize,
    audio_sample_rate: u32,
) -> Result<ResolvedTarget> {
    anyhow::ensure!(
        target.height > 0
            && target.width > 0
            && target.height.is_multiple_of(32)
            && target.width.is_multiple_of(32),
        "Ref2VA target height and width must be positive multiples of 32"
    );
    let temporal_ratio = video.temporal_ratio()?;
    let tokens_per_chunk = video.clip_length.div_ceil(temporal_ratio);
    anyhow::ensure!(
        video.clip_length > 0
            && tokens_per_chunk < video.clip_length
            && video.token_drop < tokens_per_chunk,
        "visual VAE clip/token geometry cannot represent Ref2VA frame alignment"
    );
    anyhow::ensure!(
        target.num_frames > 0,
        "Ref2VA target frame count must be positive"
    );
    let remainder = target.num_frames % video.clip_length;
    let alignment = tokens_per_chunk
        .checked_add(video.clip_length)
        .and_then(|value| value.checked_sub(remainder))
        .context("Ref2VA frame alignment overflow")?
        % video.clip_length;
    let aligned_num_frames = target
        .num_frames
        .checked_add(alignment)
        .context("aligned Ref2VA frame count overflow")?;
    let minimum_frames = REF2VA_MIN_DURATION_SECONDS * REF2VA_FPS;
    let maximum_frames = REF2VA_MAX_DURATION_SECONDS * REF2VA_FPS;
    anyhow::ensure!(
        (minimum_frames..=maximum_frames).contains(&aligned_num_frames),
        "Ref2VA target must run between 5 and 15 seconds at 24 fps after VAE alignment"
    );
    let cycles = (aligned_num_frames - tokens_per_chunk) / video.clip_length;
    let latent_frames = (cycles + 1)
        .checked_mul(tokens_per_chunk)
        .and_then(|frames| frames.checked_sub(video.token_drop))
        .context("target video latent frame count overflow")?;
    let spatial_ratio = video.spatial_ratio()?;
    anyhow::ensure!(
        target.height.is_multiple_of(spatial_ratio) && target.width.is_multiple_of(spatial_ratio),
        "Ref2VA target dimensions are not divisible by the visual VAE compression ratio"
    );
    let audio_numerator = aligned_num_frames
        .checked_mul(audio_sample_rate as usize)
        .context("target audio latent count overflow")?;
    let audio_denominator = REF2VA_FPS
        .checked_mul(audio_hop)
        .context("target audio latent denominator overflow")?;
    let audio_latents = round_ratio_ties_even(audio_numerator, audio_denominator)?;
    anyhow::ensure!(audio_latents > 0, "Ref2VA target has no audio latents");
    Ok(ResolvedTarget {
        target: Ref2vaTarget {
            num_frames: aligned_num_frames,
            ..target
        },
        latent_frames,
        latent_height: target.height / spatial_ratio,
        latent_width: target.width / spatial_ratio,
        audio_latents,
    })
}

fn select_video_vae_frames<'a>(
    frames: &'a [RgbImage],
    config: &VideoVaeEncoderConfig,
) -> Result<&'a [RgbImage]> {
    anyhow::ensure!(!frames.is_empty(), "reference video must contain frames");
    let temporal_ratio = config.temporal_ratio()?;
    let tokens_per_chunk = config.clip_length.div_ceil(temporal_ratio);
    let cycles = frames
        .len()
        .saturating_sub(tokens_per_chunk)
        .checked_div(config.clip_length)
        .context("visual VAE clip length must be positive")?
        .max(1);
    let selected = cycles
        .checked_mul(config.clip_length)
        .and_then(|frames| frames.checked_add(tokens_per_chunk))
        .context("reference video frame selection overflow")?
        .min(frames.len());
    Ok(&frames[..selected])
}

fn truncate_video(frames: &[RgbImage], target_num_frames: usize) -> Result<&[RgbImage]> {
    anyhow::ensure!(!frames.is_empty(), "reference video must contain frames");
    let frames = &frames[..frames.len().min(target_num_frames)];
    let minimum = REF2VA_FPS / REF2VA_QWEN_SAMPLE_FPS + 1;
    anyhow::ensure!(
        frames.len() >= minimum,
        "a 24 fps Ref2VA video must contain at least {minimum} frames for Qwen's 2 fps temporal pairs"
    );
    Ok(frames)
}

fn validate_visual_geometry(frames: &[RgbImage], name: &str) -> Result<()> {
    let first = frames
        .first()
        .with_context(|| format!("{name} has no frames"))?;
    anyhow::ensure!(
        first.height().is_multiple_of(32) && first.width().is_multiple_of(32),
        "{name} dimensions must be multiples of 32 before Ref2VA encoding"
    );
    anyhow::ensure!(
        frames
            .iter()
            .all(|frame| frame.height() == first.height() && frame.width() == first.width()),
        "all {name} frames must have equal dimensions"
    );
    Ok(())
}

fn validate_reference_counts(references: &[Ref2vaReference]) -> Result<()> {
    anyhow::ensure!(
        !references.is_empty(),
        "Ref2VA requires at least one reference"
    );
    anyhow::ensure!(
        references.len() <= MAX_REFERENCES,
        "Ref2VA accepts at most {MAX_REFERENCES} references"
    );
    let images = references
        .iter()
        .filter(|entry| matches!(entry, Ref2vaReference::Image(_)))
        .count();
    let videos = references
        .iter()
        .filter(|entry| matches!(entry, Ref2vaReference::Video { .. }))
        .count();
    let audios = references
        .iter()
        .filter(|entry| matches!(entry, Ref2vaReference::Audio(_)))
        .count();
    anyhow::ensure!(
        images <= MAX_IMAGES,
        "Ref2VA accepts at most {MAX_IMAGES} images"
    );
    anyhow::ensure!(
        videos <= MAX_VIDEOS,
        "Ref2VA accepts at most {MAX_VIDEOS} videos"
    );
    anyhow::ensure!(
        audios <= MAX_AUDIOS,
        "Ref2VA accepts at most {MAX_AUDIOS} audio clips"
    );
    anyhow::ensure!(
        images + videos > 0,
        "Ref2VA cannot condition on standalone audio alone"
    );
    Ok(())
}

fn sample_video_for_qwen(frames: &[RgbImage]) -> Result<(Vec<RgbImage>, Vec<f64>)> {
    let stride = REF2VA_FPS / REF2VA_QWEN_SAMPLE_FPS;
    let sampled = (0..frames.len())
        .step_by(stride)
        .map(|index| frames[index].clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        sampled.len() >= 2,
        "Qwen must see at least two sampled reference-video frames"
    );
    let mut timestamps = (0..sampled.len())
        .map(|index| index as f64 / REF2VA_QWEN_SAMPLE_FPS as f64)
        .collect::<Vec<_>>();
    if !timestamps.len().is_multiple_of(2) {
        timestamps.push(
            *timestamps
                .last()
                .context("sampled video has no timestamp")?,
        );
    }
    let block_timestamps = timestamps
        .chunks_exact(2)
        .map(|pair| (pair[0] + pair[1]) / 2.)
        .collect();
    Ok((sampled, block_timestamps))
}

fn build_video_presentations(
    videos: Option<&PreprocessedVision>,
    timestamps: &[Vec<f64>],
) -> Result<Vec<VideoPresentation>> {
    let Some(videos) = videos else {
        anyhow::ensure!(
            timestamps.is_empty(),
            "video timestamps exist without Qwen video input"
        );
        return Ok(Vec::new());
    };
    anyhow::ensure!(
        videos.grids().len() == timestamps.len(),
        "Qwen video grids and timestamp sets differ"
    );
    videos
        .grids()
        .iter()
        .zip(timestamps)
        .map(|(grid, timestamps)| {
            anyhow::ensure!(
                grid.temporal == timestamps.len(),
                "Qwen temporal blocks and timestamp labels differ"
            );
            let tokens_per_block = (grid.height / REF2VA_QWEN_SPATIAL_MERGE)
                .checked_mul(grid.width / REF2VA_QWEN_SPATIAL_MERGE)
                .context("Qwen video block token count overflow")?;
            Ok(VideoPresentation {
                tokens_per_block,
                timestamps: timestamps.clone(),
            })
        })
        .collect()
}

fn build_presentation_with(
    prompt: &str,
    references: &[Ref2vaReference],
    image_counts: &[usize],
    videos: &[VideoPresentation],
    specials: PresentationSpecialTokens,
    mut tokenize_text: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut token_ids = Vec::new();
    let mut tags = Vec::new();
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut audio_index = 0usize;

    let emit_text = |value: &str,
                     token_ids: &mut Vec<u32>,
                     tags: &mut Vec<u32>,
                     tokenize_text: &mut dyn FnMut(&str) -> Result<Vec<u32>>|
     -> Result<()> {
        let ids = tokenize_text(value)?;
        token_ids.extend_from_slice(&ids);
        tags.extend(std::iter::repeat_n(layout::TEXT_TAG, ids.len()));
        Ok(())
    };
    let emit_vision = |pad: u32, count: usize, token_ids: &mut Vec<u32>, tags: &mut Vec<u32>| {
        token_ids.push(specials.vision_start);
        token_ids.extend(std::iter::repeat_n(pad, count));
        token_ids.push(specials.vision_end);
        tags.extend(std::iter::repeat_n(layout::VIDEO_TAG, count + 2));
    };

    for reference in references {
        let has_audio = matches!(reference, Ref2vaReference::Audio(_))
            || matches!(
                reference,
                Ref2vaReference::Video {
                    soundtrack: Some(_),
                    ..
                }
            );
        if has_audio {
            audio_index += 1;
            emit_text(
                &format!("<Audio {audio_index}>: "),
                &mut token_ids,
                &mut tags,
                &mut tokenize_text,
            )?;
        }
        match reference {
            Ref2vaReference::Image(_) => {
                image_index += 1;
                emit_text(
                    &format!("<Picture {image_index}>: "),
                    &mut token_ids,
                    &mut tags,
                    &mut tokenize_text,
                )?;
                let count = *image_counts
                    .get(image_index - 1)
                    .context("image presentation has no matching Qwen grid")?;
                emit_vision(specials.image_pad, count, &mut token_ids, &mut tags);
            }
            Ref2vaReference::Video { .. } => {
                video_index += 1;
                emit_text(
                    &format!("<Video {video_index}>: "),
                    &mut token_ids,
                    &mut tags,
                    &mut tokenize_text,
                )?;
                let video = videos
                    .get(video_index - 1)
                    .context("video presentation has no matching Qwen grid")?;
                for &timestamp in &video.timestamps {
                    emit_text(
                        &format_timestamp(timestamp),
                        &mut token_ids,
                        &mut tags,
                        &mut tokenize_text,
                    )?;
                    emit_vision(
                        specials.video_pad,
                        video.tokens_per_block,
                        &mut token_ids,
                        &mut tags,
                    );
                }
            }
            Ref2vaReference::Audio(_) => {}
        }
    }
    anyhow::ensure!(image_index == image_counts.len(), "unused Qwen image grids");
    anyhow::ensure!(video_index == videos.len(), "unused Qwen video grids");
    emit_text(prompt, &mut token_ids, &mut tags, &mut tokenize_text)?;
    anyhow::ensure!(
        !token_ids.is_empty(),
        "Ref2VA presentation tokenized to nothing"
    );
    Ok((token_ids, tags))
}

fn rgb_frames_tensor(frames: &[RgbImage], device: &Device) -> Result<Tensor> {
    let first = frames.first().context("RGB frame sequence is empty")?;
    validate_visual_geometry(frames, "visual reference")?;
    let bytes_per_frame = first
        .height()
        .checked_mul(first.width())
        .and_then(|pixels| pixels.checked_mul(3))
        .context("RGB frame size overflow")?;
    let mut pixels = Vec::with_capacity(
        bytes_per_frame
            .checked_mul(frames.len())
            .context("RGB video size overflow")?,
    );
    for frame in frames {
        pixels.extend_from_slice(frame.pixels());
    }
    Tensor::from_vec(
        pixels,
        (frames.len(), first.height(), first.width(), 3),
        device,
    )?
    .to_dtype(DType::F32)?
    .permute((3, 0, 1, 2))?
    .unsqueeze(0)
    .map_err(Into::into)
}

fn random_normal(shape: Shape, rng: &mut StdRng, device: &Device) -> Result<Tensor> {
    let values = (0..shape.elem_count())
        .map(|_| StandardNormal.sample(rng))
        .collect::<Vec<f32>>();
    Tensor::from_vec(values, shape, device).map_err(Into::into)
}

fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| anyhow::anyhow!("failed to tokenize Ref2VA presentation: {error}"))
}

fn required_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .with_context(|| format!("H3 tokenizer is missing {token}"))
}

fn format_timestamp(timestamp: f64) -> String {
    let rounded = round_ties_even_f64(timestamp * 10.) / 10.;
    format!("<{rounded:.1} seconds>")
}

fn round_ties_even_f64(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 || (fraction == 0.5 && (floor as i64) % 2 == 0) {
        floor
    } else {
        floor + 1.
    }
}

fn round_ratio_ties_even(numerator: usize, denominator: usize) -> Result<usize> {
    anyhow::ensure!(denominator > 0, "rounding denominator must be non-zero");
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let twice = remainder
        .checked_mul(2)
        .context("rounding remainder overflow")?;
    if twice < denominator || (twice == denominator && quotient.is_multiple_of(2)) {
        Ok(quotient)
    } else {
        quotient.checked_add(1).context("rounded value overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(value: u8) -> RgbImage {
        RgbImage::new(32, 32, vec![value; 32 * 32 * 3]).unwrap()
    }

    fn fake_prompt() -> PromptEncoding {
        PromptEncoding {
            embeddings: Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap(),
            text_token_tags: vec![layout::TEXT_TAG, layout::VIDEO_TAG],
            token_ids: vec![1, 2],
            numerical_contract: crate::text_encoder::test_qwen_numerical_contract(2),
        }
    }

    fn video_config() -> VideoVaeEncoderConfig {
        VideoVaeEncoderConfig {
            class_name: "AutoencoderKLMiniMaxH3".to_owned(),
            in_channels: 3,
            latent_channels: 24,
            block_out_channels: vec![32; 6],
            layers_per_block: 1,
            spatial_downsample_factors: vec![2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: vec![1, 2, 2, 1, 1, 1],
            norm_num_groups: 32,
            norm_eps: 1e-6,
            spatial_padding_mode: "reflect".to_owned(),
            clip_length: 17,
            token_drop: 3,
            latents_mean: vec![0.; 24],
            latents_std: vec![1.; 24],
        }
    }

    #[test]
    fn presentation_preserves_reference_order_and_audio_before_video() {
        let references = vec![
            Ref2vaReference::Image(image(1)),
            Ref2vaReference::Audio(Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap()),
            Ref2vaReference::Video {
                frames: vec![image(2); 13],
                soundtrack: Some(Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap()),
            },
        ];
        let specials = PresentationSpecialTokens {
            vision_start: 100,
            vision_end: 101,
            image_pad: 102,
            video_pad: 103,
        };
        let video = VideoPresentation {
            tokens_per_block: 2,
            timestamps: vec![0.25, 1.0],
        };
        let id_for_text = |text: &str| -> Result<Vec<u32>> {
            let id = match text {
                "<Picture 1>: " => 10,
                "<Audio 1>: " => 11,
                "<Audio 2>: " => 12,
                "<Video 1>: " => 13,
                "<0.2 seconds>" => 14,
                "<1.0 seconds>" => 15,
                "prompt" => 16,
                other => anyhow::bail!("unexpected presentation text {other:?}"),
            };
            Ok(vec![id])
        };
        let (ids, tags) =
            build_presentation_with("prompt", &references, &[2], &[video], specials, id_for_text)
                .unwrap();
        assert_eq!(
            ids,
            vec![
                10, 100, 102, 102, 101, 11, 12, 13, 14, 100, 103, 103, 101, 15, 100, 103, 103, 101,
                16,
            ]
        );
        assert_eq!(tags.len(), ids.len());
        assert_eq!(&tags[1..5], &[layout::VIDEO_TAG; 4]);
        assert_eq!(&tags[9..13], &[layout::VIDEO_TAG; 4]);
        assert_eq!(&tags[14..18], &[layout::VIDEO_TAG; 4]);
        for index in [0usize, 5, 6, 7, 8, 13, 18] {
            assert_eq!(tags[index], layout::TEXT_TAG);
        }
    }

    #[test]
    fn qwen_sampling_and_timestamp_rounding_match_the_reference() {
        let frames = vec![image(0); 25];
        let (sampled, timestamps) = sample_video_for_qwen(&frames).unwrap();
        assert_eq!(sampled.len(), 3);
        assert_eq!(timestamps, vec![0.25, 1.0]);
        assert_eq!(format_timestamp(timestamps[0]), "<0.2 seconds>");
    }

    #[test]
    fn encoded_assembly_draws_and_shapes_every_denoiser_input() {
        let encoded = vec![EncodedReference {
            block: ReferenceBlock::Image {
                latent_frames: 1,
                latent_height: 2,
                latent_width: 2,
            },
            visual: Some(Tensor::zeros((1, 1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap()),
            audio_rows: None,
        }];
        let resolved = ResolvedTarget {
            target: Ref2vaTarget {
                num_frames: 124,
                height: 32,
                width: 32,
            },
            latent_frames: 1,
            latent_height: 2,
            latent_width: 2,
            audio_latents: 2,
        };
        let prepared = assemble_encoded(
            fake_prompt(),
            encoded,
            resolved,
            [1, 2, 2],
            1,
            2,
            7,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(prepared.condition_video_rows.dims(), &[1, 4]);
        assert_eq!(prepared.condition_audio_rows.dims(), &[0, 2]);
        assert_eq!(
            prepared.initial_target_video_latents.dims(),
            &[1, 1, 1, 2, 2]
        );
        assert_eq!(prepared.initial_target_audio_latents.dims(), &[2, 2, 2]);
        assert_eq!(prepared.layout.condition_video_rows(), 1);
        assert_eq!(prepared.layout.condition_audio_rows(), 0);

        let mut expected_rng = StdRng::seed_from_u64(7);
        let _condition_noise = random_normal(
            Shape::from_dims(&[1, 1, 1, 2, 2]),
            &mut expected_rng,
            &Device::Cpu,
        )
        .unwrap();
        let expected_video = random_normal(
            Shape::from_dims(&[1, 1, 1, 2, 2]),
            &mut expected_rng,
            &Device::Cpu,
        )
        .unwrap();
        let expected_audio_rows =
            random_normal(Shape::from_dims(&[4, 2]), &mut expected_rng, &Device::Cpu).unwrap();
        assert_eq!(
            prepared
                .initial_target_video_latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected_video
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(
            layout::pack_audio(&prepared.initial_target_audio_latents)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected_audio_rows
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn target_audio_and_video_geometry_matches_the_ten_second_profile() {
        let config = video_config();
        let target = resolve_target(
            Ref2vaTarget {
                num_frames: 240,
                height: 512,
                width: 896,
            },
            &config,
            800,
            32_000,
        )
        .unwrap();
        assert_eq!(target.target.num_frames, 243);
        assert_eq!(target.latent_frames, 72);
        assert_eq!(target.latent_height, 32);
        assert_eq!(target.latent_width, 56);
        assert_eq!(target.audio_latents, 405);
    }

    #[test]
    fn reference_video_snaps_48_frames_down_to_39_before_vae_padding() {
        let frames = vec![image(0); 48];
        let selected = select_video_vae_frames(&frames, &video_config()).unwrap();
        assert_eq!(selected.len(), 39);
    }
}
