//! Streamed Qwen3-VL conditioning for MiniMax-H3's visual modes.
//!
//! The released H3 checkpoint uses an otherwise standard Qwen3-VL model as a
//! conditioner, but consumes `hidden_states[50]` instead of logits or the
//! final normalized state.  This module keeps the checkpoint read-only and
//! materializes one weight group at a time through [`ModelWeights`].

use crate::{
    core,
    layout::{TEXT_TAG, VIDEO_TAG},
    policy::{
        ExecutionBackendPolicy, H3QwenNumericalContract, H3QwenVisionGridModality,
        H3QwenVisionLinearGeometry,
    },
    text_encoder::PromptEncoding,
};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use ff_core::weights::{CachePolicy, ModelWeights, WeightAccessStats, WeightSource};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, io::Cursor, num::NonZeroUsize, path::Path};
use tokenizers::Tokenizer;

/// Whether this device may run the pinned exact kernels.
///
/// Every exact-kernel dispatch in this crate goes through here, so a host the
/// pinned profile does not cover takes the portable Candle path instead of
/// failing at the first transcribed operator.
/// Whether this device may run the kernels compiled from this repository.
fn tuned_cuda(device: &candle_core::Device) -> bool {
    crate::cuda::tuned_kernels_available(device)
}

/// Whether this host's vendor libraries are the reference ones, which is what
/// the cuBLASLt and cuDNN operators reproduce.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn reference_cuda(device: &candle_core::Device) -> bool {
    crate::cuda::profile::reference_libraries_available(device)
}

const LAYER_NORM_EPS: f64 = 1e-6;
const DEFAULT_TARGET_HIDDEN_STATE: usize = 50;
const MAX_KEYFRAMES: usize = 2;
const QWEN_VERIFIED_FL_PATCH_ROWS: usize = 4_032;
const QWEN_VERIFIED_REF_PATCH_ROWS: usize = 28_224;
const QWEN_VERIFIED_FL_MERGER_ROWS: usize = 1_008;
const QWEN_VERIFIED_REF_MERGER_ROWS: usize = 7_056;

/// An owned, row-major RGB8 image.
///
/// FL2VA callers normally construct this after putting the keyframe on the
/// resolved H3 canvas.  Keeping that resize outside the conditioner mirrors
/// the official pipeline: the first frame is stretched to the canvas and the
/// optional last frame is cover-cropped before Qwen preprocessing begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        anyhow::ensure!(
            width > 0 && height > 0,
            "RGB image dimensions must be non-zero"
        );
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .context("RGB image size overflow")?;
        anyhow::ensure!(
            pixels.len() == expected,
            "RGB image has {} bytes, expected {expected} for {width}x{height}",
            pixels.len()
        );
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Decode a single, non-animated RGB8 PNG without modifying the source.
    pub fn from_png(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read keyframe PNG {}", path.display()))?;
        Self::from_png_bytes(&bytes)
            .with_context(|| format!("failed to decode keyframe PNG {}", path.display()))
    }

    /// Decode a single, non-animated RGB8 PNG from memory.
    pub fn from_png_bytes(bytes: &[u8]) -> Result<Self> {
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().context("failed to read PNG header")?;
        let info = reader.info();
        anyhow::ensure!(
            info.animation_control.is_none() && info.frame_control.is_none(),
            "keyframe PNG must not be animated"
        );
        anyhow::ensure!(
            info.color_type == png::ColorType::Rgb && info.bit_depth == png::BitDepth::Eight,
            "keyframe PNG must use RGB8 encoding"
        );
        let png_width = info.width;
        let png_height = info.height;
        let width = usize::try_from(png_width).context("PNG width exceeds usize")?;
        let height = usize::try_from(png_height).context("PNG height exceeds usize")?;
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .context("decoded PNG size overflow")?;
        anyhow::ensure!(
            reader.output_buffer_size() == Some(expected),
            "decoded PNG buffer size does not match RGB dimensions"
        );
        let mut pixels = vec![0u8; expected];
        let output = reader
            .next_frame(&mut pixels)
            .context("failed to decode PNG pixels")?;
        anyhow::ensure!(
            output.width == png_width
                && output.height == png_height
                && output.color_type == png::ColorType::Rgb
                && output.bit_depth == png::BitDepth::Eight
                && output.buffer_size() == expected,
            "decoded PNG output does not match RGB8 dimensions"
        );
        reader.finish().context("failed to finish PNG decoding")?;
        Self::new(width, height, pixels)
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// MiniMax-H3's released FL2VA canvas geometry for an aspect ratio.
///
/// The result is `(height, width)`, both divisible by 32.  This is the exact
/// scalar geometry rule used before the official pipeline resizes keyframes;
/// image interpolation itself remains a caller concern so first/last-frame
/// anchoring semantics cannot be confused.
pub fn resolve_fl2va_canvas_size(
    aspect_width: usize,
    aspect_height: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(
        aspect_width > 0 && aspect_height > 0,
        "FL2VA aspect dimensions must be positive"
    );
    let ratio = aspect_width as f64 / aspect_height as f64;
    anyhow::ensure!(
        (0.25..=4.).contains(&ratio),
        "MiniMax-H3 supports FL2VA aspect ratios from 1:4 to 4:1"
    );
    let (mut width, mut height) = if ratio >= 1. {
        (768. * ratio, 768.)
    } else {
        (768., 768. / ratio)
    };
    let maximum_pixels = (768 * 1344) as f64;
    let area = width * height;
    if area > maximum_pixels {
        let scale = (maximum_pixels / area).sqrt();
        width *= scale;
        height *= scale;
    }
    Ok((
        round_ties_even(height / 32.).max(1.) as usize * 32,
        round_ties_even(width / 32.).max(1.) as usize * 32,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionModality {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionGrid {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

impl VisionGrid {
    fn patch_count(self) -> Result<usize> {
        self.temporal
            .checked_mul(self.height)
            .and_then(|count| count.checked_mul(self.width))
            .context("vision patch count overflow")
    }

    fn merged_count(self, merge: usize) -> Result<usize> {
        anyhow::ensure!(
            self.height.is_multiple_of(merge) && self.width.is_multiple_of(merge),
            "vision grid is not divisible by the spatial merge"
        );
        self.temporal
            .checked_mul(self.height / merge)
            .and_then(|count| count.checked_mul(self.width / merge))
            .context("merged vision token count overflow")
    }
}

/// CPU-side official Qwen patch layout, reusable by Ref2VA presentation code.
pub struct PreprocessedVision {
    modality: VisionModality,
    grids: Vec<VisionGrid>,
    patches: Vec<f32>,
    patch_width: usize,
}

impl PreprocessedVision {
    pub const fn modality(&self) -> VisionModality {
        self.modality
    }

    pub fn grids(&self) -> &[VisionGrid] {
        &self.grids
    }

    pub fn patch_rows(&self) -> usize {
        self.patches.len() / self.patch_width
    }

    pub fn merged_token_counts(&self, spatial_merge_size: usize) -> Result<Vec<usize>> {
        self.grids
            .iter()
            .map(|grid| grid.merged_count(spatial_merge_size))
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct EncoderConfig {
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    text_config: LanguageConfig,
    vision_config: VisionConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct LanguageConfig {
    head_dim: usize,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: MropeConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct MropeConfig {
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
}

#[derive(Clone, Debug, Deserialize)]
struct VisionConfig {
    deepstack_visual_indexes: Vec<usize>,
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    in_channels: usize,
    num_heads: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ProcessorSize {
    longest_edge: usize,
    shortest_edge: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct ProcessorConfig {
    size: ProcessorSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

struct VisionEncoding {
    merged: Tensor,
    deepstack: Vec<Tensor>,
}

/// Qwen3-VL conditioner streamed from the original H3 text-encoder shards.
pub struct StreamedMultimodalTextEncoder {
    weights: ModelWeights,
    config: EncoderConfig,
    image_processor: ProcessorConfig,
    video_processor: ProcessorConfig,
    device: Device,
    target_hidden_state: usize,
    attention_query_chunk_size: usize,
}

impl StreamedMultimodalTextEncoder {
    pub fn open(
        component_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
        target_hidden_state: usize,
        attention_query_chunk_size: usize,
    ) -> Result<Self> {
        let component_dir = component_dir.as_ref();
        let config =
            read_json::<EncoderConfig>(&component_dir.join("config.json"), "Qwen3-VL config")?;
        let image_processor = read_json::<ProcessorConfig>(
            &component_dir.join("preprocessor_config.json"),
            "Qwen3-VL image processor config",
        )?;
        let video_processor = read_json::<ProcessorConfig>(
            &component_dir.join("video_preprocessor_config.json"),
            "Qwen3-VL video processor config",
        )?;
        let weights = ModelWeights::open(component_dir, source, cache_policy)?;
        let encoder = Self::new(
            weights,
            config,
            image_processor,
            video_processor,
            device,
            target_hidden_state,
            attention_query_chunk_size,
        )?;
        encoder.validate_released_architecture()?;
        encoder.validate_shape_inventory()?;
        Ok(encoder)
    }

    pub fn open_h3_default(
        component_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
        attention_query_chunk_size: usize,
    ) -> Result<Self> {
        let encoder = Self::open(
            component_dir,
            source,
            cache_policy,
            device,
            DEFAULT_TARGET_HIDDEN_STATE,
            attention_query_chunk_size,
        )?;
        encoder.validate_official_config()?;
        Ok(encoder)
    }

    fn new(
        weights: ModelWeights,
        config: EncoderConfig,
        image_processor: ProcessorConfig,
        video_processor: ProcessorConfig,
        device: Device,
        target_hidden_state: usize,
        attention_query_chunk_size: usize,
    ) -> Result<Self> {
        validate_config(&config, &image_processor, &video_processor)?;
        anyhow::ensure!(
            target_hidden_state > 0 && target_hidden_state < config.text_config.num_hidden_layers,
            "H3's target hidden state must be after at least one layer and before the final Qwen layer"
        );
        anyhow::ensure!(
            attention_query_chunk_size > 0,
            "attention query chunk size must be non-zero"
        );
        #[cfg(feature = "cuda")]
        if reference_cuda(&device) {
            crate::cuda::profile::validate_exact_profile(&device)
                .context("Qwen conditioner CUDA profile preflight")?;
        }
        Ok(Self {
            weights,
            config,
            image_processor,
            video_processor,
            device,
            target_hidden_state,
            attention_query_chunk_size,
        })
    }

    pub fn access_stats(&self) -> WeightAccessStats {
        self.weights.access_stats()
    }

    /// Fail before visual preprocessing or weight execution when tokenizer and model IDs differ.
    pub fn validate_tokenizer(&self, tokenizer: &Tokenizer) -> Result<()> {
        for (token, expected) in [
            ("<|vision_start|>", self.config.vision_start_token_id),
            ("<|vision_end|>", self.config.vision_end_token_id),
            ("<|image_pad|>", self.config.image_token_id),
            ("<|video_pad|>", self.config.video_token_id),
        ] {
            validate_tokenizer_special_id(tokenizer, token, expected)?;
        }
        Ok(())
    }

    /// Apply Qwen's image resize, `[-1, 1]` normalization, temporal repeat,
    /// and spatial-merge-major patchification.
    pub fn preprocess_images(&self, images: &[RgbImage]) -> Result<PreprocessedVision> {
        anyhow::ensure!(!images.is_empty(), "image batch must not be empty");
        preprocess_sequences(
            VisionModality::Image,
            &images
                .iter()
                .cloned()
                .map(|image| vec![image])
                .collect::<Vec<_>>(),
            &self.image_processor,
        )
    }

    /// Preprocess already-sampled Ref2VA video frame sequences.  Sampling and
    /// timestamp labels remain presentation concerns; this method implements
    /// the official Qwen video resize/normalization/patch layout.
    pub fn preprocess_videos(&self, videos: &[Vec<RgbImage>]) -> Result<PreprocessedVision> {
        anyhow::ensure!(!videos.is_empty(), "video batch must not be empty");
        preprocess_sequences(VisionModality::Video, videos, &self.video_processor)
    }

    /// Build and encode the official FL2VA presentation.
    ///
    /// `keyframes` must already use the resolved generation canvas.  Passing
    /// the canvas explicitly catches accidental conditioning/generation
    /// geometry drift before any model weights are touched.
    pub fn encode_fl2va_prompt(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        keyframes: &[RgbImage],
        canvas_height: usize,
        canvas_width: usize,
    ) -> Result<PromptEncoding> {
        self.validate_tokenizer(tokenizer)?;
        anyhow::ensure!(
            (1..=MAX_KEYFRAMES).contains(&keyframes.len()),
            "FL2VA requires one or two keyframes"
        );
        anyhow::ensure!(
            canvas_height > 0
                && canvas_width > 0
                && canvas_height.is_multiple_of(32)
                && canvas_width.is_multiple_of(32),
            "FL2VA canvas dimensions must be positive multiples of 32"
        );
        anyhow::ensure!(
            keyframes
                .iter()
                .all(|image| image.height == canvas_height && image.width == canvas_width),
            "every FL2VA keyframe must already match the {canvas_width}x{canvas_height} target canvas"
        );

        let images = self.preprocess_images(keyframes)?;
        let counts = images.merged_token_counts(self.config.vision_config.spatial_merge_size)?;
        let mut token_ids = Vec::new();
        let mut text_token_tags = Vec::new();
        for (index, count) in counts.into_iter().enumerate() {
            let label = format!("<Picture {}>: ", index + 1);
            let label_ids = tokenize(tokenizer, &label)?;
            token_ids.extend_from_slice(&label_ids);
            text_token_tags.extend(std::iter::repeat_n(TEXT_TAG, label_ids.len()));

            token_ids.push(self.config.vision_start_token_id);
            token_ids.extend(std::iter::repeat_n(self.config.image_token_id, count));
            token_ids.push(self.config.vision_end_token_id);
            text_token_tags.extend(std::iter::repeat_n(VIDEO_TAG, count + 2));
        }
        let prompt_ids = tokenize(tokenizer, prompt)?;
        token_ids.extend_from_slice(&prompt_ids);
        text_token_tags.extend(std::iter::repeat_n(TEXT_TAG, prompt_ids.len()));
        self.encode_preprocessed_presentation(&token_ids, &text_token_tags, Some(&images), None)
    }

    /// Encode a caller-built Qwen/H3 presentation.  This is the extension
    /// point used by Ref2VA: callers may supply image and video batches, with
    /// one contiguous pad-token run per grid (one run per temporal video
    /// block, as in Qwen3-VL's timestamped presentation).
    pub fn encode_preprocessed_presentation(
        &self,
        token_ids: &[u32],
        text_token_tags: &[u32],
        images: Option<&PreprocessedVision>,
        videos: Option<&PreprocessedVision>,
    ) -> Result<PromptEncoding> {
        anyhow::ensure!(!token_ids.is_empty(), "presentation must not be empty");
        anyhow::ensure!(
            token_ids.len() == text_token_tags.len(),
            "presentation IDs and H3 modality tags have different lengths"
        );
        anyhow::ensure!(
            token_ids
                .iter()
                .all(|&id| id < self.config.text_config.vocab_size as u32),
            "presentation token ID exceeds the Qwen vocabulary"
        );
        anyhow::ensure!(
            text_token_tags.iter().all(|&tag| tag < 3),
            "presentation contains an invalid H3 modality tag"
        );
        if let Some(images) = images {
            anyhow::ensure!(
                images.modality == VisionModality::Image,
                "image input was preprocessed as video"
            );
        }
        if let Some(videos) = videos {
            anyhow::ensure!(
                videos.modality == VisionModality::Video,
                "video input was preprocessed as image"
            );
        }

        let mm_types = token_ids
            .iter()
            .map(|&id| {
                if id == self.config.image_token_id {
                    1
                } else if id == self.config.video_token_id {
                    2
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let image_grids = images.map_or(&[][..], |batch| batch.grids.as_slice());
        let video_grids = videos.map_or(&[][..], |batch| batch.grids.as_slice());
        validate_qwen_exact_attention_geometry(
            reference_cuda(&self.device),
            token_ids.len(),
            image_grids,
            video_grids,
        )?;
        let image_total_patch_rows = images.map_or(0, PreprocessedVision::patch_rows);
        let video_total_patch_rows = videos.map_or(0, PreprocessedVision::patch_rows);
        let max_vision_segment_rows =
            image_grids
                .iter()
                .chain(video_grids)
                .try_fold(0usize, |maximum, grid| {
                    let rows = grid
                        .height
                        .checked_mul(grid.width)
                        .context("Qwen vision segment row count overflow")?;
                    Ok::<_, anyhow::Error>(maximum.max(rows))
                })?;
        let configured_query_rows = NonZeroUsize::new(self.attention_query_chunk_size)
            .context("Qwen configured query rows must be non-zero")?;
        let language_rows =
            NonZeroUsize::new(token_ids.len()).context("Qwen language rows must be non-zero")?;
        let vision_geometry = H3QwenVisionLinearGeometry::from_patch_rows(
            image_total_patch_rows,
            video_total_patch_rows,
        )?;
        let canonical_grids = canonical_ordered_vision_grids(
            &mm_types,
            image_grids,
            video_grids,
            self.config.vision_config.spatial_merge_size,
        )?;
        let backend = ExecutionBackendPolicy::from_device(&self.device);
        let numerical_contract = H3QwenNumericalContract::for_target_with_grids(
            backend,
            backend
                .is_cuda()
                .then(|| crate::policy::CudaCapabilities::from_device(&self.device)),
            configured_query_rows,
            language_rows,
            max_vision_segment_rows,
            vision_geometry,
            &canonical_grids,
        )?;
        numerical_contract.validate_for_with_grids(
            &self.device,
            configured_query_rows,
            language_rows,
            max_vision_segment_rows,
            vision_geometry,
            &canonical_grids,
        )?;
        let position_ids = multimodal_position_ids(
            &mm_types,
            image_grids,
            video_grids,
            self.config.vision_config.spatial_merge_size,
        )?;

        let image_features = images
            .map(|batch| self.encode_visual_batch(batch))
            .transpose()?;
        let video_features = videos
            .map(|batch| self.encode_visual_batch(batch))
            .transpose()?;
        let embeddings = self.encode_language(
            token_ids,
            &mm_types,
            &position_ids,
            image_features.as_ref(),
            video_features.as_ref(),
        )?;
        Ok(PromptEncoding {
            embeddings,
            text_token_tags: text_token_tags.to_vec(),
            token_ids: token_ids.to_vec(),
            numerical_contract,
        })
    }

    /// Verify every tensor shape consumed by the released Qwen3-VL
    /// conditioner.  The check reads safetensors headers only.
    pub fn validate_official_shape_inventory(&self) -> Result<()> {
        self.validate_official_config()?;
        self.validate_shape_inventory()
    }

    fn validate_official_config(&self) -> Result<()> {
        self.validate_released_architecture()?;
        anyhow::ensure!(
            self.target_hidden_state == DEFAULT_TARGET_HIDDEN_STATE,
            "released H3 conditioning must return Qwen hidden_states[{DEFAULT_TARGET_HIDDEN_STATE}]"
        );
        Ok(())
    }

    fn validate_released_architecture(&self) -> Result<()> {
        let vision = &self.config.vision_config;
        let language = &self.config.text_config;
        anyhow::ensure!(
            vision.depth == 27
                && vision.deepstack_visual_indexes == [8, 16, 24]
                && vision.hidden_size == 1152
                && vision.intermediate_size == 4304
                && vision.num_heads == 16
                && vision.num_position_embeddings == 2304
                && vision.out_hidden_size == 5120
                && vision.patch_size == 16
                && vision.temporal_patch_size == 2
                && vision.spatial_merge_size == 2,
            "checkpoint does not have the released H3 Qwen3-VL vision inventory"
        );
        anyhow::ensure!(
            language.hidden_size == 5120
                && language.intermediate_size == 25600
                && language.num_hidden_layers == 64
                && language.num_attention_heads == 64
                && language.num_key_value_heads == 8
                && language.head_dim == 128
                && language.rope_scaling.mrope_section == [24, 20, 20],
            "checkpoint does not have the released H3 Qwen3-VL language inventory"
        );
        Ok(())
    }

    fn validate_shape_inventory(&self) -> Result<()> {
        let vision = &self.config.vision_config;
        let language = &self.config.text_config;
        expect_shape(
            &self.weights,
            "model.visual.patch_embed.proj.weight",
            &[
                vision.hidden_size,
                vision.in_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
        )?;
        expect_shape(
            &self.weights,
            "model.visual.patch_embed.proj.bias",
            &[vision.hidden_size],
        )?;
        expect_shape(
            &self.weights,
            "model.visual.pos_embed.weight",
            &[vision.num_position_embeddings, vision.hidden_size],
        )?;
        for layer in 0..vision.depth {
            validate_vision_layer_shapes(&self.weights, layer, vision)?;
        }
        validate_merger_shapes(&self.weights, "model.visual.merger", vision, false)?;
        for index in 0..vision.deepstack_visual_indexes.len() {
            validate_merger_shapes(
                &self.weights,
                &format!("model.visual.deepstack_merger_list.{index}"),
                vision,
                true,
            )?;
        }
        expect_shape(
            &self.weights,
            "model.language_model.embed_tokens.weight",
            &[language.vocab_size, language.hidden_size],
        )?;
        for layer in 0..self.target_hidden_state {
            validate_language_layer_shapes(&self.weights, layer, language)?;
        }
        Ok(())
    }

    fn encode_visual_batch(&self, batch: &PreprocessedVision) -> Result<VisionEncoding> {
        let vision = &self.config.vision_config;
        anyhow::ensure!(
            batch.patch_width
                == vision.in_channels
                    * vision.temporal_patch_size
                    * vision.patch_size
                    * vision.patch_size,
            "preprocessed vision patch width does not match Qwen config"
        );
        let expected_rows = batch.grids.iter().try_fold(0usize, |total, grid| {
            total
                .checked_add(grid.patch_count()?)
                .context("vision patch row count overflow")
        })?;
        anyhow::ensure!(
            expected_rows == batch.patch_rows(),
            "preprocessed vision row count does not match its grids"
        );
        let segments = vision_attention_segments(&batch.grids)?;
        if tuned_cuda(&self.device) {
            for &(_, length) in &segments {
                anyhow::ensure!(
                    length <= core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS,
                    "Qwen vision eager CUDA attention segment has {length} rows, exceeding the verified exact softmax range 1..={}; set {}=0 to run it through Candle's kernels instead",
                    core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS,
                    crate::cuda::profile::DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE
                );
            }
        }
        let patches = Tensor::from_slice(
            &batch.patches,
            (expected_rows, batch.patch_width),
            &self.device,
        )?;
        let initial_names = [
            "model.visual.patch_embed.proj.weight".to_owned(),
            "model.visual.patch_embed.proj.bias".to_owned(),
            "model.visual.pos_embed.weight".to_owned(),
        ];
        let mut hidden =
            with_named_group(&self.weights, &initial_names, &self.device, |weights| {
                let weight = required(weights, &initial_names[0])?;
                let patch_bias = required(weights, &initial_names[1])?;
                let hidden =
                    vision_patch_projection(&patches.to_dtype(weight.dtype())?, weight, patch_bias)
                        .context("Qwen vision patch projection")?;

                let table = required(weights, &initial_names[2])?;
                let position = vision_position_embedding(
                    table,
                    &batch.grids,
                    expected_rows,
                    vision.num_position_embeddings,
                    vision.spatial_merge_size,
                )?;
                hidden
                    .add(&position.to_dtype(hidden.dtype())?)
                    .map_err(Into::into)
            })?;

        let (vision_cos, vision_sin) = vision_rotary_tables(
            &batch.grids,
            vision.spatial_merge_size,
            vision.hidden_size / vision.num_heads,
            &self.device,
        )?;
        let mut deepstack = Vec::with_capacity(vision.deepstack_visual_indexes.len());
        for layer in 0..vision.depth {
            let prefix = format!("model.visual.blocks.{layer}");
            let attention_names = [
                format!("{prefix}.norm1.weight"),
                format!("{prefix}.norm1.bias"),
                format!("{prefix}.attn.qkv.weight"),
                format!("{prefix}.attn.qkv.bias"),
                format!("{prefix}.attn.proj.weight"),
                format!("{prefix}.attn.proj.bias"),
            ];
            let mlp_names = [
                format!("{prefix}.norm2.weight"),
                format!("{prefix}.norm2.bias"),
                format!("{prefix}.mlp.linear_fc1.weight"),
                format!("{prefix}.mlp.linear_fc1.bias"),
                format!("{prefix}.mlp.linear_fc2.weight"),
                format!("{prefix}.mlp.linear_fc2.bias"),
            ];
            hidden = with_named_group(&self.weights, &attention_names, &self.device, |weights| {
                self.vision_attention(
                    weights,
                    &prefix,
                    &hidden,
                    &vision_cos,
                    &vision_sin,
                    &segments,
                )
            })?;
            hidden = with_named_group(&self.weights, &mlp_names, &self.device, |weights| {
                let normalized = vision_layer_norm(
                    &hidden,
                    required(weights, &mlp_names[0])?,
                    required(weights, &mlp_names[1])?,
                    LAYER_NORM_EPS,
                )?;
                let first = vision_block_gelu(&linear(
                    weights,
                    &mlp_names[2],
                    Some(&mlp_names[3]),
                    &normalized,
                )?)?;
                let output = linear(weights, &mlp_names[4], Some(&mlp_names[5]), &first)?;
                hidden.add(&output).map_err(Into::into)
            })?;

            if let Some(deep_index) = vision
                .deepstack_visual_indexes
                .iter()
                .position(|&value| value == layer)
            {
                let prefix = format!("model.visual.deepstack_merger_list.{deep_index}");
                deepstack.push(self.merge_vision(&hidden, &prefix, true)?);
            }
        }
        let merged = self.merge_vision(&hidden, "model.visual.merger", false)?;
        Ok(VisionEncoding { merged, deepstack })
    }

    fn vision_attention(
        &self,
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        segments: &[(usize, usize)],
    ) -> Result<Tensor> {
        let vision = &self.config.vision_config;
        let normalized = vision_layer_norm(
            hidden,
            required(weights, &format!("{prefix}.norm1.weight"))?,
            required(weights, &format!("{prefix}.norm1.bias"))?,
            LAYER_NORM_EPS,
        )?;
        let sequence = normalized.dim(0)?;
        let head_dim = vision.hidden_size / vision.num_heads;
        let qkv = linear(
            weights,
            &format!("{prefix}.attn.qkv.weight"),
            Some(&format!("{prefix}.attn.qkv.bias")),
            &normalized,
        )?
        .reshape((sequence, 3, vision.num_heads, head_dim))?;
        let query = qkv.narrow(1, 0, 1)?.squeeze(1)?;
        let key = qkv.narrow(1, 1, 1)?.squeeze(1)?;
        let value = qkv.narrow(1, 2, 1)?.squeeze(1)?;
        let query = apply_vision_rope(&query, cos, sin)?;
        let key = apply_vision_rope(&key, cos, sin)?;

        let mut attended_segments = Vec::with_capacity(segments.len());
        for &(start, length) in segments {
            let query = query.narrow(0, start, length)?.transpose(0, 1)?;
            let key = key.narrow(0, start, length)?.transpose(0, 1)?;
            let value = value.narrow(0, start, length)?.transpose(0, 1)?;
            #[cfg(feature = "cuda")]
            if reference_cuda(query.device())
                && length == 4_032
                && self.attention_query_chunk_size == 256
            {
                let mut chunks = Vec::with_capacity(length.div_ceil(256));
                for query_start in (0..length).step_by(256) {
                    let rows = 256.min(length - query_start);
                    let query = query.narrow(1, query_start, rows)?;
                    let scores_raw = crate::cuda::qwen::attention::qk_matmul(&query, &key)?;
                    let scores = crate::cuda::qwen::attention::scale_scores(&scores_raw, head_dim)?;
                    let probabilities = core::qwen_softmax_last_dim(&scores)?;
                    chunks.push(crate::cuda::qwen::attention::pv_matmul(
                        &probabilities,
                        &value,
                    )?);
                }
                let chunks = chunks.iter().collect::<Vec<_>>();
                attended_segments.push(Tensor::cat(&chunks, 1)?.transpose(0, 1)?);
                continue;
            }
            let query = query.contiguous()?;
            let key = key.contiguous()?;
            let value = value.contiguous()?;
            let key_t = key.transpose(1, 2)?.contiguous()?;
            let mut attended_chunks =
                Vec::with_capacity(length.div_ceil(self.attention_query_chunk_size));
            for query_start in (0..length).step_by(self.attention_query_chunk_size) {
                let query_length = self.attention_query_chunk_size.min(length - query_start);
                let query_chunk = query.narrow(1, query_start, query_length)?.contiguous()?;
                let scores = query_chunk
                    .matmul(&key_t)?
                    .affine(1. / (head_dim as f64).sqrt(), 0.)?;
                let probabilities = core::qwen_softmax_last_dim(&scores)?;
                attended_chunks.push(probabilities.matmul(&value)?);
            }
            let chunks = attended_chunks.iter().collect::<Vec<_>>();
            attended_segments.push(Tensor::cat(&chunks, 1)?.transpose(0, 1)?);
        }
        let segments = attended_segments.iter().collect::<Vec<_>>();
        let attended = Tensor::cat(&segments, 0)?
            .contiguous()?
            .reshape((sequence, vision.hidden_size))?;
        let output = linear(
            weights,
            &format!("{prefix}.attn.proj.weight"),
            Some(&format!("{prefix}.attn.proj.bias")),
            &attended,
        )?;
        hidden.add(&output).map_err(Into::into)
    }

    fn merge_vision(
        &self,
        hidden: &Tensor,
        prefix: &str,
        postshuffle_norm: bool,
    ) -> Result<Tensor> {
        let names = [
            format!("{prefix}.norm.weight"),
            format!("{prefix}.norm.bias"),
            format!("{prefix}.linear_fc1.weight"),
            format!("{prefix}.linear_fc1.bias"),
            format!("{prefix}.linear_fc2.weight"),
            format!("{prefix}.linear_fc2.bias"),
        ];
        let vision = &self.config.vision_config;
        let merge_unit = vision.spatial_merge_size * vision.spatial_merge_size;
        let sequence = hidden.dim(0)?;
        anyhow::ensure!(
            sequence.is_multiple_of(merge_unit),
            "vision sequence is not divisible by the spatial merge unit"
        );
        with_named_group(&self.weights, &names, &self.device, |weights| {
            let merged_hidden = vision.hidden_size * merge_unit;
            let normalized = if postshuffle_norm {
                let reshaped = hidden.reshape((sequence / merge_unit, merged_hidden))?;
                vision_layer_norm(
                    &reshaped,
                    required(weights, &names[0])?,
                    required(weights, &names[1])?,
                    LAYER_NORM_EPS,
                )?
            } else {
                vision_layer_norm(
                    hidden,
                    required(weights, &names[0])?,
                    required(weights, &names[1])?,
                    LAYER_NORM_EPS,
                )?
                .reshape((sequence / merge_unit, merged_hidden))?
            };
            let first =
                vision_merger_gelu(&linear(weights, &names[2], Some(&names[3]), &normalized)?)?;
            linear(weights, &names[4], Some(&names[5]), &first)
        })
    }

    fn encode_language(
        &self,
        token_ids: &[u32],
        mm_types: &[u8],
        position_ids: &[[usize; 3]],
        images: Option<&VisionEncoding>,
        videos: Option<&VisionEncoding>,
    ) -> Result<Tensor> {
        let language = &self.config.text_config;
        let sequence = token_ids.len();
        anyhow::ensure!(
            mm_types.len() == sequence && position_ids.len() == sequence,
            "Qwen presentation metadata does not match its token sequence"
        );
        let mut hidden = self
            .weights
            .load_rows(
                "model.language_model.embed_tokens.weight",
                token_ids,
                &self.device,
            )?
            .unsqueeze(0)?;
        let ordered = order_visual_features(mm_types, images, videos, sequence, &self.device)?;
        if let Some(ordered) = &ordered {
            hidden = replace_sequence_rows(&hidden, &ordered.merged, &ordered.selection)?;
        }

        let (cos, sin) = multimodal_rotary_tables(
            position_ids,
            language.head_dim,
            language.rope_theta,
            language.rope_scaling.mrope_section,
            hidden.dtype(),
            &self.device,
        )?;
        let causal_positions = Tensor::arange(
            0u32,
            u32::try_from(sequence).context("Qwen sequence exceeds U32 indexing")?,
            &self.device,
        )?;

        for layer in 0..self.target_hidden_state {
            let prefix = format!("model.language_model.layers.{layer}");
            let attention_names = [
                format!("{prefix}.input_layernorm.weight"),
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
                format!("{prefix}.self_attn.o_proj.weight"),
                format!("{prefix}.self_attn.q_norm.weight"),
                format!("{prefix}.self_attn.k_norm.weight"),
            ];
            let gate_names = [
                format!("{prefix}.post_attention_layernorm.weight"),
                format!("{prefix}.mlp.gate_proj.weight"),
            ];
            hidden = with_named_group(&self.weights, &attention_names, &self.device, |weights| {
                self.language_attention(weights, &prefix, &hidden, &cos, &sin, &causal_positions)
            })?;

            let up_names = [format!("{prefix}.mlp.up_proj.weight")];
            let (normalized, gate) =
                with_named_group(&self.weights, &gate_names, &self.device, |weights| {
                    let normalized = core::qwen_rms_norm(
                        &hidden,
                        required(weights, &gate_names[0])?,
                        language.rms_norm_eps,
                    )?;
                    let gate = core::silu_with_reference_rounding(&linear(
                        weights,
                        &gate_names[1],
                        None,
                        &normalized,
                    )?)?;
                    Ok((normalized, gate))
                })?;
            let down_names = [format!("{prefix}.mlp.down_proj.weight")];
            let activated = with_named_group(&self.weights, &up_names, &self.device, |weights| {
                gate.mul(&linear(weights, &up_names[0], None, &normalized)?)
                    .map_err(Into::into)
            })?;
            let mlp = with_named_group(&self.weights, &down_names, &self.device, |weights| {
                linear(weights, &down_names[0], None, &activated)
            })?;
            hidden = hidden.add(&mlp)?;

            if let Some(ordered) = &ordered
                && let Some(deepstack) = ordered.deepstack.get(layer)
            {
                let current = hidden
                    .squeeze(0)?
                    .index_select(&ordered.visual_positions, 0)?;
                let injected = current.add(&deepstack.to_dtype(current.dtype())?)?;
                hidden = replace_sequence_rows(&hidden, &injected, &ordered.selection)?;
            }
        }
        Ok(hidden)
    }

    fn language_attention(
        &self,
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        causal_positions: &Tensor,
    ) -> Result<Tensor> {
        let language = &self.config.text_config;
        let normalized = core::qwen_rms_norm(
            hidden,
            required(weights, &format!("{prefix}.input_layernorm.weight"))?,
            language.rms_norm_eps,
        )?;
        let (batch, sequence, _) = normalized.dims3()?;
        let query = linear(
            weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            None,
            &normalized,
        )?
        .reshape((
            batch,
            sequence,
            language.num_attention_heads,
            language.head_dim,
        ))?;
        let key = linear(
            weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            None,
            &normalized,
        )?
        .reshape((
            batch,
            sequence,
            language.num_key_value_heads,
            language.head_dim,
        ))?;
        let value = linear(
            weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            None,
            &normalized,
        )?
        .reshape((
            batch,
            sequence,
            language.num_key_value_heads,
            language.head_dim,
        ))?;
        let query = language_head_rms_norm(
            &query,
            required(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
            language.rms_norm_eps,
        )?;
        let key = language_head_rms_norm(
            &key,
            required(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
            language.rms_norm_eps,
        )?;
        let query = apply_language_rope(&query, cos, sin)?.transpose(1, 2)?;
        let key = apply_language_rope(&key, cos, sin)?.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let groups = language.num_attention_heads / language.num_key_value_heads;
        let key = repeat_kv(&key, groups)?.contiguous()?;
        let value = repeat_kv(&value, groups)?.contiguous()?;
        #[cfg(feature = "cuda")]
        if let (true, Some(query_rows)) = (
            reference_cuda(query.device()),
            match (sequence, self.attention_query_chunk_size) {
                (357, 357) => Some(357),
                (1_935, 256) => Some(256),
                (8_620, 256) => Some(256),
                _ => None,
            },
        ) {
            let mut attended_chunks = Vec::with_capacity(sequence.div_ceil(query_rows));
            for start in (0..sequence).step_by(query_rows) {
                let rows = query_rows.min(sequence - start);
                let query = query.narrow(2, start, rows)?;
                let scores_raw = crate::cuda::qwen::attention::qk_matmul(&query, &key)?;
                let scores =
                    crate::cuda::qwen::attention::scale_scores(&scores_raw, language.head_dim)?;
                let scores = apply_causal_mask(&scores, causal_positions, start, rows)?;
                let probabilities = core::qwen_softmax_last_dim(&scores)?;
                attended_chunks.push(crate::cuda::qwen::attention::pv_matmul(
                    &probabilities,
                    &value,
                )?);
            }
            let attended_chunks = attended_chunks.iter().collect::<Vec<_>>();
            let attended = Tensor::cat(&attended_chunks, 2)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((
                    batch,
                    sequence,
                    language.num_attention_heads * language.head_dim,
                ))?;
            let output = linear(
                weights,
                &format!("{prefix}.self_attn.o_proj.weight"),
                None,
                &attended,
            )?;
            return hidden.add(&output).map_err(Into::into);
        }
        let query = query.contiguous()?;
        let key_t = key.transpose(2, 3)?.contiguous()?;
        let mut chunks = Vec::with_capacity(sequence.div_ceil(self.attention_query_chunk_size));
        for start in (0..sequence).step_by(self.attention_query_chunk_size) {
            let length = self.attention_query_chunk_size.min(sequence - start);
            let query_chunk = query.narrow(2, start, length)?.contiguous()?;
            let scores = query_chunk
                .matmul(&key_t)?
                .affine(1. / (language.head_dim as f64).sqrt(), 0.)?;
            let scores = apply_causal_mask(&scores, causal_positions, start, length)?;
            let probabilities = core::qwen_softmax_last_dim(&scores)?;
            chunks.push(probabilities.matmul(&value)?);
        }
        let chunks = chunks.iter().collect::<Vec<_>>();
        let attended = Tensor::cat(&chunks, 2)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((
                batch,
                sequence,
                language.num_attention_heads * language.head_dim,
            ))?;
        let output = linear(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            None,
            &attended,
        )?;
        hidden.add(&output).map_err(Into::into)
    }
}

struct OrderedVision {
    merged: Tensor,
    deepstack: Vec<Tensor>,
    visual_positions: Tensor,
    selection: Tensor,
}

fn order_visual_features(
    mm_types: &[u8],
    images: Option<&VisionEncoding>,
    videos: Option<&VisionEncoding>,
    sequence: usize,
    device: &Device,
) -> Result<Option<OrderedVision>> {
    let image_rows = images
        .map(|encoding| encoding.merged.dim(0))
        .transpose()?
        .unwrap_or(0);
    let video_rows = videos
        .map(|encoding| encoding.merged.dim(0))
        .transpose()?
        .unwrap_or(0);
    let visual_rows = image_rows
        .checked_add(video_rows)
        .context("visual feature row count overflow")?;
    if visual_rows == 0 {
        anyhow::ensure!(
            mm_types.iter().all(|&kind| kind == 0),
            "presentation has visual pad tokens but no visual features"
        );
        return Ok(None);
    }

    let image_depth = images.map_or(0, |encoding| encoding.deepstack.len());
    let video_depth = videos.map_or(0, |encoding| encoding.deepstack.len());
    anyhow::ensure!(
        image_depth == 0 || video_depth == 0 || image_depth == video_depth,
        "image and video DeepStack depths differ"
    );
    let source_merged = concatenate_feature_sources(
        images.map(|encoding| &encoding.merged),
        videos.map(|encoding| &encoding.merged),
    )?
    .context("visual feature sources are empty")?;

    let mut image_cursor = 0usize;
    let mut video_cursor = 0usize;
    let mut source_indices = Vec::with_capacity(visual_rows);
    let mut visual_positions = Vec::with_capacity(visual_rows);
    let mut selection = Vec::with_capacity(sequence);
    for (position, &kind) in mm_types.iter().enumerate() {
        match kind {
            0 => selection.push(u32::try_from(position).context("sequence index exceeds u32")?),
            1 => {
                anyhow::ensure!(image_cursor < image_rows, "too many image pad tokens");
                source_indices
                    .push(u32::try_from(image_cursor).context("image feature index exceeds u32")?);
                visual_positions
                    .push(u32::try_from(position).context("visual position exceeds u32")?);
                selection.push(
                    u32::try_from(sequence + source_indices.len() - 1)
                        .context("replacement row index exceeds u32")?,
                );
                image_cursor += 1;
            }
            2 => {
                anyhow::ensure!(video_cursor < video_rows, "too many video pad tokens");
                source_indices.push(
                    u32::try_from(image_rows + video_cursor)
                        .context("video feature index exceeds u32")?,
                );
                visual_positions
                    .push(u32::try_from(position).context("visual position exceeds u32")?);
                selection.push(
                    u32::try_from(sequence + source_indices.len() - 1)
                        .context("replacement row index exceeds u32")?,
                );
                video_cursor += 1;
            }
            _ => bail!("unsupported Qwen multimodal token type {kind}"),
        }
    }
    anyhow::ensure!(
        image_cursor == image_rows && video_cursor == video_rows,
        "visual feature rows do not match presentation pad tokens"
    );
    let source_indices = Tensor::from_vec(source_indices, visual_rows, device)?;
    let merged = source_merged.index_select(&source_indices, 0)?;

    let deepstack_depth = image_depth.max(video_depth);
    let mut deepstack = Vec::with_capacity(deepstack_depth);
    for index in 0..deepstack_depth {
        let source = concatenate_feature_sources(
            images.and_then(|encoding| encoding.deepstack.get(index)),
            videos.and_then(|encoding| encoding.deepstack.get(index)),
        )?
        .with_context(|| format!("missing DeepStack source {index}"))?;
        deepstack.push(source.index_select(&source_indices, 0)?);
    }
    Ok(Some(OrderedVision {
        merged,
        deepstack,
        visual_positions: Tensor::from_vec(visual_positions, visual_rows, device)?,
        selection: Tensor::from_vec(selection, sequence, device)?,
    }))
}

fn concatenate_feature_sources(
    images: Option<&Tensor>,
    videos: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    match (images, videos) {
        (Some(images), Some(videos)) => Ok(Some(Tensor::cat(&[images, videos], 0)?)),
        (Some(images), None) => Ok(Some(images.clone())),
        (None, Some(videos)) => Ok(Some(videos.clone())),
        (None, None) => Ok(None),
    }
}

fn replace_sequence_rows(
    hidden: &Tensor,
    replacements: &Tensor,
    selection: &Tensor,
) -> Result<Tensor> {
    let hidden = hidden
        .squeeze(0)
        .context("Qwen hidden states must have batch size one")?;
    let rows = Tensor::cat(&[&hidden, replacements], 0)?;
    rows.index_select(selection, 0)?
        .unsqueeze(0)
        .map_err(Into::into)
}

fn preprocess_sequences(
    modality: VisionModality,
    sequences: &[Vec<RgbImage>],
    processor: &ProcessorConfig,
) -> Result<PreprocessedVision> {
    anyhow::ensure!(processor.patch_size > 0, "Qwen patch size must be non-zero");
    anyhow::ensure!(
        processor.temporal_patch_size > 0 && processor.merge_size > 0,
        "Qwen temporal patch and merge sizes must be non-zero"
    );
    let factor = processor
        .patch_size
        .checked_mul(processor.merge_size)
        .context("Qwen resize factor overflow")?;
    let patch_width = 3usize
        .checked_mul(processor.temporal_patch_size)
        .and_then(|value| value.checked_mul(processor.patch_size))
        .and_then(|value| value.checked_mul(processor.patch_size))
        .context("Qwen flattened patch width overflow")?;
    let mut patches = Vec::new();
    let mut grids = Vec::with_capacity(sequences.len());
    for frames in sequences {
        anyhow::ensure!(!frames.is_empty(), "vision sequence must not be empty");
        let source_height = frames[0].height;
        let source_width = frames[0].width;
        anyhow::ensure!(
            frames
                .iter()
                .all(|frame| frame.height == source_height && frame.width == source_width),
            "all frames in a vision sequence must have equal dimensions"
        );
        let (height, width) = match modality {
            VisionModality::Image => smart_resize_image(
                source_height,
                source_width,
                factor,
                processor.size.shortest_edge,
                processor.size.longest_edge,
            )?,
            VisionModality::Video => smart_resize_video(
                frames.len(),
                source_height,
                source_width,
                processor.temporal_patch_size,
                factor,
                processor.size.shortest_edge,
                processor.size.longest_edge,
            )?,
        };
        let mut resized = frames
            .iter()
            .map(|frame| resize_bicubic_antialias(frame, width, height))
            .collect::<Result<Vec<_>>>()?;
        let padding = (processor.temporal_patch_size
            - resized.len() % processor.temporal_patch_size)
            % processor.temporal_patch_size;
        if padding > 0 {
            let last = resized
                .last()
                .context("cannot pad an empty vision sequence")?
                .clone();
            resized.extend(std::iter::repeat_n(last, padding));
        }
        let grid = VisionGrid {
            temporal: resized.len() / processor.temporal_patch_size,
            height: height / processor.patch_size,
            width: width / processor.patch_size,
        };
        anyhow::ensure!(
            grid.height.is_multiple_of(processor.merge_size)
                && grid.width.is_multiple_of(processor.merge_size),
            "Qwen resized vision grid is not divisible by its merge size"
        );
        patchify_sequence(&resized, grid, processor, &mut patches)?;
        grids.push(grid);
    }
    let expected_rows = grids.iter().try_fold(0usize, |total, grid| {
        total
            .checked_add(grid.patch_count()?)
            .context("preprocessed patch row count overflow")
    })?;
    anyhow::ensure!(
        patches.len() == expected_rows * patch_width,
        "internal Qwen patchification size mismatch"
    );
    Ok(PreprocessedVision {
        modality,
        grids,
        patches,
        patch_width,
    })
}

fn patchify_sequence(
    frames: &[RgbImage],
    grid: VisionGrid,
    processor: &ProcessorConfig,
    output: &mut Vec<f32>,
) -> Result<()> {
    let patch = processor.patch_size;
    let merge = processor.merge_size;
    let temporal = processor.temporal_patch_size;
    anyhow::ensure!(
        frames.len() == grid.temporal * temporal,
        "vision frame count does not match temporal grid"
    );
    let height = grid.height * patch;
    let width = grid.width * patch;
    anyhow::ensure!(
        frames
            .iter()
            .all(|frame| frame.height == height && frame.width == width),
        "vision frame dimensions do not match patch grid"
    );
    for grid_t in 0..grid.temporal {
        for block_h in 0..grid.height / merge {
            for block_w in 0..grid.width / merge {
                for in_block_h in 0..merge {
                    for in_block_w in 0..merge {
                        let patch_h = block_h * merge + in_block_h;
                        let patch_w = block_w * merge + in_block_w;
                        for channel in 0..3 {
                            for temporal_offset in 0..temporal {
                                let frame = &frames[grid_t * temporal + temporal_offset];
                                for row in 0..patch {
                                    let source_row = patch_h * patch + row;
                                    for column in 0..patch {
                                        let source_column = patch_w * patch + column;
                                        let byte = frame.pixels
                                            [(source_row * width + source_column) * 3 + channel];
                                        output.push(
                                            (byte as f32 / 255. - processor.image_mean[channel])
                                                / processor.image_std[channel],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn smart_resize_image(
    height: usize,
    width: usize,
    factor: usize,
    minimum_pixels: usize,
    maximum_pixels: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(
        height >= factor && width >= factor,
        "Qwen image dimensions must each be at least the resize factor {factor}"
    );
    anyhow::ensure!(
        height.max(width) as f64 / height.min(width) as f64 <= 200.,
        "Qwen image aspect ratio must not exceed 200:1"
    );
    let mut resized_height = round_ties_even(height as f64 / factor as f64) as usize * factor;
    let mut resized_width = round_ties_even(width as f64 / factor as f64) as usize * factor;
    let rounded_area = resized_height
        .checked_mul(resized_width)
        .context("Qwen image resize area overflow")?;
    if rounded_area > maximum_pixels {
        let beta = ((height * width) as f64 / maximum_pixels as f64).sqrt();
        resized_height = (height as f64 / beta / factor as f64).floor() as usize * factor;
        resized_width = (width as f64 / beta / factor as f64).floor() as usize * factor;
    } else if rounded_area < minimum_pixels {
        let beta = (minimum_pixels as f64 / (height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    anyhow::ensure!(
        resized_height > 0 && resized_width > 0,
        "Qwen image resize produced a zero dimension"
    );
    Ok((resized_height, resized_width))
}

fn smart_resize_video(
    frames: usize,
    mut height: usize,
    mut width: usize,
    temporal_factor: usize,
    factor: usize,
    minimum_pixels: usize,
    maximum_pixels: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(
        frames >= temporal_factor,
        "Qwen video must contain at least {temporal_factor} sampled frames"
    );
    if height < factor || width < factor {
        let scale = (factor as f64 / height as f64).max(factor as f64 / width as f64);
        height = (height as f64 * scale) as usize;
        width = (width as f64 * scale) as usize;
    }
    anyhow::ensure!(
        height.max(width) as f64 / height.min(width) as f64 <= 200.,
        "Qwen video aspect ratio must not exceed 200:1"
    );
    let mut resized_height = round_ties_even(height as f64 / factor as f64) as usize * factor;
    let mut resized_width = round_ties_even(width as f64 / factor as f64) as usize * factor;
    let rounded_frames =
        round_ties_even(frames as f64 / temporal_factor as f64) as usize * temporal_factor;
    let rounded_volume = rounded_frames
        .checked_mul(resized_height)
        .and_then(|value| value.checked_mul(resized_width))
        .context("Qwen video resize volume overflow")?;
    if rounded_volume > maximum_pixels {
        let beta = ((frames * height * width) as f64 / maximum_pixels as f64).sqrt();
        resized_height =
            factor.max((height as f64 / beta / factor as f64).floor() as usize * factor);
        resized_width = factor.max((width as f64 / beta / factor as f64).floor() as usize * factor);
    } else if rounded_volume < minimum_pixels {
        let beta = (minimum_pixels as f64 / (frames * height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

#[derive(Clone)]
struct AxisSample {
    taps: Vec<(usize, f32)>,
}

fn resize_bicubic_antialias(image: &RgbImage, width: usize, height: usize) -> Result<RgbImage> {
    anyhow::ensure!(width > 0 && height > 0, "resize target must be non-zero");
    if image.width == width && image.height == height {
        return Ok(image.clone());
    }
    let horizontal = resize_axis_samples(image.width, width);
    let vertical = resize_axis_samples(image.height, height);
    let horizontal_len = image
        .height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("horizontal resize buffer overflow")?;
    let mut intermediate = vec![0f32; horizontal_len];
    for row in 0..image.height {
        for (column, sample) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let value = sample
                    .taps
                    .iter()
                    .map(|&(source, weight)| {
                        image.pixels[(row * image.width + source) * 3 + channel] as f32 * weight
                    })
                    .sum();
                intermediate[(row * width + column) * 3 + channel] = value;
            }
        }
    }
    let output_len = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("resize output buffer overflow")?;
    let mut pixels = vec![0u8; output_len];
    for (row, sample) in vertical.iter().enumerate() {
        for column in 0..width {
            for channel in 0..3 {
                let value: f32 = sample
                    .taps
                    .iter()
                    .map(|&(source, weight)| {
                        intermediate[(source * width + column) * 3 + channel] * weight
                    })
                    .sum();
                pixels[(row * width + column) * 3 + channel] = value.round().clamp(0., 255.) as u8;
            }
        }
    }
    RgbImage::new(width, height, pixels)
}

fn resize_axis_samples(source: usize, target: usize) -> Vec<AxisSample> {
    let scale = (source as f64 / target as f64).max(1.);
    let support = 2. * scale;
    (0..target)
        .map(|output| {
            let center = (output as f64 + 0.5) * source as f64 / target as f64 - 0.5;
            let first = (center - support).ceil() as isize;
            let last = (center + support).floor() as isize;
            let mut accumulated = BTreeMap::<usize, f64>::new();
            for input in first..=last {
                let clamped = input.clamp(0, source as isize - 1) as usize;
                let weight = cubic_kernel((center - input as f64) / scale);
                *accumulated.entry(clamped).or_default() += weight;
            }
            let total: f64 = accumulated.values().sum();
            let taps = accumulated
                .into_iter()
                .map(|(index, weight)| (index, (weight / total) as f32))
                .collect();
            AxisSample { taps }
        })
        .collect()
}

fn cubic_kernel(distance: f64) -> f64 {
    let distance = distance.abs();
    let coefficient = -0.75;
    if distance <= 1. {
        (coefficient + 2.) * distance.powi(3) - (coefficient + 3.) * distance.powi(2) + 1.
    } else if distance < 2. {
        coefficient * distance.powi(3) - 5. * coefficient * distance.powi(2)
            + 8. * coefficient * distance
            - 4. * coefficient
    } else {
        0.
    }
}

fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.
    }
}

fn learned_position_interpolation(
    grids: &[VisionGrid],
    source_side: usize,
    merge: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let rows = grids.iter().try_fold(0usize, |total, grid| {
        total
            .checked_add(grid.patch_count()?)
            .context("position interpolation row count overflow")
    })?;
    let mut indices = Vec::with_capacity(rows * 4);
    let mut weights = Vec::with_capacity(rows * 4);
    for grid in grids {
        anyhow::ensure!(
            grid.height.is_multiple_of(merge) && grid.width.is_multiple_of(merge),
            "position interpolation grid is not merge-aligned"
        );
        for _ in 0..grid.temporal {
            for block_h in 0..grid.height / merge {
                for block_w in 0..grid.width / merge {
                    for in_h in 0..merge {
                        for in_w in 0..merge {
                            let row = block_h * merge + in_h;
                            let column = block_w * merge + in_w;
                            let (h0, h1, hw0, hw1) = bilinear_axis(row, grid.height, source_side);
                            let (w0, w1, ww0, ww1) = bilinear_axis(column, grid.width, source_side);
                            for (source_h, height_weight, source_w, width_weight) in [
                                (h0, hw0, w0, ww0),
                                (h0, hw0, w1, ww1),
                                (h1, hw1, w0, ww0),
                                (h1, hw1, w1, ww1),
                            ] {
                                indices.push(
                                    u32::try_from(source_h * source_side + source_w)
                                        .context("learned position index exceeds u32")?,
                                );
                                weights.push(height_weight * width_weight);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((indices, weights))
}

fn bilinear_axis(index: usize, target_size: usize, source_size: usize) -> (usize, usize, f32, f32) {
    if target_size <= 1 {
        return (0, 0, 1., 0.);
    }
    let source = index as f64 * (source_size - 1) as f64 / (target_size - 1) as f64;
    let lower = source.floor() as usize;
    let upper = (lower + 1).min(source_size - 1);
    let upper_weight = (source - lower as f64) as f32;
    (lower, upper, 1. - upper_weight, upper_weight)
}

fn vision_attention_segments(grids: &[VisionGrid]) -> Result<Vec<(usize, usize)>> {
    let mut start = 0usize;
    let mut segments = Vec::new();
    for grid in grids {
        let length = grid
            .height
            .checked_mul(grid.width)
            .context("vision attention segment length overflow")?;
        anyhow::ensure!(length > 0, "vision attention segment must not be empty");
        for _ in 0..grid.temporal {
            segments.push((start, length));
            start = start
                .checked_add(length)
                .context("vision attention sequence length overflow")?;
        }
    }
    Ok(segments)
}

fn validate_qwen_exact_attention_geometry(
    cuda: bool,
    language_rows: usize,
    image_grids: &[VisionGrid],
    video_grids: &[VisionGrid],
) -> Result<()> {
    if !cuda {
        return Ok(());
    }
    anyhow::ensure!(
        (1..=core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS).contains(&language_rows),
        "Qwen language eager attention has {language_rows} rows, outside the verified exact softmax range 1..={} before embedding or vision payload access",
        core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS
    );
    for grids in [image_grids, video_grids] {
        let total_patch_rows = grids.iter().try_fold(0usize, |total, grid| {
            total
                .checked_add(grid.patch_count()?)
                .context("Qwen vision patch row count overflow")
        })?;
        if total_patch_rows != 0 {
            anyhow::ensure!(
                matches!(
                    total_patch_rows,
                    QWEN_VERIFIED_FL_PATCH_ROWS | QWEN_VERIFIED_REF_PATCH_ROWS
                ),
                "Qwen CUDA vision patch rows {total_patch_rows} are outside the verified real FL/Ref profiles {}|{}; set {}=0 to run this geometry through Candle's kernels instead",
                QWEN_VERIFIED_FL_PATCH_ROWS,
                QWEN_VERIFIED_REF_PATCH_ROWS,
                crate::cuda::profile::DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE
            );
            let merger_rows = total_patch_rows / 4;
            anyhow::ensure!(
                matches!(
                    merger_rows,
                    QWEN_VERIFIED_FL_MERGER_ROWS | QWEN_VERIFIED_REF_MERGER_ROWS
                ),
                "Qwen CUDA merger rows {merger_rows} are outside the verified real FL/Ref profiles"
            );
        }
        for (_, length) in vision_attention_segments(grids)? {
            anyhow::ensure!(
                length <= core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS,
                "Qwen vision eager attention segment has {length} rows, exceeding the verified exact softmax range 1..={} before embedding or vision payload access",
                core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS
            );
        }
    }
    Ok(())
}

fn canonical_ordered_vision_grids(
    mm_types: &[u8],
    image_grids: &[VisionGrid],
    video_grids: &[VisionGrid],
    merge: usize,
) -> Result<Vec<(H3QwenVisionGridModality, usize, usize, usize)>> {
    let merge_area = merge
        .checked_mul(merge)
        .filter(|area| *area > 0)
        .context("Qwen vision merge area is zero or overflows")?;
    let mut ordered = Vec::with_capacity(image_grids.len() + video_grids.len());
    let mut image_cursor = 0usize;
    let mut video_cursor = 0usize;
    let mut active_video_temporal_remaining = 0usize;
    let mut cursor = 0usize;
    while cursor < mm_types.len() {
        let modality = mm_types[cursor];
        if !matches!(modality, 1 | 2) {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < mm_types.len() && mm_types[cursor] == modality {
            cursor += 1;
        }
        let run_rows = cursor - start;
        if modality == 1 {
            anyhow::ensure!(
                active_video_temporal_remaining == 0,
                "Qwen image pad run interrupts a video grid presentation"
            );
            let grid = image_grids
                .get(image_cursor)
                .context("Qwen presentation has more image pad runs than image grids")?;
            anyhow::ensure!(
                grid.temporal == 1,
                "Qwen image grid temporal size must be one"
            );
            let expected_run = grid
                .height
                .checked_mul(grid.width)
                .and_then(|rows| rows.checked_div(merge_area))
                .context("Qwen image merger run size overflow")?;
            anyhow::ensure!(
                run_rows == expected_run,
                "Qwen image pad run disagrees with its grid"
            );
            ordered.push((
                H3QwenVisionGridModality::Image,
                grid.temporal,
                grid.height,
                grid.width,
            ));
            image_cursor += 1;
        } else {
            let grid = video_grids
                .get(video_cursor)
                .context("Qwen presentation has more video pad runs than video grids")?;
            if active_video_temporal_remaining == 0 {
                active_video_temporal_remaining = grid.temporal;
                ordered.push((
                    H3QwenVisionGridModality::Video,
                    grid.temporal,
                    grid.height,
                    grid.width,
                ));
            }
            let expected_run = grid
                .height
                .checked_mul(grid.width)
                .and_then(|rows| rows.checked_div(merge_area))
                .context("Qwen video merger run size overflow")?;
            anyhow::ensure!(
                run_rows == expected_run,
                "Qwen video pad run disagrees with its grid"
            );
            active_video_temporal_remaining = active_video_temporal_remaining
                .checked_sub(1)
                .context("Qwen video pad runs exceed grid temporal size")?;
            if active_video_temporal_remaining == 0 {
                video_cursor += 1;
            }
        }
    }
    anyhow::ensure!(
        image_cursor == image_grids.len()
            && video_cursor == video_grids.len()
            && active_video_temporal_remaining == 0,
        "not every Qwen vision grid has a complete ordered pad-run presentation"
    );
    Ok(ordered)
}

pub(crate) fn vision_rotary_tables(
    grids: &[VisionGrid],
    merge: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        head_dim.is_multiple_of(4),
        "Qwen vision head dimension must be divisible by four"
    );
    let frequencies_per_axis = head_dim / 4;
    let inverse = crate::text_encoder::qwen_inverse_frequencies(head_dim / 2, 10_000., device)?
        .reshape((1, frequencies_per_axis))?;
    let rows = grids.iter().try_fold(0usize, |total, grid| {
        total
            .checked_add(grid.patch_count()?)
            .context("vision rotary row count overflow")
    })?;
    let mut heights = Vec::with_capacity(rows);
    let mut widths = Vec::with_capacity(rows);
    for grid in grids {
        for _ in 0..grid.temporal {
            for block_h in 0..grid.height / merge {
                for block_w in 0..grid.width / merge {
                    for in_h in 0..merge {
                        for in_w in 0..merge {
                            heights.push((block_h * merge + in_h) as f32);
                            widths.push((block_w * merge + in_w) as f32);
                        }
                    }
                }
            }
        }
    }
    let height_frequencies =
        Tensor::from_vec(heights, (rows, 1), device)?.broadcast_mul(&inverse)?;
    let width_frequencies = Tensor::from_vec(widths, (rows, 1), device)?.broadcast_mul(&inverse)?;
    let frequencies = Tensor::cat(&[&height_frequencies, &width_frequencies], 1)?;
    let frequencies = Tensor::cat(&[&frequencies, &frequencies], 1)?;
    Ok((frequencies.cos()?, frequencies.sin()?))
}

fn multimodal_position_ids(
    mm_types: &[u8],
    image_grids: &[VisionGrid],
    video_grids: &[VisionGrid],
    merge: usize,
) -> Result<Vec<[usize; 3]>> {
    let mut video_blocks = Vec::new();
    for grid in video_grids {
        video_blocks.extend(std::iter::repeat_n(
            VisionGrid {
                temporal: 1,
                height: grid.height,
                width: grid.width,
            },
            grid.temporal,
        ));
    }
    let mut positions = Vec::with_capacity(mm_types.len());
    let mut current = 0usize;
    let mut image_cursor = 0usize;
    let mut video_cursor = 0usize;
    let mut start = 0usize;
    while start < mm_types.len() {
        let kind = mm_types[start];
        let mut end = start + 1;
        while end < mm_types.len() && mm_types[end] == kind {
            end += 1;
        }
        if kind == 0 {
            positions.extend((0..end - start).map(|offset| {
                let position = current + offset;
                [position, position, position]
            }));
            current = current
                .checked_add(end - start)
                .context("Qwen text position overflow")?;
        } else {
            let grid = match kind {
                1 => {
                    let grid = image_grids
                        .get(image_cursor)
                        .context("image pad run has no matching image grid")?;
                    image_cursor += 1;
                    *grid
                }
                2 => {
                    let grid = video_blocks
                        .get(video_cursor)
                        .context("video pad run has no matching temporal grid")?;
                    video_cursor += 1;
                    *grid
                }
                _ => bail!("unsupported Qwen multimodal token type {kind}"),
            };
            let expected = grid.merged_count(merge)?;
            anyhow::ensure!(
                end - start == expected,
                "Qwen vision pad run has {} tokens, expected {expected} from its grid",
                end - start
            );
            for temporal in 0..grid.temporal {
                for height in 0..grid.height / merge {
                    for width in 0..grid.width / merge {
                        positions.push([current + temporal, current + height, current + width]);
                    }
                }
            }
            current = current
                .checked_add(grid.height.max(grid.width) / merge)
                .context("Qwen vision position overflow")?;
        }
        start = end;
    }
    anyhow::ensure!(
        image_cursor == image_grids.len(),
        "not every image grid has a presentation pad run"
    );
    anyhow::ensure!(
        video_cursor == video_blocks.len(),
        "not every video temporal grid has a presentation pad run"
    );
    anyhow::ensure!(
        positions.len() == mm_types.len(),
        "Qwen M-RoPE position count mismatch"
    );
    Ok(positions)
}

pub(crate) fn multimodal_rotary_tables(
    positions: &[[usize; 3]],
    head_dim: usize,
    theta: f64,
    sections: [usize; 3],
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    anyhow::ensure!(
        head_dim.is_multiple_of(2) && sections.iter().sum::<usize>() == head_dim / 2,
        "Qwen M-RoPE sections must cover half the attention head"
    );
    let half = head_dim / 2;
    let inverse = crate::text_encoder::qwen_inverse_frequencies(head_dim, theta, device)?
        .reshape((1, half))?;
    let mut by_axis = Vec::with_capacity(3);
    for axis in 0..3 {
        let axis_positions = positions
            .iter()
            .map(|position| position[axis] as f32)
            .collect::<Vec<_>>();
        let axis_positions = Tensor::from_vec(axis_positions, (positions.len(), 1), device)?;
        by_axis.push(axis_positions.matmul(&inverse)?);
    }
    let mut columns = Vec::with_capacity(half);
    for index in 0..half {
        let axis = if index % 3 == 1 && index < sections[1] * 3 {
            1
        } else if index % 3 == 2 && index < sections[2] * 3 {
            2
        } else {
            0
        };
        columns.push(by_axis[axis].narrow(1, index, 1)?);
    }
    let columns = columns.iter().collect::<Vec<_>>();
    let frequencies = Tensor::cat(&columns, 1)?.reshape((1, positions.len(), half))?;
    let frequencies = Tensor::cat(&[&frequencies, &frequencies], 2)?;
    Ok((
        frequencies.cos()?.to_dtype(dtype)?,
        frequencies.sin()?.to_dtype(dtype)?,
    ))
}

pub(crate) fn apply_vision_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let input = input.to_dtype(DType::F32)?;
    let rotated = rotate_half(&input)?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    input
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

pub(crate) fn apply_language_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(input)?;
    let cos = cos.unsqueeze(2)?;
    let sin = sin.unsqueeze(2)?;
    input
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)
        .map_err(Into::into)
}

fn rotate_half(input: &Tensor) -> Result<Tensor> {
    let head_dim = input.dim(candle_core::D::Minus1)?;
    anyhow::ensure!(head_dim.is_multiple_of(2), "rotary dimension must be even");
    let half = head_dim / 2;
    let first = input.narrow(candle_core::D::Minus1, 0, half)?;
    let second = input.narrow(candle_core::D::Minus1, half, half)?;
    Tensor::cat(&[&second.neg()?, &first], candle_core::D::Minus1).map_err(Into::into)
}

fn repeat_kv(input: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(input.clone());
    }
    let (batch, heads, sequence, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, heads, groups, sequence, head_dim))?
        .contiguous()?
        .reshape((batch, heads * groups, sequence, head_dim))
        .map_err(Into::into)
}

fn apply_causal_mask(
    scores: &Tensor,
    positions: &Tensor,
    query_start: usize,
    query_length: usize,
) -> Result<Tensor> {
    let sequence = positions.dim(0)?;
    let keys = positions.reshape((1, 1, 1, sequence))?;
    let queries =
        positions
            .narrow(0, query_start, query_length)?
            .reshape((1, 1, query_length, 1))?;
    let future = keys.broadcast_gt(&queries)?.broadcast_as(scores.shape())?;
    let negative_infinity = Tensor::new(
        core::qwen_attention_mask_minimum(scores.dtype())?,
        scores.device(),
    )?
    .to_dtype(scores.dtype())?
    .broadcast_as(scores.shape())?;
    future
        .where_cond(&negative_infinity, scores)
        .context("failed to apply Qwen causal mask")
}

fn validate_config(
    config: &EncoderConfig,
    image_processor: &ProcessorConfig,
    video_processor: &ProcessorConfig,
) -> Result<()> {
    let vision = &config.vision_config;
    let language = &config.text_config;
    anyhow::ensure!(
        vision.depth > 0
            && vision.hidden_size > 0
            && vision.intermediate_size > 0
            && vision.out_hidden_size > 0,
        "Qwen vision dimensions must be non-zero"
    );
    anyhow::ensure!(
        vision.num_heads > 0 && vision.hidden_size.is_multiple_of(vision.num_heads),
        "Qwen vision hidden size is not divisible by its attention heads"
    );
    anyhow::ensure!(
        (vision.hidden_size / vision.num_heads).is_multiple_of(4),
        "Qwen vision head dimension must be divisible by four"
    );
    anyhow::ensure!(
        vision.in_channels == 3,
        "Qwen vision processor currently requires three RGB channels"
    );
    anyhow::ensure!(
        vision.patch_size > 0 && vision.temporal_patch_size > 0 && vision.spatial_merge_size > 0,
        "Qwen vision patch sizes must be non-zero"
    );
    anyhow::ensure!(
        vision
            .deepstack_visual_indexes
            .windows(2)
            .all(|pair| pair[0] < pair[1])
            && vision
                .deepstack_visual_indexes
                .iter()
                .all(|&index| index < vision.depth),
        "Qwen DeepStack visual indexes must be sorted, unique, and inside the vision tower"
    );
    integer_square_root(vision.num_position_embeddings)?;

    anyhow::ensure!(
        language.hidden_size > 0
            && language.intermediate_size > 0
            && language.head_dim > 0
            && language.vocab_size > 0,
        "Qwen language dimensions must be non-zero"
    );
    let mut special_ids = [
        config.image_token_id,
        config.video_token_id,
        config.vision_start_token_id,
        config.vision_end_token_id,
    ];
    anyhow::ensure!(
        special_ids
            .iter()
            .all(|id| (*id as usize) < language.vocab_size),
        "Qwen multimodal special token ID exceeds the language vocabulary"
    );
    special_ids.sort_unstable();
    anyhow::ensure!(
        special_ids.windows(2).all(|pair| pair[0] != pair[1]),
        "Qwen multimodal special token IDs must be distinct"
    );
    anyhow::ensure!(
        language.num_attention_heads > 0
            && language.num_key_value_heads > 0
            && language
                .num_attention_heads
                .is_multiple_of(language.num_key_value_heads),
        "Qwen language attention heads are incompatible with KV heads"
    );
    let mrope_width =
        language
            .rope_scaling
            .mrope_section
            .iter()
            .try_fold(0usize, |total, width| {
                total
                    .checked_add(*width)
                    .context("Qwen M-RoPE section width overflow")
            })?;
    anyhow::ensure!(
        language.head_dim.is_multiple_of(2)
            && language.rope_scaling.mrope_interleaved
            && mrope_width == language.head_dim / 2,
        "Qwen language M-RoPE configuration is incompatible with its head dimension"
    );
    anyhow::ensure!(
        language.rope_theta.is_finite()
            && language.rope_theta > 0.
            && language.rms_norm_eps.is_finite()
            && language.rms_norm_eps > 0.
            && vision.out_hidden_size == language.hidden_size,
        "Qwen language RoPE/norm values or vision-to-language width are invalid"
    );
    for (name, processor) in [("image", image_processor), ("video", video_processor)] {
        anyhow::ensure!(
            processor.patch_size == vision.patch_size
                && processor.temporal_patch_size == vision.temporal_patch_size
                && processor.merge_size == vision.spatial_merge_size,
            "Qwen {name} processor patch geometry differs from the vision tower"
        );
        anyhow::ensure!(
            processor.size.shortest_edge > 0
                && processor.size.longest_edge >= processor.size.shortest_edge,
            "Qwen {name} processor pixel bounds are invalid"
        );
        anyhow::ensure!(
            processor.image_mean.iter().all(|value| value.is_finite())
                && processor
                    .image_std
                    .iter()
                    .all(|&value| value.is_finite() && value > 0.),
            "Qwen {name} processor normalization statistics must be finite with positive standard deviation"
        );
    }
    Ok(())
}

fn integer_square_root(value: usize) -> Result<usize> {
    let root = (value as f64).sqrt() as usize;
    anyhow::ensure!(
        root.checked_mul(root) == Some(value),
        "Qwen learned position table is not square"
    );
    Ok(root)
}

fn validate_vision_layer_shapes(
    weights: &ModelWeights,
    layer: usize,
    config: &VisionConfig,
) -> Result<()> {
    let prefix = format!("model.visual.blocks.{layer}");
    for suffix in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
        expect_shape(
            weights,
            &format!("{prefix}.{suffix}"),
            &[config.hidden_size],
        )?;
    }
    expect_shape(
        weights,
        &format!("{prefix}.attn.qkv.weight"),
        &[3 * config.hidden_size, config.hidden_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.attn.qkv.bias"),
        &[3 * config.hidden_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.attn.proj.weight"),
        &[config.hidden_size, config.hidden_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.attn.proj.bias"),
        &[config.hidden_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.mlp.linear_fc1.weight"),
        &[config.intermediate_size, config.hidden_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.mlp.linear_fc1.bias"),
        &[config.intermediate_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.mlp.linear_fc2.weight"),
        &[config.hidden_size, config.intermediate_size],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.mlp.linear_fc2.bias"),
        &[config.hidden_size],
    )
}

fn validate_merger_shapes(
    weights: &ModelWeights,
    prefix: &str,
    config: &VisionConfig,
    postshuffle_norm: bool,
) -> Result<()> {
    let merged = config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
    let norm = if postshuffle_norm {
        merged
    } else {
        config.hidden_size
    };
    expect_shape(weights, &format!("{prefix}.norm.weight"), &[norm])?;
    expect_shape(weights, &format!("{prefix}.norm.bias"), &[norm])?;
    expect_shape(
        weights,
        &format!("{prefix}.linear_fc1.weight"),
        &[merged, merged],
    )?;
    expect_shape(weights, &format!("{prefix}.linear_fc1.bias"), &[merged])?;
    expect_shape(
        weights,
        &format!("{prefix}.linear_fc2.weight"),
        &[config.out_hidden_size, merged],
    )?;
    expect_shape(
        weights,
        &format!("{prefix}.linear_fc2.bias"),
        &[config.out_hidden_size],
    )
}

fn validate_language_layer_shapes(
    weights: &ModelWeights,
    layer: usize,
    config: &LanguageConfig,
) -> Result<()> {
    let prefix = format!("model.language_model.layers.{layer}");
    for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        expect_shape(
            weights,
            &format!("{prefix}.{suffix}"),
            &[config.hidden_size],
        )?;
    }
    expect_shape(
        weights,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        ],
    )?;
    for projection in ["k_proj", "v_proj"] {
        expect_shape(
            weights,
            &format!("{prefix}.self_attn.{projection}.weight"),
            &[
                config.num_key_value_heads * config.head_dim,
                config.hidden_size,
            ],
        )?;
    }
    expect_shape(
        weights,
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        ],
    )?;
    for norm in ["q_norm", "k_norm"] {
        expect_shape(
            weights,
            &format!("{prefix}.self_attn.{norm}.weight"),
            &[config.head_dim],
        )?;
    }
    for projection in ["gate_proj", "up_proj"] {
        expect_shape(
            weights,
            &format!("{prefix}.mlp.{projection}.weight"),
            &[config.intermediate_size, config.hidden_size],
        )?;
    }
    expect_shape(
        weights,
        &format!("{prefix}.mlp.down_proj.weight"),
        &[config.hidden_size, config.intermediate_size],
    )
}

fn expect_shape(weights: &ModelWeights, name: &str, expected: &[usize]) -> Result<()> {
    let metadata = weights.metadata(name)?;
    anyhow::ensure!(
        metadata.shape == expected,
        "Qwen tensor {name} has shape {:?}, expected {expected:?}",
        metadata.shape
    );
    Ok(())
}

fn tokenize(tokenizer: &Tokenizer, value: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(value, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| anyhow::anyhow!("failed to tokenize Qwen presentation segment: {error}"))
}

fn validate_tokenizer_special_id(tokenizer: &Tokenizer, token: &str, expected: u32) -> Result<()> {
    let actual = tokenizer
        .token_to_id(token)
        .with_context(|| format!("Qwen tokenizer has no {token} special token"))?;
    anyhow::ensure!(
        actual == expected,
        "Qwen tokenizer maps {token} to {actual}, but the model config expects {expected}"
    );
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, description: &str) -> Result<T> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read {description} {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid {description} {}", path.display()))
}

fn linear(
    weights: &BTreeMap<String, Tensor>,
    weight_name: &str,
    bias_name: Option<&str>,
    input: &Tensor,
) -> Result<Tensor> {
    let weight = required(weights, weight_name)?;
    let bias = bias_name.map(|name| required(weights, name)).transpose()?;
    let input = input.to_dtype(weight.dtype())?;
    #[cfg(feature = "cuda")]
    if reference_cuda(input.device())
        && let Some(bias) = bias
    {
        anyhow::ensure!(
            input.dtype() == DType::BF16
                && weight.dtype() == DType::BF16
                && bias.dtype() == DType::BF16,
            "released Qwen CUDA biased linear requires BF16 tensors"
        );
        return crate::cuda::linear::qwen_linear(&input, weight, bias)
            .with_context(|| format!("exact Qwen biased projection {weight_name}"));
    }
    Linear::new(weight.clone(), bias.cloned())
        .forward(&input)
        .with_context(|| format!("linear projection {weight_name}"))
}

fn vision_patch_projection(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if reference_cuda(input.device()) {
        anyhow::ensure!(
            input.dtype() == DType::BF16
                && weight.dtype() == DType::BF16
                && bias.dtype() == DType::BF16,
            "released Qwen CUDA patch projection requires BF16 tensors"
        );
        return crate::cuda::qwen::patch::projection(input, weight, bias).map_err(Into::into);
    }
    let output_width = weight.dim(0)?;
    let weight = weight.reshape((output_width, weight.elem_count() / output_width))?;
    Linear::new(weight, Some(bias.clone()))
        .forward(input)
        .map_err(Into::into)
}

fn vision_position_embedding(
    table: &Tensor,
    grids: &[VisionGrid],
    expected_rows: usize,
    position_embeddings: usize,
    merge: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(table.device())
        && crate::policy::position_profile_is_verified_for_dimensions(
            &grids
                .iter()
                .map(|grid| [grid.temporal, grid.height, grid.width])
                .collect::<Vec<_>>(),
        )
    {
        let grids = grids
            .iter()
            .map(|grid| [grid.temporal, grid.height, grid.width])
            .collect::<Vec<_>>();
        let position = crate::cuda::qwen::position::position_embedding(table, &grids)?;
        anyhow::ensure!(
            position.dims() == [expected_rows, table.dim(1)?],
            "exact Qwen learned-position output shape changed"
        );
        return Ok(position);
    }
    let (position_indices, position_weights) =
        learned_position_interpolation(grids, integer_square_root(position_embeddings)?, merge)?;
    let indices = Tensor::from_vec(position_indices, (expected_rows * 4,), table.device())?;
    let gathered = table
        .index_select(&indices, 0)?
        .to_dtype(DType::F32)?
        .reshape((expected_rows, 4, table.dim(1)?))?;
    let interpolation = Tensor::from_vec(position_weights, (expected_rows, 4, 1), table.device())?;
    gathered
        .broadcast_mul(&interpolation)?
        .sum(1)
        .map_err(Into::into)
}

fn vision_layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device()) {
        return crate::cuda::qwen::layer_norm::layer_norm(input, weight, bias, eps)
            .map_err(Into::into);
    }
    core::layer_norm(input, weight, bias, eps)
}

fn language_head_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device())
        && crate::cuda::qwen::attention::head_rms_norm_width128_covers(input)
    {
        return crate::cuda::qwen::attention::head_rms_norm_width128(input, weight, eps)
            .map_err(Into::into);
    }
    core::qwen_rms_norm(input, weight, eps)
}

fn vision_block_gelu(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device()) {
        return crate::cuda::qwen::gelu::vision_block_gelu_tanh(input).map_err(Into::into);
    }
    input.gelu().map_err(Into::into)
}

fn vision_merger_gelu(input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device()) {
        return crate::cuda::qwen::gelu::merger_gelu_erf(input).map_err(Into::into);
    }
    core::qwen_gelu_erf_with_reference_rounding(input)
}

fn required<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    weights
        .get(name)
        .with_context(|| format!("missing Qwen3-VL tensor {name}"))
}

fn with_named_group<T>(
    weights: &ModelWeights,
    names: &[String],
    device: &Device,
    function: impl FnOnce(&BTreeMap<String, Tensor>) -> Result<T>,
) -> Result<T> {
    let names = names.iter().map(String::as_str).collect::<Vec<_>>();
    weights.with_group(&names, device, function)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;

    fn processor() -> ProcessorConfig {
        ProcessorConfig {
            size: ProcessorSize {
                shortest_edge: 1,
                longest_edge: 1024 * 1024,
            },
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }

    #[test]
    fn ordered_vision_grids_follow_presentation_pad_runs() {
        let mut mm_types = vec![0u8, 0];
        mm_types.extend(std::iter::repeat_n(1, 1008));
        mm_types.push(0);
        for temporal in 0..7 {
            mm_types.extend(std::iter::repeat_n(2, 1008));
            if temporal != 6 {
                mm_types.push(0);
            }
        }
        let image = VisionGrid {
            temporal: 1,
            height: 48,
            width: 84,
        };
        let video = VisionGrid {
            temporal: 7,
            height: 48,
            width: 84,
        };
        let grids = canonical_ordered_vision_grids(&mm_types, &[image], &[video], 2).unwrap();
        assert_eq!(
            grids,
            vec![
                (H3QwenVisionGridModality::Image, 1, 48, 84),
                (H3QwenVisionGridModality::Video, 7, 48, 84),
            ]
        );
        mm_types.pop();
        assert!(canonical_ordered_vision_grids(&mm_types, &[image], &[video], 2).is_err());
    }

    #[test]
    fn released_fl2va_canvas_geometry_is_32_aligned() {
        assert_eq!(resolve_fl2va_canvas_size(16, 9).unwrap(), (768, 1344));
        assert_eq!(resolve_fl2va_canvas_size(9, 16).unwrap(), (1344, 768));
        assert_eq!(resolve_fl2va_canvas_size(1, 1).unwrap(), (768, 768));
        assert!(resolve_fl2va_canvas_size(5, 1).is_err());
    }

    #[test]
    fn image_preprocess_repeats_time_and_uses_merge_major_patch_order() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for row in 0..32 {
            for column in 0..32 {
                pixels.extend_from_slice(&[column as u8, row as u8, 255]);
            }
        }
        let image = RgbImage::new(32, 32, pixels).unwrap();
        let batch =
            preprocess_sequences(VisionModality::Image, &[vec![image]], &processor()).unwrap();
        assert_eq!(
            batch.grids,
            [VisionGrid {
                temporal: 1,
                height: 2,
                width: 2
            }]
        );
        assert_eq!(batch.patch_rows(), 4);
        assert_eq!(batch.patch_width, 1536);

        assert_eq!(batch.patches[0], -1.);
        assert_eq!(batch.patches[255], 15. / 127.5 - 1.);
        assert_eq!(batch.patches[256], -1.);
        assert_eq!(batch.patches[1536], 16. / 127.5 - 1.);
        assert_eq!(batch.patches[1024], 1.);
    }

    #[test]
    fn multimodal_positions_match_qwen_run_semantics() {
        let positions = multimodal_position_ids(
            &[0, 0, 1, 1, 1, 1, 0],
            &[VisionGrid {
                temporal: 1,
                height: 4,
                width: 4,
            }],
            &[],
            2,
        )
        .unwrap();
        assert_eq!(
            positions,
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4],
            ]
        );
    }

    #[test]
    fn rgb8_png_decoder_round_trips_pixels() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[1, 2, 3, 4, 5, 6]).unwrap();
            writer.finish().unwrap();
        }
        let image = RgbImage::from_png_bytes(&bytes).unwrap();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.pixels(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn tokenizer_special_id_mismatch_is_rejected_before_encoding() {
        let vocab = [
            ("[UNK]".to_owned(), 0),
            ("<|vision_start|>".to_owned(), 10),
            ("<|vision_end|>".to_owned(), 11),
            ("<|image_pad|>".to_owned(), 12),
            ("<|video_pad|>".to_owned(), 13),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        validate_tokenizer_special_id(&tokenizer, "<|vision_start|>", 10).unwrap();
        assert!(validate_tokenizer_special_id(&tokenizer, "<|vision_start|>", 99).is_err());
        assert!(validate_tokenizer_special_id(&tokenizer, "<|missing|>", 14).is_err());
    }

    #[test]
    fn text_mrope_interleaves_height_and_width_frequency_slots() {
        let positions = [[7, 11, 13]];
        let (cos, sin) = multimodal_rotary_tables(
            &positions,
            128,
            5_000_000.,
            [24, 20, 20],
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(cos.dims(), &[1, 1, 128]);
        assert_eq!(sin.dims(), &[1, 1, 128]);
        let cosine = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((cosine[0] - 7f32.cos()).abs() < 1e-6);
        assert!((cosine[1] - (11. * 5_000_000f64.powf(-2. / 128.)).cos() as f32).abs() < 1e-6);
        assert!((cosine[2] - (13. * 5_000_000f64.powf(-4. / 128.)).cos() as f32).abs() < 1e-6);
        assert_eq!(cosine[..64], cosine[64..]);
    }

    #[test]
    fn attention_transposes_are_contiguous_at_both_matmul_boundaries() {
        let vision = Tensor::arange(0f32, (256 * 16 * 72) as f32, &Device::Cpu)
            .unwrap()
            .reshape((256, 16, 72))
            .unwrap();
        let query = vision.transpose(0, 1).unwrap().contiguous().unwrap();
        let key = vision.transpose(0, 1).unwrap().contiguous().unwrap();
        let value = vision.transpose(0, 1).unwrap().contiguous().unwrap();
        let key_t = key.transpose(1, 2).unwrap().contiguous().unwrap();
        let query_chunk = query.narrow(1, 0, 32).unwrap().contiguous().unwrap();
        assert!(query_chunk.is_contiguous());
        assert!(key_t.is_contiguous());
        assert!(value.is_contiguous());
        let scores = query_chunk.matmul(&key_t).unwrap();
        assert_eq!(scores.dims(), &[16, 32, 256]);
        assert_eq!(scores.matmul(&value).unwrap().dims(), &[16, 32, 72]);

        let language = Tensor::zeros((1, 37, 8, 16), DType::F32, &Device::Cpu).unwrap();
        let query = language.transpose(1, 2).unwrap().contiguous().unwrap();
        let key = language.transpose(1, 2).unwrap().contiguous().unwrap();
        let key_t = key.transpose(2, 3).unwrap().contiguous().unwrap();
        let query_chunk = query.narrow(2, 0, 13).unwrap().contiguous().unwrap();
        assert!(query_chunk.is_contiguous());
        assert!(key_t.is_contiguous());
        assert_eq!(query_chunk.matmul(&key_t).unwrap().dims(), &[1, 8, 13, 37]);
    }
}
