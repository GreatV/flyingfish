//! MiniMax-H3 FL2VA transformer geometry, packed-sequence layout and sampling schedule.
use crate::model_spec::{equal, exact_keys, positive, u32_list};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{fs, path::Path};

#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine::minimax_h3::generate;

/// Row modality tags; the AdaLN table row of a sequence row is `timestep_index * MODALITIES + tag`.
pub const VIDEO_TAG: u8 = 0;
pub const TEXT_TAG: u8 = 1;
pub const AUDIO_TAG: u8 = 2;
pub const MODALITIES: u32 = 3;
/// Pixel frames per video VAE clip, the latent frames a clip keeps, and pixels per latent.
const VAE_CLIP_FRAMES: usize = 17;
const VAE_CLIP_LATENTS: usize = 5;
const VAE_SPATIAL_RATIO: usize = 16;
const FPS: f64 = 24.0;
const MIN_SECONDS: f64 = 5.0;
const MAX_SECONDS: f64 = 15.0;
const AUDIO_LATENTS_PER_SECOND: f64 = 40.0;
/// Stereo audio rows are packed channel-major, one block per channel.
const AUDIO_CHANNELS: usize = 2;
/// Conditioning video rows are pinned at `max(t, KEYFRAME_NOISE_AUG)`; keyframe anchors are noised to it.
pub const KEYFRAME_NOISE_AUG: f64 = 0.999;
/// Qwen3-VL hidden state that conditions the transformer (model definition `text_encoder_layer`).
pub const TEXT_ENCODER_LAYER: usize = 50;
/// Seed of the conditioning posterior, fixed independently of the request generator (model definition
/// `modular_pipeline.py:279` `keyframe_encode_seed` returns 42; `encoders.py:134` samples the posterior
/// under a fresh `manual_seed(encode_seed)`).
pub const KEYFRAME_ENCODE_SEED: u64 = 42;
/// Philox counter streams of the four native normal draws; each (seed, stream) pair is a distinct counter range.
/// No identity with the model definition's torch generator is claimed.
pub const STREAM_POSTERIOR_EPS: u64 = 0;
pub const STREAM_ANCHOR_NOISE: u64 = 1;
pub const STREAM_VIDEO_NOISE: u64 = 2;
pub const STREAM_AUDIO_ROWS: u64 = 3;
/// Latent frame `i` spans `ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5]` rotary time units.
const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const ROPE_SPATIAL_SCALE: f64 = 32.0;
/// Grids of at most 15 points follow the scalar element form of CPU `torch.linspace` (vector chunks start at 16 f32 lanes).
const MAX_GRID_POINTS: usize = 15;
/// Released spatial tile geometry of the video VAE, pixels: tile side and minimum overlap between neighbours.
pub const VAE_TILE: usize = 256;
const VAE_TILE_MIN_OVERLAP: usize = 64;
/// Video VAE pixel normalisation, ImageNet's (model definition `MiniMaxH3ModularPipeline.pixel_mean` / `pixel_std`).
pub const VAE_PIXEL_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
pub const VAE_PIXEL_STD: [f64; 3] = [0.229, 0.224, 0.225];
const VAE_KEYS: [&str; 24] = [
    "_class_name",
    "_diffusers_version",
    "in_channels",
    "out_channels",
    "latent_channels",
    "block_out_channels",
    "layers_per_block",
    "spatial_downsample_factors",
    "temporal_downsample_factors",
    "norm_num_groups",
    "norm_eps",
    "spatial_padding_mode",
    "decoder_num_layers",
    "decoder_num_attention_heads",
    "decoder_attention_head_dim",
    "decoder_num_register_tokens",
    "decoder_ffn_mult",
    "decoder_rope_theta",
    "decoder_rope_dim_ratio",
    "decoder_norm_eps",
    "clip_length",
    "token_drop",
    "latents_mean",
    "latents_std",
];
const TRANSFORMER_KEYS: [&str; 20] = [
    "_class_name",
    "_diffusers_version",
    "num_attention_heads",
    "attention_head_dim",
    "hidden_size",
    "num_layers",
    "num_refiner_layers",
    "ffn_dim",
    "in_channels",
    "audio_in_channels",
    "patch_size",
    "text_dim",
    "freq_dim",
    "time_embed_hidden_dim",
    "time_embed_dim",
    "rope_freq_dim",
    "rope_theta",
    "norm_eps",
    "qk_norm_eps",
    "final_norm_eps",
];

#[derive(Clone, Debug, PartialEq)]
pub struct H3Spec {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub ffn: usize,
    pub in_channels: usize,
    pub audio_in_channels: usize,
    /// `(t, h, w)` latent patch.
    pub patch: [usize; 3],
    pub text_dim: usize,
    pub freq_dim: usize,
    pub time_hidden: usize,
    pub time_dim: usize,
    /// Frequencies per rotary axis; `6 * rope_freq_dim` leading channels of every head rotate.
    pub rope_freq_dim: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
    pub final_norm_eps: f32,
}

impl H3Spec {
    /// Reads `transformer/config.json` and checks `vae/config.json` against the layout constants.
    pub fn read(model_dir: &Path) -> Result<Self> {
        let read = |component: &str| -> Result<Value> {
            let path = model_dir.join(component).join("config.json");
            serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
            )
            .with_context(|| format!("parse {}", path.display()))
        };
        VaeSpec::parse(&read("vae")?)?;
        Self::parse(&read("transformer")?)
    }

    pub fn parse(root: &Value) -> Result<Self> {
        exact_keys(root, &TRANSFORMER_KEYS, "MiniMax-H3 transformer config")?;
        equal(root, "_class_name", "MiniMaxH3Transformer3DModel".into())?;
        let n = |key| positive(root, key).map(|v| v as usize);
        let f = |key: &str| {
            root.get(key)
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| v as f32)
                .with_context(|| format!("missing/invalid {key}"))
        };
        let patch = u32_list(root, "patch_size")?;
        ensure!(patch.len() == 3, "patch_size is not (t, h, w)");
        let spec = Self {
            hidden: n("hidden_size")?,
            heads: n("num_attention_heads")?,
            head_dim: n("attention_head_dim")?,
            layers: n("num_layers")?,
            refiner_layers: n("num_refiner_layers")?,
            ffn: n("ffn_dim")?,
            in_channels: n("in_channels")?,
            audio_in_channels: n("audio_in_channels")?,
            patch: [patch[0] as usize, patch[1] as usize, patch[2] as usize],
            text_dim: n("text_dim")?,
            freq_dim: n("freq_dim")?,
            time_hidden: n("time_embed_hidden_dim")?,
            time_dim: n("time_embed_dim")?,
            rope_freq_dim: n("rope_freq_dim")?,
            rope_theta: f("rope_theta")?,
            norm_eps: f("norm_eps")?,
            qk_norm_eps: f("qk_norm_eps")?,
            final_norm_eps: f("final_norm_eps")?,
        };
        ensure!(
            spec.head_dim == 128,
            "unsupported MiniMax-H3 head_dim {}",
            spec.head_dim
        );
        ensure!(
            6 * spec.rope_freq_dim <= spec.head_dim,
            "rotary width exceeds head_dim"
        );
        ensure!(
            spec.patch[0] == 1 && spec.patch[1] > 0 && spec.patch[2] > 0,
            "unsupported MiniMax-H3 patch {:?}",
            spec.patch
        );
        Ok(spec)
    }

    /// Channels of one patchified video row.
    pub fn video_row_width(&self) -> usize {
        self.in_channels * self.patch.iter().product::<usize>()
    }

    /// `(name, out, in)` of the attention and FFN projections of every refiner block, then every transformer block.
    pub fn block_linears(&self) -> Vec<(String, usize, usize)> {
        let width = self.heads * self.head_dim;
        let blocks = (0..self.refiner_layers)
            .map(|i| format!("token_refiner.refiner_blocks.{i}"))
            .chain((0..self.layers).map(|i| format!("transformer_blocks.{i}")));
        blocks
            .flat_map(|block| {
                [
                    ("attn.to_q", width, self.hidden),
                    ("attn.to_k", width, self.hidden),
                    ("attn.to_v", width, self.hidden),
                    ("attn.to_out.0", self.hidden, width),
                    ("ff.net.0.proj", 2 * self.ffn, self.hidden),
                    ("ff.net.2", self.hidden, self.ffn),
                ]
                .map(|(name, out, input)| (format!("{block}.{name}"), out, input))
            })
            .collect()
    }
}

/// MiniMax-H3 video VAE configuration (`vae/config.json`): the causal 3D CNN encoder and the non-causal ViT decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct VaeSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub spatial_factors: Vec<usize>,
    pub temporal_factors: Vec<usize>,
    pub spatial_ratio: usize,
    pub temporal_ratio: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub registers: usize,
    pub ffn: usize,
    pub rope_theta: f32,
    /// Rotated channels of every head: `head_dim * decoder_rope_dim_ratio`, three axes of equal width.
    pub rope_dims: usize,
    pub norm_eps: f32,
    pub clip_length: usize,
    pub token_drop: usize,
    /// Encoder GroupNorm groups and epsilon (`norm_num_groups`, `norm_eps`).
    pub groups: usize,
    pub encoder_eps: f32,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl VaeSpec {
    pub fn read(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("vae").join("config.json");
        Self::parse(
            &serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
            )
            .with_context(|| format!("parse {}", path.display()))?,
        )
    }

    pub fn parse(root: &Value) -> Result<Self> {
        exact_keys(root, &VAE_KEYS, "MiniMax-H3 video VAE config")?;
        equal(root, "_class_name", "AutoencoderKLMiniMaxH3".into())?;
        equal(root, "spatial_padding_mode", "reflect".into())?;
        let n = |key| positive(root, key).map(|v| v as usize);
        let f = |key: &str| {
            root.get(key)
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .with_context(|| format!("missing/invalid {key}"))
        };
        let list = |key| -> Result<Vec<usize>> {
            Ok(u32_list(root, key)?
                .into_iter()
                .map(|v| v as usize)
                .collect())
        };
        let (spatial_factors, temporal_factors) = (
            list("spatial_downsample_factors")?,
            list("temporal_downsample_factors")?,
        );
        let head_dim = n("decoder_attention_head_dim")?;
        let rope = head_dim as f64 * f("decoder_rope_dim_ratio")?;
        let spec = Self {
            in_channels: n("in_channels")?,
            out_channels: n("out_channels")?,
            latent_channels: n("latent_channels")?,
            block_out_channels: list("block_out_channels")?,
            layers_per_block: n("layers_per_block")?,
            spatial_ratio: spatial_factors.iter().product(),
            temporal_ratio: temporal_factors.iter().product(),
            spatial_factors,
            temporal_factors,
            dim: n("decoder_num_attention_heads")? * head_dim,
            layers: n("decoder_num_layers")?,
            heads: n("decoder_num_attention_heads")?,
            head_dim,
            registers: n("decoder_num_register_tokens")?,
            ffn: n("decoder_ffn_mult")? * n("decoder_num_attention_heads")? * head_dim,
            rope_theta: f("decoder_rope_theta")? as f32,
            rope_dims: rope as usize,
            norm_eps: f("decoder_norm_eps")? as f32,
            clip_length: n("clip_length")?,
            token_drop: n("token_drop")?,
            groups: n("norm_num_groups")?,
            encoder_eps: f("norm_eps")? as f32,
            latents_mean: f32_list(root, "latents_mean")?,
            latents_std: f32_list(root, "latents_std")?,
        };
        let blocks = spec.block_out_channels.len();
        ensure!(
            spec.spatial_factors.len() == blocks
                && spec.temporal_factors.len() == blocks
                && spec.spatial_ratio == VAE_SPATIAL_RATIO
                && spec.clip_length == VAE_CLIP_FRAMES
                && spec.clip_length.div_ceil(spec.temporal_ratio) == VAE_CLIP_LATENTS
                && rope.fract() == 0.0
                && spec.rope_dims.is_multiple_of(6)
                && spec.rope_dims <= head_dim
                && spec.token_drop < VAE_CLIP_LATENTS
                && spec
                    .block_out_channels
                    .iter()
                    .all(|c| c.is_multiple_of(spec.groups))
                && spec
                    .spatial_factors
                    .iter()
                    .zip(&spec.temporal_factors)
                    .all(|(&s, &t)| s == 2 || (s == 1 && t == 1))
                && spec.latents_mean.len() == spec.latent_channels
                && spec.latents_std.len() == spec.latent_channels
                && spec.latents_std.iter().all(|v| *v > 0.0),
            "unsupported MiniMax-H3 video VAE geometry: {spec:?}"
        );
        Ok(spec)
    }

    /// Channels of one decoder output token: `out_channels x temporal_ratio x spatial_ratio^2` pixels.
    pub fn patch_width(&self) -> usize {
        self.out_channels * self.temporal_ratio * self.spatial_ratio * self.spatial_ratio
    }

    /// Latent frames per decoded chunk (`ceil(clip_length / temporal_ratio)`).
    pub fn chunk_latents(&self) -> usize {
        self.clip_length.div_ceil(self.temporal_ratio)
    }

    /// Latent frames a chunk shares with the next one: `(-token_drop) mod chunk_latents`.
    pub fn overlap_latents(&self) -> usize {
        (self.chunk_latents() - self.token_drop % self.chunk_latents()) % self.chunk_latents()
    }

    /// Leading pixel frames of every decoded chunk that `clip_length` leaves implicit: `(-clip_length) mod temporal_ratio`.
    pub fn frame_pre_padding(&self) -> usize {
        (self.temporal_ratio - self.clip_length % self.temporal_ratio) % self.temporal_ratio
    }

    /// Pixel frames that consecutive chunks cross-fade over.
    pub fn frame_overlap(&self) -> usize {
        (self.overlap_latents() * self.temporal_ratio).saturating_sub(self.frame_pre_padding())
    }
}

/// Spatial tiles over `length` pixels: the fewest `tile`-wide tiles whose overlaps are all at least `min_overlap`, the
/// remaining slack spread round-robin over the overlaps in `ratio` steps. Returns the tile side, starts and overlaps; a
/// length of at most one tile is one tile of that length.
pub fn vae_tiles(
    length: usize,
    tile: usize,
    min_overlap: usize,
    ratio: usize,
) -> Result<(usize, Vec<usize>, Vec<usize>)> {
    ensure!(
        length > 0 && min_overlap < tile && ratio > 0 && length.is_multiple_of(ratio),
        "invalid VAE tiling of {length} px by {tile} px tiles"
    );
    if tile >= length {
        return Ok((length, vec![0], Vec::new()));
    }
    let mut count = length.div_ceil(tile);
    while tile * count < min_overlap * (count - 1) + length {
        count += 1;
    }
    let mut overlaps = vec![min_overlap; count - 1];
    let slack = tile * count - min_overlap * (count - 1) - length;
    for i in 0..slack / ratio {
        overlaps[i % (count - 1)] += ratio;
    }
    let mut starts = vec![0];
    for overlap in &overlaps {
        starts.push(starts[starts.len() - 1] + tile - overlap);
    }
    Ok((tile, starts, overlaps))
}

const AUDIO_VAE_KEYS: [&str; 15] = [
    "_class_name",
    "_diffusers_version",
    "encoder_dim",
    "encoder_rates",
    "latent_dim",
    "latent_channels",
    "decoder_dim",
    "decoder_rates",
    "decoder_kernel_sizes",
    "num_attention_heads",
    "resblock_kernel_sizes",
    "resblock_dilation_sizes",
    "sampling_rate",
    "latents_mean",
    "latents_std",
];
/// Alias-free activation resampling of the audio decoder: ratio and Kaiser filter taps (model definition
/// `MiniMaxH3AudioActivation1d` defaults).
pub const AUDIO_RESAMPLE_RATIO: usize = 2;
pub const AUDIO_RESAMPLE_TAPS: usize = 12;

/// MiniMax-H3 audio VAE (`audio_vae/config.json`): DAC encoder, attention pre-block and BigVGAN decoder geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioVaeSpec {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub latent_channels: usize,
    pub heads: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_kernels: Vec<usize>,
    pub resblock_kernels: Vec<usize>,
    pub resblock_dilations: Vec<Vec<usize>>,
    pub sampling_rate: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl AudioVaeSpec {
    pub fn read(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("audio_vae").join("config.json");
        Self::parse(
            &serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
            )
            .with_context(|| format!("parse {}", path.display()))?,
        )
    }

    pub fn parse(root: &Value) -> Result<Self> {
        exact_keys(root, &AUDIO_VAE_KEYS, "MiniMax-H3 audio VAE config")?;
        equal(root, "_class_name", "AutoencoderKLMiniMaxH3Audio".into())?;
        let n = |key| positive(root, key).map(|v| v as usize);
        let list = |key| -> Result<Vec<usize>> {
            Ok(u32_list(root, key)?
                .into_iter()
                .map(|v| v as usize)
                .collect())
        };
        let resblock_dilations = root
            .get("resblock_dilation_sizes")
            .and_then(Value::as_array)
            .context("missing/invalid resblock_dilation_sizes")?
            .iter()
            .map(|row| {
                row.as_array()
                    .context("resblock_dilation_sizes row is not a list")?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .filter(|&v| v > 0)
                            .map(|v| v as usize)
                            .context("resblock dilation is not a positive integer")
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let spec = Self {
            encoder_dim: n("encoder_dim")?,
            encoder_rates: list("encoder_rates")?,
            latent_dim: n("latent_dim")?,
            latent_channels: n("latent_channels")?,
            heads: n("num_attention_heads")?,
            decoder_dim: n("decoder_dim")?,
            decoder_rates: list("decoder_rates")?,
            decoder_kernels: list("decoder_kernel_sizes")?,
            resblock_kernels: list("resblock_kernel_sizes")?,
            resblock_dilations,
            sampling_rate: n("sampling_rate")?,
            latents_mean: f32_list(root, "latents_mean")?,
            latents_std: f32_list(root, "latents_std")?,
        };
        let levels = spec.decoder_rates.len();
        ensure!(
            spec.decoder_rates.iter().product::<usize>()
                == spec.encoder_rates.iter().product::<usize>()
                && spec.decoder_kernels.len() == levels
                && spec
                    .decoder_rates
                    .iter()
                    .zip(&spec.decoder_kernels)
                    .all(|(&s, &k)| k > s && (k - s).is_multiple_of(2))
                && spec.decoder_dim.is_multiple_of(1 << levels)
                && spec.latent_dim.is_multiple_of(spec.latent_channels)
                && spec.latent_dim.is_multiple_of(spec.heads)
                && spec.resblock_kernels.len() == spec.resblock_dilations.len()
                && spec.resblock_kernels.iter().all(|k| k % 2 == 1)
                && spec.latents_mean.len() == spec.latent_channels
                && spec.latents_std.len() == spec.latent_channels,
            "unsupported MiniMax-H3 audio VAE geometry: {spec:?}"
        );
        Ok(spec)
    }

    /// Waveform samples per latent frame: the product of the decoder rates.
    pub fn hop(&self) -> usize {
        self.decoder_rates.iter().product()
    }
}

/// A JSON list of finite numbers as F32 (nearest).
fn f32_list(root: &Value, key: &str) -> Result<Vec<f32>> {
    root.get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing/invalid {key}"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|v| v.is_finite())
                .map(|v| v as f32)
                .with_context(|| format!("{key} contains a non-finite or non-number value"))
        })
        .collect()
}

/// Spatial tile grid of a pixel frame: tile height and width, and per axis the tile starts and neighbour overlaps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaeTiles {
    pub height: usize,
    pub width: usize,
    pub rows: (Vec<usize>, Vec<usize>),
    pub cols: (Vec<usize>, Vec<usize>),
}

impl VaeTiles {
    /// The released tiling (`VAE_TILE` px tiles, at least `VAE_TILE_MIN_OVERLAP` px overlap).
    pub fn released(spec: &VaeSpec, height: usize, width: usize) -> Result<Self> {
        let (tile_height, row_starts, row_overlaps) =
            vae_tiles(height, VAE_TILE, VAE_TILE_MIN_OVERLAP, spec.spatial_ratio)?;
        let (tile_width, col_starts, col_overlaps) =
            vae_tiles(width, VAE_TILE, VAE_TILE_MIN_OVERLAP, spec.spatial_ratio)?;
        Ok(Self {
            height: tile_height,
            width: tile_width,
            rows: (row_starts, row_overlaps),
            cols: (col_starts, col_overlaps),
        })
    }

    /// One tile over the whole frame (tiling off).
    pub fn whole(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            rows: (vec![0], Vec::new()),
            cols: (vec![0], Vec::new()),
        }
    }

    pub fn count(&self) -> usize {
        self.rows.0.len() * self.cols.0.len()
    }
}

/// Temporal decode plan of `latent_frames` latent frames: the latent tail repeated `pad` times, `chunks` decoded
/// chunks of `chunk_latents() + overlap_latents()` latent frames starting every `chunk_latents()`, and the output
/// frame count after the pad frames are cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub latent_frames: usize,
    pub pad: usize,
    pub chunks: usize,
    pub frames: usize,
}

impl DecodePlan {
    pub fn new(spec: &VaeSpec, latent_frames: usize) -> Result<Self> {
        let chunk = spec.chunk_latents();
        ensure!(latent_frames > 0, "empty latent video");
        let tokens = latent_frames + spec.token_drop;
        let pad = (chunk - tokens % chunk) % chunk;
        let chunks = (tokens + pad) / chunk - usize::from(spec.token_drop > 0);
        ensure!(
            chunks > 0,
            "{latent_frames} latent frames give no decode chunk"
        );
        let kept = chunk * spec.temporal_ratio - spec.frame_pre_padding();
        let tail = (chunk + spec.overlap_latents()) * spec.temporal_ratio
            - chunk * spec.temporal_ratio
            - spec.frame_pre_padding();
        let intra_tail = spec.clip_length % spec.temporal_ratio;
        let pad_frames: usize = (0..pad)
            .map(|k| {
                if intra_tail != 0 && (latent_frames + k).is_multiple_of(chunk) {
                    intra_tail
                } else {
                    spec.temporal_ratio
                }
            })
            .sum();
        let frames = (chunks * kept + if spec.token_drop > 0 { tail } else { 0 })
            .checked_sub(pad_frames)
            .context("pad frames exceed the decoded frames")?;
        Ok(Self {
            latent_frames,
            pad,
            chunks,
            frames,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub height: usize,
    pub width: usize,
    /// Frame count of the form `17 n + 5`.
    pub frames: usize,
    pub latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    /// Audio latents per channel.
    pub audio_latents: usize,
}

impl Geometry {
    /// Rounds `frames` up to the next `17 n + 5`; the aligned duration must lie in `[5, 15]` s at 24 fps.
    pub fn resolve(spec: &H3Spec, height: usize, width: usize, frames: usize) -> Result<Self> {
        let multiple = VAE_SPATIAL_RATIO * spec.patch[2];
        ensure!(
            height > 0
                && width > 0
                && height.is_multiple_of(multiple)
                && width.is_multiple_of(multiple),
            "canvas {height}x{width} is not a positive multiple of {multiple}"
        );
        ensure!(frames > 0, "frame count must be positive");
        let frames = frames
            + (VAE_CLIP_LATENTS + VAE_CLIP_FRAMES - frames % VAE_CLIP_FRAMES) % VAE_CLIP_FRAMES;
        let seconds = frames as f64 / FPS;
        ensure!(
            (MIN_SECONDS..=MAX_SECONDS).contains(&seconds),
            "aligned frame count {frames} is {seconds} s, outside [{MIN_SECONDS}, {MAX_SECONDS}]"
        );
        Ok(Self {
            height,
            width,
            frames,
            latent_frames: (frames - VAE_CLIP_LATENTS) / VAE_CLIP_FRAMES * VAE_CLIP_LATENTS + 2,
            latent_height: height / VAE_SPATIAL_RATIO,
            latent_width: width / VAE_SPATIAL_RATIO,
            audio_latents: (frames as f64 / FPS * AUDIO_LATENTS_PER_SECOND).round_ties_even()
                as usize,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeAnchor {
    First,
    Last,
}

/// `[text | keyframe conditions | audio | video]` rows of one request.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedLayout {
    /// `[rows, 3]` `(t, h, w)` rotary coordinates.
    pub position_ids: Vec<[f64; 3]>,
    pub token_tags: Vec<u8>,
    /// Sequence rows of the video stream, conditioning rows first.
    pub video_indices: Vec<u32>,
    /// Sequence rows of the audio stream, channel-major.
    pub audio_indices: Vec<u32>,
    pub text_indices: Vec<u32>,
    pub condition_video_rows: usize,
}

impl PackedLayout {
    pub fn build(
        spec: &H3Spec,
        geometry: &Geometry,
        text_tags: &[u8],
        anchors: &[KeyframeAnchor],
    ) -> Result<Self> {
        ensure!(
            text_tags
                .iter()
                .all(|&tag| tag == TEXT_TAG || tag == VIDEO_TAG),
            "text rows carry a tag other than text or video"
        );
        let (patch_h, patch_w) = (spec.patch[1], spec.patch[2]);
        ensure!(
            geometry.latent_height.is_multiple_of(patch_h)
                && geometry.latent_width.is_multiple_of(patch_w),
            "latent canvas is not divisible by the patch"
        );
        let grid_h = geometry.latent_height / patch_h;
        let grid_w = geometry.latent_width / patch_w;
        let frame_rows = grid_h * grid_w;
        let text = text_tags.len();
        let condition = anchors.len() * frame_rows;
        let audio_rows = geometry.audio_latents * AUDIO_CHANNELS;
        let condition_start = text;
        let audio_start = condition_start + condition;
        let video_start = audio_start + audio_rows;
        let rows = video_start + geometry.latent_frames * frame_rows;
        u32::try_from(rows).context("packed sequence exceeds u32 rows")?;

        let sqrt_area = ((geometry.latent_height * geometry.latent_width) as f64).sqrt();
        let height_grid = spatial_grid(geometry.latent_height, patch_h, sqrt_area);
        let width_grid = spatial_grid(geometry.latent_width, patch_w, sqrt_area);
        let frame_grid: Vec<[f64; 2]> = height_grid
            .iter()
            .flat_map(|&h| width_grid.iter().map(move |&w| [h, w]))
            .collect();
        let origin = text as f64;
        let spans: Vec<f64> = (0..geometry.latent_frames)
            .map(|i| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()])
            .collect();

        let mut position_ids = vec![[0.0f64; 3]; rows];
        for (row, position) in position_ids.iter_mut().take(text).enumerate() {
            position[0] = row as f64;
        }
        for (index, anchor) in anchors.iter().enumerate() {
            let time = match anchor {
                KeyframeAnchor::First => origin,
                KeyframeAnchor::Last => origin + pairwise_sum(&spans) - ROPE_FRAME_RESCALE,
            };
            let start = condition_start + index * frame_rows;
            for (position, &[h, w]) in position_ids[start..start + frame_rows]
                .iter_mut()
                .zip(&frame_grid)
            {
                *position = [time, h, w];
            }
        }
        let (first_w, last_w) = (width_grid[0], width_grid[width_grid.len() - 1]);
        for (offset, position) in position_ids[audio_start..video_start]
            .iter_mut()
            .enumerate()
        {
            let latent = offset % geometry.audio_latents;
            let w = if offset < geometry.audio_latents {
                first_w
            } else {
                last_w
            };
            *position = [origin + latent as f64, 0.0, w];
        }
        let mut elapsed = 0.0f64;
        for (frame, span) in spans.iter().enumerate() {
            let start = video_start + frame * frame_rows;
            for (position, &[h, w]) in position_ids[start..start + frame_rows]
                .iter_mut()
                .zip(&frame_grid)
            {
                *position = [origin + elapsed, h, w];
            }
            elapsed += span;
        }

        let mut token_tags = text_tags.to_vec();
        token_tags.extend(std::iter::repeat_n(VIDEO_TAG, condition));
        token_tags.extend(std::iter::repeat_n(AUDIO_TAG, audio_rows));
        token_tags.extend(std::iter::repeat_n(VIDEO_TAG, rows - video_start));
        let range = |a: usize, b: usize| (a as u32..b as u32).collect::<Vec<_>>();
        let mut video_indices = range(condition_start, audio_start);
        video_indices.extend(range(video_start, rows));
        Ok(Self {
            position_ids,
            token_tags,
            video_indices,
            audio_indices: range(audio_start, video_start),
            text_indices: range(0, text),
            condition_video_rows: condition,
        })
    }

    pub fn rows(&self) -> usize {
        self.token_tags.len()
    }
}

/// `dim / patch` coordinates `(i * step + left) * 32`, `step = (left + ratio - left) / count`, `ratio = dim / sqrt_area`.
fn spatial_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let count = dim / patch;
    let step = (left + ratio - left) / count as f64;
    (0..count)
        .map(|i| (i as f64 * step + left) * ROPE_SPATIAL_SCALE)
        .collect()
}

/// Pairwise f64 sum: sequential below 8 values, 8 interleaved accumulators up to 128, halves split at a multiple of 8.
fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        values.iter().fold(0.0, |sum, &v| sum + v)
    } else if n <= 128 {
        let mut lanes = [0.0f64; 8];
        lanes.copy_from_slice(&values[..8]);
        let blocked = n - n % 8;
        for chunk in values[8..blocked].as_chunks::<8>().0 {
            for (lane, &v) in lanes.iter_mut().zip(chunk) {
                *lane += v;
            }
        }
        let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for &v in &values[blocked..] {
            sum += v;
        }
        sum
    } else {
        let half = n / 2 - (n / 2) % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

/// Rectified-flow grid: `linspace(1, 0, points)` through `s * x / (1 + (s - 1) * x)` in f32, consecutive duplicates
/// collapsed; one model evaluation at `t = 1 - sigma` per sigma except the terminal 0.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl Schedule {
    pub fn new(grid_points: usize, shift: f64) -> Result<Self> {
        ensure!(
            (2..=MAX_GRID_POINTS).contains(&grid_points),
            "sigma grid of {grid_points} points outside [2, {MAX_GRID_POINTS}]"
        );
        ensure!(
            shift.is_finite() && shift > 0.0,
            "shift must be finite and positive"
        );
        let step = (0.0f32 - 1.0f32) / (grid_points - 1) as f32;
        let half = grid_points / 2;
        let (scale, slope) = (shift as f32, (shift - 1.0) as f32);
        let mut sigmas: Vec<f32> = Vec::with_capacity(grid_points);
        for i in 0..grid_points {
            let base = if i < half {
                1.0f32 + step * i as f32
            } else {
                0.0f32 - step * (grid_points - i - 1) as f32
            };
            let sigma = scale * base / (slope * base + 1.0f32);
            if sigmas.last() != Some(&sigma) {
                sigmas.push(sigma);
            }
        }
        let timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|&sigma| 1.0f32 - sigma)
            .collect();
        Ok(Self { sigmas, timesteps })
    }
}

/// Distinct timesteps of one evaluation, ascending, and the index of every sequence row into them.
#[derive(Clone, Debug, PartialEq)]
pub struct RowTimesteps {
    pub timesteps: Vec<f32>,
    pub indices: Vec<u32>,
}

impl RowTimesteps {
    /// Text and generated video rows at `video`, generated audio rows at `audio`, conditioning video rows at
    /// `max(video, 0.999)`.
    pub fn build(layout: &PackedLayout, video: f32, audio: f32) -> Self {
        let mut rows = vec![video; layout.rows()];
        let condition = (video as f64).max(KEYFRAME_NOISE_AUG) as f32;
        for &row in &layout.video_indices[..layout.condition_video_rows] {
            rows[row as usize] = condition;
        }
        for &row in &layout.audio_indices {
            rows[row as usize] = audio;
        }
        let mut timesteps = rows.clone();
        timesteps.sort_by(f32::total_cmp);
        timesteps.dedup();
        let indices = rows
            .iter()
            .map(|t| timesteps.partition_point(|v| v < t) as u32)
            .collect();
        Self { timesteps, indices }
    }

    pub fn adaln_indices(&self, layout: &PackedLayout) -> Vec<u32> {
        self.indices
            .iter()
            .zip(&layout.token_tags)
            .map(|(&index, &tag)| index * MODALITIES + u32::from(tag))
            .collect()
    }
}

/// One `RowTimesteps` per model evaluation; both grids must keep the same length after collapsing.
pub fn row_timestep_plan(
    layout: &PackedLayout,
    video: &Schedule,
    audio: &Schedule,
) -> Result<Vec<RowTimesteps>> {
    ensure!(
        video.timesteps.len() == audio.timesteps.len(),
        "video and audio grids collapse to {} and {} evaluations",
        video.timesteps.len(),
        audio.timesteps.len()
    );
    Ok(video
        .timesteps
        .iter()
        .zip(&audio.timesteps)
        .map(|(&v, &a)| RowTimesteps::build(layout, v, a))
        .collect())
}

/// Low-rank adapter: `rank` and the output scale `alpha / rank`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoraAdapter {
    pub rank: usize,
    pub scale: f32,
}

/// Every `(timestep, tag)` AdaLN pair addressed by any evaluation of a request, with per-evaluation row maps.
#[derive(Clone, Debug, PartialEq)]
pub struct AdalnPlan {
    /// Distinct timesteps over all evaluations, ascending.
    pub timesteps: Vec<f32>,
    /// `(index into timesteps, tag)` in ascending order.
    pub pairs: Vec<(u32, u8)>,
    /// Per evaluation, the pair of every sequence row.
    pub row_pairs: Vec<Vec<u32>>,
    /// Per evaluation, the index into `timesteps` of every sequence row.
    pub row_timesteps: Vec<Vec<u32>>,
}

impl AdalnPlan {
    pub fn build(layout: &PackedLayout, plan: &[RowTimesteps]) -> Self {
        let mut timesteps: Vec<f32> = plan
            .iter()
            .flat_map(|step| step.timesteps.clone())
            .collect();
        timesteps.sort_by(f32::total_cmp);
        timesteps.dedup();
        let global = |t: f32| timesteps.partition_point(|&v| v < t) as u32;
        let row_timesteps: Vec<Vec<u32>> = plan
            .iter()
            .map(|step| {
                let map: Vec<u32> = step.timesteps.iter().map(|&t| global(t)).collect();
                step.indices.iter().map(|&i| map[i as usize]).collect()
            })
            .collect();
        let mut pairs: Vec<(u32, u8)> = row_timesteps
            .iter()
            .flat_map(|rows| {
                rows.iter()
                    .zip(&layout.token_tags)
                    .map(|(&t, &tag)| (t, tag))
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let row_pairs = row_timesteps
            .iter()
            .map(|rows| {
                rows.iter()
                    .zip(&layout.token_tags)
                    .map(|(&t, &tag)| pairs.partition_point(|&p| p < (t, tag)) as u32)
                    .collect()
            })
            .collect();
        Self {
            timesteps,
            pairs,
            row_pairs,
            row_timesteps,
        }
    }
}

/// `silu(time_embedder(sinusoid(t)))` rounded to BF16, `[timesteps, time_dim]`: the input of every AdaLN projection.
/// Sinusoid `[cos(t f_i), sin(t f_i)]`, `f_i = exp(-ln(10000) * i / half)` in f32; the two F32 linears accumulate in
/// f64 and round once.
pub fn adaln_input(
    spec: &H3Spec,
    checkpoint: &crate::checkpoint::MappedCheckpoint,
    timesteps: &[f32],
) -> Result<Vec<half::bf16>> {
    let half = spec.freq_dim / 2;
    let log_period = -(10000.0f64.ln()) as f32;
    let frequencies: Vec<f32> = (0..half)
        .map(|i| (log_period * i as f32 / half as f32).exp())
        .collect();
    let mut x = Vec::with_capacity(timesteps.len() * spec.freq_dim);
    for &t in timesteps {
        x.extend(frequencies.iter().map(|&f| (t * f).cos()));
        x.extend(frequencies.iter().map(|&f| (t * f).sin()));
    }
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let hidden = f32_linear(
        checkpoint,
        "time_embedder.linear_1",
        &x,
        spec.freq_dim,
        spec.time_hidden,
    )?;
    let hidden: Vec<f32> = hidden.into_iter().map(silu).collect();
    let embedding = f32_linear(
        checkpoint,
        "time_embedder.linear_2",
        &hidden,
        spec.time_hidden,
        spec.time_dim,
    )?;
    Ok(embedding
        .into_iter()
        .map(|v| half::bf16::from_f32(silu(v)))
        .collect())
}

/// `x [rows, input] * weight[output, input]^T + bias` from the published F32 tensors.
fn f32_linear(
    checkpoint: &crate::checkpoint::MappedCheckpoint,
    base: &str,
    x: &[f32],
    input: usize,
    output: usize,
) -> Result<Vec<f32>> {
    let weight = checkpoint.f32_tensor(&format!("{base}.weight"), &[output, input])?;
    let bias = checkpoint.f32_tensor(&format!("{base}.bias"), &[output])?;
    Ok(x.chunks_exact(input)
        .flat_map(|row| {
            weight
                .chunks_exact(input)
                .zip(&bias)
                .map(|(w, &b)| {
                    (row.iter()
                        .zip(w)
                        .map(|(&a, &w)| f64::from(a) * f64::from(w))
                        .sum::<f64>()
                        + f64::from(b)) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

/// `[rows, 6 * rope_freq_dim]` BF16 rotary table: the cosines of each row's `3 * rope_freq_dim` angles, then their
/// sines. Angle `(axis, i)` is `f32(position[axis]) * inv_freq[i]` with `inv_freq[i] = 1 / theta^(2i / (2 *
/// rope_freq_dim))` in f32; axes in `(t, h, w)` order.
pub fn rotary_table(spec: &H3Spec, layout: &PackedLayout) -> Vec<half::bf16> {
    let width = 2 * spec.rope_freq_dim;
    let inv_freq: Vec<f32> = (0..spec.rope_freq_dim)
        .map(|i| 1.0f32 / spec.rope_theta.powf((2 * i) as f32 / width as f32))
        .collect();
    let mut table = Vec::with_capacity(layout.rows() * 3 * width);
    let mut angles = Vec::with_capacity(3 * spec.rope_freq_dim);
    for position in &layout.position_ids {
        angles.clear();
        for &coordinate in position {
            angles.extend(inv_freq.iter().map(|&f| coordinate as f32 * f));
        }
        table.extend(angles.iter().map(|a| half::bf16::from_f32(a.cos())));
        table.extend(angles.iter().map(|a| half::bf16::from_f32(a.sin())));
    }
    table
}

/// `[rows, head_dim]` f32 interleaved-mRoPE table of the Qwen3-VL conditioner: the cosines of each prompt row's
/// `head_dim / 2` angles, then their sines. Angle `j` is `f32(position[axis(j)]) * inv_freq[j]` with `inv_freq[j] =
/// 1 / f32(theta^(2j / head_dim))`, the power in f64.
pub fn mrope_table(
    plan: &crate::vision_input::PositionPlan,
    section: [u32; 3],
    theta: f32,
    head_dim: u32,
) -> Result<Vec<f32>> {
    let half = head_dim as usize / 2;
    ensure!(
        section.iter().sum::<u32>() as usize == half,
        "mRoPE section {section:?} does not cover head_dim {head_dim}"
    );
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / f64::from(theta).powf(f64::from((2 * j) as f32 / head_dim as f32)) as f32)
        .collect();
    let rows = plan.prompt_rows();
    let mut table = Vec::with_capacity(rows * 2 * half);
    let mut angles = vec![0f32; half];
    for row in 0..rows {
        for (j, angle) in angles.iter_mut().enumerate() {
            *angle = plan.rotary_position(section, j, row)? as f32 * inv_freq[j];
        }
        table.extend(angles.iter().map(|a| a.cos()));
        table.extend(angles.iter().map(|a| a.sin()));
    }
    Ok(table)
}

/// One MiniMax-H3 i2va request in the form the native path consumes. Conditioner prompt: token ids, the text tag of
/// every text row, image grids `[t, h, w]` and Qwen3-VL `pixel_values` rows. Keyframe: u8 pixels `[3, height, width]`.
/// `seed` is the request input from which the four normal draws are generated on device; the conditioning posterior
/// uses `KEYFRAME_ENCODE_SEED` instead. There is no recorded-draw path. Sampling: pixel frames, evaluation count and
/// the video / audio grid shifts.
#[derive(Clone, Debug, PartialEq)]
pub struct H3Request {
    pub token_ids: Vec<u32>,
    pub text_tags: Vec<u8>,
    pub grids: Vec<crate::vision_input::ImageGrid>,
    pub pixel_values: Vec<f32>,
    pub keyframe: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub frames: usize,
    pub seed: u64,
    pub evaluations: usize,
    pub shifts: (f64, f64),
}

/// Released canvas of MiniMax-H3 (model definition `canvas_short_edge` and `canvas_max_pixels`) and the aspect
/// ratios it was trained over.
const CANVAS_SHORT_EDGE: f64 = 768.0;
const CANVAS_MAX_PIXELS: f64 = 768.0 * 1344.0;
const CANVAS_ASPECT: (f64, f64) = (0.25, 4.0);

/// The canvas `(height, width)` of an aspect `aspect_width : aspect_height` (model definition `resolve_canvas_size`):
/// short edge 768, area capped at 768 * 1344, both axes rounded half to even to a multiple of the latent patch (32).
pub fn canvas_size(spec: &H3Spec, aspect_width: f64, aspect_height: f64) -> Result<(usize, usize)> {
    ensure!(
        aspect_width > 0.0 && aspect_height > 0.0,
        "aspect ratio {aspect_width}:{aspect_height} is not positive"
    );
    let ratio = aspect_width / aspect_height;
    ensure!(
        (CANVAS_ASPECT.0..=CANVAS_ASPECT.1).contains(&ratio),
        "aspect ratio {ratio} lies outside [{}, {}]",
        CANVAS_ASPECT.0,
        CANVAS_ASPECT.1
    );
    let (mut width, mut height) = if ratio >= 1.0 {
        (CANVAS_SHORT_EDGE * ratio, CANVAS_SHORT_EDGE)
    } else {
        (CANVAS_SHORT_EDGE, CANVAS_SHORT_EDGE / ratio)
    };
    let area = width * height;
    if area > CANVAS_MAX_PIXELS {
        let scale = (CANVAS_MAX_PIXELS / area).powf(0.5);
        (width, height) = (width * scale, height * scale);
    }
    let multiple = (VAE_SPATIAL_RATIO * spec.patch[2]) as f64;
    let snap = |v: f64| ((v / multiple).round_ties_even() * multiple).max(multiple) as usize;
    Ok((snap(height), snap(width)))
}

/// Token ids and row tags of an i2va / fl2va presentation (model definition encoders.py:283-301): per keyframe the
/// label `"<Picture {i}>: "` (text rows) and `<|vision_start|>`, `image_tokens[i]` x `<|image_pad|>`,
/// `<|vision_end|>` (video rows), then the prompt verbatim (text rows), all without special tokens.
pub fn presentation(
    tokenizer: &crate::serve::detok::Shared,
    prompt: &str,
    image_tokens: &[usize],
) -> Result<(Vec<u32>, Vec<u8>)> {
    let (start, pad, end) = (
        tokenizer.token_id("<|vision_start|>")?,
        tokenizer.token_id("<|image_pad|>")?,
        tokenizer.token_id("<|vision_end|>")?,
    );
    let (mut ids, mut tags) = (Vec::new(), Vec::new());
    for (index, &count) in image_tokens.iter().enumerate() {
        let label = tokenizer.encode_chat(&format!("<Picture {}>: ", index + 1))?;
        tags.extend(std::iter::repeat_n(TEXT_TAG, label.len()));
        ids.extend(label);
        ids.push(start);
        ids.extend(std::iter::repeat_n(pad, count));
        ids.push(end);
        tags.extend(std::iter::repeat_n(VIDEO_TAG, count + 2));
    }
    let text = tokenizer.encode_chat(prompt)?;
    tags.extend(std::iter::repeat_n(TEXT_TAG, text.len()));
    ids.extend(text);
    Ok((ids, tags))
}

/// The four documented (seed, stream) pairs of one request, in the pipeline's draw order: conditioning posterior
/// eps, anchor noise, video noise, audio rows. The posterior uses the definition's fixed encode seed; the other three
/// use the request seed. Each pair is a distinct Philox counter range; no identity with the model definition's torch
/// generator is claimed.
pub fn noise_draws(seed: u64) -> [(u64, u64); 4] {
    [
        (KEYFRAME_ENCODE_SEED, STREAM_POSTERIOR_EPS),
        (seed, STREAM_ANCHOR_NOISE),
        (seed, STREAM_VIDEO_NOISE),
        (seed, STREAM_AUDIO_ROWS),
    ]
}

/// An i2va request from a prompt and a decoded keyframe image: the keyframe stretched onto the canvas of its own
/// aspect with Lanczos-3 (model definition before_encoder.py:141-144, PIL `resize(LANCZOS)`), its Qwen3-VL
/// `pixel_values` and grid through the text encoder's image processor, the presentation, the aligned frame count,
/// the request `seed` and sampling (evaluations, video / audio shifts). The conditioning posterior is seeded with
/// `KEYFRAME_ENCODE_SEED`, fixed independently of `seed` by the model definition.
pub fn prepare_i2va(
    model: &Path,
    prompt: &str,
    keyframe: &crate::serve::image::RgbImage,
    (frames, seed): (usize, u64),
    (evaluations, shifts): (usize, (f64, f64)),
) -> Result<H3Request> {
    let spec = H3Spec::read(model)?;
    let encoder = model.join("text_encoder");
    let (height, width) = canvas_size(&spec, keyframe.width as f64, keyframe.height as f64)?;
    let canvas = crate::serve::image::RgbImage {
        width,
        height,
        pixels: crate::serve::image::resize_lanczos(
            &keyframe.pixels,
            [keyframe.height, keyframe.width, 3],
            [height, width],
            crate::serve::image::ResizeRounding::F64,
        )?,
    };
    let processor = crate::serve::image::Processor::load(&encoder, None)?
        .context("MiniMax-H3 text encoder without an image processor")?;
    let packed = processor.prepare_rgb(&canvas)?;
    let merge = (processor.merge() * processor.merge()) as usize;
    let image_tokens = packed
        .grids
        .iter()
        .map(|g| (g.t * g.h * g.w) as usize / merge)
        .collect::<Vec<_>>();
    let tokenizer = crate::serve::detok::Shared::load(&encoder)?;
    let (token_ids, text_tags) = presentation(&tokenizer, prompt, &image_tokens)?;
    let geometry = Geometry::resolve(&spec, height, width, frames)?;
    let planes = (0..3)
        .flat_map(|c| canvas.pixels.iter().skip(c).step_by(3).copied())
        .collect();
    Ok(H3Request {
        token_ids,
        text_tags,
        grids: packed.grids,
        pixel_values: packed.pixel_values,
        keyframe: planes,
        height,
        width,
        frames: geometry.frames,
        seed,
        evaluations,
        shifts,
    })
}

/// Generated media: u8 frames `[frames, height, width, 3]` and stereo PCM `[2, samples]` F32 at `sampling_rate`.
#[derive(Clone, Debug, PartialEq)]
pub struct H3Output {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub video: Vec<u8>,
    pub samples: usize,
    pub sampling_rate: usize,
    pub audio: Vec<f32>,
}
