use crate::core;
use crate::vae_tiling::split_tiles;
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module, ops};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    f32::consts::PI,
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::Arc,
};

pub trait VideoOutputSink {
    fn write_chunk(&mut self, first_frame: usize, rgb: &Tensor) -> Result<()>;

    fn finish(&mut self, total_frames: usize) -> Result<()> {
        let _ = total_frames;
        Ok(())
    }
}

pub struct PngFrameSink {
    directory: PathBuf,
    next_frame: usize,
    writer: Option<FrameWriter>,
}

/// PNG encoding and per-frame fsync on a writer thread, so the decoder's next
/// chunk runs while the previous chunk's files are being produced.
struct FrameWriter {
    sender: Option<std::sync::mpsc::SyncSender<(usize, Tensor)>>,
    handle: Option<std::thread::JoinHandle<()>>,
    failure: Arc<std::sync::Mutex<Option<String>>>,
}

impl FrameWriter {
    fn spawn(directory: PathBuf) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<(usize, Tensor)>(2);
        let failure = Arc::new(std::sync::Mutex::new(None));
        let thread_failure = Arc::clone(&failure);
        let handle = std::thread::spawn(move || {
            for (first_frame, chunk) in receiver {
                if let Err(error) = write_png_chunk(&directory, first_frame, &chunk) {
                    let mut failure = thread_failure.lock().expect("PNG writer lock");
                    if failure.is_none() {
                        *failure = Some(format!("{error:#}"));
                    }
                }
            }
        });
        Self {
            sender: Some(sender),
            handle: Some(handle),
            failure,
        }
    }

    fn send(&self, first_frame: usize, rgb: &Tensor) -> Result<()> {
        let sender = self
            .sender
            .as_ref()
            .context("PNG writer already finished")?;
        sender
            .send((first_frame, rgb.clone()))
            .map_err(|_| anyhow::anyhow!("PNG writer stopped early"))
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(sender) = self.sender.take() {
            drop(sender);
        }
        let handle = self.handle.take();
        if let Some(handle) = handle {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("PNG writer panicked"))?;
        }
        match self.failure.lock().expect("PNG writer lock").take() {
            Some(failure) => Err(anyhow::anyhow!(failure)),
            None => Ok(()),
        }
    }
}

impl Drop for FrameWriter {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

impl PngFrameSink {
    pub fn new(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        anyhow::ensure!(directory.is_dir(), "frame output is not a directory");
        Ok(Self {
            directory: directory.to_owned(),
            next_frame: 0,
            writer: Some(FrameWriter::spawn(directory.to_owned())),
        })
    }

    pub fn frames_written(&self) -> usize {
        self.next_frame
    }
}

impl VideoOutputSink for PngFrameSink {
    fn write_chunk(&mut self, first_frame: usize, rgb: &Tensor) -> Result<()> {
        anyhow::ensure!(
            first_frame == self.next_frame,
            "video chunks must be consecutive: expected frame {}, got {first_frame}",
            self.next_frame
        );
        let writer = self.writer.as_mut().context("PNG sink already finished")?;
        let frames = rgb
            .dims5()
            .context("video chunk must have shape [1, 3, frames, height, width]")?
            .2;
        writer.send(first_frame, rgb)?;
        self.next_frame = self
            .next_frame
            .checked_add(frames)
            .context("PNG frame index overflow")?;
        Ok(())
    }

    fn finish(&mut self, total_frames: usize) -> Result<()> {
        anyhow::ensure!(
            total_frames == self.next_frame,
            "decoder produced {total_frames} frames but PNG sink wrote {}",
            self.next_frame
        );
        if let Some(mut writer) = self.writer.take() {
            writer.finish()?;
        }
        sync_frame_directory(&self.directory)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct VideoVaeConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub spatial_downsample_factors: Vec<usize>,
    pub temporal_downsample_factors: Vec<usize>,
    pub decoder_num_layers: usize,
    pub decoder_num_attention_heads: usize,
    pub decoder_attention_head_dim: usize,
    pub decoder_num_register_tokens: usize,
    pub decoder_ffn_mult: usize,
    pub decoder_rope_theta: f32,
    pub decoder_rope_dim_ratio: f32,
    pub decoder_norm_eps: f64,
    pub clip_length: usize,
    pub token_drop: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl VideoVaeConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read visual VAE config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid visual VAE config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.class_name == "AutoencoderKLMiniMaxH3",
            "unsupported visual VAE class {}",
            self.class_name
        );
        for (name, value) in [
            ("out_channels", self.out_channels),
            ("latent_channels", self.latent_channels),
            ("decoder_num_layers", self.decoder_num_layers),
            (
                "decoder_num_attention_heads",
                self.decoder_num_attention_heads,
            ),
            (
                "decoder_attention_head_dim",
                self.decoder_attention_head_dim,
            ),
            ("clip_length", self.clip_length),
        ] {
            anyhow::ensure!(value > 0, "{name} must be non-zero");
        }
        anyhow::ensure!(
            self.latents_mean.len() == self.latent_channels
                && self.latents_std.len() == self.latent_channels,
            "visual latent statistics must have one value per channel"
        );
        anyhow::ensure!(
            self.latents_std.iter().all(|value| *value > 0.),
            "visual latent standard deviations must be positive"
        );
        let rope_dim = self.rope_dim();
        anyhow::ensure!(
            rope_dim > 0 && rope_dim.is_multiple_of(6),
            "visual decoder rotary dimension must be divisible by six"
        );
        anyhow::ensure!(
            self.token_drop < self.tokens_chunk_size()?,
            "token_drop must be smaller than the decoder chunk size"
        );
        Ok(())
    }

    pub fn spatial_ratio(&self) -> Result<usize> {
        checked_product(&self.spatial_downsample_factors, "spatial ratio")
    }

    pub fn temporal_ratio(&self) -> Result<usize> {
        checked_product(&self.temporal_downsample_factors, "temporal ratio")
    }

    pub fn decoded_frame_count(&self, latent_frames: usize) -> Result<usize> {
        anyhow::ensure!(latent_frames > 0, "video latent sequence is empty");
        let tokens_chunk = self.tokens_chunk_size()?;
        let temporal_ratio = self.temporal_ratio()?;
        let token_overlap = (tokens_chunk - self.token_drop % tokens_chunk) % tokens_chunk;
        let frame_pre_padding =
            (temporal_ratio - self.clip_length % temporal_ratio) % temporal_ratio;
        let num_tokens = latent_frames
            .checked_add(self.token_drop)
            .context("video temporal token count overflow")?;
        let pad_tokens = (tokens_chunk - num_tokens % tokens_chunk) % tokens_chunk;
        let padded_tokens = latent_frames
            .checked_add(pad_tokens)
            .context("padded video temporal token count overflow")?;
        let num_chunks = num_tokens
            .checked_add(pad_tokens)
            .context("video temporal chunk count overflow")?
            / tokens_chunk;
        let num_chunks = num_chunks
            .checked_sub(usize::from(self.token_drop > 0))
            .context("video latent sequence is too short to decode")?;
        anyhow::ensure!(
            num_chunks > 0,
            "video latent sequence is too short to decode"
        );

        let pad_frames = if pad_tokens > 0 {
            let intra_tail = self.clip_length % temporal_ratio;
            let tokens_before_pad = padded_tokens - pad_tokens;
            (0..pad_tokens).try_fold(0usize, |frames, index| {
                let added =
                    if intra_tail > 0 && (tokens_before_pad + index).is_multiple_of(tokens_chunk) {
                        intra_tail
                    } else {
                        temporal_ratio
                    };
                frames
                    .checked_add(added)
                    .context("video padding frame count overflow")
            })?
        } else {
            0
        };
        let chunk_frames = tokens_chunk
            .checked_mul(temporal_ratio)
            .context("decoded video chunk frame count overflow")?;
        let mut emitted = 0usize;
        let mut trailing_overlap = 0usize;
        for index in 0..num_chunks {
            let start = index
                .checked_mul(tokens_chunk)
                .context("video temporal chunk start overflow")?;
            let overlapped_tokens = tokens_chunk
                .checked_add(token_overlap)
                .context("overlapped video temporal chunk size overflow")?;
            let remaining_tokens = padded_tokens
                .checked_sub(start)
                .context("video temporal chunk starts beyond its padded input")?;
            let clip_tokens = overlapped_tokens.min(remaining_tokens);
            let decoded_frames = clip_tokens
                .checked_mul(temporal_ratio)
                .context("decoded video clip frame count overflow")?;
            let main_frames = chunk_frames
                .min(decoded_frames)
                .checked_sub(frame_pre_padding)
                .context("decoded temporal chunk is shorter than its leading padding")?;
            emitted = emitted
                .checked_add(main_frames)
                .context("decoded video frame count overflow")?;
            trailing_overlap = if decoded_frames > chunk_frames {
                chunk_frames
                    .min(decoded_frames - chunk_frames)
                    .checked_sub(frame_pre_padding)
                    .context("decoded temporal overlap is shorter than its leading padding")?
            } else {
                0
            };
        }
        emitted = emitted
            .checked_add(trailing_overlap)
            .context("decoded video frame count overflow")?;
        emitted
            .checked_sub(pad_frames)
            .context("video padding exceeds decoded frame count")
    }

    fn hidden_size(&self) -> usize {
        self.decoder_num_attention_heads * self.decoder_attention_head_dim
    }

    fn rope_dim(&self) -> usize {
        (self.decoder_attention_head_dim as f32 * self.decoder_rope_dim_ratio) as usize
    }

    fn tokens_chunk_size(&self) -> Result<usize> {
        Ok(self.clip_length.div_ceil(self.temporal_ratio()?))
    }
}

pub struct StreamedVideoVae {
    weights: ModelWeights,
    config: VideoVaeConfig,
    device: Device,
    attention_query_chunk_size: usize,
    tile_size: usize,
    tile_overlap: usize,
}

impl StreamedVideoVae {
    pub fn open(
        component_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
        attention_query_chunk_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            attention_query_chunk_size > 0,
            "attention query chunk size must be non-zero"
        );
        let component_dir = component_dir.as_ref();
        Ok(Self {
            config: VideoVaeConfig::from_file(component_dir.join("config.json"))?,
            weights: ModelWeights::open(component_dir, source, cache_policy)?,
            device,
            attention_query_chunk_size,
            tile_size: 256,
            tile_overlap: 64,
        })
    }

    pub fn config(&self) -> &VideoVaeConfig {
        &self.config
    }

    pub fn decode(&self, normalized_latents: &Tensor) -> Result<Tensor> {
        let mut chunks = Vec::new();
        self.decode_rgb_chunks(normalized_latents, |_, chunk| {
            chunks.push(chunk.clone());
            Ok(())
        })?;
        anyhow::ensure!(!chunks.is_empty(), "video decoder produced no frames");
        let refs = chunks.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 2).map_err(Into::into)
    }

    pub fn decode_to_sink(
        &self,
        normalized_latents: &Tensor,
        sink: &mut (impl VideoOutputSink + ?Sized),
    ) -> Result<usize> {
        let frames = self.decode_rgb_chunks(normalized_latents, |first_frame, chunk| {
            sink.write_chunk(first_frame, chunk)
        })?;
        sink.finish(frames)?;
        Ok(frames)
    }

    pub fn decode_to_png_frames(
        &self,
        normalized_latents: &Tensor,
        directory: impl AsRef<Path>,
    ) -> Result<usize> {
        anyhow::ensure!(
            normalized_latents.dim(0)? == 1,
            "PNG output requires a latent batch size of one"
        );
        let mut sink = PngFrameSink::new(directory)?;
        self.decode_to_sink(normalized_latents, &mut sink)
    }

    fn decode_rgb_chunks(
        &self,
        normalized_latents: &Tensor,
        mut emit: impl FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<usize> {
        let (batch, channels, frames, height, width) = normalized_latents.dims5()?;
        anyhow::ensure!(
            batch > 0 && frames > 0 && height > 0 && width > 0,
            "video latents must have non-zero batch, frame, height, and width dimensions"
        );
        anyhow::ensure!(
            channels == self.config.latent_channels,
            "video latents have {channels} channels, expected {}",
            self.config.latent_channels
        );
        anyhow::ensure!(
            normalized_latents.device().same_device(&self.device),
            "video latents are on a different device than the decoder"
        );
        let latents = normalized_latents.to_dtype(DType::F32)?;
        let mean = Tensor::from_vec(
            self.config.latents_mean.clone(),
            (1, channels, 1, 1, 1),
            &self.device,
        )?;
        let std = Tensor::from_vec(
            self.config.latents_std.clone(),
            (1, channels, 1, 1, 1),
            &self.device,
        )?;
        let latents = latents.broadcast_mul(&std)?.broadcast_add(&mean)?;
        let pixel_mean =
            Tensor::from_vec(vec![0.485f32, 0.456, 0.406], (1, 3, 1, 1, 1), &Device::Cpu)?;
        let pixel_std =
            Tensor::from_vec(vec![0.229f32, 0.224, 0.225], (1, 3, 1, 1, 1), &Device::Cpu)?;
        self.decode_temporal_chunks(&latents, |first_frame, decoded| {
            let decoded = decoded
                .broadcast_mul(&pixel_std)?
                .broadcast_add(&pixel_mean)?
                .clamp(0f32, 1f32)?;
            emit(first_frame, &decoded)
        })
    }

    fn decode_temporal_chunks(
        &self,
        latents: &Tensor,
        mut emit: impl FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<usize> {
        let tokens_chunk = self.config.tokens_chunk_size()?;
        let temporal_ratio = self.config.temporal_ratio()?;
        let token_overlap = (tokens_chunk - self.config.token_drop % tokens_chunk) % tokens_chunk;
        let frame_pre_padding =
            (temporal_ratio - self.config.clip_length % temporal_ratio) % temporal_ratio;
        let frame_overlap = token_overlap
            .checked_mul(temporal_ratio)
            .context("video frame overlap overflow")?
            .saturating_sub(frame_pre_padding);
        let original_tokens = latents.dim(2)?;
        let num_tokens = original_tokens + self.config.token_drop;
        let pad_tokens = (tokens_chunk - num_tokens % tokens_chunk) % tokens_chunk;
        let mut latents = latents.clone();
        if pad_tokens > 0 {
            let last = latents.narrow(2, original_tokens - 1, 1)?;
            let repeated = last.repeat((1, 1, pad_tokens, 1, 1))?;
            latents = Tensor::cat(&[&latents, &repeated], 2)?;
        }
        let num_chunks =
            (num_tokens + pad_tokens) / tokens_chunk - usize::from(self.config.token_drop > 0);
        anyhow::ensure!(
            num_chunks > 0,
            "video latent sequence is too short to decode"
        );
        let pad_frames = if pad_tokens > 0 {
            let intra_tail = self.config.clip_length % temporal_ratio;
            let tokens_before_pad = latents.dim(2)? - pad_tokens;
            (0..pad_tokens).try_fold(0usize, |frames, index| {
                let added =
                    if intra_tail > 0 && (tokens_before_pad + index).is_multiple_of(tokens_chunk) {
                        intra_tail
                    } else {
                        temporal_ratio
                    };
                frames
                    .checked_add(added)
                    .context("video padding frame count overflow")
            })?
        } else {
            0
        };
        let chunk_frames = tokens_chunk * temporal_ratio;
        let mut overlap: Option<Tensor> = None;
        let mut tail: Option<Tensor> = None;
        let mut emitted_frames = 0usize;
        for index in 0..num_chunks {
            let start = index * tokens_chunk;
            let clip_tokens = (tokens_chunk + token_overlap).min(latents.dim(2)? - start);
            let clip = latents.narrow(2, start, clip_tokens)?;
            let clip = self.decode_clip_tiled(&clip)?;
            for part in 0..(usize::from(self.config.token_drop > 0) + 1) {
                let frame_start = part * chunk_frames;
                if frame_start >= clip.dim(2)? {
                    continue;
                }
                let length = chunk_frames.min(clip.dim(2)? - frame_start);
                let mut chunk = clip.narrow(2, frame_start, length)?;
                if frame_pre_padding > 0 {
                    anyhow::ensure!(
                        chunk.dim(2)? > frame_pre_padding,
                        "decoded temporal chunk is shorter than its leading padding"
                    );
                    chunk =
                        chunk.narrow(2, frame_pre_padding, chunk.dim(2)? - frame_pre_padding)?;
                }
                chunk = chunk.to_device(&Device::Cpu)?;
                if part == 0 {
                    if let Some(previous) = overlap.take() {
                        chunk = blend(&previous, &chunk, frame_overlap, 2)?;
                    }
                    emit_without_padding_tail(
                        chunk,
                        pad_frames,
                        &mut tail,
                        &mut emitted_frames,
                        &mut emit,
                    )?;
                } else {
                    overlap = Some(chunk);
                }
            }
        }
        if let Some(overlap) = overlap {
            emit_without_padding_tail(
                overlap,
                pad_frames,
                &mut tail,
                &mut emitted_frames,
                &mut emit,
            )?;
        }
        if pad_frames > 0 {
            let tail = tail.context("video decoder produced fewer frames than its padding")?;
            anyhow::ensure!(
                tail.dim(2)? == pad_frames,
                "video decoder produced fewer frames than its {pad_frames}-frame padding"
            );
        }
        Ok(emitted_frames)
    }

    fn decode_clip_tiled(&self, latents: &Tensor) -> Result<Tensor> {
        let ratio = self.config.spatial_ratio()?;
        let pixel_height = latents.dim(3)? * ratio;
        let pixel_width = latents.dim(4)? * ratio;
        let (ys, y_lengths, y_overlaps) =
            split_tiles(pixel_height, self.tile_size, self.tile_overlap, ratio)?;
        let (xs, x_lengths, x_overlaps) =
            split_tiles(pixel_width, self.tile_size, self.tile_overlap, ratio)?;
        let mut cut = Vec::with_capacity(ys.len() * xs.len());
        for (&y, &y_len) in ys.iter().zip(&y_lengths) {
            for (&x, &x_len) in xs.iter().zip(&x_lengths) {
                cut.push(latents.narrow(3, y / ratio, y_len / ratio)?.narrow(
                    4,
                    x / ratio,
                    x_len / ratio,
                )?);
            }
        }
        let mut hidden = cut
            .iter()
            .map(|tile| self.decode_tile_prologue(tile))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = self.rotary(cut[0].dim(2)?, cut[0].dim(3)?, cut[0].dim(4)?)?;
        for layer in 0..self.config.decoder_num_layers {
            let names = Self::block_weight_names(layer);
            let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
            hidden = self.weights.with_group(&refs, &self.device, |weights| {
                hidden
                    .iter()
                    .map(|tile| self.block_with(weights, layer, tile, &cos, &sin))
                    .collect::<Result<Vec<_>>>()
            })?;
        }
        let mut decoded = hidden
            .iter()
            .zip(&cut)
            .map(|(tile, latents)| {
                Ok(self
                    .decode_tile_epilogue(tile, latents)?
                    .to_device(&Device::Cpu)?)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut rows = Vec::with_capacity(ys.len());
        for _ in &ys {
            let mut row = Vec::with_capacity(xs.len());
            for _ in &xs {
                row.push(
                    decoded
                        .next()
                        .context("decoded tile count does not match the tile grid")?,
                );
            }
            rows.push(row);
        }
        stitch_tiles(rows, &y_overlaps, &x_overlaps)
    }

    /// The per-tile reference order, kept so a test can prove the inverted
    /// loop above produces the same tensor. Production decoding uses the
    /// inverted order; this loads each layer's weights per tile.
    #[cfg(test)]
    fn decode_tile(&self, latents: &Tensor) -> Result<Tensor> {
        let (frames, height, width) = (latents.dim(2)?, latents.dim(3)?, latents.dim(4)?);
        let (cos, sin) = self.rotary(frames, height, width)?;
        let mut hidden = self.decode_tile_prologue(latents)?;
        for layer in 0..self.config.decoder_num_layers {
            hidden = self.block(&hidden, layer, &cos, &sin)?;
        }
        self.decode_tile_epilogue(&hidden, latents)
    }

    /// Everything before the decoder blocks: unpack, project in, append the
    /// register and class tokens.
    fn decode_tile_prologue(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, _, frames, height, width) = latents.dims5()?;
        let patches = frames * height * width;
        let hidden = latents
            .permute((0, 2, 3, 4, 1))?
            .contiguous()?
            .reshape((batch * patches, self.config.latent_channels))?;
        let mut hidden = self.channel_conv3d_1x1("post_quant_conv", &hidden)?;
        hidden = self.linear("decoder.proj_in", &hidden)?;
        let hidden_size = self.config.hidden_size();
        hidden = hidden.reshape((batch, patches, hidden_size))?;
        let register = self.weights.load("decoder.register_tokens", &self.device)?;
        let register =
            register.expand((batch, self.config.decoder_num_register_tokens, hidden_size))?;
        let class = Tensor::zeros((batch, 1, hidden_size), hidden.dtype(), &self.device)?;
        hidden = Tensor::cat(&[&hidden, &register, &class], 1)?;
        Ok(hidden)
    }

    /// Everything after the decoder blocks: final norm, project out, drop the
    /// appended tokens and unpatchify.
    fn decode_tile_epilogue(&self, hidden: &Tensor, latents: &Tensor) -> Result<Tensor> {
        let (batch, _, frames, height, width) = latents.dims5()?;
        let patches = frames * height * width;
        let mut hidden = hidden.clone();
        let names = ["decoder.norm_out.weight", "decoder.norm_out.bias"];
        hidden = self.weights.with_group(&names, &self.device, |weights| {
            core::layer_norm(
                &hidden,
                required(weights, names[0])?,
                required(weights, names[1])?,
                self.config.decoder_norm_eps,
            )
        })?;
        hidden = self.linear("decoder.proj_out", &hidden)?;
        hidden = hidden.narrow(1, 0, patches)?;
        let spatial = self.config.spatial_ratio()?;
        let temporal = self.config.temporal_ratio()?;
        hidden
            .reshape(
                &[
                    batch,
                    frames,
                    height,
                    width,
                    self.config.out_channels,
                    temporal,
                    spatial,
                    spatial,
                ][..],
            )?
            .permute(&[0usize, 4, 1, 5, 2, 6, 3, 7][..])?
            .contiguous()?
            .reshape((
                batch,
                self.config.out_channels,
                frames * temporal,
                height * spatial,
                width * spatial,
            ))
            .map_err(Into::into)
    }

    /// Every tensor one decoder block reads, in load order.
    ///
    /// The tiled decoder loads these once per layer and reuses them across the
    /// clip's tiles. Loading them per tile instead meant re-reading the whole
    /// 9.8 GB visual VAE once for every tile, which is free on a host whose
    /// page cache holds the file and storage-bound on one that cannot.
    fn block_weight_names(layer: usize) -> Vec<String> {
        let prefix = format!("decoder.transformer_blocks.{layer}");
        let mut names = vec![
            format!("{prefix}.norm1.weight"),
            format!("{prefix}.scale1"),
            format!("{prefix}.norm2.weight"),
            format!("{prefix}.scale2"),
        ];
        for projection in [
            format!("{prefix}.attn.to_q"),
            format!("{prefix}.attn.to_k"),
            format!("{prefix}.attn.to_v"),
            format!("{prefix}.attn.to_out.0"),
            format!("{prefix}.ff.net.0.proj"),
            format!("{prefix}.ff.net.2"),
        ] {
            names.push(format!("{projection}.weight"));
            names.push(format!("{projection}.bias"));
        }
        names
    }

    #[cfg(test)]
    fn block(&self, hidden: &Tensor, layer: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let names = Self::block_weight_names(layer);
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            self.block_with(weights, layer, hidden, cos, sin)
        })
    }

    /// One decoder block over already-materialized weights.
    ///
    /// The arithmetic is identical to loading each tensor as it is used; only
    /// where the bytes come from changes.
    fn block_with(
        &self,
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.transformer_blocks.{layer}");
        let linear = |name: &str, input: &Tensor| -> Result<Tensor> {
            Linear::new(
                required(weights, &format!("{name}.weight"))?.clone(),
                Some(required(weights, &format!("{name}.bias"))?.clone()),
            )
            .forward(input)
            .with_context(|| name.to_owned())
        };
        let normalized = core::rms_norm(
            &hidden.to_dtype(DType::F32)?,
            required(weights, &format!("{prefix}.norm1.weight"))?,
            self.config.decoder_norm_eps,
        )?;
        let heads = self.config.decoder_num_attention_heads;
        let head_dim = self.config.decoder_attention_head_dim;
        let (batch, sequence, _) = hidden.dims3()?;
        let project = |name: &str| -> Result<Tensor> {
            linear(&format!("{prefix}.attn.{name}"), &normalized)?
                .reshape((batch, sequence, heads, head_dim))
                .map_err(Into::into)
        };
        let query = apply_rotary(
            &rms_unit(&project("to_q")?, self.config.decoder_norm_eps)?,
            cos,
            sin,
        )?;
        let key = apply_rotary(
            &rms_unit(&project("to_k")?, self.config.decoder_norm_eps)?,
            cos,
            sin,
        )?;
        let value = project("to_v")?;
        let attended = scaled_dot_product(&query, &key, &value, self.attention_query_chunk_size)?;
        let attended = linear(&format!("{prefix}.attn.to_out.0"), &attended)?;
        let scale1 = required(weights, &format!("{prefix}.scale1"))?;
        let mut hidden = hidden.add(&attended.broadcast_mul(scale1)?)?;

        let normalized = core::rms_norm(
            &hidden.to_dtype(DType::F32)?,
            required(weights, &format!("{prefix}.norm2.weight"))?,
            self.config.decoder_norm_eps,
        )?;
        let projected = linear(&format!("{prefix}.ff.net.0.proj"), &normalized)?;
        anyhow::ensure!(
            projected.dim(D::Minus1)?
                == 2 * self.config.hidden_size() * self.config.decoder_ffn_mult,
            "visual decoder FFN width does not match config"
        );
        let inner = projected.dim(D::Minus1)? / 2;
        let values = projected.narrow(D::Minus1, 0, inner)?;
        let gates = projected.narrow(D::Minus1, inner, inner)?;
        let activated = values.mul(&ops::silu(&gates)?)?;
        let output = linear(&format!("{prefix}.ff.net.2"), &activated)?;
        let scale2 = required(weights, &format!("{prefix}.scale2"))?;
        hidden = hidden.add(&output.broadcast_mul(scale2)?)?;
        Ok(hidden)
    }

    fn linear(&self, prefix: &str, input: &Tensor) -> Result<Tensor> {
        let names = [format!("{prefix}.weight"), format!("{prefix}.bias")];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            Linear::new(
                required(weights, &names[0])?.clone(),
                Some(required(weights, &names[1])?.clone()),
            )
            .forward(input)
            .with_context(|| prefix.to_owned())
        })
    }

    fn channel_conv3d_1x1(&self, prefix: &str, input: &Tensor) -> Result<Tensor> {
        let names = [format!("{prefix}.weight"), format!("{prefix}.bias")];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            let weight = required(weights, &names[0])?;
            let out_channels = weight.dim(0)?;
            let in_channels = weight.dim(1)?;
            anyhow::ensure!(
                weight.dims() == [out_channels, in_channels, 1, 1, 1],
                "{prefix} is not a 1x1x1 convolution"
            );
            Linear::new(
                weight.reshape((out_channels, in_channels))?,
                Some(required(weights, &names[1])?.clone()),
            )
            .forward(input)
            .with_context(|| prefix.to_owned())
        })
    }

    fn rotary(&self, frames: usize, height: usize, width: usize) -> Result<(Tensor, Tensor)> {
        let rope_dim = self.config.rope_dim();
        let per_axis = rope_dim / 6;
        let mut inv_freq = Vec::with_capacity(per_axis);
        for index in 0..per_axis {
            let exponent = index as f32 / per_axis as f32;
            inv_freq.push(self.config.decoder_rope_theta.powf(-exponent));
        }
        let sequence = frames * height * width + self.config.decoder_num_register_tokens + 1;
        let mut cos = Vec::with_capacity(sequence * rope_dim);
        let mut sin = Vec::with_capacity(sequence * rope_dim);
        for t in 0..frames {
            for y in 0..height {
                for x in 0..width {
                    let coordinates = [
                        2. * ((t as f32 + 0.5) / frames as f32) - 1.,
                        2. * ((y as f32 + 0.5) / height as f32) - 1.,
                        2. * ((x as f32 + 0.5) / width as f32) - 1.,
                    ];
                    let mut angles = Vec::with_capacity(rope_dim / 2);
                    for coordinate in coordinates {
                        angles.extend(inv_freq.iter().map(|freq| 2. * PI * coordinate * freq));
                    }
                    for _ in 0..2 {
                        cos.extend(angles.iter().map(|angle| angle.cos()));
                        sin.extend(angles.iter().map(|angle| angle.sin()));
                    }
                }
            }
        }
        let suffix = (self.config.decoder_num_register_tokens + 1) * rope_dim;
        cos.extend(std::iter::repeat_n(1., suffix));
        sin.extend(std::iter::repeat_n(0., suffix));
        Ok((
            Tensor::from_vec(cos, (sequence, rope_dim), &self.device)?,
            Tensor::from_vec(sin, (sequence, rope_dim), &self.device)?,
        ))
    }
}

fn rms_unit(input: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = input.sqr()?.mean_keepdim(D::Minus1)?;
    input
        .broadcast_div(&(&mean + eps)?.sqrt()?)
        .map_err(Into::into)
}

fn scaled_dot_product(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    chunk: usize,
) -> Result<Tensor> {
    let (batch, sequence, heads, head_dim) = query.dims4()?;
    let query = query.transpose(1, 2)?.contiguous()?;
    let key = key.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
    let value = value.transpose(1, 2)?.contiguous()?;
    let mut chunks = Vec::with_capacity(sequence.div_ceil(chunk));
    let scale = 1. / (head_dim as f64).sqrt();
    for start in (0..sequence).step_by(chunk) {
        let length = chunk.min(sequence - start);
        let scores = query
            .narrow(2, start, length)?
            .affine(scale, 0.)?
            .matmul(&key)?;
        chunks.push(core::softmax_last_dim(&scores)?.matmul(&value)?);
    }
    let refs = chunks.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, sequence, heads * head_dim))
        .map_err(Into::into)
}

fn apply_rotary(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotary_dim = cos.dim(1)?;
    let head_dim = input.dim(D::Minus1)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;
    let rotating = input.narrow(D::Minus1, 0, rotary_dim)?;
    let half = rotary_dim / 2;
    let first = rotating.narrow(D::Minus1, 0, half)?;
    let second = rotating.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    let rotating = rotating
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)?;
    if rotary_dim == head_dim {
        Ok(rotating)
    } else {
        let pass = input.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[&rotating, &pass], D::Minus1).map_err(Into::into)
    }
}

fn stitch_tiles(
    tiles: Vec<Vec<Tensor>>,
    y_overlaps: &[usize],
    x_overlaps: &[usize],
) -> Result<Tensor> {
    let rows_count = tiles.len();
    let columns_count = tiles.first().context("no video tiles")?.len();
    anyhow::ensure!(columns_count > 0, "video tile rows must not be empty");
    anyhow::ensure!(
        tiles.iter().all(|row| row.len() == columns_count),
        "video tile grid must be rectangular"
    );
    anyhow::ensure!(
        y_overlaps.len() == rows_count.saturating_sub(1),
        "vertical overlap count does not match the tile grid"
    );
    anyhow::ensure!(
        x_overlaps.len() == columns_count.saturating_sub(1),
        "horizontal overlap count does not match the tile grid"
    );

    let mut vertical_columns = Vec::with_capacity(columns_count);
    for (column_index, _) in tiles[0].iter().enumerate() {
        let mut column_tiles = Vec::with_capacity(rows_count);
        for row_index in 0..rows_count {
            let mut tile = tiles[row_index][column_index].clone();
            if row_index > 0 {
                tile = blend(
                    &tiles[row_index - 1][column_index],
                    &tile,
                    y_overlaps[row_index - 1],
                    3,
                )?;
            }
            if row_index + 1 < rows_count {
                let retained = tile
                    .dim(3)?
                    .checked_sub(y_overlaps[row_index])
                    .context("vertical overlap exceeds decoded tile height")?;
                tile = tile.narrow(3, 0, retained)?;
            }
            column_tiles.push(tile);
        }
        let refs = column_tiles.iter().collect::<Vec<_>>();
        vertical_columns.push(Tensor::cat(&refs, 3)?);
    }
    drop(tiles);

    let mut result_columns = Vec::with_capacity(columns_count);
    for column_index in 0..columns_count {
        let mut column = vertical_columns[column_index].clone();
        if column_index > 0 {
            column = blend(
                &vertical_columns[column_index - 1],
                &column,
                x_overlaps[column_index - 1],
                4,
            )?;
        }
        if column_index + 1 < columns_count {
            let retained = column
                .dim(4)?
                .checked_sub(x_overlaps[column_index])
                .context("horizontal overlap exceeds decoded tile width")?;
            column = column.narrow(4, 0, retained)?;
        }
        result_columns.push(column);
    }
    let refs = result_columns.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 4).map_err(Into::into)
}

fn blend(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    let extent = extent.min(a.dim(dim)?).min(b.dim(dim)?);
    if extent == 0 {
        return Ok(b.clone());
    }
    let mut shape = vec![1; b.rank()];
    shape[dim] = extent;
    let weight_b = Tensor::arange(0f32, extent as f32, b.device())?
        .affine(1. / extent as f64, 0.)?
        .reshape(shape.clone())?;
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

fn emit_without_padding_tail(
    chunk: Tensor,
    pad_frames: usize,
    tail: &mut Option<Tensor>,
    emitted_frames: &mut usize,
    emit: &mut impl FnMut(usize, &Tensor) -> Result<()>,
) -> Result<()> {
    if pad_frames == 0 {
        emit(*emitted_frames, &chunk)?;
        *emitted_frames = emitted_frames
            .checked_add(chunk.dim(2)?)
            .context("decoded video frame count overflow")?;
        return Ok(());
    }

    let buffered = if let Some(previous) = tail.take() {
        Tensor::cat(&[&previous, &chunk], 2)?
    } else {
        chunk
    };
    let buffered_frames = buffered.dim(2)?;
    if buffered_frames <= pad_frames {
        *tail = Some(buffered);
        return Ok(());
    }

    let ready_frames = buffered_frames - pad_frames;
    let ready = buffered.narrow(2, 0, ready_frames)?;
    emit(*emitted_frames, &ready)?;
    *emitted_frames = emitted_frames
        .checked_add(ready_frames)
        .context("decoded video frame count overflow")?;
    *tail = Some(buffered.narrow(2, ready_frames, pad_frames)?);
    Ok(())
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
        .with_context(|| format!("missing visual VAE tensor {name}"))
}

pub fn write_png_frames(directory: impl AsRef<Path>, video: &Tensor) -> Result<usize> {
    let mut sink = PngFrameSink::new(directory)?;
    sink.write_chunk(0, video)?;
    let frames = sink.frames_written();
    sink.finish(frames)?;
    Ok(frames)
}

fn write_png_chunk(directory: &Path, first_frame: usize, video: &Tensor) -> Result<usize> {
    let (batch, channels, frames, height, width) = video.dims5()?;
    anyhow::ensure!(
        batch == 1 && channels == 3,
        "video must have shape [1, 3, frames, height, width]"
    );
    anyhow::ensure!(
        height <= u32::MAX as usize && width <= u32::MAX as usize,
        "PNG dimensions exceed the u32 format limit"
    );
    for local_frame in 0..frames {
        let values = video
            .narrow(2, local_frame, 1)?
            .permute((0, 2, 3, 4, 1))?
            .contiguous()?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let pixels = values
            .into_iter()
            .map(|value| (value.clamp(0., 1.) * 255.).round() as u8)
            .collect::<Vec<_>>();
        let frame = first_frame
            .checked_add(local_frame)
            .context("PNG frame index overflow")?;
        let path = directory.join(format!("frame_{frame:05}.png"));
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("failed to create frame {}", path.display()))?;
        let sync_file = file
            .try_clone()
            .with_context(|| format!("failed to duplicate frame {}", path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&pixels)?;
        writer.finish()?;
        sync_file
            .sync_all()
            .with_context(|| format!("failed to synchronize frame {}", path.display()))?;
    }
    Ok(frames)
}

/// Only Unix can make a directory entry itself durable; elsewhere the frames
/// are synchronized but their directory entry is not.
#[cfg(unix)]
fn sync_frame_directory(directory: &Path) -> Result<()> {
    fs::File::open(directory)
        .with_context(|| format!("failed to open frame directory {}", directory.display()))?
        .sync_all()
        .with_context(|| {
            format!(
                "failed to synchronize frame directory {}",
                directory.display()
            )
        })
}

#[cfg(not(unix))]
fn sync_frame_directory(_directory: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Shape, safetensors};
    use serde_json::json;
    use std::{collections::HashMap, fs};

    fn ones(shape: impl Into<Shape>) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn zeros(shape: impl Into<Shape>) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn patterned(shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| ((index * 13 % 41) as f32 - 20.) / 23.)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    fn score_scaled_dot_product_reference(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        chunk: usize,
    ) -> Tensor {
        let (batch, sequence, heads, head_dim) = query.dims4().unwrap();
        let query = query.transpose(1, 2).unwrap().contiguous().unwrap();
        let key = key
            .transpose(1, 2)
            .unwrap()
            .transpose(2, 3)
            .unwrap()
            .contiguous()
            .unwrap();
        let value = value.transpose(1, 2).unwrap().contiguous().unwrap();
        let mut chunks = Vec::with_capacity(sequence.div_ceil(chunk));
        for start in (0..sequence).step_by(chunk) {
            let length = chunk.min(sequence - start);
            let scores = query
                .narrow(2, start, length)
                .unwrap()
                .matmul(&key)
                .unwrap()
                .affine(1. / (head_dim as f64).sqrt(), 0.)
                .unwrap();
            chunks.push(
                core::softmax_last_dim(&scores)
                    .unwrap()
                    .matmul(&value)
                    .unwrap(),
            );
        }
        let refs = chunks.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 2)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((batch, sequence, heads * head_dim))
            .unwrap()
    }

    fn open_tiny_vae(directory: &Path, clip_length: usize, token_drop: usize) -> StreamedVideoVae {
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "_class_name": "AutoencoderKLMiniMaxH3",
                "out_channels": 3,
                "latent_channels": 1,
                "spatial_downsample_factors": [2],
                "temporal_downsample_factors": [2],
                "decoder_num_layers": 1,
                "decoder_num_attention_heads": 1,
                "decoder_attention_head_dim": 6,
                "decoder_num_register_tokens": 1,
                "decoder_ffn_mult": 1,
                "decoder_rope_theta": 100.0,
                "decoder_rope_dim_ratio": 1.0,
                "decoder_norm_eps": 1e-5,
                "clip_length": clip_length,
                "token_drop": token_drop,
                "latents_mean": [0.0],
                "latents_std": [1.0]
            }))
            .unwrap(),
        )
        .unwrap();
        let prefix = "decoder.transformer_blocks.0";
        let mut weights = HashMap::from([
            ("post_quant_conv.weight".to_owned(), ones((1, 1, 1, 1, 1))),
            ("post_quant_conv.bias".to_owned(), zeros(1)),
            ("decoder.proj_in.weight".to_owned(), ones((6, 1))),
            ("decoder.proj_in.bias".to_owned(), zeros(6)),
            ("decoder.register_tokens".to_owned(), zeros((1, 1, 6))),
            (format!("{prefix}.norm1.weight"), ones(6)),
            (format!("{prefix}.norm2.weight"), ones(6)),
            (format!("{prefix}.scale1"), zeros(6)),
            (format!("{prefix}.scale2"), zeros(6)),
            (format!("{prefix}.ff.net.0.proj.weight"), ones((12, 6))),
            (format!("{prefix}.ff.net.0.proj.bias"), zeros(12)),
            (format!("{prefix}.ff.net.2.weight"), ones((6, 6))),
            (format!("{prefix}.ff.net.2.bias"), zeros(6)),
            ("decoder.norm_out.weight".to_owned(), ones(6)),
            ("decoder.norm_out.bias".to_owned(), zeros(6)),
            ("decoder.proj_out.weight".to_owned(), ones((24, 6))),
            ("decoder.proj_out.bias".to_owned(), zeros(24)),
        ]);
        for name in ["to_q", "to_k", "to_v", "to_out.0"] {
            weights.insert(format!("{prefix}.attn.{name}.weight"), ones((6, 6)));
            weights.insert(format!("{prefix}.attn.{name}.bias"), zeros(6));
        }
        safetensors::save(
            &weights,
            directory.join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        StreamedVideoVae::open(
            directory,
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
            1,
        )
        .unwrap()
    }

    #[derive(Default)]
    struct ChunkLayoutSink {
        chunks: Vec<(usize, usize)>,
    }

    impl VideoOutputSink for ChunkLayoutSink {
        fn write_chunk(&mut self, first_frame: usize, rgb: &Tensor) -> Result<()> {
            self.chunks.push((first_frame, rgb.dim(2)?));
            Ok(())
        }

        fn finish(&mut self, total_frames: usize) -> Result<()> {
            anyhow::ensure!(
                self.chunks.iter().map(|(_, frames)| frames).sum::<usize>() == total_frames,
                "recorded chunks do not cover the decoded video"
            );
            Ok(())
        }
    }

    #[test]
    fn official_temporal_geometry_is_reproduced() {
        let config = VideoVaeConfig {
            class_name: "AutoencoderKLMiniMaxH3".to_owned(),
            out_channels: 3,
            latent_channels: 24,
            spatial_downsample_factors: vec![2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: vec![1, 2, 2, 1, 1, 1],
            decoder_num_layers: 36,
            decoder_num_attention_heads: 32,
            decoder_attention_head_dim: 64,
            decoder_num_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            latents_mean: vec![0.; 24],
            latents_std: vec![1.; 24],
        };
        config.validate().unwrap();
        assert_eq!(config.spatial_ratio().unwrap(), 16);
        assert_eq!(config.temporal_ratio().unwrap(), 4);
        assert_eq!(config.tokens_chunk_size().unwrap(), 5);
        assert_eq!(config.decoded_frame_count(37).unwrap(), 124);
        assert_eq!(config.decoded_frame_count(72).unwrap(), 243);
    }

    #[test]
    fn blending_preserves_requested_extent() {
        let a = Tensor::zeros((1, 1, 1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::ones((1, 1, 1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let blended = blend(&a, &b, 2, 4).unwrap();
        assert_eq!(blended.dims(), b.dims());
        assert_eq!(
            blended.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0., 0.5, 1., 1.]
        );
    }

    #[test]
    fn official_head_dim_query_scaling_is_bitwise_stable_in_f32() {
        let head_dim = 64;
        let query = patterned((1, 5, 2, head_dim)).affine(0.8, 0.1).unwrap();
        let key = patterned((1, 5, 2, head_dim)).affine(-0.6, 0.2).unwrap();
        let value = patterned((1, 5, 2, head_dim));
        let expected = score_scaled_dot_product_reference(&query, &key, &value, 2)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = scaled_dot_product(&query, &key, &value, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn official_head_dim_query_scaling_is_bitwise_stable_in_bf16_emulation() {
        let head_dim = 64;
        let bf16_round = |tensor: Tensor| {
            tensor
                .to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };
        let query = bf16_round(patterned((1, 2, 3, head_dim)).affine(0.8, 0.1).unwrap());
        let key_t = bf16_round(patterned((1, 2, head_dim, 5)).affine(-0.6, 0.2).unwrap());
        let scale = 1. / (head_dim as f64).sqrt();
        let expected = bf16_round(
            bf16_round(query.matmul(&key_t).unwrap())
                .affine(scale, 0.)
                .unwrap(),
        );
        let actual = bf16_round(
            bf16_round(query.affine(scale, 0.).unwrap())
                .matmul(&key_t)
                .unwrap(),
        );
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn tile_corner_matches_an_explicit_vertical_then_horizontal_reference() {
        let shape = (1, 1, 1, 4, 4);
        let top_left = Tensor::full(0f32, shape, &Device::Cpu).unwrap();
        let top_right = Tensor::full(100f32, shape, &Device::Cpu).unwrap();
        let bottom_left = Tensor::full(10f32, shape, &Device::Cpu).unwrap();
        let bottom_right = Tensor::full(110f32, shape, &Device::Cpu).unwrap();

        let actual = stitch_tiles(
            vec![
                vec![top_left.clone(), top_right.clone()],
                vec![bottom_left.clone(), bottom_right.clone()],
            ],
            &[2],
            &[2],
        )
        .unwrap();

        let bottom_left = blend(&top_left, &bottom_left, 2, 3).unwrap();
        let bottom_right = blend(&top_right, &bottom_right, 2, 3).unwrap();
        let top_left = top_left.narrow(3, 0, 2).unwrap();
        let top_right = top_right.narrow(3, 0, 2).unwrap();
        let left_column = Tensor::cat(&[&top_left, &bottom_left], 3).unwrap();
        let right_column = Tensor::cat(&[&top_right, &bottom_right], 3).unwrap();
        let right_column = blend(&left_column, &right_column, 2, 4).unwrap();
        let left_column = left_column.narrow(4, 0, 2).unwrap();
        let expected = Tensor::cat(&[&left_column, &right_column], 4).unwrap();

        assert_eq!(actual.dims(), &[1, 1, 1, 6, 6]);
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual, expected);

        let width = 6;
        let corner = [
            actual[2 * width + 2],
            actual[2 * width + 3],
            actual[3 * width + 2],
            actual[3 * width + 3],
        ];
        assert_eq!(corner, [0., 50., 5., 55.]);
        assert_eq!(corner[1] - corner[0], corner[3] - corner[2]);
        assert_eq!(corner[2] - corner[0], corner[3] - corner[1]);
    }

    #[test]
    fn inverted_layer_loop_matches_the_per_tile_reference_order() {
        let dir = tempfile::tempdir().unwrap();
        let vae = open_tiny_vae(dir.path(), 1, 0);
        let latents = ones((1, 1, 1, 1, 1));

        let reference = vae.decode_tile(&latents).unwrap();
        let mut hidden = vae.decode_tile_prologue(&latents).unwrap();
        let (cos, sin) = vae.rotary(1, 1, 1).unwrap();
        for layer in 0..vae.config.decoder_num_layers {
            let names = StreamedVideoVae::block_weight_names(layer);
            let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
            hidden = vae
                .weights
                .with_group(&refs, &vae.device, |weights| {
                    vae.block_with(weights, layer, &hidden, &cos, &sin)
                })
                .unwrap();
        }
        let inverted = vae.decode_tile_epilogue(&hidden, &latents).unwrap();

        assert_eq!(reference.dims(), inverted.dims());
        assert_eq!(
            reference.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            inverted.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "grouped weight loading changed the decoded tile"
        );
    }

    #[test]
    fn runs_a_tiny_streamed_vit_tile_decode() {
        let dir = tempfile::tempdir().unwrap();
        let vae = open_tiny_vae(dir.path(), 1, 0);
        let latents = ones((1, 1, 1, 1, 1));
        let decoded = vae.decode_tile(&latents).unwrap();
        assert_eq!(decoded.dims(), &[1, 3, 2, 2, 2]);
        assert!(
            decoded
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        let video = vae.decode(&latents).unwrap();
        assert_eq!(video.dims(), &[1, 3, 1, 2, 2]);
        let frame_dir = dir.path().join("frames");
        fs::create_dir(&frame_dir).unwrap();
        assert_eq!(write_png_frames(&frame_dir, &video).unwrap(), 1);
        let signature = fs::read(frame_dir.join("frame_00000.png")).unwrap();
        assert_eq!(&signature[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[cfg(unix)]
    #[test]
    fn png_output_rejects_existing_symlinks_without_modifying_their_target() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.bin");
        let frames = directory.path().join("frames");
        fs::write(&target, b"untouched").unwrap();
        fs::create_dir(&frames).unwrap();
        symlink(&target, frames.join("frame_00000.png")).unwrap();
        let video = Tensor::zeros((1, 3, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
        assert!(write_png_frames(&frames, &video).is_err());
        assert_eq!(fs::read(target).unwrap(), b"untouched");
    }

    #[test]
    fn async_png_sink_matches_inline_bytes_and_reports_worker_errors() {
        let directory = tempfile::tempdir().unwrap();
        let frames = directory.path().join("frames");
        fs::create_dir(&frames).unwrap();
        let values: Vec<f32> = (0..2 * 3 * 2 * 3)
            .map(|index| index as f32 / 36.0)
            .collect();
        let whole = Tensor::from_vec(values, (1, 3, 2, 2, 3), &Device::Cpu).unwrap();
        let split = [
            whole.narrow(2, 0, 1).unwrap(),
            whole.narrow(2, 1, 1).unwrap(),
        ];
        let inline = tempfile::tempdir().unwrap();
        let inline_dir = inline.path().join("frames");
        fs::create_dir(&inline_dir).unwrap();
        assert_eq!(write_png_frames(&inline_dir, &whole).unwrap(), 2);
        let mut sink = PngFrameSink::new(&frames).unwrap();
        sink.write_chunk(0, &split[0]).unwrap();
        sink.write_chunk(1, &split[1]).unwrap();
        sink.finish(2).unwrap();
        for frame in 0..2 {
            let name = format!("frame_{frame:05}.png");
            assert_eq!(
                fs::read(frames.join(&name)).unwrap(),
                fs::read(inline_dir.join(&name)).unwrap()
            );
        }
        let mut conflict = PngFrameSink::new(&frames).unwrap();
        conflict.write_chunk(0, &split[0]).unwrap();
        let error = conflict.finish(1).unwrap_err().to_string();
        assert!(error.contains("failed to create frame"), "{error}");
    }

    #[test]
    fn streams_overlapped_padded_chunks_directly_to_png() {
        let dir = tempfile::tempdir().unwrap();
        let vae = open_tiny_vae(dir.path(), 3, 1);
        let latents = ones((1, 1, 4, 1, 1));
        assert_eq!(vae.config().decoded_frame_count(4).unwrap(), 6);
        let frame_dir = dir.path().join("streamed_frames");
        fs::create_dir(&frame_dir).unwrap();

        let mut layout = ChunkLayoutSink::default();
        assert_eq!(vae.decode_to_sink(&latents, &mut layout).unwrap(), 6);
        assert!(layout.chunks.len() > 1);
        assert!(layout.chunks.iter().all(|(_, frames)| *frames < 6));
        for chunks in layout.chunks.windows(2) {
            assert_eq!(chunks[0].0 + chunks[0].1, chunks[1].0);
        }

        assert_eq!(vae.decode_to_png_frames(&latents, &frame_dir).unwrap(), 6);
        let mut names = fs::read_dir(&frame_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            (0..6)
                .map(|frame| format!("frame_{frame:05}.png"))
                .collect::<Vec<_>>()
        );
        for name in names {
            let bytes = fs::read(frame_dir.join(name)).unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
    }
}
