use crate::core;
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Activation, Conv1d, Conv1dConfig, Linear, Module};
use ff_core::weights::{CachePolicy, ModelWeights, WeightAccessStats, WeightSource};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};

const LAYER_NORM_EPS: f64 = 1e-5;
const DEFAULT_ATTENTION_QUERY_CHUNK_SIZE: usize = 128;
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];

#[derive(Clone, Debug, Deserialize)]
pub struct AudioVaeEncoderConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub latent_channels: usize,
    pub num_attention_heads: usize,
    pub sampling_rate: u32,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl AudioVaeEncoderConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read audio VAE config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid audio VAE config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.class_name == "AutoencoderKLMiniMaxH3Audio",
            "unsupported audio VAE class {}",
            self.class_name
        );
        anyhow::ensure!(self.encoder_dim > 0, "encoder_dim must be non-zero");
        anyhow::ensure!(
            !self.encoder_rates.is_empty() && self.encoder_rates.iter().all(|rate| *rate > 0),
            "encoder rates must be non-empty and positive"
        );
        anyhow::ensure!(self.latent_dim > 0, "latent_dim must be non-zero");
        anyhow::ensure!(self.latent_channels > 0, "latent_channels must be non-zero");
        anyhow::ensure!(
            self.num_attention_heads > 0
                && self.latent_dim.is_multiple_of(self.num_attention_heads),
            "latent_dim must be divisible by num_attention_heads"
        );
        anyhow::ensure!(self.sampling_rate > 0, "sampling_rate must be non-zero");
        anyhow::ensure!(
            self.latents_mean.len() == self.latent_channels
                && self.latents_std.len() == self.latent_channels,
            "audio latent statistics must have one value per latent channel"
        );
        anyhow::ensure!(
            self.latents_mean.iter().all(|value| value.is_finite())
                && self
                    .latents_std
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.),
            "audio latent statistics must be finite and standard deviations positive"
        );
        self.hop_length()?;
        self.final_encoder_channels()?;
        Ok(())
    }

    pub fn hop_length(&self) -> Result<usize> {
        self.encoder_rates.iter().try_fold(1usize, |value, rate| {
            value
                .checked_mul(*rate)
                .context("audio encoder hop length overflow")
        })
    }

    fn final_encoder_channels(&self) -> Result<usize> {
        self.encoder_rates
            .iter()
            .try_fold(self.encoder_dim, |channels, _| {
                channels
                    .checked_mul(2)
                    .context("audio encoder channel count overflow")
            })
    }
}

/// Streamed MiniMax-H3 audio-VAE encoder used by Ref2VA reference soundtracks.
///
/// The public boundary deliberately accepts only waveforms already carrying the
/// checkpoint's sample rate. Media decoding, duration truncation, and resampling
/// belong to the reference-input layer; silently treating another rate as 32 kHz
/// would condition the model at the wrong speed.
pub struct StreamedAudioVaeEncoder {
    weights: ModelWeights,
    config: AudioVaeEncoderConfig,
    device: Device,
    attention_query_chunk_size: usize,
}

impl StreamedAudioVaeEncoder {
    pub fn open(
        component_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
    ) -> Result<Self> {
        let component_dir = component_dir.as_ref();
        let config = AudioVaeEncoderConfig::from_file(component_dir.join("config.json"))?;
        let weights = ModelWeights::open(component_dir, source, cache_policy)?;
        validate_weight_inventory(&weights, &config)?;
        Ok(Self {
            weights,
            config,
            device,
            attention_query_chunk_size: DEFAULT_ATTENTION_QUERY_CHUNK_SIZE,
        })
    }

    pub fn with_attention_query_chunk_size(mut self, chunk_size: usize) -> Result<Self> {
        anyhow::ensure!(
            chunk_size > 0,
            "attention query chunk size must be non-zero"
        );
        self.attention_query_chunk_size = chunk_size;
        Ok(self)
    }

    pub fn config(&self) -> &AudioVaeEncoderConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn access_stats(&self) -> WeightAccessStats {
        self.weights.access_stats()
    }

    /// Encode a mono or stereo `[channels, samples]` waveform to the raw
    /// posterior mode `[2, latent_channels, latent_frames]`.
    ///
    /// Mono is duplicated to stereo, and the sample tail is zero-padded to the
    /// encoder hop (800 samples in the released checkpoint). The posterior log
    /// standard-deviation head is intentionally not evaluated: Ref2VA consumes
    /// `posterior.mode()`, which is exactly the output of `mean_proj`.
    pub fn encode_posterior_mode(&self, waveform: &Tensor, sample_rate: u32) -> Result<Tensor> {
        let (channels, samples) = waveform
            .dims2()
            .context("reference audio must be [channels, samples]")?;
        anyhow::ensure!(
            channels == 1 || channels == 2,
            "reference audio must be mono or stereo, got {channels} channels"
        );
        anyhow::ensure!(samples > 0, "reference audio must contain samples");
        anyhow::ensure!(
            sample_rate == self.config.sampling_rate,
            "reference audio carries {sample_rate} Hz, expected {} Hz; resample it before encoding",
            self.config.sampling_rate
        );
        anyhow::ensure!(
            waveform.device().same_device(&self.device),
            "reference audio is on a different device than the encoder"
        );
        anyhow::ensure!(
            matches!(waveform.dtype(), DType::F32 | DType::F16 | DType::BF16),
            "reference audio must use a floating-point dtype"
        );

        let waveform = waveform.to_dtype(DType::F32)?;
        let waveform = if channels == 1 {
            waveform.expand((2, samples))?.contiguous()?
        } else {
            waveform
        };
        let hop = self.config.hop_length()?;
        let padded_blocks = samples
            .checked_add(hop - 1)
            .context("audio sample count overflow")?
            / hop;
        let padded_samples = padded_blocks
            .checked_mul(hop)
            .context("padded audio sample count overflow")?;
        let right_pad = padded_samples - samples;
        let waveform = if right_pad > 0 {
            waveform.pad_with_zeros(D::Minus1, 0, right_pad)?
        } else {
            waveform
        };
        let mut hidden = waveform.unsqueeze(1)?;

        hidden = self.weight_norm_conv("encoder.block.0", &hidden, 3, 1, 1)?;
        let mut stage_channels = self.config.encoder_dim;
        for (stage, &stride) in self.config.encoder_rates.iter().enumerate() {
            let prefix = format!("encoder.block.{}", stage + 1);
            for (unit, dilation) in RESIDUAL_DILATIONS.into_iter().enumerate() {
                hidden =
                    self.residual_unit(&hidden, &format!("{prefix}.block.{unit}.block"), dilation)?;
            }
            hidden = self.snake(&hidden, &format!("{prefix}.block.3.alpha"))?;
            stage_channels = stage_channels
                .checked_mul(2)
                .context("audio encoder channel count overflow")?;
            hidden = self.weight_norm_conv(
                &format!("{prefix}.block.4"),
                &hidden,
                stride.div_ceil(2),
                stride,
                1,
            )?;
            anyhow::ensure!(
                hidden.dim(1)? == stage_channels,
                "audio encoder stage {stage} produced {} channels, expected {stage_channels}",
                hidden.dim(1)?
            );
        }

        let final_activation = self.config.encoder_rates.len() + 1;
        hidden = self.snake(&hidden, &format!("encoder.block.{final_activation}.alpha"))?;
        hidden = self.weight_norm_conv(
            &format!("encoder.block.{}", final_activation + 1),
            &hidden,
            1,
            1,
            1,
        )?;
        anyhow::ensure!(
            hidden.dims() == [2, self.config.latent_dim, padded_samples / hop],
            "audio encoder trunk produced {:?}, expected [2, {}, {}]",
            hidden.dims(),
            self.config.latent_dim,
            padded_samples / hop
        );

        hidden = self.pre_block(&hidden.transpose(1, 2)?)?.transpose(1, 2)?;
        let posterior_mode = self.plain_conv("mean_proj", &hidden, 0, 1, 1)?;
        anyhow::ensure!(
            posterior_mode.dims() == [2, self.config.latent_channels, padded_samples / hop],
            "audio posterior mode has unexpected shape {:?}",
            posterior_mode.dims()
        );
        posterior_mode.to_dtype(DType::F32).map_err(Into::into)
    }

    /// Encode and normalize a reference soundtrack into the channel-major rows
    /// consumed by the H3 transformer: `[2 * latent_frames, latent_channels]`.
    pub fn encode_condition_rows(&self, waveform: &Tensor, sample_rate: u32) -> Result<Tensor> {
        let latents = self.encode_posterior_mode(waveform, sample_rate)?;
        let (_, channels, frames) = latents.dims3()?;
        let mean = Tensor::from_vec(
            self.config.latents_mean.clone(),
            (1, channels, 1),
            &self.device,
        )?;
        let std = Tensor::from_vec(
            self.config.latents_std.clone(),
            (1, channels, 1),
            &self.device,
        )?;
        latents
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .permute((0, 2, 1))?
            .contiguous()?
            .reshape((2 * frames, channels))
            .map_err(Into::into)
    }

    fn residual_unit(&self, input: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let mut residual = self.snake(input, &format!("{prefix}.0.alpha"))?;
        residual =
            self.weight_norm_conv(&format!("{prefix}.1"), &residual, 3 * dilation, 1, dilation)?;
        residual = self.snake(&residual, &format!("{prefix}.2.alpha"))?;
        residual = self.weight_norm_conv(&format!("{prefix}.3"), &residual, 0, 1, 1)?;
        input.add(&residual).map_err(Into::into)
    }

    fn snake(&self, input: &Tensor, alpha_name: &str) -> Result<Tensor> {
        self.weights
            .with_group(&[alpha_name], &self.device, |weights| {
                let alpha = required(weights, alpha_name)?.to_dtype(input.dtype())?;
                let phase = input.broadcast_mul(&alpha)?;
                input
                    .add(
                        &phase
                            .sin()?
                            .sqr()?
                            .broadcast_div(&alpha.affine(1., 1e-9)?)?,
                    )
                    .map_err(Into::into)
            })
    }

    fn pre_block(&self, input: &Tensor) -> Result<Tensor> {
        let projected = self.layer_norm_linear(input, "pre_block.norm3", "pre_block.proj")?;
        let normalized = self.layer_norm(input, "pre_block.norm1")?;
        let attended = self.causal_attention(&normalized)?;
        let hidden = projected.add(&attended)?;
        let normalized = self.layer_norm(&hidden, "pre_block.norm2")?;
        let mlp = self.geglu_mlp(&normalized)?;
        hidden.add(&mlp).map_err(Into::into)
    }

    fn causal_attention(&self, input: &Tensor) -> Result<Tensor> {
        let names = [
            "pre_block.attn.qkv.weight",
            "pre_block.attn.q_bias",
            "pre_block.attn.zero_k_bias",
            "pre_block.attn.v_bias",
            "pre_block.attn.proj.weight",
            "pre_block.attn.proj.bias",
        ];
        self.weights.with_group(&names, &self.device, |weights| {
            let (batch, sequence, width) = input.dims3()?;
            anyhow::ensure!(
                width == self.config.latent_dim,
                "audio attention input width is {width}, expected {}",
                self.config.latent_dim
            );
            let heads = self.config.num_attention_heads;
            let head_dim = width / heads;
            let bias = Tensor::cat(
                &[
                    required(weights, names[1])?,
                    required(weights, names[2])?,
                    required(weights, names[3])?,
                ],
                0,
            )?;
            let qkv = Linear::new(required(weights, names[0])?.clone(), Some(bias))
                .forward(&input.to_dtype(required(weights, names[0])?.dtype())?)?
                .reshape((batch, sequence, 3, heads, head_dim))?;
            let query = qkv
                .narrow(2, 0, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()?;
            let key = qkv
                .narrow(2, 1, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()?;
            let value = qkv
                .narrow(2, 2, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()?;
            let key_t = key.transpose(2, 3)?.contiguous()?;
            let mask = causal_mask(sequence, query.dtype(), &self.device)?;
            let mut chunks = Vec::with_capacity(sequence.div_ceil(self.attention_query_chunk_size));
            for start in (0..sequence).step_by(self.attention_query_chunk_size) {
                let length = self.attention_query_chunk_size.min(sequence - start);
                let query = query
                    .narrow(2, start, length)?
                    .affine(1. / (head_dim as f64).sqrt(), 0.)?;
                let scores = query
                    .matmul(&key_t)?
                    .broadcast_add(&mask.narrow(2, start, length)?)?;
                chunks.push(core::softmax_last_dim(&scores)?.matmul(&value)?);
            }
            let refs = chunks.iter().collect::<Vec<_>>();
            let attended = Tensor::cat(&refs, 2)?.transpose(1, 2)?.mean(2)?;
            let pooled = adaptive_avg_pool_last_dim(&attended, self.config.latent_channels)?;
            Linear::new(
                required(weights, names[4])?.clone(),
                Some(required(weights, names[5])?.clone()),
            )
            .forward(&pooled.to_dtype(required(weights, names[4])?.dtype())?)
            .map_err(Into::into)
        })
    }

    fn geglu_mlp(&self, input: &Tensor) -> Result<Tensor> {
        let names = [
            "pre_block.mlp.norm.weight",
            "pre_block.mlp.norm.bias",
            "pre_block.mlp.w0.weight",
            "pre_block.mlp.w0.bias",
            "pre_block.mlp.w1.weight",
            "pre_block.mlp.w1.bias",
            "pre_block.mlp.w2.weight",
            "pre_block.mlp.w2.bias",
        ];
        self.weights.with_group(&names, &self.device, |weights| {
            let normalized = core::layer_norm(
                input,
                required(weights, names[0])?,
                required(weights, names[1])?,
                LAYER_NORM_EPS,
            )?;
            let gate = Linear::new(
                required(weights, names[2])?.clone(),
                Some(required(weights, names[3])?.clone()),
            )
            .forward(&normalized.to_dtype(required(weights, names[2])?.dtype())?)?;
            let gate = Activation::GeluPytorchTanh.forward(&gate)?;
            let value = Linear::new(
                required(weights, names[4])?.clone(),
                Some(required(weights, names[5])?.clone()),
            )
            .forward(&normalized.to_dtype(required(weights, names[4])?.dtype())?)?;
            let activated = gate.mul(&value)?;
            Linear::new(
                required(weights, names[6])?.clone(),
                Some(required(weights, names[7])?.clone()),
            )
            .forward(&activated.to_dtype(required(weights, names[6])?.dtype())?)
            .map_err(Into::into)
        })
    }

    fn layer_norm(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let names = [format!("{prefix}.weight"), format!("{prefix}.bias")];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            core::layer_norm(
                input,
                required(weights, &names[0])?,
                required(weights, &names[1])?,
                LAYER_NORM_EPS,
            )
        })
    }

    fn layer_norm_linear(&self, input: &Tensor, norm: &str, linear: &str) -> Result<Tensor> {
        let names = [
            format!("{norm}.weight"),
            format!("{norm}.bias"),
            format!("{linear}.weight"),
            format!("{linear}.bias"),
        ];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            let normalized = core::layer_norm(
                input,
                required(weights, &names[0])?,
                required(weights, &names[1])?,
                LAYER_NORM_EPS,
            )?;
            Linear::new(
                required(weights, &names[2])?.clone(),
                Some(required(weights, &names[3])?.clone()),
            )
            .forward(&normalized.to_dtype(required(weights, &names[2])?.dtype())?)
            .map_err(Into::into)
        })
    }

    fn plain_conv(
        &self,
        prefix: &str,
        input: &Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let names = [format!("{prefix}.weight"), format!("{prefix}.bias")];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            Conv1d::new(
                required(weights, &names[0])?.clone(),
                Some(required(weights, &names[1])?.clone()),
                Conv1dConfig {
                    padding,
                    stride,
                    dilation,
                    ..Default::default()
                },
            )
            .forward(input)
            .map_err(Into::into)
        })
    }

    fn weight_norm_conv(
        &self,
        prefix: &str,
        input: &Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let names = [
            format!("{prefix}.weight_g"),
            format!("{prefix}.weight_v"),
            format!("{prefix}.bias"),
        ];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            let weight =
                normalized_weight(required(weights, &names[0])?, required(weights, &names[1])?)?;
            Conv1d::new(
                weight,
                Some(required(weights, &names[2])?.clone()),
                Conv1dConfig {
                    padding,
                    stride,
                    dilation,
                    ..Default::default()
                },
            )
            .forward(input)
            .map_err(Into::into)
        })
    }
}

fn validate_weight_inventory(weights: &ModelWeights, config: &AudioVaeEncoderConfig) -> Result<()> {
    let mut expected = Vec::<(String, Vec<usize>)>::new();
    push_weight_norm_conv_shapes(&mut expected, "encoder.block.0", config.encoder_dim, 1, 7);

    let mut channels = config.encoder_dim;
    for (stage, &stride) in config.encoder_rates.iter().enumerate() {
        let prefix = format!("encoder.block.{}", stage + 1);
        for unit in 0..RESIDUAL_DILATIONS.len() {
            let prefix = format!("{prefix}.block.{unit}.block");
            expected.push((format!("{prefix}.0.alpha"), vec![1, channels, 1]));
            push_weight_norm_conv_shapes(
                &mut expected,
                &format!("{prefix}.1"),
                channels,
                channels,
                7,
            );
            expected.push((format!("{prefix}.2.alpha"), vec![1, channels, 1]));
            push_weight_norm_conv_shapes(
                &mut expected,
                &format!("{prefix}.3"),
                channels,
                channels,
                1,
            );
        }
        expected.push((format!("{prefix}.block.3.alpha"), vec![1, channels, 1]));
        let output_channels = channels
            .checked_mul(2)
            .context("audio encoder inventory channel count overflow")?;
        push_weight_norm_conv_shapes(
            &mut expected,
            &format!("{prefix}.block.4"),
            output_channels,
            channels,
            stride
                .checked_mul(2)
                .context("audio encoder inventory kernel size overflow")?,
        );
        channels = output_channels;
    }
    anyhow::ensure!(
        channels == config.final_encoder_channels()?,
        "audio encoder inventory derived inconsistent final channels"
    );
    let final_activation = config.encoder_rates.len() + 1;
    expected.push((
        format!("encoder.block.{final_activation}.alpha"),
        vec![1, channels, 1],
    ));
    push_weight_norm_conv_shapes(
        &mut expected,
        &format!("encoder.block.{}", final_activation + 1),
        config.latent_dim,
        channels,
        3,
    );

    for prefix in ["pre_block.norm1", "pre_block.norm3"] {
        push_layer_norm_shapes(&mut expected, prefix, config.latent_dim);
    }
    push_layer_norm_shapes(&mut expected, "pre_block.norm2", config.latent_channels);
    push_layer_norm_shapes(&mut expected, "pre_block.mlp.norm", config.latent_channels);
    expected.extend([
        (
            "pre_block.proj.weight".to_owned(),
            vec![config.latent_channels, config.latent_dim],
        ),
        (
            "pre_block.proj.bias".to_owned(),
            vec![config.latent_channels],
        ),
        (
            "pre_block.attn.qkv.weight".to_owned(),
            vec![
                config
                    .latent_dim
                    .checked_mul(3)
                    .context("audio QKV width overflow")?,
                config.latent_dim,
            ],
        ),
        ("pre_block.attn.q_bias".to_owned(), vec![config.latent_dim]),
        (
            "pre_block.attn.zero_k_bias".to_owned(),
            vec![config.latent_dim],
        ),
        ("pre_block.attn.v_bias".to_owned(), vec![config.latent_dim]),
        (
            "pre_block.attn.proj.weight".to_owned(),
            vec![config.latent_channels, config.latent_channels],
        ),
        (
            "pre_block.attn.proj.bias".to_owned(),
            vec![config.latent_channels],
        ),
    ]);
    let mlp_hidden = config
        .latent_channels
        .checked_mul(2)
        .context("audio pre-block MLP width overflow")?;
    for projection in ["w0", "w1"] {
        expected.push((
            format!("pre_block.mlp.{projection}.weight"),
            vec![mlp_hidden, config.latent_channels],
        ));
        expected.push((format!("pre_block.mlp.{projection}.bias"), vec![mlp_hidden]));
    }
    expected.extend([
        (
            "pre_block.mlp.w2.weight".to_owned(),
            vec![config.latent_channels, mlp_hidden],
        ),
        (
            "pre_block.mlp.w2.bias".to_owned(),
            vec![config.latent_channels],
        ),
        (
            "mean_proj.weight".to_owned(),
            vec![config.latent_channels, config.latent_channels, 1],
        ),
        ("mean_proj.bias".to_owned(), vec![config.latent_channels]),
    ]);

    for (name, shape) in expected {
        let metadata = weights
            .metadata(&name)
            .with_context(|| format!("official audio VAE is missing encoder tensor {name}"))?;
        anyhow::ensure!(
            metadata.dtype == "F32",
            "audio VAE encoder tensor {name} has dtype {}, expected F32",
            metadata.dtype
        );
        anyhow::ensure!(
            metadata.shape == shape,
            "audio VAE encoder tensor {name} has shape {:?}, expected {shape:?}",
            metadata.shape
        );
    }
    Ok(())
}

fn push_weight_norm_conv_shapes(
    expected: &mut Vec<(String, Vec<usize>)>,
    prefix: &str,
    output_channels: usize,
    input_channels: usize,
    kernel: usize,
) {
    expected.push((format!("{prefix}.weight_g"), vec![output_channels, 1, 1]));
    expected.push((
        format!("{prefix}.weight_v"),
        vec![output_channels, input_channels, kernel],
    ));
    expected.push((format!("{prefix}.bias"), vec![output_channels]));
}

fn push_layer_norm_shapes(expected: &mut Vec<(String, Vec<usize>)>, prefix: &str, width: usize) {
    expected.push((format!("{prefix}.weight"), vec![width]));
    expected.push((format!("{prefix}.bias"), vec![width]));
}

fn adaptive_avg_pool_last_dim(input: &Tensor, output_size: usize) -> Result<Tensor> {
    anyhow::ensure!(
        output_size > 0,
        "adaptive pool output size must be non-zero"
    );
    let input_size = input.dim(D::Minus1)?;
    anyhow::ensure!(input_size > 0, "adaptive pool input must be non-empty");
    let mut bins = Vec::with_capacity(output_size);
    for index in 0..output_size {
        let start = index * input_size / output_size;
        let end = ((index + 1) * input_size).div_ceil(output_size);
        bins.push(
            input
                .narrow(D::Minus1, start, end - start)?
                .mean_keepdim(D::Minus1)?,
        );
    }
    let refs = bins.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, D::Minus1).map_err(Into::into)
}

fn causal_mask(sequence: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let elements = sequence
        .checked_mul(sequence)
        .context("audio attention mask size overflow")?;
    let mut values = Vec::with_capacity(elements);
    for query in 0..sequence {
        values.extend((0..sequence).map(|key| if key > query { f32::NEG_INFINITY } else { 0. }));
    }
    Tensor::from_vec(values, (1, 1, sequence, sequence), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn normalized_weight(weight_g: &Tensor, weight_v: &Tensor) -> Result<Tensor> {
    let norm = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    weight_v
        .broadcast_mul(weight_g)?
        .broadcast_div(&norm)
        .map_err(Into::into)
}

fn required<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    weights
        .get(name)
        .with_context(|| format!("missing audio VAE encoder tensor {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors;
    use serde_json::json;
    use std::{collections::HashMap, fs};

    fn ones(shape: impl Into<candle_core::Shape>) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn zeros(shape: impl Into<candle_core::Shape>) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn add_weight_norm_conv(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        out_channels: usize,
        in_channels: usize,
        kernel: usize,
    ) {
        weights.insert(format!("{prefix}.weight_g"), ones((out_channels, 1, 1)));
        weights.insert(
            format!("{prefix}.weight_v"),
            ones((out_channels, in_channels, kernel)),
        );
        weights.insert(format!("{prefix}.bias"), zeros(out_channels));
    }

    #[test]
    fn tiny_encoder_upmixes_pads_normalizes_and_packs_channel_major() {
        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "_class_name": "AutoencoderKLMiniMaxH3Audio",
            "encoder_dim": 1,
            "encoder_rates": [2],
            "latent_dim": 2,
            "latent_channels": 1,
            "num_attention_heads": 1,
            "sampling_rate": 32000,
            "latents_mean": [0.0],
            "latents_std": [2.0]
        });
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        let mut weights = HashMap::new();
        add_weight_norm_conv(&mut weights, "encoder.block.0", 1, 1, 7);
        for unit in 0..3 {
            let prefix = format!("encoder.block.1.block.{unit}.block");
            weights.insert(format!("{prefix}.0.alpha"), ones((1, 1, 1)));
            add_weight_norm_conv(&mut weights, &format!("{prefix}.1"), 1, 1, 7);
            weights.insert(format!("{prefix}.2.alpha"), ones((1, 1, 1)));
            add_weight_norm_conv(&mut weights, &format!("{prefix}.3"), 1, 1, 1);
        }
        weights.insert("encoder.block.1.block.3.alpha".to_owned(), ones((1, 1, 1)));
        add_weight_norm_conv(&mut weights, "encoder.block.1.block.4", 2, 1, 4);
        weights.insert("encoder.block.2.alpha".to_owned(), ones((1, 2, 1)));
        add_weight_norm_conv(&mut weights, "encoder.block.3", 2, 2, 3);

        for prefix in ["pre_block.norm1", "pre_block.norm3"] {
            weights.insert(format!("{prefix}.weight"), ones(2));
            weights.insert(format!("{prefix}.bias"), zeros(2));
        }
        for prefix in ["pre_block.norm2", "pre_block.mlp.norm"] {
            weights.insert(format!("{prefix}.weight"), ones(1));
            weights.insert(format!("{prefix}.bias"), zeros(1));
        }
        weights.insert("pre_block.proj.weight".to_owned(), ones((1, 2)));
        weights.insert("pre_block.proj.bias".to_owned(), zeros(1));
        weights.insert("pre_block.attn.qkv.weight".to_owned(), ones((6, 2)));
        weights.insert("pre_block.attn.q_bias".to_owned(), zeros(2));
        weights.insert("pre_block.attn.zero_k_bias".to_owned(), zeros(2));
        weights.insert("pre_block.attn.v_bias".to_owned(), zeros(2));
        weights.insert("pre_block.attn.proj.weight".to_owned(), ones((1, 1)));
        weights.insert("pre_block.attn.proj.bias".to_owned(), zeros(1));
        for projection in ["w0", "w1"] {
            weights.insert(format!("pre_block.mlp.{projection}.weight"), ones((2, 1)));
            weights.insert(format!("pre_block.mlp.{projection}.bias"), zeros(2));
        }
        weights.insert("pre_block.mlp.w2.weight".to_owned(), ones((1, 2)));
        weights.insert("pre_block.mlp.w2.bias".to_owned(), zeros(1));
        weights.insert("mean_proj.weight".to_owned(), zeros((1, 1, 1)));
        weights.insert(
            "mean_proj.bias".to_owned(),
            Tensor::new(&[2f32], &Device::Cpu).unwrap(),
        );
        safetensors::save(
            &weights,
            dir.path().join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();

        let encoder = StreamedAudioVaeEncoder::open(
            dir.path(),
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
        )
        .unwrap()
        .with_attention_query_chunk_size(1)
        .unwrap();
        let mono = Tensor::new(&[[0.25f32, -0.5, 0.75]], &Device::Cpu).unwrap();
        let posterior = encoder.encode_posterior_mode(&mono, 32_000).unwrap();
        assert_eq!(posterior.dims(), &[2, 1, 2]);
        assert_eq!(
            posterior.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.; 4]
        );

        let rows = encoder.encode_condition_rows(&mono, 32_000).unwrap();
        assert_eq!(rows.dims(), &[4, 1]);
        assert_eq!(
            rows.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.; 4]
        );
        assert!(encoder.encode_condition_rows(&mono, 44_100).is_err());

        let wrong = tempfile::tempdir().unwrap();
        fs::write(
            wrong.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut wrong_weights = weights.clone();
        wrong_weights.insert("mean_proj.weight".to_owned(), zeros((1, 2, 1)));
        safetensors::save(
            &wrong_weights,
            wrong.path().join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        let error = StreamedAudioVaeEncoder::open(
            wrong.path(),
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
        )
        .err()
        .expect("wrong encoder header shape must fail during open")
        .to_string();
        assert!(error.contains("mean_proj.weight"));
        assert!(error.contains("expected [1, 1, 1]"));
    }

    #[test]
    fn adaptive_pool_matches_reference_bin_boundaries() {
        let input = Tensor::new(&[[[0f32, 1., 2., 3., 4.]]], &Device::Cpu).unwrap();
        let pooled = adaptive_avg_pool_last_dim(&input, 3).unwrap();
        assert_eq!(pooled.dims(), &[1, 1, 3]);
        assert_eq!(
            pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.5, 2., 3.5]
        );
    }
}
