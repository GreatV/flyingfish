use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct AudioVaeConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub latent_dim: usize,
    pub latent_channels: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub sampling_rate: u32,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl AudioVaeConfig {
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
        anyhow::ensure!(self.latent_dim > 0, "latent_dim must be non-zero");
        anyhow::ensure!(self.latent_channels > 0, "latent_channels must be non-zero");
        anyhow::ensure!(self.decoder_dim > 0, "decoder_dim must be non-zero");
        anyhow::ensure!(
            self.decoder_rates.len() == self.decoder_kernel_sizes.len(),
            "decoder rate and kernel counts differ"
        );
        anyhow::ensure!(
            !self.decoder_rates.is_empty(),
            "audio decoder must have at least one upsampling stage"
        );
        anyhow::ensure!(
            self.resblock_kernel_sizes.len() == self.resblock_dilation_sizes.len(),
            "resblock kernel and dilation counts differ"
        );
        anyhow::ensure!(
            !self.resblock_kernel_sizes.is_empty(),
            "audio decoder must have at least one residual kernel"
        );
        anyhow::ensure!(
            self.latents_mean.len() == self.latent_channels
                && self.latents_std.len() == self.latent_channels,
            "audio latent statistics must have one value per latent channel"
        );
        anyhow::ensure!(
            self.latents_std.iter().all(|value| *value > 0.),
            "audio latent standard deviations must be positive"
        );
        anyhow::ensure!(self.sampling_rate > 0, "sampling_rate must be non-zero");
        for (&rate, &kernel) in self.decoder_rates.iter().zip(&self.decoder_kernel_sizes) {
            anyhow::ensure!(rate > 0 && kernel >= rate, "invalid audio upsampler");
            anyhow::ensure!(
                (kernel - rate).is_multiple_of(2),
                "audio upsampler requires symmetric integral padding"
            );
        }
        Ok(())
    }

    pub fn hop_length(&self) -> Result<usize> {
        self.decoder_rates.iter().try_fold(1usize, |value, rate| {
            value
                .checked_mul(*rate)
                .context("audio hop length overflow")
        })
    }
}

pub struct StreamedAudioVae {
    weights: ModelWeights,
    config: AudioVaeConfig,
    device: Device,
}

impl StreamedAudioVae {
    pub fn open(
        component_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
    ) -> Result<Self> {
        let component_dir = component_dir.as_ref();
        let config = AudioVaeConfig::from_file(component_dir.join("config.json"))?;
        let weights = ModelWeights::open(component_dir, source, cache_policy)?;
        Ok(Self {
            weights,
            config,
            device,
        })
    }

    pub fn config(&self) -> &AudioVaeConfig {
        &self.config
    }

    pub fn decode(&self, normalized_latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames) = normalized_latents
            .dims3()
            .context("audio latents must be [channels, latent_channels, frames]")?;
        anyhow::ensure!(batch > 0 && frames > 0, "audio latents must be non-empty");
        anyhow::ensure!(
            channels == self.config.latent_channels,
            "audio latents have {channels} channels, expected {}",
            self.config.latent_channels
        );
        anyhow::ensure!(
            normalized_latents.device().same_device(&self.device),
            "audio latents are on a different device than the decoder"
        );

        let latents = normalized_latents.to_dtype(DType::F32)?;
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
        let latents = latents.broadcast_mul(&std)?.broadcast_add(&mean)?;

        let mut hidden = self.plain_conv("dec_in_proj", &latents, 0, 1, 1)?;
        hidden = self.weight_norm_conv("decoder.conv_pre", &hidden, 3, 1, 1, true)?;

        for (stage, (&rate, &kernel)) in self
            .config
            .decoder_rates
            .iter()
            .zip(&self.config.decoder_kernel_sizes)
            .enumerate()
        {
            hidden = self.weight_norm_conv_transpose(
                &format!("decoder.ups.{stage}.0"),
                &hidden,
                (kernel - rate) / 2,
                rate,
            )?;

            let mut sum: Option<Tensor> = None;
            for kernel_index in 0..self.config.resblock_kernel_sizes.len() {
                let block_index = stage * self.config.resblock_kernel_sizes.len() + kernel_index;
                let block = self.amp_block(&hidden, block_index, kernel_index)?;
                sum = Some(match sum {
                    Some(value) => value.add(&block)?,
                    None => block,
                });
            }
            hidden = (sum.context("audio decoder stage has no residual blocks")?
                / self.config.resblock_kernel_sizes.len() as f64)?;
        }

        hidden = self.alias_free_activation(&hidden, "decoder.activation_post")?;
        hidden = self.weight_norm_conv("decoder.conv_post", &hidden, 3, 1, 1, false)?;
        let expected_samples = frames
            .checked_mul(self.config.hop_length()?)
            .context("decoded audio length overflow")?;
        anyhow::ensure!(
            hidden.dims() == [batch, 1, expected_samples],
            "decoded waveform shape {:?}, expected [{batch}, 1, {expected_samples}]",
            hidden.dims()
        );
        hidden.clamp(-1f32, 1f32).map_err(Into::into)
    }

    fn amp_block(&self, input: &Tensor, block_index: usize, kernel_index: usize) -> Result<Tensor> {
        let prefix = format!("decoder.resblocks.{block_index}");
        let kernel = self.config.resblock_kernel_sizes[kernel_index];
        let dilations = &self.config.resblock_dilation_sizes[kernel_index];
        let mut hidden = input.clone();
        for (index, &dilation) in dilations.iter().enumerate() {
            let residual = self
                .alias_free_activation(&hidden, &format!("{prefix}.activations.{}", 2 * index))?;
            let residual = self.weight_norm_conv(
                &format!("{prefix}.convs1.{index}"),
                &residual,
                (kernel * dilation - dilation) / 2,
                1,
                dilation,
                true,
            )?;
            let residual = self.alias_free_activation(
                &residual,
                &format!("{prefix}.activations.{}", 2 * index + 1),
            )?;
            let residual = self.weight_norm_conv(
                &format!("{prefix}.convs2.{index}"),
                &residual,
                (kernel - 1) / 2,
                1,
                1,
                true,
            )?;
            hidden = hidden.add(&residual)?;
        }
        Ok(hidden)
    }

    fn alias_free_activation(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let names = [
            format!("{prefix}.act.alpha"),
            format!("{prefix}.act.beta"),
            format!("{prefix}.upsample.filter"),
            format!("{prefix}.downsample.lowpass.filter"),
        ];
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            let alpha = required(weights, &names[0])?.exp()?.reshape((1, (), 1))?;
            let beta = required(weights, &names[1])?.exp()?.reshape((1, (), 1))?;
            let channels = input.dim(1)?;

            let up_filter = required(weights, &names[2])?.expand((channels, 1, 12))?;
            let padded = input.pad_with_same(D::Minus1, 5, 5)?;
            let up = ConvTranspose1d::new(
                up_filter,
                None,
                ConvTranspose1dConfig {
                    stride: 2,
                    groups: channels,
                    ..Default::default()
                },
            )
            .forward(&padded)?;
            let length = up.dim(2)?;
            anyhow::ensure!(length >= 30, "alias-free upsample output is too short");
            let up = (up * 2f64)?.narrow(2, 15, length - 30)?;
            let phase = alpha.broadcast_mul(&up)?;
            let sine = phase.sin()?;
            let activated = up.add(&sine.sqr()?.broadcast_div(&(&beta + 1e-9)?)?)?;

            let down_filter = required(weights, &names[3])?.expand((channels, 1, 12))?;
            let activated = activated.pad_with_same(D::Minus1, 5, 6)?;
            Conv1d::new(
                down_filter,
                None,
                Conv1dConfig {
                    stride: 2,
                    groups: channels,
                    ..Default::default()
                },
            )
            .forward(&activated)
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
        bias: bool,
    ) -> Result<Tensor> {
        let mut names = vec![format!("{prefix}.weight_g"), format!("{prefix}.weight_v")];
        if bias {
            names.push(format!("{prefix}.bias"));
        }
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        self.weights.with_group(&refs, &self.device, |weights| {
            let weight =
                normalized_weight(required(weights, &names[0])?, required(weights, &names[1])?)?;
            let bias = if bias {
                Some(required(weights, &names[2])?.clone())
            } else {
                None
            };
            Conv1d::new(
                weight,
                bias,
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

    fn weight_norm_conv_transpose(
        &self,
        prefix: &str,
        input: &Tensor,
        padding: usize,
        stride: usize,
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
            let input_length = input.dim(2)?;
            let leading = (input_length - 1)
                .checked_mul(stride)
                .context("transposed-convolution length overflow")?;
            let required_padding = 2 * padding;
            let extra = if leading < required_padding {
                (required_padding - leading).div_ceil(2 * stride)
            } else {
                0
            };
            let padded = if extra > 0 {
                input.pad_with_zeros(D::Minus1, extra, extra)?
            } else {
                input.clone()
            };
            let output = ConvTranspose1d::new(
                weight,
                Some(required(weights, &names[2])?.clone()),
                ConvTranspose1dConfig {
                    padding,
                    stride,
                    ..Default::default()
                },
            )
            .forward(&padded)?;
            if extra > 0 {
                let crop = extra * stride;
                let output_length = output.dim(2)?;
                output
                    .narrow(2, crop, output_length - 2 * crop)
                    .map_err(Into::into)
            } else {
                Ok(output)
            }
        })
    }
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
        .with_context(|| format!("missing audio VAE tensor {name}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum WavSampleFormat {
    #[value(help = "Broadly compatible signed 16-bit PCM for listening and media muxing")]
    Pcm16,
    #[value(help = "Unclipped IEEE F32 samples for numerical comparison with a reference")]
    Float32,
}

pub fn write_wav(
    path: impl AsRef<Path>,
    waveform: &Tensor,
    sampling_rate: u32,
    format: WavSampleFormat,
) -> Result<()> {
    let path = path.as_ref();
    let waveform = waveform.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let (channels, mono, samples) = waveform.dims3()?;
    anyhow::ensure!(channels > 0, "waveform has no channels");
    anyhow::ensure!(mono == 1, "waveform must have shape [channels, 1, samples]");
    anyhow::ensure!(
        channels <= u16::MAX as usize,
        "waveform has too many channels"
    );
    let values = waveform.squeeze(1)?.to_vec2::<f32>()?;
    let (bits_per_sample, sample_format) = match format {
        WavSampleFormat::Pcm16 => (16, hound::SampleFormat::Int),
        WavSampleFormat::Float32 => (32, hound::SampleFormat::Float),
    };
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate: sampling_rate,
        bits_per_sample,
        sample_format,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV {}", path.display()))?;
    match format {
        WavSampleFormat::Pcm16 => {
            write_interleaved_samples(&mut writer, &values, samples, |value| {
                (value.clamp(-1., 1.) * i16::MAX as f32).round() as i16
            })?
        }
        WavSampleFormat::Float32 => {
            write_interleaved_samples(&mut writer, &values, samples, |value| value)?
        }
    }
    writer.finalize()?;
    Ok(())
}

fn write_interleaved_samples<W, T>(
    writer: &mut hound::WavWriter<W>,
    channels: &[Vec<f32>],
    samples: usize,
    convert: impl Fn(f32) -> T,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    T: hound::Sample,
{
    for sample in 0..samples {
        for channel in channels {
            writer.write_sample(convert(channel[sample]))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors;
    use serde_json::json;
    use std::{collections::HashMap, fs};

    #[test]
    fn validates_official_geometry() {
        let config = AudioVaeConfig {
            class_name: "AutoencoderKLMiniMaxH3Audio".to_owned(),
            latent_dim: 2048,
            latent_channels: 32,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: vec![9, 9, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            sampling_rate: 32_000,
            latents_mean: vec![0.; 32],
            latents_std: vec![1.; 32],
        };
        config.validate().unwrap();
        assert_eq!(config.hop_length().unwrap(), 800);
    }

    #[test]
    fn writes_interleaved_stereo_wav() {
        let dir = tempfile::tempdir().unwrap();
        let waveform =
            Tensor::new(&[[[0f32, 0.5, -0.5]], [[1f32, -1., 0.25]]], &Device::Cpu).unwrap();
        let path = dir.path().join("audio.wav");
        write_wav(&path, &waveform, 32_000, WavSampleFormat::Pcm16).unwrap();
        let reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().channels, 2);
        assert_eq!(reader.duration(), 3);
        assert_eq!(reader.into_samples::<i16>().count(), 6);

        let float_path = dir.path().join("audio-f32.wav");
        write_wav(&float_path, &waveform, 32_000, WavSampleFormat::Float32).unwrap();
        let reader = hound::WavReader::open(float_path).unwrap();
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);
        assert_eq!(reader.spec().bits_per_sample, 32);
        assert_eq!(
            reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![0., 1., 0.5, -1., -0.5, 0.25]
        );
    }

    #[test]
    fn runs_a_tiny_streamed_bigvgan_decode() {
        fn tensor(shape: impl Into<candle_core::Shape>) -> Tensor {
            Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
        }

        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "_class_name": "AutoencoderKLMiniMaxH3Audio",
            "latent_dim": 2,
            "latent_channels": 1,
            "decoder_dim": 2,
            "decoder_rates": [2],
            "decoder_kernel_sizes": [4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1]],
            "sampling_rate": 32000,
            "latents_mean": [0.0],
            "latents_std": [1.0]
        });
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        let mut weights = HashMap::from([
            ("dec_in_proj.weight".to_owned(), tensor((2, 1, 1))),
            ("dec_in_proj.bias".to_owned(), tensor(2)),
            ("decoder.conv_pre.weight_g".to_owned(), tensor((2, 1, 1))),
            ("decoder.conv_pre.weight_v".to_owned(), tensor((2, 2, 7))),
            ("decoder.conv_pre.bias".to_owned(), tensor(2)),
            ("decoder.ups.0.0.weight_g".to_owned(), tensor((2, 1, 1))),
            ("decoder.ups.0.0.weight_v".to_owned(), tensor((2, 1, 4))),
            ("decoder.ups.0.0.bias".to_owned(), tensor(1)),
            (
                "decoder.resblocks.0.convs1.0.weight_g".to_owned(),
                tensor((1, 1, 1)),
            ),
            (
                "decoder.resblocks.0.convs1.0.weight_v".to_owned(),
                tensor((1, 1, 3)),
            ),
            ("decoder.resblocks.0.convs1.0.bias".to_owned(), tensor(1)),
            (
                "decoder.resblocks.0.convs2.0.weight_g".to_owned(),
                tensor((1, 1, 1)),
            ),
            (
                "decoder.resblocks.0.convs2.0.weight_v".to_owned(),
                tensor((1, 1, 3)),
            ),
            ("decoder.resblocks.0.convs2.0.bias".to_owned(), tensor(1)),
            ("decoder.conv_post.weight_g".to_owned(), tensor((1, 1, 1))),
            ("decoder.conv_post.weight_v".to_owned(), tensor((1, 1, 7))),
        ]);
        for prefix in [
            "decoder.resblocks.0.activations.0",
            "decoder.resblocks.0.activations.1",
            "decoder.activation_post",
        ] {
            weights.insert(
                format!("{prefix}.act.alpha"),
                Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            );
            weights.insert(
                format!("{prefix}.act.beta"),
                Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            );
            weights.insert(format!("{prefix}.upsample.filter"), tensor((1, 1, 12)));
            weights.insert(
                format!("{prefix}.downsample.lowpass.filter"),
                tensor((1, 1, 12)),
            );
        }
        safetensors::save(
            &weights,
            dir.path().join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();

        let vae = StreamedAudioVae::open(
            dir.path(),
            WeightSource::Mmap,
            CachePolicy::new(1),
            Device::Cpu,
        )
        .unwrap();
        let latents = Tensor::zeros((2, 1, 3), DType::F32, &Device::Cpu).unwrap();
        let waveform = vae.decode(&latents).unwrap();
        assert_eq!(waveform.dims(), &[2, 1, 6]);
    }
}
