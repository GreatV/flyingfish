use super::WeightCacheArgs;
use super::device_parse::parse_device_single;
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use clap::Subcommand;
use flyingfish::trellis::clip::ClipModel;
use flyingfish::trellis::clip_text::ClipTokenizer;
use std::path::{Path, PathBuf};

#[derive(Debug, Subcommand)]
pub(super) enum ClipCommand {
    #[command(
        about = "Score candidate texts for an RGB PNG already resized to the model's square input"
    )]
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(
            long,
            help = "PNG at the model's exact input size (224x224 for ViT-L/14)"
        )]
        image: PathBuf,
        #[arg(long, required = true)]
        text: Vec<String>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
    },
}

pub(super) fn run(command: ClipCommand) -> Result<()> {
    let ClipCommand::Score {
        model,
        image,
        text,
        device,
        weights,
    } = command;
    let device = parse_device_single(&device)?;
    let encoder = ClipModel::open(
        &model,
        &device,
        weights.weight_source,
        weights.cache_policy()?,
    )?;
    let pixels = read_pixels(&image, encoder.image_size()?, &device)?;
    let tokenizer = ClipTokenizer::open(&model, encoder.text_length())?;
    let prompts = text.iter().map(String::as_str).collect::<Vec<_>>();
    let tokens = tokenizer.encode(&prompts, &device)?;
    let result = encoder.score(&pixels, &tokens)?;
    let logits = result.logits_per_image.squeeze(0)?.to_vec1::<f32>()?;
    let probabilities = result.probabilities.squeeze(0)?.to_vec1::<f32>()?;
    anyhow::ensure!(
        logits.iter().chain(&probabilities).all(|v| v.is_finite()),
        "CLIP produced non-finite scores"
    );
    let scores = text.iter().zip(logits).zip(probabilities).map(|((text,logit),probability)|
        serde_json::json!({"text":text,"logit":logit,"probability":probability})).collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"image":image,"scores":scores}))?
    );
    Ok(())
}

fn read_pixels(path: &Path, size: usize, device: &Device) -> Result<Tensor> {
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    anyhow::ensure!(
        reader.info().width as usize == size && reader.info().height as usize == size,
        "CLIP needs an already resized {size}x{size} PNG; got {}x{}",
        reader.info().width,
        reader.info().height
    );
    let mut bytes = vec![
        0;
        reader
            .output_buffer_size()
            .context("PNG decoded size overflow")?
    ];
    let info = reader.next_frame(&mut bytes)?;
    anyhow::ensure!(
        info.bit_depth == png::BitDepth::Eight,
        "CLIP PNG must decode to 8-bit pixels"
    );
    let (channels, gray) = match info.color_type {
        png::ColorType::Rgb => (3, false),
        png::ColorType::Rgba => (4, false),
        png::ColorType::Grayscale => (1, true),
        png::ColorType::GrayscaleAlpha => (2, true),
        _ => bail!("unsupported CLIP PNG color type"),
    };
    let mean = [0.481_454_67f32, 0.457_827_5, 0.408_210_72];
    let std = [0.268_629_55f32, 0.261_302_6, 0.275_777_1];
    let mut data = vec![0f32; 3 * size * size];
    for pixel in 0..size * size {
        for channel in 0..3 {
            let value = bytes[pixel * channels + if gray { 0 } else { channel }] as f32 / 255.;
            data[channel * size * size + pixel] = (value - mean[channel]) / std[channel];
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, size, size), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_png_preserves_normalization_bits() -> Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        let bytes: Vec<u8> = (0..256u16)
            .flat_map(|n| {
                [
                    n as u8,
                    n.wrapping_mul(37).wrapping_add(11) as u8,
                    (255 - n) as u8,
                ]
            })
            .collect();
        {
            let mut encoder = png::Encoder::new(file.reopen()?, 16, 16);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header()?.write_image_data(&bytes)?;
        }
        let pixels = read_pixels(file.path(), 16, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mean = [0x3ef6813a, 0x3eea685e, 0x3ed100ff].map(f32::from_bits);
        let std = [0x3e8989d0, 0x3e85c974, 0x3e8d32a8].map(f32::from_bits);
        for (channel, plane) in pixels.chunks_exact(256).enumerate() {
            for (pixel, &value) in plane.iter().enumerate() {
                let raw = bytes[pixel * 3 + channel] as f32 / 255.;
                let expected = (raw - mean[channel]) / std[channel];
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "channel {channel}, pixel {pixel}"
                );
            }
        }
        Ok(())
    }
}
