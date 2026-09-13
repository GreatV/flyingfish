use super::*;
use flyingfish::trellis::{clip::ClipModel, clip_text::ClipTokenizer};

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
    let device = parse_device(&device)?;
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
    #[allow(clippy::excessive_precision)]
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    let std = [0.26862954f32, 0.26130258, 0.27577711];
    let mut data = vec![0f32; 3 * size * size];
    for pixel in 0..size * size {
        for channel in 0..3 {
            let value = bytes[pixel * channels + if gray { 0 } else { channel }] as f32 / 255.;
            data[channel * size * size + pixel] = (value - mean[channel]) / std[channel];
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, size, size), device)?)
}
