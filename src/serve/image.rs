use crate::model_spec::{Architecture, ModelSpec};
use crate::vision_input::{ImageGrid, MultimodalInput};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::io::Cursor;
use std::path::Path;

const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;
const MAX_SOURCE_PIXELS: usize = 4 * 16_777_216;

pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

pub struct PackedImage {
    pub grid: ImageGrid,
    pub pixel_values: Vec<f32>,
}

pub struct PlannedImage {
    bytes: Vec<u8>,
    mime: String,
    grid: ImageGrid,
}

impl PlannedImage {
    pub fn feature_rows(&self) -> Result<usize> {
        self.grid.feature_rows()
    }
}

pub struct PreparedImages {
    pub modality_ids: Vec<u8>,
    pub pixel_values: Vec<f32>,
    pub grids: Vec<ImageGrid>,
}

impl PreparedImages {
    pub fn save(&self, ids: &[u32], path: &Path) -> Result<()> {
        let mut tensors = crate::tensors::Tensors::default();
        tensors.i64(
            "input_ids".into(),
            vec![ids.len()],
            &ids.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
        )?;
        tensors.i64(
            "modality_ids".into(),
            vec![self.modality_ids.len()],
            &self
                .modality_ids
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<_>>(),
        )?;
        let grids: Vec<_> = self
            .grids
            .iter()
            .flat_map(|g| [i64::from(g.t), i64::from(g.h), i64::from(g.w)])
            .collect();
        tensors.i64("image_grid_thw".into(), vec![self.grids.len(), 3], &grids)?;
        tensors.f32(
            "pixel_values_fp32".into(),
            vec![self.pixel_values.len() / 1536, 1536],
            &self.pixel_values,
        )?;
        tensors.save_new(
            path,
            [("format".into(), "flyingfish-image-input-v1".into())].into(),
        )
    }

    pub fn read(path: &Path, ids: &[u32]) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let integers = |name: &str| -> Result<Vec<i64>> {
            let tensor = tensors.tensor(name)?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::I64,
                "{name} must be I64"
            );
            Ok(tensor
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| i64::from_le_bytes(*v))
                .collect())
        };
        ensure!(
            tensors.tensor("input_ids")?.shape() == [ids.len()],
            "input_ids shape mismatch"
        );
        ensure!(
            integers("input_ids")?
                .iter()
                .copied()
                .eq(ids.iter().map(|&v| i64::from(v))),
            "image tensor input_ids differ from generation input"
        );
        ensure!(
            tensors.tensor("modality_ids")?.shape() == [ids.len()],
            "modality_ids shape mismatch"
        );
        let modality_ids = integers("modality_ids")?
            .into_iter()
            .map(u8::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let grid = tensors.tensor("image_grid_thw")?;
        ensure!(
            grid.shape().len() == 2 && grid.shape()[1] == 3,
            "image_grid_thw must have shape [images,3]"
        );
        let values = integers("image_grid_thw")?;
        let grids = values
            .as_chunks::<3>()
            .0
            .iter()
            .map(|g| {
                Ok(ImageGrid {
                    t: u32::try_from(g[0])?,
                    h: u32::try_from(g[1])?,
                    w: u32::try_from(g[2])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pixels = tensors.tensor("pixel_values_fp32")?;
        ensure!(
            pixels.dtype() == safetensors::Dtype::F32
                && pixels.shape().len() == 2
                && pixels.shape()[1] == 1536,
            "pixel_values_fp32 must be F32 [patches,1536]"
        );
        let pixel_values = pixels
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        Ok(Self {
            modality_ids,
            pixel_values,
            grids,
        })
    }

    pub fn input<'a>(&'a self, ids: &'a [u32]) -> MultimodalInput<'a> {
        MultimodalInput {
            input_ids: ids,
            modality_ids: &self.modality_ids,
            pixel_values: &self.pixel_values,
            grids: &self.grids,
        }
    }
}

#[derive(Deserialize)]
struct Size {
    shortest_edge: usize,
    longest_edge: usize,
}

#[derive(Deserialize)]
struct Config {
    size: Size,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

pub struct Processor {
    spec: ModelSpec,
    config: Config,
}

impl Processor {
    pub fn load(model: &Path) -> Result<Option<Self>> {
        let spec = ModelSpec::read(model)?;
        if spec.architecture != Architecture::Qwen35Text || spec.vision.is_none() {
            return Ok(None);
        }
        let config: Config =
            serde_json::from_reader(std::fs::File::open(model.join("preprocessor_config.json"))?)?;
        ensure!(
            config.patch_size == 16
                && config.temporal_patch_size == 2
                && config.merge_size == 2
                && config.image_mean == [0.5; 3]
                && config.image_std == [0.5; 3]
                && config.size.shortest_edge == 65536
                && config.size.longest_edge == 16777216,
            "unsupported image processor configuration"
        );
        Ok(Some(Self { spec, config }))
    }

    pub fn prepare_rgb(&self, image: &RgbImage) -> Result<PackedImage> {
        let count = checked_pixels(image.width, image.height)?;
        ensure!(
            image.pixels.len() == count * 3,
            "RGB byte count does not match dimensions"
        );
        let (height, width) = smart_resize(
            image.height,
            image.width,
            self.config.size.shortest_edge,
            self.config.size.longest_edge,
        )?;
        let resized = resize_bicubic(image, width, height);
        let gh = height / 16;
        let gw = width / 16;
        let mut pixels = Vec::with_capacity(width * height * 3 * 2);
        for group_y in 0..gh / 2 {
            for group_x in 0..gw / 2 {
                for merge_y in 0..2 {
                    for merge_x in 0..2 {
                        for channel in 0..3 {
                            for _temporal in 0..2 {
                                for y in 0..16 {
                                    for x in 0..16 {
                                        let row = (group_y * 2 + merge_y) * 16 + y;
                                        let col = (group_x * 2 + merge_x) * 16 + x;
                                        let value =
                                            resized[(row * width + col) * 3 + channel] as f32;
                                        pixels.push((value - 127.5) / 127.5);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(PackedImage {
            grid: ImageGrid {
                t: 1,
                h: gh as u32,
                w: gw as u32,
            },
            pixel_values: pixels,
        })
    }

    pub fn prepare_url(&self, url: &str) -> Result<PackedImage> {
        self.prepare_planned(self.plan_url(url)?)
    }

    pub fn plan_url(&self, url: &str) -> Result<PlannedImage> {
        let (bytes, mime) = load_encoded(url)?;
        let (width, height) = image_dimensions(&bytes, Some(&mime))?;
        let (height, width) = smart_resize(
            height,
            width,
            self.config.size.shortest_edge,
            self.config.size.longest_edge,
        )?;
        Ok(PlannedImage {
            bytes,
            mime,
            grid: ImageGrid {
                t: 1,
                h: (height / 16) as u32,
                w: (width / 16) as u32,
            },
        })
    }

    pub fn prepare_planned(&self, image: PlannedImage) -> Result<PackedImage> {
        let packed = self.prepare_rgb(&decode_image(&image.bytes, Some(&image.mime))?)?;
        ensure!(
            packed.grid == image.grid,
            "decoded image grid differs from header"
        );
        Ok(packed)
    }

    pub fn expanded_token_count(&self, ids: &[u32], images: &[PlannedImage]) -> Result<usize> {
        let vision = self
            .spec
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let placeholders = ids.iter().filter(|&&id| id == vision.image_token).count();
        ensure!(
            placeholders == images.len(),
            "image placeholder count differs from supplied images"
        );
        images
            .iter()
            .try_fold(ids.len() - placeholders, |total, image| {
                total
                    .checked_add(image.feature_rows()?)
                    .context("image token count overflow")
            })
    }

    pub fn expand(
        &self,
        ids: &[u32],
        images: Vec<PackedImage>,
        capacity: usize,
    ) -> Result<(Vec<u32>, PreparedImages)> {
        let vision = self
            .spec
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let mut expanded = Vec::new();
        let mut prepared = PreparedImages {
            modality_ids: Vec::new(),
            pixel_values: Vec::new(),
            grids: Vec::new(),
        };
        let mut images = images.into_iter();
        for &id in ids {
            if id == vision.image_token {
                let image = images
                    .next()
                    .context("more image placeholders than supplied images")?;
                let features = image.grid.feature_rows()?;
                ensure!(
                    expanded
                        .len()
                        .checked_add(features)
                        .is_some_and(|n| n <= capacity),
                    "image tokens exceed request capacity"
                );
                expanded.extend(std::iter::repeat_n(id, features));
                prepared
                    .modality_ids
                    .extend(std::iter::repeat_n(1, features));
                prepared.grids.push(image.grid);
                prepared.pixel_values.extend(image.pixel_values);
            } else {
                ensure!(expanded.len() < capacity, "prompt exceeds request capacity");
                expanded.push(id);
                prepared.modality_ids.push(0);
            }
        }
        ensure!(
            images.next().is_none(),
            "image has no placeholder in rendered prompt"
        );
        prepared.input(&expanded).validate(&self.spec, capacity)?;
        Ok((expanded, prepared))
    }
}

fn checked_pixels(width: usize, height: usize) -> Result<usize> {
    ensure!(width > 0 && height > 0, "image dimensions must be positive");
    let count = width
        .checked_mul(height)
        .context("image dimensions overflow")?;
    ensure!(
        count <= MAX_SOURCE_PIXELS,
        "decoded image exceeds pixel limit"
    );
    Ok(count)
}

pub fn smart_resize(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    checked_pixels(width, height)?;
    ensure!(
        height.max(width) as f64 / height.min(width) as f64 <= 200.0,
        "image aspect ratio exceeds 200"
    );
    ensure!(
        min_pixels > 0 && min_pixels <= max_pixels && max_pixels <= 16_777_216,
        "invalid image pixel budget"
    );
    let (h, w) = (height as f64, width as f64);
    let mut out_h = (h / 32.0).round_ties_even() as usize * 32;
    let mut out_w = (w / 32.0).round_ties_even() as usize * 32;
    if out_h * out_w > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        out_h = ((h / beta / 32.0).floor() as usize * 32).max(32);
        out_w = ((w / beta / 32.0).floor() as usize * 32).max(32);
    } else if out_h * out_w < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        out_h = (h * beta / 32.0).ceil() as usize * 32;
        out_w = (w * beta / 32.0).ceil() as usize * 32;
    }
    ensure!(
        out_h > 0 && out_w > 0 && out_h * out_w <= max_pixels,
        "resized image exceeds pixel budget"
    );
    Ok((out_h, out_w))
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    (0..output)
        .map(|out| {
            let center = (out as f64 + 0.5) * scale;
            let start = ((center - support + 0.5).floor() as isize).max(0) as usize;
            let end = ((center + support + 0.5).floor() as usize).min(input);
            let mut weights: Vec<_> = (start..end)
                .map(|i| cubic((i as f64 + 0.5 - center) / filter_scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= sum;
            }
            (start, weights)
        })
        .collect()
}

// V5 tensor-resize parity bounds: resized u8 <= 2, FP32 <= 4/255 + 4*EPS,
// BF16 <= that bound + 1/256; non-resized inputs are bit-exact.
fn resize_bicubic(image: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    let horizontal = if width == image.width {
        image.pixels.clone()
    } else {
        let weights = coefficients(image.width, width);
        let mut buffer = vec![0; width * image.height * 3];
        for y in 0..image.height {
            for (x, (start, weights)) in weights.iter().enumerate() {
                for channel in 0..3 {
                    let value: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| {
                            w * f64::from(image.pixels[(y * image.width + start + i) * 3 + channel])
                        })
                        .sum();
                    buffer[(y * width + x) * 3 + channel] = value.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        buffer
    };
    if height == image.height {
        return horizontal;
    }
    let weights = coefficients(image.height, height);
    let mut result = vec![0; width * height * 3];
    for (y, (start, weights)) in weights.iter().enumerate() {
        for x in 0..width {
            for channel in 0..3 {
                let value: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        w * f64::from(horizontal[((start + i) * width + x) * 3 + channel])
                    })
                    .sum();
                result[(y * width + x) * 3 + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    result
}

pub fn load_image(url: &str) -> Result<RgbImage> {
    let (bytes, mime) = load_encoded(url)?;
    decode_image(&bytes, Some(&mime))
}

fn load_encoded(url: &str) -> Result<(Vec<u8>, String)> {
    let (bytes, mime) = if url.starts_with("data:") {
        let (header, data) = url.split_once(',').context("invalid image data URL")?;
        let mime = header
            .strip_prefix("data:")
            .and_then(|h| h.strip_suffix(";base64"))
            .context("image data URL must use base64")?;
        ensure!(
            matches!(mime, "image/png" | "image/jpeg" | "image/jpg"),
            "only PNG and JPEG image URLs are supported"
        );
        ensure!(
            data.len() <= MAX_ENCODED_BYTES * 4 / 3 + 4,
            "encoded image exceeds byte limit"
        );
        (base64::decode(data)?, mime.to_owned())
    } else {
        bail!("image URL must be a base64 data URL");
    };
    Ok((bytes, mime))
}

enum ImageFormat {
    Png,
    Jpeg,
}

fn image_format(bytes: &[u8], mime: Option<&str>) -> Result<ImageFormat> {
    ensure!(
        bytes.len() <= MAX_ENCODED_BYTES,
        "encoded image exceeds byte limit"
    );
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        ensure!(
            mime.is_none_or(|m| m == "image/png"),
            "image MIME does not match PNG data"
        );
        Ok(ImageFormat::Png)
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        ensure!(
            mime.is_none_or(|m| m != "image/png"),
            "image MIME does not match JPEG data"
        );
        Ok(ImageFormat::Jpeg)
    } else {
        bail!("image is not PNG or JPEG");
    }
}

fn image_dimensions(bytes: &[u8], mime: Option<&str>) -> Result<(usize, usize)> {
    match image_format(bytes, mime)? {
        ImageFormat::Png => {
            let mut decoder = png::Decoder::new(Cursor::new(bytes));
            decoder.set_ignore_text_chunk(true);
            decoder.set_ignore_iccp_chunk(true);
            let reader = decoder.read_info()?;
            ensure!(
                reader.info().animation_control.is_none(),
                "animated PNG not supported"
            );
            Ok((reader.info().width as usize, reader.info().height as usize))
        }
        ImageFormat::Jpeg => {
            let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
            decoder.read_info()?;
            let info = decoder.info().context("JPEG has no image info")?;
            Ok((usize::from(info.width), usize::from(info.height)))
        }
    }
}

pub(super) fn decode_image(bytes: &[u8], mime: Option<&str>) -> Result<RgbImage> {
    match image_format(bytes, mime)? {
        ImageFormat::Png => decode_png(bytes),
        ImageFormat::Jpeg => decode_jpeg(bytes),
    }
}

fn decode_png(bytes: &[u8]) -> Result<RgbImage> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder.read_info()?;
    ensure!(
        reader.info().animation_control.is_none(),
        "animated PNG not supported"
    );
    let width = reader.info().width as usize;
    let height = reader.info().height as usize;
    let count = checked_pixels(width, height)?;
    let size = reader
        .output_buffer_size()
        .context("PNG output size overflow")?;
    ensure!(size <= count * 4, "unsupported PNG pixel depth");
    let mut buffer = vec![0; size];
    let output = reader.next_frame(&mut buffer)?;
    ensure!(
        output.bit_depth == png::BitDepth::Eight,
        "PNG must decode to 8-bit channels"
    );
    let channels = output.color_type.samples();
    let mut pixels = Vec::with_capacity(count * 3);
    for pixel in buffer[..output.buffer_size()].chunks_exact(channels) {
        match output.color_type {
            png::ColorType::Rgb | png::ColorType::Rgba => pixels.extend_from_slice(&pixel[..3]),
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => {
                pixels.extend_from_slice(&[pixel[0]; 3])
            }
            png::ColorType::Indexed => bail!("PNG palette was not expanded"),
        }
    }
    Ok(RgbImage {
        width,
        height,
        pixels,
    })
}

fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    decoder.read_info()?;
    let info = decoder.info().context("JPEG has no image info")?;
    let width = usize::from(info.width);
    let height = usize::from(info.height);
    let count = checked_pixels(width, height)?;
    let decoded = decoder.decode()?;
    let pixels = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => decoded,
        jpeg_decoder::PixelFormat::L8 => decoded.into_iter().flat_map(|v| [v; 3]).collect(),
        jpeg_decoder::PixelFormat::CMYK32 => {
            let mut rgb = Vec::with_capacity(count * 3);
            for p in decoded.as_chunks::<4>().0 {
                for &c in &p[..3] {
                    rgb.push(((u32::from(255 - c) * u32::from(255 - p[3]) + 127) / 255) as u8);
                }
            }
            rgb
        }
        jpeg_decoder::PixelFormat::L16 => bail!("16-bit JPEG is not supported"),
    };
    ensure!(pixels.len() == count * 3, "JPEG output size mismatch");
    Ok(RgbImage {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::detok::Shared;

    #[test]
    fn smart_resize_matches_qwen_grid_and_rejects_invalid_dimensions() -> Result<()> {
        assert_eq!(smart_resize(1024, 1024, 65536, 16777216)?, (1024, 1024));
        assert_eq!(smart_resize(1500, 900, 65536, 16777216)?, (1504, 896));
        assert_eq!(smart_resize(64, 64, 65536, 16777216)?, (256, 256));
        assert_eq!(smart_resize(80, 80, 1024, 16777216)?, (64, 64));
        assert!(smart_resize(0, 32, 65536, 16777216).is_err());
        assert!(smart_resize(1, 201, 65536, 16777216).is_err());
        Ok(())
    }

    #[test]
    fn png_data_url_decodes_rgb_without_alpha_compositing() -> Result<()> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()?
                .write_image_data(&[255, 0, 0, 0, 0, 128, 255, 255])?;
        }
        let data = base64::encode(&bytes);
        let image = load_image(&format!("data:image/png;base64,{data}"))?;
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.pixels, [255, 0, 0, 0, 128, 255]);
        assert!(load_image(&format!("data:image/jpeg;base64,{data}")).is_err());
        assert!(load_image("https://example.com/image.png").is_err());
        assert!(load_image("file:///etc/passwd").is_err());
        assert!(load_image("data:image/gif;base64,R0lGODlh").is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_JPEG_REFERENCE generated with Pillow"]
    fn jpeg_rgb_gray_and_cmyk_match_reference() -> Result<()> {
        let path = std::env::var("FF_JPEG_REFERENCE")?;
        let cases: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
        for case in cases.as_array().context("JPEG cases")? {
            let file = case["file"].as_str().context("JPEG path")?;
            let image = decode_image(&std::fs::read(file)?, None)?;
            let expected = std::fs::read(case["rgb"].as_str().context("RGB path")?)?;
            assert_eq!(image.width as u64, case["width"].as_u64().unwrap());
            assert_eq!(image.height as u64, case["height"].as_u64().unwrap());
            assert_eq!(image.pixels, expected, "{file}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_Q38_MODEL, FF_Q38_FIXTURE_DIR and FF_Q38_IMAGE_DIR"]
    fn v5_processor_pixels_grid_and_ids() -> Result<()> {
        let model = std::env::var("FF_Q38_MODEL")?;
        let fixtures = std::env::var("FF_Q38_FIXTURE_DIR")?;
        let media = std::env::var("FF_Q38_IMAGE_DIR")?;
        let processor = Processor::load(Path::new(&model))?.context("Qwen image processor")?;
        let tokenizer = Shared::load(Path::new(&model))?;
        let prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe every text block and figure in this document image, then summarize its purpose.";
        for (name, file) in [
            ("img-report", "img1.png"),
            ("img-nonsquare", "img-nonsquare.png"),
            ("img-marker", "img-marker.png"),
            ("img-tiny", "img-tiny.png"),
        ] {
            let bytes = std::fs::read(Path::new(&fixtures).join(format!("{name}.safetensors")))?;
            let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
            let integers = |key: &str| -> Result<Vec<i64>> {
                Ok(tensors
                    .tensor(key)?
                    .data()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect())
            };
            let expected_grid = integers("image_grid_thw")?;
            let image = decode_image(&std::fs::read(Path::new(&media).join(file))?, None)?;
            let packed = processor.prepare_rgb(&image)?;
            assert_eq!(
                [
                    packed.grid.t as i64,
                    packed.grid.h as i64,
                    packed.grid.w as i64
                ],
                expected_grid.as_slice(),
                "{name} grid"
            );
            let expected = tensors.tensor("pixel_values_fp32")?;
            let bf16 = tensors.tensor("pixel_values_bf16")?;
            let u8_values = tensors.tensor("pixel_values_uint8")?;
            assert_eq!(expected.shape(), [packed.grid.patch_rows()?, 1536]);
            let mut f32_max = 0.0_f32;
            let mut bf16_max = 0.0_f32;
            let mut bf16_different = 0;
            let mut u8_max = 0_u8;
            for (((actual, f32_bytes), bf16_bytes), &byte) in packed
                .pixel_values
                .iter()
                .zip(expected.data().as_chunks::<4>().0)
                .zip(bf16.data().as_chunks::<2>().0)
                .zip(u8_values.data())
            {
                let reference = f32::from_le_bytes(*f32_bytes);
                let reference_bf16 = half::bf16::from_bits(u16::from_le_bytes(*bf16_bytes));
                let actual_bf16 = half::bf16::from_f32(*actual);
                f32_max = f32_max.max((actual - reference).abs());
                bf16_max = bf16_max.max((actual_bf16.to_f32() - reference_bf16.to_f32()).abs());
                bf16_different += usize::from(actual_bf16.to_bits() != reference_bf16.to_bits());
                let actual_byte = ((actual + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                u8_max = u8_max.max(actual_byte.abs_diff(byte));
            }
            let resized = image.width != packed.grid.w as usize * 16
                || image.height != packed.grid.h as usize * 16;
            let tolerance = if resized {
                4.0 / 255.0 + 4.0 * f32::EPSILON
            } else {
                4.0 * f32::EPSILON
            };
            println!(
                "{}",
                serde_json::json!({"case":name,"grid":expected_grid,"resized":resized,"u8_max":u8_max,"fp32_max":f32_max,"bf16_max":bf16_max,"bf16_different":bf16_different,"fp32_tolerance":tolerance})
            );
            assert!(
                u8_max <= if resized { 2 } else { 0 },
                "{name} resized byte error"
            );
            assert!(f32_max <= tolerance, "{name} fp32 error");
            assert!(bf16_max <= tolerance + 1.0 / 256.0, "{name} bf16 error");
            if !resized {
                assert_eq!(bf16_different, 0, "{name} bf16 identity");
            }
            let ids = tokenizer.encode_chat(prompt)?;
            let (expanded, prepared) = processor.expand(&ids, vec![packed], 4096)?;
            let expected_ids: Vec<_> = integers("input_ids")?
                .into_iter()
                .map(|i| i as u32)
                .collect();
            assert_eq!(expanded, expected_ids, "{name} processor IDs");
            assert_eq!(prepared.grids.len(), 1);
        }
        Ok(())
    }
}
