//! Vision tower for Qwen3.5-27B: preprocessing + ViT + merger, CPU f32
//! (candle), ported from ff-h3 (Qwen3-VL family), minus video/deepstack.
//! Tower gate references HF in f32 (bf16 is chaotic pre-merger); e2e gate
//! (tests/vision_e2e.rs) covers the deployment path.

use crate::config::{VISION_ROPE_THETA, VisionConfig};
use crate::weights::Qwen35Weights;
use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;

const LAYER_NORM_EPS: f64 = 1e-6;

pub struct RgbImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "RGB image dimensions must be non-zero"
        );
        let expected = width
            .checked_mul(height)
            .and_then(|p| p.checked_mul(3))
            .context("RGB image size overflow")?;
        ensure!(
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

    /// Decode a single, non-animated RGB8 PNG.
    pub fn from_png(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).with_context(|| format!("failed to read PNG {}", path.display()))?;
        let decoder = png::Decoder::new(Cursor::new(&bytes));
        let mut reader = decoder.read_info().context("failed to read PNG header")?;
        let info = reader.info();
        ensure!(
            info.animation_control.is_none() && info.frame_control.is_none(),
            "PNG must not be animated"
        );
        ensure!(
            info.color_type == png::ColorType::Rgb && info.bit_depth == png::BitDepth::Eight,
            "PNG must use RGB8 encoding"
        );
        let width = usize::try_from(info.width).context("PNG width exceeds usize")?;
        let height = usize::try_from(info.height).context("PNG height exceeds usize")?;
        let expected = width
            .checked_mul(height)
            .and_then(|p| p.checked_mul(3))
            .context("decoded PNG size overflow")?;
        ensure!(
            reader.output_buffer_size() == Some(expected),
            "decoded PNG buffer size does not match RGB dimensions"
        );
        let mut pixels = vec![0u8; expected];
        reader
            .next_frame(&mut pixels)
            .context("failed to decode PNG pixels")?;
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

/// Patch grid in patch units (temporal, height, width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionGrid {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

impl VisionGrid {
    pub fn patch_count(self) -> Result<usize> {
        self.temporal
            .checked_mul(self.height)
            .and_then(|c| c.checked_mul(self.width))
            .context("vision patch count overflow")
    }

    pub fn merged_count(self, merge: usize) -> Result<usize> {
        ensure!(
            self.height.is_multiple_of(merge) && self.width.is_multiple_of(merge),
            "vision grid is not divisible by the spatial merge"
        );
        self.temporal
            .checked_mul(self.height / merge)
            .and_then(|c| c.checked_mul(self.width / merge))
            .context("merged vision token count overflow")
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ProcessorSize {
    longest_edge: usize,
    shortest_edge: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProcessorConfig {
    size: ProcessorSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

impl ProcessorConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let raw = fs::read_to_string(dir.join("preprocessor_config.json"))
            .with_context(|| format!("preprocessor_config.json under {}", dir.display()))?;
        serde_json::from_str(&raw).context("parse preprocessor_config.json")
    }

    pub fn merge_size(&self) -> usize {
        self.merge_size
    }
}

/// Qwen smart_resize: snap to factor, round-ties-even, pixel-budget clamped.
pub fn smart_resize_image(
    height: usize,
    width: usize,
    factor: usize,
    minimum_pixels: usize,
    maximum_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(
        height.max(width) as f64 / height.min(width) as f64 <= 200.,
        "image aspect ratio must not exceed 200:1"
    );
    // HF clamps small edges UP to the factor (max(factor, round_by_factor)),
    // it does not reject them.
    let mut resized_height = round_ties_even(height as f64 / factor as f64) as usize * factor;
    let mut resized_width = round_ties_even(width as f64 / factor as f64) as usize * factor;
    resized_height = resized_height.max(factor);
    resized_width = resized_width.max(factor);
    let rounded_area = resized_height
        .checked_mul(resized_width)
        .context("image resize area overflow")?;
    if rounded_area > maximum_pixels {
        let beta = ((height * width) as f64 / maximum_pixels as f64).sqrt();
        resized_height = (height as f64 / beta / factor as f64).floor() as usize * factor;
        resized_width = (width as f64 / beta / factor as f64).floor() as usize * factor;
    } else if rounded_area < minimum_pixels {
        let beta = (minimum_pixels as f64 / (height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    ensure!(
        resized_height > 0 && resized_width > 0,
        "image resize produced a zero dimension"
    );
    Ok((resized_height, resized_width))
}

fn round_ties_even(value: f64) -> f64 {
    value.round_ties_even()
}

#[derive(Clone)]
struct AxisSample {
    taps: Vec<(usize, f32)>,
}

/// Bicubic resize matching torchvision's uint8 antialias path (cubic a=-0.5,
/// border-clipped taps, u8-rounded intermediate). Fixture-pinned; do NOT
/// unify with ff-h3's resize (a=-0.75, different reference).
fn resize_bicubic_antialias(image: &RgbImage, width: usize, height: usize) -> Result<RgbImage> {
    ensure!(width > 0 && height > 0, "resize target must be non-zero");
    if image.width == width && image.height == height {
        return RgbImage::new(width, height, image.pixels.clone());
    }
    let horizontal = resize_axis_samples(image.width, width);
    let vertical = resize_axis_samples(image.height, height);
    let mut intermediate = vec![0f32; image.height * width * 3];
    for row in 0..image.height {
        for (column, sample) in horizontal.iter().enumerate() {
            for (channel, out) in intermediate
                [(row * width + column) * 3..(row * width + column) * 3 + 3]
                .iter_mut()
                .enumerate()
            {
                *out = sample
                    .taps
                    .iter()
                    .map(|&(source, weight)| {
                        image.pixels[(row * image.width + source) * 3 + channel] as f32 * weight
                    })
                    .sum::<f32>()
                    .round_ties_even()
                    .clamp(0., 255.);
            }
        }
    }
    let mut pixels = vec![0u8; height * width * 3];
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
                pixels[(row * width + column) * 3 + channel] =
                    value.round_ties_even().clamp(0., 255.) as u8;
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
            let mut taps = Vec::new();
            for input in first..=last {
                if !(0..source as isize).contains(&input) {
                    continue;
                }
                taps.push((
                    input as usize,
                    cubic_kernel((center - input as f64) / scale),
                ));
            }
            let total: f64 = taps.iter().map(|&(_, w)| w).sum();
            AxisSample {
                taps: taps
                    .into_iter()
                    .map(|(i, w)| (i, (w / total) as f32))
                    .collect(),
            }
        })
        .collect()
}

fn cubic_kernel(distance: f64) -> f64 {
    let distance = distance.abs();
    let coefficient = -0.5;
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

/// Merge-major flattened patches, (x/255 - mean)/std; frame duplicated to
/// fill temporal_patch_size.
fn patchify_image(
    frame: &RgbImage,
    grid: VisionGrid,
    processor: &ProcessorConfig,
    output: &mut Vec<f32>,
) -> Result<()> {
    let patch = processor.patch_size;
    let merge = processor.merge_size;
    let temporal = processor.temporal_patch_size;
    ensure!(grid.temporal == 1, "image grid temporal must be 1");
    let height = grid.height * patch;
    let width = grid.width * patch;
    ensure!(
        frame.height == height && frame.width == width,
        "image dimensions do not match patch grid"
    );
    for block_h in 0..grid.height / merge {
        for block_w in 0..grid.width / merge {
            for in_block_h in 0..merge {
                for in_block_w in 0..merge {
                    let patch_h = block_h * merge + in_block_h;
                    let patch_w = block_w * merge + in_block_w;
                    for channel in 0..3 {
                        for _temporal_offset in 0..temporal {
                            for row in 0..patch {
                                let source_row = patch_h * patch + row;
                                for column in 0..patch {
                                    let source_column = patch_w * patch + column;
                                    let byte = frame.pixels
                                        [(source_row * width + source_column) * 3 + channel];
                                    // Match the HF fast processor's op order:
                                    // rescale is a MULTIPLY by 1/255, then
                                    // normalize (x - mean) / std.
                                    output.push(
                                        (byte as f32 * (1.0 / 255.)
                                            - processor.image_mean[channel])
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
    Ok(())
}

/// Full image preprocessing: smart_resize -> bicubic -> patchify.
/// Returns (patches, grid); patches are row-major [t*h*w, 3*temporal*p*p].
pub fn preprocess_image(
    image: &RgbImage,
    processor: &ProcessorConfig,
) -> Result<(Vec<f32>, VisionGrid)> {
    let factor = processor.patch_size * processor.merge_size;
    let (height, width) = smart_resize_image(
        image.height,
        image.width,
        factor,
        processor.size.shortest_edge,
        processor.size.longest_edge,
    )?;
    let resized = resize_bicubic_antialias(image, width, height)?;
    let grid = VisionGrid {
        temporal: 1,
        height: height / processor.patch_size,
        width: width / processor.patch_size,
    };
    let mut patches = Vec::new();
    patchify_image(&resized, grid, processor, &mut patches)?;
    Ok((patches, grid))
}

fn bilinear_axis(index: usize, target_size: usize, source_size: usize) -> (usize, usize, f32, f32) {
    if target_size <= 1 {
        return (0, 0, 1., 0.);
    }
    // align_corners=true
    let source = index as f64 * (source_size - 1) as f64 / (target_size - 1) as f64;
    let lower = source.floor() as usize;
    let upper = (lower + 1).min(source_size - 1);
    let upper_weight = (source - lower as f64) as f32;
    (lower, upper, 1. - upper_weight, upper_weight)
}

/// Per-patch bilinear interpolation indices/weights into the learned
/// position table, in merge-block order. Ported from ff-h3.
fn learned_position_interpolation(
    grid: VisionGrid,
    source_side: usize,
    merge: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let rows = grid.patch_count()?;
    let mut indices = Vec::with_capacity(rows * 4);
    let mut weights = Vec::with_capacity(rows * 4);
    ensure!(
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
    Ok((indices, weights))
}

/// 2D axial rope tables, one row per patch in merge-block order,
/// [freq_h, freq_w] x2 layout (theta = VISION_ROPE_THETA, the HF default).
fn vision_rotary_tables(
    grid: VisionGrid,
    merge: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    ensure!(
        head_dim.is_multiple_of(4),
        "vision head dimension must be divisible by four"
    );
    let frequencies_per_axis = head_dim / 4;
    let inverse: Vec<f32> = (0..frequencies_per_axis)
        .map(|i| {
            (VISION_ROPE_THETA).powf(-(2.0 * i as f64) / (2 * frequencies_per_axis) as f64) as f32
        })
        .collect();
    let rows = grid.patch_count()?;
    let mut cos = Vec::with_capacity(rows * head_dim);
    let mut sin = Vec::with_capacity(rows * head_dim);
    for _ in 0..grid.temporal {
        for block_h in 0..grid.height / merge {
            for block_w in 0..grid.width / merge {
                for in_h in 0..merge {
                    for in_w in 0..merge {
                        let h = (block_h * merge + in_h) as f32;
                        let w = (block_w * merge + in_w) as f32;
                        for _half in 0..2 {
                            for &f in &inverse {
                                cos.push((h * f).cos());
                                sin.push((h * f).sin());
                            }
                            for &f in &inverse {
                                cos.push((w * f).cos());
                                sin.push((w * f).sin());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((cos, sin))
}

fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::layer_norm(
        x,
        weight,
        bias,
        LAYER_NORM_EPS as f32,
    )?)
}

fn linear(tensors: &BTreeMap<String, Tensor>, base: &str, x: &Tensor) -> Result<Tensor> {
    let weight = tensors
        .get(&format!("{base}.weight"))
        .with_context(|| format!("missing {base}.weight"))?;
    let bias = tensors
        .get(&format!("{base}.bias"))
        .with_context(|| format!("missing {base}.bias"))?;
    Ok(Linear::new(weight.clone(), Some(bias.clone())).forward(x)?)
}

/// erf GELU (merger); blocks use tanh-gelu below. Matches HF exactly:
/// merger nn.GELU() is erf, hidden_act gelu_pytorch_tanh is the blocks.
fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let shifted = x.broadcast_sub(&x.max_keepdim(candle_core::D::Minus1)?)?;
    let exp = shifted.exp()?;
    exp.broadcast_div(&exp.sum_keepdim(candle_core::D::Minus1)?)
        .map_err(Into::into)
}

/// The Qwen3.5 vision tower (CPU f32).
pub struct VisionTower {
    config: VisionConfig,
    device: Device,
    tensors: BTreeMap<String, Tensor>,
    head_dim: usize,
}

impl VisionTower {
    pub fn load(weights: &Qwen35Weights, config: &VisionConfig) -> Result<Self> {
        let device = Device::Cpu;
        let mut tensors = BTreeMap::new();
        let mut load = |name: &str| -> Result<()> {
            let (shape, values) = weights.bf16_tensor(name)?;
            tensors.insert(name.to_string(), Tensor::from_vec(values, shape, &device)?);
            Ok(())
        };
        load("model.visual.patch_embed.proj.weight")?;
        load("model.visual.patch_embed.proj.bias")?;
        load("model.visual.pos_embed.weight")?;
        for block in 0..config.depth {
            for name in [
                "norm1.weight",
                "norm1.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias",
            ] {
                load(&format!("model.visual.blocks.{block}.{name}"))?;
            }
        }
        for name in [
            "merger.norm.weight",
            "merger.norm.bias",
            "merger.linear_fc1.weight",
            "merger.linear_fc1.bias",
            "merger.linear_fc2.weight",
            "merger.linear_fc2.bias",
        ] {
            load(&format!("model.visual.{name}"))?;
        }
        ensure!(
            config.hidden_size.is_multiple_of(config.num_heads),
            "vision hidden not divisible by heads"
        );
        Ok(Self {
            head_dim: config.hidden_size / config.num_heads,
            config: config.clone(),
            device,
            tensors,
        })
    }

    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing vision tensor {name}"))
    }

    /// patches: row-major [t*h*w, in_channels*temporal*p*p] from
    /// preprocess_image. Returns row-major [merged_count, out_hidden_size].
    pub fn forward(&self, patches: &[f32], grid: VisionGrid) -> Result<Vec<f32>> {
        let v = &self.config;
        let rows = grid.patch_count()?;
        let patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size * v.patch_size;
        ensure!(
            patches.len() == rows * patch_dim,
            "patch buffer {} != {rows} x {patch_dim}",
            patches.len()
        );
        let x = Tensor::from_vec(patches.to_vec(), (rows, patch_dim), &self.device)?;

        // Patch projection: Conv3d with stride=kernel == Linear over the
        // flattened patch.
        let weight = self.tensor("model.visual.patch_embed.proj.weight")?;
        let bias = self.tensor("model.visual.patch_embed.proj.bias")?;
        let weight = weight.reshape((v.hidden_size, patch_dim))?;
        let mut x = Linear::new(weight, Some(bias.clone())).forward(&x)?;

        // Learned position table, bilinear-interpolated (align_corners).
        let table = self.tensor("model.visual.pos_embed.weight")?;
        let side = (v.num_position_embeddings as f64).sqrt() as usize;
        ensure!(
            side * side == v.num_position_embeddings,
            "position table {} is not square",
            v.num_position_embeddings
        );
        let (indices, weights) = learned_position_interpolation(grid, side, v.spatial_merge_size)?;
        let indices = Tensor::from_vec(indices, (rows * 4,), &self.device)?;
        let gathered = table
            .index_select(&indices, 0)?
            .reshape((rows, 4, v.hidden_size))?;
        let interpolation = Tensor::from_vec(weights, (rows, 4, 1), &self.device)?;
        let position = gathered.broadcast_mul(&interpolation)?.sum(1)?;
        x = x.add(&position)?;

        let (cos, sin) = vision_rotary_tables(grid, v.spatial_merge_size, self.head_dim)?;
        let cos = Tensor::from_vec(cos, (rows, self.head_dim), &self.device)?;
        let sin = Tensor::from_vec(sin, (rows, self.head_dim), &self.device)?;

        for block in 0..v.depth {
            let prefix = format!("model.visual.blocks.{block}");
            // Attention sub-block.
            let normalized = layer_norm(
                &x,
                self.tensor(&format!("{prefix}.norm1.weight"))?,
                self.tensor(&format!("{prefix}.norm1.bias"))?,
            )?;
            let qkv = linear(&self.tensors, &format!("{prefix}.attn.qkv"), &normalized)?
                .reshape((rows, 3, v.num_heads, self.head_dim))?;
            let query = qkv.narrow(1, 0, 1)?.squeeze(1)?;
            let key = qkv.narrow(1, 1, 1)?.squeeze(1)?;
            let value = qkv.narrow(1, 2, 1)?.squeeze(1)?;
            // 2D rope: x*cos + rotate_half(x)*sin, f32.
            // 2D rope per patch row: x*cos + rotate_half(x)*sin.
            let rope = |t: &Tensor| -> Result<Tensor> {
                let c = cos.unsqueeze(1)?;
                let s = sin.unsqueeze(1)?;
                Ok(t.broadcast_mul(&c)?
                    .add(&rotate_half_tensor(t)?.broadcast_mul(&s)?)?)
            };
            let query = rope(&query)?;
            let key = rope(&key)?;
            let query = query.transpose(0, 1)?.contiguous()?;
            let key = key.transpose(0, 1)?.contiguous()?;
            let value = value.transpose(0, 1)?.contiguous()?;
            let key_t = key.transpose(1, 2)?.contiguous()?;
            let scores = query
                .matmul(&key_t)?
                .affine(1. / (self.head_dim as f64).sqrt(), 0.)?;
            let probabilities = softmax_last_dim(&scores)?;
            let attended = probabilities
                .matmul(&value)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((rows, v.hidden_size))?;
            let output = linear(&self.tensors, &format!("{prefix}.attn.proj"), &attended)?;
            x = x.add(&output)?;
            // MLP sub-block (tanh gelu).
            let normalized = layer_norm(
                &x,
                self.tensor(&format!("{prefix}.norm2.weight"))?,
                self.tensor(&format!("{prefix}.norm2.bias"))?,
            )?;
            let hidden = linear(
                &self.tensors,
                &format!("{prefix}.mlp.linear_fc1"),
                &normalized,
            )?;
            let hidden = gelu_tanh(&hidden)?;
            let output = linear(&self.tensors, &format!("{prefix}.mlp.linear_fc2"), &hidden)?;
            x = x.add(&output)?;
        }

        // Merger: pre-shuffle LN over hidden, then 2x2 merge -> fc1 -> erf
        // gelu -> fc2.
        let merge_unit = v.spatial_merge_size * v.spatial_merge_size;
        ensure!(
            rows.is_multiple_of(merge_unit),
            "vision sequence is not divisible by the merge unit"
        );
        let merged_rows = rows / merge_unit;
        let merged_hidden = v.hidden_size * merge_unit;
        let normalized = layer_norm(
            &x,
            self.tensor("model.visual.merger.norm.weight")?,
            self.tensor("model.visual.merger.norm.bias")?,
        )?
        .reshape((merged_rows, merged_hidden))?;
        let first = linear(&self.tensors, "model.visual.merger.linear_fc1", &normalized)?;
        let first = gelu_erf(&first)?;
        let out = linear(&self.tensors, "model.visual.merger.linear_fc2", &first)?;
        ensure!(
            out.dims() == [merged_rows, v.out_hidden_size],
            "merger output {:?} != [{merged_rows}, {}]",
            out.dims(),
            v.out_hidden_size
        );
        Ok(out.flatten_all()?.to_vec1::<f32>()?)
    }
}

fn rotate_half_tensor(x: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(candle_core::D::Minus1)?;
    ensure!(head_dim.is_multiple_of(2), "rotary dimension must be even");
    let half = head_dim / 2;
    let first = x.narrow(candle_core::D::Minus1, 0, half)?;
    let second = x.narrow(candle_core::D::Minus1, half, half)?;
    Tensor::cat(&[&second.neg()?, &first], candle_core::D::Minus1).map_err(Into::into)
}

/// Per-token mrope positions [t,h,w] + decode delta (max(pos)+1 - seq_len).
/// `types`: 0 = text, 1 = image pad run (one per grid, in order).
pub fn multimodal_positions(
    types: &[u8],
    grids: &[VisionGrid],
    merge: usize,
) -> Result<(Vec<[i32; 3]>, i64)> {
    let mut positions = Vec::with_capacity(types.len());
    let mut current = 0usize;
    let mut image_cursor = 0usize;
    let mut start = 0usize;
    while start < types.len() {
        let kind = types[start];
        let mut end = start + 1;
        while end < types.len() && types[end] == kind {
            end += 1;
        }
        if kind == 0 {
            positions.extend((0..end - start).map(|offset| {
                let position = current + offset;
                [position, position, position]
            }));
            current = current
                .checked_add(end - start)
                .context("text position overflow")?;
        } else if kind == 1 {
            let grid = grids
                .get(image_cursor)
                .context("image pad run has no matching image grid")?;
            image_cursor += 1;
            let expected = grid.merged_count(merge)?;
            ensure!(
                end - start == expected,
                "vision pad run has {} tokens, expected {expected} from its grid",
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
                .context("vision position overflow")?;
        } else {
            bail!("unsupported multimodal token type {kind}");
        }
        start = end;
    }
    ensure!(
        image_cursor == grids.len(),
        "not every image grid has a pad run"
    );
    ensure!(
        positions.len() == types.len(),
        "mrope position count mismatch"
    );
    let max_pos = positions.iter().flatten().max().copied().unwrap_or(0);
    let delta = max_pos as i64 + 1 - types.len() as i64;
    Ok((
        positions
            .into_iter()
            .map(|p| [p[0] as i32, p[1] as i32, p[2] as i32])
            .collect(),
        delta,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::paths::checkpoint_dir;

    /// Gate 1: preprocessing vs the HF processor fixture (grid exact,
    /// pixels to 1e-6).
    #[test]
    fn preprocess_matches_hf_processor_fixture() {
        let dir = Path::new("src/testdata");
        let png = dir.join("vision_test.png");
        let fixture = dir.join("vision_processor_fixture.json");
        if !png.exists() || !fixture.exists() {
            return;
        }
        #[derive(Deserialize)]
        struct Fixture {
            grid_thw: [usize; 3],
            shape: Vec<usize>,
            pixel_values: Vec<f32>,
        }
        let fixture: Fixture = serde_json::from_str(&fs::read_to_string(fixture).unwrap()).unwrap();
        let image = RgbImage::from_png(&png).unwrap();
        // The fixture was dumped from the raw checkpoint dir; both share the
        // preprocessor_config.json contents.
        let proc = ProcessorConfig {
            size: ProcessorSize {
                longest_edge: 16777216,
                shortest_edge: 65536,
            },
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        };
        let (patches, grid) = preprocess_image(&image, &proc).unwrap();
        assert_eq!(
            [grid.temporal, grid.height, grid.width],
            fixture.grid_thw,
            "grid mismatch"
        );
        assert_eq!(
            patches.len(),
            fixture.pixel_values.len(),
            "patch element count"
        );
        assert_eq!(
            fixture.shape,
            vec![grid.temporal * grid.height * grid.width, 3 * 2 * 16 * 16],
            "fixture shape"
        );
        // f32 op-order ulps differ everywhere; the gate is on REAL
        // deviations: torchvision's uint8 rounding ties give an isolated
        // 1-level (2/255) diff; algorithmic mismatches measured 21+ levels
        // on 9% of pixels (wrong kernel/clipping variants).
        let mut max_abs = 0.0f32;
        let mut n_big = 0usize;
        for (a, b) in patches.iter().zip(&fixture.pixel_values) {
            let d = (a - b).abs();
            if d > 1e-6 {
                n_big += 1;
                max_abs = max_abs.max(d);
            }
        }
        assert!(
            max_abs <= 2.0 / 255.0 && n_big <= 8,
            "pixel max_abs {max_abs} over {n_big} big diffs (>1e-6)"
        );
    }

    /// Gate 2: mrope position builder must equal HF get_rope_index exactly,
    /// including the delta.
    #[test]
    fn positions_match_hf_rope_fixture() {
        let fixture = Path::new("src/testdata/vision_rope_fixture.json");
        if !fixture.exists() {
            return;
        }
        #[derive(Deserialize)]
        struct RopeFixture {
            input_ids: Vec<u32>,
            grid_thw: [usize; 3],
            position_ids: Vec<Vec<i32>>,
            rope_delta: f64,
        }
        let fixture: RopeFixture =
            serde_json::from_str(&fs::read_to_string(fixture).unwrap()).unwrap();
        let types: Vec<u8> = fixture
            .input_ids
            .iter()
            .map(|&t| if t == 248056 { 1 } else { 0 })
            .collect();
        let grids = [VisionGrid {
            temporal: fixture.grid_thw[0],
            height: fixture.grid_thw[1],
            width: fixture.grid_thw[2],
        }];
        let (positions, delta) = multimodal_positions(&types, &grids, 2).unwrap();
        assert_eq!(positions.len(), fixture.input_ids.len());
        for (i, p) in positions.iter().enumerate() {
            for (axis, &pv) in p.iter().enumerate() {
                assert_eq!(pv, fixture.position_ids[axis][i], "token {i} axis {axis}");
            }
        }
        assert_eq!(delta as f64, fixture.rope_delta, "rope delta");
    }

    /// Gate 3: the tower (patch embed -> pos embed -> 27 blocks -> merger)
    /// vs the HF bf16 fixture. Our side is f32, HF runs bf16 — the gate is
    /// at the bf16 rounding floor class (measured margins, not bit-exact).
    #[test]
    fn tower_matches_hf_fixture() {
        let Some(dir) = checkpoint_dir("Qwen/Qwen3.8-27B-int4") else {
            return;
        };
        let dir = &dir;
        let fixture_path = Path::new("src/testdata/vision_tower_fixture.json");
        let png = Path::new("src/testdata/vision_test.png");
        if !dir.exists() || !fixture_path.exists() || !png.exists() {
            return;
        }
        #[derive(Deserialize)]
        struct TowerFixture {
            shape: Vec<usize>,
            values: Vec<f32>,
        }
        let fixture: TowerFixture =
            serde_json::from_str(&fs::read_to_string(fixture_path).unwrap()).unwrap();
        let cfg = crate::config::Qwen35Config::from_model_dir(dir).unwrap();
        let weights = crate::weights::Qwen35Weights::open(dir).unwrap();
        let tower = VisionTower::load(&weights, cfg.vision_config.as_ref().unwrap()).unwrap();
        let image = RgbImage::from_png(png).unwrap();
        let proc = ProcessorConfig::from_model_dir(dir).unwrap();
        let (patches, grid) = preprocess_image(&image, &proc).unwrap();
        let out = tower.forward(&patches, grid).unwrap();
        assert_eq!(out.len(), fixture.values.len(), "tower output length");
        let merged = grid.merged_count(proc.merge_size).unwrap();
        assert_eq!(
            fixture.shape,
            vec![merged, cfg.vision_config.unwrap().out_hidden_size]
        );
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in out.iter().zip(&fixture.values) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / b.abs().max(1.0));
        }
        eprintln!("tower vs HF f32: max_abs {max_abs:.4} max_rel {max_rel:.4}");
        // The fixture is HF's tower in f32 (bf16 is measurably chaotic here:
        // bf16-vs-f32 HF diverges by hundreds of units pre-merger; the
        // merger LN re-normalizes, so the f32 reference is well-conditioned).
        // Measured: max_rel 0.0011, max_abs 0.023 — gate at 4x margin.
        assert!(
            max_rel <= 0.005,
            "tower max_rel {max_rel} exceeds the f32-reference class"
        );
    }
}
