use crate::model_spec::{Architecture, ModelSpec};
use crate::serve::detok::Shared;
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

/// Decoded-video metadata: decode order, average fps and duration in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoMetadata {
    pub total_num_frames: usize,
    pub fps: f64,
    pub duration: f64,
}

impl VideoMetadata {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.total_num_frames > 0,
            "video metadata: zero total frames"
        );
        ensure!(
            self.fps.is_finite() && self.fps > 0.0,
            "video metadata: invalid fps {}",
            self.fps
        );
        ensure!(
            self.duration.is_finite() && self.duration > 0.0,
            "video metadata: invalid duration {}",
            self.duration
        );
        Ok(())
    }
}

/// Python `round(value, 1)`: the nearest double to the one-decimal value.
fn round_to_one_decimal(value: f64) -> f64 {
    (value * 10.0).round_ties_even() / 10.0
}

/// `numpy.linspace(0, len - 1, count, dtype=int)` positions.
fn linspace_positions(len: usize, count: usize) -> Vec<usize> {
    if count == 1 {
        return vec![0];
    }
    let delta = (len - 1) as f64 / (count - 1) as f64;
    (0..count).map(|k| (delta * k as f64) as usize).collect()
}

/// Frames sampled from decode order, stack_frames = 1 (model definition `sample_frames`).
pub fn sample_frame_indices(
    meta: &VideoMetadata,
    max_num_frames: usize,
    stack_frames: u32,
) -> Result<Vec<usize>> {
    ensure!(
        stack_frames == 1,
        "stack_frames {stack_frames} is not claimed; first support samples stack_frames = 1"
    );
    ensure!(max_num_frames > 0, "zero max_num_frames");
    meta.validate()?;
    let mut indices = if meta.duration > max_num_frames as f64 {
        let count = (meta.duration / 0.1) as usize;
        let mut timestamps = Vec::with_capacity(count);
        for i in 0..count {
            timestamps.push(round_to_one_decimal(i as f64 * 0.1));
        }
        let mut main: Vec<usize> = timestamps
            .iter()
            .map(|&timestamp| {
                let index = (timestamp * meta.fps) as usize;
                index.min(meta.total_num_frames - 1)
            })
            .collect();
        if main.len() > max_num_frames {
            let positions = linspace_positions(main.len(), max_num_frames);
            main = positions.iter().map(|&position| main[position]).collect();
        }
        main
    } else {
        let seconds = meta.duration.ceil() as usize;
        let mut main = Vec::with_capacity(seconds);
        for i in 0..seconds {
            let index = (i as f64 * meta.fps) as usize;
            ensure!(
                index < meta.total_num_frames,
                "video metadata: sampled frame {index} outside {} frames",
                meta.total_num_frames
            );
            main.push(index);
        }
        main
    };
    ensure!(!indices.is_empty(), "video sampling produced no frames");
    indices.shrink_to_fit();
    Ok(indices)
}

pub struct PackedImage {
    pub grids: Vec<ImageGrid>,
    pub rows: u32,
    pub cols: u32,
    pub pixel_values: Vec<f32>,
}

/// Video processing options for the first support (stack_frames = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoOptions {
    pub max_num_frames: usize,
    pub stack_frames: u32,
    /// Slice cap per frame; `None` uses the processor configuration's value.
    pub max_slice_nums: Option<usize>,
}

/// One prepared video frame: its slice grid and per-crop patch grids (thumbnail, then slices).
#[derive(Debug)]
pub struct VideoFrame {
    pub rows: u32,
    pub cols: u32,
    pub grids: Vec<ImageGrid>,
}

/// Prepared video: packed crops in frame order.
#[derive(Debug)]
pub struct PackedVideo {
    pub frames: Vec<VideoFrame>,
    pub pixel_values: Vec<f32>,
}

impl PackedVideo {
    /// Per-crop patch grids in frame order, matching the prompt's crop runs.
    pub fn crops(&self) -> impl Iterator<Item = &ImageGrid> {
        self.frames.iter().flat_map(|frame| frame.grids.iter())
    }
}

pub struct PlannedImage {
    bytes: Vec<u8>,
    mime: String,
    grids: Vec<ImageGrid>,
    rows: u32,
    cols: u32,
}

impl PlannedImage {
    fn feature_rows(&self, merge: u32) -> Result<usize> {
        self.grids.iter().try_fold(0usize, |total, grid| {
            total
                .checked_add(grid.feature_rows(merge)?)
                .context("image token count overflow")
        })
    }
}

fn patch_width(spec: &ModelSpec) -> Result<usize> {
    spec.vision
        .as_ref()
        .context("image input requires a vision configuration")?
        .patch_elements(spec.architecture)
}

pub struct PreparedImages {
    pub modality_ids: Vec<u8>,
    pub pixel_values: Vec<f32>,
    pub grids: Vec<ImageGrid>,
}

impl PreparedImages {
    pub fn save(&self, spec: &ModelSpec, ids: &[u32], path: &Path) -> Result<()> {
        let width = patch_width(spec)?;
        ensure!(
            self.pixel_values.len().is_multiple_of(width),
            "pixel_values length {} is not a multiple of the model's {width}-element patch",
            self.pixel_values.len()
        );
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
            vec![self.pixel_values.len() / width, width],
            &self.pixel_values,
        )?;
        tensors.save_new(
            path,
            [("format".into(), "flyingfish-image-input-v1".into())].into(),
        )
    }

    pub fn read(spec: &ModelSpec, path: &Path, ids: &[u32]) -> Result<Self> {
        let width = patch_width(spec)?;
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
            pixels.dtype() == safetensors::Dtype::F32 && pixels.shape().len() == 2,
            "pixel_values_fp32 must be F32 [patches,{width}]"
        );
        ensure!(
            pixels.shape()[1] == width,
            "pixel_values_fp32 row width {} does not match the model's {width}-element patch",
            pixels.shape()[1]
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
struct QwenConfig {
    size: Size,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

#[derive(Clone, Deserialize)]
struct MiniCpmConfig {
    max_slice_nums: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    mrope_mode: String,
}

enum ImageConfig {
    Qwen(QwenConfig),
    MiniCpm(MiniCpmConfig),
}

struct Crop {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

/// `MiniCPMV4_6ImageProcessor.scale_resolution` defaults to 448.
const SCALE_RESOLUTION: usize = 448;

struct ImageId {
    start: u32,
    end: u32,
    digits: [u32; 10],
}

impl ImageId {
    fn read(tokenizer: &Shared) -> Result<Self> {
        let mut digits = [0u32; 10];
        for (value, slot) in digits.iter_mut().enumerate() {
            *slot = tokenizer.token_id(&value.to_string())?;
        }
        Ok(Self {
            start: tokenizer.token_id("<image_id>")?,
            end: tokenizer.token_id("</image_id>")?,
            digits,
        })
    }

    fn tag(&self, index: usize) -> Vec<u32> {
        let mut ids = vec![self.start];
        for digit in index.to_string().bytes() {
            ids.push(self.digits[usize::from(digit - b'0')]);
        }
        ids.push(self.end);
        ids
    }
}

pub struct Processor {
    spec: ModelSpec,
    config: ImageConfig,
    image_id: Option<ImageId>,
}

impl Processor {
    pub fn load(model: &Path, tokenizer: Option<&Shared>) -> Result<Option<Self>> {
        let spec = ModelSpec::read(model)?;
        let Some(vision) = spec.vision.as_ref() else {
            return Ok(None);
        };
        vision.validate(spec.architecture)?;
        let path = model.join("preprocessor_config.json");
        let config = match spec.architecture {
            Architecture::Qwen35Text => {
                let config: QwenConfig = serde_json::from_reader(std::fs::File::open(&path)?)?;
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
                ImageConfig::Qwen(config)
            }
            Architecture::MiniCpmV47 => {
                let config: MiniCpmConfig = serde_json::from_reader(std::fs::File::open(&path)?)?;
                ensure!(
                    config.max_slice_nums == 9,
                    "unsupported max_slice_nums: {}",
                    config.max_slice_nums
                );
                ensure!(
                    config.image_mean == [0.5; 3],
                    "unsupported image_mean: {:?}",
                    config.image_mean
                );
                ensure!(
                    config.image_std == [0.5; 3],
                    "unsupported image_std: {:?}",
                    config.image_std
                );
                ensure!(
                    config.mrope_mode == "canvas",
                    "unsupported mrope_mode: {:?}",
                    config.mrope_mode
                );
                ImageConfig::MiniCpm(config)
            }
            Architecture::Llama => return Ok(None),
        };
        let image_id = match spec.architecture {
            Architecture::MiniCpmV47 => Some(ImageId::read(
                tokenizer.context("MiniCPM image ids require the tokenizer")?,
            )?),
            _ => None,
        };
        Ok(Some(Self {
            spec,
            config,
            image_id,
        }))
    }

    pub fn merge(&self) -> u32 {
        self.spec
            .vision
            .as_ref()
            .map(|vision| vision.merge)
            .unwrap_or(0)
    }

    /// Samples frames and packs their crops for one decoded video (stack_frames = 1).
    pub fn prepare_video(
        &self,
        frames: &[RgbImage],
        meta: &VideoMetadata,
        options: &VideoOptions,
    ) -> Result<PackedVideo> {
        let indices = sample_frame_indices(meta, options.max_num_frames, options.stack_frames)?;
        ensure!(
            frames.len() == meta.total_num_frames,
            "decoded video holds {} frames, metadata reports {}",
            frames.len(),
            meta.total_num_frames
        );
        let ImageConfig::MiniCpm(base) = &self.config else {
            bail!("video input requires the MiniCPM image configuration")
        };
        let mut config = base.clone();
        if let Some(max) = options.max_slice_nums {
            ensure!(max > 0, "zero max_slice_nums");
            config.max_slice_nums = max;
        }
        let mut packed = PackedVideo {
            frames: Vec::with_capacity(indices.len()),
            pixel_values: Vec::new(),
        };
        for &index in &indices {
            let image = mini_cpm_prepare_rgb(&self.spec, &frames[index], &config)?;
            packed.frames.push(VideoFrame {
                rows: image.rows,
                cols: image.cols,
                grids: image.grids,
            });
            packed.pixel_values.extend_from_slice(&image.pixel_values);
        }
        Ok(packed)
    }

    pub fn prepare_rgb(&self, image: &RgbImage) -> Result<PackedImage> {
        let count = checked_pixels(image.width, image.height)?;
        ensure!(
            image.pixels.len() == count * 3,
            "RGB byte count does not match dimensions"
        );
        match &self.config {
            ImageConfig::Qwen(config) => qwen_prepare_rgb(image, config),
            ImageConfig::MiniCpm(config) => mini_cpm_prepare_rgb(&self.spec, image, config),
        }
    }

    pub fn prepare_url(&self, url: &str) -> Result<PackedImage> {
        self.prepare_planned(self.plan_url(url)?)
    }

    pub fn plan_url(&self, url: &str) -> Result<PlannedImage> {
        let (bytes, mime) = load_encoded(url)?;
        let (width, height) = image_dimensions(&bytes, Some(&mime))?;
        match &self.config {
            ImageConfig::Qwen(config) => {
                let (height, width) = smart_resize(
                    height,
                    width,
                    config.size.shortest_edge,
                    config.size.longest_edge,
                )?;
                Ok(PlannedImage {
                    bytes,
                    mime,
                    grids: vec![ImageGrid {
                        t: 1,
                        h: (height / 16) as u32,
                        w: (width / 16) as u32,
                    }],
                    rows: 0,
                    cols: 0,
                })
            }
            ImageConfig::MiniCpm(config) => {
                let vision = self
                    .spec
                    .vision
                    .as_ref()
                    .context("vision configuration missing")?;
                let (grids, rows, cols) =
                    mini_cpm_grids(height, width, vision.patch as usize, config.max_slice_nums)?;
                Ok(PlannedImage {
                    bytes,
                    mime,
                    grids,
                    rows: rows as u32,
                    cols: cols as u32,
                })
            }
        }
    }

    pub fn prepare_planned(&self, image: PlannedImage) -> Result<PackedImage> {
        let packed = self.prepare_rgb(&decode_image(&image.bytes, Some(&image.mime))?)?;
        ensure!(
            packed.grids == image.grids && packed.rows == image.rows && packed.cols == image.cols,
            "decoded image grid differs from header"
        );
        Ok(packed)
    }

    pub fn image_tokens(&self, image: &PlannedImage, index: usize) -> Result<usize> {
        let vision = self
            .spec
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        match &self.config {
            ImageConfig::Qwen(_) => image.feature_rows(vision.merge),
            ImageConfig::MiniCpm(_) => {
                let thumb = *image.grids.first().context("image has no crops")?;
                let tag = self
                    .image_id
                    .as_ref()
                    .context("image-id tokens missing")?
                    .tag(index);
                let mut tokens = tag.len() + 2 + thumb.feature_rows(vision.merge)?;
                if image.rows > 0 && image.cols > 0 {
                    ensure!(
                        image.grids.len() == 1 + image.rows as usize * image.cols as usize,
                        "crop count does not match the slice grid"
                    );
                    let slice = *image
                        .grids
                        .get(1)
                        .context("sliced image lacks a slice crop")?;
                    let per_slice = slice.feature_rows(vision.merge)?;
                    tokens += image.rows as usize - 1;
                    tokens += image.rows as usize * image.cols as usize * (2 + per_slice);
                }
                Ok(tokens)
            }
        }
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
            .enumerate()
            .try_fold(ids.len() - placeholders, |total, (index, image)| {
                total
                    .checked_add(self.image_tokens(image, index)?)
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
        match &self.config {
            ImageConfig::Qwen(_) => expand_qwen(&self.spec, vision, ids, images, capacity),
            ImageConfig::MiniCpm(_) => expand_mini_cpm(
                &self.spec,
                vision,
                self.image_id.as_ref().context("image-id tokens missing")?,
                ids,
                images,
                capacity,
            ),
        }
    }
}

fn qwen_prepare_rgb(image: &RgbImage, config: &QwenConfig) -> Result<PackedImage> {
    let (height, width) = smart_resize(
        image.height,
        image.width,
        config.size.shortest_edge,
        config.size.longest_edge,
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
                                    let value = resized[(row * width + col) * 3 + channel] as f32;
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
        grids: vec![ImageGrid {
            t: 1,
            h: gh as u32,
            w: gw as u32,
        }],
        rows: 0,
        cols: 0,
        pixel_values: pixels,
    })
}

fn mini_cpm_prepare_rgb(
    spec: &ModelSpec,
    image: &RgbImage,
    config: &MiniCpmConfig,
) -> Result<PackedImage> {
    let vision = spec
        .vision
        .as_ref()
        .context("vision configuration missing")?;
    let patch = vision.patch as usize;
    let (grids, rows, cols) =
        mini_cpm_grids(image.height, image.width, patch, config.max_slice_nums)?;
    let (source_h, source_w) = find_best_resize(
        image.height as f64,
        image.width as f64,
        SCALE_RESOLUTION,
        patch,
        rows == 0,
    );
    let mut crops = vec![Crop {
        pixels: resize_bicubic(image, source_w, source_h),
        width: source_w,
        height: source_h,
    }];
    if rows > 0 {
        let (refine_h, refine_w) = get_refine_size(
            image.height,
            image.width,
            rows,
            cols,
            SCALE_RESOLUTION,
            patch,
        )?;
        let refine = resize_bicubic(image, refine_w, refine_h);
        let patch_h = refine_h / rows;
        let patch_w = refine_w / cols;
        for row in 0..rows {
            for col in 0..cols {
                crops.push(crop_rect(
                    &refine,
                    refine_w,
                    row * patch_h,
                    col * patch_w,
                    patch_h,
                    patch_w,
                ));
            }
        }
    }
    let patch_h = crops[0].height / patch;
    let patch_w = crops[0].width / patch;
    ensure!(
        crops[0].height == patch_h * patch && crops[0].width == patch_w * patch,
        "source resize is not patch aligned"
    );
    let pixel_values = pack_mini_cpm(&crops, patch, &config.image_mean, &config.image_std)?;
    Ok(PackedImage {
        grids,
        rows: rows as u32,
        cols: cols as u32,
        pixel_values,
    })
}

fn crop_rect(
    image: &[u8],
    width: usize,
    top: usize,
    left: usize,
    height: usize,
    columns: usize,
) -> Crop {
    let mut pixels = Vec::with_capacity(height * columns * 3);
    for row in 0..height {
        let start = ((top + row) * width + left) * 3;
        pixels.extend_from_slice(&image[start..start + columns * 3]);
    }
    Crop {
        pixels,
        width: columns,
        height,
    }
}

fn pack_mini_cpm(
    crops: &[Crop],
    patch: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Vec<f32>> {
    let total_patches: usize = crops.iter().try_fold(0usize, |total, crop| {
        let row = crop.height / patch;
        let col = crop.width / patch;
        total
            .checked_add(row.checked_mul(col).context("crop patch count overflow")?)
            .context("patch count overflow")
    })?;
    let mut pixels = Vec::with_capacity(
        3usize
            .saturating_mul(patch)
            .saturating_mul(total_patches)
            .saturating_mul(patch),
    );
    for channel in 0..3 {
        for ky in 0..patch {
            for crop in crops {
                let cols = crop.width / patch;
                let rows = crop.height / patch;
                for patch_index in 0..rows * cols {
                    let (py, px) = (patch_index / cols, patch_index % cols);
                    for kx in 0..patch {
                        let y = py * patch + ky;
                        let x = px * patch + kx;
                        let value = crop.pixels[(y * crop.width + x) * 3 + channel] as f32;
                        let value = value * (1.0 / 255.0);
                        pixels.push((value - mean[channel]) / std[channel]);
                    }
                }
            }
        }
    }
    Ok(pixels)
}

fn mini_cpm_grids(
    height: usize,
    width: usize,
    patch: usize,
    max_slice_nums: usize,
) -> Result<(Vec<ImageGrid>, usize, usize)> {
    let sliced = get_sliced_grid(height, width, max_slice_nums, SCALE_RESOLUTION);
    let (source_h, source_w) = find_best_resize(
        height as f64,
        width as f64,
        SCALE_RESOLUTION,
        patch,
        sliced.is_none(),
    );
    let mut grids = vec![ImageGrid {
        t: 1,
        h: u32::try_from(source_h / patch).context("source grid overflows u32")?,
        w: u32::try_from(source_w / patch).context("source grid overflows u32")?,
    }];
    let (rows, cols) = sliced.unwrap_or((0, 0));
    if rows > 0 {
        let (refine_h, refine_w) =
            get_refine_size(height, width, rows, cols, SCALE_RESOLUTION, patch)?;
        let patch_h = refine_h / rows;
        let patch_w = refine_w / cols;
        ensure!(
            patch_h.is_multiple_of(patch) && patch_w.is_multiple_of(patch),
            "slice resize is not patch aligned"
        );
        let grid = ImageGrid {
            t: 1,
            h: u32::try_from(patch_h / patch).context("slice grid overflows u32")?,
            w: u32::try_from(patch_w / patch).context("slice grid overflows u32")?,
        };
        grids.extend(std::iter::repeat_n(grid, rows * cols));
    }
    Ok((grids, rows, cols))
}

fn ensure_divide(length: f64, divisor: usize) -> usize {
    let rounded = (length / divisor as f64).round_ties_even() as i64 * divisor as i64;
    rounded.max(divisor as i64) as usize
}

fn find_best_resize(
    height: f64,
    width: f64,
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut height, mut width) = (height, width);
    if height * width > (scale_resolution * scale_resolution) as f64 || allow_upscale {
        let aspect_ratio = width / height;
        height = (scale_resolution as f64 / aspect_ratio.sqrt()).trunc();
        width = (height * aspect_ratio).trunc();
    }
    let align = patch_size * 4;
    (ensure_divide(height, align), ensure_divide(width, align))
}

fn get_refine_size(
    height: usize,
    width: usize,
    rows: usize,
    cols: usize,
    scale_resolution: usize,
    patch_size: usize,
) -> Result<(usize, usize)> {
    let refine_width = ensure_divide(width as f64, cols);
    let refine_height = ensure_divide(height as f64, rows);
    let (best_height, best_width) = find_best_resize(
        refine_height as f64 / rows as f64,
        refine_width as f64 / cols as f64,
        scale_resolution,
        patch_size,
        true,
    );
    Ok((best_height * rows, best_width * cols))
}

fn get_sliced_grid(
    height: usize,
    width: usize,
    max_slice_nums: usize,
    scale_resolution: usize,
) -> Option<(usize, usize)> {
    let log_ratio = (width as f64 / height as f64).ln();
    let ratio = (width * height) as f64 / (scale_resolution * scale_resolution) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1usize, 1usize);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple.saturating_sub(1), multiple, multiple + 1] {
        if num_slices == 1 || num_slices > max_slice_nums {
            continue;
        }
        for num_rows in 1..=num_slices {
            if num_slices.is_multiple_of(num_rows) {
                let num_cols = num_slices / num_rows;
                let error = (log_ratio - (num_cols as f64 / num_rows as f64).ln()).abs();
                if error < min_error {
                    best = (num_rows, num_cols);
                    min_error = error;
                } else if error == min_error && num_rows > best.0 {
                    best = (num_rows, num_cols);
                }
            }
        }
    }
    Some(best)
}

fn expand_qwen(
    spec: &ModelSpec,
    vision: &crate::model_spec::VisionConfig,
    ids: &[u32],
    images: Vec<PackedImage>,
    capacity: usize,
) -> Result<(Vec<u32>, PreparedImages)> {
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
            let grid = *image.grids.first().context("image has no crops")?;
            let features = grid.feature_rows(vision.merge)?;
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
            prepared.grids.push(grid);
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
    prepared.input(&expanded).validate(spec, capacity)?;
    Ok((expanded, prepared))
}

fn expand_mini_cpm(
    spec: &ModelSpec,
    vision: &crate::model_spec::VisionConfig,
    image_id: &ImageId,
    ids: &[u32],
    images: Vec<PackedImage>,
    capacity: usize,
) -> Result<(Vec<u32>, PreparedImages)> {
    let markers = vision
        .minicpm
        .as_ref()
        .context("MiniCPM vision configuration missing")?;
    let mut expanded = Vec::new();
    let mut prepared = PreparedImages {
        modality_ids: Vec::new(),
        pixel_values: Vec::new(),
        grids: Vec::new(),
    };
    let mut packed: Vec<Vec<f32>> = Vec::new();
    let mut packed_patches: Vec<usize> = Vec::new();
    let mut images = images.into_iter();
    let mut index = 0usize;
    for &id in ids {
        if id == vision.image_token {
            let image = images
                .next()
                .context("more image placeholders than supplied images")?;
            let tag = image_id.tag(index);
            index += 1;
            let thumb = *image.grids.first().context("image has no crops")?;
            let thumb_tokens = thumb.feature_rows(vision.merge)?;
            let mut block = tag.len() + 2 + thumb_tokens;
            if image.rows > 0 && image.cols > 0 {
                ensure!(
                    image.grids.len() == 1 + image.rows as usize * image.cols as usize,
                    "crop count does not match the slice grid"
                );
                let slice = *image
                    .grids
                    .get(1)
                    .context("sliced image lacks a slice crop")?;
                let per_slice = slice.feature_rows(vision.merge)?;
                block += image.rows as usize - 1;
                block += image.rows as usize * image.cols as usize * (2 + per_slice);
            } else {
                ensure!(image.grids.len() == 1, "unsliced image carries slice crops");
            }
            ensure!(
                expanded
                    .len()
                    .checked_add(block)
                    .is_some_and(|n| n <= capacity),
                "image tokens exceed request capacity"
            );
            let tag_len = tag.len();
            expanded.extend(tag);
            prepared
                .modality_ids
                .extend(std::iter::repeat_n(0, tag_len));
            expanded.push(vision.start_token);
            prepared.modality_ids.push(0);
            expanded.extend(std::iter::repeat_n(id, thumb_tokens));
            prepared
                .modality_ids
                .extend(std::iter::repeat_n(1, thumb_tokens));
            expanded.push(vision.end_token);
            prepared.modality_ids.push(0);
            if image.rows > 0 && image.cols > 0 {
                let slice = *image
                    .grids
                    .get(1)
                    .context("sliced image lacks a slice crop")?;
                let per_slice = slice.feature_rows(vision.merge)?;
                for row in 0..image.rows {
                    if row > 0 {
                        expanded.push(markers.newline_token);
                        prepared.modality_ids.push(0);
                    }
                    for _ in 0..image.cols {
                        expanded.push(markers.slice_start_token);
                        prepared.modality_ids.push(0);
                        expanded.extend(std::iter::repeat_n(id, per_slice));
                        prepared
                            .modality_ids
                            .extend(std::iter::repeat_n(1, per_slice));
                        expanded.push(markers.slice_end_token);
                        prepared.modality_ids.push(0);
                    }
                }
            }
            packed_patches.push(image.grids.iter().try_fold(0usize, |total, grid| {
                total
                    .checked_add(grid.patch_rows()?)
                    .context("patch count overflow")
            })?);
            prepared.grids.extend(image.grids);
            packed.push(image.pixel_values);
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
    prepared.pixel_values = interleave_mini_cpm(&packed, &packed_patches, vision.patch as usize)?;
    prepared.input(&expanded).validate(spec, capacity)?;
    Ok((expanded, prepared))
}

fn interleave_mini_cpm(vectors: &[Vec<f32>], patches: &[usize], patch: usize) -> Result<Vec<f32>> {
    let total = patches.iter().try_fold(0usize, |total, &count| {
        total.checked_add(count).context("patch count overflow")
    })?;
    let expected = 3usize
        .checked_mul(patch)
        .and_then(|n| n.checked_mul(total))
        .and_then(|n| n.checked_mul(patch))
        .context("pixel count overflow")?;
    for (vector, &count) in vectors.iter().zip(patches) {
        ensure!(
            vector.len() == 3 * patch * count * patch,
            "image pixel count does not match its crops"
        );
    }
    let mut pixels = Vec::with_capacity(expected);
    for channel in 0..3 {
        for ky in 0..patch {
            for (vector, &count) in vectors.iter().zip(patches) {
                let row = count * patch;
                let start = (channel * patch + ky) * row;
                pixels.extend_from_slice(&vector[start..start + row]);
            }
        }
    }
    Ok(pixels)
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

// Tensor-resize parity bounds: resized u8 <= 2, FP32 <= 4/255 + 4*EPS, BF16 <= that + 1/256; else bit-exact.
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
    fn image_input_files_use_the_model_patch_width() -> Result<()> {
        let root: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/minicpm-config.json"))?;
        let spec = ModelSpec::minicpm(
            &root,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let width = spec
            .vision
            .as_ref()
            .unwrap()
            .patch_elements(spec.architecture)?;
        assert_eq!(width, 588);
        let prepared = PreparedImages {
            modality_ids: vec![0, 1, 1, 0],
            pixel_values: (0..16 * width).map(|i| i as f32 / 1024.0).collect(),
            grids: vec![ImageGrid { t: 1, h: 4, w: 4 }],
        };
        let ids = [1u32, 2, 3, 4];
        let path =
            std::env::temp_dir().join(format!("ff-image-roundtrip-minicpm-{}", std::process::id()));
        prepared.save(&spec, &ids, &path)?;
        let replay = PreparedImages::read(&spec, &path, &ids)?;
        assert_eq!(replay.grids, prepared.grids);
        assert_eq!(replay.modality_ids, prepared.modality_ids);
        assert_eq!(replay.pixel_values, prepared.pixel_values);
        let qwen: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/q38-config.json"))?;
        let qwen = ModelSpec::qwen(&qwen, &serde_json::Value::Null)?;
        let error = match PreparedImages::read(&qwen, &path, &ids) {
            Ok(_) => anyhow::bail!("Qwen-width read must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("588") && error.contains("1536"), "{error}");
        let mut mismatched = prepared;
        mismatched.pixel_values.pop();
        assert!(mismatched.save(&spec, &ids, &path).is_err());
        std::fs::remove_file(&path)?;
        Ok(())
    }

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
        let model = Path::new(&model);
        let processor = Processor::load(model, None)?.context("Qwen image processor")?;
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
                    packed.grids[0].t as i64,
                    packed.grids[0].h as i64,
                    packed.grids[0].w as i64
                ],
                expected_grid.as_slice(),
                "{name} grid"
            );
            let expected = tensors.tensor("pixel_values_fp32")?;
            let bf16 = tensors.tensor("pixel_values_bf16")?;
            let u8_values = tensors.tensor("pixel_values_uint8")?;
            assert_eq!(expected.shape(), [packed.grids[0].patch_rows()?, 1536]);
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
            let resized = image.width != packed.grids[0].w as usize * 16
                || image.height != packed.grids[0].h as usize * 16;
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

    #[test]
    fn video_sampling_matches_the_model_definition() -> Result<()> {
        let short = VideoMetadata {
            total_num_frames: 60,
            fps: 30.0,
            duration: 2.0,
        };
        assert_eq!(sample_frame_indices(&short, 8, 1)?, vec![0, 30]);
        assert_eq!(sample_frame_indices(&short, 128, 1)?, vec![0, 30]);
        let long = VideoMetadata {
            total_num_frames: 600,
            fps: 30.0,
            duration: 20.0,
        };
        assert_eq!(
            sample_frame_indices(&long, 8, 1)?,
            vec![0, 84, 168, 255, 339, 426, 510, 597]
        );
        let one = VideoMetadata {
            total_num_frames: 30,
            fps: 30.0,
            duration: 1.0,
        };
        assert_eq!(sample_frame_indices(&one, 8, 1)?, vec![0]);
        let fraction = VideoMetadata {
            total_num_frames: 12,
            fps: 30.0,
            duration: 0.4,
        };
        assert_eq!(sample_frame_indices(&fraction, 8, 1)?, vec![0]);
        let capped = VideoMetadata {
            total_num_frames: 3_900,
            fps: 30.0,
            duration: 130.0,
        };
        let indices = sample_frame_indices(&capped, 128, 1)?;
        assert_eq!(indices.len(), 128);
        assert_eq!(&indices[..3], &[0, 30, 60]);
        assert_eq!(&indices[125..], &[3_834, 3_864, 3_897]);
        let nine = VideoMetadata {
            total_num_frames: 216,
            fps: 24.0,
            duration: 9.0,
        };
        assert_eq!(
            sample_frame_indices(&nine, 8, 1)?,
            vec![0, 28, 60, 91, 120, 151, 182, 213]
        );
        let ten = VideoMetadata {
            total_num_frames: 240,
            fps: 24.0,
            duration: 10.0,
        };
        assert_eq!(
            sample_frame_indices(&ten, 8, 1)?,
            vec![0, 33, 67, 100, 134, 168, 201, 237]
        );
        let exactly = VideoMetadata {
            total_num_frames: 3_840,
            fps: 30.0,
            duration: 128.0,
        };
        let indices = sample_frame_indices(&exactly, 128, 1)?;
        assert_eq!(indices.len(), 128);
        assert_eq!(&indices[..2], &[0, 30]);
        assert_eq!(&indices[126..], &[3_780, 3_810]);
        Ok(())
    }

    #[test]
    fn video_sampling_validates_metadata_and_scope() {
        let meta = VideoMetadata {
            total_num_frames: 60,
            fps: 30.0,
            duration: 2.0,
        };
        let error = sample_frame_indices(&meta, 8, 3).unwrap_err().to_string();
        assert!(error.contains("stack_frames 3 is not claimed"), "{error}");
        assert!(sample_frame_indices(&meta, 0, 1).is_err());
        let zero = VideoMetadata {
            total_num_frames: 0,
            ..meta
        };
        assert!(sample_frame_indices(&zero, 8, 1).is_err());
        let bad_fps = VideoMetadata {
            fps: f64::NAN,
            ..meta
        };
        assert!(sample_frame_indices(&bad_fps, 8, 1).is_err());
        let bad_duration = VideoMetadata {
            duration: 0.0,
            ..meta
        };
        assert!(sample_frame_indices(&bad_duration, 8, 1).is_err());
        let inconsistent = VideoMetadata {
            total_num_frames: 10,
            ..meta
        };
        let error = sample_frame_indices(&inconsistent, 8, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside 10 frames"), "{error}");
    }

    fn video_processor() -> Result<Processor> {
        let root: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/minicpm-config.json"))?;
        let spec = ModelSpec::minicpm(
            &root,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        Ok(Processor {
            spec,
            config: ImageConfig::MiniCpm(MiniCpmConfig {
                max_slice_nums: 9,
                image_mean: [0.5, 0.5, 0.5],
                image_std: [0.5, 0.5, 0.5],
                mrope_mode: "canvas".into(),
            }),
            image_id: None,
        })
    }

    fn constant_frame(width: usize, height: usize, level: u8) -> RgbImage {
        RgbImage {
            width,
            height,
            pixels: vec![level; width * height * 3],
        }
    }

    #[test]
    fn video_frame_crops_pack_in_sample_order() -> Result<()> {
        let processor = video_processor()?;
        let patch = processor
            .spec
            .vision
            .as_ref()
            .unwrap()
            .patch_elements(processor.spec.architecture)?;
        let frames: Vec<RgbImage> = (0..4)
            .map(|i| constant_frame(64, 48, (i * 40) as u8))
            .collect();
        let meta = VideoMetadata {
            total_num_frames: 4,
            fps: 2.0,
            duration: 2.0,
        };
        let options = VideoOptions {
            max_num_frames: 2,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let packed = processor.prepare_video(&frames, &meta, &options)?;
        assert_eq!(packed.frames.len(), 2);
        let mut rows = 0usize;
        for frame in &packed.frames {
            assert_eq!((frame.rows, frame.cols), (0, 0));
            assert_eq!(frame.grids.len(), 1);
            rows += frame.grids[0].patch_rows()?;
        }
        assert_eq!(packed.pixel_values.len(), rows * patch);
        // Distinct frame content must produce distinct packed pixels in frame order.
        let per = rows / 2 * patch;
        assert_ne!(packed.pixel_values[..per], packed.pixel_values[per..]);
        let sliced = VideoOptions {
            max_num_frames: 1,
            stack_frames: 1,
            max_slice_nums: Some(9),
        };
        let large = vec![constant_frame(640, 480, 77), constant_frame(320, 240, 12)];
        let meta = VideoMetadata {
            total_num_frames: 2,
            fps: 1.0,
            duration: 2.0,
        };
        let packed = processor.prepare_video(&large, &meta, &sliced)?;
        assert_eq!(packed.frames.len(), 1);
        assert_eq!(packed.frames[0].grids.len(), 3);
        assert_eq!(packed.frames[0].rows * packed.frames[0].cols, 2);
        Ok(())
    }

    #[test]
    fn video_preparation_validates_frames_metadata_and_scope() -> Result<()> {
        let processor = video_processor()?;
        let frames = vec![constant_frame(64, 48, 3)];
        let options = VideoOptions {
            max_num_frames: 1,
            stack_frames: 1,
            max_slice_nums: None,
        };
        let meta = VideoMetadata {
            total_num_frames: 2,
            fps: 1.0,
            duration: 2.0,
        };
        let error = processor
            .prepare_video(&frames, &meta, &options)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("decoded video holds 1 frames, metadata reports 2"),
            "{error}"
        );
        let meta = VideoMetadata {
            total_num_frames: 1,
            fps: 1.0,
            duration: 1.0,
        };
        let stacked = VideoOptions {
            max_num_frames: 1,
            stack_frames: 3,
            max_slice_nums: None,
        };
        assert!(processor.prepare_video(&frames, &meta, &stacked).is_err());
        let zero_slices = VideoOptions {
            max_num_frames: 1,
            stack_frames: 1,
            max_slice_nums: Some(0),
        };
        assert!(
            processor
                .prepare_video(&frames, &meta, &zero_slices)
                .is_err()
        );
        Ok(())
    }
}
