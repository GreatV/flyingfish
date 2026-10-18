use crate::model_spec::{Architecture, ModelSpec};
use crate::serve::detok::Shared;
use crate::vision_input::{ImageGrid, MultimodalInput};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECODED_VIDEO_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_SOURCE_PIXELS: usize = 4 * 16_777_216;
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);
const DECODE_TIMEOUT: Duration = Duration::from_secs(600);
static VIDEO_SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

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

/// Decoded video container: RGB frames in decode order and their metadata.
pub struct DecodedVideo {
    pub frames: Vec<RgbImage>,
    pub metadata: VideoMetadata,
}

/// `numpy.linspace(0, len - 1, count, dtype=int)` positions; the endpoint is exactly `len - 1`.
fn linspace_positions(len: usize, count: usize) -> Vec<usize> {
    if count == 1 {
        return vec![0];
    }
    let delta = (len - 1) as f64 / (count - 1) as f64;
    let mut positions: Vec<usize> = (0..count).map(|k| (delta * k as f64) as usize).collect();
    positions[count - 1] = len - 1;
    positions
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
    let indices = if meta.duration > max_num_frames as f64 {
        let count = (meta.duration / 0.1) as usize;
        let mut timestamps = Vec::with_capacity(count);
        for i in 0..count {
            timestamps.push(i as f64 / 10.0);
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
    Ok(indices)
}

#[derive(Debug)]
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

/// Prepared video: one packed entry per sampled frame, in sample order.
#[derive(Debug)]
pub struct PackedVideo {
    pub frames: Vec<PackedImage>,
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

#[derive(Debug)]
pub struct PreparedInput {
    pub modality_ids: Vec<u8>,
    pub pixel_values: Vec<f32>,
    pub grids: Vec<ImageGrid>,
}

impl PreparedInput {
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
        ensure!(
            frames.len() == meta.total_num_frames,
            "decoded video holds {} frames, metadata reports {}",
            frames.len(),
            meta.total_num_frames
        );
        let mut size = None;
        for (index, frame) in frames.iter().enumerate() {
            let pixels = checked_pixels(frame.width, frame.height)?;
            ensure!(
                frame.pixels.len() == pixels * 3,
                "video frame {index} RGB byte count does not match its dimensions"
            );
            match size {
                None => size = Some((frame.width, frame.height)),
                Some(expected) => ensure!(
                    expected == (frame.width, frame.height),
                    "video frame {index} is {}x{}; frame 0 is {}x{}",
                    frame.width,
                    frame.height,
                    expected.0,
                    expected.1
                ),
            }
        }
        let indices = sample_frame_indices(meta, options.max_num_frames, options.stack_frames)?;
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
        };
        for &index in &indices {
            packed
                .frames
                .push(mini_cpm_prepare_rgb(&self.spec, &frames[index], &config)?);
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
        videos: Vec<PackedVideo>,
        capacity: usize,
    ) -> Result<(Vec<u32>, PreparedInput)> {
        let vision = self
            .spec
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        match &self.config {
            ImageConfig::Qwen(_) => expand_qwen(&self.spec, vision, ids, images, videos, capacity),
            ImageConfig::MiniCpm(_) => expand_mini_cpm(
                &self.spec,
                vision,
                self.image_id.as_ref().context("image-id tokens missing")?,
                ids,
                images,
                videos,
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

struct VisualUnit<'a> {
    tag: &'a [u32],
    token: u32,
    modality: u8,
    unit: &'a PackedImage,
}

/// Emits one visual unit's placeholder block: tag, thumbnail run, then slice rows.
fn push_visual_unit(
    vision: &crate::model_spec::VisionConfig,
    markers: &crate::model_spec::MiniCpmVision,
    expanded: &mut Vec<u32>,
    modality_ids: &mut Vec<u8>,
    visual: VisualUnit<'_>,
    capacity: usize,
) -> Result<usize> {
    let unit = visual.unit;
    let thumb = *unit.grids.first().context("visual unit has no crops")?;
    let thumb_tokens = thumb.feature_rows(vision.merge)?;
    let slices = if unit.rows > 0 && unit.cols > 0 {
        let rows = unit.rows as usize;
        let cols = unit.cols as usize;
        ensure!(
            unit.grids.len() == 1 + rows * cols,
            "crop count does not match the slice grid"
        );
        let slice = *unit
            .grids
            .get(1)
            .context("sliced unit lacks a slice crop")?;
        Some((rows, cols, slice.feature_rows(vision.merge)?))
    } else {
        ensure!(unit.grids.len() == 1, "unsliced unit carries slice crops");
        None
    };
    let mut len = visual
        .tag
        .len()
        .checked_add(2)
        .and_then(|n| n.checked_add(thumb_tokens))
        .context("visual token count overflow")?;
    if let Some((rows, cols, per_slice)) = slices {
        len = len
            .checked_add(rows - 1)
            .and_then(|n| n.checked_add(rows * cols * (2 + per_slice)))
            .context("visual token count overflow")?;
    }
    ensure!(
        expanded
            .len()
            .checked_add(len)
            .is_some_and(|n| n <= capacity),
        "visual tokens exceed request capacity"
    );
    expanded.extend_from_slice(visual.tag);
    modality_ids.extend(std::iter::repeat_n(0, visual.tag.len()));
    expanded.push(vision.start_token);
    modality_ids.push(0);
    expanded.extend(std::iter::repeat_n(visual.token, thumb_tokens));
    modality_ids.extend(std::iter::repeat_n(visual.modality, thumb_tokens));
    expanded.push(vision.end_token);
    modality_ids.push(0);
    if let Some((rows, cols, per_slice)) = slices {
        for row in 0..rows {
            if row > 0 {
                expanded.push(markers.newline_token);
                modality_ids.push(0);
            }
            for _ in 0..cols {
                expanded.push(markers.slice_start_token);
                modality_ids.push(0);
                expanded.extend(std::iter::repeat_n(visual.token, per_slice));
                modality_ids.extend(std::iter::repeat_n(visual.modality, per_slice));
                expanded.push(markers.slice_end_token);
                modality_ids.push(0);
            }
        }
    }
    Ok(len)
}

fn expand_qwen(
    spec: &ModelSpec,
    vision: &crate::model_spec::VisionConfig,
    ids: &[u32],
    images: Vec<PackedImage>,
    videos: Vec<PackedVideo>,
    capacity: usize,
) -> Result<(Vec<u32>, PreparedInput)> {
    ensure!(
        videos.is_empty(),
        "video input requires the MiniCPM image configuration"
    );
    let mut expanded = Vec::new();
    let mut prepared = PreparedInput {
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
    videos: Vec<PackedVideo>,
    capacity: usize,
) -> Result<(Vec<u32>, PreparedInput)> {
    let markers = vision
        .minicpm
        .as_ref()
        .context("MiniCPM vision configuration missing")?;
    let mut expanded = Vec::new();
    let mut prepared = PreparedInput {
        modality_ids: Vec::new(),
        pixel_values: Vec::new(),
        grids: Vec::new(),
    };
    let mut packed: Vec<Vec<f32>> = Vec::new();
    let mut packed_patches: Vec<usize> = Vec::new();
    let mut images = images.into_iter();
    let mut videos = videos.into_iter();
    let mut index = 0usize;
    for &id in ids {
        if id == vision.video_token {
            let video = videos
                .next()
                .context("more video placeholders than supplied videos")?;
            ensure!(!video.frames.is_empty(), "video has no frames");
            for frame in &video.frames {
                push_visual_unit(
                    vision,
                    markers,
                    &mut expanded,
                    &mut prepared.modality_ids,
                    VisualUnit {
                        tag: &[],
                        token: id,
                        modality: 2,
                        unit: frame,
                    },
                    capacity,
                )?;
            }
            for frame in &video.frames {
                packed_patches.push(frame.grids.iter().try_fold(0usize, |total, grid| {
                    total
                        .checked_add(grid.patch_rows()?)
                        .context("patch count overflow")
                })?);
            }
            prepared.grids.extend(
                video
                    .frames
                    .iter()
                    .flat_map(|frame| frame.grids.iter().copied()),
            );
            for frame in video.frames {
                packed.push(frame.pixel_values);
            }
        } else if id == vision.image_token {
            let image = images
                .next()
                .context("more image placeholders than supplied images")?;
            let tag = image_id.tag(index);
            index += 1;
            push_visual_unit(
                vision,
                markers,
                &mut expanded,
                &mut prepared.modality_ids,
                VisualUnit {
                    tag: &tag,
                    token: id,
                    modality: 1,
                    unit: &image,
                },
                capacity,
            )?;
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
    ensure!(
        videos.next().is_none(),
        "video has no placeholder in rendered prompt"
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

/// Decodes one MP4 data URL through the system ffprobe and ffmpeg.
pub fn load_video(url: &str) -> Result<DecodedVideo> {
    decode_video(&load_encoded_video(url)?)
}

fn load_encoded_video(url: &str) -> Result<Vec<u8>> {
    ensure!(
        url.starts_with("data:"),
        "video URL must be a base64 data URL"
    );
    let (header, data) = url.split_once(',').context("invalid video data URL")?;
    let mime = header
        .strip_prefix("data:")
        .and_then(|h| h.strip_suffix(";base64"))
        .context("video data URL must use base64")?;
    ensure!(mime == "video/mp4", "only MP4 video URLs are supported");
    ensure!(
        data.len() <= MAX_ENCODED_BYTES * 4 / 3 + 4,
        "encoded video exceeds byte limit"
    );
    base64::decode(data).context("video data URL is not valid base64")
}

/// Decodes one video container: ffprobe stream geometry, ffmpeg rgb24 frames over a pipe.
pub fn decode_video(bytes: &[u8]) -> Result<DecodedVideo> {
    ensure!(!bytes.is_empty(), "empty video container");
    ensure!(
        bytes.len() <= MAX_ENCODED_BYTES,
        "encoded video exceeds byte limit"
    );
    let source = ScratchVideo::write(bytes)?;
    let path = source
        .path()
        .to_str()
        .context("video scratch path is not UTF-8")?;
    let file = format!("file:{path}");
    let probe = tool_stdout(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            &file,
        ],
        PROBE_TIMEOUT,
        MAX_ENCODED_BYTES,
    )?;
    let probed = parse_probe(&probe)?;
    let frame_bytes = (probed.width as usize)
        .checked_mul(probed.height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("video frame byte count overflow")?;
    let expected = probed
        .frames
        .map(|count| {
            count
                .checked_mul(frame_bytes)
                .context("decoded video byte count overflow")
        })
        .transpose()?;
    if let Some(expected) = expected {
        ensure!(
            expected <= MAX_DECODED_VIDEO_BYTES,
            "decoded video is expected to be {expected} bytes over the {MAX_DECODED_VIDEO_BYTES}-byte limit"
        );
    }
    let raw = tool_stdout(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-noautorotate",
            "-i",
            &file,
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        DECODE_TIMEOUT,
        expected.unwrap_or(MAX_DECODED_VIDEO_BYTES),
    )?;
    ensure!(
        raw.len() % frame_bytes == 0,
        "ffmpeg produced {} bytes, not a multiple of the {frame_bytes}-byte {}x{} frame",
        raw.len(),
        probed.width,
        probed.height
    );
    let count = raw.len() / frame_bytes;
    ensure!(count > 0, "ffmpeg decoded no video frames");
    let frames = raw
        .chunks_exact(frame_bytes)
        .map(|pixels| RgbImage {
            width: probed.width as usize,
            height: probed.height as usize,
            pixels: pixels.to_vec(),
        })
        .collect();
    Ok(DecodedVideo {
        frames,
        metadata: VideoMetadata {
            total_num_frames: count,
            fps: probed.fps,
            duration: count as f64 / probed.fps,
        },
    })
}

enum ToolExit {
    Status(std::process::ExitStatus),
    TimedOut,
    WaitFailed(String),
}

/// Runs one tool, keeps stdout under `limit` bytes and kills it after `timeout`.
fn tool_stdout(tool: &str, args: &[&str], timeout: Duration, limit: usize) -> Result<Vec<u8>> {
    let mut child = std::process::Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("run {tool}"))?;
    let mut stdout = child.stdout.take().context("tool stdout pipe missing")?;
    let mut stderr = child.stderr.take().context("tool stderr pipe missing")?;
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let (exit_tx, exit_rx) = std::sync::mpsc::channel::<ToolExit>();
    let waiter = std::thread::Builder::new()
        .name(format!("{tool}-wait"))
        .spawn(move || {
            let deadline = std::time::Instant::now() + timeout;
            let exit = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break ToolExit::Status(status),
                    Ok(None) => {}
                    Err(error) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        break ToolExit::WaitFailed(error.to_string());
                    }
                }
                if stop_rx.try_recv().is_ok() {
                    let _ = child.kill();
                    break child.wait().map_or_else(
                        |error| ToolExit::WaitFailed(error.to_string()),
                        ToolExit::Status,
                    );
                }
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break ToolExit::TimedOut;
                }
                std::thread::sleep(Duration::from_millis(20));
            };
            let _ = exit_tx.send(exit);
        })
        .context("spawn tool waiter")?;
    let stderr_reader = std::thread::Builder::new()
        .name(format!("{tool}-stderr"))
        .spawn(move || {
            let mut buffer = Vec::new();
            let _ = stderr.read_to_end(&mut buffer);
            buffer
        })
        .context("spawn tool stderr reader")?;
    let mut output = Vec::new();
    let mut chunk = vec![0u8; 1 << 20];
    let mut failure = None;
    loop {
        match stdout.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => {
                if failure.is_none() && output.len().checked_add(read).is_none_or(|n| n > limit) {
                    failure = Some(anyhow::anyhow!(
                        "{tool} output exceeds the {limit}-byte limit"
                    ));
                    let _ = stop_tx.send(());
                }
                if failure.is_none() {
                    output.extend_from_slice(&chunk[..read]);
                }
            }
            Err(error) => {
                if failure.is_none() {
                    failure = Some(anyhow::anyhow!("read {tool} stdout: {error}"));
                }
                let _ = stop_tx.send(());
                break;
            }
        }
    }
    drop(stdout);
    let exit = exit_rx.recv().context("tool waiter stopped")?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("{tool} stderr reader panicked"))?;
    waiter
        .join()
        .map_err(|_| anyhow::anyhow!("{tool} waiter panicked"))?;
    if let Some(failure) = failure {
        return Err(failure);
    }
    match exit {
        ToolExit::TimedOut => bail!(
            "{tool} timed out after {} s and was killed",
            timeout.as_secs()
        ),
        ToolExit::WaitFailed(error) => bail!("wait for {tool}: {error}"),
        ToolExit::Status(status) => ensure!(
            status.success(),
            "{tool} failed ({}): {}",
            status,
            String::from_utf8_lossy(&stderr).trim()
        ),
    }
    Ok(output)
}

struct ScratchVideo {
    path: PathBuf,
}

impl ScratchVideo {
    fn write(bytes: &[u8]) -> Result<Self> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let directory = std::env::temp_dir();
        for _ in 0..8 {
            let counter = VIDEO_SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                "flyingfish-video-{}-{counter}.mp4",
                std::process::id()
            ));
            let mut file = match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("create video scratch file {}", path.display()));
                }
            };
            if let Err(error) = file.write_all(bytes) {
                let _ = std::fs::remove_file(&path);
                return Err(error)
                    .with_context(|| format!("write video scratch file {}", path.display()));
            }
            return Ok(Self { path });
        }
        bail!(
            "could not create a video scratch file in {}",
            directory.display()
        )
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchVideo {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[derive(Deserialize)]
struct ProbeReport {
    #[serde(default)]
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    nb_frames: Option<String>,
}

#[derive(Debug)]
struct ProbedVideo {
    width: u32,
    height: u32,
    fps: f64,
    frames: Option<usize>,
}

fn parse_probe(bytes: &[u8]) -> Result<ProbedVideo> {
    let report: ProbeReport = serde_json::from_slice(bytes).context("parse ffprobe JSON")?;
    let stream = report
        .streams
        .first()
        .context("ffprobe reported no video stream")?;
    let width = stream.width.context("ffprobe reported no video width")?;
    let height = stream.height.context("ffprobe reported no video height")?;
    ensure!(
        width > 0 && height > 0,
        "ffprobe reported a zero frame dimension"
    );
    let rate = stream
        .avg_frame_rate
        .as_deref()
        .context("ffprobe reported no average frame rate")?;
    let (numerator, denominator) = rate
        .split_once('/')
        .with_context(|| format!("ffprobe average frame rate {rate} is not a fraction"))?;
    let numerator: f64 = numerator
        .parse()
        .with_context(|| format!("ffprobe average frame rate {rate} numerator"))?;
    let denominator: f64 = denominator
        .parse()
        .with_context(|| format!("ffprobe average frame rate {rate} denominator"))?;
    let fps = numerator / denominator;
    ensure!(
        fps.is_finite() && fps > 0.0,
        "ffprobe average frame rate {rate} is not positive"
    );
    // Containers report an unknown frame count as "N/A".
    let frames = match stream.nb_frames.as_deref() {
        None | Some("N/A") => None,
        Some(text) => Some(
            text.parse()
                .with_context(|| format!("parse ffprobe frame count {text}"))?,
        ),
    };
    ensure!(
        frames.is_none_or(|count| count > 0),
        "ffprobe reported zero frames"
    );
    Ok(ProbedVideo {
        width,
        height,
        fps,
        frames,
    })
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
        let prepared = PreparedInput {
            modality_ids: vec![0, 1, 1, 0],
            pixel_values: (0..16 * width).map(|i| i as f32 / 1024.0).collect(),
            grids: vec![ImageGrid { t: 1, h: 4, w: 4 }],
        };
        let ids = [1u32, 2, 3, 4];
        let path =
            std::env::temp_dir().join(format!("ff-image-roundtrip-minicpm-{}", std::process::id()));
        prepared.save(&spec, &ids, &path)?;
        let replay = PreparedInput::read(&spec, &path, &ids)?;
        assert_eq!(replay.grids, prepared.grids);
        assert_eq!(replay.modality_ids, prepared.modality_ids);
        assert_eq!(replay.pixel_values, prepared.pixel_values);
        let qwen: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/q38-config.json"))?;
        let qwen = ModelSpec::qwen(&qwen, &serde_json::Value::Null)?;
        let error = match PreparedInput::read(&qwen, &path, &ids) {
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
            let (expanded, prepared) = processor.expand(&ids, vec![packed], Vec::new(), 4096)?;
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
        // The last linspace position is exactly len - 1: 407 s at 30 fps ends at frame 12207.
        let endpoint = VideoMetadata {
            total_num_frames: 12_210,
            fps: 30.0,
            duration: 407.0,
        };
        let indices = sample_frame_indices(&endpoint, 128, 1)?;
        assert_eq!(indices.len(), 128);
        assert_eq!(&indices[..4], &[0, 96, 192, 288]);
        assert_eq!(&indices[124..], &[11_916, 12_012, 12_108, 12_207]);
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
            image_id: Some(ImageId {
                start: 100,
                end: 101,
                digits: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            }),
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
        let total: usize = packed
            .frames
            .iter()
            .map(|frame| frame.pixel_values.len())
            .sum();
        assert_eq!(total, rows * patch);
        assert_ne!(packed.frames[0].pixel_values, packed.frames[1].pixel_values);
        let sliced = VideoOptions {
            max_num_frames: 1,
            stack_frames: 1,
            max_slice_nums: Some(9),
        };
        let large = vec![constant_frame(640, 480, 77), constant_frame(640, 480, 12)];
        let meta = VideoMetadata {
            total_num_frames: 2,
            fps: 1.0,
            duration: 2.0,
        };
        let packed = processor.prepare_video(&large, &meta, &sliced)?;
        assert_eq!(packed.frames.len(), 1);
        assert_eq!(packed.frames[0].grids.len(), 3);
        assert_eq!(packed.frames[0].rows * packed.frames[0].cols, 2);
        let single = VideoMetadata {
            total_num_frames: 1,
            fps: 1.0,
            duration: 1.0,
        };
        let only = processor.prepare_video(&[constant_frame(640, 480, 77)], &single, &sliced)?;
        assert_eq!(packed.frames[0].pixel_values, only.frames[0].pixel_values);
        assert_eq!(packed.frames[0].grids, only.frames[0].grids);
        Ok(())
    }

    #[test]
    fn video_pixels_interleave_like_the_same_frames_as_images() -> Result<()> {
        let processor = video_processor()?;
        let vision = processor.spec.vision.as_ref().unwrap();
        let frames = video_frames(2, 64, 48);
        let options = VideoOptions {
            max_num_frames: 2,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let video = processor.prepare_video(&frames, &video_meta(2), &options)?;
        let images = frames
            .iter()
            .map(|frame| processor.prepare_rgb(frame))
            .collect::<Result<Vec<_>>>()?;
        let (_, video_prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 4096)?;
        let (_, image_prepared) = processor.expand(
            &[vision.image_token, vision.image_token],
            images,
            Vec::new(),
            4096,
        )?;
        assert_eq!(video_prepared.pixel_values, image_prepared.pixel_values);
        assert_eq!(video_prepared.grids, image_prepared.grids);
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
        let mut bytes = video_frames(1, 64, 48);
        bytes[0].pixels.pop();
        let error = processor
            .prepare_video(&bytes, &meta, &options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("RGB byte count"), "{error}");
        let mut mixed = video_frames(2, 64, 48);
        mixed[1] = constant_frame(64, 32, 5);
        let error = processor
            .prepare_video(&mixed, &video_meta(2), &options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("video frame 1 is 64x32"), "{error}");
        Ok(())
    }

    fn video_frames(count: usize, width: usize, height: usize) -> Vec<RgbImage> {
        (0..count)
            .map(|i| constant_frame(width, height, (i * 37) as u8))
            .collect()
    }

    fn video_meta(count: usize) -> VideoMetadata {
        VideoMetadata {
            total_num_frames: count,
            fps: 1.0,
            duration: count as f64,
        }
    }

    #[test]
    fn ffprobe_metadata_parses_geometry_rate_and_frame_count() -> Result<()> {
        let report = serde_json::json!({
            "streams": [{"width": 64, "height": 48, "avg_frame_rate": "30000/1001",
                         "nb_frames": "3"}],
        });
        let probed = parse_probe(&serde_json::to_vec(&report)?)?;
        assert_eq!((probed.width, probed.height), (64, 48));
        assert!((probed.fps - 30000.0 / 1001.0).abs() < 1e-12);
        assert_eq!(probed.frames, Some(3));
        let mut unknown = report.clone();
        unknown["streams"][0]["nb_frames"] = serde_json::json!("N/A");
        let probed = parse_probe(&serde_json::to_vec(&unknown)?)?;
        assert_eq!(probed.frames, None);
        let mut absent = report.clone();
        assert!(
            absent["streams"][0]
                .as_object_mut()
                .unwrap()
                .remove("nb_frames")
                .is_some()
        );
        let probed = parse_probe(&serde_json::to_vec(&absent)?)?;
        assert_eq!(probed.frames, None);
        Ok(())
    }

    #[test]
    fn ffprobe_metadata_rejects_missing_or_inconsistent_fields() -> Result<()> {
        let valid = serde_json::json!({
            "streams": [{"width": 64, "height": 48, "avg_frame_rate": "10/1",
                         "nb_frames": "3"}],
        });
        let error = |value: serde_json::Value| -> Result<String> {
            match parse_probe(&serde_json::to_vec(&value)?) {
                Ok(_) => bail!("probe report must fail"),
                Err(error) => Ok(error.to_string()),
            }
        };
        let mut value = valid.clone();
        value["streams"] = serde_json::json!([]);
        assert!(error(value)?.contains("no video stream"));
        let mut value = valid.clone();
        assert!(
            value["streams"][0]
                .as_object_mut()
                .unwrap()
                .remove("width")
                .is_some()
        );
        assert!(error(value)?.contains("no video width"));
        let mut value = valid.clone();
        value["streams"][0]["height"] = serde_json::json!(0);
        assert!(error(value)?.contains("zero frame dimension"));
        let mut value = valid.clone();
        assert!(
            value["streams"][0]
                .as_object_mut()
                .unwrap()
                .remove("avg_frame_rate")
                .is_some()
        );
        assert!(error(value)?.contains("no average frame rate"));
        let mut value = valid.clone();
        value["streams"][0]["avg_frame_rate"] = serde_json::json!("0/0");
        assert!(error(value)?.contains("not positive"));
        let mut value = valid.clone();
        value["streams"][0]["avg_frame_rate"] = serde_json::json!("10");
        assert!(error(value)?.contains("not a fraction"));
        let mut value = valid.clone();
        value["streams"][0]["nb_frames"] = serde_json::json!("0");
        assert!(error(value)?.contains("zero frames"));
        let mut value = valid.clone();
        value["streams"][0]["nb_frames"] = serde_json::json!("3x");
        assert!(error(value)?.contains("frame count"));
        Ok(())
    }

    #[test]
    fn video_data_urls_require_mp4_base64() {
        for url in [
            "https://example.com/video.mp4",
            "file:///etc/passwd",
            "data:video/webm;base64,AAAA",
            "data:video/mp4,AAAA",
            "data:video/mp4;base64,!!!!",
        ] {
            assert!(load_video(url).is_err(), "{url}");
        }
    }

    #[test]
    #[ignore = "requires system ffmpeg"]
    fn video_decoding_uses_ffprobe_metadata_and_ffmpeg_frames() -> Result<()> {
        let path = std::env::temp_dir().join(format!("ff-video-clip-{}.mp4", std::process::id()));
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=10:duration=1.5",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&path)
            .status()
            .context("run ffmpeg to build the test clip")?;
        assert!(status.success());
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        let video = decode_video(&bytes)?;
        assert_eq!(video.frames.len(), 15);
        assert_eq!(video.metadata.total_num_frames, 15);
        assert_eq!(video.metadata.fps, 10.0);
        assert_eq!(video.metadata.duration, 1.5);
        assert_eq!((video.frames[0].width, video.frames[0].height), (64, 48));
        assert_eq!(video.frames[0].pixels.len(), 64 * 48 * 3);
        let processor = video_processor()?;
        let options = VideoOptions {
            max_num_frames: 4,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let packed = processor.prepare_video(&video.frames, &video.metadata, &options)?;
        assert_eq!(packed.frames.len(), 2);
        let vision = processor.spec.vision.as_ref().unwrap();
        let (expanded, _) =
            processor.expand(&[vision.video_token], Vec::new(), vec![packed], 4096)?;
        assert_eq!(
            expanded
                .iter()
                .filter(|&&id| id == vision.video_token)
                .count(),
            126
        );
        Ok(())
    }

    fn ffprobe_report(path: &Path, entries: &str) -> Result<serde_json::Value> {
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                entries,
                "-of",
                "json",
            ])
            .arg(format!("file:{}", path.display()))
            .output()
            .context("run ffprobe")?;
        assert!(probe.status.success());
        serde_json::from_slice(&probe.stdout).context("parse ffprobe JSON")
    }

    #[test]
    #[ignore = "requires system ffmpeg"]
    fn video_decoding_keeps_variable_frame_timing() -> Result<()> {
        let directory = std::env::temp_dir().join(format!("ff-video-vfr-{}", std::process::id()));
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("vfr.mp4");
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30:duration=1",
                "-vf",
                "select='eq(n,0)+eq(n,1)+eq(n,5)+eq(n,6)+eq(n,7)+eq(n,13)'",
                "-fps_mode",
                "vfr",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&path)
            .status()
            .context("run ffmpeg to build the VFR clip")?;
        assert!(status.success());
        let count = ffprobe_report(&path, "stream=nb_read_frames")?["streams"][0]["nb_read_frames"]
            .as_str()
            .context("nb_read_frames missing")?
            .parse::<usize>()?;
        assert_eq!(count, 6);
        let bytes = std::fs::read(&path)?;
        let reference = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-noautorotate", "-i"])
            .arg(format!("file:{}", path.display()))
            .args([
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .output()
            .context("run ffmpeg passthrough decode")?;
        std::fs::remove_dir_all(&directory)?;
        assert!(reference.status.success());
        let video = decode_video(&bytes)?;
        assert_eq!(video.frames.len(), count);
        assert_eq!(video.metadata.fps, 15.0);
        assert_eq!(video.metadata.duration, 0.4);
        let decoded: Vec<u8> = video
            .frames
            .iter()
            .flat_map(|frame| frame.pixels.iter().copied())
            .collect();
        assert_eq!(decoded, reference.stdout);
        Ok(())
    }

    #[test]
    #[ignore = "requires system ffmpeg"]
    fn video_metadata_uses_the_decoded_count_over_the_header_duration() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("ff-video-header-{}", std::process::id()));
        std::fs::create_dir_all(&directory)?;
        let clip = directory.join("video.mp4");
        let audio = directory.join("audio.mkv");
        let padded = directory.join("padded.mkv");
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30",
                "-frames:v",
                "300",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-y",
            ])
            .arg(&clip)
            .status()
            .context("run ffmpeg to build the 300-frame clip")?;
        assert!(status.success());
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=8000:cl=mono",
                "-t",
                "10.03",
                "-c:a",
                "pcm_s16le",
                "-y",
            ])
            .arg(&audio)
            .status()
            .context("run ffmpeg to build the 10.03 s audio track")?;
        assert!(status.success());
        let status = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&clip)
            .arg("-i")
            .arg(&audio)
            .args(["-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-y"])
            .arg(&padded)
            .status()
            .context("run ffmpeg to mux the padded container")?;
        assert!(status.success());
        let header = ffprobe_report(&padded, "format=duration")?["format"]["duration"]
            .as_str()
            .context("format duration missing")?
            .parse::<f64>()?;
        assert!((header - 10.03).abs() < 1e-9, "{header}");
        let bytes = std::fs::read(&padded)?;
        std::fs::remove_dir_all(&directory)?;
        let decoded = decode_video(&bytes)?;
        assert_eq!(decoded.metadata.total_num_frames, 300);
        assert_eq!(decoded.metadata.fps, 30.0);
        assert_eq!(decoded.metadata.duration, 10.0);
        let indices = sample_frame_indices(&decoded.metadata, 128, 1)?;
        assert_eq!(indices, (0..10).map(|i| i * 30).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    #[ignore = "requires system ffmpeg"]
    fn video_decoding_errors_name_the_failing_tool() -> Result<()> {
        let error = match decode_video(b"not a video container") {
            Ok(_) => bail!("invalid container must fail"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("ffprobe"), "{error}");
        Ok(())
    }

    #[test]
    fn video_placeholders_match_the_model_definition() -> Result<()> {
        let processor = video_processor()?;
        let vision = processor.spec.vision.as_ref().unwrap();
        let mut options = VideoOptions {
            max_num_frames: 2,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let frames = video_frames(2, 64, 48);
        let video = processor.prepare_video(&frames, &video_meta(2), &options)?;
        let (expanded, prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 4096)?;
        assert_eq!(expanded.len(), 130);
        assert_eq!(
            expanded
                .iter()
                .filter(|&&id| id == vision.video_token)
                .count(),
            126
        );
        let expected = [(0u8, 1usize), (2, 63), (0, 1), (0, 1), (2, 63), (0, 1)];
        let mut cursor = 0usize;
        for (modality, len) in expected {
            for &value in &prepared.modality_ids[cursor..cursor + len] {
                assert_eq!(value, modality);
            }
            cursor += len;
        }
        assert_eq!(cursor, 130);
        assert_eq!(prepared.grids.len(), 2);
        assert_eq!(prepared.pixel_values.len(), 2 * 28 * 36 * 588);
        let sliced = video_frames(2, 640, 480);
        options.max_slice_nums = Some(9);
        let video = processor.prepare_video(&sliced, &video_meta(2), &options)?;
        assert_eq!((video.frames[0].rows, video.frames[0].cols), (1, 2));
        let (expanded, prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 8192)?;
        assert_eq!(expanded.len(), 418);
        assert_eq!(
            expanded
                .iter()
                .filter(|&&id| id == vision.video_token)
                .count(),
            406
        );
        assert_eq!(prepared.grids.len(), 6);
        let square = video_frames(2, 1024, 1024);
        let video = processor.prepare_video(&square, &video_meta(2), &options)?;
        assert_eq!((video.frames[0].rows, video.frames[0].cols), (2, 3));
        let (expanded, _prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 16384)?;
        assert_eq!(expanded.len(), 878);
        assert_eq!(
            expanded
                .iter()
                .filter(|&&id| id == vision.video_token)
                .count(),
            848
        );
        Ok(())
    }

    #[test]
    fn image_and_video_placeholders_expand_in_prompt_order() -> Result<()> {
        let processor = video_processor()?;
        let vision = processor.spec.vision.as_ref().unwrap();
        let image = processor.prepare_rgb(&constant_frame(64, 48, 9))?;
        let options = VideoOptions {
            max_num_frames: 2,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let video = processor.prepare_video(&video_frames(2, 64, 48), &video_meta(2), &options)?;
        let (expanded, prepared) = processor.expand(
            &[vision.image_token, vision.video_token],
            vec![image],
            vec![video],
            4096,
        )?;
        assert_eq!(expanded.len(), 198);
        let video_first = expanded
            .iter()
            .position(|&id| id == vision.video_token)
            .unwrap();
        assert!(expanded[..video_first].contains(&vision.image_token));
        assert_eq!(
            prepared.modality_ids.iter().filter(|&&m| m == 1).count(),
            63
        );
        assert_eq!(
            prepared.modality_ids.iter().filter(|&&m| m == 2).count(),
            126
        );
        assert_eq!(prepared.grids.len(), 3);
        let (_, prepared) = processor.expand(
            &[vision.video_token, vision.image_token],
            vec![processor.prepare_rgb(&constant_frame(64, 48, 9))?],
            vec![processor.prepare_video(&video_frames(2, 64, 48), &video_meta(2), &options)?],
            4096,
        )?;
        assert_eq!(prepared.grids.len(), 3);
        assert_eq!(prepared.modality_ids[1], 2);
        Ok(())
    }

    #[test]
    fn video_expansion_validates_placeholder_counts() -> Result<()> {
        let processor = video_processor()?;
        let vision = processor.spec.vision.as_ref().unwrap();
        let options = VideoOptions {
            max_num_frames: 1,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let video = processor.prepare_video(&video_frames(1, 64, 48), &video_meta(1), &options)?;
        let error = processor
            .expand(
                &[vision.video_token, vision.video_token],
                Vec::new(),
                vec![video],
                4096,
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("more video placeholders than supplied videos"),
            "{error}"
        );
        let video = processor.prepare_video(&video_frames(1, 64, 48), &video_meta(1), &options)?;
        let error = processor
            .expand(&[5], Vec::new(), vec![video], 4096)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("video has no placeholder in rendered prompt"),
            "{error}"
        );
        let video = processor.prepare_video(&video_frames(1, 64, 48), &video_meta(1), &options)?;
        let (expanded, prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 4096)?;
        let mut modality = prepared.modality_ids.clone();
        let visual = modality
            .iter()
            .position(|&m| m == 2)
            .context("no video modality")?;
        modality[visual] = 3;
        let bad = PreparedInput {
            modality_ids: modality,
            pixel_values: prepared.pixel_values,
            grids: prepared.grids,
        };
        assert!(
            bad.input(&expanded)
                .validate(&processor.spec, 4096)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn video_canvas_positions_match_the_model_definition() -> Result<()> {
        let processor = video_processor()?;
        let vision = processor.spec.vision.as_ref().unwrap();
        let mut options = VideoOptions {
            max_num_frames: 2,
            stack_frames: 1,
            max_slice_nums: Some(1),
        };
        let video = processor.prepare_video(&video_frames(2, 64, 48), &video_meta(2), &options)?;
        let (expanded, prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 4096)?;
        let plan = prepared.input(&expanded).validate(&processor.spec, 4096)?;
        let axes = plan.axes();
        assert_eq!([axes[0][1], axes[1][1], axes[2][1]], [0, 0, 0]);
        assert_eq!([axes[0][66], axes[1][66], axes[2][66]], [10, 10, 10]);
        assert_eq!(plan.rope_delta(), -110);
        options.max_slice_nums = Some(9);
        let video =
            processor.prepare_video(&video_frames(2, 640, 480), &video_meta(2), &options)?;
        let (expanded, prepared) =
            processor.expand(&[vision.video_token], Vec::new(), vec![video], 8192)?;
        assert_eq!(expanded.len(), 418);
        let plan = prepared.input(&expanded).validate(&processor.spec, 8192)?;
        let axes = plan.axes();
        assert_eq!([axes[0][1], axes[1][1], axes[2][1]], [0, 0, 0]);
        assert_eq!([axes[0][66], axes[1][66], axes[2][66]], [0, 0, 0]);
        assert_eq!([axes[0][138], axes[1][138], axes[2][138]], [0, 0, 7]);
        assert_eq!([axes[0][210], axes[1][210], axes[2][210]], [15, 15, 15]);
        options.max_slice_nums = Some(1);
        let image = processor.prepare_rgb(&constant_frame(64, 48, 5))?;
        let video = processor.prepare_video(&video_frames(2, 64, 48), &video_meta(2), &options)?;
        let (expanded, prepared) = processor.expand(
            &[vision.image_token, vision.video_token],
            vec![image],
            vec![video],
            4096,
        )?;
        let plan = prepared.input(&expanded).validate(&processor.spec, 4096)?;
        let axes = plan.axes();
        assert_eq!([axes[0][69], axes[1][69], axes[2][69]], [13, 13, 13]);
        assert_eq!([axes[0][134], axes[1][134], axes[2][134]], [23, 23, 23]);
        Ok(())
    }
}
