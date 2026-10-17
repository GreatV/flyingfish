//! CPU contracts for processor-packed static images and text mRoPE positions.
use crate::model_spec::{Architecture, ModelSpec, VisionConfig};
use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageGrid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl ImageGrid {
    pub fn patch_rows(self) -> Result<usize> {
        ensure!(self.t == 1, "only static images (t=1) are supported");
        ensure!(
            self.h > 0 && self.w > 0 && self.h.is_multiple_of(2) && self.w.is_multiple_of(2),
            "image grid must have positive, merge-aligned height and width"
        );
        let rows = u64::from(self.h) * u64::from(self.w);
        ensure!(
            rows <= i32::MAX as u64 / 16,
            "image grid exceeds vision kernel index limits"
        );
        usize::try_from(rows).context("patch count exceeds host index range")
    }

    pub fn feature_rows(self) -> Result<usize> {
        Ok(self.patch_rows()? / 4)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSlot {
    pub tokens: Range<usize>,
    pub patches: Range<usize>,
    pub features: Range<usize>,
    pub grid: ImageGrid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpliceRange {
    pub chunk_rows: Range<usize>,
    pub feature_rows: Range<usize>,
}

pub struct MultimodalInput<'a> {
    pub input_ids: &'a [u32],
    /// 0 = text, 1 = image. Video and other modalities are rejected.
    pub modality_ids: &'a [u8],
    /// Merge-grouped rows, each [channel, temporal, patch_y, patch_x] flattened.
    pub pixel_values: &'a [f32],
    pub grids: &'a [ImageGrid],
}

#[derive(Clone, Debug)]
pub struct PositionPlan {
    axes: [Vec<i32>; 3],
    slots: Vec<ImageSlot>,
    capacity: usize,
    rope_delta: i64,
    patch_rows: usize,
    feature_rows: usize,
}

impl PositionPlan {
    pub fn text(rows: usize, capacity: usize) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= capacity && capacity <= i32::MAX as usize,
            "invalid text position extent"
        );
        let positions: Vec<_> = (0..rows).map(|p| p as i32).collect();
        Ok(Self {
            axes: [positions.clone(), positions.clone(), positions],
            slots: Vec::new(),
            capacity,
            rope_delta: 0,
            patch_rows: 0,
            feature_rows: 0,
        })
    }

    pub fn axes(&self) -> [&[i32]; 3] {
        self.axes.each_ref().map(Vec::as_slice)
    }
    pub fn slots(&self) -> &[ImageSlot] {
        &self.slots
    }
    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }
    pub fn patch_rows(&self) -> usize {
        self.patch_rows
    }
    pub fn feature_rows(&self) -> usize {
        self.feature_rows
    }
    pub fn prompt_rows(&self) -> usize {
        self.axes[0].len()
    }

    /// Dense (T,H,W) rows for the prompt and every possible continuation slot.
    pub fn packed_axes(&self) -> Result<Vec<i32>> {
        let mut table = Vec::with_capacity(
            self.capacity
                .checked_mul(3)
                .context("axis table overflow")?,
        );
        for physical in 0..self.capacity {
            if physical < self.prompt_rows() {
                table.extend(self.axes.iter().map(|a| a[physical]));
            } else {
                table.extend([self.decode_position(physical)?; 3]);
            }
        }
        Ok(table)
    }

    pub fn chunk(&self, rows: Range<usize>) -> Result<[&[i32]; 3]> {
        ensure!(
            rows.start < rows.end && rows.end <= self.prompt_rows(),
            "position chunk outside prompt"
        );
        Ok(self.axes.each_ref().map(|a| &a[rows.clone()]))
    }

    pub fn splices(&self, rows: Range<usize>) -> Result<Vec<SpliceRange>> {
        self.chunk(rows.clone())?;
        Ok(self
            .slots
            .iter()
            .filter_map(|slot| {
                let start = rows.start.max(slot.tokens.start);
                let end = rows.end.min(slot.tokens.end);
                (start < end).then(|| {
                    let feature_start = slot.features.start + start - slot.tokens.start;
                    SpliceRange {
                        chunk_rows: start - rows.start..end - rows.start,
                        feature_rows: feature_start..feature_start + end - start,
                    }
                })
            })
            .collect())
    }

    pub fn decode_position(&self, physical_position: usize) -> Result<i32> {
        ensure!(
            physical_position >= self.prompt_rows() && physical_position < self.capacity,
            "decode position outside continuation capacity"
        );
        let position = i64::try_from(physical_position)?
            .checked_add(self.rope_delta)
            .context("mRoPE position overflow")?;
        ensure!(position >= 0, "negative mRoPE coordinate");
        i32::try_from(position).context("mRoPE coordinate exceeds device range")
    }

    pub fn rotary_position(&self, frequency: usize, physical_position: usize) -> Result<i32> {
        let axis = Self::rotary_axis(frequency)?;
        if physical_position < self.prompt_rows() {
            Ok(self.axes[axis][physical_position])
        } else {
            self.decode_position(physical_position)
        }
    }

    pub fn rotary_axis(frequency: usize) -> Result<usize> {
        ensure!(frequency < 32, "frequency outside Q38 partial RoPE");
        Ok(if frequency % 3 == 1 {
            1
        } else if frequency % 3 == 2 && frequency < 30 {
            2
        } else {
            0
        })
    }
}

impl MultimodalInput<'_> {
    pub fn validate(&self, spec: &ModelSpec, capacity: usize) -> Result<PositionPlan> {
        ensure!(
            spec.architecture == Architecture::Qwen35Text,
            "packed images require Q38"
        );
        let vision = spec
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        vision.validate(spec.architecture)?;
        ensure!(
            capacity > 0 && capacity <= spec.max_position as usize,
            "invalid multimodal capacity"
        );
        let len = self.input_ids.len();
        ensure!(
            len > 0 && len <= capacity && self.modality_ids.len() == len,
            "token/modality extent mismatch"
        );
        ensure!(
            self.input_ids
                .iter()
                .all(|&id| id < spec.vocab && id != vision.video_token),
            "invalid token or unsupported video token"
        );
        ensure!(
            self.modality_ids.iter().all(|&m| m <= 1),
            "unsupported modality (only text/image)"
        );
        let patch_rows = self.grids.iter().try_fold(0usize, |n, g| {
            n.checked_add(g.patch_rows()?)
                .context("packed patch count overflow")
        })?;
        ensure!(
            patch_rows / 4 <= len,
            "more image features than prompt tokens"
        );
        let elements = patch_rows
            .checked_mul(vision.patch_elements(spec.architecture)?)
            .context("packed pixel size overflow")?;
        ensure!(
            self.pixel_values.len() == elements,
            "packed pixel element count mismatch: expected {elements}, got {}",
            self.pixel_values.len()
        );
        ensure!(
            self.pixel_values
                .iter()
                .all(|&v| v.is_finite() && half::bf16::from_f32(v).is_finite()),
            "nonfinite packed pixel or BF16 overflow"
        );

        let mut plan = PositionPlan {
            axes: std::array::from_fn(|_| Vec::with_capacity(len)),
            slots: Vec::new(),
            capacity,
            rope_delta: 0,
            patch_rows,
            feature_rows: patch_rows / 4,
        };
        let (mut cursor, mut current, mut patch, mut feature) = (0usize, 0i64, 0usize, 0usize);
        while cursor < len {
            let id = self.input_ids[cursor];
            if self.modality_ids[cursor] == 0 {
                ensure!(
                    id != vision.image_token,
                    "image placeholder is marked as text"
                );
                if id == vision.start_token {
                    ensure!(
                        self.modality_ids.get(cursor + 1) == Some(&1),
                        "vision start must precede image slots"
                    );
                }
                if id == vision.end_token {
                    ensure!(
                        cursor > 0 && self.modality_ids[cursor - 1] == 1,
                        "vision end without image slots"
                    );
                }
                for axis in &mut plan.axes {
                    axis.push(i32::try_from(current)?);
                }
                current += 1;
                cursor += 1;
                continue;
            }
            ensure!(
                id == vision.image_token
                    && cursor > 0
                    && self.input_ids[cursor - 1] == vision.start_token,
                "image run lacks its vision start"
            );
            let start = cursor;
            while cursor < len && self.modality_ids[cursor] == 1 {
                ensure!(
                    self.input_ids[cursor] == vision.image_token,
                    "image modality contains non-placeholder token"
                );
                cursor += 1;
            }
            ensure!(
                self.input_ids.get(cursor) == Some(&vision.end_token),
                "image run lacks its vision end"
            );
            let grid = *self
                .grids
                .get(plan.slots.len())
                .context("image run has no corresponding grid")?;
            let patches = grid.patch_rows()?;
            let features = grid.feature_rows()?;
            ensure!(
                cursor - start == features,
                "image placeholder count does not match grid"
            );
            for row in 0..grid.h / 2 {
                for col in 0..grid.w / 2 {
                    for (axis, offset) in [0i64, i64::from(row), i64::from(col)]
                        .into_iter()
                        .enumerate()
                    {
                        plan.axes[axis].push(i32::try_from(current + offset)?);
                    }
                }
            }
            plan.slots.push(ImageSlot {
                tokens: start..cursor,
                patches: patch..patch + patches,
                features: feature..feature + features,
                grid,
            });
            patch += patches;
            feature += features;
            current += i64::from(grid.h.max(grid.w) / 2);
        }
        ensure!(
            plan.slots.len() == self.grids.len() && patch == patch_rows,
            "unused image grids/pixels"
        );
        ensure!(
            plan.axes.iter().all(|axis| axis.len() == len),
            "position count differs from prompt"
        );
        let max = plan
            .axes
            .iter()
            .flatten()
            .max()
            .context("empty positions")?;
        plan.rope_delta = i64::from(*max) + 1 - i64::try_from(len)?;
        Ok(plan)
    }
}

#[derive(Clone, Debug)]
pub struct VisionPositions {
    pub coordinates: Vec<[i32; 2]>,
    /// Corner-major, matching the model helper's [4, patches] tensors.
    pub bilinear_indices: [Vec<i32>; 4],
    pub bilinear_weights: [Vec<f32>; 4],
}

impl VisionPositions {
    pub fn new(
        spec: &VisionConfig,
        architecture: Architecture,
        grids: &[ImageGrid],
        row_limit: usize,
    ) -> Result<Self> {
        ensure!(
            architecture == Architecture::Qwen35Text,
            "vision positions require the Qwen tower"
        );
        spec.validate(architecture)?;
        let total = grids.iter().try_fold(0usize, |n, g| {
            n.checked_add(g.patch_rows()?)
                .context("vision position size overflow")
        })?;
        ensure!(
            total > 0 && total <= row_limit,
            "vision positions exceed patch row budget"
        );
        let mut result = Self {
            coordinates: Vec::with_capacity(total),
            bilinear_indices: std::array::from_fn(|_| Vec::with_capacity(total)),
            bilinear_weights: std::array::from_fn(|_| Vec::with_capacity(total)),
        };
        let side = spec.position_side as i32;
        let coordinate = |i: u32, n: u32| {
            let end = (side - 1) as f32;
            let step = end / (n - 1) as f32;
            if i < n / 2 {
                step * i as f32
            } else {
                (-step).mul_add((n - 1 - i) as f32, end)
            }
        };
        for grid in grids {
            for block_y in 0..grid.h / 2 {
                for block_x in 0..grid.w / 2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let y = block_y * 2 + dy;
                            let x = block_x * 2 + dx;
                            result
                                .coordinates
                                .push([i32::try_from(y)?, i32::try_from(x)?]);
                            let h = coordinate(y, grid.h);
                            let w = coordinate(x, grid.w);
                            let hf = h as i32;
                            let wf = w as i32;
                            let hc = (hf + 1).min(side - 1);
                            let wc = (wf + 1).min(side - 1);
                            let dh = h - hf as f32;
                            let dw = w - wf as f32;
                            let indices = [
                                hf * side + wf,
                                hf * side + wc,
                                hc * side + wf,
                                hc * side + wc,
                            ];
                            let weights = [
                                (1.0 - dh) * (1.0 - dw),
                                (1.0 - dh) * dw,
                                dh * (1.0 - dw),
                                dh * dw,
                            ];
                            for k in 0..4 {
                                result.bilinear_indices[k].push(indices[k]);
                                result.bilinear_weights[k].push(weights[k]);
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}
