//! CPU contracts for processor-packed static images and text mRoPE positions.
use crate::model_spec::{Architecture, ModelSpec, VisionConfig};
use anyhow::{Context, Result, bail, ensure};
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

    pub fn feature_rows(self, merge: u32) -> Result<usize> {
        let patches = self.patch_rows()?;
        ensure!(
            self.h.is_multiple_of(merge) && self.w.is_multiple_of(merge),
            "image grid is not merge aligned"
        );
        Ok(patches / (merge as usize * merge as usize))
    }
}

/// Patch rows reordered into square `window` blocks: window row-major, then row-major inside the window.
pub struct WindowGather {
    /// Source patch row of each gathered row, images concatenated in grid order.
    pub indices: Vec<u32>,
    /// One window-end entry per window, preceded by zero; every window is `window * window` rows.
    pub cu_seqlens: Vec<i32>,
}

/// Nearest-bucket ids into a `side × side` learned table: `bucketize(right=True)` of `i / extent`
/// against `(j + 1) / side`, flattened as `bucket_h * side + bucket_w`, image after image.
pub fn nearest_position_ids(grids: &[ImageGrid], side: u32) -> Result<Vec<i32>> {
    ensure!(side > 0, "zero position table side");
    ensure!(
        side <= 46340,
        "position table side exceeds the i32 id range"
    );
    let bucket = |value: f32| -> i32 {
        let mut count = 0i32;
        while count < side as i32 {
            let boundary = ((count + 1) as f64 / f64::from(side)) as f32;
            if boundary > value {
                break;
            }
            count += 1;
        }
        count
    };
    let mut ids = Vec::new();
    for grid in grids {
        ids.reserve(grid.patch_rows()?);
        for row in 0..grid.h {
            let base = bucket(row as f32 / grid.h as f32) * side as i32;
            for column in 0..grid.w {
                ids.push(base + bucket(column as f32 / grid.w as f32));
            }
        }
    }
    Ok(ids)
}

/// Gathers every patch row into its `window × window` block, images concatenated in grid order.
pub fn window_gather(grids: &[ImageGrid], window: u32) -> Result<WindowGather> {
    ensure!(window > 0, "zero window size");
    let mut indices = Vec::new();
    let mut cu_seqlens = vec![0i32];
    let mut base = 0u32;
    for grid in grids {
        let patches = grid.patch_rows()?;
        ensure!(
            grid.h.is_multiple_of(window) && grid.w.is_multiple_of(window),
            "patch grid {}x{} is not window aligned",
            grid.h,
            grid.w
        );
        for window_row in 0..grid.h / window {
            for window_column in 0..grid.w / window {
                for row in 0..window {
                    for column in 0..window {
                        let patch = (window_row * window + row)
                            .checked_mul(grid.w)
                            .and_then(|column_index| {
                                column_index.checked_add(window_column * window + column)
                            })
                            .context("window patch index overflow")?;
                        indices.push(
                            base.checked_add(patch)
                                .context("window gather row overflow")?,
                        );
                    }
                }
                cu_seqlens.push(i32::try_from(indices.len())?);
            }
        }
        base = base
            .checked_add(u32::try_from(patches)?)
            .context("window gather row overflow")?;
    }
    Ok(WindowGather {
        indices,
        cu_seqlens,
    })
}

/// `merge × merge` row grouping over current row grids: `groups[merge²·r + j]` is the source row of
/// output row `r`, block `j = dh · merge + dw`, images concatenated in grid order.
pub fn merge_groups(grids: &[(u32, u32)], merge: u32) -> Result<Vec<u32>> {
    ensure!(merge > 0, "zero merge size");
    let merge = merge as usize;
    let mut groups = Vec::new();
    let mut base = 0u32;
    for &(rows, columns) in grids {
        ensure!(
            rows > 0
                && columns > 0
                && (rows as usize).is_multiple_of(merge)
                && (columns as usize).is_multiple_of(merge),
            "row grid is not merge aligned"
        );
        let (rows, columns) = (rows as usize, columns as usize);
        for group_row in 0..rows / merge {
            for group_column in 0..columns / merge {
                for dh in 0..merge {
                    for dw in 0..merge {
                        let row = (group_row * merge + dh)
                            .checked_mul(columns)
                            .and_then(|row| row.checked_add(group_column * merge + dw))
                            .context("merge group row overflow")?;
                        let row = u32::try_from(row).context("merge group row exceeds u32")?;
                        groups.push(base.checked_add(row).context("merge group row overflow")?);
                    }
                }
            }
        }
        base = base
            .checked_add(u32::try_from(
                rows.checked_mul(columns)
                    .context("merge group extent overflow")?,
            )?)
            .context("merge group row overflow")?;
    }
    Ok(groups)
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
    /// 0 = text, 1 = image, 2 = video.
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
        match spec.architecture {
            Architecture::Qwen35Text => self.validate_qwen(spec, capacity),
            Architecture::MiniCpmV47 => self.validate_minicpm(spec, capacity),
            Architecture::Llama => bail!("packed images require a multimodal architecture"),
        }
    }

    fn validate_qwen(&self, spec: &ModelSpec, capacity: usize) -> Result<PositionPlan> {
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
            let features = grid.feature_rows(vision.merge)?;
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

    fn validate_minicpm(&self, spec: &ModelSpec, capacity: usize) -> Result<PositionPlan> {
        let vision = spec
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        vision.validate(spec.architecture)?;
        let markers = vision
            .minicpm
            .as_ref()
            .context("MiniCPM vision configuration missing")?;
        let merge = vision.merge;
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
            self.input_ids.iter().all(|&id| id < spec.vocab),
            "invalid token id"
        );
        ensure!(
            self.modality_ids.iter().all(|&m| m <= 2),
            "unsupported modality (only text/image/video)"
        );
        let mut patch_rows = 0usize;
        let mut feature_rows = 0usize;
        for grid in self.grids {
            patch_rows = patch_rows
                .checked_add(grid.patch_rows()?)
                .context("packed patch count overflow")?;
            feature_rows = feature_rows
                .checked_add(grid.feature_rows(merge)?)
                .context("packed feature count overflow")?;
        }
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

        // One slot per image-token run; runs and crops are both in sequence order.
        let mut slots = Vec::new();
        let mut cursor = 0usize;
        let mut patch = 0usize;
        let mut feature = 0usize;
        let mut opened = false;
        while cursor < len {
            if self.modality_ids[cursor] == 0 {
                cursor += 1;
                continue;
            }
            let start = cursor;
            let run = self.modality_ids[start];
            let token = if run == 1 {
                vision.image_token
            } else {
                ensure!(run == 2, "unsupported modality {run}");
                vision.video_token
            };
            while cursor < len && self.modality_ids[cursor] == run {
                ensure!(
                    self.input_ids[cursor] == token,
                    "visual modality contains non-placeholder token"
                );
                cursor += 1;
            }
            let grid = *self
                .grids
                .get(slots.len())
                .context("visual run has no corresponding grid")?;
            let patches = grid.patch_rows()?;
            let features = grid.feature_rows(merge)?;
            ensure!(
                cursor - start == features,
                "visual placeholder count does not match grid"
            );
            let before = start
                .checked_sub(1)
                .and_then(|i| self.input_ids.get(i))
                .copied()
                .context("image run lacks its opening marker")?;
            let close = if before == vision.start_token {
                opened = true;
                vision.end_token
            } else {
                ensure!(
                    before == markers.slice_start_token && opened,
                    "image run lacks its opening marker"
                );
                markers.slice_end_token
            };
            ensure!(
                self.input_ids.get(cursor) == Some(&close),
                "image run lacks its closing marker"
            );
            slots.push(ImageSlot {
                tokens: start..cursor,
                patches: patch..patch + patches,
                features: feature..feature + features,
                grid,
            });
            patch += patches;
            feature += features;
        }
        ensure!(
            slots.len() == self.grids.len() && patch == patch_rows && feature == feature_rows,
            "unused image grids/pixels"
        );

        let (axes, rope_delta) = mini_cpm_canvas(
            self.input_ids,
            self.modality_ids,
            self.grids,
            vision,
            markers,
        )?;
        Ok(PositionPlan {
            axes,
            slots,
            capacity,
            rope_delta,
            patch_rows,
            feature_rows,
        })
    }
}

struct VisualFrame {
    thumb: std::ops::Range<usize>,
    slices: Vec<std::ops::Range<usize>>,
    end: usize,
}

struct VisualGroup {
    start: usize,
    end: usize,
    frames: Vec<VisualFrame>,
}

fn mini_cpm_canvas(
    ids: &[u32],
    modality: &[u8],
    grids: &[ImageGrid],
    vision: &VisionConfig,
    markers: &crate::model_spec::MiniCpmVision,
) -> Result<([Vec<i32>; 3], i64)> {
    let marker_ids = [
        markers.slice_start_token,
        markers.slice_end_token,
        vision.start_token,
        vision.end_token,
        markers.newline_token,
    ];
    let is_marker = |id: u32| marker_ids.contains(&id);
    let merge = vision.merge;

    let mut groups: Vec<VisualGroup> = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    let mut previous_end = 0usize;
    let mut cursor = 0usize;
    while cursor < ids.len() {
        let current = modality[cursor];
        let start = cursor;
        while cursor < ids.len() && modality[cursor] == current {
            cursor += 1;
        }
        if current == 0 {
            continue;
        }
        if let Some((g, f)) = open
            && start > 0
            && ids[start - 1] == markers.slice_start_token
        {
            groups[g].frames[f].slices.push(start..cursor);
            previous_end = cursor;
            continue;
        }
        let frame = VisualFrame {
            thumb: start..cursor,
            slices: Vec::new(),
            end: 0,
        };
        let adjacent =
            !groups.is_empty() && ids[previous_end..start - 1].iter().all(|&id| is_marker(id));
        if adjacent {
            groups
                .last_mut()
                .context("visual group missing")?
                .frames
                .push(frame);
        } else {
            groups.push(VisualGroup {
                start: start - 1,
                end: 0,
                frames: vec![frame],
            });
        }
        let g = groups.len() - 1;
        let f = groups[g].frames.len() - 1;
        open = Some((g, f));
        previous_end = cursor;
    }
    for group in &mut groups {
        for frame in &mut group.frames {
            let mut end = frame.slices.last().map_or(frame.thumb.end, |s| s.end);
            while end < ids.len() && is_marker(ids[end]) {
                end += 1;
            }
            frame.end = end;
        }
        group.end = group
            .frames
            .last()
            .context("visual group without frames")?
            .end;
    }

    let seq = ids.len();
    let mut pos = [vec![0i64; seq], vec![0i64; seq], vec![0i64; seq]];
    let fill_run = |pos: &mut [Vec<i64>; 3], range: std::ops::Range<usize>, base: i64| {
        for (offset, index) in range.enumerate() {
            let value = base + offset as i64;
            pos[0][index] = value;
            pos[1][index] = value;
            pos[2][index] = value;
        }
    };
    let mut grid_index = 0usize;
    let mut current_pos = 0i64;
    let mut current_idx = 0usize;
    for group in &groups {
        if group.start > current_idx {
            fill_run(&mut pos, current_idx..group.start, current_pos);
            current_pos += (group.start - current_idx) as i64;
        }
        let mut frame_cursor = group.start;
        for frame in &group.frames {
            let thumb = *grids.get(grid_index).context("missing thumbnail grid")?;
            grid_index += 1;
            let frame_start = frame.thumb.start - 1;
            if frame_start > frame_cursor {
                fill_run(&mut pos, frame_cursor..frame_start, current_pos);
                current_pos += (frame_start - frame_cursor) as i64 + 1;
            }
            let frame_end = frame.end.min(group.end);
            let canvas_origin = current_pos;
            let halo = (canvas_origin - 1).max(0);
            let (mut canvas_h, mut canvas_w) = (thumb.h / merge, thumb.w / merge);
            let (num_rows, num_cols, slice_h, slice_w) = if frame.slices.is_empty() {
                (0usize, 0usize, 0u32, 0u32)
            } else {
                let mut num_cols = frame.slices.len();
                for k in 0..frame.slices.len() - 1 {
                    if frame.slices[k + 1].start - frame.slices[k].end > 2 {
                        num_cols = k + 1;
                        break;
                    }
                }
                let mut num_rows = frame.slices.len() / num_cols;
                if num_rows * num_cols != frame.slices.len() {
                    num_rows = 1;
                    num_cols = frame.slices.len();
                }
                let first = *grids.get(grid_index).context("missing slice grid")?;
                (num_rows, num_cols, first.h / merge, first.w / merge)
            };
            if num_rows > 0 {
                canvas_h = num_rows as u32 * slice_h;
                canvas_w = num_cols as u32 * slice_w;
            }
            for axis in &mut pos {
                for value in &mut axis[frame_start..frame_end] {
                    *value = canvas_origin;
                }
            }
            pos[1][frame_start] = halo;
            pos[2][frame_start] = halo;
            if frame.thumb.end < frame_end {
                pos[1][frame.thumb.end] = canvas_origin + i64::from(canvas_h);
                pos[2][frame.thumb.end] = canvas_origin + i64::from(canvas_w);
            }
            vision_canvas_positions(
                &mut pos,
                CanvasPlacement {
                    origin: canvas_origin,
                    grid: thumb,
                    canvas: (canvas_h, canvas_w),
                    offset: (0, 0),
                    merge,
                    range: frame.thumb.clone(),
                },
            )?;
            for (k, slice) in frame.slices.iter().enumerate() {
                let h_off = (k / num_cols) as u32 * slice_h;
                let w_off = (k % num_cols) as u32 * slice_w;
                let grid = *grids.get(grid_index).context("missing slice grid")?;
                grid_index += 1;
                let slice_end_h = grid.h / merge;
                let slice_end_w = grid.w / merge;
                let marker = slice.start - 1;
                if marker >= frame_start {
                    pos[1][marker] = canvas_origin + i64::from(h_off);
                    pos[2][marker] = canvas_origin + i64::from(w_off);
                }
                if slice.end < frame_end {
                    pos[1][slice.end] = canvas_origin + i64::from(h_off + slice_end_h) - 1;
                    pos[2][slice.end] = canvas_origin + i64::from(w_off + slice_end_w) - 1;
                }
                vision_canvas_positions(
                    &mut pos,
                    CanvasPlacement {
                        origin: canvas_origin,
                        grid,
                        canvas: (0, 0),
                        offset: (h_off, w_off),
                        merge,
                        range: slice.clone(),
                    },
                )?;
            }
            for k in 0..frame.slices.len().saturating_sub(1) {
                let gap_start = frame.slices[k].end;
                let gap_end = frame.slices[k + 1].start;
                if gap_end - gap_start <= 2 {
                    continue;
                }
                let boundary_h = i64::from((k / num_cols + 1) as u32 * slice_h) - 1;
                let right_edge_w = i64::from(num_cols as u32 * slice_w);
                let [_, rows, columns] = &mut pos;
                for (h, w) in rows[gap_start + 1..gap_end - 1]
                    .iter_mut()
                    .zip(&mut columns[gap_start + 1..gap_end - 1])
                {
                    *h = canvas_origin + boundary_h;
                    *w = canvas_origin + right_edge_w;
                }
            }
            current_pos = canvas_origin + i64::from(canvas_h.max(canvas_w)) + 1;
            frame_cursor = frame_end;
        }
        if frame_cursor < group.end {
            fill_run(&mut pos, frame_cursor..group.end, current_pos);
            current_pos += (group.end - frame_cursor) as i64;
        }
        current_idx = group.end;
    }
    if current_idx < seq {
        fill_run(&mut pos, current_idx..seq, current_pos);
    }
    ensure!(
        grid_index == grids.len(),
        "unused image grids: {grid_index} of {}",
        grids.len()
    );
    let max = pos
        .iter()
        .flatten()
        .max()
        .copied()
        .context("empty positions")?;
    let rope_delta = max + 1 - seq as i64;
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    for (axis, values) in pos.into_iter().enumerate() {
        axes[axis] = values
            .into_iter()
            .map(|value| i32::try_from(value).context("mRoPE coordinate exceeds device range"))
            .collect::<Result<Vec<_>>>()?;
    }
    Ok((axes, rope_delta))
}

struct CanvasPlacement {
    origin: i64,
    grid: ImageGrid,
    canvas: (u32, u32),
    offset: (u32, u32),
    merge: u32,
    range: std::ops::Range<usize>,
}

fn vision_canvas_positions(pos: &mut [Vec<i64>; 3], placement: CanvasPlacement) -> Result<()> {
    let llm_h = placement.grid.h / placement.merge;
    let llm_w = placement.grid.w / placement.merge;
    ensure!(
        placement.range.len() == llm_h as usize * llm_w as usize,
        "crop token count does not match its grid"
    );
    let canvas_h = if placement.canvas.0 == 0 {
        llm_h
    } else {
        placement.canvas.0
    };
    let canvas_w = if placement.canvas.1 == 0 {
        llm_w
    } else {
        placement.canvas.1
    };
    let hs = linspace_round(canvas_h - 1, llm_h as usize);
    let ws = linspace_round(canvas_w - 1, llm_w as usize);
    let mut index = placement.range.start;
    for &h in &hs {
        for &w in &ws {
            pos[0][index] = placement.origin;
            pos[1][index] = placement.origin + h + i64::from(placement.offset.0);
            pos[2][index] = placement.origin + w + i64::from(placement.offset.1);
            index += 1;
        }
    }
    Ok(())
}

fn linspace_round(end: u32, len: usize) -> Vec<i64> {
    if len <= 1 {
        return vec![0];
    }
    let step = end as f32 / (len - 1) as f32;
    (0..len)
        .map(|i| (i as f32 * step).round_ties_even() as i64)
        .collect()
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
