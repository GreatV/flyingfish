//! MiniMax-H3 frozen-condition layouts and denoising loop.
//!
//! FL2VA and Ref2VA do not concatenate a mask to the target latent. They put
//! encoded reference rows into the same full-attention sequence, assign those
//! rows a fixed conditioning timestep, and update only the generated suffix.

use crate::{
    layout::{self, AUDIO_TAG, TEXT_TAG, VIDEO_TAG},
    model::{
        PreparedDenoiseSchedule, PreparedTransformerContext, StreamedTransformer,
        TransformerStaticInputs, TransformerStepInputs,
    },
    pipeline::{
        DenoiseCheckpointEvent, DenoiseObserver, DenoisePreparationEvent, DenoiseStepEvent,
        T2vaExecutionOptions, T2vaLatents, T2vaSchedule, resolve_run_steps,
    },
    scheduler::H3Scheduler,
};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use std::time::Instant;

pub const CONDITION_VIDEO_TIMESTEP: f32 = 0.999;
pub const CONDITION_AUDIO_TIMESTEP: f32 = 1.0;

const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const ROPE_SPATIAL_SCALE: f64 = 32.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyframeAnchor {
    First,
    Last,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceBlock {
    Image {
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
    },
    Audio {
        audio_latents: usize,
    },
    Video {
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        audio_latents: usize,
    },
}

pub struct ConditionedLayout {
    position_ids: Tensor,
    token_tags: Tensor,
    video_indices: Tensor,
    audio_indices: Tensor,
    text_indices: Tensor,
    video_indices_host: Vec<u32>,
    audio_indices_host: Vec<u32>,
    sequence_length: usize,
    text_rows: usize,
    video_rows: usize,
    audio_rows: usize,
    condition_video_rows: usize,
    condition_audio_rows: usize,
    target_video_rows: usize,
    target_audio_rows: usize,
}

impl ConditionedLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn fl2va(
        text_token_tags: &[u32],
        target_latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        target_audio_latents: usize,
        patch_size: [usize; 3],
        audio_channels: usize,
        anchors: &[KeyframeAnchor],
        device: &Device,
    ) -> Result<Self> {
        validate_common(
            text_token_tags,
            target_latent_frames,
            latent_height,
            latent_width,
            patch_size,
            audio_channels,
        )?;
        ensure!(
            target_audio_latents > 0,
            "FL2VA target audio latent count must be non-zero"
        );
        ensure!(
            (1..=2).contains(&anchors.len()),
            "FL2VA requires one or two keyframe anchors"
        );
        if anchors.len() == 2 {
            ensure!(
                anchors == [KeyframeAnchor::First, KeyframeAnchor::Last],
                "two FL2VA keyframes must be ordered first, then last"
            );
        }
        let [_, patch_h, patch_w] = patch_size;
        let rows_per_frame = checked_rows_per_frame(latent_height, latent_width, patch_h, patch_w)?;
        let text_rows = text_token_tags.len();
        let condition_video_rows = anchors
            .len()
            .checked_mul(rows_per_frame)
            .context("FL2VA condition row count overflow")?;
        let target_audio_rows = target_audio_latents
            .checked_mul(audio_channels)
            .context("FL2VA audio row count overflow")?;
        let target_video_rows = target_latent_frames
            .checked_mul(rows_per_frame)
            .context("FL2VA target video row count overflow")?;
        let condition_start = text_rows;
        let audio_start = condition_start + condition_video_rows;
        let video_start = audio_start + target_audio_rows;
        let sequence_length = video_start + target_video_rows;

        let (frame_grid, width_grid) = frame_grid(latent_height, latent_width, patch_h, patch_w);
        let mut positions = vec![0f64; sequence_length * 3];
        fill_text_positions(&mut positions, text_rows);
        let target_span = pairwise_sum(
            &(0..target_latent_frames)
                .map(frame_duration)
                .collect::<Vec<_>>(),
        );
        for (condition, anchor) in anchors.iter().enumerate() {
            let time = match anchor {
                KeyframeAnchor::First => text_rows as f64,
                KeyframeAnchor::Last => text_rows as f64 + target_span - ROPE_FRAME_RESCALE,
            };
            fill_video_block(
                &mut positions,
                condition_start + condition * rows_per_frame,
                1,
                &frame_grid,
                &[time],
            );
        }
        fill_audio_block(
            &mut positions,
            audio_start,
            target_audio_latents,
            text_rows as f64,
            &width_grid,
            audio_channels,
        )?;
        let target_times = temporal_grid(target_latent_frames, text_rows as f64);
        fill_video_block(
            &mut positions,
            video_start,
            target_latent_frames,
            &frame_grid,
            &target_times,
        );

        let text_indices = range_u32(0, text_rows)?;
        let mut video_indices_host = range_u32(condition_start, audio_start)?;
        video_indices_host.extend(range_u32(video_start, sequence_length)?);
        let audio_indices_host = range_u32(audio_start, video_start)?;
        let mut tags = vec![TEXT_TAG; sequence_length];
        tags[..text_rows].copy_from_slice(text_token_tags);
        assign_tags(&mut tags, &video_indices_host, VIDEO_TAG)?;
        assign_tags(&mut tags, &audio_indices_host, AUDIO_TAG)?;
        Self::from_parts(
            positions,
            tags,
            text_indices,
            video_indices_host,
            audio_indices_host,
            text_rows,
            condition_video_rows,
            0,
            target_video_rows,
            target_audio_rows,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ref2va(
        text_token_tags: &[u32],
        references: &[ReferenceBlock],
        target_latent_frames: usize,
        target_latent_height: usize,
        target_latent_width: usize,
        target_audio_latents: usize,
        patch_size: [usize; 3],
        audio_channels: usize,
        device: &Device,
    ) -> Result<Self> {
        validate_common(
            text_token_tags,
            target_latent_frames,
            target_latent_height,
            target_latent_width,
            patch_size,
            audio_channels,
        )?;
        ensure!(
            target_audio_latents > 0,
            "Ref2VA target audio latent count must be non-zero"
        );
        ensure!(
            !references.is_empty(),
            "Ref2VA requires at least one reference"
        );
        ensure!(
            references.len() <= 12,
            "Ref2VA accepts at most 12 references"
        );
        ensure!(
            references
                .iter()
                .any(|reference| !matches!(reference, ReferenceBlock::Audio { .. })),
            "Ref2VA cannot condition on audio alone"
        );
        let [_, patch_h, patch_w] = patch_size;
        let text_rows = text_token_tags.len();
        let target_rows_per_frame =
            checked_rows_per_frame(target_latent_height, target_latent_width, patch_h, patch_w)?;
        let target_video_rows = target_latent_frames
            .checked_mul(target_rows_per_frame)
            .context("Ref2VA target video row count overflow")?;
        let target_audio_rows = target_audio_latents
            .checked_mul(audio_channels)
            .context("Ref2VA target audio row count overflow")?;
        let mut condition_video_rows = 0usize;
        let mut condition_audio_rows = 0usize;
        for reference in references {
            if let ReferenceBlock::Audio { audio_latents } = *reference {
                ensure!(
                    audio_latents > 0,
                    "Ref2VA audio references must contain at least one latent"
                );
            }
            if let ReferenceBlock::Image { latent_frames, .. } = *reference {
                ensure!(
                    latent_frames == 1,
                    "Ref2VA image references must encode to exactly one latent frame"
                );
            }
            match *reference {
                ReferenceBlock::Image {
                    latent_frames,
                    latent_height,
                    latent_width,
                }
                | ReferenceBlock::Video {
                    latent_frames,
                    latent_height,
                    latent_width,
                    ..
                } => {
                    ensure!(
                        latent_frames > 0,
                        "reference latent frames must be non-zero"
                    );
                    let rows = latent_frames
                        .checked_mul(checked_rows_per_frame(
                            latent_height,
                            latent_width,
                            patch_h,
                            patch_w,
                        )?)
                        .context("reference video row count overflow")?;
                    condition_video_rows = condition_video_rows
                        .checked_add(rows)
                        .context("reference video row count overflow")?;
                }
                ReferenceBlock::Audio { .. } => {}
            }
            let audio_latents = match *reference {
                ReferenceBlock::Audio { audio_latents }
                | ReferenceBlock::Video { audio_latents, .. } => audio_latents,
                ReferenceBlock::Image { .. } => 0,
            };
            condition_audio_rows = condition_audio_rows
                .checked_add(
                    audio_latents
                        .checked_mul(audio_channels)
                        .context("reference audio row count overflow")?,
                )
                .context("reference audio row count overflow")?;
        }
        let sequence_length = text_rows
            .checked_add(condition_video_rows)
            .and_then(|value| value.checked_add(condition_audio_rows))
            .and_then(|value| value.checked_add(target_audio_rows))
            .and_then(|value| value.checked_add(target_video_rows))
            .context("Ref2VA sequence length overflow")?;
        let mut positions = vec![0f64; sequence_length * 3];
        fill_text_positions(&mut positions, text_rows);
        let (_, target_width_grid) =
            frame_grid(target_latent_height, target_latent_width, patch_h, patch_w);
        let mut cursor = text_rows;
        let mut rotary_time = text_rows as f64;
        let mut video_indices_host = Vec::with_capacity(condition_video_rows + target_video_rows);
        let mut audio_indices_host = Vec::with_capacity(condition_audio_rows + target_audio_rows);
        for reference in references {
            match *reference {
                ReferenceBlock::Image {
                    latent_frames,
                    latent_height,
                    latent_width,
                } => {
                    let (grid, _) = frame_grid(latent_height, latent_width, patch_h, patch_w);
                    let rows = latent_frames * grid.len();
                    video_indices_host.extend(range_u32(cursor, cursor + rows)?);
                    let times = vec![rotary_time; latent_frames];
                    fill_video_block(&mut positions, cursor, latent_frames, &grid, &times);
                    cursor += rows;
                    rotary_time += 1.0;
                }
                ReferenceBlock::Audio { audio_latents } => {
                    let rows = audio_latents * audio_channels;
                    audio_indices_host.extend(range_u32(cursor, cursor + rows)?);
                    fill_audio_block(
                        &mut positions,
                        cursor,
                        audio_latents,
                        rotary_time,
                        &target_width_grid,
                        audio_channels,
                    )?;
                    cursor += rows;
                    rotary_time += audio_latents as f64;
                }
                ReferenceBlock::Video {
                    latent_frames,
                    latent_height,
                    latent_width,
                    audio_latents,
                } => {
                    let origin = rotary_time;
                    let (grid, width_grid) =
                        frame_grid(latent_height, latent_width, patch_h, patch_w);
                    let audio_rows = audio_latents * audio_channels;
                    audio_indices_host.extend(range_u32(cursor, cursor + audio_rows)?);
                    fill_audio_block(
                        &mut positions,
                        cursor,
                        audio_latents,
                        origin,
                        &width_grid,
                        audio_channels,
                    )?;
                    cursor += audio_rows;
                    let video_rows = latent_frames * grid.len();
                    video_indices_host.extend(range_u32(cursor, cursor + video_rows)?);
                    let times = temporal_grid(latent_frames, origin);
                    fill_video_block(&mut positions, cursor, latent_frames, &grid, &times);
                    cursor += video_rows;
                    let video_span = (0..latent_frames).map(frame_duration).sum::<f64>();
                    rotary_time += (audio_latents as f64).max(video_span);
                }
            }
        }
        ensure!(
            cursor == text_rows + condition_video_rows + condition_audio_rows,
            "Ref2VA reference block accounting is inconsistent"
        );
        let target_audio_start = cursor;
        let target_video_start = target_audio_start + target_audio_rows;
        audio_indices_host.extend(range_u32(target_audio_start, target_video_start)?);
        fill_audio_block(
            &mut positions,
            target_audio_start,
            target_audio_latents,
            rotary_time,
            &target_width_grid,
            audio_channels,
        )?;
        video_indices_host.extend(range_u32(target_video_start, sequence_length)?);
        let (target_grid, _) =
            frame_grid(target_latent_height, target_latent_width, patch_h, patch_w);
        let target_times = temporal_grid(target_latent_frames, rotary_time);
        fill_video_block(
            &mut positions,
            target_video_start,
            target_latent_frames,
            &target_grid,
            &target_times,
        );
        let text_indices = range_u32(0, text_rows)?;
        let mut tags = vec![TEXT_TAG; sequence_length];
        tags[..text_rows].copy_from_slice(text_token_tags);
        assign_tags(&mut tags, &video_indices_host, VIDEO_TAG)?;
        assign_tags(&mut tags, &audio_indices_host, AUDIO_TAG)?;
        Self::from_parts(
            positions,
            tags,
            text_indices,
            video_indices_host,
            audio_indices_host,
            text_rows,
            condition_video_rows,
            condition_audio_rows,
            target_video_rows,
            target_audio_rows,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        positions: Vec<f64>,
        tags: Vec<u32>,
        text_indices: Vec<u32>,
        video_indices_host: Vec<u32>,
        audio_indices_host: Vec<u32>,
        text_rows: usize,
        condition_video_rows: usize,
        condition_audio_rows: usize,
        target_video_rows: usize,
        target_audio_rows: usize,
        device: &Device,
    ) -> Result<Self> {
        let sequence_length = tags.len();
        ensure!(
            sequence_length
                .checked_mul(3)
                .is_some_and(|expected| positions.len() == expected),
            "position count mismatch"
        );
        let video_rows = condition_video_rows
            .checked_add(target_video_rows)
            .context("conditioned video row count overflow")?;
        let audio_rows = condition_audio_rows
            .checked_add(target_audio_rows)
            .context("conditioned audio row count overflow")?;
        ensure!(
            video_indices_host.len() == video_rows,
            "video index count mismatch"
        );
        ensure!(
            audio_indices_host.len() == audio_rows,
            "audio index count mismatch"
        );
        ensure!(
            text_rows
                .checked_add(video_rows)
                .and_then(|rows| rows.checked_add(audio_rows))
                == Some(sequence_length),
            "conditioned modalities do not cover the packed sequence"
        );
        Ok(Self {
            position_ids: Tensor::from_vec(positions, (sequence_length, 3), device)?,
            token_tags: Tensor::from_vec(tags, sequence_length, device)?,
            video_indices: Tensor::from_vec(video_indices_host.clone(), video_rows, device)?,
            audio_indices: Tensor::from_vec(audio_indices_host.clone(), audio_rows, device)?,
            text_indices: Tensor::from_vec(text_indices, text_rows, device)?,
            video_indices_host,
            audio_indices_host,
            sequence_length,
            text_rows,
            video_rows,
            audio_rows,
            condition_video_rows,
            condition_audio_rows,
            target_video_rows,
            target_audio_rows,
        })
    }

    pub fn position_ids(&self) -> &Tensor {
        &self.position_ids
    }

    pub fn token_tags(&self) -> &Tensor {
        &self.token_tags
    }

    pub fn video_indices(&self) -> &Tensor {
        &self.video_indices
    }

    pub fn audio_indices(&self) -> &Tensor {
        &self.audio_indices
    }

    pub fn text_indices(&self) -> &Tensor {
        &self.text_indices
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn condition_video_rows(&self) -> usize {
        self.condition_video_rows
    }

    pub fn video_rows(&self) -> usize {
        self.video_rows
    }

    pub fn audio_rows(&self) -> usize {
        self.audio_rows
    }

    pub fn condition_audio_rows(&self) -> usize {
        self.condition_audio_rows
    }

    pub fn target_video_rows(&self) -> usize {
        self.target_video_rows
    }

    pub fn target_audio_rows(&self) -> usize {
        self.target_audio_rows
    }

    pub fn row_timesteps(
        &self,
        video_timestep: f32,
        audio_timestep: f32,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let condition_video_timestep = video_timestep.max(CONDITION_VIDEO_TIMESTEP);
        let distinct = distinct_timesteps([
            (self.sequence_length - self.audio_rows, video_timestep),
            (self.target_audio_rows, audio_timestep),
            (self.condition_video_rows, condition_video_timestep),
            (self.condition_audio_rows, CONDITION_AUDIO_TIMESTEP),
        ]);
        let video_index = timestep_index(&distinct, video_timestep)?;
        let mut inverse = vec![video_index; self.sequence_length];
        let condition_video_index = timestep_index(&distinct, condition_video_timestep)?;
        for &row in &self.video_indices_host[..self.condition_video_rows] {
            inverse[row as usize] = condition_video_index;
        }
        if self.condition_audio_rows > 0 {
            let condition_audio_index = timestep_index(&distinct, CONDITION_AUDIO_TIMESTEP)?;
            for &row in &self.audio_indices_host[..self.condition_audio_rows] {
                inverse[row as usize] = condition_audio_index;
            }
        }
        if self.target_audio_rows > 0 {
            let audio_index = timestep_index(&distinct, audio_timestep)?;
            for &row in &self.audio_indices_host[self.condition_audio_rows..] {
                inverse[row as usize] = audio_index;
            }
        }
        Ok((
            Tensor::from_vec(distinct.clone(), distinct.len(), device)?,
            Tensor::from_vec(inverse, self.sequence_length, device)?,
        ))
    }

    fn timestep_table(&self, video_timestep: f32, audio_timestep: f32) -> Vec<f32> {
        distinct_timesteps([
            (self.sequence_length - self.audio_rows, video_timestep),
            (self.target_audio_rows, audio_timestep),
            (
                self.condition_video_rows,
                video_timestep.max(CONDITION_VIDEO_TIMESTEP),
            ),
            (self.condition_audio_rows, CONDITION_AUDIO_TIMESTEP),
        ])
    }
}

enum PreparedConditionedExecution {
    Dynamic(PreparedTransformerContext),
    Precomputed(PreparedTransformerContext, PreparedDenoiseSchedule),
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_conditioned_with_observer(
    transformer: &StreamedTransformer,
    prompt_embeddings: &Tensor,
    text_token_tags: &[u32],
    condition_video_rows: &Tensor,
    condition_audio_rows: &Tensor,
    initial_target_video_latents: &Tensor,
    initial_target_audio_latents: &Tensor,
    layout: &ConditionedLayout,
    schedule: T2vaSchedule,
    options: T2vaExecutionOptions,
    observer: &mut dyn DenoiseObserver,
) -> Result<T2vaLatents> {
    let device = transformer.device();
    let config = transformer.config();
    let (video_batch, video_channels, video_frames, video_height, video_width) =
        initial_target_video_latents
            .dims5()
            .context("conditioned target video latents must be rank five")?;
    ensure!(
        video_batch == 1
            && video_channels == config.in_channels
            && video_frames > 0
            && video_height > 0
            && video_width > 0,
        "conditioned target video latents must be [1, {}, positive frames, positive height, positive width]",
        config.in_channels
    );
    let video_dims = initial_target_video_latents.dims().to_vec();
    let (audio_channels, audio_width, audio_frames) = initial_target_audio_latents
        .dims3()
        .context("conditioned target audio latents must be [channels, latent_channels, frames]")?;
    ensure!(
        audio_channels == 2 && audio_width == config.audio_in_channels && audio_frames > 0,
        "conditioned target audio latents must be [2, {}, positive frames]",
        config.audio_in_channels
    );
    let (prompt_batch, prompt_rows, prompt_width) = prompt_embeddings.dims3()?;
    ensure!(
        prompt_batch == 1 && prompt_width == config.text_dim,
        "conditioned prompt embeddings must be [1, rows, {}]",
        config.text_dim
    );
    ensure!(
        prompt_rows == text_token_tags.len() && prompt_rows == layout.text_rows,
        "conditioned prompt rows disagree with the packed layout"
    );
    ensure!(
        matches!(
            prompt_embeddings.dtype(),
            candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16
        ),
        "conditioned prompt embeddings must use a floating-point dtype"
    );
    for (name, tensor) in [
        ("condition video rows", condition_video_rows),
        ("condition audio rows", condition_audio_rows),
        ("target video latents", initial_target_video_latents),
        ("target audio latents", initial_target_audio_latents),
    ] {
        ensure!(
            tensor.device().same_device(device),
            "{name} are on the wrong device"
        );
        ensure!(
            tensor.dtype() == candle_core::DType::F32,
            "{name} must be F32 at the conditioned pipeline boundary"
        );
    }
    ensure!(
        prompt_embeddings.device().same_device(device),
        "prompt embeddings are on the wrong device"
    );
    for (name, tensor) in [
        ("layout positions", layout.position_ids()),
        ("layout tags", layout.token_tags()),
        ("layout video indices", layout.video_indices()),
        ("layout audio indices", layout.audio_indices()),
        ("layout text indices", layout.text_indices()),
    ] {
        ensure!(
            tensor.device().same_device(device),
            "{name} are on the wrong device"
        );
    }
    let patch_size = config.patch_size;
    let patch_volume = patch_size.iter().try_fold(1usize, |product, part| {
        product
            .checked_mul(*part)
            .context("conditioned video patch volume overflow")
    })?;
    let expected_video_width = config
        .in_channels
        .checked_mul(patch_volume)
        .context("conditioned video row width overflow")?;
    ensure!(
        condition_video_rows.dims2()? == (layout.condition_video_rows, expected_video_width),
        "condition video row shape disagrees with layout and transformer"
    );
    ensure!(
        condition_audio_rows.dims2()? == (layout.condition_audio_rows, config.audio_in_channels),
        "condition audio row shape disagrees with layout and transformer"
    );
    let mut video_scheduler = H3Scheduler::new(schedule.video_shift)?;
    let mut audio_scheduler = H3Scheduler::new(schedule.audio_shift)?;
    let video_timesteps = video_scheduler
        .set_timesteps(schedule.sigma_points)?
        .to_vec();
    let audio_timesteps = audio_scheduler
        .set_timesteps(schedule.sigma_points)?
        .to_vec();
    ensure!(
        video_timesteps.len() == audio_timesteps.len(),
        "schedule lengths differ"
    );
    ensure!(
        options.start_step < video_timesteps.len(),
        "conditioned start_step is outside the schedule"
    );
    if options.start_step > 0 {
        video_scheduler.seek(options.start_step)?;
        audio_scheduler.seek(options.start_step)?;
    }
    let remaining = video_timesteps.len() - options.start_step;
    let run_steps = resolve_run_steps(remaining, options.max_steps, "conditioned H3")?;
    let end_step = options.start_step + run_steps;
    let mut target_video_rows = layout::patchify_video(initial_target_video_latents, patch_size)?;
    let mut target_audio_rows = layout::pack_audio(initial_target_audio_latents)?;
    ensure!(
        target_video_rows.dim(0)? == layout.target_video_rows
            && target_audio_rows.dim(0)? == layout.target_audio_rows,
        "target row counts disagree with conditioned layout"
    );
    let timestep_tables = video_timesteps[options.start_step..end_step]
        .iter()
        .zip(&audio_timesteps[options.start_step..end_step])
        .map(|(&video, &audio)| layout.timestep_table(video, audio))
        .collect::<Vec<_>>();

    if observer.synchronize_device_timings() {
        device.synchronize()?;
    }
    let preparation_start = Instant::now();
    let context = transformer.prepare_context(&TransformerStaticInputs {
        encoder_hidden_states: prompt_embeddings,
        token_tags: layout.token_tags(),
        position_ids: layout.position_ids(),
        video_indices: layout.video_indices(),
        audio_indices: layout.audio_indices(),
        text_indices: layout.text_indices(),
    })?;
    let prepared = if options.precompute_adaln {
        PreparedConditionedExecution::Precomputed(
            context,
            transformer.prepare_denoise_schedule(&timestep_tables)?,
        )
    } else {
        PreparedConditionedExecution::Dynamic(context)
    };
    if observer.synchronize_device_timings() {
        device.synchronize()?;
    }
    observer.on_preparation_completed(DenoisePreparationEvent {
        prepared_evaluations: run_steps,
        elapsed: preparation_start.elapsed(),
        timing_synchronized: observer.synchronize_device_timings(),
    })?;

    let total_start = Instant::now();
    let checkpoint_each_evaluation = observer.checkpoint_each_evaluation();
    let mut final_checkpoint = None;
    for prepared_step in 0..run_steps {
        let step = options.start_step + prepared_step;
        let video_timestep = video_timesteps[step];
        let audio_timestep = audio_timesteps[step];
        if observer.synchronize_device_timings() {
            device.synchronize()?;
        }
        let step_start = Instant::now();
        let video_rows = Tensor::cat(&[condition_video_rows, &target_video_rows], 0)?;
        let audio_rows = Tensor::cat(&[condition_audio_rows, &target_audio_rows], 0)?;
        let (timesteps, timestep_indices) =
            layout.row_timesteps(video_timestep, audio_timestep, device)?;
        let inputs = TransformerStepInputs {
            video_hidden_states: &video_rows.unsqueeze(0)?,
            audio_hidden_states: &audio_rows.unsqueeze(0)?,
            timestep: &timesteps,
            timestep_indices: &timestep_indices,
        };
        let output = match &prepared {
            PreparedConditionedExecution::Dynamic(context) => {
                transformer.forward_step(context, &inputs)?
            }
            PreparedConditionedExecution::Precomputed(context, prepared_schedule) => transformer
                .forward_precomputed_step(context, prepared_schedule, prepared_step, &inputs)?,
        };
        let video_velocity = output.video.squeeze(0)?.narrow(
            0,
            layout.condition_video_rows,
            layout.target_video_rows,
        )?;
        let audio_velocity = output.audio.squeeze(0)?.narrow(
            0,
            layout.condition_audio_rows,
            layout.target_audio_rows,
        )?;
        target_video_rows =
            video_scheduler.step(&video_velocity, video_timestep, &target_video_rows)?;
        target_audio_rows =
            audio_scheduler.step(&audio_velocity, audio_timestep, &target_audio_rows)?;
        if observer.synchronize_device_timings() || checkpoint_each_evaluation {
            device.synchronize()?;
        }
        observer.on_step_completed(DenoiseStepEvent {
            step_index: step,
            total_steps: video_timesteps.len(),
            video_timestep,
            audio_timestep,
            step_elapsed: step_start.elapsed(),
            total_elapsed: total_start.elapsed(),
            timing_synchronized: observer.synchronize_device_timings(),
        })?;
        if checkpoint_each_evaluation {
            let checkpoint = materialize_target(
                &target_video_rows,
                &target_audio_rows,
                &video_dims,
                patch_size,
                audio_channels,
                audio_frames,
                step + 1,
            )?;
            observer.on_checkpoint_boundary(DenoiseCheckpointEvent {
                latents: &checkpoint,
                total_steps: video_timesteps.len(),
            })?;
            if step + 1 == end_step {
                final_checkpoint = Some(checkpoint);
            }
        }
    }
    match final_checkpoint {
        Some(checkpoint) => Ok(checkpoint),
        None => materialize_target(
            &target_video_rows,
            &target_audio_rows,
            &video_dims,
            patch_size,
            audio_channels,
            audio_frames,
            end_step,
        ),
    }
}

fn materialize_target(
    video_rows: &Tensor,
    audio_rows: &Tensor,
    video_dims: &[usize],
    patch_size: [usize; 3],
    audio_channels: usize,
    audio_frames: usize,
    completed_steps: usize,
) -> Result<T2vaLatents> {
    Ok(T2vaLatents {
        video: layout::unpatchify_video(
            video_rows,
            1,
            video_dims[1],
            video_dims[2],
            video_dims[3],
            video_dims[4],
            patch_size,
        )?,
        audio: layout::unpack_audio(audio_rows, audio_channels, audio_frames)?,
        completed_steps,
    })
}

fn validate_common(
    text_token_tags: &[u32],
    frames: usize,
    height: usize,
    width: usize,
    patch_size: [usize; 3],
    audio_channels: usize,
) -> Result<()> {
    ensure!(
        !text_token_tags.is_empty() && text_token_tags.iter().all(|tag| *tag < 3),
        "conditioned text tags must be non-empty valid modality tags"
    );
    ensure!(
        frames > 0 && height > 0 && width > 0,
        "latent geometry must be non-zero"
    );
    ensure!(
        patch_size[0] == 1,
        "conditioned H3 requires temporal patch size one"
    );
    ensure!(
        audio_channels == 2,
        "conditioned MiniMax-H3 requires stereo audio rows"
    );
    checked_rows_per_frame(height, width, patch_size[1], patch_size[2])?;
    Ok(())
}

fn checked_rows_per_frame(
    height: usize,
    width: usize,
    patch_h: usize,
    patch_w: usize,
) -> Result<usize> {
    ensure!(
        patch_h > 0 && patch_w > 0,
        "patch dimensions must be non-zero"
    );
    ensure!(
        height.is_multiple_of(patch_h) && width.is_multiple_of(patch_w),
        "latent canvas is not divisible by the transformer patch"
    );
    (height / patch_h)
        .checked_mul(width / patch_w)
        .context("rows per frame overflow")
}

fn frame_grid(
    height: usize,
    width: usize,
    patch_h: usize,
    patch_w: usize,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let sqrt_area = ((height * width) as f64).sqrt();
    let heights = spatial_grid(height, patch_h, sqrt_area);
    let widths = spatial_grid(width, patch_w, sqrt_area);
    let mut grid = Vec::with_capacity(heights.len() * widths.len());
    for &h in &heights {
        for &w in &widths {
            grid.push((h, w));
        }
    }
    (grid, widths)
}

fn spatial_grid(dimension: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dimension as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let stop = left + ratio;
    let count = dimension / patch;
    (0..count)
        .map(|index| (left + index as f64 * (stop - left) / count as f64) * ROPE_SPATIAL_SCALE)
        .collect()
}

fn temporal_grid(frames: usize, start: f64) -> Vec<f64> {
    let mut time = start;
    (0..frames)
        .map(|frame| {
            let current = time;
            time += frame_duration(frame);
            current
        })
        .collect()
}

fn frame_duration(frame: usize) -> f64 {
    ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[frame % ROPE_FRAMES_PER_LATENT.len()]
}

fn fill_text_positions(positions: &mut [f64], rows: usize) {
    for row in 0..rows {
        positions[row * 3] = row as f64;
    }
}

fn fill_video_block(
    positions: &mut [f64],
    start: usize,
    frames: usize,
    grid: &[(f64, f64)],
    times: &[f64],
) {
    debug_assert_eq!(times.len(), frames);
    for (frame, &time) in times.iter().enumerate() {
        for (spatial, &(height, width)) in grid.iter().enumerate() {
            let row = start + frame * grid.len() + spatial;
            positions[row * 3] = time;
            positions[row * 3 + 1] = height;
            positions[row * 3 + 2] = width;
        }
    }
}

fn fill_audio_block(
    positions: &mut [f64],
    start: usize,
    latents: usize,
    time: f64,
    width_grid: &[f64],
    channels: usize,
) -> Result<()> {
    ensure!(
        !width_grid.is_empty(),
        "audio positions require a spatial width grid"
    );
    for channel in 0..channels {
        let width = if channel == 0 {
            width_grid[0]
        } else {
            *width_grid.last().expect("width grid is non-empty")
        };
        for latent in 0..latents {
            let row = start + channel * latents + latent;
            positions[row * 3] = time + latent as f64;
            positions[row * 3 + 2] = width;
        }
    }
    Ok(())
}

fn assign_tags(tags: &mut [u32], indices: &[u32], tag: u32) -> Result<()> {
    for &index in indices {
        *tags
            .get_mut(index as usize)
            .context("modality index exceeds packed sequence")? = tag;
    }
    Ok(())
}

fn range_u32(start: usize, end: usize) -> Result<Vec<u32>> {
    (start..end)
        .map(|value| u32::try_from(value).context("packed sequence exceeds U32 indexing"))
        .collect()
}

fn distinct_timesteps<const N: usize>(assignments: [(usize, f32); N]) -> Vec<f32> {
    let mut values = assignments
        .into_iter()
        .filter_map(|(count, value)| (count > 0).then_some(value))
        .collect::<Vec<_>>();
    values.sort_by(f32::total_cmp);
    values.dedup();
    values
}

fn timestep_index(values: &[f32], target: f32) -> Result<u32> {
    values
        .iter()
        .position(|value| *value == target)
        .map(|index| index as u32)
        .context("failed to invert conditioned timestep table")
}

fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    if values.len() < 8 {
        return values.iter().fold(-0.0, |sum, value| sum + value);
    }
    if values.len() <= BLOCK {
        let mut accumulators = [0.0; 8];
        accumulators.copy_from_slice(&values[..8]);
        let mut index = 8;
        while index + 7 < values.len() {
            for lane in 0..8 {
                accumulators[lane] += values[index + lane];
            }
            index += 8;
        }
        let mut total = ((accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]))
            + ((accumulators[4] + accumulators[5]) + (accumulators[6] + accumulators[7]));
        while index < values.len() {
            total += values[index];
            index += 1;
        }
        return total;
    }
    let mut middle = values.len() / 2;
    middle -= middle % 8;
    pairwise_sum(&values[..middle]) + pairwise_sum(&values[middle..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fl2va_places_frozen_video_before_target_audio_and_video() {
        let layout = ConditionedLayout::fl2va(
            &[TEXT_TAG, TEXT_TAG],
            2,
            2,
            4,
            3,
            [1, 2, 2],
            2,
            &[KeyframeAnchor::First, KeyframeAnchor::Last],
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(layout.condition_video_rows, 4);
        assert_eq!(layout.target_audio_rows, 6);
        assert_eq!(layout.target_video_rows, 4);
        assert_eq!(
            layout.video_indices.to_vec1::<u32>().unwrap(),
            vec![2, 3, 4, 5, 12, 13, 14, 15]
        );
        assert_eq!(
            layout.audio_indices.to_vec1::<u32>().unwrap(),
            vec![6, 7, 8, 9, 10, 11]
        );
        let (times, inverse) = layout.row_timesteps(0.2, 0.4, &Device::Cpu).unwrap();
        assert_eq!(times.to_vec1::<f32>().unwrap(), vec![0.2, 0.4, 0.999]);
        let inverse = inverse.to_vec1::<u32>().unwrap();
        assert_eq!(&inverse[2..6], &[2, 2, 2, 2]);
        assert_eq!(&inverse[6..12], &[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn ref2va_interleaves_reference_blocks_and_pins_both_condition_modalities() {
        let layout = ConditionedLayout::ref2va(
            &[TEXT_TAG],
            &[
                ReferenceBlock::Image {
                    latent_frames: 1,
                    latent_height: 2,
                    latent_width: 4,
                },
                ReferenceBlock::Audio { audio_latents: 2 },
                ReferenceBlock::Video {
                    latent_frames: 2,
                    latent_height: 2,
                    latent_width: 4,
                    audio_latents: 1,
                },
            ],
            1,
            2,
            4,
            1,
            [1, 2, 2],
            2,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(layout.condition_video_rows, 6);
        assert_eq!(layout.condition_audio_rows, 6);
        assert_eq!(layout.target_video_rows, 2);
        assert_eq!(layout.target_audio_rows, 2);
        assert_eq!(layout.sequence_length, 17);
        let (times, inverse) = layout.row_timesteps(0.2, 0.4, &Device::Cpu).unwrap();
        assert_eq!(times.to_vec1::<f32>().unwrap(), vec![0.2, 0.4, 0.999, 1.0]);
        let inverse = inverse.to_vec1::<u32>().unwrap();
        for &row in &layout.video_indices_host[..layout.condition_video_rows] {
            assert_eq!(inverse[row as usize], 2);
        }
        for &row in &layout.audio_indices_host[..layout.condition_audio_rows] {
            assert_eq!(inverse[row as usize], 3);
        }
    }

    #[test]
    fn last_anchor_sum_matches_numpy_pairwise_order() {
        let spans = (0..57).map(frame_duration).collect::<Vec<_>>();
        assert_eq!(pairwise_sum(&spans).to_bits(), 0x4074_0000_0000_0001);
    }

    #[test]
    fn conditioned_resume_rejects_max_steps_beyond_the_remaining_schedule() {
        let error = resolve_run_steps(2, Some(3), "conditioned H3").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("conditioned H3 max_steps 3 exceeds the 2 remaining evaluations")
        );
        assert_eq!(resolve_run_steps(2, None, "conditioned H3").unwrap(), 2);
    }

    #[test]
    fn zero_length_audio_is_rejected_instead_of_becoming_an_empty_reference() {
        assert!(
            ConditionedLayout::ref2va(
                &[TEXT_TAG],
                &[
                    ReferenceBlock::Image {
                        latent_frames: 1,
                        latent_height: 2,
                        latent_width: 2,
                    },
                    ReferenceBlock::Audio { audio_latents: 0 },
                ],
                1,
                2,
                2,
                1,
                [1, 2, 2],
                2,
                &Device::Cpu,
            )
            .is_err()
        );
        assert!(
            ConditionedLayout::fl2va(
                &[TEXT_TAG],
                1,
                2,
                2,
                0,
                [1, 2, 2],
                2,
                &[KeyframeAnchor::First],
                &Device::Cpu,
            )
            .is_err()
        );
    }
}
