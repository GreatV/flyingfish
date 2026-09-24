use crate::{
    layout::{self, PackedLayout},
    model::{
        PreparedDenoiseSchedule, PreparedTransformerContext, StreamedTransformer,
        TransformerStaticInputs, TransformerStepInputs,
    },
    scheduler::H3Scheduler,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::time::{Duration, Instant};

pub struct T2vaLatents {
    pub video: Tensor,
    pub audio: Tensor,
    pub completed_steps: usize,
}

pub struct DenoiseCheckpointEvent<'a> {
    pub latents: &'a T2vaLatents,
    pub total_steps: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DenoiseStepEvent {
    pub step_index: usize,
    pub total_steps: usize,
    pub video_timestep: f32,
    pub audio_timestep: f32,
    pub step_elapsed: Duration,
    pub total_elapsed: Duration,
    pub timing_synchronized: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DenoisePreparationEvent {
    pub prepared_evaluations: usize,
    pub elapsed: Duration,
    pub timing_synchronized: bool,
}

pub trait DenoiseObserver {
    fn synchronize_device_timings(&self) -> bool {
        false
    }

    fn on_preparation_completed(&mut self, _event: DenoisePreparationEvent) -> Result<()> {
        Ok(())
    }

    fn checkpoint_each_evaluation(&self) -> bool {
        false
    }

    fn on_checkpoint_boundary(&mut self, _event: DenoiseCheckpointEvent<'_>) -> Result<()> {
        anyhow::bail!("denoise observer requested checkpoints without handling checkpoint events")
    }

    fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()>;
}

#[cfg(test)]
struct NoopObserver;

#[cfg(test)]
impl DenoiseObserver for NoopObserver {
    fn on_step_completed(&mut self, _event: DenoiseStepEvent) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct T2vaSchedule {
    pub sigma_points: usize,
    pub video_shift: f32,
    pub audio_shift: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct T2vaExecutionOptions {
    pub precompute_adaln: bool,
    pub start_step: usize,
    pub max_steps: Option<usize>,
}

/// The prompt a denoise loop attends to, embeddings plus per-row modality tags.
#[derive(Clone, Copy)]
pub struct PromptConditioning<'a> {
    pub embeddings: &'a Tensor,
    pub text_token_tags: &'a [u32],
}

/// The initial noise a T2VA denoise loop starts from.
#[derive(Clone, Copy)]
pub struct T2vaInitialLatents<'a> {
    pub video: &'a Tensor,
    pub audio: &'a Tensor,
}

impl Default for T2vaExecutionOptions {
    fn default() -> Self {
        Self {
            precompute_adaln: true,
            start_step: 0,
            max_steps: None,
        }
    }
}

enum PreparedExecution {
    Dynamic(PreparedTransformerContext),
    Precomputed(PreparedTransformerContext, PreparedDenoiseSchedule),
}

impl Default for T2vaSchedule {
    fn default() -> Self {
        Self {
            sigma_points: 50,
            video_shift: 12.,
            audio_shift: 3.,
        }
    }
}

pub fn denoise_t2va_with_options_and_observer(
    transformer: &StreamedTransformer,
    prompt: PromptConditioning<'_>,
    initial_latents: T2vaInitialLatents<'_>,
    schedule: T2vaSchedule,
    options: T2vaExecutionOptions,
    observer: &mut dyn DenoiseObserver,
) -> Result<T2vaLatents> {
    let PromptConditioning {
        embeddings: prompt_embeddings,
        text_token_tags,
    } = prompt;
    let T2vaInitialLatents {
        video: initial_video_latents,
        audio: initial_audio_latents,
    } = initial_latents;
    let config = transformer.config();
    anyhow::ensure!(
        prompt_embeddings.device().same_device(transformer.device()),
        "prompt embeddings are on the wrong device"
    );
    run_t2va_loop(
        T2vaLoopRequest {
            prompt_embeddings,
            text_token_tags,
            initial_video_latents,
            initial_audio_latents,
            patch_size: config.patch_size,
            sigma_points: schedule.sigma_points,
            video_shift: schedule.video_shift,
            audio_shift: schedule.audio_shift,
            start_step: options.start_step,
            max_steps: options.max_steps,
            device: transformer.device(),
        },
        observer,
        |layout, timestep_tables| {
            let context = transformer.prepare_context(&TransformerStaticInputs {
                encoder_hidden_states: prompt_embeddings,
                token_tags: layout.token_tags(),
                position_ids: layout.position_ids(),
                video_indices: layout.video_indices(),
                audio_indices: layout.audio_indices(),
                text_indices: layout.text_indices(),
            })?;
            if options.precompute_adaln {
                Ok(PreparedExecution::Precomputed(
                    context,
                    transformer.prepare_denoise_schedule(timestep_tables)?,
                ))
            } else {
                Ok(PreparedExecution::Dynamic(context))
            }
        },
        |prepared: &PreparedExecution,
         step_index,
         video_rows,
         audio_rows,
         timesteps,
         timestep_indices,
         _layout| {
            let inputs = TransformerStepInputs {
                video_hidden_states: &video_rows.unsqueeze(0)?,
                audio_hidden_states: &audio_rows.unsqueeze(0)?,
                timestep: timesteps,
                timestep_indices,
            };
            let output = match prepared {
                PreparedExecution::Dynamic(context) => {
                    transformer.forward_step(context, &inputs)?
                }
                PreparedExecution::Precomputed(context, schedule) => {
                    transformer.forward_precomputed_step(context, schedule, step_index, &inputs)?
                }
            };
            Ok((output.video.squeeze(0)?, output.audio.squeeze(0)?))
        },
    )
}

/// Everything the shared T2VA denoising loop needs about one request.
///
/// The loop takes this as one value rather than a dozen positional arguments:
/// several of them are same-typed scalars whose order would otherwise be easy
/// to transpose at a call site.
struct T2vaLoopRequest<'a> {
    prompt_embeddings: &'a Tensor,
    text_token_tags: &'a [u32],
    initial_video_latents: &'a Tensor,
    initial_audio_latents: &'a Tensor,
    patch_size: [usize; 3],
    sigma_points: usize,
    video_shift: f32,
    audio_shift: f32,
    start_step: usize,
    max_steps: Option<usize>,
    device: &'a Device,
}

fn run_t2va_loop<C>(
    request: T2vaLoopRequest<'_>,
    observer: &mut dyn DenoiseObserver,
    prepare: impl FnOnce(&PackedLayout, &[Vec<f32>]) -> Result<C>,
    predict: impl FnMut(
        &C,
        usize,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &PackedLayout,
    ) -> Result<(Tensor, Tensor)>,
) -> Result<T2vaLatents> {
    run_t2va_loop_with_synchronizer(
        request,
        observer,
        &mut |device| device.synchronize().map_err(Into::into),
        prepare,
        predict,
    )
}

fn run_t2va_loop_with_synchronizer<C>(
    request: T2vaLoopRequest<'_>,
    observer: &mut dyn DenoiseObserver,
    synchronize_timing_boundary: &mut dyn FnMut(&Device) -> Result<()>,
    prepare: impl FnOnce(&PackedLayout, &[Vec<f32>]) -> Result<C>,
    mut predict: impl FnMut(
        &C,
        usize,
        &Tensor,
        &Tensor,
        &Tensor,
        &Tensor,
        &PackedLayout,
    ) -> Result<(Tensor, Tensor)>,
) -> Result<T2vaLatents> {
    let T2vaLoopRequest {
        prompt_embeddings,
        text_token_tags,
        initial_video_latents,
        initial_audio_latents,
        patch_size,
        sigma_points,
        video_shift,
        audio_shift,
        start_step,
        max_steps,
        device,
    } = request;
    let video_dims = initial_video_latents.dims();
    anyhow::ensure!(
        video_dims.len() == 5 && video_dims[0] == 1,
        "T2VA video latents must be [1, channels, frames, height, width]"
    );
    let (audio_channels, _, audio_frames) = initial_audio_latents
        .dims3()
        .context("audio latents must be [channels, latent_channels, frames]")?;
    let (prompt_batch, text_rows, _) = prompt_embeddings
        .dims3()
        .context("prompt embeddings must be [1, rows, width]")?;
    anyhow::ensure!(
        prompt_batch == 1,
        "H3 T2VA currently supports batch size one"
    );
    anyhow::ensure!(
        text_rows == text_token_tags.len(),
        "text token tag count differs from prompt rows"
    );
    for (name, tensor) in [
        ("video latents", initial_video_latents),
        ("audio latents", initial_audio_latents),
    ] {
        anyhow::ensure!(
            tensor.device().same_device(device),
            "{name} are on the wrong device"
        );
        anyhow::ensure!(
            tensor.dtype() == DType::F32,
            "{name} must be F32 at the pipeline boundary"
        );
    }
    anyhow::ensure!(
        prompt_embeddings.device().same_device(device),
        "prompt embeddings are on the wrong device"
    );
    anyhow::ensure!(
        matches!(
            prompt_embeddings.dtype(),
            DType::F32 | DType::F16 | DType::BF16
        ),
        "prompt embeddings must use a floating-point dtype"
    );

    let mut video_scheduler = H3Scheduler::new(video_shift)?;
    let mut audio_scheduler = H3Scheduler::new(audio_shift)?;
    let video_timesteps = video_scheduler.set_timesteps(sigma_points)?.to_vec();
    let audio_timesteps = audio_scheduler.set_timesteps(sigma_points)?.to_vec();
    anyhow::ensure!(
        video_timesteps.len() == audio_timesteps.len(),
        "video/audio shifted schedules have different evaluation counts"
    );
    let schedule_steps = video_timesteps.len();
    anyhow::ensure!(
        start_step < schedule_steps,
        "start_step {start_step} is outside {schedule_steps} evaluations"
    );
    if start_step > 0 {
        video_scheduler.seek(start_step)?;
        audio_scheduler.seek(start_step)?;
    }
    let remaining = schedule_steps - start_step;
    let run_steps = resolve_run_steps(remaining, max_steps, "T2VA")?;
    let end_step = start_step + run_steps;
    let layout = PackedLayout::t2va(
        text_token_tags,
        layout::LatentGeometry {
            latent_frames: video_dims[2],
            latent_height: video_dims[3],
            latent_width: video_dims[4],
            audio_latents: audio_frames,
            patch_size,
            audio_channels,
        },
        device,
    )?;
    let mut video_rows = layout::patchify_video(initial_video_latents, patch_size)?;
    let mut audio_rows = layout::pack_audio(initial_audio_latents)?;
    anyhow::ensure!(
        video_rows.dim(0)? == layout.video_rows(),
        "video row count disagrees with layout"
    );
    anyhow::ensure!(
        audio_rows.dim(0)? == layout.audio_rows(),
        "audio row count disagrees with layout"
    );
    let timestep_tables = video_timesteps[start_step..end_step]
        .iter()
        .zip(&audio_timesteps[start_step..end_step])
        .map(|(&video_timestep, &audio_timestep)| {
            layout.t2va_timestep_table(video_timestep, audio_timestep)
        })
        .collect::<Vec<_>>();
    let synchronize_timings = observer.synchronize_device_timings();
    if synchronize_timings {
        synchronize_timing_boundary(device)?;
    }
    let preparation_start = Instant::now();
    let prepared = prepare(&layout, &timestep_tables)?;
    if synchronize_timings {
        synchronize_timing_boundary(device)?;
    }
    observer.on_preparation_completed(DenoisePreparationEvent {
        prepared_evaluations: run_steps,
        elapsed: preparation_start.elapsed(),
        timing_synchronized: synchronize_timings,
    })?;
    let total_steps = schedule_steps;
    let total_start = Instant::now();
    let checkpoint_each_evaluation = observer.checkpoint_each_evaluation();
    let mut final_checkpoint = None;
    for prepared_step_index in 0..run_steps {
        let step_index = start_step + prepared_step_index;
        let video_timestep = video_timesteps[step_index];
        let audio_timestep = audio_timesteps[step_index];
        if synchronize_timings {
            synchronize_timing_boundary(device)?;
        }
        let step_start = Instant::now();
        let (timesteps, timestep_indices) =
            layout.t2va_row_timesteps(video_timestep, audio_timestep, device)?;
        let (video_velocity, audio_velocity) = predict(
            &prepared,
            prepared_step_index,
            &video_rows,
            &audio_rows,
            &timesteps,
            &timestep_indices,
            &layout,
        )?;
        video_rows = video_scheduler.step(&video_velocity, video_timestep, &video_rows)?;
        audio_rows = audio_scheduler.step(&audio_velocity, audio_timestep, &audio_rows)?;
        if synchronize_timings || checkpoint_each_evaluation {
            drop(video_velocity);
            drop(audio_velocity);
            drop(timesteps);
            drop(timestep_indices);
        }
        if synchronize_timings {
            synchronize_timing_boundary(device)?;
        }
        observer.on_step_completed(DenoiseStepEvent {
            step_index,
            total_steps,
            video_timestep,
            audio_timestep,
            step_elapsed: step_start.elapsed(),
            total_elapsed: total_start.elapsed(),
            timing_synchronized: synchronize_timings,
        })?;
        if checkpoint_each_evaluation {
            let completed_steps = step_index
                .checked_add(1)
                .context("completed evaluation count overflow")?;
            let checkpoint = materialize_t2va_latents(
                &video_rows,
                &audio_rows,
                video_dims,
                patch_size,
                audio_channels,
                audio_frames,
                completed_steps,
            )?;
            synchronize_timing_boundary(device)?;
            observer.on_checkpoint_boundary(DenoiseCheckpointEvent {
                latents: &checkpoint,
                total_steps,
            })?;
            if completed_steps == end_step {
                final_checkpoint = Some(checkpoint);
            }
        }
    }

    match final_checkpoint {
        Some(checkpoint) => Ok(checkpoint),
        None => materialize_t2va_latents(
            &video_rows,
            &audio_rows,
            video_dims,
            patch_size,
            audio_channels,
            audio_frames,
            end_step,
        ),
    }
}

pub(crate) fn resolve_run_steps(
    remaining: usize,
    max_steps: Option<usize>,
    workflow: &str,
) -> Result<usize> {
    anyhow::ensure!(remaining > 0, "{workflow} has no remaining evaluations");
    match max_steps {
        Some(requested) => {
            anyhow::ensure!(
                requested > 0,
                "{workflow} max_steps must allow at least one evaluation"
            );
            anyhow::ensure!(
                requested <= remaining,
                "{workflow} max_steps {requested} exceeds the {remaining} remaining evaluations"
            );
            Ok(requested)
        }
        None => Ok(remaining),
    }
}

fn materialize_t2va_latents(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecordingObserver {
        steps: Vec<usize>,
        preparations: usize,
        synchronize_timings: bool,
        preparation_timing_flags: Vec<bool>,
        step_timing_flags: Vec<bool>,
        step_elapsed: Vec<Duration>,
        checkpoint_each_evaluation: bool,
        checkpoint_steps: Vec<usize>,
        checkpoint_totals: Vec<usize>,
        checkpoint_error_at: Option<usize>,
    }

    impl DenoiseObserver for RecordingObserver {
        fn synchronize_device_timings(&self) -> bool {
            self.synchronize_timings
        }

        fn on_preparation_completed(&mut self, event: DenoisePreparationEvent) -> Result<()> {
            self.preparations += 1;
            self.preparation_timing_flags
                .push(event.timing_synchronized);
            Ok(())
        }

        fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()> {
            self.steps.push(event.step_index);
            self.step_timing_flags.push(event.timing_synchronized);
            self.step_elapsed.push(event.step_elapsed);
            Ok(())
        }

        fn checkpoint_each_evaluation(&self) -> bool {
            self.checkpoint_each_evaluation
        }

        fn on_checkpoint_boundary(&mut self, event: DenoiseCheckpointEvent<'_>) -> Result<()> {
            let completed_steps = event.latents.completed_steps;
            anyhow::ensure!(
                self.checkpoint_error_at != Some(completed_steps),
                "checkpoint sentinel at evaluation {completed_steps}"
            );
            self.checkpoint_steps.push(completed_steps);
            self.checkpoint_totals.push(event.total_steps);
            Ok(())
        }
    }

    #[test]
    fn zero_velocity_loop_preserves_both_noise_streams() {
        let video_values = (0..32).map(|v| v as f32).collect::<Vec<_>>();
        let video = Tensor::from_vec(video_values.clone(), (1, 2, 2, 2, 4), &Device::Cpu).unwrap();
        let audio_values = (0..12).map(|v| v as f32).collect::<Vec<_>>();
        let audio = Tensor::from_vec(audio_values.clone(), (2, 2, 3), &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = RecordingObserver::default();
        let output = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1, 1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 2, 2],
                sigma_points: 4,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 0,
                max_steps: None,
                device: &Device::Cpu,
            },
            &mut observer,
            |_, _| Ok(()),
            |_, _, video_rows, audio_rows, _, _, _| {
                Ok((
                    Tensor::zeros_like(video_rows)?,
                    Tensor::zeros_like(audio_rows)?,
                ))
            },
        )
        .unwrap();
        assert_eq!(
            output
                .video
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            video_values
        );
        assert_eq!(
            output
                .audio
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            audio_values
        );
        assert_eq!(observer.steps, vec![0, 1, 2]);
        assert_eq!(observer.preparations, 1);
        assert_eq!(observer.preparation_timing_flags, vec![false]);
        assert_eq!(observer.step_timing_flags, vec![false, false, false]);
        assert_eq!(output.completed_steps, 3);
    }

    #[test]
    fn observer_can_request_synchronized_device_makespan_boundaries() {
        let video = Tensor::zeros((1, 2, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((1, 2, 1), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = RecordingObserver {
            synchronize_timings: true,
            ..RecordingObserver::default()
        };
        let mut synchronization_calls = 0usize;
        let mut synchronize = |device: &Device| {
            synchronization_calls += 1;
            device.synchronize().map_err(Into::into)
        };
        run_t2va_loop_with_synchronizer(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 1, 1],
                sigma_points: 2,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 0,
                max_steps: None,
                device: &Device::Cpu,
            },
            &mut observer,
            &mut synchronize,
            |_, _| Ok(()),
            |_, _, video_rows, audio_rows, _, _, _| {
                Ok((
                    Tensor::zeros_like(video_rows)?,
                    Tensor::zeros_like(audio_rows)?,
                ))
            },
        )
        .unwrap();

        assert_eq!(observer.preparation_timing_flags, vec![true]);
        assert_eq!(observer.step_timing_flags, vec![true]);
        assert_eq!(synchronization_calls, 4);
    }

    #[test]
    fn checkpoint_callback_is_synchronized_at_every_successful_evaluation() {
        let video = Tensor::zeros((1, 2, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((1, 2, 1), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = RecordingObserver {
            checkpoint_each_evaluation: true,
            ..RecordingObserver::default()
        };
        let mut synchronization_calls = 0usize;
        let mut synchronize = |device: &Device| {
            synchronization_calls += 1;
            device.synchronize().map_err(Into::into)
        };
        let output = run_t2va_loop_with_synchronizer(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 1, 1],
                sigma_points: 4,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 0,
                max_steps: None,
                device: &Device::Cpu,
            },
            &mut observer,
            &mut synchronize,
            |_, _| Ok(()),
            |_, _, video_rows, audio_rows, _, _, _| {
                Ok((
                    Tensor::zeros_like(video_rows)?,
                    Tensor::zeros_like(audio_rows)?,
                ))
            },
        )
        .unwrap();

        assert_eq!(observer.checkpoint_steps, vec![1, 2, 3]);
        assert_eq!(observer.checkpoint_totals, vec![3, 3, 3]);
        assert_eq!(synchronization_calls, 3);
        assert_eq!(output.completed_steps, 3);
    }

    #[test]
    fn checkpoint_failure_stops_before_the_next_evaluation() {
        let video = Tensor::zeros((1, 2, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((1, 2, 1), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = RecordingObserver {
            checkpoint_each_evaluation: true,
            checkpoint_error_at: Some(1),
            ..RecordingObserver::default()
        };
        let mut predictions = 0usize;
        let error = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 1, 1],
                sigma_points: 4,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 0,
                max_steps: None,
                device: &Device::Cpu,
            },
            &mut observer,
            |_, _| Ok(()),
            |_, _, video_rows, audio_rows, _, _, _| {
                predictions += 1;
                Ok((
                    Tensor::zeros_like(video_rows)?,
                    Tensor::zeros_like(audio_rows)?,
                ))
            },
        )
        .err()
        .expect("checkpoint sentinel must stop the loop");

        assert_eq!(predictions, 1);
        assert_eq!(observer.steps, vec![0]);
        assert!(error.to_string().contains("checkpoint sentinel"));
    }

    #[test]
    fn resumed_range_reports_absolute_progress() {
        let video = Tensor::zeros((1, 2, 2, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((2, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = RecordingObserver::default();
        let output = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1, 1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 2, 2],
                sigma_points: 4,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 1,
                max_steps: Some(1),
                device: &Device::Cpu,
            },
            &mut observer,
            |_, tables| {
                assert_eq!(tables.len(), 1);
                Ok(())
            },
            |_, prepared_step, video_rows, audio_rows, _, _, _| {
                assert_eq!(prepared_step, 0);
                Ok((
                    Tensor::zeros_like(video_rows)?,
                    Tensor::zeros_like(audio_rows)?,
                ))
            },
        )
        .unwrap();
        assert_eq!(observer.steps, vec![1]);
        assert_eq!(output.completed_steps, 2);
    }

    #[test]
    fn resumed_execution_produces_exactly_the_uninterrupted_latents() {
        let video = Tensor::from_vec(
            (0..8).map(|value| value as f32 / 8.0).collect::<Vec<_>>(),
            (1, 1, 2, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let audio = Tensor::from_vec(
            (0..4).map(|value| value as f32 / 4.0).collect::<Vec<_>>(),
            (1, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let prompt = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let predict = |_: &(),
                       _: usize,
                       video_rows: &Tensor,
                       audio_rows: &Tensor,
                       _: &Tensor,
                       _: &Tensor,
                       _: &PackedLayout|
         -> Result<(Tensor, Tensor)> {
            Ok((
                video_rows.affine(0.25, 0.125)?,
                audio_rows.affine(-0.5, 0.25)?,
            ))
        };

        let mut uninterrupted_observer = RecordingObserver::default();
        let uninterrupted = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 1, 1],
                sigma_points: 4,
                video_shift: 12.0,
                audio_shift: 3.0,
                start_step: 0,
                max_steps: None,
                device: &Device::Cpu,
            },
            &mut uninterrupted_observer,
            |_, _| Ok(()),
            predict,
        )
        .unwrap();

        let mut prefix_observer = RecordingObserver::default();
        let prefix = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 1, 1],
                sigma_points: 4,
                video_shift: 12.0,
                audio_shift: 3.0,
                start_step: 0,
                max_steps: Some(1),
                device: &Device::Cpu,
            },
            &mut prefix_observer,
            |_, _| Ok(()),
            predict,
        )
        .unwrap();
        let mut resume_observer = RecordingObserver::default();
        let resumed = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1],
                initial_video_latents: &prefix.video,
                initial_audio_latents: &prefix.audio,
                patch_size: [1, 1, 1],
                sigma_points: 4,
                video_shift: 12.0,
                audio_shift: 3.0,
                start_step: prefix.completed_steps,
                max_steps: None,
                device: &Device::Cpu,
            },
            &mut resume_observer,
            |_, _| Ok(()),
            predict,
        )
        .unwrap();

        assert_eq!(resumed.completed_steps, uninterrupted.completed_steps);
        assert_eq!(
            resumed
                .video
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            uninterrupted
                .video
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(
            resumed
                .audio
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            uninterrupted
                .audio
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn oversized_resumed_max_steps_is_rejected_before_preparation() {
        let video = Tensor::zeros((1, 2, 2, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((2, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = NoopObserver;
        let result = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1, 1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 2, 2],
                sigma_points: 4,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 1,
                max_steps: Some(3),
                device: &Device::Cpu,
            },
            &mut observer,
            |_, _| -> Result<()> { panic!("oversized range reached preparation") },
            |_, _, _, _, _, _, _| unreachable!(),
        );
        let Err(error) = result else {
            panic!("oversized max_steps must fail");
        };
        assert!(
            error
                .to_string()
                .contains("max_steps 3 exceeds the 2 remaining evaluations")
        );
    }

    #[test]
    fn zero_audio_rows_prepare_only_the_assigned_video_timestep() {
        let video = Tensor::zeros((1, 2, 1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((2, 2, 0), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = NoopObserver;
        let error = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1, 1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 2, 2],
                sigma_points: 3,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 0,
                max_steps: Some(1),
                device: &Device::Cpu,
            },
            &mut observer,
            |layout, tables| -> Result<()> {
                assert_eq!(layout.audio_rows(), 0);
                assert_eq!(tables.len(), 1);
                assert_eq!(tables[0].len(), 1);
                anyhow::bail!("stop after schedule preparation")
            },
            |_, _, _, _, _, _, _| unreachable!(),
        )
        .err()
        .expect("the preparation sentinel must stop the loop");
        assert!(
            error
                .to_string()
                .contains("stop after schedule preparation")
        );
    }

    #[test]
    fn zero_sized_video_geometry_is_an_error() {
        let video = Tensor::zeros((1, 2, 1, 2, 0), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((2, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let mut observer = NoopObserver;
        let error = run_t2va_loop(
            T2vaLoopRequest {
                prompt_embeddings: &prompt,
                text_token_tags: &[1, 1],
                initial_video_latents: &video,
                initial_audio_latents: &audio,
                patch_size: [1, 2, 2],
                sigma_points: 3,
                video_shift: 12.,
                audio_shift: 3.,
                start_step: 0,
                max_steps: Some(1),
                device: &Device::Cpu,
            },
            &mut observer,
            |_, _| Ok(()),
            |_, _, _, _, _, _, _| unreachable!(),
        )
        .err()
        .expect("zero video geometry must be rejected");
        assert!(
            error
                .to_string()
                .contains("video latent frame, height, and width dimensions must be non-zero")
        );
    }

    #[test]
    fn segmented_execution_matches_an_uninterrupted_schedule() {
        let video = Tensor::zeros((1, 2, 2, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let audio = Tensor::zeros((2, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let prompt = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let run = |video: &Tensor, audio: &Tensor, start_step, max_steps| {
            let mut observer = NoopObserver;
            run_t2va_loop(
                T2vaLoopRequest {
                    prompt_embeddings: &prompt,
                    text_token_tags: &[1, 1],
                    initial_video_latents: video,
                    initial_audio_latents: audio,
                    patch_size: [1, 2, 2],
                    sigma_points: 4,
                    video_shift: 12.,
                    audio_shift: 3.,
                    start_step,
                    max_steps,
                    device: &Device::Cpu,
                },
                &mut observer,
                |_, _| Ok(()),
                |_, _, video_rows, audio_rows, _, _, _| {
                    Ok((
                        Tensor::ones_like(video_rows)?,
                        Tensor::ones_like(audio_rows)?,
                    ))
                },
            )
            .unwrap()
        };
        let full = run(&video, &audio, 0, None);
        let first = run(&video, &audio, 0, Some(1));
        assert_eq!(first.completed_steps, 1);
        let resumed = run(&first.video, &first.audio, first.completed_steps, None);
        assert_eq!(resumed.completed_steps, full.completed_steps);
        assert_eq!(
            resumed
                .video
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            full.video.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            resumed
                .audio
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            full.audio.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}
