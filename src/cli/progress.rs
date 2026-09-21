use anyhow::{Context, Result};
use flyingfish::h3::pipeline::{DenoisePreparationEvent, DenoiseStepEvent};
use std::collections::VecDeque;
use std::io::{IsTerminal, Write};
use std::time::Duration;

const DEFAULT_TRAILING_WINDOW: usize = 5;
const DEFAULT_LOG_EVALUATION_INTERVAL: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputMode {
    Terminal,
    Log,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Forecast {
    Stabilizing {
        samples: usize,
        required: usize,
    },
    Ready {
        window: usize,
        mean: Duration,
        remaining: Duration,
    },
}

#[derive(Debug)]
struct TrailingWindow {
    capacity: usize,
    samples: VecDeque<Duration>,
}

impl TrailingWindow {
    fn new(capacity: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity >= 2,
            "progress forecast trailing window must contain at least two evaluations"
        );
        Ok(Self {
            capacity,
            samples: VecDeque::with_capacity(capacity),
        })
    }

    fn observe(&mut self, elapsed: Duration, remaining_evaluations: usize) -> Result<Forecast> {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(elapsed);
        if self.samples.len() < self.capacity {
            return Ok(Forecast::Stabilizing {
                samples: self.samples.len(),
                required: self.capacity,
            });
        }

        let total_nanos = self.samples.iter().try_fold(0u128, |total, sample| {
            total
                .checked_add(sample.as_nanos())
                .context("progress forecast duration sum overflow")
        })?;
        let mean_nanos = total_nanos
            / u128::try_from(self.capacity).context("progress forecast window exceeds u128")?;
        let remaining_nanos = mean_nanos
            .checked_mul(
                u128::try_from(remaining_evaluations)
                    .context("remaining evaluation count exceeds u128")?,
            )
            .context("progress forecast ETA overflow")?;
        Ok(Forecast::Ready {
            window: self.capacity,
            mean: duration_from_nanos(mean_nanos)?,
            remaining: duration_from_nanos(remaining_nanos)?,
        })
    }
}

#[derive(Debug)]
pub(super) struct DenoiseProgress {
    enabled: bool,
    mode: OutputMode,
    log_evaluation_interval: usize,
    forecast: TrailingWindow,
    prepared_evaluations: Option<usize>,
    observed_evaluations: usize,
    last_step_index: Option<usize>,
    total_steps: Option<usize>,
}

impl DenoiseProgress {
    pub(super) fn for_stderr(enabled: bool) -> Self {
        Self::new(
            enabled,
            if std::io::stderr().is_terminal() {
                OutputMode::Terminal
            } else {
                OutputMode::Log
            },
            DEFAULT_TRAILING_WINDOW,
            DEFAULT_LOG_EVALUATION_INTERVAL,
        )
        .expect("built-in progress settings are valid")
    }

    fn new(
        enabled: bool,
        mode: OutputMode,
        trailing_window: usize,
        log_evaluation_interval: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            log_evaluation_interval > 0,
            "progress log evaluation interval must be non-zero"
        );
        Ok(Self {
            enabled,
            mode,
            log_evaluation_interval,
            forecast: TrailingWindow::new(trailing_window)?,
            prepared_evaluations: None,
            observed_evaluations: 0,
            last_step_index: None,
            total_steps: None,
        })
    }

    pub(super) fn synchronize_device_timings(&self) -> bool {
        self.enabled
    }

    pub(super) fn on_preparation_completed(
        &mut self,
        event: DenoisePreparationEvent,
        output: &mut dyn Write,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            event.timing_synchronized,
            "progress forecasting requires synchronized preparation timing"
        );
        anyhow::ensure!(
            event.prepared_evaluations > 0,
            "progress preparation must cover at least one evaluation"
        );
        anyhow::ensure!(
            self.prepared_evaluations.is_none(),
            "progress preparation was reported more than once"
        );
        self.prepared_evaluations = Some(event.prepared_evaluations);
        self.write_line(
            output,
            &format!(
                "prepare: static context for {} evaluations completed in {:.2}s",
                event.prepared_evaluations,
                event.elapsed.as_secs_f64()
            ),
            false,
        )
    }

    pub(super) fn on_step_completed(
        &mut self,
        event: DenoiseStepEvent,
        output: &mut dyn Write,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            event.timing_synchronized,
            "progress forecasting requires synchronized evaluation timing"
        );
        let prepared_evaluations = self
            .prepared_evaluations
            .context("denoise progress arrived before preparation completed")?;
        anyhow::ensure!(
            event.total_steps > 0,
            "denoise progress total must be non-zero"
        );
        anyhow::ensure!(
            event.step_index < event.total_steps,
            "denoise progress step {} is outside {} evaluations",
            event.step_index + 1,
            event.total_steps
        );
        if let Some(total_steps) = self.total_steps {
            anyhow::ensure!(
                event.total_steps == total_steps,
                "denoise progress total changed from {total_steps} to {}",
                event.total_steps
            );
        } else {
            self.total_steps = Some(event.total_steps);
        }
        if let Some(last_step_index) = self.last_step_index {
            anyhow::ensure!(
                event.step_index == last_step_index + 1,
                "denoise progress is not contiguous: step {} followed step {}",
                event.step_index + 1,
                last_step_index + 1
            );
        }
        self.observed_evaluations = self
            .observed_evaluations
            .checked_add(1)
            .context("observed evaluation count overflow")?;
        anyhow::ensure!(
            self.observed_evaluations <= prepared_evaluations,
            "denoise progress exceeded the {prepared_evaluations} prepared evaluations"
        );
        self.last_step_index = Some(event.step_index);

        let remaining_evaluations = prepared_evaluations - self.observed_evaluations;
        let forecast = if self.observed_evaluations == 1 {
            Forecast::Stabilizing {
                samples: 0,
                required: self.forecast.capacity,
            }
        } else {
            self.forecast
                .observe(event.step_elapsed, remaining_evaluations)?
        };
        let invocation_complete = remaining_evaluations == 0;
        let should_emit = self.mode == OutputMode::Terminal
            || self.observed_evaluations == 1
            || self
                .observed_evaluations
                .is_multiple_of(self.log_evaluation_interval)
            || invocation_complete;
        if !should_emit {
            return Ok(());
        }

        let completed = event.step_index + 1;
        let line = match forecast {
            Forecast::Stabilizing { samples, required } => format!(
                "denoise {completed}/{}: {:.2}s/eval; forecast stabilizing ({samples}/{required})",
                event.total_steps,
                event.step_elapsed.as_secs_f64()
            ),
            Forecast::Ready {
                window,
                mean,
                remaining,
            } => format!(
                "denoise {completed}/{}: trailing {window}-eval mean {}; ETA {}",
                event.total_steps,
                format_duration(mean),
                format_duration(remaining)
            ),
        };
        self.write_line(output, &line, invocation_complete)
    }

    fn write_line(&self, output: &mut dyn Write, line: &str, finish: bool) -> Result<()> {
        match self.mode {
            OutputMode::Terminal => {
                write!(output, "\r\x1b[2K{line}")?;
                if finish {
                    writeln!(output)?;
                }
            }
            OutputMode::Log => writeln!(output, "{line}")?,
        }
        output.flush().context("failed to flush progress output")?;
        Ok(())
    }
}

fn duration_from_nanos(nanos: u128) -> Result<Duration> {
    const NANOS_PER_SECOND: u128 = 1_000_000_000;
    let seconds = u64::try_from(nanos / NANOS_PER_SECOND)
        .context("progress forecast duration exceeds u64 seconds")?;
    let subsec_nanos = u32::try_from(nanos % NANOS_PER_SECOND)
        .context("progress forecast subsecond duration exceeds u32")?;
    Ok(Duration::new(seconds, subsec_nanos))
}

fn format_duration(duration: Duration) -> String {
    if duration < Duration::from_secs(60) {
        return format!("{:.1}s", duration.as_secs_f64());
    }
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = seconds % 3600 / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else {
        format!("{minutes}m {seconds:02}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preparation(evaluations: usize) -> DenoisePreparationEvent {
        DenoisePreparationEvent {
            prepared_evaluations: evaluations,
            elapsed: Duration::from_secs(2),
            timing_synchronized: true,
        }
    }

    fn step(index: usize, total: usize, seconds: u64) -> DenoiseStepEvent {
        DenoiseStepEvent {
            step_index: index,
            total_steps: total,
            video_timestep: 1.0,
            audio_timestep: 1.0,
            step_elapsed: Duration::from_secs(seconds),
            total_elapsed: Duration::from_secs(seconds),
            timing_synchronized: true,
        }
    }

    #[test]
    fn first_evaluation_never_produces_an_eta() {
        let mut progress = DenoiseProgress::new(true, OutputMode::Log, 5, 5).unwrap();
        let mut output = Vec::new();
        progress
            .on_preparation_completed(preparation(10), &mut output)
            .unwrap();
        output.clear();
        progress
            .on_step_completed(step(0, 10, 100), &mut output)
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("forecast stabilizing (0/5)"));
        assert!(!output.contains("ETA"));
    }

    #[test]
    fn eta_appears_only_after_the_fixed_window_and_uses_its_trailing_samples() {
        let mut progress = DenoiseProgress::new(true, OutputMode::Terminal, 5, 1).unwrap();
        let mut output = Vec::new();
        progress
            .on_preparation_completed(preparation(10), &mut output)
            .unwrap();
        output.clear();
        progress
            .on_step_completed(step(0, 10, 100), &mut output)
            .unwrap();
        for (index, seconds) in [10, 20, 30, 40].into_iter().enumerate() {
            progress
                .on_step_completed(step(index + 1, 10, seconds), &mut output)
                .unwrap();
        }
        let warming = String::from_utf8(output.clone()).unwrap();
        assert!(!warming.contains("ETA"));

        output.clear();
        progress
            .on_step_completed(step(5, 10, 50), &mut output)
            .unwrap();
        let first_eta = String::from_utf8(output.clone()).unwrap();
        assert!(first_eta.contains("trailing 5-eval mean 30.0s; ETA 2m 00s"));

        output.clear();
        progress
            .on_step_completed(step(6, 10, 60), &mut output)
            .unwrap();
        let trailing_eta = String::from_utf8(output).unwrap();
        assert!(trailing_eta.contains("trailing 5-eval mean 40.0s; ETA 2m 00s"));
    }

    #[test]
    fn terminal_mode_overwrites_one_line_and_finishes_it_once() {
        let mut progress = DenoiseProgress::new(true, OutputMode::Terminal, 2, 1).unwrap();
        let mut output = Vec::new();
        progress
            .on_preparation_completed(preparation(2), &mut output)
            .unwrap();
        progress
            .on_step_completed(step(0, 2, 1), &mut output)
            .unwrap();
        progress
            .on_step_completed(step(1, 2, 1), &mut output)
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output.matches("\r\x1b[2K").count(), 3);
        assert_eq!(output.matches('\n').count(), 1);
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn log_mode_emits_phase_first_interval_and_final_lines_only() {
        let mut progress = DenoiseProgress::new(true, OutputMode::Log, 5, 5).unwrap();
        let mut output = Vec::new();
        progress
            .on_preparation_completed(preparation(10), &mut output)
            .unwrap();
        for index in 0..10 {
            progress
                .on_step_completed(step(index, 10, 1), &mut output)
                .unwrap();
        }
        let output = String::from_utf8(output).unwrap();
        let lines = output.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("prepare:"));
        assert!(lines[1].starts_with("denoise 1/10:"));
        assert!(lines[2].starts_with("denoise 5/10:"));
        assert!(lines[3].starts_with("denoise 10/10:"));
    }

    #[test]
    fn partial_invocation_always_emits_and_terminates_its_last_step() {
        let mut progress = DenoiseProgress::new(true, OutputMode::Terminal, 2, 5).unwrap();
        let mut output = Vec::new();
        progress
            .on_preparation_completed(preparation(2), &mut output)
            .unwrap();
        output.clear();
        progress
            .on_step_completed(step(40, 49, 2), &mut output)
            .unwrap();
        progress
            .on_step_completed(step(41, 49, 2), &mut output)
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("denoise 42/49:"));
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn no_progress_requests_no_synchronization_and_writes_nothing() {
        let mut progress = DenoiseProgress::new(false, OutputMode::Terminal, 5, 5).unwrap();
        assert!(!progress.synchronize_device_timings());
        let mut output = Vec::new();
        let mut unsynchronized_preparation = preparation(1);
        unsynchronized_preparation.timing_synchronized = false;
        progress
            .on_preparation_completed(unsynchronized_preparation, &mut output)
            .unwrap();
        let mut unsynchronized_step = step(0, 1, 1);
        unsynchronized_step.timing_synchronized = false;
        progress
            .on_step_completed(unsynchronized_step, &mut output)
            .unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn enabled_progress_fails_fast_on_unsynchronized_or_noncontiguous_events() {
        let mut progress = DenoiseProgress::new(true, OutputMode::Log, 2, 1).unwrap();
        let mut output = Vec::new();
        let mut unsynchronized = preparation(2);
        unsynchronized.timing_synchronized = false;
        assert!(
            progress
                .on_preparation_completed(unsynchronized, &mut output)
                .unwrap_err()
                .to_string()
                .contains("requires synchronized")
        );

        let mut progress = DenoiseProgress::new(true, OutputMode::Log, 2, 1).unwrap();
        progress
            .on_preparation_completed(preparation(2), &mut output)
            .unwrap();
        progress
            .on_step_completed(step(4, 10, 1), &mut output)
            .unwrap();
        assert!(
            progress
                .on_step_completed(step(6, 10, 1), &mut output)
                .unwrap_err()
                .to_string()
                .contains("not contiguous")
        );
    }
}
