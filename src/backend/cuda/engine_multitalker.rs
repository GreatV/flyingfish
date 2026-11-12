//! Multitalker parakeet streaming 0.6b engine: encoder, layer-0 speaker kernels
//! and the RNNT prediction/joint weights, loaded in place from the `.nemo` archive.

use crate::{audio::LogMel, checkpoint::MappedCheckpoint, multitalker::MultitalkerSpec};
use anyhow::{Result, ensure};
use std::{path::Path, sync::Arc};

use super::Device;
use super::engine_parakeet::{Linear, MultitalkerStep, ParakeetEncoder, linear};
use crate::backend::gpu::CudaStream;

/// Layer-0 speaker kernel: Linear(d, d) -> ReLU -> Linear(d, d), applied to the
/// mask-multiplied layer-0 input with a residual add, per speaker lane.
struct SpeakerFf {
    up: Linear,
    down: Linear,
}

impl SpeakerFf {
    fn load(
        checkpoint: &MappedCheckpoint,
        stream: &Arc<CudaStream>,
        prefix: &str,
        hidden: u32,
    ) -> Result<Self> {
        Ok(Self {
            up: linear(
                checkpoint,
                stream,
                &format!("{prefix}.0"),
                hidden,
                hidden,
                true,
            )?,
            down: linear(
                checkpoint,
                stream,
                &format!("{prefix}.3"),
                hidden,
                hidden,
                true,
            )?,
        })
    }
}

pub struct Multitalker {
    pub encoder: ParakeetEncoder,
    pub frontend: LogMel,
    pub spec: MultitalkerSpec,
    speaker: SpeakerFf,
    background: SpeakerFf,
}

impl Multitalker {
    pub fn load(nemo: &Path, device: &Device) -> Result<Self> {
        let spec = MultitalkerSpec::read(nemo)?;
        spec.validate()?;
        let encoder = ParakeetEncoder::load_multitalker(nemo, device)?;
        let checkpoint = MappedCheckpoint::open_nemo(nemo)?;
        let window = checkpoint.f32_tensor("preprocessor.featurizer.window", &[400])?;
        let fb = checkpoint.f32_tensor("preprocessor.featurizer.fb", &[1, 128, 257])?;
        let frontend = LogMel::with_filterbank(spec.mel(), window, fb)?;
        let stream = Arc::clone(&encoder.stream);
        let hidden = spec.d_model;
        ensure!(
            spec.spk_kernel_layers == [0],
            "multitalker loader requires the published layer-0 speaker kernels"
        );
        let speaker = SpeakerFf::load(&checkpoint, &stream, "spk_kernels.0", hidden)?;
        let background = SpeakerFf::load(&checkpoint, &stream, "bg_spk_kernels.0", hidden)?;

        stream.synchronize()?;
        Ok(Self {
            encoder,
            frontend,
            spec,
            speaker,
            background,
        })
    }

    /// Mel features `[frames, 128]` matching the published preprocessor.
    pub fn features(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let batch = self.frontend.extract(&[waveform], false, true, false)?;
        let mel_bins = self.spec.features as usize;
        Ok(batch.features[..batch.frames * mel_bins].to_vec())
    }

    /// Blank index: the vocabulary of 1024 fills logits 0..=1023, blank is 1024.
    pub fn blank(&self) -> u32 {
        self.spec.vocab_size
    }

    /// Starts a streaming session over the whole clip: `features` is the mel
    /// output of [`Self::features`], `diar` the `[frames/…, lanes]` speaker
    /// masks at the 80 ms frame grid (14 new frames per step), `masked_asr`
    /// selects the card-default spectral masking (kernel mode passes the masks
    /// through the layer-0 kernels instead).
    pub fn stream(
        &self,
        features: Vec<f32>,
        diar: Vec<f32>,
        masked_asr: bool,
        lanes: u32,
    ) -> Result<MultitalkerSession> {
        let width = self.spec.features as usize;
        ensure!(
            features.len().is_multiple_of(width)
                && (1..=4).contains(&lanes)
                && diar.len().is_multiple_of(lanes as usize),
            "multitalker session inputs"
        );
        let total_frames = features.len() / width;
        Ok(MultitalkerSession {
            features,
            total_frames,
            diar,
            masked_asr,
            lanes,
            buffer_idx: 0,
            step: 0,
        })
    }

    /// Runs one streaming step; `None` once the clip is consumed.
    pub fn stream_step(
        &self,
        session: &mut MultitalkerSession,
        stream: &mut super::engine_parakeet::MultitalkerStream,
    ) -> Result<Option<MultitalkerStep>> {
        let width = self.spec.features as usize;
        let lanes = session.lanes as usize;
        let (window_start, new_frames, cache_frames, keep_all) = if session.step == 0 {
            (0usize, session.total_frames.min(105), 0usize, false)
        } else if session.buffer_idx >= session.total_frames {
            return Ok(None);
        } else {
            let cache = 9usize.min(session.buffer_idx);
            let new = (session.total_frames - session.buffer_idx).min(112);
            (session.buffer_idx - cache, new, cache, new < 112)
        };
        let window_len = cache_frames + new_frames;
        let window_end = window_start + window_len;
        let mask_frames = 14usize;
        // The reference slices the bg mask to the pre-encode row count
        // (consecutive diar frames from the step start, no padding) while spk
        // keeps the step's 14 frames end-aligned by the encoder.
        let step_rows = self
            .encoder
            .multitalker_subsample_rows(window_len)
            .map_err(|error| anyhow::anyhow!("multitalker step rows: {error}"))?;
        let mut spk = vec![1.0f32; lanes * mask_frames];
        let mut bg = vec![0.0f32; lanes * step_rows];
        for lane in 0..lanes {
            for frame in 0..mask_frames {
                let index = lane + (session.step * 14 + frame) * lanes;
                if index < session.diar.len() {
                    spk[lane * mask_frames + frame] = session.diar[index];
                }
            }
            for frame in 0..step_rows {
                let others = (0..lanes).filter(|other| *other != lane).any(|other| {
                    let other_index = other + (session.step * 14 + frame) * lanes;
                    other_index < session.diar.len() && session.diar[other_index] > 0.5
                });
                bg[lane * step_rows + frame] = u8::from(others) as f32;
            }
        }
        let mut windows = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let mut window = vec![0.0f32; window_len * width];
            window.copy_from_slice(&session.features[window_start * width..window_end * width]);
            if session.masked_asr {
                // mask_features: the 8x-repeated diar mask left-pads to the
                // window (end-aligned) when shorter, or trims from the front of
                // the mask when longer; original-zero features take the floor.
                let mask_mel = mask_frames * 8;
                for frame in 0..window_len {
                    // mask_features trims the mask from the front when it is
                    // wider than the window, and left-pads zeros when narrower.
                    let mask = if mask_mel >= window_len {
                        let slot = frame / 8;
                        u8::from(spk[lane * mask_frames + slot] > 0.5) as f32
                    } else if frame < window_len - mask_mel {
                        0.0
                    } else {
                        let slot = (frame - (window_len - mask_mel)) / 8;
                        u8::from(spk[lane * mask_frames + slot] > 0.5) as f32
                    };
                    for bin in 0..width {
                        let value = session.features[(window_start + frame) * width + bin];
                        window[frame * width + bin] =
                            if value == 0.0 { -16.6355 } else { value * mask };
                    }
                }
            }
            windows.push(window);
        }
        if session.masked_asr {
            // Degenerate layer-0 targets of the published masked path.
            spk.fill(1.0);
            bg.fill(0.0);
        }
        session.buffer_idx += if session.step == 0 { 105 } else { 112 };
        session.step += 1;
        let masks = super::engine_parakeet::MultitalkerMasks { spk: &spk, bg: &bg };
        let kernels = super::engine_parakeet::MultitalkerKernels {
            speaker_up: &self.speaker.up,
            speaker_down: &self.speaker.down,
            background_up: &self.background.up,
            background_down: &self.background.down,
        };
        let step = self
            .encoder
            .multitalker_encode_step(stream, &windows, &masks, &kernels, keep_all)?;
        Ok(Some(step))
    }
}

/// Host-side streaming chunk state over one clip's features.
pub struct MultitalkerSession {
    features: Vec<f32>,
    total_frames: usize,
    diar: Vec<f32>,
    masked_asr: bool,
    lanes: u32,
    buffer_idx: usize,
    step: usize,
}

impl MultitalkerSession {
    pub fn steps_taken(&self) -> usize {
        self.step
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::backend::cuda::engine_eou::tests::read_wave_public as read_wave;
    use anyhow::Context;
    use serde_json::Value;

    const HOP: f64 = 0.08;
    const CROP0: f64 = 740.0;

    pub(super) fn synth_signal() -> Vec<f32> {
        let n = 16_000 * 3;
        (0..n)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let x = 0.3 * (2.0 * std::f64::consts::PI * 173.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 613.0 * t).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 101.0 * t).sin();
                let h = ((i as i64).wrapping_mul(2_654_435_761)).rem_euclid(1_000_003) as f64
                    / 1_000_003.0
                    - 0.5;
                let value = (x + 0.05 * h) as f32;
                // The reference arms consume this signal through a PCM_16 wav
                // (libsndfile floor(x * 32768) clamped to [-32768, 32767], then
                // / 32768); the engine must consume the same samples.
                (value * 32768.0).floor().clamp(-32768.0, 32767.0) / 32768.0
            })
            .collect()
    }

    fn synth_masks(frames: usize) -> Vec<f32> {
        let freq = [7.0, 11.0, 13.0, 17.0];
        let period = [97.0, 89.0, 83.0, 79.0];
        let mut masks = vec![0.0f32; frames * 4];
        for lane in 0..4 {
            for frame in 0..frames {
                masks[lane + frame * 4] = (0.5
                    + 0.5
                        * (2.0 * std::f64::consts::PI * freq[lane] * frame as f64 / period[lane])
                            .sin()) as f32;
            }
        }
        masks
    }

    fn word_spans(path: &std::path::Path) -> Result<Vec<(f64, f64)>> {
        let text = std::fs::read_to_string(path)?;
        let entries: Value = serde_json::from_str(&text)?;
        let mut spans = Vec::new();
        for entry in entries.as_array().context("word entries")? {
            let pair = entry.as_array().context("word pair")?;
            let (start, end) = (
                pair[0].as_f64().context("start")?,
                pair[1].as_f64().context("end")?,
            );
            if end > start {
                spans.push((start - CROP0, end - CROP0));
            }
        }
        Ok(spans)
    }

    fn gt_masks(path: &std::path::Path, frames: usize) -> Result<Vec<f32>> {
        let a = word_spans(&path.join("EN2002c.A.excerpt-words.json"))?;
        let c = word_spans(&path.join("EN2002c.C.excerpt-words.json"))?;
        let mut masks = vec![0.0f32; frames * 4];
        for (lane, spans) in [a, c].into_iter().enumerate() {
            for (start, end) in spans {
                let first = (start / HOP).floor() as i64;
                let last = (end / HOP).ceil() as i64;
                for frame in first.max(0)..last.min(frames as i64) {
                    let frame = frame as usize;
                    if start < (frame + 1) as f64 * HOP && end > frame as f64 * HOP {
                        masks[lane + frame * 4] = 1.0;
                    }
                }
            }
        }
        Ok(masks)
    }

    fn rotate_masks(orig: &[f32], frames: usize) -> Vec<f32> {
        let mut rotated = vec![0.0f32; frames * 4];
        for frame in 0..frames {
            let o = (13.29 + frame as f64 * HOP) % 120.0;
            let f1 = (o / HOP) as usize;
            let f2 = (((o + HOP) % 120.0) / HOP) as usize;
            for lane in 0..4 {
                let value = if f1 == f2 {
                    orig[lane + f1 * 4]
                } else {
                    orig[lane + f1 * 4].max(orig[lane + f2 * 4])
                };
                rotated[lane + frame * 4] = value;
            }
        }
        rotated
    }

    struct Inputs {
        name: &'static str,
        features: Vec<f32>,
        diar: Vec<f32>,
        lanes: u32,
    }

    /// The synth masks ride with the frozen fixture: they sit exactly at the
    /// 0.5 binarization threshold at their zero crossings, where libm ulp
    /// differences would flip frames between languages.
    fn inputs(model: &Multitalker, root: &std::path::Path, cells: &Value) -> Result<Vec<Inputs>> {
        let mut out = Vec::new();
        let features = model.features(&synth_signal())?;
        let mask_values: Vec<f32> = cells["synth3/masks"]
            .as_array()
            .context("synth3 masks missing from the frozen fixture")?
            .iter()
            .map(|value| Ok(value.as_f64().context("synth mask value")? as f32))
            .collect::<Result<_>>()?;
        out.push(Inputs {
            name: "synth3",
            features,
            diar: mask_values,
            lanes: 4,
        });
        for (name, wav, rot) in [
            ("meeting", root.join("meeting.wav"), false),
            ("arrival", root.join("arrival-swap.wav"), true),
        ] {
            let wave = read_wave(&wav)?;
            let features = model.features(&wave)?;
            let frames = features.len() / 128 + 28;
            let base = gt_masks(root, frames)?;
            let diar = if rot {
                rotate_masks(&base, frames)
            } else {
                base
            };
            // The reference dumps meeting/arrival with the full 4-column GT
            // diar (A, C, silent, silent); the session runs all four lanes.
            out.push(Inputs {
                name,
                features,
                diar,
                lanes: 4,
            });
        }
        Ok(out)
    }

    /// Flat index and lane/row of the largest difference between two `[lanes, rows, hidden]` slices.
    /// Flat index of the largest difference between two slices.
    fn argmax_flat(a: &[f32], b: &[f32]) -> usize {
        let mut flat = 0usize;
        let mut worst = 0.0f32;
        for (index, (x, y)) in a.iter().zip(b).enumerate() {
            let diff = (x - y).abs();
            if diff > worst {
                worst = diff;
                flat = index;
            }
        }
        flat
    }

    /// NaN on either side is an infinite difference (f32::max ignores NaN).
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                if x.is_nan() || y.is_nan() {
                    f32::INFINITY
                } else {
                    (x - y).abs()
                }
            })
            .fold(0.0f32, f32::max)
    }

    /// Transposed `[lanes, hidden, rows]` view of a `[lanes, rows, hidden]` slice.
    fn transposed(values: &[f32], lanes: usize, rows: usize, hidden: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; values.len()];
        for lane in 0..lanes {
            for row in 0..rows {
                for (bin, value) in values
                    [lane * rows * hidden + row * hidden..lane * rows * hidden + (row + 1) * hidden]
                    .iter()
                    .enumerate()
                {
                    out[lane * hidden * rows + bin * rows + row] = *value;
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "GPU + published archive + frozen cells; set FF_MULTITALKER_NEMO, FF_N6E_FROZEN, FF_N6E_INPUTS"]
    fn multitalker_cells_match_the_frozen_reference() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let frozen = std::env::var("FF_N6E_FROZEN").context("FF_N6E_FROZEN")?;
        let inputs_root = std::env::var("FF_N6E_INPUTS").context("FF_N6E_INPUTS")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let cells: Value = serde_json::from_str(&std::fs::read_to_string(
            Path::new(&frozen).join("frozen-cells-4090a1-fp32-kfix.json"),
        )?)?;
        let taus: Value = serde_json::from_str(&std::fs::read_to_string(
            // N6e-ref-taus3.json: fp32-basis taus for the -kfix cells.
            Path::new(&frozen).join("N6e-ref-taus3.json"),
        )?)?;
        let mel: Value = serde_json::from_str(&std::fs::read_to_string(
            Path::new(&frozen).join("reference-mel-synth3.json"),
        )?)?;
        let reference: Vec<f32> = mel["values"]
            .as_array()
            .context("reference mel values")?
            .iter()
            .map(|value| Ok(value.as_f64().context("mel value")? as f32))
            .collect::<Result<_>>()?;
        let ours = model.features(&synth_signal())?;
        let frames = ours.len() / 128;
        // The fixture stores the reference mel bin-major `[128, frames]`.
        let ref_frames = reference.len() / 128;
        let per_frame: Vec<f32> = (0..frames.min(ref_frames))
            .map(|frame| {
                ours[frame * 128..(frame + 1) * 128]
                    .iter()
                    .enumerate()
                    .map(|(bin, x)| (x - reference[bin * ref_frames + frame]).abs())
                    .fold(0.0f32, f32::max)
            })
            .collect();
        println!(
            "MEL frames ours={frames} ref={ref_frames} first16_max={:?}",
            &per_frame[..16.min(per_frame.len())]
        );
        let inputs = inputs(&model, Path::new(&inputs_root), &cells)?;
        let mut checked = 0;
        for (masked_asr, mode) in [(true, "masked"), (false, "kernel")] {
            for input in &inputs {
                let mut session = model.stream(
                    input.features.clone(),
                    input.diar.clone(),
                    masked_asr,
                    input.lanes,
                )?;
                let mut stream = model.encoder.multitalker_stream(input.lanes)?;
                let mut step = 0;
                while let Some(result) = model.stream_step(&mut session, &mut stream)? {
                    let key = format!("{mode}/{}/step{step}", input.name);
                    if let Some(cell) = cells.get(&key) {
                        let diff = max_abs(&result.layer0_in, &cell_values(cell, "x0_in"));
                        let tau = tau_of(&taus, &key, "x0_in");
                        let cell_x0 = cell_values(cell, "x0_in");
                        let flat = argmax_flat(&result.layer0_in, &cell_x0);
                        let rows = result.layer0_in.len() / (1024 * input.lanes as usize);
                        let per_row = (0..rows)
                            .map(|row| {
                                (0..input.lanes as usize)
                                    .map(|lane| {
                                        let base = (lane * rows + row) * 1024;
                                        result.layer0_in[base..base + 1024]
                                            .iter()
                                            .zip(&cell_x0[base..base + 1024])
                                            .map(|(x, y)| (x - y).abs())
                                            .fold(0.0f32, f32::max)
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>();
                        ensure!(
                            diff <= tau,
                            "{key} x0_in diff {diff:.3e} > tau {tau:.3e} at flat {flat} ours={} cell={} per_row_max={per_row:?}",
                            result.layer0_in[flat],
                            cell_x0[flat]
                        );
                        let diff = max_abs(&result.layer0_post, &cell_values(cell, "x0_post"));
                        let tau = tau_of(&taus, &key, "x0_post");
                        ensure!(diff <= tau, "{key} x0_post diff {diff:.3e} > tau {tau:.3e}");
                        let out_rows = result.rows.len() / (input.lanes as usize * 1024);
                        // Mid-stream steps trim the encoder rows to the valid
                        // 14 (keep_all only on the final step), so the returned
                        // row count is checked against the cell's enc rows.
                        let cell_rows =
                            cell_values(cell, "enc").len() / (input.lanes as usize * 1024);
                        ensure!(
                            out_rows == cell_rows,
                            "{key} rows {out_rows} vs cell enc rows {cell_rows}"
                        );
                        let enc = transposed(&result.rows, input.lanes as usize, out_rows, 1024);
                        let diff = max_abs(&enc, &cell_values(cell, "enc"));
                        let tau = tau_of(&taus, &key, "enc");
                        ensure!(diff <= tau, "{key} enc diff {diff:.3e} > tau {tau:.3e}");
                        println!("CELL {key} x0_in={diff:.3e} within tau");
                        checked += 3;
                    }
                    step += 1;
                }
                let expected = if input.name == "synth3" { 3 } else { 108 };
                ensure!(
                    step == expected,
                    "{mode}/{} steps {step} want {expected}",
                    input.name
                );
            }
        }
        ensure!(checked >= 22 * 3, "cells checked {checked}");
        println!("PASS multitalker_cells checked={checked} tau_bound=true");

        // Control: planting each speaker's diar mask on the other's lane must
        // move the meeting cells beyond 10x tau on x0_in and enc; a wrong lane
        // or speaker index reads exactly this.
        let wave = read_wave(&Path::new(&inputs_root).join("meeting.wav"))?;
        let features = model.features(&wave)?;
        let frames = features.len() / 128 + 28;
        let mut swapped = gt_masks(Path::new(&inputs_root), frames)?;
        for frame in 0..frames {
            swapped.swap(frame * 4, frame * 4 + 1);
        }
        let mut session = model.stream(features, swapped, true, 4)?;
        let mut stream = model.encoder.multitalker_stream(4)?;
        let step = model
            .stream_step(&mut session, &mut stream)?
            .context("control step")?;
        let cell = cells.get("masked/meeting/step0").context("control cell")?;
        let x0 = max_abs(&step.layer0_in, &cell_values(cell, "x0_in"));
        let tau_x0 = tau_of(&taus, "masked/meeting/step0", "x0_in");
        ensure!(
            x0 > 10.0 * tau_x0,
            "control x0_in {x0:.3e} did not clear 10x tau {tau_x0:.3e}"
        );
        let rows = step.rows.len() / (4 * 1024);
        let enc = transposed(&step.rows, 4, rows, 1024);
        let enc_diff = max_abs(&enc, &cell_values(cell, "enc"));
        let tau_enc = tau_of(&taus, "masked/meeting/step0", "enc");
        let cell_enc = cell_values(cell, "enc");
        let mut nan_stage = None;
        for (index, stage) in stream.trace.iter().enumerate() {
            if stage.iter().any(|value| value.is_nan()) {
                nan_stage = Some(index);
                break;
            }
        }
        let mut nan_block = None;
        if let Some(stage) = nan_stage {
            let k = stage.saturating_sub(1);
            for b in 0..3 {
                if let Some(_block) = stream
                    .block_trace
                    .get(k * 3 + b)
                    .filter(|block| block.iter().any(|value| value.is_nan()))
                {
                    nan_block = Some(b);
                    break;
                }
            }
        }
        println!(
            "CONTROL_PROBE rows={rows} enc_len={} cell_enc_len={} first_nan_stage={nan_stage:?} (0 = step input, k = after layer k-1) layer_substage={nan_block:?} (0 = after ff1, 1 = after attention, 2 = after conv) enc_head={:?} cell_head={:?}",
            enc.len(),
            cell_enc.len(),
            &enc[..4],
            &cell_enc[..4]
        );
        ensure!(
            enc_diff > 10.0 * tau_enc,
            "control enc {enc_diff:.3e} did not clear 10x tau {tau_enc:.3e}"
        );
        println!("PASS multitalker_cells_control swapped_speakers x0={x0:.3e} enc={enc_diff:.3e}");
        Ok(())
    }

    fn cell_values(cell: &Value, name: &str) -> Vec<f32> {
        cell.get(name)
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_f64().unwrap_or(f64::NAN) as f32)
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default()
    }

    fn tau_of(taus: &Value, key: &str, tensor: &str) -> f32 {
        taus["cell_taus"][format!("{key}/{tensor}")]["tau"]
            .as_f64()
            .unwrap_or(f64::INFINITY) as f32
    }

    #[test]
    #[ignore = "GPU + published archive; set FF_MULTITALKER_NEMO"]
    fn multitalker_lane_stack_is_batch_invariant() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let features = model.features(&synth_signal())?;
        let frames = features.len() / 128;
        let diar = synth_masks(frames + 28);
        // bg is the OR of the OTHER lanes' activity, so an invariance pair
        // must keep every lane's semantic inputs equal while varying only the
        // stack: alone vs stacked-with-silent (bg 0), 2-lane vs 4-lane with a
        // silent tail (bg 0), 2-lane vs 4-lane with the same content
        // everywhere (bg = the content's activity).
        let run = |content: usize, lanes: u32, silent_tail: bool| -> Result<Vec<Vec<f32>>> {
            let mut lane_masks = vec![0.0f32; frames * lanes as usize];
            for index in 0..lanes as usize {
                if silent_tail && index > 0 {
                    continue;
                }
                for frame in 0..frames {
                    lane_masks[index + frame * lanes as usize] = diar[content + frame * 4];
                }
            }
            let mut session = model.stream(features.clone(), lane_masks, false, lanes)?;
            let mut stream = model.encoder.multitalker_stream(lanes)?;
            let mut rows = Vec::new();
            while let Some(step) = model.stream_step(&mut session, &mut stream)? {
                rows.push(step.rows);
            }
            ensure!(
                session.steps_taken() == 3,
                "steps {}",
                session.steps_taken()
            );
            Ok(rows)
        };
        let four_silent = run(0, 4, true)?;
        let four_silent2 = run(2, 4, true)?;
        let four_content = run(0, 4, false)?;
        let alone0 = run(0, 1, false)?;
        let alone2 = run(2, 1, false)?;
        let two_silent = run(0, 2, true)?;
        let two_content = run(0, 2, false)?;
        for (alone, stacked, tag) in [
            (&alone0, &four_silent, "alone0_vs_silent4"),
            (&alone2, &four_silent2, "alone2_vs_silent4"),
            (&two_silent, &four_silent, "silent2_vs_silent4"),
            (&two_content, &four_content, "content2_vs_content4"),
        ] {
            for (step, (stacked_rows, alone_rows)) in stacked.iter().zip(alone).enumerate() {
                let hidden = 1024;
                let out_rows = alone_rows.len() / hidden;
                for row in 0..out_rows * hidden {
                    ensure!(
                        alone_rows[row].to_bits() == stacked_rows[row].to_bits(),
                        "pair {tag} step {step} row {row} differs"
                    );
                }
            }
        }
        println!("PASS multitalker_lane_invariant pairs=4 bit_identical=true");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    #[ignore = "published multitalker-parakeet-streaming-0.6b-v1 archive; set FF_MULTITALKER_NEMO"]
    fn multitalker_loads_every_published_tensor() -> Result<()> {
        let path = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&path), &device)?;
        ensure!(
            model.spec.n_layers == 24 && model.spec.vocab_size == 1024 && model.blank() == 1024,
            "multitalker loaded model shape"
        );
        println!("PASS multitalker_load blocks=24 vocab=1024 blank=1024 speaker_kernels=layer0");
        Ok(())
    }
}

#[cfg(test)]
mod trace_tests {
    use super::gate_tests::synth_signal;
    use super::*;
    use anyhow::Context;
    use serde_json::Value;

    /// Walks the per-layer outputs of synth3 step 0 against the reference layer
    /// cells and names the first layer that departs.
    #[test]
    #[ignore = "GPU + published archive + layer cells; set FF_MULTITALKER_NEMO, FF_N6E_FROZEN, FF_N6E_INPUTS"]
    fn multitalker_layer_bisection_names_the_first_diverging_layer() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let frozen = std::env::var("FF_N6E_FROZEN").context("FF_N6E_FROZEN")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let cells: Value = serde_json::from_str(&std::fs::read_to_string(
            Path::new(&frozen).join("frozen-cells-4090a1-fp32-kfix.json"),
        )?)?;
        let layers: Value = serde_json::from_str(&std::fs::read_to_string(
            Path::new(&frozen).join("layer-cells-cpu.json"),
        )?)?;
        let mask_values: Vec<f32> = cells["synth3/masks"]
            .as_array()
            .context("synth masks")?
            .iter()
            .map(|v| Ok(v.as_f64().context("mask")? as f32))
            .collect::<Result<_>>()?;
        let features = model.features(&synth_signal())?;
        let mut session = model.stream(features, mask_values, false, 4)?;
        let mut stream = model.encoder.multitalker_stream(4)?;
        let step = model
            .stream_step(&mut session, &mut stream)?
            .context("step")?;
        let reference = &layers["kernel/step0"]["layers"];
        let mut named = None;
        for index in 0..24 {
            let ours = &stream.trace[index + 1];
            let theirs: Vec<f32> = {
                let capture = reference[str::to_string(&index.to_string())]
                    .as_array()
                    .context("layer capture")?
                    .last()
                    .context("layer output")?;
                let mut flat = Vec::new();
                fn flatten(value: &Value, out: &mut Vec<f32>) {
                    match value {
                        Value::Array(items) => items.iter().for_each(|item| flatten(item, out)),
                        Value::Number(number) => {
                            out.push(number.as_f64().unwrap_or(f64::NAN) as f32)
                        }
                        _ => out.push(f32::NAN),
                    }
                }
                flatten(capture, &mut flat);
                flat
            };
            let diff = ours
                .iter()
                .zip(&theirs)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("LAYER {index:02} maxdiff {diff:.3e}");
            if diff > 1.0 && named.is_none() {
                named = Some((index, diff));
            }
        }
        let _ = step;
        match named {
            Some((index, diff)) => println!("FIRST_DIVERGING layer {index} diff {diff:.3e}"),
            None => println!("FIRST_DIVERGING none"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod inject_tests {
    use super::*;
    use anyhow::Context;
    use serde_json::Value;

    fn flatten(value: &Value, out: &mut Vec<f32>) {
        match value {
            Value::Array(items) => items.iter().for_each(|item| flatten(item, out)),
            Value::Number(number) => out.push(number.as_f64().unwrap_or(f64::NAN) as f32),
            _ => out.push(f32::NAN),
        }
    }

    /// Common-input per-layer bisection: inject the 4090 reference's captured
    /// layer input into our layer step (fresh step-0 session, no caches) and
    /// compare the output against the captured one.
    #[test]
    #[ignore = "GPU + published archive + 4090 layer cells; set FF_MULTITALKER_NEMO, FF_N6E_FROZEN"]
    fn multitalker_common_input_layers() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let frozen = std::env::var("FF_N6E_FROZEN").context("FF_N6E_FROZEN")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let cells: Value = serde_json::from_str(&std::fs::read_to_string(
            Path::new(&frozen).join("layer-cells-4090-fp32.json"),
        )?)?;
        let layers = cells["kernel/step0"]["layers"]
            .as_object()
            .context("layers")?;
        for index in 0..24 {
            let entry = layers
                .get(&index.to_string())
                .with_context(|| format!("layer {index} capture"))?;
            let mut input = Vec::new();
            flatten(entry.get("in").context("layer input")?, &mut input);
            let mut expected = Vec::new();
            flatten(entry.get("out").context("layer output")?, &mut expected);
            ensure!(
                input.len() == 4 * 14 * 1024 && expected.len() == input.len(),
                "layer {index} capture shape {} vs {}",
                input.len(),
                expected.len()
            );
            let mut session = model.encoder.multitalker_stream(4)?;
            let input_dev = model.encoder.stream.clone_htod(&input)?;
            let output = model
                .encoder
                .multitalker_inject_layer(index, &input_dev, &mut session)?;
            let ours = model.encoder.stream.clone_dtoh(&output)?;
            let diff = ours
                .iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("COMMON_INPUT layer {index:02} maxdiff {diff:.3e}");
            if index == 0 {
                let submodules = cells["kernel/step0"]["layer0_submodules"]
                    .as_object()
                    .context("submodules")?;
                for name in ["(0, 'feed_forward1')", "(0, 'self_attn')", "(0, 'conv')"] {
                    let Some(entry) = submodules.get(name) else {
                        continue;
                    };
                    let mut input = Vec::new();
                    flatten(entry.get("in").context("submodule in")?, &mut input);
                    let mut want = Vec::new();
                    flatten(entry.get("out").context("submodule out")?, &mut want);
                    let input_dev = model.encoder.stream.clone_htod(&input)?;
                    let mut probe_session = model.encoder.multitalker_stream(4)?;
                    let output = match name {
                        "(0, 'feed_forward1')" => {
                            model.encoder.multitalker_inject_ff1(0, &input_dev)?
                        }
                        "(0, 'self_attn')" => model.encoder.multitalker_inject_attn(
                            0,
                            &input_dev,
                            &mut probe_session,
                        )?,
                        _ => model.encoder.multitalker_inject_conv(
                            0,
                            &input_dev,
                            &mut probe_session,
                        )?,
                    };
                    let ours = model.encoder.stream.clone_dtoh(&output)?;
                    let (mut diff, mut scale) = (0.0f32, 0.0f32);
                    for (x, y) in ours.iter().zip(&want) {
                        diff = diff.max((x - y).abs());
                        scale = scale.max(y.abs());
                    }
                    println!("SUBMODULE {name} maxdiff {diff:.3e} scale {scale:.3e}");
                }
            }
        }
        Ok(())
    }
}
