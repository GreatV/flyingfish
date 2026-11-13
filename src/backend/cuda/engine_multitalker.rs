//! Multitalker parakeet streaming 0.6b engine: encoder, layer-0 speaker kernels
//! and the RNNT prediction/joint weights, loaded in place from the `.nemo` archive.

use crate::{audio::LogMel, checkpoint::MappedCheckpoint, multitalker::MultitalkerSpec};
use anyhow::{Context, Result, ensure};
use std::{path::Path, sync::Arc};

use super::Device;
use super::engine_parakeet::{Linear, MultitalkerStep, ParakeetEncoder, linear, tensor};
use super::ops::flat;
use crate::backend::gpu::{CudaSlice, CudaStream};
use cudarc::driver::PushKernelArg;

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

/// One prediction-network LSTM layer: gate weights `[4h, h]` with separate
/// input/recurrent biases, published as `dec_rnn.lstm.*_l{index}`.
struct RnntLayer {
    input: Linear,
    recurrent: Linear,
    bias_ih: CudaSlice<f32>,
    bias_hh: CudaSlice<f32>,
}

/// RNNT prediction and joint weights: embedding `[vocab+1, h]` (the blank row
/// doubles as SOS), a 2-layer LSTM, `joint.pred`/`joint.enc` projections and
/// the `joint.joint_net.2` head over `relu(enc_proj + pred_proj)`.
struct MultitalkerRnnt {
    embedding: CudaSlice<f32>,
    layers: Vec<RnntLayer>,
    pred_projection: Linear,
    enc_projection: Linear,
    joint_head: Linear,
}

impl MultitalkerRnnt {
    fn load(checkpoint: &MappedCheckpoint, stream: &Arc<CudaStream>, spec: &MultitalkerSpec) -> Result<Self> {
        let hidden = spec.pred_hidden as usize;
        let vocab_blank = spec.vocab_size as usize + 1;
        let gate = 4 * hidden;
        let embedding = tensor(
            checkpoint,
            stream,
            "decoder.prediction.embed.weight",
            &[vocab_blank, hidden],
        )?;
        let mut layers = Vec::with_capacity(spec.pred_rnn_layers as usize);
        for index in 0..spec.pred_rnn_layers as usize {
            // The published prednet stores raw nn.LSTM tensor names (no
            // `.weight` suffix), unlike the wrapped parakeet-TDT decoder.
            let gate_linear = |suffix: &str| -> Result<Linear> {
                Ok(Linear {
                    weight: tensor(
                        checkpoint,
                        stream,
                        &format!("decoder.prediction.dec_rnn.lstm.{suffix}_l{index}"),
                        &[gate, hidden],
                    )?,
                    bias: None,
                    output: gate as u32,
                    input: spec.pred_hidden,
                })
            };
            layers.push(RnntLayer {
                input: gate_linear("weight_ih")?,
                recurrent: gate_linear("weight_hh")?,
                bias_ih: tensor(
                    checkpoint,
                    stream,
                    &format!("decoder.prediction.dec_rnn.lstm.bias_ih_l{index}"),
                    &[gate],
                )?,
                bias_hh: tensor(
                    checkpoint,
                    stream,
                    &format!("decoder.prediction.dec_rnn.lstm.bias_hh_l{index}"),
                    &[gate],
                )?,
            });
        }
        let pred_projection = linear(
            checkpoint,
            stream,
            "joint.pred",
            spec.pred_hidden,
            spec.pred_hidden,
            true,
        )?;
        let enc_projection = linear(
            checkpoint,
            stream,
            "joint.enc",
            spec.joint_hidden,
            spec.d_model,
            true,
        )?;
        let joint_head = linear(
            checkpoint,
            stream,
            "joint.joint_net.2",
            vocab_blank as u32,
            spec.joint_hidden,
            true,
        )?;
        Ok(Self {
            embedding,
            layers,
            pred_projection,
            enc_projection,
            joint_head,
        })
    }
}

/// Per-session decode state: prediction hidden/cell and gate scratch at the
/// fixed 4-slot lane stride, the carried projected prediction output, and
/// joint scratch. Device buffers only; hypotheses live on the host session.
pub struct MultitalkerDecodeStream {
    embed_input: CudaSlice<f32>,
    layers: Vec<LayerState>,
    prediction: CudaSlice<f32>,
    joint_frame: CudaSlice<f32>,
    joint_sum: CudaSlice<f32>,
    joint_activated: CudaSlice<f32>,
    joint_logits: CudaSlice<f32>,
}

impl MultitalkerDecodeStream {}

/// One speaker lane's decode carry: the greedy hypothesis, the per-decision
/// joint margins (top1 - top2 over the 1025 raw logits) and the global row
/// ordinal of every emitted token.
pub struct MultitalkerHypothesis {
    pub tokens: Vec<u32>,
    /// Per-decision top1-top2 margin and whether the decision emitted.
    pub margins: Vec<f32>,
    pub emitted: Vec<bool>,
    pub timestamps: Vec<u32>,
    initialized: bool,
}

struct LayerState {
    hidden: CudaSlice<f32>,
    cell: CudaSlice<f32>,
    input_gates: CudaSlice<f32>,
    recurrent_gates: CudaSlice<f32>,
}

pub struct Multitalker {
    pub encoder: ParakeetEncoder,
    pub frontend: LogMel,
    pub spec: MultitalkerSpec,
    speaker: SpeakerFf,
    background: SpeakerFf,
    rnnt: MultitalkerRnnt,
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
        let rnnt = MultitalkerRnnt::load(&checkpoint, &stream, &spec)?;

        stream.synchronize()?;
        Ok(Self {
            encoder,
            frontend,
            spec,
            speaker,
            background,
            rnnt,
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

    /// Allocates the per-session decode state at the fixed 4-slot lane stride.
    pub fn decode_stream(&self) -> Result<MultitalkerDecodeStream> {
        let stream = &self.encoder.stream;
        let hidden = self.spec.pred_hidden as usize;
        let gate = 4 * hidden;
        let logits = (self.spec.vocab_size as usize + 1) * 4;
        let mut layers = Vec::with_capacity(self.spec.pred_rnn_layers as usize);
        for _ in 0..self.spec.pred_rnn_layers {
            layers.push(LayerState {
                hidden: stream.alloc_zeros(4 * hidden)?,
                cell: stream.alloc_zeros(4 * hidden)?,
                input_gates: stream.alloc_zeros(4 * gate)?,
                recurrent_gates: stream.alloc_zeros(4 * gate)?,
            });
        }
        Ok(MultitalkerDecodeStream {
            embed_input: stream.alloc_zeros(4 * hidden)?,
            layers,
            prediction: stream.alloc_zeros(4 * hidden)?,
            joint_frame: stream.alloc_zeros(4 * hidden)?,
            joint_sum: stream.alloc_zeros(4 * hidden)?,
            joint_activated: stream.alloc_zeros(4 * hidden)?,
            joint_logits: stream.alloc_zeros(logits)?,
        })
    }

    /// Advances one lane's prediction network over `label` at the fixed
    /// 4-slot GEMM shapes: embedding row copy, per-layer gate GEMMs and the
    /// `joint.pred` projection into the carried prediction output. A blank
    /// after the lane's first (SOS) step is a no-op, matching the published
    /// loop's commit-on-emit state rule.
    fn predict_lane(
        &self,
        dstream: &mut MultitalkerDecodeStream,
        lane: usize,
        label: u32,
        initialized: &mut bool,
    ) -> Result<()> {
        let stream = &self.encoder.stream;
        let hidden = self.spec.pred_hidden as usize;
        ensure!(lane < 4, "prediction lane {lane} outside the 4-slot stride");
        if *initialized && label == self.blank() {
            return Ok(());
        }
        let start = label as usize * hidden;
        stream.memcpy_dtod(
            &self.rnnt.embedding.slice(start..start + hidden),
            &mut dstream
                .embed_input
                .slice_mut(lane * hidden..(lane + 1) * hidden),
        )?;
        for index in 0..self.rnnt.layers.len() {
            let (previous, rest) = dstream.layers.split_at_mut(index);
            let LayerState {
                hidden: state_hidden,
                cell: state_cell,
                input_gates,
                recurrent_gates,
            } = &mut rest[0];
            let input: &CudaSlice<f32> = if index == 0 {
                &dstream.embed_input
            } else {
                &previous[index - 1].hidden
            };
            let layer = &self.rnnt.layers[index];
            self.encoder
                .lane_linear_into(input, 4, 4, &layer.input, input_gates)?;
            // The recurrent input is the layer's own carried hidden state.
            self.encoder
                .lane_linear_into(state_hidden, 4, 4, &layer.recurrent, recurrent_gates)?;
            unsafe {
                stream
                    .launch_builder(&self.encoder.ops.lstm)
                    .arg(&input_gates.slice(lane * 4 * hidden..(lane + 1) * 4 * hidden))
                    .arg(&recurrent_gates.slice(lane * 4 * hidden..(lane + 1) * 4 * hidden))
                    .arg(&layer.bias_ih)
                    .arg(&layer.bias_hh)
                    .arg(&state_cell.slice(lane * hidden..(lane + 1) * hidden))
                    .arg(&mut state_hidden.slice_mut(lane * hidden..(lane + 1) * hidden))
                    .arg(&self.spec.pred_hidden)
                    .launch(flat(hidden))?;
            }
        }
        let last = dstream
            .layers
            .last()
            .context("prediction LSTM stack empty")?;
        self.encoder
            .lane_linear_into(&last.hidden, 4, 4, &self.rnnt.pred_projection, &mut dstream.prediction)?;
        *initialized = true;
        Ok(())
    }

    /// Projects the step's encoder rows `[lanes, rows, d_model]` into
    /// `[4, rows, joint_hidden]` at the shape-constant 4-lane GEMM.
    fn project_step_rows(&self, step: &MultitalkerStep, out_rows: usize) -> Result<CudaSlice<f32>> {
        let width = self.spec.d_model as usize;
        let lanes = 4;
        let mut input = vec![0.0f32; lanes * out_rows * width];
        let present = step.rows.len() / (out_rows * width);
        for lane in 0..present.min(lanes) {
            let source = lane * out_rows * width;
            input[source..source + out_rows * width]
                .copy_from_slice(&step.rows[source..source + out_rows * width]);
        }
        let device_in = self.encoder.stream.clone_htod(&input)?;
        self.encoder.linear(
            &device_in,
            u32::try_from(lanes * out_rows)?,
            &self.rnnt.enc_projection,
        )
    }

    /// One joint evaluation for every lane's current row: gathers the
    /// projected row into the lane slot, adds the carried prediction output,
    /// relu, head GEMM, and reads the `[4, vocab+1]` logits back.
    fn joint_rows(
        &self,
        dstream: &mut MultitalkerDecodeStream,
        projected: &CudaSlice<f32>,
        row_offsets: &[usize; 4],
    ) -> Result<Vec<f32>> {
        let stream = &self.encoder.stream;
        let hidden = self.spec.pred_hidden as usize;
        for lane in 0..4 {
            let start = row_offsets[lane];
            stream.memcpy_dtod(
                &projected.slice(start..start + hidden),
                &mut dstream
                    .joint_frame
                    .slice_mut(lane * hidden..(lane + 1) * hidden),
            )?;
        }
        stream.memcpy_dtod(&dstream.prediction, &mut dstream.joint_sum)?;
        self.encoder
            .residual(&mut dstream.joint_sum, &dstream.joint_frame, 1.0)?;
        self.encoder.activate_into(
            &dstream.joint_sum,
            4,
            self.spec.pred_hidden,
            0,
            &mut dstream.joint_activated,
        )?;
        self.encoder.lane_linear_into(
            &dstream.joint_activated,
            4,
            4,
            &self.rnnt.joint_head,
            &mut dstream.joint_logits,
        )?;
        Ok(stream.clone_dtoh(&dstream.joint_logits)?)
    }

    /// Greedy RNNT decode of the step `stream_step` just returned: per row,
    /// joint evaluations until blank with at most `max_symbols` emissions
    /// (forced advance, no trailing blank evaluation), the prediction state
    /// advancing only on emission. Updates the session's per-speaker
    /// hypotheses (tokens, margins, global-row timestamps).
    pub fn decode_step(
        &self,
        session: &mut MultitalkerSession,
        dstream: &mut MultitalkerDecodeStream,
        step: &MultitalkerStep,
    ) -> Result<Vec<usize>> {
        let hidden = self.spec.d_model as usize;
        let joint_hidden = self.spec.pred_hidden as usize;
        let vocab_blank = self.spec.vocab_size as usize + 1;
        let out_rows = step.rows.len() / (hidden * session.lanes as usize);
        ensure!(out_rows > 0, "decode step with zero rows");
        ensure!(session.step > 0, "decode_step before stream_step");
        let active = if session.gating {
            session.active_lanes(session.step - 1)
        } else {
            (0..session.lanes as usize).collect()
        };
        if active.is_empty() {
            return Ok(active);
        }
        let projected = self.project_step_rows(step, out_rows)?;
        let mut cursor = [0usize; 4];
        let mut symbols = [0u32; 4];
        let mut live = [false; 4];
        for &lane in &active {
            ensure!(
                lane * out_rows * hidden + out_rows * hidden <= step.rows.len(),
                "decode row extent for lane {lane}"
            );
            live[lane] = true;
            let hypothesis = &mut session.hypotheses[lane];
            if !hypothesis.initialized {
                let mut initialized = hypothesis.initialized;
                self.predict_lane(dstream, lane, self.blank(), &mut initialized)?;
                hypothesis.initialized = initialized;
            }
        }
        while live.iter().any(|value| *value) {
            let mut row_offsets = [0usize; 4];
            for lane in 0..4 {
                row_offsets[lane] = lane * out_rows * joint_hidden + cursor[lane] * joint_hidden;
            }
            let logits = self.joint_rows(dstream, &projected, &row_offsets)?;
            for &lane in &active {
                if !live[lane] {
                    continue;
                }
                let slice = &logits[lane * vocab_blank..(lane + 1) * vocab_blank];
                let (mut best, mut next) = (0usize, 1usize);
                if slice[1] > slice[0] {
                    best = 1;
                    next = 0;
                }
                for (index, value) in slice.iter().enumerate().skip(2) {
                    if *value > slice[best] {
                        next = best;
                        best = index;
                    } else if *value > slice[next] {
                        next = index;
                    }
                }
                let margin = slice[best] - slice[next];
                let label = best as u32;
                let hypothesis = &mut session.hypotheses[lane];
                hypothesis.margins.push(margin);
                hypothesis.emitted.push(label != self.blank());
                if label == self.blank() {
                    cursor[lane] += 1;
                    symbols[lane] = 0;
                } else {
                    hypothesis.tokens.push(label);
                    hypothesis
                        .timestamps
                        .push((session.decoded_rows[lane] + cursor[lane]) as u32);
                    symbols[lane] += 1;
                    let mut initialized = hypothesis.initialized;
                    self.predict_lane(dstream, lane, label, &mut initialized)?;
                    hypothesis.initialized = initialized;
                    if symbols[lane] >= self.spec.max_symbols {
                        cursor[lane] += 1;
                        symbols[lane] = 0;
                    }
                }
                if cursor[lane] >= out_rows {
                    live[lane] = false;
                }
            }
        }
        for &lane in &active {
            session.decoded_rows[lane] += out_rows;
        }
        Ok(active)
    }

    /// Teacher-forced joint logits for the frozen logit cell: replays
    /// `rows`/`emits`/`tokens` (the emitted token per decision, ignored on
    /// blank) on a fresh lane slot from the SOS priming, returning the raw
    /// logits per decision.
    pub fn decode_teacher_forced(
        &self,
        dstream: &mut MultitalkerDecodeStream,
        lane: usize,
        rows: &[&[f32]],
        emits: &[bool],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let width = self.spec.d_model as usize;
        let joint_hidden = self.spec.pred_hidden as usize;
        let vocab_blank = self.spec.vocab_size as usize + 1;
        ensure!(
            rows.len() == emits.len()
                && rows.len() == tokens.len()
                && rows.iter().all(|row| row.len() == width),
            "teacher-forced decision extents"
        );
        let mut initialized = false;
        self.predict_lane(dstream, lane, self.blank(), &mut initialized)?;
        let mut projected = self.encoder.stream.alloc_zeros(4 * joint_hidden)?;
        let mut out = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let mut input = vec![0.0f32; 4 * width];
            input[lane * width..(lane + 1) * width].copy_from_slice(row);
            let device_in = self.encoder.stream.clone_htod(&input)?;
            let projected_all = self.encoder.linear(&device_in, 4, &self.rnnt.enc_projection)?;
            self.encoder.stream.memcpy_dtod(
                &projected_all.slice(lane * joint_hidden..(lane + 1) * joint_hidden),
                &mut projected.slice_mut(lane * joint_hidden..(lane + 1) * joint_hidden),
            )?;
            let mut offsets = [0usize; 4];
            offsets[lane] = lane * joint_hidden;
            let logits = self.joint_rows(dstream, &projected, &offsets)?;
            out.push(logits[lane * vocab_blank..(lane + 1) * vocab_blank].to_vec());
            if emits[index] {
                self.predict_lane(dstream, lane, tokens[index], &mut initialized)?;
            }
        }
        Ok(out)
    }

    /// Starts a streaming session over the whole clip: `features` is the mel
    /// output of [`Self::features`], `diar` the `[frames/…, lanes]` speaker
    /// masks at the 80 ms frame grid (14 new frames per step), `masked_asr`
    /// selects the card-default spectral masking (kernel mode passes the masks
    /// through the layer-0 kernels instead).
    /// `gating` selects the published wrapper semantics: the SpeakerTaggedASR
    /// loop encodes only the cache-gated active lanes, freezing an inactive
    /// lane's encoder caches at its own row count; without gating every lane
    /// encodes every step (the raw cache-aware step the M1 cells captured).
    pub fn stream(
        &self,
        features: Vec<f32>,
        diar: Vec<f32>,
        masked_asr: bool,
        lanes: u32,
        gating: bool,
    ) -> Result<MultitalkerSession> {
        let width = self.spec.features as usize;
        ensure!(
            features.len().is_multiple_of(width)
                && (1..=4).contains(&lanes)
                && diar.len().is_multiple_of(lanes as usize),
            "multitalker session inputs"
        );
        let total_frames = features.len() / width;
        let lane_count = lanes as usize;
        Ok(MultitalkerSession {
            features,
            total_frames,
            diar,
            masked_asr,
            lanes,
            gating,
            buffer_idx: 0,
            step: 0,
            hypotheses: (0..lane_count)
                .map(|_| MultitalkerHypothesis {
                    tokens: Vec::new(),
                    margins: Vec::new(),
                    emitted: Vec::new(),
                    timestamps: Vec::new(),
                    initialized: false,
                })
                .collect(),
            decoded_rows: vec![0usize; lane_count],
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
        let active = if session.gating {
            session.active_lanes(session.step)
        } else {
            (0..session.lanes as usize).collect()
        };
        session.buffer_idx += if session.step == 0 { 105 } else { 112 };
        session.step += 1;
        let masks = super::engine_parakeet::MultitalkerMasks { spk: &spk, bg: &bg };
        let kernels = super::engine_parakeet::MultitalkerKernels {
            speaker_up: &self.speaker.up,
            speaker_down: &self.speaker.down,
            background_up: &self.background.up,
            background_down: &self.background.down,
        };
        let step = self.encoder.multitanker_encode_step(
            stream,
            &windows,
            &masks,
            &kernels,
            keep_all,
            &active,
            !session.masked_asr,
        )?;
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
    gating: bool,
    buffer_idx: usize,
    step: usize,
    /// Per-speaker greedy hypotheses; index is the speaker lane.
    pub hypotheses: Vec<MultitalkerHypothesis>,
    decoded_rows: Vec<usize>,
}

impl MultitalkerSession {
    pub fn steps_taken(&self) -> usize {
        self.step
    }

    /// Active speakers for `step_index` (0-based): the cache gating rule — a
    /// lane is active when its diar mask exceeds 0.5 anywhere in the last 28
    /// frames of the cumulative stream through that step.
    fn active_lanes(&self, step_index: usize) -> Vec<usize> {
        let lanes = self.lanes as usize;
        let end = ((step_index + 1) * 14).min(self.diar.len() / lanes);
        let start = end.saturating_sub(28);
        (0..lanes)
            .filter(|lane| (start..end).any(|frame| self.diar[lane + frame * lanes] > 0.5))
            .collect()
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
                    false,
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
        let mut session = model.stream(features, swapped, true, 4, false)?;
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

    fn feature_frames(model: &Multitalker, wave: &[f32]) -> usize {
        model.features(wave).map(|f| f.len() / 128 + 28).unwrap_or(0)
    }

    fn tau_of(taus: &Value, key: &str, tensor: &str) -> f32 {
        taus["cell_taus"][format!("{key}/{tensor}")]["tau"]
            .as_f64()
            .unwrap_or(f64::INFINITY) as f32
    }

    /// Percentile of an unsorted slice (nearest-rank).
    fn percentile(values: &[f32], fraction: usize) -> f32 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[(fraction * sorted.len() / 100).min(sorted.len() - 1)]
    }

    #[test]
    #[ignore = "diagnostic; set FF_MULTITALKER_NEMO, FF_N6E_INPUTS, FF_N6E_ROWS_OUT"]
    fn multitalker_arrival_rows_dump() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let inputs_root = std::env::var("FF_N6E_INPUTS").context("FF_N6E_INPUTS")?;
        let out_path = std::env::var("FF_N6E_ROWS_OUT").context("FF_N6E_ROWS_OUT")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let root = Path::new(&inputs_root);
        let clip = std::env::var("FF_N6E_ROWS_CLIP").unwrap_or_else(|_| "arrival".into());
        let (wave, diar) = if clip == "meeting" {
            let wave = read_wave(&root.join("meeting.wav"))?;
            let features = model.features(&wave)?;
            let frames = features.len() / 128 + 28;
            (wave, gt_masks(root, frames)?)
        } else {
            let wave = read_wave(&root.join("arrival-swap.wav"))?;
            let frames = feature_frames(&model, &wave);
            let base = gt_masks(root, frames)?;
            (wave, rotate_masks(&base, frames))
        };
        let features = model.features(&wave)?;
        let mode = std::env::var("FF_N6E_ROWS_MODE").unwrap_or_else(|_| "masked".into());
        let masked_asr = mode == "masked";
        let mut session = model.stream(features, diar, masked_asr, 4, true)?;
        let mut stream = model.encoder.multitalker_stream(4)?;
        let mut dstream = model.decode_stream()?;
        let mut steps = Vec::new();
        let mut cold_dump: Option<Value> = None;
        while let Some(step) = model.stream_step(&mut session, &mut stream)? {
            let active = model.decode_step(&mut session, &mut dstream, &step)?;
            if cold_dump.is_none() && active.contains(&0) && session.steps_taken() > 1 {
                let lanes = 4;
                let width = 1024usize;
                let rows = step.rows.len() / (width * lanes);
                let mut trace: Vec<Value> = Vec::new();
                for layer in &stream.trace {
                    trace.push(serde_json::json!(layer[0..rows * width]));
                }
                cold_dump = Some(serde_json::json!({
                    "step": session.steps_taken() - 1,
                    "layer0_in_len": step.layer0_in.len(),
                    "layer0_in_lane0": step.layer0_in[0..16 * width],
                    "rows_lane0": step.rows[0..rows * width],
                    "trace_lane0": trace,
                }));
            }
            let lanes = 4;
            let width = 1024usize;
            let rows = step.rows.len() / (width * lanes);
            let mut entry = serde_json::json!({
                "active": active,
                "rows": rows,
                "lane0": [],
            });
            for row in 0..rows {
                let base = 0 * rows * width + row * width;
                entry["lane0"]
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!(step.rows[base..base + width]));
            }
            steps.push(entry);
        }
        let dump = serde_json::json!({
            "steps": steps,
            "cold": cold_dump,
        });
        std::fs::write(&out_path, serde_json::to_string(&dump)?)?;
        println!("PASS rows_dump steps={}", steps.len());
        Ok(())
    }

    #[test]
    #[ignore = "GPU + published archive + published-loop fixture; set FF_MULTITALKER_NEMO, FF_N6E_INPUTS, FF_N6E_M2B"]
    fn multitalker_rnnt_streams_match_the_published_loop() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let inputs_root = std::env::var("FF_N6E_INPUTS").context("FF_N6E_INPUTS")?;
        let m2b = std::env::var("FF_N6E_M2B").context("FF_N6E_M2B")?;
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&m2b)?)?;
        let frozen = std::env::var("FF_N6E_FROZEN").context("FF_N6E_FROZEN")?;
        let cells: Value = serde_json::from_str(&std::fs::read_to_string(
            Path::new(&frozen).join("frozen-cells-4090a1-fp32-kfix.json"),
        )?)?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let all = inputs(&model, Path::new(&inputs_root), &cells)?;
        let clips: Vec<&Inputs> = all
            .iter()
            .filter(|input| input.name == "meeting" || input.name == "arrival")
            .collect();
        ensure!(clips.len() == 2, "published-loop fixture needs both AMI clips");
        let mut diverged: Vec<String> = Vec::new();
        for (masked_asr, mode) in [(true, "masked"), (false, "kernel")] {
            for input in &clips {
                let case = format!("{mode}/{}", input.name);
                let steps = fixture[&case]["stream"]["steps"]
                    .as_array()
                    .with_context(|| format!("{case} steps"))?;
                let mut session =
                    model.stream(input.features.clone(), input.diar.clone(), masked_asr, 4, true)?;
                let mut stream = model.encoder.multitalker_stream(4)?;
                let mut dstream = model.decode_stream()?;
                let mut lengths: Vec<Vec<usize>> = vec![Vec::new(); 4];
                let mut step = 0usize;
                while let Some(result) = model.stream_step(&mut session, &mut stream)? {
                    let active = model.decode_step(&mut session, &mut dstream, &result)?;
                    let expect: Vec<usize> = steps[step]["active"]
                        .as_array()
                        .with_context(|| format!("{case} step {step} active"))?
                        .iter()
                        .map(|value| value.as_u64().unwrap_or(u64::MAX) as usize)
                        .collect();
                    ensure!(
                        active == expect,
                        "{case} step {step} active {active:?} reference {expect:?}"
                    );
                    for lane in 0..4 {
                        lengths[lane].push(session.hypotheses[lane].tokens.len());
                    }
                    step += 1;
                }
                ensure!(step == steps.len(), "{case} steps {step} vs {}", steps.len());
                let finals = fixture[&case]["stream"]["final"]
                    .as_object()
                    .with_context(|| format!("{case} finals"))?;
                for (speaker, expect) in finals {
                    let lane = speaker.parse::<usize>()?;
                    let hyp = &session.hypotheses[lane];
                    let tokens: Vec<u32> = expect["tokens"]
                        .as_array()
                        .context("final tokens")?
                        .iter()
                        .map(|value| value.as_u64().unwrap_or(u64::MAX) as u32)
                        .collect();
                    let stamps: Vec<u32> = expect["timestamps"]
                        .as_array()
                        .context("final timestamps")?
                        .iter()
                        .map(|value| value.as_u64().unwrap_or(u64::MAX) as u32)
                        .collect();
                    let first_mismatch = hyp
                        .tokens
                        .iter()
                        .zip(&tokens)
                        .position(|(a, b)| a != b);
                    if hyp.tokens != tokens {
                        let at = first_mismatch.unwrap_or(hyp.tokens.len().min(tokens.len()));
                        let lo = at.saturating_sub(8);
                        let hi = (at + 8).min(hyp.tokens.len().min(tokens.len()));
                        let our_stamps = &hyp.timestamps[lo..hi];
                        let ref_stamps = &stamps[lo..hi];
                        // Decision index of the mismatching emission and the
                        // margins around it (blank decisions included).
                        let decision_at = hyp
                            .emitted
                            .iter()
                            .enumerate()
                            .filter(|(_, value)| **value)
                            .nth(at)
                            .map(|(index, _)| index);
                        let margins_at = decision_at.map(|index| {
                            let mlo = index.saturating_sub(4);
                            let mhi = (index + 4).min(hyp.margins.len());
                            &hyp.margins[mlo..mhi]
                        });
                        println!(
                            "DIVERGE {case} speaker {lane} at token {at}: ours {:?} reference {:?} stamps ours {:?} ref {:?} decision {:?} margins {:?}",
                            &hyp.tokens[lo..hi],
                            &tokens[lo..hi],
                            our_stamps,
                            ref_stamps,
                            decision_at,
                            margins_at
                        );
                    }
                    if hyp.tokens != tokens || hyp.timestamps != stamps {
                        println!(
                            "STREAM_DIVERGED {case} speaker {lane} ours {} reference {} first_mismatch {:?}",
                            hyp.tokens.len(),
                            tokens.len(),
                            first_mismatch
                        );
                        diverged.push(format!("{case}/speaker{lane}"));
                        continue;
                    }
                    ensure!(
                        hyp.margins.iter().all(|margin| margin.is_finite() && *margin > 0.0),
                        "{case} speaker {lane} non-positive joint margin"
                    );
                    println!(
                        "M2 {case} speaker {lane} tokens {} decisions {} margins min {:.4} p10 {:.4} median {:.4}",
                        tokens.len(),
                        hyp.margins.len(),
                        hyp.margins.iter().cloned().fold(f32::INFINITY, f32::min),
                        percentile(&hyp.margins, 10),
                        percentile(&hyp.margins, 50)
                    );
                }
                if input.name == "arrival" {
                    // Control: a slot-keyed hypothesis carry (the wrong index a
                    // lane swap produces) concatenates the tokens of every
                    // speaker that passed through one batch slot; the
                    // speaker-keyed streams must differ from all of those.
                    let final_tokens = |lane: usize| -> Vec<u32> {
                        session.hypotheses[lane].tokens.clone()
                    };
                    for slot in 0..2usize {
                        let mut keyed = Vec::new();
                        for (index, step_active) in steps.iter().enumerate() {
                            let occupants: Vec<usize> = step_active["active"]
                                .as_array()
                                .context("active")?
                                .iter()
                                .map(|value| value.as_u64().unwrap_or(u64::MAX) as usize)
                                .collect();
                            let Some(occupant) = occupants.get(slot).copied() else {
                                continue;
                            };
                            let start = if index == 0 {
                                0
                            } else {
                                lengths[occupant][index - 1]
                            };
                            keyed.extend_from_slice(
                                &final_tokens(occupant)[start..lengths[occupant][index]],
                            );
                        }
                        for lane in 0..2usize {
                            ensure!(
                                session.hypotheses[lane].tokens != keyed,
                                "{case} speaker {lane} equals the slot-{slot} keyed carry ({} tokens)",
                                keyed.len()
                            );
                        }
                    }
                }
            }
        }
        ensure!(
            diverged.is_empty(),
            "streams diverged: {diverged:?}"
        );
        println!("PASS multitalker_rnnt_streams cases=4 tokens=exact timestamps=exact");
        Ok(())
    }

    #[test]
    #[ignore = "GPU + published archive + fixture arms; set FF_MULTITALKER_NEMO, FF_N6E_M2B, FF_N6E_M2B_GPU"]
    fn multitalker_rnnt_teacher_forced_logits() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let cpu_path = std::env::var("FF_N6E_M2B").context("FF_N6E_M2B")?;
        let gpu_path = std::env::var("FF_N6E_M2B_GPU").context("FF_N6E_M2B_GPU")?;
        let cpu: Value = serde_json::from_str(&std::fs::read_to_string(&cpu_path)?)?;
        let gpu: Value = serde_json::from_str(&std::fs::read_to_string(&gpu_path)?)?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        // Frozen before the candidate run: tau = 2x the worst per-decision
        // max-abs difference between the cpu and 4090-fp32 reference arms
        // (r3 fixtures, raw joint logits): 2 x 2.44140625e-04 = 4.8828125e-04.
        // Literal precision corrected 2026-11-12: the first spelling
        // (4.882812e-04) parsed one float ulp below the intended bound and
        // tripped the arm-pair guard at exactly half the tau.
        const TAU: f32 = 4.8828125e-04;
        fn f32_values(value: &Value) -> Result<Vec<f32>> {
            value
                .as_array()
                .context("float array")?
                .iter()
                .map(|item| Ok(item.as_f64().context("float item")? as f32))
                .collect()
        }
        fn active_of(step: &Value) -> Result<Vec<usize>> {
            step["active"]
                .as_array()
                .context("active")?
                .iter()
                .map(|value| Ok(value.as_u64().context("active lane")? as usize))
                .collect()
        }
        let mut pair = 0.0f32;
        let mut worst = 0.0f32;
        let mut worst_where = String::new();
        for case in ["masked/meeting", "masked/arrival", "kernel/meeting", "kernel/arrival"] {
            let steps = cpu[case]["stream"]["steps"]
                .as_array()
                .with_context(|| format!("{case} steps"))?;
            // Speaker-keyed global rows, mirroring the fixture capture.
            let mut rows: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 4];
            let mut base = [0usize; 4];
            for step in steps {
                let active = active_of(step)?;
                let lens: Vec<usize> = step["encoded_len"]
                    .as_array()
                    .context("encoded_len")?
                    .iter()
                    .map(|value| value.as_u64().unwrap_or(u64::MAX) as usize)
                    .collect();
                let block = step.get("encoded_rows");
                for (slot, speaker) in active.iter().enumerate() {
                    let count = lens.get(slot).copied().unwrap_or(0);
                    if let Some(block) = block.and_then(Value::as_array) {
                        let slot_rows: Vec<Vec<f32>> = block[slot]
                            .as_array()
                            .context("slot rows")?
                            .iter()
                            .map(f32_values)
                            .collect::<Result<_>>()?;
                        ensure!(
                            slot_rows.len() >= count,
                            "{case} step rows {} < {count}",
                            slot_rows.len()
                        );
                        for offset in 0..count {
                            ensure!(
                                slot_rows[offset].len() == 1024,
                                "{case} step row width {}",
                                slot_rows[offset].len()
                            );
                            rows[*speaker].push(slot_rows[offset].clone());
                        }
                    }
                    base[*speaker] += count;
                }
            }
            let cells = cpu[case]["cells"]
                .as_object()
                .with_context(|| format!("{case} cells"))?;
            for (speaker, cell) in cells {
                let lane = speaker.parse::<usize>()?;
                let decisions = cell["decisions"].as_array().context("decisions")?;
                let gpu_decisions = gpu[case]["cells"][speaker]["decisions"]
                    .as_array()
                    .context("gpu decisions")?;
                ensure!(
                    decisions.len() == gpu_decisions.len() && !decisions.is_empty(),
                    "{case} speaker {speaker} decision counts"
                );
                let emits: Vec<bool> = decisions
                    .iter()
                    .map(|decision| decision["kind"].as_str() == Some("emit"))
                    .collect();
                let tokens: Vec<u32> = decisions
                    .iter()
                    .map(|decision| decision["token"].as_u64().unwrap_or(u64::MAX) as u32)
                    .collect();
                let reference: Vec<Vec<f32>> = decisions
                    .iter()
                    .map(|decision| f32_values(&decision["logits"]))
                    .collect::<Result<_>>()?;
                let gpu_logits: Vec<Vec<f32>> = gpu_decisions
                    .iter()
                    .map(|decision| f32_values(&decision["logits"]))
                    .collect::<Result<_>>()?;
                for (index, decision) in decisions.iter().enumerate() {
                    ensure!(
                        gpu_decisions[index]["row"].as_u64() == decision["row"].as_u64()
                            && gpu_decisions[index]["kind"] == decision["kind"]
                            && gpu_decisions[index]["token"] == decision["token"],
                        "{case} speaker {speaker} decision {index} arms disagree"
                    );
                    let arm_pair = reference[index]
                        .iter()
                        .zip(&gpu_logits[index])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    pair = pair.max(arm_pair);
                }
                let decision_rows: Vec<&[f32]> = decisions
                    .iter()
                    .map(|decision| {
                        let row = decision["row"].as_u64().unwrap_or(u64::MAX) as usize;
                        rows[lane][row].as_slice()
                    })
                    .collect();
                let mut dstream = model.decode_stream()?;
                let candidate =
                    model.decode_teacher_forced(&mut dstream, 0, &decision_rows, &emits, &tokens)?;
                ensure!(candidate.len() == reference.len(), "candidate decisions");
                for (index, logits) in candidate.iter().enumerate() {
                    let diff = logits
                        .iter()
                        .zip(&reference[index])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if diff > worst {
                        worst = diff;
                        worst_where = format!("{case} speaker {speaker} decision {index}");
                    }
                }
            }
        }
        ensure!(pair * 2.0 <= TAU, "arm pair {pair:.3e} exceeds half the frozen tau");
        ensure!(
            worst <= TAU,
            "teacher-forced logits diff {worst:.3e} > tau {TAU:.3e} at {worst_where}"
        );
        println!(
            "PASS multitalker_rnnt_logits arm_pair={pair:.3e} worst={worst:.3e} tau={TAU:.3e}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU + published archive; set FF_MULTITALKER_NEMO, FF_N6E_INPUTS"]
    fn multitalker_decode_is_lane_invariant() -> Result<()> {
        let nemo = std::env::var("FF_MULTITALKER_NEMO").context("FF_MULTITALKER_NEMO")?;
        let inputs_root = std::env::var("FF_N6E_INPUTS").context("FF_N6E_INPUTS")?;
        let device = Device::new(0)?;
        let model = Multitalker::load(Path::new(&nemo), &device)?;
        let root = Path::new(&inputs_root);
        let clip = std::env::var("FF_N6E_ROWS_CLIP").unwrap_or_else(|_| "arrival".into());
        let (wave, diar) = if clip == "meeting" {
            let wave = read_wave(&root.join("meeting.wav"))?;
            let features = model.features(&wave)?;
            let frames = features.len() / 128 + 28;
            (wave, gt_masks(root, frames)?)
        } else {
            let wave = read_wave(&root.join("arrival-swap.wav"))?;
            let frames = feature_frames(&model, &wave);
            let base = gt_masks(root, frames)?;
            (wave, rotate_masks(&base, frames))
        };
        let features = model.features(&wave)?;
        let frames = features.len() / 128 + 28;
        // Masked mode: the per-lane path depends only on the lane's own mask,
        // so speaker 1 decodes identically alone (1 lane) and stacked (4
        // lanes). Kernel mode is excluded: its bg mask is the OR of the other
        // lanes, which a lone run cannot reproduce.
        let stacked = {
            let mut session = model.stream(features.clone(), diar.clone(), true, 4, true)?;
            let mut stream = model.encoder.multitalker_stream(4)?;
            let mut dstream = model.decode_stream()?;
            while let Some(step) = model.stream_step(&mut session, &mut stream)? {
                model.decode_step(&mut session, &mut dstream, &step)?;
            }
            session.hypotheses[1].tokens.clone()
        };
        let alone = {
            let mut solo = vec![0.0f32; frames];
            for frame in 0..frames {
                solo[frame] = diar[1 + frame * 4];
            }
            let mut session = model.stream(features.clone(), solo, true, 1, true)?;
            let mut stream = model.encoder.multitalker_stream(1)?;
            let mut dstream = model.decode_stream()?;
            while let Some(step) = model.stream_step(&mut session, &mut stream)? {
                model.decode_step(&mut session, &mut dstream, &step)?;
            }
            session.hypotheses[0].tokens.clone()
        };
        ensure!(
            stacked == alone,
            "speaker 1 decode differs alone ({} tokens) vs stacked ({} tokens)",
            alone.len(),
            stacked.len()
        );
        println!(
            "PASS multitalker_decode_lane_invariant tokens={} bit_identical=true",
            stacked.len()
        );
        Ok(())
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
            let mut session = model.stream(features.clone(), lane_masks, false, lanes, false)?;
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
        let _ = model.decode_stream()?;
        println!(
            "PASS multitalker_load blocks=24 vocab=1024 blank=1024 speaker_kernels=layer0 rnnt=embed1025+lstm2x640+joint1025"
        );
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
        let mut session = model.stream(features, mask_values, false, 4, false)?;
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
