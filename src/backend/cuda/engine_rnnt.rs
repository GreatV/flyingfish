//! Shared RNNT prediction, joint evaluation and greedy row decode.

use anyhow::{Context, Result, ensure};
use cudarc::driver::PushKernelArg;

use super::engine_parakeet::{Linear, ParakeetEncoder};
use super::ops::flat;
use crate::backend::gpu::CudaSlice;

pub(crate) struct RnntConfig {
    pub(crate) prediction_hidden: usize,
    pub(crate) layers: usize,
    pub(crate) vocab: usize,
    pub(crate) blank: usize,
    pub(crate) max_symbols: usize,
}

pub(crate) struct RnntLayer {
    pub(crate) input: Linear,
    pub(crate) recurrent: Linear,
    pub(crate) bias_ih: CudaSlice<f32>,
    pub(crate) bias_hh: CudaSlice<f32>,
}

pub(crate) struct RnntDecoder {
    pub(crate) config: RnntConfig,
    pub(crate) embedding: CudaSlice<f32>,
    pub(crate) layers: Vec<RnntLayer>,
    pub(crate) pred_projection: Linear,
    pub(crate) enc_projection: Linear,
    pub(crate) joint_head: Linear,
}

struct LayerState {
    hidden: CudaSlice<f32>,
    cell: CudaSlice<f32>,
    input_gates: CudaSlice<f32>,
    recurrent_gates: CudaSlice<f32>,
}

pub struct RnntDecodeStream {
    embed_input: CudaSlice<f32>,
    layers: Vec<LayerState>,
    prediction: CudaSlice<f32>,
    joint_frame: CudaSlice<f32>,
    joint_sum: CudaSlice<f32>,
    joint_activated: CudaSlice<f32>,
    joint_logits: CudaSlice<f32>,
}

#[derive(Default)]
pub struct RnntHypothesis {
    pub tokens: Vec<u32>,
    pub margins: Vec<f32>,
    pub emitted: Vec<bool>,
    pub timestamps: Vec<u32>,
    initialized: bool,
}

impl RnntDecoder {
    pub(crate) fn blank(&self) -> u32 {
        self.config.blank as u32
    }

    pub fn decode_stream(&self, encoder: &ParakeetEncoder) -> Result<RnntDecodeStream> {
        let stream = &encoder.stream;
        let hidden = self.config.prediction_hidden;
        let gate = 4 * hidden;
        let mut layers = Vec::with_capacity(self.config.layers);
        for _ in 0..self.config.layers {
            layers.push(LayerState {
                hidden: stream.alloc_zeros(4 * hidden)?,
                cell: stream.alloc_zeros(4 * hidden)?,
                input_gates: stream.alloc_zeros(4 * gate)?,
                recurrent_gates: stream.alloc_zeros(4 * gate)?,
            });
        }
        Ok(RnntDecodeStream {
            embed_input: stream.alloc_zeros(4 * hidden)?,
            layers,
            prediction: stream.alloc_zeros(4 * hidden)?,
            joint_frame: stream.alloc_zeros(4 * hidden)?,
            joint_sum: stream.alloc_zeros(4 * hidden)?,
            joint_activated: stream.alloc_zeros(4 * hidden)?,
            joint_logits: stream.alloc_zeros(4 * self.config.vocab)?,
        })
    }

    pub(crate) fn predict_lane(
        &self,
        encoder: &ParakeetEncoder,
        dstream: &mut RnntDecodeStream,
        lane: usize,
        label: u32,
        initialized: &mut bool,
    ) -> Result<()> {
        let stream = &encoder.stream;
        let hidden = self.config.prediction_hidden;
        ensure!(lane < 4, "prediction lane {lane} outside the 4-slot stride");
        ensure!(
            (label as usize) < self.config.vocab,
            "RNNT label outside vocabulary"
        );
        if *initialized && label as usize == self.config.blank {
            return Ok(());
        }
        let start = label as usize * hidden;
        stream.memcpy_dtod(
            &self.embedding.slice(start..start + hidden),
            &mut dstream
                .embed_input
                .slice_mut(lane * hidden..(lane + 1) * hidden),
        )?;
        for index in 0..self.layers.len() {
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
            let layer = &self.layers[index];
            encoder.lane_linear_into(input, 4, 4, &layer.input, input_gates)?;
            encoder.lane_linear_into(state_hidden, 4, 4, &layer.recurrent, recurrent_gates)?;
            unsafe {
                stream
                    .launch_builder(&encoder.ops.lstm)
                    .arg(&input_gates.slice(lane * 4 * hidden..(lane + 1) * 4 * hidden))
                    .arg(&recurrent_gates.slice(lane * 4 * hidden..(lane + 1) * 4 * hidden))
                    .arg(&layer.bias_ih)
                    .arg(&layer.bias_hh)
                    .arg(&state_cell.slice(lane * hidden..(lane + 1) * hidden))
                    .arg(&mut state_hidden.slice_mut(lane * hidden..(lane + 1) * hidden))
                    .arg(&(hidden as u32))
                    .launch(flat(hidden))?;
            }
        }
        let last = dstream
            .layers
            .last()
            .context("prediction LSTM stack empty")?;
        encoder.lane_linear_into(
            &last.hidden,
            4,
            4,
            &self.pred_projection,
            &mut dstream.prediction,
        )?;
        *initialized = true;
        Ok(())
    }

    pub(crate) fn joint_rows(
        &self,
        encoder: &ParakeetEncoder,
        dstream: &mut RnntDecodeStream,
        projected: &CudaSlice<f32>,
        row_offsets: &[usize; 4],
    ) -> Result<Vec<f32>> {
        let stream = &encoder.stream;
        let hidden = self.config.prediction_hidden;
        for (lane, &start) in row_offsets.iter().enumerate() {
            stream.memcpy_dtod(
                &projected.slice(start..start + hidden),
                &mut dstream
                    .joint_frame
                    .slice_mut(lane * hidden..(lane + 1) * hidden),
            )?;
        }
        stream.memcpy_dtod(&dstream.prediction, &mut dstream.joint_sum)?;
        encoder.residual(&mut dstream.joint_sum, &dstream.joint_frame, 1.0)?;
        encoder.activate_into(
            &dstream.joint_sum,
            4,
            self.config.prediction_hidden as u32,
            0,
            &mut dstream.joint_activated,
        )?;
        encoder.lane_linear_into(
            &dstream.joint_activated,
            4,
            4,
            &self.joint_head,
            &mut dstream.joint_logits,
        )?;
        Ok(stream.clone_dtoh(&dstream.joint_logits)?)
    }

    pub(crate) fn decode_rows(
        &self,
        encoder: &ParakeetEncoder,
        dstream: &mut RnntDecodeStream,
        projected: &CudaSlice<f32>,
        hypotheses: &mut [RnntHypothesis],
        active: &[usize],
        row_bases: &[usize; 4],
    ) -> Result<()> {
        let hidden = self.config.prediction_hidden;
        ensure!(
            projected.len().is_multiple_of(4 * hidden),
            "RNNT projected row extent"
        );
        let out_rows = projected.len() / (4 * hidden);
        ensure!(out_rows > 0, "decode step with zero rows");
        ensure!(hypotheses.len() <= 4, "RNNT lane count exceeds four");
        let mut cursor = [0usize; 4];
        let mut symbols = [0usize; 4];
        let mut live = [false; 4];
        for &lane in active {
            ensure!(lane < hypotheses.len(), "RNNT lane outside hypothesis set");
            live[lane] = true;
            let hypothesis = &mut hypotheses[lane];
            if !hypothesis.initialized {
                let mut initialized = hypothesis.initialized;
                self.predict_lane(encoder, dstream, lane, self.blank(), &mut initialized)?;
                hypothesis.initialized = initialized;
            }
        }
        while live.iter().any(|value| *value) {
            let mut row_offsets = [0usize; 4];
            for lane in 0..4 {
                let row = cursor[lane].min(out_rows - 1);
                row_offsets[lane] = (lane * out_rows + row) * hidden;
            }
            let logits = self.joint_rows(encoder, dstream, projected, &row_offsets)?;
            for &lane in active {
                if !live[lane] {
                    continue;
                }
                let slice = &logits[lane * self.config.vocab..(lane + 1) * self.config.vocab];
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
                let hypothesis = &mut hypotheses[lane];
                hypothesis.margins.push(margin);
                hypothesis.emitted.push(best != self.config.blank);
                if best == self.config.blank {
                    cursor[lane] += 1;
                    symbols[lane] = 0;
                } else {
                    hypothesis.tokens.push(label);
                    hypothesis
                        .timestamps
                        .push((row_bases[lane] + cursor[lane]) as u32);
                    symbols[lane] += 1;
                    let mut initialized = hypothesis.initialized;
                    self.predict_lane(encoder, dstream, lane, label, &mut initialized)?;
                    hypothesis.initialized = initialized;
                    if symbols[lane] >= self.config.max_symbols {
                        cursor[lane] += 1;
                        symbols[lane] = 0;
                    }
                }
                if cursor[lane] >= out_rows {
                    live[lane] = false;
                }
            }
        }
        Ok(())
    }
}
