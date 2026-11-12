//! Parakeet realtime EOU 120m engine: offline encoder, prediction, joint and greedy RNNT decode.

use crate::{audio::LogMel, checkpoint::MappedCheckpoint, eou::EouSpec};
use anyhow::{Context, Result, ensure};
use std::{path::Path, sync::Arc};

use super::Device;
use super::engine_parakeet::{Linear, LstmLayer, ParakeetEncoder, tensor};
use crate::backend::gpu::{CudaSlice, PushKernelArg};

const MAX_SYMBOLS: usize = 10;

pub struct Eou {
    encoder: ParakeetEncoder,
    frontend: LogMel,
    pub spec: EouSpec,
    embedding: CudaSlice<f32>,
    lstm: Vec<LstmLayer>,
    pred_projector: Linear,
    enc_projector: Linear,
    joint: Linear,
    input: CudaSlice<f32>,
    cache: CudaSlice<f32>,
    joint_sum: CudaSlice<f32>,
    joint_activated: CudaSlice<f32>,
    joint_logits: CudaSlice<f32>,
    selection: CudaSlice<u32>,
    projected: Option<CudaSlice<f32>>,
    initialized: bool,
}

impl Eou {
    pub fn load(nemo: &Path, device: &Device) -> Result<Self> {
        let spec = EouSpec::read(nemo)?;
        spec.validate()?;
        let encoder = ParakeetEncoder::load_eou(nemo, device)?;
        let frontend = LogMel::new(spec.mel())?;
        let checkpoint = MappedCheckpoint::open_nemo(nemo)?;
        let stream = Arc::clone(&encoder.stream);
        let hidden = spec.pred_hidden as usize;
        let classes = spec.vocab_size as usize + 1;
        let gate = 4 * hidden;
        let embedding = tensor(
            &checkpoint,
            &stream,
            "decoder.prediction.embed.weight",
            &[classes, hidden],
        )?;
        let mut lstm = Vec::with_capacity(spec.pred_rnn_layers as usize);
        for index in 0..spec.pred_rnn_layers {
            lstm.push(LstmLayer {
                input: Linear {
                    weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("decoder.prediction.dec_rnn.lstm.weight_ih_l{index}"),
                        &[gate, hidden],
                    )?,
                    bias: None,
                    output: 4 * spec.pred_hidden,
                    input: spec.pred_hidden,
                },
                recurrent: Linear {
                    weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("decoder.prediction.dec_rnn.lstm.weight_hh_l{index}"),
                        &[gate, hidden],
                    )?,
                    bias: None,
                    output: 4 * spec.pred_hidden,
                    input: spec.pred_hidden,
                },
                bias_ih: tensor(
                    &checkpoint,
                    &stream,
                    &format!("decoder.prediction.dec_rnn.lstm.bias_ih_l{index}"),
                    &[gate],
                )?,
                bias_hh: tensor(
                    &checkpoint,
                    &stream,
                    &format!("decoder.prediction.dec_rnn.lstm.bias_hh_l{index}"),
                    &[gate],
                )?,
                hidden: stream.alloc_zeros(hidden)?,
                cell: stream.alloc_zeros(hidden)?,
                input_gates: stream.alloc_zeros(gate)?,
                recurrent_gates: stream.alloc_zeros(gate)?,
            });
        }
        let pred_projector = super::engine_parakeet::linear(
            &checkpoint,
            &stream,
            "joint.pred",
            spec.pred_hidden,
            spec.pred_hidden,
            true,
        )?;
        let enc_projector = super::engine_parakeet::linear(
            &checkpoint,
            &stream,
            "joint.enc",
            spec.pred_hidden,
            spec.d_model,
            true,
        )?;
        let joint = super::engine_parakeet::linear(
            &checkpoint,
            &stream,
            "joint.joint_net.2",
            u32::try_from(classes)?,
            spec.joint_hidden,
            true,
        )?;
        let input = stream.alloc_zeros(hidden)?;
        let cache = stream.alloc_zeros(hidden)?;
        let joint_sum = stream.alloc_zeros(hidden)?;
        let joint_activated = stream.alloc_zeros(hidden)?;
        let joint_logits = stream.alloc_zeros(classes)?;
        let selection = stream.alloc_zeros(1)?;
        stream.synchronize()?;
        Ok(Self {
            encoder,
            frontend,
            spec,
            embedding,
            lstm,
            pred_projector,
            enc_projector,
            joint,
            input,
            cache,
            joint_sum,
            joint_activated,
            joint_logits,
            selection,
            projected: None,
            initialized: false,
        })
    }

    /// Mel features `[frames, 128]` matching the published preprocessor.
    pub fn features(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let batch = self.frontend.extract(&[waveform], false, true, false)?;
        let mel_bins = self.spec.features as usize;
        Ok(batch.features[..batch.frames * mel_bins].to_vec())
    }

    /// Offline encoder output `[rows, 512]` under the published [70, 1] attention window.
    pub fn encode(&self, features: &[f32], valid: usize) -> Result<Vec<f32>> {
        let (hidden, _mask, rows) = self.encoder.encode_right(features, valid, 0)?;
        Ok(hidden[..rows * self.spec.d_model as usize].to_vec())
    }

    /// Blank index: the vocabulary of 1026 fills logits 0..=1025, blank is 1026.
    pub fn blank(&self) -> u32 {
        self.spec.vocab_size
    }

    fn reset_prediction(&mut self) -> Result<()> {
        for layer in &mut self.lstm {
            self.encoder.stream.memset_zeros(&mut layer.hidden)?;
            self.encoder.stream.memset_zeros(&mut layer.cell)?;
        }
        self.encoder.stream.memset_zeros(&mut self.cache)?;
        self.encoder.stream.memset_zeros(&mut self.input)?;
        self.initialized = false;
        Ok(())
    }

    fn predict(&mut self, token: u32) -> Result<()> {
        ensure!(
            token <= self.blank(),
            "EOU prediction token outside vocabulary"
        );
        if self.initialized && token == self.blank() {
            return Ok(());
        }
        let width = self.spec.pred_hidden as usize;
        let start = token as usize * width;
        self.encoder
            .stream
            .memcpy_dtod(&self.embedding.slice(start..start + width), &mut self.input)?;
        for index in 0..self.lstm.len() {
            let (previous, current) = self.lstm.split_at_mut(index);
            let layer = &mut current[0];
            let input = if index == 0 {
                &self.input
            } else {
                &previous[index - 1].hidden
            };
            self.encoder
                .linear_into(input, 1, &layer.input, &mut layer.input_gates)?;
            self.encoder.linear_into(
                &layer.hidden,
                1,
                &layer.recurrent,
                &mut layer.recurrent_gates,
            )?;
            unsafe {
                self.encoder
                    .stream
                    .launch_builder(&self.encoder.ops.lstm)
                    .arg(&layer.input_gates)
                    .arg(&layer.recurrent_gates)
                    .arg(&layer.bias_ih)
                    .arg(&layer.bias_hh)
                    .arg(&layer.cell)
                    .arg(&layer.hidden)
                    .arg(&self.spec.pred_hidden)
                    .launch(super::ops::flat(width))?;
            }
        }
        let last = self
            .lstm
            .last()
            .context("EOU prediction LSTM stack empty")?;
        self.encoder
            .linear_into(&last.hidden, 1, &self.pred_projector, &mut self.cache)?;
        self.initialized = true;
        Ok(())
    }

    fn joint_argmax(&mut self, frame: usize) -> Result<u32> {
        let width = self.spec.pred_hidden as usize;
        let projected = self.projected.as_ref().context("EOU projector missing")?;
        let rows = projected.len() / width;
        ensure!(frame < rows, "EOU joint frame {frame} outside {rows}");
        let row = projected.slice(frame * width..(frame + 1) * width);
        self.encoder
            .stream
            .memcpy_dtod(&self.cache, &mut self.joint_sum)?;
        let count = self.joint_sum.len() as u64;
        unsafe {
            self.encoder
                .stream
                .launch_builder(&self.encoder.ops.residual)
                .arg(&self.joint_sum)
                .arg(&row)
                .arg(&self.joint_sum)
                .arg(&count)
                .arg(&1.0f32)
                .launch(super::ops::flat(self.joint_sum.len()))?;
        }
        self.encoder.activate_into(
            &self.joint_sum,
            1,
            self.spec.pred_hidden,
            0,
            &mut self.joint_activated,
        )?;
        self.encoder.linear_into(
            &self.joint_activated,
            1,
            &self.joint,
            &mut self.joint_logits,
        )?;
        unsafe {
            self.encoder
                .stream
                .launch_builder(&self.encoder.ops.argmax)
                .arg(&self.joint_logits)
                .arg(&mut self.selection.slice_mut(..1))
                .arg(&(self.joint_logits.len() as i32))
                .launch(super::ops::grid(1, 1024))?;
        }
        let selected = self.encoder.stream.clone_dtoh(&self.selection)?;
        Ok(selected[0])
    }

    /// Top1-top2 gap of the current joint logits, in logit units.
    fn joint_margin(&self) -> Result<f32> {
        let logits = self.encoder.stream.clone_dtoh(&self.joint_logits)?;
        let (mut top1, mut top2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &value in &logits {
            if value > top1 {
                top2 = top1;
                top1 = value;
            } else if value > top2 {
                top2 = value;
            }
        }
        Ok(top1 - top2)
    }

    /// Greedy RNNT decode over the offline encoder output; emits non-blank token indices.
    pub fn decode(&mut self, features: &[f32], valid: usize) -> Result<Vec<u32>> {
        self.reset_prediction()?;
        self.projected = None;
        let encoded = self.encode(features, valid)?;
        let hidden = self.spec.d_model as usize;
        ensure!(
            !encoded.is_empty() && encoded.len().is_multiple_of(hidden),
            "EOU decode encoder extent"
        );
        let device_encoded = self.encoder.stream.clone_htod(&encoded)?;
        let projected = self.encoder.linear(
            &device_encoded,
            u32::try_from(encoded.len() / hidden)?,
            &self.enc_projector,
        )?;
        self.projected = Some(projected);
        let width = self.spec.pred_hidden as usize;
        let rows = self
            .projected
            .as_ref()
            .context("EOU projector missing")?
            .len()
            / width;
        let capacity = MAX_SYMBOLS
            .checked_mul(rows)
            .context("EOU decode capacity overflow")?;
        let mut tokens: Vec<u32> = Vec::with_capacity(capacity);
        let mut frame = 0usize;
        while frame < rows {
            self.predict(self.blank())?;
            let mut symbols = 0usize;
            loop {
                let selected = self.joint_argmax(frame)?;
                ensure!(
                    tokens.len() < capacity,
                    "EOU decode capacity exceeded: tokens={} capacity={capacity}",
                    tokens.len()
                );
                if selected == self.blank() {
                    break;
                }
                tokens.push(selected);
                self.predict(selected)?;
                symbols += 1;
                if symbols >= MAX_SYMBOLS {
                    break;
                }
            }
            frame += 1;
        }
        Ok(tokens)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use anyhow::Context;
    use safetensors::SafeTensors;

    pub(crate) fn read_wave_public(path: &std::path::Path) -> Result<Vec<f32>> {
        read_wave(path)
    }

    pub(crate) fn read_wave(path: &std::path::Path) -> Result<Vec<f32>> {
        crate::media::read_wav(std::fs::File::open(path)?, 16000)
    }

    fn read_f32(st: &SafeTensors, key: &str) -> Result<Vec<f32>> {
        let tensor = st.tensor(key)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "tensor {key} dtype"
        );
        let bytes = tensor.data();
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn read_i64(st: &SafeTensors, key: &str) -> Result<Vec<i64>> {
        let tensor = st.tensor(key)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::I64,
            "tensor {key} dtype"
        );
        let bytes = tensor.data();
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect())
    }

    fn metrics(candidate: &[f32], reference: &[f32]) -> (f64, f64) {
        assert_eq!(candidate.len(), reference.len());
        let mut max_abs = 0f64;
        let mut squares = 0f64;
        for (left, right) in candidate.iter().zip(reference) {
            let diff = f64::from(*left) - f64::from(*right);
            max_abs = max_abs.max(diff.abs());
            squares += diff * diff;
        }
        (max_abs, (squares / candidate.len() as f64).sqrt())
    }

    #[test]
    #[ignore = "requires GPU lease; set FF_EOU_NEMO, FF_EOU_FROZEN, FF_EOU_ARM"]
    fn eou_offline_encoder_matches_frozen_reference() -> Result<()> {
        let nemo =
            std::path::PathBuf::from(std::env::var("FF_EOU_NEMO").context("FF_EOU_NEMO missing")?);
        let frozen = std::path::PathBuf::from(
            std::env::var("FF_EOU_FROZEN").context("FF_EOU_FROZEN missing")?,
        );
        let arm =
            std::path::PathBuf::from(std::env::var("FF_EOU_ARM").context("FF_EOU_ARM missing")?);
        let floors: serde_json::Value = serde_json::from_slice(&std::fs::read(&frozen)?)?;
        ensure!(floors["status"] == "FROZEN", "reference floors not frozen");
        let device = Device::new(0)?;
        let engine = Eou::load(&nemo, &device)?;
        let mel_bins = engine.spec.features as usize;
        for clip in ["clipA", "clipB", "prod120"] {
            let wave = read_wave(&arm.join(format!("inputs/{clip}.wav")))?;
            let features = engine.features(&wave)?;
            let rows = features.len() / mel_bins;
            let encoded = engine.encode(&features, rows)?;
            let bytes = std::fs::read(arm.join(format!("CPU-r1/media/{clip}.safetensors")))?;
            let st = SafeTensors::deserialize(&bytes)?;
            let reference_features = read_f32(&st, "features")?;
            let reference_encoded = read_f32(&st, "encoder_out")?;
            assert_eq!(features.len(), reference_features.len());
            assert_eq!(encoded.len(), reference_encoded.len());
            let (feature_abs, feature_rmse) = metrics(&features, &reference_features);
            let (encode_abs, encode_rmse) = metrics(&encoded, &reference_encoded);
            let feature_tau =
                &floors["metrics"][format!("media/{clip}.safetensors/features")]["tau"];
            let encode_tau =
                &floors["metrics"][format!("media/{clip}.safetensors/encoder_out")]["tau"];
            ensure!(
                feature_abs <= feature_tau["max_abs"].as_f64().context("feature tau")?
                    && feature_rmse <= feature_tau["rmse"].as_f64().context("feature tau")?,
                "EOU features {clip} max_abs {feature_abs:e} rmse {feature_rmse:e} outside tau"
            );
            ensure!(
                encode_abs <= encode_tau["max_abs"].as_f64().context("encoder tau")?
                    && encode_rmse <= encode_tau["rmse"].as_f64().context("encoder tau")?,
                "EOU encoder {clip} max_abs {encode_abs:e} rmse {encode_rmse:e} outside tau"
            );
            println!(
                "EOU_S1_CELL clip={clip} features max_abs={feature_abs:e} rmse={feature_rmse:e} tau=<{:.3e},{:.3e}> encoder max_abs={encode_abs:e} rmse={encode_rmse:e} tau=<{:.3e},{:.3e}> PASS",
                feature_tau["max_abs"].as_f64().unwrap(),
                feature_tau["rmse"].as_f64().unwrap(),
                encode_tau["max_abs"].as_f64().unwrap(),
                encode_tau["rmse"].as_f64().unwrap()
            );
        }
        println!("EOU_S1_ENCODER_GATE PASS clips=3 within_tau=true");
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease; set FF_EOU_NEMO, FF_EOU_FROZEN, FF_EOU_ARM"]
    fn eou_offline_decode_matches_frozen_tokens() -> Result<()> {
        let nemo =
            std::path::PathBuf::from(std::env::var("FF_EOU_NEMO").context("FF_EOU_NEMO missing")?);
        let frozen = std::path::PathBuf::from(
            std::env::var("FF_EOU_FROZEN").context("FF_EOU_FROZEN missing")?,
        );
        let arm =
            std::path::PathBuf::from(std::env::var("FF_EOU_ARM").context("FF_EOU_ARM missing")?);
        let _floors: serde_json::Value = serde_json::from_slice(&std::fs::read(&frozen)?)?;
        let device = Device::new(0)?;
        let mut engine = Eou::load(&nemo, &device)?;
        let mel_bins = engine.spec.features as usize;
        let eou = engine.spec.vocab_size as u32 - 2;
        for clip in ["clipA", "clipB", "prod120"] {
            let wave = read_wave(&arm.join(format!("inputs/{clip}.wav")))?;
            let features = engine.features(&wave)?;
            let tokens = engine.decode(&features, features.len() / mel_bins)?;
            let bytes = std::fs::read(arm.join(format!("CPU-r1/media/{clip}.safetensors")))?;
            let st = SafeTensors::deserialize(&bytes)?;
            let reference = read_i64(&st, "token_ids")?;
            let reference_eou = read_i64(&st, "eou_positions")?;
            ensure!(
                tokens.len() == reference.len(),
                "EOU decode {clip} token count {} against reference {}",
                tokens.len(),
                reference.len()
            );
            for (index, (left, right)) in tokens.iter().zip(&reference).enumerate() {
                ensure!(
                    i64::from(*left) == *right,
                    "EOU decode {clip} token {index} = {left} against {right}"
                );
            }
            let emitted: Vec<i64> = tokens
                .iter()
                .enumerate()
                .filter(|(_, token)| **token as i64 == i64::from(eou))
                .map(|(index, _)| index as i64)
                .collect();
            ensure!(
                emitted == reference_eou,
                "EOU decode {clip} eou positions {emitted:?} against {reference_eou:?}"
            );
            println!(
                "EOU_S2_CELL clip={clip} tokens={} eou_positions={} exact=true PASS",
                tokens.len(),
                emitted.len()
            );
        }
        println!("EOU_S2_DECODE_GATE PASS clips=3 token_ids_exact=true");
        Ok(())
    }
}

pub struct EouUtterance {
    pub tokens: Vec<u32>,
    pub end_step: usize,
}

impl Eou {
    /// Streaming greedy RNNT decode at the voice-agent cadence: one encoder row
    /// per step, hypothesis continuation, full session reset after <EOU>.
    /// Returns one token vector per utterance (each ends with EOU except
    /// possibly the last).
    pub fn stream_decode(&mut self, features: &[f32]) -> Result<(Vec<Vec<u32>>, f32)> {
        let (utterances, margin) = self.stream_decode_timed(features)?;
        Ok((
            utterances
                .into_iter()
                .map(|utterance| utterance.tokens)
                .collect(),
            margin,
        ))
    }

    pub fn stream_decode_timed(&mut self, features: &[f32]) -> Result<(Vec<EouUtterance>, f32)> {
        let mel_bins = self.spec.features as usize;
        ensure!(
            !features.is_empty() && features.len().is_multiple_of(mel_bins),
            "EOU streaming features extent"
        );
        let frames = features.len() / mel_bins;
        let steps = frames / 8 + 1;
        let mut session = self.encoder.eou_stream()?;
        self.reset_prediction()?;
        self.projected = None;
        let eou_token = self.spec.vocab_size - 2;
        let mut utterances: Vec<EouUtterance> = Vec::new();
        let mut min_margin = f32::INFINITY;
        let mut current: Vec<u32> = Vec::new();
        let mut window = vec![0f32; 25 * mel_bins];
        for step in 0..steps {
            // CacheAwareStreamingAudioBuffer cadence: step 0 consumes 9 frames
            // and shifts 1; step 1 carries a zero-padded 9-frame pre-encode cache;
            // later steps shift 8 with the window starting 16 frames before the row.
            if step == 0 {
                let stop = 9.min(frames);
                window[..stop * mel_bins].copy_from_slice(&features[..stop * mel_bins]);
                window[stop * mel_bins..].fill(0.0);
            } else if step == 1 {
                window[..8 * mel_bins].fill(0.0);
                let stop = 17.min(frames);
                window[8 * mel_bins..(8 + stop) * mel_bins]
                    .copy_from_slice(&features[..stop * mel_bins]);
                window[(8 + stop) * mel_bins..].fill(0.0);
            } else {
                let start = 8 * step - 16;
                for (index, value) in window.iter_mut().enumerate() {
                    let frame = start + index / mel_bins;
                    let bin = index % mel_bins;
                    *value = if frame < frames {
                        features[frame * mel_bins + bin]
                    } else {
                        0.0
                    };
                }
            }
            let length = if step == 0 { 9 } else { 25 };
            let total_rows = (frames - 1) / 8 + 1;
            let valid = usize::from(step < total_rows) + usize::from(step + 1 < total_rows);
            if valid == 0 && session.cache_len == 0 {
                continue;
            }
            let step_rows =
                self.encoder
                    .eou_encode_step(&mut session, &window[..length * mel_bins], valid)?;
            let hidden = self.spec.d_model as usize;
            let row: Vec<f32> = step_rows.rows[..hidden].to_vec();
            let device_row = self.encoder.stream.clone_htod(&row)?;
            let projected = self.encoder.linear(&device_row, 1, &self.enc_projector)?;
            self.projected = Some(projected);
            self.predict(self.blank())?;
            let mut symbols = 0usize;
            loop {
                let selected = self.joint_argmax(0)?;
                if selected == self.blank() {
                    break;
                }
                min_margin = min_margin.min(self.joint_margin()?);
                current.push(selected);
                self.predict(selected)?;
                symbols += 1;
                if symbols >= MAX_SYMBOLS {
                    break;
                }
            }
            let _ = length;
            if step_closes_utterance(&current, eou_token) {
                utterances.push(EouUtterance {
                    tokens: std::mem::take(&mut current),
                    end_step: step,
                });
                self.encoder.eou_stream_reset(&mut session)?;
                self.reset_prediction()?;
                self.projected = None;
            }
        }
        if !current.is_empty() {
            utterances.push(EouUtterance {
                tokens: current,
                end_step: steps - 1,
            });
        }
        Ok((utterances, min_margin))
    }
}

/// An utterance closes when the reference's step tokens contain the EOU
/// token at any position.
fn step_closes_utterance(current: &[u32], eou_token: u32) -> bool {
    current.contains(&eou_token)
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use anyhow::Context;
    use safetensors::SafeTensors;

    fn read_f32(st: &SafeTensors, key: &str) -> Result<Vec<f32>> {
        let tensor = st.tensor(key)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "tensor {key} dtype"
        );
        let bytes = tensor.data();
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn read_i64(st: &SafeTensors, key: &str) -> Result<Vec<i64>> {
        let tensor = st.tensor(key)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::I64,
            "tensor {key} dtype"
        );
        let bytes = tensor.data();
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect())
    }

    fn metrics(candidate: &[f32], reference: &[f32]) -> (f64, f64) {
        assert_eq!(candidate.len(), reference.len());
        let mut max_abs = 0f64;
        let mut squares = 0f64;
        for (left, right) in candidate.iter().zip(reference) {
            let diff = f64::from(*left) - f64::from(*right);
            max_abs = max_abs.max(diff.abs());
            squares += diff * diff;
        }
        (max_abs, (squares / candidate.len() as f64).sqrt())
    }

    #[test]
    #[ignore = "requires GPU lease; set FF_EOU_NEMO, FF_EOU_FROZEN, FF_EOU_ARM"]
    fn eou_streaming_matches_frozen_reference() -> Result<()> {
        let nemo =
            std::path::PathBuf::from(std::env::var("FF_EOU_NEMO").context("FF_EOU_NEMO missing")?);
        let frozen = std::path::PathBuf::from(
            std::env::var("FF_EOU_FROZEN").context("FF_EOU_FROZEN missing")?,
        );
        let arm =
            std::path::PathBuf::from(std::env::var("FF_EOU_ARM").context("FF_EOU_ARM missing")?);
        let floors: serde_json::Value = serde_json::from_slice(&std::fs::read(&frozen)?)?;
        ensure!(floors["status"] == "FROZEN", "reference floors not frozen");
        let device = Device::new(0)?;
        let mut engine = Eou::load(&nemo, &device)?;
        let wave = crate::backend::cuda::engine_eou::tests::read_wave_public(
            &arm.join("inputs/clipA.wav"),
        )?;
        let features = engine.features(&wave)?;
        {
            // step-1 enc rows against the frozen cell (post table reversal)
            let mel_bins = engine.spec.features as usize;
            let mut session = engine.encoder.eou_stream()?;
            let mut window = vec![0f32; 25 * mel_bins];
            window[..9 * mel_bins].copy_from_slice(&features[..9 * mel_bins]);
            let s0 = engine
                .encoder
                .eou_encode_step(&mut session, &window[..9 * mel_bins], 2)?;
            window[..8 * mel_bins].fill(0.0);
            window[8 * mel_bins..].copy_from_slice(&features[..17 * mel_bins]);
            let s1 = engine.encoder.eou_encode_step(&mut session, &window, 2)?;
            let cell_bytes = std::fs::read(arm.join("CPU-r1/media/clipA.safetensors"))?;
            let cell_st = SafeTensors::deserialize(&cell_bytes)?;
            let ref0 = read_f32(&cell_st, "stream_step0_enc_out")?;
            let ref1 = read_f32(&cell_st, "stream_step1_enc_out")?;
            let (m0, r0) = metrics(&s0.rows, &ref0);
            let (m1, r1) = metrics(&s1.rows, &ref1);
            println!(
                "EOU_S3_STEP step0 max_abs={m0:e} rmse={r0:e} step1 max_abs={m1:e} rmse={r1:e} s1_head={:?} ref1_head={:?}",
                &s1.rows[..4],
                &ref1[..4]
            );
        }
        for clip in ["clipA", "clipB"] {
            let wave = crate::backend::cuda::engine_eou::tests::read_wave_public(
                &arm.join(format!("inputs/{clip}.wav")),
            )?;
            let clip_features = engine.features(&wave)?;
            let (utterances, min_margin) = engine.stream_decode(&clip_features)?;
            let bytes = std::fs::read(arm.join(format!("CPU-r1/media/{clip}.safetensors")))?;
            let st = SafeTensors::deserialize(&bytes)?;
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(arm.join("CPU-r1/stream-report.json"))?)?;
            let expected_counts: Vec<usize> = report[clip]["utterance_token_counts"]
                .as_array()
                .context("utterance counts")?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            ensure!(
                utterances.len() == expected_counts.len(),
                "EOU stream utterances {} against {}",
                utterances.len(),
                expected_counts.len()
            );
            for (index, tokens) in utterances.iter().enumerate() {
                let key = format!("stream_utt{index}_token_ids");
                let reference = read_i64(&st, &key)?;
                ensure!(
                    tokens.len() == reference.len(),
                    "EOU stream utterance {index} tokens {} against {}",
                    tokens.len(),
                    reference.len()
                );
                for (position, (left, right)) in tokens.iter().zip(&reference).enumerate() {
                    ensure!(
                        i64::from(*left) == *right,
                        "EOU stream utterance {index} token {position} = {left} against {right}"
                    );
                }
                println!(
                    "EOU_S3_UTT clip={clip} index={index} tokens={} exact=true PASS",
                    tokens.len()
                );
            }
            println!(
                "EOU_S3_STREAM_GATE PASS clip={clip} utterances={} min_top1_top2_gap={min_margin:e}",
                utterances.len()
            );
        }
        Ok(())
    }

    #[test]
    fn eou_step_closes_utterance_on_mid_row_eou() {
        let mid = vec![4u32, 1019, 1024, 30];
        assert!(step_closes_utterance(&mid, 1024));
        let tail = vec![4u32, 1019, 1024];
        assert!(step_closes_utterance(&tail, 1024));
        let none = vec![4u32, 1019, 30];
        assert!(!step_closes_utterance(&none, 1024));
    }

    #[test]
    #[ignore = "GPU; set FF_EOU_NEMO to the published .nemo"]
    fn eou_encode_step_without_valid_rows_or_cache_errors() -> Result<()> {
        let nemo =
            std::path::PathBuf::from(std::env::var("FF_EOU_NEMO").context("FF_EOU_NEMO missing")?);
        let device = Device::new(0)?;
        let engine = Eou::load(&nemo, &device)?;
        let mut session = engine.encoder.eou_stream()?;
        let mel_bins = engine.spec.features as usize;
        let window = vec![0f32; 25 * mel_bins];
        let err = match engine.encoder.eou_encode_step(&mut session, &window, 0) {
            Ok(_) => panic!("a fully invalid window on an empty cache must error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("no valid rows"),
            "unexpected error: {err}"
        );
        Ok(())
    }
}
