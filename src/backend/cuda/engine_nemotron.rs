//! Nemotron 3.5 ASR prompt conditioning and RNNT transcription.

use crate::{audio::LogMel, checkpoint::MappedCheckpoint, nemotron::NemotronSpec};
use anyhow::{Result, ensure};
use std::{path::Path, sync::Arc};

use super::Device;
use super::engine_parakeet::{Linear, ParakeetEncoder, linear, tensor};
use super::engine_rnnt::{RnntConfig, RnntDecodeStream, RnntDecoder, RnntHypothesis, RnntLayer};
use crate::backend::gpu::{CudaSlice, CudaStream};

pub(crate) struct Nemotron {
    encoder: ParakeetEncoder,
    frontend: LogMel,
    spec: NemotronSpec,
    prompt_up: Linear,
    prompt_down: Linear,
    rnnt: RnntDecoder,
}

fn load_rnnt(
    checkpoint: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    spec: &NemotronSpec,
) -> Result<RnntDecoder> {
    let hidden = spec.decoder_hidden as usize;
    let vocab = spec.vocab as usize;
    let gate = 4 * hidden;
    let embedding = tensor(
        checkpoint,
        stream,
        "decoder.embedding.weight",
        &[vocab, hidden],
    )?;
    let mut layers = Vec::with_capacity(spec.decoder_layers as usize);
    for index in 0..spec.decoder_layers as usize {
        let gate_linear = |suffix: &str| -> Result<Linear> {
            Ok(Linear {
                weight: tensor(
                    checkpoint,
                    stream,
                    &format!("decoder.lstm.{suffix}_l{index}"),
                    &[gate, hidden],
                )?,
                bias: None,
                output: gate as u32,
                input: spec.decoder_hidden,
            })
        };
        layers.push(RnntLayer {
            input: gate_linear("weight_ih")?,
            recurrent: gate_linear("weight_hh")?,
            bias_ih: tensor(
                checkpoint,
                stream,
                &format!("decoder.lstm.bias_ih_l{index}"),
                &[gate],
            )?,
            bias_hh: tensor(
                checkpoint,
                stream,
                &format!("decoder.lstm.bias_hh_l{index}"),
                &[gate],
            )?,
        });
    }
    Ok(RnntDecoder {
        config: RnntConfig {
            prediction_hidden: hidden,
            layers: spec.decoder_layers as usize,
            vocab,
            blank: spec.blank as usize,
            max_symbols: spec.max_symbols as usize,
        },
        embedding,
        layers,
        pred_projection: linear(
            checkpoint,
            stream,
            "decoder.decoder_projector",
            spec.decoder_hidden,
            spec.decoder_hidden,
            true,
        )?,
        enc_projection: linear(
            checkpoint,
            stream,
            "encoder_projector",
            spec.decoder_hidden,
            spec.hidden,
            true,
        )?,
        joint_head: linear(
            checkpoint,
            stream,
            "joint.head",
            spec.vocab,
            spec.decoder_hidden,
            true,
        )?,
    })
}

impl Nemotron {
    pub(crate) fn load(directory: &Path, device: &Device) -> Result<Self> {
        let spec = NemotronSpec::read(directory)?;
        let encoder = ParakeetEncoder::load_nemotron(directory, device)?;
        let checkpoint = MappedCheckpoint::open(directory)?;
        let stream = Arc::clone(&encoder.stream);
        let frontend = LogMel::new(spec.mel)?;
        let prompt_up = linear(
            &checkpoint,
            &stream,
            "prompt_projector.linear_1",
            spec.prompt_intermediate,
            spec.hidden + spec.prompts,
            true,
        )?;
        let prompt_down = linear(
            &checkpoint,
            &stream,
            "prompt_projector.linear_2",
            spec.hidden,
            spec.prompt_intermediate,
            true,
        )?;
        let rnnt = load_rnnt(&checkpoint, &stream, &spec)?;
        stream.synchronize()?;
        Ok(Self {
            encoder,
            frontend,
            spec,
            prompt_up,
            prompt_down,
            rnnt,
        })
    }

    fn project_rows(&self, rows: &[f32]) -> Result<CudaSlice<f32>> {
        let width = self.spec.hidden as usize;
        ensure!(
            !rows.is_empty() && rows.len().is_multiple_of(width),
            "Nemotron encoder rows are misaligned"
        );
        let count = rows.len() / width;
        let prompts = self.spec.prompts as usize;
        let input_width = width + prompts;
        let mut input = vec![0.0f32; count * input_width];
        for (index, row) in rows.chunks_exact(width).enumerate() {
            let start = index * input_width;
            input[start..start + width].copy_from_slice(row);
            input[start + width + self.spec.default_prompt as usize] = 1.0;
        }
        let stream = &self.encoder.stream;
        let device_input = stream.clone_htod(&input)?;
        let up = self
            .encoder
            .linear(&device_input, u32::try_from(count)?, &self.prompt_up)?;
        let intermediate = self.spec.prompt_intermediate as usize;
        let mut activated = stream.alloc_zeros(count * intermediate)?;
        self.encoder.activate_into(
            &up,
            u32::try_from(count)?,
            self.spec.prompt_intermediate,
            0,
            &mut activated,
        )?;
        let conditioned =
            self.encoder
                .linear(&activated, u32::try_from(count)?, &self.prompt_down)?;
        let projected = self.encoder.linear(
            &conditioned,
            u32::try_from(count)?,
            &self.rnnt.enc_projection,
        )?;
        let joint_hidden = self.spec.decoder_hidden as usize;
        let mut padded = stream.alloc_zeros(4 * count * joint_hidden)?;
        stream.memcpy_dtod(&projected, &mut padded.slice_mut(0..count * joint_hidden))?;
        Ok(padded)
    }

    fn decode_rows(
        &self,
        rows: &[f32],
        row_base: usize,
        dstream: &mut RnntDecodeStream,
        hypothesis: &mut RnntHypothesis,
    ) -> Result<()> {
        let projected = self.project_rows(rows)?;
        let row_bases = [row_base, 0, 0, 0];
        self.rnnt.decode_rows(
            &self.encoder,
            dstream,
            &projected,
            std::slice::from_mut(hypothesis),
            &[0],
            &row_bases,
        )
    }

    fn decode_full(
        &self,
        waveform: &[f32],
        dstream: &mut RnntDecodeStream,
        hypothesis: &mut RnntHypothesis,
    ) -> Result<()> {
        let mel = self.frontend.extract(&[waveform], false, true, false)?;
        let valid = mel.mask.iter().filter(|flag| **flag).count();
        let (rows, _, count) = self.encoder.encode(&mel.features, valid)?;
        let width = self.spec.hidden as usize;
        ensure!(
            rows.len() >= count * width,
            "Nemotron full-clip encoder extent"
        );
        self.decode_rows(&rows[..count * width], 0, dstream, hypothesis)
    }

    fn decode_streaming(
        &self,
        waveform: &[f32],
        dstream: &mut RnntDecodeStream,
        hypothesis: &mut RnntHypothesis,
    ) -> Result<()> {
        let first_samples = 4040usize;
        let chunk_samples = 5520usize;
        let stride = 5120usize;
        let mut first = waveform[..waveform.len().min(first_samples)].to_vec();
        first.resize(first_samples, 0.0);
        let mel = self
            .frontend
            .extract(&[first.as_slice()], false, true, false)?;
        ensure!(
            mel.features.len() >= 25 * 128,
            "Nemotron first chunk extent"
        );
        let mut session = self.encoder.stream()?;
        let chunk = self
            .encoder
            .encode_chunk(&mut session, &mel.features[..25 * 128])?;
        self.decode_rows(&chunk.encoder, 0, dstream, hypothesis)?;

        let mut start = 25 * 160 - 256;
        let mut row_base = 4usize;
        while start < waveform.len() {
            let end = start.saturating_add(chunk_samples);
            let last = end >= waveform.len();
            let samples = &waveform[start..end.min(waveform.len())];
            let mel = self.frontend.extract(&[samples], false, false, last)?;
            ensure!(mel.frames <= 32, "Nemotron stream chunk extent");
            let mut features = vec![0.0f32; 32 * 128];
            features[..mel.frames * 128].copy_from_slice(&mel.features);
            let chunk = self.encoder.encode_chunk(&mut session, &features)?;
            self.decode_rows(&chunk.encoder, row_base, dstream, hypothesis)?;
            row_base += 4;
            start = start.saturating_add(stride);
        }
        Ok(())
    }

    pub(crate) fn decode(&self, waveform: &[f32], streaming: bool) -> Result<RnntHypothesis> {
        ensure!(!waveform.is_empty(), "Nemotron input is empty");
        ensure!(
            waveform.iter().all(|value| value.is_finite()),
            "Nemotron input is not finite"
        );
        let mut dstream = self.rnnt.decode_stream(&self.encoder)?;
        let mut hypothesis = RnntHypothesis::default();
        if streaming {
            self.decode_streaming(waveform, &mut dstream, &mut hypothesis)?;
        } else {
            self.decode_full(waveform, &mut dstream, &mut hypothesis)?;
        }
        Ok(hypothesis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use serde_json::Value;
    use std::path::PathBuf;

    fn fixture_pcm(path: &Path) -> Result<Vec<f32>> {
        let pcm = crate::media::read_wav(std::fs::read(path)?.as_slice(), 16000)?;
        ensure!(pcm.len() == 320000, "Nemotron decode PCM sample count");
        Ok(pcm)
    }

    fn check_mode(
        model: &Nemotron,
        directory: &Path,
        pcm: &[f32],
        expected: &Value,
        streaming: bool,
        mode: &str,
    ) -> Result<RnntHypothesis> {
        let hypothesis = model.decode(pcm, streaming)?;
        let expected_tokens: Vec<u32> = expected["tokens"]
            .as_array()
            .context("expected tokens")?
            .iter()
            .map(|value| u32::try_from(value.as_u64().context("token id")?).context("token"))
            .collect::<Result<_>>()?;
        let first_mismatch = hypothesis
            .tokens
            .iter()
            .zip(&expected_tokens)
            .position(|(actual, expected)| actual != expected);
        ensure!(
            hypothesis.tokens == expected_tokens,
            "Nemotron {mode} token mismatch first={first_mismatch:?} actual_len={} expected_len={} actual={:?} expected={:?}",
            hypothesis.tokens.len(),
            expected_tokens.len(),
            first_mismatch.map(|index| hypothesis.tokens[index]),
            first_mismatch.map(|index| expected_tokens[index])
        );
        let expected_emitted: Vec<bool> = expected["emitted"]
            .as_array()
            .context("expected emitted decisions")?
            .iter()
            .map(|value| value.as_bool().context("emitted decision"))
            .collect::<Result<_>>()?;
        ensure!(
            hypothesis.emitted == expected_emitted,
            "Nemotron {mode} blank decision mismatch actual={} expected={}",
            hypothesis.emitted.len(),
            expected_emitted.len()
        );
        let text = crate::tokenizer::decode(directory, &hypothesis.tokens)?;
        ensure!(
            text == expected["text"].as_str().context("expected text")?,
            "Nemotron {mode} text mismatch"
        );
        let margin = hypothesis
            .margins
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::INFINITY, f32::min);
        ensure!(margin.is_finite(), "Nemotron {mode} has no decisions");
        println!(
            "NEMOTRON_DECODE mode={mode} tokens={} decisions={} min_margin={margin:.9e} exact=true",
            hypothesis.tokens.len(),
            hypothesis.margins.len()
        );
        Ok(hypothesis)
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window and publisher decode fixture"]
    fn nemotron_full_and_stream_match_publisher_and_blank_mutant_changes_tokens() -> Result<()> {
        ensure!(
            std::env::var("FF_NEMOTRON_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let directory = PathBuf::from(
            std::env::var_os("FF_NEMOTRON_MODEL").context("FF_NEMOTRON_MODEL missing")?,
        );
        let pcm = fixture_pcm(&PathBuf::from(
            std::env::var_os("FF_NEMOTRON_DECODE_PCM").context("decode PCM missing")?,
        ))?;
        let expected: Value = serde_json::from_slice(&std::fs::read(PathBuf::from(
            std::env::var_os("FF_NEMOTRON_DECODE_FIXTURE").context("decode fixture missing")?,
        ))?)?;
        let device = Device::new(0)?;
        let mut model = Nemotron::load(&directory, &device)?;
        let full = check_mode(&model, &directory, &pcm, &expected["full"], false, "full")?;
        check_mode(
            &model,
            &directory,
            &pcm,
            &expected["stream"],
            true,
            "stream",
        )?;
        model.rnnt.config.blank -= 1;
        let mutant = model.decode(&pcm, false)?;
        ensure!(
            mutant.tokens != full.tokens || mutant.emitted != full.emitted,
            "blank-id mutant did not change Nemotron decisions"
        );
        println!(
            "NEMOTRON_DECODE_CONTROL blank={}->{} tokens_changed=true",
            model.rnnt.config.blank + 1,
            model.rnnt.config.blank
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window and publisher decode fixture"]
    fn nemotron_full_trims_physical_padding() -> Result<()> {
        ensure!(
            std::env::var("FF_NEMOTRON_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let directory = PathBuf::from(
            std::env::var_os("FF_NEMOTRON_MODEL").context("FF_NEMOTRON_MODEL missing")?,
        );
        let pcm = fixture_pcm(&PathBuf::from(
            std::env::var_os("FF_NEMOTRON_DECODE_PCM").context("decode PCM missing")?,
        ))?;
        let expected: Value = serde_json::from_slice(&std::fs::read(PathBuf::from(
            std::env::var_os("FF_NEMOTRON_DECODE_FIXTURE").context("decode fixture missing")?,
        ))?)?;
        let device = Device::new(0)?;
        let model = Nemotron::load(&directory, &device)?;
        check_mode(
            &model,
            &directory,
            &pcm[..15520],
            &expected["boundary"],
            false,
            "boundary",
        )?;
        Ok(())
    }
}
