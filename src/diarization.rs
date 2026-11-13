//! F32 frame-classification geometry and arrival-order speaker-cache configuration.
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine_diarization::Diarization;

use crate::{
    audio::LogMelConfig,
    checkpoint::MappedCheckpoint,
    model_spec::{equal, exact_keys, positive},
    parakeet,
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy)]
pub struct ChunkConfig {
    pub frames: u32,
    pub right: u32,
    pub fifo: u32,
    pub update: u32,
}

pub struct SpeakerCacheConfig {
    pub frames: u32,
    pub silence: u32,
    pub fifo: u32,
    pub update: u32,
    pub threshold: f32,
    pub latest_boost: f32,
    pub positive: u32,
    pub strong: u32,
    pub weak: u32,
}

pub struct DiarizationSpec {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub heads: u32,
    pub head_hidden: u32,
    pub speakers: u32,
    pub factor: u32,
    pub max_positions: u32,
    pub rope_theta: f32,
    pub mel: LogMelConfig,
    pub offline: ChunkConfig,
    pub cache: SpeakerCacheConfig,
    pub modes: BTreeMap<String, [u32; 2]>,
}

pub const DIARIZATION_THRESHOLD: f64 = 0.5;

pub fn speaker_segments(
    logits: &[f32],
    speakers: usize,
    attention_mask: Option<&[bool]>,
    threshold: f64,
    frame_seconds: f64,
) -> Result<Vec<crate::media::TextSegment>> {
    ensure!(speakers > 0, "speaker count must be positive");
    ensure!(
        !logits.is_empty() && logits.len().is_multiple_of(speakers),
        "diarization logits extent"
    );
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "non-finite logit"
    );
    ensure!(
        threshold.is_finite() && (0.0..=1.0).contains(&threshold),
        "invalid diarization threshold"
    );
    ensure!(
        frame_seconds.is_finite() && frame_seconds > 0.0,
        "invalid frame duration"
    );
    let frames = logits.len() / speakers;
    if let Some(mask) = attention_mask {
        ensure!(mask.len() == frames, "diarization mask extent");
    }
    let mut segments = Vec::new();
    for speaker in 0..speakers {
        let mut start = None;
        for index in 0..=frames {
            let active = index < frames && {
                let logit = logits[index * speakers + speaker];
                let valid = attention_mask.is_none_or(|mask| mask[index]);
                valid && 1.0 / (1.0 + (-f64::from(logit)).exp()) > threshold
            };
            match (start, active) {
                (None, true) => start = Some(index),
                (Some(begin), false) => {
                    let round = |frame: usize| {
                        let seconds = frame as f64 * frame_seconds;
                        (seconds * 100.0).round_ties_even() / 100.0
                    };
                    segments.push((round(begin), speaker, round(index)));
                    start = None;
                }
                _ => {}
            }
        }
    }
    segments.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    Ok(segments
        .into_iter()
        .map(|(start, speaker, end)| crate::media::TextSegment {
            start,
            end,
            text: format!("speaker_{speaker}"),
        })
        .collect())
}

#[cfg(feature = "cuda")]
pub fn stream_frontend_step(
    engine: &Diarization,
    wave: &[f32],
    step: usize,
    config: ChunkConfig,
    mel_index: usize,
) -> Result<(Vec<f32>, Vec<bool>, bool)> {
    let mel = engine.spec().mel;
    let factor = engine.spec().factor as usize;
    let hop = mel.hop_length as usize;
    let win = mel.win_length as usize;
    let n_fft = mel.n_fft as usize;
    let mel_bins = mel.mel_bins as usize;
    let chunk_mel = (config.frames as usize + config.right as usize) * factor;
    let first_samples = (chunk_mel - 1) * hop + win / 2;
    let later_samples = chunk_mel * hop + win;
    let start = if step == 0 {
        0
    } else {
        mel_index
            .checked_mul(hop)
            .and_then(|value| value.checked_sub(n_fft / 2))
            .context("streaming start underflow")?
    };
    let size = if step == 0 {
        first_samples
    } else {
        later_samples
    };
    let end = (start + size).min(wave.len());
    ensure!(
        start <= end && end <= wave.len(),
        "streaming waveform extent"
    );
    let last = end >= wave.len();
    let batch = engine.audio_features(&wave[start..end], step == 0, last)?;
    let valid = batch.mask.iter().filter(|flag| **flag).count();
    ensure!(
        valid * mel_bins <= batch.features.len() && valid <= batch.mask.len(),
        "streaming frontend extent"
    );
    Ok((
        batch.features[..valid * mel_bins].to_vec(),
        batch.mask[..valid].to_vec(),
        last,
    ))
}

#[cfg(any(feature = "cuda", test))]
fn emitted_stream_mask<'a>(
    mask: &'a [bool],
    logits: &[f32],
    speakers: usize,
) -> Result<&'a [bool]> {
    ensure!(
        speakers > 0 && logits.len().is_multiple_of(speakers),
        "diarization streaming logit extent"
    );
    let frames = logits.len() / speakers;
    ensure!(frames <= mask.len(), "diarization streaming mask extent");
    Ok(&mask[..frames])
}

#[cfg(feature = "cuda")]
pub fn diarize(
    directory: &Path,
    waveform: &[f32],
    stream: bool,
    threshold: f64,
    options: crate::RunOptions,
) -> Result<Vec<crate::media::TextSegment>> {
    let spec = DiarizationSpec::read(directory)?;
    let device = crate::backend::cuda::Device::new(options.device)?;
    let mut engine = Diarization::load(directory, &device)?;
    let frame_seconds = f64::from(spec.mel.hop_length) / f64::from(spec.mel.sampling_rate);
    if stream {
        let mode = spec
            .modes
            .get("low_latency")
            .context("diarization low_latency mode")?;
        let config = ChunkConfig {
            frames: mode[0],
            right: mode[1],
            fifo: spec.cache.fifo,
            update: spec.cache.update,
        };
        engine.configure(0, config)?;
        let factor = spec.factor as usize;
        let step_mel = config.frames as usize * factor;
        let mut logits = Vec::new();
        let mut mask = Vec::new();
        for step in 0.. {
            let (features, valid, last) =
                stream_frontend_step(&engine, waveform, step, config, step * step_mel)?;
            let (step_logits, _) = engine.stream_step(&features, &valid, config, last, 0)?;
            mask.extend_from_slice(emitted_stream_mask(
                &valid,
                &step_logits,
                spec.speakers as usize,
            )?);
            logits.extend(step_logits);
            if last {
                break;
            }
        }
        speaker_segments(
            &logits,
            spec.speakers as usize,
            Some(&mask),
            threshold,
            frame_seconds,
        )
    } else {
        let batch = engine.audio_features(waveform, false, false)?;
        let (logits, _) = engine.offline(&batch.features, &batch.mask, 0)?;
        speaker_segments(
            &logits,
            spec.speakers as usize,
            Some(&batch.mask),
            threshold,
            frame_seconds,
        )
    }
}

impl DiarizationSpec {
    pub fn read(directory: &Path) -> Result<Self> {
        let root = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let processor =
            serde_json::from_slice(&std::fs::read(directory.join("processor_config.json"))?)?;
        let spec = Self::from_json(&root, &processor)?;
        spec.validate_checkpoint(&MappedCheckpoint::open(directory)?)?;
        Ok(spec)
    }

    pub fn from_json(root: &Value, processor: &Value) -> Result<Self> {
        exact_keys(
            root,
            &[
                "architectures",
                "audio_config",
                "chunk_length",
                "chunk_right_context",
                "dtype",
                "fifo_length",
                "head_config",
                "initializer_range",
                "model_type",
                "speaker_cache_update_period",
                "streaming_config",
                "transformers_version",
            ],
            "diarization config",
        )?;
        for (key, value) in [
            (
                "architectures",
                json!(["Nemotron3DiarizationForAudioFrameClassification"]),
            ),
            ("dtype", json!("float32")),
            ("model_type", json!("nemotron3_diarization")),
            ("initializer_range", json!(0.02)),
            ("transformers_version", json!("5.18.0.dev0")),
        ] {
            equal(root, key, value)?;
        }
        let audio = &root["audio_config"];
        exact_keys(
            audio,
            &[
                "attention_dropout",
                "hidden_act",
                "hidden_size",
                "initializer_range",
                "intermediate_size",
                "max_position_embeddings",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_mel_bins",
                "partial_rotary_factor",
                "rope_parameters",
                "subsampling_factor",
            ],
            "diarization encoder",
        )?;
        for (key, value) in [
            ("attention_dropout", json!(0.0)),
            ("hidden_act", json!("gelu")),
            ("initializer_range", json!(0.02)),
            ("model_type", json!("nemotron3_diarization_audio")),
            ("partial_rotary_factor", json!(1.0)),
            (
                "rope_parameters",
                json!({"partial_rotary_factor":1.0,"rope_theta":10000.0,"rope_type":"default"}),
            ),
        ] {
            equal(audio, key, value)?;
        }
        let hidden = positive(audio, "hidden_size")?;
        let intermediate = positive(audio, "intermediate_size")?;
        let layers = positive(audio, "num_hidden_layers")?;
        let heads = positive(audio, "num_attention_heads")?;
        let factor = positive(audio, "subsampling_factor")?;
        let max_positions = positive(audio, "max_position_embeddings")?;
        ensure!(
            hidden == 512
                && intermediate == 2048
                && layers == 31
                && heads == 8
                && positive(audio, "num_key_value_heads")? == heads
                && positive(audio, "num_mel_bins")? == 128
                && factor == 8
                && max_positions == 5000,
            "unsupported diarization encoder geometry"
        );
        let head = &root["head_config"];
        ensure!(
            head == &json!({"audio_hidden_size":hidden,"hidden_size":192,"model_type":"",
            "num_speakers":8,"subsampling_factor":factor}),
            "unsupported diarization head"
        );
        let cache = &root["streaming_config"];
        ensure!(
            cache
                == &json!({"fifo_length":264,"latest_frames_score_boost":0.05,
            "min_positive_scores_rate":0.5,"model_type":"","num_speakers":8,
            "prediction_score_threshold":0.25,"speaker_cache_length":264,
            "speaker_cache_silence_frames_per_speaker":1,"speaker_cache_update_period":222,
            "strong_boost_rate":0.75,"subsampling_factor":factor,"weak_boost_rate":1.5}),
            "unsupported diarization speaker-cache policy"
        );
        let offline = ChunkConfig {
            frames: positive(root, "chunk_length")?,
            right: positive(root, "chunk_right_context")?,
            fifo: positive(root, "fifo_length")?,
            update: positive(root, "speaker_cache_update_period")?,
        };
        ensure!(
            offline.frames == 340
                && offline.right == 40
                && offline.fifo == 40
                && offline.update == 300,
            "unsupported diarization offline window"
        );
        exact_keys(
            processor,
            &[
                "feature_extractor",
                "processor_class",
                "streaming_mode",
                "streaming_modes",
                "subsampling_factor",
            ],
            "diarization processor",
        )?;
        equal(
            processor,
            "processor_class",
            json!("Nemotron3DiarizationProcessor"),
        )?;
        equal(processor, "streaming_mode", json!("low_latency"))?;
        equal(processor, "subsampling_factor", json!(factor))?;
        equal(
            processor,
            "streaming_modes",
            json!({"low_latency":[9,4],
            "very_low_latency":[6,2],"ultra_low_latency":[3,1]}),
        )?;
        let modes = serde_json::from_value(processor["streaming_modes"].clone())?;
        let mel = parakeet::feature_log_mel(
            &processor["feature_extractor"],
            "NemotronAsrStreamingFeatureExtractor",
        )?;
        let frames = positive(cache, "speaker_cache_length")?;
        let silence = positive(cache, "speaker_cache_silence_frames_per_speaker")?;
        let speakers = positive(head, "num_speakers")?;
        let budget = frames / speakers - silence;
        let spec = Self {
            hidden,
            intermediate,
            layers,
            heads,
            factor,
            max_positions,
            head_hidden: positive(head, "hidden_size")?,
            speakers,
            mel,
            offline,
            modes,
            rope_theta: audio["rope_parameters"]["rope_theta"]
                .as_f64()
                .context("rope theta")? as f32,
            cache: SpeakerCacheConfig {
                frames,
                silence,
                fifo: positive(cache, "fifo_length")?,
                update: positive(cache, "speaker_cache_update_period")?,
                threshold: cache["prediction_score_threshold"]
                    .as_f64()
                    .context("cache threshold")? as f32,
                latest_boost: cache["latest_frames_score_boost"]
                    .as_f64()
                    .context("cache boost")? as f32,
                positive: (f64::from(budget)
                    * cache["min_positive_scores_rate"]
                        .as_f64()
                        .context("positive rate")?)
                .floor() as u32,
                strong: (f64::from(budget)
                    * cache["strong_boost_rate"].as_f64().context("strong rate")?)
                .floor() as u32,
                weak: (f64::from(budget)
                    * cache["weak_boost_rate"].as_f64().context("weak rate")?)
                .floor() as u32,
            },
        };
        ensure!(
            spec.max_encoder_rows()? <= spec.max_positions,
            "diarization window exceeds rotary positions"
        );
        Ok(spec)
    }

    pub fn max_encoder_rows(&self) -> Result<u32> {
        let mut rows = self
            .cache
            .frames
            .checked_add(self.offline.fifo)
            .and_then(|n| n.checked_add(self.offline.frames))
            .and_then(|n| n.checked_add(self.offline.right))
            .context("offline window overflow")?;
        for &[chunk, right] in self.modes.values() {
            let n = self
                .cache
                .frames
                .checked_add(self.cache.fifo)
                .and_then(|n| n.checked_add(chunk))
                .and_then(|n| n.checked_add(right))
                .context("streaming window overflow")?;
            rows = rows.max(n);
        }
        Ok(rows)
    }

    pub(crate) fn expected_tensors(&self) -> BTreeMap<String, (Dtype, Vec<usize>)> {
        let h = self.hidden as usize;
        let i = self.intermediate as usize;
        let c = self.head_hidden as usize;
        let r = self.factor as usize;
        let mut expected = BTreeMap::new();
        let mut put = |name: String, shape: &[usize]| {
            expected.insert(name, (Dtype::F32, shape.to_vec()));
        };
        for layer in 0..self.layers {
            let prefix = format!("model.audio_tower.layers.{layer}");
            for norm in ["layer_norm1", "layer_norm2"] {
                for suffix in ["weight", "bias"] {
                    put(format!("{prefix}.{norm}.{suffix}"), &[h]);
                }
            }
            for qkv in ["q_proj", "k_proj", "v_proj"] {
                put(format!("{prefix}.self_attn.{qkv}.weight"), &[h, h]);
            }
            put(format!("{prefix}.self_attn.o_proj.weight"), &[h, h]);
            put(format!("{prefix}.self_attn.o_proj.bias"), &[h]);
            put(format!("{prefix}.mlp.fc1.weight"), &[i, h]);
            put(format!("{prefix}.mlp.fc1.bias"), &[i]);
            put(format!("{prefix}.mlp.fc2.weight"), &[h, i]);
            put(format!("{prefix}.mlp.fc2.bias"), &[h]);
        }
        put(
            "model.audio_tower.embedder.projection.weight".into(),
            &[h, r * self.mel.mel_bins as usize],
        );
        for norm in ["input_layer_norm", "layer_norm"] {
            for suffix in ["weight", "bias"] {
                put(format!("model.audio_tower.{norm}.{suffix}"), &[h]);
            }
        }
        put("model.proj.weight".into(), &[c, h]);
        put("model.proj.bias".into(), &[c]);
        put("model.upsampler.conv.weight".into(), &[r * c, c, 3]);
        put("model.upsampler.conv.bias".into(), &[r * c]);
        put("classifier.dense.weight".into(), &[c, c]);
        put("classifier.dense.bias".into(), &[c]);
        put(
            "classifier.out_proj.weight".into(),
            &[self.speakers as usize, c],
        );
        put("classifier.out_proj.bias".into(), &[self.speakers as usize]);
        put("silence_embeds".into(), &[h]);
        expected
    }

    pub fn validate_checkpoint(&self, checkpoint: &MappedCheckpoint) -> Result<usize> {
        parakeet::require_checkpoint(checkpoint, &self.expected_tensors(), "diarization")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn speaker_segments_preserve_overlap_mask_and_default_threshold() -> Result<()> {
        let logits = [10.0, -10.0, 10.0, 10.0, -10.0, 10.0, -10.0, 10.0];
        let mask = [true, true, true, false];
        let segments = speaker_segments(&logits, 2, Some(&mask), DIARIZATION_THRESHOLD, 0.01)?;
        assert_eq!(segments.len(), 2);
        assert_eq!((segments[0].start, segments[0].end), (0.0, 0.02));
        assert_eq!((segments[1].start, segments[1].end), (0.01, 0.03));
        assert_eq!(segments[0].text, "speaker_0");
        assert_eq!(segments[1].text, "speaker_1");
        let mut rttm = Vec::new();
        crate::media::write_rttm(&mut rttm, "clip", &segments)?;
        assert_eq!(
            String::from_utf8(rttm)?,
            "SPEAKER clip 1 0.000 0.020 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER clip 1 0.010 0.020 <NA> <NA> speaker_1 <NA> <NA>\n"
        );
        assert_eq!(DIARIZATION_THRESHOLD, 0.5);
        assert!(speaker_segments(&[0.0], 1, None, DIARIZATION_THRESHOLD, 0.01)?.is_empty());
        Ok(())
    }

    #[test]
    fn streaming_mask_tracks_emitted_frames_without_right_context() -> Result<()> {
        let mask = [true; 104];
        let emitted = vec![0.0; 72 * 8];
        assert_eq!(emitted_stream_mask(&mask, &emitted, 8)?.len(), 72);
        let final_logits = vec![0.0; 32 * 8];
        let mut accumulated = emitted_stream_mask(&mask, &emitted, 8)?.to_vec();
        accumulated.extend_from_slice(emitted_stream_mask(&mask[72..], &final_logits, 8)?);
        assert_eq!(accumulated.len(), 104);
        let too_many = vec![0.0; 105 * 8];
        assert!(emitted_stream_mask(&mask, &too_many, 8).is_err());
        Ok(())
    }

    #[test]
    fn published_layout_rejects_same_size_convolution_axis_swap() -> Result<()> {
        let config = json!({
            "architectures":["Nemotron3DiarizationForAudioFrameClassification"],
            "audio_config":{
                "attention_dropout":0.0,"hidden_act":"gelu","hidden_size":512,
                "initializer_range":0.02,"intermediate_size":2048,"max_position_embeddings":5000,
                "model_type":"nemotron3_diarization_audio","num_attention_heads":8,
                "num_hidden_layers":31,"num_key_value_heads":8,"num_mel_bins":128,
                "partial_rotary_factor":1.0,"rope_parameters":{
                    "partial_rotary_factor":1.0,"rope_theta":10000.0,"rope_type":"default"
                },"subsampling_factor":8
            },
            "chunk_length":340,"chunk_right_context":40,"dtype":"float32","fifo_length":40,
            "head_config":{"audio_hidden_size":512,"hidden_size":192,"model_type":"",
                "num_speakers":8,"subsampling_factor":8},
            "initializer_range":0.02,"model_type":"nemotron3_diarization",
            "speaker_cache_update_period":300,
            "streaming_config":{"fifo_length":264,"latest_frames_score_boost":0.05,
                "min_positive_scores_rate":0.5,"model_type":"","num_speakers":8,
                "prediction_score_threshold":0.25,"speaker_cache_length":264,
                "speaker_cache_silence_frames_per_speaker":1,"speaker_cache_update_period":222,
                "strong_boost_rate":0.75,"subsampling_factor":8,"weak_boost_rate":1.5},
            "transformers_version":"5.18.0.dev0"
        });
        let processor = json!({
            "feature_extractor":{"feature_extractor_type":"NemotronAsrStreamingFeatureExtractor",
                "feature_size":128,"hop_length":160,"n_fft":512,"padding_side":"right",
                "padding_value":0.0,"preemphasis":0.97,"return_attention_mask":true,
                "sampling_rate":16000,"win_length":400},
            "processor_class":"Nemotron3DiarizationProcessor","streaming_mode":"low_latency",
            "streaming_modes":{"low_latency":[9,4],"ultra_low_latency":[3,1],"very_low_latency":[6,2]},
            "subsampling_factor":8
        });
        let spec = DiarizationSpec::from_json(&config, &processor)?;
        let directory = std::env::temp_dir().join(format!(
            "flyingfish-diarization-layout-{}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let write = |swapped: bool| -> Result<()> {
            let mut header = serde_json::Map::new();
            let mut offset = 0usize;
            for (name, (_, mut shape)) in spec.expected_tensors() {
                if swapped && name == "model.upsampler.conv.weight" {
                    shape.swap(1, 2);
                }
                let bytes = shape.iter().product::<usize>() * 4;
                header.insert(
                    name,
                    json!({"dtype":"F32","shape":shape,"data_offsets":[offset,offset+bytes]}),
                );
                offset += bytes;
            }
            let mut header = serde_json::to_vec(&header)?;
            header.resize(header.len().next_multiple_of(8), b' ');
            let mut file = std::fs::File::create(directory.join("model.safetensors"))?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            file.set_len((8 + header.len() + offset) as u64)?;
            Ok(())
        };
        write(false)?;
        let checkpoint = MappedCheckpoint::open(&directory)?;
        assert_eq!(spec.validate_checkpoint(&checkpoint)?, 417);
        drop(checkpoint);
        write(true)?;
        let checkpoint = MappedCheckpoint::open(&directory)?;
        assert!(spec.validate_checkpoint(&checkpoint).is_err());
        drop(checkpoint);
        std::fs::remove_file(directory.join("model.safetensors"))?;
        std::fs::remove_dir(directory)?;
        Ok(())
    }
}
