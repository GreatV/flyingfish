use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::backend::{Breeze, BreezeTokenizer, TextPlant};
use crate::{
    model_spec::{equal, positive},
    sampling::Sampling,
};

#[derive(Clone, Debug)]
pub struct BreezeSpeech {
    pub samples: Vec<f32>,
    pub sampling_rate: usize,
}

#[derive(Clone, Copy)]
struct GenerationConfig {
    sampling: Sampling,
    max_new_tokens: usize,
}

fn plain_prompt(text: &str) -> String {
    format!("[S0]{text}")
}

fn generation_config_value(root: Value) -> Result<GenerationConfig> {
    let allowed = [
        "bos_token_id",
        "depth_decoder_do_sample",
        "depth_decoder_temperature",
        "do_sample",
        "eos_token_id",
        "max_new_tokens",
        "pad_token_id",
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
        "transformers_version",
    ];
    let object = root.as_object().context("Breeze generation config")?;
    for key in object.keys() {
        ensure!(
            allowed.contains(&key.as_str()),
            "unknown Breeze generation key {key}"
        );
    }
    equal(&root, "do_sample", true.into())?;
    equal(&root, "depth_decoder_do_sample", true.into())?;
    equal(&root, "bos_token_id", 2.into())?;
    equal(&root, "eos_token_id", json!([1]))?;
    equal(&root, "pad_token_id", 0.into())?;
    let repetition_penalty = root
        .get("repetition_penalty")
        .map_or(Some(1.1), Value::as_f64)
        .context("Breeze repetition penalty")?;
    let temperature = root["temperature"].as_f64().context("Breeze temperature")?;
    let depth_temperature = root["depth_decoder_temperature"]
        .as_f64()
        .context("Breeze depth temperature")?;
    ensure!(
        temperature == depth_temperature,
        "Breeze backbone/depth temperatures differ"
    );
    let top_k = root
        .get("top_k")
        .map_or(Some(50), Value::as_u64)
        .context("Breeze top-k")?;
    let top_p = root
        .get("top_p")
        .map_or(Some(1.0), Value::as_f64)
        .context("Breeze top-p")?;
    let max_new_tokens = positive(&root, "max_new_tokens")? as usize;
    ensure!(
        max_new_tokens <= 750,
        "Breeze generation limit {max_new_tokens} exceeds 750"
    );
    Ok(GenerationConfig {
        sampling: Sampling::new(
            temperature as f32,
            top_k as usize,
            top_p as f32,
            repetition_penalty as f32,
        )?,
        max_new_tokens,
    })
}

#[cfg(feature = "cuda")]
fn generation_config(directory: &Path) -> Result<GenerationConfig> {
    generation_config_value(serde_json::from_slice(
        &std::fs::read(directory.join("generation_config.json"))
            .context("read Breeze generation_config.json")?,
    )?)
}

#[cfg(feature = "cuda")]
pub fn speak(
    directory: &Path,
    text: &str,
    seed: u64,
    options: crate::RunOptions,
) -> Result<BreezeSpeech> {
    let config = generation_config(directory)?;
    let prompt = plain_prompt(text);
    let ids = crate::tokenizer::encode(directory, &prompt)?;
    let mut model = Breeze::load_for_generation(
        directory,
        &options.runtime_dir,
        options.device,
        ids.len(),
        config.max_new_tokens,
    )?;
    let projected = model.text(&ids, TextPlant::Nominal)?;
    let codes = model.sample_decode(&projected, ids.len(), config.sampling, seed)?;
    let frames = codes.len() / 16;
    drop(model);
    let device = crate::backend::cuda::Device::new(options.device)?;
    let mut codec = BreezeTokenizer::load_for_frames(
        &directory.join("audio_tokenizer/model.safetensors"),
        &device,
        frames,
    )?;
    let samples = codec.decode(&codes)?;
    ensure!(
        samples.len() == frames * 1920 && samples.iter().all(|value| value.is_finite()),
        "Breeze waveform extent"
    );
    Ok(BreezeSpeech {
        samples,
        sampling_rate: 24_000,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_prompt_uses_the_published_speaker_prefix() {
        assert_eq!(plain_prompt("Hello"), "[S0]Hello");
        assert_eq!(plain_prompt("[笑] 欢迎"), "[S0][笑] 欢迎");
    }

    #[test]
    #[ignore = "requires the Breeze-TTS-2 checkpoint"]
    fn plain_prompt_token_ids_match_the_published_fixture() -> Result<()> {
        let directory = std::env::var_os("N8_WEIGHTS").context("N8_WEIGHTS missing")?;
        let ids = crate::tokenizer::encode(Path::new(&directory), &plain_prompt("Hello"))?;
        assert_eq!(ids, [2, 262146, 9259]);
        Ok(())
    }

    #[test]
    fn generation_config_uses_published_sampling_and_defaults() -> Result<()> {
        let root = json!({
            "bos_token_id": 2,
            "depth_decoder_do_sample": true,
            "depth_decoder_temperature": 0.9,
            "do_sample": true,
            "eos_token_id": [1],
            "max_new_tokens": 750,
            "pad_token_id": 0,
            "temperature": 0.9,
            "transformers_version": "4.57.3"
        });
        let config = generation_config_value(root)?;
        assert_eq!(config.max_new_tokens, 750);
        assert_eq!(config.sampling.temperature, 0.9);
        assert_eq!(config.sampling.top_k, 50);
        assert_eq!(config.sampling.top_p, 1.0);
        assert_eq!(config.sampling.repetition_penalty, 1.1);
        Ok(())
    }
}
