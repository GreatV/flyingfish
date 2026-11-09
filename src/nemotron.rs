//! Nemotron 3.5 ASR streaming encoder contract. Full-clip lookahead is 3.
use crate::{
    audio::LogMelConfig,
    checkpoint::MappedCheckpoint,
    model_spec::{equal, exact_keys, positive, u32_list},
    parakeet::{self, FastConformer, conv_out},
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

pub struct NemotronSpec {
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    pub conv_kernel: u32,
    pub max_position: u32,
    pub decoder_hidden: u32,
    pub decoder_layers: u32,
    pub vocab: u32,
    pub blank: u32,
    pub prompts: u32,
    pub prompt_intermediate: u32,
    pub mel: LogMelConfig,
}

pub fn subsample_rows(valid: usize) -> Result<usize> {
    let mut rows = valid;
    for _ in 0..3 {
        rows = conv_out(rows, 2, 1)?;
    }
    Ok(rows)
}

impl NemotronSpec {
    pub fn read(directory: &Path) -> Result<Self> {
        let root = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let processor =
            serde_json::from_slice(&std::fs::read(directory.join("processor_config.json"))?)?;
        let generation =
            serde_json::from_slice(&std::fs::read(directory.join("generation_config.json"))?)?;
        let spec = Self::from_json(&root, &processor, &generation)?;
        spec.validate_checkpoint(&MappedCheckpoint::open(directory)?)?;
        Ok(spec)
    }

    pub fn from_json(root: &Value, processor: &Value, generation: &Value) -> Result<Self> {
        exact_keys(
            root,
            &[
                "architectures",
                "blank_token_id",
                "decoder_hidden_size",
                "default_prompt_id",
                "dtype",
                "durations",
                "encoder_config",
                "hidden_act",
                "initializer_range",
                "is_encoder_decoder",
                "max_symbols_per_step",
                "model_type",
                "num_decoder_layers",
                "num_prompts",
                "pad_token_id",
                "prompt_intermediate_size",
                "transformers_version",
                "vocab_size",
            ],
            "Nemotron config",
        )?;
        for (key, value) in [
            (
                "architectures",
                serde_json::json!(["Nemotron3_5AsrForRNNT"]),
            ),
            ("model_type", "nemotron3_5_asr".into()),
            ("dtype", "float32".into()),
            ("hidden_act", "relu".into()),
            ("initializer_range", 0.02.into()),
            ("is_encoder_decoder", true.into()),
            ("transformers_version", "5.13.0.dev0".into()),
            ("durations", serde_json::json!([])),
        ] {
            equal(root, key, value)?;
        }
        ensure!(
            u32_list(root, "durations")?.is_empty(),
            "Nemotron durations are not empty"
        );
        let encoder = root
            .get("encoder_config")
            .context("encoder_config missing")?;
        exact_keys(
            encoder,
            &[
                "activation_dropout",
                "attention_bias",
                "attention_dropout",
                "conv_kernel_size",
                "convolution_bias",
                "default_num_lookahead_tokens",
                "dropout",
                "dropout_positions",
                "hidden_act",
                "hidden_size",
                "initializer_range",
                "intermediate_size",
                "layerdrop",
                "max_position_embeddings",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_mel_bins",
                "scale_input",
                "sliding_window",
                "subsampling_conv_channels",
                "subsampling_conv_kernel_size",
                "subsampling_conv_stride",
                "subsampling_factor",
                "supported_num_lookahead_tokens",
            ],
            "Nemotron encoder config",
        )?;
        for (key, value) in [
            ("model_type", "nemotron_asr_streaming_encoder".into()),
            ("hidden_act", "silu".into()),
            ("initializer_range", 0.02.into()),
            ("attention_bias", false.into()),
            ("convolution_bias", false.into()),
            ("scale_input", false.into()),
            ("activation_dropout", 0.1.into()),
            ("attention_dropout", 0.1.into()),
            ("dropout", 0.1.into()),
            ("dropout_positions", 0.0.into()),
            ("layerdrop", 0.1.into()),
            ("default_num_lookahead_tokens", 3.into()),
            (
                "supported_num_lookahead_tokens",
                serde_json::json!([3, 0, 6, 13]),
            ),
            ("subsampling_conv_kernel_size", 3.into()),
            ("subsampling_conv_stride", 2.into()),
            ("subsampling_factor", 8.into()),
        ] {
            equal(encoder, key, value)?;
        }
        let hidden = positive(encoder, "hidden_size")?;
        let heads = positive(encoder, "num_attention_heads")?;
        let layers = positive(encoder, "num_hidden_layers")?;
        let max_position = positive(encoder, "max_position_embeddings")?;
        let conv_kernel = positive(encoder, "conv_kernel_size")?;
        let sliding = positive(encoder, "sliding_window")?;
        ensure!(
            hidden == 1024
                && heads == 8
                && positive(encoder, "num_key_value_heads")? == heads
                && layers == 24
                && positive(encoder, "intermediate_size")? == 4096
                && positive(encoder, "num_mel_bins")? == 128
                && max_position == 5000
                && positive(encoder, "subsampling_conv_channels")? == 256
                && conv_kernel == 9
                && sliding == 57,
            "unsupported Nemotron encoder geometry"
        );
        let lookahead = u32_list(encoder, "supported_num_lookahead_tokens")?;
        ensure!(
            lookahead == [3, 0, 6, 13],
            "unsupported Nemotron lookahead list"
        );
        let blank = positive(root, "blank_token_id")?;
        let vocab = positive(root, "vocab_size")?;
        let decoder_hidden = positive(root, "decoder_hidden_size")?;
        let decoder_layers = positive(root, "num_decoder_layers")?;
        let prompts = positive(root, "num_prompts")?;
        let prompt_intermediate = positive(root, "prompt_intermediate_size")?;
        equal(root, "pad_token_id", 0.into())?;
        ensure!(
            blank == 13087
                && vocab == 13088
                && decoder_hidden == 640
                && decoder_layers == 2
                && prompts == 128
                && prompt_intermediate == 2048
                && positive(root, "max_symbols_per_step")? == 10
                && positive(root, "default_prompt_id")? == 101,
            "unsupported Nemotron decoder geometry"
        );
        exact_keys(
            processor,
            &[
                "blank_token",
                "default_num_lookahead_tokens",
                "feature_extractor",
                "num_prompts",
                "processor_class",
                "prompt_dictionary",
                "supported_num_lookahead_tokens",
            ],
            "Nemotron processor config",
        )?;
        for (key, value) in [
            ("blank_token", "<blank>".into()),
            ("processor_class", "Nemotron3_5AsrProcessor".into()),
            ("default_num_lookahead_tokens", 3.into()),
            ("num_prompts", 128.into()),
            (
                "supported_num_lookahead_tokens",
                serde_json::json!([3, 0, 6, 13]),
            ),
        ] {
            equal(processor, key, value)?;
        }
        let prompt_dictionary = processor
            .get("prompt_dictionary")
            .context("prompt_dictionary missing")?;
        ensure!(
            prompt_dictionary.as_object().is_some(),
            "prompt_dictionary is not an object"
        );
        equal(prompt_dictionary, "auto", 101.into())?;
        let features = processor
            .get("feature_extractor")
            .context("Nemotron feature_extractor missing")?;
        let mel = parakeet::feature_log_mel(features, "NemotronAsrStreamingFeatureExtractor")?;
        exact_keys(
            generation,
            &[
                "_from_model_config",
                "decoder_start_token_id",
                "output_attentions",
                "output_hidden_states",
                "pad_token_id",
                "transformers_version",
            ],
            "Nemotron generation config",
        )?;
        for (key, value) in [
            ("_from_model_config", true.into()),
            ("decoder_start_token_id", blank.into()),
            ("output_attentions", false.into()),
            ("output_hidden_states", false.into()),
            ("pad_token_id", 0.into()),
            ("transformers_version", "5.13.0.dev0".into()),
        ] {
            equal(generation, key, value)?;
        }
        Ok(Self {
            hidden,
            layers,
            heads,
            conv_kernel,
            max_position,
            decoder_hidden,
            decoder_layers,
            vocab,
            blank,
            prompts,
            prompt_intermediate,
            mel,
        })
    }

    pub fn conformer(&self) -> Result<FastConformer> {
        let lookahead = 3u32;
        let chunk_size = lookahead + 1;
        let left_context = 56u32;
        ensure!(
            left_context.is_multiple_of(chunk_size),
            "sliding-window left context {left_context} does not divide by chunk {chunk_size}"
        );
        let mut frequency = 128usize;
        for _ in 0..3 {
            frequency = conv_out(frequency, 2, 1)?;
        }
        ensure!(frequency == 17, "causal mel frequency {frequency}");
        let linear_in = u32::try_from(256 * frequency)?;
        ensure!(
            self.conv_kernel == 9 && linear_in == 4352,
            "Nemotron conv geometry changed"
        );
        Ok(FastConformer {
            layers: self.layers,
            hidden: self.hidden,
            heads: self.heads,
            conv_kernel: self.conv_kernel,
            attention_bias: false,
            convolution_bias: false,
            prefix: "encoder",
            max_position: self.max_position,
            conv_batch_norm: false,
            pad_left: 2,
            pad_right: 1,
            depthwise_left: self.conv_kernel - 1,
            linear_in,
            chunk_size,
            left_chunks: left_context / chunk_size,
            conv_names: [
                "subsampling.conv_in",
                "subsampling.layers.0.depthwise_conv",
                "subsampling.layers.1.depthwise_conv",
            ],
            pointwise_names: [
                "subsampling.layers.0.pointwise_conv",
                "subsampling.layers.1.pointwise_conv",
            ],
            linear_name: "subsampling.linear",
            attention_names: [
                "q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj", "bias_u", "bias_v",
            ],
            conv_norm_name: "norm",
        })
    }

    fn expected_tensors(&self) -> Result<BTreeMap<String, (Dtype, Vec<usize>)>> {
        let mut expected = BTreeMap::new();
        self.conformer()?.insert_tensors(&mut expected);
        let hidden = self.hidden as usize;
        let decoder = self.decoder_hidden as usize;
        let prompts = self.prompts as usize;
        let intermediate = self.prompt_intermediate as usize;
        let vocab = self.vocab as usize;
        let lstm = decoder * 4;
        let mut put = |name: String, shape: &[usize]| {
            expected.insert(name, (Dtype::F32, shape.to_vec()));
        };
        put(
            "prompt_projector.linear_1.weight".into(),
            &[intermediate, hidden + prompts],
        );
        put("prompt_projector.linear_1.bias".into(), &[intermediate]);
        put(
            "prompt_projector.linear_2.weight".into(),
            &[hidden, intermediate],
        );
        put("prompt_projector.linear_2.bias".into(), &[hidden]);
        put("encoder_projector.weight".into(), &[decoder, hidden]);
        put("encoder_projector.bias".into(), &[decoder]);
        put("decoder.embedding.weight".into(), &[vocab, decoder]);
        for layer in 0..self.decoder_layers {
            put(format!("decoder.lstm.weight_ih_l{layer}"), &[lstm, decoder]);
            put(format!("decoder.lstm.weight_hh_l{layer}"), &[lstm, decoder]);
            put(format!("decoder.lstm.bias_ih_l{layer}"), &[lstm]);
            put(format!("decoder.lstm.bias_hh_l{layer}"), &[lstm]);
        }
        put(
            "decoder.decoder_projector.weight".into(),
            &[decoder, decoder],
        );
        put("decoder.decoder_projector.bias".into(), &[decoder]);
        put("joint.head.weight".into(), &[vocab, decoder]);
        put("joint.head.bias".into(), &[vocab]);
        Ok(expected)
    }

    pub fn validate_checkpoint(&self, checkpoint: &MappedCheckpoint) -> Result<usize> {
        parakeet::require_checkpoint(checkpoint, &self.expected_tensors()?, "Nemotron")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parakeet::position_rows_fit;

    fn published() -> (Value, Value, Value) {
        let root = serde_json::json!({
            "architectures": ["Nemotron3_5AsrForRNNT"],
            "blank_token_id": 13087,
            "decoder_hidden_size": 640,
            "default_prompt_id": 101,
            "dtype": "float32",
            "durations": [],
            "encoder_config": {
                "activation_dropout": 0.1,
                "attention_bias": false,
                "attention_dropout": 0.1,
                "conv_kernel_size": 9,
                "convolution_bias": false,
                "default_num_lookahead_tokens": 3,
                "dropout": 0.1,
                "dropout_positions": 0.0,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "layerdrop": 0.1,
                "max_position_embeddings": 5000,
                "model_type": "nemotron_asr_streaming_encoder",
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 8,
                "num_mel_bins": 128,
                "scale_input": false,
                "sliding_window": 57,
                "subsampling_conv_channels": 256,
                "subsampling_conv_kernel_size": 3,
                "subsampling_conv_stride": 2,
                "subsampling_factor": 8,
                "supported_num_lookahead_tokens": [3, 0, 6, 13]
            },
            "hidden_act": "relu",
            "initializer_range": 0.02,
            "is_encoder_decoder": true,
            "max_symbols_per_step": 10,
            "model_type": "nemotron3_5_asr",
            "num_decoder_layers": 2,
            "num_prompts": 128,
            "pad_token_id": 0,
            "prompt_intermediate_size": 2048,
            "transformers_version": "5.13.0.dev0",
            "vocab_size": 13088
        });
        let processor = serde_json::json!({
            "blank_token": "<blank>",
            "default_num_lookahead_tokens": 3,
            "feature_extractor": {
                "feature_extractor_type": "NemotronAsrStreamingFeatureExtractor",
                "feature_size": 128,
                "hop_length": 160,
                "n_fft": 512,
                "padding_side": "right",
                "padding_value": 0.0,
                "preemphasis": 0.97,
                "return_attention_mask": true,
                "sampling_rate": 16000,
                "win_length": 400
            },
            "num_prompts": 128,
            "processor_class": "Nemotron3_5AsrProcessor",
            "prompt_dictionary": {"auto": 101},
            "supported_num_lookahead_tokens": [3, 0, 6, 13]
        });
        let generation = serde_json::json!({
            "_from_model_config": true,
            "decoder_start_token_id": 13087,
            "output_attentions": false,
            "output_hidden_states": false,
            "pad_token_id": 0,
            "transformers_version": "5.13.0.dev0"
        });
        (root, processor, generation)
    }

    #[test]
    fn nemotron_geometry_is_causal_and_chunked() -> Result<()> {
        let (root, processor, generation) = published();
        let spec = NemotronSpec::from_json(&root, &processor, &generation)?;
        let conformer = spec.conformer()?;
        ensure!(
            conformer.layers == 24
                && conformer.hidden == 1024
                && conformer.heads == 8
                && !conformer.attention_bias
                && !conformer.convolution_bias
                && !conformer.conv_batch_norm
                && conformer.prefix == "encoder"
                && conformer.pad_left == 2
                && conformer.pad_right == 1
                && conformer.depthwise_left == 8
                && conformer.linear_in == 4352
                && conformer.chunk_size == 4
                && conformer.left_chunks == 14
                && conformer.max_position == 5000
                && conformer.conv_names
                    == [
                        "subsampling.conv_in",
                        "subsampling.layers.0.depthwise_conv",
                        "subsampling.layers.1.depthwise_conv",
                    ]
                && conformer.pointwise_names
                    == [
                        "subsampling.layers.0.pointwise_conv",
                        "subsampling.layers.1.pointwise_conv",
                    ],
            "Nemotron conformer description changed"
        );
        ensure!(
            subsample_rows(16)? == 3
                && subsample_rows(64)? == 9
                && subsample_rows(3000)? == 376
                && subsample_rows(2999)? == 376,
            "causal row counts changed"
        );
        ensure!(
            conv_out(conv_out(conv_out(16, 1, 1)?, 1, 1)?, 1, 1)? == 2
                && conv_out(conv_out(conv_out(64, 1, 1)?, 1, 1)?, 1, 1)? == 8
                && conv_out(conv_out(conv_out(3000, 1, 1)?, 1, 1)?, 1, 1)? == 375,
            "centered row counts changed"
        );
        ensure!(
            position_rows_fit(5000, spec.max_position).is_ok()
                && position_rows_fit(5001, spec.max_position).is_err()
                && position_rows_fit(0, spec.max_position).is_err(),
            "position limit accepted the wrong row count"
        );
        let tensors = spec.expected_tensors()?;
        let elements: usize = tensors
            .values()
            .map(|(_, shape)| shape.iter().copied().product::<usize>())
            .sum();
        ensure!(
            tensors.len() == 655 && elements == 637_997_088,
            "tensor contract census changed: tensors={} elements={elements}",
            tensors.len()
        );
        let mut changed = root.clone();
        changed["encoder_config"]["num_hidden_layers"] = 32.into();
        ensure!(
            NemotronSpec::from_json(&changed, &processor, &generation).is_err(),
            "changed encoder layer count accepted"
        );
        changed["encoder_config"]["num_hidden_layers"] = 24.into();
        changed["architectures"] = serde_json::json!(["ParakeetForRNNT"]);
        ensure!(
            NemotronSpec::from_json(&changed, &processor, &generation).is_err(),
            "changed architecture accepted"
        );
        let mut other_extractor = processor.clone();
        other_extractor["feature_extractor"]["feature_extractor_type"] =
            "ParakeetFeatureExtractor".into();
        ensure!(
            NemotronSpec::from_json(&root, &other_extractor, &generation).is_err(),
            "Parakeet feature extractor accepted"
        );
        Ok(())
    }

    #[test]
    #[ignore = "published checkpoint; set FF_NEMOTRON_MODEL"]
    fn nemotron_published_tensor_contract() -> Result<()> {
        let model = std::env::var("FF_NEMOTRON_MODEL").context("FF_NEMOTRON_MODEL missing")?;
        let directory = Path::new(&model);
        let spec = NemotronSpec::read(directory)?;
        let tensors = spec.validate_checkpoint(&MappedCheckpoint::open(directory)?)?;
        ensure!(tensors == 655, "published tensor count changed");
        println!("PASS NEMOTRON_TENSOR_CONTRACT tensors=655 dtypes=native shapes=exact");
        Ok(())
    }
}
