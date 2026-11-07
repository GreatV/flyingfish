//! Canary published configuration and 1507-tensor contract.
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine_canary::Canary;

use crate::{
    audio::LogMelConfig,
    checkpoint::MappedCheckpoint,
    model_spec::{equal, exact_keys, positive},
    parakeet::{self, FastConformer},
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

pub struct CanarySpec {
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    pub conv_kernel: u32,
    pub attention_bias: bool,
    pub convolution_bias: bool,
    pub vocab: u32,
    pub decoder_layers: u32,
    pub mel: LogMelConfig,
}

impl CanarySpec {
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
                "bos_token_id",
                "decoder_config",
                "decoder_start_token_id",
                "dtype",
                "encoder_config",
                "eos_token_id",
                "initializer_range",
                "is_encoder_decoder",
                "model_type",
                "pad_token_id",
                "tie_word_embeddings",
                "transformers_version",
                "use_cache",
                "vocab_size",
            ],
            "Canary config",
        )?;
        for (key, value) in [
            (
                "architectures",
                serde_json::json!(["CanaryForConditionalGeneration"]),
            ),
            ("model_type", "canary".into()),
            ("dtype", "float32".into()),
            ("initializer_range", 0.02.into()),
            ("is_encoder_decoder", true.into()),
            ("tie_word_embeddings", true.into()),
            ("bos_token_id", 4.into()),
            ("eos_token_id", 3.into()),
            ("pad_token_id", 2.into()),
            ("decoder_start_token_id", 7.into()),
            ("use_cache", true.into()),
        ] {
            equal(root, key, value)?;
        }
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
                "subsampling_conv_channels",
                "subsampling_conv_kernel_size",
                "subsampling_conv_stride",
                "subsampling_factor",
            ],
            "Canary encoder config",
        )?;
        for (key, value) in [
            ("model_type", "parakeet_encoder".into()),
            ("hidden_act", "silu".into()),
            ("initializer_range", 0.02.into()),
            ("attention_bias", true.into()),
            ("convolution_bias", true.into()),
            ("scale_input", false.into()),
            ("activation_dropout", 0.1.into()),
            ("attention_dropout", 0.1.into()),
            ("dropout", 0.1.into()),
            ("dropout_positions", 0.0.into()),
            ("layerdrop", 0.1.into()),
            ("subsampling_conv_kernel_size", 3.into()),
            ("subsampling_conv_stride", 2.into()),
            ("subsampling_factor", 8.into()),
        ] {
            equal(encoder, key, value)?;
        }
        let hidden = positive(encoder, "hidden_size")?;
        let heads = positive(encoder, "num_attention_heads")?;
        let layers = positive(encoder, "num_hidden_layers")?;
        ensure!(
            hidden == 1024
                && heads == 8
                && positive(encoder, "num_key_value_heads")? == heads
                && layers == 32
                && positive(encoder, "intermediate_size")? == 4096
                && positive(encoder, "num_mel_bins")? == 128
                && positive(encoder, "max_position_embeddings")? == 5000
                && positive(encoder, "subsampling_conv_channels")? == 256
                && positive(encoder, "conv_kernel_size")? == 9,
            "unsupported Canary encoder geometry"
        );
        let attention_bias = encoder["attention_bias"]
            .as_bool()
            .context("invalid attention_bias")?;
        let convolution_bias = encoder["convolution_bias"]
            .as_bool()
            .context("invalid convolution_bias")?;
        let decoder = root
            .get("decoder_config")
            .context("decoder_config missing")?;
        exact_keys(
            decoder,
            &[
                "attention_bias",
                "attention_dropout",
                "bos_token_id",
                "eos_token_id",
                "head_dim",
                "hidden_act",
                "hidden_size",
                "initializer_range",
                "intermediate_size",
                "max_position_embeddings",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "pad_token_id",
                "use_cache",
                "vocab_size",
            ],
            "Canary decoder config",
        )?;
        for (key, value) in [
            ("model_type", "canary_decoder".into()),
            ("attention_bias", true.into()),
            ("hidden_act", "relu".into()),
            ("initializer_range", 0.02.into()),
            ("attention_dropout", 0.1.into()),
            ("bos_token_id", 4.into()),
            ("eos_token_id", 3.into()),
            ("pad_token_id", 2.into()),
            ("use_cache", true.into()),
        ] {
            equal(decoder, key, value)?;
        }
        let vocab = positive(root, "vocab_size")?;
        let decoder_layers = positive(decoder, "num_hidden_layers")?;
        ensure!(
            vocab == 16384
                && positive(decoder, "vocab_size")? == vocab
                && positive(decoder, "hidden_size")? == hidden
                && positive(decoder, "intermediate_size")? == 4096
                && positive(decoder, "head_dim")? == 128
                && positive(decoder, "num_attention_heads")? == 8
                && positive(decoder, "num_key_value_heads")? == 8
                && decoder_layers == 8
                && positive(decoder, "max_position_embeddings")? == 1024,
            "unsupported Canary decoder geometry"
        );
        exact_keys(
            processor,
            &["feature_extractor", "processor_class"],
            "Canary processor config",
        )?;
        equal(processor, "processor_class", "CanaryProcessor".into())?;
        let features = processor
            .get("feature_extractor")
            .context("Canary feature_extractor missing")?;
        let mel = parakeet::feature_log_mel(features, "ParakeetFeatureExtractor")?;
        exact_keys(
            generation,
            &[
                "_from_model_config",
                "bos_token_id",
                "decoder_start_token_id",
                "eos_token_id",
                "output_attentions",
                "output_hidden_states",
                "pad_token_id",
                "transformers_version",
                "use_cache",
            ],
            "Canary generation config",
        )?;
        for (key, value) in [
            ("_from_model_config", true.into()),
            ("bos_token_id", 4.into()),
            ("decoder_start_token_id", 7.into()),
            ("eos_token_id", 3.into()),
            ("pad_token_id", 2.into()),
            ("output_attentions", false.into()),
            ("output_hidden_states", false.into()),
            ("use_cache", true.into()),
        ] {
            equal(generation, key, value)?;
        }
        Ok(Self {
            hidden,
            layers,
            heads,
            conv_kernel: 9,
            attention_bias,
            convolution_bias,
            vocab,
            decoder_layers,
            mel,
        })
    }

    pub fn conformer(&self) -> FastConformer {
        FastConformer {
            layers: self.layers,
            hidden: self.hidden,
            heads: self.heads,
            conv_kernel: self.conv_kernel,
            attention_bias: self.attention_bias,
            convolution_bias: self.convolution_bias,
            prefix: "model.encoder",
            max_position: 5000,
            conv_batch_norm: true,
            pad_left: 1,
            pad_right: 1,
            depthwise_left: self.conv_kernel / 2,
            linear_in: 4096,
            chunk_size: 0,
            left_chunks: 0,
            conv_names: [
                "subsampling.layers.0",
                "subsampling.layers.2",
                "subsampling.layers.5",
            ],
            pointwise_names: ["subsampling.layers.3", "subsampling.layers.6"],
        }
    }

    pub fn validate_checkpoint(&self, checkpoint: &MappedCheckpoint) -> Result<usize> {
        let mut expected = BTreeMap::new();
        self.conformer().insert_tensors(&mut expected);
        let h = self.hidden as usize;
        let inter = 4096usize;
        let vocab = self.vocab as usize;
        let put = |expected: &mut BTreeMap<String, (Dtype, Vec<usize>)>,
                   name: String,
                   dtype: Dtype,
                   shape: &[usize]| {
            expected.insert(name, (dtype, shape.to_vec()));
        };
        let weight_bias = |expected: &mut BTreeMap<String, (Dtype, Vec<usize>)>,
                           name: &str,
                           weight: &[usize],
                           bias: &[usize]| {
            put(expected, format!("{name}.weight"), Dtype::F32, weight);
            put(expected, format!("{name}.bias"), Dtype::F32, bias);
        };
        put(
            &mut expected,
            "model.decoder.embed_tokens.weight".into(),
            Dtype::F32,
            &[vocab, h],
        );
        weight_bias(
            &mut expected,
            "model.decoder.embedding_layernorm",
            &[h],
            &[h],
        );
        weight_bias(&mut expected, "model.decoder.norm", &[h], &[h]);
        weight_bias(&mut expected, "proj_out", &[vocab, h], &[vocab]);
        for layer in 0..self.decoder_layers {
            let prefix = format!("model.decoder.layers.{layer}");
            for attention in ["self_attn", "encoder_attn"] {
                for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    weight_bias(
                        &mut expected,
                        &format!("{prefix}.{attention}.{projection}"),
                        &[h, h],
                        &[h],
                    );
                }
            }
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "final_layernorm",
            ] {
                weight_bias(&mut expected, &format!("{prefix}.{norm}"), &[h], &[h]);
            }
            weight_bias(
                &mut expected,
                &format!("{prefix}.mlp.fc1"),
                &[inter, h],
                &[inter],
            );
            weight_bias(
                &mut expected,
                &format!("{prefix}.mlp.fc2"),
                &[h, inter],
                &[h],
            );
        }
        let count = parakeet::require_checkpoint(checkpoint, &expected, "Canary")?;
        let embed = checkpoint.tensor("model.decoder.embed_tokens.weight")?;
        let projection = checkpoint.tensor("proj_out.weight")?;
        ensure!(
            embed.data() == projection.data(),
            "proj_out.weight and model.decoder.embed_tokens.weight differ at byte {:?}",
            embed
                .data()
                .iter()
                .zip(projection.data())
                .position(|(left, right)| left != right)
        );
        Ok(count)
    }
}
