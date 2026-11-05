//! Parakeet TDT published configuration and tensor contract.
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine_parakeet::ParakeetEncoder;

use crate::{
    audio::LogMelConfig,
    checkpoint::MappedCheckpoint,
    model_spec::{equal, exact_keys, positive, u32_list},
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

pub struct ParakeetSpec {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub heads: u32,
    pub max_position: u32,
    pub conv_kernel: u32,
    pub subsampling_channels: u32,
    pub decoder_hidden: u32,
    pub decoder_layers: u32,
    pub vocab: u32,
    pub blank: u32,
    pub max_symbols: u32,
    pub durations: Vec<u32>,
    pub mel: LogMelConfig,
}

impl ParakeetSpec {
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
                "dtype",
                "durations",
                "encoder_config",
                "hidden_act",
                "initializer_range",
                "is_encoder_decoder",
                "max_symbols_per_step",
                "model_type",
                "num_decoder_layers",
                "pad_token_id",
                "transformers_version",
                "vocab_size",
            ],
            "Parakeet config",
        )?;
        for (key, value) in [
            ("architectures", serde_json::json!(["ParakeetForTDT"])),
            ("model_type", "parakeet_tdt".into()),
            ("dtype", "float32".into()),
            ("hidden_act", "relu".into()),
            ("initializer_range", 0.02.into()),
            ("is_encoder_decoder", true.into()),
            ("pad_token_id", 2.into()),
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
            "Parakeet encoder config",
        )?;
        for (key, value) in [
            ("model_type", "parakeet_encoder".into()),
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
            ("subsampling_conv_kernel_size", 3.into()),
            ("subsampling_conv_stride", 2.into()),
            ("subsampling_factor", 8.into()),
        ] {
            equal(encoder, key, value)?;
        }
        let hidden = positive(encoder, "hidden_size")?;
        let heads = positive(encoder, "num_attention_heads")?;
        ensure!(
            hidden == 1024
                && heads == 8
                && positive(encoder, "num_key_value_heads")? == heads
                && positive(encoder, "num_hidden_layers")? == 24
                && positive(encoder, "intermediate_size")? == 4096
                && positive(encoder, "num_mel_bins")? == 128
                && positive(encoder, "max_position_embeddings")? == 5000
                && positive(encoder, "subsampling_conv_channels")? == 256
                && positive(encoder, "conv_kernel_size")? == 9,
            "unsupported Parakeet encoder geometry"
        );
        let vocab = positive(root, "vocab_size")?;
        let blank = positive(root, "blank_token_id")?;
        let durations = u32_list(root, "durations")?;
        ensure!(
            vocab == 8193
                && blank == 8192
                && positive(root, "decoder_hidden_size")? == 640
                && positive(root, "num_decoder_layers")? == 2
                && positive(root, "max_symbols_per_step")? == 10
                && durations == [0, 1, 2, 3, 4],
            "unsupported Parakeet TDT decoder geometry"
        );
        exact_keys(
            processor,
            &["blank_token", "feature_extractor", "processor_class"],
            "Parakeet processor config",
        )?;
        equal(processor, "processor_class", "ParakeetProcessor".into())?;
        equal(processor, "blank_token", "<blank>".into())?;
        let features = processor
            .get("feature_extractor")
            .context("Parakeet feature_extractor missing")?;
        exact_keys(
            features,
            &[
                "feature_extractor_type",
                "feature_size",
                "hop_length",
                "n_fft",
                "padding_side",
                "padding_value",
                "preemphasis",
                "return_attention_mask",
                "sampling_rate",
                "win_length",
            ],
            "Parakeet feature extractor",
        )?;
        for (key, value) in [
            ("feature_extractor_type", "ParakeetFeatureExtractor".into()),
            ("padding_side", "right".into()),
            ("padding_value", 0.0.into()),
            ("return_attention_mask", true.into()),
        ] {
            equal(features, key, value)?;
        }
        ensure!(
            positive(features, "feature_size")? == 128
                && positive(features, "hop_length")? == 160
                && positive(features, "n_fft")? == 512
                && positive(features, "sampling_rate")? == 16000
                && positive(features, "win_length")? == 400,
            "unsupported Parakeet mel geometry"
        );
        let preemphasis = features["preemphasis"]
            .as_f64()
            .context("invalid preemphasis")? as f32;
        ensure!(preemphasis == 0.97, "unsupported preemphasis");
        exact_keys(
            generation,
            &[
                "_from_model_config",
                "decoder_start_token_id",
                "eos_token_id",
                "output_attentions",
                "output_hidden_states",
                "pad_token_id",
                "suppress_tokens",
                "transformers_version",
            ],
            "Parakeet generation config",
        )?;
        for (key, value) in [
            ("_from_model_config", true.into()),
            ("decoder_start_token_id", blank.into()),
            ("eos_token_id", 3.into()),
            ("pad_token_id", 2.into()),
            ("output_attentions", false.into()),
            ("output_hidden_states", false.into()),
            (
                "suppress_tokens",
                serde_json::json!([8193, 8194, 8195, 8196, 8197]),
            ),
        ] {
            equal(generation, key, value)?;
        }
        Ok(Self {
            hidden,
            intermediate: 4096,
            layers: 24,
            heads,
            max_position: 5000,
            conv_kernel: 9,
            subsampling_channels: 256,
            decoder_hidden: 640,
            decoder_layers: 2,
            vocab,
            blank,
            max_symbols: 10,
            durations,
            mel: LogMelConfig {
                sampling_rate: 16000,
                n_fft: 512,
                win_length: 400,
                hop_length: 160,
                mel_bins: 128,
                preemphasis,
                log_guard: 2.0f32.powi(-24),
                norm_eps: 1e-5,
            },
        })
    }

    pub fn validate_checkpoint(&self, checkpoint: &MappedCheckpoint) -> Result<usize> {
        let mut expected = BTreeMap::new();
        let mut insert = |name: String, dtype: Dtype, shape: &[usize]| {
            expected.insert(name, (dtype, shape.to_vec()));
        };
        let h = self.hidden as usize;
        let inter = self.intermediate as usize;
        for layer in 0..self.layers {
            let prefix = format!("encoder.layers.{layer}");
            for branch in ["feed_forward1", "feed_forward2"] {
                insert(
                    format!("{prefix}.{branch}.linear1.weight"),
                    Dtype::F32,
                    &[inter, h],
                );
                insert(
                    format!("{prefix}.{branch}.linear2.weight"),
                    Dtype::F32,
                    &[h, inter],
                );
            }
            for norm in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
            ] {
                insert(format!("{prefix}.{norm}.weight"), Dtype::F32, &[h]);
                insert(format!("{prefix}.{norm}.bias"), Dtype::F32, &[h]);
            }
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
                insert(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    Dtype::F32,
                    &[h, h],
                );
            }
            insert(format!("{prefix}.self_attn.bias_u"), Dtype::F32, &[8, 128]);
            insert(format!("{prefix}.self_attn.bias_v"), Dtype::F32, &[8, 128]);
            insert(
                format!("{prefix}.conv.pointwise_conv1.weight"),
                Dtype::F32,
                &[2 * h, h, 1],
            );
            insert(
                format!("{prefix}.conv.pointwise_conv2.weight"),
                Dtype::F32,
                &[h, h, 1],
            );
            insert(
                format!("{prefix}.conv.depthwise_conv.weight"),
                Dtype::F32,
                &[h, 1, self.conv_kernel as usize],
            );
            for name in ["weight", "bias", "running_mean", "running_var"] {
                insert(format!("{prefix}.conv.norm.{name}"), Dtype::F32, &[h]);
            }
            insert(
                format!("{prefix}.conv.norm.num_batches_tracked"),
                Dtype::I64,
                &[],
            );
        }
        let channels = self.subsampling_channels as usize;
        for layer in [0, 2, 5] {
            insert(
                format!("encoder.subsampling.layers.{layer}.weight"),
                Dtype::F32,
                &[channels, 1, 3, 3],
            );
            insert(
                format!("encoder.subsampling.layers.{layer}.bias"),
                Dtype::F32,
                &[channels],
            );
        }
        for layer in [3, 6] {
            insert(
                format!("encoder.subsampling.layers.{layer}.weight"),
                Dtype::F32,
                &[channels, channels, 1, 1],
            );
            insert(
                format!("encoder.subsampling.layers.{layer}.bias"),
                Dtype::F32,
                &[channels],
            );
        }
        insert(
            "encoder.subsampling.linear.weight".into(),
            Dtype::F32,
            &[h, 4096],
        );
        insert("encoder.subsampling.linear.bias".into(), Dtype::F32, &[h]);
        insert("encoder_projector.weight".into(), Dtype::F32, &[640, h]);
        insert("encoder_projector.bias".into(), Dtype::F32, &[640]);
        insert("decoder.embedding.weight".into(), Dtype::F32, &[8193, 640]);
        insert(
            "decoder.decoder_projector.weight".into(),
            Dtype::F32,
            &[640, 640],
        );
        insert("decoder.decoder_projector.bias".into(), Dtype::F32, &[640]);
        for layer in 0..self.decoder_layers {
            for kind in ["weight_ih", "weight_hh"] {
                insert(
                    format!("decoder.lstm.{kind}_l{layer}"),
                    Dtype::F32,
                    &[2560, 640],
                );
            }
            for kind in ["bias_ih", "bias_hh"] {
                insert(format!("decoder.lstm.{kind}_l{layer}"), Dtype::F32, &[2560]);
            }
        }
        insert("joint.head.weight".into(), Dtype::F32, &[8198, 640]);
        insert("joint.head.bias".into(), Dtype::F32, &[8198]);
        let mut actual = BTreeMap::new();
        for shard in checkpoint.views()?.values() {
            for (name, tensor) in shard.iter() {
                actual.insert(name.to_owned(), (tensor.dtype(), tensor.shape().to_vec()));
            }
        }
        if let Some(name) = expected.keys().find(|name| !actual.contains_key(*name)) {
            anyhow::bail!("Parakeet tensor missing: {name}");
        }
        if let Some(name) = actual.keys().find(|name| !expected.contains_key(*name)) {
            anyhow::bail!("Parakeet tensor unexpected: {name}");
        }
        for (name, entry) in &expected {
            ensure!(
                actual[name] == *entry,
                "Parakeet tensor contract mismatch: {name}"
            );
        }
        Ok(expected.len())
    }
}
