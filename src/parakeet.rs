//! Parakeet TDT published configuration and tensor contract.
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine_parakeet::{ParakeetEncoder, ParakeetTdt};

use crate::{
    audio::LogMelConfig,
    checkpoint::MappedCheckpoint,
    model_spec::{equal, exact_keys, positive, u32_list},
};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy)]
pub struct FastConformer {
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub conv_kernel: u32,
    pub attention_bias: bool,
    pub convolution_bias: bool,
    pub prefix: &'static str,
}

impl FastConformer {
    pub fn insert_tensors(&self, expected: &mut BTreeMap<String, (Dtype, Vec<usize>)>) {
        let hidden = self.hidden as usize;
        let intermediate = 4096usize;
        let channels = 256usize;
        let heads = self.heads as usize;
        let head_dim = hidden / heads;
        let kernel = self.conv_kernel as usize;
        let mut put = |name: String, dtype: Dtype, shape: &[usize]| {
            expected.insert(name, (dtype, shape.to_vec()));
        };
        for layer in 0..self.layers {
            let prefix = format!("{}.layers.{layer}", self.prefix);
            for branch in ["feed_forward1", "feed_forward2"] {
                put(
                    format!("{prefix}.{branch}.linear1.weight"),
                    Dtype::F32,
                    &[intermediate, hidden],
                );
                put(
                    format!("{prefix}.{branch}.linear2.weight"),
                    Dtype::F32,
                    &[hidden, intermediate],
                );
                if self.attention_bias {
                    put(
                        format!("{prefix}.{branch}.linear1.bias"),
                        Dtype::F32,
                        &[intermediate],
                    );
                    put(
                        format!("{prefix}.{branch}.linear2.bias"),
                        Dtype::F32,
                        &[hidden],
                    );
                }
            }
            for norm in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
            ] {
                put(format!("{prefix}.{norm}.weight"), Dtype::F32, &[hidden]);
                put(format!("{prefix}.{norm}.bias"), Dtype::F32, &[hidden]);
            }
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
                put(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    Dtype::F32,
                    &[hidden, hidden],
                );
            }
            if self.attention_bias {
                for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    put(
                        format!("{prefix}.self_attn.{projection}.bias"),
                        Dtype::F32,
                        &[hidden],
                    );
                }
            }
            put(
                format!("{prefix}.self_attn.bias_u"),
                Dtype::F32,
                &[heads, head_dim],
            );
            put(
                format!("{prefix}.self_attn.bias_v"),
                Dtype::F32,
                &[heads, head_dim],
            );
            put(
                format!("{prefix}.conv.pointwise_conv1.weight"),
                Dtype::F32,
                &[2 * hidden, hidden, 1],
            );
            put(
                format!("{prefix}.conv.pointwise_conv2.weight"),
                Dtype::F32,
                &[hidden, hidden, 1],
            );
            put(
                format!("{prefix}.conv.depthwise_conv.weight"),
                Dtype::F32,
                &[hidden, 1, kernel],
            );
            if self.convolution_bias {
                put(
                    format!("{prefix}.conv.pointwise_conv1.bias"),
                    Dtype::F32,
                    &[2 * hidden],
                );
                put(
                    format!("{prefix}.conv.pointwise_conv2.bias"),
                    Dtype::F32,
                    &[hidden],
                );
                put(
                    format!("{prefix}.conv.depthwise_conv.bias"),
                    Dtype::F32,
                    &[hidden],
                );
            }
            for name in ["weight", "bias", "running_mean", "running_var"] {
                put(format!("{prefix}.conv.norm.{name}"), Dtype::F32, &[hidden]);
            }
            put(
                format!("{prefix}.conv.norm.num_batches_tracked"),
                Dtype::I64,
                &[],
            );
        }
        for layer in [0, 2, 5] {
            put(
                format!("{}.subsampling.layers.{layer}.weight", self.prefix),
                Dtype::F32,
                &[channels, 1, 3, 3],
            );
            put(
                format!("{}.subsampling.layers.{layer}.bias", self.prefix),
                Dtype::F32,
                &[channels],
            );
        }
        for layer in [3, 6] {
            put(
                format!("{}.subsampling.layers.{layer}.weight", self.prefix),
                Dtype::F32,
                &[channels, channels, 1, 1],
            );
            put(
                format!("{}.subsampling.layers.{layer}.bias", self.prefix),
                Dtype::F32,
                &[channels],
            );
        }
        put(
            format!("{}.subsampling.linear.weight", self.prefix),
            Dtype::F32,
            &[hidden, intermediate],
        );
        put(
            format!("{}.subsampling.linear.bias", self.prefix),
            Dtype::F32,
            &[hidden],
        );
    }
}

pub fn feature_log_mel(features: &Value) -> Result<LogMelConfig> {
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
        "unsupported mel geometry"
    );
    let preemphasis = features["preemphasis"]
        .as_f64()
        .context("invalid preemphasis")? as f32;
    ensure!(preemphasis == 0.97, "unsupported preemphasis");
    Ok(LogMelConfig {
        sampling_rate: 16000,
        n_fft: 512,
        win_length: 400,
        hop_length: 160,
        mel_bins: 128,
        preemphasis,
        log_guard: 2.0f32.powi(-24),
        norm_eps: 1e-5,
    })
}

pub(crate) fn require_checkpoint(
    checkpoint: &MappedCheckpoint,
    expected: &BTreeMap<String, (Dtype, Vec<usize>)>,
    label: &str,
) -> Result<usize> {
    let mut actual = BTreeMap::new();
    for shard in checkpoint.views()?.values() {
        for (name, tensor) in shard.iter() {
            actual.insert(name.to_owned(), (tensor.dtype(), tensor.shape().to_vec()));
        }
    }
    if let Some(name) = expected.keys().find(|name| !actual.contains_key(*name)) {
        anyhow::bail!("{label} tensor missing: {name}");
    }
    if let Some(name) = actual.keys().find(|name| !expected.contains_key(*name)) {
        anyhow::bail!("{label} tensor unexpected: {name}");
    }
    for (name, entry) in expected {
        ensure!(
            actual[name] == *entry,
            "{label} tensor contract mismatch: {name}"
        );
    }
    Ok(expected.len())
}

pub struct ParakeetSpec {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub heads: u32,
    pub max_position: u32,
    pub conv_kernel: u32,
    pub attention_bias: bool,
    pub convolution_bias: bool,
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
        let attention_bias = encoder["attention_bias"]
            .as_bool()
            .context("invalid attention_bias")?;
        let convolution_bias = encoder["convolution_bias"]
            .as_bool()
            .context("invalid convolution_bias")?;
        let vocab = positive(root, "vocab_size")?;
        let blank = positive(root, "blank_token_id")?;
        let durations = u32_list(root, "durations")?;
        let decoder_hidden = positive(root, "decoder_hidden_size")?;
        let decoder_layers = positive(root, "num_decoder_layers")?;
        ensure!(
            vocab == 8193
                && blank == 8192
                && decoder_hidden == 640
                && decoder_layers == 2
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
        let mel = feature_log_mel(features)?;
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
            attention_bias,
            convolution_bias,
            subsampling_channels: 256,
            decoder_hidden,
            decoder_layers,
            vocab,
            blank,
            max_symbols: 10,
            durations,
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
            prefix: "encoder",
        }
    }

    pub fn validate_checkpoint(&self, checkpoint: &MappedCheckpoint) -> Result<usize> {
        let mut expected = BTreeMap::new();
        self.conformer().insert_tensors(&mut expected);
        let mut insert = |name: String, dtype: Dtype, shape: &[usize]| {
            expected.insert(name, (dtype, shape.to_vec()));
        };
        let h = self.hidden as usize;
        let d = self.decoder_hidden as usize;
        let gate = 4 * d;
        let vocab = self.vocab as usize;
        let joint = vocab + self.durations.len();
        insert("encoder_projector.weight".into(), Dtype::F32, &[d, h]);
        insert("encoder_projector.bias".into(), Dtype::F32, &[d]);
        insert("decoder.embedding.weight".into(), Dtype::F32, &[vocab, d]);
        insert(
            "decoder.decoder_projector.weight".into(),
            Dtype::F32,
            &[d, d],
        );
        insert("decoder.decoder_projector.bias".into(), Dtype::F32, &[d]);
        for layer in 0..self.decoder_layers {
            for kind in ["weight_ih", "weight_hh"] {
                insert(
                    format!("decoder.lstm.{kind}_l{layer}"),
                    Dtype::F32,
                    &[gate, d],
                );
            }
            for kind in ["bias_ih", "bias_hh"] {
                insert(format!("decoder.lstm.{kind}_l{layer}"), Dtype::F32, &[gate]);
            }
        }
        insert("joint.head.weight".into(), Dtype::F32, &[joint, d]);
        insert("joint.head.bias".into(), Dtype::F32, &[joint]);
        require_checkpoint(checkpoint, &expected, "Parakeet")
    }
}
