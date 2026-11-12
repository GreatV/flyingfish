//! Multitalker parakeet streaming 0.6b: spec from the .nemo model_config.yaml, read in place.

#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine_multitalker::Multitalker;

use crate::checkpoint::tar_member_bytes;
use crate::eou::yaml;
use anyhow::{Context, Result, bail, ensure};
use memmap2::MmapOptions;
use std::{fs::File, path::Path};

pub struct MultitalkerSpec {
    pub sample_rate: u32,
    pub features: u32,
    pub n_fft: u32,
    pub window_size: f64,
    pub window_stride: f64,
    pub frame_splicing: u32,
    pub dither: f64,
    pub pad_to: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub feat_in: u32,
    pub subsampling_factor: u32,
    pub subsampling_conv_channels: u32,
    pub causal_downsampling: bool,
    pub ff_expansion_factor: u32,
    pub n_heads: u32,
    pub att_left: u32,
    pub att_right: u32,
    pub att_context_modes: u32,
    pub pos_emb_max_len: u32,
    pub conv_kernel_size: u32,
    pub pred_hidden: u32,
    pub pred_rnn_layers: u32,
    pub vocab_size: u32,
    pub blank_as_pad: bool,
    pub joint_hidden: u32,
    pub num_classes: u32,
    pub num_speakers: u32,
    pub spk_kernel_type: String,
    pub spk_kernel_layers: Vec<u32>,
    pub add_bg_spk_kernel: bool,
    pub max_symbols: u32,
}

fn u32_of(value: &yaml::Value, key: &str) -> Result<u32> {
    u32::try_from(value.integer(key)?).with_context(|| key.to_owned())
}

impl MultitalkerSpec {
    /// Reads `model_config.yaml` out of the `.nemo` tar archive in place.
    pub fn read(nemo: &Path) -> Result<Self> {
        let file = File::open(nemo)?;
        let map = unsafe { MmapOptions::new().map(&file)? };
        let text = std::str::from_utf8(tar_member_bytes(&map, "model_config.yaml")?)?;
        let root = yaml::parse(text)?;
        let preprocessor = root.mapping("preprocessor")?;
        let encoder = root.mapping("encoder")?;
        let decoder = root.mapping("decoder")?;
        let prednet = decoder.mapping("prednet")?;
        let joint = root.mapping("joint")?;
        let jointnet = joint.mapping("jointnet")?;
        let model_defaults = root.mapping("model_defaults")?;
        let decoding = root.mapping("decoding")?;
        let greedy = decoding.mapping("greedy")?;
        ensure!(
            preprocessor.text("normalize")? == "NA",
            "multitalker preprocessor normalization"
        );
        ensure!(
            encoder.text("subsampling")? == "dw_striding"
                && encoder.text("self_attention_model")? == "rel_pos"
                && encoder.text("att_context_style")? == "chunked_limited"
                && encoder.text("conv_context_size")? == "causal"
                && encoder.text("conv_norm_type")? == "layer_norm"
                && !encoder.boolean("xscaling")?
                && !encoder.boolean("use_bias")?
                && encoder.boolean("untie_biases")?,
            "multitalker encoder architecture variant"
        );
        ensure!(
            jointnet.text("activation")? == "relu",
            "multitalker joint activation"
        );
        ensure!(decoder.boolean("blank_as_pad")?, "multitalker blank as pad");
        // Inference runs the first att_context_size mode on every layer; the
        // remaining modes are training-time alternatives only.
        let context = encoder.sequence("att_context_size")?;
        let mut att_left = None;
        let mut att_right = None;
        let pair_value = |value: &yaml::Value| -> Result<i64> {
            let yaml::Value::Scalar(text) = value else {
                bail!("att_context_size entry is not a scalar pair");
            };
            text.parse::<i64>()
                .with_context(|| format!("att_context_size entry {text} is not an integer"))
        };
        for mode in context {
            let yaml::Value::Sequence(pair) = mode else {
                bail!("att_context_size entry is not a pair");
            };
            ensure!(pair.len() == 2, "att_context_size entry is not a pair");
            let (left, right) = (pair_value(&pair[0])?, pair_value(&pair[1])?);
            let (left, right) = (u32::try_from(left)?, u32::try_from(right)?);
            match att_left {
                None => {
                    att_left = Some(left);
                    att_right = Some(right);
                }
                Some(seen) => ensure!(seen == left, "att_context_size left contexts differ"),
            }
            ensure!(
                left.is_multiple_of(right + 1),
                "chunked_limited att_context_size [{left}, {right}] violates left % (right + 1) == 0"
            );
        }
        let (att_left, att_right) = (
            att_left.context("att_context_size empty")?,
            att_right.context("att_context_size empty")?,
        );
        let spk_kernel_layers = root
            .sequence("spk_kernel_layers")?
            .iter()
            .map(|value| match value {
                yaml::Value::Scalar(text) => text
                    .parse::<u32>()
                    .map_err(|e| anyhow::anyhow!("spk_kernel_layers entry: {e}")),
                _ => bail!("spk_kernel_layers entry is not a scalar"),
            })
            .collect::<Result<Vec<u32>>>()?;
        Ok(Self {
            sample_rate: u32_of(&root, "sample_rate")?,
            features: u32_of(preprocessor, "features")?,
            n_fft: u32_of(preprocessor, "n_fft")?,
            window_size: preprocessor.float("window_size")?,
            window_stride: preprocessor.float("window_stride")?,
            frame_splicing: u32_of(preprocessor, "frame_splicing")?,
            dither: preprocessor.float("dither")?,
            pad_to: u32_of(preprocessor, "pad_to")?,
            n_layers: u32_of(encoder, "n_layers")?,
            d_model: u32_of(encoder, "d_model")?,
            feat_in: u32_of(encoder, "feat_in")?,
            subsampling_factor: u32_of(encoder, "subsampling_factor")?,
            subsampling_conv_channels: u32_of(encoder, "subsampling_conv_channels")?,
            causal_downsampling: encoder.boolean("causal_downsampling")?,
            ff_expansion_factor: u32_of(encoder, "ff_expansion_factor")?,
            n_heads: u32_of(encoder, "n_heads")?,
            att_left,
            att_right,
            att_context_modes: u32::try_from(context.len())?,
            pos_emb_max_len: u32_of(encoder, "pos_emb_max_len")?,
            conv_kernel_size: u32_of(encoder, "conv_kernel_size")?,
            pred_hidden: u32_of(prednet, "pred_hidden")?,
            pred_rnn_layers: u32_of(prednet, "pred_rnn_layers")?,
            vocab_size: u32_of(decoder, "vocab_size")?,
            blank_as_pad: decoder.boolean("blank_as_pad")?,
            joint_hidden: u32_of(jointnet, "joint_hidden")?,
            num_classes: u32_of(joint, "num_classes")?,
            num_speakers: u32_of(model_defaults, "num_speakers")?,
            spk_kernel_type: root.text("spk_kernel_type")?.to_owned(),
            spk_kernel_layers,
            add_bg_spk_kernel: root.boolean("add_bg_spk_kernel")?,
            max_symbols: u32_of(greedy, "max_symbols")?,
        })
    }

    pub fn conformer(&self) -> crate::parakeet::FastConformer {
        crate::parakeet::FastConformer {
            layers: self.n_layers,
            hidden: self.d_model,
            heads: self.n_heads,
            conv_kernel: self.conv_kernel_size,
            attention_bias: false,
            convolution_bias: false,
            prefix: "encoder",
            max_position: self.pos_emb_max_len,
            conv_batch_norm: false,
            pad_left: 2,
            pad_right: 1,
            depthwise_left: self.conv_kernel_size - 1,
            // 256 channels times 128/8 + 1 subsampled rows = 4352
            linear_in: 4352,
            // chunked_limited [70, 13]: chunk = right + 1 = 14 rows, 5 left chunks
            chunk_size: 14,
            left_chunks: 5,
            conv_names: [
                "pre_encode.conv.0",
                "pre_encode.conv.2",
                "pre_encode.conv.5",
            ],
            pointwise_names: ["pre_encode.conv.3", "pre_encode.conv.6"],
            linear_name: "pre_encode.out",
            attention_names: [
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_out",
                "linear_pos",
                "pos_bias_u",
                "pos_bias_v",
            ],
            conv_norm_name: "batch_norm",
        }
    }

    pub fn mel(&self) -> crate::audio::LogMelConfig {
        crate::audio::LogMelConfig {
            sampling_rate: self.sample_rate,
            n_fft: self.n_fft,
            win_length: (self.window_size * f64::from(self.sample_rate)) as u32,
            hop_length: (self.window_stride * f64::from(self.sample_rate)) as u32,
            mel_bins: self.features,
            preemphasis: 0.97,
            log_guard: 2.0f32.powi(-24),
            norm_eps: 1e-5,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate == 16_000
                && self.features == 128
                && self.n_fft == 512
                && self.window_size == 0.025
                && self.window_stride == 0.01
                && self.frame_splicing == 1
                && self.dither == 1e-5
                && self.pad_to == 0,
            "multitalker preprocessor mismatch"
        );
        ensure!(
            self.n_layers == 24
                && self.d_model == 1024
                && self.feat_in == 128
                && self.subsampling_factor == 8
                && self.subsampling_conv_channels == 256
                && self.causal_downsampling
                && self.ff_expansion_factor == 4
                && self.n_heads == 8
                && self.att_left == 70
                && self.att_right == 13
                && self.att_context_modes == 4
                && self.pos_emb_max_len == 5000
                && self.conv_kernel_size == 9,
            "multitalker encoder mismatch"
        );
        ensure!(
            self.pred_hidden == 640
                && self.pred_rnn_layers == 2
                && self.vocab_size == 1024
                && self.blank_as_pad
                && self.joint_hidden == 640
                && self.num_classes == 1024,
            "multitalker decoder/joint mismatch"
        );
        ensure!(
            self.num_speakers == 4
                && self.spk_kernel_type == "ff"
                && self.spk_kernel_layers == [0]
                && self.add_bg_spk_kernel
                && self.max_symbols == 10,
            "multitalker speaker-kernel mismatch"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::tar_archive;
    use anyhow::Context;

    const CONFIG_HEAD: &str = "sample_rate: 16000
freeze_diar: true
model_defaults:
  enc_hidden: 1024
  pred_hidden: 640
  joint_hidden: 640
  num_speakers: 4
preprocessor:
  sample_rate: 16000
  normalize: NA
  window_size: 0.025
  window_stride: 0.01
  features: 128
  n_fft: 512
  frame_splicing: 1
  dither: 1.0e-05
  pad_to: 0
encoder:
  feat_in: 128
  n_layers: 24
  d_model: 1024
  use_bias: false
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 256
  causal_downsampling: true
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
";

    const CONFIG_TAIL: &str = "  att_context_style: chunked_limited
  xscaling: false
  untie_biases: true
  pos_emb_max_len: 5000
  conv_kernel_size: 9
  conv_norm_type: layer_norm
  conv_context_size: causal
decoder:
  blank_as_pad: true
  prednet:
    pred_hidden: 640
    pred_rnn_layers: 2
  vocab_size: 1024
joint:
  num_classes: 1024
  jointnet:
    joint_hidden: 640
    activation: relu
    encoder_hidden: 1024
    pred_hidden: 640
decoding:
  strategy: greedy_batch
  greedy:
    max_symbols: 10
spk_kernel_type: ff
spk_kernel_layers:
- 0
add_bg_spk_kernel: true
";

    fn config_nemo(att: &str) -> Result<std::path::PathBuf> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ff-multitalker-{}-{}.nemo",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(
            &path,
            tar_archive(&[(
                "model_config.yaml",
                format!("{CONFIG_HEAD}  att_context_size:\n{att}{CONFIG_TAIL}").as_bytes(),
            )]),
        )?;
        Ok(path)
    }

    const PUBLISHED: &str = "  - - 70
    - 13
  - - 70
    - 6
  - - 70
    - 1
  - - 70
    - 0
";

    #[test]
    fn multitalker_spec_reads_the_published_context_modes() -> Result<()> {
        let path = config_nemo(PUBLISHED)?;
        let spec = MultitalkerSpec::read(&path)?;
        std::fs::remove_file(&path)?;
        spec.validate()?;
        ensure!(
            spec.att_left == 70
                && spec.att_right == 13
                && spec.att_context_modes == 4
                && spec.spk_kernel_layers == [0]
                && spec.num_speakers == 4,
            "multitalker spec fields"
        );
        let conformer = spec.conformer();
        ensure!(
            conformer.chunk_size == 14 && conformer.left_chunks == 5 && conformer.layers == 24,
            "multitalker conformer mapping"
        );
        println!("PASS multitalker_spec att=[70,13] modes=4 chunk=14 left=5");
        Ok(())
    }

    #[test]
    fn multitalker_spec_rejects_a_left_context_that_breaks_chunking() -> Result<()> {
        let broken = "  - - 71
    - 13
";
        let path = config_nemo(broken)?;
        let error = MultitalkerSpec::read(&path)
            .err()
            .context("spec read must fail")?;
        std::fs::remove_file(&path)?;
        ensure!(
            error.to_string().contains("left % (right + 1)"),
            "chunked_limited violation must be named, got: {error}"
        );
        println!("PASS multitalker_spec_rejects left=71 right=13");
        Ok(())
    }

    #[test]
    fn multitalker_spec_rejects_mixed_left_contexts() -> Result<()> {
        let broken = "  - - 70
    - 13
  - - 56
    - 6
";
        let path = config_nemo(broken)?;
        let error = MultitalkerSpec::read(&path)
            .err()
            .context("spec read must fail")?;
        std::fs::remove_file(&path)?;
        ensure!(
            error.to_string().contains("left contexts differ"),
            "mixed left contexts must be named, got: {error}"
        );
        println!("PASS multitalker_spec_rejects mixed left contexts");
        Ok(())
    }
}
