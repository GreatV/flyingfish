//! Parakeet realtime EOU 120m: spec from the .nemo model_config.yaml, read in place.

#[cfg(feature = "cuda")]
pub use crate::backend::cuda::engine_eou::{Eou, EouUtterance};

use crate::checkpoint::tar_member_bytes;
use anyhow::{Context, Result, bail, ensure};
use memmap2::MmapOptions;
use std::{fs::File, path::Path};

pub(crate) mod yaml {
    use anyhow::{Context, Result, bail, ensure};
    use std::collections::BTreeMap;

    pub(crate) enum Value {
        Scalar(String),
        Sequence(Vec<Value>),
        Mapping(BTreeMap<String, Value>),
    }

    struct Line {
        number: usize,
        indent: usize,
        text: String,
    }

    fn strip_comment(line: &str, number: usize) -> Result<String> {
        ensure!(!line.contains('\t'), "yaml line {number} contains a tab");
        let bytes = line.as_bytes();
        let mut quote: Option<u8> = None;
        let mut index = 0usize;
        while index < bytes.len() {
            let character = bytes[index];
            if let Some(open) = quote {
                if character == open {
                    if open == b'\'' && bytes.get(index + 1) == Some(&b'\'') {
                        index += 2;
                        continue;
                    }
                    quote = None;
                }
            } else if (character == b'\'' || character == b'"')
                && (index == 0 || bytes[index - 1] == b' ')
                && bytes.get(index + 1) != Some(&b' ')
            {
                quote = Some(character);
            } else if character == b'#' && (index == 0 || bytes[index - 1] == b' ') {
                return Ok(line[..index].to_owned());
            }
            index += 1;
        }
        ensure!(
            quote.is_none(),
            "yaml line {number} has an unterminated quote"
        );
        Ok(line.to_owned())
    }

    fn scalar(text: &str, number: usize) -> Result<String> {
        let text = text.trim();
        ensure!(!text.is_empty(), "yaml line {number} has an empty scalar");
        ensure!(
            !text.starts_with(['&', '*', '!', '%', '|', '>'])
                && text != "---"
                && text != "..."
                && !text.starts_with("--- ")
                && !text.starts_with("... "),
            "yaml line {number} uses an unsupported construct: {text}"
        );
        if let Some(stripped) = text.strip_prefix('\'') {
            let inner = stripped
                .strip_suffix('\'')
                .with_context(|| format!("yaml line {number} scalar"))?;
            return Ok(inner.replace("''", "'"));
        }
        if let Some(stripped) = text.strip_prefix('"') {
            let inner = stripped
                .strip_suffix('"')
                .with_context(|| format!("yaml line {number} scalar"))?;
            ensure!(
                !inner.contains('\\'),
                "yaml line {number} uses string escapes"
            );
            return Ok(inner.to_owned());
        }
        ensure!(
            !text.contains(['[', ']', '{', '}']),
            "yaml line {number} scalar has structural characters: {text}"
        );
        Ok(text.to_owned())
    }

    fn split_key(line: &Line) -> Result<(String, &str)> {
        let text = line.text.trim_start();
        let index = text
            .find(": ")
            .or_else(|| text.strip_suffix(':').map(|_| text.len() - 1));
        let index = index.with_context(|| format!("yaml line {} is not a mapping", line.number))?;
        let key = scalar(&text[..index], line.number)?;
        let value = text[index + 1..].trim();
        Ok((key, value))
    }

    struct Cursor {
        lines: Vec<Line>,
        position: usize,
    }

    impl Cursor {
        fn block(&mut self, indent: usize) -> Result<Value> {
            let line = self
                .lines
                .get(self.position)
                .context("yaml block ends early")?;
            if line.text.trim_start().starts_with('-') {
                return self.sequence(indent);
            }
            self.mapping(indent)
        }

        fn sequence(&mut self, indent: usize) -> Result<Value> {
            let mut items = Vec::new();
            while self.position < self.lines.len() {
                let (number, line_indent, text) = {
                    let line = &self.lines[self.position];
                    if line.indent < indent {
                        break;
                    }
                    if line.indent == indent && !line.text.trim_start().starts_with('-') {
                        break;
                    }
                    ensure!(
                        line.indent == indent,
                        "yaml line {} sequence indent",
                        line.number
                    );
                    (line.number, line.indent, line.text.trim_start().to_owned())
                };
                let rest = text
                    .strip_prefix("- ")
                    .or_else(|| text.strip_prefix('-'))
                    .with_context(|| format!("yaml line {number} sequence item"))?
                    .to_owned();
                self.position += 1;
                if rest.trim().is_empty() {
                    let child_indent = self
                        .lines
                        .get(self.position)
                        .context("yaml sequence item empty")?
                        .indent;
                    ensure!(
                        child_indent > indent,
                        "yaml line {number} sequence item empty"
                    );
                    items.push(self.block(child_indent)?);
                } else if rest.trim().contains(": ") || rest.trim().ends_with(':') {
                    let item = Line {
                        number,
                        indent: line_indent + 2,
                        text: format!("{}{}", " ".repeat(line_indent + 2), rest),
                    };
                    self.lines.insert(self.position, item);
                    items.push(self.mapping(line_indent + 2)?);
                } else if rest.trim_start().starts_with('-') {
                    let item = Line {
                        number,
                        indent: line_indent + 2,
                        text: format!("{}{}", " ".repeat(line_indent + 2), rest),
                    };
                    self.lines.insert(self.position, item);
                    items.push(self.sequence(line_indent + 2)?);
                } else {
                    items.push(Value::Scalar(scalar(&rest, number)?));
                }
            }
            ensure!(!items.is_empty(), "yaml empty sequence");
            Ok(Value::Sequence(items))
        }

        fn mapping(&mut self, indent: usize) -> Result<Value> {
            let mut map = BTreeMap::new();
            while self.position < self.lines.len() {
                let (number, key, value) = {
                    let line = &self.lines[self.position];
                    if line.indent < indent {
                        break;
                    }
                    ensure!(
                        line.indent == indent,
                        "yaml line {} mapping indent",
                        line.number
                    );
                    let (key, value) = split_key(line)?;
                    (line.number, key, value.to_owned())
                };
                self.position += 1;
                let entry = if value.is_empty() {
                    let child = self.lines.get(self.position);
                    match child {
                        Some(child) if child.indent > indent => self.block(child.indent)?,
                        Some(child)
                            if child.indent == indent
                                && child.text.trim_start().starts_with('-') =>
                        {
                            self.sequence(indent)?
                        }
                        _ => bail!("yaml line {number} key {key} has no value"),
                    }
                } else if value.starts_with('[') {
                    let inner = value
                        .strip_prefix('[')
                        .and_then(|v| v.strip_suffix(']'))
                        .with_context(|| format!("yaml line {number} flow sequence"))?;
                    let mut items = Vec::new();
                    for part in inner.split(',') {
                        let part = part.trim();
                        if part.is_empty() {
                            continue;
                        }
                        items.push(Value::Scalar(scalar(part, number)?));
                    }
                    Value::Sequence(items)
                } else {
                    Value::Scalar(scalar(&value, number)?)
                };
                ensure!(
                    map.insert(key, entry).is_none(),
                    "yaml duplicate key on line {number}"
                );
            }
            ensure!(!map.is_empty(), "yaml empty mapping");
            Ok(Value::Mapping(map))
        }
    }

    pub(crate) fn parse(text: &str) -> Result<Value> {
        let mut lines = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let stripped = strip_comment(raw, index + 1)?;
            let trimmed = stripped.trim();
            if trimmed.is_empty() {
                continue;
            }
            ensure!(
                trimmed != "---" && trimmed != "..." && !trimmed.starts_with('%'),
                "yaml line {} document marker or directive",
                index + 1
            );
            let indent = stripped.len() - stripped.trim_start().len();
            lines.push(Line {
                number: index + 1,
                indent,
                text: stripped,
            });
        }
        ensure!(!lines.is_empty(), "yaml document is empty");
        ensure!(lines[0].indent == 0, "yaml first line is indented");
        let mut cursor = Cursor { lines, position: 0 };
        let root = cursor.block(0)?;
        ensure!(
            cursor.position == cursor.lines.len(),
            "yaml trailing content"
        );
        Ok(root)
    }

    impl Value {
        fn entry(&self, key: &str) -> Result<&Value> {
            let Value::Mapping(map) = self else {
                bail!("yaml value is not a mapping when reading {key}");
            };
            map.get(key)
                .with_context(|| format!("yaml key missing: {key}"))
        }
        pub(crate) fn mapping(&self, key: &str) -> Result<&Value> {
            let value = self.entry(key)?;
            ensure!(
                matches!(value, Value::Mapping(_)),
                "yaml key {key} is not a mapping"
            );
            Ok(value)
        }
        pub(crate) fn sequence(&self, key: &str) -> Result<&[Value]> {
            let value = self.entry(key)?;
            let Value::Sequence(items) = value else {
                bail!("yaml key {key} is not a sequence");
            };
            Ok(items)
        }
        pub(crate) fn text(&self, key: &str) -> Result<&str> {
            let value = self.entry(key)?;
            let Value::Scalar(text) = value else {
                bail!("yaml key {key} is not a scalar");
            };
            Ok(text)
        }
        pub(crate) fn integer(&self, key: &str) -> Result<i64> {
            self.text(key)?
                .parse::<i64>()
                .with_context(|| format!("yaml key {key} is not an integer"))
        }
        pub(crate) fn float(&self, key: &str) -> Result<f64> {
            self.text(key)?
                .parse::<f64>()
                .with_context(|| format!("yaml key {key} is not a float"))
        }
        pub(crate) fn boolean(&self, key: &str) -> Result<bool> {
            match self.text(key)? {
                "true" => Ok(true),
                "false" => Ok(false),
                other => bail!("yaml key {key} is not a boolean: {other}"),
            }
        }
    }
}

pub struct EouSpec {
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
    pub pos_emb_max_len: u32,
    pub conv_kernel_size: u32,
    pub pred_hidden: u32,
    pub pred_rnn_layers: u32,
    pub vocab_size: u32,
    pub blank_as_pad: bool,
    pub joint_hidden: u32,
    pub num_classes: u32,
}

fn u32_of(value: &yaml::Value, key: &str) -> Result<u32> {
    u32::try_from(value.integer(key)?).with_context(|| key.to_owned())
}

impl EouSpec {
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
        ensure!(
            root.text("token_init_method")? == "constant",
            "EOU token init method"
        );
        ensure!(
            preprocessor.text("normalize")? == "NA",
            "EOU preprocessor normalization"
        );
        let context = encoder.sequence("att_context_size")?;
        let context_value = |index: usize| -> Result<u32> {
            let yaml::Value::Scalar(text) = context
                .get(index)
                .with_context(|| format!("att_context_size item {index}"))?
            else {
                bail!("att_context_size item {index} is not a scalar");
            };
            Ok(u32::try_from(text.parse::<i64>()?)?)
        };
        ensure!(
            encoder.text("subsampling")? == "dw_striding"
                && encoder.text("self_attention_model")? == "rel_pos"
                && encoder.text("att_context_style")? == "chunked_limited"
                && encoder.text("conv_context_size")? == "causal"
                && encoder.text("conv_norm_type")? == "layer_norm"
                && !encoder.boolean("xscaling")?
                && !encoder.boolean("use_bias")?,
            "EOU encoder architecture variant"
        );
        ensure!(
            jointnet.text("activation")? == "relu",
            "EOU joint activation"
        );
        ensure!(decoder.boolean("blank_as_pad")?, "EOU blank as pad");
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
            att_left: context_value(0)?,
            att_right: context_value(1)?,
            pos_emb_max_len: u32_of(encoder, "pos_emb_max_len")?,
            conv_kernel_size: u32_of(encoder, "conv_kernel_size")?,
            pred_hidden: u32_of(prednet, "pred_hidden")?,
            pred_rnn_layers: u32_of(prednet, "pred_rnn_layers")?,
            vocab_size: u32_of(decoder, "vocab_size")?,
            blank_as_pad: decoder.boolean("blank_as_pad")?,
            joint_hidden: u32_of(jointnet, "joint_hidden")?,
            num_classes: u32_of(joint, "num_classes")?,
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
            // chunked_limited [70, 1]: chunk = right + 1 = 2 rows, 35 left chunks
            chunk_size: 2,
            left_chunks: 35,
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
            "EOU preprocessor mismatch"
        );
        ensure!(
            self.n_layers == 17
                && self.d_model == 512
                && self.feat_in == 128
                && self.subsampling_factor == 8
                && self.subsampling_conv_channels == 256
                && self.causal_downsampling
                && self.ff_expansion_factor == 4
                && self.n_heads == 8
                && self.att_left == 70
                && self.att_right == 1
                && self.pos_emb_max_len == 5000
                && self.conv_kernel_size == 9,
            "EOU encoder mismatch"
        );
        ensure!(
            self.pred_hidden == 640
                && self.pred_rnn_layers == 1
                && self.vocab_size == 1026
                && self.blank_as_pad
                && self.joint_hidden == 640
                && self.num_classes == 1026,
            "EOU decoder/joint mismatch"
        );
        Ok(())
    }
}

fn unigram_decode(vocabulary: &str, ids: &[u32]) -> Result<String> {
    let mut pieces = Vec::new();
    for line in vocabulary.lines() {
        let (piece, score) = line.rsplit_once('\t').context("EOU vocabulary row")?;
        let score = score.parse::<f64>()?;
        ensure!(score.is_finite(), "EOU vocabulary score is not finite");
        pieces.push((piece.to_string(), score));
    }
    let model = tokenizers::models::unigram::Unigram::from(pieces, Some(0), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let builder: tokenizers::tokenizer::TokenizerBuilder<
        _,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    > = tokenizers::tokenizer::TokenizerBuilder::new();
    let builder = builder.with_model(model).with_decoder(Some(
        tokenizers::decoders::DecoderWrapper::Metaspace(
            tokenizers::pre_tokenizers::metaspace::Metaspace::new(
                '▁',
                tokenizers::pre_tokenizers::metaspace::PrependScheme::Always,
                true,
            ),
        ),
    ));
    let tokenizer = builder
        .build()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    tokenizer
        .decode(ids, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

pub fn eou_decode(nemo: &Path, ids: &[u32]) -> Result<String> {
    let file = File::open(nemo)?;
    let map = unsafe { MmapOptions::new().map(&file)? };
    let bytes = &map[crate::checkpoint::tar_member_suffix(&map, "_tokenizer.vocab")?];
    unigram_decode(std::str::from_utf8(bytes)?, ids)
}

#[cfg(feature = "cuda")]
pub fn transcribe(
    nemo: &Path,
    waveform: &[f32],
    stream: bool,
    options: crate::RunOptions,
) -> Result<crate::media::Transcription> {
    let spec = EouSpec::read(nemo)?;
    spec.validate()?;
    let device = crate::backend::cuda::Device::new(options.device)?;
    let mut model = Eou::load(nemo, &device)?;
    let features = model.features(waveform)?;
    if stream {
        let (utterances, _) = model.stream_decode_timed(&features)?;
        let mut segments = Vec::new();
        let mut start = 0.0;
        for utterance in utterances {
            let ids: Vec<_> = utterance
                .tokens
                .into_iter()
                .filter(|token| *token != spec.vocab_size - 2)
                .collect();
            let text = eou_decode(nemo, &ids)?;
            let end = utterance.end_step as f64 * 0.08;
            segments.push(crate::media::TextSegment { start, end, text });
            start = end;
        }
        let text = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(crate::media::Transcription {
            text,
            segments,
            words: Vec::new(),
        })
    } else {
        let valid = features.len() / spec.features as usize;
        let ids = model.decode(&features, valid)?;
        let text = eou_decode(nemo, &ids)?;
        crate::media::Transcription::whole_clip(text, waveform.len() as f64 / 16000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::tar_archive;
    use anyhow::Context;

    #[test]
    fn unigram_decodes_published_vocabulary_form() -> Result<()> {
        let decoded = unigram_decode(
            "<unk>\t0.0\n▁hello\t-1.0\n▁world\t-2.0\n▁again\t-3.0",
            &[1, 2],
        )?;
        assert_eq!(decoded, "hello world");
        Ok(())
    }

    const CONFIG: &str = "\
token_init_method: constant
sample_rate: 16000
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
  n_layers: 17
  d_model: 512
  use_bias: false
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 256
  causal_downsampling: true
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size:
  - 70
  - 1
  att_context_style: chunked_limited
  xscaling: false
  pos_emb_max_len: 5000
  conv_kernel_size: 9
  conv_norm_type: layer_norm
  conv_context_size: causal
decoder:
  vocab_size: 1026
  blank_as_pad: true
  prednet:
    pred_hidden: 640
    pred_rnn_layers: 1
joint:
  num_classes: 1026
  jointnet:
    joint_hidden: 640
    activation: relu
    encoder_hidden: 512
    pred_hidden: 640
";

    fn config_nemo(yaml: &str) -> Result<std::path::PathBuf> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ff-eou-{}-{}.nemo",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(
            &path,
            tar_archive(&[("model_config.yaml", yaml.as_bytes())]),
        )?;
        Ok(path)
    }

    #[test]
    fn eou_spec_reads_a_valid_config() -> Result<()> {
        let path = config_nemo(CONFIG)?;
        let spec = EouSpec::read(&path)?;
        std::fs::remove_file(&path)?;
        spec.validate()?;
        ensure!(
            spec.att_left == 70 && spec.att_right == 1 && spec.vocab_size == 1026,
            "eou spec values"
        );
        println!("PASS eou_spec_yaml layers=17 att=[70,1] vocab=1026");
        Ok(())
    }

    #[test]
    fn eou_spec_rejects_config_mutations() -> Result<()> {
        for (label, yaml) in [
            (
                "alias",
                CONFIG.replace("sample_rate: 16000", "sample_rate: *ref"),
            ),
            ("anchor", CONFIG.replace("n_heads: 8", "n_heads: &eight 8")),
            ("tag", CONFIG.replace("pad_to: 0", "pad_to: !!int 0")),
            ("document marker", format!("---\n{CONFIG}")),
            ("tab", CONFIG.replace("  features: 128", "\tfeatures: 128")),
            (
                "duplicate key",
                CONFIG.replace(
                    "sample_rate: 16000\n",
                    "sample_rate: 16000\nsample_rate: 16001\n",
                ),
            ),
            ("missing key", CONFIG.replace("  n_layers: 17\n", "")),
            (
                "wrong type",
                CONFIG.replace("n_layers: 17", "n_layers: seventeen"),
            ),
            (
                "flow mapping",
                CONFIG.replace(
                    "pred_hidden: 640\n    pred_rnn_layers: 1",
                    "pred_hidden: 640",
                ),
            ),
        ] {
            let path = config_nemo(&yaml)?;
            let error = EouSpec::read(&path)
                .err()
                .with_context(|| label.to_owned())?;
            std::fs::remove_file(&path)?;
            ensure!(
                !error.to_string().is_empty(),
                "eou yaml mutation {label} produced no error"
            );
        }
        println!(
            "PASS eou_spec_yaml_rejects alias=true anchor=true tag=true document_marker=true tab=true duplicate_key=true missing_key=true wrong_type=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "published parakeet_realtime_eou_120m-v1 archive; set FF_EOU_NEMO"]
    fn eou_spec_matches_the_published_config() -> Result<()> {
        let nemo = std::env::var("FF_EOU_NEMO").context("FF_EOU_NEMO missing")?;
        let spec = EouSpec::read(std::path::Path::new(&nemo))?;
        spec.validate()?;
        println!(
            "PASS eou_spec_published layers={} d_model={} att=[{},{}] vocab={} classes={}",
            spec.n_layers,
            spec.d_model,
            spec.att_left,
            spec.att_right,
            spec.vocab_size,
            spec.num_classes
        );
        Ok(())
    }
}
