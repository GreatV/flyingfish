//! The CLIP text tower TRELLIS conditions on.
//!
//! `TrellisTextTo3DPipeline._init_text_cond_model` loads
//! `openai/clip-vit-large-patch14` through `transformers.CLIPTextModel` and
//! `encode_text` takes its `last_hidden_state` after tokenizing with
//! `max_length=77, padding='max_length', truncation=True`. So the conditioning
//! is the full 77-position sequence after the final layer norm, not a pooled or
//! projected embedding: `text_projection` is present in the checkpoint and
//! unused.
//!
//! Three details of CLIP's text encoder decide the numbers: the attention is
//! causal, the activation is `quick_gelu` rather than a true GELU, and the
//! layer norms come before each residual branch.

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use ff_core::weights::ModelWeights;
use std::path::Path;

/// CLIP ViT-L/14's text side, as its `config.json` states it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClipTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

impl ClipTextConfig {
    /// The published `openai/clip-vit-large-patch14` text configuration.
    pub const VIT_LARGE_PATCH14: Self = Self {
        hidden_size: 768,
        intermediate_size: 3072,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        max_position_embeddings: 77,
        vocab_size: 49_408,
    };

    pub fn head_dim(&self) -> Result<usize> {
        anyhow::ensure!(
            self.num_attention_heads > 0
                && self.hidden_size.is_multiple_of(self.num_attention_heads),
            "CLIP hidden size {} does not divide into {} heads",
            self.hidden_size,
            self.num_attention_heads
        );
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

/// `layer_norm_eps` from the published configuration.
const LAYER_NORM_EPS: f64 = 1e-5;
/// `quick_gelu` is `x * sigmoid(1.702 * x)`.
const QUICK_GELU_SCALE: f64 = 1.702;
/// CLIP's `pad_token`, which its tokenizer configuration sets to the
/// end-of-text token rather than a dedicated one.
const PADDING_TOKEN: &str = "<|endoftext|>";

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn load(weights: &ModelWeights, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), device)?,
            bias: get(weights, &format!("{prefix}.bias"), device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&self.weight.t()?)?
            .broadcast_add(&self.bias)
            .map_err(Into::into)
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn load(weights: &ModelWeights, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), device)?,
            bias: get(weights, &format!("{prefix}.bias"), device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered
            .broadcast_div(&(variance + LAYER_NORM_EPS)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
            .map_err(Into::into)
    }
}

struct Layer {
    layer_norm1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    layer_norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// The loaded text tower.
pub struct ClipTextModel {
    config: ClipTextConfig,
    token_embedding: Tensor,
    position_embedding: Tensor,
    layers: Vec<Layer>,
    final_layer_norm: LayerNorm,
    causal_mask: Tensor,
    device: Device,
}

impl ClipTextModel {
    /// Load from a checkpoint directory holding `model.safetensors`.
    pub fn load(config: ClipTextConfig, weights: &ModelWeights, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("text_model.encoder.layers.{index}");
            layers.push(Layer {
                layer_norm1: LayerNorm::load(weights, &format!("{prefix}.layer_norm1"), device)?,
                q_proj: Linear::load(weights, &format!("{prefix}.self_attn.q_proj"), device)?,
                k_proj: Linear::load(weights, &format!("{prefix}.self_attn.k_proj"), device)?,
                v_proj: Linear::load(weights, &format!("{prefix}.self_attn.v_proj"), device)?,
                out_proj: Linear::load(weights, &format!("{prefix}.self_attn.out_proj"), device)?,
                layer_norm2: LayerNorm::load(weights, &format!("{prefix}.layer_norm2"), device)?,
                fc1: Linear::load(weights, &format!("{prefix}.mlp.fc1"), device)?,
                fc2: Linear::load(weights, &format!("{prefix}.mlp.fc2"), device)?,
            });
        }
        let positions = config.max_position_embeddings;
        let mask: Vec<f32> = (0..positions)
            .flat_map(|query| {
                (0..positions).map(move |key| if key > query { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        Ok(Self {
            config,
            token_embedding: get(
                weights,
                "text_model.embeddings.token_embedding.weight",
                device,
            )?,
            position_embedding: get(
                weights,
                "text_model.embeddings.position_embedding.weight",
                device,
            )?,
            layers,
            final_layer_norm: LayerNorm::load(weights, "text_model.final_layer_norm", device)?,
            causal_mask: Tensor::from_vec(mask, (1, 1, positions, positions), device)?,
            device: device.clone(),
        })
    }

    /// Open the published checkpoint directory directly.
    pub fn open(
        directory: impl AsRef<Path>,
        config: ClipTextConfig,
        device: &Device,
    ) -> Result<Self> {
        use ff_core::weights::{CachePolicy, WeightSource};

        let weights =
            ModelWeights::open(directory.as_ref(), WeightSource::Mmap, CachePolicy::new(1))
                .context("failed to open the CLIP text checkpoint")?;
        Self::load(config, &weights, device)
    }

    pub fn config(&self) -> ClipTextConfig {
        self.config
    }

    /// `last_hidden_state` for a batch of already-tokenized prompts.
    ///
    /// `tokens` is `[batch, max_position_embeddings]`, padded as the reference
    /// pads: to the full length, every time.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, length) = tokens.dims2()?;
        anyhow::ensure!(
            length == self.config.max_position_embeddings,
            "CLIP conditioning is padded to {} positions, got {length}",
            self.config.max_position_embeddings
        );
        let flat = tokens.flatten_all()?;
        let embedded = self.token_embedding.index_select(&flat, 0)?.reshape((
            batch,
            length,
            self.config.hidden_size,
        ))?;
        let positions = Tensor::arange(0u32, length as u32, &self.device)?;
        let position_embedding = self.position_embedding.index_select(&positions, 0)?;
        let mut hidden = embedded.broadcast_add(&position_embedding.unsqueeze(0)?)?;

        let heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim()?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        for layer in &self.layers {
            let residual = hidden.clone();
            let normalized = layer.layer_norm1.forward(&hidden)?;
            let shape = (batch, length, heads, head_dim);
            let q = layer
                .q_proj
                .forward(&normalized)?
                .reshape(shape)?
                .transpose(1, 2)?
                .contiguous()?;
            let k = layer
                .k_proj
                .forward(&normalized)?
                .reshape(shape)?
                .transpose(1, 2)?
                .contiguous()?;
            let v = layer
                .v_proj
                .forward(&normalized)?
                .reshape(shape)?
                .transpose(1, 2)?
                .contiguous()?;
            let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
            let scores = scores.broadcast_add(&self.causal_mask)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attended = weights
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, length, heads * head_dim))?;
            hidden = (residual + layer.out_proj.forward(&attended)?)?;

            let residual = hidden.clone();
            let normalized = layer.layer_norm2.forward(&hidden)?;
            let expanded = layer.fc1.forward(&normalized)?;
            let activated = quick_gelu(&expanded)?;
            hidden = (residual + layer.fc2.forward(&activated)?)?;
        }
        self.final_layer_norm.forward(&hidden)
    }
}

/// `x * sigmoid(1.702 * x)`, CLIP's activation.
fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::sigmoid(&(x * QUICK_GELU_SCALE)?)?;
    (x * gate).map_err(Into::into)
}

fn get(weights: &ModelWeights, name: &str, device: &Device) -> Result<Tensor> {
    weights
        .load(name, device)
        .with_context(|| format!("failed to load {name}"))?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

/// CLIP's BPE tokenizer, configured as `encode_text` configures it.
pub struct ClipTokenizer {
    tokenizer: tokenizers::Tokenizer,
    length: usize,
}

impl ClipTokenizer {
    /// Load `tokenizer.json` from a checkpoint directory.
    pub fn open(directory: impl AsRef<Path>, length: usize) -> Result<Self> {
        let path = directory.as_ref().join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| anyhow::anyhow!("failed to read {}: {error}", path.display()))?;
        let pad_id = tokenizer
            .token_to_id(PADDING_TOKEN)
            .with_context(|| format!("CLIP vocabulary has no {PADDING_TOKEN} token to pad with"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: length,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("failed to configure truncation: {error}"))?
            .with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(length),
                pad_id,
                pad_token: PADDING_TOKEN.to_owned(),
                ..Default::default()
            }));
        Ok(Self { tokenizer, length })
    }

    /// Token ids for a batch of prompts, as `[batch, length]`.
    pub fn encode(&self, prompts: &[&str], device: &Device) -> Result<Tensor> {
        let encodings = self
            .tokenizer
            .encode_batch(prompts.to_vec(), true)
            .map_err(|error| anyhow::anyhow!("failed to tokenize: {error}"))?;
        let mut ids = Vec::with_capacity(prompts.len() * self.length);
        for encoding in &encodings {
            let row = encoding.get_ids();
            anyhow::ensure!(
                row.len() == self.length,
                "tokenizer produced {} ids, expected {}",
                row.len(),
                self.length
            );
            ids.extend_from_slice(row);
        }
        Tensor::from_vec(ids, (prompts.len(), self.length), device).map_err(Into::into)
    }
}
