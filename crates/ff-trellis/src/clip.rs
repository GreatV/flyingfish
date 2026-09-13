//! CLIP's vision tower and projected text/image similarity, using the existing
//! text encoder and streaming one vision projection at a time.
use crate::clip_text::{ClipTextConfig, ClipTextModel};
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use serde_json::Value;
use std::path::Path;

pub struct ClipModel {
    weights: ModelWeights,
    text: ClipTextModel,
    config: Value,
    device: Device,
}

pub struct Similarity {
    pub image_features: Tensor,
    pub text_features: Tensor,
    pub logits_per_image: Tensor,
    pub probabilities: Tensor,
}

fn dimension(config: &Value, name: &str) -> Result<usize> {
    let n = usize::try_from(
        config[name]
            .as_u64()
            .with_context(|| format!("missing CLIP dimension {name}"))?,
    )?;
    anyhow::ensure!(n > 0, "CLIP dimension {name} must be positive");
    Ok(n)
}

impl ClipModel {
    pub fn open(
        root: &Path,
        device: &Device,
        source: WeightSource,
        cache: CachePolicy,
    ) -> Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        anyhow::ensure!(
            config["architectures"][0] == "CLIPModel",
            "expected a CLIPModel checkpoint"
        );
        let text_config = &config["text_config"];
        let vision = &config["vision_config"];
        for tower in [text_config, vision] {
            anyhow::ensure!(
                tower["hidden_act"] == "quick_gelu" && tower["layer_norm_eps"] == 1e-5,
                "CLIP tower requires quick_gelu and layer_norm_eps=1e-5"
            );
        }
        let hidden = dimension(vision, "hidden_size")?;
        let heads = dimension(vision, "num_attention_heads")?;
        let image_size = dimension(vision, "image_size")?;
        let patch_size = dimension(vision, "patch_size")?;
        anyhow::ensure!(
            hidden.is_multiple_of(heads) && image_size.is_multiple_of(patch_size),
            "invalid CLIP vision geometry"
        );
        dimension(vision, "num_hidden_layers")?;
        let text_config = ClipTextConfig {
            hidden_size: dimension(text_config, "hidden_size")?,
            intermediate_size: dimension(text_config, "intermediate_size")?,
            num_hidden_layers: dimension(text_config, "num_hidden_layers")?,
            num_attention_heads: dimension(text_config, "num_attention_heads")?,
            max_position_embeddings: dimension(text_config, "max_position_embeddings")?,
            vocab_size: dimension(text_config, "vocab_size")?,
        };
        text_config.head_dim()?;
        let weights = ModelWeights::open(root, source, cache)?;
        let text = ClipTextModel::load(text_config, &weights, device)?;
        Ok(Self {
            weights,
            text,
            config,
            device: device.clone(),
        })
    }
    pub fn image_size(&self) -> Result<usize> {
        dimension(&self.config["vision_config"], "image_size")
    }
    pub fn text_length(&self) -> usize {
        self.text.config().max_position_embeddings
    }
    fn get(&self, name: &str) -> Result<Tensor> {
        Ok(self
            .weights
            .load(name, &self.device)?
            .to_dtype(DType::F32)?)
    }
    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let x = x.broadcast_matmul(&self.get(&format!("{prefix}.weight"))?.t()?)?;
        if self.weights.contains(&format!("{prefix}.bias")) {
            Ok(x.broadcast_add(&self.get(&format!("{prefix}.bias"))?)?)
        } else {
            Ok(x)
        }
    }
    fn norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        Ok(centered
            .broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.get(&format!("{prefix}.weight"))?)?
            .broadcast_add(&self.get(&format!("{prefix}.bias"))?)?)
    }

    /// Pixel values must already be RGB, resized/cropped to the configured
    /// square and CLIP-normalized, in `[batch, 3, height, width]` order.
    pub fn image_features(&self, pixels: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = pixels.dims4()?;
        let size = self.image_size()?;
        anyhow::ensure!(
            batch > 0 && channels == 3 && height == size && width == size,
            "CLIP pixels must have shape [batch,3,{size},{size}]"
        );
        let config = &self.config["vision_config"];
        let patch = dimension(config, "patch_size")?;
        let h = dimension(config, "num_attention_heads")?;
        let hidden = dimension(config, "hidden_size")?;
        let dim = hidden / h;
        let pixels = pixels
            .to_device(&self.device)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let embedded = pixels.conv2d(
            &self.get("vision_model.embeddings.patch_embedding.weight")?,
            0,
            patch,
            1,
            1,
        )?;
        let patches = embedded.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let cls = self
            .get("vision_model.embeddings.class_embedding")?
            .reshape((1, 1, hidden))?
            .broadcast_as((batch, 1, hidden))?;
        let mut x = Tensor::cat(&[cls, patches], 1)?.broadcast_add(
            &self
                .get("vision_model.embeddings.position_embedding.weight")?
                .unsqueeze(0)?,
        )?;
        x = self.norm(&x, "vision_model.pre_layrnorm")?;
        let length = x.dim(1)?;
        for layer in 0..dimension(config, "num_hidden_layers")? {
            let p = format!("vision_model.encoder.layers.{layer}");
            let norm = self.norm(&x, &format!("{p}.layer_norm1"))?;
            let project = |name: &str| -> Result<Tensor> {
                Ok(self
                    .linear(&norm, &format!("{p}.self_attn.{name}"))?
                    .reshape((batch, length, h, dim))?
                    .transpose(1, 2)?
                    .contiguous()?)
            };
            let q = project("q_proj")?;
            let k = project("k_proj")?;
            let v = project("v_proj")?;
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (dim as f64).sqrt())?;
            let output = candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, length, hidden))?;
            x = (&x + self.linear(&output, &format!("{p}.self_attn.out_proj"))?)?;
            let ff = self.linear(
                &self.norm(&x, &format!("{p}.layer_norm2"))?,
                &format!("{p}.mlp.fc1"),
            )?;
            let gate = candle_nn::ops::sigmoid(&(&ff * 1.702)?)?;
            x = (&x + self.linear(&(&ff * gate)?, &format!("{p}.mlp.fc2"))?)?;
        }
        let pooled = self.norm(
            &x.narrow(1, 0, 1)?.squeeze(1)?,
            "vision_model.post_layernorm",
        )?;
        unit(&self.linear(&pooled, "visual_projection")?)
    }

    pub fn text_features(&self, tokens: &Tensor) -> Result<Tensor> {
        let tokens = tokens.to_device(&self.device)?.to_dtype(DType::U32)?;
        let hidden = self.text.forward(&tokens)?;
        let (batch, length, width) = hidden.dims3()?;
        anyhow::ensure!(batch > 0, "CLIP text batch must be nonempty");
        let eos = self.config["text_config"]["eos_token_id"]
            .as_u64()
            .context("missing CLIP EOS")? as u32;
        let rows = tokens.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let mut indices = Vec::with_capacity(batch);
        for (batch, row) in rows.iter().enumerate() {
            let index = if eos == 2 {
                let max = row.iter().max().context("empty CLIP token row")?;
                row.iter().position(|id| id == max)
            } else {
                row.iter().position(|&id| id == eos)
            }
            .context("CLIP token row has no EOS")?;
            indices.push(u32::try_from(batch * length + index)?);
        }
        let pooled = hidden
            .reshape((batch * length, width))?
            .index_select(&Tensor::from_vec(indices, batch, &self.device)?, 0)?;
        unit(&self.linear(&pooled, "text_projection")?)
    }

    pub fn score(&self, pixels: &Tensor, tokens: &Tensor) -> Result<Similarity> {
        let image_features = self.image_features(pixels)?;
        let text_features = self.text_features(tokens)?;
        let logits_per_image = image_features
            .matmul(&text_features.t()?)?
            .broadcast_mul(&self.get("logit_scale")?.exp()?)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&logits_per_image)?;
        Ok(Similarity {
            image_features,
            text_features,
            logits_per_image,
            probabilities,
        })
    }
}

fn unit(x: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_div(&x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?)?)
}
