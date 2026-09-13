//! DINOv3 conditioning for TRELLIS.2, including patch-only 2D RoPE.
use crate::slat_ops::{Ops, attend, layer_norm};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use std::path::Path;

pub struct DinoV3 {
    ops: Ops,
    patch: usize,
    width: usize,
    heads: usize,
    layers: usize,
    registers: usize,
    eps: f64,
    theta: f64,
    prefix: String,
}
impl DinoV3 {
    pub fn open(root: &Path, device: &Device, query_chunk: usize) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        anyhow::ensure!(
            config["architectures"][0] == "DINOv3ViTModel"
                && config["hidden_act"] == "gelu"
                && config["use_gated_mlp"] == false,
            "TRELLIS.2 requires a DINOv3 ViT with a GELU MLP"
        );
        let n = |key: &str| -> Result<usize> {
            let value = usize::try_from(
                config[key]
                    .as_u64()
                    .with_context(|| format!("missing DINOv3 {key}"))?,
            )?;
            anyhow::ensure!(value > 0, "DINOv3 {key} must be positive");
            Ok(value)
        };
        let width = n("hidden_size")?;
        let heads = n("num_attention_heads")?;
        let eps = config["layer_norm_eps"]
            .as_f64()
            .context("missing DINOv3 normalization epsilon")?;
        let theta = config["rope_theta"]
            .as_f64()
            .context("missing DINOv3 RoPE theta")?;
        anyhow::ensure!(
            width.is_multiple_of(heads)
                && (width / heads).is_multiple_of(4)
                && query_chunk > 0
                && eps.is_finite()
                && eps > 0.
                && theta.is_finite()
                && theta > 0.,
            "invalid DINOv3 geometry or numerics"
        );
        let weights = ModelWeights::open(root, WeightSource::Mmap, CachePolicy::new(1))?;
        let prefix = ["layer", "model.layer", "encoder.layer"]
            .into_iter()
            .find(|p| weights.contains(&format!("{p}.0.norm1.weight")))
            .context("missing DINOv3 transformer layers")?
            .to_string();
        Ok(Self {
            ops: Ops {
                weights,
                device: device.clone(),
                dtype: DType::F32,
                query_chunk,
                voxel_chunk: 256,
            },
            patch: n("patch_size")?,
            width,
            heads,
            layers: n("num_hidden_layers")?,
            registers: n("num_register_tokens")?,
            eps,
            theta,
            prefix,
        })
    }

    pub fn encode(&self, rgb: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = rgb.dims4()?;
        anyhow::ensure!(
            batch == 1
                && channels == 3
                && height > 0
                && width > 0
                && height.is_multiple_of(self.patch)
                && width.is_multiple_of(self.patch),
            "DINOv3 input must be one RGB image with dimensions divisible by its patch size"
        );
        let device = &self.ops.device;
        let mean = Tensor::new(&[0.485f32, 0.456, 0.406], device)?.reshape((1, 3, 1, 1))?;
        let std = Tensor::new(&[0.229f32, 0.224, 0.225], device)?.reshape((1, 3, 1, 1))?;
        let pixels = rgb
            .to_device(device)?
            .to_dtype(DType::F32)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .contiguous()?;
        let get = |name: &str| self.ops.get(name, DType::F32);
        let patches = pixels
            .conv2d(
                &get("embeddings.patch_embeddings.weight")?,
                0,
                self.patch,
                1,
                1,
            )?
            .broadcast_add(
                &get("embeddings.patch_embeddings.bias")?.reshape((1, self.width, 1, 1))?,
            )?
            .flatten_from(2)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut x = Tensor::cat(
            &[
                get("embeddings.cls_token")?,
                get("embeddings.register_tokens")?,
                patches,
            ],
            1,
        )?
        .squeeze(0)?;
        let rows = x.dim(0)?;
        let prefix = self.registers + 1;
        let patch_rows = rows - prefix;
        let dim = self.width / self.heads;
        let hp = height / self.patch;
        let wp = width / self.patch;
        let coords = (0..hp)
            .flat_map(|y| {
                (0..wp).flat_map(move |x| {
                    [
                        2. * ((y as f32 + 0.5) / hp as f32) - 1.,
                        2. * ((x as f32 + 0.5) / wp as f32) - 1.,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let frequencies = (0..dim / 4)
            .map(|i| 1f32 / (self.theta as f32).powf((4 * i) as f32 / dim as f32))
            .collect::<Vec<_>>();
        let angles = (Tensor::from_vec(coords, (patch_rows, 2, 1), device)?
            * std::f64::consts::TAU)?
            .broadcast_mul(&Tensor::from_vec(frequencies, (1, 1, dim / 4), device)?)?
            .reshape((patch_rows, dim / 2))?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        let rotate = |q: &Tensor| -> Result<Tensor> {
            let patches = q
                .narrow(0, prefix, patch_rows)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let rotated = candle_nn::rotary_emb::rope(&patches, &cos, &sin)?
                .squeeze(0)?
                .transpose(0, 1)?;
            Ok(Tensor::cat(&[q.narrow(0, 0, prefix)?, rotated], 0)?)
        };
        for layer in 0..self.layers {
            let p = format!("{}.{layer}", self.prefix);
            let norm = self.ops.norm(&x, Some(&format!("{p}.norm1")), self.eps)?;
            let project = |name: &str| -> Result<Tensor> {
                Ok(self
                    .ops
                    .linear(&norm, &format!("{p}.attention.{name}"))?
                    .reshape((rows, self.heads, dim))?)
            };
            let q = rotate(&project("q_proj")?)?;
            let k = rotate(&project("k_proj")?)?;
            let v = project("v_proj")?;
            let out = attend(&q, &k, &v, self.ops.query_chunk)?.reshape((rows, self.width))?;
            x = (&x
                + self
                    .ops
                    .linear(&out, &format!("{p}.attention.o_proj"))?
                    .broadcast_mul(&get(&format!("{p}.layer_scale1.lambda1"))?)?)?;
            let norm = self.ops.norm(&x, Some(&format!("{p}.norm2")), self.eps)?;
            let h = self
                .ops
                .linear(&norm, &format!("{p}.mlp.up_proj"))?
                .gelu_erf()?;
            x = (&x
                + self
                    .ops
                    .linear(&h, &format!("{p}.mlp.down_proj"))?
                    .broadcast_mul(&get(&format!("{p}.layer_scale2.lambda1"))?)?)?;
        }
        Ok(layer_norm(&x, 1e-5)?.unsqueeze(0)?)
    }
}
