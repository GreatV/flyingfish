//! DINOv2 with registers as TRELLIS-1 consumes it: all pre-final-norm tokens,
//! including CLS/registers, followed by an unaffine LayerNorm with eps=1e-5.
use crate::slat_ops::{Ops, attend, layer_norm};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use std::path::Path;

pub struct DinoV2 {
    ops: Ops,
    size: usize,
    patch: usize,
    width: usize,
    heads: usize,
    layers: usize,
    registers: usize,
    eps: f64,
}

impl DinoV2 {
    pub fn open(root: &Path, device: &Device, query_chunk: usize) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        anyhow::ensure!(
            config["architectures"][0] == "Dinov2WithRegistersModel"
                && config["hidden_act"] == "gelu"
                && config["use_swiglu_ffn"] == false,
            "conditioner must be DINOv2 with registers and a GELU MLP"
        );
        let n = |key: &str| -> Result<usize> {
            let value = usize::try_from(
                config[key]
                    .as_u64()
                    .with_context(|| format!("missing DINOv2 {key}"))?,
            )?;
            anyhow::ensure!(value > 0, "DINOv2 {key} must be positive");
            Ok(value)
        };
        let size = n("image_size")?;
        let patch = n("patch_size")?;
        let width = n("hidden_size")?;
        let heads = n("num_attention_heads")?;
        anyhow::ensure!(
            size.is_multiple_of(patch) && width.is_multiple_of(heads) && query_chunk > 0,
            "invalid DINOv2 geometry"
        );
        let eps = config["layer_norm_eps"]
            .as_f64()
            .context("missing DINOv2 layer_norm_eps")?;
        anyhow::ensure!(
            eps.is_finite() && eps > 0.,
            "invalid DINOv2 LayerNorm epsilon"
        );
        let weights = ModelWeights::open(root, WeightSource::Mmap, CachePolicy::new(1))?;
        Ok(Self {
            size,
            patch,
            width,
            heads,
            layers: n("num_hidden_layers")?,
            registers: n("num_register_tokens")?,
            eps,
            ops: Ops {
                weights,
                device: device.clone(),
                dtype: DType::F32,
                query_chunk,
                voxel_chunk: 256,
            },
        })
    }

    /// Prepared RGB values in `[0,1]`, shape `[1,3,image_size,image_size]`.
    /// Background removal, crop selection and resampling belong to the caller.
    pub fn encode(&self, rgb: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            rgb.dims() == [1, 3, self.size, self.size],
            "DINOv2 expects one prepared {}x{} RGB image",
            self.size,
            self.size
        );
        let pixels = rgb.to_device(&self.ops.device)?.to_dtype(DType::F32)?;
        let mean =
            Tensor::new(&[0.485f32, 0.456, 0.406], &self.ops.device)?.reshape((1, 3, 1, 1))?;
        let std =
            Tensor::new(&[0.229f32, 0.224, 0.225], &self.ops.device)?.reshape((1, 3, 1, 1))?;
        let pixels = pixels
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .contiguous()?;
        let get = |name: &str| self.ops.get(name, DType::F32);
        let patches = pixels
            .conv2d(
                &get("embeddings.patch_embeddings.projection.weight")?,
                0,
                self.patch,
                1,
                1,
            )?
            .broadcast_add(
                &get("embeddings.patch_embeddings.projection.bias")?
                    .reshape((1, self.width, 1, 1))?,
            )?
            .flatten_from(2)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut x = Tensor::cat(&[get("embeddings.cls_token")?, patches], 1)?
            .broadcast_add(&get("embeddings.position_embeddings")?)?;
        let registers = get("embeddings.register_tokens")?;
        anyhow::ensure!(
            registers.dims() == [1, self.registers, self.width],
            "DINOv2 register shape mismatch"
        );
        x = Tensor::cat(
            &[
                x.narrow(1, 0, 1)?,
                registers,
                x.narrow(1, 1, x.dim(1)? - 1)?,
            ],
            1,
        )?
        .squeeze(0)?;
        let rows = x.dim(0)?;
        for layer in 0..self.layers {
            let p = format!("encoder.layer.{layer}");
            let norm = self.ops.norm(&x, Some(&format!("{p}.norm1")), self.eps)?;
            let project = |name: &str| -> Result<Tensor> {
                Ok(self
                    .ops
                    .linear(&norm, &format!("{p}.attention.attention.{name}"))?
                    .reshape((rows, self.heads, self.width / self.heads))?)
            };
            let attention = attend(
                &project("query")?,
                &project("key")?,
                &project("value")?,
                self.ops.query_chunk,
            )?
            .reshape((rows, self.width))?;
            let attention = self
                .ops
                .linear(&attention, &format!("{p}.attention.output.dense"))?
                .broadcast_mul(&get(&format!("{p}.layer_scale1.lambda1"))?)?;
            x = (&x + attention)?;
            let norm = self.ops.norm(&x, Some(&format!("{p}.norm2")), self.eps)?;
            let h = self
                .ops
                .linear(&norm, &format!("{p}.mlp.fc1"))?
                .gelu_erf()?;
            x = (&x
                + self
                    .ops
                    .linear(&h, &format!("{p}.mlp.fc2"))?
                    .broadcast_mul(&get(&format!("{p}.layer_scale2.lambda1"))?)?)?;
        }
        Ok(layer_norm(&x, 1e-5)?.unsqueeze(0)?)
    }
}
