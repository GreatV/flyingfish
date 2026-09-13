//! The dense sparse-structure flow transformer.
//!
//! Transcribed from `microsoft/TRELLIS`, `trellis/models/sparse_structure_flow.py`
//! and the transformer and attention modules it composes. The operation order,
//! the places the reference widens to F32 and narrows back, and the eps values
//! are the parts that decide the numbers, so they are reproduced rather than
//! rewritten:
//!
//! - the block torso runs in the checkpoint's own dtype, F16 for every
//!   published sparse-structure flow model;
//! - `LayerNorm32` computes in F32 and casts back, with eps `1e-6` inside the
//!   blocks and PyTorch's default `1e-5` for the final unaffine normalization
//!   `F.layer_norm(h, h.shape[-1:])` before the output projection;
//! - `MultiHeadRMSNorm` is `l2_normalize(x.float()) * gamma * sqrt(head_dim)`,
//!   which is an RMS normalization written through `F.normalize`, with that
//!   function's `1e-12` floor on the norm.
//!
//! Only the TRELLIS-1 block shape is implemented. The 1.3B model in
//! `TRELLIS.2-4B` shares one modulation projection across blocks and gives each
//! block its own `modulation` parameter, which is a different algebra from a
//! different codebase; [`SparseStructureFlow::load`] refuses it rather than
//! guessing at it.

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use ff_core::weights::ModelWeights;

use crate::config::{PositionEmbeddingMode, SparseStructureFlowArgs};

/// Eps of the affine and unaffine `LayerNorm32`s inside a block.
const BLOCK_NORM_EPS: f64 = 1e-6;
/// Eps of `F.layer_norm(h, h.shape[-1:])`, which takes PyTorch's default.
const OUTPUT_NORM_EPS: f64 = 1e-5;
/// Floor `F.normalize` puts under the norm it divides by.
const L2_NORMALIZE_EPS: f64 = 1e-12;
/// Width of the sinusoidal timestep encoding the embedder consumes.
const TIMESTEP_FREQUENCY_CHANNELS: usize = 256;
const TIMESTEP_MAX_PERIOD: f64 = 10_000.0;

/// A `[out, in]` weight with its bias, applied as PyTorch's `nn.Linear`.
struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn load(weights: &ModelWeights, prefix: &str, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: load(weights, &format!("{prefix}.weight"), device, dtype)?,
            bias: load(weights, &format!("{prefix}.bias"), device, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = x.broadcast_matmul(&self.weight.t()?)?;
        output.broadcast_add(&self.bias).map_err(Into::into)
    }
}

/// `LayerNorm32` with learned scale and shift.
struct AffineLayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl AffineLayerNorm {
    fn load(weights: &ModelWeights, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: load(weights, &format!("{prefix}.weight"), device, DType::F32)?,
            bias: load(weights, &format!("{prefix}.bias"), device, DType::F32)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let normalized = layer_norm(x, BLOCK_NORM_EPS)?;
        normalized
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

/// `LayerNorm32` without affine parameters: widen to F32, normalize, narrow.
fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let wide = x.to_dtype(DType::F32)?;
    let mean = wide.mean_keepdim(D::Minus1)?;
    let centered = wide.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered
        .broadcast_div(&(variance + eps)?.sqrt()?)
        .map_err(Into::into)
}

/// `MultiHeadRMSNorm`: `F.normalize(x.float(), dim=-1) * gamma * sqrt(dim)`.
struct MultiHeadRmsNorm {
    /// `[heads, head_dim]`, already multiplied by the reference's `scale`.
    gain: Tensor,
}

impl MultiHeadRmsNorm {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let gamma = load(weights, &format!("{prefix}.gamma"), device, DType::F32)?;
        let gain = (gamma * (head_dim as f64).sqrt())?;
        Ok(Self { gain })
    }

    /// `x` is `[batch, tokens, heads, head_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let wide = x.to_dtype(DType::F32)?;
        let norm = wide.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm = norm.maximum(L2_NORMALIZE_EPS)?;
        wide.broadcast_div(&norm)?
            .broadcast_mul(&self.gain)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

struct SelfAttention {
    to_qkv: Linear,
    to_out: Linear,
    q_norm: Option<MultiHeadRmsNorm>,
    k_norm: Option<MultiHeadRmsNorm>,
    heads: usize,
    head_dim: usize,
}

struct CrossAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    q_norm: Option<MultiHeadRmsNorm>,
    k_norm: Option<MultiHeadRmsNorm>,
    heads: usize,
    head_dim: usize,
}

struct Block {
    modulation: Linear,
    self_attn: SelfAttention,
    norm2: AffineLayerNorm,
    cross_attn: CrossAttention,
    mlp_in: Linear,
    mlp_out: Linear,
}

/// The loaded model.
pub struct SparseStructureFlow {
    args: SparseStructureFlowArgs,
    dtype: DType,
    input_layer: Linear,
    pos_emb: Tensor,
    t_embed_in: Linear,
    t_embed_out: Linear,
    blocks: Vec<Block>,
    out_layer: Linear,
    device: Device,
}

impl SparseStructureFlow {
    pub fn load(
        args: &SparseStructureFlowArgs,
        weights: &ModelWeights,
        device: &Device,
    ) -> Result<Self> {
        Self::load_with_dtype(args, weights, device, None)
    }

    /// As [`Self::load`], with the torso precision overridden.
    ///
    /// Only the parity gate uses the override: running both implementations in
    /// F32 separates a transcription error from F16 rounding, which is the
    /// difference between a bug and the cost of the reference's own dtype
    /// policy.
    pub fn load_with_dtype(
        args: &SparseStructureFlowArgs,
        weights: &ModelWeights,
        device: &Device,
        torso_dtype: Option<DType>,
    ) -> Result<Self> {
        anyhow::ensure!(
            !args.share_mod,
            "this sparse-structure flow model shares one modulation projection across blocks, \
             which is the TRELLIS.2 block algebra; only the TRELLIS-1 shape is transcribed here"
        );
        anyhow::ensure!(
            args.pe_mode == PositionEmbeddingMode::Ape,
            "only the learned absolute position embedding is transcribed; this model uses rope"
        );
        let dtype = torso_dtype.unwrap_or(if args.use_fp16 && !device.is_cpu() {
            DType::F16
        } else {
            DType::F32
        });
        let head_dim = args.head_dim()?;

        let mut blocks = Vec::with_capacity(args.num_blocks);
        for index in 0..args.num_blocks {
            let prefix = format!("blocks.{index}");
            blocks.push(Block {
                modulation: Linear::load(
                    weights,
                    &format!("{prefix}.adaLN_modulation.1"),
                    device,
                    dtype,
                )?,
                self_attn: SelfAttention {
                    to_qkv: Linear::load(
                        weights,
                        &format!("{prefix}.self_attn.to_qkv"),
                        device,
                        dtype,
                    )?,
                    to_out: Linear::load(
                        weights,
                        &format!("{prefix}.self_attn.to_out"),
                        device,
                        dtype,
                    )?,
                    q_norm: args
                        .qk_rms_norm
                        .then(|| {
                            MultiHeadRmsNorm::load(
                                weights,
                                &format!("{prefix}.self_attn.q_rms_norm"),
                                head_dim,
                                device,
                            )
                        })
                        .transpose()?,
                    k_norm: args
                        .qk_rms_norm
                        .then(|| {
                            MultiHeadRmsNorm::load(
                                weights,
                                &format!("{prefix}.self_attn.k_rms_norm"),
                                head_dim,
                                device,
                            )
                        })
                        .transpose()?,
                    heads: args.num_heads,
                    head_dim,
                },
                norm2: AffineLayerNorm::load(weights, &format!("{prefix}.norm2"), device)?,
                cross_attn: CrossAttention {
                    to_q: Linear::load(
                        weights,
                        &format!("{prefix}.cross_attn.to_q"),
                        device,
                        dtype,
                    )?,
                    to_kv: Linear::load(
                        weights,
                        &format!("{prefix}.cross_attn.to_kv"),
                        device,
                        dtype,
                    )?,
                    to_out: Linear::load(
                        weights,
                        &format!("{prefix}.cross_attn.to_out"),
                        device,
                        dtype,
                    )?,
                    q_norm: args
                        .qk_rms_norm_cross
                        .then(|| {
                            MultiHeadRmsNorm::load(
                                weights,
                                &format!("{prefix}.cross_attn.q_rms_norm"),
                                head_dim,
                                device,
                            )
                        })
                        .transpose()?,
                    k_norm: args
                        .qk_rms_norm_cross
                        .then(|| {
                            MultiHeadRmsNorm::load(
                                weights,
                                &format!("{prefix}.cross_attn.k_rms_norm"),
                                head_dim,
                                device,
                            )
                        })
                        .transpose()?,
                    heads: args.num_heads,
                    head_dim,
                },
                mlp_in: Linear::load(weights, &format!("{prefix}.mlp.mlp.0"), device, dtype)?,
                mlp_out: Linear::load(weights, &format!("{prefix}.mlp.mlp.2"), device, dtype)?,
            });
        }

        Ok(Self {
            args: args.clone(),
            dtype,
            input_layer: Linear::load(weights, "input_layer", device, DType::F32)?,
            pos_emb: load(weights, "pos_emb", device, DType::F32)?,
            t_embed_in: Linear::load(weights, "t_embedder.mlp.0", device, DType::F32)?,
            t_embed_out: Linear::load(weights, "t_embedder.mlp.2", device, DType::F32)?,
            blocks,
            out_layer: Linear::load(weights, "out_layer", device, DType::F32)?,
            device: device.clone(),
        })
    }

    pub fn args(&self) -> &SparseStructureFlowArgs {
        &self.args
    }

    /// Velocity at `timesteps` for latents `x` under conditioning `cond`.
    ///
    /// `x` is `[batch, in_channels, resolution, resolution, resolution]`,
    /// `timesteps` is `[batch]`, and `cond` is `[batch, tokens, cond_channels]`.
    pub fn forward(&self, x: &Tensor, timesteps: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let resolution = self.args.resolution;
        let batch = x.dim(0)?;
        anyhow::ensure!(
            x.dims()
                == [
                    batch,
                    self.args.in_channels,
                    resolution,
                    resolution,
                    resolution
                ],
            "sparse-structure latent has shape {:?}, expected {:?}",
            x.dims(),
            [
                batch,
                self.args.in_channels,
                resolution,
                resolution,
                resolution
            ]
        );
        anyhow::ensure!(
            cond.dim(0)? == batch && cond.dim(D::Minus1)? == self.args.cond_channels,
            "conditioning has shape {:?}, expected [{batch}, tokens, {}]",
            cond.dims(),
            self.args.cond_channels
        );
        anyhow::ensure!(
            timesteps.dims() == [batch],
            "timesteps have shape {:?}, expected [{batch}]",
            timesteps.dims()
        );

        let patch = self.args.patch_size;
        let tokens = patchify(&x.to_dtype(DType::F32)?, patch)?;
        let channels = tokens.dim(1)?;
        let mut h = tokens
            .reshape((batch, channels, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        h = self.input_layer.forward(&h)?;
        h = h.broadcast_add(&self.pos_emb.unsqueeze(0)?)?;

        let modulation = self.timestep_embedding(timesteps)?;

        let mut h = h.to_dtype(self.dtype)?;
        let modulation = modulation.to_dtype(self.dtype)?;
        let cond = cond.to_dtype(self.dtype)?;
        for block in &self.blocks {
            h = block.forward(&h, &modulation, &cond)?;
        }

        let h = h.to_dtype(DType::F32)?;
        let h = layer_norm(&h, OUTPUT_NORM_EPS)?;
        let h = self.out_layer.forward(&h)?;

        let side = resolution / patch;
        let channels = h.dim(D::Minus1)?;
        let h = h
            .transpose(1, 2)?
            .reshape((batch, channels, side, side, side))?
            .contiguous()?;
        unpatchify(&h, patch)
    }

    /// `TimestepEmbedder`: a sinusoidal encoding through a two-layer MLP.
    fn timestep_embedding(&self, timesteps: &Tensor) -> Result<Tensor> {
        let half = TIMESTEP_FREQUENCY_CHANNELS / 2;
        let scale = -TIMESTEP_MAX_PERIOD.ln() / half as f64;
        let frequencies: Vec<f32> = (0..half)
            .map(|index| (scale * index as f64).exp() as f32)
            .collect();
        let frequencies = Tensor::from_vec(frequencies, (1, half), &self.device)?;
        let arguments = timesteps
            .to_dtype(DType::F32)?
            .reshape(((), 1))?
            .broadcast_mul(&frequencies)?;
        let encoding = Tensor::cat(&[arguments.cos()?, arguments.sin()?], D::Minus1)?;
        let hidden = self.t_embed_in.forward(&encoding)?;
        self.t_embed_out.forward(&candle_nn::ops::silu(&hidden)?)
    }
}

impl Block {
    fn forward(&self, x: &Tensor, modulation: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = self
            .modulation
            .forward(&candle_nn::ops::silu(modulation)?)?;
        let channels = x.dim(D::Minus1)?;
        let part = |index: usize| -> Result<Tensor> {
            Ok(modulation
                .narrow(D::Minus1, index * channels, channels)?
                .unsqueeze(1)?)
        };
        let (shift_msa, scale_msa, gate_msa) = (part(0)?, part(1)?, part(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (part(3)?, part(4)?, part(5)?);

        let h = layer_norm(x, BLOCK_NORM_EPS)?.to_dtype(x.dtype())?;
        let h = h
            .broadcast_mul(&(&scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let h = self.self_attn.forward(&h)?;
        let x = (x + h.broadcast_mul(&gate_msa)?)?;

        let h = self.norm2.forward(&x)?;
        let h = self.cross_attn.forward(&h, cond)?;
        let x = (&x + h)?;

        let h = layer_norm(&x, BLOCK_NORM_EPS)?.to_dtype(x.dtype())?;
        let h = h
            .broadcast_mul(&(&scale_mlp + 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        let h = self.mlp_in.forward(&h)?;
        let h = h.gelu()?;
        let h = self.mlp_out.forward(&h)?;
        (x + h.broadcast_mul(&gate_mlp)?).map_err(Into::into)
    }
}

impl SelfAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let qkv = self.to_qkv.forward(x)?;
        let qkv = qkv.reshape((batch, length, 3, self.heads, self.head_dim))?;
        let mut q = qkv.i((.., .., 0))?.contiguous()?;
        let mut k = qkv.i((.., .., 1))?.contiguous()?;
        let v = qkv.i((.., .., 2))?.contiguous()?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        let attended = attend(&q, &k, &v, self.head_dim)?;
        self.to_out
            .forward(&attended.reshape((batch, length, self.heads * self.head_dim))?)
    }
}

impl CrossAttention {
    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let (batch, length, _) = x.dims3()?;
        let context = cond.dim(1)?;
        let mut q = self
            .to_q
            .forward(x)?
            .reshape((batch, length, self.heads, self.head_dim))?;
        let kv =
            self.to_kv
                .forward(cond)?
                .reshape((batch, context, 2, self.heads, self.head_dim))?;
        let mut k = kv.i((.., .., 0))?.contiguous()?;
        let v = kv.i((.., .., 1))?.contiguous()?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        let attended = attend(&q, &k, &v, self.head_dim)?;
        self.to_out
            .forward(&attended.reshape((batch, length, self.heads * self.head_dim))?)
    }
}

/// Scaled dot-product attention over `[batch, tokens, heads, head_dim]`.
///
/// Accumulated in F32 even when the operands are F16, which is what the
/// backends the reference actually dispatches to do: xformers, FlashAttention
/// and PyTorch's fused SDPA all keep the score and the weighted sum in F32 and
/// write F16. Its `_naive_sdpa` fallback does not, and an F16 `q @ k^T` scaled
/// only afterwards overflows on this model — cross-attention queries reach
/// ±600 against keys near ±9, so a 64-wide dot product leaves the F16 range and
/// the softmax becomes NaN.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
    let dtype = q.dtype();
    let q = q.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?;
    let k = k.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?;
    let v = v.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    let weights = candle_nn::ops::softmax_last_dim(&scores)?;
    weights
        .matmul(&v)?
        .transpose(1, 2)?
        .to_dtype(dtype)?
        .contiguous()
        .map_err(Into::into)
}

/// `patchify`: fold each `patch^3` cube into the channel dimension.
fn patchify(x: &Tensor, patch: usize) -> Result<Tensor> {
    if patch == 1 {
        return Ok(x.clone());
    }
    let (batch, channels, depth, height, width) = x.dims5()?;
    for (axis, size) in [("depth", depth), ("height", height), ("width", width)] {
        anyhow::ensure!(
            size.is_multiple_of(patch),
            "sparse-structure {axis} {size} is not divisible by patch size {patch}"
        );
    }
    x.reshape(vec![
        batch,
        channels,
        depth / patch,
        patch,
        height / patch,
        patch,
        width / patch,
        patch,
    ])?
    .permute([0, 1, 3, 5, 7, 2, 4, 6])?
    .contiguous()?
    .reshape((
        batch,
        channels * patch.pow(3),
        depth / patch,
        height / patch,
        width / patch,
    ))
    .map_err(Into::into)
}

/// `unpatchify`: the inverse of [`patchify`].
fn unpatchify(x: &Tensor, patch: usize) -> Result<Tensor> {
    if patch == 1 {
        return Ok(x.clone());
    }
    let (batch, channels, depth, height, width) = x.dims5()?;
    let volume = patch.pow(3);
    anyhow::ensure!(
        channels.is_multiple_of(volume),
        "sparse-structure channel count {channels} is not divisible by {volume}"
    );
    x.reshape(vec![
        batch,
        channels / volume,
        patch,
        patch,
        patch,
        depth,
        height,
        width,
    ])?
    .permute([0, 1, 5, 2, 6, 3, 7, 4])?
    .contiguous()?
    .reshape((
        batch,
        channels / volume,
        depth * patch,
        height * patch,
        width * patch,
    ))
    .map_err(Into::into)
}

fn load(weights: &ModelWeights, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
    weights
        .load(name, device)
        .with_context(|| format!("failed to load {name}"))?
        .to_dtype(dtype)
        .map_err(Into::into)
}

use candle_core::IndexOp as _;

impl crate::sampler::VelocityModel for SparseStructureFlow {
    fn velocity(&self, x: &Tensor, timesteps: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.forward(x, timesteps, cond)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `patchify` and `unpatchify` are inverses, which is what lets the model
    /// fold a patch into the channel dimension and unfold it afterwards.
    #[test]
    fn patchify_round_trips_through_unpatchify() {
        let device = Device::Cpu;
        let x = Tensor::arange(0f32, (2 * 3 * 4 * 4 * 4) as f32, &device)
            .unwrap()
            .reshape((2, 3, 4, 4, 4))
            .unwrap();
        for patch in [1usize, 2, 4] {
            let folded = patchify(&x, patch).unwrap();
            assert_eq!(
                folded.dims(),
                [2, 3 * patch.pow(3), 4 / patch, 4 / patch, 4 / patch]
            );
            let restored = unpatchify(&folded, patch).unwrap();
            assert_eq!(restored.dims(), x.dims());
            assert_eq!(
                restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "patch {patch} did not round trip"
            );
        }
    }
}
