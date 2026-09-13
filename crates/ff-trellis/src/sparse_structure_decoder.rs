//! The dense 3D convolutional decoder from a structure latent to a voxel grid.
//!
//! Transcribed from `microsoft/TRELLIS`,
//! `trellis/models/sparse_structure_vae.py`. The published decoder uses
//! `norm_type="layer"`, so its normalization is `ChannelLayerNorm32`: permute
//! the channels last, normalize over them in F32, permute back. Upsampling is a
//! convolution to eight times the output channels followed by a 3D pixel
//! shuffle, so the block list interleaves residual blocks with those.
//!
//! The oracle this is checked against is exported with cuDNN's TF32
//! convolutions disabled. PyTorch enables them by default on Ampere and later,
//! which costs about three decimal digits: on this decoder's input convolution
//! alone, TF32 and F32 differ by 3.5e-4 relative, which is larger than the
//! error the whole eight-block decoder accumulates in F32. An oracle exported
//! with the default left in place measures cuDNN's precision policy rather than
//! this transcription.

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use ff_core::weights::ModelWeights;

use crate::config::SparseStructureCoderArgs;
use crate::conv3d::{conv3d, pixel_shuffle_3d};

/// `nn.LayerNorm`'s default eps, which `ChannelLayerNorm32` inherits.
const LAYER_NORM_EPS: f64 = 1e-5;

struct Conv3d {
    weight: Tensor,
    bias: Tensor,
    padding: usize,
}

impl Conv3d {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        padding: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), device, dtype)?,
            bias: get(weights, &format!("{prefix}.bias"), device, dtype)?,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        conv3d(x, &self.weight, Some(&self.bias), self.padding)
    }
}

/// `ChannelLayerNorm32`: normalize across channels, computed in F32.
struct ChannelLayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl ChannelLayerNorm {
    fn load(weights: &ModelWeights, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{prefix}.weight"), device, DType::F32)?,
            bias: get(weights, &format!("{prefix}.bias"), device, DType::F32)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let moved = x
            .permute([0, 2, 3, 4, 1])?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let mean = moved.mean_keepdim(D::Minus1)?;
        let centered = moved.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered
            .broadcast_div(&(variance + LAYER_NORM_EPS)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?
            .permute([0, 4, 1, 2, 3])?
            .contiguous()?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

/// `ResBlock3d`. The skip is a 1×1×1 convolution only when the width changes,
/// which in this decoder it never does.
struct ResBlock {
    norm1: ChannelLayerNorm,
    conv1: Conv3d,
    norm2: ChannelLayerNorm,
    conv2: Conv3d,
    skip: Option<Conv3d>,
}

impl ResBlock {
    fn load(weights: &ModelWeights, prefix: &str, device: &Device, dtype: DType) -> Result<Self> {
        let skip = weights
            .contains(&format!("{prefix}.skip_connection.weight"))
            .then(|| {
                Conv3d::load(
                    weights,
                    &format!("{prefix}.skip_connection"),
                    0,
                    device,
                    dtype,
                )
            })
            .transpose()?;
        Ok(Self {
            norm1: ChannelLayerNorm::load(weights, &format!("{prefix}.norm1"), device)?,
            conv1: Conv3d::load(weights, &format!("{prefix}.conv1"), 1, device, dtype)?,
            norm2: ChannelLayerNorm::load(weights, &format!("{prefix}.norm2"), device)?,
            conv2: Conv3d::load(weights, &format!("{prefix}.conv2"), 1, device, dtype)?,
            skip,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let residual = match &self.skip {
            None => x.clone(),
            Some(skip) => skip.forward(x)?,
        };
        (h + residual).map_err(Into::into)
    }
}

/// One entry of the decoder's flat block list.
enum Block {
    Residual(ResBlock),
    /// A convolution to eight times the width, then a 3D pixel shuffle.
    Upsample(Conv3d),
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Residual(block) => block.forward(x),
            Self::Upsample(conv) => pixel_shuffle_3d(&conv.forward(x)?, 2),
        }
    }
}

/// The loaded decoder.
pub struct SparseStructureDecoder {
    dtype: DType,
    input_layer: Conv3d,
    middle: Vec<ResBlock>,
    blocks: Vec<Block>,
    out_norm: ChannelLayerNorm,
    out_conv: Conv3d,
}

impl SparseStructureDecoder {
    pub fn load(
        args: &SparseStructureCoderArgs,
        weights: &ModelWeights,
        device: &Device,
    ) -> Result<Self> {
        Self::load_with_dtype(args, weights, device, None)
    }

    /// As [`Self::load`], with the torso precision overridden for parity work.
    pub fn load_with_dtype(
        args: &SparseStructureCoderArgs,
        weights: &ModelWeights,
        device: &Device,
        torso_dtype: Option<DType>,
    ) -> Result<Self> {
        anyhow::ensure!(
            !args.channels.is_empty(),
            "a sparse-structure decoder needs at least one channel width"
        );
        let dtype = torso_dtype.unwrap_or(if args.use_fp16 && !device.is_cpu() {
            DType::F16
        } else {
            DType::F32
        });

        let mut middle = Vec::with_capacity(args.num_res_blocks_middle);
        for index in 0..args.num_res_blocks_middle {
            middle.push(ResBlock::load(
                weights,
                &format!("middle_block.{index}"),
                device,
                dtype,
            )?);
        }

        let mut blocks = Vec::new();
        let mut ordinal = 0usize;
        for (index, _) in args.channels.iter().enumerate() {
            for _ in 0..args.num_res_blocks {
                blocks.push(Block::Residual(ResBlock::load(
                    weights,
                    &format!("blocks.{ordinal}"),
                    device,
                    dtype,
                )?));
                ordinal += 1;
            }
            if index + 1 < args.channels.len() {
                blocks.push(Block::Upsample(Conv3d::load(
                    weights,
                    &format!("blocks.{ordinal}.conv"),
                    1,
                    device,
                    dtype,
                )?));
                ordinal += 1;
            }
        }

        Ok(Self {
            dtype,
            input_layer: Conv3d::load(weights, "input_layer", 1, device, DType::F32)?,
            middle,
            blocks,
            out_norm: ChannelLayerNorm::load(weights, "out_layer.0", device)?,
            out_conv: Conv3d::load(weights, "out_layer.2", 1, device, DType::F32)?,
        })
    }

    /// Decode `[batch, latent_channels, side, side, side]` to occupancy logits.
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let input_dtype = latent.dtype();
        let mut h = self.input_layer.forward(&latent.to_dtype(DType::F32)?)?;
        h = h.to_dtype(self.dtype)?;
        for block in &self.middle {
            h = block.forward(&h)?;
        }
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        h = h.to_dtype(input_dtype)?;
        let h = self.out_norm.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;
        self.out_conv.forward(&h)
    }
}

fn get(weights: &ModelWeights, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
    weights
        .load(name, device)
        .with_context(|| format!("failed to load {name}"))?
        .to_dtype(dtype)
        .map_err(Into::into)
}
