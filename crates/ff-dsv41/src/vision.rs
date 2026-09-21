//! DeepSeek-ViT tower and the aligner that maps it onto the backbone.
//!
//! Reference: `inference/vision.py`. Every projection here carries a bias;
//! the rotary is the half-split form (`x1`, `x2` are the channel halves),
//! not the adjacent-pair form the backbone uses.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

use crate::config::VisionConfig;
use crate::math::rms_norm;

fn linear_with_bias(
    weight: &Tensor,
    bias: &Tensor,
    input: &[f32],
    out: usize,
    inn: usize,
) -> Result<Vec<f32>> {
    ensure!(
        input.len() == inn,
        "activation of {} does not match the {inn}-wide projection",
        input.len()
    );
    let weights = weight
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read vision weight")?;
    let biases = bias
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read vision bias")?;
    let mut output = vec![0.0f32; out];
    for (row, slot) in output.iter_mut().enumerate() {
        let mut sum = biases.get(row).copied().unwrap_or(0.0);
        for column in 0..inn {
            sum += weights[row * inn + column] * input[column];
        }
        *slot = sum;
    }
    Ok(output)
}

/// Half-split 2D rotary: the grid interleaves height and width angle pairs
/// per patch row, and each head rotates the two halves of its channels.
fn vision_cos_sin(
    rows: usize,
    columns: usize,
    rope_dim: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let inv_freq = (0..rope_dim / 2)
        .map(|index| theta.powf(-(index as f64) / (rope_dim / 2) as f64))
        .collect::<Vec<_>>();
    let mut angles = Vec::with_capacity(rows * columns * rope_dim);
    for row in 0..rows {
        for column in 0..columns {
            for position in [row, column] {
                for freq in &inv_freq {
                    angles.push((position as f64 * freq) as f32);
                }
            }
        }
    }
    let count = angles.len();
    let cos: Vec<f32> = angles.iter().map(|angle| angle.cos()).collect();
    let sin: Vec<f32> = angles.iter().map(|angle| angle.sin()).collect();
    Ok((
        Tensor::from_vec(cos, (count,), device).map_err(anyhow::Error::from)?,
        Tensor::from_vec(sin, (count,), device).map_err(anyhow::Error::from)?,
    ))
}

fn apply_half_rotary(
    values: &mut [f32],
    token: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    // The per-token angle row spans rope_dim = cos.len() channels; each head
    // rotates that span of its own channels, split into halves.
    let span = cos.len();
    let half = span / 2;
    for head in 0..heads {
        let base = token * heads * head_dim + head * head_dim;
        for d in 0..half {
            let x1 = values[base + d];
            let x2 = values[base + half + d];
            let c = cos[d];
            let s = sin[d];
            values[base + d] = x1 * c - x2 * s;
            values[base + half + d] = x2 * c + x1 * s;
        }
    }
}

/// Bidirectional single-head attention over one image (softmax in F32).
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; tokens * heads * head_dim];
    for head in 0..heads {
        let mut scores = vec![0.0f32; tokens * tokens];
        let mut max = f32::NEG_INFINITY;
        for query in 0..tokens {
            for key in 0..tokens {
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q[(query * heads + head) * head_dim + d]
                        * k[(key * heads + head) * head_dim + d];
                }
                scores[query * tokens + key] = dot * scale;
                max = max.max(scores[query * tokens + key]);
            }
        }
        for query in 0..tokens {
            let mut total = 0.0;
            let mut weights = vec![0.0f32; tokens];
            for key in 0..tokens {
                weights[key] = (scores[query * tokens + key] - max).exp();
                total += weights[key];
            }
            for key in 0..tokens {
                for d in 0..head_dim {
                    output[(query * heads + head) * head_dim + d] +=
                        weights[key] / total * v[(key * heads + head) * head_dim + d];
                }
            }
        }
    }
    output
}

/// The whole tower: patch projection, rotary blocks, final norm. Weights are
/// materialized host tensors keyed by the checkpoint names.
pub struct VisionTower {
    pub config: VisionConfig,
    pub patch_weight: Tensor,
    pub patch_bias: Tensor,
    pub blocks: Vec<VisionBlock>,
    pub norm_weight: Tensor,
}

pub struct VisionBlock {
    pub norm1: Tensor,
    pub wqkv_weight: Tensor,
    pub wqkv_bias: Tensor,
    pub wo_weight: Tensor,
    pub wo_bias: Tensor,
    pub norm2: Tensor,
    pub w1_weight: Tensor,
    pub w1_bias: Tensor,
    pub w2_weight: Tensor,
    pub w2_bias: Tensor,
}

impl VisionTower {
    pub fn forward(&self, patches: &Tensor, rows: usize, columns: usize) -> Result<Tensor> {
        let dims = patches.dims();
        ensure!(
            dims.len() == 3 && dims[0] == rows * columns,
            "patches must be [tokens, 3, patch²]"
        );
        let device = patches.device();
        let patch_values = patches
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read flattened patches")?;
        let hidden = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let head_dim = hidden / heads;
        let patch_width = dims[2];
        let tokens = patch_values.len() / patch_width;
        let mut values = Vec::with_capacity(tokens * hidden);
        for token in 0..tokens {
            values.extend(linear_with_bias(
                &self.patch_weight,
                &self.patch_bias,
                &patch_values[token * patch_width..(token + 1) * patch_width],
                hidden,
                patch_width,
            )?);
        }
        let rope_dim = head_dim / 2;
        let (cos, sin) = vision_cos_sin(rows, columns, rope_dim, self.config.rope_theta, device)?;
        let cos_values = cos.to_vec1::<f32>()?;
        let sin_values = sin.to_vec1::<f32>()?;
        let tokens = rows * columns;
        for block in &self.blocks {
            let normed = rms_norm_flat(&values, &block.norm1, tokens, hidden)?;
            let mut qkv = Vec::with_capacity(tokens * 3 * hidden);
            for token in 0..tokens {
                qkv.extend(linear_with_bias(
                    &block.wqkv_weight,
                    &block.wqkv_bias,
                    &normed[token * hidden..(token + 1) * hidden],
                    3 * hidden,
                    hidden,
                )?);
            }
            let mut q = vec![0.0f32; tokens * hidden];
            let mut k = vec![0.0f32; tokens * hidden];
            let mut v = vec![0.0f32; tokens * hidden];
            for token in 0..tokens {
                for channel in 0..hidden {
                    q[token * hidden + channel] = qkv[token * 3 * hidden + channel];
                    k[token * hidden + channel] = qkv[token * 3 * hidden + hidden + channel];
                    v[token * hidden + channel] = qkv[token * 3 * hidden + 2 * hidden + channel];
                }
            }
            for token in 0..tokens {
                let cos_token = &cos_values[token * rope_dim..(token + 1) * rope_dim];
                let sin_token = &sin_values[token * rope_dim..(token + 1) * rope_dim];
                apply_half_rotary(&mut q, token, heads, head_dim, cos_token, sin_token);
                apply_half_rotary(&mut k, token, heads, head_dim, cos_token, sin_token);
            }
            let attended = attention(
                &q,
                &k,
                &v,
                tokens,
                heads,
                head_dim,
                (head_dim as f32).recip().sqrt(),
            );
            let mut projected = Vec::with_capacity(tokens * hidden);
            for token in 0..tokens {
                projected.extend(linear_with_bias(
                    &block.wo_weight,
                    &block.wo_bias,
                    &attended[token * hidden..(token + 1) * hidden],
                    hidden,
                    hidden,
                )?);
            }
            for (slot, delta) in values.iter_mut().zip(projected.iter()) {
                *slot += delta;
            }
            let normed = rms_norm_flat(&values, &block.norm2, tokens, hidden)?;
            let mut mlp = Vec::with_capacity(tokens * hidden);
            let inter = self.config.intermediate_size;
            for token in 0..tokens {
                let gate_up = linear_with_bias(
                    &block.w1_weight,
                    &block.w1_bias,
                    &normed[token * hidden..(token + 1) * hidden],
                    2 * inter,
                    hidden,
                )?;
                let mut inner = vec![0.0f32; inter];
                for row in 0..inter {
                    let gate = gate_up[row];
                    let up = gate_up[inter + row];
                    inner[row] = gate / (1.0 + (-gate).exp()) * up;
                }
                mlp.extend(linear_with_bias(
                    &block.w2_weight,
                    &block.w2_bias,
                    &inner,
                    hidden,
                    inter,
                )?);
            }
            for (slot, delta) in values.iter_mut().zip(mlp.iter()) {
                *slot += delta;
            }
        }
        let normalized = rms_norm_flat(&values, &self.norm_weight, tokens, hidden)?;
        Tensor::from_vec(normalized, (tokens, hidden), device).map_err(anyhow::Error::from)
    }
}

fn rms_norm_flat(
    values: &[f32],
    weight: &Tensor,
    tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    let tensor = Tensor::from_vec(values.to_vec(), (tokens, hidden), weight.device())
        .map_err(anyhow::Error::from)?;
    Ok(rms_norm(&tensor, weight, 1e-6)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

/// The aligner: 3x3 unfold downsample then a two-layer GELU MLP with biases.
pub struct Aligner {
    pub downsample_ratio: usize,
    pub w1_weight: Tensor,
    pub w1_bias: Tensor,
    pub w2_weight: Tensor,
    pub w2_bias: Tensor,
    pub hidden: usize,
}

impl Aligner {
    pub fn forward(&self, features: &Tensor, rows: usize, columns: usize) -> Result<Tensor> {
        let dims = features.dims();
        ensure!(
            dims.len() == 2 && dims[0] == rows * columns,
            "features must be [tokens, hidden]"
        );
        let ratio = self.downsample_ratio;
        let pad_rows = rows.div_ceil(ratio) * ratio - rows;
        let pad_columns = columns.div_ceil(ratio) * ratio - columns;
        let grid_rows = rows + pad_rows;
        let grid_columns = columns + pad_columns;
        let hidden = dims[1];
        let device = features.device();
        let values = features
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read aligner features")?;
        let out_rows = grid_rows / ratio;
        let out_columns = grid_columns / ratio;
        let cell = ratio * ratio * hidden;
        let mut unfolded = vec![0.0f32; out_rows * out_columns * cell];
        let padded = |row: usize, column: usize, channel: usize| -> f32 {
            if row < rows && column < columns {
                values[(row * columns + column) * hidden + channel]
            } else {
                0.0
            }
        };
        for block_row in 0..out_rows {
            for block_column in 0..out_columns {
                let mut slot = 0;
                for sub_row in 0..ratio {
                    for sub_column in 0..ratio {
                        for channel in 0..hidden {
                            unfolded[(block_row * out_columns + block_column) * cell + slot] =
                                padded(
                                    block_row * ratio + sub_row,
                                    block_column * ratio + sub_column,
                                    channel,
                                );
                            slot += 1;
                        }
                    }
                }
            }
        }
        let mut output = Vec::with_capacity(out_rows * out_columns * self.hidden);
        for token in 0..out_rows * out_columns {
            let hidden_state = linear_with_bias(
                &self.w1_weight,
                &self.w1_bias,
                &unfolded[token * cell..(token + 1) * cell],
                self.hidden,
                cell,
            )?;
            // GELU, tanh approximation (matching torch's approximate="tanh").
            let activated = hidden_state
                .iter()
                .map(|value| {
                    let inner = 0.797_884_6f32 * (value + 0.044715 * value * value * value);
                    0.5 * value * (1.0 + inner.tanh())
                })
                .collect::<Vec<_>>();
            output.extend(linear_with_bias(
                &self.w2_weight,
                &self.w2_bias,
                &activated,
                self.hidden,
                self.hidden,
            )?);
        }
        Tensor::from_vec(output, (out_rows * out_columns, self.hidden), device)
            .map_err(anyhow::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn half_rotary_interleaves_row_and_column_angles() {
        let device = Device::Cpu;
        let (cos, _sin) = vision_cos_sin(2, 3, 4, 100.0, &device).unwrap();
        assert_eq!(cos.dims(), [2 * 3 * 4]);
        let cos_values = cos.to_vec1::<f32>().unwrap();
        // Token (1, 2) carries its row angle first, then its column angle;
        // at rope_dim 4 each axis contributes two frequencies.
        let token = 5;
        assert!((cos_values[token * 4] - 1.0f64.cos() as f32).abs() < 1e-4);
        assert!((cos_values[token * 4 + 2] - 2.0f64.cos() as f32).abs() < 1e-4);
    }

    #[test]
    fn tower_projects_each_patch_and_rotates_the_rope_span() {
        let device = candle_core::Device::Cpu;
        let hidden = 8;
        let heads = 2;
        let config = crate::config::VisionConfig {
            num_hidden_layers: 1,
            hidden_size: hidden,
            num_attention_heads: heads,
            intermediate_size: 8,
            patch_size: 2,
            rope_theta: 100.0,
            downsample_ratio: 1,
            max_image_tokens: 64,
            min_pixels: 16,
            max_wh_ratio: None,
        };
        let weight = |rows: usize, columns: usize| {
            Tensor::from_vec(
                (0..rows * columns)
                    .map(|index| ((index % 5) as f32 - 2.0) / 32.0)
                    .collect(),
                (rows, columns),
                &device,
            )
            .unwrap()
        };
        let ones = |rows: usize| Tensor::from_vec(vec![1.0f32; rows], (rows,), &device).unwrap();
        let block = VisionBlock {
            norm1: ones(hidden),
            wqkv_weight: weight(3 * hidden, hidden),
            wqkv_bias: Tensor::zeros(3 * hidden, DType::F32, &device).unwrap(),
            wo_weight: weight(hidden, hidden),
            wo_bias: Tensor::zeros(hidden, DType::F32, &device).unwrap(),
            norm2: ones(hidden),
            w1_weight: weight(2 * 8, hidden),
            w1_bias: Tensor::zeros(2 * 8, DType::F32, &device).unwrap(),
            w2_weight: weight(hidden, 8),
            w2_bias: Tensor::zeros(hidden, DType::F32, &device).unwrap(),
        };
        let tower = VisionTower {
            config,
            patch_weight: weight(hidden, 3 * 2 * 2),
            patch_bias: Tensor::zeros(hidden, DType::F32, &device).unwrap(),
            blocks: vec![block],
            norm_weight: ones(hidden),
        };
        // A 2x3 patch grid: six patches, each [3, 2, 2].
        let patches = Tensor::from_vec(
            (0..6 * 3 * 4)
                .map(|index| ((index % 7) as f32 - 3.0) / 16.0)
                .collect(),
            (6, 3, 4),
            &device,
        )
        .unwrap();
        let output = tower.forward(&patches, 2, 3).unwrap();
        assert_eq!(output.dims(), [6, hidden]);
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        // Distinct patch inputs must reach distinct outputs; a projection that
        // collapsed all patches onto one row would emit identical rows.
        let rows: Vec<&[f32]> = values.chunks(hidden).collect();
        assert!(
            rows.windows(2).any(|pair| pair[0] != pair[1]),
            "all patch rows identical: {values:?}"
        );
    }

    #[test]
    fn aligner_gelu_follows_the_tanh_approximation() {
        // gelu(1) via the tanh form is 0.841192; the earlier wrong formula
        // (tanh(x/sqrt(2))) gives 0.813 SME — the gap pins the constants.
        let device = candle_core::Device::Cpu;
        let features = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();
        let identity = |rows: usize| {
            let mut values = vec![0.0f32; rows * rows];
            for index in 0..rows {
                values[index * rows + index] = 1.0;
            }
            Tensor::from_vec(values, (rows, rows), &device).unwrap()
        };
        let aligner = Aligner {
            downsample_ratio: 1,
            w1_weight: identity(2),
            w1_bias: Tensor::zeros(2, DType::F32, &device).unwrap(),
            w2_weight: identity(2),
            w2_bias: Tensor::zeros(2, DType::F32, &device).unwrap(),
            hidden: 2,
        };
        let output = aligner.forward(&features, 1, 1).unwrap();
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let inner = 0.797_884_6f32 * (1.0 + 0.044715 * 1.0);
        let expected = 0.5 * (1.0 + inner.tanh());
        assert!(
            (values[0] - expected).abs() < 1e-5,
            "{} vs {expected}",
            values[0]
        );
        assert_eq!(values[1], 0.0);
    }

    #[test]
    fn aligner_unfolds_three_by_three_cells_with_padding() {
        let device = Device::Cpu;
        let hidden = 2;
        // A 3x2 grid padded to 3x3 unfolds into one 3x3 cell.
        let features = Tensor::from_vec(
            (0..3 * 2 * hidden).map(|index| index as f32).collect(),
            (6, hidden),
            &device,
        )
        .unwrap();
        let aligner = Aligner {
            downsample_ratio: 3,
            w1_weight: Tensor::from_vec(
                (0..6 * 18)
                    .map(|index| (index % 3) as f32 / 100.0)
                    .collect(),
                (6, 18),
                &device,
            )
            .unwrap(),
            w1_bias: Tensor::zeros(6, DType::F32, &device).unwrap(),
            w2_weight: Tensor::from_vec(
                (0..6 * 6).map(|index| (index % 2) as f32 / 50.0).collect(),
                (6, 6),
                &device,
            )
            .unwrap(),
            w2_bias: Tensor::zeros(6, DType::F32, &device).unwrap(),
            hidden,
        };
        let output = aligner.forward(&features, 3, 2).unwrap();
        assert_eq!(output.dims(), [1, 2]);
    }
}
