//! Exact learned-position interpolation for the released Qwen3-VL vision grids.
//!
//! Official Transformers constructs bilinear taps with CUDA F32 arithmetic.
//! Computing the same formula through host F64 and casting at the end changes
//! thousands of weight bits at the real FL/Ref sizes. This module preserves
//! every F32 rounding boundary and rejects grids outside the measured profile.

use candle_core::{DType, Tensor};

const SOURCE_SIDE: usize = 48;
const SPATIAL_MERGE: usize = 2;
const HIDDEN_SIZE: usize = 1_152;

pub(crate) const BACKEND: &str = "transformers838763bf-qwen3vl-host-stepwise-f32-bitexact-to-cuda-f32-bilinear-align-corners-taps/candle011-u32-gather-f32-weighted-sum4-bf16-cast/real-grid-profile:image1x48x84|video7x48x84-v1";

fn validate_grids(grids: &[[usize; 3]]) -> candle_core::Result<usize> {
    if grids != [[1, 48, 84]] && grids != [[7, 48, 84]] {
        candle_core::bail!(
            "unverified Qwen learned-position grids {grids:?}; supported ordered grids are [[1,48,84]] and [[7,48,84]]"
        )
    }
    grids
        .iter()
        .try_fold(0usize, |total, [temporal, height, width]| {
            temporal
                .checked_mul(*height)
                .and_then(|rows| rows.checked_mul(*width))
                .and_then(|rows| total.checked_add(rows))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Qwen learned-position row count overflow".into())
                })
        })
}

fn axis_taps(index: usize, target_size: usize) -> (usize, usize, f32, f32) {
    let scaled = index as f32 * (SOURCE_SIDE - 1) as f32;
    let source = scaled / (target_size - 1).max(1) as f32;
    let lower = source.floor() as usize;
    let upper = (lower + 1).min(SOURCE_SIDE - 1);
    let upper_weight = source - lower as f32;
    let lower_weight = 1.0_f32 - upper_weight;
    (lower, upper, lower_weight, upper_weight)
}

fn interpolation_indices_weights(
    grids: &[[usize; 3]],
) -> candle_core::Result<(Vec<u32>, Vec<f32>)> {
    let rows = validate_grids(grids)?;
    let capacity = rows
        .checked_mul(4)
        .ok_or_else(|| candle_core::Error::Msg("Qwen interpolation capacity overflow".into()))?;
    let mut indices = Vec::with_capacity(capacity);
    let mut weights = Vec::with_capacity(capacity);
    for &[temporal, height, width] in grids {
        if !height.is_multiple_of(SPATIAL_MERGE) || !width.is_multiple_of(SPATIAL_MERGE) {
            candle_core::bail!("Qwen learned-position grid is not 2x2 merge-aligned")
        }
        for _ in 0..temporal {
            for block_height in 0..height / SPATIAL_MERGE {
                for block_width in 0..width / SPATIAL_MERGE {
                    for inner_height in 0..SPATIAL_MERGE {
                        for inner_width in 0..SPATIAL_MERGE {
                            let row = block_height * SPATIAL_MERGE + inner_height;
                            let column = block_width * SPATIAL_MERGE + inner_width;
                            let (h0, h1, hw0, hw1) = axis_taps(row, height);
                            let (w0, w1, ww0, ww1) = axis_taps(column, width);
                            for (source_h, height_weight, source_w, width_weight) in [
                                (h0, hw0, w0, ww0),
                                (h0, hw0, w1, ww1),
                                (h1, hw1, w0, ww0),
                                (h1, hw1, w1, ww1),
                            ] {
                                indices.push(
                                    u32::try_from(source_h * SOURCE_SIDE + source_w).map_err(
                                        |_| {
                                            candle_core::Error::Msg(
                                                "Qwen learned-position index exceeds u32".into(),
                                            )
                                        },
                                    )?,
                                );
                                weights.push(height_weight * width_weight);
                            }
                        }
                    }
                }
            }
        }
    }
    if indices.len() != capacity || weights.len() != capacity {
        candle_core::bail!("Qwen learned-position tap count is inconsistent")
    }
    Ok((indices, weights))
}

struct PositionBoundaries {
    position_bf16: Tensor,
}

fn position_boundaries(
    table: &Tensor,
    grids: &[[usize; 3]],
) -> candle_core::Result<PositionBoundaries> {
    let rows = validate_grids(grids)?;
    let candle_core::Device::Cuda(cuda) = table.device() else {
        candle_core::bail!("exact Qwen learned-position interpolation is CUDA-only")
    };
    crate::cuda::device::require_tuned_kernel(cuda)?;
    if table.dtype() != DType::BF16
        || table.dims() != [SOURCE_SIDE * SOURCE_SIDE, HIDDEN_SIZE]
        || !table.is_contiguous()
        || table.layout().start_offset() != 0
    {
        candle_core::bail!(
            "exact Qwen learned-position table must be zero-offset contiguous BF16 [2304,1152]"
        )
    }
    let (indices, weights) = interpolation_indices_weights(grids)?;
    let indices = Tensor::from_vec(indices, (rows, 4), table.device())?;
    let weights = Tensor::from_vec(weights, (rows, 4), table.device())?;
    let gathered = table
        .index_select(&indices.flatten_all()?, 0)?
        .to_dtype(DType::F32)?
        .reshape((rows, 4, HIDDEN_SIZE))?;
    let position_f32 = gathered
        .broadcast_mul(&weights.reshape((rows, 4, 1))?)?
        .sum(1)?;
    let position_bf16 = position_f32.to_dtype(DType::BF16)?;
    Ok(PositionBoundaries { position_bf16 })
}

pub(crate) fn position_embedding(
    table: &Tensor,
    grids: &[[usize; 3]],
) -> candle_core::Result<Tensor> {
    Ok(position_boundaries(table, grids)?.position_bf16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rejects_unverified_grid_before_tensor_execution() {
        let table = Tensor::zeros((2_304, 1_152), DType::BF16, &Device::Cpu).unwrap();
        let error = position_embedding(&table, &[[1, 42, 96]])
            .unwrap_err()
            .to_string();
        assert!(error.contains("unverified"), "unexpected error: {error}");
    }

    #[test]
    fn f32_axis_rounding_has_the_pinned_bits() {
        let (_, _, lower, upper) = axis_taps(1, 84);
        assert_eq!(lower.to_bits(), 0x3e_de_12_82);
        assert_eq!(upper.to_bits(), 0x3f_10_f6_bf);
        assert_eq!((1.0_f32 - lower).to_bits(), upper.to_bits());
    }
}
