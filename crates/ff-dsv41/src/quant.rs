//! FP8 and FP4 dequantization for the V4.1 checkpoint layout.
//!
//! Body weights are E4M3 with one E8M0 scale per 32-by-32 block (ceil-divided
//! on both axes); routed experts are two E2M1 values per byte with one E8M0
//! scale per 32 elements along the reduction axis. Scales arrive as raw
//! E8M0 bytes because the expert payload's I8 dtype has no candle
//! representation. Verified against the safetensors headers in
//! `models/deepseek-ai/DeepSeek-V4.1-Flash`.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};

use crate::config::{FP4_WEIGHT_BLOCK, FP8_WEIGHT_BLOCK};

/// One E8M0 byte is the power-of-two scale `2^(byte - 127)`.
pub fn e8m0_to_f32(bits: u8) -> f32 {
    2.0f32.powi(bits as i32 - 127)
}

/// Decode one E4M3 byte to F32 (sign bit, 4 exponent bits with bias 7, 3
/// mantissa bits; 0x7F is NaN in the fnu variant, which never appears as a
/// stored weight).
pub fn e4m3_byte_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((bits >> 3) & 0x0f) as i32;
    let mantissa = (bits & 0x07) as i32;
    if exponent == 0 {
        sign * (mantissa as f32) * 2.0f32.powi(-9)
    } else {
        sign * ((1 << 3 | mantissa) as f32) * 2.0f32.powi(exponent - 10)
    }
}

/// E8M0 scale bytes as an F32 tensor of the given shape.
pub fn e8m0_scales_to_f32(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    ensure!(
        bytes.len() == shape.iter().product::<usize>(),
        "{} E8M0 bytes do not fill the shape {:?}",
        bytes.len(),
        shape
    );
    let values = bytes
        .iter()
        .map(|bits| e8m0_to_f32(*bits))
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape.to_vec(), device).map_err(anyhow::Error::from)
}

fn expand_block_scales(
    scales: &Tensor,
    rows: usize,
    columns: usize,
    block: usize,
) -> Result<Tensor> {
    let (scale_rows, scale_columns) = scales.dims2().context("block scales must be rank 2")?;
    ensure!(
        scale_rows == rows.div_ceil(block) && scale_columns == columns.div_ceil(block),
        "block scale shape [{scale_rows}, {scale_columns}] does not cover \
         [{rows}, {columns}] in {block}-by-{block} blocks"
    );
    Ok(scales
        .reshape((scale_rows, 1, scale_columns, 1))?
        .broadcast_as((scale_rows, block, scale_columns, block))?
        .reshape((scale_rows * block, scale_columns * block))?
        .narrow(0, 0, rows)?
        .narrow(1, 0, columns)?)
}

/// `output[row, column] = weight[row, column] * scale[row/32, column/32]`.
///
/// `scale_bytes` holds the E8M0 scales for the ceil-divided block grid of
/// the weight's `[rows, columns]` shape, row-major.
pub fn dequantize_fp8_block(
    weight: &Tensor,
    scale_bytes: &[u8],
    device: &Device,
) -> Result<Tensor> {
    ensure!(
        weight.dtype() == DType::F8E4M3,
        "FP8 body weight must have dtype F8E4M3, found {:?}",
        weight.dtype()
    );
    let (rows, columns) = weight
        .dims2()
        .context("FP8 body weight must be a rank-2 matrix")?;
    let scale_shape = [
        rows.div_ceil(FP8_WEIGHT_BLOCK),
        columns.div_ceil(FP8_WEIGHT_BLOCK),
    ];
    let scales = e8m0_scales_to_f32(scale_bytes, &scale_shape, device)?;
    let expanded = expand_block_scales(&scales, rows, columns, FP8_WEIGHT_BLOCK)?;
    Ok((weight.to_dtype(DType::F32)? * expanded)?)
}

/// The E2M1 decode table; the low nibble of each byte is the earlier element.
const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Experts store `[out, in/2]` packed bytes with one E8M0 scale per
/// `FP4_WEIGHT_BLOCK` elements along the input axis.
///
/// `output[row, column] = nibble(row, column) * scale[row, column/32]`.
pub fn dequantize_fp4_packed(
    payload: &[u8],
    rows: usize,
    columns: usize,
    scale_bytes: &[u8],
    device: &Device,
) -> Result<Tensor> {
    ensure!(
        payload.len() == rows * columns.div_ceil(2),
        "{} payload bytes do not pack the [{rows}, {columns}] expert",
        payload.len()
    );
    let scale_columns = columns.div_ceil(FP4_WEIGHT_BLOCK);
    ensure!(
        scale_bytes.len() == rows * scale_columns,
        "{} scale bytes do not cover [{rows}, {columns}] elements \
         in {FP4_WEIGHT_BLOCK}-element groups",
        scale_bytes.len()
    );
    let mut nibbles = Vec::with_capacity(payload.len() * 2);
    for byte in payload {
        nibbles.push(FP4_VALUES[(byte & 0x0f) as usize]);
        nibbles.push(FP4_VALUES[(byte >> 4) as usize]);
    }
    let values = Tensor::from_vec(nibbles, (rows, columns), device).map_err(anyhow::Error::from)?;
    let scales = e8m0_scales_to_f32(scale_bytes, &[rows, scale_columns], device)?;
    let expanded = scales
        .reshape((rows, scale_columns, 1))?
        .broadcast_as((rows, scale_columns, FP4_WEIGHT_BLOCK))?
        .reshape((rows, scale_columns * FP4_WEIGHT_BLOCK))?
        .narrow(1, 0, columns)?;
    Ok((values * expanded)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors::Load as _;
    use ff_core::paths::checkpoint_dir;

    #[test]
    fn e8m0_decodes_powers_of_two() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(130), 8.0);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
    }

    #[test]
    fn fp4_low_nibble_is_the_earlier_element() {
        // byte 0x10 -> low nibble 0 (0.0), high nibble 1 (0.5)
        // byte 0x98 -> low nibble 8 (negative zero entry), high nibble 9 (-0.5)
        let decoded = dequantize_fp4_packed(&[0x10, 0x98], 1, 4, &[127], &Device::Cpu).unwrap();
        let values = decoded.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, [0.0, 0.5, 0.0, -0.5]);
    }

    #[test]
    fn fp4_scales_multiply_per_32_elements() {
        let decoded = dequantize_fp4_packed(&[0x11; 64], 1, 128, &[128; 4], &Device::Cpu).unwrap();
        assert_eq!(decoded.dims(), [1, 128]);
        let values = decoded.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn fp8_block_scales_cover_ceil_divided_axes() {
        let weight = Tensor::zeros((33usize, 65usize), DType::F8E4M3, &Device::Cpu).unwrap();
        let decoded = dequantize_fp8_block(&weight, &[127; 6], &Device::Cpu).unwrap();
        assert_eq!(decoded.dims(), [33, 65]);
        assert!(dequantize_fp8_block(&weight, &[127; 4], &Device::Cpu).is_err());
    }

    #[test]
    fn e4m3_subnormals_decode_on_the_m8_scale() {
        // Exponent 0 bytes are subnormals: m/8 * 2^-6 = m * 2^-9.
        assert_eq!(e4m3_byte_to_f32(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3_byte_to_f32(0x07), 7.0 * 2.0f32.powi(-9));
        assert_eq!(e4m3_byte_to_f32(0x81), -2.0f32.powi(-9));
        // Normal range stays on the 2^(exp-10) scale; 0x08 is the smallest
        // normal (m=0, exp=1), not a subnormal.
        assert_eq!(e4m3_byte_to_f32(0x08), 2.0f32.powi(-6));
        assert_eq!(e4m3_byte_to_f32(0x38), 1.0);
        assert_eq!(e4m3_byte_to_f32(0x40), 2.0);
    }

    #[test]
    fn dequantization_round_trips_a_real_shard_payload() {
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        let path = dir.join("model-00009-of-00048.safetensors");
        if !path.exists() {
            return;
        }
        let file = std::fs::File::open(&path).unwrap();
        let mapped = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let tensors = safetensors::SafeTensors::deserialize(&mapped).unwrap();

        let weight = tensors
            .tensor("layers.6.attn.wq_a.weight")
            .unwrap()
            .load(&Device::Cpu)
            .unwrap();
        let scale_bytes = tensors.tensor("layers.6.attn.wq_a.scale").unwrap().data();
        assert_eq!(weight.dtype(), DType::F8E4M3);
        assert_eq!(weight.dims(), [1280, 5120]);
        assert_eq!(scale_bytes.len(), 40 * 160);
        let decoded = dequantize_fp8_block(&weight, scale_bytes, &Device::Cpu).unwrap();
        assert_eq!(decoded.dims(), [1280, 5120]);

        let expert = tensors.tensor("layers.6.ffn.experts.3.w1.weight").unwrap();
        assert_eq!(expert.dtype(), safetensors::Dtype::I8);
        assert_eq!(expert.shape(), [2304, 2560]);
        let expert_scale = tensors.tensor("layers.6.ffn.experts.3.w1.scale").unwrap();
        assert_eq!(expert_scale.shape(), [2304, 160]);
        let decoded =
            dequantize_fp4_packed(expert.data(), 2304, 5120, expert_scale.data(), &Device::Cpu)
                .unwrap();
        assert_eq!(decoded.dims(), [2304, 5120]);
    }
}
