//! Published E4M3 codes and the W8A16 effective-weight rounding contract.
//! Format: NVIDIA CUDA 13.0 Math API, __nv_fp8_e4m3 (bias 7; no infinity).
use half::bf16;

pub fn decode_e4m3(code: u8) -> f32 {
    let sign = u32::from(code & 0x80) << 24;
    let magnitude = code & 0x7f;
    if magnitude == 0x7f {
        return f32::from_bits(sign | 0x7fc0_0000);
    }
    let exponent = magnitude >> 3;
    let mantissa = magnitude & 7;
    if exponent == 0 {
        let value = f32::from(mantissa) * (1.0 / 512.0);
        return f32::from_bits(value.to_bits() | sign);
    }
    f32::from_bits(sign | ((u32::from(exponent) + 120) << 23) | (u32::from(mantissa) << 20))
}

pub fn effective_weight(code: u8, scale: bf16) -> bf16 {
    bf16::from_f32(decode_e4m3(code) * scale.to_f32())
}
