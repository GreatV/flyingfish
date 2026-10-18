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

/// Scale layout of a published FP8 linear weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8ScaleLayout {
    /// BF16 `[ceil(N/128), ceil(K/128)]`: one scale per 128x128 block.
    Block128,
    /// BF16 `[N, 1]`: one scale per output channel.
    PerChannel,
}

impl Fp8ScaleLayout {
    /// Stored scale extent for one `[n, k]` weight.
    pub fn scale_shape(self, n: u32, k: u32) -> [usize; 2] {
        match self {
            Self::Block128 => [n.div_ceil(128) as usize, k.div_ceil(128) as usize],
            Self::PerChannel => [n as usize, 1],
        }
    }

    /// Checkpoint tensor suffix of the scale beside `...weight`.
    pub fn scale_suffix(self) -> &'static str {
        match self {
            Self::Block128 => "weight_scale_inv",
            Self::PerChannel => "weight_scale",
        }
    }
}

impl crate::model_spec::LinearFormat {
    pub fn fp8_layout(self) -> Option<Fp8ScaleLayout> {
        match self {
            Self::Bf16 => None,
            Self::Fp8E4m3Block128 => Some(Fp8ScaleLayout::Block128),
            Self::Fp8E4m3Channel => Some(Fp8ScaleLayout::PerChannel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    /// The anchor: effective_weight over the layer-0 expert-0 native FP8
    /// bytes equals agy's torch-dequantized BF16 dump bitwise.
    #[ignore = "CPU-only external fixture; set FF_MINICPM_REFERENCE_DIR"]
    #[test]
    fn effective_weight_matches_real_expert_bitwise() -> anyhow::Result<()> {
        let dir = std::env::var_os("FF_MINICPM_REFERENCE_DIR")
            .context("FF_MINICPM_REFERENCE_DIR missing")?;
        let f = std::fs::read(std::path::Path::new(&dir).join("expert_l0e0_dequant.safetensors"))
            .context("read expert_l0e0_dequant.safetensors")?;
        let tensors = safetensors::SafeTensors::deserialize(&f)?;
        let pairs = [
            ("gate", "gate_proj.weight_fp8", "gate_proj.weight_scale"),
            ("up", "up_proj.weight_fp8", "up_proj.weight_scale"),
            ("down", "down_proj.weight_fp8", "down_proj.weight_scale"),
        ];
        for (want_name, code_name, scale_name) in pairs {
            let want = tensors.tensor(want_name)?;
            let codes = tensors.tensor(code_name)?;
            let scales = tensors.tensor(scale_name)?;
            anyhow::ensure!(
                want.dtype() == safetensors::Dtype::BF16,
                "{want_name}: expected BF16, got {:?}",
                want.dtype()
            );
            anyhow::ensure!(
                codes.dtype() == safetensors::Dtype::F8_E4M3,
                "{code_name}: expected F8_E4M3, got {:?}",
                codes.dtype()
            );
            anyhow::ensure!(
                scales.dtype() == safetensors::Dtype::BF16,
                "{scale_name}: expected BF16, got {:?}",
                scales.dtype()
            );
            let want_b = want.data();
            let code_b = codes.data();
            let scale_b = scales.data();
            anyhow::ensure!(
                code_b.len() * 2 == want_b.len(),
                "{want_name}: extents want {} code {}",
                want_b.len(),
                code_b.len()
            );
            // Per-output-channel scales: one BF16 scale per weight row.
            let k = code_b.len() / (scale_b.len() / 2);
            anyhow::ensure!(
                scale_b.len() / 2 * k == code_b.len(),
                "{want_name}: scale rows do not divide the codes"
            );
            let mut bad = None;
            for i in 0..code_b.len() {
                let row = i / k;
                let s = bf16::from_le_bytes([scale_b[row * 2], scale_b[row * 2 + 1]]);
                let got = effective_weight(code_b[i], s);
                let exp = bf16::from_le_bytes([want_b[i * 2], want_b[i * 2 + 1]]);
                if got.to_bits() != exp.to_bits() {
                    bad = Some((i, got, exp));
                    break;
                }
            }
            anyhow::ensure!(
                bad.is_none(),
                "{want_name}: effective_weight mismatch at {bad:?}"
            );
            eprintln!("{want_name}: {} weights bitwise-equal", code_b.len());
        }
        Ok(())
    }
}
