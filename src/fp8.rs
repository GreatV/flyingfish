//! Published E4M3 codes and the W8A16 effective-weight rounding contract.
//! Format: NVIDIA CUDA 13.0 Math API, __nv_fp8_e4m3 (bias 7; no infinity).
use half::bf16;
use safetensors::Dtype;

/// Row-major scales for a published `[N, K]` FP8 weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fp8BlockScale {
    block: [u32; 2],
    dtype: Dtype,
}

impl Fp8BlockScale {
    pub fn new(block: [u32; 2], dtype: Dtype) -> anyhow::Result<Self> {
        anyhow::ensure!(block.iter().all(|&v| v > 0), "zero FP8 scale block extent");
        anyhow::ensure!(
            matches!(dtype, Dtype::BF16 | Dtype::F32 | Dtype::F8_E8M0),
            "unsupported FP8 block scale dtype {dtype:?}"
        );
        Ok(Self { block, dtype })
    }

    pub fn block(self) -> [u32; 2] {
        self.block
    }

    pub fn dtype(self) -> Dtype {
        self.dtype
    }

    pub fn scale_shape(self, n: u32, k: u32) -> [usize; 2] {
        [
            n.div_ceil(self.block[0]) as usize,
            k.div_ceil(self.block[1]) as usize,
        ]
    }

    pub fn scale_bytes(self, n: u32, k: u32) -> anyhow::Result<usize> {
        use anyhow::Context;
        anyhow::ensure!(n > 0 && k > 0, "empty FP8 weight extent");
        self.scale_shape(n, k)
            .into_iter()
            .try_fold(self.dtype.bitsize() / 8, |bytes, extent| {
                bytes.checked_mul(extent)
            })
            .context("FP8 block scale byte extent overflow")
    }

    pub fn validate_scale(
        self,
        n: u32,
        k: u32,
        dtype: Dtype,
        shape: &[usize],
        bytes: usize,
    ) -> anyhow::Result<()> {
        let expected_bytes = self.scale_bytes(n, k)?;
        anyhow::ensure!(
            dtype == self.dtype && shape == self.scale_shape(n, k) && bytes == expected_bytes,
            "FP8 block scale mismatch: weight=[{n},{k}] block={:?} expected_dtype={:?} expected_shape={:?} expected_bytes={expected_bytes} got_dtype={dtype:?} got_shape={shape:?} got_bytes={bytes}",
            self.block,
            self.dtype,
            self.scale_shape(n, k),
        );
        Ok(())
    }
}

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
    pub fn block_scale(self) -> Option<Fp8BlockScale> {
        match self {
            Self::Block128 => Some(Fp8BlockScale {
                block: [128, 128],
                dtype: Dtype::BF16,
            }),
            Self::PerChannel => None,
        }
    }

    pub fn scale_dtype(self) -> Dtype {
        self.block_scale()
            .map_or(Dtype::BF16, |block| block.dtype())
    }

    /// Stored scale extent for one `[n, k]` weight.
    pub fn scale_shape(self, n: u32, k: u32) -> [usize; 2] {
        match self {
            Self::Block128 => self.block_scale().unwrap().scale_shape(n, k),
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
pub(crate) fn fake_quant_reference(x: &[bf16]) -> anyhow::Result<(bf16, Vec<u8>, Vec<bf16>)> {
    anyhow::ensure!(
        !x.is_empty() && x.iter().all(|v| v.is_finite()),
        "invalid E4M3 reference row"
    );
    let amax = x.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
    let scale = bf16::from_f32(amax / 448.0);
    let scale = if scale == bf16::ZERO {
        bf16::EPSILON
    } else {
        scale
    };
    let codes = x
        .iter()
        .map(|v| {
            let value = bf16::from_f32(v.to_f32() / scale.to_f32()).to_f32();
            let magnitude = value.abs().min(448.0);
            let mut low = 0u8;
            let mut high = 126u8;
            while low < high {
                let middle = low + (high - low) / 2;
                if decode_e4m3(middle) < magnitude {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            let upper = low;
            let lower = upper.saturating_sub(1);
            let dl = magnitude - decode_e4m3(lower);
            let du = decode_e4m3(upper) - magnitude;
            let code = if dl < du || (dl == du && lower & 1 == 0) {
                lower
            } else {
                upper
            };
            code | ((value.to_bits() >> 24) as u8 & 128)
        })
        .collect::<Vec<_>>();
    let fake = codes
        .iter()
        .map(|&code| effective_weight(code, scale))
        .collect();
    Ok((scale, codes, fake))
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) fn linear_reference(weight: &[bf16], x: &[bf16], k: usize) -> anyhow::Result<Vec<bf16>> {
    anyhow::ensure!(
        k > 0 && weight.len().is_multiple_of(k) && x.len().is_multiple_of(k),
        "invalid linear reference extents"
    );
    let mut output = Vec::with_capacity(x.len() / k * (weight.len() / k));
    for input in x.chunks_exact(k) {
        for row in weight.chunks_exact(k) {
            let mut lanes = [0.0f32; 32];
            for (lane, value) in lanes.iter_mut().enumerate() {
                let mut sums = [0.0f32; 4];
                if k.is_multiple_of(1024) {
                    for base in (0..k).step_by(1024) {
                        for word in 0..4 {
                            for byte in 0..4 {
                                let col = base + lane * 16 + word * 4 + byte;
                                sums[byte] =
                                    row[col].to_f32().mul_add(input[col].to_f32(), sums[byte]);
                                let col = col + 512;
                                let index = (byte + 2) & 3;
                                sums[index] =
                                    row[col].to_f32().mul_add(input[col].to_f32(), sums[index]);
                            }
                        }
                    }
                } else {
                    for col in (lane..k).step_by(32) {
                        sums[0] = row[col].to_f32().mul_add(input[col].to_f32(), sums[0]);
                    }
                }
                *value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            }
            for shift in [16, 8, 4, 2, 1] {
                let prior = lanes;
                for lane in 0..32 - shift {
                    lanes[lane] += prior[lane + shift];
                }
            }
            output.push(bf16::from_f32(lanes[0]));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn block_scale_native_extents_and_controls() -> anyhow::Result<()> {
        for (block, dtype, n, k, shape, bytes) in [
            ([128, 128], Dtype::F32, 2048, 4096, [16, 32], 2048),
            ([32, 32], Dtype::F8_E8M0, 2304, 5120, [72, 160], 11520),
            ([128, 128], Dtype::F32, 257, 385, [3, 4], 48),
            ([32, 32], Dtype::F8_E8M0, 65, 97, [3, 4], 12),
        ] {
            let descriptor = Fp8BlockScale::new(block, dtype)?;
            descriptor.validate_scale(n, k, dtype, &shape, bytes)?;
            assert!(
                descriptor
                    .validate_scale(n, k, dtype, &shape, bytes - 1)
                    .is_err()
            );
            assert!(
                descriptor
                    .validate_scale(n, k, dtype, &[shape[1], shape[0]], bytes)
                    .is_err()
            );
            assert!(
                descriptor
                    .validate_scale(n, k, Dtype::BF16, &shape, bytes)
                    .is_err()
            );
            let wrong_block = Fp8BlockScale::new([block[0] * 2, block[1]], dtype)?;
            assert!(
                wrong_block
                    .validate_scale(n, k, dtype, &shape, bytes)
                    .is_err()
            );
        }
        let legacy = Fp8ScaleLayout::Block128.block_scale().unwrap();
        legacy.validate_scale(257, 385, Dtype::BF16, &[3, 4], 24)?;
        assert!(Fp8BlockScale::new([0, 32], Dtype::F8_E8M0).is_err());
        assert!(Fp8BlockScale::new([32, 0], Dtype::F8_E8M0).is_err());
        assert!(Fp8BlockScale::new([32, 32], Dtype::F8_E4M3).is_err());
        assert!(legacy.scale_bytes(0, 128).is_err());
        assert!(
            Fp8BlockScale::new([1, 1], Dtype::F32)?
                .scale_bytes(u32::MAX, u32::MAX)
                .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only external fixture; set FF_MINICPM_REFERENCE_DIR"]
    fn token_quantization_matches_t13_bitwise() -> anyhow::Result<()> {
        let root = std::env::var_os("FF_MINICPM_REFERENCE_DIR")
            .context("FF_MINICPM_REFERENCE_DIR missing")?;
        let bytes =
            std::fs::read(std::path::Path::new(&root).join("fixtures/text_32_mixer.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        for (source, prefix) in [
            ("layer_03_mixer_in_norm", "layer_03_attn_in"),
            ("layer_03_o_proj_in", "layer_03_o_proj_in"),
        ] {
            let input = tensors.tensor(source)?;
            let scales = tensors.tensor(&format!("{prefix}_scale"))?;
            let codes = tensors.tensor(&format!("{prefix}_fp8_q"))?;
            let fake = tensors.tensor(&format!("{prefix}_fake_quant"))?;
            anyhow::ensure!(
                input.dtype() == safetensors::Dtype::BF16 && input.shape().len() == 2,
                "invalid {source}"
            );
            let k = input.shape()[1];
            anyhow::ensure!(
                k > 0
                    && input.shape()[0] == 32
                    && scales.dtype() == safetensors::Dtype::BF16
                    && scales.shape() == [32, 1]
                    && codes.dtype() == safetensors::Dtype::F8_E4M3
                    && codes.shape() == input.shape()
                    && fake.dtype() == safetensors::Dtype::BF16
                    && fake.shape() == input.shape(),
                "invalid {prefix} reference extents"
            );
            for (row, values) in input.data().chunks_exact(k * 2).enumerate() {
                let x = values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| bf16::from_le_bytes([v[0], v[1]]))
                    .collect::<Vec<_>>();
                let (s, c, f) = fake_quant_reference(&x)?;
                anyhow::ensure!(
                    s.to_le_bytes() == scales.data()[row * 2..row * 2 + 2],
                    "{prefix} scale row {row}"
                );
                anyhow::ensure!(
                    c == codes.data()[row * k..(row + 1) * k],
                    "{prefix} codes row {row}"
                );
                anyhow::ensure!(
                    f.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .eq(fake.data()[row * k * 2..(row + 1) * k * 2].iter().copied()),
                    "{prefix} fake row {row}"
                );
            }
            eprintln!(
                "PASS T13_quantization_oracle source={source} stored_input={:?} byte_mismatches=0",
                input.shape()
            );
        }
        let small = bf16::from_f32(bf16::EPSILON.to_f32() * 0.5);
        let (scale, _, _) = fake_quant_reference(&[small, -small])?;
        let wrong = bf16::from_f32(small.to_f32().max(bf16::EPSILON.to_f32()) / 448.0);
        anyhow::ensure!(
            scale.to_bits() != wrong.to_bits(),
            "max(amax,eps) control undetected"
        );
        let (zero_scale, zero_codes, _) = fake_quant_reference(&[bf16::ZERO, bf16::NEG_ZERO])?;
        anyhow::ensure!(
            zero_scale == bf16::EPSILON && zero_codes == [0, 128],
            "zero-scale boundary mismatch"
        );
        Ok(())
    }

    /// Effective weights equal the published layer-0 expert-0 BF16 reference bitwise.
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
