use anyhow::{Result, ensure};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

#[derive(Default)]
pub struct Tensors {
    values: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
}

impl Tensors {
    fn add(&mut self, name: String, dtype: Dtype, shape: Vec<usize>, bytes: Vec<u8>) -> Result<()> {
        let n = shape.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(b)
                .ok_or_else(|| anyhow::anyhow!("tensor shape overflow"))
        })?;
        ensure!(
            n.checked_mul(dtype.bitsize()).map(|bits| bits.div_ceil(8)) == Some(bytes.len()),
            "typed tensor shape mismatch {name}"
        );
        ensure!(
            !self.values.contains_key(&name),
            "duplicate typed tensor {name}"
        );
        self.values.insert(name, (dtype, shape, bytes));
        Ok(())
    }
    pub fn f32(&mut self, name: String, shape: Vec<usize>, values: &[f32]) -> Result<()> {
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite tensor {name}"
        );
        self.add(
            name,
            Dtype::F32,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    pub fn i32(&mut self, name: String, shape: Vec<usize>, values: &[i32]) -> Result<()> {
        self.add(
            name,
            Dtype::I32,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    pub fn i64(&mut self, name: String, shape: Vec<usize>, values: &[i64]) -> Result<()> {
        self.add(
            name,
            Dtype::I64,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    #[cfg(test)]
    pub fn u8(&mut self, name: String, shape: Vec<usize>, values: &[u8]) -> Result<()> {
        self.add(name, Dtype::U8, shape, values.to_vec())
    }
    #[cfg(test)]
    pub fn bf16(&mut self, name: String, shape: Vec<usize>, values: &[half::bf16]) -> Result<()> {
        self.add(
            name,
            Dtype::BF16,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    pub fn u64(&mut self, name: String, shape: Vec<usize>, values: &[u64]) -> Result<()> {
        self.add(
            name,
            Dtype::U64,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    pub fn save_new(&self, path: &Path, metadata: HashMap<String, String>) -> Result<()> {
        let parent = path.parent().unwrap_or(Path::new(".")).canonicalize()?;
        ensure!(
            !parent
                .components()
                .any(|c| matches!(c,std::path::Component::Normal(p) if p=="models")),
            "typed dump cannot be inside models"
        );
        let path = parent.join(
            path.file_name()
                .ok_or_else(|| anyhow::anyhow!("dump filename missing"))?,
        );
        let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let result = (|| -> Result<()> {
            let views = self
                .values
                .iter()
                .map(|(name, (dtype, shape, data))| {
                    Ok((name.as_str(), TensorView::new(*dtype, shape.clone(), data)?))
                })
                .collect::<Result<Vec<_>>>()?;
            serialize_to_file(views, Some(metadata), &tmp)?;
            std::fs::hard_link(&tmp, &path)?;
            Ok(())
        })();
        std::fs::remove_file(tmp)?;
        result
    }
}

/// Published BF16 weight bits become FP16 bits. The exponent is `(bits >> 7) & 0xFF` and the mantissa is `bits & 0x7F`.
/// Exponent 143 through 255 is a load error naming `name`. A nonzero value with exponent 102 or below stores +0.
/// Exponent 103 with mantissa 0 is `2^-24` and stays. The zero encoding keeps its sign. `report` prints a flush count above zero.
pub fn bf16_weight_to_fp16(name: &str, bytes: &mut [u8], report: bool) -> Result<u64> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "{name}: BF16 weight is {} bytes",
        bytes.len()
    );
    for pair in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        if fp16_from_bf16_bits(bits).is_err() {
            anyhow::bail!("{name}: BF16 weight is non-finite or above 65504");
        }
    }
    let mut flushed = 0u64;
    for pair in bytes.as_chunks_mut::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        let (out, flush) = match fp16_from_bf16_bits(bits) {
            Ok(mapped) => mapped,
            Err(()) => anyhow::bail!("{name}: BF16 weight is non-finite or above 65504"),
        };
        flushed += u64::from(flush);
        *pair = out.to_le_bytes();
    }
    if report && flushed > 0 {
        eprintln!("{name}: BF16 to FP16 flushed {flushed} values to +0");
    }
    Ok(flushed)
}

/// Generated BF16 table words become the FP16 encoding [`bf16_weight_to_fp16`] stores.
pub fn bf16_table_to_runtime_half(name: &str, table: &mut [half::bf16]) -> Result<()> {
    let mut bytes = Vec::with_capacity(table.len() * 2);
    for value in table.iter() {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bf16_weight_to_fp16(name, &mut bytes, false)?;
    for (value, pair) in table.iter_mut().zip(bytes.as_chunks::<2>().0) {
        *value = half::bf16::from_bits(u16::from_le_bytes(*pair));
    }
    Ok(())
}

/// `fp16` rewrites a BF16 tensor through [`bf16_weight_to_fp16`]. F32, I8, and U8 bytes stay as stored.
pub fn convert_loaded_weight(
    name: &str,
    dtype: Dtype,
    bytes: &mut [u8],
    fp16: bool,
) -> Result<u64> {
    if !fp16 || dtype != Dtype::BF16 {
        return Ok(0);
    }
    bf16_weight_to_fp16(name, bytes, false)
}

fn fp16_from_bf16_bits(bits: u16) -> Result<(u16, bool), ()> {
    let exp = (bits >> 7) & 0xFF;
    let man = bits & 0x7F;
    let sign = bits >> 15;
    if exp == 255 || exp >= 143 {
        return Err(());
    }
    if exp == 0 && man == 0 {
        return Ok((sign << 15, false));
    }
    if exp <= 102 {
        return Ok((0, true));
    }
    let mag = fp16_magnitude(exp, man)?;
    Ok(((sign << 15) | mag, false))
}

fn fp16_magnitude(exp: u16, man: u16) -> Result<u16, ()> {
    let sig = 128u32 + u32::from(man);
    let mag = if exp >= 113 {
        let fp_exp = u32::from(exp - 112);
        let fp_man = u32::from(man) << 3;
        (fp_exp << 10) | fp_man
    } else {
        let shift = 110 - i32::from(exp);
        if shift > 0 {
            round_ties_to_even(sig, shift.unsigned_abs())
        } else {
            sig << shift.unsigned_abs()
        }
    };
    u16::try_from(mag).map_err(|_| ())
}

fn round_ties_to_even(value: u32, shift: u32) -> u32 {
    let half = 1u32 << (shift - 1);
    let trunc = value >> shift;
    let rem = value & ((half << 1) - 1);
    if rem > half || (rem == half && trunc & 1 == 1) {
        trunc + 1
    } else {
        trunc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestor_bit63_is_stored_as_native_u64() {
        let bits = (1u64 << 63) | 1;
        let mut tensors = Tensors::default();
        tensors.u64("anc".into(), vec![1], &[bits]).unwrap();
        let (dtype, shape, data) = tensors.values.get("anc").unwrap();
        assert_eq!(*dtype, Dtype::U64);
        let view = TensorView::new(*dtype, shape.clone(), data).unwrap();
        let bytes = safetensors::tensor::serialize([("anc", view)], None).unwrap();
        let loaded = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let anc = loaded.tensor("anc").unwrap();
        assert_eq!(u64::from_le_bytes(anc.data().try_into().unwrap()), bits);
    }

    #[test]
    fn sm75_rotary_table_stores_fp16_one() {
        let mut one = [half::bf16::from_bits(0x3f80)];
        bf16_table_to_runtime_half("rotary", &mut one).unwrap();
        assert_eq!(one[0].to_bits(), 0x3c00);
        let mut signed = [half::bf16::from_bits(0xbf80), half::bf16::from_bits(0x3f00)];
        bf16_table_to_runtime_half("rotary", &mut signed).unwrap();
        assert_eq!(signed[0].to_bits(), 0xbc00);
        assert_eq!(signed[1].to_bits(), 0x3800);
    }

    #[test]
    fn sm75_adaln_embedding_stores_fp16_one() {
        let mut one = [half::bf16::from_bits(0x3f80)];
        bf16_table_to_runtime_half("adaln", &mut one).unwrap();
        assert_eq!(one[0].to_bits(), 0x3c00);
    }

    #[test]
    fn sm75_conditioner_text_stores_fp16_one() {
        let mut one = [half::bf16::from_f32(1.0)];
        bf16_table_to_runtime_half("image_text", &mut one).unwrap();
        assert_eq!(one[0].to_bits(), 0x3c00);
        let mut video = [half::bf16::from_f32(1.0)];
        bf16_table_to_runtime_half("video_text", &mut video).unwrap();
        assert_eq!(video[0].to_bits(), 0x3c00);
    }

    fn word(sign: u16, exp: u16, man: u16) -> [u8; 2] {
        ((sign << 15) | (exp << 7) | man).to_le_bytes()
    }

    #[test]
    fn bf16_weight_to_fp16_classifies_exponent_bins() {
        let name = "model.language_model.layers.1.mlp.gate_proj.weight";
        let mut kept = word(0, 142, 127);
        assert_eq!(bf16_weight_to_fp16(name, &mut kept, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(kept), 0x7bf8);
        let mut normal = word(0, 142, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut normal, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(normal), 0x7800);
        let mut negative = word(1, 142, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut negative, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(negative), 0xf800);
        let mut min_normal = word(0, 113, 0);
        assert_eq!(
            bf16_weight_to_fp16(name, &mut min_normal, false).unwrap(),
            0
        );
        assert_eq!(u16::from_le_bytes(min_normal), 0x0400);
        let mut subnormal = word(0, 110, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut subnormal, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(subnormal), 0x0080);
        let mut exact = word(0, 103, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut exact, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(exact), 0x0001);
        let mut exact_neg = word(1, 103, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut exact_neg, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(exact_neg), 0x8001);
        let mut down = word(0, 103, 63);
        assert_eq!(bf16_weight_to_fp16(name, &mut down, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(down), 0x0001);
        let mut up = word(0, 103, 64);
        assert_eq!(bf16_weight_to_fp16(name, &mut up, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(up), 0x0002);
        let mut pos0 = word(0, 0, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut pos0, false).unwrap(), 0);
        assert_eq!(pos0, [0, 0]);
        let mut neg0 = word(1, 0, 0);
        assert_eq!(bf16_weight_to_fp16(name, &mut neg0, false).unwrap(), 0);
        assert_eq!(u16::from_le_bytes(neg0), 0x8000);
        for (sign, exp, man) in [(0, 102, 0), (1, 102, 127), (0, 0, 1), (1, 1, 1)] {
            let mut bits = word(sign, exp, man);
            assert_eq!(bf16_weight_to_fp16(name, &mut bits, false).unwrap(), 1);
            assert_eq!(bits, [0, 0]);
        }
        let mut mixed = Vec::new();
        mixed.extend(word(0, 142, 0));
        mixed.extend(word(1, 102, 3));
        mixed.extend(word(1, 0, 0));
        assert_eq!(bf16_weight_to_fp16(name, &mut mixed, false).unwrap(), 1);
        assert_eq!(&mixed[0..2], &0x7800u16.to_le_bytes());
        assert_eq!(&mixed[2..4], &[0, 0]);
        assert_eq!(&mixed[4..6], &0x8000u16.to_le_bytes());
        for (sign, exp, man) in [
            (0, 143, 0),
            (1, 143, 127),
            (0, 254, 0),
            (0, 255, 0),
            (1, 255, 0),
            (0, 255, 1),
            (1, 255, 0x7f),
        ] {
            let mut bits = word(sign, exp, man);
            let before = bits;
            let err = bf16_weight_to_fp16(name, &mut bits, false).unwrap_err();
            assert!(err.to_string().contains(name), "{err}");
            assert_eq!(bits, before);
        }
        let mut partial = Vec::new();
        partial.extend(word(0, 142, 0));
        partial.extend(word(0, 143, 0));
        let before = partial.clone();
        let err = bf16_weight_to_fp16(name, &mut partial, false).unwrap_err();
        assert!(err.to_string().contains(name), "{err}");
        assert_eq!(partial, before);
        let mut f32_bytes = 1.0e10f32.to_le_bytes();
        let f32_before = f32_bytes;
        assert_eq!(
            convert_loaded_weight("vae.conv.weight", Dtype::F32, &mut f32_bytes, true).unwrap(),
            0
        );
        assert_eq!(f32_bytes, f32_before);
        let mut i8_bytes = [0x7f];
        assert_eq!(
            convert_loaded_weight("qweight", Dtype::I8, &mut i8_bytes, true).unwrap(),
            0
        );
        assert_eq!(i8_bytes, [0x7f]);
        let mut u8_bytes = [0xff];
        assert_eq!(
            convert_loaded_weight("scale", Dtype::U8, &mut u8_bytes, true).unwrap(),
            0
        );
        assert_eq!(u8_bytes, [0xff]);
        let mut stored = word(0, 102, 1);
        let stored_before = stored;
        assert_eq!(
            convert_loaded_weight(name, Dtype::BF16, &mut stored, false).unwrap(),
            0
        );
        assert_eq!(stored, stored_before);
        assert_eq!(
            convert_loaded_weight(name, Dtype::BF16, &mut stored, true).unwrap(),
            1
        );
        assert_eq!(stored, [0, 0]);
    }
}
