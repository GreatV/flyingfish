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
}
