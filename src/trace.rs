use anyhow::{Result, ensure};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize_to_file},
};
use std::{collections::BTreeMap, path::Path};

#[derive(Default)]
pub struct Trace {
    tensors: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
}

impl Trace {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let mut result = Self::default();
        for (name, tensor) in tensors.tensors() {
            ensure!(tensor.dtype() == Dtype::F32, "trace {name} must be FP32");
            result
                .tensors
                .insert(name, (tensor.shape().to_vec(), tensor.data().to_vec()));
        }
        Ok(result)
    }

    pub fn prefill_parts(&self) -> Result<usize> {
        let mut parts = Vec::new();
        for name in self.tensors.keys() {
            if let Some(part) = name
                .strip_prefix("prefill.")
                .and_then(|s| s.strip_suffix(".embed"))
            {
                parts.push(part.parse::<usize>()?);
            }
        }
        parts.sort_unstable();
        ensure!(
            !parts.is_empty() && parts.iter().copied().eq(0..parts.len()),
            "trace prefill parts must be contiguous from 0"
        );
        Ok(parts.len())
    }

    pub fn values(&self, name: &str) -> Result<Vec<f32>> {
        let (_, bytes) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing trace tensor {name}"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("float width")))
            .collect())
    }

    pub fn add(&mut self, name: String, shape: Vec<usize>, data: Vec<f32>) -> Result<()> {
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "nonfinite tensor {name}"
        );
        ensure!(
            shape.iter().product::<usize>() == data.len(),
            "trace shape mismatch {name}"
        );
        let bytes = data.into_iter().flat_map(f32::to_le_bytes).collect();
        ensure!(
            self.tensors.insert(name.clone(), (shape, bytes)).is_none(),
            "duplicate trace {name}"
        );
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let views = self
            .tensors
            .iter()
            .map(|(name, (shape, data))| {
                Ok((
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, path)?;
        Ok(())
    }
}

impl Trace {
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing trace tensor {name}"))?
            .0)
    }
}
