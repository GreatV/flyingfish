//! Mmap-backed tensor access for the Edge0 checkpoint.

use crate::int4::{GROUP_SIZE, GroupQuant, bf16_to_f32};
use anyhow::{Result, ensure};
use ff_core::weights::{CachePolicy, ModelWeights, WeightSource};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::Path;

/// Whole stacked expert tensor: packed payload + scales/biases + geometry.
pub type StackedProjection = (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize);

pub struct Edge0Weights {
    weights: ModelWeights,
    lora: HashMap<String, (Vec<f32>, Vec<f32>)>,
    pub lora_rank: usize,
}

const LORA_FILE: &str = "lora_edge0_35b.safetensors";

fn open_source(root: &Path, file: Option<&str>) -> Result<ModelWeights> {
    let policy = CachePolicy::unbounded_units();
    match file {
        Some(file) => ModelWeights::open_component(root, file, WeightSource::Mmap, policy),
        None => ModelWeights::open(root, WeightSource::Mmap, policy),
    }
}

impl Edge0Weights {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let weights = open_source(model_dir, None)?;
        let mut lora = HashMap::new();
        let mut lora_rank = 0usize;
        if model_dir.join(LORA_FILE).is_file() {
            let adapter = open_source(model_dir, Some(LORA_FILE))?;
            let mut pairs: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
            let mut pair_rank: HashMap<String, usize> = HashMap::new();
            for name in adapter.tensor_names() {
                let base = name.rsplit_once(".lora_").map(|(b, _)| b.to_string());
                let Some(base) = base else { continue };
                let shape = adapter.raw_tensor_metadata(name)?.shape;
                let is_a = name.ends_with("lora_A");
                let values = adapter.with_tensor_bytes(name, |data| Ok(f16_to_f32(data)))?;
                let rank = if is_a {
                    values.len() / shape[1]
                } else {
                    shape[1]
                };
                if let Some(prev) = pair_rank.insert(base.clone(), rank) {
                    anyhow::ensure!(
                        prev == rank,
                        "lora pair {base}: A rank {prev} != B rank {rank}"
                    );
                }
                let entry = pairs
                    .entry(base)
                    .or_insert_with(|| (Vec::new(), Vec::new()));
                if is_a {
                    entry.0 = values;
                } else {
                    entry.1 = values;
                }
            }
            // All adapter pairs must have the same rank.
            for (base, (a, b)) in &pairs {
                anyhow::ensure!(
                    !a.is_empty() && !b.is_empty(),
                    "lora pair {base}: missing a half — a lone adapter reads out of bounds downstream"
                );
            }
            let mut ranks = pair_rank.iter();
            if let Some((_, &first)) = ranks.next() {
                for (base, &r) in ranks {
                    anyhow::ensure!(
                        r == first,
                        "lora rank mismatch: {base} has {r}, expected {first} (kernels take one rank per launch)"
                    );
                }
                lora_rank = first;
            }
            lora = pairs;
        }
        Ok(Self {
            weights,
            lora,
            lora_rank,
        })
    }

    pub fn lora_for(&self, projection: &str) -> Option<(&[f32], &[f32], usize)> {
        self.lora
            .get(projection)
            .map(|(a, b)| (a.as_slice(), b.as_slice(), self.lora_rank))
    }

    pub fn has(&self, name: &str) -> bool {
        self.weights.contains(name)
    }

    pub fn f32_named(&self, name: &str) -> Result<Vec<f32>> {
        let dtype = self.weights.raw_tensor_metadata(name)?.dtype;
        self.weights.with_tensor_bytes(name, |data| {
            Ok(match dtype {
                Dtype::BF16 => bf16_to_f32(data),
                Dtype::F16 => f16_to_f32(data),
                other => anyhow::bail!("tensor {name} is {other:?}, expected bf16/f16"),
            })
        })
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.weights.raw_tensor_metadata(name)?.shape)
    }

    /// Weighed (not derived) weight bytes by bucket: experts, static, and
    /// embed+lm_head, each split into packed vs scale/bias. The exact
    /// replacement for hand-derived geometry formulas.
    pub fn bucket_bytes(&self) -> Result<(u64, u64, u64, u64, u64, u64)> {
        let mut expert = (0u64, 0u64);
        let mut statik = (0u64, 0u64);
        let mut embed = (0u64, 0u64);
        for name in self.weights.tensor_names() {
            let bytes = self.weights.raw_tensor_metadata(name)?.bytes as u64;
            let is_sb = name.ends_with(".scales") || name.ends_with(".biases");
            let bucket = if name.contains(".switch_mlp.") {
                &mut expert
            } else if name.contains("embed_tokens") || name.contains("lm_head") {
                &mut embed
            } else {
                &mut statik
            };
            if is_sb {
                bucket.1 += bytes;
            } else {
                bucket.0 += bytes;
            }
        }
        Ok((expert.0, expert.1, statik.0, statik.1, embed.0, embed.1))
    }

    /// Load `{projection}.weight` as a GroupQuant with `{projection}.scales`
    /// and `{projection}.biases`. The width (4 or 8 bits) is derived from
    /// the scales shape, which pins the true input dimension.
    pub fn quant_projection(&self, projection: &str) -> Result<GroupQuant> {
        let weight_name = format!("{projection}.weight");
        let meta = self.weights.raw_tensor_metadata(&weight_name)?;
        let shape = meta.shape;
        ensure!(
            shape.len() == 2,
            "{projection}.weight is not 2-D: {shape:?}"
        );
        let out_dim = shape[0];
        ensure!(out_dim > 0, "{projection}.weight has zero rows");
        let scales = self.f32_named(&format!("{projection}.scales"))?;
        let biases = self.f32_named(&format!("{projection}.biases"))?;
        let groups = scales.len() / out_dim;
        let in_dim = groups * GROUP_SIZE;
        let bits = if shape[1] * 8 == in_dim {
            4
        } else if shape[1] * 4 == in_dim {
            8
        } else {
            anyhow::bail!(
                "{projection}: packed width {} matches neither int4 nor int8 for in_dim {in_dim}",
                shape[1]
            )
        };
        ensure!(
            meta.dtype == Dtype::U32,
            "expected U32 payload, got {:?}",
            meta.dtype
        );
        let packed = self
            .weights
            .with_tensor_bytes(&weight_name, |data| Ok(le_words(data)))?;
        GroupQuant::new(packed, scales, biases, out_dim, in_dim, bits)
    }

    /// Load the WHOLE stacked tensor for a layer+projection: packed
    /// [256, rows, words], scales/biases [256, rows, groups] — the batched
    /// kernel's base+expert-index addressing needs whole-tensor residency.
    pub fn stacked_projection(&self, layer: usize, projection: &str) -> Result<StackedProjection> {
        let name = format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}");
        let weight_name = format!("{name}.weight");
        let shape = self.weights.raw_tensor_metadata(&weight_name)?.shape;
        ensure!(shape.len() == 3, "{name}.weight is not stacked: {shape:?}");
        let rows = shape[1];
        let in_dim = shape[2] * 8;
        let packed = self
            .weights
            .with_tensor_bytes(&weight_name, |data| Ok(le_words(data)))?;
        let s = self.bf16_named(&format!("{name}.scales"))?;
        let b = self.bf16_named(&format!("{name}.biases"))?;
        Ok((packed, s, b, rows, in_dim))
    }

    fn bf16_named(&self, name: &str) -> Result<Vec<f32>> {
        self.weights
            .with_tensor_bytes(name, |data| Ok(bf16_to_f32(data)))
    }

    /// Load one expert's projection from the stacked `switch_mlp` tensor.
    pub fn quant_expert(
        &self,
        layer: usize,
        expert: usize,
        projection: &str,
    ) -> Result<GroupQuant> {
        let name = format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}");
        let weight_name = format!("{name}.weight");
        let shape = self.weights.raw_tensor_metadata(&weight_name)?.shape;
        ensure!(shape.len() == 3, "{name}.weight is not stacked: {shape:?}");
        ensure!(
            expert < shape[0],
            "{name}: expert {expert} out of range; tensor stacks {}",
            shape[0]
        );
        let rows = shape[1];
        let packed_cols = shape[2];
        let in_dim = packed_cols * 8;
        let groups = in_dim / GROUP_SIZE;

        let word_bytes = rows * packed_cols * 4;
        let offset = expert * word_bytes;
        let words = self.weights.with_tensor_bytes(&weight_name, |payload| {
            ensure!(
                offset + word_bytes <= payload.len(),
                "{weight_name}: expert {expert} lies past the payload"
            );
            Ok(le_words(&payload[offset..offset + word_bytes]))
        })?;

        let slice_bf16 = |tensor: &str| -> Result<Vec<f32>> {
            self.weights.with_tensor_bytes(tensor, |raw| {
                let start = expert * rows * groups * 2;
                let end = start + rows * groups * 2;
                ensure!(
                    end <= raw.len(),
                    "{tensor} too short for expert {expert}: need {end} bytes, have {}",
                    raw.len()
                );
                Ok(bf16_to_f32(&raw[start..end]))
            })
        };
        GroupQuant::new(
            words,
            slice_bf16(&format!("{name}.scales"))?,
            slice_bf16(&format!("{name}.biases"))?,
            rows,
            in_dim,
            4,
        )
    }
}

fn f16_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|pair| {
            let bits = u16::from(pair[0]) | (u16::from(pair[1]) << 8);
            f32::from(half::f16::from_bits(bits))
        })
        .collect()
}

fn le_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    const DENSE: &str = "language_model.model.layers.0.self_attn.q_proj";

    fn words(count: usize) -> Vec<u8> {
        (0..count as u32)
            .flat_map(|w| w.wrapping_mul(2654435761).to_le_bytes())
            .collect()
    }

    fn halves(count: usize, one: u16) -> Vec<u8> {
        (0..count as u16)
            .flat_map(|v| (one + v).to_le_bytes())
            .collect()
    }

    fn write(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
            })
            .collect();
        safetensors::serialize_to_file(views, None, path).unwrap();
    }

    fn fixture(dir: &Path) -> Vec<(&'static str, Dtype, Vec<usize>, Vec<u8>)> {
        let tensors = vec![
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                Dtype::U32,
                vec![2, 3, 8],
                words(48),
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
                Dtype::BF16,
                vec![2, 3, 1],
                halves(6, 0x3f80),
            ),
            (
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
                Dtype::BF16,
                vec![2, 3, 1],
                halves(6, 0x3e80),
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                Dtype::U32,
                vec![3, 8],
                words(24),
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.scales",
                Dtype::BF16,
                vec![3, 1],
                halves(3, 0x3f80),
            ),
            (
                "language_model.model.layers.0.self_attn.q_proj.biases",
                Dtype::BF16,
                vec![3, 1],
                halves(3, 0x3e80),
            ),
        ];
        write(&dir.join("model-00001-of-00001.safetensors"), &tensors);
        let map: serde_json::Map<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, ..)| (name.to_string(), "model-00001-of-00001.safetensors".into()))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 0},
                "weight_map": map,
            }))
            .unwrap(),
        )
        .unwrap();
        write(
            &dir.join(LORA_FILE),
            &[
                ("proj.lora_A", Dtype::F16, vec![2, 4], halves(8, 0x3c00)),
                ("proj.lora_B", Dtype::F16, vec![3, 2], halves(6, 0x3c00)),
            ],
        );
        tensors
    }

    #[test]
    fn shared_traversal_yields_the_tensors_of_the_files() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = fixture(dir.path());
        let weights = Edge0Weights::open(dir.path()).unwrap();
        for (name, dtype, shape, data) in &tensors {
            assert!(weights.has(name), "{name}");
            assert_eq!(&weights.shape(name).unwrap(), shape, "{name}");
            let meta = weights.weights.raw_tensor_metadata(name).unwrap();
            assert_eq!(meta.dtype, *dtype, "{name}");
            let bytes = weights
                .weights
                .with_tensor_bytes(name, |bytes| Ok(bytes.to_vec()))
                .unwrap();
            assert_eq!(&bytes, data, "{name}");
        }
        let total: u64 = tensors.iter().map(|t| t.3.len() as u64).sum();
        let (ep, es, sp, ss, hp, hs) = weights.bucket_bytes().unwrap();
        assert_eq!(ep + es + sp + ss + hp + hs, total);

        let stacked = weights.stacked_projection(0, "gate_proj").unwrap();
        assert_eq!(stacked.0, le_words(&tensors[0].3));
        assert_eq!((stacked.3, stacked.4), (3, 64));
        assert_eq!(stacked.1, bf16_to_f32(&tensors[1].3));
        assert_eq!(stacked.2, bf16_to_f32(&tensors[2].3));

        let expert = weights.quant_expert(0, 1, "gate_proj").unwrap();
        assert_eq!(expert.packed, le_words(&tensors[0].3[96..]));
        assert_eq!(expert.scales, bf16_to_f32(&tensors[1].3[6..]));
        assert_eq!(expert.biases, bf16_to_f32(&tensors[2].3[6..]));
        assert!(weights.quant_expert(0, 2, "gate_proj").is_err());

        let dense = weights.quant_projection(DENSE).unwrap();
        assert_eq!(dense.packed, le_words(&tensors[3].3));
        assert_eq!(weights.lora_rank, 2);
        let (a, b, rank) = weights.lora_for("proj").unwrap();
        assert_eq!((a.len(), b.len(), rank), (8, 6, 2));
    }

    #[test]
    fn a_missing_tensor_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        let weights = Edge0Weights::open(dir.path()).unwrap();
        let error = weights.shape("absent.weight").unwrap_err();
        assert!(format!("{error:#}").contains("absent.weight"));
    }
}
