//! Mmap-backed tensor access for the Edge0 checkpoint.

use crate::int4::{GROUP_SIZE, GroupQuant, bf16_to_f32};
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::{fs::File, path::Path};

/// Whole stacked expert tensor: packed payload + scales/biases + geometry.
pub type StackedProjection = (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize);

struct Shard {
    _map: Mmap,
    tensors: SafeTensors<'static>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let map = unsafe { Mmap::map(&File::open(path)?) }?;
        let bytes: &'static [u8] = unsafe { std::mem::transmute(&map[..]) };
        Ok(Self {
            _map: map,
            tensors: SafeTensors::deserialize(bytes)?,
        })
    }
}

pub struct Edge0Weights {
    shards: Vec<Shard>,
    index: HashMap<String, usize>,
    lora: HashMap<String, (Vec<f32>, Vec<f32>)>,
    pub lora_rank: usize,
}

const LORA_FILE: &str = "lora_edge0_35b.safetensors";

impl Edge0Weights {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let mut shard_paths: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
                    .unwrap_or(false)
            })
            .collect();
        shard_paths.sort();
        ensure!(
            !shard_paths.is_empty(),
            "no model-*.safetensors under {}",
            model_dir.display()
        );
        let mut shards = Vec::new();
        let mut index = HashMap::new();
        for (position, path) in shard_paths.iter().enumerate() {
            let shard =
                Shard::open(path).with_context(|| format!("failed to mmap {}", path.display()))?;
            for name in shard.tensors.names() {
                index.insert(name.to_string(), position);
            }
            shards.push(shard);
        }
        let mut lora = HashMap::new();
        let mut lora_rank = 0usize;
        let lora_path = model_dir.join(LORA_FILE);
        if lora_path.is_file() {
            let map = unsafe { Mmap::map(&File::open(&lora_path)?) }?;
            let bytes: &'static [u8] = unsafe { std::mem::transmute(&map[..]) };
            let view = SafeTensors::deserialize(bytes)?;
            let mut pairs: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
            let mut pair_rank: HashMap<String, usize> = HashMap::new();
            for name in view.names() {
                let base = name.rsplit_once(".lora_").map(|(b, _)| b.to_string());
                let Some(base) = base else { continue };
                let tensor = view.tensor(name)?;
                let is_a = name.ends_with("lora_A");
                let values = f16_to_f32(tensor.data());
                let rank = if is_a {
                    values.len() / tensor.shape()[1]
                } else {
                    tensor.shape()[1]
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
            // The kernels take one rank per launch — verify uniform instead
            // of the old silent global-max.
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
            shards,
            index,
            lora,
            lora_rank,
        })
    }

    pub fn lora_for(&self, projection: &str) -> Option<(&[f32], &[f32], usize)> {
        if let Some(skip) = std::env::var_os("EDGE0_SKIP_LORA") {
            let skip = skip.to_string_lossy();
            if skip
                .split(',')
                .any(|p| !p.is_empty() && projection.contains(p))
            {
                return None;
            }
        }
        self.lora
            .get(projection)
            .map(|(a, b)| (a.as_slice(), b.as_slice(), self.lora_rank))
    }

    fn view(&self, name: &str) -> Result<safetensors::tensor::TensorView<'static>> {
        let shard = self
            .index
            .get(name)
            .and_then(|position| self.shards.get(*position))
            .with_context(|| format!("tensor {name} not found"))?;
        shard
            .tensors
            .tensor(name)
            .with_context(|| format!("failed to view {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn f32_named(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.view(name)?;
        let data = tensor.data();
        Ok(match tensor.dtype() {
            safetensors::Dtype::BF16 => bf16_to_f32(data),
            safetensors::Dtype::F16 => f16_to_f32(data),
            other => anyhow::bail!("tensor {name} is {other:?}, expected bf16/f16"),
        })
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.view(name)?.shape().to_vec())
    }

    /// Weighed (not derived) weight bytes by bucket: experts, static, and
    /// embed+lm_head, each split into packed vs scale/bias. The exact
    /// replacement for hand-derived geometry formulas.
    pub fn bucket_bytes(&self) -> Result<(u64, u64, u64, u64, u64, u64)> {
        let mut expert = (0u64, 0u64);
        let mut statik = (0u64, 0u64);
        let mut embed = (0u64, 0u64);
        for name in self.index.keys() {
            let tensor = self.view(name)?;
            let dtype_size = match tensor.dtype() {
                safetensors::Dtype::U32 | safetensors::Dtype::F32 => 4,
                safetensors::Dtype::BF16
                | safetensors::Dtype::F16
                | safetensors::Dtype::I16
                | safetensors::Dtype::U16 => 2,
                safetensors::Dtype::U8 | safetensors::Dtype::I8 => 1,
                other => anyhow::bail!("unexpected dtype {other:?} for {name}"),
            };
            let bytes: u64 = tensor.shape().iter().map(|d| *d as u64).product::<u64>() * dtype_size;
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
        let weight = self.view(&format!("{projection}.weight"))?;
        let shape = weight.shape();
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
        let packed = packed_words(&weight)?;
        GroupQuant::new(packed, scales, biases, out_dim, in_dim, bits)
    }

    /// Load the WHOLE stacked tensor for a layer+projection: packed
    /// [256, rows, words], scales/biases [256, rows, groups] — the batched
    /// kernel's base+expert-index addressing needs whole-tensor residency.
    pub fn stacked_projection(&self, layer: usize, projection: &str) -> Result<StackedProjection> {
        let name = format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}");
        let weight = self.view(&format!("{name}.weight"))?;
        let scales = self.view(&format!("{name}.scales"))?;
        let biases = self.view(&format!("{name}.biases"))?;
        let shape = weight.shape();
        ensure!(shape.len() == 3, "{name}.weight is not stacked: {shape:?}");
        let rows = shape[1];
        let in_dim = shape[2] * 8;
        let packed = le_words(weight.data());
        let s = bf16_to_f32(scales.data());
        let b = bf16_to_f32(biases.data());
        Ok((packed, s, b, rows, in_dim))
    }

    /// Load one expert's projection from the stacked `switch_mlp` tensor.
    pub fn quant_expert(
        &self,
        layer: usize,
        expert: usize,
        projection: &str,
    ) -> Result<GroupQuant> {
        let name = format!("language_model.model.layers.{layer}.mlp.switch_mlp.{projection}");
        let weight = self.view(&format!("{name}.weight"))?;
        let scales = self.view(&format!("{name}.scales"))?;
        let biases = self.view(&format!("{name}.biases"))?;
        let shape = weight.shape();
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
        let payload = weight.data();
        let offset = expert * word_bytes;
        let words = le_words(&payload[offset..offset + word_bytes]);

        let slice_bf16 = |view: &safetensors::tensor::TensorView<'static>| -> Result<Vec<f32>> {
            let raw = view.data();
            let start = expert * rows * groups * 2;
            let end = start + rows * groups * 2;
            ensure!(
                end <= raw.len(),
                "{name}.scales/biases too short for expert {expert}: need {end} bytes, have {}",
                raw.len()
            );
            Ok(bf16_to_f32(&raw[start..end]))
        };
        GroupQuant::new(
            words,
            slice_bf16(&scales)?,
            slice_bf16(&biases)?,
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

fn packed_words(view: &safetensors::tensor::TensorView<'static>) -> Result<Vec<u32>> {
    ensure!(
        view.dtype() == safetensors::Dtype::U32,
        "expected U32 payload, got {:?}",
        view.dtype()
    );
    Ok(le_words(view.data()))
}
