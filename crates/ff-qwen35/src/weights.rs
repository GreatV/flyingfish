//! Mmap-backed tensor access for a requant-ed Qwen3.8-27B checkpoint
//! (the int4 layout written by `src/bin/requant.rs`).

use anyhow::{Context, Result, ensure};
use ff_edge0::int4::{GROUP_SIZE, GroupQuant, bf16_to_f32};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::{fs::File, path::Path};

struct Shard {
    _map: Mmap,
    tensors: SafeTensors<'static>,
}

pub struct Qwen35Weights {
    shards: Vec<Shard>,
    index: HashMap<String, usize>,
}

impl Qwen35Weights {
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
            let map = unsafe { Mmap::map(&File::open(path)?) }
                .with_context(|| format!("failed to mmap {}", path.display()))?;
            let bytes: &'static [u8] = unsafe { std::mem::transmute(&map[..]) };
            let shard = SafeTensors::deserialize(bytes)
                .with_context(|| format!("parse {}", path.display()))?;
            for name in shard.names() {
                index.insert(name.to_string(), position);
            }
            shards.push(Shard {
                _map: map,
                tensors: shard,
            });
        }
        Ok(Self { shards, index })
    }

    fn view(&self, name: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        let &shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))?;
        let t = &self.shards[shard].tensors;
        let tensor = t.tensor(name)?;
        Ok((tensor.shape().to_vec(), tensor.data().to_vec()))
    }

    /// True when `{name}.weight` is a U32-packed int4 tensor (requant
    /// output) rather than a plain bf16 matrix (raw checkpoint).
    pub fn is_quant(&self, name: &str) -> bool {
        let Some(&shard) = self.index.get(&format!("{name}.weight")) else {
            return false;
        };
        self.shards[shard]
            .tensors
            .tensor(&format!("{name}.weight"))
            .map(|t| t.dtype() == safetensors::Dtype::U32)
            .unwrap_or(false)
    }

    /// Zero-copy tensor view straight off the mmap (file-backed pages are
    /// reclaimable; a materialized f32 copy is not).
    fn view_ref(&self, name: &str) -> Result<(Vec<usize>, &'static [u8])> {
        let &shard = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))?;
        let tensor = self.shards[shard].tensors.tensor(name)?;
        Ok((tensor.shape().to_vec(), tensor.data()))
    }

    /// Streaming bf16 matvec over the mmap: per-element widening, threaded
    /// over row chunks, no f32 matrix residency.
    pub fn bf16_matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let (shape, data) = self.view_ref(&format!("{name}.weight"))?;
        ensure!(shape.len() == 2, "{name}.weight is not 2-D: {shape:?}");
        let (rows, cols) = (shape[0], shape[1]);
        ensure!(x.len() == cols, "{name}: x {} != in_dim {cols}", x.len());
        let mut y = vec![0f32; rows];
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .min(16);
        let chunk = rows.div_ceil(threads);
        std::thread::scope(|s| {
            for (t, y_chunk) in y.chunks_mut(chunk).enumerate() {
                let (data, x) = (data, x);
                s.spawn(move || {
                    let r0 = t * chunk;
                    for (o, slot) in y_chunk.iter_mut().enumerate() {
                        let row = &data[(r0 + o) * cols * 2..(r0 + o + 1) * cols * 2];
                        let mut acc = 0f32;
                        for (c, xv) in x.iter().enumerate() {
                            let w = f32::from_bits(
                                (u32::from(row[c * 2]) | (u32::from(row[c * 2 + 1]) << 8)) << 16,
                            );
                            acc += w * xv;
                        }
                        *slot = acc;
                    }
                });
            }
        });
        Ok(y)
    }

    /// One bf16 row widened (embed gather).
    pub fn bf16_row(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let (shape, data) = self.view_ref(&format!("{name}.weight"))?;
        ensure!(shape.len() == 2 && row < shape[0], "{name}: row {row}");
        let cols = shape[1];
        Ok(bf16_to_f32(&data[row * cols * 2..(row + 1) * cols * 2]))
    }

    /// bf16 small tensor (norms, conv, A_log, dt_bias) widened to f32.
    pub fn f32_named(&self, name: &str) -> Result<Vec<f32>> {
        let (_shape, data) = self.view(name)?;
        Ok(bf16_to_f32(&data))
    }

    /// bf16 tensor of any rank (the vision tower's conv/2-D weights),
    /// widened to f32. Shape preserved.
    pub fn bf16_tensor(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, data) = self.view(name)?;
        Ok((shape, bf16_to_f32(&data)))
    }

    /// `{name}.weight` (U32 packed) + `.scales`/`.biases` (bf16) -> GroupQuant.
    pub fn quant_projection(&self, name: &str) -> Result<GroupQuant> {
        let (shape, packed) = self.view(&format!("{name}.weight"))?;
        ensure!(shape.len() == 2, "{name}.weight is not 2-D: {shape:?}");
        let out_dim = shape[0];
        let (_s_shape, scales) = self.view(&format!("{name}.scales"))?;
        let (_b_shape, biases) = self.view(&format!("{name}.biases"))?;
        let scales = bf16_to_f32(&scales);
        let biases = bf16_to_f32(&biases);
        let packed: Vec<u32> = packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let groups = scales.len() / out_dim;
        let in_dim = groups * GROUP_SIZE;
        ensure!(
            shape[1] * 8 == in_dim,
            "{name}: weight width {} words implies in={} but scales imply {in_dim}",
            shape[1],
            shape[1] * 8
        );
        GroupQuant::new(packed, scales, biases, out_dim, in_dim, 4)
    }
}
