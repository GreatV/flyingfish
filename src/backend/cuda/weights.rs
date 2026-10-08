use super::Device;
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, PinnedHostSlice};
use half::bf16;
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::Path,
    sync::Arc,
};

pub struct Layer {
    pub input_norm: CudaSlice<bf16>,
    pub qkv: CudaSlice<bf16>,
    pub o: CudaSlice<bf16>,
    pub post_norm: CudaSlice<bf16>,
    pub gu: CudaSlice<bf16>,
    pub down: CudaSlice<bf16>,
}

pub struct Weights {
    pub embed: CudaSlice<bf16>,
    pub layers: Vec<Layer>,
    pub norm: CudaSlice<bf16>,
    pub head: CudaSlice<bf16>,
    pub bytes: usize,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

pub(crate) struct Loader<'a> {
    tensors: BTreeMap<String, SafeTensors<'a>>,
    index: Index,
    seen: BTreeSet<String>,
    staging: PinnedHostSlice<bf16>,
    stream: Arc<CudaStream>,
    pub(crate) bytes: usize,
}

impl Loader<'_> {
    pub(crate) fn load(&mut self, specs: &[(String, Vec<usize>)]) -> Result<CudaSlice<bf16>> {
        let count: usize = specs.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
        let mut dst = unsafe { self.stream.alloc::<bf16>(count)? };
        let mut offset = 0;
        for (name, shape) in specs {
            let file = self
                .index
                .weight_map
                .get(name)
                .with_context(|| format!("missing tensor {name}"))?;
            let t = self.tensors[file].tensor(name)?;
            ensure!(
                t.dtype() == Dtype::BF16,
                "tensor {name}: expected BF16, found {:?}",
                t.dtype()
            );
            ensure!(
                t.shape() == shape,
                "tensor {name}: expected {shape:?}, found {:?}",
                t.shape()
            );
            ensure!(self.seen.insert(name.clone()), "tensor {name} loaded twice");
            for chunk in t.data().chunks(self.staging.len() * 2) {
                let len = chunk.len() / 2;
                let buf = self.staging.as_mut_slice()?;
                decode_bf16(chunk, &mut buf[..len]);
                let mut view = dst.slice_mut(offset..offset + len);
                self.stream
                    .memcpy_htod(&self.staging.as_slice()?[..len], &mut view)?;
                self.stream.synchronize()?;
                offset += len;
            }
            self.bytes += t.data().len();
        }
        Ok(dst)
    }
}

pub(crate) fn checkpoint<T>(
    dir: &Path,
    d: &Device,
    read: impl FnOnce(&mut Loader<'_>) -> Result<T>,
) -> Result<T> {
    let index: Index = if dir.join("model.safetensors.index.json").is_file() {
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?
    } else {
        let file = File::open(dir.join("model.safetensors"))?;
        let map = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&map)?;
        Index {
            weight_map: tensors
                .names()
                .into_iter()
                .map(|name| (name.to_owned(), "model.safetensors".to_owned()))
                .collect(),
        }
    };
    let mut maps: BTreeMap<String, Mmap> = BTreeMap::new();
    for name in index.weight_map.values() {
        ensure!(
            Path::new(name)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe shard path {name}"
        );
        if !maps.contains_key(name) {
            let file = File::open(dir.join(name)).with_context(|| format!("open shard {name}"))?;
            maps.insert(name.clone(), unsafe { MmapOptions::new().map(&file)? });
        }
    }
    let tensors = maps
        .iter()
        .map(|(name, map)| Ok((name.clone(), SafeTensors::deserialize(map)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let staging = unsafe { d.ctx.alloc_pinned::<bf16>(1024 * 1024)? };
    let mut l = Loader {
        tensors,
        index,
        seen: BTreeSet::new(),
        staging,
        stream: d.upload.clone(),
        bytes: 0,
    };
    let value = read(&mut l)?;
    ensure!(
        l.seen.len() == l.index.weight_map.len(),
        "checkpoint contains unexpected tensors"
    );
    drop(l);
    d.finish_upload()?;
    Ok(value)
}

fn decode_bf16(bytes: &[u8], values: &mut [bf16]) {
    for (value, bytes) in values.iter_mut().zip(bytes.as_chunks::<2>().0) {
        *value = bf16::from_bits(u16::from_le_bytes(*bytes));
    }
}

impl Weights {
    pub fn load(dir: &Path, c: &Config, d: &Device) -> Result<Self> {
        checkpoint(dir, d, |l| {
            let h = c.hidden_size;
            let f = c.intermediate_size;
            let embed = l.load(&[("model.embed_tokens.weight".into(), vec![c.vocab_size, h])])?;
            let mut layers = Vec::new();
            for i in 0..c.num_hidden_layers {
                let prefix = format!("model.layers.{i}");
                let mut one = |suffix: &str, shape: Vec<usize>| {
                    l.load(&[(format!("{prefix}.{suffix}.weight"), shape)])
                };
                let input_norm = one("input_layernorm", vec![h])?;
                let post_norm = one("post_attention_layernorm", vec![h])?;
                let o = one("self_attn.o_proj", vec![h, h])?;
                let down = one("mlp.down_proj", vec![h, f])?;
                let qkv = l.load(&[
                    (format!("{prefix}.self_attn.q_proj.weight"), vec![h, h]),
                    (
                        format!("{prefix}.self_attn.k_proj.weight"),
                        vec![c.kv_dim(), h],
                    ),
                    (
                        format!("{prefix}.self_attn.v_proj.weight"),
                        vec![c.kv_dim(), h],
                    ),
                ])?;
                let gu = l.load(&[
                    (format!("{prefix}.mlp.gate_proj.weight"), vec![f, h]),
                    (format!("{prefix}.mlp.up_proj.weight"), vec![f, h]),
                ])?;
                layers.push(Layer {
                    input_norm,
                    qkv,
                    o,
                    post_norm,
                    gu,
                    down,
                });
            }
            let norm = l.load(&[("model.norm.weight".into(), vec![h])])?;
            let head = l.load(&[("lm_head.weight".into(), vec![c.vocab_size, h])])?;
            let bytes = l.bytes;
            Ok(Self {
                embed,
                layers,
                norm,
                head,
                bytes,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::decode_bf16;
    use half::bf16;

    #[test]
    fn checkpoint_bytes_preserve_bf16_bits() {
        let bits = [0u16, 0x8000, 0x0001, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1];
        let mut bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.push(0xab);
        let mut values = vec![bf16::ZERO; bits.len()];
        decode_bf16(&bytes, &mut values);
        assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    }
}
