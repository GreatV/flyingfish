use super::Device;
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, PinnedHostSlice};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
    sync::mpsc,
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

use crate::checkpoint::{Index, MappedCheckpoint, TensorManifest};

pub(crate) struct Loader<'a> {
    source: &'a MappedCheckpoint,
    read_buffer: Vec<u8>,
    tensors: BTreeMap<String, SafeTensors<'a>>,
    index: Index,
    seen: BTreeSet<String>,
    staging: PinnedHostSlice<bf16>,
    stream: Arc<CudaStream>,
    byte_staging: Option<PinnedHostSlice<u8>>,
    verify_upload: bool,
    upload_hashers: BTreeMap<String, UploadHashWorker>,
    pub(crate) bytes: usize,
}

enum UploadHashMessage {
    Begin(String),
    Bytes(Vec<u8>),
}

struct UploadHashWorker {
    sender: mpsc::SyncSender<UploadHashMessage>,
    handle: Option<std::thread::JoinHandle<(usize, String)>>,
}

impl UploadHashWorker {
    fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel(8);
        let handle = std::thread::spawn(move || {
            let mut hash = Sha256::new();
            let mut hashed_bytes = 0usize;
            let mut tensor_open = false;
            for message in receiver {
                match message {
                    UploadHashMessage::Begin(name) => {
                        if tensor_open {
                            hash.update([0]);
                        }
                        hash.update(name.as_bytes());
                        hash.update([0]);
                        tensor_open = true;
                    }
                    UploadHashMessage::Bytes(bytes) => {
                        hashed_bytes = hashed_bytes.saturating_add(bytes.len());
                        hash.update(&bytes);
                    }
                }
            }
            if tensor_open {
                hash.update([0]);
            }
            (hashed_bytes, format!("{:x}", hash.finalize()))
        });
        Self {
            sender,
            handle: Some(handle),
        }
    }

    fn begin(&self, name: &str) -> Result<()> {
        self.sender
            .send(UploadHashMessage::Begin(name.to_owned()))
            .context("upload hash worker exited")
    }

    fn bytes(&self, bytes: &[u8]) -> Result<()> {
        self.sender
            .send(UploadHashMessage::Bytes(bytes.to_vec()))
            .context("upload hash worker exited")
    }

    fn finish(mut self) -> Result<(usize, String)> {
        let handle = self
            .handle
            .take()
            .context("upload hash worker already finished")?;
        drop(self.sender);
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("upload hash worker panicked"))
    }
}

impl Loader<'_> {
    pub(crate) fn load(&mut self, specs: &[(String, Vec<usize>)]) -> Result<CudaSlice<bf16>> {
        let count = specs.iter().try_fold(0usize, |total, (_, shape)| {
            let n = shape.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d).context("weight shape overflow")
            })?;
            total.checked_add(n).context("weight allocation overflow")
        })?;
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
            if let Some(hasher) = self.upload_hashers.get(file) {
                hasher.begin(name)?;
            }
            for mapped_chunk in t.data().chunks(self.staging.len() * 2) {
                let chunk = if self.verify_upload {
                    let len = mapped_chunk.len();
                    self.source.read_mapped_range(
                        file,
                        mapped_chunk,
                        &mut self.read_buffer[..len],
                    )?;
                    &self.read_buffer[..len]
                } else {
                    mapped_chunk
                };
                if let Some(hasher) = self.upload_hashers.get(file) {
                    hasher.bytes(chunk)?;
                }
                let len = chunk.len() / 2;
                let buf = self.staging.as_mut_slice()?;
                decode_bf16(chunk, &mut buf[..len]);
                if self.verify_upload {
                    ensure!(
                        buf[..len].iter().all(|v| v.to_f32().is_finite()),
                        "nonfinite BF16 tensor {name}"
                    );
                }
                let mut view = dst.slice_mut(offset..offset + len);
                self.stream
                    .memcpy_htod(&self.staging.as_slice()?[..len], &mut view)?;
                self.stream.synchronize()?;
                if self.verify_upload {
                    let received = self.stream.clone_dtoh(&view)?;
                    ensure!(
                        received
                            .iter()
                            .flat_map(|v| v.to_bits().to_le_bytes())
                            .eq(chunk.iter().copied()),
                        "upload byte mismatch for {name}"
                    );
                }
                offset += len;
            }
            self.bytes = self
                .bytes
                .checked_add(t.data().len())
                .context("uploaded byte count overflow")?;
        }
        Ok(dst)
    }
}

pub(crate) fn checkpoint<T>(
    dir: &Path,
    d: &Device,
    read: impl FnOnce(&mut Loader<'_>) -> Result<T>,
) -> Result<T> {
    let mapped = MappedCheckpoint::open(dir)?;
    let (value, _) = checkpoint_mapped(&mapped, d, None, false, read)?;
    Ok(value)
}

pub(super) fn checkpoint_mapped<T>(
    mapped: &MappedCheckpoint,
    d: &Device,
    manifest: Option<&TensorManifest>,
    hash_uploads: bool,
    read: impl FnOnce(&mut Loader<'_>) -> Result<T>,
) -> Result<(T, Option<(usize, String)>)> {
    let index = mapped.index.clone();
    let tensors = mapped.views()?;
    // Verified uploads read staging on the CPU, so they need cacheable pinned memory.
    let staging = unsafe {
        if manifest.is_some() {
            d.ctx.alloc_pinned_with_flags::<bf16>(1024 * 1024, 0)?
        } else {
            d.ctx.alloc_pinned::<bf16>(1024 * 1024)?
        }
    };
    let upload_shards = if hash_uploads {
        manifest
            .map(|manifest| {
                manifest
                    .entries
                    .values()
                    .filter(|entry| entry.load)
                    .map(|entry| entry.shard.clone())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default()
    } else {
        BTreeSet::new()
    };
    let mut l = Loader {
        source: mapped,
        read_buffer: if manifest.is_some() {
            vec![0; 2 * 1024 * 1024]
        } else {
            Vec::new()
        },
        tensors,
        index,
        seen: manifest
            .map(|m| {
                m.entries
                    .iter()
                    .filter(|(_, e)| !e.load)
                    .map(|(n, _)| n.clone())
                    .collect()
            })
            .unwrap_or_default(),
        staging,
        stream: d.upload.clone(),
        byte_staging: if manifest.is_some() {
            Some(unsafe { d.ctx.alloc_pinned_with_flags::<u8>(2 * 1024 * 1024, 0)? })
        } else {
            None
        },
        verify_upload: manifest.is_some(),
        upload_hashers: upload_shards
            .into_iter()
            .map(|shard| (shard.clone(), UploadHashWorker::new()))
            .collect(),
        bytes: 0,
    };
    let value = read(&mut l)?;
    let upload_digest = if hash_uploads {
        let mut aggregate = Sha256::new();
        let mut hashed_bytes = 0usize;
        let upload_hashers = std::mem::take(&mut l.upload_hashers);
        for (shard, worker) in upload_hashers {
            let (count, digest) = worker.finish()?;
            hashed_bytes = hashed_bytes.saturating_add(count);
            aggregate.update(shard.as_bytes());
            aggregate.update([0]);
            aggregate.update(digest.as_bytes());
            aggregate.update([0]);
        }
        Some((hashed_bytes, format!("{:x}", aggregate.finalize())))
    } else {
        None
    };
    ensure!(
        l.seen.len() == l.index.weight_map.len(),
        "checkpoint contains unexpected tensors"
    );
    drop(l);
    d.finish_upload()?;
    Ok((value, upload_digest))
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

impl Loader<'_> {
    pub(super) fn load_fp8(&mut self, name: &str, shape: &[usize]) -> Result<CudaSlice<u8>> {
        let file = self
            .index
            .weight_map
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let t = self.tensors[file].tensor(name)?;
        ensure!(
            t.dtype() == Dtype::F8_E4M3 && t.shape() == shape,
            "FP8 dtype/shape mismatch: {name}"
        );
        ensure!(
            self.seen.insert(name.into()),
            "tensor {name} consumed twice"
        );
        if let Some(hasher) = self.upload_hashers.get(file) {
            hasher.begin(name)?;
        }
        let staging = self
            .byte_staging
            .as_mut()
            .context("FP8 upload requires audited manifest")?;
        let mut dst = unsafe { self.stream.alloc::<u8>(t.data().len())? };
        for (i, mapped_chunk) in t.data().chunks(staging.len()).enumerate() {
            let len = mapped_chunk.len();
            self.source.read_mapped_range(
                file,
                mapped_chunk,
                &mut staging.as_mut_slice()?[..len],
            )?;
            if let Some(hasher) = self.upload_hashers.get(file) {
                hasher.bytes(&staging.as_slice()?[..len])?;
            }
            ensure!(
                staging.as_slice()?[..len].iter().all(|c| c & 0x7f != 0x7f),
                "nonfinite FP8 code in {name}"
            );
            let start = i * staging.len();
            let mut view = dst.slice_mut(start..start + len);
            self.stream
                .memcpy_htod(&staging.as_slice()?[..len], &mut view)?;
            self.stream.synchronize()?;
            let received = self.stream.clone_dtoh(&view)?;
            ensure!(
                received.as_slice() == &staging.as_slice()?[..len],
                "FP8 upload byte mismatch for {name}"
            );
        }

        self.bytes = self
            .bytes
            .checked_add(t.data().len())
            .context("uploaded byte count overflow")?;
        Ok(dst)
    }
}

pub(crate) enum StoredTensor {
    Bf16(CudaSlice<bf16>),
    Fp8(CudaSlice<u8>),
}
pub(crate) struct WeightCatalog {
    pub tensors: BTreeMap<String, StoredTensor>,
    pub manifest: TensorManifest,
    pub bytes: usize,
    pub upload_content_sha256: Option<String>,
    pub upload_hashed_bytes: Option<usize>,
    pub load_wall_ms: f64,
}
impl WeightCatalog {
    /// Move shared global tensors into the existing Engine weight owner.
    pub fn take_globals(&mut self) -> Result<Weights> {
        let mut take = |name: &str| match self.tensors.remove(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 global tensor {name}"),
        };
        Ok(Weights {
            embed: take("model.language_model.embed_tokens.weight")?,
            norm: take("model.language_model.norm.weight")?,
            head: take("lm_head.weight")?,
            layers: Vec::new(),
            bytes: self.bytes,
        })
    }
    pub fn load(dir: &Path, spec: &crate::model_spec::ModelSpec, d: &Device) -> Result<Self> {
        Self::load_inner(dir, spec, d, crate::checkpoint::LoadPolicy::TextOnly, false)
    }

    pub fn load_for_prefix_cache(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
    ) -> Result<Self> {
        Self::load_inner(dir, spec, d, crate::checkpoint::LoadPolicy::TextOnly, true)
    }
    pub fn load_with_policy(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
        policy: crate::checkpoint::LoadPolicy,
    ) -> Result<Self> {
        Self::load_inner(dir, spec, d, policy, false)
    }

    fn load_inner(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
        policy: crate::checkpoint::LoadPolicy,
        hash_uploads: bool,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        let mapped = MappedCheckpoint::open(dir)?;
        let manifest = mapped.q38_manifest_with_policy(spec, policy)?;
        let ((tensors, bytes), upload_digest) =
            checkpoint_mapped(&mapped, d, Some(&manifest), hash_uploads, |loader| {
                let mut tensors = BTreeMap::new();
                for (name, entry) in &manifest.entries {
                    if !entry.load {
                        continue;
                    }
                    let value = match entry.dtype {
                        Dtype::BF16 => {
                            StoredTensor::Bf16(loader.load(&[(name.clone(), entry.shape.clone())])?)
                        }
                        Dtype::F8_E4M3 => StoredTensor::Fp8(loader.load_fp8(name, &entry.shape)?),
                        _ => anyhow::bail!("unsupported loaded dtype for {name}"),
                    };
                    tensors.insert(name.clone(), value);
                }
                Ok((tensors, loader.bytes))
            })?;
        ensure!(
            bytes == manifest.resident_bytes,
            "resident/upload byte count mismatch"
        );
        Ok(Self {
            tensors,
            manifest,
            bytes,
            upload_content_sha256: upload_digest.as_ref().map(|x| x.1.clone()),
            upload_hashed_bytes: upload_digest.map(|x| x.0),
            load_wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
    pub fn bf16(&self, name: &str) -> Result<&CudaSlice<bf16>> {
        match self.tensors.get(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 tensor {name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use sha2::{Digest, Sha256};

    #[test]
    fn checkpoint_bytes_preserve_bf16_bits() {
        let bits = [0u16, 0x8000, 0x0001, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1];
        let mut bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.push(0xab);
        let mut values = vec![bf16::ZERO; bits.len()];
        decode_bf16(&bytes, &mut values);
        assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    }

    #[test]
    fn upload_hash_worker_hashes_exact_tensor_bytes() {
        let worker = UploadHashWorker::new();
        worker.begin("tensor.a").unwrap();
        worker.bytes(b"abc").unwrap();
        worker.begin("tensor.b").unwrap();
        worker.bytes(b"de").unwrap();
        let (count, digest) = worker.finish().unwrap();
        let mut expected = Sha256::new();
        expected.update(b"tensor.a");
        expected.update([0]);
        expected.update(b"abc");
        expected.update([0]);
        expected.update(b"tensor.b");
        expected.update([0]);
        expected.update(b"de");
        expected.update([0]);
        assert_eq!(count, 5);
        assert_eq!(digest, format!("{:x}", expected.finalize()));
    }
}
