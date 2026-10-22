use super::Device;
use crate::{
    config::Config,
    experts::{
        arena::{HostBlocks, SlotLayout},
        catalogue::ExpertGeometry,
        plan::Capacity,
    },
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, sys};
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

/// Stable native expert slots, persistent staging and host data plus spare blocks.
pub struct ExpertStorage {
    layout: SlotLayout,
    down_offset: u64,
    device: CudaSlice<bf16>,
    host: Option<Arc<HostBlocks>>,
    context: Arc<CudaContext>,
    registered: bool,
}

impl ExpertStorage {
    pub fn new(
        stream: &Arc<CudaStream>,
        upload: &Arc<CudaStream>,
        geometry: ExpertGeometry,
        capacity: &Capacity,
    ) -> Result<Self> {
        ensure!(
            Arc::ptr_eq(stream.context(), upload.context()),
            "expert allocation streams have different contexts"
        );
        let down_offset = capacity.layout.down_offset();
        let layout = SlotLayout::new(geometry, capacity)?;
        let bytes = layout.allocation_bytes()?;
        let host = (layout.host_blocks() > 0)
            .then(|| {
                HostBlocks::new(
                    layout.host_blocks(),
                    usize::try_from(capacity.slot_bytes() / 2)?,
                )
            })
            .transpose()?;
        let allocation = stream
            .alloc_zeros(bytes / size_of::<bf16>())
            .with_context(|| format!("allocate native expert slots and staging: bytes={bytes}"))?;
        upload.wait(&stream.record_event(None)?)?;
        Ok(Self {
            layout,
            down_offset,
            device: allocation,
            host,
            context: stream.context().clone(),
            registered: false,
        })
    }

    pub fn layout(&self) -> &SlotLayout {
        &self.layout
    }

    pub fn host(&self) -> Option<&Arc<HostBlocks>> {
        self.host.as_ref()
    }

    fn native_pointers(&self, range: std::ops::Range<usize>) -> Result<(u64, u64)> {
        ensure!(
            range.end <= self.device.len() * size_of::<bf16>(),
            "native expert range {range:?} exceeds allocation bytes={}",
            self.device.len() * size_of::<bf16>()
        );
        let (base, _read) = self.device.device_ptr(self.device.stream());
        let gate_up = base
            .checked_add(range.start as u64)
            .context("native expert address overflow")?;
        let down = gate_up
            .checked_add(self.down_offset)
            .context("native down address overflow")?;
        Ok((gate_up, down))
    }

    /// Native gate-up/down addresses; both address the full slot for FP8.
    pub fn slot(&self, layer: u32, rank: u32) -> Result<(u64, u64)> {
        self.native_pointers(self.layout.resident_slot(layer, rank)?)
    }

    /// Native gate-up and down device addresses of one persistent staging slot.
    pub fn staging(&self, slot: u32) -> Result<(u64, u64)> {
        self.native_pointers(self.layout.staging_slot(slot)?)
    }

    /// The ledger must evict and claim every slot in this range before writes begin.
    pub fn borrowed_tail(&self, slots: u32) -> Result<(u64, usize)> {
        let range = self.layout.borrowed_tail(slots)?;
        let (base, _read) = self.device.device_ptr(self.device.stream());
        Ok((
            base.checked_add(range.start as u64)
                .context("borrowed span address overflow")?,
            range.len(),
        ))
    }

    /// Setup calls this only for a resource-profiled pinned host plan.
    pub fn register_host(&mut self) -> Result<()> {
        ensure!(!self.registered, "expert host arena is already registered");
        let host = self
            .host
            .as_ref()
            .context("cannot register an empty host expert arena")?;
        let (address, bytes) = host.host_span();
        self.context
            .bind_to_thread()
            .context("bind expert host registration context")?;
        // SAFETY: the exact owned allocation remains alive until unregister after all device uses.
        unsafe { sys::cuMemHostRegister_v2(address as *mut std::ffi::c_void, bytes, 0).result() }
            .with_context(|| format!("register expert host arena: bytes={bytes}"))?;
        self.registered = true;
        Ok(())
    }

    /// Completes all context readers and copies before releasing page-lock registration.
    #[cfg(test)]
    fn finish(&mut self) -> Result<()> {
        self.context
            .synchronize()
            .context("finish expert storage readers and copies")?;
        if self.registered {
            let (address, bytes) = self
                .host
                .as_ref()
                .context("registered expert host arena missing")?
                .host_span();
            // SAFETY: context synchronization completed every device use of the live allocation.
            unsafe { sys::cuMemHostUnregister(address as *mut std::ffi::c_void).result() }
                .with_context(|| format!("unregister expert host arena: bytes={bytes}"))?;
            self.registered = false;
        }
        self.context
            .check_err()
            .context("expert storage deferred CUDA error")?;
        Ok(())
    }
}

impl Drop for ExpertStorage {
    fn drop(&mut self) {
        self.context.record_err(self.context.synchronize());
        if self.registered {
            self.context.record_err(self.context.bind_to_thread());
            if let Some(host) = &self.host {
                let (address, _) = host.host_span();
                // SAFETY: the host allocation remains owned and context uses have been synchronized.
                self.context.record_err(unsafe {
                    sys::cuMemHostUnregister(address as *mut std::ffi::c_void).result()
                });
            }
        }
    }
}

use crate::checkpoint::{Index, MappedCheckpoint, TensorEntry, TensorManifest};

fn upload_entry(entry: &TensorEntry) -> bool {
    entry.load && !entry.is_routed_expert()
}

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
    pub linear_format: crate::model_spec::LinearFormat,
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
        let embed_name = if self
            .tensors
            .contains_key("model.language_model.embed_tokens.weight")
        {
            "model.language_model.embed_tokens.weight"
        } else {
            "model.embed_tokens.weight"
        };
        let norm_name = if self
            .tensors
            .contains_key("model.language_model.norm.weight")
        {
            "model.language_model.norm.weight"
        } else {
            "model.norm.weight"
        };
        let mut take = |name: &str| match self.tensors.remove(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 global tensor {name}"),
        };
        Ok(Weights {
            embed: take(embed_name)?,
            norm: take(norm_name)?,
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
        let manifest = mapped.manifest(spec, policy)?;
        let upload_bytes = manifest
            .entries
            .values()
            .filter(|entry| upload_entry(entry))
            .try_fold(0usize, |sum, entry| {
                sum.checked_add(entry.bytes)
                    .context("resident upload byte count overflow")
            })?;
        ensure!(
            !hash_uploads || upload_bytes == manifest.resident_bytes,
            "prefix-cache identity does not cover deferred expert bytes: uploaded={upload_bytes} selected={}",
            manifest.resident_bytes
        );
        let ((tensors, bytes), upload_digest) =
            checkpoint_mapped(&mapped, d, Some(&manifest), hash_uploads, |loader| {
                let mut tensors = BTreeMap::new();
                for (name, entry) in &manifest.entries {
                    if !entry.load {
                        continue;
                    }
                    if !upload_entry(entry) {
                        ensure!(
                            loader.seen.insert(name.clone()),
                            "expert tensor delegated twice: name={name}"
                        );
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
            bytes == upload_bytes,
            "resident/upload byte count mismatch: uploaded={bytes} planned={upload_bytes} selected={}",
            manifest.resident_bytes
        );
        Ok(Self {
            linear_format: spec.linear_format,
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

    fn expert_storage_case(geometry: crate::experts::catalogue::ExpertGeometry) -> Result<()> {
        use crate::experts::plan::Capacity;
        use cudarc::driver::result;
        let expert_layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?;
        let slot_bytes = expert_layout.slot_bytes();
        let mut counts = vec![1; geometry.layers as usize];
        counts[0] += 1;
        let capacity = Capacity {
            layout: expert_layout,
            span_rows: 8,
            wave_rows: 2,
            device_expert_bytes: u64::from(geometry.layers + 1) * slot_bytes,
            host_expert_bytes: 3 * slot_bytes,
            slots_per_layer: counts.clone(),
            host_blocks: 2,
            spare_blocks: 1,
            disk_bytes: 0,
            worker_count: 1,
            disk_workers: 0,
            prefill_staging_slots: 2,
            pinned: true,
        };
        let device = crate::backend::cuda::Device::new(0)?;
        let mut storage = ExpertStorage::new(&device.stream, &device.upload, geometry, &capacity)?;
        let host = storage.host().context("host expert arena missing")?.clone();
        let elements = slot_bytes as usize / size_of::<bf16>();
        let pattern = |i: usize, block: u32| {
            bf16::from_bits(0x3c00 | ((i * 17 + block as usize * 59) % 251) as u16)
        };
        for block in 0..host.blocks() {
            let mut claim = host.claim(block)?;
            for (index, value) in claim.slice()?.iter_mut().enumerate() {
                *value = pattern(index, block);
            }
        }
        storage.register_host()?;
        let (registered, registered_bytes) = host.host_span();
        assert_eq!(registered_bytes, 3 * slot_bytes as usize);
        let mut flags = u32::MAX;
        unsafe { sys::cuMemHostGetFlags(&mut flags, registered as *mut std::ffi::c_void).result() }
            .context("read expert host registration flags")?;
        let unified_addressing = device
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)?
            != 0;
        let allowed_host_flags = if unified_addressing {
            sys::CU_MEMHOSTALLOC_DEVICEMAP
        } else {
            0
        };
        ensure!(
            flags & !allowed_host_flags == 0,
            "expert host registration flags={flags} allowed={allowed_host_flags} unified_addressing={unified_addressing} portable_bit={} writecombined_bit={}",
            flags & sys::CU_MEMHOSTALLOC_PORTABLE,
            flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED
        );
        println!(
            "PASS expert_storage_registration flags={flags} unified_addressing={unified_addressing} portable=false writecombined=false"
        );
        let upload = &device.upload;
        let stream = &device.stream;
        let (first, first_down) = storage.slot(0, 0)?;
        assert_eq!(first_down - first, capacity.layout.down_offset());
        let lease = host.lease(&[0])?;
        // SAFETY: the registered block is leased through completion and the destination owns a full slot.
        unsafe { result::memcpy_htod_async(first, lease.read(0)?, upload.cu_stream())? };
        let ready = upload.record_event(None)?;
        assert!(host.claim(0).is_err());
        stream.wait(&ready)?;
        let first_range = storage.layout().resident_slot(0, 0)?;
        let observed = stream.clone_dtoh(
            &storage
                .device
                .slice(first_range.start / 2..first_range.end / 2),
        )?;
        assert!(
            observed
                .iter()
                .enumerate()
                .all(|(i, &value)| value == pattern(i, 0))
        );
        ready.synchronize()?;
        drop(lease);
        let claim = host.claim(2)?;
        // SAFETY: this exact registered destination block is writer-claimed through the copy event.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                claim.pointer()? as *mut std::ffi::c_void,
                first,
                slot_bytes as usize,
                upload.cu_stream(),
            )
            .result()?
        };
        let written = upload.record_event(None)?;
        assert!(host.lease(&[2]).is_err());
        written.synchronize()?;
        drop(claim);
        let lease = host.lease(&[2])?;
        assert!(
            lease
                .read(2)?
                .iter()
                .enumerate()
                .all(|(i, &value)| value == pattern(i, 0))
        );
        drop(lease);
        let staged = storage.staging(1)?;
        let lease = host.lease(&[1])?;
        // SAFETY: the source lease and both full-slot destination ranges remain live through completion.
        unsafe {
            result::memcpy_htod_async(staged.0, lease.read(1)?, upload.cu_stream())?;
            result::memcpy_htod_async(storage.slot(0, 1)?.0, lease.read(1)?, upload.cu_stream())?;
        }
        let ready = upload.record_event(None)?;
        stream.wait(&ready)?;
        let mut ledger = crate::experts::plan::TierLedger::new(
            SlotLayout::new(geometry, &capacity)?,
            0,
            &(0..geometry.layers)
                .map(|layer| if layer == 0 { vec![0, 1] } else { vec![0] })
                .collect::<Vec<_>>(),
        )?;
        assert_eq!(ledger.borrow_tail(1, 0)?.experts, vec![(0, 1)]);
        let (borrowed, borrowed_bytes) = storage.borrowed_tail(1)?;
        assert_eq!(borrowed, storage.slot(0, 1)?.0);
        assert_eq!(borrowed_bytes, slot_bytes as usize);
        // SAFETY: the ledger evicted the tail, the stream waits for its last writer, and the storage is live.
        unsafe { result::memset_d8_async(borrowed, 0x5a, borrowed_bytes, stream.cu_stream())? };
        let staged_range = storage.layout().staging_slot(1)?;
        let staged_values = stream.clone_dtoh(
            &storage
                .device
                .slice(staged_range.start / 2..staged_range.end / 2),
        )?;
        assert!(
            staged_values
                .iter()
                .enumerate()
                .all(|(i, &value)| value == pattern(i, 1))
        );
        let first_values = stream.clone_dtoh(
            &storage
                .device
                .slice(first_range.start / 2..first_range.end / 2),
        )?;
        assert_eq!(first_values, observed);
        let tail = storage.layout().borrowed_tail(1)?;
        let tail_values = stream.clone_dtoh(&storage.device.slice(tail.start / 2..tail.end / 2))?;
        assert!(tail_values.iter().all(|value| value.to_bits() == 0x5a5a));
        ready.synchronize()?;
        drop(lease);
        storage.finish()?;
        storage.register_host()?;
        storage.finish()?;
        let lease = host.lease(&[0])?;
        let mut ring = unsafe { device.ctx.alloc_pinned::<bf16>(elements)? };
        ring.as_mut_slice()?.copy_from_slice(lease.read(0)?);
        drop(lease);
        let mut destination = storage
            .device
            .slice_mut(first_range.start / 2..first_range.end / 2);
        upload.memcpy_htod(&ring, &mut destination)?;
        stream.wait(&upload.record_event(None)?)?;
        let pageable_values = stream.clone_dtoh(&destination)?;
        assert_eq!(pageable_values, observed);
        storage.finish()?;
        device.ctx.check_err()?;
        println!(
            "PASS expert_storage_case layers={} hidden={} intermediate={} native_slot_bytes={slot_bytes} pinned=true pageable_staging=true tail_staging_disjoint=true leases=true re_register=true",
            geometry.layers, geometry.hidden, geometry.intermediate
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn expert_storage_smoke_gpu() -> Result<()> {
        expert_storage_case(crate::experts::catalogue::ExpertGeometry {
            first_layer: 0,
            layers: 3,
            experts: 8,
            hidden: 128,
            intermediate: 64,
            top_k: 2,
        })
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn expert_storage_native_gpu() -> Result<()> {
        let path = std::env::var_os("FF_MODEL")
            .context("FF_MODEL is required for native expert geometry")?;
        let spec = crate::model_spec::ModelSpec::read(std::path::Path::new(&path))?;
        expert_storage_case(crate::experts::catalogue::ExpertGeometry::from_model(
            &spec,
        )?)
    }

    #[test]
    fn expert_upload_partition_preserves_other_three_dimensional_weights() {
        let entries = [
            ("mlp.experts.gate_up_proj", vec![2, 4, 4], true, Dtype::BF16),
            ("mlp.experts.down_proj", vec![2, 4, 2], true, Dtype::BF16),
            (
                "mlp.experts.2.gate_proj.weight",
                vec![4, 4],
                true,
                Dtype::F8_E4M3,
            ),
            (
                "mlp.experts.2.gate_proj.weight_scale",
                vec![4, 1],
                true,
                Dtype::BF16,
            ),
            (
                "linear_attn.conv1d.weight",
                vec![4, 1, 4],
                true,
                Dtype::BF16,
            ),
            ("mlp.gate.weight", vec![2, 4], true, Dtype::BF16),
            ("vision.projection.weight", vec![4, 4], false, Dtype::BF16),
        ]
        .map(|(role, shape, load, dtype)| TensorEntry {
            dtype,
            bytes: shape.iter().product::<usize>() * dtype.bitsize() / 8,
            shape,
            shard: "native.safetensors".into(),
            load,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: Some(0),
            role: role.into(),
            scale_name: None,
        });
        let uploaded = entries
            .iter()
            .filter(|entry| upload_entry(entry))
            .collect::<Vec<_>>();
        assert_eq!(uploaded.iter().map(|entry| entry.bytes).sum::<usize>(), 48);
        assert_eq!(
            uploaded
                .iter()
                .map(|entry| entry.role.as_str())
                .collect::<Vec<_>>(),
            ["linear_attn.conv1d.weight", "mlp.gate.weight"]
        );
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.load && !upload_entry(entry))
                .map(|entry| entry.bytes)
                .sum::<usize>(),
            120
        );
    }
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
