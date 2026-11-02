use super::Device;
use crate::backend::gpu::{CudaContext, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice};
use crate::config::Config;
use crate::experts::{
    arena::{HostBlocks, SlotLayout},
    catalogue::ExpertGeometry,
    plan::Capacity,
};
use anyhow::{Context, Result, ensure};
#[cfg(feature = "cuda")]
use cudarc::driver::sys;
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
    sync::mpsc,
    time::Instant,
};

const UPLOAD_CHUNK_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Default, serde::Serialize)]
struct UploadTimings {
    init_ms: f64,
    device_alloc_ms: f64,
    source_read_ms: f64,
    hash_ms: f64,
    decode_ms: f64,
    finite_check_ms: f64,
    h2d_ms: f64,
    dtoh_ms: f64,
    compare_ms: f64,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub struct Layer {
    pub input_norm: Option<CudaSlice<bf16>>,
    pub qkv: CudaSlice<bf16>,
    pub o: CudaSlice<bf16>,
    pub post_norm: Option<CudaSlice<bf16>>,
    pub gu: CudaSlice<bf16>,
    pub down: CudaSlice<bf16>,
}

pub struct Weights {
    pub embed: Option<CudaSlice<bf16>>,
    pub layers: Vec<Layer>,
    pub norm: Option<CudaSlice<bf16>>,
    pub head: CudaSlice<bf16>,
    pub bytes: usize,
}

pub(crate) struct LoopvlVisionLayer {
    pub input_norm: CudaSlice<bf16>,
    pub post_norm: CudaSlice<bf16>,
    pub q_norm: CudaSlice<bf16>,
    pub k_norm: CudaSlice<bf16>,
    pub qkv: CudaSlice<bf16>,
    pub o: CudaSlice<bf16>,
    pub gu: CudaSlice<bf16>,
    pub down: CudaSlice<bf16>,
}

pub(crate) struct LoopvlVisionWeights {
    pub layers: Vec<LoopvlVisionLayer>,
    pub patch_weight: CudaSlice<bf16>,
    pub patch_bias: CudaSlice<bf16>,
    pub norm: CudaSlice<bf16>,
    pub fc1_weight: CudaSlice<u8>,
    pub fc1_bias: CudaSlice<u8>,
    pub fc2_weight: CudaSlice<u8>,
    pub fc2_bias: CudaSlice<u8>,
    pub visual_scale: CudaSlice<u8>,
    pub gate_visual: CudaSlice<u8>,
    pub gate_query: CudaSlice<u8>,
    pub gate_out: CudaSlice<u8>,
    pub gate_bias: CudaSlice<u8>,
    pub raw_alpha: CudaSlice<u8>,
    pub bytes: usize,
}

/// Stable native expert slots, persistent staging and host data plus spare blocks.
pub struct ExpertStorage {
    layout: SlotLayout,
    down_offset: u64,
    device: CudaSlice<bf16>,
    peer_device: Option<CudaSlice<bf16>>,
    host: Option<Arc<HostBlocks>>,
    context: Arc<CudaContext>,
    peer_context: Option<Arc<CudaContext>>,
    registered: bool,
}

pub(crate) struct CohortStorage {
    storage: ExpertStorage,
    policy: crate::experts::runtime::Residency,
    catalogue: crate::experts::catalogue::ExpertCatalogue,
    disk: crate::experts::disk::DiskReader,
    backing: Option<crate::experts::arena::BlockLease>,
    claims: Vec<crate::experts::arena::WriteClaim>,
    ready: crate::backend::gpu::CudaEvent,
    readers: crate::backend::gpu::CudaEvent,
    upload: Arc<CudaStream>,
    host_blocks: u32,
    staging: u32,
}

impl CohortStorage {
    pub(crate) fn load(
        device: &Device,
        directory: &Path,
        catalogue: crate::experts::catalogue::ExpertCatalogue,
        capacity: &Capacity,
        admission: crate::experts::plan::Admission,
    ) -> Result<Self> {
        use crate::experts::{
            disk::DiskReader,
            runtime::{LoadInputs, Residency},
        };
        let geometry = catalogue.geometry();
        let initial: Vec<Vec<u32>> = capacity
            .slots_per_layer
            .iter()
            .map(|&count| (0..count).collect())
            .collect();
        let mut storage =
            ExpertStorage::new(&device.stream, &device.upload, None, geometry, capacity)?;
        if storage.host().is_some() {
            storage.register_host()?;
        }
        let disk = DiskReader::open(&catalogue, directory, 2)?;
        let (policy, loaded) = Residency::load(LoadInputs {
            geometry,
            capacity: capacity.clone(),
            admission,
            max_rows: capacity.wave_rows,
            initial: &initial,
            host: storage.host(),
            catalogue: &catalogue,
            reader: &disk,
        })?;
        let mut staging =
            super::copy::pinned_zeros::<bf16>(&device.ctx, catalogue.slot_bytes() as usize / 2)?;
        for (layer, experts) in initial.iter().enumerate() {
            for &expert in experts {
                let slot =
                    disk.read_expert(layer as u32, expert, &catalogue, catalogue.slot_bytes())?;
                staging.as_mut_slice()?.copy_from_slice(&disk.values(slot)?);
                // The pinned slot and destination remain live until upload synchronization.
                unsafe {
                    super::copy::htod_async(
                        storage.slot(layer as u32, expert)?.0,
                        staging.as_slice()?,
                        &device.upload,
                    )?;
                }
                device.upload.synchronize()?;
            }
        }
        let backing = storage
            .host()
            .map(|arena| arena.lease(&(0..capacity.host_blocks).collect::<Vec<_>>()))
            .transpose()?;
        let ready = device.ctx.new_event(None)?;
        ready.record(&device.upload)?;
        device.stream.wait(&ready)?;
        println!(
            "COHORT_STORAGE gpu_slots={} host_blocks={} staging_slots={} slot_bytes={} host_bytes={} disk_bytes={}",
            capacity.slots_total(),
            capacity.host_blocks,
            capacity.prefill_staging_slots,
            catalogue.slot_bytes(),
            loaded.host_bytes,
            capacity.disk_bytes
        );
        Ok(Self {
            storage,
            policy,
            catalogue,
            disk,
            backing,
            claims: Vec::with_capacity(capacity.prefill_staging_slots as usize),
            ready,
            readers: device.ctx.new_event(None)?,
            upload: device.upload.clone(),
            host_blocks: capacity.host_blocks,
            staging: capacity.prefill_staging_slots,
        })
    }

    pub(crate) fn tier(&self, layer: u32, expert: u32) -> Result<crate::experts::plan::Tier> {
        self.policy.ledger().tier(layer, expert)
    }

    pub(crate) fn resident(&self, layer: u32, expert: u32) -> Result<(u64, u64)> {
        ensure!(
            self.tier(layer, expert)? == crate::experts::plan::Tier::Gpu,
            "cohort expert is not resident"
        );
        self.storage.slot(layer, expert)
    }

    pub(crate) fn wave_capacity(&self) -> u32 {
        self.staging
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn verify_tiers(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        use crate::experts::plan::Tier;
        let geometry = self.catalogue.geometry();
        let mut observed = vec![bf16::ZERO; self.catalogue.slot_bytes() as usize / 2];
        let mut controls = 0;
        for tier in [Tier::Gpu, Tier::Host, Tier::Disk] {
            let mut selected = None;
            for layer in 0..geometry.layers {
                for expert in 0..geometry.experts {
                    if self.tier(layer, expert)? == tier {
                        selected = Some((layer, expert));
                        break;
                    }
                }
                if selected.is_some() {
                    break;
                }
            }
            let (layer, expert) = selected.context("Runner FAIL forced tier unavailable")?;
            let address = if tier == Tier::Gpu {
                self.resident(layer, expert)?.0
            } else {
                self.stage(stream, layer, &[expert])?;
                self.staged(0)?.0
            };
            // The allocation and output span remain live through synchronization.
            unsafe { super::copy::dtoh_async(&mut observed, address, stream)? };
            stream.synchronize()?;
            let slot = self.disk.read_expert(
                layer,
                expert,
                &self.catalogue,
                self.catalogue.slot_bytes(),
            )?;
            let expected = self.disk.values(slot)?;
            let exact = observed.as_slice() == expected.values();
            let slot = self.disk.read_expert(
                layer,
                (expert + 1) % geometry.experts,
                &self.catalogue,
                self.catalogue.slot_bytes(),
            )?;
            let wrong = self.disk.values(slot)?;
            let detected = observed.as_slice() != wrong.values();
            controls += usize::from(detected);
            println!(
                "CLOVER_TIER {} tier={tier:?} layer={layer} expert={expert} exact={exact} wrong_expert_control={detected} bytes={}",
                if exact && detected { "PASS" } else { "DEFECT" },
                self.catalogue.slot_bytes()
            );
            ensure!(exact && detected, "Clover tier copy/index defect");
        }
        ensure!(controls == 3, "Clover tier controls incomplete");
        Ok(())
    }

    pub(crate) fn staged(&self, slot: u32) -> Result<(u64, u64)> {
        self.storage.staging(slot)
    }

    pub(crate) fn stage(
        &mut self,
        stream: &Arc<CudaStream>,
        layer: u32,
        experts: &[u32],
    ) -> Result<[u64; 2]> {
        use crate::experts::plan::Tier;
        ensure!(
            !experts.is_empty() && experts.len() <= self.staging as usize,
            "cohort wave outside staging capacity"
        );
        self.ready.synchronize()?;
        self.claims.clear();
        self.readers.record(stream)?;
        self.upload.wait(&self.readers)?;
        let mut bytes = [0u64; 2];
        for (slot, &expert) in experts.iter().enumerate() {
            let destination = self.storage.staging(slot as u32)?.0;
            match self.tier(layer, expert)? {
                Tier::Gpu => anyhow::bail!("resident expert requested as a staging miss"),
                Tier::Host => {
                    let block = self.policy.host_block(layer, expert)?;
                    let values = self
                        .backing
                        .as_ref()
                        .context("cohort host backing missing")?
                        .read(block)?;
                    // The immutable host lease spans every upload and GPU reader.
                    unsafe {
                        super::copy::htod_async(destination, values, &self.upload)?;
                    }
                    bytes[0] += self.catalogue.slot_bytes();
                }
                Tier::Disk => {
                    let arena = self
                        .storage
                        .host()
                        .context("cohort disk staging has no host arena")?;
                    let mut claim = arena.claim(self.host_blocks + slot as u32)?;
                    let disk_slot = self.disk.read_expert(
                        layer,
                        expert,
                        &self.catalogue,
                        self.catalogue.slot_bytes(),
                    )?;
                    claim
                        .slice()?
                        .copy_from_slice(&self.disk.values(disk_slot)?);
                    // The writer claim remains owned until ready synchronizes the upload.
                    unsafe {
                        super::copy::htod_async(destination, claim.slice()?, &self.upload)?;
                    }
                    self.claims.push(claim);
                    bytes[1] += self.catalogue.slot_bytes();
                }
            }
        }
        self.ready.record(&self.upload)?;
        stream.wait(&self.ready)?;
        Ok(bytes)
    }
}

impl ExpertStorage {
    pub fn new(
        stream: &Arc<CudaStream>,
        upload: &Arc<CudaStream>,
        peer: Option<(&Arc<CudaStream>, &Arc<CudaStream>)>,
        geometry: ExpertGeometry,
        capacity: &Capacity,
    ) -> Result<Self> {
        ensure!(
            Arc::ptr_eq(stream.context(), upload.context()),
            "expert allocation streams have different contexts"
        );
        let down_offset = capacity.layout.down_offset();
        let layout = SlotLayout::new(geometry, capacity)?;
        let gpu0_bytes = layout.allocation_bytes()?;
        let peer_bytes = layout.peer_resident_bytes();
        #[cfg(feature = "rocm")]
        ensure!(
            peer.is_none() && peer_bytes == 0,
            "HIP expert peer storage is not implemented"
        );
        let host = (layout.host_blocks() > 0)
            .then(|| {
                HostBlocks::new(
                    layout.host_blocks(),
                    usize::try_from(capacity.slot_bytes() / 2)?,
                )
            })
            .transpose()?;
        let allocation = stream
            .alloc_zeros(gpu0_bytes / size_of::<bf16>())
            .with_context(|| {
                format!("allocate native expert slots and staging on GPU 0: bytes={gpu0_bytes}")
            })?;
        upload.wait(&stream.record_event(None)?)?;
        let (peer_device, peer_context) = if peer_bytes > 0 {
            let (peer_stream, peer_upload) =
                peer.context("peer expert bytes planned without a peer device stream")?;
            ensure!(
                Arc::ptr_eq(peer_stream.context(), peer_upload.context()),
                "peer expert allocation streams have different contexts"
            );
            let peer_alloc = peer_stream
                .alloc_zeros(peer_bytes / size_of::<bf16>())
                .with_context(|| {
                    format!("allocate native expert slots on peer GPU: bytes={peer_bytes}")
                })?;
            peer_upload.wait(&peer_stream.record_event(None)?)?;
            (Some(peer_alloc), Some(peer_stream.context().clone()))
        } else {
            (None, None)
        };
        Ok(Self {
            layout,
            down_offset,
            device: allocation,
            peer_device,
            host,
            context: stream.context().clone(),
            peer_context,
            registered: false,
        })
    }

    pub fn layout(&self) -> &SlotLayout {
        &self.layout
    }

    pub fn host(&self) -> Option<&Arc<HostBlocks>> {
        self.host.as_ref()
    }

    #[cfg(all(test, feature = "cuda"))]
    pub fn has_peer(&self) -> bool {
        self.peer_device.is_some()
    }

    fn native_pointers(&self, range: std::ops::Range<usize>) -> Result<(u64, u64)> {
        let peer_bytes = self.layout.peer_resident_bytes();
        let (base, local_start) = if range.end <= peer_bytes {
            let peer = self
                .peer_device
                .as_ref()
                .context("peer expert slot requested without peer allocation")?;
            ensure!(
                range.end <= peer.len() * size_of::<bf16>(),
                "peer expert range {range:?} exceeds peer allocation bytes={}",
                peer.len() * size_of::<bf16>()
            );
            let (base, _read) = peer.device_ptr(peer.stream());
            (base, range.start)
        } else {
            ensure!(
                range.start >= peer_bytes,
                "expert range {range:?} straddles peer boundary {peer_bytes}"
            );
            let local_start = range.start - peer_bytes;
            let local_end = range.end - peer_bytes;
            ensure!(
                local_end <= self.device.len() * size_of::<bf16>(),
                "native expert range {local_start}..{local_end} exceeds GPU 0 allocation bytes={}",
                self.device.len() * size_of::<bf16>()
            );
            let (base, _read) = self.device.device_ptr(self.device.stream());
            (base, local_start)
        };
        let gate_up = base
            .checked_add(local_start as u64)
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
        let peer_bytes = self.layout.peer_resident_bytes();
        let local_start = range
            .start
            .checked_sub(peer_bytes)
            .context("borrowed tail lies inside peer expert range")?;
        let (base, _read) = self.device.device_ptr(self.device.stream());
        Ok((
            base.checked_add(local_start as u64)
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
        unsafe {
            #[cfg(feature = "cuda")]
            sys::cuMemHostRegister_v2(address as *mut std::ffi::c_void, bytes, 0)
                .result()
                .with_context(|| format!("register expert host arena: bytes={bytes}"))?;
            #[cfg(feature = "rocm")]
            self.context
                .register_host(address, bytes, false)
                .with_context(|| format!("register expert host arena: bytes={bytes}"))?;
        }
        self.registered = true;
        Ok(())
    }

    /// Completes all context readers and copies before releasing page-lock registration.
    #[cfg(all(test, feature = "cuda"))]
    fn finish(&mut self) -> Result<()> {
        self.context
            .synchronize()
            .context("finish expert storage readers and copies")?;
        if let Some(peer_ctx) = &self.peer_context {
            peer_ctx
                .synchronize()
                .context("finish peer expert storage readers and copies")?;
        }
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
        if let Some(peer_ctx) = &self.peer_context {
            peer_ctx
                .check_err()
                .context("peer expert storage deferred CUDA error")?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for ExpertStorage {
    fn drop(&mut self) {
        self.context.record_err(self.context.synchronize());
        if let Some(peer_ctx) = &self.peer_context {
            peer_ctx.record_err(peer_ctx.synchronize());
        }
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

#[cfg(feature = "rocm")]
impl Drop for ExpertStorage {
    fn drop(&mut self) {
        crate::backend::hip::drop_result(
            self.context.synchronize().context("finish expert storage"),
        );
        if let Some(peer) = &self.peer_context {
            crate::backend::hip::drop_result(
                peer.synchronize().context("finish peer expert storage"),
            );
        }
        if self.registered
            && let Some(host) = &self.host
        {
            let (address, _) = host.host_span();
            crate::backend::hip::drop_result(
                unsafe { self.context.unregister_host(address) }
                    .context("unregister expert host storage"),
            );
        }
    }
}

use crate::checkpoint::TensorEntry;
use crate::checkpoint::{Index, MappedCheckpoint, TensorManifest};

fn upload_entry(entry: &TensorEntry) -> bool {
    entry.is_non_expert_loaded()
}

pub(crate) struct Loader<'a> {
    source: &'a MappedCheckpoint,
    read_buffer: Vec<u8>,
    tensors: BTreeMap<String, SafeTensors<'a>>,
    index: Index,
    seen: BTreeSet<String>,
    staging: PinnedHostSlice<bf16>,
    staging_ready: crate::backend::gpu::CudaEvent,
    stream: Arc<CudaStream>,
    #[cfg(feature = "rocm")]
    compute: Arc<CudaStream>,
    byte_staging: Option<PinnedHostSlice<u8>>,
    verify_upload: bool,
    upload_hashers: BTreeMap<String, UploadHashWorker>,
    timings: UploadTimings,
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
    /// Marks one manifest tensor consumed without uploading it.
    pub(super) fn defer(&mut self, name: &str) -> Result<()> {
        ensure!(
            self.seen.insert(name.to_owned()),
            "tensor {name} deferred twice"
        );
        Ok(())
    }
    pub(crate) fn load(&mut self, specs: &[(String, Vec<usize>)]) -> Result<CudaSlice<bf16>> {
        let count = specs.iter().try_fold(0usize, |total, (_, shape)| {
            let n = shape.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d).context("weight shape overflow")
            })?;
            total.checked_add(n).context("weight allocation overflow")
        })?;
        let phase = Instant::now();
        let mut dst = unsafe { self.stream.alloc::<bf16>(count)? };
        self.timings.device_alloc_ms += elapsed_ms(phase);
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
            for mapped_chunk in t.data().chunks(UPLOAD_CHUNK_BYTES) {
                let phase = Instant::now();
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
                self.timings.source_read_ms += elapsed_ms(phase);
                let phase = Instant::now();
                if let Some(hasher) = self.upload_hashers.get(file) {
                    hasher.bytes(chunk)?;
                }
                self.timings.hash_ms += elapsed_ms(phase);
                let len = chunk.len() / 2;
                let phase = Instant::now();
                self.staging_ready.synchronize()?;
                let buf = self.staging.as_mut_slice()?;
                decode_bf16(chunk, &mut buf[..len]);
                self.timings.decode_ms += elapsed_ms(phase);
                let phase = Instant::now();
                if self.verify_upload {
                    ensure!(
                        buf[..len].iter().all(|v| v.to_bits() & 0x7f80 != 0x7f80),
                        "nonfinite BF16 tensor {name}"
                    );
                }
                self.timings.finite_check_ms += elapsed_ms(phase);
                let phase = Instant::now();
                let mut view = dst.slice_mut(offset..offset + len);
                self.stream
                    .memcpy_htod(&self.staging.as_slice()?[..len], &mut view)?;
                self.staging_ready.record(&self.stream)?;
                self.timings.h2d_ms += elapsed_ms(phase);
                let phase = Instant::now();
                if self.verify_upload {
                    let received = self.stream.clone_dtoh(&view)?;
                    self.timings.dtoh_ms += elapsed_ms(phase);
                    let phase = Instant::now();
                    ensure!(
                        received.iter().zip(chunk.as_chunks::<2>().0).all(
                            |(received, expected)| {
                                received.to_bits() == u16::from_le_bytes(*expected)
                            }
                        ),
                        "upload byte mismatch for {name}"
                    );
                    self.timings.compare_ms += elapsed_ms(phase);
                }
                offset += len;
            }
            self.bytes = self
                .bytes
                .checked_add(t.data().len())
                .context("uploaded byte count overflow")?;
        }
        #[cfg(feature = "rocm")]
        dst.handoff(&self.compute)?;
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
    let init_started = Instant::now();
    let index = mapped.index.clone();
    let tensors = mapped.views()?;
    // Verified uploads read staging on the CPU, so they need cacheable pinned memory.
    let staging = unsafe {
        if manifest.is_some() {
            d.ctx
                .alloc_pinned_with_flags::<bf16>(UPLOAD_CHUNK_BYTES / 2, 0)?
        } else {
            #[cfg(feature = "cuda")]
            {
                d.ctx.alloc_pinned::<bf16>(UPLOAD_CHUNK_BYTES / 2)?
            }
            #[cfg(feature = "rocm")]
            {
                d.ctx
                    .alloc_pinned_with_flags::<bf16>(UPLOAD_CHUNK_BYTES / 2, 0)?
            }
        }
    };
    #[cfg(feature = "cuda")]
    let staging_ready = d.ctx.new_event(Some(
        cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC,
    ))?;
    #[cfg(feature = "rocm")]
    let staging_ready = d.ctx.new_event(Some(1))?;
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
    let init_ms = elapsed_ms(init_started);
    let mut l = Loader {
        source: mapped,
        read_buffer: if manifest.is_some() {
            vec![0; UPLOAD_CHUNK_BYTES]
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
        staging_ready,
        stream: d.upload.clone(),
        #[cfg(feature = "rocm")]
        compute: d.stream.clone(),
        byte_staging: if manifest.is_some() {
            Some(unsafe { d.ctx.alloc_pinned_with_flags::<u8>(UPLOAD_CHUNK_BYTES, 0)? })
        } else {
            None
        },
        verify_upload: manifest.is_some(),
        upload_hashers: upload_shards
            .into_iter()
            .map(|shard| (shard.clone(), UploadHashWorker::new()))
            .collect(),
        timings: UploadTimings {
            init_ms,
            ..UploadTimings::default()
        },
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
    pub fn embedding(&self) -> &CudaSlice<bf16> {
        self.embed.as_ref().unwrap_or(&self.head)
    }

    pub fn load_loopvl(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
    ) -> Result<(Self, CudaSlice<bf16>)> {
        let mapped = MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let (weights, _) = checkpoint_mapped(&mapped, d, Some(&manifest), false, |l| {
            let h = spec.hidden as usize;
            let f = spec.intermediate as usize;
            let mut layers = Vec::new();
            for stack in ["L", "H"] {
                for i in 0..16 {
                    let p = format!("language_model.model.{stack}_module.layers.{i}");
                    let qkv = l
                        .load(&["q", "k", "v", "gate"].map(|role| {
                            (format!("{p}.self_attn.{role}_proj.weight"), vec![h, h])
                        }))?;
                    let gu = l.load(
                        &["gate", "up"]
                            .map(|role| (format!("{p}.mlp.{role}_proj.weight"), vec![f, h])),
                    )?;
                    layers.push(Layer {
                        input_norm: None,
                        post_norm: None,
                        qkv,
                        gu,
                        o: l.load(&[(format!("{p}.self_attn.o_proj.weight"), vec![h, h])])?,
                        down: l.load(&[(format!("{p}.mlp.down_proj.weight"), vec![h, f])])?,
                    });
                }
            }
            let embed = l.load(&[(
                "language_model.model.embed_tokens.weight".into(),
                vec![spec.vocab as usize, h],
            )])?;
            let head = l.load(&[(
                "language_model.lm_head.weight".into(),
                vec![spec.vocab as usize, h],
            )])?;
            let initial = l.load(&[("language_model.model.z_L_init".into(), vec![h])])?;
            ensure!(
                l.bytes == manifest.resident_bytes,
                "LoopVL upload byte count mismatch"
            );
            Ok((
                Self {
                    embed: Some(embed),
                    head,
                    norm: None,
                    layers,
                    bytes: l.bytes,
                },
                initial,
            ))
        })?;
        Ok(weights)
    }

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
                    input_norm: Some(input_norm),
                    qkv,
                    o,
                    post_norm: Some(post_norm),
                    gu,
                    down,
                });
            }
            let norm = l.load(&[("model.norm.weight".into(), vec![h])])?;
            let head = l.load(&[("lm_head.weight".into(), vec![c.vocab_size, h])])?;
            let bytes = l.bytes;
            Ok(Self {
                embed: Some(embed),
                layers,
                norm: Some(norm),
                head,
                bytes,
            })
        })
    }
}

impl LoopvlVisionWeights {
    pub(crate) fn load(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
    ) -> Result<Self> {
        let mapped = MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::VisionOnly)?;
        let (weights, _) = checkpoint_mapped(&mapped, d, Some(&manifest), false, |l| {
            let mut layers = Vec::new();
            for layer in 0..28 {
                let p = format!("vision_encoder.model.encoder.layers.{layer}");
                let qkv = l.load(&["q", "k", "v"].map(|role| {
                    (
                        format!("{p}.self_attn.{role}_proj.weight"),
                        if role == "q" {
                            vec![2048, 1024]
                        } else {
                            vec![1024, 1024]
                        },
                    )
                }))?;
                let gu = l.load(
                    &["gate", "up"]
                        .map(|role| (format!("{p}.mlp.{role}_proj.weight"), vec![3072, 1024])),
                )?;
                layers.push(LoopvlVisionLayer {
                    input_norm: l.load(&[(format!("{p}.input_layernorm.weight"), vec![1024])])?,
                    post_norm: l
                        .load(&[(format!("{p}.post_attention_layernorm.weight"), vec![1024])])?,
                    q_norm: l.load(&[(format!("{p}.self_attn.q_norm.weight"), vec![128])])?,
                    k_norm: l.load(&[(format!("{p}.self_attn.k_norm.weight"), vec![128])])?,
                    qkv,
                    o: l.load(&[(format!("{p}.self_attn.o_proj.weight"), vec![1024, 2048])])?,
                    gu,
                    down: l.load(&[(format!("{p}.mlp.down_proj.weight"), vec![1024, 3072])])?,
                });
            }
            let f32 = |l: &mut Loader<'_>, name: &str, shape: &[usize]| {
                l.load_bytes(name, shape, Dtype::F32)
            };
            let weights = Self {
                layers,
                patch_weight: l.load(&[(
                    "vision_encoder.model.embeddings.patch_embedding.weight".into(),
                    vec![1024, 3, 14, 14],
                )])?,
                patch_bias: l.load(&[(
                    "vision_encoder.model.embeddings.patch_embedding.bias".into(),
                    vec![1024],
                )])?,
                norm: l.load(&[(
                    "vision_encoder.model.encoder.norm.weight".into(),
                    vec![1024],
                )])?,
                fc1_weight: f32(l, "projector.fc1.weight", &[1536, 1024])?,
                fc1_bias: f32(l, "projector.fc1.bias", &[1536])?,
                fc2_weight: f32(l, "projector.fc2.weight", &[1536, 1536])?,
                fc2_bias: f32(l, "projector.fc2.bias", &[1536])?,
                visual_scale: f32(l, "projector.visual_embedding_scale", &[])?,
                gate_visual: f32(
                    l,
                    "language_model.model.visual_gate.visual_proj.weight",
                    &[256, 1536],
                )?,
                gate_query: f32(
                    l,
                    "language_model.model.visual_gate.query_proj.weight",
                    &[256, 1536],
                )?,
                gate_out: f32(
                    l,
                    "language_model.model.visual_gate.out_proj.weight",
                    &[1, 256],
                )?,
                gate_bias: f32(l, "language_model.model.visual_gate.out_proj.bias", &[1])?,
                raw_alpha: f32(l, "language_model.model.visual_anchor.raw_alpha", &[2])?,
                bytes: l.bytes,
            };
            ensure!(
                weights.bytes == manifest.resident_bytes,
                "LoopVL vision upload byte count mismatch"
            );
            Ok(weights)
        })?;
        Ok(weights)
    }
}

impl Loader<'_> {
    pub(super) fn load_bytes(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: Dtype,
    ) -> Result<CudaSlice<u8>> {
        let file = self
            .index
            .weight_map
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let t = self.tensors[file].tensor(name)?;
        ensure!(
            t.dtype() == dtype && t.shape() == shape,
            "native dtype/shape mismatch: {name}"
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
            .context("native byte upload requires audited manifest")?;
        let mut dst = unsafe { self.stream.alloc::<u8>(t.data().len())? };
        for (i, mapped_chunk) in t.data().chunks(UPLOAD_CHUNK_BYTES).enumerate() {
            let len = mapped_chunk.len();
            self.source.read_mapped_range(
                file,
                mapped_chunk,
                &mut staging.as_mut_slice()?[..len],
            )?;
            if let Some(hasher) = self.upload_hashers.get(file) {
                hasher.bytes(&staging.as_slice()?[..len])?;
            }
            let valid = match dtype {
                Dtype::F8_E4M3 => staging.as_slice()?[..len].iter().all(|c| c & 0x7f != 0x7f),
                Dtype::F8_E8M0 => staging.as_slice()?[..len].iter().all(|c| *c != 0xff),
                Dtype::BF16 => {
                    ensure!(len % 2 == 0, "unaligned BF16 chunk in {name}");
                    staging.as_slice()?[..len]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|c| bf16::from_bits(u16::from_le_bytes(*c)).is_finite())
                }
                Dtype::F32 => {
                    ensure!(len % 4 == 0, "unaligned F32 chunk in {name}");
                    staging.as_slice()?[..len]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|c| f32::from_le_bytes(*c).is_finite())
                }
                // Packed E2M1 nibble pairs; every byte is a valid code pair.
                Dtype::I8 => true,
                _ => anyhow::bail!("unsupported native byte dtype {dtype:?} for {name}"),
            };
            ensure!(valid, "nonfinite native value in {name}");
            let start = i * staging.len();
            let mut view = dst.slice_mut(start..start + len);
            self.stream
                .memcpy_htod(&staging.as_slice()?[..len], &mut view)?;
            let received = self.stream.clone_dtoh(&view)?;
            ensure!(
                received.as_slice() == &staging.as_slice()?[..len],
                "native upload byte mismatch for {name}"
            );
        }

        self.bytes = self
            .bytes
            .checked_add(t.data().len())
            .context("uploaded byte count overflow")?;
        #[cfg(feature = "rocm")]
        dst.handoff(&self.compute)?;
        Ok(dst)
    }
}

#[cfg(feature = "cuda")]
struct DenseLayer {
    slot: Option<usize>,
    next: Option<usize>,
    next_disk: Option<usize>,
    bytes: usize,
    host: Option<PinnedHostSlice<u8>>,
}

#[cfg(feature = "cuda")]
struct DenseSlot {
    device: CudaSlice<u8>,
    ready: cudarc::driver::CudaEvent,
    retired: cudarc::driver::CudaEvent,
    retired_valid: bool,
    loaded: Option<usize>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DenseStreaming {
    layers: Vec<DenseLayer>,
    slots: Vec<DenseSlot>,
    disk: Option<super::engine::Reader>,
    events: Option<super::engine::Events>,
    read_ahead: super::engine::ReadAhead,
    next_disk_read: Option<usize>,
    upload: Arc<CudaStream>,
    pub plan: crate::prefill::DensePlacement,
    #[cfg(test)]
    skip_reclaim: bool,
}

#[cfg(feature = "cuda")]
impl Drop for DenseStreaming {
    fn drop(&mut self) {
        let context = self.upload.context();
        let _ = self.read_ahead.receive_reader(&mut self.disk, true);
        context.record_err(context.synchronize());
    }
}

#[cfg(feature = "cuda")]
impl DenseStreaming {
    pub(crate) fn new(
        dir: &Path,
        manifest: &TensorManifest,
        plan: crate::prefill::DensePlacement,
        d: &Device,
    ) -> Result<Self> {
        use crate::experts::{catalogue::NativeRange, disk::DiskReader};
        let mut layers: Vec<_> = plan
            .slots
            .iter()
            .map(|&slot| DenseLayer {
                slot,
                next: None,
                next_disk: None,
                bytes: 0,
                host: None,
            })
            .collect();
        let mut ranges = vec![Vec::new(); layers.len()];
        for entry in manifest
            .entries
            .values()
            .filter(|entry| entry.is_linear_projection() && entry.layer.is_some())
        {
            let layer_index = entry.layer.context("dense layer missing")? as usize;
            let layer = &mut layers[layer_index];
            if layer.slot.is_none() {
                continue;
            }
            let target = u64::try_from(layer.bytes)?;
            layer.bytes = layer
                .bytes
                .checked_add(entry.bytes)
                .context("dense layer bytes overflow")?;
            if !plan.host[layer_index] {
                let offset = entry
                    .data_start
                    .checked_add(entry.data_offsets[0])
                    .context("dense source offset overflow")?;
                ranges[layer_index].push((
                    target,
                    NativeRange {
                        shard: Arc::from(entry.shard.as_str()),
                        offset,
                        len: entry.bytes as u64,
                    },
                ));
            }
        }
        let first_disk = (0..layers.len()).find(|&i| layers[i].slot.is_some() && !plan.host[i]);
        let mut next = None;
        let mut next_disk = first_disk;
        for (index, layer) in layers.iter_mut().enumerate().rev() {
            layer.next = next;
            layer.next_disk = next_disk;
            if layer.slot.is_some() {
                next = Some(index);
                if !plan.host[index] {
                    next_disk = Some(index);
                }
            }
        }
        for (index, layer) in layers.iter_mut().enumerate() {
            if plan.host[index] {
                layer.host = Some(unsafe { d.ctx.alloc_pinned_with_flags::<u8>(layer.bytes, 0)? });
            }
        }
        let (disk, events, read_ahead) = if plan.disk_bytes > 0 {
            let disk = DiskReader::open_ranges(&ranges, plan.slot_bytes, dir, 2)?;
            let reader =
                super::engine::Reader::new_ranges(d, disk, ranges, UPLOAD_CHUNK_BYTES as u64)?;
            let events = super::engine::Events::new(&d.ctx, 8)?;
            let read_ahead = super::engine::ReadAhead::new(2, 2, 1)?;
            (Some(reader), Some(events), read_ahead)
        } else {
            (None, None, super::engine::ReadAhead::new(2, 0, 1)?)
        };
        let slots = (0..2)
            .map(|_| {
                Ok(DenseSlot {
                    device: d.upload.alloc_zeros::<u8>(plan.slot_bytes)?,
                    ready: d
                        .ctx
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
                    retired: d
                        .ctx
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
                    retired_valid: false,
                    loaded: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            slots,
            disk,
            events,
            read_ahead,
            next_disk_read: first_disk,
            upload: d.upload.clone(),
            plan,
            #[cfg(test)]
            skip_reclaim: false,
        })
    }

    fn defer(
        &mut self,
        loader: &mut Loader<'_>,
        name: &str,
        entry: &TensorEntry,
        layer: usize,
        offset: usize,
    ) -> Result<()> {
        loader.defer(name)?;
        let binding = self
            .layers
            .get_mut(layer)
            .context("dense layer outside plan")?;
        let end = offset
            .checked_add(entry.bytes)
            .context("dense tensor offset overflow")?;
        ensure!(end <= binding.bytes, "dense tensor exceeds layer slot");
        let file = loader
            .index
            .weight_map
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let t = loader.tensors[file].tensor(name)?;
        ensure!(
            t.dtype() == Dtype::BF16,
            "tensor {name}: expected BF16, found {:?}",
            t.dtype()
        );
        ensure!(
            t.shape() == entry.shape,
            "tensor {name}: expected {:?}, found {:?}",
            entry.shape,
            t.shape()
        );
        ensure!(
            t.data().len() == entry.bytes,
            "deferred tensor {name} length {} differs from {}",
            t.data().len(),
            entry.bytes
        );
        let mut cursor = offset;
        for mapped_chunk in t.data().chunks(UPLOAD_CHUNK_BYTES) {
            let len = mapped_chunk.len();
            let chunk = if loader.verify_upload {
                loader.source.read_mapped_range(
                    file,
                    mapped_chunk,
                    &mut loader.read_buffer[..len],
                )?;
                &loader.read_buffer[..len]
            } else {
                mapped_chunk
            };
            ensure!(len % 2 == 0, "unaligned BF16 chunk in {name}");
            ensure!(
                chunk
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|c| bf16::from_bits(u16::from_le_bytes(*c)).is_finite()),
                "nonfinite BF16 tensor {name}"
            );
            if let Some(host) = binding.host.as_mut() {
                host.as_mut_slice()?[cursor..cursor + len].copy_from_slice(chunk);
            }
            cursor += len;
        }
        Ok(())
    }

    fn prime(&mut self) -> Result<()> {
        if let Some(first) = self.layers.iter().position(|layer| layer.slot == Some(0)) {
            self.prefetch(first)?;
        }
        if let Some(second) = self.layers.iter().position(|layer| layer.slot == Some(1)) {
            self.prefetch(second)?;
        }
        self.pump_disk()
    }

    fn pump_disk(&mut self) -> Result<()> {
        if self.plan.disk_bytes == 0 {
            return Ok(());
        }
        self.read_ahead.receive_reader(&mut self.disk, false)?;
        if !self.read_ahead.is_pending() {
            self.read_ahead.reclaim_reader(&mut self.disk, false)?;
        }
        while let Some(next_disk) = self.next_disk_read {
            if !self.read_ahead.try_enqueue(next_disk as u32, 0)? {
                break;
            }
            self.next_disk_read = self.layers[next_disk].next_disk;
        }
        self.read_ahead
            .drive_reader(&mut self.disk, self.upload.context())
    }

    #[cfg(test)]
    pub(crate) fn flush_ring_counts(&mut self) -> Result<(u64, u64)> {
        self.upload.synchronize()?;
        self.read_ahead.receive_reader(&mut self.disk, true)?;
        while self.read_ahead.reclaim_reader(&mut self.disk, true)? {}
        let counts = (self.read_ahead.claims, self.read_ahead.reclaims);
        self.pump_disk()?;
        Ok(counts)
    }

    pub(crate) fn pointer(&self, layer: usize, offset: usize, bytes: usize) -> Result<u64> {
        let binding = self.layers.get(layer).context("dense layer outside plan")?;
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= binding.bytes),
            "dense tensor exceeds layer slot"
        );
        let slot = &self.slots[binding.slot.context("dense layer is resident")?];
        Ok(slot.device.device_ptr(&self.upload).0 + offset as u64)
    }

    fn prefetch(&mut self, index: usize) -> Result<()> {
        let Some(slot_index) = self.layers[index].slot else {
            return Ok(());
        };
        if self.slots[slot_index].loaded == Some(index) {
            return self.pump_disk();
        }
        if self.slots[slot_index].retired_valid {
            self.upload.wait(&self.slots[slot_index].retired)?;
        }
        let layer_bytes = self.layers[index].bytes;
        let pointer = self.slots[slot_index].device.device_ptr(&self.upload).0;
        if let Some(host) = &self.layers[index].host {
            let source = &host.as_slice()?[..layer_bytes];
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    pointer,
                    source,
                    self.upload.cu_stream(),
                )?;
            }
        } else {
            if self.next_disk_read == Some(index) {
                self.next_disk_read = self.layers[index].next_disk;
            }
            let ring = self.read_ahead.demand_item(
                &mut self.disk,
                self.upload.context(),
                index as u32,
                0,
            )?;
            let reader = self
                .disk
                .as_mut()
                .context("dense disk reader missing after demand")?;
            let values = reader.disk.values(ring)?;
            // SAFETY: the ring slot holds at least `layer_bytes` initialized bytes for the lease's lifetime.
            let raw =
                unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, layer_bytes) };
            unsafe {
                cudarc::driver::result::memcpy_htod_async(pointer, raw, self.upload.cu_stream())?;
            }
            #[cfg(test)]
            if self.skip_reclaim {
                drop(values);
                self.read_ahead.release(ring)?;
            } else {
                let copied = self
                    .events
                    .as_mut()
                    .context("dense disk events missing")?
                    .record(&self.upload)?;
                values.copied(copied.id)?;
                reader.copied[ring as usize] = Some(copied);
            }
            #[cfg(not(test))]
            {
                let copied = self
                    .events
                    .as_mut()
                    .context("dense disk events missing")?
                    .record(&self.upload)?;
                values.copied(copied.id)?;
                reader.copied[ring as usize] = Some(copied);
            }
        }
        self.slots[slot_index].ready.record(&self.upload)?;
        self.slots[slot_index].loaded = Some(index);
        self.pump_disk()
    }

    pub(crate) fn begin(&mut self, layer: usize, stream: &Arc<CudaStream>) -> Result<()> {
        self.prefetch(layer)?;
        if let Some(slot) = self.layers[layer].slot {
            stream.wait(&self.slots[slot].ready)?;
        }
        Ok(())
    }

    pub(crate) fn end(&mut self, layer: usize, stream: &Arc<CudaStream>) -> Result<()> {
        if let Some(slot) = self.layers[layer].slot {
            self.slots[slot].retired.record(stream)?;
            self.slots[slot].retired_valid = true;
            if let Some(next) = self.layers[layer].next {
                self.prefetch(next)?;
            }
        }
        Ok(())
    }
}

pub(crate) enum StoredTensor {
    Bf16(CudaSlice<bf16>),
    Bf16Pair {
        buffer: Arc<CudaSlice<bf16>>,
        offset: usize,
        len: usize,
    },
    Bytes(CudaSlice<u8>),
    #[cfg(feature = "cuda")]
    Slot {
        pointer: u64,
        bytes: usize,
    },
}
impl StoredTensor {
    fn contiguous_projection_pair(a: &TensorEntry, b: &TensorEntry) -> bool {
        a.load
            && b.load
            && upload_entry(a)
            && upload_entry(b)
            && !a.native_byte_storage()
            && !b.native_byte_storage()
            && a.dtype == Dtype::BF16
            && b.dtype == Dtype::BF16
            && a.layer.is_some()
            && a.layer == b.layer
            && a.shard == b.shard
            && a.data_start == b.data_start
            && a.data_offsets[1] == b.data_offsets[0]
            && a.shape.len() == 2
            && b.shape.len() == 2
            && a.shape[1] == b.shape[1]
            && matches!(
                (a.role.as_str(), b.role.as_str()),
                ("mlp.gate_proj.weight", "mlp.up_proj.weight")
                    | ("self_attn.q_proj.weight", "self_attn.v_proj.weight")
            )
    }

    fn load_pair(
        loader: &mut Loader<'_>,
        first: (&String, &TensorEntry),
        second: (&String, &TensorEntry),
    ) -> Result<(Self, Self)> {
        let len_a = first.1.shape.iter().product::<usize>();
        let len_b = second.1.shape.iter().product::<usize>();
        let buffer = Arc::new(loader.load(&[
            (first.0.clone(), first.1.shape.clone()),
            (second.0.clone(), second.1.shape.clone()),
        ])?);
        Ok((
            Self::Bf16Pair {
                buffer: Arc::clone(&buffer),
                offset: 0,
                len: len_a,
            },
            Self::Bf16Pair {
                buffer,
                offset: len_a,
                len: len_b,
            },
        ))
    }

    fn load(loader: &mut Loader<'_>, name: &str, entry: &TensorEntry) -> Result<Self> {
        if entry.native_byte_storage() {
            Ok(Self::Bytes(loader.load_bytes(
                name,
                &entry.shape,
                entry.dtype,
            )?))
        } else {
            ensure!(
                entry.dtype == Dtype::BF16,
                "unsupported loaded dtype for {name}"
            );
            Ok(Self::Bf16(
                loader.load(&[(name.to_owned(), entry.shape.clone())])?,
            ))
        }
    }
}

pub(crate) struct WeightCatalog {
    #[cfg(feature = "cuda")]
    pub dense: Option<DenseStreaming>,
    pub linear_format: crate::model_spec::LinearFormat,
    pub tensors: BTreeMap<String, StoredTensor>,
    pub manifest: TensorManifest,
    pub bytes: usize,
    #[cfg(feature = "cuda")]
    pub upload_content_sha256: Option<String>,
    pub upload_hashed_bytes: Option<usize>,
    pub load_wall_ms: f64,
}
impl WeightCatalog {
    #[cfg(feature = "cuda")]
    fn dense_host_fixed(
        layers: &[usize],
        ranges: &[(usize, u64, u64, u64)],
        budget: usize,
        available_host: usize,
    ) -> Result<(usize, crate::prefill::DensePlacement)> {
        use crate::prefill::DensePlacement;
        let mut layer_staging = vec![0usize; layers.len()];
        for &(layer, offset, len, alignment) in ranges {
            let staging =
                crate::experts::disk::DiskReader::aligned_staging_bytes(offset, len, alignment)?;
            let slot = layer_staging
                .get_mut(layer)
                .context("dense staging layer out of bounds")?;
            *slot = (*slot).max(staging);
        }
        ensure!(
            !layers.is_empty() && layer_staging.iter().all(|&n| n > 0),
            "invalid dense layer staging extent"
        );
        let preliminary = DensePlacement::new(layers, budget, 0)?;
        let base_reserve = 4usize << 30;
        if available_host >= base_reserve {
            let host_only = DensePlacement::new(layers, budget, available_host - base_reserve)?;
            if host_only.disk_bytes == 0 {
                return Ok((base_reserve, host_only));
            }
        }
        let mut disk_staging = 0usize;
        for _ in 0..=layers.len() {
            let host_fixed = preliminary
                .slot_bytes
                .checked_mul(2)
                .and_then(|n| n.checked_add(disk_staging))
                .and_then(|n| n.checked_add(base_reserve))
                .context("dense host staging overflow")?;
            ensure!(
                available_host >= host_fixed,
                "dense pinned staging and host reserve exceed available RAM: need={host_fixed} available={available_host}"
            );
            let plan = DensePlacement::new(layers, budget, available_host - host_fixed)?;
            let required_staging = layer_staging
                .iter()
                .enumerate()
                .filter(|&(i, _)| plan.slots[i].is_some() && !plan.host[i])
                .map(|(_, &bytes)| bytes)
                .max()
                .unwrap_or(0);
            if required_staging <= disk_staging {
                return Ok((host_fixed, plan));
            }
            disk_staging = required_staging;
        }
        anyhow::bail!("dense disk staging extent did not converge")
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn load_dense(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
        transient_bytes: usize,
        media_bytes: usize,
        prefix_cache: bool,
    ) -> Result<Self> {
        use crate::prefill::DensePlacement;
        ensure!(
            spec.moe.is_none() && spec.linear_format.fp8_layout().is_none(),
            "dense layer streaming requires BF16 dense weights"
        );
        let mapped = MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let mut layers = vec![0usize; spec.layers.len()];
        let mut shard_alignments = BTreeMap::new();
        let mut ranges = Vec::new();
        let mut fixed = 0usize;
        for entry in manifest.entries.values().filter(|e| e.load) {
            if entry.is_linear_projection()
                && let Some(layer) = entry.layer
            {
                ensure!(entry.dtype == Dtype::BF16, "dense layer matrix is not BF16");
                let layer = layer as usize;
                layers[layer] = layers[layer]
                    .checked_add(entry.bytes)
                    .context("dense layer size overflow")?;
                let alignment = if let Some(&align) = shard_alignments.get(&entry.shard) {
                    align
                } else {
                    let align =
                        crate::experts::disk::SpanReader::open(&dir.join(&entry.shard), true)?
                            .alignment;
                    shard_alignments.insert(entry.shard.clone(), align);
                    align
                };
                let offset = entry
                    .data_start
                    .checked_add(entry.data_offsets[0])
                    .context("dense source offset overflow")?;
                ranges.push((layer, offset, entry.bytes as u64, alignment));
            } else {
                fixed = fixed
                    .checked_add(entry.bytes)
                    .context("dense fixed weights overflow")?;
            }
        }
        let (free, total) = d.ctx.mem_get_info()?;
        let budget =
            DensePlacement::device_budget(free, total, fixed, transient_bytes, media_bytes)?;
        let preliminary = DensePlacement::new(&layers, budget, 0)?;
        if preliminary.slot_bytes == 0 {
            return Self::load_inner(
                dir,
                spec,
                d,
                crate::checkpoint::LoadPolicy::TextOnly,
                prefix_cache,
            );
        }
        ensure!(
            !prefix_cache,
            "prefix-cache identity for streamed dense weights is not bound"
        );
        let available_host = usize::try_from(super::calibrate::host_memory(Path::new("/"))?)?;
        let (_, plan) = Self::dense_host_fixed(&layers, &ranges, budget, available_host)?;
        let memory_bus_bits = d
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let copy_engines = d
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?;
        let mut pci_name = [0i8; 32];
        unsafe {
            sys::cuDeviceGetPCIBusId(
                pci_name.as_mut_ptr(),
                pci_name.len() as i32,
                d.ctx.cu_device(),
            )
            .result()?;
        }
        let pci_name = unsafe { std::ffi::CStr::from_ptr(pci_name.as_ptr()) }.to_str()?;
        let pci = Path::new("/sys/bus/pci/devices").join(pci_name);
        let link_speed = std::fs::read_to_string(pci.join("max_link_speed"))
            .context("probe dense PCIe link speed")?;
        let link_width: usize = std::fs::read_to_string(pci.join("current_link_width"))
            .context("probe dense PCIe link width")?
            .trim()
            .parse()?;
        ensure!(
            link_width > 0 && !link_speed.trim().is_empty(),
            "invalid dense PCIe capability"
        );
        eprintln!(
            "{}",
            serde_json::json!({"dense_placement": &plan, "fixed_bytes": fixed, "transient_bytes": transient_bytes, "host_available_bytes": available_host, "copy_engines": copy_engines, "memory_bus_bits": memory_bus_bits, "sms": d.info.sms, "pcie_link_width": link_width, "pcie_max_link_speed": link_speed.trim()})
        );
        let started = Instant::now();
        let mut dense = DenseStreaming::new(dir, &manifest, plan, d)?;
        let mut offsets = vec![0; layers.len()];
        let gemma = matches!(
            spec.architecture,
            crate::model_spec::Architecture::Gemma4
                | crate::model_spec::Architecture::EmbeddingGemma2
        );
        let ((tensors, uploaded), _) =
            checkpoint_mapped(&mapped, d, Some(&manifest), false, |loader| {
                let mut tensors = BTreeMap::new();
                let mut iter = manifest.entries.iter().peekable();
                while let Some((name, entry)) = iter.next() {
                    if !entry.load {
                        continue;
                    }
                    if entry.is_linear_projection()
                        && let Some(layer) = entry.layer
                        && dense.plan.slots[layer as usize].is_some()
                    {
                        let layer = layer as usize;
                        let offset = offsets[layer];
                        let pointer = dense.pointer(layer, offset, entry.bytes)?;
                        offsets[layer] += entry.bytes;
                        dense.defer(loader, name, entry, layer, offset)?;
                        tensors.insert(
                            name.clone(),
                            StoredTensor::Slot {
                                pointer,
                                bytes: entry.bytes,
                            },
                        );
                        continue;
                    }
                    if gemma
                        && let Some(&(next_name, next_entry)) = iter.peek()
                        && StoredTensor::contiguous_projection_pair(entry, next_entry)
                        && next_entry
                            .layer
                            .is_none_or(|layer| dense.plan.slots[layer as usize].is_none())
                    {
                        iter.next();
                        let (first, second) = StoredTensor::load_pair(
                            loader,
                            (name, entry),
                            (next_name, next_entry),
                        )?;
                        tensors.insert(name.clone(), first);
                        tensors.insert(next_name.clone(), second);
                        continue;
                    }
                    tensors.insert(name.clone(), StoredTensor::load(loader, name, entry)?);
                }
                Ok((tensors, loader.bytes))
            })?;
        dense.prime()?;
        ensure!(
            uploaded == fixed + dense.plan.resident_bytes,
            "dense resident byte count mismatch"
        );
        let bytes = uploaded
            .checked_add(2 * dense.plan.slot_bytes)
            .context("dense device extent overflow")?;
        Ok(Self {
            dense: Some(dense),
            linear_format: spec.linear_format,
            tensors,
            manifest,
            bytes,
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: elapsed_ms(started),
        })
    }

    pub fn take_bf16(&mut self, name: &str, shape: &[usize]) -> Result<CudaSlice<bf16>> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .context("BF16 manifest entry missing")?;
        anyhow::ensure!(
            entry.load && entry.dtype == safetensors::Dtype::BF16 && entry.shape == shape,
            "invalid BF16 tensor {name}"
        );
        match self.tensors.remove(name) {
            Some(StoredTensor::Bf16(buffer)) => Ok(buffer),
            _ => anyhow::bail!("tensor not uploaded as BF16: {name}"),
        }
    }

    /// Move shared global tensors into the existing Engine weight owner.
    pub fn take_globals(&mut self) -> Result<Weights> {
        let embed_name = if self
            .tensors
            .contains_key("model.language_model.embed_tokens.weight")
        {
            "model.language_model.embed_tokens.weight"
        } else if self.tensors.contains_key("model.embed_tokens.weight") {
            "model.embed_tokens.weight"
        } else {
            "language_model.embed_tokens.weight"
        };
        let norm_name = if self
            .tensors
            .contains_key("model.language_model.norm.weight")
        {
            "model.language_model.norm.weight"
        } else if self.tensors.contains_key("model.norm.weight") {
            "model.norm.weight"
        } else {
            "language_model.norm.weight"
        };
        let has_head = self.tensors.contains_key("lm_head.weight");
        let mut take = |name: &str| match self.tensors.remove(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 global tensor {name}"),
        };
        let (embed, head) = if has_head {
            (Some(take(embed_name)?), take("lm_head.weight")?)
        } else {
            (None, take(embed_name)?)
        };
        Ok(Weights {
            embed,
            norm: Some(take(norm_name)?),
            head,
            layers: Vec::new(),
            bytes: self.bytes,
        })
    }
    pub fn load(dir: &Path, spec: &crate::model_spec::ModelSpec, d: &Device) -> Result<Self> {
        Self::load_inner(dir, spec, d, crate::checkpoint::LoadPolicy::TextOnly, false)
    }

    #[cfg(feature = "cuda")]
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
        let phase = Instant::now();
        let mapped = MappedCheckpoint::open(dir)?;
        let mapped_open_ms = elapsed_ms(phase);
        let phase = Instant::now();
        let manifest = mapped.manifest(spec, policy)?;
        let manifest_ms = elapsed_ms(phase);
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
        let gemma = matches!(
            spec.architecture,
            crate::model_spec::Architecture::Gemma4
                | crate::model_spec::Architecture::EmbeddingGemma2
        );
        let phase = Instant::now();
        let ((tensors, bytes, upload_timings), upload_digest) =
            checkpoint_mapped(&mapped, d, Some(&manifest), hash_uploads, |loader| {
                let mut tensors = BTreeMap::new();
                let mut iter = manifest.entries.iter().peekable();
                while let Some((name, entry)) = iter.next() {
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
                    if gemma
                        && let Some(&(next_name, next_entry)) = iter.peek()
                        && StoredTensor::contiguous_projection_pair(entry, next_entry)
                    {
                        iter.next();
                        let (first, second) = StoredTensor::load_pair(
                            loader,
                            (name, entry),
                            (next_name, next_entry),
                        )?;
                        tensors.insert(name.clone(), first);
                        tensors.insert(next_name.clone(), second);
                        continue;
                    }
                    let value = StoredTensor::load(loader, name, entry)?;
                    tensors.insert(name.clone(), value);
                }
                Ok((tensors, loader.bytes, loader.timings))
            })?;
        let checkpoint_wall_ms = elapsed_ms(phase);
        ensure!(
            bytes == upload_bytes,
            "resident/upload byte count mismatch: uploaded={bytes} planned={upload_bytes} selected={}",
            manifest.resident_bytes
        );
        let load_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "{}",
            serde_json::json!({"weight_load_phase": {
                "bytes": bytes,
                "tensors": tensors.len(),
                "load_wall_ms": load_wall_ms,
                "mapped_open_ms": mapped_open_ms,
                "manifest_ms": manifest_ms,
                "checkpoint_wall_ms": checkpoint_wall_ms,
                "chunk_bytes": UPLOAD_CHUNK_BYTES,
                "timings": upload_timings
            }})
        );
        Ok(Self {
            #[cfg(feature = "cuda")]
            dense: None,
            linear_format: spec.linear_format,
            tensors,
            manifest,
            bytes,
            #[cfg(feature = "cuda")]
            upload_content_sha256: upload_digest.as_ref().map(|x| x.1.clone()),
            upload_hashed_bytes: upload_digest.map(|x| x.0),
            load_wall_ms,
        })
    }
    pub fn bf16(&self, name: &str) -> Result<&CudaSlice<bf16>> {
        match self.tensors.get(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 tensor {name}"),
        }
    }

    pub(crate) fn matrix(
        &self,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<super::contracts::Tensor<bf16, 2>> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .context("matrix manifest entry missing")?;
        ensure!(
            entry.shape.len() >= 2,
            "matrix manifest rank differs: {name}"
        );
        let rows = u32::try_from(entry.shape[0])?;
        let width = u32::try_from(entry.shape[1..].iter().product::<usize>())?;
        match self.tensors.get(name) {
            Some(StoredTensor::Bf16(values)) => {
                super::contracts::view(values, stream, 0, [rows, width], [u64::from(width), 1])
            }
            Some(StoredTensor::Bf16Pair {
                buffer,
                offset,
                len,
            }) => {
                ensure!(
                    *len == (rows as usize) * (width as usize),
                    "paired matrix slice extent differs: {name}"
                );
                super::contracts::view(
                    buffer,
                    stream,
                    *offset,
                    [rows, width],
                    [u64::from(width), 1],
                )
            }
            _ => anyhow::bail!("missing/non-BF16 tensor {name}"),
        }
    }

    pub(crate) fn vector(
        &self,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<super::contracts::Tensor<bf16, 1>> {
        let values = self.bf16(name)?;
        super::contracts::view(values, stream, 0, [u32::try_from(values.len())?], [1])
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use half::bf16;
    use sha2::{Digest, Sha256};

    #[test]
    #[ignore = "requires CUDA and a dense streaming window"]
    fn dense_slots_preserve_native_bytes_across_host_disk_reuse_gpu() -> Result<()> {
        use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys as bs};
        use std::io::Write;
        let root = std::env::temp_dir().join(format!("ff-dense-slots-{}", std::process::id()));
        std::fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let device = Device::new(0)?;
            let shapes = [vec![257, 264], vec![15360, 3840]];
            let per_layer = shapes
                .iter()
                .map(|s| s.iter().product::<usize>() * 2)
                .sum::<usize>();
            let mut header = serde_json::Map::new();
            let mut offset = 0usize;
            let mut entries = BTreeMap::new();
            for layer in 0..4u32 {
                for (tensor, shape) in shapes.iter().enumerate() {
                    let name = format!("model.layers.{layer}.matrix{tensor}.weight");
                    let bytes = shape.iter().product::<usize>() * 2;
                    header.insert(name.clone(), serde_json::json!({"dtype":"BF16", "shape":shape, "data_offsets":[offset, offset + bytes]}));
                    entries.insert(
                        name,
                        TensorEntry {
                            dtype: Dtype::BF16,
                            shape: shape.clone(),
                            shard: "model.safetensors".into(),
                            bytes,
                            load: true,
                            data_start: 0,
                            data_offsets: [offset as u64, (offset + bytes) as u64],
                            scope: crate::checkpoint::TensorScope::Text,
                            layer: Some(layer),
                            role: format!("matrix{tensor}.weight"),
                            scale_name: None,
                        },
                    );
                    offset += bytes;
                }
            }
            let mut header = serde_json::to_vec(&header)?;
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            for entry in entries.values_mut() {
                entry.data_start = 8 + header.len() as u64;
            }
            let expected: Vec<Vec<u8>> = (0..4)
                .map(|layer| {
                    (0..per_layer / 2)
                        .flat_map(|i| {
                            (0x3c00u16 + ((i * 19 + layer * 173) % 1009) as u16).to_le_bytes()
                        })
                        .collect()
                })
                .collect();
            let mut file = std::fs::File::create(root.join("model.safetensors"))?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            for bytes in &expected {
                file.write_all(bytes)?;
            }
            file.sync_all()?;
            drop(file);
            let weight_map: BTreeMap<String, String> = entries
                .keys()
                .map(|name| (name.clone(), "model.safetensors".to_owned()))
                .collect();
            std::fs::write(
                root.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({
                    "metadata": {"total_size": per_layer * 4},
                    "weight_map": weight_map
                }))?,
            )?;
            let manifest = TensorManifest {
                entries,
                resident_bytes: per_layer * 4,
            };
            let mapped = MappedCheckpoint::open(&root)?;
            let populate = |dense: &mut DenseStreaming| -> Result<()> {
                let mut offsets = [0usize; 4];
                checkpoint_mapped(&mapped, &device, Some(&manifest), false, |loader| {
                    for (name, entry) in &manifest.entries {
                        let layer = entry.layer.context("test layer missing")? as usize;
                        let offset = offsets[layer];
                        offsets[layer] += entry.bytes;
                        dense.defer(loader, name, entry, layer, offset)?;
                    }
                    Ok(())
                })?;
                Ok(())
            };
            let shard_alignment =
                crate::experts::disk::SpanReader::open(&root.join("model.safetensors"), true)?
                    .alignment;
            let mut ranges = Vec::new();
            let mut max_staging = 0usize;
            for entry in manifest.entries.values() {
                let layer = entry.layer.context("test layer missing")? as usize;
                let offset = entry
                    .data_start
                    .checked_add(entry.data_offsets[0])
                    .context("test source offset overflow")?;
                let staging = crate::experts::disk::DiskReader::aligned_staging_bytes(
                    offset,
                    entry.bytes as u64,
                    shard_alignment,
                )?;
                max_staging = max_staging.max(staging);
                ranges.push((layer, offset, entry.bytes as u64, shard_alignment));
            }
            ensure!(
                max_staging > UPLOAD_CHUNK_BYTES,
                "test largest tensor staging {max_staging} does not exceed UPLOAD_CHUNK_BYTES"
            );
            let (host_fixed, host_only_plan) = WeightCatalog::dense_host_fixed(
                &[per_layer; 4],
                &ranges,
                2 * per_layer,
                (4 << 30) + 4 * per_layer,
            )?;
            ensure!(
                host_fixed == (4 << 30)
                    && host_only_plan.host_bytes == 4 * per_layer
                    && host_only_plan.disk_bytes == 0,
                "host-only placement reserved disk staging bytes: host_fixed={host_fixed} plan={host_only_plan:?}"
            );
            let (disk_fixed, disk_plan) = WeightCatalog::dense_host_fixed(
                &[per_layer; 4],
                &ranges,
                2 * per_layer,
                (4 << 30) + 3 * per_layer + UPLOAD_CHUNK_BYTES,
            )?;
            ensure!(
                disk_fixed == (4 << 30) + 2 * per_layer + max_staging
                    && disk_plan.host_bytes == 0
                    && disk_plan.disk_bytes == 4 * per_layer,
                "disk-spill placement did not reserve derived disk staging bytes: disk_fixed={disk_fixed} max_staging={max_staging} plan={disk_plan:?}"
            );
            let stream = &device.stream;
            let mut outputs = (0..4)
                .map(|_| stream.alloc_zeros::<u8>(per_layer))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let blas = CudaBlas::new(stream.clone())?;
            let (n, k) = (shapes[1][0], shapes[1][1]);
            let input = stream.clone_htod(
                &(0..18 * k)
                    .map(|i| bf16::from_f32(((i * 37 % 997) as f32 - 498.0) / 512.0))
                    .collect::<Vec<_>>(),
            )?;
            let mut products = (0..4)
                .map(|_| stream.alloc_zeros::<bf16>(18 * n))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let product = |weights: &CudaSlice<u8>, output: &mut CudaSlice<bf16>| -> Result<()> {
                let weight = unsafe { weights.transmute::<bf16>(per_layer / 2) }
                    .context("dense test matrix extent")?;
                let weight = weight.slice(shapes[0].iter().product::<usize>()..);
                for (start, rows) in [(0, 1), (1, 17)] {
                    let config = GemmConfig {
                        transa: bs::cublasOperation_t::CUBLAS_OP_T,
                        transb: bs::cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: rows as i32,
                        k: k as i32,
                        alpha: bf16::ONE,
                        lda: k as i32,
                        ldb: k as i32,
                        beta: bf16::ZERO,
                        ldc: n as i32,
                    };
                    unsafe {
                        blas.gemm(
                            config,
                            &weight,
                            &input.slice(start * k..(start + rows) * k),
                            &mut output.slice_mut(start * n..(start + rows) * n),
                        )?;
                    }
                }
                Ok(())
            };
            let mut resident_products = Vec::new();
            for (layer, (values, output)) in expected.iter().zip(&mut outputs).enumerate() {
                let resident = stream.clone_htod(values)?;
                stream.memcpy_dtod(&resident, output)?;
                ensure!(
                    stream.clone_dtoh(output)? == *values,
                    "resident native byte baseline differs"
                );
                product(&resident, &mut products[layer])?;
                let values = stream.clone_dtoh(&products[layer])?;
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite resident product"
                );
                resident_products.push(values);
            }
            println!("PASS dense_resident layers=4 layer_bytes={per_layer} byte_mismatches=0");
            for host_layers in [0, 2, 4] {
                let plan = crate::prefill::DensePlacement::new(
                    &[per_layer; 4],
                    2 * per_layer,
                    host_layers * per_layer,
                )?;
                ensure!(
                    plan.slots.iter().all(Option::is_some),
                    "streaming gate has a resident layer"
                );
                let mut dense = DenseStreaming::new(&root, &manifest, plan, &device)?;
                if host_layers == 4 {
                    ensure!(
                        dense.disk.is_none() && dense.events.is_none(),
                        "host-only placement allocated disk ring resources"
                    );
                }
                populate(&mut dense)?;
                for slot in &mut dense.slots {
                    slot.device = device.upload.clone_htod(&vec![0xcdu8; per_layer + 64])?;
                }
                dense.prime()?;
                ensure!(
                    dense.pointer(0, per_layer, 2).is_err(),
                    "slot overflow control was admitted"
                );
                for repeat in 0..3 {
                    for (layer, output) in outputs.iter_mut().enumerate() {
                        dense.begin(layer, stream)?;
                        let source = dense.pointer(layer, 0, per_layer)?;
                        let target = output.device_ptr(stream).0;
                        unsafe {
                            cudarc::driver::result::memcpy_dtod_async(
                                target,
                                source,
                                per_layer,
                                stream.cu_stream(),
                            )?;
                        }
                        let slot = dense.plan.slots[layer].context("streamed test slot missing")?;
                        product(&dense.slots[slot].device, &mut products[layer])?;
                        dense.end(layer, stream)?;
                    }
                    for (layer, output) in outputs.iter().enumerate() {
                        let actual = stream.clone_dtoh(output)?;
                        ensure!(
                            actual == expected[layer],
                            "dense slot bytes differ: host_layers={host_layers} repeat={repeat} layer={layer}"
                        );
                        ensure!(
                            actual != expected[(layer + 2) % 4],
                            "wrong-slot-generation control is not discriminating"
                        );
                        let actual = stream.clone_dtoh(&products[layer])?;
                        ensure!(
                            actual == resident_products[layer],
                            "dense resident/streamed products differ: host_layers={host_layers} repeat={repeat} layer={layer}"
                        );
                        ensure!(
                            actual != resident_products[(layer + 2) % 4],
                            "wrong-generation product control is not discriminating"
                        );
                    }
                    device.upload.synchronize()?;
                    for slot in &dense.slots {
                        let guard = stream.clone_dtoh(&slot.device.slice(per_layer..))?;
                        ensure!(
                            guard == [0xcd; 64],
                            "dense upload overwrote unused slot capacity"
                        );
                    }
                }
                let (claims, reclaims) = dense.flush_ring_counts()?;
                if host_layers == 4 {
                    ensure!(
                        claims == 0 && reclaims == 0,
                        "host-only placement claimed disk ring slots: claims={claims} reclaims={reclaims}"
                    );
                } else {
                    ensure!(
                        claims == reclaims && claims >= (3 * (4 - host_layers)) as u64,
                        "disk ring claim/reclaim mismatch: host_layers={host_layers} claims={claims} reclaims={reclaims}"
                    );
                }
                println!(
                    "PASS dense_slots host_layers={host_layers} disk_layers={} layers=4 repeats=3 slot_bytes={per_layer} claims={claims} reclaims={reclaims} byte_mismatches=0 product_rows=1,17 product_bits=0 wrong_generation=detected bounds=detected tail_guards=preserved",
                    4 - host_layers
                );
            }
            unsafe extern "C" fn wait_for_gate(user_data: *mut std::ffi::c_void) {
                let gate = unsafe { &*(user_data as *const std::sync::atomic::AtomicU32) };
                while gate.load(std::sync::atomic::Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
            }
            for skip_reclaim in [false, true] {
                let plan = crate::prefill::DensePlacement::new(&[per_layer; 4], 2 * per_layer, 0)?;
                let mut dense = DenseStreaming::new(&root, &manifest, plan, &device)?;
                populate(&mut dense)?;
                dense.skip_reclaim = skip_reclaim;
                let gate = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                unsafe {
                    sys::cuLaunchHostFunc(
                        dense.upload.cu_stream(),
                        Some(wait_for_gate),
                        std::sync::Arc::as_ptr(&gate) as *mut std::ffi::c_void,
                    )
                    .result()
                    .context("hold dense upload stream")?;
                }
                dense.prefetch(0)?;
                dense.prefetch(1)?;
                if skip_reclaim {
                    dense.read_ahead.receive_reader(&mut dense.disk, true)?;
                }
                gate.store(1, std::sync::atomic::Ordering::Release);
                dense.upload.synchronize()?;
                let actual = stream.clone_dtoh(&dense.slots[0].device)?;
                if skip_reclaim {
                    ensure!(
                        actual == expected[2] && actual != expected[0],
                        "reclaim mutant did not overwrite slot 0 with layer 2 while upload was held"
                    );
                } else {
                    ensure!(
                        actual == expected[0],
                        "guarded ring slot 0 did not preserve layer 0 bytes"
                    );
                }
            }
            println!(
                "PASS dense_reclaim_control host_only_no_staging=true guarded_exact=true skip_reclaim_overwritten_with_layer2=true"
            );
            Ok(())
        })();
        std::fs::remove_dir_all(&root)?;
        result
    }

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
            peer_expert_bytes: 0,
            gpu0_layers: geometry.layers,
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
        let mut storage =
            ExpertStorage::new(&device.stream, &device.upload, None, geometry, &capacity)?;
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
    #[ignore = "requires CUDA"]
    fn native_f32_upload_gpu() -> Result<()> {
        use std::io::Write;
        let root = std::env::temp_dir().join(format!("ff-f32-upload-{}", std::process::id()));
        let result = (|| {
            std::fs::create_dir_all(&root)?;
            let mut header = serde_json::to_vec(&serde_json::json!({
                "scale": {"dtype":"F32", "shape":[2,2], "data_offsets":[0,16]}
            }))?;
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let data: Vec<u8> = [1.0f32, -2.0, 0.5, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let mut file = std::fs::File::create(root.join("model.safetensors"))?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            file.write_all(&data)?;
            drop(file);
            std::fs::write(
                root.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({
                    "metadata":{"total_size":16},
                    "weight_map":{"scale":"model.safetensors"}
                }))?,
            )?;
            let data_start = 8 + header.len() as u64;
            let entry = TensorEntry {
                dtype: Dtype::F32,
                shape: vec![2, 2],
                shard: "model.safetensors".into(),
                bytes: 16,
                load: true,
                data_start,
                data_offsets: [0, 16],
                scope: crate::checkpoint::TensorScope::Text,
                layer: None,
                role: "scale".into(),
                scale_name: None,
            };
            let manifest = TensorManifest {
                entries: [("scale".to_owned(), entry)].into(),
                resident_bytes: 16,
            };
            let mapped = MappedCheckpoint::open(&root)?;
            let device = crate::backend::cuda::Device::new(0)?;
            let (uploaded, _) =
                checkpoint_mapped(&mapped, &device, Some(&manifest), false, |loader| {
                    let StoredTensor::Bytes(uploaded) =
                        StoredTensor::load(loader, "scale", &manifest.entries["scale"])?
                    else {
                        anyhow::bail!("native F32 upload used typed storage");
                    };
                    ensure!(loader.bytes == 16, "F32 upload accounting mismatch");
                    Ok(uploaded)
                })?;
            let observed = device.upload.clone_dtoh(&uploaded)?;
            ensure!(
                observed.as_slice() == data,
                "F32 native upload changed the published bytes"
            );
            println!("PASS native_f32_upload shape=[2,2] bytes=16");
            for role in ["hc_attn_fn", "hc_ffn_fn"] {
                let model = crate::model_spec::ModelSpec::glm5_next(
                    &serde_json::from_str(include_str!(
                        "../../../tests/data/glm53-flash-config.json"
                    ))?,
                    &serde_json::Value::Null,
                )?;
                let mult = model
                    .glm5_next
                    .as_ref()
                    .context("mHC spec missing")?
                    .hc_mult as usize;
                let shape = vec![mult * (mult + 2), mult * model.hidden as usize];
                let data: Vec<u8> = (0..shape.iter().product::<usize>())
                    .flat_map(|i| {
                        bf16::from_f32(((i * 37 + 11) % 251) as f32 / 64.0 - 2.0)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect();
                let name = format!("model.language_model.layers.0.{role}");
                let mut header = serde_json::to_vec(&serde_json::json!({
                    &name: {"dtype":"BF16", "shape":shape, "data_offsets":[0,data.len()]}
                }))?;
                while !header.len().is_multiple_of(8) {
                    header.push(b' ');
                }
                let mut file = std::fs::File::create(root.join("model.safetensors"))?;
                file.write_all(&(header.len() as u64).to_le_bytes())?;
                file.write_all(&header)?;
                file.write_all(&data)?;
                drop(file);
                std::fs::write(
                    root.join("model.safetensors.index.json"),
                    serde_json::to_vec(&serde_json::json!({
                        "metadata":{"total_size":data.len()}, "weight_map":{&name:"model.safetensors"}
                    }))?,
                )?;
                let entry = TensorEntry {
                    dtype: Dtype::BF16,
                    shape: shape.clone(),
                    shard: "model.safetensors".into(),
                    bytes: data.len(),
                    load: true,
                    data_start: 8 + header.len() as u64,
                    data_offsets: [0, data.len() as u64],
                    scope: crate::checkpoint::TensorScope::Text,
                    layer: Some(0),
                    role: role.into(),
                    scale_name: None,
                };
                let manifest = TensorManifest {
                    entries: [(name.clone(), entry)].into(),
                    resident_bytes: data.len(),
                };
                let mapped = MappedCheckpoint::open(&root)?;
                let (uploaded, _) =
                    checkpoint_mapped(&mapped, &device, Some(&manifest), false, |loader| {
                        let value = StoredTensor::load(loader, &name, &manifest.entries[&name])?;
                        let StoredTensor::Bytes(uploaded) = value else {
                            anyhow::bail!("mHC BF16 upload used typed storage: {role}");
                        };
                        ensure!(
                            loader.bytes == data.len(),
                            "mHC upload byte accounting mismatch"
                        );
                        Ok(uploaded)
                    })?;
                ensure!(
                    device.upload.clone_dtoh(&uploaded)? == data,
                    "mHC native bytes changed: {role}"
                );
                println!(
                    "PASS native_mhc_upload role={role} dtype=BF16 shape={shape:?} bytes={} bit_changes=0",
                    data.len()
                );
            }
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(&root);
        result?;
        cleanup?;
        Ok(())
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

    #[test]
    fn dense_host_fixed_reserves_shard_aligned_staging_extent() -> Result<()> {
        let per_layer = 200_000_000usize;
        let offset = 4096u64;
        let len = 100_000_000u64;
        let staging_4096 =
            crate::experts::disk::DiskReader::aligned_staging_bytes(offset, len, 4096)?;
        let staging_8192 =
            crate::experts::disk::DiskReader::aligned_staging_bytes(offset, len, 8192)?;
        assert_eq!(staging_4096, 100_003_840);
        assert_eq!(staging_8192, staging_4096 + 4096);
        let available_host = (4 << 30) + 3 * per_layer + staging_4096;
        let ranges_8192 = [0, 1, 2, 3].map(|layer| (layer, offset, len, 8192));
        let (disk_fixed, disk_plan) = WeightCatalog::dense_host_fixed(
            &[per_layer; 4],
            &ranges_8192,
            2 * per_layer,
            available_host,
        )?;
        assert_eq!(disk_fixed, (4 << 30) + 2 * per_layer + staging_8192);
        assert_eq!(disk_plan.host_bytes, 0);
        assert_eq!(disk_plan.disk_bytes, 4 * per_layer);
        let ranges_4096 = [0, 1, 2, 3].map(|layer| (layer, offset, len, 4096));
        let (under_fixed, under_plan) = WeightCatalog::dense_host_fixed(
            &[per_layer; 4],
            &ranges_4096,
            2 * per_layer,
            available_host,
        )?;
        assert_eq!(under_fixed, (4 << 30) + 2 * per_layer + staging_4096);
        assert_eq!(under_plan.host_bytes, per_layer);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn checkpoint_manifestless_upload_waits_for_staging_dma_gpu() -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use safetensors::tensor::{TensorView, serialize_to_file};

        struct ScratchDir(std::path::PathBuf);
        impl Drop for ScratchDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        const SPIN_PTX: &str = "\
.version 7.0
.target sm_80
.address_size 64
.visible .entry spin_ns(.param .u64 duration_ns) {
    .reg .pred %p;
    .reg .u64 %start, %now, %elapsed, %limit;
    ld.param.u64 %limit, [duration_ns];
    mov.u64 %start, %globaltimer;
LOOP:
    mov.u64 %now, %globaltimer;
    sub.u64 %elapsed, %now, %start;
    setp.lt.u64 %p, %elapsed, %limit;
    @%p bra LOOP;
    ret;
}
";

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let dir = ScratchDir(std::env::temp_dir().join(format!(
            "ff-manifestless-upload-{}-{nanos}",
            std::process::id()
        )));
        std::fs::create_dir_all(&dir.0)?;
        let shard_path = dir.0.join("model.safetensors");

        let chunk_words = UPLOAD_CHUNK_BYTES / size_of::<bf16>();
        let b_words = 65_537usize;
        let mut a_bytes = vec![0u8; 2 * chunk_words * size_of::<bf16>()];
        for i in 0..chunk_words {
            let w0 = 0x3c00u16 | ((i % 251) as u16);
            a_bytes[2 * i..2 * i + 2].copy_from_slice(&w0.to_le_bytes());
            let j = chunk_words + i;
            let w1 = 0x4000u16 | (((i + 37) % 241) as u16);
            a_bytes[2 * j..2 * j + 2].copy_from_slice(&w1.to_le_bytes());
        }
        let mut b_bytes = vec![0u8; b_words * size_of::<bf16>()];
        for i in 0..b_words {
            let wb = 0x4200u16 | (((i + 97) % 239) as u16);
            b_bytes[2 * i..2 * i + 2].copy_from_slice(&wb.to_le_bytes());
        }
        serialize_to_file(
            [
                (
                    "a",
                    TensorView::new(Dtype::BF16, vec![2, chunk_words], &a_bytes)?,
                ),
                ("b", TensorView::new(Dtype::BF16, vec![b_words], &b_bytes)?),
            ],
            None,
            &shard_path,
        )?;

        let raw_file = std::fs::read(&shard_path)?;
        let file_tensors = SafeTensors::deserialize(&raw_file)?;
        let expected_a = file_tensors.tensor("a")?.data();
        let expected_b = file_tensors.tensor("b")?.data();

        let device = crate::backend::cuda::Device::new(0)?;
        let spin = device
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(SPIN_PTX))?
            .load_function("spin_ns")?;
        let duration_ns = 500_000_000u64;
        let launch_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        for (pass, expected_tracking) in [(0usize, true), (1usize, false)] {
            ensure!(
                device.ctx.is_event_tracking() == expected_tracking,
                "pass={pass} expected event_tracking={expected_tracking}, observed={}",
                device.ctx.is_event_tracking()
            );
            unsafe {
                device
                    .upload
                    .launch_builder(&spin)
                    .arg(&duration_ns)
                    .launch(launch_cfg)?;
            }
            let (a_dev, b_dev) = checkpoint(&dir.0, &device, |l| {
                let a = l.load(&[("a".into(), vec![2, chunk_words])])?;
                let b = l.load(&[("b".into(), vec![b_words])])?;
                Ok((a, b))
            })?;
            let got_a = device.stream.clone_dtoh(&a_dev)?;
            let got_b = device.stream.clone_dtoh(&b_dev)?;
            let mismatch_bytes = |got: &[bf16], expected: &[u8]| -> usize {
                got.iter()
                    .zip(expected.as_chunks::<2>().0)
                    .map(|(word, bytes)| {
                        let actual = word.to_bits().to_le_bytes();
                        usize::from(actual[0] != bytes[0]) + usize::from(actual[1] != bytes[1])
                    })
                    .sum()
            };
            let a_mismatches = mismatch_bytes(&got_a, expected_a);
            let b_mismatches = mismatch_bytes(&got_b, expected_b);
            ensure!(
                a_mismatches == 0 && b_mismatches == 0,
                "manifest-less upload byte mismatch pass={pass} event_tracking={expected_tracking} a_byte_mismatches={a_mismatches} b_byte_mismatches={b_mismatches}"
            );
        }
        println!(
            "PASS checkpoint_manifestless_upload passes=2 chunk_bytes={UPLOAD_CHUNK_BYTES} a_bytes={} b_bytes={} byte_mismatches=0",
            expected_a.len(),
            expected_b.len()
        );
        Ok(())
    }
}

/// DiffusionGemma decoder layer. The published BF16 tensors are placed
/// adjacent at upload; values are unchanged. Sliding layers pack q|k|v; full
/// layers pack q|k and derive V from the shared projection (k_eq_v). Experts
/// hold the fused [experts, 2*inter, hidden] / [experts, hidden, inter] form
/// of the checkpoint, device-resident.
pub(crate) struct DiffusionLayer {
    pub qkv: CudaSlice<bf16>,
    pub q_norm: CudaSlice<bf16>,
    pub k_norm: CudaSlice<bf16>,
    pub o: CudaSlice<bf16>,
    pub input_norm: CudaSlice<bf16>,
    pub post_attn: CudaSlice<bf16>,
    pub pre_ffn: CudaSlice<bf16>,
    pub pre_ffn2: CudaSlice<bf16>,
    pub post1: CudaSlice<bf16>,
    pub post2: CudaSlice<bf16>,
    pub post: CudaSlice<bf16>,
    pub dense_gu: CudaSlice<bf16>,
    pub dense_down: CudaSlice<bf16>,
    pub router_proj: CudaSlice<bf16>,
    pub router_scale: CudaSlice<bf16>,
    pub per_expert_scale: CudaSlice<bf16>,
    pub layer_scalar: CudaSlice<bf16>,
}

pub(crate) struct DiffusionWeights {
    pub embed: CudaSlice<bf16>,
    pub final_norm: CudaSlice<bf16>,
    pub layers: Vec<DiffusionLayer>,
    pub encoder_scalars: Vec<CudaSlice<bf16>>,
    pub sc_pre_norm: CudaSlice<bf16>,
    pub sc_gu: CudaSlice<bf16>,
    pub sc_down: CudaSlice<bf16>,
}

impl DiffusionWeights {
    pub(crate) fn load(
        dir: &Path,
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
    ) -> Result<Self> {
        let mapped = MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let (weights, _) = checkpoint_mapped(&mapped, d, Some(&manifest), false, |l| {
            let h = spec.hidden as usize;
            let dense = spec.intermediate as usize;
            let moe = spec.moe.as_ref().context("DiffusionGemma MoE missing")?;
            let e = moe.experts as usize;
            let mut layers = Vec::new();
            for (i, plan) in spec.layers.iter().enumerate() {
                let g = plan.geometry.context("layer geometry missing")?;
                let dim = g.head_dim as usize;
                let kvh = g.kv_heads.unwrap_or(spec.kv_heads) as usize;
                let qh = spec.q_heads as usize;
                let p = format!("model.decoder.layers.{i}");
                let mut qkv_specs = vec![
                    (format!("{p}.self_attn.q_proj.weight"), vec![qh * dim, h]),
                    (format!("{p}.self_attn.k_proj.weight"), vec![kvh * dim, h]),
                ];
                if plan.sliding_window.is_some() {
                    qkv_specs.push((format!("{p}.self_attn.v_proj.weight"), vec![kvh * dim, h]));
                }
                let qkv = l.load(&qkv_specs)?;

                let layer = DiffusionLayer {
                    qkv,
                    q_norm: l.load(&[(format!("{p}.self_attn.q_norm.weight"), vec![dim])])?,
                    k_norm: l.load(&[(format!("{p}.self_attn.k_norm.weight"), vec![dim])])?,
                    o: l.load(&[(format!("{p}.self_attn.o_proj.weight"), vec![h, qh * dim])])?,
                    input_norm: l.load(&[(format!("{p}.input_layernorm.weight"), vec![h])])?,
                    post_attn: l
                        .load(&[(format!("{p}.post_attention_layernorm.weight"), vec![h])])?,
                    pre_ffn: l
                        .load(&[(format!("{p}.pre_feedforward_layernorm.weight"), vec![h])])?,
                    pre_ffn2: l
                        .load(&[(format!("{p}.pre_feedforward_layernorm_2.weight"), vec![h])])?,
                    post1: l
                        .load(&[(format!("{p}.post_feedforward_layernorm_1.weight"), vec![h])])?,
                    post2: l
                        .load(&[(format!("{p}.post_feedforward_layernorm_2.weight"), vec![h])])?,
                    post: l.load(&[(format!("{p}.post_feedforward_layernorm.weight"), vec![h])])?,
                    dense_gu: l.load(&[
                        (format!("{p}.mlp.gate_proj.weight"), vec![dense, h]),
                        (format!("{p}.mlp.up_proj.weight"), vec![dense, h]),
                    ])?,
                    dense_down: l.load(&[(format!("{p}.mlp.down_proj.weight"), vec![h, dense])])?,
                    router_proj: l.load(&[(format!("{p}.router.proj.weight"), vec![e, h])])?,
                    router_scale: l.load(&[(format!("{p}.router.scale"), vec![h])])?,
                    per_expert_scale: l
                        .load(&[(format!("{p}.router.per_expert_scale"), vec![e])])?,
                    layer_scalar: l.load(&[(format!("{p}.layer_scalar"), vec![1])])?,
                };
                layers.push(layer);
            }
            let mut encoder_scalars = Vec::new();
            for i in 0..spec.layers.len() {
                encoder_scalars.push(l.load(&[(
                    format!("model.encoder.language_model.layers.{i}.layer_scalar"),
                    vec![1],
                )])?);
                let p = format!("model.decoder.layers.{i}");
                for name in [
                    format!("{p}.experts.gate_up_proj"),
                    format!("{p}.experts.down_proj"),
                ] {
                    l.defer(&name)?;
                }
            }
            let embed = l.load(&[(
                "model.decoder.embed_tokens.weight".to_owned(),
                vec![spec.vocab as usize, h],
            )])?;
            let final_norm = l.load(&[("model.decoder.norm.weight".to_owned(), vec![h])])?;
            Ok(Self {
                embed,
                final_norm,
                layers,
                encoder_scalars,
                sc_pre_norm: l.load(&[(
                    "model.decoder.self_conditioning.pre_norm.weight".to_owned(),
                    vec![h],
                )])?,
                sc_gu: l.load(&[
                    (
                        "model.decoder.self_conditioning.gate_proj.weight".to_owned(),
                        vec![dense, h],
                    ),
                    (
                        "model.decoder.self_conditioning.up_proj.weight".to_owned(),
                        vec![dense, h],
                    ),
                ])?,
                sc_down: l.load(&[(
                    "model.decoder.self_conditioning.down_proj.weight".to_owned(),
                    vec![h, dense],
                )])?,
            })
        })?;
        Ok(weights)
    }
}
