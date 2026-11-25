use super::Device;
use crate::backend::gpu::{CudaContext, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice};
use crate::backend::gpu::{CudaFunction, PushKernelArg};
use crate::config::Config;
use crate::experts::{
    arena::{HostBlocks, SlotLayout},
    catalogue::ExpertGeometry,
    plan::Capacity,
};
use crate::gguf::GgmlType;
use anyhow::bail;
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

/// Fused projection groups of one classic decoder layer, derived from Config.
/// qkv is `[qkv_dim, hidden]` in `[q; k; v]` order; gu is
/// `[2 * intermediate, hidden]` in `[gate; up]` order.
pub(crate) struct ClassicFused {
    pub qkv: [(String, Vec<usize>); 3],
    pub gu: [(String, Vec<usize>); 2],
}

impl ClassicFused {
    pub(crate) fn new(c: &Config, layer: usize) -> Self {
        let prefix = format!("model.layers.{layer}");
        let h = c.hidden_size;
        let kv = c.kv_dim();
        let f = c.intermediate_size;
        Self {
            qkv: [
                (format!("{prefix}.self_attn.q_proj.weight"), vec![h, h]),
                (format!("{prefix}.self_attn.k_proj.weight"), vec![kv, h]),
                (format!("{prefix}.self_attn.v_proj.weight"), vec![kv, h]),
            ],
            gu: [
                (format!("{prefix}.mlp.gate_proj.weight"), vec![f, h]),
                (format!("{prefix}.mlp.up_proj.weight"), vec![f, h]),
            ],
        }
    }

    /// Rejects any q/k/v or gate/up reorder. Call before binding the fused groups.
    pub(crate) fn validate(&self, layer: usize) -> Result<()> {
        let prefix = format!("model.layers.{layer}");
        let want_qkv = [
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.v_proj.weight"),
        ];
        let want_gu = [
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
        ];
        ensure!(
            self.qkv.iter().map(|p| &p.0).eq(want_qkv.iter()),
            "classic qkv binding out of order at layer {layer}"
        );
        ensure!(
            self.gu.iter().map(|p| &p.0).eq(want_gu.iter()),
            "classic gate/up binding out of order at layer {layer}"
        );
        Ok(())
    }
}

fn classic_qkv_name(layer: usize) -> String {
    format!("model.layers.{layer}.self_attn.qkv_proj.weight")
}

fn classic_gu_name(layer: usize) -> String {
    format!("model.layers.{layer}.mlp.gu_proj.weight")
}

#[cfg(feature = "cuda")]
fn slice_stream(s: &CudaSlice<bf16>) -> &Arc<CudaStream> {
    s.stream()
}

fn classic_order(
    dir: &Path,
    spec: &crate::model_spec::ModelSpec,
) -> Result<Option<Vec<Vec<String>>>> {
    if spec.architecture != crate::model_spec::Architecture::Llama {
        return Ok(None);
    }
    let c = Config::read(dir)?;
    let mut order = Vec::with_capacity(spec.layers.len());
    for layer in 0..spec.layers.len() {
        let fused = ClassicFused::new(&c, layer);
        fused.validate(layer)?;
        let p = format!("model.layers.{layer}");
        let names = vec![
            fused.qkv[0].0.clone(),
            fused.qkv[1].0.clone(),
            fused.qkv[2].0.clone(),
            fused.gu[0].0.clone(),
            fused.gu[1].0.clone(),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.mlp.down_proj.weight"),
        ];
        crate::checkpoint::validate_classic_order(layer, &names)?;
        order.push(names);
    }
    Ok(Some(order))
}

fn classic_fused_specs(
    manifest: &TensorManifest,
    order: &[String],
    layer: usize,
) -> Result<ClassicFused> {
    ensure!(
        order.len() == 7,
        "classic layer {layer} has {} projections",
        order.len()
    );
    let shape = |index: usize| -> Result<Vec<usize>> {
        Ok(manifest
            .entries
            .get(&order[index])
            .with_context(|| format!("classic tensor {} missing", order[index]))?
            .shape
            .clone())
    };
    let fused = ClassicFused {
        qkv: [
            (order[0].clone(), shape(0)?),
            (order[1].clone(), shape(1)?),
            (order[2].clone(), shape(2)?),
        ],
        gu: [(order[3].clone(), shape(3)?), (order[4].clone(), shape(4)?)],
    };
    fused.validate(layer)?;
    Ok(fused)
}

fn emit_classic_resident(
    loader: &mut Loader<'_>,
    tensors: &mut BTreeMap<String, StoredTensor>,
    manifest: &TensorManifest,
    order: &[String],
    layer: usize,
) -> Result<()> {
    let fused = classic_fused_specs(manifest, order, layer)?;
    let qkv = loader.load(&fused.qkv)?;
    let gu = loader.load(&fused.gu)?;
    tensors.insert(classic_qkv_name(layer), StoredTensor::Bf16(qkv));
    tensors.insert(classic_gu_name(layer), StoredTensor::Bf16(gu));
    for name in &order[5..] {
        let entry = manifest
            .entries
            .get(name)
            .with_context(|| format!("classic tensor {name} missing"))?;
        tensors.insert(name.clone(), StoredTensor::load(loader, name, entry)?);
    }
    Ok(())
}

pub(crate) struct ClassicCatalog {
    order: Vec<Vec<String>>,
    #[cfg(feature = "cuda")]
    bindings: Vec<ClassicLayer>,
}

#[cfg(feature = "cuda")]
pub(crate) struct ClassicLayer {
    pub qkv: super::linear::LinearWeightRef,
    pub o: super::linear::LinearWeightRef,
    pub gu: super::linear::LinearWeightRef,
    pub down: super::linear::LinearWeightRef,
    pub input_norm: super::contracts::Tensor<bf16, 1>,
    pub post_norm: super::contracts::Tensor<bf16, 1>,
}

#[cfg(feature = "cuda")]
pub(crate) struct SlotWeight<'a> {
    ptr: u64,
    len: usize,
    stream: &'a Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl<'a> SlotWeight<'a> {
    pub(crate) fn new(
        weight: super::linear::LinearWeightRef,
        stream: &'a Arc<CudaStream>,
    ) -> Result<Self> {
        let super::linear::LinearWeightRef::Bf16(tensor) = weight else {
            bail!("classic streamed weight is not BF16");
        };
        let len = usize::try_from(tensor.shape[0])?
            .checked_mul(usize::try_from(tensor.shape[1])?)
            .context("classic slot element count overflow")?;
        Ok(Self {
            ptr: tensor.ptr,
            len,
            stream,
        })
    }
}

#[cfg(feature = "cuda")]
impl cudarc::driver::DeviceSlice<bf16> for SlotWeight<'_> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

#[cfg(feature = "cuda")]
impl cudarc::driver::DevicePtr<bf16> for SlotWeight<'_> {
    fn device_ptr<'b>(
        &'b self,
        _stream: &'b CudaStream,
    ) -> (sys::CUdeviceptr, cudarc::driver::SyncOnDrop<'b>) {
        (self.ptr, cudarc::driver::SyncOnDrop::Record(None))
    }
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

/// MiniMax-Music3 synthesis weights, all F32 as published (raw byte slices;
/// the engine reinterprets them as f32 at setup).
#[cfg(feature = "cuda")]
pub struct Music3SynWeights {
    pub flow: Music3FlowWeights,
    pub cond: Music3CondWeights,
    pub vocoder: Music3VocoderWeights,
}

#[cfg(feature = "cuda")]
pub struct Music3FlowWeights {
    pub blocks: Vec<Music3FlowBlock>,
    /// k=1 conv over the concat channels, stored [out, in].
    pub preprocess: CudaSlice<u8>,
    pub postprocess: CudaSlice<u8>,
    pub proj_in: CudaSlice<u8>,
    pub proj_out: CudaSlice<u8>,
    /// Fourier projection weight [fourier_dim / 2, 1].
    pub time_proj: CudaSlice<u8>,
    pub time_l1_w: CudaSlice<u8>,
    pub time_l1_b: CudaSlice<u8>,
    pub time_l2_w: CudaSlice<u8>,
    pub time_l2_b: CudaSlice<u8>,
}

#[cfg(feature = "cuda")]
pub struct Music3FlowBlock {
    pub to_q: CudaSlice<u8>,
    pub to_k: CudaSlice<u8>,
    pub to_v: CudaSlice<u8>,
    pub to_out: CudaSlice<u8>,
    pub norm1_w: CudaSlice<u8>,
    pub norm1_b: CudaSlice<u8>,
    pub norm2_w: CudaSlice<u8>,
    pub norm2_b: CudaSlice<u8>,
    pub ff_in_w: CudaSlice<u8>,
    pub ff_in_b: CudaSlice<u8>,
    pub ff_out_w: CudaSlice<u8>,
    pub ff_out_b: CudaSlice<u8>,
}

#[cfg(feature = "cuda")]
pub struct Music3CondWeights {
    /// Conv1d k=3 over the fused condition layers, stored [out, in, 3].
    pub proj_w: CudaSlice<u8>,
    pub proj_b: CudaSlice<u8>,
    pub layer_weight_logits: CudaSlice<u8>,
    pub layer_scale: CudaSlice<u8>,
}

#[cfg(feature = "cuda")]
pub struct Music3VocoderWeights {
    pub dec_in_proj_w: CudaSlice<u8>,
    pub dec_in_proj_b: CudaSlice<u8>,
    pub conv_in_g: CudaSlice<u8>,
    pub conv_in_v: CudaSlice<u8>,
    pub conv_in_b: CudaSlice<u8>,
    pub blocks: Vec<Music3VocoderBlock>,
    pub snake_out_alpha: CudaSlice<u8>,
    pub conv_out_g: CudaSlice<u8>,
    pub conv_out_v: CudaSlice<u8>,
    pub conv_out_b: CudaSlice<u8>,
}

#[cfg(feature = "cuda")]
pub struct Music3VocoderBlock {
    pub snake1_alpha: CudaSlice<u8>,
    /// Weight-norm transposed conv: g [out, 1, 1], v [in, out, kernel].
    pub conv_t_g: CudaSlice<u8>,
    pub conv_t_v: CudaSlice<u8>,
    pub conv_t_b: CudaSlice<u8>,
    /// Dilations 1, 3, 9 in unit order.
    pub units: Vec<Music3VocoderResUnit>,
}

#[cfg(feature = "cuda")]
pub struct Music3VocoderResUnit {
    pub snake1_alpha: CudaSlice<u8>,
    pub snake2_alpha: CudaSlice<u8>,
    pub c1_g: CudaSlice<u8>,
    pub c1_v: CudaSlice<u8>,
    pub c1_b: CudaSlice<u8>,
    pub c2_g: CudaSlice<u8>,
    pub c2_v: CudaSlice<u8>,
    pub c2_b: CudaSlice<u8>,
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
                let guard = disk.values(slot)?;
                guard.copy_logical(staging.as_mut_slice()?)?;
                drop(guard);
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
                    let guard = self.disk.values(disk_slot)?;
                    guard.copy_logical(claim.slice()?)?;
                    drop(guard);
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

#[cfg(test)]
mod fused_order_tests {
    use crate::checkpoint::classic_layer_names;

    #[test]
    fn fused_order_pins_qkv_and_gate_up_halves() {
        let p = "model.layers.3";
        let correct = classic_layer_names(3);
        assert_eq!(
            correct,
            [
                format!("{p}.self_attn.q_proj.weight"),
                format!("{p}.self_attn.k_proj.weight"),
                format!("{p}.self_attn.v_proj.weight"),
                format!("{p}.mlp.gate_proj.weight"),
                format!("{p}.mlp.up_proj.weight"),
                format!("{p}.self_attn.o_proj.weight"),
                format!("{p}.mlp.down_proj.weight"),
            ]
        );
        assert_ne!(classic_layer_names(3), classic_layer_names(4));
    }

    #[test]
    fn fused_order_swaps_change_the_binding() {
        let correct = classic_layer_names(3);
        for (x, y) in [(0, 1), (1, 2), (0, 2), (3, 4)] {
            let mut swapped = correct.clone();
            swapped.swap(x, y);
            assert_ne!(swapped, correct, "swap {x}/{y} must change the binding");
        }
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

    #[cfg(feature = "rocm")]
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
                let fused = ClassicFused::new(c, i);
                fused.validate(i)?;
                let qkv = l.load(&fused.qkv)?;
                let gu = l.load(&fused.gu)?;
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
                    input_norm: l.load(&[(
                        format!("{p}.{}", crate::checkpoint::GLM_INPUT_LAYERNORM),
                        vec![1024],
                    )])?,
                    post_norm: l.load(&[(
                        format!("{p}.{}", crate::checkpoint::GLM_POST_ATTENTION_LAYERNORM),
                        vec![1024],
                    )])?,
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

#[cfg(feature = "cuda")]
impl Music3SynWeights {
    pub(crate) fn load(dir: &Path, spec: &crate::music3::SynSpec, d: &Device) -> Result<Self> {
        let dim = spec.flow_dim() as usize;
        let concat = spec.concat_channels() as usize;
        let ff = spec.flow_ff_inner as usize;
        let mapped_transformer = crate::checkpoint::MappedCheckpoint::open_component(
            &dir.join("transformer"),
            "diffusion_pytorch_model",
        )?;
        let manifest_transformer = spec.manifest_transformer(&mapped_transformer)?;
        let (flow, _) = checkpoint_mapped(
            &mapped_transformer,
            d,
            Some(&manifest_transformer),
            false,
            |l| {
                let bf16 = |l: &mut Loader<'_>, name: &str, shape: &[usize]| {
                    l.load_bytes_bf16_from_f32(name, shape)
                };
                let mut blocks = Vec::with_capacity(spec.flow_layers as usize);
                for i in 0..spec.flow_layers as usize {
                    let p = format!("transformer_blocks.{i}");
                    blocks.push(Music3FlowBlock {
                        to_q: bf16(l, &format!("{p}.attn.to_q.weight"), &[dim, dim])?,
                        to_k: bf16(l, &format!("{p}.attn.to_k.weight"), &[dim, dim])?,
                        to_v: bf16(l, &format!("{p}.attn.to_v.weight"), &[dim, dim])?,
                        to_out: bf16(l, &format!("{p}.attn.to_out.0.weight"), &[dim, dim])?,
                        norm1_w: bf16(l, &format!("{p}.norm1.weight"), &[dim])?,
                        norm1_b: bf16(l, &format!("{p}.norm1.bias"), &[dim])?,
                        norm2_w: bf16(l, &format!("{p}.norm2.weight"), &[dim])?,
                        norm2_b: bf16(l, &format!("{p}.norm2.bias"), &[dim])?,
                        ff_in_w: bf16(l, &format!("{p}.ff_in.weight"), &[2 * ff, dim])?,
                        ff_in_b: bf16(l, &format!("{p}.ff_in.bias"), &[2 * ff])?,
                        ff_out_w: bf16(l, &format!("{p}.ff_out.weight"), &[dim, ff])?,
                        ff_out_b: bf16(l, &format!("{p}.ff_out.bias"), &[dim])?,
                    });
                }
                Ok(Music3FlowWeights {
                    blocks,
                    preprocess: bf16(l, "preprocess_conv.weight", &[concat, concat, 1])?,
                    postprocess: bf16(
                        l,
                        "postprocess_conv.weight",
                        &[spec.in_channels as usize, spec.in_channels as usize, 1],
                    )?,
                    proj_in: bf16(l, "proj_in.weight", &[dim, concat])?,
                    proj_out: bf16(l, "proj_out.weight", &[spec.in_channels as usize, dim])?,
                    time_proj: bf16(l, "time_proj.weight", &[spec.fourier_dim as usize / 2, 1])?,
                    time_l1_w: bf16(
                        l,
                        "time_embed.linear_1.weight",
                        &[dim, spec.fourier_dim as usize],
                    )?,
                    time_l1_b: bf16(l, "time_embed.linear_1.bias", &[dim])?,
                    time_l2_w: bf16(l, "time_embed.linear_2.weight", &[dim, dim])?,
                    time_l2_b: bf16(l, "time_embed.linear_2.bias", &[dim])?,
                })
            },
        )?;

        let cond_hidden = spec.cond_hidden as usize;
        let layers = spec.cond_layers as usize;
        let mapped_condition_encoder = crate::checkpoint::MappedCheckpoint::open_component(
            &dir.join("condition_encoder"),
            "diffusion_pytorch_model",
        )?;
        let manifest_condition_encoder = spec.manifest_condition(&mapped_condition_encoder)?;
        let (cond, _) = checkpoint_mapped(
            &mapped_condition_encoder,
            d,
            Some(&manifest_condition_encoder),
            false,
            |l| {
                let bf16 = |l: &mut Loader<'_>, name: &str, shape: &[usize]| {
                    l.load_bytes_bf16_from_f32(name, shape)
                };
                Ok(Music3CondWeights {
                    proj_w: bf16(l, "proj.weight", &[spec.cond_out as usize, cond_hidden, 3])?,
                    proj_b: bf16(l, "proj.bias", &[spec.cond_out as usize])?,
                    layer_weight_logits: bf16(l, "layer_weight_logits", &[layers])?,
                    layer_scale: bf16(l, "layer_scale", &[1])?,
                })
            },
        )?;

        // DAC-style decoder: each stage halves the channels while upsampling
        // by its ratio; conv_t kernel = 2 * stride, res units at stage output.
        let hidden = spec.decoder_hidden as usize;
        let mut block_dims = Vec::with_capacity(spec.upsampling.len());
        let mut stage_input = hidden;
        for ratio in &spec.upsampling {
            let output = stage_input / 2;
            block_dims.push((stage_input, output, *ratio as usize));
            stage_input = output;
        }
        let tail = stage_input;
        let mapped_vocoder = crate::checkpoint::MappedCheckpoint::open_component(
            &dir.join("vocoder"),
            "diffusion_pytorch_model",
        )?;
        let manifest_vocoder = spec.manifest_vocoder(&mapped_vocoder)?;
        let (vocoder, _) =
            checkpoint_mapped(&mapped_vocoder, d, Some(&manifest_vocoder), false, |l| {
                let bf16 = |l: &mut Loader<'_>, name: &str, shape: &[usize]| {
                    l.load_bytes_bf16_from_f32(name, shape)
                };
                let mut blocks = Vec::with_capacity(block_dims.len());
                for (i, &(in_dim, out_dim, stride)) in block_dims.iter().enumerate() {
                    let kernel = 2 * stride;
                    let b = format!("blocks.{i}");
                    let mut units = Vec::with_capacity(3);
                    for u in 1..=3 {
                        let ru = format!("{b}.res_unit{u}");
                        units.push(Music3VocoderResUnit {
                            snake1_alpha: bf16(l, &format!("{ru}.snake1.alpha"), &[1, out_dim, 1])?,
                            snake2_alpha: bf16(l, &format!("{ru}.snake2.alpha"), &[1, out_dim, 1])?,
                            c1_g: bf16(l, &format!("{ru}.conv1.weight_g"), &[out_dim, 1, 1])?,
                            c1_v: bf16(l, &format!("{ru}.conv1.weight_v"), &[out_dim, out_dim, 7])?,
                            c1_b: bf16(l, &format!("{ru}.conv1.bias"), &[out_dim])?,
                            c2_g: bf16(l, &format!("{ru}.conv2.weight_g"), &[out_dim, 1, 1])?,
                            c2_v: bf16(l, &format!("{ru}.conv2.weight_v"), &[out_dim, out_dim, 1])?,
                            c2_b: bf16(l, &format!("{ru}.conv2.bias"), &[out_dim])?,
                        });
                    }
                    blocks.push(Music3VocoderBlock {
                        snake1_alpha: bf16(l, &format!("{b}.snake1.alpha"), &[1, in_dim, 1])?,
                        conv_t_g: bf16(l, &format!("{b}.conv_t1.weight_g"), &[in_dim, 1, 1])?,
                        conv_t_v: bf16(
                            l,
                            &format!("{b}.conv_t1.weight_v"),
                            &[in_dim, out_dim, kernel],
                        )?,
                        conv_t_b: bf16(l, &format!("{b}.conv_t1.bias"), &[out_dim])?,
                        units,
                    });
                }
                Ok(Music3VocoderWeights {
                    dec_in_proj_w: bf16(
                        l,
                        "dec_in_proj.weight",
                        &[
                            spec.decoder_input as usize,
                            spec.vocoder_stream_channels() as usize,
                            1,
                        ],
                    )?,
                    dec_in_proj_b: bf16(l, "dec_in_proj.bias", &[spec.decoder_input as usize])?,
                    conv_in_g: bf16(l, "conv_in.weight_g", &[hidden, 1, 1])?,
                    conv_in_v: bf16(
                        l,
                        "conv_in.weight_v",
                        &[hidden, spec.decoder_input as usize, 7],
                    )?,
                    conv_in_b: bf16(l, "conv_in.bias", &[hidden])?,
                    blocks,
                    snake_out_alpha: bf16(l, "snake_out.alpha", &[1, tail, 1])?,
                    conv_out_g: bf16(l, "conv_out.weight_g", &[1, 1, 1])?,
                    conv_out_v: bf16(l, "conv_out.weight_v", &[1, tail, 7])?,
                    conv_out_b: bf16(l, "conv_out.bias", &[1])?,
                })
            })?;
        Ok(Self {
            flow,
            cond,
            vocoder,
        })
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
                Dtype::I8 | Dtype::U8 => true,
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
    #[cfg(feature = "cuda")]
    pub(super) fn load_bytes_bf16_from_f32(
        &mut self,
        name: &str,
        shape: &[usize],
    ) -> Result<CudaSlice<u8>> {
        let source = self.load_bytes(name, shape, Dtype::F32)?;
        let host = self.stream.clone_dtoh(&source)?;
        ensure!(host.len() % 4 == 0, "F32 source extent for {name}");
        let mut converted = Vec::with_capacity(host.len() / 2);
        for bytes in host.as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*bytes);
            converted.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
        }
        Ok(self.stream.clone_htod(&converted)?)
    }
}

/// The GGUF dequant kernels, loaded once from the `q4_0_linear` cubin.
pub(crate) struct QuantKernels {
    dequant_bf16: CudaFunction,
    dequant_f32: CudaFunction,
    swap_halves: CudaFunction,
    f16_to_bf16: CudaFunction,
}

/// One dequant launch: the source bytes, the destination buffer and its element type.
pub(crate) struct DequantCopy {
    pub source: u64,
    pub destination: u64,
    pub elements: usize,
    pub dtype: Dtype,
}

impl QuantKernels {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = super::cubin::module(ctx, "q4_0_linear")?;
        Ok(Self {
            dequant_bf16: module.load_function("q4_0_dequant_bf16")?,
            dequant_f32: module.load_function("q4_0_dequant_f32")?,
            swap_halves: module.load_function("q4_0_dequant_swap_halves_bf16")?,
            f16_to_bf16: module.load_function("f16_to_bf16")?,
        })
    }

    /// Dequantizes `entry`'s `copy.source` bytes into `copy.destination` (`copy.elements` of `copy.dtype`).
    pub(crate) fn dequant(
        &self,
        s: &Arc<CudaStream>,
        name: &str,
        entry: &TensorEntry,
        copy: DequantCopy,
    ) -> Result<()> {
        let DequantCopy {
            source,
            destination,
            elements,
            dtype,
        } = copy;
        match entry.quant.context("quantized entry")? {
            GgmlType::F16 => {
                ensure!(
                    dtype == Dtype::BF16,
                    "{name}: F16 GGUF source cannot fill a {dtype:?} buffer"
                );
                let n = elements as u64;
                unsafe {
                    s.launch_builder(&self.f16_to_bf16)
                        .arg(&source)
                        .arg(&destination)
                        .arg(&(n as i64))
                        .launch(super::ops::flat(n as usize))?;
                }
            }
            GgmlType::Q4_0 => {
                let blocks = entry.bytes / 18;
                ensure!(
                    blocks * 18 == entry.bytes && blocks * 32 == elements,
                    "{name}: {} source bytes do not fill a {elements}-element buffer",
                    entry.bytes
                );
                if entry.swap_halves {
                    ensure!(
                        dtype == Dtype::BF16 && entry.shape.len() == 2,
                        "{name}: swap_halves needs a 2-D BF16 tensor"
                    );
                    let (rows, cols) = (entry.shape[0] as u64, entry.shape[1] as u64);
                    unsafe {
                        s.launch_builder(&self.swap_halves)
                            .arg(&source)
                            .arg(&destination)
                            .arg(&(rows as i64))
                            .arg(&(cols as i64))
                            .launch(super::ops::flat((rows as usize) * (cols as usize) / 32))?;
                    }
                } else if dtype == Dtype::BF16 {
                    unsafe {
                        s.launch_builder(&self.dequant_bf16)
                            .arg(&source)
                            .arg(&destination)
                            .arg(&(blocks as i64))
                            .launch(super::ops::flat(blocks))?;
                    }
                } else if dtype == Dtype::F32 {
                    unsafe {
                        s.launch_builder(&self.dequant_f32)
                            .arg(&source)
                            .arg(&destination)
                            .arg(&(blocks as i64))
                            .launch(super::ops::flat(blocks))?;
                    }
                } else {
                    bail!("{name}: Q4_0 source cannot fill a {dtype:?} buffer");
                }
            }
            other => bail!("{name}: unsupported GGUF type {other:?}"),
        }
        Ok(())
    }
}

/// Whether a stored GGUF type dequantizes into its device slot; the other accepted types pass through.
pub(crate) fn dequantizes(quant: Option<GgmlType>) -> bool {
    matches!(quant, Some(GgmlType::Q4_0 | GgmlType::F16))
}

/// Device slot bytes of a streamed linear: the BF16 tile for a dequantized entry, the stored bytes otherwise.
pub(crate) fn device_slot_bytes(name: &str, entry: &TensorEntry) -> Result<usize> {
    match entry.quant {
        Some(quant) if dequantizes(Some(quant)) => {
            ensure!(entry.shape.len() == 2, "{name}: quantized entry is not 2-D");
            Ok(2 * entry.shape[0] * entry.shape[1])
        }
        Some(GgmlType::F32 | GgmlType::Bf16) => Ok(entry.bytes),
        Some(other) => bail!("{name}: quantized entry type {other:?} has no device path"),
        None => Ok(entry.bytes),
    }
}

/// Finiteness of a quantized entry's stored source: every F16 value for F16, each 18-byte Q4_0 block's scale.
pub(crate) fn quant_source_finite(name: &str, entry: &TensorEntry, bytes: &[u8]) -> Result<()> {
    match entry.quant {
        None if entry.dtype == safetensors::Dtype::F32 => {
            let (words, rest) = bytes.as_chunks::<4>();
            ensure!(
                rest.is_empty() && words.iter().all(|v| f32::from_le_bytes(*v).is_finite()),
                "{name}: nonfinite or partial F32 source"
            );
        }

        Some(GgmlType::F16) => {
            let (words, rest) = bytes.as_chunks::<2>();
            ensure!(
                rest.is_empty(),
                "{name}: F16 source is not a whole number of values"
            );
            ensure!(
                words
                    .iter()
                    .all(|v| half::f16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                "{name}: nonfinite F16 value"
            );
        }
        Some(GgmlType::Q4_0) => {
            let (blocks, rest) = bytes.as_chunks::<18>();
            ensure!(
                rest.is_empty(),
                "{name}: Q4_0 source is not a whole number of blocks"
            );
            ensure!(
                blocks
                    .iter()
                    .all(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()),
                "{name}: nonfinite Q4_0 scale"
            );
        }
        _ => {}
    }
    Ok(())
}

/// Rewrites BF16 entries in a disk-tier layer. F32, I8, U8, and quantized entries stay.
fn rewrite_fp16_weights(layer: &DenseLayer, bytes: &mut [u8], report: bool) -> Result<()> {
    for entry in &layer.entries {
        if entry.narrow_f32 || entry.entry.quant.is_some() || entry.entry.dtype != Dtype::BF16 {
            continue;
        }
        let end = entry
            .source_offset
            .checked_add(entry.entry.bytes)
            .context("FP16 weight span")?;
        ensure!(end <= bytes.len(), "{}: FP16 weight span", entry.name);
        crate::tensors::bf16_weight_to_fp16(
            &entry.name,
            &mut bytes[entry.source_offset..end],
            report,
        )?;
    }
    Ok(())
}

/// Finiteness of a streamed layer's quantized source, over every entry's stored bytes.
fn check_layer_source(layer: &DenseLayer, source: &[u8]) -> Result<()> {
    for entry in &layer.entries {
        let bytes = &source[entry.source_offset..entry.source_offset + entry.entry.bytes];
        quant_source_finite(&entry.name, &entry.entry, bytes)?;
    }
    Ok(())
}

struct DenseEntry {
    name: String,
    device_offset: usize,
    source_offset: usize,
    entry: TensorEntry,
    narrow_f32: bool,
}

struct DenseLayer {
    slot: Option<usize>,
    next: Option<usize>,
    next_disk: Option<usize>,
    bytes: usize,
    source_bytes: usize,
    needs_staging: bool,
    entries: Vec<DenseEntry>,
    host: Option<PinnedHostSlice<u8>>,
    checked: bool,
}

struct DenseSlot {
    device: CudaSlice<u8>,
    ready: crate::backend::gpu::CudaEvent,
    retired: crate::backend::gpu::CudaEvent,
    retired_valid: bool,
    loaded: Option<usize>,
}

/// One layer range the disk supply reads into its pinned slot.
#[cfg(any(feature = "cuda", feature = "rocm"))]
struct SupplyRange {
    source_offset: usize,
    shard: Arc<str>,
    offset: u64,
    len: u64,
}

/// Per-layer state of the disk supply: queued by the consumer's pump, claimed into a
/// slot by a worker, published for the consumer's demand, retired after its upload.
#[derive(Clone)]
enum SupplyLayer {
    Waiting,
    Queued,
    Reading,
    Ready(usize),
    Consumed,
    Failed,
}

struct SupplyState {
    layer: Vec<SupplyLayer>,
    /// Completed reads per layer.
    reads: Vec<u64>,
    pending: std::collections::VecDeque<usize>,
    free: Vec<bool>,
    outstanding: usize,
    failure: Option<String>,
    published: u64,
    retired: u64,
}

impl SupplyState {
    /// Admits `layer` unless it is already in flight; `false` when capacity holds.
    /// A layer the consumer will demand next stays at or ahead of the pump cursor.
    fn admit(&mut self, layer: usize, capacity: usize) -> bool {
        if !matches!(
            self.layer[layer],
            SupplyLayer::Waiting | SupplyLayer::Consumed
        ) {
            return true;
        }
        if self.outstanding >= capacity {
            return false;
        }
        self.layer[layer] = SupplyLayer::Queued;
        self.pending.push_back(layer);
        self.outstanding += 1;
        true
    }

    /// The layer's published slot, `None` while its read is queued, in flight or failed.
    /// A layer with no read queued or in flight errors naming its next visit.
    fn ready(&self, layer: usize) -> Result<Option<usize>> {
        match self.layer[layer] {
            SupplyLayer::Ready(slot) => Ok(Some(slot)),
            SupplyLayer::Queued | SupplyLayer::Reading | SupplyLayer::Failed => Ok(None),
            SupplyLayer::Waiting | SupplyLayer::Consumed => bail!(
                "dense supply layer {layer} is demanded on visit {} with no read queued or in flight",
                self.reads[layer] + 1
            ),
        }
    }

    /// Ends the layer's entry without an upload: a queued read leaves `pending`, a published
    /// slot frees. `false` when the layer holds no entry.
    fn release(&mut self, layer: usize) -> bool {
        if matches!(
            self.layer[layer],
            SupplyLayer::Waiting | SupplyLayer::Consumed
        ) {
            return false;
        }
        if let SupplyLayer::Ready(slot) = self.layer[layer] {
            self.free[slot] = true;
        }
        if matches!(self.layer[layer], SupplyLayer::Queued) {
            self.pending.retain(|queued| *queued != layer);
        }
        self.layer[layer] = SupplyLayer::Consumed;
        self.outstanding = self
            .outstanding
            .checked_sub(1)
            .expect("dense supply entry without outstanding count");
        true
    }
}

/// State shared between the dense consumer and the supply workers.
#[cfg(any(feature = "cuda", feature = "rocm"))]
struct SupplyShared {
    ranges: Vec<Vec<SupplyRange>>,
    source_bytes: Vec<usize>,
    slot_base: Vec<*mut u8>,
    readers: BTreeMap<String, crate::experts::disk::SpanReader>,
    events: Vec<crate::backend::gpu::CudaEvent>,
    context: Arc<CudaContext>,
    state: std::sync::Mutex<SupplyState>,
    changed: std::sync::Condvar,
    abort: std::sync::atomic::AtomicBool,
}

// SAFETY: a slot's base is written only by its owning worker between the free claim and
// the ready publish, and read only by the consumer between demand and retire.
unsafe impl Send for SupplyShared {}
unsafe impl Sync for SupplyShared {}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl SupplyShared {
    /// Stores the abort flag under the state mutex, then wakes every waiter: a worker that has
    /// tested the flag and not yet waited holds that mutex, so it cannot miss the wakeup.
    fn stop(&self) {
        let state = self.state.lock().expect("dense supply lock");
        self.abort.store(true, std::sync::atomic::Ordering::SeqCst);
        drop(state);
        self.changed.notify_all();
    }
}

/// Concurrent disk supply of a dense streaming tier: `workers` threads read whole disk
/// layers through the aligned direct reader into per-layer pinned slots; the consumer
/// uploads from a slot and retires it by event, and a worker reuses a slot only after its
/// event completed. A failed read replays on every demand; `Drop` joins the workers and
/// synchronizes the context before the slots free.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct LayerSupply {
    shared: Arc<SupplyShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Registered slot buffers; the destructor unregisters them after the workers joined and
    /// the context synchronized.
    registrations: SupplySlots,
}

/// Registered slot buffers of a dense supply. `Drop` unregisters every address before its
/// buffer frees: along a failed `LayerSupply::spawn` and at the supply's end.
#[cfg(any(feature = "cuda", feature = "rocm"))]
struct SupplySlots {
    context: Arc<CudaContext>,
    buffers: Vec<(usize, crate::experts::disk::AlignedBytes)>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl SupplySlots {
    /// Unregisters every buffer; a call after the first finds none.
    fn unregister(&mut self) {
        if self.buffers.is_empty() {
            return;
        }
        let context = &self.context;
        context.record_err(context.bind_to_thread());
        for (address, _) in self.buffers.drain(..) {
            // SAFETY: `LayerSupply::spawn` registered each address once; it is unregistered
            // here once, before its buffer frees.
            #[cfg(feature = "cuda")]
            context.record_err(
                unsafe {
                    cudarc::driver::sys::cuMemHostUnregister(address as *mut std::ffi::c_void)
                }
                .result(),
            );
            #[cfg(feature = "rocm")]
            context.record_err(unsafe { context.unregister_host(address as u64) });
            #[cfg(test)]
            SUPPLY_UNREGISTERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl Drop for SupplySlots {
    fn drop(&mut self) {
        self.unregister();
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl LayerSupply {
    fn spawn(
        workers: usize,
        ranges: Vec<Vec<SupplyRange>>,
        source_bytes: Vec<usize>,
        slot_bytes: usize,
        dir: &Path,
        context: &Arc<CudaContext>,
    ) -> Result<Self> {
        ensure!(workers > 0, "dense supply worker count is zero");
        ensure!(
            ranges.len() == source_bytes.len(),
            "dense supply ranges and sizes differ"
        );
        let disk_layers = ranges.iter().filter(|ranges| !ranges.is_empty()).count();
        ensure!(disk_layers > 0, "dense supply holds no disk layers");
        let slot_count = workers.min(disk_layers);
        ensure!(slot_bytes > 0, "dense supply slot bytes is zero");
        let mut readers = BTreeMap::new();
        for shard in ranges
            .iter()
            .flatten()
            .map(|range| range.shard.as_ref())
            .collect::<BTreeSet<_>>()
        {
            readers.insert(
                shard.to_string(),
                crate::experts::disk::SpanReader::open(&dir.join(shard), true)?,
            );
        }
        let alignment = readers
            .values()
            .map(|reader| reader.alignment as usize)
            .max()
            .unwrap_or(4096);
        context.bind_to_thread()?;
        let mut slots = SupplySlots {
            context: context.clone(),
            buffers: Vec::with_capacity(slot_count),
        };
        let mut slot_base = Vec::with_capacity(slot_count);
        let mut events = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let mut buffer = crate::experts::disk::AlignedBytes::new(slot_bytes, alignment)?;
            let base = buffer.as_mut().as_mut_ptr();
            // SAFETY: the buffer lives in `slots` until a drop of it unregisters the address
            // before the buffer frees.
            #[cfg(feature = "cuda")]
            unsafe {
                cudarc::driver::sys::cuMemHostRegister_v2(base.cast(), slot_bytes, 0)
                    .result()
                    .context("register dense supply slot")?;
            }
            #[cfg(feature = "rocm")]
            unsafe {
                context
                    .register_host(base as u64, slot_bytes, false)
                    .context("register dense supply slot")?;
            }
            slots.buffers.push((base as usize, buffer));
            slot_base.push(base);
            events.push(context.new_event(None)?);
        }
        let state = SupplyState {
            layer: vec![SupplyLayer::Waiting; ranges.len()],
            reads: vec![0; ranges.len()],
            pending: std::collections::VecDeque::new(),
            free: vec![true; slot_count],
            outstanding: 0,
            failure: None,
            published: 0,
            retired: 0,
        };
        let shared = Arc::new(SupplyShared {
            ranges,
            source_bytes,
            slot_base,
            readers,
            events,
            context: context.clone(),
            state: std::sync::Mutex::new(state),
            changed: std::sync::Condvar::new(),
            abort: std::sync::atomic::AtomicBool::new(false),
        });
        let mut handles = Vec::with_capacity(slot_count);
        for slot in 0..slot_count {
            match spawn_supply_worker(&shared, slot, alignment) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    shared.stop();
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            shared,
            handles,
            registrations: slots,
        })
    }

    /// Admits layers from the demand cursor while capacity holds, for at most one turn of the
    /// cyclic `chain` from the cursor's position. The cursor moves only past layers that are
    /// admitted or already in flight, so it stops at the first layer capacity holds.
    fn pump(&self, cursor: &mut Option<usize>, chain: impl Fn(usize) -> Option<usize>) {
        let capacity = self.handles.len();
        let turn = *cursor;
        let mut state = self.shared.state.lock().expect("dense supply lock");
        while let Some(index) = *cursor {
            if !state.admit(index, capacity) {
                break;
            }
            *cursor = chain(index);
            if *cursor == turn {
                break;
            }
        }
        drop(state);
        self.shared.changed.notify_all();
    }

    /// Waits the layer's pinned slot. The bytes stay live until [`Self::retire`]. A layer with
    /// no read queued or in flight errors.
    fn demand(&self, layer: usize) -> Result<(*const u8, usize)> {
        let mut state = self.shared.state.lock().expect("dense supply lock");
        loop {
            if let Some(failure) = &state.failure {
                bail!("dense supply read failed: {failure}");
            }
            if let Some(slot) = state.ready(layer)? {
                return Ok((self.shared.slot_base[slot] as *const u8, slot));
            }
            state = self.shared.changed.wait(state).expect("dense supply lock");
        }
    }

    /// Retires the layer's slot after its upload was issued on `stream`.
    fn retire(&self, layer: usize, slot: usize, stream: &Arc<CudaStream>) -> Result<()> {
        self.shared.events[slot].record(stream)?;
        let mut state = self.shared.state.lock().expect("dense supply lock");
        state.release(layer);
        state.retired = state
            .retired
            .checked_add(1)
            .expect("dense supply retired overflow");
        drop(state);
        self.shared.changed.notify_all();
        Ok(())
    }

    /// Ends the read-ahead entry of a layer the consumer finds loaded in its device slot, so
    /// no upload reads it. An entry whose read is in flight is waited for.
    fn release(&self, layer: usize) -> Result<()> {
        let mut state = self.shared.state.lock().expect("dense supply lock");
        while matches!(state.layer[layer], SupplyLayer::Reading) && state.failure.is_none() {
            state = self.shared.changed.wait(state).expect("dense supply lock");
        }
        if let Some(failure) = &state.failure {
            bail!("dense supply read failed: {failure}");
        }
        let released = state.release(layer);
        drop(state);
        if released {
            self.shared.changed.notify_all();
        }
        Ok(())
    }

    /// Waits until no layer is queued or reading; returns published and retired counts.
    #[cfg(test)]
    fn flush_counts(&self) -> Result<(u64, u64)> {
        let mut state = self.shared.state.lock().expect("dense supply lock");
        loop {
            if let Some(failure) = &state.failure {
                bail!("dense supply read failed: {failure}");
            }
            if state.pending.is_empty()
                && state
                    .layer
                    .iter()
                    .all(|layer| !matches!(layer, SupplyLayer::Reading))
            {
                let counts = (state.published, state.retired);
                return Ok(counts);
            }
            state = self.shared.changed.wait(state).expect("dense supply lock");
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl Drop for LayerSupply {
    fn drop(&mut self) {
        self.shared.stop();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        let context = &self.shared.context;
        context.record_err(context.synchronize());
        self.registrations.unregister();
    }
}

/// Test park of a supply worker between its abort check and its wait, in microseconds.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
static SUPPLY_WAIT_PARK_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Test slot whose supply worker's spawn fails; `usize::MAX` for none.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
static SUPPLY_SPAWN_FAIL_SLOT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Test count of supply slot addresses passed to the unregister call.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
static SUPPLY_UNREGISTERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Spawns supply worker `slot`. A test refuses the worker that `SUPPLY_SPAWN_FAIL_SLOT` names.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn spawn_supply_worker(
    shared: &Arc<SupplyShared>,
    slot: usize,
    alignment: usize,
) -> Result<std::thread::JoinHandle<()>> {
    #[cfg(test)]
    if slot == SUPPLY_SPAWN_FAIL_SLOT.load(std::sync::atomic::Ordering::SeqCst) {
        bail!("the spawn of dense supply worker {slot} is refused");
    }
    std::thread::Builder::new()
        .name(format!("dense-supply-{slot}"))
        .spawn({
            let shared = Arc::clone(shared);
            move || supply_worker(shared, slot, alignment)
        })
        .context("spawn dense supply worker")
}

/// One supply worker: claims the next queued layer into its slot, waits the slot's
/// previous upload event, reads the layer's ranges through the aligned direct reader,
/// and publishes the slot or records the failure.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn supply_worker(shared: Arc<SupplyShared>, slot: usize, alignment: usize) {
    if let Err(error) = shared.context.bind_to_thread() {
        let mut state = shared.state.lock().expect("dense supply lock");
        state.failure = Some(format!("{error:#}"));
        shared.changed.notify_all();
        return;
    }
    let mut bounce = crate::experts::disk::AlignedBytes::new(alignment, alignment)
        .expect("dense supply bounce allocation");
    loop {
        let layer = {
            let mut state = shared.state.lock().expect("dense supply lock");
            loop {
                if shared.abort.load(std::sync::atomic::Ordering::SeqCst) || state.failure.is_some()
                {
                    return;
                }
                if let Some(index) = state.pending.front().copied()
                    && matches!(state.layer[index], SupplyLayer::Queued)
                    && state.free[slot]
                {
                    state.pending.pop_front();
                    state.layer[index] = SupplyLayer::Reading;
                    state.free[slot] = false;
                    break index;
                }
                #[cfg(test)]
                if SUPPLY_WAIT_PARK_MICROS.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                    std::thread::sleep(std::time::Duration::from_micros(
                        SUPPLY_WAIT_PARK_MICROS.load(std::sync::atomic::Ordering::SeqCst),
                    ));
                }
                state = shared.changed.wait(state).expect("dense supply lock");
            }
        };
        let outcome = (|| -> Result<()> {
            shared.events[slot].synchronize()?;
            read_supply_layer(&shared, layer, slot, &mut bounce, alignment)
        })();
        let mut state = shared.state.lock().expect("dense supply lock");
        match outcome {
            Ok(()) => {
                state.layer[layer] = SupplyLayer::Ready(slot);
                state.reads[layer] = state.reads[layer]
                    .checked_add(1)
                    .expect("dense supply read count overflow");
                state.published = state
                    .published
                    .checked_add(1)
                    .expect("dense supply published overflow");
            }
            Err(error) => {
                state.layer[layer] = SupplyLayer::Failed;
                state.failure = Some(format!("{error:#}"));
            }
        }
        drop(state);
        shared.changed.notify_all();
    }
}

/// Reads the layer's ranges through the aligned direct reader into its slot.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn read_supply_layer(
    shared: &SupplyShared,
    layer: usize,
    slot: usize,
    bounce: &mut crate::experts::disk::AlignedBytes,
    alignment: usize,
) -> Result<()> {
    let bytes = shared.source_bytes[layer];
    // SAFETY: the worker owns the slot between its claim and the ready publish.
    let destination = unsafe { std::slice::from_raw_parts_mut(shared.slot_base[slot], bytes) };
    for range in &shared.ranges[layer] {
        let reader = shared
            .readers
            .get(range.shard.as_ref())
            .with_context(|| format!("dense layer {layer}: shard not open"))?;
        let head = (range.offset as usize) % alignment;
        let begin = range
            .offset
            .checked_sub(head as u64)
            .context("dense supply aligned begin underflow")?;
        let width = range
            .len
            .checked_add(head as u64)
            .and_then(|bytes| bytes.checked_add((alignment - 1) as u64))
            .context("dense supply span overflow")?
            & !((alignment - 1) as u64);
        let width = width
            .min(reader.file_len().saturating_sub(begin))
            .max(range.len + head as u64) as usize;
        if bounce.len() < width + alignment {
            *bounce = crate::experts::disk::AlignedBytes::new(width + alignment, alignment)?;
        }
        let span = bounce.as_mut();
        reader.read_direct(begin, width as u64, span)?;
        let source = &span[head..head + range.len as usize];
        destination[range.source_offset..range.source_offset + range.len as usize]
            .copy_from_slice(source);
    }
    Ok(())
}

pub(crate) struct DenseStreaming {
    layers: Vec<DenseLayer>,
    slots: Vec<DenseSlot>,
    supply: Option<LayerSupply>,
    next_disk_read: Option<usize>,
    upload: Arc<CudaStream>,
    staging: Option<CudaSlice<u8>>,
    quant: Option<QuantKernels>,
    narrow_f32: Option<CudaFunction>,
    pub plan: crate::prefill::DensePlacement,
    /// sm below 80. Disk reads of BF16 weights are rewritten to FP16 before the copy.
    fp16_weights: bool,
    fp16_scratch: Vec<u8>,
}

impl Drop for DenseStreaming {
    fn drop(&mut self) {
        let context = self.upload.context();
        let _ = self.supply.take();
        context.record_err(context.synchronize());
    }
}

impl DenseStreaming {
    pub(crate) fn new(
        dir: &Path,
        manifest: &TensorManifest,
        plan: crate::prefill::DensePlacement,
        d: &Device,
        order: Option<&[Vec<String>]>,
    ) -> Result<Self> {
        Self::allocate(dir, Some((manifest, order)), plan, d, None, None)
    }

    pub(crate) fn with_f32_cast(
        dir: &Path,
        manifest: &TensorManifest,
        plan: crate::prefill::DensePlacement,
        d: &Device,
        order: &[Vec<String>],
        narrow: fn(&str) -> bool,
    ) -> Result<Self> {
        Self::allocate(
            dir,
            Some((manifest, Some(order))),
            plan,
            d,
            None,
            Some(narrow),
        )
    }

    pub(crate) fn new_preloaded(
        dir: &Path,
        plan: crate::prefill::DensePlacement,
        d: &Device,
        layer_bytes: &[usize],
    ) -> Result<Self> {
        ensure!(
            plan.disk_bytes == 0 && layer_bytes.len() == plan.slots.len(),
            "preloaded layer layout differs from host placement"
        );
        Self::allocate(dir, None, plan, d, Some(layer_bytes), None)
    }

    fn allocate(
        dir: &Path,
        input: Option<(&TensorManifest, Option<&[Vec<String>]>)>,
        plan: crate::prefill::DensePlacement,
        d: &Device,
        preloaded: Option<&[usize]>,
        narrow: Option<fn(&str) -> bool>,
    ) -> Result<Self> {
        use crate::experts::catalogue::NativeRange;
        let fp16_weights = d.info.arch.fp16_stored_half();
        let mut layers: Vec<_> = plan
            .slots
            .iter()
            .map(|&slot| DenseLayer {
                slot,
                next: None,
                next_disk: None,
                bytes: 0,
                source_bytes: 0,
                needs_staging: false,
                entries: Vec::new(),
                host: None,
                checked: false,
            })
            .collect();
        let mut ranges = vec![Vec::new(); layers.len()];
        let mut add = |name: &String, layer_index: usize, entry: &TensorEntry| -> Result<()> {
            let layer = &mut layers[layer_index];
            if layer.slot.is_none() {
                return Ok(());
            }
            let device_offset = layer.bytes;
            let source_offset = layer.source_bytes;
            let narrow_f32 = narrow.is_some_and(|narrow| narrow(name));
            let device_bytes = if narrow_f32 {
                ensure!(
                    entry.dtype == Dtype::F32
                        && entry.quant.is_none()
                        && entry.bytes.is_multiple_of(4),
                    "{name}: F32 to BF16 stream source"
                );
                entry.bytes / 2
            } else {
                device_slot_bytes(name, entry)?
            };
            layer.needs_staging |= narrow_f32 || dequantizes(entry.quant);
            layer.bytes = layer
                .bytes
                .checked_add(device_bytes)
                .context("dense layer bytes overflow")?;
            layer.source_bytes = source_offset
                .checked_add(entry.bytes)
                .context("dense layer source bytes overflow")?;
            layer.entries.push(DenseEntry {
                name: name.clone(),
                device_offset,
                source_offset,
                entry: entry.clone(),
                narrow_f32,
            });
            if !plan.host[layer_index] {
                let offset = entry
                    .data_start
                    .checked_add(entry.data_offsets[0])
                    .context("dense source offset overflow")?;
                ranges[layer_index].push((
                    u64::try_from(source_offset)?,
                    NativeRange {
                        shard: Arc::from(entry.shard.as_str()),
                        offset,
                        len: entry.bytes as u64,
                    },
                ));
            }
            Ok(())
        };
        if let Some((manifest, Some(order))) = input {
            for (layer_index, names) in order.iter().enumerate() {
                for name in names {
                    let entry = manifest
                        .entries
                        .get(name)
                        .with_context(|| format!("dense ordered tensor {name} missing"))?;
                    let stored = entry.is_linear_projection()
                        || (entry.load
                            && entry.shape.len() == 2
                            && entry.role.ends_with(".qweight"));
                    ensure!(
                        (stored && entry.layer == Some(layer_index as u32))
                            || (entry.load
                                && entry.layer.is_none()
                                && entry.quant.is_none()
                                && entry.dtype == safetensors::Dtype::F32),
                        "dense ordered tensor {name} is outside layer {layer_index}"
                    );
                    add(name, layer_index, entry)?;
                }
            }
        } else if let Some((manifest, None)) = input {
            for (name, entry) in manifest
                .entries
                .iter()
                .filter(|(_, entry)| entry.is_linear_projection() && entry.layer.is_some())
            {
                let layer_index = entry.layer.context("dense layer missing")? as usize;
                add(name, layer_index, entry)?;
            }
        } else if let Some(sizes) = preloaded {
            for (i, (layer, &bytes)) in layers.iter_mut().zip(sizes).enumerate() {
                if layer.slot.is_some() {
                    ensure!(
                        plan.host[i] && bytes > 0 && bytes <= plan.slot_bytes,
                        "preloaded layer {i} exceeds its pinned slot"
                    );
                    layer.bytes = bytes;
                    layer.source_bytes = bytes;
                }
            }
        }
        let staging_bytes = layers
            .iter()
            .filter(|layer| layer.needs_staging)
            .map(|layer| layer.source_bytes)
            .max()
            .unwrap_or(0);
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
        #[cfg(test)]
        let mut allocation_before = d.ctx.mem_get_info()?.0;
        #[cfg(test)]
        let mut allocation_record =
            |phase: &str, logical_bytes: usize, host_bytes: usize| -> Result<()> {
                let after = d.ctx.mem_get_info()?.0;
                eprintln!(
                    "{}",
                    serde_json::json!({"dense_allocation": {
                        "phase": phase, "logical_bytes": logical_bytes, "host_bytes": host_bytes,
                        "free_before": allocation_before, "free_after": after,
                        "synchronized": false
                    }})
                );
                allocation_before = after;
                Ok(())
            };
        for (index, layer) in layers.iter_mut().enumerate() {
            if plan.host[index] {
                layer.host =
                    Some(unsafe { d.ctx.alloc_pinned_with_flags::<u8>(layer.source_bytes, 0)? });
                #[cfg(test)]
                allocation_record(&format!("pinned_layer{index}"), 0, layer.source_bytes)?;
            }
        }
        let slot_count = plan
            .slots
            .iter()
            .flatten()
            .copied()
            .max()
            .map(|index| index + 1)
            .context("dense streaming plan has no slot")?;
        let supply = if plan.disk_bytes > 0 {
            let disk_slot_bytes = layers
                .iter()
                .enumerate()
                .filter(|(index, layer)| layer.slot.is_some() && !plan.host[*index])
                .map(|(_, layer)| layer.source_bytes)
                .max()
                .context("dense disk plan has no disk layers")?;
            let supply_ranges = ranges
                .into_iter()
                .map(|mut layer_ranges| {
                    layer_ranges.sort_by_key(|(_, native)| (native.shard.clone(), native.offset));
                    layer_ranges
                        .into_iter()
                        .map(|(source_offset, native)| {
                            Ok(SupplyRange {
                                source_offset: usize::try_from(source_offset)
                                    .context("dense supply source offset")?,
                                shard: native.shard,
                                offset: native.offset,
                                len: native.len,
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let source_bytes = layers
                .iter()
                .map(|layer| layer.source_bytes)
                .collect::<Vec<_>>();
            let manifest = input
                .map(|(manifest, _)| manifest)
                .context("dense supply needs a manifest")?;
            let workers = super::engine::minimax_h3::dense_supply_workers(dir, manifest)?;
            Some(LayerSupply::spawn(
                workers,
                supply_ranges,
                source_bytes,
                disk_slot_bytes,
                dir,
                &d.ctx,
            )?)
        } else {
            None
        };
        #[cfg(test)]
        allocation_record("disk_reader_events", 0, 0)?;
        let slots = (0..slot_count)
            .map(|_| {
                let device = d.upload.alloc_zeros::<u8>(plan.slot_bytes)?;
                #[cfg(test)]
                allocation_record("slot_tensor", plan.slot_bytes, 0)?;
                let ready = d
                    .ctx
                    .new_event(Some(crate::backend::gpu::EVENT_DISABLE_TIMING))?;
                #[cfg(test)]
                allocation_record("slot_ready_event", 0, 0)?;
                let retired = d
                    .ctx
                    .new_event(Some(crate::backend::gpu::EVENT_DISABLE_TIMING))?;
                #[cfg(test)]
                allocation_record("slot_retired_event", 0, 0)?;
                Ok(DenseSlot {
                    device,
                    ready,
                    retired,
                    retired_valid: false,
                    loaded: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let staging = (staging_bytes > 0)
            .then(|| d.upload.alloc_zeros::<u8>(staging_bytes))
            .transpose()?;
        let entries = || layers.iter().flat_map(|layer| &layer.entries);
        let quant = entries()
            .any(|entry| dequantizes(entry.entry.quant))
            .then(|| QuantKernels::new(&d.ctx))
            .transpose()?;
        let narrow_f32 = if entries().any(|entry| entry.narrow_f32) {
            let symbol = d.info.arch.decoder_f32_narrow_symbol();
            Some(super::cubin::module(&d.ctx, "vision_ops")?.load_function(symbol)?)
        } else {
            None
        };
        #[cfg(test)]
        allocation_record("staging_and_quant_modules", staging_bytes, 0)?;
        Ok(Self {
            layers,
            slots,
            supply,
            next_disk_read: first_disk,
            upload: d.upload.clone(),
            staging,
            quant,
            narrow_f32,
            plan,
            fp16_weights,
            fp16_scratch: Vec::new(),
        })
    }

    #[cfg(any(feature = "cuda", test))]
    fn defer(
        &mut self,
        loader: &mut Loader<'_>,
        name: &str,
        entry: &TensorEntry,
        layer: usize,
        offset: usize,
    ) -> Result<()> {
        loader.defer(name)?;
        ensure!(
            entry.quant.is_none(),
            "tensor {name}: the loader dense path holds BF16 only, not a quantized entry"
        );
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

    #[cfg(any(feature = "cuda", test))]
    fn prime(&mut self) -> Result<()> {
        for slot in 0..self.slots.len() {
            if let Some(first) = self
                .layers
                .iter()
                .position(|layer| layer.slot == Some(slot))
            {
                self.prefetch(first)?;
            }
        }
        self.pump_disk();
        Ok(())
    }

    fn pump_disk(&mut self) {
        let Some(supply) = self.supply.as_ref() else {
            return;
        };
        let layers = &self.layers;
        supply.pump(&mut self.next_disk_read, |index| layers[index].next_disk);
    }

    /// Waits every outstanding supply layer; test counts of published and retired slots.
    #[cfg(test)]
    pub(crate) fn flush_ring_counts(&mut self) -> Result<(u64, u64)> {
        self.upload.synchronize()?;
        let (published, retired) = match self.supply.as_ref() {
            Some(supply) => supply.flush_counts()?,
            None => (0, 0),
        };
        self.pump_disk();
        Ok((published, retired))
    }

    /// Test control: shift one layer entry's source offset by `delta` bytes.
    #[cfg(test)]
    pub(crate) fn shift_source_offset(
        &mut self,
        layer: usize,
        name: &str,
        delta: isize,
    ) -> Result<()> {
        let entry = self
            .layers
            .get_mut(layer)
            .context("dense layer outside plan")?
            .entries
            .iter_mut()
            .find(|entry| entry.name == name)
            .context("dense entry missing")?;
        entry.source_offset = entry
            .source_offset
            .checked_add_signed(delta)
            .context("dense source offset overflow")?;
        Ok(())
    }

    /// Pinned host copy of a host-tier layer, its linear projections packed in manifest-name order; `None` for
    /// resident and disk-tier layers. Callers outside the loader fill it before the first `begin`.
    pub(crate) fn host_layer(&mut self, layer: usize) -> Result<Option<&mut [u8]>> {
        let binding = self
            .layers
            .get_mut(layer)
            .context("dense layer outside plan")?;
        Ok(match binding.host.as_mut() {
            Some(host) => Some(host.as_mut_slice()?),
            None => None,
        })
    }

    /// Worker count of the disk supply; 0 for a tier without disk layers.
    pub(crate) fn supply_workers(&self) -> usize {
        self.supply
            .as_ref()
            .map_or(0, |supply| supply.handles.len())
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
            if self.layers[index].host.is_none() {
                self.supply
                    .as_ref()
                    .context("dense disk supply missing at a loaded disk layer")?
                    .release(index)?;
            }
            self.pump_disk();
            return Ok(());
        }
        if self.slots[slot_index].retired_valid {
            self.upload.wait(&self.slots[slot_index].retired)?;
        }
        let source_bytes = self.layers[index].source_bytes;
        let check_source = !self.layers[index].checked;
        let needs_staging = self.layers[index].needs_staging;
        let pointer = self.slots[slot_index].device.device_ptr(&self.upload).0;
        let staging = needs_staging
            .then(|| {
                self.staging
                    .as_ref()
                    .context("converted layer without a staging buffer")
                    .map(|staging| staging.device_ptr(&self.upload).0)
            })
            .transpose()?;
        let source_pointer = staging.unwrap_or(pointer);
        if let Some(host) = &self.layers[index].host {
            let source = &host.as_slice()?[..source_bytes];
            if check_source {
                check_layer_source(&self.layers[index], source)?;
            }
            unsafe {
                super::copy::htod_async(source_pointer, source, &self.upload)?;
            }
        } else {
            self.pump_disk();
            let (base, slot) = self
                .supply
                .as_ref()
                .context("dense disk supply missing after demand")?
                .demand(index)?;
            // SAFETY: the slot holds `source_bytes` initialized bytes until the retire.
            let raw = unsafe { std::slice::from_raw_parts(base, source_bytes) };
            if check_source {
                check_layer_source(&self.layers[index], raw)?;
            }
            let fp16 = self.fp16_weights;
            if fp16 {
                // The previous HtoD from fp16_scratch has finished before that buffer is reused.
                self.upload.synchronize()?;
            }
            let mut scratch = if fp16 {
                std::mem::take(&mut self.fp16_scratch)
            } else {
                Vec::new()
            };
            if fp16 {
                if scratch.len() < source_bytes {
                    scratch.resize(source_bytes, 0);
                }
                scratch[..source_bytes].copy_from_slice(raw);
                rewrite_fp16_weights(
                    &self.layers[index],
                    &mut scratch[..source_bytes],
                    check_source,
                )?;
            }
            let payload = if fp16 { &scratch[..source_bytes] } else { raw };
            unsafe {
                super::copy::htod_async(source_pointer, payload, &self.upload)?;
            }
            if fp16 {
                self.fp16_scratch = scratch;
            }
            self.supply
                .as_ref()
                .context("dense disk supply missing after demand")?
                .retire(index, slot, &self.upload)?;
        }
        if check_source {
            self.layers[index].checked = true;
        }
        if needs_staging {
            for entry in &self.layers[index].entries {
                let destination = pointer + entry.device_offset as u64;
                if entry.narrow_f32 {
                    let count = u32::try_from(entry.entry.bytes / 4)?;
                    unsafe {
                        self.upload
                            .launch_builder(
                                self.narrow_f32
                                    .as_ref()
                                    .context("missing F32 cast kernel")?,
                            )
                            .arg(&(source_pointer + entry.source_offset as u64))
                            .arg(&destination)
                            .arg(&count)
                            .launch(super::ops::flat(count as usize))?;
                    }
                } else if dequantizes(entry.entry.quant) {
                    let quant = self.quant.as_ref().context("missing dequant kernels")?;
                    let elements = entry.entry.shape[0] * entry.entry.shape[1];
                    quant.dequant(
                        &self.upload,
                        &entry.name,
                        &entry.entry,
                        DequantCopy {
                            source: source_pointer + entry.source_offset as u64,
                            destination,
                            elements,
                            dtype: entry.entry.dtype,
                        },
                    )?;
                } else {
                    unsafe {
                        super::copy::dtod_raw(
                            destination,
                            source_pointer + entry.source_offset as u64,
                            entry.entry.bytes,
                            &self.upload,
                        )?;
                    }
                }
            }
        }
        self.slots[slot_index].ready.record(&self.upload)?;
        self.slots[slot_index].loaded = Some(index);
        self.pump_disk();
        Ok(())
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
                (
                    crate::checkpoint::GLM_MLP_GATE_PROJ,
                    crate::checkpoint::GLM_MLP_UP_PROJ
                ) | (
                    crate::checkpoint::GLM_ATTN_Q_PROJ,
                    crate::checkpoint::GLM_ATTN_V_PROJ
                )
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

#[cfg(feature = "cuda")]
fn emit_classic_streamed(
    loader: &mut Loader<'_>,
    tensors: &mut BTreeMap<String, StoredTensor>,
    dense: &mut DenseStreaming,
    manifest: &TensorManifest,
    order: &[String],
    offsets: &mut [usize],
    layer: usize,
) -> Result<()> {
    classic_fused_specs(manifest, order, layer)?;
    let mut qkv: Option<(u64, usize)> = None;
    let mut gu: Option<(u64, usize)> = None;
    for (index, name) in order.iter().enumerate() {
        let entry = manifest
            .entries
            .get(name)
            .with_context(|| format!("classic tensor {name} missing"))?;
        let offset = offsets[layer];
        let pointer = dense.pointer(layer, offset, entry.bytes)?;
        offsets[layer] = offset
            .checked_add(entry.bytes)
            .context("classic dense offset overflow")?;
        dense.defer(loader, name, entry, layer, offset)?;
        match index {
            0 => qkv = Some((pointer, entry.bytes)),
            1 | 2 => {
                let (base, bytes) = qkv.context("classic qkv group missing")?;
                qkv = Some((base, bytes + entry.bytes));
            }
            3 => gu = Some((pointer, entry.bytes)),
            4 => {
                let (base, bytes) = gu.context("classic gate/up group missing")?;
                gu = Some((base, bytes + entry.bytes));
            }
            5 | 6 => {
                tensors.insert(
                    name.clone(),
                    StoredTensor::Slot {
                        pointer,
                        bytes: entry.bytes,
                    },
                );
            }
            _ => unreachable!("classic layer has seven projections"),
        }
    }
    let (qkv_pointer, qkv_bytes) = qkv.context("classic qkv group missing")?;
    let (gu_pointer, gu_bytes) = gu.context("classic gate/up group missing")?;
    tensors.insert(
        classic_qkv_name(layer),
        StoredTensor::Slot {
            pointer: qkv_pointer,
            bytes: qkv_bytes,
        },
    );
    tensors.insert(
        classic_gu_name(layer),
        StoredTensor::Slot {
            pointer: gu_pointer,
            bytes: gu_bytes,
        },
    );
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) struct PcieLink {
    pub width: usize,
    pub max_speed: String,
}

#[cfg(feature = "cuda")]
pub(crate) fn sense_pcie(d: &Device) -> Result<PcieLink> {
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
    let width: usize = std::fs::read_to_string(pci.join("current_link_width"))
        .context("probe dense PCIe link width")?
        .trim()
        .parse()?;
    let max_speed = link_speed.trim().to_owned();
    ensure!(
        width > 0 && !max_speed.is_empty(),
        "invalid dense PCIe capability"
    );
    crate::prefill::pcie_payload_bytes_per_second(&max_speed, width)?;
    Ok(PcieLink { width, max_speed })
}

pub(crate) struct WeightCatalog {
    #[cfg(feature = "cuda")]
    pub dense: Option<DenseStreaming>,
    pub classic: Option<ClassicCatalog>,
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
    #[cfg(any(feature = "cuda", test))]
    fn dense_host_fixed(
        layers: &[usize],
        ranges: &[(usize, u64, u64, u64)],
        budget: usize,
        available_host: usize,
        supply_workers: usize,
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
        let base_reserve = 4usize << 30;
        if available_host >= base_reserve {
            let host_only =
                DensePlacement::new(layers, None, budget, available_host - base_reserve)?;
            if host_only.disk_bytes == 0 {
                return Ok((base_reserve, host_only));
            }
        }
        // The charge assumed never falls below the charge the plan needs: one slot and one
        // largest aligned extent per worker the plan's disk layers spawn.
        let mut host_fixed = base_reserve;
        let passes = (layers.len() + 1)
            .checked_mul(supply_workers + 1)
            .context("dense host staging overflow")?;
        for _ in 0..=passes {
            ensure!(
                available_host >= host_fixed,
                "dense pinned staging and host reserve exceed available RAM: need={host_fixed} available={available_host}"
            );
            let plan = DensePlacement::new(layers, None, budget, available_host - host_fixed)?;
            let disk: Vec<usize> = (0..layers.len())
                .filter(|&i| plan.slots[i].is_some() && !plan.host[i])
                .collect();
            let required = crate::prefill::supply_reserve((
                disk.iter().map(|&layer| layers[layer]).max().unwrap_or(0),
                disk.iter()
                    .map(|&layer| layer_staging[layer])
                    .max()
                    .unwrap_or(0),
                disk.len(),
                supply_workers,
            ))?
            .checked_add(base_reserve)
            .context("dense host staging overflow")?;
            if required <= host_fixed {
                return Ok((host_fixed, plan));
            }
            host_fixed = required;
        }
        anyhow::bail!("dense disk staging extent did not converge")
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dense_fixed_bytes(manifest: &crate::checkpoint::TensorManifest) -> Result<usize> {
        manifest
            .entries
            .values()
            .filter(|entry| entry.load)
            .filter(|entry| !(entry.is_linear_projection() && entry.layer.is_some()))
            .try_fold(0usize, |bytes, entry| {
                bytes
                    .checked_add(entry.bytes)
                    .context("dense fixed weights overflow")
            })
    }

    /// Streams the layers of `dir` that do not fit the device budget. `runtime` is the
    /// directory the disk supply's worker count is seeded from.
    #[cfg(feature = "cuda")]
    pub(crate) fn load_dense(
        (dir, runtime): (&Path, &Path),
        spec: &crate::model_spec::ModelSpec,
        d: &Device,
        transient_bytes: usize,
        media_bytes: usize,
        prefix_cache: bool,
        host_bytes: usize,
    ) -> Result<Self> {
        use crate::prefill::DensePlacement;
        ensure!(
            spec.moe.is_none() && spec.linear_format == crate::model_spec::LinearFormat::Bf16,
            "dense layer streaming requires BF16 dense weights"
        );
        let mapped = MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let order = classic_order(dir, spec)?;
        let mut layers = vec![0usize; spec.layers.len()];
        let mut shard_alignments = BTreeMap::new();
        let mut ranges = Vec::new();
        let fixed = Self::dense_fixed_bytes(&manifest)?;
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
            }
        }
        let (free, total) = d.ctx.mem_get_info()?;
        let budget =
            DensePlacement::device_budget(free, total, fixed, transient_bytes, media_bytes)?;
        let preliminary = DensePlacement::new(&layers, None, budget, 0)?;
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
        let available_host = available_host
            .checked_sub(host_bytes)
            .context("dense host feature reservation exceeds available RAM")?;
        let (_, _, supply_workers) =
            super::engine::minimax_h3::seed_dense_supply_workers(dir, &manifest, runtime)?;
        let (_, plan) =
            Self::dense_host_fixed(&layers, &ranges, budget, available_host, supply_workers)?;
        let memory_bus_bits = d
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let copy_engines = d
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?;
        let pcie = sense_pcie(d)?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"dense_placement": &plan, "fixed_bytes": fixed, "transient_bytes": transient_bytes, "host_available_bytes": available_host, "copy_engines": copy_engines, "memory_bus_bits": memory_bus_bits, "sms": d.info.sms, "pcie_link_width": pcie.width, "pcie_max_link_speed": pcie.max_speed})
            );
        }
        let started = Instant::now();
        let mut dense = DenseStreaming::new(dir, &manifest, plan, d, order.as_deref())?;
        let mut offsets = vec![0; layers.len()];
        let gemma = matches!(
            spec.architecture,
            crate::model_spec::Architecture::Gemma4
                | crate::model_spec::Architecture::EmbeddingGemma2
        );
        let ((tensors, uploaded), _) =
            checkpoint_mapped(&mapped, d, Some(&manifest), false, |loader| {
                let mut tensors = BTreeMap::new();
                let mut processed = BTreeSet::new();
                let mut iter = manifest.entries.iter().peekable();
                while let Some((name, entry)) = iter.next() {
                    if !entry.load {
                        continue;
                    }
                    if let Some(order) = &order
                        && entry.is_linear_projection()
                        && let Some(layer) = entry.layer
                    {
                        let layer = layer as usize;
                        if processed.insert(layer) {
                            if dense.plan.slots[layer].is_some() {
                                emit_classic_streamed(
                                    loader,
                                    &mut tensors,
                                    &mut dense,
                                    &manifest,
                                    &order[layer],
                                    &mut offsets,
                                    layer,
                                )?;
                            } else {
                                emit_classic_resident(
                                    loader,
                                    &mut tensors,
                                    &manifest,
                                    &order[layer],
                                    layer,
                                )?;
                            }
                        }
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
        Self {
            dense: Some(dense),
            classic: order.map(|order| ClassicCatalog {
                order,
                bindings: Vec::new(),
            }),
            linear_format: spec.linear_format,
            tensors,
            manifest,
            bytes,
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: elapsed_ms(started),
        }
        .bind_classic(&d.stream)
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
            .contains_key(crate::checkpoint::GLM_EMBED_TOKENS)
        {
            crate::checkpoint::GLM_EMBED_TOKENS
        } else if self.tensors.contains_key("model.embed_tokens.weight") {
            "model.embed_tokens.weight"
        } else {
            "language_model.embed_tokens.weight"
        };
        let norm_name = if self
            .tensors
            .contains_key(crate::checkpoint::GLM_LANGUAGE_NORM)
        {
            crate::checkpoint::GLM_LANGUAGE_NORM
        } else if self.tensors.contains_key("model.norm.weight") {
            "model.norm.weight"
        } else {
            "language_model.norm.weight"
        };
        let head_entry = self.tensors.get(crate::checkpoint::GLM_LM_HEAD);
        let native_head = matches!(head_entry, Some(StoredTensor::Bytes(_)));
        let has_head = head_entry.is_some();
        let mut take = |name: &str| match self.tensors.remove(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 global tensor {name}"),
        };
        let (embed, head) = if native_head {
            let embed = take(embed_name)?;
            let empty = embed.stream().alloc_zeros::<bf16>(0)?;
            (Some(embed), empty)
        } else if has_head {
            (
                Some(take(embed_name)?),
                take(crate::checkpoint::GLM_LM_HEAD)?,
            )
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

    pub(crate) fn classic_layers(&self) -> usize {
        self.classic
            .as_ref()
            .map_or(0, |classic| classic.order.len())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn classic_layer(&self, layer: usize) -> Result<&ClassicLayer> {
        self.classic
            .as_ref()
            .and_then(|classic| classic.bindings.get(layer))
            .with_context(|| format!("classic layer {layer} binding missing"))
    }

    #[cfg(feature = "cuda")]
    fn classic_norm_view(
        &self,
        layer: usize,
        kind: &str,
        stream: &Arc<CudaStream>,
    ) -> Result<super::contracts::Tensor<bf16, 1>> {
        let weight = self.bf16(&format!("model.layers.{layer}.{kind}.weight"))?;
        let elements = u32::try_from(weight.len())?;
        super::contracts::view(weight, stream, 0, [elements], [1])
    }

    #[cfg(feature = "cuda")]
    fn bind_classic(mut self, stream: &Arc<CudaStream>) -> Result<Self> {
        let bindings = (0..self.classic_layers())
            .map(|layer| self.classic_binding(layer, stream))
            .collect::<Result<Vec<_>>>()?;
        if let Some(classic) = self.classic.as_mut() {
            classic.bindings = bindings;
        }
        Ok(self)
    }

    #[cfg(feature = "cuda")]
    fn classic_weight(
        &self,
        name: &str,
        shape: [usize; 2],
    ) -> Result<super::linear::LinearWeightRef> {
        let rows = u32::try_from(shape[0])?;
        let width = u32::try_from(shape[1])?;
        let bytes = shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(2))
            .context("classic fused group byte extent overflow")?;
        match self
            .tensors
            .get(name)
            .with_context(|| format!("classic fused group {name} missing"))?
        {
            #[cfg(feature = "cuda")]
            StoredTensor::Slot {
                pointer,
                bytes: extent,
            } => {
                ensure!(
                    *extent == bytes,
                    "classic fused slot extent differs: {name}"
                );
                Ok(super::linear::LinearWeightRef::Bf16(unsafe {
                    super::contracts::Tensor::new(
                        *pointer,
                        [rows, width],
                        [u64::from(width), 1],
                        *extent as u64,
                    )?
                }))
            }
            StoredTensor::Bf16(values) => {
                ensure!(
                    values.len() * 2 == bytes,
                    "classic fused resident extent differs: {name}"
                );
                Ok(super::linear::LinearWeightRef::Bf16(
                    super::contracts::view(
                        values,
                        slice_stream(values),
                        0,
                        [rows, width],
                        [u64::from(width), 1],
                    )?,
                ))
            }
            _ => anyhow::bail!("classic fused group is not a BF16 tensor: {name}"),
        }
    }

    #[cfg(feature = "cuda")]
    fn classic_binding(&self, layer: usize, stream: &Arc<CudaStream>) -> Result<ClassicLayer> {
        let order = self
            .classic
            .as_ref()
            .context("classic catalogue binding missing")?
            .order
            .get(layer)
            .with_context(|| format!("classic layer {layer} outside catalogue"))?;
        crate::checkpoint::validate_classic_order(layer, order)?;
        let shapes: Vec<Vec<usize>> = order
            .iter()
            .map(|name| {
                Ok(self
                    .manifest
                    .entries
                    .get(name)
                    .with_context(|| format!("classic tensor {name} missing"))?
                    .shape
                    .clone())
            })
            .collect::<Result<_>>()?;
        let h = *shapes[0]
            .get(1)
            .context("classic q_proj has no input width")?;
        ensure!(
            shapes[..6].iter().all(|s| s.len() == 2 && s[1] == h),
            "classic layer {layer} projection widths differ"
        );
        ensure!(shapes[0][0] == h, "classic q_proj row width differs");
        ensure!(
            shapes[1] == shapes[2] && shapes[1][0] <= h,
            "classic k/v widths differ"
        );
        ensure!(shapes[3] == shapes[4], "classic gate/up widths differ");
        ensure!(shapes[5] == [h, h], "classic o_proj layout differs");
        ensure!(
            shapes[6] == [h, shapes[3][0]],
            "classic down_proj layout differs"
        );
        let qkv_rows = shapes[0][0] + shapes[1][0] + shapes[2][0];
        let gu_rows = shapes[3][0] + shapes[4][0];
        let qkv = self.classic_weight(&classic_qkv_name(layer), [qkv_rows, h])?;
        let gu = self.classic_weight(&classic_gu_name(layer), [gu_rows, h])?;
        let input_norm = self.classic_norm_view(layer, "input_layernorm", stream)?;
        let post_norm = self.classic_norm_view(layer, "post_attention_layernorm", stream)?;
        let o = self.linear_ref(&order[5], stream)?;
        let down = self.linear_ref(&order[6], stream)?;
        ensure!(
            matches!(o, super::linear::LinearWeightRef::Bf16(_))
                && matches!(down, super::linear::LinearWeightRef::Bf16(_)),
            "classic layer {layer} has non-BF16 o/down projections"
        );
        Ok(ClassicLayer {
            qkv,
            o,
            gu,
            down,
            input_norm,
            post_norm,
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
        let classic = classic_order(dir, spec)?;
        let phase = Instant::now();
        let ((tensors, bytes, upload_timings), upload_digest) =
            checkpoint_mapped(&mapped, d, Some(&manifest), hash_uploads, |loader| {
                let mut tensors = BTreeMap::new();
                let mut processed = BTreeSet::new();
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
                    if let Some(order) = &classic
                        && entry.is_linear_projection()
                        && let Some(layer) = entry.layer
                    {
                        let layer = layer as usize;
                        if processed.insert(layer) {
                            emit_classic_resident(
                                loader,
                                &mut tensors,
                                &manifest,
                                &order[layer],
                                layer,
                            )?;
                        }
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
        if crate::backend::diagnostics_enabled() {
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
        }
        #[cfg(feature = "cuda")]
        let catalog = Self {
            dense: None,
            classic: classic.map(|order| ClassicCatalog {
                order,
                bindings: Vec::new(),
            }),
            linear_format: spec.linear_format,
            tensors,
            manifest,
            bytes,
            upload_content_sha256: upload_digest.as_ref().map(|x| x.1.clone()),
            upload_hashed_bytes: upload_digest.map(|x| x.0),
            load_wall_ms,
        };
        #[cfg(feature = "cuda")]
        return catalog.bind_classic(&d.stream);
        #[cfg(not(feature = "cuda"))]
        Ok(Self {
            classic: classic.map(|order| ClassicCatalog { order }),
            linear_format: spec.linear_format,
            tensors,
            manifest,
            bytes,
            upload_hashed_bytes: upload_digest.map(|x| x.0),
            load_wall_ms,
        })
    }
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn bf16_mut(&mut self, name: &str) -> Result<&mut CudaSlice<bf16>> {
        match self.tensors.get_mut(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 tensor {name}"),
        }
    }

    pub fn bf16(&self, name: &str) -> Result<&CudaSlice<bf16>> {
        match self.tensors.get(name) {
            Some(StoredTensor::Bf16(v)) => Ok(v),
            _ => anyhow::bail!("missing/non-BF16 tensor {name}"),
        }
    }

    #[cfg(feature = "cuda")]
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
            #[cfg(feature = "cuda")]
            Some(StoredTensor::Slot { pointer, bytes }) => unsafe {
                super::contracts::Tensor::new(
                    *pointer,
                    [rows, width],
                    [u64::from(width), 1],
                    *bytes as u64,
                )
            },
            _ => anyhow::bail!("missing/non-BF16 tensor {name}"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn vector(
        &self,
        stream: &Arc<CudaStream>,
        name: &str,
    ) -> Result<super::contracts::Tensor<bf16, 1>> {
        let values = self.bf16(name)?;
        super::contracts::view(values, stream, 0, [u32::try_from(values.len())?], [1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use sha2::{Digest, Sha256};

    /// The supply's published reads that no upload retired and that wait in no slot.
    fn released_entries(dense: &DenseStreaming) -> u64 {
        dense.supply.as_ref().map_or(0, |supply| {
            released_count(&supply.shared.state.lock().expect("dense supply lock"))
        })
    }

    /// `published` less the retired entries and the entries waiting in a slot.
    fn released_count(state: &SupplyState) -> u64 {
        let waiting = state
            .layer
            .iter()
            .filter(|layer| matches!(layer, SupplyLayer::Ready(_)))
            .count() as u64;
        state.published - state.retired - waiting
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn demand_cursor_layer_is_admitted_under_capacity() {
        let fresh = || SupplyState {
            layer: vec![SupplyLayer::Waiting; 4],
            reads: vec![0; 4],
            pending: std::collections::VecDeque::new(),
            free: vec![true; 1],
            outstanding: 0,
            failure: None,
            published: 0,
            retired: 0,
        };
        let chain = |index: usize| (index < 3).then_some(index + 1);
        let sweep = |state: &mut SupplyState, cursor: &mut Option<usize>| {
            while let Some(index) = *cursor {
                if !state.admit(index, 1) {
                    break;
                }
                *cursor = chain(index);
            }
        };
        // Admission owns the cursor: every layer at or ahead of it is admitted or in flight
        // before the cursor moves on.
        let mut state = fresh();
        let mut cursor = Some(0);
        sweep(&mut state, &mut cursor);
        assert!(
            !matches!(state.layer[0], SupplyLayer::Waiting),
            "layer 0 left waiting behind the demand cursor"
        );
        assert_eq!(state.pending.front(), Some(&0));
        assert!(
            state
                .ready(0)
                .expect("an admitted layer has a read in flight")
                .is_none()
        );
        // Negative control: a cursor advanced past layer 0 before any admission strands it.
        let mut stranded = fresh();
        let mut skip = chain(0);
        sweep(&mut stranded, &mut skip);
        assert!(matches!(stranded.layer[0], SupplyLayer::Waiting));
        assert_eq!(stranded.pending.front(), Some(&1));
        // A demand for a layer with no read queued or in flight errors naming layer and visit.
        let error = format!(
            "{:#}",
            stranded
                .ready(0)
                .expect_err("a layer behind the cursor has no read")
        );
        assert!(
            error.contains("layer 0") && error.contains("visit 1"),
            "{error}"
        );
        stranded.layer[1] = SupplyLayer::Consumed;
        stranded.reads[1] = 1;
        let error = format!(
            "{:#}",
            stranded
                .ready(1)
                .expect_err("a consumed layer behind the cursor has no read")
        );
        assert!(
            error.contains("layer 1") && error.contains("visit 2"),
            "{error}"
        );
        // A layer the consumer finds loaded in its device slot ends its entry without an
        // upload, and the entry's capacity returns to the supply.
        let mut loaded = fresh();
        assert!(loaded.admit(0, 1));
        loaded.pending.clear();
        loaded.layer[0] = SupplyLayer::Ready(0);
        loaded.free[0] = false;
        assert!(
            !loaded.admit(1, 1),
            "a published entry left its capacity to the next layer"
        );
        assert!(loaded.release(0));
        assert!(
            matches!(loaded.layer[0], SupplyLayer::Consumed) && loaded.free[0],
            "a released published entry keeps its slot"
        );
        assert_eq!(loaded.outstanding, 0);
        assert!(loaded.admit(1, 1));
        assert_eq!(loaded.pending.front(), Some(&1));
        assert!(loaded.release(1));
        assert!(
            loaded.pending.is_empty() && matches!(loaded.layer[1], SupplyLayer::Consumed),
            "a released queued entry stays queued"
        );
        assert_eq!(loaded.outstanding, 0);
        assert!(!loaded.release(1));
        assert!(!loaded.release(3));
        assert!(loaded.admit(1, 1));
        // Of the published entries, those neither retired nor waiting in a slot were released.
        let mut counted = fresh();
        counted.published = 3;
        counted.retired = 1;
        counted.outstanding = 1;
        counted.layer[0] = SupplyLayer::Ready(0);
        counted.free[0] = false;
        assert_eq!(released_count(&counted), 1);
        assert!(counted.release(0));
        assert_eq!(released_count(&counted), 2);
        assert!(counted.admit(1, 1));
        assert!(counted.release(1));
        assert_eq!(released_count(&counted), 2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires GPU; a worker parked between its abort check and its wait must be woken by the drop"]
    fn supply_drop_wakes_a_worker_parking_before_its_wait_gpu() -> Result<()> {
        use std::sync::atomic::Ordering;
        use std::sync::mpsc;
        use std::time::Duration;
        let root =
            std::env::temp_dir().join(format!("ff-dense-supply-drop-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("model.safetensors"), vec![0u8; 8192])?;
        // The park holds the worker on the state for 200 ms between its abort check and its
        // wait. The supply is created, parked and dropped on one thread; the drop starts 50 ms
        // after the spawn, inside the park, and must return well inside 5 s.
        SUPPLY_WAIT_PARK_MICROS.store(200_000, Ordering::SeqCst);
        let (spawned, did_spawn) = mpsc::channel::<()>();
        let (reported, reports) = mpsc::channel::<Result<(), String>>();
        let dir = root.clone();
        std::thread::spawn(move || {
            let outcome = (|| -> Result<()> {
                let device = crate::backend::cuda::Device::new(0)?;
                let shard: Arc<str> = "model.safetensors".into();
                let supply = LayerSupply::spawn(
                    1,
                    vec![vec![SupplyRange {
                        source_offset: 0,
                        shard,
                        offset: 0,
                        len: 4096,
                    }]],
                    vec![4096],
                    4096,
                    &dir,
                    &device.ctx,
                )?;
                let _ = spawned.send(());
                std::thread::sleep(Duration::from_millis(50));
                drop(supply);
                Ok(())
            })();
            let _ = reported.send(outcome.map_err(|error| format!("{error:#}")));
        });
        did_spawn
            .recv_timeout(Duration::from_secs(60))
            .context("the supply was not created within 60 s")?;
        let dropped = reports.recv_timeout(Duration::from_secs(5));
        SUPPLY_WAIT_PARK_MICROS.store(0, Ordering::SeqCst);
        std::fs::remove_dir_all(&root)?;
        match dropped {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => anyhow::bail!("{error}"),
            Err(_) => anyhow::bail!(
                "the supply's drop did not return within 5 s: a worker parked between its abort check and its wait was not woken"
            ),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires GPU; a failed supply spawn must unregister the slots it registered"]
    fn supply_spawn_failure_unregisters_its_slots_gpu() -> Result<()> {
        use std::sync::atomic::Ordering;
        let root =
            std::env::temp_dir().join(format!("ff-dense-supply-spawn-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("model.safetensors"), vec![0u8; 8192])?;
        let device = crate::backend::cuda::Device::new(0)?;
        let shard: Arc<str> = "model.safetensors".into();
        let range = |offset: u64| SupplyRange {
            source_offset: 0,
            shard: shard.clone(),
            offset,
            len: 4096,
        };
        // Both slots are registered before the second worker's spawn is refused.
        SUPPLY_SPAWN_FAIL_SLOT.store(1, Ordering::SeqCst);
        SUPPLY_UNREGISTERS.store(0, Ordering::SeqCst);
        let failed = LayerSupply::spawn(
            2,
            vec![vec![range(0)], vec![range(4096)]],
            vec![4096; 2],
            4096,
            &root,
            &device.ctx,
        );
        SUPPLY_SPAWN_FAIL_SLOT.store(usize::MAX, Ordering::SeqCst);
        let unregistered = SUPPLY_UNREGISTERS.load(Ordering::SeqCst);
        std::fs::remove_dir_all(&root)?;
        let error = failed
            .err()
            .context("a refused worker spawn built a dense supply")?;
        let error = format!("{error:#}");
        ensure!(
            error.contains("is refused"),
            "the supply's spawn failed elsewhere: {error}"
        );
        ensure!(
            unregistered == 2,
            "a failed dense supply spawn unregistered {unregistered} of 2 registered slots"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn classic_streamed_slots_bind_fused_order_per_layer() -> Result<()> {
        use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope, classic_layer_names};
        let h = 64usize;
        let kv = 32usize;
        let f = 96usize;
        let rows = [h, kv, kv, f, f, h, h];
        let widths = [h, h, h, h, h, h, f];
        let mut entries = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        let mut order = Vec::new();
        for layer in 0..2usize {
            let names = classic_layer_names(layer).to_vec();
            let base = 0x1000u64 + (layer as u64) * 0x10000;
            let mut offset = 0usize;
            for (index, name) in names.iter().enumerate() {
                let shape = vec![rows[index], widths[index]];
                let bytes = rows[index] * widths[index] * 2;
                entries.insert(
                    name.clone(),
                    TensorEntry {
                        dtype: Dtype::BF16,
                        shape,
                        shard: String::new(),
                        bytes,
                        load: true,
                        data_start: 0,
                        data_offsets: [0, bytes as u64],
                        scope: TensorScope::Text,
                        layer: Some(layer as u32),
                        role: name.clone(),
                        scale_name: None,
                        quant: None,
                        swap_halves: false,
                    },
                );
                tensors.insert(
                    name.clone(),
                    StoredTensor::Slot {
                        pointer: base + offset as u64,
                        bytes,
                    },
                );
                offset += bytes;
            }
            let qkv_bytes = (0..3).map(|i| rows[i] * widths[i] * 2).sum::<usize>();
            let gu_bytes = (3..5).map(|i| rows[i] * widths[i] * 2).sum::<usize>();
            tensors.insert(
                classic_qkv_name(layer),
                StoredTensor::Slot {
                    pointer: base,
                    bytes: qkv_bytes,
                },
            );
            tensors.insert(
                classic_gu_name(layer),
                StoredTensor::Slot {
                    pointer: base + qkv_bytes as u64,
                    bytes: gu_bytes,
                },
            );
            order.push(names);
        }
        let catalog = WeightCatalog {
            dense: None,
            classic: Some(ClassicCatalog {
                order: order.clone(),
                bindings: Vec::new(),
            }),
            linear_format: crate::model_spec::LinearFormat::Bf16,
            tensors,
            manifest: TensorManifest {
                entries,
                resident_bytes: 0,
            },
            bytes: 0,
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: 0.0,
        };
        let mut pointers = Vec::new();
        for (layer, names) in order.iter().enumerate() {
            let fused = classic_fused_specs(&catalog.manifest, names, layer)?;
            let qkv_rows: usize = fused.qkv.iter().map(|(_, s)| s[0]).sum();
            let gu_rows: usize = fused.gu.iter().map(|(_, s)| s[0]).sum();
            let base = 0x1000u64 + (layer as u64) * 0x10000;
            let qkv_bytes = (0..3).map(|i| rows[i] * widths[i] * 2).sum::<usize>();
            let gu_bytes = (3..5).map(|i| rows[i] * widths[i] * 2).sum::<usize>();
            let o_bytes = h * h * 2;
            let expected = [
                (classic_qkv_name(layer), [qkv_rows, h], base),
                (
                    classic_gu_name(layer),
                    [gu_rows, h],
                    base + qkv_bytes as u64,
                ),
                (
                    names[5].clone(),
                    [h, h],
                    base + (qkv_bytes + gu_bytes) as u64,
                ),
                (
                    names[6].clone(),
                    [h, f],
                    base + (qkv_bytes + gu_bytes + o_bytes) as u64,
                ),
            ];
            for (name, shape, pointer) in expected {
                let crate::backend::cuda::linear::LinearWeightRef::Bf16(weight) =
                    catalog.classic_weight(&name, shape)?
                else {
                    bail!("classic slot binding is not BF16: {name}");
                };
                assert_eq!(weight.ptr, pointer, "slot pointer for {name}");
                assert_eq!(
                    weight.shape,
                    [shape[0] as u32, shape[1] as u32],
                    "slot shape for {name}"
                );
                pointers.push(weight.ptr);
            }
        }
        assert_ne!(
            pointers[0], pointers[4],
            "layer 0 and layer 1 must bind different slots"
        );
        let mut swapped = order[1].clone();
        swapped.swap(0, 1);
        assert!(
            classic_fused_specs(&catalog.manifest, &swapped, 1).is_err(),
            "q/k swap must be rejected"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU and a dense streaming window"]
    fn dense_slots_preserve_native_bytes_across_host_disk_reuse_gpu() -> Result<()> {
        #[cfg(feature = "cuda")]
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
                            quant: None,
                            swap_halves: false,
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
                1,
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
                (4 << 30) + per_layer + max_staging,
                1,
            )?;
            ensure!(
                disk_fixed == (4 << 30) + per_layer + max_staging
                    && disk_plan.host_bytes == 0
                    && disk_plan.disk_bytes == 4 * per_layer,
                "disk-spill placement did not reserve the spawned supply slot and bounce: disk_fixed={disk_fixed} max_staging={max_staging} plan={disk_plan:?}"
            );
            let stream = &device.stream;
            let mut outputs = (0..4)
                .map(|_| stream.alloc_zeros::<u8>(per_layer))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            #[cfg(feature = "cuda")]
            let blas = CudaBlas::new(stream.clone())?;
            let (n, k) = (shapes[1][0], shapes[1][1]);
            #[cfg(feature = "rocm")]
            let blas = {
                let blas = super::super::blas::Blas::new(stream.clone())?;
                for rows in [1u32, 17] {
                    blas.prepare_fixed(rows, n as u32, k as u32)?;
                }
                blas
            };
            let input = stream.clone_htod(
                &(0..18 * k)
                    .map(|i| bf16::from_f32(((i * 37 % 997) as f32 - 498.0) / 512.0))
                    .collect::<Vec<_>>(),
            )?;
            let mut products = (0..4)
                .map(|_| stream.alloc_zeros::<bf16>(18 * n))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            #[cfg(feature = "cuda")]
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
            #[cfg(feature = "rocm")]
            let product = |weight_base: u64, output: &mut CudaSlice<bf16>| -> Result<()> {
                let matrix0_elems = shapes[0].iter().product::<usize>();
                let weight_ptr = weight_base + (matrix0_elems * size_of::<bf16>()) as u64;
                let weight = unsafe {
                    super::super::contracts::Tensor::new(
                        weight_ptr,
                        [n as u32, k as u32],
                        [k as u64, 1],
                        (n * k * size_of::<bf16>()) as u64,
                    )?
                };
                let input_ptr = input.device_ptr(stream).0;
                let out_ptr = output.device_ptr(stream).0;
                for (start, rows) in [(0usize, 1usize), (1usize, 17usize)] {
                    let x = unsafe {
                        super::super::contracts::Tensor::new(
                            input_ptr + (start * k * size_of::<bf16>()) as u64,
                            [rows as u32, k as u32],
                            [k as u64, 1],
                            (rows * k * size_of::<bf16>()) as u64,
                        )?
                    };
                    let mut y = super::super::contracts::TensorMut(unsafe {
                        super::super::contracts::Tensor::new(
                            out_ptr + (start * n * size_of::<bf16>()) as u64,
                            [rows as u32, n as u32],
                            [n as u64, 1],
                            (rows * n * size_of::<bf16>()) as u64,
                        )?
                    });
                    blas.linear_bf16(weight, x, &mut y)?;
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
                #[cfg(feature = "cuda")]
                product(&resident, &mut products[layer])?;
                #[cfg(feature = "rocm")]
                product(resident.device_ptr(stream).0, &mut products[layer])?;
                let values = stream.clone_dtoh(&products[layer])?;
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite resident product"
                );
                resident_products.push(values);
            }
            println!("PASS dense_resident layers=4 layer_bytes={per_layer} byte_mismatches=0");
            for host_layers in [0, 2, 3, 4] {
                let plan = crate::prefill::DensePlacement::new(
                    &[per_layer; 4],
                    None,
                    2 * per_layer,
                    host_layers * per_layer,
                )?;
                ensure!(
                    plan.slots.iter().all(Option::is_some),
                    "streaming gate has a resident layer"
                );
                let mut dense = DenseStreaming::new(&root, &manifest, plan, &device, None)?;
                if host_layers == 4 {
                    ensure!(
                        dense.supply.is_none(),
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
                            super::super::copy::dtod_raw(target, source, per_layer, stream)?;
                        }
                        #[cfg(feature = "cuda")]
                        {
                            let slot =
                                dense.plan.slots[layer].context("streamed test slot missing")?;
                            product(&dense.slots[slot].device, &mut products[layer])?;
                        }
                        #[cfg(feature = "rocm")]
                        product(source, &mut products[layer])?;
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
                        let guard = device.upload.clone_dtoh(&slot.device.slice(per_layer..))?;
                        ensure!(
                            guard == [0xcd; 64],
                            "dense upload overwrote unused slot capacity"
                        );
                    }
                }
                let (claims, reclaims) = dense.flush_ring_counts()?;
                let released = released_entries(&dense);
                let slot_count = dense
                    .supply
                    .as_ref()
                    .map_or(0, |supply| supply.handles.len() as u64);
                if host_layers == 4 {
                    ensure!(
                        claims == 0 && reclaims == 0,
                        "host-only placement claimed disk ring slots: claims={claims} reclaims={reclaims}"
                    );
                } else {
                    // Every published read is retired by an upload, released at a layer the
                    // consumer finds loaded, or waiting in a slot at the count.
                    ensure!(
                        reclaims == (3 * (4 - host_layers)) as u64
                            && claims == reclaims + slot_count + released,
                        "disk ring claim/reclaim/release mismatch: host_layers={host_layers} claims={claims} reclaims={reclaims} released={released} slots={slot_count}"
                    );
                }
                println!(
                    "PASS dense_slots host_layers={host_layers} disk_layers={} layers=4 repeats=3 slot_bytes={per_layer} claims={claims} reclaims={reclaims} released={released} byte_mismatches=0 product_rows=1,17 product_bits=0 wrong_generation=detected bounds=detected tail_guards=preserved",
                    4 - host_layers
                );
            }
            let plan = crate::prefill::DensePlacement::new(
                &[per_layer; 4],
                None,
                2 * per_layer,
                2 * per_layer,
            )?;
            let mut dense = DenseStreaming::new(&root, &manifest, plan, &device, None)?;
            let layer = dense
                .layers
                .iter()
                .position(|layer| layer.slot.is_some() && layer.host.is_none())
                .context("mixed dense test needs a disk layer")?;
            let supply = dense
                .supply
                .as_ref()
                .context("mixed dense test lost the supply")?;
            for _ in 0..2 {
                let mut cursor = Some(layer);
                supply.pump(&mut cursor, |_| None);
                let (base, slot) = supply.demand(layer)?;
                let bytes = unsafe { std::slice::from_raw_parts(base, per_layer) };
                ensure!(
                    bytes == expected[layer],
                    "repeated disk key read different bytes"
                );
                supply.retire(layer, slot, &dense.upload)?;
            }
            ensure!(
                dense.flush_ring_counts()? == (2, 2),
                "repeated disk key did not retire both copies"
            );
            drop(dense);
            println!("PASS dense_repeated_disk_key published=2 retired=2 byte_mismatches=0");

            unsafe extern "C" fn wait_for_gate(user_data: *mut std::ffi::c_void) {
                let gate = unsafe { &*(user_data as *const std::sync::atomic::AtomicU32) };
                while gate.load(std::sync::atomic::Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
            }
            {
                let window = (per_layer / 2).min(4 << 20);
                let data_start = manifest
                    .entries
                    .values()
                    .next()
                    .context("dense test entries")?
                    .data_start;
                let ranges = (0..3usize)
                    .map(|layer| {
                        vec![SupplyRange {
                            source_offset: 0,
                            shard: "model.safetensors".into(),
                            offset: data_start + (layer as u64) * (per_layer as u64),
                            len: window as u64,
                        }]
                    })
                    .collect::<Vec<_>>();
                let source_bytes = vec![window; 3];
                let supply =
                    LayerSupply::spawn(1, ranges, source_bytes, window, &root, &device.ctx)?;
                let uploaded = device.upload.clone_htod(&vec![0u8; window])?;
                let gate = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                unsafe {
                    super::super::launch_host_func(
                        &device.upload,
                        wait_for_gate,
                        std::sync::Arc::as_ptr(&gate) as *mut std::ffi::c_void,
                    )
                    .context("hold dense upload stream")?;
                }
                let chain = |index: usize| (index < 2).then_some(index + 1);
                let mut cursor = Some(0);
                supply.pump(&mut cursor, chain);
                let (base, slot) = supply.demand(0)?;
                unsafe {
                    super::super::copy::htod_async(
                        uploaded.device_ptr(&device.upload).0,
                        std::slice::from_raw_parts(base, window),
                        &device.upload,
                    )?;
                }
                supply.retire(0, slot, &device.upload)?;
                supply.pump(&mut cursor, chain);
                let release = std::sync::Arc::clone(&gate);
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    release.store(1, std::sync::atomic::Ordering::Release);
                });
                let (next, _) = supply.demand(1)?;
                let overwritten = unsafe { std::slice::from_raw_parts(next, window) };
                ensure!(
                    overwritten == &expected[1][..window],
                    "reused supply slot did not hold the next layer"
                );
                drop(supply);
                device.upload.synchronize()?;
                let actual = stream.clone_dtoh(&uploaded)?;
                ensure!(
                    actual == expected[0][..window].to_vec(),
                    "retired supply slot was rewritten before its upload completed"
                );
            }
            println!(
                "PASS dense_reclaim_control held_upload_preserved=true slot_reuse_after_event=true"
            );
            Ok(())
        })();
        std::fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    #[ignore = "requires GPU"]
    fn dense_streaming_staging_dequant_cast_and_preloaded_gpu() -> Result<()> {
        let device = Device::new(0)?;
        let stream = &device.stream;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("ff-dense-stage-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let result = (|| -> Result<()> {
            let entry = |shape: Vec<usize>,
                         bytes: usize,
                         dtype: Dtype,
                         quant: Option<GgmlType>,
                         swap_halves: bool,
                         layer: u32,
                         role: &str| TensorEntry {
                dtype,
                bytes,
                shape,
                shard: "model.safetensors".into(),
                load: true,
                data_start: 0,
                data_offsets: [0, bytes as u64],
                scope: crate::checkpoint::TensorScope::Text,
                layer: Some(layer),
                role: role.into(),
                scale_name: None,
                quant,
                swap_halves,
            };

            // 1) with_f32_cast + passthrough BF16 + shift_source_offset negative control + nonfinite reject
            let elems = 32usize;
            let mut cast_entries = BTreeMap::new();
            let mut order = Vec::new();
            for l in 0..3u32 {
                let cast_name = format!("model.layers.{l}.cast.weight");
                let pass_name = format!("model.layers.{l}.pass.weight");
                cast_entries.insert(
                    cast_name.clone(),
                    entry(
                        vec![4, 8],
                        elems * 4,
                        Dtype::F32,
                        None,
                        false,
                        l,
                        "cast.weight",
                    ),
                );
                cast_entries.insert(
                    pass_name.clone(),
                    entry(
                        vec![4, 8],
                        elems * 2,
                        Dtype::BF16,
                        None,
                        false,
                        l,
                        "pass.weight",
                    ),
                );
                order.push(vec![cast_name, pass_name]);
            }
            let cast_manifest = TensorManifest {
                entries: cast_entries,
                resident_bytes: 0,
            };
            let per_layer_slot = elems * 2 + elems * 2;
            let per_layer_host = elems * 4 + elems * 2;
            let cast_plan = crate::prefill::DensePlacement::with_allocations(
                &[per_layer_slot; 3],
                &[per_layer_slot; 3],
                Some(&[per_layer_host; 3]),
                2 * per_layer_slot,
                3 * per_layer_host,
                0,
                2,
            )?;
            ensure!(
                cast_plan.slots == [Some(0), Some(1), Some(0)] && cast_plan.disk_bytes == 0,
                "expected 3 host layers sharing 2 slots: {cast_plan:?}"
            );
            let mut cast_dense = DenseStreaming::with_f32_cast(
                &root,
                &cast_manifest,
                cast_plan.clone(),
                &device,
                &order,
                |name| name.ends_with(".cast.weight"),
            )?;
            let mut f32_inputs = Vec::new();
            let mut bf16_inputs = Vec::new();
            for l in 0..3usize {
                let host = cast_dense
                    .host_layer(l)?
                    .context("missing cast host layer")?;
                let floats: Vec<f32> = (0..elems)
                    .map(|i| {
                        let raw = if i % 7 == 0 && l == 0 {
                            0x3f80_8000u32
                        } else {
                            let v = ((i as f32 - 15.5) * 0.125) + (l as f32) * 1.375;
                            v.to_bits()
                        };
                        f32::from_bits(raw)
                    })
                    .collect();
                for (i, &val) in floats.iter().enumerate() {
                    host[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
                let pass: Vec<u16> = (0..elems)
                    .map(|i| 0x3c00u16 + ((i * 11 + l * 43) % 199) as u16)
                    .collect();
                let base = elems * 4;
                for (i, &w) in pass.iter().enumerate() {
                    host[base + i * 2..base + (i + 1) * 2].copy_from_slice(&w.to_le_bytes());
                }
                f32_inputs.push(floats);
                bf16_inputs.push(pass);
            }
            for l in 0..3usize {
                cast_dense.begin(l, stream)?;
                let cast_ptr = cast_dense.pointer(l, 0, elems * 2)?;
                let pass_ptr = cast_dense.pointer(l, elems * 2, elems * 2)?;
                let mut got_cast = vec![0u16; elems];
                let mut got_pass = vec![0u16; elems];
                unsafe {
                    super::super::copy::dtoh_async(&mut got_cast, cast_ptr, stream)?;
                    super::super::copy::dtoh_async(&mut got_pass, pass_ptr, stream)?;
                }
                stream.synchronize()?;
                cast_dense.end(l, stream)?;
                let exp_cast: Vec<u16> = f32_inputs[l]
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_bits())
                    .collect();
                ensure!(
                    got_cast == exp_cast,
                    "F32->BF16 streamed cast bits differ on layer {l}"
                );
                ensure!(
                    got_pass == bf16_inputs[l],
                    "BF16 passthrough bits differ on layer {l}"
                );
            }
            // Negative control: shift layer 0 cast offset by 4 bytes while slot 0 holds layer 1.
            cast_dense.shift_source_offset(0, "model.layers.0.cast.weight", 4)?;
            cast_dense.begin(0, stream)?;
            let mut shifted_cast = vec![0u16; elems];
            unsafe {
                super::super::copy::dtoh_async(
                    &mut shifted_cast,
                    cast_dense.pointer(0, 0, elems * 2)?,
                    stream,
                )?;
            }
            stream.synchronize()?;
            cast_dense.end(0, stream)?;
            let exp_cast0: Vec<u16> = f32_inputs[0]
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect();
            ensure!(
                shifted_cast != exp_cast0 && shifted_cast[0] == exp_cast0[1],
                "shift_source_offset negative control did not shift layer 0 cast output"
            );
            // Non-finite F32 rejection in unchecked host_layer.
            let mut nan_dense = DenseStreaming::with_f32_cast(
                &root,
                &cast_manifest,
                cast_plan,
                &device,
                &order,
                |name| name.ends_with(".cast.weight"),
            )?;
            nan_dense.host_layer(0)?.context("nan host")?[0..4]
                .copy_from_slice(&f32::NAN.to_le_bytes());
            ensure!(
                nan_dense.begin(0, stream).is_err(),
                "non-finite F32 cast input must be rejected"
            );

            // 2) Q4_0 / Q4_0 swap_halves / F16 dequant via DenseStreaming + Q4_0->F32 via QuantKernels
            let mut q_entries = BTreeMap::new();
            let q4_name = "model.layers.0.q4.weight".to_owned();
            let swap_name = "model.layers.0.swap.weight".to_owned();
            let f16_name = "model.layers.0.f16.weight".to_owned();
            let q4_shape = vec![2usize, 64];
            let swap_shape = vec![4usize, 32];
            let f16_shape = vec![2usize, 16];
            q_entries.insert(
                f16_name.clone(),
                entry(
                    f16_shape.clone(),
                    64,
                    Dtype::BF16,
                    Some(GgmlType::F16),
                    false,
                    0,
                    "f16.weight",
                ),
            );
            q_entries.insert(
                q4_name.clone(),
                entry(
                    q4_shape.clone(),
                    72,
                    Dtype::BF16,
                    Some(GgmlType::Q4_0),
                    false,
                    0,
                    "q4.weight",
                ),
            );
            q_entries.insert(
                swap_name.clone(),
                entry(
                    swap_shape.clone(),
                    72,
                    Dtype::BF16,
                    Some(GgmlType::Q4_0),
                    true,
                    0,
                    "swap.weight",
                ),
            );
            for l in 1..3u32 {
                let extra = format!("model.layers.{l}.q4.weight");
                q_entries.insert(
                    extra.clone(),
                    entry(
                        q4_shape.clone(),
                        72,
                        Dtype::BF16,
                        Some(GgmlType::Q4_0),
                        false,
                        l,
                        "q4.weight",
                    ),
                );
            }
            let q_manifest = TensorManifest {
                entries: q_entries,
                resident_bytes: 0,
            };
            let q_order = [
                vec![q4_name.clone(), swap_name.clone(), f16_name.clone()],
                vec!["model.layers.1.q4.weight".to_owned()],
                vec!["model.layers.2.q4.weight".to_owned()],
            ];
            let q_slot_bytes = 2 * (128 + 128 + 32);
            let q_host_bytes = 72 + 72 + 64;
            let q_plan = crate::prefill::DensePlacement::with_allocations(
                &[q_slot_bytes; 3],
                &[q_slot_bytes; 3],
                Some(&[q_host_bytes, 72, 72]),
                2 * q_slot_bytes,
                q_host_bytes + 144,
                0,
                2,
            )?;
            ensure!(
                q_plan.slots == [Some(0), Some(1), Some(0)],
                "expected quant test 3 layers across 2 streaming slots: {q_plan:?}"
            );
            let mut q_dense =
                DenseStreaming::new(&root, &q_manifest, q_plan, &device, Some(&q_order))?;
            let make_q4_blocks = |blocks: usize, seed: usize| -> (Vec<u8>, Vec<f32>) {
                let scales = [0x3c00u16, 0x3800, 0xbc00, 0x4000];
                let mut raw = vec![0u8; blocks * 18];
                let mut f32_ref = vec![0.0f32; blocks * 32];
                for b in 0..blocks {
                    let sb = scales[(b + seed) % scales.len()];
                    let d = half::f16::from_bits(sb).to_f32();
                    raw[b * 18..b * 18 + 2].copy_from_slice(&sb.to_le_bytes());
                    for j in 0..16 {
                        let lo = ((j * 3 + b + seed) % 16) as u8;
                        let hi = ((j * 5 + b * 2 + seed) % 16) as u8;
                        raw[b * 18 + 2 + j] = lo | (hi << 4);
                        f32_ref[b * 32 + j] = f32::from(i16::from(lo) - 8) * d;
                        f32_ref[b * 32 + 16 + j] = f32::from(i16::from(hi) - 8) * d;
                    }
                }
                (raw, f32_ref)
            };
            let (q4_raw, q4_f32) = make_q4_blocks(4, 1);
            let (swap_raw, swap_f32) = make_q4_blocks(4, 3);
            let f16_bits: Vec<u16> = (0..32)
                .map(|i| half::f16::from_f32(((i as f32) - 12.0) * 0.25).to_bits())
                .collect();
            {
                let host = q_dense.host_layer(0)?.context("missing quant host layer")?;
                host[..72].copy_from_slice(&q4_raw);
                host[72..144].copy_from_slice(&swap_raw);
                for (i, &bits) in f16_bits.iter().enumerate() {
                    host[144 + i * 2..144 + (i + 1) * 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
            q_dense.begin(0, stream)?;
            let mut got_q4 = vec![0u16; 128];
            let mut got_swap = vec![0u16; 128];
            let mut got_f16 = vec![0u16; 32];
            unsafe {
                super::super::copy::dtoh_async(&mut got_q4, q_dense.pointer(0, 0, 256)?, stream)?;
                super::super::copy::dtoh_async(
                    &mut got_swap,
                    q_dense.pointer(0, 256, 256)?,
                    stream,
                )?;
                super::super::copy::dtoh_async(&mut got_f16, q_dense.pointer(0, 512, 64)?, stream)?;
            }
            stream.synchronize()?;
            q_dense.end(0, stream)?;
            let exp_q4: Vec<u16> = q4_f32
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect();
            ensure!(got_q4 == exp_q4, "Q4_0 -> BF16 dequant bits differ");
            let mut exp_swap = vec![0u16; 128];
            for row in 0..4usize {
                let dst_row = if row < 2 { row + 2 } else { row - 2 };
                for c in 0..32usize {
                    exp_swap[dst_row * 32 + c] = bf16::from_f32(swap_f32[row * 32 + c]).to_bits();
                }
            }
            ensure!(got_swap == exp_swap, "Q4_0 swap_halves dequant bits differ");
            let exp_f16: Vec<u16> = f16_bits
                .iter()
                .map(|&b| bf16::from_f32(half::f16::from_bits(b).to_f32()).to_bits())
                .collect();
            ensure!(got_f16 == exp_f16, "F16 -> BF16 dequant bits differ");

            // Also verify QuantKernels F32 dequant path.
            let quant = QuantKernels::new(&device.ctx)?;
            let dev_src = stream.clone_htod(&q4_raw)?;
            let dev_dst = stream.alloc_zeros::<f32>(128)?;
            quant.dequant(
                stream,
                "q4_f32",
                &entry(
                    q4_shape,
                    72,
                    Dtype::F32,
                    Some(GgmlType::Q4_0),
                    false,
                    0,
                    "q4.weight",
                ),
                DequantCopy {
                    source: dev_src.device_ptr(stream).0,
                    destination: dev_dst.device_ptr(stream).0,
                    elements: 128,
                    dtype: Dtype::F32,
                },
            )?;
            let got_f32 = stream.clone_dtoh(&dev_dst)?;
            ensure!(
                got_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    == q4_f32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "Q4_0 -> F32 dequant bits differ"
            );

            // 3) DenseStreaming::new_preloaded
            let pre_bytes = 64usize;
            let pre_plan = crate::prefill::DensePlacement::with_allocations(
                &[pre_bytes; 3],
                &[pre_bytes; 3],
                None,
                2 * pre_bytes,
                3 * pre_bytes,
                0,
                2,
            )?;
            let mut pre = DenseStreaming::new_preloaded(&root, pre_plan, &device, &[pre_bytes; 3])?;
            let p0: Vec<u8> = (0..pre_bytes)
                .map(|i| (i as u8).wrapping_mul(3).wrapping_add(1))
                .collect();
            let p1: Vec<u8> = (0..pre_bytes)
                .map(|i| (i as u8).wrapping_mul(7).wrapping_add(5))
                .collect();
            let p2: Vec<u8> = (0..pre_bytes)
                .map(|i| (i as u8).wrapping_mul(11).wrapping_add(9))
                .collect();
            pre.host_layer(0)?
                .context("pre host 0")?
                .copy_from_slice(&p0);
            pre.host_layer(1)?
                .context("pre host 1")?
                .copy_from_slice(&p1);
            pre.host_layer(2)?
                .context("pre host 2")?
                .copy_from_slice(&p2);
            for (layer, expected) in [(0usize, &p0), (1, &p1), (2, &p2), (0, &p0)] {
                pre.begin(layer, stream)?;
                let mut got = vec![0u8; pre_bytes];
                unsafe {
                    super::super::copy::dtoh_async(
                        &mut got,
                        pre.pointer(layer, 0, pre_bytes)?,
                        stream,
                    )?;
                }
                stream.synchronize()?;
                pre.end(layer, stream)?;
                ensure!(got == *expected, "new_preloaded layer {layer} bytes differ");
            }
            println!(
                "PASS dense_streaming_staging_dequant_cast_and_preloaded f32_cast=exact q4_bf16=exact q4_swap=exact q4_f32=exact f16_bf16=exact preloaded=exact shift_control=detected nan_control=rejected"
            );
            Ok(())
        })();
        std::fs::remove_dir_all(&root)?;
        result
    }

    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
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
                quant: None,
                swap_halves: false,
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
            for role in [
                crate::checkpoint::GLM_HC_ATTN_FN,
                crate::checkpoint::GLM_HC_FFN_FN,
            ] {
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
                    quant: None,
                    swap_halves: false,
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
            (
                crate::checkpoint::GLM_MLP_GATE,
                vec![2, 4],
                true,
                Dtype::BF16,
            ),
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
            quant: None,
            swap_halves: false,
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
            ["linear_attn.conv1d.weight", crate::checkpoint::GLM_MLP_GATE]
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
    fn quantized_device_slot_and_source_finite_checks() {
        let entry = |shape: Vec<usize>, bytes: usize, quant: Option<GgmlType>| TensorEntry {
            dtype: Dtype::BF16,
            bytes,
            shape,
            shard: "model.gguf".into(),
            load: true,
            data_start: 0,
            data_offsets: [0, 0],
            scope: crate::checkpoint::TensorScope::Text,
            layer: Some(0),
            role: crate::checkpoint::GLM_MLP_GATE_PROJ.into(),
            scale_name: None,
            quant,
            swap_halves: false,
        };
        let shape = vec![64usize, 32];
        let elements = 64 * 32;
        for quant in [Some(GgmlType::Q4_0), Some(GgmlType::F16)] {
            let slot =
                device_slot_bytes("t", &entry(shape.clone(), 18 * elements / 32, quant)).unwrap();
            assert_eq!(slot, 2 * elements, "{quant:?} slot");
        }
        for quant in [None, Some(GgmlType::F32), Some(GgmlType::Bf16)] {
            let slot = device_slot_bytes("t", &entry(shape.clone(), 4 * elements, quant)).unwrap();
            assert_eq!(slot, 4 * elements, "{quant:?} slot");
        }
        for quant in [
            Some(GgmlType::Other(8)),
            Some(GgmlType::Other(12)),
            Some(GgmlType::Other(9)),
        ] {
            assert!(
                device_slot_bytes("t", &entry(shape.clone(), 18 * elements / 32, quant)).is_err(),
                "{quant:?} has no device path"
            );
        }
        let mut native = entry(vec![2], 8, None);
        native.dtype = safetensors::Dtype::F32;
        let finite: Vec<_> = [0.15625001f32, -1.0000001]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(quant_source_finite("f32", &native, &finite).is_ok());
        let invalid: Vec<_> = [1.0f32, f32::NAN]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(quant_source_finite("f32", &native, &invalid).is_err());
        assert!(quant_source_finite("f32", &native, &finite[..7]).is_err());
        let nan = 0x7e00u16.to_le_bytes();
        let mut f16 = vec![0u8; 8];
        f16[2..4].copy_from_slice(&nan);
        assert!(
            quant_source_finite("t", &entry(vec![], 8, Some(GgmlType::F16)), &f16).is_err(),
            "NaN F16 value accepted"
        );
        let mut q4 = vec![0u8; 18 * 2];
        q4[18..20].copy_from_slice(&nan);
        assert!(
            quant_source_finite("t", &entry(vec![], 18 * 2, Some(GgmlType::Q4_0)), &q4).is_err(),
            "NaN Q4_0 scale accepted"
        );
        q4[18..20].copy_from_slice(&0x3c00u16.to_le_bytes());
        assert!(
            quant_source_finite("t", &entry(vec![], 18 * 2, Some(GgmlType::Q4_0)), &q4).is_ok(),
            "finite Q4_0 scale rejected"
        );
        assert!(
            quant_source_finite("t", &entry(vec![], 8, None), &f16).is_ok(),
            "unquantized source checked"
        );
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
        let base_reserve = 4usize << 30;
        let layers = [per_layer; 16];
        for (alignment, staging) in [(8192u64, staging_8192), (4096, staging_4096)] {
            let ranges: [(_, u64, u64, u64); 16] =
                std::array::from_fn(|layer| (layer, offset, len, alignment));
            for workers in [1usize, 2, 4, 8] {
                // Every layer on disk: one slot and one bounce per worker the plan spawns.
                let spawned = workers.min(16) * (per_layer + staging);
                let (fixed, plan) = WeightCatalog::dense_host_fixed(
                    &layers,
                    &ranges,
                    2 * per_layer,
                    base_reserve + spawned,
                    workers,
                )?;
                assert_eq!(
                    plan.host_bytes, 0,
                    "workers {workers} alignment {alignment}"
                );
                assert_eq!(
                    fixed,
                    base_reserve + spawned,
                    "workers {workers} alignment {alignment}"
                );
                // One layer above the charge for fifteen disk layers at the 4096 extent: that
                // layer stays on the host tier only while every bounce holds the 4096 extent.
                let available =
                    base_reserve + workers.min(15) * (per_layer + staging_4096) + per_layer;
                let (fixed, plan) = WeightCatalog::dense_host_fixed(
                    &layers,
                    &ranges,
                    2 * per_layer,
                    available,
                    workers,
                )?;
                let (host, disk_layers) = if alignment == 4096 {
                    (per_layer, 15)
                } else {
                    (0, 16)
                };
                assert_eq!(
                    plan.host_bytes, host,
                    "workers {workers} alignment {alignment}"
                );
                assert_eq!(
                    fixed,
                    base_reserve + workers.min(disk_layers) * (per_layer + staging),
                    "workers {workers} alignment {alignment}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
pub(crate) struct DiffusionWeights {
    pub embed: CudaSlice<bf16>,
    pub final_norm: CudaSlice<bf16>,
    pub layers: Vec<DiffusionLayer>,
    pub encoder_scalars: Vec<CudaSlice<bf16>>,
    pub sc_pre_norm: CudaSlice<bf16>,
    pub sc_gu: CudaSlice<bf16>,
    pub sc_down: CudaSlice<bf16>,
}

#[cfg(feature = "cuda")]
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
                    input_norm: l.load(&[(
                        format!("{p}.{}", crate::checkpoint::GLM_INPUT_LAYERNORM),
                        vec![h],
                    )])?,
                    post_attn: l.load(&[(
                        format!("{p}.{}", crate::checkpoint::GLM_POST_ATTENTION_LAYERNORM),
                        vec![h],
                    )])?,
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

#[cfg(test)]
mod classic_fused_tests {
    use super::*;

    fn config() -> Config {
        Config {
            model_type: "llama".into(),
            hidden_size: 7168,
            intermediate_size: 22016,
            num_hidden_layers: 52,
            num_attention_heads: 56,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 166912,
            max_position_embeddings: 524288,
            rms_norm_eps: 1e-5,
            rope_theta: 120000000.0,
            rope_scaling: None,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            eos_token_id: vec![166101],
            attention_bias: false,
            mlp_bias: false,
        }
    }

    #[test]
    fn classic_fused_binding_pins_qkv_and_gate_up_order() -> Result<()> {
        let c = config();
        let fused = ClassicFused::new(&c, 7);
        fused.validate(7)?;
        assert_eq!(fused.qkv[0].1, vec![c.hidden_size, c.hidden_size]);
        assert_eq!(fused.qkv[1].1, vec![c.kv_dim(), c.hidden_size]);
        assert_eq!(fused.qkv[2].1, vec![c.kv_dim(), c.hidden_size]);
        assert_eq!(fused.gu[0].1, vec![c.intermediate_size, c.hidden_size]);
        assert_eq!(fused.gu[1].1, vec![c.intermediate_size, c.hidden_size]);
        Ok(())
    }

    #[test]
    fn classic_fused_binding_rejects_swapped_orders() {
        let c = config();
        let mut swapped_qk = ClassicFused::new(&c, 3);
        swapped_qk.qkv.swap(0, 1);
        assert!(swapped_qk.validate(3).is_err(), "q/k swap must be rejected");
        let mut swapped_kv = ClassicFused::new(&c, 3);
        swapped_kv.qkv.swap(1, 2);
        assert!(swapped_kv.validate(3).is_err(), "k/v swap must be rejected");
        let mut swapped_gu = ClassicFused::new(&c, 3);
        swapped_gu.gu.swap(0, 1);
        assert!(
            swapped_gu.validate(3).is_err(),
            "gate/up swap must be rejected"
        );
        assert!(
            ClassicFused::new(&c, 3).validate(4).is_err(),
            "layer mismatch must be rejected"
        );
    }
}
