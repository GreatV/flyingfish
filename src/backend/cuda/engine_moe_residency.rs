//! Native expert DMA ownership and the CPU residency runtime.
use super::*;
use crate::backend::gpu::CudaFunction;
use crate::experts::{
    arena::{BlockLease, WriteClaim},
    catalogue::ExpertCatalogue,
    disk::{DiskExperts, DiskReader},
    plan::Tier,
    runtime::{MoveSource, PromptMove, Residency},
};
use std::cell::RefCell;

#[path = "engine_moe_prefill.rs"]
mod prefill;
pub(in crate::backend::cuda::engine) use prefill::Prefill;
pub(super) use prefill::{cpu_bound, peer_prefill_bytes};

#[derive(Clone)]
struct Ticket {
    id: u64,
    event: Arc<CudaEvent>,
}

struct Events {
    pool: Vec<Arc<CudaEvent>>,
    ids: Vec<u64>,
    epoch: u64,
    cursor: usize,
}

fn event_index(id: u64, ids: &[u64]) -> Result<usize> {
    ensure!(id > 0 && !ids.is_empty(), "invalid expert event id={id}");
    let index = ((id - 1) % ids.len() as u64) as usize;
    ensure!(
        ids[index] == id,
        "stale expert event id={id} current={} index={index}",
        ids[index]
    );
    Ok(index)
}

impl Events {
    fn new(context: &Arc<CudaContext>, count: usize) -> Result<Self> {
        ensure!(count > 0, "empty expert event pool");
        Ok(Self {
            pool: (0..count)
                .map(|_| Ok(Arc::new(context.new_event(None)?)))
                .collect::<Result<_>>()?,
            ids: vec![0; count],
            epoch: 0,
            cursor: 0,
        })
    }

    fn claim(&mut self) -> Result<Ticket> {
        let index = (0..self.pool.len())
            .map(|offset| (self.cursor + offset) % self.pool.len())
            .find(|&index| Arc::strong_count(&self.pool[index]) == 1)
            .context("expert event pool has no unowned handle")?;
        self.epoch = self
            .epoch
            .checked_add(1)
            .context("expert event epoch overflow")?;
        let id = self
            .epoch
            .checked_mul(self.pool.len() as u64)
            .and_then(|base| base.checked_add(index as u64 + 1))
            .context("expert event id overflow")?;
        self.ids[index] = id;
        self.cursor = (index + 1) % self.pool.len();
        Ok(Ticket {
            id,
            event: self.pool[index].clone(),
        })
    }

    fn record(&mut self, stream: &Arc<CudaStream>) -> Result<Ticket> {
        let ticket = self.claim()?;
        ticket.event.record(stream)?;
        Ok(ticket)
    }

    fn wait(&self, stream: &Arc<CudaStream>, id: u64) -> Result<()> {
        let index = event_index(id, &self.ids)?;
        ensure!(
            Arc::strong_count(&self.pool[index]) > 1,
            "expert event id={id} has no lifetime owner"
        );
        stream.wait(&self.pool[index])?;
        Ok(())
    }
}

struct Reader {
    disk: DiskReader,
    context: Arc<CudaContext>,
    registrations: Vec<u64>,
    copied: Vec<Option<Ticket>>,
}

impl Reader {
    fn new(
        device: &Device,
        portable: bool,
        disk: DiskReader,
        catalogue: &ExpertCatalogue,
    ) -> Result<Self> {
        let slots = disk.ring_slots() as usize;
        let mut reader = Self {
            disk,
            context: device.ctx.clone(),
            registrations: Vec::with_capacity(slots),
            copied: vec![None; slots],
        };
        device.ctx.bind_to_thread()?;
        #[cfg(feature = "cuda")]
        let flags = if portable {
            cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE
        } else {
            0
        };
        for _ in 0..slots {
            let slot = reader.disk.read_expert(0, 0, catalogue)?;
            let values = reader.disk.values(slot)?;
            let address = values.as_ptr() as u64;
            let bytes = std::mem::size_of_val(values.values());
            ensure!(
                !reader.registrations.contains(&address),
                "disk ring registration repeated address={address:#x}"
            );
            // SAFETY: the reader owns every stable ring allocation through unregister.
            #[cfg(feature = "cuda")]
            let result = unsafe {
                cudarc::driver::sys::cuMemHostRegister_v2(
                    address as *mut std::ffi::c_void,
                    bytes,
                    flags,
                )
                .result()
            };
            #[cfg(feature = "rocm")]
            let result = unsafe { device.ctx.register_host(address, bytes, portable) };
            result
                .with_context(|| format!("register expert disk ring: slot={slot} bytes={bytes}"))?;
            reader.registrations.push(address);
        }
        Ok(reader)
    }

    fn read(&mut self, layer: u32, expert: u32, catalogue: &ExpertCatalogue) -> Result<u32> {
        for (slot, event) in self.copied.iter_mut().enumerate() {
            let complete = if let Some(ticket) = event.as_ref() {
                ticket.event.try_is_complete()?
            } else {
                false
            };
            if complete {
                let completed = event.take().context("expert ring event disappeared")?;
                self.disk.reclaim(slot as u32, completed.id)?;
            }
        }
        if self.copied.iter().all(Option::is_some) {
            let (slot, _) = self
                .copied
                .iter()
                .enumerate()
                .min_by_key(|(_, event)| event.as_ref().map(|ticket| ticket.id))
                .context("expert disk ring has no slots")?;
            let event = self.copied[slot]
                .take()
                .context("expert disk ring event missing")?;
            event.event.synchronize()?;
            self.disk.reclaim(slot as u32, event.id)?;
        }
        self.disk.read_expert(layer, expert, catalogue)
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        for ticket in self.copied.iter().flatten() {
            self.context.record_err(ticket.event.synchronize());
        }
        self.context.record_err(self.context.synchronize());
        self.context.record_err(self.context.bind_to_thread());
        for &address in &self.registrations {
            // SAFETY: all ring DMA is complete and the owned allocation is still live.
            #[cfg(feature = "cuda")]
            let result = unsafe {
                cudarc::driver::sys::cuMemHostUnregister(address as *mut std::ffi::c_void).result()
            };
            #[cfg(feature = "rocm")]
            let result = unsafe { self.context.unregister_host(address) };
            self.context.record_err(result);
        }
    }
}

#[derive(Default)]
struct HostUse {
    claim: Option<WriteClaim>,
    event: Option<Ticket>,
    staged: Option<usize>,
}

struct Stage {
    values: PinnedHostSlice<bf16>,
    event: Option<Ticket>,
    destination: Option<u32>,
}

pub(in crate::backend::cuda::engine) struct Transfers {
    host: Option<Arc<HostBlocks>>,
    backing: Option<BlockLease>,
    host_use: Vec<RefCell<HostUse>>,
    stages: Vec<RefCell<Stage>>,
    h2d_cursor: usize,
    d2h_cursor: usize,
    per_direction: usize,
    h2d: Arc<CudaStream>,
    d2h: Arc<CudaStream>,
    pinned: bool,
}

impl Transfers {
    fn new(
        device: &Device,
        storage: &super::super::super::weights::ExpertStorage,
        pinned: bool,
        engines: u32,
        backing_blocks: u32,
    ) -> Result<Self> {
        ensure!(
            engines > 0,
            "expert copies require at least one measured copy engine"
        );
        let host = storage.host().cloned();
        ensure!(
            !pinned || host.is_some(),
            "pinned expert plan has no host arena"
        );
        let host_use = (0..host.as_ref().map_or(0, |arena| arena.blocks()))
            .map(|_| RefCell::new(HostUse::default()))
            .collect();
        let elements = storage.layout().slot_bytes() / 2;
        let per_direction = engines.min(2) as usize;
        let stages = if !pinned && host.is_some() {
            2 * per_direction
        } else {
            0
        };
        let backing = if backing_blocks == 0 {
            None
        } else {
            Some(
                host.as_ref()
                    .context("immutable expert backing has no arena")?
                    .lease(&(0..backing_blocks).collect::<Vec<_>>())?,
            )
        };
        Ok(Self {
            host,
            backing,
            host_use,
            stages: (0..stages)
                .map(|_| {
                    Ok(RefCell::new(Stage {
                        values: pinned_zeros(&device.ctx, elements)?,
                        event: None,
                        destination: None,
                    }))
                })
                .collect::<Result<_>>()?,
            h2d_cursor: 0,
            d2h_cursor: per_direction,
            per_direction,
            h2d: device.upload.clone(),
            d2h: if engines >= 2 {
                device.ctx.new_stream()?
            } else {
                device.upload.clone()
            },
            pinned,
        })
    }

    fn flush_stage(&self, index: usize) -> Result<()> {
        let mut stage = self
            .stages
            .get(index)
            .context("expert staging index outside allocation")?
            .borrow_mut();
        if let Some(event) = stage.event.as_ref() {
            event.event.synchronize()?;
        }
        if let Some(block) = stage.destination.take() {
            let mut target = self.host_use[block as usize].borrow_mut();
            target
                .claim
                .as_mut()
                .context("staged expert write-back has no writer claim")?
                .slice()?
                .copy_from_slice(stage.values.as_slice()?);
            target.staged = None;
            target.claim.take();
        }
        stage.event = None;
        Ok(())
    }

    fn ready_host(&self, block: u32) -> Result<()> {
        let usage = self
            .host_use
            .get(block as usize)
            .context("expert host block outside arena")?;
        let staged = usage.borrow().staged;
        if let Some(index) = staged {
            return self.flush_stage(index);
        }
        let mut usage = usage.borrow_mut();
        if usage.claim.is_some() {
            usage
                .event
                .as_ref()
                .context("expert host claim has no DMA completion")?
                .event
                .synchronize()?;
            usage.claim.take();
        }
        Ok(())
    }

    fn claim(&self, block: u32) -> Result<u64> {
        let arena = self.host.as_ref().context("expert DMA has no host arena")?;
        let mut usage = self
            .host_use
            .get(block as usize)
            .context("expert host block outside arena")?
            .borrow_mut();
        if usage.claim.is_none() {
            usage.claim = Some(arena.claim(block)?);
        }
        usage
            .claim
            .as_ref()
            .context("expert host writer claim missing")?
            .pointer()
    }

    fn wait_host(&self, block: u32, stream: &Arc<CudaStream>) -> Result<()> {
        let usage = self
            .host_use
            .get(block as usize)
            .context("expert host block outside arena")?
            .borrow();
        if let Some(event) = &usage.event {
            stream.wait(&event.event)?;
        }
        Ok(())
    }

    fn source_host(&mut self, block: u32) -> Result<(u64, Option<usize>)> {
        if self.pinned {
            if let Some(backing) = &self.backing {
                return Ok((backing.read(block)?.as_ptr() as u64, None));
            }
            let pointer = self.claim(block)?;
            self.wait_host(block, &self.h2d)?;
            return Ok((pointer, None));
        }
        self.ready_host(block)?;
        let index = self.h2d_cursor;
        self.h2d_cursor = (self.h2d_cursor + 1) % self.per_direction;
        self.flush_stage(index)?;
        let lease = if self.backing.is_none() {
            Some(
                self.host
                    .as_ref()
                    .context("expert host source missing")?
                    .lease(&[block])?,
            )
        } else {
            None
        };
        let lease = self
            .backing
            .as_ref()
            .or(lease.as_ref())
            .context("expert host source has no read lease")?;
        let mut stage = self.stages[index].borrow_mut();
        stage
            .values
            .as_mut_slice()?
            .copy_from_slice(lease.read(block)?);
        Ok((stage.values.as_ptr()? as u64, Some(index)))
    }

    fn write_back(
        &mut self,
        pointer: u64,
        block: u32,
        bytes: usize,
        events: &mut Events,
    ) -> Result<Ticket> {
        let stage = if self.pinned {
            None
        } else {
            self.ready_host(block)?;
            let index = self.d2h_cursor;
            self.d2h_cursor = self.per_direction
                + (self.d2h_cursor - self.per_direction + 1) % self.per_direction;
            self.flush_stage(index)?;
            Some(index)
        };
        let claimed = self.claim(block)?;
        self.wait_host(block, &self.d2h)?;
        let destination = if let Some(index) = stage {
            self.stages[index].borrow_mut().values.as_mut_ptr()? as u64
        } else {
            claimed
        };
        self.d2h.context().bind_to_thread()?;
        // SAFETY: the full native slot and its claimed host or pinned destination stay owned.
        unsafe {
            copy::dtoh_raw(
                destination as *mut std::ffi::c_void,
                pointer,
                bytes,
                self.d2h.as_ref(),
            )
        }
        .with_context(|| format!("expert write-back: block={block} bytes={bytes}"))?;
        let completed = events.record(&self.d2h)?;
        let mut usage = self.host_use[block as usize].borrow_mut();
        usage.event = Some(completed.clone());
        usage.staged = stage;
        if let Some(index) = stage {
            let mut stage = self.stages[index].borrow_mut();
            stage.event = Some(completed.clone());
            stage.destination = Some(block);
        }
        Ok(completed)
    }

    fn copied_host(&self, block: u32, staging: Option<usize>, event: &Ticket) {
        self.host_use[block as usize].borrow_mut().event = Some(event.clone());
        if let Some(index) = staging {
            self.stages[index].borrow_mut().event = Some(event.clone());
        }
    }

    fn finish(&self) -> Result<()> {
        if self.backing.is_some() {
            self.h2d.synchronize()?;
        }
        for index in 0..self.stages.len() {
            self.flush_stage(index)?;
        }
        for block in 0..self.host_use.len() {
            self.ready_host(block as u32)?;
        }
        Ok(())
    }

    pub fn ready_weights(&self, weights: &[ExpertWeights]) -> Result<()> {
        let arena = self
            .host
            .as_ref()
            .context("CPU host weights have no host arena")?;
        for weights in weights {
            self.ready_host(u32::try_from(
                weights.gate_up.start / arena.block_elements(),
            )?)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Physical {
    pointer: u64,
    copy: Option<Ticket>,
    reader: Option<Ticket>,
    waited: bool,
}

pub(super) struct Load<'a> {
    pub capacity: &'a crate::experts::plan::Capacity,
    pub admission: crate::experts::plan::Admission,
    pub initial: &'a [Vec<u32>],
    pub catalogue: Arc<ExpertCatalogue>,
    pub directory: &'a Path,
    pub engines: u32,
    pub max_rows: u32,
    pub swiglu_rounding: crate::model_spec::SwigluRounding,
    pub swiglu_limit: f32,
    pub routed_weight: crate::model_spec::RoutedWeightBoundary,
}

struct PeerSlots {
    gpu0_layers: u32,
    upload: Arc<CudaStream>,
    gate_up: CudaSlice<u64>,
    down: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    synced: CudaEvent,
    publish: CudaFunction,
}

pub(in crate::backend::cuda::engine) struct Slots {
    pub storage: super::super::super::weights::ExpertStorage,
    gate_up: CudaSlice<u64>,
    down: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    peer: Option<PeerSlots>,
    pub published: Vec<u64>,
    policy: Residency,
    catalogue: Arc<ExpertCatalogue>,
    disk: Option<Reader>,
    transfers: Transfers,
    events: Events,
    pub(super) refill: CudaEvent,
    physical: Vec<Physical>,
    publish: CudaFunction,
    routes: Vec<crate::experts::runtime::Route>,
    host_experts: Vec<u32>,
    host_weights: Vec<ExpertWeights>,
    moves: Vec<PromptMove>,
    readers: Vec<u32>,
    disk_cpu: Option<DiskExperts>,
    pub request_open: bool,
    #[cfg(all(test, feature = "cuda"))]
    pending_reads: u64,
    #[cfg(all(test, feature = "cuda"))]
    delay_request: bool,
}

fn check_publish(
    experts: u32,
    incoming: u32,
    victim: Option<u32>,
    gu: u64,
    down: u64,
    generation: u64,
    previous: u64,
) -> Result<()> {
    ensure!(
        experts > 0
            && experts <= i32::MAX as u32
            && incoming < experts
            && victim.is_none_or(|victim| victim < experts && victim != incoming),
        "expert publication bounds: experts={experts} incoming={incoming} victim={victim:?}"
    );
    ensure!(
        gu > 0 && down > 0 && gu.is_multiple_of(16) && down.is_multiple_of(16) && generation > 0,
        "expert publication pointer/generation: gu={gu:#x} down={down:#x} generation={generation}"
    );
    ensure!(
        previous < generation,
        "expert publication epoch: incoming={incoming} previous={previous} generation={generation}"
    );
    Ok(())
}

fn check_lifetime(expected: [Option<u64>; 2], owned: [Option<u64>; 2]) -> Result<()> {
    ensure!(
        expected == owned,
        "expert slot lifetime differs: residency={expected:?} backend={owned:?}"
    );
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
pub(super) fn delay_stream(stream: &Arc<CudaStream>) -> Result<()> {
    unsafe extern "C" fn delay(_: *mut std::ffi::c_void) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    stream.context().bind_to_thread()?;
    // SAFETY: the callback uses no CUDA API, shared data, or outstanding device work.
    unsafe {
        cudarc::driver::sys::cuLaunchHostFunc(
            stream.cu_stream(),
            Some(delay),
            std::ptr::null_mut(),
        )
        .result()?;
    }
    Ok(())
}

impl Slots {
    pub(super) fn load(
        device: &Device,
        peer_device: Option<&Device>,
        geometry: ExpertGeometry,
        input: Load<'_>,
    ) -> Result<Self> {
        let total_started = std::time::Instant::now();
        ensure!(
            input.catalogue.geometry() == geometry,
            "expert slot/catalogue geometry differs"
        );
        ensure!(
            input.capacity.layout == *input.catalogue.layout(),
            "expert slot/catalogue layout differs: planned={:?} catalogue={:?}",
            input.capacity.layout,
            input.catalogue.layout()
        );
        ensure!(
            input.engines > 0,
            "expert setup has no measured copy engines"
        );
        let phase_started = std::time::Instant::now();
        let has_peer = input.capacity.gpu0_layers < geometry.layers;
        let peer_streams = if has_peer {
            let peer = peer_device.context("multi-GPU expert capacity requires peer device")?;
            Some((&peer.stream, &peer.upload))
        } else {
            None
        };
        let mut storage = super::super::super::weights::ExpertStorage::new(
            &device.stream,
            &device.upload,
            peer_streams,
            geometry,
            input.capacity,
        )?;
        let storage_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        if input.capacity.pinned {
            storage.register_host()?;
        }
        let host_register_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let disk = Reader::new(
            device,
            has_peer,
            DiskReader::open(&input.catalogue, input.directory, input.engines.min(2))?,
            &input.catalogue,
        )?;
        let reader_open_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let (policy, host_load) = Residency::load(crate::experts::runtime::LoadInputs {
            geometry,
            capacity: input.capacity.clone(),
            admission: input.admission,
            max_rows: input.max_rows,
            initial: input.initial,
            host: storage.host(),
            catalogue: &input.catalogue,
            reader: &disk.disk,
        })?;
        let host_fill_ms = host_load.seconds * 1000.0;
        let phase_started = std::time::Instant::now();
        let physical_count = input.capacity.slots_total() as usize;
        let mut physical = (0..physical_count)
            .map(|_| Physical::default())
            .collect::<Vec<_>>();
        let bytes = storage.layout().slot_bytes();
        for layer in 0..geometry.layers {
            for rank in 0..storage.layout().slot_count(layer)? {
                let index = storage.layout().resident_slot(layer, rank)?.start / bytes;
                ensure!(
                    index < physical.len(),
                    "physical expert slot {index} outside allocation"
                );
                physical[index].pointer = storage.slot(layer, rank)?.0;
            }
        }
        ensure!(
            physical.iter().all(|slot| slot.pointer != 0),
            "physical expert slot map has a hole"
        );
        let physical_map_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let transfers = Transfers::new(
            device,
            &storage,
            input.capacity.pinned,
            input.engines,
            if policy.ledger().has_full_host_backing() {
                input.capacity.host_blocks
            } else {
                0
            },
        )?;
        let transfers_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let events = Events::new(
            &device.ctx,
            2 * physical_count
                + transfers.host_use.len()
                + transfers.stages.len()
                + disk.copied.len()
                + 2 * input.capacity.prefill_staging_slots as usize
                + 4,
        )?;
        let events_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let logical = geometry.layers as usize * geometry.experts as usize;
        let peer = if has_peer {
            let peer = peer_device.context("multi-GPU expert capacity requires peer device")?;
            let gate_up = peer.stream.alloc_zeros(logical)?;
            let down = peer.stream.alloc_zeros(logical)?;
            let valid = peer.stream.alloc_zeros(logical)?;
            let generation = peer.stream.alloc_zeros(logical)?;
            peer.upload.wait(&peer.stream.record_event(None)?)?;
            Some(PeerSlots {
                gpu0_layers: input.capacity.gpu0_layers,
                upload: peer.upload.clone(),
                gate_up,
                down,
                valid,
                generation,
                synced: peer.ctx.new_event(None)?,
                publish: super::super::super::cubin::module(&peer.ctx, "moe_route")?
                    .load_function("moe_slot_publish")?,
            })
        } else {
            None
        };
        let mut slots = Self {
            storage,
            gate_up: device.stream.alloc_zeros(logical)?,
            down: device.stream.alloc_zeros(logical)?,
            valid: device.stream.alloc_zeros(logical)?,
            generation: device.stream.alloc_zeros(logical)?,
            peer,
            published: vec![0; logical],
            policy,
            catalogue: input.catalogue.clone(),
            disk: Some(disk),
            transfers,
            events,
            refill: device.ctx.new_event(None)?,
            physical,
            publish: super::super::super::cubin::module(&device.ctx, "moe_route")?
                .load_function("moe_slot_publish")?,
            routes: Vec::with_capacity(input.max_rows as usize * geometry.top_k as usize),
            host_experts: Vec::with_capacity(geometry.experts as usize),
            host_weights: Vec::with_capacity(geometry.experts as usize),
            moves: Vec::with_capacity(physical_count),
            readers: Vec::with_capacity(
                physical_count.max(input.max_rows as usize * geometry.top_k as usize),
            ),
            disk_cpu: if input.capacity.disk_workers > 0 {
                Some(DiskExperts::new_with_swiglu_rounding(
                    input.catalogue.clone(),
                    input.directory,
                    input.capacity.disk_workers,
                    input.max_rows,
                    input.swiglu_rounding,
                    input.routed_weight,
                    input.swiglu_limit,
                )?)
            } else {
                None
            },
            request_open: false,
            #[cfg(all(test, feature = "cuda"))]
            pending_reads: 0,
            #[cfg(all(test, feature = "cuda"))]
            delay_request: false,
        };
        let slots_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        device.upload.wait(&device.stream.record_event(None)?)?;
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                if slots.policy.ledger().tier(layer, expert)? == crate::experts::plan::Tier::Gpu {
                    let slot = slots.policy.ledger().slot(layer, expert)?;
                    let generation = slots.policy.generation(layer, expert)?;
                    let block = slots.policy.host_block(layer, expert)?;
                    let source = if slots.policy.ledger().has_full_host_backing()
                        && block != crate::experts::runtime::NO_BLOCK
                    {
                        MoveSource::Host(block)
                    } else {
                        MoveSource::Disk
                    };
                    slots.install(layer, expert, None, slot, generation, source)?;
                }
            }
        }
        slots.refill.record(&slots.transfers.h2d)?;
        let resident_upload_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "{}",
            serde_json::json!({"expert_slots_phase": {
                "storage_new_ms": storage_new_ms,
                "host_register_ms": host_register_ms,
                "reader_open_ms": reader_open_ms,
                "host_fill_ms": host_fill_ms,
                "host_fill_bytes": host_load.host_bytes,
                "physical_map_ms": physical_map_ms,
                "transfers_new_ms": transfers_new_ms,
                "events_new_ms": events_new_ms,
                "slots_new_ms": slots_new_ms,
                "resident_upload_ms": resident_upload_ms,
                "total_ms": total_started.elapsed().as_secs_f64() * 1000.0
            }})
        );
        Ok(slots)
    }

    pub fn bindings(&self, layer: usize, stream: &Arc<CudaStream>) -> Result<SlotBindings> {
        let geometry = self.storage.layout().geometry();
        ensure!(
            layer < geometry.layers as usize,
            "expert layer={layer} outside layers={}",
            geometry.layers
        );
        let offset = layer * geometry.experts as usize;
        if let Some(peer) = &self.peer
            && layer >= peer.gpu0_layers as usize
        {
            return Ok(SlotBindings {
                gate_up: view(&peer.gate_up, stream, offset, [geometry.experts], [1])?,
                down: view(&peer.down, stream, offset, [geometry.experts], [1])?,
                valid: view(&peer.valid, stream, offset, [geometry.experts], [1])?,
                generation: view(&peer.generation, stream, offset, [geometry.experts], [1])?,
            });
        }
        Ok(SlotBindings {
            gate_up: view(&self.gate_up, stream, offset, [geometry.experts], [1])?,
            down: view(&self.down, stream, offset, [geometry.experts], [1])?,
            valid: view(&self.valid, stream, offset, [geometry.experts], [1])?,
            generation: view(&self.generation, stream, offset, [geometry.experts], [1])?,
        })
    }

    fn publish(
        &mut self,
        layer: u32,
        expert: u32,
        victim: Option<u32>,
        slot: u32,
        generation: u64,
    ) -> Result<Ticket> {
        let geometry = self.storage.layout().geometry();
        let pointer = self
            .physical
            .get(slot as usize)
            .context("publish physical slot outside allocation")?
            .pointer;
        let down = pointer
            .checked_add(self.catalogue.layout().down_offset())
            .context("expert down pointer overflow")?;
        let offset = layer as usize * geometry.experts as usize;
        let previous = *self
            .published
            .get(offset + expert as usize)
            .context("expert publication layer/expert outside generation table")?;
        check_publish(
            geometry.experts,
            expert,
            victim,
            pointer,
            down,
            generation,
            previous,
        )?;
        let on_peer = self
            .peer
            .as_ref()
            .is_some_and(|peer| layer >= peer.gpu0_layers);
        if on_peer {
            let peer = self.peer.as_ref().context("peer slots missing")?;
            let stream = peer.upload.clone();
            let function = peer.publish.clone();
            let tables = self.bindings(layer as usize, &stream)?;
            stream.context().bind_to_thread()?;
            // SAFETY: checked layer/expert extents and owned peer tables cover both scalar updates.
            unsafe {
                stream
                    .launch_builder(&function)
                    .arg(&tables.gate_up.ptr)
                    .arg(&tables.down.ptr)
                    .arg(&tables.valid.ptr)
                    .arg(&tables.generation.ptr)
                    .arg(&(geometry.experts as i32))
                    .arg(&(expert as i32))
                    .arg(&victim.map_or(-1, |value| value as i32))
                    .arg(&pointer)
                    .arg(&down)
                    .arg(&generation)
                    .launch(super::super::super::ops::grid(1, 1))?;
            }
            let peer = self.peer.as_ref().context("peer slots missing")?;
            peer.synced.record(&peer.upload)?;
            self.transfers.h2d.wait(&peer.synced)?;
            self.transfers.h2d.context().bind_to_thread()?;
        } else {
            #[cfg(feature = "cuda")]
            let tables = self.bindings(layer as usize, &self.transfers.h2d)?;
            #[cfg(feature = "rocm")]
            let tables = self.bindings(layer as usize, self.gate_up.stream())?;
            let stream = &self.transfers.h2d;
            // SAFETY: apply_move waits slot copy/reader tickets on h2d; checked table updates precede the publication ticket waited by prepare_routes.
            unsafe {
                stream
                    .launch_builder(&self.publish)
                    .arg(&tables.gate_up.ptr)
                    .arg(&tables.down.ptr)
                    .arg(&tables.valid.ptr)
                    .arg(&tables.generation.ptr)
                    .arg(&(geometry.experts as i32))
                    .arg(&(expert as i32))
                    .arg(&victim.map_or(-1, |value| value as i32))
                    .arg(&pointer)
                    .arg(&down)
                    .arg(&generation)
                    .launch(super::super::super::ops::grid(1, 1))?;
            }
        }
        let event = self.events.record(&self.transfers.h2d)?;
        self.physical[slot as usize].copy = Some(event.clone());
        self.physical[slot as usize].reader = None;
        self.physical[slot as usize].waited = false;
        if let Some(victim) = victim {
            self.published[offset + victim as usize] = 0;
        }
        self.published[offset + expert as usize] = generation;
        self.policy.record_copy(&[slot], event.id)?;
        Ok(event)
    }

    fn install(
        &mut self,
        layer: u32,
        expert: u32,
        victim: Option<u32>,
        slot: u32,
        generation: u64,
        source: MoveSource,
    ) -> Result<()> {
        let on_peer = self
            .peer
            .as_ref()
            .is_some_and(|peer| layer >= peer.gpu0_layers);
        self.transfers.h2d.context().bind_to_thread()?;
        let (pointer, staging, ring) = match source {
            MoveSource::Host(block) => {
                let (pointer, staging) = self.transfers.source_host(block)?;
                (pointer, staging, None)
            }
            MoveSource::Disk => {
                let disk = self
                    .disk
                    .as_mut()
                    .context("decode disk reader is owned by prefill IO")?;
                let ring = disk.read(layer, expert, &self.catalogue)?;
                let values = disk.disk.values(ring)?;
                let pointer = values.as_ptr() as u64;
                if on_peer {
                    let peer = self.peer.as_ref().context("peer slots missing")?;
                    peer.upload.context().bind_to_thread()?;
                    // SAFETY: the portable registered ring guard remains alive until the copy is submitted.
                    unsafe {
                        copy::htod_raw(
                            self.physical[slot as usize].pointer,
                            pointer as *const std::ffi::c_void,
                            usize::try_from(self.catalogue.slot_bytes())?,
                            peer.upload.as_ref(),
                        )
                    }
                    .context("copy native disk expert into its peer physical slot")?;
                    peer.synced.record(&peer.upload)?;
                    self.transfers.h2d.wait(&peer.synced)?;
                    self.transfers.h2d.context().bind_to_thread()?;
                } else {
                    // SAFETY: the registered ring guard remains alive until the copy is submitted.
                    unsafe {
                        copy::htod_raw(
                            self.physical[slot as usize].pointer,
                            pointer as *const std::ffi::c_void,
                            usize::try_from(self.catalogue.slot_bytes())?,
                            self.transfers.h2d.as_ref(),
                        )
                    }
                    .context("copy native disk expert into its physical slot")?;
                }
                let copy = self.events.record(&self.transfers.h2d)?;
                values.copied(copy.id)?;
                disk.copied[ring as usize] = Some(copy);
                (pointer, None, Some(ring))
            }
        };
        if ring.is_none() {
            if on_peer {
                let peer = self.peer.as_ref().context("peer slots missing")?;
                peer.upload.context().bind_to_thread()?;
                // SAFETY: the source remains claimed or staged through the recorded copy.
                unsafe {
                    copy::htod_raw(
                        self.physical[slot as usize].pointer,
                        pointer as *const std::ffi::c_void,
                        usize::try_from(self.catalogue.slot_bytes())?,
                        peer.upload.as_ref(),
                    )
                }
                .context("copy native host expert into its peer physical slot")?;
                peer.synced.record(&peer.upload)?;
                self.transfers.h2d.wait(&peer.synced)?;
                self.transfers.h2d.context().bind_to_thread()?;
            } else {
                self.transfers.h2d.context().bind_to_thread()?;
                // SAFETY: the pinned source remains claimed or staged through the recorded copy.
                unsafe {
                    copy::htod_raw(
                        self.physical[slot as usize].pointer,
                        pointer as *const std::ffi::c_void,
                        usize::try_from(self.catalogue.slot_bytes())?,
                        self.transfers.h2d.as_ref(),
                    )
                }
                .context("copy native host expert into its physical slot")?;
            }
        }
        let event = self.publish(layer, expert, victim, slot, generation)?;
        if let MoveSource::Host(block) = source {
            self.transfers.copied_host(block, staging, &event);
        }
        Ok(())
    }

    fn apply_move(&mut self, movement: PromptMove) -> Result<()> {
        let physical = self
            .physical
            .get(movement.slot as usize)
            .context("move slot outside allocation")?;
        let lifetime = self.policy.reclaim(movement.slot)?;
        check_lifetime(
            lifetime,
            [
                physical.copy.as_ref().map(|t| t.id),
                physical.reader.as_ref().map(|t| t.id),
            ],
        )?;
        for id in lifetime.into_iter().flatten() {
            self.events.wait(&self.transfers.h2d, id)?;
            self.events.wait(&self.transfers.d2h, id)?;
        }
        if let Some(id) = movement.dest_wait {
            self.events.wait(&self.transfers.d2h, id)?;
        }
        if let Some(slot) = movement.chain_slot {
            let ticket = self
                .physical
                .get(slot as usize)
                .and_then(|slot| slot.copy.as_ref())
                .context("chained swap has no preceding copy")?;
            self.events.wait(&self.transfers.d2h, ticket.id)?;
        }
        let slot = self
            .physical
            .get_mut(movement.slot as usize)
            .context("expert move targets an unallocated physical slot")?;
        slot.copy = None;
        slot.reader = None;
        if let Some(block) = movement.dest_block {
            ensure!(
                movement.victim.is_some(),
                "expert move writes back without a victim"
            );
            let event = self.transfers.write_back(
                slot.pointer,
                block,
                usize::try_from(self.catalogue.slot_bytes())?,
                &mut self.events,
            )?;
            self.transfers.h2d.wait(&event.event)?;
        }
        self.install(
            movement.layer,
            movement.expert,
            movement.victim,
            movement.slot,
            movement.generation,
            movement.source,
        )
    }

    pub(super) fn begin_request(&mut self, frequency: &[Vec<f64>]) -> Result<()> {
        self.moves.clear();
        self.moves
            .extend_from_slice(self.policy.begin_request(frequency)?);
        for index in 0..self.moves.len() {
            #[cfg(all(test, feature = "cuda"))]
            if index + 1 == self.moves.len() && std::mem::take(&mut self.delay_request) {
                delay_stream(&self.transfers.h2d)?;
            }
            self.apply_move(self.moves[index])?;
        }
        self.refill.record(&self.transfers.h2d)?;
        self.request_open = true;
        Ok(())
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(in crate::backend::cuda::engine) fn capture_placement(
        &self,
    ) -> Result<Vec<(Tier, Option<u32>)>> {
        let geometry = self.catalogue.geometry();
        let mut placement = Vec::with_capacity((geometry.layers * geometry.experts) as usize);
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                let tier = self.policy.ledger().tier(layer, expert)?;
                let slot = if tier == Tier::Gpu {
                    Some(self.policy.ledger().slot(layer, expert)?)
                } else {
                    None
                };
                placement.push((tier, slot));
            }
        }
        Ok(placement)
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(in crate::backend::cuda::engine) fn pin_capture_request(
        &mut self,
        frequency: &[Vec<f64>],
    ) -> Result<Vec<(Tier, Option<u32>)>> {
        if self.request_open {
            self.end_request()?;
        }
        self.begin_request(frequency)?;
        self.end_request()?;
        self.capture_placement()
    }

    pub fn end_request(&mut self) -> Result<()> {
        self.policy.end_request()?;
        self.transfers.finish()?;
        if let Some(disk) = &self.disk_cpu {
            disk.check_ready()?;
        }
        self.request_open = false;
        Ok(())
    }

    pub fn begin_token(&mut self) -> Result<()> {
        self.policy.begin_token()
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(in crate::backend::cuda::engine) fn freeze_probe_admissions(&mut self) -> Result<()> {
        self.transfers.finish()?;
        self.policy.freeze_probe_admissions()
    }

    pub fn finish_token(&mut self) -> Result<()> {
        self.moves.clear();
        for published in self.policy.finish_token()? {
            let source = match (published.from_host, published.source_block) {
                (true, Some(block)) => MoveSource::Host(block),
                (false, None) => MoveSource::Disk,
                other => anyhow::bail!("admission source contract differs: {other:?}"),
            };
            self.moves.push(PromptMove {
                layer: published.layer,
                expert: published.expert,
                victim: Some(published.victim),
                source,
                dest_block: published.dest_block,
                dest_wait: published.dest_wait,
                chain_slot: None,
                slot: published.slot,
                generation: published.generation,
            });
        }
        for index in 0..self.moves.len() {
            #[cfg(all(test, feature = "cuda"))]
            if index + 1 == self.moves.len() && std::mem::take(&mut self.delay_request) {
                delay_stream(&self.transfers.h2d)?;
            }
            self.apply_move(self.moves[index])?;
        }
        Ok(())
    }

    fn prepare_routes(&mut self, device: &Device, layer: u32, decode: &Decode) -> Result<()> {
        let geometry = self.catalogue.geometry();
        let (ids, weights) = decode.routed()?;
        self.routes.clear();
        self.readers.clear();
        self.host_experts.clear();
        self.host_weights.clear();
        for (choice, (&id, &weight)) in ids.iter().zip(weights).enumerate() {
            ensure!(
                id >= 0 && (id as u32) < geometry.experts,
                "route layer={layer} choice={choice} expert={id} outside {}",
                geometry.experts
            );
            let expert = id as u32;
            self.routes.push(crate::experts::runtime::Route {
                row: choice as u32 / geometry.top_k,
                choice: choice as u32,
                expert,
                weight,
            });
            if self.policy.ledger().tier(layer, expert)? == Tier::Gpu {
                let slot = self.policy.ledger().slot(layer, expert)?;
                let physical = &mut self.physical[slot as usize];
                if !physical.waited {
                    let id = self
                        .policy
                        .first_read(slot)?
                        .context("resident expert has no copy event")?;
                    ensure!(
                        physical.copy.as_ref().is_some_and(|copy| copy.id == id),
                        "slot={slot} copy event differs from residency id={id}"
                    );
                    #[cfg(all(test, feature = "cuda"))]
                    if !physical
                        .copy
                        .as_ref()
                        .context("pending copy owner missing")?
                        .event
                        .try_is_complete()?
                    {
                        self.pending_reads += 1;
                    }
                    self.events.wait(&device.stream, id)?;
                    physical.waited = true;
                }
                if !self.readers.contains(&slot) {
                    self.readers.push(slot);
                }
            }
        }
        for expert in 0..geometry.experts {
            if !self.routes.iter().any(|route| route.expert == expert)
                || self.policy.ledger().tier(layer, expert)? != Tier::Host
            {
                continue;
            }
            let block = self.policy.host_block(layer, expert)?;
            let range = self
                .transfers
                .host
                .as_ref()
                .context("host tier has no arena")?
                .block_range(block)?;
            self.host_experts.push(expert);
            self.host_weights.push(ExpertWeights::from_slot(
                Shape {
                    hidden: geometry.hidden as usize,
                    intermediate: geometry.intermediate as usize,
                },
                range,
                self.catalogue.layout(),
            )?);
        }
        Ok(())
    }

    pub fn run(
        &mut self,
        device: &Device,
        decode: &mut Decode,
        layer: u32,
        ffn: BoundFfn,
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let prewake = layer == self.catalogue.geometry().first_layer;
        if prewake {
            decode.pool.prewake()?;
        }
        if let Err(error) = decode.route(device, layer as usize, &ffn, x) {
            if prewake {
                decode.pool.stop_prewake()?;
            }
            return Err(error);
        }
        self.complete(device, decode, layer, ffn, (x, y), None)
    }

    pub(in crate::backend::cuda::engine) fn all_resident(&self) -> bool {
        let geometry = self.catalogue.geometry();
        self.physical.len() == geometry.layers as usize * geometry.experts as usize
    }

    pub(super) fn record_all_readers(&mut self, device: &Device) -> Result<()> {
        self.readers.clear();
        self.readers.extend(0..u32::try_from(self.physical.len())?);
        let event = self.events.record(&device.stream)?;
        self.policy.record_readers(&self.readers, event.id)?;
        for physical in &mut self.physical {
            physical.reader = Some(event.clone());
        }
        Ok(())
    }

    pub(super) fn record_peer_readers(&mut self, device: &Device) -> Result<()> {
        let Some(peer) = &self.peer else {
            return Ok(());
        };
        let geometry = self.catalogue.geometry();
        let peer_slots = geometry
            .layers
            .saturating_sub(peer.gpu0_layers)
            .checked_mul(geometry.experts)
            .context("peer slot count overflow")?;
        if peer_slots == 0 {
            return Ok(());
        }
        ensure!(
            (peer_slots as usize) <= self.physical.len(),
            "peer slots={peer_slots} exceed physical slots={}",
            self.physical.len()
        );
        self.readers.clear();
        self.readers.extend(0..peer_slots);
        let event = self.events.record(&device.stream)?;
        self.policy.record_readers(&self.readers, event.id)?;
        for physical in &mut self.physical[..peer_slots as usize] {
            physical.waited = true;
            physical.reader = Some(event.clone());
        }
        Ok(())
    }

    pub(super) fn run_graph(
        &mut self,
        device: &Device,
        decode: &mut Decode,
        layer: u32,
        graph_layer: usize,
        ffn: BoundFfn,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
    ) -> Result<()> {
        let prewake = layer == self.catalogue.geometry().first_layer;
        if prewake {
            decode.pool.prewake()?;
        }
        if let Err(error) = decode.read_route(device, layer as usize, &ffn, activation.0) {
            if prewake {
                decode.pool.stop_prewake()?;
            }
            return Err(error);
        }
        self.complete(device, decode, layer, ffn, activation, Some(graph_layer))
    }

    fn complete(
        &mut self,
        device: &Device,
        decode: &mut Decode,
        layer: u32,
        ffn: BoundFfn,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
        graph: Option<usize>,
    ) -> Result<()> {
        let result = self.complete_inner(device, decode, layer, ffn, activation, graph);
        if result.is_err() {
            decode.pool.stop_prewake()?;
        }
        result
    }

    fn complete_inner(
        &mut self,
        device: &Device,
        decode: &mut Decode,
        layer: u32,
        ffn: BoundFfn,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
        graph: Option<usize>,
    ) -> Result<()> {
        self.prepare_routes(device, layer, decode)?;
        let bindings = self.bindings(layer as usize, decode.slot_stream(device, layer as usize))?;
        let dispatch = self.policy.dispatch(layer, &self.routes)?;
        ensure!(
            dispatch.host_experts == self.host_experts,
            "host expert compact order differs from residency at layer={layer}"
        );
        if dispatch.host.is_empty() {
            decode.pool.stop_prewake()?;
        }
        decode.finish_with(
            device,
            ffn,
            bindings,
            Dispatch {
                active: &dispatch.active,
                generation: &dispatch.generation,
                host: self.transfers.host.clone(),
                experts: &self.host_weights,
                assignments: &dispatch.host,
                miss_rows: &dispatch.miss_rows,
                host_dma: Some(&self.transfers),
                disk: &dispatch.disk,
                disk_cpu: self.disk_cpu.as_mut(),
                layer,
            },
            activation,
            graph,
        )?;
        if !self.readers.is_empty() {
            let event = self.events.record(&device.stream)?;
            self.policy.record_readers(&self.readers, event.id)?;
            for &slot in &self.readers {
                self.physical[slot as usize].reader = Some(event.clone());
            }
        }
        Ok(())
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn finish_routed(
        &mut self,
        device: &Device,
        decode: &mut Decode,
        layer: u32,
        ffn: BoundFfn,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
    ) -> Result<()> {
        self.complete(device, decode, layer, ffn, activation, None)
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn trace(&self, trace: &mut Trace) -> Result<()> {
        let geometry = self.catalogue.geometry();
        self.policy.ledger().check()?;
        let mut tiers = Vec::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                tiers.push(match self.policy.ledger().tier(layer, expert)? {
                    Tier::Gpu => 0.0,
                    Tier::Host => 1.0,
                    Tier::Disk => 2.0,
                });
            }
        }
        let shape = vec![geometry.layers as usize, geometry.experts as usize];
        trace.add("residency.tier".into(), shape.clone(), tiers)?;
        trace.add(
            "residency.generation".into(),
            shape,
            self.published
                .iter()
                .map(|&generation| generation as f32)
                .collect(),
        )?;
        trace.add(
            "stats.copy_wait_pending".into(),
            vec![1],
            vec![self.pending_reads as f32],
        )?;
        for (name, values) in [
            (
                "residency.moves",
                self.moves
                    .iter()
                    .flat_map(|movement| {
                        [
                            movement.layer as f32,
                            movement.expert as f32,
                            movement.victim.map_or(-1.0, |victim| victim as f32),
                        ]
                    })
                    .collect(),
            ),
            (
                "residency.source",
                self.moves
                    .iter()
                    .map(|movement| u8::from(matches!(movement.source, MoveSource::Disk)) as f32)
                    .collect(),
            ),
            (
                "residency.writeback",
                self.moves
                    .iter()
                    .map(|movement| movement.dest_block.map_or(-1.0, |block| block as f32))
                    .collect(),
            ),
        ] {
            let width = if name == "residency.moves" { 3 } else { 1 };
            trace.add(name.into(), vec![self.moves.len(), width], values)?;
        }
        Ok(())
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn delay_copies(&mut self) {
        self.delay_request = true;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn event_ids_reject_reused_handle_generations() -> Result<()> {
        let mut ids = vec![5, 6, 7, 8];
        assert_eq!(event_index(6, &ids)?, 1);
        ids[1] = 10;
        assert!(event_index(6, &ids).is_err());
        assert_eq!(event_index(10, &ids)?, 1);
        assert!(event_index(0, &ids).is_err());
        assert!(event_index(1, &[]).is_err());
        Ok(())
    }

    #[test]
    fn slot_publications_and_lifetimes_reject_stale_state() -> Result<()> {
        check_publish(16, 3, Some(1), 0x1000, 0x2000, 7, 0)?;
        check_publish(16, 3, None, 0x1000, 0x2000, 8, 7)?;
        assert!(check_publish(16, 3, None, 0x1000, 0x2000, 7, 7).is_err());
        assert!(check_publish(16, 3, None, 0x1000, 0x2000, 6, 7).is_err());
        assert!(check_publish(16, 16, Some(1), 0x1000, 0x2000, 7, 0).is_err());
        assert!(check_publish(16, 3, Some(3), 0x1000, 0x2000, 7, 0).is_err());
        assert!(check_publish(16, 3, Some(1), 0x1002, 0x2000, 7, 0).is_err());
        assert!(check_publish(16, 3, Some(1), 0x1000, 0x2000, 0, 0).is_err());
        check_lifetime([Some(3), Some(7)], [Some(3), Some(7)])?;
        assert!(check_lifetime([Some(3), Some(7)], [Some(3), None]).is_err());
        assert!(check_lifetime([Some(3), Some(7)], [Some(3), Some(6)]).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "authorized GPU window; device traps run in separate processes"]
    fn slot_publication_epochs_gpu() -> Result<()> {
        use cudarc::driver::sys::CUresult;
        use std::{io::Write, process::Command};

        let Ok(case) = std::env::var("FF_MOE_CHILD") else {
            for case in [
                "publication_valid",
                "publication_equal",
                "publication_older",
                "publication_victim",
            ] {
                let output = Command::new(std::env::current_exe()?)
                    .args([
                        "backend::cuda::engine::moe::residency::tests::slot_publication_epochs_gpu",
                        "--exact",
                        "--include-ignored",
                        "--test-threads=1",
                        "--nocapture",
                    ])
                    .env("FF_MOE_CHILD", case)
                    .output()?;
                let log = String::from_utf8_lossy(&output.stdout);
                let expected = if case == "publication_valid" { 0 } else { 3 };
                ensure!(
                    output.status.code() == Some(expected)
                        && log.contains("running 1 test")
                        && log.contains(&format!("PASS slot_publication_epoch case={case}")),
                    "publication child case={case} exit={:?} stdout={log} stderr={}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr)
                );
                print!("{log}");
            }
            return Ok(());
        };
        let device = Device::new(0)?;
        let stream = &device.stream;
        let module = super::super::super::super::cubin::module(&device.ctx, "moe_route")?;
        let function = module.load_function("moe_slot_publish")?;
        let cases = match case.as_str() {
            "publication_valid" => vec![(1, 7, 8, -1), (0, 99, 1, -1), (1, 7, u64::MAX, 1)],
            "publication_equal" => vec![(1, 7, 7, -1)],
            "publication_older" => vec![(1, 7, 6, -1)],
            "publication_victim" => vec![(1, 7, 8, 0)],
            _ => anyhow::bail!("unknown publication child case={case}"),
        };
        for (incoming_valid, previous, epoch, victim) in cases {
            let mut gate_up = stream.clone_htod(&[0x1000u64, 0x5000])?;
            let mut down = stream.clone_htod(&[0x2000u64, 0x6000])?;
            let mut valid = stream.clone_htod(&[incoming_valid, 1u32])?;
            let mut generation = stream.clone_htod(&[previous, 13u64])?;
            // SAFETY: the four tables each own two entries; the kernel stores metadata only.
            let launch = unsafe {
                stream
                    .launch_builder(&function)
                    .arg(&mut gate_up)
                    .arg(&mut down)
                    .arg(&mut valid)
                    .arg(&mut generation)
                    .arg(&2i32)
                    .arg(&0i32)
                    .arg(&victim)
                    .arg(&0x3000u64)
                    .arg(&0x4000u64)
                    .arg(&epoch)
                    .launch(super::super::super::super::ops::grid(1, 1))
            };
            let outcome = launch.and_then(|_| stream.synchronize());
            if case != "publication_valid" {
                let error = outcome
                    .context("DEFECT stale publication did not trap")
                    .expect_err("DEFECT stale publication did not trap");
                let driver = error
                    .downcast_ref::<cudarc::driver::DriverError>()
                    .context("publication child did not return a driver error")?;
                ensure!(
                    matches!(
                        driver.0,
                        CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
                            | CUresult::CUDA_ERROR_LAUNCH_FAILED
                    ),
                    "publication child case={case} unrelated driver error={driver:?}"
                );
                println!("PASS slot_publication_epoch case={case} error={driver:?}");
                std::io::stdout().flush()?;
                std::process::exit(3);
            }
            outcome?;
            let removed = victim == 1;
            ensure!(
                stream.clone_dtoh(&gate_up)? == [0x3000, if removed { 0 } else { 0x5000 }]
                    && stream.clone_dtoh(&down)? == [0x4000, if removed { 0 } else { 0x6000 }]
                    && stream.clone_dtoh(&valid)? == [1, u32::from(!removed)]
                    && stream.clone_dtoh(&generation)? == [epoch, if removed { 0 } else { 13 }],
                "DEFECT publication metadata differs previous={previous} epoch={epoch} victim={victim}"
            );
        }
        println!(
            "PASS slot_publication_epoch case={case} newer=true inactive=true victim_cleared=true"
        );
        Ok(())
    }
}
