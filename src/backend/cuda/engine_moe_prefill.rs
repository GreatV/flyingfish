//! Layer-major prefill over borrowed expert slots.
use super::*;
use crate::backend::cuda::moe::prefill::prefill_scratch_bytes;
use crate::backend::cuda::moe::prefill::{
    LANE16_ROWS, MoePrefillContribute, MoePrefillFinish, PrefillOps, arithmetic_mask,
};
use crate::experts::{
    plan::{Capacity, Reserves, ServiceTable},
    span::{SpanPlan, SpanRates, StagingLeases},
};
use crate::prefill::ExpertSpanLayout;
use std::fmt::Write;
use std::ops::Range;

#[path = "engine_moe_prefetch.rs"]
mod prefetch;
use prefetch::ReadAhead;

struct Span {
    base: u64,
    layout: ExpertSpanLayout,
    geometry: ExpertGeometry,
    positions: Vec<u32>,
}

impl Span {
    fn view<T: Copy, const R: usize>(
        &self,
        region: &Range<usize>,
        offset: usize,
        shape: [u32; R],
        stride: [u64; R],
    ) -> Result<Tensor<T, R>> {
        ensure!(
            offset < region.len() && region.end <= self.layout.bytes,
            "prefill span view outside region={region:?} offset={offset} bytes={}",
            self.layout.bytes
        );
        // SAFETY: the slot arena owns this borrowed range through refill; callers order all uses.
        unsafe {
            Tensor::new(
                self.base + (region.start + offset) as u64,
                shape,
                stride,
                (region.len() - offset) as u64,
            )
        }
    }

    fn hidden(&self, region: &Range<usize>, base: u32, rows: u32) -> Result<Tensor<bf16, 2>> {
        ensure!(
            rows > 0
                && base
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.layout.rows),
            "prefill hidden rows={rows} base={base} capacity={}",
            self.layout.rows
        );
        let hidden = self.geometry.hidden;
        self.view(
            region,
            base as usize * hidden as usize * 2,
            [rows, hidden],
            [u64::from(hidden), 1],
        )
    }

    fn choices<T: Copy>(
        &self,
        region: &Range<usize>,
        base: u32,
        rows: u32,
    ) -> Result<Tensor<T, 2>> {
        ensure!(
            rows > 0
                && base
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.layout.rows),
            "prefill choices rows={rows} base={base} capacity={}",
            self.layout.rows
        );
        let k = self.geometry.top_k;
        self.view(
            region,
            base as usize * k as usize * size_of::<T>(),
            [rows, k],
            [u64::from(k), 1],
        )
    }

    fn contributions(&self, base: u32, rows: u32) -> Result<Tensor<bf16, 3>> {
        ensure!(
            rows > 0
                && base
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.layout.rows),
            "prefill contributions rows={rows} base={base} capacity={}",
            self.layout.rows
        );
        let (k, h) = (self.geometry.top_k, self.geometry.hidden);
        self.view(
            &self.layout.contributions,
            base as usize * k as usize * h as usize * 2,
            [rows, k, h],
            [u64::from(k) * u64::from(h), u64::from(h), 1],
        )
    }
}

struct Tables {
    gate_up: CudaSlice<u64>,
    down: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    host_gate_up: PinnedHostSlice<u64>,
    host_down: PinnedHostSlice<u64>,
    host_valid: PinnedHostSlice<u32>,
    host_generation: PinnedHostSlice<u64>,
}

impl Tables {
    fn new(device: &Device, experts: usize, cohorts: usize) -> Result<Self> {
        let words = experts
            .checked_mul(cohorts)
            .context("prefill table extent overflow")?;
        Ok(Self {
            gate_up: device.stream.alloc_zeros(words)?,
            down: device.stream.alloc_zeros(words)?,
            valid: device.stream.alloc_zeros(words)?,
            generation: device.stream.alloc_zeros(words)?,
            host_gate_up: pinned_zeros(&device.ctx, words)?,
            host_down: pinned_zeros(&device.ctx, words)?,
            host_valid: pinned_zeros(&device.ctx, words)?,
            host_generation: pinned_zeros(&device.ctx, words)?,
        })
    }

    fn set(
        &mut self,
        experts: u32,
        cohort: usize,
        expert: u32,
        pointer: (u64, u64),
        epoch: u64,
    ) -> Result<()> {
        ensure!(
            expert < experts && epoch > 0,
            "prefill table expert={expert} experts={experts} epoch={epoch}"
        );
        let index = cohort * experts as usize + expert as usize;
        ensure!(
            index < self.host_gate_up.len(),
            "prefill cohort={cohort} exceeds table allocation"
        );
        self.host_gate_up.as_mut_slice()?[index] = pointer.0;
        self.host_down.as_mut_slice()?[index] = pointer.1;
        self.host_valid.as_mut_slice()?[index] = 1;
        self.host_generation.as_mut_slice()?[index] = epoch;
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        self.host_gate_up.as_mut_slice()?.fill(0);
        self.host_down.as_mut_slice()?.fill(0);
        self.host_valid.as_mut_slice()?.fill(0);
        self.host_generation.as_mut_slice()?.fill(0);
        Ok(())
    }

    fn upload(&mut self, stream: &Arc<CudaStream>, cohorts: usize, experts: u32) -> Result<()> {
        let words = cohorts
            .checked_mul(experts as usize)
            .context("prefill upload table extent overflow")?;
        ensure!(
            words > 0 && words <= self.host_gate_up.len(),
            "prefill upload cohorts={cohorts} outside table allocation"
        );
        // SAFETY: each pinned layer table stays immutable until the next layer's route receipt.
        unsafe {
            let (gu, _gu) = self.gate_up.device_ptr_mut(stream);
            let (down, _down) = self.down.device_ptr_mut(stream);
            let (valid, _valid) = self.valid.device_ptr_mut(stream);
            let (generation, _generation) = self.generation.device_ptr_mut(stream);
            result::memcpy_htod_async(
                gu,
                &self.host_gate_up.as_slice()?[..words],
                stream.cu_stream(),
            )?;
            result::memcpy_htod_async(
                down,
                &self.host_down.as_slice()?[..words],
                stream.cu_stream(),
            )?;
            result::memcpy_htod_async(
                valid,
                &self.host_valid.as_slice()?[..words],
                stream.cu_stream(),
            )?;
            result::memcpy_htod_async(
                generation,
                &self.host_generation.as_slice()?[..words],
                stream.cu_stream(),
            )?;
        }
        Ok(())
    }

    fn bindings(
        &self,
        stream: &Arc<CudaStream>,
        cohort: usize,
        experts: u32,
    ) -> Result<SlotBindings> {
        let offset = cohort
            .checked_mul(experts as usize)
            .context("prefill cohort table offset overflow")?;
        ensure!(
            offset
                .checked_add(experts as usize)
                .is_some_and(|end| end <= self.host_gate_up.len()),
            "prefill bindings cohort={cohort} outside table allocation"
        );
        Ok(SlotBindings {
            gate_up: view(&self.gate_up, stream, offset, [experts], [1])?,
            down: view(&self.down, stream, offset, [experts], [1])?,
            valid: view(&self.valid, stream, offset, [experts], [1])?,
            generation: view(&self.generation, stream, offset, [experts], [1])?,
        })
    }
}

struct Gpu {
    mask: CudaFunction,
    scratch: CudaSlice<u8>,
    tables: Tables,
    host_ids: PinnedHostSlice<i32>,
    host_weights: PinnedHostSlice<bf16>,
    mailbox: Arc<CudaStream>,
    staging_readers: Vec<Option<Ticket>>,
    staging_copy: Option<Ticket>,
    resident_slots: Vec<u32>,
    cpu_arithmetic: Vec<bool>,
    epoch: u64,
    read_ahead: ReadAhead,
}

struct Cpu {
    x: PinnedHostSlice<bf16>,
    out: PinnedHostSlice<bf16>,
    done: PinnedHostSlice<u32>,
    choices: Vec<u32>,
    rows: Vec<u32>,
    row_map: Vec<u32>,
    experts: Vec<ExpertWeights>,
    assignments: Vec<Assignment>,
    disk: Vec<crate::experts::runtime::Route>,
}

type Wave = (usize, usize, usize, bool, String);

pub(in crate::backend::cuda::engine) struct Prefill {
    geometry: ExpertGeometry,
    capacity: Capacity,
    reserves: Reserves,
    service: ServiceTable,
    workers: u32,
    cpu_ceiling: u32,
    h2d_bytes_per_second: f64,
    span: Option<Span>,
    gpu: Gpu,
    cpu: Cpu,
    frequency: Vec<Vec<f64>>,
    rows: u64,
    failed: bool,
    mixer_rows: usize,
    workspace_rows: usize,
    waves: Vec<Wave>,
    layer_names: Vec<String>,
    counts: Vec<u32>,
    resident: Vec<bool>,
    forecast_counts: Vec<u32>,
    forecast_resident: Vec<bool>,
    plan: SpanPlan,
    forecast_plan: SpanPlan,
    leases: StagingLeases,
    pub stats: [u64; 8],
}

impl Drop for Prefill {
    fn drop(&mut self) {
        self.gpu
            .mailbox
            .context()
            .record_err(self.gpu.mailbox.synchronize());
    }
}

/// CPU workspace rows/choices, constrained expert rows, and measured expert rows.
pub(in crate::backend::cuda::engine) fn cpu_bound(
    geometry: ExpertGeometry,
    slot_bytes: u64,
    rows: u32,
    service: &ServiceTable,
    h2d_bytes_per_second: f64,
) -> Result<(usize, usize, u32, u32)> {
    ensure!(
        rows > 0
            && service.max_rows() > 0
            && geometry.experts > 0
            && geometry.top_k > 0
            && geometry.top_k <= geometry.experts,
        "prefill CPU bound rows={rows} measured_rows={} geometry={geometry:?}",
        service.max_rows()
    );
    let mut counts = vec![0; geometry.experts as usize];
    let resident = vec![false; geometry.experts as usize];
    let mut plan = SpanPlan::new(&geometry, slot_bytes, 1, service.worker_count()?)?;
    let mut cpu_rows = 0;
    for row_count in 1..=service.max_rows().min(rows) {
        counts[0] = row_count;
        plan.update(
            &counts,
            &resident,
            row_count,
            1,
            &SpanRates {
                service,
                h2d_bytes_per_second,
                cpu_row_limit: service.max_rows(),
            },
        )?;
        if !plan.cpu().is_empty() {
            cpu_rows = row_count;
        }
    }
    let measured_cpu_rows = cpu_rows;
    cpu_rows = cpu_rows.min(LANE16_ROWS);
    let choices = (u64::from(rows) * u64::from(geometry.top_k))
        .min(u64::from(geometry.experts) * u64::from(cpu_rows))
        .max(1);
    ensure!(
        choices > 0 && choices <= 1 << 16,
        "prefill CPU assignment bound={choices} outside1..65536"
    );
    Ok((
        usize::try_from(choices.min(u64::from(rows)))?,
        usize::try_from(choices)?,
        cpu_rows,
        measured_cpu_rows,
    ))
}

impl Prefill {
    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn allocated_workspace_bytes(&self) -> u64 {
        self.gpu.scratch.len() as u64
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn capture_frequency(&self) -> Vec<Vec<f64>> {
        self.frequency.clone()
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn capture_resident_limit(
        &self,
        rows: usize,
    ) -> Result<u32> {
        let rows = u32::try_from(rows)?.min(self.capacity.span_rows);
        ensure!(rows > 0, "capture request has no rows");
        let borrowed = u32::try_from(
            self.reserves
                .span_bytes(rows)?
                .div_ceil(self.capacity.slot_bytes()),
        )?;
        self.capacity
            .slots_total()
            .checked_sub(borrowed)
            .context("capture borrow exceeds resident slots")
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn reference_geometry(&self) -> [usize; 4] {
        [
            self.capacity.span_rows as usize,
            self.capacity.wave_rows as usize,
            self.mixer_rows,
            self.workspace_rows,
        ]
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn reseed_request(&self, slots: &mut Slots) -> Result<()> {
        slots.end_request()?;
        slots.begin_request(&self.frequency)
    }

    pub(in crate::backend::cuda::engine) fn new(
        device: &Device,
        slots: &mut Slots,
        capacity: &Capacity,
        reserves: Reserves,
        service: ServiceTable,
        h2d_bytes_per_second: f64,
        windows: (usize, usize),
    ) -> Result<Self> {
        let (workspace_rows, mixer_rows) = windows;
        let geometry = slots.catalogue.geometry();
        ensure!(
            mixer_rows > 0 && mixer_rows <= workspace_rows,
            "prefill mixer_rows={mixer_rows} exceed workspace_rows={workspace_rows}"
        );
        ensure!(
            capacity.span_rows > 0
                && capacity.wave_rows > 0
                && capacity.wave_rows <= capacity.span_rows
                && (capacity.wave_rows * geometry.top_k * 2).is_multiple_of(16)
                && capacity.prefill_staging_slots > 0,
            "prefill span/wave/staging geometry span={} wave={} top_k={} staging={}",
            capacity.span_rows,
            capacity.wave_rows,
            geometry.top_k,
            capacity.prefill_staging_slots
        );
        ensure!(
            geometry.experts <= geometry.top_k * geometry.hidden,
            "prefill router logits exceed contribution arena: experts={} top_k={} hidden={}",
            geometry.experts,
            geometry.top_k,
            geometry.hidden
        );
        ensure!(
            h2d_bytes_per_second.is_finite() && h2d_bytes_per_second > 0.0,
            "prefill H2D rate={h2d_bytes_per_second}"
        );
        ExpertSpanLayout::new(
            &reserves,
            geometry,
            capacity.slot_bytes(),
            capacity.span_rows,
        )?
        .validate_wave(capacity.wave_rows, workspace_rows)?;
        let borrowed = reserves
            .span_bytes(capacity.span_rows)?
            .div_ceil(capacity.slot_bytes());
        let host_union = u64::from(geometry.layers) * u64::from(geometry.experts)
            - u64::from(capacity.slots_total());
        if capacity.disk_bytes == 0 && u64::from(capacity.host_blocks) >= host_union + borrowed {
            slots.disk = None;
        }
        let rings = slots
            .disk
            .as_ref()
            .map_or(0, |reader| reader.disk.ring_slots());
        let assignments = capacity.span_rows as usize * geometry.top_k as usize;
        let cpu = cpu_bound(
            geometry,
            capacity.slot_bytes(),
            capacity.span_rows,
            &service,
            h2d_bytes_per_second,
        )?;
        let workers = service.worker_count()?;
        let cohorts =
            (geometry.experts as usize).div_ceil(capacity.prefill_staging_slots as usize) + 1;
        Ok(Self {
            geometry,
            capacity: capacity.clone(),
            reserves,
            service,
            workers,
            cpu_ceiling: cpu.2,
            h2d_bytes_per_second,
            span: None,
            gpu: Gpu {
                mask: super::super::super::super::cubin::module(&device.ctx, "moe_route")?
                    .load_function("moe_route_mask")?,
                scratch: device
                    .stream
                    .alloc_zeros(usize::try_from(prefill_scratch_bytes(
                        capacity.wave_rows as usize,
                        capacity.span_rows as usize,
                        geometry.top_k as usize,
                        geometry.experts as usize,
                        geometry.hidden as usize,
                        geometry.intermediate as usize,
                        capacity.layout.fp8_slot(),
                    )?)?)?,
                tables: Tables::new(device, geometry.experts as usize, cohorts)?,
                host_ids: pinned_zeros(&device.ctx, assignments)?,
                host_weights: pinned_zeros(&device.ctx, assignments)?,
                mailbox: device.ctx.new_stream()?,
                staging_readers: vec![None; capacity.prefill_staging_slots as usize],
                staging_copy: None,
                resident_slots: Vec::with_capacity(geometry.experts as usize),
                cpu_arithmetic: vec![false; geometry.experts as usize],
                epoch: 0,
                read_ahead: ReadAhead::new(
                    capacity.prefill_staging_slots,
                    rings,
                    geometry.experts,
                )?,
            },
            cpu: Cpu {
                x: pinned_zeros(&device.ctx, cpu.0 * geometry.hidden as usize)?,
                out: pinned_zeros(&device.ctx, cpu.1 * geometry.hidden as usize)?,
                done: {
                    let mut words = pinned_zeros(&device.ctx, cpu.1)?;
                    words.as_mut_slice()?.fill(1);
                    words
                },
                choices: Vec::with_capacity(cpu.1),
                rows: Vec::with_capacity(cpu.0),
                row_map: vec![u32::MAX; capacity.span_rows as usize],
                experts: Vec::with_capacity(geometry.experts as usize),
                assignments: Vec::with_capacity(cpu.1),
                disk: Vec::with_capacity(cpu.1),
            },
            frequency: vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize],
            rows: 0,
            failed: false,
            mixer_rows,
            workspace_rows,
            waves: (0..capacity.span_rows as usize)
                .step_by(mixer_rows)
                .map(|base| {
                    (
                        base,
                        0,
                        0,
                        false,
                        String::with_capacity("prefill.".len() + usize::MAX.ilog10() as usize + 1),
                    )
                })
                .collect(),
            layer_names: (0..geometry.layers)
                .map(|layer| format!("layer.{layer}"))
                .collect(),
            counts: vec![0; geometry.experts as usize],
            resident: vec![false; geometry.experts as usize],
            forecast_counts: vec![0; geometry.experts as usize],
            forecast_resident: vec![false; geometry.experts as usize],
            plan: SpanPlan::new(
                &geometry,
                capacity.slot_bytes(),
                capacity.prefill_staging_slots,
                workers,
            )?,
            forecast_plan: SpanPlan::new(
                &geometry,
                capacity.slot_bytes(),
                capacity.prefill_staging_slots,
                workers,
            )?,
            leases: StagingLeases::new(capacity.prefill_staging_slots)?,
            stats: [0; 8],
        })
    }

    fn begin(&mut self, device: &Device, slots: &mut Slots, rows: u32) -> Result<()> {
        ensure!(self.span.is_none(), "prefill span is already borrowed");
        ensure!(
            !slots.request_open,
            "prefill cannot begin while a decode request is open; end_request or reset is required"
        );
        self.gpu.read_ahead.finish(slots)?;
        self.gpu.read_ahead.stats = [0; 6];
        let rows = rows.min(self.capacity.span_rows);
        let layout = ExpertSpanLayout::new(
            &self.reserves,
            self.geometry,
            self.capacity.slot_bytes(),
            rows,
        )?;
        layout.validate_wave(self.capacity.wave_rows.min(rows), self.workspace_rows)?;
        let borrowed = u32::try_from(
            self.reserves
                .span_bytes(rows)?
                .div_ceil(self.capacity.slot_bytes()),
        )?;
        let mut occupants = vec![None; self.capacity.slots_total() as usize];
        for layer in 0..self.geometry.layers {
            for expert in 0..self.geometry.experts {
                if slots.policy.ledger().tier(layer, expert)? == Tier::Gpu {
                    occupants[slots.policy.ledger().slot(layer, expert)? as usize] =
                        Some((layer, expert));
                }
            }
        }
        let borrow = slots.policy.borrow(borrowed)?;
        let (base, bytes) = slots.storage.borrowed_tail(borrowed)?;
        ensure!(
            layout.bytes <= bytes,
            "borrowed prefill bytes={bytes} required={}",
            layout.bytes
        );
        let mut victim = 0;
        for &position in &borrow.positions {
            let physical = &slots.physical[position as usize];
            let lifetime = slots.policy.reclaim(position)?;
            check_lifetime(
                lifetime,
                [
                    physical.copy.as_ref().map(|t| t.id),
                    physical.reader.as_ref().map(|t| t.id),
                ],
            )?;
            for id in lifetime.into_iter().flatten() {
                slots.events.wait(&device.stream, id)?;
                slots.events.wait(&slots.transfers.d2h, id)?;
            }
            if let Some((layer, expert)) = occupants[position as usize] {
                ensure!(
                    borrow.experts.get(victim) == Some(&(layer, expert)),
                    "borrowed victim order differs at physical slot={position}"
                );
                if borrow.write_back[victim] {
                    if let Some(id) = borrow.host_waits[victim] {
                        slots.events.wait(&slots.transfers.d2h, id)?;
                    }
                    let block = slots.policy.host_block(layer, expert)?;
                    let copied = slots.transfers.write_back(
                        physical.pointer,
                        block,
                        usize::try_from(self.capacity.slot_bytes())?,
                        &mut slots.events,
                    )?;
                    device.stream.wait(&copied.event)?;
                }
                let logical = layer as usize * self.geometry.experts as usize + expert as usize;
                // SAFETY: these scalar metadata words belong to the evicted logical expert; prior readers are ordered above.
                unsafe {
                    for (pointer, bytes) in [
                        (
                            slots.gate_up.device_ptr(&device.stream).0 + (logical * 8) as u64,
                            8,
                        ),
                        (
                            slots.down.device_ptr(&device.stream).0 + (logical * 8) as u64,
                            8,
                        ),
                        (
                            slots.valid.device_ptr(&device.stream).0 + (logical * 4) as u64,
                            4,
                        ),
                        (
                            slots.generation.device_ptr(&device.stream).0 + (logical * 8) as u64,
                            8,
                        ),
                    ] {
                        result::memset_d8_async(pointer, 0, bytes, device.stream.cu_stream())?;
                    }
                }
                slots.published[logical] = 0;
                victim += 1;
            }
            slots.physical[position as usize].copy = None;
            slots.physical[position as usize].reader = None;
            slots.physical[position as usize].waited = false;
        }
        ensure!(
            victim == borrow.experts.len(),
            "borrow did not visit every victim"
        );
        let ready = slots.events.record(&device.stream)?;
        slots.policy.record_copy(&borrow.positions, ready.id)?;
        for &position in &borrow.positions {
            slots.physical[position as usize].copy = Some(ready.clone());
        }
        self.frequency.iter_mut().for_each(|layer| layer.fill(0.0));
        self.rows = 0;
        self.stats = [0; 8];
        self.stats[4] = borrow.d2h_bytes;
        self.stats[5] = borrow.disk_bytes;
        self.span = Some(Span {
            base,
            layout,
            geometry: self.geometry,
            positions: borrow.positions,
        });
        Ok(())
    }

    fn end(&mut self, device: &Device, slots: &mut Slots) -> Result<()> {
        self.gpu.read_ahead.finish(slots)?;
        let span = self
            .span
            .as_ref()
            .context("prefill refill has no borrowed span")?;
        ensure!(self.rows > 0, "prefill request has no routed rows");
        let reader = slots.events.record(&device.stream)?;
        slots.policy.record_readers(&span.positions, reader.id)?;
        for &position in &span.positions {
            slots.physical[position as usize].reader = Some(reader.clone());
        }
        let positions = slots.policy.refill()?;
        ensure!(
            positions.len() == span.positions.len()
                && positions.iter().all(|slot| span.positions.contains(slot)),
            "prefill refill positions differ from borrowed span"
        );
        for layer in &mut self.frequency {
            for count in layer {
                *count /= self.rows as f64;
            }
        }
        slots.begin_request(&self.frequency)?;
        for movement in &slots.moves {
            self.stats[6 + usize::from(movement.source == MoveSource::Disk)] +=
                self.capacity.slot_bytes();
            if movement.dest_block.is_some() {
                self.stats[4] += self.capacity.slot_bytes();
            }
        }
        self.span = None;
        Ok(())
    }

    fn ffn(
        &mut self,
        device: &Device,
        slots: &mut Slots,
        decode: &mut Decode,
        layer: u32,
        rows: u32,
        ffn: BoundFfn,
    ) -> Result<()> {
        let BoundFfn::Moe { router, .. } = ffn else {
            anyhow::bail!("prefill layer={layer} has a dense FFN");
        };
        let span_rows = self
            .span
            .as_ref()
            .context("prefill FFN has no borrowed storage")?
            .layout
            .rows;
        ensure!(
            layer < self.geometry.layers && rows > 0 && rows <= span_rows,
            "prefill FFN layer={layer} rows={rows} geometry={:?}",
            self.geometry
        );
        let next = self.lookahead(slots, layer, rows)?;
        let span = self
            .span
            .as_ref()
            .context("prefill FFN has no borrowed storage")?;
        let stream = &device.stream;
        ensure!(
            self.gpu.scratch.len() as u64
                >= prefill_scratch_bytes(
                    self.capacity.wave_rows.min(rows) as usize,
                    rows as usize,
                    self.geometry.top_k as usize,
                    self.geometry.experts as usize,
                    self.geometry.hidden as usize,
                    self.geometry.intermediate as usize,
                    self.capacity.layout.fp8_slot(),
                )?,
            "prefill wave scratch bytes={} are below the rows={rows} wave={} bound",
            self.gpu.scratch.len(),
            self.capacity.wave_rows
        );
        let x = span.hidden(&span.layout.normalized, 0, rows)?;
        let ids = span.choices::<i32>(&span.layout.ids, 0, rows)?;
        let weights = span.choices::<bf16>(&span.layout.weights, 0, rows)?;
        let mut logits = TensorMut(span.view(
            &span.layout.contributions,
            0,
            [rows, self.geometry.experts],
            [u64::from(self.geometry.experts), 1],
        )?);
        decode.route.route_prefill(
            stream,
            x,
            router,
            &mut TensorMut(ids),
            &mut TensorMut(weights),
            &mut logits,
        )?;
        let routed = slots.events.record(stream)?;
        self.gpu.mailbox.wait(&routed.event)?;
        let choices = rows as usize * self.geometry.top_k as usize;
        // SAFETY: the route receipt copies exactly the live prefix into persistent pinned buffers.
        unsafe {
            result::memcpy_dtoh_async(
                &mut self.gpu.host_ids.as_mut_slice()?[..choices],
                ids.ptr,
                self.gpu.mailbox.cu_stream(),
            )?;
            result::memcpy_dtoh_async(
                &mut self.gpu.host_weights.as_mut_slice()?[..choices],
                weights.ptr,
                self.gpu.mailbox.cu_stream(),
            )?;
        }
        self.gpu.mailbox.synchronize()?;
        let host_ids = &self.gpu.host_ids.as_slice()?[..choices];
        let host_weights = &self.gpu.host_weights.as_slice()?[..choices];
        self.counts.fill(0);
        for (choice, (&expert, weight)) in host_ids.iter().zip(host_weights).enumerate() {
            ensure!(
                expert >= 0 && expert < self.geometry.experts as i32 && weight.to_f32().is_finite(),
                "prefill route layer={layer} choice={choice} expert={expert} weight={weight}"
            );
            let row_start = choice / self.geometry.top_k as usize * self.geometry.top_k as usize;
            ensure!(
                !host_ids[row_start..choice].contains(&expert),
                "prefill duplicate route layer={layer} choice={choice} expert={expert}"
            );
            self.counts[expert as usize] += 1;
        }
        for (expert, resident) in self.resident.iter_mut().enumerate() {
            *resident = slots.policy.ledger().tier(layer, expert as u32)? == Tier::Gpu;
        }
        self.plan.update(
            &self.counts,
            &self.resident,
            rows,
            self.capacity.wave_rows.min(rows),
            &SpanRates {
                service: &self.service,
                h2d_bytes_per_second: self.h2d_bytes_per_second,
                cpu_row_limit: self.cpu_ceiling,
            },
        )?;
        self.plan.pack_waves(&self.counts, host_ids)?;
        let plan = &self.plan;
        arithmetic_mask(&self.counts, &mut self.gpu.cpu_arithmetic)?;
        for &expert in plan.cpu() {
            ensure!(
                self.counts[expert as usize] <= self.cpu_ceiling,
                "prefill CPU expert layer={layer} expert={expert} rows={} exceed setup ceiling={}",
                self.counts[expert as usize],
                self.cpu_ceiling
            );
        }
        for (total, &count) in self.frequency[layer as usize].iter_mut().zip(&self.counts) {
            *total += f64::from(count);
        }
        if layer == 0 {
            self.rows += u64::from(rows);
        }
        self.gpu.read_ahead.begin_layer(slots, layer, plan)?;
        self.cpu.prepare(&self.gpu, span, slots, layer, plan)?;
        self.gpu
            .prepare_tables(&device.stream, slots, layer, plan)?;
        if !self.cpu.experts.is_empty() {
            slots.transfers.ready_weights(&self.cpu.experts)?;
        }
        let cpu_rows = self.cpu.rows.len();
        let cpu_choices = self.cpu.choices.len();
        let cpu_hidden = &self.cpu.x.as_slice()?[..cpu_rows * self.geometry.hidden as usize];
        let (gpu_result, host_result, disk_started) = {
            let pending = if self.cpu.assignments.is_empty() {
                None
            } else {
                Some(
                    decode.pool.submit(Job {
                        arena: slots
                            .transfers
                            .host
                            .as_ref()
                            .context("prefill CPU cohort has no host arena")?
                            .clone(),
                        hidden: cpu_hidden,
                        rows: cpu_rows,
                        experts: &self.cpu.experts,
                        assignments: &self.cpu.assignments,
                        out: &mut self.cpu.out.as_mut_slice()?
                            [..cpu_choices * self.geometry.hidden as usize],
                    })?,
                )
            };
            let disk_start = if self.cpu.disk.is_empty() {
                Ok(())
            } else {
                match slots.disk_cpu.as_mut() {
                    Some(pool) => pool.start(layer, cpu_hidden, cpu_rows, &self.cpu.disk),
                    None => Err(anyhow::anyhow!("prefill disk CPU routes have no workers")),
                }
            };
            let disk_started = !self.cpu.disk.is_empty() && disk_start.is_ok();
            let gpu_result = disk_start.and_then(|()| {
                self.gpu.contribute(
                    device,
                    slots,
                    span,
                    &mut decode.prefill,
                    (layer, plan, ffn, &mut self.leases),
                    next.then_some((layer + 1, &self.forecast_plan)),
                )
            });
            let host_result = match pending {
                Some(job) => decode.pool.finish(job),
                None => Ok(()),
            };
            (gpu_result, host_result, disk_started)
        };
        let disk_result = if !disk_started {
            Ok(())
        } else {
            slots
                .disk_cpu
                .as_mut()
                .context("prefill disk completion has no workers")?
                .finish(self.cpu.out.as_mut_slice()?)
        };
        gpu_result?;
        host_result?;
        disk_result?;
        self.cpu
            .upload(&self.gpu.mailbox, stream, span, &mut slots.events)?;
        // SAFETY: the complete routed union contains exactly rows*top_k u32 words.
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(
                span.base + span.layout.active.start as u64,
                1,
                choices,
                stream.cu_stream(),
            )
            .result()?;
        }
        self.stats[0] += 1;
        self.stats[1] += u64::from(rows);
        self.stats[2] += plan.streamed_bytes();
        self.stats[3] += cpu_choices as u64;
        Ok(())
    }

    fn lookahead(&mut self, slots: &Slots, layer: u32, rows: u32) -> Result<bool> {
        if layer + 1 == self.geometry.layers {
            return Ok(false);
        }
        let expected =
            u64::from(rows) * u64::from(self.geometry.top_k) / u64::from(self.geometry.experts);
        let copy_seconds = self.capacity.slot_bytes() as f64 / self.h2d_bytes_per_second;
        if expected <= u64::from(self.service.max_rows())
            || copy_seconds
                >= self
                    .service
                    .rows_time(self.service.max_rows(), 1, self.workers)?
        {
            return Ok(false);
        }
        for (expert, resident) in self.forecast_resident.iter_mut().enumerate() {
            *resident = slots.policy.ledger().tier(layer + 1, expert as u32)? == Tier::Gpu;
        }
        self.forecast_counts.fill(u32::try_from(expected)?);
        self.forecast_plan.update(
            &self.forecast_counts,
            &self.forecast_resident,
            rows,
            rows.min(self.capacity.wave_rows),
            &SpanRates {
                service: &self.service,
                h2d_bytes_per_second: self.h2d_bytes_per_second,
                cpu_row_limit: self.cpu_ceiling,
            },
        )?;
        Ok(true)
    }

    fn finish_wave(
        &mut self,
        device: &Device,
        decode: &mut Decode,
        ffn: BoundFfn,
        base: u32,
        rows: u32,
        y: Tensor<bf16, 2>,
    ) -> Result<()> {
        let BoundFfn::Moe {
            shared,
            shared_gate,
            ..
        } = ffn
        else {
            anyhow::bail!("prefill finish received a dense FFN");
        };
        let span = self
            .span
            .as_ref()
            .context("prefill finish has no borrowed span")?;
        decode.prefill.prefill_finish(
            &device.stream,
            MoePrefillFinish {
                x: span.hidden(&span.layout.normalized, base, rows)?,
                ids: span.choices(&span.layout.ids, base, rows)?,
                final_active: span.choices(&span.layout.active, base, rows)?,
                contributions: span.contributions(base, rows)?,
                done: span.choices(&span.layout.done, base, rows)?,
                experts: self.geometry.experts,
                shared_gate: shared[0],
                shared_up: shared[1],
                shared_down: shared[2],
                shared_expert_gate: shared_gate,
                y: TensorMut(y),
            },
            &qwen::scratch(&self.gpu.scratch, &device.stream),
        )?;
        Ok(())
    }
}

fn device_copy(
    stream: &Arc<CudaStream>,
    source: Tensor<bf16, 2>,
    target: Tensor<bf16, 2>,
) -> Result<()> {
    ensure!(
        source.shape == target.shape
            && source.stride == target.stride
            && source.stride == [u64::from(source.shape[1]), 1],
        "prefill dense copy shape/stride differs: source={source:?} target={target:?}"
    );
    let bytes = source.shape[0] as usize * source.shape[1] as usize * 2;
    // SAFETY: both validated views cover the same dense extent and stay owned through this stream.
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            target.ptr,
            source.ptr,
            bytes,
            stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

impl Engine {
    pub(in crate::backend::cuda::engine) fn check_moe_reset(&self) -> Result<()> {
        if let Some(prefill) = &self.moe_prefill {
            ensure!(
                !prefill.failed && prefill.span.is_none(),
                "MoE prefill failed={} borrowed_rows={:?}; reload is required",
                prefill.failed,
                prefill.span.as_ref().map(|span| span.layout.rows)
            );
        }
        Ok(())
    }
    pub(in crate::backend::cuda::engine) fn prefill_moe_chunks(
        &mut self,
        ids: &[u32],
        trace: Option<&mut Trace>,
        keep: usize,
        after_chunk: &mut impl FnMut(&mut Self, &[u32]) -> Result<()>,
    ) -> Result<u32> {
        ensure!(
            self.spec.is_none() || self.hidden_capture.is_some(),
            "MoE draft prefill requires the bound target capture"
        );
        ensure!(
            self.moe_prefill.is_some() && self.moe_slots.is_some() && self.moe.is_some(),
            "MoE prefill binding missing: executor={} slots={} CPU_pool={}",
            self.moe_prefill.is_some(),
            self.moe_slots.is_some(),
            self.moe.is_some()
        );
        let mut prefill = self
            .moe_prefill
            .take()
            .context("MoE layer-major prefill executor is not bound")?;
        let mut slots = self
            .moe_slots
            .take()
            .context("MoE prefill slot arena is not bound")?;
        let mut decode = self
            .moe
            .take()
            .context("MoE CPU worker pool is not bound")?;
        let mut waves = std::mem::take(&mut prefill.waves);
        let result = self.run_moe_prefill(
            &mut prefill,
            &mut slots,
            &mut decode,
            (ids, &mut waves),
            (trace, keep),
            after_chunk,
        );
        prefill.waves = waves;
        if result.is_err() {
            prefill.failed = true;
        }
        self.moe_prefill = Some(prefill);
        self.moe_slots = Some(slots);
        self.moe = Some(decode);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn moe_wave(&mut self, position: usize, rows: usize) -> Result<crate::backend::Step<'static>> {
        self.position = position;
        self.prepare_prefill_rows(rows)?;
        self.state.set_sequence(
            &self.device.stream,
            u32::try_from(position)?,
            u32::try_from(rows)?,
            u32::try_from(self.workspace_rows)?,
        )?;
        self.device
            .stream
            .memcpy_htod(&[i32::try_from(position)?], &mut self.work.position)?;
        Ok(crate::backend::Step {
            rows,
            input: crate::backend::Input::Ids,
            head: Head::None,
            mask: crate::backend::Mask::Causal,
        })
    }

    fn run_moe_prefill(
        &mut self,
        prefill: &mut Prefill,
        slots: &mut Slots,
        decode: &mut Decode,
        input: (&[u32], &mut [Wave]),
        tracing: (Option<&mut Trace>, usize),
        after_chunk: &mut impl FnMut(&mut Self, &[u32]) -> Result<()>,
    ) -> Result<u32> {
        use crate::backend::Operator as O;
        let (ids, waves) = input;
        let (mut trace, keep) = tracing;
        let start = self.position;
        let end = start + ids.len();
        let trace_start = end.saturating_sub(keep);
        let all_layers = trace.as_ref().is_some_and(|trace| trace.all_layers);
        ensure!(
            self.chunk == prefill.mixer_rows && self.workspace_rows == prefill.workspace_rows,
            "prefill mixer/workspace={}/{} differ from bound engine rows={}/{}",
            prefill.mixer_rows,
            prefill.workspace_rows,
            self.chunk,
            self.workspace_rows
        );
        prefill.begin(&self.device, slots, u32::try_from(ids.len())?)?;
        let mut part = 0usize;
        for (span_index, admitted) in ids.chunks(prefill.capacity.span_rows as usize).enumerate() {
            if span_index > 0 {
                prefill.gpu.read_ahead.finish(slots)?;
            }
            let span_start = start + span_index * prefill.capacity.span_rows as usize;
            let wave_count = admitted.len().div_ceil(self.chunk);
            ensure!(
                wave_count <= waves.len(),
                "prefill wave_count={wave_count} exceed setup bound={}",
                waves.len()
            );
            let waves = &mut waves[..wave_count];
            for (base, rows, skip, selected, prefix) in waves.iter_mut() {
                *rows = self.chunk.min(admitted.len() - *base);
                *skip = trace_start.saturating_sub(span_start + *base).min(*rows);
                *selected = trace.is_some() && *skip < *rows;
                prefix.clear();
                if *selected {
                    if all_layers {
                        prefix.push_str("layer");
                    } else {
                        write!(prefix, "prefill.{part}")?;
                    }
                    part += 1;
                }
            }
            for (base, rows, skip, selected, prefix) in waves.iter() {
                let step = self.moe_wave(span_start + base, *rows)?;
                self.device.stream.memcpy_htod(
                    &admitted[*base..base + rows],
                    &mut self.work.ids.slice_mut(..*rows),
                )?;
                let mut target = if *selected {
                    trace.as_deref_mut().map(|trace| Target {
                        trace,
                        prefix,
                        skip: *skip,
                    })
                } else {
                    None
                };
                self.run_operator(O::Embed, 0, step, &mut target)?;
                let span = prefill
                    .span
                    .as_ref()
                    .context("embedded prefill has no borrowed span")?;
                device_copy(
                    &self.device.stream,
                    qwen::matrix(
                        &self.work.x,
                        &self.device.stream,
                        0,
                        *rows,
                        self.model_spec.hidden as usize,
                    )?,
                    span.hidden(&span.layout.residual, *base as u32, *rows as u32)?,
                )?;
            }
            for layer in 0..self.model_spec.layers.len() {
                for (base, rows, skip, selected, prefix) in waves.iter() {
                    let step = self.moe_wave(span_start + base, *rows)?;
                    self.select_qwen_positions()?;
                    let span = prefill
                        .span
                        .as_ref()
                        .context("prefill mixer has no borrowed span")?;
                    device_copy(
                        &self.device.stream,
                        span.hidden(&span.layout.residual, *base as u32, *rows as u32)?,
                        qwen::matrix(
                            &self.work.x,
                            &self.device.stream,
                            0,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?,
                    )?;
                    let mut target = if *selected {
                        trace.as_deref_mut().map(|trace| Target {
                            trace,
                            prefix,
                            skip: *skip,
                        })
                    } else {
                        None
                    };
                    self.qwen_layer_begin(layer, step)?;
                    self.run_operator(O::Norm, layer, step, &mut target)?;
                    self.qwen_mixer(layer, step, &mut target)?;
                    self.run_operator(O::PostNorm, layer, step, &mut target)?;
                    device_copy(
                        &self.device.stream,
                        qwen::matrix(
                            &self.work.x,
                            &self.device.stream,
                            0,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?,
                        span.hidden(&span.layout.residual, *base as u32, *rows as u32)?,
                    )?;
                    device_copy(
                        &self.device.stream,
                        qwen::matrix(
                            &self.work.n,
                            &self.device.stream,
                            0,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?,
                        span.hidden(&span.layout.normalized, *base as u32, *rows as u32)?,
                    )?;
                }
                let ffn = self.moe_ffn_binding(layer)?;
                prefill.ffn(
                    &self.device,
                    slots,
                    decode,
                    layer as u32,
                    admitted.len() as u32,
                    ffn,
                )?;
                for base in (0..admitted.len()).step_by(prefill.capacity.wave_rows as usize) {
                    let rows = (prefill.capacity.wave_rows as usize).min(admitted.len() - base);
                    prefill.finish_wave(
                        &self.device,
                        decode,
                        ffn,
                        base as u32,
                        rows as u32,
                        qwen::matrix(
                            &self.work.out,
                            &self.device.stream,
                            0,
                            rows,
                            self.model_spec.hidden as usize,
                        )?,
                    )?;
                    let span = prefill
                        .span
                        .as_ref()
                        .context("prefill residual join has no borrowed span")?;
                    device_copy(
                        &self.device.stream,
                        span.hidden(&span.layout.residual, base as u32, rows as u32)?,
                        qwen::matrix(
                            &self.work.x,
                            &self.device.stream,
                            0,
                            rows,
                            self.model_spec.hidden as usize,
                        )?,
                    )?;
                    device_copy(
                        &self.device.stream,
                        qwen::matrix(
                            &self.work.out,
                            &self.device.stream,
                            0,
                            rows,
                            self.model_spec.hidden as usize,
                        )?,
                        span.hidden(&span.layout.normalized, base as u32, rows as u32)?,
                    )?;
                    // SAFETY: the two owned dense row windows cover rows*hidden values on this stream.
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&self.ops.residual)
                            .arg(&mut self.work.x)
                            .arg(&self.work.out)
                            .arg(&(i32::try_from(rows * self.model_spec.hidden as usize)?))
                            .launch(super::super::super::super::ops::flat(
                                rows * self.model_spec.hidden as usize,
                            ))?;
                    }
                    if let Some(capture) = &mut self.hidden_capture
                        && let Some(slot) = capture.layer_slots[layer]
                    {
                        copy::columns(
                            &self.device.stream,
                            &self.work.x,
                            &mut capture.values.slice_mut(base * capture.width..),
                            copy::Rect {
                                rows,
                                width: self.model_spec.hidden as usize,
                                src_pitch: self.model_spec.hidden as usize,
                                src_col: 0,
                                dst_pitch: capture.width,
                                dst_col: slot as usize * self.model_spec.hidden as usize,
                            },
                        )?;
                    }
                    device_copy(
                        &self.device.stream,
                        qwen::matrix(
                            &self.work.x,
                            &self.device.stream,
                            0,
                            rows,
                            self.model_spec.hidden as usize,
                        )?,
                        span.hidden(&span.layout.residual, base as u32, rows as u32)?,
                    )?;
                }
                for (base, rows, skip, selected, prefix) in waves.iter() {
                    if !selected {
                        continue;
                    }
                    let span = prefill
                        .span
                        .as_ref()
                        .context("prefill trace has no borrowed span")?;
                    device_copy(
                        &self.device.stream,
                        span.hidden(&span.layout.residual, *base as u32, *rows as u32)?,
                        qwen::matrix(
                            &self.work.x,
                            &self.device.stream,
                            0,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?,
                    )?;
                    device_copy(
                        &self.device.stream,
                        span.hidden(&span.layout.normalized, *base as u32, *rows as u32)?,
                        qwen::matrix(
                            &self.work.out,
                            &self.device.stream,
                            0,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?,
                    )?;
                    let mut target = trace.as_deref_mut().map(|trace| Target {
                        trace,
                        prefix,
                        skip: *skip,
                    });
                    if layer == self.trace_layer {
                        trace_value(
                            &mut target,
                            &self.device.stream,
                            "ffn",
                            &self.work.out,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?;
                    }
                    let layers = self.model_spec.layers.len();
                    if !all_layers
                        || layer == 0
                        || layer == self.trace_layer
                        || layer == layers / 2
                        || layer + 1 == layers
                        || self
                            .hidden_capture
                            .as_ref()
                            .is_some_and(|capture| capture.layer_slots[layer].is_some())
                    {
                        let name = if all_layers {
                            &prefill.layer_names[layer]["layer.".len()..]
                        } else {
                            &prefill.layer_names[layer]
                        };
                        trace_value(
                            &mut target,
                            &self.device.stream,
                            name,
                            &self.work.x,
                            *rows,
                            self.model_spec.hidden as usize,
                        )?;
                    }
                }
            }
            for (base, rows, skip, selected, prefix) in waves.iter() {
                let mut step = self.moe_wave(span_start + base, *rows)?;
                self.device.stream.memcpy_htod(
                    &admitted[*base..base + rows],
                    &mut self.work.ids.slice_mut(..*rows),
                )?;
                let span = prefill
                    .span
                    .as_ref()
                    .context("prefill head has no borrowed span")?;
                device_copy(
                    &self.device.stream,
                    span.hidden(&span.layout.residual, *base as u32, *rows as u32)?,
                    qwen::matrix(
                        &self.work.x,
                        &self.device.stream,
                        0,
                        *rows,
                        self.model_spec.hidden as usize,
                    )?,
                )?;
                if span_start + base + rows == end {
                    step.head = Head::Last;
                }
                let mut target = if *selected {
                    trace.as_deref_mut().map(|trace| Target {
                        trace,
                        prefix,
                        skip: *skip,
                    })
                } else {
                    None
                };
                self.run_operator(O::Head, 0, step, &mut target)?;
                self.resident_ids_len = *rows;
                self.last_input_kind = Some(PrefixLastInputKind::PrefillChunk);
                after_chunk(self, &admitted[*base..base + rows])?;
                self.committed_prefix_ids
                    .extend_from_slice(&admitted[*base..base + rows]);
            }
            if self.spec.is_some() {
                self.inject_capture(admitted.len(), span_start, None)?;
            }
        }
        prefill.end(&self.device, slots)?;
        self.position = end;
        self.device
            .stream
            .memcpy_htod(&[i32::try_from(end)?], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[i32::try_from(end + 1)?], &mut self.work.length)?;
        if end < self.capacity {
            self.state.set_sequence(
                &self.device.stream,
                u32::try_from(end)?,
                1,
                self.workspace_rows as u32,
            )?;
        }
        let token = self.token()?;
        self.logits_cache.mark_valid();
        self.pending_token = Some(token);
        Ok(token)
    }
}

impl Cpu {
    fn prepare(
        &mut self,
        gpu: &Gpu,
        span: &Span,
        slots: &Slots,
        layer: u32,
        plan: &SpanPlan,
    ) -> Result<()> {
        self.choices.clear();
        self.rows.clear();
        self.row_map.fill(u32::MAX);
        self.experts.clear();
        self.assignments.clear();
        self.disk.clear();
        let geometry = span.geometry;
        let choices = plan.span_rows() as usize * geometry.top_k as usize;
        let ids = &gpu.host_ids.as_slice()?[..choices];
        let weights = &gpu.host_weights.as_slice()?[..choices];
        for &expert in plan.cpu() {
            let compact = self.experts.len() as u32;
            let tier = slots.policy.ledger().tier(layer, expert)?;
            if tier == Tier::Host {
                let block = slots.policy.host_block(layer, expert)?;
                let range = slots
                    .transfers
                    .host
                    .as_ref()
                    .context("prefill host tier has no arena")?
                    .block_range(block)?;
                self.experts.push(ExpertWeights::from_slot(
                    Shape {
                        hidden: geometry.hidden as usize,
                        intermediate: geometry.intermediate as usize,
                    },
                    range,
                    slots.catalogue.layout(),
                )?);
            }
            ensure!(
                tier != Tier::Gpu,
                "prefill CPU expert={expert} is GPU-resident"
            );
            for (choice, _) in ids
                .iter()
                .enumerate()
                .filter(|(_, id)| **id == expert as i32)
            {
                let index = self.choices.len();
                ensure!(
                    index < self.done.len(),
                    "prefill CPU assignments exceed bound={}",
                    self.done.len()
                );
                self.choices.push(choice as u32);
                let row = choice / geometry.top_k as usize;
                if self.row_map[row] == u32::MAX {
                    let compact_row = self.rows.len();
                    ensure!(
                        (compact_row + 1) * geometry.hidden as usize <= self.x.len(),
                        "prefill CPU rows exceed input capacity={}",
                        self.x.len()
                    );
                    self.row_map[row] = compact_row as u32;
                    self.rows.push(row as u32);
                    let first = compact_row * geometry.hidden as usize;
                    let source = span.base
                        + span.layout.normalized.start as u64
                        + (row * geometry.hidden as usize * 2) as u64;
                    // SAFETY: the routed row and compact pinned destination both hold hidden BF16 values.
                    unsafe {
                        result::memcpy_dtoh_async(
                            &mut self.x.as_mut_slice()?[first..first + geometry.hidden as usize],
                            source,
                            gpu.mailbox.cu_stream(),
                        )?;
                    }
                }
                if tier == Tier::Host {
                    self.assignments.push(Assignment {
                        row: self.row_map[row],
                        choice: index as u32,
                        expert: compact,
                        weight: weights[choice],
                    });
                } else {
                    self.disk.push(crate::experts::runtime::Route {
                        row: self.row_map[row],
                        choice: index as u32,
                        expert,
                        weight: weights[choice],
                    });
                }
            }
        }
        gpu.mailbox.synchronize()?;
        Ok(())
    }

    fn upload(
        &self,
        mailbox: &Arc<CudaStream>,
        compute: &Arc<CudaStream>,
        span: &Span,
        events: &mut Events,
    ) -> Result<()> {
        if self.choices.is_empty() {
            return Ok(());
        }
        let ready = events.record(compute)?;
        mailbox.wait(&ready.event)?;
        let hidden = span.geometry.hidden as usize;
        let values = self.out.as_slice()?;
        let done = self.done.as_slice()?;
        for (index, &choice) in self.choices.iter().enumerate() {
            // SAFETY: each compact CPU result owns exactly one routed contribution and completion word.
            unsafe {
                result::memcpy_htod_async(
                    span.base
                        + span.layout.contributions.start as u64
                        + u64::from(choice) * hidden as u64 * 2,
                    &values[index * hidden..(index + 1) * hidden],
                    mailbox.cu_stream(),
                )?;
                result::memcpy_htod_async(
                    span.base + span.layout.done.start as u64 + u64::from(choice) * 4,
                    &done[index..index + 1],
                    mailbox.cu_stream(),
                )?;
            }
        }
        let uploaded = events.record(mailbox)?;
        compute.wait(&uploaded.event)?;
        Ok(())
    }
}

impl Gpu {
    fn prepare_tables(
        &mut self,
        compute: &Arc<CudaStream>,
        slots: &mut Slots,
        layer: u32,
        plan: &SpanPlan,
    ) -> Result<()> {
        let geometry = slots.catalogue.geometry();
        self.tables.clear()?;
        for &expert in plan.resident() {
            let position = slots.policy.ledger().slot(layer, expert)?;
            let physical = &mut slots.physical[position as usize];
            if !physical.waited {
                let id = slots
                    .policy
                    .first_read(position)?
                    .context("prefill resident has no copy event")?;
                ensure!(
                    physical.copy.as_ref().is_some_and(|copy| copy.id == id),
                    "prefill resident slot={position} copy ownership differs"
                );
                slots.events.wait(compute, id)?;
                physical.waited = true;
            }
            self.tables.set(
                geometry.experts,
                0,
                expert,
                (
                    physical.pointer,
                    physical.pointer + slots.catalogue.layout().down_offset(),
                ),
                slots.policy.generation(layer, expert)?,
            )?;
        }
        for (cohort, experts) in plan.cohorts().enumerate() {
            self.epoch = self
                .epoch
                .checked_add(1)
                .context("prefill staging epoch overflow")?;
            for (position, &expert) in experts.iter().enumerate() {
                self.tables.set(
                    geometry.experts,
                    cohort + 1,
                    expert,
                    slots.storage.staging(position as u32)?,
                    self.epoch,
                )?;
            }
        }
        self.tables
            .upload(compute, 1 + plan.cohort_count() as usize, geometry.experts)
    }

    fn contribute(
        &mut self,
        device: &Device,
        slots: &mut Slots,
        span: &Span,
        ops: &mut PrefillOps,
        run: (u32, &SpanPlan, BoundFfn, &mut StagingLeases),
        next: Option<(u32, &SpanPlan)>,
    ) -> Result<()> {
        let (layer, plan, ffn, leases) = run;
        let stream = &device.stream;
        if plan.staged().is_empty()
            && let Some((layer, next)) = next
        {
            self.read_ahead.predict(slots, layer, next)?;
        }
        // SAFETY: done is reset once for the whole layer before any cohort writes it.
        unsafe {
            result::memset_d8_async(
                span.base + span.layout.done.start as u64,
                0,
                plan.span_rows() as usize * span.geometry.top_k as usize * 4,
                stream.cu_stream(),
            )?;
        }
        if !plan.resident().is_empty() {
            let ticket = self.cohort(device, slots, span, ops, plan, (layer, 0, ffn, None))?;
            self.resident_slots.clear();
            for &expert in plan.resident() {
                self.resident_slots
                    .push(slots.policy.ledger().slot(layer, expert)?);
            }
            slots
                .policy
                .record_readers(&self.resident_slots, ticket.id)?;
            for &position in &self.resident_slots {
                slots.physical[position as usize].reader = Some(ticket.clone());
            }
        }
        for (cohort, experts) in plan.cohorts().enumerate() {
            leases.reset(u32::try_from(plan.cohort_waves(cohort + 1)?.len())?)?;
            for (position, &expert) in experts.iter().enumerate() {
                if let Some(reader) = self.staging_readers[position].take() {
                    slots.transfers.h2d.wait(&reader.event)?;
                }
                self.copy_expert(slots, layer, expert, position as u32)?;
            }
            let copied = slots.events.record(&slots.transfers.h2d)?;
            leases.load(experts, copied.id)?;
            self.staging_copy = Some(copied);
            self.read_ahead.prime(slots, layer, plan)?;
            if cohort + 1 == plan.cohort_count() as usize
                && let Some((layer, next)) = next
            {
                self.read_ahead.predict(slots, layer, next)?;
            }
            self.cohort(
                device,
                slots,
                span,
                ops,
                plan,
                (layer, cohort + 1, ffn, Some(leases)),
            )?;
            for &id in leases.release()? {
                ensure!(
                    self.staging_readers
                        .iter()
                        .flatten()
                        .any(|reader| reader.id == id),
                    "prefill released reader={id} has no owner"
                );
            }
        }
        Ok(())
    }

    fn copy_expert(
        &mut self,
        slots: &mut Slots,
        layer: u32,
        expert: u32,
        position: u32,
    ) -> Result<()> {
        let pointer = slots.storage.staging(position)?.0;
        let bytes = usize::try_from(slots.catalogue.slot_bytes())?;
        let (source, host) = match slots.policy.ledger().tier(layer, expert)? {
            Tier::Host => {
                let block = slots.policy.host_block(layer, expert)?;
                let (source, stage) = slots.transfers.source_host(block)?;
                (source, Some((block, stage)))
            }
            Tier::Disk => {
                let ring = self.read_ahead.demand(slots, layer, expert)?;
                let disk = slots
                    .disk
                    .as_mut()
                    .context("prefill disk DMA has no reader owner")?;
                let values = disk.disk.values(ring)?;
                let source = values.as_ptr() as u64;
                // SAFETY: the registered disk ring remains leased until its H2D event.
                unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        pointer,
                        source as *const std::ffi::c_void,
                        bytes,
                        slots.transfers.h2d.cu_stream(),
                    )
                    .result()?;
                }
                let copied = slots.events.record(&slots.transfers.h2d)?;
                values.copied(copied.id)?;
                disk.copied[ring as usize] = Some(copied);
                return Ok(());
            }
            Tier::Gpu => anyhow::bail!(
                "prefill staged expert layer={layer} expert={expert} is already resident"
            ),
        };
        // SAFETY: the native expert's claimed or staged source remains live through the copy event.
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                pointer,
                source as *const std::ffi::c_void,
                bytes,
                slots.transfers.h2d.cu_stream(),
            )
            .result()?;
        }
        if let Some((block, stage)) = host {
            let copied = slots.events.record(&slots.transfers.h2d)?;
            slots.transfers.copied_host(block, stage, &copied);
        }
        Ok(())
    }

    fn cohort(
        &mut self,
        device: &Device,
        slots: &mut Slots,
        span: &Span,
        ops: &mut PrefillOps,
        plan: &SpanPlan,
        cohort: (u32, usize, BoundFfn, Option<&mut StagingLeases>),
    ) -> Result<Ticket> {
        let (layer, cohort, ffn, mut leases) = cohort;
        let staged_slots = if leases.is_some() {
            ensure!(cohort > 0, "prefill resident cohort has staging leases");
            plan.cohorts()
                .nth(cohort - 1)
                .context("prefill staging cohort exceeds the layer plan")?
                .len()
        } else {
            0
        };
        let BoundFfn::Moe {
            shared,
            shared_gate,
            ..
        } = ffn
        else {
            anyhow::bail!("prefill contribute received a dense FFN");
        };
        let stream = &device.stream;
        let geometry = span.geometry;
        let tables = self.tables.bindings(stream, cohort, geometry.experts)?;
        let ids = span.choices::<i32>(&span.layout.ids, 0, plan.span_rows())?;
        let active = span.choices::<u32>(&span.layout.active, 0, plan.span_rows())?;
        let generation = span.choices::<u64>(&span.layout.generation, 0, plan.span_rows())?;
        let assignments = plan
            .span_rows()
            .checked_mul(geometry.top_k)
            .context("prefill mask assignment overflow")?;
        ensure!(
            assignments <= i32::MAX as u32,
            "prefill mask assignments={assignments} exceed i32"
        );
        // SAFETY: validated dense assignment and expert tables cover every mask element.
        unsafe {
            stream
                .launch_builder(&self.mask)
                .arg(&ids.ptr)
                .arg(&tables.valid.ptr)
                .arg(&tables.generation.ptr)
                .arg(&active.ptr)
                .arg(&generation.ptr)
                .arg(&(assignments as i32))
                .arg(&(geometry.experts as i32))
                .launch(super::super::super::super::ops::grid(
                    assignments.div_ceil(256) as usize,
                    256,
                ))?;
        }
        let wait_copy = if leases.is_some() {
            Some(
                self.staging_copy
                    .as_ref()
                    .context("prefill staging cohort has no copy event")?
                    .event
                    .as_ref(),
            )
        } else {
            None
        };
        let range = cohort * geometry.experts as usize..(cohort + 1) * geometry.experts as usize;
        let mut last = None;
        for (wave_index, &(base, rows, active)) in plan.cohort_waves(cohort)?.iter().enumerate() {
            let ticket = slots.events.claim()?;
            ops.prefill_contribute(
                stream,
                MoePrefillContribute {
                    cpu_arithmetic: &self.cpu_arithmetic,
                    fp8_slot: slots.catalogue.layout().fp8_slot(),
                    wave: Some((cohort, base, active)),
                    wait_copy: (wave_index == 0).then_some(wait_copy).flatten(),
                    reader_event: Some(&ticket.event),
                    x: span.hidden(&span.layout.normalized, base, rows)?,
                    ids: span.choices(&span.layout.ids, base, rows)?,
                    weights: span.choices(&span.layout.weights, base, rows)?,
                    active: span.choices(&span.layout.active, base, rows)?,
                    assignment_gen: span.choices(&span.layout.generation, base, rows)?,
                    slots_gate_up: tables.gate_up,
                    slots_down: tables.down,
                    slot_valid: tables.valid,
                    slot_generation: tables.generation,
                    slots_gate_up_host: &self.tables.host_gate_up.as_slice()?[range.clone()],
                    slots_down_host: &self.tables.host_down.as_slice()?[range.clone()],
                    shared_gate: shared[0],
                    shared_up: shared[1],
                    shared_down: shared[2],
                    shared_expert_gate: shared_gate,
                    contributions: span.contributions(base, rows)?,
                    done: TensorMut(span.choices(&span.layout.done, base, rows)?),
                },
                &qwen::scratch(&self.scratch, stream),
            )?;
            if let Some(leases) = leases.as_mut() {
                for position in 0..staged_slots {
                    leases.record_wave(position as u32, ticket.id)?;
                    self.staging_readers[position] = Some(ticket.clone());
                }
            }
            last = Some(ticket);
            self.read_ahead.prime(slots, layer, plan)?;
        }
        last.context("prefill cohort has no row waves")
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::tests::{
        graph_state, hybrid_spec, tier_capacity, tier_fixture, tier_service,
    };
    use super::*;
    use crate::backend::cuda::moe::reference::MoeFixture;
    use crate::cpu::expert_ffn::{
        Contribution, Fp8Contribution, Isa, Scratch, contribution, fp8_contribution,
    };

    fn reserves(geometry: ExpertGeometry) -> Reserves {
        Reserves {
            device_fixed_bytes: 0,
            host_fixed_bytes: 0,
            residual_row_bytes: u64::from(geometry.hidden) * 2,
            normalized_row_bytes: u64::from(geometry.hidden) * 2,
            contribution_row_bytes: u64::from(geometry.top_k) * u64::from(geometry.hidden) * 2,
            done_row_bytes: u64::from(geometry.top_k) * 4,
            route_row_bytes: u64::from(geometry.top_k) * 18,
            prefill_staging_slots: 2,
            disk_ring_slots: 2,
        }
    }

    #[test]
    #[ignore = "requires CUDA, FF_MODEL and FF_MINICPM_REFERENCE_DIR"]
    fn published_expert_contributions_match_across_tiers_gpu() -> Result<()> {
        use crate::backend::cuda::moe::testing;
        use crate::cpu::expert_ffn::Job;
        let directory = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let fixtures = PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("reference directory missing")?,
        );
        let model = crate::model_spec::ModelSpec::read(&directory)?;
        let geometry = ExpertGeometry::from_model(&model)?;
        ensure!(
            model.expert_format == Some(crate::model_spec::ExpertFormat::Bf16)
                && geometry.hidden == 2048
                && geometry.intermediate == 512
                && geometry.experts == 256,
            "published BF16 expert identity geometry required: {geometry:?}"
        );
        let checkpoint = crate::checkpoint::MappedCheckpoint::open(&directory)?;
        let catalogue = Arc::new(ExpertCatalogue::from_manifest(
            &checkpoint.manifest(&model, crate::checkpoint::LoadPolicy::TextOnly)?,
            geometry,
            crate::model_spec::ExpertFormat::Bf16,
        )?);
        drop(checkpoint);
        let expert = 160u32;
        let reader = DiskReader::open(&catalogue, &directory, 1)?;
        let slot = reader.read_expert(0, expert, &catalogue)?;
        let native = reader.values(slot)?.values().to_vec();
        let hidden = geometry.hidden as usize;
        let intermediate = geometry.intermediate as usize;
        let cut = usize::try_from(catalogue.layout().down_offset() / 2)?;
        let host = HostBlocks::new(1, native.len())?;
        host.claim(0)?.slice()?.copy_from_slice(&native);
        let weights = [ExpertWeights {
            gate_up: 0..cut,
            gate_up_stride: hidden,
            down: cut..native.len(),
            down_stride: intermediate,
        }];
        let bytes = std::fs::read(fixtures.join("text_32_layer00_moe.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let read = |tensors: &safetensors::SafeTensors<'_>,
                    name: &str,
                    shape: &[usize]|
         -> Result<Vec<bf16>> {
            let tensor = tensors.tensor(name)?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::BF16 && tensor.shape() == shape,
                "expert identity fixture {name} dtype={:?} shape={:?}",
                tensor.dtype(),
                tensor.shape()
            );
            Ok(tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| bf16::from_bits(u16::from_le_bytes(*value)))
                .collect())
        };
        let input = read(&tensors, "layer_00_moe_in_norm", &[32, hidden])?;
        let routing = read(
            &tensors,
            "layer_00_router_topk_weights",
            &[32, geometry.top_k as usize],
        )?;
        let reference_bytes = std::fs::read(fixtures.join("expert-tier-reference.safetensors"))?;
        let reference = safetensors::SafeTensors::deserialize(&reference_bytes)?;
        ensure!(
            read(&reference, "input", &[32, hidden])? == input
                && read(&reference, "routing", &[32, geometry.top_k as usize])? == routing,
            "expert tier reference input differs from T0"
        );
        let reference_bound = |a: &[bf16], b: &[bf16]| {
            let delta = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0.0f32, f32::max);
            let twice = delta + delta;
            let rounded = bf16::from_f32(twice);
            bf16::from_bits(rounded.to_bits() + u16::from(rounded.to_f32() < twice)).to_f32()
        };
        let device = Device::new(0)?;
        let stream = &device.stream;
        let domain: Vec<_> = (0..=u16::MAX)
            .flat_map(|bits| [bf16::from_bits(bits), bf16::ONE])
            .collect();
        let domain_device = stream.clone_htod(&domain)?;
        let silu_device = stream.alloc_zeros::<bf16>(usize::from(u16::MAX) + 1)?;
        let silu =
            crate::backend::cuda::cubin::module(&device.ctx, "act")?.load_function("silu_mul")?;
        // SAFETY: the dense [65536,2] input and [65536,1] output cover the BF16 domain.
        unsafe {
            stream
                .launch_builder(&silu)
                .arg(&domain_device.device_ptr(stream).0)
                .arg(&silu_device.device_ptr(stream).0)
                .arg(&65536i32)
                .arg(&1i32)
                .launch(crate::backend::cuda::ops::grid(256, 256))?;
        }
        let silu_map = stream.clone_dtoh(&silu_device)?;
        let silu_mismatches = silu_map
            .iter()
            .enumerate()
            .filter(|(bits, actual)| {
                let value = bf16::from_bits(*bits as u16).to_f32();
                value.is_finite()
                    && actual.to_bits()
                        != bf16::from_f32(value * (1.0 / (1.0 + (-value).exp()))).to_bits()
            })
            .count();
        ensure!(
            silu_mismatches == 0,
            "BF16 SiLU CPU/GPU map mismatches={silu_mismatches}"
        );
        println!("PASS expert_tier_silu_map finite_inputs=65280 bit_mismatches=0");
        let native_device = stream.clone_htod(&native)?;
        let gate_up_pointer = native_device.device_ptr(stream).0;
        let down_pointer = gate_up_pointer + (cut * 2) as u64;
        let mut host_gate_up = vec![0u64; geometry.experts as usize];
        let mut host_down = host_gate_up.clone();
        let mut valid = vec![0u32; geometry.experts as usize];
        host_gate_up[expert as usize] = gate_up_pointer;
        host_down[expert as usize] = down_pointer;
        valid[expert as usize] = 1;
        let tables = testing::SlotTable {
            gate_up: stream.clone_htod(&host_gate_up)?,
            down: stream.clone_htod(&host_down)?,
            valid: stream.clone_htod(&valid)?,
            generation: stream
                .clone_htod(&valid.iter().copied().map(u64::from).collect::<Vec<_>>())?,
        };
        let pool = ExpertPool::new(2, Shape::new(hidden, intermediate)?, 32, 32, None)?;
        let mut disk = DiskExperts::new(catalogue, &directory, 1, 32)?;
        let ops = PrefillOps::new(&device.ctx, stream, crate::model_spec::ExpertFormat::Bf16)?;
        let mut defects = 0;
        for (rows, weighted) in [
            (1usize, false),
            (32, false),
            (32, true),
            (LANE16_ROWS as usize, true),
            (LANE16_ROWS as usize + 1, true),
        ] {
            let x = &input[..rows * hidden];
            let routing: Vec<_> = (0..rows)
                .map(|row| {
                    if weighted {
                        routing[row * geometry.top_k as usize]
                    } else {
                        bf16::ONE
                    }
                })
                .collect();
            ensure!(
                routing
                    .iter()
                    .all(|value| value.is_finite() && *value > bf16::ZERO)
                    && (!weighted || routing.iter().all(|value| *value != bf16::ONE)),
                "expert identity routing weights invalid"
            );
            let assignments: Vec<_> = routing
                .iter()
                .enumerate()
                .map(|(row, &weight)| Assignment {
                    row: row as u32,
                    choice: row as u32,
                    expert: 0,
                    weight,
                })
                .collect();
            let routes: Vec<_> = assignments
                .iter()
                .map(|item| crate::experts::runtime::Route {
                    row: item.row,
                    choice: item.choice,
                    expert,
                    weight: item.weight,
                })
                .collect();
            let x_device = stream.clone_htod(x)?;
            let ids = stream.clone_htod(&vec![expert as i32; rows])?;
            let routing_device = stream.clone_htod(&routing)?;
            let active = stream.clone_htod(&vec![1u32; rows])?;
            let generation = stream.clone_htod(&vec![1u64; rows])?;
            let layout = crate::backend::cuda::moe::prefill::prefill_layout(
                rows,
                rows,
                1,
                geometry.experts as usize,
                hidden,
                intermediate,
                None,
            )?;
            let scratch = stream.alloc_zeros::<u8>(usize::try_from(prefill_scratch_bytes(
                rows,
                rows,
                1,
                geometry.experts as usize,
                hidden,
                intermediate,
                None,
            )?)?)?;
            let boundary = rows == LANE16_ROWS as usize || rows == LANE16_ROWS as usize + 1;
            let mut arithmetic = vec![true; geometry.experts as usize];
            if boundary {
                let mut counts = vec![0; geometry.experts as usize];
                counts[expert as usize] = rows as u32;
                arithmetic_mask(&counts, &mut arithmetic)?;
                ensure!(
                    arithmetic[expert as usize] == (rows <= LANE16_ROWS as usize),
                    "expert arithmetic boundary rows={rows}"
                );
            }
            let run_gpu = |tables: &testing::SlotTable,
                           host_gate_up: &[u64],
                           host_down: &[u64]|
             -> Result<Vec<bf16>> {
                let contributions = stream.alloc_zeros::<bf16>(rows * hidden)?;
                let done = stream.alloc_zeros::<u32>(rows)?;
                {
                    let _range =
                        crate::backend::cuda::profile::Range::new(true, c"expert_tier_grouped");
                    ops.prefill_contribute(
                        stream,
                        MoePrefillContribute {
                            cpu_arithmetic: &arithmetic,
                            fp8_slot: None,
                            wave: None,
                            wait_copy: None,
                            reader_event: None,
                            x: qwen::matrix(&x_device, stream, 0, rows, hidden)?,
                            ids: testing::dense(&ids, stream, rows as u32, 1)?,
                            weights: testing::dense(&routing_device, stream, rows as u32, 1)?,
                            active: testing::dense(&active, stream, rows as u32, 1)?,
                            assignment_gen: testing::dense(&generation, stream, rows as u32, 1)?,
                            slots_gate_up: testing::view(
                                &tables.gate_up,
                                stream,
                                [geometry.experts],
                                [1],
                            )?,
                            slots_down: testing::view(
                                &tables.down,
                                stream,
                                [geometry.experts],
                                [1],
                            )?,
                            slot_valid: testing::view(
                                &tables.valid,
                                stream,
                                [geometry.experts],
                                [1],
                            )?,
                            slot_generation: testing::view(
                                &tables.generation,
                                stream,
                                [geometry.experts],
                                [1],
                            )?,
                            slots_gate_up_host: host_gate_up,
                            slots_down_host: host_down,
                            shared_gate: qwen::matrix(
                                &native_device,
                                stream,
                                0,
                                intermediate,
                                hidden,
                            )?,
                            shared_up: qwen::matrix(
                                &native_device,
                                stream,
                                intermediate * hidden,
                                intermediate,
                                hidden,
                            )?,
                            shared_down: qwen::matrix(
                                &native_device,
                                stream,
                                cut,
                                hidden,
                                intermediate,
                            )?,
                            shared_expert_gate: testing::view(
                                &x_device,
                                stream,
                                [geometry.hidden],
                                [1],
                            )?,
                            contributions: testing::view(
                                &contributions,
                                stream,
                                [rows as u32, 1, geometry.hidden],
                                [geometry.hidden as u64, geometry.hidden as u64, 1],
                            )?,
                            done: testing::mut_view(&done, stream, [rows as u32, 1], [1, 1])?,
                        },
                        &qwen::scratch(&scratch, stream),
                    )?;
                    stream.synchronize()?;
                }
                ensure!(
                    stream.clone_dtoh(&done)?.iter().all(|&value| value == 1),
                    "grouped expert did not complete every assignment"
                );
                Ok(stream.clone_dtoh(&contributions)?)
            };
            let gpu = run_gpu(&tables, &host_gate_up, &host_down)?;
            let mut cpu = vec![bf16::ZERO; rows * hidden];
            let pending = pool.submit(Job {
                arena: host.clone(),
                hidden: x,
                rows,
                experts: &weights,
                assignments: &assignments,
                out: &mut cpu,
            })?;
            pool.finish(pending)?;
            let mut from_disk = vec![bf16::ZERO; rows * hidden];
            disk.start(0, x, rows, &routes)?;
            disk.finish(&mut from_disk)?;
            if boundary {
                let staged = stream.clone_htod(&native)?;
                let mut staged_gate_up = host_gate_up.clone();
                let mut staged_down = host_down.clone();
                staged_gate_up[expert as usize] = staged.device_ptr(stream).0;
                staged_down[expert as usize] = staged.device_ptr(stream).0 + (cut * 2) as u64;
                let staged_tables = testing::SlotTable {
                    gate_up: stream.clone_htod(&staged_gate_up)?,
                    down: stream.clone_htod(&staged_down)?,
                    valid: stream.clone_htod(&valid)?,
                    generation: stream
                        .clone_htod(&valid.iter().copied().map(u64::from).collect::<Vec<_>>())?,
                };
                let streamed = run_gpu(&staged_tables, &staged_gate_up, &staged_down)?;
                let mut comparisons = vec![
                    ("gpu_resident_streamed", &gpu, &streamed),
                    ("cpu_pool_disk", &cpu, &from_disk),
                ];
                if arithmetic[expert as usize] {
                    comparisons
                        .extend([("gpu_cpu_pool", &gpu, &cpu), ("gpu_disk", &gpu, &from_disk)]);
                }
                for (name, a, b) in comparisons {
                    ensure!(
                        a.iter().chain(b).all(|value| value.is_finite()),
                        "nonfinite expert boundary contribution {name}"
                    );
                    let mismatches = a
                        .iter()
                        .zip(b)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    defects += usize::from(mismatches != 0);
                    println!(
                        "{} expert_arithmetic_boundary rows={rows} lane16={} comparison={name} weighted=true bit_mismatches={mismatches} cpu_reference_only={}",
                        if mismatches == 0 { "PASS" } else { "DEFECT" },
                        arithmetic[expert as usize],
                        !arithmetic[expert as usize]
                    );
                }
                continue;
            }
            if !weighted {
                let packed = stream.clone_dtoh(&scratch)?;
                let mut cpu_stages: [Vec<bf16>; 4] = std::array::from_fn(|_| Vec::new());
                let mut gpu_stages: [Vec<bf16>; 4] = std::array::from_fn(|_| Vec::new());
                let shape = Shape::new(hidden, intermediate)?;
                let mut cpu_scratch = Scratch::new(&shape);
                let mut cpu_row = vec![bf16::ZERO; hidden];
                let mut seen = vec![false; rows];
                let gpu_values = |offset: u64, count: usize| -> Result<Vec<bf16>> {
                    let offset = usize::try_from(offset)?;
                    Ok(packed
                        .get(offset..offset + count * 2)
                        .context("expert GPU stage exceeds scratch")?
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|value| bf16::from_bits(u16::from_le_bytes(*value)))
                        .collect())
                };
                for row in 0..rows {
                    contribution(
                        Isa::detect(),
                        Contribution {
                            shape: &shape,
                            arena: &native,
                            weights: &weights[0],
                            hidden: &x[row * hidden..(row + 1) * hidden],
                            routing: bf16::ONE,
                        },
                        &mut cpu_row,
                        &mut cpu_scratch,
                    )?;
                    let pool_mismatches = cpu_row
                        .iter()
                        .zip(&cpu[row * hidden..(row + 1) * hidden])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    defects += usize::from(pool_mismatches != 0);
                    println!(
                        "expert_tier_cpu_stage_binding rows={rows} row={row} pool_bit_mismatches={pool_mismatches}"
                    );
                    for (target, values) in cpu_stages.iter_mut().zip(cpu_scratch.stages()) {
                        target.extend_from_slice(values);
                    }
                    let position_at = usize::try_from(layout.pos_of)? + row * 4;
                    let position = usize::try_from(u32::from_le_bytes(
                        packed
                            .get(position_at..position_at + 4)
                            .context("expert position exceeds scratch")?
                            .try_into()?,
                    ))?;
                    ensure!(
                        position < rows && !seen[position],
                        "expert row permutation invalid: row={row} position={position}"
                    );
                    seen[position] = true;
                    for (target, (offset, width)) in gpu_stages.iter_mut().zip([
                        (
                            layout.h1 + (position * 2 * intermediate * 2) as u64,
                            intermediate,
                        ),
                        (
                            layout.h1 + ((position * 2 + 1) * intermediate * 2) as u64,
                            intermediate,
                        ),
                        (
                            layout.act + (position * intermediate * 2) as u64,
                            intermediate,
                        ),
                        (layout.y_perm + (position * hidden * 2) as u64, hidden),
                    ]) {
                        target.extend(gpu_values(offset, width)?);
                    }
                }
                for (stage, (name, width)) in [
                    ("gate", intermediate),
                    ("up", intermediate),
                    ("activated", intermediate),
                    ("down", hidden),
                ]
                .into_iter()
                .enumerate()
                {
                    let batch = read(&reference, &format!("batch{rows}_{name}"), &[rows, width])?;
                    let per_row = read(&reference, &format!("perrow32_{name}"), &[32, width])?;
                    let fp64 = read(&reference, &format!("fp64_{name}"), &[32, width])?;
                    let bound = reference_bound(&batch, &per_row[..rows * width]);
                    let sources = [
                        ("gpu_grouped", gpu_stages[stage].as_slice()),
                        ("cpu", cpu_stages[stage].as_slice()),
                        ("official_batch", batch.as_slice()),
                        ("official_perrow", &per_row[..rows * width]),
                        ("fp64", &fp64[..rows * width]),
                    ];
                    for a in 0..sources.len() {
                        for b in a + 1..sources.len() {
                            let (left, right) = (sources[a], sources[b]);
                            ensure!(
                                left.1.iter().chain(right.1).all(|value| value.is_finite()),
                                "nonfinite expert stage={name}"
                            );
                            let mismatches = left
                                .1
                                .iter()
                                .zip(right.1)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            let maximum = left
                                .1
                                .iter()
                                .zip(right.1)
                                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                                .fold(0.0f32, f32::max);
                            let first = left
                                .1
                                .iter()
                                .zip(right.1)
                                .enumerate()
                                .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                                .map(|(index, (a, b))| {
                                    (index, a.to_bits(), b.to_bits(), a.to_f32(), b.to_f32())
                                });
                            println!(
                                "expert_tier_stage rows={rows} stage={name} left={} right={} stored=[{rows},{width}] bit_mismatches={mismatches} max_abs={maximum} first={first:?}",
                                left.0, right.0
                            );
                            if a < 2 && (b == 2 || b == 3) {
                                let pass = maximum <= bound && (bound != 0.0 || mismatches == 0);
                                defects += usize::from(!pass);
                                println!(
                                    "{} expert_tier_stage_reference rows={rows} stage={name} tier={} reference={} max_abs={maximum} bound={bound} bit_mismatches={mismatches}",
                                    if pass { "PASS" } else { "DEFECT" },
                                    left.0,
                                    right.0
                                );
                            }
                        }
                    }
                }
            }
            let weighted_reference = |name: &str, shape: &[usize]| -> Result<Vec<bf16>> {
                Ok(read(&reference, name, shape)?[..rows * hidden]
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        bf16::from_f32(value.to_f32() * routing[index / hidden].to_f32())
                    })
                    .collect())
            };
            let batch = weighted_reference(&format!("batch{rows}_down"), &[rows, hidden])?;
            let per_row = weighted_reference("perrow32_down", &[32, hidden])?;
            let bound = reference_bound(&batch, &per_row);
            for (tier, actual) in [("gpu", &gpu), ("cpu", &cpu), ("disk", &from_disk)] {
                for (name, expected) in [("batch", &batch), ("per_row", &per_row)] {
                    let mismatches = actual
                        .iter()
                        .zip(expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let maximum = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                        .fold(0.0f32, f32::max);
                    let pass = maximum <= bound && (bound != 0.0 || mismatches == 0);
                    defects += usize::from(!pass);
                    println!(
                        "{} expert_tier_weighted_reference rows={rows} weighted={weighted} tier={tier} reference={name} max_abs={maximum} bound={bound} bit_mismatches={mismatches}",
                        if pass { "PASS" } else { "DEFECT" }
                    );
                }
            }
            for (name, a, b) in [
                ("gpu_vs_cpu_pool", &gpu, &cpu),
                ("gpu_vs_disk", &gpu, &from_disk),
                ("cpu_pool_vs_disk", &cpu, &from_disk),
            ] {
                ensure!(
                    a.iter().chain(b).all(|value| value.is_finite()),
                    "nonfinite expert tier contribution {name}"
                );
                let mismatches = a
                    .iter()
                    .zip(b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let maximum = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                    .fold(0.0f32, f32::max);
                let first = a
                    .iter()
                    .zip(b)
                    .enumerate()
                    .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                    .map(|(index, (a, b))| {
                        (index, a.to_bits(), b.to_bits(), a.to_f32(), b.to_f32())
                    });
                println!(
                    "{} expert_tier_identity layer=0 expert={expert} rows={rows} weighted={weighted} comparison={name} stored=[rows,2048] bit_mismatches={mismatches} max_abs={maximum} first={first:?} cpu_isa={}",
                    if mismatches == 0 { "PASS" } else { "DEFECT" },
                    pool.isa().name()
                );
                defects += usize::from(mismatches != 0);
            }
        }
        ensure!(
            defects == 0,
            "expert tier contribution identity DEFECT cells={defects}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and FLYINGFISH_RUNTIME_DIR"]
    fn resident_graph_waits_for_prefill_refill_gpu() -> Result<()> {
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("graph refill runtime missing")?,
        );
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let mut capacity = capacity(
            geometry,
            8,
            false,
            2,
            true,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        capacity.slots_per_layer.fill(geometry.experts);
        capacity.device_expert_bytes = capacity.slot_bytes()
            * (u64::from(geometry.layers) * u64::from(geometry.experts)
                + u64::from(capacity.prefill_staging_slots));
        let initial =
            vec![(0..geometry.experts).rev().collect::<Vec<_>>(); geometry.layers as usize];
        let mut fixture = tier_fixture(3, 8);
        fixture.router_w.fill(bf16::ZERO);
        let mut engines = (0..2)
            .map(|arm| {
                Engine::load_moe_placed(
                    &fixture,
                    &runtime.join(format!("resident-refill-{arm}")),
                    &capacity,
                    &initial,
                    crate::experts::plan::Admission {
                        budget_bytes_per_token: u64::from(capacity.spare_blocks)
                            * capacity.slot_bytes(),
                    },
                    2,
                    8,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidate = engines.pop().context("graph refill candidate missing")?;
        let mut reference = engines.pop().context("graph refill reference missing")?;
        for engine in [&mut reference, &mut candidate] {
            engine.closure.freeze_fixed()?;
            engine.declare_graphs()?;
            engine.capture()?;
            engine.closure.expect_captures(1)?;
            engine.closure.ready(Instant::now())?;
            engine.reset()?;
        }
        reference.prefill(&[1, 2, 3, 4, 5], None)?;
        candidate
            .moe_slots
            .as_mut()
            .context("graph refill slots missing")?
            .delay_request = true;
        candidate.prefill(&[1, 2, 3, 4, 5], None)?;
        let slots = candidate
            .moe_slots
            .as_ref()
            .context("graph refill slots missing")?;
        ensure!(
            slots.all_resident() && !slots.moves.is_empty(),
            "refill control issued no resident copies"
        );
        candidate.set_token(7)?;
        candidate.decode(true, None)?;
        reference.set_token(7)?;
        reference.decode(false, None)?;
        let a = reference.decode_logits()?;
        let b = candidate.decode_logits()?;
        ensure!(
            a.iter().chain(&b).all(|value| value.is_finite()),
            "graph refill nonfinite logits"
        );
        let logits = a
            .iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let a = graph_state(&reference)?;
        let b = graph_state(&candidate)?;
        let state = a.iter().zip(&b).filter(|(a, b)| a != b).count() + a.len().abs_diff(b.len());
        println!(
            "{} resident_prefill_graph_refill logits_bit_mismatches={logits} state_bit_mismatches={state} delayed_upload=true",
            if logits == 0 && state == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        ensure!(
            logits == 0 && state == 0,
            "resident prefill graph refill DEFECT"
        );
        Ok(())
    }

    #[test]
    fn borrowed_span_views_preserve_global_choice_and_tail_offsets() -> Result<()> {
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?
        .slot_bytes();
        let service = ServiceTable::new(
            "fixture",
            [1, 64]
                .into_iter()
                .map(|rows| crate::experts::plan::ServiceCell {
                    workers: 2,
                    rows,
                    experts: 1,
                    seconds: f64::from(rows) * 0.00002,
                })
                .collect(),
        )?;
        assert_eq!(
            cpu_bound(geometry, slot_bytes, 263, &service, 1_000_000_000.0)?,
            (32, 32, 2, 2)
        );
        assert_eq!(
            cpu_bound(geometry, slot_bytes, 263, &service, 1_000_000_000_000.0)?,
            (1, 1, 0, 0)
        );
        assert_eq!(
            cpu_bound(geometry, slot_bytes, 263, &service, 1_000_000.0)?,
            (256, 256, 16, 64)
        );
        let mut counts = vec![0; geometry.experts as usize];
        counts[..5].copy_from_slice(&[0, 1, 15, 16, 17]);
        for measured in [15, 16, 64] {
            let copy_seconds = (f64::from(measured) + 0.5) * 0.00002;
            let h2d = slot_bytes as f64 / copy_seconds;
            let cpu = cpu_bound(geometry, slot_bytes, 263, &service, h2d)?;
            assert_eq!(cpu.3, measured);
            assert_eq!(cpu.2, measured.min(LANE16_ROWS));
            let mut mask = vec![false; geometry.experts as usize];
            arithmetic_mask(&counts, &mut mask)?;
            assert_eq!(&mask[..5], &[true, true, true, true, false]);
            let mut plan = SpanPlan::new(&geometry, slot_bytes, 1, 2)?;
            plan.update(
                &counts,
                &vec![false; geometry.experts as usize],
                263,
                4,
                &SpanRates {
                    service: &service,
                    h2d_bytes_per_second: h2d,
                    cpu_row_limit: cpu.2,
                },
            )?;
            assert!(
                plan.cpu()
                    .iter()
                    .all(|&expert| counts[expert as usize] <= cpu.2)
            );
            assert!(plan.staged().contains(&4));
            assert!(!plan.cpu().contains(&4));
            println!(
                "PASS deterministic_arithmetic_cpu measured={measured} effective={} boundary={LANE16_ROWS} counts=0,1,15,16,17",
                cpu.2
            );
        }
        for span_rows in [8, 17, 37, 263] {
            let layout =
                ExpertSpanLayout::new(&reserves(geometry), geometry, slot_bytes, span_rows)?;
            layout.validate_wave(4, 8)?;
            assert!(layout.validate_wave(4, 3).is_err());
            if span_rows >= 17 {
                assert!(layout.validate_wave(12, 8).is_err());
            }
            let span = Span {
                base: 0x1000,
                layout,
                geometry,
                positions: vec![7],
            };
            for base in (0..span_rows).step_by(4) {
                let rows = 4.min(span_rows - base);
                let x = span.hidden(&span.layout.normalized, base, rows)?;
                let ids = span.choices::<i32>(&span.layout.ids, base, rows)?;
                let weights = span.choices::<bf16>(&span.layout.weights, base, rows)?;
                let contributions = span.contributions(base, rows)?;
                assert_eq!(
                    x.ptr,
                    span.base + span.layout.normalized.start as u64 + u64::from(base) * 256
                );
                assert_eq!(
                    ids.ptr,
                    span.base + span.layout.ids.start as u64 + u64::from(base) * 16
                );
                assert_eq!(
                    weights.ptr,
                    span.base + span.layout.weights.start as u64 + u64::from(base) * 8
                );
                assert_eq!(
                    contributions.ptr,
                    span.base + span.layout.contributions.start as u64 + u64::from(base) * 1024
                );
                assert_eq!(contributions.shape, [rows, 4, 128]);
            }
            let tail = span.hidden(&span.layout.normalized, 0, 3)?;
            assert_eq!(tail.ptr, span.base + span.layout.normalized.start as u64);
            assert!(span.contributions(span_rows - 1, 2).is_err());
            assert!(span.choices::<bf16>(&span.layout.weights, 1, 1).is_err());
        }
        for pinned in [false, true] {
            for engines in [1, 2] {
                let capacity = capacity(
                    geometry,
                    17,
                    pinned,
                    engines,
                    true,
                    crate::model_spec::ExpertFormat::Bf16,
                )?;
                assert_eq!(capacity.host_blocks, 25);
                assert_eq!(capacity.spare_blocks, 2);
                assert_eq!(capacity.host_expert_bytes, 27 * slot_bytes);
                assert_eq!(capacity.disk_bytes, 0);
                assert_eq!(capacity.disk_workers, 0);
                assert_eq!(capacity.pinned, pinned);
            }
        }
        let capacity = capacity(
            geometry,
            17,
            false,
            2,
            false,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        assert_eq!(
            reference_cohorts(&[0, 3, 4, 5], 0, &capacity)?,
            (vec![4, 5], vec![vec![0, 3]])
        );
        assert_eq!(
            reference_cohorts(&[0, 3, 4, 5], 1, &capacity)?,
            (vec![3, 4, 5], vec![vec![0]])
        );
        assert_eq!(
            reference_cohorts(&[0, 3, 4, 5].repeat(3), 1, &capacity)?,
            (vec![], vec![vec![0], vec![3, 4], vec![5]])
        );
        println!(
            "PASS prefill_span_cpu_views spans=8,17,37,263 global_choice_offsets=true borrowed_tail_prefix=true wave_workspace_bound=true derived_host_surplus_blocks=1 GPU_executed=false"
        );
        Ok(())
    }

    fn inputs(variant: u32, rows: usize) -> MoeFixture {
        let mut fixture = tier_fixture(variant, 8);
        let source = fixture.x.clone();
        let pattern = (0..rows).map(|row| {
            if row + 7 < rows {
                0
            } else if row + 3 < rows {
                1
            } else if row + 1 < rows {
                2
            } else {
                3
            }
        });
        fixture.x = pattern
            .flat_map(|row| source[row * 128..(row + 1) * 128].iter().copied())
            .collect();
        fixture.dims.rows = rows;
        fixture
    }

    fn capacity(
        geometry: ExpertGeometry,
        rows: u32,
        pinned: bool,
        engines: u32,
        host_complete: bool,
        format: crate::model_spec::ExpertFormat,
    ) -> Result<Capacity> {
        let mut capacity = tier_capacity(geometry, pinned, format)?;
        capacity.span_rows = rows;
        capacity.wave_rows = 4;
        capacity.prefill_staging_slots = 2;
        if rows > 37 && format == crate::model_spec::ExpertFormat::Bf16 {
            capacity.slots_per_layer = vec![8, 8];
            capacity.device_expert_bytes = 16 * capacity.slot_bytes();
            capacity.disk_bytes = 8 * capacity.slot_bytes();
        } else if rows > 37 {
            let borrowed = crate::experts::plan::span_borrow_slots(
                &reserves(geometry),
                capacity.slot_bytes(),
                rows,
                rows,
            )?;
            let per_layer = (borrowed + geometry.layers)
                .div_ceil(geometry.layers)
                .max(8);
            ensure!(
                per_layer <= geometry.experts,
                "prefill fixture exceeds expert count"
            );
            capacity.slots_per_layer = vec![per_layer; geometry.layers as usize];
            capacity.device_expert_bytes =
                u64::from(capacity.slots_total()) * capacity.slot_bytes();
            capacity.disk_bytes = u64::from(
                geometry.layers * geometry.experts - capacity.slots_total() - capacity.host_blocks,
            ) * capacity.slot_bytes();
        }
        if host_complete {
            let reserves = reserves(geometry);
            let slot_bytes = capacity.slot_bytes();
            let borrowed =
                crate::experts::plan::span_borrow_slots(&reserves, slot_bytes, rows, rows)?;
            let non_gpu = geometry.layers * geometry.experts - capacity.slots_total();
            let host_bytes = u64::from(non_gpu + borrowed + capacity.spare_blocks) * slot_bytes;
            let service = ServiceTable::new(
                "fixture",
                vec![crate::experts::plan::ServiceCell {
                    workers: capacity.worker_count,
                    rows: 1,
                    experts: 1,
                    seconds: 0.00002,
                }],
            )?;
            return crate::experts::plan::capacity(
                &crate::experts::plan::PlanInputs {
                    geometry,
                    layout: capacity.layout.clone(),
                    device_free_bytes: capacity.device_expert_bytes
                        + reserves.persistent_bytes(slot_bytes)?,
                    host_available_bytes: host_bytes + reserves.host_bytes(slot_bytes)?,
                    h2d_bytes_per_second: 1_000_000_000.0,
                    d2h_bytes_per_second: 1_000_000_000.0,
                    copy_engines: engines,
                    disk_readers: 0,
                    host_registration_bytes: if pinned { host_bytes } else { 0 },
                    service: &service,
                    reserves: &reserves,
                    span_rows: rows,
                    wave_rows: 4,
                },
                capacity.spare_blocks,
            );
        }
        Ok(capacity)
    }

    struct Reference {
        ids: Vec<i32>,
        weights: Vec<bf16>,
        contributions: Vec<bf16>,
        output: Vec<bf16>,
    }

    fn reference(
        engine: &Engine,
        fixture: &MoeFixture,
        layer: u32,
        wave: usize,
        cohorts: impl FnOnce(&[i32]) -> Result<(Vec<u32>, Vec<Vec<u32>>)>,
    ) -> Result<Reference> {
        let stream = &engine.device.stream;
        let rows = fixture.dims.rows;
        let decode = engine
            .moe
            .as_ref()
            .context("prefill reference executor missing")?;
        let ffn = engine.moe_ffn_binding(layer as usize)?;
        let BoundFfn::Moe {
            router,
            shared,
            shared_gate,
            ..
        } = ffn
        else {
            anyhow::bail!("prefill reference has a dense FFN");
        };
        let input = stream.clone_htod(&fixture.x)?;
        let ids = stream.alloc_zeros::<i32>(rows * 4)?;
        let weights = stream.alloc_zeros::<bf16>(rows * 4)?;
        let logits = stream.alloc_zeros::<bf16>(rows * 16)?;
        decode.route.route_prefill(
            stream,
            qwen::matrix(&input, stream, 0, rows, 128)?,
            router,
            &mut TensorMut(view(&ids, stream, 0, [rows as u32, 4], [4, 1])?),
            &mut TensorMut(view(&weights, stream, 0, [rows as u32, 4], [4, 1])?),
            &mut TensorMut(qwen::matrix(&logits, stream, 0, rows, 16)?),
        )?;
        let host_ids = stream.clone_dtoh(&ids)?;
        let host_weights = stream.clone_dtoh(&weights)?;
        let mut counts = [0; 16];
        for &expert in &host_ids {
            counts[expert as usize] += 1;
        }
        let mut arithmetic = [false; 16];
        arithmetic_mask(&counts, &mut arithmetic)?;
        let (cpu_experts, cohorts) = cohorts(&host_ids)?;
        let slots = engine
            .moe_slots
            .as_ref()
            .context("prefill reference slots missing")?;
        let table = slots.bindings(layer as usize, stream)?;
        let pointers = (0..16)
            .map(|expert| {
                let position = slots.policy.ledger().slot(layer, expert)?;
                let pointer = slots.physical[position as usize].pointer;
                Ok((pointer, pointer + slots.catalogue.layout().down_offset()))
            })
            .collect::<Result<Vec<_>>>()?;
        let gate_up = pointers.iter().map(|p| p.0).collect::<Vec<_>>();
        let down = pointers.iter().map(|p| p.1).collect::<Vec<_>>();
        let generation = stream.clone_htod(
            &host_ids
                .iter()
                .map(|&expert| slots.published[layer as usize * 16 + expert as usize])
                .collect::<Vec<_>>(),
        )?;
        let mut active = stream.alloc_zeros::<u32>(rows * 4)?;
        let mut contributions = stream.alloc_zeros::<bf16>(rows * 4 * 128)?;
        let mut done = stream.alloc_zeros::<u32>(rows * 4)?;
        let y = stream.alloc_zeros::<bf16>(rows * 128)?;
        let scratch = stream.alloc_zeros::<u8>(prefill_scratch_bytes(
            wave,
            wave,
            4,
            16,
            128,
            64,
            slots.catalogue.layout().fp8_slot(),
        )? as usize)?;
        for experts in &cohorts {
            let mask = host_ids
                .iter()
                .map(|&expert| u32::from(experts.contains(&(expert as u32))))
                .collect::<Vec<_>>();
            stream.memcpy_htod(&mask, &mut active)?;
            for base in (0..rows).step_by(wave) {
                let count = wave.min(rows - base);
                decode.prefill.prefill_contribute(
                    stream,
                    MoePrefillContribute {
                        cpu_arithmetic: &arithmetic,
                        fp8_slot: slots.catalogue.layout().fp8_slot(),
                        wave: None,
                        wait_copy: None,
                        reader_event: None,
                        x: qwen::matrix(&input, stream, base * 128, count, 128)?,
                        ids: view(&ids, stream, base * 4, [count as u32, 4], [4, 1])?,
                        weights: view(&weights, stream, base * 4, [count as u32, 4], [4, 1])?,
                        active: view(&active, stream, base * 4, [count as u32, 4], [4, 1])?,
                        assignment_gen: view(
                            &generation,
                            stream,
                            base * 4,
                            [count as u32, 4],
                            [4, 1],
                        )?,
                        slots_gate_up: table.gate_up,
                        slots_down: table.down,
                        slot_valid: table.valid,
                        slot_generation: table.generation,
                        slots_gate_up_host: &gate_up,
                        slots_down_host: &down,
                        shared_gate: shared[0],
                        shared_up: shared[1],
                        shared_down: shared[2],
                        shared_expert_gate: shared_gate,
                        contributions: view(
                            &contributions,
                            stream,
                            base * 4 * 128,
                            [count as u32, 4, 128],
                            [512, 128, 1],
                        )?,
                        done: TensorMut(view(&done, stream, base * 4, [count as u32, 4], [4, 1])?),
                    },
                    &qwen::scratch(&scratch, stream),
                )?;
                stream.synchronize()?;
            }
        }
        let mut expected = stream.clone_dtoh(&contributions)?;
        let shape = Shape::new(128, 64)?;
        let mut cpu_scratch = Scratch::new(&shape);
        let native = ExpertWeights {
            gate_up: 0..16384,
            gate_up_stride: 128,
            down: 16384..24576,
            down_stride: 64,
        };
        for &expert in &cpu_experts {
            let expert = expert as usize;
            let mut native_slot = vec![0u8; slots.catalogue.slot_bytes() as usize];
            let arena = if slots.catalogue.layout().fp8_slot().is_some() {
                stream.synchronize()?;
                // SAFETY: the resident oracle owns this initialized native slot until this read completes.
                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        native_slot.as_mut_ptr().cast(),
                        pointers[expert].0,
                        native_slot.len(),
                    )
                    .result()?;
                }
                Vec::new()
            } else {
                fixture.gate_up[expert * 16384..(expert + 1) * 16384]
                    .iter()
                    .chain(&fixture.down[expert * 8192..(expert + 1) * 8192])
                    .map(|&value| {
                        if layer == 0 {
                            value
                        } else {
                            bf16::from_f32(-0.75 * value.to_f32())
                        }
                    })
                    .collect::<Vec<_>>()
            };
            for (choice, (&id, &weight)) in host_ids.iter().zip(&host_weights).enumerate() {
                if id as usize != expert {
                    continue;
                }
                let row = choice / 4;
                let hidden = &fixture.x[row * 128..(row + 1) * 128];
                let out = &mut expected[choice * 128..(choice + 1) * 128];
                if let Some(slot_layout) = slots.catalogue.layout().fp8_slot() {
                    fp8_contribution(
                        Isa::detect(),
                        Fp8Contribution {
                            shape: &shape,
                            slot: &native_slot,
                            slot_layout,
                            hidden,
                            routing: weight,
                        },
                        out,
                        &mut cpu_scratch,
                    )?;
                } else {
                    contribution(
                        Isa::detect(),
                        Contribution {
                            shape: &shape,
                            arena: &arena,
                            weights: &native,
                            hidden,
                            routing: weight,
                        },
                        out,
                        &mut cpu_scratch,
                    )?;
                }
            }
        }
        stream.memcpy_htod(&expected, &mut contributions)?;
        let mut completed = stream.clone_dtoh(&done)?;
        for (choice, &expert) in host_ids.iter().enumerate() {
            let gpu = cohorts
                .iter()
                .any(|cohort| cohort.contains(&(expert as u32)));
            ensure!(
                completed[choice] == u32::from(gpu),
                "Runner FAIL direct cohort done choice={choice} expert={expert} done={} expected={gpu}",
                completed[choice]
            );
            if cpu_experts.contains(&(expert as u32)) {
                completed[choice] = 1;
            }
            ensure!(
                completed[choice] == 1,
                "Runner FAIL direct reference choice={choice} is uncovered"
            );
        }
        stream.memcpy_htod(&completed, &mut done)?;
        stream.memcpy_htod(&vec![1u32; rows * 4], &mut active)?;
        for base in (0..rows).step_by(wave) {
            let count = wave.min(rows - base);
            decode.prefill.prefill_finish(
                stream,
                MoePrefillFinish {
                    x: qwen::matrix(&input, stream, base * 128, count, 128)?,
                    ids: view(&ids, stream, base * 4, [count as u32, 4], [4, 1])?,
                    final_active: view(&active, stream, base * 4, [count as u32, 4], [4, 1])?,
                    contributions: view(
                        &contributions,
                        stream,
                        base * 512,
                        [count as u32, 4, 128],
                        [512, 128, 1],
                    )?,
                    done: view(&done, stream, base * 4, [count as u32, 4], [4, 1])?,
                    experts: 16,
                    shared_gate: shared[0],
                    shared_up: shared[1],
                    shared_down: shared[2],
                    shared_expert_gate: shared_gate,
                    y: TensorMut(qwen::matrix(&y, stream, base * 128, count, 128)?),
                },
                &qwen::scratch(&scratch, stream),
            )?;
            stream.synchronize()?;
        }
        Ok(Reference {
            ids: host_ids,
            weights: host_weights,
            contributions: expected,
            output: stream.clone_dtoh(&y)?,
        })
    }

    fn same_bits<T: Copy>(a: &[T], b: &[T], bits: impl Fn(T) -> u32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| bits(a) == bits(b))
    }

    fn gate(
        row_cases: &[u32],
        variants: &[u32],
        modes: &[(bool, u32)],
        host_complete: bool,
        format: crate::model_spec::ExpertFormat,
    ) -> Result<()> {
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("prefill gate runtime missing")?,
        );
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let mut defects = 0;
        let mut total_coverage = [0u64; 5];
        for &variant in variants {
            for &rows in row_cases {
                let fixture = inputs(variant, rows as usize);
                let oracle = Engine::load_moe_test(
                    &fixture,
                    &runtime.join(format!("ref-{variant}-{rows}")),
                    8,
                    format,
                )?;
                let mut expected = Vec::new();
                let reference_capacity = tier_capacity(geometry, false, format)?;
                let ceiling = cpu_bound(
                    geometry,
                    reference_capacity.slot_bytes(),
                    rows,
                    &tier_service(reference_capacity.worker_count)?,
                    1_000_000_000.0,
                )?
                .2;
                for layer in 0..2 {
                    let first = reference(&oracle, &fixture, layer, 4, |_| {
                        Ok((vec![], vec![(0..16).collect()]))
                    })?;
                    let second = reference(&oracle, &fixture, layer, 4, |_| {
                        Ok((vec![], vec![(0..16).collect()]))
                    })?;
                    ensure!(
                        first.ids == second.ids
                            && same_bits(&first.weights, &second.weights, |x| u32::from(
                                x.to_bits()
                            ))
                            && same_bits(&first.contributions, &second.contributions, |x| {
                                u32::from(x.to_bits())
                            })
                            && same_bits(&first.output, &second.output, |x| u32::from(x.to_bits())),
                        "Runner FAIL prefill same-form repeat layer={layer} rows={rows}"
                    );
                    expected.push(first);
                }
                for (pinned, engines, backed) in modes.iter().flat_map(|&(pinned, engines)| {
                    [false, true]
                        .into_iter()
                        .filter(move |&backed| !backed || host_complete)
                        .map(move |backed| (pinned, engines, backed))
                }) {
                    let directory =
                        runtime.join(format!("span-{variant}-{rows}-{pinned}-{engines}-{backed}"));
                    let mut capacity =
                        capacity(geometry, rows, pinned, engines, host_complete, format)?;
                    if backed {
                        capacity.host_blocks = geometry.layers * geometry.experts;
                        capacity.spare_blocks = 0;
                        capacity.host_expert_bytes =
                            u64::from(capacity.host_blocks + capacity.spare_blocks)
                                * capacity.slot_bytes();
                        capacity.disk_bytes = 0;
                        capacity.disk_workers = 0;
                    }
                    let initial = vec![(0..capacity.slots_per_layer[0]).collect::<Vec<_>>(); 2];
                    let mut engine = Engine::load_moe_placed(
                        &fixture,
                        &directory,
                        &capacity,
                        &initial,
                        crate::experts::plan::Admission {
                            budget_bytes_per_token: 2 * capacity.slot_bytes(),
                        },
                        engines,
                        8,
                    )?;
                    let mut prefill = engine
                        .moe_prefill
                        .take()
                        .context("prefill gate executor missing")?;
                    ensure!(
                        prefill.cpu_ceiling == ceiling,
                        "prefill oracle CPU ceiling mismatch"
                    );
                    let mut slots = engine
                        .moe_slots
                        .take()
                        .context("prefill gate arena missing")?;
                    let mut decode = engine.moe.take().context("prefill gate CPU pool missing")?;
                    ensure!(
                        slots.request_open,
                        "Runner FAIL prefill lifecycle control has no open request"
                    );
                    ensure!(
                        prefill.begin(&engine.device, &mut slots, rows).is_err()
                            && prefill.span.is_none()
                            && slots.request_open,
                        "DEFECT prefill silently ended an open request"
                    );
                    slots.end_request()?;
                    if host_complete {
                        super::super::delay_stream(&slots.transfers.d2h)?;
                    }
                    {
                        let _range =
                            crate::backend::cuda::profile::Range::new(true, c"engine_moe_borrow");
                        prefill.begin(&engine.device, &mut slots, rows)?;
                    }
                    let span = prefill.span.as_ref().context("prefill gate span missing")?;
                    slots.policy.ledger().check()?;
                    if host_complete {
                        ensure!(
                            (prefill.stats[4] == 0) == backed
                                && prefill.stats[5] == 0
                                && slots.disk.is_none(),
                            "Runner FAIL host borrow fixture: backed={backed} D2H_bytes={} dropped_bytes={} disk_reader={}",
                            prefill.stats[4],
                            prefill.stats[5],
                            slots.disk.is_some()
                        );
                    }
                    let output = engine
                        .device
                        .stream
                        .alloc_zeros::<bf16>(rows as usize * 128)?;
                    let input = engine.device.stream.clone_htod(&fixture.x)?;
                    let span_input = span.hidden(&span.layout.normalized, 0, rows)?;
                    device_copy(
                        &engine.device.stream,
                        qwen::matrix(&input, &engine.device.stream, 0, rows as usize, 128)?,
                        span_input,
                    )?;
                    let mut coverage = [0u64; 5];
                    let mut forecast = Vec::new();
                    let mut expected_cancellations = 0;
                    if rows == 263 {
                        ensure!(
                            rows * geometry.top_k / geometry.experts > prefill.service.max_rows()
                                && (capacity.slot_bytes() as f64 / prefill.h2d_bytes_per_second)
                                    < prefill.service.rows_time(
                                        prefill.service.max_rows(),
                                        1,
                                        prefill.service.worker_count()?,
                                    )?,
                            "Runner FAIL S263 uniform forecast is not wholly streamed"
                        );
                    }
                    for (layer, all_resident) in expected.iter().enumerate() {
                        let prior_coverage = coverage;
                        let mut counts = vec![0; geometry.experts as usize];
                        for &expert in &all_resident.ids {
                            counts[expert as usize] += 1;
                        }
                        let resident = (0..geometry.experts)
                            .map(|expert| {
                                Ok(slots.policy.ledger().tier(layer as u32, expert)? == Tier::Gpu)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let mut expected_plan = SpanPlan::new(
                            &geometry,
                            slots.catalogue.slot_bytes(),
                            2,
                            prefill.workers,
                        )?;
                        expected_plan.update(
                            &counts,
                            &resident,
                            rows,
                            4,
                            &SpanRates {
                                service: &prefill.service,
                                h2d_bytes_per_second: prefill.h2d_bytes_per_second,
                                cpu_row_limit: prefill.cpu_ceiling,
                            },
                        )?;
                        expected_cancellations += forecast
                            .iter()
                            .filter(|expert| {
                                !expected_plan
                                    .cohorts()
                                    .any(|cohort| cohort.contains(expert))
                            })
                            .count() as u64;
                        forecast.clear();
                        if rows == 263 && layer + 1 < geometry.layers as usize {
                            for expert in 0..geometry.experts {
                                if slots.policy.ledger().tier(layer as u32 + 1, expert)?
                                    == Tier::Disk
                                {
                                    forecast.push(expert);
                                }
                            }
                        }
                        coverage[0] += expected_plan
                            .resident()
                            .iter()
                            .map(|&expert| u64::from(counts[expert as usize]))
                            .sum::<u64>();
                        for &expert in expected_plan.cpu() {
                            coverage[if slots.policy.ledger().tier(layer as u32, expert)?
                                == Tier::Host
                            {
                                1
                            } else {
                                2
                            }] += u64::from(counts[expert as usize]);
                        }
                        for &expert in expected_plan.cohorts().flatten() {
                            coverage[if slots.policy.ledger().tier(layer as u32, expert)?
                                == Tier::Host
                            {
                                3
                            } else {
                                4
                            }] += 1;
                        }
                        let ffn = engine.moe_ffn_binding(layer)?;
                        {
                            let _range =
                                crate::backend::cuda::profile::Range::new(true, c"engine_moe_span");
                            prefill.ffn(
                                &engine.device,
                                &mut slots,
                                &mut decode,
                                layer as u32,
                                rows,
                                ffn,
                            )?;
                            if !prefill.cpu.choices.is_empty() {
                                super::super::delay_stream(&prefill.gpu.mailbox)?;
                                prefill.cpu.upload(
                                    &prefill.gpu.mailbox,
                                    &engine.device.stream,
                                    prefill.span.as_ref().context("upload span missing")?,
                                    &mut slots.events,
                                )?;
                                let uploaded = slots.events.record(&prefill.gpu.mailbox)?;
                                ensure!(
                                    !uploaded.event.try_is_complete()?,
                                    "DEFECT prefill CPU upload blocked the host until completion"
                                );
                                println!(
                                    "PASS prefill_upload_async layer={layer} rows={rows} variant={variant} pinned={pinned} engines={engines} backed={backed} upload_pending=true delay_ms=100"
                                );
                            }
                            super::super::delay_stream(&engine.device.stream)?;
                            for base in (0..rows).step_by(4) {
                                let count = 4.min(rows - base);
                                prefill.finish_wave(
                                    &engine.device,
                                    &mut decode,
                                    ffn,
                                    base,
                                    count,
                                    qwen::matrix(
                                        &output,
                                        &engine.device.stream,
                                        base as usize * 128,
                                        count as usize,
                                        128,
                                    )?,
                                )?;
                            }
                            engine.device.stream.synchronize()?;
                        }
                        let reference = reference(&oracle, &fixture, layer as u32, 4, |_| {
                            Ok((expected_plan.cpu().to_vec(), vec![(0..16).collect()]))
                        })?;
                        let actual = engine.device.stream.clone_dtoh(&output)?;
                        let span = prefill.span.as_ref().context("prefill gate span missing")?;
                        let mut contributions = vec![bf16::ZERO; rows as usize * 4 * 128];
                        let mut done = vec![0u32; rows as usize * 4];
                        // SAFETY: the borrowed span covers both live prefixes through stream synchronization.
                        unsafe {
                            result::memcpy_dtoh_async(
                                &mut contributions,
                                span.contributions(0, rows)?.ptr,
                                engine.device.stream.cu_stream(),
                            )?;
                            result::memcpy_dtoh_async(
                                &mut done,
                                span.choices::<u32>(&span.layout.done, 0, rows)?.ptr,
                                engine.device.stream.cu_stream(),
                            )?;
                        }
                        engine.device.stream.synchronize()?;
                        ensure!(
                            actual
                                .iter()
                                .chain(&reference.output)
                                .chain(&contributions)
                                .chain(&reference.contributions)
                                .all(|value| value.to_f32().is_finite()),
                            "Runner FAIL prefill output nonfinite layer={layer} rows={rows}"
                        );
                        let output_mismatches = actual
                            .iter()
                            .zip(&reference.output)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        let wave_mismatches = actual
                            .chunks(4 * 128)
                            .zip(reference.output.chunks(4 * 128))
                            .map(|(a, b)| {
                                a.iter()
                                    .zip(b)
                                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                                    .count()
                            })
                            .collect::<Vec<_>>();
                        println!(
                            "{} prefill_span_waves layer={layer} rows={rows} variant={variant} pinned={pinned} engines={engines} wave_mismatches={wave_mismatches:?} delay_ms=100 ref_outside_NVTX=true",
                            if output_mismatches == 0 {
                                "PASS"
                            } else {
                                "DEFECT"
                            }
                        );
                        let mut cohort_mismatches = [0usize; 2];
                        for (choice, (&expert, values)) in reference
                            .ids
                            .iter()
                            .zip(contributions.as_chunks::<128>().0)
                            .enumerate()
                        {
                            let cpu = expected_plan.cpu().contains(&(expert as u32));
                            cohort_mismatches[usize::from(cpu)] += values
                                .iter()
                                .zip(&reference.contributions[choice * 128..(choice + 1) * 128])
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                        }
                        let mismatches = output_mismatches
                            + cohort_mismatches.iter().sum::<usize>()
                            + done.iter().filter(|&&value| value != 1).count()
                            + prefill.gpu.host_ids.as_slice()?[..rows as usize * 4]
                                .iter()
                                .zip(&reference.ids)
                                .filter(|(a, b)| a != b)
                                .count()
                            + prefill.gpu.host_weights.as_slice()?[..rows as usize * 4]
                                .iter()
                                .zip(&reference.weights)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                        defects += usize::from(mismatches != 0);
                        let cell_coverage: [u64; 5] =
                            std::array::from_fn(|index| coverage[index] - prior_coverage[index]);
                        println!(
                            "{} prefill_span layer={layer} rows={rows} wave=4 variant={variant} pinned={pinned} engines={engines} backed={backed} bit_mismatches={mismatches} T3_contribution_mismatches={} CPU_contribution_mismatches={} combine_mismatches={output_mismatches} coverage={cell_coverage:?} ref_outside_NVTX=true",
                            if mismatches == 0 { "PASS" } else { "DEFECT" },
                            cohort_mismatches[0],
                            cohort_mismatches[1]
                        );
                    }
                    {
                        let _range =
                            crate::backend::cuda::profile::Range::new(true, c"engine_moe_refill");
                        prefill.end(&engine.device, &mut slots)?;
                        engine.device.stream.synchronize()?;
                    }
                    ensure!(
                        prefill.span.is_none() && slots.request_open,
                        "Runner FAIL prefill request boundary not restored"
                    );
                    ensure!(
                        coverage[0] > 0
                            && (format != crate::model_spec::ExpertFormat::Bf16 || coverage[1] > 0)
                            && coverage[3] > 0
                            && if host_complete {
                                coverage[2] == 0 && coverage[4] == 0
                            } else {
                                coverage[2] > 0 && coverage[4] > 0
                            },
                        "Runner FAIL prefill fixture coverage={coverage:?} host_complete={host_complete}"
                    );
                    for (total, count) in total_coverage.iter_mut().zip(coverage) {
                        *total += count;
                    }
                    ensure!(
                        rows != 263 || expected_cancellations > 0,
                        "Runner FAIL prefill S263 has no wrong-prediction cancellation decisions"
                    );
                    let state_mismatch = prefill.stats[0] != 2
                        || prefill.stats[1] != 2 * u64::from(rows)
                        || prefill.stats[2] != (coverage[3] + coverage[4]) * capacity.slot_bytes()
                        || prefill.stats[3] != coverage[1] + coverage[2]
                        || prefill.gpu.read_ahead.stats[4] != expected_cancellations
                        || prefill.gpu.read_ahead.stats[5] != 1;
                    defects += usize::from(state_mismatch);
                    if let Err(error) = slots.policy.ledger().check() {
                        defects += 1;
                        println!(
                            "DEFECT prefill_span_ledger rows={rows} variant={variant} pinned={pinned} engines={engines} error={error:#}"
                        );
                    }
                    println!(
                        "{} prefill_span_coverage rows={rows} variant={variant} pinned={pinned} engines={engines} backed={backed} stats={:?} IO={:?} expected={coverage:?} expected_cancellations={expected_cancellations} ref_outside_NVTX=true",
                        if state_mismatch { "DEFECT" } else { "PASS" },
                        prefill.stats,
                        prefill.gpu.read_ahead.stats
                    );
                    slots.end_request()?;
                }
            }
        }
        ensure!(
            total_coverage[0] > 0
                && total_coverage[1] > 0
                && total_coverage[3] > 0
                && (host_complete || total_coverage[2] > 0 && total_coverage[4] > 0),
            "Runner FAIL prefill gate coverage={total_coverage:?} host_complete={host_complete}"
        );
        ensure!(defects == 0, "DEFECT prefill span cells={defects}");
        println!(
            "PASS prefill_span_gate format={format:?} rows={row_cases:?} variants={variants:?} modes={modes:?} host_complete={host_complete} borrowed=true refill=true CPU_submit=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "authorized GPU window"]
    fn prefill_span_smoke_gpu() -> Result<()> {
        gate(
            &[17],
            &[3],
            &[(false, 1)],
            false,
            crate::model_spec::ExpertFormat::Bf16,
        )
    }

    #[test]
    #[ignore = "requires CUDA and FLYINGFISH_RUNTIME_DIR"]
    fn prefill_borrow_uses_disk_cpu_workers_gpu() -> Result<()> {
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("prefill runtime missing")?,
        );
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let rows = 8;
        let fixture = inputs(3, rows);
        let mut capacity = capacity(
            geometry,
            rows as u32,
            false,
            2,
            true,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        capacity.host_blocks = geometry.layers * geometry.experts - capacity.slots_total();
        capacity.host_expert_bytes =
            u64::from(capacity.host_blocks + capacity.spare_blocks) * capacity.slot_bytes();
        capacity.disk_workers = 1;
        ensure!(capacity.disk_bytes == 0, "fixture has an initial disk tier");
        let initial = vec![(0..capacity.slots_per_layer[0]).collect::<Vec<_>>(); 2];
        let mut engine = Engine::load_moe_placed(
            &fixture,
            &runtime.join("borrow-disk"),
            &capacity,
            &initial,
            crate::experts::plan::Admission {
                budget_bytes_per_token: 2 * capacity.slot_bytes(),
            },
            2,
            8,
        )?;
        let oracle = Engine::load_moe_test(
            &fixture,
            &runtime.join("borrow-oracle"),
            8,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        let mut prefill = engine.moe_prefill.take().context("prefill missing")?;
        let mut slots = engine.moe_slots.take().context("slots missing")?;
        let mut decode = engine.moe.take().context("decode missing")?;
        ensure!(slots.disk_cpu.is_some(), "borrow has no disk CPU workers");
        slots.end_request()?;
        prefill.begin(&engine.device, &mut slots, rows as u32)?;
        ensure!(prefill.stats[5] > 0, "borrow introduced no disk victims");
        let input = engine.device.stream.clone_htod(&fixture.x)?;
        let span = prefill.span.as_ref().context("span missing")?;
        device_copy(
            &engine.device.stream,
            qwen::matrix(&input, &engine.device.stream, 0, rows, 128)?,
            span.hidden(&span.layout.normalized, 0, rows as u32)?,
        )?;
        let mut disk_choices = 0;
        for layer in 0..geometry.layers {
            prefill.ffn(
                &engine.device,
                &mut slots,
                &mut decode,
                layer,
                rows as u32,
                engine.moe_ffn_binding(layer as usize)?,
            )?;
            let expected = reference(&oracle, &fixture, layer, 4, |_| {
                Ok((prefill.plan.cpu().to_vec(), vec![(0..16).collect()]))
            })?;
            let span = prefill.span.as_ref().context("span missing")?;
            let mut actual = vec![bf16::ZERO; rows * 4 * 128];
            // SAFETY: the borrowed span owns all live contribution rows until refill.
            unsafe {
                result::memcpy_dtoh_async(
                    &mut actual,
                    span.contributions(0, rows as u32)?.ptr,
                    engine.device.stream.cu_stream(),
                )?;
            }
            engine.device.stream.synchronize()?;
            for (choice, &expert) in expected.ids.iter().enumerate() {
                if prefill.plan.cpu().contains(&(expert as u32))
                    && slots.policy.ledger().tier(layer, expert as u32)? == Tier::Disk
                {
                    disk_choices += 1;
                    ensure!(
                        same_bits(
                            &actual[choice * 128..(choice + 1) * 128],
                            &expected.contributions[choice * 128..(choice + 1) * 128],
                            |x| u32::from(x.to_bits())
                        ),
                        "DEFECT borrowed disk contribution layer={layer} choice={choice}"
                    );
                }
            }
        }
        ensure!(
            disk_choices > 0,
            "fixture selected no borrowed disk CPU expert"
        );
        prefill.end(&engine.device, &mut slots)?;
        engine.device.stream.synchronize()?;
        slots.policy.ledger().check()?;
        slots.end_request()?;
        println!(
            "PASS prefill_borrow_disk initial_disk_bytes=0 disk_choices={disk_choices} contribution_bit_mismatches=0 refill=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "authorized GPU window"]
    fn prefill_span_gpu() -> Result<()> {
        gate(
            &[17, 263],
            &[3, 7],
            &[(false, 1), (false, 2), (true, 1), (true, 2)],
            false,
            crate::model_spec::ExpertFormat::Bf16,
        )
    }

    #[test]
    #[ignore = "authorized GPU window"]
    fn prefill_borrow_host_gpu() -> Result<()> {
        gate(
            &[17],
            &[3, 7],
            &[(false, 1), (false, 2), (true, 1), (true, 2)],
            true,
            crate::model_spec::ExpertFormat::Bf16,
        )
    }

    #[test]
    #[ignore = "GPU window and FLYINGFISH_RUNTIME_DIR required"]
    fn fp8_prefill_residency_gpu() -> Result<()> {
        gate(
            &[17, 263],
            &[3, 7],
            &[(false, 1), (true, 2)],
            false,
            crate::model_spec::ExpertFormat::Fp8W8a16,
        )?;
        gate(
            &[17],
            &[3, 7],
            &[(false, 1), (true, 2)],
            true,
            crate::model_spec::ExpertFormat::Fp8W8a16,
        )?;
        Ok(())
    }

    fn reference_order(
        engine: &mut Engine,
        ids: &[u32],
        chunk: usize,
    ) -> Result<(Vec<bf16>, Vec<f32>)> {
        engine.reset()?;
        engine
            .moe_slots
            .as_mut()
            .context("prefill order reference has no slots")?
            .begin_request(&vec![vec![1.0; 16]; 2])?;
        let mut hidden = Vec::new();
        for values in ids.chunks(chunk) {
            engine.state.set_sequence(
                &engine.device.stream,
                engine.position as u32,
                values.len() as u32,
                engine.workspace_rows as u32,
            )?;
            engine
                .device
                .stream
                .memcpy_htod(values, &mut engine.work.ids.slice_mut(..values.len()))?;
            engine
                .device
                .stream
                .memcpy_htod(&[engine.position as i32], &mut engine.work.position)?;
            engine.forward(
                values.len(),
                false,
                engine.position + values.len() == ids.len(),
                None,
            )?;
            hidden.extend(
                engine
                    .device
                    .stream
                    .clone_dtoh(&engine.work.x.slice(..values.len() * 128))?,
            );
            engine.position += values.len();
        }
        Ok((hidden, engine.decode_logits()?))
    }

    fn max_difference<T: Copy>(a: &[T], b: &[T], value: impl Fn(T) -> f32) -> Result<f32> {
        ensure!(
            a.len() == b.len() && !a.is_empty(),
            "Runner FAIL prefill comparison extents {} vs {}",
            a.len(),
            b.len()
        );
        ensure!(
            a.iter().chain(b).all(|&x| value(x).is_finite()),
            "Runner FAIL prefill comparison contains nonfinite values"
        );
        Ok(a.iter()
            .zip(b)
            .map(|(&a, &b)| (value(a) - value(b)).abs())
            .fold(0.0, f32::max))
    }

    fn reference_cohorts(
        ids: &[i32],
        layer: u32,
        capacity: &Capacity,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>)> {
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        ensure!(
            !ids.is_empty() && ids.len().is_multiple_of(geometry.top_k as usize),
            "order reference choices={} are not complete top_k={} rows",
            ids.len(),
            geometry.top_k
        );
        let layout = crate::experts::arena::SlotLayout::new(geometry, capacity)?;
        let borrowed = crate::experts::plan::span_borrow_slots(
            &reserves(geometry),
            capacity.slot_bytes(),
            capacity.span_rows,
            capacity.span_rows,
        )?;
        let mut resident = vec![false; geometry.experts as usize];
        for expert in 0..layout.slot_count(layer)? {
            let slot = layout.resident_slot(layer, expert)?.start / layout.slot_bytes();
            resident[expert as usize] = slot < (capacity.slots_total() - borrowed) as usize;
        }
        let mut counts = vec![0; geometry.experts as usize];
        for &expert in ids {
            ensure!(
                (0..geometry.experts as i32).contains(&expert),
                "order reference expert={expert} outside0..{}",
                geometry.experts
            );
            counts[expert as usize] += 1;
        }
        let mut plan = SpanPlan::new(
            &geometry,
            capacity.slot_bytes(),
            capacity.prefill_staging_slots,
            capacity.worker_count,
        )?;
        plan.update(
            &counts,
            &resident,
            u32::try_from(ids.len() / geometry.top_k as usize)?,
            capacity
                .wave_rows
                .min((ids.len() / geometry.top_k as usize) as u32),
            &SpanRates {
                service: &tier_service(capacity.worker_count)?,
                h2d_bytes_per_second: 1_000_000_000.0,
                cpu_row_limit: LANE16_ROWS,
            },
        )?;
        let mut cohorts = Vec::new();
        if !plan.resident().is_empty() {
            cohorts.push(plan.resident().to_vec());
        }
        cohorts.extend(
            plan.staged()
                .chunks(capacity.prefill_staging_slots as usize)
                .map(<[u32]>::to_vec),
        );
        Ok((plan.cpu().to_vec(), cohorts))
    }

    fn reference_layer_order(
        engine: &mut Engine,
        ids: &[u32],
        schedule: [usize; 3],
        placement: Option<&Capacity>,
    ) -> Result<(Vec<bf16>, Vec<f32>)> {
        use crate::backend::Operator as O;
        let [span_rows, chunk, wave] = schedule;
        ensure!(
            span_rows > 0
                && chunk > 0
                && wave > 0
                && chunk <= engine.workspace_rows
                && wave <= engine.workspace_rows,
            "order reference schedule={schedule:?} workspace={}",
            engine.workspace_rows
        );
        engine.reset()?;
        let mut fixture = tier_fixture(3, 8);
        let mut hidden = Vec::new();
        for (span_index, admitted) in ids.chunks(span_rows).enumerate() {
            let start = span_index * span_rows;
            let mut residual = Vec::new();
            for (index, values) in admitted.chunks(chunk).enumerate() {
                let step = engine.moe_wave(start + index * chunk, values.len())?;
                engine
                    .device
                    .stream
                    .memcpy_htod(values, &mut engine.work.ids.slice_mut(..values.len()))?;
                engine.run_operator(O::Embed, 0, step, &mut None)?;
                residual.extend(
                    engine
                        .device
                        .stream
                        .clone_dtoh(&engine.work.x.slice(..values.len() * 128))?,
                );
            }
            for layer in 0..engine.model_spec.layers.len() {
                let mut normalized = Vec::new();
                for base in (0..admitted.len()).step_by(chunk) {
                    let rows = chunk.min(admitted.len() - base);
                    let step = engine.moe_wave(start + base, rows)?;
                    engine.select_qwen_positions()?;
                    engine.device.stream.memcpy_htod(
                        &residual[base * 128..(base + rows) * 128],
                        &mut engine.work.x.slice_mut(..rows * 128),
                    )?;
                    engine.qwen_layer_begin(layer, step)?;
                    engine.run_operator(O::Norm, layer, step, &mut None)?;
                    engine.qwen_mixer(layer, step, &mut None)?;
                    engine.run_operator(O::PostNorm, layer, step, &mut None)?;
                    residual[base * 128..(base + rows) * 128].copy_from_slice(
                        &engine
                            .device
                            .stream
                            .clone_dtoh(&engine.work.x.slice(..rows * 128))?,
                    );
                    normalized.extend(
                        engine
                            .device
                            .stream
                            .clone_dtoh(&engine.work.n.slice(..rows * 128))?,
                    );
                }
                fixture.x = normalized;
                fixture.dims.rows = admitted.len();
                let ffn = reference(
                    engine,
                    &fixture,
                    layer as u32,
                    wave,
                    |ids| match placement {
                        Some(capacity) => reference_cohorts(ids, layer as u32, capacity),
                        None => Ok((vec![], vec![(0..16).collect()])),
                    },
                )?;
                for base in (0..admitted.len()).step_by(wave) {
                    let rows = wave.min(admitted.len() - base);
                    let step = engine.moe_wave(start + base, rows)?;
                    engine.device.stream.memcpy_htod(
                        &residual[base * 128..(base + rows) * 128],
                        &mut engine.work.x.slice_mut(..rows * 128),
                    )?;
                    engine.device.stream.memcpy_htod(
                        &ffn.output[base * 128..(base + rows) * 128],
                        &mut engine.work.out.slice_mut(..rows * 128),
                    )?;
                    engine.run_operator(O::NextNorm, layer, step, &mut None)?;
                    residual[base * 128..(base + rows) * 128].copy_from_slice(
                        &engine
                            .device
                            .stream
                            .clone_dtoh(&engine.work.x.slice(..rows * 128))?,
                    );
                }
            }
            for base in (0..admitted.len()).step_by(chunk) {
                let rows = chunk.min(admitted.len() - base);
                let mut step = engine.moe_wave(start + base, rows)?;
                engine.device.stream.memcpy_htod(
                    &residual[base * 128..(base + rows) * 128],
                    &mut engine.work.x.slice_mut(..rows * 128),
                )?;
                if start + base + rows == ids.len() {
                    step.head = Head::Last;
                }
                engine.run_operator(O::Head, 0, step, &mut None)?;
            }
            hidden.extend(residual);
        }
        let logits = engine
            .device
            .stream
            .clone_dtoh(&engine.work.logits.slice(..32))?;
        Ok((hidden, logits.into_iter().map(bf16::to_f32).collect()))
    }

    #[test]
    #[ignore = "authorized GPU window"]
    fn prefill_layer_order_gpu() -> Result<()> {
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR")
                .context("prefill order gate runtime missing")?,
        );
        let fixture = tier_fixture(3, 8);
        let ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3];
        let mut oracle = Engine::load_moe_test(
            &fixture,
            &runtime.join("order-reference"),
            8,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        let baseline = reference_order(&mut oracle, &ids, 8)?;
        let repeated = reference_order(&mut oracle, &ids, 8)?;
        ensure!(
            same_bits(&baseline.0, &repeated.0, |x| u32::from(x.to_bits()))
                && same_bits(&baseline.1, &repeated.1, f32::to_bits),
            "Runner FAIL prefill order same-form repeat differs"
        );
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let capacities = [
            capacity(
                geometry,
                17,
                false,
                2,
                false,
                crate::model_spec::ExpertFormat::Bf16,
            )?,
            capacity(
                geometry,
                8,
                false,
                2,
                false,
                crate::model_spec::ExpertFormat::Bf16,
            )?,
        ];
        let mut floor_cases = Vec::new();
        for chunk in [1, 4, 2] {
            let other = reference_order(&mut oracle, &ids, chunk)?;
            let repeat = reference_order(&mut oracle, &ids, chunk)?;
            floor_cases.push((format!("chunk{chunk}"), other, repeat));
        }
        let mut wide = Engine::load_moe_test(
            &fixture,
            &runtime.join("order-reference-wide"),
            16,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        floor_cases.push((
            "chunk16".into(),
            reference_layer_order(&mut wide, &ids, [16, 16, 4], None)?,
            reference_layer_order(&mut wide, &ids, [16, 16, 4], None)?,
        ));
        for capacity in &capacities {
            for wave in [2, 8] {
                let schedule = [capacity.span_rows as usize, 8, wave];
                floor_cases.push((
                    format!("span{}-wave{wave}", capacity.span_rows),
                    reference_layer_order(&mut oracle, &ids, schedule, Some(capacity))?,
                    reference_layer_order(&mut oracle, &ids, schedule, Some(capacity))?,
                ));
            }
        }
        let mut floor = [0.0f32; 2];
        for (case, other, repeat) in floor_cases {
            ensure!(
                same_bits(&other.0, &repeat.0, |x| u32::from(x.to_bits()))
                    && same_bits(&other.1, &repeat.1, f32::to_bits),
                "Runner FAIL prefill floor reference repeat differs case={case}"
            );
            let hidden = max_difference(&baseline.0, &other.0, bf16::to_f32)?;
            let logits = max_difference(&baseline.1, &other.1, |x| x)?;
            floor[0] = floor[0].max(hidden);
            floor[1] = floor[1].max(logits);
            println!(
                "PASS prefill_order_floor_cell case={case} hidden_max_abs={hidden} logits_max_abs={logits} same_form_bitwise=true ref_outside_NVTX=true"
            );
        }
        println!(
            "PASS prefill_order_reference rows=17 chunks=8,4,1,2,16 waves=4,2,8 reference_cells=8 hidden_floor={} logits_floor={} same_form_bitwise=true",
            floor[0], floor[1]
        );
        let mut identities = Vec::new();
        for capacity in &capacities {
            let schedule = [capacity.span_rows as usize, 8, 4];
            let expected = reference_layer_order(&mut oracle, &ids, schedule, Some(capacity))?;
            let repeated = reference_layer_order(&mut oracle, &ids, schedule, Some(capacity))?;
            ensure!(
                same_bits(&expected.0, &repeated.0, |x| u32::from(x.to_bits()))
                    && same_bits(&expected.1, &repeated.1, f32::to_bits),
                "Runner FAIL same-order reference repeat differs span_rows={}",
                capacity.span_rows
            );
            max_difference(&expected.0, &repeated.0, bf16::to_f32)?;
            max_difference(&expected.1, &repeated.1, |x| x)?;
            println!(
                "PASS prefill_order_identity_reference span_rows={} chunk=8 wave=4 hidden_words={} logits_words={} same_form_bitwise=true ref_outside_NVTX=true",
                capacity.span_rows,
                expected.0.len(),
                expected.1.len()
            );
            identities.push(expected);
        }
        let mut defects = 0;
        for (capacity, identity) in capacities.iter().zip(&identities) {
            let span_rows = capacity.span_rows;
            let mut engine = Engine::load_moe_placed(
                &fixture,
                &runtime.join(format!("order-candidate-{span_rows}")),
                capacity,
                &[vec![0, 1, 2, 3], vec![0, 1, 2, 3]],
                crate::experts::plan::Admission {
                    budget_bytes_per_token: 2 * capacity.slot_bytes(),
                },
                2,
                8,
            )?;
            engine.closure.freeze_fixed()?;
            engine.closure.ready(Instant::now())?;
            engine.reset()?;
            let mut hidden = Vec::new();
            {
                let _range =
                    crate::backend::cuda::profile::Range::new(true, c"engine_moe_layer_order");
                engine.prefill_chunks(&ids, None, 0, |engine, values| {
                    hidden.extend(
                        engine
                            .device
                            .stream
                            .clone_dtoh(&engine.work.x.slice(..values.len() * 128))?,
                    );
                    Ok(())
                })?;
                engine.device.stream.synchronize()?;
            }
            let logits = engine.decode_logits()?;
            max_difference(&identity.0, &hidden, bf16::to_f32)?;
            max_difference(&identity.1, &logits, |x| x)?;
            let identity_hidden = identity
                .0
                .iter()
                .zip(&hidden)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let identity_logits = identity
                .1
                .iter()
                .zip(&logits)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let identity_pass = identity_hidden == 0 && identity_logits == 0;
            println!(
                "{} prefill_order_identity span_rows={span_rows} chunk=8 wave=4 hidden_bit_mismatches={identity_hidden} logits_bit_mismatches={identity_logits} hidden_words={} logits_words={} ref_outside_NVTX=true",
                if identity_pass { "PASS" } else { "DEFECT" },
                hidden.len(),
                logits.len()
            );
            let hidden_error = max_difference(&baseline.0, &hidden, bf16::to_f32)?;
            let logits_error = max_difference(&baseline.1, &logits, |x| x)?;
            let hidden_bitwise = same_bits(&baseline.0, &hidden, |x| u32::from(x.to_bits()));
            let logits_bitwise = same_bits(&baseline.1, &logits, f32::to_bits);
            let drains = engine
                .moe_prefill
                .as_ref()
                .context("prefill order has no executor")?
                .gpu
                .read_ahead
                .stats[5];
            let spans = ids.len().div_ceil(span_rows as usize);
            let pass = (if floor[0] == 0.0 {
                hidden_bitwise
            } else {
                hidden_error <= floor[0]
            }) && (if floor[1] == 0.0 {
                logits_bitwise
            } else {
                logits_error <= floor[1]
            }) && engine.position == ids.len()
                && engine.committed_prefix_ids == ids
                && engine.pending_token.is_some()
                && drains == spans as u64;
            defects += usize::from(!pass || !identity_pass);
            println!(
                "{} prefill_layer_order rows=17 span_rows={span_rows} spans={spans} last_span_rows={} drains={drains} layers=2 GDN=true attention_d256=true rope_kv=false hidden_max_abs={hidden_error} hidden_floor={} logits_max_abs={logits_error} logits_floor={} hidden_bitwise={hidden_bitwise} logits_bitwise={logits_bitwise} ref_outside_NVTX=true",
                if pass { "PASS" } else { "DEFECT" },
                ids.len() - (spans - 1) * span_rows as usize,
                floor[0],
                floor[1]
            );
        }
        ensure!(
            defects == 0,
            "DEFECT prefill layer order or sequence/drain boundary cells={defects}"
        );
        Ok(())
    }
}
