//! Eager routed-FFN execution over caller-provided residency decisions.
use super::*;
use crate::backend::cuda::{
    contracts::{Tensor, TensorMut},
    linear::BoundFfn,
    moe::{
        combine::{CombineOps, MoeCombine},
        decode::{DecodeOps, MoeDecode, decode_scratch_bytes},
        route::RouteOps,
    },
};
use crate::cpu::expert_ffn::{Assignment, ExpertPool, ExpertWeights, Job, Shape};
use crate::experts::{arena::HostBlocks, catalogue::ExpertGeometry};
use cudarc::driver::{CudaEvent, DevicePtr, DeviceRepr, HostSlice, ValidAsZeroBits, result};

fn pinned_zeros<T: DeviceRepr + ValidAsZeroBits>(
    context: &Arc<CudaContext>,
    len: usize,
) -> Result<PinnedHostSlice<T>> {
    // SAFETY: flags0 allocates CPU-readable pinned storage, initialized before creating typed slices.
    let mut allocation = unsafe { context.alloc_pinned_with_flags(len, 0)? };
    unsafe {
        std::ptr::write_bytes(allocation.as_mut_ptr()?, 0, len);
    }
    Ok(allocation)
}

#[derive(Clone, Copy)]
pub(super) struct SlotBindings {
    pub gate_up: Tensor<u64, 1>,
    pub down: Tensor<u64, 1>,
    pub valid: Tensor<u32, 1>,
    pub generation: Tensor<u64, 1>,
}

pub(super) struct Slots {
    storage: super::super::weights::ExpertStorage,
    gate_up: CudaSlice<u64>,
    down: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    published: Vec<u64>,
}

impl Slots {
    fn bindings(&self, layer: usize, stream: &Arc<CudaStream>) -> Result<SlotBindings> {
        let geometry = self.storage.layout().geometry();
        ensure!(
            layer < geometry.layers as usize,
            "MoE slot layer={layer} outside layers={}",
            geometry.layers
        );
        let experts = geometry.experts;
        let offset = layer * experts as usize;
        Ok(SlotBindings {
            gate_up: qwen::view(&self.gate_up, stream, offset, [experts], [1])?,
            down: qwen::view(&self.down, stream, offset, [experts], [1])?,
            valid: qwen::view(&self.valid, stream, offset, [experts], [1])?,
            generation: qwen::view(&self.generation, stream, offset, [experts], [1])?,
        })
    }
}

impl Engine {
    pub(super) fn run_moe_ffn(
        &mut self,
        layer: usize,
        rows: usize,
        ffn: BoundFfn,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        let decode = self
            .moe
            .as_mut()
            .context("MoE decode executor is not bound")?;
        let slots = self
            .moe_slots
            .as_ref()
            .context("MoE physical slot tables are not bound")?;
        let stream = &self.device.stream;
        let hidden = self.model_spec.hidden as usize;
        let x = qwen::matrix(&self.work.n, stream, 0, rows, hidden)?;
        let bindings = slots.bindings(layer, stream)?;
        decode.route(&self.device, ffn, x)?;
        let (ids, _) = decode.routed()?;
        let mut generation = [0u64; 8 * 32];
        let experts = slots.storage.layout().geometry().experts as usize;
        for (choice, &expert) in ids.iter().enumerate() {
            ensure!(
                expert >= 0 && (expert as usize) < experts,
                "MoE route choice={choice} expert={expert} outside experts={experts}"
            );
            generation[choice] = slots.published[layer * experts + expert as usize];
            ensure!(
                generation[choice] != 0,
                "MoE all-resident binding lacks layer={layer} expert={expert}"
            );
        }
        let count = ids.len();
        decode.finish(
            &self.device,
            ffn,
            bindings,
            Dispatch {
                active: &[1u32; 8 * 32][..count],
                generation: &generation[..count],
                host: None,
                experts: &[],
                assignments: &[],
                miss_rows: &[],
            },
            (
                x,
                &mut TensorMut(qwen::matrix(&self.work.out, stream, 0, rows, hidden)?),
            ),
        )?;
        if let Some(target) = trace.as_mut() {
            let prefix = if target.trace.all_layers {
                format!("layer_{layer}")
            } else {
                target.prefix.to_owned()
            };
            decode.trace(stream, target.trace, &prefix)?;
            trace_value(trace, stream, "ffn", &self.work.out, rows, hidden)?;
        }
        Ok(())
    }
}

pub(super) struct Dispatch<'a> {
    pub active: &'a [u32],
    pub generation: &'a [u64],
    pub host: Option<Arc<HostBlocks>>,
    pub experts: &'a [ExpertWeights],
    pub assignments: &'a [Assignment],
    pub miss_rows: &'a [u32],
}

pub(super) struct Decode {
    geometry: ExpertGeometry,
    max_rows: usize,
    rows: usize,
    pending: bool,
    input_ptr: u64,
    router_ptr: u64,
    route: RouteOps,
    contribute: DecodeOps,
    combine: CombineOps,
    ids: CudaSlice<i32>,
    weights: CudaSlice<bf16>,
    active: CudaSlice<u32>,
    final_active: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    contributions: CudaSlice<bf16>,
    done: CudaSlice<u32>,
    shared: CudaSlice<bf16>,
    scale: CudaSlice<bf16>,
    scratch: CudaSlice<u8>,
    host_ids: PinnedHostSlice<i32>,
    host_weights: PinnedHostSlice<bf16>,
    host_active: PinnedHostSlice<u32>,
    host_generation: PinnedHostSlice<u64>,
    host_x: PinnedHostSlice<bf16>,
    cpu_x: Vec<bf16>,
    cpu_out: PinnedHostSlice<bf16>,
    cpu_done: PinnedHostSlice<u32>,
    pool: ExpertPool,
    pub stats: [u64; 5],
}

impl Decode {
    pub fn new(
        device: &Device,
        geometry: ExpertGeometry,
        max_rows: usize,
        workers: u32,
    ) -> Result<Self> {
        ensure!(
            (1..=8).contains(&max_rows),
            "MoE decode row bound={max_rows} outside1..8"
        );
        ensure!(
            geometry.top_k > 0 && geometry.top_k <= 32 && geometry.top_k <= geometry.experts,
            "MoE decode top_k={} experts={}",
            geometry.top_k,
            geometry.experts
        );
        let choices = max_rows
            .checked_mul(geometry.top_k as usize)
            .context("MoE route extent overflow")?;
        let elements = choices
            .checked_mul(geometry.hidden as usize)
            .context("MoE contribution extent overflow")?;
        let hidden = max_rows
            .checked_mul(geometry.hidden as usize)
            .context("MoE hidden extent overflow")?;
        let stream = &device.stream;
        let context = &device.ctx;
        Ok(Self {
            geometry,
            max_rows,
            rows: 0,
            pending: false,
            input_ptr: 0,
            router_ptr: 0,
            route: RouteOps::new(context, stream)?,
            contribute: DecodeOps::new(context)?,
            combine: CombineOps::new(context)?,
            ids: stream.alloc_zeros(choices)?,
            weights: stream.alloc_zeros(choices)?,
            active: stream.alloc_zeros(choices)?,
            final_active: stream.clone_htod(&vec![1u32; choices])?,
            generation: stream.alloc_zeros(choices)?,
            contributions: stream.alloc_zeros(elements)?,
            done: stream.alloc_zeros(choices)?,
            shared: stream.alloc_zeros(hidden)?,
            scale: stream.alloc_zeros(max_rows)?,
            scratch: stream.alloc_zeros(usize::try_from(decode_scratch_bytes(
                max_rows,
                geometry.top_k as usize,
                geometry.intermediate as usize,
            ))?)?,
            host_ids: pinned_zeros(context, choices)?,
            host_weights: pinned_zeros(context, choices)?,
            host_active: pinned_zeros(context, choices)?,
            host_generation: pinned_zeros(context, choices)?,
            host_x: pinned_zeros(context, hidden)?,
            cpu_x: vec![bf16::ZERO; hidden],
            cpu_out: pinned_zeros(context, elements)?,
            cpu_done: pinned_zeros(context, choices)?,
            pool: ExpertPool::new(
                workers as usize,
                Shape::new(geometry.hidden as usize, geometry.intermediate as usize)?,
                max_rows,
                choices,
            )?,
            stats: [0; 5],
        })
    }

    pub fn route(&mut self, device: &Device, ffn: BoundFfn, x: Tensor<bf16, 2>) -> Result<()> {
        let BoundFfn::Moe { router, .. } = ffn else {
            anyhow::bail!("MoE route received a dense FFN");
        };
        ensure!(
            !self.pending,
            "MoE route has an unfinished contribution/combine phase"
        );
        let rows = x.shape[0] as usize;
        ensure!(
            rows > 0 && rows <= self.max_rows && x.shape[1] == self.geometry.hidden,
            "MoE input shape={:?} row_bound={} hidden={}",
            x.shape,
            self.max_rows,
            self.geometry.hidden
        );
        ensure!(
            router.shape == [self.geometry.experts, self.geometry.hidden],
            "MoE router shape={:?} geometry={:?}",
            router.shape,
            self.geometry
        );
        let k = self.geometry.top_k;
        let stream = &device.stream;
        self.route.route(
            stream,
            x,
            router,
            &mut TensorMut(qwen::view(
                &self.ids,
                stream,
                0,
                [rows as u32, k],
                [u64::from(k), 1],
            )?),
            &mut TensorMut(qwen::view(
                &self.weights,
                stream,
                0,
                [rows as u32, k],
                [u64::from(k), 1],
            )?),
        )?;
        device.upload.wait(&stream.record_event(None)?)?;
        let count = rows * k as usize;
        {
            let (ids, _ids) = self.ids.device_ptr(&device.upload);
            let (weights, _weights) = self.weights.device_ptr(&device.upload);
            let (host_ids, _ids_write) =
                unsafe { self.host_ids.stream_synced_mut_slice(&device.upload) };
            let (host_weights, _weights_write) =
                unsafe { self.host_weights.stream_synced_mut_slice(&device.upload) };
            let (host_x, _x_write) = unsafe { self.host_x.stream_synced_mut_slice(&device.upload) };
            // SAFETY: each copied prefix has its exact logical extent inside the owned mailbox.
            unsafe {
                result::memcpy_dtoh_async(&mut host_ids[..count], ids, device.upload.cu_stream())?;
                result::memcpy_dtoh_async(
                    &mut host_weights[..count],
                    weights,
                    device.upload.cu_stream(),
                )?;
                result::memcpy_dtoh_async(
                    &mut host_x[..rows * self.geometry.hidden as usize],
                    x.ptr,
                    device.upload.cu_stream(),
                )?;
            }
        }
        device.upload.synchronize()?;
        self.rows = rows;
        self.input_ptr = x.ptr;
        self.router_ptr = router.ptr;
        self.pending = true;
        self.stats[0] += 1;
        Ok(())
    }

    pub fn routed(&self) -> Result<(&[i32], &[bf16])> {
        let count = self.rows * self.geometry.top_k as usize;
        Ok((
            &self.host_ids.as_slice()?[..count],
            &self.host_weights.as_slice()?[..count],
        ))
    }

    pub fn finish(
        &mut self,
        device: &Device,
        ffn: BoundFfn,
        slots: SlotBindings,
        dispatch: Dispatch<'_>,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
    ) -> Result<CudaEvent> {
        let BoundFfn::Moe {
            router,
            shared,
            shared_gate,
            ..
        } = ffn
        else {
            anyhow::bail!("MoE completion received a dense FFN");
        };
        let (x, y) = activation;
        ensure!(
            self.pending && x.ptr == self.input_ptr && router.ptr == self.router_ptr,
            "MoE completion phase/input mismatch pending={} input={:#x} routed_input={:#x} router={:#x} routed_router={:#x}",
            self.pending,
            x.ptr,
            self.input_ptr,
            router.ptr,
            self.router_ptr
        );
        let (rows, k, hidden) = (
            self.rows,
            self.geometry.top_k as usize,
            self.geometry.hidden as usize,
        );
        ensure!(
            rows > 0 && x.shape == [rows as u32, hidden as u32],
            "MoE completion shape={:?} routed_rows={rows}",
            x.shape
        );
        ensure!(
            shared[0].shape[0] == self.geometry.intermediate,
            "MoE shared intermediate={} expert intermediate={}",
            shared[0].shape[0],
            self.geometry.intermediate
        );
        validate_dispatch(rows, k, self.routed()?.1, &dispatch)?;
        let stream = &device.stream;
        self.host_active.as_mut_slice()?[..rows * k].copy_from_slice(dispatch.active);
        self.host_generation.as_mut_slice()?[..rows * k].copy_from_slice(dispatch.generation);
        {
            let (active, _active_read) =
                unsafe { self.host_active.stream_synced_slice(&device.upload) };
            let (generation, _generation_read) =
                unsafe { self.host_generation.stream_synced_slice(&device.upload) };
            let (active_ptr, _active_write) = self.active.device_ptr_mut(&device.upload);
            let (generation_ptr, _generation_write) =
                self.generation.device_ptr_mut(&device.upload);
            // SAFETY: owned pinned prefixes and device destinations each contain rows*top_k elements.
            unsafe {
                result::memcpy_htod_async(
                    active_ptr,
                    &active[..rows * k],
                    device.upload.cu_stream(),
                )?;
                result::memcpy_htod_async(
                    generation_ptr,
                    &generation[..rows * k],
                    device.upload.cu_stream(),
                )?;
            }
        }
        stream.wait(&device.upload.record_event(None)?)?;
        {
            let (done_ptr, _write) = self.done.device_ptr_mut(stream);
            // SAFETY: done owns max_rows*top_k words and is ordered after the preceding layer's combine.
            unsafe {
                result::memset_d8_async(
                    done_ptr,
                    0,
                    rows * k * size_of::<u32>(),
                    stream.cu_stream(),
                )?
            };
        }
        let ids = qwen::view(&self.ids, stream, 0, [rows as u32, k as u32], [k as u64, 1])?;
        let active = qwen::view(
            &self.active,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        let contributions = qwen::view(
            &self.contributions,
            stream,
            0,
            [rows as u32, k as u32, hidden as u32],
            [(k * hidden) as u64, hidden as u64, 1],
        )?;
        let mut shared_out = TensorMut(qwen::matrix(&self.shared, stream, 0, rows, hidden)?);
        let mut done = TensorMut(qwen::view(
            &self.done,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?);
        let reader = self.contribute.contribute(
            stream,
            &MoeDecode {
                x,
                ids,
                active,
                weights: qwen::view(
                    &self.weights,
                    stream,
                    0,
                    [rows as u32, k as u32],
                    [k as u64, 1],
                )?,
                slots_gate_up: slots.gate_up,
                slots_down: slots.down,
                slot_valid: slots.valid,
                slot_generation: slots.generation,
                assignment_gen: qwen::view(
                    &self.generation,
                    stream,
                    0,
                    [rows as u32, k as u32],
                    [k as u64, 1],
                )?,
                shared_gate: shared[0],
                shared_up: shared[1],
                shared_down: shared[2],
                shared_expert_gate: shared_gate,
                contributions,
                shared_out: &mut shared_out,
                done: &mut done,
            },
            &qwen::scratch(&self.scratch, stream),
        )?;
        if !dispatch.assignments.is_empty() {
            self.run_cpu(device, &dispatch)?;
            device.upload.wait(&reader)?;
            let values = self.cpu_out.as_slice()?;
            let done_values = self.cpu_done.as_mut_slice()?;
            for assignment in dispatch.assignments {
                let choice = assignment.choice as usize;
                let first = choice * hidden;
                done_values[choice] = 1;
                // SAFETY: each validated CPU choice owns its contribution and done word; mailboxes live through publication.
                unsafe {
                    result::memcpy_htod_async(
                        contributions.ptr + (first * 2) as u64,
                        &values[first..first + hidden],
                        device.upload.cu_stream(),
                    )?;
                    result::memcpy_htod_async(
                        done.read().ptr + (choice * 4) as u64,
                        &done_values[choice..choice + 1],
                        device.upload.cu_stream(),
                    )?;
                }
            }
            let published = device.upload.record_event(None)?;
            stream.wait(&published)?;
            published.synchronize()?;
            self.stats[2] += dispatch.assignments.len() as u64;
            self.stats[3] += 1;
        }
        let mut scale = TensorMut(qwen::view(
            &self.scale,
            stream,
            0,
            [rows as u32, 1],
            [1, 1],
        )?);
        self.combine
            .shared_scale(stream, x, shared_gate, &mut scale)?;
        let final_active = qwen::view(
            &self.final_active,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        self.stats[1] += dispatch.active.iter().map(|&n| u64::from(n)).sum::<u64>();
        self.stats[4] += 1;
        let completed = self.combine.combine(
            stream,
            &MoeCombine {
                ids,
                final_active,
                done: done.read(),
                contributions,
                shared_out: shared_out.read(),
                x,
                shared_expert_gate: shared_gate,
                scale,
                y: TensorMut(y.read()),
            },
        )?;
        self.pending = false;
        Ok(completed)
    }

    fn run_cpu(&mut self, device: &Device, dispatch: &Dispatch<'_>) -> Result<()> {
        let hidden = self.geometry.hidden as usize;
        let source = self.host_x.as_slice()?;
        for (compacted, &row) in dispatch.miss_rows.iter().enumerate() {
            ensure!(
                (row as usize) < self.rows,
                "MoE CPU row={row} outside rows={}",
                self.rows
            );
            self.cpu_x[compacted * hidden..(compacted + 1) * hidden]
                .copy_from_slice(&source[row as usize * hidden..(row as usize + 1) * hidden]);
        }
        self.pool.run(Job {
            arena: dispatch
                .host
                .as_ref()
                .context("MoE CPU assignments have no host arena")?
                .clone(),
            hidden: &self.cpu_x[..dispatch.miss_rows.len() * hidden],
            rows: dispatch.miss_rows.len(),
            experts: dispatch.experts,
            assignments: dispatch.assignments,
            out: &mut self.cpu_out.as_mut_slice()?
                [..self.rows * self.geometry.top_k as usize * hidden],
        })?;
        device.ctx.check_err()?;
        Ok(())
    }

    pub fn trace(&self, stream: &Arc<CudaStream>, trace: &mut Trace, prefix: &str) -> Result<()> {
        let (rows, k, hidden) = (
            self.rows,
            self.geometry.top_k as usize,
            self.geometry.hidden as usize,
        );
        let (ids, weights) = self.routed()?;
        trace.add(
            format!("{prefix}.moe.ids"),
            vec![rows, k],
            ids.iter().map(|&v| v as f32).collect(),
        )?;
        trace.add(
            format!("{prefix}.moe.weights"),
            vec![rows, k],
            weights.iter().map(|v| v.to_f32()).collect(),
        )?;
        for (name, data, shape) in [
            ("contributions", &self.contributions, vec![rows, k, hidden]),
            ("shared", &self.shared, vec![rows, hidden]),
        ] {
            let len = shape.iter().product::<usize>();
            trace.add(
                format!("{prefix}.moe.{name}"),
                shape,
                stream
                    .clone_dtoh(&data.slice(..len))?
                    .into_iter()
                    .map(bf16::to_f32)
                    .collect(),
            )?;
        }
        trace.add(
            format!("{prefix}.moe.done"),
            vec![rows, k],
            stream
                .clone_dtoh(&self.done.slice(..rows * k))?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        )?;
        Ok(())
    }
}

fn validate_dispatch(
    rows: usize,
    top_k: usize,
    weights: &[bf16],
    dispatch: &Dispatch<'_>,
) -> Result<()> {
    ensure!(
        (1..=8).contains(&rows) && (1..=32).contains(&top_k),
        "MoE dispatch rows={rows} top_k={top_k} exceed decode bounds"
    );
    let count = rows * top_k;
    ensure!(
        dispatch.active.len() == count
            && dispatch.generation.len() == count
            && weights.len() == count,
        "MoE dispatch lengths active={} generation={} weights={} expected={count}",
        dispatch.active.len(),
        dispatch.generation.len(),
        weights.len()
    );
    ensure!(
        dispatch.miss_rows.len() <= rows
            && dispatch.miss_rows.iter().all(|&r| (r as usize) < rows)
            && dispatch.miss_rows.windows(2).all(|r| r[0] < r[1]),
        "MoE CPU compacted rows={:?} outside unique ascending0..{rows}",
        dispatch.miss_rows
    );
    ensure!(
        dispatch.assignments.is_empty() || dispatch.host.is_some(),
        "MoE CPU assignments have no host arena"
    );
    let mut coverage = [0u8; 8 * 32];
    for assignment in dispatch.assignments {
        let choice = assignment.choice as usize;
        ensure!(
            choice < count
                && (assignment.row as usize) < dispatch.miss_rows.len()
                && (assignment.expert as usize) < dispatch.experts.len(),
            "MoE CPU assignment choice={} row={} expert={} outside count={count} rows={} experts={}",
            assignment.choice,
            assignment.row,
            assignment.expert,
            dispatch.miss_rows.len(),
            dispatch.experts.len()
        );
        ensure!(
            dispatch.miss_rows[assignment.row as usize] as usize == choice / top_k,
            "MoE CPU compacted row={} does not own choice={choice}",
            assignment.row
        );
        ensure!(
            assignment.weight == weights[choice],
            "MoE CPU choice={choice} routing weight mismatch"
        );
        ensure!(
            coverage[choice] == 0,
            "MoE CPU choice={choice} has duplicate assignments"
        );
        coverage[choice] = 1;
    }
    for (choice, &active) in dispatch.active.iter().enumerate() {
        ensure!(
            active <= 1 && coverage[choice] == u8::from(active == 0),
            "MoE choice={choice} active={active} CPU coverage={} does not cover exactly once",
            coverage[choice]
        );
        ensure!(
            active == 0 || dispatch.generation[choice] > 0,
            "MoE GPU choice={choice} has generation=0"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "engine_moe_tests.rs"]
mod tests;
