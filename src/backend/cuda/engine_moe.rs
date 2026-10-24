//! Eager routed-FFN execution over caller-provided residency decisions.
use super::super::copy::pinned_zeros;
use super::*;
use crate::backend::cuda::{
    calibrate::host_memory,
    contracts::{Tensor, TensorMut, view},
    linear::BoundFfn,
    moe::{
        combine::{CombineOps, MoeCombine},
        decode::{DecodeOps, MoeDecode, decode_scratch_bytes, decode_scratch_bytes_mixed},
        prefill::{LANE16_ROWS, PrefillOps},
        route::RouteOps,
    },
};
use crate::cpu::expert_ffn::{Assignment, ExpertPool, ExpertWeights, Job, Shape};
use crate::experts::{arena::HostBlocks, catalogue::ExpertGeometry};
use cudarc::driver::{CudaEvent, DevicePtr, HostSlice, result};

#[path = "engine_moe_residency.rs"]
mod residency;
pub(super) use residency::{Prefill, Slots};
#[path = "engine_moe_graph.rs"]
mod graphs;

pub(super) struct Setup {
    pub workspace: crate::prefill::ExpertWorkspaceBudget,
    capacity: crate::experts::plan::Capacity,
    admission: crate::experts::plan::Admission,
    measured: super::super::calibrate::ExpertCalibration,
    catalogue: Arc<crate::experts::catalogue::ExpertCatalogue>,
    decode_rows: usize,
}

fn decode_device_bytes(
    geometry: ExpertGeometry,
    shared_intermediate: u32,
    rows: usize,
) -> Result<usize> {
    ensure!(
        (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows),
        "expert decode rows={rows} exceeds skinny bound={} geometry={geometry:?}",
        crate::linear_plan::MAX_SKINNY_ROWS
    );
    let hidden = geometry.hidden as usize;
    let k = geometry.top_k as usize;
    let row = hidden
        .checked_mul(2)
        .and_then(|n| n.checked_add(26))
        .and_then(|n| n.checked_mul(k))
        .and_then(|n| n.checked_add(hidden * 2 + 2))
        .and_then(|n| n.checked_add(geometry.experts as usize * 2))
        .context("expert decode row byte count overflow")?;
    let scratch = decode_scratch_bytes_mixed(
        rows,
        k,
        geometry.intermediate as usize,
        shared_intermediate as usize,
    );
    rows.checked_mul(row)
        .and_then(|n| n.checked_add(usize::try_from(scratch).ok()?))
        .context("expert decode device reserve overflow")
}

fn host_workspace(
    geometry: ExpertGeometry,
    span_rows: u32,
    workers: u32,
    copies: u32,
) -> Result<u64> {
    let (h, i, k) = (
        u64::from(geometry.hidden),
        u64::from(geometry.intermediate),
        u64::from(geometry.top_k),
    );
    let choices = (u64::from(span_rows) * k)
        .min(u64::from(geometry.experts) * u64::from(LANE16_ROWS))
        .max(1);
    let rows = u64::from(span_rows).min(choices);
    let workers = u64::from(workers);
    ensure!(workers > 0, "expert host workspace has zero workers");
    let metadata = (size_of::<ExpertWeights>() + size_of::<Assignment>()) as u64;
    let local = rows * h * 6
        + rows
        + choices * i * 2
        + i * 4
        + (choices * h.max(i)).div_ceil(workers) * 2
        + choices * metadata;
    let shared = rows * h * 2 + choices * (i + h) * 2 + choices * metadata;
    let mailboxes = rows * h * 8 + choices * (h * 4 + 16);
    let disk = u64::from(copies) * (rows * h * 2 + rows * k * (h * 2 + 16));
    (local * workers)
        .checked_add(shared)
        .and_then(|n| n.checked_add(mailboxes))
        .and_then(|n| n.checked_add(disk))
        .context("expert host workspace overflow")
}

impl Setup {
    pub fn new(
        device: &Device,
        model: &crate::model_spec::ModelSpec,
        manifest: &crate::checkpoint::TensorManifest,
        directory: &Path,
        runtime: &Path,
        write: bool,
        (request, decode_rows, verify_rows, context_row_bytes): (
            crate::prefill::WorkspaceRequest,
            usize,
            usize,
            usize,
        ),
    ) -> Result<Self> {
        use crate::experts::plan::PlanInputs;
        use crate::prefill::{ExpertWorkspaceBudget, ExpertWorkspaceRequest};
        let total_started = std::time::Instant::now();
        let phase_started = total_started;
        let geometry = ExpertGeometry::from_model(model)?;
        let shared_intermediate = model
            .moe
            .context("MoE geometry missing")?
            .shared_intermediate;
        let decode_extra = decode_device_bytes(geometry, shared_intermediate, decode_rows)?
            .checked_sub(decode_device_bytes(geometry, shared_intermediate, 1)?)
            .context("expert decode reserve underflow")?;
        let catalogue = Arc::new(crate::experts::catalogue::ExpertCatalogue::from_manifest(
            manifest,
            geometry,
            model.expert_format.context("MoE expert_format missing")?,
        )?);
        let catalogue_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let calibration = super::super::closure::Closure::new(
            write,
            format!(
                "calibrate the expert service table for {}",
                directory.display()
            ),
        );
        let measured = super::super::calibrate::load_experts(
            device,
            runtime,
            geometry,
            model.expert_format.context("MoE expert_format missing")?,
            &calibration,
        )?;
        let expert_calibration_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let available = host_memory(Path::new("/"))?;
        let copies = measured.copy_engines.min(2);
        let slot = catalogue.slot_bytes();
        let workers = measured.service.worker_count()?;
        let policy = crate::gdn_plan::GdnPolicy::for_device(&device.info);
        let vision = if model.vision.is_some() {
            crate::checkpoint::MappedCheckpoint::open(directory)?
                .manifest(model, crate::checkpoint::LoadPolicy::VisionOnly)?
                .resident_bytes
                .checked_add(request.total / 16)
                .context("expert vision reserve overflow")?
        } else {
            0
        };
        let mut candidates = Vec::new();
        let mut failures = Vec::new();
        let mut spans = (0..usize::BITS)
            .map(|shift| 1usize << shift)
            .take_while(|&rows| rows <= request.capacity)
            .chain([request.capacity])
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(rows) = request.requested {
            spans.retain(|&span| span >= rows);
        }
        let choice_row_alignment = match (geometry.top_k * 2) % 16 {
            0 => 1,
            8 => 2,
            4 | 12 => 4,
            2 | 6 | 10 | 14 => 8,
            _ => 16,
        };
        spans.retain(|&span| span >= choice_row_alignment);
        spans.retain(|&span| span >= decode_rows);
        for span in spans {
            let span_rows = u32::try_from(span)?;
            let mut wave_rows = 1u32;
            // L2 bounds activation scratch.
            while wave_rows <= span_rows / 2
                && crate::backend::cuda::moe::prefill::prefill_scratch_bytes_mixed(
                    wave_rows as usize * 2,
                    span_rows as usize,
                    geometry.top_k as usize,
                    geometry.experts as usize,
                    geometry.hidden as usize,
                    (geometry.intermediate as usize, shared_intermediate as usize),
                    None,
                )? <= device.info.l2_bytes as u64 / 2
            {
                wave_rows *= 2;
            }
            wave_rows = wave_rows.next_multiple_of(u32::try_from(choice_row_alignment)?);
            let scratch = usize::try_from(
                crate::backend::cuda::moe::prefill::prefill_scratch_bytes_mixed(
                    wave_rows as usize,
                    span_rows as usize,
                    geometry.top_k as usize,
                    geometry.experts as usize,
                    geometry.hidden as usize,
                    (geometry.intermediate as usize, shared_intermediate as usize),
                    catalogue.layout().fp8_slot(),
                )?,
            )?;
            let decode_scratch = usize::try_from(decode_scratch_bytes(
                1,
                geometry.top_k as usize,
                geometry.intermediate as usize,
            ))?;
            let fixed = vision
                .checked_add(scratch)
                .and_then(|n| {
                    n.checked_add(geometry.layers as usize * geometry.experts as usize * 28)
                })
                .and_then(|n| n.checked_add(geometry.hidden as usize * 2 + 2))
                .and_then(|n| n.checked_add(decode_scratch))
                .and_then(|n| n.checked_add(decode_extra))
                .and_then(|n| {
                    n.checked_add(super::super::moe::prefill::stage_device_bytes() as usize)
                })
                .context("expert fixed device reserve overflow")?;
            let host = host_workspace(geometry, span_rows, workers, copies)?
                .checked_add(super::super::moe::prefill::stage_host_pinned_bytes())
                .and_then(|n| n.checked_add(available / 16))
                .and_then(|n| n.checked_add(u64::from(copies) * slot * 5))
                .context("expert fixed host reserve overflow")?;
            let candidate = ExpertWorkspaceBudget::new(
                ExpertWorkspaceRequest {
                    mixer: crate::prefill::WorkspaceRequest {
                        fixed_transient_bytes: request
                            .fixed_transient_bytes
                            .checked_add(fixed)
                            .and_then(|bytes| {
                                context_row_bytes
                                    .checked_mul(span)
                                    .and_then(|context| bytes.checked_add(context))
                            })
                            .context("expert mixer fixed reserve overflow")?,
                        row_bytes: request
                            .row_bytes
                            .checked_sub(context_row_bytes)
                            .context("draft context row reserve exceeds combined row bytes")?,
                        ..request
                    },
                    span_rows,
                    wave_rows,
                    wave_workspace_bytes: 0,
                    device_headroom_bytes: request.total / 16,
                    host_fixed_bytes: host,
                    prefill_staging_slots: copies,
                    disk_ring_slots: copies,
                },
                geometry,
                slot,
                |rows| {
                    qwen::Runtime::workspace(
                        model,
                        rows.max(wave_rows as usize),
                        request.capacity,
                        policy,
                        verify_rows,
                    )
                },
                |rows| Ok(qwen::default_chunk(rows)),
            );
            match candidate {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => failures.push(format!("span={span}: {error}")),
            }
        }
        ensure!(
            !candidates.is_empty(),
            "no admitted expert workspace: {}",
            failures.join("; ")
        );
        let registration = measured.host_registration_bytes;
        let first = &candidates[0].candidate;
        let (index, capacity, admission) = ExpertWorkspaceBudget::select(
            &PlanInputs {
                geometry,
                layout: catalogue.layout().clone(),
                device_free_bytes: request.free_after_resident as u64,
                host_available_bytes: available,
                h2d_bytes_per_second: measured.h2d_bytes_per_second,
                d2h_bytes_per_second: measured.d2h_bytes_per_second,
                copy_engines: measured.copy_engines,
                disk_readers: copies,
                host_registration_bytes: registration,
                service: &measured.service,
                reserves: &first.reserves,
                span_rows: first.span_rows,
                wave_rows: first.wave_rows,
            },
            u32::try_from(request.capacity)?,
            &candidates,
        )?;
        let workspace = candidates.swap_remove(index);
        let cpu = residency::cpu_bound(
            geometry,
            capacity.slot_bytes(),
            capacity.span_rows,
            &measured.service,
            measured.h2d_bytes_per_second,
        )?;
        eprintln!(
            "{}",
            serde_json::json!({"expert_memory_plan": {
                "span_rows":capacity.span_rows,"wave_rows":capacity.wave_rows,
                "mixer_rows":workspace.mixer.plan.chunk,"workspace_rows":workspace.workspace_rows,
                "borrowed_slots":workspace.borrowed_slots,"slots_per_layer":capacity.slots_per_layer,
                "host_blocks":capacity.host_blocks,"spare_blocks":capacity.spare_blocks,
                "host_backing_complete":u64::from(capacity.host_blocks)==u64::from(geometry.layers)*u64::from(geometry.experts),
                "disk_bytes":capacity.disk_bytes,"pinned":capacity.pinned,
                "host_available_bytes":available,"host_registration_bytes":registration,
                "device_fixed_bytes":workspace.candidate.reserves.device_fixed_bytes,
                "host_fixed_bytes":workspace.candidate.reserves.host_fixed_bytes,
                "vision_reserved_bytes":vision,"budget_bytes_per_token":admission.budget_bytes_per_token,
                "cpu_rows_measured":cpu.3,"cpu_rows_limit":cpu.2,
                "cpu_arithmetic_rows":super::super::moe::prefill::LANE16_ROWS,
                "decode_rows":decode_rows,"decode_extra_device_bytes":decode_extra,
                "context_span_rows":capacity.span_rows,"context_span_row_bytes":context_row_bytes,
                "context_span_bytes":context_row_bytes.checked_mul(capacity.span_rows as usize).context("draft context span bytes overflow")?
            }})
        );
        eprintln!(
            "{}",
            serde_json::json!({"expert_setup_phase": {
                "catalogue_ms": catalogue_ms,
                "expert_calibration_ms": expert_calibration_ms,
                "planning_ms": phase_started.elapsed().as_secs_f64() * 1000.0,
                "total_ms": total_started.elapsed().as_secs_f64() * 1000.0
            }})
        );
        Ok(Self {
            workspace,
            capacity,
            admission,
            measured,
            catalogue,
            decode_rows,
        })
    }

    pub fn bind(self, engine: &mut Engine) -> Result<()> {
        let total_started = std::time::Instant::now();
        let geometry = self.catalogue.geometry();
        let initial = crate::experts::plan::initial_residency(
            &vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize],
            &self.capacity.slots_per_layer,
            &geometry,
        )?;
        let cpu = residency::cpu_bound(
            geometry,
            self.capacity.slot_bytes(),
            self.capacity.span_rows,
            &self.measured.service,
            self.measured.h2d_bytes_per_second,
        )?;
        let phase_started = std::time::Instant::now();
        let mut slots = Slots::load(
            &engine.device,
            geometry,
            residency::Load {
                capacity: &self.capacity,
                admission: self.admission,
                initial: &initial,
                directory: engine
                    .qwen
                    .as_ref()
                    .context("expert hybrid runtime missing")?
                    .checkpoint_path(),
                catalogue: self.catalogue,
                engines: self.measured.copy_engines,
                max_rows: u32::try_from(cpu.0.max(self.decode_rows))?,
                swiglu_rounding: engine.model_spec.swiglu_rounding(),
                routed_weight: engine.model_spec.routed_weight_boundary(),
            },
        )?;
        let slots_load_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        engine.moe = Some(Decode::new(
            &engine.device,
            DecodeOptions {
                geometry,
                max_rows: self.decode_rows,
                workers: self.capacity.worker_count,
                cpu_bounds: (
                    cpu.0.max(self.decode_rows),
                    cpu.1.max(self.decode_rows * geometry.top_k as usize),
                ),
                format: engine
                    .model_spec
                    .expert_format
                    .context("MoE expert_format missing")?,
                shared_intermediate: engine
                    .model_spec
                    .moe
                    .context("MoE shared geometry missing")?
                    .shared_intermediate,
                swiglu_rounding: engine.model_spec.swiglu_rounding(),
                routed_weight: engine.model_spec.routed_weight_boundary(),
            },
        )?);
        let decode_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        engine.moe_prefill = Some(Prefill::new(
            &engine.device,
            &mut slots,
            &self.capacity,
            self.workspace.candidate.reserves,
            self.measured.service,
            self.measured.h2d_bytes_per_second,
            (
                engine
                    .model_spec
                    .moe
                    .context("MoE shared geometry missing")?
                    .shared_intermediate,
                self.workspace.workspace_rows,
                self.workspace.mixer.plan.chunk,
            ),
        )?);
        let prefill_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        engine.device.finish_upload()?;
        engine.moe_slots = Some(slots);
        eprintln!(
            "{}",
            serde_json::json!({"expert_bind_phase": {
                "slots_load_ms": slots_load_ms,
                "decode_new_ms": decode_new_ms,
                "prefill_new_ms": prefill_new_ms,
                "total_ms": total_started.elapsed().as_secs_f64() * 1000.0
            }})
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct SlotBindings {
    pub gate_up: Tensor<u64, 1>,
    pub down: Tensor<u64, 1>,
    pub valid: Tensor<u32, 1>,
    pub generation: Tensor<u64, 1>,
}

impl Engine {
    pub(super) fn bind_invariant_arithmetic(&mut self) -> Result<()> {
        self.closure.bind_fixed()?;
        let executor = self
            .moe
            .as_mut()
            .context("invariant target arithmetic requires MoE")?;
        ensure!(
            executor.geometry.hidden.is_multiple_of(256)
                && executor.geometry.intermediate.is_multiple_of(256)
                && u64::from(executor.geometry.hidden)
                    >= 2 * u64::from(executor.geometry.intermediate),
            "invariant shared geometry hidden={} inter={}",
            executor.geometry.hidden,
            executor.geometry.intermediate
        );
        self.blas
            .bind_invariant(&self.device.ctx, crate::linear_plan::MAX_SKINNY_ROWS)?;
        executor.contribute = DecodeOps::new_invariant_with_boundaries(
            &self.device.ctx,
            self.model_spec.swiglu_rounding(),
            self.model_spec.routed_weight_boundary(),
        )?;
        eprintln!(
            "draft_target_arithmetic lane16_routed=true skinny_shared=true skinny_typed=true rows={}",
            crate::linear_plan::MAX_SKINNY_ROWS
        );
        Ok(())
    }

    pub(super) fn run_moe_ffn(
        &mut self,
        layer: usize,
        rows: usize,
        ffn: BoundFfn,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        let slot_layer = u32::try_from(self.moe_slot_layer(layer)?)?;
        let decode = self
            .moe
            .as_mut()
            .context("MoE decode executor is not bound")?;
        let slots = self
            .moe_slots
            .as_mut()
            .context("MoE physical slot tables are not bound")?;
        let stream = &self.device.stream;
        let hidden = self.model_spec.hidden as usize;
        let x = qwen::matrix(&self.work.n, stream, 0, rows, hidden)?;
        slots.run(
            &self.device,
            decode,
            slot_layer,
            ffn,
            x,
            &mut TensorMut(qwen::matrix(&self.work.out, stream, 0, rows, hidden)?),
        )?;
        if let Some(target) = trace
            .as_mut()
            .filter(|target| target.trace.all_layers || layer == self.trace_layer)
        {
            let (prefix, name) = if target.trace.all_layers {
                (format!("{}.{layer}", target.prefix), format!("{layer}.ffn"))
            } else {
                (target.prefix.to_owned(), "ffn".to_owned())
            };
            decode.trace(stream, target.trace, &prefix)?;
            trace_value(trace, stream, &name, &self.work.out, rows, hidden)?;
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
    pub host_dma: Option<&'a residency::Transfers>,
    pub disk: &'a [crate::experts::runtime::Route],
    pub disk_cpu: Option<&'a mut crate::experts::disk::DiskExperts>,
    pub layer: u32,
}

pub(super) struct Decode {
    geometry: ExpertGeometry,
    shared_intermediate: u32,
    fp8_slot: Option<crate::experts::catalogue::Fp8Slot>,
    max_rows: usize,
    phase: DecodePhase,
    route: RouteOps,
    contribute: DecodeOps,
    prefill: PrefillOps,
    combine: CombineOps,
    ids: CudaSlice<i32>,
    weights: CudaSlice<bf16>,
    route_logits: CudaSlice<bf16>,
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
    mailbox: Arc<CudaStream>,
    segments: Vec<graphs::Segments>,
    verify_segments: std::collections::BTreeMap<usize, Vec<graphs::Segments>>,
    mailbox_ready: CudaEvent,
    reader: Arc<CudaEvent>,
    mask: cudarc::driver::CudaFunction,
    pub stats: [u64; 5],
}

#[derive(Default)]
struct DecodePhase {
    rows: usize,
    pending: bool,
    input_ptr: u64,
    router_ptr: u64,
}

impl DecodePhase {
    fn begin(&mut self, rows: usize, input_ptr: u64, router_ptr: u64) -> Result<()> {
        ensure!(
            !self.pending && rows > 0,
            "MoE route phase is not idle or has zero rows"
        );
        self.rows = rows;
        self.input_ptr = input_ptr;
        self.router_ptr = router_ptr;
        self.pending = true;
        Ok(())
    }

    fn check(&self, input_ptr: u64, router_ptr: u64) -> Result<()> {
        ensure!(
            self.pending && input_ptr == self.input_ptr && router_ptr == self.router_ptr,
            "MoE completion phase/input mismatch pending={} input={input_ptr:#x} routed_input={:#x} router={router_ptr:#x} routed_router={:#x}",
            self.pending,
            self.input_ptr,
            self.router_ptr
        );
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        ensure!(self.pending, "MoE completion has no pending route");
        self.pending = false;
        Ok(())
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Drop for Decode {
    fn drop(&mut self) {
        self.mailbox
            .context()
            .record_err(self.mailbox.synchronize());
    }
}

pub struct DecodeOptions {
    pub geometry: ExpertGeometry,
    pub max_rows: usize,
    pub workers: u32,
    pub cpu_bounds: (usize, usize),
    pub format: crate::model_spec::ExpertFormat,
    pub shared_intermediate: u32,
    pub swiglu_rounding: crate::model_spec::SwigluRounding,
    pub routed_weight: crate::model_spec::RoutedWeightBoundary,
}

impl Decode {
    pub fn new(device: &Device, options: DecodeOptions) -> Result<Self> {
        let DecodeOptions {
            geometry,
            max_rows,
            workers,
            cpu_bounds,
            format,
            shared_intermediate,
            swiglu_rounding,
            routed_weight,
        } = options;
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&max_rows),
            "MoE decode row bound={max_rows} exceeds skinny bound={}",
            crate::linear_plan::MAX_SKINNY_ROWS
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
        ensure!(
            cpu_bounds.0 >= max_rows && cpu_bounds.1 >= choices,
            "MoE CPU bounds={cpu_bounds:?} below decode rows={max_rows} choices={choices}"
        );
        let stream = &device.stream;
        let context = &device.ctx;
        let layout = crate::experts::catalogue::ExpertLayout::new(format, &geometry)?;
        let decode = Self {
            geometry,
            shared_intermediate,
            fp8_slot: layout.fp8_slot().cloned(),
            max_rows,
            phase: DecodePhase::default(),
            route: RouteOps::new(context, stream)?,
            contribute: DecodeOps::new_with_boundaries(context, swiglu_rounding, routed_weight)?,
            prefill: PrefillOps::new_with_swiglu_rounding(
                context,
                stream,
                format,
                swiglu_rounding,
            )?,
            combine: CombineOps::new(context)?,
            ids: stream.alloc_zeros(choices)?,
            weights: stream.alloc_zeros(choices)?,
            route_logits: stream.alloc_zeros(max_rows * geometry.experts as usize)?,
            active: stream.alloc_zeros(choices)?,
            final_active: stream.clone_htod(&vec![1u32; choices])?,
            generation: stream.alloc_zeros(choices)?,
            contributions: stream.alloc_zeros(elements)?,
            done: stream.alloc_zeros(choices)?,
            shared: stream.alloc_zeros(hidden)?,
            scale: stream.alloc_zeros(max_rows)?,
            scratch: stream.alloc_zeros(usize::try_from(decode_scratch_bytes_mixed(
                max_rows,
                geometry.top_k as usize,
                geometry.intermediate as usize,
                shared_intermediate as usize,
            ))?)?,
            host_ids: pinned_zeros(context, choices)?,
            host_weights: pinned_zeros(context, choices)?,
            host_active: pinned_zeros(context, choices)?,
            host_generation: pinned_zeros(context, choices)?,
            host_x: pinned_zeros(context, hidden)?,
            cpu_x: vec![bf16::ZERO; hidden],
            cpu_out: pinned_zeros(context, elements)?,
            cpu_done: pinned_zeros(context, choices)?,
            mailbox: context.new_stream()?,
            segments: Vec::new(),
            verify_segments: std::collections::BTreeMap::new(),
            mailbox_ready: context.new_event(None)?,
            reader: Arc::new(context.new_event(None)?),
            mask: super::super::cubin::module(context, "moe_route")?
                .load_function("moe_route_mask")?,
            pool: ExpertPool::new_with_boundaries(
                workers as usize,
                Shape::new(geometry.hidden as usize, geometry.intermediate as usize)?,
                cpu_bounds.0,
                cpu_bounds.1,
                layout.fp8_slot().cloned(),
                swiglu_rounding,
                routed_weight,
            )?,
            stats: [0; 5],
        };
        let allocated = (decode.ids.len()
            + decode.active.len()
            + decode.final_active.len()
            + decode.done.len())
            * size_of::<u32>()
            + (decode.weights.len() + decode.route_logits.len()) * size_of::<bf16>()
            + decode.generation.len() * size_of::<u64>()
            + (decode.contributions.len() + decode.shared.len() + decode.scale.len())
                * size_of::<bf16>()
            + decode.scratch.len();
        let planned = decode_device_bytes(geometry, shared_intermediate, max_rows)?;
        ensure!(
            allocated == planned,
            "MoE decode row buffers allocation={allocated} differs from planned={planned} rows={max_rows}"
        );
        Ok(decode)
    }

    pub fn route(&mut self, device: &Device, ffn: &BoundFfn, x: Tensor<bf16, 2>) -> Result<()> {
        self.route_device(device, ffn, x)?;
        self.read_route(device, ffn, x)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.mailbox.synchronize()?;
        self.phase.reset();
        Ok(())
    }

    fn route_device(&self, device: &Device, ffn: &BoundFfn, x: Tensor<bf16, 2>) -> Result<()> {
        let BoundFfn::Moe { router, score, .. } = ffn else {
            anyhow::bail!("MoE route received a dense FFN");
        };
        ensure!(
            !self.phase.pending,
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
            x,
            *router,
            &mut TensorMut(view(
                &self.ids,
                stream,
                0,
                [rows as u32, k],
                [u64::from(k), 1],
            )?),
            &mut TensorMut(view(
                &self.weights,
                stream,
                0,
                [rows as u32, k],
                [u64::from(k), 1],
            )?),
            &mut TensorMut(view(
                &self.route_logits,
                stream,
                0,
                [rows as u32, self.geometry.experts],
                [u64::from(self.geometry.experts), 1],
            )?),
            *score,
        )
    }

    fn read_route(&mut self, device: &Device, ffn: &BoundFfn, x: Tensor<bf16, 2>) -> Result<()> {
        let BoundFfn::Moe { router, .. } = ffn else {
            anyhow::bail!("MoE route mailbox received a dense FFN");
        };
        let rows = x.shape[0] as usize;
        ensure!(
            !self.phase.pending
                && rows > 0
                && rows <= self.max_rows
                && x.shape[1] == self.geometry.hidden,
            "MoE route mailbox pending={} shape={:?} row_bound={}",
            self.phase.pending,
            x.shape,
            self.max_rows
        );
        let k = self.geometry.top_k;
        let stream = &device.stream;
        self.reader.record(stream)?;
        self.mailbox.wait(&self.reader)?;
        let count = rows * k as usize;
        {
            let (ids, _ids) = self.ids.device_ptr(&self.mailbox);
            let (weights, _weights) = self.weights.device_ptr(&self.mailbox);
            let (host_ids, _ids_write) =
                unsafe { self.host_ids.stream_synced_mut_slice(&self.mailbox) };
            let (host_weights, _weights_write) =
                unsafe { self.host_weights.stream_synced_mut_slice(&self.mailbox) };
            let (host_x, _x_write) = unsafe { self.host_x.stream_synced_mut_slice(&self.mailbox) };
            // SAFETY: each copied prefix has its exact logical extent inside the owned mailbox.
            unsafe {
                result::memcpy_dtoh_async(&mut host_ids[..count], ids, self.mailbox.cu_stream())?;
                result::memcpy_dtoh_async(
                    &mut host_weights[..count],
                    weights,
                    self.mailbox.cu_stream(),
                )?;
                result::memcpy_dtoh_async(
                    &mut host_x[..rows * self.geometry.hidden as usize],
                    x.ptr,
                    self.mailbox.cu_stream(),
                )?;
            }
        }
        self.mailbox.synchronize()?;
        self.phase.begin(rows, x.ptr, router.ptr)?;
        self.stats[0] += 1;
        Ok(())
    }

    pub fn routed(&self) -> Result<(&[i32], &[bf16])> {
        let count = self.phase.rows * self.geometry.top_k as usize;
        Ok((
            &self.host_ids.as_slice()?[..count],
            &self.host_weights.as_slice()?[..count],
        ))
    }

    #[cfg(test)]
    pub fn finish(
        &mut self,
        device: &Device,
        ffn: BoundFfn,
        slots: SlotBindings,
        dispatch: Dispatch<'_>,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
    ) -> Result<()> {
        self.finish_with(device, ffn, slots, dispatch, activation, None)
    }

    fn segment(&self, rows: usize, layer: usize) -> Result<&graphs::Segments> {
        let layers = if rows == 1 {
            &self.segments
        } else {
            self.verify_segments
                .get(&rows)
                .with_context(|| format!("MoE verify graph width={rows} missing"))?
        };
        layers
            .get(layer)
            .with_context(|| format!("MoE graph segment missing rows={rows} layer={layer}"))
    }

    fn finish_with(
        &mut self,
        device: &Device,
        ffn: BoundFfn,
        slots: SlotBindings,
        mut dispatch: Dispatch<'_>,
        activation: (Tensor<bf16, 2>, &mut TensorMut<bf16, 2>),
        graph: Option<usize>,
    ) -> Result<()> {
        let BoundFfn::Moe {
            router, ref shared, ..
        } = ffn
        else {
            anyhow::bail!("MoE completion received a dense FFN");
        };
        let (x, y) = activation;
        self.phase.check(x.ptr, router.ptr)?;
        let (rows, k, hidden) = (
            self.phase.rows,
            self.geometry.top_k as usize,
            self.geometry.hidden as usize,
        );
        ensure!(
            rows > 0 && x.shape == [rows as u32, hidden as u32],
            "MoE completion shape={:?} routed_rows={rows}",
            x.shape
        );
        ensure!(
            shared[0].shape[0] == self.shared_intermediate,
            "MoE shared intermediate={} expected={}",
            shared[0].shape[0],
            self.shared_intermediate
        );
        validate_dispatch(rows, k, self.routed()?.1, &dispatch)?;
        let stream = &device.stream;
        self.host_active.as_mut_slice()?[..rows * k].copy_from_slice(dispatch.active);
        self.host_generation.as_mut_slice()?[..rows * k].copy_from_slice(dispatch.generation);
        {
            let (active, _active_read) =
                unsafe { self.host_active.stream_synced_slice(&self.mailbox) };
            let (generation, _generation_read) =
                unsafe { self.host_generation.stream_synced_slice(&self.mailbox) };
            let (active_ptr, _active_write) = self.active.device_ptr_mut(&self.mailbox);
            let (generation_ptr, _generation_write) = self.generation.device_ptr_mut(&self.mailbox);
            // SAFETY: owned pinned prefixes and device destinations each contain rows*top_k elements.
            unsafe {
                result::memcpy_htod_async(
                    active_ptr,
                    &active[..rows * k],
                    self.mailbox.cu_stream(),
                )?;
                result::memcpy_htod_async(
                    generation_ptr,
                    &generation[..rows * k],
                    self.mailbox.cu_stream(),
                )?;
            }
        }
        self.mailbox_ready.record(&self.mailbox)?;
        stream.wait(&self.mailbox_ready)?;
        if let Some(layer) = graph {
            self.segment(rows, layer)?.contribute.launch()?;
        } else {
            self.contribute_device(device, ffn.clone(), slots, x)?;
        }
        self.reader.record(stream)?;
        let reader = self.reader.clone();
        self.upload_cpu(device, &mut dispatch, &reader)?;
        if let Some(layer) = graph {
            self.segment(rows, layer)?.join.launch()?;
        } else {
            self.combine_device(device, ffn, x, y)?;
        }
        self.stats[1] += dispatch.active.iter().map(|&n| u64::from(n)).sum::<u64>();
        self.stats[4] += 1;
        self.phase.finish()?;
        Ok(())
    }

    fn contribute_device(
        &mut self,
        device: &Device,
        ffn: BoundFfn,
        slots: SlotBindings,
        x: Tensor<bf16, 2>,
    ) -> Result<()> {
        let BoundFfn::Moe {
            ref shared,
            shared_gate,
            ..
        } = ffn
        else {
            anyhow::bail!("MoE contribution received a dense FFN");
        };
        let (rows, k, hidden) = (
            x.shape[0] as usize,
            self.geometry.top_k as usize,
            self.geometry.hidden as usize,
        );
        let stream = &device.stream;
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
        let ids = view(&self.ids, stream, 0, [rows as u32, k as u32], [k as u64, 1])?;
        let active = view(
            &self.active,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        let contributions = view(
            &self.contributions,
            stream,
            0,
            [rows as u32, k as u32, hidden as u32],
            [(k * hidden) as u64, hidden as u64, 1],
        )?;
        let mut shared_out = TensorMut(qwen::matrix(&self.shared, stream, 0, rows, hidden)?);
        let mut done = TensorMut(view(
            &self.done,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?);
        let input = MoeDecode {
            x,
            ids,
            active,
            weights: view(
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
            assignment_gen: view(
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
        };
        let scratch = qwen::scratch(&self.scratch, stream);
        if let Some(slot) = self.fp8_slot.as_ref() {
            self.contribute.contribute_w8a16(
                stream,
                &input,
                slot,
                &scratch,
                self.geometry.intermediate,
            )
        } else {
            self.contribute
                .contribute(stream, &input, &scratch, self.geometry.intermediate)
        }
    }

    fn upload_cpu(
        &mut self,
        device: &Device,
        dispatch: &mut Dispatch<'_>,
        reader: &CudaEvent,
    ) -> Result<()> {
        let (rows, k, hidden) = (
            self.phase.rows,
            self.geometry.top_k as usize,
            self.geometry.hidden as usize,
        );
        let stream = &device.stream;
        let contributions = view(
            &self.contributions,
            stream,
            0,
            [rows as u32, k as u32, hidden as u32],
            [(k * hidden) as u64, hidden as u64, 1],
        )?;
        let done = view(
            &self.done,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        if !dispatch.assignments.is_empty() || !dispatch.disk.is_empty() {
            self.run_cpu(device, dispatch)?;
            self.mailbox.wait(reader)?;
            let values = self.cpu_out.as_slice()?;
            let done_values = self.cpu_done.as_mut_slice()?;
            for choice in dispatch
                .assignments
                .iter()
                .map(|a| a.choice)
                .chain(dispatch.disk.iter().map(|a| a.choice))
            {
                let choice = choice as usize;
                let first = choice * hidden;
                done_values[choice] = 1;
                // SAFETY: each validated CPU choice owns its contribution and done word; mailboxes live through publication.
                unsafe {
                    result::memcpy_htod_async(
                        contributions.ptr + (first * 2) as u64,
                        &values[first..first + hidden],
                        self.mailbox.cu_stream(),
                    )?;
                    result::memcpy_htod_async(
                        done.ptr + (choice * 4) as u64,
                        &done_values[choice..choice + 1],
                        self.mailbox.cu_stream(),
                    )?;
                }
            }
            self.mailbox_ready.record(&self.mailbox)?;
            stream.wait(&self.mailbox_ready)?;
            self.stats[2] += (dispatch.assignments.len() + dispatch.disk.len()) as u64;
            self.stats[3] += 1;
        }
        Ok(())
    }

    fn combine_device(
        &self,
        device: &Device,
        ffn: BoundFfn,
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let BoundFfn::Moe { shared_gate, .. } = ffn else {
            anyhow::bail!("MoE combine received a dense FFN");
        };
        let (rows, k, hidden) = (
            x.shape[0] as usize,
            self.geometry.top_k as usize,
            self.geometry.hidden as usize,
        );
        let stream = &device.stream;
        let ids = view(&self.ids, stream, 0, [rows as u32, k as u32], [k as u64, 1])?;
        let contributions = view(
            &self.contributions,
            stream,
            0,
            [rows as u32, k as u32, hidden as u32],
            [(k * hidden) as u64, hidden as u64, 1],
        )?;
        let shared_out = qwen::matrix(&self.shared, stream, 0, rows, hidden)?;
        let done = view(
            &self.done,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        let mut gate_view = None;
        let mut scale = None;
        if let Some(gate) = shared_gate {
            let mut written = TensorMut(view(&self.scale, stream, 0, [rows as u32, 1], [1, 1])?);
            self.combine.shared_scale(stream, x, gate, &mut written)?;
            gate_view = Some(gate);
            scale = Some(written);
        }
        let final_active = view(
            &self.final_active,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        self.combine.combine(
            stream,
            &MoeCombine {
                ids,
                final_active,
                done,
                contributions,
                shared_out,
                x,
                shared_expert_gate: gate_view,
                scale,
                y: TensorMut(y.read()),
            },
        )?;
        Ok(())
    }

    fn run_cpu(&mut self, device: &Device, dispatch: &mut Dispatch<'_>) -> Result<()> {
        let hidden = self.geometry.hidden as usize;
        let source = self.host_x.as_slice()?;
        for (compacted, &row) in dispatch.miss_rows.iter().enumerate() {
            ensure!(
                (row as usize) < self.phase.rows,
                "MoE CPU row={row} outside rows={}",
                self.phase.rows
            );
            self.cpu_x[compacted * hidden..(compacted + 1) * hidden]
                .copy_from_slice(&source[row as usize * hidden..(row as usize + 1) * hidden]);
        }
        let cpu_hidden = &self.cpu_x[..dispatch.miss_rows.len() * hidden];
        let output = &mut self.cpu_out.as_mut_slice()?
            [..self.phase.rows * self.geometry.top_k as usize * hidden];
        let contribute = |out: &mut [bf16]| -> Result<()> {
            if !dispatch.assignments.is_empty() {
                if let Some(dma) = dispatch.host_dma {
                    dma.ready_weights(dispatch.experts)?;
                }
                self.pool.run(Job {
                    arena: dispatch
                        .host
                        .as_ref()
                        .context("MoE CPU assignments have no host arena")?
                        .clone(),
                    hidden: cpu_hidden,
                    rows: dispatch.miss_rows.len(),
                    experts: dispatch.experts,
                    assignments: dispatch.assignments,
                    out,
                })?;
            }
            Ok(())
        };
        if dispatch.disk.is_empty() {
            contribute(output)?;
        } else {
            dispatch
                .disk_cpu
                .as_deref_mut()
                .context("disk routes have no bound CPU workers")?
                .run(
                    dispatch.layer,
                    cpu_hidden,
                    dispatch.miss_rows.len(),
                    dispatch.disk,
                    output,
                    contribute,
                )?;
        }
        device.ctx.check_err()?;
        Ok(())
    }

    pub fn trace(&self, stream: &Arc<CudaStream>, trace: &mut Trace, prefix: &str) -> Result<()> {
        let (rows, k, hidden) = (
            self.phase.rows,
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
        (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows) && (1..=32).contains(&top_k),
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
    ensure!(
        dispatch.disk.is_empty() || dispatch.disk_cpu.is_some(),
        "disk CPU assignments have no bound worker service"
    );
    let mut coverage = [0u8; crate::linear_plan::MAX_SKINNY_ROWS * 32];
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
    for route in dispatch.disk {
        let choice = route.choice as usize;
        ensure!(
            choice < count && (route.row as usize) < dispatch.miss_rows.len(),
            "disk route choice={} row={} outside count={count} rows={}",
            route.choice,
            route.row,
            dispatch.miss_rows.len()
        );
        ensure!(
            dispatch.miss_rows[route.row as usize] as usize == choice / top_k
                && route.weight == weights[choice]
                && coverage[choice] == 0,
            "disk route choice={choice} compact row/weight/coverage differs"
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

#[cfg(test)]
#[path = "engine_moe_e2e.rs"]
mod e2e;
