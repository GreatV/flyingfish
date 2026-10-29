//! Eager routed-FFN execution over caller-provided residency decisions.
use super::super::copy::pinned_zeros;
use super::*;
use crate::backend::cuda::{
    calibrate::host_memory,
    contracts::{Tensor, TensorMut, view},
    linear::BoundFfn,
    moe::{
        combine::{CombineOps, MoeCombineResidual, MoeCombineRouted, MoeJoinRoutedResidual},
        decode::{DecodeOps, MoeDecode, decode_scratch_bytes, decode_scratch_bytes_mixed},
        prefill::{LANE16_ROWS, PrefillOps},
        route::RouteOps,
    },
};
use crate::backend::gpu::{CudaEvent, CudaFunction, DevicePtr};
use crate::cpu::expert_ffn::{Assignment, ExpertPool, ExpertWeights, Job, Shape};
use crate::experts::{arena::HostBlocks, catalogue::ExpertGeometry, plan::PeerSendLayout};
#[cfg(feature = "cuda")]
use cudarc::driver::HostSlice;

#[path = "engine_moe_residency.rs"]
mod residency;
pub(super) use residency::{Prefill, Slots};
#[path = "engine_moe_graph.rs"]
mod graphs;

pub(super) struct Setup {
    pub workspace: crate::prefill::ExpertWorkspaceBudget,
    directory: PathBuf,
    capacity: crate::experts::plan::Capacity,
    admission: crate::experts::plan::Admission,
    measured: super::super::calibrate::ExpertCalibration,
    catalogue: Arc<crate::experts::catalogue::ExpertCatalogue>,
    decode_rows: usize,
    routed_wave_rows: u32,
    peer_wave_rows: u32,
    peer: Option<Device>,
    non_expert_decode_bytes: u64,
    cold_schedule: Option<crate::dspark::VerifyCostSchedule>,
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
        .and_then(|n| n.checked_add(hidden * 2 + size_of::<u32>()))
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

fn routed_wave_rows(
    geometry: ExpertGeometry,
    capacity: &crate::experts::plan::Capacity,
    workspace_rows: usize,
    shared: (u32, crate::model_spec::RoutedWeightBoundary),
) -> Result<(u32, u64)> {
    use super::super::moe::prefill::{W8A16_TILE_ROWS, prefill_scratch_bytes_mixed_with_boundary};
    let Some(fp8) = capacity.layout.fp8_slot() else {
        return Ok((capacity.wave_rows, 0));
    };
    let slots = u64::from(geometry.layers) * u64::from(geometry.experts);
    if capacity.gpu0_layers != geometry.layers || u64::from(capacity.slots_total()) != slots {
        return Ok((capacity.wave_rows, 0));
    }
    ensure!(
        geometry.top_k > 0
            && capacity.wave_rows > 0
            && capacity.wave_rows <= capacity.span_rows
            && capacity.wave_rows as usize <= workspace_rows,
        "invalid resident routed wave geometry"
    );
    let available = capacity
        .device_expert_bytes
        .checked_sub(
            slots
                .checked_mul(capacity.slot_bytes())
                .context("resident expert bytes overflow")?,
        )
        .context("resident expert slots exceed device budget")?;
    let target = u64::from(geometry.experts)
        .checked_mul(u64::from(W8A16_TILE_ROWS))
        .context("resident routed tile extent overflow")?
        .div_ceil(u64::from(geometry.top_k))
        .max(u64::from(capacity.wave_rows))
        .min(u64::from(capacity.span_rows))
        .min(u64::try_from(workspace_rows)?);
    let alignment = 8u32 >> geometry.top_k.trailing_zeros().min(3);
    ensure!(
        capacity.wave_rows.is_multiple_of(alignment),
        "resident routed wave is not BF16 pointer aligned"
    );
    let scratch = |rows: u32| {
        prefill_scratch_bytes_mixed_with_boundary(
            rows as usize,
            capacity.span_rows as usize,
            geometry.top_k as usize,
            geometry.experts as usize,
            geometry.hidden as usize,
            (geometry.intermediate as usize, shared.0 as usize),
            (Some(fp8), shared.1),
        )
    };
    let baseline = scratch(capacity.wave_rows)?;
    let (mut low, mut high) = (
        capacity.wave_rows / alignment,
        u32::try_from(target)? / alignment,
    );
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let extra = scratch(mid * alignment)?
            .checked_sub(baseline)
            .context("resident routed scratch is not monotone")?;
        if extra <= available {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let rows = low * alignment;
    Ok((rows, scratch(rows)? - baseline))
}

fn peer_routed_wave_rows(
    geometry: ExpertGeometry,
    capacity: &crate::experts::plan::Capacity,
    shared: (u32, crate::model_spec::RoutedWeightBoundary),
) -> Result<(u32, u64)> {
    use super::super::moe::prefill::W8A16_TILE_ROWS;
    let Some(fp8) = capacity.layout.fp8_slot() else {
        return Ok((capacity.wave_rows, 0));
    };
    if capacity.gpu0_layers >= geometry.layers {
        return Ok((capacity.wave_rows, 0));
    }
    ensure!(
        geometry.top_k > 0 && capacity.wave_rows > 0 && capacity.wave_rows <= capacity.span_rows,
        "invalid peer routed wave geometry"
    );
    let available = capacity
        .peer_expert_bytes
        .checked_sub(
            u64::from(capacity.peer_slots())
                .checked_mul(capacity.slot_bytes())
                .context("peer expert bytes overflow")?,
        )
        .context("peer expert slots exceed peer budget")?;
    let target = u64::from(geometry.experts)
        .checked_mul(u64::from(W8A16_TILE_ROWS))
        .context("peer routed tile extent overflow")?
        .div_ceil(u64::from(geometry.top_k))
        .max(u64::from(capacity.wave_rows))
        .min(u64::from(capacity.span_rows));
    let alignment = 8u32 >> geometry.top_k.trailing_zeros().min(3);
    ensure!(
        capacity.wave_rows.is_multiple_of(alignment),
        "peer routed wave is not BF16 pointer aligned"
    );
    let bytes = |rows: u32| -> Result<u64> {
        Ok(u64::try_from(residency::peer_prefill_bytes(
            geometry,
            capacity.span_rows,
            rows,
            shared,
            Some(fp8),
        )?)?)
    };
    let baseline = bytes(capacity.wave_rows)?;
    let (mut low, mut high) = (
        capacity.wave_rows / alignment,
        u32::try_from(target)? / alignment,
    );
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let extra = bytes(mid * alignment)?
            .checked_sub(baseline)
            .context("peer routed scratch is not monotone")?;
        if extra <= available {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let rows = low * alignment;
    Ok((rows, bytes(rows)? - baseline))
}

impl Setup {
    pub fn new(
        device: &Device,
        model: &crate::model_spec::ModelSpec,
        manifest: &crate::checkpoint::TensorManifest,
        directory: &Path,
        runtime: &Path,
        write: bool,
        (request, decode_rows, verify_rows, context_row_bytes, draft_config): (
            crate::prefill::WorkspaceRequest,
            usize,
            usize,
            usize,
            Option<&crate::dspark::DraftConfig>,
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
            if draft_config.is_some() {
                crate::cpu::expert_ffn::Form::Lane16
            } else {
                crate::cpu::expert_ffn::Form::Lane8
            },
            &calibration,
        )?;
        let expert_calibration_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let available = host_memory(Path::new("/"))?;
        #[cfg(test)]
        let (available, request) = tests::MEMORY_BUDGET.with(|budget| -> Result<_> {
            let Some((device_bytes, host_bytes)) = budget.get() else {
                return Ok((available, request));
            };
            ensure!(
                device_bytes <= request.free_after_resident && host_bytes <= available,
                "residency fixture budget exceeds available memory"
            );
            Ok((
                host_bytes,
                crate::prefill::WorkspaceRequest {
                    free_after_resident: device_bytes,
                    ..request
                },
            ))
        })?;
        let copies = measured.copy_engines.min(2);
        let slot = catalogue.slot_bytes();
        let workers = measured.service.pool_worker_count()?;
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
                && crate::backend::cuda::moe::prefill::prefill_scratch_bytes_mixed_with_boundary(
                    wave_rows as usize * 2,
                    span_rows as usize,
                    geometry.top_k as usize,
                    geometry.experts as usize,
                    geometry.hidden as usize,
                    (geometry.intermediate as usize, shared_intermediate as usize),
                    (None, model.routed_weight_boundary()),
                )? <= device.info.l2_bytes as u64 / 2
            {
                wave_rows *= 2;
            }
            wave_rows = wave_rows.next_multiple_of(u32::try_from(choice_row_alignment)?);
            let scratch = usize::try_from(
                crate::backend::cuda::moe::prefill::prefill_scratch_bytes_mixed_with_boundary(
                    wave_rows as usize,
                    span_rows as usize,
                    geometry.top_k as usize,
                    geometry.experts as usize,
                    geometry.hidden as usize,
                    (geometry.intermediate as usize, shared_intermediate as usize),
                    (
                        catalogue.layout().fp8_slot(),
                        model.routed_weight_boundary(),
                    ),
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
                .and_then(|n| n.checked_add(geometry.hidden as usize * 2 + 2 + size_of::<u32>()))
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
        #[cfg(feature = "cuda")]
        let (peer_device, peer_unsupported_sm) = device.peer_with_unsupported_sm()?;
        #[cfg(feature = "rocm")]
        let (peer_device, peer_unsupported_sm) = (device.peer()?, None::<u32>);
        let peer_inputs = if let Some(peer_dev) = &peer_device {
            peer_dev.ctx.bind_to_thread()?;
            let (peer_free, peer_total) = peer_dev.ctx.mem_get_info()?;
            device.ctx.bind_to_thread()?;
            let max_span = candidates
                .iter()
                .map(|c| c.candidate.span_rows)
                .max()
                .context("no candidate spans")?;
            let max_wave = candidates
                .iter()
                .map(|c| c.candidate.wave_rows)
                .max()
                .context("no candidate waves")?;
            let peer_decode = decode_device_bytes(geometry, shared_intermediate, decode_rows)?;
            let peer_prefill = residency::peer_prefill_bytes(
                geometry,
                max_span,
                max_wave,
                (shared_intermediate, model.routed_weight_boundary()),
                catalogue.layout().fp8_slot(),
            )?;
            let peer_tables = geometry.layers as usize * geometry.experts as usize * 28;
            let peer_reserve = (peer_total / 16)
                .checked_add(peer_decode)
                .and_then(|n| n.checked_add(peer_prefill))
                .and_then(|n| n.checked_add(peer_tables))
                .context("peer GPU reserve overflow")?;
            (measured.peer_transfers.len() >= 2).then_some(crate::experts::plan::PeerInputs {
                device_free_bytes: peer_free as u64,
                device_reserved_bytes: peer_reserve as u64,
                decode_transfer: measured.peer_transfers[0],
                prefill_transfers: &measured.peer_transfers[1..],
            })
        } else {
            None
        };
        let first = &candidates[0].candidate;
        let (index, mut capacity, admission) = ExpertWorkspaceBudget::select_with_peer(
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
            peer_inputs,
            u32::try_from(request.capacity)?,
            &candidates,
        )?;
        let peer = (capacity.gpu0_layers < geometry.layers)
            .then_some(peer_device)
            .flatten();
        let mut workspace = candidates.swap_remove(index);
        let (routed_wave_rows, extra_scratch) = routed_wave_rows(
            geometry,
            &capacity,
            workspace.workspace_rows,
            (shared_intermediate, model.routed_weight_boundary()),
        )?;
        workspace.candidate.reserves.device_fixed_bytes = workspace
            .candidate
            .reserves
            .device_fixed_bytes
            .checked_add(extra_scratch)
            .context("resident routed fixed reserve overflow")?;
        workspace.mixer.planned_transient_bytes = workspace
            .mixer
            .planned_transient_bytes
            .checked_add(usize::try_from(extra_scratch)?)
            .context("resident routed workspace overflow")?;
        capacity.device_expert_bytes = capacity
            .device_expert_bytes
            .checked_sub(extra_scratch)
            .context("resident routed scratch exceeds device budget")?;
        let (peer_wave_rows, peer_extra_scratch) = peer_routed_wave_rows(
            geometry,
            &capacity,
            (shared_intermediate, model.routed_weight_boundary()),
        )?;
        capacity.peer_expert_bytes = capacity
            .peer_expert_bytes
            .checked_sub(peer_extra_scratch)
            .context("peer routed scratch exceeds peer budget")?;
        let host_backed = capacity.host_backed(&geometry);
        let non_expert_decode_bytes = manifest.non_expert_decode_bytes()?;
        let cold_schedule = draft_config
            .map(|draft| {
                crate::dspark::VerifyCostSchedule::derive(
                    draft,
                    (model, non_expert_decode_bytes),
                    (
                        &capacity,
                        admission,
                        &measured.service,
                        measured.copy_engines,
                    ),
                    (
                        [
                            device.info.memory_bytes_per_second,
                            peer.as_ref()
                                .map_or(0.0, |p| p.info.memory_bytes_per_second),
                            measured.h2d_bytes_per_second,
                            measured.d2h_bytes_per_second,
                        ],
                        [
                            device.info.l2_bytes as u64,
                            peer.as_ref().map_or(0, |p| p.info.l2_bytes as u64),
                        ],
                    ),
                    &measured.peer_transfers,
                    &vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize],
                )
            })
            .transpose()?;
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
                "routed_wave_rows":routed_wave_rows,"routed_extra_scratch_bytes":extra_scratch,
                "peer_wave_rows":peer_wave_rows,"peer_extra_scratch_bytes":peer_extra_scratch,
                "mixer_rows":workspace.mixer.plan.chunk,"workspace_rows":workspace.workspace_rows,
                "borrowed_slots":workspace.borrowed_slots,"slots_per_layer":capacity.slots_per_layer,
                "gpu0_layers":capacity.gpu0_layers,"peer_expert_bytes":capacity.peer_expert_bytes,
                "peer_transport":peer.is_some().then(|| serde_json::json!({"decode":"captured_portable_bounce_4k","prefill":"eager_peer_memcpy_8m_68m"})),
                "peer_unsupported_sm":peer_unsupported_sm,
                "host_blocks":capacity.host_blocks,"spare_blocks":capacity.spare_blocks,
                "host_backing_complete":host_backed,
                "disk_bytes":capacity.disk_bytes,"pinned":capacity.pinned,
                "host_available_bytes":available,"host_registration_bytes":registration,
                "device_fixed_bytes":workspace.candidate.reserves.device_fixed_bytes,
                "host_fixed_bytes":workspace.candidate.reserves.host_fixed_bytes,
                "vision_reserved_bytes":vision,"budget_bytes_per_token":admission.budget_bytes_per_token,
                "cpu_rows_measured":cpu.3,"cpu_rows_limit":cpu.2,
                "cpu_arithmetic_rows":super::super::moe::prefill::LANE16_ROWS,
                "decode_rows":decode_rows,"decode_extra_device_bytes":decode_extra,
                "context_span_rows":capacity.span_rows,"context_span_row_bytes":context_row_bytes,
                "context_span_bytes":context_row_bytes.checked_mul(capacity.span_rows as usize).context("draft context span bytes overflow")?,
                "dspark_verify_schedule":cold_schedule
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
            directory: directory.to_owned(),
            capacity,
            admission,
            measured,
            catalogue,
            decode_rows,
            routed_wave_rows,
            peer_wave_rows,
            peer,
            non_expert_decode_bytes,
            cold_schedule,
        })
    }

    pub fn peer_tail_elems(&self) -> Result<usize> {
        if self.peer.is_none() {
            return Ok(0);
        }
        let geometry = self.catalogue.geometry();
        PeerSendLayout::tail_bf16_elems(
            self.decode_rows,
            geometry.hidden as usize,
            geometry.top_k as usize,
        )
    }

    pub fn bind(self, engine: &mut Engine) -> Result<()> {
        let total_started = std::time::Instant::now();
        if let (Some(spec), Some(schedule)) = (engine.spec.as_mut(), self.cold_schedule) {
            spec.non_expert_decode_bytes = self.non_expert_decode_bytes;
            spec.cold_verify_schedule = schedule;
            spec.verify_schedule = schedule;
        }
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
            self.peer.as_ref(),
            geometry,
            residency::Load {
                capacity: &self.capacity,
                admission: self.admission,
                initial: &initial,
                directory: &self.directory,
                catalogue: self.catalogue,
                engines: self.measured.copy_engines,
                max_rows: u32::try_from(cpu.0.max(self.decode_rows))?,
                swiglu_rounding: engine.model_spec.swiglu_rounding(),
                swiglu_limit: engine.model_spec.swiglu_limit(),
                routed_weight: engine.model_spec.routed_weight_boundary(),
            },
        )?;
        let slots_load_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        let send_ptr = {
            let (pointer, _guard) = engine.work.n.device_ptr(&engine.device.stream);
            pointer
        };
        let send_cap_bytes = engine.work.n.len() * size_of::<bf16>();
        let cpu_bounds = (
            cpu.0.max(self.decode_rows),
            cpu.1.max(self.decode_rows * geometry.top_k as usize),
        );
        let participants = self
            .measured
            .service
            .participant_table(u32::try_from(cpu_bounds.0)?, u32::try_from(cpu_bounds.1)?)?;
        engine.moe = Some(Decode::new(
            &engine.device,
            DecodeOptions {
                peer: self.peer.as_ref().map(|peer| PeerBind {
                    device: peer,
                    gpu0_layers: self.capacity.gpu0_layers,
                    send_ptr,
                    send_cap_bytes,
                }),
                geometry,
                residual_output: engine.model_spec.architecture
                    != crate::model_spec::Architecture::Flint,
                max_rows: self.decode_rows,
                workers: self.capacity.worker_count,
                cpu_bounds,
                participants,
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
                swiglu_limit: engine.model_spec.swiglu_limit(),
                routed_weight: engine.model_spec.routed_weight_boundary(),
            },
        )?);
        let decode_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        let phase_started = std::time::Instant::now();
        engine.moe_prefill = Some(Prefill::new(
            (&engine.device, self.peer.as_ref()),
            &mut slots,
            &self.capacity,
            self.workspace.candidate.reserves,
            self.measured.service,
            (
                self.measured.h2d_bytes_per_second,
                self.measured.d2h_bytes_per_second,
                self.measured.copy_engines,
                self.measured.peer_transfers,
            ),
            (
                engine
                    .model_spec
                    .moe
                    .context("MoE shared geometry missing")?
                    .shared_intermediate,
                self.workspace.workspace_rows,
                self.workspace.mixer.plan.chunk,
                (self.routed_wave_rows, self.peer_wave_rows),
                (
                    engine.model_spec.swiglu_rounding(),
                    engine.model_spec.routed_weight_boundary(),
                ),
            ),
        )?);
        let prefill_new_ms = phase_started.elapsed().as_secs_f64() * 1000.0;
        engine.device.finish_upload()?;
        if let Some(peer) = &self.peer {
            peer.finish_upload()?;
            engine.device.ctx.bind_to_thread()?;
        }
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
        if let Some(peer) = executor.peer.as_mut() {
            peer.contribute = DecodeOps::new_invariant_with_boundaries(
                &peer.device.ctx,
                self.model_spec.swiglu_rounding(),
                self.model_spec.routed_weight_boundary(),
            )?;
            self.device.ctx.bind_to_thread()?;
        }
        executor.form = crate::cpu::expert_ffn::Form::Lane16;
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
        let output = if decode.residual_output {
            qwen::matrix(&self.work.x, stream, 0, rows, hidden)?
        } else {
            qwen::matrix(&self.work.out, stream, 0, rows, hidden)?
        };
        slots.run(
            &self.device,
            decode,
            slot_layer,
            ffn,
            x,
            &mut TensorMut(output),
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
            decode.trace(stream, slot_layer as usize, target.trace, &prefix)?;
            let output = if decode.residual_output {
                &self.work.x
            } else {
                &self.work.out
            };
            trace_value(trace, stream, &name, output, rows, hidden)?;
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

pub(super) struct PeerBind<'a> {
    pub device: &'a Device,
    pub gpu0_layers: u32,
    pub send_ptr: u64,
    pub send_cap_bytes: usize,
}

pub(super) struct PeerBounce {
    ptr: *mut u8,
    routed_off: usize,
    _allocation: PinnedHostSlice<u8>,
}

unsafe impl Send for PeerBounce {}
unsafe impl Sync for PeerBounce {}

impl PeerBounce {
    fn new(
        context: &Arc<CudaContext>,
        max_send_bytes: usize,
        max_routed_bytes: usize,
    ) -> Result<Self> {
        let routed_off = max_send_bytes.next_multiple_of(16);
        let total = (routed_off + max_routed_bytes).next_multiple_of(16);
        #[cfg(feature = "cuda")]
        let portable = cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE;
        #[cfg(feature = "rocm")]
        let portable = 1;
        // The portable allocation is owned through the last peer DMA completion.
        let mut allocation = unsafe { context.alloc_pinned_with_flags::<u8>(total, portable)? };
        let raw = allocation.as_mut_ptr()?;
        unsafe {
            std::ptr::write_bytes(raw, 0, total);
        }
        Ok(Self {
            ptr: raw.cast::<u8>(),
            routed_off,
            _allocation: allocation,
        })
    }

    fn send_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr.cast()
    }

    fn routed_ptr(&self) -> *mut std::ffi::c_void {
        unsafe { self.ptr.add(self.routed_off).cast() }
    }
}

pub(super) struct PeerDecode {
    pub gpu0_layers: u32,
    pub send_ptr: u64,
    pub send_cap_bytes: usize,
    pub device: Device,
    pub contribute: DecodeOps,
    pub combine: CombineOps,
    pub mask: CudaFunction,
    pub recv: CudaSlice<u8>,
    pub active: CudaSlice<u32>,
    pub final_active: CudaSlice<u32>,
    pub generation: CudaSlice<u64>,
    pub contributions: CudaSlice<bf16>,
    pub done: CudaSlice<u32>,
    pub routed_dev1: CudaSlice<bf16>,
    pub routed_dev0: CudaSlice<bf16>,
    pub scratch: CudaSlice<u8>,
    pub bounce: PeerBounce,
    pub fork: CudaEvent,
    pub join: CudaEvent,
    #[cfg(all(test, feature = "cuda"))]
    pub perturb_word: Option<(usize, u16)>,
}

pub(super) struct Decode {
    form: crate::cpu::expert_ffn::Form,
    geometry: ExpertGeometry,
    combine_blocks: u32,
    residual_output: bool,
    shared_intermediate: u32,
    fp8_slot: Option<crate::experts::catalogue::Fp8Slot>,
    max_rows: usize,
    phase: DecodePhase,
    route: RouteOps,
    contribute: DecodeOps,
    prefill: PrefillOps,
    combine: CombineOps,
    pub(super) peer: Option<PeerDecode>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<bf16>,
    route_logits: CudaSlice<bf16>,
    active: CudaSlice<u32>,
    final_active: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    contributions: CudaSlice<bf16>,
    done: CudaSlice<u32>,
    shared: CudaSlice<bf16>,
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
    peer_tail: Option<(usize, CudaGraph)>,
    verify_peer_tails: std::collections::BTreeMap<usize, (usize, CudaGraph)>,
    mailbox_ready: CudaEvent,
    reader: Arc<CudaEvent>,
    mask: CudaFunction,
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
        if let Some(peer) = &self.peer {
            peer.device.ctx.record_err(peer.device.stream.synchronize());
        }
    }
}

pub(super) struct DecodeOptions<'a> {
    pub peer: Option<PeerBind<'a>>,
    pub geometry: ExpertGeometry,
    pub residual_output: bool,
    pub max_rows: usize,
    pub workers: u32,
    pub cpu_bounds: (usize, usize),
    pub participants: crate::experts::plan::ParticipantTable,
    pub format: crate::model_spec::ExpertFormat,
    pub shared_intermediate: u32,
    pub swiglu_rounding: crate::model_spec::SwigluRounding,
    pub swiglu_limit: f32,
    pub routed_weight: crate::model_spec::RoutedWeightBoundary,
}

impl Decode {
    pub fn new(device: &Device, options: DecodeOptions<'_>) -> Result<Self> {
        let DecodeOptions {
            peer: peer_info,
            geometry,
            residual_output,
            max_rows,
            workers,
            cpu_bounds,
            participants,
            format,
            shared_intermediate,
            swiglu_rounding,
            swiglu_limit,
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
        let scratch_bytes = usize::try_from(decode_scratch_bytes_mixed(
            max_rows,
            geometry.top_k as usize,
            geometry.intermediate as usize,
            shared_intermediate as usize,
        ))?;
        let peer = if let Some(bind) = peer_info {
            let max_send = PeerSendLayout::for_rows(
                max_rows,
                geometry.hidden as usize,
                geometry.top_k as usize,
            )?;
            ensure!(
                bind.send_ptr != 0
                    && bind.send_ptr.is_multiple_of(16)
                    && bind.send_cap_bytes >= max_send.send_bytes,
                "peer decode send region {:#x} bytes={} cannot hold max send bytes={}",
                bind.send_ptr,
                bind.send_cap_bytes,
                max_send.send_bytes
            );
            context.bind_to_thread()?;
            let bounce = PeerBounce::new(context, max_send.send_bytes, hidden * size_of::<bf16>())?;
            let pstream = &bind.device.stream;
            let pctx = &bind.device.ctx;
            #[cfg(feature = "cuda")]
            let untimed = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
            #[cfg(feature = "rocm")]
            let untimed = Some(2);
            let pd = PeerDecode {
                gpu0_layers: bind.gpu0_layers,
                send_ptr: bind.send_ptr,
                send_cap_bytes: bind.send_cap_bytes,
                device: bind.device.clone(),
                contribute: DecodeOps::new_with_boundaries(pctx, swiglu_rounding, routed_weight)?,
                combine: CombineOps::new(pctx)?,
                mask: super::super::cubin::module(pctx, "moe_route")?
                    .load_function("moe_route_mask")?,
                recv: pstream.alloc_zeros(max_send.send_bytes)?,
                active: pstream.alloc_zeros(choices)?,
                final_active: pstream.clone_htod(&vec![1u32; choices])?,
                generation: pstream.alloc_zeros(choices)?,
                contributions: pstream.alloc_zeros(elements)?,
                done: pstream.alloc_zeros(choices)?,
                routed_dev1: pstream.alloc_zeros(hidden)?,
                routed_dev0: stream.alloc_zeros(hidden)?,
                scratch: pstream.alloc_zeros(scratch_bytes)?,
                bounce,
                fork: context.new_event(untimed)?,
                join: pctx.new_event(untimed)?,
                #[cfg(all(test, feature = "cuda"))]
                perturb_word: None,
            };
            context.bind_to_thread()?;
            Some(pd)
        } else {
            None
        };
        let decode = Self {
            form: crate::cpu::expert_ffn::Form::Lane8,
            geometry,
            combine_blocks: if residual_output {
                geometry.hidden.div_ceil(256)
            } else {
                1
            },
            residual_output,
            shared_intermediate,
            fp8_slot: layout.fp8_slot().cloned(),
            max_rows,
            phase: DecodePhase::default(),
            route: RouteOps::new(context, stream, geometry.experts, max_rows)?,
            contribute: DecodeOps::new_with_boundaries(context, swiglu_rounding, routed_weight)?,
            prefill: PrefillOps::new_with_boundaries(
                context,
                stream,
                format,
                swiglu_rounding,
                routed_weight,
            )?,
            combine: CombineOps::new(context)?,
            peer,
            ids: stream.alloc_zeros(choices)?,
            weights: stream.alloc_zeros(choices)?,
            route_logits: stream.alloc_zeros(max_rows * geometry.experts as usize)?,
            active: stream.alloc_zeros(choices)?,
            final_active: stream.clone_htod(&vec![1u32; choices])?,
            generation: stream.alloc_zeros(choices)?,
            contributions: stream.alloc_zeros(elements)?,
            done: stream.alloc_zeros(choices)?,
            shared: stream.alloc_zeros(hidden)?,
            scratch: stream.alloc_zeros(scratch_bytes)?,
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
            peer_tail: None,
            verify_peer_tails: std::collections::BTreeMap::new(),
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
                crate::cpu::expert_ffn::ExpertBoundaries {
                    swiglu_rounding,
                    swiglu_limit,
                    routed_weight,
                },
                participants,
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
            + (decode.contributions.len() + decode.shared.len()) * size_of::<bf16>()
            + decode.scratch.len()
            + decode.route.workspace_bytes();
        let planned = decode_device_bytes(geometry, shared_intermediate, max_rows)?;
        ensure!(
            allocated == planned,
            "MoE decode row buffers allocation={allocated} differs from planned={planned} rows={max_rows}"
        );
        Ok(decode)
    }

    pub(super) fn slot_stream<'a>(
        &'a self,
        device: &'a Device,
        layer: usize,
    ) -> &'a Arc<CudaStream> {
        if let Some(peer) = &self.peer
            && layer >= peer.gpu0_layers as usize
        {
            &peer.device.stream
        } else {
            &device.stream
        }
    }

    pub fn route(
        &mut self,
        device: &Device,
        layer: usize,
        ffn: &BoundFfn,
        x: Tensor<bf16, 2>,
    ) -> Result<()> {
        self.route_device(device, layer, ffn, x)?;
        self.read_route(device, layer, ffn, x)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.mailbox.synchronize()?;
        self.phase.reset();
        Ok(())
    }

    fn route_device(
        &self,
        device: &Device,
        layer: usize,
        ffn: &BoundFfn,
        x: Tensor<bf16, 2>,
    ) -> Result<()> {
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
        let (mut ids_view, mut weights_view) = if let Some(peer) = self
            .peer
            .as_ref()
            .filter(|peer| layer >= peer.gpu0_layers as usize)
        {
            ensure!(
                x.ptr == peer.send_ptr,
                "peer decode layer={layer} input ptr={:#x} differs from send region {:#x}",
                x.ptr,
                peer.send_ptr
            );
            let layout = PeerSendLayout::for_rows(rows, self.geometry.hidden as usize, k as usize)?;
            ensure!(
                layout.send_bytes <= peer.send_cap_bytes,
                "peer decode layer={layer} send bytes={} exceed capacity={}",
                layout.send_bytes,
                peer.send_cap_bytes
            );
            // SAFETY: send_ptr owns send_cap_bytes on GPU 0 and ids/weights offsets are 16-byte aligned past x.
            unsafe {
                (
                    TensorMut::new(
                        peer.send_ptr + layout.ids_off as u64,
                        [rows as u32, k],
                        [u64::from(k), 1],
                        layout.ids_bytes as u64,
                    )?,
                    TensorMut::new(
                        peer.send_ptr + layout.weights_off as u64,
                        [rows as u32, k],
                        [u64::from(k), 1],
                        layout.weights_bytes as u64,
                    )?,
                )
            }
        } else {
            (
                TensorMut(view(
                    &self.ids,
                    stream,
                    0,
                    [rows as u32, k],
                    [u64::from(k), 1],
                )?),
                TensorMut(view(
                    &self.weights,
                    stream,
                    0,
                    [rows as u32, k],
                    [u64::from(k), 1],
                )?),
            )
        };
        self.route.route(
            x,
            *router,
            &mut ids_view,
            &mut weights_view,
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

    fn read_route(
        &mut self,
        device: &Device,
        layer: usize,
        ffn: &BoundFfn,
        x: Tensor<bf16, 2>,
    ) -> Result<()> {
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
            let (ids_ptr, weights_ptr, _ids_guard, _weights_guard) = if let Some(peer) = self
                .peer
                .as_ref()
                .filter(|peer| layer >= peer.gpu0_layers as usize)
            {
                ensure!(
                    x.ptr == peer.send_ptr,
                    "peer route mailbox layer={layer} input ptr={:#x} differs from send region {:#x}",
                    x.ptr,
                    peer.send_ptr
                );
                let layout =
                    PeerSendLayout::for_rows(rows, self.geometry.hidden as usize, k as usize)?;
                (
                    peer.send_ptr + layout.ids_off as u64,
                    peer.send_ptr + layout.weights_off as u64,
                    None,
                    None,
                )
            } else {
                #[cfg(feature = "cuda")]
                let (ids, g_ids) = self.ids.device_ptr(&self.mailbox);
                #[cfg(feature = "rocm")]
                let (ids, g_ids) = self.ids.device_ptr(self.ids.stream());
                #[cfg(feature = "cuda")]
                let (weights, g_weights) = self.weights.device_ptr(&self.mailbox);
                #[cfg(feature = "rocm")]
                let (weights, g_weights) = self.weights.device_ptr(self.weights.stream());
                (ids, weights, Some(g_ids), Some(g_weights))
            };
            let (host_ids, _ids_write) =
                unsafe { self.host_ids.stream_synced_mut_slice(&self.mailbox) };
            let (host_weights, _weights_write) =
                unsafe { self.host_weights.stream_synced_mut_slice(&self.mailbox) };
            let (host_x, _x_write) = unsafe { self.host_x.stream_synced_mut_slice(&self.mailbox) };
            // SAFETY: mailbox waits self.reader after compute writes; mailbox synchronization completes these owned-prefix reads.
            unsafe {
                copy::dtoh_async(&mut host_ids[..count], ids_ptr, self.mailbox.as_ref())?;
                copy::dtoh_async(
                    &mut host_weights[..count],
                    weights_ptr,
                    self.mailbox.as_ref(),
                )?;
                copy::dtoh_async(
                    &mut host_x[..rows * self.geometry.hidden as usize],
                    x.ptr,
                    self.mailbox.as_ref(),
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
        let on_peer = self
            .peer
            .as_ref()
            .is_some_and(|peer| dispatch.layer >= peer.gpu0_layers);
        if on_peer {
            ensure!(
                dispatch.assignments.is_empty()
                    && dispatch.disk.is_empty()
                    && dispatch.active.iter().all(|&a| a == 1),
                "peer decode layer={} requires full GPU residency",
                dispatch.layer
            );
        } else {
            self.host_active.as_mut_slice()?[..rows * k].copy_from_slice(dispatch.active);
            self.host_generation.as_mut_slice()?[..rows * k].copy_from_slice(dispatch.generation);
            {
                let (active, _active_read) =
                    unsafe { self.host_active.stream_synced_slice(&self.mailbox) };
                let (generation, _generation_read) =
                    unsafe { self.host_generation.stream_synced_slice(&self.mailbox) };
                #[cfg(feature = "cuda")]
                let (active_ptr, _active_write) = self.active.device_ptr_mut(&self.mailbox);
                #[cfg(feature = "rocm")]
                let active_owner = self.active.stream().clone();
                #[cfg(feature = "rocm")]
                let (active_ptr, _active_write) = self.active.device_ptr_mut(&active_owner);
                #[cfg(feature = "cuda")]
                let (generation_ptr, _generation_write) =
                    self.generation.device_ptr_mut(&self.mailbox);
                #[cfg(feature = "rocm")]
                let generation_owner = self.generation.stream().clone();
                #[cfg(feature = "rocm")]
                let (generation_ptr, _generation_write) =
                    self.generation.device_ptr_mut(&generation_owner);
                // SAFETY: read_route waits self.reader before mailbox writes; compute waits self.mailbox_ready before reading the owned rows*top_k spans.
                unsafe {
                    copy::htod_async(active_ptr, &active[..rows * k], self.mailbox.as_ref())?;
                    copy::htod_async(
                        generation_ptr,
                        &generation[..rows * k],
                        self.mailbox.as_ref(),
                    )?;
                }
            }
            self.mailbox_ready.record(&self.mailbox)?;
            stream.wait(&self.mailbox_ready)?;
        }
        if let Some(layer) = graph {
            self.segment(rows, layer)?.contribute.launch()?;
        } else {
            self.contribute_device(device, dispatch.layer as usize, ffn.clone(), slots, x)?;
        }
        self.reader.record(stream)?;
        let reader = self.reader.clone();
        self.upload_cpu(device, &mut dispatch, &reader)?;
        if let Some(layer) = graph {
            self.segment(rows, layer)?.join.launch()?;
        } else {
            self.combine_device(device, dispatch.layer as usize, ffn, x, y)?;
        }
        self.stats[1] += dispatch.active.iter().map(|&n| u64::from(n)).sum::<u64>();
        self.stats[4] += 1;
        self.phase.finish()?;
        Ok(())
    }

    fn contribute_device(
        &mut self,
        device: &Device,
        layer: usize,
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
        if let Some(peer) = self
            .peer
            .as_mut()
            .filter(|peer| layer >= peer.gpu0_layers as usize)
        {
            ensure!(
                x.ptr == peer.send_ptr,
                "peer decode layer={layer} input ptr={:#x} differs from send region {:#x}",
                x.ptr,
                peer.send_ptr
            );
            let layout = PeerSendLayout::for_rows(rows, hidden, k)?;
            ensure!(
                layout.send_bytes <= peer.send_cap_bytes,
                "peer decode layer={layer} send bytes={} exceed capacity={}",
                layout.send_bytes,
                peer.send_cap_bytes
            );
            let pstream = &peer.device.stream;
            // SAFETY: contiguous GPU 0 send packet, portable bounce region, and GPU 1 recv slice stay live and ordered by fork/join events.
            unsafe {
                copy::dtoh_raw(
                    peer.bounce.send_ptr(),
                    peer.send_ptr,
                    layout.send_bytes,
                    stream.as_ref(),
                )
                .context("cuMemcpyDtoHAsync decode peer send 0->h")?;
            }
            peer.fork
                .record(stream)
                .context("cuEventRecord decode peer fork")?;
            peer.device
                .ctx
                .bind_to_thread()
                .context("cuCtxSetCurrent decode peer")?;
            pstream
                .wait(&peer.fork)
                .context("cuStreamWaitEvent decode peer fork")?;
            let recv_base = peer.recv.device_ptr(pstream).0;
            unsafe {
                copy::htod_raw(
                    recv_base,
                    peer.bounce.send_ptr(),
                    layout.send_bytes,
                    pstream.as_ref(),
                )
                .context("cuMemcpyHtoDAsync decode peer recv h->1")?;
            }
            // SAFETY: recv_base owns layout.send_bytes on GPU 1 and x/ids/weights offsets are 16-byte aligned and disjoint.
            let (p_x, p_ids, p_weights) = unsafe {
                (
                    Tensor::<bf16, 2>::new(
                        recv_base,
                        [rows as u32, hidden as u32],
                        [hidden as u64, 1],
                        layout.ids_off as u64,
                    )?,
                    Tensor::<i32, 2>::new(
                        recv_base + layout.ids_off as u64,
                        [rows as u32, k as u32],
                        [k as u64, 1],
                        layout.ids_bytes as u64,
                    )?,
                    Tensor::<bf16, 2>::new(
                        recv_base + layout.weights_off as u64,
                        [rows as u32, k as u32],
                        [k as u64, 1],
                        layout.weights_bytes as u64,
                    )?,
                )
            };
            let p_active = view(
                &peer.active,
                pstream,
                0,
                [rows as u32, k as u32],
                [k as u64, 1],
            )?;
            let p_gen = view(
                &peer.generation,
                pstream,
                0,
                [rows as u32, k as u32],
                [k as u64, 1],
            )?;
            // SAFETY: every table covers the expert domain and each assignment buffer covers rows*top_k on GPU 1.
            unsafe {
                pstream
                    .launch_builder(&peer.mask)
                    .arg(&p_ids.ptr)
                    .arg(&slots.valid.ptr)
                    .arg(&slots.generation.ptr)
                    .arg(&p_active.ptr)
                    .arg(&p_gen.ptr)
                    .arg(&((rows * k) as i32))
                    .arg(&(self.geometry.experts as i32))
                    .launch(grid((rows * k).div_ceil(256), 256))
                    .context("cuLaunchKernel decode peer moe_route_mask")?;
                let (done_ptr, _write) = peer.done.device_ptr_mut(pstream);
                copy::memset_bytes(done_ptr, 0, rows * k * size_of::<u32>(), pstream.as_ref())
                    .context("cuMemsetD8Async decode peer done")?;
            }
            let p_contributions = view(
                &peer.contributions,
                pstream,
                0,
                [rows as u32, k as u32, hidden as u32],
                [(k * hidden) as u64, hidden as u64, 1],
            )?;
            let mut p_routed =
                TensorMut(qwen::matrix(&peer.routed_dev1, pstream, 0, rows, hidden)?);
            let mut p_done = TensorMut(view(
                &peer.done,
                pstream,
                0,
                [rows as u32, k as u32],
                [k as u64, 1],
            )?);
            let peer_input = MoeDecode {
                x: p_x,
                ids: p_ids,
                active: p_active,
                weights: p_weights,
                slots_gate_up: slots.gate_up,
                slots_down: slots.down,
                slot_valid: slots.valid,
                slot_generation: slots.generation,
                assignment_gen: p_gen,
                shared_gate: shared[0],
                shared_up: shared[1],
                shared_down: shared[2],
                shared_expert_gate: shared_gate,
                contributions: p_contributions,
                shared_out: &mut p_routed,
                done: &mut p_done,
            };
            let p_scratch = qwen::scratch(&peer.scratch, pstream);
            if let Some(slot) = self.fp8_slot.as_ref() {
                peer.contribute
                    .contribute_routed_w8a16(
                        pstream,
                        &peer_input,
                        slot,
                        &p_scratch,
                        self.geometry.intermediate,
                    )
                    .context("decode peer contribute_routed_w8a16")?;
            } else {
                peer.contribute
                    .contribute_routed(pstream, &peer_input, &p_scratch, self.geometry.intermediate)
                    .context("decode peer contribute_routed")?;
            }
            let p_final_active = view(
                &peer.final_active,
                pstream,
                0,
                [rows as u32, k as u32],
                [k as u64, 1],
            )?;
            peer.combine
                .combine_routed(
                    pstream,
                    &MoeCombineRouted {
                        ids: p_ids,
                        final_active: p_final_active,
                        done: p_done.read(),
                        contributions: p_contributions,
                        routed_out: TensorMut(p_routed.read()),
                    },
                )
                .context("decode peer combine_routed")?;
            let src_routed = peer.routed_dev1.device_ptr(pstream).0;
            let routed_bytes = layout.recv_bytes;
            // SAFETY: peer.routed_dev1 and portable bounce region stay live and ordered before join.
            unsafe {
                copy::dtoh_raw(
                    peer.bounce.routed_ptr(),
                    src_routed,
                    routed_bytes,
                    pstream.as_ref(),
                )
                .context("cuMemcpyDtoHAsync decode peer routed 1->h")?;
            }
            peer.join
                .record(pstream)
                .context("cuEventRecord decode peer join")?;
            device
                .ctx
                .bind_to_thread()
                .context("cuCtxSetCurrent decode gpu0")?;
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
            let gpu0_input = MoeDecode {
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
            return self
                .contribute
                .contribute_shared(stream, &gpu0_input, &scratch, self.geometry.intermediate)
                .context("decode gpu0 contribute_shared");
        }
        {
            let (done_ptr, _write) = self.done.device_ptr_mut(stream);
            // SAFETY: done owns max_rows*top_k words and is ordered after the preceding layer's combine.
            unsafe {
                copy::memset_bytes(done_ptr, 0, rows * k * size_of::<u32>(), stream.as_ref())?
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
                    copy::htod_async(
                        contributions.ptr + (first * 2) as u64,
                        &values[first..first + hidden],
                        self.mailbox.as_ref(),
                    )?;
                    copy::htod_async(
                        done.ptr + (choice * 4) as u64,
                        &done_values[choice..choice + 1],
                        self.mailbox.as_ref(),
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
        layer: usize,
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
        let shared_out = qwen::matrix(&self.shared, stream, 0, rows, hidden)?;
        let (residual, y_out) = if self.residual_output {
            (Some(TensorMut(y.read())), None)
        } else {
            (None, Some(TensorMut(y.read())))
        };
        if let Some(peer) = self
            .peer
            .as_ref()
            .filter(|peer| layer >= peer.gpu0_layers as usize)
        {
            stream
                .wait(&peer.join)
                .context("cuStreamWaitEvent decode peer join")?;
            let routed_in = qwen::matrix(&peer.routed_dev0, stream, 0, rows, hidden)?;
            // SAFETY: portable bounce region was written before peer.join and routed_in is live on GPU 0.
            unsafe {
                copy::htod_raw(
                    routed_in.ptr,
                    peer.bounce.routed_ptr(),
                    rows * hidden * size_of::<bf16>(),
                    stream.as_ref(),
                )
                .context("cuMemcpyHtoDAsync decode peer routed h->0")?;
            }
            #[cfg(all(test, feature = "cuda"))]
            if let Some((word_idx, bits)) = peer.perturb_word {
                ensure!(
                    word_idx < rows * hidden,
                    "perturbed word={word_idx} outside routed_dev0 len={}",
                    rows * hidden
                );
                // SAFETY: word_idx is inside the owned peer.routed_dev0 buffer on GPU 0.
                unsafe {
                    cudarc::driver::sys::cuMemsetD16Async(
                        routed_in.ptr + (word_idx * 2) as u64,
                        bits,
                        1,
                        stream.cu_stream(),
                    )
                    .result()
                    .context("cuMemsetD16Async perturb routed_peer")?;
                }
            }
            return self
                .combine
                .join_routed_residual(
                    stream,
                    &MoeJoinRoutedResidual {
                        routed_in,
                        shared_out,
                        x,
                        shared_expert_gate: shared_gate,
                        residual,
                        y_out,
                    },
                    self.combine_blocks,
                )
                .context("decode gpu0 join_routed_residual");
        }
        let ids = view(&self.ids, stream, 0, [rows as u32, k as u32], [k as u64, 1])?;
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
        let final_active = view(
            &self.final_active,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?;
        self.combine.combine_residual(
            stream,
            &MoeCombineResidual {
                ids,
                final_active,
                done,
                contributions,
                shared_out,
                x,
                shared_expert_gate: shared_gate,
                residual,
                y_out,
            },
            self.combine_blocks,
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
                    form: self.form,
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
                    crate::experts::disk::Job {
                        layer: dispatch.layer,
                        hidden: cpu_hidden,
                        rows: dispatch.miss_rows.len(),
                        routes: dispatch.disk,
                        form: self.form,
                    },
                    output,
                    contribute,
                )?;
        }
        device.ctx.check_err()?;
        Ok(())
    }

    pub fn trace(
        &self,
        stream: &Arc<CudaStream>,
        layer: usize,
        trace: &mut Trace,
        prefix: &str,
    ) -> Result<()> {
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
        let (contrib_stream, contrib_slice, done_slice) = if let Some(peer) = self
            .peer
            .as_ref()
            .filter(|peer| layer >= peer.gpu0_layers as usize)
        {
            (&peer.device.stream, &peer.contributions, &peer.done)
        } else {
            (stream, &self.contributions, &self.done)
        };
        for (name, data_stream, data, shape) in [
            (
                "contributions",
                contrib_stream,
                contrib_slice,
                vec![rows, k, hidden],
            ),
            ("shared", stream, &self.shared, vec![rows, hidden]),
        ] {
            let len = shape.iter().product::<usize>();
            trace.add(
                format!("{prefix}.moe.{name}"),
                shape,
                data_stream
                    .clone_dtoh(&data.slice(..len))?
                    .into_iter()
                    .map(bf16::to_f32)
                    .collect(),
            )?;
        }
        trace.add(
            format!("{prefix}.moe.done"),
            vec![rows, k],
            contrib_stream
                .clone_dtoh(&done_slice.slice(..rows * k))?
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

#[cfg(all(test, feature = "cuda"))]
#[path = "engine_moe_e2e.rs"]
mod e2e;
