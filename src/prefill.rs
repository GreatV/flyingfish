use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn dense_prompt_bytes(
    spec: &crate::model_spec::ModelSpec,
    capacity: usize,
    chunk: usize,
    kv: crate::kv::KvFormat,
) -> Result<usize> {
    ensure!(
        capacity > 0 && chunk > 0 && chunk <= capacity,
        "invalid dense prefill extent"
    );
    if spec.moe.is_some()
        || spec.linear_format.fp8_layout().is_none()
        || kv != crate::kv::KvFormat::Bf16
        || chunk <= crate::mtp::MAX_VERIFY_ROWS
        || capacity == chunk
    {
        return Ok(0);
    }
    capacity
        .checked_mul(spec.hidden as usize)
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<u16>()))
        .context("dense prefill prompt buffer extent overflow")
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn chunk_end(start: usize, expected: usize, rows: usize, total: usize) -> Result<usize> {
    ensure!(
        start == expected && rows > 0,
        "prefill chunks are out of order"
    );
    let end = start
        .checked_add(rows)
        .context("prefill chunk extent overflow")?;
    ensure!(end <= total, "prefill chunk exceeds the request");
    Ok(end)
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub chunk: usize,
    pub requested_chunk: Option<usize>,
    pub free_bytes: usize,
    pub reserve_bytes: usize,
    pub row_bytes: usize,
    pub row_limit: usize,
    pub alignment: usize,
}

impl Plan {
    pub fn new(
        c: &Config,
        capacity: usize,
        requested: Option<usize>,
        free: usize,
        total: usize,
        alignment: usize,
        draft_row_bytes: usize,
    ) -> Result<Self> {
        let row_bytes = (4 * c.hidden_size + c.qkv_dim() + 3 * c.intermediate_size) * 2
            + 4
            + c.num_attention_heads * 4
            + draft_row_bytes;
        Self::budget(capacity, requested, free, total, alignment, row_bytes)
    }

    fn budget(
        capacity: usize,
        requested: Option<usize>,
        free: usize,
        total: usize,
        alignment: usize,
        row_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            alignment > 0 && row_bytes > 0,
            "invalid prefill alignment or row size"
        );
        ensure!(free <= total, "free VRAM exceeds total VRAM");
        let reserve_bytes = total / 16;
        let available = free.checked_sub(reserve_bytes).ok_or_else(|| {
            anyhow::anyhow!(
                "prefill VRAM budget: required reserve={reserve_bytes} available={free}"
            )
        })?;
        let row_limit = capacity.min(available / row_bytes);
        ensure!(
            row_limit > 0,
            "prefill workspace cannot fit one row: required={row_bytes} available={available}"
        );
        let chunk = if let Some(chunk) = requested {
            ensure!(
                chunk > 0 && chunk <= row_limit,
                "requested prefill chunk {chunk} exceeds row budget {row_limit}: required={} available={available}",
                chunk
                    .checked_mul(row_bytes)
                    .context("prefill byte count overflow")?
            );
            chunk
        } else if row_limit < alignment {
            row_limit
        } else {
            let tiles = row_limit / alignment;
            alignment * (1usize << (usize::BITS - 1 - tiles.leading_zeros()))
        };
        Ok(Self {
            chunk,
            requested_chunk: requested,
            free_bytes: free,
            reserve_bytes,
            row_bytes,
            row_limit,
            alignment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Plan, WorkspaceBudget, WorkspaceRequest, chunk_end, dense_prompt_bytes};

    #[test]
    fn dense_prefill_chunk_order_rejects_a_swapped_predecessor() -> anyhow::Result<()> {
        let mut expected = 0;
        for (start, rows) in [(0, 8192), (8192, 8128)] {
            expected = chunk_end(start, expected, rows, 16320)?;
        }
        assert_eq!(expected, 16320);
        assert!(chunk_end(8192, 0, 8128, 16320).is_err());
        assert!(chunk_end(0, 0, 16321, 16320).is_err());
        assert!(chunk_end(usize::MAX, usize::MAX, 1, usize::MAX).is_err());
        println!(
            "PASS dense_prefill_chunk_order_control rows=16320 swapped_start=8192 expected_start=0 rejected=true"
        );
        Ok(())
    }

    #[test]
    fn dense_prompt_buffers_preserve_exact_capacity_bounds() -> anyhow::Result<()> {
        use crate::{
            gdn_plan::GdnPolicy,
            kv::KvFormat,
            model_spec::{LinearFormat, ModelSpec},
            op_contract::GdnSpec,
        };
        let root = serde_json::from_str(include_str!("../tests/data/q38-config.json"))?;
        let mut model = ModelSpec::qwen(&root, &serde_json::json!({}))?;
        model.linear_format = LinearFormat::fp8_block([128, 128], safetensors::Dtype::BF16)?;
        assert_eq!(model.hidden, 5120);
        let gdn = GdnSpec::from_model(&model)?;
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model)?.row_bytes()?;
        for (capacity, expected) in [(1024, 20_971_520), (4096, 83_886_080), (16384, 335_544_320)] {
            let chunk = capacity / 2;
            let bytes = dense_prompt_bytes(&model, capacity, chunk, KvFormat::Bf16)?;
            assert_eq!(bytes, expected);
            assert_eq!(
                dense_prompt_bytes(&model, capacity, capacity, KvFormat::Bf16)?,
                0
            );
            assert_eq!(
                dense_prompt_bytes(&model, capacity, chunk, KvFormat::Affine8)?,
                0
            );
            let workspace = |rows| {
                Ok(usize::try_from(
                    GdnPolicy::P2.workspace(&gdn, rows as u32)?.bytes,
                )?)
            };
            let fixed = bytes + 4096;
            let total = 48usize << 30;
            let live = fixed + row_bytes * chunk + workspace(chunk)?;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(chunk),
                free_after_resident: total / 16 + live,
                total,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(request, workspace, Ok)?;
            assert_eq!(budget.plan.chunk, chunk);
            assert_eq!(budget.plan.row_limit, chunk);
            assert_eq!(budget.planned_transient_bytes, live);
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    workspace,
                    Ok
                )
                .is_err()
            );
            println!(
                "PASS dense_prompt_capacity capacity={capacity} chunk={chunk} layout=[2,{capacity},5120] prompt_bytes={bytes} planned_transient_bytes={live} boundary_minus_one_rejected=true"
            );
        }
        assert!(dense_prompt_bytes(&model, usize::MAX, 32, KvFormat::Bf16).is_err());
        assert!(dense_prompt_bytes(&model, 32, 33, KvFormat::Bf16).is_err());
        model.linear_format = LinearFormat::Bf16;
        assert_eq!(dense_prompt_bytes(&model, 16384, 8192, KvFormat::Bf16)?, 0);
        let root = serde_json::from_str(include_str!("../tests/data/minicpm-config.json"))?;
        let mut moe =
            ModelSpec::minicpm(&root, &serde_json::json!({"eos_token_id":[248044,248046]}))?;
        moe.linear_format = LinearFormat::Fp8E4m3Channel;
        for capacity in [1024, 4096, 16384] {
            assert_eq!(
                dense_prompt_bytes(&moe, capacity, capacity / 2, KvFormat::Bf16)?,
                0
            );
        }
        Ok(())
    }

    #[test]
    fn budget_bounds_and_override() {
        let plan = Plan::budget(33024, None, 160000, 160000, 64, 4).unwrap();
        assert_eq!(plan.chunk, 32768);
        assert!(plan.chunk * plan.row_bytes + plan.reserve_bytes <= plan.free_bytes);
        assert!(Plan::budget(33024, Some(33025), 160000, 160000, 64, 4).is_err());
        assert_eq!(
            Plan::budget(33024, Some(33024), 160000, 160000, 64, 4)
                .unwrap()
                .chunk,
            33024
        );
        let exact = Plan::budget(16, Some(16), 72, 128, 1, 4).unwrap();
        assert_eq!(exact.row_limit, 16);
        assert!(Plan::budget(16, Some(16), 71, 128, 1, 4).is_err());
    }

    #[test]
    fn memory_limit_and_small_sequence() {
        assert_eq!(
            Plan::budget(32768, None, 20000, 160000, 64, 4)
                .unwrap()
                .chunk,
            2048
        );
        assert_eq!(
            Plan::budget(17, None, 160000, 160000, 64, 4).unwrap().chunk,
            17
        );
        assert!(Plan::budget(32768, None, 10000, 160000, 64, 4).is_err());
    }
}

/// Shared budget input for models with persistent mixer state and per-operator scratch.
/// `free_after_resident` is sampled after weights, KV/state and layer arena allocation;
/// they are recorded in the ledger but are not subtracted a second time.
#[derive(Debug, Serialize)]
pub struct WorkspaceBudget {
    pub plan: Plan,
    pub operator_workspace_bytes: usize,
    pub planned_transient_bytes: usize,
    pub accounting: &'static str,
}
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceRequest {
    pub capacity: usize,
    pub requested: Option<usize>,
    pub free_after_resident: usize,
    pub total: usize,
    pub alignment: usize,
    pub row_bytes: usize,
    pub fixed_transient_bytes: usize,
}
impl WorkspaceBudget {
    /// `workspace(rows)` must be monotone and include every simultaneously live
    /// operator temporary, with alignment. Sequential operators contribute their max.
    /// `select` bounds the DEFAULT chunk only; an explicit `requested` chunk is
    /// admitted as given, up to the same row limit.
    pub fn new(
        request: WorkspaceRequest,
        workspace: impl Fn(usize) -> Result<usize>,
        select: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Self> {
        Self::with_reserve(request, request.total / 16, 1, workspace, select)
    }

    fn with_reserve(
        request: WorkspaceRequest,
        reserve: usize,
        minimum_rows: usize,
        workspace: impl Fn(usize) -> Result<usize>,
        select: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Self> {
        let WorkspaceRequest {
            capacity,
            requested,
            free_after_resident,
            total,
            alignment,
            row_bytes,
            fixed_transient_bytes,
        } = request;
        ensure!(
            capacity > 0 && alignment > 0 && row_bytes > 0 && free_after_resident <= total,
            "invalid workspace budget"
        );
        ensure!(
            minimum_rows > 0 && minimum_rows <= capacity,
            "workspace row floor {minimum_rows} outside capacity {capacity}"
        );
        let available = free_after_resident.checked_sub(reserve).with_context(|| {
            format!("VRAM reserve {reserve} exceeds free memory {free_after_resident}")
        })?;
        let bytes = |rows: usize| -> Result<usize> {
            let operator_bytes = workspace(rows)?;
            rows.max(minimum_rows)
                .checked_mul(row_bytes)
                .and_then(|v| v.checked_add(fixed_transient_bytes))
                .and_then(|v| v.checked_add(operator_bytes))
                .context("workspace byte count overflow")
        };
        let one_row = bytes(1)?;
        ensure!(
            one_row <= available,
            "workspace cannot fit one row: required={one_row} available={available} row_bytes={row_bytes} fixed_bytes={fixed_transient_bytes}"
        );
        let (mut low, mut high) = (1usize, capacity);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if bytes(mid)? <= available {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let row_limit = low;
        let chunk = if let Some(n) = requested {
            ensure!(
                n > 0 && n <= row_limit,
                "requested chunk {n} exceeds exact workspace budget: capacity={capacity} row_limit={row_limit} available_bytes={available} row_bytes={row_bytes} fixed_transient_bytes={fixed_transient_bytes}"
            );
            n
        } else {
            let admitted = if row_limit < alignment {
                row_limit
            } else {
                let tiles = row_limit / alignment;
                alignment * (1usize << (usize::BITS - 1 - tiles.leading_zeros()))
            };
            let selected = select(admitted)?;
            ensure!(
                selected > 0
                    && selected <= admitted
                    && (selected == admitted || selected.is_multiple_of(alignment)),
                "default chunk selection must preserve admission and alignment"
            );
            selected
        };
        let scratch = workspace(chunk)?;
        Ok(Self {
            plan: Plan {
                chunk,
                requested_chunk: requested,
                free_bytes: free_after_resident,
                reserve_bytes: reserve,
                row_bytes,
                row_limit,
                alignment,
            },
            operator_workspace_bytes: scratch,
            planned_transient_bytes: bytes(chunk)?,
            accounting: "free_after_resident",
        })
    }
}

/// Candidate row domains and measured resources before expert-slot allocation.
#[derive(Clone, Copy, Debug)]
pub struct ExpertWorkspaceRequest {
    pub mixer: WorkspaceRequest,
    pub span_rows: u32,
    pub wave_rows: u32,
    pub wave_workspace_bytes: usize,
    pub device_headroom_bytes: usize,
    /// CPU/disk scratch, pinned staging, runtime state and measured host headroom.
    pub host_fixed_bytes: u64,
    pub prefill_staging_slots: u32,
    pub disk_ring_slots: u32,
}

#[derive(Debug)]
pub struct ExpertWorkspaceBudget {
    pub mixer: WorkspaceBudget,
    pub workspace_rows: usize,
    /// Whole tail slots borrowed at the candidate's span bound.
    pub borrowed_slots: u32,
    pub candidate: crate::experts::plan::SpanCandidate,
}

#[derive(Debug)]
#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) struct ExpertSpanLayout {
    pub rows: u32,
    pub residual: std::ops::Range<usize>,
    pub normalized: std::ops::Range<usize>,
    pub contributions: std::ops::Range<usize>,
    pub done: std::ops::Range<usize>,
    pub ids: std::ops::Range<usize>,
    pub weights: std::ops::Range<usize>,
    pub active: std::ops::Range<usize>,
    pub bytes: usize,
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
impl ExpertSpanLayout {
    pub fn validate_wave(&self, wave_rows: u32, workspace_rows: usize) -> Result<()> {
        ensure!(
            wave_rows > 0 && wave_rows <= self.rows && wave_rows as usize <= workspace_rows,
            "expert wave_rows={wave_rows} exceed span_rows={} or workspace_rows={workspace_rows}",
            self.rows
        );
        Ok(())
    }

    pub fn new(
        reserves: &crate::experts::plan::Reserves,
        geometry: crate::experts::catalogue::ExpertGeometry,
        slot_bytes: u64,
        rows: u32,
    ) -> Result<Self> {
        ensure!(rows > 0, "zero expert span rows={rows}");
        let hidden = u64::from(geometry.hidden) * 2;
        let choices = u64::from(geometry.top_k);
        ensure!(
            hidden > 0
                && choices > 0
                && reserves.residual_row_bytes == hidden
                && reserves.normalized_row_bytes == hidden
                && reserves.contribution_row_bytes == hidden * choices
                && reserves.done_row_bytes == choices * 4
                && reserves.route_row_bytes == choices * (4 + 2 + 4),
            "expert span reserves disagree with geometry={geometry:?} rows={rows}"
        );
        let mut bytes = 0usize;
        let mut region = |row_bytes: u64| -> Result<std::ops::Range<usize>> {
            let first = bytes
                .checked_add(15)
                .context("expert span alignment overflow")?
                & !15;
            let count = usize::try_from(
                row_bytes
                    .checked_mul(u64::from(rows))
                    .context("expert span region overflow")?,
            )?;
            bytes = first
                .checked_add(count)
                .context("expert span extent overflow")?;
            Ok(first..bytes)
        };
        let residual = region(reserves.residual_row_bytes)?;
        let normalized = region(reserves.normalized_row_bytes)?;
        let contributions = region(reserves.contribution_row_bytes)?;
        let done = region(reserves.done_row_bytes)?;
        let ids = region(choices * 4)?;
        let weights = region(choices * 2)?;
        drop(region);
        if weights.len() % 16 != 0 {
            bytes = bytes
                .checked_add(15)
                .context("expert span weight padding overflow")?
                & !15;
        }
        let mut region = |row_bytes: u64| -> Result<std::ops::Range<usize>> {
            let first = bytes
                .checked_add(15)
                .context("expert span alignment overflow")?
                & !15;
            let count = usize::try_from(
                row_bytes
                    .checked_mul(u64::from(rows))
                    .context("expert span region overflow")?,
            )?;
            bytes = first
                .checked_add(count)
                .context("expert span extent overflow")?;
            Ok(first..bytes)
        };
        let active = region(choices * 4)?;
        let raw = reserves.span_bytes(rows)?;
        let slot = slot_bytes;
        ensure!(
            slot.is_multiple_of(16) && bytes as u64 <= raw.div_ceil(slot) * slot,
            "aligned expert span bytes={bytes} exceed borrowed bytes={} raw={raw}",
            raw.div_ceil(slot) * slot
        );
        Ok(Self {
            rows,
            residual,
            normalized,
            contributions,
            done,
            ids,
            weights,
            active,
            bytes,
        })
    }
}

impl ExpertWorkspaceBudget {
    pub fn new(
        request: ExpertWorkspaceRequest,
        geometry: crate::experts::catalogue::ExpertGeometry,
        slot_bytes: u64,
        mixer_workspace: impl Fn(usize) -> Result<usize>,
        select: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Self> {
        use crate::experts::plan::{Reserves, SpanCandidate, span_borrow_slots};
        let ExpertWorkspaceRequest {
            mut mixer,
            span_rows,
            wave_rows,
            wave_workspace_bytes,
            device_headroom_bytes,
            host_fixed_bytes,
            prefill_staging_slots,
            disk_ring_slots,
        } = request;
        ensure!(
            mixer.free_after_resident <= mixer.total,
            "expert free-after-resident bytes {} exceed total bytes {}",
            mixer.free_after_resident,
            mixer.total
        );
        ensure!(
            geometry.layers > 0
                && geometry.experts > 0
                && geometry.hidden > 0
                && geometry.intermediate > 0
                && geometry.top_k > 0
                && geometry.top_k <= geometry.experts,
            "invalid expert workspace geometry: {geometry:?}"
        );
        ensure!(
            span_rows > 0 && span_rows as usize <= mixer.capacity,
            "expert span rows {span_rows} outside capacity {}",
            mixer.capacity
        );
        ensure!(
            wave_rows > 0 && wave_rows <= span_rows,
            "expert wave rows {wave_rows} outside span {span_rows}"
        );
        let hidden_bytes = u64::from(geometry.hidden) * 2;
        let choices = u64::from(geometry.top_k);
        let mut reserves = Reserves {
            device_fixed_bytes: 0,
            host_fixed_bytes,
            residual_row_bytes: hidden_bytes,
            normalized_row_bytes: hidden_bytes,
            contribution_row_bytes: hidden_bytes
                .checked_mul(choices)
                .context("expert contribution row overflow")?,
            done_row_bytes: choices * size_of::<u32>() as u64,
            // ids i32, weights BF16 and final-active u32.
            route_row_bytes: choices
                * (size_of::<i32>() + size_of::<half::bf16>() + size_of::<u32>()) as u64,
            prefill_staging_slots,
            disk_ring_slots,
        };
        let staging_bytes = reserves.persistent_bytes(slot_bytes)?;
        let span_bytes = reserves.span_bytes(span_rows)?;
        let borrowed_slots = span_borrow_slots(&reserves, slot_bytes, span_rows, span_rows)?;
        let routed_slots = u64::from(geometry.layers) * u64::from(geometry.experts);
        ensure!(
            u64::from(borrowed_slots) + u64::from(geometry.layers) <= routed_slots,
            "span bytes {span_bytes} borrow {borrowed_slots} slots, leaving fewer than one per layer: routed_slots={routed_slots} layers={}",
            geometry.layers
        );
        let borrowed_bytes = u64::from(borrowed_slots)
            .checked_mul(slot_bytes)
            .context("borrowed expert slot extent overflow")?;
        // Decode retains one contribution/done row and two route snapshots.
        let decode_mailbox_bytes = choices
            .checked_mul(
                2 * (size_of::<i32>()
                    + size_of::<half::bf16>()
                    + size_of::<u32>()
                    + size_of::<u64>()) as u64,
            )
            .and_then(|n| n.checked_add(reserves.contribution_row_bytes))
            .and_then(|n| n.checked_add(reserves.done_row_bytes))
            .context("decode mailbox extent overflow")?;
        let minimum_experts = u64::from(geometry.layers)
            .checked_mul(slot_bytes)
            .context("minimum expert capacity overflow")?;
        let withheld = usize::try_from(
            borrowed_bytes
                .checked_add(staging_bytes)
                .and_then(|n| n.checked_add(minimum_experts))
                .and_then(|n| n.checked_add(decode_mailbox_bytes))
                .context("expert workspace reservation overflow")?,
        )?;
        mixer.free_after_resident = mixer.free_after_resident.checked_sub(withheld).with_context(|| {
            format!(
                "borrowed span slots {borrowed_bytes} plus staging {staging_bytes}, minimum slots {minimum_experts} and decode mailboxes {decode_mailbox_bytes} exceed free bytes {}",
                mixer.free_after_resident
            )
        })?;
        mixer.capacity = span_rows as usize;
        let mixer = WorkspaceBudget::with_reserve(
            mixer,
            device_headroom_bytes,
            wave_rows as usize,
            |rows| Ok(mixer_workspace(rows)?.max(wave_workspace_bytes)),
            select,
        )
        .with_context(|| {
            format!(
                "expert mixer/wave admission failed: span_rows={span_rows} wave_rows={wave_rows} borrowed_slots={borrowed_slots} minimum_expert_bytes={minimum_experts} headroom_bytes={device_headroom_bytes}"
            )
        })?;
        let workspace_rows = mixer.plan.chunk.max(wave_rows as usize);
        reserves.device_fixed_bytes = u64::try_from(
            mixer
                .planned_transient_bytes
                .checked_add(device_headroom_bytes)
                .context("expert fixed reserve overflow")?,
        )?
        .checked_add(decode_mailbox_bytes)
        .context("expert fixed decode reserve overflow")?;
        Ok(Self {
            mixer,
            workspace_rows,
            borrowed_slots,
            candidate: SpanCandidate {
                span_rows,
                wave_rows,
                reserves,
            },
        })
    }

    pub fn select(
        inputs: &crate::experts::plan::PlanInputs<'_>,
        prefill_rows: u32,
        candidates: &[Self],
    ) -> Result<(
        usize,
        crate::experts::plan::Capacity,
        crate::experts::plan::Admission,
    )> {
        Self::select_with_peer(inputs, None, prefill_rows, candidates)
    }

    pub fn select_with_peer(
        inputs: &crate::experts::plan::PlanInputs<'_>,
        peer: Option<crate::experts::plan::PeerInputs<'_>>,
        prefill_rows: u32,
        candidates: &[Self],
    ) -> Result<(
        usize,
        crate::experts::plan::Capacity,
        crate::experts::plan::Admission,
    )> {
        use crate::experts::plan::{Admission, admission_budget, select_span_with_peer};
        let budget_bytes_per_token = admission_budget(
            &inputs.geometry,
            inputs.service,
            inputs.h2d_bytes_per_second,
            inputs.d2h_bytes_per_second,
            inputs.copy_engines,
        )?;
        let mut admission = Admission {
            budget_bytes_per_token,
        };
        let span_candidates = candidates
            .iter()
            .map(|budget| budget.candidate.clone())
            .collect::<Vec<_>>();
        let (index, mut capacity) = select_span_with_peer(
            inputs,
            admission.hard_cap(inputs.layout.slot_bytes())?,
            prefill_rows,
            &span_candidates,
            peer,
        )?;
        if capacity.gpu0_layers < inputs.geometry.layers {
            admission.budget_bytes_per_token = admission_budget(
                &crate::experts::catalogue::ExpertGeometry {
                    layers: capacity.gpu0_layers,
                    ..inputs.geometry
                },
                inputs.service,
                inputs.h2d_bytes_per_second,
                inputs.d2h_bytes_per_second,
                inputs.copy_engines,
            )?;
            let (_, updated) = select_span_with_peer(
                inputs,
                admission.hard_cap(inputs.layout.slot_bytes())?,
                prefill_rows,
                &span_candidates[index..=index],
                peer,
            )?;
            capacity = updated;
        }
        Ok((index, capacity, admission))
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    fn request(capacity: usize, requested: Option<usize>, free: usize) -> WorkspaceRequest {
        WorkspaceRequest {
            capacity,
            requested,
            free_after_resident: free,
            total: 160_000,
            alignment: 32,
            row_bytes: 4,
            fixed_transient_bytes: 512,
        }
    }

    /// The default chunk is the memory-admitted row limit rounded DOWN to the
    /// alignment and then bounded by `select`; an explicit request is honoured up to
    /// the row limit WITHOUT `select`.
    #[test]
    fn default_chunk_is_bounded_by_select_and_requests_are_not() -> Result<()> {
        let cap = |admitted: usize| Ok(admitted.min(2048));
        let wide = WorkspaceBudget::new(request(262_144, None, 160_000), |r| Ok(8 * r), cap)?;
        assert_eq!(wide.plan.chunk, 2048, "the default is clamped");
        assert!(wide.plan.chunk <= wide.plan.row_limit);
        let exact = WorkspaceBudget::new(request(262_144, None, 160_000), |r| Ok(8 * r), Ok)?;
        assert!(
            exact.plan.chunk > 2048,
            "select may pass the admitted chunk through"
        );
        let explicit =
            WorkspaceBudget::new(request(262_144, Some(8192), 160_000), |r| Ok(8 * r), cap)?;
        assert_eq!(
            explicit.plan.chunk, 8192,
            "an explicit chunk is never clamped"
        );
        let over =
            WorkspaceBudget::new(request(262_144, Some(1 << 20), 160_000), |r| Ok(8 * r), cap);
        assert!(over.is_err(), "an explicit chunk still faces the row limit");
        Ok(())
    }

    #[test]
    fn default_chunk_is_the_admitted_limit_rounded_down() -> Result<()> {
        for capacity in [
            1, 17, 31, 32, 511, 512, 513, 1021, 1024, 2047, 2048, 2049, 4096, 8192,
        ] {
            for free in [11_000, 14_000, 25_000, 160_000] {
                let budget =
                    WorkspaceBudget::new(request(capacity, None, free), |r| Ok(8 * r), Ok)?;
                let chunk = budget.plan.chunk;
                assert!(chunk > 0 && chunk <= budget.plan.row_limit);
                assert!(chunk == budget.plan.row_limit || chunk.is_multiple_of(32));
                assert_eq!(budget.plan.requested_chunk, None);
                assert_eq!(budget.operator_workspace_bytes, 8 * chunk);
                assert_eq!(budget.planned_transient_bytes, 4 * chunk + 512 + 8 * chunk);
            }
        }
        Ok(())
    }

    #[test]
    fn every_budget_field_is_carried_and_requests_are_bounded() -> Result<()> {
        let budget = WorkspaceBudget::new(request(8192, Some(2048), 160_000), |r| Ok(8 * r), Ok)?;
        assert_eq!(budget.plan.chunk, 2048);
        assert_eq!(budget.plan.requested_chunk, Some(2048));
        assert_eq!(budget.plan.free_bytes, 160_000);
        assert_eq!(budget.plan.reserve_bytes, 160_000 / 16);
        assert_eq!(budget.plan.row_bytes, 4);
        assert_eq!(budget.plan.alignment, 32);
        assert_eq!(budget.accounting, "free_after_resident");
        assert!(
            WorkspaceBudget::new(request(8192, Some(1 << 20), 160_000), |r| Ok(8 * r), Ok).is_err()
        );
        assert!(WorkspaceBudget::new(request(8192, Some(0), 160_000), |r| Ok(8 * r), Ok).is_err());
        assert!(WorkspaceBudget::new(request(0, None, 160_000), |r| Ok(8 * r), Ok).is_err());
        assert!(WorkspaceBudget::new(request(8192, None, 1), |r| Ok(8 * r), Ok).is_err());
        Ok(())
    }

    #[test]
    fn expert_span_destinations_and_wave_floor_precede_cache_capacity() -> Result<()> {
        use crate::experts::{
            catalogue::ExpertGeometry,
            plan::{PlanInputs, ServiceCell, ServiceTable},
        };
        let model = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let geometry = ExpertGeometry::from_model(&model)?;
        let layout = crate::experts::catalogue::ExpertLayout::new(
            model.expert_format.context("MoE expert_format missing")?,
            &geometry,
        )?;
        let slot_bytes = layout.slot_bytes();
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model)?.row_bytes()?;
        let request = ExpertWorkspaceRequest {
            mixer: WorkspaceRequest {
                capacity: 65_536,
                requested: Some(37),
                free_after_resident: 24 << 30,
                total: 24 << 30,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: 8192,
            },
            span_rows: 65_536,
            wave_rows: 71,
            wave_workspace_bytes: 71 * 937,
            device_headroom_bytes: 4096,
            host_fixed_bytes: 16384,
            prefill_staging_slots: 2,
            disk_ring_slots: 3,
        };
        let candidates = [97, 65_536]
            .map(|span_rows| {
                ExpertWorkspaceBudget::new(
                    ExpertWorkspaceRequest {
                        span_rows,
                        ..request
                    },
                    geometry,
                    slot_bytes,
                    |r| Ok(r * 257),
                    Ok,
                )
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let budget = &candidates[1];
        let reserves = &budget.candidate.reserves;
        assert_eq!(budget.mixer.plan.chunk, 37);
        assert_eq!(budget.workspace_rows, 71);
        assert_eq!(budget.mixer.operator_workspace_bytes, 71 * 937);
        assert_eq!(reserves.residual_row_bytes, 4096);
        assert_eq!(reserves.normalized_row_bytes, 4096);
        assert_eq!(reserves.contribution_row_bytes, 32768);
        assert_eq!(reserves.done_row_bytes, 32);
        assert_eq!(reserves.route_row_bytes, 80);
        assert_eq!(budget.borrowed_slots, 428);
        let fixed = 8192 + 71 * row_bytes + 71 * 937 + 4096 + 33088;
        assert_eq!(reserves.device_fixed_bytes, fixed as u64);
        assert_eq!(reserves.span_bytes(request.span_rows)?, 65_536 * 41072);
        for rows in [1, 97, request.span_rows] {
            let layout = ExpertSpanLayout::new(reserves, geometry, slot_bytes, rows)?;
            layout.validate_wave(1, 1)?;
            assert!(layout.validate_wave(1, 0).is_err());
            if rows >= 12 {
                assert!(layout.validate_wave(12, 8).is_err());
            }
            assert_eq!(layout.bytes, rows as usize * 41072);
            let regions = [
                &layout.residual,
                &layout.normalized,
                &layout.contributions,
                &layout.done,
                &layout.ids,
                &layout.weights,
                &layout.active,
            ];
            assert!(regions.iter().all(|r| r.start % 16 == 0));
            assert!(regions.windows(2).all(|r| r[0].end <= r[1].start));
            assert_eq!(
                layout.contributions.len(),
                layout.rows as usize * 8 * 2048 * 2
            );
            assert_eq!(layout.active.end, layout.bytes);
        }
        assert_eq!(
            reserves.persistent_bytes(slot_bytes)?,
            fixed as u64 + 2 * 6_291_456
        );
        assert_eq!(reserves.host_bytes(slot_bytes)?, 16384 + 3 * 6_291_456);
        let service = ServiceTable::new(
            "fixture",
            vec![
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 1.0 / 65536.0,
                },
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 8,
                    seconds: 0.001,
                },
            ],
        )?;
        let inputs = PlanInputs {
            geometry,
            layout: layout.clone(),
            device_free_bytes: request.mixer.free_after_resident as u64,
            host_available_bytes: 48 << 30,
            h2d_bytes_per_second: (16u64 << 30) as f64,
            d2h_bytes_per_second: (16u64 << 30) as f64,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: 0,
            service: &service,
            reserves,
            span_rows: request.span_rows,
            wave_rows: request.wave_rows,
        };
        let (index, capacity, admission) =
            ExpertWorkspaceBudget::select(&inputs, 65_536, &candidates)?;
        assert_eq!(index, 1);
        assert_eq!(admission.budget_bytes_per_token, 10 << 20);
        assert_eq!(capacity.spare_blocks, 1);
        assert_eq!(capacity.span_rows, 65_536);
        assert_eq!(capacity.wave_rows, 71);
        let slots: u64 = capacity.slots_per_layer.iter().map(|&n| u64::from(n)).sum();
        let remaining = inputs.device_free_bytes - reserves.persistent_bytes(slot_bytes)?;
        assert_eq!(slots, remaining / slot_bytes);
        assert!(slots - u64::from(budget.borrowed_slots) >= u64::from(geometry.layers));
        let short = &candidates[0];
        assert_eq!(short.borrowed_slots, 1);
        assert_eq!(
            short.candidate.reserves.device_fixed_bytes,
            reserves.device_fixed_bytes
        );
        let short_capacity = crate::experts::plan::capacity(
            &PlanInputs {
                span_rows: 97,
                reserves: &short.candidate.reserves,
                ..inputs.clone()
            },
            1,
        )?;
        assert_eq!(short_capacity.slots_per_layer, capacity.slots_per_layer);
        for changed in [
            PlanInputs {
                copy_engines: 1,
                ..inputs.clone()
            },
            PlanInputs {
                d2h_bytes_per_second: (8u64 << 30) as f64,
                ..inputs.clone()
            },
        ] {
            let (_, capacity, admission) =
                ExpertWorkspaceBudget::select(&changed, 65_536, &candidates)?;
            assert_eq!(admission.budget_bytes_per_token, 5 << 20);
            assert!(capacity.host_blocks > 0);
            assert_eq!(capacity.spare_blocks, 1);
        }
        assert!(
            ExpertWorkspaceBudget::select(
                &PlanInputs {
                    d2h_bytes_per_second: f64::NAN,
                    ..inputs
                },
                65_536,
                &candidates,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn expert_workspace_exact_boundary_preserves_minimum_slots_and_headroom() -> Result<()> {
        use crate::experts::catalogue::ExpertGeometry;
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 2,
            experts: 8,
            hidden: 16,
            intermediate: 8,
            top_k: 2,
        };
        let mut request = ExpertWorkspaceRequest {
            mixer: WorkspaceRequest {
                capacity: 16,
                requested: Some(3),
                free_after_resident: 1 << 20,
                total: 1 << 20,
                alignment: 1,
                row_bytes: 128,
                fixed_transient_bytes: 1024,
            },
            span_rows: 16,
            wave_rows: 5,
            wave_workspace_bytes: 2048,
            device_headroom_bytes: 4096,
            host_fixed_bytes: 512,
            prefill_staging_slots: 2,
            disk_ring_slots: 1,
        };
        let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?
        .slot_bytes();
        let budget = ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok)?;
        let exact = budget.candidate.reserves.persistent_bytes(slot_bytes)?
            + (u64::from(budget.borrowed_slots) + u64::from(geometry.layers)) * slot_bytes;
        request.mixer.free_after_resident = usize::try_from(exact)?;
        let admitted =
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok)?;
        assert_eq!(admitted.workspace_rows, 5);
        assert_eq!(admitted.mixer.operator_workspace_bytes, 2048);
        assert_eq!(
            admitted.candidate.reserves.persistent_bytes(slot_bytes)?
                + (u64::from(admitted.borrowed_slots) + u64::from(geometry.layers)) * slot_bytes,
            exact
        );
        request.mixer.free_after_resident -= 1;
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.mixer.free_after_resident = 1 << 20;
        request.wave_rows = 17;
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.wave_rows = 5;
        request.mixer.requested = Some(17);
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.mixer.requested = Some(3);
        request.mixer.free_after_resident = request.mixer.total + 1;
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.mixer.free_after_resident = request.mixer.total;
        let oversized = ExpertGeometry {
            hidden: u32::MAX,
            experts: u32::MAX,
            top_k: u32::MAX,
            ..geometry
        };
        assert!(
            ExpertWorkspaceBudget::new(request, oversized, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );

        let peer_geom = crate::experts::catalogue::ExpertGeometry {
            first_layer: 0,
            layers: 4,
            experts: 8,
            hidden: 128,
            intermediate: 64,
            top_k: 2,
        };
        let peer_layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &peer_geom,
        )?;
        let peer_slot = peer_layout.slot_bytes();
        let mut peer_req = request;
        peer_req.mixer.requested = None;
        peer_req.mixer.free_after_resident = usize::try_from(peer_slot * 32 + (1 << 20))?;
        peer_req.mixer.total = peer_req.mixer.free_after_resident;
        let peer_cand =
            ExpertWorkspaceBudget::new(peer_req, peer_geom, peer_slot, |r| Ok(r * 13), Ok)?;
        let table = crate::experts::plan::ServiceTable::new(
            "test",
            vec![
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 0.001,
                },
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 2,
                    seconds: 0.002,
                },
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 16,
                    experts: 1,
                    seconds: 0.016,
                },
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 16,
                    experts: 2,
                    seconds: 0.032,
                },
            ],
        )?;
        let plan_inputs = crate::experts::plan::PlanInputs {
            geometry: peer_geom,
            layout: peer_layout,
            device_free_bytes: peer_cand.candidate.reserves.persistent_bytes(peer_slot)?
                + u64::from(peer_cand.borrowed_slots) * peer_slot
                + 12 * peer_slot,
            host_available_bytes: peer_cand.candidate.reserves.host_bytes(peer_slot)?
                + 64 * peer_slot,
            h2d_bytes_per_second: (peer_slot as f64) * 1000.0,
            d2h_bytes_per_second: (peer_slot as f64) * 1000.0,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: 64 * peer_slot,
            service: &table,
            reserves: &peer_cand.candidate.reserves,
            span_rows: peer_cand.candidate.span_rows,
            wave_rows: peer_cand.candidate.wave_rows,
        };
        let (_, single_cap, single_adm) =
            ExpertWorkspaceBudget::select(&plan_inputs, 16, std::slice::from_ref(&peer_cand))?;
        let xfer = [(8_192u64, 1e10, 1e10), (65_536u64, 1e10, 1e10)];
        let (_, split_cap, split_adm) = ExpertWorkspaceBudget::select_with_peer(
            &plan_inputs,
            Some(crate::experts::plan::PeerInputs {
                device_free_bytes: 16 * peer_slot,
                device_reserved_bytes: 0,
                decode_transfer: xfer[0],
                prefill_transfers: &xfer,
            }),
            16,
            std::slice::from_ref(&peer_cand),
        )?;
        assert_eq!(single_cap.gpu0_layers, 4);
        assert_eq!(split_cap.gpu0_layers, 2);
        assert_eq!(
            split_adm.budget_bytes_per_token * 2,
            single_adm.budget_bytes_per_token
        );
        Ok(())
    }
}
