//! Q38 MTP row, identity and allocation contracts, independent of CUDA execution.
use crate::{
    model_spec::{Architecture, ModelSpec},
    op_contract::{AttentionWorkspace, ExecutionPhase},
    work_layout::RowWorkSpec,
};
use anyhow::{Context, Result, ensure};

pub mod batched;
pub mod transaction;

pub(crate) const MAX_VERIFY_ROWS: usize = 5;

/// Target verification used for a complete MTP request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum VerifyPolicy {
    #[cfg(test)]
    Decode,
    #[default]
    Batched,
}

pub fn check_request(prompt: usize, count: usize, capacity: usize) -> Result<()> {
    ensure!(
        prompt > 0 && count > 0 && prompt.checked_add(count).is_some_and(|n| n <= capacity),
        "MTP request needs a nonempty prompt, positive output count and a reserved next-token slot"
    );
    Ok(())
}

/// Token publication times exclude model setup and graph capture.
pub struct Output {
    pub ids: Vec<u32>,
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MtpOptions {
    pub depth: usize,
    pub hidden_tap: MtpHiddenTap,
    pub recurrent_tap: MtpRecurrentTap,
    pub graph: bool,
}
impl MtpOptions {
    pub fn validate(self) -> Result<()> {
        self.verify_rows(None).map(|_| ())
    }

    pub(crate) fn verify_rows(self, chunk: Option<usize>) -> Result<usize> {
        ensure!(
            (1..MAX_VERIFY_ROWS).contains(&self.depth),
            "MTP chain depth must be 1..=4"
        );
        let rows = self.depth + 1;
        if let Some(chunk) = chunk {
            ensure!(
                chunk >= rows,
                "MTP chunk {chunk} cannot fit {rows} verification rows for depth {}",
                self.depth
            );
        }
        Ok(rows)
    }
}

pub const HIDDEN: usize = 5120;
pub const VOCAB: usize = 248320;
pub const WEIGHT_BYTES: usize = 477_199_744;
pub const PROBE_WEIGHT_BYTES: usize = WEIGHT_BYTES + 2 * VOCAB * HIDDEN * 2;
pub const ARENA_BYTES: usize = 744_488_960;
pub const KV_ROW_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MtpHiddenTap {
    PreFinalNorm,
    PostFinalNorm,
}
impl MtpHiddenTap {
    pub fn fixture_key(self) -> &'static str {
        match self {
            Self::PreFinalNorm => "target.head_input.prenorm",
            Self::PostFinalNorm => "target.head_input.postnorm",
        }
    }
    pub fn logits_key(self) -> &'static str {
        match self {
            Self::PreFinalNorm => "mtp.logits",
            Self::PostFinalNorm => "mtp.logits.postnorm_h2",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MtpRecurrentTap {
    PreMtpNorm,
    PostMtpNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpPlan {
    pub hidden_tap: MtpHiddenTap,
    pub recurrent_tap: MtpRecurrentTap,
    pub max_rows: usize,
    pub capacity: usize,
    pub phase: ExecutionPhase,
}
impl MtpPlan {
    pub const CONTRACT: &'static str = "q38-mtp-v1-emb-hidden-rope-j-slot-jminus1-rms1plusw";
    pub fn validate(&self, spec: &ModelSpec) -> Result<()> {
        ensure!(
            spec.architecture == Architecture::Qwen35Text
                && spec.hidden as usize == HIDDEN
                && spec.vocab as usize == VOCAB
                && spec.intermediate == 17408
                && spec.q_heads == 24
                && spec.kv_heads == 4
                && spec.head_dim == 256
                && spec.q_width == 6144
                && spec.rotary_dim == 64
                && spec.rms_eps == 1e-6
                && spec.rope_theta == 1e7,
            "MTP requires Q38 geometry"
        );
        ensure!(
            self.capacity >= 32
                && self.capacity <= spec.max_position as usize
                && self.capacity < i32::MAX as usize,
            "invalid MTP capacity/absolute position range"
        );
        ensure!(
            self.max_rows > 0
                && self.max_rows <= self.capacity
                && (self.phase != ExecutionPhase::Decode || self.max_rows == 1),
            "invalid MTP rows/phase"
        );
        Ok(())
    }
    pub fn check_rows(&self, prefix: usize, rows: usize) -> Result<()> {
        ensure!(
            rows > 0
                && rows <= self.max_rows
                && prefix
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
            "MTP write exceeds plan/cache capacity"
        );
        Ok(())
    }
    pub fn check_absolute_rows(
        &self,
        prefix: usize,
        rows: usize,
        max_position: usize,
    ) -> Result<()> {
        self.check_rows(prefix, rows)?;
        ensure!(
            prefix
                .checked_add(rows)
                .is_some_and(|last| last < max_position),
            "MTP shifted absolute position exceeds model limit"
        );
        Ok(())
    }
    pub fn key_chunk(&self) -> u32 {
        256.min((self.capacity / 32 * 32) as u32)
    }
    pub fn memory(&self, spec: &ModelSpec) -> Result<MtpMemory> {
        self.memory_with_head_rows(spec, self.max_rows)
    }
    pub fn memory_with_head_rows(&self, spec: &ModelSpec, head_rows: usize) -> Result<MtpMemory> {
        self.validate(spec)?;
        ensure!(
            head_rows > 0 && head_rows <= self.max_rows,
            "invalid MTP head capacity"
        );
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .context("MTP memory multiplication overflow")
        };
        let rows = self.max_rows;
        let kv = mul(self.capacity, KV_ROW_BYTES)?;
        let row_work = mul(rows, runtime_row_bytes(spec)?)?;
        // Work includes one logits row plus token/position/length. Extra storage
        // holds concat, embedding, incoming hidden, all-row head and positions.
        let extra = mul(head_rows, VOCAB * 2)?;
        let fixed = VOCAB * 2 + 12 + 64 * 8 + 32 + 8;
        let scratch = scratch_bytes(self.capacity, rows)?;
        let blas = 4 * 1024 * 1024;
        let total = [
            PROBE_WEIGHT_BYTES,
            ARENA_BYTES,
            kv,
            row_work,
            extra,
            fixed,
            scratch,
            blas,
        ]
        .into_iter()
        .try_fold(0usize, |a, b| {
            a.checked_add(b).context("MTP total memory overflow")
        })?;
        Ok(MtpMemory {
            weights: PROBE_WEIGHT_BYTES,
            arena: ARENA_BYTES,
            kv,
            work: row_work + extra + fixed,
            scratch,
            blas,
            total,
        })
    }
}

fn runtime_row_bytes(spec: &ModelSpec) -> Result<usize> {
    RowWorkSpec::new(spec)?
        .row_bytes()?
        .checked_add(4 * HIDDEN * 2 + 3 * 4 + 24 * 4)
        .context("MTP runtime row bytes overflow")
}
fn scratch_bytes(capacity: usize, rows: usize) -> Result<usize> {
    let prep = rows
        .checked_mul(2048)
        .and_then(|n| n.checked_add(255))
        .context("MTP scratch overflow")?
        & !255;
    let repair = usize::try_from(
        AttentionWorkspace::new(
            &crate::op_contract::FullAttentionSpec::q38(),
            rows.min(4) as u32,
            capacity as u32,
            256.min((capacity / 32 * 32) as u32),
        )?
        .bytes,
    )?;
    Ok(prep.max(repair))
}
#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) struct AttachmentBudget {
    pub fixed_bytes: usize,
    pub row_bytes: usize,
    capacity: usize,
}
#[cfg(any(feature = "cuda", feature = "rocm", test))]
impl AttachmentBudget {
    pub fn new(spec: &ModelSpec, capacity: usize) -> Result<Self> {
        let row_bytes = runtime_row_bytes(spec)?
            .checked_add(2 * HIDDEN * 2)
            .context("MTP bootstrap row overflow")?;
        let scratch = scratch_bytes(capacity, 1)?;
        let fixed_bytes = attachment_bytes(spec, capacity, 1, true)?
            .checked_sub(row_bytes)
            .and_then(|n| n.checked_sub(scratch))
            .context("MTP fixed budget underflow")?;
        Ok(Self {
            fixed_bytes,
            row_bytes,
            capacity,
        })
    }
    pub fn workspace(&self, rows: usize) -> Result<usize> {
        scratch_bytes(self.capacity, rows)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn checkpoint_bytes(spec: &ModelSpec, rows: usize) -> Result<usize> {
    use crate::{model_spec::Mixer, op_contract::GdnSpec};
    ensure!(rows > 0, "empty checkpoint window");
    let kv = rows
        .checked_mul(spec.kv_heads as usize)
        .and_then(|n| n.checked_mul(spec.head_dim as usize))
        .and_then(|n| n.checked_mul(4))
        .context("checkpoint KV overflow")?;
    let gdn = if spec.layers.iter().any(|l| l.mixer == Mixer::Gdn) {
        let g = GdnSpec::from_model(spec)?;
        usize::try_from(g.image_bytes()?)?
    } else {
        0
    };
    let mut bytes: usize = if spec.gdn.is_some() { 36 } else { 4 };
    for layer in &spec.layers {
        bytes = bytes
            .checked_add(match layer.mixer {
                Mixer::FullAttention => kv,
                Mixer::Gdn => gdn,
            })
            .context("checkpoint byte count overflow")?;
    }
    Ok(bytes)
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn component_spec(spec: &ModelSpec) -> Result<ModelSpec> {
    let mut component = spec.clone();
    let ffn = if spec.moe.is_some() {
        crate::model_spec::FfnKind::Moe
    } else {
        crate::model_spec::FfnKind::Dense
    };
    component.layers = vec![crate::model_spec::LayerPlan {
        geometry: None,
        mixer: crate::model_spec::Mixer::FullAttention,
        state_index: 0,
        rope: true,
        sliding_window: None,
        ffn,
    }];
    if spec.architecture == crate::model_spec::Architecture::Glm5Next {
        component.gdn = None;
    }
    Ok(component)
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn snapshot_bytes_for_rows(spec: &ModelSpec, verify_rows: usize) -> Result<usize> {
    ensure!(
        (2..=MAX_VERIFY_ROWS).contains(&verify_rows),
        "invalid MTP snapshot verify_rows={verify_rows}"
    );
    let layers = spec
        .layers
        .iter()
        .filter(|l| l.mixer == crate::model_spec::Mixer::Gdn)
        .count();
    if layers == 0 {
        return Ok(0);
    }
    let g = crate::op_contract::GdnSpec::from_model(spec)?;
    usize::try_from(
        g.image_bytes()?
            .checked_mul((verify_rows - 1) as u64)
            .context("snapshot byte overflow")?,
    )?
    .checked_mul(layers)
    .context("snapshot layer byte overflow")
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn snapshot_bytes(spec: &ModelSpec) -> Result<usize> {
    snapshot_bytes_for_rows(spec, MAX_VERIFY_ROWS)
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlmMtpCommitPlan {
    pub accepted: usize,
    pub consumed: usize,
    pub next_position: usize,
    pub dsa_committed_keys: usize,
    pub select_prefix_rows: Option<usize>,
    pub replay_target_passes: usize,
    pub commit_mtp_row0: bool,
    pub carry_verify_row1_hidden: bool,
    pub next_pending_token: u32,
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn glm_mtp_commit_plan(
    start_pos: usize,
    draft_token: u32,
    v0: u32,
    v1: u32,
) -> GlmMtpCommitPlan {
    if draft_token == v0 {
        GlmMtpCommitPlan {
            accepted: 1,
            consumed: 2,
            next_position: start_pos + 2,
            dsa_committed_keys: start_pos + 2,
            select_prefix_rows: None,
            replay_target_passes: 0,
            commit_mtp_row0: true,
            carry_verify_row1_hidden: true,
            next_pending_token: v1,
        }
    } else {
        GlmMtpCommitPlan {
            accepted: 0,
            consumed: 1,
            next_position: start_pos + 1,
            dsa_committed_keys: start_pos + 1,
            select_prefix_rows: Some(1),
            replay_target_passes: 0,
            commit_mtp_row0: false,
            carry_verify_row1_hidden: false,
            next_pending_token: v0,
        }
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GlmMtpDsaTruncation {
    pub restore_slots: Vec<usize>,
    pub incomplete_pool: Option<usize>,
    pub restore_pools: Vec<usize>,
    pub meta: crate::op_contract::DsaMeta,
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn glm_mtp_dsa_truncation(
    spec: crate::op_contract::DsaSpec,
    capacity: usize,
    prefix: usize,
    active_rows: usize,
    accepted: usize,
) -> Result<GlmMtpDsaTruncation> {
    let cap_u32 = u32::try_from(capacity)?;
    ensure!(
        accepted > 0 && accepted < active_rows && prefix + active_rows <= capacity,
        "invalid DSA prefix truncation prefix={prefix} active_rows={active_rows} accepted={accepted} capacity={capacity}"
    );
    let pool_size = spec.pool as usize;
    let pool_cap = spec.pool_capacity(cap_u32) as usize;
    let restore_slots = (accepted..active_rows)
        .map(|row| (prefix + row) % capacity)
        .collect();
    let new_keys = prefix + accepted;
    let verified_keys = prefix + active_rows;
    let incomplete_pool =
        (!new_keys.is_multiple_of(pool_size)).then_some((new_keys / pool_size) % pool_cap);
    let restore_pools = (new_keys.div_ceil(pool_size)..verified_keys.div_ceil(pool_size))
        .map(|p| p % pool_cap)
        .collect();
    let meta = crate::op_contract::DsaMeta {
        prefix: i32::try_from(prefix)?,
        valid_rows: i32::try_from(accepted)?,
        key_count: i32::try_from(new_keys)?,
        capacity: i32::try_from(capacity)?,
        first_key: 0,
        reserved: [0; 3],
    };
    meta.validate(cap_u32)?;
    Ok(GlmMtpDsaTruncation {
        restore_slots,
        incomplete_pool,
        restore_pools,
        meta,
    })
}

#[cfg(test)]
pub(crate) fn glm_mtp_break_even_alpha(
    b_draft: u64,
    mean_b_verify: f64,
    b_step1: u64,
    replay_on_reject: bool,
) -> f64 {
    if replay_on_reject {
        (b_draft as f64 + mean_b_verify) / (2.0 * b_step1 as f64)
    } else {
        (b_draft as f64 + mean_b_verify) / b_step1 as f64 - 1.0
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn glm_attachment_bytes(
    spec: &ModelSpec,
    capacity: usize,
    verify_rows: usize,
) -> Result<usize> {
    ensure!(
        spec.architecture == Architecture::Glm5Next,
        "GLM MTP attachment requires Glm5Next"
    );
    let dsa = crate::op_contract::DsaSpec::from_model(spec)?;
    let gdn = crate::op_contract::GdnSpec::from_model(spec)?;
    let moe = spec.moe.context("GLM MoE specification missing")?;
    let h = spec.hidden as usize;
    let vocab = spec.vocab as usize;
    let inter = moe.expert_intermediate as usize;
    let experts = moe.experts as usize;
    let top_k = moe.experts_per_tok as usize;
    let dsa_bytes = usize::try_from(dsa.state_bytes(u32::try_from(capacity)?)?)?;
    let gdn_bytes = usize::try_from(gdn.image_bytes()?)?;
    let gdn_layers = spec
        .layers
        .iter()
        .filter(|l| l.mixer == crate::model_spec::Mixer::Gdn)
        .count();
    let dsa_layers = spec.layers.len() - gdn_layers;
    let expert_proj_bytes = inter * h + (inter / 128) * (h / 128) * 4;
    let runtime_weights = (5 * h
        + h * 2 * h
        + 512 * h
        + 512
        + 32768 * 512
        + 1536 * h
        + 1536
        + 16384 * 1536
        + h * 16384
        + experts * h)
        * 2
        + experts * 4
        + 3 * expert_proj_bytes
        + experts * 6 * 8
        + experts * 4
        + top_k * 3 * expert_proj_bytes;
    let native_moe_shared = 2 * (h + (h / 128) * 4) + (inter + (inter / 128) * 4);
    let native_moe_buffers = experts * 4
        + 2 * top_k * 4
        + h
        + (h / 128) * 4
        + 3 * top_k * inter * 2
        + top_k * inter
        + top_k * (inter / 128) * 4
        + 2 * top_k * h * 2
        + 3 * inter * 2
        + h * 2
        + 2 * top_k * 4
        + 2 * 8 * 4
        + 4
        + 4 * 1024 * 1024;
    let runtime_scratch =
        (12 * h + 2 * h + 2 * 512 + 2 * 1536 + 2 * 16384) * 2 + vocab * 4 + 2051 * 4;
    [
        runtime_weights,
        native_moe_shared,
        native_moe_buffers,
        runtime_scratch,
        dsa_bytes,
        gdn_layers * gdn_bytes + dsa_layers * dsa_bytes + 36,
        dsa_bytes + 4,
        snapshot_bytes_for_rows(spec, verify_rows)?,
        verify_rows * vocab * 4,
        (4 * HIDDEN + vocab) * 2 + 8,
    ]
    .into_iter()
    .try_fold(0usize, |sum, n| {
        sum.checked_add(n)
            .context("GLM MTP attachment byte overflow")
    })
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn batch_bytes(spec: &ModelSpec) -> Result<usize> {
    checkpoint_bytes(spec, 5)?
        .checked_add(snapshot_bytes(spec)?)
        .and_then(|n| n.checked_add((5 * VOCAB + 6 * HIDDEN) * 2 + 60))
        .context("MTP batch buffer overflow")
}

/// Additional resident and workspace bytes, excluding the Engine's shared BLAS.
/// The ordinary workspace planner applies its existing reserve to this total.
#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn attachment_bytes(
    spec: &ModelSpec,
    capacity: usize,
    rows: usize,
    batched: bool,
) -> Result<usize> {
    let plan = MtpPlan {
        hidden_tap: MtpHiddenTap::PostFinalNorm,
        recurrent_tap: MtpRecurrentTap::PostMtpNorm,
        max_rows: rows,
        capacity,
        phase: ExecutionPhase::Prefill,
    };
    let memory = plan.memory_with_head_rows(spec, 1)?;
    let bootstrap = rows
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(HIDDEN * 2))
        .context("MTP bootstrap buffer overflow")?;
    [
        memory.total - memory.weights - memory.blas,
        WEIGHT_BYTES,
        checkpoint_bytes(spec, 5)?,
        checkpoint_bytes(&component_spec(spec)?, 5)?,
        bootstrap,
        (4 * HIDDEN + VOCAB) * 2 + 8,
        if batched { batch_bytes(spec)? } else { 0 },
    ]
    .into_iter()
    .try_fold(0usize, |sum, n| {
        sum.checked_add(n).context("MTP attachment byte overflow")
    })
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct MtpMemory {
    pub weights: usize,
    pub arena: usize,
    pub kv: usize,
    pub work: usize,
    pub scratch: usize,
    pub blas: usize,
    pub total: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpRow {
    pub hidden_row: usize,
    pub token: u32,
    pub position: i32,
    pub slot: usize,
}
#[cfg(test)]
pub fn teacher_forced_rows(sequence: &[u32]) -> Result<Vec<MtpRow>> {
    ensure!(
        sequence.len() >= 2 && sequence.len() <= i32::MAX as usize,
        "MTP fixture needs at least two tokens within the position range"
    );
    ensure!(
        sequence.iter().all(|&t| (t as usize) < VOCAB),
        "MTP token outside vocabulary"
    );
    Ok(sequence
        .iter()
        .copied()
        .enumerate()
        .skip(1)
        .map(|(j, token)| MtpRow {
            hidden_row: j - 1,
            token,
            position: j as i32,
            slot: j - 1,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verify_rows_fit_the_selected_depth_and_chunk() -> Result<()> {
        for depth in 1..MAX_VERIFY_ROWS {
            let options = MtpOptions {
                depth,
                hidden_tap: MtpHiddenTap::PostFinalNorm,
                recurrent_tap: MtpRecurrentTap::PostMtpNorm,
                graph: false,
            };
            assert_eq!(options.verify_rows(None)?, depth + 1);
            assert_eq!(options.verify_rows(Some(depth + 1))?, depth + 1);
            assert!(options.verify_rows(Some(depth)).is_err());
            assert_eq!(options.verify_rows(Some(4)).is_ok(), depth <= 3);
        }
        Ok(())
    }

    fn spec() -> ModelSpec {
        ModelSpec::qwen(
            &serde_json::from_str(include_str!("../tests/data/q38-config.json")).unwrap(),
            &serde_json::Value::Null,
        )
        .unwrap()
    }

    fn glm_spec() -> ModelSpec {
        ModelSpec::glm5_next(
            &serde_json::from_str(include_str!("../tests/data/glm53-flash-config.json")).unwrap(),
            &serde_json::json!({"eos_token_id": [154820, 154827, 154829]}),
        )
        .unwrap()
    }

    #[test]
    fn mtp_component_uses_full_attention() {
        let component = component_spec(&spec()).unwrap();
        assert_eq!(component.layers.len(), 1);
        let layer = &component.layers[0];
        assert_eq!(layer.mixer, crate::model_spec::Mixer::FullAttention);
        assert_eq!(layer.state_index, 0);
        assert!(layer.rope);
        assert_eq!(layer.sliding_window, None);
        assert_eq!(layer.ffn, crate::model_spec::FfnKind::Dense);
        assert!(component.gdn.is_some());

        let glm_component = component_spec(&glm_spec()).unwrap();
        assert!(glm_component.gdn.is_none());
    }

    #[test]
    fn glm_mtp_rejection_commits_prefix_one_without_target_replay() -> Result<()> {
        let reject = glm_mtp_commit_plan(33, 9001, 42, 77);
        assert_eq!(reject.accepted, 0);
        assert_eq!(reject.consumed, 1);
        assert_eq!(reject.next_position, 34);
        assert_eq!(reject.dsa_committed_keys, 34);
        assert_eq!(reject.select_prefix_rows, Some(1));
        assert_eq!(reject.replay_target_passes, 0);
        assert!(!reject.commit_mtp_row0);
        assert!(!reject.carry_verify_row1_hidden);
        assert_eq!(reject.next_pending_token, 42);

        let accept = glm_mtp_commit_plan(33, 42, 42, 77);
        assert_eq!(accept.accepted, 1);
        assert_eq!(accept.consumed, 2);
        assert_eq!(accept.next_position, 35);
        assert_eq!(accept.dsa_committed_keys, 35);
        assert_eq!(accept.select_prefix_rows, None);
        assert_eq!(accept.replay_target_passes, 0);
        assert!(accept.commit_mtp_row0);
        assert!(accept.carry_verify_row1_hidden);
        assert_eq!(accept.next_pending_token, 77);

        let dsa = crate::op_contract::DsaSpec::glm53();
        let incomplete = glm_mtp_dsa_truncation(dsa, 64, 2, 2, 1)?;
        assert_eq!(incomplete.restore_slots, vec![3]);
        assert_eq!(incomplete.incomplete_pool, Some(0));
        assert!(incomplete.restore_pools.is_empty());
        assert_eq!(incomplete.meta.prefix, 2);
        assert_eq!(incomplete.meta.valid_rows, 1);
        assert_eq!(incomplete.meta.key_count, 3);

        let completed = glm_mtp_dsa_truncation(dsa, 64, 3, 2, 1)?;
        assert_eq!(completed.restore_slots, vec![4]);
        assert_eq!(completed.incomplete_pool, None);
        assert_eq!(completed.restore_pools, vec![1]);
        assert_eq!(completed.meta.prefix, 3);
        assert_eq!(completed.meta.valid_rows, 1);
        assert_eq!(completed.meta.key_count, 4);

        let glm = glm_spec();
        assert_eq!(snapshot_bytes_for_rows(&glm, 2)?, 152_633_344);
        assert!(glm_attachment_bytes(&glm, 64, 2)? > 152_633_344);

        let b_step1 = 15_344_926_720u64;
        let b_draft = 1_714_025_088u64;
        let mean_b_verify = 21_072_128_530.0;
        let alpha_star_no_replay = glm_mtp_break_even_alpha(b_draft, mean_b_verify, b_step1, false);
        let alpha_star_replay = glm_mtp_break_even_alpha(b_draft, mean_b_verify, b_step1, true);
        assert!(alpha_star_no_replay < 0.50);
        assert!(alpha_star_replay > 0.74);
        Ok(())
    }

    fn plan() -> MtpPlan {
        MtpPlan {
            hidden_tap: MtpHiddenTap::PreFinalNorm,
            recurrent_tap: MtpRecurrentTap::PostMtpNorm,
            max_rows: 116,
            capacity: 128,
            phase: ExecutionPhase::Prefill,
        }
    }
    #[test]
    fn fixture_rows_are_shifted_once() {
        let rows = teacher_forced_rows(&[101, 202, 303]).unwrap();
        assert_eq!(
            rows,
            vec![
                MtpRow {
                    hidden_row: 0,
                    token: 202,
                    position: 1,
                    slot: 0
                },
                MtpRow {
                    hidden_row: 1,
                    token: 303,
                    position: 2,
                    slot: 1
                }
            ]
        );
        assert!(teacher_forced_rows(&[1]).is_err());
        assert!(teacher_forced_rows(&[1, VOCAB as u32]).is_err());
    }
    #[test]
    fn memory_and_capacity_are_checked() {
        let p = plan();
        let m = p.memory(&spec()).unwrap();
        assert_eq!(m.weights, 5_562_793_344);
        assert_eq!(m.kv, 128 * 4096);
        assert!(m.total > m.weights + m.arena + m.kv);
        assert!(p.check_rows(127, 1).is_ok());
        assert!(p.check_rows(128, 1).is_err());
        assert!(p.check_rows(usize::MAX, 1).is_err());
        for (max_rows, capacity, phase) in [
            (0, 128, ExecutionPhase::Prefill),
            (129, 128, ExecutionPhase::Prefill),
            (2, 128, ExecutionPhase::Decode),
            (1, usize::MAX, ExecutionPhase::Prefill),
        ] {
            assert!(
                MtpPlan {
                    max_rows,
                    capacity,
                    phase,
                    ..p
                }
                .memory(&spec())
                .is_err()
            );
        }
        use crate::checkpoint::{LoadPolicy as L, TensorScope as S};
        assert!(L::MtpProbe.selects_tensor(S::Text, "lm_head.weight"));
        assert!(!L::MtpProbe.selects_tensor(S::Text, "model.language_model.norm.weight"));
        assert!(!L::MtpProbe.selects_tensor(
            S::Text,
            "model.language_model.layers.0.input_layernorm.weight"
        ));
        assert!(L::MtpProbe.selects(S::Mtp));
        assert!(!L::MtpProbe.selects(S::Vision));
        assert!(!L::TextOnly.selects(S::Mtp));
        assert!(L::TextWithMtp.selects(S::Text));
    }

    #[test]
    fn canonical_repair_scratch_covers_bound_rows() -> Result<()> {
        for capacity in [32, 64, 128, 4096, 18432] {
            for rows in [1, 2, 3, 4, 16, 53, 116, 2048] {
                if rows > capacity {
                    continue;
                }
                let bytes = scratch_bytes(capacity, rows)?;
                assert!(bytes.is_multiple_of(256));
                for repair_rows in 1..=rows.min(4) {
                    let required = AttentionWorkspace::new(
                        &crate::op_contract::FullAttentionSpec::q38(),
                        repair_rows as u32,
                        capacity as u32,
                        256.min((capacity / 32 * 32) as u32),
                    )?
                    .bytes as usize;
                    assert!(
                        bytes >= required,
                        "capacity={capacity} bound_rows={rows} repair_rows={repair_rows}: allocated={bytes} required={required}"
                    );
                }
            }
        }
        assert_eq!(scratch_bytes(64, 1)?, 27_136);
        assert_eq!(scratch_bytes(64, 2)?, 53_760);
        assert_eq!(scratch_bytes(64, 3)?, 80_896);
        Ok(())
    }

    #[test]
    fn maximum_capacity_checks_shifted_positions_at_execution() -> Result<()> {
        let mut s = spec();
        s.max_position = 128;
        let p = MtpPlan {
            max_rows: 16,
            capacity: 128,
            ..plan()
        };
        p.validate(&s)?;
        p.check_absolute_rows(125, 2, 128)?;
        assert!(p.check_absolute_rows(126, 2, 128).is_err());
        assert!(p.check_absolute_rows(usize::MAX, 1, 128).is_err());
        assert!(MtpPlan { capacity: 129, ..p }.validate(&s).is_err());
        assert!(p.check_rows(127, 1).is_ok());
        Ok(())
    }

    #[test]
    fn snapshot_and_attachment_bytes_share_the_workspace_budget() -> Result<()> {
        let s = spec();
        assert_eq!(snapshot_bytes(&s)?, 627572736);
        for capacity in [4096, 131072] {
            let budget = AttachmentBudget::new(&s, capacity)?;
            for rows in [4, 5] {
                assert_eq!(
                    budget.fixed_bytes + rows * budget.row_bytes + budget.workspace(rows)?,
                    attachment_bytes(&s, capacity, rows, true)?
                );
            }
        }
        let p = MtpPlan {
            max_rows: 16,
            capacity: 128,
            ..plan()
        };
        assert_eq!(
            p.memory(&s)?.total - p.memory_with_head_rows(&s, 1)?.total,
            15 * VOCAB * 2
        );
        let b = AttachmentBudget::new(&s, 128)?;
        for rows in [1, 2, 5, 15, 16, 32, 128] {
            assert_eq!(
                b.fixed_bytes + rows * b.row_bytes + b.workspace(rows)?,
                attachment_bytes(&s, 128, rows, true)?
            );
        }
        let total = 8usize << 30;
        let request = crate::prefill::WorkspaceRequest {
            capacity: 128,
            requested: None,
            total,
            alignment: 8,
            free_after_resident: total / 16
                + b.fixed_bytes
                + 20 * (64 + b.row_bytes)
                + b.workspace(20)?
                + 1024,
            row_bytes: 64,
            fixed_transient_bytes: 0,
        };
        let plain = crate::prefill::WorkspaceBudget::new(request, |_| Ok(1024), Ok)?;
        let combined = crate::prefill::WorkspaceBudget::new(
            crate::prefill::WorkspaceRequest {
                row_bytes: request.row_bytes + b.row_bytes,
                fixed_transient_bytes: b.fixed_bytes,
                ..request
            },
            |rows| Ok(1024 + b.workspace(rows)?),
            Ok,
        )?;
        assert!(combined.plan.chunk < plain.plan.chunk);
        assert_eq!(combined.plan.chunk, 16);
        assert_eq!(combined.plan.reserve_bytes, total / 16);
        assert!(
            combined.planned_transient_bytes + combined.plan.reserve_bytes
                <= request.free_after_resident
        );
        assert!(
            crate::prefill::WorkspaceBudget::new(
                crate::prefill::WorkspaceRequest {
                    requested: Some(32),
                    row_bytes: request.row_bytes + b.row_bytes,
                    fixed_transient_bytes: b.fixed_bytes,
                    ..request
                },
                |rows| Ok(1024 + b.workspace(rows)?),
                Ok,
            )
            .is_err()
        );
        Ok(())
    }
}

#[cfg(test)]
mod resident_tests {
    use super::*;
    #[test]
    fn request_reserves_next_slot_and_rejects_overflow() {
        assert!(check_request(31, 1, 32).is_ok());
        for (p, n, c) in [
            (0, 1, 32),
            (1, 0, 32),
            (32, 1, 32),
            (usize::MAX, 1, usize::MAX),
        ] {
            assert!(check_request(p, n, c).is_err());
        }
    }
}
