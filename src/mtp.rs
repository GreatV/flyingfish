//! Q38 MTP row, identity and allocation contracts, independent of CUDA execution.
use crate::{
    model_spec::{Architecture, ModelSpec},
    op_contract::{AttentionWorkspace, ExecutionPhase},
    work_layout::RowWorkSpec,
};
use anyhow::{Context, Result, ensure};

pub mod batched;
pub mod transaction;

/// Target verification used for a complete MTP request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum VerifyPolicy {
    #[cfg(test)]
    Decode,
    #[default]
    Batched,
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
        ensure!(
            (1..=3).contains(&self.depth),
            "MTP chain depth must be 1..=3"
        );
        Ok(())
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
                && self.capacity < spec.max_position as usize
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
    pub fn key_chunk(&self) -> u32 {
        256.min((self.capacity / 32 * 32) as u32)
    }
    pub fn memory(&self, spec: &ModelSpec) -> Result<MtpMemory> {
        self.validate(spec)?;
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .context("MTP memory multiplication overflow")
        };
        let rows = self.max_rows;
        let kv = mul(self.capacity, KV_ROW_BYTES)?;
        let row_work = mul(rows, RowWorkSpec::new(spec)?.row_bytes()?)?;
        // Work includes one logits row plus token/position/length. Extra storage
        // holds concat, embedding, incoming hidden, all-row head and positions.
        let extra = mul(rows, 4 * HIDDEN * 2 + VOCAB * 2 + 3 * 4 + 24 * 4)?;
        let fixed = VOCAB * 2 + 12 + 64 * 8 + 32;
        let prep = (mul(rows, 2048)?
            .checked_add(255)
            .context("MTP scratch overflow")?)
            & !255;
        let decode = usize::try_from(
            AttentionWorkspace::q38(1, self.capacity as u32, self.key_chunk())?.bytes,
        )?;
        let scratch = prep.max(decode);
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
    fn spec() -> ModelSpec {
        ModelSpec::qwen(
            &serde_json::from_str(include_str!("../tests/data/q38-config.json")).unwrap(),
            &serde_json::Value::Null,
        )
        .unwrap()
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
}
