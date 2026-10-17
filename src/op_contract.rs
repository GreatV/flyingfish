//! CPU-side invariants shared by operator binders.
use anyhow::{Context, Result, ensure};

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SequenceMeta {
    pub prefix: i32,
    pub valid_rows: i32,
    pub key_count: i32,
    pub capacity: i32,
    pub causal: i32,
    pub reserved: [i32; 3],
}
impl SequenceMeta {
    pub fn validate(&self, capacity: u32, max_rows: u32) -> Result<()> {
        ensure!(
            capacity <= i32::MAX as u32 && capacity > 0,
            "invalid capacity"
        );
        ensure!(
            self.capacity == capacity as i32 && self.causal == 1,
            "sequence plan mismatch"
        );
        ensure!(
            self.prefix >= 0 && self.valid_rows > 0 && self.valid_rows as u32 <= max_rows,
            "invalid active rows"
        );
        ensure!(
            self.prefix.checked_add(self.valid_rows) == Some(self.key_count)
                && self.key_count <= self.capacity,
            "invalid key_count"
        );
        ensure!(self.reserved == [0; 3], "nonzero reserved metadata");
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AttentionWorkspace {
    pub max_parts: u64,
    pub segment_rows: u64,
    pub p_max: u64,
    pub p_sum: u64,
    pub p_out: u64,
    pub k_norm: u64,
    pub bytes: u64,
    pub alignment: u64,
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .context("workspace multiplication overflow")
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("workspace addition overflow")
}
fn align(n: u64) -> Result<u64> {
    Ok(add(n, 255)? & !255)
}
impl AttentionWorkspace {
    pub fn q38(max_rows: u32, capacity: u32, key_chunk: u32) -> Result<Self> {
        ensure!(
            max_rows > 0 && max_rows <= capacity && capacity <= i32::MAX as u32,
            "invalid attention limits"
        );
        ensure!(
            key_chunk > 0 && key_chunk.is_multiple_of(32),
            "key_chunk must be a positive multiple of narrow KT=32"
        );
        let max_parts = u64::from(capacity).div_ceil(u64::from(key_chunk));
        let segment_rows = u64::from(max_rows.min(64));
        let e = mul(mul(segment_rows, 24)?, max_parts)?;
        let scalar = align(mul(e, 4)?)?;
        let p_out = mul(scalar, 2)?;
        let k_norm = add(p_out, align(mul(mul(e, 256)?, 4)?)?)?;
        let bytes = add(k_norm, align(mul(u64::from(max_rows), 2048)?)?)?;
        Ok(Self {
            max_parts,
            segment_rows,
            p_max: 0,
            p_sum: scalar,
            p_out,
            k_norm,
            bytes,
            alignment: 256,
        })
    }
}
const _: () = assert!(std::mem::size_of::<SequenceMeta>() == 32);
const _: () = assert!(std::mem::offset_of!(SequenceMeta, causal) == 16);

#[derive(Clone, Copy, Debug)]
pub struct FullAttentionSpec {
    pub hq: u32,
    pub hkv: u32,
    pub d: u32,
    pub scale: f32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
}
impl FullAttentionSpec {
    pub const fn q38() -> Self {
        Self {
            hq: 24,
            hkv: 4,
            d: 256,
            scale: 0.0625,
            rms_eps: 1e-6,
            rope_theta: 1e7,
            rotary_dim: 64,
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hq == 24
                && self.hkv == 4
                && self.d == 256
                && self.scale == 0.0625
                && self.rms_eps == 1e-6
                && self.rope_theta == 1e7
                && self.rotary_dim == 64,
            "unsupported full-attention specification"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AttentionLimits {
    pub max_rows: u32,
    pub capacity: u32,
    pub key_chunk: u32,
    pub causal: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceReq {
    pub bytes: u64,
    pub alignment: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug)]
pub struct GdnSpec {
    pub hk: u32,
    pub hv: u32,
    pub dk: u32,
    pub dv: u32,
    pub conv_width: u32,
    pub history_width: u32,
    pub chunk: u32,
    pub prefill_value_split: u32,
    pub decode_value_split: u32,
    pub rms_eps: f32,
    pub l2_eps: f32,
}
impl GdnSpec {
    pub const PLAN_ID: &'static str = "gdn-plan-v2-c32-vs1-fp32";
    pub const fn q38() -> Self {
        Self {
            hk: 16,
            hv: 48,
            dk: 128,
            dv: 128,
            conv_width: 4,
            history_width: 3,
            chunk: 32,
            prefill_value_split: 1,
            decode_value_split: 2,
            rms_eps: 1e-6,
            l2_eps: 1e-6,
        }
    }
    /// Projection/history channel width derived from the normalized GDN geometry.
    pub fn conv_dim(&self) -> Result<u32> {
        self.validate()?;
        self.hk
            .checked_mul(self.dk)
            .and_then(|q| q.checked_mul(2))
            .and_then(|qk| self.hv.checked_mul(self.dv).and_then(|v| qk.checked_add(v)))
            .context("GDN convolution width overflow")
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hk == 16
                && self.hv == 48
                && self.dk == 128
                && self.dv == 128
                && self.conv_width == 4
                && self.history_width == 3
                && self.chunk == 32
                && self.prefill_value_split == 1
                && self.decode_value_split == 2
                && self.rms_eps == 1e-6
                && self.l2_eps == 1e-6,
            "unsupported GDN plan; expected {}",
            Self::PLAN_ID
        );
        Ok(())
    }
    pub fn state_bytes(&self) -> Result<u64> {
        self.validate()?;
        mul(
            mul(
                mul(u64::from(self.hv), u64::from(self.dk))?,
                u64::from(self.dv),
            )?,
            4,
        )
    }
    pub fn history_bytes(&self) -> Result<u64> {
        self.validate()?;
        mul(
            mul(u64::from(self.conv_dim()?), u64::from(self.history_width))?,
            4,
        )
    }
}
