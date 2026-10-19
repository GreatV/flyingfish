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
            self.capacity == capacity as i32 && [0, 1].contains(&self.causal),
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
    pub fn new(
        spec: &FullAttentionSpec,
        max_rows: u32,
        capacity: u32,
        key_chunk: u32,
    ) -> Result<Self> {
        spec.validate()?;
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
        let e = mul(mul(segment_rows, u64::from(spec.hq))?, max_parts)?;
        let scalar = align(mul(e, 4)?)?;
        let p_out = mul(scalar, 2)?;
        let k_norm = add(p_out, align(mul(mul(e, 256)?, 4)?)?)?;
        let bytes = add(
            k_norm,
            align(mul(mul(u64::from(max_rows), u64::from(spec.hkv))?, 512)?)?,
        )?;
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

#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub fn from_model(spec: &crate::model_spec::ModelSpec) -> Result<Self> {
        ensure!(
            spec.q_heads.checked_mul(spec.head_dim) == Some(spec.q_width),
            "attention q_width={} differs from q_heads={} * head_dim={}",
            spec.q_width,
            spec.q_heads,
            spec.head_dim
        );
        let bound = Self {
            hq: spec.q_heads,
            hkv: spec.kv_heads,
            d: spec.head_dim,
            scale: 1.0 / (spec.head_dim as f32).sqrt(),
            rms_eps: spec.rms_eps,
            rope_theta: spec.rope_theta,
            rotary_dim: spec.rotary_dim,
        };
        bound.validate()?;
        Ok(bound)
    }

    pub fn kernel_module(&self) -> Result<&'static str> {
        match (self.hq, self.hkv) {
            (24, 4) => Ok("tcmqa_d256"),
            (16, 2) => Ok("tcmqa_d256_q16_kv2"),
            (8, 2) => Ok("tcmqa_d256_q8_kv2"),
            _ => anyhow::bail!("unsupported attention hq={} hkv={}", self.hq, self.hkv),
        }
    }

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
            matches!(
                (self.hq, self.hkv, self.rotary_dim),
                (24, 4, 64) | (16, 2, 64) | (8, 2, 256)
            ) && self.d == 256
                && self.scale == 0.0625
                && self.rms_eps == 1e-6
                && self.rope_theta == 1e7,
            "unsupported full-attention specification: {self:?}"
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
    pub qk_norm_fp32: bool,
}
impl GdnSpec {
    pub const PLAN_ID: &'static str = "gdn-plan-v2-c32-vs1-fp32";
    pub fn from_model(spec: &crate::model_spec::ModelSpec) -> Result<Self> {
        let g = spec
            .gdn
            .as_ref()
            .context("GDN geometry missing from model spec")?;
        let bound = Self {
            hk: g.key_heads,
            hv: g.value_heads,
            dk: g.key_head_dim,
            dv: g.value_head_dim,
            conv_width: g.conv_kernel,
            history_width: g
                .conv_kernel
                .checked_sub(1)
                .context("GDN history width underflow")?,
            rms_eps: spec.rms_eps,
            qk_norm_fp32: g.qk_norm_fp32,
            ..Self::q38()
        };
        bound.validate()?;
        Ok(bound)
    }
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
            qk_norm_fp32: false,
        }
    }
    pub fn kernel_suffix(&self) -> Result<&'static str> {
        match self.hv {
            48 => Ok(""),
            32 => Ok("_h32"),
            _ => anyhow::bail!("unsupported GDN hv={}", self.hv),
        }
    }

    pub fn gate_kernel(&self) -> Result<&'static str> {
        match self.hv {
            48 => Ok("q38_gdn_gates"),
            32 => Ok("gdn_gates_h32"),
            _ => anyhow::bail!("unsupported GDN hv={}", self.hv),
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
            matches!((self.hv, self.qk_norm_fp32), (32, true) | (48, false)),
            "unsupported GDN hv={} qk_norm_fp32={}",
            self.hv,
            self.qk_norm_fp32
        );
        ensure!(
            self.hk == 16
                && matches!(self.hv, 32 | 48)
                && self.dk == 128
                && self.dv == 128
                && self.conv_width == 4
                && self.history_width == 3
                && self.chunk == 32
                && self.prefill_value_split == 1
                && self.decode_value_split == 2
                && self.rms_eps == 1e-6
                && self.l2_eps == 1e-6,
            "unsupported GDN plan {self:?}; expected {}",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_fields_determine_mixer_work_and_state_capacity() -> Result<()> {
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let mut model = crate::model_spec::ModelSpec::qwen(&root, &serde_json::Value::Null)?;
        let original = GdnSpec::from_model(&model)?;
        let work = crate::work_layout::RowWorkSpec::new(&model)?;
        assert_eq!((work.projection, work.mixer, work.ab), (14336, 6144, 96));
        assert_eq!(original.state_bytes()?, 48 * 128 * 128 * 4);
        assert_eq!(
            FullAttentionSpec::from_model(&model)?.kernel_module()?,
            "tcmqa_d256"
        );
        assert_eq!(original.kernel_suffix()?, "");
        assert_eq!(original.gate_kernel()?, "q38_gdn_gates");
        model.hidden = 2048;
        model.q_heads = 16;
        model.kv_heads = 2;
        model.q_width = 4096;
        model.gdn.as_mut().unwrap().value_heads = 32;
        model.gdn.as_mut().unwrap().qk_norm_fp32 = true;
        let changed = GdnSpec::from_model(&model)?;
        let attention = FullAttentionSpec::from_model(&model)?;
        let work = crate::work_layout::RowWorkSpec::new(&model)?;
        assert_eq!((work.projection, work.mixer, work.ab), (9216, 4096, 64));
        assert_eq!(changed.state_bytes()?, 32 * 128 * 128 * 4);
        assert_eq!(changed.history_bytes()?, 8192 * 3 * 4);
        assert_eq!(attention.kernel_module()?, "tcmqa_d256_q16_kv2");
        assert_eq!(changed.kernel_suffix()?, "_h32");
        assert_eq!(changed.gate_kernel()?, "gdn_gates_h32");
        let unsupported = GdnSpec { hv: 64, ..changed };
        assert!(
            unsupported
                .kernel_suffix()
                .unwrap_err()
                .to_string()
                .contains("hv=64")
        );
        assert!(
            unsupported
                .gate_kernel()
                .unwrap_err()
                .to_string()
                .contains("hv=64")
        );
        let unsupported = FullAttentionSpec {
            hq: 16,
            hkv: 4,
            ..attention
        };
        assert!(
            unsupported
                .kernel_module()
                .unwrap_err()
                .to_string()
                .contains("hq=16 hkv=4")
        );
        assert_eq!(
            (attention.hq, attention.hkv, changed.conv_dim()?),
            (16, 2, 8192)
        );
        assert_eq!(
            model.architecture,
            crate::model_spec::Architecture::Qwen35Text
        );
        Ok(())
    }

    #[test]
    fn attention_workspace_covers_each_bound_head_geometry() -> Result<()> {
        for (hq, hkv, rotary_dim) in [(24, 4, 64), (16, 2, 64), (8, 2, 256)] {
            let spec = FullAttentionSpec {
                hq,
                hkv,
                rotary_dim,
                ..FullAttentionSpec::q38()
            };
            for rows in [1, 2, 5, 8, 37, 512, 4096] {
                let w = AttentionWorkspace::new(&spec, rows, 8192, 128)?;
                let aligned = |n: u64| (n + 255) & !255;
                let elements = u64::from(rows.min(64)) * u64::from(hq) * 64;
                assert_eq!(w.p_sum, aligned(elements * 4));
                assert_eq!(w.p_out, 2 * aligned(elements * 4));
                assert_eq!(w.k_norm, w.p_out + aligned(elements * 256 * 4));
                assert_eq!(
                    w.bytes - w.k_norm,
                    aligned(u64::from(rows) * u64::from(hkv) * 256 * 2)
                );
            }
        }
        let unsupported = FullAttentionSpec {
            hkv: 4,
            hq: 16,
            ..FullAttentionSpec::q38()
        };
        assert!(AttentionWorkspace::new(&unsupported, 1, 8192, 128).is_err());
        Ok(())
    }
}
