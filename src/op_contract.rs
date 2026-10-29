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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GdnGateMode {
    PerHead,
    PerChannel { lower_bound: Option<f32> },
}

#[derive(Clone, Copy, Debug)]
pub struct GdnSpec {
    pub gate_mode: GdnGateMode,
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
    pub const PLAN_ID: &'static str = "gdn-plan-v4-c32-vs1-l2fp32";
    pub fn from_model(spec: &crate::model_spec::ModelSpec) -> Result<Self> {
        let g = spec
            .gdn
            .as_ref()
            .context("GDN geometry missing from model spec")?;
        let bound = Self {
            gate_mode: if spec.architecture == crate::model_spec::Architecture::Glm5Next {
                GdnGateMode::PerChannel {
                    lower_bound: Some(
                        spec.glm5_next
                            .as_ref()
                            .context("GLM gate configuration missing")?
                            .gate_lower_bound,
                    ),
                }
            } else {
                GdnGateMode::PerHead
            },
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
            ..Self::q38()
        };
        bound.validate()?;
        Ok(bound)
    }
    pub const fn q38() -> Self {
        Self {
            gate_mode: GdnGateMode::PerHead,
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
    pub fn kernel_suffix(&self) -> Result<&'static str> {
        match self.gate_mode {
            GdnGateMode::PerChannel { .. } => Ok("_per_channel"),
            GdnGateMode::PerHead => match self.hv {
                48 => Ok(""),
                32 => Ok("_h32"),
                _ => anyhow::bail!("unsupported GDN hv={}", self.hv),
            },
        }
    }

    pub fn gate_kernel(&self) -> Result<&'static str> {
        match self.gate_mode {
            GdnGateMode::PerChannel { .. } => Ok("gdn_gates_per_channel"),
            GdnGateMode::PerHead => match self.hv {
                48 => Ok("q38_gdn_gates"),
                32 => Ok("gdn_gates_h32"),
                _ => anyhow::bail!("unsupported GDN hv={}", self.hv),
            },
        }
    }

    /// Ordered recurrent rows supported by the geometry's snapshot kernels.
    pub fn snapshot_limit(&self) -> Result<u32> {
        self.validate()?;
        match self.gate_mode {
            GdnGateMode::PerChannel { .. } => Ok(crate::mtp::MAX_VERIFY_ROWS as u32),
            GdnGateMode::PerHead => match self.hv {
                32 => Ok(crate::linear_plan::MAX_SKINNY_ROWS as u32),
                48 => Ok(crate::mtp::MAX_VERIFY_ROWS as u32),
                _ => anyhow::bail!("unsupported GDN hv={}", self.hv),
            },
        }
    }
    /// Packed A/B row width: per-head stores two head vectors; per-channel adds hv*dv A values.
    pub fn ab_width(&self) -> Result<u32> {
        self.validate()?;
        match self.gate_mode {
            GdnGateMode::PerHead => self.hv.checked_mul(2),
            GdnGateMode::PerChannel { .. } => self
                .hv
                .checked_mul(self.dv.checked_add(1).context("GDN AB width overflow")?),
        }
        .context("GDN AB width overflow")
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
        let gate_valid = match self.gate_mode {
            GdnGateMode::PerHead => {
                matches!(self.hv, 32 | 48) && self.hk == 16 && self.rms_eps == 1e-6
            }
            GdnGateMode::PerChannel { lower_bound } => {
                self.hk == 64
                    && self.hv == 64
                    && self.rms_eps == 1e-5
                    && lower_bound.is_none_or(|value| value.is_finite() && value <= 0.0)
            }
        };
        ensure!(
            gate_valid
                && self.dk == 128
                && self.dv == 128
                && self.conv_width == 4
                && self.history_width == 3
                && self.chunk == 32
                && self.prefill_value_split == 1
                && self.decode_value_split == 2
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
    pub fn image_bytes(&self) -> Result<u64> {
        self.state_bytes()?
            .checked_add(self.history_bytes()?)
            .context("GDN state image byte overflow")
    }
    pub fn input_row_bytes(&self) -> Result<u64> {
        self.validate()?;
        let ab = u64::from(self.ab_width()?);
        let elems = u64::from(self.conv_dim()?)
            .checked_add(ab)
            .context("GDN verify input width overflow")?;
        mul(elems, 2)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DsaMeta {
    pub prefix: i32,
    pub valid_rows: i32,
    pub key_count: i32,
    pub capacity: i32,
    pub first_key: i32,
    pub reserved: [i32; 3],
}
impl DsaMeta {
    pub fn validate(&self, capacity: u32) -> Result<()> {
        ensure!(
            capacity > 0 && capacity <= i32::MAX as u32,
            "invalid DSA capacity"
        );
        ensure!(
            self.capacity == capacity as i32
                && self.prefix >= 0
                && self.valid_rows > 0
                && self.valid_rows <= self.capacity
                && self.first_key >= 0
                && self.key_count > 0,
            "invalid DSA sequence metadata"
        );
        ensure!(
            self.prefix.checked_add(self.valid_rows) == Some(self.key_count)
                && self.first_key <= self.key_count
                && self.reserved == [0; 3],
            "invalid DSA sequence extents"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DsaKvForm {
    Direct,
    Projected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DsaSpec {
    pub kv_form: DsaKvForm,
    pub index_heads: u32,
    pub index_dim: u32,
    pub pool: u32,
    pub index_topk: u32,
    pub q_heads: u32,
    pub qk_dim: u32,
    pub v_dim: u32,
    pub latent_dim: u32,
}
impl DsaSpec {
    pub const fn glm53() -> Self {
        Self {
            kv_form: DsaKvForm::Projected,
            index_heads: 32,
            index_dim: 128,
            pool: 4,
            index_topk: 2048,
            q_heads: 64,
            qk_dim: 256,
            v_dim: 256,
            latent_dim: 512,
        }
    }
    pub fn from_model(spec: &crate::model_spec::ModelSpec) -> Result<Self> {
        let g = spec
            .glm5_next
            .as_ref()
            .context("DSA geometry missing from model spec")?;
        let bound = Self {
            kv_form: DsaKvForm::Projected,
            index_heads: g.index_heads,
            index_dim: g.index_head_dim,
            pool: g.index_kpool,
            index_topk: g.index_topk,
            q_heads: spec.q_heads,
            qk_dim: spec.head_dim,
            v_dim: g.v_head_dim,
            latent_dim: g.kv_lora_rank,
        };
        ensure!(
            bound.qk_dim == g.qk_nope_head_dim,
            "DSA rope geometry is not bound"
        );
        bound.validate()?;
        Ok(bound)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (self.index_heads, self.index_dim, self.q_heads) == (32, 128, 64),
            "unsupported DSA specification: {self:?}"
        );
        match self.kv_form {
            DsaKvForm::Direct => ensure!(
                (
                    self.pool,
                    self.index_topk,
                    self.qk_dim,
                    self.v_dim,
                    self.latent_dim
                ) == (1, 512, 512, 512, 512),
                "direct DSA requires unpooled 512-wide KV and 512 selected positions"
            ),
            DsaKvForm::Projected => ensure!(
                (
                    self.pool,
                    self.index_topk,
                    self.qk_dim,
                    self.v_dim,
                    self.latent_dim
                ) == (4, 2048, 256, 256, 512),
                "projected DSA requires pool4, topk2048, K/V256 and latent512"
            ),
        }
        Ok(())
    }
    pub const fn selected_pools(&self) -> u32 {
        self.index_topk / self.pool
    }
    pub const fn selected_width(&self) -> u32 {
        self.index_topk + self.pool - 1
    }
    pub const fn packed_width(&self) -> u32 {
        2 * self.index_dim + 1
    }
    pub const fn pool_capacity(&self, capacity: u32) -> u32 {
        capacity.div_ceil(self.pool)
    }
    pub fn state_bytes(&self, capacity: u32) -> Result<u64> {
        self.validate()?;
        ensure!(
            capacity > 0 && capacity <= i32::MAX as u32,
            "DSA state capacity outside positive i32"
        );
        let pools = u64::from(self.pool_capacity(capacity));
        let cap = u64::from(capacity);
        let elements = |width: u32, count: u64, bytes: u64| -> Result<u64> {
            count
                .checked_mul(u64::from(width))
                .and_then(|v| v.checked_mul(bytes))
                .context("DSA state extent overflow")
        };
        let bytes = add(
            elements(self.latent_dim, cap, 2)?,
            elements(self.packed_width(), cap, 2)?,
        )?;
        let bytes = add(bytes, elements(self.index_dim, pools, 2)?)?;
        let bytes = add(bytes, elements(1, cap, 4)?)?;
        let bytes = add(bytes, elements(1, pools, 4)?)?;
        add(bytes, 32)
    }
    pub fn index_workspace_bytes(&self, rows: u32) -> Result<u64> {
        self.validate()?;
        ensure!(rows > 0, "DSA workspace rows must be positive");
        let per_row =
            u64::from(self.selected_pools()) * 8 + 4 + u64::from(self.selected_width()) * 4;
        per_row
            .checked_mul(u64::from(rows))
            .context("DSA index workspace overflow")
    }
}
const _: () = assert!(std::mem::size_of::<DsaMeta>() == 32);

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
        let changed = GdnSpec::from_model(&model)?;
        let attention = FullAttentionSpec::from_model(&model)?;
        let work = crate::work_layout::RowWorkSpec::new(&model)?;
        assert_eq!((work.projection, work.mixer, work.ab), (9216, 4096, 64));
        assert_eq!(changed.state_bytes()?, 32 * 128 * 128 * 4);
        assert_eq!(changed.history_bytes()?, 8192 * 3 * 4);
        assert_eq!(attention.kernel_module()?, "tcmqa_d256_q16_kv2");
        assert_eq!(changed.kernel_suffix()?, "_h32");
        assert_eq!(changed.gate_kernel()?, "gdn_gates_h32");
        assert_eq!(original.snapshot_limit()?, 5);
        assert_eq!(changed.snapshot_limit()?, 16);
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

    #[test]
    fn dsa_metadata_capacity_and_padding() -> Result<()> {
        let base = DsaMeta {
            prefix: 0,
            valid_rows: 4,
            key_count: 4,
            capacity: 4,
            first_key: 0,
            reserved: [0; 3],
        };
        let mut defects = 0;
        for (name, meta, expected) in [
            (
                "append_overwrite",
                DsaMeta {
                    valid_rows: 5,
                    key_count: 5,
                    ..base
                },
                false,
            ),
            (
                "all_padding",
                DsaMeta {
                    first_key: 4,
                    ..base
                },
                true,
            ),
        ] {
            let accepted = meta.validate(4).is_ok();
            defects += usize::from(accepted != expected);
            println!("DSA_METADATA case={name} accepted={accepted} expected={expected}");
        }
        let overflow = DsaSpec::glm53().state_bytes(u32::MAX).is_ok();
        defects += usize::from(overflow);
        println!("DSA_METADATA case=state_i32_capacity accepted={overflow} expected=false");
        ensure!(defects == 0, "DEFECT DSA metadata count={defects}");
        Ok(())
    }

    #[test]
    fn dsa_state_and_index_workspace_cover_production_bounds() -> Result<()> {
        let spec = DsaSpec::glm53();
        assert_eq!(spec.state_bytes(4096)?, 6_582_304);
        assert_eq!(spec.index_workspace_bytes(1)?, 12_304);
        for capacity in [3u32, 4, 2047, 2048, 2051, 4096, 1_048_576] {
            let pools = u64::from(capacity.div_ceil(4));
            let expected =
                u64::from(capacity) * (512 * 2 + 257 * 2 + 4) + pools * (128 * 2 + 4) + 32;
            assert_eq!(spec.state_bytes(capacity)?, expected);
            for rows in [1, 5, 512] {
                assert_eq!(spec.index_workspace_bytes(rows)?, 12_304 * u64::from(rows));
            }
            let meta = DsaMeta {
                prefix: capacity as i32 - 1,
                valid_rows: 1,
                key_count: capacity as i32,
                capacity: capacity as i32,
                first_key: 0,
                reserved: [0; 3],
            };
            meta.validate(capacity)?;
        }
        let direct = DsaSpec {
            kv_form: DsaKvForm::Direct,
            pool: 1,
            index_topk: 512,
            qk_dim: 512,
            v_dim: 512,
            ..spec
        };
        direct.validate()?;
        assert!(DsaMeta::default().validate(4096).is_err());
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/glm53-flash-config.json"
        )))?;
        let model = crate::model_spec::ModelSpec::glm5_next(
            &root,
            &serde_json::json!({"eos_token_id": [154820, 154827, 154829]}),
        )?;
        assert_eq!(DsaSpec::from_model(&model)?, spec);
        Ok(())
    }
}
