//! Shared row-work geometry; excludes weights, persistent state and operator scratch.
use crate::{
    model_spec::{Architecture, ModelSpec},
    op_contract::{FullAttentionSpec, GdnSpec},
};
use anyhow::{Context, Result};
#[derive(Clone, Copy, Debug)]
pub struct RowWorkSpec {
    pub hidden: usize,
    pub intermediate: usize,
    pub projection: usize,
    pub mixer: usize,
    pub extra_mixer: bool,
    pub ab: usize,
}
impl RowWorkSpec {
    pub fn new(spec: &ModelSpec) -> Result<Self> {
        let hidden = spec.hidden as usize;
        let intermediate = spec.intermediate as usize;
        let q = spec.q_width as usize;
        let kv = (spec.kv_heads as usize)
            .checked_mul(spec.head_dim as usize)
            .context("KV width overflow")?;
        let (projection, mixer, extra_mixer, ab) = match spec.architecture {
            Architecture::LoopVl => (2 * q + 2 * kv, q, false, 0),
            Architecture::Llama => (
                q.checked_add(kv.checked_mul(2).context("KV width overflow")?)
                    .context("projection width overflow")?,
                q,
                false,
                0,
            ),
            Architecture::Flint => (
                q.checked_add(kv.checked_mul(2).context("KV width overflow")?)
                    .context("projection width overflow")?,
                q,
                true,
                0,
            ),
            Architecture::Glm5Next => {
                anyhow::bail!("GLM-5.3 row work is not bound yet")
            }
            Architecture::Qwen35Text | Architecture::MiniCpmV47 => {
                FullAttentionSpec::from_model(spec)?;
                let g = GdnSpec::from_model(spec)?;
                let gqkv = g.conv_dim()? as usize;
                let full = q
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(2 * kv))
                    .context("Q/gate/K/V width overflow")?;
                (
                    full.max(gqkv),
                    q.max((g.hv * g.dv) as usize),
                    true,
                    (2 * g.hv) as usize,
                )
            }
        };
        Ok(Self {
            hidden,
            intermediate,
            projection,
            mixer,
            extra_mixer,
            ab,
        })
    }
    pub fn row_bytes(&self) -> Result<usize> {
        let fields = [
            self.hidden,
            self.hidden,
            self.hidden,
            self.projection,
            self.mixer,
            self.intermediate,
            self.intermediate,
            self.intermediate,
            if self.extra_mixer { self.mixer } else { 0 },
            if self.extra_mixer { self.mixer } else { 0 },
            self.ab,
        ];
        fields
            .iter()
            .try_fold(0usize, |sum, &n| {
                sum.checked_add(n).context("row width overflow")
            })?
            .checked_mul(2)
            .and_then(|n| n.checked_add(4))
            .context("row bytes overflow")
    }
    pub fn elements(rows: usize, width: usize) -> Result<usize> {
        rows.checked_mul(width).context("row allocation overflow")
    }
}
