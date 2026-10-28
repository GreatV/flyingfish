//! Shared row-work geometry; excludes weights, persistent state and operator scratch.
use crate::{
    model_spec::{Architecture, ModelSpec},
    op_contract::{DsaSpec, FullAttentionSpec, GdnSpec},
};
use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Debug)]
pub struct DsaRowLayout {
    pub query: Range<usize>,
    pub query_residual: Range<usize>,
    pub latent: Range<usize>,
    pub index_query: Range<usize>,
    pub index_weights: Range<usize>,
    pub packed: Range<usize>,
}

impl DsaRowLayout {
    pub fn new(spec: &ModelSpec, rows: usize) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= i32::MAX as usize,
            "invalid DSA row extent"
        );
        let d = DsaSpec::from_model(spec)?;
        let glm = spec.glm5_next.as_ref().context("DSA geometry missing")?;
        let mut end = 0u32;
        let mut span = |width: u32| -> Result<Range<usize>> {
            ensure!(width > 0, "empty DSA projection width");
            let start = end;
            end = end
                .checked_add(width)
                .context("DSA projection width overflow")?;
            Ok(RowWorkSpec::elements(rows, start as usize)?
                ..RowWorkSpec::elements(rows, end as usize)?)
        };
        Ok(Self {
            query: span(
                d.q_heads
                    .checked_mul(d.qk_dim)
                    .context("DSA query width overflow")?,
            )?,
            query_residual: span(glm.q_lora_rank)?,
            latent: span(d.latent_dim)?,
            index_query: span(
                d.index_heads
                    .checked_mul(d.index_dim)
                    .context("DSA index width overflow")?,
            )?,
            index_weights: span(d.index_heads)?,
            packed: span(d.packed_width())?,
        })
    }
}

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
        let mut intermediate = spec.intermediate as usize;
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
                let g = spec.gdn.as_ref().context("GLM KDA geometry missing")?;
                let d = DsaSpec::from_model(spec)?;
                let key = g
                    .key_heads
                    .checked_mul(g.key_head_dim)
                    .context("KDA key width overflow")?;
                let value = g
                    .value_heads
                    .checked_mul(g.value_head_dim)
                    .context("KDA value width overflow")?;
                let kda_projection = key
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(value))
                    .context("KDA projection width overflow")?;
                let dsa_projection = DsaRowLayout::new(spec, 1)?.packed.end;
                let ab = value
                    .checked_add(g.value_heads)
                    .context("KDA gate width overflow")?;
                (
                    (kda_projection as usize).max(dsa_projection),
                    value.max(d.q_heads * d.v_dim) as usize,
                    true,
                    ab as usize,
                )
            }
            Architecture::DeepseekV41 => {
                let moe = spec
                    .moe
                    .as_ref()
                    .context("DeepSeek-V4.1 MoE geometry missing")?;
                // The projection buffer holds the wq_b output (q_width) and the
                // wkv output (head_dim) side by side: [rows, q_width + head_dim].
                // The FFN row takes the wider of the routed and shared expert
                // intermediates; the mixer buffer holds the post-wo_b attention
                // output: [rows, hidden].
                intermediate =
                    (moe.expert_intermediate as usize).max(moe.shared_intermediate as usize);
                (
                    q.checked_add(spec.head_dim as usize)
                        .context("projection width overflow")?,
                    hidden,
                    false,
                    0,
                )
            }
            Architecture::Gemma4 => {
                let mut head_dim = 0usize;
                for layer in &spec.layers {
                    let geometry = layer.geometry.as_ref().with_context(|| {
                        format!(
                            "missing per-layer attention geometry at {}",
                            layer.state_index
                        )
                    })?;
                    head_dim = head_dim.max(geometry.head_dim as usize);
                    intermediate = intermediate.max(geometry.intermediate as usize);
                }
                if let Some(width) = spec.per_layer_input {
                    intermediate = intermediate.max(spec.layers.len() * width as usize);
                }
                let q = (spec.q_heads as usize)
                    .checked_mul(head_dim)
                    .context("per-layer Q width overflow")?;
                let kv = (spec.kv_heads as usize)
                    .checked_mul(head_dim)
                    .context("per-layer KV width overflow")?;
                (
                    q.checked_add(
                        kv.checked_mul(2)
                            .context("per-layer KV projection overflow")?,
                    )
                    .context("per-layer projection width overflow")?,
                    q,
                    false,
                    0,
                )
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
                    g.ab_width()? as usize,
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
