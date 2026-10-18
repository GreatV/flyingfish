//! Qwen bindings and operators of the shared Engine.
use super::*;
use crate::backend::cuda::contracts::view;
use crate::backend::cuda::{
    attention_d256::{AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings},
    contracts::{RowDomain, Scratch, Tensor, TensorMut},
    gdn::{CUDA_GRID_YZ_MAX, GdnBindings, GdnOps, GdnPlan},
    linear::{LayerLinearPlan, LinearWeightRef},
    ops::{CommonOpsRef, GatedRmsInput, ResidualRmsCapture, ResidualRmsInput, RmsMode, RmsOptions},
    weights::WeightCatalog,
};
/// The workspace alignment the default-chunk cap must preserve; the same value the budget call passes.
const CHUNK_ALIGNMENT: usize = 32;
/// The default chunk is bounded by the largest alignment multiple at or under the CUDA gridDim.y/z limit the GDN prep pair launches rows on.
fn default_chunk(admitted: usize) -> usize {
    admitted.min(CUDA_GRID_YZ_MAX as usize / CHUNK_ALIGNMENT * CHUNK_ALIGNMENT)
}

use crate::gdn_plan::GdnPolicy;
use crate::{
    linear_plan::LayerDequantPlan,
    model_spec::Mixer,
    op_contract::{AttentionLimits, ExecutionPhase, FullAttentionSpec, GdnSpec},
};
use cudarc::driver::{CudaFunction, DevicePtr};
use std::collections::BTreeSet;

fn all_layers(trace: &Option<Target<'_>>) -> bool {
    trace.as_ref().is_some_and(|t| t.trace.all_layers)
}

fn debug_name(trace: &Option<Target<'_>>, layer: usize, point: &str) -> String {
    if all_layers(trace) {
        format!("{layer}.debug.{point}")
    } else {
        format!("debug.{point}")
    }
}

fn require_qwen35_geometry(spec: &crate::model_spec::ModelSpec) -> Result<()> {
    use crate::model_spec::{Architecture, LinearFormat};
    let (hidden, intermediate, vocab, hq, hkv, layers, hv, format) = match spec.architecture {
        Architecture::Qwen35Text => (
            5120,
            17408,
            248320,
            24,
            4,
            64,
            48,
            LinearFormat::Fp8E4m3Block128,
        ),
        Architecture::MiniCpmV47 => (2048, 512, 248144, 16, 2, 40, 32, LinearFormat::Bf16),
        Architecture::Llama => anyhow::bail!(
            "hybrid engine does not implement architecture={:?}",
            spec.architecture
        ),
    };
    for (name, value, expected) in [
        ("hidden_size", spec.hidden, hidden),
        ("intermediate_size", spec.intermediate, intermediate),
        ("vocab_size", spec.vocab, vocab),
        ("num_attention_heads", spec.q_heads, hq),
        ("num_key_value_heads", spec.kv_heads, hkv),
        ("head_dim", spec.head_dim, 256),
        ("q_width", spec.q_width, hq * 256),
        ("max_position_embeddings", spec.max_position, 262144),
        ("rotary_dim", spec.rotary_dim, 64),
    ] {
        ensure!(
            value == expected,
            "hybrid engine does not implement {name}={value}"
        );
    }
    ensure!(
        spec.layers.len() == layers,
        "hybrid engine does not implement num_hidden_layers={}",
        spec.layers.len()
    );
    ensure!(
        spec.rms_eps == 1e-6,
        "hybrid engine does not implement rms_norm_eps={}",
        spec.rms_eps
    );
    ensure!(
        spec.rope_theta == 1e7,
        "hybrid engine does not implement rope_theta={}",
        spec.rope_theta
    );
    ensure!(
        spec.linear_format == format,
        "hybrid engine does not implement linear_format={:?} for {:?}",
        spec.linear_format,
        spec.architecture
    );
    let gdn = spec
        .gdn
        .as_ref()
        .context("hybrid engine does not implement gdn=None")?;
    for (name, value, expected) in [
        ("linear_num_key_heads", gdn.key_heads, 16),
        ("linear_num_value_heads", gdn.value_heads, hv),
        ("linear_key_head_dim", gdn.key_head_dim, 128),
        ("linear_value_head_dim", gdn.value_head_dim, 128),
        ("linear_conv_kernel_dim", gdn.conv_kernel, 4),
    ] {
        ensure!(
            value == expected,
            "hybrid engine does not implement {name}={value}"
        );
    }
    Ok(())
}

pub(super) struct Runtime {
    attention: FullAttentionSpec,
    gdn: GdnSpec,
    checkpoint_dir: PathBuf,
    vision: Option<super::vision_runtime::VisionRuntime>,
    vision_bytes: usize,
    positions: CudaSlice<i32>,
    active_positions: CudaSlice<i32>,
    gather_positions: CudaFunction,
    catalog: WeightCatalog,
    arena: Option<CudaSlice<u8>>,
    scratch: CudaSlice<u8>,
    kv_scratch: Option<(CudaSlice<bf16>, CudaSlice<bf16>)>,
    mask: CudaSlice<u64>,
    layers: Vec<Layer>,
    common: CommonOpsRef,
    advance: CudaFunction,
    prepared: BTreeSet<u32>,
    workspace_rows: u32,
}
struct Layer {
    linear: LayerLinearPlan,
    input_norm: Tensor<bf16, 1>,
    post_norm: Tensor<bf16, 1>,
    mixer: BoundMixer,
}
enum BoundMixer {
    Attention {
        image_prefill: PrefillPlans<FullAttentionPlan>,
        text_prefill: PrefillPlans<FullAttentionPlan>,
        decode: Box<FullAttentionPlan>,
        compact_small: Option<PrefillPlans<FullAttentionPlan>>,
    },
    Gdn {
        norm: Tensor<bf16, 1>,
        prefill: PrefillPlans<GdnPlan>,
        verify: Vec<GdnPlan>,
        decode: Box<GdnPlan>,
    },
}

/// Prebound logical extents for MTP verify/replay. All plans share the Engine's
/// existing allocations; projection offsets remain based on workspace_rows.
struct PrefillPlans<T> {
    full: Box<T>,
    small: Vec<T>,
    max_rows: usize,
}
/// Either a prebound plan borrowed as-is or a plan narrowed at dispatch.
enum Narrowed<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}
impl<T> std::ops::Deref for Narrowed<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Narrowed::Borrowed(plan) => plan,
            Narrowed::Owned(plan) => plan,
        }
    }
}
impl<T> PrefillPlans<T> {
    fn bind(full: T, max_rows: usize, narrow: impl Fn(&T, u32) -> Result<T>) -> Result<Self> {
        ensure!(max_rows > 0, "prefill plan requires rows");
        let small = (1..=crate::mtp::MAX_VERIFY_ROWS.min(max_rows))
            .map(|rows| narrow(&full, rows as u32))
            .collect::<Result<_>>()?;
        Ok(Self {
            full: Box::new(full),
            small,
            max_rows,
        })
    }
    /// Rows 1..=5 borrow prebound small plans; larger sizes narrow at dispatch.
    fn select(
        &self,
        rows: usize,
        narrow: &dyn Fn(&T, u32) -> Result<T>,
    ) -> Result<Narrowed<'_, T>> {
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "prefill rows outside binding"
        );
        if let Some(small) = self.small.get(rows - 1) {
            return Ok(Narrowed::Borrowed(small));
        }
        Ok(Narrowed::Owned(narrow(&self.full, rows as u32)?))
    }
}
pub(super) fn matrix(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 2>> {
    view(
        b,
        s,
        offset,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}
pub(super) fn heads(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    h: u32,
    d: u32,
) -> Result<Tensor<bf16, 3>> {
    view(
        b,
        s,
        offset,
        [u32::try_from(rows)?, h, d],
        [u64::from(h) * u64::from(d), u64::from(d), 1],
    )
}
pub(super) fn vector(b: &CudaSlice<bf16>, s: &Arc<CudaStream>) -> Result<Tensor<bf16, 1>> {
    view(b, s, 0, [u32::try_from(b.len())?], [1])
}
pub(super) fn scratch(b: &CudaSlice<u8>, s: &Arc<CudaStream>) -> Scratch {
    let (ptr, _read) = b.device_ptr(s);
    Scratch {
        ptr,
        bytes: b.len() as u64,
    }
}
pub(super) fn limits(rows: usize, capacity: usize) -> Result<AttentionLimits> {
    ensure!(capacity >= 32, "Q38 attention capacity must be at least 32");
    Ok(AttentionLimits {
        max_rows: u32::try_from(rows)?,
        capacity: u32::try_from(capacity)?,
        key_chunk: 256.min((capacity / 32 * 32) as u32),
        causal: 1,
    })
}
fn nested_verify_tcmqa(device_prefix: bool, rows: usize) -> Result<bool> {
    if !device_prefix {
        return Ok(false);
    }
    ensure!(
        rows > 0 && rows <= 5,
        "nested device-prefix attention supports rows 1..=5"
    );
    Ok(true)
}
impl Runtime {
    pub(super) fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_dir
    }
    #[cfg(test)]
    pub(super) fn position_values(&self, stream: &Arc<CudaStream>) -> Result<Vec<i32>> {
        Ok(stream.clone_dtoh(&self.positions)?)
    }
    pub(super) fn vision_bytes(&self) -> usize {
        self.vision_bytes
    }
    fn row_bytes(spec: &crate::model_spec::ModelSpec) -> Result<usize> {
        crate::work_layout::RowWorkSpec::new(spec)?
            .row_bytes()?
            .checked_add(spec.q_heads as usize * size_of::<f32>())
            .and_then(|n| n.checked_add(3 * size_of::<i32>()))
            .context("Q38 row workspace including FA2 LSE overflow")
    }
    fn fixed_bytes(spec: &crate::model_spec::ModelSpec, capacity: usize) -> Result<usize> {
        capacity
            .checked_mul(3 * size_of::<i32>())
            .and_then(|n| n.checked_add(spec.vocab as usize * 2 + 12 + 64 * size_of::<u64>()))
            .context("Q38 fixed workspace overflow")
    }
    fn workspace(
        model: &crate::model_spec::ModelSpec,
        rows: usize,
        capacity: usize,
        policy: GdnPolicy,
    ) -> Result<usize> {
        let attention = FullAttentionSpec::from_model(model)?;
        let gdn = GdnSpec::from_model(model)?;
        let attention_prefill =
            FullAttentionOps::preprocess_workspace(&attention, limits(rows, capacity)?)?;
        let attention_decode = FullAttentionOps::workspace(&attention, limits(1, capacity)?)?;
        let attention_nested_verify = FullAttentionOps::workspace(
            &attention,
            limits(rows.min(crate::mtp::MAX_VERIFY_ROWS), capacity)?,
        )?;
        let spec = gdn;
        let prefill = GdnOps::workspace(&spec, u32::try_from(rows)?, policy)?;
        let decode = GdnOps::workspace(&spec, 1, GdnPolicy::LEGACY)?;
        ensure!(
            [
                attention_prefill.alignment,
                attention_decode.alignment,
                attention_nested_verify.alignment,
                prefill.alignment,
                decode.alignment,
            ]
            .iter()
            .all(|&a| a == 256),
            "operator scratch alignment differs from Engine arena"
        );
        Ok(usize::try_from(
            attention_prefill
                .bytes
                .max(attention_decode.bytes)
                .max(attention_nested_verify.bytes)
                .max(prefill.bytes)
                .max(decode.bytes),
        )?)
    }
    fn new(
        engine: &Engine,
        checkpoint_dir: &Path,
        catalog: WeightCatalog,
        arena: Option<CudaSlice<u8>>,
        scratch_buffer: CudaSlice<u8>,
        kv_scratch: Option<(CudaSlice<bf16>, CudaSlice<bf16>)>,
        verify_rows: usize,
    ) -> Result<Self> {
        let s = &engine.device.stream;
        let rows = engine.workspace_rows;
        let attention = FullAttentionSpec::from_model(&engine.model_spec)?;
        let gdn = GdnSpec::from_model(&engine.model_spec)?;
        let q_width = engine.model_spec.q_width;
        let kv_width = attention.hkv * attention.d;
        let gdn_policy = GdnPolicy::for_device(engine.device.info.arch, engine.device.info.sms);
        let common = CommonOpsRef::new(&engine.device.ctx)?;
        let positions = s.clone_htod(
            &crate::vision_input::PositionPlan::text(1, engine.capacity)?.packed_axes()?,
        )?;
        let active_positions = s.alloc_zeros(rows * 3)?;
        let mask = s.clone_htod(
            &(0..64)
                .map(|i| {
                    if i == 63 {
                        u64::MAX
                    } else {
                        (1u64 << (i + 1)) - 1
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        let mut layers = Vec::new();
        let w = &engine.work;
        let mw = w.mixer.as_ref().context("Q38 mixer work missing")?;
        for (i, layout) in engine.model_spec.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{i}.");
            let weight = |name: &str| vector(catalog.bf16(&format!("{prefix}{name}"))?, s);
            let plan = LayerDequantPlan::from_manifest(
                &catalog.manifest,
                i as u32,
                engine.model_spec.linear_format,
            )?;
            let arena_view = arena.as_ref().map(|b| scratch(b, s));
            let linear = LayerLinearPlan::bind(&plan, &catalog, arena_view.as_ref(), s)?;
            let mixer = match layout.mixer {
                Mixer::FullAttention => {
                    let kv = &engine.state.kv[layout.state_index as usize];
                    ensure!(
                        kv.compact.is_some() == kv_scratch.is_some(),
                        "KV format/prefill scratch mismatch at setup"
                    );
                    let bind = |active_rows, positions, compact: bool| {
                        let (k, v, cap) = if compact {
                            (
                                &kv.k,
                                &kv.v,
                                kv.compact
                                    .as_ref()
                                    .context("compact cache missing")?
                                    .layout
                                    .ring,
                            )
                        } else if let Some((k, v)) = &kv_scratch {
                            (k, v, engine.capacity)
                        } else {
                            (&kv.k, &kv.v, engine.capacity)
                        };
                        FullAttentionOps::prepare(
                            &engine.device.ctx,
                            &attention,
                            limits(active_rows, engine.capacity)?,
                            AttentionBindings {
                                compact: if compact {
                                    kv.compact.as_ref().map(|c| c.view(&kv.k, &kv.v, s))
                                } else {
                                    None
                                },
                                q_gate: heads(&w.qkv, s, 0, active_rows, attention.hq, 512)?,
                                k: heads(
                                    &w.qkv,
                                    s,
                                    rows * (2 * q_width as usize),
                                    active_rows,
                                    attention.hkv,
                                    256,
                                )?,
                                v: heads(
                                    &w.qkv,
                                    s,
                                    rows * (2 * q_width as usize + kv_width as usize),
                                    active_rows,
                                    attention.hkv,
                                    256,
                                )?,
                                q: TensorMut(heads(&mw.q, s, 0, active_rows, attention.hq, 256)?),
                                gate: Some(TensorMut(heads(
                                    &mw.gate,
                                    s,
                                    0,
                                    active_rows,
                                    attention.hq,
                                    256,
                                )?)),
                                norm_mode: crate::backend::cuda::ops::RmsMode::ZeroCentered,
                                k_cache: TensorMut(view(
                                    k,
                                    s,
                                    0,
                                    [attention.hkv, cap as u32, 256],
                                    [(cap * 256) as u64, 256, 1],
                                )?),
                                v_cache: TensorMut(view(
                                    v,
                                    s,
                                    0,
                                    [attention.hkv, cap as u32, 256],
                                    [(cap * 256) as u64, 256, 1],
                                )?),
                                core_out: TensorMut(heads(
                                    &w.attn,
                                    s,
                                    0,
                                    active_rows,
                                    attention.hq,
                                    256,
                                )?),
                                q_norm: weight("self_attn.q_norm.weight")?,
                                k_norm: weight("self_attn.k_norm.weight")?,
                                causal_mask: view(&mask, s, 0, [64], [1])?,
                                positions,
                            },
                            common.clone(),
                        )
                    };
                    let decode = bind(
                        1,
                        PositionBindings::Axes(view(&active_positions, s, 0, [1, 3], [3, 1])?),
                        kv.compact.is_some(),
                    )?;
                    ensure!(
                        decode.workspace().bytes <= scratch_buffer.len() as u64,
                        "bound decode attention exceeds Engine scratch budget"
                    );
                    if layout.state_index == 0 {
                        eprintln!(
                            "{}",
                            serde_json::json!({"q38_decode_attention_plan": {
                                "segment_rows":decode.segment_rows(), "segments":decode.segments(),
                                "max_parts":decode.max_parts(), "phase1_smem_bytes":decode.phase1_smem_bytes(),
                                "phase2_smem_bytes":decode.phase2_smem_bytes(), "workspace_bytes":decode.workspace().bytes,
                                "positions":"Axes"
                            }})
                        );
                    }
                    let text_prefill = PrefillPlans::bind(
                        bind(rows, PositionBindings::Linear, false)?,
                        rows,
                        FullAttentionPlan::for_rows,
                    )?;
                    ensure!(
                        text_prefill
                            .small
                            .iter()
                            .all(|plan| plan.workspace().bytes <= scratch_buffer.len() as u64),
                        "nested verify attention exceeds Engine scratch budget"
                    );
                    BoundMixer::Attention {
                        image_prefill: PrefillPlans::bind(
                            bind(
                                rows,
                                PositionBindings::Axes(view(
                                    &active_positions,
                                    s,
                                    0,
                                    [rows as u32, 3],
                                    [3, 1],
                                )?),
                                false,
                            )?,
                            rows,
                            FullAttentionPlan::for_rows,
                        )?,
                        text_prefill,
                        compact_small: if kv.compact.is_some() {
                            Some(PrefillPlans::bind(
                                bind(verify_rows.min(rows), PositionBindings::Linear, true)?,
                                verify_rows.min(rows),
                                FullAttentionPlan::for_rows,
                            )?)
                        } else {
                            None
                        },
                        decode: Box::new(decode),
                    }
                }
                Mixer::Gdn => {
                    let spec = gdn;
                    let cd = spec.conv_dim()?;
                    let state = &engine.state.gdn[layout.state_index as usize];
                    let conv = catalog.bf16(&format!("{prefix}linear_attn.conv1d.weight"))?;
                    let bind = |active_rows| {
                        GdnOps::prepare(
                            &engine.device.ctx,
                            &spec,
                            active_rows as u32,
                            GdnBindings {
                                qkv: matrix(&w.qkv, s, 0, active_rows, cd as usize)?,
                                a: matrix(&mw.ab, s, 0, active_rows, spec.hv as usize)?,
                                b: matrix(
                                    &mw.ab,
                                    s,
                                    rows * spec.hv as usize,
                                    active_rows,
                                    spec.hv as usize,
                                )?,
                                conv_weight: matrix(
                                    conv,
                                    s,
                                    0,
                                    cd as usize,
                                    spec.conv_width as usize,
                                )?,
                                a_log: weight("linear_attn.A_log")?,
                                dt_bias: weight("linear_attn.dt_bias")?,
                                core_out: TensorMut(heads(
                                    &w.attn,
                                    s,
                                    0,
                                    active_rows,
                                    spec.hv,
                                    spec.dv,
                                )?),
                                s: TensorMut(view(
                                    &state.s,
                                    s,
                                    0,
                                    [spec.hv, spec.dk, spec.dv],
                                    [
                                        u64::from(spec.dk) * u64::from(spec.dv),
                                        u64::from(spec.dv),
                                        1,
                                    ],
                                )?),
                                history: TensorMut(view(
                                    &state.history,
                                    s,
                                    0,
                                    [cd, spec.history_width],
                                    [u64::from(spec.history_width), 1],
                                )?),
                            },
                            common.clone(),
                            // The planner's derived chunkwise-P2 split for a plan bound
                            // at `active_rows`; steps of at most MIN_VERIFY_ROWS keep 0.
                            gdn_policy.cw_p2_for_step(active_rows as u32),
                        )
                    };
                    BoundMixer::Gdn {
                        norm: weight("linear_attn.norm.weight")?,
                        prefill: PrefillPlans::bind(bind(rows)?, rows, |plan, rows| {
                            plan.for_rows(rows, None)
                        })?,
                        verify: Vec::new(),
                        decode: Box::new(bind(1)?),
                    }
                }
            };
            layers.push(Layer {
                linear,
                input_norm: weight("input_layernorm.weight")?,
                post_norm: weight("post_attention_layernorm.weight")?,
                mixer,
            });
        }
        let module = crate::backend::cuda::cubin::module(&engine.device.ctx, "q38_common")?;
        Ok(Self {
            attention,
            gdn,
            checkpoint_dir: checkpoint_dir
                .canonicalize()
                .context("resolve Q38 checkpoint directory")?,
            vision: None,
            vision_bytes: 0,
            positions,
            active_positions,
            gather_positions: crate::backend::cuda::cubin::module(
                &engine.device.ctx,
                "q38_vision_io",
            )?
            .load_function("q38_text_positions")?,
            catalog,
            arena,
            scratch: scratch_buffer,
            kv_scratch,
            mask,
            layers,
            common,
            advance: module.load_function("q38_advance")?,
            prepared: BTreeSet::new(),
            workspace_rows: u32::try_from(rows)?,
        })
    }
    fn prepare(
        &mut self,
        blas: &Blas,
        rows: u32,
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<()> {
        self.prepare_dimensions(blas, rows, spec.vocab, spec.hidden)
    }
    fn prepare_dimensions(
        &mut self,
        blas: &Blas,
        rows: u32,
        vocab: u32,
        hidden: u32,
    ) -> Result<()> {
        if self.prepared.contains(&rows) {
            return Ok(());
        }
        for layer in &mut self.layers {
            layer.linear.prepare(blas, rows)?;
        }
        blas.prepare_fixed(1, vocab, hidden)?;
        self.prepared.insert(rows);
        Ok(())
    }
    pub(super) fn prepare_prefill_rows(
        &mut self,
        blas: &Blas,
        rows: u32,
        vocab: u32,
        hidden: u32,
    ) -> Result<()> {
        let bound = bucketed_prefill_rows(rows, self.workspace_rows)?;
        self.prepare_dimensions(blas, bound, vocab, hidden)
    }
    fn linear_rows(&self, rows: u32, phase: ExecutionPhase) -> Result<u32> {
        if phase == ExecutionPhase::Decode {
            return Ok(rows);
        }
        let bound = bucketed_prefill_rows(rows, self.workspace_rows)?;
        ensure!(
            self.prepared.contains(&bound),
            "Q38 prefill Lt bucket {bound} was not bound before Ready"
        );
        Ok(bound)
    }
    fn ensure_complete(&self) -> Result<()> {
        // These allocations own every address retained by the operator plans.
        ensure!(
            self.catalog.bytes > 0
                && self.arena.as_ref().is_none_or(|a| !a.is_empty())
                && self.mask.len() == 64,
            "Q38 owners missing"
        );
        Ok(())
    }
}
impl Engine {
    pub(super) fn bind_mtp_snapshots(
        &mut self,
        snapshots: &[super::super::state::GdnState],
        verify_rows: usize,
    ) -> Result<()> {
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let stream = &self.device.stream;
        let g = q.gdn;
        let cd = g.conv_dim()?;
        let mut states = snapshots.iter();
        for layer in &mut q.layers {
            if let BoundMixer::Gdn {
                prefill, verify, ..
            } = &mut layer.mixer
            {
                let state = states.next().context("missing GDN snapshot allocation")?;
                let s = view(
                    &state.s,
                    stream,
                    0,
                    [4, g.hv, g.dk, g.dv],
                    [
                        u64::from(g.hv * g.dk * g.dv),
                        u64::from(g.dk * g.dv),
                        u64::from(g.dv),
                        1,
                    ],
                )?;
                let h = view(
                    &state.history,
                    stream,
                    0,
                    [4, cd, g.history_width],
                    [
                        u64::from(cd * g.history_width),
                        u64::from(g.history_width),
                        1,
                    ],
                )?;
                *verify = (1..=verify_rows)
                    .map(|rows| prefill.full.for_rows(rows as u32, Some((s, h))))
                    .collect::<Result<_>>()?;
            }
        }
        ensure!(states.next().is_none(), "extra GDN snapshot allocation");
        Ok(())
    }

    pub(super) fn reset_qwen_positions(&mut self) -> Result<()> {
        if let Some(q) = self.qwen.as_mut() {
            let positions =
                crate::vision_input::PositionPlan::text(1, self.capacity)?.packed_axes()?;
            self.device
                .stream
                .memcpy_htod(&positions, &mut q.positions)?;
        }
        Ok(())
    }

    pub fn prefill_multimodal(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        mut trace: Option<&mut Trace>,
    ) -> Result<u32> {
        self.closure.assert_ready()?;
        ensure!(
            !trace.as_ref().is_some_and(|t| t.all_layers) || input.input_ids.len() <= self.chunk,
            "all-layer multimodal trace requires one complete prompt chunk"
        );
        self.prepare_multimodal(input, trace.as_deref_mut())?;
        let result = self.prefill(input.input_ids, trace);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    pub(super) fn prepare_multimodal(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        trace: Option<&mut Trace>,
    ) -> Result<()> {
        ensure!(self.qwen.is_some(), "packed images require Qwen");
        ensure!(
            self.state.kv.iter().all(|kv| kv.compact.is_none()),
            "8-bit KV is qualified for text only"
        );
        ensure!(
            !self.poisoned && self.position == 0 && self.images.is_none(),
            "multimodal prefill requires a healthy, reset sequence"
        );
        let positions = input
            .validate(&self.model_spec, self.capacity)?
            .packed_axes()?;
        let request_bytes = super::vision_runtime::VisionRuntime::request_bytes(
            &self.model_spec,
            input,
            self.capacity,
        )?;
        self.prepare_vision_weights(request_bytes)?;
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let vision = q.vision.as_ref().context("vision runtime missing")?;
        let (free, total) = self.device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16)
                .is_some_and(|n| request_bytes <= n),
            "vision request exceeds memory budget: required={request_bytes} free={free}"
        );
        self.poisoned = true;
        let request = vision.prepare(&self.device, &mut self.blas, input, self.capacity)?;
        let images = vision.encode(&self.device.stream, &self.blas, request, trace)?;
        self.device
            .stream
            .memcpy_htod(&positions, &mut q.positions)?;
        self.device.stream.synchronize()?;
        self.images = Some(images);
        self.poisoned = false;
        Ok(())
    }

    pub fn prepare_resident_vision(&mut self) -> Result<()> {
        if self.model_spec.vision.is_some() {
            self.prepare_vision_weights(0)?;
        }
        Ok(())
    }

    pub(crate) fn check_resident_vision_budget(
        dir: &Path,
        options: &crate::spec::Options,
    ) -> Result<()> {
        let spec = crate::model_spec::ModelSpec::read(dir)?;
        if spec.vision.is_none() {
            return Ok(());
        }
        let checkpoint = crate::checkpoint::MappedCheckpoint::open(dir)?;
        let main_bytes = checkpoint
            .manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?
            .resident_bytes;
        let vision_bytes = checkpoint
            .manifest(&spec, crate::checkpoint::LoadPolicy::VisionOnly)?
            .resident_bytes;
        let kv_bytes = spec
            .layers
            .iter()
            .filter(|layer| layer.mixer == Mixer::FullAttention)
            .count()
            .checked_mul(options.capacity)
            .and_then(|n| n.checked_mul(spec.kv_heads as usize))
            .and_then(|n| n.checked_mul(spec.head_dim as usize))
            .and_then(|n| n.checked_mul(4))
            .context("resident KV byte count overflow")?;
        let mtp_bytes = crate::mtp::attachment_bytes(&spec, options.capacity, 1, true)?;
        let minimum = [main_bytes, vision_bytes, kv_bytes, mtp_bytes]
            .into_iter()
            .try_fold(0usize, |sum, n| {
                sum.checked_add(n).context("resident byte count overflow")
            })?;
        let device = Device::new(options.device)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16).is_some_and(|n| minimum <= n),
            "co-resident MTP/vision/KV minimum exceeds memory budget: main_weights_bytes={main_bytes} mtp_minimum_bytes={mtp_bytes} vision_bytes={vision_bytes} kv_bytes={kv_bytes} minimum_required_bytes={minimum} free_bytes={free} reserve_bytes={}",
            total / 16
        );
        Ok(())
    }

    fn prepare_vision_weights(&mut self, request_bytes: usize) -> Result<()> {
        let kv_bytes = self
            .state
            .kv
            .iter()
            .map(|v| (v.k.len() + v.v.len()) * size_of::<bf16>())
            .sum::<usize>();
        let mtp_bytes = if self.mtp.is_some() {
            crate::mtp::attachment_bytes(&self.model_spec, self.capacity, self.chunk, true)?
        } else {
            0
        };
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        if q.vision.is_none() {
            let manifest = crate::checkpoint::MappedCheckpoint::open(&q.checkpoint_dir)?
                .manifest(&self.model_spec, crate::checkpoint::LoadPolicy::VisionOnly)?;
            let needed = manifest
                .resident_bytes
                .checked_add(request_bytes)
                .context("vision resident/workspace overflow")?;
            let (free, total) = self.device.ctx.mem_get_info()?;
            eprintln!(
                "{}",
                serde_json::json!({"serve_resident_budget": {
                    "main_weights_bytes": self.weights.bytes, "mtp_attachment_bytes": mtp_bytes,
                    "vision_weights_bytes": manifest.resident_bytes, "kv_bytes": kv_bytes,
                    "request_bytes": request_bytes, "free_bytes": free, "reserve_bytes": total / 16,
                    "additional_required_bytes": needed, "chunk": self.chunk,
                }})
            );
            ensure!(
                free.checked_sub(total / 16).is_some_and(|n| needed <= n),
                "co-resident MTP/vision/KV exceeds memory budget: mtp_bytes={mtp_bytes} vision_bytes={} kv_bytes={kv_bytes} request_bytes={request_bytes} additional_required_bytes={needed} free_bytes={free} reserve_bytes={}",
                manifest.resident_bytes,
                total / 16
            );
            self.poisoned = true;
            let mut catalog = WeightCatalog::load_with_policy(
                &q.checkpoint_dir,
                &self.model_spec,
                &self.device,
                crate::checkpoint::LoadPolicy::VisionOnly,
            )?;
            q.vision = Some(super::vision_runtime::VisionRuntime::from_catalog(
                &self.model_spec,
                &mut catalog,
                &self.device,
                q.common.clone(),
            )?);
            ensure!(catalog.tensors.is_empty(), "vision weights left unbound");
            q.vision_bytes = catalog.bytes;
            self.device.finish_upload()?;
            self.poisoned = false;
        }
        Ok(())
    }

    pub(super) fn load_qwen(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        mode: super::SetupMode,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        require_qwen35_geometry(&model_spec)?;
        ensure!(model_spec.moe.is_none(), "MoE FFN execution is not bound");
        let write = !matches!(mode, super::SetupMode::Ready);
        let reserve_mtp = matches!(mode, super::SetupMode::Mtp { .. });
        let verify_rows = match mode {
            super::SetupMode::Mtp { verify_rows } => verify_rows,
            _ => crate::mtp::MAX_VERIFY_ROWS,
        };
        let started = Instant::now();
        let attention = FullAttentionSpec::from_model(&model_spec)?;
        let gdn = GdnSpec::from_model(&model_spec)?;
        let kv_width = attention.hkv * attention.d;
        let config = model_spec.config();
        let capture_layers = settings
            .draft_model
            .as_deref()
            .map(|path| crate::dspark::read_capture_layers(path, &config))
            .transpose()?;
        ensure!(
            !settings.spec_budget.is_tree(),
            "Qwen DSpark proposal/tree execution is not bound"
        );
        let capture_row_bytes = capture_layers
            .as_ref()
            .map(|ids| {
                ids.len()
                    .checked_mul(config.hidden_size)
                    .and_then(|n| n.checked_mul(size_of::<bf16>()))
                    .context("capture row byte count overflow")
            })
            .transpose()?
            .unwrap_or(0);
        ensure!(
            capacity >= 32 && capacity <= model_spec.max_position as usize,
            "invalid Q38 capacity"
        );
        // Reject an incomplete operator plan before allocating model weights.
        Runtime::workspace(&model_spec, 1, capacity, GdnPolicy::LEGACY)?;
        let runtime = settings
            .runtime_dir
            .as_deref()
            .context("Q38 setup requires runtime directory")?
            .to_owned();
        std::fs::create_dir_all(&runtime)?;
        let closure = Rc::new(super::super::closure::Closure::new(
            write,
            "Q38 uses fixed operator plans".into(),
        ));
        let device = Device::new(ordinal)?;
        let gdn_policy = GdnPolicy::for_device(device.info.arch, device.info.sms);
        let prefix_cache = settings.prefix_cache;
        let mut catalog = if prefix_cache {
            WeightCatalog::load_for_prefix_cache(dir, &model_spec, &device)?
        } else {
            WeightCatalog::load(dir, &model_spec, &device)?
        };
        let prefix_identity_load_ms = catalog.load_wall_ms;
        let mut prefix_identity = if prefix_cache {
            let upload_sha = catalog
                .upload_content_sha256
                .clone()
                .context("prefix-cache upload hash missing")?;
            let upload_bytes = catalog
                .upload_hashed_bytes
                .context("prefix-cache upload byte count missing")?;
            ensure!(
                upload_bytes == catalog.bytes,
                "prefix-cache upload hash byte count differs from uploaded weights"
            );
            Some(PrefixIdentity::from_prefix_cache(
                dir,
                &text_rope_context_sha256(capacity)?,
                &model_spec,
                &upload_sha,
                upload_bytes,
            )?)
        } else {
            None
        };
        if let Some(identity) = prefix_identity.as_mut() {
            identity.attention_kv_dtype = settings.kv_format.tag().into();
        }
        let weights = catalog.take_globals()?;
        let s = &device.stream;
        let state = super::super::state::StateStore::new_with_format(
            &model_spec,
            capacity,
            settings.kv_format,
            s,
        )?;
        let kv_scratch = if settings.kv_format == crate::kv::KvFormat::Bf16 {
            None
        } else {
            Some((
                s.alloc_zeros::<bf16>(capacity * kv_width as usize)?,
                s.alloc_zeros::<bf16>(capacity * kv_width as usize)?,
            ))
        };
        let capture_checkpoint = Some(super::super::state::StateCheckpoint::new(
            &state,
            &model_spec,
            1,
            s,
        )?);
        let arena_bytes = (0..model_spec.layers.len())
            .map(|i| {
                LayerDequantPlan::from_manifest(
                    &catalog.manifest,
                    i as u32,
                    model_spec.linear_format,
                )
                .map(|p| p.bytes)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing Q38 layers")?;
        let arena = if arena_bytes == 0 {
            None
        } else {
            Some(s.alloc_zeros::<u8>(usize::try_from(arena_bytes)?)?)
        };
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        if model_spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Block128 {
            blas.enable_fp8(&device.ctx)?;
        }
        device.finish_upload()?;
        let (free, total) = device.ctx.mem_get_info()?;
        let mtp_budget = reserve_mtp
            .then(|| crate::mtp::AttachmentBudget::new(&model_spec, capacity))
            .transpose()?;
        let budget = crate::prefill::WorkspaceBudget::new(
            crate::prefill::WorkspaceRequest {
                capacity,
                requested: requested_chunk,
                free_after_resident: free,
                total,
                alignment: CHUNK_ALIGNMENT,
                row_bytes: Runtime::row_bytes(&model_spec)?
                    .checked_add(mtp_budget.as_ref().map_or(0, |b| b.row_bytes))
                    .and_then(|bytes| bytes.checked_add(capture_row_bytes))
                    .context("combined row bytes overflow")?,
                fixed_transient_bytes: Runtime::fixed_bytes(&model_spec, capacity)?
                    .checked_add(mtp_budget.as_ref().map_or(0, |b| b.fixed_bytes))
                    .context("combined fixed bytes overflow")?,
            },
            |rows| {
                let mut target = Runtime::workspace(&model_spec, rows, capacity, gdn_policy)?;
                if settings.kv_format != crate::kv::KvFormat::Bf16 {
                    target = target.max(
                        crate::op_contract::AttentionWorkspace::new(
                            &attention,
                            verify_rows as u32,
                            capacity as u32,
                            limits(verify_rows, capacity)?.key_chunk,
                        )?
                        .bytes as usize,
                    );
                }
                target
                    .checked_add(
                        mtp_budget
                            .as_ref()
                            .map(|b| b.workspace(rows))
                            .transpose()?
                            .unwrap_or(0),
                    )
                    .context("Q38 combined workspace overflow")
            },
            |admitted| Ok(default_chunk(admitted)),
        )?;
        let rows = budget.plan.chunk;
        ensure!(
            !reserve_mtp || rows >= verify_rows,
            "MTP chunk {rows} cannot fit {verify_rows} verification rows"
        );
        let target_workspace = Runtime::workspace(&model_spec, rows, capacity, gdn_policy)?.max(
            if settings.kv_format == crate::kv::KvFormat::Bf16 {
                0
            } else {
                crate::op_contract::AttentionWorkspace::new(
                    &attention,
                    verify_rows as u32,
                    capacity as u32,
                    limits(verify_rows, capacity)?.key_chunk,
                )?
                .bytes as usize
            },
        );
        eprintln!(
            "{}",
            serde_json::json!({
                "q38_memory_plan": budget,
                "mtp_attachment_planned_bytes": if reserve_mtp { crate::mtp::attachment_bytes(&model_spec, capacity, rows, true)? } else { 0 },
                "resident_weights_bytes": weights.bytes,
                "prefix_cache": {
                    "enabled": prefix_cache,
                    "load_wall_ms": prefix_identity_load_ms,
                    "hashed_bytes": catalog.upload_hashed_bytes.unwrap_or(0)
                },
                "persistent_state_bytes": state.bytes(),
                "kv_format": settings.kv_format.tag(),
                "kv_prefill_scratch_bytes": kv_scratch.as_ref().map_or(0,|(k,v)|(k.len()+v.len())*size_of::<bf16>()),
                "capture_checkpoint_bytes": capture_checkpoint.as_ref().context("capture checkpoint missing")?.bytes(),
                "target_hidden_capture_bytes": rows.checked_mul(capture_row_bytes).context("capture byte count overflow")?,
                "layer_arena_bytes": arena_bytes,
                "attention_prefill_preprocess_bytes": FullAttentionOps::preprocess_workspace(&attention, limits(rows, capacity)?)?.bytes,
                "attention_prefill_lse_bytes": rows * model_spec.q_heads as usize * size_of::<f32>(),
                "attention_decode_workspace_bytes": FullAttentionOps::workspace(&attention, limits(1, capacity)?)?.bytes,
                "attention_nested_verify_workspace_bytes": FullAttentionOps::workspace(&attention, limits(rows.min(5), capacity)?)?.bytes,
                "attention_prefill_backend": "FA2 d256",
                "attention_decode_backend": "TCMQA d256",
                "gdn_prefill_workspace_bytes": GdnOps::workspace(&gdn, u32::try_from(rows)?, gdn_policy)?.bytes,
                "gdn_decode_workspace_bytes": GdnOps::workspace(&gdn, 1, GdnPolicy::LEGACY)?.bytes,
                "bound_prefill_max_rows": rows,
                "bound_decode_max_rows": 1,
                "gdn_plan": gdn_policy.identity(),
                "gdn_workspace_policy": "max_complete_prebound_plan_not_sum",
                "selection_source": "fixed"
            })
        );
        let scratch_buffer = s.alloc_zeros::<u8>(target_workspace)?;
        let work = Work::new(s, &model_spec, rows)?;
        let flash = Flash::new(s.clone(), rows, model_spec.q_heads as usize)?;
        let hidden_capture = capture_layers
            .as_ref()
            .map(|ids| HiddenCapture::new(ids, &config, rows, s))
            .transpose()?;
        let mut engine = Self {
            #[cfg(test)]
            capture_fail_after_layer: None,
            #[cfg(test)]
            capture_fail_stage: None,
            qwen_device_prefix: false,
            forward_rows: (1..=rows.min(16)).collect(),
            graphs: vec![Graph::Decode],
            closure,
            graph: None,
            poisoned: false,
            spec_budget: settings.spec_budget,
            spec_graph: settings.spec_graph,
            tree_builder: settings.tree_builder,
            compare_builders: false,
            profile_rounds: false,
            runtime,
            spec: None,
            hidden_capture,
            weights,
            state,
            capture_checkpoint,
            work,
            ops,
            blas,
            flash: Some(flash),
            decode: None,
            device,
            config,
            model_spec,
            capacity,
            chunk: rows,
            workspace_rows: rows,
            trace_layer: 0,
            position: 0,
            prefix_identity,
            prefix_identity_load_ms,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
            qwen: None,
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
        };
        engine.qwen = Some(Runtime::new(
            &engine,
            dir,
            catalog,
            arena,
            scratch_buffer,
            kv_scratch,
            verify_rows,
        )?);
        engine.prepare(1)?;
        engine.prepare_prefill_plans()?;
        engine
            .qwen
            .as_ref()
            .context("Q38 runtime missing")?
            .ensure_complete()?;
        engine.closure.freeze_fixed()?;
        if !write {
            engine.reset()?;
            engine.prepare_graphs()?;
            engine.closure.expect_captures(engine.graphs.len())?;
            engine.check_graphs()?;
            engine.reset()?;
            engine.closure.ready(started)?;
        }
        Ok(engine)
    }
    pub(super) fn prepare_qwen(&mut self, rows: usize) -> Result<()> {
        self.qwen.as_mut().context("Q38 runtime missing")?.prepare(
            &self.blas,
            u32::try_from(rows)?,
            &self.model_spec,
        )
    }
    pub(super) fn prepare_prefill_rows(&mut self, rows: usize) -> Result<()> {
        if self.qwen.is_some() {
            let vocab = self.model_spec.vocab;
            let hidden = self.model_spec.hidden;
            let q = self.qwen.as_mut().context("Q38 runtime missing")?;
            return q.prepare_prefill_rows(&self.blas, u32::try_from(rows)?, vocab, hidden);
        }
        self.prepare(rows)
    }
    pub(super) fn qwen_advance(&mut self) -> Result<()> {
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        let meta = self.state.seq_ref(&self.device.stream)?;
        unsafe {
            self.device
                .stream
                .launch_builder(&q.advance)
                .arg(&meta.ptr)
                .launch(grid(1, 1))?;
        }
        self.work.advance(&self.ops, &self.device.stream)?;
        self.state.compact_commit(&self.device.stream, false)
    }
    pub(super) fn qwen_layer_begin(
        &mut self,
        i: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        if step.input == crate::backend::Input::Ids
            && !crate::linear_plan::direct_fp8_rows(step.rows)
        {
            self.qwen.as_mut().context("Q38 runtime missing")?.layers[i]
                .linear
                .materialize(&self.blas, &self.device.stream)?;
        }
        Ok(())
    }
    pub(super) fn qwen_mixer(
        &mut self,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        let layer = &q.layers[i];
        let attention = q.attention;
        let gdn = q.gdn;
        let q_width = self.model_spec.q_width;
        let kv_width = attention.hkv * attention.d;
        let conv_width = 2 * gdn.hk * gdn.dk + gdn.hv * gdn.dv;
        let value_width = gdn.hv * gdn.dv;
        let hidden = self.model_spec.hidden as usize;
        let s = &self.device.stream;
        let rows = step.rows;
        let phase = if step.input == crate::backend::Input::Token {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let linear_rows = q.linear_rows(rows as u32, phase)? as usize;
        let w = &mut self.work;
        let mw = w.mixer.as_ref().context("Q38 mixer work missing")?;
        let meta = self.state.seq_ref(s)?;
        let domain = RowDomain {
            meta,
            row_base: 0,
            row_capacity: rows as u32,
        };
        match &layer.mixer {
            BoundMixer::Attention {
                image_prefill,
                text_prefill,
                decode,
                compact_small,
            } => {
                for (role, width, offset) in [
                    ("self_attn.q_proj.weight", 2 * q_width as usize, 0),
                    (
                        "self_attn.k_proj.weight",
                        kv_width as usize,
                        self.workspace_rows * 2 * q_width as usize,
                    ),
                    (
                        "self_attn.v_proj.weight",
                        kv_width as usize,
                        self.workspace_rows * (2 * q_width as usize + kv_width as usize),
                    ),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, hidden)?,
                        &mut TensorMut(matrix(&w.qkv, s, offset, linear_rows, width)?),
                    )?;
                }
                match phase {
                    ExecutionPhase::Prefill => {
                        if nested_verify_tcmqa(self.qwen_device_prefix, rows)? {
                            let kv = &self.state.kv[self.model_spec.layers[i].state_index as usize];
                            let plans = if kv.compact.is_some() {
                                compact_small
                                    .as_ref()
                                    .context("compact verify plan missing")?
                            } else {
                                text_prefill
                            };
                            let narrowed = plans.select(rows, &FullAttentionPlan::for_rows)?;
                            FullAttentionOps::run(&narrowed, s, meta, scratch(&q.scratch, s))?;
                        } else {
                            let kv = &self.state.kv[self.model_spec.layers[i].state_index as usize];
                            if let Some(c) = &kv.compact {
                                let (k, v) = q
                                    .kv_scratch
                                    .as_ref()
                                    .context("compact KV prefill scratch missing")?;
                                ensure!(
                                    self.images.is_none(),
                                    "8-bit KV is qualified for text only"
                                );
                                c.dequant_prefix(
                                    c.view(&kv.k, &kv.v, s),
                                    meta,
                                    k,
                                    v,
                                    self.position,
                                    s,
                                )?;
                            }
                            let plans = if self.images.is_some() {
                                image_prefill
                            } else {
                                text_prefill
                            };
                            let narrowed = plans.select(rows, &FullAttentionPlan::for_rows)?;
                            let prefill: &FullAttentionPlan = &narrowed;
                            FullAttentionOps::preprocess(prefill, s, meta, scratch(&q.scratch, s))?;
                            let kv = &self.state.kv[self.model_spec.layers[i].state_index as usize];
                            let (k_cache, v_cache) = if kv.compact.is_some() {
                                let (k, v) = q
                                    .kv_scratch
                                    .as_ref()
                                    .context("compact KV prefill scratch missing")?;
                                (k, v)
                            } else {
                                (&kv.k, &kv.v)
                            };
                            let shape = Shape {
                                rows,
                                start: self.position,
                                q_heads: attention.hq as usize,
                                kv_heads: attention.hkv as usize,
                                capacity: self.capacity,
                                head_dim: 256,
                                causal: true,
                            };
                            self.flash
                                .as_mut()
                                .context("Q38 prefill attention is not bound")?
                                .run(
                                    &mw.q,
                                    q_width as usize,
                                    k_cache,
                                    v_cache,
                                    &mut w.attn,
                                    shape,
                                )?;
                            if let Some(c) = &kv.compact {
                                let (k, v) = q
                                    .kv_scratch
                                    .as_ref()
                                    .context("compact KV prefill scratch missing")?;
                                c.finish(
                                    c.view(&kv.k, &kv.v, s),
                                    meta,
                                    Some((k, v, rows)),
                                    true,
                                    s,
                                )?;
                            }
                        }
                    }
                    ExecutionPhase::Decode => {
                        FullAttentionOps::run(decode, s, meta, scratch(&q.scratch, s))?;
                    }
                }
                q.common.gate.sigmoid_mul(
                    s,
                    domain,
                    heads(&w.attn, s, 0, rows, attention.hq, 256)?,
                    heads(&mw.gate, s, 0, rows, attention.hq, 256)?,
                    &mut TensorMut(heads(&w.attn, s, 0, rows, attention.hq, 256)?),
                )?;
                layer.linear.run(
                    &self.blas,
                    "self_attn.o_proj.weight",
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, q_width as usize)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
                )?;
            }
            BoundMixer::Gdn {
                norm,
                prefill,
                verify,
                decode,
            } => {
                let spec = gdn;
                let conv = conv_width as usize;
                let value = value_width as usize;
                let hv = spec.hv as usize;
                for (role, output, offset, width) in [
                    ("linear_attn.in_proj_qkv.weight", &w.qkv, 0, conv),
                    ("linear_attn.in_proj_z.weight", &mw.gate, 0, value),
                    ("linear_attn.in_proj_a.weight", &mw.ab, 0, hv),
                    (
                        "linear_attn.in_proj_b.weight",
                        &mw.ab,
                        self.workspace_rows * hv,
                        hv,
                    ),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, hidden)?,
                        &mut TensorMut(matrix(output, s, offset, linear_rows, width)?),
                    )?;
                }
                let narrowed;
                let plan: &GdnPlan = if self.qwen_device_prefix {
                    ensure!(
                        phase == ExecutionPhase::Prefill,
                        "MTP verify must remain Prefill"
                    );
                    verify
                        .get(rows.checked_sub(1).context("MTP verify requires rows")?)
                        .context("MTP verify GDN plan missing")?
                } else {
                    match phase {
                        ExecutionPhase::Prefill => {
                            narrowed =
                                prefill.select(rows, &|plan, rows| plan.for_rows(rows, None))?;
                            &narrowed
                        }
                        ExecutionPhase::Decode => decode.as_ref(),
                    }
                };
                if self.qwen_device_prefix {
                    GdnOps::run_verify(plan, s, meta, scratch(&q.scratch, s))?;
                } else {
                    GdnOps::run(plan, s, meta, phase, scratch(&q.scratch, s))?;
                }
                q.common.norm.rms_gated(
                    s,
                    domain,
                    GatedRmsInput {
                        x: heads(&w.attn, s, 0, rows, spec.hv, spec.dv)?,
                        weight: *norm,
                        gate: heads(&mw.gate, s, 0, rows, spec.hv, spec.dv)?,
                    },
                    &mut TensorMut(heads(&w.attn, s, 0, rows, spec.hv, spec.dv)?),
                    spec.rms_eps,
                )?;
                layer.linear.run(
                    &self.blas,
                    "linear_attn.out_proj.weight",
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, value)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
                )?;
            }
        }
        let all = all_layers(trace);
        if all || i == self.trace_layer {
            let name = debug_name(trace, i, "mixer_out");
            let mixer_width = match &layer.mixer {
                BoundMixer::Attention { .. } => q_width,
                BoundMixer::Gdn { .. } => value_width,
            };
            trace_value(trace, s, &name, &w.attn, rows, mixer_width as usize)?;
            let name = debug_name(trace, i, "o_proj");
            trace_value(trace, s, &name, &w.out, rows, hidden)?;
            if matches!(self.model_spec.layers[i].mixer, Mixer::Gdn)
                && (!all || i == 0)
                && let Some(t) = trace.as_mut()
            {
                let state = &self.state.gdn[self.model_spec.layers[i].state_index as usize];
                let prefix = if all {
                    format!("{}.{i}", t.prefix)
                } else {
                    t.prefix.to_owned()
                };
                let suffix = if all { ".e2e" } else { "" };
                t.trace.add(
                    format!("{prefix}.debug.gdn_s{suffix}"),
                    vec![gdn.hv as usize, gdn.dk as usize, gdn.dv as usize],
                    s.clone_dtoh(&state.s)?,
                )?;
                t.trace.add(
                    format!("{prefix}.debug.gdn_history{suffix}"),
                    vec![conv_width as usize, gdn.history_width as usize],
                    s.clone_dtoh(&state.history)?,
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn qwen_operator(
        &mut self,
        op: crate::backend::Operator,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        use crate::backend::Operator as O;
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        let layer = &q.layers[i];
        let s = &self.device.stream;
        let w = &mut self.work;
        let rows = step.rows;
        let h = self.model_spec.hidden as usize;
        let f = self.model_spec.intermediate as usize;
        let phase = if step.input == crate::backend::Input::Token {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let linear_rows = q.linear_rows(rows as u32, phase)? as usize;
        let domain = RowDomain {
            meta: self.state.seq_ref(s)?,
            row_base: 0,
            row_capacity: rows as u32,
        };
        match op {
            O::Embed => {
                let (table, _table) = q.positions.device_ptr(s);
                let (active, _active) = q.active_positions.device_ptr(s);
                let max_rows = self.workspace_rows as u32;
                unsafe {
                    s.launch_builder(&q.gather_positions)
                        .arg(&table)
                        .arg(&active)
                        .arg(&domain.meta.ptr)
                        .arg(&max_rows)
                        .launch(grid(
                            (self.workspace_rows * 3).div_ceil(256).min(65535),
                            256,
                        ))?;
                }
                let ids = if step.input == crate::backend::Input::Token {
                    &w.token
                } else {
                    &w.ids
                };
                unsafe {
                    s.launch_builder(&self.ops.embed)
                        .arg(&self.weights.embed)
                        .arg(ids)
                        .arg(&mut w.x)
                        .arg(&(rows as i32))
                        .arg(&(h as i32))
                        .launch(flat(rows * h))?;
                }
                if step.input == crate::backend::Input::Ids
                    && let Some(images) = &self.images
                {
                    images.splice(
                        s,
                        self.position,
                        &mut TensorMut(matrix(&w.x, s, 0, rows, h)?),
                    )?;
                }
                trace_value(trace, s, "embed", &w.x, rows, h)?;
            }
            O::Norm => {
                if i == 0
                    || !self
                        .hidden_capture
                        .as_ref()
                        .is_some_and(|capture| capture.layer_slots[i - 1].is_some())
                {
                    q.common.norm.rms_with(
                        s,
                        domain,
                        heads(&w.x, s, 0, rows, 1, h as u32)?,
                        layer.input_norm,
                        &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                        RmsOptions {
                            mode: RmsMode::ZeroCentered,
                            eps: self.model_spec.rms_eps,
                        },
                    )?;
                }
                if all_layers(trace) || i == self.trace_layer {
                    let name = debug_name(trace, i, "norm1");
                    trace_value(trace, s, &name, &w.n, rows, h)?;
                }
            }
            O::PostNorm | O::NextNorm => {
                if matches!(op, O::PostNorm) {
                    q.common.norm.rms_add_with(
                        s,
                        domain,
                        ResidualRmsInput {
                            x: &mut TensorMut(heads(&w.x, s, 0, rows, 1, h as u32)?),
                            residual: heads(&w.out, s, 0, rows, 1, h as u32)?,
                            weight: layer.post_norm,
                        },
                        &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                        RmsOptions {
                            mode: RmsMode::ZeroCentered,
                            eps: self.model_spec.rms_eps,
                        },
                    )?;
                    if all_layers(trace) || i == self.trace_layer {
                        let name = debug_name(trace, i, "post_attention");
                        trace_value(trace, s, &name, &w.x, rows, h)?;
                        let name = debug_name(trace, i, "norm2");
                        trace_value(trace, s, &name, &w.n, rows, h)?;
                    }
                } else {
                    if let Some(capture) = &mut self.hidden_capture
                        && let Some(slot) = capture.layer_slots[i]
                    {
                        let weight = if let Some(next) = q.layers.get(i + 1) {
                            next.input_norm
                        } else {
                            vector(&self.weights.norm, s)?
                        };
                        self.ops.add_rmsnorm_capture(
                            s,
                            Some(domain),
                            ResidualRmsCapture {
                                input: ResidualRmsInput {
                                    x: &mut TensorMut(heads(&w.x, s, 0, rows, 1, h as u32)?),
                                    residual: heads(&w.out, s, 0, rows, 1, h as u32)?,
                                    weight,
                                },
                                capture: &mut TensorMut(view(
                                    &capture.values,
                                    s,
                                    0,
                                    [rows as u32, capture.taps as u32, h as u32],
                                    [capture.width as u64, h as u64, 1],
                                )?),
                                slot,
                            },
                            &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                            RmsOptions {
                                mode: RmsMode::ZeroCentered,
                                eps: self.model_spec.rms_eps,
                            },
                        )?;
                    } else {
                        unsafe {
                            s.launch_builder(&self.ops.residual)
                                .arg(&mut w.x)
                                .arg(&w.out)
                                .arg(&((rows * h) as i32))
                                .launch(flat(rows * h))?;
                        }
                    }
                    if trace.is_some()
                        && (!all_layers(trace)
                            || [0, 3, 7, 31, 32, 47, 63].contains(&i)
                            || self
                                .hidden_capture
                                .as_ref()
                                .is_some_and(|capture| capture.layer_slots[i].is_some()))
                    {
                        let name = if all_layers(trace) {
                            i.to_string()
                        } else {
                            format!("layer.{i}")
                        };
                        trace_value(trace, s, &name, &w.x, rows, h)?;
                    }
                }
            }
            O::GateUp => {
                for (role, offset) in [
                    ("mlp.gate_proj.weight", 0),
                    ("mlp.up_proj.weight", self.workspace_rows * f),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, h)?,
                        &mut TensorMut(matrix(&w.gu, s, offset, linear_rows, f)?),
                    )?;
                }
            }
            O::Activate => {
                let gate = heads(&w.gu, s, 0, rows, 1, f as u32)?;
                let up = heads(&w.gu, s, self.workspace_rows * f, rows, 1, f as u32)?;
                let mut activated = TensorMut(heads(&w.act, s, 0, rows, 1, f as u32)?);
                q.common.gate.swiglu(s, domain, gate, up, &mut activated)?;
            }
            O::Down => layer.linear.run(
                &self.blas,
                "mlp.down_proj.weight",
                phase,
                matrix(&w.act, s, 0, linear_rows, f)?,
                &mut TensorMut(matrix(&w.out, s, 0, linear_rows, h)?),
            )?,
            O::Head => {
                if step.head == Head::None {
                    return Ok(());
                }
                ensure!(
                    step.head == Head::Last,
                    "Q38 all-row speculative head is not bound"
                );
                if !self.hidden_capture.as_ref().is_some_and(|capture| {
                    capture.layer_slots[self.model_spec.layers.len() - 1].is_some()
                }) {
                    q.common.norm.rms_with(
                        s,
                        domain,
                        heads(&w.x, s, 0, rows, 1, h as u32)?,
                        vector(&self.weights.norm, s)?,
                        &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                        RmsOptions {
                            mode: RmsMode::ZeroCentered,
                            eps: self.model_spec.rms_eps,
                        },
                    )?;
                }
                self.blas.linear_typed(
                    phase,
                    LinearWeightRef::Bf16(matrix(
                        &self.weights.head,
                        s,
                        0,
                        self.model_spec.vocab as usize,
                        h,
                    )?),
                    matrix(&w.n, s, (rows - 1) * h, 1, h)?,
                    &mut TensorMut(matrix(&w.logits, s, 0, 1, self.model_spec.vocab as usize)?),
                )?;
                unsafe {
                    s.launch_builder(&self.ops.argmax)
                        .arg(&w.logits)
                        .arg(&mut w.token)
                        .arg(&(self.model_spec.vocab as i32))
                        .launch(grid(1, 1024))?;
                }
                if let Some(t) = trace.as_mut() {
                    let values = s.clone_dtoh(&w.logits)?;
                    t.trace.add(
                        format!("{}.logits", t.prefix),
                        vec![1, 1, self.model_spec.vocab as usize],
                        values.into_iter().map(bf16::to_f32).collect(),
                    )?;
                }
            }
            _ => anyhow::bail!("operator {op:?} is not a Q38 common stage"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::default_chunk;

    fn geometry() -> crate::model_spec::ModelSpec {
        crate::model_spec::ModelSpec::qwen(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )))
            .unwrap(),
            &serde_json::json!({}),
        )
        .unwrap()
    }

    #[test]
    fn capability_check_matches_compiled_mixer_geometries() -> Result<()> {
        let qwen: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let spec = crate::model_spec::ModelSpec::qwen(&qwen, &serde_json::json!({}))?;
        require_qwen35_geometry(&spec)?;
        let minicpm = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        require_qwen35_geometry(&minicpm)?;
        for original in [&spec, &minicpm] {
            for field in [
                "hidden_size",
                "intermediate_size",
                "vocab_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "q_width",
                "max_position_embeddings",
                "rotary_dim",
                "num_hidden_layers",
                "rms_norm_eps",
                "rope_theta",
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
                "gdn",
                "linear_format",
                "architecture",
            ] {
                let mut bad = original.clone();
                match field {
                    "hidden_size" => bad.hidden += 1,
                    "intermediate_size" => bad.intermediate += 1,
                    "vocab_size" => bad.vocab += 1,
                    "num_attention_heads" => bad.q_heads += 1,
                    "num_key_value_heads" => bad.kv_heads += 1,
                    "head_dim" => bad.head_dim += 1,
                    "q_width" => bad.q_width += 1,
                    "max_position_embeddings" => bad.max_position += 1,
                    "rotary_dim" => bad.rotary_dim += 1,
                    "num_hidden_layers" => {
                        bad.layers.pop();
                    }
                    "rms_norm_eps" => bad.rms_eps *= 2.0,
                    "rope_theta" => bad.rope_theta *= 2.0,
                    "linear_num_key_heads" => bad.gdn.as_mut().unwrap().key_heads += 1,
                    "linear_num_value_heads" => bad.gdn.as_mut().unwrap().value_heads += 1,
                    "linear_key_head_dim" => bad.gdn.as_mut().unwrap().key_head_dim += 1,
                    "linear_value_head_dim" => bad.gdn.as_mut().unwrap().value_head_dim += 1,
                    "linear_conv_kernel_dim" => bad.gdn.as_mut().unwrap().conv_kernel += 1,
                    "gdn" => bad.gdn = None,
                    "linear_format" => {
                        bad.linear_format = match original.linear_format {
                            crate::model_spec::LinearFormat::Bf16 => {
                                crate::model_spec::LinearFormat::Fp8E4m3Block128
                            }
                            crate::model_spec::LinearFormat::Fp8E4m3Block128 => {
                                crate::model_spec::LinearFormat::Bf16
                            }
                            crate::model_spec::LinearFormat::Fp8E4m3Channel => {
                                crate::model_spec::LinearFormat::Bf16
                            }
                        }
                    }
                    "architecture" => bad.architecture = crate::model_spec::Architecture::Llama,
                    _ => unreachable!(),
                }
                let error = require_qwen35_geometry(&bad).unwrap_err().to_string();
                assert!(error.contains(&format!("{field}=")), "{field}: {error}");
            }
        }
        let error = Engine::load_qwen(
            Path::new("unused-model"),
            0,
            32,
            Some(1),
            Settings {
                tree_builder: crate::backend::TreeBuilder::Waves,
                spec_budget: crate::backend::SpecBudget::Chain,
                spec_graph: false,
                draft_model: None,
                prefix_cache: false,
                kv_format: crate::kv::KvFormat::Bf16,
                runtime_dir: None,
            },
            super::super::SetupMode::Prepare,
            minicpm,
        )
        .err()
        .context("MoE model reached device setup")?;
        assert!(error.to_string().contains("MoE FFN execution"), "{error}");
        Ok(())
    }

    #[test]
    fn the_default_chunk_is_bounded_by_the_cuda_grid_yz_limit() {
        assert_eq!(default_chunk(100_000), 65_504);
        assert_eq!(default_chunk(100_000) % 32, 0);
        assert_eq!(default_chunk(500), 500);
    }

    /// An admission of 65536 rows selects 65504, a multiple of the budget alignment.
    #[test]
    fn the_default_chunk_cap_preserves_the_budget_alignment() -> Result<()> {
        let budget = WorkspaceBudget::new(
            WorkspaceRequest {
                capacity: 70000,
                requested: None,
                free_after_resident: 1 << 40,
                total: 1 << 40,
                alignment: super::CHUNK_ALIGNMENT,
                row_bytes: 1,
                fixed_transient_bytes: 0,
            },
            |rows| Ok(rows * 4),
            |admitted| Ok(default_chunk(admitted)),
        )?;
        assert_eq!(budget.plan.chunk, 65_504);
        assert_eq!(budget.plan.chunk % super::CHUNK_ALIGNMENT, 0);
        Ok(())
    }

    /// A stand-in plan recording the rows it was narrowed to.
    #[derive(Clone, Debug, PartialEq)]
    struct FakePlan {
        max_rows: usize,
    }
    impl FakePlan {
        fn for_rows(plan: &FakePlan, rows: u32) -> Result<FakePlan> {
            ensure!(
                rows > 0 && rows as usize <= plan.max_rows,
                "rows outside binding"
            );
            Ok(FakePlan {
                max_rows: rows as usize,
            })
        }
    }

    #[test]
    fn select_returns_the_plan_narrowed_to_the_steps_live_rows() {
        let plans = PrefillPlans::bind(FakePlan { max_rows: 2048 }, 2048, FakePlan::for_rows)
            .expect("bind");
        // A step size never seen before narrows host-side at dispatch; no preparation step exists.
        assert_eq!(
            plans.select(1021, &FakePlan::for_rows).unwrap().max_rows,
            1021
        );
        assert_eq!(
            plans.select(700, &FakePlan::for_rows).unwrap().max_rows,
            700
        );
        // Rows 1..=5 borrow prebound small plans.
        for rows in 1..=5 {
            assert_eq!(
                plans.select(rows, &FakePlan::for_rows).unwrap().max_rows,
                rows
            );
        }
        assert!(plans.select(0, &FakePlan::for_rows).is_err());
        assert!(plans.select(2049, &FakePlan::for_rows).is_err());
    }

    #[test]
    fn a_step_only_runs_a_plan_narrowed_to_its_own_rows() {
        let plans = PrefillPlans::bind(FakePlan { max_rows: 2048 }, 2048, FakePlan::for_rows)
            .expect("bind");
        assert_eq!(
            plans.select(1021, &FakePlan::for_rows).unwrap().max_rows,
            1021,
            "a 1021-row step must not run a plan sized for the admitted chunk"
        );
    }

    #[test]
    fn every_step_size_of_a_chunked_prompt_gets_its_own_plan() {
        // 4093 tokens over a 2048 chunk: two 2048 steps and one 997 tail.
        let plans = PrefillPlans::bind(FakePlan { max_rows: 2048 }, 2048, FakePlan::for_rows)
            .expect("bind");
        for step in [2048usize, 997] {
            assert_eq!(
                plans.select(step, &FakePlan::for_rows).unwrap().max_rows,
                step
            );
        }
    }

    use super::*;
    use crate::prefill::{WorkspaceBudget, WorkspaceRequest};

    #[test]
    fn prefill_linear_rows_use_setup_bound_buckets() -> Result<()> {
        for (rows, workspace, bound) in [
            (1u32, 2048u32, 1u32),
            (7, 2048, 7),
            (16, 2048, 16),
            (17, 2048, 32),
            (31, 2048, 32),
            (2037, 2048, 2048),
            (17, 20, 20),
        ] {
            assert_eq!(bucketed_prefill_rows(rows, workspace)?, bound);
        }
        assert!(bucketed_prefill_rows(0, 2048).is_err());
        assert!(bucketed_prefill_rows(2049, 2048).is_err());
        Ok(())
    }

    #[test]
    fn small_prefill_plans_select_structural_rows_without_rebinding() -> Result<()> {
        for capacity in [1, 2, 3, 4, 5, 16, 2048, 4096] {
            let narrow = |&full: &usize, rows: u32| {
                assert_eq!(full, capacity);
                Ok(rows as usize)
            };
            let plans = PrefillPlans::bind(capacity, capacity, narrow)?;
            assert_eq!(plans.small, (1..=5.min(capacity)).collect::<Vec<_>>());
            for rows in 1..=capacity {
                assert_eq!(*plans.select(rows, &narrow)?, rows);
            }
            for rows in 1..=5.min(capacity) {
                ensure!(
                    matches!(plans.select(rows, &narrow)?, Narrowed::Borrowed(_)),
                    "small extent {rows} did not select a prebound plan"
                );
            }
            assert!(plans.select(0, &narrow).is_err());
            assert!(plans.select(capacity + 1, &narrow).is_err());
            assert!(plans.select(usize::MAX, &narrow).is_err());
            let budget =
                Runtime::workspace(&geometry(), capacity, 4096, GdnPolicy::for_device(89, 128))?
                    as u64;
            for rows in 1..=5.min(capacity) {
                assert!(
                    GdnOps::workspace(
                        &crate::op_contract::GdnSpec::q38(),
                        rows as u32,
                        GdnPolicy::LEGACY
                    )?
                    .bytes
                        <= budget
                );
                assert!(
                    FullAttentionOps::preprocess_workspace(
                        &FullAttentionSpec::q38(),
                        limits(rows, 4096)?
                    )?
                    .bytes
                        <= budget
                );
            }
        }
        assert!(PrefillPlans::bind(0, 0, |_, rows| Ok(rows as usize)).is_err());
        Ok(())
    }

    #[test]
    fn nested_verify_routes_only_small_rows_to_tcmqa() -> Result<()> {
        for rows in [1, 4, 65, 2048] {
            assert!(!nested_verify_tcmqa(false, rows)?);
        }
        for rows in 1..=5 {
            assert!(nested_verify_tcmqa(true, rows)?);
        }
        assert!(nested_verify_tcmqa(true, 0).is_err());
        assert!(nested_verify_tcmqa(true, 6).is_err());
        Ok(())
    }

    #[test]
    fn workspace_binds_nested_verify_tcmqa_rows() -> Result<()> {
        let spec = crate::model_spec::ModelSpec::qwen(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )))?,
            &serde_json::json!({}),
        )?;
        for capacity in [256, 16_832] {
            let required =
                FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(5, capacity)?)?;
            let allocated =
                Runtime::workspace(&geometry(), 5, capacity, GdnPolicy::for_device(89, 128))?;
            ensure!(
                required.bytes <= allocated as u64,
                "nested verify workspace {required:?} exceeds allocation {allocated}"
            );
            let mtp = crate::mtp::AttachmentBudget::new(&spec, capacity)?;
            let row_bytes = Runtime::row_bytes(&spec)? + mtp.row_bytes;
            let fixed = Runtime::fixed_bytes(&spec, capacity)? + mtp.fixed_bytes;
            let workspace =
                |rows| {
                    Ok(Runtime::workspace(
                        &geometry(),
                        rows,
                        capacity,
                        GdnPolicy::for_device(89, 128),
                    )? + mtp.workspace(rows)?)
                };
            let total = 24usize << 30;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(5),
                free_after_resident: total / 16 + fixed + 5 * row_bytes + workspace(5)?,
                total,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(request, workspace, Ok)?;
            assert_eq!(budget.plan.chunk, 5);
            assert_eq!(budget.plan.row_limit, 5);
            assert_eq!(
                budget.planned_transient_bytes + budget.plan.reserve_bytes,
                request.free_after_resident
            );
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    workspace,
                    Ok,
                )
                .is_err()
            );
            let admitted = WorkspaceBudget::new(
                WorkspaceRequest {
                    requested: None,
                    free_after_resident: total / 4,
                    ..request
                },
                workspace,
                |rows| Ok(default_chunk(rows)),
            )?;
            assert!(admitted.plan.chunk >= 5);
            assert!(admitted.planned_transient_bytes + admitted.plan.reserve_bytes <= total / 4);
            println!(
                "MTP_M5_ADMISSION_PASS capacity={capacity} total={total} free_after_resident={} minimum_free={} chunk={} row_limit={} planned_transient={} reserve={}",
                total / 4,
                request.free_after_resident,
                admitted.plan.chunk,
                admitted.plan.row_limit,
                admitted.planned_transient_bytes,
                admitted.plan.reserve_bytes,
            );
        }
        Ok(())
    }

    #[test]
    fn fa2_workspace_budget_includes_lse_and_decode_scratch() -> Result<()> {
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let spec = crate::model_spec::ModelSpec::qwen(&root, &serde_json::json!({}))?;
        let work_row_bytes = crate::work_layout::RowWorkSpec::new(&spec)?.row_bytes()?;
        let capacity = 256;
        let total = 1 << 30;
        let fixed = spec.vocab as usize * 2 + 12 + 64 * size_of::<u64>() + capacity * 12;
        for rows in [1, 65, 128] {
            let scratch = Runtime::workspace(&geometry(), rows, capacity, GdnPolicy::LEGACY)?;
            for required in [
                FullAttentionOps::preprocess_workspace(
                    &FullAttentionSpec::q38(),
                    limits(rows, capacity)?,
                )?,
                FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?,
                GdnOps::workspace(
                    &crate::op_contract::GdnSpec::q38(),
                    rows as u32,
                    GdnPolicy::LEGACY,
                )?,
                GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), 1, GdnPolicy::LEGACY)?,
            ] {
                ensure!(
                    scratch as u64 >= required.bytes,
                    "operator scratch underbudgeted"
                );
            }
            let lse = rows * 24 * 4;
            let allocated = rows * work_row_bytes + lse + rows * 12 + fixed + scratch;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(rows),
                free_after_resident: total / 16 + allocated,
                total,
                alignment: 32,
                row_bytes: Runtime::row_bytes(&spec)?,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(
                request,
                |r| Runtime::workspace(&geometry(), r, capacity, GdnPolicy::LEGACY),
                Ok,
            )?;
            assert_eq!(budget.planned_transient_bytes, allocated);
            assert_eq!(budget.plan.row_limit, rows);
            // The same request must fail if even one byte of the allocation is missing.
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    |r| Runtime::workspace(&geometry(), r, capacity, GdnPolicy::LEGACY),
                    Ok,
                )
                .is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn scan_budget_accounts_for_every_prebound_bucket() -> Result<()> {
        let spec = crate::model_spec::ModelSpec::qwen(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )))?,
            &serde_json::json!({}),
        )?;
        let policy = GdnPolicy::for_device(89, 128);
        let capacity = 8192;
        let total = 8usize << 30;
        let row_bytes = Runtime::row_bytes(&spec)?;
        let fixed = Runtime::fixed_bytes(&spec, capacity)?;
        for rows in [512, 513, 1021, 1024, 2048, 4096, 8192] {
            let scratch = Runtime::workspace(&geometry(), rows, capacity, policy)?;
            let gdn = policy.workspace(&crate::op_contract::GdnSpec::q38(), rows as u32)?;
            assert!(scratch as u64 >= gdn.bytes);
            let required = rows * row_bytes + fixed + scratch;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(rows),
                free_after_resident: total / 16 + required,
                total,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(
                request,
                |r| Runtime::workspace(&geometry(), r, capacity, policy),
                Ok,
            )?;
            assert_eq!(budget.plan.chunk, rows);
            assert_eq!(budget.operator_workspace_bytes, scratch);
            assert_eq!(budget.planned_transient_bytes, required);
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    |r| Runtime::workspace(&geometry(), r, capacity, policy),
                    Ok,
                )
                .is_err()
            );
        }
        Ok(())
    }
}
