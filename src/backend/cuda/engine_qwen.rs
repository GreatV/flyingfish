//! Qwen bindings and operators of the shared Engine.
use super::*;
use crate::backend::cuda::{
    attention_d256::{AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings},
    contracts::{RowDomain, Scratch, Tensor, TensorMut},
    gdn::{GdnBindings, GdnOps, GdnPlan},
    linear::{LayerLinearPlan, LinearWeightRef},
    ops::{CommonOpsRef, GatedRmsInput, ResidualRmsInput, RmsMode, RmsOptions},
    weights::WeightCatalog,
};
/// Default prefill chunk ceiling in rows; an explicit `--chunk` is not clamped.
const DEFAULT_CHUNK_ROWS: usize = 2048;

use crate::gdn_plan::GdnPolicy;
use crate::{
    linear_plan::LayerDequantPlan,
    model_spec::Mixer,
    op_contract::{AttentionLimits, ExecutionPhase, FullAttentionSpec},
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

pub(super) struct Runtime {
    checkpoint_dir: PathBuf,
    vision: Option<super::vision_runtime::VisionRuntime>,
    vision_bytes: usize,
    positions: CudaSlice<i32>,
    active_positions: CudaSlice<i32>,
    gather_positions: CudaFunction,
    catalog: WeightCatalog,
    arena: CudaSlice<u8>,
    scratch: CudaSlice<u8>,
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
        prefill: Box<FullAttentionPlan>,
        text_prefill: PrefillPlans<FullAttentionPlan>,
        decode: Box<FullAttentionPlan>,
    },
    Gdn {
        norm: Tensor<bf16, 1>,
        prefill: PrefillPlans<GdnPlan>,
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
/// Either a snapshot-bound plan borrowed as-is or a plan narrowed at dispatch.
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
        let small = (1..=4.min(max_rows))
            .map(|rows| narrow(&full, rows as u32))
            .collect::<Result<_>>()?;
        Ok(Self {
            full: Box::new(full),
            small,
            max_rows,
        })
    }
    /// The plan narrowed to this step's live rows; 1..=4 keep the snapshot-bound small plans and larger sizes narrow host-side at dispatch.
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
pub(super) fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    offset: usize,
    shape: [u32; R],
    stride: [u64; R],
) -> Result<Tensor<T, R>> {
    ensure!(offset < b.len(), "view offset outside allocation");
    let elements = shape
        .iter()
        .zip(&stride)
        .try_fold(1u64, |extent, (&n, &step)| {
            ensure!(n > 0, "zero view dimension");
            extent
                .checked_add(
                    u64::from(n - 1)
                        .checked_mul(step)
                        .context("view stride overflow")?,
                )
                .context("view extent overflow")
        })?;
    ensure!(
        elements <= (b.len() - offset) as u64,
        "view exceeds allocation"
    );
    let bytes = elements
        .checked_mul(size_of::<T>() as u64)
        .context("view byte overflow")?;
    let (ptr, _read) = b.device_ptr(s);
    unsafe { Tensor::new(ptr + (offset * size_of::<T>()) as u64, shape, stride, bytes) }
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
    fn workspace(rows: usize, capacity: usize, policy: GdnPolicy) -> Result<usize> {
        let attention_prefill = FullAttentionOps::preprocess_workspace(limits(rows, capacity)?)?;
        let attention_decode =
            FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?;
        let spec = crate::op_contract::GdnSpec::q38();
        let prefill = GdnOps::workspace(&spec, u32::try_from(rows)?, policy)?;
        let decode = GdnOps::workspace(&spec, 1, GdnPolicy::LEGACY)?;
        ensure!(
            [
                attention_prefill.alignment,
                attention_decode.alignment,
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
                .max(prefill.bytes)
                .max(decode.bytes),
        )?)
    }
    fn new(
        engine: &Engine,
        checkpoint_dir: &Path,
        catalog: WeightCatalog,
        arena: CudaSlice<u8>,
        scratch_buffer: CudaSlice<u8>,
    ) -> Result<Self> {
        let s = &engine.device.stream;
        let rows = engine.workspace_rows;
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
            let plan = LayerDequantPlan::from_manifest(&catalog.manifest, i as u32)?;
            let linear = LayerLinearPlan::bind(&plan, &catalog, &scratch(&arena, s), s)?;
            let mixer = match layout.mixer {
                Mixer::FullAttention => {
                    let kv = &engine.state.kv[layout.state_index as usize];
                    let bind = |active_rows, positions| {
                        FullAttentionOps::prepare(
                            &engine.device.ctx,
                            &FullAttentionSpec::q38(),
                            limits(active_rows, engine.capacity)?,
                            AttentionBindings {
                                q_gate: heads(&w.qkv, s, 0, active_rows, 24, 512)?,
                                k: heads(&w.qkv, s, rows * 12288, active_rows, 4, 256)?,
                                v: heads(&w.qkv, s, rows * (12288 + 1024), active_rows, 4, 256)?,
                                q: TensorMut(heads(&mw.q, s, 0, active_rows, 24, 256)?),
                                gate: TensorMut(heads(&mw.gate, s, 0, active_rows, 24, 256)?),
                                k_cache: TensorMut(view(
                                    &kv.k,
                                    s,
                                    0,
                                    [4, engine.capacity as u32, 256],
                                    [(engine.capacity * 256) as u64, 256, 1],
                                )?),
                                v_cache: TensorMut(view(
                                    &kv.v,
                                    s,
                                    0,
                                    [4, engine.capacity as u32, 256],
                                    [(engine.capacity * 256) as u64, 256, 1],
                                )?),
                                core_out: TensorMut(heads(&w.attn, s, 0, active_rows, 24, 256)?),
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
                    BoundMixer::Attention {
                        prefill: Box::new(bind(
                            rows,
                            PositionBindings::Axes(view(
                                &active_positions,
                                s,
                                0,
                                [rows as u32, 3],
                                [3, 1],
                            )?),
                        )?),
                        text_prefill: PrefillPlans::bind(
                            bind(rows, PositionBindings::Linear)?,
                            rows,
                            FullAttentionPlan::for_rows,
                        )?,
                        decode: Box::new(decode),
                    }
                }
                Mixer::Gdn => {
                    let spec = crate::op_contract::GdnSpec::q38();
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
                            // The planner's derived chunkwise-P1 split for a plan bound
                            // at `active_rows`; steps of at most MIN_VERIFY_ROWS keep 0.
                            gdn_policy.p1_split_for_step(active_rows as u32),
                        )
                    };
                    BoundMixer::Gdn {
                        norm: weight("linear_attn.norm.weight")?,
                        prefill: PrefillPlans::bind(bind(rows)?, rows, GdnPlan::for_rows)?,
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
            self.catalog.bytes > 0 && !self.arena.is_empty() && self.mask.len() == 64,
            "Q38 owners missing"
        );
        Ok(())
    }
}
impl Engine {
    pub(super) fn bind_mtp_snapshots(
        &mut self,
        snapshots: &[super::super::state::GdnState],
    ) -> Result<()> {
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let stream = &self.device.stream;
        let mut states = snapshots.iter();
        for layer in &mut q.layers {
            if let BoundMixer::Gdn { prefill, .. } = &mut layer.mixer {
                let state = states.next().context("missing GDN snapshot allocation")?;
                let s = view(
                    &state.s,
                    stream,
                    0,
                    [4, 48, 128, 128],
                    [48 * 128 * 128, 128 * 128, 128, 1],
                )?;
                let h = view(&state.history, stream, 0, [4, 10240, 3], [10240 * 3, 3, 1])?;
                for plan in &mut prefill.small {
                    plan.bind_snapshots(s, h)?;
                }
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
        let main_bytes = checkpoint.q38_manifest(&spec)?.resident_bytes;
        let vision_bytes = checkpoint
            .q38_manifest_with_policy(&spec, crate::checkpoint::LoadPolicy::VisionOnly)?
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
                .q38_manifest_with_policy(
                    &self.model_spec,
                    crate::checkpoint::LoadPolicy::VisionOnly,
                )?;
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
        let write = !matches!(mode, super::SetupMode::Ready);
        let reserve_mtp = matches!(mode, super::SetupMode::Mtp);
        let started = Instant::now();
        ensure!(
            settings.draft_model.is_none(),
            "Q38 MTP/tree execution is not bound"
        );
        ensure!(
            capacity >= 32 && capacity <= model_spec.max_position as usize,
            "invalid Q38 capacity"
        );
        // Reject an incomplete operator plan before allocating model weights.
        Runtime::workspace(1, capacity, GdnPolicy::LEGACY)?;
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
        let prefix_identity = if prefix_cache {
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
        let weights = catalog.take_globals()?;
        let s = &device.stream;
        let state = super::super::state::StateStore::new(&model_spec, capacity, s)?;
        let capture_checkpoint = Some(super::super::state::StateCheckpoint::new(
            &state,
            &model_spec,
            1,
            s,
        )?);
        let arena_bytes = (0..model_spec.layers.len())
            .map(|i| LayerDequantPlan::from_manifest(&catalog.manifest, i as u32).map(|p| p.bytes))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing Q38 layers")?;
        let arena = s.alloc_zeros::<u8>(usize::try_from(arena_bytes)?)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        blas.enable_fp8(&device.ctx)?;
        device.finish_upload()?;
        let (free, total) = device.ctx.mem_get_info()?;
        let mtp_budget = reserve_mtp
            .then(|| crate::mtp::AttachmentBudget::new(&model_spec, capacity))
            .transpose()?;
        let mut admitted_default_chunk = None;
        let budget = crate::prefill::WorkspaceBudget::new(
            crate::prefill::WorkspaceRequest {
                capacity,
                requested: requested_chunk,
                free_after_resident: free,
                total,
                alignment: 32,
                row_bytes: Runtime::row_bytes(&model_spec)?
                    .checked_add(mtp_budget.as_ref().map_or(0, |b| b.row_bytes))
                    .context("combined row bytes overflow")?,
                fixed_transient_bytes: Runtime::fixed_bytes(&model_spec, capacity)?
                    .checked_add(mtp_budget.as_ref().map_or(0, |b| b.fixed_bytes))
                    .context("combined fixed bytes overflow")?,
            },
            |rows| {
                let target = Runtime::workspace(rows, capacity, gdn_policy)?;
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
            |admitted| {
                admitted_default_chunk = Some(admitted);
                Ok(admitted.min(DEFAULT_CHUNK_ROWS))
            },
        )?;
        let rows = budget.plan.chunk;
        if let Some(admitted) = admitted_default_chunk
            && admitted != rows
        {
            eprintln!(
                "{}",
                serde_json::json!({"q38_default_chunk_policy": {
                    "admitted_chunk": admitted, "selected_chunk": rows,
                    "arch": device.info.arch, "sms": device.info.sms,
                    "requested_chunk": requested_chunk,
                }})
            );
        }
        ensure!(
            !reserve_mtp || rows >= 4,
            "MTP workspace cannot fit four verification rows"
        );
        let target_workspace = Runtime::workspace(rows, capacity, gdn_policy)?;
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
                "capture_checkpoint_bytes": capture_checkpoint.as_ref().context("capture checkpoint missing")?.bytes(),
                "layer_arena_bytes": arena_bytes,
                "attention_prefill_preprocess_bytes": FullAttentionOps::preprocess_workspace(limits(rows, capacity)?)?.bytes,
                "attention_prefill_lse_bytes": rows * model_spec.q_heads as usize * size_of::<f32>(),
                "attention_decode_workspace_bytes": FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?.bytes,
                "attention_prefill_backend": "FA2 d256",
                "attention_decode_backend": "TCMQA d256",
                "gdn_prefill_workspace_bytes": GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), u32::try_from(rows)?, gdn_policy)?.bytes,
                "gdn_decode_workspace_bytes": GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), 1, GdnPolicy::LEGACY)?.bytes,
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
        let config = model_spec.config();
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
        engine.qwen = Some(Runtime::new(&engine, dir, catalog, arena, scratch_buffer)?);
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
        self.work.advance(&self.ops, &self.device.stream)
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
                .materialize(self.blas.fp8_ops()?, &self.device.stream)?;
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
                prefill,
                text_prefill,
                decode,
            } => {
                for (role, width, offset) in [
                    ("self_attn.q_proj.weight", 12288, 0),
                    ("self_attn.k_proj.weight", 1024, self.workspace_rows * 12288),
                    (
                        "self_attn.v_proj.weight",
                        1024,
                        self.workspace_rows * (12288 + 1024),
                    ),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, 5120)?,
                        &mut TensorMut(matrix(&w.qkv, s, offset, linear_rows, width)?),
                    )?;
                }
                match phase {
                    ExecutionPhase::Prefill => {
                        let narrowed;
                        let prefill: &FullAttentionPlan = if self.images.is_some() {
                            prefill
                        } else {
                            narrowed = text_prefill.select(rows, &FullAttentionPlan::for_rows)?;
                            &narrowed
                        };
                        FullAttentionOps::preprocess(prefill, s, meta, scratch(&q.scratch, s))?;
                        let kv = &self.state.kv[self.model_spec.layers[i].state_index as usize];
                        let shape = Shape {
                            rows,
                            start: self.position,
                            q_heads: 24,
                            kv_heads: 4,
                            capacity: self.capacity,
                            head_dim: 256,
                            causal: true,
                        };
                        let launch = if self.qwen_device_prefix {
                            super::super::flash::Launch::Device {
                                shape,
                                key_count: self
                                    .state
                                    .sequence
                                    .as_ref()
                                    .context("Q38 sequence missing")?
                                    .slice(2..3),
                            }
                        } else {
                            shape.into()
                        };
                        self.flash
                            .as_mut()
                            .context("Q38 prefill attention is not bound")?
                            .run(&mw.q, 24 * 256, &kv.k, &kv.v, &mut w.attn, launch)?;
                    }
                    ExecutionPhase::Decode => {
                        FullAttentionOps::run(decode, s, meta, scratch(&q.scratch, s))?;
                    }
                }
                q.common.gate.sigmoid_mul(
                    s,
                    domain,
                    heads(&w.attn, s, 0, rows, 24, 256)?,
                    heads(&mw.gate, s, 0, rows, 24, 256)?,
                    &mut TensorMut(heads(&w.attn, s, 0, rows, 24, 256)?),
                )?;
                layer.linear.run(
                    &self.blas,
                    "self_attn.o_proj.weight",
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, 6144)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, 5120)?),
                )?;
            }
            BoundMixer::Gdn {
                norm,
                prefill,
                decode,
            } => {
                let spec = crate::op_contract::GdnSpec::q38();
                let conv = spec.conv_dim()? as usize;
                let value = (spec.hv * spec.dv) as usize;
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
                        matrix(&w.n, s, 0, linear_rows, 5120)?,
                        &mut TensorMut(matrix(output, s, offset, linear_rows, width)?),
                    )?;
                }
                let narrowed;
                let plan = match phase {
                    ExecutionPhase::Prefill => {
                        narrowed = prefill.select(rows, &GdnPlan::for_rows)?;
                        &narrowed
                    }
                    ExecutionPhase::Decode => decode.as_ref(),
                };
                if self.qwen_device_prefix {
                    ensure!(
                        phase == ExecutionPhase::Prefill,
                        "MTP verify must remain Prefill"
                    );
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
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, 5120)?),
                )?;
            }
        }
        let all = all_layers(trace);
        if all || i == self.trace_layer {
            let name = debug_name(trace, i, "mixer_out");
            trace_value(trace, s, &name, &w.attn, rows, 6144)?;
            let name = debug_name(trace, i, "o_proj");
            trace_value(trace, s, &name, &w.out, rows, 5120)?;
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
                    vec![48, 128, 128],
                    s.clone_dtoh(&state.s)?,
                )?;
                t.trace.add(
                    format!("{prefix}.debug.gdn_history{suffix}"),
                    vec![10240, 3],
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
                    unsafe {
                        s.launch_builder(&self.ops.residual)
                            .arg(&mut w.x)
                            .arg(&w.out)
                            .arg(&((rows * h) as i32))
                            .launch(flat(rows * h))?;
                    }
                    if trace.is_some()
                        && (!all_layers(trace) || [0, 3, 7, 31, 32, 47, 63].contains(&i))
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
    /// A stand-in plan recording the rows it was narrowed to, so selection is testable without a device.
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
        // 1..=4 keep the snapshot-bound small plans.
        for rows in 1..=4 {
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
            assert_eq!(plans.small, (1..=4.min(capacity)).collect::<Vec<_>>());
            for rows in 1..=capacity {
                assert_eq!(*plans.select(rows, &narrow)?, rows);
            }
            assert!(plans.select(0, &narrow).is_err());
            assert!(plans.select(capacity + 1, &narrow).is_err());
            assert!(plans.select(usize::MAX, &narrow).is_err());
            let budget = Runtime::workspace(capacity, 4096, GdnPolicy::for_device(89, 128))? as u64;
            for rows in 1..=4.min(capacity) {
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
                    FullAttentionOps::preprocess_workspace(limits(rows, 4096)?)?.bytes <= budget
                );
            }
        }
        assert!(PrefillPlans::bind(0, 0, |_, rows| Ok(rows as usize)).is_err());
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
            let scratch = Runtime::workspace(rows, capacity, GdnPolicy::LEGACY)?;
            for required in [
                FullAttentionOps::preprocess_workspace(limits(rows, capacity)?)?,
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
                |r| Runtime::workspace(r, capacity, GdnPolicy::LEGACY),
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
                    |r| Runtime::workspace(r, capacity, GdnPolicy::LEGACY),
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
            let scratch = Runtime::workspace(rows, capacity, policy)?;
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
            let budget =
                WorkspaceBudget::new(request, |r| Runtime::workspace(r, capacity, policy), Ok)?;
            assert_eq!(budget.plan.chunk, rows);
            assert_eq!(budget.operator_workspace_bytes, scratch);
            assert_eq!(budget.planned_transient_bytes, required);
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    |r| Runtime::workspace(r, capacity, policy),
                    Ok,
                )
                .is_err()
            );
        }
        Ok(())
    }
}
