//! Qwen bindings and operators of the shared Engine.
use super::*;
use crate::backend::cuda::{
    attention_d256::{AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings},
    contracts::{RowDomain, Scratch, Tensor, TensorMut},
    gdn::{GdnBindings, GdnOps, GdnPlan},
    linear::{LayerLinearPlan, LinearWeightRef},
    ops::{CommonOpsRef, GatedRmsInput, RmsMode, RmsOptions},
    weights::WeightCatalog,
};
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
    multiply: CudaFunction,
    advance: CudaFunction,
    prepared: BTreeSet<u32>,
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
        text_prefill: Box<FullAttentionPlan>,
        decode: Box<FullAttentionPlan>,
    },
    Gdn {
        norm: Tensor<bf16, 1>,
        prefill: Box<GdnPlan>,
        decode: Box<GdnPlan>,
    },
}
fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
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
fn matrix(
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
fn heads(
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
fn vector(b: &CudaSlice<bf16>, s: &Arc<CudaStream>) -> Result<Tensor<bf16, 1>> {
    view(b, s, 0, [u32::try_from(b.len())?], [1])
}
fn scratch(b: &CudaSlice<u8>, s: &Arc<CudaStream>) -> Scratch {
    let (ptr, _read) = b.device_ptr(s);
    Scratch {
        ptr,
        bytes: b.len() as u64,
    }
}
fn limits(rows: usize, capacity: usize) -> Result<AttentionLimits> {
    ensure!(capacity >= 32, "Q38 attention capacity must be at least 32");
    Ok(AttentionLimits {
        max_rows: u32::try_from(rows)?,
        capacity: u32::try_from(capacity)?,
        key_chunk: 256.min((capacity / 32 * 32) as u32),
        causal: 1,
    })
}
impl Runtime {
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
    fn workspace(rows: usize, capacity: usize) -> Result<usize> {
        let attention_prefill = FullAttentionOps::preprocess_workspace(limits(rows, capacity)?)?;
        let attention_decode =
            FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?;
        let spec = crate::op_contract::GdnSpec::q38();
        let prefill = GdnOps::workspace(&spec, u32::try_from(rows)?)?;
        let decode = GdnOps::workspace(&spec, 1)?;
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
                        text_prefill: Box::new(bind(rows, PositionBindings::Linear)?),
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
                        )
                    };
                    BoundMixer::Gdn {
                        norm: weight("linear_attn.norm.weight")?,
                        prefill: Box::new(bind(rows)?),
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
            multiply: module.load_function("q38_multiply")?,
            advance: module.load_function("q38_advance")?,
            prepared: BTreeSet::new(),
        })
    }
    fn prepare(
        &mut self,
        blas: &Blas,
        rows: u32,
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<()> {
        if self.prepared.contains(&rows) {
            return Ok(());
        }
        for layer in &self.layers {
            layer.linear.prepare(blas, rows)?;
        }
        blas.prepare_fixed(1, spec.vocab, spec.hidden)?;
        self.prepared.insert(rows);
        Ok(())
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
            ensure!(
                free.checked_sub(total / 16).is_some_and(|n| needed <= n),
                "vision weights and request exceed memory budget: required={needed} free={free}"
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

    pub(super) fn load_qwen(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        write: bool,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
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
        Runtime::workspace(1, capacity)?;
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
        let mut catalog = WeightCatalog::load(dir, &model_spec, &device)?;
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
        let budget = crate::prefill::WorkspaceBudget::new(
            crate::prefill::WorkspaceRequest {
                capacity,
                requested: requested_chunk,
                free_after_resident: free,
                total,
                alignment: 32,
                row_bytes: Runtime::row_bytes(&model_spec)?,
                fixed_transient_bytes: Runtime::fixed_bytes(&model_spec, capacity)?,
            },
            |rows| Runtime::workspace(rows, capacity),
        )?;
        let rows = budget.plan.chunk;
        eprintln!(
            "{}",
            serde_json::json!({
                "q38_memory_plan": budget,
                "resident_weights_bytes": weights.bytes,
                "persistent_state_bytes": state.bytes(),
                "capture_checkpoint_bytes": capture_checkpoint.as_ref().context("capture checkpoint missing")?.bytes(),
                "layer_arena_bytes": arena_bytes,
                "attention_prefill_preprocess_bytes": FullAttentionOps::preprocess_workspace(limits(rows, capacity)?)?.bytes,
                "attention_prefill_lse_bytes": rows * model_spec.q_heads as usize * size_of::<f32>(),
                "attention_decode_workspace_bytes": FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?.bytes,
                "attention_prefill_backend": "FA2 d256",
                "attention_decode_backend": "TCMQA d256",
                "gdn_prefill_workspace_bytes": GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), u32::try_from(rows)?)?.bytes,
                "gdn_decode_workspace_bytes": GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), 1)?.bytes,
                "bound_prefill_max_rows": rows,
                "bound_decode_max_rows": 1,
                "gdn_plan": crate::op_contract::GdnSpec::PLAN_ID,
                "selection_source": "fixed"
            })
        );
        let scratch_buffer = s.alloc_zeros::<u8>(budget.operator_workspace_bytes)?;
        let work = Work::new(s, &model_spec, rows)?;
        let flash = Flash::new(s.clone(), rows, model_spec.q_heads as usize)?;
        let config = model_spec.config();
        let mut engine = Self {
            #[cfg(test)]
            capture_fail_after_layer: None,
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
            qwen: None,
            images: None,
        };
        engine.qwen = Some(Runtime::new(&engine, dir, catalog, arena, scratch_buffer)?);
        engine.prepare(1)?;
        engine.prepare(rows)?;
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
                        matrix(&w.n, s, 0, rows, 5120)?,
                        &mut TensorMut(matrix(&w.qkv, s, offset, rows, width)?),
                    )?;
                }
                match phase {
                    ExecutionPhase::Prefill => {
                        let prefill = if self.images.is_some() {
                            prefill
                        } else {
                            text_prefill
                        };
                        FullAttentionOps::preprocess(prefill, s, meta, scratch(&q.scratch, s))?;
                        let kv = &self.state.kv[self.model_spec.layers[i].state_index as usize];
                        self.flash
                            .as_mut()
                            .context("Q38 prefill attention is not bound")?
                            .run(
                                &mw.q,
                                24 * 256,
                                &kv.k,
                                &kv.v,
                                &mut w.attn,
                                Shape {
                                    rows,
                                    start: self.position,
                                    q_heads: 24,
                                    kv_heads: 4,
                                    capacity: self.capacity,
                                    head_dim: 256,
                                    causal: true,
                                },
                            )?;
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
                    matrix(&w.attn, s, 0, rows, 6144)?,
                    &mut TensorMut(matrix(&w.out, s, 0, rows, 5120)?),
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
                        matrix(&w.n, s, 0, rows, 5120)?,
                        &mut TensorMut(matrix(output, s, offset, rows, width)?),
                    )?;
                }
                let plan = match phase {
                    ExecutionPhase::Prefill => prefill,
                    ExecutionPhase::Decode => decode,
                };
                GdnOps::run(plan, s, meta, phase, scratch(&q.scratch, s))?;
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
                    matrix(&w.attn, s, 0, rows, value)?,
                    &mut TensorMut(matrix(&w.out, s, 0, rows, 5120)?),
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
                unsafe {
                    s.launch_builder(&self.ops.residual)
                        .arg(&mut w.x)
                        .arg(&w.out)
                        .arg(&((rows * h) as i32))
                        .launch(flat(rows * h))?;
                }
                if matches!(op, O::PostNorm) {
                    q.common.norm.rms_with(
                        s,
                        domain,
                        heads(&w.x, s, 0, rows, 1, h as u32)?,
                        layer.post_norm,
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
                } else if trace.is_some()
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
            O::GateUp => {
                for (role, offset) in [
                    ("mlp.gate_proj.weight", 0),
                    ("mlp.up_proj.weight", self.workspace_rows * f),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, rows, h)?,
                        &mut TensorMut(matrix(&w.gu, s, offset, rows, f)?),
                    )?;
                }
            }
            O::Activate => {
                q.common.gate.silu(
                    s,
                    domain,
                    heads(&w.gu, s, 0, rows, 1, f as u32)?,
                    &mut TensorMut(heads(&w.act, s, 0, rows, 1, f as u32)?),
                )?;
                let a = matrix(&w.act, s, 0, rows, f)?;
                let up = matrix(&w.gu, s, self.workspace_rows * f, rows, f)?;
                unsafe {
                    s.launch_builder(&q.multiply)
                        .arg(&a.ptr)
                        .arg(&up.ptr)
                        .arg(&a.ptr)
                        .arg(&((rows * f) as u64))
                        .launch(flat(rows * f))?;
                }
            }
            O::Down => layer.linear.run(
                &self.blas,
                "mlp.down_proj.weight",
                phase,
                matrix(&w.act, s, 0, rows, f)?,
                &mut TensorMut(matrix(&w.out, s, 0, rows, h)?),
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
    use super::*;
    use crate::prefill::{WorkspaceBudget, WorkspaceRequest};

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
            let scratch = Runtime::workspace(rows, capacity)?;
            for required in [
                FullAttentionOps::preprocess_workspace(limits(rows, capacity)?)?,
                FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?,
                GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), rows as u32)?,
                GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), 1)?,
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
            let budget = WorkspaceBudget::new(request, |r| Runtime::workspace(r, capacity))?;
            assert_eq!(budget.planned_transient_bytes, allocated);
            assert_eq!(budget.plan.row_limit, rows);
            // The same request must fail if even one byte of the allocation is missing.
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    |r| Runtime::workspace(r, capacity),
                )
                .is_err()
            );
        }
        Ok(())
    }
}
