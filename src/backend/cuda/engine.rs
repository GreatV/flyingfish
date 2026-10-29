use super::copy;
use super::flash::Flash;
use crate::backend::TreeBuilder;
use crate::backend::gpu::{CudaContext, PinnedHostSlice};
use crate::backend::gpu::{CudaGraph, CudaSlice, CudaStream, DevicePtrMut, PushKernelArg};
#[cfg(feature = "cuda")]
use crate::prefix_state::{
    PrefixBoundary, PrefixBuilder, PrefixGeometry, PrefixIdentity, PrefixRecordKind,
    PrefixRestoreReceipt, PrefixSaveReceipt, PrefixSnapshot, sha256, token_ids_sha256,
    validate_transfer_bytes,
};
use crate::prefix_state::{PrefixLastInputKind, PrefixLogitsCache};
#[cfg(feature = "cuda")]
use crate::vision_input::PositionPlan;
use crate::{
    backend::cuda::weights::Weights,
    backend::cuda::{
        Device,
        blas::Blas,
        decode::Attention,
        ops::{Ops, flat, grid},
        verification::Shape,
    },
    backend::{Event, Head, Logits, Settings, SpecBudget, Target},
    config::Config,
    trace::Trace,
};
use anyhow::{Context, Result, ensure};
#[cfg(feature = "cuda")]
use cudarc::driver::sys;
use half::bf16;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::Instant,
};

#[cfg(feature = "cuda")]
#[path = "engine_deepseek.rs"]
pub(crate) mod deepseek;
#[path = "engine_moe.rs"]
mod moe;
#[path = "engine_mtp.rs"]
#[cfg(feature = "cuda")]
mod mtp;
#[path = "engine_mtp_generate.rs"]
#[cfg(feature = "cuda")]
mod mtp_generate;
#[cfg(test)]
#[path = "engine_mtp_launch_tests.rs"]
#[cfg(feature = "cuda")]
mod mtp_launch_tests;
#[path = "engine_qwen.rs"]
mod qwen;
#[path = "engine_recurrent.rs"]
mod recurrent;
#[path = "engine_vision.rs"]
mod vision_runtime;

fn trace_value(
    trace: &mut Option<Target<'_>>,
    stream: &Arc<CudaStream>,
    name: &str,
    data: &CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    if let Some(t) = trace.as_mut() {
        ensure!(t.skip < rows, "trace selection is empty");
        let data = stream.clone_dtoh(&data.slice(t.skip * width..rows * width))?;
        t.trace.add(
            format!("{}.{name}", t.prefix),
            vec![1, rows - t.skip, width],
            data.into_iter().map(bf16::to_f32).collect(),
        )?;
    }
    Ok(())
}

pub(super) fn bucketed_prefill_rows(rows: u32, workspace_rows: u32) -> Result<u32> {
    ensure!(
        rows > 0 && rows <= workspace_rows,
        "prefill rows {rows} outside workspace {workspace_rows}"
    );
    if rows <= 16 {
        return Ok(rows);
    }
    Ok(rows.next_power_of_two().min(workspace_rows))
}

pub(super) fn spec_workspace_rows(
    chunk: usize,
    budget: Option<SpecBudget>,
    verify_rows: usize,
) -> usize {
    budget
        .map_or(chunk, |budget| budget.workspace_rows(chunk))
        .max(verify_rows)
}

pub(crate) use super::state::Kv;

pub(crate) struct Work {
    pub(crate) ids: CudaSlice<u32>,
    pub(crate) position: CudaSlice<i32>,
    pub(crate) length: CudaSlice<i32>,
    pub(crate) x: CudaSlice<bf16>,
    pub(crate) n: CudaSlice<bf16>,
    pub(crate) qkv: CudaSlice<bf16>,
    pub(crate) attn: CudaSlice<bf16>,
    pub(crate) out: CudaSlice<bf16>,
    pub(crate) gu: CudaSlice<bf16>,
    pub(crate) act: CudaSlice<bf16>,
    pub(crate) logits: CudaSlice<bf16>,
    pub(crate) token: CudaSlice<u32>,
    pub(crate) mixer: Option<MixerWork>,
    #[cfg(feature = "cuda")]
    pub(super) peer_tail_elems: usize,
}

pub(crate) struct MixerWork {
    pub(crate) q: CudaSlice<bf16>,
    pub(crate) gate: CudaSlice<bf16>,
    pub(crate) ab: CudaSlice<bf16>,
}
impl MixerWork {
    fn new(
        stream: &Arc<CudaStream>,
        shape: &crate::work_layout::RowWorkSpec,
        rows: usize,
    ) -> Result<Option<Self>> {
        use crate::work_layout::RowWorkSpec as W;
        if !shape.extra_mixer {
            return Ok(None);
        }
        Ok(Some(Self {
            q: stream.alloc_zeros(W::elements(rows, shape.mixer)?)?,
            gate: stream.alloc_zeros(W::elements(rows, shape.mixer)?)?,
            ab: stream.alloc_zeros(W::elements(rows, shape.ab)?)?,
        }))
    }
}

impl Work {
    fn new(
        stream: &Arc<CudaStream>,
        spec: &crate::model_spec::ModelSpec,
        rows: usize,
    ) -> Result<Self> {
        Self::new_with_peer_tail(stream, spec, rows, 0)
    }

    pub(super) fn new_with_peer_tail(
        stream: &Arc<CudaStream>,
        spec: &crate::model_spec::ModelSpec,
        rows: usize,
        peer_tail_elems: usize,
    ) -> Result<Self> {
        use crate::work_layout::RowWorkSpec as W;
        let shape = W::new(spec)?;
        let n_elems = W::elements(rows, shape.hidden)?
            .checked_add(peer_tail_elems)
            .context("peer send workspace overflow")?;
        Ok(Self {
            ids: stream.alloc_zeros(rows)?,
            position: stream.alloc_zeros(1)?,
            length: stream.alloc_zeros(1)?,
            x: stream.alloc_zeros(W::elements(rows, shape.hidden)?)?,
            n: stream.alloc_zeros(n_elems)?,
            qkv: stream.alloc_zeros(W::elements(rows, shape.projection)?)?,
            attn: stream.alloc_zeros(W::elements(rows, shape.mixer)?)?,
            out: stream.alloc_zeros(W::elements(rows, shape.hidden)?)?,
            gu: stream.alloc_zeros(W::elements(
                rows,
                shape
                    .intermediate
                    .checked_mul(2)
                    .context("MLP width overflow")?,
            )?)?,
            act: stream.alloc_zeros(W::elements(rows, shape.intermediate)?)?,
            logits: stream.alloc_zeros(spec.vocab as usize)?,
            token: stream.alloc_zeros(1)?,
            mixer: MixerWork::new(stream, &shape, rows)?,
            #[cfg(feature = "cuda")]
            peer_tail_elems,
        })
    }
    fn grow(
        &mut self,
        stream: &Arc<CudaStream>,
        spec: &crate::model_spec::ModelSpec,
        rows: usize,
    ) -> Result<()> {
        use crate::work_layout::RowWorkSpec as W;
        ensure!(rows >= self.ids.len(), "workspace rows cannot shrink");
        let shape = W::new(spec)?;
        let ids = stream.alloc_zeros(rows)?;
        let x = stream.alloc_zeros(W::elements(rows, shape.hidden)?)?;
        #[cfg(feature = "cuda")]
        let peer_tail = self.peer_tail_elems;
        #[cfg(feature = "rocm")]
        let peer_tail = 0;
        let n_elems = W::elements(rows, shape.hidden)?
            .checked_add(peer_tail)
            .context("peer send workspace overflow")?;
        let n = stream.alloc_zeros(n_elems)?;
        let qkv = stream.alloc_zeros(W::elements(rows, shape.projection)?)?;
        let attn = stream.alloc_zeros(W::elements(rows, shape.mixer)?)?;
        let out = stream.alloc_zeros(W::elements(rows, shape.hidden)?)?;
        let gu = stream.alloc_zeros(W::elements(
            rows,
            shape
                .intermediate
                .checked_mul(2)
                .context("MLP width overflow")?,
        )?)?;
        let act = stream.alloc_zeros(W::elements(rows, shape.intermediate)?)?;
        let mixer = MixerWork::new(stream, &shape, rows)?;
        self.ids = ids;
        self.x = x;
        self.n = n;
        self.qkv = qkv;
        self.attn = attn;
        self.out = out;
        self.gu = gu;
        self.act = act;
        self.mixer = mixer;
        Ok(())
    }

    fn add_norm(
        &mut self,
        ops: &Ops,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<bf16>,
        rows: usize,
        c: &Config,
    ) -> Result<()> {
        let (res, _res) = self.x.device_ptr_mut(stream);
        unsafe {
            stream
                .launch_builder(&ops.add_norm)
                .arg(&self.out)
                .arg(&res)
                .arg(weight)
                .arg(&mut self.n)
                .arg(&res)
                .arg(&(rows as i32))
                .arg(&(c.hidden_size as i32))
                .arg(&c.rms_norm_eps)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }

    fn advance(&mut self, ops: &Ops, stream: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&ops.advance)
                .arg(&mut self.position)
                .launch(grid(1, 1))?;
            stream
                .launch_builder(&ops.advance)
                .arg(&mut self.length)
                .launch(grid(1, 1))?;
        }
        Ok(())
    }
}

pub(crate) struct Views<'a> {
    pub device: &'a Device,
    pub config: &'a Config,
    pub weights: &'a Weights,
    pub kv: &'a mut [Kv],
    pub work: &'a mut Work,
    pub capacity: usize,
    pub head_stride: usize,
    pub pos_stride: usize,
}

impl Views<'_> {
    fn check(&self, rows: usize) -> Result<()> {
        let c = self.config;
        ensure!(
            self.device.info.warp == 32,
            "CUDA workspace requires warp32"
        );
        ensure!(
            self.head_stride == self.capacity * c.head_dim && self.pos_stride == c.head_dim,
            "KV view strides do not match capacity/head dimension"
        );
        ensure!(
            self.weights.layers.len() == c.num_hidden_layers
                && self.weights.head.len() == c.vocab_size * c.hidden_size,
            "weight view does not match configured layers/head"
        );
        ensure!(
            self.kv.len() == c.num_hidden_layers
                && self
                    .kv
                    .iter()
                    .all(|kv| kv.k.len() == self.capacity * c.kv_dim()
                        && kv.v.len() == self.capacity * c.kv_dim()),
            "KV view does not match configured layers/capacity"
        );
        ensure!(
            rows > 0
                && self.work.ids.len() >= rows
                && self.work.x.len() >= rows * c.hidden_size
                && self.work.n.len() >= rows * c.hidden_size
                && self.work.qkv.len() >= rows * c.qkv_dim()
                && self.work.attn.len() >= rows * c.hidden_size
                && self.work.out.len() >= rows * c.hidden_size
                && self.work.gu.len() >= rows * 2 * c.intermediate_size
                && self.work.act.len() >= rows * c.intermediate_size,
            "workspace views cannot hold {rows} rows"
        );
        Ok(())
    }
}

struct HiddenCapture {
    values: CudaSlice<bf16>,
    layer_slots: Vec<Option<u32>>,
    taps: usize,
    width: usize,
}

impl HiddenCapture {
    fn new(ids: &[usize], config: &Config, rows: usize, stream: &Arc<CudaStream>) -> Result<Self> {
        let mut layer_slots = vec![None; config.num_hidden_layers];
        for (slot, &layer) in ids.iter().enumerate() {
            // Slots store decoder-layer outputs, hidden_states[layer + 1].
            let entry = layer_slots
                .get_mut(layer)
                .context("capture layer outside target")?;
            ensure!(entry.is_none(), "duplicate capture layer {layer}");
            *entry = Some(u32::try_from(slot)?);
        }
        let width = ids
            .len()
            .checked_mul(config.hidden_size)
            .filter(|&width| width > 0 && width <= i32::MAX as usize)
            .context("capture row width overflow")?;
        let elements = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(size_of::<bf16>()))
            .context("capture allocation size overflow")?
            / size_of::<bf16>();
        Ok(Self {
            values: stream.alloc_zeros(elements)?,
            layer_slots,
            taps: ids.len(),
            width,
        })
    }

    fn grow(&mut self, rows: usize, stream: &Arc<CudaStream>) -> Result<()> {
        let elements = rows
            .checked_mul(self.width)
            .and_then(|n| n.checked_mul(size_of::<bf16>()))
            .context("capture allocation size overflow")?
            / size_of::<bf16>();
        self.values = stream.alloc_zeros(elements)?;
        Ok(())
    }
}

struct Spec {
    #[cfg(test)]
    forced_proposals: Option<Vec<u32>>,
    eos: Vec<u32>,
    tree_verify_graph: Option<CudaGraph>,
    tree_commit_graph: Option<CudaGraph>,
    tree_draft_graph: Option<CudaGraph>,
    draft_graph: Option<CudaGraph>,
    verify_graph: Option<CudaGraph>,
    moe_verify_graphs: std::collections::BTreeMap<usize, CudaGraph>,
    inject_graphs: Vec<Option<CudaGraph>>,
    tree: Option<super::tree::State>,
    tree_attention: Option<super::verification::Verification>,
    max_verify_rows: usize,
    non_expert_decode_bytes: u64,
    verify_schedule: crate::dspark::VerifyCostSchedule,
    cold_verify_schedule: crate::dspark::VerifyCostSchedule,
    active_rows: usize,
    graph: bool,
    verification: Option<super::verification::Verification>,
    draft: super::draft::Draft,
    logits: CudaSlice<bf16>,
    predictions: CudaSlice<u32>,
    row_counts: CudaSlice<i32>,
    checkpoint: Option<super::state::StateCheckpoint>,
    gdn_inputs: Vec<super::state::GdnVerifyInput>,
}

#[derive(Clone, Copy)]
enum SpecPhase {
    Draft,
    Verify(usize),
    Inject(usize),
}

impl SpecPhase {
    fn rows(self, spec: &Spec) -> Result<usize> {
        match self {
            Self::Draft => Ok(spec.draft.config.block_size),
            Self::Verify(rows) | Self::Inject(rows) => Ok(rows),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Verify(_) => "verify",
            Self::Inject(_) => "inject",
        }
    }
}

impl Spec {
    fn new(draft: super::draft::Draft, engine: &Engine) -> Result<Self> {
        let max_verify_rows = if engine.model_spec.moe.is_some() {
            draft.config.target_rows(&engine.model_spec)?
        } else {
            draft.config.verify_rows()?
        };
        let rows = if engine.spec_budget.is_tree() {
            engine.spec_budget.rows()
        } else {
            max_verify_rows
        };
        ensure!(
            rows <= engine.workspace_rows && rows <= engine.capacity,
            "DSpark allocated rows={rows} workspace={} capacity={}",
            engine.workspace_rows,
            engine.capacity
        );
        let stream = &engine.device.stream;
        let counts: Vec<_> = (1..=rows).map(|row| row as i32).collect();
        let row_counts = stream.clone_htod(&counts)?;
        stream.synchronize()?;
        let checkpoint = engine
            .model_spec
            .moe
            .as_ref()
            .map(|_| {
                super::state::StateCheckpoint::new(&engine.state, &engine.model_spec, rows, stream)
            })
            .transpose()?;
        let gdn_inputs = if checkpoint.is_some() {
            let gdn_spec = crate::op_contract::GdnSpec::from_model(&engine.model_spec)?;
            engine
                .state
                .gdn
                .iter()
                .map(|_| super::state::GdnVerifyInput::new(&gdn_spec, rows - 1, stream))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let spec = Self {
            #[cfg(test)]
            forced_proposals: None,
            eos: Vec::new(),
            tree_verify_graph: None,
            tree_commit_graph: None,
            tree_draft_graph: None,
            tree: if engine.spec_budget.is_tree() {
                Some(super::tree::State::new(
                    &engine.device,
                    &engine.config,
                    &engine.state.kv,
                    engine.capacity,
                    rows,
                )?)
            } else {
                None
            },
            tree_attention: None,
            max_verify_rows,
            non_expert_decode_bytes: 0,
            verify_schedule: crate::dspark::VerifyCostSchedule::UNCONSTRAINED,
            cold_verify_schedule: crate::dspark::VerifyCostSchedule::UNCONSTRAINED,
            active_rows: 0,
            graph: engine.spec_graph,
            draft_graph: None,
            verify_graph: None,
            moe_verify_graphs: std::collections::BTreeMap::new(),
            inject_graphs: (0..max_verify_rows).map(|_| None).collect(),
            verification: None,
            draft,
            logits: stream.alloc_zeros(rows * engine.config.vocab_size)?,
            predictions: stream.alloc_zeros(rows)?,
            row_counts,
            checkpoint,
            gdn_inputs,
        };
        if let Some(checkpoint) = &spec.checkpoint {
            let actual = checkpoint.bytes()
                + spec
                    .gdn_inputs
                    .iter()
                    .map(super::state::GdnVerifyInput::bytes)
                    .sum::<usize>();
            let planned = spec.draft.config.target_state_bytes(&engine.model_spec)?;
            ensure!(
                actual == planned,
                "DSpark state allocation={actual} differs from planned={planned} rows={rows}"
            );
        }
        Ok(spec)
    }
    fn graph_slot(&mut self, phase: SpecPhase) -> Result<&mut Option<CudaGraph>> {
        Ok(match phase {
            SpecPhase::Draft => &mut self.draft_graph,
            SpecPhase::Verify(_) => &mut self.verify_graph,
            SpecPhase::Inject(rows) => {
                let index = rows.checked_sub(1).context("inject Graph rows=0")?;
                let bound = self.inject_graphs.len();
                self.inject_graphs
                    .get_mut(index)
                    .with_context(|| format!("inject Graph rows={rows} exceeds bound={bound}"))?
            }
        })
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Graph {
    Decode,
    MoeRoute(usize),
    MoeContribute(usize),
    MoeJoin(usize),
    MoePeerTail(usize),
    Draft,
    Verify,
    TreeDraft,
    TreeVerify,
    TreeCommit,
    Inject(usize),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureFault {
    Begin,
    End,
    Instantiate,
}

#[derive(Clone, Copy)]
enum SetupMode {
    Ready,
    Prepare,
    #[cfg(feature = "cuda")]
    Mtp {
        verify_rows: usize,
    },
}

impl SetupMode {
    fn writes(self) -> bool {
        match self {
            Self::Ready => false,
            Self::Prepare => true,
            #[cfg(feature = "cuda")]
            Self::Mtp { .. } => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(feature = "cuda")]
enum LoadDispatch {
    Qwen,
    Llama,
    Recurrent,
    Deepseek,
}

#[cfg(feature = "cuda")]
fn load_dispatch(architecture: crate::model_spec::Architecture) -> Result<LoadDispatch> {
    use crate::model_spec::Architecture;
    Ok(match architecture {
        Architecture::Qwen35Text => LoadDispatch::Qwen,
        Architecture::Llama => LoadDispatch::Llama,
        Architecture::MiniCpmV47 => LoadDispatch::Qwen,
        Architecture::Flint => LoadDispatch::Qwen,
        Architecture::LoopVl => LoadDispatch::Recurrent,
        Architecture::DeepseekV41 => LoadDispatch::Deepseek,
        Architecture::Glm5Next => LoadDispatch::Qwen,
        Architecture::Gemma4 => LoadDispatch::Qwen,
    })
}

pub struct Engine {
    qwen_device_prefix: bool,
    #[cfg(test)]
    capture_fail_after_layer: Option<(usize, usize)>,
    #[cfg(test)]
    capture_fail_stage: Option<CaptureFault>,
    #[cfg(test)]
    flint_window_capture: Option<(Vec<usize>, Vec<usize>, PathBuf)>,
    forward_rows: std::collections::BTreeSet<usize>,
    graphs: Vec<Graph>,
    closure: Rc<super::closure::Closure>,
    graph: Option<CudaGraph>,
    poisoned: bool,
    spec_budget: SpecBudget,
    spec_graph: bool,
    tree_builder: TreeBuilder,
    compare_builders: bool,
    profile_rounds: bool,
    runtime: PathBuf,
    spec: Option<Spec>,
    hidden_capture: Option<HiddenCapture>,
    qwen: Option<qwen::Runtime>,
    recurrent: Option<recurrent::Runtime>,
    #[cfg(feature = "cuda")]
    deepseek_mixer: Option<deepseek::DeepseekMixer>,
    #[cfg(feature = "cuda")]
    deepseek_ffn: Option<deepseek::DeepseekFfn>,
    #[cfg(feature = "cuda")]
    deepseek_engram: Option<deepseek::DeepseekEngram>,
    hc: Option<super::hc::HcRuntime>,
    glm_dsa: Option<qwen::GlmDsaOutputs>,
    moe: Option<moe::Decode>,
    moe_slots: Option<moe::Slots>,
    moe_prefill: Option<moe::Prefill>,
    #[cfg(test)]
    moe_test_ffn: Option<super::linear::BoundFfn>,
    #[cfg(test)]
    flint_legacy_overflow_prefill: bool,
    flint_frozen_decode_length: bool,
    #[cfg(feature = "cuda")]
    mtp: Option<mtp::Chain>,
    #[cfg(feature = "cuda")]
    mtp_state: Option<super::state::StateStore>,
    #[cfg(feature = "cuda")]
    mtp_epoch: u64,
    images: Option<vision_runtime::EncodedImages>,
    weights: Weights,
    state: super::state::StateStore,
    capture_checkpoint: Option<super::state::StateCheckpoint>,
    work: Work,
    ops: Ops,
    blas: Blas,
    flash: Option<Flash>,
    decode: Option<Attention>,
    logits_f32: Option<CudaSlice<f32>>,
    device: Device,
    config: Config,
    model_spec: crate::model_spec::ModelSpec,
    capacity: usize,
    chunk: usize,
    workspace_rows: usize,
    trace_layer: usize,
    position: usize,
    #[cfg(feature = "cuda")]
    prefix_identity: Option<PrefixIdentity>,
    #[cfg(feature = "cuda")]
    prefix_identity_load_ms: f64,
    logits_cache: PrefixLogitsCache,
    committed_prefix_ids: Vec<u32>,
    pending_token: Option<u32>,
    resident_ids_len: usize,
    last_input_kind: Option<PrefixLastInputKind>,
}

#[cfg(feature = "cuda")]
pub struct TextPrefixSaveRequest<'a> {
    pub prefix_ids: &'a [u32],
    pub block_tokens: usize,
    pub transfer_bytes: usize,
}

#[cfg(feature = "cuda")]
pub struct TextPrefixRestoreIntent<'a> {
    pub prefix_ids: &'a [u32],
    pub pending_next_token: u32,
    pub transfer_bytes: usize,
}

#[cfg(feature = "cuda")]
pub struct TextPrefixSaveOutput {
    pub snapshot: PrefixSnapshot,
    pub receipt: PrefixSaveReceipt,
}

#[cfg(feature = "cuda")]
fn pinned_transfer_ring(
    ctx: &Arc<CudaContext>,
    transfer_bytes: usize,
) -> Result<PinnedHostSlice<u8>> {
    validate_transfer_bytes(transfer_bytes)?;
    Ok(unsafe { ctx.alloc_pinned_with_flags::<u8>(transfer_bytes, 0)? })
}

#[cfg(feature = "cuda")]
pub(super) fn dtoh_pinned_chunks<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    source: &CudaSlice<T>,
    source_element_offset: usize,
    bytes: usize,
    ring: &mut PinnedHostSlice<u8>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes);
    let mut source_byte_offset = source_element_offset
        .checked_mul(std::mem::size_of::<T>())
        .context("prefix D2H source offset overflow")?;
    let mut left = bytes;
    while left > 0 {
        let chunk = left.min(ring.len());
        copy::dtoh_bytes(stream, source, source_byte_offset, ring, 0, chunk)?;
        stream.synchronize()?;
        out.extend_from_slice(&ring.as_slice()?[..chunk]);
        source_byte_offset += chunk;
        left -= chunk;
    }
    ensure!(out.len() == bytes, "prefix D2H byte count mismatch");
    Ok(out)
}

#[cfg(feature = "cuda")]
pub(super) fn htod_pinned_chunks<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    destination: &mut CudaSlice<T>,
    destination_element_offset: usize,
    bytes: &[u8],
    ring: &mut PinnedHostSlice<u8>,
) -> Result<()> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let chunk = (bytes.len() - offset).min(ring.len());
        {
            let destination = ring.as_mut_slice()?;
            destination[..chunk].copy_from_slice(&bytes[offset..offset + chunk]);
        }
        copy::htod_bytes(
            stream,
            destination,
            destination_element_offset * std::mem::size_of::<T>() + offset,
            ring,
            0,
            chunk,
        )?;
        stream.synchronize()?;
        offset += chunk;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn text_rope_context_sha256(capacity: usize) -> Result<String> {
    let axes = PositionPlan::text(1, capacity)?.packed_axes()?;
    let mut bytes = Vec::with_capacity(axes.len() * 4);
    for axis in axes {
        bytes.extend_from_slice(&axis.to_le_bytes());
    }
    Ok(sha256(&bytes))
}

#[cfg(feature = "cuda")]
type PrefixAttentionTensors = Vec<(usize, Vec<Vec<u8>>, Vec<Vec<u8>>)>;

#[cfg(feature = "cuda")]
fn prefix_attention_host_tensors(
    snapshot: &PrefixSnapshot,
    head_dim: usize,
) -> Result<PrefixAttentionTensors> {
    let prefix_tokens = snapshot.manifest.boundary.prefix_tokens;
    let block_tokens = snapshot.manifest.block_tokens;
    let prefix_bytes = prefix_tokens * head_dim * 2;
    let mut by_layer = std::collections::BTreeMap::new();
    for (receipt, payload) in snapshot.records() {
        if receipt.key.kind != PrefixRecordKind::AttentionKvBlock {
            continue;
        }
        let segment_bytes = receipt.key.valid_tokens * head_dim * 2;
        ensure!(
            !payload.is_empty() && payload.len() % (2 * segment_bytes) == 0,
            "malformed attention payload extent"
        );
        let kv_heads = payload.len() / (2 * segment_bytes);
        let entry = by_layer
            .entry(receipt.key.layer)
            .or_insert_with(|| (vec![Vec::new(); kv_heads], vec![Vec::new(); kv_heads]));
        ensure!(
            entry.0.len() == kv_heads && entry.1.len() == kv_heads,
            "attention head count changes inside a prefix snapshot"
        );
        for tensor in [&mut entry.0, &mut entry.1] {
            if tensor.iter().any(|x| x.is_empty()) {
                tensor.iter_mut().for_each(|x| x.resize(prefix_bytes, 0));
            }
        }
        let block = receipt.key.token_start / block_tokens;
        let destination_offset = block * block_tokens * head_dim * 2;
        let tensors: [&mut Vec<Vec<u8>>; 2] = [entry.0.as_mut(), entry.1.as_mut()];
        for (tensor_index, tensor) in tensors.into_iter().enumerate() {
            for (head, destination) in tensor.iter_mut().enumerate() {
                let payload_offset = tensor_index * kv_heads * segment_bytes + head * segment_bytes;
                destination[destination_offset..destination_offset + segment_bytes]
                    .copy_from_slice(&payload[payload_offset..payload_offset + segment_bytes]);
            }
        }
    }
    Ok(by_layer
        .into_iter()
        .map(|(layer, (k, v))| (layer, k, v))
        .collect())
}

impl Engine {
    pub fn load(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
    ) -> Result<Self> {
        Self::load_kind(dir, ordinal, capacity, requested_chunk, settings, false)
    }
    #[cfg(feature = "cuda")]
    pub fn calibrate(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        chunk: Option<usize>,
        settings: Settings,
    ) -> Result<Option<Self>> {
        let started = Instant::now();
        let spec = crate::model_spec::ModelSpec::read(dir)?;
        if spec.moe.is_some() && settings.draft_model.is_none() {
            let geometry = crate::experts::catalogue::ExpertGeometry::from_model(&spec)?;
            let runtime = super::calibrate::runtime_dir(
                settings
                    .runtime_dir
                    .as_deref()
                    .context("expert calibration requires a runtime directory")?,
                dir,
                true,
            )?;
            let closure = super::closure::Closure::new(
                true,
                format!(
                    "{:?} --model {dir:?} --device {ordinal} --runtime-dir {runtime:?} calibrate",
                    std::env::current_exe()?,
                ),
            );
            let device = Device::new(ordinal)?;
            let measured = super::calibrate::load_experts(
                &device,
                &runtime,
                geometry,
                spec.expert_format.context("MoE expert_format missing")?,
                crate::cpu::expert_ffn::Form::Lane8,
                &closure,
            )?;
            let budget = crate::experts::plan::admission_budget(
                &geometry,
                &measured.service,
                measured.h2d_bytes_per_second,
                measured.d2h_bytes_per_second,
                measured.copy_engines,
            )?;
            eprintln!(
                "{}",
                serde_json::json!({"expert_setup_calibrated":{"budget_bytes_per_token":budget,"hard_step_cap":crate::experts::plan::Admission { budget_bytes_per_token: budget }.hard_cap(crate::experts::catalogue::ExpertLayout::new(spec.expert_format.context("MoE expert_format missing")?, &geometry)?.slot_bytes())?}})
            );
            if spec.architecture != crate::model_spec::Architecture::Flint {
                return Ok(None);
            }
        }
        let mut engine = Self::load_kind(dir, ordinal, capacity, chunk, settings, true)?;
        if spec.architecture != crate::model_spec::Architecture::Flint {
            return Ok(None);
        }
        engine.finish_flint_calibration(dir, started)?;
        Ok(Some(engine))
    }

    fn finish_flint_calibration(&mut self, _dir: &Path, started: Instant) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let profile = super::closure::Profile::new(
                &self.config,
                self.capacity,
                self.chunk,
                None,
                _dir,
                None,
                None,
            )?;
            let bucket = crate::backend::setup::attention_bucket(self.capacity)?;
            let path = super::calibrate::cache_path(
                &self.runtime,
                &self.device.key,
                bucket,
                self.capacity,
            );
            self.closure.preflight(&[path])?;
            self.closure
                .finish(profile, &self.runtime, self.device.key.clone())?;
        }
        self.reset()?;
        self.prepare_graphs()?;
        self.closure.expect_captures(self.graphs.len())?;
        self.check_graphs()?;
        self.reset()?;
        self.closure.ready(started)
    }
    fn load_kind(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        write: bool,
    ) -> Result<Self> {
        Self::load_kind_for(
            dir,
            ordinal,
            capacity,
            requested_chunk,
            settings,
            if write {
                SetupMode::Prepare
            } else {
                SetupMode::Ready
            },
        )
    }
    #[cfg(feature = "cuda")]
    fn load_kind_for(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        mode: SetupMode,
    ) -> Result<Self> {
        let write = mode.writes();
        let started = Instant::now();
        let model_spec = crate::model_spec::ModelSpec::read(dir)?;
        match load_dispatch(model_spec.architecture)? {
            LoadDispatch::Recurrent => {
                return Self::load_recurrent(
                    dir,
                    ordinal,
                    capacity,
                    requested_chunk,
                    settings,
                    model_spec,
                );
            }
            LoadDispatch::Deepseek => {
                #[cfg(not(feature = "cuda"))]
                anyhow::bail!("DeepSeek native operators require CUDA");
                #[cfg(feature = "cuda")]
                return Self::load_deepseek(
                    dir,
                    ordinal,
                    capacity,
                    requested_chunk,
                    settings,
                    model_spec,
                );
            }
            LoadDispatch::Qwen => {
                return Self::load_qwen(
                    dir,
                    ordinal,
                    capacity,
                    requested_chunk,
                    settings,
                    mode,
                    model_spec,
                );
            }
            LoadDispatch::Llama => {}
        }
        ensure!(
            settings.kv_format == crate::kv::KvFormat::Bf16,
            "8-bit KV requires the Qwen3.8 text runtime"
        );
        let draft_path = settings.draft_model.clone();
        let quote = |p: &Path| format!("'{}'", p.display().to_string().replace('\'', "'\"'\"'"));
        let runtime_arg = settings
            .runtime_dir
            .as_deref()
            .context("CUDA setup requires --runtime-dir or FLYINGFISH_RUNTIME_DIR")?;
        let mut command = format!(
            "{} --model {} --device {ordinal} --capacity {capacity} --runtime-dir {}",
            quote(&std::env::current_exe()?),
            quote(dir),
            quote(runtime_arg)
        );
        if let Some(path) = &draft_path {
            command.push_str(&format!(
                " --draft-model {} --spec-budget {}",
                quote(path),
                match settings.spec_budget {
                    SpecBudget::Chain => "chain",
                    SpecBudget::Tree16 => "tree16",
                    SpecBudget::Tree32 => "tree32",
                    SpecBudget::Tree64 => "tree64",
                }
            ));
        }
        if let Some(chunk) = requested_chunk {
            command.push_str(&format!(" --chunk {chunk}"));
        }
        command.push_str(" calibrate");
        let closure = Rc::new(super::closure::Closure::new(write, command));
        let config = Config::read(dir)?;
        let draft_config = draft_path
            .as_deref()
            .map(|path| crate::dspark::DraftConfig::read(path, &config))
            .transpose()?;
        if let Some(draft) = &draft_config {
            ensure!(
                draft.block_size == 7,
                "unsupported Llama DSpark block_size={}; requires block_size=7",
                draft.block_size
            );
            let rows = draft.verify_rows()?;
            ensure!(
                rows <= crate::linear_plan::MAX_SKINNY_ROWS,
                "DSpark block_size={} verify_rows={rows} exceeds skinny bound={}",
                draft.block_size,
                crate::linear_plan::MAX_SKINNY_ROWS
            );
            ensure!(
                !settings.spec_budget.is_tree() || rows == SpecBudget::Chain.rows(),
                "tree draft block_size={} requires the declared chain rows={}",
                draft.block_size,
                SpecBudget::Chain.rows()
            );
            ensure!(
                rows < capacity,
                "DSpark block_size={} verify_rows={rows} exceeds capacity={capacity}",
                draft.block_size
            );
        }
        let model_spec = crate::model_spec::ModelSpec::from_llama(&config)?;
        ensure!(
            capacity > 0
                && capacity <= config.max_position_embeddings
                && capacity < i32::MAX as usize,
            "invalid KV capacity {capacity}"
        );
        ensure!(
            !settings.spec_budget.is_tree() || capacity >= settings.spec_budget.rows(),
            "tree budget {} exceeds KV capacity {capacity}",
            settings.spec_budget.rows()
        );
        let runtime = super::calibrate::runtime_dir(
            settings.runtime_dir.as_deref().context(
                "CUDA attention calibration requires --runtime-dir or FLYINGFISH_RUNTIME_DIR",
            )?,
            dir,
            write,
        )?;
        let device = Device::new(ordinal)?;
        let weights = Weights::load(dir, &config, &device).context("load target weights")?;
        let s = &device.stream;
        let state = super::state::StateStore::new(&model_spec, capacity, s)?;
        let capture_checkpoint = Some(super::state::StateCheckpoint::new(
            &state,
            &model_spec,
            1,
            s,
        )?);
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, model_spec.swiglu_rounding())
            .context("load CUDA operator modules")?;
        eprintln!("{}", serde_json::json!({"backend_device":device.info}));
        let blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        let workspace_rows = if let Some(draft) = &draft_config {
            if settings.spec_budget.is_tree() {
                settings.spec_budget.rows()
            } else {
                draft.verify_rows()?
            }
        } else {
            settings.spec_budget.rows().min(capacity)
        };
        let chunk = workspace_rows;
        let mut work = Work::new(s, &model_spec, workspace_rows)?;
        s.memcpy_htod(&[1i32], &mut work.length)?;
        let flash = if model_spec.architecture == crate::model_spec::Architecture::Gemma4 {
            None
        } else {
            Some(Flash::new(
                s.clone(),
                workspace_rows,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )?)
        };
        s.synchronize()?;
        let mut engine = Self {
            #[cfg(test)]
            #[cfg(feature = "cuda")]
            capture_fail_after_layer: None,
            #[cfg(test)]
            #[cfg(feature = "cuda")]
            capture_fail_stage: None,
            #[cfg(test)]
            flint_window_capture: None,
            qwen: None,
            #[cfg(feature = "cuda")]
            deepseek_mixer: None,
            #[cfg(feature = "cuda")]
            deepseek_ffn: None,
            #[cfg(feature = "cuda")]
            deepseek_engram: None,
            recurrent: None,
            hc: None,
            glm_dsa: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            #[cfg(test)]
            moe_test_ffn: None,
            #[cfg(test)]
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            #[cfg(feature = "cuda")]
            mtp: None,
            #[cfg(feature = "cuda")]
            mtp_state: None,
            #[cfg(feature = "cuda")]
            mtp_epoch: 0,
            images: None,
            qwen_device_prefix: false,
            forward_rows: std::collections::BTreeSet::new(),
            #[cfg(feature = "cuda")]
            graphs: Vec::new(),
            closure,
            poisoned: false,
            spec_budget: settings.spec_budget,
            #[cfg(feature = "cuda")]
            spec_graph: settings.spec_graph,
            #[cfg(feature = "cuda")]
            tree_builder: settings.tree_builder,
            #[cfg(feature = "cuda")]
            compare_builders: false,
            #[cfg(feature = "cuda")]
            profile_rounds: false,
            runtime,
            #[cfg(feature = "cuda")]
            spec: None,
            #[cfg(feature = "cuda")]
            hidden_capture: None,
            #[cfg(feature = "cuda")]
            graph: None,
            weights,
            state,
            #[cfg(feature = "cuda")]
            capture_checkpoint,
            work,
            ops,
            blas,
            #[cfg(feature = "cuda")]
            flash,
            decode: None,
            logits_f32: None,
            device,
            config,
            model_spec,
            capacity,
            chunk,
            workspace_rows,
            trace_layer: 0,
            position: 0,
            #[cfg(feature = "cuda")]
            prefix_identity: None,
            #[cfg(feature = "cuda")]
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        if let Some(path) = settings.draft_model {
            let path = path.canonicalize()?;
            let draft = super::draft::Draft::new(
                &path,
                &engine.device,
                &engine.config,
                engine.capacity,
                engine.workspace_rows,
                &mut engine.blas,
                super::markov::ScoreRounding::Bf16,
            )?;
            engine.hidden_capture = Some(HiddenCapture::new(
                &draft.config.target_layer_ids,
                &engine.config,
                engine.workspace_rows,
                &engine.device.stream,
            )?);
            let verify_rows = draft.config.verify_rows()?;
            engine.prepare(verify_rows)?;
            engine.blas.prepare(
                verify_rows,
                engine.config.vocab_size,
                engine.config.hidden_size,
            )?;
            let rows = if engine.spec_budget.is_tree() {
                engine.spec_budget.rows()
            } else {
                verify_rows
            };
            if engine.spec_budget.is_tree() {
                engine.prepare(rows)?;
                engine
                    .blas
                    .prepare(rows, engine.config.vocab_size, engine.config.hidden_size)?;
            }
            engine.spec = Some(Spec::new(draft, &engine)?);
        }
        engine.size_workspace(requested_chunk)?;
        engine.declare_graphs()?;
        if let Some(spec) = &engine.spec {
            eprintln!(
                "{}",
                serde_json::json!({"spec_execution":{"graph":engine.spec_graph,"budget":engine.spec_budget,"draft_rows":spec.draft.config.block_size,"verify_rows":spec.draft.config.verify_rows()?,"workspace_rows":engine.workspace_rows,"inject_rows":spec.inject_graphs.len(),"acceptance":"host greedy prefix"}})
            );
        }
        engine.device.stream.synchronize()?;
        let (free, total) = engine.device.ctx.mem_get_info()?;
        let kv_bytes: usize = engine
            .state
            .kv
            .iter()
            .map(|kv| (kv.k.len() + kv.v.len()) * size_of::<bf16>())
            .sum();
        eprintln!(
            "weights_bytes={} kv_bytes={} device_used_bytes={} device_total_bytes={}",
            engine.weights.bytes,
            kv_bytes,
            total - free,
            total
        );
        engine.initialize(dir, draft_path.as_deref(), write, started)?;
        Ok(engine)
    }

    fn grow_rows(&mut self, rows: usize, chunk: usize) -> Result<()> {
        let old_rows = self.workspace_rows;
        let old_chunk = self.chunk;
        ensure!(rows >= old_rows, "workspace cannot shrink during load");
        #[cfg(feature = "cuda")]
        ensure!(
            self.deepseek_engram.is_none() || rows == old_rows,
            "Engram workspace is fixed at setup"
        );
        if rows != old_rows {
            self.work
                .grow(&self.device.stream, &self.model_spec, rows)?;
            if let Some(hc) = &mut self.hc {
                hc.grow(&self.device.stream, rows)?;
            }
            #[cfg(feature = "cuda")]
            if let Some(mixer) = &mut self.deepseek_mixer {
                mixer.grow(&self.device.stream, rows)?;
            }
            #[cfg(feature = "cuda")]
            if let Some(ffn) = &mut self.deepseek_ffn {
                ffn.grow(&self.device.stream, rows)?;
            }
            if let Some(spec) = &mut self.spec {
                spec.draft.grow(rows)?;
            }
            if let Some(capture) = &mut self.hidden_capture {
                capture.grow(rows, &self.device.stream)?;
            }
        }
        if rows != old_rows || chunk != old_chunk || self.flash.is_none() {
            self.flash = Some(Flash::new(
                self.device.stream.clone(),
                rows,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )?);
        }
        self.workspace_rows = rows;
        self.chunk = chunk;
        Ok(())
    }

    fn size_workspace(&mut self, requested_chunk: Option<usize>) -> Result<()> {
        self.device.finish_upload()?;
        self.device.stream.synchronize()?;
        let (free, total) = self.device.ctx.mem_get_info()?;
        let plan = crate::prefill::Plan::new(
            &self.config,
            self.capacity,
            requested_chunk,
            free,
            total,
            Flash::tile(self.config.head_dim)?,
            self.hidden_capture
                .as_ref()
                .map_or(0, |capture| capture.width * 2)
                + self
                    .spec
                    .as_ref()
                    .map(|spec| spec.draft.config.workspace_row_bytes())
                    .transpose()?
                    .unwrap_or(0),
        )?;
        let rows = spec_workspace_rows(
            plan.chunk,
            if cfg!(feature = "cuda") || self.spec.is_some() {
                Some(self.spec_budget)
            } else {
                None
            },
            self.spec
                .as_ref()
                .map(|spec| spec.draft.config.verify_rows())
                .transpose()?
                .unwrap_or(1),
        );
        let required = rows
            .checked_mul(plan.row_bytes)
            .context("workspace byte count overflow")?;
        let available = free.saturating_sub(plan.reserve_bytes);
        ensure!(
            rows <= plan.row_limit,
            "workspace {rows} rows exceeds row budget {}: required={required} available={available}",
            plan.row_limit
        );
        eprintln!(
            "{}",
            serde_json::json!({"prefill_memory_plan":plan,"draft_allocated":self.spec.is_some()})
        );
        self.poisoned = true;
        self.grow_rows(rows, plan.chunk).with_context(|| {
            format!("allocate prefill workspace: required={required} available={available}")
        })?;
        if cfg!(feature = "cuda") || self.qwen.is_some() {
            self.prepare_prefill_plans()?;
        }
        self.poisoned = false;
        Ok(())
    }

    fn prepare_prefill_plans(&mut self) -> Result<()> {
        if self.qwen.is_none() {
            self.prepare(self.chunk)?;
            return Ok(());
        }
        let workspace = self.workspace_rows;
        let mut rows: std::collections::BTreeSet<usize> = (1..=workspace.min(16)).collect();
        let mut bucket = 32usize;
        while workspace > 16 && bucket < workspace {
            rows.insert(bucket);
            bucket *= 2;
        }
        rows.insert(workspace);
        for row in rows {
            self.prepare(row)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_decode(&mut self) -> Result<()> {
        if self.model_spec.architecture == crate::model_spec::Architecture::DeepseekV41 {
            return self
                .deepseek_mixer
                .as_ref()
                .context("DeepSeek-V4.1 decode native mixer is not bound")?
                .check_decode_ready();
        }
        if self.qwen.is_some()
            && self.model_spec.architecture != crate::model_spec::Architecture::Flint
        {
            return Ok(());
        }
        if self.decode.is_some() {
            return Ok(());
        }
        let calibrated = super::calibrate::load(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            self.capacity,
            &self.closure,
        )
        .context("attention setup calibration")?;
        let setup = crate::backend::setup::Setup::select(
            self.device.info.clone(),
            &self.config,
            self.capacity,
            |_info, capacity| calibrated.choose(capacity),
        )?;
        #[cfg(feature = "cuda")]
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"rows":1,"device":setup.device,"selection":setup.selection,"attention":setup.attention,"calibration":calibrated}})
        );
        let decode = Attention::new(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            self.capacity,
            setup.attention,
        )?;
        self.decode = Some(decode);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn bind_recorded_decode(&mut self) -> Result<()> {
        if self.decode.is_some() {
            return Ok(());
        }
        let calibrated =
            super::calibrate::load_recorded(&self.device, &self.runtime, self.capacity)?;
        let setup = crate::backend::setup::Setup::select(
            self.device.info.clone(),
            &self.config,
            self.capacity,
            |_info, capacity| calibrated.choose(capacity),
        )?;
        self.decode = Some(Attention::new(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            self.capacity,
            setup.attention,
        )?);
        Ok(())
    }

    pub(crate) fn views(&mut self) -> Views<'_> {
        Views {
            device: &self.device,
            config: &self.config,
            weights: &self.weights,
            kv: &mut self.state.kv,
            work: &mut self.work,
            capacity: self.capacity,
            head_stride: self.capacity * self.config.head_dim,
            pos_stride: self.config.head_dim,
        }
    }

    pub fn device_info(&self) -> &crate::backend::setup::DeviceInfo {
        &self.device.info
    }

    fn declare_graphs(&mut self) -> Result<()> {
        self.graphs = 'plans: {
            if self.spec.as_ref().is_some_and(|spec| !spec.graph) {
                break 'plans Vec::new();
            }
            if self.model_spec.moe.is_some() {
                let widths = match &self.spec {
                    Some(spec) => crate::dspark::MIN_MOE_VERIFY_ROWS..=spec.max_verify_rows,
                    None => 1..=1,
                };
                let mut graphs: Vec<_> = if self
                    .moe_slots
                    .as_ref()
                    .context("MoE graph declaration has no slots")?
                    .all_resident()
                {
                    widths
                        .map(|_| {
                            if self.spec.is_some() {
                                Graph::Verify
                            } else {
                                Graph::Decode
                            }
                        })
                        .collect()
                } else {
                    let tail_start = self.moe_peer_tail_start()?;
                    let seg_end = tail_start.unwrap_or(self.model_spec.layers.len());
                    widths
                        .flat_map(|_| {
                            (0..seg_end)
                                .flat_map(|layer| {
                                    [
                                        Graph::MoeRoute(layer),
                                        Graph::MoeContribute(layer),
                                        Graph::MoeJoin(layer),
                                    ]
                                })
                                .chain(tail_start.map(Graph::MoePeerTail))
                        })
                        .collect()
                };
                if let Some(spec) = &self.spec {
                    graphs.push(Graph::Draft);
                    graphs.extend((1..=spec.inject_graphs.len()).map(Graph::Inject));
                }
                break 'plans graphs;
            }
            match &self.spec {
                None => vec![Graph::Decode],
                Some(spec) if !spec.graph => Vec::new(),
                Some(spec) => {
                    let mut graphs = if self.spec_budget.is_tree() {
                        vec![Graph::TreeDraft, Graph::TreeVerify, Graph::TreeCommit]
                    } else {
                        vec![Graph::Draft, Graph::Verify]
                    };
                    graphs.extend((1..=spec.inject_graphs.len()).map(Graph::Inject));
                    graphs
                }
            }
        };
        eprintln!(
            "{}",
            serde_json::json!({"backend_graph_declaration":self.graphs})
        );
        Ok(())
    }

    fn has_graph(&self, graph: Graph) -> bool {
        if matches!(graph, Graph::Decode) {
            return self.graph.is_some();
        }
        if let Graph::MoeRoute(layer) | Graph::MoeContribute(layer) | Graph::MoeJoin(layer) = graph
        {
            let seg_end = self
                .moe_peer_tail_start()
                .ok()
                .flatten()
                .unwrap_or(self.model_spec.layers.len());
            return layer < seg_end && self.has_moe_segments();
        }
        #[cfg(feature = "cuda")]
        if let Graph::MoePeerTail(start) = graph {
            return self.moe_peer_tail_start().ok().flatten() == Some(start)
                && self.has_moe_peer_tail();
        }
        let Some(spec) = &self.spec else {
            return false;
        };
        match graph {
            Graph::Decode => self.graph.is_some(),
            Graph::MoeRoute(_)
            | Graph::MoeContribute(_)
            | Graph::MoeJoin(_)
            | Graph::MoePeerTail(_) => false,
            Graph::Draft => spec.draft_graph.is_some(),
            Graph::Verify => {
                if self.model_spec.moe.is_some() {
                    return self.has_moe_verify_graphs();
                }
                spec.verify_graph.is_some()
            }
            Graph::TreeDraft => spec.tree_draft_graph.is_some(),
            Graph::TreeVerify => spec.tree_verify_graph.is_some(),
            Graph::TreeCommit => spec.tree_commit_graph.is_some(),
            Graph::Inject(rows) => rows
                .checked_sub(1)
                .and_then(|r| spec.inject_graphs.get(r))
                .is_some_and(Option::is_some),
        }
    }
    fn spec_verify_graph_ready(&self) -> bool {
        if self.has_moe_segments() {
            return true;
        }
        self.has_graph(Graph::Verify)
    }
    fn check_graphs(&self) -> Result<()> {
        for &graph in &self.graphs {
            ensure!(
                self.has_graph(graph),
                "production Graph is missing: {graph:?}"
            );
        }
        Ok(())
    }
    pub fn check_ready(&self) -> Result<()> {
        ensure!(
            !self.poisoned,
            "engine state is poisoned; reset/reload is required"
        );
        self.closure.assert_ready()?;
        self.check_graphs()?;
        Ok(())
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    #[cfg(feature = "cuda")]
    fn prepare(&mut self, rows: usize) -> Result<()> {
        if self.qwen.is_some() {
            return self.prepare_qwen(rows);
        }
        let c = &self.config;
        for (o, i) in [
            (c.qkv_dim(), c.hidden_size),
            (c.hidden_size, c.hidden_size),
            (2 * c.intermediate_size, c.hidden_size),
            (c.hidden_size, c.intermediate_size),
        ] {
            self.blas.prepare(rows, o, i)?;
        }
        Ok(())
    }

    fn snapshot(
        &self,
        trace: &mut Trace,
        name: String,
        data: &CudaSlice<bf16>,
        skip: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let values = self
            .device
            .stream
            .clone_dtoh(&data.slice(skip * width..rows * width))?;
        trace.add(
            name,
            vec![1, rows - skip, width],
            values.into_iter().map(bf16::to_f32).collect(),
        )
    }

    fn forward(
        &mut self,
        rows: usize,
        use_token: bool,
        head: bool,
        trace: Option<Target<'_>>,
    ) -> Result<()> {
        let step = crate::backend::Step {
            rows,
            input: if use_token {
                crate::backend::Input::Token
            } else {
                crate::backend::Input::Ids
            },
            head: if head { Head::Last } else { Head::None },
            mask: crate::backend::Mask::Causal,
        };
        self.run_step(step, trace)
    }

    fn check_step(&self, step: crate::backend::Step<'_>) -> Result<()> {
        ensure!(
            self.recurrent.is_none(),
            "recurrent execution requires the prefix/decode program"
        );
        if self.qwen.is_some() {
            ensure!(
                matches!(step.mask, crate::backend::Mask::Causal)
                    && (step.head != Head::All
                        || self.model_spec.moe.is_some()
                        || self.model_spec.architecture == crate::model_spec::Architecture::Gemma4),
                "hybrid forward requires causal mask and a bound MoE chain for head={:?}",
                step.head
            );
        }
        let linear_rows = if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            self.spec_budget.rows()
        } else {
            step.rows
        };
        let available_rows = self
            .workspace_rows
            .min(self.capacity.saturating_sub(self.position));
        step.check(available_rows, linear_rows, &self.forward_rows)?;
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(
            !(self.spec_budget.is_tree()
                && step.head == Head::All
                && matches!(step.mask, crate::backend::Mask::Causal)),
            "causal all-row verification requires a chain spec budget"
        );
        if step.input == crate::backend::Input::Ids
            && step.head == Head::All
            && matches!(step.mask, crate::backend::Mask::Causal)
            && self.model_spec.architecture != crate::model_spec::Architecture::Gemma4
        {
            let spec = self
                .spec
                .as_ref()
                .context("causal verification requires draft geometry")?;
            let valid = if self.model_spec.moe.is_some() {
                (crate::dspark::MIN_MOE_VERIFY_ROWS..=spec.max_verify_rows).contains(&step.rows)
            } else {
                step.rows == spec.max_verify_rows
            };
            ensure!(
                valid,
                "causal all-row verification rows={} differs from the loaded chain geometry",
                step.rows
            );
        }
        Ok(())
    }

    fn prepare_step(&mut self, step: crate::backend::Step<'_>) -> Result<()> {
        self.check_step(step)?;
        let rows = if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            self.spec_budget.rows()
        } else {
            step.rows
        };
        if step.input == crate::backend::Input::Ids {
            self.prepare_prefill_rows(rows)?;
        } else {
            self.prepare(rows)?;
        }
        if step.head == Head::All {
            self.blas
                .prepare(rows, self.config.vocab_size, self.config.hidden_size)?;
        }
        Ok(())
    }

    pub fn run_step(
        &mut self,
        step: crate::backend::Step<'_>,
        trace: Option<Target<'_>>,
    ) -> Result<()> {
        let previous = self.qwen_device_prefix;
        if self.qwen.is_some()
            && step.head == Head::All
            && self.model_spec.architecture != crate::model_spec::Architecture::Gemma4
        {
            self.qwen_device_prefix = true;
        }
        let result = self.run_step_inner(step, trace);
        self.qwen_device_prefix = previous;
        if (cfg!(feature = "rocm") || self.qwen.is_some()) && result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn run_step_inner(
        &mut self,
        step: crate::backend::Step<'_>,
        mut trace: Option<Target<'_>>,
    ) -> Result<()> {
        self.prepare_step(step)?;
        if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            self.prepare_tree(step)?;
        }
        if step.input == crate::backend::Input::Token {
            self.ensure_decode()?;
        } else if step.head == Head::All && matches!(step.mask, crate::backend::Mask::Causal) {
            {
                if self.model_spec.architecture != crate::model_spec::Architecture::Gemma4 {
                    ensure!(
                        step.rows <= crate::linear_plan::MAX_SKINNY_ROWS,
                        "chain verification rows={} exceeds skinny bound={}",
                        step.rows,
                        crate::linear_plan::MAX_SKINNY_ROWS
                    );
                    self.ensure_verification()?;
                }
                if self.qwen.is_some() {
                    self.state.set_sequence(
                        &self.device.stream,
                        self.position as u32,
                        step.rows as u32,
                        self.workspace_rows as u32,
                    )?;
                }
            }
        }
        if let Some(slots) = self.moe_slots.as_mut() {
            #[cfg(feature = "cuda")]
            if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next
                && !slots.request_open
            {
                slots.begin_staging_request()?;
            }
            slots.begin_token()?;
        }
        let executed = (|| {
            self.run_operator(crate::backend::Operator::Embed, 0, step, &mut trace)?;
            if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
                self.hc
                    .as_mut()
                    .context("GLM-5.3 mHC runtime is not bound")?
                    .begin_embedded(&self.device.stream, step.rows, &self.work.x)?;
            }
            for i in 0..self.config.num_hidden_layers {
                self.run_layer(i, step, &mut trace)?;
            }
            self.run_operator(crate::backend::Operator::Head, 0, step, &mut trace)
        })();
        let completed = if let Some(slots) = self.moe_slots.as_mut() {
            slots.finish_token()
        } else {
            Ok(())
        };
        executed?;
        completed?;
        if step.head == Head::None {
            self.logits_cache.invalidate();
        } else {
            self.logits_cache.mark_valid();
        }
        Ok(())
    }

    /// Host-side validation for the tree step; the Embed operator writes the tree metadata.
    fn prepare_tree(&mut self, step: crate::backend::Step<'_>) -> Result<()> {
        self.spec_budget
            .check_tree(step, self.position, self.capacity)?;
        let spec = self
            .spec
            .as_ref()
            .context("tree verification requires enabled draft state")?;
        ensure!(
            spec.predictions.len() >= step.rows,
            "tree predictions exceed allocated workspace"
        );
        let tree = spec.tree.as_ref().context("tree State is not enabled")?;
        ensure!(
            tree.rows() == step.rows && tree.budget() == self.spec_budget.rows(),
            "tree metadata rows/budget do not match forward"
        );
        self.views().check(step.rows)?;
        self.ensure_tree_attention()?;
        Ok(())
    }

    fn run_attention(&mut self, layer: usize, step: crate::backend::Step<'_>) -> Result<()> {
        if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            let spec = self
                .spec
                .as_mut()
                .context("tree attention requires draft State")?;
            let tree = spec.tree.as_ref().context("tree State is not enabled")?;
            let kv = &self.state.kv[layer];
            return spec
                .tree_attention
                .as_mut()
                .context("tree attention is not calibrated")?
                .run_tree(
                    super::verification::Query {
                        qkv: &self.work.qkv,
                        k: &kv.k,
                        v: &kv.v,
                        out: &mut self.work.attn,
                        prefix: tree.prefix(),
                        shape: Shape {
                            rows: tree.budget(),
                            start: self.position,
                            q_heads: self.config.num_attention_heads,
                            kv_heads: self.config.num_key_value_heads,
                            capacity: self.capacity,
                            head_dim: self.config.head_dim,
                            causal: false,
                            window_left: None,
                        },
                    },
                    tree.ancestors(),
                );
        }
        if step.input == crate::backend::Input::Token {
            self.ensure_decode()?;
        }
        let w = &mut self.work;
        let kv = &self.state.kv[layer];
        let c = &self.config;
        if step.input == crate::backend::Input::Token {
            self.decode
                .as_mut()
                .context("decode attention has not been initialized")?
                .run(
                    &self.device.stream,
                    &w.qkv,
                    &kv.k,
                    &kv.v,
                    &mut w.attn,
                    &w.length,
                )
        } else {
            let shape = Shape {
                rows: step.rows,
                start: self.position,
                q_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                capacity: self.capacity,
                head_dim: c.head_dim,
                causal: true,
                window_left: None,
            };
            if step.head == Head::All {
                return self
                    .spec
                    .as_mut()
                    .context("verification attention requires draft state")?
                    .verification
                    .as_mut()
                    .context("verification attention has not been initialized")?
                    .run(&w.qkv, &kv.k, &kv.v, &mut w.attn, &w.position, shape);
            }
            self.flash
                .as_mut()
                .context("dense prefill attention is not bound")?
                .run(
                    &w.qkv,
                    (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim,
                    &kv.k,
                    &kv.v,
                    &mut w.attn,
                    shape,
                )
        }
    }

    pub fn verification_impl(&self) -> &'static str {
        if self.spec_budget.is_tree() {
            return self
                .spec
                .as_ref()
                .and_then(|s| s.tree_attention.as_ref())
                .map(|s| s.implementation())
                .unwrap_or("tree attention has not been initialized");
        }
        self.spec
            .as_ref()
            .and_then(|s| s.verification.as_ref())
            .map(|s| s.implementation())
            .unwrap_or("verification has not been initialized")
    }
    #[cfg(feature = "cuda")]
    pub fn profile_rounds(&mut self, enabled: bool) {
        self.profile_rounds = enabled;
    }

    #[cfg(test)]
    pub(crate) fn gemma_prefill_all(&mut self, ids: &[u32]) -> Result<()> {
        ensure!(!ids.is_empty(), "Gemma 4 prefill requires token IDs");
        self.device.stream.memcpy_htod(ids, &mut self.work.ids)?;
        self.run_step(
            crate::backend::Step {
                rows: ids.len(),
                input: crate::backend::Input::Ids,
                head: Head::All,
                mask: crate::backend::Mask::Causal,
            },
            None,
        )?;
        self.position += ids.len();
        self.device
            .stream
            .memcpy_htod(&[self.position as i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[(self.position + 1) as i32], &mut self.work.length)?;
        if self.qwen.is_some() && self.position < self.capacity {
            self.state.set_sequence(
                &self.device.stream,
                self.position as u32,
                1,
                self.workspace_rows as u32,
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn gemma_prefill_trace(
        &mut self,
        ids: &[u32],
        trace: &mut crate::trace::Trace,
    ) -> Result<()> {
        ensure!(!ids.is_empty(), "Gemma 4 prefill requires token IDs");
        self.device.stream.memcpy_htod(ids, &mut self.work.ids)?;
        self.run_step(
            crate::backend::Step {
                rows: ids.len(),
                input: crate::backend::Input::Ids,
                head: Head::All,
                mask: crate::backend::Mask::Causal,
            },
            Some(Target {
                trace,
                prefix: "",
                skip: 0,
            }),
        )?;
        self.position += ids.len();
        self.device
            .stream
            .memcpy_htod(&[self.position as i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[(self.position + 1) as i32], &mut self.work.length)?;
        if self.qwen.is_some() && self.position < self.capacity {
            self.state.set_sequence(
                &self.device.stream,
                self.position as u32,
                1,
                self.workspace_rows as u32,
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn gemma_rope_table(&self, full: bool, positions: &[usize]) -> Result<Vec<bf16>> {
        let ops = self
            .qwen
            .as_ref()
            .and_then(|q| q.gemma.as_ref())
            .context("Gemma 4 operations missing")?;
        let d = if full { 512 } else { 256 };
        let table = &ops.rope_tables[usize::from(full)];
        let mut out = Vec::with_capacity(positions.len() * d);
        for &position in positions {
            ensure!(
                position < self.capacity,
                "RoPE table position outside capacity"
            );
            out.extend(
                self.device
                    .stream
                    .clone_dtoh(&table.slice(position * d..(position + 1) * d))?,
            );
        }
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn gemma_position(&self) -> Result<(usize, i32)> {
        Ok((
            self.position,
            self.device.stream.clone_dtoh(&self.work.position)?[0],
        ))
    }

    #[cfg(test)]
    pub(crate) fn gemma_hidden_state(&self) -> Result<(Vec<bf16>, Vec<bf16>)> {
        let h = self.model_spec.hidden as usize;
        Ok((
            self.device.stream.clone_dtoh(&self.work.x.slice(..h))?,
            self.device.stream.clone_dtoh(&self.work.n.slice(..h))?,
        ))
    }

    #[cfg(test)]
    pub(crate) fn gemma_all_logits(&self, row: usize) -> Result<Vec<bf16>> {
        self.qwen
            .as_ref()
            .context("Gemma 4 runtime missing")?
            .gemma_logits(&self.device.stream, row, self.model_spec.vocab as usize)
    }

    pub fn verification_logits(&self, row: usize) -> Result<Vec<f32>> {
        let spec = self.spec.as_ref().context("draft state is not enabled")?;
        ensure!(
            row < spec.active_rows,
            "verification logit row {row} exceeds active rows {}",
            spec.active_rows
        );
        let n = self.config.vocab_size;
        Ok(self
            .device
            .stream
            .clone_dtoh(&spec.logits.slice(row * n..(row + 1) * n))?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    pub fn run_layer(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        self.prepare_step(step)?;
        ensure!(
            layer < self.config.num_hidden_layers,
            "backend layer {layer} is outside model"
        );
        if self.qwen.is_some() {
            self.qwen_layer_begin(layer, step)?;
        }
        self.run_layer_program(layer, step, trace)
    }

    fn run_layer_program(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(engram) = &self.deepseek_engram {
            engram.run(
                &self.device.stream,
                layer,
                step.rows,
                self.hc.as_mut().context("Engram mHC stream missing")?,
            )?;
        }
        let program = crate::model::layer_program(&self.model_spec, layer)?;
        let stages: &[crate::model::LayerStage] = match &program {
            crate::model::LayerProgram::Standard(stages) => &stages[..],
            crate::model::LayerProgram::Hc(stages) => &stages[..],
            crate::model::LayerProgram::PerLayerInput(stages) => &stages[..],
        };
        for (position, &stage) in stages.iter().enumerate() {
            match stage {
                crate::model::LayerStage::Common(op) => {
                    self.run_operator(op, layer, step, trace)?
                }
                crate::model::LayerStage::Mixer(mixer) => {
                    self.run_mixer(mixer, layer, step, trace)?
                }
                crate::model::LayerStage::Ffn => self.run_ffn(layer, step, trace)?,
                crate::model::LayerStage::Hc(kind) => {
                    self.run_hc_stage(position, kind, layer, step)?
                }
                crate::model::LayerStage::PerLayerInput => {
                    self.qwen_per_layer_input(layer, step, trace)?
                }
            }
            #[cfg(test)]
            if let Some(calls) = self.qwen.as_ref().and_then(|q| q.glm_calls.as_ref()) {
                calls.borrow_mut()[layer][position] += step.rows;
            }
        }
        #[cfg(test)]
        self.check_capture_layer(layer)?;
        Ok(())
    }

    fn run_hc_stage(
        &mut self,
        stage: usize,
        kind: crate::model::HcStage,
        layer: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        let rows = if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            self.spec_budget.rows()
        } else {
            step.rows
        };
        let runtime = self.hc.as_mut().context("mHC runtime is not bound")?;
        let plan = crate::model::hc_stage_plan(stage, kind, runtime.binding.policy)?;
        match plan {
            crate::model::HcStagePlan::Pre { sublayer, source } => runtime.hc_pre(
                &self.device.stream,
                layer,
                sublayer,
                source,
                rows,
                &mut self.work.x,
            ),
            crate::model::HcStagePlan::Post { sublayer } => {
                runtime.hc_post(&self.device.stream, layer, sublayer, rows, &self.work.out)
            }
        }
    }

    #[cfg(test)]
    fn check_capture_layer(&mut self, layer: usize) -> Result<()> {
        if let Some((fail_layer, remaining)) = &mut self.capture_fail_after_layer
            && *fail_layer == layer
        {
            *remaining = remaining
                .checked_sub(1)
                .context("invalid failure injection countdown")?;
            ensure!(
                *remaining != 0,
                "injected capture failure after layer {layer}"
            );
        }
        Ok(())
    }

    fn run_mixer(
        &mut self,
        mixer: crate::model_spec::Mixer,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        use crate::model_spec::{Architecture, Mixer};
        match (self.model_spec.architecture, mixer) {
            (
                Architecture::Qwen35Text
                | Architecture::MiniCpmV47
                | Architecture::Flint
                | Architecture::Glm5Next
                | Architecture::Gemma4,
                _,
            ) => self.qwen_mixer(layer, step, trace),
            (Architecture::Llama, Mixer::FullAttention) => {
                for op in crate::model::LLAMA_MIXER {
                    self.run_operator(op, layer, step, trace)?;
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (Architecture::DeepseekV41, _) => self.deepseek_mixer_stage(layer, step),
            _ => anyhow::bail!(
                "mixer {mixer:?} has no bound layer runtime; Q38 setup is not complete"
            ),
        }
    }

    pub fn run_decode_step(&mut self, trace: Option<Target<'_>>) -> Result<()> {
        if self.recurrent.is_some() {
            self.run_recurrent(1, true, trace)?;
            return self.work.advance(&self.ops, &self.device.stream);
        }
        self.ensure_decode()?;
        self.forward(1, true, true, trace)?;
        if self.qwen.is_some() {
            return self.qwen_advance();
        }
        self.work.advance(&self.ops, &self.device.stream)
    }

    pub fn run_operator(
        &mut self,
        op: crate::backend::Operator,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        self.prepare_step(step)?;
        ensure!(
            i < self.config.num_hidden_layers,
            "backend operator layer {i} is outside model"
        );
        if self.qwen.is_some() {
            return self.qwen_operator(op, i, step, trace);
        }
        #[cfg(feature = "cuda")]
        if self.model_spec.architecture == crate::model_spec::Architecture::DeepseekV41 {
            ensure!(
                trace.is_none(),
                "DeepSeek-V4.1 operators record no trace tensors; --dump and --trace-layer are unsupported"
            );
            return self.deepseek_operator(op, i, step);
        }
        let is_tree = matches!(step.mask, crate::backend::Mask::Tree { .. });
        if is_tree && matches!(op, crate::backend::Operator::RopeKv) {
            let tree = self
                .spec
                .as_ref()
                .context("tree operator requires draft State")?
                .tree
                .as_ref()
                .context("tree State is not enabled")?;
            let kv = &mut self.state.kv[i];
            return tree.rope_kv(&mut self.work.qkv, &mut kv.k, &mut kv.v);
        }
        let rows = if is_tree {
            self.spec_budget.rows()
        } else {
            step.rows
        };
        #[cfg(feature = "cuda")]
        let linear_rows = rows;
        #[cfg(feature = "rocm")]
        let linear_rows = bucketed_prefill_rows(rows as u32, self.workspace_rows as u32)? as usize;
        let trace_rows = step.rows;
        let use_token = matches!(step.input, crate::backend::Input::Token);
        let head = step.head == Head::Last;
        let c = &self.config;
        let w = &mut self.work;
        let s = &self.device.stream;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let nr = rows as i32;
        let hd = h as i32;
        let fd = f as i32;
        let qh = c.num_attention_heads as i32;
        let kh = c.num_key_value_heads as i32;
        let dim = c.head_dim as i32;
        let cap = self.capacity as i32;
        let layer = &self.weights.layers[i];
        match op {
            crate::backend::Operator::Embed => {
                // Fused gather + layer-0 RMSNorm; Embed immediately precedes layer 0's Norm.
                if is_tree {
                    {
                        let tree = self
                            .spec
                            .as_ref()
                            .context("tree embed requires draft State")?
                            .tree
                            .as_ref()
                            .context("tree State is not enabled")?;
                        tree.prepare_embed_norm(
                            &w.position,
                            self.weights.embedding(),
                            layer
                                .input_norm
                                .as_ref()
                                .context("weighted input norm missing")?,
                            super::tree::EmbedOut {
                                ids: &mut w.ids,
                                x: &mut w.x,
                                y: &mut w.n,
                            },
                            c,
                        )?;
                    }
                } else {
                    let ids = if use_token { &w.token } else { &w.ids };
                    unsafe {
                        s.launch_builder(&self.ops.embed_rmsnorm)
                            .arg(self.weights.embedding())
                            .arg(ids)
                            .arg(&mut w.x)
                            .arg(
                                layer
                                    .input_norm
                                    .as_ref()
                                    .context("weighted input norm missing")?,
                            )
                            .arg(&mut w.n)
                            .arg(&nr)
                            .arg(&hd)
                            .arg(&c.rms_norm_eps)
                            .launch(grid(rows, 256))?;
                    }
                }
                trace_value(trace, s, "embed", &w.x, trace_rows, h)?;
            }
            crate::backend::Operator::Norm => {
                // The layer-0 norm runs in Embed; Norm has no work on other layers.
            }
            crate::backend::Operator::Qkv => {
                self.blas
                    .linear(
                        &layer.qkv,
                        &w.n.slice(..linear_rows * h),
                        &mut w.qkv.slice_mut(..linear_rows * c.qkv_dim()),
                        linear_rows,
                        c.qkv_dim(),
                        h,
                    )
                    .with_context(|| format!("layer {i} qkv"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.norm1", &w.n, trace_rows, h)?;
                    trace_value(trace, s, "debug.qkv", &w.qkv, trace_rows, c.qkv_dim())?;
                }
            }
            crate::backend::Operator::RopeKv => {
                unsafe {
                    // Fused rotation + KV write over rows x heads x (head_dim/2) element pairs.
                    let kv = &mut self.state.kv[i];
                    s.launch_builder(&self.ops.rope_kv)
                        .arg(&mut w.qkv)
                        .arg(&mut kv.k)
                        .arg(&mut kv.v)
                        .arg(&w.position)
                        .arg(&nr)
                        .arg(&qh)
                        .arg(&kh)
                        .arg(&dim)
                        .arg(&c.rope_theta)
                        .arg(&cap)
                        .launch(flat(
                            rows * (c.num_attention_heads + 2 * c.num_key_value_heads)
                                * (c.head_dim / 2),
                        ))?;
                }
            }
            crate::backend::Operator::Attention => {
                self.run_attention(i, step)?;
            }
            crate::backend::Operator::Output => {
                self.blas
                    .linear(
                        &layer.o,
                        &w.attn.slice(..linear_rows * h),
                        &mut w.out.slice_mut(..linear_rows * h),
                        linear_rows,
                        h,
                        h,
                    )
                    .with_context(|| format!("layer {i} output projection"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.rope", &w.qkv, trace_rows, c.qkv_dim())?;
                    trace_value(trace, s, "debug.attention", &w.attn, trace_rows, h)?;
                    trace_value(trace, s, "debug.o", &w.out, trace_rows, h)?;
                }
            }
            crate::backend::Operator::PostNorm => {
                w.add_norm(
                    &self.ops,
                    s,
                    layer
                        .post_norm
                        .as_ref()
                        .context("weighted post norm missing")?,
                    rows,
                    c,
                )?;
            }
            crate::backend::Operator::GateUp => {
                self.blas
                    .linear(
                        &layer.gu,
                        &w.n.slice(..linear_rows * h),
                        &mut w.gu.slice_mut(..linear_rows * 2 * f),
                        linear_rows,
                        2 * f,
                        h,
                    )
                    .with_context(|| format!("layer {i} gate/up"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.norm2", &w.n, trace_rows, h)?;
                    trace_value(trace, s, "debug.gu", &w.gu, trace_rows, 2 * f)?;
                }
            }
            crate::backend::Operator::Activate => unsafe {
                s.launch_builder(&self.ops.act)
                    .arg(&w.gu)
                    .arg(&mut w.act)
                    .arg(&nr)
                    .arg(&fd)
                    .arg(&(self.ops.swiglu_rounding() as i32))
                    .launch(flat(rows * f))?;
            },
            crate::backend::Operator::Down => {
                self.blas
                    .linear(
                        &layer.down,
                        &w.act.slice(..linear_rows * f),
                        &mut w.out.slice_mut(..linear_rows * h),
                        linear_rows,
                        h,
                        f,
                    )
                    .with_context(|| format!("layer {i} down"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.act", &w.act, trace_rows, f)?;
                    trace_value(trace, s, "debug.down", &w.out, trace_rows, h)?;
                }
            }
            crate::backend::Operator::NextNorm => {
                let weight = if let Some(next) = self.weights.layers.get(i + 1) {
                    next.input_norm
                        .as_ref()
                        .context("weighted input norm missing")?
                } else {
                    self.weights
                        .norm
                        .as_ref()
                        .context("weighted final norm missing")?
                };
                {
                    if let Some(binding) = self.hidden_capture.as_mut()
                        && let Some(slot) = binding.layer_slots[i]
                    {
                        use super::contracts::TensorMut;
                        use super::ops::{
                            ResidualRmsCapture, ResidualRmsInput, RmsMode, RmsOptions,
                        };
                        let taps = binding.taps;
                        let mut x = TensorMut(super::contracts::view(
                            &w.x,
                            s,
                            0,
                            [rows as u32, 1, h as u32],
                            [h as u64, h as u64, 1],
                        )?);
                        let mut y = TensorMut(super::contracts::view(
                            &w.n,
                            s,
                            0,
                            [rows as u32, 1, h as u32],
                            [h as u64, h as u64, 1],
                        )?);
                        let mut capture = TensorMut(super::contracts::view(
                            &binding.values,
                            s,
                            0,
                            [rows as u32, taps as u32, h as u32],
                            [(taps * h) as u64, h as u64, 1],
                        )?);
                        self.ops.add_rmsnorm_capture(
                            s,
                            None,
                            ResidualRmsCapture {
                                input: ResidualRmsInput {
                                    x: &mut x,
                                    residual: super::contracts::view(
                                        &w.out,
                                        s,
                                        0,
                                        [rows as u32, 1, h as u32],
                                        [h as u64, h as u64, 1],
                                    )?,
                                    weight: super::contracts::view(weight, s, 0, [h as u32], [1])?,
                                },
                                capture: &mut capture,
                                slot,
                            },
                            &mut y,
                            RmsOptions {
                                mode: RmsMode::Standard,
                                eps: c.rms_norm_eps,
                            },
                        )?;
                    } else {
                        w.add_norm(&self.ops, s, weight, rows, c)?;
                    }
                }
                trace_value(trace, s, &format!("layer.{i}"), &w.x, trace_rows, h)?;
            }
            crate::backend::Operator::Head => {
                if step.head == Head::All {
                    let spec = self
                        .spec
                        .as_mut()
                        .context("all-row logits require enabled DSpark state")?;
                    ensure!(
                        rows <= spec.predictions.len(),
                        "verification rows exceed allocated predictions"
                    );
                    self.blas.linear(
                        &self.weights.head,
                        &w.n.slice(..rows * h),
                        &mut spec.logits.slice_mut(..rows * c.vocab_size),
                        rows,
                        c.vocab_size,
                        h,
                    )?;
                    let valid_rows = if is_tree {
                        spec.tree
                            .as_ref()
                            .context("tree State is not enabled")?
                            .valid_rows()
                            .slice(..1)
                    } else {
                        spec.row_counts.slice(rows - 1..rows)
                    };
                    unsafe {
                        s.launch_builder(&self.ops.argmax_rows)
                            .arg(&spec.logits)
                            .arg(&mut spec.predictions)
                            .arg(&(c.vocab_size as i32))
                            .arg(&valid_rows)
                            .launch(grid(rows, 1024))?;
                    }
                    spec.active_rows = step.rows;
                    return Ok(());
                }
                if !head && trace.is_none() {
                    return Ok(());
                }
                if let Some(t) = trace.as_mut() {
                    self.snapshot(
                        t.trace,
                        format!("{}.norm", t.prefix),
                        &self.work.n,
                        t.skip,
                        rows,
                        h,
                    )?;
                }
                if !head {
                    return Ok(());
                }
                let w = &mut self.work;
                let input = w.n.slice((rows - 1) * h..rows * h);
                self.blas
                    .linear(
                        &self.weights.head,
                        &input,
                        &mut w.logits.slice_mut(..c.vocab_size),
                        1,
                        c.vocab_size,
                        h,
                    )
                    .context("lm_head")?;
                let logits = w.logits.slice(..c.vocab_size);
                let vocab = c.vocab_size as i32;
                unsafe {
                    s.launch_builder(&self.ops.argmax)
                        .arg(&logits)
                        .arg(&mut w.token)
                        .arg(&vocab)
                        .launch(grid(1, 1024))?;
                }
                if let Some(t) = trace.as_mut() {
                    self.snapshot(
                        t.trace,
                        format!("{}.logits", t.prefix),
                        &self.work.logits,
                        0,
                        1,
                        c.vocab_size,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.check_moe_reset()?;
        self.poisoned = true;
        {
            if let Some(slots) = self.moe_slots.as_mut()
                && slots.request_open
            {
                slots.end_request()?;
            }
            #[cfg(feature = "cuda")]
            if let Some(chain) = self.mtp.as_mut() {
                chain.invalidate_verify();
            }
            #[cfg(feature = "cuda")]
            {
                self.mtp_epoch = self
                    .mtp_epoch
                    .checked_add(1)
                    .context("MTP state epoch exhausted")?;
            }
            self.device.stream.synchronize()?;
            if let Some(decode) = self.moe.as_mut() {
                decode.reset()?;
            }
            self.images = None;
            if let Some(recurrent) = self.recurrent.as_mut() {
                recurrent.reset(&self.device.stream, self.capacity)?;
            }
            self.reset_qwen_positions()?;
        }
        self.state.reset(&self.device.stream)?;
        #[cfg(feature = "cuda")]
        if self.deepseek_ffn.is_some()
            && let Some(slots) = self.moe_slots.as_mut()
        {
            slots.begin_staging_request()?;
        }
        #[cfg(feature = "cuda")]
        if let Some(engram) = &mut self.deepseek_engram {
            engram.service.reset();
            engram.prepared = None;
        }
        #[cfg(test)]
        if let Some(calls) = self.qwen.as_ref().and_then(|q| q.glm_calls.as_ref()) {
            calls.borrow_mut().fill([0; 10]);
        }
        if let Some(spec) = self.spec.as_mut() {
            spec.active_rows = 0;
            spec.verify_schedule = spec.cold_verify_schedule;
        }
        self.position = 0;
        self.logits_cache.invalidate();
        self.committed_prefix_ids.clear();
        self.pending_token = None;
        self.resident_ids_len = 0;
        self.last_input_kind = None;
        self.device
            .stream
            .memcpy_htod(&[0i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[1i32], &mut self.work.length)?;
        self.poisoned = false;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn save_text_prefix(
        &mut self,
        request: TextPrefixSaveRequest<'_>,
    ) -> Result<TextPrefixSaveOutput> {
        ensure!(
            self.model_spec.architecture != crate::model_spec::Architecture::Glm5Next,
            "GLM DSA prefix persistence is not bound"
        );
        let started = Instant::now();
        self.closure.assert_ready()?;
        ensure!(!self.poisoned, "cannot save a poisoned engine prefix");
        ensure!(self.qwen.is_some(), "text prefix save requires Qwen3.8");
        ensure!(
            self.images.is_none(),
            "multimodal prefix save is unsupported"
        );
        ensure!(
            self.spec.as_ref().is_none_or(|spec| spec.active_rows == 0),
            "cannot save during a speculative transaction"
        );
        let prefix_identity = self
            .prefix_identity
            .clone()
            .context("prefix save requires --prefix-cache at engine load")?;
        let identity_load_ms = self.prefix_identity_load_ms;
        let prefix = request.prefix_ids.len();
        ensure!(
            prefix > 0 && prefix == self.position,
            "prefix boundary mismatch"
        );
        ensure!(
            self.committed_prefix_ids.len() == prefix,
            "engine committed token count differs from requested prefix"
        );
        ensure!(
            self.committed_prefix_ids.as_slice() == request.prefix_ids,
            "prefix token ids differ from the engine-committed sequence"
        );
        ensure!(
            request
                .prefix_ids
                .iter()
                .all(|token| (*token as usize) < self.config.vocab_size),
            "prefix token outside vocabulary"
        );
        self.device.stream.synchronize()?;
        let sequence = self
            .state
            .sequence
            .as_ref()
            .context("Qwen sequence metadata missing")?;
        let sequence = self.device.stream.clone_dtoh(sequence)?;
        ensure!(
            sequence
                == [
                    prefix as i32,
                    1,
                    prefix as i32 + 1,
                    self.capacity as i32,
                    1,
                    0,
                    0,
                    0
                ],
            "device sequence metadata differs from committed text prefix"
        );
        let last_token = request
            .prefix_ids
            .last()
            .copied()
            .context("prefix token ids are empty")?;
        let pending_next_token = self.token()?;
        self.pending_token = Some(pending_next_token);
        ensure!(
            (pending_next_token as usize) < self.config.vocab_size,
            "pending prefix token outside vocabulary"
        );
        let last_input_kind = self
            .last_input_kind
            .context("engine has no committed text input source")?;
        match last_input_kind {
            PrefixLastInputKind::PrefillChunk => {
                let resident_ids = self.device.stream.clone_dtoh(&self.work.ids)?;
                ensure!(
                    self.resident_ids_len > 0,
                    "engine has no resident prefill ids to audit"
                );
                let last_input_index = self.resident_ids_len - 1;
                ensure!(
                    resident_ids.get(last_input_index).copied() == Some(last_token),
                    "last prefix input differs from the resident last submitted chunk"
                );
            }
            PrefixLastInputKind::DecodeToken => {
                ensure!(
                    self.committed_prefix_ids.last().copied() == Some(last_token),
                    "last prefix input differs from the committed decode token"
                );
            }
        }
        let geometry = PrefixGeometry::new(
            &self.model_spec,
            self.capacity,
            prefix,
            request.block_tokens,
            &prefix_identity,
        )?;
        let boundary = PrefixBoundary {
            prefix_tokens: prefix,
            last_token,
            last_input_kind,
            pending_next_token,
            token_ids_sha256: token_ids_sha256(request.prefix_ids)?,
            rope_context_sha256: text_rope_context_sha256(self.capacity)?,
        };
        let builder = PrefixBuilder::new(geometry, boundary)?;
        let mut ring = pinned_transfer_ring(&self.device.ctx, request.transfer_bytes)?;
        let result = self.save_text_prefix_into(builder, &mut ring);
        if result.is_err() {
            self.poisoned = true;
            self.device.stream.synchronize()?;
        }
        let snapshot = result?;
        let receipt = PrefixSaveReceipt {
            prefix_tokens: prefix,
            state_root: snapshot.manifest.state_root.clone(),
            total_state_bytes: snapshot.manifest.total_state_bytes,
            transfer_bytes: request.transfer_bytes,
            identity_load_ms,
            identity_hashed_bytes: prefix_identity.hashed_weight_bytes,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        Ok(TextPrefixSaveOutput { snapshot, receipt })
    }

    #[cfg(feature = "cuda")]
    fn save_text_prefix_into(
        &mut self,
        mut builder: PrefixBuilder,
        ring: &mut PinnedHostSlice<u8>,
    ) -> Result<PrefixSnapshot> {
        let stream = self.device.stream.clone();
        let capacity = self.capacity;
        let dim = self.model_spec.head_dim as usize;
        let block_tokens = builder.geometry.block_tokens;
        let prefix_tokens = builder.geometry.prefix_tokens;
        for (physical, layer) in self.model_spec.layers.iter().enumerate() {
            match layer.mixer {
                crate::model_spec::Mixer::FullAttention => {
                    let state_index = layer.state_index as usize;
                    let kv = &mut self.state.kv[state_index];
                    if kv.compact.is_some() {
                        let c = kv.compact.as_ref().context("compact cache missing")?;
                        ensure!(
                            stream.clone_dtoh(&c.cursor)?
                                == [
                                    prefix_tokens as i32,
                                    crate::kv::sealed_until(prefix_tokens) as i32,
                                    0,
                                    0
                                ],
                            "prefix cache has an uncommitted compact cursor"
                        );
                        for block in 0..prefix_tokens.div_ceil(block_tokens) {
                            let start = block * block_tokens;
                            let end = prefix_tokens.min(start + block_tokens);
                            let payload = kv.compact_prefix_block(
                                start..end,
                                prefix_tokens,
                                &stream,
                                ring,
                                None,
                            )?;
                            builder.add(
                                PrefixRecordKind::AttentionKvBlock,
                                physical,
                                block,
                                payload,
                            )?;
                        }
                        continue;
                    }
                    let prefix_bytes = prefix_tokens * dim * 2;
                    let mut resident = [
                        Vec::with_capacity(builder.geometry.kv_heads * prefix_bytes),
                        Vec::with_capacity(builder.geometry.kv_heads * prefix_bytes),
                    ];
                    for tensor in [&kv.k, &kv.v] {
                        let tensor_index = usize::from(std::ptr::eq(tensor, &kv.v));
                        for head in 0..builder.geometry.kv_heads {
                            let source_element_offset = head * capacity * dim;
                            let bytes = dtoh_pinned_chunks(
                                &stream,
                                tensor,
                                source_element_offset,
                                prefix_bytes,
                                ring,
                            )?;
                            resident[tensor_index].extend_from_slice(&bytes);
                        }
                    }
                    for block in 0..prefix_tokens.div_ceil(block_tokens) {
                        let token_start = block * block_tokens;
                        let valid = (prefix_tokens - token_start).min(block_tokens);
                        let segment_bytes = valid * dim * 2;
                        let mut payload =
                            Vec::with_capacity(builder.geometry.attention_block_bytes(valid)?);
                        for tensor in &resident {
                            for head in 0..builder.geometry.kv_heads {
                                let source_offset = head * prefix_bytes + token_start * dim * 2;
                                payload.extend_from_slice(
                                    &tensor[source_offset..source_offset + segment_bytes],
                                );
                            }
                        }
                        builder.add(
                            PrefixRecordKind::AttentionKvBlock,
                            physical,
                            block,
                            payload,
                        )?;
                    }
                    drop(resident);
                }
                crate::model_spec::Mixer::Gdn => {
                    let state_index = layer.state_index as usize;
                    let state = &self.state.gdn[state_index];
                    let s = dtoh_pinned_chunks(
                        &stream,
                        &state.s,
                        0,
                        builder.geometry.gdn_s_bytes,
                        ring,
                    )?;
                    builder.add(PrefixRecordKind::GdnS, physical, 0, s)?;
                    let history = dtoh_pinned_chunks(
                        &stream,
                        &state.history,
                        0,
                        builder.geometry.gdn_history_bytes,
                        ring,
                    )?;
                    builder.add(PrefixRecordKind::GdnHistory, physical, 0, history)?;
                }
            }
        }
        builder.finish()
    }

    #[cfg(feature = "cuda")]
    pub fn restore_text_prefix(
        &mut self,
        snapshot: &PrefixSnapshot,
        intent: TextPrefixRestoreIntent<'_>,
    ) -> Result<PrefixRestoreReceipt> {
        ensure!(
            self.model_spec.architecture != crate::model_spec::Architecture::Glm5Next,
            "GLM DSA prefix persistence is not bound"
        );
        let started = Instant::now();
        validate_transfer_bytes(intent.transfer_bytes)?;
        snapshot.verify_prefix_and_pending_within_vocab(
            intent.prefix_ids,
            intent.pending_next_token,
            self.config.vocab_size,
        )?;
        self.closure.assert_ready()?;
        ensure!(!self.poisoned, "cannot restore into a poisoned engine");
        ensure!(self.qwen.is_some(), "text prefix restore requires Qwen3.8");
        ensure!(
            self.images.is_none(),
            "multimodal prefix restore is unsupported"
        );
        ensure!(
            self.spec.as_ref().is_none_or(|spec| spec.active_rows == 0),
            "cannot restore during a speculative transaction"
        );
        let prefix_identity = self
            .prefix_identity
            .clone()
            .context("prefix restore requires --prefix-cache at engine load")?;
        let identity_load_ms = self.prefix_identity_load_ms;
        ensure!(
            snapshot.manifest.identity == prefix_identity,
            "prefix snapshot identity differs from the loaded engine"
        );
        let geometry = PrefixGeometry::new(
            &self.model_spec,
            self.capacity,
            snapshot.manifest.boundary.prefix_tokens,
            snapshot.manifest.block_tokens,
            &prefix_identity,
        )?;
        geometry.verify_manifest(&snapshot.manifest)?;
        ensure!(
            snapshot.manifest.boundary.rope_context_sha256
                == text_rope_context_sha256(self.capacity)?,
            "prefix rope context differs from this engine"
        );
        let ring = pinned_transfer_ring(&self.device.ctx, intent.transfer_bytes)?;
        self.device.stream.synchronize()?;
        self.logits_cache.invalidate();
        self.poisoned = true;
        if let Some(chain) = self.mtp.as_mut() {
            chain.invalidate_verify();
        }
        let result = self.restore_text_prefix_from(snapshot, intent.prefix_ids, ring);
        if let Err(error) = result {
            self.device.stream.synchronize()?;
            return Err(error);
        }
        self.mtp_epoch = self
            .mtp_epoch
            .checked_add(1)
            .context("MTP state epoch exhausted")?;
        if let Some(spec) = self.spec.as_mut() {
            spec.active_rows = 0;
            spec.verify_schedule = spec.cold_verify_schedule;
        }
        self.poisoned = false;
        // It remains invalid until the restored pending token is actually run.
        Ok(PrefixRestoreReceipt {
            prefix_tokens: snapshot.manifest.boundary.prefix_tokens,
            state_root: snapshot.manifest.state_root.clone(),
            total_state_bytes: snapshot.manifest.total_state_bytes,
            transfer_bytes: intent.transfer_bytes,
            identity_load_ms,
            identity_hashed_bytes: prefix_identity.hashed_weight_bytes,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[cfg(feature = "cuda")]
    fn restore_text_prefix_from(
        &mut self,
        snapshot: &PrefixSnapshot,
        prefix_ids: &[u32],
        mut ring: PinnedHostSlice<u8>,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        let capacity = self.capacity;
        let dim = self.model_spec.head_dim as usize;
        let compact =
            crate::kv::KvFormat::from_tag(&snapshot.manifest.identity.attention_kv_dtype)?
                != crate::kv::KvFormat::Bf16;
        let attention_tensors = if compact {
            Vec::new()
        } else {
            prefix_attention_host_tensors(snapshot, dim)?
        };
        if compact {
            let prefix = snapshot.manifest.boundary.prefix_tokens;
            for (receipt, payload) in snapshot.records() {
                if receipt.key.kind != PrefixRecordKind::AttentionKvBlock {
                    continue;
                }
                let layer = &self.model_spec.layers[receipt.key.layer];
                self.state.kv[layer.state_index as usize].compact_prefix_block(
                    receipt.key.token_start..receipt.key.token_end,
                    prefix,
                    &stream,
                    &mut ring,
                    Some(payload),
                )?;
            }
        }
        for (physical, k_heads, v_heads) in attention_tensors {
            let layer = self
                .model_spec
                .layers
                .get(physical)
                .with_context(|| format!("prefix layer {physical} outside model"))?;
            ensure!(
                matches!(layer.mixer, crate::model_spec::Mixer::FullAttention),
                "prefix K/V record maps to a non-attention layer"
            );
            let kv = &mut self.state.kv[layer.state_index as usize];
            for (head, bytes) in k_heads.iter().enumerate() {
                htod_pinned_chunks(&stream, &mut kv.k, head * capacity * dim, bytes, &mut ring)?;
            }
            for (head, bytes) in v_heads.iter().enumerate() {
                htod_pinned_chunks(&stream, &mut kv.v, head * capacity * dim, bytes, &mut ring)?;
            }
        }
        for (receipt, payload) in snapshot.records() {
            if receipt.key.kind == PrefixRecordKind::AttentionKvBlock {
                continue;
            }
            let physical = receipt.key.layer;
            let layer = self
                .model_spec
                .layers
                .get(physical)
                .with_context(|| format!("prefix layer {physical} outside model"))?;
            let state_index = layer.state_index as usize;
            match receipt.key.kind {
                PrefixRecordKind::AttentionKvBlock => continue,
                PrefixRecordKind::GdnS => {
                    ensure!(
                        matches!(layer.mixer, crate::model_spec::Mixer::Gdn),
                        "prefix S record maps to a non-GDN layer"
                    );
                    htod_pinned_chunks(
                        &stream,
                        &mut self.state.gdn[state_index].s,
                        0,
                        payload,
                        &mut ring,
                    )?;
                }
                PrefixRecordKind::GdnHistory => {
                    ensure!(
                        matches!(layer.mixer, crate::model_spec::Mixer::Gdn),
                        "prefix history record maps to a non-GDN layer"
                    );
                    htod_pinned_chunks(
                        &stream,
                        &mut self.state.gdn[state_index].history,
                        0,
                        payload,
                        &mut ring,
                    )?;
                }
            }
        }
        let prefix = snapshot.manifest.boundary.prefix_tokens;
        let last_input_kind = snapshot.manifest.boundary.last_input_kind;
        stream.synchronize()?;
        self.state
            .set_sequence(&stream, prefix as u32, 1, self.workspace_rows as u32)?;
        self.reset_qwen_positions()?;
        stream.memcpy_htod(&[prefix as i32], &mut self.work.position)?;
        stream.memcpy_htod(&[prefix as i32 + 1], &mut self.work.length)?;
        let resident_len = match last_input_kind {
            PrefixLastInputKind::PrefillChunk => {
                let resident_len = prefix.min(self.chunk);
                let resident_ids_start = prefix - resident_len;
                stream.memcpy_htod(
                    &prefix_ids[resident_ids_start..],
                    &mut self.work.ids.slice_mut(..resident_len),
                )?;
                resident_len
            }
            // The committed decode input is host-audited; the next prefill
            // overwrites work.ids before it can be consumed.
            PrefixLastInputKind::DecodeToken => self.resident_ids_len.min(self.chunk),
        };
        self.set_token(snapshot.manifest.boundary.pending_next_token)?;
        self.position = prefix;
        for kv in &self.state.kv {
            if let Some(c) = &kv.compact {
                c.cursor_mode(
                    c.view(&kv.k, &kv.v, &stream),
                    self.state.seq_ref(&stream)?,
                    2,
                    &stream,
                )?;
            }
        }
        self.committed_prefix_ids = prefix_ids.to_vec();
        self.resident_ids_len = resident_len;
        self.last_input_kind = Some(last_input_kind);
        stream.synchronize()?;
        Ok(())
    }

    pub fn prefill(&mut self, ids: &[u32], trace: Option<&mut Trace>) -> Result<u32> {
        self.prefill_inner(ids, trace, ids.len())
    }

    pub fn prefill_tail(&mut self, ids: &[u32], trace: &mut Trace, keep: usize) -> Result<u32> {
        ensure!(
            keep > 0 && keep <= ids.len(),
            "invalid hidden trace tail length {keep}"
        );
        self.prefill_inner(ids, Some(trace), keep)
    }

    fn prefill_inner(
        &mut self,
        ids: &[u32],
        trace: Option<&mut Trace>,
        keep: usize,
    ) -> Result<u32> {
        if self.recurrent.is_some() {
            return self.prefill_recurrent(ids, ids.len(), trace, keep);
        }
        self.prefill_chunks(ids, trace, keep, |_, _| Ok(()))
    }

    fn prefill_chunks(
        &mut self,
        ids: &[u32],
        mut trace: Option<&mut Trace>,
        keep: usize,
        mut after_chunk: impl FnMut(&mut Self, &[u32]) -> Result<()>,
    ) -> Result<u32> {
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(!ids.is_empty(), "empty input token ids");
        let all_layers = trace.as_ref().is_some_and(|t| t.all_layers);
        ensure!(
            !all_layers
                || (self.qwen.is_some()
                    && self.position == 0
                    && ids.len() <= self.chunk
                    && keep == ids.len()),
            "all-layer trace requires one complete Qwen prefill from position zero"
        );
        ensure!(
            self.position + ids.len() <= self.capacity,
            "prefill exceeds KV capacity"
        );
        ensure!(
            ids.iter().all(|&x| (x as usize) < self.config.vocab_size),
            "input token id outside vocabulary"
        );
        if self.model_spec.moe.is_some()
            && !matches!(
                self.model_spec.architecture,
                crate::model_spec::Architecture::DeepseekV41
                    | crate::model_spec::Architecture::Glm5Next
            )
        {
            return self.prefill_moe_chunks(ids, trace, keep, &mut after_chunk);
        }
        #[cfg(feature = "cuda")]
        if ids.len() > self.chunk
            && self.images.is_none()
            && self.qwen.as_ref().is_some_and(|q| q.prompt.is_some())
        {
            return self.prefill_dense_layers(ids, trace, keep, &mut after_chunk);
        }
        let trace_start = self.position + ids.len() - keep;
        let end = self.position + ids.len();
        let mut part = 0usize;
        for chunk in ids.chunks(self.chunk) {
            #[cfg(feature = "cuda")]
            self.prepare_deepseek_engram(chunk, None)?;
            let skip = trace_start.saturating_sub(self.position).min(chunk.len());
            let selected = trace.is_some() && skip < chunk.len();
            self.prepare_prefill_rows(chunk.len())?;
            if self.qwen.is_some() {
                self.state.set_sequence(
                    &self.device.stream,
                    self.position as u32,
                    chunk.len() as u32,
                    self.workspace_rows as u32,
                )?;
            }
            self.device
                .stream
                .memcpy_htod(chunk, &mut self.work.ids.slice_mut(..chunk.len()))?;
            self.resident_ids_len = chunk.len();
            self.last_input_kind = Some(PrefixLastInputKind::PrefillChunk);
            self.device
                .stream
                .memcpy_htod(&[self.position as i32], &mut self.work.position)?;
            let prefix = if all_layers {
                "layer".to_owned()
            } else {
                format!("prefill.{part}")
            };
            self.forward(
                chunk.len(),
                false,
                self.position + chunk.len() == end,
                if selected {
                    trace.as_deref_mut().map(|t| Target {
                        trace: t,
                        prefix: prefix.as_str(),
                        skip,
                    })
                } else {
                    None
                },
            )?;
            after_chunk(self, chunk)?;
            if selected {
                part += 1;
            }
            if self.spec.is_some() {
                self.inject_capture(chunk.len(), self.position, None)?;
            }
            self.committed_prefix_ids.extend_from_slice(chunk);
            self.position += chunk.len();
        }
        self.device
            .stream
            .memcpy_htod(&[self.position as i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[(self.position + 1) as i32], &mut self.work.length)?;
        if self.qwen.is_some() && self.position < self.capacity {
            self.state.set_sequence(
                &self.device.stream,
                self.position as u32,
                1,
                self.workspace_rows as u32,
            )?;
        }
        let pending_next_token = self.token()?;
        self.pending_token = Some(pending_next_token);
        Ok(pending_next_token)
    }

    fn advance(&mut self, graph: bool, trace: Option<(&mut Trace, &str)>) -> Result<()> {
        #[cfg(feature = "cuda")]
        ensure!(
            !graph || self.spec.is_none(),
            "graph decode is not declared for draft-enabled engines"
        );

        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(self.position < self.capacity, "decode exceeds KV capacity");
        ensure!(
            !graph || trace.is_none(),
            "graph tracing is unsupported; use eager tracing"
        );
        let committed_token = match self.pending_token {
            Some(token) => token,
            None => self.token()?,
        };
        #[cfg(feature = "cuda")]
        self.prepare_deepseek_engram(&[committed_token], None)?;
        if graph {
            {
                let result = if self.moe.is_some() {
                    self.run_moe_graph(1)
                } else {
                    #[cfg(feature = "cuda")]
                    let launched = self
                        .graph
                        .as_ref()
                        .context("decode graph is not captured")?
                        .launch()
                        .map_err(Into::into);
                    #[cfg(feature = "rocm")]
                    let launched = self
                        .graph
                        .as_ref()
                        .context("decode graph is not captured")?
                        .launch();
                    launched
                };
                if let Err(error) = result {
                    self.poisoned = true;
                    return Err(error);
                }
                self.logits_cache.mark_valid();
            }
        } else {
            self.run_decode_step(trace.map(|(trace, prefix)| Target {
                trace,
                prefix,
                skip: 0,
            }))?;
        }
        self.committed_prefix_ids.push(committed_token);
        self.pending_token = None;
        self.last_input_kind = Some(PrefixLastInputKind::DecodeToken);
        self.position += 1;
        Ok(())
    }

    pub fn decode(&mut self, graph: bool, trace: Option<(&mut Trace, &str)>) -> Result<u32> {
        self.advance(graph, trace)?;
        let pending_next_token = self.token()?;
        self.pending_token = Some(pending_next_token);
        Ok(pending_next_token)
    }

    pub fn step(&mut self, graph: bool) -> Result<()> {
        self.advance(graph, None)
    }

    pub fn rewind(&mut self, position: usize, token: u32) -> Result<()> {
        ensure!(
            self.recurrent.is_none(),
            "LoopVL rewind requires rebuilding the bidirectional prefix"
        );
        ensure!(
            self.qwen.is_none(),
            "Q38 rewind requires a committed GDN checkpoint; scalar rewind is unsupported"
        );
        ensure!(
            position <= self.position,
            "rewind position exceeds current cache length"
        );
        self.device.stream.synchronize()?;
        self.device
            .stream
            .memcpy_htod(&[position as i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[(position + 1) as i32], &mut self.work.length)?;
        self.committed_prefix_ids.truncate(position);
        self.set_token(token)?;
        self.position = position;
        self.device.stream.synchronize()?;
        Ok(())
    }

    pub fn weight_bytes(&self) -> usize {
        let extra = self.qwen.as_ref().map_or(0, qwen::Runtime::vision_bytes);
        self.weights.bytes + extra
    }

    pub fn decode_weight_bytes(&self) -> usize {
        if self.recurrent.is_some() {
            return self
                .weights
                .layers
                .iter()
                .map(|l| (l.qkv.len() + l.o.len() + l.gu.len() + l.down.len()) * 2)
                .sum::<usize>()
                * 4
                + (self.weights.head.len() + 2 * self.config.hidden_size) * 2;
        }
        self.weights.bytes - self.weights.embedding().len() * size_of::<bf16>()
            + self.config.hidden_size * size_of::<bf16>()
    }

    #[cfg(feature = "cuda")]
    pub fn kv_read_bytes(&self, tokens: usize) -> usize {
        self.state
            .kv
            .iter()
            .map(|kv| {
                let Some(_) = &kv.compact else {
                    return tokens * self.config.kv_dim() * 4;
                };
                let sealed = crate::kv::sealed_until(tokens);
                let coded = sealed.saturating_sub(crate::kv::SINK_TOKENS);
                let stats = sealed / crate::kv::GROUP_TOKENS * 4 * 256 * 2 * 4 + coded * 4 * 2 * 4;
                coded * 4 * 256 * 2 + (tokens - coded) * 4 * 256 * 4 + stats + 16
            })
            .sum()
    }

    /// DeepSeek-V4.1 and GLM-5.3 decode eagerly.
    pub fn graph_decode(&self) -> bool {
        if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
            return false;
        }
        #[cfg(feature = "cuda")]
        if self.deepseek_mixer.is_some() {
            return false;
        }
        true
    }

    pub fn capture(&mut self) -> Result<()> {
        ensure!(
            self.model_spec.architecture != crate::model_spec::Architecture::Glm5Next,
            "GLM storage-only expert graph segments are not bound"
        );
        ensure!(
            self.graph_decode(),
            "DeepSeek-V4.1 decode has no bound graph path"
        );
        if self.moe.is_some() {
            return self.capture_moe();
        }
        ensure!(
            self.spec.is_none(),
            "graph decode is not declared for draft-enabled engines"
        );
        if self.graph.is_some() {
            return Ok(());
        }
        self.closure.capture()?;
        #[cfg(feature = "cuda")]
        if let Some(chain) = self.mtp.as_mut() {
            chain.invalidate_verify();
        }
        self.ensure_decode()?;
        ensure!(
            self.position < self.capacity,
            "graph capture exceeds KV capacity"
        );
        ensure!(!self.poisoned, "capture requires healthy engine state");
        let s = self.device.stream.clone();
        let saved_position = s.clone_dtoh(&self.work.position)?;
        let saved_length = s.clone_dtoh(&self.work.length)?;
        let mut checkpoint = self
            .capture_checkpoint
            .take()
            .context("capture checkpoint missing")?;
        let result = (|| -> Result<CudaGraph> {
            checkpoint.save(&self.state, &self.work.token, self.position, &s, None)?;
            self.run_decode_step(None)?;
            s.synchronize()?;
            checkpoint.restore(&mut self.state, &mut self.work.token, &s)?;
            s.memcpy_htod(&saved_position, &mut self.work.position)?;
            s.memcpy_htod(&saved_length, &mut self.work.length)?;
            s.synchronize()?;
            #[cfg(test)]
            ensure!(
                self.capture_fail_stage != Some(CaptureFault::Begin),
                "injected capture begin failure"
            );
            #[cfg(feature = "cuda")]
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            #[cfg(feature = "rocm")]
            let capture = s.begin_capture()?;
            let launches = self.run_decode_step(None);
            // End capture even when a host-side launch check fails.
            #[cfg(feature = "cuda")]
            let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
            #[cfg(feature = "rocm")]
            let captured = capture.finish().map(Some);
            launches?;
            #[cfg(test)]
            ensure!(
                self.capture_fail_stage != Some(CaptureFault::End),
                "injected capture end failure"
            );
            let graph = captured?.context("empty decode graph")?;
            // end_capture also instantiates; the injection exercises disposal and rollback before publish.
            #[cfg(test)]
            ensure!(
                self.capture_fail_stage != Some(CaptureFault::Instantiate),
                "injected capture instantiate failure"
            );
            graph.upload()?;
            s.synchronize()?;
            Ok(graph)
        })();
        let restored = (|| -> Result<()> {
            checkpoint.restore(&mut self.state, &mut self.work.token, &s)?;
            s.memcpy_htod(&saved_position, &mut self.work.position)?;
            s.memcpy_htod(&saved_length, &mut self.work.length)?;
            s.synchronize()?;
            Ok(())
        })();
        self.capture_checkpoint = Some(checkpoint);
        if let Err(error) = restored {
            self.poisoned = true;
            return Err(error.context(format!(
                "capture state restore failed; capture error: {:?}",
                result.as_ref().err()
            )));
        }
        match result {
            Ok(graph) => self.graph = Some(graph),
            Err(error) => {
                self.poisoned = true;
                return Err(error.context("decode capture transaction failed"));
            }
        }
        Ok(())
    }

    pub fn token(&self) -> Result<u32> {
        Ok(self.device.stream.clone_dtoh(&self.work.token)?[0])
    }

    pub fn set_token(&mut self, token: u32) -> Result<()> {
        ensure!(
            (token as usize) < self.config.vocab_size,
            "token outside vocabulary"
        );
        self.device
            .stream
            .memcpy_htod(&[token], &mut self.work.token)?;
        self.pending_token = Some(token);
        Ok(())
    }

    pub fn decode_logits(&self) -> Result<Vec<f32>> {
        self.logits_cache.read()?;
        if let Some(logits) = &self.logits_f32 {
            return self
                .device
                .stream
                .clone_dtoh(logits)
                .context("read FP32 logits");
        }
        let values = self
            .device
            .stream
            .clone_dtoh(&self.work.logits.slice(..self.config.vocab_size))?;
        Ok(values.into_iter().map(bf16::to_f32).collect())
    }

    pub fn event(&self) -> Result<Event> {
        #[cfg(feature = "cuda")]
        let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        #[cfg(feature = "rocm")]
        let flags = Some(0);
        Ok(Event(self.device.ctx.new_event(flags)?))
    }

    pub fn record(&self, event: &Event) -> Result<()> {
        event
            .0
            .record(&self.device.stream)
            .context("record engine event")
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device
            .stream
            .synchronize()
            .context("synchronize engine stream")
    }

    pub fn sample_buffer(&self, count: usize) -> Result<Logits> {
        Ok(Logits(
            self.device
                .stream
                .alloc_zeros(count * self.config.vocab_size)?,
        ))
    }

    pub fn copy_logits(&self, dst: &mut Logits, step: usize) -> Result<()> {
        self.logits_cache.read()?;
        let width = self.config.vocab_size;
        ensure!(
            (step + 1)
                .checked_mul(width)
                .is_some_and(|n| n <= dst.0.len()),
            "logit sample destination size mismatch"
        );
        self.device.stream.memcpy_dtod(
            &self.work.logits.slice(..width),
            &mut dst.0.slice_mut(step * width..(step + 1) * width),
        )?;
        Ok(())
    }

    pub fn check_samples(&self, src: &Logits) -> Result<usize> {
        let values = self.device.stream.clone_dtoh(&src.0)?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite logits in benchmark samples"
        );
        Ok(values.len())
    }

    #[cfg(feature = "cuda")]
    pub fn profiler_start(&self) -> Result<()> {
        Ok(cudarc::driver::profiler_start()?)
    }

    #[cfg(feature = "cuda")]
    pub fn profiler_stop(&self) -> Result<()> {
        Ok(cudarc::driver::profiler_stop()?)
    }

    pub fn check_logits(&self) -> Result<()> {
        let values = self.decode_logits()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite model logits"
        );
        Ok(())
    }

    pub fn trace_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(
            layer < self.config.num_hidden_layers,
            "trace layer {layer} is outside model"
        );
        self.trace_layer = layer;
        Ok(())
    }

    pub fn decode_linear(&self) -> &'static str {
        if self.recurrent.is_some() {
            return "BF16 skinny decode, FP32 accumulation; fixed BF16 Lt prefill";
        }
        if self.qwen.is_some() {
            return "fp8-storage-gemv/fp32-accum; fixed-bf16-lt";
        }
        self.blas.decode_impl()
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// Device bytes in use right now; the runner reads it while the prefill allocation is live.
    pub fn live_device_bytes(&self) -> Result<u64> {
        let (free, total) = self.device.ctx.mem_get_info()?;
        Ok(u64::try_from(total - free)?)
    }

    pub fn attention_chunk(&self) -> Option<usize> {
        self.decode.as_ref().map(Attention::chunk)
    }

    pub fn attention_plan(&self) -> Option<crate::backend::setup::AttentionPlan> {
        self.decode.as_ref().map(Attention::plan)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Err(error) = self.device.stream.synchronize() {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr().lock(),
                "model stream synchronization failed during drop: {error}"
            );
        }
    }
}

impl Engine {
    fn ensure_verification(&mut self) -> Result<()> {
        let spec = self.spec.as_mut().context("draft state is not enabled")?;
        if self.qwen.is_some() {
            ensure!(
                spec.checkpoint.is_some() && spec.gdn_inputs.len() == self.state.gdn.len(),
                "MoE verification lacks transaction state or recurrent replay inputs"
            );
            return Ok(());
        }
        if spec.verification.is_some() {
            return Ok(());
        }
        let rows = spec.draft.config.verify_rows()?;
        #[cfg(feature = "cuda")]
        let calibrated = super::multi_calibrate::load(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            crate::backend::setup::MultiShape {
                rows,
                causal: true,
                capacity: self.capacity,
            },
            &self.closure,
        )
        .context("MQ verification setup calibration")?;
        #[cfg(feature = "cuda")]
        let (attention, selection) = calibrated.choose(self.capacity)?;
        #[cfg(feature = "rocm")]
        let attention = {
            self.closure.bind_fixed()?;
            super::verification::hip_tcmqa_plan(&self.device.ctx, self.capacity, rows)?
        };
        #[cfg(feature = "cuda")]
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{
                "rows":rows,"attention":attention,"selection":selection,"calibration":calibrated,
                "length":"device position counts prefix KV before the query block"
            }})
        );
        spec.verification = Some(super::verification::Verification::with_plan(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows,
                causal: true,
            },
            attention,
        )?);
        Ok(())
    }

    fn ensure_tree_attention(&mut self) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("tree requires draft State")?
            .tree_attention
            .is_some()
        {
            return Ok(());
        }
        let rows = self.spec_budget.rows();
        #[cfg(feature = "cuda")]
        let measured = super::multi_calibrate::load_tree(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            rows,
            self.capacity,
            &self.closure,
        )?;
        #[cfg(feature = "cuda")]
        let (plan, selection) = measured.choose(self.capacity)?;
        #[cfg(feature = "rocm")]
        let plan = {
            self.closure.bind_fixed()?;
            super::verification::hip_tcmqa_plan(&self.device.ctx, self.capacity, rows)?
        };
        let attention = super::verification::Verification::with_plan(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows,
                causal: false,
            },
            plan,
        )?;
        #[cfg(feature = "cuda")]
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"role":"tree","rows":rows,"attention":plan,"selection":selection,"calibration":measured}})
        );
        let spec = self.spec.as_mut().context("tree requires draft State")?;
        spec.tree_attention = Some(attention);
        Ok(())
    }

    pub fn verify_tree(
        &mut self,
        tree: &crate::tree::Tree,
        trace: Option<Target<'_>>,
    ) -> Result<Vec<u32>> {
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(
            tree.prefix() == self.position,
            "tree prefix does not match committed KV"
        );
        self.spec
            .as_mut()
            .context("tree verification requires draft State")?
            .tree
            .as_mut()
            .context("tree verification requires --spec-budget tree16/tree32/tree64")?
            .load(tree)?;
        let parents: Vec<_> = tree.nodes().iter().map(|n| n.parent).collect();
        let positions: Vec<_> = tree.nodes().iter().map(|n| n.position).collect();
        let step = crate::backend::Step {
            rows: tree.nodes().len(),
            input: crate::backend::Input::Ids,
            head: Head::All,
            mask: crate::backend::Mask::Tree {
                parents: &parents,
                positions: &positions,
            },
        };
        let graph = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .graph;
        ensure!(
            !graph || trace.is_none(),
            "tree Graph tracing requires --spec-graph false"
        );
        if graph {
            self.ensure_tree_verify_graph(step)?;
            let spec = self.spec.as_mut().context("draft State is not enabled")?;
            spec.tree_verify_graph
                .as_ref()
                .context("tree verify Graph is not captured")?
                .launch()?;
            spec.active_rows = tree.nodes().len();
        } else {
            self.run_step(step, trace)?;
        }
        let spec = self.spec.as_ref().context("draft State is not enabled")?;
        let values = self
            .device
            .stream
            .clone_dtoh(&spec.predictions.slice(..tree.nodes().len()))?;
        Ok(values)
    }

    fn ensure_tree_verify_graph(&mut self, step: crate::backend::Step<'_>) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree_verify_graph
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let s = self.device.stream.clone();
        let timer = std::time::Instant::now();
        let token = s.clone_dtoh(&self.work.token)?;
        let position = s.clone_dtoh(&self.work.position)?;
        let length = s.clone_dtoh(&self.work.length)?;
        self.run_step(step, None)?;
        s.synchronize()?;
        s.memcpy_htod(&token, &mut self.work.token)?;
        s.memcpy_htod(&position, &mut self.work.position)?;
        s.memcpy_htod(&length, &mut self.work.length)?;
        s.synchronize()?;
        #[cfg(feature = "cuda")]
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        #[cfg(feature = "rocm")]
        let capture = s.begin_capture()?;
        let launched = self.run_step(step, None);
        #[cfg(feature = "cuda")]
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        #[cfg(feature = "rocm")]
        let captured = capture.finish().map(Some);
        launched?;
        let graph = captured?.context("empty tree verify Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .tree_verify_graph = Some(graph);
        eprintln!(
            "{}",
            serde_json::json!({"tree_graph_setup":{"phase":"verify","budget":self.spec_budget.rows(),"wall_ms":timer.elapsed().as_secs_f64()*1000.0}})
        );
        Ok(())
    }

    fn tree_gather_scatter(&mut self) -> Result<()> {
        let spec = self.spec.as_mut().context("draft State is not enabled")?;
        let state = spec.tree.as_mut().context("tree State is not enabled")?;
        state.gather_path(
            &self.state.kv,
            &self
                .hidden_capture
                .as_ref()
                .context("target capture missing")?
                .values,
        )?;
        state.scatter(&mut self.state.kv)
    }

    fn capture_tree_commit(&mut self) -> Result<()> {
        self.closure.capture()?;
        let s = self.device.stream.clone();
        s.synchronize()?;
        #[cfg(feature = "cuda")]
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        #[cfg(feature = "rocm")]
        let capture = s.begin_capture()?;
        let launched = self.tree_gather_scatter();
        #[cfg(feature = "cuda")]
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        #[cfg(feature = "rocm")]
        let captured = capture.finish().map(Some);
        launched?;
        let graph = captured?.context("empty tree commit Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .tree_commit_graph = Some(graph);
        Ok(())
    }

    fn tree_inject(&mut self, start: usize) -> Result<()> {
        let spec = self.spec.as_mut().context("draft State is not enabled")?;
        let state = spec.tree.as_ref().context("tree State is not enabled")?;
        spec.draft.inject_device(
            state.compact_hidden(),
            state.keep(),
            start,
            state.prefix(),
            super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
            },
        )
    }

    fn capture_tree_inject(&mut self, start: usize, rows: usize) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .inject_graphs[rows - 1]
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let s = self.device.stream.clone();
        self.tree_inject(start)?;
        s.synchronize()?;
        #[cfg(feature = "cuda")]
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        #[cfg(feature = "rocm")]
        let capture = s.begin_capture()?;
        let launched = self.tree_inject(start);
        #[cfg(feature = "cuda")]
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        #[cfg(feature = "rocm")]
        let captured = capture.finish().map(Some);
        launched?;
        let graph = captured?.context("empty tree inject Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .inject_graphs[rows - 1] = Some(graph);
        Ok(())
    }

    pub fn tree_metadata(&self) -> Result<crate::backend::TreeMetadata> {
        let tree = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree
            .as_ref()
            .context("tree State is not enabled")?;
        let s = &self.device.stream;
        let n = tree.rows();
        Ok(crate::backend::TreeMetadata {
            ids: s.clone_dtoh(&tree.ids().slice(..n))?,
            ancestors: s.clone_dtoh(&tree.ancestors().slice(..n))?,
            positions: s.clone_dtoh(&tree.positions().slice(..n))?,
            slots: s.clone_dtoh(&tree.slots().slice(..n))?,
            valid_rows: s.clone_dtoh(tree.valid_rows())?[0],
            prefix: s.clone_dtoh(tree.prefix())?[0],
        })
    }

    pub fn tree_hidden(&self) -> Result<Vec<f32>> {
        let spec = self.spec.as_ref().context("draft State is not enabled")?;
        let rows = spec
            .tree
            .as_ref()
            .context("tree State is not enabled")?
            .rows();
        let capture = self
            .hidden_capture
            .as_ref()
            .context("target capture missing")?;
        Ok(self
            .device
            .stream
            .clone_dtoh(&capture.values.slice(..rows * capture.width))?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    pub fn poison_tree_padding(&mut self, tree: &crate::tree::Tree) -> Result<()> {
        let n = tree.nodes().len();
        let budget = self.spec_budget.rows();
        ensure!(
            self.spec_budget.is_tree() && n < budget && tree.prefix() + budget <= self.capacity,
            "padding negative control requires actual N<tree budget within capacity"
        );
        let d = self.config.head_dim;
        let values = vec![bf16::NAN; (budget - n) * d];
        for kv in &mut self.state.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = (head * self.capacity + tree.prefix() + n) * d;
                self.device
                    .stream
                    .memcpy_htod(&values, &mut kv.k.slice_mut(lo..lo + values.len()))?;
                self.device
                    .stream
                    .memcpy_htod(&values, &mut kv.v.slice_mut(lo..lo + values.len()))?;
            }
        }
        self.device.stream.synchronize()?;
        Ok(())
    }

    pub fn commit_tree(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
    ) -> Result<()> {
        let spec = self.spec.as_ref().context("draft State is not enabled")?;
        let predictions = self
            .device
            .stream
            .clone_dtoh(&spec.predictions.slice(..tree.nodes().len()))?;
        self.commit_tree_checked(tree, plan, eos, &predictions)
    }

    pub fn commit_tree_fixture(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
        reference_predictions: &[u32],
    ) -> Result<()> {
        ensure!(
            reference_predictions.len() == tree.nodes().len()
                && reference_predictions
                    .iter()
                    .all(|&v| (v as usize) < self.config.vocab_size),
            "teacher-forced reference predictions have invalid extent/token"
        );
        self.commit_tree_checked(tree, plan, eos, reference_predictions)
    }

    fn commit_tree_checked(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
        predictions: &[u32],
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        let mut cursor = crate::tree::Cursor {
            position: self.position,
            token: self.token()?,
        };
        tree.check_commit(plan, predictions, eos, &cursor)?;
        let result = (|| -> Result<()> {
            let spec = self.spec.as_mut().context("draft State is not enabled")?;
            spec.tree
                .as_mut()
                .context("tree State is not enabled")?
                .load_commit(plan)?;
            let graph = spec.graph;
            self.device.stream.synchronize()?;
            let s = self.device.stream.clone();
            let cached = graph
                && self
                    .spec
                    .as_ref()
                    .context("draft State is not enabled")?
                    .tree_commit_graph
                    .is_some();
            if cached {
                self.spec
                    .as_ref()
                    .context("draft State is not enabled")?
                    .tree_commit_graph
                    .as_ref()
                    .context("tree commit Graph missing")?
                    .launch()?;
            } else {
                self.tree_gather_scatter()?;
            }
            s.synchronize()?;
            if graph && !cached {
                self.capture_tree_commit()?;
            }
            if graph {
                self.capture_tree_inject(self.position, plan.rows.len())?;
            }
            if graph {
                self.spec
                    .as_ref()
                    .context("draft State is not enabled")?
                    .inject_graphs[plan.rows.len() - 1]
                    .as_ref()
                    .context("tree inject Graph missing")?
                    .launch()?;
            } else {
                self.tree_inject(self.position)?;
            }
            s.synchronize()?;
            let spec = self.spec.as_ref().context("draft State is not enabled")?;
            spec.tree
                .as_ref()
                .context("tree State is not enabled")?
                .select_logits(&spec.logits, &mut self.work.logits)?;
            s.memcpy_htod(&[plan.position as i32], &mut self.work.position)?;
            s.memcpy_htod(&[(plan.position + 1) as i32], &mut self.work.length)?;
            s.memcpy_htod(&[plan.next], &mut self.work.token)?;
            s.synchronize()?;
            self.logits_cache.mark_valid();
            tree.finish(plan, predictions, eos, &mut cursor)?;
            self.position = cursor.position;
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result.context("tree commit failed; reset/reload required")
    }

    pub fn tree_kv_path(&self, path: &[usize]) -> Result<Vec<f32>> {
        let state = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree
            .as_ref()
            .context("tree State is not enabled")?;
        ensure!(
            !path.is_empty()
                && path[0] == 0
                && path.iter().all(|&i| i < state.rows())
                && path.windows(2).all(|r| r[0] < r[1]),
            "tree KV path is invalid"
        );
        let prefix = self.device.stream.clone_dtoh(state.prefix())?[0];
        ensure!(
            prefix >= 0 && prefix as usize == self.position,
            "tree KV path must be read before commit"
        );
        let mut result = Vec::new();
        let d = self.config.head_dim;
        for layer in &self.state.kv {
            for data in [&layer.k, &layer.v] {
                for head in 0..self.config.num_key_value_heads {
                    for &row in path {
                        let lo = (head * self.capacity + self.position + row) * d;
                        result.extend(
                            self.device
                                .stream
                                .clone_dtoh(&data.slice(lo..lo + d))?
                                .into_iter()
                                .map(bf16::to_f32),
                        );
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn tree_injected_hidden(&self) -> Result<Vec<f32>> {
        let state = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree
            .as_ref()
            .context("tree State is not enabled")?;
        ensure!(
            state.keep() > 0,
            "tree injected hidden requires a committed path"
        );
        Ok(self
            .device
            .stream
            .clone_dtoh(
                &state
                    .compact_hidden()
                    .slice(..state.keep() * 5 * self.config.hidden_size),
            )?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    fn inject_capture(
        &mut self,
        rows: usize,
        start: usize,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        let spec = self.spec.as_mut().context("draft state is not enabled")?;
        spec.draft.inject(
            &self
                .hidden_capture
                .as_ref()
                .context("target capture missing")?
                .values,
            rows,
            start,
            super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
            },
            trace,
        )
    }

    fn run_spec_device(&mut self, phase: SpecPhase, start: usize) -> Result<()> {
        let rows = phase.rows(self.spec.as_ref().context("draft is not enabled")?)?;
        match phase {
            SpecPhase::Draft => {
                self.spec
                    .as_mut()
                    .context("draft is not enabled")?
                    .draft
                    .propose_device(
                        &self.work.token,
                        &self.work.position,
                        start,
                        (self.weights.embedding(), &self.weights.head),
                        super::draft::Compute {
                            ops: &self.ops,
                            blas: &mut self.blas,
                        },
                    )?;
                if self.model_spec.moe.is_some() {
                    self.spec
                        .as_ref()
                        .context("MoE draft missing")?
                        .draft
                        .copy_proposals(&mut self.work.ids.slice_mut(1..rows + 1))?;
                    self.device
                        .stream
                        .memcpy_dtod(&self.work.token, &mut self.work.ids.slice_mut(..1))?;
                }
                Ok(())
            }
            SpecPhase::Verify(_) => {
                self.spec
                    .as_ref()
                    .context("draft is not enabled")?
                    .draft
                    .copy_proposals(&mut self.work.ids.slice_mut(1..rows))?;
                self.device
                    .stream
                    .memcpy_dtod(&self.work.token, &mut self.work.ids.slice_mut(..1))?;
                self.run_step(
                    crate::backend::Step {
                        rows,
                        input: crate::backend::Input::Ids,
                        head: Head::All,
                        mask: crate::backend::Mask::Causal,
                    },
                    None,
                )
            }
            SpecPhase::Inject(rows) => {
                let spec = self.spec.as_mut().context("draft is not enabled")?;
                spec.draft.inject_device(
                    &self
                        .hidden_capture
                        .as_ref()
                        .context("target capture missing")?
                        .values,
                    rows,
                    start,
                    &self.work.position,
                    super::draft::Compute {
                        ops: &self.ops,
                        blas: &mut self.blas,
                    },
                )
            }
        }
    }

    fn ensure_spec_graph(&mut self, phase: SpecPhase, start: usize) -> Result<()> {
        if self.model_spec.moe.is_some() && matches!(phase, SpecPhase::Verify(_)) {
            ensure!(
                self.has_moe_segments() || self.has_moe_verify_graphs(),
                "MoE verification graphs were not bound before Ready"
            );
            return Ok(());
        }
        let rows = phase.rows(self.spec.as_ref().context("draft is not enabled")?)?;
        if self
            .spec
            .as_mut()
            .context("draft is not enabled")?
            .graph_slot(phase)?
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let timer = std::time::Instant::now();
        let s = self.device.stream.clone();
        let token = s.clone_dtoh(&self.work.token)?;
        let position = s.clone_dtoh(&self.work.position)?;
        let length = s.clone_dtoh(&self.work.length)?;
        if self.model_spec.moe.is_some() {
            self.spec
                .as_mut()
                .context("MoE draft missing")?
                .draft
                .checkpoint_kv(start)?;
        }
        let warmed = self.run_spec_device(phase, start);
        s.synchronize()?;
        if self.model_spec.moe.is_some() {
            self.spec
                .as_mut()
                .context("MoE draft missing")?
                .draft
                .restore_kv(start, 0)?;
        }
        s.memcpy_htod(&token, &mut self.work.token)?;
        s.memcpy_htod(&position, &mut self.work.position)?;
        s.memcpy_htod(&length, &mut self.work.length)?;
        s.synchronize()?;
        warmed?;
        #[cfg(feature = "cuda")]
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        #[cfg(feature = "rocm")]
        let capture = s.begin_capture()?;
        let launched = self.run_spec_device(phase, start);
        #[cfg(feature = "cuda")]
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        #[cfg(feature = "rocm")]
        let captured = capture.finish().map(Some);
        launched?;
        let graph = captured?.context("empty spec Graph")?;
        graph.upload()?;
        s.synchronize()?;
        *self
            .spec
            .as_mut()
            .context("draft is not enabled")?
            .graph_slot(phase)? = Some(graph);
        eprintln!(
            "{}",
            serde_json::json!({"spec_graph_setup":{"phase":phase.name(),"rows":rows,"capacity":self.capacity,"wall_ms":timer.elapsed().as_secs_f64()*1000.0,"prefix":"device position","warmup":"selected implementation; token/position/length restored"}})
        );
        Ok(())
    }

    fn launch_spec_graph(&mut self, phase: SpecPhase) -> Result<()> {
        if self.model_spec.moe.is_some()
            && let SpecPhase::Verify(rows) = phase
        {
            self.state.set_sequence(
                &self.device.stream,
                self.position as u32,
                rows as u32,
                self.workspace_rows as u32,
            )?;
            self.run_moe_graph(rows)?;
            self.spec.as_mut().context("MoE chain missing")?.active_rows = rows;
            return Ok(());
        }
        let spec = self.spec.as_mut().context("draft is not enabled")?;
        spec.graph_slot(phase)?
            .as_ref()
            .context("spec Graph has not been captured")?
            .launch()?;
        if let SpecPhase::Verify(rows) = phase {
            spec.active_rows = rows;
        }
        Ok(())
    }

    fn run_tree_draft_device(&mut self, start: usize) -> Result<()> {
        self.spec
            .as_mut()
            .context("draft is not enabled")?
            .draft
            .forward_base_device(
                &self.work.token,
                &self.work.position,
                start,
                (self.weights.embedding(), &self.weights.head),
                super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                },
            )
    }

    fn ensure_tree_draft_graph(&mut self, start: usize) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("draft is not enabled")?
            .tree_draft_graph
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let timer = std::time::Instant::now();
        let s = self.device.stream.clone();
        self.run_tree_draft_device(start)?;
        s.synchronize()?;
        #[cfg(feature = "cuda")]
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        #[cfg(feature = "rocm")]
        let capture = s.begin_capture()?;
        let launched = self.run_tree_draft_device(start);
        #[cfg(feature = "cuda")]
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        #[cfg(feature = "rocm")]
        let captured = capture.finish().map(Some);
        launched?;
        let graph = captured?.context("empty tree draft Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft is not enabled")?
            .tree_draft_graph = Some(graph);
        eprintln!(
            "{}",
            serde_json::json!({"tree_graph_setup":{"phase":"draft","rows":7,"capacity":self.capacity,"wall_ms":timer.elapsed().as_secs_f64()*1000.0}})
        );
        Ok(())
    }

    pub fn draft_config(&self) -> Result<&crate::dspark::DraftConfig> {
        Ok(&self
            .spec
            .as_ref()
            .context("draft is not enabled")?
            .draft
            .config)
    }
    pub fn spec_round(
        &mut self,
        limit: usize,
        mut trace: Option<&mut Trace>,
        prefix: &str,
    ) -> Result<crate::dspark::Round> {
        #[cfg(feature = "cuda")]
        ensure!(
            self.qwen.is_none() || self.model_spec.moe.is_some(),
            "DSpark verification is not bound for {:?}",
            self.model_spec.architecture
        );
        let transaction = self.model_spec.moe.is_some();
        let origin = self.position;
        let pending = self.pending_token;
        let resident = self.resident_ids_len;
        let last_input = self.last_input_kind;
        if transaction {
            self.closure.assert_ready()?;
            ensure!(
                !self.poisoned,
                "DSpark transaction requires reset after an error"
            );
            let spec = self
                .spec
                .as_mut()
                .context("MoE chain requires a loaded draft")?;
            let rows = spec.max_verify_rows;
            ensure!(
                limit > 0
                    && !self.spec_budget.is_tree()
                    && origin
                        .checked_add(rows)
                        .is_some_and(|end| end <= self.capacity),
                "MoE chain limit={limit} prefix={origin} rows={rows} capacity={} budget={:?}",
                self.capacity,
                self.spec_budget
            );
            spec.checkpoint
                .as_mut()
                .context("MoE chain checkpoint missing")?
                .save_rows(
                    &self.state,
                    &self.work.token,
                    origin,
                    rows,
                    &self.device.stream,
                    None,
                )?;
            spec.draft.checkpoint_kv(origin)?;
        }
        let result = (|| {
            self.closure.assert_ready()?;
            if self.spec_budget.is_tree() {
                return self.tree_round(limit, trace, prefix);
            }
            self.ensure_verification()?;
            self.spec
                .as_mut()
                .context("draft state is not enabled")?
                .draft
                .setup_attention(
                    &self.device,
                    &self.ops,
                    &self.config,
                    &self.runtime,
                    &self.closure,
                )?;
            let _round_range = super::profile::Range::new(self.profile_rounds, c"dspark_round");
            let max_rows = self
                .spec
                .as_ref()
                .context("draft is not enabled")?
                .max_verify_rows;
            ensure!(limit > 0, "spec round output limit must be positive");
            ensure!(
                self.position
                    .checked_add(max_rows)
                    .is_some_and(|end| end <= self.capacity),
                "spec verification position={} rows={max_rows} exceeds capacity={}",
                self.position,
                self.capacity
            );
            let graph = self.spec.as_ref().context("draft is not enabled")?.graph;
            ensure!(
                !graph || trace.is_none(),
                "spec Graph tracing requires --spec-graph false"
            );
            let start = self.position;
            if graph {
                if transaction {
                    ensure!(
                        self.has_graph(Graph::Draft) && self.spec_verify_graph_ready(),
                        "MoE draft/verify graphs were not bound before Ready"
                    );
                } else {
                    self.ensure_spec_graph(SpecPhase::Draft, start)?;
                    self.ensure_spec_graph(SpecPhase::Verify(max_rows), start)?;
                }
            }
            let anchor = if graph && !transaction {
                None
            } else {
                Some(self.token()?)
            };
            let draft_range = super::profile::Range::new(self.profile_rounds, c"draft");
            let draft_start = std::time::Instant::now();
            let proposals = if graph {
                self.launch_spec_graph(SpecPhase::Draft)?;
                self.spec
                    .as_mut()
                    .context("draft is not enabled")?
                    .draft
                    .proposals()?
            } else {
                let anchor = anchor.context("eager draft anchor missing")?;
                let spec = self.spec.as_mut().context("draft is not enabled")?;
                spec.draft.propose(
                    anchor,
                    start,
                    (self.weights.embedding(), &self.weights.head),
                    super::draft::Compute {
                        ops: &self.ops,
                        blas: &mut self.blas,
                    },
                    trace.as_deref_mut().map(|t| (t, prefix)),
                    None,
                )?
            };
            self.device.stream.synchronize()?;
            let draft_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
            drop(draft_range);
            let base_rows = if transaction {
                let spec = self.spec.as_ref().context("draft is not enabled")?;
                let confidences = spec.draft.confidences()?;
                crate::dspark::select_verify_rows(
                    &confidences,
                    crate::dspark::MIN_MOE_VERIFY_ROWS,
                    &spec.verify_schedule,
                )?
            } else {
                max_rows
            };
            #[cfg(test)]
            let mut forced_applied = false;
            #[cfg(test)]
            let (rows, proposals) = if let Some(forced) = self
                .spec
                .as_mut()
                .context("chain missing")?
                .forced_proposals
                .take()
            {
                let forced_rows = forced.len() + 1;
                ensure!(
                    if transaction {
                        (crate::dspark::MIN_MOE_VERIFY_ROWS..=max_rows).contains(&forced_rows)
                    } else {
                        forced_rows == max_rows
                    } && forced
                        .iter()
                        .all(|&id| (id as usize) < self.config.vocab_size),
                    "forced chain proposals exceed the loaded geometry"
                );
                forced_applied = true;
                (forced_rows, forced)
            } else {
                (base_rows, proposals[..base_rows - 1].to_vec())
            };
            #[cfg(not(test))]
            let (rows, proposals) = (base_rows, proposals[..base_rows - 1].to_vec());
            #[cfg(test)]
            if graph && transaction && forced_applied {
                let ids: Vec<_> = std::iter::once(anchor.context("forced graph anchor missing")?)
                    .chain(proposals.iter().copied())
                    .collect();
                self.device
                    .stream
                    .memcpy_htod(&ids, &mut self.work.ids.slice_mut(..rows))?;
            }
            if !graph {
                let mut ids = vec![anchor.context("eager verify anchor missing")?];
                ids.extend(&proposals);
                self.device
                    .stream
                    .memcpy_htod(&ids, &mut self.work.ids.slice_mut(..rows))?;
                self.device
                    .stream
                    .memcpy_htod(&[start as i32], &mut self.work.position)?;
            }
            let verify_start = std::time::Instant::now();
            let verify_range = super::profile::Range::new(self.profile_rounds, c"verify");
            if graph {
                self.launch_spec_graph(SpecPhase::Verify(rows))?;
            } else {
                self.run_step(
                    crate::backend::Step {
                        rows,
                        input: crate::backend::Input::Ids,
                        head: Head::All,
                        mask: crate::backend::Mask::Causal,
                    },
                    trace.as_deref_mut().map(|trace| Target {
                        trace,
                        prefix,
                        skip: 0,
                    }),
                )?;
            }
            let predictions = self.device.stream.clone_dtoh(
                &self
                    .spec
                    .as_ref()
                    .context("draft is not enabled")?
                    .predictions
                    .slice(..rows),
            )?;
            let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
            drop(verify_range);
            self.position += rows;
            let accepted = crate::dspark::accept(&proposals, &predictions)?;
            let mut output = proposals[..accepted.proposals].to_vec();
            output.push(accepted.bonus);
            output.truncate(limit);
            if transaction
                && let Some(index) = output.iter().position(|token| {
                    self.spec
                        .as_ref()
                        .is_some_and(|spec| spec.eos.contains(token))
                })
            {
                output.truncate(index + 1);
            }
            let committed = output.len();
            let accepted_proposals = accepted.proposals.min(committed - 1);
            let bonus = *output.last().context("spec round emitted no token")?;
            if graph {
                if transaction {
                    ensure!(
                        self.has_graph(Graph::Inject(committed)),
                        "MoE commit graph rows={committed} missing before Ready"
                    );
                } else {
                    self.ensure_spec_graph(SpecPhase::Inject(committed), start)?;
                }
            }
            let inject_start = std::time::Instant::now();
            let _inject_range = super::profile::Range::new(self.profile_rounds, c"inject");
            if graph {
                self.launch_spec_graph(SpecPhase::Inject(committed))?;
            } else {
                self.inject_capture(committed, start, trace.as_deref_mut().map(|t| (t, prefix)))?;
            }
            let s = &self.device.stream;
            let spec = self.spec.as_ref().context("draft is not enabled")?;
            s.memcpy_dtod(
                &spec.logits.slice(
                    (committed - 1) * self.config.vocab_size..committed * self.config.vocab_size,
                ),
                &mut self.work.logits,
            )?;
            self.logits_cache.mark_valid();
            if transaction {
                {
                    let spec = self.spec.as_mut().context("MoE chain missing")?;
                    spec.checkpoint
                        .as_ref()
                        .context("MoE chain checkpoint missing")?
                        .restore_for_replay(&mut self.state, committed, rows, s)?;
                    spec.draft.restore_kv(start, committed)?;
                    if committed < rows {
                        self.replay_gdn_prefix(start, committed)?;
                    }
                    let s = &self.device.stream;
                    self.state.set_sequence(
                        s,
                        (start + committed - 1) as u32,
                        1,
                        self.workspace_rows as u32,
                    )?;
                    s.memcpy_htod(&[(start + committed - 1) as i32], &mut self.work.position)?;
                    s.memcpy_htod(&[(start + committed) as i32], &mut self.work.length)?;
                    self.qwen_advance()?;
                    self.committed_prefix_ids
                        .push(anchor.context("MoE committed anchor missing")?);
                    self.committed_prefix_ids
                        .extend_from_slice(&output[..committed - 1]);
                    self.resident_ids_len = committed;
                    self.last_input_kind = Some(PrefixLastInputKind::DecodeToken);
                    self.position = start + committed;
                    self.set_token(bonus)?;
                    self.device.stream.synchronize()?;
                }
            } else {
                self.rewind(start + committed, bonus)?;
            }
            if let Some(t) = trace {
                t.add(
                    format!("{prefix}.target_argmax"),
                    vec![rows],
                    predictions.iter().map(|&v| v as f32).collect(),
                )?;
                t.add(
                    format!("{prefix}.m"),
                    vec![1],
                    vec![accepted_proposals as f32],
                )?;
                t.add(
                    format!("{prefix}.bonus"),
                    vec![1],
                    vec![accepted.bonus as f32],
                )?;
            }
            Ok(crate::dspark::Round {
                logits_rows: (0..committed).collect(),
                proposals,
                predictions,
                accepted: accepted_proposals,
                committed,
                output,
                position: self.position,
                draft_ms,
                verify_ms,
                inject_ms: inject_start.elapsed().as_secs_f64() * 1000.0,
                tree: None,
            })
        })();
        if transaction && result.is_err() {
            self.poisoned = true;
            let spec = self
                .spec
                .as_mut()
                .context("failed MoE chain lost its draft")?;
            spec.checkpoint
                .as_ref()
                .context("failed MoE chain lost its checkpoint")?
                .restore(&mut self.state, &mut self.work.token, &self.device.stream)
                .context("restore target after MoE chain failure")?;
            spec.draft
                .restore_kv(origin, 0)
                .context("restore draft after MoE chain failure")?;
            self.device
                .stream
                .memcpy_htod(&[origin as i32], &mut self.work.position)?;
            self.device
                .stream
                .memcpy_htod(&[origin as i32 + 1], &mut self.work.length)?;
            self.committed_prefix_ids.truncate(origin);
            if last_input == Some(PrefixLastInputKind::PrefillChunk) {
                self.device.stream.memcpy_htod(
                    &self.committed_prefix_ids[origin - resident..origin],
                    &mut self.work.ids.slice_mut(..resident),
                )?;
            }
            self.position = origin;
            self.pending_token = pending;
            self.resident_ids_len = resident;
            self.last_input_kind = last_input;
            self.logits_cache.invalidate();
            self.device.stream.synchronize()?;
        }
        result
    }

    pub fn set_spec_graph(&mut self, graph: bool) -> Result<()> {
        self.device.stream.synchronize()?;
        ensure!(
            !graph || !self.graphs.is_empty(),
            "spec Graphs were not prepared; load with --spec-graph true"
        );
        self.spec.as_mut().context("draft is not enabled")?.graph = graph;
        eprintln!("{}", serde_json::json!({"spec_execution":{"graph":graph}}));
        Ok(())
    }

    pub fn set_spec_eos(&mut self, eos: &[u32]) -> Result<()> {
        ensure!(
            eos.iter().all(|&v| (v as usize) < self.config.vocab_size),
            "spec EOS token is outside vocabulary"
        );
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .eos = eos.to_vec();
        Ok(())
    }

    fn tree_round(
        &mut self,
        limit: usize,
        trace: Option<&mut Trace>,
        prefix: &str,
    ) -> Result<crate::dspark::Round> {
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(limit > 0, "tree round output limit must be positive");
        let start = self.position;
        ensure!(
            start
                .checked_add(self.spec_budget.rows())
                .is_some_and(|end| end <= self.capacity),
            "tree scratch exceeds KV capacity"
        );
        let anchor = self.token()?;
        let draft_start = std::time::Instant::now();
        let graph = self
            .spec
            .as_ref()
            .context("tree round requires enabled draft State")?
            .graph;
        let spec = self
            .spec
            .as_mut()
            .context("tree round requires enabled draft State")?;
        spec.draft.setup_attention(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            &self.closure,
        )?;
        spec.draft
            .setup_markov(&self.device, &self.ops, &self.runtime, &self.closure)?;
        if graph {
            self.ensure_tree_draft_graph(start)?;
            self.spec
                .as_mut()
                .context("tree round requires enabled draft State")?
                .tree_draft_graph
                .as_ref()
                .context("tree draft Graph is not captured")?
                .launch()?;
        } else {
            let spec = self
                .spec
                .as_mut()
                .context("tree round requires enabled draft State")?;
            spec.draft.forward_base(
                anchor,
                start,
                (self.weights.embedding(), &self.weights.head),
                super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                },
            )?;
        }
        self.device.stream.synchronize()?;
        let base_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
        let build_start = std::time::Instant::now();
        let (tree, stats) = self.build_tree(self.tree_builder, anchor, start)?;
        let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
        if self.compare_builders {
            let other = match self.tree_builder {
                TreeBuilder::Serial => TreeBuilder::Waves,
                TreeBuilder::Waves => TreeBuilder::Serial,
            };
            let (reference, _) = self.build_tree(other, anchor, start)?;
            ensure!(
                tree.nodes().len() == reference.nodes().len(),
                "tree builders return different node counts"
            );
            for (row, (actual, expected)) in tree.nodes().iter().zip(reference.nodes()).enumerate()
            {
                ensure!(
                    actual == expected && actual.logp.to_bits() == expected.logp.to_bits(),
                    "tree builders differ at node {row}: actual={actual:?}, expected={expected:?}"
                );
            }
        }
        let draft_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
        let verify_start = std::time::Instant::now();
        let predictions = self.verify_tree(
            &tree,
            trace.map(|trace| Target {
                trace,
                prefix,
                skip: 0,
            }),
        )?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        let eos = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .eos
            .clone();
        let plan = tree.select(&predictions, limit, &eos)?;
        let commit_start = std::time::Instant::now();
        self.commit_tree(&tree, &plan, &eos)?;
        let commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;
        Ok(crate::dspark::Round {
            logits_rows: plan.rows.clone(),
            proposals: tree.nodes()[1..].iter().map(|n| n.token).collect(),
            predictions,
            accepted: plan.matched,
            committed: plan.output.len(),
            output: plan.output,
            position: plan.position,
            draft_ms,
            verify_ms,
            inject_ms: commit_ms,
            tree: Some(crate::dspark::TreeTiming {
                builder: self.tree_builder,
                base_ms,
                build_ms,
                waves: stats.waves,
                requests: stats.requests,
                used_requests: stats.used_requests,
                batch_sizes: stats.batch_sizes,
                compared_serial: self.compare_builders,
            }),
        })
    }

    fn build_tree(
        &mut self,
        builder: TreeBuilder,
        anchor: u32,
        prefix: usize,
    ) -> Result<(crate::tree::Tree, crate::tree::WaveStats)> {
        let spec = self
            .spec
            .as_mut()
            .context("tree building requires draft state")?;
        match builder {
            TreeBuilder::Waves => crate::tree::Tree::best_first_waves(
                anchor,
                prefix,
                self.capacity,
                self.spec_budget.rows(),
                7,
                |requests| {
                    spec.draft
                        .distributions_batch(requests)?
                        .iter()
                        .map(crate::backend::Top4::distribution)
                        .collect()
                },
            ),
            TreeBuilder::Serial => {
                let mut cache = std::collections::HashMap::new();
                let mut count = 0;
                let tree = crate::tree::Tree::best_first(
                    anchor,
                    prefix,
                    self.capacity,
                    self.spec_budget.rows(),
                    7,
                    |row, token| {
                        if let Some(value) = cache.get(&(row, token)) {
                            return Ok(crate::tree::Distribution::clone(value));
                        }
                        let value = spec
                            .draft
                            .distributions_batch(&[(row, token)])?
                            .into_iter()
                            .next()
                            .context("empty serial Markov result")?
                            .distribution()?;
                        count += 1;
                        cache.insert((row, token), value.clone());
                        Ok(value)
                    },
                )?;
                Ok((
                    tree,
                    crate::tree::WaveStats {
                        waves: count,
                        requests: count,
                        batch_sizes: vec![1; count],
                        used_requests: count,
                    },
                ))
            }
        }
    }

    pub fn compare_tree_builders(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !enabled || self.spec_budget.is_tree(),
            "builder comparison requires a tree budget"
        );
        self.compare_builders = enabled;
        Ok(())
    }

    pub fn distributions_batch(
        &mut self,
        requests: &[(u8, u32)],
    ) -> Result<Vec<crate::backend::Top4>> {
        ensure!(
            self.spec_budget.is_tree(),
            "distributions_batch requires a tree spec budget"
        );
        self.closure.assert_ready()?;
        self.spec
            .as_mut()
            .context("draft state is not enabled")?
            .draft
            .distributions_batch(requests)
    }

    pub fn spec_state_bits(&self, start: usize, rows: usize) -> Result<Vec<u8>> {
        ensure!(
            rows > 0 && start + rows == self.position,
            "spec bit comparison must cover newly committed prefix"
        );
        let s = &self.device.stream;
        let mut out = Vec::new();
        super::copy::bits(s, &self.work.logits, &mut out)?;
        let spec = self.spec.as_ref().context("draft is not enabled")?;
        super::copy::bits(s, &spec.logits, &mut out)?;
        spec.draft.state_bits(start, rows, &mut out)?;
        let d = self.config.head_dim;
        for kv in &self.state.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = (head * self.capacity + start) * d;
                let hi = lo + rows * d;
                super::copy::bits(s, &kv.k.slice(lo..hi), &mut out)?;
                super::copy::bits(s, &kv.v.slice(lo..hi), &mut out)?;
            }
        }
        for x in s
            .clone_dtoh(&self.work.position)?
            .into_iter()
            .chain(s.clone_dtoh(&self.work.length)?)
        {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for x in s.clone_dtoh(&self.work.token)? {
            out.extend_from_slice(&x.to_le_bytes());
        }
        Ok(out)
    }

    pub fn inject_hidden(&mut self, hidden: &[f32], start: usize, rows: usize) -> Result<()> {
        let limit = super::closure::injection_limit(
            super::closure::small_rows(self.chunk, self.capacity),
            self.spec
                .as_ref()
                .context("draft geometry missing")?
                .draft
                .config
                .verify_rows()?,
        );
        ensure!(
            rows > 0 && (rows > 16 || rows <= limit),
            "injection rows {rows} exceed declared small-row limit {limit}"
        );
        ensure!(self.spec.is_some(), "draft is not enabled");
        let capture = self
            .hidden_capture
            .as_mut()
            .context("target capture missing")?;
        ensure!(
            rows.checked_mul(capture.width) == Some(hidden.len()),
            "injected hidden shape mismatch"
        );
        ensure!(rows <= self.workspace_rows, "injection exceeds workspace");
        let values: Vec<bf16> = hidden.iter().copied().map(bf16::from_f32).collect();
        self.device
            .stream
            .memcpy_htod(&values, &mut capture.values.slice_mut(..values.len()))?;
        self.inject_capture(rows, start, None)
    }

    pub fn draft_proposals(
        &mut self,
        anchor: u32,
        position: usize,
        trace: &mut Trace,
        prefix: &str,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        ensure!(
            self.qwen.is_none(),
            "Qwen DSpark proposal execution is not bound"
        );
        let spec = self.spec.as_mut().context("draft is not enabled")?;
        spec.draft.setup_attention(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            &self.closure,
        )?;
        spec.draft.propose(
            anchor,
            position,
            (self.weights.embedding(), &self.weights.head),
            super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
            },
            Some((trace, prefix)),
            forced,
        )
    }

    pub fn draft_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        self.spec
            .as_ref()
            .context("draft is not enabled")?
            .draft
            .snapshot_kv(trace, start, rows)
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

impl Engine {
    pub fn kv_snapshot(&self, start: usize, rows: usize) -> Result<Vec<f32>> {
        ensure!(
            start
                .checked_add(rows)
                .is_some_and(|end| end <= self.position),
            "KV snapshot exceeds valid prefix"
        );
        let c = &self.config;
        let mut output = Vec::new();
        for layer in &self.state.kv {
            if let Some(compact) = &layer.compact {
                let s = &self.device.stream;
                let cursor = s.clone_dtoh(&compact.cursor)?;
                ensure!(
                    cursor
                        == [
                            self.position as i32,
                            crate::kv::sealed_until(self.position) as i32,
                            0,
                            0
                        ],
                    "KV snapshot requires a committed compact cache"
                );
                for (buffer, codes, stats, is_k) in [
                    (&layer.k, &compact.k_code, &compact.k_stats, true),
                    (&layer.v, &compact.v_code, &compact.v_stats, false),
                ] {
                    for head in 0..c.num_key_value_heads {
                        for pos in start..start + rows {
                            if pos < crate::kv::SINK_TOKENS || pos >= cursor[1] as usize {
                                let lo = (head * compact.layout.ring
                                    + crate::kv::ring_slot(pos, compact.layout.ring)?)
                                    * c.head_dim;
                                output.extend(
                                    s.clone_dtoh(&buffer.slice(lo..lo + c.head_dim))?
                                        .into_iter()
                                        .map(bf16::to_f32),
                                );
                            } else {
                                let lo = (head * self.capacity + pos) * c.head_dim;
                                let code = s.clone_dtoh(&codes.slice(lo..lo + c.head_dim))?;
                                let lo = if is_k {
                                    (head * self.capacity.div_ceil(crate::kv::GROUP_TOKENS)
                                        + pos / crate::kv::GROUP_TOKENS)
                                        * c.head_dim
                                        * 2
                                } else {
                                    (head * self.capacity + pos) * 2
                                };
                                let statistic = s.clone_dtoh(
                                    &stats.slice(lo..lo + if is_k { c.head_dim * 2 } else { 2 }),
                                )?;
                                for (d, &word) in code.iter().enumerate() {
                                    let i = if is_k { 2 * d } else { 0 };
                                    let scaled = f32::from(word) * statistic[i + 1];
                                    output.push(bf16::from_f32(scaled + statistic[i]).to_f32());
                                }
                            }
                        }
                    }
                }
                continue;
            }
            for buffer in [&layer.k, &layer.v] {
                for head in 0..c.num_key_value_heads {
                    let values = self.device.stream.clone_dtoh(&buffer.slice(
                        (head * self.capacity + start) * c.head_dim
                            ..(head * self.capacity + start + rows) * c.head_dim,
                    ))?;
                    output.extend(values.into_iter().map(bf16::to_f32));
                }
            }
        }
        Ok(output)
    }
}

impl Engine {
    #[cfg(feature = "cuda")]
    fn initialize(
        &mut self,
        model: &Path,
        draft: Option<&Path>,
        write: bool,
        started: Instant,
    ) -> Result<()> {
        let timer = Instant::now();
        let budget = self.spec.as_ref().map(|_| self.spec_budget);
        self.forward_rows = super::closure::forward_rows(
            super::closure::small_rows(self.chunk, self.capacity),
            budget,
            self.spec.as_ref().map(|spec| &spec.draft.config),
        )?;
        let profile = super::closure::Profile::new(
            &self.config,
            self.capacity,
            self.chunk,
            budget,
            model,
            draft,
            self.spec.as_ref().map(|s| &s.draft.config),
        )?;
        self.blas.declare(self.workspace_rows, &profile.pairs);
        let key = self.device.key.clone();
        let bucket = crate::backend::setup::attention_bucket(self.capacity)?;
        let mut paths: Vec<_> = profile
            .linear
            .iter()
            .map(|&shape| self.blas.choice_path(shape))
            .collect();
        use super::closure::Attention as Kind;
        for &kind in &profile.attention {
            paths.push(match kind {
                Kind::Decode => {
                    super::calibrate::cache_path(&self.runtime, &key, bucket, self.capacity)
                }
                Kind::Draft | Kind::Verify | Kind::Tree => {
                    let spec = self
                        .spec
                        .as_ref()
                        .context("attention calibration requires draft geometry")?;
                    let rows = match kind {
                        Kind::Draft => SpecPhase::Draft.rows(spec)?,
                        Kind::Verify => spec.max_verify_rows,
                        Kind::Tree => self.spec_budget.rows(),
                        Kind::Decode => anyhow::bail!("decode attention has no multirow cache"),
                    };
                    super::multi_calibrate::cache_path(
                        &self.runtime,
                        &key,
                        bucket,
                        crate::backend::setup::MultiShape {
                            rows,
                            causal: kind == Kind::Verify,
                            capacity: self.capacity,
                        },
                        kind == Kind::Tree,
                    )
                }
            });
        }
        if budget.is_some_and(SpecBudget::is_tree) {
            paths.push(super::markov_calibrate::cache_path(
                &self.runtime,
                &key,
                self.spec
                    .as_ref()
                    .context("tree draft missing")?
                    .draft
                    .markov_shape(),
            ));
        }
        paths.push(profile.path(&self.runtime, &key)?);
        self.closure.preflight(&paths)?;
        if write {
            let c = &self.config;
            for &shape in &profile.linear {
                let values = super::random(shape.rows * shape.input, 1);
                let input = self.device.stream.clone_htod(&values)?;
                let layer = &self.weights.layers[0];
                let weight = if shape.output == c.vocab_size && shape.input == c.hidden_size {
                    Some(&self.weights.head)
                } else {
                    [
                        (&layer.qkv, c.qkv_dim(), c.hidden_size),
                        (&layer.o, c.hidden_size, c.hidden_size),
                        (&layer.gu, 2 * c.intermediate_size, c.hidden_size),
                        (&layer.down, c.hidden_size, c.intermediate_size),
                    ]
                    .into_iter()
                    .find(|(_, o, i)| *o == shape.output && *i == shape.input)
                    .map(|(w, _, _)| w)
                };
                if let Some(weight) = weight {
                    self.blas.calibrate_shape(weight, &input.slice(..), shape)?;
                } else {
                    self.spec
                        .as_ref()
                        .context("draft shape without draft")?
                        .draft
                        .calibrate_shape(&mut self.blas, &input.slice(..), shape)?;
                }
            }
        } else {
            self.blas.load_choices(&profile.linear)?;
        }
        for &kind in &profile.attention {
            match kind {
                Kind::Decode => self.ensure_decode()?,
                Kind::Draft => self
                    .spec
                    .as_mut()
                    .context("draft state missing")?
                    .draft
                    .setup_attention(
                        &self.device,
                        &self.ops,
                        &self.config,
                        &self.runtime,
                        &self.closure,
                    )?,
                Kind::Verify => self.ensure_verification()?,
                Kind::Tree => self.ensure_tree_attention()?,
            }
        }
        if budget.is_some_and(SpecBudget::is_tree) {
            self.spec
                .as_mut()
                .context("draft state missing")?
                .draft
                .setup_markov(&self.device, &self.ops, &self.runtime, &self.closure)?;
        }
        self.device.stream.synchronize()?;
        self.closure.finish(profile, &self.runtime, key)?;
        let choices_ms = timer.elapsed().as_secs_f64() * 1000.0;
        if write {
            return Ok(());
        }
        let capture = Instant::now();
        self.prepare_graphs()?;
        self.closure.expect_captures(self.graphs.len())?;
        self.check_graphs()?;
        let capture_ms = capture.elapsed().as_secs_f64() * 1000.0;
        let restore = Instant::now();
        self.restore_setup()?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup_times":{"choices_ms":choices_ms,"graphs_ms":capture_ms,"restore_ms":restore.elapsed().as_secs_f64()*1000.0}})
        );
        self.closure.ready(started)
    }

    fn prepare_graphs(&mut self) -> Result<()> {
        self.set_token(0)?;
        if self.graphs.is_empty() {
            return Ok(());
        }
        if self.moe.is_some() {
            if self.spec.is_none() {
                return self.capture();
            }
            self.spec
                .as_mut()
                .context("MoE chain missing")?
                .draft
                .checkpoint_kv(0)?;
            self.ensure_spec_graph(SpecPhase::Draft, 0)?;
            self.launch_spec_graph(SpecPhase::Draft)?;
            self.device.stream.synchronize()?;
            self.spec
                .as_mut()
                .context("MoE chain missing")?
                .draft
                .restore_kv(0, 0)?;
            self.capture()?;
            let rows = self
                .spec
                .as_ref()
                .context("MoE chain missing")?
                .inject_graphs
                .len();
            for rows in 1..=rows {
                self.spec
                    .as_mut()
                    .context("MoE chain missing")?
                    .draft
                    .checkpoint_kv(0)?;
                self.ensure_spec_graph(SpecPhase::Inject(rows), 0)?;
                self.launch_spec_graph(SpecPhase::Inject(rows))?;
                self.device.stream.synchronize()?;
                self.spec
                    .as_mut()
                    .context("MoE chain missing")?
                    .draft
                    .restore_kv(0, 0)?;
            }
            self.spec.as_mut().context("MoE chain missing")?.active_rows = 0;
            return Ok(());
        }
        if self.spec.is_none() {
            self.capture()?;
            self.graph
                .as_ref()
                .context("decode Graph missing")?
                .launch()?;
            self.device.stream.synchronize()?;
            self.check_logits()?;
            return Ok(());
        }
        let injections: Vec<_> = self
            .graphs
            .iter()
            .filter_map(|graph| match graph {
                Graph::Inject(rows) => Some(*rows),
                _ => None,
            })
            .collect();
        if !self.spec_budget.is_tree() {
            let verify_rows = self
                .spec
                .as_ref()
                .context("draft state missing")?
                .max_verify_rows;
            self.ensure_spec_graph(SpecPhase::Draft, 0)?;
            self.ensure_spec_graph(SpecPhase::Verify(verify_rows), 0)?;
            self.launch_spec_graph(SpecPhase::Draft)?;
            self.launch_spec_graph(SpecPhase::Verify(verify_rows))?;
            for rows in injections {
                self.ensure_spec_graph(SpecPhase::Inject(rows), 0)?;
                self.launch_spec_graph(SpecPhase::Inject(rows))?;
            }
        } else {
            self.ensure_tree_draft_graph(0)?;
            let count = self.spec_budget.rows();
            let tokens: Vec<_> = (0..count)
                .map(|r| if r < 8 { 0 } else { r as u32 })
                .collect();
            let parents: Vec<_> = (0..count)
                .map(|r| {
                    if r == 0 {
                        -1
                    } else if r < 8 {
                        (r - 1) as i32
                    } else {
                        0
                    }
                })
                .collect();
            let tree = crate::tree::Tree::edges(&tokens, &parents, 0, self.capacity)?;
            self.spec
                .as_mut()
                .context("draft state missing")?
                .tree
                .as_mut()
                .context("tree state missing")?
                .load(&tree)?;
            let positions: Vec<_> = tree.nodes().iter().map(|n| n.position).collect();
            let step = crate::backend::Step {
                rows: count,
                input: crate::backend::Input::Ids,
                head: Head::All,
                mask: crate::backend::Mask::Tree {
                    parents: &parents,
                    positions: &positions,
                },
            };
            self.ensure_tree_verify_graph(step)?;
            self.spec
                .as_ref()
                .context("draft state missing")?
                .tree_verify_graph
                .as_ref()
                .context("tree verify Graph missing")?
                .launch()?;
            let predictions = vec![0; count];
            for rows in injections {
                let plan = tree.select(&predictions, rows, &[])?;
                self.spec
                    .as_mut()
                    .context("draft state missing")?
                    .tree
                    .as_mut()
                    .context("tree state missing")?
                    .load_commit(&plan)?;
                self.tree_gather_scatter()?;
                if rows == 1 {
                    self.capture_tree_commit()?;
                }
                self.spec
                    .as_ref()
                    .context("draft state missing")?
                    .tree_commit_graph
                    .as_ref()
                    .context("tree commit Graph missing")?
                    .launch()?;
                self.capture_tree_inject(0, rows)?;
                self.spec
                    .as_ref()
                    .context("draft state missing")?
                    .inject_graphs[rows - 1]
                    .as_ref()
                    .context("tree inject Graph missing")?
                    .launch()?;
            }
            let spec = self.spec.as_mut().context("draft state missing")?;
            spec.tree_draft_graph
                .as_ref()
                .context("tree draft Graph missing")?
                .launch()?;
            for p in [1, 8, 64] {
                spec.draft.distributions_batch(&vec![(0, 0); p])?;
            }
        }
        self.device.stream.synchronize()?;
        Ok(())
    }

    fn restore_setup(&mut self) -> Result<()> {
        let s = self.device.stream.clone();
        s.synchronize()?;
        let rows = if self.spec.is_some() {
            if self.spec_budget.is_tree() {
                self.spec_budget.rows()
            } else {
                self.spec
                    .as_ref()
                    .context("draft geometry missing")?
                    .draft
                    .config
                    .verify_rows()?
            }
        } else {
            1
        };
        for kv in &mut self.state.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = head * self.capacity * self.config.head_dim;
                let end = lo + rows * self.config.head_dim;
                s.memset_zeros(&mut kv.k.slice_mut(lo..end))?;
                s.memset_zeros(&mut kv.v.slice_mut(lo..end))?;
            }
        }
        for value in [
            &mut self.work.x,
            &mut self.work.n,
            &mut self.work.qkv,
            &mut self.work.attn,
            &mut self.work.out,
            &mut self.work.gu,
            &mut self.work.act,
            &mut self.work.logits,
        ] {
            s.memset_zeros(value)?;
        }
        s.memset_zeros(&mut self.work.ids)?;
        s.memset_zeros(&mut self.work.token)?;
        if let Some(capture) = &mut self.hidden_capture {
            s.memset_zeros(&mut capture.values)?;
        }
        if let Some(spec) = &mut self.spec {
            spec.draft.reset_setup()?;
            s.memset_zeros(&mut spec.logits)?;
            s.memset_zeros(&mut spec.predictions)?;
            spec.active_rows = 0;
            if let Some(tree) = &mut spec.tree {
                tree.reset_setup()?;
            }
        }
        self.reset()?;
        s.synchronize()?;
        ensure!(
            self.position == 0
                && self.token()? == 0
                && s.clone_dtoh(&self.work.position)? == [0]
                && s.clone_dtoh(&self.work.length)? == [1],
            "setup state was not restored"
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
#[test]
fn glm_dispatch_selects_the_hybrid_runtime() -> Result<()> {
    assert!(matches!(
        load_dispatch(crate::model_spec::Architecture::Glm5Next)?,
        LoadDispatch::Qwen
    ));
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[path = "engine_tests.rs"]
mod tests;

impl Engine {
    fn run_ffn(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        if self.run_moe_if_bound(layer, step, trace)? {
            return Ok(());
        }
        for op in [
            crate::backend::Operator::GateUp,
            crate::backend::Operator::Activate,
            crate::backend::Operator::Down,
        ] {
            self.run_operator(op, layer, step, trace)
                .with_context(|| format!("dense FFN operator {op:?} layer={layer}"))?;
        }
        Ok(())
    }
}

#[cfg(feature = "rocm")]
impl Engine {
    pub(crate) fn load_mtp(
        _: &std::path::Path,
        _: &crate::spec::Options,
        _: usize,
        _: bool,
    ) -> Result<Self> {
        anyhow::bail!("HIP load_mtp is unsupported")
    }
    pub(crate) fn generate_mtp_prepared(
        &mut self,
        _: &[u32],
        _: usize,
        _: bool,
        _: crate::mtp::VerifyPolicy,
        _: bool,
        _: &mut dyn FnMut(&[u32], usize) -> bool,
    ) -> Result<crate::mtp::Output> {
        anyhow::bail!("HIP generate_mtp_prepared is unsupported")
    }
    pub(crate) fn reset_mtp(&mut self) -> Result<()> {
        anyhow::bail!("HIP reset_mtp is unsupported")
    }
    pub(crate) fn generate_mtp(
        _dir: &Path,
        _ids: &[u32],
        _options: &crate::spec::Options,
        _depth: usize,
        _policy: crate::mtp::VerifyPolicy,
        _graph: bool,
    ) -> Result<crate::mtp::Output> {
        anyhow::bail!("HIP generate_mtp is unsupported")
    }
    pub fn profiler_start(&self) -> Result<()> {
        anyhow::bail!("HIP profiler_start is unsupported")
    }
    pub fn profiler_stop(&self) -> Result<()> {
        anyhow::bail!("HIP profiler_stop is unsupported")
    }
    pub fn profile_rounds(&mut self, enabled: bool) {
        assert!(!enabled, "HIP speculative round profiling is unsupported");
    }
    pub fn kv_read_bytes(&self, tokens: usize) -> usize {
        self.state.kv.len() * tokens * self.config.kv_dim() * 4
    }
}

#[cfg(feature = "rocm")]
#[derive(serde::Serialize, serde::Deserialize)]
struct HipSetup {
    identity: super::blas::Identity,
    profile: super::closure::Profile,
    linear: Vec<super::blas::Choice>,
    decode: crate::backend::setup::AttentionPlan,
    prefill: usize,
}

#[cfg(all(test, feature = "rocm"))]
#[test]
#[ignore = "requires GPU lease, FF_ARGMAX_MODEL and FLYINGFISH_RUNTIME_DIR"]
fn hip_engine_capture_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
    let settings = Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        runtime_dir: Some(std::env::var("FLYINGFISH_RUNTIME_DIR")?.into()),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    drop(Engine::calibrate(
        &model,
        0,
        256,
        Some(33),
        settings.clone(),
    )?);
    for (stage, layer) in [
        (None, None),
        (Some(CaptureFault::Begin), None),
        (Some(CaptureFault::End), None),
        (Some(CaptureFault::Instantiate), None),
        (None, Some((3, 1))),
        (None, Some((3, 2))),
    ] {
        let mut engine = Engine::load(&model, 0, 256, Some(33), settings.clone())?;
        let ids: Vec<_> = (0..17).map(|i| ((i * 269 + 23) % 130560) as u32).collect();
        engine.prefill(&ids, None)?;
        let s = engine.device.stream.clone();
        let snapshot = |engine: &Engine| -> Result<_> {
            let mut kv = Vec::new();
            for layer in &engine.state.kv {
                kv.extend(s.clone_dtoh(&layer.k)?);
                kv.extend(s.clone_dtoh(&layer.v)?);
            }
            Ok((
                kv,
                s.clone_dtoh(&engine.work.token)?,
                s.clone_dtoh(&engine.work.position)?,
                s.clone_dtoh(&engine.work.length)?,
            ))
        };
        let before = snapshot(&engine)?;
        engine.graph = None;
        engine.closure = Rc::new(super::closure::Closure::new(false, "calibrate".into()));
        engine.initialize_hip(&model, None)?;
        engine.capture_fail_stage = stage;
        engine.capture_fail_after_layer = layer;
        let result = engine.capture();
        ensure!(
            snapshot(&engine)? == before
                && engine.position == 17
                && engine.committed_prefix_ids == ids,
            "HIP capture failed to restore KV/token/position/length"
        );
        if stage.is_some() || layer.is_some() {
            ensure!(
                result.is_err() && engine.graph.is_none() && engine.poisoned,
                "injected capture fault did not poison/dispose"
            );
            ensure!(
                engine.decode(false, None).is_err(),
                "poisoned engine decoded"
            );
        } else {
            result?;
            ensure!(
                engine.graph.is_some() && !engine.poisoned,
                "successful capture did not publish graph"
            );
        }
        engine.capture_fail_stage = None;
        engine.capture_fail_after_layer = None;
        engine.reset()?;
        ensure!(
            !engine.poisoned && engine.position == 0,
            "capture failure reset did not recover"
        );
        engine.closure.ready(Instant::now())?;
        engine.prefill(&ids, None)?;
        engine.decode(false, None)?;
        println!(
            "HIP_CAPTURE_TRANSACTION stage={stage:?} layer={layer:?} exact_restore=true reset_eager_recovery=true PASS"
        );
    }
    Ok(())
}

#[cfg(feature = "rocm")]
impl Engine {
    pub fn calibrate(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        chunk: Option<usize>,
        settings: Settings,
    ) -> Result<Option<Self>> {
        let started = Instant::now();
        let mut engine = Self::load_kind(dir, ordinal, capacity, chunk, settings, true)?;
        if engine.model_spec.architecture == crate::model_spec::Architecture::Flint {
            engine.finish_flint_calibration(dir, started)?;
        }
        Ok(Some(engine))
    }

    fn load_kind_for(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        mode: SetupMode,
    ) -> Result<Self> {
        let started = Instant::now();
        let write = mode.writes();
        let model_spec = crate::model_spec::ModelSpec::read(dir)?;
        if model_spec.architecture == crate::model_spec::Architecture::LoopVl {
            return Self::load_recurrent(
                dir,
                ordinal,
                capacity,
                requested_chunk,
                settings,
                model_spec,
            );
        }
        if matches!(
            model_spec.architecture,
            crate::model_spec::Architecture::MiniCpmV47
                | crate::model_spec::Architecture::Qwen35Text
                | crate::model_spec::Architecture::Flint
        ) {
            return Self::load_qwen(
                dir,
                ordinal,
                capacity,
                requested_chunk,
                settings,
                mode,
                model_spec,
            );
        }
        ensure!(
            model_spec.architecture == crate::model_spec::Architecture::Llama,
            "HIP dense setup requires Llama; architecture {:?} is unsupported",
            model_spec.architecture
        );
        ensure!(
            !settings.prefix_cache && settings.kv_format == crate::kv::KvFormat::Bf16,
            "HIP dense setup requires BF16 KV without persistent prefix cache"
        );
        let config = Config::read(dir)?;
        ensure!(
            config.num_attention_heads == 16
                && config.num_key_value_heads == 2
                && config.head_dim == 128,
            "HIP dense attention requires Q16/KV2/d128"
        );
        ensure!(
            capacity > 0
                && capacity <= config.max_position_embeddings
                && capacity <= i32::MAX as usize / 128,
            "HIP KV capacity exceeds model/kernel bounds"
        );
        let runtime = super::closure::runtime_dir(
            settings
                .runtime_dir
                .as_deref()
                .context("HIP setup requires a runtime directory")?,
            dir,
            write,
        )?;
        let command = format!(
            "flyingfish --model {} --capacity {capacity} --runtime-dir {} calibrate",
            dir.display(),
            runtime.display()
        );
        let closure = Rc::new(super::closure::Closure::new(write, command));
        let device = Device::new(ordinal)?;
        let weights = Weights::load(dir, &config, &device).context("HIP load dense weights")?;
        let s = &device.stream;
        let state = super::state::StateStore::new(&model_spec, capacity, s)?;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, model_spec.swiglu_rounding())?;
        let blas = Blas::new(s.clone())?;
        let draft_config = settings
            .draft_model
            .as_deref()
            .map(|path| crate::dspark::DraftConfig::read(path, &config))
            .transpose()?;
        if draft_config.is_some() {
            device.ctx.require_wmma()?;
        }
        let workspace_rows = draft_config
            .as_ref()
            .map(|draft| {
                Ok::<_, anyhow::Error>(spec_workspace_rows(
                    1,
                    Some(settings.spec_budget),
                    draft.verify_rows()?,
                ))
            })
            .transpose()?
            .unwrap_or(1);
        ensure!(
            workspace_rows <= capacity,
            "HIP DSpark workspace exceeds KV capacity"
        );
        let work = Work::new(s, &model_spec, workspace_rows)?;
        let capture_checkpoint = Some(super::state::StateCheckpoint::new(
            &state,
            &model_spec,
            1,
            s,
        )?);
        let mut engine = Self {
            images: None,
            qwen_device_prefix: false,
            qwen: None,
            recurrent: None,
            hc: None,
            glm_dsa: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            #[cfg(test)]
            flint_window_capture: None,
            #[cfg(test)]
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            logits_f32: None,
            #[cfg(test)]
            moe_test_ffn: None,
            graphs: Vec::new(),
            spec: None,
            hidden_capture: None,
            spec_graph: settings.spec_graph,
            tree_builder: settings.tree_builder,
            compare_builders: false,
            profile_rounds: false,
            graph: None,
            capture_checkpoint,
            #[cfg(test)]
            capture_fail_after_layer: None,
            #[cfg(test)]
            capture_fail_stage: None,
            forward_rows: (1..=workspace_rows.min(16)).collect(),
            closure,
            poisoned: false,
            spec_budget: settings.spec_budget,
            runtime,
            weights,
            state,
            work,
            ops,
            blas,
            decode: None,
            device,
            config,
            model_spec,
            capacity,
            chunk: workspace_rows,
            workspace_rows,
            trace_layer: 0,
            position: 0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
            flash: None,
        };
        if let Some(path) = &settings.draft_model {
            let draft = super::draft::Draft::new(
                path,
                &engine.device,
                &engine.config,
                capacity,
                workspace_rows,
                &mut engine.blas,
                super::markov::ScoreRounding::Bf16,
            )?;
            engine.hidden_capture = Some(HiddenCapture::new(
                &draft.config.target_layer_ids,
                &engine.config,
                workspace_rows,
                &engine.device.stream,
            )?);
            engine.spec = Some(Spec::new(draft, &engine)?);
        }
        engine.size_workspace(requested_chunk)?;
        engine.declare_graphs()?;
        engine.initialize_hip(dir, settings.draft_model.as_deref())?;
        engine.prepare_prefill_plans()?;
        if let Some(spec) = &mut engine.spec {
            spec.draft.setup_attention(
                &engine.device,
                &engine.ops,
                &engine.config,
                &engine.runtime,
                &engine.closure,
            )?;
            spec.draft.setup_markov(
                &engine.device,
                &engine.ops,
                &engine.runtime,
                &engine.closure,
            )?;
        }
        if engine.spec.is_some() {
            if engine.spec_budget.is_tree() {
                engine.ensure_tree_attention()?;
            } else {
                engine.ensure_verification()?;
            }
        }
        engine.reset()?;
        engine.prepare_graphs()?;
        engine.restore_setup()?;
        engine.closure.expect_captures(engine.graphs.len())?;
        engine.check_graphs()?;
        engine.closure.ready(started)?;
        Ok(engine)
    }

    fn initialize_hip(&mut self, model: &Path, draft_path: Option<&Path>) -> Result<()> {
        use super::closure::{Profile, hash, write_cache};
        let budget = self.spec.as_ref().map(|_| self.spec_budget);
        self.forward_rows = super::closure::forward_rows(
            super::closure::small_rows(self.chunk, self.capacity),
            budget,
            self.spec.as_ref().map(|s| &s.draft.config),
        )?;
        let profile = Profile::hip(
            &self.config,
            self.capacity,
            self.chunk,
            model,
            budget,
            draft_path,
        )?;
        let identity = self.blas.identity().clone();
        let tag = hash(&serde_json::to_vec(&(&identity, &profile))?);
        let path = self.runtime.join(format!("hip-dense-{tag}.json"));
        let manifest = self.runtime.join(format!("hip-closure-{tag}.json"));
        self.closure.preflight(&[path.clone(), manifest.clone()])?;
        let cached = self.closure.access(&path)?;
        let decode = super::verification::hip_plan(
            &self.device.ctx,
            self.capacity,
            1,
            false,
            self.config.num_attention_heads,
        )?;
        let prefill = Flash::tile(self.config.head_dim)?;
        let setup = if cached {
            let setup: HipSetup = serde_json::from_slice(&std::fs::read(&path)?)?;
            ensure!(
                setup.identity == identity
                    && setup.profile == profile
                    && setup.decode == decode
                    && setup.prefill == prefill,
                "HIP setup identity/profile/capability plan mismatch"
            );
            ensure!(
                setup
                    .linear
                    .iter()
                    .map(|c| c.shape)
                    .collect::<std::collections::BTreeSet<_>>()
                    == profile.linear
                    && setup.linear.len() == profile.linear.len(),
                "HIP setup linear closure is incomplete or duplicated"
            );
            for choice in &setup.linear {
                ensure!(
                    !self.blas.wmma_candidate(choice.shape) || choice.setup_median_ms.is_some(),
                    "HIP linear closure lacks WMMA/library setup selection"
                );
                self.blas.prepare_choice(choice.shape, None, Some(choice))?;
            }
            setup
        } else {
            let mut linear = Vec::with_capacity(profile.linear.len());
            for &shape in &profile.linear {
                if self.blas.wmma_candidate(shape) {
                    self.closure.measurement()?;
                    let values: Vec<_> = (0..shape.rows * shape.input)
                        .map(|i| bf16::from_f32(((i * 17 + 31) % 257) as f32 / 256.0 - 0.5))
                        .collect();
                    let input = self.device.stream.clone_htod(&values)?;
                    let c = &self.config;
                    let layer = &self.weights.layers[0];
                    let weight = [
                        (&self.weights.head, c.vocab_size, c.hidden_size),
                        (&layer.qkv, c.qkv_dim(), c.hidden_size),
                        (&layer.o, c.hidden_size, c.hidden_size),
                        (&layer.gu, 2 * c.intermediate_size, c.hidden_size),
                        (&layer.down, c.hidden_size, c.intermediate_size),
                    ]
                    .into_iter()
                    .find(|(_, output, input)| *output == shape.output && *input == shape.input)
                    .map(|(weight, _, _)| weight);
                    let choice = if let Some(weight) = weight {
                        self.blas.calibrate_shape(shape, weight, &input)?
                    } else {
                        let draft = &self
                            .spec
                            .as_ref()
                            .context("HIP calibration shape has no model/draft binding")?
                            .draft;
                        self.blas.calibrate_shape(
                            shape,
                            &draft.calibration_weight(shape)?,
                            &input,
                        )?
                    };
                    linear.push(choice);
                } else {
                    linear.push(self.blas.prepare_choice(shape, None, None)?);
                }
            }
            let setup = HipSetup {
                identity: identity.clone(),
                profile,
                linear,
                decode,
                prefill,
            };
            write_cache(&path, &setup)?;
            setup
        };
        self.decode = Some(Attention::new(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            self.capacity,
            setup.decode,
        )?);
        self.closure
            .finish_with_key(setup.profile, manifest, identity)?;
        eprintln!(
            "{}",
            serde_json::json!({"hip_setup": {"cached": cached, "measurements": if cached { 0 } else { setup.linear.iter().filter(|c| c.setup_median_ms.is_some()).count() }, "decode": decode, "prefill": prefill, "prefill_tile_rows": prefill, "gemm_rows": self.chunk, "calibration": "untuned correctness-only"}})
        );
        Ok(())
    }

    fn prepare(&mut self, rows: usize) -> Result<()> {
        if self.qwen.is_some() {
            return self.prepare_qwen(rows);
        }
        let rows = bucketed_prefill_rows(rows as u32, self.workspace_rows as u32)? as usize;
        let c = &self.config;
        for (output, input) in [
            (c.qkv_dim(), c.hidden_size),
            (c.hidden_size, c.hidden_size),
            (2 * c.intermediate_size, c.hidden_size),
            (c.hidden_size, c.intermediate_size),
        ] {
            self.blas.prepare(rows, output, input)?;
        }
        Ok(())
    }
    fn ensure_decode(&self) -> Result<()> {
        if self.qwen.is_some() {
            return Ok(());
        }
        ensure!(
            self.decode.is_some(),
            "HIP decode was not bound during setup"
        );
        Ok(())
    }
}

#[cfg(test)]
#[test]
#[ignore = "requires a full-model GPU window and FF_Q38_MODEL/FLYINGFISH_RUNTIME_DIR"]
fn q38_engine_capture_gpu() -> Result<()> {
    use crate::backend::{Input, Mask, Step};
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "full-model GPU window not acknowledged"
    );
    let dir = PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("FF_Q38_MODEL missing")?);
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime directory missing")?,
    );
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(runtime),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    // Calibration/setup returns Frozen before production graph capture for a fixed plan.
    let image_fixture = std::env::var_os("FF_Q38_IMAGE_CAPTURE_FIXTURE");
    let capacity = if image_fixture.is_some() { 128 } else { 1312 };
    let chunk = if image_fixture.is_some() { 9 } else { 8 };
    let mut engine = Engine::load_kind(&dir, 0, capacity, Some(chunk), settings, true)?;
    ensure!(engine.qwen.is_some(), "G7 fixture requires Q38");
    let s = engine.device.stream.clone();
    engine.reset()?;
    let ids = if let Some(path) = image_fixture.as_ref() {
        use safetensors::{Dtype, SafeTensors};
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;
        let integers = |name: &str| -> Result<Vec<i64>> {
            let t = tensors.tensor(name)?;
            ensure!(
                t.dtype() == Dtype::I64,
                "image fixture integer dtype mismatch"
            );
            Ok(t.data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| i64::from_le_bytes(*v))
                .collect())
        };
        let ids = integers("input_ids")?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let modalities = integers("mm_token_type_ids")?
            .into_iter()
            .map(u8::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let grid = integers("image_grid_thw")?;
        ensure!(grid.len() == 3, "capture fixture must contain one image");
        let grids = [crate::vision_input::ImageGrid {
            t: grid[0].try_into()?,
            h: grid[1].try_into()?,
            w: grid[2].try_into()?,
        }];
        let pixels = tensors.tensor("pixel_values_fp32")?;
        ensure!(pixels.dtype() == Dtype::F32, "capture pixels must be FP32");
        let pixels: Vec<_> = pixels
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let input = crate::vision_input::MultimodalInput {
            input_ids: &ids,
            modality_ids: &modalities,
            grids: &grids,
            pixel_values: &pixels,
        };
        engine.prepare_multimodal(&input, None)?;
        ensure!(
            ids.len() % chunk == 1 && ids.len() + 3 <= capacity,
            "use the tiny/marker fixture for M1 tail and replay room"
        );
        ids
    } else {
        (0..257).map(|i| 1 + i % 31).collect()
    };
    for part in ids.chunks(chunk) {
        engine.prepare(part.len())?;
        engine
            .state
            .set_sequence(&s, engine.position as u32, part.len() as u32, chunk as u32)?;
        s.memcpy_htod(part, &mut engine.work.ids.slice_mut(..part.len()))?;
        engine.run_step(
            Step {
                rows: part.len(),
                input: Input::Ids,
                head: Head::Last,
                mask: Mask::Causal,
            },
            None,
        )?;
        engine.position += part.len();
    }
    let prefix = ids.len();
    s.memcpy_htod(&[prefix as i32], &mut engine.work.position)?;
    s.memcpy_htod(&[prefix as i32 + 1], &mut engine.work.length)?;
    engine
        .state
        .set_sequence(&s, prefix as u32, 1, chunk as u32)?;
    s.synchronize()?;

    fn state_bits(engine: &Engine) -> Result<Vec<u8>> {
        let s = &engine.device.stream;
        let mut bits = Vec::with_capacity(engine.state.bytes() + 32);
        for state in &engine.state.gdn {
            for buffer in [&state.s, &state.history] {
                for value in s.clone_dtoh(buffer)? {
                    bits.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        for kv in &engine.state.kv {
            for buffer in [&kv.k, &kv.v] {
                for value in s.clone_dtoh(buffer)? {
                    bits.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        for value in s.clone_dtoh(engine.state.sequence.as_ref().context("metadata missing")?)? {
            bits.extend_from_slice(&value.to_le_bytes());
        }
        for value in engine
            .qwen
            .as_ref()
            .context("Q38 runtime missing")?
            .position_values(s)?
        {
            bits.extend_from_slice(&value.to_le_bytes());
        }
        for value in s.clone_dtoh(&engine.work.token)? {
            bits.extend_from_slice(&value.to_le_bytes());
        }
        for buffer in [&engine.work.position, &engine.work.length] {
            for value in s.clone_dtoh(buffer)? {
                bits.extend_from_slice(&value.to_le_bytes());
            }
        }
        bits.extend_from_slice(&engine.position.to_le_bytes());
        Ok(bits)
    }
    let committed = state_bits(&engine)?;
    for occurrence in [1, 2] {
        engine.capture_fail_after_layer = Some((0, occurrence));
        ensure!(
            engine.capture().is_err(),
            "injected warmup failure was ignored"
        );
        ensure!(
            engine.poisoned && engine.graph.is_none(),
            "failed capture published a graph or remained healthy"
        );
        ensure!(
            state_bits(&engine)? == committed,
            "failed warmup did not restore committed state"
        );
        engine.capture_fail_after_layer = None;
        // The fixture reuses the restored state; ordinary load returns its capture error.
        engine.poisoned = false;
    }
    engine.capture()?;
    ensure!(
        state_bits(&engine)? == committed,
        "nonzero-prefix capture changed committed state"
    );
    // Capture's write window is one row. The replay comparison needs three rows.
    let mut seed = crate::backend::cuda::state::StateCheckpoint::new(
        &engine.state,
        &engine.model_spec,
        3,
        &s,
    )?;
    seed.save(&engine.state, &engine.work.token, prefix, &s, None)?;
    let mut replay_tokens = Vec::new();
    for _ in 0..3 {
        engine
            .graph
            .as_ref()
            .context("decode graph missing")?
            .launch()?;
        engine.position += 1;
        replay_tokens.push(engine.token()?);
    }
    engine.check_logits()?;
    let replay_state = state_bits(&engine)?;
    let replay_logits = s.clone_dtoh(&engine.work.logits)?;
    seed.restore(&mut engine.state, &mut engine.work.token, &s)?;
    s.memcpy_htod(&[prefix as i32], &mut engine.work.position)?;
    s.memcpy_htod(&[prefix as i32 + 1], &mut engine.work.length)?;
    engine.position = prefix;
    ensure!(
        state_bits(&engine)? == committed,
        "replay seed restore changed committed state"
    );
    let mut eager_tokens = Vec::new();
    for _ in 0..3 {
        engine.run_decode_step(None)?;
        engine.position += 1;
        eager_tokens.push(engine.token()?);
    }
    ensure!(eager_tokens == replay_tokens, "eager/graph tokens differ");
    ensure!(
        state_bits(&engine)? == replay_state,
        "eager/graph S/history/KV/meta/token differ"
    );
    ensure!(
        s.clone_dtoh(&engine.work.logits)? == replay_logits,
        "eager/graph logits differ"
    );
    engine.closure.expect_captures(3)?;
    engine.check_graphs()?;
    engine.reset()?;
    ensure!(engine.images.is_none(), "reset retained image features");
    ensure!(
        engine
            .qwen
            .as_ref()
            .context("Q38 runtime missing")?
            .position_values(&s)?
            == crate::vision_input::PositionPlan::text(1, capacity)?.packed_axes()?,
        "reset retained image coordinates"
    );
    for state in &engine.state.gdn {
        for buffer in [&state.s, &state.history] {
            ensure!(
                s.clone_dtoh(buffer)?.iter().all(|v| v.to_bits() == 0),
                "reset retained recurrent state"
            );
        }
    }
    engine.closure.ready(Instant::now())?;
    engine.check_ready()?;
    engine.poisoned = true;
    ensure!(
        engine.check_ready().is_err(),
        "poisoned engine reported Ready"
    );
    engine.reset()?;
    engine.check_ready()?;
    eprintln!(
        "Q38_ENGINE_G7 nonzero_capture_restore/warmup_and_capture_failure_restore/repeated_graph_eager/state_bits/logits/reset/ready_poison PASS"
    );
    Ok(())
}
